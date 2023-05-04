use assert2::assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    householder::make_householder_in_place,
    mul::{inner_prod::inner_prod_with_conj, matmul},
    parallelism_degree, temp_mat_req, temp_mat_zeroed, zipped, ComplexField, Conj, Entity, MatMut,
    Parallelism,
};
use reborrow::*;

use crate::tridiag_real_evd::norm2;

pub fn make_hessenberg_in_place_req<E: Entity>(
    n: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    StackReq::try_all_of([
        temp_mat_req::<E>(n, 1)?,
        temp_mat_req::<E>(n, 1)?,
        temp_mat_req::<E>(n, 1)?,
        temp_mat_req::<E>(n, parallelism_degree(parallelism))?,
        temp_mat_req::<E>(n, parallelism_degree(parallelism))?,
    ])
}

pub fn make_hessenberg_in_place<E: ComplexField>(
    a: MatMut<'_, E>,
    householder: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    assert!(a.nrows() == a.ncols());
    assert!(a.row_stride() == 1);

    let n = a.nrows();
    if n < 2 {
        return;
    }

    let mut a = a;
    let mut householder = householder;

    let (mut u, stack) = temp_mat_zeroed::<E>(n, 1, stack);
    let (mut y, stack) = temp_mat_zeroed::<E>(n, 1, stack);
    let (mut z, stack) = temp_mat_zeroed::<E>(n, 1, stack);

    let (mut v, stack) = temp_mat_zeroed::<E>(n, parallelism_degree(parallelism), stack);
    let (mut w, _) = temp_mat_zeroed::<E>(n, parallelism_degree(parallelism), stack);

    let mut u = u.as_mut();
    let mut y = y.as_mut();
    let mut z = z.as_mut();
    let mut v = v.as_mut();
    let mut w = w.as_mut();

    let arch = pulp::Arch::new();
    for k in 0..n - 1 {
        let a_cur = a.rb_mut().submatrix(k, k, n - k, n - k);
        let [mut a11, mut a12, mut a21, mut a22] = a_cur.split_at(1, 1);

        let [_, u] = u.rb_mut().split_at_row(k);
        let [nu, mut u21] = u.split_at_row(1);
        let [_, y] = y.rb_mut().split_at_row(k);
        let [psi, mut y21] = y.split_at_row(1);
        let [_, z] = z.rb_mut().split_at_row(k);
        let [zeta, mut z21] = z.split_at_row(1);

        let [_, v] = v.rb_mut().split_at_row(k);
        let [_, v21] = v.split_at_row(1);
        let mut v21 = v21.subcols(0, parallelism_degree(parallelism));

        let [_, w] = w.rb_mut().split_at_row(k);
        let [_, w21] = w.split_at_row(1);
        let mut w21 = w21.subcols(0, parallelism_degree(parallelism));

        if k > 0 {
            let nu = nu.read(0, 0);
            let psi = psi.read(0, 0);
            let zeta = zeta.read(0, 0);

            a11.write(
                0,
                0,
                a11.read(0, 0)
                    .sub(&(nu.mul(&psi.conj())).add(&zeta.mul(&nu.conj()))),
            );
            zipped!(a12.rb_mut(), y21.rb().transpose(), u21.rb().transpose()).for_each(
                |mut a, y, u| {
                    let y = y.read();
                    let u = u.read();
                    a.write(a.read().sub(&(nu.mul(&y.conj())).add(&zeta.mul(&u.conj()))));
                },
            );
            zipped!(a21.rb_mut(), u21.rb(), z21.rb()).for_each(|mut a, u, z| {
                let z = z.read();
                let u = u.read();
                a.write(a.read().sub(&(u.mul(&psi.conj())).add(&z.mul(&nu.conj()))));
            });
        }

        let (tau, new_head) = {
            let [head, tail] = a21.rb_mut().split_at_row(1);
            let norm2 = norm2(tail.rb());
            make_householder_in_place(Some(tail), head.read(0, 0), norm2)
        };
        a21.write(0, 0, E::one());
        let tau_inv = tau.inv();
        householder.write(k, 0, tau);

        {
            if k > 0 {
                matmul(
                    a22.rb_mut(),
                    u21.rb(),
                    y21.rb().adjoint(),
                    Some(E::one()),
                    E::one().neg(),
                    parallelism,
                );
                matmul(
                    a22.rb_mut(),
                    z21.rb(),
                    u21.rb().adjoint(),
                    Some(E::one()),
                    E::one().neg(),
                    parallelism,
                );
            }
            matmul(
                y21.rb_mut(),
                a22.rb().adjoint(),
                a21.rb(),
                None,
                E::one(),
                parallelism,
            );
            matmul(
                z21.rb_mut(),
                a22.rb(),
                a21.rb(),
                None,
                E::one(),
                parallelism,
            );
        }
        zipped!(u21.rb_mut(), a21.rb()).for_each(|mut dst, src| dst.write(src.read()));
        a21.write(0, 0, new_head);

        let beta = inner_prod_with_conj(u21.rb(), Conj::Yes, z21.rb(), Conj::No)
            .scale_power_of_two(&E::Real::from_f64(0.5));

        zipped!(y21.rb_mut(), u21.rb()).for_each(|mut y, u| {
            let u = u.read();
            let beta = beta.conj();
            y.write(y.read().sub(&beta.mul(&u.mul(&tau_inv))).mul(&tau_inv));
        });
        zipped!(z21.rb_mut(), u21.rb()).for_each(|mut z, u| {
            let u = u.read();
            z.write(z.read().sub(&beta.mul(&u.mul(&tau_inv))).mul(&tau_inv));
        });

        let mut a_right = a.rb_mut().submatrix(0, k + 1, k + 1, n - k - 1);
        for i in 0..k + 1 {
            let mut row = a_right.rb_mut().row(i);
            let dot = inner_prod_with_conj(row.rb().transpose(), Conj::No, u21.rb(), Conj::No)
                .mul(&tau_inv);
            zipped!(row.rb_mut(), u21.rb().transpose())
                .for_each(|mut a, u| a.write(a.read().sub(&u.read().conj().mul(&dot))));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{
        c64,
        householder::{
            apply_block_householder_sequence_on_the_right_in_place_req,
            apply_block_householder_sequence_on_the_right_in_place_with_conj,
            apply_block_householder_sequence_transpose_on_the_left_in_place_req,
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
        },
        Mat,
    };

    macro_rules! make_stack {
        ($req: expr $(,)?) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn test_make_hessenberg() {
        let n = 10;
        let parallelism = Parallelism::None;
        let a = Mat::with_dims(n, n, |_, _| c64::new(rand::random(), rand::random()));

        let mut h = a.clone();
        let mut householder = Mat::zeros(n - 1, 1);
        make_hessenberg_in_place(
            h.as_mut(),
            householder.as_mut(),
            parallelism,
            make_stack!(make_hessenberg_in_place_req::<c64>(n, parallelism)),
        );

        let mut copy = a.clone();
        apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            h.as_ref().submatrix(1, 0, n - 1, n - 1),
            householder.as_ref().transpose(),
            Conj::Yes,
            copy.as_mut().submatrix(1, 0, n - 1, n),
            parallelism,
            make_stack!(
                apply_block_householder_sequence_transpose_on_the_left_in_place_req::<c64>(
                    n - 1,
                    1,
                    n
                )
            ),
        );
        apply_block_householder_sequence_on_the_right_in_place_with_conj(
            h.as_ref().submatrix(1, 0, n - 1, n - 1),
            householder.as_ref().transpose(),
            Conj::No,
            copy.as_mut().submatrix(0, 1, n, n - 1),
            parallelism,
            make_stack!(
                apply_block_householder_sequence_on_the_right_in_place_req::<c64>(n - 1, 1, n)
            ),
        );

        for j in 0..n {
            for i in 0..Ord::min(n, j + 2) {
                assert_approx_eq!(copy.read(i, j), h.read(i, j));
            }
        }

        for j in 0..n {
            for i in j + 2..n {
                assert_approx_eq!(copy.read(i, j), c64::zero());
            }
        }
    }
}
