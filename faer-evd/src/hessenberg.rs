#[cfg(feature = "std")]
use assert2::assert;
use core::slice;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    householder::{
        apply_block_householder_on_the_right_in_place_req,
        apply_block_householder_on_the_right_in_place_with_conj, make_householder_in_place,
        upgrade_householder_factor,
    },
    mul::{inner_prod::inner_prod_with_conj, matmul},
    parallelism_degree, temp_mat_req, temp_mat_uninit, temp_mat_zeroed, zipped, ComplexField, Conj,
    Entity, MatMut, MatRef, Parallelism, SimdCtx,
};
use reborrow::*;

use crate::tridiag_real_evd::norm2;

pub fn make_hessenberg_in_place_req<E: Entity>(
    n: usize,
    householder_blocksize: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    StackReq::try_any_of([
        StackReq::try_all_of([
            temp_mat_req::<E>(n, 1)?,
            temp_mat_req::<E>(n, 1)?,
            temp_mat_req::<E>(n, 1)?,
            temp_mat_req::<E>(n, parallelism_degree(parallelism))?,
            temp_mat_req::<E>(n, parallelism_degree(parallelism))?,
        ])?,
        apply_block_householder_on_the_right_in_place_req::<E>(n, householder_blocksize, n)?,
    ])
}

struct HessenbergFusedUpdate<'a, E: ComplexField> {
    a: MatMut<'a, E>,
    v: MatMut<'a, E>,
    w: MatMut<'a, E>,

    u: MatRef<'a, E>,
    y: MatRef<'a, E>,
    z: MatRef<'a, E>,

    x: MatRef<'a, E>,
}

impl<E: ComplexField> pulp::WithSimd for HessenbergFusedUpdate<'_, E> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            mut a,
            mut v,
            mut w,
            u,
            y,
            z,
            x,
        } = self;

        debug_assert_eq!(a.row_stride(), 1);
        debug_assert_eq!(v.row_stride(), 1);
        debug_assert_eq!(w.row_stride(), 1);
        debug_assert_eq!(u.row_stride(), 1);
        debug_assert_eq!(y.row_stride(), 1);
        debug_assert_eq!(z.row_stride(), 1);
        debug_assert_eq!(x.row_stride(), 1);

        let m = a.nrows();
        let n = a.ncols();

        debug_assert!(m > 0);

        let lane_count = core::mem::size_of::<E::SimdUnit<S>>() / core::mem::size_of::<E::Unit>();

        unsafe {
            let prefix = ((m - 1) % lane_count) + 1;

            let u_ = u;
            let x_ = x;

            let w = E::faer_map(
                w.rb_mut().as_ptr(),
                #[inline(always)]
                |ptr| slice::from_raw_parts_mut(ptr, m),
            );
            let u = E::faer_map(
                u.as_ptr(),
                #[inline(always)]
                |ptr| slice::from_raw_parts(ptr, m),
            );
            let z = E::faer_map(
                z.as_ptr(),
                #[inline(always)]
                |ptr| slice::from_raw_parts(ptr, m),
            );
            let x = E::faer_map(
                x.as_ptr(),
                #[inline(always)]
                |ptr| slice::from_raw_parts(ptr, m),
            );

            let (mut w_prefix, w_suffix) = E::faer_unzip(E::faer_map(
                w,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let (u_prefix, u_suffix) = E::faer_unzip(E::faer_map(
                u,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));
            let (z_prefix, z_suffix) = E::faer_unzip(E::faer_map(
                z,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));
            let (x_prefix, x_suffix) = E::faer_unzip(E::faer_map(
                x,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));

            let w_suffix = faer_core::simd::slice_as_mut_simd::<E, S>(w_suffix).0;
            let u_suffix = faer_core::simd::slice_as_simd::<E, S>(u_suffix).0;
            let z_suffix = faer_core::simd::slice_as_simd::<E, S>(z_suffix).0;
            let x_suffix = faer_core::simd::slice_as_simd::<E, S>(x_suffix).0;

            let (mut w_head, mut w_tail) = E::faer_as_arrays_mut::<4, _>(w_suffix);
            let (u_head, u_tail) = E::faer_as_arrays::<4, _>(u_suffix);
            let (z_head, z_tail) = E::faer_as_arrays::<4, _>(z_suffix);
            let (x_head, x_tail) = E::faer_as_arrays::<4, _>(x_suffix);

            let zero = E::faer_zero();

            for j in 0..n {
                let a = E::faer_map(
                    a.rb_mut().ptr_at(0, j),
                    #[inline(always)]
                    |ptr| slice::from_raw_parts_mut(ptr, m),
                );

                let (a_prefix, a_suffix) = E::faer_unzip(E::faer_map(
                    a,
                    #[inline(always)]
                    |slice| slice.split_at_mut(prefix),
                ));
                let a_suffix = faer_core::simd::slice_as_mut_simd::<E, S>(a_suffix).0;
                let (a_head, a_tail) = E::faer_as_arrays_mut::<4, _>(a_suffix);

                let y_rhs = E::faer_simd_splat(simd, y.read(j, 0).faer_conj().faer_neg());
                let u_rhs = E::faer_simd_splat(simd, u_.read(j, 0).faer_conj().faer_neg());
                let x_rhs = E::faer_simd_splat(simd, x_.read(j, 0));

                let mut sum0 = E::faer_simd_splat(simd, zero);
                let mut sum1 = E::faer_simd_splat(simd, zero);
                let mut sum2 = E::faer_simd_splat(simd, zero);
                let mut sum3 = E::faer_simd_splat(simd, zero);

                let mut a_prefix_ =
                    E::faer_partial_load_last(simd, E::faer_rb(E::faer_as_ref(&a_prefix)));
                let u_prefix = E::faer_partial_load_last(simd, E::faer_copy(&u_prefix));
                let z_prefix = E::faer_partial_load_last(simd, E::faer_copy(&z_prefix));

                a_prefix_ = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&u_prefix),
                    E::faer_copy(&y_rhs),
                    a_prefix_,
                );
                a_prefix_ = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&z_prefix),
                    E::faer_copy(&u_rhs),
                    a_prefix_,
                );

                E::faer_partial_store_last(simd, a_prefix, E::faer_copy(&a_prefix_));

                let mut w_prefix_ =
                    E::faer_partial_load_last(simd, E::faer_rb(E::faer_as_ref(&w_prefix)));
                w_prefix_ = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&a_prefix_),
                    E::faer_copy(&x_rhs),
                    w_prefix_,
                );
                E::faer_partial_store_last(
                    simd,
                    E::faer_rb_mut(E::faer_as_mut(&mut w_prefix)),
                    w_prefix_,
                );

                let x_prefix = E::faer_partial_load_last(simd, E::faer_copy(&x_prefix));
                sum0 = E::faer_simd_conj_mul_adde(
                    simd,
                    E::faer_copy(&a_prefix_),
                    E::faer_copy(&x_prefix),
                    sum0,
                );

                for ((((a, w), x), u), z) in E::faer_into_iter(a_head)
                    .zip(E::faer_into_iter(E::faer_rb_mut(E::faer_as_mut(
                        &mut w_head,
                    ))))
                    .zip(E::faer_into_iter(E::faer_copy(&x_head)))
                    .zip(E::faer_into_iter(E::faer_copy(&u_head)))
                    .zip(E::faer_into_iter(E::faer_copy(&z_head)))
                {
                    let [mut a0, mut a1, mut a2, mut a3] =
                        E::faer_unzip4(E::faer_deref(E::faer_rb(E::faer_as_ref(&a))));
                    let [mut w0, mut w1, mut w2, mut w3] =
                        E::faer_unzip4(E::faer_deref(E::faer_rb(E::faer_as_ref(&w))));

                    let [x0, x1, x2, x3] = E::faer_unzip4(E::faer_deref(x));
                    let [u0, u1, u2, u3] = E::faer_unzip4(E::faer_deref(u));
                    let [z0, z1, z2, z3] = E::faer_unzip4(E::faer_deref(z));

                    a0 = E::faer_simd_mul_adde(simd, E::faer_copy(&u0), E::faer_copy(&y_rhs), a0);
                    a0 = E::faer_simd_mul_adde(simd, E::faer_copy(&z0), E::faer_copy(&u_rhs), a0);

                    a1 = E::faer_simd_mul_adde(simd, E::faer_copy(&u1), E::faer_copy(&y_rhs), a1);
                    a1 = E::faer_simd_mul_adde(simd, E::faer_copy(&z1), E::faer_copy(&u_rhs), a1);

                    a2 = E::faer_simd_mul_adde(simd, E::faer_copy(&u2), E::faer_copy(&y_rhs), a2);
                    a2 = E::faer_simd_mul_adde(simd, E::faer_copy(&z2), E::faer_copy(&u_rhs), a2);

                    a3 = E::faer_simd_mul_adde(simd, E::faer_copy(&u3), E::faer_copy(&y_rhs), a3);
                    a3 = E::faer_simd_mul_adde(simd, E::faer_copy(&z3), E::faer_copy(&u_rhs), a3);

                    E::faer_map(
                        E::faer_zip(
                            a,
                            E::faer_zip(
                                E::faer_zip(E::faer_copy(&a0), E::faer_copy(&a1)),
                                E::faer_zip(E::faer_copy(&a2), E::faer_copy(&a3)),
                            ),
                        ),
                        #[inline(always)]
                        |(a, ((a0, a1), (a2, a3)))| {
                            a[0] = a0;
                            a[1] = a1;
                            a[2] = a2;
                            a[3] = a3;
                        },
                    );

                    w0 = E::faer_simd_mul_adde(simd, E::faer_copy(&a0), E::faer_copy(&x_rhs), w0);
                    w1 = E::faer_simd_mul_adde(simd, E::faer_copy(&a1), E::faer_copy(&x_rhs), w1);
                    w2 = E::faer_simd_mul_adde(simd, E::faer_copy(&a2), E::faer_copy(&x_rhs), w2);
                    w3 = E::faer_simd_mul_adde(simd, E::faer_copy(&a3), E::faer_copy(&x_rhs), w3);

                    E::faer_map(
                        E::faer_zip(
                            w,
                            E::faer_zip(
                                E::faer_zip(E::faer_copy(&w0), E::faer_copy(&w1)),
                                E::faer_zip(E::faer_copy(&w2), E::faer_copy(&w3)),
                            ),
                        ),
                        #[inline(always)]
                        |(w, ((w0, w1), (w2, w3)))| {
                            w[0] = w0;
                            w[1] = w1;
                            w[2] = w2;
                            w[3] = w3;
                        },
                    );

                    sum0 = E::faer_simd_conj_mul_adde(
                        simd,
                        E::faer_copy(&a0),
                        E::faer_copy(&x0),
                        sum0,
                    );
                    sum1 = E::faer_simd_conj_mul_adde(
                        simd,
                        E::faer_copy(&a1),
                        E::faer_copy(&x1),
                        sum1,
                    );
                    sum2 = E::faer_simd_conj_mul_adde(
                        simd,
                        E::faer_copy(&a2),
                        E::faer_copy(&x2),
                        sum2,
                    );
                    sum3 = E::faer_simd_conj_mul_adde(
                        simd,
                        E::faer_copy(&a3),
                        E::faer_copy(&x3),
                        sum3,
                    );
                }

                sum0 = E::faer_simd_add(simd, sum0, sum1);
                sum2 = E::faer_simd_add(simd, sum2, sum3);

                sum0 = E::faer_simd_add(simd, sum0, sum2);

                for ((((a, w), x), u), z) in E::faer_into_iter(a_tail)
                    .zip(E::faer_into_iter(E::faer_rb_mut(E::faer_as_mut(
                        &mut w_tail,
                    ))))
                    .zip(E::faer_into_iter(E::faer_copy(&x_tail)))
                    .zip(E::faer_into_iter(E::faer_copy(&u_tail)))
                    .zip(E::faer_into_iter(E::faer_copy(&z_tail)))
                {
                    let mut a0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&a)));
                    let mut w0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w)));

                    let x0 = E::faer_deref(x);
                    let u0 = E::faer_deref(u);
                    let z0 = E::faer_deref(z);

                    a0 = E::faer_simd_mul_adde(simd, E::faer_copy(&u0), E::faer_copy(&y_rhs), a0);
                    a0 = E::faer_simd_mul_adde(simd, E::faer_copy(&z0), E::faer_copy(&u_rhs), a0);

                    E::faer_map(
                        E::faer_zip(a, E::faer_copy(&a0)),
                        #[inline(always)]
                        |(a, a0)| *a = a0,
                    );

                    w0 = E::faer_simd_mul_adde(simd, E::faer_copy(&a0), E::faer_copy(&x_rhs), w0);

                    E::faer_map(
                        E::faer_zip(w, E::faer_copy(&w0)),
                        #[inline(always)]
                        |(w, w0)| *w = w0,
                    );

                    sum0 = E::faer_simd_conj_mul_adde(
                        simd,
                        E::faer_copy(&a0),
                        E::faer_copy(&x0),
                        sum0,
                    );
                }

                let sum = E::faer_simd_reduce_add(simd, sum0);
                v.write(j, 0, sum);
            }
        }
    }
}

pub fn make_hessenberg_in_place<E: ComplexField>(
    a: MatMut<'_, E>,
    householder: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    assert!(a.nrows() == a.ncols());
    assert!(a.row_stride() == 1);

    let n = a.nrows();
    if n < 2 {
        return;
    }

    let mut a = a;
    let mut householder = householder;

    let mut stack = stack;

    {
        let (mut u, stack) = temp_mat_zeroed::<E>(n, 1, stack.rb_mut());
        let (mut y, stack) = temp_mat_zeroed::<E>(n, 1, stack);
        let (mut z, stack) = temp_mat_zeroed::<E>(n, 1, stack);

        let (mut v, stack) = temp_mat_zeroed::<E>(n, 1, stack);
        let (mut w, _) = temp_mat_zeroed::<E>(n, parallelism_degree(parallelism), stack);

        let mut u = u.as_mut();
        let mut y = y.as_mut();
        let mut z = z.as_mut();
        let mut v = v.as_mut();
        let mut w = w.as_mut();

        let arch = E::Simd::default();
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
            let [_, mut v21] = v.split_at_row(1);

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
                    a11.read(0, 0).faer_sub(
                        (nu.faer_mul(psi.faer_conj())).faer_add(zeta.faer_mul(nu.faer_conj())),
                    ),
                );
                zipped!(a12.rb_mut(), y21.rb().transpose(), u21.rb().transpose()).for_each(
                    |mut a, y, u| {
                        let y = y.read();
                        let u = u.read();
                        a.write(a.read().faer_sub(
                            (nu.faer_mul(y.faer_conj())).faer_add(zeta.faer_mul(u.faer_conj())),
                        ));
                    },
                );
                zipped!(a21.rb_mut(), u21.rb(), z21.rb()).for_each(|mut a, u, z| {
                    let z = z.read();
                    let u = u.read();
                    a.write(a.read().faer_sub(
                        (u.faer_mul(psi.faer_conj())).faer_add(z.faer_mul(nu.faer_conj())),
                    ));
                });
            }

            let (tau, new_head) = {
                let [head, tail] = a21.rb_mut().split_at_row(1);
                let norm2 = norm2(tail.rb());
                make_householder_in_place(Some(tail), head.read(0, 0), norm2)
            };
            a21.write(0, 0, E::faer_one());
            let tau_inv = tau.faer_inv();
            householder.write(k, 0, tau);

            if k > 0 {
                w21.fill_zeros();
                arch.dispatch(HessenbergFusedUpdate {
                    a: a22.rb_mut(),
                    v: v21.rb_mut(),
                    w: w21.rb_mut().col(0),
                    u: u21.rb(),
                    y: y21.rb(),
                    z: z21.rb(),
                    x: a21.rb(),
                });

                y21.rb_mut().clone_from(v21.rb());
                z21.rb_mut().clone_from(w21.rb().col(0));
            } else {
                matmul(
                    y21.rb_mut(),
                    a22.rb().adjoint(),
                    a21.rb(),
                    None,
                    E::faer_one(),
                    parallelism,
                );
                matmul(
                    z21.rb_mut(),
                    a22.rb(),
                    a21.rb(),
                    None,
                    E::faer_one(),
                    parallelism,
                );
            }

            zipped!(u21.rb_mut(), a21.rb()).for_each(|mut dst, src| dst.write(src.read()));
            a21.write(0, 0, new_head);

            let beta = inner_prod_with_conj(u21.rb(), Conj::Yes, z21.rb(), Conj::No)
                .faer_scale_power_of_two(E::Real::faer_from_f64(0.5));

            zipped!(y21.rb_mut(), u21.rb()).for_each(|mut y, u| {
                let u = u.read();
                let beta = beta.faer_conj();
                y.write(
                    y.read()
                        .faer_sub(beta.faer_mul(u.faer_mul(tau_inv)))
                        .faer_mul(tau_inv),
                );
            });
            zipped!(z21.rb_mut(), u21.rb()).for_each(|mut z, u| {
                let u = u.read();
                z.write(
                    z.read()
                        .faer_sub(beta.faer_mul(u.faer_mul(tau_inv)))
                        .faer_mul(tau_inv),
                );
            });
        }
    }

    let mut householder = householder.transpose();
    let householder_blocksize = householder.nrows();
    let mut k_base = 0;
    while k_base < n - 1 {
        let bs = Ord::min(householder_blocksize, n - 1 - k_base);

        let mut householder = householder.rb_mut().submatrix(0, k_base, bs, bs);
        let full_essentials = a.rb().submatrix(1, 0, n - 1, n - 1);
        let essentials = full_essentials.submatrix(k_base, k_base, n - 1 - k_base, bs);

        for k in 0..bs {
            householder.write(k, k, householder.read(0, k));
        }
        upgrade_householder_factor(householder, essentials, bs, 1, parallelism);
        k_base += bs;
    }

    let mut k_base = 0;
    while k_base < n - 1 {
        let bs = Ord::min(householder_blocksize, n - 1 - k_base);

        let householder = householder.rb().submatrix(0, k_base, bs, bs);
        let full_essentials = a.rb().submatrix(1, 0, n - 1, n - 1);
        let essentials = full_essentials.submatrix(k_base, k_base, n - 1 - k_base, bs);

        for k_local in 0..bs {
            let k = k_base + k_local;

            let mut a21 = unsafe { a.rb().col(k).subrows(k + 1, n - k - 1).const_cast() };
            let old_head = a21.read(0, 0);
            a21.write(0, 0, E::faer_one());

            let mut a_right = unsafe { a.rb().submatrix(0, k + 1, k + 1, n - k - 1).const_cast() };
            let tau_inv = householder.read(k_local, k_local).faer_inv();

            let nrows = k_local + 1;
            let (mut dot, _) = temp_mat_uninit::<E>(nrows, 1, stack.rb_mut());
            let mut dot = dot.as_mut();
            matmul(
                dot.rb_mut(),
                a_right.rb().subrows(k_base, nrows),
                a21.rb(),
                None,
                tau_inv.faer_neg(),
                parallelism,
            );
            matmul(
                a_right.rb_mut().subrows(k_base, nrows),
                dot.rb(),
                a21.rb().adjoint(),
                Some(E::faer_one()),
                E::faer_one(),
                parallelism,
            );

            a21.write(0, 0, old_head);
        }

        let mut a_right = unsafe {
            a.rb()
                .submatrix(0, k_base + 1, k_base, n - 1 - k_base)
                .const_cast()
        };
        apply_block_householder_on_the_right_in_place_with_conj(
            essentials,
            householder,
            Conj::No,
            a_right.rb_mut(),
            parallelism,
            stack.rb_mut(),
        );

        k_base += bs;
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
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn test_make_hessenberg() {
        for n in [10, 20, 64, 128, 1024] {
            for parallelism in [Parallelism::None, Parallelism::Rayon(4)] {
                let a = Mat::from_fn(n, n, |_, _| c64::new(rand::random(), rand::random()));

                let mut h = a.clone();
                let householder_blocksize = Ord::min(8, n - 1);
                let mut householder = Mat::zeros(n - 1, householder_blocksize);
                make_hessenberg_in_place(
                    h.as_mut(),
                    householder.as_mut(),
                    parallelism,
                    make_stack!(make_hessenberg_in_place_req::<c64>(
                        n,
                        householder_blocksize,
                        parallelism
                    )),
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
                        apply_block_householder_sequence_on_the_right_in_place_req::<c64>(
                            n - 1,
                            1,
                            n
                        )
                    ),
                );

                for j in 0..n {
                    for i in 0..Ord::min(n, j + 2) {
                        assert_approx_eq!(copy.read(i, j), h.read(i, j));
                    }
                }

                for j in 0..n {
                    for i in j + 2..n {
                        assert_approx_eq!(copy.read(i, j), c64::faer_zero());
                    }
                }
            }
        }
    }
}
