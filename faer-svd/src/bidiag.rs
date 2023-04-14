use assert2::assert;
use coe::Coerce;
use core::slice;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    c32, c64, for_each_raw, householder::make_householder_in_place, mul::matmul, par_split_indices,
    parallelism_degree, temp_mat_req, temp_mat_uninit, temp_mat_zeroed, zipped, ComplexField, Conj,
    Entity, MatMut, MatRef, Parallelism,
};
use pulp::Simd;
use reborrow::*;

pub fn bidiagonalize_in_place_req<E: Entity>(
    m: usize,
    n: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        temp_mat_req::<E>(n, 1)?,
        temp_mat_req::<E>(m, 1)?,
        temp_mat_req::<E>(
            m,
            match parallelism {
                Parallelism::None => 1,
                Parallelism::Rayon(n_threads) => {
                    if n_threads == 0 {
                        rayon::current_num_threads()
                    } else {
                        n_threads
                    }
                }
            },
        )?,
    ])
}

pub fn bidiagonalize_in_place<E: ComplexField>(
    mut a: MatMut<'_, E>,
    mut householder_left: MatMut<'_, E>,
    mut householder_right: MatMut<'_, E>,
    parallelism: Parallelism,
    mut stack: DynStack<'_>,
) {
    let m = a.nrows();
    let n = a.ncols();

    assert!(m >= n);

    let n_threads = match parallelism {
        Parallelism::None => 1,
        Parallelism::Rayon(n_threads) => {
            if n_threads == 0 {
                rayon::current_num_threads()
            } else {
                n_threads
            }
        }
    };

    let (mut y, mut stack) = unsafe { temp_mat_uninit::<E>(n, 1, stack.rb_mut()) };
    let mut y = y.as_mut();
    let (mut z, mut stack) = unsafe { temp_mat_uninit::<E>(m, 1, stack.rb_mut()) };
    let mut z = z.as_mut();

    let (mut z_tmp, _) = temp_mat_zeroed::<E>(m, n_threads, stack.rb_mut());
    let mut z_tmp = z_tmp.as_mut();

    let mut tl = E::zero();
    let mut tr = E::zero();
    let mut a01 = E::zero();

    for k in 0..n {
        let [a_left, a_right] = a.rb_mut().split_at_col(k);
        let [mut a_top, mut a_cur] = a_right.split_at_row(k);

        let m = a_cur.nrows();
        let n = a_cur.ncols();

        let [mut a_col, a_right] = a_cur.rb_mut().split_at_col(1);
        let [mut a_row, mut a_next] = a_right.split_at_row(1);

        if k > 0 {
            let u = a_left.rb().submatrix(k, k - 1, m, 1);
            let mut v = a_top.rb_mut().submatrix(k - 1, 0, 1, n);
            let y = y.rb().submatrix(k - 1, 0, n, 1);
            let z = z.rb().submatrix(k - 1, 0, m, 1);

            let f0 = y.read(0, 0).conj().mul(&tl.inv());
            let f1 = v.read(0, 0).conj().mul(&tr.inv());

            zipped!(a_col.rb_mut(), u, z).for_each(|mut a, b, c| {
                a.write(a.read().sub(&f0.mul(&b.read())).sub(&f1.mul(&c.read())))
            });

            let f0 = u.read(0, 0).mul(&tl.inv());
            let f1 = z.read(0, 0).mul(&tr.inv());
            zipped!(
                a_row.rb_mut(),
                y.submatrix(1, 0, n - 1, 1).transpose(),
                v.rb().submatrix(0, 1, 1, n - 1),
            )
            .for_each(|mut a, b, c| {
                a.write(
                    a.read()
                        .sub(&f0.mul(&b.read().conj()))
                        .sub(&f1.mul(&c.read().conj())),
                )
            });

            v.write(0, 0, a01);
        }

        let mut y = y.rb_mut().submatrix(k, 0, n - 1, 1);
        let mut z = z.rb_mut().submatrix(k, 0, m - 1, 1);
        let z_tmp = z_tmp.rb_mut().submatrix(k, 0, m - 1, n_threads);

        let tl_prev = tl.clone();
        let a00;
        (tl, a00) = {
            let head = a_col.read(0, 0);
            let essential = a_col.rb_mut().col(0).subrows(1, m - 1);
            let mut tail_squared_norm = E::Real::zero();
            for idx in 0..m - 1 {
                let x = essential.read(idx, 0);
                tail_squared_norm = tail_squared_norm.add(&x.mul(&x.conj()).real());
            }
            make_householder_in_place(Some(essential), head, tail_squared_norm)
        };
        a_col.write(0, 0, a00);
        householder_left.write(k, 0, tl.clone());

        if n == 1 {
            break;
        }

        let u = a_col.rb().submatrix(1, 0, m - 1, 1);

        bidiag_fused_op(
            k,
            m,
            n,
            tl_prev,
            tl.clone(),
            tr.clone(),
            z_tmp,
            a_left,
            a_top,
            a_next.rb_mut(),
            y.rb_mut(),
            parallelism,
            z.rb_mut(),
            u,
            a_row.rb_mut(),
        );

        (tr, a01) = {
            let head = a_row.read(0, 0);
            let essential = a_row.rb().row(0).subcols(1, n - 2).transpose();
            let mut tail_squared_norm = E::Real::zero();
            for idx in 0..n - 2 {
                let x = essential.read(idx, 0);
                tail_squared_norm = tail_squared_norm.add(&x.mul(&x.conj()).real());
            }
            make_householder_in_place(None, head, tail_squared_norm)
        };
        householder_right.write(k, 0, tr.clone());

        let diff = a_row.read(0, 0).sub(&a01);

        if diff != E::zero() {
            let f = diff.inv().conj();
            zipped!(a_row.rb_mut().row(0).subcols(1, n - 2).transpose())
                .for_each(|mut x| x.write(x.read().conj().mul(&f)));

            zipped!(z.rb_mut().col(0), a_next.rb().col(0))
                .for_each(|mut z, a| z.write(f.mul(&z.read().sub(&a01.conj().mul(&a.read())))));
        }

        a_row.write(0, 0, E::one());
        let b = faer_core::mul::inner_prod::inner_prod_with_conj(
            y.rb().col(0),
            Conj::Yes,
            a_row.rb().row(0).transpose(),
            Conj::No,
        );

        let factor = b.mul(&tl.inv()).neg();
        zipped!(z.rb_mut(), u).for_each(|mut z, u| z.write(z.read().add(&u.read().mul(&factor))));
    }
}

#[allow(dead_code)]
fn bidiag_fused_op_reference<E: ComplexField>(
    k: usize,
    m: usize,
    n: usize,
    tl_prev: E,
    tl: E,
    tr: E,
    _z_tmp: MatMut<'_, E>,
    a_left: MatMut<'_, E>,
    a_top: MatMut<'_, E>,
    mut a_next: MatMut<'_, E>,
    mut y: MatMut<'_, E>,
    parallelism: Parallelism,
    mut z: MatMut<'_, E>,
    u: MatRef<'_, E>,
    mut a_row: MatMut<'_, E>,
) {
    if k > 0 {
        let u_prev = a_left.rb().submatrix(k + 1, k - 1, m - 1, 1);
        let v_prev = a_top.rb().submatrix(k - 1, 1, 1, n - 1);
        matmul(
            a_next.rb_mut(),
            u_prev,
            y.rb().adjoint(),
            Some(E::one()),
            tl_prev.inv().neg(),
            parallelism,
        );
        matmul(
            a_next.rb_mut(),
            z.rb(),
            v_prev.conjugate(),
            Some(E::one()),
            tr.inv().neg(),
            parallelism,
        );

        matmul(
            y.rb_mut(),
            a_next.rb().adjoint(),
            u,
            None,
            E::one(),
            parallelism,
        );
        zipped!(y.rb_mut(), a_row.rb().transpose())
            .for_each(|mut dst, src| dst.write(dst.read().add(&src.read().conj())));
        let tl_inv = tl.inv();
        zipped!(a_row.rb_mut(), y.rb().transpose())
            .for_each(|mut dst, src| dst.write(dst.read().sub(&src.read().conj().mul(&tl_inv))));
        matmul(
            z.rb_mut(),
            a_next.rb(),
            a_row.rb().adjoint(),
            None,
            E::one(),
            parallelism,
        );
    } else {
        matmul(
            y.rb_mut(),
            a_next.rb().adjoint(),
            u,
            None,
            E::one(),
            parallelism,
        );
        zipped!(y.rb_mut(), a_row.rb().transpose())
            .for_each(|mut dst, src| dst.write(dst.read().add(&src.read().conj())));
        let tl_inv = tl.inv();
        zipped!(a_row.rb_mut(), y.rb().transpose())
            .for_each(|mut dst, src| dst.write(dst.read().sub(&src.read().conj().mul(&tl_inv))));
        matmul(
            z.rb_mut(),
            a_next.rb(),
            a_row.rb().adjoint(),
            None,
            E::one(),
            parallelism,
        );
    }
}

fn bidiag_fused_op_step0_f64(
    arch: pulp::Arch,

    // update a_next
    a_j: &mut [f64],
    z: &[f64],
    u_prev: &[f64],
    u_rhs: f64,
    z_rhs: f64,

    // compute yj
    u: &[f64],
) -> f64 {
    struct Impl<'a> {
        a_j: &'a mut [f64],
        z: &'a [f64],
        u_prev: &'a [f64],
        u_rhs: f64,
        z_rhs: f64,
        u: &'a [f64],
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = f64;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self {
                a_j,
                z,
                u_prev,
                u_rhs,
                z_rhs,
                u,
            } = self;
            let (a_j_head, a_j_tail) = S::f64s_as_mut_simd(a_j);
            let (z_head, z_tail) = S::f64s_as_simd(z);
            let (u_prev_head, u_prev_tail) = S::f64s_as_simd(u_prev);
            let (u_head, u_tail) = S::f64s_as_simd(u);

            let (a_j_head4, a_j_head1) = pulp::as_arrays_mut::<4, _>(a_j_head);
            let (z_head4, z_head1) = pulp::as_arrays::<4, _>(z_head);
            let (u_prev_head4, u_prev_head1) = pulp::as_arrays::<4, _>(u_prev_head);
            let (u_head4, u_head1) = pulp::as_arrays::<4, _>(u_head);

            let mut sum_v0 = simd.f64s_splat(0.0_f64);
            let mut sum_v1 = simd.f64s_splat(0.0_f64);
            let mut sum_v2 = simd.f64s_splat(0.0_f64);
            let mut sum_v3 = simd.f64s_splat(0.0_f64);

            let u_rhs_v = simd.f64s_splat(u_rhs);
            let z_rhs_v = simd.f64s_splat(z_rhs);

            for (((aij, zi), u_prev_i), ui) in a_j_head4
                .iter_mut()
                .zip(z_head4)
                .zip(u_prev_head4)
                .zip(u_head4)
            {
                let aij_new0 = simd.f64s_mul_adde(
                    u_prev_i[0],
                    simd.f64s_neg(u_rhs_v),
                    simd.f64s_mul_adde(zi[0], simd.f64s_neg(z_rhs_v), aij[0]),
                );
                let aij_new1 = simd.f64s_mul_adde(
                    u_prev_i[1],
                    simd.f64s_neg(u_rhs_v),
                    simd.f64s_mul_adde(zi[1], simd.f64s_neg(z_rhs_v), aij[1]),
                );
                let aij_new2 = simd.f64s_mul_adde(
                    u_prev_i[2],
                    simd.f64s_neg(u_rhs_v),
                    simd.f64s_mul_adde(zi[2], simd.f64s_neg(z_rhs_v), aij[2]),
                );
                let aij_new3 = simd.f64s_mul_adde(
                    u_prev_i[3],
                    simd.f64s_neg(u_rhs_v),
                    simd.f64s_mul_adde(zi[3], simd.f64s_neg(z_rhs_v), aij[3]),
                );
                sum_v0 = simd.f64s_mul_adde(aij_new0, ui[0], sum_v0);
                sum_v1 = simd.f64s_mul_adde(aij_new1, ui[1], sum_v1);
                sum_v2 = simd.f64s_mul_adde(aij_new2, ui[2], sum_v2);
                sum_v3 = simd.f64s_mul_adde(aij_new3, ui[3], sum_v3);
                aij[0] = aij_new0;
                aij[1] = aij_new1;
                aij[2] = aij_new2;
                aij[3] = aij_new3;
            }

            sum_v0 = simd.f64s_add(sum_v0, sum_v1);
            sum_v2 = simd.f64s_add(sum_v2, sum_v3);
            sum_v0 = simd.f64s_add(sum_v0, sum_v2);

            for (((aij, zi), u_prev_i), ui) in a_j_head1
                .iter_mut()
                .zip(z_head1)
                .zip(u_prev_head1)
                .zip(u_head1)
            {
                let aij_new = simd.f64s_mul_adde(
                    *u_prev_i,
                    simd.f64s_neg(u_rhs_v),
                    simd.f64s_mul_adde(*zi, simd.f64s_neg(z_rhs_v), *aij),
                );
                sum_v0 = simd.f64s_mul_adde(aij_new, *ui, sum_v0);
                *aij = aij_new;
            }

            let mut sum = simd.f64s_reduce_sum(sum_v0);

            for (((aij, zi), u_prev_i), ui) in
                a_j_tail.iter_mut().zip(z_tail).zip(u_prev_tail).zip(u_tail)
            {
                let aij_new = f64::mul_add(*u_prev_i, -u_rhs, f64::mul_add(*zi, -z_rhs, *aij));
                sum = f64::mul_add(aij_new, *ui, sum);
                *aij = aij_new;
            }

            sum
        }
    }

    arch.dispatch(Impl {
        a_j,
        z,
        u_prev,
        u_rhs,
        z_rhs,
        u,
    })
}

fn bidiag_fused_op_step1_f64(
    arch: pulp::Arch,

    // update z
    z: &mut [f64],
    a_j: &[f64],
    rhs: f64,
) {
    struct Impl<'a> {
        z: &'a mut [f64],
        a_j: &'a [f64],
        rhs: f64,
    }
    impl pulp::WithSimd for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self { z, a_j, rhs } = self;
            let (z_head, z_tail) = S::f64s_as_mut_simd(z);
            let (a_j_head, a_j_tail) = S::f64s_as_simd(a_j);
            let rhs_v = simd.f64s_splat(rhs);

            for (zi, aij) in z_head.iter_mut().zip(a_j_head) {
                *zi = simd.f64s_mul_adde(*aij, rhs_v, *zi);
            }
            for (zi, aij) in z_tail.iter_mut().zip(a_j_tail) {
                *zi = f64::mul_add(*aij, rhs, *zi);
            }
        }
    }
    arch.dispatch(Impl { z, a_j, rhs })
}

fn bidiag_fused_op_step0_c64(
    arch: pulp::Arch,

    // update a_next
    a_j: &mut [c64],
    z: &[c64],
    u_prev: &[c64],
    u_rhs: c64,
    z_rhs: c64,

    // compute yj
    u: &[c64],
) -> c64 {
    struct Impl<'a> {
        a_j: &'a mut [c64],
        z: &'a [c64],
        u_prev: &'a [c64],
        u_rhs: c64,
        z_rhs: c64,
        u: &'a [c64],
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = c64;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self {
                a_j,
                z,
                u_prev,
                u_rhs,
                z_rhs,
                u,
            } = self;
            let u_rhs = u_rhs.into();
            let z_rhs = z_rhs.into();

            let (a_j_head, a_j_tail) = S::c64s_as_mut_simd(bytemuck::cast_slice_mut(a_j));
            let (z_head, z_tail) = S::c64s_as_simd(bytemuck::cast_slice(z));
            let (u_prev_head, u_prev_tail) = S::c64s_as_simd(bytemuck::cast_slice(u_prev));
            let (u_head, u_tail) = S::c64s_as_simd(bytemuck::cast_slice(u));

            let (a_j_head4, a_j_head1) = pulp::as_arrays_mut::<4, _>(a_j_head);
            let (z_head4, z_head1) = pulp::as_arrays::<4, _>(z_head);
            let (u_prev_head4, u_prev_head1) = pulp::as_arrays::<4, _>(u_prev_head);
            let (u_head4, u_head1) = pulp::as_arrays::<4, _>(u_head);

            let mut sum_v0 = simd.c64s_splat(c64::zero().into());
            let mut sum_v1 = simd.c64s_splat(c64::zero().into());
            let mut sum_v2 = simd.c64s_splat(c64::zero().into());
            let mut sum_v3 = simd.c64s_splat(c64::zero().into());

            let u_rhs_v = simd.c64s_neg(simd.c64s_splat(u_rhs));
            let z_rhs_v = simd.c64s_neg(simd.c64s_splat(z_rhs));

            for (((aij, zi), u_prev_i), ui) in a_j_head4
                .iter_mut()
                .zip(z_head4)
                .zip(u_prev_head4)
                .zip(u_head4)
            {
                let aij_new0 = simd.c64s_mul_adde(
                    u_prev_i[0],
                    u_rhs_v,
                    simd.c64s_mul_adde(zi[0], z_rhs_v, aij[0]),
                );
                let aij_new1 = simd.c64s_mul_adde(
                    u_prev_i[1],
                    u_rhs_v,
                    simd.c64s_mul_adde(zi[1], z_rhs_v, aij[1]),
                );
                let aij_new2 = simd.c64s_mul_adde(
                    u_prev_i[2],
                    u_rhs_v,
                    simd.c64s_mul_adde(zi[2], z_rhs_v, aij[2]),
                );
                let aij_new3 = simd.c64s_mul_adde(
                    u_prev_i[3],
                    u_rhs_v,
                    simd.c64s_mul_adde(zi[3], z_rhs_v, aij[3]),
                );
                sum_v0 = simd.c64s_conj_mul_adde(aij_new0, ui[0], sum_v0);
                sum_v1 = simd.c64s_conj_mul_adde(aij_new1, ui[1], sum_v1);
                sum_v2 = simd.c64s_conj_mul_adde(aij_new2, ui[2], sum_v2);
                sum_v3 = simd.c64s_conj_mul_adde(aij_new3, ui[3], sum_v3);
                aij[0] = aij_new0;
                aij[1] = aij_new1;
                aij[2] = aij_new2;
                aij[3] = aij_new3;
            }

            sum_v0 = simd.c64s_add(sum_v0, sum_v1);
            sum_v2 = simd.c64s_add(sum_v2, sum_v3);
            sum_v0 = simd.c64s_add(sum_v0, sum_v2);

            for (((aij, zi), u_prev_i), ui) in a_j_head1
                .iter_mut()
                .zip(z_head1)
                .zip(u_prev_head1)
                .zip(u_head1)
            {
                let aij_new =
                    simd.c64s_mul_adde(*u_prev_i, u_rhs_v, simd.c64s_mul_adde(*zi, z_rhs_v, *aij));
                sum_v0 = simd.c64s_conj_mul_adde(aij_new, *ui, sum_v0);
                *aij = aij_new;
            }

            let mut sum = simd.c64s_reduce_sum(sum_v0);

            for (((aij, zi), u_prev_i), ui) in
                a_j_tail.iter_mut().zip(z_tail).zip(u_prev_tail).zip(u_tail)
            {
                let aij_new = *aij - *u_prev_i * u_rhs - *zi * z_rhs;
                sum = aij_new.conj() * *ui + sum;
                *aij = aij_new;
            }

            sum.into()
        }
    }

    arch.dispatch(Impl {
        a_j,
        z,
        u_prev,
        u_rhs,
        z_rhs,
        u,
    })
}

fn bidiag_fused_op_step1_c64(
    arch: pulp::Arch,

    // update z
    z: &mut [c64],
    a_j: &[c64],
    rhs: c64,
) {
    struct Impl<'a> {
        z: &'a mut [c64],
        a_j: &'a [c64],
        rhs: c64,
    }
    impl pulp::WithSimd for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self { z, a_j, rhs } = self;
            let rhs = rhs.into();
            let (z_head, z_tail) = S::c64s_as_mut_simd(bytemuck::cast_slice_mut(z));
            let (a_j_head, a_j_tail) = S::c64s_as_simd(bytemuck::cast_slice(a_j));
            let rhs_v = simd.c64s_splat(rhs);

            for (zi, aij) in z_head.iter_mut().zip(a_j_head) {
                *zi = simd.c64s_mul_adde(*aij, rhs_v, *zi);
            }
            for (zi, aij) in z_tail.iter_mut().zip(a_j_tail) {
                *zi = *aij * rhs + *zi;
            }
        }
    }
    arch.dispatch(Impl { z, a_j, rhs })
}

fn bidiag_fused_op_step0_f32(
    arch: pulp::Arch,

    // update a_next
    a_j: &mut [f32],
    z: &[f32],
    u_prev: &[f32],
    u_rhs: f32,
    z_rhs: f32,

    // compute yj
    u: &[f32],
) -> f32 {
    struct Impl<'a> {
        a_j: &'a mut [f32],
        z: &'a [f32],
        u_prev: &'a [f32],
        u_rhs: f32,
        z_rhs: f32,
        u: &'a [f32],
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = f32;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self {
                a_j,
                z,
                u_prev,
                u_rhs,
                z_rhs,
                u,
            } = self;
            let (a_j_head, a_j_tail) = S::f32s_as_mut_simd(a_j);
            let (z_head, z_tail) = S::f32s_as_simd(z);
            let (u_prev_head, u_prev_tail) = S::f32s_as_simd(u_prev);
            let (u_head, u_tail) = S::f32s_as_simd(u);

            let (a_j_head4, a_j_head1) = pulp::as_arrays_mut::<4, _>(a_j_head);
            let (z_head4, z_head1) = pulp::as_arrays::<4, _>(z_head);
            let (u_prev_head4, u_prev_head1) = pulp::as_arrays::<4, _>(u_prev_head);
            let (u_head4, u_head1) = pulp::as_arrays::<4, _>(u_head);

            let mut sum_v0 = simd.f32s_splat(0.0_f32);
            let mut sum_v1 = simd.f32s_splat(0.0_f32);
            let mut sum_v2 = simd.f32s_splat(0.0_f32);
            let mut sum_v3 = simd.f32s_splat(0.0_f32);

            let u_rhs_v = simd.f32s_splat(u_rhs);
            let z_rhs_v = simd.f32s_splat(z_rhs);

            for (((aij, zi), u_prev_i), ui) in a_j_head4
                .iter_mut()
                .zip(z_head4)
                .zip(u_prev_head4)
                .zip(u_head4)
            {
                let aij_new0 = simd.f32s_mul_adde(
                    u_prev_i[0],
                    simd.f32s_neg(u_rhs_v),
                    simd.f32s_mul_adde(zi[0], simd.f32s_neg(z_rhs_v), aij[0]),
                );
                let aij_new1 = simd.f32s_mul_adde(
                    u_prev_i[1],
                    simd.f32s_neg(u_rhs_v),
                    simd.f32s_mul_adde(zi[1], simd.f32s_neg(z_rhs_v), aij[1]),
                );
                let aij_new2 = simd.f32s_mul_adde(
                    u_prev_i[2],
                    simd.f32s_neg(u_rhs_v),
                    simd.f32s_mul_adde(zi[2], simd.f32s_neg(z_rhs_v), aij[2]),
                );
                let aij_new3 = simd.f32s_mul_adde(
                    u_prev_i[3],
                    simd.f32s_neg(u_rhs_v),
                    simd.f32s_mul_adde(zi[3], simd.f32s_neg(z_rhs_v), aij[3]),
                );
                sum_v0 = simd.f32s_mul_adde(aij_new0, ui[0], sum_v0);
                sum_v1 = simd.f32s_mul_adde(aij_new1, ui[1], sum_v1);
                sum_v2 = simd.f32s_mul_adde(aij_new2, ui[2], sum_v2);
                sum_v3 = simd.f32s_mul_adde(aij_new3, ui[3], sum_v3);
                aij[0] = aij_new0;
                aij[1] = aij_new1;
                aij[2] = aij_new2;
                aij[3] = aij_new3;
            }

            sum_v0 = simd.f32s_add(sum_v0, sum_v1);
            sum_v2 = simd.f32s_add(sum_v2, sum_v3);
            sum_v0 = simd.f32s_add(sum_v0, sum_v2);

            for (((aij, zi), u_prev_i), ui) in a_j_head1
                .iter_mut()
                .zip(z_head1)
                .zip(u_prev_head1)
                .zip(u_head1)
            {
                let aij_new = simd.f32s_mul_adde(
                    *u_prev_i,
                    simd.f32s_neg(u_rhs_v),
                    simd.f32s_mul_adde(*zi, simd.f32s_neg(z_rhs_v), *aij),
                );
                sum_v0 = simd.f32s_mul_adde(aij_new, *ui, sum_v0);
                *aij = aij_new;
            }

            let mut sum = simd.f32s_reduce_sum(sum_v0);

            for (((aij, zi), u_prev_i), ui) in
                a_j_tail.iter_mut().zip(z_tail).zip(u_prev_tail).zip(u_tail)
            {
                let aij_new = f32::mul_add(*u_prev_i, -u_rhs, f32::mul_add(*zi, -z_rhs, *aij));
                sum = f32::mul_add(aij_new, *ui, sum);
                *aij = aij_new;
            }

            sum
        }
    }

    arch.dispatch(Impl {
        a_j,
        z,
        u_prev,
        u_rhs,
        z_rhs,
        u,
    })
}

fn bidiag_fused_op_step1_f32(
    arch: pulp::Arch,

    // update z
    z: &mut [f32],
    a_j: &[f32],
    rhs: f32,
) {
    struct Impl<'a> {
        z: &'a mut [f32],
        a_j: &'a [f32],
        rhs: f32,
    }
    impl pulp::WithSimd for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self { z, a_j, rhs } = self;
            let (z_head, z_tail) = S::f32s_as_mut_simd(z);
            let (a_j_head, a_j_tail) = S::f32s_as_simd(a_j);
            let rhs_v = simd.f32s_splat(rhs);

            for (zi, aij) in z_head.iter_mut().zip(a_j_head) {
                *zi = simd.f32s_mul_adde(*aij, rhs_v, *zi);
            }
            for (zi, aij) in z_tail.iter_mut().zip(a_j_tail) {
                *zi = f32::mul_add(*aij, rhs, *zi);
            }
        }
    }
    arch.dispatch(Impl { z, a_j, rhs })
}

fn bidiag_fused_op_step0_c32(
    arch: pulp::Arch,

    // update a_next
    a_j: &mut [c32],
    z: &[c32],
    u_prev: &[c32],
    u_rhs: c32,
    z_rhs: c32,

    // compute yj
    u: &[c32],
) -> c32 {
    struct Impl<'a> {
        a_j: &'a mut [c32],
        z: &'a [c32],
        u_prev: &'a [c32],
        u_rhs: c32,
        z_rhs: c32,
        u: &'a [c32],
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = c32;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self {
                a_j,
                z,
                u_prev,
                u_rhs,
                z_rhs,
                u,
            } = self;
            let u_rhs = u_rhs.into();
            let z_rhs = z_rhs.into();

            let (a_j_head, a_j_tail) = S::c32s_as_mut_simd(bytemuck::cast_slice_mut(a_j));
            let (z_head, z_tail) = S::c32s_as_simd(bytemuck::cast_slice(z));
            let (u_prev_head, u_prev_tail) = S::c32s_as_simd(bytemuck::cast_slice(u_prev));
            let (u_head, u_tail) = S::c32s_as_simd(bytemuck::cast_slice(u));

            let (a_j_head4, a_j_head1) = pulp::as_arrays_mut::<4, _>(a_j_head);
            let (z_head4, z_head1) = pulp::as_arrays::<4, _>(z_head);
            let (u_prev_head4, u_prev_head1) = pulp::as_arrays::<4, _>(u_prev_head);
            let (u_head4, u_head1) = pulp::as_arrays::<4, _>(u_head);

            let mut sum_v0 = simd.c32s_splat(c32::zero().into());
            let mut sum_v1 = simd.c32s_splat(c32::zero().into());
            let mut sum_v2 = simd.c32s_splat(c32::zero().into());
            let mut sum_v3 = simd.c32s_splat(c32::zero().into());

            let u_rhs_v = simd.c32s_neg(simd.c32s_splat(u_rhs));
            let z_rhs_v = simd.c32s_neg(simd.c32s_splat(z_rhs));

            for (((aij, zi), u_prev_i), ui) in a_j_head4
                .iter_mut()
                .zip(z_head4)
                .zip(u_prev_head4)
                .zip(u_head4)
            {
                let aij_new0 = simd.c32s_mul_adde(
                    u_prev_i[0],
                    u_rhs_v,
                    simd.c32s_mul_adde(zi[0], z_rhs_v, aij[0]),
                );
                let aij_new1 = simd.c32s_mul_adde(
                    u_prev_i[1],
                    u_rhs_v,
                    simd.c32s_mul_adde(zi[1], z_rhs_v, aij[1]),
                );
                let aij_new2 = simd.c32s_mul_adde(
                    u_prev_i[2],
                    u_rhs_v,
                    simd.c32s_mul_adde(zi[2], z_rhs_v, aij[2]),
                );
                let aij_new3 = simd.c32s_mul_adde(
                    u_prev_i[3],
                    u_rhs_v,
                    simd.c32s_mul_adde(zi[3], z_rhs_v, aij[3]),
                );
                sum_v0 = simd.c32s_conj_mul_adde(aij_new0, ui[0], sum_v0);
                sum_v1 = simd.c32s_conj_mul_adde(aij_new1, ui[1], sum_v1);
                sum_v2 = simd.c32s_conj_mul_adde(aij_new2, ui[2], sum_v2);
                sum_v3 = simd.c32s_conj_mul_adde(aij_new3, ui[3], sum_v3);
                aij[0] = aij_new0;
                aij[1] = aij_new1;
                aij[2] = aij_new2;
                aij[3] = aij_new3;
            }

            sum_v0 = simd.c32s_add(sum_v0, sum_v1);
            sum_v2 = simd.c32s_add(sum_v2, sum_v3);
            sum_v0 = simd.c32s_add(sum_v0, sum_v2);

            for (((aij, zi), u_prev_i), ui) in a_j_head1
                .iter_mut()
                .zip(z_head1)
                .zip(u_prev_head1)
                .zip(u_head1)
            {
                let aij_new =
                    simd.c32s_mul_adde(*u_prev_i, u_rhs_v, simd.c32s_mul_adde(*zi, z_rhs_v, *aij));
                sum_v0 = simd.c32s_conj_mul_adde(aij_new, *ui, sum_v0);
                *aij = aij_new;
            }

            let mut sum = simd.c32s_reduce_sum(sum_v0);

            for (((aij, zi), u_prev_i), ui) in
                a_j_tail.iter_mut().zip(z_tail).zip(u_prev_tail).zip(u_tail)
            {
                let aij_new = *aij - *u_prev_i * u_rhs - *zi * z_rhs;
                sum = aij_new.conj() * *ui + sum;
                *aij = aij_new;
            }

            sum.into()
        }
    }

    arch.dispatch(Impl {
        a_j,
        z,
        u_prev,
        u_rhs,
        z_rhs,
        u,
    })
}

fn bidiag_fused_op_step1_c32(
    arch: pulp::Arch,

    // update z
    z: &mut [c32],
    a_j: &[c32],
    rhs: c32,
) {
    struct Impl<'a> {
        z: &'a mut [c32],
        a_j: &'a [c32],
        rhs: c32,
    }
    impl pulp::WithSimd for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self { z, a_j, rhs } = self;
            let rhs = rhs.into();
            let (z_head, z_tail) = S::c32s_as_mut_simd(bytemuck::cast_slice_mut(z));
            let (a_j_head, a_j_tail) = S::c32s_as_simd(bytemuck::cast_slice(a_j));
            let rhs_v = simd.c32s_splat(rhs);

            for (zi, aij) in z_head.iter_mut().zip(a_j_head) {
                *zi = simd.c32s_mul_adde(*aij, rhs_v, *zi);
            }
            for (zi, aij) in z_tail.iter_mut().zip(a_j_tail) {
                *zi = *aij * rhs + *zi;
            }
        }
    }
    arch.dispatch(Impl { z, a_j, rhs })
}

fn bidiag_fused_op_process_batch<E: ComplexField>(
    arch: pulp::Arch,
    mut z_tmp: MatMut<'_, E>,
    mut a_next: MatMut<'_, E>,
    mut a_row: MatMut<'_, E>,
    u: MatRef<'_, E>,
    u_prev: MatRef<'_, E>,
    v_prev: MatRef<'_, E>,
    mut y: MatMut<'_, E>,
    z: MatRef<'_, E>,
    tl_prev_inv: E,
    tr_prev_inv: E,
    tl_inv: E,
) {
    let ncols = a_next.ncols();
    let nrows = a_next.nrows();
    for j in 0..ncols {
        let u_rhs = y.read(j, 0).conj().mul(&tl_prev_inv);
        let z_rhs = v_prev.read(0, j).conj().mul(&tr_prev_inv);

        let yj = if coe::is_same::<f64, E>() {
            let a_next: MatMut<'_, f64> = a_next.rb_mut().coerce();
            let z: MatRef<'_, f64> = z.coerce();
            let u_prev: MatRef<'_, f64> = u_prev.coerce();
            let u: MatRef<'_, f64> = u.coerce();
            unsafe {
                coe::coerce_static(bidiag_fused_op_step0_f64(
                    arch,
                    slice::from_raw_parts_mut(a_next.ptr_at(0, j), nrows),
                    slice::from_raw_parts(z.as_ptr(), nrows),
                    slice::from_raw_parts(u_prev.as_ptr(), nrows),
                    coe::coerce_static(u_rhs),
                    coe::coerce_static(z_rhs),
                    slice::from_raw_parts(u.as_ptr(), nrows),
                ))
            }
        } else if coe::is_same::<c64, E>() {
            let a_next: MatMut<'_, c64> = a_next.rb_mut().coerce();
            let z: MatRef<'_, c64> = z.coerce();
            let u_prev: MatRef<'_, c64> = u_prev.coerce();
            let u: MatRef<'_, c64> = u.coerce();
            unsafe {
                coe::coerce_static(bidiag_fused_op_step0_c64(
                    arch,
                    slice::from_raw_parts_mut(a_next.ptr_at(0, j), nrows),
                    slice::from_raw_parts(z.as_ptr(), nrows),
                    slice::from_raw_parts(u_prev.as_ptr(), nrows),
                    coe::coerce_static(u_rhs),
                    coe::coerce_static(z_rhs),
                    slice::from_raw_parts(u.as_ptr(), nrows),
                ))
            }
        } else if coe::is_same::<f32, E>() {
            let a_next: MatMut<'_, f32> = a_next.rb_mut().coerce();
            let z: MatRef<'_, f32> = z.coerce();
            let u_prev: MatRef<'_, f32> = u_prev.coerce();
            let u: MatRef<'_, f32> = u.coerce();
            unsafe {
                coe::coerce_static(bidiag_fused_op_step0_f32(
                    arch,
                    slice::from_raw_parts_mut(a_next.ptr_at(0, j), nrows),
                    slice::from_raw_parts(z.as_ptr(), nrows),
                    slice::from_raw_parts(u_prev.as_ptr(), nrows),
                    coe::coerce_static(u_rhs),
                    coe::coerce_static(z_rhs),
                    slice::from_raw_parts(u.as_ptr(), nrows),
                ))
            }
        } else if coe::is_same::<c32, E>() {
            let a_next: MatMut<'_, c32> = a_next.rb_mut().coerce();
            let z: MatRef<'_, c32> = z.coerce();
            let u_prev: MatRef<'_, c32> = u_prev.coerce();
            let u: MatRef<'_, c32> = u.coerce();
            unsafe {
                coe::coerce_static(bidiag_fused_op_step0_c32(
                    arch,
                    slice::from_raw_parts_mut(a_next.ptr_at(0, j), nrows),
                    slice::from_raw_parts(z.as_ptr(), nrows),
                    slice::from_raw_parts(u_prev.as_ptr(), nrows),
                    coe::coerce_static(u_rhs),
                    coe::coerce_static(z_rhs),
                    slice::from_raw_parts(u.as_ptr(), nrows),
                ))
            }
        } else {
            let mut yj = E::zero();
            for i in 0..nrows {
                unsafe {
                    a_next.write_unchecked(
                        i,
                        j,
                        a_next
                            .read_unchecked(i, j)
                            .sub(&u_prev.read_unchecked(i, 0).mul(&u_rhs))
                            .sub(&z.read_unchecked(i, 0).mul(&z_rhs)),
                    );

                    yj = yj.add(
                        &(a_next.read_unchecked(i, j))
                            .conj()
                            .mul(&u.read_unchecked(i, 0)),
                    );
                }
            }

            yj
        };
        y.write(j, 0, yj.add(&a_row.read(0, j).conj()));
        a_row.write(
            0,
            j,
            a_row.read(0, j).sub(&y.read(j, 0).conj().mul(&tl_inv)),
        );

        let rhs = a_row.read(0, j).conj();

        if coe::is_same::<f64, E>() {
            let a_next: MatMut<'_, f64> = a_next.rb_mut().coerce();
            let mut z_tmp: MatMut<'_, f64> = z_tmp.rb_mut().coerce();
            unsafe {
                bidiag_fused_op_step1_f64(
                    arch,
                    slice::from_raw_parts_mut(z_tmp.rb_mut().as_ptr(), nrows),
                    slice::from_raw_parts(a_next.rb().ptr_at(0, j), nrows),
                    coe::coerce_static(rhs),
                );
            }
        } else if coe::is_same::<c64, E>() {
            let a_next: MatMut<'_, c64> = a_next.rb_mut().coerce();
            let mut z_tmp: MatMut<'_, c64> = z_tmp.rb_mut().coerce();
            unsafe {
                bidiag_fused_op_step1_c64(
                    arch,
                    slice::from_raw_parts_mut(z_tmp.rb_mut().as_ptr(), nrows),
                    slice::from_raw_parts(a_next.rb().ptr_at(0, j), nrows),
                    coe::coerce_static(rhs),
                );
            }
        } else if coe::is_same::<f32, E>() {
            let a_next: MatMut<'_, f32> = a_next.rb_mut().coerce();
            let mut z_tmp: MatMut<'_, f32> = z_tmp.rb_mut().coerce();
            unsafe {
                bidiag_fused_op_step1_f32(
                    arch,
                    slice::from_raw_parts_mut(z_tmp.rb_mut().as_ptr(), nrows),
                    slice::from_raw_parts(a_next.rb().ptr_at(0, j), nrows),
                    coe::coerce_static(rhs),
                );
            }
        } else if coe::is_same::<c32, E>() {
            let a_next: MatMut<'_, c32> = a_next.rb_mut().coerce();
            let mut z_tmp: MatMut<'_, c32> = z_tmp.rb_mut().coerce();
            unsafe {
                bidiag_fused_op_step1_c32(
                    arch,
                    slice::from_raw_parts_mut(z_tmp.rb_mut().as_ptr(), nrows),
                    slice::from_raw_parts(a_next.rb().ptr_at(0, j), nrows),
                    coe::coerce_static(rhs),
                );
            }
        } else {
            for i in 0..nrows {
                unsafe {
                    let zi = z_tmp.read_unchecked(i, 0);
                    let aij = a_next.read_unchecked(i, j);
                    z_tmp.write_unchecked(i, 0, zi.add(&aij.mul(&rhs)));
                }
            }
        }
    }
}

fn bidiag_fused_op<E: ComplexField>(
    k: usize,
    m: usize,
    n: usize,
    tl_prev: E,
    tl: E,
    tr: E,
    mut z_tmp: MatMut<'_, E>,
    a_left: MatMut<'_, E>,
    a_top: MatMut<'_, E>,
    mut a_next: MatMut<'_, E>,
    mut y: MatMut<'_, E>,
    parallelism: Parallelism,
    mut z: MatMut<'_, E>,
    u: MatRef<'_, E>,
    mut a_row: MatMut<'_, E>,
) {
    let parallelism = if m * n < 128 * 128 {
        Parallelism::None
    } else {
        parallelism
    };
    if k > 0 {
        if a_next.row_stride() == 1 {
            let arch = pulp::Arch::new();

            let n_threads = parallelism_degree(parallelism);

            let u_prev = a_left.rb().submatrix(k + 1, k - 1, m - 1, 1).col(0);
            let v_prev = a_top.rb().submatrix(k - 1, 1, 1, n - 1).row(0);

            let tl_prev_inv = tl_prev.inv();
            let tr_prev_inv = tr.inv();
            let tl_inv = tl.inv();

            assert!(a_next.row_stride() == 1);
            assert!(u_prev.row_stride() == 1);
            assert!(u.row_stride() == 1);
            assert!(z.row_stride() == 1);

            match n_threads {
                1 => {
                    bidiag_fused_op_process_batch(
                        arch,
                        z_tmp.rb_mut().col(0),
                        a_next,
                        a_row.row(0),
                        u.col(0),
                        u_prev,
                        v_prev,
                        y.col(0),
                        z.rb().col(0),
                        tl_prev_inv,
                        tr_prev_inv,
                        tl_inv,
                    );
                }
                n_threads => {
                    for_each_raw(
                        n_threads,
                        |idx| {
                            let (col_start, ncols) =
                                par_split_indices(a_next.ncols(), idx, n_threads);
                            let z_tmp = unsafe { z_tmp.rb().col(idx).const_cast() };
                            let a_next =
                                unsafe { a_next.rb().subcols(col_start, ncols).const_cast() };
                            let a_row =
                                unsafe { a_row.rb().subcols(col_start, ncols).const_cast() };
                            let y = unsafe { y.rb().subrows(col_start, ncols).const_cast() };
                            let v_prev = v_prev.subcols(col_start, ncols);

                            bidiag_fused_op_process_batch(
                                arch,
                                z_tmp,
                                a_next,
                                a_row,
                                u,
                                u_prev,
                                v_prev,
                                y,
                                z.rb(),
                                tl_prev_inv.clone(),
                                tr_prev_inv.clone(),
                                tl_inv.clone(),
                            );
                        },
                        parallelism,
                    );
                }
            }

            let mut idx = 0;
            let mut first_init = true;
            while idx < n_threads {
                let bs = <usize as Ord>::min(2, n_threads - idx);
                let z_block = z_tmp.rb_mut().submatrix(0, idx, m - 1, bs);

                match bs {
                    1 => {
                        let z0 = unsafe { z_block.rb().col(0).const_cast() };
                        if first_init {
                            zipped!(z.rb_mut().col(0), z0).for_each(|mut z, mut z0| {
                                z.write(z0.read());
                                z0.write(E::zero());
                            });
                        } else {
                            zipped!(z.rb_mut().col(0), z0).for_each(|mut z, mut z0| {
                                z.write(z.read().add(&z0.read()));
                                z0.write(E::zero());
                            });
                        }
                    }
                    2 => {
                        let z0 = unsafe { z_block.rb().col(0).const_cast() };
                        let z1 = unsafe { z_block.rb().col(1).const_cast() };
                        if first_init {
                            zipped!(z.rb_mut().col(0), z0, z1).for_each(|mut z, mut z0, mut z1| {
                                z.write(z0.read().add(&z1.read()));
                                z0.write(E::zero());
                                z1.write(E::zero());
                            });
                        } else {
                            zipped!(z.rb_mut().col(0), z0, z1).for_each(|mut z, mut z0, mut z1| {
                                z.write(z.read().add(&z0.read().add(&z1.read())));
                                z0.write(E::zero());
                                z1.write(E::zero());
                            });
                        }
                    }
                    _ => unreachable!(),
                }
                idx += bs;
                first_init = false;
            }
        } else {
            let u_prev = a_left.rb().submatrix(k + 1, k - 1, m - 1, 1);
            let v_prev = a_top.rb().submatrix(k - 1, 1, 1, n - 1);
            matmul(
                a_next.rb_mut(),
                u_prev,
                y.rb().adjoint(),
                Some(E::one()),
                tl_prev.inv().neg(),
                parallelism,
            );
            matmul(
                a_next.rb_mut(),
                z.rb(),
                v_prev.conjugate(),
                Some(E::one()),
                tr.inv().neg(),
                parallelism,
            );

            matmul(
                y.rb_mut(),
                a_next.rb().adjoint(),
                u,
                None,
                E::one(),
                parallelism,
            );
            zipped!(y.rb_mut(), a_row.rb().transpose())
                .for_each(|mut dst, src| dst.write(dst.read().add(&src.read().conj())));
            let tl_inv = tl.inv();
            zipped!(a_row.rb_mut(), y.rb().transpose()).for_each(|mut dst, src| {
                dst.write(dst.read().sub(&src.read().conj().mul(&tl_inv)))
            });
            matmul(
                z.rb_mut(),
                a_next.rb(),
                a_row.rb().adjoint(),
                None,
                E::one(),
                parallelism,
            );
        }
    } else {
        matmul(
            y.rb_mut(),
            a_next.rb().adjoint(),
            u,
            None,
            E::one(),
            parallelism,
        );
        zipped!(y.rb_mut(), a_row.rb().transpose())
            .for_each(|mut dst, src| dst.write(dst.read().add(&src.read().conj())));
        let tl_inv = tl.inv();
        zipped!(a_row.rb_mut(), y.rb().transpose())
            .for_each(|mut dst, src| dst.write(dst.read().sub(&src.read().conj().mul(&tl_inv))));

        matmul(
            z.rb_mut(),
            a_next.rb(),
            a_row.rb().adjoint(),
            None,
            E::one(),
            parallelism,
        );
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
            apply_block_householder_sequence_on_the_right_in_place_with_conj,
            apply_block_householder_sequence_transpose_on_the_left_in_place_req,
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
            apply_block_householder_sequence_transpose_on_the_right_in_place_req,
        },
        Mat,
    };

    macro_rules! make_stack {
        ($req: expr $(,)?) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn bidiag_f64() {
        let mat = Mat::with_dims(15, 10, |_, _| rand::random::<f64>());

        let m = mat.nrows();
        let n = mat.ncols();

        let mut bid = mat.clone();
        let mut tau_left = Mat::zeros(n, 1);
        let mut tau_right = Mat::zeros(n - 1, 1);

        bidiagonalize_in_place(
            bid.as_mut(),
            tau_left.as_mut().col(0),
            tau_right.as_mut().col(0),
            Parallelism::None,
            make_stack!(bidiagonalize_in_place_req::<f64>(m, n, Parallelism::None)),
        );

        let mut copy = mat.clone();
        apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            bid.as_ref(),
            tau_left.as_ref().transpose(),
            Conj::No,
            copy.as_mut(),
            Parallelism::None,
            make_stack!(
                apply_block_householder_sequence_transpose_on_the_left_in_place_req::<f64>(m, 1, n),
            ),
        );

        apply_block_householder_sequence_on_the_right_in_place_with_conj(
            bid.as_ref().submatrix(0, 1, m, n - 1).transpose(),
            tau_right.as_ref().transpose(),
            Conj::No,
            copy.as_mut().submatrix(0, 1, m, n - 1),
            Parallelism::None,
            make_stack!(
                apply_block_householder_sequence_transpose_on_the_right_in_place_req::<f64>(
                    n - 1,
                    1,
                    m,
                )
            ),
        );

        for j in 0..n {
            for i in (0..j.saturating_sub(1)).chain(j + 1..m) {
                assert_approx_eq!(copy.read(i, j), 0.0);
            }
        }
    }

    #[test]
    fn bidiag_c64() {
        let mat = Mat::with_dims(15, 10, |_, _| c64::new(rand::random(), rand::random()));

        let m = mat.nrows();
        let n = mat.ncols();

        let mut bid = mat.clone();
        let mut tau_left = Mat::zeros(n, 1);
        let mut tau_right = Mat::zeros(n - 1, 1);

        bidiagonalize_in_place(
            bid.as_mut(),
            tau_left.as_mut().col(0),
            tau_right.as_mut().col(0),
            Parallelism::Rayon(0),
            make_stack!(bidiagonalize_in_place_req::<c64>(
                m,
                n,
                Parallelism::Rayon(0)
            )),
        );

        let mut copy = mat.clone();
        apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            bid.as_ref(),
            tau_left.as_ref().transpose(),
            Conj::Yes,
            copy.as_mut(),
            Parallelism::Rayon(0),
            make_stack!(
                apply_block_householder_sequence_transpose_on_the_left_in_place_req::<c64>(m, 1, n),
            ),
        );

        apply_block_householder_sequence_on_the_right_in_place_with_conj(
            bid.as_ref().submatrix(0, 1, m, n - 1).transpose(),
            tau_right.as_ref().transpose(),
            Conj::No,
            copy.as_mut().submatrix(0, 1, m, n - 1),
            Parallelism::Rayon(0),
            make_stack!(
                apply_block_householder_sequence_transpose_on_the_right_in_place_req::<c64>(
                    n - 1,
                    1,
                    m,
                )
            ),
        );

        for j in 0..n {
            for i in (0..j.saturating_sub(1)).chain(j + 1..m) {
                assert_approx_eq!(copy.read(i, j), c64::zero());
            }
        }
    }
}
