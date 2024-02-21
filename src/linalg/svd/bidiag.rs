use crate::{
    assert,
    linalg::{matmul::matmul, temp_mat_req, temp_mat_uninit, temp_mat_zeroed},
    unzipped,
    utils::thread::{for_each_raw, par_split_indices, parallelism_degree},
    zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism,
};
use core::slice;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_entity::*;
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
        temp_mat_req::<E>(m, parallelism_degree(parallelism))?,
    ])
}

pub fn bidiagonalize_in_place<E: ComplexField>(
    mut a: MatMut<'_, E>,
    mut householder_left: MatMut<'_, E>,
    mut householder_right: MatMut<'_, E>,
    parallelism: Parallelism,
    mut stack: PodStack<'_>,
) {
    let m = a.nrows();
    let n = a.ncols();

    assert!(m >= n);

    let n_threads = parallelism_degree(parallelism);

    let (mut y, mut stack) = temp_mat_uninit::<E>(n, 1, stack.rb_mut());
    let mut y = y.as_mut();
    let (mut z, mut stack) = temp_mat_uninit::<E>(m, 1, stack.rb_mut());
    let mut z = z.as_mut();

    let (mut z_tmp, _) = temp_mat_zeroed::<E>(m, n_threads, stack.rb_mut());
    let mut z_tmp = z_tmp.as_mut();

    let mut tl = E::faer_zero();
    let mut tr = E::faer_zero();
    let mut a01 = E::faer_zero();

    for k in 0..n {
        let (a_left, a_right) = a.rb_mut().split_at_col_mut(k);
        let (mut a_top, mut a_cur) = a_right.split_at_row_mut(k);

        let m = a_cur.nrows();
        let n = a_cur.ncols();

        let (mut a_col, a_right) = a_cur.rb_mut().split_at_col_mut(1);
        let (mut a_row, mut a_next) = a_right.split_at_row_mut(1);

        if k > 0 {
            let u = a_left.rb().submatrix(k, k - 1, m, 1);
            let mut v = a_top.rb_mut().submatrix_mut(k - 1, 0, 1, n);
            let y = y.rb().submatrix(k - 1, 0, n, 1);
            let z = z.rb().submatrix(k - 1, 0, m, 1);

            let f0 = y.read(0, 0).faer_conj().faer_mul(tl.faer_inv());
            let f1 = v.read(0, 0).faer_conj().faer_mul(tr.faer_inv());

            zipped!(a_col.rb_mut(), u, z).for_each(|unzipped!(mut a, b, c)| {
                a.write(
                    a.read()
                        .faer_sub(f0.faer_mul(b.read()))
                        .faer_sub(f1.faer_mul(c.read())),
                )
            });

            let f0 = u.read(0, 0).faer_mul(tl.faer_inv());
            let f1 = z.read(0, 0).faer_mul(tr.faer_inv());
            zipped!(
                a_row.rb_mut(),
                y.submatrix(1, 0, n - 1, 1).transpose(),
                v.rb().submatrix(0, 1, 1, n - 1),
            )
            .for_each(|unzipped!(mut a, b, c)| {
                a.write(
                    a.read()
                        .faer_sub(f0.faer_mul(b.read().faer_conj()))
                        .faer_sub(f1.faer_mul(c.read().faer_conj())),
                )
            });

            v.write(0, 0, a01);
        }

        let mut y = y.rb_mut().submatrix_mut(k, 0, n - 1, 1);
        let mut z = z.rb_mut().submatrix_mut(k, 0, m - 1, 1);
        let z_tmp = z_tmp.rb_mut().submatrix_mut(k, 0, m - 1, n_threads);

        let tl_prev = tl;
        let a00;
        (tl, a00) = {
            let head = a_col.read(0, 0);
            let essential = a_col.rb_mut().col_mut(0).subrows_mut(1, m - 1);
            let tail_norm = essential.norm_l2();
            crate::linalg::householder::make_householder_in_place(
                Some(essential.as_2d_mut()),
                head,
                tail_norm,
            )
        };
        a_col.write(0, 0, a00);
        householder_left.write(k, 0, tl);

        if n == 1 {
            break;
        }

        let u = a_col.rb().submatrix(1, 0, m - 1, 1);

        bidiag_fused_op(
            k,
            m,
            n,
            tl_prev,
            tl,
            tr,
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
            let tail_norm = essential.norm_l2();
            crate::linalg::householder::make_householder_in_place(None, head, tail_norm)
        };
        householder_right.write(k, 0, tr);

        let diff = a_row.read(0, 0).faer_sub(a01);

        if diff != E::faer_zero() {
            let f = diff.faer_inv().faer_conj();
            zipped!(a_row
                .rb_mut()
                .row_mut(0)
                .subcols_mut(1, n - 2)
                .transpose_mut()
                .as_2d_mut())
            .for_each(|unzipped!(mut x)| x.write(x.read().faer_conj().faer_mul(f)));

            zipped!(
                z.rb_mut().col_mut(0).as_2d_mut(),
                a_next.rb().col(0).as_2d(),
            )
            .for_each(|unzipped!(mut z, a)| {
                z.write(f.faer_mul(z.read().faer_sub(a01.faer_conj().faer_mul(a.read()))))
            });
        }

        a_row.write(0, 0, E::faer_one());
        let b = crate::linalg::matmul::inner_prod::inner_prod_with_conj(
            y.rb().col(0).as_2d(),
            Conj::Yes,
            a_row.rb().row(0).transpose().as_2d(),
            Conj::No,
        );

        let factor = b.faer_mul(tl.faer_inv()).faer_neg();
        zipped!(z.rb_mut(), u)
            .for_each(|unzipped!(mut z, u)| z.write(z.read().faer_add(u.read().faer_mul(factor))));
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
            Some(E::faer_one()),
            tl_prev.faer_inv().faer_neg(),
            parallelism,
        );
        matmul(
            a_next.rb_mut(),
            z.rb(),
            v_prev.conjugate(),
            Some(E::faer_one()),
            tr.faer_inv().faer_neg(),
            parallelism,
        );

        matmul(
            y.rb_mut(),
            a_next.rb().adjoint(),
            u,
            None,
            E::faer_one(),
            parallelism,
        );
        zipped!(y.rb_mut(), a_row.rb().transpose()).for_each(|unzipped!(mut dst, src)| {
            dst.write(dst.read().faer_add(src.read().faer_conj()))
        });
        let tl_inv = tl.faer_inv();
        zipped!(a_row.rb_mut(), y.rb().transpose()).for_each(|unzipped!(mut dst, src)| {
            dst.write(dst.read().faer_sub(src.read().faer_conj().faer_mul(tl_inv)))
        });
        matmul(
            z.rb_mut(),
            a_next.rb(),
            a_row.rb().adjoint(),
            None,
            E::faer_one(),
            parallelism,
        );
    } else {
        matmul(
            y.rb_mut(),
            a_next.rb().adjoint(),
            u,
            None,
            E::faer_one(),
            parallelism,
        );
        zipped!(y.rb_mut(), a_row.rb().transpose()).for_each(|unzipped!(mut dst, src)| {
            dst.write(dst.read().faer_add(src.read().faer_conj()))
        });
        let tl_inv = tl.faer_inv();
        zipped!(a_row.rb_mut(), y.rb().transpose()).for_each(|unzipped!(mut dst, src)| {
            dst.write(dst.read().faer_sub(src.read().faer_conj().faer_mul(tl_inv)))
        });
        matmul(
            z.rb_mut(),
            a_next.rb(),
            a_row.rb().adjoint(),
            None,
            E::faer_one(),
            parallelism,
        );
    }
}

fn bidiag_fused_op_step0<E: ComplexField>(
    arch: E::Simd,

    // update a_next
    mut a_j: GroupFor<E, &mut [UnitFor<E>]>,
    z: GroupFor<E, &[UnitFor<E>]>,
    u_prev: GroupFor<E, &[UnitFor<E>]>,
    u_rhs: E,
    z_rhs: E,

    // compute yj
    u: GroupFor<E, &[UnitFor<E>]>,
) -> E {
    struct Impl<'a, E: ComplexField> {
        a_j: GroupFor<E, &'a mut [UnitFor<E>]>,
        z: GroupFor<E, &'a [UnitFor<E>]>,
        u_prev: GroupFor<E, &'a [UnitFor<E>]>,
        u_rhs: E,
        z_rhs: E,
        u: GroupFor<E, &'a [UnitFor<E>]>,
    }

    impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
        type Output = E;

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

            let (a_j_head, a_j_tail) = faer_entity::slice_as_mut_simd::<E, S>(a_j);
            let (z_head, z_tail) = faer_entity::slice_as_simd::<E, S>(z);
            let (u_prev_head, u_prev_tail) = faer_entity::slice_as_simd::<E, S>(u_prev);
            let (u_head, u_tail) = faer_entity::slice_as_simd::<E, S>(u);

            let (a_j_head4, a_j_head1) = E::faer_as_arrays_mut::<4, _>(a_j_head);
            let (z_head4, z_head1) = E::faer_as_arrays::<4, _>(z_head);
            let (u_prev_head4, u_prev_head1) = E::faer_as_arrays::<4, _>(u_prev_head);
            let (u_head4, u_head1) = E::faer_as_arrays::<4, _>(u_head);

            let mut sum_v0 = E::faer_simd_splat(simd, E::faer_zero());
            let mut sum_v1 = E::faer_simd_splat(simd, E::faer_zero());
            let mut sum_v2 = E::faer_simd_splat(simd, E::faer_zero());
            let mut sum_v3 = E::faer_simd_splat(simd, E::faer_zero());

            let u_rhs_v = E::faer_simd_neg(simd, E::faer_simd_splat(simd, u_rhs));
            let z_rhs_v = E::faer_simd_neg(simd, E::faer_simd_splat(simd, z_rhs));

            for (((aij, zi), u_prev_i), ui) in E::faer_into_iter(a_j_head4)
                .zip(E::faer_into_iter(z_head4))
                .zip(E::faer_into_iter(u_prev_head4))
                .zip(E::faer_into_iter(u_head4))
            {
                let [u_prev_i0, u_prev_i1, u_prev_i2, u_prev_i3] =
                    E::faer_unzip4(E::faer_deref(u_prev_i));
                let [zi0, zi1, zi2, zi3] = E::faer_unzip4(E::faer_deref(zi));
                let [ui0, ui1, ui2, ui3] = E::faer_unzip4(E::faer_deref(ui));
                let [aij0, aij1, aij2, aij3] =
                    E::faer_unzip4(E::faer_deref(E::faer_rb(E::faer_as_ref(&aij))));

                let aij_new0 = E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(u_prev_i0),
                    u_rhs_v,
                    E::faer_simd_mul_adde(
                        simd,
                        into_copy::<E, _>(zi0),
                        z_rhs_v,
                        into_copy::<E, _>(aij0),
                    ),
                );
                let aij_new1 = E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(u_prev_i1),
                    u_rhs_v,
                    E::faer_simd_mul_adde(
                        simd,
                        into_copy::<E, _>(zi1),
                        z_rhs_v,
                        into_copy::<E, _>(aij1),
                    ),
                );
                let aij_new2 = E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(u_prev_i2),
                    u_rhs_v,
                    E::faer_simd_mul_adde(
                        simd,
                        into_copy::<E, _>(zi2),
                        z_rhs_v,
                        into_copy::<E, _>(aij2),
                    ),
                );
                let aij_new3 = E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(u_prev_i3),
                    u_rhs_v,
                    E::faer_simd_mul_adde(
                        simd,
                        into_copy::<E, _>(zi3),
                        z_rhs_v,
                        into_copy::<E, _>(aij3),
                    ),
                );
                sum_v0 = E::faer_simd_conj_mul_adde(simd, aij_new0, into_copy::<E, _>(ui0), sum_v0);
                sum_v1 = E::faer_simd_conj_mul_adde(simd, aij_new1, into_copy::<E, _>(ui1), sum_v1);
                sum_v2 = E::faer_simd_conj_mul_adde(simd, aij_new2, into_copy::<E, _>(ui2), sum_v2);
                sum_v3 = E::faer_simd_conj_mul_adde(simd, aij_new3, into_copy::<E, _>(ui3), sum_v3);

                E::faer_map(
                    E::faer_zip(
                        aij,
                        E::faer_zip(
                            E::faer_zip(from_copy::<E, _>(aij_new0), from_copy::<E, _>(aij_new1)),
                            E::faer_zip(from_copy::<E, _>(aij_new2), from_copy::<E, _>(aij_new3)),
                        ),
                    ),
                    #[inline(always)]
                    |(aij, ((aij0, aij1), (aij2, aij3)))| {
                        aij[0] = aij0;
                        aij[1] = aij1;
                        aij[2] = aij2;
                        aij[3] = aij3;
                    },
                );
            }

            sum_v0 = E::faer_simd_add(simd, sum_v0, sum_v1);
            sum_v2 = E::faer_simd_add(simd, sum_v2, sum_v3);
            sum_v0 = E::faer_simd_add(simd, sum_v0, sum_v2);

            for (((aij, zi), u_prev_i), ui) in E::faer_into_iter(a_j_head1)
                .zip(E::faer_into_iter(z_head1))
                .zip(E::faer_into_iter(u_prev_head1))
                .zip(E::faer_into_iter(u_head1))
            {
                let aij_new = E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_deref(u_prev_i)),
                    u_rhs_v,
                    E::faer_simd_mul_adde(
                        simd,
                        into_copy::<E, _>(E::faer_deref(zi)),
                        z_rhs_v,
                        into_copy::<E, _>(E::faer_deref(E::faer_rb(E::faer_as_ref(&aij)))),
                    ),
                );
                sum_v0 = E::faer_simd_conj_mul_adde(
                    simd,
                    aij_new,
                    into_copy::<E, _>(E::faer_deref(ui)),
                    sum_v0,
                );
                E::faer_map(
                    E::faer_zip(aij, from_copy::<E, _>(aij_new)),
                    #[inline(always)]
                    |(aij, aij_new)| *aij = aij_new,
                );
            }

            let zi = E::faer_partial_load(simd, z_tail);
            let u_prev_i = E::faer_partial_load(simd, u_prev_tail);
            let ui = E::faer_partial_load(simd, u_tail);
            let aij_load = E::faer_partial_load(simd, E::faer_rb(E::faer_as_ref(&a_j_tail)));

            let aij_new = E::faer_simd_mul_adde(
                simd,
                u_prev_i,
                u_rhs_v,
                E::faer_simd_mul_adde(simd, zi, z_rhs_v, aij_load),
            );

            sum_v0 = E::faer_simd_conj_mul_adde(simd, aij_new, ui, sum_v0);
            E::faer_partial_store(simd, a_j_tail, aij_new);

            E::faer_simd_reduce_add(simd, sum_v0)
        }
    }

    arch.dispatch(Impl {
        a_j: E::faer_rb_mut(E::faer_as_mut(&mut a_j)),
        z: E::faer_rb(E::faer_as_ref(&z)),
        u_prev: E::faer_rb(E::faer_as_ref(&u_prev)),
        u_rhs,
        z_rhs,
        u: E::faer_rb(E::faer_as_ref(&u)),
    })
}

fn bidiag_fused_op_step1<'a, E: ComplexField>(
    arch: E::Simd,

    // update z
    z: GroupFor<E, &'a mut [UnitFor<E>]>,
    a_j: GroupFor<E, &'a [UnitFor<E>]>,
    rhs: E,
) {
    struct Impl<'a, E: ComplexField> {
        z: GroupFor<E, &'a mut [UnitFor<E>]>,
        a_j: GroupFor<E, &'a [UnitFor<E>]>,
        rhs: E,
    }
    impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self { z, a_j, rhs } = self;
            let (z_head, z_tail) = faer_entity::slice_as_mut_simd::<E, S>(z);
            let (a_j_head, a_j_tail) = faer_entity::slice_as_simd::<E, S>(a_j);
            let rhs_v = E::faer_simd_splat(simd, rhs);

            for (zi, aij) in E::faer_into_iter(z_head).zip(E::faer_into_iter(a_j_head)) {
                let new_zi = E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_deref(aij)),
                    rhs_v,
                    into_copy::<E, _>(E::faer_deref(E::faer_rb(E::faer_as_ref(&zi)))),
                );
                E::faer_map(
                    E::faer_zip(zi, from_copy::<E, _>(new_zi)),
                    #[inline(always)]
                    |(zi, new_zi)| *zi = new_zi,
                );
            }

            let aij = E::faer_partial_load(simd, a_j_tail);
            let zi_load = E::faer_partial_load(simd, E::faer_rb(E::faer_as_ref(&z_tail)));
            let new_zi = E::faer_simd_mul_adde(simd, aij, rhs_v, zi_load);
            E::faer_partial_store(simd, z_tail, new_zi);
        }
    }
    arch.dispatch(Impl { z, a_j, rhs })
}

fn bidiag_fused_op_process_batch<E: ComplexField>(
    arch: E::Simd,
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
        let u_rhs = y.read(j, 0).faer_conj().faer_mul(tl_prev_inv);
        let z_rhs = v_prev.read(0, j).faer_conj().faer_mul(tr_prev_inv);

        let yj = {
            let a_next = a_next.rb_mut();
            unsafe {
                bidiag_fused_op_step0::<E>(
                    arch,
                    E::faer_map(
                        a_next.ptr_at_mut(0, j),
                        #[inline(always)]
                        |ptr| slice::from_raw_parts_mut(ptr, nrows),
                    ),
                    E::faer_map(
                        z.as_ptr(),
                        #[inline(always)]
                        |ptr| slice::from_raw_parts(ptr, nrows),
                    ),
                    E::faer_map(
                        u_prev.as_ptr(),
                        #[inline(always)]
                        |ptr| slice::from_raw_parts(ptr, nrows),
                    ),
                    u_rhs,
                    z_rhs,
                    E::faer_map(
                        u.as_ptr(),
                        #[inline(always)]
                        |ptr| slice::from_raw_parts(ptr, nrows),
                    ),
                )
            }
        };
        y.write(j, 0, yj.faer_add(a_row.read(0, j).faer_conj()));
        a_row.write(
            0,
            j,
            a_row
                .read(0, j)
                .faer_sub(y.read(j, 0).faer_conj().faer_mul(tl_inv)),
        );

        let rhs = a_row.read(0, j).faer_conj();

        let a_next = a_next.rb();
        let z_tmp = z_tmp.rb_mut();
        unsafe {
            bidiag_fused_op_step1(
                arch,
                E::faer_map(z_tmp.as_ptr_mut(), |ptr| {
                    slice::from_raw_parts_mut(ptr, nrows)
                }),
                E::faer_map(a_next.ptr_at(0, j), |ptr| slice::from_raw_parts(ptr, nrows)),
                rhs,
            );
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
            let arch = E::Simd::default();

            let n_threads = parallelism_degree(parallelism);

            let u_prev = a_left.rb().submatrix(k + 1, k - 1, m - 1, 1).col(0);
            let v_prev = a_top.rb().submatrix(k - 1, 1, 1, n - 1).row(0);

            let tl_prev_inv = tl_prev.faer_inv();
            let tr_prev_inv = tr.faer_inv();
            let tl_inv = tl.faer_inv();

            assert!(a_next.row_stride() == 1);
            assert!(u_prev.row_stride() == 1);
            assert!(u.row_stride() == 1);
            assert!(z.row_stride() == 1);

            match n_threads {
                1 => {
                    bidiag_fused_op_process_batch(
                        arch,
                        z_tmp.rb_mut().col_mut(0).as_2d_mut(),
                        a_next,
                        a_row.row_mut(0).as_2d_mut(),
                        u.col(0).as_2d(),
                        u_prev.as_2d(),
                        v_prev.as_2d(),
                        y.col_mut(0).as_2d_mut(),
                        z.rb().col(0).as_2d(),
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
                                z_tmp.as_2d_mut(),
                                a_next,
                                a_row,
                                u,
                                u_prev.as_2d(),
                                v_prev.as_2d(),
                                y,
                                z.rb(),
                                tl_prev_inv,
                                tr_prev_inv,
                                tl_inv,
                            );
                        },
                        parallelism,
                    );
                }
            }

            let mut idx = 0;
            let mut first_init = true;
            while idx < n_threads {
                let bs = Ord::min(2, n_threads - idx);
                let z_block = z_tmp.rb_mut().submatrix_mut(0, idx, m - 1, bs);

                match bs {
                    1 => {
                        let z0 = unsafe { z_block.rb().col(0).const_cast() };
                        if first_init {
                            zipped!(z.rb_mut().col_mut(0).as_2d_mut(), z0.as_2d_mut()).for_each(
                                |unzipped!(mut z, mut z0)| {
                                    z.write(z0.read());
                                    z0.write(E::faer_zero());
                                },
                            );
                        } else {
                            zipped!(z.rb_mut().col_mut(0).as_2d_mut(), z0.as_2d_mut()).for_each(
                                |unzipped!(mut z, mut z0)| {
                                    z.write(z.read().faer_add(z0.read()));
                                    z0.write(E::faer_zero());
                                },
                            );
                        }
                    }
                    2 => {
                        let z0 = unsafe { z_block.rb().col(0).const_cast() };
                        let z1 = unsafe { z_block.rb().col(1).const_cast() };
                        if first_init {
                            zipped!(
                                z.rb_mut().col_mut(0).as_2d_mut(),
                                z0.as_2d_mut(),
                                z1.as_2d_mut(),
                            )
                            .for_each(
                                |unzipped!(mut z, mut z0, mut z1)| {
                                    z.write(z0.read().faer_add(z1.read()));
                                    z0.write(E::faer_zero());
                                    z1.write(E::faer_zero());
                                },
                            );
                        } else {
                            zipped!(
                                z.rb_mut().col_mut(0).as_2d_mut(),
                                z0.as_2d_mut(),
                                z1.as_2d_mut(),
                            )
                            .for_each(
                                |unzipped!(mut z, mut z0, mut z1)| {
                                    z.write(z.read().faer_add(z0.read().faer_add(z1.read())));
                                    z0.write(E::faer_zero());
                                    z1.write(E::faer_zero());
                                },
                            );
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
                Some(E::faer_one()),
                tl_prev.faer_inv().faer_neg(),
                parallelism,
            );
            matmul(
                a_next.rb_mut(),
                z.rb(),
                v_prev.conjugate(),
                Some(E::faer_one()),
                tr.faer_inv().faer_neg(),
                parallelism,
            );

            matmul(
                y.rb_mut(),
                a_next.rb().adjoint(),
                u,
                None,
                E::faer_one(),
                parallelism,
            );
            zipped!(y.rb_mut(), a_row.rb().transpose()).for_each(|unzipped!(mut dst, src)| {
                dst.write(dst.read().faer_add(src.read().faer_conj()))
            });
            let tl_inv = tl.faer_inv();
            zipped!(a_row.rb_mut(), y.rb().transpose()).for_each(|unzipped!(mut dst, src)| {
                dst.write(dst.read().faer_sub(src.read().faer_conj().faer_mul(tl_inv)))
            });
            matmul(
                z.rb_mut(),
                a_next.rb(),
                a_row.rb().adjoint(),
                None,
                E::faer_one(),
                parallelism,
            );
        }
    } else {
        matmul(
            y.rb_mut(),
            a_next.rb().adjoint(),
            u,
            None,
            E::faer_one(),
            parallelism,
        );
        zipped!(y.rb_mut(), a_row.rb().transpose()).for_each(|unzipped!(mut dst, src)| {
            dst.write(dst.read().faer_add(src.read().faer_conj()))
        });
        let tl_inv = tl.faer_inv();
        zipped!(a_row.rb_mut(), y.rb().transpose()).for_each(|unzipped!(mut dst, src)| {
            dst.write(dst.read().faer_sub(src.read().faer_conj().faer_mul(tl_inv)))
        });

        matmul(
            z.rb_mut(),
            a_next.rb(),
            a_row.rb().adjoint(),
            None,
            E::faer_one(),
            parallelism,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert,
        complex_native::c64,
        linalg::householder::{
            apply_block_householder_sequence_on_the_right_in_place_with_conj,
            apply_block_householder_sequence_transpose_on_the_left_in_place_req,
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
            apply_block_householder_sequence_transpose_on_the_right_in_place_req,
        },
        Mat,
    };
    use assert_approx_eq::assert_approx_eq;

    macro_rules! make_stack {
        ($req: expr $(,)?) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn bidiag_f64() {
        let mat = Mat::from_fn(15, 10, |_, _| rand::random::<f64>());

        let m = mat.nrows();
        let n = mat.ncols();

        let mut bid = mat.clone();
        let mut tau_left = Mat::zeros(n, 1);
        let mut tau_right = Mat::zeros(n - 1, 1);

        bidiagonalize_in_place(
            bid.as_mut(),
            tau_left.as_mut().col_mut(0).as_2d_mut(),
            tau_right.as_mut().col_mut(0).as_2d_mut(),
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
            copy.as_mut().submatrix_mut(0, 1, m, n - 1),
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
        let mat = Mat::from_fn(15, 10, |_, _| c64::new(rand::random(), rand::random()));

        let m = mat.nrows();
        let n = mat.ncols();

        let mut bid = mat.clone();
        let mut tau_left = Mat::zeros(n, 1);
        let mut tau_right = Mat::zeros(n - 1, 1);

        bidiagonalize_in_place(
            bid.as_mut(),
            tau_left.as_mut().col_mut(0).as_2d_mut(),
            tau_right.as_mut().col_mut(0).as_2d_mut(),
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
            copy.as_mut().submatrix_mut(0, 1, m, n - 1),
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
                assert_approx_eq!(copy.read(i, j), c64::faer_zero());
            }
        }
    }
}
