#[cfg(feature = "std")]
use assert2::{assert, debug_assert};
use core::slice;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    c32, c64,
    householder::upgrade_householder_factor,
    mul::inner_prod::{self, inner_prod_with_conj_arch},
    permutation::{swap_cols, PermutationMut},
    simd, transmute_unchecked, zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism,
    SimdCtx,
};
use pulp::{as_arrays, as_arrays_mut, Simd};
use reborrow::*;

pub use crate::no_pivoting::compute::recommended_blocksize;

#[inline(always)]
fn update_and_norm2_simd_impl<'a, E: ComplexField, S: Simd>(
    simd: S,
    a: E::Group<&'a mut [E::Unit]>,
    b: E::Group<&'a [E::Unit]>,
    k: E,
) -> E::Real {
    let mut acc0 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());
    let mut acc1 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());
    let mut acc2 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());
    let mut acc3 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());
    let mut acc4 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());
    let mut acc5 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());
    let mut acc6 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());
    let mut acc7 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());

    let (a, a_rem) = simd::slice_as_mut_simd::<E, S>(a);
    let (b, b_rem) = simd::slice_as_simd::<E, S>(b);

    let k_ = k;
    let k = E::faer_simd_splat(simd, k_);

    let (a, a_remv) = E::faer_as_arrays_mut::<8, _>(a);
    let (b, b_remv) = E::faer_as_arrays::<8, _>(b);

    for (a, b) in E::faer_into_iter(a).zip(E::faer_into_iter(b)) {
        let [mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7] =
            E::faer_unzip8(E::faer_deref(E::faer_rb(E::faer_as_ref(&a))));
        let [b0, b1, b2, b3, b4, b5, b6, b7] = E::faer_unzip8(E::faer_deref(b));

        a0 = E::faer_simd_mul_adde(simd, E::faer_copy(&k), b0, a0);
        acc0 = E::faer_simd_abs2_adde(simd, E::faer_copy(&a0), acc0);

        a1 = E::faer_simd_mul_adde(simd, E::faer_copy(&k), b1, a1);
        acc1 = E::faer_simd_abs2_adde(simd, E::faer_copy(&a1), acc1);

        a2 = E::faer_simd_mul_adde(simd, E::faer_copy(&k), b2, a2);
        acc2 = E::faer_simd_abs2_adde(simd, E::faer_copy(&a2), acc2);

        a3 = E::faer_simd_mul_adde(simd, E::faer_copy(&k), b3, a3);
        acc3 = E::faer_simd_abs2_adde(simd, E::faer_copy(&a3), acc3);

        a4 = E::faer_simd_mul_adde(simd, E::faer_copy(&k), b4, a4);
        acc4 = E::faer_simd_abs2_adde(simd, E::faer_copy(&a4), acc4);

        a5 = E::faer_simd_mul_adde(simd, E::faer_copy(&k), b5, a5);
        acc5 = E::faer_simd_abs2_adde(simd, E::faer_copy(&a5), acc5);

        a6 = E::faer_simd_mul_adde(simd, E::faer_copy(&k), b6, a6);
        acc6 = E::faer_simd_abs2_adde(simd, E::faer_copy(&a6), acc6);

        a7 = E::faer_simd_mul_adde(simd, E::faer_copy(&k), b7, a7);
        acc7 = E::faer_simd_abs2_adde(simd, E::faer_copy(&a7), acc7);

        E::faer_map(
            E::faer_zip(
                a,
                E::faer_zip(
                    E::faer_zip(E::faer_zip(a0, a1), E::faer_zip(a2, a3)),
                    E::faer_zip(E::faer_zip(a4, a5), E::faer_zip(a6, a7)),
                ),
            ),
            #[inline(always)]
            |(a, (((a0, a1), (a2, a3)), ((a4, a5), (a6, a7))))| {
                a[0] = a0;
                a[1] = a1;
                a[2] = a2;
                a[3] = a3;
                a[4] = a4;
                a[5] = a5;
                a[6] = a6;
                a[7] = a7;
            },
        );
    }

    acc0 = E::Real::faer_simd_add(simd, acc0, acc1);
    acc2 = E::Real::faer_simd_add(simd, acc2, acc3);
    acc4 = E::Real::faer_simd_add(simd, acc4, acc5);
    acc6 = E::Real::faer_simd_add(simd, acc6, acc7);

    acc0 = E::Real::faer_simd_add(simd, acc0, acc2);
    acc4 = E::Real::faer_simd_add(simd, acc4, acc6);

    for (a, b) in E::faer_into_iter(a_remv).zip(E::faer_into_iter(b_remv)) {
        let new_a = E::faer_simd_mul_adde(
            simd,
            E::faer_copy(&k),
            E::faer_deref(b),
            E::faer_deref(E::faer_rb(E::faer_as_ref(&a))),
        );
        E::faer_map(
            E::faer_zip(a, E::faer_copy(&new_a)),
            #[inline(always)]
            |(a, new_a)| *a = new_a,
        );
        acc0 = E::faer_simd_abs2_adde(simd, new_a, acc0);
    }

    acc0 = E::Real::faer_simd_add(simd, acc0, acc4);

    let a_load = E::faer_partial_load(simd, E::faer_rb(E::faer_as_ref(&a_rem)));
    let b = E::faer_partial_load(simd, b_rem);
    let new_a = E::faer_simd_mul_adde(simd, E::faer_copy(&k), b, a_load);
    E::faer_partial_store(simd, a_rem, E::faer_copy(&new_a));
    acc0 = E::faer_simd_abs2_adde(simd, new_a, acc0);

    E::Real::faer_simd_reduce_add(simd, acc0)
}

#[inline(always)]
fn update_and_norm2_simd_impl_c32<'a, S: Simd>(
    simd: S,
    a: &'a mut [c32],
    b: &'a [c32],
    k: c32,
) -> f32 {
    let k = k.into();
    let mut acc0 = simd.f32s_splat(0.0);
    let mut acc1 = simd.f32s_splat(0.0);
    let mut acc2 = simd.f32s_splat(0.0);
    let mut acc3 = simd.f32s_splat(0.0);
    let mut acc4 = simd.f32s_splat(0.0);
    let mut acc5 = simd.f32s_splat(0.0);
    let mut acc6 = simd.f32s_splat(0.0);
    let mut acc7 = simd.f32s_splat(0.0);

    let (a, a_rem) = S::c32s_as_mut_simd(bytemuck::cast_slice_mut(a));
    let (b, b_rem) = S::c32s_as_simd(bytemuck::cast_slice(b));

    let (a, a_remv) = as_arrays_mut::<8, _>(a);
    let (b, b_remv) = as_arrays::<8, _>(b);

    let vk = simd.c32s_splat(k);

    #[inline(always)]
    fn accumulate<S: Simd>(simd: S, acc: S::f32s, a: S::c32s) -> S::f32s {
        if coe::is_same::<S, pulp::Scalar>() {
            let norm2: c32 = bytemuck::cast(simd.c32s_abs2(a));
            bytemuck::cast(norm2.re)
        } else {
            simd.f32s_mul_add_e(bytemuck::cast(a), bytemuck::cast(a), acc)
        }
    }

    for (a, b) in a.iter_mut().zip(b.iter()) {
        a[0] = simd.c32s_mul_add_e(vk, b[0], a[0]);
        acc0 = accumulate(simd, acc0, a[0]);

        a[1] = simd.c32s_mul_add_e(vk, b[1], a[1]);
        acc1 = accumulate(simd, acc1, a[1]);

        a[2] = simd.c32s_mul_add_e(vk, b[2], a[2]);
        acc2 = accumulate(simd, acc2, a[2]);

        a[3] = simd.c32s_mul_add_e(vk, b[3], a[3]);
        acc3 = accumulate(simd, acc3, a[3]);

        a[4] = simd.c32s_mul_add_e(vk, b[4], a[4]);
        acc4 = accumulate(simd, acc4, a[4]);

        a[5] = simd.c32s_mul_add_e(vk, b[5], a[5]);
        acc5 = accumulate(simd, acc5, a[5]);

        a[6] = simd.c32s_mul_add_e(vk, b[6], a[6]);
        acc6 = accumulate(simd, acc6, a[6]);

        a[7] = simd.c32s_mul_add_e(vk, b[7], a[7]);
        acc7 = accumulate(simd, acc7, a[7]);
    }

    for (a, b) in a_remv.iter_mut().zip(b_remv.iter()) {
        *a = simd.c32s_mul_add_e(vk, *b, *a);
        acc0 = accumulate(simd, acc0, *a);
    }

    acc0 = simd.f32s_add(acc0, acc1);
    acc2 = simd.f32s_add(acc2, acc3);
    acc4 = simd.f32s_add(acc4, acc5);
    acc6 = simd.f32s_add(acc6, acc7);

    acc0 = simd.f32s_add(acc0, acc2);
    acc4 = simd.f32s_add(acc4, acc6);

    acc0 = simd.f32s_add(acc0, acc4);

    let mut acc = simd.f32s_reduce_sum(acc0);

    for (a, b) in a_rem.iter_mut().zip(b_rem.iter()) {
        *a = k * *b + *a;
        acc = a.re * a.re + a.im * a.im;
    }

    acc
}

#[inline(always)]
fn update_and_norm2_simd_impl_c64<'a, S: Simd>(
    simd: S,
    a: &'a mut [c64],
    b: &'a [c64],
    k: c64,
) -> f64 {
    let k = k.into();
    let mut acc0 = simd.f64s_splat(0.0);
    let mut acc1 = simd.f64s_splat(0.0);
    let mut acc2 = simd.f64s_splat(0.0);
    let mut acc3 = simd.f64s_splat(0.0);
    let mut acc4 = simd.f64s_splat(0.0);
    let mut acc5 = simd.f64s_splat(0.0);
    let mut acc6 = simd.f64s_splat(0.0);
    let mut acc7 = simd.f64s_splat(0.0);

    let (a, a_rem) = S::c64s_as_mut_simd(bytemuck::cast_slice_mut(a));
    let (b, b_rem) = S::c64s_as_simd(bytemuck::cast_slice(b));

    let (a, a_remv) = as_arrays_mut::<8, _>(a);
    let (b, b_remv) = as_arrays::<8, _>(b);

    let vk = simd.c64s_splat(k);

    #[inline(always)]
    fn accumulate<S: Simd>(simd: S, acc: S::f64s, a: S::c64s) -> S::f64s {
        if coe::is_same::<S, pulp::Scalar>() {
            let norm2: c64 = bytemuck::cast(simd.c64s_abs2(a));
            bytemuck::cast(norm2.re)
        } else {
            simd.f64s_mul_add_e(bytemuck::cast(a), bytemuck::cast(a), acc)
        }
    }

    for (a, b) in a.iter_mut().zip(b.iter()) {
        a[0] = simd.c64s_mul_add_e(vk, b[0], a[0]);
        acc0 = accumulate(simd, acc0, a[0]);

        a[1] = simd.c64s_mul_add_e(vk, b[1], a[1]);
        acc1 = accumulate(simd, acc1, a[1]);

        a[2] = simd.c64s_mul_add_e(vk, b[2], a[2]);
        acc2 = accumulate(simd, acc2, a[2]);

        a[3] = simd.c64s_mul_add_e(vk, b[3], a[3]);
        acc3 = accumulate(simd, acc3, a[3]);

        a[4] = simd.c64s_mul_add_e(vk, b[4], a[4]);
        acc4 = accumulate(simd, acc4, a[4]);

        a[5] = simd.c64s_mul_add_e(vk, b[5], a[5]);
        acc5 = accumulate(simd, acc5, a[5]);

        a[6] = simd.c64s_mul_add_e(vk, b[6], a[6]);
        acc6 = accumulate(simd, acc6, a[6]);

        a[7] = simd.c64s_mul_add_e(vk, b[7], a[7]);
        acc7 = accumulate(simd, acc7, a[7]);
    }

    for (a, b) in a_remv.iter_mut().zip(b_remv.iter()) {
        *a = simd.c64s_mul_add_e(vk, *b, *a);
        acc0 = accumulate(simd, acc0, *a);
    }

    acc0 = simd.f64s_add(acc0, acc1);
    acc2 = simd.f64s_add(acc2, acc3);
    acc4 = simd.f64s_add(acc4, acc5);
    acc6 = simd.f64s_add(acc6, acc7);

    acc0 = simd.f64s_add(acc0, acc2);
    acc4 = simd.f64s_add(acc4, acc6);

    acc0 = simd.f64s_add(acc0, acc4);

    let mut acc = simd.f64s_reduce_sum(acc0);

    for (a, b) in a_rem.iter_mut().zip(b_rem.iter()) {
        *a = k * *b + *a;
        acc = a.re * a.re + a.im * a.im;
    }

    acc
}

struct UpdateAndNorm2<'a, E: ComplexField> {
    a: E::Group<&'a mut [E::Unit]>,
    b: E::Group<&'a [E::Unit]>,
    k: E,
}

impl<E: ComplexField> pulp::WithSimd for UpdateAndNorm2<'_, E> {
    type Output = E::Real;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { a, b, k } = self;
        if coe::is_same::<c32, E>() {
            return coe::coerce_static(unsafe {
                update_and_norm2_simd_impl_c32(
                    simd,
                    transmute_unchecked(a),
                    transmute_unchecked(b),
                    transmute_unchecked(k),
                )
            });
        }
        if coe::is_same::<c64, E>() {
            return coe::coerce_static(unsafe {
                update_and_norm2_simd_impl_c64(
                    simd,
                    transmute_unchecked(a),
                    transmute_unchecked(b),
                    transmute_unchecked(k),
                )
            });
        }
        update_and_norm2_simd_impl(simd, a, b, k)
    }
}

#[inline(always)]
fn norm2<E: ComplexField>(arch: E::Simd, a: MatRef<'_, E>) -> E::Real {
    inner_prod_with_conj_arch(arch, a, Conj::Yes, a, Conj::No).faer_real()
}

#[inline(always)]
fn update_and_norm2<E: ComplexField>(
    arch: E::Simd,
    a: MatMut<'_, E>,
    b: MatRef<'_, E>,
    k: E,
) -> E::Real {
    let colmajor = a.row_stride() == 1 && b.row_stride() == 1;
    if colmajor {
        let a_len = a.nrows();

        let a = E::faer_map(
            a.as_ptr(),
            #[inline(always)]
            |ptr| unsafe { slice::from_raw_parts_mut(ptr, a_len) },
        );
        let b = E::faer_map(
            b.as_ptr(),
            #[inline(always)]
            |ptr| unsafe { slice::from_raw_parts(ptr, a_len) },
        );
        return arch.dispatch(UpdateAndNorm2 { a, b, k });
    }

    let mut acc = E::Real::faer_zero();
    zipped!(a, b).for_each(|mut a_, b| {
        let a = a_.read();
        let b = b.read();

        a_.write(a.faer_add(k.faer_mul(b)));
        acc = acc.faer_add((a.faer_conj().faer_mul(a)).faer_real());
    });

    acc
}

fn qr_in_place_colmajor<E: ComplexField>(
    mut matrix: MatMut<'_, E>,
    mut householder_coeffs: MatMut<'_, E>,
    col_perm: &mut [usize],
    parallelism: Parallelism,
    disable_parallelism: fn(usize, usize) -> bool,
) -> usize {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = Ord::min(m, n);

    debug_assert!(householder_coeffs.nrows() == size);

    let mut n_transpositions = 0;

    if size == 0 {
        return n_transpositions;
    }

    let mut biggest_col_idx = 0;
    let mut biggest_col_value = E::Real::faer_zero();

    let arch = E::Simd::default();

    for j in 0..n {
        let col_value = norm2(arch, matrix.rb().col(j));
        if col_value > biggest_col_value {
            biggest_col_value = col_value;
            biggest_col_idx = j;
        }
    }

    for k in 0..size {
        let mut matrix_right = matrix.rb_mut().submatrix(0, k, m, n - k);

        col_perm.swap(k, k + biggest_col_idx);
        if biggest_col_idx > 0 {
            n_transpositions += 1;
            swap_cols(matrix_right.rb_mut(), 0, biggest_col_idx);
        }

        let mut matrix = matrix.rb_mut().submatrix(k, k, m - k, n - k);
        let m = matrix.nrows();
        let n = matrix.ncols();

        let [_, _, first_col, last_cols] = matrix.rb_mut().split_at(0, 1);
        let first_col = first_col.col(0);

        let [mut first_head, mut first_tail] = first_col.split_at_row(1);
        let tail_squared_norm = norm2(arch, first_tail.rb());
        let (tau, beta) = faer_core::householder::make_householder_in_place(
            Some(first_tail.rb_mut()),
            first_head.read(0, 0),
            tail_squared_norm,
        );
        first_head.write(0, 0, beta);
        let tau_inv = tau.faer_inv();
        householder_coeffs.write(k, 0, tau);

        let first_tail = first_tail.rb();

        if n == 0 {
            return n_transpositions;
        }

        let extra_parallelism = if disable_parallelism(m, n) {
            Parallelism::None
        } else {
            parallelism
        };

        match extra_parallelism {
            Parallelism::None => {
                biggest_col_value = E::Real::faer_zero();
                biggest_col_idx = 0;

                process_cols(
                    arch,
                    last_cols,
                    0,
                    first_tail,
                    tau_inv,
                    &mut biggest_col_value,
                    &mut biggest_col_idx,
                );
            }
            #[cfg(feature = "rayon")]
            Parallelism::Rayon(_) => {
                use faer_core::{for_each_raw, par_split_indices, parallelism_degree, Ptr};
                let n_threads = parallelism_degree(parallelism);

                let mut biggest_col = vec![(E::Real::faer_zero(), 0_usize); n_threads];
                {
                    let biggest_col = Ptr(biggest_col.as_mut_ptr());
                    for_each_raw(
                        n_threads,
                        |idx| {
                            let (col_start, ncols) =
                                par_split_indices(last_cols.ncols(), idx, n_threads);
                            let matrix =
                                unsafe { last_cols.rb().subcols(col_start, ncols).const_cast() };

                            let mut local_biggest_col_value = E::Real::faer_zero();
                            let mut local_biggest_col_idx = 0;

                            process_cols(
                                arch,
                                matrix,
                                col_start,
                                first_tail,
                                tau_inv,
                                &mut local_biggest_col_value,
                                &mut local_biggest_col_idx,
                            );
                            unsafe {
                                *{ biggest_col }.0 =
                                    (local_biggest_col_value, local_biggest_col_idx);
                            }
                        },
                        parallelism,
                    );
                }

                biggest_col_value = E::Real::faer_zero();
                biggest_col_idx = 0;

                for (col_value, col_idx) in biggest_col {
                    if col_value > biggest_col_value {
                        biggest_col_value = col_value;
                        biggest_col_idx = col_idx;
                    }
                }
            }
        }
    }

    n_transpositions
}

struct ProcessCols<'a, E: ComplexField> {
    matrix: MatMut<'a, E>,
    offset: usize,
    first_tail: MatRef<'a, E>,
    tau_inv: E,
    biggest_col_value: &'a mut E::Real,
    biggest_col_idx: &'a mut usize,
}

impl<E: ComplexField> pulp::WithSimd for ProcessCols<'_, E> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self {
            mut matrix,
            offset,
            first_tail: first_tail_,
            tau_inv,
            biggest_col_value,
            biggest_col_idx,
        } = self;

        debug_assert_eq!(matrix.row_stride(), 1);
        debug_assert_eq!(first_tail_.row_stride(), 1);

        if matrix.nrows() == 0 {
            return;
        }

        let m = matrix.nrows() - 1;
        let first_tail = E::faer_map(
            first_tail_.as_ptr(),
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
        );

        for j in 0..matrix.ncols() {
            let [mut col_head, col_tail] = matrix.rb_mut().col(j).split_at_row(1);
            let col_head_ = col_head.read(0, 0);

            let col_tail = E::faer_map(
                col_tail.as_ptr(),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
            );

            let dot = inner_prod::AccConjAxB::<'_, E> {
                a: E::faer_rb(E::faer_as_ref(&first_tail)),
                b: E::faer_rb(E::faer_as_ref(&col_tail)),
            }
            .with_simd(simd)
            .faer_add(col_head_);

            let k = (tau_inv.faer_mul(dot)).faer_neg();
            col_head.write(0, 0, col_head_.faer_add(k));

            let col_value = UpdateAndNorm2 {
                a: col_tail,
                b: E::faer_rb(E::faer_as_ref(&first_tail)),
                k,
            }
            .with_simd(simd);

            if col_value > *biggest_col_value {
                *biggest_col_value = col_value;
                *biggest_col_idx = j + offset;
            }
        }
    }
}

#[inline(always)]
fn process_cols<E: ComplexField>(
    arch: E::Simd,
    mut matrix: MatMut<'_, E>,
    offset: usize,
    first_tail: MatRef<'_, E>,
    tau_inv: E,
    biggest_col_value: &mut E::Real,
    biggest_col_idx: &mut usize,
) {
    if matrix.row_stride() == 1 {
        arch.dispatch(ProcessCols {
            matrix,
            offset,
            first_tail,
            tau_inv,
            biggest_col_value,
            biggest_col_idx,
        });
    } else {
        for j in 0..matrix.ncols() {
            let [mut col_head, col_tail] = matrix.rb_mut().col(j).split_at_row(1);
            let col_head_ = col_head.read(0, 0);

            let dot = col_head_.faer_add(inner_prod_with_conj_arch(
                arch,
                first_tail,
                Conj::Yes,
                col_tail.rb(),
                Conj::No,
            ));
            let k = (tau_inv.faer_mul(dot)).faer_neg();
            col_head.write(0, 0, col_head_.faer_add(k));

            let col_value = update_and_norm2(arch, col_tail, first_tail, k);
            if col_value > *biggest_col_value {
                *biggest_col_value = col_value;
                *biggest_col_idx = j + offset;
            }
        }
    }
}

fn default_disable_parallelism(m: usize, n: usize) -> bool {
    let prod = m * n;
    prod < 192 * 256
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct ColPivQrComputeParams {
    /// At which size the parallelism should be disabled. `None` to automatically determine this
    /// threshold.
    pub disable_parallelism: Option<fn(nrows: usize, ncols: usize) -> bool>,
}

impl ColPivQrComputeParams {
    fn normalize(self) -> fn(usize, usize) -> bool {
        self.disable_parallelism
            .unwrap_or(default_disable_parallelism)
    }
}

/// Computes the size and alignment of required workspace for performing a QR decomposition
/// with column pivoting.
pub fn qr_in_place_req<E: Entity>(
    nrows: usize,
    ncols: usize,
    blocksize: usize,
    parallelism: Parallelism,
    params: ColPivQrComputeParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = nrows;
    let _ = ncols;
    let _ = parallelism;
    let _ = blocksize;
    let _ = &params;
    Ok(StackReq::default())
}

/// Computes the QR decomposition with pivoting of a rectangular matrix $A$, into a unitary matrix
/// $Q$, represented as a block Householder sequence, and an upper trapezoidal matrix $R$, such
/// that $$AP^\top = QR.$$
///
/// The Householder bases of $Q$ are stored in the strictly lower trapezoidal part of `matrix` with
/// an implicit unit diagonal, and its upper triangular Householder factors are stored in
/// `householder_factor`, blockwise in chunks of `blocksize×blocksize`.
///
/// The block size is chosed as the number of rows of `householder_factor`.
///
/// After the function returns, `col_perm` contains the order of the columns after pivoting, i.e.
/// the result is the same as computing the non-pivoted QR decomposition of the matrix `matrix[:,
/// col_perm]`. `col_perm_inv` contains its inverse permutation.
///
/// # Output
///
/// - The number of transpositions that constitute the permutation.
/// - a structure representing the permutation $P$.
///
/// # Panics
///
/// - Panics if the number of columns of the householder factor is not equal to the minimum of the
/// number of rows and the number of columns of the input matrix.
/// - Panics if the block size is zero.
/// - Panics if the length of `col_perm` and `col_perm_inv` is not equal to the number of columns
/// of `matrix`.
/// - Panics if the provided memory in `stack` is insufficient (see [`qr_in_place_req`]).
pub fn qr_in_place<'out, E: ComplexField>(
    matrix: MatMut<'_, E>,
    householder_factor: MatMut<'_, E>,
    col_perm: &'out mut [usize],
    col_perm_inv: &'out mut [usize],
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: ColPivQrComputeParams,
) -> (usize, PermutationMut<'out>) {
    let _ = &stack;
    let disable_parallelism = params.normalize();
    let m = matrix.nrows();
    let n = matrix.ncols();

    assert!(col_perm.len() == n);
    assert!(col_perm_inv.len() == n);

    for (j, p) in col_perm.iter_mut().enumerate() {
        *p = j;
    }

    let mut householder_factor = householder_factor;
    let householder_coeffs = householder_factor.rb_mut().row(0).transpose();

    let mut matrix = matrix;

    let n_transpositions = qr_in_place_colmajor(
        matrix.rb_mut(),
        householder_coeffs,
        col_perm,
        parallelism,
        disable_parallelism,
    );

    fn div_ceil(a: usize, b: usize) -> usize {
        let (div, rem) = (a / b, a % b);
        if rem == 0 {
            div
        } else {
            div + 1
        }
    }

    let blocksize = householder_factor.nrows();
    if blocksize > 1 {
        let size = householder_factor.ncols();
        let n_blocks = div_ceil(size, blocksize);

        let qr_factors = matrix.rb();

        let func = |idx: usize| {
            let j = idx * blocksize;
            let blocksize = Ord::min(blocksize, size - j);
            let mut householder = unsafe { householder_factor.rb().const_cast() }
                .submatrix(0, j, blocksize, blocksize);

            for i in 0..blocksize {
                let coeff = householder.read(0, i);
                householder.write(i, i, coeff);
            }

            let qr = qr_factors.submatrix(j, j, m - j, blocksize);

            upgrade_householder_factor(householder, qr, blocksize, 1, parallelism);
        };

        match parallelism {
            Parallelism::None => (0..n_blocks).for_each(func),
            #[cfg(feature = "rayon")]
            Parallelism::Rayon(_) => {
                use rayon::prelude::*;
                (0..n_blocks).into_par_iter().for_each(func)
            }
        }
    }

    for (j, &p) in col_perm.iter().enumerate() {
        col_perm_inv[p] = j;
    }

    (n_transpositions, unsafe {
        PermutationMut::new_unchecked(col_perm, col_perm_inv)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{
        c64,
        householder::{
            apply_block_householder_sequence_on_the_left_in_place_req,
            apply_block_householder_sequence_on_the_left_in_place_with_conj,
        },
        mul::matmul,
        zip::Diag,
        Conj, Mat, MatRef,
    };
    use rand::random;

    macro_rules! make_stack {
        ($req: expr $(,)?) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    fn reconstruct_factors<E: ComplexField>(
        qr_factors: MatRef<'_, E>,
        householder: MatRef<'_, E>,
    ) -> (Mat<E>, Mat<E>) {
        let m = qr_factors.nrows();
        let n = qr_factors.ncols();

        let mut q = Mat::zeros(m, m);
        let mut r = Mat::zeros(m, n);

        r.as_mut()
            .cwise()
            .zip(qr_factors)
            .for_each_triangular_upper(Diag::Include, |mut a, b| a.write(b.read()));

        q.as_mut()
            .diagonal()
            .cwise()
            .for_each(|mut a| a.write(E::faer_one()));

        apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr_factors,
            householder,
            Conj::No,
            q.as_mut(),
            Parallelism::Rayon(0),
            make_stack!(
                apply_block_householder_sequence_on_the_left_in_place_req::<E>(
                    m,
                    householder.nrows(),
                    m
                )
            ),
        );

        (q, r)
    }

    #[test]
    fn test_qr_f64() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            for (m, n) in [(2, 2), (2, 4), (4, 2), (4, 4), (63, 63), (1024, 1024)] {
                let mut mat = Mat::<f64>::from_fn(m, n, |_, _| random());
                let mat_orig = mat.clone();
                let size = m.min(n);
                let blocksize = 8;
                let mut householder = Mat::zeros(blocksize, size);
                let mut perm = vec![0; n];
                let mut perm_inv = vec![0; n];

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    make_stack!(qr_in_place_req::<f64>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let mut qr = Mat::zeros(m, n);
                matmul(
                    qr.as_mut(),
                    q.as_ref(),
                    r.as_ref(),
                    None,
                    1.0,
                    Parallelism::Rayon(8),
                );

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(qr.read(i, j), mat_orig.read(i, perm[j]));
                    }
                }
            }
        }
    }

    #[test]
    fn test_qr_c64() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            for (m, n) in [(2, 2), (2, 4), (4, 2), (4, 4), (63, 63)] {
                let mut mat = Mat::<c64>::from_fn(m, n, |_, _| c64::new(random(), random()));
                let mat_orig = mat.clone();
                let size = m.min(n);
                let blocksize = 8;
                let mut householder = Mat::zeros(blocksize, size);
                let mut perm = vec![0; n];
                let mut perm_inv = vec![0; n];

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    make_stack!(qr_in_place_req::<c64>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let mut qr = Mat::zeros(m, n);
                let mut qhq = Mat::zeros(m, m);
                matmul(
                    qr.as_mut(),
                    q.as_ref(),
                    r.as_ref(),
                    None,
                    c64::faer_one(),
                    Parallelism::Rayon(8),
                );

                matmul(
                    qhq.as_mut(),
                    q.as_ref().adjoint(),
                    q.as_ref(),
                    None,
                    c64::faer_one(),
                    Parallelism::Rayon(8),
                );

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(qr.read(i, j), mat_orig.read(i, perm[j]));
                    }
                }
            }
        }
    }

    #[test]
    fn test_qr_f32() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            for (m, n) in [(2, 2), (2, 4), (4, 2), (4, 4), (63, 63), (1024, 1024)] {
                let mut mat = Mat::<f32>::from_fn(m, n, |_, _| random());
                let mat_orig = mat.clone();
                let size = m.min(n);
                let blocksize = 8;
                let mut householder = Mat::zeros(blocksize, size);
                let mut perm = vec![0; n];
                let mut perm_inv = vec![0; n];

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    make_stack!(qr_in_place_req::<f32>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let mut qr = Mat::zeros(m, n);
                matmul(
                    qr.as_mut(),
                    q.as_ref(),
                    r.as_ref(),
                    None,
                    1.0,
                    Parallelism::Rayon(8),
                );

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(qr.read(i, j), mat_orig.read(i, perm[j]), 1e-4);
                    }
                }
            }
        }
    }

    #[test]
    fn test_qr_c32() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            for (m, n) in [(2, 2), (2, 4), (4, 2), (4, 4), (63, 63)] {
                let mut mat = Mat::<c32>::from_fn(m, n, |_, _| c32::new(random(), random()));
                let mat_orig = mat.clone();
                let size = m.min(n);
                let blocksize = 8;
                let mut householder = Mat::zeros(blocksize, size);
                let mut perm = vec![0; n];
                let mut perm_inv = vec![0; n];

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    make_stack!(qr_in_place_req::<c32>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let mut qr = Mat::zeros(m, n);
                let mut qhq = Mat::zeros(m, m);
                matmul(
                    qr.as_mut(),
                    q.as_ref(),
                    r.as_ref(),
                    None,
                    c32::faer_one(),
                    Parallelism::Rayon(8),
                );

                matmul(
                    qhq.as_mut(),
                    q.as_ref().adjoint(),
                    q.as_ref(),
                    None,
                    c32::faer_one(),
                    Parallelism::Rayon(8),
                );

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(qr.read(i, j), mat_orig.read(i, perm[j]), 1e-4);
                    }
                }
            }
        }
    }

    #[test]
    fn test_qr_c32_zeros() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            for (m, n) in [(2, 2), (2, 4), (4, 2), (4, 4), (63, 63)] {
                let mut mat = Mat::<c32>::zeros(m, n);
                let mat_orig = mat.clone();
                let size = m.min(n);
                let blocksize = 8;
                let mut householder = Mat::zeros(blocksize, size);
                let mut perm = vec![0; n];
                let mut perm_inv = vec![0; n];

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    make_stack!(qr_in_place_req::<c32>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let mut qr = Mat::zeros(m, n);
                let mut qhq = Mat::zeros(m, m);
                matmul(
                    qr.as_mut(),
                    q.as_ref(),
                    r.as_ref(),
                    None,
                    c32::faer_one(),
                    Parallelism::Rayon(8),
                );

                matmul(
                    qhq.as_mut(),
                    q.as_ref().adjoint(),
                    q.as_ref(),
                    None,
                    c32::faer_one(),
                    Parallelism::Rayon(8),
                );

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(qr.read(i, j), mat_orig.read(i, perm[j]), 1e-4);
                    }
                }
            }
        }
    }

    #[test]
    fn test_qr_c32_ones() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            for (m, n) in [(2, 2), (2, 4), (4, 2), (4, 4), (63, 63)] {
                let mut mat = Mat::<c32>::from_fn(m, n, |_, _| c32::faer_one());
                let mat_orig = mat.clone();
                let size = m.min(n);
                let blocksize = 8;
                let mut householder = Mat::zeros(blocksize, size);
                let mut perm = vec![0; n];
                let mut perm_inv = vec![0; n];

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    make_stack!(qr_in_place_req::<c32>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let mut qr = Mat::zeros(m, n);
                let mut qhq = Mat::zeros(m, m);
                matmul(
                    qr.as_mut(),
                    q.as_ref(),
                    r.as_ref(),
                    None,
                    c32::faer_one(),
                    Parallelism::Rayon(8),
                );

                matmul(
                    qhq.as_mut(),
                    q.as_ref().adjoint(),
                    q.as_ref(),
                    None,
                    c32::faer_one(),
                    Parallelism::Rayon(8),
                );

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(qr.read(i, j), mat_orig.read(i, perm[j]), 1e-4);
                    }
                }
            }
        }
    }

    #[test]
    fn test_qr_c32_rank_2() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            for (m, n) in [(2, 2), (2, 4), (4, 2), (4, 4), (63, 63)] {
                let u0 = Mat::from_fn(m, 1, |_, _| c32::new(random(), random()));
                let v0 = Mat::from_fn(1, n, |_, _| c32::new(random(), random()));
                let u1 = Mat::from_fn(m, 1, |_, _| c32::new(random(), random()));
                let v1 = Mat::from_fn(1, n, |_, _| c32::new(random(), random()));

                let mut mat = u0 * v0 + u1 * v1;
                let mat_orig = mat.clone();
                let size = m.min(n);
                let blocksize = 8;
                let mut householder = Mat::zeros(blocksize, size);
                let mut perm = vec![0; n];
                let mut perm_inv = vec![0; n];

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    make_stack!(qr_in_place_req::<c32>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let mut qr = Mat::zeros(m, n);
                let mut qhq = Mat::zeros(m, m);
                matmul(
                    qr.as_mut(),
                    q.as_ref(),
                    r.as_ref(),
                    None,
                    c32::faer_one(),
                    Parallelism::Rayon(8),
                );

                matmul(
                    qhq.as_mut(),
                    q.as_ref().adjoint(),
                    q.as_ref(),
                    None,
                    c32::faer_one(),
                    Parallelism::Rayon(8),
                );

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(qr.read(i, j), mat_orig.read(i, perm[j]), 1e-4);
                    }
                }
            }
        }
    }
}
