pub use crate::linalg::qr::no_pivoting::compute::recommended_blocksize;
use crate::{
    assert,
    complex_native::{c32, c64},
    debug_assert,
    linalg::{
        householder::upgrade_householder_factor,
        matmul::inner_prod::{self, inner_prod_with_conj_arch},
    },
    perm::{swap_cols_idx as swap_cols, PermRef},
    unzipped,
    utils::{simd::*, slice::*, DivCeil},
    zipped, ComplexField, Conj, Entity, Index, MatMut, MatRef, Parallelism, SignedIndex,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_entity::*;
use pulp::Simd;
use reborrow::*;

#[inline(always)]
fn update_and_norm2_simd_impl<'a, E: ComplexField, S: Simd>(
    simd: S,
    a: GroupFor<E, &'a mut [UnitFor<E>]>,
    b: GroupFor<E, &'a [UnitFor<E>]>,
    k: E,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> E::Real {
    let simd_real = SimdFor::<E::Real, S>::new(simd);
    let simd = SimdFor::<E, S>::new(simd);

    let a = SliceGroupMut::<'_, E>::new(a);
    let b = SliceGroup::<'_, E>::new(b);

    let (a_head, a_body, a_tail) = simd.as_aligned_simd_mut(a, offset);
    let (b_head, b_body, b_tail) = simd.as_aligned_simd(b, offset);

    let k = simd.splat(k);

    let (a_body8, a_body1) = a_body.as_arrays_mut::<8>();
    let (b_body8, b_body1) = b_body.as_arrays::<8>();

    #[inline(always)]
    fn process_init<E: ComplexField, S: Simd>(
        simd: SimdFor<E, S>,
        mut a: impl Write<Output = SimdGroupFor<E, S>>,
        b: impl Read<Output = SimdGroupFor<E, S>>,
        k: SimdGroupFor<E, S>,
    ) -> SimdGroupFor<E::Real, S> {
        let zero = simd.splat(E::faer_zero());
        let a_ = simd.mul_add_e(k, b.read_or(zero), a.read_or(zero));
        a.write(a_);
        simd.abs2(a_)
    }

    #[inline(always)]
    fn process<E: ComplexField, S: Simd>(
        simd: SimdFor<E, S>,
        mut a: impl Write<Output = SimdGroupFor<E, S>>,
        b: impl Read<Output = SimdGroupFor<E, S>>,
        k: SimdGroupFor<E, S>,
        acc: SimdGroupFor<E::Real, S>,
    ) -> SimdGroupFor<E::Real, S> {
        let zero = simd.splat(E::faer_zero());
        let a_ = simd.mul_add_e(k, b.read_or(zero), a.read_or(zero));
        a.write(a_);
        simd.abs2_add_e(a_, acc)
    }

    let mut acc0 = process_init(simd, a_head, b_head, k);
    let mut acc1 = simd_real.splat(E::Real::faer_zero());
    let mut acc2 = simd_real.splat(E::Real::faer_zero());
    let mut acc3 = simd_real.splat(E::Real::faer_zero());
    let mut acc4 = simd_real.splat(E::Real::faer_zero());
    let mut acc5 = simd_real.splat(E::Real::faer_zero());
    let mut acc6 = simd_real.splat(E::Real::faer_zero());
    let mut acc7 = simd_real.splat(E::Real::faer_zero());

    for ([a0, a1, a2, a3, a4, a5, a6, a7], [b0, b1, b2, b3, b4, b5, b6, b7]) in a_body8
        .into_mut_iter()
        .map(RefGroupMut::unzip)
        .zip(b_body8.into_ref_iter().map(RefGroup::unzip))
    {
        acc0 = process(simd, a0, b0, k, acc0);
        acc1 = process(simd, a1, b1, k, acc1);
        acc2 = process(simd, a2, b2, k, acc2);
        acc3 = process(simd, a3, b3, k, acc3);
        acc4 = process(simd, a4, b4, k, acc4);
        acc5 = process(simd, a5, b5, k, acc5);
        acc6 = process(simd, a6, b6, k, acc6);
        acc7 = process(simd, a7, b7, k, acc7);
    }
    for (a0, b0) in a_body1.into_mut_iter().zip(b_body1.into_ref_iter()) {
        acc0 = process(simd, a0, b0, k, acc0);
    }

    acc0 = process(simd, a_tail, b_tail, k, acc0);

    acc0 = simd_real.add(acc0, acc1);
    acc2 = simd_real.add(acc2, acc3);
    acc4 = simd_real.add(acc4, acc5);
    acc6 = simd_real.add(acc6, acc7);

    acc0 = simd_real.add(acc0, acc2);
    acc4 = simd_real.add(acc4, acc6);

    acc0 = simd_real.add(acc0, acc4);

    simd_real.reduce_add(simd_real.rotate_left(acc0, offset.rotate_left_amount()))
}

#[inline(always)]
fn update_and_norm2_simd_impl_c32<'a, S: Simd>(
    simd: S,
    a: &'a mut [c32],
    b: &'a [c32],
    k: c32,
    offset: pulp::Offset<SimdMaskFor<c32, S>>,
) -> f32 {
    let simd_real = SimdFor::<f32, S>::new(simd);
    let simd = SimdFor::<c32, S>::new(simd);

    let a = SliceGroupMut::<'_, c32>::new(a);
    let b = SliceGroup::<'_, c32>::new(b);

    let (a_head, a_body, a_tail) = simd.as_aligned_simd_mut(a, offset);
    let (b_head, b_body, b_tail) = simd.as_aligned_simd(b, offset);

    let (a_body8, a_body1) = a_body.as_arrays_mut::<8>();
    let (b_body8, b_body1) = b_body.as_arrays::<8>();

    #[inline(always)]
    fn process_init<S: Simd>(
        simd: SimdFor<c32, S>,
        mut a: impl Write<Output = SimdGroupFor<c32, S>>,
        b: impl Read<Output = SimdGroupFor<c32, S>>,
        k: SimdGroupFor<c32, S>,
    ) -> SimdGroupFor<f32, S> {
        let zero = simd.splat(c32 { re: 0.0, im: 0.0 });
        let a_ = simd.mul_add_e(k, b.read_or(zero), a.read_or(zero));
        a.write(a_);
        accumulate_init(simd.simd, a_)
    }

    #[inline(always)]
    fn process<S: Simd>(
        simd: SimdFor<c32, S>,
        mut a: impl Write<Output = SimdGroupFor<c32, S>>,
        b: impl Read<Output = SimdGroupFor<c32, S>>,
        k: SimdGroupFor<c32, S>,
        acc: SimdGroupFor<f32, S>,
    ) -> SimdGroupFor<f32, S> {
        let zero = simd.splat(c32 { re: 0.0, im: 0.0 });
        let a_ = simd.mul_add_e(k, b.read_or(zero), a.read_or(zero));
        a.write(a_);
        accumulate(simd.simd, acc, a_)
    }

    #[inline(always)]
    fn accumulate_init<S: Simd>(simd: S, a: S::c32s) -> S::f32s {
        if coe::is_same::<S, pulp::Scalar>() {
            let norm2: c32 = bytemuck::cast(simd.c32s_abs2(a));
            bytemuck::cast(norm2.re)
        } else {
            simd.f32s_mul(bytemuck::cast(a), bytemuck::cast(a))
        }
    }

    #[inline(always)]
    fn accumulate<S: Simd>(simd: S, acc: S::f32s, a: S::c32s) -> S::f32s {
        if coe::is_same::<S, pulp::Scalar>() {
            let norm2: c32 = bytemuck::cast(simd.c32s_abs2(a));
            let acc: f32 = bytemuck::cast(acc);
            bytemuck::cast(norm2.re + acc)
        } else {
            simd.f32s_mul_add_e(bytemuck::cast(a), bytemuck::cast(a), acc)
        }
    }

    let k = simd.splat(k);
    let mut acc0 = process_init(simd, a_head, b_head, k);
    let mut acc1 = simd_real.splat(0.0);
    let mut acc2 = simd_real.splat(0.0);
    let mut acc3 = simd_real.splat(0.0);
    let mut acc4 = simd_real.splat(0.0);
    let mut acc5 = simd_real.splat(0.0);
    let mut acc6 = simd_real.splat(0.0);
    let mut acc7 = simd_real.splat(0.0);

    for ([a0, a1, a2, a3, a4, a5, a6, a7], [b0, b1, b2, b3, b4, b5, b6, b7]) in a_body8
        .into_mut_iter()
        .map(RefGroupMut::unzip)
        .zip(b_body8.into_ref_iter().map(RefGroup::unzip))
    {
        acc0 = process(simd, a0, b0, k, acc0);
        acc1 = process(simd, a1, b1, k, acc1);
        acc2 = process(simd, a2, b2, k, acc2);
        acc3 = process(simd, a3, b3, k, acc3);
        acc4 = process(simd, a4, b4, k, acc4);
        acc5 = process(simd, a5, b5, k, acc5);
        acc6 = process(simd, a6, b6, k, acc6);
        acc7 = process(simd, a7, b7, k, acc7);
    }
    for (a0, b0) in a_body1.into_mut_iter().zip(b_body1.into_ref_iter()) {
        acc0 = process(simd, a0, b0, k, acc0);
    }

    acc0 = process(simd, a_tail, b_tail, k, acc0);

    acc0 = simd_real.add(acc0, acc1);
    acc2 = simd_real.add(acc2, acc3);
    acc4 = simd_real.add(acc4, acc5);
    acc6 = simd_real.add(acc6, acc7);

    acc0 = simd_real.add(acc0, acc2);
    acc4 = simd_real.add(acc4, acc6);

    acc0 = simd_real.add(acc0, acc4);

    simd.simd.f32s_reduce_sum(
        simd.simd
            .f32s_rotate_left(acc0, 2 * offset.rotate_left_amount()),
    )
}

#[inline(always)]
fn update_and_norm2_simd_impl_c64<'a, S: Simd>(
    simd: S,
    a: &'a mut [c64],
    b: &'a [c64],
    k: c64,
    offset: pulp::Offset<SimdMaskFor<c64, S>>,
) -> f64 {
    let simd_real = SimdFor::<f64, S>::new(simd);
    let simd = SimdFor::<c64, S>::new(simd);

    let a = SliceGroupMut::<'_, c64>::new(a);
    let b = SliceGroup::<'_, c64>::new(b);

    let (a_head, a_body, a_tail) = simd.as_aligned_simd_mut(a, offset);
    let (b_head, b_body, b_tail) = simd.as_aligned_simd(b, offset);

    let (a_body8, a_body1) = a_body.as_arrays_mut::<8>();
    let (b_body8, b_body1) = b_body.as_arrays::<8>();

    #[inline(always)]
    fn process_init<S: Simd>(
        simd: SimdFor<c64, S>,
        mut a: impl Write<Output = SimdGroupFor<c64, S>>,
        b: impl Read<Output = SimdGroupFor<c64, S>>,
        k: SimdGroupFor<c64, S>,
    ) -> SimdGroupFor<f64, S> {
        let zero = simd.splat(c64 { re: 0.0, im: 0.0 });
        let a_ = simd.mul_add_e(k, b.read_or(zero), a.read_or(zero));
        a.write(a_);
        accumulate_init(simd.simd, a_)
    }

    #[inline(always)]
    fn process<S: Simd>(
        simd: SimdFor<c64, S>,
        mut a: impl Write<Output = SimdGroupFor<c64, S>>,
        b: impl Read<Output = SimdGroupFor<c64, S>>,
        k: SimdGroupFor<c64, S>,
        acc: SimdGroupFor<f64, S>,
    ) -> SimdGroupFor<f64, S> {
        let zero = simd.splat(c64 { re: 0.0, im: 0.0 });
        let a_ = simd.mul_add_e(k, b.read_or(zero), a.read_or(zero));
        a.write(a_);
        accumulate(simd.simd, acc, a_)
    }

    #[inline(always)]
    fn accumulate_init<S: Simd>(simd: S, a: S::c64s) -> S::f64s {
        if coe::is_same::<S, pulp::Scalar>() {
            let norm2: c64 = bytemuck::cast(simd.c64s_abs2(a));
            bytemuck::cast(norm2.re)
        } else {
            simd.f64s_mul(bytemuck::cast(a), bytemuck::cast(a))
        }
    }

    #[inline(always)]
    fn accumulate<S: Simd>(simd: S, acc: S::f64s, a: S::c64s) -> S::f64s {
        if coe::is_same::<S, pulp::Scalar>() {
            let norm2: c64 = bytemuck::cast(simd.c64s_abs2(a));
            let acc: f64 = bytemuck::cast(acc);
            bytemuck::cast(norm2.re + acc)
        } else {
            simd.f64s_mul_add_e(bytemuck::cast(a), bytemuck::cast(a), acc)
        }
    }

    let k = simd.splat(k);
    let mut acc0 = process_init(simd, a_head, b_head, k);
    let mut acc1 = simd_real.splat(0.0);
    let mut acc2 = simd_real.splat(0.0);
    let mut acc3 = simd_real.splat(0.0);
    let mut acc4 = simd_real.splat(0.0);
    let mut acc5 = simd_real.splat(0.0);
    let mut acc6 = simd_real.splat(0.0);
    let mut acc7 = simd_real.splat(0.0);

    for ([a0, a1, a2, a3, a4, a5, a6, a7], [b0, b1, b2, b3, b4, b5, b6, b7]) in a_body8
        .into_mut_iter()
        .map(RefGroupMut::unzip)
        .zip(b_body8.into_ref_iter().map(RefGroup::unzip))
    {
        acc0 = process(simd, a0, b0, k, acc0);
        acc1 = process(simd, a1, b1, k, acc1);
        acc2 = process(simd, a2, b2, k, acc2);
        acc3 = process(simd, a3, b3, k, acc3);
        acc4 = process(simd, a4, b4, k, acc4);
        acc5 = process(simd, a5, b5, k, acc5);
        acc6 = process(simd, a6, b6, k, acc6);
        acc7 = process(simd, a7, b7, k, acc7);
    }
    for (a0, b0) in a_body1.into_mut_iter().zip(b_body1.into_ref_iter()) {
        acc0 = process(simd, a0, b0, k, acc0);
    }

    acc0 = process(simd, a_tail, b_tail, k, acc0);

    acc0 = simd_real.add(acc0, acc1);
    acc2 = simd_real.add(acc2, acc3);
    acc4 = simd_real.add(acc4, acc5);
    acc6 = simd_real.add(acc6, acc7);

    acc0 = simd_real.add(acc0, acc2);
    acc4 = simd_real.add(acc4, acc6);

    acc0 = simd_real.add(acc0, acc4);

    simd.simd.f64s_reduce_sum(
        simd.simd
            .f64s_rotate_left(acc0, 2 * offset.rotate_left_amount()),
    )
}

struct UpdateAndNorm2<'a, E: ComplexField> {
    a: GroupFor<E, &'a mut [UnitFor<E>]>,
    b: GroupFor<E, &'a [UnitFor<E>]>,
    k: E,
}

impl<E: ComplexField> UpdateAndNorm2<'_, E> {
    #[inline(always)]
    fn with_simd_and_offset<S: Simd>(
        self,
        simd: S,
        offset: pulp::Offset<SimdMaskFor<E, S>>,
    ) -> E::Real {
        let Self { a, b, k } = self;
        if coe::is_same::<c32, E>() {
            return coe::coerce_static(unsafe {
                update_and_norm2_simd_impl_c32(
                    simd,
                    transmute_unchecked(a),
                    transmute_unchecked(b),
                    transmute_unchecked(k),
                    transmute_unchecked(offset),
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
                    transmute_unchecked(offset),
                )
            });
        }
        update_and_norm2_simd_impl(simd, a, b, k, offset)
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
    let _ = arch;
    let mut acc = E::Real::faer_zero();
    zipped!(a, b).for_each(|unzipped!(mut a_, b)| {
        let a = a_.read();
        let b = b.read();

        a_.write(a.faer_add(k.faer_mul(b)));
        acc = acc.faer_add((a.faer_conj().faer_mul(a)).faer_real());
    });

    acc
}

fn qr_in_place_colmajor<I: Index, E: ComplexField>(
    mut matrix: MatMut<'_, E>,
    mut householder_coeffs: MatMut<'_, E>,
    col_perm: &mut [I],
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
        let col_value = norm2(arch, matrix.rb().col(j).as_2d());
        if col_value > biggest_col_value {
            biggest_col_value = col_value;
            biggest_col_idx = j;
        }
    }

    for k in 0..size {
        let mut matrix_right = matrix.rb_mut().submatrix_mut(0, k, m, n - k);

        col_perm.swap(k, k + biggest_col_idx);
        if biggest_col_idx > 0 {
            n_transpositions += 1;
            swap_cols(matrix_right.rb_mut(), 0, biggest_col_idx);
        }

        let mut matrix = matrix.rb_mut().submatrix_mut(k, k, m - k, n - k);
        let m = matrix.nrows();
        let n = matrix.ncols();

        let (_, _, first_col, last_cols) = matrix.rb_mut().split_at_mut(0, 1);
        let first_col = first_col.col_mut(0);

        let (mut first_head, mut first_tail) = first_col.split_at_mut(1);
        let tail_norm = first_tail.norm_l2();

        let (tau, beta) = crate::linalg::householder::make_householder_in_place(
            Some(first_tail.rb_mut().as_2d_mut()),
            first_head.read(0),
            tail_norm,
        );
        first_head.write(0, beta);
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
                    first_tail.as_2d(),
                    tau_inv,
                    &mut biggest_col_value,
                    &mut biggest_col_idx,
                );
            }
            #[cfg(feature = "rayon")]
            Parallelism::Rayon(_) => {
                use crate::utils::thread::{
                    for_each_raw, par_split_indices, parallelism_degree, Ptr,
                };
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
                                first_tail.as_2d(),
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
            offset: col_offset,
            first_tail,
            tau_inv,
            biggest_col_value,
            biggest_col_idx,
        } = self;

        debug_assert_eq!(matrix.row_stride(), 1);
        debug_assert_eq!(first_tail.row_stride(), 1);

        if matrix.nrows() == 0 {
            return;
        }

        let first = SliceGroup::<'_, E>::new(first_tail.try_get_contiguous_col(0));
        let simd = SimdFor::<E, S>::new(simd);
        let offset = simd.align_offset(first.rb());

        for j in 0..matrix.ncols() {
            let (mut col0, col) =
                SliceGroupMut::<'_, E>::new(matrix.rb_mut().try_get_contiguous_col_mut(j))
                    .split_at(1);
            let col0_ = col0.read(0);

            let dot = inner_prod::with_simd_and_offset(simd, YesConj, first, col.rb(), offset)
                .faer_add(col0.read(0));

            let k = (dot.faer_mul(tau_inv)).faer_neg();
            col0.write(0, col0_.faer_add(k));

            let col_value = UpdateAndNorm2 {
                a: col.into_inner(),
                b: first.into_inner(),
                k,
            }
            .with_simd_and_offset(simd.simd, offset);

            if col_value > *biggest_col_value {
                *biggest_col_value = col_value;
                *biggest_col_idx = j + col_offset;
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
            let (mut col_head, col_tail) = matrix.rb_mut().col_mut(j).split_at_mut(1);
            let col_head_ = col_head.read(0);

            let dot = col_head_.faer_add(inner_prod_with_conj_arch(
                arch,
                first_tail,
                Conj::Yes,
                col_tail.rb().as_2d(),
                Conj::No,
            ));
            let k = (tau_inv.faer_mul(dot)).faer_neg();
            col_head.write(0, col_head_.faer_add(k));

            let col_value = update_and_norm2(arch, col_tail.as_2d_mut(), first_tail, k);
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
pub fn qr_in_place_req<I: Index, E: Entity>(
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

#[derive(Copy, Clone, Debug)]
pub struct ColPivQrInfo {
    pub transposition_count: usize,
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
pub fn qr_in_place<'out, I: Index, E: ComplexField>(
    matrix: MatMut<'_, E>,
    householder_factor: MatMut<'_, E>,
    col_perm: &'out mut [I],
    col_perm_inv: &'out mut [I],
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: ColPivQrComputeParams,
) -> (ColPivQrInfo, PermRef<'out, I>) {
    fn implementation<'out, I: Index, E: ComplexField>(
        matrix: MatMut<'_, E>,
        householder_factor: MatMut<'_, E>,
        col_perm: &'out mut [I],
        col_perm_inv: &'out mut [I],
        parallelism: Parallelism,
        stack: PodStack<'_>,
        params: ColPivQrComputeParams,
    ) -> (usize, PermRef<'out, I>) {
        {
            let truncate = <I::Signed as SignedIndex>::truncate;

            let _ = &stack;
            let disable_parallelism = params.normalize();
            let m = matrix.nrows();
            let n = matrix.ncols();

            assert!(all(col_perm.len() == n, col_perm_inv.len() == n));

            #[cfg(feature = "perf-warn")]
            if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(QR_WARN) {
                if matrix.col_stride().unsigned_abs() == 1 {
                    log::warn!(target: "faer_perf", "QR with column pivoting prefers column-major matrix. Found row-major matrix.");
                } else {
                    log::warn!(target: "faer_perf", "QR with column pivoting prefers column-major matrix. Found matrix with generic strides.");
                }
            }

            for (j, p) in col_perm.iter_mut().enumerate() {
                *p = I::from_signed(truncate(j));
            }

            let mut householder_factor = householder_factor;
            let householder_coeffs = householder_factor.rb_mut().row_mut(0).transpose_mut();

            let mut matrix = matrix;

            let n_transpositions = qr_in_place_colmajor(
                matrix.rb_mut(),
                householder_coeffs.as_2d_mut(),
                col_perm,
                parallelism,
                disable_parallelism,
            );

            let blocksize = householder_factor.nrows();
            if blocksize > 1 {
                let size = householder_factor.ncols();
                let n_blocks = size.msrv_div_ceil(blocksize);

                let qr_factors = matrix.rb();

                let func = |idx: usize| {
                    let j = idx * blocksize;
                    let blocksize = Ord::min(blocksize, size - j);
                    let mut householder = unsafe { householder_factor.rb().const_cast() }
                        .submatrix_mut(0, j, blocksize, blocksize);

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
                col_perm_inv[p.to_signed().zx()] = I::from_signed(truncate(j));
            }

            (n_transpositions, unsafe {
                PermRef::new_unchecked(col_perm, col_perm_inv)
            })
        }
    }

    let (n_transpositions, perm) = implementation(
        matrix,
        householder_factor,
        I::canonicalize_mut(col_perm),
        I::canonicalize_mut(col_perm_inv),
        parallelism,
        stack,
        params,
    );
    (
        ColPivQrInfo {
            transposition_count: n_transpositions,
        },
        perm.uncanonicalized::<I>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert,
        complex_native::c64,
        linalg::{
            householder::{
                apply_block_householder_sequence_on_the_left_in_place_req,
                apply_block_householder_sequence_on_the_left_in_place_with_conj,
            },
            matmul::matmul,
            zip::Diag,
        },
        Conj, Mat, MatRef,
    };
    use assert_approx_eq::assert_approx_eq;
    use matrixcompare::assert_matrix_eq;
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

        let mut q = Mat::<E>::zeros(m, m);
        let mut r = Mat::<E>::zeros(m, n);

        zipped!(r.as_mut(), qr_factors)
            .for_each_triangular_upper(Diag::Include, |unzipped!(mut a, b)| a.write(b.read()));

        zipped!(q.as_mut().diagonal_mut().column_vector_mut().as_2d_mut())
            .for_each(|unzipped!(mut a)| a.write(E::faer_one()));

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
                let mut perm = vec![0usize; n];
                let mut perm_inv = vec![0usize; n];

                let (_, p) = qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    make_stack!(qr_in_place_req::<usize, f64>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let qr = &q * &r;
                let qhq = &q * q.adjoint();
                assert_matrix_eq!(qr, &mat_orig * p.rb().inverse(), comp = abs, tol = 1e-10);
                assert_matrix_eq!(qhq, Mat::<f64>::identity(m, m), comp = abs, tol = 1e-10);
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
                let mut perm = vec![0usize; n];
                let mut perm_inv = vec![0usize; n];

                let (_, p) = qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    make_stack!(qr_in_place_req::<usize, c64>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let qr = &q * &r;
                let qhq = &q * q.adjoint();
                assert!((&qr - &mat_orig * p.rb().inverse()).norm_max() < 1e-10);
                assert!((&qhq - Mat::<c64>::identity(m, m)).norm_max() < 1e-10);
            }
        }
    }

    #[test]
    fn test_qr_f32() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            for (m, n) in [
                (128, 4),
                (2, 2),
                (2, 4),
                (4, 2),
                (4, 4),
                (63, 63),
                (1024, 1024),
            ] {
                let mut mat = Mat::<f32>::from_fn(m, n, |_, _| random());
                let mat_orig = mat.clone();
                let size = m.min(n);
                let blocksize = 8;
                let mut householder = Mat::zeros(blocksize, size);
                let mut perm = vec![0usize; n];
                let mut perm_inv = vec![0usize; n];

                let (_, p) = qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    make_stack!(qr_in_place_req::<usize, f32>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let qr = &q * &r;
                let qhq = &q * q.adjoint();
                assert!((&qr - &mat_orig * p.rb().inverse()).norm_max() < 1e-4);
                assert!((&qhq - Mat::<f32>::identity(m, m)).norm_max() < 1e-4);
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
                let mut perm = vec![0usize; n];
                let mut perm_inv = vec![0usize; n];

                let (_, p) = qr_in_place(
                    mat.as_mut(),
                    householder.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    parallelism,
                    make_stack!(qr_in_place_req::<usize, c32>(
                        m,
                        n,
                        blocksize,
                        parallelism,
                        Default::default()
                    )),
                    Default::default(),
                );

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
                let qr = &q * &r;
                let qhq = &q * q.adjoint();
                assert!((&qr - &mat_orig * p.rb().inverse()).norm_max() < 1e-4);
                assert!((&qhq - Mat::<c32>::identity(m, m)).norm_max() < 1e-4);
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
                    make_stack!(qr_in_place_req::<usize, c32>(
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

                for (j, &pj) in perm.iter().enumerate() {
                    for i in 0..m {
                        assert_approx_eq!(qr.read(i, j), mat_orig.read(i, pj), 1e-4);
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
                    make_stack!(qr_in_place_req::<usize, c32>(
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

                for (j, &pj) in perm.iter().enumerate() {
                    for i in 0..m {
                        assert_approx_eq!(qr.read(i, j), mat_orig.read(i, pj), 1e-4);
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
                    make_stack!(qr_in_place_req::<usize, c32>(
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

                for (j, &pj) in perm.iter().enumerate() {
                    for i in 0..m {
                        assert_approx_eq!(qr.read(i, j), mat_orig.read(i, pj), 1e-4);
                    }
                }
            }
        }
    }
}
