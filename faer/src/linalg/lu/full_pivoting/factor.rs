use faer_traits::{Real, RealMarker};
use linalg::matmul::matmul;
use pulp::Simd;

use crate::{
    internal_prelude::*,
    perm::{swap_cols_idx, swap_rows_idx},
    utils::thread::par_split_indices,
};

#[inline(always)]
fn best_value<T: ComplexField, S: Simd>(
    simd: &SimdCtx<T, S>,
    best_value: RealMarker<T::SimdVec<S>>,
    best_indices: T::SimdIndex<S>,
    value: T::SimdVec<S>,
    indices: T::SimdIndex<S>,
) -> (RealMarker<T::SimdVec<S>>, T::SimdIndex<S>) {
    let value = simd.abs1(value);
    let is_better = simd.gt(value, best_value);
    (
        RealMarker(simd.select(is_better, value.0, best_value.0)),
        simd.iselect(is_better, indices, best_indices),
    )
}

#[inline(always)]
fn best_score<T: ComplexField, S: Simd>(
    simd: &SimdCtx<T, S>,
    best_score: RealMarker<T::SimdVec<S>>,
    best_indices: T::SimdIndex<S>,
    score: RealMarker<T::SimdVec<S>>,
    indices: T::SimdIndex<S>,
) -> (RealMarker<T::SimdVec<S>>, T::SimdIndex<S>) {
    let is_better = simd.gt(score, best_score);
    (
        RealMarker(simd.select(is_better, score.0, best_score.0)),
        simd.iselect(is_better, indices, best_indices),
    )
}

#[inline(always)]
fn best_score_2d<T: ComplexField, S: Simd>(
    simd: &SimdCtx<T, S>,
    best_score: RealMarker<T::SimdVec<S>>,
    best_row: T::SimdIndex<S>,
    best_col: T::SimdIndex<S>,
    score: RealMarker<T::SimdVec<S>>,
    row: T::SimdIndex<S>,
    col: T::SimdIndex<S>,
) -> (RealMarker<T::SimdVec<S>>, T::SimdIndex<S>, T::SimdIndex<S>) {
    let is_better = simd.gt(score, best_score);
    (
        RealMarker(simd.select(is_better, score.0, best_score.0)),
        simd.iselect(is_better, row, best_row),
        simd.iselect(is_better, col, best_col),
    )
}

#[inline(always)]
#[math]
fn reduce_2d<T: ComplexField, S: Simd>(
    simd: &SimdCtx<T, S>,
    best_values: RealMarker<T::SimdVec<S>>,
    best_row: T::SimdIndex<S>,
    best_col: T::SimdIndex<S>,
) -> (usize, usize, Real<T>) {
    let best_val = simd.reduce_max_real(best_values);

    let best_val_splat = simd.splat_real(&best_val);
    let is_best = simd.ge(best_values, best_val_splat);
    let idx = simd.first_true_mask(is_best);

    let best_row =
        bytemuck::cast_slice::<T::SimdIndex<S>, T::Index>(core::slice::from_ref(&best_row))[idx];
    let best_col =
        bytemuck::cast_slice::<T::SimdIndex<S>, T::Index>(core::slice::from_ref(&best_col))[idx];

    (best_row.zx(), best_col.zx(), best_val)
}

#[inline(always)]
#[math]
fn best_in_col_simd<'M, T: ComplexField, S: Simd>(
    simd: SimdCtx<'M, T, S>,
    data: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
) -> (RealMarker<T::SimdVec<S>>, T::SimdIndex<S>) {
    let (head, body4, body1, tail) = simd.batch_indices::<4>();

    let iota = T::simd_iota(&simd.0);
    let lane_count = size_of::<T::SimdVec<S>>() / size_of::<T>();

    let inc1 = simd.isplat(T::Index::truncate(lane_count));
    let inc4 = simd.isplat(T::Index::truncate(4 * lane_count));

    let mut best_val0 = simd.splat_real(&zero());
    let mut best_val1 = simd.splat_real(&zero());
    let mut best_val2 = simd.splat_real(&zero());
    let mut best_val3 = simd.splat_real(&zero());

    let mut best_idx0 = simd.isplat(T::Index::truncate(0));
    let mut best_idx1 = simd.isplat(T::Index::truncate(0));
    let mut best_idx2 = simd.isplat(T::Index::truncate(0));
    let mut best_idx3 = simd.isplat(T::Index::truncate(0));

    let mut idx0 = simd.iadd(
        iota,
        simd.isplat(T::Index::truncate(simd.offset().wrapping_neg())),
    );
    let mut idx1 = simd.iadd(idx0, inc1);
    let mut idx2 = simd.iadd(idx1, inc1);
    let mut idx3 = simd.iadd(idx2, inc1);

    if let Some(i0) = head {
        (best_val0, best_idx0) = best_value(&simd, best_val0, best_idx0, simd.read(data, i0), idx0);
        idx0 = simd.iadd(idx0, inc1);
    }

    for [i0, i1, i2, i3] in body4 {
        (best_val0, best_idx0) = best_value(&simd, best_val0, best_idx0, simd.read(data, i0), idx0);
        (best_val1, best_idx1) = best_value(&simd, best_val1, best_idx1, simd.read(data, i1), idx1);
        (best_val2, best_idx2) = best_value(&simd, best_val2, best_idx2, simd.read(data, i2), idx2);
        (best_val3, best_idx3) = best_value(&simd, best_val3, best_idx3, simd.read(data, i3), idx3);

        idx0 = simd.iadd(idx0, inc4);
        idx1 = simd.iadd(idx1, inc4);
        idx2 = simd.iadd(idx2, inc4);
        idx3 = simd.iadd(idx3, inc4);
    }

    for i0 in body1 {
        (best_val0, best_idx0) = best_value(&simd, best_val0, best_idx0, simd.read(data, i0), idx0);
        idx0 = simd.iadd(idx0, inc1);
    }

    if let Some(i0) = tail {
        (best_val0, best_idx0) = best_value(&simd, best_val0, best_idx0, simd.read(data, i0), idx0);
    }

    (best_val0, best_idx0) = best_score(&simd, best_val0, best_idx0, best_val1, best_idx1);
    (best_val2, best_idx2) = best_score(&simd, best_val2, best_idx2, best_val3, best_idx3);
    best_score(&simd, best_val0, best_idx0, best_val2, best_idx2)
}

#[inline(always)]
#[math]
fn update_and_best_in_col_simd<'M, T: ComplexField, S: Simd>(
    simd: SimdCtx<'M, T, S>,
    data: ColMut<'_, T, Dim<'M>, ContiguousFwd>,
    lhs: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
    rhs: T,
) -> (RealMarker<T::SimdVec<S>>, T::SimdIndex<S>) {
    let mut data = data;

    let (head, body4, body1, tail) = simd.batch_indices::<3>();

    let iota = T::simd_iota(&simd.0);
    let lane_count = size_of::<T::SimdVec<S>>() / size_of::<T>();

    let inc1 = simd.isplat(T::Index::truncate(lane_count));
    let inc3 = simd.isplat(T::Index::truncate(3 * lane_count));

    let mut best_val0 = simd.splat_real(&zero());
    let mut best_val1 = simd.splat_real(&zero());
    let mut best_val2 = simd.splat_real(&zero());

    let mut best_idx0 = simd.isplat(T::Index::truncate(0));
    let mut best_idx1 = simd.isplat(T::Index::truncate(0));
    let mut best_idx2 = simd.isplat(T::Index::truncate(0));

    let mut idx0 = simd.iadd(
        iota,
        simd.isplat(T::Index::truncate(simd.offset().wrapping_neg())),
    );
    let mut idx1 = simd.iadd(idx0, inc1);
    let mut idx2 = simd.iadd(idx1, inc1);

    let rhs = simd.splat(&-rhs);

    if let Some(i0) = head {
        let mut x0 = simd.read(data.rb(), i0);
        let l0 = simd.read(lhs, i0);
        x0 = simd.mul_add(l0, rhs, x0);

        (best_val0, best_idx0) = best_value(&simd, best_val0, best_idx0, x0, idx0);
        idx0 = simd.iadd(idx0, inc1);

        simd.write(data.rb_mut(), i0, x0);
    }

    for [i0, i1, i2] in body4 {
        let mut x0 = simd.read(data.rb(), i0);
        let l0 = simd.read(lhs, i0);
        x0 = simd.mul_add(l0, rhs, x0);
        (best_val0, best_idx0) =
            best_value(&simd, best_val0, best_idx0, simd.read(data.rb(), i0), idx0);
        simd.write(data.rb_mut(), i0, x0);

        let mut x1 = simd.read(data.rb(), i1);
        let l1 = simd.read(lhs, i1);
        x1 = simd.mul_add(l1, rhs, x1);
        (best_val1, best_idx1) =
            best_value(&simd, best_val1, best_idx1, simd.read(data.rb(), i1), idx1);
        simd.write(data.rb_mut(), i1, x1);

        let mut x2 = simd.read(data.rb(), i2);
        let l2 = simd.read(lhs, i2);
        x2 = simd.mul_add(l2, rhs, x2);
        (best_val2, best_idx2) =
            best_value(&simd, best_val2, best_idx2, simd.read(data.rb(), i2), idx2);
        simd.write(data.rb_mut(), i2, x2);

        idx0 = simd.iadd(idx0, inc3);
        idx1 = simd.iadd(idx1, inc3);
        idx2 = simd.iadd(idx2, inc3);
    }

    for i0 in body1 {
        let mut x0 = simd.read(data.rb(), i0);
        let l0 = simd.read(lhs, i0);
        x0 = simd.mul_add(l0, rhs, x0);

        (best_val0, best_idx0) = best_value(&simd, best_val0, best_idx0, x0, idx0);
        idx0 = simd.iadd(idx0, inc1);

        simd.write(data.rb_mut(), i0, x0);
    }

    if let Some(i0) = tail {
        let mut x0 = simd.read(data.rb(), i0);
        let l0 = simd.read(lhs, i0);
        x0 = simd.mul_add(l0, rhs, x0);

        (best_val0, best_idx0) = best_value(&simd, best_val0, best_idx0, x0, idx0);

        simd.write(data.rb_mut(), i0, x0);
    }

    (best_val0, best_idx0) = best_score(&simd, best_val0, best_idx0, best_val1, best_idx1);
    best_score(&simd, best_val0, best_idx0, best_val2, best_idx2)
}

#[inline(always)]
fn best_in_mat_simd<'M, 'N, T: ComplexField>(
    data: MatRef<'_, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
) -> (usize, usize, Real<T>) {
    struct Impl<'a, 'M, 'N, T: ComplexField> {
        data: MatRef<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
    }

    impl<'a, 'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'N, T> {
        type Output = (usize, usize, Real<T>);

        #[math]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self { data } = self;

            let M = data.nrows();
            let N = data.ncols();
            let simd = SimdCtx::<'_, T, S>::new(T::simd_ctx(simd), M);

            let mut best_row = simd.isplat(T::Index::truncate(0));
            let mut best_col = simd.isplat(T::Index::truncate(0));
            let mut best_val = simd.splat_real(&zero());

            for j in N.indices() {
                let col = data.col(j);
                let (best_val_j, best_row_j) = best_in_col_simd(simd, col);

                (best_val, best_row, best_col) = best_score_2d(
                    &simd,
                    best_val,
                    best_row,
                    best_col,
                    best_val_j,
                    best_row_j,
                    simd.isplat(T::Index::truncate(*j)),
                );
            }
            reduce_2d(&simd, best_val, best_row, best_col)
        }
    }

    T::Arch::default().dispatch(Impl { data })
}

#[inline(always)]
fn update_and_best_in_mat_simd<'M, 'N, T: ComplexField>(
    data: MatMut<'_, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
    lhs: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
    rhs: RowRef<'_, T, Dim<'N>>,
    align: usize,
) -> (usize, usize, Real<T>) {
    struct Impl<'a, 'M, 'N, T: ComplexField> {
        data: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
        lhs: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
        rhs: RowRef<'a, T, Dim<'N>>,
        align: usize,
    }

    impl<'a, 'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'N, T> {
        type Output = (usize, usize, Real<T>);

        #[math]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self {
                data,
                lhs,
                rhs,
                align,
            } = self;

            let M = data.nrows();
            let N = data.ncols();
            let simd = SimdCtx::<'_, T, S>::new_align(T::simd_ctx(simd), M, align);

            let mut best_row = simd.isplat(T::Index::truncate(0));
            let mut best_col = simd.isplat(T::Index::truncate(0));
            let mut best_val = simd.splat_real(&zero());
            let mut data = data;

            for j in N.indices() {
                let data = data.rb_mut().col_mut(j);
                let rhs = copy(rhs[j]);
                let (best_val_j, best_row_j) = update_and_best_in_col_simd(simd, data, lhs, rhs);

                (best_val, best_row, best_col) = best_score_2d(
                    &simd,
                    best_val,
                    best_row,
                    best_col,
                    best_val_j,
                    best_row_j,
                    simd.isplat(T::Index::truncate(*j)),
                );
            }
            reduce_2d(&simd, best_val, best_row, best_col)
        }
    }

    T::Arch::default().dispatch(Impl {
        data,
        lhs,
        rhs,
        align,
    })
}

#[math]
fn best_in_matrix_fallback<'M, 'N, T: ComplexField>(
    data: MatRef<'_, T, Dim<'M>, Dim<'N>>,
) -> (usize, usize, Real<T>) {
    let mut max = zero();
    let mut row = 0;
    let mut col = 0;

    let (M, N) = data.shape();

    for j in N.indices() {
        for i in M.indices() {
            let abs = abs1(data[(i, j)]);
            if abs > max {
                row = *i;
                col = *j;
                max = abs;
            }
        }
    }

    (row, col, max)
}

#[math]
fn best_in_matrix<'M, 'N, T: ComplexField>(
    data: MatRef<'_, T, Dim<'M>, Dim<'N>>,
) -> (usize, usize, Real<T>) {
    if const { T::SIMD_CAPABILITIES.is_simd() } {
        if let Some(dst) = data.try_as_col_major() {
            best_in_mat_simd(dst)
        } else {
            best_in_matrix_fallback(data)
        }
    } else {
        best_in_matrix_fallback(data)
    }
}
#[math]
fn rank_one_update_and_best_in_matrix<'M, 'N, T: ComplexField>(
    mut dst: MatMut<'_, T, Dim<'M>, Dim<'N>>,
    lhs: ColRef<'_, T, Dim<'M>>,
    rhs: RowRef<'_, T, Dim<'N>>,
    align: usize,
) -> (usize, usize, Real<T>) {
    if const { T::SIMD_CAPABILITIES.is_simd() } {
        if let (Some(dst), Some(lhs)) =
            (dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major())
        {
            update_and_best_in_mat_simd(dst, lhs, rhs, align)
        } else {
            matmul(
                dst.rb_mut(),
                Accum::Add,
                lhs.as_mat(),
                rhs.as_mat(),
                -one(),
                Par::Seq,
            );
            best_in_matrix(dst.rb())
        }
    } else {
        matmul(
            dst.rb_mut(),
            Accum::Add,
            lhs.as_mat(),
            rhs.as_mat(),
            -one(),
            Par::Seq,
        );
        best_in_matrix(dst.rb())
    }
}

#[math]
fn lu_in_place_unblocked<'M, 'N, T: ComplexField>(
    A: MatMut<'_, T, Dim<'M>, Dim<'N>>,
    row_trans: &mut [usize],
    col_trans: &mut [usize],
    par: Par,
    transpose: bool,
    params: FullPivLuParams,
) -> usize {
    let mut n_trans = 0;

    let (M, N) = A.shape();
    if *M == 0 || *N == 0 {
        return 0;
    }

    let mut par = par;

    let mut A = A;
    let (mut max_row, mut max_col, mut max_score) = best_in_matrix(A.rb());

    for (ki, kj) in core::iter::zip(M.indices(), N.indices()) {
        if max_score < min_positive() {
            for (k, (row, col)) in
                core::iter::zip(&mut row_trans[*ki..], &mut col_trans[*kj..]).enumerate()
            {
                *row = k + *ki;
                *col = k + *kj;
            }
            break;
        }

        row_trans[*ki] = max_row;
        col_trans[*kj] = max_col;

        if max_row != *ki {
            let row = M.idx(max_row);
            swap_rows_idx(A.rb_mut(), ki, row);
            n_trans += 1;
        }
        if max_col != *kj {
            let col = N.idx(max_col);
            swap_cols_idx(A.rb_mut(), kj, col);
            n_trans += 1;
        }

        let inv = recip(A[(ki, kj)]);
        if transpose {
            for j in kj.next().to(N.end()) {
                A[(ki, j)] = A[(ki, j)] * inv;
            }
        } else {
            for i in ki.next().to(M.end()) {
                A[(i, kj)] = A[(i, kj)] * inv;
            }
        }

        if ki.next() == M.end() || kj.next() == N.end() {
            break;
        }

        if (*M - *ki.next()) * (*N - *kj.next()) < params.par_threshold {
            par = Par::Seq;
        }

        ghost_tree!(ROWS(TOP, BOT), COLS(LEFT, RIGHT), {
            let (l![top, bot], (disjoint_rows, ..)) = M.split(l![ki, ..], ROWS);
            let (l![left, right], (disjoint_cols, ..)) = N.split(l![kj, ..], COLS);

            let l![A0, A1] = A.rb_mut().row_segments_mut(l![top, bot], disjoint_rows);
            let l![_, A01] = A0.col_segments_mut(l![left, right], disjoint_cols);
            let l![A10, mut A11] = A1.col_segments_mut(l![left, right], disjoint_cols);

            let lhs = A10.rb();
            let rhs = A01.rb();

            match par {
                Par::Seq => {
                    (max_row, max_col, max_score) = rank_one_update_and_best_in_matrix(
                        A11.rb_mut(),
                        lhs,
                        rhs,
                        simd_align(*ki.next()),
                    );
                }
                #[cfg(feature = "rayon")]
                Par::Rayon(nthreads) => {
                    use rayon::prelude::*;
                    let nthreads = nthreads.get();

                    let mut best = core::iter::repeat_with(|| (0, 0, zero()))
                        .take(nthreads)
                        .collect::<Vec<_>>();
                    let full_cols = *A11.ncols();

                    best.par_iter_mut()
                        .zip_eq(A11.rb_mut().par_col_partition_mut(nthreads))
                        .zip_eq(rhs.transpose().par_partition(nthreads))
                        .enumerate()
                        .for_each(|(idx, (((max_row, max_col, max_score), A11), rhs))| {
                            with_dim!(N, A11.ncols());

                            (*max_row, *max_col, *max_score) = {
                                let (a, mut b, c) = rank_one_update_and_best_in_matrix(
                                    A11.as_col_shape_mut(N),
                                    lhs,
                                    rhs.transpose().as_col_shape(N),
                                    simd_align(*ki.next()),
                                );
                                b += par_split_indices(full_cols, idx, nthreads).0;
                                (a, b, c)
                            };
                        });

                    max_row = 0;
                    max_col = 0;
                    max_score = zero();

                    for (row, col, val) in best {
                        if val > max_score {
                            max_row = row;
                            max_col = col;
                            max_score = val;
                        }
                    }
                }
            }
        });

        max_row += *ki.next();
        max_col += *kj.next();
    }

    n_trans
}

/// LU factorization tuning parameters.
#[derive(Copy, Clone)]
#[non_exhaustive]
pub struct FullPivLuParams {
    /// At which size the parallelism should be disabled. `None` to automatically determine this
    /// threshold.
    pub par_threshold: usize,
}

impl Default for FullPivLuParams {
    #[inline]
    fn default() -> Self {
        Self {
            par_threshold: 256 * 512,
        }
    }
}

#[inline]
pub fn lu_in_place_scratch<I: Index, T: ComplexField>(
    nrows: usize,
    ncols: usize,
    par: Par,
    params: FullPivLuParams,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    _ = params;
    let size = Ord::min(nrows, ncols);
    StackReq::try_new::<usize>(size)?.try_array(2)
}

#[derive(Copy, Clone, Debug)]
pub struct FullPivLuInfo {
    pub transposition_count: usize,
}

pub fn lu_in_place<'out, 'M, 'N, I: Index, T: ComplexField>(
    mat: MatMut<'_, T, Dim<'M>, Dim<'N>>,
    row_perm: &'out mut Array<'M, I>,
    row_perm_inv: &'out mut Array<'M, I>,
    col_perm: &'out mut Array<'N, I>,
    col_perm_inv: &'out mut Array<'N, I>,
    par: Par,
    stack: &mut DynStack,
    params: FullPivLuParams,
) -> (
    FullPivLuInfo,
    PermRef<'out, I, Dim<'M>>,
    PermRef<'out, I, Dim<'N>>,
) {
    #[cfg(feature = "perf-warn")]
    if (mat.col_stride().unsigned_abs() == 1 || mat.row_stride().unsigned_abs() != 1)
        && crate::__perf_warn!(LU_WARN)
    {
        log::warn!(target: "faer_perf", "LU with full pivoting prefers column-major or row-major matrix. Found matrix with generic strides.");
    }

    let (M, N) = mat.shape();

    let size = Ord::min(*M, *N);

    let (mut row_transpositions, stack) = stack.make_with(size, |_| 0);
    let row_transpositions = row_transpositions.as_mut();
    let (mut col_transpositions, _) = stack.make_with(size, |_| 0);
    let col_transpositions = col_transpositions.as_mut();

    let n_transpositions = if mat.row_stride().abs() < mat.col_stride().abs() {
        lu_in_place_unblocked(
            mat,
            row_transpositions,
            col_transpositions,
            par,
            false,
            params,
        )
    } else {
        lu_in_place_unblocked(
            mat.transpose_mut(),
            col_transpositions,
            row_transpositions,
            par,
            true,
            params,
        )
    };

    for i in M.indices() {
        row_perm[i] = I::truncate(*i);
    }
    for (i, t) in row_transpositions.iter().copied().enumerate() {
        row_perm.as_mut().swap(i, t);
    }
    for i in M.indices() {
        row_perm_inv[M.idx(row_perm[i].zx())] = I::truncate(*i);
    }

    for j in N.indices() {
        col_perm[j] = I::truncate(*j);
    }
    for (i, t) in col_transpositions.iter().copied().enumerate() {
        col_perm.as_mut().swap(i, t);
    }
    for j in N.indices() {
        col_perm_inv[N.idx(col_perm[j].zx())] = I::truncate(*j);
    }

    unsafe {
        (
            FullPivLuInfo {
                transposition_count: n_transpositions,
            },
            PermRef::new_unchecked(
                Idx::from_slice_ref_unchecked(row_perm.as_ref()),
                Idx::from_slice_ref_unchecked(row_perm_inv.as_ref()),
                M,
            ),
            PermRef::new_unchecked(
                Idx::from_slice_ref_unchecked(col_perm.as_ref()),
                Idx::from_slice_ref_unchecked(col_perm_inv.as_ref()),
                N,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, c64, stats::prelude::*, utils::approx::*, Mat};
    use dyn_stack::GlobalMemBuffer;

    #[test]
    fn test_flu() {
        let rng = &mut StdRng::seed_from_u64(0);

        for par in [Par::Seq, Par::rayon(8)] {
            for m in [8, 16, 24, 32, 128, 255, 256, 257] {
                with_dim!(M, m);
                with_dim!(N, 8);

                let approx_eq = CwiseMat(ApproxEq {
                    abs_tol: 1e-10,
                    rel_tol: 1e-10,
                });

                let A = CwiseMatDistribution {
                    nrows: M,
                    ncols: N,
                    dist: ComplexDistribution::new(StandardNormal, StandardNormal),
                }
                .rand::<Mat<c64, Dim, Dim>>(rng);
                let A = A.as_ref();

                let mut LU = A.cloned();
                let row_perm = &mut *vec![0usize; *M];
                let row_perm_inv = &mut *vec![0usize; *M];
                let row_perm = Array::from_mut(row_perm, M);
                let row_perm_inv = Array::from_mut(row_perm_inv, M);

                let col_perm = &mut *vec![0usize; *N];
                let col_perm_inv = &mut *vec![0usize; *N];
                let col_perm = Array::from_mut(col_perm, N);
                let col_perm_inv = Array::from_mut(col_perm_inv, N);

                let (_, p, q) = lu_in_place(
                    LU.as_mut(),
                    row_perm,
                    row_perm_inv,
                    col_perm,
                    col_perm_inv,
                    par,
                    DynStack::new(&mut GlobalMemBuffer::new(
                        lu_in_place_scratch::<usize, c64>(*N, *N, par, default()).unwrap(),
                    )),
                    Default::default(),
                );

                let mut L = LU.as_ref().cloned();
                let mut U = LU.as_ref().cloned();

                for j in N.indices() {
                    let i = M.check(*j);
                    for i in IdxInc::ZERO.to(i.excl()) {
                        if *i >= *j {
                            break;
                        }
                        L[(i, j)] = c64::ZERO;
                    }
                    L[(i, j)] = c64::ONE;
                }
                for j in N.indices() {
                    let i = M.check(*j);
                    for i in i.next().to(M.end()) {
                        U[(i, j)] = c64::ZERO;
                    }
                }
                let L = L.as_ref();
                let U = U.as_ref();

                let U = U.subrows(IdxInc::ZERO, N);

                assert!(p.inverse() * L * U * q ~ A);
            }
            // for n in [16, 24, 32, 128, 255, 256, 257] {
            //     with_dim!(N, n);
            //     let approx_eq = CwiseMat(ApproxEq {
            //         abs_tol: 1e-10,
            //         rel_tol: 1e-10,
            //     });

            //     let A = CwiseMatDistribution {
            //         nrows: N,
            //         ncols: N,
            //         dist: StandardNormal,
            //     }
            //     .rand::<Mat<f64, Dim, Dim>>(rng);
            //     let A = A.as_ref();

            //     let mut LU = A.cloned();
            //     let row_perm = &mut *vec![0usize; n];
            //     let row_perm_inv = &mut *vec![0usize; n];
            //     let row_perm = Array::from_mut(row_perm, N);
            //     let row_perm_inv = Array::from_mut(row_perm_inv, N);

            //     let col_perm = &mut *vec![0usize; n];
            //     let col_perm_inv = &mut *vec![0usize; n];
            //     let col_perm = Array::from_mut(col_perm, N);
            //     let col_perm_inv = Array::from_mut(col_perm_inv, N);

            //     let (_, p, q) = lu_in_place(
            //         LU.as_mut(),
            //         row_perm,
            //         row_perm_inv,
            //         col_perm,
            //         col_perm_inv,
            //         par,
            //         DynStack::new(&mut GlobalMemBuffer::new(
            //             lu_in_place_scratch::<usize, f64>(*N, *N, par, default()).unwrap(),
            //         )),
            //         Default::default(),
            //     );

            //     let mut L = LU.as_ref().cloned();
            //     let mut U = LU.as_ref().cloned();

            //     for j in N.indices() {
            //         for i in IdxInc::ZERO.to(j.excl()) {
            //             L[(i, j)] = 0.0;
            //         }
            //         L[(j, j)] = 1.0;
            //     }
            //     for j in N.indices() {
            //         for i in j.next().to(N.end()) {
            //             U[(i, j)] = 0.0;
            //         }
            //     }
            //     let L = L.as_ref();
            //     let U = U.as_ref();

            //     assert!(p.inverse() * L * U * q ~ A);
            // }
        }
    }
}
