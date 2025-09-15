use faer_traits::{Real, RealReg};
use linalg::matmul::matmul;
use pulp::Simd;

use crate::internal_prelude::*;
use crate::perm::{swap_cols_idx, swap_rows_idx};
use crate::utils::thread::par_split_indices;

#[inline(always)]
fn best_value<T: ComplexField, S: Simd>(
	simd: &SimdCtx<T, S>,
	best_value: RealReg<T::SimdVec<S>>,
	best_indices: T::SimdIndex<S>,
	value: T::SimdVec<S>,
	indices: T::SimdIndex<S>,
) -> (RealReg<T::SimdVec<S>>, T::SimdIndex<S>) {
	let value = simd.abs1(value);
	let is_better = (**simd).gt(value, best_value);
	(
		RealReg(simd.select(is_better, value.0, best_value.0)),
		simd.iselect(is_better, indices, best_indices),
	)
}

#[inline(always)]
fn best_score<T: ComplexField, S: Simd>(
	simd: &SimdCtx<T, S>,
	best_score: RealReg<T::SimdVec<S>>,
	best_indices: T::SimdIndex<S>,
	score: RealReg<T::SimdVec<S>>,
	indices: T::SimdIndex<S>,
) -> (RealReg<T::SimdVec<S>>, T::SimdIndex<S>) {
	let is_better = (**simd).gt(score, best_score);
	(
		RealReg(simd.select(is_better, score.0, best_score.0)),
		simd.iselect(is_better, indices, best_indices),
	)
}

#[inline(always)]
fn best_score_2d<T: ComplexField, S: Simd>(
	simd: &SimdCtx<T, S>,
	best_score: RealReg<T::SimdVec<S>>,
	best_row: T::SimdIndex<S>,
	best_col: T::SimdIndex<S>,
	score: RealReg<T::SimdVec<S>>,
	row: T::SimdIndex<S>,
	col: T::SimdIndex<S>,
) -> (RealReg<T::SimdVec<S>>, T::SimdIndex<S>, T::SimdIndex<S>) {
	let is_better = (**simd).gt(score, best_score);
	(
		RealReg(simd.select(is_better, score.0, best_score.0)),
		simd.iselect(is_better, row, best_row),
		simd.iselect(is_better, col, best_col),
	)
}

#[inline(always)]
#[math]
fn reduce_2d<T: ComplexField, S: Simd>(
	simd: &SimdCtx<T, S>,
	best_values: RealReg<T::SimdVec<S>>,
	best_row: T::SimdIndex<S>,
	best_col: T::SimdIndex<S>,
) -> (usize, usize, Real<T>) {
	let best_val = simd.reduce_max_real(best_values);

	let best_val_splat = simd.splat_real(&best_val);
	let is_best = (**simd).ge(best_values, best_val_splat);
	let idx = simd.first_true_mask(is_best);

	let best_row = bytemuck::cast_slice::<T::SimdIndex<S>, T::Index>(core::slice::from_ref(&best_row))[idx];
	let best_col = bytemuck::cast_slice::<T::SimdIndex<S>, T::Index>(core::slice::from_ref(&best_col))[idx];

	(best_row.zx(), best_col.zx(), best_val)
}

#[inline(always)]
#[math]
fn best_in_col_simd<'M, T: ComplexField, S: Simd>(
	simd: SimdCtx<'M, T, S>,
	data: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
) -> (RealReg<T::SimdVec<S>>, T::SimdIndex<S>) {
	let (head, body4, body1, tail) = simd.batch_indices::<4>();

	let iota = T::simd_iota(&simd.0);
	let lane_count = core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>();

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

	let mut idx0 = simd.iadd(iota, simd.isplat(T::Index::truncate(simd.offset().wrapping_neg())));

	if let Some(i0) = head {
		(best_val0, best_idx0) = best_value(&simd, best_val0, best_idx0, simd.read(data, i0), idx0);
		idx0 = simd.iadd(idx0, inc1);
	}

	let mut idx1 = simd.iadd(idx0, inc1);
	let mut idx2 = simd.iadd(idx1, inc1);
	let mut idx3 = simd.iadd(idx2, inc1);
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
) -> (RealReg<T::SimdVec<S>>, T::SimdIndex<S>) {
	let mut data = data;

	let (head, body4, body1, tail) = simd.batch_indices::<3>();

	let iota = T::simd_iota(&simd.0);
	let lane_count = core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>();

	let inc1 = simd.isplat(T::Index::truncate(lane_count));
	let inc3 = simd.isplat(T::Index::truncate(3 * lane_count));

	let mut best_val0 = simd.splat_real(&zero());
	let mut best_val1 = simd.splat_real(&zero());
	let mut best_val2 = simd.splat_real(&zero());

	let mut best_idx0 = simd.isplat(T::Index::truncate(0));
	let mut best_idx1 = simd.isplat(T::Index::truncate(0));
	let mut best_idx2 = simd.isplat(T::Index::truncate(0));

	let mut idx0 = simd.iadd(iota, simd.isplat(T::Index::truncate(simd.offset().wrapping_neg())));
	let rhs = simd.splat(&-rhs);

	if let Some(i0) = head {
		let mut x0 = simd.read(data.rb(), i0);
		let l0 = simd.read(lhs, i0);
		x0 = simd.mul_add(l0, rhs, x0);

		(best_val0, best_idx0) = best_value(&simd, best_val0, best_idx0, x0, idx0);
		idx0 = simd.iadd(idx0, inc1);

		simd.write(data.rb_mut(), i0, x0);
	}

	let mut idx1 = simd.iadd(idx0, inc1);
	let mut idx2 = simd.iadd(idx1, inc1);
	for [i0, i1, i2] in body4 {
		let mut x0 = simd.read(data.rb(), i0);
		let l0 = simd.read(lhs, i0);
		x0 = simd.mul_add(l0, rhs, x0);
		(best_val0, best_idx0) = best_value(&simd, best_val0, best_idx0, x0, idx0);
		simd.write(data.rb_mut(), i0, x0);

		let mut x1 = simd.read(data.rb(), i1);
		let l1 = simd.read(lhs, i1);
		x1 = simd.mul_add(l1, rhs, x1);
		(best_val1, best_idx1) = best_value(&simd, best_val1, best_idx1, x1, idx1);
		simd.write(data.rb_mut(), i1, x1);

		let mut x2 = simd.read(data.rb(), i2);
		let l2 = simd.read(lhs, i2);
		x2 = simd.mul_add(l2, rhs, x2);
		(best_val2, best_idx2) = best_value(&simd, best_val2, best_idx2, x2, idx2);
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
fn best_in_mat_simd<T: ComplexField>(data: MatRef<'_, T, usize, usize, ContiguousFwd>) -> (usize, usize, Real<T>) {
	struct Impl<'a, 'M, 'N, T: ComplexField> {
		data: MatRef<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
	}

	impl<'a, 'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'N, T> {
		type Output = (usize, usize, Real<T>);

		#[math]
		#[inline(always)]
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

	with_dim!(M, data.nrows());
	with_dim!(N, data.ncols());
	dispatch!(Impl { data: data.as_shape(M, N) }, Impl, T)
}

#[inline(always)]
fn update_and_best_in_mat_simd<T: ComplexField>(
	data: MatMut<'_, T, usize, usize, ContiguousFwd>,
	lhs: ColRef<'_, T, usize, ContiguousFwd>,
	rhs: RowRef<'_, T, usize>,
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
		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let Self { data, lhs, rhs, align } = self;

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

	with_dim!(M, data.nrows());
	with_dim!(N, data.ncols());
	dispatch!(
		Impl {
			data: data.as_shape_mut(M, N),
			lhs: lhs.as_row_shape(M),
			rhs: rhs.as_col_shape(N),
			align,
		},
		Impl,
		T
	)
}

#[math]
fn best_in_matrix_fallback<T: ComplexField>(data: MatRef<'_, T>) -> (usize, usize, Real<T>) {
	let mut max = zero();
	let mut row = 0;
	let mut col = 0;

	let (m, n) = data.shape();

	for j in 0..n {
		for i in 0..m {
			let abs = abs1(data[(i, j)]);
			if abs > max {
				row = i;
				col = j;
				max = abs;
			}
		}
	}

	(row, col, max)
}

#[math]
fn best_in_matrix<T: ComplexField>(data: MatRef<'_, T>) -> (usize, usize, Real<T>) {
	if try_const! { T::SIMD_CAPABILITIES.is_simd() } {
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
fn rank_one_update_and_best_in_matrix<T: ComplexField>(
	mut dst: MatMut<'_, T>,
	lhs: ColRef<'_, T>,
	rhs: RowRef<'_, T>,
	align: usize,
) -> (usize, usize, Real<T>) {
	if try_const! { T::SIMD_CAPABILITIES.is_simd() } {
		if let (Some(dst), Some(lhs)) = (dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major()) {
			update_and_best_in_mat_simd(dst, lhs, rhs, align)
		} else {
			matmul(dst.rb_mut(), Accum::Add, lhs.as_mat(), rhs.as_mat(), -one::<T>(), Par::Seq);
			best_in_matrix(dst.rb())
		}
	} else {
		matmul(dst.rb_mut(), Accum::Add, lhs.as_mat(), rhs.as_mat(), -one::<T>(), Par::Seq);
		best_in_matrix(dst.rb())
	}
}

#[math]
fn lu_in_place_unblocked<T: ComplexField>(
	A: MatMut<'_, T>,
	row_trans: &mut [usize],
	col_trans: &mut [usize],
	par: Par,
	transpose: bool,
	params: Spec<FullPivLuParams, T>,
) -> usize {
	let params = params.config;
	let mut n_trans = 0;

	let (m, n) = A.shape();
	if m == 0 || n == 0 {
		return 0;
	}

	let mut par = par;

	let mut A = A;
	let (mut max_row, mut max_col, mut max_score) = best_in_matrix(A.rb());

	for k in 0..Ord::min(m, n) {
		if max_score < min_positive() {
			for (i, (row, col)) in core::iter::zip(&mut row_trans[k..], &mut col_trans[k..]).enumerate() {
				*row = i + k;
				*col = i + k;
			}
			break;
		}

		row_trans[k] = max_row;
		col_trans[k] = max_col;

		if max_row != k {
			swap_rows_idx(A.rb_mut(), k, max_row);
			n_trans += 1;
		}
		if max_col != k {
			swap_cols_idx(A.rb_mut(), k, max_col);
			n_trans += 1;
		}

		let inv = recip(A[(k, k)]);
		if transpose {
			for j in k + 1..n {
				A[(k, j)] = A[(k, j)] * inv;
			}
		} else {
			for i in k + 1..m {
				A[(i, k)] = A[(i, k)] * inv;
			}
		}

		if k + 1 == Ord::min(m, n) {
			break;
		}
		if (m - k - 1) * (n - k - 1) < params.par_threshold {
			par = Par::Seq;
		}

		let (_, A01, A10, mut A11) = A.rb_mut().split_at_mut(k + 1, k + 1);

		let lhs = A10.col(k);
		let rhs = A01.row(k);

		match par {
			Par::Seq => {
				(max_row, max_col, max_score) = rank_one_update_and_best_in_matrix(A11.rb_mut(), lhs, rhs, simd_align(k + 1));
			},
			#[cfg(feature = "rayon")]
			Par::Rayon(nthreads) => {
				use rayon::prelude::*;
				let nthreads = nthreads.get();

				let mut best = core::iter::repeat_with(|| (0, 0, zero())).take(nthreads).collect::<alloc::vec::Vec<_>>();
				let full_cols = A11.ncols();

				best.par_iter_mut()
					.zip_eq(A11.rb_mut().par_col_partition_mut(nthreads))
					.zip_eq(rhs.par_partition(nthreads))
					.enumerate()
					.for_each(|(idx, (((max_row, max_col, max_score), A11), rhs))| {
						(*max_row, *max_col, *max_score) = {
							let (a, mut b, c) = rank_one_update_and_best_in_matrix(A11, lhs, rhs, simd_align(k + 1));
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
			},
		}

		max_row += k + 1;
		max_col += k + 1;
	}

	n_trans
}

/// $LU$ factorization tuning parameters
#[derive(Copy, Clone, Debug)]
pub struct FullPivLuParams {
	/// threshold at which size parallelism should be disabled
	pub par_threshold: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for FullPivLuParams {
	#[inline]
	fn auto() -> Self {
		Self {
			par_threshold: 256 * 512,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

#[inline]
pub fn lu_in_place_scratch<I: Index, T: ComplexField>(nrows: usize, ncols: usize, par: Par, params: Spec<FullPivLuParams, T>) -> StackReq {
	_ = par;
	_ = params;
	let size = Ord::min(nrows, ncols);
	StackReq::new::<usize>(size).array(2)
}

#[derive(Copy, Clone, Debug)]
pub struct FullPivLuInfo {
	pub transposition_count: usize,
}

pub fn lu_in_place<'out, I: Index, T: ComplexField>(
	mat: MatMut<'_, T>,
	row_perm: &'out mut [I],
	row_perm_inv: &'out mut [I],
	col_perm: &'out mut [I],
	col_perm_inv: &'out mut [I],
	par: Par,
	stack: &mut MemStack,
	params: Spec<FullPivLuParams, T>,
) -> (FullPivLuInfo, PermRef<'out, I>, PermRef<'out, I>) {
	#[cfg(feature = "perf-warn")]
	if (mat.col_stride().unsigned_abs() == 1 || mat.row_stride().unsigned_abs() != 1) && crate::__perf_warn!(LU_WARN) {
		log::warn!(target: "faer_perf", "LU with full pivoting prefers column-major or row-major matrix. Found matrix with generic strides.");
	}

	let (M, N) = mat.shape();

	let size = Ord::min(M, N);

	let (mut row_transpositions, stack) = stack.make_with(size, |_| 0);
	let row_transpositions = row_transpositions.as_mut();
	let (mut col_transpositions, _) = stack.make_with(size, |_| 0);
	let col_transpositions = col_transpositions.as_mut();

	let n_transpositions = if mat.row_stride().abs() < mat.col_stride().abs() {
		lu_in_place_unblocked(mat, row_transpositions, col_transpositions, par, false, params)
	} else {
		lu_in_place_unblocked(mat.transpose_mut(), col_transpositions, row_transpositions, par, true, params)
	};

	for i in 0..M {
		row_perm[i] = I::truncate(i);
	}
	for (i, t) in row_transpositions.iter().copied().enumerate() {
		row_perm.as_mut().swap(i, t);
	}
	for i in 0..M {
		row_perm_inv[row_perm[i].zx()] = I::truncate(i);
	}

	for j in 0..N {
		col_perm[j] = I::truncate(j);
	}
	for (i, t) in col_transpositions.iter().copied().enumerate() {
		col_perm.as_mut().swap(i, t);
	}
	for j in 0..N {
		col_perm_inv[col_perm[j].zx()] = I::truncate(j);
	}

	unsafe {
		(
			FullPivLuInfo {
				transposition_count: n_transpositions,
			},
			PermRef::new_unchecked(row_perm, row_perm_inv, M),
			PermRef::new_unchecked(col_perm, col_perm_inv, N),
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, assert, c64};
	use dyn_stack::MemBuffer;

	#[test]
	fn test_flu() {
		let rng = &mut StdRng::seed_from_u64(0);

		for par in [Par::Seq, Par::rayon(8)] {
			for m in [8, 16, 24, 32, 128, 255, 256, 257] {
				let n = 8;

				let approx_eq = CwiseMat(ApproxEq {
					abs_tol: 1e-10,
					rel_tol: 1e-10,
				});

				let A = CwiseMatDistribution {
					nrows: m,
					ncols: n,
					dist: ComplexDistribution::new(StandardNormal, StandardNormal),
				}
				.rand::<Mat<c64>>(rng);
				let A = A.as_ref();

				let mut LU = A.cloned();
				let row_perm = &mut *vec![0usize; m];
				let row_perm_inv = &mut *vec![0usize; m];

				let col_perm = &mut *vec![0usize; n];
				let col_perm_inv = &mut *vec![0usize; n];

				let (_, p, q) = lu_in_place(
					LU.as_mut(),
					row_perm,
					row_perm_inv,
					col_perm,
					col_perm_inv,
					par,
					MemStack::new(&mut MemBuffer::new(lu_in_place_scratch::<usize, c64>(n, n, par, default()))),
					default(),
				);

				let mut L = LU.as_ref().cloned();
				let mut U = LU.as_ref().cloned();

				for j in 0..n {
					for i in 0..j {
						L[(i, j)] = c64::ZERO;
					}
					L[(j, j)] = c64::ONE;
				}
				for j in 0..n {
					for i in j + 1..m {
						U[(i, j)] = c64::ZERO;
					}
				}
				let L = L.as_ref();
				let U = U.as_ref();

				let U = U.subrows(0, n);

				assert!(p.inverse() * L * U * q ~ A);
			}

			for n in [16, 24, 32, 128, 255, 256, 257] {
				let approx_eq = CwiseMat(ApproxEq {
					abs_tol: 1e-10,
					rel_tol: 1e-10,
				});

				let A = CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: StandardNormal,
				}
				.rand::<Mat<f64>>(rng);
				let A = A.as_ref();

				let mut LU = A.cloned();
				let row_perm = &mut *vec![0usize; n];
				let row_perm_inv = &mut *vec![0usize; n];

				let col_perm = &mut *vec![0usize; n];
				let col_perm_inv = &mut *vec![0usize; n];

				let (_, p, q) = lu_in_place(
					LU.as_mut(),
					row_perm,
					row_perm_inv,
					col_perm,
					col_perm_inv,
					par,
					MemStack::new(&mut MemBuffer::new(lu_in_place_scratch::<usize, f64>(n, n, par, default()))),
					default(),
				);

				let mut L = LU.as_ref().cloned();
				let mut U = LU.as_ref().cloned();

				for j in 0..n {
					for i in 0..j {
						L[(i, j)] = 0.0;
					}
					L[(j, j)] = 1.0;
				}
				for j in 0..n {
					for i in j + 1..n {
						U[(i, j)] = 0.0;
					}
				}
				let L = L.as_ref();
				let U = U.as_ref();

				assert!(p.inverse() * L * U * q ~ A);
			}
		}
	}

	#[test]
	fn test_gh238() {
		#[rustfmt::skip]
		let A = [
			[-2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
			[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		];

		let A = MatRef::from_row_major_array(&A).to_owned();

		let b = [
			-0.0, -0.0, -2.0, -2.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
		];

		let b = ColRef::from_slice(&b);

		let lu = A.full_piv_lu();
		assert!(lu.solve(b).is_all_finite());
	}
}
