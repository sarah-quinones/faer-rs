use crate::assert;
use crate::internal_prelude_sp::*;
use core::cell::UnsafeCell;

/// info about the matrix multiplication operation to help split the workload between multiple
/// threads
pub struct SparseMatMulInfo {
	flops_prefix_sum: alloc::vec::Vec<f64>,
}

/// performs a symbolic matrix multiplication of a sparse matrix `lhs` by a sparse matrix `rhs`,
/// and returns the result.
///
/// # note
/// allows unsorted matrices, and produces a sorted output.
#[track_caller]
pub fn sparse_sparse_matmul_symbolic<I: Index>(
	lhs: SymbolicSparseColMatRef<'_, I>,
	rhs: SymbolicSparseColMatRef<'_, I>,
) -> Result<(SymbolicSparseColMat<I>, SparseMatMulInfo), FaerError> {
	assert!(lhs.ncols() == rhs.nrows());

	let m = lhs.nrows();
	let n = rhs.ncols();

	let mut col_ptr = try_zeroed::<I>(n + 1)?;
	let mut row_idx = alloc::vec::Vec::new();
	let mut work = try_collect(repeat_n!(I::truncate(usize::MAX), m))?;
	let mut info = try_zeroed::<f64>(n + 1)?;

	for j in 0..n {
		let mut count = 0usize;
		let mut flops = 0.0f64;
		for k in rhs.row_idx_of_col(j) {
			for i in lhs.row_idx_of_col(k) {
				if work[i] != I::truncate(j) {
					row_idx.try_reserve(1).ok().ok_or(FaerError::OutOfMemory)?;
					row_idx.push(I::truncate(i));
					work[i] = I::truncate(j);

					count += 1;
				}
			}
			flops += lhs.row_idx_of_col_raw(k).len() as f64;
		}

		info[j + 1] = info[j] + flops;
		col_ptr[j + 1] = col_ptr[j] + I::truncate(count);
		if col_ptr[j + 1] > I::from_signed(I::Signed::MAX) {
			return Err(FaerError::IndexOverflow);
		}
		row_idx[col_ptr[j].zx()..col_ptr[j + 1].zx()].sort_unstable();
	}

	unsafe {
		Ok((
			SymbolicSparseColMat::new_unchecked(m, n, col_ptr, None, row_idx),
			SparseMatMulInfo { flops_prefix_sum: info },
		))
	}
}

/// computes the layout of the workspace required to perform the numeric matrix
/// multiplication into `dst`.
pub fn sparse_sparse_matmul_numeric_scratch<I: Index, T: ComplexField>(dst: SymbolicSparseColMatRef<'_, I>, par: Par) -> StackReq {
	temp_mat_scratch::<T>(dst.nrows(), par.degree())
}

/// performs a numeric matrix multiplication of a sparse matrix `lhs` by a sparse matrix `rhs`
/// multiplied by `alpha`, and stores or adds the result to `dst`.
///
/// # note
/// `lhs` and `rhs` are allowed to be unsorted matrices.
#[track_caller]
#[math]
pub fn sparse_sparse_matmul_numeric<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(
	dst: SparseColMatMut<'_, I, T>,
	beta: Accum,
	lhs: SparseColMatRef<'_, I, LhsT>,
	rhs: SparseColMatRef<'_, I, RhsT>,
	alpha: T,
	info: &SparseMatMulInfo,
	par: Par,
	stack: &mut MemStack,
) {
	assert!(all(dst.nrows() == lhs.nrows(), dst.ncols() == rhs.ncols(), lhs.ncols() == rhs.nrows()));
	let m = lhs.nrows();
	let n = rhs.ncols();
	let mut dst = dst;
	if let Accum::Replace = beta {
		for j in 0..n {
			dst.rb_mut().val_of_col_mut(j).fill(zero());
		}
	}
	let alpha = &alpha;

	let (c_symbolic, c_values) = dst.parts_mut();

	let total_flop_count = info.flops_prefix_sum[n];

	let (mut work, _) = temp_mat_zeroed::<T, _, _>(m, par.degree(), stack);
	let work = work.as_mat_mut();
	let work = work.rb();

	#[derive(Copy, Clone)]
	struct SyncWrapper<T>(T);
	unsafe impl<T> Sync for SyncWrapper<T> {}
	unsafe impl<T> Send for SyncWrapper<T> {}

	let c_values = SyncWrapper(&*UnsafeCell::from_mut(c_values));

	let nthreads = par.degree();
	let job = &|tid: usize| {
		assert!(tid < nthreads);

		fn partition_fn(total_flop_count: f64, nthreads: usize, tid: usize) -> impl FnMut(&f64) -> bool {
			move |&x| x < total_flop_count * (tid as f64 / nthreads as f64)
		}

		let mut work = unsafe { work.col(tid).const_cast().try_as_col_major_mut().unwrap() };
		let col_start = info.flops_prefix_sum.partition_point(partition_fn(total_flop_count, nthreads, tid));
		let col_end = col_start + info.flops_prefix_sum[col_start..].partition_point(partition_fn(total_flop_count, nthreads, tid + 1));

		// SAFETY: UnsafeCell<[T]> ~ [T] ~ [UnsafeCell<T>]
		let c_values = unsafe { &*({ c_values }.0 as *const UnsafeCell<[T]> as *const [UnsafeCell<T>]) };

		for j in col_start..col_end {
			for (k, b_k) in iter::zip(rhs.row_idx_of_col(j), rhs.val_of_col(j)) {
				let b_k = Conj::apply(b_k) * *alpha;

				for (i, a_i) in iter::zip(lhs.row_idx_of_col(k), lhs.val_of_col(k)) {
					let a_i = Conj::apply(a_i);
					work[i] = work[i] + a_i * b_k;
				}
			}
			// SAFETY: UnsafeCell<[T]> ~ [T] ~ [UnsafeCell<T>]
			// and only thread `tid` has access to the range of column `j`
			// since `col_start..col_end` denote disjoint ranges for each `tid`
			let c_values =
				unsafe { &mut *UnsafeCell::raw_get((&c_values[c_symbolic.col_range(j)]) as *const [UnsafeCell<T>] as *const UnsafeCell<[T]>) };

			for (i, c_i) in iter::zip(c_symbolic.row_idx_of_col(j), c_values) {
				*c_i = *c_i + work[i];
				work[i] = zero();
			}
		}
	};

	match par {
		Par::Seq => {
			job(0);
		},
		#[cfg(feature = "rayon")]
		Par::Rayon(nthreads) => {
			use rayon::prelude::*;

			(0..nthreads.get()).into_par_iter().for_each(|tid| {
				job(tid);
			});
		},
	}
}

/// performs a numeric matrix multiplication of a sparse matrix `lhs` by a sparse matrix `rhs`
/// multiplied by `alpha`, and returns the result.
///
/// # note
/// `lhs` and `rhs` are allowed to be unsorted matrices.
#[track_caller]
pub fn sparse_sparse_matmul<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(
	lhs: SparseColMatRef<'_, I, LhsT>,
	rhs: SparseColMatRef<'_, I, RhsT>,
	alpha: T,
	par: Par,
) -> Result<SparseColMat<I, T>, FaerError> {
	assert!(lhs.ncols() == rhs.nrows());

	let (symbolic, info) = sparse_sparse_matmul_symbolic(lhs.symbolic(), rhs.symbolic())?;
	let mut val = alloc::vec::Vec::new();
	val.try_reserve_exact(symbolic.row_idx().len()).ok().ok_or(FaerError::OutOfMemory)?;
	val.resize(symbolic.row_idx().len(), zero());

	sparse_sparse_matmul_numeric(
		SparseColMatMut::new(symbolic.rb(), &mut val),
		Accum::Add,
		lhs,
		rhs,
		alpha,
		&info,
		par,
		MemStack::new(&mut MemBuffer::try_new(sparse_sparse_matmul_numeric_scratch::<I, T>(symbolic.rb(), par))?),
	);

	Ok(SparseColMat::new(symbolic, val))
}

/// multiplies a sparse matrix `lhs` by a dense matrix `rhs`, and stores or adds the result to
/// `dst`. see [`faer::linalg::matmul::matmul`](crate::linalg::matmul::matmul) for more details.
///
/// # note
/// allows unsorted matrices.
#[track_caller]
#[math]
pub fn sparse_dense_matmul<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(
	dst: MatMut<'_, T>,
	beta: Accum,
	lhs: SparseColMatRef<'_, I, LhsT>,
	rhs: MatRef<'_, RhsT>,
	alpha: T,
	par: Par,
) {
	assert!(all(dst.nrows() == lhs.nrows(), dst.ncols() == rhs.ncols(), lhs.ncols() == rhs.nrows()));

	// TODO: parallelize this
	let _ = par;
	let mut dst = dst;

	if let Accum::Replace = beta {
		dst.fill(zero());
	}
	with_dim!(M, dst.nrows());
	with_dim!(N, dst.ncols());
	with_dim!(K, lhs.ncols());

	let mut dst = dst.as_shape_mut(M, N);
	let lhs = lhs.as_shape(M, K);
	let rhs = rhs.as_shape(K, N);

	for j in N.indices() {
		for depth in K.indices() {
			let rhs_kj = Conj::apply(&rhs[(depth, j)]) * alpha;
			for (i, lhs_ik) in iter::zip(lhs.row_idx_of_col(depth), lhs.val_of_col(depth)) {
				dst[(i, j)] = dst[(i, j)] + Conj::apply(lhs_ik) * rhs_kj;
			}
		}
	}
}

/// multiplies a dense matrix `lhs` by a sparse matrix `rhs`, and stores or adds the result to
/// `dst`. see [`faer::linalg::matmul::matmul`](crate::linalg::matmul::matmul) for more details.
///
/// # note
/// allows unsorted matrices.
#[track_caller]
#[math]
pub fn dense_sparse_matmul<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(
	dst: MatMut<'_, T>,
	beta: Accum,
	lhs: MatRef<'_, LhsT>,
	rhs: SparseColMatRef<'_, I, RhsT>,
	alpha: T,
	par: Par,
) {
	assert!(all(dst.nrows() == lhs.nrows(), dst.ncols() == rhs.ncols(), lhs.ncols() == rhs.nrows()));

	// TODO: parallelize this
	let _ = par;

	with_dim!(M, dst.nrows());
	with_dim!(N, dst.ncols());
	with_dim!(K, lhs.ncols());

	let mut dst = dst.as_shape_mut(M, N);
	let lhs = lhs.as_shape(M, K);
	let rhs = rhs.as_shape(K, N);

	for i in M.indices() {
		for j in N.indices() {
			let mut acc = zero::<T>();
			for (depth, rhs_kj) in iter::zip(rhs.row_idx_of_col(j), rhs.val_of_col(j)) {
				let l = Conj::apply(&lhs[(i, depth)]);
				let r = Conj::apply(rhs_kj);
				acc = acc + l * r;
			}
			match beta {
				Accum::Replace => dst[(i, j)] = alpha * acc,
				Accum::Add => dst[(i, j)] = dst[(i, j)] + alpha * acc,
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;

	#[test]
	fn test_sp_matmul() {
		let a = SparseColMat::<usize, f64>::try_new_from_triplets(
			5,
			4,
			&[
				Triplet::new(0, 0, 1.0),
				Triplet::new(1, 0, 2.0),
				Triplet::new(3, 0, 3.0),
				//
				Triplet::new(1, 1, 5.0),
				Triplet::new(4, 1, 6.0),
				//
				Triplet::new(0, 2, 7.0),
				Triplet::new(2, 2, 8.0),
				//
				Triplet::new(0, 3, 9.0),
				Triplet::new(2, 3, 10.0),
				Triplet::new(3, 3, 11.0),
				Triplet::new(4, 3, 12.0),
			],
		)
		.unwrap();

		let b = SparseColMat::<usize, f64>::try_new_from_triplets(
			4,
			6,
			&[
				Triplet::new(0, 0, 1.0),
				Triplet::new(1, 0, 2.0),
				Triplet::new(3, 0, 3.0),
				//
				Triplet::new(1, 1, 5.0),
				Triplet::new(3, 1, 6.0),
				//
				Triplet::new(1, 2, 7.0),
				Triplet::new(3, 2, 8.0),
				//
				Triplet::new(1, 3, 9.0),
				Triplet::new(3, 3, 10.0),
				//
				Triplet::new(1, 4, 11.0),
				Triplet::new(3, 4, 12.0),
				//
				Triplet::new(1, 5, 13.0),
				Triplet::new(3, 5, 14.0),
			],
		)
		.unwrap();

		let c = sparse_sparse_matmul(a.rb(), b.rb(), 2.0, Par::rayon(12)).unwrap();

		assert!(c.to_dense() == Scale(2.0) * a.to_dense() * b.to_dense());
	}
}
