use crate::internal_prelude_sp::*;
use crate::{assert, debug_assert};

/// sorts `row_indices` and `values` simultaneously so that `row_indices` is nonincreasing.
pub fn sort_indices<I: Index, T>(col_ptr: &[I], col_nnz: Option<&[I]>, row_idx: &mut [I], val: &mut [T]) {
	assert!(col_ptr.len() > 0);

	let n = col_ptr.len() - 1;
	for j in 0..n {
		let start = col_ptr[j].zx();
		let end = col_nnz.map(|nnz| start + nnz[j].zx()).unwrap_or(col_ptr[j + 1].zx());
		unsafe { crate::sort::sort_indices(&mut row_idx[start..end], &mut val[start..end]) };
	}
}

/// sorts and deduplicates `row_indices` and `values` simultaneously so that `row_indices` is
/// nonincreasing and contains no duplicate indices.
pub fn sort_dedup_indices<I: Index, T: ComplexField>(col_ptr: &[I], col_nnz: &mut [I], row_idx: &mut [I], val: &mut [T]) {
	assert!(col_ptr.len() > 0);

	let n = col_ptr.len() - 1;
	for j in 0..n {
		let start = col_ptr[j].zx();
		let end = start + col_nnz[j].zx();
		unsafe { crate::sort::sort_indices(&mut row_idx[start..end], &mut val[start..end]) };

		let mut prev = I::truncate(usize::MAX);

		let mut writer = start;
		let mut reader = start;
		while reader < end {
			let cur = row_idx[reader];
			if cur == prev {
				writer -= 1;
				val[writer] = add(&val[writer], &val[reader]);
			} else {
				val[writer] = copy(&val[reader]);
			}

			prev = cur;
			reader += 1;
			writer += 1;
		}

		col_nnz[j] = I::truncate(writer - start);
	}
}

/// computes the workspace layout required to apply a two sided permutation to a
/// self-adjoint sparse matrix
pub fn permute_self_adjoint_scratch<I: Index>(dim: usize) -> StackReq {
	StackReq::new::<I>(dim)
}

/// computes the workspace layout required to apply a two sided permutation to a
/// self-adjoint sparse matrix and deduplicate its elements
pub fn permute_dedup_self_adjoint_scratch<I: Index>(dim: usize) -> StackReq {
	StackReq::new::<I>(dim)
}

/// computes the self-adjoint permutation $P A P^\top$ of the matrix $A$
///
/// the result is stored in `new_col_ptrs`, `new_row_indices`
///
/// # note
/// allows unsorted matrices, producing a sorted output. duplicate entries are kept, however
pub fn permute_self_adjoint<'out, N: Shape, I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
	new_val: &'out mut [T],
	new_col_ptr: &'out mut [I],
	new_row_idx: &'out mut [I],
	A: SparseColMatRef<'_, I, C, N, N>,
	perm: PermRef<'_, I, N>,
	in_side: Side,
	out_side: Side,
	stack: &mut MemStack,
) -> SparseColMatMut<'out, I, T, N, N> {
	let n = A.nrows();
	with_dim!(N, n.unbound());

	permute_self_adjoint_imp(
		new_val,
		new_col_ptr,
		new_row_idx,
		A.as_shape(N, N).canonical(),
		Conj::get::<C>(),
		perm.as_shape(N),
		in_side,
		out_side,
		true,
		stack,
	)
	.as_shape_mut(n, n)
}

/// computes the self-adjoint permutation $P A P^\top$ of the matrix $A$ without sorting the row
/// indices, and returns a view over it
///
/// the result is stored in `new_col_ptrs`, `new_row_indices`
///
/// # note
/// allows unsorted matrices, producing an sorted output
pub fn permute_self_adjoint_to_unsorted<'out, N: Shape, I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
	new_val: &'out mut [T],
	new_col_ptr: &'out mut [I],
	new_row_idx: &'out mut [I],
	A: SparseColMatRef<'_, I, C, N, N>,
	perm: PermRef<'_, I, N>,
	in_side: Side,
	out_side: Side,
	stack: &mut MemStack,
) -> SparseColMatMut<'out, I, T, N, N> {
	let n = A.nrows();
	with_dim!(N, n.unbound());

	permute_self_adjoint_imp(
		new_val,
		new_col_ptr,
		new_row_idx,
		A.as_shape(N, N).canonical(),
		Conj::get::<C>(),
		perm.as_shape(N),
		in_side,
		out_side,
		false,
		stack,
	)
	.as_shape_mut(n, n)
}

/// computes the self-adjoint permutation $P A P^\top$ of the matrix $A$ and deduplicate the
/// elements of the output matrix
///
/// the result is stored in `new_col_ptrs`, `new_row_indices`
///
/// # note
/// allows unsorted matrices, producing a sorted output. duplicate entries are merged
pub fn permute_dedup_self_adjoint<'out, N: Shape, I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
	new_val: &'out mut [T],
	new_col_ptr: &'out mut [I],
	new_row_idx: &'out mut [I],
	A: SparseColMatRef<'_, I, C, N, N>,
	perm: PermRef<'_, I, N>,
	in_side: Side,
	out_side: Side,
	stack: &mut MemStack,
) -> SparseColMatMut<'out, I, T, N, N> {
	let n = A.nrows();
	with_dim!(N, n.unbound());

	permute_dedup_self_adjoint_imp(
		new_val,
		new_col_ptr,
		new_row_idx,
		A.as_shape(N, N).canonical(),
		Conj::get::<C>(),
		perm.as_shape(N),
		in_side,
		out_side,
		stack,
	)
	.as_shape_mut(n, n)
}

fn permute_self_adjoint_imp<'N, 'out, I: Index, T: ComplexField>(
	new_val: &'out mut [T],
	new_col_ptr: &'out mut [I],
	new_row_idx: &'out mut [I],
	A: SparseColMatRef<'_, I, T, Dim<'N>, Dim<'N>>,
	conj_A: Conj,
	perm: PermRef<'_, I, Dim<'N>>,
	in_side: Side,
	out_side: Side,
	sort: bool,
	stack: &mut MemStack,
) -> SparseColMatMut<'out, I, T, Dim<'N>, Dim<'N>> {
	// old_i <= old_j => -old_i >= -old_j
	// reverse the order with bitwise not
	// x + !x == MAX
	// x + !x + 1 == 0
	// !x = -1 - x

	// if we flipped the side of A, then we need to check old_i <= old_j instead
	let src_to_cmp = {
		let mask = match in_side {
			Side::Lower => 0,
			Side::Upper => usize::MAX,
		};
		move |i: usize| mask ^ i
	};

	let dst_to_cmp = {
		let mask = match out_side {
			Side::Lower => 0,
			Side::Upper => usize::MAX,
		};
		move |i: usize| mask ^ i
	};

	let conj_A = conj_A.is_conj();

	// in_side/out_side are assumed Side::Lower

	let N = A.ncols();
	let n = *N;

	assert!(new_col_ptr.len() == n + 1);
	let (_, perm_inv) = perm.bound_arrays();

	let (mut cur_row_pos, _) = stack.collect(repeat_n!(I::truncate(0), n));
	let cur_row_pos = Array::from_mut(&mut cur_row_pos, N);

	let col_counts = &mut *cur_row_pos;
	for old_j in N.indices() {
		let new_j = perm_inv[old_j].zx();

		let old_j_cmp = src_to_cmp(*old_j);
		let new_j_cmp = dst_to_cmp(*new_j);

		for old_i in A.row_idx_of_col(old_j) {
			let new_i = perm_inv[old_i].zx();

			let old_i_cmp = src_to_cmp(*old_i);
			let new_i_cmp = dst_to_cmp(*new_i);

			if old_i_cmp >= old_j_cmp {
				let lower = new_i_cmp >= new_j_cmp;
				let new_j = if lower { new_j } else { new_i };

				// cannot overflow because A.compute_nnz() <= I::Signed::MAX
				// col_counts[new_j] always >= 0
				col_counts[new_j] += I::truncate(1);
			}
		}
	}

	// col_counts[_] >= 0
	// cumulative sum cannot overflow because it's <= A.compute_nnz()

	new_col_ptr[0] = I::truncate(0);
	for (count, [ci0, ci1]) in iter::zip(col_counts.as_mut(), windows2(Cell::as_slice_of_cells(Cell::from_mut(&mut *new_col_ptr)))) {
		let ci0 = ci0.get();
		ci1.set(ci0 + *count);
		*count = ci0;
	}

	// new_col_ptr is non-decreasing
	let nnz = new_col_ptr[n].zx();
	let new_row_idx = &mut new_row_idx[..nnz];
	let new_val = &mut new_val[..nnz];

	{
		with_dim!(NNZ, nnz);
		let new_val = Array::from_mut(new_val, NNZ);
		let new_row_idx = Array::from_mut(new_row_idx, NNZ);

		let conj_if = |cond: bool, x: &T| -> T {
			if try_const! { T::IS_REAL } {
				copy(x)
			} else {
				if cond != conj_A { conj(x) } else { copy(x) }
			}
		};

		for old_j in N.indices() {
			let new_j = perm_inv[old_j].zx();

			let old_j_cmp = src_to_cmp(*old_j);
			let new_j_cmp = dst_to_cmp(*new_j);

			for (old_i, val) in iter::zip(A.row_idx_of_col(old_j), A.val_of_col(old_j)) {
				let new_i = perm_inv[old_i].zx();

				let old_i_cmp = src_to_cmp(*old_i);
				let new_i_cmp = dst_to_cmp(*new_i);

				if old_i_cmp >= old_j_cmp {
					let lower = new_i_cmp >= new_j_cmp;

					let (new_j, new_i) = if lower { (new_j, new_i) } else { (new_i, new_j) };

					let cur_row_pos = &mut cur_row_pos[new_j];

					// SAFETY: cur_row_pos < NNZ
					let row_pos = unsafe { Idx::new_unchecked(cur_row_pos.zx(), NNZ) };

					*cur_row_pos += I::truncate(1);

					new_val[row_pos] = conj_if(!lower, val);
					new_row_idx[row_pos] = I::truncate(*new_i);
				}
			}
		}
	}

	if sort {
		sort_indices(new_col_ptr, None, new_row_idx, new_val);
	}
	// SAFETY:
	// 0. new_col_ptr is non-decreasing
	// 1. all written row indices are less than n

	unsafe { SparseColMatMut::new(SymbolicSparseColMatRef::new_unchecked(N, N, new_col_ptr, None, new_row_idx), new_val) }
}

fn permute_dedup_self_adjoint_imp<'N, 'out, I: Index, T: ComplexField>(
	new_val: &'out mut [T],
	new_col_ptr: &'out mut [I],
	new_row_idx: &'out mut [I],
	A: SparseColMatRef<'_, I, T, Dim<'N>, Dim<'N>>,
	conj_A: Conj,
	perm: PermRef<'_, I, Dim<'N>>,
	in_side: Side,
	out_side: Side,
	stack: &mut MemStack,
) -> SparseColMatMut<'out, I, T, Dim<'N>, Dim<'N>> {
	let N = A.nrows();

	permute_self_adjoint_imp(new_val, new_col_ptr, new_row_idx, A, conj_A, perm, in_side, out_side, false, stack);

	{
		let new_col_ptr = Cell::as_slice_of_cells(Cell::from_mut(new_col_ptr));

		let start = Array::from_ref(&new_col_ptr[..*N], N);
		let end = Array::from_ref(&new_col_ptr[1..], N);
		let mut writer = 0usize;

		for j in N.indices() {
			let start = start[j].replace(I::truncate(writer)).zx();
			let end = end[j].get().zx();

			unsafe {
				crate::sort::sort_indices(&mut new_row_idx[start..end], &mut new_val[start..end]);
			}

			let mut prev = I::truncate(usize::MAX);

			let mut reader = start;
			while reader < end {
				let cur = new_row_idx[reader];

				if cur == prev {
					// same element, add
					writer -= 1;
					new_val[writer] = add(&new_val[writer], &new_val[reader]);
				} else {
					// new element, copy
					new_row_idx[writer] = new_row_idx[reader];
					new_val[writer] = copy(&new_val[reader]);
				}

				prev = cur;
				reader += 1;
				writer += 1;
			}
		}
		new_col_ptr[*N].set(I::truncate(writer));
	}

	unsafe { SparseColMatMut::new(SymbolicSparseColMatRef::new_unchecked(N, N, new_col_ptr, None, new_row_idx), new_val) }
}

/// computes the workspace layout required to transpose a matrix
pub fn transpose_scratch<I: Index>(nrows: usize, ncols: usize) -> StackReq {
	_ = ncols;
	StackReq::new::<usize>(nrows)
}

/// computes the workspace layout required to transpose a matrix and deduplicate the
/// output elements
pub fn transpose_dedup_scratch<I: Index>(nrows: usize, ncols: usize) -> StackReq {
	_ = ncols;
	StackReq::new::<usize>(nrows).array(2)
}

/// computes the transpose of the matrix $A$ and returns a view over it.
///
/// the result is stored in `new_col_ptrs`, `new_row_indices` and `new_values`.
///
/// # note
/// allows unsorted matrices, producing a sorted output. duplicate entries are kept, however
pub fn transpose<'out, Rows: Shape, Cols: Shape, I: Index, T: Clone>(
	new_val: &'out mut [T],
	new_col_ptr: &'out mut [I],
	new_row_idx: &'out mut [I],
	A: SparseColMatRef<'_, I, T, Rows, Cols>,
	stack: &mut MemStack,
) -> SparseColMatMut<'out, I, T, Cols, Rows> {
	let (m, n) = A.shape();
	with_dim!(M, m.unbound());
	with_dim!(N, n.unbound());

	transpose_imp(T::clone, new_val, new_col_ptr, new_row_idx, A.as_shape(M, N), stack).as_shape_mut(n, m)
}

/// computes the adjoint of the matrix $A$ and returns a view over it.
///
/// the result is stored in `new_col_ptrs`, `new_row_indices` and `new_values`.
///
/// # note
/// allows unsorted matrices, producing a sorted output. duplicate entries are kept, however
pub fn adjoint<'out, Rows: Shape, Cols: Shape, I: Index, T: ComplexField>(
	new_val: &'out mut [T],
	new_col_ptr: &'out mut [I],
	new_row_idx: &'out mut [I],
	A: SparseColMatRef<'_, I, T, Rows, Cols>,
	stack: &mut MemStack,
) -> SparseColMatMut<'out, I, T, Cols, Rows> {
	let (m, n) = A.shape();
	with_dim!(M, m.unbound());
	with_dim!(N, n.unbound());

	transpose_imp(conj::<T>, new_val, new_col_ptr, new_row_idx, A.as_shape(M, N), stack).as_shape_mut(n, m)
}

/// computes the transpose of the matrix $A$ and returns a view over it.
///
/// the result is stored in `new_col_ptrs`, `new_row_indices` and `new_values`.
///
/// # note
/// allows unsorted matrices, producing a sorted output. duplicate entries are merged
pub fn transpose_dedup<'out, Rows: Shape, Cols: Shape, I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
	new_val: &'out mut [T],
	new_col_ptr: &'out mut [I],
	new_row_idx: &'out mut [I],
	A: SparseColMatRef<'_, I, C, Rows, Cols>,
	stack: &mut MemStack,
) -> SparseColMatMut<'out, I, T, Cols, Rows> {
	let (m, n) = A.shape();
	with_dim!(M, m.unbound());
	with_dim!(N, n.unbound());

	transpose_dedup_imp(new_val, new_col_ptr, new_row_idx, A.as_shape(M, N), stack).as_shape_mut(n, m)
}

fn transpose_imp<'ROWS, 'COLS, 'out, I: Index, T>(
	clone: impl Fn(&T) -> T,
	new_val: &'out mut [T],
	new_col_ptr: &'out mut [I],
	new_row_idx: &'out mut [I],
	A: SparseColMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
	stack: &mut MemStack,
) -> SparseColMatMut<'out, I, T, Dim<'COLS>, Dim<'ROWS>> {
	let (M, N) = A.shape();
	assert!(new_col_ptr.len() == *M + 1);
	let (mut col_count, _) = stack.collect(repeat_n!(I::truncate(0), *M));
	let col_count = Array::from_mut(&mut col_count, M);

	// can't overflow because the total count is A.compute_nnz() <= I::Signed::MAX
	for j in N.indices() {
		for i in A.row_idx_of_col(j) {
			col_count[i] += I::truncate(1);
		}
	}

	new_col_ptr[0] = I::truncate(0);

	// col_count elements are >= 0
	for (j, [pj0, pj1]) in iter::zip(M.indices(), windows2(Cell::as_slice_of_cells(Cell::from_mut(new_col_ptr)))) {
		let cj = &mut col_count[j];
		let pj = pj0.get();
		// new_col_ptr is non-decreasing
		pj1.set(pj + *cj);

		// *cj = cur_row_pos
		*cj = pj;
	}

	let new_row_idx = &mut new_row_idx[..new_col_ptr[*M].zx()];
	let new_val = &mut new_val[..new_col_ptr[*M].zx()];
	let cur_row_pos = col_count;

	for j in N.indices() {
		for (i, val) in iter::zip(A.row_idx_of_col(j), A.val_of_col(j)) {
			let ci = &mut cur_row_pos[i];
			// SAFETY: see below
			unsafe {
				let ci = ci.zx();
				*new_row_idx.get_unchecked_mut(ci) = I::truncate(*j);
				*new_val.get_unchecked_mut(ci) = clone(val);
			}
			*ci += I::truncate(1);
		}
	}
	// cur_row_pos[i] == col_ptr[i] + col_count[i] == col_ptr[i + 1] <= col_ptr[m]
	// so all the unchecked accesses were valid and non-overlapping, which means the entire array is
	// filled.
	debug_assert!(cur_row_pos.as_ref() == &new_col_ptr[1..]);

	// SAFETY:
	// 0. new_col_ptr is non-decreasing
	// 1. all written row indices are less than n
	unsafe { SparseColMatMut::new(SymbolicSparseColMatRef::new_unchecked(N, M, new_col_ptr, None, new_row_idx), new_val) }
}

fn transpose_dedup_imp<'ROWS, 'COLS, 'out, I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
	new_val: &'out mut [T],
	new_col_ptr: &'out mut [I],
	new_row_idx: &'out mut [I],
	A: SparseColMatRef<'_, I, C, Dim<'ROWS>, Dim<'COLS>>,
	stack: &mut MemStack,
) -> SparseColMatMut<'out, I, T, Dim<'COLS>, Dim<'ROWS>> {
	let (M, N) = A.shape();
	assert!(new_col_ptr.len() == *M + 1);
	let A = A.canonical();

	let sentinel = I::truncate(usize::MAX);
	let (mut col_count, stack) = stack.collect(repeat_n!(I::truncate(0), *M));
	let (mut last_seen, _) = stack.collect(repeat_n!(sentinel, *M));

	let col_count = Array::from_mut(&mut col_count, M);
	let last_seen = Array::from_mut(&mut last_seen, M);

	// can't overflow because the total count is A.compute_nnz() <= I::Signed::MAX
	for j in N.indices() {
		for i in A.row_idx_of_col(j) {
			let j = I::truncate(*j);
			if last_seen[i] == j {
				continue;
			}
			last_seen[i] = j;
			col_count[i] += I::truncate(1);
		}
	}

	new_col_ptr[0] = I::truncate(0);

	// col_count elements are >= 0
	for (j, [pj0, pj1]) in iter::zip(M.indices(), windows2(Cell::as_slice_of_cells(Cell::from_mut(new_col_ptr)))) {
		let cj = &mut col_count[j];
		let pj = pj0.get();
		// new_col_ptr is non-decreasing
		pj1.set(pj + *cj);

		// *cj = cur_row_pos
		*cj = pj;
	}

	last_seen.as_mut().fill(sentinel);

	let new_row_idx = &mut new_row_idx[..new_col_ptr[*M].zx()];
	let new_val = &mut new_val[..new_col_ptr[*M].zx()];
	let cur_row_pos = col_count;

	for j in N.indices() {
		for (i, val) in iter::zip(A.row_idx_of_col(j), A.val_of_col(j)) {
			let ci = &mut cur_row_pos[i];

			let val = if Conj::get::<C>().is_conj() { conj(val) } else { copy(val) };

			let j = I::truncate(*j);
			// SAFETY: see below
			unsafe {
				if last_seen[i] == j {
					let ci = ci.zx() - 1;
					*new_val.get_unchecked_mut(ci) = add(new_val.get_unchecked(ci), &val);
				} else {
					last_seen[i] = j;
					*ci += I::truncate(1);

					let ci = ci.zx() - 1;
					{
						*new_row_idx.get_unchecked_mut(ci) = j;
						*new_val.get_unchecked_mut(ci) = val;
					}
				}
			}
		}
	}
	// cur_row_pos[i] == col_ptr[i] + col_count[i] == col_ptr[i + 1] <= col_ptr[m]
	// so all the unchecked accesses were valid and non-overlapping, which means the entire array is
	// filled.
	debug_assert!(cur_row_pos.as_ref() == &new_col_ptr[1..]);

	// SAFETY:
	// 0. new_col_ptr is non-decreasing
	// 1. all written row indices are less than n
	unsafe { SparseColMatMut::new(SymbolicSparseColMatRef::new_unchecked(N, M, new_col_ptr, None, new_row_idx), new_val) }
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use dyn_stack::MemBuffer;

	#[test]
	fn test_transpose() {
		let nrows = 5;
		let ncols = 3;
		let A = SparseColMatRef::new(
			SymbolicSparseColMatRef::new_unsorted_checked(
				nrows,
				ncols,
				&[0usize, 4, 8, 11],
				None,
				&[
					0, 0, 2, 4, //
					2, 1, 1, 0, //
					0, 1, 3,
				],
			),
			&[
				1.0, 2.0, 3.0, 4.0, //
				11.0, 12.0, 13.0, 14.0, //
				21.0, 22.0, 23.0,
			],
		);
		let nnz = A.compute_nnz();

		let new_col_ptr = &mut *vec![0usize; nrows + 1];
		let new_row_idx = &mut *vec![0usize; nnz];
		let new_val = &mut *vec![0.0; nnz];
		{
			let out = transpose(
				new_val,
				new_col_ptr,
				new_row_idx,
				A,
				MemStack::new(&mut MemBuffer::new(transpose_scratch::<usize>(nrows, ncols))),
			)
			.into_const();

			let target = SparseColMatRef::new(
				SymbolicSparseColMatRef::new_unsorted_checked(
					ncols,
					nrows,
					&[0usize, 4, 7, 9, 10, 11],
					None,
					&[
						0, 0, 1, 2, //
						1, 1, 2, //
						0, 1, //
						2, //
						0,
					],
				),
				&[
					1.0, 2.0, 14.0, 21.0, //
					12.0, 13.0, 22.0, //
					3.0, 11.0, //
					23.0, //
					4.0,
				],
			);

			assert!(all(
				out.col_ptr() == target.col_ptr(),
				out.row_idx() == target.row_idx(),
				out.val() == target.val()
			));
		}

		{
			let out = transpose_dedup(
				new_val,
				new_col_ptr,
				new_row_idx,
				A,
				MemStack::new(&mut MemBuffer::new(transpose_dedup_scratch::<usize>(nrows, ncols))),
			)
			.into_const();

			let target = SparseColMatRef::new(
				SymbolicSparseColMatRef::new_unsorted_checked(
					ncols,
					nrows,
					&[0usize, 3, 5, 7, 8, 9],
					None,
					&[
						0, 1, 2, //
						1, 2, //
						0, 1, //
						2, //
						0,
					],
				),
				&[
					3.0, 14.0, 21.0, //
					25.0, 22.0, //
					3.0, 11.0, //
					23.0, //
					4.0,
				],
			);

			assert!(all(
				out.col_ptr() == target.col_ptr(),
				out.row_idx() == target.row_idx(),
				out.val() == target.val()
			));
		}
	}

	#[test]
	fn test_permute_self_adjoint() {
		let n = 5;
		let rng = &mut StdRng::seed_from_u64(0);
		let diag_rng = &mut StdRng::seed_from_u64(1);

		let mut rand = || ComplexDistribution::new(StandardNormal, StandardNormal).rand::<c64>(rng);
		let mut rand_diag = || c64::new(StandardNormal.rand(diag_rng), 0.0);

		let val = &[
			rand_diag(),
			rand_diag(),
			rand(),
			rand(),
			//
			rand(),
			rand_diag(),
			rand_diag(),
			rand(),
			//
			rand(),
			rand(),
			rand(),
			//
			rand(),
			rand_diag(),
			rand(),
			//
			rand_diag(),
			rand(),
			rand(),
		];

		let A = SparseColMatRef::new(
			SymbolicSparseColMatRef::new_unsorted_checked(
				n,
				n,
				&[0usize, 4, 8, 11, 14, 17],
				None,
				&[
					0, 0, 2, 4, //
					2, 1, 1, 0, //
					0, 1, 3, //
					2, 3, 4, //
					4, 3, 2, //
				],
			),
			val,
		);
		let nnz = A.compute_nnz();

		let perm_fwd = &mut *vec![0, 4, 1, 3, 2usize];
		let perm_bwd = &mut *vec![0; 5];
		for i in 0..n {
			perm_bwd[perm_fwd[i]] = i;
		}

		let perm = PermRef::new_checked(perm_fwd, perm_bwd, n);

		let new_col_ptr = &mut *vec![0usize; n + 1];
		let new_row_idx = &mut *vec![0usize; nnz];
		let new_val = &mut *vec![c64::ZERO; nnz];

		for f in [permute_self_adjoint_to_unsorted, permute_self_adjoint, permute_dedup_self_adjoint] {
			for (in_side, out_side) in [
				(Side::Lower, Side::Lower),
				(Side::Lower, Side::Upper),
				(Side::Upper, Side::Lower),
				(Side::Upper, Side::Upper),
			] {
				let mut out = f(
					new_val,
					new_col_ptr,
					new_row_idx,
					A,
					perm,
					in_side,
					out_side,
					MemStack::new(&mut MemBuffer::new(permute_self_adjoint_scratch::<usize>(n))),
				)
				.to_dense();

				let mut A = A.to_dense();

				match in_side {
					Side::Lower => {
						z!(&mut A).for_each_triangular_upper(linalg::zip::Diag::Skip, |uz!(x)| *x = c64::ZERO);
						for j in 0..n {
							for i in 0..j {
								A[(i, j)] = A[(j, i)].conj();
							}
						}
					},
					Side::Upper => {
						z!(&mut A).for_each_triangular_lower(linalg::zip::Diag::Skip, |uz!(x)| *x = c64::ZERO);
						for j in 0..n {
							for i in j + 1..n {
								A[(i, j)] = A[(j, i)].conj();
							}
						}
					},
				}

				match out_side {
					Side::Lower => {
						for j in 0..n {
							for i in 0..j {
								out[(i, j)] = out[(j, i)].conj();
							}
						}
					},
					Side::Upper => {
						for j in 0..n {
							for i in j + 1..n {
								out[(i, j)] = out[(j, i)].conj();
							}
						}
					},
				}

				assert!(out == perm * &A * perm.inverse());
			}
		}
	}
}
