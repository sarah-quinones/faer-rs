use super::*;
use crate::assert;
use crate::internal_prelude::*;

/// returns the resulting matrix obtained by applying `f` to the elements from `lhs` and `rhs`,
/// skipping entries that are unavailable in both of `lhs` and `rhs`.
///
/// # panics
/// panics if `lhs` and `rhs` don't have matching dimensions.  
#[track_caller]
pub fn binary_op<I: Index, T, LhsT, RhsT>(
	lhs: SparseColMatRef<'_, I, LhsT>,
	rhs: SparseColMatRef<'_, I, RhsT>,
	f: impl FnMut(Option<&LhsT>, Option<&RhsT>) -> T,
) -> Result<SparseColMat<I, T>, FaerError> {
	assert!(lhs.nrows() == rhs.nrows());
	assert!(lhs.ncols() == rhs.ncols());
	let mut f = f;
	let m = lhs.nrows();
	let n = lhs.ncols();

	let mut col_ptr = try_zeroed::<I>(n + 1)?;

	let mut nnz = 0usize;
	for j in 0..n {
		let lhs = lhs.row_idx_of_col_raw(j);
		let rhs = rhs.row_idx_of_col_raw(j);

		let mut lhs_pos = 0usize;
		let mut rhs_pos = 0usize;
		while lhs_pos < lhs.len() && rhs_pos < rhs.len() {
			let lhs = lhs[lhs_pos];
			let rhs = rhs[rhs_pos];

			lhs_pos += (lhs <= rhs) as usize;
			rhs_pos += (rhs <= lhs) as usize;
			nnz += 1;
		}
		nnz += lhs.len() - lhs_pos;
		nnz += rhs.len() - rhs_pos;
		col_ptr[j + 1] = I::truncate(nnz);
	}

	if nnz > I::Signed::MAX.zx() {
		return Err(FaerError::IndexOverflow);
	}

	let mut row_idx = try_zeroed(nnz)?;
	let mut values = alloc::vec::Vec::new();
	values.try_reserve_exact(nnz).map_err(|_| FaerError::OutOfMemory)?;

	let mut nnz = 0usize;
	for j in 0..n {
		let lhs_values = lhs.val_of_col(j);
		let rhs_values = rhs.val_of_col(j);
		let lhs = lhs.row_idx_of_col_raw(j);
		let rhs = rhs.row_idx_of_col_raw(j);

		let mut lhs_pos = 0usize;
		let mut rhs_pos = 0usize;
		while lhs_pos < lhs.len() && rhs_pos < rhs.len() {
			let lhs = lhs[lhs_pos];
			let rhs = rhs[rhs_pos];

			match lhs.cmp(&rhs) {
				core::cmp::Ordering::Less => {
					row_idx[nnz] = lhs;
					values.push(f(Some(&lhs_values[lhs_pos]), None));
				},
				core::cmp::Ordering::Equal => {
					row_idx[nnz] = lhs;
					values.push(f(Some(&lhs_values[lhs_pos]), Some(&rhs_values[rhs_pos])));
				},
				core::cmp::Ordering::Greater => {
					row_idx[nnz] = rhs;
					values.push(f(None, Some(&rhs_values[rhs_pos])));
				},
			}

			lhs_pos += (lhs <= rhs) as usize;
			rhs_pos += (rhs <= lhs) as usize;
			nnz += 1;
		}
		row_idx[nnz..nnz + lhs.len() - lhs_pos].copy_from_slice(&lhs[lhs_pos..]);
		for src in &lhs_values[lhs_pos..lhs.len()] {
			values.push(f(Some(src), None));
		}
		nnz += lhs.len() - lhs_pos;

		row_idx[nnz..nnz + rhs.len() - rhs_pos].copy_from_slice(&rhs[rhs_pos..]);
		for src in &rhs_values[rhs_pos..rhs.len()] {
			values.push(f(None, Some(src)));
		}
		nnz += rhs.len() - rhs_pos;
	}

	Ok(SparseColMat::<I, T>::new(
		SymbolicSparseColMat::<I>::new_checked(m, n, col_ptr, None, row_idx),
		values,
	))
}

/// returns the resulting matrix obtained by applying `f` to the elements from `dst` and `src`
/// skipping entries that are unavailable in both of them.  
/// the sparsity patter of `dst` is unchanged.
///
/// # panics
/// panics if `src` and `dst` don't have matching dimensions.  
/// panics if `src` contains an index that's unavailable in `dst`.  
#[track_caller]
pub fn binary_op_assign_into<I: Index, T, SrcT>(
	dst: SparseColMatMut<'_, I, T>,
	src: SparseColMatRef<'_, I, SrcT>,
	f: impl FnMut(&mut T, Option<&SrcT>),
) {
	{
		assert!(dst.nrows() == src.nrows());
		assert!(dst.ncols() == src.ncols());

		let n = dst.ncols();
		let mut dst = dst;
		let mut f = f;

		for j in 0..n {
			let (dst, dst_val) = dst.rb_mut().parts_mut();

			let dst_val = &mut dst_val[dst.col_range(j)];
			let src_val = src.val_of_col(j);

			let dst = dst.row_idx_of_col_raw(j);
			let src = src.row_idx_of_col_raw(j);

			let mut dst_pos = 0usize;
			let mut src_pos = 0usize;

			while src_pos < src.len() {
				let src = src[src_pos];

				if dst[dst_pos] < src {
					f(&mut dst_val[dst_pos], None);
					dst_pos += 1;
					continue;
				}

				assert!(dst[dst_pos] == src);

				f(&mut dst_val[dst_pos], Some(&src_val[src_pos]));

				src_pos += 1;
				dst_pos += 1;
			}
			while dst_pos < dst.len() {
				f(&mut dst_val[dst_pos], None);
				dst_pos += 1;
			}
		}
	}
}

/// returns the resulting matrix obtained by applying `f` to the elements from `dst`, `lhs` and
/// `rhs`, skipping entries that are unavailable in all of `dst`, `lhs` and `rhs`.  
/// the sparsity patter of `dst` is unchanged.
///
/// # panics
/// panics if `lhs`, `rhs` and `dst` don't have matching dimensions.  
/// panics if `lhs` or `rhs` contains an index that's unavailable in `dst`.  
#[track_caller]
pub fn ternary_op_assign_into<I: Index, T, LhsT, RhsT>(
	dst: SparseColMatMut<'_, I, T>,
	lhs: SparseColMatRef<'_, I, LhsT>,
	rhs: SparseColMatRef<'_, I, RhsT>,
	f: impl FnMut(&mut T, Option<&LhsT>, Option<&RhsT>),
) {
	{
		assert!(dst.nrows() == lhs.nrows());
		assert!(dst.ncols() == lhs.ncols());
		assert!(dst.nrows() == rhs.nrows());
		assert!(dst.ncols() == rhs.ncols());

		let n = dst.ncols();
		let mut dst = dst;
		let mut f = f;

		for j in 0..n {
			let (dst, dst_val) = dst.rb_mut().parts_mut();

			let dst_val = &mut dst_val[dst.col_range(j)];
			let lhs_val = lhs.val_of_col(j);
			let rhs_val = rhs.val_of_col(j);

			let dst = dst.row_idx_of_col_raw(j);
			let rhs = rhs.row_idx_of_col_raw(j);
			let lhs = lhs.row_idx_of_col_raw(j);

			let mut dst_pos = 0usize;
			let mut lhs_pos = 0usize;
			let mut rhs_pos = 0usize;

			while lhs_pos < lhs.len() && rhs_pos < rhs.len() {
				let lhs = lhs[lhs_pos];
				let rhs = rhs[rhs_pos];

				if dst[dst_pos] < Ord::min(lhs, rhs) {
					f(&mut dst_val[dst_pos], None, None);
					dst_pos += 1;
					continue;
				}

				assert!(dst[dst_pos] == Ord::min(lhs, rhs));

				match lhs.cmp(&rhs) {
					core::cmp::Ordering::Less => {
						f(&mut dst_val[dst_pos], Some(&lhs_val[lhs_pos]), None);
					},
					core::cmp::Ordering::Equal => {
						f(&mut dst_val[dst_pos], Some(&lhs_val[lhs_pos]), Some(&rhs_val[rhs_pos]));
					},
					core::cmp::Ordering::Greater => {
						f(&mut dst_val[dst_pos], None, Some(&rhs_val[rhs_pos]));
					},
				}

				lhs_pos += (lhs <= rhs) as usize;
				rhs_pos += (rhs <= lhs) as usize;
				dst_pos += 1;
			}
			while lhs_pos < lhs.len() {
				let lhs = lhs[lhs_pos];
				if dst[dst_pos] < lhs {
					f(&mut dst_val[dst_pos], None, None);
					dst_pos += 1;
					continue;
				}
				f(&mut dst_val[dst_pos], Some(&lhs_val[lhs_pos]), None);
				lhs_pos += 1;
				dst_pos += 1;
			}
			while rhs_pos < rhs.len() {
				let rhs = rhs[rhs_pos];
				if dst[dst_pos] < rhs {
					f(&mut dst_val[dst_pos], None, None);
					dst_pos += 1;
					continue;
				}
				f(&mut dst_val[dst_pos], None, Some(&rhs_val[rhs_pos]));
				rhs_pos += 1;
				dst_pos += 1;
			}
			while rhs_pos < rhs.len() {
				let rhs = rhs[rhs_pos];
				dst_pos += dst[dst_pos..].binary_search(&rhs).unwrap();
				f(&mut dst_val[dst_pos], None, Some(&rhs_val[rhs_pos]));
				rhs_pos += 1;
			}
		}
	}
}

/// returns the sparsity pattern containing the union of those of `lhs` and `rhs`.
///
/// # panics
/// panics if `lhs` and `rhs` don't have matching dimensions.  
#[track_caller]
#[inline]
pub fn union_symbolic<I: Index>(
	lhs: SymbolicSparseColMatRef<'_, I>,
	rhs: SymbolicSparseColMatRef<'_, I>,
) -> Result<SymbolicSparseColMat<I>, FaerError> {
	Ok(binary_op(
		SparseColMatRef::<I, Symbolic>::new(lhs, Symbolic::materialize(lhs.compute_nnz())),
		SparseColMatRef::<I, Symbolic>::new(rhs, Symbolic::materialize(rhs.compute_nnz())),
		#[inline(always)]
		|_, _| Symbolic,
	)?
	.into_parts()
	.0)
}

/// returns the sum of `lhs` and `rhs`.
///
/// # panics
/// panics if `lhs` and `rhs` don't have matching dimensions.  
#[track_caller]
#[inline]
pub fn add<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(
	lhs: SparseColMatRef<'_, I, LhsT>,
	rhs: SparseColMatRef<'_, I, RhsT>,
) -> Result<SparseColMat<I, T>, FaerError> {
	binary_op(lhs, rhs, |lhs, rhs| match (lhs.map(Conj::apply), rhs.map(Conj::apply)) {
		(None, None) => zero(),
		(None, Some(rhs)) => rhs,
		(Some(lhs), None) => lhs,
		(Some(lhs), Some(rhs)) => faer_traits::math_utils::add(&lhs, &rhs),
	})
}

/// returns the difference of `lhs` and `rhs`.
///
/// # panics
/// panics if `lhs` and `rhs` don't have matching dimensions.  
#[track_caller]
#[inline]
pub fn sub<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(
	lhs: SparseColMatRef<'_, I, LhsT>,
	rhs: SparseColMatRef<'_, I, RhsT>,
) -> Result<SparseColMat<I, T>, FaerError> {
	binary_op(lhs, rhs, |lhs, rhs| match (lhs.map(Conj::apply), rhs.map(Conj::apply)) {
		(None, None) => zero(),
		(None, Some(rhs)) => rhs,
		(Some(lhs), None) => lhs,
		(Some(lhs), Some(rhs)) => faer_traits::math_utils::sub(&lhs, &rhs),
	})
}

/// computes the sum of `dst` and `src` and stores the result in `dst` without changing its
/// symbolic structure.
///
/// # panics
/// panics if `dst` and `rhs` don't have matching dimensions.  
/// panics if `rhs` contains an index that's unavailable in `dst`.  
pub fn add_assign<I: Index, T: ComplexField, RhsT: Conjugate<Canonical = T>>(dst: SparseColMatMut<'_, I, T>, rhs: SparseColMatRef<'_, I, RhsT>) {
	binary_op_assign_into(dst, rhs, |dst, rhs| {
		*dst = faer_traits::math_utils::add(dst, &match rhs {
			Some(rhs) => Conj::apply(rhs),
			None => zero(),
		})
	})
}

/// computes the difference of `dst` and `src` and stores the result in `dst` without changing its
/// symbolic structure.
///
/// # panics
/// panics if `dst` and `rhs` don't have matching dimensions.  
/// panics if `rhs` contains an index that's unavailable in `dst`.  
pub fn sub_assign<I: Index, T: ComplexField, RhsT: Conjugate<Canonical = T>>(dst: SparseColMatMut<'_, I, T>, rhs: SparseColMatRef<'_, I, RhsT>) {
	binary_op_assign_into(dst, rhs, |dst, rhs| {
		*dst = faer_traits::math_utils::sub(dst, &match rhs {
			Some(rhs) => Conj::apply(rhs),
			None => zero(),
		})
	})
}

/// computes the sum of `lhs` and `rhs`, storing the result in `dst` without changing its
/// symbolic structure.
///
/// # panics
/// panics if `dst`, `lhs` and `rhs` don't have matching dimensions.  
/// panics if `lhs` or `rhs` contains an index that's unavailable in `dst`.  
#[track_caller]
#[inline]
pub fn add_into<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(
	dst: SparseColMatMut<'_, I, T>,
	lhs: SparseColMatRef<'_, I, LhsT>,
	rhs: SparseColMatRef<'_, I, RhsT>,
) {
	ternary_op_assign_into(dst, lhs, rhs, |dst, lhs, rhs| {
		*dst = match (lhs.map(Conj::apply), rhs.map(Conj::apply)) {
			(None, None) => zero(),
			(None, Some(rhs)) => rhs,
			(Some(lhs), None) => lhs,
			(Some(lhs), Some(rhs)) => faer_traits::math_utils::add(&lhs, &rhs),
		};
	})
}

/// computes the difference of `lhs` and `rhs`, storing the result in `dst` without changing its
/// symbolic structure.
///
/// # panics
/// panics if `dst`, `lhs` and `rhs` don't have matching dimensions.  
/// panics if `lhs` or `rhs` contains an index that's unavailable in `dst`.  
#[track_caller]
#[inline]
pub fn sub_into<I: Index, T: ComplexField, LhsT: Conjugate<Canonical = T>, RhsT: Conjugate<Canonical = T>>(
	dst: SparseColMatMut<'_, I, T>,
	lhs: SparseColMatRef<'_, I, LhsT>,
	rhs: SparseColMatRef<'_, I, RhsT>,
) {
	ternary_op_assign_into(dst, lhs, rhs, |dst, lhs, rhs| {
		*dst = match (lhs.map(Conj::apply), rhs.map(Conj::apply)) {
			(None, None) => zero(),
			(None, Some(rhs)) => rhs,
			(Some(lhs), None) => lhs,
			(Some(lhs), Some(rhs)) => faer_traits::math_utils::sub(&lhs, &rhs),
		};
	})
}
