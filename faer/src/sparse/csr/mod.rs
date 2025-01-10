use super::*;
use crate::internal_prelude::*;
use crate::{Idx, MaybeIdx, assert, debug_assert};
use core::ops::Range;
use core::{fmt, iter};

pub struct SymbolicSparseRowMatRef<'a, I, Rows = usize, Cols = usize> {
	pub(super) nrows: Rows,
	pub(super) ncols: Cols,
	pub(super) row_ptr: &'a [I],
	pub(super) row_nnz: Option<&'a [I]>,
	pub(super) col_idx: &'a [I],
}

pub struct SparseRowMatRef<'a, I, T, Rows = usize, Cols = usize> {
	pub(super) symbolic: SymbolicSparseRowMatRef<'a, I, Rows, Cols>,
	pub(super) val: &'a [T],
}

pub struct SparseRowMatMut<'a, I, T, Rows = usize, Cols = usize> {
	pub(super) symbolic: SymbolicSparseRowMatRef<'a, I, Rows, Cols>,
	pub(super) val: &'a mut [T],
}

#[derive(Clone)]
pub struct SymbolicSparseRowMat<I, Rows = usize, Cols = usize> {
	pub(super) nrows: Rows,
	pub(super) ncols: Cols,
	pub(super) row_ptr: alloc::vec::Vec<I>,
	pub(super) row_nnz: Option<alloc::vec::Vec<I>>,
	pub(super) col_idx: alloc::vec::Vec<I>,
}

#[derive(Clone)]
pub struct SparseRowMat<I, T, Rows = usize, Cols = usize> {
	pub(super) symbolic: SymbolicSparseRowMat<I, Rows, Cols>,
	pub(super) val: alloc::vec::Vec<T>,
}

impl<'a, I, Rows: Copy, Cols: Copy> Copy for SymbolicSparseRowMatRef<'a, I, Rows, Cols> {}
impl<'a, I, T, Rows: Copy, Cols: Copy> Copy for SparseRowMatRef<'a, I, T, Rows, Cols> {}

impl<'a, I, Rows: Copy, Cols: Copy> Clone for SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
impl<'a, I, T, Rows: Copy, Cols: Copy> Clone for SparseRowMatRef<'a, I, T, Rows, Cols> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

impl<'a, I, Rows: Copy, Cols: Copy> IntoConst for SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
	type Target = SymbolicSparseRowMatRef<'a, I, Rows, Cols>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}

impl<'short, 'a, I, Rows: Copy, Cols: Copy> ReborrowMut<'short> for SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
	type Target = SymbolicSparseRowMatRef<'short, I, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}

impl<'short, 'a, I, Rows: Copy, Cols: Copy> Reborrow<'short> for SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
	type Target = SymbolicSparseRowMatRef<'short, I, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}

impl<'a, I, T, Rows: Copy, Cols: Copy> IntoConst for SparseRowMatRef<'a, I, T, Rows, Cols> {
	type Target = SparseRowMatRef<'a, I, T, Rows, Cols>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> ReborrowMut<'short> for SparseRowMatRef<'a, I, T, Rows, Cols> {
	type Target = SparseRowMatRef<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> Reborrow<'short> for SparseRowMatRef<'a, I, T, Rows, Cols> {
	type Target = SparseRowMatRef<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}

impl<'a, I, T, Rows: Copy, Cols: Copy> IntoConst for SparseRowMatMut<'a, I, T, Rows, Cols> {
	type Target = SparseRowMatRef<'a, I, T, Rows, Cols>;

	#[inline]
	fn into_const(self) -> Self::Target {
		SparseRowMatRef {
			symbolic: self.symbolic,
			val: self.val,
		}
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> ReborrowMut<'short> for SparseRowMatMut<'a, I, T, Rows, Cols> {
	type Target = SparseRowMatMut<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		SparseRowMatMut {
			symbolic: self.symbolic,
			val: self.val,
		}
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> Reborrow<'short> for SparseRowMatMut<'a, I, T, Rows, Cols> {
	type Target = SparseRowMatRef<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		SparseRowMatRef {
			symbolic: self.symbolic,
			val: self.val,
		}
	}
}

impl<'short, I, T, Rows: Copy, Cols: Copy> ReborrowMut<'short> for SparseRowMat<I, T, Rows, Cols> {
	type Target = SparseRowMatMut<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		SparseRowMatMut {
			symbolic: self.symbolic.rb(),
			val: &mut self.val,
		}
	}
}

impl<'short, I, T, Rows: Copy, Cols: Copy> Reborrow<'short> for SparseRowMat<I, T, Rows, Cols> {
	type Target = SparseRowMatRef<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		SparseRowMatRef {
			symbolic: self.symbolic.rb(),
			val: &self.val,
		}
	}
}

impl<'short, I, Rows: Copy, Cols: Copy> Reborrow<'short> for SymbolicSparseRowMat<I, Rows, Cols> {
	type Target = SymbolicSparseRowMatRef<'short, I, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		SymbolicSparseRowMatRef {
			nrows: self.nrows,
			ncols: self.ncols,
			row_ptr: &self.row_ptr,
			row_nnz: self.row_nnz.as_deref(),
			col_idx: &self.col_idx,
		}
	}
}

#[inline(always)]
#[track_caller]
fn assume_row_ptr<I: Index>(nrows: usize, ncols: usize, row_ptr: &[I], row_nnz: Option<&[I]>, col_idx: &[I]) {
	assert!(all(ncols <= I::Signed::MAX.zx(), nrows <= I::Signed::MAX.zx(),));
	assert!(row_ptr.len() == nrows + 1);
	assert!(row_ptr[nrows].zx() <= col_idx.len());
	if let Some(row_nnz) = row_nnz {
		assert!(row_nnz.len() == nrows);
	}
}

#[track_caller]
fn check_row_ptr<I: Index>(nrows: usize, ncols: usize, row_ptr: &[I], row_nnz: Option<&[I]>, col_idx: &[I]) {
	assert!(all(ncols <= I::Signed::MAX.zx(), nrows <= I::Signed::MAX.zx(),));
	assert!(row_ptr.len() == nrows + 1);
	if let Some(row_nnz) = row_nnz {
		assert!(row_nnz.len() == nrows);
		for (&nnz_i, &[row, row_next]) in iter::zip(row_nnz, windows2(row_ptr)) {
			assert!(row <= row_next);
			assert!(nnz_i <= row_next - row);
		}
	} else {
		for &[row, row_next] in windows2(row_ptr) {
			assert!(row <= row_next);
		}
	}
	assert!(row_ptr[nrows].zx() <= col_idx.len());
}

#[track_caller]
fn check_col_idx<I: Index>(nrows: usize, ncols: usize, row_ptr: &[I], row_nnz: Option<&[I]>, col_idx: &[I]) {
	_ = nrows;
	if let Some(row_nnz) = row_nnz {
		for (nnz, &r) in iter::zip(row_nnz, row_ptr) {
			let r = r.zx();
			let nnz = nnz.zx();
			let col_idx = &col_idx[r..r + nnz];
			if !col_idx.is_empty() {
				let mut j = col_idx[0].zx();
				for &j_next in &col_idx[1..] {
					let j_next = j_next.zx();
					assert!(j < j_next);
					j = j_next;
				}
				assert!(j < ncols);
			}
		}
	} else {
		for &[r, r_next] in windows2(row_ptr) {
			let col_idx = &col_idx[r.zx()..r_next.zx()];
			if !col_idx.is_empty() {
				let mut j = col_idx[0].zx();
				for &j_next in &col_idx[1..] {
					let j_next = j_next.zx();
					assert!(j < j_next);
					j = j_next;
				}
				assert!(j < ncols);
			}
		}
	}
}

#[track_caller]
fn check_col_idx_unsorted<I: Index>(nrows: usize, ncols: usize, row_ptr: &[I], row_nnz: Option<&[I]>, col_idx: &[I]) {
	_ = nrows;

	if let Some(row_nnz) = row_nnz {
		for (&nnz, &r) in iter::zip(row_nnz, row_ptr) {
			let r = r.zx();
			let nnz = nnz.zx();
			for &j in &col_idx[r..r + nnz] {
				let j = j.zx();
				assert!(j < ncols);
			}
		}
	} else {
		for &[r, r_next] in windows2(row_ptr) {
			for &j in &col_idx[r.zx()..r_next.zx()] {
				let j = j.zx();
				assert!(j < ncols);
			}
		}
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index> SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
	#[inline]
	#[track_caller]
	pub unsafe fn new_unchecked(nrows: Rows, ncols: Cols, row_ptr: &'a [I], row_nnz: Option<&'a [I]>, col_idx: &'a [I]) -> Self {
		assume_row_ptr(nrows.unbound(), ncols.unbound(), row_ptr, row_nnz, col_idx);

		Self {
			nrows,
			ncols,
			row_ptr,
			row_nnz,
			col_idx,
		}
	}

	#[inline]
	#[track_caller]
	pub fn new_checked(nrows: Rows, ncols: Cols, row_ptr: &'a [I], row_nnz: Option<&'a [I]>, col_idx: &'a [I]) -> Self {
		check_row_ptr(nrows.unbound(), ncols.unbound(), row_ptr, row_nnz, col_idx);
		check_col_idx(nrows.unbound(), ncols.unbound(), row_ptr, row_nnz, col_idx);

		Self {
			nrows,
			ncols,
			row_ptr,
			row_nnz,
			col_idx,
		}
	}

	#[inline]
	pub fn parts(self) -> (Rows, Cols, &'a [I], Option<&'a [I]>, &'a [I]) {
		(self.nrows, self.ncols, self.row_ptr, self.row_nnz, self.col_idx)
	}

	#[inline]
	pub fn nrows(&self) -> Rows {
		self.nrows
	}

	#[inline]
	pub fn ncols(&self) -> Cols {
		self.ncols
	}

	#[inline]
	pub fn shape(&self) -> (Rows, Cols) {
		(self.nrows, self.ncols)
	}

	#[inline]
	pub fn transpose(self) -> SymbolicSparseColMatRef<'a, I, Cols, Rows> {
		SymbolicSparseColMatRef {
			nrows: self.ncols,
			ncols: self.nrows,
			col_ptr: self.row_ptr,
			col_nnz: self.row_nnz,
			row_idx: self.col_idx,
		}
	}

	#[inline]
	pub fn to_owned(&self) -> Result<SymbolicSparseRowMat<I, Rows, Cols>, FaerError> {
		Ok(self.transpose().to_owned()?.into_transpose())
	}

	#[inline]
	pub fn to_col_major(&self) -> Result<SymbolicSparseColMat<I, Rows, Cols>, FaerError> {
		Ok(self.transpose().to_row_major()?.into_transpose())
	}

	#[inline]
	pub fn compute_nnz(&self) -> usize {
		self.transpose().compute_nnz()
	}

	#[inline]
	pub fn row_ptr(&self) -> &'a [I] {
		self.row_ptr
	}

	#[inline]
	pub fn row_nnz(&self) -> Option<&'a [I]> {
		self.row_nnz
	}

	#[inline]
	pub fn col_idx(&self) -> &'a [I] {
		self.col_idx
	}

	#[inline]
	#[track_caller]
	pub fn row_range(&self, i: Idx<Rows>) -> Range<usize> {
		assert!(i < self.nrows());
		unsafe { self.row_range_unchecked(i) }
	}

	#[inline]
	#[track_caller]
	pub unsafe fn row_range_unchecked(&self, i: Idx<Rows>) -> Range<usize> {
		debug_assert!(i < self.nrows());

		self.transpose().col_range_unchecked(i)
	}

	#[inline]
	#[track_caller]
	pub fn col_idx_of_row_raw(&self, i: Idx<Rows>) -> &'a [Idx<Cols, I>] {
		unsafe {
			let slice = self.col_idx.get_unchecked(self.row_range(i));
			let len = slice.len();
			core::slice::from_raw_parts(slice.as_ptr() as *const Idx<Cols, I>, len)
		}
	}

	#[inline]
	#[track_caller]
	pub fn col_idx_of_row(&self, i: Idx<Rows>) -> impl 'a + Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Cols>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		self.col_idx_of_row_raw(i)
			.iter()
			.map(|&j| unsafe { Idx::<Cols>::new_unbound(j.unbound().zx()) })
	}

	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SymbolicSparseRowMatRef<'a, I, V, H> {
		assert!(all(self.nrows.unbound() == nrows.unbound(), self.ncols.unbound() == ncols.unbound()));

		SymbolicSparseRowMatRef {
			nrows,
			ncols,
			row_ptr: self.row_ptr,
			row_nnz: self.row_nnz,
			col_idx: self.col_idx,
		}
	}

	#[inline]
	pub fn as_dyn(self) -> SymbolicSparseRowMatRef<'a, I> {
		SymbolicSparseRowMatRef {
			nrows: self.nrows.unbound(),
			ncols: self.ncols.unbound(),
			row_ptr: self.row_ptr,
			row_nnz: self.row_nnz,
			col_idx: self.col_idx,
		}
	}
}

impl<Rows: Shape, Cols: Shape, I: Index> SymbolicSparseRowMat<I, Rows, Cols> {
	#[inline]
	#[track_caller]
	pub unsafe fn new_unchecked(
		nrows: Rows,
		ncols: Cols,
		row_ptr: alloc::vec::Vec<I>,
		row_nnz: Option<alloc::vec::Vec<I>>,
		col_idx: alloc::vec::Vec<I>,
	) -> Self {
		assume_row_ptr(nrows.unbound(), ncols.unbound(), &row_ptr, row_nnz.as_deref(), &col_idx);

		Self {
			nrows,
			ncols,
			row_ptr,
			row_nnz,
			col_idx,
		}
	}

	#[inline]
	#[track_caller]
	pub fn new_checked(
		nrows: Rows,
		ncols: Cols,
		row_ptr: alloc::vec::Vec<I>,
		row_nnz: Option<alloc::vec::Vec<I>>,
		col_idx: alloc::vec::Vec<I>,
	) -> Self {
		check_row_ptr(nrows.unbound(), ncols.unbound(), &row_ptr, row_nnz.as_deref(), &col_idx);
		check_col_idx(nrows.unbound(), ncols.unbound(), &row_ptr, row_nnz.as_deref(), &col_idx);

		Self {
			nrows,
			ncols,
			row_ptr,
			row_nnz,
			col_idx,
		}
	}

	#[inline]
	#[track_caller]
	pub fn new_unsorted_checked(
		nrows: Rows,
		ncols: Cols,
		row_ptr: alloc::vec::Vec<I>,
		row_nnz: Option<alloc::vec::Vec<I>>,
		col_idx: alloc::vec::Vec<I>,
	) -> Self {
		check_row_ptr(nrows.unbound(), ncols.unbound(), &row_ptr, row_nnz.as_deref(), &col_idx);
		check_col_idx_unsorted(nrows.unbound(), ncols.unbound(), &row_ptr, row_nnz.as_deref(), &col_idx);

		Self {
			nrows,
			ncols,
			row_ptr,
			row_nnz,
			col_idx,
		}
	}

	#[inline]
	pub fn parts(&self) -> (Rows, Cols, &'_ [I], Option<&'_ [I]>, &'_ [I]) {
		(self.nrows, self.ncols, &self.row_ptr, self.row_nnz.as_deref(), &self.col_idx)
	}

	#[inline]
	pub fn into_parts(self) -> (Rows, Cols, alloc::vec::Vec<I>, Option<alloc::vec::Vec<I>>, alloc::vec::Vec<I>) {
		(self.nrows, self.ncols, self.row_ptr, self.row_nnz, self.col_idx)
	}

	#[inline]
	pub fn nrows(&self) -> Rows {
		self.nrows
	}

	#[inline]
	pub fn ncols(&self) -> Cols {
		self.ncols
	}

	#[inline]
	pub fn shape(&self) -> (Rows, Cols) {
		(self.nrows, self.ncols)
	}

	#[inline]
	pub fn transpose(&self) -> SymbolicSparseColMatRef<'_, I, Cols, Rows> {
		self.rb().transpose()
	}

	#[inline]
	pub fn into_transpose(self) -> SymbolicSparseColMat<I, Cols, Rows> {
		SymbolicSparseColMat {
			nrows: self.ncols,
			ncols: self.nrows,
			col_ptr: self.row_ptr,
			col_nnz: self.row_nnz,
			row_idx: self.col_idx,
		}
	}

	#[inline]
	pub fn to_owned(&self) -> Result<SymbolicSparseRowMat<I, Rows, Cols>, FaerError> {
		self.rb().to_owned()
	}

	#[inline]
	pub fn to_col_major(&self) -> Result<SymbolicSparseColMat<I, Rows, Cols>, FaerError> {
		self.rb().to_col_major()
	}

	#[inline]
	pub fn compute_nnz(&self) -> usize {
		self.rb().compute_nnz()
	}

	#[inline]
	pub fn row_ptr(&self) -> &'_ [I] {
		&self.row_ptr
	}

	#[inline]
	pub fn row_nnz(&self) -> Option<&'_ [I]> {
		self.row_nnz.as_deref()
	}

	#[inline]
	pub fn col_idx(&self) -> &'_ [I] {
		&self.col_idx
	}

	#[inline]
	#[track_caller]
	pub fn row_range(&self, i: Idx<Rows>) -> Range<usize> {
		self.rb().row_range(i)
	}

	#[inline]
	#[track_caller]
	pub unsafe fn row_range_unchecked(&self, i: Idx<Rows>) -> Range<usize> {
		self.rb().row_range_unchecked(i)
	}

	#[inline]
	#[track_caller]
	pub fn col_idx_of_row_raw(&self, i: Idx<Rows>) -> &'_ [Idx<Cols, I>] {
		self.rb().col_idx_of_row_raw(i)
	}

	#[inline]
	#[track_caller]
	pub fn col_idx_of_row(&self, i: Idx<Rows>) -> impl '_ + Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Cols>> {
		self.rb().col_idx_of_row(i)
	}

	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(&self, nrows: V, ncols: H) -> SymbolicSparseRowMatRef<'_, I, V, H> {
		self.rb().as_shape(nrows, ncols)
	}

	#[inline]
	#[track_caller]
	pub fn into_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SymbolicSparseRowMat<I, V, H> {
		assert!(all(self.nrows().unbound() == nrows.unbound(), self.ncols().unbound() == ncols.unbound()));
		SymbolicSparseRowMat {
			nrows,
			ncols,
			row_ptr: self.row_ptr,
			row_nnz: self.row_nnz,
			col_idx: self.col_idx,
		}
	}

	#[inline]
	pub fn as_dyn(&self) -> SymbolicSparseRowMatRef<'_, I> {
		self.rb().as_dyn()
	}

	#[inline]
	pub fn into_dyn(self) -> SymbolicSparseRowMat<I> {
		SymbolicSparseRowMat {
			nrows: self.nrows.unbound(),
			ncols: self.ncols.unbound(),
			row_ptr: self.row_ptr,
			row_nnz: self.row_nnz,
			col_idx: self.col_idx,
		}
	}

	#[inline]
	pub fn try_new_from_indices(nrows: Rows, ncols: Cols, idx: &[Pair<Idx<Rows, I>, Idx<Cols, I>>]) -> Result<(Self, Argsort<I>), CreationError> {
		let (symbolic, argsort) = SymbolicSparseColMat::try_new_from_indices_impl(
			ncols,
			nrows,
			|i| Pair {
				row: idx[i].col,
				col: idx[i].row,
			},
			|_, _| false,
			idx.len(),
		)?;

		Ok((symbolic.into_transpose(), argsort))
	}

	#[inline]
	pub fn try_new_from_nonnegative_indices(
		nrows: Rows,
		ncols: Cols,
		idx: &[Pair<MaybeIdx<Rows, I>, MaybeIdx<Cols, I>>],
	) -> Result<(Self, Argsort<I>), CreationError> {
		let (symbolic, argsort) = SymbolicSparseColMat::try_new_from_indices_impl(
			ncols,
			nrows,
			|i| Pair {
				row: unsafe { Idx::<Cols, I>::new_unbound(I::from_signed(idx[i].col.unbound())) },
				col: unsafe { Idx::<Rows, I>::new_unbound(I::from_signed(idx[i].row.unbound())) },
			},
			|row, col| {
				let row = row.unbound().to_signed();
				let col = col.unbound().to_signed();
				let zero = I::Signed::truncate(0);

				row < zero || col < zero
			},
			idx.len(),
		)?;
		Ok((symbolic.into_transpose(), argsort))
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> SparseRowMatRef<'a, I, T, Rows, Cols> {
	#[inline]
	#[track_caller]
	pub fn new(symbolic: SymbolicSparseRowMatRef<'a, I, Rows, Cols>, val: &'a [T]) -> Self {
		assert!(symbolic.col_idx().len() == val.len());
		Self { symbolic, val }
	}

	#[inline]
	pub fn parts(self) -> (SymbolicSparseRowMatRef<'a, I, Rows, Cols>, &'a [T]) {
		(self.symbolic, self.val)
	}

	#[inline]
	pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
		self.symbolic
	}

	#[inline]
	pub fn val(self) -> &'a [T] {
		self.val
	}

	#[inline]
	#[track_caller]
	pub fn val_of_row(self, i: Idx<Rows>) -> &'a [T] {
		unsafe { self.val.get_unchecked(self.row_range(i)) }
	}

	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseRowMatRef<'a, I, T, V, H> {
		SparseRowMatRef {
			symbolic: self.symbolic.as_shape(nrows, ncols),
			val: self.val,
		}
	}

	/// returns a reference to the value at the given index, or `None` if the symbolic structure
	/// doesn't contain it, or contains multiple indices with the given index
	///
	/// # panics
	/// - panics if `row >= self.nrows()`
	/// - panics if `col >= self.ncols()`
	#[track_caller]
	pub fn get(self, row: Idx<Rows>, col: Idx<Cols>) -> Option<&'a T> {
		assert!(row < self.nrows());
		assert!(col < self.ncols());
		let col = I::truncate(col.unbound());
		let rowl = row.unbound();
		let start = self.symbolic().as_dyn().col_idx_of_row_raw(rowl).partition_point(partition_by_lt(col));
		let end = start + self.symbolic().as_dyn().col_idx_of_row_raw(rowl)[start..].partition_point(partition_by_le(col));

		if end == start + 1 { Some(&self.val_of_row(row)[start]) } else { None }
	}

	#[inline]
	pub fn as_dyn(self) -> SparseRowMatRef<'a, I, T> {
		SparseRowMatRef {
			symbolic: self.symbolic.as_dyn(),
			val: self.val,
		}
	}

	#[inline]
	pub fn transpose(self) -> SparseColMatRef<'a, I, T, Cols, Rows> {
		SparseColMatRef {
			symbolic: self.symbolic.transpose(),
			val: self.val,
		}
	}

	#[inline]
	pub fn conjugate(self) -> SparseRowMatRef<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.transpose().conjugate().transpose()
	}

	#[inline]
	pub fn adjoint(self) -> SparseColMatRef<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.transpose().conjugate()
	}

	#[inline]
	pub fn canonical(self) -> SparseRowMatRef<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		self.transpose().canonical().transpose()
	}

	#[inline]
	pub fn to_col_major(&self) -> Result<SparseColMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate,
	{
		Ok(self.transpose().to_row_major()?.into_transpose())
	}

	#[inline]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		fn imp<'ROWS, 'COLS, I: Index, T: Conjugate>(
			src: SparseRowMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
		) -> Mat<T::Canonical, Dim<'ROWS>, Dim<'COLS>> {
			let src = src.canonical();

			let mut out = Mat::zeros(src.nrows(), src.ncols());
			let M = src.nrows();

			for i in M.indices() {
				for (j, val) in iter::zip(src.col_idx_of_row(i), src.val_of_row(i)) {
					if try_const! { Conj::get::<T>().is_conj() } {
						out[(i, j)] = add(&out[(i, j)], &conj(val));
					} else {
						out[(i, j)] = add(&out[(i, j)], val);
					}
				}
			}

			out
		}
		with_dim!(ROWS, self.nrows().unbound());
		with_dim!(COLS, self.ncols().unbound());
		let this = self.as_shape(ROWS, COLS);

		imp(this).into_shape(self.nrows(), self.ncols())
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> SparseRowMatMut<'a, I, T, Rows, Cols> {
	#[inline]
	#[track_caller]
	pub fn new(symbolic: SymbolicSparseRowMatRef<'a, I, Rows, Cols>, val: &'a mut [T]) -> Self {
		assert!(symbolic.col_idx().len() == val.len());
		Self { symbolic, val }
	}

	#[inline]
	pub fn parts(self) -> (SymbolicSparseRowMatRef<'a, I, Rows, Cols>, &'a [T]) {
		(self.symbolic, self.val)
	}

	#[inline]
	pub fn parts_mut(self) -> (SymbolicSparseRowMatRef<'a, I, Rows, Cols>, &'a mut [T]) {
		(self.symbolic, self.val)
	}

	#[inline]
	pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
		self.symbolic
	}

	#[inline]
	pub fn val(self) -> &'a [T] {
		self.val
	}

	#[inline]
	pub fn val_mut(self) -> &'a mut [T] {
		self.val
	}

	#[inline]
	#[track_caller]
	pub fn val_of_row(self, i: Idx<Rows>) -> &'a [T] {
		unsafe { self.val.get_unchecked(self.row_range(i)) }
	}

	#[inline]
	#[track_caller]
	pub fn val_of_row_mut(self, j: Idx<Rows>) -> &'a mut [T] {
		unsafe { self.val.get_unchecked_mut(self.row_range(j)) }
	}

	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseRowMatRef<'a, I, T, V, H> {
		SparseRowMatRef {
			symbolic: self.symbolic.as_shape(nrows, ncols),
			val: self.val,
		}
	}

	#[inline]
	#[track_caller]
	pub fn as_shape_mut<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseRowMatMut<'a, I, T, V, H> {
		SparseRowMatMut {
			symbolic: self.symbolic.as_shape(nrows, ncols),
			val: self.val,
		}
	}

	/// returns a reference to the value at the given index, or `None` if the symbolic structure
	/// doesn't contain it, or contains multiple indices with the given index
	///
	/// # panics
	/// - panics if `row >= self.nrows()`
	/// - panics if `col >= self.ncols()`
	#[track_caller]
	pub fn get_mut(self, row: Idx<Rows>, col: Idx<Cols>) -> Option<&'a mut T> {
		assert!(row < self.nrows());
		assert!(col < self.ncols());
		let col = I::truncate(col.unbound());
		let rowl = row.unbound();
		let start = self.symbolic().as_dyn().col_idx_of_row_raw(rowl).partition_point(partition_by_lt(col));
		let end = start + self.symbolic().as_dyn().col_idx_of_row_raw(rowl)[start..].partition_point(partition_by_le(col));

		if end == start + 1 {
			Some(&mut self.val_of_row_mut(row)[start])
		} else {
			None
		}
	}

	#[inline]
	pub fn as_dyn(self) -> SparseRowMatRef<'a, I, T> {
		SparseRowMatRef {
			symbolic: self.symbolic.as_dyn(),
			val: self.val,
		}
	}

	#[inline]
	pub fn as_dyn_mut(self) -> SparseRowMatMut<'a, I, T> {
		SparseRowMatMut {
			symbolic: self.symbolic.as_dyn(),
			val: self.val,
		}
	}

	#[inline]
	pub fn transpose(self) -> SparseColMatRef<'a, I, T, Cols, Rows> {
		SparseColMatRef {
			symbolic: self.symbolic.transpose(),
			val: self.val,
		}
	}

	#[inline]
	pub fn transpose_mut(self) -> SparseColMatMut<'a, I, T, Cols, Rows> {
		SparseColMatMut {
			symbolic: self.symbolic.transpose(),
			val: self.val,
		}
	}

	#[inline]
	pub fn conjugate(self) -> SparseRowMatRef<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseRowMatRef {
			symbolic: self.symbolic,
			val: unsafe { core::slice::from_raw_parts(self.val.as_ptr() as *const T::Conj, len) },
		}
	}

	#[inline]
	pub fn conjugate_mut(self) -> SparseRowMatMut<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseRowMatMut {
			symbolic: self.symbolic,
			val: unsafe { core::slice::from_raw_parts_mut(self.val.as_mut_ptr() as *mut T::Conj, len) },
		}
	}

	#[inline]
	pub fn adjoint(self) -> SparseColMatRef<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline]
	pub fn adjoint_mut(self) -> SparseColMatMut<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate_mut().transpose_mut()
	}

	#[inline]
	pub fn canonical(self) -> SparseRowMatRef<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseRowMatRef {
			symbolic: self.symbolic,
			val: unsafe { core::slice::from_raw_parts(self.val.as_ptr() as *const T::Canonical, len) },
		}
	}

	#[inline]
	pub fn canonical_mut(self) -> SparseRowMatMut<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseRowMatMut {
			symbolic: self.symbolic,
			val: unsafe { core::slice::from_raw_parts_mut(self.val.as_mut_ptr() as *mut T::Canonical, len) },
		}
	}

	#[inline]
	pub fn to_col_major(&self) -> Result<SparseColMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate,
	{
		self.rb().to_col_major()
	}

	#[inline]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb().to_dense()
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T> SparseRowMat<I, T, Rows, Cols> {
	#[inline]
	#[track_caller]
	pub fn new(symbolic: SymbolicSparseRowMat<I, Rows, Cols>, val: alloc::vec::Vec<T>) -> Self {
		assert!(symbolic.col_idx().len() == val.len());
		Self { symbolic, val }
	}

	#[inline]
	pub fn parts(&self) -> (SymbolicSparseRowMatRef<'_, I, Rows, Cols>, &'_ [T]) {
		(self.symbolic.rb(), &self.val)
	}

	#[inline]
	pub fn parts_mut(&mut self) -> (SymbolicSparseRowMatRef<'_, I, Rows, Cols>, &'_ mut [T]) {
		(self.symbolic.rb(), &mut self.val)
	}

	#[inline]
	pub fn into_parts(self) -> (SymbolicSparseRowMat<I, Rows, Cols>, alloc::vec::Vec<T>) {
		(self.symbolic, self.val)
	}

	#[inline]
	pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'_, I, Rows, Cols> {
		self.symbolic.rb()
	}

	#[inline]
	pub fn val(&self) -> &'_ [T] {
		&self.val
	}

	#[inline]
	pub fn val_mut(&mut self) -> &'_ mut [T] {
		&mut self.val
	}

	#[inline]
	#[track_caller]
	pub fn val_of_row(&self, j: Idx<Rows>) -> &'_ [T] {
		unsafe { self.val.get_unchecked(self.row_range(j)) }
	}

	#[inline]
	#[track_caller]
	pub fn val_of_row_mut(&mut self, j: Idx<Rows>) -> &'_ mut [T] {
		unsafe { self.val.get_unchecked_mut(self.symbolic.row_range(j)) }
	}

	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(&self, nrows: V, ncols: H) -> SparseRowMatRef<'_, I, T, V, H> {
		SparseRowMatRef {
			symbolic: self.symbolic.as_shape(nrows, ncols),
			val: &self.val,
		}
	}

	#[inline]
	#[track_caller]
	pub fn as_shape_mut<V: Shape, H: Shape>(&mut self, nrows: V, ncols: H) -> SparseRowMatMut<'_, I, T, V, H> {
		SparseRowMatMut {
			symbolic: self.symbolic.as_shape(nrows, ncols),
			val: &mut self.val,
		}
	}

	#[inline]
	#[track_caller]
	pub fn into_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseRowMat<I, T, V, H> {
		SparseRowMat {
			symbolic: self.symbolic.into_shape(nrows, ncols),
			val: self.val,
		}
	}

	#[inline]
	pub fn as_dyn(&self) -> SparseRowMatRef<'_, I, T> {
		SparseRowMatRef {
			symbolic: self.symbolic.as_dyn(),
			val: &self.val,
		}
	}

	#[inline]
	pub fn as_dyn_mut(&mut self) -> SparseRowMatMut<'_, I, T> {
		SparseRowMatMut {
			symbolic: self.symbolic.as_dyn(),
			val: &mut self.val,
		}
	}

	#[inline]
	pub fn into_dyn(self) -> SparseRowMat<I, T> {
		SparseRowMat {
			symbolic: self.symbolic.into_dyn(),
			val: self.val,
		}
	}

	#[inline]
	pub fn transpose(&self) -> SparseColMatRef<'_, I, T, Cols, Rows> {
		SparseColMatRef {
			symbolic: self.symbolic.transpose(),
			val: &self.val,
		}
	}

	#[inline]
	pub fn transpose_mut(&mut self) -> SparseColMatMut<'_, I, T, Cols, Rows> {
		SparseColMatMut {
			symbolic: self.symbolic.transpose(),
			val: &mut self.val,
		}
	}

	#[inline]
	pub fn into_transpose(self) -> SparseColMat<I, T, Cols, Rows> {
		SparseColMat {
			symbolic: self.symbolic.into_transpose(),
			val: self.val,
		}
	}

	#[inline]
	pub fn conjugate(&self) -> SparseRowMatRef<'_, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb().conjugate()
	}

	#[inline]
	pub fn conjugate_mut(&mut self) -> SparseRowMatMut<'_, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb_mut().conjugate_mut()
	}

	#[inline]
	pub fn into_conjugate(self) -> SparseRowMat<I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let mut vec = core::mem::ManuallyDrop::new(self.val);
		let len = vec.len();
		let cap = vec.capacity();
		let ptr = vec.as_mut_ptr();

		SparseRowMat {
			symbolic: self.symbolic,
			val: unsafe { alloc::vec::Vec::from_raw_parts(ptr as *mut T::Conj, len, cap) },
		}
	}

	#[inline]
	pub fn adjoint(&self) -> SparseColMatRef<'_, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline]
	pub fn adjoint_mut(&mut self) -> SparseColMatMut<'_, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate_mut().transpose_mut()
	}

	#[inline]
	pub fn into_adjoint(self) -> SparseColMat<I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.into_conjugate().into_transpose()
	}

	#[inline]
	pub fn canonical(&self) -> SparseRowMatRef<'_, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseRowMatRef {
			symbolic: self.symbolic.rb(),
			val: unsafe { core::slice::from_raw_parts(self.val.as_ptr() as *const T::Canonical, len) },
		}
	}

	#[inline]
	pub fn canonical_mut(&mut self) -> SparseRowMatMut<'_, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseRowMatMut {
			symbolic: self.symbolic.rb(),
			val: unsafe { core::slice::from_raw_parts_mut(self.val.as_mut_ptr() as *mut T::Canonical, len) },
		}
	}

	#[inline]
	pub fn into_canonical(self) -> SparseRowMat<I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let mut vec = core::mem::ManuallyDrop::new(self.val);
		let len = vec.len();
		let cap = vec.capacity();
		let ptr = vec.as_mut_ptr();

		SparseRowMat {
			symbolic: self.symbolic,
			val: unsafe { alloc::vec::Vec::from_raw_parts(ptr as *mut T::Canonical, len, cap) },
		}
	}

	#[inline]
	pub fn to_col_major(&self) -> Result<SparseColMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate,
	{
		self.rb().to_col_major()
	}

	#[inline]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb().to_dense()
	}

	#[track_caller]
	pub fn new_from_argsort(symbolic: SymbolicSparseRowMat<I, Rows, Cols>, argsort: &Argsort<I>, val: &[T]) -> Result<Self, FaerError>
	where
		T: ComplexField,
	{
		Ok(SparseColMat::new_from_argsort(symbolic.into_transpose(), argsort, val)?.into_transpose())
	}

	#[track_caller]
	pub fn try_new_from_triplets(nrows: Rows, ncols: Cols, entries: &[Triplet<Idx<Rows, I>, Idx<Cols, I>, T>]) -> Result<Self, CreationError>
	where
		T: ComplexField,
	{
		let (symbolic, argsort) = SymbolicSparseColMat::try_new_from_indices_impl(
			ncols,
			nrows,
			|i| Pair {
				row: entries[i].col,
				col: entries[i].row,
			},
			|_, _| false,
			entries.len(),
		)?;

		Ok(SparseColMat::new_from_argsort_impl(symbolic, &argsort, |i| entries[i].val.clone(), entries.len())?.into_transpose())
	}

	#[track_caller]
	pub fn try_new_from_nonnegative_triplets(
		nrows: Rows,
		ncols: Cols,
		entries: &[Triplet<MaybeIdx<Rows, I>, MaybeIdx<Cols, I>, T>],
	) -> Result<Self, CreationError>
	where
		T: ComplexField,
	{
		let (symbolic, argsort) = SymbolicSparseColMat::try_new_from_indices_impl(
			ncols,
			nrows,
			|i| Pair {
				row: unsafe { Idx::<Cols, I>::new_unbound(I::from_signed(entries[i].col.unbound())) },
				col: unsafe { Idx::<Rows, I>::new_unbound(I::from_signed(entries[i].row.unbound())) },
			},
			|row, col| {
				let row = row.unbound().to_signed();
				let col = col.unbound().to_signed();
				let zero = I::Signed::truncate(0);
				row < zero || col < zero
			},
			entries.len(),
		)?;

		Ok(SparseColMat::new_from_argsort_impl(symbolic, &argsort, |i| entries[i].val.clone(), entries.len())?.into_transpose())
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseRowMatRef<'a, I, T, Rows, Cols> {
	type Target = SymbolicSparseRowMatRef<'a, I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.symbolic
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseRowMatMut<'a, I, T, Rows, Cols> {
	type Target = SymbolicSparseRowMatRef<'a, I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.symbolic
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseRowMat<I, T, Rows, Cols> {
	type Target = SymbolicSparseRowMat<I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.symbolic
	}
}

impl<Rows: Shape, Cols: Shape, I: Index> fmt::Debug for SymbolicSparseRowMatRef<'_, I, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fn imp<'ROWS, 'COLS, I: Index>(mat: SymbolicSparseRowMatRef<'_, I, Dim<'ROWS>, Dim<'COLS>>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
			struct Entries<'a, 'ROWS, 'COLS, I>(SymbolicSparseRowMatRef<'a, I, Dim<'ROWS>, Dim<'COLS>>);

			impl<'ROWS, 'COLS, I: Index> fmt::Debug for Entries<'_, 'ROWS, 'COLS, I> {
				fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
					let mat = self.0;

					f.debug_list()
						.entries(
							mat.nrows()
								.indices()
								.flat_map(|row| mat.col_idx_of_row(row).map(move |col| Pair { row, col })),
						)
						.finish()
				}
			}

			f.debug_struct("SymbolicSparseRowMat")
				.field("nrows", &mat.nrows)
				.field("ncols", &mat.ncols)
				.field("entries", &Entries(mat))
				.finish()
		}
		with_dim!(ROWS, self.nrows().unbound());
		with_dim!(COLS, self.ncols().unbound());

		imp(self.as_shape(ROWS, COLS), f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index> fmt::Debug for SymbolicSparseRowMat<I, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.rb().fmt(f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for SparseRowMatRef<'_, I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fn imp<'ROWS, 'COLS, I: Index, T: fmt::Debug>(
			mat: SparseRowMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>,
			f: &mut fmt::Formatter<'_>,
		) -> fmt::Result {
			struct Entries<'a, 'ROWS, 'COLS, I, T>(SparseRowMatRef<'a, I, T, Dim<'ROWS>, Dim<'COLS>>);

			impl<'ROWS, 'COLS, I: Index, T: fmt::Debug> fmt::Debug for Entries<'_, 'ROWS, 'COLS, I, T> {
				fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
					let mat = self.0;

					f.debug_list()
						.entries(mat.nrows().indices().flat_map(|row| {
							let col_idx = mat.col_idx_of_row(row);
							let val = mat.val_of_row(row);

							iter::zip(col_idx, val).map(move |(col, val)| Triplet {
								row,
								col,
								val: crate::hacks::hijack_debug(val),
							})
						}))
						.finish()
				}
			}

			f.debug_struct("SparseRowMat")
				.field("nrows", &mat.nrows)
				.field("ncols", &mat.ncols)
				.field("entries", &Entries(mat))
				.finish()
		}

		with_dim!(ROWS, self.nrows().unbound());
		with_dim!(COLS, self.ncols().unbound());

		imp(self.as_shape(ROWS, COLS), f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for SparseRowMatMut<'_, I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.rb().fmt(f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for SparseRowMat<I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.rb().fmt(f)
	}
}
