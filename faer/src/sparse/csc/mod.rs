use super::*;
use crate::internal_prelude::*;
use crate::{Idx, assert, debug_assert};
use core::ops::Range;
use core::{fmt, iter};

pub struct SymbolicSparseColMatRef<'a, I, Rows = usize, Cols = usize> {
	pub(super) nrows: Rows,
	pub(super) ncols: Cols,
	pub(super) col_ptr: &'a [I],
	pub(super) col_nnz: Option<&'a [I]>,
	pub(super) row_idx: &'a [I],
}

pub struct SparseColMatRef<'a, I, T, Rows = usize, Cols = usize> {
	pub(super) symbolic: SymbolicSparseColMatRef<'a, I, Rows, Cols>,
	pub(super) val: &'a [T],
}

pub struct SparseColMatMut<'a, I, T, Rows = usize, Cols = usize> {
	pub(super) symbolic: SymbolicSparseColMatRef<'a, I, Rows, Cols>,
	pub(super) val: &'a mut [T],
}

#[derive(Clone)]
pub struct SymbolicSparseColMat<I, Rows = usize, Cols = usize> {
	pub(super) nrows: Rows,
	pub(super) ncols: Cols,
	pub(super) col_ptr: alloc::vec::Vec<I>,
	pub(super) col_nnz: Option<alloc::vec::Vec<I>>,
	pub(super) row_idx: alloc::vec::Vec<I>,
}

#[derive(Clone)]
pub struct SparseColMat<I, T, Rows = usize, Cols = usize> {
	pub(super) symbolic: SymbolicSparseColMat<I, Rows, Cols>,
	pub(super) val: alloc::vec::Vec<T>,
}

impl<'a, I, Rows: Copy, Cols: Copy> Copy for SymbolicSparseColMatRef<'a, I, Rows, Cols> {}
impl<'a, I, T, Rows: Copy, Cols: Copy> Copy for SparseColMatRef<'a, I, T, Rows, Cols> {}

impl<'a, I, Rows: Copy, Cols: Copy> Clone for SymbolicSparseColMatRef<'a, I, Rows, Cols> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
impl<'a, I, T, Rows: Copy, Cols: Copy> Clone for SparseColMatRef<'a, I, T, Rows, Cols> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

impl<'a, I, Rows: Copy, Cols: Copy> IntoConst for SymbolicSparseColMatRef<'a, I, Rows, Cols> {
	type Target = SymbolicSparseColMatRef<'a, I, Rows, Cols>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}

impl<'short, 'a, I, Rows: Copy, Cols: Copy> ReborrowMut<'short> for SymbolicSparseColMatRef<'a, I, Rows, Cols> {
	type Target = SymbolicSparseColMatRef<'short, I, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}

impl<'short, 'a, I, Rows: Copy, Cols: Copy> Reborrow<'short> for SymbolicSparseColMatRef<'a, I, Rows, Cols> {
	type Target = SymbolicSparseColMatRef<'short, I, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}

impl<'a, I, T, Rows: Copy, Cols: Copy> IntoConst for SparseColMatRef<'a, I, T, Rows, Cols> {
	type Target = SparseColMatRef<'a, I, T, Rows, Cols>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> ReborrowMut<'short> for SparseColMatRef<'a, I, T, Rows, Cols> {
	type Target = SparseColMatRef<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> Reborrow<'short> for SparseColMatRef<'a, I, T, Rows, Cols> {
	type Target = SparseColMatRef<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}

impl<'a, I, T, Rows: Copy, Cols: Copy> IntoConst for SparseColMatMut<'a, I, T, Rows, Cols> {
	type Target = SparseColMatRef<'a, I, T, Rows, Cols>;

	#[inline]
	fn into_const(self) -> Self::Target {
		SparseColMatRef {
			symbolic: self.symbolic,
			val: self.val,
		}
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> ReborrowMut<'short> for SparseColMatMut<'a, I, T, Rows, Cols> {
	type Target = SparseColMatMut<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		SparseColMatMut {
			symbolic: self.symbolic,
			val: self.val,
		}
	}
}

impl<'short, 'a, I, T, Rows: Copy, Cols: Copy> Reborrow<'short> for SparseColMatMut<'a, I, T, Rows, Cols> {
	type Target = SparseColMatRef<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		SparseColMatRef {
			symbolic: self.symbolic,
			val: self.val,
		}
	}
}

impl<'short, I, T, Rows: Copy, Cols: Copy> ReborrowMut<'short> for SparseColMat<I, T, Rows, Cols> {
	type Target = SparseColMatMut<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		SparseColMatMut {
			symbolic: self.symbolic.rb(),
			val: &mut self.val,
		}
	}
}

impl<'short, I, T, Rows: Copy, Cols: Copy> Reborrow<'short> for SparseColMat<I, T, Rows, Cols> {
	type Target = SparseColMatRef<'short, I, T, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		SparseColMatRef {
			symbolic: self.symbolic.rb(),
			val: &self.val,
		}
	}
}

impl<'short, I, Rows: Copy, Cols: Copy> Reborrow<'short> for SymbolicSparseColMat<I, Rows, Cols> {
	type Target = SymbolicSparseColMatRef<'short, I, Rows, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		SymbolicSparseColMatRef {
			nrows: self.nrows,
			ncols: self.ncols,
			col_ptr: &self.col_ptr,
			col_nnz: self.col_nnz.as_deref(),
			row_idx: &self.row_idx,
		}
	}
}

#[inline(always)]
#[track_caller]
fn assume_col_ptr<I: Index>(nrows: usize, ncols: usize, col_ptr: &[I], col_nnz: Option<&[I]>, row_idx: &[I]) {
	assert!(all(ncols <= I::Signed::MAX.zx(), nrows <= I::Signed::MAX.zx(),));
	assert!(col_ptr.len() == ncols + 1);
	assert!(col_ptr[ncols].zx() <= row_idx.len());
	if let Some(nnz_per_row) = col_nnz {
		assert!(nnz_per_row.len() == ncols);
	}
}

#[track_caller]
fn check_col_ptr<I: Index>(nrows: usize, ncols: usize, col_ptr: &[I], col_nnz: Option<&[I]>, row_idx: &[I]) {
	assert!(all(ncols <= I::Signed::MAX.zx(), nrows <= I::Signed::MAX.zx(),));
	assert!(col_ptr.len() == ncols + 1);
	if let Some(nnz_per_col) = col_nnz {
		assert!(nnz_per_col.len() == ncols);
		for (&nnz_j, &[col, col_next]) in iter::zip(nnz_per_col, windows2(col_ptr)) {
			assert!(col <= col_next);
			assert!(nnz_j <= col_next - col);
		}
	} else {
		for &[col, col_next] in windows2(col_ptr) {
			assert!(col <= col_next);
		}
	}
	assert!(col_ptr[ncols].zx() <= row_idx.len());
}

#[track_caller]
fn check_row_idx<I: Index>(nrows: usize, ncols: usize, col_ptr: &[I], col_nnz: Option<&[I]>, row_idx: &[I]) {
	_ = ncols;
	if let Some(col_nnz) = col_nnz {
		for (nnz, &c) in iter::zip(col_nnz, col_ptr) {
			let c = c.zx();
			let nnz = nnz.zx();
			let row_idx = &row_idx[c..c + nnz];
			if !row_idx.is_empty() {
				let mut i = row_idx[0].zx();
				for &i_next in &row_idx[1..] {
					let i_next = i_next.zx();
					assert!(i < i_next);
					i = i_next;
				}
				assert!(i < nrows);
			}
		}
	} else {
		for &[c, c_next] in windows2(col_ptr) {
			let row_idx = &row_idx[c.zx()..c_next.zx()];
			if !row_idx.is_empty() {
				let mut i = row_idx[0].zx();
				for &i_next in &row_idx[1..] {
					let i_next = i_next.zx();
					assert!(i < i_next);
					i = i_next;
				}
				assert!(i < nrows);
			}
		}
	}
}

#[track_caller]
fn check_row_idx_unsorted<I: Index>(nrows: usize, ncols: usize, col_ptr: &[I], col_nnz: Option<&[I]>, row_idx: &[I]) {
	_ = ncols;

	if let Some(col_nnz) = col_nnz {
		for (nnz, &c) in iter::zip(col_nnz, col_ptr) {
			let c = c.zx();
			let nnz = nnz.zx();
			for &i in &row_idx[c..c + nnz] {
				let i = i.zx();
				assert!(i < nrows);
			}
		}
	} else {
		for &[c, c_next] in windows2(col_ptr) {
			for &i in &row_idx[c.zx()..c_next.zx()] {
				let i = i.zx();
				assert!(i < nrows);
			}
		}
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index> SymbolicSparseColMatRef<'a, I, Rows, Cols> {
	#[inline]
	#[track_caller]
	pub unsafe fn new_unchecked(nrows: Rows, ncols: Cols, col_ptr: &'a [I], col_nnz: Option<&'a [I]>, row_idx: &'a [I]) -> Self {
		assume_col_ptr(nrows.unbound(), ncols.unbound(), col_ptr, col_nnz, row_idx);

		Self {
			nrows,
			ncols,
			col_ptr,
			col_nnz,
			row_idx,
		}
	}

	#[inline]
	#[track_caller]
	pub fn new_checked(nrows: Rows, ncols: Cols, col_ptr: &'a [I], col_nnz: Option<&'a [I]>, row_idx: &'a [I]) -> Self {
		check_col_ptr(nrows.unbound(), ncols.unbound(), col_ptr, col_nnz, row_idx);
		check_row_idx(nrows.unbound(), ncols.unbound(), col_ptr, col_nnz, row_idx);

		Self {
			nrows,
			ncols,
			col_ptr,
			col_nnz,
			row_idx,
		}
	}

	#[inline]
	#[track_caller]
	pub fn new_unsorted_checked(nrows: Rows, ncols: Cols, col_ptr: &'a [I], col_nnz: Option<&'a [I]>, row_idx: &'a [I]) -> Self {
		check_col_ptr(nrows.unbound(), ncols.unbound(), col_ptr, col_nnz, row_idx);
		check_row_idx_unsorted(nrows.unbound(), ncols.unbound(), col_ptr, col_nnz, row_idx);

		Self {
			nrows,
			ncols,
			col_ptr,
			col_nnz,
			row_idx,
		}
	}

	#[inline]
	pub fn into_parts(self) -> (Rows, Cols, &'a [I], Option<&'a [I]>, &'a [I]) {
		(self.nrows, self.ncols, self.col_ptr, self.col_nnz, self.row_idx)
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
	pub fn transpose(self) -> SymbolicSparseRowMatRef<'a, I, Cols, Rows> {
		SymbolicSparseRowMatRef {
			nrows: self.ncols,
			ncols: self.nrows,
			row_ptr: self.col_ptr,
			row_nnz: self.col_nnz,
			col_idx: self.row_idx,
		}
	}

	#[inline]
	pub fn to_owned(&self) -> Result<SymbolicSparseColMat<I, Rows, Cols>, FaerError> {
		Ok(SymbolicSparseColMat {
			nrows: self.nrows,
			ncols: self.ncols,
			col_ptr: try_collect(self.col_ptr.iter().copied())?,
			col_nnz: self.col_nnz.map(|col_nnz| try_collect(col_nnz.iter().copied())).transpose()?,
			row_idx: try_collect(self.row_idx.iter().copied())?,
		})
	}

	#[inline]
	pub fn to_row_major(&self) -> Result<SymbolicSparseRowMat<I, Rows, Cols>, FaerError> {
		todo!()
	}

	#[inline]
	pub fn compute_nnz(&self) -> usize {
		fn imp<I: Index>(col_ptr: &[I], col_nnz: Option<&[I]>, ncols: usize) -> usize {
			match col_nnz {
				Some(col_nnz) => {
					let mut nnz = 0usize;
					for &nnz_j in col_nnz {
						// can't overflow
						nnz += nnz_j.zx();
					}
					nnz
				},
				None => col_ptr[ncols].zx() - col_ptr[0].zx(),
			}
		}
		imp(self.col_ptr, self.col_nnz, self.ncols.unbound())
	}

	#[inline]
	pub fn col_ptr(&self) -> &'a [I] {
		self.col_ptr
	}

	#[inline]
	pub fn col_nnz(&self) -> Option<&'a [I]> {
		self.col_nnz
	}

	#[inline]
	pub fn row_idx(&self) -> &'a [I] {
		self.row_idx
	}

	#[inline]
	#[track_caller]
	pub fn col_range(&self, j: Idx<Cols>) -> Range<usize> {
		assert!(j < self.ncols());
		unsafe { self.col_range_unchecked(j) }
	}

	#[inline]
	#[track_caller]
	pub unsafe fn col_range_unchecked(&self, j: Idx<Cols>) -> Range<usize> {
		debug_assert!(j < self.ncols());
		let j = j.unbound();

		let start = self.col_ptr.get_unchecked(j).zx();
		let end = self.col_nnz.map(|col_nnz| (col_nnz.get_unchecked(j).zx() + start)).unwrap_or(self.col_ptr.get_unchecked(j + 1).zx());

		start..end
	}

	#[inline]
	#[track_caller]
	pub fn row_idx_of_col_raw(&self, j: Idx<Cols>) -> &'a [Idx<Rows, I>] {
		unsafe {
			let slice = self.row_idx.get_unchecked(self.col_range(j));
			let len = slice.len();
			core::slice::from_raw_parts(slice.as_ptr() as *const Idx<Rows, I>, len)
		}
	}

	#[inline]
	#[track_caller]
	pub fn row_idx_of_col(&self, j: Idx<Cols>) -> impl 'a + Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Rows>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		self.row_idx_of_col_raw(j).iter().map(|&i| unsafe { Idx::<Rows>::new_unbound(i.unbound().zx()) })
	}

	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SymbolicSparseColMatRef<'a, I, V, H> {
		assert!(all(self.nrows.unbound() == nrows.unbound(), self.ncols.unbound() == ncols.unbound()));

		SymbolicSparseColMatRef {
			nrows,
			ncols,
			col_ptr: self.col_ptr,
			col_nnz: self.col_nnz,
			row_idx: self.row_idx,
		}
	}

	#[inline]
	pub fn as_dyn(self) -> SymbolicSparseColMatRef<'a, I> {
		SymbolicSparseColMatRef {
			nrows: self.nrows.unbound(),
			ncols: self.ncols.unbound(),
			col_ptr: self.col_ptr,
			col_nnz: self.col_nnz,
			row_idx: self.row_idx,
		}
	}
}

impl<Rows: Shape, Cols: Shape, I: Index> SymbolicSparseColMat<I, Rows, Cols> {
	#[inline]
	#[track_caller]
	pub unsafe fn new_unchecked(nrows: Rows, ncols: Cols, col_ptr: alloc::vec::Vec<I>, col_nnz: Option<alloc::vec::Vec<I>>, row_idx: alloc::vec::Vec<I>) -> Self {
		assume_col_ptr(nrows.unbound(), ncols.unbound(), &col_ptr, col_nnz.as_deref(), &row_idx);

		Self {
			nrows,
			ncols,
			col_ptr,
			col_nnz,
			row_idx,
		}
	}

	#[inline]
	#[track_caller]
	pub fn new_checked(nrows: Rows, ncols: Cols, col_ptr: alloc::vec::Vec<I>, col_nnz: Option<alloc::vec::Vec<I>>, row_idx: alloc::vec::Vec<I>) -> Self {
		check_col_ptr(nrows.unbound(), ncols.unbound(), &col_ptr, col_nnz.as_deref(), &row_idx);
		check_row_idx(nrows.unbound(), ncols.unbound(), &col_ptr, col_nnz.as_deref(), &row_idx);

		Self {
			nrows,
			ncols,
			col_ptr,
			col_nnz,
			row_idx,
		}
	}

	#[inline]
	#[track_caller]
	pub fn new_unsorted_checked(nrows: Rows, ncols: Cols, col_ptr: alloc::vec::Vec<I>, col_nnz: Option<alloc::vec::Vec<I>>, row_idx: alloc::vec::Vec<I>) -> Self {
		check_col_ptr(nrows.unbound(), ncols.unbound(), &col_ptr, col_nnz.as_deref(), &row_idx);
		check_row_idx_unsorted(nrows.unbound(), ncols.unbound(), &col_ptr, col_nnz.as_deref(), &row_idx);

		Self {
			nrows,
			ncols,
			col_ptr,
			col_nnz,
			row_idx,
		}
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
	pub fn transpose(&self) -> SymbolicSparseRowMatRef<'_, I, Cols, Rows> {
		self.rb().transpose()
	}

	#[inline]
	pub fn into_transpose(self) -> SymbolicSparseRowMat<I, Cols, Rows> {
		SymbolicSparseRowMat {
			nrows: self.ncols,
			ncols: self.nrows,
			row_ptr: self.col_ptr,
			row_nnz: self.col_nnz,
			col_idx: self.row_idx,
		}
	}

	#[inline]
	pub fn to_owned(&self) -> Result<SymbolicSparseColMat<I, Rows, Cols>, FaerError> {
		self.rb().to_owned()
	}

	#[inline]
	pub fn to_row_major(&self) -> Result<SymbolicSparseRowMat<I, Rows, Cols>, FaerError> {
		self.rb().to_row_major()
	}

	#[inline]
	pub fn compute_nnz(&self) -> usize {
		self.rb().compute_nnz()
	}

	#[inline]
	pub fn col_ptr(&self) -> &'_ [I] {
		&self.col_ptr
	}

	#[inline]
	pub fn col_nnz(&self) -> Option<&'_ [I]> {
		self.col_nnz.as_deref()
	}

	#[inline]
	pub fn row_idx(&self) -> &'_ [I] {
		&self.row_idx
	}

	#[inline]
	#[track_caller]
	pub fn col_range(&self, j: Idx<Cols>) -> Range<usize> {
		self.rb().col_range(j)
	}

	#[inline]
	#[track_caller]
	pub unsafe fn col_range_unchecked(&self, j: Idx<Cols>) -> Range<usize> {
		self.rb().col_range_unchecked(j)
	}

	#[inline]
	#[track_caller]
	pub fn row_idx_of_col_raw(&self, j: Idx<Cols>) -> &'_ [Idx<Rows, I>] {
		self.rb().row_idx_of_col_raw(j)
	}

	#[inline]
	#[track_caller]
	pub fn row_idx_of_col(&self, j: Idx<Cols>) -> impl '_ + Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Rows>> {
		self.rb().row_idx_of_col(j)
	}

	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(&self, nrows: V, ncols: H) -> SymbolicSparseColMatRef<'_, I, V, H> {
		self.rb().as_shape(nrows, ncols)
	}

	#[inline]
	#[track_caller]
	pub fn into_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SymbolicSparseColMat<I, V, H> {
		assert!(all(self.nrows().unbound() == nrows.unbound(), self.ncols().unbound() == ncols.unbound()));
		SymbolicSparseColMat {
			nrows,
			ncols,
			col_ptr: self.col_ptr,
			col_nnz: self.col_nnz,
			row_idx: self.row_idx,
		}
	}

	#[inline]
	pub fn as_dyn(&self) -> SymbolicSparseColMatRef<'_, I> {
		self.rb().as_dyn()
	}

	#[inline]
	pub fn into_dyn(self) -> SymbolicSparseColMat<I> {
		SymbolicSparseColMat {
			nrows: self.nrows.unbound(),
			ncols: self.ncols.unbound(),
			col_ptr: self.col_ptr,
			col_nnz: self.col_nnz,
			row_idx: self.row_idx,
		}
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> SparseColMatRef<'a, I, T, Rows, Cols> {
	#[inline]
	#[track_caller]
	pub fn new(symbolic: SymbolicSparseColMatRef<'a, I, Rows, Cols>, val: &'a [T]) -> Self {
		assert!(symbolic.row_idx().len() == val.len());
		Self { symbolic, val }
	}

	#[inline]
	pub fn parts(self) -> (SymbolicSparseColMatRef<'a, I, Rows, Cols>, &'a [T]) {
		(self.symbolic, self.val)
	}

	#[inline]
	pub fn symbolic(&self) -> SymbolicSparseColMatRef<'a, I, Rows, Cols> {
		self.symbolic
	}

	#[inline]
	pub fn val(self) -> &'a [T] {
		self.val
	}

	#[inline]
	#[track_caller]
	pub fn val_of_col(self, j: Idx<Cols>) -> &'a [T] {
		unsafe { self.val.get_unchecked(self.col_range(j)) }
	}

	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseColMatRef<'a, I, T, V, H> {
		SparseColMatRef {
			symbolic: self.symbolic.as_shape(nrows, ncols),
			val: self.val,
		}
	}

	#[inline]
	pub fn as_dyn(self) -> SparseColMatRef<'a, I, T> {
		SparseColMatRef {
			symbolic: self.symbolic.as_dyn(),
			val: self.val,
		}
	}

	#[inline]
	pub fn transpose(self) -> SparseRowMatRef<'a, I, T, Cols, Rows> {
		SparseRowMatRef {
			symbolic: self.symbolic.transpose(),
			val: self.val,
		}
	}

	#[inline]
	pub fn conjugate(self) -> SparseColMatRef<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseColMatRef {
			symbolic: self.symbolic,
			val: unsafe { core::slice::from_raw_parts(self.val.as_ptr() as *const T::Conj, len) },
		}
	}

	#[inline]
	pub fn adjoint(self) -> SparseRowMatRef<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline]
	pub fn canonical(self) -> SparseColMatRef<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseColMatRef {
			symbolic: self.symbolic,
			val: unsafe { core::slice::from_raw_parts(self.val.as_ptr() as *const T::Canonical, len) },
		}
	}

	#[inline]
	pub fn to_row_major(&self) -> Result<SparseRowMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate<Canonical: ComplexField>,
	{
		todo!()
	}

	#[inline]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate<Canonical: ComplexField>,
	{
		fn imp<'ROWS, 'COLS, I: Index, T: Conjugate<Canonical: ComplexField>>(src: SparseColMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>) -> Mat<T::Canonical, Dim<'ROWS>, Dim<'COLS>> {
			let src = src.canonical();

			let mut out = Mat::zeros(src.nrows(), src.ncols());
			let N = src.ncols();

			for j in N.indices() {
				for (i, val) in iter::zip(src.row_idx_of_col(j), src.val_of_col(j)) {
					if const { Conj::get::<T>().is_conj() } {
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

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> SparseColMatMut<'a, I, T, Rows, Cols> {
	#[inline]
	#[track_caller]
	pub fn new(symbolic: SymbolicSparseColMatRef<'a, I, Rows, Cols>, val: &'a mut [T]) -> Self {
		assert!(symbolic.row_idx().len() == val.len());
		Self { symbolic, val }
	}

	#[inline]
	pub fn parts(self) -> (SymbolicSparseColMatRef<'a, I, Rows, Cols>, &'a [T]) {
		(self.symbolic, self.val)
	}

	#[inline]
	pub fn parts_mut(self) -> (SymbolicSparseColMatRef<'a, I, Rows, Cols>, &'a mut [T]) {
		(self.symbolic, self.val)
	}

	#[inline]
	pub fn symbolic(&self) -> SymbolicSparseColMatRef<'a, I, Rows, Cols> {
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
	pub fn val_of_col(self, j: Idx<Cols>) -> &'a [T] {
		unsafe { self.val.get_unchecked(self.col_range(j)) }
	}

	#[inline]
	#[track_caller]
	pub fn val_of_col_mut(self, j: Idx<Cols>) -> &'a mut [T] {
		unsafe { self.val.get_unchecked_mut(self.col_range(j)) }
	}

	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseColMatRef<'a, I, T, V, H> {
		SparseColMatRef {
			symbolic: self.symbolic.as_shape(nrows, ncols),
			val: self.val,
		}
	}

	#[inline]
	#[track_caller]
	pub fn as_shape_mut<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseColMatMut<'a, I, T, V, H> {
		SparseColMatMut {
			symbolic: self.symbolic.as_shape(nrows, ncols),
			val: self.val,
		}
	}

	#[inline]
	pub fn as_dyn(self) -> SparseColMatRef<'a, I, T> {
		SparseColMatRef {
			symbolic: self.symbolic.as_dyn(),
			val: self.val,
		}
	}

	#[inline]
	pub fn as_dyn_mut(self) -> SparseColMatMut<'a, I, T> {
		SparseColMatMut {
			symbolic: self.symbolic.as_dyn(),
			val: self.val,
		}
	}

	#[inline]
	pub fn transpose(self) -> SparseRowMatRef<'a, I, T, Cols, Rows> {
		SparseRowMatRef {
			symbolic: self.symbolic.transpose(),
			val: self.val,
		}
	}

	#[inline]
	pub fn transpose_mut(self) -> SparseRowMatMut<'a, I, T, Cols, Rows> {
		SparseRowMatMut {
			symbolic: self.symbolic.transpose(),
			val: self.val,
		}
	}

	#[inline]
	pub fn conjugate(self) -> SparseColMatRef<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseColMatRef {
			symbolic: self.symbolic,
			val: unsafe { core::slice::from_raw_parts(self.val.as_ptr() as *const T::Conj, len) },
		}
	}

	#[inline]
	pub fn conjugate_mut(self) -> SparseColMatMut<'a, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseColMatMut {
			symbolic: self.symbolic,
			val: unsafe { core::slice::from_raw_parts_mut(self.val.as_mut_ptr() as *mut T::Conj, len) },
		}
	}

	#[inline]
	pub fn adjoint(self) -> SparseRowMatRef<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline]
	pub fn adjoint_mut(self) -> SparseRowMatMut<'a, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate_mut().transpose_mut()
	}

	#[inline]
	pub fn canonical(self) -> SparseColMatRef<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseColMatRef {
			symbolic: self.symbolic,
			val: unsafe { core::slice::from_raw_parts(self.val.as_ptr() as *const T::Canonical, len) },
		}
	}

	#[inline]
	pub fn canonical_mut(self) -> SparseColMatMut<'a, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseColMatMut {
			symbolic: self.symbolic,
			val: unsafe { core::slice::from_raw_parts_mut(self.val.as_mut_ptr() as *mut T::Canonical, len) },
		}
	}

	#[inline]
	pub fn to_row_major(&self) -> Result<SparseRowMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate<Canonical: ComplexField>,
	{
		self.rb().to_row_major()
	}

	#[inline]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate<Canonical: ComplexField>,
	{
		self.rb().to_dense()
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T> SparseColMat<I, T, Rows, Cols> {
	#[inline]
	#[track_caller]
	pub fn new(symbolic: SymbolicSparseColMat<I, Rows, Cols>, val: alloc::vec::Vec<T>) -> Self {
		assert!(symbolic.row_idx().len() == val.len());
		Self { symbolic, val }
	}

	#[inline]
	pub fn parts(&self) -> (SymbolicSparseColMatRef<'_, I, Rows, Cols>, &'_ [T]) {
		(self.symbolic.rb(), &self.val)
	}

	#[inline]
	pub fn parts_mut(&mut self) -> (SymbolicSparseColMatRef<'_, I, Rows, Cols>, &'_ mut [T]) {
		(self.symbolic.rb(), &mut self.val)
	}

	#[inline]
	pub fn into_parts(self) -> (SymbolicSparseColMat<I, Rows, Cols>, alloc::vec::Vec<T>) {
		(self.symbolic, self.val)
	}

	#[inline]
	pub fn symbolic(&self) -> SymbolicSparseColMatRef<'_, I, Rows, Cols> {
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
	pub fn val_of_col(&self, j: Idx<Cols>) -> &'_ [T] {
		unsafe { self.val.get_unchecked(self.col_range(j)) }
	}

	#[inline]
	#[track_caller]
	pub fn val_of_col_mut(&mut self, j: Idx<Cols>) -> &'_ mut [T] {
		unsafe { self.val.get_unchecked_mut(self.symbolic.col_range(j)) }
	}

	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(&self, nrows: V, ncols: H) -> SparseColMatRef<'_, I, T, V, H> {
		SparseColMatRef {
			symbolic: self.symbolic.as_shape(nrows, ncols),
			val: &self.val,
		}
	}

	#[inline]
	#[track_caller]
	pub fn as_shape_mut<V: Shape, H: Shape>(&mut self, nrows: V, ncols: H) -> SparseColMatMut<'_, I, T, V, H> {
		SparseColMatMut {
			symbolic: self.symbolic.as_shape(nrows, ncols),
			val: &mut self.val,
		}
	}

	#[inline]
	#[track_caller]
	pub fn into_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> SparseColMat<I, T, V, H> {
		SparseColMat {
			symbolic: self.symbolic.into_shape(nrows, ncols),
			val: self.val,
		}
	}

	#[inline]
	pub fn as_dyn(&self) -> SparseColMatRef<'_, I, T> {
		SparseColMatRef {
			symbolic: self.symbolic.as_dyn(),
			val: &self.val,
		}
	}

	#[inline]
	pub fn as_dyn_mut(&mut self) -> SparseColMatMut<'_, I, T> {
		SparseColMatMut {
			symbolic: self.symbolic.as_dyn(),
			val: &mut self.val,
		}
	}

	#[inline]
	pub fn into_dyn(self) -> SparseColMat<I, T> {
		SparseColMat {
			symbolic: self.symbolic.into_dyn(),
			val: self.val,
		}
	}

	#[inline]
	pub fn transpose(&self) -> SparseRowMatRef<'_, I, T, Cols, Rows> {
		SparseRowMatRef {
			symbolic: self.symbolic.transpose(),
			val: &self.val,
		}
	}

	#[inline]
	pub fn transpose_mut(&mut self) -> SparseRowMatMut<'_, I, T, Cols, Rows> {
		SparseRowMatMut {
			symbolic: self.symbolic.transpose(),
			val: &mut self.val,
		}
	}

	#[inline]
	pub fn into_transpose(self) -> SparseRowMat<I, T, Cols, Rows> {
		SparseRowMat {
			symbolic: self.symbolic.into_transpose(),
			val: self.val,
		}
	}

	#[inline]
	pub fn conjugate(&self) -> SparseColMatRef<'_, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb().conjugate()
	}

	#[inline]
	pub fn conjugate_mut(&mut self) -> SparseColMatMut<'_, I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb_mut().conjugate_mut()
	}

	#[inline]
	pub fn into_conjugate(self) -> SparseColMat<I, T::Conj, Rows, Cols>
	where
		T: Conjugate,
	{
		let mut vec = core::mem::ManuallyDrop::new(self.val);
		let len = vec.len();
		let cap = vec.capacity();
		let ptr = vec.as_mut_ptr();

		SparseColMat {
			symbolic: self.symbolic,
			val: unsafe { alloc::vec::Vec::from_raw_parts(ptr as *mut T::Conj, len, cap) },
		}
	}

	#[inline]
	pub fn adjoint(&self) -> SparseRowMatRef<'_, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline]
	pub fn adjoint_mut(&mut self) -> SparseRowMatMut<'_, I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.conjugate_mut().transpose_mut()
	}

	#[inline]
	pub fn into_adjoint(self) -> SparseRowMat<I, T::Conj, Cols, Rows>
	where
		T: Conjugate,
	{
		self.into_conjugate().into_transpose()
	}

	#[inline]
	pub fn canonical(&self) -> SparseColMatRef<'_, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseColMatRef {
			symbolic: self.symbolic.rb(),
			val: unsafe { core::slice::from_raw_parts(self.val.as_ptr() as *const T::Canonical, len) },
		}
	}

	#[inline]
	pub fn canonical_mut(&mut self) -> SparseColMatMut<'_, I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let len = self.val.len();
		SparseColMatMut {
			symbolic: self.symbolic.rb(),
			val: unsafe { core::slice::from_raw_parts_mut(self.val.as_mut_ptr() as *mut T::Canonical, len) },
		}
	}

	#[inline]
	pub fn into_canonical(self) -> SparseColMat<I, T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		let mut vec = core::mem::ManuallyDrop::new(self.val);
		let len = vec.len();
		let cap = vec.capacity();
		let ptr = vec.as_mut_ptr();

		SparseColMat {
			symbolic: self.symbolic,
			val: unsafe { alloc::vec::Vec::from_raw_parts(ptr as *mut T::Canonical, len, cap) },
		}
	}

	#[inline]
	pub fn to_row_major(&self) -> Result<SparseRowMat<I, T::Canonical, Rows, Cols>, FaerError>
	where
		T: Conjugate<Canonical: ComplexField>,
	{
		self.rb().to_row_major()
	}

	#[inline]
	pub fn to_dense(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate<Canonical: ComplexField>,
	{
		self.rb().to_dense()
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseColMatRef<'a, I, T, Rows, Cols> {
	type Target = SymbolicSparseColMatRef<'a, I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.symbolic
	}
}

impl<'a, Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseColMatMut<'a, I, T, Rows, Cols> {
	type Target = SymbolicSparseColMatRef<'a, I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.symbolic
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T> core::ops::Deref for SparseColMat<I, T, Rows, Cols> {
	type Target = SymbolicSparseColMat<I, Rows, Cols>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.symbolic
	}
}

impl<Rows: Shape, Cols: Shape, I: Index> fmt::Debug for SymbolicSparseColMatRef<'_, I, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fn imp<'ROWS, 'COLS, I: Index>(mat: SymbolicSparseColMatRef<'_, I, Dim<'ROWS>, Dim<'COLS>>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
			struct Entries<'a, 'ROWS, 'COLS, I>(SymbolicSparseColMatRef<'a, I, Dim<'ROWS>, Dim<'COLS>>);

			impl<'ROWS, 'COLS, I: Index> fmt::Debug for Entries<'_, 'ROWS, 'COLS, I> {
				fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
					let mat = self.0;

					f.debug_list()
						.entries(mat.ncols().indices().flat_map(|col| mat.row_idx_of_col(col).map(move |row| Pair { row, col })))
						.finish()
				}
			}

			f.debug_struct("SymbolicSparseColMat")
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
impl<Rows: Shape, Cols: Shape, I: Index> fmt::Debug for SymbolicSparseColMat<I, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.rb().fmt(f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for SparseColMatRef<'_, I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fn imp<'ROWS, 'COLS, I: Index, T: fmt::Debug>(mat: SparseColMatRef<'_, I, T, Dim<'ROWS>, Dim<'COLS>>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
			struct Entries<'a, 'ROWS, 'COLS, I, T>(SparseColMatRef<'a, I, T, Dim<'ROWS>, Dim<'COLS>>);

			impl<'ROWS, 'COLS, I: Index, T: fmt::Debug> fmt::Debug for Entries<'_, 'ROWS, 'COLS, I, T> {
				fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
					let mat = self.0;

					f.debug_list()
						.entries(mat.ncols().indices().flat_map(|col| {
							let row_idx = mat.row_idx_of_col(col);
							let val = mat.val_of_col(col);

							iter::zip(row_idx, val).map(move |(row, val)| Triplet {
								row,
								col,
								val: crate::hacks::hijack_debug(val),
							})
						}))
						.finish()
				}
			}

			f.debug_struct("SparseColMat")
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

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for SparseColMatMut<'_, I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.rb().fmt(f)
	}
}

impl<Rows: Shape, Cols: Shape, I: Index, T: fmt::Debug> fmt::Debug for SparseColMat<I, T, Rows, Cols> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.rb().fmt(f)
	}
}
