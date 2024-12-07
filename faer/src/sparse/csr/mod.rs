use crate::internal_prelude::*;
use crate::{Idx, assert, debug_assert};
use core::iter;
use core::marker::PhantomData;
use core::ops::Range;

use super::*;

pub struct SymbolicSparseRowMatRef<'a, I, Rows, Cols> {
	pub(super) nrows: Rows,
	pub(super) ncols: Cols,
	pub(super) row_ptr: &'a [I],
	pub(super) row_nnz: Option<&'a [I]>,
	pub(super) col_idx: &'a [I],
}

pub struct SparseRowMatRef<'a, I, T, Rows, Cols> {
	pub(super) symbolic: SymbolicSparseRowMatRef<'a, I, Rows, Cols>,
	pub(super) val: &'a [T],
}

pub struct SparseRowMatMut<'a, I, T, Rows, Cols> {
	pub(super) symbolic: SymbolicSparseRowMatRef<'a, I, Rows, Cols>,
	pub(super) val: &'a mut [T],
}

#[derive(Clone)]
pub struct SymbolicSparseRowMat<I, Rows, Cols> {
	pub(super) nrows: Rows,
	pub(super) ncols: Cols,
	pub(super) row_ptr: alloc::vec::Vec<I>,
	pub(super) row_nnz: Option<alloc::vec::Vec<I>>,
	pub(super) col_idx: alloc::vec::Vec<I>,
}

#[derive(Clone)]
pub struct SparseRowMat<I, T, Rows, Cols> {
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
