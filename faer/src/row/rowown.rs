use crate::internal_prelude::*;
use crate::{ContiguousFwd, Idx, IdxInc, TryReserveError};
use core::ops::{Index, IndexMut};
use faer_traits::Real;

use super::RowIndex;

#[derive(Clone)]
pub struct Row<T, Cols: Shape = usize> {
	trans: Col<T, Cols>,
}

impl<T, Cols: Shape> Row<T, Cols> {
	#[inline]
	pub fn from_fn(nrows: Cols, f: impl FnMut(Idx<Cols>) -> T) -> Self {
		Self {
			trans: Col::from_fn(nrows, f),
		}
	}

	#[inline]
	pub fn zeros(ncols: Cols) -> Self
	where
		T: ComplexField,
	{
		Self { trans: Col::zeros(ncols) }
	}

	#[inline]
	pub fn ones(ncols: Cols) -> Self
	where
		T: ComplexField,
	{
		Self { trans: Col::ones(ncols) }
	}

	#[inline]
	pub fn full(ncols: Cols, value: T) -> Self
	where
		T: Clone,
	{
		Self {
			trans: Col::full(ncols, value),
		}
	}

	#[inline]
	pub fn try_reserve(&mut self, new_row_capacity: usize) -> Result<(), TryReserveError> {
		self.trans.try_reserve(new_row_capacity)
	}

	#[track_caller]
	pub fn reserve(&mut self, new_row_capacity: usize) {
		self.trans.reserve(new_row_capacity)
	}

	#[inline]
	pub fn resize_with(&mut self, new_nrows: Cols, f: impl FnMut(Idx<Cols>) -> T) {
		self.trans.resize_with(new_nrows, f);
	}

	#[inline]
	pub fn truncate(&mut self, new_nrows: Cols) {
		self.trans.truncate(new_nrows);
	}

	#[inline]
	pub fn into_col_shape<V: Shape>(self, nrows: V) -> Row<T, V> {
		Row {
			trans: self.trans.into_row_shape(nrows),
		}
	}

	#[inline]
	pub fn into_diagonal(self) -> Diag<T, Cols> {
		Diag { inner: self.trans }
	}
}

impl<T, Cols: Shape> Row<T, Cols> {
	#[inline]
	pub fn nrows(&self) -> usize {
		self.trans.ncols()
	}

	#[inline]
	pub fn ncols(&self) -> Cols {
		self.trans.nrows()
	}

	#[inline]
	pub fn as_ref(&self) -> RowRef<'_, T, Cols> {
		self.trans.as_ref().transpose()
	}

	#[inline]
	pub fn as_mut(&mut self) -> RowMut<'_, T, Cols> {
		self.trans.as_mut().transpose_mut()
	}

	#[inline]
	pub fn norm_max(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.as_ref().transpose().norm_max()
	}

	#[inline]
	pub fn norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.as_ref().transpose().norm_l2()
	}
}

impl<T: core::fmt::Debug, Cols: Shape> core::fmt::Debug for Row<T, Cols> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.as_ref().fmt(f)
	}
}

impl<T, Cols: Shape> Index<Idx<Cols>> for Row<T, Cols> {
	type Output = T;

	#[inline]
	#[track_caller]
	fn index(&self, col: Idx<Cols>) -> &Self::Output {
		self.as_ref().at(col)
	}
}

impl<T, Cols: Shape> IndexMut<Idx<Cols>> for Row<T, Cols> {
	#[inline]
	#[track_caller]
	fn index_mut(&mut self, col: Idx<Cols>) -> &mut Self::Output {
		self.as_mut().at_mut(col)
	}
}

impl<T, Cols: Shape> Row<T, Cols> {
	#[inline(always)]
	pub fn as_ptr(&self) -> *const T {
		self.as_ref().as_ptr()
	}

	#[inline(always)]
	pub fn shape(&self) -> (usize, Cols) {
		self.as_ref().shape()
	}

	#[inline(always)]
	pub fn col_stride(&self) -> isize {
		self.as_ref().col_stride()
	}

	#[inline(always)]
	pub fn ptr_at(&self, col: IdxInc<Cols>) -> *const T {
		self.as_ref().ptr_at(col)
	}

	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at(&self, col: Idx<Cols>) -> *const T {
		self.as_ref().ptr_inbounds_at(col)
	}

	#[inline]
	#[track_caller]
	pub fn split_at_col(&self, col: IdxInc<Cols>) -> (RowRef<'_, T, usize>, RowRef<'_, T, usize>) {
		self.as_ref().split_at_col(col)
	}

	#[inline(always)]
	pub fn transpose(&self) -> ColRef<'_, T, Cols> {
		self.as_ref().transpose()
	}

	#[inline(always)]
	pub fn conjugate(&self) -> RowRef<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.as_ref().conjugate()
	}

	#[inline(always)]
	pub fn canonical(&self) -> RowRef<'_, T::Canonical, Cols>
	where
		T: Conjugate,
	{
		self.as_ref().canonical()
	}

	#[inline(always)]
	pub fn adjoint(&self) -> ColRef<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.as_ref().adjoint()
	}

	#[track_caller]
	#[inline(always)]
	pub fn get<ColRange>(&self, col: ColRange) -> <RowRef<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowRef<'a, T, Cols>: RowIndex<ColRange>,
	{
		<RowRef<'_, T, Cols> as RowIndex<ColRange>>::get(self.as_ref(), col)
	}

	#[track_caller]
	#[inline(always)]
	pub unsafe fn get_unchecked<ColRange>(&self, col: ColRange) -> <RowRef<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowRef<'a, T, Cols>: RowIndex<ColRange>,
	{
		unsafe { <RowRef<'_, T, Cols> as RowIndex<ColRange>>::get_unchecked(self.as_ref(), col) }
	}

	#[inline]
	pub fn reverse_cols(&self) -> RowRef<'_, T, Cols> {
		self.as_ref().reverse_cols()
	}

	#[inline]
	pub fn subcols<V: Shape>(&self, col_start: IdxInc<Cols>, ncols: V) -> RowRef<'_, T, V> {
		self.as_ref().subcols(col_start, ncols)
	}

	#[inline]
	pub fn as_col_shape<V: Shape>(&self, ncols: V) -> RowRef<'_, T, V> {
		self.as_ref().as_col_shape(ncols)
	}

	#[inline]
	pub fn as_dyn_cols(&self) -> RowRef<'_, T, usize> {
		self.as_ref().as_dyn_cols()
	}

	#[inline]
	pub fn as_dyn_stride(&self) -> RowRef<'_, T, Cols, isize> {
		self.as_ref().as_dyn_stride()
	}

	#[inline]
	pub fn iter(&self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = &'_ T> {
		self.as_ref().iter()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	pub fn par_iter(&self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = &'_ T>
	where
		T: Sync,
	{
		self.as_ref().par_iter()
	}

	#[inline]
	pub fn try_as_row_major(&self) -> Option<RowRef<'_, T, Cols, ContiguousFwd>> {
		self.as_ref().try_as_row_major()
	}

	#[inline]
	pub fn as_diagonal(&self) -> DiagRef<'_, T, Cols> {
		self.as_ref().as_diagonal()
	}

	#[inline(always)]
	pub unsafe fn const_cast(&self) -> RowMut<'_, T, Cols> {
		self.as_ref().const_cast()
	}

	#[inline]
	pub fn as_mat(&self) -> MatRef<'_, T, usize, Cols, isize> {
		self.as_ref().as_mat()
	}

	#[inline]
	pub fn as_mat_mut(&mut self) -> MatMut<'_, T, usize, Cols, isize> {
		self.as_mut().as_mat_mut()
	}
}

impl<T, Cols: Shape> Row<T, Cols> {
	#[inline(always)]
	pub fn as_ptr_mut(&mut self) -> *mut T {
		self.as_mut().as_ptr_mut()
	}

	#[inline(always)]
	pub fn ptr_at_mut(&mut self, col: IdxInc<Cols>) -> *mut T {
		self.as_mut().ptr_at_mut(col)
	}

	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at_mut(&mut self, col: Idx<Cols>) -> *mut T {
		self.as_mut().ptr_inbounds_at_mut(col)
	}

	#[inline]
	#[track_caller]
	pub fn split_at_col_mut(&mut self, col: IdxInc<Cols>) -> (RowMut<'_, T, usize>, RowMut<'_, T, usize>) {
		self.as_mut().split_at_col_mut(col)
	}

	#[inline(always)]
	pub fn transpose_mut(&mut self) -> ColMut<'_, T, Cols> {
		self.as_mut().transpose_mut()
	}

	#[inline(always)]
	pub fn conjugate_mut(&mut self) -> RowMut<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.as_mut().conjugate_mut()
	}

	#[inline(always)]
	pub fn canonical_mut(&mut self) -> RowMut<'_, T::Canonical, Cols>
	where
		T: Conjugate,
	{
		self.as_mut().canonical_mut()
	}

	#[inline(always)]
	pub fn adjoint_mut(&mut self) -> ColMut<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.as_mut().adjoint_mut()
	}

	#[track_caller]
	#[inline(always)]
	pub fn get_mut<ColRange>(&mut self, col: ColRange) -> <RowMut<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowMut<'a, T, Cols>: RowIndex<ColRange>,
	{
		<RowMut<'_, T, Cols> as RowIndex<ColRange>>::get(self.as_mut(), col)
	}

	#[track_caller]
	#[inline(always)]
	pub unsafe fn get_mut_unchecked<ColRange>(&mut self, col: ColRange) -> <RowMut<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowMut<'a, T, Cols>: RowIndex<ColRange>,
	{
		unsafe { <RowMut<'_, T, Cols> as RowIndex<ColRange>>::get_unchecked(self.as_mut(), col) }
	}

	#[inline]
	pub fn reverse_cols_mut(&mut self) -> RowMut<'_, T, Cols> {
		self.as_mut().reverse_cols_mut()
	}

	#[inline]
	pub fn subcols_mut<V: Shape>(&mut self, col_start: IdxInc<Cols>, ncols: V) -> RowMut<'_, T, V> {
		self.as_mut().subcols_mut(col_start, ncols)
	}

	#[inline]
	pub fn as_col_shape_mut<V: Shape>(&mut self, ncols: V) -> RowMut<'_, T, V> {
		self.as_mut().as_col_shape_mut(ncols)
	}

	#[inline]
	pub fn as_dyn_cols_mut(&mut self) -> RowMut<'_, T, usize> {
		self.as_mut().as_dyn_cols_mut()
	}

	#[inline]
	pub fn as_dyn_stride_mut(&mut self) -> RowMut<'_, T, Cols, isize> {
		self.as_mut().as_dyn_stride_mut()
	}

	#[inline]
	pub fn iter_mut(&mut self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = &'_ mut T> {
		self.as_mut().iter_mut()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	pub fn par_iter_mut(&mut self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = &'_ mut T>
	where
		T: Send,
	{
		self.as_mut().par_iter_mut()
	}

	#[inline]
	pub fn copy_from<RhsT: Conjugate<Canonical = T>>(&mut self, other: impl AsRowRef<T = RhsT, Cols = Cols>)
	where
		T: ComplexField,
	{
		self.as_mut().copy_from(other)
	}

	#[inline]
	pub fn try_as_row_major_mut(&mut self) -> Option<RowMut<'_, T, Cols, ContiguousFwd>> {
		self.as_mut().try_as_row_major_mut()
	}

	#[inline]
	pub fn as_diagonal_mut(&mut self) -> DiagMut<'_, T, Cols> {
		self.as_mut().as_diagonal_mut()
	}
}
