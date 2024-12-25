use super::{AsColRef, ColIndex};
use crate::internal_prelude::*;
use crate::{Idx, IdxInc, TryReserveError};
use core::ops::{Index, IndexMut};
use faer_traits::Real;

#[derive(Clone)]
pub struct Col<T, Rows: Shape = usize> {
	column: Mat<T, Rows, usize>,
}

#[inline]
fn idx_to_pair<T, R>(f: impl FnMut(T) -> R) -> impl FnMut(T, usize) -> R {
	let mut f = f;
	#[inline(always)]
	move |i, _| f(i)
}

impl<T, Rows: Shape> Col<T, Rows> {
	pub fn from_fn(nrows: Rows, f: impl FnMut(Idx<Rows>) -> T) -> Self {
		Self {
			column: Mat::from_fn(nrows, 1, idx_to_pair(f)),
		}
	}

	#[inline]
	pub fn zeros(nrows: Rows) -> Self
	where
		T: ComplexField,
	{
		Self {
			column: Mat::zeros(nrows, 1),
		}
	}

	#[inline]
	pub fn ones(nrows: Rows) -> Self
	where
		T: ComplexField,
	{
		Self { column: Mat::ones(nrows, 1) }
	}

	#[inline]
	pub fn full(nrows: Rows, value: T) -> Self
	where
		T: Clone,
	{
		Self {
			column: Mat::full(nrows, 1, value),
		}
	}

	pub fn try_reserve(&mut self, new_row_capacity: usize) -> Result<(), TryReserveError> {
		self.column.try_reserve(new_row_capacity, 1)
	}

	#[track_caller]
	pub fn reserve(&mut self, new_row_capacity: usize) {
		self.column.reserve(new_row_capacity, 1)
	}

	pub fn resize_with(&mut self, new_nrows: Rows, f: impl FnMut(Idx<Rows>) -> T) {
		self.column.resize_with(new_nrows, 1, idx_to_pair(f));
	}

	pub fn truncate(&mut self, new_nrows: Rows) {
		self.column.truncate(new_nrows, 1);
	}

	#[inline]
	pub fn into_row_shape<V: Shape>(self, nrows: V) -> Col<T, V> {
		Col {
			column: self.column.into_shape(nrows, 1),
		}
	}

	#[inline]
	pub fn into_diagonal(self) -> Diag<T, Rows> {
		Diag { inner: self }
	}
}

impl<T, Rows: Shape> Col<T, Rows> {
	#[inline]
	pub fn nrows(&self) -> Rows {
		self.column.nrows()
	}

	#[inline]
	pub fn ncols(&self) -> usize {
		self.column.ncols()
	}

	#[inline]
	pub fn as_ref(&self) -> ColRef<'_, T, Rows> {
		self.column.as_ref().col(0)
	}

	#[inline]
	pub fn as_mut(&mut self) -> ColMut<'_, T, Rows> {
		self.column.as_mut().col_mut(0)
	}

	#[inline]
	pub fn norm_max(&self) -> Real<T>
	where
		T: Conjugate,
	{
		linalg::reductions::norm_max::norm_max(self.as_ref().canonical().as_dyn_stride().as_dyn_rows().as_mat())
	}

	#[inline]
	pub fn norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		linalg::reductions::norm_l2::norm_l2(self.as_ref().canonical().as_dyn_stride().as_dyn_rows().as_mat())
	}
}

impl<T: core::fmt::Debug, Rows: Shape> core::fmt::Debug for Col<T, Rows> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.as_ref().fmt(f)
	}
}

impl<T, Rows: Shape> Index<Idx<Rows>> for Col<T, Rows> {
	type Output = T;

	#[inline]
	#[track_caller]
	fn index(&self, row: Idx<Rows>) -> &Self::Output {
		self.as_ref().at(row)
	}
}

impl<T, Rows: Shape> IndexMut<Idx<Rows>> for Col<T, Rows> {
	#[inline]
	#[track_caller]
	fn index_mut(&mut self, row: Idx<Rows>) -> &mut Self::Output {
		self.as_mut().at_mut(row)
	}
}

impl<T, Rows: Shape> Col<T, Rows> {
	#[inline(always)]
	pub fn as_ptr(&self) -> *const T {
		self.as_ref().as_ptr()
	}

	#[inline(always)]
	pub fn shape(&self) -> (Rows, usize) {
		(self.nrows(), self.ncols())
	}

	#[inline(always)]
	pub fn row_stride(&self) -> isize {
		1
	}

	#[inline(always)]
	pub fn ptr_at(&self, row: IdxInc<Rows>) -> *const T {
		self.as_ref().ptr_at(row)
	}

	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>) -> *const T {
		self.as_ref().ptr_inbounds_at(row)
	}

	#[inline]
	#[track_caller]
	pub fn split_at_row(&self, row: IdxInc<Rows>) -> (ColRef<'_, T, usize>, ColRef<'_, T, usize>) {
		self.as_ref().split_at_row(row)
	}

	#[inline(always)]
	pub fn transpose(&self) -> RowRef<'_, T, Rows> {
		self.as_ref().transpose()
	}

	#[inline(always)]
	pub fn conjugate(&self) -> ColRef<'_, T::Conj, Rows>
	where
		T: Conjugate,
	{
		self.as_ref().conjugate()
	}

	#[inline(always)]
	pub fn canonical(&self) -> ColRef<'_, T::Canonical, Rows>
	where
		T: Conjugate,
	{
		self.as_ref().canonical()
	}

	#[inline(always)]
	pub fn adjoint(&self) -> RowRef<'_, T::Conj, Rows>
	where
		T: Conjugate,
	{
		self.as_ref().adjoint()
	}

	#[track_caller]
	#[inline(always)]
	pub fn get<RowRange>(&self, row: RowRange) -> <ColRef<'_, T, Rows> as ColIndex<RowRange>>::Target
	where
		for<'a> ColRef<'a, T, Rows>: ColIndex<RowRange>,
	{
		<ColRef<'_, T, Rows> as ColIndex<RowRange>>::get(self.as_ref(), row)
	}

	#[track_caller]
	#[inline(always)]
	pub unsafe fn get_unchecked<RowRange>(&self, row: RowRange) -> <ColRef<'_, T, Rows> as ColIndex<RowRange>>::Target
	where
		for<'a> ColRef<'a, T, Rows>: ColIndex<RowRange>,
	{
		unsafe { <ColRef<'_, T, Rows> as ColIndex<RowRange>>::get_unchecked(self.as_ref(), row) }
	}

	#[inline]
	pub fn reverse_rows(&self) -> ColRef<'_, T, Rows> {
		self.as_ref().reverse_rows()
	}

	#[inline]
	pub fn subrows<V: Shape>(&self, row_start: IdxInc<Rows>, nrows: V) -> ColRef<'_, T, V> {
		self.as_ref().subrows(row_start, nrows)
	}

	#[inline]
	pub fn as_row_shape<V: Shape>(&self, nrows: V) -> ColRef<'_, T, V> {
		self.as_ref().as_row_shape(nrows)
	}

	#[inline]
	pub fn as_dyn_rows(&self) -> ColRef<'_, T, usize> {
		self.as_ref().as_dyn_rows()
	}

	#[inline]
	pub fn as_dyn_stride(&self) -> ColRef<'_, T, Rows, isize> {
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
	#[track_caller]
	#[cfg(feature = "rayon")]
	pub fn par_partition(&self, count: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColRef<'_, T, usize>>
	where
		T: Sync,
	{
		self.as_ref().par_partition(count)
	}

	#[inline]
	pub fn try_as_col_major(&self) -> Option<ColRef<'_, T, Rows, ContiguousFwd>> {
		self.as_ref().try_as_col_major()
	}

	#[inline]
	pub fn try_as_col_major_mut(&mut self) -> Option<ColMut<'_, T, Rows, ContiguousFwd>> {
		self.as_ref().try_as_col_major().map(|x| unsafe { x.const_cast() })
	}

	#[inline]
	pub fn as_mat(&self) -> MatRef<'_, T, Rows, usize, isize> {
		self.as_ref().as_mat()
	}

	#[inline]
	pub fn as_mat_mut(&mut self) -> MatMut<'_, T, Rows, usize, isize> {
		unsafe { self.as_ref().as_mat().const_cast() }
	}

	#[inline]
	pub fn as_diagonal(&self) -> DiagRef<'_, T, Rows> {
		DiagRef { inner: self.as_ref() }
	}
}

impl<T, Rows: Shape> Col<T, Rows> {
	#[inline(always)]
	pub fn as_ptr_mut(&mut self) -> *mut T {
		self.as_mut().as_ptr_mut()
	}

	#[inline(always)]
	pub fn ptr_at_mut(&mut self, row: IdxInc<Rows>) -> *mut T {
		self.as_mut().ptr_at_mut(row)
	}

	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at_mut(&mut self, row: Idx<Rows>) -> *mut T {
		self.as_mut().ptr_inbounds_at_mut(row)
	}

	#[inline]
	#[track_caller]
	pub fn split_at_row_mut(&mut self, row: IdxInc<Rows>) -> (ColMut<'_, T, usize>, ColMut<'_, T, usize>) {
		self.as_mut().split_at_row_mut(row)
	}

	#[inline(always)]
	pub fn transpose_mut(&mut self) -> RowMut<'_, T, Rows> {
		self.as_mut().transpose_mut()
	}

	#[inline(always)]
	pub fn conjugate_mut(&mut self) -> ColMut<'_, T::Conj, Rows>
	where
		T: Conjugate,
	{
		self.as_mut().conjugate_mut()
	}

	#[inline(always)]
	pub fn canonical_mut(&mut self) -> ColMut<'_, T::Canonical, Rows>
	where
		T: Conjugate,
	{
		self.as_mut().canonical_mut()
	}

	#[inline(always)]
	pub fn adjoint_mut(&mut self) -> RowMut<'_, T::Conj, Rows>
	where
		T: Conjugate,
	{
		self.as_mut().adjoint_mut()
	}

	#[track_caller]
	#[inline(always)]
	pub fn get_mut<RowRange>(&mut self, row: RowRange) -> <ColMut<'_, T, Rows> as ColIndex<RowRange>>::Target
	where
		for<'a> ColMut<'a, T, Rows>: ColIndex<RowRange>,
	{
		<ColMut<'_, T, Rows> as ColIndex<RowRange>>::get(self.as_mut(), row)
	}

	#[track_caller]
	#[inline(always)]
	pub unsafe fn get_mut_unchecked<RowRange>(&mut self, row: RowRange) -> <ColMut<'_, T, Rows> as ColIndex<RowRange>>::Target
	where
		for<'a> ColMut<'a, T, Rows>: ColIndex<RowRange>,
	{
		unsafe { <ColMut<'_, T, Rows> as ColIndex<RowRange>>::get_unchecked(self.as_mut(), row) }
	}

	#[inline]
	pub fn reverse_rows_mut(&mut self) -> ColMut<'_, T, Rows> {
		self.as_mut().reverse_rows_mut()
	}

	#[inline]
	pub fn subrows_mut<V: Shape>(&mut self, row_start: IdxInc<Rows>, nrows: V) -> ColMut<'_, T, V> {
		self.as_mut().subrows_mut(row_start, nrows)
	}

	#[inline]
	pub fn as_row_shape_mut<V: Shape>(&mut self, nrows: V) -> ColMut<'_, T, V> {
		self.as_mut().as_row_shape_mut(nrows)
	}

	#[inline]
	pub fn as_dyn_rows_mut(&mut self) -> ColMut<'_, T, usize> {
		self.as_mut().as_dyn_rows_mut()
	}

	#[inline]
	pub fn as_dyn_stride_mut(&mut self) -> ColMut<'_, T, Rows, isize> {
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
	#[track_caller]
	#[cfg(feature = "rayon")]
	pub fn par_partition_mut(&mut self, count: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColMut<'_, T, usize>>
	where
		T: Send,
	{
		self.as_mut().par_partition_mut(count)
	}

	#[inline]
	pub fn as_diagonal_mut(&mut self) -> DiagMut<'_, T, Rows> {
		self.as_mut().as_diagonal_mut()
	}

	#[inline]
	#[track_caller]
	pub fn copy_from<RhsT: Conjugate<Canonical = T>>(&mut self, rhs: impl AsColRef<T = RhsT, Rows = Rows>)
	where
		T: ComplexField,
	{
		self.as_mut().copy_from(rhs)
	}
}

impl<'short, T, Rows: Shape> Reborrow<'short> for Col<T, Rows> {
	type Target = ColRef<'short, T, Rows>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		self.as_ref()
	}
}
impl<'short, T, Rows: Shape> ReborrowMut<'short> for Col<T, Rows> {
	type Target = ColMut<'short, T, Rows>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		self.as_mut()
	}
}
