use crate::internal_prelude::*;
use crate::{ContiguousFwd, Idx, IdxInc, TryReserveError};
use core::ops::{Index, IndexMut};
use faer_traits::Real;

use super::RowIndex;

/// heap allocated resizable row vector.
///
/// # note
///
/// the memory layout of `Row` is guaranteed to be row-major, meaning that it has a column stride
/// of `1`.
#[derive(Clone)]
pub struct Row<T, Cols: Shape = usize> {
	pub(crate) trans: Col<T, Cols>,
}

impl<T, Cols: Shape> Row<T, Cols> {
	/// returns a new row with dimension `nrows`, filled with the provided function
	#[inline]
	pub fn from_fn(nrows: Cols, f: impl FnMut(Idx<Cols>) -> T) -> Self {
		Self {
			trans: Col::from_fn(nrows, f),
		}
	}

	/// returns a new row with dimension `nrows`, filled with zeros
	#[inline]
	pub fn zeros(ncols: Cols) -> Self
	where
		T: ComplexField,
	{
		Self { trans: Col::zeros(ncols) }
	}

	/// returns a new row with dimension `nrows`, filled with ones
	#[inline]
	pub fn ones(ncols: Cols) -> Self
	where
		T: ComplexField,
	{
		Self { trans: Col::ones(ncols) }
	}

	/// returns a new row with dimension `nrows`, filled with `value`
	#[inline]
	pub fn full(ncols: Cols, value: T) -> Self
	where
		T: Clone,
	{
		Self {
			trans: Col::full(ncols, value),
		}
	}

	/// reserves the minimum capacity for `col_capacity` columns without reallocating, or returns an
	/// error in case of failure. does nothing if the capacity is already sufficient
	#[inline]
	pub fn try_reserve(&mut self, new_row_capacity: usize) -> Result<(), TryReserveError> {
		self.trans.try_reserve(new_row_capacity)
	}

	/// reserves the minimum capacity for `col_capacity` columns without reallocating. does nothing
	/// if the capacity is already sufficient
	#[track_caller]
	pub fn reserve(&mut self, new_row_capacity: usize) {
		self.trans.reserve(new_row_capacity)
	}

	/// resizes the row in-place so that the new dimension is `new_ncols`.
	/// new elements are created with the given function `f`, so that elements at index `j`
	/// are created by calling `f(j)`
	#[inline]
	pub fn resize_with(&mut self, new_nrows: Cols, f: impl FnMut(Idx<Cols>) -> T) {
		self.trans.resize_with(new_nrows, f);
	}

	/// truncates the row so that its new dimensions are `new_ncols`.  
	/// the new dimension must be smaller than or equal to the current dimension
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// - `new_ncols > self.ncols()`
	#[inline]
	pub fn truncate(&mut self, new_nrows: Cols) {
		self.trans.truncate(new_nrows);
	}

	/// see [`RowRef::as_col_shape`]
	#[inline]
	pub fn into_col_shape<V: Shape>(self, nrows: V) -> Row<T, V> {
		Row {
			trans: self.trans.into_row_shape(nrows),
		}
	}

	/// see [`RowRef::as_diagonal`]
	#[inline]
	pub fn into_diagonal(self) -> Diag<T, Cols> {
		Diag { inner: self.trans }
	}

	/// see [`RowRef::transpose`]
	#[inline]
	pub fn into_transpose(self) -> Col<T, Cols> {
		self.trans
	}
}

impl<T, Cols: Shape> Row<T, Cols> {
	/// returns the number of rows of the row (always 1)
	#[inline]
	pub fn nrows(&self) -> usize {
		self.trans.ncols()
	}

	/// returns the number of columns of the row
	#[inline]
	pub fn ncols(&self) -> Cols {
		self.trans.nrows()
	}

	#[inline]
	/// returns a view over `self`
	pub fn as_ref(&self) -> RowRef<'_, T, Cols> {
		self.trans.as_ref().transpose()
	}

	#[inline]
	/// returns a view over `self`
	pub fn as_mut(&mut self) -> RowMut<'_, T, Cols> {
		self.trans.as_mut().transpose_mut()
	}

	#[inline]
	/// see [`RowRef::norm_max`]
	pub fn norm_max(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().norm_max()
	}

	#[inline]
	/// see [`RowRef::norm_l2`]
	pub fn norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().norm_l2()
	}

	#[inline]
	/// see [`RowRef::squared_norm_l2`]
	pub fn squared_norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().squared_norm_l2()
	}

	#[inline]
	/// see [`RowRef::norm_l1`]
	pub fn norm_l1(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().norm_l1()
	}

	#[inline]
	/// see [`RowRef::sum`]
	pub fn sum(&self) -> T::Canonical
	where
		T: Conjugate,
	{
		self.rb().as_mat().sum()
	}

	/// see [`RowRef::cloned`]
	#[inline]
	pub fn cloned(&self) -> Row<T, Cols>
	where
		T: Clone,
	{
		self.rb().cloned()
	}

	/// see [`RowRef::to_owned`]
	#[inline]
	pub fn to_owned(&self) -> Row<T::Canonical, Cols>
	where
		T: Conjugate,
	{
		self.rb().to_owned()
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
	/// see [`RowRef::as_ptr`]
	pub fn as_ptr(&self) -> *const T {
		self.as_ref().as_ptr()
	}

	#[inline(always)]
	/// see [`RowRef::shape`]
	pub fn shape(&self) -> (usize, Cols) {
		self.as_ref().shape()
	}

	#[inline(always)]
	/// see [`RowRef::col_stride`]
	pub fn col_stride(&self) -> isize {
		self.as_ref().col_stride()
	}

	#[inline(always)]
	/// see [`RowRef::ptr_at`]
	pub fn ptr_at(&self, col: IdxInc<Cols>) -> *const T {
		self.as_ref().ptr_at(col)
	}

	#[inline(always)]
	#[track_caller]
	/// see [`RowRef::ptr_inbounds_at`]
	pub unsafe fn ptr_inbounds_at(&self, col: Idx<Cols>) -> *const T {
		self.as_ref().ptr_inbounds_at(col)
	}

	#[inline]
	#[track_caller]
	/// see [`RowRef::split_at_col`]
	pub fn split_at_col(&self, col: IdxInc<Cols>) -> (RowRef<'_, T, usize>, RowRef<'_, T, usize>) {
		self.as_ref().split_at_col(col)
	}

	#[inline(always)]
	/// see [`RowRef::transpose`]
	pub fn transpose(&self) -> ColRef<'_, T, Cols> {
		self.as_ref().transpose()
	}

	#[inline(always)]
	/// see [`RowRef::conjugate`]
	pub fn conjugate(&self) -> RowRef<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.as_ref().conjugate()
	}

	#[inline(always)]
	/// see [`RowRef::canonical`]
	pub fn canonical(&self) -> RowRef<'_, T::Canonical, Cols>
	where
		T: Conjugate,
	{
		self.as_ref().canonical()
	}

	#[inline(always)]
	/// see [`RowRef::adjoint`]
	pub fn adjoint(&self) -> ColRef<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.as_ref().adjoint()
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowRef::get`]
	pub fn get<ColRange>(&self, col: ColRange) -> <RowRef<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowRef<'a, T, Cols>: RowIndex<ColRange>,
	{
		<RowRef<'_, T, Cols> as RowIndex<ColRange>>::get(self.as_ref(), col)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowRef::get_unchecked`]
	pub unsafe fn get_unchecked<ColRange>(&self, col: ColRange) -> <RowRef<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowRef<'a, T, Cols>: RowIndex<ColRange>,
	{
		unsafe { <RowRef<'_, T, Cols> as RowIndex<ColRange>>::get_unchecked(self.as_ref(), col) }
	}

	#[inline]
	/// see [`RowRef::reverse_cols`]
	pub fn reverse_cols(&self) -> RowRef<'_, T, Cols> {
		self.as_ref().reverse_cols()
	}

	#[inline]
	/// see [`RowRef::subcols`]
	pub fn subcols<V: Shape>(&self, col_start: IdxInc<Cols>, ncols: V) -> RowRef<'_, T, V> {
		self.as_ref().subcols(col_start, ncols)
	}

	#[inline]
	/// see [`RowRef::as_col_shape`]
	pub fn as_col_shape<V: Shape>(&self, ncols: V) -> RowRef<'_, T, V> {
		self.as_ref().as_col_shape(ncols)
	}

	#[inline]
	/// see [`RowRef::as_dyn_cols`]
	pub fn as_dyn_cols(&self) -> RowRef<'_, T, usize> {
		self.as_ref().as_dyn_cols()
	}

	#[inline]
	/// see [`RowRef::as_dyn_stride`]
	pub fn as_dyn_stride(&self) -> RowRef<'_, T, Cols, isize> {
		self.as_ref().as_dyn_stride()
	}

	#[inline]
	/// see [`RowRef::iter`]
	pub fn iter(&self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = &'_ T> {
		self.as_ref().iter()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`RowRef::par_iter`]
	pub fn par_iter(&self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = &'_ T>
	where
		T: Sync,
	{
		self.as_ref().par_iter()
	}

	#[inline]
	/// see [`RowRef::try_as_row_major`]
	pub fn try_as_row_major(&self) -> Option<RowRef<'_, T, Cols, ContiguousFwd>> {
		self.as_ref().try_as_row_major()
	}

	#[inline]
	/// see [`RowRef::as_diagonal`]
	pub fn as_diagonal(&self) -> DiagRef<'_, T, Cols> {
		self.as_ref().as_diagonal()
	}

	#[inline(always)]
	/// see [`RowRef::const_cast`]
	pub unsafe fn const_cast(&self) -> RowMut<'_, T, Cols> {
		self.as_ref().const_cast()
	}

	#[inline]
	/// see [`RowRef::as_mat`]
	pub fn as_mat(&self) -> MatRef<'_, T, usize, Cols, isize> {
		self.as_ref().as_mat()
	}

	#[inline]
	/// see [`RowRef::as_mat`]
	pub fn as_mat_mut(&mut self) -> MatMut<'_, T, usize, Cols, isize> {
		self.as_mut().as_mat_mut()
	}
}

impl<T, Cols: Shape> Row<T, Cols> {
	#[inline(always)]
	/// see [`RowMut::as_ptr_mut`]
	pub fn as_ptr_mut(&mut self) -> *mut T {
		self.as_mut().as_ptr_mut()
	}

	#[inline(always)]
	/// see [`RowMut::ptr_at_mut`]
	pub fn ptr_at_mut(&mut self, col: IdxInc<Cols>) -> *mut T {
		self.as_mut().ptr_at_mut(col)
	}

	#[inline(always)]
	#[track_caller]
	/// see [`RowMut::ptr_inbounds_at_mut`]
	pub unsafe fn ptr_inbounds_at_mut(&mut self, col: Idx<Cols>) -> *mut T {
		self.as_mut().ptr_inbounds_at_mut(col)
	}

	#[inline]
	#[track_caller]
	/// see [`RowMut::split_at_col_mut`]
	pub fn split_at_col_mut(&mut self, col: IdxInc<Cols>) -> (RowMut<'_, T, usize>, RowMut<'_, T, usize>) {
		self.as_mut().split_at_col_mut(col)
	}

	#[inline(always)]
	/// see [`RowMut::transpose_mut`]
	pub fn transpose_mut(&mut self) -> ColMut<'_, T, Cols> {
		self.as_mut().transpose_mut()
	}

	#[inline(always)]
	/// see [`RowMut::conjugate_mut`]
	pub fn conjugate_mut(&mut self) -> RowMut<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.as_mut().conjugate_mut()
	}

	#[inline(always)]
	/// see [`RowMut::canonical_mut`]
	pub fn canonical_mut(&mut self) -> RowMut<'_, T::Canonical, Cols>
	where
		T: Conjugate,
	{
		self.as_mut().canonical_mut()
	}

	#[inline(always)]
	/// see [`RowMut::adjoint_mut`]
	pub fn adjoint_mut(&mut self) -> ColMut<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.as_mut().adjoint_mut()
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowMut::get_mut`]
	pub fn get_mut<ColRange>(&mut self, col: ColRange) -> <RowMut<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowMut<'a, T, Cols>: RowIndex<ColRange>,
	{
		<RowMut<'_, T, Cols> as RowIndex<ColRange>>::get(self.as_mut(), col)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowMut::get_mut_unchecked`]
	pub unsafe fn get_mut_unchecked<ColRange>(&mut self, col: ColRange) -> <RowMut<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowMut<'a, T, Cols>: RowIndex<ColRange>,
	{
		unsafe { <RowMut<'_, T, Cols> as RowIndex<ColRange>>::get_unchecked(self.as_mut(), col) }
	}

	#[inline]
	/// see [`RowMut::reverse_cols_mut`]
	pub fn reverse_cols_mut(&mut self) -> RowMut<'_, T, Cols> {
		self.as_mut().reverse_cols_mut()
	}

	#[inline]
	/// see [`RowMut::subcols_mut`]
	pub fn subcols_mut<V: Shape>(&mut self, col_start: IdxInc<Cols>, ncols: V) -> RowMut<'_, T, V> {
		self.as_mut().subcols_mut(col_start, ncols)
	}

	#[inline]
	/// see [`RowMut::as_col_shape_mut`]
	pub fn as_col_shape_mut<V: Shape>(&mut self, ncols: V) -> RowMut<'_, T, V> {
		self.as_mut().as_col_shape_mut(ncols)
	}

	#[inline]
	/// see [`RowMut::as_dyn_cols_mut`]
	pub fn as_dyn_cols_mut(&mut self) -> RowMut<'_, T, usize> {
		self.as_mut().as_dyn_cols_mut()
	}

	#[inline]
	/// see [`RowMut::as_dyn_stride_mut`]
	pub fn as_dyn_stride_mut(&mut self) -> RowMut<'_, T, Cols, isize> {
		self.as_mut().as_dyn_stride_mut()
	}

	#[inline]
	/// see [`RowMut::iter_mut`]
	pub fn iter_mut(&mut self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = &'_ mut T> {
		self.as_mut().iter_mut()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`RowMut::par_iter_mut`]
	pub fn par_iter_mut(&mut self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = &'_ mut T>
	where
		T: Send,
	{
		self.as_mut().par_iter_mut()
	}

	#[inline]
	/// see [`RowMut::copy_from`]
	pub fn copy_from<RhsT: Conjugate<Canonical = T>>(&mut self, other: impl AsRowRef<T = RhsT, Cols = Cols>)
	where
		T: ComplexField,
	{
		self.as_mut().copy_from(other)
	}

	#[inline]
	/// see [`RowMut::try_as_row_major_mut`]
	pub fn try_as_row_major_mut(&mut self) -> Option<RowMut<'_, T, Cols, ContiguousFwd>> {
		self.as_mut().try_as_row_major_mut()
	}

	#[inline]
	/// see [`RowMut::as_diagonal_mut`]
	pub fn as_diagonal_mut(&mut self) -> DiagMut<'_, T, Cols> {
		self.as_mut().as_diagonal_mut()
	}
}

impl<'short, T, Cols: Shape> Reborrow<'short> for Row<T, Cols> {
	type Target = RowRef<'short, T, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		self.as_ref()
	}
}
impl<'short, T, Cols: Shape> ReborrowMut<'short> for Row<T, Cols> {
	type Target = RowMut<'short, T, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		self.as_mut()
	}
}

#[cfg(feature = "std")]
impl<T> From<std::vec::Vec<T>> for Row<T> {
	#[inline]
	fn from(vec: std::vec::Vec<T>) -> Self {
		let n = vec.len();
		let row = Row::from_fn(n, |i| unsafe { std::ptr::read(&vec[i as usize]) });
		std::mem::forget(vec);
		row
	}
}

#[cfg(feature = "std")]
impl<T> From<Row<T>> for std::vec::Vec<T> {
	#[inline]
	fn from(mut row: Row<T>) -> Self {
		let n = row.ncols();
		let mut vec = std::vec::Vec::with_capacity(n);
		for i in 0..n {
			unsafe {
				let ptr = row.ptr_inbounds_at_mut(i);
				vec.push(std::ptr::read(ptr));
			}
		}
		std::mem::forget(row);
		vec
	}
}

#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
	use super::*;

	#[test]
	fn test_from_vec() {
		// Test with integers
		let vec = vec![1, 2, 3, 4, 5];
		let row = Row::from(vec);

		// Check dimensions
		assert_eq!(row.ncols(), 5);
		assert_eq!(row.nrows(), 1);

		// Check elements
		for i in 0..5 {
			assert_eq!(row[i], i + 1);
		}

		// Test with floating point
		let vec = vec![1.0, 2.5, 3.7, 4.2, 5.9];
		let row = Row::from(vec);

		// Check dimensions
		assert_eq!(row.ncols(), 5);
		assert_eq!(row.nrows(), 1);

		// Check elements
		assert_eq!(row[0], 1.0);
		assert_eq!(row[1], 2.5);
		assert_eq!(row[2], 3.7);
		assert_eq!(row[3], 4.2);
		assert_eq!(row[4], 5.9);
	}

	#[test]
	fn test_empty_vec() {
		let vec: Vec<f64> = vec![];
		let row = Row::from(vec);

		assert_eq!(row.ncols(), 0);
		assert_eq!(row.nrows(), 1);
	}

	#[test]
	fn test_into_vec() {
		let row = Row::from_fn(5, |i| i as f64);
		let vec: Vec<f64> = row.into();

		assert_eq!(vec.len(), 5);
		for i in 0..5 {
			assert_eq!(vec[i], i as f64);
		}
	}
}
