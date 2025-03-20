use super::{AsColRef, ColIndex};
use crate::internal_prelude::*;
use crate::{Idx, IdxInc, TryReserveError};
use core::ops::{Index, IndexMut};
use faer_traits::Real;

/// heap allocated resizable column vector.
///
/// # note
///
/// the memory layout of `Col` is guaranteed to be column-major, meaning that it has a row stride
/// of `1`.
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
	/// returns a new column with dimension `nrows`, filled with the provided function
	pub fn from_fn(nrows: Rows, f: impl FnMut(Idx<Rows>) -> T) -> Self {
		Self {
			column: Mat::from_fn(nrows, 1, idx_to_pair(f)),
		}
	}

	/// returns a new column with dimension `nrows`, filled with zeros
	#[inline]
	pub fn zeros(nrows: Rows) -> Self
	where
		T: ComplexField,
	{
		Self {
			column: Mat::zeros(nrows, 1),
		}
	}

	/// returns a new column with dimension `nrows`, filled with ones
	#[inline]
	pub fn ones(nrows: Rows) -> Self
	where
		T: ComplexField,
	{
		Self { column: Mat::ones(nrows, 1) }
	}

	/// returns a new column with dimension `nrows`, filled with `value`
	#[inline]
	pub fn full(nrows: Rows, value: T) -> Self
	where
		T: Clone,
	{
		Self {
			column: Mat::full(nrows, 1, value),
		}
	}

	/// reserves the minimum capacity for `row_capacity` rows without reallocating, or returns an
	/// error in case of failure. does nothing if the capacity is already sufficient
	pub fn try_reserve(&mut self, new_row_capacity: usize) -> Result<(), TryReserveError> {
		self.column.try_reserve(new_row_capacity, 1)
	}

	/// reserves the minimum capacity for `row_capacity` rows without reallocating. does nothing if
	/// the capacity is already sufficient
	#[track_caller]
	pub fn reserve(&mut self, new_row_capacity: usize) {
		self.column.reserve(new_row_capacity, 1)
	}

	/// resizes the column in-place so that the new dimension is `new_nrows`.
	/// new elements are created with the given function `f`, so that elements at index `i`
	/// are created by calling `f(i)`
	pub fn resize_with(&mut self, new_nrows: Rows, f: impl FnMut(Idx<Rows>) -> T) {
		self.column.resize_with(new_nrows, 1, idx_to_pair(f));
	}

	/// truncates the column so that its new dimensions are `new_nrows`.  
	/// the new dimension must be smaller than or equal to the current dimension
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// - `new_nrows > self.nrows()`
	pub fn truncate(&mut self, new_nrows: Rows) {
		self.column.truncate(new_nrows, 1);
	}

	/// see [`ColRef::as_row_shape`]
	#[inline]
	pub fn into_row_shape<V: Shape>(self, nrows: V) -> Col<T, V> {
		Col {
			column: self.column.into_shape(nrows, 1),
		}
	}

	/// see [`ColRef::as_diagonal`]
	#[inline]
	pub fn into_diagonal(self) -> Diag<T, Rows> {
		Diag { inner: self }
	}

	/// see [`ColRef::transpose`]
	#[inline]
	pub fn into_transpose(self) -> Row<T, Rows> {
		Row { trans: self }
	}
}

impl<T, Rows: Shape> Col<T, Rows> {
	/// returns the number of rows of the column
	#[inline]
	pub fn nrows(&self) -> Rows {
		self.column.nrows()
	}

	/// returns the number of columns of the column (always `1`)
	#[inline]
	pub fn ncols(&self) -> usize {
		1
	}

	/// returns a view over `self`
	#[inline]
	pub fn as_ref(&self) -> ColRef<'_, T, Rows> {
		self.column.as_ref().col(0)
	}

	/// returns a view over `self`
	#[inline]
	pub fn as_mut(&mut self) -> ColMut<'_, T, Rows> {
		self.column.as_mut().col_mut(0)
	}

	#[inline]
	/// see [`ColRef::norm_max`]
	pub fn norm_max(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().norm_max()
	}

	#[inline]
	/// see [`ColRef::norm_l2`]
	pub fn norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().norm_l2()
	}

	#[inline]
	/// see [`ColRef::squared_norm_l2`]
	pub fn squared_norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().squared_norm_l2()
	}

	#[inline]
	/// see [`ColRef::norm_l1`]
	pub fn norm_l1(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().norm_l1()
	}

	#[inline]
	/// see [`ColRef::sum`]
	pub fn sum(&self) -> T::Canonical
	where
		T: Conjugate,
	{
		self.rb().as_mat().sum()
	}

	/// see [`ColRef::cloned`]
	#[inline]
	pub fn cloned(&self) -> Col<T, Rows>
	where
		T: Clone,
	{
		self.rb().cloned()
	}

	/// see [`ColRef::to_owned`]
	#[inline]
	pub fn to_owned(&self) -> Col<T::Canonical, Rows>
	where
		T: Conjugate,
	{
		self.rb().to_owned()
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
	/// see [`ColRef::as_ptr`]
	pub fn as_ptr(&self) -> *const T {
		self.as_ref().as_ptr()
	}

	#[inline(always)]
	/// see [`ColRef::shape`]
	pub fn shape(&self) -> (Rows, usize) {
		(self.nrows(), self.ncols())
	}

	#[inline(always)]
	/// see [`ColRef::row_stride`]
	pub fn row_stride(&self) -> isize {
		1
	}

	#[inline(always)]
	/// see [`ColRef::ptr_at`]
	pub fn ptr_at(&self, row: IdxInc<Rows>) -> *const T {
		self.as_ref().ptr_at(row)
	}

	#[inline(always)]
	#[track_caller]
	/// see [`ColRef::ptr_inbounds_at`]
	pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>) -> *const T {
		self.as_ref().ptr_inbounds_at(row)
	}

	#[inline]
	#[track_caller]
	/// see [`ColRef::split_at_row`]
	pub fn split_at_row(&self, row: IdxInc<Rows>) -> (ColRef<'_, T, usize>, ColRef<'_, T, usize>) {
		self.as_ref().split_at_row(row)
	}

	#[inline(always)]
	/// see [`ColRef::transpose`]
	pub fn transpose(&self) -> RowRef<'_, T, Rows> {
		self.as_ref().transpose()
	}

	#[inline(always)]
	/// see [`ColRef::conjugate`]
	pub fn conjugate(&self) -> ColRef<'_, T::Conj, Rows>
	where
		T: Conjugate,
	{
		self.as_ref().conjugate()
	}

	#[inline(always)]
	/// see [`ColRef::canonical`]
	pub fn canonical(&self) -> ColRef<'_, T::Canonical, Rows>
	where
		T: Conjugate,
	{
		self.as_ref().canonical()
	}

	#[inline(always)]
	/// see [`ColRef::adjoint`]
	pub fn adjoint(&self) -> RowRef<'_, T::Conj, Rows>
	where
		T: Conjugate,
	{
		self.as_ref().adjoint()
	}

	#[track_caller]
	#[inline(always)]
	/// see [`ColRef::get`]
	pub fn get<RowRange>(&self, row: RowRange) -> <ColRef<'_, T, Rows> as ColIndex<RowRange>>::Target
	where
		for<'a> ColRef<'a, T, Rows>: ColIndex<RowRange>,
	{
		<ColRef<'_, T, Rows> as ColIndex<RowRange>>::get(self.as_ref(), row)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`ColRef::get_unchecked`]
	pub unsafe fn get_unchecked<RowRange>(&self, row: RowRange) -> <ColRef<'_, T, Rows> as ColIndex<RowRange>>::Target
	where
		for<'a> ColRef<'a, T, Rows>: ColIndex<RowRange>,
	{
		unsafe { <ColRef<'_, T, Rows> as ColIndex<RowRange>>::get_unchecked(self.as_ref(), row) }
	}

	#[inline]
	/// see [`ColRef::reverse_rows`]
	pub fn reverse_rows(&self) -> ColRef<'_, T, Rows> {
		self.as_ref().reverse_rows()
	}

	#[inline]
	/// see [`ColRef::subrows`]
	pub fn subrows<V: Shape>(&self, row_start: IdxInc<Rows>, nrows: V) -> ColRef<'_, T, V> {
		self.as_ref().subrows(row_start, nrows)
	}

	#[inline]
	/// see [`ColRef::as_row_shape`]
	pub fn as_row_shape<V: Shape>(&self, nrows: V) -> ColRef<'_, T, V> {
		self.as_ref().as_row_shape(nrows)
	}

	#[inline]
	/// see [`ColRef::as_dyn_rows`]
	pub fn as_dyn_rows(&self) -> ColRef<'_, T, usize> {
		self.as_ref().as_dyn_rows()
	}

	#[inline]
	/// see [`ColRef::as_dyn_stride`]
	pub fn as_dyn_stride(&self) -> ColRef<'_, T, Rows, isize> {
		self.as_ref().as_dyn_stride()
	}

	#[inline]
	/// see [`ColRef::iter`]
	pub fn iter(&self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = &'_ T> {
		self.as_ref().iter()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`ColRef::par_iter`]
	pub fn par_iter(&self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = &'_ T>
	where
		T: Sync,
	{
		self.as_ref().par_iter()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`ColRef::par_partition`]
	pub fn par_partition(&self, count: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColRef<'_, T, usize>>
	where
		T: Sync,
	{
		self.as_ref().par_partition(count)
	}

	#[inline]
	/// see [`ColRef::try_as_col_major`]
	pub fn try_as_col_major(&self) -> Option<ColRef<'_, T, Rows, ContiguousFwd>> {
		self.as_ref().try_as_col_major()
	}

	#[inline]
	/// see [`ColRef::try_as_col_major`]
	pub fn try_as_col_major_mut(&mut self) -> Option<ColMut<'_, T, Rows, ContiguousFwd>> {
		self.as_ref().try_as_col_major().map(|x| unsafe { x.const_cast() })
	}

	#[inline]
	/// see [`ColRef::as_mat`]
	pub fn as_mat(&self) -> MatRef<'_, T, Rows, usize, isize> {
		self.as_ref().as_mat()
	}

	#[inline]
	/// see [`ColRef::as_mat`]
	pub fn as_mat_mut(&mut self) -> MatMut<'_, T, Rows, usize, isize> {
		unsafe { self.as_ref().as_mat().const_cast() }
	}

	#[inline]
	/// see [`ColRef::as_diagonal`]
	pub fn as_diagonal(&self) -> DiagRef<'_, T, Rows> {
		DiagRef { inner: self.as_ref() }
	}
}

impl<T, Rows: Shape> Col<T, Rows> {
	#[inline(always)]
	/// see [`ColMut::as_ptr_mut`]
	pub fn as_ptr_mut(&mut self) -> *mut T {
		self.as_mut().as_ptr_mut()
	}

	#[inline(always)]
	/// see [`ColMut::ptr_at_mut`]
	pub fn ptr_at_mut(&mut self, row: IdxInc<Rows>) -> *mut T {
		self.as_mut().ptr_at_mut(row)
	}

	#[inline(always)]
	#[track_caller]
	/// see [`ColMut::ptr_inbounds_at_mut`]
	pub unsafe fn ptr_inbounds_at_mut(&mut self, row: Idx<Rows>) -> *mut T {
		self.as_mut().ptr_inbounds_at_mut(row)
	}

	#[inline]
	#[track_caller]
	/// see [`ColMut::split_at_row_mut`]
	pub fn split_at_row_mut(&mut self, row: IdxInc<Rows>) -> (ColMut<'_, T, usize>, ColMut<'_, T, usize>) {
		self.as_mut().split_at_row_mut(row)
	}

	#[inline(always)]
	/// see [`ColMut::transpose_mut`]
	pub fn transpose_mut(&mut self) -> RowMut<'_, T, Rows> {
		self.as_mut().transpose_mut()
	}

	#[inline(always)]
	/// see [`ColMut::conjugate_mut`]
	pub fn conjugate_mut(&mut self) -> ColMut<'_, T::Conj, Rows>
	where
		T: Conjugate,
	{
		self.as_mut().conjugate_mut()
	}

	#[inline(always)]
	/// see [`ColMut::canonical_mut`]
	pub fn canonical_mut(&mut self) -> ColMut<'_, T::Canonical, Rows>
	where
		T: Conjugate,
	{
		self.as_mut().canonical_mut()
	}

	#[inline(always)]
	/// see [`ColMut::adjoint_mut`]
	pub fn adjoint_mut(&mut self) -> RowMut<'_, T::Conj, Rows>
	where
		T: Conjugate,
	{
		self.as_mut().adjoint_mut()
	}

	#[track_caller]
	#[inline(always)]
	/// see [`ColMut::get_mut`]
	pub fn get_mut<RowRange>(&mut self, row: RowRange) -> <ColMut<'_, T, Rows> as ColIndex<RowRange>>::Target
	where
		for<'a> ColMut<'a, T, Rows>: ColIndex<RowRange>,
	{
		<ColMut<'_, T, Rows> as ColIndex<RowRange>>::get(self.as_mut(), row)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`ColMut::get_mut_unchecked`]
	pub unsafe fn get_mut_unchecked<RowRange>(&mut self, row: RowRange) -> <ColMut<'_, T, Rows> as ColIndex<RowRange>>::Target
	where
		for<'a> ColMut<'a, T, Rows>: ColIndex<RowRange>,
	{
		unsafe { <ColMut<'_, T, Rows> as ColIndex<RowRange>>::get_unchecked(self.as_mut(), row) }
	}

	#[inline]
	/// see [`ColMut::reverse_rows_mut`]
	pub fn reverse_rows_mut(&mut self) -> ColMut<'_, T, Rows> {
		self.as_mut().reverse_rows_mut()
	}

	#[inline]
	/// see [`ColMut::subrows_mut`]
	pub fn subrows_mut<V: Shape>(&mut self, row_start: IdxInc<Rows>, nrows: V) -> ColMut<'_, T, V> {
		self.as_mut().subrows_mut(row_start, nrows)
	}

	#[inline]
	/// see [`ColMut::as_row_shape_mut`]
	pub fn as_row_shape_mut<V: Shape>(&mut self, nrows: V) -> ColMut<'_, T, V> {
		self.as_mut().as_row_shape_mut(nrows)
	}

	#[inline]
	/// see [`ColMut::as_dyn_rows_mut`]
	pub fn as_dyn_rows_mut(&mut self) -> ColMut<'_, T, usize> {
		self.as_mut().as_dyn_rows_mut()
	}

	#[inline]
	/// see [`ColMut::as_dyn_stride_mut`]
	pub fn as_dyn_stride_mut(&mut self) -> ColMut<'_, T, Rows, isize> {
		self.as_mut().as_dyn_stride_mut()
	}

	#[inline]
	/// see [`ColMut::iter_mut`]
	pub fn iter_mut(&mut self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = &'_ mut T> {
		self.as_mut().iter_mut()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`ColMut::par_iter_mut`]
	pub fn par_iter_mut(&mut self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = &'_ mut T>
	where
		T: Send,
	{
		self.as_mut().par_iter_mut()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`ColMut::par_partition_mut`]
	pub fn par_partition_mut(&mut self, count: usize) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColMut<'_, T, usize>>
	where
		T: Send,
	{
		self.as_mut().par_partition_mut(count)
	}

	#[inline]
	/// see [`ColMut::as_diagonal_mut`]
	pub fn as_diagonal_mut(&mut self) -> DiagMut<'_, T, Rows> {
		self.as_mut().as_diagonal_mut()
	}

	#[inline]
	#[track_caller]
	/// see [`ColMut::copy_from`]
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

#[cfg(feature = "std")]
impl<T> From<std::vec::Vec<T>> for Col<T> {
	#[inline]
	fn from(vec: std::vec::Vec<T>) -> Self {
		let n = vec.len();
		let column = Col::from_fn(n, |i| unsafe { std::ptr::read(&vec[i as usize]) });
		std::mem::forget(vec);
		column
	}
}

#[cfg(feature = "std")]
impl<T> From<Col<T>> for std::vec::Vec<T> {
	#[inline]
	fn from(mut col: Col<T>) -> Self {
		let n = col.nrows();
		let mut vec = std::vec::Vec::with_capacity(n);
		for i in 0..n {
			unsafe {
				let ptr = col.ptr_inbounds_at_mut(i);
				vec.push(std::ptr::read(ptr));
			}
		}
		std::mem::forget(col);
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
		let col = Col::from(vec);

		// Check dimensions
		assert_eq!(col.nrows(), 5);
		assert_eq!(col.ncols(), 1);

		// Check elements
		for i in 0..5 {
			assert_eq!(col[i], i + 1);
		}

		// Test with floating point
		let vec = vec![1.0, 2.5, 3.7, 4.2, 5.9];
		let col = Col::from(vec);

		// Check dimensions
		assert_eq!(col.nrows(), 5);
		assert_eq!(col.ncols(), 1);

		// Check elements
		assert_eq!(col[0], 1.0);
		assert_eq!(col[1], 2.5);
		assert_eq!(col[2], 3.7);
		assert_eq!(col[3], 4.2);
		assert_eq!(col[4], 5.9);
	}

	#[test]
	fn test_empty_vec() {
		let vec: Vec<f64> = vec![];
		let col = Col::from(vec);

		assert_eq!(col.nrows(), 0);
		assert_eq!(col.ncols(), 1);
	}

	#[test]
	fn test_into_vec() {
		let col = Col::from_fn(5, |i| i as f64);
		let vec: Vec<f64> = col.into();

		assert_eq!(vec.len(), 5);
		for i in 0..5 {
			assert_eq!(vec[i], i as f64);
		}
	}
}
