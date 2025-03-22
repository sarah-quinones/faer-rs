use crate::internal_prelude::*;
use crate::utils::bound::{Array, Dim, Partition};
use crate::{ContiguousFwd, Idx, IdxInc};
use core::ops::{Index, IndexMut};
use equator::{assert, debug_assert};
use faer_traits::Real;

use super::RowIndex;

/// mutable view over a row vector, similar to a mutable reference to a strided
/// [prim@slice]
///
/// # note
///
/// unlike a slice, the data pointed to by `RowMut<'_, T>` is allowed to be partially or fully
/// uninitialized under certain conditions. in this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly or
/// indirectly through any of the numerical library routines, unless it is explicitly permitted
pub struct RowMut<'a, T, Cols = usize, CStride = isize> {
	pub(crate) trans: ColMut<'a, T, Cols, CStride>,
}

impl<'short, T, Rows: Copy, RStride: Copy> Reborrow<'short> for RowMut<'_, T, Rows, RStride> {
	type Target = RowRef<'short, T, Rows, RStride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		RowRef { trans: self.trans.rb() }
	}
}
impl<'short, T, Rows: Copy, RStride: Copy> ReborrowMut<'short> for RowMut<'_, T, Rows, RStride> {
	type Target = RowMut<'short, T, Rows, RStride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		RowMut { trans: self.trans.rb_mut() }
	}
}
impl<'a, T, Rows: Copy, RStride: Copy> IntoConst for RowMut<'a, T, Rows, RStride> {
	type Target = RowRef<'a, T, Rows, RStride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		RowRef {
			trans: self.trans.into_const(),
		}
	}
}

impl<'a, T, Cols: Shape, CStride: Stride> RowMut<'a, T, Cols, CStride> {
	/// creates a `RowMut` from pointers to the column vector data, number of rows, and row stride
	///
	/// # safety
	/// this function has the same safety requirements as
	/// [`MatMut::from_raw_parts_mut(ptr, 1, ncols, 0, col_stride)`]
	#[inline(always)]
	#[track_caller]
	pub unsafe fn from_raw_parts_mut(ptr: *mut T, ncols: Cols, col_stride: CStride) -> Self {
		Self {
			trans: ColMut::from_raw_parts_mut(ptr, ncols, col_stride),
		}
	}

	/// returns a pointer to the row data
	#[inline(always)]
	pub fn as_ptr(&self) -> *const T {
		self.trans.as_ptr()
	}

	/// returns the number of rows of the row (always 1)
	#[inline(always)]
	pub fn nrows(&self) -> usize {
		1
	}

	/// returns the number of columns of the row
	#[inline(always)]
	pub fn ncols(&self) -> Cols {
		self.trans.nrows()
	}

	/// returns the number of rows and columns of the row
	#[inline(always)]
	pub fn shape(&self) -> (usize, Cols) {
		(self.nrows(), self.ncols())
	}

	/// returns the column stride of the row
	#[inline(always)]
	pub fn col_stride(&self) -> CStride {
		self.trans.row_stride()
	}

	/// returns a raw pointer to the element at the given index
	#[inline(always)]
	pub fn ptr_at(&self, col: IdxInc<Cols>) -> *const T {
		self.trans.ptr_at(col)
	}

	/// returns a raw pointer to the element at the given index, assuming the provided index
	/// is within the row bounds
	///
	/// # safety
	/// the behavior is undefined if any of the following conditions are violated:
	/// * `col < self.ncols()`
	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at(&self, col: Idx<Cols>) -> *const T {
		debug_assert!(all(col < self.ncols()));
		self.trans.ptr_inbounds_at(col)
	}

	#[inline]
	#[track_caller]
	/// see [`RowRef::split_at_col`]
	pub fn split_at_col(self, col: IdxInc<Cols>) -> (RowRef<'a, T, usize, CStride>, RowRef<'a, T, usize, CStride>) {
		self.into_const().split_at_col(col)
	}

	#[inline(always)]
	/// see [`RowRef::transpose`]
	pub fn transpose(self) -> ColRef<'a, T, Cols, CStride> {
		self.into_const().transpose()
	}

	#[inline(always)]
	/// see [`RowRef::conjugate`]
	pub fn conjugate(self) -> RowRef<'a, T::Conj, Cols, CStride>
	where
		T: Conjugate,
	{
		self.into_const().conjugate()
	}

	#[inline(always)]
	/// see [`RowRef::canonical`]
	pub fn canonical(self) -> RowRef<'a, T::Canonical, Cols, CStride>
	where
		T: Conjugate,
	{
		self.into_const().canonical()
	}

	#[inline(always)]
	/// see [`RowRef::adjoint`]
	pub fn adjoint(self) -> ColRef<'a, T::Conj, Cols, CStride>
	where
		T: Conjugate,
	{
		self.into_const().adjoint()
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowRef::get`]
	pub fn get<ColRange>(self, col: ColRange) -> <RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::Target
	where
		RowRef<'a, T, Cols, CStride>: RowIndex<ColRange>,
	{
		<RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::get(self.into_const(), col)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowRef::get_unchecked`]
	pub unsafe fn get_unchecked<ColRange>(self, col: ColRange) -> <RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::Target
	where
		RowRef<'a, T, Cols, CStride>: RowIndex<ColRange>,
	{
		unsafe { <RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::get_unchecked(self.into_const(), col) }
	}

	#[inline]
	/// see [`RowRef::reverse_cols`]
	pub fn reverse_cols(self) -> RowRef<'a, T, Cols, CStride::Rev> {
		self.into_const().reverse_cols()
	}

	#[inline]
	/// see [`RowRef::subcols`]
	pub fn subcols<V: Shape>(self, col_start: IdxInc<Cols>, ncols: V) -> RowRef<'a, T, V, CStride> {
		self.into_const().subcols(col_start, ncols)
	}

	#[inline]
	#[track_caller]
	/// see [`RowRef::as_col_shape`]
	pub fn as_col_shape<V: Shape>(self, ncols: V) -> RowRef<'a, T, V, CStride> {
		self.into_const().as_col_shape(ncols)
	}

	#[inline]
	/// see [`RowRef::as_dyn_cols`]
	pub fn as_dyn_cols(self) -> RowRef<'a, T, usize, CStride> {
		self.into_const().as_dyn_cols()
	}

	#[inline]
	/// see [`RowRef::as_dyn_stride`]
	pub fn as_dyn_stride(self) -> RowRef<'a, T, Cols, isize> {
		self.into_const().as_dyn_stride()
	}

	#[inline]
	/// see [`RowRef::iter`]
	pub fn iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a T>
	where
		Cols: 'a,
	{
		self.trans.iter()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`RowRef::par_iter`]
	pub fn par_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a T>
	where
		T: Sync,
		Cols: 'a,
	{
		self.trans.par_iter()
	}

	#[inline]
	/// see [`RowRef::try_as_row_major`]
	pub fn try_as_row_major(self) -> Option<RowRef<'a, T, Cols, ContiguousFwd>> {
		self.into_const().try_as_row_major()
	}

	#[inline]
	/// see [`RowRef::as_diagonal`]
	pub fn as_diagonal(self) -> DiagRef<'a, T, Cols, CStride> {
		DiagRef {
			inner: self.trans.into_const(),
		}
	}

	#[inline(always)]
	#[doc(hidden)]
	pub unsafe fn const_cast(self) -> RowMut<'a, T, Cols, CStride> {
		RowMut {
			trans: self.trans.const_cast(),
		}
	}

	#[inline]
	/// returns a view over `self`
	pub fn as_ref(&self) -> RowRef<'_, T, Cols, CStride> {
		self.rb()
	}

	#[inline]
	/// returns a view over `self`
	pub fn as_mut(&mut self) -> RowMut<'_, T, Cols, CStride> {
		self.rb_mut()
	}

	#[inline]
	/// see [`RowRef::as_mat`]
	pub fn as_mat(self) -> MatRef<'a, T, usize, Cols, isize, CStride> {
		self.into_const().as_mat()
	}

	#[inline]
	/// see [`RowRef::as_mat`]
	pub fn as_mat_mut(self) -> MatMut<'a, T, usize, Cols, isize, CStride> {
		unsafe { self.into_const().as_mat().const_cast() }
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

impl<'a, T, Cols: Shape, CStride: Stride> RowMut<'a, T, Cols, CStride> {
	#[inline(always)]
	/// see [`RowRef::as_ptr`]
	pub fn as_ptr_mut(&self) -> *mut T {
		self.trans.as_ptr_mut()
	}

	#[inline(always)]
	/// see [`RowRef::ptr_at`]
	pub fn ptr_at_mut(&self, col: IdxInc<Cols>) -> *mut T {
		self.trans.ptr_at_mut(col)
	}

	#[inline(always)]
	#[track_caller]
	/// see [`RowRef::ptr_inbounds_at`]
	pub unsafe fn ptr_inbounds_at_mut(&self, col: Idx<Cols>) -> *mut T {
		debug_assert!(all(col < self.ncols()));
		self.trans.ptr_inbounds_at_mut(col)
	}

	#[inline]
	#[track_caller]
	/// see [`RowRef::split_at_col`]
	pub fn split_at_col_mut(self, col: IdxInc<Cols>) -> (RowMut<'a, T, usize, CStride>, RowMut<'a, T, usize, CStride>) {
		let (a, b) = self.into_const().split_at_col(col);
		unsafe { (a.const_cast(), b.const_cast()) }
	}

	#[inline(always)]
	/// see [`RowRef::transpose`]
	pub fn transpose_mut(self) -> ColMut<'a, T, Cols, CStride> {
		self.trans
	}

	#[inline(always)]
	/// see [`RowRef::conjugate`]
	pub fn conjugate_mut(self) -> RowMut<'a, T::Conj, Cols, CStride>
	where
		T: Conjugate,
	{
		unsafe { self.into_const().conjugate().const_cast() }
	}

	#[inline(always)]
	/// see [`RowRef::canonical`]
	pub fn canonical_mut(self) -> RowMut<'a, T::Canonical, Cols, CStride>
	where
		T: Conjugate,
	{
		unsafe { self.into_const().canonical().const_cast() }
	}

	#[inline(always)]
	/// see [`RowRef::adjoint`]
	pub fn adjoint_mut(self) -> ColMut<'a, T::Conj, Cols, CStride>
	where
		T: Conjugate,
	{
		unsafe { self.into_const().adjoint().const_cast() }
	}

	#[inline(always)]
	#[track_caller]
	pub(crate) fn at_mut(self, col: Idx<Cols>) -> &'a mut T {
		assert!(all(col < self.ncols()));
		unsafe { self.at_mut_unchecked(col) }
	}

	#[inline(always)]
	#[track_caller]
	pub(crate) unsafe fn at_mut_unchecked(self, col: Idx<Cols>) -> &'a mut T {
		&mut *self.ptr_inbounds_at_mut(col)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowRef::get`]
	pub fn get_mut<ColRange>(self, col: ColRange) -> <RowMut<'a, T, Cols, CStride> as RowIndex<ColRange>>::Target
	where
		RowMut<'a, T, Cols, CStride>: RowIndex<ColRange>,
	{
		<RowMut<'a, T, Cols, CStride> as RowIndex<ColRange>>::get(self, col)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowRef::get`]
	pub unsafe fn get_mut_unchecked<ColRange>(self, col: ColRange) -> <RowMut<'a, T, Cols, CStride> as RowIndex<ColRange>>::Target
	where
		RowMut<'a, T, Cols, CStride>: RowIndex<ColRange>,
	{
		unsafe { <RowMut<'a, T, Cols, CStride> as RowIndex<ColRange>>::get_unchecked(self, col) }
	}

	#[inline]
	/// see [`RowRef::reverse_cols`]
	pub fn reverse_cols_mut(self) -> RowMut<'a, T, Cols, CStride::Rev> {
		unsafe { self.into_const().reverse_cols().const_cast() }
	}

	#[inline]
	/// see [`RowRef::subcols`]
	pub fn subcols_mut<V: Shape>(self, col_start: IdxInc<Cols>, ncols: V) -> RowMut<'a, T, V, CStride> {
		unsafe { self.into_const().subcols(col_start, ncols).const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`RowRef::as_col_shape`]
	pub fn as_col_shape_mut<V: Shape>(self, ncols: V) -> RowMut<'a, T, V, CStride> {
		unsafe { self.into_const().as_col_shape(ncols).const_cast() }
	}

	#[inline]
	/// see [`RowRef::as_dyn_cols`]
	pub fn as_dyn_cols_mut(self) -> RowMut<'a, T, usize, CStride> {
		unsafe { self.into_const().as_dyn_cols().const_cast() }
	}

	#[inline]
	/// see [`RowRef::as_dyn_stride`]
	pub fn as_dyn_stride_mut(self) -> RowMut<'a, T, Cols, isize> {
		unsafe { self.into_const().as_dyn_stride().const_cast() }
	}

	#[inline]
	/// see [`RowRef::iter`]
	pub fn iter_mut(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a mut T>
	where
		Cols: 'a,
	{
		self.trans.iter_mut()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`RowRef::par_iter`]
	pub fn par_iter_mut(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a mut T>
	where
		T: Send,
		Cols: 'a,
	{
		self.trans.par_iter_mut()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`RowRef::par_partition`]
	pub fn par_partition(self, count: usize) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowRef<'a, T, usize, CStride>>
	where
		T: Sync,
		Cols: 'a,
	{
		self.into_const().par_partition(count)
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`RowRef::par_partition`]
	pub fn par_partition_mut(self, count: usize) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowMut<'a, T, usize, CStride>>
	where
		T: Send,
		Cols: 'a,
	{
		use crate::mat::matmut::SyncCell;
		use rayon::prelude::*;
		unsafe {
			self.as_type::<SyncCell<T>>()
				.into_const()
				.par_partition(count)
				.map(|col| col.const_cast().as_type::<T>())
		}
	}

	pub(crate) unsafe fn as_type<U>(self) -> RowMut<'a, U, Cols, CStride> {
		RowMut::from_raw_parts_mut(self.as_ptr_mut() as *mut U, self.ncols(), self.col_stride())
	}

	#[inline]
	/// copies `other` into `self`
	pub fn copy_from<RhsT: Conjugate<Canonical = T>>(&mut self, other: impl AsRowRef<T = RhsT, Cols = Cols>)
	where
		T: ComplexField,
	{
		self.rb_mut().transpose_mut().copy_from(other.as_row_ref().transpose());
	}

	#[inline]
	/// see [`RowRef::try_as_row_major`]
	pub fn try_as_row_major_mut(self) -> Option<RowMut<'a, T, Cols, ContiguousFwd>> {
		self.into_const().try_as_row_major().map(|x| unsafe { x.const_cast() })
	}

	#[inline]
	/// see [`RowRef::as_diagonal`]
	pub fn as_diagonal_mut(self) -> DiagMut<'a, T, Cols, CStride> {
		DiagMut { inner: self.trans }
	}

	/// fills all the elements of `self` with `value`
	pub fn fill(&mut self, value: T)
	where
		T: Clone,
	{
		self.rb_mut().transpose_mut().fill(value)
	}

	#[inline]
	pub(crate) fn __at_mut(self, i: Idx<Cols>) -> &'a mut T {
		self.at_mut(i)
	}
}

impl<'a, T, Rows: Shape> RowMut<'a, T, Rows, ContiguousFwd> {
	/// returns a reference over the elements as a slice
	#[inline]
	pub fn as_slice(self) -> &'a [T] {
		self.transpose().as_slice()
	}
}

impl<'a, 'ROWS, T> RowMut<'a, T, Dim<'ROWS>, ContiguousFwd> {
	/// returns a reference over the elements as a lifetime-bound slice
	#[inline]
	pub fn as_array(self) -> &'a Array<'ROWS, T> {
		self.transpose().as_array()
	}
}

impl<'a, T, Cols: Shape> RowMut<'a, T, Cols, ContiguousFwd> {
	/// returns a reference over the elements as a slice
	#[inline]
	pub fn as_slice_mut(self) -> &'a mut [T] {
		self.transpose_mut().as_slice_mut()
	}
}

impl<'a, 'COLS, T> RowMut<'a, T, Dim<'COLS>, ContiguousFwd> {
	/// returns a reference over the elements as a lifetime-bound slice
	#[inline]
	pub fn as_array_mut(self) -> &'a mut Array<'COLS, T> {
		self.transpose_mut().as_array_mut()
	}
}

impl<'COLS, 'a, T, CStride: Stride> RowMut<'a, T, Dim<'COLS>, CStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_cols_with<'LEFT, 'RIGHT>(
		self,
		col: Partition<'LEFT, 'RIGHT, 'COLS>,
	) -> (RowRef<'a, T, Dim<'LEFT>, CStride>, RowRef<'a, T, Dim<'RIGHT>, CStride>) {
		let (a, b) = self.split_at_col(col.midpoint());
		(a.as_col_shape(col.head), b.as_col_shape(col.tail))
	}
}

impl<'COLS, 'a, T, CStride: Stride> RowMut<'a, T, Dim<'COLS>, CStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_cols_with_mut<'LEFT, 'RIGHT>(
		self,
		col: Partition<'LEFT, 'RIGHT, 'COLS>,
	) -> (RowMut<'a, T, Dim<'LEFT>, CStride>, RowMut<'a, T, Dim<'RIGHT>, CStride>) {
		let (a, b) = self.split_at_col_mut(col.midpoint());
		(a.as_col_shape_mut(col.head), b.as_col_shape_mut(col.tail))
	}
}

impl<T: core::fmt::Debug, Cols: Shape, CStride: Stride> core::fmt::Debug for RowMut<'_, T, Cols, CStride> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.rb().fmt(f)
	}
}

impl<T, Cols: Shape, RStride: Stride> Index<Idx<Cols>> for RowRef<'_, T, Cols, RStride> {
	type Output = T;

	#[inline]
	#[track_caller]
	fn index(&self, col: Idx<Cols>) -> &Self::Output {
		self.at(col)
	}
}

impl<T, Cols: Shape, RStride: Stride> Index<Idx<Cols>> for RowMut<'_, T, Cols, RStride> {
	type Output = T;

	#[inline]
	#[track_caller]
	fn index(&self, col: Idx<Cols>) -> &Self::Output {
		self.rb().at(col)
	}
}

impl<T, Cols: Shape, RStride: Stride> IndexMut<Idx<Cols>> for RowMut<'_, T, Cols, RStride> {
	#[inline]
	#[track_caller]
	fn index_mut(&mut self, col: Idx<Cols>) -> &mut Self::Output {
		self.rb_mut().at_mut(col)
	}
}

impl<'a, T, Cols: Shape, CStride: Stride> RowMut<'a, T, Cols, CStride>
where
	T: RealField,
{
	/// Returns the maximum element in the row, or `None` if the row is empty
	pub fn max(&self) -> Option<T> {
		self.rb().as_dyn_cols().as_dyn_stride().internal_max()
	}

	/// Returns the minimum element in the row, or `None` if the row is empty
	pub fn min(&self) -> Option<T> {
		self.rb().as_dyn_cols().as_dyn_stride().internal_min()
	}
}

#[cfg(test)]
mod tests {
	use crate::Row;

	#[test]
	fn test_row_min() {
		let row: Row<f64> = Row::from_fn(5, |x| (x + 1) as f64);
		let rowmut = row.as_ref();
		assert_eq!(rowmut.min(), Some(1.0));

		let empty: Row<f64> = Row::from_fn(0, |_| 0.0);
		let emptymut = empty.as_ref();
		assert_eq!(emptymut.min(), None);
	}

	#[test]
	fn test_row_max() {
		let row: Row<f64> = Row::from_fn(5, |x| (x + 1) as f64);
		let rowmut = row.as_ref();
		assert_eq!(rowmut.max(), Some(5.0));

		let empty: Row<f64> = Row::from_fn(0, |_| 0.0);
		let emptymut = empty.as_ref();
		assert_eq!(emptymut.max(), None);
	}
}
