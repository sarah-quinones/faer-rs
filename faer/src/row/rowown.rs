use super::*;
use crate::{ContiguousFwd, Idx, IdxInc, TryReserveError, assert};

/// see [`super::Row`]
#[derive(Clone)]
pub struct Own<T, Cols: Shape = usize> {
	pub(crate) trans: Col<T, Cols>,
}

impl<T, Cols: Shape> Row<T, Cols> {
	/// returns a new row with dimension `ncols`, filled with the provided function
	#[inline]
	pub fn from_fn(ncols: Cols, f: impl FnMut(Idx<Cols>) -> T) -> Self {
		Self {
			0: Own {
				trans: Col::from_fn(ncols, f),
			},
		}
	}

	/// returns a new row with dimension `ncols`, filled with zeros
	#[inline]
	pub fn zeros(ncols: Cols) -> Self
	where
		T: ComplexField,
	{
		Self {
			0: Own { trans: Col::zeros(ncols) },
		}
	}

	/// returns a new row with dimension `ncols`, filled with ones
	#[inline]
	pub fn ones(ncols: Cols) -> Self
	where
		T: ComplexField,
	{
		Self {
			0: Own { trans: Col::ones(ncols) },
		}
	}

	/// returns a new row with dimension `ncols`, filled with `value`
	#[inline]
	pub fn full(ncols: Cols, value: T) -> Self
	where
		T: Clone,
	{
		Self {
			0: Own {
				trans: Col::full(ncols, value),
			},
		}
	}

	/// reserves the minimum capacity for `col_capacity` columns without reallocating, or returns an
	/// error in case of failure. does nothing if the capacity is already sufficient
	#[inline]
	pub fn try_reserve(&mut self, new_row_capacity: usize) -> Result<(), TryReserveError> {
		self.0.trans.try_reserve(new_row_capacity)
	}

	/// reserves the minimum capacity for `col_capacity` columns without reallocating. does nothing
	/// if the capacity is already sufficient
	#[track_caller]
	pub fn reserve(&mut self, new_row_capacity: usize) {
		self.0.trans.reserve(new_row_capacity)
	}

	/// resizes the row in-place so that the new dimension is `new_ncols`.
	/// new elements are created with the given function `f`, so that elements at index `j`
	/// are created by calling `f(j)`
	#[inline]
	pub fn resize_with(&mut self, new_ncols: Cols, f: impl FnMut(Idx<Cols>) -> T) {
		self.0.trans.resize_with(new_ncols, f);
	}

	/// truncates the row so that its new dimensions are `new_ncols`.  
	/// the new dimension must be smaller than or equal to the current dimension
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// - `new_ncols > self.ncols()`
	#[inline]
	pub fn truncate(&mut self, new_ncols: Cols) {
		self.0.trans.truncate(new_ncols);
	}

	/// see [`RowRef::as_col_shape`]
	#[inline]
	pub fn into_col_shape<V: Shape>(self, ncols: V) -> Row<T, V> {
		assert!(all(self.ncols().unbound() == ncols.unbound()));
		Row {
			0: Own {
				trans: self.0.trans.into_row_shape(ncols),
			},
		}
	}

	/// see [`RowRef::as_diagonal`]
	#[inline]
	pub fn into_diagonal(self) -> Diag<T, Cols> {
		Diag {
			0: crate::diag::Own { inner: self.0.trans },
		}
	}

	/// see [`RowRef::transpose`]
	#[inline]
	pub fn into_transpose(self) -> Col<T, Cols> {
		self.0.trans
	}
}

impl<T, Cols: Shape> Row<T, Cols> {
	/// returns the number of rows of the row (always 1)
	#[inline]
	pub fn nrows(&self) -> usize {
		self.0.trans.ncols()
	}

	/// returns the number of columns of the row
	#[inline]
	pub fn ncols(&self) -> Cols {
		self.0.trans.nrows()
	}
}

impl<T: core::fmt::Debug, Cols: Shape> core::fmt::Debug for Own<T, Cols> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.rb().fmt(f)
	}
}

impl<T, Cols: Shape> Row<T, Cols> {
	#[inline(always)]
	/// see [`RowRef::as_ptr`]
	pub fn as_ptr(&self) -> *const T {
		self.rb().as_ptr()
	}

	#[inline(always)]
	/// see [`RowRef::shape`]
	pub fn shape(&self) -> (usize, Cols) {
		self.rb().shape()
	}

	#[inline(always)]
	/// see [`RowRef::col_stride`]
	pub fn col_stride(&self) -> isize {
		self.rb().col_stride()
	}

	#[inline(always)]
	/// see [`RowRef::ptr_at`]
	pub fn ptr_at(&self, col: IdxInc<Cols>) -> *const T {
		self.rb().ptr_at(col)
	}

	#[inline(always)]
	#[track_caller]
	/// see [`RowRef::ptr_inbounds_at`]
	pub unsafe fn ptr_inbounds_at(&self, col: Idx<Cols>) -> *const T {
		self.rb().ptr_inbounds_at(col)
	}

	#[inline]
	#[track_caller]
	/// see [`RowRef::split_at_col`]
	pub fn split_at_col(&self, col: IdxInc<Cols>) -> (RowRef<'_, T, usize>, RowRef<'_, T, usize>) {
		self.rb().split_at_col(col)
	}

	#[inline(always)]
	/// see [`RowRef::transpose`]
	pub fn transpose(&self) -> ColRef<'_, T, Cols> {
		self.rb().transpose()
	}

	#[inline(always)]
	/// see [`RowRef::conjugate`]
	pub fn conjugate(&self) -> RowRef<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.rb().conjugate()
	}

	#[inline(always)]
	/// see [`RowRef::canonical`]
	pub fn canonical(&self) -> RowRef<'_, T::Canonical, Cols>
	where
		T: Conjugate,
	{
		self.rb().canonical()
	}

	#[inline(always)]
	/// see [`RowRef::adjoint`]
	pub fn adjoint(&self) -> ColRef<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.rb().adjoint()
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowRef::get`]
	pub fn get<ColRange>(&self, col: ColRange) -> <RowRef<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowRef<'a, T, Cols>: RowIndex<ColRange>,
	{
		<RowRef<'_, T, Cols> as RowIndex<ColRange>>::get(self.rb(), col)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowRef::get_unchecked`]
	pub unsafe fn get_unchecked<ColRange>(&self, col: ColRange) -> <RowRef<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowRef<'a, T, Cols>: RowIndex<ColRange>,
	{
		unsafe { <RowRef<'_, T, Cols> as RowIndex<ColRange>>::get_unchecked(self.rb(), col) }
	}

	#[inline]
	/// see [`RowRef::reverse_cols`]
	pub fn reverse_cols(&self) -> RowRef<'_, T, Cols> {
		self.rb().reverse_cols()
	}

	#[inline]
	/// see [`RowRef::subcols`]
	pub fn subcols<V: Shape>(&self, col_start: IdxInc<Cols>, ncols: V) -> RowRef<'_, T, V> {
		self.rb().subcols(col_start, ncols)
	}

	#[inline]
	/// see [`RowRef::as_col_shape`]
	pub fn as_col_shape<V: Shape>(&self, ncols: V) -> RowRef<'_, T, V> {
		self.rb().as_col_shape(ncols)
	}

	#[inline]
	/// see [`RowRef::as_dyn_cols`]
	pub fn as_dyn_cols(&self) -> RowRef<'_, T, usize> {
		self.rb().as_dyn_cols()
	}

	#[inline]
	/// see [`RowRef::as_dyn_stride`]
	pub fn as_dyn_stride(&self) -> RowRef<'_, T, Cols, isize> {
		self.rb().as_dyn_stride()
	}

	#[inline]
	/// see [`RowRef::iter`]
	pub fn iter(&self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = &'_ T> {
		self.rb().iter()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`RowRef::par_iter`]
	pub fn par_iter(&self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = &'_ T>
	where
		T: Sync,
	{
		self.rb().par_iter()
	}

	#[inline]
	/// see [`RowRef::try_as_row_major`]
	pub fn try_as_row_major(&self) -> Option<RowRef<'_, T, Cols, ContiguousFwd>> {
		self.rb().try_as_row_major()
	}

	#[inline]
	/// see [`RowRef::as_diagonal`]
	pub fn as_diagonal(&self) -> DiagRef<'_, T, Cols> {
		self.rb().as_diagonal()
	}

	#[inline(always)]
	/// see [`RowRef::const_cast`]
	pub unsafe fn const_cast(&self) -> RowMut<'_, T, Cols> {
		self.rb().const_cast()
	}

	#[inline]
	/// see [`RowRef::as_mat`]
	pub fn as_mat(&self) -> MatRef<'_, T, usize, Cols, isize> {
		self.rb().as_mat()
	}

	#[inline]
	/// see [`RowRef::as_mat`]
	pub fn as_mat_mut(&mut self) -> MatMut<'_, T, usize, Cols, isize> {
		self.rb_mut().as_mat_mut()
	}
}

impl<T, Cols: Shape> Row<T, Cols> {
	#[inline(always)]
	/// see [`RowMut::as_ptr_mut`]
	pub fn as_ptr_mut(&mut self) -> *mut T {
		self.rb_mut().as_ptr_mut()
	}

	#[inline(always)]
	/// see [`RowMut::ptr_at_mut`]
	pub fn ptr_at_mut(&mut self, col: IdxInc<Cols>) -> *mut T {
		self.rb_mut().ptr_at_mut(col)
	}

	#[inline(always)]
	#[track_caller]
	/// see [`RowMut::ptr_inbounds_at_mut`]
	pub unsafe fn ptr_inbounds_at_mut(&mut self, col: Idx<Cols>) -> *mut T {
		self.rb_mut().ptr_inbounds_at_mut(col)
	}

	#[inline]
	#[track_caller]
	/// see [`RowMut::split_at_col_mut`]
	pub fn split_at_col_mut(&mut self, col: IdxInc<Cols>) -> (RowMut<'_, T, usize>, RowMut<'_, T, usize>) {
		self.rb_mut().split_at_col_mut(col)
	}

	#[inline(always)]
	/// see [`RowMut::transpose_mut`]
	pub fn transpose_mut(&mut self) -> ColMut<'_, T, Cols> {
		self.rb_mut().transpose_mut()
	}

	#[inline(always)]
	/// see [`RowMut::conjugate_mut`]
	pub fn conjugate_mut(&mut self) -> RowMut<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.rb_mut().conjugate_mut()
	}

	#[inline(always)]
	/// see [`RowMut::canonical_mut`]
	pub fn canonical_mut(&mut self) -> RowMut<'_, T::Canonical, Cols>
	where
		T: Conjugate,
	{
		self.rb_mut().canonical_mut()
	}

	#[inline(always)]
	/// see [`RowMut::adjoint_mut`]
	pub fn adjoint_mut(&mut self) -> ColMut<'_, T::Conj, Cols>
	where
		T: Conjugate,
	{
		self.rb_mut().adjoint_mut()
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowMut::get_mut`]
	pub fn get_mut<ColRange>(&mut self, col: ColRange) -> <RowMut<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowMut<'a, T, Cols>: RowIndex<ColRange>,
	{
		<RowMut<'_, T, Cols> as RowIndex<ColRange>>::get(self.rb_mut(), col)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`RowMut::get_mut_unchecked`]
	pub unsafe fn get_mut_unchecked<ColRange>(&mut self, col: ColRange) -> <RowMut<'_, T, Cols> as RowIndex<ColRange>>::Target
	where
		for<'a> RowMut<'a, T, Cols>: RowIndex<ColRange>,
	{
		unsafe { <RowMut<'_, T, Cols> as RowIndex<ColRange>>::get_unchecked(self.rb_mut(), col) }
	}

	#[inline]
	/// see [`RowMut::reverse_cols_mut`]
	pub fn reverse_cols_mut(&mut self) -> RowMut<'_, T, Cols> {
		self.rb_mut().reverse_cols_mut()
	}

	#[inline]
	/// see [`RowMut::subcols_mut`]
	pub fn subcols_mut<V: Shape>(&mut self, col_start: IdxInc<Cols>, ncols: V) -> RowMut<'_, T, V> {
		self.rb_mut().subcols_mut(col_start, ncols)
	}

	#[inline]
	/// see [`RowMut::as_col_shape_mut`]
	pub fn as_col_shape_mut<V: Shape>(&mut self, ncols: V) -> RowMut<'_, T, V> {
		self.rb_mut().as_col_shape_mut(ncols)
	}

	#[inline]
	/// see [`RowMut::as_dyn_cols_mut`]
	pub fn as_dyn_cols_mut(&mut self) -> RowMut<'_, T, usize> {
		self.rb_mut().as_dyn_cols_mut()
	}

	#[inline]
	/// see [`RowMut::as_dyn_stride_mut`]
	pub fn as_dyn_stride_mut(&mut self) -> RowMut<'_, T, Cols, isize> {
		self.rb_mut().as_dyn_stride_mut()
	}

	#[inline]
	/// see [`RowMut::iter_mut`]
	pub fn iter_mut(&mut self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = &'_ mut T> {
		self.rb_mut().iter_mut()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`RowMut::par_iter_mut`]
	pub fn par_iter_mut(&mut self) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = &'_ mut T>
	where
		T: Send,
	{
		self.rb_mut().par_iter_mut()
	}

	#[inline]
	/// see [`RowMut::try_as_row_major_mut`]
	pub fn try_as_row_major_mut(&mut self) -> Option<RowMut<'_, T, Cols, ContiguousFwd>> {
		self.rb_mut().try_as_row_major_mut()
	}

	#[inline]
	/// see [`RowMut::as_diagonal_mut`]
	pub fn as_diagonal_mut(&mut self) -> DiagMut<'_, T, Cols> {
		self.rb_mut().as_diagonal_mut()
	}
}

impl<'short, T, Cols: Shape> Reborrow<'short> for Own<T, Cols> {
	type Target = Ref<'short, T, Cols>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		Ref { trans: self.trans.rb() }
	}
}
impl<'short, T, Cols: Shape> ReborrowMut<'short> for Own<T, Cols> {
	type Target = Mut<'short, T, Cols>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		Mut { trans: self.trans.rb_mut() }
	}
}

impl<T, Cols: Shape> Row<T, Cols>
where
	T: RealField,
{
	/// Returns the maximum element in the row, or `None` if the row is empty
	pub fn max(&self) -> Option<T> {
		self.as_dyn_cols().as_dyn_stride().internal_max()
	}

	/// Returns the minimum element in the row, or `None` if the row is empty
	pub fn min(&self) -> Option<T> {
		self.as_dyn_cols().as_dyn_stride().internal_min()
	}
}

impl<T> FromIterator<T> for Row<T> {
	fn from_iter<I>(iter: I) -> Self
	where
		I: IntoIterator<Item = T>,
	{
		Row {
			0: Own {
				trans: Col::from_iter_imp(iter.into_iter()),
			},
		}
	}
}

#[cfg(test)]
mod tests {
	use crate::Row;

	#[test]
	fn test_row_min() {
		let row: Row<f64> = Row::from_fn(5, |x| (x + 1) as f64);
		assert_eq!(row.min(), Some(1.0));

		let empty: Row<f64> = Row::from_fn(0, |_| 0.0);
		assert_eq!(empty.min(), None);
	}

	#[test]
	fn test_row_max() {
		let row: Row<f64> = Row::from_fn(5, |x| (x + 1) as f64);
		assert_eq!(row.max(), Some(5.0));

		let empty: Row<f64> = Row::from_fn(0, |_| 0.0);
		assert_eq!(empty.max(), None);
	}

	#[test]
	fn test_from_iter() {
		let row: Row<i32> = (0..10).collect();
		assert_eq!(row.ncols(), 10);
		assert_eq!(row[0], 0);
		assert_eq!(row[9], 9);
	}
}
