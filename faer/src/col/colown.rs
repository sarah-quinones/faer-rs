use super::*;
use crate::{Idx, IdxInc, TryReserveError};

/// see [`super::Col`]
#[derive(Clone)]
pub struct Own<T, Rows: Shape = usize> {
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
			0: Own {
				column: Mat::from_fn(nrows, 1, idx_to_pair(f)),
			},
		}
	}

	/// returns a new column with dimension `nrows`, filled with zeros
	#[inline]
	pub fn zeros(nrows: Rows) -> Self
	where
		T: ComplexField,
	{
		Self {
			0: Own {
				column: Mat::zeros(nrows, 1),
			},
		}
	}

	/// returns a new column with dimension `nrows`, filled with ones
	#[inline]
	pub fn ones(nrows: Rows) -> Self
	where
		T: ComplexField,
	{
		Self {
			0: Own { column: Mat::ones(nrows, 1) },
		}
	}

	/// returns a new column with dimension `nrows`, filled with `value`
	#[inline]
	pub fn full(nrows: Rows, value: T) -> Self
	where
		T: Clone,
	{
		Self {
			0: Own {
				column: Mat::full(nrows, 1, value),
			},
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
			0: Own {
				column: self.0.column.into_shape(nrows, 1),
			},
		}
	}

	/// see [`ColRef::as_diagonal`]
	#[inline]
	pub fn into_diagonal(self) -> Diag<T, Rows> {
		Diag {
			0: crate::diag::Own { inner: self },
		}
	}

	/// see [`ColRef::transpose`]
	#[inline]
	pub fn into_transpose(self) -> Row<T, Rows> {
		Row {
			0: crate::row::Own { trans: self },
		}
	}
}

impl<T> Col<T> {
	pub(crate) fn from_iter_imp<I: Iterator<Item = T>>(iter: I) -> Self {
		let mut iter = iter;
		let (min, max) = iter.size_hint();

		if max == Some(min) {
			// optimization for exact len iterators

			let cap = min;
			let mut col = Mat::<T>::with_capacity(cap, 1);

			let mut count = 0;
			iter.take(cap).for_each(|item| unsafe {
				col.as_ptr_mut().add(count).write(item);
				count += 1;

				if const { core::mem::needs_drop::<T>() } {
					col.set_dims(count, 1);
				}
			});
			unsafe {
				col.set_dims(count, 1);
			}
			assert_eq!(count, cap);

			Col { 0: Own { column: col } }
		} else {
			let mut cap = Ord::max(4, min);
			let mut col = Mat::<T>::with_capacity(cap, 1);

			let mut count = 0;
			loop {
				let expected = cap;
				iter.by_ref().take(cap - count).for_each(|item| unsafe {
					col.as_ptr_mut().add(count).write(item);
					count += 1;
					if const { core::mem::needs_drop::<T>() } {
						col.set_dims(count, 1);
					}
				});
				unsafe {
					col.set_dims(count, 1);
				}

				if count < expected {
					break;
				}
				if let Some(item) = iter.next() {
					cap = cap.checked_mul(2).unwrap();
					col.reserve(cap, 1);

					unsafe {
						col.as_ptr_mut().add(count).write(item);
						count += 1;
						col.set_dims(count, 1);
					}
				} else {
					break;
				}
			}

			Col { 0: Own { column: col } }
		}
	}
}

impl<T> FromIterator<T> for Col<T> {
	fn from_iter<I>(iter: I) -> Self
	where
		I: IntoIterator<Item = T>,
	{
		Self::from_iter_imp(iter.into_iter()).into()
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
}

impl<T: core::fmt::Debug, Rows: Shape> core::fmt::Debug for Own<T, Rows> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.rb().fmt(f)
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
		self.rb().as_diagonal()
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
}

impl<'short, T, Rows: Shape> Reborrow<'short> for Own<T, Rows> {
	type Target = Ref<'short, T, Rows>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		Ref {
			imp: ColView {
				nrows: self.column.nrows(),
				row_stride: 1,
				ptr: unsafe { NonNull::new_unchecked(self.column.as_ptr() as *mut T) },
			},
			__marker: core::marker::PhantomData,
		}
	}
}
impl<'short, T, Rows: Shape> ReborrowMut<'short> for Own<T, Rows> {
	type Target = Mut<'short, T, Rows>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		Mut {
			imp: ColView {
				nrows: self.column.nrows(),
				row_stride: 1,
				ptr: unsafe { NonNull::new_unchecked(self.column.as_ptr_mut()) },
			},
			__marker: core::marker::PhantomData,
		}
	}
}

impl<T, Rows: Shape> Col<T, Rows>
where
	T: RealField,
{
	/// Returns the maximum element in the column, or `None` if the column is empty
	pub fn max(&self) -> Option<T> {
		self.as_dyn_rows().as_dyn_stride().internal_max()
	}

	/// Returns the minimum element in the column, or `None` if the column is empty
	pub fn min(&self) -> Option<T> {
		self.as_dyn_rows().as_dyn_stride().internal_min()
	}
}

#[cfg(test)]
mod tests {
	use crate::Col;

	#[test]
	fn test_col_min() {
		let col: Col<f64> = Col::from_fn(5, |x| (x + 1) as f64);
		assert_eq!(col.min(), Some(1.0));

		let empty: Col<f64> = Col::from_fn(0, |_| 0.0);
		assert_eq!(empty.min(), None);
	}

	#[test]
	fn test_col_max() {
		let col: Col<f64> = Col::from_fn(5, |x| (x + 1) as f64);
		assert_eq!(col.max(), Some(5.0));

		let empty: Col<f64> = Col::from_fn(0, |_| 0.0);
		assert_eq!(empty.max(), None);
	}

	#[test]
	fn test_from_iter() {
		let col: Col<i32> = (0..10).collect();
		assert_eq!(col.nrows(), 10);
		assert_eq!(col[0], 0);
		assert_eq!(col[9], 9);
	}
}
