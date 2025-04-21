use super::*;
use crate::mat::matmut::SyncCell;
use crate::utils::bound::{Array, Dim, Partition};
use crate::{ContiguousFwd, Idx, IdxInc};
use core::marker::PhantomData;
use core::ptr::NonNull;
use equator::assert;
use generativity::Guard;

/// see [`super::ColMut`]
pub struct Mut<'a, T, Rows = usize, RStride = isize> {
	pub(super) imp: ColView<T, Rows, RStride>,
	pub(super) __marker: PhantomData<&'a mut T>,
}

impl<'short, T, Rows: Copy, RStride: Copy> Reborrow<'short> for Mut<'_, T, Rows, RStride> {
	type Target = Ref<'short, T, Rows, RStride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		Ref {
			imp: self.imp,
			__marker: PhantomData,
		}
	}
}
impl<'short, T, Rows: Copy, RStride: Copy> ReborrowMut<'short> for Mut<'_, T, Rows, RStride> {
	type Target = Mut<'short, T, Rows, RStride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		Mut {
			imp: self.imp,
			__marker: PhantomData,
		}
	}
}
impl<'a, T, Rows: Copy, RStride: Copy> IntoConst for Mut<'a, T, Rows, RStride> {
	type Target = Ref<'a, T, Rows, RStride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		Ref {
			imp: self.imp,
			__marker: PhantomData,
		}
	}
}

unsafe impl<T: Sync, Rows: Sync, RStride: Sync> Sync for Mut<'_, T, Rows, RStride> {}
unsafe impl<T: Send, Rows: Send, RStride: Send> Send for Mut<'_, T, Rows, RStride> {}

impl<'a, T> ColMut<'a, T> {
	/// creates a column view over the given element
	#[inline]
	pub fn from_mut(value: &'a mut T) -> Self {
		unsafe { ColMut::from_raw_parts_mut(value as *mut T, 1, 1) }
	}

	/// creates a `ColMut` from slice views over the column vector data, the result has the same
	/// number of rows as the length of the input slice
	#[inline]
	pub fn from_slice_mut(slice: &'a mut [T]) -> Self {
		let len = slice.len();
		unsafe { Self::from_raw_parts_mut(slice.as_mut_ptr(), len, 1) }
	}
}

impl<'a, T, Rows: Shape, RStride: Stride> ColMut<'a, T, Rows, RStride> {
	/// creates a `ColMut` from pointers to the column vector data, number of rows, and row stride
	///
	/// # safety
	/// this function has the same safety requirements as
	/// [`MatMut::from_raw_parts(ptr, nrows, 1, row_stride, 0)`]
	#[inline(always)]
	#[track_caller]
	pub const unsafe fn from_raw_parts_mut(ptr: *mut T, nrows: Rows, row_stride: RStride) -> Self {
		Self {
			0: Mut {
				imp: ColView {
					ptr: NonNull::new_unchecked(ptr),
					nrows,
					row_stride,
				},
				__marker: PhantomData,
			},
		}
	}

	/// returns a pointer to the column data
	#[inline(always)]
	pub fn as_ptr(&self) -> *const T {
		self.rb().as_ptr()
	}

	/// returns the number of rows of the column
	#[inline(always)]
	pub fn nrows(&self) -> Rows {
		self.imp.nrows
	}

	/// returns the number of columns of the column (always `1`)
	#[inline(always)]
	pub fn ncols(&self) -> usize {
		1
	}

	/// returns the number of rows and columns of the column
	#[inline(always)]
	pub fn shape(&self) -> (Rows, usize) {
		(self.nrows(), self.ncols())
	}

	/// returns the row stride of the column, specified in number of elements, not in bytes
	#[inline(always)]
	pub fn row_stride(&self) -> RStride {
		self.imp.row_stride
	}

	/// returns a raw pointer to the element at the given index
	#[inline(always)]
	pub fn ptr_at(&self, row: IdxInc<Rows>) -> *const T {
		self.rb().ptr_at(row)
	}

	/// returns a raw pointer to the element at the given index, assuming the provided index
	/// is within the column bounds
	///
	/// # safety
	/// the behavior is undefined if any of the following conditions are violated:
	/// * `row < self.nrows()`
	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>) -> *const T {
		self.rb().ptr_inbounds_at(row)
	}

	#[inline]
	#[track_caller]
	/// see [`ColRef::split_at_row`]
	pub fn split_at_row(self, row: IdxInc<Rows>) -> (ColRef<'a, T, usize, RStride>, ColRef<'a, T, usize, RStride>) {
		self.into_const().split_at_row(row)
	}

	#[inline(always)]
	/// see [`ColRef::transpose`]
	pub fn transpose(self) -> RowRef<'a, T, Rows, RStride> {
		self.into_const().transpose()
	}

	#[inline(always)]
	/// see [`ColRef::conjugate`]
	pub fn conjugate(self) -> ColRef<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		self.into_const().conjugate()
	}

	#[inline(always)]
	/// see [`ColRef::canonical`]
	pub fn canonical(self) -> ColRef<'a, T::Canonical, Rows, RStride>
	where
		T: Conjugate,
	{
		self.into_const().canonical()
	}

	#[inline(always)]
	/// see [`ColRef::adjoint`]
	pub fn adjoint(self) -> RowRef<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		self.into_const().adjoint()
	}

	#[track_caller]
	#[inline(always)]
	/// see [`ColRef::get`]
	pub fn get<RowRange>(self, row: RowRange) -> <ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::Target
	where
		ColRef<'a, T, Rows, RStride>: ColIndex<RowRange>,
	{
		<ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::get(self.into_const(), row)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`ColRef::get_unchecked`]
	pub unsafe fn get_unchecked<RowRange>(self, row: RowRange) -> <ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::Target
	where
		ColRef<'a, T, Rows, RStride>: ColIndex<RowRange>,
	{
		unsafe { <ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::get_unchecked(self.into_const(), row) }
	}

	#[inline]
	/// see [`ColRef::reverse_rows`]
	pub fn reverse_rows(self) -> ColRef<'a, T, Rows, RStride::Rev> {
		self.into_const().reverse_rows()
	}

	#[inline]
	/// see [`ColRef::subrows`]
	pub fn subrows<V: Shape>(self, row_start: IdxInc<Rows>, nrows: V) -> ColRef<'a, T, V, RStride> {
		self.into_const().subrows(row_start, nrows)
	}

	#[inline]
	#[track_caller]
	/// see [`ColRef::as_row_shape`]
	pub fn as_row_shape<V: Shape>(self, nrows: V) -> ColRef<'a, T, V, RStride> {
		self.into_const().as_row_shape(nrows)
	}

	#[inline]
	/// see [`ColRef::as_dyn_rows`]
	pub fn as_dyn_rows(self) -> ColRef<'a, T, usize, RStride> {
		self.into_const().as_dyn_rows()
	}

	#[inline]
	/// see [`ColRef::as_dyn_stride`]
	pub fn as_dyn_stride(self) -> ColRef<'a, T, Rows, isize> {
		self.into_const().as_dyn_stride()
	}

	#[inline]
	/// see [`ColRef::iter`]
	pub fn iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a T>
	where
		Rows: 'a,
	{
		self.into_const().iter()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`ColRef::par_iter`]
	pub fn par_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a T>
	where
		T: Sync,
		Rows: 'a,
	{
		self.into_const().par_iter()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`ColRef::par_partition`]
	pub fn par_partition(self, count: usize) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColRef<'a, T, usize, RStride>>
	where
		T: Sync,
		Rows: 'a,
	{
		self.into_const().par_partition(count)
	}

	#[inline]
	/// see [`ColRef::try_as_col_major`]
	pub fn try_as_col_major(self) -> Option<ColRef<'a, T, Rows, ContiguousFwd>> {
		self.into_const().try_as_col_major()
	}

	#[inline]
	/// see [`ColRef::try_as_col_major`]
	pub fn try_as_col_major_mut(self) -> Option<ColMut<'a, T, Rows, ContiguousFwd>> {
		self.into_const().try_as_col_major().map(|x| unsafe { x.const_cast() })
	}

	#[inline(always)]
	#[doc(hidden)]
	pub unsafe fn const_cast(self) -> ColMut<'a, T, Rows, RStride> {
		self
	}

	#[inline]
	#[doc(hidden)]
	pub fn bind_r<'N>(self, row: Guard<'N>) -> ColMut<'a, T, Dim<'N>, RStride> {
		unsafe { ColMut::from_raw_parts_mut(self.as_ptr_mut(), self.nrows().bind(row), self.row_stride()) }
	}

	#[inline]
	/// see [`ColRef::as_mat`]
	pub fn as_mat(self) -> MatRef<'a, T, Rows, usize, RStride, isize> {
		self.into_const().as_mat()
	}

	#[inline]
	/// see [`ColRef::as_mat`]
	pub fn as_mat_mut(self) -> MatMut<'a, T, Rows, usize, RStride, isize> {
		unsafe { self.into_const().as_mat().const_cast() }
	}

	#[inline]
	/// see [`ColRef::as_diagonal`]
	pub fn as_diagonal(self) -> DiagRef<'a, T, Rows, RStride> {
		DiagRef {
			0: crate::diag::Ref { inner: self.into_const() },
		}
	}
}

impl<T, Rows: Shape, RStride: Stride, Inner: for<'short> ReborrowMut<'short, Target = Mut<'short, T, Rows, RStride>>> generic::Col<Inner> {
	/// copies `other` into `self`
	#[inline]
	pub fn copy_from<RhsT: Conjugate<Canonical = T>>(&mut self, other: impl AsColRef<T = RhsT, Rows = Rows>)
	where
		T: ComplexField,
	{
		let other = other.as_col_ref();
		let this = self.rb_mut();

		assert!(all(this.nrows() == other.nrows(), this.ncols() == other.ncols(),));
		let m = this.nrows();

		with_dim!(M, m.unbound());
		imp(
			self.rb_mut().as_row_shape_mut(M).as_dyn_stride_mut(),
			other.as_row_shape(M).canonical(),
			Conj::get::<RhsT>(),
		);

		pub fn imp<'M, 'N, T: ComplexField>(this: ColMut<'_, T, Dim<'M>>, other: ColRef<'_, T, Dim<'M>>, conj_: Conj) {
			match conj_ {
				Conj::No => {
					zip!(this, other).for_each(|unzip!(dst, src)| *dst = copy(&src));
				},
				Conj::Yes => {
					zip!(this, other).for_each(|unzip!(dst, src)| *dst = conj(&src));
				},
			}
		}
	}

	/// fills all the elements of `self` with `value`
	#[inline]
	pub fn fill(&mut self, value: T)
	where
		T: Clone,
	{
		fn cloner<T: Clone>(value: T) -> impl for<'a> FnMut(crate::linalg::zip::Last<&'a mut T>) {
			#[inline(always)]
			move |x| *x.0 = value.clone()
		}
		z!(self.rb_mut().as_dyn_rows_mut()).for_each(cloner::<T>(value));
	}

	#[inline]
	/// returns a view over `self`
	pub fn as_mut(&mut self) -> ColMut<'_, T, Rows, RStride> {
		self.rb_mut()
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> ColMut<'a, T, Rows, RStride> {
	#[inline(always)]
	/// see [`ColRef::as_ptr`]
	pub fn as_ptr_mut(&self) -> *mut T {
		self.rb().as_ptr() as *mut T
	}

	#[inline(always)]
	/// see [`ColRef::ptr_at`]
	pub fn ptr_at_mut(&self, row: IdxInc<Rows>) -> *mut T {
		self.rb().ptr_at(row) as *mut T
	}

	#[inline(always)]
	#[track_caller]
	/// see [`ColRef::ptr_inbounds_at`]
	pub unsafe fn ptr_inbounds_at_mut(&self, row: Idx<Rows>) -> *mut T {
		self.rb().ptr_inbounds_at(row) as *mut T
	}

	#[inline]
	#[track_caller]
	/// see [`ColRef::split_at_row`]
	pub fn split_at_row_mut(self, row: IdxInc<Rows>) -> (ColMut<'a, T, usize, RStride>, ColMut<'a, T, usize, RStride>) {
		let (a, b) = self.into_const().split_at_row(row);
		unsafe { (a.const_cast(), b.const_cast()) }
	}

	#[inline(always)]
	/// see [`ColRef::transpose`]
	pub fn transpose_mut(self) -> RowMut<'a, T, Rows, RStride> {
		unsafe { self.into_const().transpose().const_cast() }
	}

	#[inline(always)]
	/// see [`ColRef::conjugate`]
	pub fn conjugate_mut(self) -> ColMut<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		unsafe { self.into_const().conjugate().const_cast() }
	}

	#[inline(always)]
	/// see [`ColRef::canonical`]
	pub fn canonical_mut(self) -> ColMut<'a, T::Canonical, Rows, RStride>
	where
		T: Conjugate,
	{
		unsafe { self.into_const().canonical().const_cast() }
	}

	#[inline(always)]
	/// see [`ColRef::adjoint`]
	pub fn adjoint_mut(self) -> RowMut<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		unsafe { self.into_const().adjoint().const_cast() }
	}

	#[inline(always)]
	#[track_caller]
	pub(crate) fn at_mut(self, row: Idx<Rows>) -> &'a mut T {
		assert!(all(row < self.nrows()));
		unsafe { self.at_mut_unchecked(row) }
	}

	#[inline(always)]
	#[track_caller]
	pub(crate) unsafe fn at_mut_unchecked(self, row: Idx<Rows>) -> &'a mut T {
		&mut *self.ptr_inbounds_at_mut(row)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`ColRef::get`]
	pub fn get_mut<RowRange>(self, row: RowRange) -> <ColMut<'a, T, Rows, RStride> as ColIndex<RowRange>>::Target
	where
		ColMut<'a, T, Rows, RStride>: ColIndex<RowRange>,
	{
		<ColMut<'a, T, Rows, RStride> as ColIndex<RowRange>>::get(self, row)
	}

	#[track_caller]
	#[inline(always)]
	/// see [`ColRef::get_unchecked`]
	pub unsafe fn get_mut_unchecked<RowRange>(self, row: RowRange) -> <ColMut<'a, T, Rows, RStride> as ColIndex<RowRange>>::Target
	where
		ColMut<'a, T, Rows, RStride>: ColIndex<RowRange>,
	{
		unsafe { <ColMut<'a, T, Rows, RStride> as ColIndex<RowRange>>::get_unchecked(self, row) }
	}

	#[inline]
	/// see [`ColRef::reverse_rows`]
	pub fn reverse_rows_mut(self) -> ColMut<'a, T, Rows, RStride::Rev> {
		unsafe { self.into_const().reverse_rows().const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`ColRef::subrows`]
	pub fn subrows_mut<V: Shape>(self, row_start: IdxInc<Rows>, nrows: V) -> ColMut<'a, T, V, RStride> {
		unsafe { self.into_const().subrows(row_start, nrows).const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`ColRef::as_row_shape`]
	pub fn as_row_shape_mut<V: Shape>(self, nrows: V) -> ColMut<'a, T, V, RStride> {
		unsafe { self.into_const().as_row_shape(nrows).const_cast() }
	}

	#[inline]
	/// see [`ColRef::as_dyn_rows`]
	pub fn as_dyn_rows_mut(self) -> ColMut<'a, T, usize, RStride> {
		unsafe { self.into_const().as_dyn_rows().const_cast() }
	}

	#[inline]
	/// see [`ColRef::as_dyn_stride`]
	pub fn as_dyn_stride_mut(self) -> ColMut<'a, T, Rows, isize> {
		unsafe { self.into_const().as_dyn_stride().const_cast() }
	}

	#[inline]
	/// see [`ColRef::iter`]
	pub fn iter_mut(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a mut T>
	where
		Rows: 'a,
	{
		let this = self.into_const();
		Rows::indices(Rows::start(), this.nrows().end()).map(move |j| unsafe { this.const_cast().at_mut_unchecked(j) })
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`ColRef::par_iter`]
	pub fn par_iter_mut(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a mut T>
	where
		T: Send,
		Rows: 'a,
	{
		unsafe {
			let this = self.as_type::<SyncCell<T>>().into_const();

			use rayon::prelude::*;
			(0..this.nrows().unbound()).into_par_iter().map(move |j| {
				let ptr = this.const_cast().at_mut_unchecked(Idx::<Rows>::new_unbound(j));
				&mut *(ptr as *mut SyncCell<T> as *mut T)
			})
		}
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`ColRef::par_partition`]
	pub fn par_partition_mut(self, count: usize) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColMut<'a, T, usize, RStride>>
	where
		T: Send,
		Rows: 'a,
	{
		use rayon::prelude::*;
		unsafe {
			self.as_type::<SyncCell<T>>()
				.into_const()
				.par_partition(count)
				.map(|col| col.const_cast().as_type::<T>())
		}
	}

	pub(crate) unsafe fn as_type<U>(self) -> ColMut<'a, U, Rows, RStride> {
		ColMut::from_raw_parts_mut(self.as_ptr_mut() as *mut U, self.nrows(), self.row_stride())
	}

	#[inline]
	/// see [`ColRef::as_diagonal`]
	pub fn as_diagonal_mut(self) -> DiagMut<'a, T, Rows, RStride> {
		DiagMut {
			0: crate::diag::Mut { inner: self },
		}
	}

	#[inline]
	#[track_caller]
	pub(crate) fn __at_mut(self, i: Idx<Rows>) -> &'a mut T {
		self.at_mut(i)
	}
}

impl<'a, T, Rows: Shape> ColMut<'a, T, Rows, ContiguousFwd> {
	/// returns a reference over the elements as a slice
	#[inline]
	pub fn as_slice_mut(self) -> &'a mut [T] {
		unsafe { core::slice::from_raw_parts_mut(self.as_ptr_mut(), self.nrows().unbound()) }
	}
}

impl<'a, 'ROWS, T> ColMut<'a, T, Dim<'ROWS>, ContiguousFwd> {
	/// returns a reference over the elements as a lifetime-bound slice
	#[inline]
	pub fn as_array_mut(self) -> &'a mut Array<'ROWS, T> {
		unsafe { &mut *(self.as_slice_mut() as *mut [_] as *mut Array<'ROWS, T>) }
	}
}

impl<'ROWS, 'a, T, RStride: Stride> ColMut<'a, T, Dim<'ROWS>, RStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_rows_with<'TOP, 'BOT>(
		self,
		row: Partition<'TOP, 'BOT, 'ROWS>,
	) -> (ColRef<'a, T, Dim<'TOP>, RStride>, ColRef<'a, T, Dim<'BOT>, RStride>) {
		let (a, b) = self.split_at_row(row.midpoint());
		(a.as_row_shape(row.head), b.as_row_shape(row.tail))
	}
}

impl<'ROWS, 'a, T, RStride: Stride> ColMut<'a, T, Dim<'ROWS>, RStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_rows_with_mut<'TOP, 'BOT>(
		self,
		row: Partition<'TOP, 'BOT, 'ROWS>,
	) -> (ColMut<'a, T, Dim<'TOP>, RStride>, ColMut<'a, T, Dim<'BOT>, RStride>) {
		let (a, b) = self.split_at_row_mut(row.midpoint());
		(a.as_row_shape_mut(row.head), b.as_row_shape_mut(row.tail))
	}
}

impl<T: core::fmt::Debug, Rows: Shape, RStride: Stride> core::fmt::Debug for Mut<'_, T, Rows, RStride> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.rb().fmt(f)
	}
}

impl<'a, T, Rows: Shape> ColMut<'a, T, Rows>
where
	T: RealField,
{
	/// Returns the maximum element in the column, or `None` if the column is empty
	pub fn max(&self) -> Option<T> {
		self.rb().as_dyn_rows().as_dyn_stride().internal_max()
	}

	/// Returns the minimum element in the column, or `None` if the column is empty
	pub fn min(&self) -> Option<T> {
		self.rb().as_dyn_rows().as_dyn_stride().internal_min()
	}
}

#[cfg(test)]
mod tests {
	use crate::Col;

	#[test]
	fn test_col_min() {
		let mut col: Col<f64> = Col::from_fn(5, |x| (x + 1) as f64);
		let colmut = col.as_mut();
		assert_eq!(colmut.min(), Some(1.0));

		let mut empty: Col<f64> = Col::from_fn(0, |_| 0.0);
		let emptymut = empty.as_mut();
		assert_eq!(emptymut.min(), None);
	}

	#[test]
	fn test_col_max() {
		let mut col: Col<f64> = Col::from_fn(5, |x| (x + 1) as f64);
		let colmut = col.as_mut();
		assert_eq!(colmut.max(), Some(5.0));

		let mut empty: Col<f64> = Col::from_fn(0, |_| 0.0);
		let emptymut = empty.as_mut();
		assert_eq!(emptymut.max(), None);
	}
}
