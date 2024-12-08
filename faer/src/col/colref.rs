use super::{ColIndex, ColView};
use crate::internal_prelude::*;
use crate::utils::bound::{Array, Dim, Partition};
use crate::{ContiguousFwd, Idx, IdxInc};
use core::marker::PhantomData;
use core::ptr::NonNull;
use equator::assert;
use faer_traits::Real;
use generativity::Guard;

pub struct ColRef<'a, T, Rows = usize, RStride = isize> {
	pub(super) imp: ColView<T, Rows, RStride>,
	pub(super) __marker: PhantomData<&'a T>,
}

impl<T, Rows: Copy, RStride: Copy> Copy for ColRef<'_, T, Rows, RStride> {}
impl<T, Rows: Copy, RStride: Copy> Clone for ColRef<'_, T, Rows, RStride> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

impl<'short, T, Rows: Copy, RStride: Copy> Reborrow<'short> for ColRef<'_, T, Rows, RStride> {
	type Target = ColRef<'short, T, Rows, RStride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}
impl<'short, T, Rows: Copy, RStride: Copy> ReborrowMut<'short> for ColRef<'_, T, Rows, RStride> {
	type Target = ColRef<'short, T, Rows, RStride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}
impl<'a, T, Rows: Copy, RStride: Copy> IntoConst for ColRef<'a, T, Rows, RStride> {
	type Target = ColRef<'a, T, Rows, RStride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}

unsafe impl<T: Sync, Rows: Sync, RStride: Sync> Sync for ColRef<'_, T, Rows, RStride> {}
unsafe impl<T: Sync, Rows: Send, RStride: Send> Send for ColRef<'_, T, Rows, RStride> {}

impl<'a, T> ColRef<'a, T> {
	#[inline]
	pub fn from_slice(slice: &'a [T]) -> Self {
		let len = slice.len();
		unsafe { Self::from_raw_parts(slice.as_ptr(), len, 1) }
	}
}

impl<'a, T, Rows: Shape, RStride: Stride> ColRef<'a, T, Rows, RStride> {
	#[inline(always)]
	#[track_caller]
	pub unsafe fn from_raw_parts(ptr: *const T, nrows: Rows, row_stride: RStride) -> Self {
		Self {
			imp: ColView {
				ptr: NonNull::new_unchecked(ptr as *mut T),
				nrows,
				row_stride,
			},
			__marker: PhantomData,
		}
	}

	#[inline(always)]
	pub fn as_ptr(&self) -> *const T {
		self.imp.ptr.as_ptr() as *const T
	}

	#[inline(always)]
	pub fn nrows(&self) -> Rows {
		self.imp.nrows
	}

	#[inline(always)]
	pub fn ncols(&self) -> usize {
		1
	}

	#[inline(always)]
	pub fn shape(&self) -> (Rows, usize) {
		(self.nrows(), self.ncols())
	}

	#[inline(always)]
	pub fn row_stride(&self) -> RStride {
		self.imp.row_stride
	}

	#[inline(always)]
	pub fn ptr_at(&self, row: IdxInc<Rows>) -> *const T {
		let ptr = self.as_ptr();

		if row >= self.nrows() {
			ptr
		} else {
			ptr.wrapping_offset(row.unbound() as isize * self.row_stride().element_stride())
		}
	}

	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>) -> *const T {
		self.as_ptr().offset(row.unbound() as isize * self.row_stride().element_stride())
	}

	#[inline]
	#[track_caller]
	pub fn split_at_row(self, row: IdxInc<Rows>) -> (ColRef<'a, T, usize, RStride>, ColRef<'a, T, usize, RStride>) {
		assert!(all(row <= self.nrows()));
		let rs = self.row_stride();

		let top = self.as_ptr();
		let bot = self.ptr_at(row);
		unsafe {
			(
				ColRef::from_raw_parts(top, row.unbound(), rs),
				ColRef::from_raw_parts(bot, self.nrows().unbound() - row.unbound(), rs),
			)
		}
	}

	#[inline(always)]
	pub fn transpose(self) -> RowRef<'a, T, Rows, RStride> {
		RowRef { trans: self }
	}

	#[inline(always)]
	pub fn conjugate(self) -> ColRef<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		unsafe { ColRef::from_raw_parts(self.as_ptr() as *const T::Conj, self.nrows(), self.row_stride()) }
	}

	#[inline(always)]
	pub fn canonical(self) -> ColRef<'a, T::Canonical, Rows, RStride>
	where
		T: Conjugate,
	{
		unsafe { ColRef::from_raw_parts(self.as_ptr() as *const T::Canonical, self.nrows(), self.row_stride()) }
	}

	#[inline(always)]
	pub fn adjoint(self) -> RowRef<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline(always)]
	#[track_caller]
	pub(crate) fn at(self, row: Idx<Rows>) -> &'a T {
		assert!(all(row < self.nrows()));
		unsafe { self.at_unchecked(row) }
	}

	#[inline(always)]
	#[track_caller]
	pub(crate) unsafe fn at_unchecked(self, row: Idx<Rows>) -> &'a T {
		&*self.ptr_inbounds_at(row)
	}

	#[track_caller]
	#[inline(always)]
	pub fn get<RowRange>(self, row: RowRange) -> <ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::Target
	where
		ColRef<'a, T, Rows, RStride>: ColIndex<RowRange>,
	{
		<ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::get(self, row)
	}

	#[track_caller]
	#[inline(always)]
	pub unsafe fn get_unchecked<RowRange>(self, row: RowRange) -> <ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::Target
	where
		ColRef<'a, T, Rows, RStride>: ColIndex<RowRange>,
	{
		unsafe { <ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::get_unchecked(self, row) }
	}

	#[inline]
	pub fn reverse_rows(self) -> ColRef<'a, T, Rows, RStride::Rev> {
		let row = unsafe { IdxInc::<Rows>::new_unbound(self.nrows().unbound().saturating_sub(1)) };
		let ptr = self.ptr_at(row);
		unsafe { ColRef::from_raw_parts(ptr, self.nrows(), self.row_stride().rev()) }
	}

	#[inline]
	#[track_caller]
	pub fn subrows<V: Shape>(self, row_start: IdxInc<Rows>, nrows: V) -> ColRef<'a, T, V, RStride> {
		assert!(all(row_start <= self.nrows()));
		{
			let nrows = nrows.unbound();
			let full_nrows = self.nrows().unbound();
			let row_start = row_start.unbound();
			assert!(all(nrows <= full_nrows - row_start));
		}
		let rs = self.row_stride();

		unsafe { ColRef::from_raw_parts(self.ptr_at(row_start), nrows, rs) }
	}

	#[inline]
	#[track_caller]
	pub fn as_row_shape<V: Shape>(self, nrows: V) -> ColRef<'a, T, V, RStride> {
		assert!(all(self.nrows().unbound() == nrows.unbound()));
		unsafe { ColRef::from_raw_parts(self.as_ptr(), nrows, self.row_stride()) }
	}

	#[inline]
	pub fn as_dyn_rows(self) -> ColRef<'a, T, usize, RStride> {
		unsafe { ColRef::from_raw_parts(self.as_ptr(), self.nrows().unbound(), self.row_stride()) }
	}

	#[inline]
	pub fn as_dyn_stride(self) -> ColRef<'a, T, Rows, isize> {
		unsafe { ColRef::from_raw_parts(self.as_ptr(), self.nrows(), self.row_stride().element_stride()) }
	}

	#[inline]
	pub fn iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a T>
	where
		Rows: 'a,
	{
		Rows::indices(Rows::start(), self.nrows().end()).map(move |j| unsafe { self.at_unchecked(j) })
	}

	#[inline]
	#[cfg(feature = "rayon")]
	pub fn par_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a T>
	where
		T: Sync,
		Rows: 'a,
	{
		use rayon::prelude::*;
		(0..self.nrows().unbound())
			.into_par_iter()
			.map(move |j| unsafe { self.at_unchecked(Idx::<Rows>::new_unbound(j)) })
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	pub fn par_partition(self, count: usize) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColRef<'a, T, usize, RStride>>
	where
		T: Sync,
		Rows: 'a,
	{
		use rayon::prelude::*;

		let this = self.as_dyn_rows();

		assert!(count > 0);
		(0..count).into_par_iter().map(move |chunk_idx| {
			let (start, len) = crate::utils::thread::par_split_indices(this.nrows(), chunk_idx, count);
			this.subrows(start, len)
		})
	}

	#[inline]
	pub fn cloned(self) -> Col<T, Rows>
	where
		T: Clone,
	{
		fn imp<'M, T: Clone, RStride: Stride>(this: ColRef<'_, T, Dim<'M>, RStride>) -> Col<T, Dim<'M>> {
			Col::from_fn(this.nrows(), |i| this.at(i).clone())
		}

		with_dim!(M, self.nrows().unbound());
		imp(self.as_row_shape(M)).into_row_shape(self.nrows())
	}

	#[inline]
	pub fn to_owned(self) -> Col<T::Canonical, Rows>
	where
		T: Conjugate,
	{
		fn imp<'M, T, RStride: Stride>(this: ColRef<'_, T, Dim<'M>, RStride>) -> Col<T::Canonical, Dim<'M>>
		where
			T: Conjugate,
		{
			Col::from_fn(this.nrows(), |i| Conj::apply::<T>(this.at(i)))
		}

		with_dim!(M, self.nrows().unbound());
		imp(self.as_row_shape(M)).into_row_shape(self.nrows())
	}

	#[inline]
	pub fn try_as_col_major(self) -> Option<ColRef<'a, T, Rows, ContiguousFwd>> {
		if self.row_stride().element_stride() == 1 {
			Some(unsafe { ColRef::from_raw_parts(self.as_ptr(), self.nrows(), ContiguousFwd) })
		} else {
			None
		}
	}

	#[inline(always)]
	pub unsafe fn const_cast(self) -> ColMut<'a, T, Rows, RStride> {
		ColMut::from_raw_parts_mut(self.as_ptr() as *mut T, self.nrows(), self.row_stride())
	}

	#[inline]
	pub fn as_mat(self) -> MatRef<'a, T, Rows, usize, RStride, isize> {
		unsafe { MatRef::from_raw_parts(self.as_ptr(), self.nrows(), self.ncols(), self.row_stride(), 0) }
	}

	#[inline]
	pub fn as_ref(&self) -> ColRef<'_, T, Rows, RStride> {
		*self
	}

	#[inline]
	pub fn bind_r<'N>(self, row: Guard<'N>) -> ColRef<'a, T, Dim<'N>, RStride> {
		unsafe { ColRef::from_raw_parts(self.as_ptr(), self.nrows().bind(row), self.row_stride()) }
	}

	#[inline(always)]
	#[track_caller]
	pub fn read(&self, row: Idx<Rows>) -> T
	where
		T: Clone,
	{
		self.at(row).clone()
	}

	#[inline]
	#[track_caller]
	pub(crate) fn __at(self, i: Idx<Rows>) -> &'a T {
		self.at(i)
	}

	#[inline]
	pub fn as_diagonal(self) -> DiagRef<'a, T, Rows, RStride> {
		DiagRef { inner: self }
	}

	#[inline]
	pub fn norm_max(&self) -> Real<T>
	where
		T: Conjugate,
	{
		linalg::reductions::norm_max::norm_max(self.canonical().as_dyn_stride().as_dyn_rows().as_mat())
	}

	#[inline]
	pub fn norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		linalg::reductions::norm_l2::norm_l2(self.canonical().as_dyn_stride().as_dyn_rows().as_mat())
	}

	#[inline]
	pub fn norm_l2_squared(&self) -> Real<T>
	where
		T: Conjugate,
	{
		linalg::reductions::norm_l2_sqr::norm_l2_sqr(self.canonical().as_dyn_stride().as_dyn_rows().as_mat())
	}
}

impl<'a, T, Rows: Shape> ColRef<'a, T, Rows, ContiguousFwd> {
	#[inline]
	pub fn as_slice(self) -> &'a [T] {
		unsafe { core::slice::from_raw_parts(self.as_ptr(), self.nrows().unbound()) }
	}
}

impl<'a, 'ROWS, T> ColRef<'a, T, Dim<'ROWS>, ContiguousFwd> {
	#[inline]
	pub fn as_array(self) -> &'a Array<'ROWS, T> {
		unsafe { &*(self.as_slice() as *const [_] as *const Array<'ROWS, T>) }
	}
}

impl<'ROWS, 'a, T, RStride: Stride> ColRef<'a, T, Dim<'ROWS>, RStride> {
	#[inline]
	pub fn split_rows_with<'TOP, 'BOT>(
		self,
		row: Partition<'TOP, 'BOT, 'ROWS>,
	) -> (ColRef<'a, T, Dim<'TOP>, RStride>, ColRef<'a, T, Dim<'BOT>, RStride>) {
		let (a, b) = self.split_at_row(row.midpoint());
		(a.as_row_shape(row.head), b.as_row_shape(row.tail))
	}
}

impl<T: core::fmt::Debug, Rows: Shape, RStride: Stride> core::fmt::Debug for ColRef<'_, T, Rows, RStride> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.transpose().fmt(f)
	}
}
