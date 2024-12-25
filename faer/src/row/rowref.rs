use super::RowIndex;
use crate::internal_prelude::*;
use crate::utils::bound::{Array, Dim, Partition};
use crate::{ContiguousFwd, Idx, IdxInc};
use equator::{assert, debug_assert};
use faer_traits::Real;

pub struct RowRef<'a, T, Cols = usize, CStride = isize> {
	pub(crate) trans: ColRef<'a, T, Cols, CStride>,
}

impl<T, Rows: Copy, CStride: Copy> Copy for RowRef<'_, T, Rows, CStride> {}
impl<T, Rows: Copy, CStride: Copy> Clone for RowRef<'_, T, Rows, CStride> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

impl<'short, T, Rows: Copy, CStride: Copy> Reborrow<'short> for RowRef<'_, T, Rows, CStride> {
	type Target = RowRef<'short, T, Rows, CStride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}
impl<'short, T, Rows: Copy, CStride: Copy> ReborrowMut<'short> for RowRef<'_, T, Rows, CStride> {
	type Target = RowRef<'short, T, Rows, CStride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}
impl<'a, T, Rows: Copy, CStride: Copy> IntoConst for RowRef<'a, T, Rows, CStride> {
	type Target = RowRef<'a, T, Rows, CStride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}

unsafe impl<T: Sync, Rows: Sync, CStride: Sync> Sync for RowRef<'_, T, Rows, CStride> {}
unsafe impl<T: Sync, Rows: Send, CStride: Send> Send for RowRef<'_, T, Rows, CStride> {}

impl<'a, T, Cols: Shape, CStride: Stride> RowRef<'a, T, Cols, CStride> {
	#[inline(always)]
	#[track_caller]
	pub unsafe fn from_raw_parts(ptr: *const T, ncols: Cols, col_stride: CStride) -> Self {
		Self {
			trans: ColRef::from_raw_parts(ptr, ncols, col_stride),
		}
	}

	#[inline(always)]
	pub fn as_ptr(&self) -> *const T {
		self.trans.as_ptr()
	}

	#[inline(always)]
	pub fn nrows(&self) -> usize {
		1
	}

	#[inline(always)]
	pub fn ncols(&self) -> Cols {
		self.trans.nrows()
	}

	#[inline(always)]
	pub fn shape(&self) -> (usize, Cols) {
		(self.nrows(), self.ncols())
	}

	#[inline(always)]
	pub fn col_stride(&self) -> CStride {
		self.trans.row_stride()
	}

	#[inline(always)]
	pub fn ptr_at(&self, col: IdxInc<Cols>) -> *const T {
		self.trans.ptr_at(col)
	}

	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at(&self, col: Idx<Cols>) -> *const T {
		debug_assert!(all(col < self.ncols()));
		self.trans.ptr_inbounds_at(col)
	}

	#[inline]
	#[track_caller]
	pub fn split_at_col(self, col: IdxInc<Cols>) -> (RowRef<'a, T, usize, CStride>, RowRef<'a, T, usize, CStride>) {
		assert!(all(col <= self.ncols()));
		let rs = self.col_stride();

		let top = self.as_ptr();
		let bot = self.ptr_at(col);
		unsafe {
			(
				RowRef::from_raw_parts(top, col.unbound(), rs),
				RowRef::from_raw_parts(bot, self.ncols().unbound() - col.unbound(), rs),
			)
		}
	}

	#[inline(always)]
	pub fn transpose(self) -> ColRef<'a, T, Cols, CStride> {
		self.trans
	}

	#[inline(always)]
	pub fn conjugate(self) -> RowRef<'a, T::Conj, Cols, CStride>
	where
		T: Conjugate,
	{
		RowRef {
			trans: self.trans.conjugate(),
		}
	}

	#[inline(always)]
	pub fn canonical(self) -> RowRef<'a, T::Canonical, Cols, CStride>
	where
		T: Conjugate,
	{
		RowRef {
			trans: self.trans.canonical(),
		}
	}

	#[inline(always)]
	pub fn adjoint(self) -> ColRef<'a, T::Conj, Cols, CStride>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline(always)]
	pub(crate) fn at(self, col: Idx<Cols>) -> &'a T {
		assert!(all(col < self.ncols()));
		unsafe { self.at_unchecked(col) }
	}

	#[inline(always)]
	pub(crate) unsafe fn at_unchecked(self, col: Idx<Cols>) -> &'a T {
		&*self.ptr_inbounds_at(col)
	}

	#[track_caller]
	#[inline(always)]
	pub fn get<ColRange>(self, col: ColRange) -> <RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::Target
	where
		RowRef<'a, T, Cols, CStride>: RowIndex<ColRange>,
	{
		<RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::get(self, col)
	}

	#[track_caller]
	#[inline(always)]
	pub unsafe fn get_unchecked<ColRange>(self, col: ColRange) -> <RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::Target
	where
		RowRef<'a, T, Cols, CStride>: RowIndex<ColRange>,
	{
		unsafe { <RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::get_unchecked(self, col) }
	}

	#[inline]
	pub fn reverse_cols(self) -> RowRef<'a, T, Cols, CStride::Rev> {
		RowRef {
			trans: self.trans.reverse_rows(),
		}
	}

	#[inline]
	pub fn subcols<V: Shape>(self, col_start: IdxInc<Cols>, ncols: V) -> RowRef<'a, T, V, CStride> {
		assert!(all(col_start <= self.ncols()));
		{
			let ncols = ncols.unbound();
			let full_ncols = self.ncols().unbound();
			let col_start = col_start.unbound();
			assert!(all(ncols <= full_ncols - col_start));
		}
		let cs = self.col_stride();
		unsafe { RowRef::from_raw_parts(self.ptr_at(col_start), ncols, cs) }
	}

	#[inline]
	#[track_caller]
	pub fn as_col_shape<V: Shape>(self, ncols: V) -> RowRef<'a, T, V, CStride> {
		assert!(all(self.ncols().unbound() == ncols.unbound()));
		unsafe { RowRef::from_raw_parts(self.as_ptr(), ncols, self.col_stride()) }
	}

	#[inline]
	pub fn as_dyn_cols(self) -> RowRef<'a, T, usize, CStride> {
		unsafe { RowRef::from_raw_parts(self.as_ptr(), self.ncols().unbound(), self.col_stride()) }
	}

	#[inline]
	pub fn as_dyn_stride(self) -> RowRef<'a, T, Cols, isize> {
		unsafe { RowRef::from_raw_parts(self.as_ptr(), self.ncols(), self.col_stride().element_stride()) }
	}

	#[inline]
	pub fn iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a T>
	where
		Cols: 'a,
	{
		self.trans.iter()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	pub fn par_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a T>
	where
		T: Sync,
		Cols: 'a,
	{
		self.trans.par_iter()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	pub fn par_partition(self, count: usize) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowRef<'a, T, usize, CStride>>
	where
		T: Sync,
		Cols: 'a,
	{
		use rayon::prelude::*;
		self.transpose().par_partition(count).map(ColRef::transpose)
	}

	#[inline]
	pub fn try_as_row_major(self) -> Option<RowRef<'a, T, Cols, ContiguousFwd>> {
		if self.col_stride().element_stride() == 1 {
			Some(unsafe { RowRef::from_raw_parts(self.as_ptr(), self.ncols(), ContiguousFwd) })
		} else {
			None
		}
	}

	#[inline(always)]
	pub unsafe fn const_cast(self) -> RowMut<'a, T, Cols, CStride> {
		RowMut {
			trans: self.trans.const_cast(),
		}
	}

	#[inline]
	pub fn as_ref(&self) -> RowRef<'_, T, Cols, CStride> {
		*self
	}

	#[inline]
	pub fn as_mat(self) -> MatRef<'a, T, usize, Cols, isize, CStride> {
		self.transpose().as_mat().transpose()
	}

	#[inline]
	pub fn as_diagonal(self) -> DiagRef<'a, T, Cols, CStride> {
		DiagRef { inner: self.trans }
	}

	#[inline]
	pub(crate) fn __at(self, i: Idx<Cols>) -> &'a T {
		self.at(i)
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

	#[inline]
	pub fn to_owned(&self) -> Row<T::Canonical, Cols>
	where
		T: Conjugate,
	{
		Row {
			trans: self.trans.to_owned(),
		}
	}
}

impl<'a, T, Rows: Shape> RowRef<'a, T, Rows, ContiguousFwd> {
	#[inline]
	pub fn as_slice(self) -> &'a [T] {
		self.transpose().as_slice()
	}
}

impl<'a, 'ROWS, T> RowRef<'a, T, Dim<'ROWS>, ContiguousFwd> {
	#[inline]
	pub fn as_array(self) -> &'a Array<'ROWS, T> {
		self.transpose().as_array()
	}
}

impl<'COLS, 'a, T, CStride: Stride> RowRef<'a, T, Dim<'COLS>, CStride> {
	#[inline]
	pub fn split_cols_with<'LEFT, 'RIGHT>(
		self,
		col: Partition<'LEFT, 'RIGHT, 'COLS>,
	) -> (RowRef<'a, T, Dim<'LEFT>, CStride>, RowRef<'a, T, Dim<'RIGHT>, CStride>) {
		let (a, b) = self.split_at_col(col.midpoint());
		(a.as_col_shape(col.head), b.as_col_shape(col.tail))
	}
}

impl<T: core::fmt::Debug, Cols: Shape, CStride: Stride> core::fmt::Debug for RowRef<'_, T, Cols, CStride> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		fn imp<T: core::fmt::Debug>(f: &mut core::fmt::Formatter<'_>, this: RowRef<'_, T, Dim<'_>>) -> core::fmt::Result {
			f.debug_list()
				.entries(this.ncols().indices().map(|j| crate::hacks::hijack_debug(this.at(j))))
				.finish()
		}

		with_dim!(N, self.ncols().unbound());
		imp(f, self.as_col_shape(N).as_dyn_stride())
	}
}
