use super::RowIndex;
use crate::internal_prelude::*;
use crate::utils::bound::{Array, Dim, Partition};
use crate::{ContiguousFwd, Idx, IdxInc};
use equator::{assert, debug_assert};
use faer_traits::Real;

/// immutable view over a row vector, similar to an immutable reference to a strided
/// [prim@slice]
///
/// # note
///
/// unlike a slice, the data pointed to by `RowRef<'_, T>` is allowed to be partially or fully
/// uninitialized under certain conditions. in this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly or
/// indirectly through any of the numerical library routines, unless it is explicitly permitted
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
	/// creates a `RowRef` from pointers to the column vector data, number of rows, and row stride
	///
	/// # safety
	/// this function has the same safety requirements as
	/// [`MatRef::from_raw_parts(ptr, 1, ncols, 0, col_stride)`]
	#[inline(always)]
	#[track_caller]
	pub unsafe fn from_raw_parts(ptr: *const T, ncols: Cols, col_stride: CStride) -> Self {
		Self {
			trans: ColRef::from_raw_parts(ptr, ncols, col_stride),
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

	/// splits the row vertically at the given column into two parts and returns an array of
	/// each subrow, in the following order:
	/// * left
	/// * right
	///
	/// # panics
	/// the function panics if the following condition is violated:
	/// * `col <= self.ncols()`
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

	/// returns a view over the transpose of `self`
	#[inline(always)]
	pub fn transpose(self) -> ColRef<'a, T, Cols, CStride> {
		self.trans
	}

	/// returns a view over the conjugate of `self`
	#[inline(always)]
	pub fn conjugate(self) -> RowRef<'a, T::Conj, Cols, CStride>
	where
		T: Conjugate,
	{
		RowRef {
			trans: self.trans.conjugate(),
		}
	}

	/// returns an unconjugated view over `self`
	#[inline(always)]
	pub fn canonical(self) -> RowRef<'a, T::Canonical, Cols, CStride>
	where
		T: Conjugate,
	{
		RowRef {
			trans: self.trans.canonical(),
		}
	}

	/// returns a view over the conjugate transpose of `self`
	#[inline(always)]
	pub fn adjoint(self) -> ColRef<'a, T::Conj, Cols, CStride>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline(always)]
	#[track_caller]
	pub(crate) fn at(self, col: Idx<Cols>) -> &'a T {
		assert!(all(col < self.ncols()));
		unsafe { self.at_unchecked(col) }
	}

	#[inline(always)]
	#[track_caller]
	pub(crate) unsafe fn at_unchecked(self, col: Idx<Cols>) -> &'a T {
		&*self.ptr_inbounds_at(col)
	}

	/// returns a reference to the element at the given index, or a subrow if
	/// `col` is a range, with bound checks
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `col` must be contained in `[0, self.ncols())`
	#[track_caller]
	#[inline(always)]
	pub fn get<ColRange>(self, col: ColRange) -> <RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::Target
	where
		RowRef<'a, T, Cols, CStride>: RowIndex<ColRange>,
	{
		<RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::get(self, col)
	}

	/// returns a reference to the element at the given index, or a subrow if
	/// `col` is a range, without bound checks
	///
	/// # panics
	/// the behavior is undefined if any of the following conditions are violated:
	/// * `col` must be contained in `[0, self.ncols())`
	#[track_caller]
	#[inline(always)]
	pub unsafe fn get_unchecked<ColRange>(self, col: ColRange) -> <RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::Target
	where
		RowRef<'a, T, Cols, CStride>: RowIndex<ColRange>,
	{
		unsafe { <RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::get_unchecked(self, col) }
	}

	/// returns a view over the `self`, with the columns in reversed order
	#[inline]
	pub fn reverse_cols(self) -> RowRef<'a, T, Cols, CStride::Rev> {
		RowRef {
			trans: self.trans.reverse_rows(),
		}
	}

	/// returns a view over the subrow starting at column `col_start`, and with number of
	/// columns `ncols`
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `col_start <= self.ncols()`
	/// * `ncols <= self.ncols() - col_start`
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

	/// returns the input row with the given column shape after checking that it matches the
	/// current column shape
	#[inline]
	#[track_caller]
	pub fn as_col_shape<V: Shape>(self, ncols: V) -> RowRef<'a, T, V, CStride> {
		assert!(all(self.ncols().unbound() == ncols.unbound()));
		unsafe { RowRef::from_raw_parts(self.as_ptr(), ncols, self.col_stride()) }
	}

	/// returns the input row with dynamic column shape
	#[inline]
	pub fn as_dyn_cols(self) -> RowRef<'a, T, usize, CStride> {
		unsafe { RowRef::from_raw_parts(self.as_ptr(), self.ncols().unbound(), self.col_stride()) }
	}

	/// returns the input row with dynamic stride
	#[inline]
	pub fn as_dyn_stride(self) -> RowRef<'a, T, Cols, isize> {
		unsafe { RowRef::from_raw_parts(self.as_ptr(), self.ncols(), self.col_stride().element_stride()) }
	}

	/// returns an iterator over the elements of the row
	#[inline]
	pub fn iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a T>
	where
		Cols: 'a,
	{
		self.trans.iter()
	}

	/// returns a parallel iterator over the elements of the row
	#[inline]
	#[cfg(feature = "rayon")]
	pub fn par_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a T>
	where
		T: Sync,
		Cols: 'a,
	{
		self.trans.par_iter()
	}

	/// returns a parallel iterator that provides exactly `count` successive chunks of the elements
	/// of this row
	///
	/// only available with the `rayon` feature
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

	/// returns a view over the row with a static column stride equal to `+1`, or `None` otherwise
	#[inline]
	pub fn try_as_row_major(self) -> Option<RowRef<'a, T, Cols, ContiguousFwd>> {
		if self.col_stride().element_stride() == 1 {
			Some(unsafe { RowRef::from_raw_parts(self.as_ptr(), self.ncols(), ContiguousFwd) })
		} else {
			None
		}
	}

	#[inline(always)]
	#[doc(hidden)]
	pub unsafe fn const_cast(self) -> RowMut<'a, T, Cols, CStride> {
		RowMut {
			trans: self.trans.const_cast(),
		}
	}

	/// returns a view over `self`
	#[inline]
	pub fn as_ref(&self) -> RowRef<'_, T, Cols, CStride> {
		*self
	}

	/// returns a matrix view over `self`
	#[inline]
	pub fn as_mat(self) -> MatRef<'a, T, usize, Cols, isize, CStride> {
		self.transpose().as_mat().transpose()
	}

	/// interprets the row as a diagonal matrix
	#[inline]
	pub fn as_diagonal(self) -> DiagRef<'a, T, Cols, CStride> {
		DiagRef { inner: self.trans }
	}

	#[inline]
	pub(crate) fn __at(self, i: Idx<Cols>) -> &'a T {
		self.at(i)
	}

	/// returns the maximum norm of `self`
	#[inline]
	pub fn norm_max(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().norm_max()
	}

	/// returns the l2 norm of `self`
	#[inline]
	pub fn norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().norm_l2()
	}

	/// returns the squared l2 norm of `self`
	#[inline]
	pub fn squared_norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().squared_norm_l2()
	}

	/// returns the l1 norm of `self`
	#[inline]
	pub fn norm_l1(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().as_mat().norm_l1()
	}

	/// returns the sum of the elements of `self`
	#[inline]
	pub fn sum(&self) -> T::Canonical
	where
		T: Conjugate,
	{
		self.rb().as_mat().sum()
	}

	/// returns a newly allocated row holding the cloned values of `self`
	#[inline]
	pub fn cloned(&self) -> Row<T, Cols>
	where
		T: Clone,
	{
		self.transpose().cloned().into_transpose()
	}

	/// returns a newly allocated row holding the (possibly conjugated) values of `self`
	#[inline]
	pub fn to_owned(&self) -> Row<T::Canonical, Cols>
	where
		T: Conjugate,
	{
		self.transpose().to_owned().into_transpose()
	}
}

impl<'a, T, Rows: Shape> RowRef<'a, T, Rows, ContiguousFwd> {
	/// returns a reference over the elements as a slice
	#[inline]
	pub fn as_slice(self) -> &'a [T] {
		self.transpose().as_slice()
	}
}

impl<'a, 'ROWS, T> RowRef<'a, T, Dim<'ROWS>, ContiguousFwd> {
	/// returns a reference over the elements as a lifetime-bound slice
	#[inline]
	pub fn as_array(self) -> &'a Array<'ROWS, T> {
		self.transpose().as_array()
	}
}

impl<'COLS, 'a, T, CStride: Stride> RowRef<'a, T, Dim<'COLS>, CStride> {
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

impl<'a, T> RowRef<'a, T, usize, isize>
where
	T: RealField,
{
	/// Returns the maximum element in the row, or `None` if the row is empty
	pub(crate) fn internal_max(self) -> Option<T> {
		if self.nrows().unbound() == 0 || self.ncols() == 0 {
			return None;
		}

		let mut max_val = self.get(0);

		self.iter().for_each(|val| {
			if val > max_val {
				max_val = val;
			}
		});

		Some((*max_val).clone())
	}

	/// Returns the minimum element in the row, or `None` if the row is empty
	pub(crate) fn internal_min(self) -> Option<T> {
		if self.nrows().unbound() == 0 || self.ncols() == 0 {
			return None;
		}

		let mut min_val = self.get(0);

		self.iter().for_each(|val| {
			if val < min_val {
				min_val = val;
			}
		});

		Some((*min_val).clone())
	}
}

impl<'a, T, Cols: Shape, CStride: Stride> RowRef<'a, T, Cols, CStride>
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

#[cfg(test)]
mod tests {
	use crate::Row;

	#[test]
	fn test_row_min() {
		let row: Row<f64> = Row::from_fn(5, |x| (x + 1) as f64);
		let rowref = row.as_ref();
		assert_eq!(rowref.min(), Some(1.0));

		let empty: Row<f64> = Row::from_fn(0, |_| 0.0);
		let emptyref = empty.as_ref();
		assert_eq!(emptyref.min(), None);
	}

	#[test]
	fn test_row_max() {
		let row: Row<f64> = Row::from_fn(5, |x| (x + 1) as f64);
		let rowref = row.as_ref();
		assert_eq!(rowref.max(), Some(5.0));

		let empty: Row<f64> = Row::from_fn(0, |_| 0.0);
		let emptyref = empty.as_ref();
		assert_eq!(emptyref.max(), None);
	}
}
