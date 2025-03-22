use super::*;
use crate::internal_prelude::*;
use crate::utils::bound::{Dim, Partition};
use crate::{Conj, ContiguousFwd, Idx, IdxInc, unzip, zip};
use core::ops::{Index, IndexMut};
use equator::assert;
use faer_traits::{ComplexField, Real};
use generativity::{Guard, make_guard};
use linalg::zip::Last;
use matref::MatRef;

/// mutable view over a matrix, similar to a mutable reference to a 2d strided [prim@slice]
///
/// # note
///
/// unlike a slice, the data pointed to by `MatMut<'_, T>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, either directly or indirectly through any of the
/// numerical library routines, unless it is explicitly permitted
///
/// # move semantics
/// since `MatMut` mutably borrows data, it cannot be [`Copy`]. this means that if we pass a
/// `MatMut` to a function that takes it by value, or use a method that consumes `self` like
/// [`MatMut::transpose_mut`], this renders the original variable unusable
///
/// ```compile_fail
/// use faer::{Mat, MatMut};
///
/// fn takes_matmut(view: MatMut<'_, f64>) {}
///
/// let mut matrix = Mat::new();
/// let view = matrix.as_mut();
///
/// takes_matmut(view); // `view` is moved (passed by value)
/// takes_matmut(view); // this fails to compile since `view` was moved
/// ```
/// the way to get around it is to use the [`reborrow::ReborrowMut`] trait, which allows us to
/// mutably borrow a `MatMut` to obtain another `MatMut` for the lifetime of the borrow.
/// it's also similarly possible to immutably borrow a `MatMut` to obtain a `MatRef` for the
/// lifetime of the borrow, using [`reborrow::Reborrow`]
/// ```
/// use faer::{Mat, MatMut, MatRef};
/// use reborrow::*;
///
/// fn takes_matmut(view: MatMut<'_, f64>) {}
/// fn takes_matref(view: MatRef<'_, f64>) {}
///
/// let mut matrix = Mat::new();
/// let mut view = matrix.as_mut();
///
/// takes_matmut(view.rb_mut());
/// takes_matmut(view.rb_mut());
/// takes_matref(view.rb());
/// // view is still usable here
/// ```
pub struct MatMut<'a, T, Rows = usize, Cols = usize, RStride = isize, CStride = isize> {
	pub(super) imp: MatView<T, Rows, Cols, RStride, CStride>,
	pub(super) __marker: PhantomData<&'a mut T>,
}

#[repr(transparent)]
pub(crate) struct SyncCell<T>(T);
unsafe impl<T> Sync for SyncCell<T> {}

impl<'short, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Reborrow<'short> for MatMut<'_, T, Rows, Cols, RStride, CStride> {
	type Target = MatRef<'short, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		MatRef {
			imp: self.imp,
			__marker: PhantomData,
		}
	}
}
impl<'short, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> ReborrowMut<'short> for MatMut<'_, T, Rows, Cols, RStride, CStride> {
	type Target = MatMut<'short, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		MatMut {
			imp: self.imp,
			__marker: PhantomData,
		}
	}
}
impl<'a, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> IntoConst for MatMut<'a, T, Rows, Cols, RStride, CStride> {
	type Target = MatRef<'a, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		MatRef {
			imp: self.imp,
			__marker: PhantomData,
		}
	}
}

unsafe impl<T: Sync, Rows: Sync, Cols: Sync, RStride: Sync, CStride: Sync> Sync for MatMut<'_, T, Rows, Cols, RStride, CStride> {}
unsafe impl<T: Send, Rows: Send, Cols: Send, RStride: Send, CStride: Send> Send for MatMut<'_, T, Rows, Cols, RStride, CStride> {}

impl<'a, T> MatMut<'a, T> {
	/// equivalent to `MatMut::from_row_major_slice_mut(array.as_flattened_mut(), ROWS, COLS)`
	#[inline]
	pub fn from_row_major_array_mut<const ROWS: usize, const COLS: usize>(array: &'a mut [[T; COLS]; ROWS]) -> Self {
		unsafe { Self::from_raw_parts_mut(array as *mut _ as *mut T, ROWS, COLS, COLS as isize, 1) }
	}

	/// equivalent to `MatMut::from_column_major_slice_mut(array.as_flattened_mut(), ROWS, COLS)`
	#[inline]
	pub fn from_column_major_array_mut<const ROWS: usize, const COLS: usize>(array: &'a mut [[T; ROWS]; COLS]) -> Self {
		unsafe { Self::from_raw_parts_mut(array as *mut _ as *mut T, ROWS, COLS, 1, ROWS as isize) }
	}
}

impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> MatMut<'a, T, Rows, Cols, RStride, CStride> {
	/// creates a `MatMut` from a pointer to the matrix data, dimensions, and strides
	///
	/// the row (resp. column) stride is the offset from the memory address of a given matrix
	/// element at index `(row: i, col: j)`, to the memory address of the matrix element at
	/// index `(row: i + 1, col: 0)` (resp. `(row: 0, col: i + 1)`). this offset is specified in
	/// number of elements, not in bytes
	///
	/// # safety
	/// the behavior is undefined if any of the following conditions are violated:
	/// * for each matrix unit, the entire memory region addressed by the matrix must be contained
	/// within a single allocation, accessible in its entirety by the corresponding pointer in
	/// `ptr`
	/// * for each matrix unit, the corresponding pointer must be non null and properly aligned,
	/// even for a zero-sized matrix.
	/// * the values accessible by the matrix must be initialized at some point before they are
	///   read, or
	/// references to them are formed
	/// * no aliasing (including self aliasing) is allowed. in other words, none of the elements
	/// accessible by any matrix unit may be accessed for reads or writes by any other means for
	/// the duration of the lifetime `'a`. no two elements within a single matrix unit may point to
	/// the same address (such a thing can be achieved with a zero stride, for example), and no two
	/// matrix units may point to the same address
	///
	/// # example
	///
	/// ```
	/// use faer::{MatMut, mat};
	///
	/// // row major matrix with 2 rows, 3 columns, with a column at the end that we want to skip.
	/// // the row stride is the pointer offset from the address of 1.0 to the address of 4.0,
	/// // which is 4
	/// // the column stride is the pointer offset from the address of 1.0 to the address of 2.0,
	/// // which is 1
	/// let mut data = [[1.0, 2.0, 3.0, f64::NAN], [4.0, 5.0, 6.0, f64::NAN]];
	/// let mut matrix =
	/// 	unsafe { MatMut::from_raw_parts_mut(data.as_mut_ptr() as *mut f64, 2, 3, 4, 1) };
	///
	/// let expected = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
	/// assert_eq!(expected.as_ref(), matrix);
	/// ```
	#[inline]
	#[track_caller]
	pub unsafe fn from_raw_parts_mut(ptr: *mut T, nrows: Rows, ncols: Cols, row_stride: RStride, col_stride: CStride) -> Self {
		Self {
			imp: MatView {
				ptr: NonNull::new_unchecked(ptr),
				nrows,
				ncols,
				row_stride,
				col_stride,
			},
			__marker: PhantomData,
		}
	}

	/// returns a pointer to the matrix data
	#[inline]
	pub fn as_ptr(&self) -> *const T {
		self.imp.ptr.as_ptr()
	}

	/// returns the number of rows of the matrix
	#[inline]
	pub fn nrows(&self) -> Rows {
		self.imp.nrows
	}

	/// returns the number of columns of the matrix
	#[inline]
	pub fn ncols(&self) -> Cols {
		self.imp.ncols
	}

	/// returns the number of rows and columns of the matrix
	#[inline]
	pub fn shape(&self) -> (Rows, Cols) {
		(self.nrows(), self.ncols())
	}

	/// returns the row stride of the matrix, specified in number of elements, not in bytes
	#[inline]
	pub fn row_stride(&self) -> RStride {
		self.imp.row_stride
	}

	/// returns the column stride of the matrix, specified in number of elements, not in bytes
	#[inline]
	pub fn col_stride(&self) -> CStride {
		self.imp.col_stride
	}

	/// returns a raw pointer to the element at the given index
	#[inline]
	pub fn ptr_at(&self, row: IdxInc<Rows>, col: IdxInc<Cols>) -> *const T {
		self.rb().ptr_at(row, col)
	}

	/// returns a raw pointer to the element at the given index, assuming the provided index
	/// is within the matrix bounds
	///
	/// # safety
	/// the behavior is undefined if any of the following conditions are violated:
	/// * `row < self.nrows()`
	/// * `col < self.ncols()`
	#[inline]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>, col: Idx<Cols>) -> *const T {
		self.rb().ptr_inbounds_at(row, col)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::split_at`]
	pub fn split_at(
		self,
		row: IdxInc<Rows>,
		col: IdxInc<Cols>,
	) -> (
		MatRef<'a, T, usize, usize, RStride, CStride>,
		MatRef<'a, T, usize, usize, RStride, CStride>,
		MatRef<'a, T, usize, usize, RStride, CStride>,
		MatRef<'a, T, usize, usize, RStride, CStride>,
	) {
		self.into_const().split_at(row, col)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::split_at_row`]
	pub fn split_at_row(self, row: IdxInc<Rows>) -> (MatRef<'a, T, usize, Cols, RStride, CStride>, MatRef<'a, T, usize, Cols, RStride, CStride>) {
		self.into_const().split_at_row(row)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::split_at_col`]
	pub fn split_at_col(self, col: IdxInc<Cols>) -> (MatRef<'a, T, Rows, usize, RStride, CStride>, MatRef<'a, T, Rows, usize, RStride, CStride>) {
		self.into_const().split_at_col(col)
	}

	#[inline]
	/// see [`MatRef::transpose`]
	pub fn transpose(self) -> MatRef<'a, T, Cols, Rows, CStride, RStride> {
		MatRef {
			imp: MatView {
				ptr: self.imp.ptr,
				nrows: self.imp.ncols,
				ncols: self.imp.nrows,
				row_stride: self.imp.col_stride,
				col_stride: self.imp.row_stride,
			},
			__marker: PhantomData,
		}
	}

	#[inline]
	/// see [`MatRef::conjugate`]
	pub fn conjugate(self) -> MatRef<'a, T::Conj, Rows, Cols, RStride, CStride>
	where
		T: Conjugate,
	{
		self.into_const().conjugate()
	}

	#[inline]
	/// see [`MatRef::canonical`]
	pub fn canonical(self) -> MatRef<'a, T::Canonical, Rows, Cols, RStride, CStride>
	where
		T: Conjugate,
	{
		self.into_const().canonical()
	}

	#[inline]
	/// see [`MatRef::adjoint`]
	pub fn adjoint(self) -> MatRef<'a, T::Conj, Cols, Rows, CStride, RStride>
	where
		T: Conjugate,
	{
		self.into_const().adjoint()
	}

	#[inline]
	/// see [`MatRef::reverse_rows`]
	pub fn reverse_rows(self) -> MatRef<'a, T, Rows, Cols, RStride::Rev, CStride> {
		self.into_const().reverse_rows()
	}

	#[inline]
	/// see [`MatRef::reverse_cols`]
	pub fn reverse_cols(self) -> MatRef<'a, T, Rows, Cols, RStride, CStride::Rev> {
		self.into_const().reverse_cols()
	}

	#[inline]
	/// see [`MatRef::reverse_rows_and_cols`]
	pub fn reverse_rows_and_cols(self) -> MatRef<'a, T, Rows, Cols, RStride::Rev, CStride::Rev> {
		self.into_const().reverse_rows_and_cols()
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::submatrix`]
	pub fn submatrix<V: Shape, H: Shape>(
		self,
		row_start: IdxInc<Rows>,
		col_start: IdxInc<Cols>,
		nrows: V,
		ncols: H,
	) -> MatRef<'a, T, V, H, RStride, CStride> {
		self.into_const().submatrix(row_start, col_start, nrows, ncols)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::subrows`]
	pub fn subrows<V: Shape>(self, row_start: IdxInc<Rows>, nrows: V) -> MatRef<'a, T, V, Cols, RStride, CStride> {
		self.into_const().subrows(row_start, nrows)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::subcols`]
	pub fn subcols<H: Shape>(self, col_start: IdxInc<Cols>, ncols: H) -> MatRef<'a, T, Rows, H, RStride, CStride> {
		self.into_const().subcols(col_start, ncols)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::as_shape`]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> MatRef<'a, T, V, H, RStride, CStride> {
		self.into_const().as_shape(nrows, ncols)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::as_row_shape`]
	pub fn as_row_shape<V: Shape>(self, nrows: V) -> MatRef<'a, T, V, Cols, RStride, CStride> {
		self.into_const().as_row_shape(nrows)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::as_col_shape`]
	pub fn as_col_shape<H: Shape>(self, ncols: H) -> MatRef<'a, T, Rows, H, RStride, CStride> {
		self.into_const().as_col_shape(ncols)
	}

	#[inline]
	/// see [`MatRef::as_dyn_stride`]
	pub fn as_dyn_stride(self) -> MatRef<'a, T, Rows, Cols, isize, isize> {
		self.into_const().as_dyn_stride()
	}

	#[inline]
	/// see [`MatRef::as_dyn`]
	pub fn as_dyn(self) -> MatRef<'a, T, usize, usize, RStride, CStride> {
		self.into_const().as_dyn()
	}

	#[inline]
	/// see [`MatRef::as_dyn_rows`]
	pub fn as_dyn_rows(self) -> MatRef<'a, T, usize, Cols, RStride, CStride> {
		self.into_const().as_dyn_rows()
	}

	#[inline]
	/// see [`MatRef::as_dyn_cols`]
	pub fn as_dyn_cols(self) -> MatRef<'a, T, Rows, usize, RStride, CStride> {
		self.into_const().as_dyn_cols()
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::row`]
	pub fn row(self, i: Idx<Rows>) -> RowRef<'a, T, Cols, CStride> {
		self.into_const().row(i)
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::col`]
	pub fn col(self, j: Idx<Cols>) -> ColRef<'a, T, Rows, RStride> {
		self.into_const().col(j)
	}

	#[inline]
	/// see [`MatRef::col_iter`]
	pub fn col_iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = ColRef<'a, T, Rows, RStride>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().col_iter()
	}

	#[inline]
	/// see [`MatRef::row_iter`]
	pub fn row_iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = RowRef<'a, T, Cols, CStride>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().row_iter()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_col_iter`]
	pub fn par_col_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColRef<'a, T, Rows, RStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().par_col_iter()
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_row_iter`]
	pub fn par_row_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowRef<'a, T, Cols, CStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().par_row_iter()
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_col_chunks`]
	pub fn par_col_chunks(
		self,
		chunk_size: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, T, Rows, usize, RStride, CStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().par_col_chunks(chunk_size)
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_col_partition`]
	pub fn par_col_partition(
		self,
		count: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, T, Rows, usize, RStride, CStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().par_col_partition(count)
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_row_chunks`]
	pub fn par_row_chunks(
		self,
		chunk_size: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, T, usize, Cols, RStride, CStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().par_row_chunks(chunk_size)
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_row_partition`]
	pub fn par_row_partition(
		self,
		count: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, T, usize, Cols, RStride, CStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().par_row_partition(count)
	}

	#[inline]
	/// see [`MatRef::try_as_col_major`]
	pub fn try_as_col_major(self) -> Option<MatRef<'a, T, Rows, Cols, ContiguousFwd, CStride>> {
		self.into_const().try_as_col_major()
	}

	#[inline]
	/// see [`MatRef::try_as_row_major`]
	pub fn try_as_row_major(self) -> Option<MatRef<'a, T, Rows, Cols, RStride, ContiguousFwd>> {
		self.into_const().try_as_row_major()
	}

	#[doc(hidden)]
	#[inline]
	pub unsafe fn const_cast(self) -> MatMut<'a, T, Rows, Cols, RStride, CStride> {
		self
	}

	#[inline]
	/// returns a view over `self`
	pub fn as_ref(&self) -> MatRef<'_, T, Rows, Cols, RStride, CStride> {
		self.rb()
	}

	#[inline]
	/// returns a view over `self`
	pub fn as_mut(&mut self) -> MatMut<'_, T, Rows, Cols, RStride, CStride> {
		self.rb_mut()
	}

	/// see [`MatRef::)]`]	#[doc(hidden)]
	#[inline]
	pub fn bind<'M, 'N>(self, row: Guard<'M>, col: Guard<'N>) -> MatMut<'a, T, Dim<'M>, Dim<'N>, RStride, CStride> {
		unsafe {
			MatMut::from_raw_parts_mut(
				self.as_ptr_mut(),
				self.nrows().bind(row),
				self.ncols().bind(col),
				self.row_stride(),
				self.col_stride(),
			)
		}
	}

	#[doc(hidden)]
	#[inline]
	pub fn bind_r<'M>(self, row: Guard<'M>) -> MatMut<'a, T, Dim<'M>, Cols, RStride, CStride> {
		unsafe {
			MatMut::from_raw_parts_mut(
				self.as_ptr_mut(),
				self.nrows().bind(row),
				self.ncols(),
				self.row_stride(),
				self.col_stride(),
			)
		}
	}

	#[doc(hidden)]
	#[inline]
	pub fn bind_c<'N>(self, col: Guard<'N>) -> MatMut<'a, T, Rows, Dim<'N>, RStride, CStride> {
		unsafe {
			MatMut::from_raw_parts_mut(
				self.as_ptr_mut(),
				self.nrows(),
				self.ncols().bind(col),
				self.row_stride(),
				self.col_stride(),
			)
		}
	}

	#[inline]
	/// see [`MatRef::norm_max`]
	pub fn norm_max(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().norm_max()
	}

	#[inline]
	/// see [`MatRef::norm_l2`]
	pub fn norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().norm_l2()
	}

	#[inline]
	/// see [`MatRef::squared_norm_l2`]
	pub fn squared_norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().squared_norm_l2()
	}

	#[inline]
	/// see [`MatRef::norm_l1`]
	pub fn norm_l1(&self) -> Real<T>
	where
		T: Conjugate,
	{
		self.rb().norm_l1()
	}

	#[inline]
	#[math]
	/// see [`MatRef::sum`]
	pub fn sum(&self) -> T::Canonical
	where
		T: Conjugate,
	{
		self.rb().sum()
	}

	#[inline]
	/// see [`MatRef::determinant`]
	pub fn determinant(&self) -> T::Canonical
	where
		T: Conjugate,
	{
		self.rb().determinant()
	}

	#[track_caller]
	#[inline]
	/// see [`MatRef::get`]
	pub fn get<RowRange, ColRange>(
		self,
		row: RowRange,
		col: ColRange,
	) -> <MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::Target
	where
		MatRef<'a, T, Rows, Cols, RStride, CStride>: MatIndex<RowRange, ColRange>,
	{
		<MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::get(self.into_const(), row, col)
	}

	#[track_caller]
	#[inline]
	/// see [`MatRef::get_unchecked`]
	///
	/// # safety
	/// same as [`MatRef::get_unchecked`]
	pub unsafe fn get_unchecked<RowRange, ColRange>(
		self,
		row: RowRange,
		col: ColRange,
	) -> <MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::Target
	where
		MatRef<'a, T, Rows, Cols, RStride, CStride>: MatIndex<RowRange, ColRange>,
	{
		unsafe { <MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::get_unchecked(self.into_const(), row, col) }
	}

	/// see [`MatRef::get`]
	#[track_caller]
	#[inline]
	pub fn get_mut<RowRange, ColRange>(
		self,
		row: RowRange,
		col: ColRange,
	) -> <MatMut<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::Target
	where
		MatMut<'a, T, Rows, Cols, RStride, CStride>: MatIndex<RowRange, ColRange>,
	{
		<MatMut<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::get(self, row, col)
	}

	/// see [`MatRef::get_unchecked`]
	///
	/// # safety
	/// same as [`MatRef::get_unchecked`]
	#[track_caller]
	#[inline]
	pub unsafe fn get_mut_unchecked<RowRange, ColRange>(
		self,
		row: RowRange,
		col: ColRange,
	) -> <MatMut<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::Target
	where
		MatMut<'a, T, Rows, Cols, RStride, CStride>: MatIndex<RowRange, ColRange>,
	{
		unsafe { <MatMut<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::get_unchecked(self, row, col) }
	}

	#[inline]
	/// see [`MatRef::cloned`]
	pub fn cloned(&self) -> Mat<T, Rows, Cols>
	where
		T: Clone,
	{
		self.rb().cloned()
	}

	/// see [`MatRef::to_owned`]
	#[inline]
	pub fn to_owned(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		self.rb().to_owned()
	}
}

impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> MatMut<'a, T, Rows, Cols, RStride, CStride> {
	#[inline]
	/// see [`MatRef::as_ptr`]
	pub fn as_ptr_mut(&self) -> *mut T {
		self.imp.ptr.as_ptr()
	}

	#[inline]
	/// see [`MatRef::ptr_at`]
	pub fn ptr_at_mut(&self, row: IdxInc<Rows>, col: IdxInc<Cols>) -> *mut T {
		self.rb().ptr_at(row, col) as *mut T
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::ptr_inbounds_at`]
	pub unsafe fn ptr_inbounds_at_mut(&self, row: Idx<Rows>, col: Idx<Cols>) -> *mut T {
		self.rb().ptr_inbounds_at(row, col) as *mut T
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::split_at`]
	pub fn split_at_mut(
		self,
		row: IdxInc<Rows>,
		col: IdxInc<Cols>,
	) -> (
		MatMut<'a, T, usize, usize, RStride, CStride>,
		MatMut<'a, T, usize, usize, RStride, CStride>,
		MatMut<'a, T, usize, usize, RStride, CStride>,
		MatMut<'a, T, usize, usize, RStride, CStride>,
	) {
		let (a, b, c, d) = self.into_const().split_at(row, col);
		unsafe { (a.const_cast(), b.const_cast(), c.const_cast(), d.const_cast()) }
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::split_at_row`]
	pub fn split_at_row_mut(self, row: IdxInc<Rows>) -> (MatMut<'a, T, usize, Cols, RStride, CStride>, MatMut<'a, T, usize, Cols, RStride, CStride>) {
		let (a, b) = self.into_const().split_at_row(row);
		unsafe { (a.const_cast(), b.const_cast()) }
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::split_at_col`]
	pub fn split_at_col_mut(self, col: IdxInc<Cols>) -> (MatMut<'a, T, Rows, usize, RStride, CStride>, MatMut<'a, T, Rows, usize, RStride, CStride>) {
		let (a, b) = self.into_const().split_at_col(col);
		unsafe { (a.const_cast(), b.const_cast()) }
	}

	#[inline]
	/// see [`MatRef::transpose`]
	pub fn transpose_mut(self) -> MatMut<'a, T, Cols, Rows, CStride, RStride> {
		MatMut {
			imp: MatView {
				ptr: self.imp.ptr,
				nrows: self.imp.ncols,
				ncols: self.imp.nrows,
				row_stride: self.imp.col_stride,
				col_stride: self.imp.row_stride,
			},
			__marker: PhantomData,
		}
	}

	#[inline]
	/// see [`MatRef::conjugate`]
	pub fn conjugate_mut(self) -> MatMut<'a, T::Conj, Rows, Cols, RStride, CStride>
	where
		T: Conjugate,
	{
		unsafe { self.into_const().conjugate().const_cast() }
	}

	#[inline]
	/// see [`MatRef::canonical`]
	pub fn canonical_mut(self) -> MatMut<'a, T::Canonical, Rows, Cols, RStride, CStride>
	where
		T: Conjugate,
	{
		unsafe { self.into_const().canonical().const_cast() }
	}

	#[inline]
	/// see [`MatRef::adjoint`]
	pub fn adjoint_mut(self) -> MatMut<'a, T::Conj, Cols, Rows, CStride, RStride>
	where
		T: Conjugate,
	{
		unsafe { self.into_const().adjoint().const_cast() }
	}

	#[inline]
	#[track_caller]
	pub(crate) fn at_mut(self, row: Idx<Rows>, col: Idx<Cols>) -> &'a mut T {
		assert!(all(row < self.nrows(), col < self.ncols()));
		unsafe { self.at_mut_unchecked(row, col) }
	}

	#[inline]
	#[track_caller]
	pub(crate) unsafe fn at_mut_unchecked(self, row: Idx<Rows>, col: Idx<Cols>) -> &'a mut T {
		&mut *self.ptr_inbounds_at_mut(row, col)
	}

	#[inline]
	/// see [`MatRef::reverse_rows`]
	pub fn reverse_rows_mut(self) -> MatMut<'a, T, Rows, Cols, RStride::Rev, CStride> {
		unsafe { self.into_const().reverse_rows().const_cast() }
	}

	#[inline]
	/// see [`MatRef::reverse_cols`]
	pub fn reverse_cols_mut(self) -> MatMut<'a, T, Rows, Cols, RStride, CStride::Rev> {
		unsafe { self.into_const().reverse_cols().const_cast() }
	}

	#[inline]
	/// see [`MatRef::reverse_rows_and_cols`]
	pub fn reverse_rows_and_cols_mut(self) -> MatMut<'a, T, Rows, Cols, RStride::Rev, CStride::Rev> {
		unsafe { self.into_const().reverse_rows_and_cols().const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::submatrix`]
	pub fn submatrix_mut<V: Shape, H: Shape>(
		self,
		row_start: IdxInc<Rows>,
		col_start: IdxInc<Cols>,
		nrows: V,
		ncols: H,
	) -> MatMut<'a, T, V, H, RStride, CStride> {
		unsafe { self.into_const().submatrix(row_start, col_start, nrows, ncols).const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::subrows`]
	pub fn subrows_mut<V: Shape>(self, row_start: IdxInc<Rows>, nrows: V) -> MatMut<'a, T, V, Cols, RStride, CStride> {
		unsafe { self.into_const().subrows(row_start, nrows).const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::subcols`]
	pub fn subcols_mut<H: Shape>(self, col_start: IdxInc<Cols>, ncols: H) -> MatMut<'a, T, Rows, H, RStride, CStride> {
		unsafe { self.into_const().subcols(col_start, ncols).const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::as_shape`]
	pub fn as_shape_mut<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> MatMut<'a, T, V, H, RStride, CStride> {
		unsafe { self.into_const().as_shape(nrows, ncols).const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::as_row_shape`]
	pub fn as_row_shape_mut<V: Shape>(self, nrows: V) -> MatMut<'a, T, V, Cols, RStride, CStride> {
		unsafe { self.into_const().as_row_shape(nrows).const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::as_col_shape`]
	pub fn as_col_shape_mut<H: Shape>(self, ncols: H) -> MatMut<'a, T, Rows, H, RStride, CStride> {
		unsafe { self.into_const().as_col_shape(ncols).const_cast() }
	}

	#[inline]
	/// see [`MatRef::as_dyn_stride`]
	pub fn as_dyn_stride_mut(self) -> MatMut<'a, T, Rows, Cols, isize, isize> {
		unsafe { self.into_const().as_dyn_stride().const_cast() }
	}

	#[inline]
	/// see [`MatRef::as_dyn`]
	pub fn as_dyn_mut(self) -> MatMut<'a, T, usize, usize, RStride, CStride> {
		unsafe { self.into_const().as_dyn().const_cast() }
	}

	#[inline]
	/// see [`MatRef::as_dyn_rows`]
	pub fn as_dyn_rows_mut(self) -> MatMut<'a, T, usize, Cols, RStride, CStride> {
		unsafe { self.into_const().as_dyn_rows().const_cast() }
	}

	#[inline]
	/// see [`MatRef::as_dyn_cols`]
	pub fn as_dyn_cols_mut(self) -> MatMut<'a, T, Rows, usize, RStride, CStride> {
		unsafe { self.into_const().as_dyn_cols().const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::row`]
	pub fn row_mut(self, i: Idx<Rows>) -> RowMut<'a, T, Cols, CStride> {
		unsafe { self.into_const().row(i).const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`MatRef::col`]
	pub fn col_mut(self, j: Idx<Cols>) -> ColMut<'a, T, Rows, RStride> {
		unsafe { self.into_const().col(j).const_cast() }
	}

	#[inline]
	/// see [`MatRef::col_iter`]
	pub fn col_iter_mut(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = ColMut<'a, T, Rows, RStride>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().col_iter().map(|x| unsafe { x.const_cast() })
	}

	#[inline]
	/// see [`MatRef::row_iter`]
	pub fn row_iter_mut(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = RowMut<'a, T, Cols, CStride>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		self.into_const().row_iter().map(|x| unsafe { x.const_cast() })
	}

	#[inline]
	pub(crate) unsafe fn as_type<U>(self) -> MatMut<'a, U, Rows, Cols, RStride, CStride> {
		MatMut::from_raw_parts_mut(
			self.as_ptr_mut() as *mut U,
			self.nrows(),
			self.ncols(),
			self.row_stride(),
			self.col_stride(),
		)
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_col_iter`]
	pub fn par_col_iter_mut(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColMut<'a, T, Rows, RStride>>
	where
		T: Send,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;
		unsafe {
			self.as_type::<SyncCell<T>>()
				.into_const()
				.par_col_iter()
				.map(|x| x.const_cast())
				.map(|x| x.as_type())
		}
	}

	#[inline]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_row_iter`]
	pub fn par_row_iter_mut(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowMut<'a, T, Cols, CStride>>
	where
		T: Send,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;
		unsafe {
			self.as_type::<SyncCell<T>>()
				.into_const()
				.par_row_iter()
				.map(|x| x.const_cast())
				.map(|x| x.as_type())
		}
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_col_chunks`]
	pub fn par_col_chunks_mut(
		self,
		chunk_size: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, T, Rows, usize, RStride, CStride>>
	where
		T: Send,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;
		unsafe {
			self.as_type::<SyncCell<T>>()
				.into_const()
				.par_col_chunks(chunk_size)
				.map(|x| x.const_cast())
				.map(|x| x.as_type())
		}
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_col_partition`]
	pub fn par_col_partition_mut(
		self,
		count: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, T, Rows, usize, RStride, CStride>>
	where
		T: Send,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;
		unsafe {
			self.as_type::<SyncCell<T>>()
				.into_const()
				.par_col_partition(count)
				.map(|x| x.const_cast())
				.map(|x| x.as_type())
		}
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_row_chunks`]
	pub fn par_row_chunks_mut(
		self,
		chunk_size: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, T, usize, Cols, RStride, CStride>>
	where
		T: Send,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;
		unsafe {
			self.as_type::<SyncCell<T>>()
				.into_const()
				.par_row_chunks(chunk_size)
				.map(|x| x.const_cast())
				.map(|x| x.as_type())
		}
	}

	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	/// see [`MatRef::par_row_partition`]
	pub fn par_row_partition_mut(
		self,
		count: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, T, usize, Cols, RStride, CStride>>
	where
		T: Send,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;
		unsafe {
			self.as_type::<SyncCell<T>>()
				.into_const()
				.par_row_partition(count)
				.map(|x| x.const_cast())
				.map(|x| x.as_type())
		}
	}

	#[inline]
	/// see [`MatRef::split_first_row`]
	pub fn split_first_row_mut(self) -> Option<(RowMut<'a, T, Cols, CStride>, MatMut<'a, T, usize, Cols, RStride, CStride>)> {
		if let Some(i0) = self.nrows().idx_inc(1) {
			let (head, tail) = self.split_at_row_mut(i0);
			Some((head.row_mut(0), tail))
		} else {
			None
		}
	}

	#[inline]
	/// see [`MatRef::split_first_col`]
	pub fn split_first_col_mut(self) -> Option<(ColMut<'a, T, Rows, RStride>, MatMut<'a, T, Rows, usize, RStride, CStride>)> {
		if let Some(i0) = self.ncols().idx_inc(1) {
			let (head, tail) = self.split_at_col_mut(i0);
			Some((head.col_mut(0), tail))
		} else {
			None
		}
	}

	#[inline]
	/// see [`MatRef::split_last_row`]
	pub fn split_last_row_mut(self) -> Option<(RowMut<'a, T, Cols, CStride>, MatMut<'a, T, usize, Cols, RStride, CStride>)> {
		if self.nrows().unbound() > 0 {
			let i0 = self.nrows().checked_idx_inc(self.nrows().unbound() - 1);
			let (head, tail) = self.split_at_row_mut(i0);
			Some((tail.row_mut(0), head))
		} else {
			None
		}
	}

	#[inline]
	/// see [`MatRef::split_last_col`]
	pub fn split_last_col_mut(self) -> Option<(ColMut<'a, T, Rows, RStride>, MatMut<'a, T, Rows, usize, RStride, CStride>)> {
		if self.ncols().unbound() > 0 {
			let i0 = self.ncols().checked_idx_inc(self.ncols().unbound() - 1);
			let (head, tail) = self.split_at_col_mut(i0);
			Some((tail.col_mut(0), head))
		} else {
			None
		}
	}

	#[inline]
	/// see [`MatRef::split_first_row`]
	pub fn split_first_row(self) -> Option<(RowRef<'a, T, Cols, CStride>, MatRef<'a, T, usize, Cols, RStride, CStride>)> {
		self.into_const().split_first_row()
	}

	#[inline]
	/// see [`MatRef::split_first_col`]
	pub fn split_first_col(self) -> Option<(ColRef<'a, T, Rows, RStride>, MatRef<'a, T, Rows, usize, RStride, CStride>)> {
		self.into_const().split_first_col()
	}

	#[inline]
	/// see [`MatRef::split_last_row`]
	pub fn split_last_row(self) -> Option<(RowRef<'a, T, Cols, CStride>, MatRef<'a, T, usize, Cols, RStride, CStride>)> {
		self.into_const().split_last_row()
	}

	#[inline]
	/// see [`MatRef::split_last_col`]
	pub fn split_last_col(self) -> Option<(ColRef<'a, T, Rows, RStride>, MatRef<'a, T, Rows, usize, RStride, CStride>)> {
		self.into_const().split_last_col()
	}

	#[inline]
	/// see [`MatRef::try_as_col_major`]
	pub fn try_as_col_major_mut(self) -> Option<MatMut<'a, T, Rows, Cols, ContiguousFwd, CStride>> {
		self.into_const().try_as_col_major().map(|x| unsafe { x.const_cast() })
	}

	#[inline]
	/// see [`MatRef::try_as_row_major`]
	pub fn try_as_row_major_mut(self) -> Option<MatMut<'a, T, Rows, Cols, RStride, ContiguousFwd>> {
		self.into_const().try_as_row_major().map(|x| unsafe { x.const_cast() })
	}

	/// returns two views over the given columns
	///
	/// # panics
	/// panics if `i0 == i1`
	#[inline]
	#[track_caller]
	pub fn two_cols_mut(self, i0: Idx<Cols>, i1: Idx<Cols>) -> (ColMut<'a, T, Rows, RStride>, ColMut<'a, T, Rows, RStride>) {
		assert!(i0 != i1);
		let this = self.into_const();
		unsafe { (this.col(i0).const_cast(), this.col(i1).const_cast()) }
	}

	/// returns two views over the given rows
	///
	/// # panics
	/// panics if `i0 == i1`
	#[inline]
	#[track_caller]
	pub fn two_rows_mut(self, i0: Idx<Rows>, i1: Idx<Rows>) -> (RowMut<'a, T, Cols, CStride>, RowMut<'a, T, Cols, CStride>) {
		assert!(i0 != i1);
		let this = self.into_const();
		unsafe { (this.row(i0).const_cast(), this.row(i1).const_cast()) }
	}

	/// copies the lower triangular half of `other`, including the diagonal, into `self`
	#[inline]
	#[track_caller]
	pub fn copy_from_triangular_lower<RhsT: Conjugate<Canonical = T>>(&mut self, other: impl AsMatRef<T = RhsT, Rows = Rows, Cols = Cols>)
	where
		T: ComplexField,
	{
		let other = other.as_mat_ref();

		assert!(all(self.nrows() == other.nrows(), self.ncols() == other.ncols(),));
		let (m, n) = self.shape();

		make_guard!(M);
		make_guard!(N);
		let M = m.bind(M);
		let N = n.bind(N);
		let this = self.rb_mut().as_shape_mut(M, N).as_dyn_stride_mut();
		let other = other.as_shape(M, N);
		imp(this, other.canonical(), Conj::get::<RhsT>());

		#[math]
		pub fn imp<'M, 'N, T: ComplexField>(this: MatMut<'_, T, Dim<'M>, Dim<'N>>, other: MatRef<'_, T, Dim<'M>, Dim<'N>>, conj_: Conj) {
			match conj_ {
				Conj::No => {
					zip!(this, other).for_each_triangular_lower(crate::linalg::zip::Diag::Include, |unzip!(dst, src)| *dst = copy(&src));
				},
				Conj::Yes => {
					zip!(this, other).for_each_triangular_lower(crate::linalg::zip::Diag::Include, |unzip!(dst, src)| *dst = conj(&src));
				},
			}
		}
	}

	/// copies the upper triangular half of `other`, including the diagonal, into `self`
	#[inline]
	#[track_caller]
	pub fn copy_from_triangular_upper<RhsT: Conjugate<Canonical = T>>(&mut self, other: impl AsMatRef<T = RhsT, Rows = Rows, Cols = Cols>)
	where
		T: ComplexField,
	{
		(*self)
			.rb_mut()
			.transpose_mut()
			.copy_from_triangular_lower(other.as_mat_ref().transpose())
	}

	/// copies `other` into `self`
	#[inline]
	#[track_caller]
	pub fn copy_from<RhsT: Conjugate<Canonical = T>>(&mut self, other: impl AsMatRef<T = RhsT, Rows = Rows, Cols = Cols>)
	where
		T: ComplexField,
	{
		let other = other.as_mat_ref();

		assert!(all(self.nrows() == other.nrows(), self.ncols() == other.ncols(),));
		let (m, n) = self.shape();

		make_guard!(M);
		make_guard!(N);
		let M = m.bind(M);
		let N = n.bind(N);
		let this = self.rb_mut().as_shape_mut(M, N).as_dyn_stride_mut();
		let other = other.as_shape(M, N);
		imp(this, other.canonical(), Conj::get::<RhsT>());

		#[math]
		pub fn imp<'M, 'N, T: ComplexField>(this: MatMut<'_, T, Dim<'M>, Dim<'N>>, other: MatRef<'_, T, Dim<'M>, Dim<'N>>, conj_: Conj) {
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

	/// copies the lower triangular half of `other`, excluding the diagonal, into `self`
	#[inline]
	#[track_caller]
	pub fn copy_from_strict_triangular_lower<RhsT: Conjugate<Canonical = T>>(&mut self, other: impl AsMatRef<T = RhsT, Rows = Rows, Cols = Cols>)
	where
		T: ComplexField,
	{
		let other = other.as_mat_ref();

		assert!(all(self.nrows() == other.nrows(), self.ncols() == other.ncols(),));
		let (m, n) = self.shape();

		make_guard!(M);
		make_guard!(N);
		let M = m.bind(M);
		let N = n.bind(N);
		let this = self.rb_mut().as_shape_mut(M, N).as_dyn_stride_mut();
		let other = other.as_shape(M, N);
		imp(this, other.canonical(), Conj::get::<RhsT>());

		#[math]
		pub fn imp<'M, 'N, T: ComplexField>(this: MatMut<'_, T, Dim<'M>, Dim<'N>>, other: MatRef<'_, T, Dim<'M>, Dim<'N>>, conj_: Conj) {
			match conj_ {
				Conj::No => {
					zip!(this, other).for_each_triangular_lower(crate::linalg::zip::Diag::Skip, |unzip!(dst, src)| *dst = copy(&src));
				},
				Conj::Yes => {
					zip!(this, other).for_each_triangular_lower(crate::linalg::zip::Diag::Skip, |unzip!(dst, src)| *dst = conj(&src));
				},
			}
		}
	}

	/// copies the upper triangular half of `other`, excluding the diagonal, into `self`
	#[inline]
	#[track_caller]
	pub fn copy_from_strict_triangular_upper<RhsT: Conjugate<Canonical = T>>(&mut self, other: impl AsMatRef<T = RhsT, Rows = Rows, Cols = Cols>)
	where
		T: ComplexField,
	{
		(*self)
			.rb_mut()
			.transpose_mut()
			.copy_from_strict_triangular_lower(other.as_mat_ref().transpose())
	}

	/// fills all the elements of `self` with `value`
	#[inline]
	pub fn fill(&mut self, value: T)
	where
		T: Clone,
	{
		fn cloner<T: Clone>(value: T) -> impl for<'a> FnMut(Last<&'a mut T>) {
			#[inline]
			move |x| *x.0 = value.clone()
		}
		z!(self.rb_mut().as_dyn_mut()).for_each(cloner::<T>(value));
	}

	#[inline]
	#[track_caller]
	pub(crate) fn read(&self, row: Idx<Rows>, col: Idx<Cols>) -> T
	where
		T: Clone,
	{
		self.rb().read(row, col)
	}

	#[inline]
	#[track_caller]
	pub(crate) fn write(&mut self, i: Idx<Rows>, j: Idx<Cols>, value: T) {
		*self.rb_mut().at_mut(i, j) = value;
	}

	#[inline]
	pub(crate) fn __at_mut(self, (i, j): (Idx<Rows>, Idx<Cols>)) -> &'a mut T {
		self.at_mut(i, j)
	}
}

impl<'a, T, Rows: Shape, Cols: Shape> MatMut<'a, T, Rows, Cols> {
	/// creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
	/// the data is interpreted in a column-major format, so that the first chunk of `nrows`
	/// values from the slices goes in the first column of the matrix, the second chunk of `nrows`
	/// values goes in the second column, and so on
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `nrows * ncols == slice.len()`
	///
	/// # example
	/// ```
	/// use faer::{MatMut, mat};
	///
	/// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
	/// let view = MatMut::from_column_major_slice_mut(&mut slice, 3, 2);
	///
	/// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
	/// assert_eq!(expected, view);
	/// ```
	#[inline]
	#[track_caller]
	pub fn from_column_major_slice_mut(slice: &'a mut [T], nrows: Rows, ncols: Cols) -> Self
	where
		T: Sized,
	{
		from_slice_assert(nrows.unbound(), ncols.unbound(), slice.len());

		unsafe { Self::from_raw_parts_mut(slice.as_mut_ptr(), nrows, ncols, 1, nrows.unbound() as isize) }
	}

	/// creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
	/// the data is interpreted in a column-major format, where the beginnings of two consecutive
	/// columns are separated by `col_stride` elements.
	#[inline]
	#[track_caller]
	pub fn from_column_major_slice_with_stride_mut(slice: &'a mut [T], nrows: Rows, ncols: Cols, col_stride: usize) -> Self
	where
		T: Sized,
	{
		from_strided_column_major_slice_mut_assert(nrows.unbound(), ncols.unbound(), col_stride, slice.len());

		unsafe { Self::from_raw_parts_mut(slice.as_mut_ptr(), nrows, ncols, 1, col_stride as isize) }
	}

	/// creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
	/// the data is interpreted in a row-major format, so that the first chunk of `ncols`
	/// values from the slices goes in the first column of the matrix, the second chunk of `ncols`
	/// values goes in the second column, and so on
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `nrows * ncols == slice.len()`
	///
	/// # example
	/// ```
	/// use faer::{MatMut, mat};
	///
	/// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
	/// let view = MatMut::from_row_major_slice_mut(&mut slice, 3, 2);
	///
	/// let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
	/// assert_eq!(expected, view);
	/// ```
	#[inline]
	#[track_caller]
	pub fn from_row_major_slice_mut(slice: &'a mut [T], nrows: Rows, ncols: Cols) -> Self
	where
		T: Sized,
	{
		MatMut::from_column_major_slice_mut(slice, ncols, nrows).transpose_mut()
	}

	/// creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
	/// the data is interpreted in a row-major format, where the beginnings of two consecutive
	/// rows are separated by `row_stride` elements.
	#[inline]
	#[track_caller]
	pub fn from_row_major_slice_with_stride_mut(slice: &'a mut [T], nrows: Rows, ncols: Cols, row_stride: usize) -> Self
	where
		T: Sized,
	{
		from_strided_row_major_slice_mut_assert(nrows.unbound(), ncols.unbound(), row_stride, slice.len());

		unsafe { Self::from_raw_parts_mut(slice.as_mut_ptr(), nrows, ncols, 1, row_stride as isize) }
	}
}

impl<'ROWS, 'COLS, 'a, T, RStride: Stride, CStride: Stride> MatMut<'a, T, Dim<'ROWS>, Dim<'COLS>, RStride, CStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_with_mut<'TOP, 'BOT, 'LEFT, 'RIGHT>(
		self,
		row: Partition<'TOP, 'BOT, 'ROWS>,
		col: Partition<'LEFT, 'RIGHT, 'COLS>,
	) -> (
		MatMut<'a, T, Dim<'TOP>, Dim<'LEFT>, RStride, CStride>,
		MatMut<'a, T, Dim<'TOP>, Dim<'RIGHT>, RStride, CStride>,
		MatMut<'a, T, Dim<'BOT>, Dim<'LEFT>, RStride, CStride>,
		MatMut<'a, T, Dim<'BOT>, Dim<'RIGHT>, RStride, CStride>,
	) {
		let (a, b, c, d) = self.split_at_mut(row.midpoint(), col.midpoint());
		(
			a.as_shape_mut(row.head, col.head),
			b.as_shape_mut(row.head, col.tail),
			c.as_shape_mut(row.tail, col.head),
			d.as_shape_mut(row.tail, col.tail),
		)
	}
}

impl<'ROWS, 'a, T, Cols: Shape, RStride: Stride, CStride: Stride> MatMut<'a, T, Dim<'ROWS>, Cols, RStride, CStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_rows_with_mut<'TOP, 'BOT>(
		self,
		row: Partition<'TOP, 'BOT, 'ROWS>,
	) -> (
		MatMut<'a, T, Dim<'TOP>, Cols, RStride, CStride>,
		MatMut<'a, T, Dim<'BOT>, Cols, RStride, CStride>,
	) {
		let (a, b) = self.split_at_row_mut(row.midpoint());
		(a.as_row_shape_mut(row.head), b.as_row_shape_mut(row.tail))
	}
}

impl<'COLS, 'a, T, Rows: Shape, RStride: Stride, CStride: Stride> MatMut<'a, T, Rows, Dim<'COLS>, RStride, CStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_cols_with_mut<'LEFT, 'RIGHT>(
		self,
		col: Partition<'LEFT, 'RIGHT, 'COLS>,
	) -> (
		MatMut<'a, T, Rows, Dim<'LEFT>, RStride, CStride>,
		MatMut<'a, T, Rows, Dim<'RIGHT>, RStride, CStride>,
	) {
		let (a, b) = self.split_at_col_mut(col.midpoint());
		(a.as_col_shape_mut(col.head), b.as_col_shape_mut(col.tail))
	}
}

impl<'ROWS, 'COLS, 'a, T, RStride: Stride, CStride: Stride> MatMut<'a, T, Dim<'ROWS>, Dim<'COLS>, RStride, CStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_with<'TOP, 'BOT, 'LEFT, 'RIGHT>(
		self,
		row: Partition<'TOP, 'BOT, 'ROWS>,
		col: Partition<'LEFT, 'RIGHT, 'COLS>,
	) -> (
		MatRef<'a, T, Dim<'TOP>, Dim<'LEFT>, RStride, CStride>,
		MatRef<'a, T, Dim<'TOP>, Dim<'RIGHT>, RStride, CStride>,
		MatRef<'a, T, Dim<'BOT>, Dim<'LEFT>, RStride, CStride>,
		MatRef<'a, T, Dim<'BOT>, Dim<'RIGHT>, RStride, CStride>,
	) {
		let (a, b, c, d) = self.split_at(row.midpoint(), col.midpoint());
		(
			a.as_shape(row.head, col.head),
			b.as_shape(row.head, col.tail),
			c.as_shape(row.tail, col.head),
			d.as_shape(row.tail, col.tail),
		)
	}
}

impl<'ROWS, 'a, T, Cols: Shape, RStride: Stride, CStride: Stride> MatMut<'a, T, Dim<'ROWS>, Cols, RStride, CStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_rows_with<'TOP, 'BOT>(
		self,
		row: Partition<'TOP, 'BOT, 'ROWS>,
	) -> (
		MatRef<'a, T, Dim<'TOP>, Cols, RStride, CStride>,
		MatRef<'a, T, Dim<'BOT>, Cols, RStride, CStride>,
	) {
		let (a, b) = self.split_at_row(row.midpoint());
		(a.as_row_shape(row.head), b.as_row_shape(row.tail))
	}
}

impl<'COLS, 'a, T, Rows: Shape, RStride: Stride, CStride: Stride> MatMut<'a, T, Rows, Dim<'COLS>, RStride, CStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_cols_with<'LEFT, 'RIGHT>(
		self,
		col: Partition<'LEFT, 'RIGHT, 'COLS>,
	) -> (
		MatRef<'a, T, Rows, Dim<'LEFT>, RStride, CStride>,
		MatRef<'a, T, Rows, Dim<'RIGHT>, RStride, CStride>,
	) {
		let (a, b) = self.split_at_col(col.midpoint());
		(a.as_col_shape(col.head), b.as_col_shape(col.tail))
	}
}

impl<'a, T, Dim: Shape, RStride: Stride, CStride: Stride> MatMut<'a, T, Dim, Dim, RStride, CStride> {
	/// see [`MatRef::diagonal`]
	#[inline]
	pub fn diagonal(self) -> DiagRef<'a, T, Dim, isize> {
		self.into_const().diagonal()
	}
}

impl<'a, T, Dim: Shape, RStride: Stride, CStride: Stride> MatMut<'a, T, Dim, Dim, RStride, CStride> {
	/// see [`MatRef::diagonal`]
	#[inline]
	pub fn diagonal_mut(self) -> DiagMut<'a, T, Dim, isize> {
		unsafe { self.into_const().diagonal().column_vector().const_cast().as_diagonal_mut() }
	}
}

impl<T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> Index<(Idx<Rows>, Idx<Cols>)> for MatMut<'_, T, Rows, Cols, RStride, CStride> {
	type Output = T;

	#[inline]
	#[track_caller]
	fn index(&self, (row, col): (Idx<Rows>, Idx<Cols>)) -> &Self::Output {
		self.rb().at(row, col)
	}
}

impl<T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IndexMut<(Idx<Rows>, Idx<Cols>)> for MatMut<'_, T, Rows, Cols, RStride, CStride> {
	#[inline]
	#[track_caller]
	fn index_mut(&mut self, (row, col): (Idx<Rows>, Idx<Cols>)) -> &mut Self::Output {
		self.rb_mut().at_mut(row, col)
	}
}

impl<'a, T: core::fmt::Debug, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> core::fmt::Debug
	for MatMut<'a, T, Rows, Cols, RStride, CStride>
{
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.rb().fmt(f)
	}
}

impl<'a, T, Rows: Shape, Cols: Shape> MatMut<'a, T, Rows, Cols>
where
	T: RealField,
{
	/// see [MatRef::min]
	pub fn min(self) -> Option<T> {
		MatRef::internal_min(self.rb().as_dyn())
	}

	/// see [MatRef::min]
	pub fn max(self) -> Option<T> {
		MatRef::internal_max(self.rb().as_dyn())
	}
}

#[track_caller]
#[inline]
fn from_strided_column_major_slice_mut_assert(nrows: usize, ncols: usize, col_stride: usize, len: usize) {
	if nrows > 0 && ncols > 0 {
		// we don't have to worry about size == usize::MAX == slice.len(), because the length of a
		// slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
		// we don't care
		let last = usize::checked_mul(col_stride, ncols - 1).and_then(|last_col| last_col.checked_add(nrows - 1));
		let Some(last) = last else {
			panic!("address computation of the last matrix element overflowed");
		};
		assert!(all(col_stride >= nrows, last < len));
	}
}

#[track_caller]
#[inline]
fn from_strided_row_major_slice_mut_assert(nrows: usize, ncols: usize, row_stride: usize, len: usize) {
	if nrows > 0 && ncols > 0 {
		// we don't have to worry about size == usize::MAX == slice.len(), because the length of a
		// slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
		// we don't care
		let last = usize::checked_mul(row_stride, nrows - 1).and_then(|last_row| last_row.checked_add(ncols - 1));
		let Some(last) = last else {
			panic!("address computation of the last matrix element overflowed");
		};
		assert!(all(row_stride >= ncols, last < len));
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_min() {
		let mut m = mat![
			[1.0, 5.0, 3.0],
			[4.0, 2.0, 9.0],
			[7.0, 8.0, 6.0], //
		];

		assert_eq!(m.as_mut().min(), Some(1.0));

		let mut empty: Mat<f64> = Mat::new();
		assert_eq!(empty.as_mut().min(), None);
	}

	#[test]
	fn test_max() {
		let mut m = mat![
			[1.0, 5.0, 3.0],
			[4.0, 2.0, 9.0],
			[7.0, 8.0, 6.0], //
		];

		assert_eq!(m.as_mut().max(), Some(9.0));

		let mut empty: Mat<f64> = Mat::new();
		assert_eq!(empty.as_mut().max(), None);
	}
}
