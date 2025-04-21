use super::{Mat, MatMut, MatRef, *};
use crate::col::ColRef;
use crate::internal_prelude::*;
use crate::row::RowRef;
use crate::utils::bound::{Dim, Partition};
use crate::{ContiguousFwd, Idx, IdxInc};
use equator::{assert, debug_assert};
use faer_traits::Real;
use generativity::Guard;

/// see [`super::MatRef`]
pub struct Ref<'a, T, Rows = usize, Cols = usize, RStride = isize, CStride = isize> {
	pub(super) imp: MatView<T, Rows, Cols, RStride, CStride>,
	pub(super) __marker: PhantomData<&'a T>,
}

impl<T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Copy for Ref<'_, T, Rows, Cols, RStride, CStride> {}
impl<T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Clone for Ref<'_, T, Rows, Cols, RStride, CStride> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

impl<'short, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Reborrow<'short> for Ref<'_, T, Rows, Cols, RStride, CStride> {
	type Target = Ref<'short, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}
impl<'short, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> ReborrowMut<'short> for Ref<'_, T, Rows, Cols, RStride, CStride> {
	type Target = Ref<'short, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}
impl<'a, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> IntoConst for Ref<'a, T, Rows, Cols, RStride, CStride> {
	type Target = Ref<'a, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}

unsafe impl<T: Sync, Rows: Sync, Cols: Sync, RStride: Sync, CStride: Sync> Sync for Ref<'_, T, Rows, Cols, RStride, CStride> {}
unsafe impl<T: Sync, Rows: Send, Cols: Send, RStride: Send, CStride: Send> Send for Ref<'_, T, Rows, Cols, RStride, CStride> {}

#[track_caller]
#[inline]
fn from_strided_column_major_slice_assert(nrows: usize, ncols: usize, col_stride: usize, len: usize) {
	if nrows > 0 && ncols > 0 {
		let last = usize::checked_mul(col_stride, ncols - 1).and_then(|last_col| last_col.checked_add(nrows - 1));
		let Some(last) = last else {
			panic!("address computation of the last matrix element overflowed");
		};
		assert!(last < len);
	}
}

impl<'a, T> MatRef<'a, T> {
	/// equivalent to `MatRef::from_row_major_slice(array.as_flattened(), ROWS, COLS)`
	#[inline]
	pub fn from_row_major_array<const ROWS: usize, const COLS: usize>(array: &'a [[T; COLS]; ROWS]) -> Self {
		unsafe { Self::from_raw_parts(array as *const _ as *const T, ROWS, COLS, COLS as isize, 1) }
	}

	/// equivalent to `MatRef::from_column_major_slice(array.as_flattened(), ROWS, COLS)`
	#[inline]
	pub fn from_column_major_array<const ROWS: usize, const COLS: usize>(array: &'a [[T; ROWS]; COLS]) -> Self {
		unsafe { Self::from_raw_parts(array as *const _ as *const T, ROWS, COLS, 1, ROWS as isize) }
	}

	/// creates a `1×1` view over the given element
	#[inline]
	pub fn from_ref(value: &'a T) -> Self {
		unsafe { MatRef::from_raw_parts(value as *const T, 1, 1, 0, 0) }
	}
}

impl<'a, T, Rows: Shape, Cols: Shape> MatRef<'a, T, Rows, Cols> {
	/// creates a `MatRef` from a view over a single element, repeated `nrows×ncols` times
	#[inline]
	pub fn from_repeated_ref(value: &'a T, nrows: Rows, ncols: Cols) -> Self {
		unsafe { MatRef::from_raw_parts(value as *const T, nrows, ncols, 0, 0) }
	}

	/// creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
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
	/// use faer::{MatRef, mat};
	///
	/// let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
	/// let view = MatRef::from_column_major_slice(&slice, 3, 2);
	///
	/// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
	/// assert_eq!(expected, view);
	/// ```
	#[inline]
	#[track_caller]
	pub fn from_column_major_slice(slice: &'a [T], nrows: Rows, ncols: Cols) -> Self {
		from_slice_assert(nrows.unbound(), ncols.unbound(), slice.len());

		unsafe { MatRef::from_raw_parts(slice.as_ptr(), nrows, ncols, 1, nrows.unbound() as isize) }
	}

	/// creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
	/// the data is interpreted in a column-major format, where the beginnings of two consecutive
	/// columns are separated by `col_stride` elements
	#[inline]
	#[track_caller]
	pub fn from_column_major_slice_with_stride(slice: &'a [T], nrows: Rows, ncols: Cols, col_stride: usize) -> Self {
		from_strided_column_major_slice_assert(nrows.unbound(), ncols.unbound(), col_stride, slice.len());

		unsafe { MatRef::from_raw_parts(slice.as_ptr(), nrows, ncols, 1, col_stride as isize) }
	}

	/// creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
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
	/// use faer::{MatRef, mat};
	///
	/// let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
	/// let view = MatRef::from_row_major_slice(&slice, 3, 2);
	///
	/// let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
	/// assert_eq!(expected, view);
	/// ```
	#[inline]
	#[track_caller]
	pub fn from_row_major_slice(slice: &'a [T], nrows: Rows, ncols: Cols) -> Self {
		MatRef::from_column_major_slice(slice, ncols, nrows).transpose()
	}

	/// creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
	/// the data is interpreted in a row-major format, where the beginnings of two consecutive
	/// rows are separated by `row_stride` elements
	#[inline]
	#[track_caller]
	pub fn from_row_major_slice_with_stride(slice: &'a [T], nrows: Rows, ncols: Cols, row_stride: usize) -> Self {
		MatRef::from_column_major_slice_with_stride(slice, ncols, nrows, row_stride).transpose()
	}
}

impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> MatRef<'a, T, Rows, Cols, RStride, CStride> {
	/// creates a `MatRef` from a pointer to the matrix data, dimensions, and strides
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
	/// * for each matrix unit, the corresponding pointer must be properly aligned,
	/// even for a zero-sized matrix
	/// * the values accessible by the matrix must be initialized at some point before they are
	/// read, or references to them are formed
	/// * no mutable aliasing is allowed. in other words, none of the elements accessible by any
	/// matrix unit may be accessed for writes by any other means for the duration of the lifetime
	/// `'a`
	///
	/// # example
	///
	/// ```
	/// use faer::{MatRef, mat};
	///
	/// // row major matrix with 2 rows, 3 columns, with a column at the end that we want to skip.
	/// // the row stride is the pointer offset from the address of 1.0 to the address of 4.0,
	/// // which is 4
	/// // the column stride is the pointer offset from the address of 1.0 to the address of 2.0,
	/// // which is 1
	/// let data = [[1.0, 2.0, 3.0, f64::NAN], [4.0, 5.0, 6.0, f64::NAN]];
	/// let matrix = unsafe { MatRef::from_raw_parts(data.as_ptr() as *const f64, 2, 3, 4, 1) };
	///
	/// let expected = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
	/// assert_eq!(expected.as_ref(), matrix);
	/// ```
	#[inline]
	#[track_caller]
	pub const unsafe fn from_raw_parts(ptr: *const T, nrows: Rows, ncols: Cols, row_stride: RStride, col_stride: CStride) -> Self {
		Self(Ref {
			imp: MatView {
				ptr: NonNull::new_unchecked(ptr as *mut T),
				nrows,
				ncols,
				row_stride,
				col_stride,
			},
			__marker: PhantomData,
		})
	}

	/// returns a pointer to the matrix data
	#[inline]
	pub fn as_ptr(&self) -> *const T {
		self.imp.ptr.as_ptr() as *const T
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
		let ptr = self.as_ptr();

		if row >= self.nrows() || col >= self.ncols() {
			ptr
		} else {
			ptr.wrapping_offset(row.unbound() as isize * self.row_stride().element_stride())
				.wrapping_offset(col.unbound() as isize * self.col_stride().element_stride())
		}
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
		debug_assert!(all(row < self.nrows(), col < self.ncols()));
		self.as_ptr()
			.offset(row.unbound() as isize * self.row_stride().element_stride())
			.offset(col.unbound() as isize * self.col_stride().element_stride())
	}

	/// splits the matrix horizontally and vertically at the given index into four corners and
	/// returns an array of each submatrix, in the following order:
	/// * top left
	/// * top right
	/// * bottom left
	/// * bottom right
	///
	/// # safety
	/// the function panics if any of the following conditions are violated:
	/// * `row <= self.nrows()`
	/// * `col <= self.ncols()`
	#[inline]
	#[track_caller]
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
		assert!(all(row <= self.nrows(), col <= self.ncols()));

		let rs = self.row_stride();
		let cs = self.col_stride();

		let top_left = self.ptr_at(Rows::start(), Cols::start());
		let top_right = self.ptr_at(Rows::start(), col);
		let bot_left = self.ptr_at(row, Cols::start());
		let bot_right = self.ptr_at(row, col);

		unsafe {
			(
				MatRef::from_raw_parts(top_left, row.unbound(), col.unbound(), rs, cs),
				MatRef::from_raw_parts(top_right, row.unbound(), self.ncols().unbound() - col.unbound(), rs, cs),
				MatRef::from_raw_parts(bot_left, self.nrows().unbound() - row.unbound(), col.unbound(), rs, cs),
				MatRef::from_raw_parts(
					bot_right,
					self.nrows().unbound() - row.unbound(),
					self.ncols().unbound() - col.unbound(),
					rs,
					cs,
				),
			)
		}
	}

	/// splits the matrix horizontally at the given row into two parts and returns an array of
	/// each submatrix, in the following order:
	/// * top
	/// * bottom
	///
	/// # panics
	/// the function panics if the following condition is violated:
	/// * `row <= self.nrows()`
	#[inline]
	#[track_caller]
	pub fn split_at_row(self, row: IdxInc<Rows>) -> (MatRef<'a, T, usize, Cols, RStride, CStride>, MatRef<'a, T, usize, Cols, RStride, CStride>) {
		assert!(all(row <= self.nrows()));

		let rs = self.row_stride();
		let cs = self.col_stride();

		let top = self.ptr_at(Rows::start(), Cols::start());
		let bot = self.ptr_at(row, Cols::start());

		unsafe {
			(
				MatRef::from_raw_parts(top, row.unbound(), self.ncols(), rs, cs),
				MatRef::from_raw_parts(bot, self.nrows().unbound() - row.unbound(), self.ncols(), rs, cs),
			)
		}
	}

	/// splits the matrix vertically at the given column into two parts and returns an array of
	/// each submatrix, in the following order:
	/// * left
	/// * right
	///
	/// # panics
	/// the function panics if the following condition is violated:
	/// * `col <= self.ncols()`
	#[inline]
	#[track_caller]
	pub fn split_at_col(self, col: IdxInc<Cols>) -> (MatRef<'a, T, Rows, usize, RStride, CStride>, MatRef<'a, T, Rows, usize, RStride, CStride>) {
		assert!(all(col <= self.ncols()));

		let rs = self.row_stride();
		let cs = self.col_stride();

		let left = self.ptr_at(Rows::start(), Cols::start());
		let right = self.ptr_at(Rows::start(), col);

		unsafe {
			(
				MatRef::from_raw_parts(left, self.nrows(), col.unbound(), rs, cs),
				MatRef::from_raw_parts(right, self.nrows(), self.ncols().unbound() - col.unbound(), rs, cs),
			)
		}
	}

	/// returns a view over the transpose of `self`
	///
	/// # example
	/// ```
	/// use faer::mat;
	///
	/// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
	/// let view = matrix.as_ref();
	/// let transpose = view.transpose();
	///
	/// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
	/// assert_eq!(expected.as_ref(), transpose);
	/// ```
	#[inline]
	pub fn transpose(self) -> MatRef<'a, T, Cols, Rows, CStride, RStride> {
		MatRef {
			0: Ref {
				imp: MatView {
					ptr: self.imp.ptr,
					nrows: self.imp.ncols,
					ncols: self.imp.nrows,
					row_stride: self.imp.col_stride,
					col_stride: self.imp.row_stride,
				},
				__marker: PhantomData,
			},
		}
	}

	/// returns a view over the conjugate of `self`
	#[inline]
	pub fn conjugate(self) -> MatRef<'a, T::Conj, Rows, Cols, RStride, CStride>
	where
		T: Conjugate,
	{
		unsafe {
			MatRef::from_raw_parts(
				self.as_ptr() as *const T::Conj,
				self.nrows(),
				self.ncols(),
				self.row_stride(),
				self.col_stride(),
			)
		}
	}

	/// returns an unconjugated view over `self`
	#[inline]
	pub fn canonical(self) -> MatRef<'a, T::Canonical, Rows, Cols, RStride, CStride>
	where
		T: Conjugate,
	{
		unsafe {
			MatRef::from_raw_parts(
				self.as_ptr() as *const T::Canonical,
				self.nrows(),
				self.ncols(),
				self.row_stride(),
				self.col_stride(),
			)
		}
	}

	#[inline]
	#[doc(hidden)]
	pub fn __canonicalize(self) -> (MatRef<'a, T::Canonical, Rows, Cols, RStride, CStride>, Conj)
	where
		T: Conjugate,
	{
		(self.canonical(), Conj::get::<T>())
	}

	/// returns a view over the conjugate transpose of `self`.
	#[inline]
	pub fn adjoint(self) -> MatRef<'a, T::Conj, Cols, Rows, CStride, RStride>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline]
	#[track_caller]
	pub(crate) fn at(self, row: Idx<Rows>, col: Idx<Cols>) -> &'a T {
		assert!(all(row < self.nrows(), col < self.ncols()));
		unsafe { self.at_unchecked(row, col) }
	}

	#[inline]
	#[track_caller]
	pub(crate) unsafe fn at_unchecked(self, row: Idx<Rows>, col: Idx<Cols>) -> &'a T {
		&*self.ptr_inbounds_at(row, col)
	}

	/// returns a view over the `self`, with the rows in reversed order
	///
	/// # example
	/// ```
	/// use faer::mat;
	///
	/// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
	/// let view = matrix.as_ref();
	/// let reversed_rows = view.reverse_rows();
	///
	/// let expected = mat![[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]];
	/// assert_eq!(expected.as_ref(), reversed_rows);
	/// ```
	#[inline]
	pub fn reverse_rows(self) -> MatRef<'a, T, Rows, Cols, RStride::Rev, CStride> {
		let row = unsafe { IdxInc::<Rows>::new_unbound(self.nrows().unbound().saturating_sub(1)) };
		let ptr = self.ptr_at(row, Cols::start());
		unsafe { MatRef::from_raw_parts(ptr, self.nrows(), self.ncols(), self.row_stride().rev(), self.col_stride()) }
	}

	/// returns a view over the `self`, with the columns in reversed order
	///
	/// # example
	/// ```
	/// use faer::mat;
	///
	/// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
	/// let view = matrix.as_ref();
	/// let reversed_cols = view.reverse_cols();
	///
	/// let expected = mat![[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]];
	/// assert_eq!(expected.as_ref(), reversed_cols);
	/// ```
	#[inline]
	pub fn reverse_cols(self) -> MatRef<'a, T, Rows, Cols, RStride, CStride::Rev> {
		let col = unsafe { IdxInc::<Cols>::new_unbound(self.ncols().unbound().saturating_sub(1)) };
		let ptr = self.ptr_at(Rows::start(), col);
		unsafe { MatRef::from_raw_parts(ptr, self.nrows(), self.ncols(), self.row_stride(), self.col_stride().rev()) }
	}

	/// returns a view over the `self`, with the rows and the columns in reversed order
	///
	/// # example
	/// ```
	/// use faer::mat;
	///
	/// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
	/// let view = matrix.as_ref();
	/// let reversed = view.reverse_rows_and_cols();
	///
	/// let expected = mat![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
	/// assert_eq!(expected.as_ref(), reversed);
	/// ```
	#[inline]
	pub fn reverse_rows_and_cols(self) -> MatRef<'a, T, Rows, Cols, RStride::Rev, CStride::Rev> {
		self.reverse_rows().reverse_cols()
	}

	/// returns a view over the submatrix starting at index `(row_start, col_start)`, and with
	/// dimensions `(nrows, ncols)`
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `row_start <= self.nrows()`
	/// * `col_start <= self.ncols()`
	/// * `nrows <= self.nrows() - row_start`
	/// * `ncols <= self.ncols() - col_start`
	///
	/// # example
	/// ```
	/// use faer::mat;
	///
	/// let matrix = mat![
	/// 	[1.0, 5.0, 9.0], //
	/// 	[2.0, 6.0, 10.0],
	/// 	[3.0, 7.0, 11.0],
	/// 	[4.0, 8.0, 12.0f64],
	/// ];
	///
	/// let view = matrix.as_ref();
	/// let submatrix = view.submatrix(
	/// 	/* row_start: */ 2, /* col_start: */ 1, /* nrows: */ 2, /* ncols: */ 2,
	/// );
	///
	/// let expected = mat![[7.0, 11.0], [8.0, 12.0f64]];
	/// assert_eq!(expected.as_ref(), submatrix);
	/// ```
	#[inline]
	#[track_caller]
	pub fn submatrix<V: Shape, H: Shape>(
		self,
		row_start: IdxInc<Rows>,
		col_start: IdxInc<Cols>,
		nrows: V,
		ncols: H,
	) -> MatRef<'a, T, V, H, RStride, CStride> {
		assert!(all(row_start <= self.nrows(), col_start <= self.ncols()));
		{
			let nrows = nrows.unbound();
			let full_nrows = self.nrows().unbound();
			let row_start = row_start.unbound();
			let ncols = ncols.unbound();
			let full_ncols = self.ncols().unbound();
			let col_start = col_start.unbound();
			assert!(all(nrows <= full_nrows - row_start, ncols <= full_ncols - col_start,));
		}
		let rs = self.row_stride();
		let cs = self.col_stride();

		unsafe { MatRef::from_raw_parts(self.ptr_at(row_start, col_start), nrows, ncols, rs, cs) }
	}

	/// returns a view over the submatrix starting at row `row_start`, and with number of rows
	/// `nrows`
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `row_start <= self.nrows()`
	/// * `nrows <= self.nrows() - row_start`
	///
	/// # example
	/// ```
	/// use faer::mat;
	///
	/// let matrix = mat![
	/// 	[1.0, 5.0, 9.0], //
	/// 	[2.0, 6.0, 10.0],
	/// 	[3.0, 7.0, 11.0],
	/// 	[4.0, 8.0, 12.0f64],
	/// ];
	///
	/// let view = matrix.as_ref();
	/// let subrows = view.subrows(/* row_start: */ 1, /* nrows: */ 2);
	///
	/// let expected = mat![[2.0, 6.0, 10.0], [3.0, 7.0, 11.0],];
	/// assert_eq!(expected.as_ref(), subrows);
	/// ```
	#[inline]
	#[track_caller]
	pub fn subrows<V: Shape>(self, row_start: IdxInc<Rows>, nrows: V) -> MatRef<'a, T, V, Cols, RStride, CStride> {
		assert!(all(row_start <= self.nrows()));
		{
			let nrows = nrows.unbound();
			let full_nrows = self.nrows().unbound();
			let row_start = row_start.unbound();
			assert!(all(nrows <= full_nrows - row_start));
		}
		let rs = self.row_stride();
		let cs = self.col_stride();

		unsafe { MatRef::from_raw_parts(self.ptr_at(row_start, Cols::start()), nrows, self.ncols(), rs, cs) }
	}

	/// returns a view over the submatrix starting at column `col_start`, and with number of
	/// columns `ncols`
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `col_start <= self.ncols()`
	/// * `ncols <= self.ncols() - col_start`
	///
	/// # example
	/// ```
	/// use faer::mat;
	///
	/// let matrix = mat![
	/// 	[1.0, 5.0, 9.0], //
	/// 	[2.0, 6.0, 10.0],
	/// 	[3.0, 7.0, 11.0],
	/// 	[4.0, 8.0, 12.0f64],
	/// ];
	///
	/// let view = matrix.as_ref();
	/// let subcols = view.subcols(/* col_start: */ 2, /* ncols: */ 1);
	///
	/// let expected = mat![[9.0], [10.0], [11.0], [12.0f64]];
	/// assert_eq!(expected.as_ref(), subcols);
	/// ```
	#[inline]
	#[track_caller]
	pub fn subcols<H: Shape>(self, col_start: IdxInc<Cols>, ncols: H) -> MatRef<'a, T, Rows, H, RStride, CStride> {
		assert!(all(col_start <= self.ncols()));
		{
			let ncols = ncols.unbound();
			let full_ncols = self.ncols().unbound();
			let col_start = col_start.unbound();
			assert!(all(ncols <= full_ncols - col_start));
		}
		let rs = self.row_stride();
		let cs = self.col_stride();

		unsafe { MatRef::from_raw_parts(self.ptr_at(Rows::start(), col_start), self.nrows(), ncols, rs, cs) }
	}

	/// returns the input matrix with the given shape after checking that it matches the
	/// current shape
	#[inline]
	#[track_caller]
	pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> MatRef<'a, T, V, H, RStride, CStride> {
		assert!(all(self.nrows().unbound() == nrows.unbound(), self.ncols().unbound() == ncols.unbound(),));
		unsafe { MatRef::from_raw_parts(self.as_ptr(), nrows, ncols, self.row_stride(), self.col_stride()) }
	}

	/// returns the input matrix with the given row shape after checking that it matches the
	/// current row shape
	#[inline]
	#[track_caller]
	pub fn as_row_shape<V: Shape>(self, nrows: V) -> MatRef<'a, T, V, Cols, RStride, CStride> {
		assert!(all(self.nrows().unbound() == nrows.unbound()));
		unsafe { MatRef::from_raw_parts(self.as_ptr(), nrows, self.ncols(), self.row_stride(), self.col_stride()) }
	}

	/// returns the input matrix with the given column shape after checking that it matches the
	/// current column shape
	#[inline]
	#[track_caller]
	pub fn as_col_shape<H: Shape>(self, ncols: H) -> MatRef<'a, T, Rows, H, RStride, CStride> {
		assert!(all(self.ncols().unbound() == ncols.unbound()));
		unsafe { MatRef::from_raw_parts(self.as_ptr(), self.nrows(), ncols, self.row_stride(), self.col_stride()) }
	}

	/// returns the input matrix with dynamic stride
	#[inline]
	pub fn as_dyn_stride(self) -> MatRef<'a, T, Rows, Cols, isize, isize> {
		unsafe {
			MatRef::from_raw_parts(
				self.as_ptr(),
				self.nrows(),
				self.ncols(),
				self.row_stride().element_stride(),
				self.col_stride().element_stride(),
			)
		}
	}

	/// returns the input matrix with dynamic shape
	#[inline]
	pub fn as_dyn(self) -> MatRef<'a, T, usize, usize, RStride, CStride> {
		unsafe {
			MatRef::from_raw_parts(
				self.as_ptr(),
				self.nrows().unbound(),
				self.ncols().unbound(),
				self.row_stride(),
				self.col_stride(),
			)
		}
	}

	/// returns the input matrix with dynamic row shape
	#[inline]
	pub fn as_dyn_rows(self) -> MatRef<'a, T, usize, Cols, RStride, CStride> {
		unsafe { MatRef::from_raw_parts(self.as_ptr(), self.nrows().unbound(), self.ncols(), self.row_stride(), self.col_stride()) }
	}

	/// returns the input matrix with dynamic column shape
	#[inline]
	pub fn as_dyn_cols(self) -> MatRef<'a, T, Rows, usize, RStride, CStride> {
		unsafe { MatRef::from_raw_parts(self.as_ptr(), self.nrows(), self.ncols().unbound(), self.row_stride(), self.col_stride()) }
	}

	/// returns a view over the row at the given index
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `row_idx < self.nrows()`
	#[inline]
	#[track_caller]
	pub fn row(self, i: Idx<Rows>) -> RowRef<'a, T, Cols, CStride> {
		assert!(i < self.nrows());

		unsafe { RowRef::from_raw_parts(self.ptr_at(i.into(), Cols::start()), self.ncols(), self.col_stride()) }
	}

	/// returns a view over the column at the given index
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `col_idx < self.ncols()`
	#[inline]
	#[track_caller]
	pub fn col(self, j: Idx<Cols>) -> ColRef<'a, T, Rows, RStride> {
		assert!(j < self.ncols());

		unsafe { ColRef::from_raw_parts(self.ptr_at(Rows::start(), j.into()), self.nrows(), self.row_stride()) }
	}

	/// returns an iterator over the columns of the matrix
	#[inline]
	pub fn col_iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = ColRef<'a, T, Rows, RStride>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		Cols::indices(Cols::start(), self.ncols().end()).map(move |j| self.col(j))
	}

	/// returns an iterator over the rows of the matrix
	#[inline]
	pub fn row_iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = RowRef<'a, T, Cols, CStride>>
	where
		Rows: 'a,
		Cols: 'a,
	{
		Rows::indices(Rows::start(), self.nrows().end()).map(move |i| self.row(i))
	}

	/// returns a parallel iterator over the columns of the matrix
	#[inline]
	#[cfg(feature = "rayon")]
	pub fn par_col_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColRef<'a, T, Rows, RStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;

		#[inline]
		fn col_fn<T, Rows: Shape, RStride: Stride, CStride: Stride>(
			col: MatRef<'_, T, Rows, usize, RStride, CStride>,
		) -> ColRef<'_, T, Rows, RStride> {
			col.col(0)
		}

		self.par_col_chunks(1).map(col_fn)
	}

	/// returns a parallel iterator over the rows of the matrix
	#[inline]
	#[cfg(feature = "rayon")]
	pub fn par_row_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowRef<'a, T, Cols, CStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;
		self.transpose().par_col_iter().map(ColRef::transpose)
	}

	/// returns a parallel iterator that provides successive chunks of the columns of this
	/// matrix, with each having at most `chunk_size` columns
	///
	/// if the number of columns is a multiple of `chunk_size`, then all chunks have
	/// `chunk_size` columns
	///
	/// only available with the `rayon` feature
	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	pub fn par_col_chunks(
		self,
		chunk_size: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, T, Rows, usize, RStride, CStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;

		let this = self.as_dyn_cols();

		assert!(chunk_size > 0);
		let chunk_count = this.ncols().msrv_div_ceil(chunk_size);
		(0..chunk_count).into_par_iter().map(move |chunk_idx| {
			let pos = chunk_size * chunk_idx;
			this.subcols(pos, Ord::min(chunk_size, this.ncols() - pos))
		})
	}

	/// returns a parallel iterator that provides exactly `count` successive chunks of the columns
	/// of this matrix
	///
	/// only available with the `rayon` feature
	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	pub fn par_col_partition(
		self,
		count: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, T, Rows, usize, RStride, CStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;

		let this = self.as_dyn_cols();

		assert!(count > 0);
		(0..count).into_par_iter().map(move |chunk_idx| {
			let (start, len) = crate::utils::thread::par_split_indices(this.ncols(), chunk_idx, count);
			this.subcols(start, len)
		})
	}

	/// returns a parallel iterator that provides successive chunks of the rows of this matrix,
	/// with each having at most `chunk_size` rows
	///
	/// if the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
	/// rows
	///
	/// only available with the `rayon` feature
	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	pub fn par_row_chunks(
		self,
		chunk_size: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, T, usize, Cols, RStride, CStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;
		self.transpose().par_col_chunks(chunk_size).map(MatRef::transpose)
	}

	/// returns a parallel iterator that provides exactly `count` successive chunks of the rows
	/// of this matrix
	///
	/// only available with the `rayon` feature
	#[inline]
	#[track_caller]
	#[cfg(feature = "rayon")]
	pub fn par_row_partition(
		self,
		count: usize,
	) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, T, usize, Cols, RStride, CStride>>
	where
		T: Sync,
		Rows: 'a,
		Cols: 'a,
	{
		use rayon::prelude::*;
		self.transpose().par_col_partition(count).map(MatRef::transpose)
	}

	/// returns a reference to the first row and a view over the remaining ones if the matrix has
	/// at least one row, otherwise `None`
	#[inline]
	pub fn split_first_row(self) -> Option<(RowRef<'a, T, Cols, CStride>, MatRef<'a, T, usize, Cols, RStride, CStride>)> {
		if let Some(i0) = self.nrows().idx_inc(1) {
			let (head, tail) = self.split_at_row(i0);
			Some((head.row(0), tail))
		} else {
			None
		}
	}

	/// returns a reference to the first column and a view over the remaining ones if the matrix has
	/// at least one column, otherwise `None`
	#[inline]
	pub fn split_first_col(self) -> Option<(ColRef<'a, T, Rows, RStride>, MatRef<'a, T, Rows, usize, RStride, CStride>)> {
		if let Some(i0) = self.ncols().idx_inc(1) {
			let (head, tail) = self.split_at_col(i0);
			Some((head.col(0), tail))
		} else {
			None
		}
	}

	/// returns a reference to the last row and a view over the remaining ones if the matrix has
	/// at least one row, otherwise `None`
	#[inline]
	pub fn split_last_row(self) -> Option<(RowRef<'a, T, Cols, CStride>, MatRef<'a, T, usize, Cols, RStride, CStride>)> {
		if self.nrows().unbound() > 0 {
			let i0 = self.nrows().checked_idx_inc(self.nrows().unbound() - 1);
			let (head, tail) = self.split_at_row(i0);
			Some((tail.row(0), head))
		} else {
			None
		}
	}

	/// returns a reference to the last column and a view over the remaining ones if the matrix has
	/// at least one column, otherwise `None`
	#[inline]
	pub fn split_last_col(self) -> Option<(ColRef<'a, T, Rows, RStride>, MatRef<'a, T, Rows, usize, RStride, CStride>)> {
		if self.ncols().unbound() > 0 {
			let i0 = self.ncols().checked_idx_inc(self.ncols().unbound() - 1);
			let (head, tail) = self.split_at_col(i0);
			Some((tail.col(0), head))
		} else {
			None
		}
	}

	/// returns a view over the matrix with a static column stride equal to `+1`, or `None`
	/// otherwise
	#[inline]
	pub fn try_as_row_major(self) -> Option<MatRef<'a, T, Rows, Cols, RStride, ContiguousFwd>> {
		if self.col_stride().element_stride() == 1 {
			Some(unsafe { MatRef::from_raw_parts(self.as_ptr(), self.nrows(), self.ncols(), self.row_stride(), ContiguousFwd) })
		} else {
			None
		}
	}

	#[doc(hidden)]
	#[inline]
	pub fn bind<'M, 'N>(self, row: Guard<'M>, col: Guard<'N>) -> MatRef<'a, T, Dim<'M>, Dim<'N>, RStride, CStride> {
		unsafe {
			MatRef::from_raw_parts(
				self.as_ptr(),
				self.nrows().bind(row),
				self.ncols().bind(col),
				self.row_stride(),
				self.col_stride(),
			)
		}
	}

	#[doc(hidden)]
	#[inline]
	pub fn bind_r<'M>(self, row: Guard<'M>) -> MatRef<'a, T, Dim<'M>, Cols, RStride, CStride> {
		unsafe { MatRef::from_raw_parts(self.as_ptr(), self.nrows().bind(row), self.ncols(), self.row_stride(), self.col_stride()) }
	}

	#[doc(hidden)]
	#[inline]
	pub fn bind_c<'N>(self, col: Guard<'N>) -> MatRef<'a, T, Rows, Dim<'N>, RStride, CStride> {
		unsafe { MatRef::from_raw_parts(self.as_ptr(), self.nrows(), self.ncols().bind(col), self.row_stride(), self.col_stride()) }
	}

	#[doc(hidden)]
	#[inline]
	pub unsafe fn const_cast(self) -> MatMut<'a, T, Rows, Cols, RStride, CStride> {
		MatMut::from_raw_parts_mut(self.as_ptr() as *mut T, self.nrows(), self.ncols(), self.row_stride(), self.col_stride())
	}

	/// returns a view over the matrix with a static row stride equal to `+1`, or `None` otherwise
	#[inline]
	pub fn try_as_col_major(self) -> Option<MatRef<'a, T, Rows, Cols, ContiguousFwd, CStride>> {
		if self.row_stride().element_stride() == 1 {
			Some(unsafe { MatRef::from_raw_parts(self.as_ptr(), self.nrows(), self.ncols(), ContiguousFwd, self.col_stride()) })
		} else {
			None
		}
	}

	/// returns references to the element at the given index, or submatrices if either `row`
	/// or `col` is a range, with bound checks
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `row` must be contained in `[0, self.nrows())`
	/// * `col` must be contained in `[0, self.ncols())`
	#[track_caller]
	#[inline]
	pub fn get<RowRange, ColRange>(
		self,
		row: RowRange,
		col: ColRange,
	) -> <MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::Target
	where
		MatRef<'a, T, Rows, Cols, RStride, CStride>: MatIndex<RowRange, ColRange>,
	{
		<MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::get(self, row, col)
	}

	/// equivalent to `self.get(row, ..)`
	#[track_caller]
	#[inline]
	pub fn get_r<RowRange>(self, row: RowRange) -> <MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, core::ops::RangeFull>>::Target
	where
		MatRef<'a, T, Rows, Cols, RStride, CStride>: MatIndex<RowRange, core::ops::RangeFull>,
	{
		<MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, core::ops::RangeFull>>::get(self, row, ..)
	}

	/// equivalent to `self.get(.., col)`
	#[track_caller]
	#[inline]
	pub fn get_c<ColRange>(self, col: ColRange) -> <MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<core::ops::RangeFull, ColRange>>::Target
	where
		MatRef<'a, T, Rows, Cols, RStride, CStride>: MatIndex<core::ops::RangeFull, ColRange>,
	{
		<MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<core::ops::RangeFull, ColRange>>::get(self, .., col)
	}

	/// returns references to the element at the given index, or submatrices if either `row`
	/// or `col` is a range, without bound checks
	///
	/// # safety
	/// the behavior is undefined if any of the following conditions are violated:
	/// * `row` must be contained in `[0, self.nrows())`
	/// * `col` must be contained in `[0, self.ncols())`
	#[track_caller]
	#[inline]
	pub unsafe fn get_unchecked<RowRange, ColRange>(
		self,
		row: RowRange,
		col: ColRange,
	) -> <MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::Target
	where
		MatRef<'a, T, Rows, Cols, RStride, CStride>: MatIndex<RowRange, ColRange>,
	{
		unsafe { <MatRef<'a, T, Rows, Cols, RStride, CStride> as MatIndex<RowRange, ColRange>>::get_unchecked(self, row, col) }
	}

	#[inline]
	pub(crate) fn __at(self, (i, j): (Idx<Rows>, Idx<Cols>)) -> &'a T {
		self.at(i, j)
	}
}

impl<
	T,
	Rows: Shape,
	Cols: Shape,
	RStride: Stride,
	CStride: Stride,
	Inner: for<'short> Reborrow<'short, Target = Ref<'short, T, Rows, Cols, RStride, CStride>>,
> generic::Mat<Inner>
{
	/// returns a view over `self`
	#[inline]
	pub fn as_ref(&self) -> MatRef<'_, T, Rows, Cols, RStride, CStride> {
		self.rb()
	}

	/// returns a newly allocated matrix holding the cloned values of `self`
	#[inline]
	pub fn cloned(&self) -> Mat<T, Rows, Cols>
	where
		T: Clone,
	{
		fn imp<'M, 'N, T: Clone, RStride: Stride, CStride: Stride>(
			this: MatRef<'_, T, Dim<'M>, Dim<'N>, RStride, CStride>,
		) -> Mat<T, Dim<'M>, Dim<'N>> {
			Mat::from_fn(this.nrows(), this.ncols(), |i, j| this.at(i, j).clone())
		}

		let this = self.rb();

		with_dim!(M, this.nrows().unbound());
		with_dim!(N, this.ncols().unbound());
		imp(this.as_shape(M, N)).into_shape(this.nrows(), this.ncols())
	}

	/// returns a newly allocated matrix holding the (possibly conjugated) values of `self`
	#[inline]
	pub fn to_owned(&self) -> Mat<T::Canonical, Rows, Cols>
	where
		T: Conjugate,
	{
		fn imp<'M, 'N, T, RStride: Stride, CStride: Stride>(
			this: MatRef<'_, T, Dim<'M>, Dim<'N>, RStride, CStride>,
		) -> Mat<T::Canonical, Dim<'M>, Dim<'N>>
		where
			T: Conjugate,
		{
			Mat::from_fn(this.nrows(), this.ncols(), |i, j| Conj::apply::<T>(this.at(i, j)))
		}

		let this = self.rb();
		with_dim!(M, this.nrows().unbound());
		with_dim!(N, this.ncols().unbound());
		imp(this.as_shape(M, N)).into_shape(this.nrows(), this.ncols())
	}

	/// returns the maximum norm of `self`
	#[inline]
	pub fn norm_max(&self) -> Real<T>
	where
		T: Conjugate,
	{
		linalg::reductions::norm_max::norm_max(self.rb().canonical().as_dyn_stride().as_dyn())
	}

	/// returns the l2 norm of `self`
	#[inline]
	pub fn norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		linalg::reductions::norm_l2::norm_l2(self.rb().canonical().as_dyn_stride().as_dyn())
	}

	/// returns the squared l2 norm of `self`
	#[inline]
	pub fn squared_norm_l2(&self) -> Real<T>
	where
		T: Conjugate,
	{
		linalg::reductions::norm_l2_sqr::norm_l2_sqr(self.rb().canonical().as_dyn_stride().as_dyn())
	}

	/// returns the l1 norm of `self`
	#[inline]
	pub fn norm_l1(&self) -> Real<T>
	where
		T: Conjugate,
	{
		linalg::reductions::norm_l1::norm_l1(self.rb().canonical().as_dyn_stride().as_dyn())
	}

	/// returns the sum of the elements of `self`
	#[inline]
	#[math]
	pub fn sum(&self) -> T::Canonical
	where
		T: Conjugate,
	{
		let val = linalg::reductions::sum::sum(self.rb().canonical().as_dyn_stride().as_dyn());
		if try_const! { Conj::get::<T>().is_conj() } { conj(val) } else { val }
	}

	/// returns the determinant of `self`
	#[inline]
	#[math]
	pub fn determinant(&self) -> T::Canonical
	where
		T: Conjugate,
	{
		let det = linalg::reductions::determinant::determinant(self.rb().canonical().as_dyn_stride().as_dyn());
		if const { T::IS_CANONICAL } { det } else { conj(det) }
	}

	/// kronecker product of two matrices
	///
	/// the kronecker product of two matrices $A$ and $B$ is a block matrix
	/// $B$ with the following structure:
	///
	/// ```text
	/// C = [ a[(0, 0)] * B    , a[(0, 1)] * B    , ... , a[(0, n-1)] * B    ]
	///     [ a[(1, 0)] * B    , a[(1, 1)] * B    , ... , a[(1, n-1)] * B    ]
	///     [ ...              , ...              , ... , ...              ]
	///     [ a[(m-1, 0)] * B  , a[(m-1, 1)] * B  , ... , a[(m-1, n-1)] * B  ]
	/// ```
	///
	/// # panics
	///
	/// panics if `dst` does not have the correct dimensions. the dimensions
	/// of `dst` must be `A.nrows() * B.nrows()` by `A.ncols() * B.ncols()`.
	///
	/// # example
	///
	/// ```
	/// use faer::linalg::kron::kron;
	/// use faer::{Mat, mat};
	///
	/// let a = mat![[1.0, 2.0], [3.0, 4.0]];
	/// let b = mat![[0.0, 5.0], [6.0, 7.0]];
	/// let c = mat![
	/// 	[0.0, 5.0, 0.0, 10.0],
	/// 	[6.0, 7.0, 12.0, 14.0],
	/// 	[0.0, 15.0, 0.0, 20.0],
	/// 	[18.0, 21.0, 24.0, 28.0],
	/// ];
	/// let mut dst = Mat::zeros(4, 4);
	/// kron(dst.as_mut(), a.as_ref(), b.as_ref());
	/// assert_eq!(dst, c);
	/// ```
	#[inline]
	pub fn kron(&self, rhs: impl AsMatRef<T: Conjugate<Canonical = T::Canonical>>) -> Mat<T::Canonical>
	where
		T: Conjugate,
	{
		fn imp<T: ComplexField>(lhs: MatRef<'_, impl Conjugate<Canonical = T>>, rhs: MatRef<'_, impl Conjugate<Canonical = T>>) -> Mat<T> {
			let mut out = Mat::zeros(lhs.nrows() * rhs.nrows(), lhs.ncols() * rhs.ncols());
			linalg::kron::kron(out.rb_mut(), lhs, rhs);
			out
		}

		imp(self.rb().as_dyn().as_dyn_stride(), rhs.as_mat_ref().as_dyn().as_dyn_stride())
	}

	/// returns `true` if all of the elements of `self` are finite.
	/// otherwise returns `false`.
	#[inline]
	pub fn is_all_finite(&self) -> bool
	where
		T: Conjugate,
	{
		fn imp<T: ComplexField>(A: MatRef<'_, T>) -> bool {
			with_dim!({
				let M = A.nrows();
				let N = A.ncols();
			});

			let A = A.as_shape(M, N);

			for j in N.indices() {
				for i in M.indices() {
					if !is_finite(&A[(i, j)]) {
						return false;
					}
				}
			}

			true
		}

		imp(self.rb().as_dyn().as_dyn_stride().canonical())
	}

	/// returns `true` if any of the elements of `self` is `NaN`.
	/// otherwise returns `false`.
	#[inline]
	pub fn has_nan(&self) -> bool
	where
		T: Conjugate,
	{
		fn imp<T: ComplexField>(A: MatRef<'_, T>) -> bool {
			with_dim!({
				let M = A.nrows();
				let N = A.ncols();
			});

			let A = A.as_shape(M, N);

			for j in N.indices() {
				for i in M.indices() {
					if is_nan(&A[(i, j)]) {
						return true;
					}
				}
			}

			false
		}

		imp(self.rb().as_dyn().as_dyn_stride().canonical())
	}
}

impl<'a, T, Dim: Shape, RStride: Stride, CStride: Stride> MatRef<'a, T, Dim, Dim, RStride, CStride> {
	/// returns the diagonal of the matrix
	#[inline]
	pub fn diagonal(self) -> DiagRef<'a, T, Dim, isize> {
		let k = Ord::min(self.nrows(), self.ncols());
		DiagRef {
			0: crate::diag::Ref {
				inner: unsafe { ColRef::from_raw_parts(self.as_ptr(), k, self.row_stride().element_stride() + self.col_stride().element_stride()) },
			},
		}
	}
}

impl<'ROWS, 'COLS, 'a, T, RStride: Stride, CStride: Stride> MatRef<'a, T, Dim<'ROWS>, Dim<'COLS>, RStride, CStride> {
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

impl<'ROWS, 'a, T, Cols: Shape, RStride: Stride, CStride: Stride> MatRef<'a, T, Dim<'ROWS>, Cols, RStride, CStride> {
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

impl<'COLS, 'a, T, Rows: Shape, RStride: Stride, CStride: Stride> MatRef<'a, T, Rows, Dim<'COLS>, RStride, CStride> {
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

impl<'a, T: core::fmt::Debug, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> core::fmt::Debug
	for Ref<'a, T, Rows, Cols, RStride, CStride>
{
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		fn imp<'M, 'N, T: core::fmt::Debug>(this: MatRef<'_, T, Dim<'M>, Dim<'N>>, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
			writeln!(f, "[")?;
			for i in this.nrows().indices() {
				this.row(i).fmt(f)?;
				f.write_str(",\n")?;
			}
			write!(f, "]")
		}

		let this = generic::Mat::from_inner_ref(self);

		with_dim!(M, this.nrows().unbound());
		with_dim!(N, this.ncols().unbound());
		imp(this.as_shape(M, N).as_dyn_stride(), f)
	}
}

impl<'a, T> MatRef<'a, T, usize, usize>
where
	T: RealField,
{
	pub(crate) fn internal_max(self) -> Option<T> {
		if self.nrows().unbound() == 0 || self.ncols().unbound() == 0 {
			return None;
		}

		let mut max_val = self.get(0, 0);

		let this = if self.row_stride().unsigned_abs() == 1 { self.transpose() } else { self };

		let this = if this.col_stride() > 0 { this } else { this.reverse_cols() };

		this.row_iter().for_each(|row| {
			row.iter().for_each(|val| {
				if val > max_val {
					max_val = &val;
				}
			});
		});

		Some((*max_val).clone())
	}

	pub(crate) fn internal_min(self) -> Option<T> {
		if self.nrows().unbound() == 0 || self.ncols().unbound() == 0 {
			return None;
		}

		let mut min_val = self.get(0, 0);

		let this = if self.row_stride().unsigned_abs() == 1 { self.transpose() } else { self };

		let this = if this.col_stride() > 0 { this } else { this.reverse_cols() };

		this.row_iter().for_each(|row| {
			row.iter().for_each(|val| {
				if val < min_val {
					min_val = &val;
				}
			});
		});

		Some((*min_val).clone())
	}
}

impl<'a, T, Rows: Shape, Cols: Shape> MatRef<'a, T, Rows, Cols>
where
	T: RealField,
{
	/// Returns the maximum element in the matrix
	///
	/// # Returns
	///
	/// * `Option<T>` - The maximum element in the matrix, or `None` if the matrix is empty
	///
	/// # Examples
	///
	/// ```
	/// use faer::{Mat, mat};
	///
	/// let m = mat![[1.0, 5.0, 3.0], [4.0, 2.0, 9.0], [7.0, 8.0, 6.0],];
	///
	/// assert_eq!(m.max(), Some(9.0));
	///
	/// let empty: Mat<f64> = Mat::new();
	/// assert_eq!(empty.max(), None);
	/// ```
	pub fn max(self) -> Option<T> {
		MatRef::internal_max(self.as_dyn())
	}

	/// Returns the minimum element in the matrix
	///
	/// # Returns
	///
	/// * `Option<T>` - The minimum element in the matrix, or `None` if the matrix is empty
	///
	/// # Examples
	///
	/// ```
	/// use faer::{Mat, mat};
	///
	/// let m = mat![[1.0, 5.0, 3.0], [4.0, 2.0, 9.0], [7.0, 8.0, 6.0],];
	///
	/// assert_eq!(m.min(), Some(1.0));
	///
	/// let empty: Mat<f64> = Mat::new();
	/// assert_eq!(empty.min(), None);
	/// ```
	pub fn min(self) -> Option<T> {
		MatRef::internal_min(self.as_dyn())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_min() {
		let m = mat![
			[1.0, 5.0, 3.0],
			[4.0, 2.0, 9.0],
			[7.0, 8.0, 6.0], //
		];

		assert_eq!(m.as_ref().min(), Some(1.0));

		let empty: Mat<f64> = Mat::new();
		assert_eq!(empty.as_ref().min(), None);
	}

	#[test]
	fn test_max() {
		let m = mat![
			[1.0, 5.0, 3.0],
			[4.0, 2.0, 9.0],
			[7.0, 8.0, 6.0], //
		];

		assert_eq!(m.as_ref().max(), Some(9.0));

		let empty: Mat<f64> = Mat::new();
		assert_eq!(empty.as_ref().max(), None);
	}
}
