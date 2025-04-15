use crate::{Shape, Stride, Unbind};
use core::marker::PhantomData;
use core::ptr::NonNull;
use faer_traits::{ComplexField, Conjugate};
use reborrow::*;

pub(crate) struct MatView<T: ?Sized, Rows, Cols, RStride, CStride> {
	ptr: NonNull<T>,
	nrows: Rows,
	ncols: Cols,
	row_stride: RStride,
	col_stride: CStride,
}

/// represents a type that can be used to slice a matrix, such as an index or a range of indices
pub trait MatIndex<RowRange, ColRange> {
	/// sliced view type
	type Target;

	/// slice `this` using `row` and `col`
	fn get(this: Self, row: RowRange, col: ColRange) -> Self::Target;

	/// slice `this` using `row` and `col` without bound checks
	unsafe fn get_unchecked(this: Self, row: RowRange, col: ColRange) -> Self::Target;
}

impl<T: ?Sized, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Copy for MatView<T, Rows, Cols, RStride, CStride> {}
impl<T: ?Sized, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Clone for MatView<T, Rows, Cols, RStride, CStride> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

#[inline]
#[track_caller]
fn from_slice_assert(nrows: usize, ncols: usize, len: usize) {
	let size = usize::checked_mul(nrows, ncols);
	assert!(size == Some(len));
}

mod mat_index;

pub(crate) mod matmut;
pub(crate) mod matown;
pub(crate) mod matref;

pub use matmut::Mut;
pub use matown::Own;
pub use matref::Ref;

/// heap allocated resizable matrix, similar to a 2d [`alloc::vec::Vec`]
///
/// # note
///
/// the memory layout of `Own` is guaranteed to be column-major, meaning that it has a row stride
/// of `1`, and an unspecified column stride that can be queried with [`Mat::col_stride`]
///
/// this implies that while each individual column is stored contiguously in memory, the matrix as
/// a whole may not necessarily be contiguous. the implementation may add padding at the end of
/// each column when overaligning each column can provide a performance gain
///
/// let us consider a 3×4 matrix
///
/// ```notcode
///  0 │ 3 │ 6 │  9
/// ───┼───┼───┼───
///  1 │ 4 │ 7 │ 10
/// ───┼───┼───┼───
///  2 │ 5 │ 8 │ 11
/// ```
/// the memory representation of the data held by such a matrix could look like the following:
///
/// ```notcode
/// [0, 1, 2, x, 3, 4, 5, x, 6, 7, 8, x, 9, 10, 11, x]
/// ```
///
/// where `x` represents padding elements
pub type Mat<T, Rows = usize, Cols = usize> = generic::Mat<Own<T, Rows, Cols>>;

/// immutable view over a matrix, similar to an immutable reference to a 2d strided [prim@slice]
///
/// # Note
///
/// unlike a slice, the data pointed to by `MatRef<'_, T>` is allowed to be partially or fully
/// uninitialized under certain conditions. in this case, care must be taken to not perform any
/// operations that read the uninitialized values, either directly or indirectly through any of the
/// numerical library routines, unless it is explicitly permitted
pub type MatRef<'a, T, Rows = usize, Cols = usize, RStride = isize, CStride = isize> = generic::Mat<Ref<'a, T, Rows, Cols, RStride, CStride>>;

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
pub type MatMut<'a, T, Rows = usize, Cols = usize, RStride = isize, CStride = isize> = generic::Mat<Mut<'a, T, Rows, Cols, RStride, CStride>>;

/// generic `Mat` wrapper
pub mod generic {
	use crate::{Idx, Shape, Stride};
	use core::fmt::Debug;
	use core::ops::{Index, IndexMut};
	use reborrow::*;

	/// generic `Mat` wrapper
	#[derive(Copy, Clone)]
	#[repr(transparent)]
	pub struct Mat<Inner>(pub Inner);

	impl<Inner: Debug> Debug for Mat<Inner> {
		#[inline(always)]
		fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
			self.0.fmt(f)
		}
	}

	impl<Inner> Mat<Inner> {
		/// wrap by reference
		#[inline(always)]
		pub fn from_inner_ref(inner: &Inner) -> &Self {
			unsafe { &*(inner as *const Inner as *const Self) }
		}

		/// wrap by mutable reference
		#[inline(always)]
		pub fn from_inner_mut(inner: &mut Inner) -> &mut Self {
			unsafe { &mut *(inner as *mut Inner as *mut Self) }
		}
	}

	impl<Inner> core::ops::Deref for Mat<Inner> {
		type Target = Inner;

		#[inline(always)]
		fn deref(&self) -> &Self::Target {
			&self.0
		}
	}

	impl<Inner> core::ops::DerefMut for Mat<Inner> {
		#[inline(always)]
		fn deref_mut(&mut self) -> &mut Self::Target {
			&mut self.0
		}
	}

	impl<'short, Inner: Reborrow<'short>> Reborrow<'short> for Mat<Inner> {
		type Target = Mat<Inner::Target>;

		#[inline(always)]
		fn rb(&'short self) -> Self::Target {
			Mat(self.0.rb())
		}
	}

	impl<'short, Inner: ReborrowMut<'short>> ReborrowMut<'short> for Mat<Inner> {
		type Target = Mat<Inner::Target>;

		#[inline(always)]
		fn rb_mut(&'short mut self) -> Self::Target {
			Mat(self.0.rb_mut())
		}
	}

	impl<Inner: IntoConst> IntoConst for Mat<Inner> {
		type Target = Mat<Inner::Target>;

		#[inline(always)]
		fn into_const(self) -> Self::Target {
			Mat(self.0.into_const())
		}
	}

	impl<
		T,
		Rows: Shape,
		Cols: Shape,
		RStride: Stride,
		CStride: Stride,
		Inner: for<'short> Reborrow<'short, Target = super::Ref<'short, T, Rows, Cols, RStride, CStride>>,
	> Index<(Idx<Rows>, Idx<Cols>)> for Mat<Inner>
	{
		type Output = T;

		#[inline]
		#[track_caller]
		fn index(&self, (row, col): (Idx<Rows>, Idx<Cols>)) -> &Self::Output {
			self.rb().at(row, col)
		}
	}

	impl<
		T,
		Rows: Shape,
		Cols: Shape,
		RStride: Stride,
		CStride: Stride,
		Inner: for<'short> Reborrow<'short, Target = super::Ref<'short, T, Rows, Cols, RStride, CStride>>
			+ for<'short> ReborrowMut<'short, Target = super::Mut<'short, T, Rows, Cols, RStride, CStride>>,
	> IndexMut<(Idx<Rows>, Idx<Cols>)> for Mat<Inner>
	{
		#[inline]
		#[track_caller]
		fn index_mut(&mut self, (row, col): (Idx<Rows>, Idx<Cols>)) -> &mut Self::Output {
			self.rb_mut().at_mut(row, col)
		}
	}
}

/// trait for types that can be converted to a matrix view
pub trait AsMatRef {
	/// scalar type
	type T;
	/// row dimension type
	type Rows: Shape;
	/// column dimension type
	type Cols: Shape;
	/// owned matrix type
	type Owned: AsMat<Self::T, T = Self::T, Rows = Self::Rows, Cols = Self::Cols, Owned = Self::Owned>;

	/// returns a view over `self`
	fn as_mat_ref(&self) -> MatRef<Self::T, Self::Rows, Self::Cols>;
}

/// trait for types that can be converted to a matrix view
pub trait AsMatMut: AsMatRef {
	/// returns a view over `self`
	fn as_mat_mut(&mut self) -> MatMut<Self::T, Self::Rows, Self::Cols>;
}

/// trait for owning matrix types
pub trait AsMat<T>: AsMatMut {
	/// returns a matrix with dimensions `(rows, cols)` filled with zeros
	fn zeros(rows: Self::Rows, cols: Self::Cols) -> Self
	where
		T: ComplexField;

	/// returns a matrix with dimensions `(rows, cols)` filled with zeros
	fn truncate(&mut self, rows: Self::Rows, cols: Self::Cols);
}

impl<M: AsMatRef> AsMatRef for &M {
	type Cols = M::Cols;
	type Owned = M::Owned;
	type Rows = M::Rows;
	type T = M::T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<Self::T, Self::Rows, Self::Cols> {
		(**self).as_mat_ref()
	}
}
impl<M: AsMatRef> AsMatRef for &mut M {
	type Cols = M::Cols;
	type Owned = M::Owned;
	type Rows = M::Rows;
	type T = M::T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<Self::T, Self::Rows, Self::Cols> {
		(**self).as_mat_ref()
	}
}
impl<M: AsMatMut> AsMatMut for &mut M {
	#[inline]
	fn as_mat_mut(&mut self) -> MatMut<Self::T, Self::Rows, Self::Cols> {
		(**self).as_mat_mut()
	}
}

impl<T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> AsMatRef for MatRef<'_, T, Rows, Cols, RStride, CStride> {
	type Cols = Cols;
	type Owned = Mat<T, Rows, Cols>;
	type Rows = Rows;
	type T = T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<T, Rows, Cols> {
		self.as_dyn_stride()
	}
}

impl<T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> AsMatRef for MatMut<'_, T, Rows, Cols, RStride, CStride> {
	type Cols = Cols;
	type Owned = Mat<T, Rows, Cols>;
	type Rows = Rows;
	type T = T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<T, Rows, Cols> {
		self.rb().as_dyn_stride()
	}
}

impl<T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> AsMatMut for MatMut<'_, T, Rows, Cols, RStride, CStride> {
	#[inline]
	fn as_mat_mut(&mut self) -> MatMut<T, Rows, Cols> {
		self.rb_mut().as_dyn_stride_mut()
	}
}

impl<T, Rows: Shape, Cols: Shape> AsMatRef for Mat<T, Rows, Cols> {
	type Cols = Cols;
	type Owned = Mat<T, Rows, Cols>;
	type Rows = Rows;
	type T = T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<T, Rows, Cols> {
		self.as_dyn_stride()
	}
}

impl<T, Rows: Shape, Cols: Shape> AsMat<T> for Mat<T, Rows, Cols> {
	#[inline]
	fn zeros(rows: Rows, cols: Cols) -> Self
	where
		T: ComplexField,
	{
		Mat::zeros(rows, cols)
	}

	#[track_caller]
	#[inline]
	fn truncate(&mut self, rows: Self::Rows, cols: Self::Cols) {
		self.truncate(rows, cols)
	}
}

impl<T, Rows: Shape, Cols: Shape> AsMatMut for Mat<T, Rows, Cols> {
	#[inline]
	fn as_mat_mut(&mut self) -> MatMut<T, Rows, Cols> {
		self.as_dyn_stride_mut()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::prelude::*;

	#[test]
	fn test_mat() {
		let _x = crate::mat![[0.0, 1.0]];
		let mat = Mat::from_fn(3, 4, |i, j| i as f64 + j as f64);

		let mat = mat.as_ref().cloned();
		let mat = mat.as_ref();

		for i in 0..3 {
			for j in 0..4 {
				zip!(&mat).map(|x| x).as_ref().at(i, j);
			}
		}
	}

	#[test]
	fn test_mat_complex() {
		let _x = mat![[c64::new(0.0, 0.0), c64::new(1.0, 0.0)]];
		let mat = Mat::from_fn(3, 4, |i, j| c64::new(i as f64 + j as f64, 0.0));
		{
			let _conj = mat.as_ref().conjugate();
		}

		let mat = mat.as_ref().cloned();
		let mat = mat.as_ref();

		for i in 0..3 {
			for j in 0..4 {
				zip!(&mat).map(|x| x).as_ref().at(i, j);
			}
		}
	}
}
