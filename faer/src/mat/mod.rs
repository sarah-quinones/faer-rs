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

pub use matmut::MatMut;
pub use matown::Mat;
pub use matref::MatRef;

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

	#[test]
	fn test_mat_resize_with() {
            let f_test_case = |nr, nc, fill| {
                let mut mat: Mat<f64> = Mat::zeros(nr, nc);
                // extend by one row and fill
                mat.resize_with(mat.nrows()+1, mat.ncols(), |_, _| fill);
                assert_eq!(mat.get(.., ..).sum(), fill*nc as f64);

                let mut mat: Mat<f64> = Mat::zeros(nr, nc);
                // extend by one column and fill
                mat.resize_with(mat.nrows(), mat.ncols()+1, |_, _| fill);
                assert_eq!(mat.get(.., ..).sum(), fill*nr as f64);
            };
            f_test_case(4, 3, 1.0);
            f_test_case(3, 4, 2.0);
            f_test_case(1, 1, 3.0);
            f_test_case(2, 2, 4.0);
        }
}
