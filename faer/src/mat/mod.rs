use crate::{slice_len, Shape, Stride, Unbind};
use core::{marker::PhantomData, ptr::NonNull};
use faer_traits::{help, ComplexContainer, ConjUnit, Container, Unit};
use reborrow::*;

pub(crate) struct MatView<C: Container, T: ?Sized, Rows, Cols, RStride, CStride> {
    ptr: C::OfCopy<NonNull<T>>,
    nrows: Rows,
    ncols: Cols,
    row_stride: RStride,
    col_stride: CStride,
}

impl<C: Container, T: ?Sized, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Copy
    for MatView<C, T, Rows, Cols, RStride, CStride>
{
}
impl<C: Container, T: ?Sized, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Clone
    for MatView<C, T, Rows, Cols, RStride, CStride>
{
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

pub(crate) mod matmut;
pub(crate) mod matown;
pub(crate) mod matref;

pub use matmut::MatMut as MatMutGeneric;
pub use matown::Mat as MatGeneric;
pub use matref::MatRef as MatRefGeneric;

pub type MatRef<'a, T, Rows = usize, Cols = usize, RStride = isize, CStride = isize> =
    MatRefGeneric<'a, Unit, T, Rows, Cols, RStride, CStride>;
pub type MatMut<'a, T, Rows = usize, Cols = usize, RStride = isize, CStride = isize> =
    MatMutGeneric<'a, Unit, T, Rows, Cols, RStride, CStride>;
pub type Mat<T, Rows = usize, Cols = usize> = MatGeneric<Unit, T, Rows, Cols>;

pub trait AsMatMut<C: Container, T, Rows: Shape, Cols: Shape> {
    fn as_mat_mut(&mut self) -> MatMutGeneric<C, T, Rows, Cols>;
}
pub trait AsMatRef<C: Container, T, Rows: Shape, Cols: Shape> {
    fn as_mat_ref(&self) -> MatRefGeneric<C, T, Rows, Cols>;
}

impl<C: Container, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride>
    AsMatRef<C, T, Rows, Cols> for MatRefGeneric<'_, C, T, Rows, Cols, RStride, CStride>
{
    #[inline]
    fn as_mat_ref(&self) -> MatRefGeneric<C, T, Rows, Cols> {
        self.as_dyn_stride()
    }
}

impl<C: Container, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride>
    AsMatRef<C, T, Rows, Cols> for MatMutGeneric<'_, C, T, Rows, Cols, RStride, CStride>
{
    #[inline]
    fn as_mat_ref(&self) -> MatRefGeneric<C, T, Rows, Cols> {
        self.rb().as_dyn_stride()
    }
}

impl<C: Container, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride>
    AsMatMut<C, T, Rows, Cols> for MatMutGeneric<'_, C, T, Rows, Cols, RStride, CStride>
{
    #[inline]
    fn as_mat_mut(&mut self) -> MatMutGeneric<C, T, Rows, Cols> {
        self.rb_mut().as_dyn_stride_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zipped;
    use num_complex::Complex64;

    #[test]
    fn test_mat() {
        let _x = crate::mat![[0.0, 1.0]];
        let mat = Mat::from_fn(3, 4, |i, j| i as f64 + j as f64);

        let mat = mat.as_ref().cloned();
        let mat = mat.as_ref();

        for i in 0..3 {
            for j in 0..4 {
                dbg!(zipped!(&mat).map(|x| x).as_ref().at(i, j));
            }
        }
    }

    #[test]
    fn test_mat_complex() {
        let _x = crate::mat![[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]];
        let mat = Mat::from_fn(3, 4, |i, j| Complex64::new(i as f64 + j as f64, 0.0));
        {
            let _conj = mat.as_ref().conjugate();
        }

        let mat = mat.as_ref().cloned();
        let mat = mat.as_ref();

        for i in 0..3 {
            for j in 0..4 {
                dbg!(zipped!(&mat).map(|x| x).as_ref().at(i, j));
            }
        }
    }
}
