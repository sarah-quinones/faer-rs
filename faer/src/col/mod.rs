use crate::internal_prelude::*;
use core::ptr::NonNull;
use faer_traits::Unit;

struct ColView<C: Container, T: ?Sized, Rows, RStride> {
    ptr: C::OfCopy<NonNull<T>>,
    nrows: Rows,
    row_stride: RStride,
}

impl<C: Container, T: ?Sized, Rows: Copy, RStride: Copy> Copy for ColView<C, T, Rows, RStride> {}
impl<C: Container, T: ?Sized, Rows: Copy, RStride: Copy> Clone for ColView<C, T, Rows, RStride> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

pub(crate) mod colmut;
pub(crate) mod colown;
pub(crate) mod colref;

pub use colmut::ColMut as ColMutGeneric;
pub use colown::Col as ColGeneric;
pub use colref::ColRef as ColRefGeneric;

pub type ColRef<'a, T, Rows = usize, RStride = isize> = ColRefGeneric<'a, Unit, T, Rows, RStride>;
pub type ColMut<'a, T, Rows = usize, RStride = isize> = ColMutGeneric<'a, Unit, T, Rows, RStride>;
pub type Col<T, Rows = usize> = ColGeneric<Unit, T, Rows>;

pub trait AsColMut<C: Container, T, Rows: Shape> {
    fn as_col_mut(&mut self) -> ColMutGeneric<C, T, Rows>;
}
pub trait AsColRef<C: Container, T, Rows: Shape> {
    fn as_col_ref(&self) -> ColRefGeneric<C, T, Rows>;
}

impl<C: Container, T, Rows: Shape, CStride: Stride> AsColRef<C, T, Rows>
    for ColRefGeneric<'_, C, T, Rows, CStride>
{
    #[inline]
    fn as_col_ref(&self) -> ColRefGeneric<C, T, Rows> {
        self.as_dyn_stride()
    }
}

impl<C: Container, T, Rows: Shape, CStride: Stride> AsColRef<C, T, Rows>
    for ColMutGeneric<'_, C, T, Rows, CStride>
{
    #[inline]
    fn as_col_ref(&self) -> ColRefGeneric<C, T, Rows> {
        self.rb().as_dyn_stride()
    }
}

impl<C: Container, T, Rows: Shape, CStride: Stride> AsColMut<C, T, Rows>
    for ColMutGeneric<'_, C, T, Rows, CStride>
{
    #[inline]
    fn as_col_mut(&mut self) -> ColMutGeneric<C, T, Rows> {
        self.rb_mut().as_dyn_stride_mut()
    }
}
