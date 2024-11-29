use crate::internal_prelude::*;
use core::ptr::NonNull;

pub trait ColIndex<RowRange> {
    type Target;

    fn get(this: Self, row: RowRange) -> Self::Target;
    unsafe fn get_unchecked(this: Self, row: RowRange) -> Self::Target;
}

struct ColView<T: ?Sized, Rows, RStride> {
    ptr: NonNull<T>,
    nrows: Rows,
    row_stride: RStride,
}

impl<T: ?Sized, Rows: Copy, RStride: Copy> Copy for ColView<T, Rows, RStride> {}
impl<T: ?Sized, Rows: Copy, RStride: Copy> Clone for ColView<T, Rows, RStride> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

mod col_index;

pub(crate) mod colmut;
pub(crate) mod colown;
pub(crate) mod colref;

pub use colmut::ColMut;
pub use colown::Col;
pub use colref::ColRef;

pub trait AsColMut: AsColRef {
    fn as_col_mut(&mut self) -> ColMut<Self::T, Self::Rows>;
}
pub trait AsColRef {
    type T;
    type Rows: Shape;

    fn as_col_ref(&self) -> ColRef<Self::T, Self::Rows>;
}

impl<T, Rows: Shape, RStride: Stride> AsColRef for ColRef<'_, T, Rows, RStride> {
    type T = T;
    type Rows = Rows;

    #[inline]
    fn as_col_ref(&self) -> ColRef<T, Rows> {
        self.as_dyn_stride()
    }
}

impl<T, Rows: Shape, RStride: Stride> AsColRef for ColMut<'_, T, Rows, RStride> {
    type T = T;
    type Rows = Rows;

    #[inline]
    fn as_col_ref(&self) -> ColRef<T, Rows> {
        self.rb().as_dyn_stride()
    }
}

impl<T, Rows: Shape, RStride: Stride> AsColMut for ColMut<'_, T, Rows, RStride> {
    #[inline]
    fn as_col_mut(&mut self) -> ColMut<T, Rows> {
        self.rb_mut().as_dyn_stride_mut()
    }
}

impl<T, Rows: Shape> AsColRef for Col<T, Rows> {
    type T = T;
    type Rows = Rows;

    #[inline]
    fn as_col_ref(&self) -> ColRef<T, Rows> {
        self.as_dyn_stride()
    }
}

impl<T, Rows: Shape> AsColMut for Col<T, Rows> {
    #[inline]
    fn as_col_mut(&mut self) -> ColMut<T, Rows> {
        self.as_dyn_stride_mut()
    }
}
