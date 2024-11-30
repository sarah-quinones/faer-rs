use crate::{internal_prelude::*, Shape};

pub trait RowIndex<ColRange> {
    type Target;

    fn get(this: Self, col: ColRange) -> Self::Target;
    unsafe fn get_unchecked(this: Self, col: ColRange) -> Self::Target;
}

mod row_index;

pub(crate) mod rowmut;
pub(crate) mod rowown;
pub(crate) mod rowref;

pub use rowmut::RowMut;
pub use rowown::Row;
pub use rowref::RowRef;

pub trait AsRowMut: AsRowRef {
    fn as_row_mut(&mut self) -> RowMut<Self::T, Self::Cols>;
}
pub trait AsRowRef {
    type T;
    type Cols: Shape;

    fn as_row_ref(&self) -> RowRef<Self::T, Self::Cols>;
}

impl<T, Cols: Shape, CStride: Stride> AsRowRef for RowRef<'_, T, Cols, CStride> {
    type T = T;
    type Cols = Cols;

    #[inline]
    fn as_row_ref(&self) -> RowRef<T, Cols> {
        self.as_dyn_stride()
    }
}

impl<T, Cols: Shape, CStride: Stride> AsRowRef for RowMut<'_, T, Cols, CStride> {
    type T = T;
    type Cols = Cols;

    #[inline]
    fn as_row_ref(&self) -> RowRef<T, Cols> {
        self.rb().as_dyn_stride()
    }
}

impl<T, Cols: Shape, CStride: Stride> AsRowMut for RowMut<'_, T, Cols, CStride> {
    #[inline]
    fn as_row_mut(&mut self) -> RowMut<T, Cols> {
        self.rb_mut().as_dyn_stride_mut()
    }
}

impl<T, Cols: Shape> AsRowRef for Row<T, Cols> {
    type T = T;
    type Cols = Cols;

    #[inline]
    fn as_row_ref(&self) -> RowRef<T, Cols> {
        self.as_dyn_stride()
    }
}

impl<T, Cols: Shape> AsRowMut for Row<T, Cols> {
    #[inline]
    fn as_row_mut(&mut self) -> RowMut<T, Cols> {
        self.as_dyn_stride_mut()
    }
}
impl<M: AsRowRef> AsRowRef for &M {
    type T = M::T;
    type Cols = M::Cols;

    #[inline]
    fn as_row_ref(&self) -> RowRef<Self::T, Self::Cols> {
        (**self).as_row_ref()
    }
}
impl<M: AsRowRef> AsRowRef for &mut M {
    type T = M::T;
    type Cols = M::Cols;

    #[inline]
    fn as_row_ref(&self) -> RowRef<Self::T, Self::Cols> {
        (**self).as_row_ref()
    }
}
impl<M: AsRowMut> AsRowMut for &mut M {
    #[inline]
    fn as_row_mut(&mut self) -> RowMut<Self::T, Self::Cols> {
        (**self).as_row_mut()
    }
}
