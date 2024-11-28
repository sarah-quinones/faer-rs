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

pub trait AsRowMut<T, Cols: Shape> {
    fn as_row_mut(&mut self) -> RowMut<T, Cols>;
}
pub trait AsRowRef<T, Cols: Shape> {
    fn as_row_ref(&self) -> RowRef<T, Cols>;
}

impl<T, Cols: Shape, CStride: Stride> AsRowRef<T, Cols> for RowRef<'_, T, Cols, CStride> {
    #[inline]
    fn as_row_ref(&self) -> RowRef<T, Cols> {
        self.as_dyn_stride()
    }
}

impl<T, Cols: Shape, CStride: Stride> AsRowRef<T, Cols> for RowMut<'_, T, Cols, CStride> {
    #[inline]
    fn as_row_ref(&self) -> RowRef<T, Cols> {
        self.rb().as_dyn_stride()
    }
}

impl<T, Cols: Shape, CStride: Stride> AsRowMut<T, Cols> for RowMut<'_, T, Cols, CStride> {
    #[inline]
    fn as_row_mut(&mut self) -> RowMut<T, Cols> {
        self.rb_mut().as_dyn_stride_mut()
    }
}

impl<T, Cols: Shape> AsRowRef<T, Cols> for Row<T, Cols> {
    #[inline]
    fn as_row_ref(&self) -> RowRef<T, Cols> {
        self.as_dyn_stride()
    }
}

impl<T, Cols: Shape> AsRowMut<T, Cols> for Row<T, Cols> {
    #[inline]
    fn as_row_mut(&mut self) -> RowMut<T, Cols> {
        self.as_dyn_stride_mut()
    }
}
