use crate::{internal_prelude::*, Shape};
use faer_traits::Unit;

pub(crate) mod rowmut;
pub(crate) mod rowown;
pub(crate) mod rowref;

pub use rowmut::RowMut as RowMutGeneric;
pub use rowown::Row as RowGeneric;
pub use rowref::RowRef as RowRefGeneric;

pub type RowRef<'a, T, Cols = usize, CStride = isize> = RowRefGeneric<'a, Unit, T, Cols, CStride>;
pub type RowMut<'a, T, Cols = usize, CStride = isize> = RowMutGeneric<'a, Unit, T, Cols, CStride>;
pub type Row<T, Cols = usize> = RowGeneric<Unit, T, Cols>;

pub trait AsRowMut<C: Container, T, Cols: Shape> {
    fn as_row_mut(&mut self) -> RowMutGeneric<C, T, Cols>;
}
pub trait AsRowRef<C: Container, T, Cols: Shape> {
    fn as_row_ref(&self) -> RowRefGeneric<C, T, Cols>;
}

impl<C: Container, T, Cols: Shape, CStride: Stride> AsRowRef<C, T, Cols>
    for RowRefGeneric<'_, C, T, Cols, CStride>
{
    #[inline]
    fn as_row_ref(&self) -> RowRefGeneric<C, T, Cols> {
        self.as_dyn_stride()
    }
}

impl<C: Container, T, Cols: Shape, CStride: Stride> AsRowRef<C, T, Cols>
    for RowMutGeneric<'_, C, T, Cols, CStride>
{
    #[inline]
    fn as_row_ref(&self) -> RowRefGeneric<C, T, Cols> {
        self.rb().as_dyn_stride()
    }
}

impl<C: Container, T, Cols: Shape, CStride: Stride> AsRowMut<C, T, Cols>
    for RowMutGeneric<'_, C, T, Cols, CStride>
{
    #[inline]
    fn as_row_mut(&mut self) -> RowMutGeneric<C, T, Cols> {
        self.rb_mut().as_dyn_stride_mut()
    }
}
