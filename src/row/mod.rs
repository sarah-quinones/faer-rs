use crate::{
    col::{VecImpl, VecOwnImpl},
    utils::slice::*,
    Conj,
};
use core::{marker::PhantomData, ptr::NonNull};
use faer_entity::*;
use reborrow::*;

/// Represents a type that can be used to slice a row, such as an index or a range of indices.
pub trait RowIndex<ColRange>: crate::seal::Seal + Sized {
    /// Resulting type of the indexing operation.
    type Target;

    /// Index the row at `col`, without bound checks.
    #[allow(clippy::missing_safety_doc)]
    unsafe fn get_unchecked(this: Self, col: ColRange) -> Self::Target {
        <Self as RowIndex<ColRange>>::get(this, col)
    }
    /// Index the row at `col`.
    fn get(this: Self, col: ColRange) -> Self::Target;
}

/// Trait for types that can be converted to a row view.
pub trait AsRowRef<E: Entity> {
    /// Convert to a row view.
    fn as_row_ref(&self) -> RowRef<'_, E>;
}
/// Trait for types that can be converted to a mutable row view.
pub trait AsRowMut<E: Entity> {
    /// Convert to a mutable row view.
    fn as_row_mut(&mut self) -> RowMut<'_, E>;
}

mod row_index;

mod rowref;
pub use rowref::{from_raw_parts, from_slice, RowRef};

mod rowmut;
pub use rowmut::{from_raw_parts_mut, from_slice_mut, RowMut};

mod rowown;
pub use rowown::Row;
