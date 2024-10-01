use crate::{col::VecImpl, mat::*, utils::slice::*, Conj, Shape};
use coe::Coerce;
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
    /// Column dimension of the row.
    type C: Shape;

    /// Convert to a row view.
    fn as_row_ref(&self) -> RowRef<'_, E, Self::C>;
}
/// Trait for types that can be converted to a mutable row view.
pub trait AsRowMut<E: Entity>: AsRowRef<E> {
    /// Convert to a mutable row view.
    fn as_row_mut(&mut self) -> RowMut<'_, E, Self::C>;
}

impl<E: Entity, T: AsRowRef<E>> AsRowRef<E> for &T {
    type C = T::C;

    fn as_row_ref(&self) -> RowRef<'_, E, Self::C> {
        (**self).as_row_ref()
    }
}
impl<E: Entity, T: AsRowRef<E>> AsRowRef<E> for &mut T {
    type C = T::C;

    fn as_row_ref(&self) -> RowRef<'_, E, Self::C> {
        (**self).as_row_ref()
    }
}

impl<E: Entity, T: AsRowMut<E>> AsRowMut<E> for &mut T {
    fn as_row_mut(&mut self) -> RowMut<'_, E, Self::C> {
        (**self).as_row_mut()
    }
}

mod row_index;

mod rowref;
pub use rowref::{
    from_raw_parts, from_ref, from_ref_generic, from_repeated_ref, from_repeated_ref_generic,
    from_slice, from_slice_generic, RowRef,
};

mod rowmut;
pub use rowmut::{
    from_mut, from_mut_generic, from_raw_parts_mut, from_slice_mut, from_slice_mut_generic, RowMut,
};

mod rowown;
pub use rowown::Row;

/// Type that can be interpreted as a batch of row vectors. Can be a single row or a matrix.
pub trait RowBatch<E: Conjugate>: As2D<E> {
    /// Corresponding owning type.
    type Owned: RowBatch<E::Canonical>;

    /// Constructor of the owned type that initializes the values to zero.
    fn new_owned_zeros(nrows: usize, ncols: usize) -> Self::Owned;

    /// Constructor of the owned type that copies the values.
    fn new_owned_copied(src: &Self) -> Self::Owned;

    /// Resize an owned column or matrix.
    fn resize_owned(owned: &mut Self::Owned, nrows: usize, ncols: usize);
}

/// Type that can be interpreted as a mutable batch of row vectors. Can be a single row or a
/// matrix.
pub trait RowBatchMut<E: Conjugate>: As2DMut<E> + RowBatch<E> {}

impl<E: Conjugate, T: RowBatch<E>> RowBatch<E> for &T {
    type Owned = T::Owned;

    #[inline]
    #[track_caller]
    fn new_owned_zeros(nrows: usize, ncols: usize) -> Self::Owned {
        T::new_owned_zeros(nrows, ncols)
    }

    #[inline]
    fn new_owned_copied(src: &Self) -> Self::Owned {
        T::new_owned_copied(src)
    }

    #[inline]
    fn resize_owned(owned: &mut Self::Owned, nrows: usize, ncols: usize) {
        T::resize_owned(owned, nrows, ncols)
    }
}

impl<E: Conjugate, T: RowBatch<E>> RowBatch<E> for &mut T {
    type Owned = T::Owned;

    #[inline]
    #[track_caller]
    fn new_owned_zeros(nrows: usize, ncols: usize) -> Self::Owned {
        T::new_owned_zeros(nrows, ncols)
    }

    #[inline]
    fn new_owned_copied(src: &Self) -> Self::Owned {
        T::new_owned_copied(src)
    }

    #[inline]
    fn resize_owned(owned: &mut Self::Owned, nrows: usize, ncols: usize) {
        T::resize_owned(owned, nrows, ncols)
    }
}

impl<E: Conjugate, T: RowBatchMut<E>> RowBatchMut<E> for &mut T {}

impl<'a, FromE: Entity, ToE: Entity> Coerce<RowRef<'a, ToE>> for RowRef<'a, FromE> {
    #[inline(always)]
    fn coerce(self) -> RowRef<'a, ToE> {
        assert!(coe::is_same::<FromE, ToE>());
        unsafe { transmute_unchecked::<RowRef<'a, FromE>, RowRef<'a, ToE>>(self) }
    }
}
impl<'a, FromE: Entity, ToE: Entity> Coerce<RowMut<'a, ToE>> for RowMut<'a, FromE> {
    #[inline(always)]
    fn coerce(self) -> RowMut<'a, ToE> {
        assert!(coe::is_same::<FromE, ToE>());
        unsafe { transmute_unchecked::<RowMut<'a, FromE>, RowMut<'a, ToE>>(self) }
    }
}
