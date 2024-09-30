use crate::{mat::*, utils::slice::*, Conj, Shape};
use coe::Coerce;
use core::{marker::PhantomData, ptr::NonNull};
use faer_entity::*;
use reborrow::*;

#[repr(C)]
pub(crate) struct VecImpl<E: Entity, N: Shape = usize> {
    pub(crate) ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    pub(crate) len: N,
    pub(crate) stride: isize,
}
#[repr(C)]
pub(crate) struct VecOwnImpl<E: Entity> {
    pub(crate) ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    pub(crate) len: usize,
}

impl<E: Entity, N: Shape> Copy for VecImpl<E, N> {}
impl<E: Entity, N: Shape> Clone for VecImpl<E, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<E: Entity, N: Shape> Sync for VecImpl<E, N> {}
unsafe impl<E: Entity, N: Shape> Send for VecImpl<E, N> {}
unsafe impl<E: Entity> Sync for VecOwnImpl<E> {}
unsafe impl<E: Entity> Send for VecOwnImpl<E> {}

/// Represents a type that can be used to slice a column, such as an index or a range of indices.
pub trait ColIndex<RowRange>: crate::seal::Seal + Sized {
    /// Resulting type of the indexing operation.
    type Target;

    /// Index the column at `row`, without bound checks.
    #[allow(clippy::missing_safety_doc)]
    unsafe fn get_unchecked(this: Self, row: RowRange) -> Self::Target {
        <Self as ColIndex<RowRange>>::get(this, row)
    }
    /// Index the column at `row`.
    fn get(this: Self, row: RowRange) -> Self::Target;
}

/// Trait for types that can be converted to a column view.
pub trait AsColRef<E: Entity> {
    /// Convert to a column view.
    fn as_col_ref(&self) -> ColRef<'_, E>;
}
/// Trait for types that can be converted to a mutable column view.
pub trait AsColMut<E: Entity>: AsColRef<E> {
    /// Convert to a mutable column view.
    fn as_col_mut(&mut self) -> ColMut<'_, E>;
}

impl<E: Entity, T: AsColRef<E>> AsColRef<E> for &T {
    fn as_col_ref(&self) -> ColRef<'_, E> {
        (**self).as_col_ref()
    }
}
impl<E: Entity, T: AsColRef<E>> AsColRef<E> for &mut T {
    fn as_col_ref(&self) -> ColRef<'_, E> {
        (**self).as_col_ref()
    }
}

impl<E: Entity, T: AsColMut<E>> AsColMut<E> for &mut T {
    fn as_col_mut(&mut self) -> ColMut<'_, E> {
        (**self).as_col_mut()
    }
}

mod col_index;

mod colref;
pub use colref::{
    from_raw_parts, from_ref, from_ref_generic, from_repeated_ref, from_repeated_ref_generic,
    from_slice, from_slice_generic, ColRef,
};

mod colmut;
pub use colmut::{
    from_mut, from_mut_generic, from_raw_parts_mut, from_slice_mut, from_slice_mut_generic, ColMut,
};

mod colown;
pub use colown::Col;

/// Type that can be interpreted as a batch of column vectors. Can be a single column or a matrix.
pub trait ColBatch<E: Conjugate>: As2D<E> {
    /// Corresponding owning type.
    type Owned: ColBatchMut<E::Canonical>;

    /// Constructor of the owned type that initializes the values to zero.
    fn new_owned_zeros(nrows: usize, ncols: usize) -> Self::Owned;

    /// Constructor of the owned type that copies the values.
    fn new_owned_copied(src: &Self) -> Self::Owned;

    /// Resize an owned column or matrix.
    fn resize_owned(owned: &mut Self::Owned, nrows: usize, ncols: usize);
}

/// Type that can be interpreted as a mutable batch of column vectors. Can be a single column or a
/// matrix.
pub trait ColBatchMut<E: Conjugate>: As2DMut<E> + ColBatch<E> {}

impl<E: Conjugate, T: ColBatch<E>> ColBatch<E> for &T {
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

impl<E: Conjugate, T: ColBatch<E>> ColBatch<E> for &mut T {
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

impl<E: Conjugate, T: ColBatchMut<E>> ColBatchMut<E> for &mut T {}

impl<'a, FromE: Entity, ToE: Entity> Coerce<ColRef<'a, ToE>> for ColRef<'a, FromE> {
    #[inline(always)]
    fn coerce(self) -> ColRef<'a, ToE> {
        assert!(coe::is_same::<FromE, ToE>());
        unsafe { transmute_unchecked::<ColRef<'a, FromE>, ColRef<'a, ToE>>(self) }
    }
}
impl<'a, FromE: Entity, ToE: Entity> Coerce<ColMut<'a, ToE>> for ColMut<'a, FromE> {
    #[inline(always)]
    fn coerce(self) -> ColMut<'a, ToE> {
        assert!(coe::is_same::<FromE, ToE>());
        unsafe { transmute_unchecked::<ColMut<'a, FromE>, ColMut<'a, ToE>>(self) }
    }
}
