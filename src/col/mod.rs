use crate::{utils::slice::*, Conj};
use core::{marker::PhantomData, ptr::NonNull};
use faer_entity::*;
use reborrow::*;

#[repr(C)]
struct VecImpl<E: Entity> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    len: usize,
    stride: isize,
}
#[repr(C)]
struct VecOwnImpl<E: Entity> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    len: usize,
}

impl<E: Entity> Copy for VecImpl<E> {}
impl<E: Entity> Clone for VecImpl<E> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<E: Entity> Sync for VecImpl<E> {}
unsafe impl<E: Entity> Send for VecImpl<E> {}
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
pub trait AsColMut<E: Entity> {
    /// Convert to a mutable column view.
    fn as_col_mut(&mut self) -> ColMut<'_, E>;
}

// AS COL
const _: () = {
    impl<E: Entity> AsColRef<E> for ColRef<'_, E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            *self
        }
    }
    impl<E: Entity> AsColRef<E> for &'_ ColRef<'_, E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            **self
        }
    }
    impl<E: Entity> AsColRef<E> for ColMut<'_, E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            (*self).rb()
        }
    }
    impl<E: Entity> AsColRef<E> for &'_ ColMut<'_, E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            (**self).rb()
        }
    }
    impl<E: Entity> AsColRef<E> for Col<E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            (*self).as_ref()
        }
    }
    impl<E: Entity> AsColRef<E> for &'_ Col<E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            (**self).as_ref()
        }
    }

    impl<E: Entity> AsColMut<E> for ColMut<'_, E> {
        #[inline]
        fn as_col_mut(&mut self) -> ColMut<'_, E> {
            (*self).rb_mut()
        }
    }

    impl<E: Entity> AsColMut<E> for &'_ mut ColMut<'_, E> {
        #[inline]
        fn as_col_mut(&mut self) -> ColMut<'_, E> {
            (**self).rb_mut()
        }
    }

    impl<E: Entity> AsColMut<E> for Col<E> {
        #[inline]
        fn as_col_mut(&mut self) -> ColMut<'_, E> {
            (*self).as_mut()
        }
    }

    impl<E: Entity> AsColMut<E> for &'_ mut Col<E> {
        #[inline]
        fn as_col_mut(&mut self) -> ColMut<'_, E> {
            (**self).as_mut()
        }
    }
};

// COL INDEX
const _: () = {
    // RangeFull
    // Range
    // RangeInclusive
    // RangeTo
    // RangeToInclusive
    // usize

    use core::ops::RangeFull;
    type Range = core::ops::Range<usize>;
    type RangeInclusive = core::ops::RangeInclusive<usize>;
    type RangeFrom = core::ops::RangeFrom<usize>;
    type RangeTo = core::ops::RangeTo<usize>;
    type RangeToInclusive = core::ops::RangeToInclusive<usize>;

    impl<E: Entity> ColIndex<RangeFull> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFull) -> Self {
            let _ = row;
            this
        }
    }

    impl<E: Entity> ColIndex<Range> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: Range) -> Self {
            this.subrows(row.start, row.end - row.start)
        }
    }

    impl<E: Entity> ColIndex<RangeInclusive> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeInclusive) -> Self {
            assert!(*row.end() != usize::MAX);
            <Self as ColIndex<Range>>::get(this, *row.start()..*row.end() + 1)
        }
    }

    impl<E: Entity> ColIndex<RangeFrom> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFrom) -> Self {
            let nrows = this.nrows();
            <Self as ColIndex<Range>>::get(this, row.start..nrows)
        }
    }
    impl<E: Entity> ColIndex<RangeTo> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeTo) -> Self {
            <Self as ColIndex<Range>>::get(this, 0..row.end)
        }
    }

    impl<E: Entity> ColIndex<RangeToInclusive> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeToInclusive) -> Self {
            assert!(row.end != usize::MAX);
            <Self as ColIndex<Range>>::get(this, 0..row.end + 1)
        }
    }

    impl<'a, E: Entity> ColIndex<usize> for ColRef<'a, E> {
        type Target = GroupFor<E, &'a E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, row: usize) -> Self::Target {
            unsafe { E::faer_map(this.ptr_inbounds_at(row), |ptr: *const _| &*ptr) }
        }

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: usize) -> Self::Target {
            assert!(row < this.nrows());
            unsafe { <Self as ColIndex<usize>>::get_unchecked(this, row) }
        }
    }

    impl<E: Entity> ColIndex<RangeFull> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFull) -> Self {
            let _ = row;
            this
        }
    }

    impl<E: Entity> ColIndex<Range> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: Range) -> Self {
            this.subrows_mut(row.start, row.end - row.start)
        }
    }

    impl<E: Entity> ColIndex<RangeInclusive> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeInclusive) -> Self {
            assert!(*row.end() != usize::MAX);
            <Self as ColIndex<Range>>::get(this, *row.start()..*row.end() + 1)
        }
    }

    impl<E: Entity> ColIndex<RangeFrom> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFrom) -> Self {
            let nrows = this.nrows();
            <Self as ColIndex<Range>>::get(this, row.start..nrows)
        }
    }
    impl<E: Entity> ColIndex<RangeTo> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeTo) -> Self {
            <Self as ColIndex<Range>>::get(this, 0..row.end)
        }
    }

    impl<E: Entity> ColIndex<RangeToInclusive> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeToInclusive) -> Self {
            assert!(row.end != usize::MAX);
            <Self as ColIndex<Range>>::get(this, 0..row.end + 1)
        }
    }

    impl<'a, E: Entity> ColIndex<usize> for ColMut<'a, E> {
        type Target = GroupFor<E, &'a mut E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, row: usize) -> Self::Target {
            unsafe { E::faer_map(this.ptr_inbounds_at_mut(row), |ptr: *mut _| &mut *ptr) }
        }

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: usize) -> Self::Target {
            assert!(row < this.nrows());
            unsafe { <Self as ColIndex<usize>>::get_unchecked(this, row) }
        }
    }
};

mod colref;
pub use colref::ColRef;

mod colmut;
pub use colmut::ColMut;

mod colown;
pub use colown::Col;

/// Creates a `ColRef` from pointers to the column vector data, number of rows, and row stride.
///
/// # Safety:
/// This function has the same safety requirements as
/// [`mat::from_raw_parts(ptr, nrows, 1, row_stride, 0)`]
#[inline(always)]
pub unsafe fn from_raw_parts<'a, E: Entity>(
    ptr: GroupFor<E, *const E::Unit>,
    nrows: usize,
    row_stride: isize,
) -> ColRef<'a, E> {
    ColRef::__from_raw_parts(ptr, nrows, row_stride)
}

/// Creates a `ColMut` from pointers to the column vector data, number of rows, and row stride.
///
/// # Safety:
/// This function has the same safety requirements as
/// [`mat::from_raw_parts_mut(ptr, nrows, 1, row_stride, 0)`]
#[inline(always)]
pub unsafe fn from_raw_parts_mut<'a, E: Entity>(
    ptr: GroupFor<E, *mut E::Unit>,
    nrows: usize,
    row_stride: isize,
) -> ColMut<'a, E> {
    ColMut::__from_raw_parts(ptr, nrows, row_stride)
}

/// Creates a `ColRef` from slice views over the column vector data, The result has the same
/// number of rows as the length of the input slice.
#[inline(always)]
pub fn from_slice<E: Entity>(slice: GroupFor<E, &[E::Unit]>) -> ColRef<'_, E> {
    let nrows = SliceGroup::<'_, E>::new(E::faer_copy(&slice)).len();

    unsafe {
        from_raw_parts(
            E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.as_ptr(),
            ),
            nrows,
            1,
        )
    }
}

/// Creates a `ColMut` from slice views over the column vector data, The result has the same
/// number of rows as the length of the input slice.
#[inline(always)]
pub fn from_slice_mut<E: Entity>(slice: GroupFor<E, &mut [E::Unit]>) -> ColMut<'_, E> {
    let nrows = SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&slice))).len();

    unsafe {
        from_raw_parts_mut(
            E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.as_mut_ptr(),
            ),
            nrows,
            1,
        )
    }
}

impl<'a, E: Entity> core::fmt::Debug for ColRef<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_2d().fmt(f)
    }
}
impl<'a, E: Entity> core::fmt::Debug for ColMut<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<E: Entity> core::fmt::Debug for Col<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<E: SimpleEntity> core::ops::Index<usize> for ColRef<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, row: usize) -> &E {
        self.get(row)
    }
}

impl<E: SimpleEntity> core::ops::Index<usize> for ColMut<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, row: usize) -> &E {
        (*self).rb().get(row)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<usize> for ColMut<'_, E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, row: usize) -> &mut E {
        (*self).rb_mut().get_mut(row)
    }
}

impl<E: SimpleEntity> core::ops::Index<usize> for Col<E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, row: usize) -> &E {
        self.as_ref().get(row)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<usize> for Col<E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, row: usize) -> &mut E {
        self.as_mut().get_mut(row)
    }
}
