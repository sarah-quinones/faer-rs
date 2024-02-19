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

// AS ROW
const _: () = {
    impl<E: Entity> AsRowRef<E> for RowRef<'_, E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            *self
        }
    }
    impl<E: Entity> AsRowRef<E> for &'_ RowRef<'_, E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            **self
        }
    }
    impl<E: Entity> AsRowRef<E> for RowMut<'_, E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            (*self).rb()
        }
    }
    impl<E: Entity> AsRowRef<E> for &'_ RowMut<'_, E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            (**self).rb()
        }
    }
    impl<E: Entity> AsRowRef<E> for Row<E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            (*self).as_ref()
        }
    }
    impl<E: Entity> AsRowRef<E> for &'_ Row<E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            (**self).as_ref()
        }
    }

    impl<E: Entity> AsRowMut<E> for RowMut<'_, E> {
        #[inline]
        fn as_row_mut(&mut self) -> RowMut<'_, E> {
            (*self).rb_mut()
        }
    }

    impl<E: Entity> AsRowMut<E> for &'_ mut RowMut<'_, E> {
        #[inline]
        fn as_row_mut(&mut self) -> RowMut<'_, E> {
            (**self).rb_mut()
        }
    }

    impl<E: Entity> AsRowMut<E> for Row<E> {
        #[inline]
        fn as_row_mut(&mut self) -> RowMut<'_, E> {
            (*self).as_mut()
        }
    }

    impl<E: Entity> AsRowMut<E> for &'_ mut Row<E> {
        #[inline]
        fn as_row_mut(&mut self) -> RowMut<'_, E> {
            (**self).as_mut()
        }
    }
};
// ROW INDEX
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

    impl<E: Entity> RowIndex<RangeFull> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeFull) -> Self {
            let _ = col;
            this
        }
    }

    impl<E: Entity> RowIndex<Range> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: Range) -> Self {
            this.subcols(col.start, col.end - col.start)
        }
    }

    impl<E: Entity> RowIndex<RangeInclusive> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeInclusive) -> Self {
            assert!(*col.end() != usize::MAX);
            <Self as RowIndex<Range>>::get(this, *col.start()..*col.end() + 1)
        }
    }

    impl<E: Entity> RowIndex<RangeFrom> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeFrom) -> Self {
            let ncols = this.ncols();
            <Self as RowIndex<Range>>::get(this, col.start..ncols)
        }
    }
    impl<E: Entity> RowIndex<RangeTo> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeTo) -> Self {
            <Self as RowIndex<Range>>::get(this, 0..col.end)
        }
    }

    impl<E: Entity> RowIndex<RangeToInclusive> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeToInclusive) -> Self {
            assert!(col.end != usize::MAX);
            <Self as RowIndex<Range>>::get(this, 0..col.end + 1)
        }
    }

    impl<'a, E: Entity> RowIndex<usize> for RowRef<'a, E> {
        type Target = GroupFor<E, &'a E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, col: usize) -> Self::Target {
            unsafe { E::faer_map(this.ptr_inbounds_at(col), |ptr: *const _| &*ptr) }
        }

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: usize) -> Self::Target {
            assert!(col < this.ncols());
            unsafe { <Self as RowIndex<usize>>::get_unchecked(this, col) }
        }
    }

    impl<E: Entity> RowIndex<RangeFull> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeFull) -> Self {
            let _ = col;
            this
        }
    }

    impl<E: Entity> RowIndex<Range> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: Range) -> Self {
            this.subcols_mut(col.start, col.end - col.start)
        }
    }

    impl<E: Entity> RowIndex<RangeInclusive> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeInclusive) -> Self {
            assert!(*col.end() != usize::MAX);
            <Self as RowIndex<Range>>::get(this, *col.start()..*col.end() + 1)
        }
    }

    impl<E: Entity> RowIndex<RangeFrom> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeFrom) -> Self {
            let ncols = this.ncols();
            <Self as RowIndex<Range>>::get(this, col.start..ncols)
        }
    }

    impl<E: Entity> RowIndex<RangeTo> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeTo) -> Self {
            <Self as RowIndex<Range>>::get(this, 0..col.end)
        }
    }

    impl<E: Entity> RowIndex<RangeToInclusive> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeToInclusive) -> Self {
            assert!(col.end != usize::MAX);
            <Self as RowIndex<Range>>::get(this, 0..col.end + 1)
        }
    }

    impl<'a, E: Entity> RowIndex<usize> for RowMut<'a, E> {
        type Target = GroupFor<E, &'a mut E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, col: usize) -> Self::Target {
            unsafe { E::faer_map(this.ptr_inbounds_at_mut(col), |ptr: *mut _| &mut *ptr) }
        }

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: usize) -> Self::Target {
            assert!(col < this.ncols());
            unsafe { <Self as RowIndex<usize>>::get_unchecked(this, col) }
        }
    }
};

mod rowref;
pub use rowref::RowRef;

mod rowmut;
pub use rowmut::RowMut;

mod rowown;
pub use rowown::Row;

/// Creates a `RowRef` from pointers to the row vector data, number of columns, and column
/// stride.
///
/// # Safety:
/// This function has the same safety requirements as
/// [`mat::from_raw_parts(ptr, 1, ncols, 0, col_stride)`]
#[inline(always)]
pub unsafe fn from_raw_parts<'a, E: Entity>(
    ptr: GroupFor<E, *const E::Unit>,
    ncols: usize,
    col_stride: isize,
) -> RowRef<'a, E> {
    RowRef::__from_raw_parts(ptr, ncols, col_stride)
}

/// Creates a `RowMut` from pointers to the row vector data, number of columns, and column
/// stride.
///
/// # Safety:
/// This function has the same safety requirements as
/// [`mat::from_raw_parts_mut(ptr, 1, ncols, 0, col_stride)`]
#[inline(always)]
pub unsafe fn from_raw_parts_mut<'a, E: Entity>(
    ptr: GroupFor<E, *mut E::Unit>,
    ncols: usize,
    col_stride: isize,
) -> RowMut<'a, E> {
    RowMut::__from_raw_parts(ptr, ncols, col_stride)
}

/// Creates a `RowRef` from slice views over the row vector data, The result has the same
/// number of columns as the length of the input slice.
#[inline(always)]
pub fn from_slice<E: Entity>(slice: GroupFor<E, &[E::Unit]>) -> RowRef<'_, E> {
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

/// Creates a `RowMut` from slice views over the row vector data, The result has the same
/// number of columns as the length of the input slice.
#[inline(always)]
pub fn from_slice_mut<E: Entity>(slice: GroupFor<E, &mut [E::Unit]>) -> RowMut<'_, E> {
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

impl<'a, E: Entity> core::fmt::Debug for RowRef<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_2d().fmt(f)
    }
}
impl<'a, E: Entity> core::fmt::Debug for RowMut<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}
impl<E: Entity> core::fmt::Debug for Row<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<E: SimpleEntity> core::ops::Index<usize> for RowRef<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, col: usize) -> &E {
        self.get(col)
    }
}

impl<E: SimpleEntity> core::ops::Index<usize> for RowMut<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, col: usize) -> &E {
        (*self).rb().get(col)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<usize> for RowMut<'_, E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, col: usize) -> &mut E {
        (*self).rb_mut().get_mut(col)
    }
}

impl<E: SimpleEntity> core::ops::Index<usize> for Row<E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, col: usize) -> &E {
        self.as_ref().get(col)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<usize> for Row<E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, col: usize) -> &mut E {
        self.as_mut().get_mut(col)
    }
}
