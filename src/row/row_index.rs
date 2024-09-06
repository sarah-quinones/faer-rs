// RangeFull
// Range
// RangeInclusive
// RangeTo
// RangeToInclusive
// usize

use super::*;
use crate::assert;
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
