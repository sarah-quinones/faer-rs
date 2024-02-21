// RangeFull
// Range
// RangeInclusive
// RangeTo
// RangeToInclusive
// usize

use super::*;
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
