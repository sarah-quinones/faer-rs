use super::*;
use crate::assert;

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

impl<E: Entity, RowRange> MatIndex<RowRange, RangeFrom> for MatRef<'_, E>
where
    Self: MatIndex<RowRange, Range>,
{
    type Target = <Self as MatIndex<RowRange, Range>>::Target;

    #[track_caller]
    #[inline(always)]
    fn get(
        this: Self,
        row: RowRange,
        col: RangeFrom,
    ) -> <Self as MatIndex<RowRange, Range>>::Target {
        let ncols = this.ncols();
        <Self as MatIndex<RowRange, Range>>::get(this, row, col.start..ncols)
    }
}
impl<E: Entity, RowRange> MatIndex<RowRange, RangeTo> for MatRef<'_, E>
where
    Self: MatIndex<RowRange, Range>,
{
    type Target = <Self as MatIndex<RowRange, Range>>::Target;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RowRange, col: RangeTo) -> <Self as MatIndex<RowRange, Range>>::Target {
        <Self as MatIndex<RowRange, Range>>::get(this, row, 0..col.end)
    }
}
impl<E: Entity, RowRange> MatIndex<RowRange, RangeToInclusive> for MatRef<'_, E>
where
    Self: MatIndex<RowRange, Range>,
{
    type Target = <Self as MatIndex<RowRange, Range>>::Target;

    #[track_caller]
    #[inline(always)]
    fn get(
        this: Self,
        row: RowRange,
        col: RangeToInclusive,
    ) -> <Self as MatIndex<RowRange, Range>>::Target {
        assert!(col.end != usize::MAX);
        <Self as MatIndex<RowRange, Range>>::get(this, row, 0..col.end + 1)
    }
}
impl<E: Entity, RowRange> MatIndex<RowRange, RangeInclusive> for MatRef<'_, E>
where
    Self: MatIndex<RowRange, Range>,
{
    type Target = <Self as MatIndex<RowRange, Range>>::Target;

    #[track_caller]
    #[inline(always)]
    fn get(
        this: Self,
        row: RowRange,
        col: RangeInclusive,
    ) -> <Self as MatIndex<RowRange, Range>>::Target {
        assert!(*col.end() != usize::MAX);
        <Self as MatIndex<RowRange, Range>>::get(this, row, *col.start()..*col.end() + 1)
    }
}
impl<E: Entity, RowRange> MatIndex<RowRange, RangeFull> for MatRef<'_, E>
where
    Self: MatIndex<RowRange, Range>,
{
    type Target = <Self as MatIndex<RowRange, Range>>::Target;

    #[track_caller]
    #[inline(always)]
    fn get(
        this: Self,
        row: RowRange,
        col: RangeFull,
    ) -> <Self as MatIndex<RowRange, Range>>::Target {
        let _ = col;
        let ncols = this.ncols();
        <Self as MatIndex<RowRange, Range>>::get(this, row, 0..ncols)
    }
}

impl<E: Entity> MatIndex<RangeFull, Range> for MatRef<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeFull, col: Range) -> Self {
        let _ = row;
        assert!(col.start <= col.end);
        this.subcols(col.start, col.end - col.start)
    }
}
impl<'a, E: Entity> MatIndex<RangeFull, usize> for MatRef<'a, E> {
    type Target = ColRef<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeFull, col: usize) -> Self::Target {
        let _ = row;
        this.col(col)
    }
}

impl<E: Entity> MatIndex<Range, Range> for MatRef<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: Range, col: Range) -> Self {
        assert!(all(row.start <= row.end, col.start <= col.end));
        this.submatrix(
            row.start,
            col.start,
            row.end - row.start,
            col.end - col.start,
        )
    }
}
impl<'a, E: Entity> MatIndex<Range, usize> for MatRef<'a, E> {
    type Target = ColRef<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: Range, col: usize) -> Self::Target {
        assert!(row.start <= row.end);
        this.submatrix(row.start, col, row.end - row.start, 1)
            .col(0)
    }
}

impl<E: Entity> MatIndex<RangeInclusive, Range> for MatRef<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeInclusive, col: Range) -> Self {
        assert!(*row.end() != usize::MAX);
        <Self as MatIndex<Range, Range>>::get(this, *row.start()..*row.end() + 1, col)
    }
}
impl<'a, E: Entity> MatIndex<RangeInclusive, usize> for MatRef<'a, E> {
    type Target = ColRef<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeInclusive, col: usize) -> Self::Target {
        assert!(*row.end() != usize::MAX);
        <Self as MatIndex<Range, usize>>::get(this, *row.start()..*row.end() + 1, col)
    }
}

impl<E: Entity> MatIndex<RangeFrom, Range> for MatRef<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeFrom, col: Range) -> Self {
        let nrows = this.nrows();
        <Self as MatIndex<Range, Range>>::get(this, row.start..nrows, col)
    }
}
impl<'a, E: Entity> MatIndex<RangeFrom, usize> for MatRef<'a, E> {
    type Target = ColRef<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeFrom, col: usize) -> Self::Target {
        let nrows = this.nrows();
        <Self as MatIndex<Range, usize>>::get(this, row.start..nrows, col)
    }
}
impl<E: Entity> MatIndex<RangeTo, Range> for MatRef<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeTo, col: Range) -> Self {
        <Self as MatIndex<Range, Range>>::get(this, 0..row.end, col)
    }
}
impl<'a, E: Entity> MatIndex<RangeTo, usize> for MatRef<'a, E> {
    type Target = ColRef<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeTo, col: usize) -> Self::Target {
        <Self as MatIndex<Range, usize>>::get(this, 0..row.end, col)
    }
}

impl<E: Entity> MatIndex<RangeToInclusive, Range> for MatRef<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeToInclusive, col: Range) -> Self {
        assert!(row.end != usize::MAX);
        <Self as MatIndex<Range, Range>>::get(this, 0..row.end + 1, col)
    }
}
impl<'a, E: Entity> MatIndex<RangeToInclusive, usize> for MatRef<'a, E> {
    type Target = ColRef<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeToInclusive, col: usize) -> Self::Target {
        assert!(row.end != usize::MAX);
        <Self as MatIndex<Range, usize>>::get(this, 0..row.end + 1, col)
    }
}

impl<'a, E: Entity> MatIndex<usize, Range> for MatRef<'a, E> {
    type Target = RowRef<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: usize, col: Range) -> Self::Target {
        assert!(col.start <= col.end);
        this.submatrix(row, col.start, 1, col.end - col.start)
            .row(0)
    }
}

impl<E: Entity, RowRange> MatIndex<RowRange, RangeFrom> for MatMut<'_, E>
where
    Self: MatIndex<RowRange, Range>,
{
    type Target = <Self as MatIndex<RowRange, Range>>::Target;

    #[track_caller]
    #[inline(always)]
    fn get(
        this: Self,
        row: RowRange,
        col: RangeFrom,
    ) -> <Self as MatIndex<RowRange, Range>>::Target {
        let ncols = this.ncols();
        <Self as MatIndex<RowRange, Range>>::get(this, row, col.start..ncols)
    }
}
impl<E: Entity, RowRange> MatIndex<RowRange, RangeTo> for MatMut<'_, E>
where
    Self: MatIndex<RowRange, Range>,
{
    type Target = <Self as MatIndex<RowRange, Range>>::Target;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RowRange, col: RangeTo) -> <Self as MatIndex<RowRange, Range>>::Target {
        <Self as MatIndex<RowRange, Range>>::get(this, row, 0..col.end)
    }
}
impl<E: Entity, RowRange> MatIndex<RowRange, RangeToInclusive> for MatMut<'_, E>
where
    Self: MatIndex<RowRange, Range>,
{
    type Target = <Self as MatIndex<RowRange, Range>>::Target;

    #[track_caller]
    #[inline(always)]
    fn get(
        this: Self,
        row: RowRange,
        col: RangeToInclusive,
    ) -> <Self as MatIndex<RowRange, Range>>::Target {
        assert!(col.end != usize::MAX);
        <Self as MatIndex<RowRange, Range>>::get(this, row, 0..col.end + 1)
    }
}
impl<E: Entity, RowRange> MatIndex<RowRange, RangeInclusive> for MatMut<'_, E>
where
    Self: MatIndex<RowRange, Range>,
{
    type Target = <Self as MatIndex<RowRange, Range>>::Target;

    #[track_caller]
    #[inline(always)]
    fn get(
        this: Self,
        row: RowRange,
        col: RangeInclusive,
    ) -> <Self as MatIndex<RowRange, Range>>::Target {
        assert!(*col.end() != usize::MAX);
        <Self as MatIndex<RowRange, Range>>::get(this, row, *col.start()..*col.end() + 1)
    }
}
impl<E: Entity, RowRange> MatIndex<RowRange, RangeFull> for MatMut<'_, E>
where
    Self: MatIndex<RowRange, Range>,
{
    type Target = <Self as MatIndex<RowRange, Range>>::Target;

    #[track_caller]
    #[inline(always)]
    fn get(
        this: Self,
        row: RowRange,
        col: RangeFull,
    ) -> <Self as MatIndex<RowRange, Range>>::Target {
        let _ = col;
        let ncols = this.ncols();
        <Self as MatIndex<RowRange, Range>>::get(this, row, 0..ncols)
    }
}

impl<E: Entity> MatIndex<RangeFull, Range> for MatMut<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeFull, col: Range) -> Self {
        let _ = row;
        assert!(col.start <= col.end);
        this.subcols_mut(col.start, col.end - col.start)
    }
}
impl<'a, E: Entity> MatIndex<RangeFull, usize> for MatMut<'a, E> {
    type Target = ColMut<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeFull, col: usize) -> Self::Target {
        let _ = row;
        this.col_mut(col)
    }
}

impl<E: Entity> MatIndex<Range, Range> for MatMut<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: Range, col: Range) -> Self {
        assert!(all(row.start <= row.end, col.start <= col.end));
        this.submatrix_mut(
            row.start,
            col.start,
            row.end - row.start,
            col.end - col.start,
        )
    }
}
impl<'a, E: Entity> MatIndex<Range, usize> for MatMut<'a, E> {
    type Target = ColMut<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: Range, col: usize) -> Self::Target {
        assert!(row.start <= row.end);
        this.submatrix_mut(row.start, col, row.end - row.start, 1)
            .col_mut(0)
    }
}

impl<E: Entity> MatIndex<RangeInclusive, Range> for MatMut<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeInclusive, col: Range) -> Self {
        assert!(*row.end() != usize::MAX);
        <Self as MatIndex<Range, Range>>::get(this, *row.start()..*row.end() + 1, col)
    }
}
impl<'a, E: Entity> MatIndex<RangeInclusive, usize> for MatMut<'a, E> {
    type Target = ColMut<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeInclusive, col: usize) -> Self::Target {
        assert!(*row.end() != usize::MAX);
        <Self as MatIndex<Range, usize>>::get(this, *row.start()..*row.end() + 1, col)
    }
}

impl<E: Entity> MatIndex<RangeFrom, Range> for MatMut<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeFrom, col: Range) -> Self {
        let nrows = this.nrows();
        <Self as MatIndex<Range, Range>>::get(this, row.start..nrows, col)
    }
}
impl<'a, E: Entity> MatIndex<RangeFrom, usize> for MatMut<'a, E> {
    type Target = ColMut<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeFrom, col: usize) -> Self::Target {
        let nrows = this.nrows();
        <Self as MatIndex<Range, usize>>::get(this, row.start..nrows, col)
    }
}
impl<E: Entity> MatIndex<RangeTo, Range> for MatMut<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeTo, col: Range) -> Self {
        <Self as MatIndex<Range, Range>>::get(this, 0..row.end, col)
    }
}
impl<'a, E: Entity> MatIndex<RangeTo, usize> for MatMut<'a, E> {
    type Target = ColMut<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeTo, col: usize) -> Self::Target {
        <Self as MatIndex<Range, usize>>::get(this, 0..row.end, col)
    }
}

impl<E: Entity> MatIndex<RangeToInclusive, Range> for MatMut<'_, E> {
    type Target = Self;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeToInclusive, col: Range) -> Self {
        assert!(row.end != usize::MAX);
        <Self as MatIndex<Range, Range>>::get(this, 0..row.end + 1, col)
    }
}
impl<'a, E: Entity> MatIndex<RangeToInclusive, usize> for MatMut<'a, E> {
    type Target = ColMut<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: RangeToInclusive, col: usize) -> Self::Target {
        assert!(row.end != usize::MAX);
        <Self as MatIndex<Range, usize>>::get(this, 0..row.end + 1, col)
    }
}

impl<'a, E: Entity> MatIndex<usize, Range> for MatMut<'a, E> {
    type Target = RowMut<'a, E>;

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: usize, col: Range) -> Self::Target {
        assert!(col.start <= col.end);
        this.submatrix_mut(row, col.start, 1, col.end - col.start)
            .row_mut(0)
    }
}

impl<'a, E: Entity> MatIndex<usize, usize> for MatRef<'a, E> {
    type Target = GroupFor<E, &'a E::Unit>;

    #[track_caller]
    #[inline(always)]
    unsafe fn get_unchecked(this: Self, row: usize, col: usize) -> Self::Target {
        unsafe { E::faer_map(this.ptr_inbounds_at(row, col), |ptr| &*ptr) }
    }

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: usize, col: usize) -> Self::Target {
        assert!(all(row < this.nrows(), col < this.ncols()));
        unsafe { <Self as MatIndex<usize, usize>>::get_unchecked(this, row, col) }
    }
}

impl<'a, E: Entity> MatIndex<usize, usize> for MatMut<'a, E> {
    type Target = GroupFor<E, &'a mut E::Unit>;

    #[track_caller]
    #[inline(always)]
    unsafe fn get_unchecked(this: Self, row: usize, col: usize) -> Self::Target {
        unsafe { E::faer_map(this.ptr_inbounds_at_mut(row, col), |ptr| &mut *ptr) }
    }

    #[track_caller]
    #[inline(always)]
    fn get(this: Self, row: usize, col: usize) -> Self::Target {
        assert!(all(row < this.nrows(), col < this.ncols()));
        unsafe { <Self as MatIndex<usize, usize>>::get_unchecked(this, row, col) }
    }
}
