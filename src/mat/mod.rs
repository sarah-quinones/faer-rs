use crate::assert;
use crate::col::{Col, ColMut, ColRef};
use crate::row::{Row, RowMut, RowRef};
use crate::utils::slice::*;
use crate::Conj;
use coe::Coerce;
use core::marker::PhantomData;
use core::ptr::NonNull;
use faer_entity::*;
use reborrow::*;

#[repr(C)]
struct MatImpl<E: Entity> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
}
#[repr(C)]
struct MatOwnImpl<E: Entity> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    nrows: usize,
    ncols: usize,
}

unsafe impl<E: Entity> Sync for MatImpl<E> {}
unsafe impl<E: Entity> Send for MatImpl<E> {}
unsafe impl<E: Entity> Sync for MatOwnImpl<E> {}
unsafe impl<E: Entity> Send for MatOwnImpl<E> {}

impl<E: Entity> Copy for MatImpl<E> {}
impl<E: Entity> Clone for MatImpl<E> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

/// Represents a type that can be used to slice a matrix, such as an index or a range of indices.
pub trait MatIndex<RowRange, ColRange>: crate::seal::Seal + Sized {
    /// Resulting type of the indexing operation.
    type Target;

    /// Index the matrix at `(row, col)`, without bound checks.
    #[allow(clippy::missing_safety_doc)]
    unsafe fn get_unchecked(this: Self, row: RowRange, col: ColRange) -> Self::Target {
        <Self as MatIndex<RowRange, ColRange>>::get(this, row, col)
    }
    /// Index the matrix at `(row, col)`.
    fn get(this: Self, row: RowRange, col: ColRange) -> Self::Target;
}

/// Trait for types that can be converted to a matrix view.
///
/// This trait is implemented for types of the matrix family, like [`Mat`],
/// [`MatRef`], and [`MatMut`], but not for types like [`Col`], [`Row`], or
/// their families. For a more general trait, see [`As2D`].
pub trait AsMatRef<E: Entity> {
    /// Convert to a matrix view.
    fn as_mat_ref(&self) -> MatRef<'_, E>;
}
/// Trait for types that can be converted to a mutable matrix view.
///
/// This trait is implemented for types of the matrix family, like [`Mat`],
/// [`MatRef`], and [`MatMut`], but not for types like [`Col`], [`Row`], or
/// their families. For a more general trait, see [`As2D`].
pub trait AsMatMut<E: Entity> {
    /// Convert to a mutable matrix view.
    fn as_mat_mut(&mut self) -> MatMut<'_, E>;
}

/// Trait for types that can be converted to a 2D matrix view.
///
/// This trait is implemented for any type that can be represented as a
/// 2D matrix view, like [`Mat`], [`Row`], [`Col`], and their respective
/// references and mutable references. For a trait specific to the matrix
/// family, see [`AsMatRef`] or [`AsMatMut`].
pub trait As2D<E: Entity> {
    /// Convert to a 2D matrix view.
    fn as_2d_ref(&self) -> MatRef<'_, E>;
}
/// Trait for types that can be converted to a mutable 2D matrix view.
///
/// This trait is implemented for any type that can be represented as a
/// 2D matrix view, like [`Mat`], [`Row`], [`Col`], and their respective
/// references and mutable references. For a trait specific to the matrix
/// family, see [`AsMatRef`] or [`AsMatMut`].
pub trait As2DMut<E: Entity> {
    /// Convert to a mutable 2D matrix view.
    fn as_2d_mut(&mut self) -> MatMut<'_, E>;
}

// MAT INDEX
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
        fn get(
            this: Self,
            row: RowRange,
            col: RangeTo,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
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
            this.submatrix(
                row.start,
                col.start,
                row.end - row.start,
                col.end - col.start,
            )
        }
    }
    impl<E: Entity> MatIndex<Range, usize> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: Range, col: usize) -> Self {
            this.submatrix(row.start, col, row.end - row.start, 1)
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
    impl<E: Entity> MatIndex<RangeInclusive, usize> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeInclusive, col: usize) -> Self {
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
    impl<E: Entity> MatIndex<RangeFrom, usize> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFrom, col: usize) -> Self {
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
    impl<E: Entity> MatIndex<RangeTo, usize> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeTo, col: usize) -> Self {
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
    impl<E: Entity> MatIndex<RangeToInclusive, usize> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeToInclusive, col: usize) -> Self {
            assert!(row.end != usize::MAX);
            <Self as MatIndex<Range, usize>>::get(this, 0..row.end + 1, col)
        }
    }

    impl<E: Entity> MatIndex<usize, Range> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: usize, col: Range) -> Self {
            this.submatrix(row, col.start, 1, col.end - col.start)
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
        fn get(
            this: Self,
            row: RowRange,
            col: RangeTo,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
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
            this.submatrix_mut(
                row.start,
                col.start,
                row.end - row.start,
                col.end - col.start,
            )
        }
    }
    impl<E: Entity> MatIndex<Range, usize> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: Range, col: usize) -> Self {
            this.submatrix_mut(row.start, col, row.end - row.start, 1)
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
    impl<E: Entity> MatIndex<RangeInclusive, usize> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeInclusive, col: usize) -> Self {
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
    impl<E: Entity> MatIndex<RangeFrom, usize> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFrom, col: usize) -> Self {
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
    impl<E: Entity> MatIndex<RangeTo, usize> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeTo, col: usize) -> Self {
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
    impl<E: Entity> MatIndex<RangeToInclusive, usize> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeToInclusive, col: usize) -> Self {
            assert!(row.end != usize::MAX);
            <Self as MatIndex<Range, usize>>::get(this, 0..row.end + 1, col)
        }
    }

    impl<E: Entity> MatIndex<usize, Range> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: usize, col: Range) -> Self {
            this.submatrix_mut(row, col.start, 1, col.end - col.start)
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
};

// AS MAT
const _: () = {
    impl<E: Entity> AsMatRef<E> for MatRef<'_, E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            *self
        }
    }
    impl<E: Entity> AsMatRef<E> for &'_ MatRef<'_, E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            **self
        }
    }
    impl<E: Entity> AsMatRef<E> for MatMut<'_, E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            (*self).rb()
        }
    }
    impl<E: Entity> AsMatRef<E> for &'_ MatMut<'_, E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            (**self).rb()
        }
    }
    impl<E: Entity> AsMatRef<E> for Mat<E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            (*self).as_ref()
        }
    }
    impl<E: Entity> AsMatRef<E> for &'_ Mat<E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            (**self).as_ref()
        }
    }

    impl<E: Entity> AsMatMut<E> for MatMut<'_, E> {
        #[inline]
        fn as_mat_mut(&mut self) -> MatMut<'_, E> {
            (*self).rb_mut()
        }
    }

    impl<E: Entity> AsMatMut<E> for &'_ mut MatMut<'_, E> {
        #[inline]
        fn as_mat_mut(&mut self) -> MatMut<'_, E> {
            (**self).rb_mut()
        }
    }

    impl<E: Entity> AsMatMut<E> for Mat<E> {
        #[inline]
        fn as_mat_mut(&mut self) -> MatMut<'_, E> {
            (*self).as_mut()
        }
    }

    impl<E: Entity> AsMatMut<E> for &'_ mut Mat<E> {
        #[inline]
        fn as_mat_mut(&mut self) -> MatMut<'_, E> {
            (**self).as_mut()
        }
    }
};

// AS 2D
const _: () = {
    // Matrix family
    impl<E: Entity> As2D<E> for &'_ MatRef<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            **self
        }
    }

    impl<E: Entity> As2D<E> for MatRef<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            *self
        }
    }

    impl<E: Entity> As2D<E> for &'_ MatMut<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            (**self).rb()
        }
    }

    impl<E: Entity> As2D<E> for MatMut<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            (*self).rb()
        }
    }

    impl<E: Entity> As2D<E> for &'_ Mat<E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            (**self).as_ref()
        }
    }

    impl<E: Entity> As2D<E> for Mat<E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            (*self).as_ref()
        }
    }

    // Row Family
    impl<E: Entity> As2D<E> for &'_ RowRef<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            self.as_2d()
        }
    }

    impl<E: Entity> As2D<E> for RowRef<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            self.as_2d()
        }
    }

    impl<E: Entity> As2D<E> for &'_ RowMut<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            (**self).rb().as_2d()
        }
    }

    impl<E: Entity> As2D<E> for RowMut<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            self.rb().as_2d()
        }
    }

    impl<E: Entity> As2D<E> for &'_ Row<E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            (**self).as_ref().as_2d()
        }
    }

    impl<E: Entity> As2D<E> for Row<E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            self.as_ref().as_2d()
        }
    }

    // Col Family
    impl<E: Entity> As2D<E> for &'_ ColRef<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            self.as_2d()
        }
    }

    impl<E: Entity> As2D<E> for ColRef<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            self.as_2d()
        }
    }

    impl<E: Entity> As2D<E> for &'_ ColMut<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            (**self).rb().as_2d()
        }
    }

    impl<E: Entity> As2D<E> for ColMut<'_, E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            self.rb().as_2d()
        }
    }

    impl<E: Entity> As2D<E> for &'_ Col<E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            (**self).as_ref().as_2d()
        }
    }

    impl<E: Entity> As2D<E> for Col<E> {
        #[inline]
        fn as_2d_ref(&self) -> MatRef<'_, E> {
            self.as_ref().as_2d()
        }
    }
};

// AS 2D MUT
const _: () = {
    // Matrix family
    impl<E: Entity> As2DMut<E> for &'_ mut MatMut<'_, E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            (**self).rb_mut()
        }
    }

    impl<E: Entity> As2DMut<E> for MatMut<'_, E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            (*self).rb_mut()
        }
    }

    impl<E: Entity> As2DMut<E> for &'_ mut Mat<E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            (**self).as_mut()
        }
    }

    impl<E: Entity> As2DMut<E> for Mat<E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            (*self).as_mut()
        }
    }

    // Row Family
    impl<E: Entity> As2DMut<E> for &'_ mut RowMut<'_, E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            (**self).rb_mut().as_2d_mut()
        }
    }

    impl<E: Entity> As2DMut<E> for RowMut<'_, E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            self.rb_mut().as_2d_mut()
        }
    }

    impl<E: Entity> As2DMut<E> for &'_ mut Row<E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            (**self).as_mut().as_2d_mut()
        }
    }

    impl<E: Entity> As2DMut<E> for Row<E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            self.as_mut().as_2d_mut()
        }
    }

    // Col Family
    impl<E: Entity> As2DMut<E> for &'_ mut ColMut<'_, E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            (**self).rb_mut().as_2d_mut()
        }
    }

    impl<E: Entity> As2DMut<E> for ColMut<'_, E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            self.rb_mut().as_2d_mut()
        }
    }

    impl<E: Entity> As2DMut<E> for &'_ mut Col<E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            (**self).as_mut().as_2d_mut()
        }
    }

    impl<E: Entity> As2DMut<E> for Col<E> {
        #[inline]
        fn as_2d_mut(&mut self) -> MatMut<'_, E> {
            self.as_mut().as_2d_mut()
        }
    }
};

impl<'a, FromE: Entity, ToE: Entity> Coerce<MatRef<'a, ToE>> for MatRef<'a, FromE> {
    #[inline(always)]
    fn coerce(self) -> MatRef<'a, ToE> {
        assert!(coe::is_same::<FromE, ToE>());
        unsafe { transmute_unchecked::<MatRef<'a, FromE>, MatRef<'a, ToE>>(self) }
    }
}
impl<'a, FromE: Entity, ToE: Entity> Coerce<MatMut<'a, ToE>> for MatMut<'a, FromE> {
    #[inline(always)]
    fn coerce(self) -> MatMut<'a, ToE> {
        assert!(coe::is_same::<FromE, ToE>());
        unsafe { transmute_unchecked::<MatMut<'a, FromE>, MatMut<'a, ToE>>(self) }
    }
}

mod matref;
pub use matref::MatRef;

mod matmut;
pub use matmut::MatMut;

mod matown;
pub use matown::Mat;

pub(crate) mod matalloc;

/// Creates a `MatRef` from pointers to the matrix data, dimensions, and strides.
///
/// The row (resp. column) stride is the offset from the memory address of a given matrix
/// element at indices `(row: i, col: j)`, to the memory address of the matrix element at
/// indices `(row: i + 1, col: 0)` (resp. `(row: 0, col: i + 1)`). This offset is specified in
/// number of elements, not in bytes.
///
/// # Safety
/// The behavior is undefined if any of the following conditions are violated:
/// * For each matrix unit, the entire memory region addressed by the matrix must be contained
/// within a single allocation, accessible in its entirety by the corresponding pointer in
/// `ptr`.
/// * For each matrix unit, the corresponding pointer must be properly aligned,
/// even for a zero-sized matrix.
/// * The values accessible by the matrix must be initialized at some point before they are
/// read, or references to them are formed.
/// * No mutable aliasing is allowed. In other words, none of the elements accessible by any
/// matrix unit may be accessed for writes by any other means for the duration of the lifetime
/// `'a`.
///
/// # Example
///
/// ```
/// use faer::mat;
///
/// // row major matrix with 2 rows, 3 columns, with a column at the end that we want to skip.
/// // the row stride is the pointer offset from the address of 1.0 to the address of 4.0,
/// // which is 4.
/// // the column stride is the pointer offset from the address of 1.0 to the address of 2.0,
/// // which is 1.
/// let data = [[1.0, 2.0, 3.0, f64::NAN], [4.0, 5.0, 6.0, f64::NAN]];
/// let matrix = unsafe { mat::from_raw_parts::<f64>(data.as_ptr() as *const f64, 2, 3, 4, 1) };
///
/// let expected = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// assert_eq!(expected.as_ref(), matrix);
/// ```
#[inline(always)]
pub unsafe fn from_raw_parts<'a, E: Entity>(
    ptr: GroupFor<E, *const E::Unit>,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
) -> MatRef<'a, E> {
    MatRef::__from_raw_parts(ptr, nrows, ncols, row_stride, col_stride)
}

/// Creates a `MatMut` from pointers to the matrix data, dimensions, and strides.
///
/// The row (resp. column) stride is the offset from the memory address of a given matrix
/// element at indices `(row: i, col: j)`, to the memory address of the matrix element at
/// indices `(row: i + 1, col: 0)` (resp. `(row: 0, col: i + 1)`). This offset is specified in
/// number of elements, not in bytes.
///
/// # Safety
/// The behavior is undefined if any of the following conditions are violated:
/// * For each matrix unit, the entire memory region addressed by the matrix must be contained
/// within a single allocation, accessible in its entirety by the corresponding pointer in
/// `ptr`.
/// * For each matrix unit, the corresponding pointer must be non null and properly aligned,
/// even for a zero-sized matrix.
/// * The values accessible by the matrix must be initialized at some point before they are
///   read, or
/// references to them are formed.
/// * No aliasing (including self aliasing) is allowed. In other words, none of the elements
/// accessible by any matrix unit may be accessed for reads or writes by any other means for
/// the duration of the lifetime `'a`. No two elements within a single matrix unit may point to
/// the same address (such a thing can be achieved with a zero stride, for example), and no two
/// matrix units may point to the same address.
///
/// # Example
///
/// ```
/// use faer::mat;
///
/// // row major matrix with 2 rows, 3 columns, with a column at the end that we want to skip.
/// // the row stride is the pointer offset from the address of 1.0 to the address of 4.0,
/// // which is 4.
/// // the column stride is the pointer offset from the address of 1.0 to the address of 2.0,
/// // which is 1.
/// let mut data = [[1.0, 2.0, 3.0, f64::NAN], [4.0, 5.0, 6.0, f64::NAN]];
/// let mut matrix =
///     unsafe { mat::from_raw_parts_mut::<f64>(data.as_mut_ptr() as *mut f64, 2, 3, 4, 1) };
///
/// let expected = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// assert_eq!(expected.as_ref(), matrix);
/// ```
#[inline(always)]
pub unsafe fn from_raw_parts_mut<'a, E: Entity>(
    ptr: GroupFor<E, *mut E::Unit>,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
) -> MatMut<'a, E> {
    MatMut::__from_raw_parts(ptr, nrows, ncols, row_stride, col_stride)
}

impl<'a, E: Entity> core::fmt::Debug for MatRef<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        struct DebugRow<'a, T: Entity>(MatRef<'a, T>);

        impl<'a, T: Entity> core::fmt::Debug for DebugRow<'a, T> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let mut j = 0;
                f.debug_list()
                    .entries(core::iter::from_fn(|| {
                        let ret = if j < self.0.ncols() {
                            Some(T::faer_from_units(T::faer_deref(self.0.get(0, j))))
                        } else {
                            None
                        };
                        j += 1;
                        ret
                    }))
                    .finish()
            }
        }

        writeln!(f, "[")?;
        for i in 0..self.nrows() {
            let row = self.subrows(i, 1);
            DebugRow(row).fmt(f)?;
            f.write_str(",\n")?;
        }
        write!(f, "]")
    }
}

impl<'a, E: Entity> core::fmt::Debug for MatMut<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<E: Entity> core::fmt::Debug for Mat<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for MatRef<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &E {
        self.get(row, col)
    }
}

impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for MatMut<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &E {
        (*self).rb().get(row, col)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for MatMut<'_, E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut E {
        (*self).rb_mut().get_mut(row, col)
    }
}

impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for Mat<E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &E {
        self.as_ref().get(row, col)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for Mat<E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut E {
        self.as_mut().get_mut(row, col)
    }
}

/// Creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a column-major format, so that the first chunk of `nrows`
/// values from the slices goes in the first column of the matrix, the second chunk of `nrows`
/// values goes in the second column, and so on.
///
/// # Panics
/// The function panics if any of the following conditions are violated:
/// * `nrows * ncols == slice.len()`
///
/// # Example
/// ```
/// use faer::mat;
///
/// let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
/// let view = mat::from_column_major_slice::<f64>(&slice, 3, 2);
///
/// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
/// assert_eq!(expected, view);
/// ```
#[track_caller]
#[inline(always)]
pub fn from_column_major_slice<'a, E: Entity>(
    slice: GroupFor<E, &'a [E::Unit]>,
    nrows: usize,
    ncols: usize,
) -> MatRef<'a, E> {
    from_slice_assert(
        nrows,
        ncols,
        SliceGroup::<'_, E>::new(E::faer_copy(&slice)).len(),
    );

    unsafe {
        from_raw_parts(
            E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.as_ptr(),
            ),
            nrows,
            ncols,
            1,
            nrows as isize,
        )
    }
}

/// Creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a row-major format, so that the first chunk of `ncols`
/// values from the slices goes in the first column of the matrix, the second chunk of `ncols`
/// values goes in the second column, and so on.
///
/// # Panics
/// The function panics if any of the following conditions are violated:
/// * `nrows * ncols == slice.len()`
///
/// # Example
/// ```
/// use faer::mat;
///
/// let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
/// let view = mat::from_row_major_slice::<f64>(&slice, 3, 2);
///
/// let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// assert_eq!(expected, view);
/// ```
#[track_caller]
#[inline(always)]
pub fn from_row_major_slice<'a, E: Entity>(
    slice: GroupFor<E, &'a [E::Unit]>,
    nrows: usize,
    ncols: usize,
) -> MatRef<'a, E> {
    from_column_major_slice(slice, ncols, nrows).transpose()
}

/// Creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a column-major format, where the beginnings of two consecutive
/// columns are separated by `col_stride` elements.
#[track_caller]
pub fn from_column_major_slice_with_stride<'a, E: Entity>(
    slice: GroupFor<E, &'a [E::Unit]>,
    nrows: usize,
    ncols: usize,
    col_stride: usize,
) -> MatRef<'a, E> {
    from_strided_column_major_slice_assert(
        nrows,
        ncols,
        col_stride,
        SliceGroup::<'_, E>::new(E::faer_copy(&slice)).len(),
    );

    unsafe {
        from_raw_parts(
            E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.as_ptr(),
            ),
            nrows,
            ncols,
            1,
            col_stride as isize,
        )
    }
}

/// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a column-major format, so that the first chunk of `nrows`
/// values from the slices goes in the first column of the matrix, the second chunk of `nrows`
/// values goes in the second column, and so on.
///
/// # Panics
/// The function panics if any of the following conditions are violated:
/// * `nrows * ncols == slice.len()`
///
/// # Example
/// ```
/// use faer::mat;
///
/// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
/// let view = mat::from_column_major_slice_mut::<f64>(&mut slice, 3, 2);
///
/// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
/// assert_eq!(expected, view);
/// ```
#[track_caller]
pub fn from_column_major_slice_mut<'a, E: Entity>(
    slice: GroupFor<E, &'a mut [E::Unit]>,
    nrows: usize,
    ncols: usize,
) -> MatMut<'a, E> {
    from_slice_assert(
        nrows,
        ncols,
        SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&slice))).len(),
    );
    unsafe {
        from_raw_parts_mut(
            E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.as_mut_ptr(),
            ),
            nrows,
            ncols,
            1,
            nrows as isize,
        )
    }
}

/// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a row-major format, so that the first chunk of `ncols`
/// values from the slices goes in the first column of the matrix, the second chunk of `ncols`
/// values goes in the second column, and so on.
///
/// # Panics
/// The function panics if any of the following conditions are violated:
/// * `nrows * ncols == slice.len()`
///
/// # Example
/// ```
/// use faer::mat;
///
/// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
/// let view = mat::from_row_major_slice_mut::<f64>(&mut slice, 3, 2);
///
/// let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// assert_eq!(expected, view);
/// ```
#[inline(always)]
#[track_caller]
pub fn from_row_major_slice_mut<'a, E: Entity>(
    slice: GroupFor<E, &'a mut [E::Unit]>,
    nrows: usize,
    ncols: usize,
) -> MatMut<'a, E> {
    from_column_major_slice_mut(slice, ncols, nrows).transpose_mut()
}

/// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a column-major format, where the beginnings of two consecutive
/// columns are separated by `col_stride` elements.
#[track_caller]
pub fn from_column_major_slice_with_stride_mut<'a, E: Entity>(
    slice: GroupFor<E, &'a mut [E::Unit]>,
    nrows: usize,
    ncols: usize,
    col_stride: usize,
) -> MatMut<'a, E> {
    from_strided_column_major_slice_mut_assert(
        nrows,
        ncols,
        col_stride,
        SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&slice))).len(),
    );
    unsafe {
        from_raw_parts_mut(
            E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.as_mut_ptr(),
            ),
            nrows,
            ncols,
            1,
            col_stride as isize,
        )
    }
}

#[track_caller]
#[inline]
fn from_slice_assert(nrows: usize, ncols: usize, len: usize) {
    // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
    // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
    // we don't care
    let size = usize::checked_mul(nrows, ncols).unwrap_or(usize::MAX);
    assert!(size == len);
}

#[track_caller]
#[inline]
fn from_strided_column_major_slice_assert(
    nrows: usize,
    ncols: usize,
    col_stride: usize,
    len: usize,
) {
    // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
    // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
    // we don't care
    let last = usize::checked_mul(col_stride, ncols - 1)
        .and_then(|last_col| last_col.checked_add(nrows - 1))
        .unwrap_or(usize::MAX);
    assert!(last < len);
}

#[track_caller]
#[inline]
fn from_strided_column_major_slice_mut_assert(
    nrows: usize,
    ncols: usize,
    col_stride: usize,
    len: usize,
) {
    // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
    // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
    // we don't care
    let last = usize::checked_mul(col_stride, ncols - 1)
        .and_then(|last_col| last_col.checked_add(nrows - 1))
        .unwrap_or(usize::MAX);
    assert!(all(col_stride >= nrows, last < len));
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::Matrix<E> for MatRef<'_, E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Dense(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::DenseAccess<E> for MatRef<'_, E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::Matrix<E> for MatMut<'_, E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Dense(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::DenseAccess<E> for MatMut<'_, E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::Matrix<E> for Mat<E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Dense(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::DenseAccess<E> for Mat<E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}
