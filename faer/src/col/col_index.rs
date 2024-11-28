use super::*;
use crate::{assert, debug_assert, into_range::IntoRange, Idx, IdxInc};

impl<'a, R: Shape, T, Rs: Stride, RowRange: IntoRange<IdxInc<R>, Len<R>: 'a>> ColIndex<RowRange>
    for ColRef<'a, T, R, Rs>
{
    type Target = ColRef<'a, T, RowRange::Len<R>, Rs>;

    #[track_caller]
    #[inline]
    fn get(this: Self, row: RowRange) -> Self::Target {
        let row = row.into_range(R::start(), this.nrows().end());
        assert!(all(row.start <= row.end, row.end <= this.nrows()));
        let nrows =
            unsafe { RowRange::Len::<R>::new_unbound(row.end.unbound() - row.start.unbound()) };

        this.subrows(row.start, nrows)
    }

    #[track_caller]
    #[inline]
    unsafe fn get_unchecked(this: Self, row: RowRange) -> Self::Target {
        let row = row.into_range(R::start(), this.nrows().end());

        debug_assert!(all(row.start <= row.end, row.end <= this.nrows(),));
        let nrows =
            unsafe { RowRange::Len::<R>::new_unbound(row.end.unbound() - row.start.unbound()) };

        this.subrows(row.start, nrows)
    }
}

impl<'a, R: Shape, T, Rs: Stride, RowRange: IntoRange<IdxInc<R>, Len<R>: 'a>> ColIndex<RowRange>
    for ColMut<'a, T, R, Rs>
{
    type Target = ColMut<'a, T, RowRange::Len<R>, Rs>;

    #[track_caller]
    #[inline]
    fn get(this: Self, row: RowRange) -> Self::Target {
        let row = row.into_range(R::start(), this.nrows().end());
        assert!(all(row.start <= row.end, row.end <= this.nrows()));
        let nrows =
            unsafe { RowRange::Len::<R>::new_unbound(row.end.unbound() - row.start.unbound()) };

        this.subrows_mut(row.start, nrows)
    }

    #[track_caller]
    #[inline]
    unsafe fn get_unchecked(this: Self, row: RowRange) -> Self::Target {
        let row = row.into_range(R::start(), this.nrows().end());

        debug_assert!(all(row.start <= row.end, row.end <= this.nrows(),));
        let nrows =
            unsafe { RowRange::Len::<R>::new_unbound(row.end.unbound() - row.start.unbound()) };

        this.subrows_mut(row.start, nrows)
    }
}

macro_rules! idx_impl {
    ($R: ty $(, $tt: tt)?) => {
        impl<'a $(, $tt)?, T, Rs: Stride> ColIndex<Idx<$R>> for ColRef<'a, T, $R, Rs> {
            type Target = &'a T;

            #[track_caller]
            #[inline]
            fn get(this: Self, row: Idx<$R>) -> Self::Target {
                this.at(row)
            }

            #[track_caller]
            #[inline]
            unsafe fn get_unchecked(this: Self, row: Idx<$R>) -> Self::Target {
                this.at_unchecked(row)
            }
        }

        impl<'a $(, $tt)?, T, Rs: Stride> ColIndex<Idx<$R>> for ColMut<'a, T, $R, Rs> {
            type Target = &'a mut T;

            #[track_caller]
            #[inline]
            fn get(this: Self, row: Idx<$R>) -> Self::Target {
                this.at_mut(row)
            }

            #[track_caller]
            #[inline]
            unsafe fn get_unchecked(this: Self, row: Idx<$R>) -> Self::Target {
                this.at_mut_unchecked(row)
            }
        }
    };
}

idx_impl!(usize);
idx_impl!(Dim<'N>, 'N);
