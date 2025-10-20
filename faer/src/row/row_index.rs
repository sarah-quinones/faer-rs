use super::*;
use crate::into_range::IntoRange;
use crate::{Idx, IdxInc, assert, debug_assert};

impl<'a, C: Shape, T, Cs: Stride, ColRange: IntoRange<IdxInc<C>, Len<C>: 'a>> RowIndex<ColRange> for RowRef<'a, T, C, Cs> {
	type Target = RowRef<'a, T, ColRange::Len<C>, Cs>;

	#[track_caller]
	#[inline]

	fn get(this: Self, col: ColRange) -> Self::Target {
		let col = col.into_range(C::start(), this.ncols().end());

		assert!(all(col.start <= col.end, col.end <= this.ncols()));

		let ncols = unsafe { ColRange::Len::<C>::new_unbound(col.end.unbound() - col.start.unbound()) };

		this.subcols(col.start, ncols)
	}

	#[track_caller]
	#[inline]

	unsafe fn get_unchecked(this: Self, col: ColRange) -> Self::Target {
		let col = col.into_range(C::start(), this.ncols().end());

		debug_assert!(all(col.start <= col.end, col.end <= this.ncols(),));

		let ncols = unsafe { ColRange::Len::<C>::new_unbound(col.end.unbound() - col.start.unbound()) };

		this.subcols(col.start, ncols)
	}
}

impl<'a, C: Shape, T, Cs: Stride, ColRange: IntoRange<IdxInc<C>, Len<C>: 'a>> RowIndex<ColRange> for RowMut<'a, T, C, Cs> {
	type Target = RowMut<'a, T, ColRange::Len<C>, Cs>;

	#[track_caller]
	#[inline]

	fn get(this: Self, col: ColRange) -> Self::Target {
		let col = col.into_range(C::start(), this.ncols().end());

		assert!(all(col.start <= col.end, col.end <= this.ncols()));

		let ncols = unsafe { ColRange::Len::<C>::new_unbound(col.end.unbound() - col.start.unbound()) };

		this.subcols_mut(col.start, ncols)
	}

	#[track_caller]
	#[inline]

	unsafe fn get_unchecked(this: Self, col: ColRange) -> Self::Target {
		let col = col.into_range(C::start(), this.ncols().end());

		debug_assert!(all(col.start <= col.end, col.end <= this.ncols(),));

		let ncols = unsafe { ColRange::Len::<C>::new_unbound(col.end.unbound() - col.start.unbound()) };

		this.subcols_mut(col.start, ncols)
	}
}

macro_rules! idx_impl {
    ($C:ty $(, $tt:tt)?) => {
        impl <'a $(, $tt)?, T, Cs : Stride > RowIndex < Idx <$C >> for RowRef <'a, T, $C,
        Cs > { type Target = &'a T; #[track_caller] #[inline] fn get(this : Self, col :
        Idx <$C >) -> Self::Target { this.at(col) } #[track_caller] #[inline] unsafe fn
        get_unchecked(this : Self, col : Idx <$C >) -> Self::Target { this
        .at_unchecked(col) } } impl <'a $(, $tt)?, T, Cs : Stride > RowIndex < Idx <$C >>
        for RowMut <'a, T, $C, Cs > { type Target = &'a mut T; #[track_caller] #[inline]
        fn get(this : Self, col : Idx <$C >) -> Self::Target { this.at_mut(col) }
        #[track_caller] #[inline] unsafe fn get_unchecked(this : Self, col : Idx <$C >)
        -> Self::Target { this.at_mut_unchecked(col) } }
    };
}

idx_impl!(usize);

idx_impl!(Dim <'N >, 'N);
