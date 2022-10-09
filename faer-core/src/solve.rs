use crate::{MatMut, MatRef};

use assert2::assert as fancy_assert;
use assert2::debug_assert as fancy_debug_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use num_traits::{One, Zero};
use reborrow::*;

pub mod triangular {
    use core::ops::{Add, Mul, Neg};

    use super::*;

    pub fn solve_unit_lower_triangular_in_place_req<T: 'static>(
        dim: usize,
        rhs_ncols: usize,
        n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n = dim;
        let k = rhs_ncols;
        if n <= 1 {
            return Ok(StackReq::default());
        }
        StackReq::try_any_of([
            solve_unit_lower_triangular_in_place_req::<T>(dim / 2, rhs_ncols, n_threads)?,
            crate::mul::matmul_req::<T>(n - n / 2, n / 2, k, n_threads)?,
            solve_unit_lower_triangular_in_place_req::<T>(dim - dim / 2, rhs_ncols, n_threads)?,
        ])
    }

    pub fn solve_unit_upper_triangular_in_place_req<T: 'static>(
        dim: usize,
        rhs_ncols: usize,
        n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n = dim;
        let k = rhs_ncols;
        if n <= 1 {
            return Ok(StackReq::default());
        }
        StackReq::try_any_of([
            solve_unit_upper_triangular_in_place_req::<T>(dim / 2, rhs_ncols, n_threads)?,
            crate::mul::matmul_req::<T>(n - n / 2, n / 2, k, n_threads)?,
            solve_unit_upper_triangular_in_place_req::<T>(dim - dim / 2, rhs_ncols, n_threads)?,
        ])
    }

    #[track_caller]
    #[inline]
    pub fn solve_unit_lower_triangular_in_place<T>(
        triangular_lower: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        fancy_assert!(triangular_lower.nrows() == triangular_lower.ncols());
        fancy_assert!(rhs.nrows() == triangular_lower.ncols());

        unsafe {
            solve_unit_lower_triangular_in_place_unchecked(triangular_lower, rhs, n_threads, stack);
        }
    }

    #[track_caller]
    #[inline]
    pub fn solve_unit_upper_triangular_in_place<T>(
        triangular_upper: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        fancy_assert!(triangular_upper.nrows() == triangular_upper.ncols());
        fancy_assert!(rhs.nrows() == triangular_upper.ncols());

        unsafe {
            solve_unit_upper_triangular_in_place_unchecked(triangular_upper, rhs, n_threads, stack);
        }
    }

    pub unsafe fn solve_unit_lower_triangular_in_place_unchecked<T>(
        tril: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        fancy_debug_assert!(tril.nrows() == tril.ncols());
        fancy_debug_assert!(rhs.nrows() == tril.ncols());

        let n = tril.nrows();

        match n {
            0 | 1 => return,
            _ => (),
        }

        let mut stack = stack;
        let bs = n / 2;

        let (tril_top_left, _, tril_bot_left, tril_bot_right) = tril.split_at_unchecked(bs, bs);
        let (_, mut rhs_top, _, mut rhs_bot) = rhs.split_at_unchecked(bs, 0);

        solve_unit_lower_triangular_in_place_unchecked(
            tril_top_left,
            rhs_top.rb_mut(),
            n_threads,
            stack.rb_mut(),
        );

        crate::mul::matmul(
            rhs_bot.rb_mut(),
            tril_bot_left,
            rhs_top.into_const(),
            Some(&T::one()),
            &-&T::one(),
            n_threads,
            stack.rb_mut(),
        );

        solve_unit_lower_triangular_in_place_unchecked(tril_bot_right, rhs_bot, n_threads, stack);
    }

    pub(crate) unsafe fn solve_unit_upper_triangular_in_place_unchecked<T>(
        triu: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        fancy_debug_assert!(triu.nrows() == triu.ncols());
        fancy_debug_assert!(rhs.nrows() == triu.ncols());

        let n = triu.nrows();

        match n {
            0 | 1 => return,
            _ => (),
        }

        let mut stack = stack;
        let bs = n - n / 2;

        let (triu_top_left, triu_top_right, _, triu_bot_right) = triu.split_at_unchecked(bs, bs);
        let (_, mut rhs_top, _, mut rhs_bot) = rhs.split_at_unchecked(bs, 0);

        solve_unit_upper_triangular_in_place_unchecked(
            triu_bot_right,
            rhs_bot.rb_mut(),
            n_threads,
            stack.rb_mut(),
        );

        crate::mul::matmul(
            rhs_top.rb_mut(),
            triu_top_right,
            rhs_bot.into_const(),
            Some(&T::one()),
            &-&T::one(),
            n_threads,
            stack.rb_mut(),
        );

        solve_unit_upper_triangular_in_place_unchecked(triu_top_left, rhs_top, n_threads, stack);
    }
}
