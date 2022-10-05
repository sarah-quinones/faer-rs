use crate::{MatMut, MatRef};

use assert2::debug_assert as fancy_debug_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use num_traits::{One, Zero};
use reborrow::*;

pub mod triangular {
    use core::ops::{Add, Mul, Neg};

    use super::*;

    pub fn solve_tri_lower_with_implicit_unit_diagonal_in_place_req<T: 'static>(
        max_tril_dim: usize,
        max_rhs_ncols: usize,
        max_n_threads: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n = max_tril_dim;
        let k = max_rhs_ncols;
        crate::backend::mul::mat_x_mat_req::<T>(n - n / 2, n / 2, k, max_n_threads)
    }

    pub unsafe fn solve_tri_lower_with_implicit_unit_diagonal_in_place_unchecked<T>(
        tril: MatRef<'_, T>,
        rhs: MatMut<'_, T>,
        n_threads_hint: usize,
        stack: DynStack<'_>,
    ) where
        T: Zero + One + Clone + Send + Sync + 'static,
        for<'a> &'a T: Add<&'a T, Output = T> + Mul<&'a T, Output = T> + Neg<Output = T>,
    {
        fancy_debug_assert!(tril.nrows() == tril.ncols());
        fancy_debug_assert!(rhs.nrows() == tril.ncols());

        let n = tril.nrows();

        match n {
            0 | 1 => return,
            _ => (),
        }

        let mut stack = stack;

        let (tril_top_left, _, tril_bot_left, tril_bot_right) =
            tril.split_at_unchecked(n / 2, n / 2);
        let (_, mut rhs_top, _, mut rhs_bot) = rhs.split_at_unchecked(n / 2, 0);

        solve_tri_lower_with_implicit_unit_diagonal_in_place_unchecked(
            tril_top_left,
            rhs_top.rb_mut(),
            n_threads_hint,
            stack.rb_mut(),
        );

        crate::backend::mul::mat_x_mat(
            rhs_bot.rb_mut(),
            tril_bot_left,
            rhs_top.into_const(),
            Some(&T::one()),
            &-&T::one(),
            n_threads_hint,
            stack.rb_mut(),
        );

        solve_tri_lower_with_implicit_unit_diagonal_in_place_unchecked(
            tril_bot_right,
            rhs_bot,
            n_threads_hint,
            stack,
        );
    }
}
