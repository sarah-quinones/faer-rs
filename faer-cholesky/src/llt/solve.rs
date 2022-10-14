use core::ops::{Add, Mul, Neg};

use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{solve, MatMut, MatRef, Zero};
use num_traits::{Inv, One};
use reborrow::*;

use assert2::assert as fancy_assert;

pub fn solve_in_place_req<T: 'static>(
    cholesky_dim: usize,
    rhs_ncols: usize,
    n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    use solve::triangular::*;
    solve_triangular_in_place_req::<T>(cholesky_dim, rhs_ncols, n_threads)
}

#[track_caller]
pub fn solve_in_place<T>(
    cholesky_factors: MatRef<'_, T>,
    rhs: MatMut<'_, T>,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    let n = cholesky_factors.nrows();

    fancy_assert!(cholesky_factors.nrows() == cholesky_factors.ncols());
    fancy_assert!(rhs.nrows() == n);

    let mut rhs = rhs;
    let mut stack = stack;

    solve::triangular::solve_lower_triangular_in_place(
        cholesky_factors,
        rhs.rb_mut(),
        n_threads,
        stack.rb_mut(),
    );

    solve::triangular::solve_upper_triangular_in_place(
        cholesky_factors.transpose(),
        rhs.rb_mut(),
        n_threads,
        stack.rb_mut(),
    );
}
