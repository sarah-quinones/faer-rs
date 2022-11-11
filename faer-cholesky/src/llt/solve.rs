use faer_core::{solve, ComplexField, MatMut, MatRef, Parallelism};
use reborrow::*;

use assert2::assert as fancy_assert;

#[track_caller]
pub fn solve_in_place<T: ComplexField>(
    cholesky_factors: MatRef<'_, T>,
    rhs: MatMut<'_, T>,
    conj_lhs: bool,
    conj_rhs: bool,
    parallelism: Parallelism,
) {
    let n = cholesky_factors.nrows();

    fancy_assert!(cholesky_factors.nrows() == cholesky_factors.ncols());
    fancy_assert!(rhs.nrows() == n);

    let mut rhs = rhs;

    solve::triangular::solve_lower_triangular_in_place(
        cholesky_factors,
        rhs.rb_mut(),
        conj_lhs,
        conj_rhs,
        parallelism,
    );

    solve::triangular::solve_upper_triangular_in_place(
        cholesky_factors.transpose(),
        rhs.rb_mut(),
        !conj_lhs,
        false,
        parallelism,
    );
}
