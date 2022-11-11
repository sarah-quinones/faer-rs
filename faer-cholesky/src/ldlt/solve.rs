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
    let k = rhs.ncols();

    fancy_assert!(cholesky_factors.nrows() == cholesky_factors.ncols());
    fancy_assert!(rhs.nrows() == n);

    let mut rhs = rhs;

    solve::triangular::solve_unit_lower_triangular_in_place(
        cholesky_factors,
        rhs.rb_mut(),
        conj_lhs,
        conj_rhs,
        parallelism,
    );

    for j in 0..k {
        for i in 0..n {
            let d = *unsafe { cholesky_factors.get_unchecked(i, i) };
            let rhs = unsafe { rhs.rb_mut().get_unchecked(i, j) };
            *rhs = *rhs / d;
        }
    }

    solve::triangular::solve_unit_upper_triangular_in_place(
        cholesky_factors.transpose(),
        rhs.rb_mut(),
        !conj_lhs,
        false,
        parallelism,
    );
}
