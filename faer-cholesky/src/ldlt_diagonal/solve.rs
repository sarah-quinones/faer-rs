use faer_core::{solve, ComplexField, Conj, MatMut, MatRef, Parallelism};
use reborrow::*;

use assert2::assert as fancy_assert;

/// Given the Cholesky factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
/// computes the solution of the linear system:
/// $$\text{Op}_A(A)X = \text{Op}_B(B).$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
/// $\text{Op}_B$ is either the identity or the conjugation depending on the value of `conj_rhs`.  
///
/// The solution of the linear system is stored in `rhs`.
#[track_caller]
pub fn solve_in_place<T: ComplexField>(
    cholesky_factors: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    let n = cholesky_factors.nrows();
    let k = rhs.ncols();

    fancy_assert!(cholesky_factors.nrows() == cholesky_factors.ncols());
    fancy_assert!(rhs.nrows() == n);

    let mut rhs = rhs;

    solve::solve_unit_lower_triangular_in_place(
        cholesky_factors,
        conj_lhs,
        rhs.rb_mut(),
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

    solve::solve_unit_upper_triangular_in_place(
        cholesky_factors.transpose(),
        match conj_lhs {
            Conj::No => Conj::Yes,
            Conj::Yes => Conj::No,
        },
        rhs.rb_mut(),
        Conj::No,
        parallelism,
    );
}

/// Given the Cholesky factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
/// computes the solution of the linear system:
/// $$\text{Op}_A(A)^\top X = \text{Op}_B(B).$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
/// $\text{Op}_B$ is either the identity or the conjugation depending on the value of `conj_rhs`.  
///
/// The solution of the linear system is stored in `rhs`.
#[track_caller]
pub fn solve_transpose_in_place<T: ComplexField>(
    cholesky_factors: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    // (L D L.*).T = conj(L D L.*)
    solve_in_place(
        cholesky_factors,
        match conj_lhs {
            Conj::No => Conj::Yes,
            Conj::Yes => Conj::No,
        },
        rhs,
        conj_rhs,
        parallelism,
    )
}
