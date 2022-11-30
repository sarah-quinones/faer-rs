use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{solve, ComplexField, Conj, MatMut, MatRef, Parallelism};
use reborrow::*;

use assert2::assert as fancy_assert;

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix in place, given its Cholesky decomposition.
pub fn solve_in_place_req<T: 'static>(
    cholesky_dimension: usize,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = cholesky_dimension;
    let _ = rhs_ncols;
    let _ = parallelism;
    Ok(StackReq::default())
}

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix out of place, given its Cholesky decomposition.
pub fn solve_to_req<T: 'static>(
    cholesky_dimension: usize,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = cholesky_dimension;
    let _ = rhs_ncols;
    let _ = parallelism;
    Ok(StackReq::default())
}

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// the transpose of a matrix in place, given its Cholesky decomposition.
pub fn solve_transpose_in_place_req<T: 'static>(
    cholesky_dimension: usize,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = cholesky_dimension;
    let _ = rhs_ncols;
    let _ = parallelism;
    Ok(StackReq::default())
}

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// the transpose of a matrix out of place, given its Cholesky decomposition.
pub fn solve_transpose_to_req<T: 'static>(
    cholesky_dimension: usize,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = cholesky_dimension;
    let _ = rhs_ncols;
    let _ = parallelism;
    Ok(StackReq::default())
}

/// Given the Cholesky factor of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
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
    stack: DynStack<'_>,
) {
    let _ = &stack;
    let n = cholesky_factors.nrows();

    fancy_assert!(cholesky_factors.nrows() == cholesky_factors.ncols());
    fancy_assert!(rhs.nrows() == n);

    let mut rhs = rhs;

    solve::solve_lower_triangular_in_place(
        cholesky_factors,
        conj_lhs,
        rhs.rb_mut(),
        conj_rhs,
        parallelism,
    );

    solve::solve_upper_triangular_in_place(
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

/// Given the Cholesky factor of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
/// computes the solution of the linear system:
/// $$\text{Op}_A(A)X = \text{Op}_B(B).$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
/// $\text{Op}_B$ is either the identity or the conjugation depending on the value of `conj_rhs`.  
///
/// The solution of the linear system is stored in `dst`.
#[track_caller]
pub fn solve_to<T: ComplexField>(
    dst: MatMut<'_, T>,
    cholesky_factors: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatRef<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut dst = dst;
    dst.rb_mut()
        .cwise()
        .zip(rhs)
        .for_each(|dst, src| *dst = *src);
    solve_in_place(
        cholesky_factors,
        conj_lhs,
        dst,
        conj_rhs,
        parallelism,
        stack,
    )
}

/// Given the Cholesky factor of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
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
    stack: DynStack<'_>,
) {
    // (L L.*).T = conj(L L.*)
    solve_in_place(
        cholesky_factors,
        match conj_lhs {
            Conj::No => Conj::Yes,
            Conj::Yes => Conj::No,
        },
        rhs,
        conj_rhs,
        parallelism,
        stack,
    )
}

/// Given the Cholesky factor of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
/// computes the solution of the linear system:
/// $$\text{Op}_A(A)^\top X = \text{Op}_B(B).$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
/// $\text{Op}_B$ is either the identity or the conjugation depending on the value of `conj_rhs`.  
///
/// The solution of the linear system is stored in `dst`.
#[track_caller]
pub fn solve_transpose_to<T: ComplexField>(
    dst: MatMut<'_, T>,
    cholesky_factors: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatRef<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut dst = dst;
    dst.rb_mut()
        .cwise()
        .zip(rhs)
        .for_each(|dst, src| *dst = *src);
    solve_transpose_in_place(
        cholesky_factors,
        conj_lhs,
        dst,
        conj_rhs,
        parallelism,
        stack,
    )
}
