use assert2::assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{solve, zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism};
use reborrow::*;

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix in place, given its Cholesky decomposition.
pub fn solve_in_place_req<E: Entity>(
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
pub fn solve_req<E: Entity>(
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
pub fn solve_transpose_in_place_req<E: Entity>(
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
pub fn solve_transpose_req<E: Entity>(
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
/// $$\text{Op}_A(A)X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `rhs`.
#[track_caller]
pub fn solve_in_place_with_conj<E: ComplexField>(
    cholesky_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let _ = &stack;
    let n = cholesky_factors.nrows();

    assert!(cholesky_factors.nrows() == cholesky_factors.ncols());
    assert!(rhs.nrows() == n);

    let mut rhs = rhs;

    solve::solve_lower_triangular_in_place_with_conj(
        cholesky_factors,
        conj_lhs,
        rhs.rb_mut(),
        parallelism,
    );

    solve::solve_upper_triangular_in_place_with_conj(
        cholesky_factors.transpose(),
        conj_lhs.compose(Conj::Yes),
        rhs.rb_mut(),
        parallelism,
    );
}

/// Given the Cholesky factor of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
/// computes the solution of the linear system:
/// $$\text{Op}_A(A)X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `dst`.
#[track_caller]
pub fn solve_with_conj<E: ComplexField>(
    dst: MatMut<'_, E>,
    cholesky_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut dst = dst;
    zipped!(dst.rb_mut(), rhs).for_each(|mut dst, src| dst.write(src.read()));
    solve_in_place_with_conj(cholesky_factors, conj_lhs, dst, parallelism, stack)
}

/// Given the Cholesky factor of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
/// computes the solution of the linear system:
/// $$\text{Op}_A(A)^\top X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `rhs`.
#[track_caller]
pub fn solve_transpose_in_place_with_conj<E: ComplexField>(
    cholesky_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    // (L L.*).T = conj(L L.*)
    solve_in_place_with_conj(
        cholesky_factors,
        conj_lhs.compose(Conj::Yes),
        rhs,
        parallelism,
        stack,
    )
}

/// Given the Cholesky factor of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
/// computes the solution of the linear system:
/// $$\text{Op}_A(A)^\top X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `dst`.
#[track_caller]
pub fn solve_transpose_with_conj<E: ComplexField>(
    dst: MatMut<'_, E>,
    cholesky_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut dst = dst;
    zipped!(dst.rb_mut(), rhs).for_each(|mut dst, src| dst.write(src.read()));
    solve_transpose_in_place_with_conj(cholesky_factors, conj_lhs, dst, parallelism, stack)
}
