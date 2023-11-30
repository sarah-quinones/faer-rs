use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    assert, solve, unzipped, zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism,
};
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
///
/// # Panics
///
/// Panics if any of these conditions is violated:
///
/// * `cholesky_factor` must be square of dimension `n`.
/// * `rhs` must have `n` rows.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`solve_in_place_req`]).
#[track_caller]
pub fn solve_in_place_with_conj<E: ComplexField>(
    cholesky_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let _ = &stack;
    let n = cholesky_factor.nrows();

    assert!(all(
        cholesky_factor.nrows() == cholesky_factor.ncols(),
        rhs.nrows() == n,
    ));

    let mut rhs = rhs;

    solve::solve_lower_triangular_in_place_with_conj(
        cholesky_factor,
        conj_lhs,
        rhs.rb_mut(),
        parallelism,
    );

    solve::solve_upper_triangular_in_place_with_conj(
        cholesky_factor.transpose(),
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
///
/// # Panics
///
/// Panics if any of these conditions is violated:
///
/// * `cholesky_factor` must be square of dimension `n`.
/// * `rhs` must have `n` rows.
/// * `dst` must have `n` rows, and the same number of columns as `rhs`.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`solve_req`]).
#[track_caller]
pub fn solve_with_conj<E: ComplexField>(
    dst: MatMut<'_, E>,
    cholesky_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let mut dst = dst;
    zipped!(dst.rb_mut(), rhs).for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
    solve_in_place_with_conj(cholesky_factor, conj_lhs, dst, parallelism, stack)
}

/// Given the Cholesky factor of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
/// computes the solution of the linear system:
/// $$\text{Op}_A(A)^\top X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `rhs`.
///
/// # Panics
///
/// Panics if any of these conditions is violated:
///
/// * `cholesky_factor` must be square of dimension `n`.
/// * `rhs` must have `n` rows.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`solve_transpose_in_place_req`]).
#[track_caller]
pub fn solve_transpose_in_place_with_conj<E: ComplexField>(
    cholesky_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    // (L L.*).T = conj(L L.*)
    solve_in_place_with_conj(
        cholesky_factor,
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
///
/// # Panics
///
/// Panics if any of these conditions is violated:
///
/// * `cholesky_factor` must be square of dimension `n`.
/// * `rhs` must have `n` rows.
/// * `dst` must have `n` rows, and the same number of columns as `rhs`.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`solve_transpose_req`]).
#[track_caller]
pub fn solve_transpose_with_conj<E: ComplexField>(
    dst: MatMut<'_, E>,
    cholesky_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let mut dst = dst;
    zipped!(dst.rb_mut(), rhs).for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
    solve_transpose_in_place_with_conj(cholesky_factor, conj_lhs, dst, parallelism, stack)
}
