use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{solve, zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism};
use reborrow::*;

#[cfg(feature = "std")]
use assert2::assert;

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix in place, given its diagonal LDLT decomposition.
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
/// matrix out of place, given its diagonal LDLT decomposition.
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
/// the transpose of a matrix in place, given its diagonal LDLT decomposition.
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
/// the transpose of a matrix out of place, given its diagonal LDLT decomposition.
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

/// Given the Cholesky factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
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
/// * `cholesky_factors` must be square of dimension `n`.
/// * `rhs` must have `n` rows.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`solve_in_place_req`]).
#[track_caller]
pub fn solve_in_place_with_conj<E: ComplexField>(
    cholesky_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let n = cholesky_factors.nrows();
    let k = rhs.ncols();
    let _ = &stack;

    assert!(cholesky_factors.nrows() == cholesky_factors.ncols());
    assert!(rhs.nrows() == n);

    let mut rhs = rhs;

    solve::solve_unit_lower_triangular_in_place_with_conj(
        cholesky_factors,
        conj_lhs,
        rhs.rb_mut(),
        parallelism,
    );

    for j in 0..k {
        for i in 0..n {
            let d = unsafe { cholesky_factors.read_unchecked(i, i) };
            let rhs_elem = unsafe { rhs.read_unchecked(i, j) };
            unsafe {
                rhs.write_unchecked(i, j, rhs_elem.mul(d.inv()));
            }
        }
    }

    solve::solve_unit_upper_triangular_in_place_with_conj(
        cholesky_factors.transpose(),
        conj_lhs.compose(Conj::Yes),
        rhs.rb_mut(),
        parallelism,
    );
}

/// Given the Cholesky factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
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
/// * `cholesky_factors` must be square of dimension `n`.
/// * `rhs` must have `n` rows.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`solve_transpose_in_place_req`]).
#[track_caller]
pub fn solve_transpose_in_place_with_conj<E: ComplexField>(
    cholesky_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    // (L D L.*).T = conj(L D L.*)
    solve_in_place_with_conj(
        cholesky_factors,
        match conj_lhs {
            Conj::No => Conj::Yes,
            Conj::Yes => Conj::No,
        },
        rhs,
        parallelism,
        stack,
    )
}

/// Given the Cholesky factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
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
/// * `cholesky_factors` must be square of dimension `n`.
/// * `rhs` must have `n` rows.
/// * `dst` must have `n` rows, and the same number of columns as `rhs`.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`solve_transpose_req`]).
#[track_caller]
pub fn solve_transpose_with_conj<E: ComplexField>(
    dst: MatMut<'_, E>,
    cholesky_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let mut dst = dst;
    zipped!(dst.rb_mut(), rhs).for_each(|mut dst, src| dst.write(src.read()));
    solve_transpose_in_place_with_conj(cholesky_factors, conj_lhs, dst, parallelism, stack)
}

/// Given the Cholesky factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
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
/// * `cholesky_factors` must be square of dimension `n`.
/// * `rhs` must have `n` rows.
/// * `dst` must have `n` rows, and the same number of columns as `rhs`.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`solve_req`]).
#[track_caller]
pub fn solve_with_conj<E: ComplexField>(
    dst: MatMut<'_, E>,
    cholesky_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let mut dst = dst;
    zipped!(dst.rb_mut(), rhs).for_each(|mut dst, src| dst.write(src.read()));
    solve_in_place_with_conj(cholesky_factors, conj_lhs, dst, parallelism, stack)
}
