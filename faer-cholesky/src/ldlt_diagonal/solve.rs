use dyn_stack::DynStack;
use faer_core::{solve, zipped, ComplexField, Conj, MatMut, MatRef, Parallelism};
use reborrow::*;

use assert2::assert;

/// Given the Cholesky factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
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
            let d = unsafe { cholesky_factors.read_unchecked(i, i).clone() };
            let rhs_elem = unsafe { rhs.read_unchecked(i, j) };
            unsafe {
                rhs.write_unchecked(i, j, rhs_elem.mul(&d.inv()));
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
#[track_caller]
pub fn solve_transpose_in_place_with_conj<E: ComplexField>(
    cholesky_factors: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
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

/// Given the Cholesky factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function
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
