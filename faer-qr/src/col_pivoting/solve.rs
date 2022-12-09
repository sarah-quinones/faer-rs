use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    permutation::{permute_rows, permute_rows_in_place, permute_rows_in_place_req, PermutationRef},
    ComplexField, Conj, MatMut, MatRef, Parallelism,
};
use reborrow::*;

use crate::no_pivoting;

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix in place, given its QR decomposition with column pivoting.
#[inline]
pub fn solve_in_place_req<T: 'static>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_any_of([
        no_pivoting::solve::solve_in_place_req::<T>(qr_size, qr_blocksize, rhs_ncols)?,
        permute_rows_in_place_req::<T>(qr_size, rhs_ncols)?,
    ])
}

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// the transpose of a matrix in place, given its QR decomposition with column pivoting.
#[inline]
pub fn solve_transpose_in_place_req<T: 'static>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_any_of([
        no_pivoting::solve::solve_transpose_in_place_req::<T>(qr_size, qr_blocksize, rhs_ncols)?,
        permute_rows_in_place_req::<T>(qr_size, rhs_ncols)?,
    ])
}

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix out of place, given its QR decomposition with column pivoting.
#[inline]
pub fn solve_req<T: 'static>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_any_of([
        no_pivoting::solve::solve_req::<T>(qr_size, qr_blocksize, rhs_ncols)?,
        permute_rows_in_place_req::<T>(qr_size, rhs_ncols)?,
    ])
}

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// the transpose of a matrix ouf of place, given its QR decomposition with column pivoting.
#[inline]
pub fn solve_transpose_req<T: 'static>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_any_of([
        no_pivoting::solve::solve_transpose_req::<T>(qr_size, qr_blocksize, rhs_ncols)?,
        permute_rows_in_place_req::<T>(qr_size, rhs_ncols)?,
    ])
}

/// Given the QR factors with column pivoting of a matrix $A$ and a matrix $B$ stored in `rhs`,
/// this function computes the solution of the linear system:
/// $$\text{Op}_A(A)X = \text{Op}_B(B).$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
/// $\text{Op}_B$ is either the identity or the conjugation depending on the value of `conj_rhs`.  
///
/// The solution of the linear system is stored in `rhs`.
///
/// # Panics
///
/// - Panics if `qr_factors` is not a square matrix.
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `col_perm` doesn't have the same dimension as `qr_factors`.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lu_factors`.
/// - Panics if the provided memory in `stack` is insufficient.
#[track_caller]
pub fn solve_in_place<T: ComplexField>(
    qr_factors: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    col_perm: PermutationRef<'_>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut rhs = rhs;
    let mut stack = stack;
    no_pivoting::solve::solve_in_place(
        qr_factors,
        householder_factor,
        conj_lhs,
        rhs.rb_mut(),
        conj_rhs,
        parallelism,
        stack.rb_mut(),
    );
    permute_rows_in_place(rhs, col_perm.inverse(), stack);
}

/// Given the QR factors with column pivoting of a matrix $A$ and a matrix $B$ stored in `rhs`,
/// this function computes the solution of the linear system:
/// $$\text{Op}_A(A)\top X = \text{Op}_B(B).$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
/// $\text{Op}_B$ is either the identity or the conjugation depending on the value of `conj_rhs`.  
///
/// The solution of the linear system is stored in `rhs`.
///
/// # Panics
///
/// - Panics if `qr_factors` is not a square matrix.
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `col_perm` doesn't have the same dimension as `qr_factors`.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lu_factors`.
/// - Panics if the provided memory in `stack` is insufficient.
#[track_caller]
pub fn solve_transpose_in_place<T: ComplexField>(
    qr_factors: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    col_perm: PermutationRef<'_>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut rhs = rhs;
    let mut stack = stack;
    permute_rows_in_place(rhs.rb_mut(), col_perm, stack.rb_mut());
    no_pivoting::solve::solve_transpose_in_place(
        qr_factors,
        householder_factor,
        conj_lhs,
        rhs.rb_mut(),
        conj_rhs,
        parallelism,
        stack.rb_mut(),
    );
}

/// Given the QR factors with column pivoting of a matrix $A$ and a matrix $B$ stored in `rhs`,
/// this function computes the solution of the linear system:
/// $$\text{Op}_A(A)X = \text{Op}_B(B).$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
/// $\text{Op}_B$ is either the identity or the conjugation depending on the value of `conj_rhs`.  
///
/// The solution of the linear system is stored in `dst`.
///
/// # Panics
///
/// - Panics if `qr_factors` is not a square matrix.
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `col_perm` doesn't have the same dimension as `qr_factors`.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lu_factors`.
/// - Panics if `rhs` and `dst` don't have the same shape.
/// - Panics if the provided memory in `stack` is insufficient.
#[track_caller]
pub fn solve<T: ComplexField>(
    dst: MatMut<'_, T>,
    qr_factors: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    col_perm: PermutationRef<'_>,
    conj_lhs: Conj,
    rhs: MatRef<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut dst = dst;
    let mut stack = stack;
    no_pivoting::solve::solve(
        dst.rb_mut(),
        qr_factors,
        householder_factor,
        conj_lhs,
        rhs,
        conj_rhs,
        parallelism,
        stack.rb_mut(),
    );
    permute_rows_in_place(dst, col_perm.inverse(), stack);
}

/// Given the QR factors with column pivoting of a matrix $A$ and a matrix $B$ stored in `rhs`,
/// this function computes the solution of the linear system:
/// $$\text{Op}_A(A)^\top X = \text{Op}_B(B).$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
/// $\text{Op}_B$ is either the identity or the conjugation depending on the value of `conj_rhs`.  
///
/// The solution of the linear system is stored in `dst`.
///
/// # Panics
///
/// - Panics if `qr_factors` is not a square matrix.
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `col_perm` doesn't have the same dimension as `qr_factors`.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lu_factors`.
/// - Panics if `rhs` and `dst` don't have the same shape.
/// - Panics if the provided memory in `stack` is insufficient.
#[track_caller]
pub fn solve_transpose<T: ComplexField>(
    dst: MatMut<'_, T>,
    qr_factors: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    col_perm: PermutationRef<'_>,
    conj_lhs: Conj,
    rhs: MatRef<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut dst = dst;
    let mut stack = stack;
    permute_rows(dst.rb_mut(), rhs, col_perm);
    no_pivoting::solve::solve_transpose_in_place(
        qr_factors,
        householder_factor,
        conj_lhs,
        dst.rb_mut(),
        conj_rhs,
        parallelism,
        stack.rb_mut(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::col_pivoting::compute::{qr_in_place, qr_in_place_req, recommended_blocksize};
    use assert2::assert as fancy_assert;
    use faer_core::{c32, c64, mul::matmul, Mat};
    use rand::random;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req))
        };
    }

    fn test_solve_in_place<T: ComplexField>(mut random: impl FnMut() -> T, epsilon: T::Real) {
        let n = 32;
        let k = 6;

        let a = Mat::with_dims(|_, _| random(), n, n);
        let rhs = Mat::with_dims(|_, _| random(), n, k);

        let mut qr = a.clone();
        let blocksize = recommended_blocksize::<f64>(n, n);
        let mut householder = Mat::with_dims(|_, _| T::zero(), blocksize, n);
        let mut perm = vec![0; n];
        let mut perm_inv = vec![0; n];

        let (_, perm) = qr_in_place(
            qr.as_mut(),
            householder.as_mut(),
            &mut perm,
            &mut perm_inv,
            Parallelism::None,
            make_stack!(qr_in_place_req::<T>(n, n, blocksize, Parallelism::None).unwrap()),
            Default::default(),
        );

        let qr = qr.as_ref();

        for conj_lhs in [Conj::No, Conj::Yes] {
            for conj_rhs in [Conj::No, Conj::Yes] {
                let mut sol = rhs.clone();
                solve_in_place(
                    qr,
                    householder.as_ref(),
                    perm.rb(),
                    conj_lhs,
                    sol.as_mut(),
                    conj_rhs,
                    Parallelism::None,
                    make_stack!(solve_in_place_req::<T>(n, blocksize, k).unwrap()),
                );

                let mut rhs_reconstructed = rhs.clone();
                matmul(
                    rhs_reconstructed.as_mut(),
                    Conj::No,
                    a.as_ref(),
                    conj_lhs,
                    sol.as_ref(),
                    Conj::No,
                    None,
                    T::one(),
                    Parallelism::None,
                );

                for j in 0..k {
                    for i in 0..n {
                        let target = match conj_rhs {
                            Conj::No => rhs[(i, j)],
                            Conj::Yes => rhs[(i, j)].conj(),
                        };

                        fancy_assert!((rhs_reconstructed[(i, j)] - target).abs() < epsilon)
                    }
                }
            }
        }
    }

    fn test_solve_transpose_in_place<T: ComplexField>(
        mut random: impl FnMut() -> T,
        epsilon: T::Real,
    ) {
        let n = 32;
        let k = 6;

        let a = Mat::with_dims(|_, _| random(), n, n);
        let rhs = Mat::with_dims(|_, _| random(), n, k);

        let mut qr = a.clone();
        let blocksize = recommended_blocksize::<f64>(n, n);
        let mut householder = Mat::with_dims(|_, _| T::zero(), blocksize, n);
        let mut perm = vec![0; n];
        let mut perm_inv = vec![0; n];

        let (_, perm) = qr_in_place(
            qr.as_mut(),
            householder.as_mut(),
            &mut perm,
            &mut perm_inv,
            Parallelism::None,
            make_stack!(qr_in_place_req::<T>(n, n, blocksize, Parallelism::None).unwrap()),
            Default::default(),
        );

        let qr = qr.as_ref();

        for conj_lhs in [Conj::No, Conj::Yes] {
            for conj_rhs in [Conj::No, Conj::Yes] {
                let mut sol = rhs.clone();
                solve_transpose_in_place(
                    qr,
                    householder.as_ref(),
                    perm.rb(),
                    conj_lhs,
                    sol.as_mut(),
                    conj_rhs,
                    Parallelism::None,
                    make_stack!(solve_transpose_in_place_req::<T>(n, blocksize, k).unwrap()),
                );

                let mut rhs_reconstructed = rhs.clone();
                matmul(
                    rhs_reconstructed.as_mut(),
                    Conj::No,
                    a.as_ref().transpose(),
                    conj_lhs,
                    sol.as_ref(),
                    Conj::No,
                    None,
                    T::one(),
                    Parallelism::None,
                );

                for j in 0..k {
                    for i in 0..n {
                        let target = match conj_rhs {
                            Conj::No => rhs[(i, j)],
                            Conj::Yes => rhs[(i, j)].conj(),
                        };

                        fancy_assert!((rhs_reconstructed[(i, j)] - target).abs() < epsilon)
                    }
                }
            }
        }
    }

    #[test]
    fn test_solve_in_place_f64() {
        test_solve_in_place(random::<f64>, 1e-6);
    }
    #[test]
    fn test_solve_in_place_f32() {
        test_solve_in_place(random::<f32>, 1e-1);
    }

    #[test]
    fn test_solve_in_place_c64() {
        test_solve_in_place(|| c64::new(random(), random()), 1e-6);
    }

    #[test]
    fn test_solve_in_place_c32() {
        test_solve_in_place(|| c32::new(random(), random()), 1e-1);
    }

    #[test]
    fn test_solve_transpose_in_place_f64() {
        test_solve_transpose_in_place(random::<f64>, 1e-6);
    }

    #[test]
    fn test_solve_transpose_in_place_f32() {
        test_solve_transpose_in_place(random::<f32>, 1e-1);
    }

    #[test]
    fn test_solve_transpose_in_place_c64() {
        test_solve_transpose_in_place(|| c64::new(random(), random()), 1e-6);
    }

    #[test]
    fn test_solve_transpose_in_place_c32() {
        test_solve_transpose_in_place(|| c32::new(random(), random()), 1e-1);
    }
}
