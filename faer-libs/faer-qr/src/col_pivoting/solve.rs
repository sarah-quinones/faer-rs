use crate::no_pivoting;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    permutation::{
        permute_rows, permute_rows_in_place, permute_rows_in_place_req, Index, PermutationRef,
    },
    ComplexField, Conj, Entity, MatMut, MatRef, Parallelism,
};
use reborrow::*;

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix in place, given its QR decomposition with column pivoting.
#[inline]
pub fn solve_in_place_req<E: Entity, I: Index>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_any_of([
        no_pivoting::solve::solve_in_place_req::<E>(qr_size, qr_blocksize, rhs_ncols)?,
        permute_rows_in_place_req::<E, I>(qr_size, rhs_ncols)?,
    ])
}

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// the transpose of a matrix in place, given its QR decomposition with column pivoting.
#[inline]
pub fn solve_transpose_in_place_req<E: Entity, I: Index>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_any_of([
        no_pivoting::solve::solve_transpose_in_place_req::<E>(qr_size, qr_blocksize, rhs_ncols)?,
        permute_rows_in_place_req::<E, I>(qr_size, rhs_ncols)?,
    ])
}

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix out of place, given its QR decomposition with column pivoting.
#[inline]
pub fn solve_req<E: Entity, I: Index>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_any_of([
        no_pivoting::solve::solve_req::<E>(qr_size, qr_blocksize, rhs_ncols)?,
        permute_rows_in_place_req::<E, I>(qr_size, rhs_ncols)?,
    ])
}

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// the transpose of a matrix ouf of place, given its QR decomposition with column pivoting.
#[inline]
pub fn solve_transpose_req<E: Entity, I: Index>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_any_of([
        no_pivoting::solve::solve_transpose_req::<E>(qr_size, qr_blocksize, rhs_ncols)?,
        permute_rows_in_place_req::<E, I>(qr_size, rhs_ncols)?,
    ])
}

/// Given the QR factors with column pivoting of a matrix $A$ and a matrix $B$ stored in `rhs`,
/// this function computes the solution of the linear system in the sense of least squares:
/// $$\text{Op}_A(A)X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `rhs`.
///
/// # Panics
///
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `col_perm` doesn't have the same dimension as `qr_factors`.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lu_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see [`solve_in_place_req`]).
#[track_caller]
pub fn solve_in_place<E: ComplexField, I: Index>(
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    col_perm: PermutationRef<'_, I>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let mut rhs = rhs;
    let mut stack = stack;
    no_pivoting::solve::solve_in_place(
        qr_factors,
        householder_factor,
        conj_lhs,
        rhs.rb_mut(),
        parallelism,
        stack.rb_mut(),
    );
    let size = qr_factors.ncols();
    permute_rows_in_place(rhs.subrows(0, size), col_perm.inverse(), stack);
}

/// Given the QR factors with column pivoting of a matrix $A$ and a matrix $B$ stored in `rhs`,
/// this function computes the solution of the linear system in the sense of least squares::
/// $$\text{Op}_A(A)\top X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `rhs`.
///
/// # Panics
///
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `col_perm` doesn't have the same dimension as `qr_factors`.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lu_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see
///   [`solve_transpose_in_place_req`]).
#[track_caller]
pub fn solve_transpose_in_place<E: ComplexField, I: Index>(
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    col_perm: PermutationRef<'_, I>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let mut rhs = rhs;
    let mut stack = stack;
    permute_rows_in_place(rhs.rb_mut(), col_perm, stack.rb_mut());
    no_pivoting::solve::solve_transpose_in_place(
        qr_factors,
        householder_factor,
        conj_lhs,
        rhs.rb_mut(),
        parallelism,
        stack.rb_mut(),
    );
}

/// Given the QR factors with column pivoting of a matrix $A$ and a matrix $B$ stored in `rhs`,
/// this function computes the solution of the linear system:
/// $$\text{Op}_A(A)X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
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
/// - Panics if the provided memory in `stack` is insufficient (see [`solve_req`]).
#[track_caller]
pub fn solve<E: ComplexField, I: Index>(
    dst: MatMut<'_, E>,
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    col_perm: PermutationRef<'_, I>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let mut dst = dst;
    let mut stack = stack;
    no_pivoting::solve::solve(
        dst.rb_mut(),
        qr_factors,
        householder_factor,
        conj_lhs,
        rhs,
        parallelism,
        stack.rb_mut(),
    );
    permute_rows_in_place(dst, col_perm.inverse(), stack);
}

/// Given the QR factors with column pivoting of a matrix $A$ and a matrix $B$ stored in `rhs`,
/// this function computes the solution of the linear system:
/// $$\text{Op}_A(A)^\top X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
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
/// - Panics if the provided memory in `stack` is insufficient (see [`solve_transpose_req`]).
#[track_caller]
pub fn solve_transpose<E: ComplexField, I: Index>(
    dst: MatMut<'_, E>,
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    col_perm: PermutationRef<'_, I>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let mut dst = dst;
    let mut stack = stack;
    permute_rows(dst.rb_mut(), rhs, col_perm);
    no_pivoting::solve::solve_transpose_in_place(
        qr_factors,
        householder_factor,
        conj_lhs,
        dst.rb_mut(),
        parallelism,
        stack.rb_mut(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::col_pivoting::compute::{qr_in_place, qr_in_place_req, recommended_blocksize};
    use assert2::assert as fancy_assert;
    use faer_core::{c32, c64, mul::matmul_with_conj, Mat};
    use rand::random;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    fn test_solve_in_place<E: ComplexField>(mut random: impl FnMut() -> E, epsilon: E::Real) {
        let n = 32;
        let k = 6;

        let a = Mat::from_fn(n, n, |_, _| random());
        let rhs = Mat::from_fn(n, k, |_, _| random());

        let mut qr = a.clone();
        let blocksize = recommended_blocksize::<f64>(n, n);
        let mut householder = Mat::from_fn(blocksize, n, |_, _| E::faer_zero());
        let mut perm = vec![0usize; n];
        let mut perm_inv = vec![0usize; n];

        let (_, perm) = qr_in_place(
            qr.as_mut(),
            householder.as_mut(),
            &mut perm,
            &mut perm_inv,
            Parallelism::None,
            make_stack!(qr_in_place_req::<E, usize>(
                n,
                n,
                blocksize,
                Parallelism::None,
                Default::default(),
            )),
            Default::default(),
        );

        let qr = qr.as_ref();

        for conj_lhs in [Conj::No, Conj::Yes] {
            let mut sol = rhs.clone();
            solve_in_place(
                qr,
                householder.as_ref(),
                perm.rb(),
                conj_lhs,
                sol.as_mut(),
                Parallelism::None,
                make_stack!(solve_in_place_req::<E, usize>(n, blocksize, k)),
            );

            let mut rhs_reconstructed = rhs.clone();
            matmul_with_conj(
                rhs_reconstructed.as_mut(),
                a.as_ref(),
                conj_lhs,
                sol.as_ref(),
                Conj::No,
                None,
                E::faer_one(),
                Parallelism::None,
            );

            for j in 0..k {
                for i in 0..n {
                    fancy_assert!(
                        (rhs_reconstructed.read(i, j).faer_sub(rhs.read(i, j))).faer_abs()
                            < epsilon
                    )
                }
            }
        }
    }

    fn test_solve_transpose_in_place<E: ComplexField>(
        mut random: impl FnMut() -> E,
        epsilon: E::Real,
    ) {
        let n = 32;
        let k = 6;

        let a = Mat::from_fn(n, n, |_, _| random());
        let rhs = Mat::from_fn(n, k, |_, _| random());

        let mut qr = a.clone();
        let blocksize = recommended_blocksize::<f64>(n, n);
        let mut householder = Mat::from_fn(blocksize, n, |_, _| E::faer_zero());
        let mut perm = vec![0usize; n];
        let mut perm_inv = vec![0; n];

        let (_, perm) = qr_in_place(
            qr.as_mut(),
            householder.as_mut(),
            &mut perm,
            &mut perm_inv,
            Parallelism::None,
            make_stack!(qr_in_place_req::<E, usize>(
                n,
                n,
                blocksize,
                Parallelism::None,
                Default::default(),
            )),
            Default::default(),
        );

        let qr = qr.as_ref();

        for conj_lhs in [Conj::No, Conj::Yes] {
            let mut sol = rhs.clone();
            solve_transpose_in_place(
                qr,
                householder.as_ref(),
                perm.rb(),
                conj_lhs,
                sol.as_mut(),
                Parallelism::None,
                make_stack!(solve_transpose_in_place_req::<E, usize>(n, blocksize, k)),
            );

            let mut rhs_reconstructed = rhs.clone();
            matmul_with_conj(
                rhs_reconstructed.as_mut(),
                a.as_ref().transpose(),
                conj_lhs,
                sol.as_ref(),
                Conj::No,
                None,
                E::faer_one(),
                Parallelism::None,
            );

            for j in 0..k {
                for i in 0..n {
                    fancy_assert!(
                        (rhs_reconstructed.read(i, j).faer_sub(rhs.read(i, j))).faer_abs()
                            < epsilon
                    )
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
