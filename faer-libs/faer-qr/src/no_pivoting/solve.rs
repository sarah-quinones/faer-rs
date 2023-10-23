#[cfg(feature = "std")]
use assert2::assert;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    householder::{
        apply_block_householder_sequence_on_the_left_in_place_with_conj,
        apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
    },
    solve, temp_mat_req, zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism,
};
use reborrow::*;

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix in place, given its QR decomposition.
#[inline]
pub fn solve_in_place_req<E: Entity>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = qr_size;
    temp_mat_req::<E>(qr_blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// the transpose of a matrix in place, given its QR decomposition.
#[inline]
pub fn solve_transpose_in_place_req<E: Entity>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = qr_size;
    temp_mat_req::<E>(qr_blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for solving a linear system defined by a
/// matrix out of place, given its QR decomposition.
#[inline]
pub fn solve_req<E: Entity>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = qr_size;
    temp_mat_req::<E>(qr_blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for solving a linear system defined by
/// the transpose of a matrix out of place, given its QR decomposition.
#[inline]
pub fn solve_transpose_req<E: Entity>(
    qr_size: usize,
    qr_blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = qr_size;
    temp_mat_req::<E>(qr_blocksize, rhs_ncols)
}

/// Given the QR factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function computes
/// the solution of the linear system in the sense of least squares:
/// $$\text{Op}_A(A)X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `rhs`.
///
/// # Panics
///
/// - Panics if `qr_factors` is not a tall matrix.
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `rhs` doesn't have the same number of rows as the number of columns of `qr_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see [`solve_in_place_req`]).
#[track_caller]
pub fn solve_in_place<E: ComplexField>(
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    // conjᵃ(H₀ × ... × Hₖ₋₁ × R) X = conjᵇ(B)
    // X = conjᵃ(R)⁻¹ × conjᵃ(Hₖ₋₁) × ... × conjᵃ(H₀) × conjᵇ(B)
    let m = qr_factors.nrows();
    let n = qr_factors.ncols();
    let size = Ord::min(m, n);
    let blocksize = householder_factor.nrows();
    assert!(qr_factors.nrows() >= qr_factors.ncols());
    assert!((householder_factor.nrows(), householder_factor.ncols()) == (blocksize, size));
    assert!(rhs.nrows() == qr_factors.nrows());

    let mut rhs = rhs;
    let mut stack = stack;
    apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        qr_factors,
        householder_factor,
        conj_lhs.compose(Conj::Yes),
        rhs.rb_mut(),
        parallelism,
        stack.rb_mut(),
    );

    solve::solve_upper_triangular_in_place_with_conj(
        qr_factors.submatrix(0, 0, size, size),
        conj_lhs,
        rhs.subrows(0, size),
        parallelism,
    );
}

/// Given the QR factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function computes
/// the solution of the linear system:
/// $$\text{Op}_A(A)^\top X = B.$$
///
/// $\text{Op}_A$ is either the identity or the conjugation depending on the value of `conj_lhs`.  
///
/// The solution of the linear system is stored in `rhs`.
///
/// # Panics
///
/// - Panics if `qr_factors` is not a square matrix.
/// - Panics if the number of columns of `householder_factor` isn't the same as the minimum of the
/// number of rows and the number of columns of `qr_factors`.
/// - Panics if the block size is zero.
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `qr_factors`.
/// - Panics if the provided memory in `stack` is insufficient (see
///   [`solve_transpose_in_place_req`]).
#[track_caller]
pub fn solve_transpose_in_place<E: ComplexField>(
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    // conjᵃ(H₀ × ... × Hₖ₋₁ × R)ᵀ X = conjᵇ(B)
    // conjᵃ(Rᵀ × Hₖ₋₁ᵀ × ... × H₀ᵀ) X = conjᵇ(B)
    // X = conj(conjᵃ(H₀)) × ... × conj(conjᵃ(Hₖ₋₁)) × (conjᵃ(R)ᵀ)⁻¹ × conjᵇ(B)
    assert!(qr_factors.nrows() == qr_factors.ncols());
    let size = qr_factors.nrows();
    let blocksize = householder_factor.nrows();
    assert!((householder_factor.nrows(), householder_factor.ncols()) == (blocksize, size));
    assert!(rhs.nrows() == qr_factors.nrows());

    let mut rhs = rhs;
    let mut stack = stack;

    solve::solve_lower_triangular_in_place_with_conj(
        qr_factors.transpose(),
        conj_lhs,
        rhs.rb_mut(),
        parallelism,
    );
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        qr_factors,
        householder_factor,
        conj_lhs.compose(Conj::Yes),
        rhs.rb_mut(),
        parallelism,
        stack.rb_mut(),
    );
}

/// Given the QR factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function computes
/// the solution of the linear system:
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
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `qr_factors`.
/// - Panics if `rhs` and `dst` don't have the same shape.
/// - Panics if the provided memory in `stack` is insufficient (see [`solve_req`]).
#[track_caller]
pub fn solve<E: ComplexField>(
    dst: MatMut<'_, E>,
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let mut dst = dst;
    zipped!(dst.rb_mut(), rhs).for_each(|mut dst, src| dst.write(src.read()));
    solve_in_place(
        qr_factors,
        householder_factor,
        conj_lhs,
        dst,
        parallelism,
        stack,
    );
}

/// Given the QR factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this function computes
/// the solution of the linear system:
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
/// - Panics if `rhs` doesn't have the same number of rows as the dimension of `qr_factors`.
/// - Panics if `rhs` and `dst` don't have the same shape.
/// - Panics if the provided memory in `stack` is insufficient (see [`solve_transpose_req`]).
#[track_caller]
pub fn solve_transpose<E: ComplexField>(
    dst: MatMut<'_, E>,
    qr_factors: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let mut dst = dst;
    zipped!(dst.rb_mut(), rhs).for_each(|mut dst, src| dst.write(src.read()));
    solve_transpose_in_place(
        qr_factors,
        householder_factor,
        conj_lhs,
        dst,
        parallelism,
        stack,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::no_pivoting::compute::{qr_in_place, qr_in_place_req, recommended_blocksize};
    use assert2::assert;
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
        qr_in_place(
            qr.as_mut(),
            householder.as_mut(),
            Parallelism::None,
            make_stack!(qr_in_place_req::<E>(
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
                conj_lhs,
                sol.as_mut(),
                Parallelism::None,
                make_stack!(solve_in_place_req::<E>(n, blocksize, k)),
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
                    assert!(
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
        qr_in_place(
            qr.as_mut(),
            householder.as_mut(),
            Parallelism::None,
            make_stack!(qr_in_place_req::<E>(
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
                conj_lhs,
                sol.as_mut(),
                Parallelism::None,
                make_stack!(solve_transpose_in_place_req::<E>(n, blocksize, k)),
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
                    assert!(
                        (rhs_reconstructed.read(i, j).faer_sub(rhs.read(i, j))).faer_abs()
                            < epsilon
                    )
                }
            }
        }
    }

    fn random_c64() -> c64 {
        c64 {
            re: random(),
            im: random(),
        }
    }
    fn random_c32() -> c32 {
        c32 {
            re: random(),
            im: random(),
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
        test_solve_in_place(random_c64, 1e-6);
    }

    #[test]
    fn test_solve_in_place_c32() {
        test_solve_in_place(random_c32, 1e-1);
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
        test_solve_transpose_in_place(random_c64, 1e-6);
    }

    #[test]
    fn test_solve_transpose_in_place_c32() {
        test_solve_transpose_in_place(random_c32, 1e-1);
    }
}
