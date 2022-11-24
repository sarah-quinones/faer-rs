use faer_core::{
    mul::triangular::{self, BlockStructure},
    ComplexField, Conj, MatMut, MatRef, Parallelism,
};

/// Computes the lower triangular part of the reconstructed matrix, given its Cholesky
/// decomposition, and stores the result in `dst`.
///
/// # Panics
///
/// - Panics if `cholesky_factor` is not a square matrix.
/// - Panics if the destination shape doesn't match the shape of the matrix.
#[track_caller]
pub fn reconstruct_lower_to<T: ComplexField>(
    dst: MatMut<'_, T>,
    cholesky_factor: MatRef<'_, T>,
    parallelism: Parallelism,
) {
    triangular::matmul(
        dst,
        BlockStructure::TriangularLower,
        Conj::No,
        cholesky_factor,
        BlockStructure::TriangularLower,
        Conj::No,
        cholesky_factor.transpose(),
        BlockStructure::TriangularUpper,
        Conj::Yes,
        None,
        T::one(),
        parallelism,
    );
}
