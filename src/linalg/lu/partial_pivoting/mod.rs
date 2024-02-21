//! The partial pivoting LU decomposition is such that:
//! $$PA = LU,$$
//! where $P$ is a permutation matrix, $L$ is a unit lower triangular matrix, and $U$ is
//! an upper triangular matrix.

/// Computing the decomposition.
pub mod compute;
/// Reconstructing the inverse of the original matrix from the decomposition.
pub mod inverse;
/// Reconstructing the original matrix from the decomposition.
pub mod reconstruct;
/// Solving a linear system usin the decomposition.
pub mod solve;
