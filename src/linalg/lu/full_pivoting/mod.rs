//! The full pivoting LU decomposition is such that:
//! $$PAQ^\top = LU,$$
//! where $P$ and $Q$ are permutation matrices, $L$ is a unit lower triangular matrix, and $U$ is
//! an upper triangular matrix.
//!
//! The full pivoting LU decomposition is more numerically stable than the one with partial
//! pivoting, but is more expensive to compute.

/// Computing the decomposition.
pub mod compute;
/// Reconstructing the inverse of the original matrix from the decomposition.
pub mod inverse;
/// Reconstructing the inverse of the original matrix from the decomposition.
pub mod reconstruct;
/// Solving a linear system usin the decomposition.
pub mod solve;
