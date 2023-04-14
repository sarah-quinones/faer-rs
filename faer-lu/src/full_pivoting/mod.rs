//! The full pivoting LU decomposition is such that:
//! $$PAQ^\top = LU,$$
//! where $P$ and $Q$ are permutation matrices, $L$ is a unit lower triangular matrix, and $U$ is
//! an upper triangular matrix.
//!
//! The full pivoting LU decomposition is more numerically stable than the one with partial
//! pivoting, but is more expensive to compute.

pub mod compute;
pub mod inverse;
pub mod reconstruct;
pub mod solve;
