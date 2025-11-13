//! the full pivoting $LU$ decomposition is such that:
//! $$P A Q^\top = LU$$
//! where $P$ and $Q$ are permutation matrices, $L$ is a unit lower triangular matrix, and $U$ is
//! an upper triangular matrix.
#![allow(missing_docs)]
pub mod factor;
pub mod inverse;
pub mod reconstruct;
pub mod solve;
