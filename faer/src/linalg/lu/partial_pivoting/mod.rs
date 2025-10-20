//! the partial pivoting $LU$ decomposition is such that:
//! $$PA = LU$$
//! where $P$ is a permutation matrix, $L$ is a unit lower triangular matrix, and $U$ is
//! an upper triangular matrix.
#![allow(missing_docs)]

pub mod factor;
pub mod inverse;
pub mod reconstruct;
pub mod solve;
