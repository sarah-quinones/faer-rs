//! The partial pivoting LU decomposition is such that:
//! $$PA = LU,$$
//! where $P$ is a permutation matrix, $L$ is a unit lower triangular matrix, and $U$ is
//! an upper triangular matrix.

mod compute;
mod inverse;
mod reconstruct;
mod solve;

pub use compute::*;
pub use inverse::*;
pub use reconstruct::*;
pub use solve::*;
