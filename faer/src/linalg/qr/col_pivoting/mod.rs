//! the $QR$ decomposition with column pivoting decomposes a matrix $A$ into the product
//! $$AP^T = QR$$
//! where $P$ is a permutation matrix, $Q$ is a unitary matrix (represented as a block householder
//! sequence), and $R$ is an upper trapezoidal matrix.
#![allow(missing_docs)]

pub mod factor;
pub mod inverse;
pub mod reconstruct;
pub mod solve;
