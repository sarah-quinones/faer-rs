//! The QR decomposition decomposes a matrix $A$ into the product
//! $$A = QR,$$
//! where $Q$ is a unitary matrix (represented as a block Householder sequence), and $R$ is an upper
//! trapezoidal matrix.

/// Computing the decomposition.
pub mod compute;
/// Reconstructing the inverse of the original matrix from the decomposition.
pub mod inverse;
/// Reconstructing the original matrix from the decomposition.
pub mod reconstruct;
/// Solving a linear system using the decomposition.
pub mod solve;
