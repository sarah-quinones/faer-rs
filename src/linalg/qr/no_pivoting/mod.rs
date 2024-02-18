//! The QR decomposition decomposes a matrix $A$ into the product
//! $$A = QR,$$
//! where $Q$ is a unitary matrix (represented as a block Householder sequence), and $R$ is an upper
//! trapezoidal matrix.

pub mod compute;
pub mod inverse;
pub mod reconstruct;
pub mod solve;
