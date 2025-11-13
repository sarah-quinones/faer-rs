//! the $QR$ decomposition decomposes a matrix $A$ into the product
//! $$A = QR$$
//! where $Q$ is a unitary matrix (represented as a block householder sequence), and $R$ is an upper
//! trapezoidal matrix.
#![allow(missing_docs)]
pub mod factor;
pub mod inverse;
pub mod reconstruct;
pub mod solve;
