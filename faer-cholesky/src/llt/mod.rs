//! The Cholesky decomposition of a hermitian positive definite matrix $A$ is such that:
//! $$A = LL^H,$$
//! where $L$ is a lower triangular matrix.

pub mod compute;
pub mod inverse;
pub mod reconstruct;
pub mod solve;
pub mod update;

#[derive(Debug, Clone, Copy)]
pub struct CholeskyError;
