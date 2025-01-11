//! the $L D L^\top$ decomposition of a self-adjoint positive definite matrix $A$ is such that:
//! $$A = L D L^H$$
//! where $L$ is a unit lower triangular matrix, and $D$ is a diagonal matrix
#![allow(missing_docs)]

pub mod factor;
pub mod solve;
pub mod update;

pub mod inverse;
pub mod reconstruct;
