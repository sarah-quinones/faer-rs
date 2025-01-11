//! the $L L^\top$ decomposition of a self-adjoint positive definite matrix $A$ is such that:
//! $$A = L L^H$$
//! where $L$ is a lower triangular matrix
#![allow(missing_docs)]

pub mod factor;
pub mod solve;
pub mod update;

pub mod inverse;
pub mod reconstruct;
