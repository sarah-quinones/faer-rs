//! the pivoted $L L^\top$ decomposition of a self-adjoint positive definite matrix $A$ is such
//! that: $$P A P^\top = L L^H$$
//! where $L$ is a unit lower triangular matrix and $P$ is a permutation matrix
#![allow(missing_docs)]

pub mod factor;
pub mod inverse;
pub mod reconstruct;
pub mod solve;
