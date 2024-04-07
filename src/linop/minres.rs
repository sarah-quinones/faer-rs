use crate::{
    linalg::{householder, matmul::matmul, qr, temp_mat_req, temp_mat_uninit},
    linop::{BiLinOp, BiPrecond, InitialGuessStatus},
    prelude::*,
    utils::DivCeil,
    ComplexField, Conj, Parallelism, RealField,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use equator::{assert, debug_assert};
use reborrow::*;

#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct MinResParams<E: ComplexField> {
    pub initial_guess: InitialGuessStatus,
    pub abs_tolerance: E::Real,
    pub rel_tolerance: E::Real,
    pub max_iters: usize,
}

impl<E: ComplexField> Default for MinResParams<E> {
    #[inline]
    fn default() -> Self {
        Self {
            initial_guess: InitialGuessStatus::MaybeNonZero,
            abs_tolerance: E::Real::faer_zero(),
            rel_tolerance: E::Real::faer_epsilon().faer_mul(E::Real::faer_from_f64(128.0)),
            max_iters: usize::MAX,
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct MinResInfo<E: ComplexField> {
    pub abs_residual: E::Real,
    pub rel_residual: E::Real,
    pub iter_count: usize,
}

#[derive(Copy, Clone, Debug)]
pub enum MinResError<E: ComplexField> {
    NoConvergence {
        abs_residual: E::Real,
        rel_residual: E::Real,
    },
}

#[allow(dead_code)]
#[cfg(test)]
pub fn minres_prototype<E: ComplexField>(
    mut x: MatMut<'_, E>,
    A: MatRef<'_, E>,
    b: MatRef<'_, E>,
    params: MinResParams<E>,
) {
}
