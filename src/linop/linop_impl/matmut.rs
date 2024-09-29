use crate::{
    linop::{BiLinOp, BiPrecond, LinOp, Precond},
    ComplexField, Conjugate, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> LinOp<E> for MatMut<'_, ViewE> {
    #[inline]
    fn nrows(&self) -> usize {
        (*self).nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        (*self).ncols()
    }

    #[inline]
    fn apply_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        _ = (rhs_ncols, parallelism);
        Ok(StackReq::empty())
    }

    #[inline]
    #[track_caller]
    fn apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        _ = stack;
        let this = self.rb();
        crate::linalg::matmul::matmul(out, this, rhs, None, E::faer_one(), parallelism);
    }

    #[inline]
    #[track_caller]
    fn conj_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        _ = stack;
        let this = self.rb().conjugate();
        crate::linalg::matmul::matmul(out, this, rhs, None, E::faer_one(), parallelism);
    }
}

impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> BiLinOp<E> for MatMut<'_, ViewE> {
    #[inline]
    fn transpose_apply_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        _ = (rhs_ncols, parallelism);
        Ok(StackReq::empty())
    }

    #[inline]
    #[track_caller]
    fn transpose_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        _ = stack;
        let this = self.rb().transpose();
        crate::linalg::matmul::matmul(out, this, rhs, None, E::faer_one(), parallelism);
    }

    #[inline]
    #[track_caller]
    fn adjoint_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        _ = stack;
        let this = self.rb().adjoint();
        crate::linalg::matmul::matmul(out, this, rhs, None, E::faer_one(), parallelism);
    }
}

impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> Precond<E> for MatMut<'_, ViewE> {}
impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> BiPrecond<E> for MatMut<'_, ViewE> {}
