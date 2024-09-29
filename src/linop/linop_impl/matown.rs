use crate::{
    linop::{BiLinOp, BiPrecond, LinOp, Precond},
    ComplexField, Conjugate, Mat, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};

impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> LinOp<E> for Mat<ViewE> {
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
        let this = self.as_ref();
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
        let this = self.as_ref().conjugate();
        crate::linalg::matmul::matmul(out, this, rhs, None, E::faer_one(), parallelism);
    }
}

impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> BiLinOp<E> for Mat<ViewE> {
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
        let this = self.as_ref().transpose();
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
        let this = self.as_ref().adjoint();
        crate::linalg::matmul::matmul(out, this, rhs, None, E::faer_one(), parallelism);
    }
}

impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> Precond<E> for Mat<ViewE> {}
impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> BiPrecond<E> for Mat<ViewE> {}
