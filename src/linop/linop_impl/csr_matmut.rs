use crate::{
    linop::{BiLinOp, BiPrecond, LinOp, Precond},
    sparse::SparseRowMatMut,
    ComplexField, Conjugate, Index, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};

impl<E: ComplexField, I: Index, ViewE: Conjugate<Canonical = E>> LinOp<E>
    for SparseRowMatMut<'_, I, ViewE>
{
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
        self.as_ref().apply_req(rhs_ncols, parallelism)
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
        self.as_ref().apply(out, rhs, parallelism, stack)
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
        self.as_ref().conj_apply(out, rhs, parallelism, stack)
    }
}

impl<E: ComplexField, I: Index, ViewE: Conjugate<Canonical = E>> BiLinOp<E>
    for SparseRowMatMut<'_, I, ViewE>
{
    #[inline]
    fn transpose_apply_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        self.as_ref().transpose_apply_req(rhs_ncols, parallelism)
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
        self.as_ref().transpose_apply(out, rhs, parallelism, stack)
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
        self.as_ref().adjoint_apply(out, rhs, parallelism, stack)
    }
}

impl<E: ComplexField, I: Index, ViewE: Conjugate<Canonical = E>> Precond<E>
    for SparseRowMatMut<'_, I, ViewE>
{
}
impl<E: ComplexField, I: Index, ViewE: Conjugate<Canonical = E>> BiPrecond<E>
    for SparseRowMatMut<'_, I, ViewE>
{
}
