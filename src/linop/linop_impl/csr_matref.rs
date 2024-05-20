use crate::{
    linop::{BiLinOp, BiPrecond, LinOp, Precond},
    sparse::SparseRowMatRef,
    ComplexField, Conjugate, Index, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};

impl<E: ComplexField, I: Index, ViewE: Conjugate<Canonical = E>> LinOp<E>
    for SparseRowMatRef<'_, I, ViewE>
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
        stack: PodStack<'_>,
    ) {
        _ = stack;
        crate::sparse::linalg::matmul::dense_sparse_matmul(
            out.transpose_mut(),
            rhs.transpose(),
            self.transpose(),
            None,
            E::faer_one(),
            parallelism,
        );
    }

    #[inline]
    #[track_caller]
    fn conj_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) {
        _ = stack;
        let this = self.conjugate();
        crate::sparse::linalg::matmul::dense_sparse_matmul(
            out.transpose_mut(),
            rhs.transpose(),
            this.transpose(),
            None,
            E::faer_one(),
            parallelism,
        );
    }
}

impl<E: ComplexField, I: Index, ViewE: Conjugate<Canonical = E>> BiLinOp<E>
    for SparseRowMatRef<'_, I, ViewE>
{
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
        stack: PodStack<'_>,
    ) {
        _ = stack;
        let this = self.transpose();
        crate::sparse::linalg::matmul::sparse_dense_matmul(
            out,
            this,
            rhs,
            None,
            E::faer_one(),
            parallelism,
        );
    }

    #[inline]
    #[track_caller]
    fn adjoint_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) {
        _ = stack;
        let this = self.adjoint();
        crate::sparse::linalg::matmul::sparse_dense_matmul(
            out,
            this,
            rhs,
            None,
            E::faer_one(),
            parallelism,
        );
    }
}

impl<E: ComplexField, I: Index, ViewE: Conjugate<Canonical = E>> Precond<E>
    for SparseRowMatRef<'_, I, ViewE>
{
}
impl<E: ComplexField, I: Index, ViewE: Conjugate<Canonical = E>> BiPrecond<E>
    for SparseRowMatRef<'_, I, ViewE>
{
}
