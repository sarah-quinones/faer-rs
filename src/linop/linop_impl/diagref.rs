use crate::{
    diag::DiagRef,
    linop::{BiLinOp, BiPrecond, LinOp, Precond},
    ComplexField, Conjugate, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> LinOp<E> for DiagRef<'_, ViewE> {
    #[inline]
    fn apply_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let _ = rhs_ncols;
        let _ = parallelism;
        Ok(StackReq::empty())
    }

    #[inline]
    fn nrows(&self) -> usize {
        self.column_vector().nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.column_vector().nrows()
    }

    fn apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        let k = rhs.ncols();
        _ = parallelism;
        _ = stack;
        let mut out = out;
        for j in 0..k {
            crate::zipped!(
                __rw,
                out.rb_mut().col_mut(j),
                rhs.col(j),
                self.column_vector()
            )
            .for_each(|crate::unzipped!(mut out, rhs, d)| {
                out.write(rhs.read() * d.read().canonicalize())
            });
        }
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        self.column_vector()
            .conjugate()
            .column_vector_as_diagonal()
            .apply(out, rhs, parallelism, stack)
    }
}

impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> BiLinOp<E> for DiagRef<'_, ViewE> {
    #[inline]
    fn transpose_apply_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        self.apply_req(rhs_ncols, parallelism)
    }

    fn transpose_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        self.apply(out, rhs, parallelism, stack);
    }

    fn adjoint_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        self.conj_apply(out, rhs, parallelism, stack);
    }
}

impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> Precond<E> for DiagRef<'_, ViewE> {
    fn apply_in_place_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let _ = rhs_ncols;
        let _ = parallelism;
        Ok(StackReq::empty())
    }

    fn apply_in_place(&self, rhs: MatMut<'_, E>, parallelism: Parallelism, stack: &mut PodStack) {
        _ = parallelism;
        _ = stack;
        let mut rhs = rhs;
        let k = rhs.ncols();
        for j in 0..k {
            crate::zipped!(__rw, rhs.rb_mut().col_mut(j), self.column_vector()).for_each(
                |crate::unzipped!(mut out, d)| out.write(out.read() * d.read().canonicalize()),
            );
        }
    }

    fn conj_apply_in_place(
        &self,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        self.column_vector()
            .conjugate()
            .column_vector_as_diagonal()
            .apply_in_place(rhs, parallelism, stack)
    }
}
impl<E: ComplexField, ViewE: Conjugate<Canonical = E>> BiPrecond<E> for DiagRef<'_, ViewE> {
    fn transpose_apply_in_place_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        self.apply_in_place_req(rhs_ncols, parallelism)
    }

    fn transpose_apply_in_place(
        &self,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        self.apply_in_place(rhs, parallelism, stack)
    }

    fn adjoint_apply_in_place(
        &self,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        self.conj_apply_in_place(rhs, parallelism, stack)
    }
}
