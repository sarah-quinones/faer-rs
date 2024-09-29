use crate::{
    linalg::{temp_mat_req, temp_mat_uninit},
    ComplexField, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

/// Biconjugate gradient stabilized method.
pub mod bicgstab;
/// Conjugate gradient method.
pub mod conjugate_gradient;
/// Least squares minimal residual.
pub mod lsmr;

mod linop_impl;

/// Specifies whether the initial guess should be assumed to be zero or not.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum InitialGuessStatus {
    /// Initial guess is already zeroed.
    Zero,
    /// Initial guess may contain non-zero values.
    #[default]
    MaybeNonZero,
}

/// Identity preconditioner, no-op for most operations.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct IdentityPrecond {
    /// Dimension of the preconditioner, equal to the dimension of the solution.
    pub dim: usize,
}

impl<E: ComplexField> LinOp<E> for IdentityPrecond {
    #[inline]
    #[track_caller]
    fn apply_req(
        &self,
        _rhs_ncols: usize,
        _parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        Ok(StackReq::empty())
    }

    #[inline]
    fn nrows(&self) -> usize {
        self.dim
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.dim
    }

    #[inline]
    #[track_caller]
    fn apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        _parallelism: Parallelism,
        _stack: &mut PodStack,
    ) {
        { out }.copy_from(rhs);
    }
    #[inline]
    #[track_caller]
    fn conj_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        _parallelism: Parallelism,
        _stack: &mut PodStack,
    ) {
        { out }.copy_from(rhs);
    }
}
impl<E: ComplexField> BiLinOp<E> for IdentityPrecond {
    #[inline]
    fn transpose_apply_req(
        &self,
        _rhs_ncols: usize,
        _parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        Ok(StackReq::empty())
    }

    #[inline]
    #[track_caller]
    fn transpose_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        _parallelism: Parallelism,
        _stack: &mut PodStack,
    ) {
        { out }.copy_from(rhs);
    }
    #[inline]
    #[track_caller]
    fn adjoint_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        _parallelism: Parallelism,
        _stack: &mut PodStack,
    ) {
        { out }.copy_from(rhs);
    }
}
impl<E: ComplexField> Precond<E> for IdentityPrecond {
    fn apply_in_place_req(
        &self,
        _rhs_ncols: usize,
        _parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        Ok(StackReq::empty())
    }

    fn apply_in_place(
        &self,
        _rhs: MatMut<'_, E>,
        _parallelism: Parallelism,
        _stack: &mut PodStack,
    ) {
    }

    fn conj_apply_in_place(
        &self,
        _rhs: MatMut<'_, E>,
        _parallelism: Parallelism,
        _stack: &mut PodStack,
    ) {
    }
}
impl<E: ComplexField> BiPrecond<E> for IdentityPrecond {
    fn transpose_apply_in_place_req(
        &self,
        _rhs_ncols: usize,
        _parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        Ok(StackReq::empty())
    }

    fn transpose_apply_in_place(
        &self,
        _rhs: MatMut<'_, E>,
        _parallelism: Parallelism,
        _stack: &mut PodStack,
    ) {
    }

    fn adjoint_apply_in_place(
        &self,
        _rhs: MatMut<'_, E>,
        _parallelism: Parallelism,
        _stack: &mut PodStack,
    ) {
    }
}

/// Linear operator from a finite-dimensional vector space.
pub trait LinOp<E: ComplexField>: Sync + core::fmt::Debug {
    /// Computes the workspace size and alignment required to apply `self` or the conjugate of
    /// `self` to a matrix with `rhs_ncols` columns.
    fn apply_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow>;

    /// Output dimension of the operator.
    fn nrows(&self) -> usize;
    /// Input dimension of the operator.
    fn ncols(&self) -> usize;

    /// Applies `self` to `rhs`, and stores the result in `out`.
    fn apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    );

    /// Applies the conjugate of `self` to `rhs`, and stores the result in `out`.
    fn conj_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    );
}

/// Linear operator that can be applied from either the right or the left side.
pub trait BiLinOp<E: ComplexField>: LinOp<E> {
    /// Computes the workspace size and alignment required to apply the transpose or adjoint of
    /// `self` to a matrix with `rhs_ncols` columns.
    fn transpose_apply_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow>;

    /// Applies the transpose of `self` to `rhs`, and stores the result in `out`.
    fn transpose_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    );

    /// Applies the adjoint of `self` to `rhs`, and stores the result in `out`.
    fn adjoint_apply(
        &self,
        out: MatMut<'_, E>,
        rhs: MatRef<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    );
}

/// Preconditioner for a linear system.
///
/// Same as [`LinOp`] except that it can be applied in place.
pub trait Precond<E: ComplexField>: LinOp<E> {
    /// Computes the workspace size and alignment required to apply `self` or the conjugate of
    /// `self` to a matrix with `rhs_ncols` columns in place.
    fn apply_in_place_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        temp_mat_req::<E>(self.nrows(), rhs_ncols)?.try_and(self.apply_req(rhs_ncols, parallelism)?)
    }

    /// Applies `self` to `rhs`, and stores the result in `rhs`.
    #[track_caller]
    fn apply_in_place(&self, rhs: MatMut<'_, E>, parallelism: Parallelism, stack: &mut PodStack) {
        let (mut tmp, stack) = temp_mat_uninit::<E>(self.nrows(), rhs.ncols(), stack);
        self.apply(tmp.rb_mut(), rhs.rb(), parallelism, stack);
        { rhs }.copy_from(&tmp);
    }

    /// Applies the conjugate of `self` to `rhs`, and stores the result in `rhs`.
    #[track_caller]
    fn conj_apply_in_place(
        &self,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        let (mut tmp, stack) = temp_mat_uninit::<E>(self.nrows(), rhs.ncols(), stack);
        self.conj_apply(tmp.rb_mut(), rhs.rb(), parallelism, stack);
        { rhs }.copy_from(&tmp);
    }
}

/// Preconditioner for a linear system that can bee applied from either the right or the left side.
///
/// Same as [`BiLinOp`] except that it can be applied in place.
pub trait BiPrecond<E: ComplexField>: Precond<E> + BiLinOp<E> {
    /// Computes the workspace size and alignment required to apply the transpose or adjoint of
    /// `self` to a matrix with `rhs_ncols` columns in place.
    fn transpose_apply_in_place_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        temp_mat_req::<E>(self.nrows(), rhs_ncols)?
            .try_and(self.transpose_apply_req(rhs_ncols, parallelism)?)
    }

    /// Applies the transpose of `self` to `rhs`, and stores the result in `rhs`.
    #[track_caller]
    fn transpose_apply_in_place(
        &self,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        let (mut tmp, stack) = temp_mat_uninit::<E>(self.nrows(), rhs.ncols(), stack);
        self.transpose_apply(tmp.rb_mut(), rhs.rb(), parallelism, stack);
        { rhs }.copy_from(&tmp);
    }

    /// Applies the adjoint of `self` to `rhs`, and stores the result in `rhs`.
    #[track_caller]
    fn adjoint_apply_in_place(
        &self,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        let (mut tmp, stack) = temp_mat_uninit::<E>(self.nrows(), rhs.ncols(), stack);
        self.adjoint_apply(tmp.rb_mut(), rhs.rb(), parallelism, stack);
        { rhs }.copy_from(&tmp);
    }
}

impl<E: ComplexField, T: ?Sized + LinOp<E>> LinOp<E> for &T {
    #[inline]
    #[track_caller]
    fn apply_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        (**self).apply_req(rhs_ncols, parallelism)
    }

    #[inline]
    fn nrows(&self) -> usize {
        (**self).nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        (**self).ncols()
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
        (**self).apply(out, rhs, parallelism, stack)
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
        (**self).conj_apply(out, rhs, parallelism, stack)
    }
}

impl<E: ComplexField, T: ?Sized + BiLinOp<E>> BiLinOp<E> for &T {
    #[inline]
    #[track_caller]
    fn transpose_apply_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        (**self).transpose_apply_req(rhs_ncols, parallelism)
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
        (**self).transpose_apply(out, rhs, parallelism, stack)
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
        (**self).adjoint_apply(out, rhs, parallelism, stack)
    }
}

impl<E: ComplexField, T: ?Sized + Precond<E>> Precond<E> for &T {
    fn apply_in_place_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        (**self).apply_in_place_req(rhs_ncols, parallelism)
    }

    fn apply_in_place(&self, rhs: MatMut<'_, E>, parallelism: Parallelism, stack: &mut PodStack) {
        (**self).apply_in_place(rhs, parallelism, stack);
    }

    fn conj_apply_in_place(
        &self,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        (**self).conj_apply_in_place(rhs, parallelism, stack);
    }
}

impl<E: ComplexField, T: ?Sized + BiPrecond<E>> BiPrecond<E> for &T {
    fn transpose_apply_in_place_req(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        (**self).transpose_apply_in_place_req(rhs_ncols, parallelism)
    }

    fn transpose_apply_in_place(
        &self,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        (**self).transpose_apply_in_place(rhs, parallelism, stack);
    }

    fn adjoint_apply_in_place(
        &self,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) {
        (**self).adjoint_apply_in_place(rhs, parallelism, stack);
    }
}
