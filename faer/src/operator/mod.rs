use crate::internal_prelude_sp::*;

/// Biconjugate gradient stabilized method.
pub mod bicgstab;
/// Conjugate gradient method.
pub mod conjugate_gradient;
/// Least squares minimal residual.
pub mod lsmr;

mod operator_impl;

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

/// Linear operator from a finite-dimensional vector space.
pub trait LinOp<T: ComplexField>: Sync + core::fmt::Debug {
	/// Computes the workspace size and alignment required to apply `self` or the conjugate of
	/// `self` to a matrix with `rhs_ncols` columns.
	fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq;

	/// Output dimension of the operator.
	fn nrows(&self) -> usize;
	/// Input dimension of the operator.
	fn ncols(&self) -> usize;

	/// Applies `self` to `rhs`, and stores the result in `out`.
	fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack);

	/// Applies the conjugate of `self` to `rhs`, and stores the result in `out`.
	fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack);
}

/// Linear operator that can be applied from either the right or the left side.
pub trait BiLinOp<T: ComplexField>: LinOp<T> {
	/// Computes the workspace size and alignment required to apply the transpose or adjoint of
	/// `self` to a matrix with `rhs_ncols` columns.
	fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq;

	/// Applies the transpose of `self` to `rhs`, and stores the result in `out`.
	fn transpose_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack);

	/// Applies the adjoint of `self` to `rhs`, and stores the result in `out`.
	fn adjoint_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack);
}

/// Preconditioner for a linear system.
///
/// Same as [`LinOp`] except that it can be applied in place.
pub trait Precond<T: ComplexField>: LinOp<T> {
	/// Computes the workspace size and alignment required to apply `self` or the conjugate of
	/// `self` to a matrix with `rhs_ncols` columns in place.
	fn apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		temp_mat_scratch::<T>(self.nrows(), rhs_ncols).and(self.apply_scratch(rhs_ncols, par))
	}

	/// Applies `self` to `rhs`, and stores the result in `rhs`.
	#[track_caller]
	fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		let (mut tmp, stack) = temp_mat_zeroed::<T, _, _>(self.nrows(), rhs.ncols(), stack);
		let mut tmp = tmp.as_mat_mut();
		self.apply(tmp.rb_mut(), rhs.rb(), par, stack);
		{ rhs }.copy_from(&tmp);
	}

	/// Applies the conjugate of `self` to `rhs`, and stores the result in `rhs`.
	#[track_caller]
	fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		let (mut tmp, stack) = temp_mat_zeroed::<T, _, _>(self.nrows(), rhs.ncols(), stack);
		let mut tmp = tmp.as_mat_mut();

		self.conj_apply(tmp.rb_mut(), rhs.rb(), par, stack);
		{ rhs }.copy_from(&tmp);
	}
}

/// Preconditioner for a linear system that can bee applied from either the right or the left side.
///
/// Same as [`BiLinOp`] except that it can be applied in place.
pub trait BiPrecond<T: ComplexField>: Precond<T> + BiLinOp<T> {
	/// Computes the workspace size and alignment required to apply the transpose or adjoint of
	/// `self` to a matrix with `rhs_ncols` columns in place.
	fn transpose_apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		temp_mat_scratch::<T>(self.nrows(), rhs_ncols).and(self.transpose_apply_scratch(rhs_ncols, par))
	}

	/// Applies the transpose of `self` to `rhs`, and stores the result in `rhs`.
	#[track_caller]
	fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		let (mut tmp, stack) = temp_mat_zeroed::<T, _, _>(self.nrows(), rhs.ncols(), stack);
		let mut tmp = tmp.as_mat_mut();
		self.transpose_apply(tmp.rb_mut(), rhs.rb(), par, stack);
		{ rhs }.copy_from(&tmp);
	}

	/// Applies the adjoint of `self` to `rhs`, and stores the result in `rhs`.
	#[track_caller]
	fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		let (mut tmp, stack) = temp_mat_zeroed::<T, _, _>(self.nrows(), rhs.ncols(), stack);
		let mut tmp = tmp.as_mat_mut();

		self.adjoint_apply(tmp.rb_mut(), rhs.rb(), par, stack);
		{ rhs }.copy_from(&tmp);
	}
}

impl<T: ComplexField, M: Sized + LinOp<T>> LinOp<T> for &M {
	#[inline]
	#[track_caller]
	fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		(**self).apply_scratch(rhs_ncols, par)
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
	fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		(**self).apply(out, rhs, par, stack)
	}

	#[inline]
	#[track_caller]
	fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		(**self).conj_apply(out, rhs, par, stack)
	}
}

impl<T: ComplexField, M: Sized + BiLinOp<T>> BiLinOp<T> for &M {
	#[inline]
	#[track_caller]
	fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		(**self).transpose_apply_scratch(rhs_ncols, par)
	}

	#[inline]
	#[track_caller]
	fn transpose_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		(**self).transpose_apply(out, rhs, par, stack)
	}

	#[inline]
	#[track_caller]
	fn adjoint_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		(**self).adjoint_apply(out, rhs, par, stack)
	}
}

impl<T: ComplexField, M: Sized + Precond<T>> Precond<T> for &M {
	fn apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		(**self).apply_in_place_scratch(rhs_ncols, par)
	}

	fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		(**self).apply_in_place(rhs, par, stack);
	}

	fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		(**self).conj_apply_in_place(rhs, par, stack);
	}
}

impl<T: ComplexField, M: Sized + BiPrecond<T>> BiPrecond<T> for &M {
	fn transpose_apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		(**self).transpose_apply_in_place_scratch(rhs_ncols, par)
	}

	fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		(**self).transpose_apply_in_place(rhs, par, stack);
	}

	fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		(**self).adjoint_apply_in_place(rhs, par, stack);
	}
}
