use super::*;

impl<T: ComplexField, ViewT: Conjugate<Canonical = T>> LinOp<T> for MatRef<'_, ViewT> {
	#[inline]

	fn nrows(&self) -> usize {
		(*self).nrows()
	}

	#[inline]

	fn ncols(&self) -> usize {
		(*self).ncols()
	}

	#[inline]

	fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		_ = (rhs_ncols, par);

		StackReq::EMPTY
	}

	#[inline]
	#[track_caller]

	fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		_ = stack;

		linalg::matmul::matmul(out, Accum::Replace, *self, rhs, one::<T>(), par);
	}

	#[inline]
	#[track_caller]

	fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		_ = stack;

		let this = self.conjugate();

		linalg::matmul::matmul(out, Accum::Replace, this, rhs, one::<T>(), par);
	}
}

impl<T: ComplexField, ViewT: Conjugate<Canonical = T>> BiLinOp<T> for MatRef<'_, ViewT> {
	#[inline]

	fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		_ = (rhs_ncols, par);

		StackReq::EMPTY
	}

	#[inline]
	#[track_caller]

	fn transpose_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		_ = stack;

		let this = self.transpose();

		linalg::matmul::matmul(out, Accum::Replace, this, rhs, one::<T>(), par);
	}

	#[inline]
	#[track_caller]

	fn adjoint_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		_ = stack;

		let this = self.adjoint();

		linalg::matmul::matmul(out, Accum::Replace, this, rhs, one::<T>(), par);
	}
}

impl<T: ComplexField, ViewT: Conjugate<Canonical = T>> Precond<T> for MatRef<'_, ViewT> {}

impl<T: ComplexField, ViewT: Conjugate<Canonical = T>> BiPrecond<T> for MatRef<'_, ViewT> {}
