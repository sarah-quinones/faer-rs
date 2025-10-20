use super::*;

impl<T: ComplexField, I: Index, ViewT: Conjugate<Canonical = T>> LinOp<T> for SparseRowMatRef<'_, I, ViewT> {
	#[inline]

	fn nrows(&self) -> usize {
		(**self).nrows()
	}

	#[inline]

	fn ncols(&self) -> usize {
		(**self).ncols()
	}

	#[inline]

	fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		_ = (rhs_ncols, par);

		StackReq::EMPTY
	}

	#[inline]
	#[track_caller]

	fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		self.rb().transpose().transpose_apply(out, rhs, par, stack)
	}

	#[inline]
	#[track_caller]

	fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		self.rb().transpose().adjoint_apply(out, rhs, par, stack)
	}
}

impl<T: ComplexField, I: Index, ViewT: Conjugate<Canonical = T>> BiLinOp<T> for SparseRowMatRef<'_, I, ViewT> {
	#[inline]

	fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		_ = (rhs_ncols, par);

		StackReq::EMPTY
	}

	#[inline]
	#[track_caller]

	fn transpose_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		self.rb().transpose().apply(out, rhs, par, stack)
	}

	#[inline]
	#[track_caller]

	fn adjoint_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		self.rb().transpose().conj_apply(out, rhs, par, stack)
	}
}

impl<T: ComplexField, I: Index, ViewT: Conjugate<Canonical = T>> Precond<T> for SparseRowMatRef<'_, I, ViewT> {}

impl<T: ComplexField, I: Index, ViewT: Conjugate<Canonical = T>> BiPrecond<T> for SparseRowMatRef<'_, I, ViewT> {}
