use super::*;
impl<I: Index, T: ComplexField, ViewT: Conjugate<Canonical = T>> LinOp<T> for SparseColMat<I, ViewT> {
	#[inline]
	fn nrows(&self) -> usize {
		self.rb().nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		self.rb().ncols()
	}

	#[inline]
	fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		self.rb().apply_scratch(rhs_ncols, par)
	}

	#[inline]
	#[track_caller]
	fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		self.rb().apply(out, rhs, par, stack)
	}

	#[inline]
	#[track_caller]
	fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		self.rb().conj_apply(out, rhs, par, stack)
	}
}
impl<I: Index, T: ComplexField, ViewT: Conjugate<Canonical = T>> BiLinOp<T> for SparseColMat<I, ViewT> {
	#[inline]
	fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		self.rb().transpose_apply_scratch(rhs_ncols, par)
	}

	#[inline]
	#[track_caller]
	fn transpose_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		self.rb().transpose_apply(out, rhs, par, stack)
	}

	#[inline]
	#[track_caller]
	fn adjoint_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		self.rb().adjoint_apply(out, rhs, par, stack)
	}
}
impl<I: Index, T: ComplexField, ViewT: Conjugate<Canonical = T>> Precond<T> for SparseColMat<I, ViewT> {}
impl<I: Index, T: ComplexField, ViewT: Conjugate<Canonical = T>> BiPrecond<T> for SparseColMat<I, ViewT> {}
