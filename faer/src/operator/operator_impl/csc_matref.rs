use super::*;

impl<T: ComplexField, I: Index, ViewT: Conjugate<Canonical = T>> LinOp<T> for SparseColMatRef<'_, I, ViewT> {
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
		_ = stack;
		linalg_sp::matmul::sparse_dense_matmul(out, Accum::Replace, *self, rhs, one(), par);
	}

	#[inline]
	#[track_caller]
	fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		_ = stack;
		let this = self.conjugate();
		linalg_sp::matmul::sparse_dense_matmul(out, Accum::Replace, this, rhs, one(), par);
	}
}

impl<T: ComplexField, I: Index, ViewT: Conjugate<Canonical = T>> BiLinOp<T> for SparseColMatRef<'_, I, ViewT> {
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
		linalg_sp::matmul::dense_sparse_matmul(out.transpose_mut(), Accum::Replace, rhs.transpose(), this.transpose(), one(), par);
	}

	#[inline]
	#[track_caller]
	fn adjoint_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		_ = stack;
		let this = self.adjoint();
		linalg_sp::matmul::dense_sparse_matmul(out.transpose_mut(), Accum::Replace, rhs.transpose(), this.transpose(), one(), par);
	}
}

impl<T: ComplexField, I: Index, ViewT: Conjugate<Canonical = T>> Precond<T> for SparseColMatRef<'_, I, ViewT> {}
impl<T: ComplexField, I: Index, ViewT: Conjugate<Canonical = T>> BiPrecond<T> for SparseColMatRef<'_, I, ViewT> {}
