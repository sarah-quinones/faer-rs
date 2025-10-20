use super::*;

impl<T: ComplexField, ViewT: Conjugate<Canonical = T>> LinOp<T> for DiagRef<'_, ViewT> {
	#[inline]

	fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		let _ = rhs_ncols;

		let _ = par;

		StackReq::EMPTY
	}

	#[inline]

	fn nrows(&self) -> usize {
		self.column_vector().nrows()
	}

	#[inline]

	fn ncols(&self) -> usize {
		self.column_vector().nrows()
	}

	fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		let k = rhs.ncols();

		_ = par;

		_ = stack;

		let mut out = out;

		for j in 0..k {
			z!(out.rb_mut().col_mut(j), rhs.col(j), self.column_vector()).for_each(|uz!(out, rhs, d): Zip![&mut _, ..]| {
				*out = rhs * Conj::apply(d);
			});
		}
	}

	fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		self.column_vector().conjugate().as_diagonal().apply(out, rhs, par, stack)
	}
}

impl<T: ComplexField, ViewT: Conjugate<Canonical = T>> BiLinOp<T> for DiagRef<'_, ViewT> {
	#[inline]

	fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		self.apply_scratch(rhs_ncols, par)
	}

	fn transpose_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		self.apply(out, rhs, par, stack);
	}

	fn adjoint_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
		self.conj_apply(out, rhs, par, stack);
	}
}

impl<T: ComplexField, ViewT: Conjugate<Canonical = T>> Precond<T> for DiagRef<'_, ViewT> {
	fn apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		let _ = rhs_ncols;

		let _ = par;

		StackReq::EMPTY
	}

	fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		_ = par;

		_ = stack;

		let mut rhs = rhs;

		let k = rhs.ncols();

		for j in 0..k {
			z!(rhs.rb_mut().col_mut(j), self.column_vector()).for_each(|uz!(out, d): Zip![&mut _, ..]| *out *= Conj::apply(d));
		}
	}

	fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		self.column_vector().conjugate().as_diagonal().apply_in_place(rhs, par, stack)
	}
}

impl<T: ComplexField, ViewT: Conjugate<Canonical = T>> BiPrecond<T> for DiagRef<'_, ViewT> {
	fn transpose_apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		self.apply_in_place_scratch(rhs_ncols, par)
	}

	fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		self.apply_in_place(rhs, par, stack)
	}

	fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
		self.conj_apply_in_place(rhs, par, stack)
	}
}
