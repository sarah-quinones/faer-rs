use crate::assert;
use crate::internal_prelude::*;

pub fn inverse_scratch<I: Index, T: ComplexField>(dim: usize, par: Par) -> Result<StackReq, SizeOverflow> {
	_ = par;
	super::solve::solve_in_place_scratch::<I, T>(dim, dim, par)
}

#[track_caller]
#[math]
pub fn inverse<I: Index, T: ComplexField>(out: MatMut<'_, T>, L: MatRef<'_, T>, diagonal: DiagRef<'_, T>, subdiagonal: DiagRef<'_, T>, perm: PermRef<'_, I>, par: Par, stack: &mut DynStack) {
	let n = L.nrows();
	assert!(all(
		out.nrows() == n,
		out.ncols() == n,
		L.nrows() == n,
		L.ncols() == n,
		diagonal.dim() == n,
		subdiagonal.dim() == n,
		perm.len() == n,
	));

	let mut out = out;
	out.fill(zero());
	out.rb_mut().diagonal_mut().fill(one());

	super::solve::solve_in_place(L, diagonal, subdiagonal, perm, out.rb_mut(), par, stack);
}
