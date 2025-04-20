use crate::internal_prelude::*;
use crate::{assert, get_global_parallelism};
use alloc::vec;
use dyn_stack::MemBuffer;

#[math]
pub fn determinant<T: ComplexField>(mat: MatRef<'_, T>) -> T::Canonical {
	assert!(mat.nrows() == mat.ncols());

	let par = get_global_parallelism();
	let (m, n) = mat.shape();
	let mut row_perm_fwd = vec![0usize; m];
	let mut row_perm_bwd = vec![0usize; m];

	let mut factors = mat.to_owned();
	let count = linalg::lu::partial_pivoting::factor::lu_in_place(
		factors.as_mat_mut(),
		&mut row_perm_fwd,
		&mut row_perm_bwd,
		par,
		MemStack::new(&mut MemBuffer::new(
			linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T>(m, n, par, default()),
		)),
		default(),
	)
	.0
	.transposition_count;

	let mut det = one();
	for i in 0..factors.nrows() {
		det = mul(det, factors[(i, i)]);
	}
	if count % 2 == 0 { det } else { neg(det) }
}
