use crate::{assert, get_global_parallelism};
use crate::internal_prelude::*;
use alloc::vec;
use dyn_stack::MemBuffer;

/// Returns the determinant of `self`.
#[math]
pub fn determinant<T: ComplexField>(mat: MatRef<'_, T>) -> T::Real {
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
	).0.transposition_count;

	let mut det = one::<T>();
	for i in 0..factors.nrows() {
		det = mul(det, factors.as_mat_ref().read(i, i));
	}
	if count % 2 == 0 {
		real(det)
	} else {
		real(neg(det))
	}
}