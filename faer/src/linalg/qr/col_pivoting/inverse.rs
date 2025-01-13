use crate::assert;
use crate::internal_prelude::*;
use linalg::householder::apply_block_householder_sequence_transpose_on_the_right_in_place_scratch;

pub fn inverse_scratch<I: Index, T: ComplexField>(dim: usize, blocksize: usize, par: Par) -> StackReq {
	_ = par;
	StackReq::or(
		apply_block_householder_sequence_transpose_on_the_right_in_place_scratch::<T>(dim, blocksize, dim),
		crate::perm::permute_cols_in_place_scratch::<I, T>(dim, dim),
	)
}

#[track_caller]
pub fn inverse<I: Index, T: ComplexField>(
	out: MatMut<'_, T>,
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	R: MatRef<'_, T>,
	col_perm: PermRef<'_, I>,
	par: Par,
	stack: &mut MemStack,
) {
	// A P^-1 = Q R
	// A^-1 = P^-1 R^-1 Q^-1

	let n = Q_basis.ncols();
	let blocksize = Q_coeff.nrows();
	assert!(all(
		blocksize > 0,
		Q_basis.nrows() == n,
		Q_basis.ncols() == n,
		Q_coeff.ncols() == n,
		R.nrows() == n,
		R.ncols() == n,
		out.nrows() == n,
		out.ncols() == n,
		col_perm.len() == n,
	));

	let mut out = out;
	out.fill(zero());
	linalg::triangular_inverse::invert_upper_triangular(out.rb_mut(), R, par);
	linalg::householder::apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj(
		Q_basis,
		Q_coeff,
		Conj::Yes,
		out.rb_mut(),
		par,
		stack,
	);
	crate::perm::permute_rows_in_place(out.rb_mut(), col_perm.inverse(), stack);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use linalg::qr::col_pivoting::*;

	#[test]
	fn test_inverse() {
		let rng = &mut StdRng::seed_from_u64(0);
		let n = 50;
		let A = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);

		let mut QR = A.to_owned();
		let mut H = Mat::zeros(4, n);
		let col_perm_fwd = &mut *vec![0usize; n];
		let col_perm_bwd = &mut *vec![0usize; n];

		let (_, col_perm) = factor::qr_in_place(
			QR.as_mut(),
			H.as_mut(),
			col_perm_fwd,
			col_perm_bwd,
			Par::Seq,
			MemStack::new(&mut { MemBuffer::new(factor::qr_in_place_scratch::<usize, c64>(n, n, 4, Par::Seq, default())) }),
			default(),
		);

		let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));

		let mut A_inv = Mat::zeros(n, n);
		inverse::inverse(
			A_inv.as_mut(),
			QR.as_ref(),
			H.as_ref(),
			QR.as_ref(),
			col_perm,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(inverse::inverse_scratch::<usize, c64>(n, 4, Par::Seq))),
		);

		assert!(A_inv * A ~ Mat::identity(n, n));
	}
}
