use crate::assert;
use crate::internal_prelude::*;

pub fn inverse_scratch<T: ComplexField>(dim: usize, block_size: usize, par: Par) -> StackReq {
	_ = par;
	linalg::householder::apply_block_householder_sequence_transpose_on_the_right_in_place_scratch::<T>(dim, block_size, dim)
}

#[track_caller]
pub fn inverse<T: ComplexField>(
	out: MatMut<'_, T>,
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	R: MatRef<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	// A = Q R
	// A^-1 = R^-1 Q^-1

	let n = Q_basis.ncols();
	let block_size = Q_coeff.nrows();
	assert!(all(
		block_size > 0,
		Q_basis.nrows() == n,
		Q_basis.ncols() == n,
		Q_coeff.ncols() == n,
		R.nrows() == n,
		R.ncols() == n,
		out.nrows() == n,
		out.ncols() == n,
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
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use linalg::qr::no_pivoting::*;

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
		let mut Q_coeff = Mat::zeros(4, n);

		factor::qr_in_place(
			QR.as_mut(),
			Q_coeff.as_mut(),
			Par::Seq,
			MemStack::new(&mut { MemBuffer::new(factor::qr_in_place_scratch::<c64>(n, n, 4, Par::Seq, default())) }),
			default(),
		);

		let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));

		let mut A_inv = Mat::zeros(n, n);
		inverse::inverse(
			A_inv.as_mut(),
			QR.as_ref(),
			Q_coeff.as_ref(),
			QR.as_ref(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(inverse::inverse_scratch::<c64>(n, 4, Par::Seq))),
		);

		assert!(A_inv * A ~ Mat::identity(n, n));
	}
}
