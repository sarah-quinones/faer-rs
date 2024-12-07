use crate::assert;
use crate::internal_prelude::*;

pub fn reconstruct_scratch<T: ComplexField>(nrows: usize, ncols: usize, blocksize: usize, par: Par) -> Result<StackReq, SizeOverflow> {
	_ = par;
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(nrows, blocksize, ncols)
}

#[track_caller]
pub fn reconstruct<T: ComplexField>(out: MatMut<'_, T>, Q_basis: MatRef<'_, T>, Q_coeff: MatRef<'_, T>, R: MatRef<'_, T>, par: Par, stack: &mut DynStack) {
	let m = Q_basis.nrows();
	let n = R.ncols();
	let size = Ord::min(m, n);
	assert!(all(
		out.nrows() == m,
		out.ncols() == n,
		Q_basis.nrows() == m,
		Q_basis.ncols() == size,
		Q_coeff.ncols() == size,
		R.nrows() == size,
		R.ncols() == n,
	));

	let mut out = out;
	out.fill(zero());
	out.rb_mut().get_mut(..size, ..n).copy_from_triangular_upper(R);

	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(Q_basis, Q_coeff, Conj::No, out.rb_mut(), par, stack);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::GlobalMemBuffer;
	use linalg::qr::no_pivoting::*;

	#[test]
	#[azucar::infer]
	fn test_reconstruct() {
		let rng = &mut StdRng::seed_from_u64(0);
		for (m, n) in [(100, 50), (50, 100)] {
			let A = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);
			let size = Ord::min(m, n);

			let mut QR = A.to_owned();
			let mut Q_coeff = Mat::zeros(4, size);

			factor::qr_in_place(
				QR.as_mut(),
				Q_coeff.as_mut(),
				Par::Seq,
				DynStack::new(&mut { GlobalMemBuffer::new(factor::qr_in_place_scratch::<c64>(m, n, 4, Par::Seq, _).unwrap()) }),
				_,
			);

			let approx_eq = CwiseMat(ApproxEq::<c64>::eps() * (n as f64));

			let mut A_rec = Mat::zeros(m, n);
			reconstruct::reconstruct(
				A_rec.as_mut(),
				QR.get(.., ..size),
				Q_coeff.as_ref(),
				QR.get(..size, ..),
				Par::Seq,
				DynStack::new(&mut GlobalMemBuffer::new(reconstruct::reconstruct_scratch::<c64>(m, n, 4, Par::Seq).unwrap())),
			);

			assert!(A_rec ~ A);
		}
	}
}
