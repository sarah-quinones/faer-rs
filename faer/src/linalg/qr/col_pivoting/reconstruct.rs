use crate::assert;
use crate::internal_prelude::*;

pub fn reconstruct_scratch<I: Index, T: ComplexField>(nrows: usize, ncols: usize, blocksize: usize, par: Par) -> StackReq {
	_ = par;
	StackReq::or(
		linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(nrows, blocksize, ncols),
		crate::perm::permute_cols_in_place_scratch::<I, T>(nrows, ncols),
	)
}

#[track_caller]
pub fn reconstruct<I: Index, T: ComplexField>(
	out: MatMut<'_, T>,
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	R: MatRef<'_, T>,
	col_perm: PermRef<'_, I>,
	par: Par,
	stack: &mut MemStack,
) {
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
		col_perm.len() == n,
	));

	let mut out = out;
	out.fill(zero());
	out.rb_mut().get_mut(..size, ..n).copy_from_triangular_upper(R);

	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(Q_basis, Q_coeff, Conj::No, out.rb_mut(), par, stack);
	crate::perm::permute_cols_in_place(out.rb_mut(), col_perm.inverse(), stack);
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
	fn test_reconstruct() {
		let rng = &mut StdRng::seed_from_u64(0);
		for (m, n) in [(100, 50), (50, 100)] {
			let size = Ord::min(m, n);

			let A = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			let mut QR = A.to_owned();
			let mut Q_coeff = Mat::zeros(4, Ord::min(m, n));
			let col_perm_fwd = &mut *vec![0usize; n];
			let col_perm_bwd = &mut *vec![0usize; n];

			let (_, col_perm) = factor::qr_in_place(
				QR.as_mut(),
				Q_coeff.as_mut(),
				col_perm_fwd,
				col_perm_bwd,
				Par::Seq,
				MemStack::new(&mut { MemBuffer::new(factor::qr_in_place_scratch::<usize, c64>(m, n, 4, Par::Seq, default())) }),
				default(),
			);

			let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));

			let mut A_rec = Mat::zeros(m, n);
			reconstruct::reconstruct(
				A_rec.as_mut(),
				QR.get(.., ..size),
				Q_coeff.as_ref(),
				QR.get(..size, ..),
				col_perm,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(reconstruct::reconstruct_scratch::<usize, c64>(m, n, 4, Par::Seq))),
			);

			assert!(A_rec ~ A);
		}
	}
}
