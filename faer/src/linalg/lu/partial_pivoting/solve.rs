use crate::assert;
use crate::internal_prelude::*;
use crate::perm::{permute_rows_in_place, permute_rows_in_place_scratch};

pub fn solve_in_place_scratch<I: Index, T: ComplexField>(LU_dim: usize, rhs_ncols: usize, par: Par) -> StackReq {
	_ = par;
	permute_rows_in_place_scratch::<I, T>(LU_dim, rhs_ncols)
}

pub fn solve_transpose_in_place_scratch<I: Index, T: ComplexField>(LU_dim: usize, rhs_ncols: usize, par: Par) -> StackReq {
	_ = par;
	permute_rows_in_place_scratch::<I, T>(LU_dim, rhs_ncols)
}

#[track_caller]
pub fn solve_in_place_with_conj<I: Index, T: ComplexField>(
	L: MatRef<'_, T>,
	U: MatRef<'_, T>,
	row_perm: PermRef<'_, I>,
	conj_LU: Conj,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	// LU = PA
	// P^-1 LU = A
	// A^-1 = U^-1 L^-1 P

	let n = L.nrows();

	assert!(all(
		L.nrows() == n,
		L.ncols() == n,
		U.nrows() == n,
		U.ncols() == n,
		row_perm.len() == n,
		rhs.nrows() == n,
	));

	let mut rhs = rhs;
	permute_rows_in_place(rhs.rb_mut(), row_perm, stack);

	linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(L, conj_LU, rhs.rb_mut(), par);

	linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(U, conj_LU, rhs.rb_mut(), par);
}

#[track_caller]
pub fn solve_transpose_in_place_with_conj<I: Index, T: ComplexField>(
	L: MatRef<'_, T>,
	U: MatRef<'_, T>,
	row_perm: PermRef<'_, I>,
	conj_LU: Conj,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	// LU = PA
	// P^-1 LU = A
	// A^-T = (U^-1 L^-1 P).T
	// A^-T = P^-1 L^-T U^-T

	let n = L.nrows();

	assert!(all(
		L.nrows() == n,
		L.ncols() == n,
		U.nrows() == n,
		U.ncols() == n,
		row_perm.len() == n,
		rhs.nrows() == n,
	));

	let mut rhs = rhs;

	linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(U.transpose(), conj_LU, rhs.rb_mut(), par);
	linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(L.transpose(), conj_LU, rhs.rb_mut(), par);

	permute_rows_in_place(rhs.rb_mut(), row_perm.inverse(), stack);
}

#[track_caller]
pub fn solve_in_place<I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
	L: MatRef<'_, C>,
	U: MatRef<'_, C>,
	row_perm: PermRef<'_, I>,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	solve_in_place_with_conj(L.canonical(), U.canonical(), row_perm, Conj::get::<C>(), rhs, par, stack)
}

#[track_caller]
pub fn solve_transpose_in_place<I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
	L: MatRef<'_, C>,
	U: MatRef<'_, C>,
	row_perm: PermRef<'_, I>,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	solve_transpose_in_place_with_conj(L.canonical(), U.canonical(), row_perm, Conj::get::<C>(), rhs, par, stack)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use linalg::lu::partial_pivoting::*;

	#[azucar::infer]
	#[test]
	fn test_solve() {
		let rng = &mut StdRng::seed_from_u64(0);
		let n = 50;
		let k = 3;

		let A = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);

		let B = CwiseMatDistribution {
			nrows: n,
			ncols: k,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);

		let mut LU = A.to_owned();
		let row_perm_fwd = &mut *vec![0usize; n];
		let row_perm_bwd = &mut *vec![0usize; n];

		let row_perm = factor::lu_in_place(
			LU.as_mut(),
			row_perm_fwd,
			row_perm_bwd,
			Par::Seq,
			MemStack::new(&mut { MemBuffer::new(factor::lu_in_place_scratch::<usize, c64>(n, n, Par::Seq, _)) }),
			_,
		)
		.1;

		let approx_eq = CwiseMat(ApproxEq::eps() * 8.0 * (n as f64));

		{
			let mut X = B.to_owned();
			solve::solve_in_place(
				LU.as_ref(),
				LU.as_ref(),
				row_perm,
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(solve::solve_in_place_scratch::<usize, c64>(n, k, Par::Seq))),
			);

			assert!(&A * &X ~ B);
		}
		{
			let mut X = B.to_owned();
			solve::solve_transpose_in_place(
				LU.as_ref(),
				LU.as_ref(),
				row_perm,
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(solve::solve_in_place_scratch::<usize, c64>(n, k, Par::Seq))),
			);

			assert!(A.transpose() * &X ~ B);
		}
		{
			let mut X = B.to_owned();
			solve::solve_in_place(
				LU.conjugate(),
				LU.conjugate(),
				row_perm,
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(solve::solve_in_place_scratch::<usize, c64>(n, k, Par::Seq))),
			);

			assert!(A.conjugate() * &X ~ B);
		}
		{
			let mut X = B.to_owned();
			solve::solve_transpose_in_place(
				LU.conjugate(),
				LU.conjugate(),
				row_perm,
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(solve::solve_in_place_scratch::<usize, c64>(n, k, Par::Seq))),
			);

			assert!(A.adjoint() * &X ~ B);
		}
	}
}
