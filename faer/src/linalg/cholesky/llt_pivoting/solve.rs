use crate::assert;
use crate::internal_prelude::*;
use crate::perm::permute_rows;

pub fn solve_in_place_scratch<I: Index, T: ComplexField>(dim: usize, rhs_ncols: usize, par: Par) -> StackReq {
	_ = par;
	temp_mat_scratch::<T>(dim, rhs_ncols)
}

#[math]
#[track_caller]
pub fn solve_in_place_with_conj<I: Index, T: ComplexField>(
	L: MatRef<'_, T>,
	perm: PermRef<'_, I>,
	conj_lhs: Conj,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let n = L.nrows();
	let k = rhs.ncols();
	assert!(all(L.nrows() == n, L.ncols() == n, rhs.nrows() == n));

	let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
	let mut tmp = tmp.as_mat_mut();
	let mut rhs = rhs;

	permute_rows(tmp.rb_mut(), rhs.rb(), perm);
	linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(L, conj_lhs, tmp.rb_mut(), par);
	linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(L.transpose(), conj_lhs.compose(Conj::Yes), tmp.rb_mut(), par);
	permute_rows(rhs.rb_mut(), tmp.rb(), perm.inverse());
}

#[math]
#[track_caller]
pub fn solve_in_place<I: Index, T: ComplexField, C: Conjugate<Canonical = T>>(
	L: MatRef<'_, C>,
	perm: PermRef<'_, I>,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	solve_in_place_with_conj(L.canonical(), perm, Conj::get::<C>(), rhs, par, stack);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use linalg::cholesky::llt_pivoting;

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

		let A = &A * A.adjoint();
		let mut L = A.to_owned();
		let perm_fwd = &mut *vec![0usize; n];
		let perm_bwd = &mut *vec![0usize; n];

		let (_, perm) = llt_pivoting::factor::cholesky_in_place(
			L.as_mut(),
			perm_fwd,
			perm_bwd,
			Par::Seq,
			MemStack::new(&mut { MemBuffer::new(llt_pivoting::factor::cholesky_in_place_scratch::<usize, c64>(n, Par::Seq, default())) }),
			default(),
		)
		.unwrap();

		let approx_eq = CwiseMat(ApproxEq::eps() * 8.0 * (n as f64));

		{
			let mut X = B.to_owned();
			llt_pivoting::solve::solve_in_place(
				L.as_ref(),
				perm,
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(llt_pivoting::solve::solve_in_place_scratch::<usize, c64>(
					n,
					k,
					Par::Seq,
				))),
			);

			assert!(&A * &X ~ B);
		}

		{
			let mut X = B.to_owned();
			llt_pivoting::solve::solve_in_place(
				L.conjugate(),
				perm,
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(llt_pivoting::solve::solve_in_place_scratch::<usize, c64>(
					n,
					k,
					Par::Seq,
				))),
			);

			assert!(A.conjugate() * &X ~ B);
		}
	}
}
