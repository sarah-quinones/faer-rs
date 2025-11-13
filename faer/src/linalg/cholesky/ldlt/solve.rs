use crate::assert;
use crate::internal_prelude::*;
pub fn solve_in_place_scratch<T: ComplexField>(dim: usize, rhs_ncols: usize, par: Par) -> StackReq {
	_ = (dim, rhs_ncols, par);
	StackReq::EMPTY
}
#[track_caller]
pub fn solve_in_place_with_conj<T: ComplexField>(
	L: MatRef<'_, T>,
	D: DiagRef<'_, T>,
	conj_lhs: Conj,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let n = L.nrows();
	_ = stack;
	assert!(all(L.nrows() == n, L.ncols() == n, D.dim() == n, rhs.nrows() == n,));
	let mut rhs = rhs;
	linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(L, conj_lhs, rhs.rb_mut(), par);
	{
		with_dim!(N, rhs.nrows());
		with_dim!(K, rhs.ncols());
		let D = D.as_shape(N);
		let mut rhs = rhs.rb_mut().as_shape_mut(N, K);
		for j in K.indices() {
			for i in N.indices() {
				let d = &D[i].real().recip();
				rhs[(i, j)] = rhs[(i, j)].mul_real(d);
			}
		}
	}
	linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(L.transpose(), conj_lhs.compose(Conj::Yes), rhs.rb_mut(), par);
}
#[track_caller]
pub fn solve_in_place<T: ComplexField, C: Conjugate<Canonical = T>>(
	L: MatRef<'_, C>,
	D: DiagRef<'_, C>,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	solve_in_place_with_conj(L.canonical(), D.canonical(), Conj::get::<C>(), rhs, par, stack);
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use linalg::cholesky::ldlt;
	#[test]
	fn test_solve() {
		let rng = &mut StdRng::seed_from_u64(0);
		for n in [50, 200, 400] {
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
			ldlt::factor::cholesky_in_place(
				L.as_mut(),
				Default::default(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(ldlt::factor::cholesky_in_place_scratch::<c64>(
					n,
					Par::Seq,
					default(),
				))),
				default(),
			)
			.unwrap();
			let approx_eq = CwiseMat(ApproxEq::eps() * 8.0 * (n as f64));
			{
				let mut X = B.to_owned();
				ldlt::solve::solve_in_place(
					L.as_ref(),
					L.diagonal(),
					X.as_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(ldlt::solve::solve_in_place_scratch::<c64>(n, k, Par::Seq))),
				);
				assert!(& A * & X ~ B);
			}
			{
				let mut X = B.to_owned();
				ldlt::solve::solve_in_place(
					L.conjugate(),
					L.conjugate().diagonal(),
					X.as_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(ldlt::solve::solve_in_place_scratch::<c64>(n, k, Par::Seq))),
				);
				assert!(A.conjugate() * & X ~ B);
			}
		}
	}
}
