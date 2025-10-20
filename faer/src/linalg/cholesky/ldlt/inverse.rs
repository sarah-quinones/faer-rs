use crate::assert;
use crate::internal_prelude::*;
use linalg::matmul::triangular::BlockStructure;

pub fn inverse_scratch<T: ComplexField>(dim: usize, par: Par) -> StackReq {
	_ = par;

	temp_mat_scratch::<T>(dim, dim)
}

#[track_caller]

pub fn inverse<T: ComplexField>(out: MatMut<'_, T>, L: MatRef<'_, T>, D: DiagRef<'_, T>, par: Par, stack: &mut MemStack) {
	let mut out = out;

	let n = out.nrows();

	assert!(all(out.nrows() == n, out.ncols() == n, L.nrows() == n, L.ncols() == n, D.dim() == n,));

	let (mut L_inv, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };

	let mut L_inv = L_inv.as_mat_mut();

	linalg::triangular_inverse::invert_unit_lower_triangular(L_inv.rb_mut(), L, par);

	{
		with_dim!(N, n);

		let mut L_inv = L_inv.rb_mut().as_shape_mut(N, N);

		let D = D.as_shape(N);

		for j in N.indices() {
			let d = D[j].real().recip();

			L_inv[(j, j)] = d.to_cplx();
		}

		for j in N.indices() {
			for i in j.next().to(N.end()) {
				let d = L_inv[(i, i)].real();

				L_inv[(j, i)] = L_inv[(i, j)].conj().mul_real(&d);
			}
		}
	}

	let L_inv = L_inv.rb();

	linalg::matmul::triangular::matmul(
		out.rb_mut(),
		BlockStructure::TriangularLower,
		Accum::Replace,
		L_inv,
		BlockStructure::TriangularUpper,
		L_inv,
		BlockStructure::UnitTriangularLower,
		one(),
		par,
	);
}

#[cfg(test)]

mod tests {

	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use linalg::cholesky::ldlt::*;

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

		let A = &A * A.adjoint();

		let mut L = A.to_owned();

		factor::cholesky_in_place(
			L.as_mut(),
			Default::default(),
			Par::Seq,
			MemStack::new(&mut { MemBuffer::new(factor::cholesky_in_place_scratch::<c64>(n, Par::Seq, default())) }),
			default(),
		)
		.unwrap();

		let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));

		let mut A_inv = Mat::zeros(n, n);

		inverse::inverse(
			A_inv.as_mut(),
			L.as_ref(),
			L.diagonal(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(inverse::inverse_scratch::<c64>(n, Par::Seq))),
		);

		for j in 0..n {
			for i in 0..j {
				A_inv[(i, j)] = A_inv[(j, i)].conj();
			}
		}

		assert!(A_inv * A ~ Mat::identity(n, n));
	}
}
