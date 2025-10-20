use crate::assert;
use crate::internal_prelude::*;
use linalg::matmul::triangular::BlockStructure;

pub fn reconstruct_scratch<T: ComplexField>(dim: usize, par: Par) -> StackReq {
	_ = par;

	temp_mat_scratch::<T>(dim, dim)
}

#[track_caller]

pub fn reconstruct<T: ComplexField>(out: MatMut<'_, T>, L: MatRef<'_, T>, D: DiagRef<'_, T>, par: Par, stack: &mut MemStack) {
	let mut out = out;

	_ = stack;

	let n = out.nrows();

	assert!(all(out.nrows() == n, out.ncols() == n, L.nrows() == n, L.ncols() == n, D.dim() == n,));

	let (mut LxD, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };

	let mut LxD = LxD.as_mat_mut();

	{
		with_dim!(N, n);

		let mut LxD = LxD.rb_mut().as_shape_mut(N, N);

		let L = L.as_shape(N, N);

		let D = D.as_shape(N);

		for j in N.indices() {
			let d = &D[j].real();

			LxD[(j, j)] = d.to_cplx();

			for i in j.next().to(N.end()) {
				LxD[(i, j)] = L[(i, j)].mul_real(d);
			}
		}
	}

	let LxD = LxD.rb();

	linalg::matmul::triangular::matmul(
		out.rb_mut(),
		BlockStructure::TriangularLower,
		Accum::Replace,
		LxD,
		BlockStructure::TriangularLower,
		L.adjoint(),
		BlockStructure::UnitTriangularUpper,
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

	fn test_reconstruct() {
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
			MemStack::new(&mut MemBuffer::new(factor::cholesky_in_place_scratch::<c64>(n, Par::Seq, default()))),
			default(),
		)
		.unwrap();

		let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));

		let mut A_rec = Mat::zeros(n, n);

		reconstruct::reconstruct(
			A_rec.as_mut(),
			L.as_ref(),
			L.diagonal(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(reconstruct::reconstruct_scratch::<c64>(n, Par::Seq))),
		);

		for j in 0..n {
			for i in 0..j {
				A_rec[(i, j)] = A_rec[(j, i)].conj();
			}
		}

		assert!(A_rec ~ A);
	}
}
