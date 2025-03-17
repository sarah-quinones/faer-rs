use crate::assert;
use crate::internal_prelude::*;
use linalg::matmul::triangular::BlockStructure;

pub fn reconstruct_scratch<I: Index, T: ComplexField>(dim: usize, par: Par) -> StackReq {
	_ = (dim, par);
	temp_mat_scratch::<T>(dim, dim)
}

#[track_caller]
#[math]
pub fn reconstruct<I: Index, T: ComplexField>(out: MatMut<'_, T>, L: MatRef<'_, T>, perm: PermRef<'_, I>, par: Par, stack: &mut MemStack) {
	let mut out = out;
	let n = out.nrows();

	assert!(all(out.nrows() == n, out.ncols() == n, L.nrows() == n, L.ncols() == n,));

	let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut tmp = tmp.as_mat_mut();

	linalg::matmul::triangular::matmul(
		tmp.rb_mut(),
		BlockStructure::TriangularLower,
		Accum::Replace,
		L,
		BlockStructure::TriangularLower,
		L.adjoint(),
		BlockStructure::TriangularUpper,
		one(),
		par,
	);

	let p = perm.arrays().1;

	for j in 0..n {
		let jj = p[j].zx();
		for i in j..n {
			let ii = p[i].zx();

			if ii >= jj {
				out[(i, j)] = copy(tmp[(ii, jj)]);
			} else {
				out[(i, j)] = conj(tmp[(jj, ii)]);
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use linalg::cholesky::llt_pivoting::*;

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
		let perm_fwd = &mut *vec![0usize; n];
		let perm_bwd = &mut *vec![0usize; n];

		let (_, perm) = factor::cholesky_in_place(
			L.as_mut(),
			perm_fwd,
			perm_bwd,
			Par::Seq,
			MemStack::new(&mut { MemBuffer::new(factor::cholesky_in_place_scratch::<usize, c64>(n, Par::Seq, default())) }),
			default(),
		)
		.unwrap();

		let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));

		let mut A_rec = Mat::zeros(n, n);
		reconstruct::reconstruct(
			A_rec.as_mut(),
			L.as_ref(),
			perm,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(reconstruct::reconstruct_scratch::<usize, c64>(n, Par::Seq))),
		);

		for j in 0..n {
			for i in 0..j {
				A_rec[(i, j)] = A_rec[(j, i)].conj();
			}
		}

		assert!(A_rec ~ A);
	}
}
