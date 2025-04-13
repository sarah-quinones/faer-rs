use crate::assert;
use crate::internal_prelude::*;
use linalg::matmul::triangular::BlockStructure;

pub fn reconstruct_scratch<I: Index, T: ComplexField>(dim: usize, par: Par) -> StackReq {
	_ = par;
	temp_mat_scratch::<T>(dim, dim)
}

#[track_caller]
#[math]
pub fn reconstruct<I: Index, T: ComplexField>(
	out: MatMut<'_, T>,
	L: MatRef<'_, T>,
	diagonal: DiagRef<'_, T>,
	subdiagonal: DiagRef<'_, T>,
	perm: PermRef<'_, I>,
	par: Par,
	stack: &mut MemStack,
) {
	let n = L.nrows();
	assert!(all(
		out.nrows() == n,
		out.ncols() == n,
		L.nrows() == n,
		L.ncols() == n,
		diagonal.dim() == n,
		subdiagonal.dim() == n,
		perm.len() == n,
	));

	let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut tmp = tmp.as_mat_mut();
	let mut out = out;
	let s = subdiagonal;

	out.fill(zero());
	out.rb_mut().diagonal_mut().fill(one());
	out.copy_from_strict_triangular_lower(L);

	let mut j = 0;
	while j < n {
		if s[j] == zero() {
			let d = real(L[(j, j)]);

			for i in 0..n {
				out[(i, j)] = mul_real(out[(i, j)], d);
			}

			j += 1;
		} else {
			let akp1k = copy(s[j]);
			let ak = real(L[(j, j)]);
			let akp1 = real(L[(j + 1, j + 1)]);

			for i in 0..n {
				let xk = copy(out[(i, j)]);
				let xkp1 = copy(out[(i, j + 1)]);

				out[(i, j)] = mul_real(xk, ak) + (xkp1 * akp1k);
				out[(i, j + 1)] = mul_real(xkp1, akp1) + (xk * conj(akp1k));
			}

			j += 2;
		}
	}

	linalg::matmul::triangular::matmul(
		tmp.rb_mut(),
		BlockStructure::TriangularLower,
		Accum::Replace,
		L,
		BlockStructure::UnitTriangularLower,
		out.rb().adjoint(),
		BlockStructure::Rectangular,
		one(),
		par,
	);

	let perm_inv = perm.arrays().1;
	for j in 0..n {
		let pj = perm_inv[j].zx();
		for i in j..n {
			let pi = perm_inv[i].zx();

			out[(i, j)] = if pi >= pj { copy(tmp[(pi, pj)]) } else { conj(tmp[(pj, pi)]) };
		}
	}

	for j in 0..n {
		out[(j, j)] = from_real(real(out[(j, j)]));
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use linalg::cholesky::lblt::*;

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

		let A = &A + A.adjoint();
		let mut LB = A.to_owned();
		let mut subdiag = Diag::zeros(n);
		let perm_fwd = &mut *vec![0usize; n];
		let perm_bwd = &mut *vec![0usize; n];

		let (_, perm) = factor::cholesky_in_place(
			LB.as_mut(),
			subdiag.as_mut(),
			perm_fwd,
			perm_bwd,
			Par::Seq,
			MemStack::new(&mut { MemBuffer::new(factor::cholesky_in_place_scratch::<usize, c64>(n, Par::Seq, default())) }),
			default(),
		);

		let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));

		let mut A_rec = Mat::zeros(n, n);
		reconstruct::reconstruct(
			A_rec.as_mut(),
			LB.as_ref(),
			LB.diagonal(),
			subdiag.as_ref(),
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
