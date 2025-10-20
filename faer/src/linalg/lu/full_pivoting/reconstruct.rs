use crate::assert;
use crate::internal_prelude::*;
use linalg::matmul::triangular::BlockStructure;

pub fn reconstruct_scratch<I: Index, T: ComplexField>(nrows: usize, ncols: usize, par: Par) -> StackReq {
	_ = par;

	temp_mat_scratch::<T>(nrows, ncols)
}

#[track_caller]

pub fn reconstruct<I: Index, T: ComplexField>(
	out: MatMut<'_, T>,
	L: MatRef<'_, T>,
	U: MatRef<'_, T>,
	row_perm: PermRef<'_, I>,
	col_perm: PermRef<'_, I>,
	par: Par,
	stack: &mut MemStack,
) {
	let m = L.nrows();

	let n = U.ncols();

	let size = Ord::min(m, n);

	assert!(all(out.nrows() == m, out.ncols() == n, row_perm.len() == m, col_perm.len() == n,));

	let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(m, n, stack) };

	let mut tmp = tmp.as_mat_mut();

	let mut out = out;

	linalg::matmul::triangular::matmul(
		tmp.rb_mut().get_mut(..size, ..size),
		BlockStructure::Rectangular,
		Accum::Replace,
		L.get(..size, ..size),
		BlockStructure::UnitTriangularLower,
		U.get(..size, ..size),
		BlockStructure::TriangularUpper,
		one(),
		par,
	);

	if m > n {
		linalg::matmul::triangular::matmul(
			tmp.rb_mut().get_mut(size.., ..size),
			BlockStructure::Rectangular,
			Accum::Replace,
			L.get(size.., ..size),
			BlockStructure::Rectangular,
			U.get(..size, ..size),
			BlockStructure::TriangularUpper,
			one(),
			par,
		);
	}

	if m < n {
		linalg::matmul::triangular::matmul(
			tmp.rb_mut().get_mut(..size, size..),
			BlockStructure::Rectangular,
			Accum::Replace,
			L.get(..size, ..size),
			BlockStructure::UnitTriangularLower,
			U.get(..size, size..),
			BlockStructure::Rectangular,
			one(),
			par,
		);
	}

	with_dim!(M, m);

	with_dim!(N, n);

	let row_perm = row_perm.as_shape(M).bound_arrays().1;

	let col_perm = col_perm.as_shape(N).bound_arrays().1;

	let tmp = tmp.rb().as_shape(M, N);

	let mut out = out.rb_mut().as_shape_mut(M, N);

	for j in N.indices() {
		for i in M.indices() {
			out[(i, j)] = tmp[(row_perm[i].zx(), col_perm[j].zx())].clone();
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
	use linalg::lu::full_pivoting::*;

	#[test]

	fn test_reconstruct() {
		let rng = &mut StdRng::seed_from_u64(0);

		for (m, n) in [(100, 50), (50, 100)] {
			let A = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			let mut LU = A.to_owned();

			let row_perm_fwd = &mut *vec![0usize; m];

			let row_perm_bwd = &mut *vec![0usize; m];

			let col_perm_fwd = &mut *vec![0usize; n];

			let col_perm_bwd = &mut *vec![0usize; n];

			let (_, row_perm, col_perm) = factor::lu_in_place(
				LU.as_mut(),
				row_perm_fwd,
				row_perm_bwd,
				col_perm_fwd,
				col_perm_bwd,
				Par::Seq,
				MemStack::new(&mut { MemBuffer::new(factor::lu_in_place_scratch::<usize, c64>(m, n, Par::Seq, default())) }),
				default(),
			);

			let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));

			let mut A_rec = Mat::zeros(m, n);

			reconstruct::reconstruct(
				A_rec.as_mut(),
				LU.as_ref(),
				LU.as_ref(),
				row_perm,
				col_perm,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(reconstruct::reconstruct_scratch::<usize, c64>(m, n, Par::Seq))),
			);

			assert!(A_rec ~ A);
		}
	}
}
