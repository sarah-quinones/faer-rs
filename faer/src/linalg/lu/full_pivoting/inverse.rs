use crate::assert;
use crate::internal_prelude::*;
use linalg::matmul::triangular::BlockStructure;

pub fn inverse_scratch<I: Index, T: ComplexField>(dim: usize, par: Par) -> StackReq {
	_ = par;

	temp_mat_scratch::<T>(dim, dim)
}

#[track_caller]

pub fn inverse<I: Index, T: ComplexField>(
	out: MatMut<'_, T>,
	L: MatRef<'_, T>,
	U: MatRef<'_, T>,
	row_perm: PermRef<'_, I>,
	col_perm: PermRef<'_, I>,
	par: Par,
	stack: &mut MemStack,
) {
	let n = L.ncols();

	assert!(all(
		L.nrows() == n,
		L.ncols() == n,
		U.nrows() == n,
		U.ncols() == n,
		out.nrows() == n,
		out.ncols() == n,
		row_perm.len() == n,
	));

	let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };

	let mut tmp = tmp.as_mat_mut();

	let mut out = out;

	linalg::triangular_inverse::invert_unit_lower_triangular(out.rb_mut(), L, par);

	linalg::triangular_inverse::invert_upper_triangular(out.rb_mut(), U, par);

	linalg::matmul::triangular::matmul(
		tmp.rb_mut(),
		BlockStructure::Rectangular,
		Accum::Replace,
		out.rb(),
		BlockStructure::TriangularUpper,
		out.rb(),
		BlockStructure::UnitTriangularLower,
		one(),
		par,
	);

	with_dim!(N, n);

	let (row_perm, col_perm) = (col_perm.as_shape(N).bound_arrays().1, row_perm.as_shape(N).bound_arrays().1);

	let tmp = tmp.rb().as_shape(N, N);

	let mut out = out.rb_mut().as_shape_mut(N, N);

	for j in N.indices() {
		for i in N.indices() {
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

	fn test_inverse() {
		let rng = &mut StdRng::seed_from_u64(0);

		let n = 50;

		let A = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);

		let mut LU = A.to_owned();

		let row_perm_fwd = &mut *vec![0usize; n];

		let row_perm_bwd = &mut *vec![0usize; n];

		let col_perm_fwd = &mut *vec![0usize; n];

		let col_perm_bwd = &mut *vec![0usize; n];

		let (_, row_perm, col_perm) = factor::lu_in_place(
			LU.as_mut(),
			row_perm_fwd,
			row_perm_bwd,
			col_perm_fwd,
			col_perm_bwd,
			Par::Seq,
			MemStack::new(&mut { MemBuffer::new(factor::lu_in_place_scratch::<usize, c64>(n, n, Par::Seq, default())) }),
			default(),
		);

		let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));

		let mut A_inv = Mat::zeros(n, n);

		inverse::inverse(
			A_inv.as_mut(),
			LU.as_ref(),
			LU.as_ref(),
			row_perm,
			col_perm,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(inverse::inverse_scratch::<usize, c64>(n, Par::Seq))),
		);

		assert!(& A_inv * & A ~ Mat::identity(n, n));
	}
}
