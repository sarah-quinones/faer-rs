use crate::assert;
use crate::internal_prelude::*;
pub fn solve_lstsq_in_place_scratch<T: ComplexField>(
	qr_nrows: usize,
	qr_ncols: usize,
	qr_block_size: usize,
	rhs_ncols: usize,
	par: Par,
) -> StackReq {
	_ = qr_ncols;
	_ = par;
	linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(qr_nrows, qr_block_size, rhs_ncols)
}
pub fn solve_in_place_scratch<T: ComplexField>(
	qr_dim: usize,
	qr_block_size: usize,
	rhs_ncols: usize,
	par: Par,
) -> StackReq {
	solve_lstsq_in_place_scratch::<T>(
		qr_dim,
		qr_dim,
		qr_block_size,
		rhs_ncols,
		par,
	)
}
pub fn solve_transpose_in_place_scratch<T: ComplexField>(
	qr_dim: usize,
	qr_block_size: usize,
	rhs_ncols: usize,
	par: Par,
) -> StackReq {
	_ = par;
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(qr_dim, qr_block_size, rhs_ncols)
}
#[track_caller]
pub fn solve_lstsq_in_place_with_conj<T: ComplexField>(
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	R: MatRef<'_, T>,
	conj_QR: Conj,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let m = Q_basis.nrows();
	let n = Q_basis.ncols();
	let size = Ord::min(m, n);
	let block_size = Q_coeff.nrows();
	assert!(all(
		block_size > 0,
		rhs.nrows() == m,
		Q_basis.nrows() >= Q_basis.ncols(),
		Q_coeff.ncols() == size,
		R.nrows() >= size,
		R.ncols() == n,
	));
	let mut rhs = rhs;
	let mut stack = stack;
	linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
		Q_basis,
		Q_coeff,
		conj_QR.compose(Conj::Yes),
		rhs.rb_mut(),
		par,
		stack.rb_mut(),
	);
	linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(
		R.get(..size, ..),
		conj_QR,
		rhs.subrows_mut(0, size),
		par,
	);
}
#[track_caller]
pub fn solve_lstsq_in_place<T: ComplexField, C: Conjugate<Canonical = T>>(
	Q_basis: MatRef<'_, C>,
	Q_coeff: MatRef<'_, C>,
	R: MatRef<'_, C>,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	solve_lstsq_in_place_with_conj(
		Q_basis.canonical(),
		Q_coeff.canonical(),
		R.canonical(),
		Conj::get::<C>(),
		rhs,
		par,
		stack,
	);
}
#[track_caller]
pub fn solve_in_place_with_conj<T: ComplexField>(
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	R: MatRef<'_, T>,
	conj_QR: Conj,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let n = Q_basis.nrows();
	let block_size = Q_coeff.nrows();
	assert!(all(
		block_size > 0,
		rhs.nrows() == n,
		Q_basis.nrows() == n,
		Q_basis.ncols() == n,
		Q_coeff.ncols() == n,
		R.nrows() == n,
		R.ncols() == n,
	));
	solve_lstsq_in_place_with_conj(
		Q_basis, Q_coeff, R, conj_QR, rhs, par, stack,
	);
}
#[track_caller]
pub fn solve_in_place<T: ComplexField, C: Conjugate<Canonical = T>>(
	Q_basis: MatRef<'_, C>,
	Q_coeff: MatRef<'_, C>,
	R: MatRef<'_, C>,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	solve_in_place_with_conj(
		Q_basis.canonical(),
		Q_coeff.canonical(),
		R.canonical(),
		Conj::get::<C>(),
		rhs,
		par,
		stack,
	);
}
#[track_caller]
pub fn solve_transpose_in_place_with_conj<T: ComplexField>(
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	R: MatRef<'_, T>,
	conj_QR: Conj,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let n = Q_basis.nrows();
	let block_size = Q_coeff.nrows();
	assert!(all(
		block_size > 0,
		rhs.nrows() == n,
		Q_basis.nrows() == n,
		Q_basis.ncols() == n,
		Q_coeff.ncols() == n,
		R.nrows() == n,
		R.ncols() == n,
	));
	let mut rhs = rhs;
	let mut stack = stack;
	linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(
		R.transpose(),
		conj_QR,
		rhs.rb_mut(),
		par,
	);
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
		Q_basis,
		Q_coeff,
		conj_QR.compose(Conj::Yes),
		rhs.rb_mut(),
		par,
		stack.rb_mut(),
	);
}
#[track_caller]
pub fn solve_transpose_in_place<
	T: ComplexField,
	C: Conjugate<Canonical = T>,
>(
	Q_basis: MatRef<'_, C>,
	Q_coeff: MatRef<'_, C>,
	R: MatRef<'_, C>,
	rhs: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	solve_transpose_in_place_with_conj(
		Q_basis.canonical(),
		Q_coeff.canonical(),
		R.canonical(),
		Conj::get::<C>(),
		rhs,
		par,
		stack,
	);
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use linalg::qr::no_pivoting::*;
	#[test]
	fn test_lstsq() {
		let rng = &mut StdRng::seed_from_u64(0);
		let m = 100;
		let n = 50;
		let k = 3;
		let A = CwiseMatDistribution {
			nrows: m,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);
		let B = CwiseMatDistribution {
			nrows: m,
			ncols: k,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);
		let mut QR = A.to_owned();
		let mut H = Mat::zeros(4, n);
		factor::qr_in_place(
			QR.as_mut(),
			H.as_mut(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(factor::qr_in_place_scratch::<
				c64,
			>(
				m, n, 4, Par::Seq, default()
			))),
			default(),
		);
		let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));
		{
			let mut X = B.to_owned();
			solve::solve_lstsq_in_place(
				QR.as_ref(),
				H.as_ref(),
				QR.as_ref(),
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					solve::solve_lstsq_in_place_scratch::<c64>(
						m,
						n,
						4,
						k,
						Par::Seq,
					),
				)),
			);
			let X = X.get(..n, ..);
			assert!(A.adjoint() * & A * & X ~ A.adjoint() * & B);
		}
		{
			let mut X = B.to_owned();
			solve::solve_lstsq_in_place(
				QR.conjugate(),
				H.conjugate(),
				QR.conjugate(),
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					solve::solve_lstsq_in_place_scratch::<c64>(
						m,
						n,
						4,
						k,
						Par::Seq,
					),
				)),
			);
			let X = X.get(..n, ..);
			assert!(A.transpose() * A.conjugate() * & X ~ A.transpose() * & B);
		}
	}
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
		let mut QR = A.to_owned();
		let mut H = Mat::zeros(4, n);
		factor::qr_in_place(
			QR.as_mut(),
			H.as_mut(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(factor::qr_in_place_scratch::<
				c64,
			>(
				n, n, 4, Par::Seq, default()
			))),
			default(),
		);
		let approx_eq = CwiseMat(ApproxEq::eps() * (n as f64));
		{
			let mut X = B.to_owned();
			solve::solve_in_place(
				QR.as_ref(),
				H.as_ref(),
				QR.as_ref(),
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					solve::solve_in_place_scratch::<c64>(n, 4, k, Par::Seq),
				)),
			);
			assert!(& A * & X ~ B);
		}
		{
			let mut X = B.to_owned();
			solve::solve_in_place(
				QR.conjugate(),
				H.conjugate(),
				QR.conjugate(),
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					solve::solve_in_place_scratch::<c64>(n, 4, k, Par::Seq),
				)),
			);
			assert!(A.conjugate() * & X ~ B);
		}
		{
			let mut X = B.to_owned();
			solve::solve_transpose_in_place(
				QR.as_ref(),
				H.as_ref(),
				QR.as_ref(),
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					solve::solve_transpose_in_place_scratch::<c64>(
						n,
						4,
						k,
						Par::Seq,
					),
				)),
			);
			assert!(A.transpose() * & X ~ B);
		}
		{
			let mut X = B.to_owned();
			solve::solve_transpose_in_place(
				QR.conjugate(),
				H.conjugate(),
				QR.conjugate(),
				X.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					solve::solve_transpose_in_place_scratch::<c64>(
						n,
						4,
						k,
						Par::Seq,
					),
				)),
			);
			assert!(A.adjoint() * & X ~ B);
		}
	}
}
