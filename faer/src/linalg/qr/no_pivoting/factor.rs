use crate::assert;
use crate::internal_prelude::*;
use linalg::householder::{self};

// R R R R|R R R | A A A
// H R R R|R R R | A A A
// H H 0 R|R R R | A A A
//        |------| A A A
// H H H 0|R R R | A A A
// H H H H|0 R R | A A A

/// information about the resulting $QR$ factorization.
#[derive(Copy, Clone, Debug)]
pub struct QrInfo {
	/// estimated rank of the matrix.
	pub rank: usize,
}

#[math]
fn qr_in_place_unblocked<T: ComplexField>(A: MatMut<'_, T>, H: RowMut<'_, T>, row_start: usize, col_start: usize) -> usize {
	let mut A = A;
	let mut H = H;

	let (m, n) = A.shape();

	let mut col = col_start;
	let mut row = row_start;

	while row < m && col < n {
		let norm = A.rb().col(col).get(..row).norm_l2();

		let (mut A00, A01, A10, A11) = A.rb_mut().split_at_mut(row + 1, col + 1);
		let (mut A10l, A10r) = A10.split_at_col_mut(col);
		let mut A10r = A10r.col_mut(0);

		let A01 = A01.row_mut(row);
		let A00 = &mut A00[(row, col)];

		let (info, v) = if row == col {
			let info = householder::make_householder_in_place(A00, A10r.rb_mut());
			(info, A10r.rb())
		} else {
			let info = householder::make_householder_out_of_place(A00, A10l.rb_mut().col_mut(row), A10r.rb());
			let nrows = A10r.nrows();

			A10r.rb_mut().get_mut(..Ord::min(nrows, col - row)).fill(zero());
			(info, A10l.rb().col(row))
		};

		let norm = hypot(info.norm, norm);
		let eps = eps::<T::Real>();
		let leeway = from_f64::<T::Real>((m - row) as f64 * 16.0);
		let threshold = eps * leeway * norm;

		if info.norm > threshold {
			let tau_inv = recip(info.tau);
			H[row] = from_real(info.tau);

			for (head, tail) in core::iter::zip(A01.iter_mut(), A11.col_iter_mut()) {
				let dot = *head + linalg::matmul::dot::inner_prod(v.transpose(), Conj::Yes, tail.rb(), Conj::No);
				let k = -mul_real(dot, tau_inv);
				*head = *head + k;
				z!(tail, v).for_each(|uz!(dst, src)| {
					*dst = *dst + k * *src;
				});
			}
			row += 1;
		}
		col += 1;
	}

	row
}

/// the recommended block size to use for a $QR$ decomposition of a matrix with the given shape.
#[inline]
pub fn recommended_blocksize<T: ComplexField>(nrows: usize, ncols: usize) -> usize {
	let prod = nrows * ncols;
	let size = nrows.min(ncols);

	(if prod > 8192 * 8192 {
		256
	} else if prod > 2048 * 2048 {
		128
	} else if prod > 1024 * 1024 {
		64
	} else if prod > 512 * 512 {
		48
	} else if prod > 128 * 128 {
		32
	} else if prod > 32 * 32 {
		8
	} else if prod > 16 * 16 {
		4
	} else {
		1
	})
	.min(size)
	.max(1)
}

/// $QR$ factorization tuning parameters.
#[derive(Debug, Copy, Clone)]
pub struct QrParams {
	/// threshold at which blocking algorithms should be disabled
	pub blocking_threshold: usize,
	/// threshold at which the parallelism should be disabled
	pub par_threshold: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for QrParams {
	#[inline]
	fn auto() -> Self {
		Self {
			blocking_threshold: 48 * 48,
			par_threshold: 192 * 256,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

#[math]
fn qr_in_place_blocked<T: ComplexField>(
	A: MatMut<'_, T>,
	H: MatMut<'_, T>,
	row_start: usize,
	col_start: usize,
	par: Par,
	stack: &mut MemStack,
	params: Spec<QrParams, T>,
) -> usize {
	let params = params.config;

	let (m, n) = A.shape();
	let size = Ord::min(m, n);
	let blocksize = H.nrows();

	assert!(blocksize > 0);

	if blocksize == 1 {
		return qr_in_place_unblocked(A, H.row_mut(0), row_start, col_start);
	}
	let sub_blocksize = if m * n < params.blocking_threshold { 1 } else { blocksize / 2 };

	let mut A = A;
	let mut H = H;

	let mut col = col_start;
	let mut row = row_start;

	while row < size && col < n {
		let blocksize = Ord::min(blocksize, Ord::min(size - row, n - col));
		let sub_blocksize = Ord::min(blocksize, sub_blocksize);

		let mut A = A.rb_mut();
		let mut H = H.rb_mut();

		//
		//  offset col_start
		//       v v
		// R R R R|A A|A A
		// H 0 R R|A A|A A
		// H H 0 R|A A|A A
		// H H H 0|R A|A A
		// H H H H|0 R|A A
		// H H H H|H 0|A A

		// householder coeffs
		//
		// H H H H H 0 0 0 ∞ 0 0 0
		//   H H H   ∞ 0 0   ∞ 0 0
		//     H H     ∞ 0     ∞ 0
		//       H       ∞       ∞

		let new_row = qr_in_place_blocked(
			A.rb_mut().subcols_mut(0, col + blocksize),
			H.rb_mut().subrows_mut(0, sub_blocksize),
			row,
			col,
			par,
			stack,
			params.into(),
		);
		let offset = new_row - row;

		let mut k = 0;
		while k < offset {
			let sub_blocksize = Ord::min(sub_blocksize, offset - k);

			if k > 0 {
				let mut H = H.rb_mut().subcols_mut(row + k, sub_blocksize);

				let (H0, H1) = H.rb_mut().split_at_row_mut(k);
				let H0 = H0.rb().subrows(0, sub_blocksize);
				let H1 = H1.subrows_mut(0, sub_blocksize);

				{ H1 }.copy_from_triangular_upper(H0);
			}
			k += sub_blocksize;
		}

		let (Q0, A1) = A.rb_mut().get_mut(row.., ..).split_at_col_mut(col + blocksize);
		let Q0 = Q0.rb().get(.., row..row + offset);
		let mut H = H.rb_mut().get_mut(..offset, row..row + offset);

		householder::upgrade_householder_factor(H.rb_mut(), Q0, blocksize, sub_blocksize, par);
		if A1.ncols() > 0 {
			householder::apply_block_householder_transpose_on_the_left_in_place_with_conj(Q0, H.rb(), Conj::Yes, A1, par, stack)
		};

		col += blocksize;
		row += offset;
	}
	row
}

#[track_caller]
pub fn qr_in_place<T: ComplexField>(A: MatMut<'_, T>, Q_coeff: MatMut<'_, T>, par: Par, stack: &mut MemStack, params: Spec<QrParams, T>) -> QrInfo {
	let blocksize = Q_coeff.nrows();
	assert!(all(blocksize > 0, Q_coeff.ncols() == Ord::min(A.nrows(), A.ncols()),));

	#[cfg(feature = "perf-warn")]
	if A.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(QR_WARN) {
		if A.col_stride().unsigned_abs() == 1 {
			log::warn!(target: "faer_perf", "QR prefers column-major matrix. Found row-major matrix.");
		} else {
			log::warn!(target: "faer_perf", "QR prefers column-major matrix. Found matrix with generic strides.");
		}
	}

	let mut Q_coeff = Q_coeff;
	let rank = qr_in_place_blocked(A, Q_coeff.rb_mut(), 0, 0, par, stack, params);
	Q_coeff.rb_mut().get_mut(.., rank..).fill(zero());

	let mut col = rank / blocksize * blocksize;
	let n = Q_coeff.ncols();
	while col < n {
		let blocksize = Ord::min(blocksize, n - col);

		let start = Ord::max(rank, col);

		Q_coeff
			.rb_mut()
			.get_mut(start - col.., start..col + blocksize)
			.diagonal_mut()
			.fill(infinity());

		col += blocksize;
	}

	QrInfo { rank }
}

/// computes the size and alignment of required workspace for performing a qr
/// decomposition with no pivoting
#[inline]
pub fn qr_in_place_scratch<T: ComplexField>(nrows: usize, ncols: usize, blocksize: usize, par: Par, params: Spec<QrParams, T>) -> StackReq {
	let _ = par;
	let _ = nrows;
	let _ = &params;
	temp_mat_scratch::<T>(blocksize, ncols)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, Row, assert, c64};
	use dyn_stack::MemBuffer;

	#[test]
	fn test_qr() {
		let rng = &mut StdRng::seed_from_u64(0);
		for rank in [1, 2, 3, 4, 5, 100, usize::MAX] {
			for par in [Par::Seq, Par::rayon(8)] {
				for n in [2, 4, 8, 16, 24, 32, 127, 128, 257] {
					let rank = Ord::min(n, rank);

					let approx_eq = CwiseMat(ApproxEq {
						abs_tol: 1e-10,
						rel_tol: 1e-10,
					});

					let A0 = CwiseMatDistribution {
						nrows: n,
						ncols: rank,
						dist: ComplexDistribution::new(StandardNormal, StandardNormal),
					}
					.rand::<Mat<c64>>(rng);
					let A1 = CwiseMatDistribution {
						nrows: rank,
						ncols: n,
						dist: ComplexDistribution::new(StandardNormal, StandardNormal),
					}
					.rand::<Mat<c64>>(rng);

					let A = &A0 * &A1;
					let A = A.as_ref();

					let mut QR = A.cloned();
					let mut H = Row::zeros(n);

					let mut params: QrParams = auto!(c64);
					params.blocking_threshold = usize::MAX;

					let params = params.into();
					let computed_rank = qr_in_place(
						QR.as_mut(),
						H.as_mat_mut(),
						Par::Seq,
						MemStack::new(&mut MemBuffer::new(qr_in_place_scratch::<c64>(n, n, 1, Par::Seq, params))),
						params,
					);
					assert!(computed_rank.rank == rank);

					let mut Q = Mat::<c64>::zeros(n, n);
					let mut R = QR.as_ref().cloned();

					for j in 0..n {
						Q[(j, j)] = c64::ONE;
					}

					householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
						QR.as_ref(),
						H.as_mat(),
						Conj::No,
						Q.as_mut(),
						Par::Seq,
						MemStack::new(&mut MemBuffer::new(
							householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<c64>(n, 1, n),
						)),
					);

					for j in 0..n {
						for i in j + 1..n {
							R[(i, j)] = c64::ZERO;
						}
					}

					assert!(Q * R ~ A);
				}

				for n in [2, 3, 4, 8, 16, 24, 32, 128, 255, 256, 257, 512] {
					let bs = 15;
					let rank = Ord::min(n, rank);

					let approx_eq = CwiseMat(ApproxEq {
						abs_tol: 1e-10,
						rel_tol: 1e-10,
					});

					let A0 = CwiseMatDistribution {
						nrows: n,
						ncols: rank,
						dist: ComplexDistribution::new(StandardNormal, StandardNormal),
					}
					.rand::<Mat<c64>>(rng);
					let A1 = CwiseMatDistribution {
						nrows: rank,
						ncols: n,
						dist: ComplexDistribution::new(StandardNormal, StandardNormal),
					}
					.rand::<Mat<c64>>(rng);

					let A = &A0 * &A1;
					let A = A.as_ref();
					let mut QR = A.cloned();
					let mut H = Mat::zeros(bs, n);

					let computed_rank = qr_in_place(
						QR.as_mut(),
						H.as_mut(),
						par,
						MemStack::new(&mut MemBuffer::new(qr_in_place_scratch::<c64>(n, n, bs, par, default()))),
						default(),
					);
					assert!(computed_rank.rank == rank);

					let mut Q = Mat::<c64>::zeros(n, n);
					let mut R = QR.as_ref().cloned();

					for j in 0..n {
						Q[(j, j)] = c64::ONE;
					}

					householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
						QR.as_ref(),
						H.as_ref(),
						Conj::No,
						Q.as_mut(),
						Par::Seq,
						MemStack::new(&mut MemBuffer::new(
							householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<c64>(n, bs, n),
						)),
					);

					for j in 0..n {
						for i in j + 1..n {
							R[(i, j)] = c64::ZERO;
						}
					}

					assert!(Q * R ~ A);
				}

				let n = 20;
				for m in [2, 3, 4, 8, 16, 24, 32, 128, 255, 256, 257, 512] {
					let size = Ord::min(m, n);
					let bs = 15;
					let rank = Ord::min(size, rank);

					let approx_eq = CwiseMat(ApproxEq {
						abs_tol: 1e-10,
						rel_tol: 1e-10,
					});

					let A0 = CwiseMatDistribution {
						nrows: m,
						ncols: rank,
						dist: ComplexDistribution::new(StandardNormal, StandardNormal),
					}
					.rand::<Mat<c64>>(rng);
					let A1 = CwiseMatDistribution {
						nrows: rank,
						ncols: n,
						dist: ComplexDistribution::new(StandardNormal, StandardNormal),
					}
					.rand::<Mat<c64>>(rng);

					let A = &A0 * &A1;
					let A = A.as_ref();
					let mut QR = A.cloned();
					let mut H = Mat::zeros(bs, size);

					let computed_rank = qr_in_place(
						QR.as_mut(),
						H.as_mut(),
						par,
						MemStack::new(&mut MemBuffer::new(qr_in_place_scratch::<c64>(m, n, bs, par, default()))),
						default(),
					);
					assert!(computed_rank.rank == rank);

					let mut Q = Mat::<c64, _, _>::zeros(m, m);
					let mut R = QR.as_ref().cloned();

					for j in 0..m {
						Q[(j, j)] = c64::ONE;
					}

					householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
						QR.as_ref().subcols(0, size),
						H.as_ref(),
						Conj::No,
						Q.as_mut(),
						Par::Seq,
						MemStack::new(&mut MemBuffer::new(
							householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<c64>(m, bs, m),
						)),
					);

					for j in 0..n {
						for i in j + 1..m {
							R[(i, j)] = c64::ZERO;
						}
					}

					assert!(Q * R ~ A);
				}
			}
		}
	}
}
