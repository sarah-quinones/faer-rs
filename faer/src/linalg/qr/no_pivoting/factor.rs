use crate::assert;
use crate::internal_prelude::*;
use linalg::householder::{self, HouseholderInfo};

#[math]
fn qr_in_place_unblocked<T: ComplexField>(A: MatMut<'_, T>, H: RowMut<'_, T>) {
	let mut A = A;
	let mut H = H;

	let size = H.ncols();

	for k in 0..size {
		let (mut A00, mut A01, mut A10, mut A11) = A.rb_mut().split_at_mut(k + 1, k + 1);

		let A00 = &mut A00[(k, k)];
		let mut A01 = A01.rb_mut().row_mut(k);
		let mut A10 = A10.rb_mut().col_mut(k);

		let HouseholderInfo { tau, .. } = householder::make_householder_in_place(A00, A10.rb_mut());

		let tau_inv = recip(tau);
		H[k] = from_real(tau);

		for (head, mut tail) in core::iter::zip(A01.rb_mut().iter_mut(), A11.rb_mut().col_iter_mut()) {
			let dot = *head + linalg::matmul::dot::inner_prod(A10.rb().transpose(), Conj::Yes, tail.rb(), Conj::No);
			let k = -mul_real(dot, tau_inv);
			*head = *head + k;
			z!(tail.rb_mut(), A10.rb()).for_each(|uz!(dst, src)| {
				*dst = *dst + k * *src;
			});
		}
	}
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
fn qr_in_place_blocked<T: ComplexField>(A: MatMut<'_, T>, H: MatMut<'_, T>, par: Par, stack: &mut MemStack, params: Spec<QrParams, T>) {
	let params = params.config;

	let m = A.nrows();
	let n = A.ncols();
	let size = H.ncols();
	let blocksize = H.nrows();

	assert!(blocksize > 0);

	if blocksize == 1 {
		return qr_in_place_unblocked(A, H.row_mut(0));
	}
	let sub_blocksize = if m * n < params.blocking_threshold { 1 } else { blocksize / 2 };

	let mut A = A;
	let mut H = H;

	let mut j = 0;
	while j < size {
		let blocksize = Ord::min(blocksize, size - j);
		let sub_blocksize = Ord::min(blocksize, sub_blocksize);

		let mut A = A.rb_mut().get_mut(j.., j..);
		let mut H = H.rb_mut().submatrix_mut(0, j, blocksize, blocksize);

		qr_in_place_blocked(
			A.rb_mut().subcols_mut(0, blocksize),
			H.rb_mut().subrows_mut(0, sub_blocksize),
			par,
			stack,
			params.into(),
		);

		let mut k = 0;
		while k < blocksize {
			let sub_blocksize = Ord::min(sub_blocksize, blocksize - k);

			if k > 0 {
				let mut H = H.rb_mut().subcols_mut(k, sub_blocksize);

				let (H0, H1) = H.rb_mut().split_at_row_mut(k);
				let H0 = H0.rb().subrows(0, sub_blocksize);
				let H1 = H1.subrows_mut(0, sub_blocksize);

				{ H1 }.copy_from_triangular_upper(H0);
			}
			k += sub_blocksize;
		}

		let (A0, A1) = A.rb_mut().split_at_col_mut(blocksize);
		let A0 = A0.rb();

		householder::upgrade_householder_factor(H.rb_mut(), A0, blocksize, sub_blocksize, par);
		if A1.ncols() > 0 {
			householder::apply_block_householder_transpose_on_the_left_in_place_with_conj(A0, H.rb(), Conj::Yes, A1, par, stack)
		};

		j += blocksize;
	}
}

#[track_caller]
pub fn qr_in_place<T: ComplexField>(A: MatMut<'_, T>, Q_coeff: MatMut<'_, T>, par: Par, stack: &mut MemStack, params: Spec<QrParams, T>) {
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

	qr_in_place_blocked(A, Q_coeff, par, stack, params);
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
	fn test_unblocked_qr() {
		let rng = &mut StdRng::seed_from_u64(0);

		for par in [Par::Seq, Par::rayon(8)] {
			for n in [2, 4, 8, 16, 24, 32, 127, 128, 257] {
				let approx_eq = CwiseMat(ApproxEq {
					abs_tol: 1e-10,
					rel_tol: 1e-10,
				});

				let A = CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: ComplexDistribution::new(StandardNormal, StandardNormal),
				}
				.rand::<Mat<c64>>(rng);
				let A = A.as_ref();

				let mut QR = A.cloned();
				let mut H = Row::zeros(n);

				qr_in_place_unblocked(QR.as_mut(), H.as_mut());

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

			for n in [2, 3, 4, 8, 16, 24, 32, 128, 255, 256, 257] {
				let bs = 15;

				let approx_eq = CwiseMat(ApproxEq {
					abs_tol: 1e-10,
					rel_tol: 1e-10,
				});

				let A = CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: ComplexDistribution::new(StandardNormal, StandardNormal),
				}
				.rand::<Mat<c64>>(rng);
				let A = A.as_ref();
				let mut QR = A.cloned();
				let mut H = Mat::zeros(bs, n);

				qr_in_place_blocked(
					QR.as_mut(),
					H.as_mut(),
					par,
					MemStack::new(&mut MemBuffer::new(qr_in_place_scratch::<c64>(n, n, bs, par, default()))),
					default(),
				);

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
			for m in [2, 3, 4, 8, 16, 24, 32, 128, 255, 256, 257] {
				let size = Ord::min(m, n);
				let bs = 15;

				let approx_eq = CwiseMat(ApproxEq {
					abs_tol: 1e-10,
					rel_tol: 1e-10,
				});

				let A = CwiseMatDistribution {
					nrows: m,
					ncols: n,
					dist: ComplexDistribution::new(StandardNormal, StandardNormal),
				}
				.rand::<Mat<c64>>(rng);
				let A = A.as_ref();
				let mut QR = A.cloned();
				let mut H = Mat::zeros(bs, size);

				qr_in_place_blocked(
					QR.as_mut(),
					H.as_mut(),
					par,
					MemStack::new(&mut MemBuffer::new(qr_in_place_scratch::<c64>(m, n, bs, par, default()))),
					default(),
				);

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
