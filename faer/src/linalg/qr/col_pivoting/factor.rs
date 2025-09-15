use crate::assert;
use crate::internal_prelude::*;
use crate::perm::swap_cols_idx;
use linalg::householder::{self, HouseholderInfo};
use pulp::Simd;

pub use super::super::no_pivoting::factor::recommended_blocksize;

// B11 += A10 * dot
// B01 += l * dot
// dot  = -tau_inv * (B01 + B10^H * B11)
// B01 += dot
// norm-= abs2(B01)
#[math]
fn update_mat_and_dot_simd<T: ComplexField>(
	norm: RowMut<'_, T>,
	dot: RowMut<'_, T>,
	B01: RowMut<'_, T>,
	B11: MatMut<'_, T, usize, usize, ContiguousFwd>,
	A10: ColRef<'_, T, usize, ContiguousFwd>,
	B10: ColRef<'_, T, usize, ContiguousFwd>,
	l: T,
	tau_inv: T::Real,
	align: usize,
) {
	struct Impl<'a, 'M, 'N, T: ComplexField> {
		norm: RowMut<'a, T, Dim<'N>>,
		dot: RowMut<'a, T, Dim<'N>>,
		B01: RowMut<'a, T, Dim<'N>>,
		B11: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		A10: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		B10: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		l: T,
		tau_inv: T::Real,
		align: usize,
	}
	impl<'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'_, 'M, 'N, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let Self {
				mut norm,
				mut dot,
				B01: mut u,
				mut B11,
				A10,
				B10,
				l,
				tau_inv,
				align,
			} = self;

			let m = B11.nrows();
			let n = B11.ncols();

			let simd = SimdCtx::<'_, T, S>::new_align(T::simd_ctx(simd), m, align);

			let (head, body4, body1, tail) = simd.batch_indices::<4>();

			let mut j = n.indices();

			loop {
				match (j.next(), j.next(), j.next(), j.next()) {
					(Some(j0), Some(j1), Some(j2), Some(j3)) => {
						let b0 = copy(dot[j0]);
						let b1 = copy(dot[j1]);
						let b2 = copy(dot[j2]);
						let b3 = copy(dot[j3]);

						let rhs0 = simd.splat(&b0);
						let rhs1 = simd.splat(&b1);
						let rhs2 = simd.splat(&b2);
						let rhs3 = simd.splat(&b3);

						let mut acc0 = simd.zero();
						let mut acc1 = simd.zero();
						let mut acc2 = simd.zero();
						let mut acc3 = simd.zero();

						macro_rules! do_it {
							($i: expr) => {{
								let i = $i;

								let lhs0 = simd.read(A10, i);
								let lhs1 = simd.read(B10, i);

								let mut dst0 = simd.read(B11.rb().col(j0), i);
								dst0 = simd.mul_add(lhs0, rhs0, dst0);
								acc0 = simd.conj_mul_add(lhs1, dst0, acc0);
								simd.write(B11.rb_mut().col_mut(j0), i, dst0);

								let mut dst1 = simd.read(B11.rb().col(j1), i);
								dst1 = simd.mul_add(lhs0, rhs1, dst1);
								acc1 = simd.conj_mul_add(lhs1, dst1, acc1);
								simd.write(B11.rb_mut().col_mut(j1), i, dst1);

								let mut dst2 = simd.read(B11.rb().col(j2), i);
								dst2 = simd.mul_add(lhs0, rhs2, dst2);
								acc2 = simd.conj_mul_add(lhs1, dst2, acc2);
								simd.write(B11.rb_mut().col_mut(j2), i, dst2);

								let mut dst3 = simd.read(B11.rb().col(j3), i);
								dst3 = simd.mul_add(lhs0, rhs3, dst3);
								acc3 = simd.conj_mul_add(lhs1, dst3, acc3);
								simd.write(B11.rb_mut().col_mut(j3), i, dst3);
							}};
						}

						if let Some(i) = head {
							do_it!(i);
						}

						for [i0, i1, i2, i3] in body4.clone() {
							do_it!(i0);
							do_it!(i1);
							do_it!(i2);
							do_it!(i3);
						}
						for i in body1.clone() {
							do_it!(i);
						}
						if let Some(i) = tail {
							do_it!(i);
						}

						let tmp = u[j0] + l * b0;
						let d0 = mul_real(tmp + simd.reduce_sum(acc0), -tau_inv);
						u[j0] = tmp + d0;
						dot[j0] = d0;
						norm[j0] = from_real(sqrt(abs2(norm[j0]) - abs2(u[j0])));

						let tmp = u[j1] + l * b1;
						let d1 = mul_real(tmp + simd.reduce_sum(acc1), -tau_inv);
						u[j1] = tmp + d1;
						dot[j1] = d1;
						norm[j1] = from_real(sqrt(abs2(norm[j1]) - abs2(u[j1])));

						let tmp = u[j2] + l * b2;
						let d2 = mul_real(tmp + simd.reduce_sum(acc2), -tau_inv);
						u[j2] = tmp + d2;
						dot[j2] = d2;
						norm[j2] = from_real(sqrt(abs2(norm[j2]) - abs2(u[j2])));

						let tmp = u[j3] + l * b3;
						let d3 = mul_real(tmp + simd.reduce_sum(acc3), -tau_inv);
						u[j3] = tmp + d3;
						dot[j3] = d3;
						norm[j3] = from_real(sqrt(abs2(norm[j3]) - abs2(u[j3])));
					},
					(j0, j1, j2, j3) => {
						for j0 in [j0, j1, j2, j3].into_iter().flatten() {
							let b0 = copy(dot[j0]);
							let rhs0 = simd.splat(&b0);

							let mut acc0 = simd.zero();

							macro_rules! do_it {
								($i: expr) => {{
									let i = $i;

									let lhs0 = simd.read(A10, i);
									let lhs1 = simd.read(B10, i);

									let mut dst0 = simd.read(B11.rb().col(j0), i);
									dst0 = simd.mul_add(lhs0, rhs0, dst0);
									acc0 = simd.conj_mul_add(lhs1, dst0, acc0);
									simd.write(B11.rb_mut().col_mut(j0), i, dst0);
								}};
							}

							if let Some(i) = head {
								do_it!(i);
							}
							for [i0, i1, i2, i3] in body4.clone() {
								do_it!(i0);
								do_it!(i1);
								do_it!(i2);
								do_it!(i3);
							}

							for i in body1.clone() {
								do_it!(i);
							}
							if let Some(i) = tail {
								do_it!(i);
							}

							let tmp = u[j0] + l * b0;
							let d0 = mul_real(tmp + simd.reduce_sum(acc0), -tau_inv);
							u[j0] = tmp + d0;
							dot[j0] = d0;
							norm[j0] = from_real(sqrt(abs2(norm[j0]) - abs2(u[j0])));
						}
						break;
					},
				}
			}
		}
	}

	with_dim!(M, B11.nrows());
	with_dim!(N, B11.ncols());
	dispatch!(
		Impl {
			norm: norm.as_col_shape_mut(N),
			dot: dot.as_col_shape_mut(N),
			B01: B01.as_col_shape_mut(N),
			B11: B11.as_shape_mut(M, N),
			A10: A10.as_row_shape(M),
			B10: B10.as_row_shape(M),
			l,
			tau_inv,
			align
		},
		Impl,
		T
	)
}

#[math]

/// $QR$ factorization with column pivoting tuning parameters
#[derive(Copy, Clone, Debug)]
pub struct ColPivQrParams {
	/// threshold at which blocking algorithms should be disabled
	pub blocking_threshold: usize,
	/// threshold at which the parallelism should be disabled
	pub par_threshold: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for ColPivQrParams {
	#[inline]
	fn auto() -> Self {
		Self {
			blocking_threshold: 48 * 48,
			par_threshold: 192 * 256,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

#[track_caller]
#[math]
fn qr_in_place_unblocked<'out, I: Index, T: ComplexField>(
	A: MatMut<'_, T>,
	H: RowMut<'_, T>,
	col_perm: &'out mut [I],
	col_perm_inv: &'out mut [I],
	par: Par,
	stack: &mut MemStack,
	params: Spec<ColPivQrParams, T>,
) -> (ColPivQrInfo, PermRef<'out, I>) {
	let m = A.nrows();
	let n = A.ncols();
	let size = H.ncols();

	let params = params.config;
	let mut A = A;
	let mut H = H;
	let mut par = par;

	assert!(size == Ord::min(m, n));
	for j in 0..n {
		col_perm[j] = I::truncate(j);
	}

	let mut n_trans = 0;

	'main: {
		if size == 0 {
			break 'main;
		}

		let (mut dot, stack) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
		let (mut norm, stack) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
		let _ = stack;

		let mut dot = dot.as_mat_mut().col_mut(0).transpose_mut();
		let mut norm = norm.as_mat_mut().col_mut(0).transpose_mut();

		let mut best = zero();

		let threshold = sqrt(eps::<T::Real>());

		for j in 0..n {
			let val = A.rb().col(j).norm_l2();
			norm[j] = from_real(val);

			if val > best {
				best = val;
			}
		}

		let scale_fwd = copy(best);
		let scale_bwd = recip(best);

		zip!(A.rb_mut()).for_each(|unzip!(a)| *a = mul_real(*a, scale_bwd));

		for j in 0..n {
			norm[j] = from_real(real(norm[j]) * scale_bwd);
		}
		best = best * scale_bwd;
		let mut best_threshold = best * threshold;

		'unscale: {
			for k in 0..size {
				let mut new_best = zero::<T::Real>();
				let mut best_col = k;
				for j in k..n {
					let val = real(norm[j]);
					if val > new_best {
						new_best = val;
						best_col = j;
					}
				}

				let delayed_update = T::SIMD_CAPABILITIES.is_simd() && A.row_stride() == 1 && k > 0 && new_best >= best_threshold;

				if k > 0 && !delayed_update {
					let (_, _, A10, mut A11) = A.rb_mut().split_at_mut(k, k);
					let dot = dot.rb().get(k..);
					let A10 = A10.rb().col(k - 1);

					linalg::matmul::matmul(A11.rb_mut(), Accum::Add, A10, dot, one(), par);

					best = zero();
					for j in k..n {
						let val = A11.rb().col(j - k).norm_l2();

						norm[j] = from_real(val);

						if val > best {
							best = val;
							best_col = j;
						}
					}
					best_threshold = best * threshold;
				}

				if best_col != k {
					n_trans += 1;
					col_perm.as_mut().swap(best_col, k);
					swap_cols_idx(A.rb_mut(), best_col, k);
					swap_cols_idx(dot.rb_mut().as_mat_mut(), best_col, k);
					swap_cols_idx(norm.rb_mut().as_mat_mut(), best_col, k);
				}

				let (_, _, A10, mut A11) = A.rb_mut().split_at_mut(k, k);
				let A10 = A10.rb();
				let dot0 = dot.rb_mut().get_mut(k..);

				let (mut B00, B01, B10, mut B11) = A11.rb_mut().split_at_mut(1, 1);
				let B00 = &mut B00[(0, 0)];
				let mut B01 = B01.row_mut(0);
				let mut B10 = B10.col_mut(0);

				let l = if delayed_update {
					let A10 = A10.col(k - 1);
					copy(A10[0])
				} else {
					zero()
				};
				let r = copy(dot0[0]);

				let mut dot = dot.rb_mut().get_mut(k + 1..);
				let mut norm = norm.rb_mut().get_mut(k + 1..);

				if delayed_update {
					let A10 = A10.col(k - 1).get(1..);

					*B00 = *B00 + l * r;
					zip!(B10.rb_mut(), A10).for_each(|unzip!(x, y)| {
						*x = *x + r * *y;
					});
				}

				let HouseholderInfo { tau, .. } = householder::make_householder_in_place(B00, B10.rb_mut());
				let tau_inv = recip(tau);
				H[k] = from_real(tau);

				if k + 1 == size {
					if delayed_update {
						zip!(B01.rb_mut(), dot.rb()).for_each(|unzip!(x, y)| {
							*x = *x + l * *y;
						});
					}
					break 'unscale;
				}

				if (m - k - 1) * (n - k - 1) < params.par_threshold {
					par = Par::Seq;
				}

				if delayed_update {
					let A10 = A10.col(k - 1).get(1..);

					match par {
						Par::Seq => {
							update_mat_and_dot_simd(
								norm.rb_mut(),
								dot.rb_mut(),
								B01.rb_mut(),
								B11.rb_mut().try_as_col_major_mut().unwrap(),
								A10.try_as_col_major().unwrap(),
								B10.rb().try_as_col_major().unwrap(),
								l,
								tau_inv,
								simd_align(k + 1),
							);
						},
						#[cfg(feature = "rayon")]
						Par::Rayon(nthreads) => {
							let nthreads = nthreads.get();
							use rayon::prelude::*;
							norm.par_partition_mut(nthreads)
								.zip(dot.par_partition_mut(nthreads))
								.zip(B01.par_partition_mut(nthreads))
								.zip(B11.par_col_partition_mut(nthreads))
								.for_each(|(((norm, dot), B01), B11)| {
									update_mat_and_dot_simd(
										norm,
										dot,
										B01,
										B11.try_as_col_major_mut().unwrap(),
										A10.try_as_col_major().unwrap(),
										B10.rb().try_as_col_major().unwrap(),
										copy(l),
										copy(tau_inv),
										simd_align(k + 1),
									);
								});
						},
					}
				} else {
					dot.copy_from(B01.rb());
					linalg::matmul::matmul(dot.rb_mut(), Accum::Add, B10.rb().adjoint(), B11.rb(), one(), par);

					zip!(B01.rb_mut(), dot.rb_mut(), norm.rb_mut()).for_each(|unzip!(a, dot, norm)| {
						*dot = mul_real(-*dot, tau_inv);
						*a = *a + *dot;
						*norm = from_real(sqrt(abs2(*norm) - abs2(*a)));
					});
				}
			}
		}
		zip!(A.rb_mut()).for_each_triangular_upper(linalg::zip::Diag::Include, |unzip!(a)| *a = mul_real(*a, scale_fwd));
	}

	for j in 0..n {
		col_perm_inv[col_perm[j].zx()] = I::truncate(j);
	}

	(
		ColPivQrInfo {
			transposition_count: n_trans,
		},
		unsafe { PermRef::new_unchecked(col_perm, col_perm_inv, n) },
	)
}

/// computes the layout of required workspace for performing a qr decomposition
/// with column pivoting
pub fn qr_in_place_scratch<I: Index, T: ComplexField>(
	nrows: usize,
	ncols: usize,
	blocksize: usize,
	par: Par,
	params: Spec<ColPivQrParams, T>,
) -> StackReq {
	let _ = nrows;
	let _ = ncols;
	let _ = par;
	let _ = blocksize;
	let _ = &params;
	linalg::temp_mat_scratch::<T>(ncols, 2)
}

/// information about the resulting $QR$ factorization.
#[derive(Copy, Clone, Debug)]
pub struct ColPivQrInfo {
	/// number of transpositions that were performed, can be used to compute the determinant of
	/// $P$.
	pub transposition_count: usize,
}

#[track_caller]
#[math]
pub fn qr_in_place<'out, I: Index, T: ComplexField>(
	A: MatMut<'_, T>,
	Q_coeff: MatMut<'_, T>,
	col_perm: &'out mut [I],
	col_perm_inv: &'out mut [I],
	par: Par,
	stack: &mut MemStack,
	params: Spec<ColPivQrParams, T>,
) -> (ColPivQrInfo, PermRef<'out, I>) {
	let mut A = A;
	let mut H = Q_coeff;
	let size = H.ncols();
	let blocksize = H.nrows();

	let ret = qr_in_place_unblocked(A.rb_mut(), H.rb_mut().row_mut(0), col_perm, col_perm_inv, par, stack, params);

	let mut j = 0;
	while j < size {
		let blocksize = Ord::min(blocksize, size - j);

		let mut H = H.rb_mut().subcols_mut(j, blocksize).subrows_mut(0, blocksize);

		for j in 0..blocksize {
			H[(j, j)] = copy(H[(0, j)]);
		}

		let A = A.rb().get(j.., j..j + blocksize);

		householder::upgrade_householder_factor(H.rb_mut(), A, blocksize, 1, par);
		j += blocksize;
	}
	ret
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, assert, c64};
	use dyn_stack::MemBuffer;

	#[test]
	fn test_unblocked_qr() {
		let rng = &mut StdRng::seed_from_u64(0);

		for par in [Par::Seq, Par::rayon(8)] {
			for n in [2, 3, 4, 8, 16, 24, 32, 128, 255] {
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

				let col_perm = &mut *vec![0usize; n];
				let col_perm_inv = &mut *vec![0usize; n];

				let q = qr_in_place(
					QR.as_mut(),
					H.as_mut(),
					col_perm,
					col_perm_inv,
					par,
					MemStack::new(&mut MemBuffer::new(qr_in_place_scratch::<usize, c64>(n, n, bs, par, default()))),
					default(),
				)
				.1;

				let mut Q = Mat::<c64, _, _>::zeros(n, n);
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

				assert!(Q * R * q ~ A);
			}

			let n = 20;
			for m in [2, 3, 4, 8, 16, 24, 32, 128, 255] {
				let bs = 15;
				let size = Ord::min(m, n);

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

				let col_perm = &mut *vec![0usize; n];
				let col_perm_inv = &mut *vec![0usize; n];

				let q = qr_in_place(
					QR.as_mut(),
					H.as_mut(),
					col_perm,
					col_perm_inv,
					par,
					MemStack::new(&mut MemBuffer::new(qr_in_place_scratch::<usize, c64>(m, n, bs, par, default()))),
					default(),
				)
				.1;

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

				assert!(Q * R * q ~ A);
			}
		}
	}
}
