use crate::assert;
use crate::internal_prelude::*;
use linalg::householder;
use linalg::matmul::triangular::BlockStructure;
use linalg::matmul::{self, dot};

/// tridiagonalization tuning parameters
#[derive(Copy, Clone, Debug)]
pub struct TridiagParams {
	/// threshold at which parallelism should be disabled
	pub par_threshold: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for TridiagParams {
	fn auto() -> Self {
		Self {
			par_threshold: 192 * 256,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

/// computes the size and alignment of the workspace required to compute a self-adjoint matrix's
/// tridiagonalization
pub fn tridiag_in_place_scratch<T: ComplexField>(dim: usize, par: Par, params: Spec<TridiagParams, T>) -> StackReq {
	_ = par;
	_ = params;
	StackReq::all_of(&[temp_mat_scratch::<T>(dim, 1).array(2), temp_mat_scratch::<T>(dim, par.degree())])
}

#[math]
fn tridiag_fused_op_simd<T: ComplexField>(
	A: MatMut<'_, T, usize, usize, ContiguousFwd>,
	y2: ColMut<'_, T, usize>,
	z2: ColMut<'_, T, usize, ContiguousFwd>,

	ry2: ColRef<'_, T, usize>,
	rz2: ColRef<'_, T, usize, ContiguousFwd>,

	u0: ColRef<'_, T, usize, ContiguousFwd>,
	u1: ColRef<'_, T, usize>,
	u2: ColRef<'_, T, usize>,
	v2: ColRef<'_, T, usize, ContiguousFwd>,

	f: T,
	align: usize,
) {
	struct Impl<'a, 'M, 'N, T: ComplexField> {
		A: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		y2: ColMut<'a, T, Dim<'N>>,
		z2: ColMut<'a, T, Dim<'M>, ContiguousFwd>,

		ry2: ColRef<'a, T, Dim<'N>>,
		rz2: ColRef<'a, T, Dim<'M>, ContiguousFwd>,

		u0: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		u1: ColRef<'a, T, Dim<'N>>,
		u2: ColRef<'a, T, Dim<'N>>,
		v2: ColRef<'a, T, Dim<'M>, ContiguousFwd>,

		f: T,
		align: usize,
	}

	impl<'a, 'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'N, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self {
				mut A,
				mut y2,
				mut z2,
				ry2,
				rz2,
				u0,
				u1,
				u2,
				v2,
				f,
				mut align,
			} = self;

			let simd = T::simd_ctx(simd);
			let (m, n) = A.shape();
			{
				let simd = SimdCtx::<T, S>::new_align(simd, m, align);
				let (head, body, tail) = simd.indices();

				if let Some(i0) = head {
					simd.write(z2.rb_mut(), i0, simd.zero());
				}
				for i0 in body {
					simd.write(z2.rb_mut(), i0, simd.zero());
				}
				if let Some(i0) = tail {
					simd.write(z2.rb_mut(), i0, simd.zero());
				}
			}

			for j in n.indices() {
				let i = m.idx_inc(*j);
				with_dim!(m, *m - *j);

				let simd = SimdCtx::<T, S>::new_align(simd, m, align);
				align -= 1;

				let mut A = A.rb_mut().col_mut(j).subrows_mut(i, m);

				let mut z = z2.rb_mut().subrows_mut(i, m);
				let rz = rz2.subrows(i, m);
				let ua = u0.subrows(i, m);
				let v = v2.subrows(i, m);

				let y = y2.rb_mut().at_mut(j);
				let ry = simd.splat(&(-ry2[j]));
				let ub = simd.splat(&(-u1[j]));
				let uc = simd.splat(&(f * u2[j]));

				let mut acc0 = simd.zero();
				let mut acc1 = simd.zero();
				let mut acc2 = simd.zero();
				let mut acc3 = simd.zero();

				let (head, body4, body1, tail) = simd.batch_indices::<4>();
				if let Some(i0) = head {
					let mut a = simd.read(A.rb(), i0);
					a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
					a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
					simd.write(A.rb_mut(), i0, a);

					let tmp = simd.read(z.rb(), i0);
					simd.write(z.rb_mut(), i0, simd.mul_add(a, uc, tmp));

					acc0 = simd.conj_mul_add(a, simd.read(v, i0), acc0);
				}

				for [i0, i1, i2, i3] in body4 {
					{
						let mut a = simd.read(A.rb(), i0);
						a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
						a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
						simd.write(A.rb_mut(), i0, a);

						let tmp = simd.read(z.rb(), i0);
						simd.write(z.rb_mut(), i0, simd.mul_add(a, uc, tmp));

						acc0 = simd.conj_mul_add(a, simd.read(v, i0), acc0);
					}
					{
						let mut a = simd.read(A.rb(), i1);
						a = simd.conj_mul_add(ry, simd.read(ua, i1), a);
						a = simd.conj_mul_add(ub, simd.read(rz, i1), a);
						simd.write(A.rb_mut(), i1, a);

						let tmp = simd.read(z.rb(), i1);
						simd.write(z.rb_mut(), i1, simd.mul_add(a, uc, tmp));

						acc1 = simd.conj_mul_add(a, simd.read(v, i1), acc1);
					}
					{
						let mut a = simd.read(A.rb(), i2);
						a = simd.conj_mul_add(ry, simd.read(ua, i2), a);
						a = simd.conj_mul_add(ub, simd.read(rz, i2), a);
						simd.write(A.rb_mut(), i2, a);

						let tmp = simd.read(z.rb(), i2);
						simd.write(z.rb_mut(), i2, simd.mul_add(a, uc, tmp));

						acc2 = simd.conj_mul_add(a, simd.read(v, i2), acc2);
					}
					{
						let mut a = simd.read(A.rb(), i3);
						a = simd.conj_mul_add(ry, simd.read(ua, i3), a);
						a = simd.conj_mul_add(ub, simd.read(rz, i3), a);
						simd.write(A.rb_mut(), i3, a);

						let tmp = simd.read(z.rb(), i3);
						simd.write(z.rb_mut(), i3, simd.mul_add(a, uc, tmp));

						acc3 = simd.conj_mul_add(a, simd.read(v, i3), acc3);
					}
				}
				for i0 in body1 {
					let mut a = simd.read(A.rb(), i0);
					a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
					a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
					simd.write(A.rb_mut(), i0, a);

					let tmp = simd.read(z.rb(), i0);
					simd.write(z.rb_mut(), i0, simd.mul_add(a, uc, tmp));

					acc0 = simd.conj_mul_add(a, simd.read(v, i0), acc0);
				}
				if let Some(i0) = tail {
					let mut a = simd.read(A.rb(), i0);
					a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
					a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
					simd.write(A.rb_mut(), i0, a);

					let tmp = simd.read(z.rb(), i0);
					simd.write(z.rb_mut(), i0, simd.mul_add(a, uc, tmp));

					acc0 = simd.conj_mul_add(a, simd.read(v, i0), acc0);
				}

				acc0 = simd.add(acc0, acc1);
				acc2 = simd.add(acc2, acc3);
				acc0 = simd.add(acc0, acc2);

				let acc0 = simd.reduce_sum(acc0);
				let i0 = m.idx(0);
				*y = f * (acc0 - A[i0] * v[i0]);
			}
		}
	}

	with_dim!(M, A.nrows());
	with_dim!(N, A.ncols());

	dispatch!(
		Impl {
			A: A.as_shape_mut(M, N),
			y2: y2.as_row_shape_mut(N),
			z2: z2.as_row_shape_mut(M),
			ry2: ry2.as_row_shape(N),
			rz2: rz2.as_row_shape(M),
			u0: u0.as_row_shape(M),
			u1: u1.as_row_shape(N),
			u2: u2.as_row_shape(N),
			v2: v2.as_row_shape(M),
			f,
			align,
		},
		Impl,
		T
	)
}

#[math]
fn tridiag_fused_op<T: ComplexField>(
	A: MatMut<'_, T>,
	y2: ColMut<'_, T>,
	z2: ColMut<'_, T>,

	ry2: ColRef<'_, T>,
	rz2: ColRef<'_, T>,

	u0: ColRef<'_, T>,
	u1: ColRef<'_, T>,
	u2: ColRef<'_, T>,
	v2: ColRef<'_, T>,

	f: T,
	align: usize,
) {
	let mut A = A;
	let mut z2 = z2;

	if try_const! { T::SIMD_CAPABILITIES.is_simd() } {
		if let (Some(A), Some(z2), Some(rz2), Some(u0), Some(v2)) = (
			A.rb_mut().try_as_col_major_mut(),
			z2.rb_mut().try_as_col_major_mut(),
			rz2.try_as_col_major(),
			u0.try_as_col_major(),
			v2.try_as_col_major(),
		) {
			tridiag_fused_op_simd(A, y2, z2, ry2, rz2, u0, u1, u2, v2, f, align);
		} else {
			tridiag_fused_op_fallback(A, y2, z2, ry2, rz2, u0, u1, u2, v2, f);
		}
	} else {
		tridiag_fused_op_fallback(A, y2, z2, ry2, rz2, u0, u1, u2, v2, f);
	}
}

#[math]
fn tridiag_fused_op_fallback<T: ComplexField>(
	A: MatMut<'_, T>,
	y2: ColMut<'_, T>,
	z2: ColMut<'_, T>,

	ry2: ColRef<'_, T>,
	rz2: ColRef<'_, T>,

	u0: ColRef<'_, T>,
	u1: ColRef<'_, T>,
	u2: ColRef<'_, T>,
	v2: ColRef<'_, T>,

	f: T,
) {
	let par = Par::Seq;

	let mut A = A;
	let mut y2 = y2;

	let n = A.ncols();

	let (mut A0, mut A1) = A.rb_mut().split_at_row_mut(n);
	let (u00, u01) = u0.split_at_row(n);
	let (v20, v21) = v2.split_at_row(n);
	let (mut z20, mut z21) = z2.split_at_row_mut(n);

	let (rz20, rz21) = rz2.split_at_row(n);

	matmul::triangular::matmul(
		A0.rb_mut(),
		BlockStructure::TriangularLower,
		Accum::Add,
		u00,
		BlockStructure::Rectangular,
		ry2.adjoint(),
		BlockStructure::Rectangular,
		-one::<T>(),
		par,
	);
	matmul::triangular::matmul(
		A0.rb_mut(),
		BlockStructure::TriangularLower,
		Accum::Add,
		rz20,
		BlockStructure::Rectangular,
		u1.adjoint(),
		BlockStructure::Rectangular,
		-one::<T>(),
		par,
	);
	matmul::matmul(A1.rb_mut(), Accum::Add, u01, ry2.adjoint(), -one::<T>(), par);
	matmul::matmul(A1.rb_mut(), Accum::Add, rz21, u1.adjoint(), -one::<T>(), par);

	matmul::triangular::matmul(
		z20.rb_mut(),
		BlockStructure::Rectangular,
		Accum::Replace,
		A0.rb(),
		BlockStructure::TriangularLower,
		u2,
		BlockStructure::Rectangular,
		f.clone(),
		par,
	);
	matmul::triangular::matmul(
		y2.rb_mut(),
		BlockStructure::Rectangular,
		Accum::Replace,
		A0.rb().adjoint(),
		BlockStructure::StrictTriangularUpper,
		v20,
		BlockStructure::Rectangular,
		f.clone(),
		par,
	);

	matmul::matmul(z21.rb_mut(), Accum::Replace, A1.rb(), u2, f.clone(), par);
	matmul::matmul(y2.rb_mut(), Accum::Add, A1.rb().adjoint(), v21, f.clone(), par);
}

/// computes a self-adjoint matrix $A$'s tridiagonalization such that $A = Q T Q^H$
///
/// $T$ is a self-adjoint tridiagonal matrix stored in $A$'s diagonal and subdiagonal
///
/// $Q$ is a sequence of householder reflections stored in the unit lower triangular half of $A$
/// (excluding the diagonal), with the householder coefficients being stored in `householder`
#[math]
pub fn tridiag_in_place<T: ComplexField>(
	A: MatMut<'_, T>,
	householder: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<TridiagParams, T>,
) {
	let params = params.config;
	let mut A = A;
	let mut H = householder;
	let mut par = par;
	let n = A.nrows();
	let b = H.nrows();

	assert!(H.ncols() == n.saturating_sub(1));

	if n == 0 {
		return;
	}

	let (mut y, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
	let (mut w, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
	let (mut z, _) = unsafe { temp_mat_uninit(n, par.degree(), stack) };
	let mut y = y.as_mat_mut().col_mut(0);
	let mut w = w.as_mat_mut().col_mut(0);
	let mut z = z.as_mat_mut();

	{
		let mut H = H.rb_mut().row_mut(0);
		for k in 0..n {
			let (_, A01, A10, A11) = A.rb_mut().split_at_mut(k, k);

			let (_, _) = A01.split_first_col().unwrap();
			let (_, A20) = A10.split_first_row_mut().unwrap();
			let (mut A11, _, A21, mut A22) = A11.split_at_mut(1, 1);

			let mut A21 = A21.col_mut(0);

			let a11 = &mut A11[(0, 0)];

			let (y1, mut y2) = y.rb_mut().split_at_row_mut(k).1.split_at_row_mut(1);
			let y1 = copy(y1[0]);

			if k > 0 {
				let p = k - 1;

				let u2 = (A20.rb()).col(p);

				*a11 = *a11 - y1 - conj(y1);

				z!(A21.rb_mut(), u2, y2.rb()).for_each(|uz!(a, u, y)| {
					*a = *a - conj(y1) * *u - *y;
				});
			}

			if k + 1 == n {
				break;
			}

			let rem = n - k - 1;
			if rem * rem / 2 < params.par_threshold {
				par = Par::Seq;
			}

			let k1 = k + 1;

			let tau_inv;
			{
				let (mut a11, mut x2) = A21.rb_mut().split_at_row_mut(1);
				let a11 = &mut a11[0];

				let householder::HouseholderInfo { tau, .. } = householder::make_householder_in_place(a11, x2.rb_mut());

				tau_inv = recip(real(tau));
				H[k] = from_real(tau);

				let mut z2 = z.rb_mut().split_at_row_mut(k + 2).1;
				let mut w2 = w.rb_mut().split_at_row_mut(k + 2).1;

				let (mut y1, mut y2) = y2.rb_mut().split_at_row_mut(1);
				let y1 = &mut y1[0];

				let (A1, A2) = A22.rb_mut().split_at_row_mut(1);
				let A1 = A1.row_mut(0);

				let (mut a11, _) = A1.split_at_col_mut(1);
				let a11 = &mut a11[0];

				let (A21, mut A22) = A2.split_at_col_mut(1);
				let mut A21 = A21.col_mut(0);

				if k > 0 {
					let p = k - 1;

					let (u1, u2) = (A20.rb()).col(p).split_at_row(1);
					let u1 = copy(u1[0]);

					*a11 = *a11 - u1 * conj(y1) - *y1 * conj(u1);

					z!(A21.rb_mut(), u2.rb(), y2.rb()).for_each(|uz!(a, u, y)| {
						*a = *a - *u * conj(y1) - *y * conj(u1);
					});

					w2.copy_from(y2.rb());

					match par {
						Par::Seq => {
							let mut z2 = z2.rb_mut().col_mut(0);
							tridiag_fused_op(
								A22.rb_mut(),
								y2.rb_mut(),
								z2.rb_mut(),
								w2.rb(),
								w2.rb(),
								u2.rb(),
								u2.rb(),
								x2.rb(),
								x2.rb(),
								from_real(tau_inv),
								simd_align(k1 + 1),
							);
							z!(y2.rb_mut(), z2.rb_mut()).for_each(|uz!(y, z)| *y = *y + *z);
						},
						#[cfg(feature = "rayon")]
						Par::Rayon(nthreads) => {
							use rayon::prelude::*;
							let nthreads = nthreads.get();
							let mut z2 = z2.rb_mut().subcols_mut(0, nthreads);

							let n2 = A22.ncols();
							assert!((n2 as u64) < (1u64 << 50)); // to check that integers can be
							// represented exactly as floats

							let idx_to_col_start = |idx: usize| {
								let idx_as_percent = idx as f64 / nthreads as f64;
								let col_start_percent = 1.0f64 - libm::sqrt(1.0f64 - idx_as_percent);
								(col_start_percent * n2 as f64) as usize
							};

							{
								let A22 = A22.rb();
								let y2 = y2.rb();

								let f = from_real(tau_inv);
								z2.rb_mut().par_col_iter_mut().enumerate().for_each(|(idx, mut z2)| {
									let first = idx_to_col_start(idx);
									let last_col = idx_to_col_start(idx + 1);
									let nrows = n2 - first;
									let ncols = last_col - first;

									let mut A = unsafe { A22.rb().subcols(first, ncols).subrows(first, nrows).const_cast() };

									{
										let y2 = unsafe { y2.subrows(first, ncols).const_cast() };
										let mut z2 = z2.rb_mut().subrows_mut(first, nrows);

										let ry2 = w2.rb().subrows(first, ncols);
										let rz2 = w2.rb().subrows(first, nrows);

										let u0 = u2.subrows(first, nrows);
										let u1 = u2.subrows(first, ncols);
										let u2 = x2.rb().subrows(first, ncols);
										let v2 = x2.rb().subrows(first, nrows);

										tridiag_fused_op(
											A.rb_mut(),
											y2,
											z2.rb_mut(),
											ry2,
											rz2,
											u0,
											u1,
											u2,
											v2,
											copy(f),
											n.next_power_of_two() - (k1 + 1) - first,
										);
									}

									z2.rb_mut().subrows_mut(0, first).fill(zero());
								});
							}

							for mut z2 in z2.rb_mut().col_iter_mut() {
								z!(y2.rb_mut(), z2.rb_mut()).for_each(|uz!(y, z)| *y = *y + *z);
							}
						},
					}
				} else {
					matmul::triangular::matmul(
						y2.rb_mut(),
						BlockStructure::Rectangular,
						Accum::Replace,
						A22.rb(),
						BlockStructure::TriangularLower,
						x2.rb(),
						BlockStructure::Rectangular,
						from_real(tau_inv),
						par,
					);
					matmul::triangular::matmul(
						y2.rb_mut(),
						BlockStructure::Rectangular,
						Accum::Add,
						A22.rb().adjoint(),
						BlockStructure::StrictTriangularUpper,
						x2.rb(),
						BlockStructure::Rectangular,
						from_real(tau_inv),
						par,
					);
				}

				z!(y2.rb_mut(), A21.rb()).for_each(|uz!(y, a)| *y = *y + mul_real(*a, tau_inv));

				*y1 = mul_real(*a11 + dot::inner_prod(A21.rb().transpose(), Conj::Yes, x2.rb(), Conj::No), tau_inv);

				let b = mul_real(
					mul_pow2(*y1 + dot::inner_prod(x2.rb().transpose(), Conj::Yes, y2.rb(), Conj::No), from_f64(0.5)),
					tau_inv,
				);
				*y1 = *y1 - b;
				z!(y2.rb_mut(), x2.rb()).for_each(|uz!(y, u)| {
					*y = *y - b * *u;
				});
			}
		}
	}

	if n > 0 {
		let n = n - 1;
		let A = A.rb().submatrix(1, 0, n, n);
		let mut H = H.rb_mut().subcols_mut(0, n);

		let mut j = 0;
		while j < n {
			let b = Ord::min(b, n - j);

			let mut H = H.rb_mut().submatrix_mut(0, j, b, b);

			for k in 0..b {
				H[(k, k)] = copy(H[(0, k)]);
			}

			householder::upgrade_householder_factor(H.rb_mut(), A.submatrix(j, j, n - j, b), b, 1, par);
			j += b;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, assert, c64};
	use dyn_stack::MemBuffer;

	#[test]
	fn test_tridiag_real() {
		let rng = &mut StdRng::seed_from_u64(0);

		for n in [2, 3, 4, 8, 16] {
			let A = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);

			let A = A.rb() + A.adjoint();

			let b = 3;
			let mut H = Mat::zeros(b, n - 1);

			let mut V = A.clone();
			let mut V = V.as_mut();
			tridiag_in_place(
				V.rb_mut(),
				H.rb_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(StackReq::all_of(&[
					householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(n - 1, b, n),
					tridiag_in_place_scratch::<f64>(n, Par::Seq, default()),
				]))),
				default(),
			);

			let mut A = A.clone();
			let mut A = A.as_mut();

			for iter in 0..2 {
				let mut A = if iter == 0 { A.rb_mut() } else { A.rb_mut().transpose_mut() };

				let n = n - 1;

				let V = V.rb().submatrix(1, 0, n, n);
				let mut A = A.rb_mut().subrows_mut(1, n);
				let H = H.as_ref();

				householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
					V,
					H.as_ref(),
					if iter == 0 { Conj::Yes } else { Conj::No },
					A.rb_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(
						householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(n, b, n + 1),
					)),
				);
			}

			let approx_eq = CwiseMat(ApproxEq::<f64>::eps());
			for j in 0..n {
				for i in 0..n {
					if i > j + 1 || j > i + 1 {
						V[(i, j)] = 0.0;
					}
				}
			}
			for i in 0..n {
				if i + 1 < n {
					V[(i, i + 1)] = V[(i + 1, i)];
				}
			}

			assert!(V ~ A);
		}
	}

	#[test]
	fn test_tridiag_cplx() {
		let rng = &mut StdRng::seed_from_u64(0);

		for n in [2, 3, 4, 8, 16] {
			let A = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			let A = A.rb() + A.adjoint();

			let b = 3;
			let mut H = Mat::zeros(b, n - 1);

			let mut V = A.clone();
			let mut V = V.as_mut();
			tridiag_in_place(
				V.rb_mut(),
				H.as_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(tridiag_in_place_scratch::<c64>(n, Par::Seq, default()))),
				default(),
			);

			let mut A = A.clone();
			let mut A = A.as_mut();

			for iter in 0..2 {
				let mut A = if iter == 0 { A.rb_mut() } else { A.rb_mut().transpose_mut() };

				let n = n - 1;

				let V = V.rb().submatrix(1, 0, n, n);
				let mut A = A.rb_mut().subrows_mut(1, n);
				let H = H.as_ref();

				householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
					V,
					H.as_ref(),
					if iter == 0 { Conj::Yes } else { Conj::No },
					A.rb_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(
						householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<c64>(n, b, n + 1),
					)),
				);
			}

			let approx_eq = CwiseMat(ApproxEq::eps());
			for j in 0..n {
				for i in 0..n {
					if i > j + 1 || j > i + 1 {
						V[(i, j)] = c64::ZERO;
					}
				}
			}
			for i in 0..n {
				if i + 1 < n {
					V[(i, i + 1)] = V[(i + 1, i)].conj();
				}
			}

			assert!(V ~ A);
		}
	}
}
