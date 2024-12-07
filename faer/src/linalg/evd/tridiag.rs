use crate::assert;
use crate::internal_prelude::*;
use linalg::householder;
use linalg::matmul::triangular::BlockStructure;
use linalg::matmul::{self, dot};

/// QR factorization tuning parameters.
#[derive(Copy, Clone, Debug)]
pub struct TridiagParams {
	/// At which size the parallelism should be disabled.
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

pub fn tridiag_in_place_scratch<T: ComplexField>(dim: usize, par: Par, params: Spec<TridiagParams, T>) -> Result<StackReq, SizeOverflow> {
	_ = par;
	_ = params;
	StackReq::try_all_of([temp_mat_scratch::<T>(dim, 1)?.try_array(2)?, temp_mat_scratch::<T>(dim, par.degree())?])
}

#[azucar::reborrow]
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
					simd.write(&mut z2, i0, simd.zero());
				}
				for i0 in body {
					simd.write(&mut z2, i0, simd.zero());
				}
				if let Some(i0) = tail {
					simd.write(&mut z2, i0, simd.zero());
				}
			}

			for j in n.indices() {
				let i = m.idx_inc(*j);
				with_dim!(m, *m - *j);

				let simd = SimdCtx::<T, S>::new_align(simd, m, align);
				align -= 1;

				let mut A = (&mut A).col_mut(j).subrows_mut(i, m);

				let mut z = (&mut z2).subrows_mut(i, m);
				let rz = rz2.subrows(i, m);
				let ua = u0.subrows(i, m);
				let v = v2.subrows(i, m);

				let y = (&mut y2).at_mut(j);
				let ry = simd.splat(&(-ry2[j]));
				let ub = simd.splat(&(-u1[j]));
				let uc = simd.splat(&(f * u2[j]));

				let mut acc0 = simd.zero();
				let mut acc1 = simd.zero();
				let mut acc2 = simd.zero();
				let mut acc3 = simd.zero();

				let (head, body4, body1, tail) = simd.batch_indices::<4>();
				if let Some(i0) = head {
					let mut a = simd.read(&A, i0);
					a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
					a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
					simd.write(&mut A, i0, a);

					let tmp = simd.read(&z, i0);
					simd.write(&mut z, i0, simd.mul_add(a, uc, tmp));

					acc0 = simd.conj_mul_add(a, simd.read(v, i0), acc0);
				}

				for [i0, i1, i2, i3] in body4 {
					{
						let mut a = simd.read(&A, i0);
						a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
						a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
						simd.write(&mut A, i0, a);

						let tmp = simd.read(&z, i0);
						simd.write(&mut z, i0, simd.mul_add(a, uc, tmp));

						acc0 = simd.conj_mul_add(a, simd.read(v, i0), acc0);
					}
					{
						let mut a = simd.read(&A, i1);
						a = simd.conj_mul_add(ry, simd.read(ua, i1), a);
						a = simd.conj_mul_add(ub, simd.read(rz, i1), a);
						simd.write(&mut A, i1, a);

						let tmp = simd.read(&z, i1);
						simd.write(&mut z, i1, simd.mul_add(a, uc, tmp));

						acc1 = simd.conj_mul_add(a, simd.read(v, i1), acc1);
					}
					{
						let mut a = simd.read(&A, i2);
						a = simd.conj_mul_add(ry, simd.read(ua, i2), a);
						a = simd.conj_mul_add(ub, simd.read(rz, i2), a);
						simd.write(&mut A, i2, a);

						let tmp = simd.read(&z, i2);
						simd.write(&mut z, i2, simd.mul_add(a, uc, tmp));

						acc2 = simd.conj_mul_add(a, simd.read(v, i2), acc2);
					}
					{
						let mut a = simd.read(&A, i3);
						a = simd.conj_mul_add(ry, simd.read(ua, i3), a);
						a = simd.conj_mul_add(ub, simd.read(rz, i3), a);
						simd.write(&mut A, i3, a);

						let tmp = simd.read(&z, i3);
						simd.write(&mut z, i3, simd.mul_add(a, uc, tmp));

						acc3 = simd.conj_mul_add(a, simd.read(v, i3), acc3);
					}
				}
				for i0 in body1 {
					let mut a = simd.read(&A, i0);
					a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
					a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
					simd.write(&mut A, i0, a);

					let tmp = simd.read(&z, i0);
					simd.write(&mut z, i0, simd.mul_add(a, uc, tmp));

					acc0 = simd.conj_mul_add(a, simd.read(v, i0), acc0);
				}
				if let Some(i0) = tail {
					let mut a = simd.read(&A, i0);
					a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
					a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
					simd.write(&mut A, i0, a);

					let tmp = simd.read(&z, i0);
					simd.write(&mut z, i0, simd.mul_add(a, uc, tmp));

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

#[azucar::reborrow]
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

	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if let (Some(A), Some(z2), Some(rz2), Some(u0), Some(v2)) = (
			(&mut A).try_as_col_major_mut(),
			(&mut z2).try_as_col_major_mut(),
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

#[azucar::reborrow]
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

	let (mut A0, mut A1) = (&mut A).split_at_row_mut(n);
	let (u00, u01) = u0.split_at_row(n);
	let (v20, v21) = v2.split_at_row(n);
	let (mut z20, mut z21) = z2.split_at_row_mut(n);

	let (rz20, rz21) = rz2.split_at_row(n);

	matmul::triangular::matmul(
		&mut A0,
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
		&mut A0,
		BlockStructure::TriangularLower,
		Accum::Add,
		rz20,
		BlockStructure::Rectangular,
		u1.adjoint(),
		BlockStructure::Rectangular,
		-one::<T>(),
		par,
	);
	matmul::matmul(&mut A1, Accum::Add, u01, ry2.adjoint(), -one::<T>(), par);
	matmul::matmul(&mut A1, Accum::Add, rz21, u1.adjoint(), -one::<T>(), par);

	matmul::triangular::matmul(
		&mut z20,
		BlockStructure::Rectangular,
		Accum::Replace,
		&A0,
		BlockStructure::TriangularLower,
		u2,
		BlockStructure::Rectangular,
		f.clone(),
		par,
	);
	matmul::triangular::matmul(
		&mut y2,
		BlockStructure::Rectangular,
		Accum::Replace,
		(&A0).adjoint(),
		BlockStructure::StrictTriangularUpper,
		v20,
		BlockStructure::Rectangular,
		f.clone(),
		par,
	);

	matmul::matmul(&mut z21, Accum::Replace, &A1, u2, f.clone(), par);
	matmul::matmul(&mut y2, Accum::Add, (&A1).adjoint(), v21, f.clone(), par);
}

#[azucar::reborrow]
#[math]
pub fn tridiag_in_place<T: ComplexField>(A: MatMut<'_, T>, H: MatMut<'_, T>, par: Par, stack: &mut DynStack, params: Spec<TridiagParams, T>) {
	let params = params.into_inner();
	let mut A = A;
	let mut H = H;
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
		let mut H = (&mut H).row_mut(0);
		for k in 0..n {
			let (_, A01, A10, A11) = (&mut A).split_at_mut(k, k);

			let (_, _) = A01.split_first_col().unwrap();
			let (_, A20) = A10.split_first_row_mut().unwrap();
			let (mut A11, _, A21, mut A22) = A11.split_at_mut(1, 1);

			let mut A21 = A21.col_mut(0);

			let a11 = &mut A11[(0, 0)];

			let (y1, mut y2) = (&mut y).split_at_row_mut(k).1.split_at_row_mut(1);
			let y1 = copy(y1[0]);

			if k > 0 {
				let p = k - 1;

				let u2 = (&A20).col(p);

				*a11 = *a11 - y1 - conj(y1);

				z!(&mut A21, u2, &y2).for_each(|uz!(a, u, y)| {
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
				let (mut a11, mut x2) = (&mut A21).split_at_row_mut(1);
				let a11 = &mut a11[0];

				let (tau, _) = householder::make_householder_in_place(a11, &mut x2);

				tau_inv = recip(real(tau));
				H[k] = tau;

				let mut z2 = (&mut z).split_at_row_mut(k + 2).1;
				let mut w2 = (&mut w).split_at_row_mut(k + 2).1;

				let (mut y1, mut y2) = (&mut y2).split_at_row_mut(1);
				let y1 = &mut y1[0];

				let (A1, A2) = (&mut A22).split_at_row_mut(1);
				let A1 = A1.row_mut(0);

				let (mut a11, _) = A1.split_at_col_mut(1);
				let a11 = &mut a11[0];

				let (A21, mut A22) = A2.split_at_col_mut(1);
				let mut A21 = A21.col_mut(0);

				if k > 0 {
					let p = k - 1;

					let (u1, u2) = (&A20).col(p).split_at_row(1);
					let u1 = copy(u1[0]);

					*a11 = *a11 - u1 * conj(y1) - *y1 * conj(u1);

					z!(&mut A21, &u2, &y2).for_each(|uz!(a, u, y)| {
						*a = *a - *u * conj(y1) - *y * conj(u1);
					});

					w2.copy_from(&y2);

					match par {
						Par::Seq => {
							let mut z2 = (&mut z2).col_mut(0);
							tridiag_fused_op(&mut A22, &mut y2, &mut z2, &w2, &w2, &u2, &u2, &x2, &x2, from_real(tau_inv), simd_align(k1 + 1));
							z!(&mut y2, &mut z2).for_each(|uz!(y, z)| *y = *y + *z);
						},
						#[cfg(feature = "rayon")]
						Par::Rayon(nthreads) => {
							use rayon::prelude::*;
							let nthreads = nthreads.get();
							let mut z2 = (&mut z2).subcols_mut(0, nthreads);

							let n2 = A22.ncols();
							assert!((n2 as u64) < (1u64 << 50)); // to check that integers can be
							// represented exactly as floats

							let idx_to_col_start = |idx: usize| {
								let idx_as_percent = idx as f64 / nthreads as f64;
								let col_start_percent = 1.0f64 - libm::sqrt(1.0f64 - idx_as_percent);
								(col_start_percent * n2 as f64) as usize
							};

							{
								let A22 = &A22;
								let y2 = &y2;

								let f = from_real(tau_inv);
								(&mut z2).par_col_iter_mut().enumerate().for_each(|(idx, mut z2)| {
									let first = idx_to_col_start(idx);
									let last_col = idx_to_col_start(idx + 1);
									let nrows = n2 - first;
									let ncols = last_col - first;

									let mut A = unsafe { (&A22).subcols(first, ncols).subrows(first, nrows).const_cast() };

									{
										let y2 = unsafe { y2.subrows(first, ncols).const_cast() };
										let mut z2 = (&mut z2).subrows_mut(first, nrows);

										let ry2 = (&w2).subrows(first, ncols);
										let rz2 = (&w2).subrows(first, nrows);

										let u0 = u2.subrows(first, nrows);
										let u1 = u2.subrows(first, ncols);
										let u2 = (&x2).subrows(first, ncols);
										let v2 = (&x2).subrows(first, nrows);

										tridiag_fused_op(&mut A, y2, &mut z2, ry2, rz2, u0, u1, u2, v2, copy(f), n.next_power_of_two() - (k1 + 1) - first);
									}

									(&mut z2).subrows_mut(0, first).fill(zero());
								});
							}

							for mut z2 in (&mut z2).col_iter_mut() {
								z!(&mut y2, &mut z2).for_each(|uz!(y, z)| *y = *y + *z);
							}
						},
					}
				} else {
					matmul::triangular::matmul(
						&mut y2,
						BlockStructure::Rectangular,
						Accum::Replace,
						&A22,
						BlockStructure::TriangularLower,
						&x2,
						BlockStructure::Rectangular,
						from_real(tau_inv),
						par,
					);
					matmul::triangular::matmul(
						&mut y2,
						BlockStructure::Rectangular,
						Accum::Add,
						(&A22).adjoint(),
						BlockStructure::StrictTriangularUpper,
						&x2,
						BlockStructure::Rectangular,
						from_real(tau_inv),
						par,
					);
				}

				z!(&mut y2, &A21).for_each(|uz!(y, a)| *y = *y + mul_real(*a, tau_inv));

				*y1 = mul_real(*a11 + dot::inner_prod((&A21).transpose(), Conj::Yes, &x2, Conj::No), tau_inv);

				let b = mul_real(mul_pow2(*y1 + dot::inner_prod((&x2).transpose(), Conj::Yes, &y2, Conj::No), from_f64(0.5)), tau_inv);
				*y1 = *y1 - b;
				z!(&mut y2, &x2).for_each(|uz!(y, u)| {
					*y = *y - b * *u;
				});
			}
		}
	}

	if n > 0 {
		let n = n - 1;
		let A = (&A).submatrix(1, 0, n, n);
		let mut H = (&mut H).subcols_mut(0, n);

		let mut j = 0;
		while j < n {
			let b = Ord::min(b, n - j);

			let mut H = (&mut H).submatrix_mut(0, j, b, b);

			for k in 0..b {
				H[(k, k)] = copy(H[(0, k)]);
			}

			householder::upgrade_householder_factor(&mut H, A.submatrix(j, j, n - j, b), b, 1, par);
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
	use dyn_stack::GlobalMemBuffer;

	#[azucar::reborrow]
	#[azucar::infer]
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

			let A = &A + A.adjoint();

			let b = 3;
			let mut H = Mat::zeros(b, n - 1);

			let mut V = A.clone();
			let mut V = V.as_mut();
			tridiag_in_place(
				&mut V,
				&mut H,
				Par::Seq,
				DynStack::new(&mut GlobalMemBuffer::new(StackReq::all_of([
					householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(n - 1, b, n).unwrap(),
					tridiag_in_place_scratch::<f64>(n, Par::Seq, _).unwrap(),
				]))),
				_,
			);

			let mut A = A.clone();
			let mut A = A.as_mut();

			for iter in 0..2 {
				let mut A = if iter == 0 { &mut A } else { (&mut A).transpose_mut() };

				let n = n - 1;

				let V = (&V).submatrix(1, 0, n, n);
				let mut A = (&mut A).subrows_mut(1, n);
				let H = H.as_ref();

				householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
					V,
					H.as_ref(),
					if iter == 0 { Conj::Yes } else { Conj::No },
					&mut A,
					Par::Seq,
					DynStack::new(&mut GlobalMemBuffer::new(
						householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(n, b, n + 1).unwrap(),
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

	#[azucar::reborrow]
	#[azucar::infer]
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

			let A = &A + A.adjoint();

			let b = 3;
			let mut H = Mat::zeros(b, n - 1);

			let mut V = A.clone();
			let mut V = V.as_mut();
			tridiag_in_place(
				&mut V,
				H.as_mut(),
				Par::Seq,
				DynStack::new(&mut GlobalMemBuffer::new(tridiag_in_place_scratch::<c64>(n, Par::Seq, _).unwrap())),
				_,
			);

			let mut A = A.clone();
			let mut A = A.as_mut();

			for iter in 0..2 {
				let mut A = if iter == 0 { &mut A } else { (&mut A).transpose_mut() };

				let n = n - 1;

				let V = (&V).submatrix(1, 0, n, n);
				let mut A = (&mut A).subrows_mut(1, n);
				let H = H.as_ref();

				householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
					V,
					H.as_ref(),
					if iter == 0 { Conj::Yes } else { Conj::No },
					&mut A,
					Par::Seq,
					DynStack::new(&mut GlobalMemBuffer::new(
						householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<c64>(n, b, n + 1).unwrap(),
					)),
				);
			}

			let approx_eq = CwiseMat(ApproxEq::<c64>::eps());
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
