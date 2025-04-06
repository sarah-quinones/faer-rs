use crate::assert;
use crate::internal_prelude::*;
use linalg::householder::*;
use linalg::matmul::{dot, matmul};

/// computes the size and alignment of the workspace required to compute a matrix's
/// bidiagonalization
pub fn bidiag_in_place_scratch<T: ComplexField>(nrows: usize, ncols: usize, par: Par, params: Spec<BidiagParams, T>) -> StackReq {
	_ = par;
	_ = params;
	StackReq::all_of(&[temp_mat_scratch::<T>(nrows, 1), temp_mat_scratch::<T>(ncols, 1)])
}

/// bidiagonalization tuning parameters.
#[derive(Debug, Copy, Clone)]
pub struct BidiagParams {
	/// threshold at which parallelism should be disabled
	pub par_threshold: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for BidiagParams {
	fn auto() -> Self {
		Self {
			par_threshold: 192 * 256,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

/// computes a matrix $A$'s bidiagonalization such that $A = U B V^H$
///
/// $B$ is a bidiagonal matrix stored in $A$'s diagonal and superdiagonal
///
/// $U$ is a sequence of householder reflections stored in the unit lower triangular half of $A$,
/// with the householder coefficients being stored in `H_left`
///
/// $V$ is a sequence of householder reflections stored in the unit upper triangular half of $A$
/// (excluding the diagonal), with the householder coefficients being stored in `H_right`
#[math]
pub fn bidiag_in_place<T: ComplexField>(
	A: MatMut<'_, T>,
	H_left: MatMut<'_, T>,
	H_right: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<BidiagParams, T>,
) {
	let params = params.config;
	let m = A.nrows();
	let n = A.ncols();
	let size = Ord::min(m, n);
	let bl = H_left.nrows();
	let br = H_right.nrows();

	assert!(H_left.ncols() == size);
	assert!(H_right.ncols() == size.saturating_sub(1));

	let (mut y, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
	let (mut z, _) = unsafe { temp_mat_uninit(m, 1, stack) };

	let mut y = y.as_mat_mut().col_mut(0).transpose_mut();
	let mut z = z.as_mat_mut().col_mut(0);

	let mut A = A;
	let mut Hl = H_left;
	let mut Hr = H_right;
	let mut par = par;

	{
		let mut Hl = Hl.rb_mut().row_mut(0);
		let mut Hr = Hr.rb_mut().row_mut(0);

		for k in 0..size {
			let mut A = A.rb_mut();

			let (_, A01, A10, A11) = A.rb_mut().split_at_mut(k, k);

			let (_, A02) = A01.split_first_col().unwrap();
			let (A10, A20) = A10.split_first_row_mut().unwrap();
			let (mut A11, A12, A21, mut A22) = A11.split_at_mut(1, 1);

			let mut A12 = A12.row_mut(0);
			let mut A21 = A21.col_mut(0);

			let a11 = &mut A11[(0, 0)];

			let (y1, mut y2) = y.rb_mut().split_at_col_mut(k).1.split_at_col_mut(1);
			let (z1, mut z2) = z.rb_mut().split_at_row_mut(k).1.split_at_row_mut(1);

			let y1 = copy(y1[0]);
			let z1 = copy(z1[0]);

			if k > 0 {
				let k1 = k - 1;

				let up0 = copy(A10[k1]);
				let up = A20.rb().col(k1);
				let vp = A02.rb().row(k1);

				*a11 = *a11 - up0 * y1 - z1;
				z!(A21.rb_mut(), up.rb(), z2.rb()).for_each(|uz!(a, u, z)| *a = *a - *u * y1 - *z);
				z!(A12.rb_mut(), y2.rb(), vp.rb()).for_each(|uz!(a, y, v)| *a = *a - up0 * *y - z1 * *v);
			}

			let HouseholderInfo { tau: tl, .. } = make_householder_in_place(a11, A21.rb_mut());
			let tl_inv = recip(tl);
			Hl[k] = from_real(tl);

			if (m - k - 1) * (n - k - 1) < params.par_threshold {
				par = Par::Seq;
			}

			if k > 0 {
				let k1 = k - 1;

				let up = A20.rb().col(k1);
				let vp = A02.row(k1);

				match par {
					Par::Seq => bidiag_fused_op(A22.rb_mut(), A21.rb(), up.rb(), z2.rb(), y2.rb_mut(), vp.rb(), simd_align(k + 1)),
					#[cfg(feature = "rayon")]
					Par::Rayon(nthreads) => {
						use rayon::prelude::*;
						let nthreads = nthreads.get();

						A22.rb_mut()
							.par_col_partition_mut(nthreads)
							.zip_eq(y2.rb_mut().par_partition_mut(nthreads))
							.zip_eq(vp.par_partition(nthreads))
							.for_each(|((A22, y2), vp)| {
								bidiag_fused_op(A22, A21.rb(), up.rb(), z2.rb(), y2, vp.rb(), simd_align(k + 1));
							});
					},
				}
			} else {
				matmul(y2.rb_mut(), Accum::Replace, A21.rb().adjoint(), A22.rb(), one(), par);
			}

			z!(y2.rb_mut(), A12.rb_mut()).for_each(|uz!(y, a)| {
				*y = mul_real(*y + *a, tl_inv);
				*a = *a - *y;
			});
			let norm = A12.rb().norm_l2();
			let norm_inv = recip(norm);
			if norm != zero() {
				z!(A12.rb_mut()).for_each(|uz!(a)| *a = mul_real(a, norm_inv));
			}
			matmul(z2.rb_mut(), Accum::Replace, A22.rb(), A12.rb().adjoint(), one(), par);

			if k + 1 == size {
				break;
			}

			let (mut A12_a, mut A12_b) = A12.rb_mut().split_at_col_mut(1);
			let A22_a = A22.rb().col(0);
			let (y2_a, y2_b) = y2.rb().split_at_col(1);
			let y2_a = &y2_a[0];

			let a12_a = &mut A12_a[0];

			let HouseholderInfo {
				tau: tr,
				head_with_beta_inv: m,
				..
			} = make_householder_in_place(a12_a, A12_b.rb_mut().transpose_mut());
			let tr_inv = recip(tr);
			Hr[k] = from_real(tr);
			let beta = copy(*a12_a);
			*a12_a = mul_real(*a12_a, norm);

			let b = *y2_a + dot::inner_prod(y2_b, Conj::No, A12_b.rb().transpose(), Conj::Yes);

			if m != infinity() {
				z!(z2.rb_mut(), A21.rb(), A22_a.rb()).for_each(|uz!(z, u, a)| {
					let w = *z - *a * conj(beta);
					let w = w * conj(m);
					let w = w - *u * b;
					*z = mul_real(w, tr_inv);
				});
			} else {
				z!(z2.rb_mut(), A21.rb(), A22_a.rb()).for_each(|uz!(z, u, a)| {
					let w = *a - *u * b;
					*z = mul_real(w, tr_inv);
				});
			}
		}
	}

	let mut j = 0;
	while j < size {
		let bl = Ord::min(bl, size - j);

		let mut Hl = Hl.rb_mut().get_mut(..bl, j..j + bl);
		for k in 0..bl {
			Hl[(k, k)] = copy(Hl[(0, k)]);
		}

		upgrade_householder_factor(Hl.rb_mut(), A.rb().get(j.., j..j + bl), bl, 1, par);

		j += bl;
	}

	if size > 0 {
		let size = size - 1;
		let A = A.rb().get(..size, 1..);

		let mut Hr = Hr.rb_mut().get_mut(.., ..size);

		let mut j = 0;
		while j < size {
			let br = Ord::min(br, size - j);

			let mut Hr = Hr.rb_mut().get_mut(..br, j..j + br);

			for k in 0..br {
				Hr[(k, k)] = copy(Hr[(0, k)]);
			}

			upgrade_householder_factor(Hr.rb_mut(), A.transpose().get(j.., j..j + br), br, 1, par);
			j += br;
		}
	}
}

#[math]
fn bidiag_fused_op<T: ComplexField>(
	A22: MatMut<'_, T>,
	u: ColRef<'_, T>,
	up: ColRef<'_, T>,
	z: ColRef<'_, T>,
	y: RowMut<'_, T>,
	vp: RowRef<'_, T>,
	align: usize,
) {
	let mut A22 = A22;

	if try_const! { T::SIMD_CAPABILITIES.is_simd() } {
		if let (Some(A22), Some(u), Some(up), Some(z)) = (
			A22.rb_mut().try_as_col_major_mut(),
			u.try_as_col_major(),
			up.try_as_col_major(),
			z.try_as_col_major(),
		) {
			bidiag_fused_op_simd(A22, u, up, z, y, vp, align);
		} else {
			bidiag_fused_op_fallback(A22, u, up, z, y, vp);
		}
	} else {
		bidiag_fused_op_fallback(A22, u, up, z, y, vp);
	}
}

#[math]
fn bidiag_fused_op_fallback<T: ComplexField>(
	A22: MatMut<'_, T>,
	u: ColRef<'_, T>,
	up: ColRef<'_, T>,
	z: ColRef<'_, T>,
	y: RowMut<'_, T>,
	vp: RowRef<'_, T>,
) {
	let mut A22 = A22;
	let mut y = y;

	matmul(A22.rb_mut(), Accum::Add, up, y.rb(), -one::<T>(), Par::Seq);
	matmul(A22.rb_mut(), Accum::Add, z, vp, -one::<T>(), Par::Seq);
	matmul(y.rb_mut(), Accum::Replace, u.adjoint(), A22.rb(), one(), Par::Seq);
}

#[math]
fn bidiag_fused_op_simd<'M, 'N, T: ComplexField>(
	A22: MatMut<'_, T, usize, usize, ContiguousFwd>,
	u: ColRef<'_, T, usize, ContiguousFwd>,
	up: ColRef<'_, T, usize, ContiguousFwd>,
	z: ColRef<'_, T, usize, ContiguousFwd>,

	y: RowMut<'_, T, usize>,
	vp: RowRef<'_, T, usize>,

	align: usize,
) {
	struct Impl<'a, 'M, 'N, T: ComplexField> {
		A22: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		u: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		up: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		z: ColRef<'a, T, Dim<'M>, ContiguousFwd>,

		y: RowMut<'a, T, Dim<'N>>,
		vp: RowRef<'a, T, Dim<'N>>,

		align: usize,
	}

	impl<'a, 'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'N, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self {
				mut A22,
				u,
				up,
				z,
				mut y,
				vp,
				align,
			} = self;

			let m = A22.nrows();
			let n = A22.ncols();
			let simd = SimdCtx::<T, S>::new_align(T::simd_ctx(simd), m, align);
			let (head, body4, body1, tail) = simd.batch_indices::<4>();

			for j in n.indices() {
				let mut a = A22.rb_mut().col_mut(j);

				let mut acc0 = simd.zero();
				let mut acc1 = simd.zero();
				let mut acc2 = simd.zero();
				let mut acc3 = simd.zero();

				let yj = simd.splat(&-y[j]);
				let vj = simd.splat(&-vp[j]);

				if let Some(i0) = head {
					let mut a0 = simd.read(a.rb(), i0);
					a0 = simd.mul_add(simd.read(up, i0), yj, a0);
					a0 = simd.mul_add(simd.read(z, i0), vj, a0);
					simd.write(a.rb_mut(), i0, a0);

					acc0 = simd.conj_mul_add(simd.read(u, i0), a0, acc0);
				}

				for [i0, i1, i2, i3] in body4.clone() {
					{
						let mut a0 = simd.read(a.rb(), i0);
						a0 = simd.mul_add(simd.read(up, i0), yj, a0);
						a0 = simd.mul_add(simd.read(z, i0), vj, a0);
						simd.write(a.rb_mut(), i0, a0);

						acc0 = simd.conj_mul_add(simd.read(u, i0), a0, acc0);
					}
					{
						let mut a1 = simd.read(a.rb(), i1);
						a1 = simd.mul_add(simd.read(up, i1), yj, a1);
						a1 = simd.mul_add(simd.read(z, i1), vj, a1);
						simd.write(a.rb_mut(), i1, a1);

						acc1 = simd.conj_mul_add(simd.read(u, i1), a1, acc1);
					}
					{
						let mut a2 = simd.read(a.rb(), i2);
						a2 = simd.mul_add(simd.read(up, i2), yj, a2);
						a2 = simd.mul_add(simd.read(z, i2), vj, a2);
						simd.write(a.rb_mut(), i2, a2);

						acc2 = simd.conj_mul_add(simd.read(u, i2), a2, acc2);
					}
					{
						let mut a3 = simd.read(a.rb(), i3);
						a3 = simd.mul_add(simd.read(up, i3), yj, a3);
						a3 = simd.mul_add(simd.read(z, i3), vj, a3);
						simd.write(a.rb_mut(), i3, a3);

						acc3 = simd.conj_mul_add(simd.read(u, i3), a3, acc3);
					}
				}

				for i0 in body1.clone() {
					let mut a0 = simd.read(a.rb(), i0);
					a0 = simd.mul_add(simd.read(up, i0), yj, a0);
					a0 = simd.mul_add(simd.read(z, i0), vj, a0);
					simd.write(a.rb_mut(), i0, a0);

					acc0 = simd.conj_mul_add(simd.read(u, i0), a0, acc0);
				}
				if let Some(i0) = tail {
					let mut a0 = simd.read(a.rb(), i0);
					a0 = simd.mul_add(simd.read(up, i0), yj, a0);
					a0 = simd.mul_add(simd.read(z, i0), vj, a0);
					simd.write(a.rb_mut(), i0, a0);

					acc0 = simd.conj_mul_add(simd.read(u, i0), a0, acc0);
				}

				acc0 = simd.add(acc0, acc1);
				acc2 = simd.add(acc2, acc3);
				acc0 = simd.add(acc0, acc2);

				y[j] = simd.reduce_sum(acc0);
			}
		}
	}

	with_dim!(M, A22.nrows());
	with_dim!(N, A22.ncols());

	dispatch!(
		Impl {
			A22: A22.as_shape_mut(M, N),
			u: u.as_row_shape(M),
			up: up.as_row_shape(M),
			z: z.as_row_shape(M),
			y: y.as_col_shape_mut(N),
			vp: vp.as_col_shape(N),
			align,
		},
		Impl,
		T
	)
}

#[cfg(test)]
mod tests {
	use std::mem::MaybeUninit;

	use dyn_stack::MemBuffer;

	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, assert, c64};

	#[test]
	fn test_bidiag_real() {
		let rng = &mut StdRng::seed_from_u64(0);

		for (m, n) in [(8, 4), (8, 8)] {
			let size = Ord::min(m, n);

			let A = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);

			let bl = 4;
			let br = 3;
			let mut Hl = Mat::zeros(bl, size);
			let mut Hr = Mat::zeros(br, size - 1);

			let mut UV = A.clone();
			bidiag_in_place(
				UV.rb_mut(),
				Hl.rb_mut(),
				Hr.rb_mut(),
				Par::Seq,
				MemStack::new(&mut [MaybeUninit::uninit(); 1024]),
				default(),
			);

			let mut A = A.clone();
			let mut A = A.as_mut();

			apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
				UV.rb().get(.., ..size),
				Hl.rb(),
				Conj::Yes,
				A.rb_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(n - 1, 1, m),
				)),
			);

			let V = UV.rb().get(..size - 1, 1..size);
			let A1 = A.rb_mut().get_mut(.., 1..size);
			let Hr = Hr.as_ref();

			apply_block_householder_sequence_on_the_right_in_place_with_conj(
				V.transpose(),
				Hr.as_ref(),
				Conj::Yes,
				A1,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					apply_block_householder_sequence_on_the_right_in_place_scratch::<f64>(n - 1, 1, m),
				)),
			);

			let approx_eq = CwiseMat(ApproxEq::<f64>::eps());
			for j in 0..n {
				for i in 0..m {
					if i > j || j > i + 1 {
						UV[(i, j)] = 0.0;
					}
				}
			}

			assert!(UV ~ A);
		}
	}

	#[test]
	fn test_bidiag_cplx() {
		let rng = &mut StdRng::seed_from_u64(0);

		for (m, n) in [(8, 4), (8, 8)] {
			let size = Ord::min(m, n);
			let A = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			let bl = 4;
			let br = 3;
			let mut Hl = Mat::zeros(bl, size);
			let mut Hr = Mat::zeros(br, size - 1);

			let mut UV = A.clone();
			let mut UV = UV.as_mut();
			bidiag_in_place(
				UV.rb_mut(),
				Hl.rb_mut(),
				Hr.rb_mut(),
				Par::Seq,
				MemStack::new(&mut [MaybeUninit::uninit(); 1024]),
				default(),
			);

			let mut A = A.clone();
			let mut A = A.as_mut();

			apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
				UV.rb().subcols(0, size),
				Hl.rb(),
				Conj::Yes,
				A.rb_mut(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<c64>(n - 1, 1, m),
				)),
			);

			let V = UV.rb().get(..size - 1, 1..size);
			let A1 = A.rb_mut().get_mut(.., 1..size);
			let Hr = Hr.rb();

			apply_block_householder_sequence_on_the_right_in_place_with_conj(
				V.transpose(),
				Hr,
				Conj::Yes,
				A1,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					apply_block_householder_sequence_on_the_right_in_place_scratch::<c64>(n - 1, 1, m),
				)),
			);

			let approx_eq = CwiseMat(ApproxEq::eps());
			for j in 0..n {
				for i in 0..m {
					if i > j || j > i + 1 {
						UV[(i, j)] = c64::ZERO;
					}
				}
			}

			assert!(UV ~ A);
		}
	}
}
