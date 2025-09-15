use crate::assert;
use crate::internal_prelude::*;
use linalg::householder::{self, HouseholderInfo};
use linalg::matmul::triangular::BlockStructure;
use linalg::matmul::{self, dot, matmul};
use linalg::triangular_solve;

/// hessenberg factorization tuning parameters
#[derive(Copy, Clone, Debug)]
pub struct HessenbergParams {
	/// threshold at which parallelism should be disabled
	pub par_threshold: usize,
	/// threshold at which blocking should be disabled
	pub blocking_threshold: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for HessenbergParams {
	fn auto() -> Self {
		Self {
			par_threshold: 192 * 256,
			blocking_threshold: 256 * 256,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

/// computes the layout of the workspace required to compute a matrix's hessenberg
/// decomposition
pub fn hessenberg_in_place_scratch<T: ComplexField>(dim: usize, blocksize: usize, par: Par, params: Spec<HessenbergParams, T>) -> StackReq {
	let params = params.config;
	let _ = par;
	let n = dim;
	if n * n < params.blocking_threshold {
		StackReq::any_of(&[StackReq::all_of(&[
			temp_mat_scratch::<T>(n, 1).array(3),
			temp_mat_scratch::<T>(n, par.degree()),
		])])
	} else {
		StackReq::all_of(&[
			temp_mat_scratch::<T>(n, blocksize),
			temp_mat_scratch::<T>(blocksize, 1),
			StackReq::any_of(&[
				StackReq::all_of(&[temp_mat_scratch::<T>(n, 1), temp_mat_scratch::<T>(n, par.degree())]),
				temp_mat_scratch::<T>(n, blocksize),
			]),
		])
	}
}

#[math]
fn hessenberg_fused_op_simd<T: ComplexField>(
	A: MatMut<'_, T, usize, usize, ContiguousFwd>,

	l_out: RowMut<'_, T, usize>,
	r_out: ColMut<'_, T, usize, ContiguousFwd>,
	l_in: RowRef<'_, T, usize, ContiguousFwd>,
	r_in: ColRef<'_, T, usize>,

	l0: ColRef<'_, T, usize, ContiguousFwd>,
	l1: ColRef<'_, T, usize, ContiguousFwd>,
	r0: RowRef<'_, T, usize>,
	r1: RowRef<'_, T, usize>,
	align: usize,
) {
	struct Impl<'a, 'M, 'N, T: ComplexField> {
		A: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,

		l_out: RowMut<'a, T, Dim<'N>>,
		r_out: ColMut<'a, T, Dim<'M>, ContiguousFwd>,
		l_in: RowRef<'a, T, Dim<'M>, ContiguousFwd>,
		r_in: ColRef<'a, T, Dim<'N>>,

		l0: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		l1: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		r0: RowRef<'a, T, Dim<'N>>,
		r1: RowRef<'a, T, Dim<'N>>,
		align: usize,
	}

	impl<'a, 'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'N, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self {
				mut A,
				mut l_out,
				mut r_out,
				l_in,
				r_in,
				l0,
				l1,
				r0,
				r1,
				align,
			} = self;

			let (m, n) = A.shape();

			let simd = SimdCtx::<T, S>::new_align(T::simd_ctx(simd), m, align);

			{
				let (head, body, tail) = simd.indices();
				if let Some(i) = head {
					simd.write(r_out.rb_mut(), i, simd.zero());
				}
				for i in body {
					simd.write(r_out.rb_mut(), i, simd.zero());
				}
				if let Some(i) = tail {
					simd.write(r_out.rb_mut(), i, simd.zero());
				}
			}

			let (head, body4, body1, tail) = simd.batch_indices::<4>();

			let l_in = l_in.transpose();

			for j in n.indices() {
				let mut A = A.rb_mut().col_mut(j);
				let r_in = simd.splat(r_in.at(j));
				let r0 = simd.splat(&(-r0[j]));
				let r1 = simd.splat(&(-r1[j]));

				let mut acc0 = simd.zero();
				let mut acc1 = simd.zero();
				let mut acc2 = simd.zero();
				let mut acc3 = simd.zero();

				if let Some(i0) = head {
					let mut a0 = simd.read(A.rb(), i0);
					a0 = simd.mul_add(simd.read(l0, i0), r0, a0);
					a0 = simd.conj_mul_add(r1, simd.read(l1, i0), a0);
					simd.write(A.rb_mut(), i0, a0);
					acc0 = simd.conj_mul_add(simd.read(l_in, i0), a0, acc0);
					let tmp = simd.read(r_out.rb(), i0);
					simd.write(r_out.rb_mut(), i0, simd.mul_add(a0, r_in, tmp));
				}
				for [i0, i1, i2, i3] in body4.clone() {
					{
						let mut a0 = simd.read(A.rb(), i0);
						a0 = simd.mul_add(simd.read(l0, i0), r0, a0);
						a0 = simd.conj_mul_add(r1, simd.read(l1, i0), a0);
						simd.write(A.rb_mut(), i0, a0);
						acc0 = simd.conj_mul_add(simd.read(l_in, i0), a0, acc0);
						let tmp = simd.read(r_out.rb(), i0);
						simd.write(r_out.rb_mut(), i0, simd.mul_add(a0, r_in, tmp));
					}
					{
						let mut a1 = simd.read(A.rb(), i1);
						a1 = simd.mul_add(simd.read(l0, i1), r0, a1);
						a1 = simd.conj_mul_add(r1, simd.read(l1, i1), a1);
						simd.write(A.rb_mut(), i1, a1);
						acc1 = simd.conj_mul_add(simd.read(l_in, i1), a1, acc1);
						let tmp = simd.read(r_out.rb(), i1);
						simd.write(r_out.rb_mut(), i1, simd.mul_add(a1, r_in, tmp));
					}
					{
						let mut a2 = simd.read(A.rb(), i2);
						a2 = simd.mul_add(simd.read(l0, i2), r0, a2);
						a2 = simd.conj_mul_add(r1, simd.read(l1, i2), a2);
						simd.write(A.rb_mut(), i2, a2);
						acc2 = simd.conj_mul_add(simd.read(l_in, i2), a2, acc2);
						let tmp = simd.read(r_out.rb(), i2);
						simd.write(r_out.rb_mut(), i2, simd.mul_add(a2, r_in, tmp));
					}
					{
						let mut a3 = simd.read(A.rb(), i3);
						a3 = simd.mul_add(simd.read(l0, i3), r0, a3);
						a3 = simd.conj_mul_add(r1, simd.read(l1, i3), a3);
						simd.write(A.rb_mut(), i3, a3);
						acc3 = simd.conj_mul_add(simd.read(l_in, i3), a3, acc3);
						let tmp = simd.read(r_out.rb(), i3);
						simd.write(r_out.rb_mut(), i3, simd.mul_add(a3, r_in, tmp));
					}
				}
				for i0 in body1.clone() {
					let mut a0 = simd.read(A.rb(), i0);
					a0 = simd.mul_add(simd.read(l0, i0), r0, a0);
					a0 = simd.conj_mul_add(r1, simd.read(l1, i0), a0);
					simd.write(A.rb_mut(), i0, a0);
					acc0 = simd.conj_mul_add(simd.read(l_in, i0), a0, acc0);
					let tmp = simd.read(r_out.rb(), i0);
					simd.write(r_out.rb_mut(), i0, simd.mul_add(a0, r_in, tmp));
				}
				if let Some(i0) = tail {
					let mut a0 = simd.read(A.rb(), i0);
					a0 = simd.mul_add(simd.read(l0, i0), r0, a0);
					a0 = simd.conj_mul_add(r1, simd.read(l1, i0), a0);
					simd.write(A.rb_mut(), i0, a0);
					acc0 = simd.conj_mul_add(simd.read(l_in, i0), a0, acc0);
					let tmp = simd.read(r_out.rb(), i0);
					simd.write(r_out.rb_mut(), i0, simd.mul_add(a0, r_in, tmp));
				}

				acc0 = simd.add(acc0, acc1);
				acc2 = simd.add(acc2, acc3);
				acc0 = simd.add(acc0, acc2);

				let l_out = l_out.rb_mut().at_mut(j);
				*l_out = simd.reduce_sum(acc0);
			}
		}
	}

	with_dim!(M, A.nrows());
	with_dim!(N, A.ncols());

	dispatch!(
		Impl {
			A: A.as_shape_mut(M, N),
			l_out: l_out.as_col_shape_mut(N),
			r_out: r_out.as_row_shape_mut(M),
			l_in: l_in.as_col_shape(M),
			r_in: r_in.as_row_shape(N),
			l0: l0.as_row_shape(M),
			l1: l1.as_row_shape(M),
			r0: r0.as_col_shape(N),
			r1: r1.as_col_shape(N),
			align,
		},
		Impl,
		T
	)
}

#[math]
fn hessenberg_fused_op_fallback<T: ComplexField>(
	A: MatMut<'_, T>,

	l_out: RowMut<'_, T>,
	r_out: ColMut<'_, T>,
	l_in: RowRef<'_, T>,
	r_in: ColRef<'_, T>,

	l0: ColRef<'_, T>,
	l1: ColRef<'_, T>,
	r0: RowRef<'_, T>,
	r1: RowRef<'_, T>,
) {
	let mut A = A;

	matmul(A.rb_mut(), Accum::Add, l0.as_mat(), r0.as_mat(), -one::<T>(), Par::Seq);
	matmul(A.rb_mut(), Accum::Add, l1.as_mat(), r1.as_mat().conjugate(), -one::<T>(), Par::Seq);

	matmul(r_out.as_mat_mut(), Accum::Replace, A.rb(), r_in.as_mat(), one(), Par::Seq);
	matmul(l_out.as_mat_mut(), Accum::Replace, l_in.as_mat().conjugate(), A.rb(), one(), Par::Seq);
}

fn hessenberg_fused_op<T: ComplexField>(
	A: MatMut<'_, T>,

	l_out: RowMut<'_, T>,
	r_out: ColMut<'_, T>,
	l_in: RowRef<'_, T>,
	r_in: ColRef<'_, T>,

	l0: ColRef<'_, T>,
	l1: ColRef<'_, T>,
	r0: RowRef<'_, T>,
	r1: RowRef<'_, T>,
	align: usize,
) {
	let mut A = A;
	let mut r_out = r_out;

	if try_const! { T::SIMD_CAPABILITIES.is_simd() } {
		if let (Some(A), Some(r_out), Some(l_in), Some(l0), Some(l1)) = (
			A.rb_mut().try_as_col_major_mut(),
			r_out.rb_mut().try_as_col_major_mut(),
			l_in.try_as_row_major(),
			l0.try_as_col_major(),
			l1.try_as_col_major(),
		) {
			hessenberg_fused_op_simd(A, l_out, r_out, l_in, r_in, l0, l1, r0, r1, align);
		} else {
			hessenberg_fused_op_fallback(A, l_out, r_out, l_in, r_in, l0, l1, r0, r1);
		}
	} else {
		hessenberg_fused_op_fallback(A, l_out, r_out, l_in, r_in, l0, l1, r0, r1);
	}
}

#[math]
fn hessenberg_rearranged_unblocked<T: ComplexField>(A: MatMut<'_, T>, H: MatMut<'_, T>, par: Par, stack: &mut MemStack, params: HessenbergParams) {
	assert!(all(A.nrows() == A.ncols(), H.ncols() == A.ncols().saturating_sub(1)));

	let n = A.nrows();
	let b = H.nrows();

	if n == 0 {
		return;
	}

	let mut A = A;
	let mut H = H;
	let mut par = par;

	{
		let mut H = H.rb_mut().row_mut(0);
		let (mut y, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
		let (mut z, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
		let (mut v, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
		let (mut w, _) = unsafe { temp_mat_uninit(n, par.degree(), stack) };

		let mut y = y.as_mat_mut().col_mut(0).transpose_mut();
		let mut z = z.as_mat_mut().col_mut(0);
		let mut v = v.as_mat_mut().col_mut(0).transpose_mut();
		let mut w = w.as_mat_mut();

		for k in 0..n {
			let (_, A01, A10, A11) = A.rb_mut().split_at_mut(k, k);

			let (_, mut A02) = A01.split_first_col_mut().unwrap();
			let (_, A20) = A10.split_first_row_mut().unwrap();
			let (mut A11, A12, A21, mut A22) = A11.split_at_mut(1, 1);

			let mut A12 = A12.row_mut(0);
			let mut A21 = A21.col_mut(0);

			let A11 = &mut A11[(0, 0)];

			let (y1, mut y2) = y.rb_mut().split_at_col_mut(k).1.split_at_col_mut(1);
			let y1 = copy(y1[0]);

			let (z1, mut z2) = z.rb_mut().split_at_row_mut(k).1.split_at_row_mut(1);
			let z1 = copy(z1[0]);

			let (_, mut v2) = v.rb_mut().split_at_col_mut(k).1.split_at_col_mut(1);
			let (mut w0, w12) = w.rb_mut().split_at_row_mut(k);
			let (_, mut w2) = w12.split_at_row_mut(1);

			if k > 0 {
				let p = k - 1;
				let u2 = A20.rb().col(p);

				*A11 = *A11 - y1 - z1;
				z!(&mut A12, &y2, u2.rb().transpose()).for_each(|uz!(a, y, u): Zip!(&mut T, &T, &T)| *a = *a - *y - z1 * conj(*u));
				z!(&mut A21, &u2, &z2).for_each(|uz!(a, u, z): Zip!(&mut T, &T, &T)| *a = *a - *u * y1 - *z);
			}

			{
				let n = n - k - 1;
				if n * n < params.par_threshold {
					par = Par::Seq;
				}
			}

			if k + 1 == n {
				break;
			}

			let beta;
			let tau_inv;
			{
				let (mut A11, mut A21) = A21.rb_mut().split_at_row_mut(1);
				let A11 = &mut A11[0];

				let HouseholderInfo { tau, .. } = householder::make_householder_in_place(A11, A21.rb_mut());
				tau_inv = recip(tau);
				beta = copy(*A11);
				*A11 = one();

				H[k] = from_real(tau);
			}

			let x2 = A21.rb();

			if k > 0 {
				let p = k - 1;
				let u2 = A20.rb().col(p);
				hessenberg_fused_op(
					A22.rb_mut(),
					v2.rb_mut(),
					w2.rb_mut().col_mut(0),
					x2.transpose(),
					x2,
					u2,
					z2.rb(),
					y2.rb(),
					u2.transpose(),
					simd_align(k + 1),
				);
				y2.copy_from(v2.rb());
				z2.copy_from(w2.rb().col(0));
			} else {
				matmul(z2.rb_mut().as_mat_mut(), Accum::Replace, A22.rb(), x2.as_mat(), one(), par);
				matmul(y2.rb_mut().as_mat_mut(), Accum::Replace, x2.adjoint().as_mat(), A22.rb(), one(), par);
			}

			let u2 = x2;

			let b = mul_real(
				mul_pow2(dot::inner_prod(u2.rb().transpose(), Conj::Yes, z2.rb(), Conj::No), from_f64(0.5)),
				tau_inv,
			);
			z!(&mut y2, u2.transpose()).for_each(|uz!(y, u): Zip!(&mut T, &T)| *y = mul_real(*y - b * conj(*u), tau_inv));
			z!(&mut z2, u2).for_each(|uz!(z, u): Zip!(&mut T, &T)| *z = mul_real(*z - b * *u, tau_inv));

			let dot = mul_real(dot::inner_prod(A12.rb(), Conj::No, u2.rb(), Conj::No), tau_inv);
			z!(&mut A12, u2.transpose()).for_each(|uz!(a, u): Zip!(&mut T, &T)| *a = *a - dot * conj(u));

			matmul(w0.rb_mut().col_mut(0).as_mat_mut(), Accum::Replace, A02.rb(), u2.as_mat(), one(), par);
			matmul(
				A02.rb_mut(),
				Accum::Add,
				w0.rb().col(0).as_mat(),
				u2.adjoint().as_mat(),
				-from_real::<T>(&tau_inv),
				par,
			);

			A21[0] = beta;
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

#[math]
fn hessenberg_gqvdg_unblocked<T: ComplexField>(
	A: MatMut<'_, T>,
	Z: MatMut<'_, T>,
	H: MatMut<'_, T>,
	beta: ColMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: HessenbergParams,
) {
	let n = A.nrows();
	let b = H.nrows();
	let mut A = A;
	let mut H = H;
	let mut Z = Z;
	_ = params;

	let (mut x, _) = unsafe { temp_mat_uninit(n, 1, stack) };
	let mut x = x.as_mat_mut().col_mut(0);
	let mut beta = beta;

	for k in 0..b {
		let mut x0 = x.rb_mut().subrows_mut(0, k);
		let (T00, T01, _, T11) = H.rb_mut().split_at_mut(k, k);
		let (mut T01, _) = T01.split_first_col_mut().unwrap();
		let (mut T11, _, _, _) = T11.split_at_mut(1, 1);

		let T11 = &mut T11[(0, 0)];

		let (U0, A12) = A.rb_mut().split_at_col_mut(k);
		let (mut A1, A2) = A12.split_first_col_mut().unwrap();

		let (Z0, Z12) = Z.rb_mut().split_at_col_mut(k);
		let (mut Z1, _) = Z12.split_first_col_mut().unwrap();

		let U0 = U0.rb();
		let Z0 = Z0.rb();
		let T00 = T00.rb();

		let (U00, U10) = U0.split_at_row(k);
		let (U10, U20) = U10.split_first_row().unwrap();

		x0.copy_from(U10.adjoint());
		triangular_solve::solve_upper_triangular_in_place(T00, x0.rb_mut().as_mat_mut(), par);
		matmul::matmul(A1.rb_mut().as_mat_mut(), Accum::Add, Z0, x0.rb().as_mat(), -one::<T>(), par);

		let (mut A01, A11) = A1.rb_mut().split_at_row_mut(k);
		let (mut A11, mut A21) = A11.split_at_row_mut(1);
		let A11 = &mut A11[0];

		{
			matmul::triangular::matmul(
				x0.rb_mut().as_mat_mut(),
				BlockStructure::Rectangular,
				Accum::Replace,
				U00.adjoint(),
				BlockStructure::StrictTriangularUpper,
				A01.rb().as_mat(),
				BlockStructure::Rectangular,
				one(),
				par,
			);
			z!(x0.rb_mut(), U10.transpose()).for_each(|uz!(x, u): Zip!(&mut T, &T)| *x = *x + *A11 * conj(*u));
			matmul::matmul(x0.rb_mut().as_mat_mut(), Accum::Add, U20.adjoint(), A21.rb().as_mat(), one(), par);
		}
		{
			triangular_solve::solve_lower_triangular_in_place(T00.adjoint(), x0.rb_mut().as_mat_mut(), par);
		}
		{
			matmul::triangular::matmul(
				A01.rb_mut().as_mat_mut(),
				BlockStructure::Rectangular,
				Accum::Add,
				U00,
				BlockStructure::StrictTriangularLower,
				x0.rb().as_mat(),
				BlockStructure::Rectangular,
				-one::<T>(),
				par,
			);
			*A11 = *A11 - dot::inner_prod(U10, Conj::No, x0.rb(), Conj::No);
			matmul::matmul(A21.rb_mut().as_mat_mut(), Accum::Add, U20, x0.rb().as_mat(), -one::<T>(), par);
		}

		if k + 1 < n {
			let (mut A11, mut A21) = A21.rb_mut().split_at_row_mut(1);
			let A11 = &mut A11[0];

			let HouseholderInfo { tau, .. } = householder::make_householder_in_place(A11, A21.rb_mut());

			beta[k] = copy(A11);
			*A11 = one();
			*T11 = from_real(tau);
		} else {
			*T11 = infinity();
		}

		matmul::matmul(Z1.rb_mut().as_mat_mut(), Accum::Replace, A2.rb(), A21.rb().as_mat(), one(), par);

		matmul::matmul(T01.rb_mut().as_mat_mut(), Accum::Replace, U20.adjoint(), A21.rb().as_mat(), one(), par);
	}
}

/// computes a matrix $A$'s hessenberg decomposition such that $A = Q H Q^H$
///
/// $H$ is a hessenberg matrix stored in the upper triangular half of $A$ (plus the subdiagonal)
///
/// $Q$ is a sequence of householder reflections stored in the unit lower triangular half of $A$
/// (excluding the diagonal), with the householder coefficients being stored in `householder`
#[track_caller]
pub fn hessenberg_in_place<T: ComplexField>(
	A: MatMut<'_, T>,
	householder: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<HessenbergParams, T>,
) {
	let params = params.config;
	assert!(all(A.nrows() == A.ncols(), householder.ncols() == A.ncols().saturating_sub(1)));

	let n = A.nrows().unbound();

	if n * n < params.blocking_threshold {
		hessenberg_rearranged_unblocked(A, householder, par, stack, params);
	} else {
		hessenberg_gqvdg_blocked(A, householder, par, stack, params);
	}
}

#[math]
fn hessenberg_gqvdg_blocked<T: ComplexField>(A: MatMut<'_, T>, H: MatMut<'_, T>, par: Par, stack: &mut MemStack, params: HessenbergParams) {
	let n = A.nrows();
	let b = H.nrows();
	let mut A = A;
	let mut H = H;
	let (mut Z, stack) = unsafe { temp_mat_uninit(n, b, stack) };
	let mut Z = Z.as_mat_mut();

	let mut j = 0;
	while j < n {
		let bs = Ord::min(b, n - j);
		let bs_u = Ord::min(bs, n - j - 1);

		let (mut beta, stack) = unsafe { temp_mat_uninit(bs, 1, stack) };
		let mut beta = beta.as_mat_mut().col_mut(0);

		{
			let mut T11 = H.rb_mut().submatrix_mut(0, j, bs_u, bs_u);
			{
				let A11 = A.rb_mut().submatrix_mut(j, j, n - j, n - j);
				let Z1 = Z.rb_mut().submatrix_mut(j, 0, n - j, bs);

				hessenberg_gqvdg_unblocked(A11, Z1, T11.rb_mut(), beta.rb_mut(), par, stack, params);
			}

			let (mut X, _) = unsafe { temp_mat_uninit(n, bs_u, stack) };
			let mut X = X.as_mat_mut();

			let (mut X0, X12) = X.rb_mut().split_at_row_mut(j);
			let (_, mut X2) = X12.split_at_row_mut(bs_u);

			let (_, Z12) = Z.rb_mut().subcols_mut(0, bs_u).split_at_row_mut(j);
			let (mut Z1, mut Z2) = Z12.split_at_row_mut(bs_u);

			let (_, A01, _, A11) = A.rb_mut().split_at_mut(j, j);
			let (mut A01, mut A02) = A01.split_at_col_mut(bs_u);
			let (A11, mut A12, A21, mut A22) = A11.split_at_mut(bs_u, bs_u);

			let U1 = A11.rb();
			let U2 = A21.rb();

			let T1 = T11.rb();

			matmul::triangular::matmul(
				X0.rb_mut(),
				BlockStructure::Rectangular,
				Accum::Replace,
				A01.rb(),
				BlockStructure::Rectangular,
				U1,
				BlockStructure::StrictTriangularLower,
				one(),
				par,
			);
			matmul::matmul(X0.rb_mut(), Accum::Add, A02.rb(), U2, one(), par);

			triangular_solve::solve_lower_triangular_in_place(T1.transpose(), X0.rb_mut().transpose_mut(), par);

			matmul::triangular::matmul(
				A01.rb_mut(),
				BlockStructure::Rectangular,
				Accum::Add,
				X0.rb(),
				BlockStructure::Rectangular,
				U1.adjoint(),
				BlockStructure::StrictTriangularUpper,
				-one::<T>(),
				par,
			);
			matmul::matmul(A02.rb_mut(), Accum::Add, X0.rb(), U2.adjoint(), -one::<T>(), par);

			triangular_solve::solve_lower_triangular_in_place(T1.transpose(), Z1.rb_mut().transpose_mut(), par);
			triangular_solve::solve_lower_triangular_in_place(T1.transpose(), Z2.rb_mut().transpose_mut(), par);

			matmul::matmul(A12.rb_mut(), Accum::Add, Z1.rb(), U2.adjoint(), -one::<T>(), par);
			matmul::matmul(A22.rb_mut(), Accum::Add, Z2.rb(), U2.adjoint(), -one::<T>(), par);

			let mut X = X2.rb_mut().transpose_mut();

			matmul::triangular::matmul(
				X.rb_mut(),
				BlockStructure::Rectangular,
				Accum::Replace,
				U1.adjoint(),
				BlockStructure::StrictTriangularUpper,
				A12.rb(),
				BlockStructure::Rectangular,
				one(),
				par,
			);
			matmul::matmul(X.rb_mut(), Accum::Add, U2.adjoint(), A22.rb(), one(), par);

			triangular_solve::solve_lower_triangular_in_place(T1.adjoint(), X.rb_mut(), par);

			matmul::triangular::matmul(
				A12.rb_mut(),
				BlockStructure::Rectangular,
				Accum::Add,
				U1,
				BlockStructure::StrictTriangularLower,
				X.rb(),
				BlockStructure::Rectangular,
				-one::<T>(),
				par,
			);
			matmul::matmul(A22.rb_mut(), Accum::Add, U2, X.rb(), -one::<T>(), par);
		}

		let n = n - j;
		let mut A = A.rb_mut().submatrix_mut(j, j, n, bs);
		for k in 0..bs {
			if k + 1 < n {
				A[(k + 1, k)] = copy(beta[k]);
			}
		}

		j += bs;
	}
}

#[cfg(test)]
mod tests {
	use dyn_stack::MemBuffer;
	use std::mem::MaybeUninit;

	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, assert, c64};

	#[test]
	fn test_hessenberg_real() {
		let rng = &mut StdRng::seed_from_u64(0);

		for n in [3, 4, 8, 16] {
			let A = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);

			let b = 3;
			let mut H = Mat::zeros(b, n - 1);

			let mut V = A.clone();
			let mut V = V.as_mut();
			hessenberg_rearranged_unblocked(
				V.rb_mut(),
				H.as_mut(),
				Par::Seq,
				MemStack::new(&mut [MaybeUninit::uninit(); 1024]),
				auto!(f64),
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
					if i > j + 1 {
						V[(i, j)] = 0.0;
					}
				}
			}

			assert!(V ~ A);
		}
	}

	#[test]
	fn test_hessenberg_cplx() {
		let rng = &mut StdRng::seed_from_u64(0);

		for n in [1, 2, 3, 4, 8, 16] {
			for par in [Par::Seq, Par::rayon(4)] {
				let A = CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: ComplexDistribution::new(StandardNormal, StandardNormal),
				}
				.rand::<Mat<c64>>(rng);

				let b = 3;
				let mut H = Mat::zeros(b, n - 1);

				let mut V = A.clone();
				let mut V = V.as_mut();
				hessenberg_rearranged_unblocked(
					V.rb_mut(),
					H.as_mut(),
					par,
					MemStack::new(&mut [MaybeUninit::uninit(); 8 * 1024]),
					HessenbergParams {
						par_threshold: 0,
						..auto!(c64)
					}
					.into(),
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
						if i > j + 1 {
							V[(i, j)] = c64::ZERO;
						}
					}
				}

				assert!(V ~ A);
			}
		}
	}

	#[test]
	fn test_hessenberg_cplx_gqvdg() {
		let rng = &mut StdRng::seed_from_u64(0);

		for n in [2, 3, 4, 8, 16, 21] {
			for par in [Par::Seq, Par::rayon(4)] {
				let b = 4;

				let A = CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: ComplexDistribution::new(StandardNormal, StandardNormal),
				}
				.rand::<Mat<c64, _, _>>(rng);

				let mut H = Mat::zeros(b, n - 1);

				let mut V = A.clone();
				let mut V = V.as_mut();
				hessenberg_gqvdg_blocked(
					V.rb_mut(),
					H.as_mut(),
					par,
					MemStack::new(&mut [MaybeUninit::uninit(); 16 * 1024]),
					HessenbergParams {
						par_threshold: 0,
						..auto!(c64)
					}
					.into(),
				);

				let mut A = A.clone();
				let mut A = A.as_mut();

				for iter in 0..2 {
					let mut A = if iter == 0 { A.rb_mut() } else { A.rb_mut().transpose_mut() };

					let n = n - 1;

					let V = V.rb().submatrix(1, 0, n, n);
					let mut A = A.rb_mut().subrows_mut(1, n);
					let H = H.as_ref().subcols(0, n);

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
						if i > j + 1 {
							V[(i, j)] = c64::ZERO;
						}
					}
				}

				assert!(V ~ A);
			}
		}
	}
}
