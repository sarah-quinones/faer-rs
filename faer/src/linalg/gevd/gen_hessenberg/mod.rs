use crate::internal_prelude::*;
use crate::utils;
use equator::assert;
use linalg::matmul::{matmul, triangular as tr};

/// computes the layout of the workspace required to compute a matrix pair's generalized
/// hessenberg decomposition

pub fn generalized_hessenberg_scratch<T: ComplexField>(n: usize, params: GeneralizedHessenbergParams) -> StackReq {
	if params.block_size <= 1 || n <= params.block_size {
		StackReq::EMPTY
	} else {
		linalg::temp_mat_scratch::<T>(
			n,
			match params.block_size.checked_mul(6) {
				Some(n) => n,
				None => return StackReq::OVERFLOW,
			},
		)
	}
}

pub(crate) fn make_givens<T: ComplexField>(f: T, g: T) -> (T::Real, T, T) {
	if g == zero() {
		(one(), zero(), f)
	} else if f == zero() {
		let c = zero::<T::Real>();

		let d = g.abs();

		let d_inv = d.recip();

		let s = g.conj().mul_real(&d_inv);

		let r = d.to_cplx();

		(c, s, r)
	} else {
		let f1 = f.abs();

		let g1 = g.abs();

		let h1 = f1.hypot(g1);

		let ref h1_inv = h1.recip();

		let c = f1 * h1_inv;

		let r = f.mul_real(c.recip());

		let s = g.conj() * r.mul_real(h1_inv).mul_real(h1_inv);

		(c, s, r)
	}
}

pub(crate) fn rot<T: ComplexField>(c: T::Real, s: T, x: RowMut<'_, T>, y: RowMut<'_, T>) {
	let (c, s) = &(c, s);

	zip!(x, y).for_each(|unzip!(x, y): Zip!(&mut _, &mut _)| {
		(*x, *y) = (x.mul_real(c) + &*y * s, y.mul_real(c) - &*x * s.conj());
	});
}

/// generalized hessenberg factorization tuning parameters
#[derive(Copy, Clone, Debug)]

pub struct GeneralizedHessenbergParams {
	/// algorithm blocking parameter
	pub block_size: usize,
	/// threshold at which blocking should be disabled
	pub blocking_threshold: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for GeneralizedHessenbergParams {
	fn auto() -> Self {
		Self {
			block_size: 32,
			blocking_threshold: 256 * 256,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

pub(crate) fn trot<T: ComplexField>(c: T::Real, s: T, x: ColMut<'_, T>, y: ColMut<'_, T>) {
	rot(c, -s.conj(), x.transpose_mut(), y.transpose_mut());
}

fn generalized_hessenberg_unblocked<T: ComplexField>(
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	Q: Option<MatMut<'_, T>>,
	Z: Option<MatMut<'_, T>>,
	Q_is_I: bool,
	Z_is_I: bool,
	par: Par,
	stack: &mut MemStack,
	params: GeneralizedHessenbergParams,
) {
	_ = params;

	_ = stack;

	_ = Q_is_I;

	_ = Z_is_I;

	_ = par;

	let mut A = A;

	let mut B = B;

	let mut Q = Q;

	let mut Z = Z;

	let n = A.nrows();

	let (Q_nrows, Q_ncols) = Q.rb().map(|Q| (Q.nrows(), Q.ncols())).unwrap_or((n, n));

	let (Z_nrows, Z_ncols) = Z.rb().map(|Z| (Z.nrows(), Z.ncols())).unwrap_or((n, n));

	assert!(all(
		A.nrows() == n,
		A.ncols() == n,
		B.nrows() == n,
		B.ncols() == n,
		Q_nrows == n,
		Q_ncols == n,
		Z_nrows == n,
		Z_ncols == n,
	));

	if n <= 2 {
		return;
	}

	for jcol in 0..n - 2 {
		for jrow in (jcol + 2..n).rev() {
			let (c, s, r) = make_givens(A[(jrow - 1, jcol)].copy(), A[(jrow, jcol)].copy());

			A[(jrow - 1, jcol)] = r;

			A[(jrow, jcol)] = zero();

			let (x, y) = A.rb_mut().get_mut(.., jcol + 1..).two_rows_mut(jrow - 1, jrow);

			rot(c.copy(), s.copy(), x, y);

			let (x, y) = B.rb_mut().get_mut(.., jrow - 1..).two_rows_mut(jrow - 1, jrow);

			rot(c.copy(), s.copy(), x, y);

			if let Some(mut Q) = Q.rb_mut() {
				let (x, y) = Q.rb_mut().two_cols_mut(jrow - 1, jrow);

				rot(c.copy(), s.conj(), x.transpose_mut(), y.transpose_mut());
			}

			let (c, s, r) = make_givens(B[(jrow, jrow)].copy(), B[(jrow, jrow - 1)].copy());

			B[(jrow, jrow)] = r;

			B[(jrow, jrow - 1)] = zero();

			let (x, y) = A.rb_mut().two_cols_mut(jrow, jrow - 1);

			rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());

			let (x, y) = B.rb_mut().get_mut(..jrow, ..).two_cols_mut(jrow, jrow - 1);

			rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());

			if let Some(mut Z) = Z.rb_mut() {
				let (x, y) = Z.rb_mut().two_cols_mut(jrow, jrow - 1);

				rot(c, s, x.transpose_mut(), y.transpose_mut());
			}
		}
	}
}

fn apply_U_from_the_right<T: ComplexField>(M: MatMut<'_, T>, U: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
	let n = U.nrows() / 2;

	let (U11, U12, U21, U22) = U.split_at(n, n);

	let mut M = M;

	let (mut tmp, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(M.nrows(), M.ncols(), stack) };

	let mut tmp = tmp.as_mat_mut();

	let (mut T1, mut T2) = tmp.rb_mut().split_at_col_mut(n);

	let (M1, M2) = M.rb().split_at_col(n);

	utils::thread::join_raw(
		|par| {
			matmul(T1.rb_mut(), Accum::Replace, M1.rb(), U11, one(), par);

			tr::matmul(
				T1.rb_mut(),
				tr::BlockStructure::Rectangular,
				Accum::Add,
				M2.rb(),
				tr::BlockStructure::Rectangular,
				U21,
				tr::BlockStructure::TriangularUpper,
				one(),
				par,
			);
		},
		|par| {
			tr::matmul(
				T2.rb_mut(),
				tr::BlockStructure::Rectangular,
				Accum::Replace,
				M1.rb(),
				tr::BlockStructure::Rectangular,
				U12,
				tr::BlockStructure::TriangularLower,
				one(),
				par,
			);

			matmul(T2.rb_mut(), Accum::Add, M2.rb(), U22, one(), par);
		},
		par,
	);

	M.copy_from(tmp);
}

/// computes a matrix pair $(A, B)$'s generalized hessenberg decomposition such that
/// - B is an upper triangular matrix
/// - $A = Q H Z^H, A = Q T Z^H$,
/// - $H$ is a hessenberg matrix stored in the upper triangular half of $A$ (plus the subdiagonal),
/// - $T$ is an upper triangular matrix,
/// - $Q$ and $Z$ are unitary matrices.
///
/// # warning
/// $B$ is assumed to be upper triangular on input.
///
/// $Q$ and $Z$ are postmultiplied into the input-output parameters `Q` and `Z`.
///
/// i.e.: $Q_{\text{out}} = Q_{\text{in}} * Q$ and $Z_{\text{out}} = Z_{\text{in}} * Z$.
///
/// if this behavior is not desired then $Q$ and $Z$ should be overwritten by the identity matrix
/// before calling this function.

pub fn generalized_hessenberg<T: ComplexField>(
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	Q_inout: Option<MatMut<'_, T>>,
	Z_inout: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: GeneralizedHessenbergParams,
) {
	let n = A.nrows();

	let (Q_nrows, Q_ncols) = Q_inout.rb().map(|Q| (Q.nrows(), Q.ncols())).unwrap_or((n, n));

	let (Z_nrows, Z_ncols) = Z_inout.rb().map(|Z| (Z.nrows(), Z.ncols())).unwrap_or((n, n));

	let Q_is_I = Q_inout.rb().is_some_and(|Q| Q.is_identity());

	let Z_is_I = Z_inout.rb().is_some_and(|Z| Z.is_identity());

	assert!(all(
		A.nrows() == n,
		A.ncols() == n,
		B.nrows() == n,
		B.ncols() == n,
		Q_nrows == n,
		Q_ncols == n,
		Z_nrows == n,
		Z_ncols == n,
	));

	if n <= 2 {
		return;
	}

	if params.block_size <= 1 || A.nrows() <= params.block_size {
		return generalized_hessenberg_unblocked(A, B, Q_inout, Z_inout, Q_is_I, Z_is_I, par, stack, params);
	}

	let mut A = A;

	let mut B = B;

	let mut Q = Q_inout;

	let mut Z = Z_inout;

	let mut jcol = 0;

	while jcol < n - 2 {
		let nnb = params.block_size.clamp(1, n - 2 - jcol);

		let top = if jcol < 2 { 0 } else { jcol + 1 };

		let n2nb = ((n - 2 - jcol) / nnb).saturating_sub(1);

		let nblst = (n - 1 - jcol) - n2nb * nnb;

		alloca!('stack: {
			let mut work0 = unsafe { mat![uninit::<T>, nblst, nblst] };

			let mut work1 = unsafe { mat![uninit::<T>, 2 * nnb, 2 * nnb * n2nb] };
		});

		work0.fill(zero());

		work0.rb_mut().diagonal_mut().fill(one());

		work1.fill(zero());

		for i in 0..n2nb {
			work1.rb_mut().subcols_mut(2 * nnb * i, 2 * nnb).diagonal_mut().fill(one());
		}

		for j in jcol..jcol + nnb {
			for i in (j + 2..n).rev() {
				let (c, s, r) = make_givens(A[(i - 1, j)].copy(), A[(i, j)].copy());

				A[(i - 1, j)] = r;

				A[(i, j)] = c.to_cplx();

				B[(i, j)] = s;
			}

			let mut len = 2 + j - jcol;

			let jrow = j + n2nb * nnb + 2;

			for i in (jrow..n).rev() {
				let ref c = A[(i, j)].real();

				let ref s = B[(i, j)].copy();

				let col = nblst - (n - i) - 1;

				for jj in nblst - len..nblst {
					let ref temp0 = work0[(jj, col)].copy();

					let ref temp1 = work0[(jj, col + 1)].copy();

					work0[(jj, col + 1)] = temp1.mul_real(c) - s * temp0;

					work0[(jj, col)] = temp0.mul_real(c) + s.conj() * temp1;
				}

				len += 1;
			}

			let mut j0 = jrow;

			let mut idx = 0usize;

			while j0 > j + 2 {
				j0 -= nnb;

				let mut start = nnb;

				let mut len = 2 + j - jcol;

				let mut col = nnb + j - jcol;

				let mut U = work1.rb_mut().submatrix_mut(0, idx * 2 * nnb, 2 * nnb, 2 * nnb);

				for i in (j0..j0 + nnb).rev() {
					let ref c = A[(i, j)].real();

					let ref s = B[(i, j)].copy();

					col -= 1;

					start -= 1;

					for jj in start..start + len {
						let ref temp0 = U[(jj, col)].copy();

						let ref temp1 = U[(jj, col + 1)].copy();

						U[(jj, col + 1)] = temp1.mul_real(c) - s * temp0;

						U[(jj, col)] = s.conj() * temp1 + temp0.mul_real(c);
					}

					len += 1;
				}

				idx += 1;
			}

			for jj in (j + 1..n).rev() {
				for i in (j + 2..Ord::min(n, jj + 2)).rev() {
					let ref c = A[(i, j)].real();

					let ref s = B[(i, j)].copy();

					let ref temp0 = B[(i - 1, jj)].copy();

					let ref temp1 = B[(i, jj)].copy();

					B[(i, jj)] = temp1.mul_real(c) - s.conj() * temp0;

					B[(i - 1, jj)] = temp0.mul_real(c) + s * temp1;
				}

				if jj + 1 < n {
					let (c, s, r) = make_givens(B[(jj + 1, jj + 1)].copy(), B[(jj + 1, jj)].copy());

					B[(jj + 1, jj + 1)] = r;

					B[(jj + 1, jj)] = zero();

					let (bjj, bjj1) = B.rb_mut().get_mut(top..jj + 1, ..).two_cols_mut(jj, jj + 1);

					rot(c.copy(), s.copy(), bjj1.transpose_mut(), bjj.transpose_mut());

					A[(jj + 1, j)] = c.to_cplx();

					B[(jj + 1, j)] = -s.conj();
				}
			}

			let jj = (n - 2 - j) % 3;

			let mut i = n - 1 - j;

			while i > jj + 1 {
				i -= 3;

				let ref c0 = A[(j + i + 1, j)].real();

				let ref s0 = -&B[(j + i + 1, j)];

				let ref c1 = A[(j + i + 2, j)].real();

				let ref s1 = -&B[(j + i + 2, j)];

				let ref c2 = A[(j + i + 3, j)].real();

				let ref s2 = -&B[(j + i + 3, j)];

				for k in top..n {
					let ref temp0 = A[(k, j + i + 0)].copy();

					let ref temp1 = A[(k, j + i + 1)].copy();

					let ref temp2 = A[(k, j + i + 2)].copy();

					let ref temp3 = A[(k, j + i + 3)].copy();

					A[(k, j + i + 3)] = temp3.mul_real(c2) + temp2 * s2.conj();

					let temp2 = temp2.mul_real(c2) - temp3 * s2;

					A[(k, j + i + 2)] = temp2.mul_real(c1) + temp1 * s1.conj();

					let temp1 = temp1.mul_real(c1) - temp2 * s1;

					A[(k, j + i + 1)] = temp1.mul_real(c0) + temp0 * s0.conj();

					let temp0 = temp0.mul_real(c0) - temp1 * s0;

					A[(k, j + i + 0)] = temp0;
				}
			}

			for i in (1..jj + 1).rev() {
				let c = A[(j + i + 1, j)].real();

				let s = B[(j + i + 1, j)].copy();

				let (aj1, aj) = A.rb_mut().get_mut(top.., ..).two_cols_mut(j + i + 1, j + i);

				trot(c, s, aj1, aj);
			}

			if j < jcol + nnb - 1 {
				let len = j - jcol + 1;

				let jrow = n - nblst;

				{
					let (mut work2, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(nblst, 1, stack) };

					let mut work2 = work2.as_mat_mut().col_mut(0);

					matmul(
						work2.rb_mut().get_mut(..len),
						Accum::Replace,
						work0.rb().get(.., ..len).adjoint(),
						A.rb().get(jrow.., j + 1),
						one(),
						par,
					);

					tr::matmul(
						work2.rb_mut().get_mut(len..),
						tr::BlockStructure::Rectangular,
						Accum::Replace,
						work0.rb().get(..nblst - len, len..).adjoint(),
						tr::BlockStructure::TriangularUpper,
						A.rb().get(jrow..jrow + nblst - len, j + 1),
						tr::BlockStructure::Rectangular,
						one(),
						par,
					);

					matmul(
						work2.rb_mut().get_mut(len..),
						Accum::Add,
						work0.rb().get(nblst - len.., len..).adjoint(),
						A.rb().get(jrow + nblst - len.., j + 1),
						one(),
						par,
					);

					for i in jrow..jrow + nblst {
						A[(i, j + 1)] = work2[i - jrow].copy();
					}
				}

				let mut j0 = jrow;

				let mut idx = 0usize;

				while j0 > jcol + nnb {
					j0 -= nnb;

					let U = work1.rb().submatrix(0, idx * 2 * nnb, nnb + len, nnb + len);

					let (U11, U12, U21, U22) = U.split_at(nnb, len);

					let (mut work2, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(nnb + len, 1, stack) };

					let mut work2 = work2.as_mat_mut().col_mut(0);

					matmul(
						work2.rb_mut().get_mut(..len),
						Accum::Replace,
						U11.adjoint(),
						A.rb().get(j0..j0 + nnb, j + 1),
						one(),
						par,
					);

					tr::matmul(
						work2.rb_mut().get_mut(..len),
						tr::BlockStructure::Rectangular,
						Accum::Add,
						U21.adjoint(),
						tr::BlockStructure::TriangularLower,
						A.rb().get(j0 + nnb..j0 + nnb + len, j + 1),
						tr::BlockStructure::Rectangular,
						one(),
						par,
					);

					tr::matmul(
						work2.rb_mut().get_mut(len..),
						tr::BlockStructure::Rectangular,
						Accum::Replace,
						U12.adjoint(),
						tr::BlockStructure::TriangularUpper,
						A.rb().get(j0..j0 + nnb, j + 1),
						tr::BlockStructure::Rectangular,
						one(),
						par,
					);

					matmul(
						work2.rb_mut().get_mut(len..),
						Accum::Add,
						U22.adjoint(),
						A.rb().get(j0 + nnb..j0 + nnb + len, j + 1),
						one(),
						par,
					);

					for i in j0..j0 + len + nnb {
						A[(i, j + 1)] = work2[i - j0].copy();
					}

					idx += 1;
				}
			}
		}

		let cola = n - jcol - nnb;

		let j = n - nblst;

		{
			let (mut work2, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(nblst, cola, stack) };

			let mut work2 = work2.as_mat_mut();

			matmul(
				work2.rb_mut(),
				Accum::Replace,
				work0.rb().adjoint(),
				A.rb().get(j.., jcol + nnb..),
				one(),
				par,
			);

			A.rb_mut().get_mut(j.., jcol + nnb..).copy_from(&work2);
		}

		let mut j0 = j;

		let mut idx = 0usize;

		while j0 > jcol + nnb {
			j0 -= nnb;

			let U = work1.rb().submatrix(0, idx * 2 * nnb, 2 * nnb, 2 * nnb);

			let (U11, U12, U21, U22) = U.split_at(nnb, nnb);

			let (mut work2, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(2 * nnb, cola, stack) };

			let mut work2 = work2.as_mat_mut();

			let (mut work2_top, mut work2_bot) = work2.rb_mut().split_at_row_mut(nnb);

			utils::thread::join_raw(
				|par| {
					matmul(
						work2_top.rb_mut(),
						Accum::Replace,
						U11.adjoint(),
						A.rb().get(j0..j0 + nnb, jcol + nnb..),
						one(),
						par,
					);

					tr::matmul(
						work2_top.rb_mut(),
						tr::BlockStructure::Rectangular,
						Accum::Add,
						U21.adjoint(),
						tr::BlockStructure::TriangularLower,
						A.rb().get(j0 + nnb..j0 + 2 * nnb, jcol + nnb..),
						tr::BlockStructure::Rectangular,
						one(),
						par,
					);
				},
				|par| {
					tr::matmul(
						work2_bot.rb_mut(),
						tr::BlockStructure::Rectangular,
						Accum::Replace,
						U12.adjoint(),
						tr::BlockStructure::TriangularUpper,
						A.rb().get(j0..j0 + nnb, jcol + nnb..),
						tr::BlockStructure::Rectangular,
						one(),
						par,
					);

					matmul(
						work2_bot.rb_mut(),
						Accum::Add,
						U22.adjoint(),
						A.rb().get(j0 + nnb..j0 + 2 * nnb, jcol + nnb..),
						one(),
						par,
					);
				},
				par,
			);

			A.rb_mut().get_mut(j0..j0 + 2 * nnb, jcol + nnb..).copy_from(&work2);

			idx += 1;
		}

		if let Some(mut Q) = Q.rb_mut() {
			let topq;

			let nh;

			if Q_is_I {
				topq = Ord::max(1, j - jcol);

				nh = n - topq;
			} else {
				topq = 0;

				nh = n;
			}

			{
				let (mut work2, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(nh, nblst, stack) };

				let mut work2 = work2.as_mat_mut();

				matmul(work2.rb_mut(), Accum::Replace, Q.rb().get(topq.., j..), work0.rb(), one(), par);

				Q.rb_mut().get_mut(topq.., j..).copy_from(&work2);
			}

			let mut j0 = j;

			let mut idx = 0usize;

			while j0 > jcol + nnb {
				j0 -= nnb;

				let topq;

				if Q_is_I {
					topq = Ord::max(1, j0 - jcol);
				} else {
					topq = 0;
				}

				let U = work1.rb().submatrix(0, idx * 2 * nnb, 2 * nnb, 2 * nnb);

				apply_U_from_the_right(Q.rb_mut().get_mut(topq.., j0..j0 + 2 * nnb), U, par, stack);

				idx += 1;
			}
		}

		if Z.is_some() || top > 0 {
			work0.fill(zero());

			work0.rb_mut().diagonal_mut().fill(one());

			work1.fill(zero());

			for i in 0..n2nb {
				work1
					.rb_mut()
					.subcols_mut(2 * nnb * i, 2 * nnb)
					.diagonal_mut()
					.column_vector_mut()
					.fill(one());
			}

			for j in jcol..jcol + nnb {
				let mut len = 2 + j - jcol;

				let jrow = j + n2nb * nnb + 2;

				for i in (jrow..n).rev() {
					let ref c = A[(i, j)].real();

					let ref s = B[(i, j)].copy();

					A[(i, j)] = zero();

					B[(i, j)] = zero();

					let col = nblst - (n - i) - 1;

					for jj in nblst - len..nblst {
						let ref temp0 = work0[(jj, col)].copy();

						let ref temp1 = work0[(jj, col + 1)].copy();

						work0[(jj, col + 1)] = temp1.mul_real(c) - temp0 * s.conj();

						work0[(jj, col)] = temp0.mul_real(c) + temp1 * s;
					}

					len += 1;
				}

				let mut j0 = jrow;

				let mut idx = 0usize;

				while j0 > j + 2 {
					j0 -= nnb;

					let mut len = 2 + j - jcol;

					let mut col = nnb + j - jcol;

					let mut start = nnb;

					let mut U = work1.rb_mut().submatrix_mut(0, idx * 2 * nnb, 2 * nnb, 2 * nnb);

					for i in (j0..j0 + nnb).rev() {
						let ref c = A[(i, j)].real();

						let ref s = B[(i, j)].copy();

						A[(i, j)] = zero();

						B[(i, j)] = zero();

						col -= 1;

						start -= 1;

						for jj in start..start + len {
							let ref temp0 = U[(jj, col)].copy();

							let ref temp1 = U[(jj, col + 1)].copy();

							U[(jj, col + 1)] = temp1.mul_real(c) - temp0 * s.conj();

							U[(jj, col)] = temp0.mul_real(c) + temp1 * s;
						}

						len += 1;
					}

					idx += 1;
				}
			}
		} else {
			zip!(A.rb_mut().get_mut(jcol + 2.., jcol..n - jcol - 2)).for_each(|unzip!(x)| *x = zero());

			zip!(B.rb_mut().get_mut(jcol + 2.., jcol..n - jcol - 2)).for_each(|unzip!(x)| *x = zero());
		}

		if top > 0 {
			for mut M in [A.rb_mut(), B.rb_mut()] {
				let j = n - nblst;

				{
					let (mut work2, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(top, nblst, stack) };

					let mut work2 = work2.as_mat_mut();

					matmul(work2.rb_mut(), Accum::Replace, M.rb().get(..top, j..), work0.rb(), one(), par);

					M.rb_mut().get_mut(..top, j..).copy_from(&work2);
				}

				let mut j0 = j;

				let mut idx = 0usize;

				while j0 > jcol + nnb {
					j0 -= nnb;

					let U = work1.rb().submatrix(0, idx * 2 * nnb, 2 * nnb, 2 * nnb);

					apply_U_from_the_right(M.rb_mut().get_mut(..top, j0..j0 + 2 * nnb), U, par, stack);

					idx += 1;
				}
			}
		}

		if let Some(mut Z) = Z.rb_mut() {
			let topq;

			let nh;

			if Z_is_I {
				topq = Ord::max(1, j - jcol);

				nh = n - topq;
			} else {
				topq = 0;

				nh = n;
			}

			{
				let (mut work2, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(nh, nblst, stack) };

				let mut work2 = work2.as_mat_mut();

				matmul(work2.rb_mut(), Accum::Replace, Z.rb().get(topq.., j..), work0.rb(), one(), par);

				Z.rb_mut().get_mut(topq.., j..).copy_from(&work2);
			}

			let mut j0 = j;

			let mut idx = 0usize;

			while j0 > jcol + nnb {
				j0 -= nnb;

				let topq;

				if Z_is_I {
					topq = Ord::max(1, j0 - jcol);
				} else {
					topq = 0;
				}

				let U = work1.rb().submatrix(0, idx * 2 * nnb, 2 * nnb, 2 * nnb);

				apply_U_from_the_right(Z.rb_mut().get_mut(topq.., j0..j0 + 2 * nnb), U, par, stack);

				idx += 1;
			}
		}

		jcol += nnb;
	}
}

#[cfg(test)]

mod tests {

	use super::*;
	use crate::{linalg, stats};
	use dyn_stack::MemBuffer;
	use equator::assert;
	use stats::prelude::*;

	#[test]

	fn test_givens() {
		let rng = &mut StdRng::seed_from_u64(0);

		let rand = ComplexDistribution::new(StandardUniform, StandardUniform);

		let mut sample = || -> c32 { rand.sample(rng) };

		let shift = c32::new(0.5, 0.5);

		let x = sample() - shift;

		let y = sample() - shift;

		let (c, s, r) = make_givens(x, y);

		assert!((x * c + y * s - r).norm() < 1e-6);

		assert!((-x * s.conj() + y * c).norm() < 1e-6);
	}

	#[test]

	fn test_hessenberg() {
		let rng = &mut StdRng::seed_from_u64(0);

		for n in [12, 35, 128, 255] {
			let rand = stats::CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			};

			let mut sample = || -> Mat<c64> { rand.rand(rng) };

			let A = sample();

			let mut B = sample();

			zip!(&mut B).for_each_triangular_lower(linalg::zip::Diag::Skip, |unzip!(x)| {
				*x = 0.0.into();
			});

			let B = B;

			let mut mem = MemBuffer::new(generalized_hessenberg_scratch::<c64>(
				n,
				GeneralizedHessenbergParams {
					block_size: 32,
					..auto!(c64)
				},
			));

			for gen_hessenberg in [
				|A: MatMut<'_, _>, B: MatMut<'_, _>, Q: Option<MatMut<'_, _>>, Z: Option<MatMut<'_, _>>, _, _, par, stack: &mut MemStack, params| {
					generalized_hessenberg(A, B, Q, Z, par, stack, params)
				},
				generalized_hessenberg_unblocked,
			] {
				for (Q_is_I, Z_is_I) in [(false, false), (true, true), (true, false), (false, true)] {
					let Q0 = &if Q_is_I {
						Mat::identity(n, n)
					} else {
						UnitaryMat {
							dim: n,
							standard_normal: ComplexDistribution::new(StandardNormal, StandardNormal),
						}
						.rand::<Mat<c64>>(rng)
					};

					let Z0 = &if Z_is_I {
						Mat::identity(n, n)
					} else {
						UnitaryMat {
							dim: n,
							standard_normal: ComplexDistribution::new(StandardNormal, StandardNormal),
						}
						.rand::<Mat<c64>>(rng)
					};

					let mut Q = Q0.to_owned();

					let mut Z = Z0.to_owned();

					let mut H = A.clone();

					let mut T = B.clone();

					gen_hessenberg(
						H.as_mut(),
						T.as_mut(),
						Some(Q.as_mut()),
						Some(Z.as_mut()),
						Q_is_I,
						Z_is_I,
						Par::Seq,
						MemStack::new(&mut mem),
						GeneralizedHessenbergParams {
							block_size: 32,
							..auto!(c64)
						},
					);

					Q = Q0.adjoint() * &Q;

					Z = Z0.adjoint() * &Z;

					assert!((&Q * &H * Z.adjoint() - &A).norm_max() < 1e-13);

					assert!((&Q * &T * Z.adjoint() - &B).norm_max() < 1e-13);

					for j in 0..n {
						for i in j + 1..n {
							assert!(T[(i, j)] == c64::ZERO);
						}

						for i in j + 2..n {
							assert!(H[(i, j)] == c64::ZERO);
						}
					}
				}
			}
		}
	}
}
