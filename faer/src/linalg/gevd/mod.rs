//! low level implementation of the generalized eigenvalue decomposition of a square matrix pair.
//!
//! the generalized eigenvalue decomposition of a matrix pair $(A, B)$ of shape $(n, n)$ is a
//! decomposition into components $S$, $U$ such that:
//!
//! - $U$ has shape $(n, n)$ and is invertible
//! - $S$ has shape $(n, n)$ and is a diagonal matrix
//! - and finally:
//!
//! $$A = B U S U^{-1}$$
//!
//! if $A$ is self-adjoint and $B$ is positive definite, then $U$ can be made unitary ($U^{-1} =
//! U^H$), and $S$ is real valued.
//!
//! the implementation can choose to compute the left and/or right generalized eigenvectors
//! separately, and the generalized eigenvalues (diagonal of $S$) are represented as a ratio $\alpha
//! / \beta$ to improve numerical stability.

use crate::internal_prelude::*;
use complex::Complex;
use equator::assert;

pub use linalg::evd::ComputeEigenvectors;

/// generalized schur decomposition parameters
#[derive(Clone, Copy, Debug)]
pub struct GeneralizedSchurParams {
	/// An estimate of the relative cost of flops within the near-the-diagonal shift chase compared
	/// to flops within the matmul calls of a QZ sweep.
	pub relative_cost_estimate_of_shift_chase_to_matmul: fn(matrix_dimension: usize, active_block_dimension: usize) -> usize,
	/// Function that returns the number of shifts to use for a given matrix size
	pub recommended_shift_count: fn(matrix_dimension: usize, active_block_dimension: usize) -> usize,
	/// Function that returns the deflation window to use for a given matrix size
	pub recommended_deflation_window: fn(matrix_dimension: usize, active_block_dimension: usize) -> usize,
	/// Threshold to switch between blocked and unblocked code
	pub blocking_threshold: usize,
	/// Threshold of percent of aggressive-early-deflation window that must converge to skip a
	/// sweep
	pub nibble_threshold: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

/// schur to eigendecomposition conversion parameters
#[derive(Clone, Copy, Debug)]
pub struct GevdFromSchurParams {
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

/// eigendecomposition tuning parameters
#[derive(Clone, Copy, Debug)]
pub struct GevdParams {
	/// hessenberg parameters
	pub hessenberg: gen_hessenberg::GeneralizedHessenbergParams,
	/// schur from hessenberg conversion parameters
	pub schur: GeneralizedSchurParams,
	/// eigendecomposition from schur conversion parameters
	pub evd_from_schur: GevdFromSchurParams,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

fn default_relative_cost_estimate_of_shift_chase_to_matmul(n: usize, nh: usize) -> usize {
	_ = (n, nh);
	10
}

impl<T: ComplexField> Auto<T> for GeneralizedSchurParams {
	fn auto() -> Self {
		let schur: linalg::evd::SchurParams = auto!(T);

		Self {
			relative_cost_estimate_of_shift_chase_to_matmul: default_relative_cost_estimate_of_shift_chase_to_matmul,
			recommended_shift_count: schur.recommended_shift_count,
			recommended_deflation_window: schur.recommended_deflation_window,
			blocking_threshold: schur.blocking_threshold,
			nibble_threshold: schur.nibble_threshold,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

impl<T: ComplexField> Auto<T> for GevdParams {
	fn auto() -> Self {
		Self {
			hessenberg: auto!(T),
			schur: auto!(T),
			evd_from_schur: auto!(T),
			non_exhaustive: NonExhaustive(()),
		}
	}
}

impl<T: ComplexField> Auto<T> for GevdFromSchurParams {
	fn auto() -> Self {
		Self {
			non_exhaustive: NonExhaustive(()),
		}
	}
}

/// hessenberg decomposition
pub mod gen_hessenberg;

/// $QZ$ decomposition for complex matrices
pub mod qz_cplx;
/// $QZ$ decomposition for real matrices
pub mod qz_real;

#[track_caller]
#[math]
fn compute_gevd_generic<T: ComplexField>(
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	alpha_re: ColMut<'_, T>,
	alpha_im: ColMut<'_, T>,
	beta: ColMut<'_, T>,
	u_left: Option<MatMut<'_, T>>,
	u_right: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: GevdParams,

	hessenberg_to_qz: fn(
		A: MatMut<'_, T>,
		B: MatMut<'_, T>,
		Q: Option<MatMut<'_, T>>,
		Z: Option<MatMut<'_, T>>,
		alphar: ColMut<'_, T>,
		alphai: ColMut<'_, T>,
		beta: ColMut<'_, T>,
		eigenvectors: ComputeEigenvectors,
		par: Par,
		params: GeneralizedSchurParams,
		stack: &mut MemStack,
	),
	qz_to_gevd: fn(A: MatRef<'_, T>, B: MatRef<'_, T>, Q: Option<MatMut<'_, T>>, Z: Option<MatMut<'_, T>>, par: Par, stack: &mut MemStack),
) {
	let n = A.nrows();
	assert!(all(
		A.nrows() == n,
		A.ncols() == n,
		B.nrows() == n,
		B.ncols() == n,
		alpha_re.nrows() == n,
		if const { T::IS_REAL } { alpha_im.nrows() } else { n } == n,
		beta.nrows() == n,
	));
	if let Some(u_left) = u_left.rb() {
		assert!(all(u_left.nrows() == n, u_left.ncols() == n));
	}
	if let Some(u_right) = u_right.rb() {
		assert!(all(u_right.nrows() == n, u_right.ncols() == n));
	}

	if n == 0 {
		return;
	}

	#[cfg(feature = "perf-warn")]
	{
		for u in [u_left.rb(), u_right.rb()] {
			if let Some(matrix) = u.rb() {
				if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(GEVD_WARN) {
					if matrix.col_stride().unsigned_abs() == 1 {
						log::warn!(target: "faer_perf", "GEVD prefers column-major eigenvector matrix. Found row-major matrix.");
					} else {
						log::warn!(target: "faer_perf", "GEVD prefers column-major eigenvector matrix. Found matrix with generic strides.");
					}
				}
			}
		}

		for M in [A.rb(), B.rb()] {
			if M.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(GEVD_WARN) {
				if M.col_stride().unsigned_abs() == 1 {
					log::warn!(target: "faer_perf", "GEVD prefers column-major input matrix. Found row-major matrix.");
				} else {
					log::warn!(target: "faer_perf", "GEVD prefers column-major input matrix. Found matrix with generic strides.");
				}
			}
		}
	}

	if !(A.is_all_finite() && B.is_all_finite()) {
		{ alpha_re }.fill(nan());
		{ alpha_im }.fill(nan());
		{ beta }.fill(nan());
		for u in [u_left, u_right] {
			if let Some(mut u) = u {
				u.fill(nan());
			}
		}
		return;
	}
	let need_qz = u_left.is_some() || u_right.is_some();

	let mut A = A;
	let mut B = B;
	let mut u_left = u_left;
	let mut u_right = u_right;
	let mut alpha_re = alpha_re;
	let mut alpha_im = alpha_im;
	let mut beta = beta;

	for u in [u_left.rb_mut(), u_right.rb_mut()] {
		if let Some(u) = u {
			u.diagonal_mut().column_vector_mut().fill(one());
		}
	}

	{
		let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n, n);
		let (mut householder, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(blocksize, n, stack) };
		let mut householder = householder.as_mat_mut();
		linalg::qr::no_pivoting::factor::qr_in_place(B.rb_mut(), householder.rb_mut(), par, stack, default());
		linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
			B.rb(),
			householder.rb(),
			Conj::Yes,
			A.rb_mut(),
			par,
			stack,
		);

		if let Some(u_left) = u_left.rb_mut() {
			linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
				B.rb(),
				householder.rb(),
				Conj::No,
				u_left,
				par,
				stack,
			);
		}

		zip!(B.rb_mut()).for_each_triangular_lower(linalg::zip::Diag::Skip, |unzip!(x)| *x = zero());
	}

	gen_hessenberg::generalized_hessenberg(A.rb_mut(), B.rb_mut(), u_left.rb_mut(), u_right.rb_mut(), par, stack, params.hessenberg);

	hessenberg_to_qz(
		A.rb_mut(),
		B.rb_mut(),
		u_left.rb_mut(),
		u_right.rb_mut(),
		alpha_re.rb_mut(),
		alpha_im.rb_mut(),
		beta.rb_mut(),
		if need_qz { ComputeEigenvectors::Yes } else { ComputeEigenvectors::No },
		par,
		params.schur,
		stack,
	);

	qz_to_gevd(A.rb(), B.rb(), u_left.rb_mut(), u_right.rb_mut(), par, stack);
}

#[math]
fn solve_shifted_1x1<T: ComplexField>(smin: T::Real, ca: T::Real, A: T, d0: T, B: &mut T, w: T) {
	let safmin = min_positive::<T::Real>();
	let smlnum = safmin + safmin;
	let smin = max(smin, smlnum);

	let CR = mul_real(A, ca) - w * d0;

	let cmax = abs(CR);
	if cmax < smin {
		// use smin * I
		let smin_inv = recip(smin);
		*B = mul_real(*B, smin_inv);
	}

	// w is real
	let C = recip(CR);
	*B = *B * C;
}

#[math]
fn solve_complex_shifted_1x1<T: RealField>(smin: T, ca: T, A: MatRef<'_, T>, d0: T, mut B: MatMut<'_, T>, wr: T, wi: T) {
	let nw = B.ncols();
	let safmin = min_positive::<T>();
	let smlnum = safmin + safmin;
	let smin = max(smin, smlnum);

	// Compute the real part of  C = ca A - w D
	let CR = ca * A[(0, 0)] - wr * d0;
	let CI = -wi * d0;

	let cmax = max(abs(CR), abs(CI));
	if cmax < smin {
		// use smin * I
		let smin_inv = recip(smin);
		zip!(B.rb_mut()).for_each(|unzip!(x)| *x = *x * smin_inv);
	}

	if nw == 1 {
		// w is real
		let C = recip(CR);
		zip!(B.rb_mut()).for_each(|unzip!(x)| *x = *x * C);
	} else {
		// w is complex
		let (Br, Bi) = B.two_cols_mut(0, 1);
		let C = recip(Complex { re: CR, im: CI });
		zip!(Br, Bi).for_each(|unzip!(re, im)| {
			(*re, *im) = (*re * C.re - *im * C.im, *re * C.im + *im * C.re);
		});
	}
}

#[math]
fn solve_complex_shifted_2x2<T: RealField>(smin: T, ca: T, A: MatRef<'_, T>, d0: T, d1: T, mut B: MatMut<'_, T>, wr: T, wi: T, stack: &mut MemStack) {
	let nw = B.ncols();
	let zero = zero::<T>;
	let safmin = min_positive::<T>();
	let smlnum = safmin + safmin;
	let smin = max(smin, smlnum);

	// Compute the real part of  C = ca A - w D

	if nw == 1 {
		// w is real

		let (mut C, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(2, 2, stack) };
		let mut C = C.as_mat_mut();
		let mut row_perm_fwd = [0usize; 2];
		let mut row_perm_inv = [0usize; 2];
		let mut col_perm_fwd = [0usize; 2];
		let mut col_perm_inv = [0usize; 2];

		C[(0, 0)] = ca * A[(0, 0)] - wr * d0;
		C[(1, 0)] = ca * A[(1, 0)];
		C[(0, 1)] = ca * A[(0, 1)];
		C[(1, 1)] = ca * A[(1, 1)] - wr * d1;

		let cmax = C.norm_max();
		if cmax < smin {
			// use smin * I
			let smin_inv = recip(smin);
			zip!(B.rb_mut()).for_each(|unzip!(x)| *x = *x * smin_inv);
		}
		let (_, row_perm, col_perm) = linalg::lu::full_pivoting::factor::lu_in_place(
			C.rb_mut(),
			&mut row_perm_fwd,
			&mut row_perm_inv,
			&mut col_perm_fwd,
			&mut col_perm_inv,
			Par::Seq,
			stack,
			Default::default(),
		);
		linalg::lu::full_pivoting::solve::solve_in_place(C.rb(), C.rb(), row_perm, col_perm, B.rb_mut(), Par::Seq, stack);
	} else {
		// w is complex

		let (mut C, stack) = unsafe { linalg::temp_mat_uninit::<Complex<T>, _, _>(2, 2, stack) };
		let mut C = C.as_mat_mut();
		let mut row_perm_fwd = [0usize; 2];
		let mut row_perm_inv = [0usize; 2];
		let mut col_perm_fwd = [0usize; 2];
		let mut col_perm_inv = [0usize; 2];

		C[(0, 0)] = Complex {
			re: ca * A[(0, 0)] - wr * d0,
			im: -wi * d0,
		};

		C[(1, 0)] = Complex {
			re: ca * A[(1, 0)],
			im: zero(),
		};

		C[(0, 1)] = Complex {
			re: ca * A[(0, 1)],
			im: zero(),
		};

		C[(1, 1)] = Complex {
			re: ca * A[(1, 1)] - wr * d1,
			im: -wi * d1,
		};

		let cmax = C.norm_max();
		if cmax < smin {
			// use smin * I
			let smin_inv = recip(smin);
			zip!(B.rb_mut()).for_each(|unzip!(x)| *x = *x * smin_inv);
		}

		let (_, row_perm, col_perm) = linalg::lu::full_pivoting::factor::lu_in_place(
			C.rb_mut(),
			&mut row_perm_fwd,
			&mut row_perm_inv,
			&mut col_perm_fwd,
			&mut col_perm_inv,
			Par::Seq,
			stack,
			Default::default(),
		);

		let n = B.nrows();
		let (Br, Bi) = B.two_cols_mut(0, 1);
		let (mut B, stack) = unsafe { linalg::temp_mat_uninit::<Complex<T>, _, _>(n, 1, stack) };
		let mut B = B.as_mat_mut().col_mut(0);
		zip!(&mut B, &Br, &Bi).for_each(|unzip!(z, re, im)| {
			*z = Complex {
				re: copy(*re),
				im: copy(*im),
			}
		});
		linalg::lu::full_pivoting::solve::solve_in_place(C.rb(), C.rb(), row_perm, col_perm, B.rb_mut().as_mat_mut(), Par::Seq, stack);

		zip!(Br, Bi, &B).for_each(|unzip!(re, im, z)| (*re, *im) = (copy(z.re), copy(z.im)));
	}
}

#[math]
fn qz_to_gevd_real<T: RealField>(
	A: MatRef<'_, T>,
	B: MatRef<'_, T>,
	Q: Option<MatMut<'_, T>>,
	Z: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
) {
	let n = A.nrows();
	if n == 0 {
		return;
	}

	let one = one::<T>;
	let zero = zero::<T>;

	let ulp = eps::<T>();
	let safmin = min_positive::<T>();
	let smallnum = safmin * from_f64(n as f64);
	let small = smallnum * recip(ulp);
	let bignum = recip(smallnum);
	let big = recip(small);

	let (mut acolnorm, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
	let acolnorm = acolnorm.as_mat_mut();
	let (mut bcolnorm, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
	let bcolnorm = bcolnorm.as_mat_mut();

	let mut acolnorm = acolnorm.col_mut(0);
	let mut bcolnorm = bcolnorm.col_mut(0);
	let mut anorm = zero();
	let mut bnorm = zero();

	let mut j = 0;
	while j < n {
		let cplx = j + 1 < n && A[(j + 1, j)] != zero();
		if !cplx {
			let a = A.rb().col(j).get(..j).norm_l1();
			acolnorm[j] = copy(a);
			anorm = max(anorm, a + abs(A[(j, j)]));

			j += 1;
		} else {
			for jc in j..j + 2 {
				let a = A.rb().col(jc).get(..j).norm_l1();
				acolnorm[jc] = copy(a);
				anorm = max(anorm, a + abs(A[(j, jc)]) + abs(A[(j + 1, jc)]));
			}
			j += 2
		}
	}
	for j in 0..n {
		let b = B.rb().col(j).get(..j).norm_l1();
		bcolnorm[j] = copy(b);
		bnorm = max(bnorm, b + abs(B[(j, j)]));
	}
	let ascale = recip(max(anorm, safmin));
	let bscale = recip(max(bnorm, safmin));

	// left eigenvectors
	if let Some(mut u) = Q {
		let mut je = 0usize;
		while je < n {
			let cplx_eigval = je + 1 < n && A[(je + 1, je)] != zero();
			let nw = if cplx_eigval { 2 } else { 1 };

			if !cplx_eigval && max(abs(A[(je, je)]), abs(B[(je, je)])) < safmin {
				u.rb_mut().col_mut(je).fill(zero());
				u[(je, je)] = one();

				je += 1;
				continue;
			}

			let mut acoef;
			let mut acoefa;
			let mut bcoefa;
			let mut bcoefr;
			let mut bcoefi;
			let mut xmax;

			let (mut rhs, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
			let mut rhs = rhs.as_mat_mut();

			if !cplx_eigval {
				// real eigenvalue
				let temp = max(max(abs(A[(je, je)]) * ascale, abs(B[(je, je)]) * bscale), safmin);
				let salfar = (temp * A[(je, je)]) * ascale;
				let sbeta = (temp * B[(je, je)]) * bscale;

				acoef = sbeta * ascale;
				bcoefr = salfar * bscale;
				bcoefi = zero();

				// scale to avoid underflow
				let mut scale = one();
				let lsa = abs(sbeta) >= safmin && abs(acoef) < small;
				let lsb = abs(salfar) >= safmin && abs(bcoefr) < small;

				if lsa {
					scale = (small / abs(sbeta)) * min(anorm, big);
				}
				if lsb {
					scale = max(scale, (small / abs(salfar)) * min(bnorm, big));
				}
				if lsa || lsb {
					scale = min(scale, one() / (safmin * max(one(), max(abs(acoef), abs(bcoefr)))));
					if lsa {
						acoef = ascale * (scale * sbeta)
					} else {
						acoef = scale * acoef
					}
					if lsb {
						bcoefr = bscale * (scale * salfar)
					} else {
						bcoefr = scale * bcoefr
					}
				}
				acoefa = abs(acoef);
				bcoefa = abs(bcoefr);
				rhs[(je, 0)] = one();
				xmax = one();
			} else {
				// complex eigenvalue
				let (scale, _, wr, _, wi) = qz_real::generalized_eigval_2x2(
					(copy(A[(je, je)]), copy(A[(je, je + 1)]), copy(A[(je + 1, je)]), copy(A[(je + 1, je + 1)])),
					(copy(B[(je, je)]), copy(B[(je, je + 1)]), copy(B[(je + 1, je)]), copy(B[(je + 1, je + 1)])),
				);

				acoef = scale;
				bcoefr = wr;
				bcoefi = -wi;

				// scale to avoid over/underflow
				acoefa = abs(acoef);
				bcoefa = abs(bcoefr) + abs(bcoefi);
				let mut scale = one();
				if acoefa * ulp < safmin && acoefa >= safmin {
					scale = (safmin / ulp) / acoefa
				}
				if bcoefa * ulp < safmin && bcoefa >= safmin {
					scale = max(scale, (safmin / ulp) / bcoefa)
				}
				if safmin * acoefa > ascale {
					scale = ascale / (safmin * acoefa)
				}
				if safmin * bcoefa > bscale {
					scale = min(scale, bscale / (safmin * bcoefa))
				}
				if scale != one() {
					acoef = scale * acoef;
					acoefa = abs(acoef);
					bcoefr = scale * bcoefr;
					bcoefi = scale * bcoefi;
					bcoefa = abs(bcoefr) + abs(bcoefi);
				}

				// compute first two components of eigenvector

				let temp = acoef * A[(je + 1, je)];
				let temp2r = acoef * A[(je, je)] - bcoefr * B[(je, je)];
				let temp2i = -bcoefi * B[(je, je)];
				if abs(temp) > abs(temp2r) + abs(temp2i) {
					rhs[(je, 0)] = one();
					rhs[(je, 1)] = zero();
					rhs[(je + 1, 0)] = -temp2r / temp;
					rhs[(je + 1, 1)] = -temp2i / temp;
				} else {
					rhs[(je + 1, 0)] = one();
					rhs[(je + 1, 1)] = zero();
					let temp = acoef * A[(je, je + 1)];
					rhs[(je, 0)] = (bcoefr * B[(je + 1, je + 1)] - acoef * A[(je + 1, je + 1)]) / temp;
					rhs[(je, 1)] = bcoefi * B[(je + 1, je + 1)] / temp;
				}
				xmax = max(abs(rhs[(je, 0)]) + abs(rhs[(je, 1)]), abs(rhs[(je + 1, 0)]) + abs(rhs[(je + 1, 1)]))
			}

			let dmin = max(max(ulp * acoefa * anorm, ulp * bcoefa * bnorm), safmin);
			let mut j = je + nw;
			while j < n {
				let cplx = j + 1 < n && A[(j + 1, j)] != zero();
				let na = if cplx { 2 } else { 1 };

				let xscale = recip(xmax);

				let mut temp = max(max(acolnorm[j], bcolnorm[j]), acoefa * acolnorm[j] + bcoefa * bcolnorm[j]);

				let b0 = copy(B[(j, j)]);
				let b1;

				if cplx {
					temp = max(
						max(temp, acoefa * acolnorm[j + 1] + bcoefa * bcolnorm[j + 1]),
						max(acolnorm[j + 1], bcolnorm[j + 1]),
					);
					b1 = copy(B[(j + 1, j + 1)]);
				} else {
					b1 = zero();
				}
				if temp > bignum * xscale {
					for jw in 0..nw {
						for jr in je..j {
							rhs[(jr, jw)] = xscale * rhs[(jr, jw)];
						}
					}
					xmax = xmax * xscale;
				}

				// Compute dot products
				//
				//       j-1
				// SUM = sum  conjg( a*S(k,j) - b*P(k,j) )*x(k)
				//       k=je
				//
				// To reduce the op count, this is done as
				//
				// _        j-1                  _        j-1
				// a*conjg( sum  S(k,j)*x(k) ) - b*conjg( sum  P(k,j)*x(k) )
				//          k=je                          k=je
				//
				// which may cause underflow problems if A or B are close
				// to underflow.  (T.g., less than SMALL.)

				let (mut sums, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(na, nw, stack) };
				let mut sums = sums.as_mat_mut();
				let (mut sump, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(na, nw, stack) };
				let mut sump = sump.as_mat_mut();
				for jw in 0..nw {
					for ja in 0..na {
						sums[(ja, jw)] = zero();
						sump[(ja, jw)] = zero();

						for jr in je..j {
							sums[(ja, jw)] = sums[(ja, jw)] + A[(jr, j + ja)] * rhs[(jr, jw)];
							sump[(ja, jw)] = sump[(ja, jw)] + B[(jr, j + ja)] * rhs[(jr, jw)];
						}
					}
				}

				for ja in 0..na {
					if cplx_eigval {
						rhs[(j + ja, 0)] = -acoef * sums[(ja, 0)] + bcoefr * sump[(ja, 0)] - bcoefi * sump[(ja, 1)];
						rhs[(j + ja, 1)] = -acoef * sums[(ja, 1)] + bcoefr * sump[(ja, 1)] + bcoefi * sump[(ja, 0)];
					} else {
						rhs[(j + ja, 0)] = -acoef * sums[(ja, 0)] + bcoefr * sump[(ja, 0)];
					}
				}
				// Solve  ( a A - b B ).T  y = SUM(,)
				// with scaling and perturbation of the denominator
				if cplx {
					solve_complex_shifted_2x2(
						copy(dmin),
						copy(acoef),
						A.submatrix(j, j, na, na).transpose(),
						b0,
						b1,
						rhs.rb_mut().subrows_mut(j, na),
						copy(bcoefr),
						copy(bcoefi),
						stack,
					);
				} else {
					solve_complex_shifted_1x1(
						copy(dmin),
						copy(acoef),
						A.submatrix(j, j, na, na).transpose(),
						b0,
						rhs.rb_mut().subrows_mut(j, na),
						copy(bcoefr),
						copy(bcoefi),
					);
				}

				j += na;
			}

			let (mut tmp, _) = linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
			let mut tmp = tmp.as_mat_mut();
			linalg::matmul::matmul(tmp.rb_mut(), Accum::Replace, u.rb().get(.., je..), rhs.rb().get(je.., ..), one(), par);

			let mut u = u.rb_mut().subcols_mut(je, nw);
			u.copy_from(&tmp);
			let scale = recip(u.norm_l2());
			zip!(u).for_each(|unzip!(u)| {
				*u = *u * scale;
			});

			je += nw;
		}
	}
	// right eigenvectors
	if let Some(mut u) = Z {
		let mut je = n;
		while je > 0 {
			je -= 1;
			let cplx_eigval = je > 0 && A[(je, je - 1)] != zero();
			let nw = if cplx_eigval { 2 } else { 1 };
			je -= nw - 1;

			if !cplx_eigval && max(abs(A[(je, je)]), abs(B[(je, je)])) < safmin {
				u.rb_mut().col_mut(je).fill(zero());
				u[(je, je)] = one();

				continue;
			}

			let mut acoef;
			let mut acoefa;
			let mut bcoefa;
			let mut bcoefr;
			let mut bcoefi;

			let (mut rhs, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
			let mut rhs = rhs.as_mat_mut();

			if !cplx_eigval {
				// real eigenvalue
				let temp = max(max(abs(A[(je, je)]) * ascale, abs(B[(je, je)]) * bscale), safmin);
				let salfar = (temp * A[(je, je)]) * ascale;
				let sbeta = (temp * B[(je, je)]) * bscale;

				acoef = sbeta * ascale;
				bcoefr = salfar * bscale;
				bcoefi = zero();

				// scale to avoid underflow
				let mut scale = one();
				let lsa = abs(sbeta) >= safmin && abs(acoef) < small;
				let lsb = abs(salfar) >= safmin && abs(bcoefr) < small;

				if lsa {
					scale = (small / abs(sbeta)) * min(anorm, big);
				}
				if lsb {
					scale = max(scale, (small / abs(salfar)) * min(bnorm, big));
				}
				if lsa || lsb {
					scale = min(scale, one() / (safmin * max(one(), max(abs(acoef), abs(bcoefr)))));
					if lsa {
						acoef = ascale * (scale * sbeta)
					} else {
						acoef = scale * acoef
					}
					if lsb {
						bcoefr = bscale * (scale * salfar)
					} else {
						bcoefr = scale * bcoefr
					}
				}
				acoefa = abs(acoef);
				bcoefa = abs(bcoefr);
				rhs[(je, 0)] = one();

				for jr in 0..je {
					rhs[(jr, 0)] = bcoefr * B[(jr, je)] - acoef * A[(jr, je)];
				}
			} else {
				// complex eigenvalue
				let (scale, _, wr, _, wi) = qz_real::generalized_eigval_2x2(
					(copy(A[(je, je)]), copy(A[(je, je + 1)]), copy(A[(je + 1, je)]), copy(A[(je + 1, je + 1)])),
					(copy(B[(je, je)]), copy(B[(je, je + 1)]), copy(B[(je + 1, je)]), copy(B[(je + 1, je + 1)])),
				);

				acoef = scale;
				bcoefr = wr;
				bcoefi = -wi;

				// scale to avoid over/underflow
				acoefa = abs(acoef);
				bcoefa = abs(bcoefr) + abs(bcoefi);
				let mut scale = one();
				if acoefa * ulp < safmin && acoefa >= safmin {
					scale = (safmin / ulp) / acoefa
				}
				if bcoefa * ulp < safmin && bcoefa >= safmin {
					scale = max(scale, (safmin / ulp) / bcoefa)
				}
				if safmin * acoefa > ascale {
					scale = ascale / (safmin * acoefa)
				}
				if safmin * bcoefa > bscale {
					scale = min(scale, bscale / (safmin * bcoefa))
				}
				if scale != one() {
					acoef = scale * acoef;
					acoefa = abs(acoef);
					bcoefr = scale * bcoefr;
					bcoefi = scale * bcoefi;
					bcoefa = abs(bcoefr) + abs(bcoefi);
				}

				// compute first two components of eigenvector

				let temp = acoef * A[(je + 1, je)];
				let temp2r = acoef * A[(je + 1, je + 1)] - bcoefr * B[(je + 1, je + 1)];
				let temp2i = -bcoefi * B[(je + 1, je + 1)];
				if abs(temp) > abs(temp2r) + abs(temp2i) {
					rhs[(je + 1, 0)] = one();
					rhs[(je + 1, 1)] = zero();
					rhs[(je, 0)] = -temp2r / temp;
					rhs[(je, 1)] = -temp2i / temp;
				} else {
					rhs[(je, 0)] = one();
					rhs[(je, 1)] = zero();
					let temp = acoef * A[(je, je + 1)];
					rhs[(je + 1, 0)] = (bcoefr * B[(je, je)] - acoef * A[(je, je)]) / temp;
					rhs[(je + 1, 1)] = bcoefi * B[(je, je)] / temp;
				}

				let creala = acoef * rhs[(je, 0)];
				let cimaga = acoef * rhs[(je, 1)];
				let crealb = bcoefr * rhs[(je, 0)] - bcoefi * rhs[(je, 1)];
				let cimagb = bcoefi * rhs[(je, 0)] + bcoefr * rhs[(je, 1)];
				let cre2a = acoef * rhs[(je + 1, 0)];
				let cim2a = acoef * rhs[(je + 1, 1)];
				let cre2b = bcoefr * rhs[(je + 1, 0)] - bcoefi * rhs[(je + 1, 1)];
				let cim2b = bcoefi * rhs[(je + 1, 0)] + bcoefr * rhs[(je + 1, 1)];
				for jr in 0..je {
					rhs[(jr, 0)] = -creala * A[(jr, je)] + crealb * B[(jr, je)] - cre2a * A[(jr, je + 1)] + cre2b * B[(jr, je + 1)];
					rhs[(jr, 1)] = -cimaga * A[(jr, je)] + cimagb * B[(jr, je)] - cim2a * A[(jr, je + 1)] + cim2b * B[(jr, je + 1)];
				}
			}
			let dmin = max(max(ulp * acoefa * anorm, ulp * bcoefa * bnorm), safmin);
			let mut j = je;
			while j > 0 {
				j -= 1;
				let cplx = j > 0 && A[(j, j - 1)] != zero();
				let na = if cplx { 2 } else { 1 };
				j -= na - 1;

				let b0 = copy(B[(j, j)]);
				let b1;

				if cplx {
					b1 = copy(B[(j + 1, j + 1)]);
				} else {
					b1 = zero();
				}

				if cplx {
					solve_complex_shifted_2x2(
						copy(dmin),
						copy(acoef),
						A.submatrix(j, j, na, na),
						b0,
						b1,
						rhs.rb_mut().subrows_mut(j, na),
						copy(bcoefr),
						copy(bcoefi),
						stack,
					);
				} else {
					solve_complex_shifted_1x1(
						copy(dmin),
						copy(acoef),
						A.submatrix(j, j, na, na),
						b0,
						rhs.rb_mut().subrows_mut(j, na),
						copy(bcoefr),
						copy(bcoefi),
					);
				}

				// Compute the contributions of the off-diagonals of
				// column j (and j+1, if 2-by-2 block) of A and B to the
				// sums.

				for ja in 0..na {
					if cplx_eigval {
						let creala = acoef * rhs[(j + ja, 0)];
						let cimaga = acoef * rhs[(j + ja, 1)];
						let crealb = bcoefr * rhs[(j + ja, 0)] - bcoefi * rhs[(j + ja, 1)];
						let cimagb = bcoefi * rhs[(j + ja, 0)] + bcoefr * rhs[(j + ja, 1)];
						for jr in 0..j {
							rhs[(jr, 0)] = rhs[(jr, 0)] - creala * A[(jr, j + ja)] + crealb * B[(jr, j + ja)];
							rhs[(jr, 1)] = rhs[(jr, 1)] - cimaga * A[(jr, j + ja)] + cimagb * B[(jr, j + ja)];
						}
					} else {
						let creala = acoef * rhs[(j + ja, 0)];
						let crealb = bcoefr * rhs[(j + ja, 0)];

						for jr in 0..j {
							rhs[(jr, 0)] = rhs[(jr, 0)] - creala * A[(jr, j + ja)] + crealb * B[(jr, j + ja)];
						}
					}
				}
			}
			let (mut tmp, _) = linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
			let mut tmp = tmp.as_mat_mut();
			linalg::matmul::matmul(
				tmp.rb_mut(),
				Accum::Replace,
				u.rb().get(.., ..je + nw),
				rhs.rb().get(..je + nw, ..),
				one(),
				par,
			);

			let mut u = u.rb_mut().subcols_mut(je, nw);
			u.copy_from(&tmp);
			let scale = recip(u.norm_l2());
			zip!(u).for_each(|unzip!(u)| {
				*u = *u * scale;
			});
		}
	}
}

#[math]
fn qz_to_gevd_cplx<T: ComplexField>(
	A: MatRef<'_, T>,
	B: MatRef<'_, T>,
	Q: Option<MatMut<'_, T>>,
	Z: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
) {
	let n = A.nrows();
	if n == 0 {
		return;
	}

	let one = one::<T::Real>;

	let ulp = eps::<T::Real>();
	let safmin = min_positive::<T::Real>();
	let smallnum = safmin * from_f64(n as f64);
	let small = smallnum * recip(ulp);
	let bignum = recip(smallnum);
	let big = recip(small);

	let (mut acolnorm, stack) = linalg::temp_mat_zeroed::<T::Real, _, _>(n, 1, stack);
	let acolnorm = acolnorm.as_mat_mut();
	let (mut bcolnorm, stack) = linalg::temp_mat_zeroed::<T::Real, _, _>(n, 1, stack);
	let bcolnorm = bcolnorm.as_mat_mut();

	let mut acolnorm = acolnorm.col_mut(0);
	let mut bcolnorm = bcolnorm.col_mut(0);
	let mut anorm = zero::<T::Real>();
	let mut bnorm = zero::<T::Real>();

	let mut j = 0;
	while j < n {
		let a = A.rb().col(j).get(..j).norm_l1();
		acolnorm[j] = copy(a);
		anorm = max(anorm, a + abs(A[(j, j)]));

		j += 1;
	}
	for j in 0..n {
		let b = B.rb().col(j).get(..j).norm_l1();
		bcolnorm[j] = copy(b);
		bnorm = max(bnorm, b + abs(B[(j, j)]));
	}
	let ascale = recip(max(anorm, safmin));
	let bscale = recip(max(bnorm, safmin));

	// left eigenvectors
	if let Some(mut u) = Q {
		let mut je = 0usize;
		while je < n {
			if max(abs(A[(je, je)]), abs(B[(je, je)])) < safmin {
				u.rb_mut().col_mut(je).fill(zero());
				u[(je, je)] = from_real(one());

				je += 1;
				continue;
			}

			let mut acoef;
			let acoefa;
			let bcoefa;
			let mut bcoef;
			let mut xmax;

			let (mut rhs, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
			let mut rhs = rhs.as_mat_mut().col_mut(0);

			{
				// real eigenvalue
				let temp = max(max(abs(A[(je, je)]) * ascale, abs(B[(je, je)]) * bscale), safmin);
				let salfar = mul_real(mul_real(A[(je, je)], temp), ascale);
				let sbeta = real(B[(je, je)]) * temp * bscale;

				acoef = sbeta * ascale;
				bcoef = mul_real(salfar, bscale);

				// scale to avoid underflow
				let mut scale = one();
				let lsa = abs(sbeta) >= safmin && abs(acoef) < small;
				let lsb = abs(salfar) >= safmin && abs(bcoef) < small;

				if lsa {
					scale = (small / abs(sbeta)) * min(anorm, big);
				}
				if lsb {
					scale = max(scale, (small / abs(salfar)) * min(bnorm, big));
				}
				if lsa || lsb {
					scale = min(scale, one() / (safmin * max(one(), max(abs(acoef), abs(bcoef)))));
					if lsa {
						acoef = sbeta * scale * ascale
					} else {
						acoef = acoef * scale
					}
					if lsb {
						bcoef = mul_real(mul_real(salfar, scale), bscale)
					} else {
						bcoef = mul_real(bcoef, scale)
					}
				}
				acoefa = abs(acoef);
				bcoefa = abs(bcoef);
				rhs[je] = from_real(one());
				xmax = one();
			}

			let dmin = max(max(ulp * acoefa * anorm, ulp * bcoefa * bnorm), safmin);
			let mut j = je + 1;
			while j < n {
				let xscale = recip(xmax);

				let temp = max(max(acolnorm[j], bcolnorm[j]), acoefa * acolnorm[j] + bcoefa * bcolnorm[j]);

				let b0 = copy(B[(j, j)]);

				if temp > bignum * xscale {
					for jr in je..j {
						rhs[jr] = mul_real(rhs[jr], xscale);
					}
					xmax = xmax * xscale;
				}

				// Compute dot products
				//
				//       j-1
				// SUM = sum  conjg( a*S(k,j) - b*P(k,j) )*x(k)
				//       k=je
				//
				// To reduce the op count, this is done as
				//
				// _        j-1                  _        j-1
				// a*conjg( sum  S(k,j)*x(k) ) - b*conjg( sum  P(k,j)*x(k) )
				//          k=je                          k=je
				//
				// which may cause underflow problems if A or B are close
				// to underflow.  (T.g., less than SMALL.)

				let mut sums = zero::<T>();
				let mut sump = zero::<T>();

				for jr in je..j {
					sums = sums + conj(A[(jr, j)]) * rhs[jr];
					sump = sump + conj(B[(jr, j)]) * rhs[jr];
				}

				rhs[j] = conj(bcoef) * sump - mul_real(sums, acoef);
				// Solve  ( a A - b B ).T  y = SUM(,)
				// with scaling and perturbation of the denominator

				solve_shifted_1x1(copy(dmin), copy(acoef), conj(A[(j, j)]), conj(b0), &mut rhs[j], conj(bcoef));

				j += 1;
			}

			let (mut tmp, _) = linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
			let mut tmp = tmp.as_mat_mut().col_mut(0);
			linalg::matmul::matmul(
				tmp.rb_mut(),
				Accum::Replace,
				u.rb().get(.., je..),
				rhs.rb().get(je..),
				from_real(one()),
				par,
			);

			let mut u = u.rb_mut().col_mut(je);
			u.copy_from(&tmp);
			let scale = recip(u.norm_l2());
			zip!(u).for_each(|unzip!(u)| {
				*u = mul_real(*u, scale);
			});

			je += 1;
		}
	}
	// right eigenvectors
	if let Some(mut u) = Z {
		let mut je = n;
		while je > 0 {
			je -= 1;

			if max(abs(A[(je, je)]), abs(B[(je, je)])) < safmin {
				u.rb_mut().col_mut(je).fill(zero());
				u[(je, je)] = from_real(one());

				continue;
			}

			let mut acoef;
			let acoefa;
			let bcoefa;
			let mut bcoefr;

			let (mut rhs, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
			let mut rhs = rhs.as_mat_mut().col_mut(0);

			{
				// real eigenvalue
				let temp = max(max(abs(A[(je, je)]) * ascale, abs(B[(je, je)]) * bscale), safmin);
				let salfar = mul_real(mul_real(A[(je, je)], temp), ascale);
				let sbeta = real(B[(je, je)]) * temp * bscale;

				acoef = sbeta * ascale;
				bcoefr = mul_real(salfar, bscale);

				// scale to avoid underflow
				let mut scale = one();
				let lsa = abs(sbeta) >= safmin && abs(acoef) < small;
				let lsb = abs(salfar) >= safmin && abs(bcoefr) < small;

				if lsa {
					scale = (small / abs(sbeta)) * min(anorm, big);
				}
				if lsb {
					scale = max(scale, (small / abs(salfar)) * min(bnorm, big));
				}
				if lsa || lsb {
					scale = min(scale, one() / (safmin * max(one(), max(abs(acoef), abs(bcoefr)))));
					if lsa {
						acoef = sbeta * scale * ascale
					} else {
						acoef = acoef * scale
					}
					if lsb {
						bcoefr = mul_real(mul_real(salfar, scale), bscale)
					} else {
						bcoefr = mul_real(bcoefr, scale)
					}
				}
				acoefa = abs(acoef);
				bcoefa = abs(bcoefr);
				rhs[je] = from_real(one());

				for jr in 0..je {
					rhs[jr] = bcoefr * B[(jr, je)] - mul_real(A[(jr, je)], acoef);
				}
			}
			let dmin = max(max(ulp * acoefa * anorm, ulp * bcoefa * bnorm), safmin);
			let mut j = je;
			while j > 0 {
				j -= 1;

				let b0 = copy(B[(j, j)]);

				solve_shifted_1x1(copy(dmin), copy(acoef), copy(A[(j, j)]), b0, &mut rhs[j], copy(bcoefr));

				// Compute the contributions of the off-diagonals of
				// column j (and j+1, if 2-by-2 block) of A and B to the
				// sums.

				let creala = mul_real(rhs[j], acoef);
				let crealb = bcoefr * rhs[j];

				for jr in 0..j {
					rhs[jr] = rhs[jr] - creala * A[(jr, j)] + crealb * B[(jr, j)];
				}
			}
			let (mut tmp, _) = linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
			let mut tmp = tmp.as_mat_mut().col_mut(0);
			linalg::matmul::matmul(
				tmp.rb_mut(),
				Accum::Replace,
				u.rb().get(.., ..je + 1),
				rhs.rb().get(..je + 1),
				from_real(one()),
				par,
			);

			let mut u = u.rb_mut().col_mut(je);
			u.copy_from(&tmp);
			let scale = recip(u.norm_l2());
			zip!(u).for_each(|unzip!(u)| {
				*u = mul_real(*u, scale);
			});
		}
	}
}

/// computes the layout of the workspace required to compute a matrix pair's
/// generalized eigendecomposition
pub fn gevd_scratch<T: ComplexField>(dim: usize, left: ComputeEigenvectors, right: ComputeEigenvectors, par: Par, params: GevdParams) -> StackReq {
	let _ = (left, right);

	let n = dim;
	let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n, n);

	StackReq::any_of(&[
		linalg::temp_mat_scratch::<T>(blocksize, n).and(
			linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(n, n, blocksize, par, default())
				.or(linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(n, blocksize, n)),
		),
		gen_hessenberg::generalized_hessenberg_scratch::<T>(n, auto!(T)),
		if const { T::IS_REAL } {
			qz_real::hessenberg_to_qz_scratch::<T::Real>(n, par, params.schur)
		} else {
			qz_cplx::hessenberg_to_qz_scratch::<T>(n, par, params.schur)
		},
	])
}

/// computes the real matrix pair $(A, B)$'s eigendecomposition
///
/// the eigenvalues are stored in $S$, the left eigenvectors in $U_L$, and the right eigenvectors in
/// $U_R$
#[track_caller]
pub fn gevd_real<T: RealField>(
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	S_re: ColMut<'_, T>,
	S_im: ColMut<'_, T>,
	beta: ColMut<'_, T>,
	U_left: Option<MatMut<'_, T>>,
	U_right: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: GevdParams,
) {
	compute_gevd_generic(
		A,
		B,
		S_re,
		S_im,
		beta,
		U_left,
		U_right,
		par,
		stack,
		params,
		qz_real::hessenberg_to_qz,
		qz_to_gevd_real,
	)
}

/// computes the complex matrix pair $(A, B)$'s eigendecomposition
///
/// the eigenvalues are stored in $S$, the left eigenvectors in $U_L$, and the right eigenvectors in
/// $U_R$
#[track_caller]
pub fn gevd_cplx<T: ComplexField>(
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	S: ColMut<'_, T>,
	beta: ColMut<'_, T>,
	U_left: Option<MatMut<'_, T>>,
	U_right: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: GevdParams,
) {
	compute_gevd_generic(
		A,
		B,
		S,
		ColMut::from_slice_mut(&mut []),
		beta,
		U_left,
		U_right,
		par,
		stack,
		params,
		|A: MatMut<'_, T>,
		 B: MatMut<'_, T>,
		 Q: Option<MatMut<'_, T>>,
		 Z: Option<MatMut<'_, T>>,
		 alphar: ColMut<'_, T>,
		 _: ColMut<'_, T>,
		 beta: ColMut<'_, T>,
		 eigenvectors: ComputeEigenvectors,
		 par: Par,
		 params: GeneralizedSchurParams,
		 stack: &mut MemStack| qz_cplx::hessenberg_to_qz(A, B, Q, Z, alphar, beta, eigenvectors, par, params, stack),
		qz_to_gevd_cplx,
	)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils;
	use dyn_stack::MemBuffer;
	use equator::assert;

	#[test]
	fn test_lishen_() {
		let approx_eq = utils::approx::CwiseMat(utils::approx::ApproxEq::<f64>::eps() * 128.0);

		let A = &[
			[
				0.0000000067116330731078674,
				-0.0000000024189842394753674,
				0.00000000032505193899740187,
				0.0,
				0.0,
				0.0,
			],
			[
				-0.0000000024189842394753674,
				0.000000011866543705551487,
				-0.00000001582083742901503,
				0.0,
				0.0,
				0.0,
			],
			[
				0.00000000032505193899740187,
				-0.00000001582083742901503,
				0.00000002523087227632826,
				0.0,
				0.0,
				0.0,
			],
			[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0, 1.0f64],
		];

		let B = &[
			[
				0.00016038376001067668,
				-0.00003674235041182116,
				-0.0000029397921270058582,
				-1.0,
				-0.0,
				-0.0,
			],
			[-0.00003674235041182116, 0.0002038454538304322, -0.00021213416901900504, -0.0, -1.0, -0.0],
			[
				-0.0000029397921270058582,
				-0.00021213416901900504,
				0.0003954706173886255,
				-0.0,
				-0.0,
				-1.0,
			],
			[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0, 0.0, 0.0f64],
		];

		let A = MatRef::from_row_major_array(A);
		let B = MatRef::from_row_major_array(B);

		let n = A.nrows();

		{
			let mut H = A.to_owned();
			let mut T = B.to_owned();
			let mut alpha_re = Col::<f64>::zeros(n);
			let mut alpha_im = Col::<f64>::zeros(n);
			let mut beta = Col::<f64>::zeros(n);

			let mut UL = Mat::<f64>::identity(n, n);
			let mut UR = Mat::<f64>::identity(n, n);

			gevd_real(
				H.as_mut(),
				T.as_mut(),
				alpha_re.rb_mut(),
				alpha_im.rb_mut(),
				beta.rb_mut(),
				Some(UL.as_mut()),
				Some(UR.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(gevd_scratch::<f64>(
					n,
					ComputeEigenvectors::Yes,
					ComputeEigenvectors::Yes,
					Par::Seq,
					auto!(f64),
				))),
				auto!(f64),
			);

			{
				let mut i = 0;
				while i < n {
					if alpha_im[i] == 0.0 {
						let u = UR.col(i);
						let a = alpha_re[i];
						let b = beta[i];

						assert!((b / a) * &A * u ~ B * u);

						i += 1;
					} else {
						let ur = UR.col(i);
						let ui = UR.col(i + 1);
						let ar = alpha_re[i];
						let ai = alpha_im[i];
						let b = beta[i];

						assert!((b / (ar.abs() + ai.abs())) * &A * ur ~ B * (ar * ur + ai * ui) / (ar.abs() + ai.abs()));
						assert!((b / (ar.abs() + ai.abs())) * &A * ui ~ B * (ar * ui - ai * ur) / (ar.abs() + ai.abs()));

						i += 2;
					}
				}
			}

			{
				let mut i = 0;
				while i < n {
					if alpha_im[i] == 0.0 {
						let u = UL.col(i);
						let a = alpha_re[i];
						let b = beta[i];

						assert!(b / a * u.adjoint() * &A ~ u.adjoint() * B);

						i += 1;
					} else {
						let ur = UL.col(i);
						let ui = UL.col(i + 1);
						let ar = alpha_re[i];
						let ai = alpha_im[i];
						let b = beta[i];

						assert!((b / (ar.abs() + ai.abs())) * ur.adjoint() * &A ~ (ar * ur + ai * ui).adjoint() / (ar.abs() + ai.abs()) * B);
						assert!((b / (ar.abs() + ai.abs())) * ui.adjoint() * &A ~ (ar * ui - ai * ur).adjoint() / (ar.abs() + ai.abs()) * B);

						i += 2;
					}
				}
			}
		}
	}

	#[test]
	fn test_cplx() {
		let rng = &mut StdRng::seed_from_u64(1);
		let approx_eq = utils::approx::CwiseMat(utils::approx::ApproxEq::<f64>::eps() * 128.0);
		let n = 20;

		let A = &CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);

		let B = &CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);

		{
			let mut H = A.to_owned();
			let mut T = B.to_owned();
			let mut alpha_re = Col::<c64>::zeros(n);
			let mut beta = Col::<c64>::zeros(n);

			let mut UL = Mat::<c64>::identity(n, n);
			let mut UR = Mat::<c64>::identity(n, n);

			gevd_cplx(
				H.as_mut(),
				T.as_mut(),
				alpha_re.rb_mut(),
				beta.rb_mut(),
				Some(UL.as_mut()),
				Some(UR.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(gevd_scratch::<c64>(
					n,
					ComputeEigenvectors::Yes,
					ComputeEigenvectors::Yes,
					Par::Seq,
					auto!(c64),
				))),
				auto!(c64),
			);

			{
				let mut i = 0;
				while i < n {
					let u = UR.col(i);
					let a = alpha_re[i];
					let b = beta[i];

					assert!(Scale(b / a) * A * u ~ B * u);

					i += 1;
				}
			}

			{
				let mut i = 0;
				while i < n {
					let u = UL.col(i);
					let a = alpha_re[i];
					let b = beta[i];

					assert!(Scale(b / a) * u.adjoint() * A ~ u.adjoint() * B);

					i += 1;
				}
			}
		}
	}
}
