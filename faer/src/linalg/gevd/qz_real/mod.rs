use crate::internal_prelude::*;
use equator::assert;
use linalg::matmul::matmul;

use super::GeneralizedSchurParams;
use super::gen_hessenberg::{make_givens, rot};

#[math]
fn copy_sign<T: RealField>(a: T, b: T) -> T {
	if b >= zero() { a } else { -a }
}

#[inline]
fn rot_rows<T: RealField>(c: T, s: T, A: MatMut<'_, T>, i: usize, j: usize) {
	let (x, y) = A.two_rows_mut(i, j);
	rot(c, s, x, y)
}

#[inline]
fn rot_cols<T: RealField>(c: T, s: T, A: MatMut<'_, T>, i: usize, j: usize) {
	let (x, y) = A.two_cols_mut(i, j);
	rot(c, s, x.transpose_mut(), y.transpose_mut())
}

// solve A * R - L * B = C
// solve D * R - L * E = F
// (R, L) is stored in (C, F)
#[math]
fn solve_sylvester_single_block<T: RealField>(
	A: MatRef<'_, T>,
	B: MatRef<'_, T>,
	mut C: MatMut<'_, T>,
	D: MatRef<'_, T>,
	E: MatRef<'_, T>,
	mut F: MatMut<'_, T>,
	stack: &mut MemStack,
) {
	let n1 = A.nrows();
	let n2 = B.nrows();

	let nz = n1 * n2 * 2;

	let (mut z, stack) = linalg::temp_mat_zeroed::<T, _, _>(nz, nz, stack);
	let mut z = z.as_mat_mut();
	let (mut rhs, stack) = linalg::temp_mat_zeroed::<T, _, _>(nz, 1, stack);
	let rhs = rhs.as_mat_mut();
	let (row_perm_fwd, stack) = unsafe { stack.make_raw::<usize>(nz) };
	let (row_perm_inv, stack) = unsafe { stack.make_raw::<usize>(nz) };
	let (col_perm_fwd, stack) = unsafe { stack.make_raw::<usize>(nz) };
	let (col_perm_inv, stack) = unsafe { stack.make_raw::<usize>(nz) };
	let stack = stack;

	let mut rhs = rhs.col_mut(0);

	{
		if n1 == 1 && n2 == 1 {
			z[(1, 1)] = copy(A[(0, 0)]);
			z[(2, 1)] = copy(D[(0, 0)]);
			z[(1, 2)] = -B[(0, 0)];
			z[(2, 2)] = -E[(0, 0)];

			rhs[0] = copy(C[(0, 0)]);
			rhs[1] = copy(F[(0, 0)]);
		} else if n1 == 1 && n2 == 2 {
			z[(0, 0)] = copy(A[(0, 0)]);
			z[(1, 0)] = zero();
			z[(2, 0)] = copy(D[(0, 0)]);
			z[(3, 0)] = zero();

			z[(0, 1)] = zero();
			z[(1, 1)] = copy(A[(0, 0)]);
			z[(2, 1)] = zero();
			z[(3, 1)] = copy(D[(0, 0)]);

			z[(0, 2)] = -B[(0, 0)];
			z[(1, 2)] = -B[(0, 1)];
			z[(2, 2)] = -E[(0, 0)];
			z[(3, 2)] = -E[(0, 1)];

			z[(0, 3)] = -B[(1, 0)];
			z[(1, 3)] = -B[(1, 1)];
			z[(2, 3)] = zero();
			z[(3, 3)] = -E[(1, 1)];

			rhs[0] = copy(C[(0, 0)]);
			rhs[1] = copy(C[(0, 1)]);
			rhs[2] = copy(F[(0, 0)]);
			rhs[3] = copy(F[(0, 1)]);
		} else if n1 == 2 && n2 == 1 {
			z[(0, 0)] = copy(A[(0, 0)]);
			z[(1, 0)] = copy(A[(1, 0)]);
			z[(2, 0)] = copy(D[(0, 0)]);
			z[(3, 0)] = zero();

			z[(0, 1)] = copy(A[(0, 1)]);
			z[(1, 1)] = copy(A[(1, 1)]);
			z[(2, 1)] = copy(D[(0, 1)]);
			z[(3, 1)] = copy(D[(1, 1)]);

			z[(0, 2)] = -B[(0, 0)];
			z[(1, 2)] = zero();
			z[(2, 2)] = -E[(0, 0)];
			z[(3, 2)] = zero();

			z[(0, 3)] = zero();
			z[(1, 3)] = -B[(0, 0)];
			z[(2, 3)] = zero();
			z[(3, 3)] = -E[(0, 0)];

			rhs[0] = copy(C[(0, 0)]);
			rhs[1] = copy(C[(1, 0)]);
			rhs[2] = copy(F[(0, 0)]);
			rhs[3] = copy(F[(1, 0)]);
		} else if n1 == 2 && n2 == 2 {
			z[(0, 0)] = copy(A[(0, 0)]);
			z[(1, 0)] = copy(A[(1, 0)]);
			z[(4, 0)] = copy(D[(0, 0)]);

			z[(0, 1)] = copy(A[(0, 1)]);
			z[(1, 1)] = copy(A[(1, 1)]);
			z[(4, 1)] = copy(D[(0, 1)]);
			z[(5, 1)] = copy(D[(1, 1)]);

			z[(2, 2)] = copy(A[(0, 0)]);
			z[(3, 2)] = copy(A[(1, 0)]);
			z[(6, 2)] = copy(D[(0, 0)]);

			z[(2, 3)] = copy(A[(0, 1)]);
			z[(3, 3)] = copy(A[(1, 1)]);
			z[(6, 3)] = copy(D[(0, 1)]);
			z[(7, 3)] = copy(D[(1, 1)]);

			z[(0, 4)] = -B[(0, 0)];
			z[(2, 4)] = -B[(0, 1)];
			z[(4, 4)] = -E[(0, 0)];
			z[(6, 4)] = -E[(0, 1)];

			z[(1, 5)] = -B[(0, 0)];
			z[(3, 5)] = -B[(0, 1)];
			z[(5, 5)] = -E[(0, 0)];
			z[(7, 5)] = -E[(0, 1)];

			z[(0, 6)] = -B[(1, 0)];
			z[(2, 6)] = -B[(1, 1)];
			z[(6, 6)] = -E[(1, 1)];

			z[(1, 7)] = -B[(1, 0)];
			z[(3, 7)] = -B[(1, 1)];
			z[(7, 7)] = -E[(1, 1)];

			rhs[0] = copy(C[(0, 0)]);
			rhs[1] = copy(C[(1, 0)]);
			rhs[2] = copy(C[(0, 1)]);
			rhs[3] = copy(C[(1, 1)]);
			rhs[4] = copy(F[(0, 0)]);
			rhs[5] = copy(F[(1, 0)]);
			rhs[6] = copy(F[(0, 1)]);
			rhs[7] = copy(F[(1, 1)]);
		} else {
			unreachable!();
		}
	}

	let (_, row_perm, col_perm) = linalg::lu::full_pivoting::factor::lu_in_place(
		z.rb_mut(),
		row_perm_fwd,
		row_perm_inv,
		col_perm_fwd,
		col_perm_inv,
		Par::Seq,
		stack,
		Default::default(),
	);
	linalg::lu::full_pivoting::solve::solve_in_place(z.rb(), z.rb(), row_perm, col_perm, rhs.rb_mut().as_mat_mut(), Par::Seq, stack);

	if n1 == 1 && n2 == 1 {
		C[(0, 0)] = copy(rhs[0]);
		F[(0, 0)] = copy(rhs[1]);
	} else if n1 == 1 && n2 == 2 {
		C[(0, 0)] = copy(rhs[0]);
		C[(0, 1)] = copy(rhs[1]);
		F[(0, 0)] = copy(rhs[2]);
		F[(0, 1)] = copy(rhs[3]);
	} else if n1 == 2 && n2 == 1 {
		C[(0, 0)] = copy(rhs[0]);
		C[(1, 0)] = copy(rhs[1]);
		F[(0, 0)] = copy(rhs[2]);
		F[(1, 0)] = copy(rhs[3]);
	} else if n1 == 2 && n2 == 2 {
		C[(0, 0)] = copy(rhs[0]);
		C[(1, 0)] = copy(rhs[1]);
		C[(0, 1)] = copy(rhs[2]);
		C[(1, 1)] = copy(rhs[3]);
		F[(0, 0)] = copy(rhs[4]);
		F[(1, 0)] = copy(rhs[5]);
		F[(0, 1)] = copy(rhs[6]);
		F[(1, 1)] = copy(rhs[7]);
	} else {
		unreachable!();
	}
}

#[math]
fn svd_triu_2x2<T: RealField>(f: T, g: T, h: T) -> (T, T, T, T, T, T) {
	let half = from_f64::<T>(0.5);
	let two = from_f64::<T>(2.0);
	let four = from_f64::<T>(4.0);
	let eps = eps::<T>();

	let mut ft = copy(f);
	let mut fa = abs(ft);
	let mut ht = copy(h);
	let mut ha = abs(h);

	// * PMAX points to the maximum absolute element of matrix
	// * PMAX = 1 if F largest in absolute values
	// * PMAX = 2 if G largest in absolute values
	// * PMAX = 3 if H largest in absolute values
	let mut pmax = 1;
	let swap = ha > fa;
	if swap {
		pmax = 3;
		let temp = ft;
		ft = ht;
		ht = temp;
		let temp = fa;
		fa = ha;
		ha = temp;
		// *
		// * Now FA .ge. HA
		// *
	}
	let gt = copy(g);
	let ga = abs(gt);
	let mut ssmin = zero::<T>();
	let mut ssmax = zero::<T>();
	let mut clt = zero::<T>();
	let mut crt = zero::<T>();
	let mut slt = zero::<T>();
	let mut srt = zero::<T>();

	if ga == zero() {
		// * Diagonal matrix
		ssmin = ha;
		ssmax = fa;
		clt = one();
		crt = one();
		slt = zero();
		srt = zero();
	} else {
		let mut gasmal = true;
		if ga > fa {
			pmax = 2;
			if fa / ga < eps {
				// * Case of very large GA
				gasmal = false;
				ssmax = copy(ga);
				if ha > one() {
					ssmin = fa / (ga / ha);
				} else {
					ssmin = (fa / ga) * ha;
				}
				clt = one();
				slt = ht / gt;
				srt = one();
				crt = ft / gt;
			}
		}

		if gasmal {
			// * Normal case
			let d = fa - ha;
			let mut l;
			if d == fa {
				// *
				// * Copes with infinite F or H
				// *
				l = one();
			} else {
				l = d / fa;
			}
			// * Note that 0 .le. L .le. 1
			// *
			let m = gt / ft;
			// *
			// * Note that abs(M) .le. 1/macheps
			// *
			let mut t = two - l;
			// *
			// * Note that T .ge. 1
			// *
			let mm = m * m;
			let tt = t * t;
			let s = sqrt(tt + mm);
			// *
			// * Note that 1 .le. S .le. 1 + 1/macheps
			// *
			let r;
			if l == zero() {
				r = abs(m);
			} else {
				r = sqrt(l * l + mm);
			}
			// * Note that 0 .le. R .le. 1 + 1/macheps
			let a = half * (s + r);
			// * Note that 1 .le. A .le. 1 + abs(M)
			ssmin = ha / a;
			ssmax = fa * a;
			if mm == zero() {
				// * Note that M is very tiny
				if l == zero() {
					t = copy_sign(copy(two), copy(ft)) * copy_sign(one(), gt);
				} else {
					t = gt / copy_sign(d, copy(ft)) + m / t;
				}
			} else {
				t = (m / (s + t) + m / (r + l)) * (one::<T>() + a);
			}
			l = sqrt(t * t + four);
			crt = two / l;
			srt = t / l;
			clt = (crt + srt * m) / a;
			slt = (ht / ft) * srt / a;
		}
	}

	let csl;
	let csr;
	let snl;
	let snr;

	if swap {
		csl = srt;
		snl = crt;
		csr = slt;
		snr = clt;
	} else {
		csl = clt;
		snl = slt;
		csr = crt;
		snr = srt;
	}

	let mut tsign = zero::<T>();
	let sign = copy_sign;
	// *
	// * Correct signs of SSMAX and SSMIN
	// *
	if pmax == 1 {
		tsign = sign(one(), copy(csr)) * sign(one(), copy(csl)) * sign(one(), copy(f))
	}
	if pmax == 2 {
		tsign = sign(one(), copy(snr)) * sign(one(), copy(csl)) * sign(one(), copy(g))
	}
	if pmax == 3 {
		tsign = sign(one(), copy(snr)) * sign(one(), copy(snl)) * sign(one(), copy(h))
	}
	ssmax = sign(ssmax, copy(tsign));
	ssmin = sign(ssmin, tsign * sign(one(), f) * sign(one(), h));
	(ssmin, ssmax, snr, csr, snl, csl)
}

#[math]
pub(super) fn generalized_eigval_2x2<T: RealField>((a11, a12, a21, a22): (T, T, T, T), (b11, b12, b21, b22): (T, T, T, T)) -> (T, T, T, T, T) {
	let safety = from_f64::<T>(128.0);
	let safmin = min_positive::<T>() * safety;
	let safmax = recip(safmin);
	let rtmin = sqrt(safmin);
	let rtmax = recip(rtmin);
	let zero = zero::<T>;
	let one = one::<T>;
	let half = from_f64::<T>(0.5);

	// * Scale A
	let anorm = max(safmin, max(abs(a11) + abs(a21), abs(a12) + abs(a22)));
	let ascale = one() / anorm;
	let a11 = a11 * ascale;
	let a12 = a12 * ascale;
	let a21 = a21 * ascale;
	let a22 = a22 * ascale;

	let mut b11 = b11;
	let b12 = b12;
	let _ = b21;
	let mut b22 = b22;

	// * Perturb B if necessary to insure non-singularity
	let bmin = rtmin * max(max(abs(b11), abs(b12)), max(abs(b22), rtmin));
	if abs(b11) < bmin {
		b11 = copy_sign(copy(bmin), b11);
	}
	if abs(b22) < bmin {
		b22 = copy_sign(bmin, b22);
	}

	// * Scale B
	let bnorm = max(safmin, max(abs(b11), abs(b12) + abs(b22)));
	let bsize = max(abs(b11), abs(b22));
	let bscale = one() / bsize;
	let b11 = b11 * bscale;
	let b12 = b12 * bscale;
	let b22 = b22 * bscale;

	// * Compute larger eigenvalue by method described by C. van Loan
	// *
	// * ( AS is A shifted by -SHIFT*B )

	let binv11 = one() / b11;
	let binv22 = one() / b22;
	let s1 = a11 * binv11;
	let s2 = a22 * binv22;

	let as11;
	let as12;
	let as22;
	let ss;
	let abi22;
	let pp;
	let shift;

	if abs(s1) <= abs(s2) {
		as12 = a12 - s1 * b12;
		as22 = a22 - s1 * b22;
		ss = a21 * (binv11 * binv22);
		abi22 = as22 * binv22 - ss * b12;
		pp = half * abi22;
		shift = s1;
	} else {
		as12 = a12 - s2 * b12;
		as11 = a11 - s2 * b11;
		ss = a21 * (binv11 * binv22);
		abi22 = -ss * b12;
		pp = half * (as11 * binv11 + abi22);
		shift = s2;
	}

	let qq = ss * as12;
	let discr;
	let r;

	if abs(pp * rtmin) >= one() {
		discr = abs2(pp * rtmin) + qq * safmin;
		r = sqrt(abs(discr)) * rtmax;
	} else {
		if abs2(pp) + abs(qq) <= safmin {
			discr = abs2(rtmax * pp) + qq * safmax;
			r = sqrt(abs(discr)) * rtmin;
		} else {
			discr = abs2(pp) + qq;
			r = sqrt(abs(discr));
		}
	}

	let (mut wr1, mut wr2, mut wi);

	// * Note: the test of R in the following IF is to cover the case when
	// * DISCR is small and negative and is flushed to zero during
	// * the calculation of R.  On machines which have a consistent
	// * flush-to-zero threshold and handle numbers above that
	// * threshold correctly, it would not be necessary.
	if discr >= zero() || r == zero() {
		let sum = pp + copy_sign(copy(r), copy(pp));
		let diff = pp - copy_sign(copy(r), copy(pp));
		let wbig = shift + sum;
		// * Compute smaller eigenvalue
		let mut wsmall = shift + diff;

		if abs(wbig) * half > max(safmin, abs(wsmall)) {
			let wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22);
			wsmall = wdet / wbig;
		}
		// * Choose (real) eigenvalue closest to 2,2 element of A*B**(-1)
		// * for WR1.
		if pp > abi22 {
			wr1 = min(wbig, wsmall);
			wr2 = max(wbig, wsmall);
		} else {
			wr1 = max(wbig, wsmall);
			wr2 = min(wbig, wsmall);
		}
		wi = zero();
	} else {
		wr1 = shift + pp;
		wr2 = copy(wr1);
		wi = r;
	}

	// * Further scaling to avoid underflow and overflow in computing
	// * SCALE1 and overflow in computing w*B.
	// *
	// * This scale factor (WSCALE) is bounded from above using C1 and C2,
	// * and from below using C3 and C4.
	// * C1 implements the condition  s A  must never overflow.
	// * C2 implements the condition  w B  must never overflow.
	// * C3, with C2,
	// * implement the condition that s A - w B must never overflow.
	// * C4 implements the condition  s    should not underflow.
	// * C5 implements the condition  max(s,|w|) should be at least 2.

	let c1 = bsize * (safmin * max(one(), ascale));
	let c2 = safmin * max(one(), bnorm);
	let c3 = bsize * safmin;
	let c4;
	let c5;
	if ascale <= one() && bsize <= one() {
		c4 = min(one(), (ascale / safmin) * bsize);
	} else {
		c4 = one();
	}
	if ascale <= one() || bsize <= one() {
		c5 = min(one(), ascale * bsize);
	} else {
		c5 = one();
	}
	let fuzzy1 = from_f64::<T>(1.00001);

	// * Scale first eigenvalue
	let wabs = abs(wr1) + abs(wi);
	let wsize = max(safmin, max(c1, max(fuzzy1 * (wabs * c2 + c3), min(c4, half * max(wabs, c5)))));

	let scale1: T;
	let mut scale2: T = one();

	if wsize != one() {
		let wscale = one() / wsize;
		if wsize > one() {
			scale1 = (max(ascale, bsize) * wscale) * min(ascale, bsize);
		} else {
			scale1 = (min(ascale, bsize) * wscale) * max(ascale, bsize);
		}

		wr1 = wr1 * wscale;
		if wi != zero() {
			wi = wi * wscale;
			wr2 = copy(wr1);
			scale2 = copy(scale1);
		}
	} else {
		scale1 = ascale * bsize;
		scale2 = copy(scale1);
	}

	if wi == zero() {
		let wsize = max(
			safmin,
			max(c1, max(fuzzy1 * (abs(wr2) * c2 + c3), min(c4, mul_real(max(abs(wr2), c5), half)))),
		);
		if wsize != one() {
			let wscale = one() / wsize;
			if wsize > one() {
				scale2 = (max(ascale, bsize) * wscale) * min(ascale, bsize);
			} else {
				scale2 = (min(ascale, bsize) * wscale) * max(ascale, bsize);
			}
			wr2 = wr2 * wscale;
		} else {
			scale2 = ascale * bsize;
		}
	}

	(scale1, scale2, wr1, wr2, wi)
}

#[math]
fn hessenberg_to_qz_unblocked<T: RealField>(
	ilo: usize,
	ihi: usize,
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	Q: Option<MatMut<'_, T>>,
	Z: Option<MatMut<'_, T>>,
	alphar: ColMut<'_, T>,
	alphai: ColMut<'_, T>,
	beta: ColMut<'_, T>,
	eigvals_only: bool,
) {
	let mut H = A;
	let mut T = B;
	let mut Q = Q;
	let mut Z = Z;
	let mut alphar = alphar;
	let mut alphai = alphai;
	let mut beta = beta;

	let n = H.nrows();

	let one = one::<T>;
	let zero = zero::<T>;
	let half = from_f64::<T>(0.5);
	let ulp = eps::<T>();
	let safmin = min_positive::<T>();
	let safmax = one() / safmin;

	// * Set Eigenvalues IHI+1:N
	for j in ihi + 1..n {
		if T[(j, j)] < zero() {
			if !eigvals_only {
				zip!(T.rb_mut().col_mut(j).get_mut(..j + 1)).for_each(|unzip!(x)| *x = -*x);
				zip!(H.rb_mut().col_mut(j).get_mut(..j + 1)).for_each(|unzip!(x)| *x = -*x);
			} else {
				T[(j, j)] = -T[(j, j)];
				H[(j, j)] = -H[(j, j)];
			}
			if let Some(mut Z) = Z.rb_mut() {
				zip!(Z.rb_mut().col_mut(j)).for_each(|unzip!(x)| *x = -*x);
			}
		}
		alphar[j] = copy(H[(j, j)]);
		alphai[j] = zero();
		beta[j] = copy(T[(j, j)]);
	}

	if ihi >= ilo {
		// * MAIN QZ ITERATION LOOP
		// *
		// * Initialize dynamic indices
		// *
		// * Eigenvalues ILAST+1:N have been found.
		// * Column operations modify rows IFRSTM:whatever.
		// * Row operations modify columns whatever:ILASTM.
		// *
		// * If only eigenvalues are being computed, then
		// * IFRSTM is the row of the last splitting row above row ILAST;
		// * this is always at least ILO.
		// * IITER counts iterations since the last eigenvalue was found,
		// * to tell when to use an extraordinary shift.
		// * MAXIT is the maximum number of QZ sweeps allowed.

		let mut ifirst: usize;
		let mut ilast = ihi;
		let mut ifrstm;
		let mut ilastm;

		if !eigvals_only {
			ifrstm = 0;
			ilastm = n - 1;
		} else {
			ifrstm = ilo;
			ilastm = ihi;
		}

		let mut iiter = 0;
		let maxit = 30 * (ihi + 1 - ilo);
		let mut eshift = zero();

		let anorm = H.rb().get(ilo..ihi + 1, ilo..ihi + 1).norm_max();
		let bnorm = T.rb().get(ilo..ihi + 1, ilo..ihi + 1).norm_max();

		let atol = max(safmin, ulp * anorm);
		let btol = max(safmin, ulp * bnorm);

		let ascale = recip(max(safmin, anorm));
		let bscale = recip(max(safmin, bnorm));

		'main_loop: for iter in 0..maxit {
			// * Split the matrix if possible.
			// *
			// * Two tests:
			// * 1: H(j,j-1)=0  or  j=ILO
			// * 2: T(j,j)=0

			_ = iter;

			'goto110: {
				'goto80: {
					'goto70: {
						if ilast == ilo {
							// * Special case: j=ILAST
							break 'goto80;
						} else {
							if abs(H[(ilast, ilast - 1)]) <= max(safmin, ulp * (abs(H[(ilast, ilast)]) + abs(H[(ilast - 1, ilast - 1)]))) {
								H[(ilast, ilast - 1)] = zero();
								break 'goto80;
							}
						}

						if abs(T[(ilast, ilast)]) <= btol {
							T[(ilast, ilast)] = zero();
							break 'goto70;
						}

						// * General case: j<ILAST
						for j in (ilo..ilast).rev() {
							// * Test 1: for H(j,j-1)=0 or j=ILO
							let ilazro: bool;
							let mut ilazr2: bool;

							if j == ilo {
								ilazro = true;
							} else if abs(H[(j, j - 1)]) <= max(safmin, ulp * (abs(H[(j, j)]) + abs(H[(j - 1, j - 1)]))) {
								H[(j, j - 1)] = zero();
								ilazro = true;
							} else {
								ilazro = false;
							}

							// * Test 2: for T(j,j)=0
							if abs(T[(j, j)]) < btol {
								T[(j, j)] = zero();

								// * Test 1a: Check for 2 consecutive small subdiagonals in A
								ilazr2 = false;
								if !ilazro {
									let mut temp = abs(H[(j, j - 1)]);
									let mut temp2 = abs(H[(j, j)]);
									let tempr = max(temp, temp2);
									if tempr < one() && tempr != zero() {
										temp = temp / tempr;
										temp2 = temp2 / tempr;
									}
									if temp * (ascale * abs(H[(j + 1, j)])) <= temp2 * (ascale * atol) {
										ilazr2 = true
									}
								}
								// * If both tests pass (1 & 2), i.e., the leading diagonal
								// * element of B in the block is zero, split a 1x1 block off
								// * at the top. (I.e., at the J-th row/column) The leading
								// * diagonal element of the remainder can also be zero, so
								// * this may have to be done repeatedly.
								if ilazro || ilazr2 {
									for jch in j..ilast {
										let (c, s, r) = make_givens(copy(H[(jch, jch)]), copy(H[(jch + 1, jch)]));
										H[(jch, jch)] = r;
										H[(jch + 1, jch)] = zero();

										let (x, y) = H.rb_mut().get_mut(.., jch + 1..ilastm + 1).two_rows_mut(jch, jch + 1);
										rot(copy(c), copy(s), x, y);

										let (x, y) = T.rb_mut().get_mut(.., jch + 1..ilastm + 1).two_rows_mut(jch, jch + 1);
										rot(copy(c), copy(s), x, y);

										if let Some(mut Q) = Q.rb_mut() {
											let (x, y) = Q.rb_mut().two_cols_mut(jch, jch + 1);
											rot(copy(c), copy(s), x.transpose_mut(), y.transpose_mut());
										}

										if ilazr2 {
											H[(jch, jch - 1)] = H[(jch, jch - 1)] * c;
										}
										ilazr2 = false;

										if abs(T[(jch + 1, jch + 1)]) >= btol {
											if jch + 1 >= ilast {
												break 'goto80;
											} else {
												ifirst = jch + 1;
												break 'goto110;
											}
										}
										T[(jch + 1, jch + 1)] = zero();
									}
									break 'goto70;
								} else {
									// * Only test 2 passed -- chase the zero to T(ILAST,ILAST)
									// * Then process as in the case T(ILAST,ILAST)=0
									for jch in j..ilast {
										let (c, s, r) = make_givens(copy(T[(jch, jch + 1)]), copy(T[(jch + 1, jch + 1)]));
										T[(jch, jch + 1)] = r;
										T[(jch + 1, jch + 1)] = zero();
										if jch + 1 < ilastm {
											let (x, y) = T.rb_mut().get_mut(.., jch + 2..ilastm + 1).two_rows_mut(jch, jch + 1);
											rot(copy(c), copy(s), x, y);
										}
										let (x, y) = H.rb_mut().get_mut(.., jch - 1..ilastm + 1).two_rows_mut(jch, jch + 1);
										rot(copy(c), copy(s), x, y);

										if let Some(mut Q) = Q.rb_mut() {
											let (x, y) = Q.rb_mut().two_cols_mut(jch, jch + 1);
											rot(copy(c), copy(s), x.transpose_mut(), y.transpose_mut());
										}

										let (c, s, r) = make_givens(copy(H[(jch + 1, jch)]), copy(H[(jch + 1, jch - 1)]));
										H[(jch + 1, jch)] = r;
										H[(jch + 1, jch - 1)] = zero();

										let (x, y) = H.rb_mut().get_mut(ifrstm..jch + 1, ..).two_cols_mut(jch, jch - 1);
										rot(copy(c), copy(s), x.transpose_mut(), y.transpose_mut());

										let (x, y) = T.rb_mut().get_mut(ifrstm..jch, ..).two_cols_mut(jch, jch - 1);
										rot(copy(c), copy(s), x.transpose_mut(), y.transpose_mut());

										if let Some(mut Z) = Z.rb_mut() {
											let (x, y) = Z.rb_mut().two_cols_mut(jch, jch - 1);
											rot(copy(c), copy(s), x.transpose_mut(), y.transpose_mut());
										}
									}
									break 'goto70;
								}
							} else if ilazro {
								// * Only test 1 passed -- work on J:ILAST
								ifirst = j;
								break 'goto110;
							}

							// * Neither test passed -- try next J
						}
						// * (Drop-through is "impossible")
						unreachable!();
					}
					// goto70

					// * T(ILAST,ILAST)=0 -- clear H(ILAST,ILAST-1) to split off a
					// * 1x1 block.
					let (c, s, r) = make_givens(copy(H[(ilast, ilast)]), copy(H[(ilast, ilast - 1)]));
					H[(ilast, ilast)] = r;
					H[(ilast, ilast - 1)] = zero();

					let (x, y) = H.rb_mut().get_mut(ifrstm..ilast, ..).two_cols_mut(ilast, ilast - 1);
					rot(copy(c), copy(s), x.transpose_mut(), y.transpose_mut());

					let (x, y) = T.rb_mut().get_mut(ifrstm..ilast, ..).two_cols_mut(ilast, ilast - 1);
					rot(copy(c), copy(s), x.transpose_mut(), y.transpose_mut());

					if let Some(mut Z) = Z.rb_mut() {
						let (x, y) = Z.rb_mut().two_cols_mut(ilast, ilast - 1);
						rot(copy(c), copy(s), x.transpose_mut(), y.transpose_mut());
					}
				}
				// goto80

				// * H(ILAST,ILAST-1)=0 -- Standardize B, set ALPHAR, ALPHAI,
				// * and BETA
				if T[(ilast, ilast)] < zero() {
					if !eigvals_only {
						for j in ifrstm..ilast + 1 {
							H[(j, ilast)] = -H[(j, ilast)];
							T[(j, ilast)] = -T[(j, ilast)];
						}
					} else {
						H[(ilast, ilast)] = -H[(ilast, ilast)];
						T[(ilast, ilast)] = -T[(ilast, ilast)];
					}

					if let Some(mut Z) = Z.rb_mut() {
						for j in 0..n {
							Z[(j, ilast)] = -Z[(j, ilast)];
						}
					}
				}

				alphar[ilast] = copy(H[(ilast, ilast)]);
				alphai[ilast] = zero();
				beta[ilast] = copy(T[(ilast, ilast)]);

				// * Go to next block -- exit if finished.
				ilast = ilast.wrapping_sub(1);
				if ilast == usize::MAX || ilast < ilo {
					return;
				}

				// * Reset counters
				iiter = 0;
				eshift = zero();
				if eigvals_only {
					ilastm = ilast;
					if ifrstm > ilast {
						ifrstm = ilo;
					}
				}
				continue 'main_loop;
			}
			// goto110
			iiter += 1;
			if eigvals_only {
				ifrstm = ifirst;
			}

			let mut wr;
			let mut s1;
			let mut s2: T;
			let mut wr2;
			let mut wi;
			let mut temp;

			'goto200: {
				// * Compute single shifts.
				// *
				// * At this point, IFIRST < ILAST, and the diagonal elements of
				// * T(IFIRST:ILAST,IFIRST,ILAST) are larger than BTOL (in
				// * magnitude)
				if iiter % 10 == 0 {
					// * Exceptional shift.  Chosen for no particularly good reason.
					// * (Single shift only.)
					if from_f64::<T>(maxit as f64) * safmin * abs(H[(ilast, ilast - 1)]) < abs(T[(ilast - 1, ilast - 1)]) {
						eshift = div(H[(ilast, ilast - 1)], T[(ilast - 1, ilast - 1)]);
					} else {
						eshift = eshift + recip(safmin * from_f64::<T>(maxit as f64));
					}
					s1 = one();
					wr = copy(eshift);
				} else {
					// * Shifts based on the generalized eigenvalues of the
					// * bottom-right 2x2 block of A and B. The first eigenvalue
					// * returned by DLAG2 is the Wilkinson shift (AEP p.512),
					(s1, s2, wr, wr2, wi) = generalized_eigval_2x2(
						(
							copy(H[(ilast - 1, ilast - 1)]),
							copy(H[(ilast - 1, ilast)]),
							copy(H[(ilast, ilast - 1)]),
							copy(H[(ilast, ilast)]),
						),
						(
							copy(T[(ilast - 1, ilast - 1)]),
							copy(T[(ilast - 1, ilast)]),
							copy(T[(ilast, ilast - 1)]),
							copy(T[(ilast, ilast)]),
						),
					);
					(copy(s1), copy(s2), copy(wr), copy(wr2), copy(wi));

					if abs((wr / s1) * T[(ilast, ilast)] - H[(ilast, ilast)]) > abs((wr2 / s2) * T[(ilast, ilast)] - H[(ilast, ilast)]) {
						core::mem::swap(&mut wr, &mut wr2);
						core::mem::swap(&mut s1, &mut s2);
					}
					// temp = max(s1, safmin * max(one(), max(abs(wr), abs(wi))));
					if wi != zero() {
						break 'goto200;
					};
				}
				let mut scale;
				// * Fiddle with shift to avoid overflow
				temp = min(ascale, one()) * (half * safmax);
				if s1 > temp {
					scale = temp / s1;
				} else {
					scale = one();
				}

				temp = min(bscale, one()) * (half * safmax);
				if abs(wr) > temp {
					scale = min(scale, temp / abs(wr));
				}
				s1 = scale * s1;
				wr = scale * wr;

				let mut istart;
				'goto130: {
					// * Now check for two consecutive small subdiagonals.
					for j in (ifirst + 1..ilast).rev() {
						istart = j;
						temp = abs(s1 * H[(j, j - 1)]);
						let mut temp2 = abs(s1 * H[(j, j)] - wr * T[(j, j)]);
						let tempr = max(temp, temp2);

						if tempr < one() && tempr != zero() {
							temp = temp / tempr;
							temp2 = temp2 / tempr;
						}
						if abs((ascale * H[(j + 1, j)]) * temp) <= (ascale * atol) * temp2 {
							break 'goto130;
						}
					}

					istart = ifirst;
				}
				// goto130

				single_shift_sweep(s1, istart, wr, ilast, ilastm, ifrstm, H.rb_mut(), T.rb_mut(), Q.rb_mut(), Z.rb_mut());
				continue 'main_loop;
			}
			// goto200

			// * Use Francis double-shift
			// *
			// * Note: the Francis double-shift should work with real shifts,
			// * but only if the block is at least 3x3.
			// * This code may break if this point is reached with
			// * a 2x2 block with real eigenvalues.

			if ifirst + 1 == ilast {
				// * Special case -- 2x2 block with complex eigenvectors
				// *
				// * Step 1: Standardize, that is, rotate so that
				// *
				// * |   ( B11  0  )
				// * B = (         )  with B11 non-negative.
				// * |   (  0  B22 )
				let (mut b22, mut b11, mut sr, mut cr, sl, cl) =
					svd_triu_2x2(copy(T[(ilast - 1, ilast - 1)]), copy(T[(ilast - 1, ilast)]), copy(T[(ilast, ilast)]));

				if b11 < zero() {
					cr = -cr;
					sr = -sr;
					b11 = -b11;
					b22 = -b22;
				}

				let (x, y) = H.rb_mut().subcols_mut(ilast - 1, ilastm + 1 - ifirst).two_rows_mut(ilast - 1, ilast);
				rot(copy(cl), copy(sl), x, y);

				let (x, y) = H.rb_mut().subrows_mut(ifrstm, ilast + 1 - ifrstm).two_cols_mut(ilast - 1, ilast);
				rot(copy(cr), copy(sr), x.transpose_mut(), y.transpose_mut());

				if ilast < ilastm {
					let (x, y) = T.rb_mut().subcols_mut(ilast + 1, ilastm - ilast).two_rows_mut(ilast - 1, ilast);
					rot(copy(cl), copy(sl), x, y);
				}
				if ifrstm + 1 < ilast {
					let (x, y) = T.rb_mut().subrows_mut(ifrstm, ifirst - ifrstm).two_cols_mut(ilast - 1, ilast);
					rot(copy(cr), copy(sr), x.transpose_mut(), y.transpose_mut());
				}

				if let Some(mut Q) = Q.rb_mut() {
					let (x, y) = Q.rb_mut().two_cols_mut(ilast - 1, ilast);
					rot(copy(cl), copy(sl), x.transpose_mut(), y.transpose_mut());
				}
				if let Some(mut Z) = Z.rb_mut() {
					let (x, y) = Z.rb_mut().two_cols_mut(ilast - 1, ilast);
					rot(copy(cr), copy(sr), x.transpose_mut(), y.transpose_mut());
				}

				T[(ilast - 1, ilast - 1)] = copy(b11);
				T[(ilast - 1, ilast)] = zero();
				T[(ilast, ilast - 1)] = zero();
				T[(ilast, ilast)] = copy(b22);

				// * If B22 is negative, negate column ILAST
				if b22 < zero() {
					for j in ifrstm..ilast + 1 {
						H[(j, ilast)] = -H[(j, ilast)];
						T[(j, ilast)] = -T[(j, ilast)];
					}
					if let Some(mut Z) = Z.rb_mut() {
						for j in 0..n {
							Z[(j, ilast)] = -Z[(j, ilast)];
						}
					}
					b22 = -b22;
				}

				// * Step 2: Compute ALPHAR, ALPHAI, and BETA (see refs.)
				// *
				// * Recompute shift

				(s1, _, wr, _, wi) = generalized_eigval_2x2(
					(
						copy(H[(ilast - 1, ilast - 1)]),
						copy(H[(ilast - 1, ilast)]),
						copy(H[(ilast, ilast - 1)]),
						copy(H[(ilast, ilast)]),
					),
					(
						copy(T[(ilast - 1, ilast - 1)]),
						copy(T[(ilast - 1, ilast)]),
						copy(T[(ilast, ilast - 1)]),
						copy(T[(ilast, ilast)]),
					),
				);

				// *f If standardization has perturbed the shift onto real line,
				// * do another (real single-shift) QR step.
				if wi == zero() {
					continue 'main_loop;
				}
				let s1inv = one() / s1;
				// * Do EISPACK (QZVAL) computation of alpha and beta
				// *
				let a11 = copy(H[(ilast - 1, ilast - 1)]);
				let a21 = copy(H[(ilast, ilast - 1)]);
				let a12 = copy(H[(ilast - 1, ilast)]);
				let a22 = copy(H[(ilast, ilast)]);
				// *
				// * Compute complex Givens rotation on right
				// * (Assume some element of C = (sA - wB) > unfl )
				// * |                          __
				// * |         (sA - wB) ( CZ   -SZ )
				// * |                   ( SZ    CZ )
				// *
				let c11r = s1 * a11 - wr * b11;
				let c11i = -wi * b11;
				let c12 = s1 * a12;
				let c21 = s1 * a21;
				let c22r = s1 * a22 - wr * b22;
				let c22i = -wi * b22;

				let mut cz;
				let szr;
				let szi;

				if abs(c11r) + abs(c11i) + abs(c12) > abs(c21) + abs(c22r) + abs(c22i) {
					let t1 = hypot(c12, hypot(c11r, c11i));
					cz = c12 / t1;
					szr = -c11r / t1;
					szi = -c11i / t1;
				} else {
					cz = hypot(c22r, c22i);
					if cz <= safmin {
						cz = zero();
						szr = one();
						szi = zero();
					} else {
						let tempr = c22r / cz;
						let tempi = c22i / cz;
						let t1 = hypot(cz, c21);
						cz = cz / t1;
						szr = -c21 * tempr / t1;
						szi = c21 * tempi / t1;
					}
				}
				// * Compute Givens rotation on left
				// *
				// * |         (  CQ   SQ )
				// * |         (  __      )  A or B
				// * |         ( -SQ   CQ )
				let an = abs(a11) + abs(a12) + abs(a21) + abs(a22);
				let bn = abs(b11) + abs(b22);
				let wabs = abs(wr) + abs(wi);

				let mut cq;
				let mut sqr;
				let mut sqi;
				if s1 * an > wabs * bn {
					cq = cz * b11;
					sqr = szr * b22;
					sqi = -szi * b22;
				} else {
					let a1r = cz * a11 + szr * a12;
					let a1i = szi * a12;
					let a2r = cz * a21 + szr * a22;
					let a2i = szi * a22;
					cq = hypot(a1r, a1i);
					if cq <= safmin {
						cq = zero();
						sqr = one();
						sqi = zero();
					} else {
						let tempr = a1r / cq;
						let tempi = a1i / cq;
						sqr = tempr * a2r + tempi * a2i;
						sqi = tempi * a2r - tempr * a2i;
					}
				}

				let t1 = hypot(cq, hypot(sqr, sqi));
				cq = cq / t1;
				sqr = sqr / t1;
				sqi = sqi / t1;

				// * Compute diagonal elements of QBZ
				// *
				let tempr = sqr * szr - sqi * szi;
				let tempi = sqr * szi + sqi * szr;
				let b1r = cq * cz * b11 + tempr * b22;
				let b1i = tempi * b22;
				let b1a = hypot(b1r, b1i);
				let b2r = cq * cz * b22 + tempr * b11;
				let b2i = -tempi * b11;
				let b2a = hypot(b2r, b2i);
				// *
				// * Normalize so beta > 0, and Im( alpha1 ) > 0
				// *
				beta[ilast - 1] = copy(b1a);
				beta[ilast] = copy(b2a);
				alphar[ilast - 1] = (wr * b1a) * s1inv;
				alphai[ilast - 1] = (wi * b1a) * s1inv;
				alphar[ilast] = (wr * b2a) * s1inv;
				alphai[ilast] = -(wi * b2a) * s1inv;
				// *
				// * Step 3: Go to next block -- exit if finished.
				ilast = ifirst.wrapping_sub(1);
				if ilast < ilo || ilast == usize::MAX {
					break 'main_loop;
				}

				// * Reset counters
				iiter = 0;
				eshift = zero();
				if eigvals_only {
					ilastm = ilast;
					if ifrstm > ilast {
						ifrstm = ilo;
					}
				}
				continue 'main_loop;
			} else {
				double_shift_sweep(
					copy(ascale),
					ilast,
					copy(bscale),
					ifirst,
					ilastm,
					n,
					ifrstm,
					H.rb_mut(),
					T.rb_mut(),
					Q.rb_mut(),
					Z.rb_mut(),
				);
			}
		}
	}

	// * Set Eigenvalues 1:ILO-1
	for j in 0..ilo {
		if T[(j, j)] < zero() {
			if !eigvals_only {
				zip!(T.rb_mut().col_mut(j).get_mut(..j + 1)).for_each(|unzip!(x)| *x = -*x);
				zip!(H.rb_mut().col_mut(j).get_mut(..j + 1)).for_each(|unzip!(x)| *x = -*x);
			} else {
				T[(j, j)] = -T[(j, j)];
				H[(j, j)] = -H[(j, j)];
			}
			if let Some(mut Z) = Z.rb_mut() {
				zip!(Z.rb_mut().col_mut(j)).for_each(|unzip!(x)| *x = -*x);
			}
		}
		alphar[j] = copy(H[(j, j)]);
		alphai[j] = zero();
		beta[j] = copy(T[(j, j)]);
	}
}

#[math]
fn double_shift_sweep<T: RealField>(
	ascale: T,
	ilast: usize,
	bscale: T,
	ifirst: usize,
	ilastm: usize,
	n: usize,
	ifrstm: usize,
	mut H: MatMut<'_, T>,
	mut T: MatMut<'_, T>,
	mut Q: Option<MatMut<'_, T>>,
	mut Z: Option<MatMut<'_, T>>,
) {
	let one = one::<T>;
	let zero = zero::<T>;
	let safmin = min_positive::<T>();

	// * | Usual case: 3x3 or larger block, using Francis implicit
	// * |             double-shift
	// * |
	// * |                          2
	// * | Eigenvalue equation is  w  - c w + d = 0,
	// * |
	// * |                               -1 2        -1
	// * | so compute 1st column of  (A B  )  - c A B   + d
	// * | using the formula in QZIT (from EISPACK)
	// * |
	// * | We assume that the block is at least 3x3

	let mut temp;

	let ad11 = (ascale * H[(ilast - 1, ilast - 1)]) / (bscale * T[(ilast - 1, ilast - 1)]);
	let ad21 = (ascale * H[(ilast, ilast - 1)]) / (bscale * T[(ilast - 1, ilast - 1)]);
	let ad12 = (ascale * H[(ilast - 1, ilast)]) / (bscale * T[(ilast, ilast)]);
	let ad22 = (ascale * H[(ilast, ilast)]) / (bscale * T[(ilast, ilast)]);
	let u12 = T[(ilast - 1, ilast)] / T[(ilast, ilast)];
	let ad11l = (ascale * H[(ifirst, ifirst)]) / (bscale * T[(ifirst, ifirst)]);
	let ad21l = (ascale * H[(ifirst + 1, ifirst)]) / (bscale * T[(ifirst, ifirst)]);
	let ad12l = (ascale * H[(ifirst, ifirst + 1)]) / (bscale * T[(ifirst + 1, ifirst + 1)]);
	let ad22l = (ascale * H[(ifirst + 1, ifirst + 1)]) / (bscale * T[(ifirst + 1, ifirst + 1)]);
	let ad32l = (ascale * H[(ifirst + 2, ifirst + 1)]) / (bscale * T[(ifirst + 1, ifirst + 1)]);
	let u12l = T[(ifirst, ifirst + 1)] / T[(ifirst + 1, ifirst + 1)];

	let mut v_storage = [zero(), zero(), zero()];
	let mut v = ColMut::from_slice_mut(&mut v_storage);

	v[0] = (ad11 - ad11l) * (ad22 - ad11l) - ad12 * ad21 + ad21 * u12 * ad11l + (ad12l - ad11l * u12l) * ad21l;
	v[1] = ((ad22l - ad11l) - ad21l * u12l - (ad11 - ad11l) - (ad22 - ad11l) + ad21 * u12) * ad21l;
	v[2] = ad32l * ad21l;

	let mut tau;
	{
		let (mut head, tail) = v.rb_mut().split_at_row_mut(1);
		tau = linalg::householder::make_householder_in_place(&mut head[0], tail).tau;
		tau = one() / tau;
		head[0] = one();
	}

	let istart = ifirst;
	let mut scale;
	let mut u1;
	let mut u2;

	// * Sweep
	for j in istart..ilast - 1 {
		// * All but last elements: use 3x3 Householder transforms.
		// *
		// * Zero (j-1)st column of A
		if j > istart {
			v[0] = copy(H[(j, j - 1)]);
			v[1] = copy(H[(j + 1, j - 1)]);
			v[2] = copy(H[(j + 2, j - 1)]);

			{
				let (mut head, tail) = v.rb_mut().split_at_row_mut(1);
				tau = linalg::householder::make_householder_in_place(&mut head[0], tail).tau;
				tau = one() / tau;
				H[(j, j - 1)] = copy(head[0]);
				head[0] = one();
				H[(j + 1, j - 1)] = zero();
				H[(j + 2, j - 1)] = zero();
			}
		}

		let v2 = copy(v[1]);
		let v3 = copy(v[2]);
		let t2 = tau * v2;
		let t3 = tau * v3;

		for jc in j..ilastm + 1 {
			temp = H[(j, jc)] + v2 * H[(j + 1, jc)] + v3 * H[(j + 2, jc)];
			H[(j, jc)] = H[(j, jc)] - temp * tau;
			H[(j + 1, jc)] = H[(j + 1, jc)] - temp * t2;
			H[(j + 2, jc)] = H[(j + 2, jc)] - temp * t3;
			let temp2 = T[(j, jc)] + v2 * T[(j + 1, jc)] + v3 * T[(j + 2, jc)];
			T[(j, jc)] = T[(j, jc)] - temp2 * tau;
			T[(j + 1, jc)] = T[(j + 1, jc)] - temp2 * t2;
			T[(j + 2, jc)] = T[(j + 2, jc)] - temp2 * t3;
		}

		if let Some(mut Q) = Q.rb_mut() {
			for jr in 0..n {
				temp = Q[(jr, j)] + v2 * Q[(jr, j + 1)] + v3 * Q[(jr, j + 2)];
				Q[(jr, j)] = Q[(jr, j)] - temp * tau;
				Q[(jr, j + 1)] = Q[(jr, j + 1)] - temp * t2;
				Q[(jr, j + 2)] = Q[(jr, j + 2)] - temp * t3;
			}
		}

		// * Zero j-th column of B (see DLAGBC for details)
		// *
		// * Swap rows to pivot
		let mut ilpivt = false;
		let temp = max(abs(T[(j + 1, j + 1)]), abs(T[(j + 1, j + 2)]));
		let temp2 = max(abs(T[(j + 2, j + 1)]), abs(T[(j + 2, j + 2)]));
		let mut w11;
		let mut w12;
		let mut w21;
		let mut w22;

		'goto250: {
			if max(temp, temp2) < safmin {
				scale = zero();
				u1 = one();
				u2 = zero();
				break 'goto250;
			} else if temp >= temp2 {
				w11 = copy(T[(j + 1, j + 1)]);
				w21 = copy(T[(j + 2, j + 1)]);
				w12 = copy(T[(j + 1, j + 2)]);
				w22 = copy(T[(j + 2, j + 2)]);
				u1 = copy(T[(j + 1, j)]);
				u2 = copy(T[(j + 2, j)]);
			} else {
				w21 = copy(T[(j + 1, j + 1)]);
				w11 = copy(T[(j + 2, j + 1)]);
				w22 = copy(T[(j + 1, j + 2)]);
				w12 = copy(T[(j + 2, j + 2)]);
				u2 = copy(T[(j + 1, j)]);
				u1 = copy(T[(j + 2, j)]);
			}

			// * Swap columns if nec.
			if abs(w12) > abs(w11) {
				ilpivt = true;
				let temp = w12;
				let temp2 = w22;
				w12 = w11;
				w22 = w21;
				w11 = temp;
				w21 = temp2;
			}

			// * LU-factor
			let temp = w21 / w11;
			u2 = u2 - temp * u1;
			w22 = w22 - temp * w12;
			// w21 = zero();

			// * Compute SCALE
			scale = one();
			if abs(w22) < safmin {
				scale = zero();
				u2 = one();
				u1 = -w12 / w11;
				break 'goto250;
			}
			if abs(w22) < abs(u2) {
				scale = abs(w22 / u2)
			}
			if abs(w11) < abs(u1) {
				scale = min(scale, abs(w11 / u1))
			}

			// * Solve
			u2 = (scale * u2) / w22;
			u1 = (scale * u1 - w12 * u2) / w11;
		}
		// goto250
		if ilpivt {
			let temp = u2;
			u2 = u1;
			u1 = temp;
		}

		// * Compute Householder Vector
		let t1 = sqrt(scale * scale + u1 * u1 + u2 * u2);
		tau = one() + scale / t1;
		let vs = -one() / (scale + t1);
		v[0] = one();
		v[1] = vs * u1;
		v[2] = vs * u2;

		// * Apply transformations from the right.
		let v2 = copy(v[1]);
		let v3 = copy(v[2]);
		let t2 = tau * v2;
		let t3 = tau * v3;

		for jr in ifrstm..Ord::min(j + 3, ilast) + 1 {
			let temp = H[(jr, j)] + v2 * H[(jr, j + 1)] + v3 * H[(jr, j + 2)];
			H[(jr, j)] = H[(jr, j)] - temp * tau;
			H[(jr, j + 1)] = H[(jr, j + 1)] - temp * t2;
			H[(jr, j + 2)] = H[(jr, j + 2)] - temp * t3;
		}

		for jr in ifrstm..j + 3 {
			let temp = T[(jr, j)] + v2 * T[(jr, j + 1)] + v3 * T[(jr, j + 2)];
			T[(jr, j)] = T[(jr, j)] - temp * tau;
			T[(jr, j + 1)] = T[(jr, j + 1)] - temp * t2;
			T[(jr, j + 2)] = T[(jr, j + 2)] - temp * t3;
		}

		if let Some(mut Z) = Z.rb_mut() {
			for jr in 0..n {
				let temp = Z[(jr, j)] + v2 * Z[(jr, j + 1)] + v3 * Z[(jr, j + 2)];
				Z[(jr, j)] = Z[(jr, j)] - temp * tau;
				Z[(jr, j + 1)] = Z[(jr, j + 1)] - temp * t2;
				Z[(jr, j + 2)] = Z[(jr, j + 2)] - temp * t3;
			}
		}
		T[(j + 1, j)] = zero();
		T[(j + 2, j)] = zero();
	}
	// * Last elements: Use Givens rotations
	// *
	// * Rotations from the left
	let j = ilast - 1;
	let (c, s, temp) = make_givens(copy(H[(j, j - 1)]), copy(H[(j + 1, j - 1)]));
	H[(j, j - 1)] = temp;
	H[(j + 1, j - 1)] = zero();
	for jc in j..ilastm + 1 {
		let temp = c * H[(j, jc)] + s * H[(j + 1, jc)];
		H[(j + 1, jc)] = -s * H[(j, jc)] + c * H[(j + 1, jc)];
		H[(j, jc)] = temp;
		let temp2 = c * T[(j, jc)] + s * T[(j + 1, jc)];
		T[(j + 1, jc)] = -s * T[(j, jc)] + c * T[(j + 1, jc)];
		T[(j, jc)] = temp2;
	}

	if let Some(mut Q) = Q.rb_mut() {
		for jr in 0..n {
			let temp = c * Q[(jr, j)] + s * Q[(jr, j + 1)];
			Q[(jr, j + 1)] = -s * Q[(jr, j)] + c * Q[(jr, j + 1)];
			Q[(jr, j)] = temp;
		}
	}

	// * Rotations from the right.
	let (c, s, temp) = make_givens(copy(T[(j + 1, j + 1)]), copy(T[(j + 1, j)]));
	T[(j + 1, j + 1)] = temp;
	T[(j + 1, j)] = zero();

	for jr in ifrstm..ilast + 1 {
		let temp = c * H[(jr, j + 1)] + s * H[(jr, j)];
		H[(jr, j)] = -s * H[(jr, j + 1)] + c * H[(jr, j)];
		H[(jr, j + 1)] = temp;
	}
	for jr in ifrstm..ilast {
		let temp = c * T[(jr, j + 1)] + s * T[(jr, j)];
		T[(jr, j)] = -s * T[(jr, j + 1)] + c * T[(jr, j)];
		T[(jr, j + 1)] = temp;
	}
	if let Some(mut Z) = Z.rb_mut() {
		for jr in 0..n {
			let temp = c * Z[(jr, j + 1)] + s * Z[(jr, j)];
			Z[(jr, j)] = -s * Z[(jr, j + 1)] + c * Z[(jr, j)];
			Z[(jr, j + 1)] = temp;
		}
	}
}

#[math]
fn single_shift_sweep<T: RealField>(
	s1: T,
	istart: usize,
	wr: T,
	ilast: usize,
	ilastm: usize,
	ifrstm: usize,
	mut H: MatMut<'_, T>,
	mut T: MatMut<'_, T>,
	mut Q: Option<MatMut<'_, T>>,
	mut Z: Option<MatMut<'_, T>>,
) {
	// * Do an implicit single-shift QZ sweep.
	// *
	// * Initial Q
	let zero = zero::<T>;
	let n = H.nrows();

	let mut temp = s1 * H[(istart, istart)] - wr * T[(istart, istart)];
	let temp2 = s1 * H[(istart + 1, istart)];

	let (mut c, mut s, _) = make_givens(temp, temp2);

	// * Sweep
	for j in istart..ilast {
		if j > istart {
			(c, s, temp) = make_givens(copy(H[(j, j - 1)]), copy(H[(j + 1, j - 1)]));
			H[(j, j - 1)] = temp;
			H[(j + 1, j - 1)] = zero();
		}

		for jc in j..ilastm + 1 {
			temp = c * H[(j, jc)] + s * H[(j + 1, jc)];
			H[(j + 1, jc)] = -s * H[(j, jc)] + c * H[(j + 1, jc)];
			H[(j, jc)] = temp;
			let temp2 = c * T[(j, jc)] + s * T[(j + 1, jc)];
			T[(j + 1, jc)] = -s * T[(j, jc)] + c * T[(j + 1, jc)];
			T[(j, jc)] = temp2;
		}
		if let Some(mut Q) = Q.rb_mut() {
			for jr in 0..n {
				temp = c * Q[(jr, j)] + s * Q[(jr, j + 1)];
				Q[(jr, j + 1)] = -s * Q[(jr, j)] + c * Q[(jr, j + 1)];
				Q[(jr, j)] = temp;
			}
		}

		(c, s, temp) = make_givens(copy(T[(j + 1, j + 1)]), copy(T[(j + 1, j)]));
		T[(j + 1, j + 1)] = temp;
		T[(j + 1, j)] = zero();

		for jr in ifrstm..Ord::min(j + 2, ilast) + 1 {
			temp = c * H[(jr, j + 1)] + s * H[(jr, j)];
			H[(jr, j)] = -s * H[(jr, j + 1)] + c * H[(jr, j)];
			H[(jr, j + 1)] = temp;
		}
		for jr in ifrstm..j + 1 {
			temp = c * T[(jr, j + 1)] + s * T[(jr, j)];
			T[(jr, j)] = -s * T[(jr, j + 1)] + c * T[(jr, j)];
			T[(jr, j + 1)] = temp;
		}
		if let Some(mut Z) = Z.rb_mut() {
			for jr in 0..n {
				temp = c * Z[(jr, j + 1)] + s * Z[(jr, j)];
				Z[(jr, j)] = -s * Z[(jr, j + 1)] + c * Z[(jr, j)];
				Z[(jr, j + 1)] = temp;
			}
		}
	}
}

/// computes the layout of the workspace required to compute a real matrix pair's QZ
/// decomposition, assuming the pair is already in generalized hessenberg form.
#[math]
pub fn hessenberg_to_qz_scratch<T: RealField>(n: usize, par: Par, params: GeneralizedSchurParams) -> StackReq {
	hessenberg_to_qz_blocked_scratch::<T>(n, par, params)
}

#[math]
fn hessenberg_to_qz_blocked_scratch<T: RealField>(n: usize, par: Par, params: GeneralizedSchurParams) -> StackReq {
	let nmin = Ord::max(15, params.blocking_threshold);
	if n < nmin {
		return StackReq::empty();
	}

	let nw = (n - 3) / 3;
	let nsr = (params.recommended_shift_count)(n, n);
	let rcost = (params.relative_cost_estimate_of_shift_chase_to_matmul)(n, n);

	let itemp1 = (nsr as f64 / sqrt(1.0 + 2.0 * (nsr as f64) / (rcost as f64 / 100.0 * n as f64))) as usize;
	let itemp1 = ((itemp1.saturating_sub(1)) / 4) * 4 + 4;
	let nbr = nsr + itemp1;

	let qc_aed = linalg::temp_mat_scratch::<T>(nw, nw);
	let qc_sweep = linalg::temp_mat_scratch::<T>(nbr, nbr);

	StackReq::any_of(&[
		StackReq::all_of(&[qc_aed, qc_aed, aed_scratch::<T>(n, nw, par, params)]),
		StackReq::all_of(&[qc_sweep, qc_sweep, multishift_sweep_scratch::<T>(n, nsr)]),
	])
}
#[math]
fn multishift_sweep_scratch<T: RealField>(n: usize, ns: usize) -> StackReq {
	linalg::temp_mat_scratch::<T>(n, 2 * ns)
}
#[math]
fn aed_scratch<T: RealField>(n: usize, nw: usize, par: Par, params: GeneralizedSchurParams) -> StackReq {
	StackReq::any_of(&[
		hessenberg_to_qz_blocked_scratch::<T>(nw, par, params),
		StackReq::all_of(&[
			linalg::temp_mat_scratch::<T>(4, 4),
			linalg::temp_mat_scratch::<T>(4, 4),
			linalg::temp_mat_scratch::<T>(8, 8),
			linalg::temp_mat_scratch::<T>(8, 1),
			StackReq::all_of(&[StackReq::new::<usize>(8)]),
			StackReq::any_of(&[
				linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, T>(8, 8, Par::Seq, Default::default()),
				linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, T>(n, 1, Par::Seq),
			]),
		]),
		linalg::temp_mat_scratch::<T>(nw, n),
		linalg::temp_mat_scratch::<T>(n, nw),
	])
}

/// computes a real matrix pair's QZ decomposition, assuming the pair is already in generalized
/// hessenberg form.  
/// the unitary transformations $Q$ and $Z$ resulting from the QZ decomposition are postmultiplied
/// into the input-output parameters `Q_inout` and `Z_inout`.
///
/// if both the generalized eigenvalues and eigenvectors are desired, then `eigenvectors` may be set
/// to `ComputeEigenvectors::Yes`. in this case the input matrices $A$ and $B$ are overwritten by
/// their QZ form $(S, T)$ such that $S$ is upper quasi-triangular and $T$ is upper triangular.
///
/// if only the generalized eigenvalues are desired, then `eigenvectors` may be set to
/// `ComputeEigenvectors::No`. note that in this case, the input matrices $A$ and $B$ are still
/// clobbered, and contain unspecified values on output.
#[track_caller]
#[math]
pub fn hessenberg_to_qz<T: RealField>(
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	Q_inout: Option<MatMut<'_, T>>,
	Z_inout: Option<MatMut<'_, T>>,
	alphar: ColMut<'_, T>,
	alphai: ColMut<'_, T>,
	beta: ColMut<'_, T>,
	eigenvectors: linalg::evd::ComputeEigenvectors,
	par: Par,
	params: GeneralizedSchurParams,
	stack: &mut MemStack,
) {
	let eigvals_only = eigenvectors == linalg::evd::ComputeEigenvectors::No;

	let n = A.nrows();
	let (Q_nrows, Q_ncols) = Q_inout.rb().map(|m| (m.nrows(), m.ncols())).unwrap_or((n, n));
	let (Z_nrows, Z_ncols) = Z_inout.rb().map(|m| (m.nrows(), m.ncols())).unwrap_or((n, n));
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

	if n == 0 {
		return;
	}
	hessenberg_to_qz_blocked(0, n - 1, A, B, Q_inout, Z_inout, alphar, alphai, beta, eigvals_only, par, params, stack)
}

#[math]
fn hessenberg_to_qz_blocked<T: RealField>(
	ilo: usize,
	ihi: usize,
	mut A: MatMut<'_, T>,
	mut B: MatMut<'_, T>,
	mut Q: Option<MatMut<'_, T>>,
	mut Z: Option<MatMut<'_, T>>,
	mut alphar: ColMut<'_, T>,
	mut alphai: ColMut<'_, T>,
	mut beta: ColMut<'_, T>,
	eigvals_only: bool,
	par: Par,
	params: GeneralizedSchurParams,
	stack: &mut MemStack,
) {
	let n = A.nrows();
	let ulp = eps::<T>();
	let safmin = min_positive::<T>();
	let smlnum = safmin * (from_f64::<T>(n as f64) / ulp);
	let zero = zero::<T>;
	let one = one::<T>;

	let bnorm = B.rb().get(ilo..ihi + 1, ilo..ihi + 1).norm_max();
	let btol = max(safmin, ulp * bnorm);

	let mut istart = ilo;
	let mut istop = ihi;
	let maxit = 30 * (ihi + 1 - ilo);
	let mut ld = 0usize;
	let mut eshift = zero();
	let nh = ihi - ilo + 1;

	let nmin = Ord::max(15, params.blocking_threshold);

	let nibble = params.nibble_threshold;

	let nwr = (params.recommended_deflation_window)(n, nh);

	let nsr = (params.recommended_shift_count)(n, nh);

	let rcost = (params.relative_cost_estimate_of_shift_chase_to_matmul)(n, nh);

	let itemp1 = (nsr as f64 / sqrt(1.0 + 2.0 * (nsr as f64) / (rcost as f64 / 100.0 * n as f64))) as usize;
	let itemp1 = ((itemp1.saturating_sub(1)) / 4) * 4 + 4;
	let nbr = nsr + itemp1;

	if n < nmin {
		hessenberg_to_qz_unblocked(
			ilo,
			ihi,
			A.rb_mut(),
			B.rb_mut(),
			Q.rb_mut(),
			Z.rb_mut(),
			alphar.rb_mut(),
			alphai.rb_mut(),
			beta.rb_mut(),
			eigvals_only,
		);
		return;
	}

	let nw_max = (n - 3) / 3;
	let nw_max = nw_max + nw_max % 2;
	for iter in 0..maxit {
		_ = iter;

		if istop == usize::MAX || istart + 1 >= istop {
			break;
		}

		// * Check deflations at the end
		if abs(A[(istop - 1, istop - 2)]) <= max(smlnum, ulp * (abs(A[(istop - 1, istop - 1)]) + abs(A[(istop - 2, istop - 2)]))) {
			A[(istop - 1, istop - 2)] = zero();
			istop -= 2;
			ld = 0;
			eshift = zero();
		} else if abs(A[(istop, istop - 1)]) <= max(smlnum, ulp * (abs(A[(istop, istop)]) + abs(A[(istop - 1, istop - 1)]))) {
			istop -= 1;
			ld = 0;
			eshift = zero();
		}

		// * Check deflations at the start
		if abs(A[(istart + 2, istart + 1)]) <= max(smlnum, ulp * (abs(A[(istart + 1, istart + 1)]) + abs(A[(istart + 2, istart + 2)]))) {
			A[(istart + 2, istart + 1)] = zero();
			istart = istart + 2;
			ld = 0;
			eshift = zero();
		} else if abs(A[(istart + 1, istart)]) <= max(smlnum, ulp * (abs(A[(istart, istart)]) + abs(A[(istart + 1, istart + 1)]))) {
			A[(istart + 1, istart)] = zero();
			istart = istart + 1;
			ld = 0;
			eshift = zero();
		}

		if istart + 1 >= istop {
			break;
		}

		// * Check interior deflations
		let mut istart2 = istart;
		for k in (istart + 1..istop + 1).rev() {
			if abs(A[(k, k - 1)]) <= max(smlnum, ulp * (abs(A[(k, k)]) + abs(A[(k - 1, k - 1)]))) {
				A[(k, k - 1)] = zero();
				istart2 = k;
				break;
			}
		}

		let (istartm, istopm);
		if !eigvals_only {
			istartm = 0;
			istopm = n - 1;
		} else {
			istartm = istart2;
			istopm = istop;
		}

		// * Check infinite eigenvalues, this is done without blocking so might
		// * slow down the method when many infinite eigenvalues are present
		let mut k = istop + 1;
		while k > istart2 {
			k -= 1;

			if abs(B[(k, k)]) < btol {
				// * A diagonal element of B is negligible, move it
				// * to the top and deflate it

				for k2 in (istart2 + 1..k + 1).rev() {
					let (c1, s1, temp) = make_givens(copy(B[(k2 - 1, k2)]), copy(B[(k2 - 1, k2 - 1)]));
					B[(k2 - 1, k2)] = temp;
					B[(k2 - 1, k2 - 1)] = zero();

					let (x, y) = B.rb_mut().get_mut(istartm..k2 - 1, ..).two_cols_mut(k2, k2 - 1);
					rot(copy(c1), copy(s1), x.transpose_mut(), y.transpose_mut());

					let (x, y) = A.rb_mut().get_mut(istartm..Ord::min(istop, k2 + 1) + 1, ..).two_cols_mut(k2, k2 - 1);
					rot(copy(c1), copy(s1), x.transpose_mut(), y.transpose_mut());

					if let Some(Z) = Z.rb_mut() {
						let (x, y) = Z.two_cols_mut(k2, k2 - 1);
						rot(copy(c1), copy(s1), x.transpose_mut(), y.transpose_mut());
					}

					if k2 < istop {
						let (c1, s1, temp) = make_givens(copy(A[(k2, k2 - 1)]), copy(A[(k2 + 1, k2 - 1)]));
						A[(k2, k2 - 1)] = temp;
						A[(k2 + 1, k2 - 1)] = zero();

						let (x, y) = A.rb_mut().get_mut(.., k2..istopm + 1).two_rows_mut(k2, k2 + 1);
						rot(copy(c1), copy(s1), x, y);

						let (x, y) = B.rb_mut().get_mut(.., k2..istopm + 1).two_rows_mut(k2, k2 + 1);
						rot(copy(c1), copy(s1), x, y);
						B[(k2 + 1, k2)] = zero();

						if let Some(Q) = Q.rb_mut() {
							let (x, y) = Q.two_cols_mut(k2, k2 + 1);
							rot(copy(c1), conj(s1), x.transpose_mut(), y.transpose_mut());
						}
					}
				}

				if istart2 < istop {
					let (c1, s1, temp) = make_givens(copy(A[(istart2, istart2)]), copy(A[(istart2 + 1, istart2)]));
					A[(istart2, istart2)] = temp;
					A[(istart2 + 1, istart2)] = zero();

					let (x, y) = A.rb_mut().get_mut(.., istart2 + 1..istopm + 1).two_rows_mut(istart2, istart2 + 1);
					rot(copy(c1), copy(s1), x, y);

					let (x, y) = B.rb_mut().get_mut(.., istart2 + 1..istopm + 1).two_rows_mut(istart2, istart2 + 1);
					rot(copy(c1), copy(s1), x, y);

					if let Some(Q) = Q.rb_mut() {
						let (x, y) = Q.two_cols_mut(istart2, istart2 + 1);
						rot(copy(c1), conj(s1), x.transpose_mut(), y.transpose_mut());
					}
				}

				istart2 += 1;
			}
		}

		// * istart2 now points to the top of the bottom right
		// * unreduced Hessenberg block
		if istart2 >= istop {
			istop = istart2 - 1;
			ld = 0;
			eshift = zero();
			continue;
		}

		let mut nw = nwr;
		let nshifts = nsr;
		let nblock = nbr;

		if istop + 1 - istart2 < nmin {
			// * Setting nw to the size of the subblock will make AED deflate
			// * all the eigenvalues. This is slightly more efficient than just
			// * using qz_small because the off diagonal part gets updated via BLAS.
			if istop + 1 - istart < nmin {
				nw = istop + 1 - istart;
				istart2 = istart;
			} else {
				nw = istop + 1 - istart2;
			}
		}

		nw = Ord::min(nw, nw_max);

		let (n_undeflated, n_deflated);
		{
			let (mut QC, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(nw, nw, stack) };
			let mut QC = QC.as_mat_mut();
			let (mut ZC, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(nw, nw, stack) };
			let mut ZC = ZC.as_mat_mut();

			// * Time for AED
			(n_undeflated, n_deflated) = aggressive_early_deflation(
				eigvals_only,
				istart2,
				istop,
				nw,
				A.rb_mut(),
				B.rb_mut(),
				Q.rb_mut(),
				Z.rb_mut(),
				alphar.rb_mut(),
				alphai.rb_mut(),
				beta.rb_mut(),
				QC.rb_mut(),
				ZC.rb_mut(),
				par,
				params,
				stack,
			);
		}

		if n_deflated > 0 {
			istop = istop.wrapping_sub(n_deflated);
			ld = 0;
			eshift = zero();
		}

		if (100 * n_deflated > nibble * (n_deflated + n_undeflated)) || (istop.wrapping_add(1) - istart2 < nmin) {
			// * AED has uncovered many eigenvalues. Skip a QZ sweep and run
			// * AED again.
			continue;
		}

		ld += 1;
		let mut ns = Ord::min(nshifts, istop.wrapping_sub(istart2));
		ns = Ord::min(ns, n_undeflated);
		let shiftpos = istop - n_undeflated + 1;

		// * Shuffle shifts to put double shifts in front
		// * This ensures that we don't split up a double shift
		for i in (shiftpos..istop - 1).step_by(2) {
			if alphai[i] != -alphai[i + 1] {
				for mut v in [alphar.rb_mut(), alphai.rb_mut(), beta.rb_mut()] {
					let swap = copy(v[i]);
					v[i] = copy(v[i + 1]);
					v[i + 1] = copy(v[i + 2]);
					v[i + 2] = swap;
				}
			}
		}

		if ld % 6 == 0 {
			// * Exceptional shift.  Chosen for no particularly good reason.
			if from_f64::<T>(maxit as f64) * safmin * abs(A[(istop, istop - 1)]) < abs(B[(istop - 1, istop - 1)]) {
				eshift = A[(istop, istop - 1)] / B[(istop - 1, istop - 1)];
			} else {
				eshift = eshift + one() / (safmin * (from_f64::<T>(maxit as f64)));
			}
			alphar[shiftpos] = one();
			alphar[shiftpos + 1] = zero();
			alphai[shiftpos] = zero();
			alphai[shiftpos + 1] = zero();
			beta[shiftpos] = copy(eshift);
			beta[shiftpos + 1] = copy(eshift);
			ns = 2;
		}

		let (mut QC, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(nblock, nblock, stack) };
		let mut QC = QC.as_mat_mut();
		let (mut ZC, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(nblock, nblock, stack) };
		let mut ZC = ZC.as_mat_mut();

		// * Time for a QZ sweep
		multishift_sweep(
			istart2,
			istop,
			alphar.rb_mut().subrows_mut(shiftpos, ns),
			alphai.rb_mut().subrows_mut(shiftpos, ns),
			beta.rb_mut().subrows_mut(shiftpos, ns),
			A.rb_mut(),
			B.rb_mut(),
			Q.rb_mut(),
			Z.rb_mut(),
			QC.rb_mut(),
			ZC.rb_mut(),
			eigvals_only,
			par,
			stack,
		);
	}

	// * Call ZHGEQZ to normalize the eigenvalue blocks and set the eigenvalues
	// * If all the eigenvalues have been found, ZHGEQZ will not do any iterations
	// * and only normalize the blocks. In case of a rare convergence failure,
	// * the single shift might perform better.

	hessenberg_to_qz_unblocked(
		ilo,
		ihi,
		A.rb_mut(),
		B.rb_mut(),
		Q.rb_mut(),
		Z.rb_mut(),
		alphar.rb_mut(),
		alphai.rb_mut(),
		beta.rb_mut(),
		eigvals_only,
	);
}

#[math]
fn laqz1<T: RealField>(A: MatRef<'_, T>, B: MatRef<'_, T>, sr1: T, sr2: T, si: T, beta1: T, beta2: T, mut v: ColMut<'_, T>) {
	let zero = zero::<T>;
	let one = one::<T>;

	let safmin = min_positive::<T>();
	let safmax = one() / safmin;

	let mut w1 = beta1 * A[(0, 0)] - sr1 * B[(0, 0)];
	let mut w2 = beta1 * A[(1, 0)] - sr1 * B[(1, 0)];
	let scale1 = sqrt(abs(w1)) * sqrt(abs(w2));
	if scale1 >= safmin && scale1 <= safmax {
		w1 = w1 / scale1;
		w2 = w2 / scale1;
	}

	// *
	// * Solve linear system
	// *
	w2 = w2 / B[(1, 1)];
	w1 = (w1 - B[(0, 1)] * w2) / B[(0, 0)];
	let scale2 = sqrt(abs(w1)) * sqrt(abs(w2));
	if scale2 >= safmin && scale2 <= safmax {
		w1 = w1 / scale2;
		w2 = w2 / scale2;
	}
	// *
	// * Apply second shift
	// *
	v[0] = beta2 * (A[(0, 0)] * w1 + A[(0, 1)] * w2) - sr2 * (B[(0, 0)] * w1 + B[(0, 1)] * w2);
	v[1] = beta2 * (A[(1, 0)] * w1 + A[(1, 1)] * w2) - sr2 * (B[(1, 0)] * w1 + B[(1, 1)] * w2);
	v[2] = beta2 * (A[(2, 0)] * w1 + A[(2, 1)] * w2) - sr2 * (B[(2, 0)] * w1 + B[(2, 1)] * w2);
	// *
	// * Account for imaginary part
	// *
	v[0] = v[0] + si * si * B[(0, 0)] / scale1 / scale2;
	// *
	// * Check for overflow
	// *
	if abs(v[0]) > safmax || abs(v[1]) > safmax || abs(v[2]) > safmax || is_nan(v[0]) || is_nan(v[1]) || is_nan(v[2]) {
		v[0] = zero();
		v[1] = zero();
		v[2] = zero();
	}
}

#[math]
fn chase_bulge_2x2<T: RealField>(
	k: usize,
	istartm: usize,
	istopm: usize,
	ihi: usize,
	mut A: MatMut<'_, T>,
	mut B: MatMut<'_, T>,
	qstart: usize,
	mut Q: Option<MatMut<'_, T>>,
	zstart: usize,
	mut Z: Option<MatMut<'_, T>>,
) {
	let zero = zero::<T>;

	let mut h_storage = [zero(), zero(), zero(), zero(), zero(), zero()];
	let mut h = MatMut::from_column_major_slice_mut(&mut h_storage, 2, 3);

	if k + 2 == ihi {
		// * Shift is located on the edge of the matrix, remove it
		h.copy_from(B.rb().get(ihi - 1..ihi + 1, ihi - 2..ihi + 1));

		// * Make H upper triangular
		let (c1, s1, temp) = make_givens(copy(h[(0, 0)]), copy(h[(1, 0)]));
		h[(0, 0)] = temp;
		h[(1, 0)] = zero();
		rot_rows(c1, s1, h.rb_mut().get_mut(.., 1..), 0, 1);

		let (c1, s1, _) = make_givens(copy(h[(1, 2)]), copy(h[(1, 1)]));
		rot_cols(copy(c1), copy(s1), h.rb_mut().get_mut(..1, ..), 2, 1);
		let (c2, s2, _) = make_givens(copy(h[(0, 1)]), copy(h[(0, 0)]));
		for mut M in [A.rb_mut(), B.rb_mut()] {
			rot_cols(copy(c1), copy(s1), M.rb_mut().subrows_mut(istartm, ihi + 1 - istartm), ihi, ihi - 1);
			rot_cols(copy(c2), copy(s2), M.rb_mut().subrows_mut(istartm, ihi + 1 - istartm), ihi - 1, ihi - 2);
		}
		B[(ihi - 1, ihi - 2)] = zero();
		B[(ihi, ihi - 2)] = zero();

		if let Some(mut Z) = Z.rb_mut() {
			rot_cols(copy(c1), copy(s1), Z.rb_mut(), ihi - zstart, ihi - 1 - zstart);
			rot_cols(copy(c2), copy(s2), Z.rb_mut(), ihi - 1 - zstart, ihi - 2 - zstart);
		}

		let (c1, s1, temp) = make_givens(copy(A[(ihi - 1, ihi - 2)]), copy(A[(ihi, ihi - 2)]));
		A[(ihi - 1, ihi - 2)] = temp;
		A[(ihi, ihi - 2)] = zero();
		for mut M in [A.rb_mut(), B.rb_mut()] {
			rot_rows(copy(c1), copy(s1), M.rb_mut().subcols_mut(ihi - 1, istopm + 2 - ihi), ihi - 1, ihi);
		}

		if let Some(mut Q) = Q.rb_mut() {
			rot_cols(copy(c1), copy(s1), Q.rb_mut(), ihi - 1 - qstart, ihi - qstart);
		}

		let (c1, s1, temp) = make_givens(copy(B[(ihi, ihi)]), copy(B[(ihi, ihi - 1)]));
		B[(ihi, ihi)] = temp;
		B[(ihi, ihi - 1)] = zero();
		rot_cols(copy(c1), copy(s1), B.rb_mut().subrows_mut(istartm, ihi - istartm), ihi, ihi - 1);
		rot_cols(copy(c1), copy(s1), A.rb_mut().subrows_mut(istartm, ihi + 1 - istartm), ihi, ihi - 1);
		if let Some(mut Z) = Z.rb_mut() {
			rot_cols(copy(c1), copy(s1), Z.rb_mut(), ihi - zstart, ihi - zstart - 1);
		}
	} else {
		// *
		// * Normal operation, move bulge down
		// *
		h.copy_from(B.rb().get(k + 1..k + 3, k..k + 3));
		// *
		// * Make H upper triangular
		// *

		// * Make H upper triangular
		let (c1, s1, temp) = make_givens(copy(h[(0, 0)]), copy(h[(1, 0)]));
		h[(0, 0)] = temp;
		h[(1, 0)] = zero();
		rot_rows(c1, s1, h.rb_mut(), 0, 1);

		// *
		// * Calculate Z1 and Z2
		// *

		let (c1, s1, _) = make_givens(copy(h[(1, 2)]), copy(h[(1, 1)]));
		rot_cols(copy(c1), copy(s1), h.rb_mut().get_mut(..1, ..), 2, 1);
		let (c2, s2, _) = make_givens(copy(h[(0, 1)]), copy(h[(0, 0)]));

		// *
		// * Apply transformations from the right
		// *
		rot_cols(copy(c1), copy(s1), A.rb_mut().subrows_mut(istartm, k + 4 - istartm), k + 2, k + 1);
		rot_cols(copy(c2), copy(s2), A.rb_mut().subrows_mut(istartm, k + 4 - istartm), k + 1, k);

		rot_cols(copy(c1), copy(s1), B.rb_mut().subrows_mut(istartm, k + 3 - istartm), k + 2, k + 1);
		rot_cols(copy(c2), copy(s2), B.rb_mut().subrows_mut(istartm, k + 3 - istartm), k + 1, k);

		if let Some(mut Z) = Z.rb_mut() {
			rot_cols(copy(c1), copy(s1), Z.rb_mut(), k + 2 - zstart, k + 1 - zstart);
			rot_cols(copy(c2), copy(s2), Z.rb_mut(), k + 1 - zstart, k - zstart);
		}

		B[(k + 1, k)] = zero();
		B[(k + 2, k)] = zero();

		// *
		// * Calculate Q1 and Q2
		// *
		let (c1, s1, temp) = make_givens(copy(A[(k + 2, k)]), copy(A[(k + 3, k)]));
		A[(k + 2, k)] = temp;
		A[(k + 3, k)] = zero();
		let (c2, s2, temp) = make_givens(copy(A[(k + 1, k)]), copy(A[(k + 2, k)]));
		A[(k + 1, k)] = temp;
		A[(k + 2, k)] = zero();

		// *
		// * Apply transformations from the left
		// *
		for mut M in [A.rb_mut(), B.rb_mut()] {
			rot_rows(copy(c1), copy(s1), M.rb_mut().subcols_mut(k + 1, istopm - k), k + 2, k + 3);
			rot_rows(copy(c2), copy(s2), M.rb_mut().subcols_mut(k + 1, istopm - k), k + 1, k + 2);
		}

		if let Some(mut Q) = Q.rb_mut() {
			rot_cols(copy(c1), copy(s1), Q.rb_mut(), k + 2 - qstart, k + 3 - qstart);
			rot_cols(copy(c2), copy(s2), Q.rb_mut(), k + 1 - qstart, k + 2 - qstart);
		}
	}
}

#[math]
fn aggressive_early_deflation<T: RealField>(
	eigvals_only: bool,
	ilo: usize,
	ihi: usize,
	nw: usize,
	mut A: MatMut<'_, T>,
	mut B: MatMut<'_, T>,
	mut Q: Option<MatMut<'_, T>>,
	mut Z: Option<MatMut<'_, T>>,
	mut alphar: ColMut<'_, T>,
	mut alphai: ColMut<'_, T>,
	mut beta: ColMut<'_, T>,
	mut QC: MatMut<'_, T>,
	mut ZC: MatMut<'_, T>,
	par: Par,
	params: GeneralizedSchurParams,
	stack: &mut MemStack,
) -> (usize, usize) {
	let n = A.nrows();

	let zero = zero::<T>;
	let one = one::<T>;
	let ulp = eps::<T>();
	let safmin = min_positive::<T>();
	let smlnum = safmin * (from_f64::<T>(n as f64) / ulp);

	let jw = Ord::min(nw, ihi + 1 - ilo);
	let kwtop = ihi + 1 - jw;

	let s;
	if kwtop == ilo {
		s = zero();
	} else {
		s = copy(A[(kwtop, kwtop - 1)]);
	};

	let mut ifst: usize;
	let mut ilst: usize;

	if ihi == kwtop {
		// * 1 by 1 deflation window, just try a regular deflation
		alphar[kwtop] = copy(A[(kwtop, kwtop)]);
		alphai[kwtop] = copy(A[(kwtop, kwtop)]);
		beta[kwtop] = copy(B[(kwtop, kwtop)]);
		// ns = 1;
		// nd = 0;

		if abs(s) <= max(smlnum, abs(A[(kwtop, kwtop)]) * ulp) {
			// ns = 0;
			// nd = 1;
			if kwtop > ilo {
				A[(kwtop, kwtop - 1)] = zero();
			}
		}
	}

	// * Transform window to real schur form
	let mut qc = QC.rb_mut().get_mut(..jw, ..jw);
	let mut zc = ZC.rb_mut().get_mut(..jw, ..jw);

	for mut m in [qc.rb_mut(), zc.rb_mut()] {
		m.fill(zero());
		m.diagonal_mut().column_vector_mut().fill(one());
	}

	hessenberg_to_qz_blocked(
		0,
		jw - 1,
		A.rb_mut().submatrix_mut(kwtop, kwtop, jw, jw),
		B.rb_mut().submatrix_mut(kwtop, kwtop, jw, jw),
		Some(qc.rb_mut()),
		Some(zc.rb_mut()),
		alphar.rb_mut(),
		alphai.rb_mut(),
		beta.rb_mut(),
		false,
		par,
		params,
		stack,
	);

	let mut kwbot;

	// * Deflation detection loop
	if kwtop == ilo || s == zero() {
		kwbot = kwtop.wrapping_sub(1);
	} else {
		kwbot = ihi;
		let mut k = 0;
		let mut k2 = 0;

		while k < jw {
			let mut bulge = false;
			if kwbot + 1 - kwtop >= 2 {
				bulge = A[(kwbot, kwbot - 1)] != zero();
			}

			if bulge {
				// * Try to deflate complex conjugate eigenvalue pair
				let mut temp = abs(A[(kwbot, kwbot)]) + sqrt(abs(A[(kwbot, kwbot - 1)])) * sqrt(abs(A[(kwbot - 1, kwbot)]));
				if temp == zero() {
					temp = abs(s);
				}
				if max(abs(s * qc[(0, kwbot - kwtop - 1)]), abs(s * qc[(0, kwbot - kwtop)])) <= max(smlnum, ulp * temp) {
					// * Deflatable
					kwbot -= 2;
				} else {
					// * Not deflatable, move out of the way
					ifst = kwbot - kwtop;
					ilst = k2;

					reorder_qz(
						A.rb_mut().submatrix_mut(kwtop, kwtop, jw, jw),
						B.rb_mut().submatrix_mut(kwtop, kwtop, jw, jw),
						Some(qc.rb_mut()),
						Some(zc.rb_mut()),
						ifst,
						ilst,
						stack,
					);
					k2 += 2;
				}
				k += 2;
			} else {
				// * Try to deflate eigenvalue
				let mut temp = abs(A[(kwbot, kwbot)]);
				if temp == zero() {
					temp = abs(s);
				}
				if abs(s * qc[(0, kwbot - kwtop)]) < max(smlnum, ulp * temp) {
					// * Deflatable
					kwbot -= 1;
				} else {
					// * Not deflatable, move out of the way
					ifst = kwbot - kwtop;
					ilst = k2;
					reorder_qz(
						A.rb_mut().submatrix_mut(kwtop, kwtop, jw, jw),
						B.rb_mut().submatrix_mut(kwtop, kwtop, jw, jw),
						Some(qc.rb_mut()),
						Some(zc.rb_mut()),
						ifst,
						ilst,
						stack,
					);
					k2 += 1;
				}
				k += 1;
			}
		}
	}

	// * Store eigenvalues
	let nd = ihi - kwbot;
	let ns = jw - nd;
	let mut k = kwtop;
	while k <= ihi {
		let mut bulge = false;
		if k < ihi {
			if A[(k + 1, k)] != zero() {
				bulge = true
			}
		}
		if bulge {
			let (scale1, scale2, wr1, wr2, wi) = generalized_eigval_2x2(
				(copy(A[(k, k)]), copy(A[(k, k + 1)]), copy(A[(k + 1, k)]), copy(A[(k + 1, k + 1)])),
				(copy(B[(k, k)]), copy(B[(k, k + 1)]), copy(B[(k + 1, k)]), copy(B[(k + 1, k + 1)])),
			);
			beta[k] = copy(scale1);
			beta[k + 1] = copy(scale2);
			alphar[k] = copy(wr1);
			alphar[k + 1] = copy(wr2);
			alphai[k] = copy(wi);
			alphai[k + 1] = -wi;
			k += 2;
		} else {
			alphar[k] = copy(A[(k, k)]);
			alphai[k] = zero();
			beta[k] = copy(B[(k, k)]);
			k += 1;
		}
	}

	if kwtop != ilo && s != zero() {
		// * Reflect spike back, this will create optimally packed bulges
		let scale = copy(A[(kwtop, kwtop - 1)]);
		zip!(A.rb_mut().get_mut(kwtop..kwbot + 1, kwtop - 1), qc.rb().get(0, ..jw - nd).transpose()).for_each(|unzip!(dst, src)| *dst = *src * scale);

		for k in (kwtop..kwbot).rev() {
			let (c1, s1, temp) = make_givens(copy(A[(k, kwtop - 1)]), copy(A[(k + 1, kwtop - 1)]));
			A[(k, kwtop - 1)] = temp;
			A[(k + 1, kwtop - 1)] = zero();

			let k2 = Ord::max(kwtop, k - 1);

			rot_rows(copy(c1), copy(s1), A.rb_mut().get_mut(.., k2..ihi + 1), k, k + 1);
			rot_rows(copy(c1), copy(s1), B.rb_mut().get_mut(.., k - 1..ihi + 1), k, k + 1);
			rot_cols(copy(c1), copy(s1), qc.rb_mut(), k - kwtop, k + 1 - kwtop);
		}

		// * Chase bulges down
		let istartm = kwtop;
		let istopm = ihi;
		let mut k = kwbot;
		while k > kwtop {
			k -= 1;

			if k >= kwtop + 1 && A[(k + 1, k - 1)] != zero() {
				// * Move double pole block down and remove it
				for k2 in k - 1..kwbot - 1 {
					chase_bulge_2x2(
						k2,
						kwtop,
						kwtop + jw - 1,
						kwbot,
						A.rb_mut(),
						B.rb_mut(),
						kwtop,
						Some(qc.rb_mut()),
						kwtop,
						Some(zc.rb_mut()),
					);
				}
				k -= 1;
			} else {
				// * k points to single shift
				for k2 in k..kwbot - 1 {
					let (c1, s1, temp) = make_givens(copy(B[(k2 + 1, k2 + 1)]), copy(B[(k2 + 1, k2)]));
					B[(k2 + 1, k2 + 1)] = temp;
					B[(k2 + 1, k2)] = zero();

					rot_cols(copy(c1), copy(s1), A.rb_mut().subrows_mut(istartm, k2 + 3 - istartm), k2 + 1, k2);
					rot_cols(copy(c1), copy(s1), B.rb_mut().subrows_mut(istartm, k2 + 1 - istartm), k2 + 1, k2);
					rot_cols(copy(c1), copy(s1), zc.rb_mut(), k2 + 1 - kwtop, k2 - kwtop);

					let (c1, s1, temp) = make_givens(copy(A[(k2 + 1, k2)]), copy(A[(k2 + 2, k2)]));
					A[(k2 + 1, k2)] = temp;
					A[(k2 + 2, k2)] = zero();
					rot_rows(copy(c1), copy(s1), A.rb_mut().subcols_mut(k2 + 1, istopm - k2), k2 + 1, k2 + 2);
					rot_rows(copy(c1), copy(s1), B.rb_mut().subcols_mut(k2 + 1, istopm - k2), k2 + 1, k2 + 2);
					rot_cols(copy(c1), copy(s1), qc.rb_mut(), k2 + 1 - kwtop, k2 + 2 - kwtop);
				}

				// * Remove the shift
				let (c1, s1, temp) = make_givens(copy(B[(kwbot, kwbot)]), copy(B[(kwbot, kwbot - 1)]));
				B[(kwbot, kwbot)] = temp;
				B[(kwbot, kwbot - 1)] = zero();
				rot_cols(copy(c1), copy(s1), B.rb_mut().subrows_mut(istartm, kwbot - istartm), kwbot, kwbot - 1);
				rot_cols(copy(c1), copy(s1), A.rb_mut().subrows_mut(istartm, kwbot + 1 - istartm), kwbot, kwbot - 1);
				rot_cols(copy(c1), copy(s1), zc.rb_mut(), kwbot - kwtop, kwbot - 1 - kwtop);
			}
		}
	}

	// * Apply Qc and Zc to rest of the matrix
	let istartm;
	let istopm;
	if !eigvals_only {
		istartm = 0;
		istopm = n - 1;
	} else {
		istartm = ilo;
		istopm = ihi;
	}

	if istopm - ihi > 0 {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(jw, istopm - ihi, stack) };
		let mut work = work.as_mat_mut();
		for M in [A.rb_mut(), B.rb_mut()] {
			let M: MatMut<'_, T> = M;
			let mut M = M.submatrix_mut(kwtop, ihi + 1, jw, istopm - ihi);
			matmul(work.rb_mut(), Accum::Replace, qc.rb().adjoint(), M.rb(), one(), par);
			M.copy_from(&work);
		}
	}
	if let Some(mut Q) = Q.rb_mut() {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(n, jw, stack) };
		let mut work = work.as_mat_mut();
		let mut M = Q.rb_mut().subcols_mut(kwtop, jw);
		matmul(work.rb_mut(), Accum::Replace, M.rb(), qc.rb(), one(), par);
		M.copy_from(&work);
	}

	if kwtop - istartm > 0 {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(kwtop - istartm, jw, stack) };
		let mut work = work.as_mat_mut();
		for M in [A.rb_mut(), B.rb_mut()] {
			let M: MatMut<'_, T> = M;
			let mut M = M.submatrix_mut(istartm, kwtop, kwtop - istartm, jw);
			matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);
			M.copy_from(&work);
		}
	}
	if let Some(mut Z) = Z.rb_mut() {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(n, jw, stack) };
		let mut work = work.as_mat_mut();
		let mut M = Z.rb_mut().subcols_mut(kwtop, jw);
		matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);
		M.copy_from(&work);
	}

	(ns, nd)
}

#[math]
fn swap_qz<T: RealField>(
	mut A: MatMut<'_, T>,
	mut B: MatMut<'_, T>,
	mut Q: Option<MatMut<'_, T>>,
	mut Z: Option<MatMut<'_, T>>,
	start: usize,
	n1: usize,
	n2: usize,
	stack: &mut MemStack,
) {
	let n = A.nrows();
	let m = n1 + n2;
	let zero = zero::<T>;
	let one = one::<T>;

	if n1 == 1 && n2 == 1 {
		// * CASE 1: Swap 1-by-1 and 1-by-1 blocks.
		// *
		// * Compute orthogonal QL and RQ that swap 1-by-1 and 1-by-1 blocks
		// * using Givens rotations and perform the swap tentatively.

		let mut s11 = copy(A[(start, start)]);
		let s12 = copy(A[(start, start + 1)]);
		let mut s21 = zero();
		let s22 = copy(A[(start + 1, start + 1)]);

		let mut t11 = copy(B[(start, start)]);
		let t12 = copy(B[(start, start + 1)]);
		let mut t21 = zero();
		let t22 = copy(B[(start + 1, start + 1)]);

		let f = s22 * t11 - t22 * s11;
		let g = s22 * t12 - t22 * s12;
		let sa = abs(s22) * abs(t11);
		let sb = abs(s11) * abs(t22);

		let (ir12, ir11, _) = make_givens(f, g);
		let ir21 = -ir12;

		s11 = s11 * ir11 + s12 * ir21;
		s21 = s21 * ir11 + s22 * ir21;
		t11 = t11 * ir11 + t12 * ir21;
		t21 = t21 * ir11 + t22 * ir21;

		let (li11, li21, _) = if sa >= sb { make_givens(s11, s21) } else { make_givens(t11, t21) };

		rot_cols(copy(ir11), copy(ir21), A.rb_mut().subrows_mut(0, start + 2), start, start + 1);
		rot_rows(copy(li11), copy(li21), A.rb_mut().subcols_mut(start, n - start), start, start + 1);
		rot_cols(copy(ir11), copy(ir21), B.rb_mut().subrows_mut(0, start + 2), start, start + 1);
		rot_rows(copy(li11), copy(li21), B.rb_mut().subcols_mut(start, n - start), start, start + 1);
		A[(start + 1, start)] = zero();
		B[(start + 1, start)] = zero();

		if let Some(mut Z) = Z.rb_mut() {
			rot_cols(copy(ir11), copy(ir21), Z.rb_mut(), start, start + 1);
		}
		if let Some(mut Q) = Q.rb_mut() {
			rot_cols(copy(li11), copy(li21), Q.rb_mut(), start, start + 1);
		}
	} else {
		// * CASE 2: Swap n1-by-n1 and n2-by-n2 blocks
		// * Solve the generalized Sylvester equation
		// * S11 * R - L * S22 = S12
		// * T11 * R - L * T22 = T12
		// * for R and L. Solutions in LI and IR.
		let S = A.rb().submatrix(start, start, m, m);
		let T = B.rb().submatrix(start, start, m, m);

		// S12/T12 is n1-by-n2
		let (S11, S12, _, S22) = S.split_at(n1, n1);
		let (T11, T12, _, T22) = T.split_at(n1, n1);

		let (mut L, stack) = linalg::temp_mat_zeroed::<T, _, _>(n1 + n2, n2, stack);
		let mut L = L.as_mat_mut();
		let (mut R, stack) = linalg::temp_mat_zeroed::<T, _, _>(n1, n1 + n2, stack);
		let mut R = R.as_mat_mut();

		L.rb_mut().get_mut(..n1, ..).copy_from(T12);
		L.rb_mut().get_mut(n1.., ..).diagonal_mut().column_vector_mut().fill(-one());

		R.rb_mut().get_mut(.., n1..).copy_from(S12);
		R.rb_mut().get_mut(.., ..n1).diagonal_mut().column_vector_mut().fill(one());

		{
			let mut L = L.rb_mut().get_mut(..n1, ..);
			let mut R = R.rb_mut().get_mut(.., n1..);

			solve_sylvester_single_block(S11, S22, R.rb_mut(), T11, T22, L.rb_mut(), stack);
		}

		let (mut L_householder, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(1, n2, stack) };
		let mut L_householder = L_householder.as_mat_mut();
		let (mut R_householder, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(1, n1, stack) };
		let mut R_householder = R_householder.as_mat_mut();

		// compute Q' s.t.
		// [ L ]        [ TL ]
		// [ I ] = Q' * [  0 ]
		linalg::qr::no_pivoting::factor::qr_in_place(L.rb_mut(), L_householder.rb_mut(), Par::Seq, stack, Default::default());

		// compute Z' s.t.
		// [ R.T.rev ]        [ TR ]
		// [       I ] = Z' * [  0 ]
		//
		// i.e.
		//               [ R.T.revrow ]   [ TR ]
		// Z'.T.revcol * [   I.revrow ] = [  0 ]
		//
		// i.e.
		//            [   I ]   [         0 ]
		// Z'.T.rev * [ R.T ] = [ TR.revrow ]
		//
		// i.e.
		// [ I  R ] * Z'.rev = [ 0  TR.revrow.T ]
		// [ I  R ] * Z'.rev = [ 0  TR.revrow.T ]
		let mut RT_rev = R.rb_mut().transpose_mut().reverse_rows_and_cols_mut();
		linalg::qr::no_pivoting::factor::qr_in_place(
			RT_rev.rb_mut(),
			R_householder.rb_mut().reverse_cols_mut(),
			Par::Seq,
			stack,
			Default::default(),
		);

		for j in 0..n2 {
			let mut l = L.rb_mut().col_mut(j);
			l.rb_mut().get_mut(..j).fill(zero());
			l[j] = one();
		}
		for j in 0..n1 {
			let mut r = RT_rev.rb_mut().col_mut(j);
			r.rb_mut().get_mut(..j).fill(zero());
			r[j] = one();
		}

		let L = L.rb();
		let R = R.rb();

		for mut M in [A.rb_mut(), B.rb_mut()] {
			for mut m in M.rb_mut().get_mut(start..start + m, start..).col_iter_mut() {
				for j in 0..n2 {
					let l = L.col(j);
					let tau = copy(L_householder[(0, j)]);

					let dot = (l.transpose() * m.rb()) * -recip(tau);
					zip!(m.rb_mut(), l).for_each(|unzip!(dst, src)| *dst = *dst + dot * *src);
				}
			}
			for mut m in M.rb_mut().get_mut(..start + m, start..start + m).row_iter_mut() {
				for j in (0..n1).rev() {
					let r = R.row(j);
					let tau = copy(R_householder[(0, j)]);

					let dot = (m.rb() * r.transpose()) * -recip(tau);
					zip!(m.rb_mut(), r).for_each(|unzip!(dst, src)| *dst = *dst + dot * *src);
				}
			}
		}

		if let Some(Q) = Q.rb_mut() {
			let mut Q = Q.get_mut(.., start..start + m);

			for j in 0..n2 {
				let l = L.col(j);
				let tau = copy(L_householder[(0, j)]);

				let (mut dot, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(Q.nrows(), 1, stack) };
				let mut dot = dot.as_mat_mut().col_mut(0);

				matmul(dot.rb_mut(), Accum::Replace, Q.rb(), l, one(), Par::Seq);
				matmul(Q.rb_mut(), Accum::Add, dot.rb(), l.transpose(), -recip(tau), Par::Seq);
			}
		}

		if let Some(Z) = Z.rb_mut() {
			let mut Z = Z.get_mut(.., start..start + m);

			for j in (0..n1).rev() {
				let r = R.row(j).transpose();
				let tau = copy(R_householder[(0, j)]);

				let (mut dot, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(Z.nrows(), 1, stack) };
				let mut dot = dot.as_mat_mut().col_mut(0);

				matmul(dot.rb_mut(), Accum::Replace, Z.rb(), r, one(), Par::Seq);
				matmul(Z.rb_mut(), Accum::Add, dot.rb(), r.transpose(), -recip(tau), Par::Seq);
			}
		}

		{
			let mut start = start;
			for nx in [n2, n1] {
				if nx == 2 {
					// triangularize the B section

					let mut A = A.rb_mut().get_mut(start..start + 2, start..);
					let mut B = B.rb_mut().get_mut(start..start + 2, start..);

					let (c, s, _) = make_givens(copy(B[(0, 0)]), copy(B[(1, 0)]));
					rot_rows(copy(c), copy(s), B.rb_mut(), 0, 1);
					rot_rows(copy(c), copy(s), A.rb_mut(), 0, 1);

					if let Some(Q) = Q.rb_mut() {
						rot_cols(copy(c), copy(s), Q, start, start + 1);
					}
				}

				start += nx;
			}
		}

		for j in 0..m {
			for i in j + 1..m {
				B[(start + i, start + j)] = zero();
			}
		}
		A.rb_mut().get_mut(start + n2..start + n2 + n1, start..start + n2).fill(zero());
	}
}

// B. Kagstrom; A Direct Method for Reordering Eigenvalues in the
// Generalized Real Schur Form of a Regular Matrix Pair (A, B), in
// M.S. Moonen et al (eds), Linear Algebra for Large Scale and
// Real-Time Applications, Kluwer Academic Publ. 1993, pp 195-218.
#[math]
fn reorder_qz<T: RealField>(
	mut A: MatMut<'_, T>,
	mut B: MatMut<'_, T>,
	mut Q: Option<MatMut<'_, T>>,
	mut Z: Option<MatMut<'_, T>>,
	mut ifst: usize,
	mut ilst: usize,
	stack: &mut MemStack,
) {
	let zero = zero::<T>;
	let n = A.nrows();

	// *
	// * Determine the first row of the specified block and find out
	// * if it is 1-by-1 or 2-by-2.
	// *
	if ifst > 0 {
		if A[(ifst, ifst - 1)] != zero() {
			ifst -= 1;
		}
	}
	let mut nbf = 1;
	if ifst < n - 1 {
		if A[(ifst + 1, ifst)] != zero() {
			nbf = 2;
		}
	}

	// *
	// * Determine the first row of the final block
	// * and find out if it is 1-by-1 or 2-by-2.
	// *
	if ilst > 0 {
		if A[(ilst, ilst - 1)] != zero() {
			ilst = ilst - 1;
		}
	}
	let mut nbl = 1;
	if ilst < n - 1 {
		if A[(ilst + 1, ilst)] != zero() {
			nbl = 2
		}
	}
	if ifst == ilst {
		return;
	}

	if ifst < ilst {
		if nbf == 2 && nbl == 1 {
			ilst -= 1;
		}
		if nbf == 1 && nbl == 2 {
			ilst += 1;
		}

		let mut here = ifst;
		// * Swap with next one below.
		loop {
			if nbf == 1 || nbf == 2 {
				// * Current block either 1-by-1 or 2-by-2.
				let mut nbnext = 1;
				if here + nbf + 1 <= n - 1 {
					if A[(here + nbf + 1, here + nbf)] != zero() {
						nbnext = 2;
					}
				}

				swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here, nbf, nbnext, stack);

				here += nbnext;
				// * Test if 2-by-2 block breaks into two 1-by-1 blocks.
				if nbf == 2 {
					if A[(here + 1, here)] == zero() {
						nbf = 3;
					}
				}
			} else {
				// * Current block consists of two 1-by-1 blocks, each of which
				// * must be swap individually.
				let mut nbnext = 1;
				if here + 3 <= n - 1 {
					if A[(here + 3, here + 2)] != zero() {
						nbnext = 2;
					}
				}

				swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here + 1, 1, nbnext, stack);

				if nbnext == 1 {
					// * Swap two 1-by-1 blocks.
					swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here, 1, 1, stack);
					here += 1;
				} else {
					// * Recompute NBNEXT in case of 2-by-2 split.
					if A[(here + 2, here + 1)] == zero() {
						nbnext = 1;
					}

					if nbnext == 2 {
						// * 2-by-2 block did not split.
						swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here, 1, nbnext, stack);
						here += 2;
					} else {
						// * 2-by-2 block did split.
						swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here, 1, 1, stack);
						here += 1;
						swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here, 1, 1, stack);
						here += 1;
					}
				}
			}

			if here >= ilst {
				break;
			}
		}
	} else {
		let mut here = ifst;

		loop {
			// * Swap with next one below.
			if nbf == 1 || nbf == 2 {
				// * Current block either 1-by-1 or 2-by-2.
				let mut nbnext = 1;
				if here >= 2 {
					if A[(here - 1, here - 2)] != zero() {
						nbnext = 2;
					}
				}

				swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here - nbnext, nbnext, nbf, stack);
				here -= nbnext;

				// * Test if 2-by-2 block breaks into two 1-by-1 blocks.
				if nbf == 2 {
					if A[(here + 1, here)] == zero() {
						nbf = 3;
					}
				}
			} else {
				// * Current block consists of two 1-by-1 blocks, each of which
				// * must be swap individually.
				let mut nbnext = 1;
				if here >= 2 {
					if A[(here - 1, here - 2)] != zero() {
						nbnext = 2;
					}
				}

				swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here - nbnext, nbnext, 1, stack);
				if nbnext == 1 {
					// * Swap two 1-by-1 blocks.
					swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here, nbnext, 1, stack);
					here -= 1;
				} else {
					// * Recompute NBNEXT in case of 2-by-2 split.
					if A[(here, here - 1)] == zero() {
						nbnext = 1;
					}
					if nbnext == 2 {
						// * 2-by-2 block did not split.
						swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here - 1, 2, 1, stack);
						here -= 2;
					} else {
						// * 2-by-2 block did split.
						swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here, 1, 1, stack);
						here -= 1;
						swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here, 1, 1, stack);
						here -= 1;
					}
				}
			}

			if here <= ilst {
				break;
			}
		}
	}
}

#[math]
fn multishift_sweep<T: RealField>(
	ilo: usize,
	ihi: usize,
	mut sr: ColMut<'_, T>,
	mut si: ColMut<'_, T>,
	mut ss: ColMut<'_, T>,
	mut A: MatMut<'_, T>,
	mut B: MatMut<'_, T>,
	mut Q: Option<MatMut<'_, T>>,
	mut Z: Option<MatMut<'_, T>>,
	mut QC: MatMut<'_, T>,
	mut ZC: MatMut<'_, T>,
	eigvals_only: bool,
	par: Par,
	stack: &mut MemStack,
) {
	let zero = zero::<T>;
	let one = one::<T>;

	let n = A.nrows();
	let nshifts = sr.nrows();

	if nshifts < 2 {
		return;
	}
	if ilo >= ihi {
		return;
	}

	let istartm;
	let istopm;

	if !eigvals_only {
		istartm = 0;
		istopm = n - 1;
	} else {
		istartm = ilo;
		istopm = ihi;
	}

	let nblock_desired = QC.ncols();

	// * Shuffle shifts into pairs of real shifts and pairs
	// * of complex conjugate shifts assuming complex
	// * conjugate shifts are already adjacent to one
	// * another
	for i in (0..nshifts - 2).step_by(2) {
		if si[i] != -si[i + 1] {
			for mut v in [sr.rb_mut(), si.rb_mut(), ss.rb_mut()] {
				let swap = copy(v[i]);
				v[i] = copy(v[i + 1]);
				v[i + 1] = copy(v[i + 2]);
				v[i + 2] = swap;
			}
		}
	}

	// * NSHFTS is supposed to be even, but if it is odd,
	// * then simply reduce it by one.  The shuffle above
	// * ensures that the drop shift is real and that
	// * the remaining shifts are paired.
	let ns = nshifts / 2 * 2;
	let npos = Ord::max(nblock_desired - ns, 1);

	let mut qc = QC.rb_mut().get_mut(..ns + 1, ..ns + 1);
	let mut zc = ZC.rb_mut().get_mut(..ns, ..ns);

	let mut v_storage = [zero(), zero(), zero()];
	let mut v = ColMut::from_slice_mut(&mut v_storage);

	for mut m in [qc.rb_mut(), zc.rb_mut()] {
		m.fill(zero());
		m.diagonal_mut().column_vector_mut().fill(one());
	}

	for i in (0..ns).step_by(2) {
		laqz1(
			A.rb().submatrix(ilo, ilo, 3, 3),
			B.rb().submatrix(ilo, ilo, 3, 3),
			copy(sr[i]),
			copy(sr[i + 1]),
			copy(si[i]),
			copy(ss[i]),
			copy(ss[i + 1]),
			v.rb_mut(),
		);

		let v1 = copy(v[0]);
		let mut v2 = copy(v[1]);
		let v3 = copy(v[2]);

		let (c1, s1, temp) = make_givens(v2, v3);
		v2 = temp;
		let (c2, s2, _) = make_givens(v1, v2);

		rot_rows(copy(c1), copy(s1), A.rb_mut().subcols_mut(ilo, ns), ilo + 1, ilo + 2);
		rot_rows(copy(c2), copy(s2), A.rb_mut().subcols_mut(ilo, ns), ilo + 0, ilo + 1);
		rot_rows(copy(c1), copy(s1), B.rb_mut().subcols_mut(ilo, ns), ilo + 1, ilo + 2);
		rot_rows(copy(c2), copy(s2), B.rb_mut().subcols_mut(ilo, ns), ilo + 0, ilo + 1);
		rot_cols(copy(c1), copy(s1), qc.rb_mut(), 1, 2);
		rot_cols(copy(c2), copy(s2), qc.rb_mut(), 0, 1);

		let i = i + 1;
		for j in 0..ns - i - 1 {
			chase_bulge_2x2(
				j,
				0,
				ns - 1,
				ihi - ilo,
				A.rb_mut().get_mut(ilo.., ilo..),
				B.rb_mut().get_mut(ilo.., ilo..),
				0,
				Some(qc.rb_mut()),
				0,
				Some(zc.rb_mut()),
			);
		}
	}
	// * Update the rest of the pencil

	// * Update A[(ilo:ilo+ns,ilo+ns:istopm)] and B[(ilo:ilo+ns,ilo+ns:istopm)]
	// * from the left with Qc(1:ns+1,1:ns+1)'
	let sheight = ns + 1;
	let swidth = (istopm + 1).saturating_sub(ilo + ns);

	if swidth > 0 {
		{
			let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(sheight, swidth, stack) };
			let mut work = work.as_mat_mut();
			for M in [A.rb_mut(), B.rb_mut()] {
				let mut M = M.submatrix_mut(ilo, ilo + ns, sheight, swidth);
				matmul(work.rb_mut(), Accum::Replace, qc.rb().adjoint(), M.rb(), one(), par);
				M.copy_from(&work);
			}
		}
	}
	if let Some(mut Q) = Q.rb_mut() {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(n, sheight, stack) };
		let mut work = work.as_mat_mut();
		let mut M = Q.rb_mut().get_mut(.., ilo..ilo + sheight);
		matmul(work.rb_mut(), Accum::Replace, M.rb(), qc.rb(), one(), par);
		M.copy_from(&work);
	}

	// * Update A[(istartm:ilo-1,ilo:ilo+ns-1)] and B[(istartm:ilo-1,ilo:ilo+ns-1)]
	// * from the right with Zc(1:ns,1:ns),
	let sheight = ilo.saturating_sub(istartm);
	let swidth = ns;
	if sheight > 0 {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(sheight, swidth, stack) };
		let mut work = work.as_mat_mut();
		for M in [A.rb_mut(), B.rb_mut()] {
			let mut M = M.submatrix_mut(istartm, ilo, sheight, swidth);
			matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);
			M.copy_from(&work);
		}
	}

	if let Some(mut Z) = Z.rb_mut() {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(n, swidth, stack) };
		let mut work = work.as_mat_mut();
		let mut M = Z.rb_mut().get_mut(.., ilo..ilo + swidth);
		matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);
		M.copy_from(&work);
	}

	// * The following block chases the shifts down to the bottom
	// * right block. If possible, a shift is moved down npos
	// * positions at a time
	let mut k = ilo;
	while k < ihi - ns {
		let np = Ord::min(ihi - ns - k, npos);
		// Size of the near-the-diagonal block
		let nblock = ns + np;
		// istartb points to the first row we will be updating
		let istartb = k + 1;
		// istopb points to the last column we will be updating
		let istopb = k + nblock - 1;

		let mut qc = QC.rb_mut().get_mut(..nblock, ..nblock);
		let mut zc = ZC.rb_mut().get_mut(..nblock, ..nblock);

		for mut m in [qc.rb_mut(), zc.rb_mut()] {
			m.fill(zero());
			m.diagonal_mut().column_vector_mut().fill(one());
		}

		for i in (0..ns).rev().step_by(2) {
			for j in 0..np {
				// * Move down the block with index k+i+j-1, updating
				// * the (ns+np x ns+np) block:
				// * (k:k+ns+np,k:k+ns+np-1)
				chase_bulge_2x2(
					k + i + j - 1,
					istartb,
					istopb,
					ihi,
					A.rb_mut(),
					B.rb_mut(),
					k + 1,
					Some(qc.rb_mut()),
					k,
					Some(zc.rb_mut()),
				);
			}
		}

		// * Update rest of the pencil

		// * Update A[(k+1:k+ns+np, k+ns+np:istopm)] and
		// * B[(k+1:k+ns+np, k+ns+np:istopm)]
		// * from the left with Qc(1:ns+np,1:ns+np)'
		let sheight = ns + np;
		let swidth = (istopm + 1).saturating_sub(k + ns + np);
		if swidth > 0 {
			let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(sheight, swidth, stack) };
			let mut work = work.as_mat_mut();
			for M in [A.rb_mut(), B.rb_mut()] {
				let mut M = M.get_mut(k + 1..k + 1 + sheight, k + ns + np..k + ns + np + swidth);
				matmul(work.rb_mut(), Accum::Replace, qc.rb().adjoint(), M.rb(), one(), par);
				M.copy_from(&work);
			}
		}
		if let Some(mut Q) = Q.rb_mut() {
			let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(n, sheight, stack) };
			let mut work = work.as_mat_mut();
			let mut M = Q.rb_mut().get_mut(.., k + 1..k + 1 + sheight);
			matmul(work.rb_mut(), Accum::Replace, M.rb(), qc.rb(), one(), par);
			M.copy_from(&work);
		}

		// * Update A[(istartm:k,k:k+ns+npos-1)] and B[(istartm:k,k:k+ns+npos-1)]
		// * from the right with Zc(1:ns+np,1:ns+np)
		let sheight = (k + 1).saturating_sub(istartm);
		let swidth = nblock;
		if sheight > 0 {
			let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(sheight, swidth, stack) };
			let mut work = work.as_mat_mut();
			for M in [A.rb_mut(), B.rb_mut()] {
				let mut M = M.get_mut(istartm..istartm + sheight, k..k + swidth);
				matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);
				M.copy_from(&work);
			}
		}
		if let Some(mut Z) = Z.rb_mut() {
			let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(n, swidth, stack) };
			let mut work = work.as_mat_mut();
			let mut M = Z.rb_mut().get_mut(.., k..k + swidth);
			matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);
			M.copy_from(&work);
		}

		k += np;
	}

	// * The following block removes the shifts from the bottom right corner
	// * one by one. Updates are initially applied to A[(ihi-ns+1:ihi,ihi-ns:ihi)].
	let mut qc = QC.rb_mut().get_mut(..ns, ..ns);
	let mut zc = ZC.rb_mut().get_mut(..ns + 1, ..ns + 1);

	// * istartb points to the first row we will be updating
	let istartb = ihi + 1 - ns;
	// * istopb points to the last column we will be updating
	let istopb = ihi;

	for mut m in [qc.rb_mut(), zc.rb_mut()] {
		m.fill(zero());
		m.diagonal_mut().column_vector_mut().fill(one());
	}

	for i in (0..ns).step_by(2) {
		// * Chase the shift down to the bottom right corner
		let i = i + 1;
		for ishift in ihi - i - 1..ihi - 1 {
			chase_bulge_2x2(
				ishift,
				istartb,
				istopb,
				ihi,
				A.rb_mut(),
				B.rb_mut(),
				ihi + 1 - ns,
				Some(qc.rb_mut()),
				ihi - ns,
				Some(zc.rb_mut()),
			);
		}
	}

	// * Update rest of the pencil
	//
	// * Update A[(ihi-ns+1:ihi, ihi+1:istopm)]
	// * from the left with Qc(1:ns,1:ns)'
	let sheight = ns;
	let swidth = istopm.saturating_sub(ihi);
	if swidth > 0 {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(sheight, swidth, stack) };
		let mut work = work.as_mat_mut();
		for M in [A.rb_mut(), B.rb_mut()] {
			let mut M = M.submatrix_mut(ihi + 1 - ns, ihi + 1, sheight, swidth);
			matmul(work.rb_mut(), Accum::Replace, qc.rb().adjoint(), M.rb(), one(), par);
			M.copy_from(&work);
		}
	}
	if let Some(mut Q) = Q.rb_mut() {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(n, sheight, stack) };
		let mut work = work.as_mat_mut();
		let mut M = Q.rb_mut().get_mut(.., ihi - ns + 1..ihi + 1);
		matmul(work.rb_mut(), Accum::Replace, M.rb(), qc.rb(), one(), par);
		M.copy_from(&work);
	}

	// * Update A[(istartm:ihi-ns,ihi-ns:ihi)]
	// * from the right with Zc(1:ns+1,1:ns+1)
	let sheight = (ihi + 1).saturating_sub(ns + istartm);
	let swidth = ns + 1;
	if sheight > 0 {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(sheight, swidth, stack) };
		let mut work = work.as_mat_mut();
		for M in [A.rb_mut(), B.rb_mut()] {
			let mut M = M.submatrix_mut(istartm, ihi - ns, sheight, swidth);
			matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);
			M.copy_from(&work);
		}
	}
	if let Some(mut Z) = Z.rb_mut() {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(n, swidth, stack) };
		let mut work = work.as_mat_mut();
		let mut M = Z.rb_mut().get_mut(.., ihi - ns..ihi + 1);
		matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);
		M.copy_from(&work);
	}
}

#[cfg(test)]
mod tests {
	use super::super::gen_hessenberg::{GeneralizedHessenbergParams, generalized_hessenberg, generalized_hessenberg_scratch};
	use super::*;
	use crate::utils::approx::*;
	use crate::{linalg, stats};
	use dyn_stack::MemBuffer;
	use equator::assert;
	use stats::prelude::*;

	fn make_pair(rng: &mut dyn RngCore, ns: &[usize]) -> (Mat<f64>, Mat<f64>) {
		let dist = StandardNormal;

		let n = ns.iter().sum::<usize>();
		let mut A = Mat::zeros(n, n);
		let mut B = Mat::zeros(n, n);

		let mut pos = 0;
		for &nx in ns {
			equator::assert!(any(nx == 1, nx == 2));

			for j in 0..nx {
				for i in 0..pos + nx {
					A[(i, pos + j)] = dist.sample(rng);
				}
			}
			for j in 0..nx {
				for i in 0..pos + j + 1 {
					B[(i, pos + j)] = dist.sample(rng);
				}
			}

			pos += nx;
		}

		(A, B)
	}

	fn geigval_2x2(A: MatRef<'_, f64>, B: MatRef<'_, f64>) -> (f64, f64, f64, f64) {
		let (s1, s2, wr1, wr2, wi) =
			generalized_eigval_2x2((A[(0, 0)], A[(0, 1)], A[(1, 0)], A[(1, 1)]), (B[(0, 0)], B[(0, 1)], B[(1, 0)], B[(1, 1)]));

		let mut wr1 = wr1 / s1;
		let mut wr2 = wr2 / s2;
		let wi = wi / s1;
		if wr2 < wr1 {
			core::mem::swap(&mut wr1, &mut wr2);
		}
		(wr1, wi, wr2, -wi)
	}

	fn geigval_1x1(A: MatRef<'_, f64>, B: MatRef<'_, f64>) -> f64 {
		A[(0, 0)] / B[(0, 0)]
	}

	#[test]
	fn test_swap_qz_random() {
		let approx_eq = crate::utils::approx::ApproxEq::<f64> {
			abs_tol: 1e-10,
			rel_tol: 1e-10,
		};

		let rng = &mut StdRng::seed_from_u64(0);

		for (n1, n2) in [(1, 1), (1, 2), (2, 1), (2, 2)] {
			for (ns, j) in [
				([n1, n2].as_slice(), 0),
				([2, n1, n2, 2].as_slice(), 2),
				([n1, n2, 2].as_slice(), 0),
				([2, n1, n2].as_slice(), 2),
				([1, n1, n2, 1].as_slice(), 1),
				([n1, n2, 1].as_slice(), 0),
				([1, n1, n2].as_slice(), 1),
			] {
				let n = ns.iter().sum::<usize>();

				for _ in 0..10 {
					let (A, B) = make_pair(rng, ns);
					let mut A_swap = A.clone();
					let mut B_swap = B.clone();
					let mut Q = Mat::identity(n, n);
					let mut Z = Mat::identity(n, n);

					swap_qz(
						A_swap.as_mut(),
						B_swap.as_mut(),
						Some(Q.as_mut()),
						Some(Z.as_mut()),
						j,
						n1,
						n2,
						MemStack::new(&mut [core::mem::MaybeUninit::new(0u8); 16 * 1024]),
					);

					let A1 = A.get(j..j + n1, j..j + n1);
					let A2 = A.get(j + n1..j + n1 + n2, j + n1..j + n1 + n2);
					let B1 = B.get(j..j + n1, j..j + n1);
					let B2 = B.get(j + n1..j + n1 + n2, j + n1..j + n1 + n2);

					let A1_swap = A_swap.get(j..j + n2, j..j + n2);
					let A2_swap = A_swap.get(j + n2..j + n2 + n1, j + n2..j + n2 + n1);
					let B1_swap = B_swap.get(j..j + n2, j..j + n2);
					let B2_swap = B_swap.get(j + n2..j + n2 + n1, j + n2..j + n2 + n1);

					if n1 == 1 {
						let w1 = geigval_1x1(A1, B1);
						let w2_swap = geigval_1x1(A2_swap, B2_swap);
						assert!(w1 ~ w2_swap);
					} else {
						let w1 = geigval_2x2(A1, B1);
						let w2_swap = geigval_2x2(A2_swap, B2_swap);
						assert!(all(
							w1.0 ~ w2_swap.0,
							w1.1 ~ w2_swap.1,
							w1.2 ~ w2_swap.2,
							w1.3 ~ w2_swap.3,
						));
					}

					if n2 == 1 {
						let w1 = geigval_1x1(A2, B2);
						let w2_swap = geigval_1x1(A1_swap, B1_swap);

						assert!(w1 ~ w2_swap);
					} else {
						let w1 = geigval_2x2(A2, B2);
						let w2_swap = geigval_2x2(A1_swap, B1_swap);

						assert!(all(
							w1.0 ~ w2_swap.0,
							w1.1 ~ w2_swap.1,
							w1.2 ~ w2_swap.2,
							w1.3 ~ w2_swap.3,
						));
					}

					let approx_eq = CwiseMat(approx_eq);
					let A_old = &Q * &A_swap * &Z.transpose();
					let B_old = &Q * &B_swap * &Z.transpose();
					assert!(all(A_old ~ A, B_old ~ B));
				}
			}
		}
	}

	#[test]
	fn test_swap_qz_edge_case() {
		let approx_eq = ApproxEq::<f64> {
			abs_tol: 1e-10,
			rel_tol: 1e-10,
		};

		let rng = &mut StdRng::seed_from_u64(0);

		for (n1, n2) in [(1, 1), (2, 2)] {
			for (ns, j) in [
				([n1, n2].as_slice(), 0),
				([2, n1, n2, 2].as_slice(), 2),
				([n1, n2, 2].as_slice(), 0),
				([2, n1, n2].as_slice(), 2),
				([1, n1, n2, 1].as_slice(), 1),
				([n1, n2, 1].as_slice(), 0),
				([1, n1, n2].as_slice(), 1),
			] {
				let n = ns.iter().sum::<usize>();

				for _ in 0..10 {
					let (mut A, mut B) = make_pair(rng, ns);

					let A1 = A.get(j..j + n1, j..j + n1).to_owned();
					let B1 = B.get(j..j + n1, j..j + n1).to_owned();
					A.get_mut(j + n1..j + n1 + n2, j + n1..j + n1 + n2).copy_from(A1);
					B.get_mut(j + n1..j + n1 + n2, j + n1..j + n1 + n2).copy_from(B1);

					let mut A_swap = A.clone();
					let mut B_swap = B.clone();
					let mut Q = Mat::identity(n, n);
					let mut Z = Mat::identity(n, n);

					swap_qz(
						A_swap.as_mut(),
						B_swap.as_mut(),
						Some(Q.as_mut()),
						Some(Z.as_mut()),
						j,
						n1,
						n2,
						MemStack::new(&mut [core::mem::MaybeUninit::new(0u8); 16 * 1024]),
					);

					let A1 = A.get(j..j + n1, j..j + n1);
					let A2 = A.get(j + n1..j + n1 + n2, j + n1..j + n1 + n2);
					let B1 = B.get(j..j + n1, j..j + n1);
					let B2 = B.get(j + n1..j + n1 + n2, j + n1..j + n1 + n2);

					let A1_swap = A_swap.get(j..j + n2, j..j + n2);
					let A2_swap = A_swap.get(j + n2..j + n2 + n1, j + n2..j + n2 + n1);
					let B1_swap = B_swap.get(j..j + n2, j..j + n2);
					let B2_swap = B_swap.get(j + n2..j + n2 + n1, j + n2..j + n2 + n1);

					if n1 == 1 {
						let w1 = geigval_1x1(A1, B1);
						let w2_swap = geigval_1x1(A2_swap, B2_swap);
						assert!(w1 ~ w2_swap);
					} else {
						let w1 = geigval_2x2(A1, B1);
						let w2_swap = geigval_2x2(A2_swap, B2_swap);
						assert!(all(
							w1.0 ~ w2_swap.0,
							w1.1 ~ w2_swap.1,
							w1.2 ~ w2_swap.2,
							w1.3 ~ w2_swap.3,
						));
					}

					if n2 == 1 {
						let w1 = geigval_1x1(A2, B2);
						let w2_swap = geigval_1x1(A1_swap, B1_swap);

						assert!(w1 ~ w2_swap);
					} else {
						let w1 = geigval_2x2(A2, B2);
						let w2_swap = geigval_2x2(A1_swap, B1_swap);

						assert!(all(
							w1.0 ~ w2_swap.0,
							w1.1 ~ w2_swap.1,
							w1.2 ~ w2_swap.2,
							w1.3 ~ w2_swap.3,
						));
					}

					let approx_eq = CwiseMat(approx_eq);
					let A_old = &Q * &A_swap * &Z.transpose();
					let B_old = &Q * &B_swap * &Z.transpose();
					assert!(all(A_old ~ A, B_old ~ B));
				}
			}
		}
	}

	#[test]
	fn test_qz_real_unblocked() {
		let rng = &mut StdRng::seed_from_u64(0);
		for n in [6, 12, 53, 102] {
			let rand = stats::CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			};
			let mut sample = || -> Mat<f64> { rand.sample(rng) };
			let A = sample();
			let mut B = sample();
			zip!(&mut B).for_each_triangular_lower(linalg::zip::Diag::Skip, |unzip!(x)| {
				*x = 0.0;
			});
			let B = B;

			let mut mem = MemBuffer::new(generalized_hessenberg_scratch::<f64>(
				n,
				GeneralizedHessenbergParams {
					block_size: 32,
					..auto!(f64)
				},
			));

			let mut Q = Mat::identity(n, n);
			let mut Z = Mat::identity(n, n);

			let mut A_clone = A.clone();
			let mut B_clone = B.clone();

			generalized_hessenberg(
				A_clone.as_mut(),
				B_clone.as_mut(),
				Some(Q.as_mut()),
				Some(Z.as_mut()),
				Par::Seq,
				MemStack::new(&mut mem),
				GeneralizedHessenbergParams {
					block_size: 32,
					..auto!(f64)
				},
			);

			let mut alphar = Col::zeros(n);
			let mut alphai = Col::zeros(n);
			let mut beta = Col::zeros(n);

			hessenberg_to_qz_unblocked(
				0,
				n - 1,
				A_clone.as_mut(),
				B_clone.as_mut(),
				Some(Q.as_mut()),
				Some(Z.as_mut()),
				alphar.as_mut(),
				alphai.as_mut(),
				beta.as_mut(),
				false,
			);

			assert!((&Q * &A_clone * Z.adjoint() - &A).norm_max() < 1e-13);
			assert!((&Q * &B_clone * Z.adjoint() - &B).norm_max() < 1e-13);

			for j in 0..n {
				for i in j + 1..n {
					assert!(B_clone[(i, j)] == 0.0);
				}
			}

			let mut successive = false;
			for j in 0..n {
				if j + 1 < n {
					if A_clone[(j + 1, j)] != 0.0 {
						assert!(!successive);
						successive = true;
					} else {
						successive = false;
					}
				}
				for i in j + 2..n {
					assert!(B_clone[(i, j)] == 0.0);
				}
			}
		}
	}

	#[test]
	fn test_qz_real_blocked() {
		let rng = &mut StdRng::seed_from_u64(0);
		let approx_eq = CwiseMat(ApproxEq::<f64> {
			abs_tol: 1e-10,
			rel_tol: 1e-10,
		});

		for n in [102, 251] {
			for _ in 0..10 {
				let rand = stats::CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: StandardNormal,
				};
				let mut sample = || -> Mat<f64> { rand.sample(rng) };
				let A = sample();
				let mut B = sample();
				zip!(&mut B).for_each_triangular_lower(linalg::zip::Diag::Skip, |unzip!(x)| {
					*x = 0.0;
				});
				let B = B;

				let mut mem = MemBuffer::new(generalized_hessenberg_scratch::<f64>(
					n,
					GeneralizedHessenbergParams {
						block_size: 32,
						..auto!(f64)
					},
				));

				let mut Q = Mat::identity(n, n);
				let mut Z = Mat::identity(n, n);

				let mut A_clone = A.clone();
				let mut B_clone = B.clone();

				generalized_hessenberg(
					A_clone.as_mut(),
					B_clone.as_mut(),
					Some(Q.as_mut()),
					Some(Z.as_mut()),
					Par::Seq,
					MemStack::new(&mut mem),
					GeneralizedHessenbergParams {
						block_size: 32,
						..auto!(f64)
					},
				);

				let mut alphar = Col::zeros(n);
				let mut alphai = Col::zeros(n);
				let mut beta = Col::zeros(n);

				hessenberg_to_qz_blocked(
					0,
					n - 1,
					A_clone.as_mut(),
					B_clone.as_mut(),
					Some(Q.as_mut()),
					Some(Z.as_mut()),
					alphar.as_mut(),
					alphai.as_mut(),
					beta.as_mut(),
					false,
					Par::Seq,
					auto!(f64),
					MemStack::new(&mut MemBuffer::new(hessenberg_to_qz_scratch::<f64>(n, Par::Seq, auto!(f64)))),
				);
				assert!(&Q * &A_clone * Z.adjoint() ~ A);
				assert!(&Q * &B_clone * Z.adjoint() ~ B);

				for j in 0..n {
					for i in j + 1..n {
						assert!(B_clone[(i, j)] == 0.0);
					}
				}

				let mut successive = false;
				for j in 0..n {
					if j + 1 < n {
						if A_clone[(j + 1, j)] != 0.0 {
							assert!(!successive);
							successive = true;
						} else {
							successive = false;
						}
					}
					for i in j + 2..n {
						assert!(B_clone[(i, j)] == 0.0);
					}
				}
			}
		}
	}
}
