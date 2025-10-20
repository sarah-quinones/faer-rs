use super::GeneralizedSchurParams;
use super::gen_hessenberg::{make_givens, rot, trot};
use crate::internal_prelude::*;
use equator::assert;
use linalg::matmul::matmul;

fn hessenberg_to_qz_unblocked<T: ComplexField>(
	ilo: usize,
	ihi: usize,
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	Q: Option<MatMut<'_, T>>,
	Z: Option<MatMut<'_, T>>,
	alpha: ColMut<'_, T>,
	beta: ColMut<'_, T>,
	eigvals_only: bool,
) {
	let mut H = A;

	let mut T = B;

	let mut Q = Q;

	let mut Z = Z;

	let mut alpha = alpha;

	let mut beta = beta;

	let n = H.nrows();

	let ulp = eps::<T::Real>();

	let safmin = min_positive::<T::Real>();

	for j in ihi + 1..n {
		let absb = T[(j, j)].abs();

		if absb > safmin {
			let signbc = T[(j, j)].conj().mul_real(absb.recip());

			T[(j, j)] = absb.to_cplx();

			if !eigvals_only {
				zip!(T.rb_mut().get_mut(..j - 1, j)).for_each(|unzip!(x)| *x *= &signbc);

				zip!(H.rb_mut().get_mut(..j, j)).for_each(|unzip!(x)| *x *= &signbc);
			} else {
				H[(j, j)] = &H[(j, j)] * &signbc;
			}

			if let Some(Z) = Z.rb_mut() {
				zip!(Z.col_mut(j)).for_each(|unzip!(x)| *x *= &signbc);
			}
		} else {
			T[(j, j)] = zero();
		}

		alpha[j] = H[(j, j)].copy();

		beta[j] = T[(j, j)].copy();
	}

	let mut ifirst;

	let mut ilast = ihi;

	let mut ifrstm;

	let mut ilastm;

	if !eigvals_only {
		ifrstm = 0usize;

		ilastm = n - 1;
	} else {
		ifrstm = ilo;

		ilastm = ihi;
	}

	let mut iiter = 0usize;

	let maxit = 30 * (ihi + 1 - ilo);

	let mut eshift = zero::<T>();

	let anorm = H.rb().get(ilo..ihi + 1, ilo..ihi + 1).norm_max();

	let bnorm = T.rb().get(ilo..ihi + 1, ilo..ihi + 1).norm_max();

	let atol = safmin.fmax(&ulp * &anorm);

	let btol = safmin.fmax(&ulp * &bnorm);

	let ascale = safmin.fmax(&anorm).recip();

	let bscale = safmin.fmax(&bnorm).recip();

	if ihi >= ilo {
		'main_loop: for _ in 0..maxit {
			'goto70: {
				'goto60: {
					'goto50: {
						if ilast == ilo {
							break 'goto60;
						}

						if H[(ilast, ilast - 1)].abs1() <= atol {
							H[(ilast, ilast - 1)] = zero();

							break 'goto60;
						}

						if T[(ilast, ilast)].abs1() <= btol {
							T[(ilast, ilast)] = zero();

							break 'goto50;
						}

						for j in (ilo..ilast).rev() {
							let ilazro;

							if j == ilo {
								ilazro = true;
							} else if H[(j, j - 1)].abs1() < atol {
								H[(j, j - 1)] = zero();

								ilazro = true;
							} else {
								ilazro = false;
							}

							if T[(j, j)].abs1() < btol {
								T[(j, j)] = zero();

								let mut ilazr2 = false;

								if !ilazro {
									if H[(j, j - 1)].abs1() * (&ascale * H[(j + 1, j)].abs1()) < H[(j, j)].abs1() * (&ascale * &atol) {
										ilazr2 = true;
									}
								}

								if ilazro || ilazr2 {
									for jch in j..ilast {
										let (c, s, r) = make_givens(H[(jch, jch)].copy(), H[(jch + 1, jch)].copy());

										H[(jch, jch)] = r;

										H[(jch + 1, jch)] = zero();

										let (x, y) = H.rb_mut().get_mut(.., jch + 1..ilastm + 1).two_rows_mut(jch, jch + 1);

										rot(c.copy(), s.copy(), x, y);

										let (x, y) = T.rb_mut().get_mut(.., jch + 1..ilastm + 1).two_rows_mut(jch, jch + 1);

										rot(c.copy(), s.copy(), x, y);

										if let Some(mut Q) = Q.rb_mut() {
											let (x, y) = Q.rb_mut().two_rows_mut(jch, jch + 1);

											rot(c.copy(), s.copy(), x, y);
										}

										if ilazr2 {
											H[(jch, jch - 1)] = H[(jch, jch - 1)].mul_real(&c);
										}

										ilazr2 = false;

										if T[(jch + 1, jch + 1)].abs1() >= btol {
											if jch + 1 >= ilast {
												break 'goto60;
											} else {
												ifirst = jch + 1;

												break 'goto70;
											}
										}

										T[(jch + 1, jch + 1)] = zero();
									}

									break 'goto50;
								} else {
									for jch in j..ilast {
										let (c, s, r) = make_givens(T[(jch, jch + 1)].copy(), T[(jch + 1, jch + 1)].copy());

										T[(jch, jch + 1)] = r;

										T[(jch + 1, jch + 1)] = zero();

										if jch < ilastm - 1 {
											let (x, y) = T.rb_mut().get_mut(.., jch + 2..ilastm + 1).two_rows_mut(jch, jch + 1);

											rot(c.copy(), s.copy(), x, y);
										}

										let (x, y) = H.rb_mut().get_mut(.., jch - 1..ilastm + 1).two_rows_mut(jch, jch + 1);

										rot(c.copy(), s.copy(), x, y);

										if let Some(mut Q) = Q.rb_mut() {
											let (x, y) = Q.rb_mut().two_cols_mut(jch, jch + 1);

											trot(c.copy(), -&s, x, y);
										}

										let (c, s, r) = make_givens(H[(jch + 1, jch)].copy(), H[(jch + 1, jch - 1)].copy());

										H[(jch + 1, jch)] = r;

										H[(jch + 1, jch - 1)] = zero();

										let (x, y) = H.rb_mut().get_mut(ifrstm..jch + 1, ..).two_cols_mut(jch, jch - 1);

										rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());

										let (x, y) = T.rb_mut().get_mut(ifrstm..jch, ..).two_cols_mut(jch, jch - 1);

										rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());

										if let Some(mut Z) = Z.rb_mut() {
											let (x, y) = Z.rb_mut().two_cols_mut(jch, jch - 1);

											rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());
										}
									}

									break 'goto50;
								}
							} else if ilazro {
								ifirst = j;

								break 'goto70;
							}
						}

						panic!();
					}

					let (c, s, r) = make_givens(H[(ilast, ilast)].copy(), H[(ilast, ilast - 1)].copy());

					H[(ilast, ilast)] = r;

					H[(ilast, ilast - 1)] = zero();

					let (x, y) = H.rb_mut().get_mut(ifrstm..ilast, ..).two_cols_mut(ilast, ilast - 1);

					rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());

					let (x, y) = T.rb_mut().get_mut(ifrstm..ilast, ..).two_cols_mut(ilast, ilast - 1);

					rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());

					if let Some(mut Z) = Z.rb_mut() {
						let (x, y) = Z.rb_mut().two_cols_mut(ilast, ilast - 1);

						rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());
					}
				}

				let absb = T[(ilast, ilast)].abs();

				if absb > safmin {
					let signbc = T[(ilast, ilast)].conj().mul_real(absb.recip());

					T[(ilast, ilast)] = absb.to_cplx();

					if !eigvals_only {
						zip!(T.rb_mut().get_mut(ifrstm..ilast, ilast)).for_each(|unzip!(x)| *x *= &signbc);

						zip!(H.rb_mut().get_mut(ifrstm..ilast + 1, ilast)).for_each(|unzip!(x)| *x *= &signbc);
					} else {
						H[(ilast, ilast)] = &H[(ilast, ilast)] * &signbc;
					}

					if let Some(Z) = Z.rb_mut() {
						zip!(Z.col_mut(ilast)).for_each(|unzip!(x)| *x *= &signbc);
					}
				} else {
					T[(ilast, ilast)] = zero();
				}

				alpha[ilast] = H[(ilast, ilast)].copy();

				beta[ilast] = T[(ilast, ilast)].copy();

				ilast = ilast.wrapping_sub(1);

				if ilast == usize::MAX || ilast < ilo {
					break 'main_loop;
				}

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

			iiter += 1;

			if eigvals_only {
				ifrstm = ifirst;
			}

			let shift;

			if iiter % 10 != 0 {
				let u12 = T[(ilast - 1, ilast)].mul_real(&bscale) * T[(ilast, ilast)].mul_real(&bscale).recip();

				let ad11 = H[(ilast - 1, ilast - 1)].mul_real(&ascale) * T[(ilast - 1, ilast - 1)].mul_real(&bscale).recip();

				let ad21 = H[(ilast, ilast - 1)].mul_real(&ascale) * T[(ilast - 1, ilast - 1)].mul_real(&bscale).recip();

				let ad12 = H[(ilast - 1, ilast)].mul_real(&ascale) * T[(ilast, ilast)].mul_real(&bscale).recip();

				let ad22 = H[(ilast, ilast)].mul_real(&ascale) * T[(ilast, ilast)].mul_real(&bscale).recip();

				let abi22 = &ad22 - &u12 * &ad21;

				let t1 = (&ad11 + &abi22).mul_real(from_f64::<T::Real>(0.5));

				let rtdisc = (&t1 * &t1 + &ad12 * &ad21 - &ad11 * &ad22).sqrt();

				let diff = &t1 - &abi22;

				let temp = diff.real() * rtdisc.real() + diff.imag() * rtdisc.imag();

				if temp <= zero() {
					shift = &t1 + &rtdisc;
				} else {
					shift = &t1 - &rtdisc;
				}
			} else {
				eshift = &eshift + H[(ilast, ilast - 1)].mul_real(&ascale) * T[(ilast - 1, ilast - 1)].mul_real(&bscale).recip();

				shift = eshift.copy();
			}

			let mut istart;

			let mut ctemp;

			'goto90: {
				for j in (ifirst + 1..ilast).rev() {
					istart = j;

					ctemp = H[(j, j)].mul_real(&ascale) - (&shift * T[(j, j)].mul_real(&bscale));

					let mut temp = ctemp.abs1();

					let mut temp2 = &ascale * H[(j + 1, j)].abs1();

					let ref tempr = temp.fmax(&temp2);

					if *tempr < one() && *tempr != zero() {
						temp = temp / tempr;

						temp2 = temp2 / tempr;
					}

					if H[(j, j - 1)].abs1() * &temp2 <= &temp * &atol {
						break 'goto90;
					}
				}

				istart = ifirst;

				ctemp = H[(ifirst, ifirst)].mul_real(&ascale) - &shift * T[(ifirst, ifirst)].mul_real(&bscale);
			}

			let ctemp2 = H[(istart + 1, istart)].mul_real(&ascale);

			let (mut c, mut s, _) = make_givens(ctemp, ctemp2);

			for j in istart..ilast {
				if j > istart {
					let r;

					(c, s, r) = make_givens(H[(j, j - 1)].copy(), H[(j + 1, j - 1)].copy());

					H[(j, j - 1)] = r;

					H[(j + 1, j - 1)] = zero();
				}

				for jc in j..ilastm + 1 {
					let ctemp = H[(j, jc)].mul_real(&c) + &H[(j + 1, jc)] * &s;

					H[(j + 1, jc)] = H[(j + 1, jc)].mul_real(&c) - &H[(j, jc)] * s.conj();

					H[(j, jc)] = ctemp;

					let ctemp2 = T[(j, jc)].mul_real(&c) + &T[(j + 1, jc)] * &s;

					T[(j + 1, jc)] = T[(j + 1, jc)].mul_real(&c) - &T[(j, jc)] * s.conj();

					T[(j, jc)] = ctemp2;
				}

				if let Some(mut Q) = Q.rb_mut() {
					for jr in 0..n {
						let ctemp = Q[(jr, j)].mul_real(&c) + &Q[(jr, j + 1)] * s.conj();

						Q[(jr, j + 1)] = Q[(jr, j + 1)].mul_real(&c) - &Q[(jr, j)] * &s;

						Q[(jr, j)] = ctemp;
					}
				}

				let r;

				(c, s, r) = make_givens(T[(j + 1, j + 1)].copy(), T[(j + 1, j)].copy());

				T[(j + 1, j + 1)] = r;

				T[(j + 1, j)] = zero();

				for jr in ifrstm..Ord::min(j + 3, ilast + 1) {
					let ctemp = H[(jr, j + 1)].mul_real(&c) + &H[(jr, j)] * &s;

					H[(jr, j)] = H[(jr, j)].mul_real(&c) - &H[(jr, j + 1)] * s.conj();

					H[(jr, j + 1)] = ctemp;
				}

				for jr in ifrstm..j + 1 {
					let ctemp = T[(jr, j + 1)].mul_real(&c) + &T[(jr, j)] * &s;

					T[(jr, j)] = T[(jr, j)].mul_real(&c) - &T[(jr, j + 1)] * s.conj();

					T[(jr, j + 1)] = ctemp;
				}

				if let Some(mut Z) = Z.rb_mut() {
					for jr in 0..n {
						let ctemp = Z[(jr, j + 1)].mul_real(&c) + &Z[(jr, j)] * &s;

						Z[(jr, j)] = Z[(jr, j)].mul_real(&c) - &Z[(jr, j + 1)] * s.conj();

						Z[(jr, j + 1)] = ctemp;
					}
				}
			}
		}
	}

	for j in 0..ilo {
		let absb = T[(j, j)].abs();

		if absb > safmin {
			let signbc = T[(j, j)].conj().mul_real(absb.recip());

			T[(j, j)] = absb.to_cplx();

			if !eigvals_only {
				zip!(T.rb_mut().get_mut(..j - 1, j)).for_each(|unzip!(x)| *x *= &signbc);

				zip!(H.rb_mut().get_mut(..j, j)).for_each(|unzip!(x)| *x *= &signbc);
			} else {
				H[(j, j)] = &H[(j, j)] * &signbc;
			}

			if let Some(Z) = Z.rb_mut() {
				zip!(Z.col_mut(j)).for_each(|unzip!(x)| *x *= &signbc);
			}
		} else {
			T[(j, j)] = zero();
		}

		alpha[j] = H[(j, j)].copy();

		beta[j] = T[(j, j)].copy();
	}
}

fn chase_bulge_1x1<T: ComplexField>(
	k: usize,
	istartm: usize,
	istopm: usize,
	ihi: usize,
	mut A: MatMut<'_, T>,
	mut B: MatMut<'_, T>,
	mut Q: Option<MatMut<'_, T>>,
	mut Z: Option<MatMut<'_, T>>,
	qstart: usize,
	zstart: usize,
) {
	if k + 1 == ihi {
		let (c, s, r) = make_givens(B[(ihi, ihi)].copy(), B[(ihi, ihi - 1)].copy());

		B[(ihi, ihi)] = r;

		B[(ihi, ihi - 1)] = zero();

		let (x, y) = B.rb_mut().get_mut(istartm..ihi, ..).two_cols_mut(ihi, ihi - 1);

		rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());

		let (x, y) = A.rb_mut().get_mut(istartm..ihi + 1, ..).two_cols_mut(ihi, ihi - 1);

		rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());

		if let Some(Z) = Z.rb_mut() {
			let (x, y) = Z.two_cols_mut(&ihi - &zstart, ihi - 1 - zstart);

			rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());
		}
	} else {
		let (c, s, r) = make_givens(B[(k + 1, k + 1)].copy(), B[(k + 1, k)].copy());

		B[(k + 1, k + 1)] = r;

		B[(k + 1, k)] = zero();

		let (x, y) = A.rb_mut().get_mut(istartm..k + 3, ..).two_cols_mut(k + 1, k);

		rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());

		let (x, y) = B.rb_mut().get_mut(istartm..k + 1, ..).two_cols_mut(k + 1, k);

		rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());

		if let Some(Z) = Z.rb_mut() {
			let (x, y) = Z.two_cols_mut(k + 1 - zstart, &k - &zstart);

			rot(c.copy(), s.copy(), x.transpose_mut(), y.transpose_mut());
		}

		let (c, s, r) = make_givens(A[(k + 1, k)].copy(), A[(k + 2, k)].copy());

		A[(k + 1, k)] = r;

		A[(k + 2, k)] = zero();

		let (x, y) = A.rb_mut().get_mut(.., k + 1..istopm + 1).two_rows_mut(k + 1, k + 2);

		rot(c.copy(), s.copy(), x, y);

		let (x, y) = B.rb_mut().get_mut(.., k + 1..istopm + 1).two_rows_mut(k + 1, k + 2);

		rot(c.copy(), s.copy(), x, y);

		if let Some(Q) = Q.rb_mut() {
			let (x, y) = Q.two_cols_mut(k + 1 - qstart, k + 2 - qstart);

			rot(c.copy(), s.conj(), x.transpose_mut(), y.transpose_mut());
		}
	}
}

fn multishift_sweep<T: ComplexField>(
	eigvals_only: bool,
	ilo: usize,
	ihi: usize,
	alpha: ColRef<'_, T>,
	beta: ColRef<'_, T>,
	mut A: MatMut<'_, T>,
	mut B: MatMut<'_, T>,
	mut Q: Option<MatMut<'_, T>>,
	mut Z: Option<MatMut<'_, T>>,
	mut QC: MatMut<'_, T>,
	mut ZC: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let safmin = min_positive::<T::Real>();

	let safmax = safmin.recip();

	let nblock_desired = QC.ncols();

	let n = A.nrows();

	let istartm;

	let istopm;

	if eigvals_only {
		istartm = ilo;

		istopm = ihi;
	} else {
		istartm = 0;

		istopm = n - 1;
	}

	let nshifts = alpha.nrows();

	let ns = nshifts;

	let npos = Ord::max(&nblock_desired - &ns, 1);

	let mut qc = QC.rb_mut().get_mut(..ns + 1, ..ns + 1);

	let mut zc = ZC.rb_mut().get_mut(..ns, ..ns);

	for mut m in [qc.rb_mut(), zc.rb_mut()] {
		m.fill(zero());

		m.rb_mut().diagonal_mut().column_vector_mut().fill(one());
	}

	for i in 0..ns {
		let scale = alpha[i].abs().sqrt() * beta[i].abs().sqrt();

		let (alpha, beta) = if scale >= safmin && scale <= safmax {
			(alpha[i].mul_real(scale.recip()), beta[i].mul_real(scale.recip()))
		} else {
			(alpha[i].copy(), beta[i].copy())
		};

		let mut temp2 = &beta * &A[(ilo, ilo)] - &alpha * &B[(ilo, ilo)];

		let mut temp3 = &beta * &A[(ilo + 1, ilo)];

		if temp2.abs() > safmax || temp3.abs() > safmax {
			temp2 = one();

			temp3 = zero();
		}

		let (c, s, _) = make_givens(temp2, temp3);

		let (x, y) = A.rb_mut().get_mut(.., ilo..&ilo + &ns).two_rows_mut(ilo, ilo + 1);

		rot(c.copy(), s.copy(), x, y);

		let (x, y) = B.rb_mut().get_mut(.., ilo..&ilo + &ns).two_rows_mut(ilo, ilo + 1);

		rot(c.copy(), s.copy(), x, y);

		let (x, y) = qc.rb_mut().get_mut(.., ..ns + 1).two_cols_mut(0, 1);

		rot(c.copy(), s.conj(), x.transpose_mut(), y.transpose_mut());

		for j in 0..&ns - &i - 1 {
			chase_bulge_1x1(
				j,
				0,
				ns - 1,
				&ihi - &ilo,
				A.rb_mut().get_mut(ilo.., ilo..),
				B.rb_mut().get_mut(ilo.., ilo..),
				Some(qc.rb_mut()),
				Some(zc.rb_mut()),
				0,
				0,
			);
		}
	}

	let sheight = ns + 1;

	let swidth = (istopm + 1).saturating_sub(&ilo + &ns);

	if swidth > 0 {
		{
			let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(sheight, swidth, stack) };

			let mut work = work.as_mat_mut();

			for M in [A.rb_mut(), B.rb_mut()] {
				let mut M = M.submatrix_mut(ilo, &ilo + &ns, sheight, swidth);

				matmul(work.rb_mut(), Accum::Replace, qc.rb().adjoint(), M.rb(), one(), par);

				M.copy_from(&work);
			}
		}
	}

	if let Some(mut Q) = Q.rb_mut() {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(n, sheight, stack) };

		let mut work = work.as_mat_mut();

		let mut M = Q.rb_mut().get_mut(.., ilo..&ilo + &sheight);

		matmul(work.rb_mut(), Accum::Replace, M.rb(), qc.rb(), one(), par);

		M.copy_from(&work);
	}

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

		let mut M = Z.rb_mut().get_mut(.., ilo..&ilo + &swidth);

		matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);

		M.copy_from(&work);
	}

	let mut k = ilo;

	while k < &ihi - &ns {
		let np = Ord::min(&ihi - &ns - &k, npos);

		let nblock = &ns + &np;

		let istartb = k + 1;

		let istopb = &k + &nblock - 1;

		let mut qc = QC.rb_mut().get_mut(..nblock, ..nblock);

		let mut zc = ZC.rb_mut().get_mut(..nblock, ..nblock);

		qc.fill(zero());

		qc.rb_mut().diagonal_mut().column_vector_mut().fill(one());

		zc.fill(zero());

		zc.rb_mut().diagonal_mut().column_vector_mut().fill(one());

		for i in (0..ns).rev() {
			for j in 0..np {
				chase_bulge_1x1(
					&k + &i + &j,
					istartb,
					istopb,
					ihi,
					A.rb_mut(),
					B.rb_mut(),
					Some(qc.rb_mut()),
					Some(zc.rb_mut()),
					k + 1,
					k,
				);
			}
		}

		let sheight = &ns + &np;

		let swidth = (istopm + 1).saturating_sub(&k + &ns + &np);

		if swidth > 0 {
			let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(sheight, swidth, stack) };

			let mut work = work.as_mat_mut();

			for M in [A.rb_mut(), B.rb_mut()] {
				let mut M = M.get_mut(k + 1..k + 1 + sheight, &k + &ns + &np..&k + &ns + &np + &swidth);

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

		let sheight = (k + 1).saturating_sub(istartm);

		let swidth = nblock;

		if sheight > 0 {
			let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(sheight, swidth, stack) };

			let mut work = work.as_mat_mut();

			for M in [A.rb_mut(), B.rb_mut()] {
				let mut M = M.get_mut(istartm..&istartm + &sheight, k..&k + &swidth);

				matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);

				M.copy_from(&work);
			}
		}

		if let Some(mut Z) = Z.rb_mut() {
			let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(n, swidth, stack) };

			let mut work = work.as_mat_mut();

			let mut M = Z.rb_mut().get_mut(.., k..&k + &swidth);

			matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);

			M.copy_from(&work);
		}

		k += np;
	}

	let mut qc = QC.rb_mut().get_mut(..ns, ..ns);

	let mut zc = ZC.rb_mut().get_mut(..ns + 1, ..ns + 1);

	qc.fill(zero());

	qc.rb_mut().diagonal_mut().column_vector_mut().fill(one());

	zc.fill(zero());

	zc.rb_mut().diagonal_mut().column_vector_mut().fill(one());

	let istartb = &ihi - &ns + 1;

	let istopb = ihi;

	for i in 0..ns {
		let i = i + 1;

		for ishift in &ihi - &i..ihi {
			chase_bulge_1x1(
				ishift,
				istartb,
				istopb,
				ihi,
				A.rb_mut(),
				B.rb_mut(),
				Some(qc.rb_mut()),
				Some(zc.rb_mut()),
				&ihi - &ns + 1,
				&ihi - &ns,
			);
		}
	}

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

		let mut M = Q.rb_mut().get_mut(.., &ihi - &ns + 1..ihi + 1);

		matmul(work.rb_mut(), Accum::Replace, M.rb(), qc.rb(), one(), par);

		M.copy_from(&work);
	}

	let sheight = (ihi + 1).saturating_sub(&ns + &istartm);

	let swidth = ns + 1;

	if sheight > 0 {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(sheight, swidth, stack) };

		let mut work = work.as_mat_mut();

		for M in [A.rb_mut(), B.rb_mut()] {
			let mut M = M.submatrix_mut(istartm, &ihi - &ns, sheight, swidth);

			matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);

			M.copy_from(&work);
		}
	}

	if let Some(mut Z) = Z.rb_mut() {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(n, swidth, stack) };

		let mut work = work.as_mat_mut();

		let mut M = Z.rb_mut().get_mut(.., &ihi - &ns..ihi + 1);

		matmul(work.rb_mut(), Accum::Replace, M.rb(), zc.rb(), one(), par);

		M.copy_from(&work);
	}
}

fn swap_qz<T: ComplexField>(mut A: MatMut<'_, T>, mut B: MatMut<'_, T>, mut Q: Option<MatMut<'_, T>>, mut Z: Option<MatMut<'_, T>>, j1: usize) {
	let mut s00 = A[(j1, j1)].copy();

	let s01 = A[(j1, j1 + 1)].copy();

	let mut s10 = zero::<T>();

	let s11 = A[(j1 + 1, j1 + 1)].copy();

	let mut t00 = B[(j1, j1)].copy();

	let t01 = B[(j1, j1 + 1)].copy();

	let mut t10 = zero::<T>();

	let t11 = B[(j1 + 1, j1 + 1)].copy();

	let f = &s11 * &t00 - &t11 * &s00;

	let g = &s11 * &t01 - &t11 * &s01;

	let sa = s11.abs() * t00.abs();

	let sb = s00.abs() * t11.abs();

	let (cz, sz, _) = make_givens(g, f);

	let sz = -&sz;

	s00 = s00.mul_real(&cz) + &s01 * sz.conj();

	s10 = s10.mul_real(&cz) + &s11 * sz.conj();

	t00 = t00.mul_real(&cz) + &t01 * sz.conj();

	t10 = t10.mul_real(&cz) + &t11 * sz.conj();

	let (cq, sq);

	if sa > sb {
		(cq, sq, _) = make_givens(s00, s10);
	} else {
		(cq, sq, _) = make_givens(t00, t10);
	}

	for M in [A.rb_mut(), B.rb_mut()] {
		let mut M: MatMut<'_, T> = M;

		let (x, y) = M.rb_mut().get_mut(..j1 + 2, ..).two_cols_mut(j1, j1 + 1);

		rot(cz.copy(), sz.conj(), x.transpose_mut(), y.transpose_mut());

		let (x, y) = M.rb_mut().get_mut(.., j1..).two_rows_mut(j1, j1 + 1);

		rot(cq.copy(), sq.copy(), x, y);

		M[(j1 + 1, j1)] = zero();
	}

	if let Some(Z) = Z.rb_mut() {
		let (x, y) = Z.two_cols_mut(j1, j1 + 1);

		rot(cz, sz.conj(), x.transpose_mut(), y.transpose_mut());
	}

	if let Some(Q) = Q.rb_mut() {
		let (x, y) = Q.two_cols_mut(j1, j1 + 1);

		rot(cq, sq.conj(), x.transpose_mut(), y.transpose_mut());
	}
}

fn reorder_qz<T: ComplexField>(
	mut A: MatMut<'_, T>,
	mut B: MatMut<'_, T>,
	mut Q: Option<MatMut<'_, T>>,
	mut Z: Option<MatMut<'_, T>>,
	ifst: usize,
	ilst: usize,
) {
	if ifst == ilst {
		return;
	}

	if ifst < ilst {
		let mut here = ifst;

		while here < ilst {
			swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here);

			here += 1;
		}
	} else {
		let mut here = ifst;

		while here > ilst {
			here -= 1;

			swap_qz(A.rb_mut(), B.rb_mut(), Q.rb_mut(), Z.rb_mut(), here);
		}
	}
}

fn aggressive_early_deflation<T: ComplexField>(
	eigvals_only: bool,
	ilo: usize,
	ihi: usize,
	nw: usize,
	mut A: MatMut<'_, T>,
	mut B: MatMut<'_, T>,
	mut Q: Option<MatMut<'_, T>>,
	mut Z: Option<MatMut<'_, T>>,
	mut alpha: ColMut<'_, T>,
	mut beta: ColMut<'_, T>,
	mut QC: MatMut<'_, T>,
	mut ZC: MatMut<'_, T>,
	par: Par,
	params: GeneralizedSchurParams,
	stack: &mut MemStack,
) -> (usize, usize) {
	let n = A.nrows();

	let ulp = eps::<T::Real>();

	let safmin = min_positive::<T::Real>();

	let smlnum = &safmin * (from_f64::<T::Real>(n as f64) / &ulp);

	let jw = Ord::min(nw, ihi + 1 - ilo);

	let kwtop = ihi + 1 - jw;

	let s;

	if kwtop == ilo {
		s = zero();
	} else {
		s = A[(kwtop, kwtop - 1)].copy();
	};

	let mut ifst;

	let mut ilst;

	if ihi == kwtop {
		alpha[kwtop] = A[(kwtop, kwtop)].copy();

		beta[kwtop] = B[(kwtop, kwtop)].copy();

		if s.abs() <= smlnum.fmax(A[(kwtop, kwtop)].abs() * &ulp) {
			if kwtop > ilo {
				A[(kwtop, kwtop - 1)] = zero();
			}
		}
	}

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
		alpha.rb_mut(),
		beta.rb_mut(),
		false,
		par,
		params,
		stack,
	);

	let mut kwbot;

	if kwtop == ilo || s == zero() {
		kwbot = kwtop.wrapping_sub(1);
	} else {
		kwbot = ihi;

		let mut k = 0;

		let mut k2 = 0;

		while k < jw {
			let mut tempr = A[(kwbot, kwbot)].abs();

			if tempr == zero() {
				tempr = s.abs();
			}

			if (&s * &qc[(0, kwbot - kwtop)]).abs() < smlnum.fmax(&ulp * &tempr) {
				kwbot -= 1;
			} else {
				ifst = &kwbot - &kwtop;

				ilst = k2;

				reorder_qz(
					A.rb_mut().submatrix_mut(kwtop, kwtop, jw, jw),
					B.rb_mut().submatrix_mut(kwtop, kwtop, jw, jw),
					Some(qc.rb_mut()),
					Some(zc.rb_mut()),
					ifst,
					ilst,
				);

				k2 += 1;
			}

			k += 1;
		}
	}

	let nd = ihi.wrapping_sub(kwbot);

	let ns = &jw - &nd;

	let mut k = kwtop;

	while k <= ihi {
		alpha[k] = A[(k, k)].copy();

		beta[k] = B[(k, k)].copy();

		k += 1;
	}

	if kwtop != ilo && s != zero() {
		let scale = A[(kwtop, kwtop - 1)].copy();

		zip!(A.rb_mut().get_mut(kwtop..kwbot + 1, kwtop - 1), qc.rb().get(0, ..jw - nd).transpose())
			.for_each(|unzip!(dst, src): Zip!(&mut T, &T)| *dst = src.conj() * &scale);

		for k in (kwtop..kwbot).rev() {
			let (c1, s1, temp) = make_givens(A[(k, kwtop - 1)].copy(), A[(k + 1, kwtop - 1)].copy());

			A[(k, kwtop - 1)] = temp;

			A[(k + 1, kwtop - 1)] = zero();

			let k2 = Ord::max(kwtop, k - 1);

			let (x, y) = A.rb_mut().get_mut(.., k2..ihi + 1).two_rows_mut(k, k + 1);

			rot(c1.copy(), s1.copy(), x, y);

			let (x, y) = B.rb_mut().get_mut(.., k - 1..ihi + 1).two_rows_mut(k, k + 1);

			rot(c1.copy(), s1.copy(), x, y);

			let (x, y) = qc.rb_mut().two_cols_mut(&k - &kwtop, k + 1 - kwtop);

			rot(c1.copy(), s1.conj(), x.transpose_mut(), y.transpose_mut());
		}

		let mut k = kwbot;

		while k > kwtop {
			k -= 1;

			for k2 in k..kwbot {
				chase_bulge_1x1(
					k2,
					kwtop,
					&kwtop + &jw - 1,
					kwbot,
					A.rb_mut(),
					B.rb_mut(),
					Some(qc.rb_mut()),
					Some(zc.rb_mut()),
					kwtop,
					kwtop,
				);
			}
		}
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

	if &istopm - &ihi > 0 {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(jw, &istopm - &ihi, stack) };

		let mut work = work.as_mat_mut();

		for M in [A.rb_mut(), B.rb_mut()] {
			let M: MatMut<'_, T> = M;

			let mut M = M.submatrix_mut(kwtop, ihi + 1, jw, &istopm - &ihi);

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

	if &kwtop - &istartm > 0 {
		let (mut work, _) = unsafe { linalg::temp_mat_uninit::<T, _, _>(&kwtop - &istartm, jw, stack) };

		let mut work = work.as_mat_mut();

		for M in [A.rb_mut(), B.rb_mut()] {
			let M: MatMut<'_, T> = M;

			let mut M = M.submatrix_mut(istartm, kwtop, &kwtop - &istartm, jw);

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

/// computes the layout of the workspace required to compute a complex matrix pair's QZ
/// decomposition, assuming the pair is already in generalized hessenberg form.

pub fn hessenberg_to_qz_scratch<T: ComplexField>(n: usize, par: Par, params: GeneralizedSchurParams) -> StackReq {
	let nmin = Ord::max(15, params.blocking_threshold);

	if n < nmin {
		return StackReq::EMPTY;
	}

	let nw = (n - 3) / 3;

	let nsr = (params.recommended_shift_count)(n, n);

	let rcost = (params.relative_cost_estimate_of_shift_chase_to_matmul)(n, n);

	let itemp1 = (nsr as f64 / (1.0 + 2.0 * (nsr as f64) / (rcost as f64 / 100.0 * n as f64)).sqrt()) as usize;

	let itemp1 = (itemp1.saturating_sub(1) / 4) * 4 + 4;

	let nbr = nsr + itemp1;

	let qc_aed = linalg::temp_mat_scratch::<T>(nw, nw);

	let qc_sweep = linalg::temp_mat_scratch::<T>(nbr, nbr);

	StackReq::any_of(&[
		StackReq::all_of(&[qc_aed, qc_aed, aed_scratch::<T>(n, nw, par, params)]),
		StackReq::all_of(&[qc_sweep, qc_sweep, multishift_sweep_scratch::<T>(n, nsr)]),
	])
}

fn multishift_sweep_scratch<T: ComplexField>(n: usize, ns: usize) -> StackReq {
	linalg::temp_mat_scratch::<T>(n, 2 * ns)
}

fn aed_scratch<T: ComplexField>(n: usize, nw: usize, par: Par, params: GeneralizedSchurParams) -> StackReq {
	StackReq::any_of(&[
		hessenberg_to_qz_scratch::<T>(nw, par, params),
		linalg::temp_mat_scratch::<T>(nw, n),
		linalg::temp_mat_scratch::<T>(n, nw),
	])
}

/// computes a complex matrix pair's QZ decomposition, assuming the pair is already in generalized
/// hessenberg form.
/// the unitary transformations $Q$ and $Z$ resulting from the QZ decomposition are postmultiplied
/// into the input-output parameters `Q_inout` and `Z_inout`.
///
/// if both the generalized eigenvalues and eigenvectors are desired, then `eigenvectors` may be set
/// to `ComputeEigenvectors::Yes`. in this case the input matrices $A$ and $B$ are overwritten by
/// their QZ form $(S, T)$ such that $S$ and $T$ are upper triangular.
///
/// if only the generalized eigenvalues are desired, then `eigenvectors` may be set to
/// `ComputeEigenvectors::No`. note that in this case, the input matrices $A$ and $B$ are still
/// clobbered, and contain unspecified values on output.
#[track_caller]

pub fn hessenberg_to_qz<T: ComplexField>(
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	Q: Option<MatMut<'_, T>>,
	Z: Option<MatMut<'_, T>>,
	alpha: ColMut<'_, T>,
	beta: ColMut<'_, T>,
	eigenvectors: linalg::evd::ComputeEigenvectors,
	par: Par,
	params: GeneralizedSchurParams,
	stack: &mut MemStack,
) {
	let eigvals_only = eigenvectors == linalg::evd::ComputeEigenvectors::No;

	let n = A.nrows();

	let (Q_nrows, Q_ncols) = Q.rb().map(|m| (m.nrows(), m.ncols())).unwrap_or((n, n));

	let (Z_nrows, Z_ncols) = Z.rb().map(|m| (m.nrows(), m.ncols())).unwrap_or((n, n));

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

	hessenberg_to_qz_blocked(0, n - 1, A, B, Q, Z, alpha, beta, eigvals_only, par, params, stack);
}

fn hessenberg_to_qz_blocked<T: ComplexField>(
	ilo: usize,
	ihi: usize,
	A: MatMut<'_, T>,
	B: MatMut<'_, T>,
	Q: Option<MatMut<'_, T>>,
	Z: Option<MatMut<'_, T>>,
	alpha: ColMut<'_, T>,
	beta: ColMut<'_, T>,
	eigvals_only: bool,
	par: Par,
	params: GeneralizedSchurParams,
	stack: &mut MemStack,
) {
	let mut A = A;

	let mut B = B;

	let mut Q = Q;

	let mut Z = Z;

	let mut alpha = alpha;

	let mut beta = beta;

	let n = A.nrows();

	let ulp = eps::<T::Real>();

	let safmin = min_positive::<T::Real>();

	let smlnum = &safmin * (from_f64::<T::Real>(n as f64) / &ulp);

	let bnorm = B.rb().get(ilo..ihi + 1, ilo..ihi + 1).norm_max();

	let btol = safmin.fmax(&ulp * &bnorm);

	let mut istart = ilo;

	let mut istop = ihi;

	let maxit = 30 * (ihi + 1 - ilo);

	let mut ld = 0usize;

	let mut eshift = zero();

	let nh = &ihi - &ilo + 1;

	let nmin = Ord::max(15, params.blocking_threshold);

	let nibble = params.nibble_threshold;

	let nwr = (params.recommended_deflation_window)(n, nh);

	let nsr = (params.recommended_shift_count)(n, nh);

	let rcost = (params.relative_cost_estimate_of_shift_chase_to_matmul)(n, nh);

	let itemp1 = (nsr as f64 / (1.0 + 2.0 * (nsr as f64) / ((rcost as f64 / 100.0) * n as f64)).sqrt()) as usize;

	let itemp1 = (itemp1.saturating_sub(1) / 4) * 4 + 4;

	let nbr = &nsr + &itemp1;

	if n < nmin {
		hessenberg_to_qz_unblocked(
			ilo,
			ihi,
			A.rb_mut(),
			B.rb_mut(),
			Q.rb_mut(),
			Z.rb_mut(),
			alpha.rb_mut(),
			beta.rb_mut(),
			eigvals_only,
		);

		return;
	}

	let nw_max = (n - 3) / 3;

	for iter in 0..maxit {
		_ = iter;

		if istop == usize::MAX || istart + 1 >= istop {
			break;
		}

		if A[(istop, istop - 1)].abs() <= smlnum.fmax(&ulp * (A[(istop, istop)].abs() + A[(istop - 1, istop - 1)].abs())) {
			A[(istop, istop - 1)] = zero();

			istop -= 1;

			ld = 0;

			eshift = zero();
		}

		if A[(istart + 1, istart)].abs() <= smlnum.fmax(&ulp * (A[(istart, istart)].abs() + A[(istart + 1, istart + 1)].abs())) {
			A[(istart + 1, istart)] = zero();

			istart += 1;

			ld = 0;

			eshift = zero();
		}

		if istart + 1 >= istop {
			break;
		}

		let mut istart2 = istart;

		for k in (istart + 1..istop + 1).rev() {
			if A[(k, k - 1)].abs() <= smlnum.fmax(&ulp * (A[(k, k)].abs() + A[(k - 1, k - 1)].abs())) {
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

		let mut k = istop + 1;

		while k > istart2 {
			k -= 1;

			if B[(k, k)].abs() < btol {
				for k2 in (istart2 + 1..k + 1).rev() {
					let (c1, s1, temp) = make_givens(B[(k2 - 1, k2)].copy(), B[(k2 - 1, k2 - 1)].copy());

					B[(k2 - 1, k2)] = temp;

					B[(k2 - 1, k2 - 1)] = zero();

					let (x, y) = B.rb_mut().get_mut(istartm..k2 - 1, ..).two_cols_mut(k2, k2 - 1);

					rot(c1.copy(), s1.copy(), x.transpose_mut(), y.transpose_mut());

					let (x, y) = A.rb_mut().get_mut(istartm..Ord::min(istop, k2 + 1) + 1, ..).two_cols_mut(k2, k2 - 1);

					rot(c1.copy(), s1.copy(), x.transpose_mut(), y.transpose_mut());

					if let Some(Z) = Z.rb_mut() {
						let (x, y) = Z.two_cols_mut(k2, k2 - 1);

						rot(c1.copy(), s1.copy(), x.transpose_mut(), y.transpose_mut());
					}

					if k2 < istop {
						let (c1, s1, temp) = make_givens(A[(k2, k2 - 1)].copy(), A[(k2 + 1, k2 - 1)].copy());

						A[(k2, k2 - 1)] = temp;

						A[(k2 + 1, k2 - 1)] = zero();

						let (x, y) = A.rb_mut().get_mut(.., k2..istopm + 1).two_rows_mut(k2, k2 + 1);

						rot(c1.copy(), s1.copy(), x, y);

						let (x, y) = B.rb_mut().get_mut(.., k2..istopm + 1).two_rows_mut(k2, k2 + 1);

						rot(c1.copy(), s1.copy(), x, y);

						if let Some(Q) = Q.rb_mut() {
							let (x, y) = Q.two_cols_mut(k2, k2 + 1);

							rot(c1.copy(), s1.conj(), x.transpose_mut(), y.transpose_mut());
						}
					}
				}

				if istart2 < istop {
					let (c1, s1, temp) = make_givens(A[(istart2, istart2)].copy(), A[(istart2 + 1, istart2)].copy());

					A[(istart2, istart2)] = temp;

					A[(istart2 + 1, istart2)] = zero();

					let (x, y) = A.rb_mut().get_mut(.., istart2 + 1..istopm + 1).two_rows_mut(istart2, istart2 + 1);

					rot(c1.copy(), s1.copy(), x, y);

					let (x, y) = B.rb_mut().get_mut(.., istart2 + 1..istopm + 1).two_rows_mut(istart2, istart2 + 1);

					rot(c1.copy(), s1.copy(), x, y);

					if let Some(Q) = Q.rb_mut() {
						let (x, y) = Q.two_cols_mut(istart2, istart2 + 1);

						rot(c1.copy(), s1.conj(), x.transpose_mut(), y.transpose_mut());
					}
				}

				istart2 += 1;
			}
		}

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

			(n_undeflated, n_deflated) = aggressive_early_deflation(
				eigvals_only,
				istart2,
				istop,
				nw,
				A.rb_mut(),
				B.rb_mut(),
				Q.rb_mut(),
				Z.rb_mut(),
				alpha.rb_mut(),
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

		if (100 * n_deflated > &nibble * (&n_deflated + &n_undeflated)) || (istop.wrapping_add(1) - &istart2 < nmin) {
			continue;
		}

		ld += 1;

		let mut ns = Ord::min(nshifts, istop.wrapping_sub(istart2));

		ns = Ord::min(ns, n_undeflated);

		let shiftpos = istop.wrapping_add(1) - &n_undeflated;

		if ld % 6 == 0 {
			if from_f64::<T::Real>(maxit as f64) * &safmin * A[(istop, istop - 1)].abs() < B[(istop - 1, istop - 1)].abs() {
				eshift = &A[(istop, istop - 1)] * B[(istop - 1, istop - 1)].recip();
			} else {
				eshift = &eshift + (&safmin * from_f64::<T::Real>(maxit as f64).recip()).to_cplx::<T>();
			}

			alpha[shiftpos] = one();

			beta[shiftpos] = eshift.copy();

			ns = 1;
		}

		let (mut QC, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(nblock, nblock, stack) };

		let mut QC = QC.as_mat_mut();

		let (mut ZC, stack) = unsafe { linalg::temp_mat_uninit::<T, _, _>(nblock, nblock, stack) };

		let mut ZC = ZC.as_mat_mut();

		multishift_sweep(
			eigvals_only,
			istart2,
			istop,
			alpha.rb().subrows(shiftpos, ns),
			beta.rb().subrows(shiftpos, ns),
			A.rb_mut(),
			B.rb_mut(),
			Q.rb_mut(),
			Z.rb_mut(),
			QC.rb_mut(),
			ZC.rb_mut(),
			par,
			stack,
		);
	}

	hessenberg_to_qz_unblocked(
		ilo,
		ihi,
		A.rb_mut(),
		B.rb_mut(),
		Q.rb_mut(),
		Z.rb_mut(),
		alpha.rb_mut(),
		beta.rb_mut(),
		eigvals_only,
	);
}

#[cfg(test)]

mod tests {

	use super::super::gen_hessenberg::{GeneralizedHessenbergParams, generalized_hessenberg, generalized_hessenberg_scratch};
	use super::*;
	use crate::stats::prelude::*;
	use crate::{Par, linalg, stats};
	use dyn_stack::MemBuffer;
	use equator::assert;

	#[test]

	fn test_qz_cplx_unblocked() {
		let rng = &mut StdRng::seed_from_u64(0);

		for n in [12, 35, 128, 255] {
			let rand = stats::CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			};

			let mut sample = || -> Mat<c64> { rand.sample(rng) };

			let A = sample();

			let mut B = sample();

			zip!(&mut B).for_each_triangular_lower(linalg::zip::Diag::Skip, |unzip!(x)| {
				*x = c64::ZERO;
			});

			let B = B;

			let mut mem = MemBuffer::new(generalized_hessenberg_scratch::<c64>(
				n,
				GeneralizedHessenbergParams {
					block_size: 32,
					..auto!(c64)
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
					..auto!(c64)
				},
			);

			let mut alpha = Col::zeros(n);

			let mut beta = Col::zeros(n);

			hessenberg_to_qz_unblocked(
				0,
				n - 1,
				A_clone.as_mut(),
				B_clone.as_mut(),
				Some(Q.as_mut()),
				Some(Z.as_mut()),
				alpha.as_mut(),
				beta.as_mut(),
				false,
			);

			assert!((&Q * &A_clone * Z.adjoint() - &A).norm_max() < 1e-13);

			assert!((&Q * &B_clone * Z.adjoint() - &B).norm_max() < 1e-13);

			for j in 0..n {
				for i in j + 1..n {
					assert!(B_clone[(i, j)] == c64::ZERO);
				}

				for i in j + 1..n {
					assert!(A_clone[(i, j)] == c64::ZERO);
				}
			}
		}
	}

	#[test]

	fn test_qz_cplx_blocked() {
		let rng = &mut StdRng::seed_from_u64(0);

		for n in [12, 35, 128, 255] {
			for _ in 0..10 {
				let rand = stats::CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: ComplexDistribution::new(StandardNormal, StandardNormal),
				};

				let mut sample = || -> Mat<c64> { rand.sample(rng) };

				let A = sample();

				let mut B = sample();

				zip!(&mut B).for_each_triangular_lower(linalg::zip::Diag::Skip, |unzip!(x)| {
					*x = c64::ZERO;
				});

				let B = B;

				let mut mem = MemBuffer::new(generalized_hessenberg_scratch::<c64>(
					n,
					GeneralizedHessenbergParams {
						block_size: 32,
						..auto!(c64)
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
						..auto!(c64)
					},
				);

				let mut alpha = Col::zeros(n);

				let mut beta = Col::zeros(n);

				hessenberg_to_qz_blocked(
					0,
					n - 1,
					A_clone.as_mut(),
					B_clone.as_mut(),
					Some(Q.as_mut()),
					Some(Z.as_mut()),
					alpha.as_mut(),
					beta.as_mut(),
					false,
					Par::Seq,
					auto!(c64),
					MemStack::new(&mut MemBuffer::new(hessenberg_to_qz_scratch::<c64>(n, Par::Seq, auto!(c64)))),
				);

				assert!((&Q * &A_clone * Z.adjoint() - &A).norm_max() < 1e-13);

				assert!((&Q * &B_clone * Z.adjoint() - &B).norm_max() < 1e-13);

				for j in 0..n {
					for i in j + 1..n {
						assert!(B_clone[(i, j)] == c64::ZERO);
					}

					for i in j + 1..n {
						assert!(A_clone[(i, j)] == c64::ZERO);
					}
				}
			}
		}
	}
}
