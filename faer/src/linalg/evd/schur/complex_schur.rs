// adapted from <T>LAPACK implementation
//
// https://github.com/tlapack/tlapack
// https://github.com/tlapack/tlapack/blob/master/include/tlapack/lapack/lahqr.hpp

use super::*;
use crate::{assert, debug_assert};
use linalg::householder::*;
use linalg::jacobi::JacobiRotation;
use linalg::matmul::matmul;

#[math]
fn lahqr_eig22<T: ComplexField>(a00: T, a01: T, a10: T, a11: T) -> (T, T) {
	let s = abs1(a00) + abs1(a01) + abs1(a10) + abs1(a11);
	if s == zero() {
		return (zero(), zero());
	}

	let half = from_f64::<T::Real>(0.5);
	let s_inv = recip(s);
	let a00 = mul_real(a00, s_inv);
	let a01 = mul_real(a01, s_inv);
	let a10 = mul_real(a10, s_inv);
	let a11 = mul_real(a11, s_inv);

	let tr = mul_real(a00 + a11, half);
	let det = (a00 - tr) * (a00 - tr) + (a01 * a10);

	let rtdisc = sqrt(det);
	(mul_real(tr + rtdisc, s), mul_real(tr - rtdisc, s))
}

#[math]
fn lahqr<T: ComplexField>(want_t: bool, A: MatMut<'_, T>, Z: Option<MatMut<'_, T>>, w: ColMut<'_, T>, ilo: usize, ihi: usize) -> isize {
	let n = A.nrows();
	let nh = ihi - ilo;

	let mut A = A;
	let mut Z = Z;
	let mut w = w;

	let eps = eps::<T::Real>();
	let smlnum = min_positive::<T::Real>() / eps;
	let non_convergence_limit = 10usize;
	let dat1 = from_f64::<T::Real>(0.75);
	let dat2 = from_f64::<T::Real>(-0.4375);

	if nh == 0 {
		return 0;
	}
	if nh == 1 {
		w[ilo] = A[(ilo, ilo)].clone();
	}

	let itmax = Ord::max(30, nbits::<T>() / 2).saturating_mul(Ord::max(10, nh)).saturating_mul(nh);

	// k_defl counts the number of iterations since a deflation
	let mut k_defl = 0usize;

	// istop is the end of the active subblock.
	// As more and more eigenvalues converge, it eventually
	// becomes ilo+1 and the loop ends.
	let mut istop = ihi;

	// istart is the start of the active subblock. Either
	// istart = ilo, or H(istart, istart-1) = 0. This means
	// that we can treat this subblock separately.
	let mut istart = ilo;

	for iter in 0..itmax {
		if iter + 1 == itmax {
			return istop as isize;
		}

		if istart + 1 >= istop {
			if istart + 1 == istop {
				w[istart] = A[(istart, istart)].clone();
			}
			break;
		}

		let istart_m;
		let istop_m;

		if want_t {
			istart_m = 0;
			istop_m = n;
		} else {
			istart_m = istart;
			istop_m = istop;
		}

		for i in (istart + 1..istop).rev() {
			if abs1(A[(i, i - 1)]) < smlnum {
				(A[(i, i - 1)] = zero());
				istart = i;
				break;
			}

			let mut tst = abs1(A[(i - 1, i - 1)]) + abs1(A[(i, i)]);

			if tst == zero() {
				if i >= ilo + 2 {
					tst = tst + abs1(A[(i - 1, i - 2)]);
				}
				if i + 1 < ihi {
					tst = tst + abs(A[(i + 1, i)]);
				}
			}

			if abs1(A[(i, i - 1)]) <= eps * tst {
				// The elementwise deflation test has passed
				// The following performs second deflation test due
				// to Ahues & Tisseur (LAWN 122, 1997). It has better
				// mathematical foundation and improves accuracy in some
				// examples.
				//
				// The test is |A(i,i-1)|*|A(i-1,i)| <=
				// eps*|A(i,i)|*|A(i-1,i-1)| The multiplications might overflow
				// so we do some scaling first.

				let ab = max(abs1(A[(i, i - 1)]), abs1(A[(i - 1, i)]));
				let ba = min(abs1(A[(i, i - 1)]), abs1(A[(i - 1, i)]));
				let aa = max(abs1(A[(i, i)]), abs1(A[(i, i)] - A[(i - 1, i - 1)]));
				let bb = min(abs1(A[(i, i)]), abs1(A[(i, i)] - A[(i - 1, i - 1)]));
				let s = aa + ab;
				if ba * (ab / s) <= max(smlnum, eps * (bb * (aa / s))) {
					(A[(i, i - 1)] = zero());
					istart = i;
					break;
				}
			}
		}

		if istart + 1 >= istop {
			k_defl = 0;
			w[istart] = A[(istart, istart)].clone();
			istop = istart;
			istart = ilo;
			continue;
		}

		// determine shift
		let (a00, a01, a10, a11);
		k_defl += 1;

		if k_defl % non_convergence_limit == 0 {
			// exceptional shift
			let mut s = abs(A[(istop - 1, istop - 2)]);

			if istop > ilo + 2 {
				s = s + abs(A[(istop - 2, istop - 3)]);
			}

			a00 = from_real::<T>(&mul(dat1, s)) + A[(istop - 1, istop - 1)];
			a10 = from_real::<T>(&mul(dat2, s));
			a01 = from_real::<T>(&s);
			a11 = copy(a00);
		} else {
			// wilkinson shift
			a00 = A[(istop - 2, istop - 2)].clone();
			a10 = A[(istop - 1, istop - 2)].clone();
			a01 = A[(istop - 2, istop - 1)].clone();
			a11 = A[(istop - 1, istop - 1)].clone();
		}

		let (mut s1, s2) = lahqr_eig22(a00, a01, a10, a11);
		if abs1(s1 - A[(istop - 1, istop - 1)]) > abs1(s2 - A[(istop - 1, istop - 1)]) {
			s1 = copy(s2);
		}

		// We have already checked whether the subblock has split.
		// If it has split, we can introduce any shift at the top of the new
		// subblock. Now that we know the specific shift, we can also check
		// whether we can introduce that shift somewhere else in the subblock.
		let mut istart2 = istart;
		if istart + 2 < istop {
			for i in (istart + 1..istop - 1).rev() {
				let h00 = A[(i, i)] - (s1);
				let h10 = A[(i + 1, i)].clone();

				let JacobiRotation { c: _, s: sn } = JacobiRotation::rotg(h00, h10).0;
				if abs1(conj(sn) * A[(i, i - 1)])
					<= mul(
						eps, //
						add(abs1(A[(i, i - 1)]), abs1(A[(i, i + 1)])),
					) {
					istart2 = i;
					break;
				}
			}
		}
		for i in istart2..istop - 1 {
			let (rot, r);

			if i == istart2 {
				let h00 = copy(A[(i, i)] - s1);
				let h10 = copy(A[(i + 1, i)]);

				(rot, _) = JacobiRotation::rotg(h00, h10);
				if i > istart {
					A[(i, i - 1)] = A[(i, i - 1)] * rot.c;
				}
			} else {
				(rot, r) = JacobiRotation::rotg(A[(i, i - 1)].clone(), A[(i + 1, i - 1)].clone());

				(A[(i, i - 1)] = copy(r));
				(A[(i + 1, i - 1)] = zero());
			}

			rot.adjoint()
				.apply_on_the_left_in_place(A.rb_mut().subcols_mut(i, istop_m - i).two_rows_mut(i, i + 1));

			rot.apply_on_the_right_in_place(A.rb_mut().get_mut(istart_m..Ord::min(i + 3, istop), ..).two_cols_mut(i, i + 1));

			if let Some(Z) = Z.rb_mut() {
				rot.apply_on_the_right_in_place(Z.two_cols_mut(i, i + 1));
			}
		}
	}

	0
}

#[math]
fn lahqr_shiftcolumn<T: ComplexField>(h: MatRef<'_, T>, v: ColMut<'_, T>, s1: T, s2: T) {
	let mut v = v;
	let n = h.nrows();
	if n == 2 {
		let s = abs1(h[(0, 0)] - s2) + abs1(h[(1, 0)]);
		if s == zero() {
			v[0] = zero();
			v[1] = zero();
		} else {
			let s_inv = recip(s);
			let h10s = mul_real(h[(1, 0)], s_inv);
			v[0] = h10s * h[(0, 1)] + (h[(0, 0)] - s1) * mul_real(h[(0, 0)] - s2, s_inv);
			v[1] = h10s * (h[(0, 0)] + h[(1, 1)] - s1 - s2);
		}
	} else {
		let s = abs1((h[(0, 0)] - s2)) + abs1(h[(1, 0)]) + abs1(h[(2, 0)]);
		if s == zero() {
			v[0] = zero();
			v[1] = zero();
			v[2] = zero();
		} else {
			let s_inv = recip(s);
			let h10s = mul_real(h[(1, 0)], s_inv);
			let h20s = mul_real(h[(2, 0)], s_inv);
			v[0] = (h[(0, 0)] - s1) * mul_real(h[(0, 0)] - s2, s_inv) + h[(0, 1)] * h10s + h[(0, 2)] * h20s;
			v[1] = h10s * (h[(0, 0)] + h[(1, 1)] - s1 - s2) + h[(1, 2)] * h20s;
			v[2] = h20s * (h[(0, 0)] + h[(2, 2)] - s1 - s2) + h10s * h[(2, 1)];
		}
	}
}
#[math]
fn aggressive_early_deflation<T: ComplexField>(
	want_t: bool,
	mut a: MatMut<'_, T>,
	mut z: Option<MatMut<'_, T>>,
	mut s: ColMut<'_, T>,
	ilo: usize,
	ihi: usize,
	nw: usize,
	par: Par,
	stack: &mut MemStack,
	params: SchurParams,
) -> (usize, usize) {
	let n = a.nrows();
	let nw_max = (n - 3) / 3;
	let eps = eps::<T::Real>();
	let n_T = from_f64::<T::Real>(n as f64);
	let small_num = min_positive::<T::Real>() / eps * n_T;
	let jw = Ord::min(Ord::min(nw, ihi - ilo), nw_max);
	let kwtop = ihi - jw;
	let mut s_spike = if kwtop == ilo { zero() } else { copy(a[(kwtop, kwtop - 1)]) };
	if kwtop + 1 == ihi {
		s[kwtop] = copy(a[(kwtop, kwtop)]);

		let mut ns = 1;
		let mut nd = 0;
		if abs1(s_spike) <= max(small_num, eps * abs1(a[(kwtop, kwtop)])) {
			ns = 0;
			nd = 1;
			if kwtop > ilo {
				a[(kwtop, kwtop - 1)] = zero();
			}
		}
		return (ns, nd);
	}
	let mut v = unsafe { a.rb().submatrix(n - jw, 0, jw, jw).const_cast() };
	let mut tw = unsafe { a.rb().submatrix(n - jw, jw, jw, jw).const_cast() };
	let mut wh = unsafe { a.rb().submatrix(n - jw, jw, jw, n - 2 * jw - 3).const_cast() };
	let mut wv = unsafe { a.rb().submatrix(jw + 3, 0, n - 2 * jw - 3, jw).const_cast() };
	let mut a = unsafe { a.rb().const_cast() };
	let a_window = a.rb().submatrix(kwtop, kwtop, ihi - kwtop, ihi - kwtop);
	let mut s_window = unsafe { s.rb().subrows(kwtop, ihi - kwtop).const_cast() };
	tw.fill(zero());
	for j in 0..jw {
		for i in 0..Ord::min(j + 2, jw) {
			tw[(i, j)] = copy(a_window[(i, j)]);
		}
	}
	v.fill(zero());
	v.rb_mut().diagonal_mut().fill(one());
	let infqr = if jw < params.blocking_threshold {
		lahqr(true, tw.rb_mut(), Some(v.rb_mut()), s_window.rb_mut(), 0, jw)
	} else {
		let infqr = multishift_qr(true, tw.rb_mut(), Some(v.rb_mut()), s_window.rb_mut(), 0, jw, par, stack, params).0;
		for j in 0..jw {
			for i in j + 2..jw {
				tw[(i, j)] = zero();
			}
		}
		infqr
	};
	let infqr = infqr as usize;
	let mut ns = jw;
	let nd;
	let mut ilst = infqr;
	while ilst < ns {
		#[allow(clippy::disallowed_names)]
		let mut foo = abs1(tw[(ns - 1, ns - 1)]);
		if foo == zero() {
			foo = abs1(s_spike);
		}
		if abs1(s_spike) * abs1(v[(0, ns - 1)]) <= max(small_num, eps * foo) {
			ns -= 1;
		} else {
			let ifst = ns - 1;
			schur_move(tw.rb_mut(), Some(v.rb_mut()), ifst, &mut ilst);
			ilst += 1;
		}
	}
	if ns == 0 {
		s_spike = zero();
	}
	if ns == jw {
		nd = jw - ns;
		ns -= infqr;
		return (ns, nd);
	}
	let mut sorted = false;
	let mut sorting_window_size = jw;
	while !sorted {
		sorted = true;
		let mut ilst = 0;
		let mut i1 = ns;
		while i1 + 1 < sorting_window_size {
			if i1 + 1 == jw {
				ilst -= 1;
				break;
			}
			let i2 = i1 + 1;
			let ev1 = abs1(tw[(i1, i1)]);
			let ev2 = abs1(tw[(i2, i2)]);
			if ev1 > ev2 {
				i1 = i2;
			} else {
				sorted = false;
				let ierr = schur_swap(tw.rb_mut(), Some(v.rb_mut()), i1);
				if ierr == 0 {
					i1 += 1;
				} else {
					i1 = i2;
				}
				ilst = i1;
			}
		}
		sorting_window_size = ilst;
	}
	let mut i = 0;
	while i < jw {
		s[kwtop + i] = copy(tw[(i, i)]);
		i += 1;
	}
	if s_spike != zero() {
		{
			let mut vv = wv.rb_mut().col_mut(0).subrows_mut(0, ns);
			for i in 0..ns {
				vv[i] = conj(v[(0, i)]);
			}
			let mut head = copy(vv[0]);
			let tail = vv.rb_mut().subrows_mut(1, ns - 1);
			let HouseholderInfo { tau, .. } = make_householder_in_place(&mut head, tail);
			vv[0] = one();
			let tau: T = recip(from_real(tau));
			{
				let mut tw_slice = tw.rb_mut().submatrix_mut(0, 0, ns, jw);
				let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
				let mut tmp = tmp.as_mat_mut().transpose_mut();
				matmul(tmp.rb_mut(), Accum::Replace, vv.rb().adjoint().as_mat(), tw_slice.rb(), one(), par);
				matmul(tw_slice.rb_mut(), Accum::Add, vv.rb().as_mat(), tmp.as_ref(), -tau, par);
			}
			{
				let mut tw_slice2 = tw.rb_mut().submatrix_mut(0, 0, jw, ns);
				let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
				let mut tmp = tmp.as_mat_mut();
				matmul(tmp.rb_mut(), Accum::Replace, tw_slice2.rb(), vv.rb().as_mat(), one(), par);
				matmul(tw_slice2.rb_mut(), Accum::Add, tmp.as_ref(), vv.rb().adjoint().as_mat(), -tau, par);
			}
			{
				let mut v_slice = v.rb_mut().submatrix_mut(0, 0, jw, ns);
				let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
				let mut tmp = tmp.as_mat_mut();
				matmul(tmp.rb_mut(), Accum::Replace, v_slice.rb(), vv.rb().as_mat(), one(), par);
				matmul(v_slice.rb_mut(), Accum::Add, tmp.as_ref(), vv.rb().adjoint().as_mat(), -tau, par);
			}
			vv[0] = head;
		}
		{
			let mut householder = wv.rb_mut().col_mut(0).subrows_mut(0, ns - 1);
			hessenberg::hessenberg_in_place(
				tw.rb_mut().submatrix_mut(0, 0, ns, ns),
				householder.rb_mut().as_mat_mut().transpose_mut(),
				par,
				stack,
				Default::default(),
			);
			apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
				tw.rb().submatrix(1, 0, ns - 1, ns - 1),
				householder.rb().transpose().as_mat(),
				Conj::Yes,
				unsafe { tw.rb().submatrix(1, ns, ns - 1, jw - ns).const_cast() },
				par,
				stack,
			);
			apply_block_householder_sequence_on_the_right_in_place_with_conj(
				tw.rb().submatrix(1, 0, ns - 1, ns - 1),
				householder.rb().transpose().as_mat(),
				Conj::No,
				v.rb_mut().submatrix_mut(0, 1, jw, ns - 1),
				par,
				stack,
			);
		}
	}
	if kwtop > 0 {
		a[(kwtop, kwtop - 1)] = s_spike * conj(v[(0, 0)]);
	}
	for j in 0..jw {
		for i in 0..Ord::min(j + 2, jw) {
			a[(kwtop + i, kwtop + j)] = copy(tw[(i, j)]);
		}
	}
	nd = jw - ns;
	ns -= infqr;
	let (istart_m, istop_m);
	if want_t {
		istart_m = 0;
		istop_m = n;
	} else {
		istart_m = ilo;
		istop_m = ihi;
	}
	if ihi < istop_m {
		let mut i = ihi;
		while i < istop_m {
			let iblock = Ord::min(istop_m - i, wh.ncols());
			let mut a_slice = a.rb_mut().submatrix_mut(kwtop, i, ihi - kwtop, iblock);
			let mut wh_slice = wh.rb_mut().submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
			matmul(wh_slice.rb_mut(), Accum::Replace, v.rb().adjoint(), a_slice.rb(), one(), par);
			a_slice.copy_from(wh_slice.rb());
			i += iblock;
		}
	}
	if istart_m < kwtop {
		let mut i = istart_m;
		while i < kwtop {
			let iblock = Ord::min(kwtop - i, wv.nrows());
			let mut a_slice = a.rb_mut().submatrix_mut(i, kwtop, iblock, ihi - kwtop);
			let mut wv_slice = wv.rb_mut().submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
			matmul(wv_slice.rb_mut(), Accum::Replace, a_slice.rb(), v.rb(), one(), par);
			a_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	if let Some(mut z) = z.rb_mut() {
		let mut i = 0;
		while i < n {
			let iblock = Ord::min(n - i, wv.nrows());
			let mut z_slice = z.rb_mut().submatrix_mut(i, kwtop, iblock, ihi - kwtop);
			let mut wv_slice = wv.rb_mut().submatrix_mut(0, 0, z_slice.nrows(), z_slice.ncols());
			matmul(wv_slice.rb_mut(), Accum::Replace, z_slice.rb(), v.rb(), one(), par);
			z_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	(ns, nd)
}

fn schur_move<T: ComplexField>(mut a: MatMut<'_, T>, mut q: Option<MatMut<'_, T>>, ifst: usize, ilst: &mut usize) -> isize {
	let n = a.nrows();

	// Quick return
	if n == 0 {
		return 0;
	}

	let mut here = ifst;
	if ifst < *ilst {
		while here != *ilst {
			// Size of the next eigenvalue block
			let ierr = schur_swap(a.rb_mut(), q.rb_mut(), here);
			if ierr != 0 {
				// The swap failed, return with error
				*ilst = here;
				return 1;
			}
			here += 1;
		}
	} else {
		while here != *ilst {
			// Size of the next eigenvalue block
			let ierr = schur_swap(a.rb_mut(), q.rb_mut(), here - 1);
			if ierr != 0 {
				// The swap failed, return with error
				*ilst = here;
				return 1;
			}
			here -= 1;
		}
	}

	0
}

#[math]
pub(crate) fn schur_swap<T: ComplexField>(mut a: MatMut<'_, T>, q: Option<MatMut<'_, T>>, j0: usize) -> isize {
	let n = a.nrows();
	let j1 = j0 + 1;
	let j2 = j0 + 2;
	let t00 = copy(a[(j0, j0)]);
	let t11 = copy(a[(j1, j1)]);
	let (rot, _) = JacobiRotation::<T>::rotg(copy(a[(j0, j1)]), t11 - t00);
	a[(j1, j1)] = t00;
	a[(j0, j0)] = t11;
	if j2 < n {
		rot.adjoint()
			.apply_on_the_left_in_place(a.rb_mut().get_mut(.., j2..).two_rows_mut(j0, j1));
	}
	if j0 > 0 {
		rot.apply_on_the_right_in_place(a.rb_mut().get_mut(..j0, ..).two_cols_mut(j0, j1));
	}
	if let Some(q) = q {
		rot.apply_on_the_right_in_place(q.two_cols_mut(j0, j1));
	}
	0
}
/// returns err code, number of aggressive early deflations, number of qr sweeps
#[math]
pub fn multishift_qr<T: ComplexField>(
	want_t: bool,
	a: MatMut<'_, T>,
	z: Option<MatMut<'_, T>>,
	w: ColMut<'_, T>,
	ilo: usize,
	ihi: usize,
	par: Par,
	stack: &mut MemStack,
	params: SchurParams,
) -> (isize, usize, usize) {
	assert!(a.nrows() == a.ncols());
	assert!(ilo <= ihi);
	let n = a.nrows();
	let nh = ihi - ilo;
	assert!(w.nrows() == n);
	assert!(w.ncols() == 1);
	if let Some(z) = z.rb() {
		assert!(z.nrows() == n);
		assert!(z.ncols() == n);
	}
	let mut a = a;
	let mut z = z;
	let mut w = w;
	let mut stack = stack;
	let non_convergence_limit_window = 5;
	let non_convergence_limit_shift = 6;
	let dat1 = from_f64::<T::Real>(0.75);
	let dat2 = from_f64::<T::Real>(-0.4375);
	let nmin = Ord::max(15, params.blocking_threshold);
	let nibble = params.nibble_threshold;
	let nsr = (params.recommended_shift_count)(n, nh);
	let nsr = Ord::min(Ord::min(nsr, (n.saturating_sub(3)) / 6), ihi - ilo - 1);
	let nsr = Ord::max(nsr / 2 * 2, 2);
	let nwr = (params.recommended_deflation_window)(n, nh);
	let nwr = Ord::max(nwr, 2);
	let nwr = Ord::min(Ord::min(nwr, (n.saturating_sub(1)) / 3), ihi - ilo);
	if n < nmin {
		let err = lahqr(want_t, a, z, w, ilo, ihi);
		return (err, 0, 0);
	}
	if nh == 0 {
		return (0, 0, 0);
	}
	let nw_max = (n - 3) / 3;
	let itmax = 30 * Ord::max(10, nh);
	let mut k_defl = 0;
	let mut istop = ihi;
	let mut info = 0;
	let mut nw = 0;
	let mut count_aed = 0;
	let mut count_sweep = 0;
	for iter in 0..itmax + 1 {
		if iter == itmax {
			info = istop as isize;
			break;
		}
		if ilo + 1 >= istop {
			if ilo + 1 == istop {
				w[ilo] = copy(a[(ilo, ilo)]);
			}
			break;
		}
		let mut istart = ilo;
		for i in (ilo + 1..istop).rev() {
			if a[(i, i - 1)] == zero() {
				istart = i;
				break;
			}
		}
		let nh = istop - istart;
		let nwupbd = Ord::min(nh, nw_max);
		if k_defl < non_convergence_limit_window {
			nw = Ord::min(nwupbd, nwr);
		} else {
			nw = Ord::min(nwupbd, 2 * nw);
		}
		if nh <= 4 {
			nw = nh;
		}
		if nw < nw_max {
			if nw + 1 >= nh {
				nw = nh
			};
			let kwtop = istop - nw;
			if (kwtop > istart + 2) && (abs1(a[(kwtop, kwtop - 1)]) > abs1(a[(kwtop - 1, kwtop - 2)])) {
				nw += 1;
			}
		}
		let (ls, ld) = aggressive_early_deflation(want_t, a.rb_mut(), z.rb_mut(), w.rb_mut(), istart, istop, nw, par, stack.rb_mut(), params);
		count_aed += 1;
		istop -= ld;
		if ld > 0 {
			k_defl = 0;
		}
		if ld > 0 && (100 * ld > nwr * nibble || (istop - istart) <= Ord::min(nmin, nw_max)) {
			continue;
		}
		k_defl += 1;
		let mut ns = Ord::min(nh - 1, Ord::min(Ord::max(2, ls), nsr));
		ns = ns / 2 * 2;
		let mut i_shifts = istop - ns;
		if k_defl % non_convergence_limit_shift == 0 {
			for i in (i_shifts + 1..istop).rev().step_by(2) {
				if i >= ilo + 2 {
					let ss = abs1(a[(i, i - 1)]) + abs1(a[(i - 1, i - 2)]);
					let aa = from_real::<T>(&(dat1 * ss)) + a[(i, i)];
					let bb = from_real::<T>(&ss);
					let cc = from_real::<T>(&(dat2 * ss));
					let dd = copy(aa);
					let (s1, s2) = lahqr_eig22(aa, bb, cc, dd);
					w[i - 1] = s1;
					w[i] = s2;
				} else {
					w[i - 1] = copy(a[(i, i)]);
					w[i] = copy(a[(i, i)]);
				}
			}
		} else {
			if ls <= ns / 2 {
				let mut temp = a.rb_mut().submatrix_mut(n - ns, 0, ns, ns);
				let mut shifts = w.rb_mut().subrows_mut(istop - ns, ns);
				let ierr = lahqr(false, temp.rb_mut(), None, shifts.rb_mut(), 0, ns) as usize;
				ns = ns - ierr;
				if ns < 2 {
					let aa = copy(a[(istop - 2, istop - 2)]);
					let bb = copy(a[(istop - 2, istop - 1)]);
					let cc = copy(a[(istop - 1, istop - 2)]);
					let dd = copy(a[(istop - 1, istop - 1)]);
					let (s1, s2) = lahqr_eig22(aa, bb, cc, dd);
					w[istop - 2] = s1;
					w[istop - 1] = s2;
					ns = 2;
				}
				i_shifts = istop - ns;
			}
			let mut sorted = false;
			let mut k = istop;
			while !sorted && k > i_shifts {
				sorted = true;
				for i in i_shifts..k - 1 {
					if abs1(w[i]) < abs1(w[i + 1]) {
						sorted = false;
						let wi = copy(w[i]);
						let wip1 = copy(w[i + 1]);
						w[i] = wip1;
						w[i + 1] = wi;
					}
				}
				k -= 1;
			}
			for i in (i_shifts + 2..istop).rev().step_by(2) {
				if imag(w[i]) != -imag(w[i - 1]) {
					let tmp = copy(w[i]);
					w[i] = copy(w[i - 1]);
					w[i - 1] = copy(w[i - 2]);
					w[i - 2] = tmp;
				}
			}
			if ns % 2 == 1 {
				ns -= 1;
			}
			i_shifts = istop - ns;
		}
		let mut shifts = w.rb_mut().subrows_mut(i_shifts, ns);
		multishift_qr_sweep(want_t, a.rb_mut(), z.rb_mut(), shifts.rb_mut(), istart, istop, par, stack);
		count_sweep += 1;
	}
	(info, count_aed, count_sweep)
}
#[math]
fn move_bulge<T: ComplexField>(mut h: MatMut<'_, T>, mut v: ColMut<'_, T>, s1: T, s2: T) {
	let v0 = real(v[0]);
	let v1 = copy(v[1]);
	let v2 = copy(v[2]);
	let refsum = mul_real(v2, v0) * h[(3, 2)];
	let epsilon = eps::<T::Real>();
	h[(3, 0)] = -refsum;
	h[(3, 1)] = -refsum * conj(v1);
	h[(3, 2)] = h[(3, 2)] - refsum * conj(v2);
	v[0] = copy(h[(1, 0)]);
	v[1] = copy(h[(2, 0)]);
	v[2] = copy(h[(3, 0)]);
	let mut beta = copy(v[0]);
	let tail = v.rb_mut().subrows_mut(1, 2);
	let HouseholderInfo { tau, .. } = make_householder_in_place(&mut beta, tail);
	v[0] = from_real(recip(tau));
	if h[(3, 0)] != zero() || h[(3, 1)] != zero() || h[(3, 2)] != zero() {
		h[(1, 0)] = beta;
		h[(2, 0)] = zero();
		h[(3, 0)] = zero();
	} else {
		stack_mat!(vt, 3, 1, T);
		let mut vt = vt.rb_mut().col_mut(0);
		let h2 = h.rb().submatrix(1, 1, 3, 3);
		lahqr_shiftcolumn(h2, vt.rb_mut(), s1, s2);
		let mut beta_unused = copy(vt[0]);
		let tail = vt.rb_mut().subrows_mut(1, 2);
		let HouseholderInfo { tau, .. } = make_householder_in_place(&mut beta_unused, tail);
		vt[0] = from_real(recip(tau));
		let vt0 = copy(vt[0]);
		let vt1 = copy(vt[1]);
		let vt2 = copy(vt[2]);
		let refsum = conj(vt0) * h[(1, 0)] + conj(vt1) * h[(2, 0)];
		if abs1(sub(h[(2, 0)], mul(refsum, vt1))) + abs1(mul(refsum, vt2)) > epsilon * (abs1(h[(0, 0)]) + abs1(h[(1, 1)]) + abs1(h[(2, 2)])) {
			h[(1, 0)] = beta;
			h[(2, 0)] = zero();
			h[(3, 0)] = zero();
		} else {
			h[(1, 0)] = h[(1, 0)] - refsum;
			h[(2, 0)] = zero();
			h[(3, 0)] = zero();
			v[0] = copy(vt[0]);
			v[1] = copy(vt[1]);
			v[2] = copy(vt[2]);
		}
	}
}
#[math]
fn multishift_qr_sweep<T: ComplexField>(
	want_t: bool,
	a: MatMut<'_, T>,
	mut z: Option<MatMut<'_, T>>,
	s: ColMut<'_, T>,
	ilo: usize,
	ihi: usize,
	par: Par,
	stack: &mut MemStack,
) {
	let n = a.nrows();
	assert!(n >= 12);
	let (mut v, _stack) = temp_mat_zeroed(3, s.nrows() / 2, stack);
	let mut v = v.as_mat_mut();
	let n_block_max = (n - 3) / 3;
	let n_shifts_max = Ord::min(ihi - ilo - 1, Ord::max(2, 3 * (n_block_max / 4)));
	let mut n_shifts = Ord::min(s.nrows(), n_shifts_max);
	if n_shifts % 2 == 1 {
		n_shifts -= 1;
	}
	let n_bulges = n_shifts / 2;
	let n_block_desired = Ord::min(2 * n_shifts, n_block_max);
	let mut u = unsafe { a.rb().submatrix(n - n_block_desired, 0, n_block_desired, n_block_desired).const_cast() };
	let mut wh = unsafe {
		a.rb()
			.submatrix(n - n_block_desired, n_block_desired, n_block_desired, n - 2 * n_block_desired - 3)
			.const_cast()
	};
	let mut wv = unsafe {
		a.rb()
			.submatrix(n_block_desired + 3, 0, n - 2 * n_block_desired - 3, n_block_desired)
			.const_cast()
	};
	let mut a = unsafe { a.rb().const_cast() };
	let mut i_pos_block = 0;
	introduce_bulges(
		ilo,
		ihi,
		n_block_desired,
		n_bulges,
		n_shifts,
		want_t,
		a.rb_mut(),
		z.rb_mut(),
		u.rb_mut(),
		v.rb_mut(),
		wh.rb_mut(),
		wv.rb_mut(),
		s.rb(),
		&mut i_pos_block,
		par,
	);
	move_bulges_down(
		ilo,
		ihi,
		n_block_desired,
		n_bulges,
		n_shifts,
		want_t,
		a.rb_mut(),
		z.rb_mut(),
		u.rb_mut(),
		v.rb_mut(),
		wh.rb_mut(),
		wv.rb_mut(),
		s.rb(),
		&mut i_pos_block,
		par,
	);
	remove_bulges(
		ilo,
		ihi,
		n_bulges,
		n_shifts,
		want_t,
		a.rb_mut(),
		z.rb_mut(),
		u.rb_mut(),
		v.rb_mut(),
		wh.rb_mut(),
		wv.rb_mut(),
		s.rb(),
		&mut i_pos_block,
		par,
	);
}
#[inline(never)]
#[math]
fn introduce_bulges<T: ComplexField>(
	ilo: usize,
	ihi: usize,
	n_block_desired: usize,
	n_bulges: usize,
	n_shifts: usize,
	want_t: bool,
	mut a: MatMut<'_, T>,
	mut z: Option<MatMut<'_, T>>,
	mut u: MatMut<'_, T>,
	mut v: MatMut<'_, T>,
	mut wh: MatMut<'_, T>,
	mut wv: MatMut<'_, T>,
	s: ColRef<'_, T>,
	i_pos_block: &mut usize,
	par: Par,
) {
	let n = a.nrows();
	let eps = eps::<T::Real>();
	let n_T = from_f64::<T::Real>(n as f64);
	let small_num = min_positive::<T::Real>() / eps * n_T;
	let n_block = Ord::min(n_block_desired, ihi - ilo);
	let mut istart_m = ilo;
	let mut istop_m = ilo + n_block;
	let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
	u2.fill(zero());
	u2.rb_mut().diagonal_mut().fill(one());
	for i_pos_last in ilo..ilo + n_block - 2 {
		let n_active_bulges = Ord::min(n_bulges, ((i_pos_last - ilo) / 2) + 1);
		for i_bulge in 0..n_active_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let mut v = v.rb_mut().col_mut(i_bulge);
			if i_pos == ilo {
				let h = a.rb().submatrix(ilo, ilo, 3, 3);
				let s1 = copy(s[s.nrows() - 1 - 2 * i_bulge]);
				let s2 = copy(s[s.nrows() - 1 - 2 * i_bulge - 1]);
				lahqr_shiftcolumn(h, v.rb_mut(), s1, s2);
				debug_assert!(v.nrows() == 3);
				let mut beta = copy(v[0]);
				let tail = v.rb_mut().subrows_mut(1, 2);
				let HouseholderInfo { tau, .. } = make_householder_in_place(&mut beta, tail);
				v[0] = from_real(recip(tau));
			} else {
				let mut h = a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
				let s1 = copy(s[s.nrows() - 1 - 2 * i_bulge]);
				let s2 = copy(s[s.nrows() - 1 - 2 * i_bulge - 1]);
				move_bulge(h.rb_mut(), v.rb_mut(), s1, s2);
			}
			let v0 = real(v[0]);
			let v1 = copy(v[1]);
			let v2 = copy(v[2]);
			for j in istart_m..i_pos + 3 {
				let sum = a[(j, i_pos)] + v1 * a[(j, i_pos + 1)] + v2 * a[(j, i_pos + 2)];
				a[(j, i_pos)] = a[(j, i_pos)] - mul_real(sum, v0);
				a[(j, i_pos + 1)] = a[(j, i_pos + 1)] - mul_real(sum, v0) * conj(v1);
				a[(j, i_pos + 2)] = a[(j, i_pos + 2)] - mul_real(sum, v0) * conj(v2);
			}
			let sum = a[(i_pos, i_pos)] + conj(v1) * a[(i_pos + 1, i_pos)] + conj(v2) * a[(i_pos + 2, i_pos)];
			a[(i_pos, i_pos)] = a[(i_pos, i_pos)] - mul_real(sum, v0);
			a[(i_pos + 1, i_pos)] = a[(i_pos + 1, i_pos)] - mul_real(sum, v0) * v1;
			a[(i_pos + 2, i_pos)] = a[(i_pos + 2, i_pos)] - mul_real(sum, v0) * v2;
			if (i_pos > ilo) && (a[(i_pos, i_pos - 1)] != zero()) {
				let mut tst1 = abs1(a[(i_pos - 1, i_pos - 1)]) + abs1(a[(i_pos, i_pos)]);
				if tst1 == zero() {
					if i_pos > ilo + 1 {
						tst1 = tst1 + abs1(a[(i_pos - 1, i_pos - 2)]);
					}
					if i_pos > ilo + 2 {
						tst1 = tst1 + abs1(a[(i_pos - 1, i_pos - 3)]);
					}
					if i_pos > ilo + 3 {
						tst1 = tst1 + abs1(a[(i_pos - 1, i_pos - 4)]);
					}
					if i_pos < ihi - 1 {
						tst1 = tst1 + abs1(a[(i_pos + 1, i_pos)]);
					}
					if i_pos < ihi - 2 {
						tst1 = tst1 + abs1(a[(i_pos + 2, i_pos)]);
					}
					if i_pos < ihi - 3 {
						tst1 = tst1 + abs1(a[(i_pos + 3, i_pos)]);
					}
				}
				if abs1(a[(i_pos, i_pos - 1)]) < max(small_num, eps * tst1) {
					let ab = max(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
					let ba = min(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
					let aa = max(abs1(a[(i_pos, i_pos)]), abs1((a[(i_pos, i_pos)] - a[(i_pos - 1, i_pos - 1)])));
					let bb = min(abs1(a[(i_pos, i_pos)]), abs1((a[(i_pos, i_pos)] - a[(i_pos - 1, i_pos - 1)])));
					let s = aa + ab;
					if ba * (ab / s) <= max(small_num, eps * (bb * (aa / s))) {
						a[(i_pos, i_pos - 1)] = zero();
					}
				}
			}
		}
		for i_bulge in 0..n_active_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let v = v.rb_mut().col_mut(i_bulge);
			let v0 = real(v[0]);
			let v1 = copy(v[1]);
			let v2 = copy(v[2]);
			for j in i_pos + 1..istop_m {
				let sum = a[(i_pos, j)] + conj(v1) * a[(i_pos + 1, j)] + conj(v2) * a[(i_pos + 2, j)];
				a[(i_pos, j)] = a[(i_pos, j)] - mul_real(sum, v0);
				a[(i_pos + 1, j)] = a[(i_pos + 1, j)] - mul_real(sum, v0) * v1;
				a[(i_pos + 2, j)] = a[(i_pos + 2, j)] - mul_real(sum, v0) * v2;
			}
		}
		for i_bulge in 0..n_active_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let v = v.rb_mut().col_mut(i_bulge);
			let v0 = real(v[0]);
			let v1 = copy(v[1]);
			let v2 = copy(v[2]);
			let i1 = 0;
			let i2 = Ord::min(u2.nrows(), (i_pos_last - ilo) + (i_pos_last - ilo) + 3);
			for j in i1..i2 {
				let sum = u2[(j, i_pos - ilo)] + v1 * u2[(j, i_pos - ilo + 1)] + v2 * u2[(j, i_pos - ilo + 2)];
				u2[(j, i_pos - ilo)] = u2[(j, i_pos - ilo)] - mul_real(sum, v0);
				u2[(j, i_pos - ilo + 1)] = u2[(j, i_pos - ilo + 1)] - mul_real(sum, v0) * conj(v1);
				u2[(j, i_pos - ilo + 2)] = u2[(j, i_pos - ilo + 2)] - mul_real(sum, v0) * conj(v2);
			}
		}
	}
	if want_t {
		istart_m = 0;
		istop_m = n;
	} else {
		istart_m = ilo;
		istop_m = ihi;
	}
	if ilo + n_shifts + 1 < istop_m {
		let mut i = ilo + n_block;
		while i < istop_m {
			let iblock = Ord::min(istop_m - i, wh.ncols());
			let mut a_slice = a.rb_mut().submatrix_mut(ilo, i, n_block, iblock);
			let mut wh_slice = wh.rb_mut().submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
			matmul(wh_slice.rb_mut(), Accum::Replace, u2.rb().adjoint(), a_slice.rb(), one(), par);
			a_slice.copy_from(wh_slice.rb());
			i += iblock;
		}
	}
	if istart_m < ilo {
		let mut i = istart_m;
		while i < ilo {
			let iblock = Ord::min(ilo - i, wv.nrows());
			let mut a_slice = a.rb_mut().submatrix_mut(i, ilo, iblock, n_block);
			let mut wv_slice = wv.rb_mut().submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
			matmul(wv_slice.rb_mut(), Accum::Replace, a_slice.rb(), u2.rb(), one(), par);
			a_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	if let Some(mut z) = z.rb_mut() {
		let mut i = 0;
		while i < n {
			let iblock = Ord::min(n - i, wv.nrows());
			let mut z_slice = z.rb_mut().submatrix_mut(i, ilo, iblock, n_block);
			let mut wv_slice = wv.rb_mut().submatrix_mut(0, 0, z_slice.nrows(), z_slice.ncols());
			matmul(wv_slice.rb_mut(), Accum::Replace, z_slice.rb(), u2.rb(), one(), par);
			z_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	*i_pos_block = ilo + n_block - n_shifts;
}
#[inline(never)]
#[math]
fn move_bulges_down<T: ComplexField>(
	ilo: usize,
	ihi: usize,
	n_block_desired: usize,
	n_bulges: usize,
	n_shifts: usize,
	want_t: bool,
	mut a: MatMut<'_, T>,
	mut z: Option<MatMut<'_, T>>,
	mut u: MatMut<'_, T>,
	mut v: MatMut<'_, T>,
	mut wh: MatMut<'_, T>,
	mut wv: MatMut<'_, T>,
	s: ColRef<'_, T>,
	i_pos_block: &mut usize,
	par: Par,
) {
	let n = a.nrows();
	let eps = eps::<T::Real>();
	let n_T = from_f64::<T::Real>(n as f64);
	let small_num = min_positive::<T::Real>() / eps * n_T;
	while *i_pos_block + n_block_desired < ihi {
		let n_pos = Ord::min(n_block_desired - n_shifts, ihi - n_shifts - 1 - *i_pos_block);
		let n_block = n_shifts + n_pos;
		let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
		u2.fill(zero());
		u2.rb_mut().diagonal_mut().fill(one());
		let mut istart_m = *i_pos_block;
		let mut istop_m = *i_pos_block + n_block;
		for i_pos_last in *i_pos_block + n_shifts - 2..*i_pos_block + n_shifts - 2 + n_pos {
			for i_bulge in 0..n_bulges {
				let i_pos = i_pos_last - 2 * i_bulge;
				let mut v = v.rb_mut().col_mut(i_bulge);
				let mut h = a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
				let s1 = copy(s[s.nrows() - 1 - 2 * i_bulge]);
				let s2 = copy(s[s.nrows() - 1 - 2 * i_bulge - 1]);
				move_bulge(h.rb_mut(), v.rb_mut(), s1, s2);
				let v0 = real(v[0]);
				let v1 = copy(v[1]);
				let v2 = copy(v[2]);
				for j in istart_m..i_pos + 3 {
					let sum = a[(j, i_pos)] + v1 * a[(j, i_pos + 1)] + v2 * a[(j, i_pos + 2)];
					a[(j, i_pos)] = a[(j, i_pos)] - mul_real(sum, v0);
					a[(j, i_pos + 1)] = a[(j, i_pos + 1)] - mul_real(sum, v0) * conj(v1);
					a[(j, i_pos + 2)] = a[(j, i_pos + 2)] - mul_real(sum, v0) * conj(v2);
				}
				let sum = a[(i_pos, i_pos)] + conj(v1) * a[(i_pos + 1, i_pos)] + conj(v2) * a[(i_pos + 2, i_pos)];
				a[(i_pos, i_pos)] = a[(i_pos, i_pos)] - mul_real(sum, v0);
				a[(i_pos + 1, i_pos)] = a[(i_pos + 1, i_pos)] - mul_real(sum, v0) * v1;
				a[(i_pos + 2, i_pos)] = a[(i_pos + 2, i_pos)] - mul_real(sum, v0) * v2;
				if (i_pos > ilo) && (a[(i_pos, i_pos - 1)] != zero()) {
					let mut tst1 = abs1(a[(i_pos - 1, i_pos - 1)]) + abs1(a[(i_pos, i_pos)]);
					if tst1 == zero() {
						if i_pos > ilo + 1 {
							tst1 = tst1 + abs1(a[(i_pos - 1, i_pos - 2)]);
						}
						if i_pos > ilo + 2 {
							tst1 = tst1 + abs1(a[(i_pos - 1, i_pos - 3)]);
						}
						if i_pos > ilo + 3 {
							tst1 = tst1 + abs1(a[(i_pos - 1, i_pos - 4)]);
						}
						if i_pos < ihi - 1 {
							tst1 = tst1 + abs1(a[(i_pos + 1, i_pos)]);
						}
						if i_pos < ihi - 2 {
							tst1 = tst1 + abs1(a[(i_pos + 2, i_pos)]);
						}
						if i_pos < ihi - 3 {
							tst1 = tst1 + abs1(a[(i_pos + 3, i_pos)]);
						}
					}
					if abs1(a[(i_pos, i_pos - 1)]) < max(small_num, eps * tst1) {
						let ab = max(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
						let ba = min(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
						let aa = max(abs1(a[(i_pos, i_pos)]), abs1((a[(i_pos, i_pos)] - a[(i_pos - 1, i_pos - 1)])));
						let bb = min(abs1(a[(i_pos, i_pos)]), abs1((a[(i_pos, i_pos)] - a[(i_pos - 1, i_pos - 1)])));
						let s = aa + ab;
						if ba * (ab / s) <= max(small_num, eps * (bb * (aa / s))) {
							a[(i_pos, i_pos - 1)] = zero();
						}
					}
				}
			}
			for i_bulge in 0..n_bulges {
				let i_pos = i_pos_last - 2 * i_bulge;
				let v = v.rb_mut().col_mut(i_bulge);
				let v0 = real(v[0]);
				let v1 = copy(v[1]);
				let v2 = copy(v[2]);
				for j in i_pos + 1..istop_m {
					let sum = a[(i_pos, j)] + conj(v1) * a[(i_pos + 1, j)] + conj(v2) * a[(i_pos + 2, j)];
					a[(i_pos, j)] = a[(i_pos, j)] - mul_real(sum, v0);
					a[(i_pos + 1, j)] = a[(i_pos + 1, j)] - (mul_real(sum, v0) * v1);
					a[(i_pos + 2, j)] = a[(i_pos + 2, j)] - (mul_real(sum, v0) * v2);
				}
			}
			for i_bulge in 0..n_bulges {
				let i_pos = i_pos_last - 2 * i_bulge;
				let v = v.rb_mut().col_mut(i_bulge);
				let v0 = real(v[0]);
				let v1 = copy(v[1]);
				let v2 = copy(v[2]);
				let i1 = (i_pos - *i_pos_block) - (i_pos_last + 2 - *i_pos_block - n_shifts);
				let i2 = Ord::min(u2.nrows(), (i_pos_last - *i_pos_block) + (i_pos_last + 2 - *i_pos_block - n_shifts) + 3);
				for j in i1..i2 {
					let sum = u2[(j, i_pos - *i_pos_block)] + v1 * u2[(j, i_pos - *i_pos_block + 1)] + v2 * u2[(j, i_pos - *i_pos_block + 2)];
					u2[(j, i_pos - *i_pos_block)] = u2[(j, i_pos - *i_pos_block)] - mul_real(sum, v0);
					u2[(j, i_pos - *i_pos_block + 1)] = u2[(j, i_pos - *i_pos_block + 1)] - (mul_real(sum, v0) * conj(v1));
					u2[(j, i_pos - *i_pos_block + 2)] = u2[(j, i_pos - *i_pos_block + 2)] - (mul_real(sum, v0) * conj(v2));
				}
			}
		}
		if want_t {
			istart_m = 0;
			istop_m = n;
		} else {
			istart_m = ilo;
			istop_m = ihi;
		}
		if *i_pos_block + n_block < istop_m {
			let mut i = *i_pos_block + n_block;
			while i < istop_m {
				let iblock = Ord::min(istop_m - i, wh.ncols());
				let mut a_slice = a.rb_mut().submatrix_mut(*i_pos_block, i, n_block, iblock);
				let mut wh_slice = wh.rb_mut().submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
				matmul(wh_slice.rb_mut(), Accum::Replace, u2.rb().adjoint(), a_slice.rb(), one(), par);
				a_slice.copy_from(wh_slice.rb());
				i += iblock;
			}
		}
		if istart_m < *i_pos_block {
			let mut i = istart_m;
			while i < *i_pos_block {
				let iblock = Ord::min(*i_pos_block - i, wv.nrows());
				let mut a_slice = a.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
				let mut wv_slice = wv.rb_mut().submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
				matmul(wv_slice.rb_mut(), Accum::Replace, a_slice.rb(), u2.rb(), one(), par);
				a_slice.copy_from(wv_slice.rb());
				i += iblock;
			}
		}
		if let Some(mut z) = z.rb_mut() {
			let mut i = 0;
			while i < n {
				let iblock = Ord::min(n - i, wv.nrows());
				let mut z_slice = z.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
				let mut wv_slice = wv.rb_mut().submatrix_mut(0, 0, z_slice.nrows(), z_slice.ncols());
				matmul(wv_slice.rb_mut(), Accum::Replace, z_slice.rb(), u2.rb(), one(), par);
				z_slice.copy_from(wv_slice.rb());
				i += iblock;
			}
		}
		*i_pos_block += n_pos;
	}
}
#[inline(never)]
#[math]
fn remove_bulges<T: ComplexField>(
	ilo: usize,
	ihi: usize,
	n_bulges: usize,
	n_shifts: usize,
	want_t: bool,
	mut a: MatMut<'_, T>,
	mut z: Option<MatMut<'_, T>>,
	mut u: MatMut<'_, T>,
	mut v: MatMut<'_, T>,
	mut wh: MatMut<'_, T>,
	mut wv: MatMut<'_, T>,
	s: ColRef<'_, T>,
	i_pos_block: &mut usize,
	par: Par,
) {
	let n = a.nrows();
	let eps = eps::<T::Real>();
	let small_num = min_positive::<T::Real>() / eps * from_f64::<T::Real>(n as f64);
	let n_block = ihi - *i_pos_block;
	let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
	u2.fill(zero());
	u2.rb_mut().diagonal_mut().fill(one());
	let mut istart_m = *i_pos_block;
	let mut istop_m = ihi;
	for i_pos_last in *i_pos_block + n_shifts - 2..ihi + n_shifts - 1 {
		let mut i_bulge_start = if i_pos_last + 3 > ihi { (i_pos_last + 3 - ihi) / 2 } else { 0 };
		for i_bulge in i_bulge_start..n_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			if i_pos == ihi - 2 {
				let mut v = v.rb_mut().subrows_mut(0, 2).col_mut(i_bulge);
				let mut h = a.rb_mut().subrows_mut(i_pos, 2).col_mut(i_pos - 1);
				let mut beta = copy(h[0]);
				let tail = h.rb_mut().subrows_mut(1, 1);
				let HouseholderInfo { tau, .. } = make_householder_in_place(&mut beta, tail);
				v[0] = from_real(recip(tau));
				v[1] = copy(h[1]);
				h[0] = beta;
				h[1] = zero();
				let t0 = conj(v[0]);
				let v1 = copy(v[1]);
				let t1 = t0 * v1;
				for j in istart_m..i_pos + 2 {
					let sum = a[(j, i_pos)] + (v1 * a[(j, i_pos + 1)]);
					a[(j, i_pos)] = a[(j, i_pos)] - sum * conj(t0);
					a[(j, i_pos + 1)] = a[(j, i_pos + 1)] - sum * conj(t1);
				}
				for j in i_pos..istop_m {
					let sum = a[(i_pos, j)] + (conj(v1) * a[(i_pos + 1, j)]);
					a[(i_pos, j)] = a[(i_pos, j)] - sum * t0;
					a[(i_pos + 1, j)] = a[(i_pos + 1, j)] - sum * t1;
				}
				for j in 0..u2.nrows() {
					let sum = u2[(j, i_pos - *i_pos_block)] + v1 * u2[(j, i_pos - *i_pos_block + 1)];
					u2[(j, i_pos - *i_pos_block)] = u2[(j, i_pos - *i_pos_block)] - sum * conj(t0);
					u2[(j, i_pos - *i_pos_block + 1)] = u2[(j, i_pos - *i_pos_block + 1)] - sum * conj(t1);
				}
			} else {
				let mut v = v.rb_mut().col_mut(i_bulge);
				let mut h = a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
				let s1 = copy(s[s.nrows() - 1 - 2 * i_bulge]);
				let s2 = copy(s[s.nrows() - 1 - 2 * i_bulge - 1]);
				move_bulge(h.rb_mut(), v.rb_mut(), s1, s2);
				{
					let t0 = conj(v[0]);
					let v1 = copy(v[1]);
					let t1 = t0 * v1;
					let v2 = copy(v[2]);
					let t2 = t0 * v2;
					for j in istart_m..i_pos + 3 {
						let sum = a[(j, i_pos)] + v1 * a[(j, i_pos + 1)] + v2 * a[(j, i_pos + 2)];
						a[(j, i_pos)] = a[(j, i_pos)] - sum * conj(t0);
						a[(j, i_pos + 1)] = a[(j, i_pos + 1)] - sum * conj(t1);
						a[(j, i_pos + 2)] = a[(j, i_pos + 2)] - sum * conj(t2);
					}
				}
				let v0 = real(v[0]);
				let v1 = copy(v[1]);
				let v2 = copy(v[2]);
				let sum = a[(i_pos, i_pos)] + conj(v1) * a[(i_pos + 1, i_pos)] + conj(v2) * a[(i_pos + 2, i_pos)];
				a[(i_pos, i_pos)] = a[(i_pos, i_pos)] - mul_real(sum, v0);
				a[(i_pos + 1, i_pos)] = a[(i_pos + 1, i_pos)] - (mul_real(sum, v0) * v1);
				a[(i_pos + 2, i_pos)] = a[(i_pos + 2, i_pos)] - (mul_real(sum, v0) * v2);
				if i_pos > ilo && a[(i_pos, i_pos - 1)] != zero() {
					let mut tst1 = abs1(a[(i_pos - 1, i_pos - 1)]) + abs1(a[(i_pos, i_pos)]);
					if tst1 == zero() {
						if i_pos > ilo + 1 {
							tst1 = tst1 + abs1(a[(i_pos - 1, i_pos - 2)]);
						}
						if i_pos > ilo + 2 {
							tst1 = tst1 + abs1(a[(i_pos - 1, i_pos - 3)]);
						}
						if i_pos > ilo + 3 {
							tst1 = tst1 + abs1(a[(i_pos - 1, i_pos - 4)]);
						}
						if i_pos < ihi - 1 {
							tst1 = tst1 + abs1(a[(i_pos + 1, i_pos)]);
						}
						if i_pos < ihi - 2 {
							tst1 = tst1 + abs1(a[(i_pos + 2, i_pos)]);
						}
						if i_pos < ihi - 3 {
							tst1 = tst1 + abs1(a[(i_pos + 3, i_pos)]);
						}
					}
					if abs1(a[(i_pos, i_pos - 1)]) < max(small_num, eps * tst1) {
						let ab = max(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
						let ba = min(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
						let aa = max(abs1(a[(i_pos, i_pos)]), abs1((a[(i_pos, i_pos)] - a[(i_pos - 1, i_pos - 1)])));
						let bb = min(abs1(a[(i_pos, i_pos)]), abs1((a[(i_pos, i_pos)] - a[(i_pos - 1, i_pos - 1)])));
						let s = aa + ab;
						if ba * (ab / s) <= max(small_num, eps * (bb * (aa / s))) {
							a[(i_pos, i_pos - 1)] = zero();
						}
					}
				}
			}
		}
		i_bulge_start = if i_pos_last + 4 > ihi { (i_pos_last + 4 - ihi) / 2 } else { 0 };
		for i_bulge in i_bulge_start..n_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let v = v.rb_mut().col_mut(i_bulge);
			let v0 = real(v[0]);
			let v1 = copy(v[1]);
			let v2 = copy(v[2]);
			for j in i_pos + 1..istop_m {
				let sum = a[(i_pos, j)] + conj(v1) * a[(i_pos + 1, j)] + conj(v2) * a[(i_pos + 2, j)];
				a[(i_pos, j)] = a[(i_pos, j)] - mul_real(sum, v0);
				a[(i_pos + 1, j)] = a[(i_pos + 1, j)] - mul_real(sum, v0) * v1;
				a[(i_pos + 2, j)] = a[(i_pos + 2, j)] - mul_real(sum, v0) * v2;
			}
		}
		for i_bulge in i_bulge_start..n_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let v = v.rb_mut().col_mut(i_bulge);
			let v0 = real(v[0]);
			let v1 = copy(v[1]);
			let v2 = copy(v[2]);
			let i1 = (i_pos - *i_pos_block) - (i_pos_last + 2 - *i_pos_block - n_shifts);
			let i2 = Ord::min(u2.nrows(), (i_pos_last - *i_pos_block) + (i_pos_last + 2 - *i_pos_block - n_shifts) + 3);
			for j in i1..i2 {
				let sum = u2[(j, i_pos - *i_pos_block)] + v1 * u2[(j, i_pos - *i_pos_block + 1)] + v2 * u2[(j, i_pos - *i_pos_block + 2)];
				u2[(j, i_pos - *i_pos_block)] = u2[(j, i_pos - *i_pos_block)] - mul_real(sum, v0);
				u2[(j, i_pos - *i_pos_block + 1)] = u2[(j, i_pos - *i_pos_block + 1)] - mul_real(sum, v0) * conj(v1);
				u2[(j, i_pos - *i_pos_block + 2)] = u2[(j, i_pos - *i_pos_block + 2)] - mul_real(sum, v0) * conj(v2);
			}
		}
	}
	if want_t {
		istart_m = 0;
		istop_m = n;
	} else {
		istart_m = ilo;
		istop_m = ihi;
	}
	debug_assert!(*i_pos_block + n_block == ihi);
	if ihi < istop_m {
		let mut i = ihi;
		while i < istop_m {
			let iblock = Ord::min(istop_m - i, wh.ncols());
			let mut a_slice = a.rb_mut().submatrix_mut(*i_pos_block, i, n_block, iblock);
			let mut wh_slice = wh.rb_mut().submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
			matmul(wh_slice.rb_mut(), Accum::Replace, u2.rb().adjoint(), a_slice.rb(), one(), par);
			a_slice.copy_from(wh_slice.rb());
			i += iblock;
		}
	}
	if istart_m < *i_pos_block {
		let mut i = istart_m;
		while i < *i_pos_block {
			let iblock = Ord::min(*i_pos_block - i, wv.nrows());
			let mut a_slice = a.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
			let mut wv_slice = wv.rb_mut().submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
			matmul(wv_slice.rb_mut(), Accum::Replace, a_slice.rb(), u2.rb(), one(), par);
			a_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	if let Some(mut z) = z.rb_mut() {
		let mut i = 0;
		while i < n {
			let iblock = Ord::min(n - i, wv.nrows());
			let mut z_slice = z.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
			let mut wv_slice = wv.rb_mut().submatrix_mut(0, 0, z_slice.nrows(), z_slice.ncols());
			matmul(wv_slice.rb_mut(), Accum::Replace, z_slice.rb(), u2.rb(), one(), par);
			z_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
}

#[cfg(test)]
mod tests {
	use crate::linalg::evd::schur::multishift_qr_scratch;
	use crate::prelude::*;
	use crate::utils::approx::*;
	use crate::{assert, c64};
	use dyn_stack::{MemBuffer, MemStack};
	use rand::rngs::StdRng;
	use rand::{Rng, SeedableRng};

	#[test]
	fn test_3() {
		let n = 3;
		let h = MatRef::from_row_major_array(try_const! {
			&[
				[c64::new(0.997386, 0.677592), c64::new(0.646064, 0.936948), c64::new(0.090948, 0.674011)],
				[c64::new(0.212396, 0.976794), c64::new(0.460270, 0.926436), c64::new(0.494441, 0.888187)],
				[c64::new(0.000000, 0.000000), c64::new(0.616652, 0.840012), c64::new(0.768245, 0.349193)],
			]
		});

		let mut q = Mat::from_fn(n, n, |i, j| if i == j { c64::ONE } else { c64::ZERO });
		let mut w = Col::zeros(n);
		let mut t = h.cloned();
		super::lahqr(true, t.as_mut(), Some(q.as_mut()), w.as_mut(), 0, n);

		let h_reconstructed = &q * &t * q.adjoint();

		let approx_eq = CwiseMat(ApproxEq::eps());
		assert!(h_reconstructed ~ h);
	}

	#[test]
	fn test_n() {
		let rng = &mut StdRng::seed_from_u64(4);

		for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32, 128, 256] {
			for _ in 0..10 {
				let mut h = Mat::<c64>::zeros(n, n);
				for j in 0..n {
					for i in 0..n {
						if i <= j + 1 {
							h[(i, j)] = c64::new(rng.gen(), rng.gen());
						}
					}
				}

				if n <= 128 {
					let mut q = Mat::from_fn(n, n, |i, j| if i == j { c64::ONE } else { c64::ZERO });

					let mut w = Col::zeros(n);

					let mut t = h.clone();
					super::lahqr(true, t.as_mut(), Some(q.as_mut()), w.as_mut(), 0, n);

					let h_reconstructed = &q * &t * q.adjoint();

					let mut approx_eq = CwiseMat(ApproxEq::eps());
					approx_eq.0.abs_tol *= 10.0 * (n as f64).sqrt();
					approx_eq.0.rel_tol *= 10.0 * (n as f64).sqrt();

					assert!(h_reconstructed ~ h);
				}

				{
					let mut q = Mat::zeros(n, n);
					for i in 0..n {
						q[(i, i)] = c64::ONE;
					}

					let mut w = Col::zeros(n);

					let mut t = h.as_ref().cloned();
					super::multishift_qr(
						true,
						t.as_mut(),
						Some(q.as_mut()),
						w.as_mut(),
						0,
						n,
						Par::Seq,
						MemStack::new(&mut MemBuffer::new(multishift_qr_scratch::<c64>(n, n, true, true, Par::Seq, auto!(c64)))),
						auto!(c64),
					);

					for j in 0..n {
						for i in 0..n {
							if i > j + 1 {
								t[(i, j)] = c64::ZERO;
							}
						}
					}

					let h_reconstructed = &q * &t * q.adjoint();

					let mut approx_eq = CwiseMat(ApproxEq::eps());
					approx_eq.0.abs_tol *= 10.0 * (n as f64);
					approx_eq.0.rel_tol *= 10.0 * (n as f64);

					assert!(h ~ h_reconstructed);
				}
			}
		}
	}
}
