use super::*;
use crate::{assert, debug_assert};
use linalg::householder::*;
use linalg::jacobi::JacobiRotation;
use linalg::matmul::matmul;
fn lahqr_eig22<T: ComplexField>(a00: T, a01: T, a10: T, a11: T) -> (T, T) {
	let ref s = a00.abs1() + a01.abs1() + a10.abs1() + a11.abs1();
	if *s == zero() {
		return (zero(), zero());
	}
	let ref half = from_f64::<T::Real>(0.5);
	let ref s_inv = s.recip();
	let ref a00 = a00.mul_real(s_inv);
	let ref a01 = a01.mul_real(s_inv);
	let ref a10 = a10.mul_real(s_inv);
	let ref a11 = a11.mul_real(s_inv);
	let ref tr = (a00 + a11).mul_real(half);
	let det = (a00 - tr) * (a00 - tr) + (a01 * a10);
	let ref rtdisc = det.sqrt();
	((tr + rtdisc).mul_real(s), (tr - rtdisc).mul_real(s))
}
fn lahqr<T: ComplexField>(
	want_t: bool,
	A: MatMut<'_, T>,
	Z: Option<MatMut<'_, T>>,
	w: ColMut<'_, T>,
	ilo: usize,
	ihi: usize,
) -> isize {
	let n = A.nrows();
	let nh = ihi - ilo;
	let mut A = A;
	let mut Z = Z;
	let mut w = w;
	let ref eps = eps::<T::Real>();
	let ref smlnum = min_positive::<T::Real>() / eps;
	let non_convergence_limit = 10usize;
	let ref dat1 = from_f64::<T::Real>(0.75);
	let ref dat2 = from_f64::<T::Real>(-0.4375);
	if nh == 0 {
		return 0;
	}
	if nh == 1 {
		w[ilo] = A[(ilo, ilo)].copy();
	}
	let itmax = Ord::max(30, nbits::<T>() / 2)
		.saturating_mul(Ord::max(10, nh))
		.saturating_mul(nh);
	let mut k_defl = 0usize;
	let mut istop = ihi;
	let mut istart = ilo;
	for iter in 0..itmax {
		if iter + 1 == itmax {
			return istop as isize;
		}
		if istart + 1 >= istop {
			if istart + 1 == istop {
				w[istart] = A[(istart, istart)].copy();
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
			if A[(i, i - 1)].abs1() < *smlnum {
				A[(i, i - 1)] = zero();
				istart = i;
				break;
			}
			let mut tst = A[(i - 1, i - 1)].abs1() + A[(i, i)].abs1();
			if tst == zero() {
				if i >= ilo + 2 {
					tst += A[(i - 1, i - 2)].abs1();
				}
				if i + 1 < ihi {
					tst += A[(i + 1, i)].abs1();
				}
			}
			if A[(i, i - 1)].abs1() <= eps * tst {
				let ref ab = A[(i, i - 1)].abs1().fmax(A[(i - 1, i)].abs1());
				let ref ba = A[(i, i - 1)].abs1().fmin(A[(i - 1, i)].abs1());
				let ref aa = A[(i, i)]
					.abs1()
					.fmax((&A[(i, i)] - &A[(i - 1, i - 1)]).abs1());
				let ref bb = A[(i, i)]
					.abs1()
					.fmin((&A[(i, i)] - &A[(i - 1, i - 1)]).abs1());
				let ref s = aa + ab;
				if ba * (ab / s) <= (eps * (bb * (aa / s))).fmax(smlnum) {
					A[(i, i - 1)] = zero();
					istart = i;
					break;
				}
			}
		}
		if istart + 1 >= istop {
			k_defl = 0;
			w[istart] = A[(istart, istart)].copy();
			istop = istart;
			istart = ilo;
			continue;
		}
		let (a00, a01, a10, a11);
		k_defl += 1;
		if k_defl % non_convergence_limit == 0 {
			let mut s = A[(istop - 1, istop - 2)].abs();
			if istop > ilo + 2 {
				s += A[(istop - 2, istop - 3)].abs();
			}
			a00 = (dat1 * &s).to_cplx::<T>() + &A[(istop - 1, istop - 1)];
			a10 = (dat2 * &s).to_cplx::<T>();
			a01 = s.to_cplx::<T>();
			a11 = a00.copy();
		} else {
			a00 = A[(istop - 2, istop - 2)].copy();
			a10 = A[(istop - 1, istop - 2)].copy();
			a01 = A[(istop - 2, istop - 1)].copy();
			a11 = A[(istop - 1, istop - 1)].copy();
		}
		let (mut s1, s2) = lahqr_eig22(a00, a01, a10, a11);
		if (&s1 - &A[(istop - 1, istop - 1)]).abs1()
			> (&s2 - &A[(istop - 1, istop - 1)]).abs1()
		{
			s1 = s2.copy();
		}
		let mut istart2 = istart;
		if istart + 2 < istop {
			for i in (istart + 1..istop - 1).rev() {
				let h00 = &A[(i, i)] - &s1;
				let h10 = A[(i + 1, i)].copy();
				let JacobiRotation { c: _, s: sn } =
					JacobiRotation::rotg(h00, h10).0;
				if (sn.conj() * &A[(i, i - 1)]).abs1()
					<= eps * (A[(i, i - 1)].abs1() + A[(i, i + 1)].abs1())
				{
					istart2 = i;
					break;
				}
			}
		}
		for i in istart2..istop - 1 {
			let (rot, r);
			if i == istart2 {
				let h00 = &A[(i, i)] - &s1;
				let h10 = A[(i + 1, i)].copy();
				(rot, _) = JacobiRotation::rotg(h00, h10);
				if i > istart {
					A[(i, i - 1)] *= &rot.c;
				}
			} else {
				(rot, r) = JacobiRotation::rotg(
					A[(i, i - 1)].copy(),
					A[(i + 1, i - 1)].copy(),
				);
				A[(i, i - 1)] = r.copy();
				A[(i + 1, i - 1)] = zero();
			}
			rot.adjoint().apply_on_the_left_in_place(
				A.rb_mut()
					.subcols_mut(i, istop_m - i)
					.two_rows_mut(i, i + 1),
			);
			rot.apply_on_the_right_in_place(
				A.rb_mut()
					.get_mut(istart_m..Ord::min(i + 3, istop), ..)
					.two_cols_mut(i, i + 1),
			);
			if let Some(Z) = Z.rb_mut() {
				rot.apply_on_the_right_in_place(Z.two_cols_mut(i, i + 1));
			}
		}
	}
	0
}
fn lahqr_shiftcolumn<T: ComplexField>(
	h: MatRef<'_, T>,
	v: ColMut<'_, T>,
	s1: T,
	s2: T,
) {
	let mut v = v;
	let ref s1 = s1;
	let ref s2 = s2;
	let n = h.nrows();
	if n == 2 {
		let s = (&h[(0, 0)] - s2).abs1() + h[(1, 0)].abs1();
		if s == zero() {
			v[0] = zero();
			v[1] = zero();
		} else {
			let ref s_inv = s.recip();
			let ref h10s = h[(1, 0)].mul_real(s_inv);
			v[0] = h10s * &h[(0, 1)]
				+ (&h[(0, 0)] - s1) * (&h[(0, 0)] - s2).mul_real(s_inv);
			v[1] = h10s * (&h[(0, 0)] + &h[(1, 1)] - s1 - s2);
		}
	} else {
		let s = (&h[(0, 0)] - s2).abs1() + h[(1, 0)].abs1() + h[(2, 0)].abs1();
		if s == zero() {
			v[0] = zero();
			v[1] = zero();
			v[2] = zero();
		} else {
			let ref s_inv = s.recip();
			let ref h10s = h[(1, 0)].mul_real(s_inv);
			let ref h20s = h[(2, 0)].mul_real(s_inv);
			v[0] = (&h[(0, 0)] - s1) * (&h[(0, 0)] - s2).mul_real(s_inv)
				+ &h[(0, 1)] * h10s
				+ &h[(0, 2)] * h20s;
			v[1] =
				h10s * (&h[(0, 0)] + &h[(1, 1)] - s1 - s2) + &h[(1, 2)] * h20s;
			v[2] =
				h20s * (&h[(0, 0)] + &h[(2, 2)] - s1 - s2) + h10s * &h[(2, 1)];
		}
	}
}
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
	let ref eps = eps::<T::Real>();
	let ref n_T = from_f64::<T::Real>(n as f64);
	let ref small_num = min_positive::<T::Real>() / eps * n_T;
	let jw = Ord::min(Ord::min(nw, ihi - ilo), nw_max);
	let kwtop = ihi - jw;
	let mut s_spike = if kwtop == ilo {
		zero()
	} else {
		a[(kwtop, kwtop - 1)].copy()
	};
	if kwtop + 1 == ihi {
		s[kwtop] = a[(kwtop, kwtop)].copy();
		let mut ns = 1;
		let mut nd = 0;
		if s_spike.abs1() <= (eps * a[(kwtop, kwtop)].abs1()).fmax(small_num) {
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
	let mut wh = unsafe {
		a.rb()
			.submatrix(n - jw, jw, jw, n - 2 * jw - 3)
			.const_cast()
	};
	let mut wv =
		unsafe { a.rb().submatrix(jw + 3, 0, n - 2 * jw - 3, jw).const_cast() };
	let mut a = unsafe { a.rb().const_cast() };
	let a_window = a.rb().submatrix(kwtop, kwtop, ihi - kwtop, ihi - kwtop);
	let mut s_window =
		unsafe { s.rb().subrows(kwtop, ihi - kwtop).const_cast() };
	tw.fill(zero());
	for j in 0..jw {
		for i in 0..Ord::min(j + 2, jw) {
			tw[(i, j)] = a_window[(i, j)].copy();
		}
	}
	v.fill(zero());
	v.rb_mut().diagonal_mut().fill(one());
	let infqr = if jw < params.blocking_threshold {
		lahqr(
			true,
			tw.rb_mut(),
			Some(v.rb_mut()),
			s_window.rb_mut(),
			0,
			jw,
		)
	} else {
		let infqr = multishift_qr(
			true,
			tw.rb_mut(),
			Some(v.rb_mut()),
			s_window.rb_mut(),
			0,
			jw,
			par,
			stack,
			params,
		)
		.0;
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
		let mut foo = tw[(ns - 1, ns - 1)].abs1();
		if foo == zero() {
			foo = s_spike.abs1();
		}
		if s_spike.abs1() * v[(0, ns - 1)].abs1() <= (eps * foo).fmax(small_num)
		{
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
			let ev1 = tw[(i1, i1)].abs1();
			let ev2 = tw[(i2, i2)].abs1();
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
		s[kwtop + i] = tw[(i, i)].copy();
		i += 1;
	}
	if s_spike != zero() {
		{
			let mut vv = wv.rb_mut().col_mut(0).subrows_mut(0, ns);
			for i in 0..ns {
				vv[i] = v[(0, i)].conj();
			}
			let mut head = vv[0].copy();
			let tail = vv.rb_mut().subrows_mut(1, ns - 1);
			let HouseholderInfo { tau, .. } =
				make_householder_in_place(&mut head, tail);
			vv[0] = one();
			let ref tau = tau.recip().to_cplx::<T>();
			{
				let mut tw_slice = tw.rb_mut().submatrix_mut(0, 0, ns, jw);
				let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
				let mut tmp = tmp.as_mat_mut().transpose_mut();
				matmul(
					tmp.rb_mut(),
					Accum::Replace,
					vv.rb().adjoint().as_mat(),
					tw_slice.rb(),
					one(),
					par,
				);
				matmul(
					tw_slice.rb_mut(),
					Accum::Add,
					vv.rb().as_mat(),
					tmp.as_ref(),
					-tau,
					par,
				);
			}
			{
				let mut tw_slice2 = tw.rb_mut().submatrix_mut(0, 0, jw, ns);
				let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
				let mut tmp = tmp.as_mat_mut();
				matmul(
					tmp.rb_mut(),
					Accum::Replace,
					tw_slice2.rb(),
					vv.rb().as_mat(),
					one(),
					par,
				);
				matmul(
					tw_slice2.rb_mut(),
					Accum::Add,
					tmp.as_ref(),
					vv.rb().adjoint().as_mat(),
					-tau,
					par,
				);
			}
			{
				let mut v_slice = v.rb_mut().submatrix_mut(0, 0, jw, ns);
				let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
				let mut tmp = tmp.as_mat_mut();
				matmul(
					tmp.rb_mut(),
					Accum::Replace,
					v_slice.rb(),
					vv.rb().as_mat(),
					one(),
					par,
				);
				matmul(
					v_slice.rb_mut(),
					Accum::Add,
					tmp.as_ref(),
					vv.rb().adjoint().as_mat(),
					-tau,
					par,
				);
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
		a[(kwtop, kwtop - 1)] = s_spike * v[(0, 0)].conj();
	}
	for j in 0..jw {
		for i in 0..Ord::min(j + 2, jw) {
			a[(kwtop + i, kwtop + j)] = tw[(i, j)].copy();
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
			let mut a_slice =
				a.rb_mut().submatrix_mut(kwtop, i, ihi - kwtop, iblock);
			let mut wh_slice = wh.rb_mut().submatrix_mut(
				0,
				0,
				a_slice.nrows(),
				a_slice.ncols(),
			);
			matmul(
				wh_slice.rb_mut(),
				Accum::Replace,
				v.rb().adjoint(),
				a_slice.rb(),
				one(),
				par,
			);
			a_slice.copy_from(wh_slice.rb());
			i += iblock;
		}
	}
	if istart_m < kwtop {
		let mut i = istart_m;
		while i < kwtop {
			let iblock = Ord::min(kwtop - i, wv.nrows());
			let mut a_slice =
				a.rb_mut().submatrix_mut(i, kwtop, iblock, ihi - kwtop);
			let mut wv_slice = wv.rb_mut().submatrix_mut(
				0,
				0,
				a_slice.nrows(),
				a_slice.ncols(),
			);
			matmul(
				wv_slice.rb_mut(),
				Accum::Replace,
				a_slice.rb(),
				v.rb(),
				one(),
				par,
			);
			a_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	if let Some(mut z) = z.rb_mut() {
		let mut i = 0;
		while i < n {
			let iblock = Ord::min(n - i, wv.nrows());
			let mut z_slice =
				z.rb_mut().submatrix_mut(i, kwtop, iblock, ihi - kwtop);
			let mut wv_slice = wv.rb_mut().submatrix_mut(
				0,
				0,
				z_slice.nrows(),
				z_slice.ncols(),
			);
			matmul(
				wv_slice.rb_mut(),
				Accum::Replace,
				z_slice.rb(),
				v.rb(),
				one(),
				par,
			);
			z_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	(ns, nd)
}
fn schur_move<T: ComplexField>(
	mut a: MatMut<'_, T>,
	mut q: Option<MatMut<'_, T>>,
	ifst: usize,
	ilst: &mut usize,
) -> isize {
	let n = a.nrows();
	if n == 0 {
		return 0;
	}
	let mut here = ifst;
	if ifst < *ilst {
		while here != *ilst {
			let ierr = schur_swap(a.rb_mut(), q.rb_mut(), here);
			if ierr != 0 {
				*ilst = here;
				return 1;
			}
			here += 1;
		}
	} else {
		while here != *ilst {
			let ierr = schur_swap(a.rb_mut(), q.rb_mut(), here - 1);
			if ierr != 0 {
				*ilst = here;
				return 1;
			}
			here -= 1;
		}
	}
	0
}
pub(crate) fn schur_swap<T: ComplexField>(
	mut a: MatMut<'_, T>,
	q: Option<MatMut<'_, T>>,
	j0: usize,
) -> isize {
	let n = a.nrows();
	let j1 = j0 + 1;
	let j2 = j0 + 2;
	let t00 = a[(j0, j0)].copy();
	let t11 = a[(j1, j1)].copy();
	let (rot, _) = JacobiRotation::<T>::rotg(a[(j0, j1)].copy(), &t11 - &t00);
	a[(j1, j1)] = t00;
	a[(j0, j0)] = t11;
	if j2 < n {
		rot.adjoint().apply_on_the_left_in_place(
			a.rb_mut().get_mut(.., j2..).two_rows_mut(j0, j1),
		);
	}
	if j0 > 0 {
		rot.apply_on_the_right_in_place(
			a.rb_mut().get_mut(..j0, ..).two_cols_mut(j0, j1),
		);
	}
	if let Some(q) = q {
		rot.apply_on_the_right_in_place(q.two_cols_mut(j0, j1));
	}
	0
}
/// returns err code, number of aggressive early deflations, number of qr sweeps
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
	let ref dat1 = from_f64::<T::Real>(0.75);
	let ref dat2 = from_f64::<T::Real>(-0.4375);
	let nmin = Ord::max(15, params.blocking_threshold);
	let nibble = params.nibble_threshold;
	let nsr = (params.recommended_shift_count)(n, nh);
	let nsr = Ord::min(Ord::min(nsr, n.saturating_sub(3) / 6), ihi - ilo - 1);
	let nsr = Ord::max(nsr / 2 * 2, 2);
	let nwr = (params.recommended_deflation_window)(n, nh);
	let nwr = Ord::max(nwr, 2);
	let nwr = Ord::min(Ord::min(nwr, n.saturating_sub(1) / 3), ihi - ilo);
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
				w[ilo] = a[(ilo, ilo)].copy();
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
				nw = nh;
			}
			let kwtop = istop - nw;
			if (kwtop > istart + 2)
				&& (a[(kwtop, kwtop - 1)].abs1()
					> a[(kwtop - 1, kwtop - 2)].abs1())
			{
				nw += 1;
			}
		}
		let (ls, ld) = aggressive_early_deflation(
			want_t,
			a.rb_mut(),
			z.rb_mut(),
			w.rb_mut(),
			istart,
			istop,
			nw,
			par,
			stack.rb_mut(),
			params,
		);
		count_aed += 1;
		istop -= ld;
		if ld > 0 {
			k_defl = 0;
		}
		if ld > 0
			&& (100 * ld > nwr * nibble
				|| (istop - istart) <= Ord::min(nmin, nw_max))
		{
			continue;
		}
		k_defl += 1;
		let mut ns = Ord::min(nh - 1, Ord::min(Ord::max(2, ls), nsr));
		ns = ns / 2 * 2;
		let mut i_shifts = istop - ns;
		if k_defl % non_convergence_limit_shift == 0 {
			for i in (i_shifts + 1..istop).rev().step_by(2) {
				if i >= ilo + 2 {
					let ref ss =
						a[(i, i - 1)].abs1() + a[(i - 1, i - 2)].abs1();
					let aa = (dat1 * ss).to_cplx::<T>() + &a[(i, i)];
					let bb = ss.to_cplx::<T>();
					let cc = (dat2 * ss).to_cplx::<T>();
					let dd = aa.copy();
					let (s1, s2) = lahqr_eig22(aa, bb, cc, dd);
					w[i - 1] = s1;
					w[i] = s2;
				} else {
					w[i - 1] = a[(i, i)].copy();
					w[i] = a[(i, i)].copy();
				}
			}
		} else {
			if ls <= ns / 2 {
				let mut temp = a.rb_mut().submatrix_mut(n - ns, 0, ns, ns);
				let mut shifts = w.rb_mut().subrows_mut(istop - ns, ns);
				let ierr =
					lahqr(false, temp.rb_mut(), None, shifts.rb_mut(), 0, ns)
						as usize;
				ns = ns - ierr;
				if ns < 2 {
					let aa = a[(istop - 2, istop - 2)].copy();
					let bb = a[(istop - 2, istop - 1)].copy();
					let cc = a[(istop - 1, istop - 2)].copy();
					let dd = a[(istop - 1, istop - 1)].copy();
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
					if w[i].abs1() < w[i + 1].abs1() {
						sorted = false;
						let wi = w[i].copy();
						let wip1 = w[i + 1].copy();
						w[i] = wip1;
						w[i + 1] = wi;
					}
				}
				k -= 1;
			}
			for i in (i_shifts + 2..istop).rev().step_by(2) {
				if w[i].imag() != -w[i - 1].imag() {
					let tmp = w[i].copy();
					w[i] = w[i - 1].copy();
					w[i - 1] = w[i - 2].copy();
					w[i - 2] = tmp;
				}
			}
			if ns % 2 == 1 {
				ns -= 1;
			}
			i_shifts = istop - ns;
		}
		let mut shifts = w.rb_mut().subrows_mut(i_shifts, ns);
		multishift_qr_sweep(
			want_t,
			a.rb_mut(),
			z.rb_mut(),
			shifts.rb_mut(),
			istart,
			istop,
			par,
			stack,
		);
		count_sweep += 1;
	}
	(info, count_aed, count_sweep)
}
fn move_bulge<T: ComplexField>(
	mut h: MatMut<'_, T>,
	mut v: ColMut<'_, T>,
	s1: T,
	s2: T,
) {
	let ref v0 = v[0].real();
	let ref v1 = v[1].copy();
	let ref v2 = v[2].copy();
	let ref refsum = v2.mul_real(v0) * &h[(3, 2)];
	let ref epsilon = eps::<T::Real>();
	h[(3, 0)] = -refsum;
	h[(3, 1)] = -refsum * v1.conj();
	h[(3, 2)] -= refsum * v2.conj();
	v[0] = h[(1, 0)].copy();
	v[1] = h[(2, 0)].copy();
	v[2] = h[(3, 0)].copy();
	let mut beta = v[0].copy();
	let tail = v.rb_mut().subrows_mut(1, 2);
	let HouseholderInfo { tau, .. } =
		make_householder_in_place(&mut beta, tail);
	v[0] = tau.recip().to_cplx();
	if h[(3, 0)] != zero() || h[(3, 1)] != zero() || h[(3, 2)] != zero() {
		h[(1, 0)] = beta;
		h[(2, 0)] = zero();
		h[(3, 0)] = zero();
	} else {
		stack_mat!(vt, 3, 1, T);
		let mut vt = vt.rb_mut().col_mut(0);
		let h2 = h.rb().submatrix(1, 1, 3, 3);
		lahqr_shiftcolumn(h2, vt.rb_mut(), s1, s2);
		let mut beta_unused = vt[0].copy();
		let tail = vt.rb_mut().subrows_mut(1, 2);
		let HouseholderInfo { tau, .. } =
			make_householder_in_place(&mut beta_unused, tail);
		vt[0] = tau.recip().to_cplx();
		let ref vt0 = vt[0].copy();
		let ref vt1 = vt[1].copy();
		let ref vt2 = vt[2].copy();
		let ref refsum = vt0.conj() * &h[(1, 0)] + vt1.conj() * &h[(2, 0)];
		if (&h[(2, 0)] - refsum * vt1).abs1() + (refsum * vt2).abs1()
			> epsilon * (h[(0, 0)].abs1() + h[(1, 1)].abs1() + h[(2, 2)].abs1())
		{
			h[(1, 0)] = beta;
			h[(2, 0)] = zero();
			h[(3, 0)] = zero();
		} else {
			h[(1, 0)] -= refsum;
			h[(2, 0)] = zero();
			h[(3, 0)] = zero();
			v[0] = vt[0].copy();
			v[1] = vt[1].copy();
			v[2] = vt[2].copy();
		}
	}
}
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
	let n_shifts_max =
		Ord::min(ihi - ilo - 1, Ord::max(2, 3 * (n_block_max / 4)));
	let mut n_shifts = Ord::min(s.nrows(), n_shifts_max);
	if n_shifts % 2 == 1 {
		n_shifts -= 1;
	}
	let n_bulges = n_shifts / 2;
	let n_block_desired = Ord::min(2 * n_shifts, n_block_max);
	let mut u = unsafe {
		a.rb()
			.submatrix(n - n_block_desired, 0, n_block_desired, n_block_desired)
			.const_cast()
	};
	let mut wh = unsafe {
		a.rb()
			.submatrix(
				n - n_block_desired,
				n_block_desired,
				n_block_desired,
				n - 2 * n_block_desired - 3,
			)
			.const_cast()
	};
	let mut wv = unsafe {
		a.rb()
			.submatrix(
				n_block_desired + 3,
				0,
				n - 2 * n_block_desired - 3,
				n_block_desired,
			)
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
	let ref eps = eps::<T::Real>();
	let ref n_T = from_f64::<T::Real>(n as f64);
	let ref small_num = min_positive::<T::Real>() / eps * n_T;
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
				let s1 = s[s.nrows() - 1 - 2 * i_bulge].copy();
				let s2 = s[s.nrows() - 1 - 2 * i_bulge - 1].copy();
				lahqr_shiftcolumn(h, v.rb_mut(), s1, s2);
				debug_assert!(v.nrows() == 3);
				let mut beta = v[0].copy();
				let tail = v.rb_mut().subrows_mut(1, 2);
				let HouseholderInfo { tau, .. } =
					make_householder_in_place(&mut beta, tail);
				v[0] = tau.recip().to_cplx();
			} else {
				let mut h =
					a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
				let s1 = s[s.nrows() - 1 - 2 * i_bulge].copy();
				let s2 = s[s.nrows() - 1 - 2 * i_bulge - 1].copy();
				move_bulge(h.rb_mut(), v.rb_mut(), s1, s2);
			}
			let ref v0 = v[0].real();
			let ref v1 = v[1].copy();
			let ref v2 = v[2].copy();
			for j in istart_m..i_pos + 3 {
				let ref sum = &a[(j, i_pos)]
					+ v1 * &a[(j, i_pos + 1)]
					+ v2 * &a[(j, i_pos + 2)];
				a[(j, i_pos)] -= sum.mul_real(v0);
				a[(j, i_pos + 1)] -= sum.mul_real(v0) * v1.conj();
				a[(j, i_pos + 2)] -= sum.mul_real(v0) * v2.conj();
			}
			let ref sum = &a[(i_pos, i_pos)]
				+ v1.conj() * &a[(i_pos + 1, i_pos)]
				+ v2.conj() * &a[(i_pos + 2, i_pos)];
			a[(i_pos, i_pos)] -= sum.mul_real(v0);
			a[(i_pos + 1, i_pos)] -= sum.mul_real(v0) * v1;
			a[(i_pos + 2, i_pos)] -= sum.mul_real(v0) * v2;
			if (i_pos > ilo) && (a[(i_pos, i_pos - 1)] != zero()) {
				let mut tst1 =
					a[(i_pos - 1, i_pos - 1)].abs1() + a[(i_pos, i_pos)].abs1();
				if tst1 == zero() {
					if i_pos > ilo + 1 {
						tst1 = tst1 + a[(i_pos - 1, i_pos - 2)].abs1();
					}
					if i_pos > ilo + 2 {
						tst1 = tst1 + a[(i_pos - 1, i_pos - 3)].abs1();
					}
					if i_pos > ilo + 3 {
						tst1 = tst1 + a[(i_pos - 1, i_pos - 4)].abs1();
					}
					if i_pos < ihi - 1 {
						tst1 = tst1 + a[(i_pos + 1, i_pos)].abs1();
					}
					if i_pos < ihi - 2 {
						tst1 = tst1 + a[(i_pos + 2, i_pos)].abs1();
					}
					if i_pos < ihi - 3 {
						tst1 = tst1 + a[(i_pos + 3, i_pos)].abs1();
					}
				}
				if a[(i_pos, i_pos - 1)].abs1() < small_num.fmax(eps * tst1) {
					let ref ab = a[(i_pos, i_pos - 1)]
						.abs1()
						.fmax(a[(i_pos - 1, i_pos)].abs1());
					let ref ba = a[(i_pos, i_pos - 1)]
						.abs1()
						.fmin(a[(i_pos - 1, i_pos)].abs1());
					let ref aa = a[(i_pos, i_pos)].abs1().fmax(
						(&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)])
							.abs1(),
					);
					let ref bb = a[(i_pos, i_pos)].abs1().fmin(
						(&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)])
							.abs1(),
					);
					let ref s = aa + ab;
					if ba * (ab / s) <= small_num.fmax(eps * (bb * (aa / s))) {
						a[(i_pos, i_pos - 1)] = zero();
					}
				}
			}
		}
		for i_bulge in 0..n_active_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let v = v.rb_mut().col_mut(i_bulge);
			let ref v0 = v[0].real();
			let ref v1 = v[1].copy();
			let ref v2 = v[2].copy();
			for j in i_pos + 1..istop_m {
				let ref sum = &a[(i_pos, j)]
					+ v1.conj() * &a[(i_pos + 1, j)]
					+ v2.conj() * &a[(i_pos + 2, j)];
				a[(i_pos, j)] -= sum.mul_real(v0);
				a[(i_pos + 1, j)] -= sum.mul_real(v0) * v1;
				a[(i_pos + 2, j)] -= sum.mul_real(v0) * v2;
			}
		}
		for i_bulge in 0..n_active_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let v = v.rb_mut().col_mut(i_bulge);
			let ref v0 = v[0].real();
			let ref v1 = v[1].copy();
			let ref v2 = v[2].copy();
			let i1 = 0;
			let i2 = Ord::min(
				u2.nrows(),
				(i_pos_last - ilo) + (i_pos_last - ilo) + 3,
			);
			for j in i1..i2 {
				let ref sum = &u2[(j, i_pos - ilo)]
					+ v1 * &u2[(j, i_pos - ilo + 1)]
					+ v2 * &u2[(j, i_pos - ilo + 2)];
				u2[(j, i_pos - ilo)] -= sum.mul_real(v0);
				u2[(j, i_pos - ilo + 1)] -= sum.mul_real(v0) * v1.conj();
				u2[(j, i_pos - ilo + 2)] -= sum.mul_real(v0) * v2.conj();
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
			let mut wh_slice = wh.rb_mut().submatrix_mut(
				0,
				0,
				a_slice.nrows(),
				a_slice.ncols(),
			);
			matmul(
				wh_slice.rb_mut(),
				Accum::Replace,
				u2.rb().adjoint(),
				a_slice.rb(),
				one(),
				par,
			);
			a_slice.copy_from(wh_slice.rb());
			i += iblock;
		}
	}
	if istart_m < ilo {
		let mut i = istart_m;
		while i < ilo {
			let iblock = Ord::min(ilo - i, wv.nrows());
			let mut a_slice = a.rb_mut().submatrix_mut(i, ilo, iblock, n_block);
			let mut wv_slice = wv.rb_mut().submatrix_mut(
				0,
				0,
				a_slice.nrows(),
				a_slice.ncols(),
			);
			matmul(
				wv_slice.rb_mut(),
				Accum::Replace,
				a_slice.rb(),
				u2.rb(),
				one(),
				par,
			);
			a_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	if let Some(mut z) = z.rb_mut() {
		let mut i = 0;
		while i < n {
			let iblock = Ord::min(n - i, wv.nrows());
			let mut z_slice = z.rb_mut().submatrix_mut(i, ilo, iblock, n_block);
			let mut wv_slice = wv.rb_mut().submatrix_mut(
				0,
				0,
				z_slice.nrows(),
				z_slice.ncols(),
			);
			matmul(
				wv_slice.rb_mut(),
				Accum::Replace,
				z_slice.rb(),
				u2.rb(),
				one(),
				par,
			);
			z_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	*i_pos_block = ilo + n_block - n_shifts;
}
#[inline(never)]
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
	let ref eps = eps::<T::Real>();
	let ref n_T = from_f64::<T::Real>(n as f64);
	let ref small_num = min_positive::<T::Real>() / eps * n_T;
	while *i_pos_block + n_block_desired < ihi {
		let n_pos = Ord::min(
			n_block_desired - n_shifts,
			ihi - n_shifts - 1 - *i_pos_block,
		);
		let n_block = n_shifts + n_pos;
		let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
		u2.fill(zero());
		u2.rb_mut().diagonal_mut().fill(one());
		let mut istart_m = *i_pos_block;
		let mut istop_m = *i_pos_block + n_block;
		for i_pos_last in
			*i_pos_block + n_shifts - 2..*i_pos_block + n_shifts - 2 + n_pos
		{
			for i_bulge in 0..n_bulges {
				let i_pos = i_pos_last - 2 * i_bulge;
				let mut v = v.rb_mut().col_mut(i_bulge);
				let mut h =
					a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
				let s1 = s[s.nrows() - 1 - 2 * i_bulge].copy();
				let s2 = s[s.nrows() - 1 - 2 * i_bulge - 1].copy();
				move_bulge(h.rb_mut(), v.rb_mut(), s1, s2);
				let ref v0 = v[0].real();
				let ref v1 = v[1].copy();
				let ref v2 = v[2].copy();
				for j in istart_m..i_pos + 3 {
					let ref sum = &a[(j, i_pos)]
						+ v1 * &a[(j, i_pos + 1)]
						+ v2 * &a[(j, i_pos + 2)];
					a[(j, i_pos)] -= sum.mul_real(v0);
					a[(j, i_pos + 1)] -= sum.mul_real(v0) * v1.conj();
					a[(j, i_pos + 2)] -= sum.mul_real(v0) * v2.conj();
				}
				let ref sum = &a[(i_pos, i_pos)]
					+ v1.conj() * &a[(i_pos + 1, i_pos)]
					+ v2.conj() * &a[(i_pos + 2, i_pos)];
				a[(i_pos, i_pos)] -= sum.mul_real(v0);
				a[(i_pos + 1, i_pos)] -= sum.mul_real(v0) * v1;
				a[(i_pos + 2, i_pos)] -= sum.mul_real(v0) * v2;
				if (i_pos > ilo) && (a[(i_pos, i_pos - 1)] != zero()) {
					let mut tst1 = a[(i_pos - 1, i_pos - 1)].abs1()
						+ a[(i_pos, i_pos)].abs1();
					if tst1 == zero() {
						if i_pos > ilo + 1 {
							tst1 = tst1 + a[(i_pos - 1, i_pos - 2)].abs1();
						}
						if i_pos > ilo + 2 {
							tst1 = tst1 + a[(i_pos - 1, i_pos - 3)].abs1();
						}
						if i_pos > ilo + 3 {
							tst1 = tst1 + a[(i_pos - 1, i_pos - 4)].abs1();
						}
						if i_pos < ihi - 1 {
							tst1 = tst1 + a[(i_pos + 1, i_pos)].abs1();
						}
						if i_pos < ihi - 2 {
							tst1 = tst1 + a[(i_pos + 2, i_pos)].abs1();
						}
						if i_pos < ihi - 3 {
							tst1 = tst1 + a[(i_pos + 3, i_pos)].abs1();
						}
					}
					if a[(i_pos, i_pos - 1)].abs1() < small_num.fmax(eps * tst1)
					{
						let ref ab = a[(i_pos, i_pos - 1)]
							.abs1()
							.fmax(a[(i_pos - 1, i_pos)].abs1());
						let ref ba = a[(i_pos, i_pos - 1)]
							.abs1()
							.fmin(a[(i_pos - 1, i_pos)].abs1());
						let ref aa = a[(i_pos, i_pos)].abs1().fmax(
							(&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)])
								.abs1(),
						);
						let ref bb = a[(i_pos, i_pos)].abs1().fmin(
							(&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)])
								.abs1(),
						);
						let ref s = aa + ab;
						if ba * (ab / s)
							<= small_num.fmax(eps * (bb * (aa / s)))
						{
							a[(i_pos, i_pos - 1)] = zero();
						}
					}
				}
			}
			for i_bulge in 0..n_bulges {
				let i_pos = i_pos_last - 2 * i_bulge;
				let v = v.rb_mut().col_mut(i_bulge);
				let ref v0 = v[0].real();
				let ref v1 = v[1].copy();
				let ref v2 = v[2].copy();
				for j in i_pos + 1..istop_m {
					let ref sum = &a[(i_pos, j)]
						+ v1.conj() * &a[(i_pos + 1, j)]
						+ v2.conj() * &a[(i_pos + 2, j)];
					a[(i_pos, j)] -= sum.mul_real(v0);
					a[(i_pos + 1, j)] -= sum.mul_real(v0) * v1;
					a[(i_pos + 2, j)] -= sum.mul_real(v0) * v2;
				}
			}
			for i_bulge in 0..n_bulges {
				let i_pos = i_pos_last - 2 * i_bulge;
				let v = v.rb_mut().col_mut(i_bulge);
				let ref v0 = v[0].real();
				let ref v1 = v[1].copy();
				let ref v2 = v[2].copy();
				let i1 = (i_pos - *i_pos_block)
					- (i_pos_last + 2 - *i_pos_block - n_shifts);
				let i2 = Ord::min(
					u2.nrows(),
					(i_pos_last - *i_pos_block)
						+ (i_pos_last + 2 - *i_pos_block - n_shifts)
						+ 3,
				);
				for j in i1..i2 {
					let ref sum = &u2[(j, i_pos - *i_pos_block)]
						+ v1 * &u2[(j, i_pos - *i_pos_block + 1)]
						+ v2 * &u2[(j, i_pos - *i_pos_block + 2)];
					u2[(j, i_pos - *i_pos_block)] -= sum.mul_real(v0);
					u2[(j, i_pos - *i_pos_block + 1)] -=
						sum.mul_real(v0) * v1.conj();
					u2[(j, i_pos - *i_pos_block + 2)] -=
						sum.mul_real(v0) * v2.conj();
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
				let mut a_slice =
					a.rb_mut().submatrix_mut(*i_pos_block, i, n_block, iblock);
				let mut wh_slice = wh.rb_mut().submatrix_mut(
					0,
					0,
					a_slice.nrows(),
					a_slice.ncols(),
				);
				matmul(
					wh_slice.rb_mut(),
					Accum::Replace,
					u2.rb().adjoint(),
					a_slice.rb(),
					one(),
					par,
				);
				a_slice.copy_from(wh_slice.rb());
				i += iblock;
			}
		}
		if istart_m < *i_pos_block {
			let mut i = istart_m;
			while i < *i_pos_block {
				let iblock = Ord::min(*i_pos_block - i, wv.nrows());
				let mut a_slice =
					a.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
				let mut wv_slice = wv.rb_mut().submatrix_mut(
					0,
					0,
					a_slice.nrows(),
					a_slice.ncols(),
				);
				matmul(
					wv_slice.rb_mut(),
					Accum::Replace,
					a_slice.rb(),
					u2.rb(),
					one(),
					par,
				);
				a_slice.copy_from(wv_slice.rb());
				i += iblock;
			}
		}
		if let Some(mut z) = z.rb_mut() {
			let mut i = 0;
			while i < n {
				let iblock = Ord::min(n - i, wv.nrows());
				let mut z_slice =
					z.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
				let mut wv_slice = wv.rb_mut().submatrix_mut(
					0,
					0,
					z_slice.nrows(),
					z_slice.ncols(),
				);
				matmul(
					wv_slice.rb_mut(),
					Accum::Replace,
					z_slice.rb(),
					u2.rb(),
					one(),
					par,
				);
				z_slice.copy_from(wv_slice.rb());
				i += iblock;
			}
		}
		*i_pos_block += n_pos;
	}
}
#[inline(never)]
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
	let ref eps = eps::<T::Real>();
	let ref small_num =
		min_positive::<T::Real>() / eps * from_f64::<T::Real>(n as f64);
	let n_block = ihi - *i_pos_block;
	let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
	u2.fill(zero());
	u2.rb_mut().diagonal_mut().fill(one());
	let mut istart_m = *i_pos_block;
	let mut istop_m = ihi;
	for i_pos_last in *i_pos_block + n_shifts - 2..ihi + n_shifts - 1 {
		let mut i_bulge_start = if i_pos_last + 3 > ihi {
			(i_pos_last + 3 - ihi) / 2
		} else {
			0
		};
		for i_bulge in i_bulge_start..n_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			if i_pos == ihi - 2 {
				let mut v = v.rb_mut().subrows_mut(0, 2).col_mut(i_bulge);
				let mut h = a.rb_mut().subrows_mut(i_pos, 2).col_mut(i_pos - 1);
				let mut beta = h[0].copy();
				let tail = h.rb_mut().subrows_mut(1, 1);
				let HouseholderInfo { tau, .. } =
					make_householder_in_place(&mut beta, tail);
				v[0] = tau.recip().to_cplx();
				v[1] = h[1].copy();
				h[0] = beta;
				h[1] = zero();
				let ref t0 = v[0].conj();
				let ref v1 = v[1].copy();
				let ref t1 = t0 * v1;
				for j in istart_m..i_pos + 2 {
					let ref sum = &a[(j, i_pos)] + (v1 * &a[(j, i_pos + 1)]);
					a[(j, i_pos)] -= sum * t0.conj();
					a[(j, i_pos + 1)] -= sum * t1.conj();
				}
				for j in i_pos..istop_m {
					let ref sum =
						&a[(i_pos, j)] + (v1.conj() * &a[(i_pos + 1, j)]);
					a[(i_pos, j)] -= sum * t0;
					a[(i_pos + 1, j)] -= sum * t1;
				}
				for j in 0..u2.nrows() {
					let ref sum = &u2[(j, i_pos - *i_pos_block)]
						+ v1 * &u2[(j, i_pos - *i_pos_block + 1)];
					u2[(j, i_pos - *i_pos_block)] -= sum * t0.conj();
					u2[(j, i_pos - *i_pos_block + 1)] -= sum * t1.conj();
				}
			} else {
				let mut v = v.rb_mut().col_mut(i_bulge);
				let mut h =
					a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
				let s1 = s[s.nrows() - 1 - 2 * i_bulge].copy();
				let s2 = s[s.nrows() - 1 - 2 * i_bulge - 1].copy();
				move_bulge(h.rb_mut(), v.rb_mut(), s1, s2);
				{
					let ref t0 = v[0].conj();
					let ref v1 = v[1].copy();
					let ref t1 = t0 * v1;
					let ref v2 = v[2].copy();
					let ref t2 = t0 * v2;
					for j in istart_m..i_pos + 3 {
						let ref sum = &a[(j, i_pos)]
							+ v1 * &a[(j, i_pos + 1)]
							+ v2 * &a[(j, i_pos + 2)];
						a[(j, i_pos)] -= sum * t0.conj();
						a[(j, i_pos + 1)] -= sum * t1.conj();
						a[(j, i_pos + 2)] -= sum * t2.conj();
					}
				}
				let ref v0 = v[0].real();
				let ref v1 = v[1].copy();
				let ref v2 = v[2].copy();
				let ref sum = &a[(i_pos, i_pos)]
					+ v1.conj() * &a[(i_pos + 1, i_pos)]
					+ v2.conj() * &a[(i_pos + 2, i_pos)];
				a[(i_pos, i_pos)] -= sum.mul_real(v0);
				a[(i_pos + 1, i_pos)] -= sum.mul_real(v0) * v1;
				a[(i_pos + 2, i_pos)] -= sum.mul_real(v0) * v2;
				if i_pos > ilo && a[(i_pos, i_pos - 1)] != zero() {
					let mut tst1 = a[(i_pos - 1, i_pos - 1)].abs1()
						+ a[(i_pos, i_pos)].abs1();
					if tst1 == zero() {
						if i_pos > ilo + 1 {
							tst1 = tst1 + a[(i_pos - 1, i_pos - 2)].abs1();
						}
						if i_pos > ilo + 2 {
							tst1 = tst1 + a[(i_pos - 1, i_pos - 3)].abs1();
						}
						if i_pos > ilo + 3 {
							tst1 = tst1 + a[(i_pos - 1, i_pos - 4)].abs1();
						}
						if i_pos < ihi - 1 {
							tst1 = tst1 + a[(i_pos + 1, i_pos)].abs1();
						}
						if i_pos < ihi - 2 {
							tst1 = tst1 + a[(i_pos + 2, i_pos)].abs1();
						}
						if i_pos < ihi - 3 {
							tst1 = tst1 + a[(i_pos + 3, i_pos)].abs1();
						}
					}
					if a[(i_pos, i_pos - 1)].abs1() < small_num.fmax(eps * tst1)
					{
						let ref ab = a[(i_pos, i_pos - 1)]
							.abs1()
							.fmax(a[(i_pos - 1, i_pos)].abs1());
						let ref ba = a[(i_pos, i_pos - 1)]
							.abs1()
							.fmin(a[(i_pos - 1, i_pos)].abs1());
						let ref aa = a[(i_pos, i_pos)].abs1().fmax(
							(&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)])
								.abs1(),
						);
						let ref bb = a[(i_pos, i_pos)].abs1().fmin(
							(&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)])
								.abs1(),
						);
						let ref s = aa + ab;
						if ba * (ab / s)
							<= small_num.fmax(eps * (bb * (aa / s)))
						{
							a[(i_pos, i_pos - 1)] = zero();
						}
					}
				}
			}
		}
		i_bulge_start = if i_pos_last + 4 > ihi {
			(i_pos_last + 4 - ihi) / 2
		} else {
			0
		};
		for i_bulge in i_bulge_start..n_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let v = v.rb_mut().col_mut(i_bulge);
			let ref v0 = v[0].real();
			let ref v1 = v[1].copy();
			let ref v2 = v[2].copy();
			for j in i_pos + 1..istop_m {
				let ref sum = &a[(i_pos, j)]
					+ v1.conj() * &a[(i_pos + 1, j)]
					+ v2.conj() * &a[(i_pos + 2, j)];
				a[(i_pos, j)] -= sum.mul_real(v0);
				a[(i_pos + 1, j)] -= sum.mul_real(v0) * v1;
				a[(i_pos + 2, j)] -= sum.mul_real(v0) * v2;
			}
		}
		for i_bulge in i_bulge_start..n_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let v = v.rb_mut().col_mut(i_bulge);
			let ref v0 = v[0].real();
			let ref v1 = v[1].copy();
			let ref v2 = v[2].copy();
			let i1 = (i_pos - *i_pos_block)
				- (i_pos_last + 2 - *i_pos_block - n_shifts);
			let i2 = Ord::min(
				u2.nrows(),
				(i_pos_last - *i_pos_block)
					+ (i_pos_last + 2 - *i_pos_block - n_shifts)
					+ 3,
			);
			for j in i1..i2 {
				let ref sum = &u2[(j, i_pos - *i_pos_block)]
					+ v1 * &u2[(j, i_pos - *i_pos_block + 1)]
					+ v2 * &u2[(j, i_pos - *i_pos_block + 2)];
				u2[(j, i_pos - *i_pos_block)] -= sum.mul_real(v0);
				u2[(j, i_pos - *i_pos_block + 1)] -=
					sum.mul_real(v0) * v1.conj();
				u2[(j, i_pos - *i_pos_block + 2)] -=
					sum.mul_real(v0) * v2.conj();
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
			let mut a_slice =
				a.rb_mut().submatrix_mut(*i_pos_block, i, n_block, iblock);
			let mut wh_slice = wh.rb_mut().submatrix_mut(
				0,
				0,
				a_slice.nrows(),
				a_slice.ncols(),
			);
			matmul(
				wh_slice.rb_mut(),
				Accum::Replace,
				u2.rb().adjoint(),
				a_slice.rb(),
				one(),
				par,
			);
			a_slice.copy_from(wh_slice.rb());
			i += iblock;
		}
	}
	if istart_m < *i_pos_block {
		let mut i = istart_m;
		while i < *i_pos_block {
			let iblock = Ord::min(*i_pos_block - i, wv.nrows());
			let mut a_slice =
				a.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
			let mut wv_slice = wv.rb_mut().submatrix_mut(
				0,
				0,
				a_slice.nrows(),
				a_slice.ncols(),
			);
			matmul(
				wv_slice.rb_mut(),
				Accum::Replace,
				a_slice.rb(),
				u2.rb(),
				one(),
				par,
			);
			a_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	if let Some(mut z) = z.rb_mut() {
		let mut i = 0;
		while i < n {
			let iblock = Ord::min(n - i, wv.nrows());
			let mut z_slice =
				z.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
			let mut wv_slice = wv.rb_mut().submatrix_mut(
				0,
				0,
				z_slice.nrows(),
				z_slice.ncols(),
			);
			matmul(
				wv_slice.rb_mut(),
				Accum::Replace,
				z_slice.rb(),
				u2.rb(),
				one(),
				par,
			);
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
		let h = MatRef::from_row_major_array(
			const {
				&[
					[
						c64::new(0.997386, 0.677592),
						c64::new(0.646064, 0.936948),
						c64::new(0.090948, 0.674011),
					],
					[
						c64::new(0.212396, 0.976794),
						c64::new(0.460270, 0.926436),
						c64::new(0.494441, 0.888187),
					],
					[
						c64::new(0.000000, 0.000000),
						c64::new(0.616652, 0.840012),
						c64::new(0.768245, 0.349193),
					],
				]
			},
		);
		let mut q =
			Mat::from_fn(
				n,
				n,
				|i, j| if i == j { c64::ONE } else { c64::ZERO },
			);
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
							h[(i, j)] = c64::new(rng.random(), rng.random());
						}
					}
				}
				if n <= 128 {
					let mut q = Mat::from_fn(n, n, |i, j| {
						if i == j { c64::ONE } else { c64::ZERO }
					});
					let mut w = Col::zeros(n);
					let mut t = h.cloned();
					super::lahqr(
						true,
						t.as_mut(),
						Some(q.as_mut()),
						w.as_mut(),
						0,
						n,
					);
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
						MemStack::new(&mut MemBuffer::new(
							multishift_qr_scratch::<c64>(
								n,
								n,
								true,
								true,
								Par::Seq,
								auto!(c64),
							),
						)),
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
