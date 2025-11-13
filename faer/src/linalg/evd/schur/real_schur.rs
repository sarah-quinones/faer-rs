use super::*;
use crate::{assert, debug_assert};
use linalg::householder::*;
use linalg::jacobi::JacobiRotation;
use linalg::matmul::matmul;
fn lahqr_eig22<T: RealField>(a00: T, a01: T, a10: T, a11: T) -> ((T, T), (T, T)) {
	let half = from_f64::<T>(0.5);
	let ref s = a00.abs() + a01.abs() + a10.abs() + a11.abs();
	if *s == zero() {
		return ((zero(), zero()), (zero(), zero()));
	}
	let ref a00 = a00 / s;
	let ref a01 = a01 / s;
	let ref a10 = a10 / s;
	let ref a11 = a11 / s;
	let ref tr = (a00 + a11) * half;
	let ref det = (a00 - tr).abs2() + a01 * a10;
	if *det >= zero() {
		let ref rtdisc = det.sqrt();
		((s * (tr + rtdisc), zero()), (s * (tr - rtdisc), zero()))
	} else {
		let ref rtdisc = (-det).sqrt();
		let re = s * tr;
		let im = s * rtdisc;
		((re.copy(), im.copy()), (re, -im))
	}
}
fn lasy2<T: RealField>(tl: MatRef<'_, T>, tr: MatRef<'_, T>, b: MatRef<'_, T>, x: MatMut<'_, T>) -> T {
	let mut x = x;
	let mut info = 0;
	assert!(all(tl.nrows() == 2, tr.nrows() == 2, tl.ncols() == 2, tr.ncols() == 2,));
	let ref eps = eps::<T>();
	let ref smlnum = min_positive::<T>() / eps;
	stack_mat!(btmp, 4, 1, T);
	stack_mat!(tmp, 4, 1, T);
	stack_mat!(t16, 4, 4, T);
	let mut jpiv = [0usize; 4];
	let smin = tr[(0, 0)]
		.abs1()
		.fmax(tr[(0, 1)].abs1())
		.fmax(tr[(1, 0)].abs1())
		.fmax(tr[(1, 1)].abs1())
		.fmax(tl[(0, 0)].abs1())
		.fmax(tl[(0, 1)].abs1())
		.fmax(tl[(1, 0)].abs1())
		.fmax(tl[(1, 1)].abs1());
	let smin = (eps * smin).fmax(smlnum);
	t16[(0, 0)] = &tl[(0, 0)] - &tr[(0, 0)];
	t16[(1, 1)] = &tl[(1, 1)] - &tr[(0, 0)];
	t16[(2, 2)] = &tl[(0, 0)] - &tr[(1, 1)];
	t16[(3, 3)] = &tl[(1, 1)] - &tr[(1, 1)];
	t16[(0, 1)] = tl[(0, 1)].copy();
	t16[(1, 0)] = tl[(1, 0)].copy();
	t16[(2, 3)] = tl[(0, 1)].copy();
	t16[(3, 2)] = tl[(1, 0)].copy();
	t16[(0, 2)] = -&tr[(1, 0)];
	t16[(1, 3)] = -&tr[(1, 0)];
	t16[(2, 0)] = -&tr[(0, 1)];
	t16[(3, 1)] = -&tr[(0, 1)];
	btmp[(0, 0)] = b[(0, 0)].copy();
	btmp[(1, 0)] = b[(1, 0)].copy();
	btmp[(2, 0)] = b[(0, 1)].copy();
	btmp[(3, 0)] = b[(1, 1)].copy();
	let (mut ipsv, mut jpsv);
	#[allow(clippy::needless_range_loop)]
	for i in 0..3usize {
		ipsv = i;
		jpsv = i;
		let mut xmax = zero();
		for ip in i..4 {
			for jp in i..4 {
				if t16[(ip, jp)].abs1() >= xmax {
					xmax = t16[(ip, jp)].abs1();
					ipsv = ip;
					jpsv = jp;
				}
			}
		}
		if ipsv != i {
			crate::perm::swap_rows_idx(t16.rb_mut(), ipsv, i);
			let temp = btmp[(i, 0)].copy();
			btmp[(i, 0)] = btmp[(ipsv, 0)].copy();
			btmp[(ipsv, 0)] = temp;
		}
		if jpsv != i {
			crate::perm::swap_cols_idx(t16.rb_mut(), jpsv, i);
		}
		jpiv[i] = jpsv;
		if t16[(i, i)].abs1() < smin {
			info = 1;
			t16[(i, i)] = smin.copy();
		}
		for j in i + 1..4 {
			t16[(j, i)] = &t16[(j, i)] / &t16[(i, i)];
			btmp[(j, 0)] = &btmp[(j, 0)] - &t16[(j, i)] * &btmp[(i, 0)];
			for k in i + 1..4 {
				t16[(j, k)] = &t16[(j, k)] - &t16[(j, i)] * &t16[(i, k)];
			}
		}
	}
	if t16[(3, 3)].abs1() < smin {
		info = 1;
		t16[(3, 3)] = smin.copy();
	}
	let mut scale = one::<T>();
	let ref eight = from_f64::<T>(8.0);
	if (eight * smlnum) * btmp[(0, 0)].abs1() > t16[(0, 0)].abs1()
		|| (eight * smlnum) * btmp[(1, 0)].abs1() > t16[(1, 1)].abs1()
		|| (eight * smlnum) * btmp[(2, 0)].abs1() > t16[(2, 2)].abs1()
		|| (eight * smlnum) * btmp[(3, 0)].abs1() > t16[(3, 3)].abs1()
	{
		scale = from_f64::<T>(0.125)
			/ btmp[(0, 0)]
				.abs1()
				.fmax(btmp[(1, 0)].abs1())
				.fmax(btmp[(2, 0)].abs1())
				.fmax(btmp[(3, 0)].abs1());
		btmp[(0, 0)] *= &scale;
		btmp[(1, 0)] *= &scale;
		btmp[(2, 0)] *= &scale;
		btmp[(3, 0)] *= &scale;
	}
	for i in 0..4usize {
		let k = 3 - i;
		let ref temp = t16[(k, k)].recip();
		tmp[(k, 0)] = &btmp[(k, 0)] * temp;
		for j in k + 1..4 {
			tmp[(k, 0)] = &tmp[(k, 0)] - temp * &t16[(k, j)] * &tmp[(j, 0)];
		}
	}
	for i in 0..3usize {
		if jpiv[2 - i] != 2 - i {
			let temp = tmp[(2 - i, 0)].copy();
			tmp[(2 - i, 0)] = tmp[(jpiv[2 - i], 0)].copy();
			tmp[(jpiv[2 - i], 0)] = temp;
		}
	}
	x[(0, 0)] = tmp[(0, 0)].copy();
	x[(1, 0)] = tmp[(1, 0)].copy();
	x[(0, 1)] = tmp[(2, 0)].copy();
	x[(1, 1)] = tmp[(3, 0)].copy();
	let _ = info;
	scale
}
#[inline(never)]
fn lahqr_schur22<T: RealField>(mut a: T, mut b: T, mut c: T, mut d: T) -> ((T, T, T, T), (T, T), (T, T), (T, T)) {
	let ref half = from_f64::<T>(0.5);
	let one = one::<T>();
	let ref multpl = from_f64::<T>(4.0);
	let ref eps = eps::<T>();
	let ref safmin = min_positive::<T>();
	let ref safmn2 = (safmin / eps).sqrt();
	let ref safmx2 = safmn2.recip();
	let mut cs;
	let mut sn;
	if c == zero() {
		cs = one;
		sn = zero();
	} else if b == zero() {
		cs = zero();
		sn = one;
		core::mem::swap(&mut d, &mut a);
		b = -&c;
		c = zero();
	} else if a == d && (b > zero()) != (c > zero()) {
		cs = one;
		sn = zero();
	} else {
		let mut temp = &a - &d;
		let mut p = &temp * half;
		let ref bcmax = b.abs().fmax(c.abs());
		let mut bcmin = b.abs().fmin(c.abs());
		if (b > zero()) != (c > zero()) {
			bcmin = -bcmin;
		}
		let ref bcmin = bcmin;
		let mut scale = p.abs().fmax(bcmax);
		let mut z = (&p / &scale) * &p + (bcmax / &scale) * bcmin;
		if z >= multpl * eps {
			let mut __tmp = scale.sqrt() * z.sqrt();
			if p < zero() {
				__tmp = -__tmp;
			}
			z = &p + __tmp;
			a = &d + &z;
			d -= (bcmax / &z) * bcmin;
			let ref tau = (c.abs2() + z.abs2()).sqrt();
			cs = &z / tau;
			sn = &c / tau;
			b = b - c;
			c = zero();
		} else {
			let mut sigma = &b + &c;
			for _ in 0..20 {
				scale = temp.abs().fmax(sigma.abs());
				if scale >= *safmx2 {
					sigma = sigma * safmn2;
					temp = temp * safmn2;
					continue;
				}
				if scale <= *safmn2 {
					sigma = sigma * safmx2;
					temp = temp * safmx2;
					continue;
				}
				break;
			}
			p = temp.mul_pow2(half);
			let mut tau = (sigma.abs2() + temp.abs2()).sqrt();
			cs = (one + sigma.abs() / &tau).mul_pow2(half).sqrt();
			sn = -p / (&tau * &cs);
			if sigma < zero() {
				sn = -sn;
			}
			let ref aa = &a * &cs + &b * &sn;
			let ref bb = -&a * &sn + &b * &cs;
			let ref cc = &c * &cs + &d * &sn;
			let ref dd = -&c * &sn + &d * &cs;
			a = aa * &cs + cc * &sn;
			b = bb * &cs + dd * &sn;
			c = -aa * &sn + cc * &cs;
			d = -bb * &sn + dd * &cs;
			temp = (a + d).mul_pow2(half);
			a = temp.copy();
			d = temp.copy();
			if c != zero() && b != zero() && (b > zero()) == (c > zero()) {
				let ref sab = b.abs().sqrt();
				let ref sac = c.abs().sqrt();
				p = if c > zero() { sab * sac } else { -sab * sac };
				tau = (&b + &c).abs().sqrt().recip();
				a = &temp + &p;
				d = &temp - &p;
				b -= &c;
				c = zero();
				let ref cs1 = sab * &tau;
				let ref sn1 = sac * &tau;
				temp = &cs * cs1 - &sn * sn1;
				sn = &cs * sn1 + &sn * cs1;
				cs = temp;
			}
		}
	}
	let (s1, s2) = if c != zero() {
		let temp = b.abs().sqrt() * c.abs().sqrt();
		((a.copy(), temp.copy()), (d.copy(), -temp))
	} else {
		((a.copy(), zero()), (d.copy(), zero()))
	};
	((a, b, c, d), s1, s2, (cs, sn))
}
fn lahqr_shiftcolumn<T: RealField>(h: MatRef<'_, T>, mut v: ColMut<'_, T>, s1: (T, T), s2: (T, T)) {
	debug_assert!(h.nrows() == h.ncols());
	let n = h.nrows();
	debug_assert!(v.nrows() == n);
	if n == 2 {
		let ref s = (&h[(0, 0)] - &s2.0).abs() + &s2.1.abs() + &h[(1, 0)].abs();
		if *s == zero() {
			v[0] = zero();
			v[1] = zero();
		} else {
			let ref h10s = &h[(1, 0)] / s;
			let v0 = h10s * &h[(0, 1)] + (&h[(0, 0)] - &s1.0) * ((&h[(0, 0)] - &s2.0) / s) - &s1.1 * (&s2.1 / s);
			let v1 = h10s * (&h[(0, 0)] + &h[(1, 1)] - &s1.0 - &s2.0);
			v[0] = v0;
			v[1] = v1;
		}
	} else {
		let ref s = (&h[(0, 0)] - &s2.0).abs() + s2.1.abs() + &h[(1, 0)].abs() + &h[(2, 0)].abs();
		if *s == zero() {
			v[0] = zero();
			v[1] = zero();
			v[2] = zero();
		} else {
			let ref h10s = &h[(1, 0)] / s;
			let ref h20s = &h[(2, 0)] / s;
			let v0 = (&h[(0, 0)] - &s1.0) * ((&h[(0, 0)] - &s2.0) / s) - &s1.1 * (&s2.1 / s) + &h[(0, 1)] * h10s + &h[(0, 2)] * h20s;
			let v1 = h10s * (&h[(0, 0)] + &h[(1, 1)] - &s1.0 - &s2.0) + &h[(1, 2)] * h20s;
			let v2 = h20s * (&h[(0, 0)] + &h[(2, 2)] - &s1.0 - &s2.0) + &h[(2, 1)] * h10s;
			v[0] = v0;
			v[1] = v1;
			v[2] = v2;
		}
	}
}
fn schur_move<T: RealField>(mut a: MatMut<T>, mut q: Option<MatMut<T>>, mut ifst: usize, ilst: &mut usize) -> isize {
	let n = a.nrows();
	if n == 0 {
		return 0;
	}
	if ifst > 0 && (a[(ifst, ifst - 1)] != zero()) {
		ifst -= 1;
	}
	let mut nbf = 1;
	if ifst < n - 1 && (a[(ifst + 1, ifst)] != zero()) {
		nbf = 2;
	}
	if *ilst > 0 && (a[(*ilst, *ilst - 1)] != zero()) {
		*ilst -= 1;
	}
	let mut nbl = 1;
	if (*ilst < n - 1) && (a[(*ilst + 1, *ilst)] != zero()) {
		nbl = 2;
	}
	let mut here = ifst;
	if ifst < *ilst {
		if nbf == 2 && nbl == 1 {
			*ilst -= 1;
		}
		if nbf == 1 && nbl == 2 {
			*ilst += 1;
		}
		while here != *ilst {
			let mut nbnext = 1;
			if (here + nbf + 1 < n) && (a[(here + nbf + 1, here + nbf)] != zero()) {
				nbnext = 2;
			}
			let ierr = schur_swap(a.rb_mut(), q.rb_mut(), here, nbf, nbnext);
			if ierr != 0 {
				*ilst = here;
				return 1;
			}
			here += nbnext;
		}
	} else {
		while here != *ilst {
			let mut nbnext = 1;
			if here > 1 && (a[(here - 1, here - 2)] != zero()) {
				nbnext = 2;
			}
			let ierr = schur_swap(a.rb_mut(), q.rb_mut(), here - nbnext, nbnext, nbf);
			if ierr != 0 {
				*ilst = here;
				return 1;
			}
			here -= nbnext;
		}
	}
	0
}
pub(crate) fn schur_swap_11<T: RealField>(mut a: MatMut<T>, mut q: Option<MatMut<T>>, j0: usize) -> isize {
	let n = a.nrows();
	let j1 = j0 + 1;
	let j2 = j0 + 2;
	let t00 = a[(j0, j0)].copy();
	let t11 = a[(j1, j1)].copy();
	let temp = a[(j0, j1)].copy();
	let temp2 = &t11 - &t00;
	let (rot, _) = JacobiRotation::rotg(temp, temp2);
	a[(j1, j1)] = t00;
	a[(j0, j0)] = t11;
	if j2 < n {
		rot.apply_on_the_right_in_place(a.rb_mut().transpose_mut().get_mut(j2.., ..).two_cols_mut(j0, j1));
	}
	if j0 > 0 {
		rot.apply_on_the_right_in_place(a.rb_mut().get_mut(..j0, ..).two_cols_mut(j0, j1));
	}
	if let Some(q) = q.rb_mut() {
		rot.apply_on_the_right_in_place(q.two_cols_mut(j0, j1));
	}
	0
}
pub(crate) fn schur_swap_12<T: RealField>(mut a: MatMut<T>, mut q: Option<MatMut<T>>, j0: usize) -> isize {
	let n = a.nrows();
	let j1 = j0 + 1;
	let j2 = j0 + 2;
	let j3 = j0 + 3;
	stack_mat!(b, 3, 2, T);
	b[(0, 0)] = a[(j0, j1)].copy();
	b[(1, 0)] = &a[(j1, j1)] - &a[(j0, j0)];
	b[(2, 0)] = a[(j2, j1)].copy();
	b[(0, 1)] = a[(j0, j2)].copy();
	b[(1, 1)] = a[(j1, j2)].copy();
	b[(2, 1)] = &a[(j2, j2)] - &a[(j0, j0)];
	let mut v1 = b.rb_mut().col_mut(0);
	let (head, tail) = v1.rb_mut().split_at_row_mut(1);
	let HouseholderInfo { tau: tau1, .. } = make_householder_in_place(head.at_mut(0), tail);
	let ref tau1 = tau1.recip();
	let ref v11 = b[(1, 0)].copy();
	let ref v12 = b[(2, 0)].copy();
	let ref sum = &b[(0, 1)] + v11 * &b[(1, 1)] + v12 * &b[(2, 1)];
	b[(0, 1)] -= sum * tau1;
	b[(1, 1)] -= sum * tau1 * v11;
	b[(2, 1)] -= sum * tau1 * v12;
	let mut v2 = b.rb_mut().col_mut(1).subrows_mut(1, 2);
	let (head, tail) = v2.rb_mut().split_at_row_mut(1);
	let HouseholderInfo { tau: tau2, .. } = make_householder_in_place(head.at_mut(0), tail);
	let ref tau2 = tau2.recip();
	let ref v21 = v2[1].copy();
	for j in j0..n {
		let ref sum = &a[(j0, j)] + v11 * &a[(j1, j)] + v12 * &a[(j2, j)];
		a[(j0, j)] -= sum * tau1;
		a[(j1, j)] -= sum * tau1 * v11;
		a[(j2, j)] -= sum * tau1 * v12;
		let ref sum = &a[(j1, j)] + v21 * &a[(j2, j)];
		a[(j1, j)] -= sum * tau2;
		a[(j2, j)] -= sum * tau2 * v21;
	}
	for j in 0..j3 {
		let ref sum = &a[(j, j0)] + v11 * &a[(j, j1)] + v12 * &a[(j, j2)];
		a[(j, j0)] -= sum * tau1;
		a[(j, j1)] -= sum * tau1 * v11;
		a[(j, j2)] -= sum * tau1 * v12;
		let ref sum = &a[(j, j1)] + v21 * &a[(j, j2)];
		a[(j, j1)] -= sum * tau2;
		a[(j, j2)] -= sum * tau2 * v21;
	}
	if let Some(mut q) = q.rb_mut() {
		for j in 0..n {
			let ref sum = &q[(j, j0)] + v11 * &q[(j, j1)] + v12 * &q[(j, j2)];
			q[(j, j0)] -= sum * tau1;
			q[(j, j1)] -= sum * tau1 * v11;
			q[(j, j2)] -= sum * tau1 * v12;
			let ref sum = &q[(j, j1)] + v21 * &q[(j, j2)];
			q[(j, j1)] -= sum * tau2;
			q[(j, j2)] -= sum * tau2 * v21;
		}
	}
	a[(j2, j0)] = zero();
	a[(j2, j1)] = zero();
	0
}
pub(crate) fn schur_swap_21<T: RealField>(mut a: MatMut<T>, mut q: Option<MatMut<T>>, j0: usize) -> isize {
	let n = a.nrows();
	let j1 = j0 + 1;
	let j2 = j0 + 2;
	let j3 = j0 + 3;
	stack_mat!(b, 3, 2, T);
	b[(0, 0)] = a[(j1, j2)].copy();
	b[(1, 0)] = &a[(j1, j1)] - &a[(j2, j2)];
	b[(2, 0)] = a[(j1, j0)].copy();
	b[(0, 1)] = a[(j0, j2)].copy();
	b[(1, 1)] = a[(j0, j1)].copy();
	b[(2, 1)] = &a[(j0, j0)] - &a[(j2, j2)];
	let mut v1 = b.rb_mut().col_mut(0);
	let (head, tail) = v1.rb_mut().split_at_row_mut(1);
	let HouseholderInfo { tau: tau1, .. } = make_householder_in_place(head.at_mut(0), tail);
	let ref tau1 = tau1.recip();
	let ref v11 = v1[1].copy();
	let ref v12 = v1[2].copy();
	let ref sum = &b[(0, 1)] + v11 * &b[(1, 1)] + v12 * &b[(2, 1)];
	b[(0, 1)] -= sum * tau1;
	b[(1, 1)] -= sum * tau1 * v11;
	b[(2, 1)] -= sum * tau1 * v12;
	let mut v2 = b.rb_mut().col_mut(1).subrows_mut(1, 2);
	let (head, tail) = v2.rb_mut().split_at_row_mut(1);
	let HouseholderInfo { tau: tau2, .. } = make_householder_in_place(head.at_mut(0), tail);
	let ref tau2 = tau2.recip();
	let ref v21 = v2[1].copy();
	for j in j0..n {
		let ref sum = &a[(j2, j)] + v11 * &a[(j1, j)] + v12 * &a[(j0, j)];
		a[(j2, j)] -= sum * tau1;
		a[(j1, j)] -= sum * tau1 * v11;
		a[(j0, j)] -= sum * tau1 * v12;
		let ref sum = &a[(j1, j)] + v21 * &a[(j0, j)];
		a[(j1, j)] -= sum * tau2;
		a[(j0, j)] -= sum * tau2 * v21;
	}
	for j in 0..j3 {
		let ref sum = &a[(j, j2)] + v11 * &a[(j, j1)] + v12 * &a[(j, j0)];
		a[(j, j2)] -= sum * tau1;
		a[(j, j1)] -= sum * tau1 * v11;
		a[(j, j0)] -= sum * tau1 * v12;
		let ref sum = &a[(j, j1)] + v21 * &a[(j, j0)];
		a[(j, j1)] -= sum * tau2;
		a[(j, j0)] -= sum * tau2 * v21;
	}
	if let Some(mut q) = q.rb_mut() {
		for j in 0..n {
			let ref sum = &q[(j, j2)] + v11 * &q[(j, j1)] + v12 * &q[(j, j0)];
			q[(j, j2)] -= sum * tau1;
			q[(j, j1)] -= sum * tau1 * v11;
			q[(j, j0)] -= sum * tau1 * v12;
			let ref sum = &q[(j, j1)] + v21 * &q[(j, j0)];
			q[(j, j1)] -= sum * tau2;
			q[(j, j0)] -= sum * tau2 * v21;
		}
	}
	a[(j1, j0)] = zero();
	a[(j2, j0)] = zero();
	0
}
pub(crate) fn schur_swap_22<T: RealField>(mut a: MatMut<T>, mut q: Option<MatMut<T>>, j0: usize) -> isize {
	let n = a.nrows();
	let ref epsilon = eps::<T>();
	let ref zero_threshold = min_positive::<T>();
	let j1 = j0 + 1;
	let j2 = j0 + 2;
	let j3 = j0 + 3;
	stack_mat!(d, 4, 4, T);
	let ad_slice = a.rb().submatrix(j0, j0, 4, 4);
	d.copy_from(ad_slice);
	let mut dnorm = zero::<T>();
	z!(d.rb()).for_each(|unzip!(d)| dnorm = dnorm.fmax(d.abs()));
	let eps = epsilon;
	let ref small_num = zero_threshold / eps;
	let thresh = (from_f64::<T>(10.0) * eps * dnorm).fmax(small_num);
	stack_mat!(v, 4, 2, T);
	let mut x = v.rb_mut().submatrix_mut(0, 0, 2, 2);
	let (tl, b, _, tr) = d.rb().split_at(2, 2);
	let ref scale = lasy2(tl, tr, b, x.rb_mut());
	v[(2, 0)] = -scale;
	v[(2, 1)] = zero();
	v[(3, 0)] = zero();
	v[(3, 1)] = -scale;
	let mut v1 = v.rb_mut().col_mut(0);
	let (head, tail) = v1.rb_mut().split_at_row_mut(1);
	let HouseholderInfo { tau: tau1, .. } = make_householder_in_place(head.at_mut(0), tail);
	let ref tau1 = tau1.recip();
	let ref v11 = v1[1].copy();
	let ref v12 = v1[2].copy();
	let ref v13 = v1[3].copy();
	let ref sum = &v[(0, 1)] + v11 * &v[(1, 1)] + v12 * &v[(2, 1)] + v13 * &v[(3, 1)];
	v[(0, 1)] -= sum * tau1;
	v[(1, 1)] -= sum * tau1 * v11;
	v[(2, 1)] -= sum * tau1 * v12;
	v[(3, 1)] -= sum * tau1 * v13;
	let mut v2 = v.rb_mut().col_mut(1).subrows_mut(1, 3);
	let (head, tail) = v2.rb_mut().split_at_row_mut(1);
	let HouseholderInfo { tau: tau2, .. } = make_householder_in_place(head.at_mut(0), tail);
	let ref tau2 = tau2.recip();
	let ref v21 = v2[1];
	let ref v22 = v2[2];
	for j in 0..4 {
		let ref sum = &d[(0, j)] + v11 * &d[(1, j)] + v12 * &d[(2, j)] + v13 * &d[(3, j)];
		d[(0, j)] -= sum * tau1;
		d[(1, j)] -= sum * tau1 * v11;
		d[(2, j)] -= sum * tau1 * v12;
		d[(3, j)] -= sum * tau1 * v13;
		let ref sum = &d[(1, j)] + v21 * &d[(2, j)] + v22 * &d[(3, j)];
		d[(1, j)] -= sum * tau2;
		d[(2, j)] -= sum * tau2 * v21;
		d[(3, j)] -= sum * tau2 * v22;
	}
	for j in 0..4 {
		let ref sum = &d[(j, 0)] + v11 * &d[(j, 1)] + v12 * &d[(j, 2)] + v13 * &d[(j, 3)];
		d[(j, 0)] -= sum * tau1;
		d[(j, 1)] -= sum * tau1 * v11;
		d[(j, 2)] -= sum * tau1 * v12;
		d[(j, 3)] -= sum * tau1 * v13;
		let ref sum = &d[(j, 1)] + v21 * &d[(j, 2)] + v22 * &d[(j, 3)];
		d[(j, 1)] -= sum * tau2;
		d[(j, 2)] -= sum * tau2 * v21;
		d[(j, 3)] -= sum * tau2 * v22;
	}
	if d[(2, 0)].abs().fmax(d[(2, 1)].abs()).fmax(d[(3, 0)].abs()).fmax(d[(3, 1)].abs()) > thresh {
		return 1;
	}
	for j in j0..n {
		let ref sum = &a[(j0, j)] + v11 * &a[(j1, j)] + v12 * &a[(j2, j)] + v13 * &a[(j3, j)];
		a[(j0, j)] -= sum * tau1;
		a[(j1, j)] -= sum * tau1 * v11;
		a[(j2, j)] -= sum * tau1 * v12;
		a[(j3, j)] -= sum * tau1 * v13;
		let ref sum = &a[(j1, j)] + v21 * &a[(j2, j)] + v22 * &a[(j3, j)];
		a[(j1, j)] -= sum * tau2;
		a[(j2, j)] -= sum * tau2 * v21;
		a[(j3, j)] -= sum * tau2 * v22;
	}
	for j in 0..j0 + 4 {
		let ref sum = &a[(j, j0)] + v11 * &a[(j, j1)] + v12 * &a[(j, j2)] + v13 * &a[(j, j3)];
		a[(j, j0)] -= sum * tau1;
		a[(j, j1)] -= sum * tau1 * v11;
		a[(j, j2)] -= sum * tau1 * v12;
		a[(j, j3)] -= sum * tau1 * v13;
		let ref sum = &a[(j, j1)] + v21 * &a[(j, j2)] + v22 * &a[(j, j3)];
		a[(j, j1)] -= sum * tau2;
		a[(j, j2)] -= sum * tau2 * v21;
		a[(j, j3)] -= sum * tau2 * v22;
	}
	if let Some(mut q) = q.rb_mut() {
		for j in 0..n {
			let ref sum = &q[(j, j0)] + v11 * &q[(j, j1)] + v12 * &q[(j, j2)] + v13 * &q[(j, j3)];
			q[(j, j0)] -= sum * tau1;
			q[(j, j1)] -= sum * tau1 * v11;
			q[(j, j2)] -= sum * tau1 * v12;
			q[(j, j3)] -= sum * tau1 * v13;
			let ref sum = &q[(j, j1)] + v21 * &q[(j, j2)] + v22 * &q[(j, j3)];
			q[(j, j1)] -= sum * tau2;
			q[(j, j2)] -= sum * tau2 * v21;
			q[(j, j3)] -= sum * tau2 * v22;
		}
	}
	a[(j2, j0)] = zero();
	a[(j2, j1)] = zero();
	a[(j3, j0)] = zero();
	a[(j3, j1)] = zero();
	0
}
pub(crate) fn schur_swap<T: RealField>(mut a: MatMut<T>, mut q: Option<MatMut<T>>, j0: usize, n1: usize, n2: usize) -> isize {
	let n = a.nrows();
	let j1 = j0 + 1;
	let j2 = j0 + 2;
	if n1 == 2 && (a[(j1, j0)] == zero()) {
		schur_swap(a.rb_mut(), q.rb_mut(), j1, 1, n2);
		schur_swap(a.rb_mut(), q.rb_mut(), j0, 1, n2);
		return 0;
	}
	if n2 == 2 && (a[(j0 + n1 + 1, j0 + n1)] == zero()) {
		schur_swap(a.rb_mut(), q.rb_mut(), j0, n1, 1);
		schur_swap(a.rb_mut(), q.rb_mut(), j1, n1, 1);
		return 0;
	}
	let schur_swap_impl = match (n1, n2) {
		(1, 1) => schur_swap_11::<T>,
		(2, 1) => schur_swap_21::<T>,
		(1, 2) => schur_swap_12::<T>,
		(2, 2) => schur_swap_22::<T>,
		_ => panic!(),
	};
	schur_swap_impl(a.rb_mut(), q.rb_mut(), j0);
	if n2 == 2 {
		let ((a00, a01, a10, a11), _, _, (cs, sn)) = lahqr_schur22(a[(j0, j0)].copy(), a[(j0, j1)].copy(), a[(j1, j0)].copy(), a[(j1, j1)].copy());
		let rot = JacobiRotation { c: cs, s: sn };
		a[(j0, j0)] = a00;
		a[(j0, j1)] = a01;
		a[(j1, j0)] = a10;
		a[(j1, j1)] = a11;
		if j2 < n {
			rot.apply_on_the_right_in_place(a.rb_mut().transpose_mut().get_mut(j2.., ..).two_cols_mut(j0, j1));
		}
		if j0 > 0 {
			rot.apply_on_the_right_in_place(a.rb_mut().get_mut(..j0, ..).two_cols_mut(j0, j1));
		}
		if let Some(q) = q.rb_mut() {
			rot.apply_on_the_right_in_place(q.two_cols_mut(j0, j1));
		}
	}
	if n1 == 2 {
		let j0 = j0 + n2;
		let j1 = j1 + n2;
		let j2 = j2 + n2;
		let ((a00, a01, a10, a11), _, _, (cs, sn)) = lahqr_schur22(a[(j0, j0)].copy(), a[(j0, j1)].copy(), a[(j1, j0)].copy(), a[(j1, j1)].copy());
		let rot = JacobiRotation { c: cs, s: sn };
		a[(j0, j0)] = a00;
		a[(j0, j1)] = a01;
		a[(j1, j0)] = a10;
		a[(j1, j1)] = a11;
		if j2 < n {
			rot.apply_on_the_right_in_place(a.rb_mut().transpose_mut().get_mut(j2.., ..).two_cols_mut(j0, j1));
		}
		if j0 > 0 {
			rot.apply_on_the_right_in_place(a.rb_mut().get_mut(..j0, ..).two_cols_mut(j0, j1));
		}
		if let Some(q) = q.rb_mut() {
			rot.apply_on_the_right_in_place(q.two_cols_mut(j0, j1));
		}
	}
	0
}
fn aggressive_early_deflation<T: RealField>(
	want_t: bool,
	mut a: MatMut<'_, T>,
	mut z: Option<MatMut<'_, T>>,
	mut s_re: ColMut<'_, T>,
	mut s_im: ColMut<'_, T>,
	ilo: usize,
	ihi: usize,
	nw: usize,
	par: Par,
	mut stack: &mut MemStack,
	params: SchurParams,
) -> (usize, usize) {
	let n = a.nrows();
	let epsilon = eps::<T>();
	let zero_threshold = min_positive::<T>();
	let nw_max = (n - 3) / 3;
	let ref eps = epsilon;
	let ref small_num = zero_threshold / eps * from_f64::<T>(n as f64);
	let jw = Ord::min(Ord::min(nw, ihi - ilo), nw_max);
	let kwtop = ihi - jw;
	let mut s_spike = if kwtop == ilo { zero() } else { a[(kwtop, kwtop - 1)].copy() };
	if kwtop + 1 == ihi {
		s_re[kwtop] = a[(kwtop, kwtop)].copy();
		s_im[kwtop] = zero();
		let mut ns = 1;
		let mut nd = 0;
		if s_spike.abs() <= (eps * a[(kwtop, kwtop)].abs()).fmax(small_num) {
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
	let mut s_re_window = unsafe { s_re.rb().subrows(kwtop, ihi - kwtop).const_cast() };
	let mut s_im_window = unsafe { s_im.rb().subrows(kwtop, ihi - kwtop).const_cast() };
	z!(tw.rb_mut()).for_each_triangular_lower(linalg::zip::Diag::Include, |unzip!(x)| *x = zero());
	for j in 0..jw {
		for i in 0..Ord::min(j + 2, jw) {
			tw[(i, j)] = a_window[(i, j)].copy();
		}
	}
	v.fill(zero());
	v.rb_mut().diagonal_mut().fill(one::<T>());
	let infqr = if true || jw < params.blocking_threshold {
		lahqr(true, tw.rb_mut(), Some(v.rb_mut()), s_re_window.rb_mut(), s_im_window.rb_mut(), 0, jw)
	} else {
		let infqr = multishift_qr(
			true,
			tw.rb_mut(),
			Some(v.rb_mut()),
			s_re_window.rb_mut(),
			s_im_window.rb_mut(),
			0,
			jw,
			par,
			stack.rb_mut(),
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
		let ref s_spike = s_spike;
		let mut bulge = false;
		if ns > 1 && (tw[(ns - 1, ns - 2)] != zero()) {
			bulge = true;
		}
		if !bulge {
			let mut __tmp = tw[(ns - 1, ns - 1)].abs();
			if __tmp == zero() {
				__tmp = s_spike.abs();
			}
			if s_spike.abs() * v[(0, ns - 1)].abs() <= (eps * __tmp).fmax(small_num) {
				ns -= 1;
			} else {
				let ifst = ns - 1;
				schur_move(tw.rb_mut(), Some(v.rb_mut()), ifst, &mut ilst);
				ilst += 1;
			}
		} else {
			let mut __tmp = tw[(ns - 1, ns - 1)].abs() + tw[(ns - 1, ns - 2)].abs().sqrt() * tw[(ns - 2, ns - 1)].abs().sqrt();
			if __tmp == zero() {
				__tmp = s_spike.abs();
			}
			if (s_spike * &v[(0, ns - 1)]).abs().fmax((s_spike * &v[(0, ns - 2)]).abs()) <= (eps * __tmp).fmax(small_num) {
				ns -= 2;
			} else {
				let ifst = ns - 2;
				schur_move(tw.rb_mut(), Some(v.rb_mut()), ifst, &mut ilst);
				ilst += 2;
			}
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
	let mut sorting_window_size = jw as isize;
	while !sorted {
		sorted = true;
		let mut ilst = 0isize;
		let mut i1 = ns;
		while i1 as isize + 1 < sorting_window_size {
			let mut n1 = 1;
			if tw[(i1 + 1, i1)] != zero() {
				n1 = 2;
			}
			if i1 + n1 == jw {
				ilst -= n1 as isize;
				break;
			}
			let i2 = i1 + n1;
			let mut n2 = 1;
			if i2 + 1 < jw && (tw[(i2 + 1, i2)] != zero()) {
				n2 = 2;
			}
			let (ev1, ev2);
			if n1 == 1 {
				ev1 = tw[(i1, i1)].abs();
			} else {
				ev1 = tw[(i1, i1)].abs() + tw[(i1 + 1, i1)].abs().sqrt() * tw[(i1, i1 + 1)].abs().sqrt();
			}
			if n2 == 1 {
				ev2 = tw[(i2, i2)].abs();
			} else {
				ev2 = tw[(i2, i2)].abs() + tw[(i2 + 1, i2)].abs().sqrt() * tw[(i2, i2 + 1)].abs().sqrt();
			}
			if ev1 >= ev2 {
				i1 = i2;
			} else {
				sorted = false;
				let ierr = schur_swap(tw.rb_mut(), Some(v.rb_mut()), i1, n1, n2);
				if ierr == 0 {
					i1 += n2;
				} else {
					i1 = i2;
				}
				ilst = i1 as isize;
			}
		}
		sorting_window_size = ilst;
	}
	let mut i = 0;
	while i < jw {
		let mut n1 = 1;
		if i + 1 < jw && (tw[(i + 1, i)] != zero()) {
			n1 = 2;
		}
		if n1 == 1 {
			s_re[kwtop + i] = tw[(i, i)].copy();
			s_im[kwtop + i] = zero();
		} else {
			let ((s1_re, s1_im), (s2_re, s2_im)) =
				lahqr_eig22(tw[(i, i)].copy(), tw[(i, i + 1)].copy(), tw[(i + 1, i)].copy(), tw[(i + 1, i + 1)].copy());
			s_re[kwtop + i] = s1_re;
			s_im[kwtop + i] = s1_im;
			s_re[kwtop + i + 1] = s2_re;
			s_im[kwtop + i + 1] = s2_im;
		}
		i += n1;
	}
	if s_spike != zero() {
		{
			let mut vv = wv.rb_mut().col_mut(0).subrows_mut(0, ns);
			for i in 0..ns {
				vv[i] = v[(0, i)].copy();
			}
			let mut head = vv[0].copy();
			let tail = vv.rb_mut().subrows_mut(1, ns - 1);
			let HouseholderInfo { tau, .. } = make_householder_in_place(&mut head, tail);
			let beta = head.copy();
			vv[0] = one::<T>();
			let ref tau = tau.recip();
			{
				let mut tw_slice = tw.rb_mut().submatrix_mut(0, 0, ns, jw);
				let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
				let mut tmp = tmp.as_mat_mut().transpose_mut();
				matmul(tmp.rb_mut(), Accum::Replace, vv.rb().adjoint().as_mat(), tw_slice.rb(), one::<T>(), par);
				matmul(tw_slice.rb_mut(), Accum::Add, vv.rb().as_mat(), tmp.as_ref(), -tau, par);
			}
			{
				let mut tw_slice2 = tw.rb_mut().submatrix_mut(0, 0, jw, ns);
				let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
				let mut tmp = tmp.as_mat_mut();
				matmul(tmp.rb_mut(), Accum::Replace, tw_slice2.rb(), vv.rb().as_mat(), one::<T>(), par);
				matmul(tw_slice2.rb_mut(), Accum::Add, tmp.as_ref(), vv.rb().adjoint().as_mat(), -tau, par);
			}
			{
				let mut v_slice = v.rb_mut().submatrix_mut(0, 0, jw, ns);
				let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
				let mut tmp = tmp.as_mat_mut();
				matmul(tmp.rb_mut(), Accum::Replace, v_slice.rb(), vv.rb().as_mat(), one::<T>(), par);
				matmul(v_slice.rb_mut(), Accum::Add, tmp.as_ref(), vv.rb().adjoint().as_mat(), -tau, par);
			}
			vv[0] = beta;
		}
		{
			let mut householder = wv.rb_mut().col_mut(0).subrows_mut(0, ns - 1);
			hessenberg::hessenberg_in_place(
				tw.rb_mut().submatrix_mut(0, 0, ns, ns),
				householder.rb_mut().as_mat_mut().transpose_mut(),
				par,
				stack.rb_mut(),
				Default::default(),
			);
			apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
				tw.rb().submatrix(1, 0, ns - 1, ns - 1),
				householder.rb().transpose().as_mat(),
				Conj::Yes,
				unsafe { tw.rb().submatrix(1, ns, ns - 1, jw - ns).const_cast() },
				par,
				stack.rb_mut(),
			);
			apply_block_householder_sequence_on_the_right_in_place_with_conj(
				tw.rb().submatrix(1, 0, ns - 1, ns - 1),
				householder.rb().transpose().as_mat(),
				Conj::No,
				v.rb_mut().submatrix_mut(0, 1, jw, ns - 1),
				par,
				stack.rb_mut(),
			);
		}
	}
	if kwtop > 0 {
		a[(kwtop, kwtop - 1)] = &s_spike * &v[(0, 0)];
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
			let mut a_slice = a.rb_mut().submatrix_mut(kwtop, i, ihi - kwtop, iblock);
			let mut wh_slice = wh.rb_mut().submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
			matmul(wh_slice.rb_mut(), Accum::Replace, v.rb().adjoint(), a_slice.rb(), one::<T>(), par);
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
			matmul(wv_slice.rb_mut(), Accum::Replace, a_slice.rb(), v.rb(), one::<T>(), par);
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
			matmul(wv_slice.rb_mut(), Accum::Replace, z_slice.rb(), v.rb(), one::<T>(), par);
			z_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	(ns, nd)
}
fn move_bulge<T: RealField>(mut h: MatMut<'_, T>, mut v: ColMut<'_, T>, s1: (T, T), s2: (T, T)) {
	let ref epsilon = eps::<T>();
	let ref v0 = v[0].copy();
	let ref v1 = v[1].copy();
	let ref v2 = v[2].copy();
	let ref refsum = v2.mul_real(v0) * &h[(3, 2)];
	h[(3, 0)] = -refsum;
	h[(3, 1)] = -refsum * v1;
	h[(3, 2)] -= refsum * v2;
	v[0] = h[(1, 0)].copy();
	v[1] = h[(2, 0)].copy();
	v[2] = h[(3, 0)].copy();
	let mut beta = v[0].copy();
	let tail = v.rb_mut().subrows_mut(1, 2);
	let HouseholderInfo { tau, .. } = make_householder_in_place(&mut beta, tail);
	v[0] = tau.recip();
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
		let HouseholderInfo { tau, .. } = make_householder_in_place(&mut beta_unused, tail);
		vt[0] = tau.recip();
		let ref vt0 = vt[0].copy();
		let ref vt1 = vt[1].copy();
		let ref vt2 = vt[2].copy();
		let ref refsum = vt0 * &h[(1, 0)] + vt1 * &h[(2, 0)];
		if (&h[(2, 0)] - refsum * vt1).abs1() + (refsum * vt2).abs1() > epsilon * (h[(0, 0)].abs1() + h[(1, 1)].abs1() + h[(2, 2)].abs1()) {
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
fn multishift_qr_sweep<T: RealField>(
	want_t: bool,
	a: MatMut<T>,
	mut z: Option<MatMut<T>>,
	s_re: ColMut<T>,
	s_im: ColMut<T>,
	ilo: usize,
	ihi: usize,
	par: Par,
	stack: &mut MemStack,
) {
	let n = a.nrows();
	assert!(n >= 12);
	let (mut v, _stack) = crate::linalg::temp_mat_zeroed(3, s_re.nrows() / 2, stack);
	let mut v = v.as_mat_mut();
	let n_block_max = (n - 3) / 3;
	let n_shifts_max = Ord::min(ihi - ilo - 1, Ord::max(2, 3 * (n_block_max / 4)));
	let mut n_shifts = Ord::min(s_re.nrows(), n_shifts_max);
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
		s_re.rb(),
		s_im.rb(),
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
		s_re.rb(),
		s_im.rb(),
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
		s_re.rb(),
		s_im.rb(),
		&mut i_pos_block,
		par,
	);
}
#[inline(never)]
fn introduce_bulges<T: RealField>(
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
	s_re: ColRef<'_, T>,
	s_im: ColRef<'_, T>,
	i_pos_block: &mut usize,
	parallelism: Par,
) {
	let n = a.nrows();
	let ref eps = eps::<T>();
	let ref small_num = min_positive::<T>() / eps * from_f64::<T>(n as f64);
	let n_block = Ord::min(n_block_desired, ihi - ilo);
	let mut istart_m = ilo;
	let mut istop_m = ilo + n_block;
	let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
	u2.fill(zero());
	u2.rb_mut().diagonal_mut().fill(one::<T>());
	for i_pos_last in ilo..ilo + n_block - 2 {
		let n_active_bulges = Ord::min(n_bulges, ((i_pos_last - ilo) / 2) + 1);
		for i_bulge in 0..n_active_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let mut v = v.rb_mut().col_mut(i_bulge);
			if i_pos == ilo {
				let h = a.rb().submatrix(ilo, ilo, 3, 3);
				let s1_re = s_re[s_re.nrows() - 1 - 2 * i_bulge].copy();
				let s1_im = s_im[s_im.nrows() - 1 - 2 * i_bulge].copy();
				let s2_re = s_re[s_re.nrows() - 1 - 2 * i_bulge - 1].copy();
				let s2_im = s_im[s_im.nrows() - 1 - 2 * i_bulge - 1].copy();
				lahqr_shiftcolumn(h, v.rb_mut(), (s1_re, s1_im), (s2_re, s2_im));
				debug_assert!(v.nrows() == 3);
				let mut head = v[0].copy();
				let tail = v.rb_mut().subrows_mut(1, 2);
				let HouseholderInfo { tau, .. } = make_householder_in_place(&mut head, tail);
				v[0] = tau.recip();
			} else {
				let mut h = a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
				let s1_re = s_re[s_re.nrows() - 1 - 2 * i_bulge].copy();
				let s1_im = s_im[s_im.nrows() - 1 - 2 * i_bulge].copy();
				let s2_re = s_re[s_re.nrows() - 1 - 2 * i_bulge - 1].copy();
				let s2_im = s_im[s_im.nrows() - 1 - 2 * i_bulge - 1].copy();
				move_bulge(h.rb_mut(), v.rb_mut(), (s1_re, s1_im), (s2_re, s2_im));
			}
			let ref v0 = v[0].copy();
			let ref v1 = v[1];
			let ref v2 = v[2];
			for j in istart_m..i_pos + 3 {
				let ref sum = &a[(j, i_pos)] + v1 * &a[(j, i_pos + 1)] + v2 * &a[(j, i_pos + 2)];
				a[(j, i_pos)] -= sum.mul_real(v0);
				a[(j, i_pos + 1)] -= sum.mul_real(v0) * v1;
				a[(j, i_pos + 2)] -= sum.mul_real(v0) * v2;
			}
			let ref sum = &a[(i_pos, i_pos)] + v1 * &a[(i_pos + 1, i_pos)] + v2 * &a[(i_pos + 2, i_pos)];
			a[(i_pos, i_pos)] -= sum.mul_real(v0);
			a[(i_pos + 1, i_pos)] -= sum.mul_real(v0) * v1;
			a[(i_pos + 2, i_pos)] -= sum.mul_real(v0) * v2;
			if i_pos > ilo && (a[(i_pos, i_pos - 1)] != zero()) {
				let mut tst1 = a[(i_pos - 1, i_pos - 1)].abs1() + a[(i_pos, i_pos)].abs1();
				if tst1 == zero() {
					if i_pos > ilo + 1 {
						tst1 += a[(i_pos - 1, i_pos - 2)].abs1();
					}
					if i_pos > ilo + 2 {
						tst1 += a[(i_pos - 1, i_pos - 3)].abs1();
					}
					if i_pos > ilo + 3 {
						tst1 += a[(i_pos - 1, i_pos - 4)].abs1();
					}
					if i_pos < ihi - 1 {
						tst1 += a[(i_pos + 1, i_pos)].abs1();
					}
					if i_pos < ihi - 2 {
						tst1 += a[(i_pos + 2, i_pos)].abs1();
					}
					if i_pos < ihi - 3 {
						tst1 += a[(i_pos + 3, i_pos)].abs1();
					}
				}
				if a[(i_pos, i_pos - 1)].abs1() < (eps * tst1).fmax(small_num) {
					let ref ab = a[(i_pos, i_pos - 1)].abs1().fmax(a[(i_pos - 1, i_pos)].abs1());
					let ref ba = a[(i_pos, i_pos - 1)].abs1().fmin(a[(i_pos - 1, i_pos)].abs1());
					let ref aa = a[(i_pos, i_pos)].abs1().fmax((&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)]).abs1());
					let ref bb = a[(i_pos, i_pos)].abs1().fmin((&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)]).abs1());
					let ref s = aa + ab;
					if (ba * (ab / s)) <= (eps * (bb * (aa / s))).fmax(small_num) {
						a[(i_pos, i_pos - 1)] = zero();
					}
				}
			}
		}
		for i_bulge in 0..n_active_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let v = v.rb_mut().col_mut(i_bulge);
			let ref v0 = v[0].copy();
			let ref v1 = v[1];
			let ref v2 = v[2];
			for j in i_pos + 1..istop_m {
				let ref sum = &a[(i_pos, j)] + v1 * &a[(i_pos + 1, j)] + v2 * &a[(i_pos + 2, j)];
				a[(i_pos, j)] -= sum.mul_real(v0);
				a[(i_pos + 1, j)] -= sum.mul_real(v0) * v1;
				a[(i_pos + 2, j)] -= sum.mul_real(v0) * v2;
			}
		}
		for i_bulge in 0..n_active_bulges {
			let i_pos = i_pos_last - 2 * i_bulge;
			let v = v.rb_mut().col_mut(i_bulge);
			let ref v0 = v[0].copy();
			let ref v1 = v[1];
			let ref v2 = v[2];
			let i1 = 0;
			let i2 = Ord::min(u2.nrows(), (i_pos_last - ilo) + (i_pos_last - ilo) + 3);
			for j in i1..i2 {
				let ref sum = &u2[(j, i_pos - ilo)] + v1 * &u2[(j, i_pos - ilo + 1)] + v2 * &u2[(j, i_pos - ilo + 2)];
				u2[(j, i_pos - ilo)] -= sum.mul_real(v0);
				u2[(j, i_pos - ilo + 1)] -= sum.mul_real(v0) * v1;
				u2[(j, i_pos - ilo + 2)] -= sum.mul_real(v0) * v2;
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
			matmul(
				wh_slice.rb_mut(),
				Accum::Replace,
				u2.rb().adjoint(),
				a_slice.rb(),
				one::<T>(),
				parallelism,
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
			let mut wv_slice = wv.rb_mut().submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
			matmul(wv_slice.rb_mut(), Accum::Replace, a_slice.rb(), u2.rb(), one::<T>(), parallelism);
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
			matmul(wv_slice.rb_mut(), Accum::Replace, z_slice.rb(), u2.rb(), one::<T>(), parallelism);
			z_slice.copy_from(wv_slice.rb());
			i += iblock;
		}
	}
	*i_pos_block = ilo + n_block - n_shifts;
}
#[inline(never)]
fn move_bulges_down<T: RealField>(
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
	s_re: ColRef<'_, T>,
	s_im: ColRef<'_, T>,
	i_pos_block: &mut usize,
	parallelism: Par,
) {
	let n = a.nrows();
	let ref eps = eps::<T>();
	let ref small_num = min_positive::<T>() / eps * from_f64::<T>(n as f64);
	while *i_pos_block + n_block_desired < ihi {
		let n_pos = Ord::min(n_block_desired - n_shifts, ihi - n_shifts - 1 - *i_pos_block);
		let n_block = n_shifts + n_pos;
		let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
		u2.fill(zero());
		u2.rb_mut().diagonal_mut().fill(one::<T>());
		let mut istart_m = *i_pos_block;
		let mut istop_m = *i_pos_block + n_block;
		for i_pos_last in *i_pos_block + n_shifts - 2..*i_pos_block + n_shifts - 2 + n_pos {
			for i_bulge in 0..n_bulges {
				let i_pos = i_pos_last - 2 * i_bulge;
				let mut v = v.rb_mut().col_mut(i_bulge);
				let mut h = a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
				let s1_re = s_re[s_re.nrows() - 1 - 2 * i_bulge].copy();
				let s1_im = s_im[s_im.nrows() - 1 - 2 * i_bulge].copy();
				let s2_re = s_re[s_re.nrows() - 1 - 2 * i_bulge - 1].copy();
				let s2_im = s_im[s_im.nrows() - 1 - 2 * i_bulge - 1].copy();
				move_bulge(h.rb_mut(), v.rb_mut(), (s1_re, s1_im), (s2_re, s2_im));
				let ref v0 = v[0].copy();
				let ref v1 = v[1];
				let ref v2 = v[2];
				for j in istart_m..i_pos + 3 {
					let ref sum = &a[(j, i_pos)] + v1 * &a[(j, i_pos + 1)] + v2 * &a[(j, i_pos + 2)];
					a[(j, i_pos)] -= sum.mul_real(v0);
					a[(j, i_pos + 1)] -= sum.mul_real(v0) * v1;
					a[(j, i_pos + 2)] -= sum.mul_real(v0) * v2;
				}
				let ref sum = &a[(i_pos, i_pos)] + v1 * &a[(i_pos + 1, i_pos)] + v2 * &a[(i_pos + 2, i_pos)];
				a[(i_pos, i_pos)] -= sum.mul_real(v0);
				a[(i_pos + 1, i_pos)] -= sum.mul_real(v0) * v1;
				a[(i_pos + 2, i_pos)] -= sum.mul_real(v0) * v2;
				if i_pos > ilo && a[(i_pos, i_pos - 1)] != zero() {
					let mut tst1 = a[(i_pos - 1, i_pos - 1)].abs1() + a[(i_pos, i_pos)].abs1();
					if tst1 == zero() {
						if i_pos > ilo + 1 {
							tst1 += a[(i_pos - 1, i_pos - 2)].abs1();
						}
						if i_pos > ilo + 2 {
							tst1 += a[(i_pos - 1, i_pos - 3)].abs1();
						}
						if i_pos > ilo + 3 {
							tst1 += a[(i_pos - 1, i_pos - 4)].abs1();
						}
						if i_pos < ihi - 1 {
							tst1 += a[(i_pos + 1, i_pos)].abs1();
						}
						if i_pos < ihi - 2 {
							tst1 += a[(i_pos + 2, i_pos)].abs1();
						}
						if i_pos < ihi - 3 {
							tst1 += a[(i_pos + 3, i_pos)].abs1();
						}
					}
					if a[(i_pos, i_pos - 1)].abs1() < (eps * tst1).fmax(small_num) {
						let ref ab = a[(i_pos, i_pos - 1)].abs1().fmax(a[(i_pos - 1, i_pos)].abs1());
						let ref ba = a[(i_pos, i_pos - 1)].abs1().fmin(a[(i_pos - 1, i_pos)].abs1());
						let ref aa = a[(i_pos, i_pos)].abs1().fmax((&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)]).abs1());
						let ref bb = a[(i_pos, i_pos)].abs1().fmin((&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)]).abs1());
						let ref s = aa + ab;
						if (ba * (ab / s)) <= (eps * (bb * (aa / s))).fmax(small_num) {
							a[(i_pos, i_pos - 1)] = zero();
						}
					}
				}
			}
			for i_bulge in 0..n_bulges {
				let i_pos = i_pos_last - 2 * i_bulge;
				let v = v.rb_mut().col_mut(i_bulge);
				let ref v0 = v[0].copy();
				let ref v1 = v[1];
				let ref v2 = v[2];
				for j in i_pos + 1..istop_m {
					let ref sum = &a[(i_pos, j)] + v1 * &a[(i_pos + 1, j)] + v2 * &a[(i_pos + 2, j)];
					a[(i_pos, j)] -= sum.mul_real(v0);
					a[(i_pos + 1, j)] -= sum.mul_real(v0) * v1;
					a[(i_pos + 2, j)] -= sum.mul_real(v0) * v2;
				}
			}
			for i_bulge in 0..n_bulges {
				let i_pos = i_pos_last - 2 * i_bulge;
				let v = v.rb_mut().col_mut(i_bulge);
				let ref v0 = v[0].copy();
				let ref v1 = v[1];
				let ref v2 = v[2];
				let i1 = (i_pos - *i_pos_block) - (i_pos_last + 2 - *i_pos_block - n_shifts);
				let i2 = Ord::min(u2.nrows(), (i_pos_last - *i_pos_block) + (i_pos_last + 2 - *i_pos_block - n_shifts) + 3);
				for j in i1..i2 {
					let ref sum = &u2[(j, i_pos - *i_pos_block)] + v1 * &u2[(j, i_pos - *i_pos_block + 1)] + v2 * &u2[(j, i_pos - *i_pos_block + 2)];
					u2[(j, i_pos - *i_pos_block)] -= sum.mul_real(v0);
					u2[(j, i_pos - *i_pos_block + 1)] -= sum.mul_real(v0) * v1;
					u2[(j, i_pos - *i_pos_block + 2)] -= sum.mul_real(v0) * v2;
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
				matmul(
					wh_slice.rb_mut(),
					Accum::Replace,
					u2.rb().adjoint(),
					a_slice.rb(),
					one::<T>(),
					parallelism,
				);
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
				matmul(wv_slice.rb_mut(), Accum::Replace, a_slice.rb(), u2.rb(), one::<T>(), parallelism);
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
				matmul(wv_slice.rb_mut(), Accum::Replace, z_slice.rb(), u2.rb(), one::<T>(), parallelism);
				z_slice.copy_from(wv_slice.rb());
				i += iblock;
			}
		}
		*i_pos_block += n_pos;
	}
}
#[inline(never)]
fn remove_bulges<T: RealField>(
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
	s_re: ColRef<'_, T>,
	s_im: ColRef<'_, T>,
	i_pos_block: &mut usize,
	parallelism: Par,
) {
	let n = a.nrows();
	let ref eps = eps::<T>();
	let ref small_num = min_positive::<T>() / eps * from_f64::<T>(n as f64);
	{
		let n_block = ihi - *i_pos_block;
		let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
		u2.fill(zero());
		u2.rb_mut().diagonal_mut().fill(one::<T>());
		let mut istart_m = *i_pos_block;
		let mut istop_m = ihi;
		for i_pos_last in *i_pos_block + n_shifts - 2..ihi + n_shifts - 1 {
			let mut i_bulge_start = if i_pos_last + 3 > ihi { (i_pos_last + 3 - ihi) / 2 } else { 0 };
			for i_bulge in i_bulge_start..n_bulges {
				let i_pos = i_pos_last - 2 * i_bulge;
				if i_pos == ihi - 2 {
					let mut v = v.rb_mut().subrows_mut(0, 2).col_mut(i_bulge);
					let mut h = a.rb_mut().subrows_mut(i_pos, 2).col_mut(i_pos - 1);
					let mut beta = h[0].copy();
					let tail = h.rb_mut().subrows_mut(1, 1);
					let HouseholderInfo { tau, .. } = make_householder_in_place(&mut beta, tail);
					v[0] = tau.recip();
					v[1] = h[1].copy();
					h[0] = beta;
					h[1] = zero();
					let ref t0 = v[0];
					let ref v1 = v[1];
					let ref t1 = t0 * v1;
					for j in istart_m..i_pos + 2 {
						let ref sum = &a[(j, i_pos)] + v1 * &a[(j, i_pos + 1)];
						a[(j, i_pos)] -= sum * t0;
						a[(j, i_pos + 1)] -= sum * t1;
					}
					for j in i_pos..istop_m {
						let ref sum = &a[(i_pos, j)] + v1 * &a[(i_pos + 1, j)];
						a[(i_pos, j)] -= sum * t0;
						a[(i_pos + 1, j)] -= sum * t1;
					}
					for j in 0..u2.nrows() {
						let ref sum = &u2[(j, i_pos - *i_pos_block)] + v1 * &u2[(j, i_pos - *i_pos_block + 1)];
						u2[(j, i_pos - *i_pos_block)] -= sum * t0;
						u2[(j, i_pos - *i_pos_block + 1)] -= sum * t1;
					}
				} else {
					let mut v = v.rb_mut().col_mut(i_bulge);
					let mut h = a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
					let s1_re = s_re[s_re.nrows() - 1 - 2 * i_bulge].copy();
					let s1_im = s_im[s_im.nrows() - 1 - 2 * i_bulge].copy();
					let s2_re = s_re[s_re.nrows() - 1 - 2 * i_bulge - 1].copy();
					let s2_im = s_im[s_im.nrows() - 1 - 2 * i_bulge - 1].copy();
					move_bulge(h.rb_mut(), v.rb_mut(), (s1_re, s1_im), (s2_re, s2_im));
					{
						let ref t0 = v[0];
						let ref v1 = v[1];
						let ref t1 = t0 * v1;
						let ref v2 = v[2];
						let ref t2 = t0 * v2;
						for j in istart_m..i_pos + 3 {
							let ref sum = &a[(j, i_pos)] + v1 * &a[(j, i_pos + 1)] + v2 * &a[(j, i_pos + 2)];
							a[(j, i_pos)] -= sum * t0;
							a[(j, i_pos + 1)] -= sum * t1;
							a[(j, i_pos + 2)] -= sum * t2;
						}
					}
					let ref v0 = v[0].copy();
					let ref v1 = v[1];
					let ref v2 = v[2];
					let ref sum = &a[(i_pos, i_pos)] + v1 * &a[(i_pos + 1, i_pos)] + v2 * &a[(i_pos + 2, i_pos)];
					a[(i_pos, i_pos)] -= sum.mul_real(v0);
					a[(i_pos + 1, i_pos)] -= sum.mul_real(v0) * v1;
					a[(i_pos + 2, i_pos)] -= sum.mul_real(v0) * v2;
					if i_pos > ilo && a[(i_pos, i_pos - 1)] != zero() {
						let mut tst1 = a[(i_pos - 1, i_pos - 1)].abs1() + a[(i_pos, i_pos)].abs1();
						if tst1 == zero() {
							if i_pos > ilo + 1 {
								tst1 += a[(i_pos - 1, i_pos - 2)].abs1();
							}
							if i_pos > ilo + 2 {
								tst1 += a[(i_pos - 1, i_pos - 3)].abs1();
							}
							if i_pos > ilo + 3 {
								tst1 += a[(i_pos - 1, i_pos - 4)].abs1();
							}
							if i_pos < ihi - 1 {
								tst1 += a[(i_pos + 1, i_pos)].abs1();
							}
							if i_pos < ihi - 2 {
								tst1 += a[(i_pos + 2, i_pos)].abs1();
							}
							if i_pos < ihi - 3 {
								tst1 += a[(i_pos + 3, i_pos)].abs1();
							}
						}
						if a[(i_pos, i_pos - 1)].abs1() < (eps * tst1).fmax(small_num) {
							let ref ab = a[(i_pos, i_pos - 1)].abs1().fmax(a[(i_pos - 1, i_pos)].abs1());
							let ref ba = a[(i_pos, i_pos - 1)].abs1().fmin(a[(i_pos - 1, i_pos)].abs1());
							let ref aa = a[(i_pos, i_pos)].abs1().fmax((&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)]).abs1());
							let ref bb = a[(i_pos, i_pos)].abs1().fmin((&a[(i_pos, i_pos)] - &a[(i_pos - 1, i_pos - 1)]).abs1());
							let ref s = aa + ab;
							if (ba * (ab / s)) <= (eps * (bb * (aa / s))).fmax(small_num) {
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
				let ref v0 = v[0].copy();
				let ref v1 = v[1];
				let ref v2 = v[2];
				for j in i_pos + 1..istop_m {
					let ref sum = &a[(i_pos, j)] + v1 * &a[(i_pos + 1, j)] + v2 * &a[(i_pos + 2, j)];
					a[(i_pos, j)] -= sum.mul_real(v0);
					a[(i_pos + 1, j)] -= sum.mul_real(v0) * v1;
					a[(i_pos + 2, j)] -= sum.mul_real(v0) * v2;
				}
			}
			for i_bulge in i_bulge_start..n_bulges {
				let i_pos = i_pos_last - 2 * i_bulge;
				let v = v.rb_mut().col_mut(i_bulge);
				let ref v0 = v[0].copy();
				let ref v1 = v[1];
				let ref v2 = v[2];
				let i1 = (i_pos - *i_pos_block) - (i_pos_last + 2 - *i_pos_block - n_shifts);
				let i2 = Ord::min(u2.nrows(), (i_pos_last - *i_pos_block) + (i_pos_last + 2 - *i_pos_block - n_shifts) + 3);
				for j in i1..i2 {
					let ref sum = &u2[(j, i_pos - *i_pos_block)] + v1 * &u2[(j, i_pos - *i_pos_block + 1)] + v2 * &u2[(j, i_pos - *i_pos_block + 2)];
					u2[(j, i_pos - *i_pos_block)] -= sum.mul_real(v0);
					u2[(j, i_pos - *i_pos_block + 1)] -= sum.mul_real(v0) * v1;
					u2[(j, i_pos - *i_pos_block + 2)] -= sum.mul_real(v0) * v2;
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
				matmul(
					wh_slice.rb_mut(),
					Accum::Replace,
					u2.rb().adjoint(),
					a_slice.rb(),
					one::<T>(),
					parallelism,
				);
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
				matmul(wv_slice.rb_mut(), Accum::Replace, a_slice.rb(), u2.rb(), one::<T>(), parallelism);
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
				matmul(wv_slice.rb_mut(), Accum::Replace, z_slice.rb(), u2.rb(), one::<T>(), parallelism);
				z_slice.copy_from(wv_slice.rb());
				i += iblock;
			}
		}
	}
}
pub fn multishift_qr<T: RealField>(
	want_t: bool,
	a: MatMut<'_, T>,
	z: Option<MatMut<'_, T>>,
	w_re: ColMut<'_, T>,
	w_im: ColMut<'_, T>,
	ilo: usize,
	ihi: usize,
	parallelism: Par,
	stack: &mut MemStack,
	params: SchurParams,
) -> (isize, usize, usize) {
	assert!(a.nrows() == a.ncols());
	assert!(ilo <= ihi);
	let n = a.nrows();
	let nh = ihi - ilo;
	assert!(w_re.nrows() == n);
	assert!(w_im.nrows() == n);
	if let Some(z) = z.rb() {
		assert!(z.nrows() == n);
		assert!(z.ncols() == n);
	}
	let mut a = a;
	let mut z = z;
	let mut w_re = w_re;
	let mut w_im = w_im;
	let mut stack = stack;
	let non_convergence_limit_window = 5;
	let non_convergence_limit_shift = 6;
	let ref dat1 = from_f64::<T>(0.75);
	let ref dat2 = from_f64::<T>(-0.4375);
	let nmin = Ord::max(15, params.blocking_threshold);
	let nibble = params.nibble_threshold;
	let nsr = (params.recommended_shift_count)(n, nh);
	let nsr = Ord::min(Ord::min(nsr, n.saturating_sub(3) / 6), ihi - ilo - 1);
	let nsr = Ord::max(nsr / 2 * 2, 2);
	let nwr = (params.recommended_deflation_window)(n, nh);
	let nwr = Ord::max(nwr, 2);
	let nwr = Ord::min(Ord::min(nwr, n.saturating_sub(1) / 3), ihi - ilo);
	if n < nmin {
		let err = lahqr(want_t, a, z, w_re, w_im, ilo, ihi);
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
				w_re[ilo] = a[(ilo, ilo)].copy();
				w_im[ilo] = zero();
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
			if kwtop > istart + 2 && a[(kwtop, kwtop - 1)].abs1() > a[(kwtop - 1, kwtop - 2)].abs1() {
				nw += 1;
			}
		}
		let (ls, ld) = aggressive_early_deflation(
			want_t,
			a.rb_mut(),
			z.rb_mut(),
			w_re.rb_mut(),
			w_im.rb_mut(),
			istart,
			istop,
			nw,
			parallelism,
			stack.rb_mut(),
			params,
		);
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
		let mut i_shifts = istop - ls;
		if k_defl % non_convergence_limit_shift == 0 {
			for i in (i_shifts + 1..istop).rev().step_by(2) {
				if i >= ilo + 2 {
					let ref ss = a[(i, i - 1)].abs1() + a[(i - 1, i - 2)].abs1();
					let aa = dat1 * ss + &a[(i, i)];
					let bb = ss.copy();
					let cc = dat2 * ss;
					let dd = aa.copy();
					let (s1, s2) = lahqr_eig22(aa, bb, cc, dd);
					w_re[i - 1] = s1.0;
					w_im[i - 1] = s1.1;
					w_re[i] = s2.0;
					w_im[i] = s2.1;
				} else {
					w_re[i] = a[(i, i)].copy();
					w_re[i - 1] = a[(i, i)].copy();
					w_im[i] = zero();
					w_im[i - 1] = zero();
				}
			}
		} else {
			if ls <= ns / 2 {
				let mut temp = a.rb_mut().submatrix_mut(n - ns, 0, ns, ns);
				let mut shifts_re = w_re.rb_mut().subrows_mut(istop - ns, ns);
				let mut shifts_im = w_im.rb_mut().subrows_mut(istop - ns, ns);
				let ierr = lahqr(false, temp.rb_mut(), None, shifts_re.rb_mut(), shifts_im.rb_mut(), 0, ns) as usize;
				ns = ns - ierr;
				if ns < 2 {
					let aa = a[(istop - 2, istop - 2)].copy();
					let bb = a[(istop - 2, istop - 1)].copy();
					let cc = a[(istop - 1, istop - 2)].copy();
					let dd = a[(istop - 1, istop - 1)].copy();
					let (s1, s2) = lahqr_eig22(aa, bb, cc, dd);
					w_re[istop - 2] = s1.0;
					w_im[istop - 2] = s1.1;
					w_re[istop - 1] = s2.0;
					w_im[istop - 1] = s2.1;
					ns = 2;
				}
				i_shifts = istop - ns;
			}
			let mut sorted = false;
			let mut k = istop;
			while !sorted && k > i_shifts {
				sorted = true;
				for i in i_shifts..k - 1 {
					if (w_re[i].abs() + w_im[i].abs()) < (w_re[i + 1].abs() + w_im[i + 1].abs()) {
						sorted = false;
						let wi = (w_re[i].copy(), w_im[i].copy());
						let wip1 = (w_re[i + 1].copy(), w_im[i + 1].copy());
						w_re[i] = wip1.0;
						w_im[i] = wip1.1;
						w_re[i + 1] = wi.0;
						w_im[i + 1] = wi.1;
					}
				}
				k -= 1;
			}
			for i in (i_shifts + 2..istop).rev().step_by(2) {
				if w_im[i] != -&w_im[i - 1] {
					let tmp = (w_re[i].copy(), w_im[i].copy());
					w_re[i] = w_re[i - 1].copy();
					w_im[i] = w_im[i - 1].copy();
					w_re[i - 1] = w_re[i - 2].copy();
					w_im[i - 1] = w_im[i - 2].copy();
					w_re[i - 2] = tmp.0;
					w_im[i - 2] = tmp.1;
				}
			}
			if ns > 1 && ns % 2 == 1 {
				ns -= 1;
			}
			i_shifts = istop - ns;
		}
		if ns == 2 && (w_im[i_shifts] == zero()) {
			if (&w_re[i_shifts] - &a[(istop - 1, istop - 1)]).abs() < (&w_re[i_shifts + 1] - &a[(istop - 1, istop - 1)]).abs() {
				w_re[i_shifts + 1] = w_re[i_shifts].copy();
				w_im[i_shifts + 1] = w_im[i_shifts].copy();
			} else {
				w_re[i_shifts] = w_re[i_shifts + 1].copy();
				w_im[i_shifts] = w_im[i_shifts + 1].copy();
			}
		}
		let mut shifts_re = w_re.rb_mut().subrows_mut(i_shifts, ns);
		let mut shifts_im = w_im.rb_mut().subrows_mut(i_shifts, ns);
		multishift_qr_sweep(
			want_t,
			a.rb_mut(),
			z.rb_mut(),
			shifts_re.rb_mut(),
			shifts_im.rb_mut(),
			istart,
			istop,
			parallelism,
			stack.rb_mut(),
		);
		count_sweep += 1;
	}
	(info, count_aed, count_sweep)
}
pub fn lahqr<T: RealField>(
	want_t: bool,
	a: MatMut<'_, T>,
	z: Option<MatMut<'_, T>>,
	w_re: ColMut<'_, T>,
	w_im: ColMut<'_, T>,
	ilo: usize,
	ihi: usize,
) -> isize {
	let epsilon = eps::<T>();
	let zero_threshold = min_positive::<T>();
	assert!(a.nrows() == a.ncols());
	assert!(ilo <= ihi);
	let n = a.nrows();
	let nh = ihi - ilo;
	assert!(w_re.nrows() == n);
	assert!(w_im.nrows() == n);
	assert!(w_re.ncols() == 1);
	assert!(w_im.ncols() == 1);
	if let Some(z) = z.rb() {
		assert!(z.nrows() == n);
		assert!(z.ncols() == n);
	}
	let mut a = a;
	let mut z = z;
	let mut w_re = w_re;
	let mut w_im = w_im;
	let ref one = one::<T>();
	let ref eps = epsilon;
	let ref small_num = zero_threshold / eps;
	let non_convergence_limit = 10;
	let ref dat1 = from_f64::<T>(0.75);
	let ref dat2 = from_f64::<T>(-0.4375);
	if nh == 0 {
		return 0;
	}
	if nh == 1 {
		w_re[ilo] = a[(ilo, ilo)].copy();
		w_im[ilo] = zero();
	}
	let itmax = 30 * Ord::max(10, nh);
	let mut k_defl = 0usize;
	let mut istop = ihi;
	let mut istart = ilo;
	stack_mat!(v, 3, 1, T);
	let mut v = v.rb_mut().col_mut(0);
	for iter in 0..itmax + 1 {
		if iter == itmax {
			return istop as isize;
		}
		if istart + 1 >= istop {
			if istart + 1 == istop {
				w_re[istart] = a[(istart, istart)].copy();
				w_im[istart] = zero();
			}
			break;
		}
		let istart_m;
		let istop_m;
		if !want_t {
			istart_m = istart;
			istop_m = istop;
		} else {
			istart_m = 0;
			istop_m = n;
		}
		for i in (istart + 1..istop).rev() {
			if a[(i, i - 1)].abs() < *small_num {
				a[(i, i - 1)] = zero();
				istart = i;
				break;
			}
			let mut tst = a[(i - 1, i - 1)].abs() + a[(i, i)].abs();
			if tst == zero() {
				if i >= ilo + 2 {
					tst = tst + a[(i - 1, i - 2)].abs();
				}
				if i + 1 < ihi {
					tst = tst + a[(i + 1, i)].abs();
				}
			}
			if a[(i, i - 1)].abs() <= (eps * tst) {
				let ref ab = a[(i, i - 1)].abs().fmax(a[(i - 1, i)].abs());
				let ref ba = a[(i, i - 1)].abs().fmin(a[(i - 1, i)].abs());
				let ref aa = a[(i, i)].abs().fmax((&a[(i, i)] - &a[(i - 1, i - 1)]).abs());
				let ref bb = a[(i, i)].abs().fmin((&a[(i, i)] - &a[(i - 1, i - 1)]).abs());
				let ref s = aa + ab;
				if ba * (ab / s) <= (eps * (bb * (aa / s))).fmax(small_num) {
					a[(i, i - 1)] = zero();
					istart = i;
					break;
				}
			}
		}
		if istart + 2 >= istop {
			if istart + 1 == istop {
				k_defl = 0;
				w_re[istart] = a[(istart, istart)].copy();
				w_im[istart] = zero();
				istop = istart;
				istart = ilo;
				continue;
			}
			if istart + 2 == istop {
				let ((a00, a01, a10, a11), (s1_re, s1_im), (s2_re, s2_im), (cs, sn)) = lahqr_schur22(
					a[(istart, istart)].copy(),
					a[(istart, istart + 1)].copy(),
					a[(istart + 1, istart)].copy(),
					a[(istart + 1, istart + 1)].copy(),
				);
				let rot = JacobiRotation { c: cs, s: sn };
				a[(istart, istart)] = a00;
				a[(istart, istart + 1)] = a01;
				a[(istart + 1, istart)] = a10;
				a[(istart + 1, istart + 1)] = a11;
				w_re[istart] = s1_re;
				w_im[istart] = s1_im;
				w_re[istart + 1] = s2_re;
				w_im[istart + 1] = s2_im;
				if want_t {
					if istart + 2 < istop_m {
						rot.apply_on_the_right_in_place(
							a.rb_mut()
								.transpose_mut()
								.get_mut(istart + 2..istop_m, ..)
								.two_cols_mut(istart, istart + 1),
						);
					}
					rot.apply_on_the_right_in_place(a.rb_mut().get_mut(istart_m..istart, ..).two_cols_mut(istart, istart + 1));
				}
				if let Some(z) = z.rb_mut() {
					rot.apply_on_the_right_in_place(z.two_cols_mut(istart, istart + 1));
				}
				k_defl = 0;
				istop = istart;
				istart = ilo;
				continue;
			}
		}
		let (a00, a01, a10, a11);
		k_defl += 1;
		if k_defl % non_convergence_limit == 0 {
			let mut s = a[(istop - 1, istop - 2)].abs();
			if istop > ilo + 2 {
				s = s + a[(istop - 2, istop - 3)].abs();
			}
			a00 = dat1 * &s + &a[(istop - 1, istop - 1)];
			a01 = dat2 * &s;
			a10 = s;
			a11 = a00.copy();
		} else {
			a00 = a[(istop - 2, istop - 2)].copy();
			a10 = a[(istop - 1, istop - 2)].copy();
			a01 = a[(istop - 2, istop - 1)].copy();
			a11 = a[(istop - 1, istop - 1)].copy();
		}
		let (mut s1, mut s2) = lahqr_eig22(a00, a01, a10, a11);
		if s1.1 == zero() && s2.1 == zero() {
			if (&s1.0 - &a[(istop - 1, istop - 1)]).abs() <= (&s2.0 - &a[(istop - 1, istop - 1)]).abs() {
				s2 = ((s1.0).copy(), (s1.1).copy());
			} else {
				s1 = ((s2.0).copy(), (s2.1).copy());
			}
		}
		let mut istart2 = istart;
		if istart + 3 < istop {
			for i in (istart + 1..istop - 2).rev() {
				let h = a.rb().submatrix(i, i, 3, 3);
				lahqr_shiftcolumn(h, v.rb_mut(), ((s1.0).copy(), (s1.1).copy()), ((s2.0).copy(), (s2.1).copy()));
				let mut head = v[0].copy();
				let HouseholderInfo { tau, .. } = make_householder_in_place(&mut head, v.rb_mut().subrows_mut(1, 2));
				let ref tau = tau.recip();
				let ref v0 = *tau;
				let ref v1 = v[1];
				let ref v2 = v[2];
				let ref refsum = v0 * &a[(i, i - 1)] + v1 * &a[(i + 1, i - 1)];
				if (&a[(i + 1, i - 1)] - refsum * v1).abs() + (refsum * v2).abs()
					<= eps * ((a[(i, i - 1)].abs() + a[(i, i + 1)].abs()) + a[(i + 1, i + 2)].abs())
				{
					istart2 = i;
					break;
				}
			}
		}
		for i in istart2..istop - 1 {
			let nr = Ord::min(3, istop - i);
			let mut t1;
			if i == istart2 {
				let h = a.rb().submatrix(i, i, nr, nr);
				let mut x = v.rb_mut().subrows_mut(0, nr);
				lahqr_shiftcolumn(h, x.rb_mut(), ((s1.0).copy(), (s1.1).copy()), ((s2.0).copy(), (s2.1).copy()));
				let mut beta = x[0].copy();
				let tail = x.rb_mut().subrows_mut(1, nr - 1);
				HouseholderInfo { tau: t1, .. } = make_householder_in_place(&mut beta, tail);
				v[0] = beta;
				t1 = t1.recip();
				if i > istart {
					a[(i, i - 1)] *= one - &t1;
				}
			} else {
				v[0] = a[(i, i - 1)].copy();
				v[1] = a[(i + 1, i - 1)].copy();
				if nr == 3 {
					v[2] = a[(i + 2, i - 1)].copy();
				}
				let mut beta = v[0].copy();
				let tail = v.rb_mut().subrows_mut(1, nr - 1);
				HouseholderInfo { tau: t1, .. } = make_householder_in_place(&mut beta, tail);
				t1 = t1.recip();
				v[0] = beta.copy();
				a[(i, i - 1)] = beta.copy();
				a[(i + 1, i - 1)] = zero();
				if nr == 3 {
					a[(i + 2, i - 1)] = zero();
				}
			}
			let ref t1 = t1;
			let ref v2 = v[1];
			let ref t2 = t1 * v2;
			if nr == 3 {
				let ref v3 = v[2];
				let ref t3 = t1 * v3;
				for j in i..istop_m {
					let ref sum = &a[(i, j)] + v2 * &a[(i + 1, j)] + v3 * &a[(i + 2, j)];
					a[(i, j)] -= sum * t1;
					a[(i + 1, j)] -= sum * t2;
					a[(i + 2, j)] -= sum * t3;
				}
				for j in istart_m..Ord::min(i + 4, istop) {
					let ref sum = &a[(j, i)] + v2 * &a[(j, i + 1)] + v3 * &a[(j, i + 2)];
					a[(j, i)] -= sum * t1;
					a[(j, i + 1)] -= sum * t2;
					a[(j, i + 2)] -= sum * t3;
				}
				if let Some(mut z) = z.rb_mut() {
					for j in 0..n {
						let ref sum = &z[(j, i)] + v2 * &z[(j, i + 1)] + v3 * &z[(j, i + 2)];
						z[(j, i)] -= sum * t1;
						z[(j, i + 1)] -= sum * t2;
						z[(j, i + 2)] -= sum * t3;
					}
				}
			} else {
				for j in i..istop_m {
					let ref sum = &a[(i, j)] + v2 * &a[(i + 1, j)];
					a[(i, j)] -= sum * t1;
					a[(i + 1, j)] -= sum * t2;
				}
				for j in istart_m..Ord::min(i + 3, istop) {
					let ref sum = &a[(j, i)] + v2 * &a[(j, i + 1)];
					a[(j, i)] -= sum * t1;
					a[(j, i + 1)] -= sum * t2;
				}
				if let Some(mut z) = z.rb_mut() {
					for j in 0..n {
						let ref sum = &z[(j, i)] + v2 * &z[(j, i + 1)];
						z[(j, i)] -= sum * t1;
						z[(j, i + 1)] -= sum * t2;
					}
				}
			}
		}
	}
	0
}
#[cfg(test)]
mod tests {
	use super::{lahqr, multishift_qr};
	use crate::assert;
	use crate::linalg::evd::schur::multishift_qr_scratch;
	use crate::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::{MemBuffer, MemStack};
	#[test]
	fn test_5() {
		let h = [
			[-0.417, -0.056, -2.136, 1.64, -1.793],
			[-0.842, 0.503, -1.245, -1.058, -0.909],
			[0., 2.292, 0.042, -1.118, 0.539],
			[0., 0., 1.175, -0.748, 0.009],
			[0., 0., 0., -0.989, -0.339],
		];
		let h = MatRef::from_row_major_array(&h);
		let mut q = [
			[1.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 1.0],
		];
		let mut q = MatMut::from_row_major_array_mut(&mut q);
		let mut w_re = Col::zeros(5);
		let mut w_im = Col::zeros(5);
		let mut t = h.cloned();
		lahqr(true, t.as_mut(), Some(q.as_mut()), w_re.as_mut(), w_im.as_mut(), 0, 5);
		let h_reconstructed = &q * &t * q.transpose();
		let approx_eq = CwiseMat(ApproxEq::<f64>::eps());
		assert!(h ~ h_reconstructed);
	}
	#[test]
	fn test_5_2() {
		let h = [
			[0.10, 0.97, 0.19, 0.21, 0.84],
			[0.19, 0.21, 0.05, 0.83, 0.15],
			[0.00, 0.13, 0.05, 0.20, 0.14],
			[0.00, 0.00, 0.45, 0.44, 0.67],
			[0.00, 0.00, 0.00, 0.78, 0.27],
		];
		let h = MatRef::from_row_major_array(&h);
		let mut q = [
			[1.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 1.0],
		];
		let mut q = MatMut::from_row_major_array_mut(&mut q);
		let mut w_re = Col::zeros(5);
		let mut w_im = Col::zeros(5);
		let mut t = h.cloned();
		lahqr(true, t.as_mut(), Some(q.as_mut()), w_re.as_mut(), w_im.as_mut(), 0, 5);
		let h_reconstructed = &q * &t * q.transpose();
		let approx_eq = CwiseMat(ApproxEq::<f64>::eps());
		assert!(h ~ h_reconstructed);
	}
	#[test]
	fn test_n() {
		use rand::prelude::*;
		let rng = &mut StdRng::seed_from_u64(0);
		for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32, 128, 256] {
			let mut h = Mat::zeros(n, n);
			for j in 0..n {
				for i in 0..n {
					if i <= j + 1 {
						h[(i, j)] = rng.random::<f64>();
					}
				}
			}
			if n <= 128 {
				let mut q = Mat::zeros(n, n);
				for i in 0..n {
					q[(i, i)] = 1.0;
				}
				let mut w_re = Col::zeros(n);
				let mut w_im = Col::zeros(n);
				let mut t = h.as_ref().cloned();
				lahqr(true, t.as_mut(), Some(q.as_mut()), w_re.as_mut(), w_im.as_mut(), 0, n);
				let h_reconstructed = &q * &t * q.transpose();
				let approx_eq = CwiseMat(ApproxEq::<f64>::eps());
				assert!(h ~ h_reconstructed);
			}
			{
				let mut q = Mat::zeros(n, n);
				for i in 0..n {
					q[(i, i)] = 1.0;
				}
				let mut w_re = Col::zeros(n);
				let mut w_im = Col::zeros(n);
				let mut t = h.as_ref().cloned();
				multishift_qr(
					true,
					t.as_mut(),
					Some(q.as_mut()),
					w_re.as_mut(),
					w_im.as_mut(),
					0,
					n,
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(multishift_qr_scratch::<f64>(n, n, true, true, Par::Seq, auto!(f64)))),
					auto!(f64),
				);
				for j in 0..n {
					for i in 0..n {
						if i > j + 1 {
							t[(i, j)] = 0.0;
						}
					}
				}
				let h_reconstructed = &q * &t * q.transpose();
				let approx_eq = CwiseMat(ApproxEq::<f64>::eps());
				assert!(h ~ h_reconstructed);
			}
		}
	}
}
