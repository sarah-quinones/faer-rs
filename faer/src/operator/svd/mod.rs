use crate::internal_prelude::*;
use crate::matrix_free::BiLinOp;
use linalg::matmul::matmul;
use linalg::temp_mat_zeroed;
fn iterate_lanczos<T: ComplexField>(
	A: &dyn BiLinOp<T>,
	H: MatMut<'_, T>,
	Q: MatMut<'_, T>,
	P: MatMut<'_, T>,
	start: usize,
	end: usize,
	par: Par,
	stack: &mut MemStack,
) {
	let mut P = P;
	let mut Q = Q;
	let mut H = H;
	for j in start..end + 1 {
		let (P, Pnext) = P.rb_mut().split_at_col_mut(j - 1);
		let (Q, Qnext) = Q.rb_mut().split_at_col_mut(j);
		let P = P.rb();
		let Q = Q.rb();
		let mut Pnext = Pnext.col_mut(0);
		let mut Qnext = Qnext.col_mut(0);
		let ref f = from_f64::<T::Real>(Ord::max(j, 8) as f64) * eps::<T::Real>();
		{
			A.apply(Pnext.rb_mut().as_mat_mut(), Q.col(j - 1).as_mat(), par, stack);
			for i in 0..j - 1 {
				let ref r = P.col(i).adjoint() * Pnext.rb();
				zip!(Pnext.rb_mut(), P.col(i)).for_each(|unzip!(y, x): Zip!(&mut _, &_)| *y -= r * x);
			}
			{
				let (mut converged, _) = stack.collect(core::iter::repeat_n(false, j - 1));
				loop {
					let mut all_true = true;
					for i in 0..j - 1 {
						if !converged[i] {
							all_true = false;
							let ref r = P.col(i).adjoint() * Pnext.rb();
							zip!(Pnext.rb_mut(), P.col(i)).for_each(|unzip!(y, x): Zip!(&mut _, &T)| *y -= r * x);
							converged[i] = r.abs() < f * Pnext.norm_l2();
						}
					}
					if all_true {
						break;
					}
				}
			}
			let norm = Pnext.norm_l2();
			if norm > zero() {
				let ref norm_inv = norm.recip();
				zip!(&mut Pnext).for_each(|unzip!(v)| *v = v.mul_real(norm_inv));
			} else {
				break;
			}
			H[(j - 1, j - 1)] = norm.to_cplx();
		}
		let Pnext = Pnext.rb();
		{
			A.adjoint_apply(Qnext.rb_mut().as_mat_mut(), Pnext.as_mat(), par, stack);
			for i in 0..j {
				let ref r = Q.col(i).adjoint() * Qnext.rb();
				zip!(Qnext.rb_mut(), Q.col(i)).for_each(|unzip!(y, x): Zip!(&mut _, &T)| *y -= r * x);
			}
			{
				let (mut converged, _) = stack.collect(core::iter::repeat_n(false, j));
				loop {
					let mut all_true = true;
					for i in 0..j {
						if !converged[i] {
							all_true = false;
							let ref r = Q.col(i).adjoint() * Qnext.rb();
							zip!(Qnext.rb_mut(), Q.col(i)).for_each(|unzip!(y, x): Zip!(&mut _, &T)| *y -= r * x);
							converged[i] = r.abs() < f * Qnext.norm_l2();
						}
					}
					if all_true {
						break;
					}
				}
			}
			let norm = Qnext.norm_l2();
			if norm > zero() {
				let ref norm_inv = norm.recip();
				zip!(&mut Qnext).for_each(|unzip!(v)| *v = v.mul_real(norm_inv));
			} else {
				break;
			}
			H[(j - 1, j)] = norm.to_cplx();
		}
	}
}
pub fn partial_svd_imp<T: ComplexField>(
	left_singular_vecs: MatMut<'_, T>,
	right_singular_vecs: MatMut<'_, T>,
	singular_vals: &mut [T],
	A: &dyn BiLinOp<T>,
	v0: ColRef<'_, T>,
	min_dim: usize,
	max_dim: usize,
	n_eigval: usize,
	tol: T::Real,
	restarts: usize,
	par: Par,
	stack: &mut MemStack,
) -> usize {
	let m = A.nrows();
	let n = A.ncols();
	let (mut H, stack) = temp_mat_zeroed::<T, _, _>(max_dim, max_dim + 1, stack);
	let mut H = H.as_mat_mut();
	let (mut P, stack) = temp_mat_zeroed::<T, _, _>(m, max_dim, stack);
	let mut P = P.as_mat_mut();
	let (mut Q, stack) = temp_mat_zeroed::<T, _, _>(n, max_dim + 1, stack);
	let mut Q = Q.as_mat_mut();
	let (mut tmp, stack) = temp_mat_zeroed::<T, _, _>(Ord::max(m, n), max_dim, stack);
	let mut tmp = tmp.as_mat_mut();
	let mut active = 0usize;
	if max_dim < Ord::min(m, n) {
		let (mut X, stack) = temp_mat_zeroed::<T, _, _>(max_dim, max_dim, stack);
		let mut X = X.as_mat_mut();
		let (mut Y, stack) = temp_mat_zeroed::<T, _, _>(max_dim, max_dim, stack);
		let mut Y = Y.as_mat_mut();
		let (mut residual, stack) = temp_mat_zeroed::<T, _, _>(max_dim, 1, stack);
		let mut residual = residual.as_mat_mut().col_mut(0);
		let ref f = v0.norm_l2();
		if *f > min_positive() {
			let ref f = f.recip();
			zip!(Q.rb_mut().col_mut(0), v0).for_each(|unzip!(y, x): Zip!(&mut _, &_)| *y = x.mul_real(f));
		} else {
			let n0 = n as u32;
			let n1 = (n >> 32) as u32;
			let n = from_f64::<T>(n0 as f64) + from_f64::<T>(n1 as f64);
			let ref f = n.sqrt().recip();
			zip!(Q.rb_mut().col_mut(0)).for_each(|unzip!(y)| *y = f.copy());
		}
		iterate_lanczos(A, H.as_mut(), Q.as_mut(), P.as_mut(), 1, min_dim, par, stack);
		let mut k = min_dim;
		for _ in 0..restarts {
			iterate_lanczos(A, H.as_mut(), Q.as_mut(), P.as_mut(), k + 1, max_dim, par, stack);
			let ref Hmm = H[(max_dim - 1, max_dim)].copy();
			let dim = max_dim - active;
			X.fill(zero());
			X.rb_mut().diagonal_mut().fill(one());
			Y.fill(zero());
			Y.rb_mut().diagonal_mut().fill(one());
			let mut X_slice = X.rb_mut().get_mut(active..max_dim, active..max_dim);
			let mut Y_slice = Y.rb_mut().get_mut(active..max_dim, active..max_dim);
			let mut H_slice = H.rb_mut().get_mut(active..max_dim, active..max_dim);
			{
				let (mut s, stack) = temp_mat_zeroed(dim, 1, stack);
				let mut s = s.as_mat_mut().col_mut(0).as_diagonal_mut();
				if linalg::svd::svd(
					H_slice.rb(),
					s.rb_mut(),
					Some(X_slice.rb_mut()),
					Some(Y_slice.rb_mut()),
					par,
					stack,
					default(),
				)
				.is_err()
				{
					break;
				}
				H_slice.fill(zero());
				H_slice.rb_mut().diagonal_mut().copy_from(s.rb());
			}
			for j in 0..dim {
				residual[active + j] = Hmm * &X[(max_dim - 1, active + j)];
			}
			#[derive(Copy, Clone, PartialEq, Eq, Debug)]
			enum Group {
				Lock,
				Retain,
				Purge,
			}
			let mut groups = alloc::vec![Group::Purge; max_dim];
			let nev = n_eigval;
			let mut nlock = 0usize;
			for j in 0..nev {
				if residual[j].abs() <= tol {
					groups[j] = Group::Lock;
					nlock += 1;
				} else {
					groups[j] = Group::Retain;
				}
			}
			let ideal_size = Ord::min(nlock + min_dim, (min_dim + max_dim) / 2);
			k = nev;
			for i in nev..max_dim {
				let group;
				if k < ideal_size && residual[i].abs() > tol {
					group = Group::Retain;
					k += 1;
				} else {
					group = Group::Purge;
				}
				groups[i] = group;
			}
			let mut purge = 0usize;
			while purge < active && groups[purge] == Group::Lock {
				purge += 1;
			}
			let mut lo = 0usize;
			let mut mi = 0usize;
			let mut hi = 0usize;
			while hi < max_dim {
				match groups[hi] {
					Group::Lock => {
						if hi > lo {
							H[(lo, lo)] = H[(hi, hi)].copy();
							residual[lo] = residual[hi].copy();
							{
								let (mut lo, hi) = X.rb_mut().two_cols_mut(lo, hi);
								lo.copy_from(hi);
							}
							{
								let (mut lo, hi) = Y.rb_mut().two_cols_mut(lo, hi);
								lo.copy_from(hi);
							}
						}
						lo += 1;
						mi += 1;
						hi += 1;
					},
					Group::Retain => {
						if hi > mi {
							H[(mi, mi)] = H[(hi, hi)].copy();
							{
								let (mut mi, hi) = X.rb_mut().two_cols_mut(mi, hi);
								mi.copy_from(hi);
							}
							{
								let (mut mi, hi) = Y.rb_mut().two_cols_mut(mi, hi);
								mi.copy_from(hi);
							}
						}
						mi += 1;
						hi += 1;
					},
					Group::Purge => {
						hi += 1;
					},
				}
			}
			{
				let mut P_tmp = tmp.rb_mut().get_mut(..m, purge..k);
				matmul(
					P_tmp.rb_mut(),
					Accum::Replace,
					P.rb().get(.., purge..max_dim),
					X.rb().get(purge..max_dim, purge..k),
					one(),
					par,
				);
				P.rb_mut().get_mut(.., purge..k).copy_from(&P_tmp);
			}
			{
				let mut Q_tmp = tmp.rb_mut().get_mut(..n, purge..k);
				matmul(
					Q_tmp.rb_mut(),
					Accum::Replace,
					Q.rb().get(.., purge..max_dim),
					Y.rb().get(purge..max_dim, purge..k),
					one(),
					par,
				);
				Q.rb_mut().get_mut(.., purge..k).copy_from(&Q_tmp);
			}
			for i in 0..k {
				H[(i, k)] = residual[i].copy();
			}
			let (mut x, y) = Q.rb_mut().two_cols_mut(k, max_dim);
			x.copy_from(&y);
			active = nlock;
			if nlock >= n_eigval {
				break;
			}
		}
		let n = active;
		let (mut norms, stack) = stack.make_with(n, |j| H[(j, j)].abs());
		let (mut perm, stack) = stack.make_with(n, |j| j);
		let _ = stack;
		let perm = &mut *perm;
		let norms = &mut *norms;
		perm.sort_unstable_by(|&i, &j| {
			if norms[i] > norms[j] {
				core::cmp::Ordering::Less
			} else if norms[i] < norms[j] {
				core::cmp::Ordering::Greater
			} else {
				core::cmp::Ordering::Equal
			}
		});
		let mut left_singular_vecs = left_singular_vecs;
		let mut right_singular_vecs = right_singular_vecs;
		for idx in 0..active {
			let j = perm[idx];
			left_singular_vecs.rb_mut().col_mut(idx).copy_from(P.rb().col(j));
			right_singular_vecs.rb_mut().col_mut(idx).copy_from(Q.rb().col(j));
			singular_vals[idx] = H[(j, j)].copy();
		}
		active
	} else {
		panic!()
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::prelude::*;
	use crate::stats::prelude::*;
	use dyn_stack::{MemBuffer, StackReq};
	use equator::assert;
	#[test]
	fn test_small_cplx() {
		let rng = &mut StdRng::seed_from_u64(1);
		let m = 768;
		let n = 512;
		let n_eigval = 32;
		let mat = CwiseMatDistribution {
			nrows: m,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		};
		let col = CwiseColDistribution {
			nrows: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		};
		let A: Mat<c64> = mat.sample(rng);
		let mut v0: Col<c64> = col.sample(rng);
		v0 /= v0.norm_l2();
		let A = A.as_ref();
		let v0 = v0.as_ref();
		let par = Par::Seq;
		let mem = &mut MemBuffer::new(StackReq::new::<u8>(1024 * 1024 * 512));
		let mut U = Mat::zeros(m, n_eigval);
		let mut V = Mat::zeros(n, n_eigval);
		let mut s = vec![c64::ZERO; n_eigval];
		let n_converged = partial_svd_imp(
			U.rb_mut(),
			V.rb_mut(),
			&mut s,
			&A,
			v0,
			32,
			64,
			n_eigval,
			f64::EPSILON * 128.0,
			1000,
			par,
			MemStack::new(mem),
		);
		assert!(s.iter().map(|x| x.norm()).is_sorted_by(|x, y| x >= y));
		assert!(n_converged == n_eigval);
		for j in 0..n_converged {
			assert!((A.adjoint() * (A * V.col(j)) - Scale(s[j] * s[j]) * V.col(j)).norm_l2() < 1e-10);
			assert!((A * (A.adjoint() * U.col(j)) - Scale(s[j] * s[j]) * U.col(j)).norm_l2() < 1e-10);
		}
	}
}
