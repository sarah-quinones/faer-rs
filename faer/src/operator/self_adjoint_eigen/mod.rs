use super::LinOp;
use crate::internal_prelude::*;
use crate::perm;
use linalg::matmul::matmul;
use linalg::temp_mat_zeroed;

fn iterate_lanczos<T: ComplexField>(
	A: &dyn LinOp<T>,
	H: MatMut<'_, T>,
	V: MatMut<'_, T>,
	start: usize,
	end: usize,
	par: Par,
	stack: &mut MemStack,
) {
	let mut V = V;
	let mut H = H;
	for j in start..end + 1 {
		let mut H = H.rb_mut().col_mut(j - 1);
		H.fill(zero());
		let (V, Vnext) = V.rb_mut().split_at_col_mut(j);
		let V = V.rb();
		let mut Vnext = Vnext.col_mut(0);
		A.apply(
			Vnext.rb_mut().as_mat_mut(),
			V.col(j - 1).as_mat(),
			par,
			stack,
		);
		let (mut converged, _) = stack.collect(core::iter::repeat_n(false, j));
		let mut h = H.rb_mut().get_mut(..j);
		for i in 0..j {
			let r = V.col(i).adjoint() * Vnext.rb();
			zip!(Vnext.rb_mut(), V.col(i))
				.for_each(|unzip!(y, x): Zip!(&mut T, &T)| *y -= &r * x);
			if i + 1 == j {
				h[i] = r;
			}
		}
		let ref f =
			from_f64::<T::Real>(Ord::max(j, 8) as f64) * eps::<T::Real>();
		loop {
			let mut all_true = true;
			for i in 0..j {
				if !converged[i] {
					all_true = false;
					let ref r = V.col(i).adjoint() * Vnext.rb();
					zip!(Vnext.rb_mut(), V.col(i))
						.for_each(|unzip!(y, x): Zip!(&mut T, &T)| *y -= r * x);
					if i + 1 == j {
						h[i] += r;
					}
					converged[i] = r.abs() < f * Vnext.norm_l2();
				}
			}
			if all_true {
				break;
			}
		}
		let norm = Vnext.norm_l2();
		if norm > zero() {
			let ref norm_inv = norm.recip();
			zip!(&mut Vnext)
				.for_each(|unzip!(v): Zip![&mut _]| *v = v.mul_real(norm_inv));
		} else {
			break;
		}
		H[j] = norm.to_cplx();
	}
}

pub fn partial_self_adjoint_eigen_imp<T: ComplexField>(
	eigvecs: MatMut<'_, T>,
	eigvals: &mut [T],
	A: &dyn LinOp<T>,
	v0: ColRef<'_, T>,
	min_dim: usize,
	max_dim: usize,
	n_eigval: usize,
	tol: T::Real,
	restarts: usize,
	par: Par,
	stack: &mut MemStack,
) -> usize {
	let n = A.nrows();
	let (mut H, stack) =
		temp_mat_zeroed::<T, _, _>(max_dim + 1, max_dim, stack);
	let mut H = H.as_mat_mut();
	let (mut V, stack) = temp_mat_zeroed::<T, _, _>(n, max_dim + 1, stack);
	let mut V = V.as_mat_mut();
	let (mut tmp, stack) = temp_mat_zeroed::<T, _, _>(n, max_dim, stack);
	let mut tmp = tmp.as_mat_mut();
	if max_dim < n {
		let mut active = 0usize;
		let (mut Q, stack) =
			temp_mat_zeroed::<T, _, _>(max_dim, max_dim, stack);
		let mut Q = Q.as_mat_mut();
		let (mut residual, stack) =
			temp_mat_zeroed::<T::Real, _, _>(max_dim, 1, stack);
		let mut residual = residual.as_mat_mut().col_mut(0);
		let f = v0.norm_l2();
		if f > min_positive() {
			let ref f = f.recip();
			zip!(V.rb_mut().col_mut(0), v0)
				.for_each(|unzip!(y, x): Zip!(&mut T, &T)| *y = x.mul_real(f));
		} else {
			let n0 = n as u32;
			let n1 = (n >> 32) as u32;
			let n = from_f64::<T>(n0 as f64) + from_f64::<T>(n1 as f64);
			let f = n.sqrt().recip();
			zip!(V.rb_mut().col_mut(0)).for_each(|unzip!(y)| *y = f.copy());
		}
		iterate_lanczos(A, H.as_mut(), V.as_mut(), 1, min_dim, par, stack);
		let mut k = min_dim;
		for _ in 0..restarts {
			iterate_lanczos(
				A,
				H.as_mut(),
				V.as_mut(),
				k + 1,
				max_dim,
				par,
				stack,
			);
			let ref Hmm = H[(max_dim, max_dim - 1)].copy();
			let n = max_dim - active;
			Q.fill(zero());
			Q.rb_mut().diagonal_mut().fill(one());
			let mut Q_slice =
				Q.rb_mut().get_mut(active..max_dim, active..max_dim);
			let mut H_slice =
				H.rb_mut().get_mut(active..max_dim, active..max_dim);
			{
				let (mut s, stack) = temp_mat_zeroed(n, 1, stack);
				let mut s = s.as_mat_mut().col_mut(0).as_diagonal_mut();
				if linalg::evd::self_adjoint_evd(
					H_slice.rb(),
					s.rb_mut(),
					Some(Q_slice.rb_mut()),
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
			for j in 0..n {
				let mut idx = j;
				let mut max = zero::<T::Real>();
				for i in j..n {
					let v = H_slice[(i, i)].abs();
					if v > max {
						max = v;
						idx = i;
					}
				}
				let i = idx;
				if i != j {
					let tmp = H_slice[(i, i)].copy();
					H_slice[(i, i)] = H_slice[(j, j)].copy();
					H_slice[(j, j)] = tmp;
					perm::swap_cols_idx(Q_slice.rb_mut(), i, j);
				}
			}
			let ref Hmm_abs = Hmm.abs();
			for j in 0..n {
				residual[active + j] =
					Hmm_abs * Q[(max_dim - 1, active + j)].abs();
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
				if residual[j] <= tol {
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
				if k < ideal_size && residual[i] > tol {
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
							let (mut lo, hi) = Q.rb_mut().two_cols_mut(lo, hi);
							lo.copy_from(hi);
						}
						lo += 1;
						mi += 1;
						hi += 1;
					},
					Group::Retain => {
						if hi > mi {
							H[(mi, mi)] = H[(hi, hi)].copy();
							let (mut mi, hi) = Q.rb_mut().two_cols_mut(mi, hi);
							mi.copy_from(hi);
						}
						mi += 1;
						hi += 1;
					},
					Group::Purge => {
						hi += 1;
					},
				}
			}
			let mut V_tmp = tmp.rb_mut().get_mut(.., purge..k);
			matmul(
				V_tmp.rb_mut(),
				Accum::Replace,
				V.rb().get(.., purge..max_dim),
				Q.rb().get(purge..max_dim, purge..k),
				one(),
				par,
			);
			V.rb_mut().get_mut(.., purge..k).copy_from(&V_tmp);
			let mut b_tmp = tmp.rb_mut().get_mut(0, ..);
			matmul(
				b_tmp.rb_mut(),
				Accum::Replace,
				H.rb().get(max_dim, ..),
				Q.rb(),
				one(),
				par,
			);
			H.rb_mut().get_mut(max_dim, ..).copy_from(b_tmp);
			let (mut x, y) = V.rb_mut().two_cols_mut(k, max_dim);
			x.copy_from(&y);
			let (mut x, mut y) = H.rb_mut().two_rows_mut(k, max_dim);
			x.copy_from(&y);
			y.fill(zero());
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
		let mut eigvecs = eigvecs;
		for idx in 0..active {
			let j = perm[idx];
			eigvecs.rb_mut().col_mut(idx).copy_from(V.rb().col(j));
			eigvals[idx] = H[(j, j)].copy();
		}
		active
	} else {
		panic!();
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::prelude::*;
	use crate::stats::prelude::*;
	use dyn_stack::MemBuffer;
	use equator::assert;
	#[test]
	fn test_small_cplx() {
		let rng = &mut StdRng::seed_from_u64(1);
		let n = 512;
		let n_eigval = 32;
		let mat = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		};
		let col = CwiseColDistribution {
			nrows: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		};
		let A: Mat<c64> = mat.sample(rng);
		let A = &A + A.adjoint();
		let mut v0: Col<c64> = col.sample(rng);
		v0 /= v0.norm_l2();
		let A = A.as_ref();
		let v0 = v0.as_ref();
		let par = Par::Seq;
		let mem = &mut MemBuffer::new(
			crate::matrix_free::eigen::partial_eigen_scratch(
				&A,
				n_eigval,
				par,
				default(),
			),
		);
		let mut V = Mat::zeros(n, n_eigval);
		let mut w = vec![c64::ZERO; n_eigval];
		let n_converged = partial_self_adjoint_eigen_imp(
			V.rb_mut(),
			&mut w,
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
		assert!(w.iter().map(|x| x.norm()).is_sorted_by(|x, y| x >= y));
		assert!(n_converged == n_eigval);
		for j in 0..n_converged {
			assert!((A * V.col(j) - Scale(w[j]) * V.col(j)).norm_l2() < 1e-10);
		}
	}
}
