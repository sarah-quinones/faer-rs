use super::*;
use crate::assert;

/// computes the layout of required workspace for executing the bicgstab algorithm
pub fn bicgstab_scratch<T: ComplexField>(
	left_precond: impl Precond<T>,
	right_precond: impl Precond<T>,
	mat: impl LinOp<T>,
	rhs_ncols: usize,
	par: Par,
) -> StackReq {
	fn implementation<T: ComplexField>(K1: &dyn Precond<T>, K2: &dyn Precond<T>, A: &dyn LinOp<T>, rhs_ncols: usize, par: Par) -> StackReq {
		let n = A.nrows();
		let k = rhs_ncols;

		let nk = temp_mat_scratch::<T>(n, k);
		let kk = temp_mat_scratch::<T>(k, k);
		let k_usize = StackReq::new::<usize>(k);
		let lu = crate::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, T>(k, k, par, Default::default());
		StackReq::all_of(&[
			k_usize, // row_perm
			k_usize, // row_perm_inv
			k_usize, // col_perm
			k_usize, // col_perm_inv
			kk,      // rtv
			nk,      // r
			nk,      // p
			nk,      // r_tilde
			nk,      // v
			nk,      // y
			nk,      // s
			nk,      // t
			nk,      // z
			StackReq::any_of(&[
				lu,
				A.apply_scratch(k, par),
				StackReq::all_of(&[
					nk, // y0 | z0 | ks
					K1.apply_scratch(k, par),
					K2.apply_scratch(k, par),
				]),
				StackReq::all_of(&[
					kk, // rtr | rtt
					kk, // temp
				]),
				kk, // rtr | rtt
			]),
		])
	}
	implementation(&left_precond, &right_precond, &mat, rhs_ncols, par)
}

/// algorithm parameters
#[derive(Copy, Clone, Debug)]
pub struct BicgParams<T> {
	/// whether the initial guess is implicitly zero or not
	pub initial_guess: InitialGuessStatus,
	/// absolute tolerance for convergence testing
	pub abs_tolerance: T,
	/// relative tolerance for convergence testing
	pub rel_tolerance: T,
	/// maximum number of iterations
	pub max_iters: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: RealField> Default for BicgParams<T> {
	#[inline]
	#[math]
	fn default() -> Self {
		Self {
			initial_guess: InitialGuessStatus::MaybeNonZero,
			abs_tolerance: zero(),
			rel_tolerance: eps::<T>() * from_f64::<T>(128.0),
			max_iters: usize::MAX,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

/// algorithm result
#[derive(Copy, Clone, Debug)]
pub struct BicgInfo<T> {
	/// absolute residual at the final step
	pub abs_residual: T,
	/// relative residual at the final step
	pub rel_residual: T,
	/// number of iterations executed by the algorithm
	pub iter_count: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

/// algorithm error
#[derive(Copy, Clone, Debug)]
pub enum BicgError<T> {
	/// convergence failure
	NoConvergence {
		/// absolute residual at the final step
		abs_residual: T,
		/// relative residual at the final step
		rel_residual: T,
	},
}

/// executes bicgstab using the provided preconditioners
///
/// # note
/// this function is also optimized for a rhs with multiple columns
#[track_caller]
pub fn bicgstab<T: ComplexField>(
	out: MatMut<'_, T>,
	left_precond: impl Precond<T>,
	right_precond: impl Precond<T>,
	mat: impl LinOp<T>,
	rhs: MatRef<'_, T>,
	params: BicgParams<T::Real>,
	callback: impl FnMut(MatRef<'_, T>),
	par: Par,
	stack: &mut MemStack,
) -> Result<BicgInfo<T::Real>, BicgError<T::Real>> {
	#[track_caller]
	#[math]
	fn implementation<T: ComplexField>(
		out: MatMut<'_, T>,
		left_precond: &dyn Precond<T>,
		right_precond: &dyn Precond<T>,
		mat: &dyn LinOp<T>,
		rhs: MatRef<'_, T>,
		params: BicgParams<T::Real>,
		callback: &mut dyn FnMut(MatRef<'_, T>),
		par: Par,
		stack: &mut MemStack,
	) -> Result<BicgInfo<T::Real>, BicgError<T::Real>> {
		let mut x = out;
		let A = mat;
		let K1 = left_precond;
		let K2 = right_precond;
		let b = rhs;

		assert!(A.nrows() == A.ncols());
		let n = A.nrows();
		let k = x.ncols();

		let b_norm = b.norm_l2();
		if b_norm == zero::<T::Real>() {
			x.fill(zero());
			return Ok(BicgInfo {
				abs_residual: zero::<T::Real>(),
				rel_residual: zero::<T::Real>(),
				iter_count: 0,
				non_exhaustive: NonExhaustive(()),
			});
		}

		let rel_threshold = params.rel_tolerance * b_norm;
		let abs_threshold = params.abs_tolerance;
		let threshold = if abs_threshold > rel_threshold { abs_threshold } else { rel_threshold };

		let (row_perm, stack) = unsafe { stack.make_raw::<usize>(k) };
		let (row_perm_inv, stack) = unsafe { stack.make_raw::<usize>(k) };
		let (col_perm, stack) = unsafe { stack.make_raw::<usize>(k) };
		let (col_perm_inv, stack) = unsafe { stack.make_raw::<usize>(k) };
		let (mut rtv, stack) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack) };
		let mut rtv = rtv.as_mat_mut();
		let (mut r, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
		let mut r = r.as_mat_mut();
		let (mut p, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
		let mut p = p.as_mat_mut();
		let (mut r_tilde, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
		let mut r_tilde = r_tilde.as_mat_mut();

		let abs_residual = if params.initial_guess == InitialGuessStatus::MaybeNonZero {
			A.apply(r.rb_mut(), x.rb(), par, stack);
			z!(&mut r, &b).for_each(|uz!(r, b)| *r = *b - *r);

			r.norm_l2()
		} else {
			copy(b_norm)
		};

		if abs_residual < threshold {
			return Ok(BicgInfo {
				rel_residual: abs_residual / b_norm,
				abs_residual,
				iter_count: 0,
				non_exhaustive: NonExhaustive(()),
			});
		}

		p.copy_from(&r);
		r_tilde.copy_from(&r);

		for iter in 0..params.max_iters {
			let (mut v, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
			let mut v = v.as_mat_mut();
			let (mut y, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
			let mut y = y.as_mat_mut();
			{
				let (mut y0, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
				let mut y0 = y0.as_mat_mut();
				K1.apply(y0.rb_mut(), p.rb(), par, stack);
				K2.apply(y.rb_mut(), y0.rb(), par, stack);
			}
			A.apply(v.rb_mut(), y.rb(), par, stack);

			crate::linalg::matmul::matmul(rtv.rb_mut(), Accum::Replace, r_tilde.rb().transpose(), v.rb(), one::<T>(), par);
			let (_, row_perm, col_perm) = crate::linalg::lu::full_pivoting::factor::lu_in_place(
				rtv.rb_mut(),
				row_perm,
				row_perm_inv,
				col_perm,
				col_perm_inv,
				par,
				stack,
				Default::default(),
			);
			let mut rank = k;
			let tol = eps::<T::Real>() * from_f64::<T::Real>(k as f64) * abs(rtv[(0, 0)]);
			for i in 0..k {
				if abs(rtv[(i, i)]) < tol {
					rank = i;
					break;
				}
			}

			let (mut s, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
			let mut s = s.as_mat_mut();
			{
				let (mut rtr, stack) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack) };
				let mut rtr = rtr.as_mat_mut();
				crate::linalg::matmul::matmul(rtr.rb_mut(), Accum::Replace, r_tilde.rb().transpose(), r.rb(), one::<T>(), par);
				let (mut temp, _) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack) };
				let mut temp = temp.as_mat_mut();
				crate::perm::permute_rows(temp.rb_mut(), rtr.rb(), row_perm);
				crate::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
					rtv.rb().get(..rank, ..rank),
					temp.rb_mut().get_mut(..rank, ..),
					par,
				);
				crate::linalg::triangular_solve::solve_upper_triangular_in_place(
					rtv.rb().get(..rank, ..rank),
					temp.rb_mut().get_mut(..rank, ..),
					par,
				);
				temp.rb_mut().get_mut(rank.., ..).fill(zero());
				crate::perm::permute_rows(rtr.rb_mut(), temp.rb(), col_perm.inverse());
				let alpha = rtr.rb();

				s.copy_from(&r);
				crate::linalg::matmul::matmul(s.rb_mut(), Accum::Add, v.rb(), alpha.rb(), -one::<T>(), par);
				crate::linalg::matmul::matmul(
					x.rb_mut(),
					if iter == 0 && params.initial_guess == InitialGuessStatus::Zero {
						Accum::Replace
					} else {
						Accum::Add
					},
					y.rb(),
					alpha.rb(),
					one::<T>(),
					par,
				);
			}
			let norm = s.norm_l2();
			if norm < threshold {
				return Ok(BicgInfo {
					rel_residual: norm / b_norm,
					abs_residual: norm,
					iter_count: iter + 1,
					non_exhaustive: NonExhaustive(()),
				});
			}

			let (mut t, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
			let mut t = t.as_mat_mut();
			let (mut z, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
			let mut z = z.as_mat_mut();
			{
				let (mut z0, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
				let mut z0 = z0.as_mat_mut();
				K1.apply(z0.rb_mut(), s.rb(), par, stack);
				K2.apply(z.rb_mut(), z0.rb(), par, stack);
			}
			A.apply(t.rb_mut(), z.rb(), par, stack);

			let compute_w = |kt: MatRef<'_, T>, ks: MatRef<'_, T>| {
				let mut wt = zero::<T>();
				let mut ws = zero::<T>();
				for j in 0..k {
					let kt = kt.rb().col(j);
					let ks = ks.rb().col(j);
					ws = ws + kt.transpose() * ks;
					wt = wt + kt.transpose() * kt;
				}
				recip(wt) * ws
			};

			let w = {
				let mut kt = y;
				let (mut ks, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
				let mut ks = ks.as_mat_mut();
				K1.apply(kt.rb_mut(), t.rb(), par, stack);
				K1.apply(ks.rb_mut(), s.rb(), par, stack);
				compute_w(kt.rb(), ks.rb())
			};

			z!(&mut r, &s, &t).for_each(|uz!(r, s, t)| *r = *s - w * *t);
			z!(&mut x, &z).for_each(|uz!(x, z)| *x = *x + w * *z);
			z!(&mut p, &v).for_each(|uz!(p, v)| *p = *p - w * *v);

			callback(x.rb());

			let norm = r.norm_l2();
			if norm < threshold {
				return Ok(BicgInfo {
					rel_residual: norm / b_norm,
					abs_residual: norm,
					iter_count: iter + 1,
					non_exhaustive: NonExhaustive(()),
				});
			}

			let (mut rtt, stack) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack) };
			let mut rtt = rtt.as_mat_mut();
			{
				crate::linalg::matmul::matmul(rtt.rb_mut(), Accum::Replace, r_tilde.rb().transpose(), t.rb(), one::<T>(), par);
				let (mut temp, _) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack) };
				let mut temp = temp.as_mat_mut();
				crate::perm::permute_rows(temp.rb_mut(), rtt.rb(), row_perm);
				crate::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
					rtv.rb().get(..rank, ..rank),
					temp.rb_mut().get_mut(..rank, ..),
					par,
				);
				crate::linalg::triangular_solve::solve_upper_triangular_in_place(
					rtv.rb().get(..rank, ..rank),
					temp.rb_mut().get_mut(..rank, ..),
					par,
				);
				temp.rb_mut().get_mut(rank.., ..).fill(zero());
				crate::perm::permute_rows(rtt.rb_mut(), temp.rb(), col_perm.inverse());
			}

			let beta = rtt.rb();
			let mut tmp = v;
			crate::linalg::matmul::matmul(tmp.rb_mut(), Accum::Replace, p.rb(), beta.rb(), one::<T>(), par);
			z!(&mut p, &r, &tmp).for_each(|uz!(p, r, tmp)| *p = *r - *tmp);
		}
		Err(BicgError::NoConvergence {
			rel_residual: abs_residual / b_norm,
			abs_residual,
		})
	}
	implementation(out, &left_precond, &right_precond, &mat, rhs, params, &mut { callback }, par, stack)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::mat;
	use dyn_stack::MemBuffer;
	use equator::assert;
	use rand::prelude::*;

	#[test]
	fn test_bicgstab() {
		let ref mut rng = StdRng::seed_from_u64(0);

		let ref A = mat![[2.5, -1.0], [1.0, 3.1]];
		let ref sol = mat![[2.1, 2.1], [4.1, 3.2]];
		let ref rhs = A * sol;
		let ref mut diag = Mat::<f64>::identity(2, 2);
		for i in 0..2 {
			diag[(i, i)] = f64::exp(rand::distributions::Standard.sample(rng));
		}
		let ref diag = *diag;

		let ref mut out = Mat::<f64>::zeros(2, sol.ncols());
		let mut params = BicgParams::default();
		params.max_iters = 10;
		let result = bicgstab(
			out.as_mut(),
			diag.as_ref(),
			diag.as_ref(),
			A.as_ref(),
			rhs.as_ref(),
			params,
			|_| {},
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(bicgstab_scratch(
				diag.as_ref(),
				diag.as_ref(),
				A.as_ref(),
				sol.ncols(),
				Par::Seq,
			))),
		);
		let ref out = *out;

		assert!(result.is_ok());
		assert!((A * out - rhs).norm_l2() <= params.rel_tolerance * rhs.norm_l2());
	}
}
