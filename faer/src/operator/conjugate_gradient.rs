use crate::assert;
use crate::matrix_free::*;
use linalg::cholesky::llt_pivoting::factor as piv_llt;
use linalg::matmul::triangular::BlockStructure;

/// algorithm parameters
#[derive(Copy, Clone, Debug)]
pub struct CgParams<T: RealField> {
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

/// algorithm result
#[derive(Copy, Clone, Debug)]
pub struct CgInfo<T: RealField> {
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
pub enum CgError<T: ComplexField> {
	/// operator was detected to not be positive definite
	NonPositiveDefiniteOperator,
	/// preconditioner was detected to not be positive definite
	NonPositiveDefinitePreconditioner,
	/// convergence failure
	NoConvergence {
		/// absolute residual at the final step
		abs_residual: T::Real,
		/// relative residual at the final step
		rel_residual: T::Real,
	},
}

impl<T: RealField> Default for CgParams<T> {
	#[inline]
	#[math]
	fn default() -> Self {
		Self {
			initial_guess: InitialGuessStatus::MaybeNonZero,
			abs_tolerance: zero::<T>(),
			rel_tolerance: eps::<T>() * from_f64::<T>(128.0),
			max_iters: usize::MAX,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

/// computes the layout of required workspace for executing the conjugate gradient
/// algorithm
pub fn conjugate_gradient_scratch<T: ComplexField>(precond: impl Precond<T>, mat: impl LinOp<T>, rhs_ncols: usize, par: Par) -> StackReq {
	fn implementation<T: ComplexField>(M: &dyn Precond<T>, A: &dyn LinOp<T>, rhs_ncols: usize, par: Par) -> StackReq {
		let n = A.nrows();
		let k = rhs_ncols;

		let nk = temp_mat_scratch::<T>(n, k);
		let kk = temp_mat_scratch::<T>(k, k);
		let k_usize = StackReq::new::<usize>(k);
		let chol = piv_llt::cholesky_in_place_scratch::<usize, T>(k, par, default());
		StackReq::all_of(&[
			nk,      // residual
			nk,      // p
			nk,      // z
			kk,      // rtz
			k_usize, // perm
			k_usize, // perm_inv
			StackReq::any_of(&[
				StackReq::all_of(&[
					nk, // Ap
					kk, // ptAp | rtz_new
					StackReq::any_of(&[
						A.apply_scratch(k, par),
						chol, // ptAp | rtz
						StackReq::all_of(&[
							kk, // alpha | beta
							kk, // alpha_perm | beta_perm
						]),
					]),
				]),
				M.apply_scratch(k, par),
			]),
		])
	}
	implementation(&precond, &mat, rhs_ncols, par)
}

/// executes the conjugate gradient using the provided preconditioner
///
/// # note
/// this function is also optimized for a rhs with multiple columns
#[inline]
#[track_caller]
pub fn conjugate_gradient<T: ComplexField>(
	out: MatMut<'_, T>,
	precond: impl Precond<T>,
	mat: impl LinOp<T>,
	rhs: MatRef<'_, T>,
	params: CgParams<T::Real>,
	callback: impl FnMut(MatRef<'_, T>),
	par: Par,
	stack: &mut MemStack,
) -> Result<CgInfo<T::Real>, CgError<T::Real>> {
	#[track_caller]
	#[math]
	fn implementation<T: ComplexField>(
		mut x: MatMut<'_, T>,
		M: &dyn Precond<T>,
		A: &dyn LinOp<T>,
		b: MatRef<'_, T>,

		params: CgParams<T::Real>,
		callback: &mut dyn FnMut(MatRef<'_, T>),
		par: Par,
		mut stack: &mut MemStack,
	) -> Result<CgInfo<T::Real>, CgError<T::Real>> {
		assert!(A.nrows() == A.ncols());

		let n = A.nrows();
		let k = b.ncols();
		let b_norm = b.norm_l2();
		if b_norm == zero::<T::Real>() {
			x.fill(zero());
			return Ok(CgInfo {
				abs_residual: zero::<T::Real>(),
				rel_residual: zero::<T::Real>(),
				iter_count: 0,
				non_exhaustive: NonExhaustive(()),
			});
		}

		let rel_threshold = params.rel_tolerance * b_norm;
		let abs_threshold = params.abs_tolerance;

		let threshold = if abs_threshold > rel_threshold { abs_threshold } else { rel_threshold };

		let (mut r, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
		let mut r = r.as_mat_mut();
		let (mut p, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
		let mut p = p.as_mat_mut();
		let (mut z, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
		let mut z = z.as_mat_mut();
		let (mut rtz, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack.rb_mut()) };
		let mut rtz = rtz.as_mat_mut();

		let (perm, mut stack) = unsafe { stack.rb_mut().make_raw::<usize>(k) };
		let (perm_inv, mut stack) = unsafe { stack.rb_mut().make_raw::<usize>(k) };

		let abs_residual = if params.initial_guess == InitialGuessStatus::MaybeNonZero {
			A.apply(r.rb_mut(), x.rb(), par, stack.rb_mut());
			z!(&mut r, &b).for_each(|uz!(res, rhs)| *res = *rhs - *res);
			r.norm_l2()
		} else {
			copy(b_norm)
		};

		if abs_residual < threshold {
			return Ok(CgInfo {
				rel_residual: abs_residual / b_norm,
				abs_residual,
				iter_count: 0,
				non_exhaustive: NonExhaustive(()),
			});
		}

		let tril = BlockStructure::TriangularLower;

		{
			M.apply(p.rb_mut(), r.rb(), par, stack.rb_mut());

			crate::linalg::matmul::triangular::matmul(
				rtz.rb_mut(),
				tril,
				Accum::Replace,
				r.rb().adjoint(),
				BlockStructure::Rectangular,
				p.rb(),
				BlockStructure::Rectangular,
				one::<T>(),
				par,
			);
		}
		for iter in 0..params.max_iters {
			{
				let (mut Ap, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
				let mut Ap = Ap.as_mat_mut();
				let (mut ptAp, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack.rb_mut()) };
				let mut ptAp = ptAp.as_mat_mut();

				A.apply(Ap.rb_mut(), p.rb(), par, stack.rb_mut());
				crate::linalg::matmul::triangular::matmul(
					ptAp.rb_mut(),
					tril,
					Accum::Replace,
					p.rb().adjoint(),
					BlockStructure::Rectangular,
					Ap.rb(),
					BlockStructure::Rectangular,
					one::<T>(),
					par,
				);

				let (info, llt_perm) = match piv_llt::cholesky_in_place(ptAp.rb_mut(), perm, perm_inv, par, stack.rb_mut(), Default::default()) {
					Ok(ok) => ok,
					Err(_) => return Err(CgError::NonPositiveDefiniteOperator),
				};

				let (mut alpha, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack.rb_mut()) };
				let mut alpha = alpha.as_mat_mut();
				let (mut alpha_perm, _) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack.rb_mut()) };
				let mut alpha_perm = alpha_perm.as_mat_mut();
				alpha.copy_from(&rtz);
				for j in 0..k {
					for i in 0..j {
						alpha.write(i, j, conj(alpha[(j, i)]));
					}
				}
				crate::perm::permute_rows(alpha_perm.rb_mut(), alpha.rb(), llt_perm);
				crate::linalg::triangular_solve::solve_lower_triangular_in_place(
					ptAp.rb().get(..info.rank, ..info.rank),
					alpha_perm.rb_mut().get_mut(..info.rank, ..),
					par,
				);
				crate::linalg::triangular_solve::solve_upper_triangular_in_place(
					ptAp.rb().get(..info.rank, ..info.rank).adjoint(),
					alpha_perm.rb_mut().get_mut(..info.rank, ..),
					par,
				);
				alpha_perm.rb_mut().get_mut(info.rank.., ..).fill(zero());
				crate::perm::permute_rows(alpha.rb_mut(), alpha_perm.rb(), llt_perm.inverse());

				crate::linalg::matmul::matmul(
					x.rb_mut(),
					if iter == 0 && params.initial_guess == InitialGuessStatus::Zero {
						Accum::Replace
					} else {
						Accum::Add
					},
					p.rb(),
					alpha.rb(),
					one::<T>(),
					par,
				);
				crate::linalg::matmul::matmul(r.rb_mut(), Accum::Add, Ap.rb(), alpha.rb(), -one::<T>(), par);
				callback(x.rb());
			}

			let abs_residual = r.norm_l2();
			if abs_residual < threshold {
				return Ok(CgInfo {
					rel_residual: abs_residual / b_norm,
					abs_residual,
					iter_count: iter + 1,
					non_exhaustive: NonExhaustive(()),
				});
			}

			M.apply(z.rb_mut(), r.rb(), par, stack.rb_mut());

			let (mut rtz_new, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack.rb_mut()) };
			let mut rtz_new = rtz_new.as_mat_mut();
			crate::linalg::matmul::triangular::matmul(
				rtz_new.rb_mut(),
				tril,
				Accum::Replace,
				r.rb().adjoint(),
				BlockStructure::Rectangular,
				z.rb(),
				BlockStructure::Rectangular,
				one::<T>(),
				par,
			);

			{
				let (info, llt_perm) = match piv_llt::cholesky_in_place(rtz.rb_mut(), perm, perm_inv, par, stack.rb_mut(), Default::default()) {
					Ok(ok) => ok,
					Err(_) => return Err(CgError::NonPositiveDefiniteOperator),
				};
				let (mut beta, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack.rb_mut()) };
				let mut beta = beta.as_mat_mut();
				let (mut beta_perm, _) = unsafe { temp_mat_uninit::<T, _, _>(k, k, stack.rb_mut()) };
				let mut beta_perm = beta_perm.as_mat_mut();
				beta.copy_from(&rtz_new);
				for j in 0..k {
					for i in 0..j {
						beta.write(i, j, conj(beta[(j, i)]));
					}
				}
				crate::perm::permute_rows(beta_perm.rb_mut(), beta.rb(), llt_perm);
				crate::linalg::triangular_solve::solve_lower_triangular_in_place(
					rtz.rb().get(..info.rank, ..info.rank),
					beta_perm.rb_mut().get_mut(..info.rank, ..),
					par,
				);
				crate::linalg::triangular_solve::solve_upper_triangular_in_place(
					rtz.rb().get(..info.rank, ..info.rank).adjoint(),
					beta_perm.rb_mut().get_mut(..info.rank, ..),
					par,
				);
				beta_perm.rb_mut().get_mut(info.rank.., ..).fill(zero());
				crate::perm::permute_rows(beta.rb_mut(), beta_perm.rb(), llt_perm.inverse());
				rtz.copy_from(&rtz_new);

				crate::linalg::matmul::matmul(z.rb_mut(), Accum::Add, p.rb(), beta.rb(), one::<T>(), par);
				p.copy_from(&z);
			}
		}

		Err(CgError::NoConvergence {
			rel_residual: abs_residual / b_norm,
			abs_residual,
		})
	}

	implementation(out, &precond, &mat, rhs, params, &mut { callback }, par, stack)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::{mat, matrix_free};
	use dyn_stack::MemBuffer;
	use equator::assert;
	use rand::prelude::*;

	#[test]
	fn test_cg() {
		let ref A = mat![[2.5, -1.0], [-1.0, 3.1]];
		let ref sol = mat![[2.1, 2.4], [4.1, 4.0]];
		let ref rhs = A * sol;
		let ref mut out = Mat::<f64>::zeros(2, sol.ncols());
		let mut params = CgParams::default();
		params.max_iters = 10;
		let precond = matrix_free::IdentityPrecond { dim: 2 };
		let result = conjugate_gradient(
			out.as_mut(),
			precond,
			A.as_ref(),
			rhs.as_ref(),
			params,
			|_| {},
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(conjugate_gradient_scratch(precond, A.as_ref(), 2, Par::Seq))),
		);
		let ref out = *out;

		assert!(result.is_ok());
		let result = result.unwrap();
		assert!((A * out - rhs).norm_l2() <= params.rel_tolerance * rhs.norm_l2());
		assert!(result.iter_count <= 1);
	}

	#[test]
	fn test_cg_breakdown() {
		let ref mut rng = StdRng::seed_from_u64(0);
		let n = 10;
		let k = 15;
		let ref Q: Mat<c64> = UnitaryMat {
			dim: n,
			standard_normal: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.sample(rng);
		let mut d = Col::zeros(n);
		for i in 0..n {
			d[i] = c64::new(f64::exp(rand::distributions::Standard.sample(rng)).recip(), 0.0);
		}
		let ref A = Q * d.as_ref().as_diagonal() * Q.adjoint();
		let ref mut diag = Mat::<c64>::identity(n, n);
		for i in 0..n {
			diag[(i, i)] = c64::new(f64::exp(rand::distributions::Standard.sample(rng)).recip(), 0.0);
		}
		let ref diag = *diag;
		let ref mut sol = CwiseMatDistribution {
			nrows: n,
			ncols: k,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.sample(rng);

		for i in 0..n {
			sol[(i, k - 1)] = c64::new(0.0, 0.0);
			for j in 0..k - 1 {
				let val = sol[(i, j)];
				sol[(i, k - 1)] += val;
			}
		}

		let ref sol = *sol;
		let ref rhs = A * sol;
		let ref mut out = Mat::<c64>::zeros(n, k);
		let params = CgParams::default();
		let result = conjugate_gradient(
			out.as_mut(),
			diag.as_ref(),
			A.as_ref(),
			rhs.as_ref(),
			params,
			|_| {},
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(conjugate_gradient_scratch::<c64>(
				diag.as_ref(),
				A.as_ref(),
				k,
				Par::Seq,
			))),
		);
		let ref out = *out;

		assert!(result.is_ok());
		let result = result.unwrap();
		assert!((A * out - rhs).norm_l2() <= params.rel_tolerance * rhs.norm_l2());
		assert!(result.iter_count <= 1);
	}
}
