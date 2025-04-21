use super::*;
use crate::{assert, debug_assert};
use linalg::matmul::matmul;
use linalg::{householder, qr};

/// algorithm parameters
#[derive(Copy, Clone, Debug)]
pub struct LsmrParams<T> {
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

impl<T: RealField> Default for LsmrParams<T> {
	#[inline]
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
pub struct LsmrInfo<T> {
	/// absolute residual at the final step
	pub abs_residual: T,
	/// relative residual at the final step
	pub rel_residual: T,
	/// number of iterations executed by the algorithm
	pub iter_count: usize,

	#[doc(hidden)]
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

/// algorithm error
#[derive(Copy, Clone, Debug)]
pub enum LsmrError<T> {
	/// convergence failure
	NoConvergence {
		/// absolute residual at the final step
		abs_residual: T,
		/// relative residual at the final step
		rel_residual: T,
	},
}

/// computes the size and alignment of required workspace for executing the lsmr
/// algorithm
pub fn lsmr_scratch<T: ComplexField>(right_precond: impl BiPrecond<T>, mat: impl BiLinOp<T>, rhs_ncols: usize, par: Par) -> StackReq {
	fn implementation<T: ComplexField>(M: &dyn BiPrecond<T>, A: &dyn BiLinOp<T>, rhs_ncols: usize, par: Par) -> StackReq {
		let m = A.nrows();
		let n = A.ncols();
		let mut k = rhs_ncols;

		assert!(k < isize::MAX as usize);
		if k > n {
			k = k.msrv_checked_next_multiple_of(n).unwrap();
		}
		assert!(k < isize::MAX as usize);

		let s = Ord::min(k, Ord::min(n, m));

		let mk = temp_mat_scratch::<T>(m, k);
		let nk = temp_mat_scratch::<T>(n, k);
		let ss = temp_mat_scratch::<T>(s, s);
		let ss2 = temp_mat_scratch::<T>(2 * s, 2 * s);
		let sk = temp_mat_scratch::<T>(s, k);
		let sk2 = temp_mat_scratch::<T>(2 * s, 2 * k);

		let ms_bs = qr::no_pivoting::factor::recommended_blocksize::<T>(m, s);
		let ns_bs = qr::no_pivoting::factor::recommended_blocksize::<T>(n, s);
		let ss_bs = qr::no_pivoting::factor::recommended_blocksize::<T>(2 * s, 2 * s);

		let AT = A.transpose_apply_scratch(k, par);
		let A = A.apply_scratch(k, par);
		let MT = M.transpose_apply_in_place_scratch(k, par);
		let M = M.apply_in_place_scratch(k, par);

		let m_qr = StackReq::any_of(&[
			temp_mat_scratch::<T>(ms_bs, s),
			qr::no_pivoting::factor::qr_in_place_scratch::<T>(m, s, ms_bs, par, Default::default()),
			householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(m, ms_bs, s),
		]);

		let n_qr = StackReq::any_of(&[
			temp_mat_scratch::<T>(ns_bs, s),
			qr::no_pivoting::factor::qr_in_place_scratch::<T>(n, s, ns_bs, par, Default::default()),
			householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(n, ns_bs, s),
		]);

		let s_qr = StackReq::any_of(&[
			temp_mat_scratch::<T>(ss_bs, s),
			qr::no_pivoting::factor::qr_in_place_scratch::<T>(2 * s, 2 * s, ss_bs, par, Default::default()),
			householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(2 * s, ss_bs, 2 * s),
		]);

		StackReq::all_of(&[
			mk,  // u
			nk,  // v
			sk,  // beta
			sk,  // alpha
			sk,  // zetabar
			sk,  // alphabar
			sk,  // theta
			sk2, // pbar_adjoint
			nk,  // vold
			StackReq::any_of(&[StackReq::all_of(&[mk, StackReq::any_of(&[A, M, m_qr])])]),
			StackReq::any_of(&[StackReq::all_of(&[nk, StackReq::any_of(&[AT, MT, n_qr])])]),
			ss2, // p_adjoint
			ss,  // rho
			ss,  // thetaold
			ss,  // rhobar
			ss,  // thetabar
			ss,  // zeta
			ss,  // zetabar
			StackReq::all_of(&[temp_mat_scratch::<T>(2 * s, 2 * s), s_qr]),
		])
	}

	implementation(&right_precond, &mat, rhs_ncols, par)
}

/// executes lsmr using the provided preconditioner
///
/// # note
/// this function is also optimized for a rhs with multiple columns
#[track_caller]
pub fn lsmr<T: ComplexField>(
	out: MatMut<'_, T>,
	right_precond: impl BiPrecond<T>,
	mat: impl BiLinOp<T>,
	rhs: MatRef<'_, T>,
	params: LsmrParams<T::Real>,
	callback: impl FnMut(MatRef<'_, T>),
	par: Par,
	stack: &mut MemStack,
) -> Result<LsmrInfo<T::Real>, LsmrError<T::Real>> {
	#[track_caller]
	#[math]
	fn implementation<T: ComplexField>(
		mut x: MatMut<'_, T>,
		M: &impl BiPrecond<T>,
		A: &impl BiLinOp<T>,
		b: MatRef<'_, T>,
		params: LsmrParams<T::Real>,
		callback: &mut dyn FnMut(MatRef<'_, T>),
		par: Par,
		stack: &mut MemStack,
	) -> Result<LsmrInfo<T::Real>, LsmrError<T::Real>> {
		fn thin_qr<T: ComplexField>(mut Q: MatMut<'_, T>, mut R: MatMut<'_, T>, mut mat: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
			let k = R.nrows();
			let bs = qr::no_pivoting::factor::recommended_blocksize::<T>(mat.nrows(), mat.ncols());
			let (mut house, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(bs, Ord::min(mat.nrows(), mat.ncols()), stack) };
			let mut house = house.as_mat_mut();

			qr::no_pivoting::factor::qr_in_place(mat.rb_mut(), house.rb_mut(), par, stack.rb_mut(), Default::default());

			R.fill(zero());
			R.copy_from_triangular_upper(mat.rb().get(..k, ..k));
			Q.fill(zero());
			Q.rb_mut().diagonal_mut().column_vector_mut().fill(one::<T>());
			householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
				mat.rb(),
				house.rb(),
				Conj::No,
				Q.rb_mut(),
				par,
				stack.rb_mut(),
			);
		}

		let m = A.nrows();
		let n = A.ncols();
		let mut k = b.ncols();
		{
			let out = x.rb();
			let mat = A;
			let right_precond = M;
			let rhs = b;
			assert!(all(
				right_precond.nrows() == mat.ncols(),
				right_precond.ncols() == mat.ncols(),
				rhs.nrows() == mat.nrows(),
				out.nrows() == mat.ncols(),
				out.ncols() == rhs.ncols(),
			));
		}

		if m == 0 || n == 0 || k == 0 || core::mem::size_of::<T::Unit>() == 0 {
			x.fill(zero());
			return Ok(LsmrInfo {
				abs_residual: zero::<T::Real>(),
				rel_residual: zero::<T::Real>(),
				iter_count: 0,
				non_exhaustive: NonExhaustive(()),
			});
		}

		debug_assert!(all(m < isize::MAX as usize, n < isize::MAX as usize, k < isize::MAX as usize));
		let actual_k = k;
		if k > n {
			// pad to avoid last block slowing down the rest
			k = k.msrv_checked_next_multiple_of(n).unwrap();
		}
		debug_assert!(k < isize::MAX as usize);

		let s = Ord::min(k, Ord::min(n, m));

		let mut stack = stack;

		let (mut u, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(m, k, stack.rb_mut()) };
		let mut u = u.as_mat_mut();
		let (mut beta, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, k, stack.rb_mut()) };
		let mut beta = beta.as_mat_mut();

		let (mut v, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
		let mut v = v.as_mat_mut();
		let (mut alpha, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, k, stack.rb_mut()) };
		let mut alpha = alpha.as_mat_mut();

		let (mut zetabar, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, k, stack.rb_mut()) };
		let mut zetabar = zetabar.as_mat_mut();
		let (mut alphabar, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, k, stack.rb_mut()) };
		let mut alphabar = alphabar.as_mat_mut();
		let (mut theta, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, k, stack.rb_mut()) };
		let mut theta = theta.as_mat_mut();
		let (mut pbar_adjoint, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(2 * s, 2 * k, stack.rb_mut()) };
		let mut pbar_adjoint = pbar_adjoint.as_mat_mut();

		let (mut w, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
		let mut w = w.as_mat_mut();
		let (mut wbar, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
		let mut wbar = wbar.as_mat_mut();

		{
			let (mut qr, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(m, k, stack.rb_mut()) };
			let mut qr = qr.as_mat_mut();
			if params.initial_guess == InitialGuessStatus::Zero {
				qr.rb_mut().get_mut(.., ..actual_k).copy_from(b);
				qr.rb_mut().get_mut(.., actual_k..).fill(zero());
			} else {
				A.apply(qr.rb_mut().rb_mut().get_mut(.., ..actual_k), x.rb(), par, stack.rb_mut());
				z!(qr.rb_mut().get_mut(.., ..actual_k), &b).for_each(|uz!(ax, b)| *ax = *b - *ax);
				qr.rb_mut().get_mut(.., actual_k..).fill(zero());
			}
			let mut start = 0;
			while start < k {
				let end = Ord::min(k - start, s) + start;
				let len = end - start;
				thin_qr(
					u.rb_mut().get_mut(.., start..end),
					beta.rb_mut().get_mut(..len, start..end),
					qr.rb_mut().get_mut(.., start..end),
					par,
					stack.rb_mut(),
				);
				start = end;
			}
		}

		{
			let (mut qr, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
			let mut qr = qr.as_mat_mut();
			A.adjoint_apply(qr.rb_mut(), u.rb(), par, stack.rb_mut());
			M.adjoint_apply_in_place(qr.rb_mut(), par, stack.rb_mut());
			let mut start = 0;
			while start < k {
				let end = Ord::min(k - start, s) + start;
				let len = end - start;
				thin_qr(
					v.rb_mut().get_mut(.., start..end),
					alpha.rb_mut().get_mut(..len, start..end),
					qr.rb_mut().get_mut(.., start..end),
					par,
					stack.rb_mut(),
				);
				start = end;
			}
		}

		zetabar.fill(zero());
		let mut start = 0;
		while start < k {
			let end = Ord::min(k - start, s) + start;
			let len = end - start;
			matmul(
				zetabar.rb_mut().get_mut(..len, start..end),
				Accum::Replace,
				alpha.rb().get(..len, start..end),
				beta.rb().get(..len, start..end),
				one::<T>(),
				par,
			);
			start = end;
		}
		alphabar.copy_from(&alpha);
		pbar_adjoint.fill(zero());
		let mut start = 0;
		while start < k {
			let end = Ord::min(k - start, s) + start;
			let len = end - start;
			pbar_adjoint
				.rb_mut()
				.get_mut(..2 * len, 2 * start..2 * end)
				.diagonal_mut()
				.column_vector_mut()
				.fill(one());
			start = end;
		}
		theta.fill(zero());
		w.fill(zero());
		wbar.fill(zero());

		let mut norm;
		let norm_ref = if params.initial_guess == InitialGuessStatus::Zero {
			norm = zetabar.norm_l2();
			copy(norm)
		} else {
			norm = zetabar.norm_l2();
			let (mut tmp, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, actual_k, stack.rb_mut()) };
			let mut tmp = tmp.as_mat_mut();
			A.adjoint_apply(tmp.rb_mut(), b, par, stack.rb_mut());
			M.adjoint_apply_in_place(tmp.rb_mut(), par, stack.rb_mut());
			tmp.norm_l2()
		};
		let threshold = norm_ref * params.rel_tolerance;

		if norm_ref == zero::<T::Real>() {
			x.fill(zero());
			return Ok(LsmrInfo {
				abs_residual: zero::<T::Real>(),
				rel_residual: zero::<T::Real>(),
				iter_count: 0,
				non_exhaustive: NonExhaustive(()),
			});
		}

		if norm <= threshold {
			return Ok(LsmrInfo {
				abs_residual: zero::<T::Real>(),
				rel_residual: zero::<T::Real>(),
				iter_count: 0,
				non_exhaustive: NonExhaustive(()),
			});
		}

		for iter in 0..params.max_iters {
			let (mut vold, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
			let mut vold = vold.as_mat_mut();
			{
				let (mut qr, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(m, k, stack.rb_mut()) };
				let mut qr = qr.as_mat_mut();
				vold.copy_from(&v);
				M.apply_in_place(v.rb_mut(), par, stack.rb_mut());
				A.apply(qr.rb_mut(), v.rb(), par, stack.rb_mut());

				let mut start = 0;
				while start < k {
					let s = Ord::min(k - start, s);
					let end = start + s;
					matmul(
						qr.rb_mut().get_mut(.., start..end),
						Accum::Add,
						u.rb().get(.., start..end),
						alpha.rb().get(..s, start..end).adjoint(),
						-one::<T>(),
						par,
					);
					thin_qr(
						u.rb_mut().get_mut(.., start..end),
						beta.rb_mut().get_mut(..s, start..end),
						qr.rb_mut().get_mut(.., start..end),
						par,
						stack.rb_mut(),
					);
					start = end;
				}
			}

			{
				let (mut qr, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
				let mut qr = qr.as_mat_mut();
				A.adjoint_apply(qr.rb_mut(), u.rb(), par, stack.rb_mut());
				M.adjoint_apply_in_place(qr.rb_mut(), par, stack.rb_mut());

				let mut start = 0;
				while start < k {
					let s = Ord::min(k - start, s);
					let end = start + s;
					matmul(
						qr.rb_mut().get_mut(.., start..end),
						Accum::Add,
						vold.rb().get(.., start..end),
						beta.rb().get(..s, start..end).adjoint(),
						-one::<T>(),
						par,
					);

					// now contains M v_old
					vold.rb_mut().get_mut(.., start..end).copy_from(v.rb().get(.., start..end));

					thin_qr(
						v.rb_mut().get_mut(.., start..end),
						alpha.rb_mut().get_mut(..s, start..end),
						qr.rb_mut().get_mut(.., start..end),
						par,
						stack.rb_mut(),
					);
					start = end;
				}
			}

			let mut Mvold = vold;

			let mut start = 0;
			while start < k {
				let s = Ord::min(k - start, s);
				let end = start + s;

				let mut x = x.rb_mut().get_mut(.., start..Ord::min(actual_k, end));
				let mut Mvold = Mvold.rb_mut().get_mut(.., start..end);
				let mut w = w.rb_mut().get_mut(.., start..end);
				let mut wbar = wbar.rb_mut().get_mut(.., start..end);

				let alpha = alpha.rb_mut().get_mut(..s, start..end);
				let beta = beta.rb_mut().get_mut(..s, start..end);
				let mut zetabar = zetabar.rb_mut().get_mut(..s, start..end);
				let mut alphabar = alphabar.rb_mut().get_mut(..s, start..end);
				let mut theta = theta.rb_mut().get_mut(..s, start..end);
				let mut pbar_adjoint = pbar_adjoint.rb_mut().get_mut(..2 * s, 2 * start..2 * end);

				let (mut p_adjoint, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(2 * s, 2 * s, stack.rb_mut()) };
				let mut p_adjoint = p_adjoint.as_mat_mut();

				let (mut rho, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
				let mut rho = rho.as_mat_mut();
				let (mut thetaold, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
				let mut thetaold = thetaold.as_mat_mut();
				let (mut rhobar, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
				let mut rhobar = rhobar.as_mat_mut();
				let (mut thetabar, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
				let mut thetabar = thetabar.as_mat_mut();
				let (mut zeta, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
				let mut zeta = zeta.as_mat_mut();
				let (mut zetabar_tmp, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
				let mut zetabar_tmp = zetabar_tmp.as_mat_mut();

				{
					let (mut qr, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(2 * s, s, stack.rb_mut()) };
					let mut qr = qr.as_mat_mut();
					qr.rb_mut().get_mut(..s, ..).copy_from(alphabar.rb().adjoint());
					qr.rb_mut().get_mut(s.., ..).copy_from(&beta);
					thin_qr(p_adjoint.rb_mut(), rho.rb_mut(), qr.rb_mut(), par, stack.rb_mut());
				}

				thetaold.copy_from(&theta);
				matmul(theta.rb_mut(), Accum::Replace, alpha.rb(), p_adjoint.rb().get(s.., ..s), one::<T>(), par);
				matmul(
					alphabar.rb_mut(),
					Accum::Replace,
					alpha.rb(),
					p_adjoint.rb().get(s.., s..),
					one::<T>(),
					par,
				);

				matmul(
					thetabar.rb_mut(),
					Accum::Replace,
					rho.rb(),
					pbar_adjoint.rb().get(s.., ..s),
					one::<T>(),
					par,
				);
				{
					let (mut qr, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(2 * s, s, stack.rb_mut()) };
					let mut qr = qr.as_mat_mut();
					matmul(
						qr.rb_mut().get_mut(..s, ..),
						Accum::Replace,
						pbar_adjoint.rb().adjoint().get(s.., s..),
						rho.rb().adjoint(),
						one::<T>(),
						par,
					);
					qr.rb_mut().get_mut(s.., ..).copy_from(&theta);
					thin_qr(pbar_adjoint.rb_mut(), rhobar.rb_mut(), qr.rb_mut(), par, stack.rb_mut());
				}

				matmul(
					zeta.rb_mut(),
					Accum::Replace,
					pbar_adjoint.rb().adjoint().get(..s, ..s),
					zetabar.rb(),
					one::<T>(),
					par,
				);
				matmul(
					zetabar_tmp.rb_mut(),
					Accum::Replace,
					pbar_adjoint.rb().adjoint().get(s.., ..s),
					zetabar.rb(),
					one::<T>(),
					par,
				);
				zetabar.copy_from(&zetabar_tmp);

				matmul(Mvold.rb_mut(), Accum::Add, w.rb(), thetaold.rb().adjoint(), -one::<T>(), par);
				crate::linalg::triangular_solve::solve_lower_triangular_in_place(rho.rb().transpose(), Mvold.rb_mut().transpose_mut(), par);
				w.copy_from(&Mvold);

				matmul(Mvold.rb_mut(), Accum::Add, wbar.rb(), thetabar.rb().adjoint(), -one::<T>(), par);
				crate::linalg::triangular_solve::solve_lower_triangular_in_place(rhobar.rb().transpose(), Mvold.rb_mut().transpose_mut(), par);
				wbar.copy_from(&Mvold);

				let actual_s = x.ncols();
				matmul(
					x.rb_mut(),
					if iter == 0 && params.initial_guess == InitialGuessStatus::Zero {
						Accum::Replace
					} else {
						Accum::Add
					},
					wbar.rb(),
					zeta.rb().get(.., ..actual_s),
					one::<T>(),
					par,
				);
				start = end;
			}
			norm = zetabar.norm_l2();
			callback(x.rb());
			if norm <= threshold {
				return Ok(LsmrInfo {
					rel_residual: norm / norm_ref,
					abs_residual: norm,
					iter_count: iter + 1,
					non_exhaustive: NonExhaustive(()),
				});
			}
		}

		Err(LsmrError::NoConvergence {
			rel_residual: norm / norm_ref,
			abs_residual: norm,
		})
	}
	implementation(out, &right_precond, &mat, rhs, params, &mut { callback }, par, stack)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use dyn_stack::MemBuffer;
	use equator::assert;

	#[test]
	fn test_lsmr() {
		let ref mut rng = StdRng::seed_from_u64(0);
		let m = 100;
		let n = 80;
		for k in [1, 2, 4, 7, 10, 40, 80, 100] {
			let A: Mat<c64> = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.sample(rng);
			let b: Mat<c64> = CwiseMatDistribution {
				nrows: m,
				ncols: k,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.sample(rng);
			let k = b.ncols();

			let ref mut diag = Scale(c64::new(2.0, 0.0)) * Mat::<c64>::identity(n, n);
			for i in 0..n {
				diag[(i, i)] = (128.0 * f64::exp(rand::distributions::Standard.sample(rng))).into();
			}
			for i in 0..n - 1 {
				diag[(i + 1, i)] = f64::exp(rand::distributions::Standard.sample(rng)).into();
			}

			let params = LsmrParams::default();

			let rand = CwiseMatDistribution {
				nrows: n,
				ncols: k,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			};
			let mut out = rand.sample(rng);

			let result = lsmr(
				out.as_mut(),
				diag.as_ref(),
				A.as_ref(),
				b.as_ref(),
				params,
				|_| {},
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(lsmr_scratch(diag.as_ref(), A.as_ref(), k, Par::Seq))),
			);
			assert!(result.is_ok());
			let result = result.unwrap();
			assert!(result.iter_count <= (4 * n).msrv_div_ceil(Ord::min(k, n)));
		}
	}

	#[test]
	fn test_breakdown() {
		let ref mut rng = StdRng::seed_from_u64(0);
		let m = 100;
		let n = 80;
		for k in [1, 2, 4, 7, 10, 40, 80, 100] {
			let A: Mat<c64> = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.sample(rng);
			let b: Mat<c64> = CwiseMatDistribution {
				nrows: m,
				ncols: k,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.sample(rng);
			let b = crate::concat![[b, b]];
			let k = b.ncols();

			let ref mut diag = Scale(c64::new(2.0, 0.0)) * Mat::<c64>::identity(n, n);
			for i in 0..n {
				diag[(i, i)] = (128.0 * f64::exp(rand::distributions::Standard.sample(rng))).into();
			}
			for i in 0..n - 1 {
				diag[(i + 1, i)] = f64::exp(rand::distributions::Standard.sample(rng)).into();
			}

			let params = LsmrParams::default();

			let rand = CwiseMatDistribution {
				nrows: n,
				ncols: k,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			};
			let mut out = rand.sample(rng);
			let result = lsmr(
				out.as_mut(),
				diag.as_ref(),
				A.as_ref(),
				b.as_ref(),
				params,
				|_| {},
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(lsmr_scratch(diag.as_ref(), A.as_ref(), k, Par::Seq))),
			);
			assert!(result.is_ok());
			let result = result.unwrap();
			assert!(result.iter_count <= (4 * n).msrv_div_ceil(Ord::min(k, n)));
		}
	}
}
