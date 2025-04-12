use crate::assert;
use crate::internal_prelude::*;
use crate::linalg::matmul::internal::*;
use linalg::matmul::triangular::BlockStructure;
use pulp::Simd;

#[inline(always)]
#[math]
fn simd_cholesky_row_batch<'N, T: ComplexField, S: Simd>(
	simd: T::SimdCtx<S>,
	A: MatMut<'_, T, Dim<'N>, Dim<'N>, ContiguousFwd>,
	D: RowMut<'_, T, Dim<'N>>,

	start: IdxInc<'N>,

	is_llt: bool,
	regularize: bool,
	eps: T::Real,
	delta: T::Real,
	signs: Option<&Array<'N, i8>>,
) -> Result<usize, usize> {
	let mut A = A;
	let mut D = D;

	let n = A.ncols();

	with_dim!(TAIL, *n - *start);

	let simd = SimdCtx::<T, S>::new_force_mask(simd, TAIL);
	let (idx_head, indices, idx_tail) = simd.indices();
	assert!(idx_head.is_none());
	let Some(idx_tail) = idx_tail else { panic!() };

	let mut count = 0usize;

	for j in n.indices() {
		with_dim!(LEFT, *j);

		let (A_0, Aj) = A.rb_mut().split_at_col_mut(j.into());
		let A_0 = A_0.as_col_shape(LEFT);
		let A10 = A_0.subrows(start, TAIL);

		let mut Aj = Aj.col_mut(0).subrows_mut(start, TAIL);

		{
			let D = D.rb().subcols(IdxInc::ZERO, LEFT);
			let mut Aj = Aj.rb_mut();
			let mut iter = indices.clone();
			let i0 = iter.next();
			let i1 = iter.next();
			let i2 = iter.next();

			match (i0, i1, i2) {
				(None, None, None) => {
					let mut Aij = simd.read(Aj.rb(), idx_tail);

					for k in LEFT.indices() {
						let Ak = A10.col(k);

						let D = real(D[k]);
						let D = if is_llt { one() } else { D };

						let Ajk = simd.splat(&mul_real(conj(A_0[(j, k)]), -D));

						let Aik = simd.read(Ak, idx_tail);
						Aij = simd.mul_add(Ajk, Aik, Aij);
					}
					simd.write(Aj.rb_mut(), idx_tail, Aij);
				},
				(Some(i0), None, None) => {
					let mut A0j = simd.read(Aj.rb(), i0);
					let mut Aij = simd.read(Aj.rb(), idx_tail);

					for k in LEFT.indices() {
						let Ak = A10.col(k);

						let D = real(D[k]);
						let D = if is_llt { one() } else { D };

						let Ajk = simd.splat(&mul_real(conj(A_0[(j, k)]), -D));

						let A0k = simd.read(Ak, i0);
						let Aik = simd.read(Ak, idx_tail);
						A0j = simd.mul_add(Ajk, A0k, A0j);
						Aij = simd.mul_add(Ajk, Aik, Aij);
					}
					simd.write(Aj.rb_mut(), i0, A0j);
					simd.write(Aj.rb_mut(), idx_tail, Aij);
				},
				(Some(i0), Some(i1), None) => {
					let mut A0j = simd.read(Aj.rb(), i0);
					let mut A1j = simd.read(Aj.rb(), i1);
					let mut Aij = simd.read(Aj.rb(), idx_tail);

					for k in LEFT.indices() {
						let Ak = A10.col(k);

						let D = real(D[k]);
						let D = if is_llt { one() } else { D };

						let Ajk = simd.splat(&mul_real(conj(A_0[(j, k)]), -D));

						let A0k = simd.read(Ak, i0);
						let A1k = simd.read(Ak, i1);
						let Aik = simd.read(Ak, idx_tail);
						A0j = simd.mul_add(Ajk, A0k, A0j);
						A1j = simd.mul_add(Ajk, A1k, A1j);
						Aij = simd.mul_add(Ajk, Aik, Aij);
					}
					simd.write(Aj.rb_mut(), i0, A0j);
					simd.write(Aj.rb_mut(), i1, A1j);
					simd.write(Aj.rb_mut(), idx_tail, Aij);
				},
				(Some(i0), Some(i1), Some(i2)) => {
					let mut A0j = simd.read(Aj.rb(), i0);
					let mut A1j = simd.read(Aj.rb(), i1);
					let mut A2j = simd.read(Aj.rb(), i2);
					let mut Aij = simd.read(Aj.rb(), idx_tail);

					for k in LEFT.indices() {
						let Ak = A10.col(k);

						let D = real(D[k]);
						let D = if is_llt { one() } else { D };

						let Ajk = simd.splat(&mul_real(conj(A_0[(j, k)]), -D));

						let A0k = simd.read(Ak, i0);
						let A1k = simd.read(Ak, i1);
						let A2k = simd.read(Ak, i2);
						let Aik = simd.read(Ak, idx_tail);
						A0j = simd.mul_add(Ajk, A0k, A0j);
						A1j = simd.mul_add(Ajk, A1k, A1j);
						A2j = simd.mul_add(Ajk, A2k, A2j);
						Aij = simd.mul_add(Ajk, Aik, Aij);
					}
					simd.write(Aj.rb_mut(), i0, A0j);
					simd.write(Aj.rb_mut(), i1, A1j);
					simd.write(Aj.rb_mut(), i2, A2j);
					simd.write(Aj.rb_mut(), idx_tail, Aij);
				},
				_ => {
					unreachable!();
				},
			}
		}

		let D = D.rb_mut().at_mut(j);

		if *j >= *start {
			let j_row = TAIL.idx(*j - *start);

			let mut diag = real(Aj[j_row]);

			if regularize {
				let sign = if is_llt { 1 } else { if let Some(signs) = signs { signs[j] } else { 0 } };

				let small_or_negative = diag <= eps;
				let minus_small_or_positive = diag >= -eps;

				if sign == 1 && small_or_negative {
					diag = copy(delta);
					count += 1;
				} else if sign == -1i8 && minus_small_or_positive {
					diag = neg(delta);
				} else {
					if small_or_negative && minus_small_or_positive {
						if diag < zero() {
							diag = neg(delta);
						} else {
							diag = copy(delta);
						}
					}
				}
			}

			let j = j;
			let diag = if is_llt {
				if !(diag > zero()) {
					*D = from_real(diag);
					return Err(*j);
				}
				sqrt(diag)
			} else {
				copy(diag)
			};

			*D = from_real(diag);

			if diag == zero() || !is_finite(diag) {
				return Err(*j);
			}
		}

		let diag = real(*D);

		{
			let mut Aj = Aj.rb_mut();
			let inv = simd.splat_real(&recip(diag));

			for i in indices.clone() {
				let mut Aij = simd.read(Aj.rb(), i);
				Aij = simd.mul_real(Aij, inv);
				simd.write(Aj.rb_mut(), i, Aij);
			}
			{
				let mut Aij = simd.read(Aj.rb(), idx_tail);
				Aij = simd.mul_real(Aij, inv);
				simd.write(Aj.rb_mut(), idx_tail, Aij);
			}
		}
	}

	Ok(count)
}

#[inline(always)]
#[math]
fn simd_cholesky_matrix<T: ComplexField, S: Simd>(
	simd: T::SimdCtx<S>,
	A: MatMut<'_, T, usize, usize, ContiguousFwd>,
	D: RowMut<'_, T, usize>,

	is_llt: bool,
	regularize: bool,
	eps: T::Real,
	delta: T::Real,
	signs: Option<&[i8]>,
) -> Result<usize, usize> {
	let N = A.ncols();

	let blocksize = 4 * (core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>());

	let mut A = A;
	let mut D = D;

	let mut count = 0;

	let mut j = 0;
	while j < N {
		let blocksize = Ord::min(blocksize, N - j);
		let j_next = j + blocksize;

		with_dim!(HEAD, j_next);
		let A = A.rb_mut().submatrix_mut(0, 0, HEAD, HEAD);
		let D = D.rb_mut().subcols_mut(0, HEAD);

		let signs = signs.map(|signs| Array::from_ref(&signs[..*HEAD], HEAD));

		count += simd_cholesky_row_batch(simd, A, D, HEAD.idx_inc(j), is_llt, regularize, eps.clone(), delta.clone(), signs)?;
		j += blocksize;
	}

	Ok(count)
}

fn simd_cholesky<T: ComplexField>(
	A: MatMut<'_, T>,
	D: RowMut<'_, T>,
	is_llt: bool,
	regularize: bool,
	eps: T::Real,
	delta: T::Real,
	signs: Option<&[i8]>,
) -> Result<usize, usize> {
	struct Impl<'a, T: ComplexField> {
		A: MatMut<'a, T, usize, usize, ContiguousFwd>,
		D: RowMut<'a, T>,
		is_llt: bool,
		regularize: bool,
		eps: T::Real,
		delta: T::Real,
		signs: Option<&'a [i8]>,
	}

	impl<'a, T: ComplexField> pulp::WithSimd for Impl<'a, T> {
		type Output = Result<usize, usize>;

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let Self {
				A,
				D,
				is_llt,
				regularize,
				eps,
				delta,
				signs,
			} = self;
			let simd = T::simd_ctx(simd);
			if A.nrows() > 0 {
				simd_cholesky_matrix(simd, A, D, is_llt, regularize, eps, delta, signs)
			} else {
				Ok(0)
			}
		}
	}

	let mut A = A;
	if try_const! { T::SIMD_CAPABILITIES.is_simd() } {
		if let Some(A) = A.rb_mut().try_as_col_major_mut() {
			dispatch!(
				Impl {
					A,
					D,
					is_llt,
					regularize,
					eps,
					delta,
					signs,
				},
				Impl,
				T
			)
		} else {
			cholesky_fallback(A, D, is_llt, regularize, eps.clone(), delta.clone(), signs)
		}
	} else {
		cholesky_fallback(A, D, is_llt, regularize, eps.clone(), delta.clone(), signs)
	}
}

#[math]
fn cholesky_fallback<T: ComplexField>(
	A: MatMut<'_, T>,
	D: RowMut<'_, T>,
	is_llt: bool,
	regularize: bool,
	eps: T::Real,
	delta: T::Real,
	signs: Option<&[i8]>,
) -> Result<usize, usize> {
	let n = A.nrows();
	let mut count = 0;
	let mut A = A;
	let mut D = D;

	for j in 0..n {
		for i in j..n {
			let mut sum = zero();
			for k in 0..j {
				let D = real(D[k]);
				let D = if is_llt { one() } else { D };

				sum = sum + mul_real(conj(A[(j, k)]) * A[(i, k)], D);
			}
			A[(i, j)] = A[(i, j)] - sum;
		}

		let D = D.rb_mut().at_mut(j);
		let mut diag = real(A[(j, j)]);

		if regularize {
			let sign = if is_llt { 1 } else { if let Some(signs) = signs { signs[j] } else { 0 } };

			let small_or_negative = diag <= eps;
			let minus_small_or_positive = diag >= -eps;

			if sign == 1 && small_or_negative {
				diag = copy(delta);
				count += 1;
			} else if sign == -1i8 && minus_small_or_positive {
				diag = neg(delta);
			} else {
				if small_or_negative && minus_small_or_positive {
					if diag < zero() {
						diag = neg(delta);
					} else {
						diag = copy(delta);
					}
				}
			}
		}

		let diag = if is_llt {
			if !(diag > zero()) {
				*D = from_real(diag);
				return Err(j);
			}
			sqrt(diag)
		} else {
			copy(diag)
		};
		*D = from_real(diag);

		if diag == zero() || !is_finite(diag) {
			return Err(j);
		}

		let inv = recip(diag);

		for i in j..n {
			A[(i, j)] = mul_real(A[(i, j)], inv);
		}
	}

	Ok(count)
}

#[math]
pub(crate) fn cholesky_recursion<T: ComplexField>(
	A: MatMut<'_, T>,
	D: RowMut<'_, T>,

	recursion_threshold: usize,
	blocksize: usize,
	is_llt: bool,
	regularize: bool,
	eps: &T::Real,
	delta: &T::Real,
	signs: Option<&[i8]>,
	par: Par,
) -> Result<usize, usize> {
	let n = A.ncols();
	if n <= recursion_threshold {
		simd_cholesky(A, D, is_llt, regularize, eps.clone(), delta.clone(), signs)
	} else {
		let mut count = 0;
		let blocksize = Ord::min(n.next_power_of_two() / 2, blocksize);
		let mut A = A;
		let mut D = D;

		let mut j = 0;
		while j < n {
			let blocksize = Ord::min(blocksize, n - j);

			let (mut A00, A01, mut A10, mut A11) = A.rb_mut().get_mut(j.., j..).split_at_mut(blocksize, blocksize);

			let mut D0 = D.rb_mut().subcols_mut(j, blocksize);

			let mut L10xD0 = A01.transpose_mut();

			let signs = signs.map(|signs| &signs[j..][..blocksize]);

			match cholesky_recursion(
				A00.rb_mut(),
				D0.rb_mut(),
				recursion_threshold,
				blocksize,
				is_llt,
				regularize,
				eps,
				delta,
				signs,
				par,
			) {
				Ok(local_count) => count += local_count,
				Err(fail_idx) => return Err(j + fail_idx),
			}
			let A00 = A00.rb();

			if is_llt {
				linalg::triangular_solve::solve_lower_triangular_in_place(A00.conjugate(), A10.rb_mut().transpose_mut(), par)
			} else {
				linalg::triangular_solve::solve_unit_lower_triangular_in_place(A00.conjugate(), A10.rb_mut().transpose_mut(), par)
			}
			let mut A10 = A10.rb_mut();

			if is_llt {
				linalg::matmul::triangular::matmul(
					A11.rb_mut(),
					BlockStructure::TriangularLower,
					Accum::Add,
					A10.rb(),
					BlockStructure::Rectangular,
					A10.rb().adjoint(),
					BlockStructure::Rectangular,
					-one::<T>(),
					par,
				);
			} else {
				if has_spicy_matmul::<T>() {
					for k in 0..blocksize {
						let d = real(D0[k]);
						let d = recip(d);

						for i in j + blocksize..n {
							let i = i - (j + blocksize);
							A10[(i, k)] = mul_real(A10[(i, k)], d);
						}
					}
					spicy_matmul::<usize, T>(
						A11.rb_mut(),
						BlockStructure::TriangularLower,
						None,
						None,
						Accum::Add,
						A10.rb(),
						Conj::No,
						A10.rb().transpose(),
						Conj::Yes,
						Some(D0.rb().transpose().as_diagonal()),
						-one::<T>(),
						par,
						MemStack::new(&mut []),
					);
				} else {
					for k in 0..blocksize {
						let d = real(D0[k]);
						let d = recip(d);

						for i in j + blocksize..n {
							let i = i - (j + blocksize);
							let a = copy(A10[(i, k)]);
							A10[(i, k)] = mul_real(A10[(i, k)], d);
							L10xD0[(i, k)] = a;
						}
					}
					linalg::matmul::triangular::matmul(
						A11.rb_mut(),
						BlockStructure::TriangularLower,
						Accum::Add,
						A10,
						BlockStructure::Rectangular,
						L10xD0.adjoint(),
						BlockStructure::Rectangular,
						-one::<T>(),
						par,
					);
				}
			};

			j += blocksize;
		}

		Ok(count)
	}
}

/// dynamic $LDL^\top$ regularization.
/// values below `epsilon` in absolute value, or with the wrong sign are set to `delta` with
/// their corrected sign.
#[derive(Copy, Clone, Debug)]
pub struct LdltRegularization<'a, T> {
	/// expected signs for the diagonal at each step of the decomposition.
	pub dynamic_regularization_signs: Option<&'a [i8]>,
	/// regularized value.
	pub dynamic_regularization_delta: T,
	/// regularization threshold.
	pub dynamic_regularization_epsilon: T,
}

/// info about the result of the $LDL^\top$ factorization.
#[derive(Copy, Clone, Debug)]
pub struct LdltInfo {
	/// number of pivots whose value or sign had to be corrected.
	pub dynamic_regularization_count: usize,
}

/// error in the $LDL^\top$ factorization.
#[derive(Copy, Clone, Debug)]
pub enum LdltError {
	ZeroPivot { index: usize },
}

impl core::fmt::Display for LdltError {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Debug::fmt(self, f)
	}
}
impl core::error::Error for LdltError {}

impl<T: RealField> Default for LdltRegularization<'_, T> {
	fn default() -> Self {
		Self {
			dynamic_regularization_signs: None,
			dynamic_regularization_delta: zero(),
			dynamic_regularization_epsilon: zero(),
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub struct LdltParams {
	pub recursion_threshold: usize,
	pub blocksize: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for LdltParams {
	#[inline]
	fn auto() -> Self {
		Self {
			recursion_threshold: 64,
			blocksize: 128,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

#[inline]
pub fn cholesky_in_place_scratch<T: ComplexField>(dim: usize, par: Par, params: Spec<LdltParams, T>) -> StackReq {
	_ = par;
	_ = params;
	temp_mat_scratch::<T>(dim, 1)
}

#[math]
pub fn cholesky_in_place<T: ComplexField>(
	A: MatMut<'_, T>,
	regularization: LdltRegularization<'_, T::Real>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<LdltParams, T>,
) -> Result<LdltInfo, LdltError> {
	let params = params.config;

	let n = A.nrows();
	let mut D = unsafe { temp_mat_uninit(n, 1, stack).0 };
	let D = D.as_mat_mut();
	let mut D = D.col_mut(0).transpose_mut();
	let mut A = A;

	let ret = match cholesky_recursion(
		A.rb_mut(),
		D.rb_mut(),
		params.recursion_threshold,
		params.blocksize,
		false,
		regularization.dynamic_regularization_delta > zero() && regularization.dynamic_regularization_epsilon > zero(),
		&regularization.dynamic_regularization_epsilon,
		&regularization.dynamic_regularization_delta,
		regularization.dynamic_regularization_signs.map(|signs| signs),
		par,
	) {
		Ok(count) => Ok(LdltInfo {
			dynamic_regularization_count: count,
		}),
		Err(index) => Err(LdltError::ZeroPivot { index }),
	};
	let init = if let Err(LdltError::ZeroPivot { index }) = ret { index + 1 } else { n };

	for i in 0..init {
		A[(i, i)] = copy(D[i]);
	}

	ret
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, Row, assert, c64};

	#[test]
	fn test_simd_cholesky() {
		let rng = &mut StdRng::seed_from_u64(0);

		type T = c64;

		for n in 0..=64 {
			for f in [cholesky_fallback::<T>, simd_cholesky::<T>] {
				for llt in [true, false] {
					let approx_eq = CwiseMat(ApproxEq {
						abs_tol: 1e-12,
						rel_tol: 1e-12,
					});

					let A = CwiseMatDistribution {
						nrows: n,
						ncols: n,
						dist: ComplexDistribution::new(StandardNormal, StandardNormal),
					}
					.rand::<Mat<c64>>(rng);

					let A = &A * &A.adjoint();
					let A = A.as_ref().as_shape(n, n);

					let mut L = A.cloned();
					let mut L = L.as_mut();
					let mut D = Row::zeros(n);
					let mut D = D.as_mut();

					f(L.rb_mut(), D.rb_mut(), llt, false, 0.0, 0.0, None).unwrap();

					for j in 0..n {
						for i in 0..j {
							L[(i, j)] = c64::ZERO;
						}
					}
					let L = L.rb().as_dyn_stride();

					if llt {
						assert!(L * L.adjoint() ~ A);
					} else {
						assert!(L * D.as_diagonal() * L.adjoint() ~ A);
					};
				}
			}
		}
	}

	#[test]
	fn test_cholesky() {
		let rng = &mut StdRng::seed_from_u64(0);

		for n in [2, 4, 8, 31, 127, 240] {
			for llt in [false, true] {
				let approx_eq = CwiseMat(ApproxEq {
					abs_tol: 1e-12,
					rel_tol: 1e-12,
				});

				let A = CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: ComplexDistribution::new(StandardNormal, StandardNormal),
				}
				.rand::<Mat<c64>>(rng);

				let A = &A * &A.adjoint();
				let A = A.as_ref();

				let mut L = A.cloned();
				let mut L = L.as_mut();
				let mut D = Row::zeros(n);
				let mut D = D.as_mut();

				cholesky_recursion(L.rb_mut(), D.rb_mut(), 32, 32, llt, false, &0.0, &0.0, None, Par::Seq).unwrap();

				for j in 0..n {
					for i in 0..j {
						L[(i, j)] = c64::ZERO;
					}
				}
				let L = L.rb().as_dyn_stride();

				if llt {
					assert!(L * L.adjoint() ~ A);
				} else {
					assert!(L * D.as_diagonal() * L.adjoint() ~ A);
				};
			}
		}
	}
}
