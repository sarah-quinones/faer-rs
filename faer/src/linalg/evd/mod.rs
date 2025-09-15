//! low level implementation of the eigenvalue decomposition of a square diagonalizable matrix.
//!
//! the eigenvalue decomposition of a square matrix $A$ of shape $(n, n)$ is a decomposition into
//! two components $U$, $S$:
//!
//! - $U$ has shape $(n, n)$ and is invertible
//! - $S$ has shape $(n, n)$ and is a diagonal matrix
//! - and finally:
//!
//! $$A = U S U^{-1}$$
//!
//! if $A$ is self-adjoint, then $U$ can be made unitary ($U^{-1} = U^H$), and $S$ is real valued

/// hessenberg decomposition
pub mod hessenberg;
#[doc(hidden)]
pub mod schur;

/// self-adjoint tridiagonalization
pub mod tridiag;
pub(crate) mod tridiag_evd;

use crate::assert;
use crate::internal_prelude::*;
use hessenberg::HessenbergParams;
use linalg::matmul::triangular::BlockStructure;
pub use schur::SchurParams;
use tridiag::TridiagParams;

/// eigendecomposition error
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvdError {
	/// reached max iterations
	NoConvergence,
}

/// schur to eigendecomposition conversion parameters
#[derive(Clone, Copy, Debug)]
pub struct EvdFromSchurParams {
	/// threshold at which the implementation should stop recursing
	pub recursion_threshold: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

/// eigendecomposition tuning parameters
#[derive(Clone, Copy, Debug)]
pub struct EvdParams {
	/// hessenberg parameters
	pub hessenberg: HessenbergParams,
	/// schur from hessenberg conversion parameters
	pub schur: SchurParams,
	/// eigendecomposition from schur conversion parameters
	pub evd_from_schur: EvdFromSchurParams,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

/// self-adjoint eigendecomposition tuning parameters
#[derive(Clone, Copy, Debug)]
pub struct SelfAdjointEvdParams {
	/// tridiagonalization parameters
	pub tridiag: TridiagParams,
	/// threshold at which the implementation should stop recursing
	pub recursion_threshold: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for EvdParams {
	fn auto() -> Self {
		Self {
			hessenberg: auto!(T),
			schur: auto!(T),
			evd_from_schur: auto!(T),
			non_exhaustive: NonExhaustive(()),
		}
	}
}
impl<T: ComplexField> Auto<T> for EvdFromSchurParams {
	fn auto() -> Self {
		Self {
			recursion_threshold: 64,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

impl<T: ComplexField> Auto<T> for SelfAdjointEvdParams {
	fn auto() -> Self {
		Self {
			tridiag: auto!(T),
			recursion_threshold: 128,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

/// whether the eigenvectors should be computed
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComputeEigenvectors {
	/// do not compute eigenvectors
	No,
	/// compute eigenvectors
	Yes,
}

/// computes the layout of the workspace required to compute a self-adjoint matrix's
/// eigendecomposition
#[math]
pub fn self_adjoint_evd_scratch<T: ComplexField>(
	dim: usize,
	compute_u: ComputeEigenvectors,
	par: Par,
	params: Spec<SelfAdjointEvdParams, T>,
) -> StackReq {
	let n = dim;
	let bs = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n, n);

	let prologue = StackReq::all_of(&[
		temp_mat_scratch::<T>(n, n),
		temp_mat_scratch::<T>(bs, n),
		StackReq::any_of(&[tridiag::tridiag_in_place_scratch::<T>(n, par, params.tridiag.into())]),
		temp_mat_scratch::<T::Real>(n, 1).array(2),
	]);
	if compute_u == ComputeEigenvectors::No {
		return prologue;
	}

	StackReq::all_of(&[
		prologue,
		temp_mat_scratch::<T::Real>(n, if try_const! { T::IS_REAL } { 0 } else { n }).array(2),
		StackReq::any_of(&[
			if n < params.recursion_threshold {
				StackReq::empty()
			} else {
				tridiag_evd::divide_and_conquer_scratch::<T>(n, par)
			},
			temp_mat_scratch::<T>(n, 1),
			linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(n - 1, bs, n),
		]),
	])
}

/// computes the matrix $A$'s eigendecomposition, assuming it is tridiagonal and self-adjoint
///
/// the eigenvalues are stored in $S$, and the eigenvectors in $U$ such that the eigenvalues are
/// sorted in nondecreasing order
#[math]
pub fn tridiagonal_self_adjoint_evd<T: ComplexField>(
	diag: DiagRef<'_, T>,
	subdiag: DiagRef<'_, T>,
	s: DiagMut<'_, T>,
	u: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<SelfAdjointEvdParams, T>,
) -> Result<(), EvdError> {
	let n = diag.dim();
	let (mut real_diag, stack) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
	let (mut real_offdiag, stack) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };

	let mut real_diag = real_diag.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();
	let mut real_offdiag = real_offdiag.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();

	for i in 0..n {
		real_diag[i] = real(diag[i]);

		if i + 1 < n {
			if try_const! { T::IS_REAL } {
				real_offdiag[i] = real(subdiag[i]);
			} else {
				real_offdiag[i] = abs(subdiag[i]);
			}
		} else {
			real_offdiag[i] = zero();
		}
	}

	let mut s = s;
	let mut u = match u {
		Some(u) => u,
		None => {
			tridiag_evd::qr_algorithm(real_diag.rb_mut(), real_offdiag.rb_mut(), None)?;
			for i in 0..n {
				s[i] = from_real(real_diag[i]);
			}

			return Ok(());
		},
	};

	let (mut u_real, stack) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, if T::IS_REAL { 0 } else { n }, stack) };
	let mut u_real = u_real.as_mat_mut();
	let mut u_evd = if try_const! { T::IS_REAL } {
		unsafe { core::mem::transmute(u.rb_mut()) }
	} else {
		u_real.rb_mut()
	};

	if n < params.recursion_threshold {
		tridiag_evd::qr_algorithm(real_diag.rb_mut(), real_offdiag.rb_mut(), Some(u_evd.rb_mut()))?;
	} else {
		tridiag_evd::divide_and_conquer::<T::Real>(
			real_diag.rb_mut(),
			real_offdiag.rb_mut(),
			u_evd.rb_mut(),
			par,
			stack,
			params.recursion_threshold,
		)?;
	}

	if try_const! { !T::IS_REAL } {
		let normalized = |x: T| {
			if x == zero() { one() } else { mul_real(x, recip(abs(x))) }
		};

		let (mut scale, _) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
		let mut scale = scale.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();

		let mut x = one::<T>();
		scale[0] = one();

		for i in 1..n {
			x = normalized(subdiag[i - 1] * x);
			scale[i] = copy(x);
		}
		for j in 0..n {
			z!(u.rb_mut().col_mut(j), u_real.rb().col(j), scale.rb()).for_each(|uz!(u, real, scale)| {
				*u = mul_real(*scale, *real);
			});
		}
	}

	for i in 0..n {
		s[i] = from_real(real_diag[i]);
	}

	Ok(())
}

/// computes the matrix $A$'s eigendecomposition, assuming it is self-adjoint
///
/// the eigenvalues are stored in $S$, and the eigenvectors in $U$ such that the eigenvalues are
/// sorted in nondecreasing order
///
/// only the lower triangular half of $A$ is accessed
#[math]
pub fn self_adjoint_evd<T: ComplexField>(
	A: MatRef<'_, T>,
	s: DiagMut<'_, T>,
	u: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<SelfAdjointEvdParams, T>,
) -> Result<(), EvdError> {
	let n = A.nrows();
	assert!(all(A.nrows() == A.ncols(), s.dim() == n));
	if let Some(u) = u.rb() {
		assert!(all(u.nrows() == n, u.ncols() == n));
	}
	let s = s.column_vector_mut();

	if n == 0 {
		return Ok(());
	}

	#[cfg(feature = "perf-warn")]
	if let Some(matrix) = u.rb() {
		if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(QR_WARN) {
			if matrix.col_stride().unsigned_abs() == 1 {
				log::warn!(target: "faer_perf", "EVD prefers column-major eigenvector matrix. Found row-major matrix.");
			} else {
				log::warn!(target: "faer_perf", "EVD prefers column-major eigenvector matrix. Found matrix with generic strides.");
			}
		}
	}

	let (mut trid, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut trid = trid.as_mat_mut();

	trid.copy_from_triangular_lower(A);

	let bs = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n, n);
	let (mut householder, stack) = unsafe { temp_mat_uninit::<T, _, _>(bs, n - 1, stack) };
	let mut householder = householder.as_mat_mut();

	{
		tridiag::tridiag_in_place(trid.rb_mut(), householder.rb_mut(), par, stack, params.tridiag.into());
	}

	let trid = trid.rb();

	let (mut diag, stack) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
	let (mut offdiag, stack) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };

	let mut diag = diag.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();
	let mut offdiag = offdiag.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();

	for i in 0..n {
		diag[i] = real(trid[(i, i)]);

		if i + 1 < n {
			if try_const! { T::IS_REAL } {
				offdiag[i] = real(trid[(i + 1, i)]);
			} else {
				offdiag[i] = abs(trid[(i + 1, i)]);
			}
		} else {
			offdiag[i] = zero();
		}
	}

	let mut s = s;
	let mut u = match u {
		Some(u) => u,
		None => {
			tridiag_evd::qr_algorithm(diag.rb_mut(), offdiag.rb_mut(), None)?;
			for i in 0..n {
				s[i] = from_real(diag[i]);
			}

			return Ok(());
		},
	};

	let (mut u_real, stack) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, if T::IS_REAL { 0 } else { n }, stack) };
	let mut u_real = u_real.as_mat_mut();
	let mut u_evd = if try_const! { T::IS_REAL } {
		unsafe { core::mem::transmute(u.rb_mut()) }
	} else {
		u_real.rb_mut()
	};

	if n < params.recursion_threshold {
		tridiag_evd::qr_algorithm(diag.rb_mut(), offdiag.rb_mut(), Some(u_evd.rb_mut()))?;
	} else {
		tridiag_evd::divide_and_conquer::<T::Real>(diag.rb_mut(), offdiag.rb_mut(), u_evd.rb_mut(), par, stack, params.recursion_threshold)?;
	}

	if try_const! { !T::IS_REAL } {
		let normalized = |x: T| {
			if x == zero() { one() } else { mul_real(x, recip(abs(x))) }
		};

		let (mut scale, _) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
		let mut scale = scale.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();

		let mut x = one::<T>();
		scale[0] = one();

		for i in 1..n {
			x = normalized(trid[(i, i - 1)] * x);
			scale[i] = copy(x);
		}
		for j in 0..n {
			z!(u.rb_mut().col_mut(j), u_real.rb().col(j), scale.rb()).for_each(|uz!(u, real, scale)| {
				*u = mul_real(*scale, *real);
			});
		}
	}

	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
		trid.submatrix(1, 0, n - 1, n - 1),
		householder.rb(),
		Conj::No,
		u.rb_mut().subrows_mut(1, n - 1),
		par,
		stack,
	);

	for i in 0..n {
		s[i] = from_real(diag[i]);
	}

	Ok(())
}

/// computes the layout of the workspace required to compute a self-adjoint matrix's
/// pseudoinverse, given the eigendecomposition
pub fn pseudoinverse_from_self_adjoint_evd_scratch<T: ComplexField>(dim: usize, par: Par) -> StackReq {
	_ = par;
	temp_mat_scratch::<T>(dim, dim).array(2)
}

/// computes a self-adjoint matrix's pseudoinverse, given the eigendecomposition factors $S$ and $U$
#[math]
#[track_caller]
pub fn pseudoinverse_from_self_adjoint_evd<T: ComplexField>(
	pinv: MatMut<'_, T>,
	s: DiagRef<'_, T>,
	u: MatRef<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	pseudoinverse_from_self_adjoint_evd_with_tolerance(pinv, s, u, zero(), eps::<T::Real>() * from_f64::<T::Real>(u.ncols() as f64), par, stack);
}

/// computes a self-adjoint matrix's pseudoinverse, given the eigendecomposition factors $S$ and
/// $U$, and tolerance parameters for determining zero eigenvalues
#[math]
#[track_caller]
pub fn pseudoinverse_from_self_adjoint_evd_with_tolerance<T: ComplexField>(
	pinv: MatMut<'_, T>,
	s: DiagRef<'_, T>,
	u: MatRef<'_, T>,
	abs_tol: T::Real,
	rel_tol: T::Real,
	par: Par,
	stack: &mut MemStack,
) {
	let s = s.column_vector();
	let mut pinv = pinv;
	let n = u.ncols();

	assert!(all(u.nrows() == n, u.ncols() == n, s.nrows() == n));

	let smax = s.norm_max();
	let tol = max(abs_tol, rel_tol * smax);

	let (mut u_trunc, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let (mut up_trunc, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };

	let mut u_trunc = u_trunc.as_mat_mut();
	let mut up_trunc = up_trunc.as_mat_mut();
	let mut len = 0;

	for j in 0..n {
		let x = absmax(s[j]);
		if x > tol {
			let p = recip(real(s[j]));
			u_trunc.rb_mut().col_mut(len).copy_from(u.col(j));
			z!(up_trunc.rb_mut().col_mut(len), u.col(j)).for_each(|uz!(dst, src)| *dst = mul_real(*src, p));

			len += 1;
		}
	}

	let u_trunc = u_trunc.get(.., ..len);
	let up_trunc = up_trunc.get(.., ..len);

	linalg::matmul::triangular::matmul(
		pinv.rb_mut(),
		BlockStructure::TriangularLower,
		Accum::Replace,
		up_trunc.rb(),
		BlockStructure::Rectangular,
		u_trunc.rb().adjoint(),
		BlockStructure::Rectangular,
		one(),
		par,
	);

	for j in 0..n {
		for i in 0..j {
			pinv[(i, j)] = conj(pinv[(j, i)]);
		}
	}
}

#[math]
fn dot2x1<T: RealField>(lhs0: RowRef<'_, T>, lhs1: RowRef<'_, T>, rhs: ColRef<'_, T>) -> (T, T) {
	let n = rhs.nrows();
	assert!(all(lhs0.ncols() == n, lhs1.ncols() == n));

	let mut acc00 = zero::<T>();
	let mut acc01 = zero::<T>();
	let mut acc10 = zero::<T>();
	let mut acc11 = zero::<T>();

	let n2 = n / 2 * 2;

	let mut i = 0;
	while i < n2 {
		acc00 = acc00 + lhs0[i] * rhs[i];
		acc10 = acc10 + lhs1[i] * rhs[i];

		acc01 = acc01 + lhs0[i + 1] * rhs[i + 1];
		acc11 = acc11 + lhs1[i + 1] * rhs[i + 1];

		i += 2;
	}
	while i < n {
		acc00 = acc00 + lhs0[i] * rhs[i];
		acc10 = acc10 + lhs1[i] * rhs[i];
		i += 1;
	}

	(acc00 + acc01, acc10 + acc11)
}

#[math]
fn dot2x2<T: RealField>(lhs0: RowRef<'_, T>, lhs1: RowRef<'_, T>, rhs0: ColRef<'_, T>, rhs1: ColRef<'_, T>) -> (T, T, T, T) {
	let n = rhs0.nrows();
	assert!(all(lhs0.ncols() == n, lhs1.ncols() == n));

	let mut acc0 = zero::<T>();
	let mut acc1 = zero::<T>();
	let mut acc2 = zero::<T>();
	let mut acc3 = zero::<T>();

	let mut i = 0;
	while i < n {
		acc0 = acc0 + lhs0[i] * rhs0[i];
		acc1 = acc1 + lhs1[i] * rhs0[i];
		acc2 = acc2 + lhs0[i] * rhs1[i];
		acc3 = acc3 + lhs1[i] * rhs1[i];

		i += 1;
	}

	(acc0, acc1, acc2, acc3)
}

#[math]
pub(crate) fn evd_from_real_schur_imp<T: RealField>(A: MatRef<'_, T>, V: MatMut<'_, T>, par: Par, params: EvdFromSchurParams) {
	let one = one::<T>;
	let zero = zero::<T>;

	let mut V = V;
	let n = A.nrows();

	let mut norm = zero();

	for j in 0..n {
		for i in 0..Ord::min(j + 2, n) {
			norm = norm + abs1(A[(i, j)]);
		}
	}

	let mut k = n;
	loop {
		if k == 0 {
			break;
		}
		k -= 1;

		if k == 0 || A[(k, k - 1)] == zero() {
			// real eigenvalue
			let p = copy(A[(k, k)]);

			// solve (A[:k, :k] - p I) X = -A[:k, k]
			// form rhs
			V[(k, k)] = one();
			for i in 0..k {
				V[(i, k)] = -A[(i, k)];
			}

			solve_real_shifted_upper_quasi_triangular_system(
				A.submatrix(0, 0, k, k),
				p,
				V.rb_mut().subrows_mut(0, k).col_mut(k),
				copy(norm),
				par,
				params,
			);
		} else {
			// complex eigenvalue pair
			let p = copy(A[(k, k)]);
			let q = sqrt(abs(A[(k, k - 1)])) * sqrt(abs(A[(k - 1, k)]));

			// solve (A[:k, :k] - (p + iq) I) X = rhs
			// form rhs
			if abs(A[(k - 1, k)]) >= abs(A[(k, k - 1)]) {
				V[(k - 1, k - 1)] = one();
				V[(k, k)] = q / A[(k - 1, k)];
			} else {
				V[(k - 1, k - 1)] = -q / A[(k, k - 1)];
				V[(k, k)] = one();
			}

			V[(k - 1, k)] = zero();
			V[(k, k - 1)] = zero();

			for i in 0..k - 1 {
				V[(i, k - 1)] = -V[(k - 1, k - 1)] * A[(i, k - 1)];
				V[(i, k)] = -V[(k, k)] * A[(i, k)];
			}

			solve_cplx_shifted_upper_quasi_triangular_system(
				A.submatrix(0, 0, k - 1, k - 1),
				p,
				q,
				V.rb_mut().submatrix_mut(0, k - 1, k - 1, 2),
				copy(norm),
				par,
				params,
			);

			k -= 1;
		}
	}
}

#[math]
pub(crate) fn evd_from_cplx_schur_imp<T: ComplexField>(A: MatRef<'_, T>, conj_A: Conj, V: MatMut<'_, T>, par: Par, params: EvdFromSchurParams) {
	let one = one::<T>;

	let mut V = V;
	let n = A.nrows();

	let mut norm = zero::<T::Real>();

	for j in 0..n {
		for i in 0..j + 1 {
			norm = norm + abs1(A[(i, j)]);
		}
	}

	let mut k = n;
	loop {
		if k == 0 {
			break;
		}
		k -= 1;

		let p = if conj_A == Conj::Yes { conj(A[(k, k)]) } else { copy(A[(k, k)]) };

		// solve (A[:k, :k] - p I) X = -A[:k, k]
		// form rhs
		V[(k, k)] = one();

		if conj_A == Conj::Yes {
			for i in 0..k {
				V[(i, k)] = -conj(A[(i, k)]);
			}
		} else {
			for i in 0..k {
				V[(i, k)] = -A[(i, k)];
			}
		}

		solve_shifted_upper_triangular_system(
			A.submatrix(0, 0, k, k),
			conj_A,
			p,
			V.rb_mut().subrows_mut(0, k).col_mut(k),
			copy(norm),
			par,
			params,
		);
	}
}

#[math]
fn solve_real_shifted_upper_quasi_triangular_system<T: RealField>(
	A: MatRef<'_, T>,
	p: T,
	x: ColMut<'_, T>,
	norm: T,
	par: Par,
	params: EvdFromSchurParams,
) {
	let n = A.nrows();

	let one = one::<T>;
	let zero = zero::<T>;

	let eps = eps::<T>();

	let mut x = x;

	if par.degree() == 0 || n < params.recursion_threshold {
		let mut i = n;
		loop {
			if i == 0 {
				break;
			}
			i -= 1;

			if i == 0 || A[(i, i - 1)] == zero() {
				// 1x1 block
				let dot = linalg::matmul::dot::inner_prod(A.row(i).subcols(i + 1, n - i - 1), Conj::No, x.rb().subrows(i + 1, n - i - 1), Conj::No);

				x[i] = x[i] - dot;

				let mut z = A[(i, i)] - p;

				if abs(z) < eps * norm {
					z = eps * norm;
				}

				if x[i] != zero() {
					x[i] = x[i] / z;
				}
			} else {
				let (dot0, dot1) = dot2x1(
					A.row(i - 1).subcols(i + 1, n - i - 1),
					A.row(i).subcols(i + 1, n - i - 1),
					x.rb().subrows(i + 1, n - i - 1),
				);

				x[i - 1] = x[i - 1] - dot0;
				x[i] = x[i] - dot1;

				// solve
				// [a b] [x0]   [r0]
				// [c a]×[x1] = [r1]
				//
				//  [x0]   [a  -b] [r0]
				//  [x1] = [-c  a]×[r1] / det
				let a = A[(i, i)] - p;
				let b = copy(A[(i - 1, i)]);
				let c = copy(A[(i, i - 1)]);

				let r0 = copy(x[i - 1]);
				let r1 = copy(x[i]);

				let inv_det = recip(abs2(a) - b * c);

				x[i - 1] = (a * r0 - b * r1) * inv_det;
				x[i] = (a * r1 - c * r0) * inv_det;

				i -= 1;
			}
		}
	} else {
		let mut mid = n / 2;
		if A[(mid, mid - 1)] != zero() {
			mid -= 1;
		}

		let (A00, A01, _, A11) = A.split_at(mid, mid);
		let (mut x0, mut x1) = x.rb_mut().split_at_row_mut(mid);

		solve_real_shifted_upper_quasi_triangular_system(A11, copy(p), x1.rb_mut(), copy(norm), par, params);

		linalg::matmul::matmul(x0.rb_mut().as_mat_mut(), Accum::Add, A01, x1.rb().as_mat(), -one(), par);

		solve_real_shifted_upper_quasi_triangular_system(A00, p, x0.rb_mut(), norm, par, params);
	}
}

#[math]
fn solve_cplx_shifted_upper_quasi_triangular_system<T: RealField>(
	A: MatRef<'_, T>,
	p: T,
	q: T,
	x: MatMut<'_, T>,
	norm: T,
	par: Par,
	params: EvdFromSchurParams,
) {
	let n = A.nrows();

	let one = one::<T>;
	let zero = zero::<T>;

	let eps = eps::<T>();

	let mut x = x;

	if par.degree() == 0 || n < params.recursion_threshold {
		let mut i = n;
		loop {
			if i == 0 {
				break;
			}
			i -= 1;

			if i == 0 || A[(i, i - 1)] == zero() {
				// 1x1 block
				let (re, im) = dot2x1(
					x.rb().subrows(i + 1, n - i - 1).col(0).transpose(),
					x.rb().subrows(i + 1, n - i - 1).col(1).transpose(),
					A.row(i).subcols(i + 1, n - i - 1).transpose(),
				);

				x[(i, 0)] = x[(i, 0)] - re;
				x[(i, 1)] = x[(i, 1)] - im;

				let mut z_re = A[(i, i)] - p;
				let mut z_im = -q;
				let mut z = hypot(z_re, z_im);

				if z < eps * norm {
					z_re = eps * norm;
					z_im = zero();
					z = copy(z_re);
				}

				let z_re = (z_re / z) / z;
				let z_im = (-z_im / z) / z;

				let x_re = copy(x[(i, 0)]);
				let x_im = copy(x[(i, 1)]);

				x[(i, 0)] = x_re * z_re - x_im * z_im;
				x[(i, 1)] = x_re * z_im + x_im * z_re;
			} else {
				let (re0, re1, im0, im1) = dot2x2(
					A.row(i - 1).subcols(i + 1, n - i - 1),
					A.row(i).subcols(i + 1, n - i - 1),
					x.rb().col(0).subrows(i + 1, n - i - 1),
					x.rb().col(1).subrows(i + 1, n - i - 1),
				);

				x[(i - 1, 0)] = x[(i - 1, 0)] - re0;
				x[(i - 1, 1)] = x[(i - 1, 1)] - im0;
				x[(i, 0)] = x[(i, 0)] - re1;
				x[(i, 1)] = x[(i, 1)] - im1;

				let a_re = A[(i, i)] - p;
				let a_im = -q;
				let b = copy(A[(i - 1, i)]);
				let c = copy(A[(i, i - 1)]);

				let r0_re = copy(x[(i - 1, 0)]);
				let r0_im = copy(x[(i - 1, 1)]);
				let r1_re = copy(x[(i, 0)]);
				let r1_im = copy(x[(i, 1)]);

				let mut z_re = abs2(a_re) - abs2(a_im) - b * c;
				let mut z_im = mul_pow2(a_re * a_im, from_f64::<T>(2.0));
				let mut z = hypot(z_re, z_im);

				if z < eps * norm {
					z_re = eps * norm;
					z_im = zero();
					z = copy(z_re);
				}

				let z_re = (z_re / z) / z;
				let z_im = (-z_im / z) / z;

				let x0_re = (a_re * r0_re - a_im * r0_im) - b * r1_re;
				let x0_im = (a_re * r0_im + a_im * r0_re) - b * r1_im;

				let x1_re = (a_re * r1_re - a_im * r1_im) - c * r0_re;
				let x1_im = (a_re * r1_im + a_im * r1_re) - c * r0_im;

				x[(i - 1, 0)] = x0_re * z_re - x0_im * z_im;
				x[(i - 1, 1)] = x0_re * z_im + x0_im * z_re;

				x[(i, 0)] = x1_re * z_re - x1_im * z_im;
				x[(i, 1)] = x1_re * z_im + x1_im * z_re;

				i -= 1;
			}
		}
	} else {
		let mut mid = n / 2;
		if A[(mid, mid - 1)] != zero() {
			mid -= 1;
		}

		let (A00, A01, _, A11) = A.split_at(mid, mid);
		let (mut x0, mut x1) = x.rb_mut().split_at_row_mut(mid);

		solve_cplx_shifted_upper_quasi_triangular_system(A11, copy(p), copy(q), x1.rb_mut(), copy(norm), par, params);

		linalg::matmul::matmul(x0.rb_mut(), Accum::Add, A01, x1.rb(), -one(), par);

		solve_cplx_shifted_upper_quasi_triangular_system(A00, p, q, x0.rb_mut(), norm, par, params);
	}
}

#[math]
fn solve_shifted_upper_triangular_system<T: ComplexField>(
	A: MatRef<'_, T>,
	conj_A: Conj,
	p: T,
	x: ColMut<'_, T>,
	norm: T::Real,
	par: Par,
	params: EvdFromSchurParams,
) {
	let n = A.nrows();

	let one = one::<T>;
	let zero = zero::<T>;

	let eps = eps::<T::Real>();

	let mut x = x;

	if par.degree() == 0 || n < params.recursion_threshold {
		let mut i = n;
		loop {
			if i == 0 {
				break;
			}
			i -= 1;

			// 1x1 block
			let dot = linalg::matmul::dot::inner_prod(A.row(i).subcols(i + 1, n - i - 1), conj_A, x.rb().subrows(i + 1, n - i - 1), Conj::No);

			x[i] = x[i] - dot;

			let mut z = if conj_A == Conj::Yes { conj(A[(i, i)]) } else { copy(A[(i, i)]) } - p;

			if abs(z) < eps * norm {
				z = from_real(eps * norm);
			}

			if x[i] != zero() {
				x[i] = x[i] * recip(z);
			}
		}
	} else {
		let mid = n / 2;

		let (A00, A01, _, A11) = A.split_at(mid, mid);
		let (mut x0, mut x1) = x.rb_mut().split_at_row_mut(mid);

		solve_shifted_upper_triangular_system(A11, conj_A, copy(p), x1.rb_mut(), copy(norm), par, params);

		linalg::matmul::matmul_with_conj(x0.rb_mut().as_mat_mut(), Accum::Add, A01, conj_A, x1.rb().as_mat(), Conj::No, -one(), par);

		solve_shifted_upper_triangular_system(A00, conj_A, p, x0.rb_mut(), norm, par, params);
	}
}

/// computes the layout of the workspace required to compute a matrix's
/// eigendecomposition
pub fn evd_scratch<T: ComplexField>(
	dim: usize,
	eigen_left: ComputeEigenvectors,
	eigen_right: ComputeEigenvectors,
	par: Par,
	params: Spec<EvdParams, T>,
) -> StackReq {
	let n = dim;

	if n == 0 {
		return StackReq::EMPTY;
	}

	let compute_eigen = eigen_left == ComputeEigenvectors::Yes || eigen_right == ComputeEigenvectors::Yes;
	let bs = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n - 1, n - 1);

	let H = temp_mat_scratch::<T>(n, n);
	let X = H;
	let Z = temp_mat_scratch::<T>(n, if compute_eigen { n } else { 0 });
	let householder = temp_mat_scratch::<T>(bs, n);
	let apply = linalg::householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<T>(n - 1, bs, n - 1);

	StackReq::all_of(&[
		H,
		Z,
		StackReq::any_of(&[
			householder.and(hessenberg::hessenberg_in_place_scratch::<T>(n, bs, par, params.hessenberg.into()).or(apply)),
			schur::multishift_qr_scratch::<T>(n, n, compute_eigen, compute_eigen, par, params.schur),
			X,
		]),
	])
}

#[math]
fn evd_imp<T: ComplexField>(
	A: MatRef<'_, T>,
	s: ColMut<'_, T>,
	s_im: Option<ColMut<'_, T>>,
	u_left: Option<MatMut<'_, T>>,
	u_right: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: EvdParams,
) -> Result<(), EvdError> {
	let n = A.nrows();

	if n == 0 {
		return Ok(());
	}

	for j in 0..n {
		for i in 0..n {
			if !is_finite(A[(i, j)]) {
				return Err(EvdError::NoConvergence);
			}
		}
	}

	let bs = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n - 1, n - 1);
	let mut s = s;
	let mut s_im = s_im;

	let (mut H, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let (mut Z, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, if u_left.is_some() || u_right.is_some() { n } else { 0 }, stack) };

	let mut H = H.as_mat_mut();
	let mut Z = if u_left.is_some() || u_right.is_some() {
		Some(Z.as_mat_mut())
	} else {
		None
	};

	H.copy_from(A);

	{
		let (mut householder, stack) = unsafe { temp_mat_uninit::<T, _, _>(bs, n - 1, stack) };
		let mut householder = householder.as_mat_mut();

		hessenberg::hessenberg_in_place(H.rb_mut(), householder.rb_mut(), par, stack, params.hessenberg.into());

		if let Some(mut Z) = Z.rb_mut() {
			Z.fill(zero());
			Z.rb_mut().diagonal_mut().fill(one());

			linalg::householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
				H.rb().submatrix(1, 0, n - 1, n - 1),
				householder.rb(),
				Conj::No,
				Z.rb_mut().submatrix_mut(1, 1, n - 1, n - 1),
				par,
				stack,
			);
		}

		for j in 0..n {
			for i in j + 2..n {
				H[(i, j)] = zero();
			}
		}
	}

	if try_const! { T::IS_REAL } {
		schur::real_schur::multishift_qr::<T::Real>(
			unsafe { core::mem::transmute(Z.is_some()) },
			unsafe { core::mem::transmute(H.rb_mut()) },
			unsafe { core::mem::transmute(Z.rb_mut()) },
			unsafe { core::mem::transmute(s.rb_mut()) },
			unsafe { core::mem::transmute(s_im.rb_mut().unwrap()) },
			0,
			n,
			par,
			stack,
			params.schur,
		);
	} else {
		schur::complex_schur::multishift_qr::<T>(Z.is_some(), H.rb_mut(), Z.rb_mut(), s.rb_mut(), 0, n, par, stack, params.schur);
	}

	let H = H.rb();

	if let (Some(mut u), Some(Z)) = (u_right, Z.rb()) {
		let (mut X, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
		let mut X = X.as_mat_mut();

		if try_const! { T::IS_REAL } {
			evd_from_real_schur_imp::<T::Real>(
				unsafe { core::mem::transmute(H) },
				unsafe { core::mem::transmute(X.rb_mut()) },
				par,
				params.evd_from_schur,
			);
		} else {
			evd_from_cplx_schur_imp::<T>(H, Conj::No, X.rb_mut(), par, params.evd_from_schur);
		}

		linalg::matmul::triangular::matmul(
			u.rb_mut(),
			BlockStructure::Rectangular,
			Accum::Replace,
			Z,
			BlockStructure::Rectangular,
			X.rb(),
			BlockStructure::TriangularUpper,
			one(),
			par,
		);
	}

	if let (Some(mut u), Some(Z)) = (u_left, Z.rb()) {
		let (mut X, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
		let mut X = X.as_mat_mut().reverse_rows_mut();

		if try_const! { T::IS_REAL } {
			evd_from_real_schur_imp::<T::Real>(
				unsafe { core::mem::transmute(H.transpose().reverse_rows_and_cols()) },
				unsafe { core::mem::transmute(X.rb_mut()) },
				par,
				params.evd_from_schur,
			);
		} else {
			evd_from_cplx_schur_imp::<T>(H.transpose().reverse_rows_and_cols(), Conj::Yes, X.rb_mut(), par, params.evd_from_schur);
		}

		linalg::matmul::triangular::matmul(
			u.rb_mut(),
			BlockStructure::Rectangular,
			Accum::Replace,
			Z,
			BlockStructure::Rectangular,
			X.rb().reverse_rows_and_cols(),
			BlockStructure::TriangularLower,
			one(),
			par,
		);
	}

	Ok(())
}

/// computes the matrix $A$'s eigendecomposition
///
/// the eigenvalues are stored in $S$, the left eigenvectors in $U_L$, and the right eigenvectors in
/// $U_R$
#[track_caller]
pub fn evd_cplx<T: RealField>(
	A: MatRef<'_, Complex<T>>,
	s: DiagMut<'_, Complex<T>>,
	u_left: Option<MatMut<'_, Complex<T>>>,
	u_right: Option<MatMut<'_, Complex<T>>>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<EvdParams, Complex<T>>,
) -> Result<(), EvdError> {
	let n = A.nrows();
	assert!(all(A.nrows() == n, A.ncols() == n, s.dim() == n));
	if let Some(u) = u_left.rb() {
		assert!(all(u.nrows() == n, u.ncols() == n));
	}
	if let Some(u) = u_right.rb() {
		assert!(all(u.nrows() == n, u.ncols() == n));
	}

	evd_imp(A, s.column_vector_mut(), None, u_left, u_right, par, stack, params.config)
}

/// computes the matrix $A$'s eigendecomposition
///
/// the eigenvalues are stored in $S$, the left eigenvectors in $U_L$, and the right eigenvectors in
/// $U_R$
#[track_caller]
pub fn evd_real<T: RealField>(
	A: MatRef<'_, T>,
	s_re: DiagMut<'_, T>,
	s_im: DiagMut<'_, T>,
	u_left: Option<MatMut<'_, T>>,
	u_right: Option<MatMut<'_, T>>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<EvdParams, T>,
) -> Result<(), EvdError> {
	let n = A.nrows();
	assert!(all(A.nrows() == n, A.ncols() == n, s_re.dim() == n, s_im.dim() == n));
	if let Some(u) = u_left.rb() {
		assert!(all(u.nrows() == n, u.ncols() == n));
	}
	if let Some(u) = u_right.rb() {
		assert!(all(u.nrows() == n, u.ncols() == n));
	}

	evd_imp(
		A,
		s_re.column_vector_mut(),
		Some(s_im.column_vector_mut()),
		u_left,
		u_right,
		par,
		stack,
		params.config,
	)
}

#[cfg(test)]
mod general_tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;

	fn test_cplx_evd(mat: MatRef<'_, c64>) {
		let n = mat.nrows();
		let params = Spec::new(EvdParams {
			hessenberg: auto!(c64),
			schur: auto!(c64),
			evd_from_schur: EvdFromSchurParams {
				recursion_threshold: 8,
				..auto!(c64)
			},
			..auto!(c64)
		});

		use faer_traits::math_utils::*;
		let approx_eq = CwiseMat(ApproxEq::eps() * sqrt(&from_f64(8.0 * n as f64)));

		let mut s = Mat::zeros(n, n);
		{
			let mut ul = Mat::zeros(n, n);
			let mut ur = Mat::zeros(n, n);

			evd_cplx(
				mat.as_ref(),
				s.as_mut().diagonal_mut(),
				Some(ul.as_mut()),
				Some(ur.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(evd_scratch::<c64>(
					n,
					ComputeEigenvectors::Yes,
					ComputeEigenvectors::Yes,
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();

			assert!(&ur * &s ~ mat * &ur);
			assert!(&s * ul.adjoint() ~ ul.adjoint() * mat);
		}
	}

	fn test_real_evd(mat: MatRef<'_, f64>) {
		let n = mat.nrows();
		let params = Spec::new(EvdParams {
			hessenberg: auto!(f64),
			schur: auto!(f64),
			evd_from_schur: EvdFromSchurParams {
				recursion_threshold: 8,
				..auto!(f64)
			},
			..auto!(f64)
		});

		use faer_traits::math_utils::*;
		let approx_eq = CwiseMat(ApproxEq::<f64>::eps() * sqrt(&from_f64(8.0 * n as f64)));

		let mut s_re = Diag::zeros(n);
		let mut s_im = Diag::zeros(n);
		{
			let mut ul = Mat::zeros(n, n);
			let mut ur = Mat::zeros(n, n);

			evd_real(
				mat.as_ref(),
				s_re.as_mut(),
				s_im.as_mut(),
				Some(ul.as_mut()),
				Some(ur.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(evd_scratch::<f64>(
					n,
					ComputeEigenvectors::Yes,
					ComputeEigenvectors::Yes,
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();

			let mut i = 0;
			while i < n {
				if s_im[i] == 0.0 {
					let ur = ur.col(i);
					let ul = ul.col(i);

					let s = Scale(s_re[i]);

					assert!((&ur * s).as_mat() ~ (mat * &ur).as_mat());
					assert!((&ul.adjoint() * s).as_mat() ~ (&ul.adjoint() * mat).as_mat());

					i += 1;
				} else {
					let re = ur.col(i);
					let im = ur.col(i + 1);
					let ur = &Col::from_fn(n, |i| c64::new(re[i], im[i]));

					let re = ul.col(i);
					let im = ul.col(i + 1);
					let ul = &Col::from_fn(n, |i| c64::new(re[i], im[i]));

					let mat = &Mat::from_fn(n, n, |i, j| c64::from(mat[(i, j)]));

					let s = Scale(c64::new(s_re[i], s_im[i]));

					let approx_eq = CwiseMat(ApproxEq::eps() * sqrt(&from_f64(8.0 * n as f64)));

					assert!((ur * s).as_mat() ~ (mat * ur).as_mat());
					assert!((ul.adjoint() * s).as_mat() ~ (ul.adjoint() * mat).as_mat());

					i += 2;
				}
			}
		}
	}

	#[test]
	fn test_cplx() {
		let rng = &mut StdRng::seed_from_u64(1);

		for n in [2, 4, 10, 15, 20, 50, 100, 150] {
			let mat = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			test_cplx_evd(mat.as_ref());
		}
	}

	#[test]
	fn test_real() {
		let rng = &mut StdRng::seed_from_u64(0);

		for n in [2, 4, 10, 15, 20, 50, 100, 150] {
			let mat = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);

			test_real_evd(mat.as_ref());
		}
	}
}

#[cfg(test)]
mod self_adjoint_tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;

	fn test_self_adjoint_evd<T: ComplexField>(mat: MatRef<'_, T>) {
		let n = mat.nrows();
		let params = Spec::new(SelfAdjointEvdParams {
			recursion_threshold: 8,
			..auto!(T)
		});
		use faer_traits::math_utils::*;
		let approx_eq = CwiseMat(ApproxEq::<T::Real>::eps() * sqrt(&from_f64(8.0 * n as f64)));

		let mut s = Mat::zeros(n, n);
		{
			let mut u = Mat::zeros(n, n);

			self_adjoint_evd(
				mat.as_ref(),
				s.as_mut().diagonal_mut(),
				Some(u.as_mut()),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(self_adjoint_evd_scratch::<T>(
					n,
					ComputeEigenvectors::Yes,
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();

			let reconstructed = &u * &s * u.adjoint();
			assert!(reconstructed ~ mat);
		}

		{
			let mut s2 = Mat::zeros(n, n);

			self_adjoint_evd(
				mat.as_ref(),
				s2.as_mut().diagonal_mut(),
				None,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(self_adjoint_evd_scratch::<T>(
					n,
					ComputeEigenvectors::No,
					Par::Seq,
					params,
				))),
				params,
			)
			.unwrap();

			assert!(s2 ~ s);
		}
	}

	#[test]
	fn test_real() {
		let rng = &mut StdRng::seed_from_u64(1);

		for n in [1, 2, 4, 10, 15, 20, 50, 100, 150] {
			let mat = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: StandardNormal,
			}
			.rand::<Mat<f64>>(rng);
			let mat = &mat + mat.adjoint();

			test_self_adjoint_evd(mat.as_ref());
		}
	}

	#[test]
	fn test_cplx() {
		let rng = &mut StdRng::seed_from_u64(1);

		for n in [2, 4, 10, 15, 20, 50, 100, 150] {
			let mat = CwiseMatDistribution {
				nrows: n,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);
			let mat = &mat + mat.adjoint();

			test_self_adjoint_evd(mat.as_ref());
		}
	}

	#[test]
	fn test_special() {
		for n in [1, 2, 4, 10, 15, 20, 50, 100, 150] {
			test_self_adjoint_evd(Mat::full(n, n, 0.0).as_ref());
			test_self_adjoint_evd(Mat::full(n, n, c64::ZERO).as_ref());
			test_self_adjoint_evd(Mat::full(n, n, 1.0).as_ref());
			test_self_adjoint_evd(Mat::full(n, n, c64::ONE).as_ref());
			test_self_adjoint_evd(Mat::<f64>::identity(n, n).as_ref());
			test_self_adjoint_evd(Mat::<c64>::identity(n, n).as_ref());
		}
	}

	#[test]
	fn test_pinv() {
		let rng = &mut StdRng::seed_from_u64(0);

		let n = 36;

		let mat = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: StandardNormal,
		}
		.rand::<Mat<f64>>(rng);

		let mat = &mat + mat.adjoint();

		let pinv = mat.self_adjoint_eigen(Side::Lower).unwrap().pseudoinverse();
		let err = &mat * &pinv - Mat::<f64>::identity(n, n);
		assert!(err.norm_max() < 1e-10);
	}
}
