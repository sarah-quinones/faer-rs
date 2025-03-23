use crate::internal_prelude::*;
use crate::{assert, get_global_parallelism};
use alloc::vec;
use alloc::vec::Vec;
use dyn_stack::MemBuffer;
use faer_traits::math_utils;
use linalg::svd::ComputeSvdVectors;

pub use linalg::cholesky::ldlt::factor::LdltError;
pub use linalg::cholesky::llt::factor::LltError;
pub use linalg::evd::EvdError;
pub use linalg::svd::SvdError;

/// shape info of a linear system solver
pub trait ShapeCore {
	/// returns the number of rows of the matrix
	fn nrows(&self) -> usize;
	/// returns the number of columns of the matrix
	fn ncols(&self) -> usize;
}

/// linear system solver implementation
pub trait SolveCore<T: ComplexField>: ShapeCore {
	/// solves the equation `self × x = rhs`, implicitly conjugating `self` if needed, and stores
	/// the result in `rhs`
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>);
	/// solves the equation `self.transpose() × x = rhs`, implicitly conjugating `self` if needed,
	/// and stores the result in `rhs`
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>);
}
/// least squares linear system solver implementation
pub trait SolveLstsqCore<T: ComplexField>: ShapeCore {
	/// solves the equation `self × x = rhs` in the sense of least squares, implicitly conjugating
	/// `self` if needed, and stores the result in the top rows of `rhs`
	fn solve_lstsq_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>);
}
/// dense linear system solver
pub trait DenseSolveCore<T: ComplexField>: SolveCore<T> {
	/// returns an approximation of the matrix that was used to create the decomposition
	fn reconstruct(&self) -> Mat<T>;
	/// returns an approximation of the inverse of the matrix that was used to create the
	/// decomposition
	fn inverse(&self) -> Mat<T>;
}

impl<S: ?Sized + ShapeCore> ShapeCore for &S {
	#[inline]
	fn nrows(&self) -> usize {
		(**self).nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		(**self).ncols()
	}
}

impl<T: ComplexField, S: ?Sized + SolveCore<T>> SolveCore<T> for &S {
	#[inline]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		(**self).solve_in_place_with_conj(conj, rhs)
	}

	#[inline]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		(**self).solve_transpose_in_place_with_conj(conj, rhs)
	}
}

impl<T: ComplexField, S: ?Sized + SolveLstsqCore<T>> SolveLstsqCore<T> for &S {
	#[inline]
	fn solve_lstsq_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		(**self).solve_lstsq_in_place_with_conj(conj, rhs)
	}
}

impl<T: ComplexField, S: ?Sized + DenseSolveCore<T>> DenseSolveCore<T> for &S {
	#[inline]
	fn reconstruct(&self) -> Mat<T> {
		(**self).reconstruct()
	}

	#[inline]
	fn inverse(&self) -> Mat<T> {
		(**self).inverse()
	}
}

/// [`SolveCore`] extension trait
pub trait Solve<T: ComplexField>: SolveCore<T> {
	#[track_caller]
	#[inline]
	/// solves $A x = b$
	fn solve_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.solve_in_place_with_conj(Conj::No, { rhs }.as_mat_mut().as_dyn_cols_mut());
	}
	#[track_caller]
	#[inline]
	/// solves $\bar A x = b$
	fn solve_conjugate_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.solve_in_place_with_conj(Conj::Yes, { rhs }.as_mat_mut().as_dyn_cols_mut());
	}

	#[track_caller]
	#[inline]
	/// solves $A^\top x = b$
	fn solve_transpose_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.solve_transpose_in_place_with_conj(Conj::No, { rhs }.as_mat_mut().as_dyn_cols_mut());
	}
	#[track_caller]
	#[inline]
	/// solves $A^H x = b$
	fn solve_adjoint_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.solve_transpose_in_place_with_conj(Conj::Yes, { rhs }.as_mat_mut().as_dyn_cols_mut());
	}

	#[track_caller]
	#[inline]
	/// solves $x A = b$
	fn rsolve_in_place(&self, lhs: impl AsMatMut<T = T, Cols = usize>) {
		self.solve_transpose_in_place_with_conj(Conj::No, { lhs }.as_mat_mut().as_dyn_rows_mut().transpose_mut());
	}
	#[track_caller]
	#[inline]
	/// solves $x \bar A = b$
	fn rsolve_conjugate_in_place(&self, lhs: impl AsMatMut<T = T, Cols = usize>) {
		self.solve_transpose_in_place_with_conj(Conj::Yes, { lhs }.as_mat_mut().as_dyn_rows_mut().transpose_mut());
	}

	#[track_caller]
	#[inline]
	/// solves $x A^\top = b$
	fn rsolve_transpose_in_place(&self, lhs: impl AsMatMut<T = T, Cols = usize>) {
		self.solve_in_place_with_conj(Conj::No, { lhs }.as_mat_mut().as_dyn_rows_mut().transpose_mut());
	}
	#[track_caller]
	#[inline]
	/// solves $x A^H = b$
	fn rsolve_adjoint_in_place(&self, lhs: impl AsMatMut<T = T, Cols = usize>) {
		self.solve_in_place_with_conj(Conj::Yes, { lhs }.as_mat_mut().as_dyn_rows_mut().transpose_mut());
	}

	#[track_caller]
	#[inline]
	/// solves $A x = b$
	fn solve<Rhs: AsMatRef<T = T, Rows = usize>>(&self, rhs: Rhs) -> Rhs::Owned {
		let rhs = rhs.as_mat_ref();
		let mut out = Rhs::Owned::zeros(rhs.nrows(), rhs.ncols());
		out.as_mat_mut().copy_from(rhs);
		self.solve_in_place(&mut out);
		out
	}
	#[track_caller]
	#[inline]
	/// solves $\bar A x = b$
	fn solve_conjugate<Rhs: AsMatRef<T = T, Rows = usize>>(&self, rhs: Rhs) -> Rhs::Owned {
		let rhs = rhs.as_mat_ref();
		let mut out = Rhs::Owned::zeros(rhs.nrows(), rhs.ncols());
		out.as_mat_mut().copy_from(rhs);
		self.solve_conjugate_in_place(&mut out);
		out
	}

	#[track_caller]
	#[inline]
	/// solves $A^\top x = b$
	fn solve_transpose<Rhs: AsMatRef<T = T, Rows = usize>>(&self, rhs: Rhs) -> Rhs::Owned {
		let rhs = rhs.as_mat_ref();
		let mut out = Rhs::Owned::zeros(rhs.nrows(), rhs.ncols());
		out.as_mat_mut().copy_from(rhs);
		self.solve_transpose_in_place(&mut out);
		out
	}
	#[track_caller]
	#[inline]
	/// solves $A^H x = b$
	fn solve_adjoint<Rhs: AsMatRef<T = T, Rows = usize>>(&self, rhs: Rhs) -> Rhs::Owned {
		let rhs = rhs.as_mat_ref();
		let mut out = Rhs::Owned::zeros(rhs.nrows(), rhs.ncols());
		out.as_mat_mut().copy_from(rhs);
		self.solve_adjoint_in_place(&mut out);
		out
	}

	#[track_caller]
	#[inline]
	/// solves $x A = b$
	fn rsolve<Lhs: AsMatRef<T = T, Cols = usize>>(&self, lhs: Lhs) -> Lhs::Owned {
		let lhs = lhs.as_mat_ref();
		let mut out = Lhs::Owned::zeros(lhs.nrows(), lhs.ncols());
		out.as_mat_mut().copy_from(lhs);
		self.rsolve_in_place(&mut out);
		out
	}
	#[track_caller]
	#[inline]
	/// solves $x \bar A = b$
	fn rsolve_conjugate<Lhs: AsMatRef<T = T, Cols = usize>>(&self, lhs: Lhs) -> Lhs::Owned {
		let lhs = lhs.as_mat_ref();
		let mut out = Lhs::Owned::zeros(lhs.nrows(), lhs.ncols());
		out.as_mat_mut().copy_from(lhs);
		self.rsolve_conjugate_in_place(&mut out);
		out
	}

	#[track_caller]
	#[inline]
	/// solves $x A^\top = b$
	fn rsolve_transpose<Lhs: AsMatRef<T = T, Cols = usize>>(&self, lhs: Lhs) -> Lhs::Owned {
		let lhs = lhs.as_mat_ref();
		let mut out = Lhs::Owned::zeros(lhs.nrows(), lhs.ncols());
		out.as_mat_mut().copy_from(lhs);
		self.rsolve_transpose_in_place(&mut out);
		out
	}
	#[track_caller]
	#[inline]
	/// solves $x A^H = b$
	fn rsolve_adjoint<Lhs: AsMatRef<T = T, Cols = usize>>(&self, lhs: Lhs) -> Lhs::Owned {
		let lhs = lhs.as_mat_ref();
		let mut out = Lhs::Owned::zeros(lhs.nrows(), lhs.ncols());
		out.as_mat_mut().copy_from(lhs);
		self.rsolve_adjoint_in_place(&mut out);
		out
	}
}

impl<C: Conjugate> MatRef<'_, C> {
	#[track_caller]
	/// returns the $LU$ decomposition of `self` with partial (row) pivoting
	pub fn partial_piv_lu(&self) -> PartialPivLu<C::Canonical> {
		PartialPivLu::new(self.as_mat_ref())
	}

	#[track_caller]
	/// returns the $LU$ decomposition of `self` with full pivoting
	pub fn full_piv_lu(&self) -> FullPivLu<C::Canonical> {
		FullPivLu::new(self.as_mat_ref())
	}

	#[track_caller]
	/// returns the $QR$ decomposition of `self`
	pub fn qr(&self) -> Qr<C::Canonical> {
		Qr::new(self.as_mat_ref())
	}

	#[track_caller]
	/// returns the $QR$ decomposition of `self` with column pivoting
	pub fn col_piv_qr(&self) -> ColPivQr<C::Canonical> {
		ColPivQr::new(self.as_mat_ref())
	}

	#[track_caller]
	/// returns the svd of `self`
	///
	/// singular values are nonnegative and sorted in nonincreasing order
	pub fn svd(&self) -> Result<Svd<C::Canonical>, SvdError> {
		Svd::new(self.as_mat_ref())
	}

	#[track_caller]
	/// returns the thin svd of `self`
	///
	/// singular values are nonnegative and sorted in nonincreasing order
	pub fn thin_svd(&self) -> Result<Svd<C::Canonical>, SvdError> {
		Svd::new_thin(self.as_mat_ref())
	}

	#[track_caller]
	/// returns the $L L^\top$ decomposition of `self`
	pub fn llt(&self, side: Side) -> Result<Llt<C::Canonical>, LltError> {
		Llt::new(self.as_mat_ref(), side)
	}

	#[track_caller]
	/// returns the $L D L^\top$ decomposition of `self`
	pub fn ldlt(&self, side: Side) -> Result<Ldlt<C::Canonical>, LdltError> {
		Ldlt::new(self.as_mat_ref(), side)
	}

	#[track_caller]
	/// returns the bunch-kaufman decomposition of `self`
	pub fn lblt(&self, side: Side) -> Lblt<C::Canonical> {
		Lblt::new(self.as_mat_ref(), side)
	}

	#[track_caller]
	/// returns the eigendecomposition of `self`, assuming it is self-adjoint
	///
	/// eigenvalues sorted in nondecreasing order
	pub fn self_adjoint_eigen(&self, side: Side) -> Result<SelfAdjointEigen<C::Canonical>, EvdError> {
		SelfAdjointEigen::new(self.as_mat_ref(), side)
	}

	#[track_caller]
	/// returns the eigenvalues of `self`, assuming it is self-adjoint
	///
	/// eigenvalues sorted in nondecreasing order
	pub fn self_adjoint_eigenvalues(&self, side: Side) -> Result<Vec<Real<C>>, EvdError> {
		#[track_caller]
		pub fn imp<T: ComplexField>(mut A: MatRef<'_, T>, side: Side) -> Result<Vec<T::Real>, EvdError> {
			assert!(A.nrows() == A.ncols());
			if side == Side::Upper {
				A = A.transpose();
			}
			let par = get_global_parallelism();
			let n = A.nrows();

			let mut s = Diag::<T>::zeros(n);

			linalg::evd::self_adjoint_evd(
				A,
				s.as_mut(),
				None,
				par,
				MemStack::new(&mut MemBuffer::new(linalg::evd::self_adjoint_evd_scratch::<T>(
					n,
					linalg::evd::ComputeEigenvectors::No,
					par,
					default(),
				))),
				default(),
			)?;

			Ok(s.column_vector().iter().map(|x| real(x)).collect())
		}

		imp(self.as_mat_ref().canonical(), side)
	}

	#[track_caller]
	/// returns the singular values of `self`
	///
	/// singular values are nonnegative and sorted in nonincreasing order
	pub fn singular_values(&self) -> Result<Vec<Real<C>>, SvdError> {
		pub fn imp<T: ComplexField>(A: MatRef<'_, T>) -> Result<Vec<T::Real>, SvdError> {
			let par = get_global_parallelism();
			let m = A.nrows();
			let n = A.ncols();

			let mut s = Diag::<T>::zeros(Ord::min(m, n));

			linalg::svd::svd(
				A,
				s.as_mut(),
				None,
				None,
				par,
				MemStack::new(&mut MemBuffer::new(linalg::svd::svd_scratch::<T>(
					m,
					n,
					linalg::svd::ComputeSvdVectors::No,
					linalg::svd::ComputeSvdVectors::No,
					par,
					default(),
				))),
				default(),
			)?;

			Ok(s.column_vector().iter().map(|x| real(x)).collect())
		}

		imp(self.as_mat_ref().canonical())
	}
}

impl<T: RealField> MatRef<'_, T> {
	#[track_caller]
	/// returns the eigendecomposition of `self`
	pub fn eigen_from_real(&self) -> Result<Eigen<T>, EvdError> {
		Eigen::new_from_real(self.as_mat_ref())
	}

	#[track_caller]
	/// returns the eigenvalues of `self`
	pub fn eigenvalues_from_real(&self) -> Result<Vec<Complex<T>>, EvdError> {
		let par = get_global_parallelism();

		let A = self.as_mat_ref();
		assert!(A.nrows() == A.ncols());
		let n = A.nrows();

		let mut s_re = Diag::<T>::zeros(n);
		let mut s_im = Diag::<T>::zeros(n);

		linalg::evd::evd_real(
			A,
			s_re.as_mut(),
			s_im.as_mut(),
			None,
			None,
			par,
			MemStack::new(&mut MemBuffer::new(linalg::evd::evd_scratch::<T>(
				n,
				linalg::evd::ComputeEigenvectors::No,
				linalg::evd::ComputeEigenvectors::No,
				par,
				default(),
			))),
			default(),
		)?;

		Ok(s_re
			.column_vector()
			.iter()
			.zip(s_im.column_vector().iter())
			.map(|(re, im)| Complex::new(re.clone(), im.clone()))
			.collect())
	}
}

impl<T: RealField> MatRef<'_, Complex<T>> {
	#[track_caller]
	/// returns the eigendecomposition of `self`
	pub fn eigen(&self) -> Result<Eigen<T>, EvdError> {
		Eigen::new(self.as_mat_ref())
	}

	#[track_caller]
	/// returns the eigenvalues of `self`
	pub fn eigenvalues(&self) -> Result<Vec<Complex<T>>, EvdError> {
		let par = get_global_parallelism();

		let A = self.as_mat_ref();
		assert!(A.nrows() == A.ncols());
		let n = A.nrows();

		let mut s = Diag::<Complex<T>>::zeros(n);

		linalg::evd::evd_cplx(
			A,
			s.as_mut(),
			None,
			None,
			par,
			MemStack::new(&mut MemBuffer::new(linalg::evd::evd_scratch::<Complex<T>>(
				n,
				linalg::evd::ComputeEigenvectors::No,
				linalg::evd::ComputeEigenvectors::No,
				par,
				default(),
			))),
			default(),
		)?;

		Ok(s.column_vector().iter().cloned().collect())
	}
}

impl<C: Conjugate> MatMut<'_, C> {
	#[track_caller]
	/// returns the $LU$ decomposition of `self` with partial (row) pivoting
	pub fn partial_piv_lu(&self) -> PartialPivLu<C::Canonical> {
		self.rb().partial_piv_lu()
	}

	#[track_caller]
	/// returns the $LU$ decomposition of `self` with full pivoting
	pub fn full_piv_lu(&self) -> FullPivLu<C::Canonical> {
		self.rb().full_piv_lu()
	}

	#[track_caller]
	/// returns the $QR$ decomposition of `self`
	pub fn qr(&self) -> Qr<C::Canonical> {
		self.rb().qr()
	}

	#[track_caller]
	/// returns the $QR$ decomposition of `self` with column pivoting
	pub fn col_piv_qr(&self) -> ColPivQr<C::Canonical> {
		self.rb().col_piv_qr()
	}

	#[track_caller]
	/// returns the svd of `self`
	///
	/// singular values are nonnegative and sorted in nonincreasing order
	pub fn svd(&self) -> Result<Svd<C::Canonical>, SvdError> {
		self.rb().svd()
	}

	#[track_caller]
	/// returns the thin svd of `self`
	///
	/// singular values are nonnegative and sorted in nonincreasing order
	pub fn thin_svd(&self) -> Result<Svd<C::Canonical>, SvdError> {
		self.rb().thin_svd()
	}

	#[track_caller]
	/// returns the $L L^\top$ decomposition of `self`
	pub fn llt(&self, side: Side) -> Result<Llt<C::Canonical>, LltError> {
		self.rb().llt(side)
	}

	#[track_caller]
	/// returns the $L D L^\top$ decomposition of `self`
	pub fn ldlt(&self, side: Side) -> Result<Ldlt<C::Canonical>, LdltError> {
		self.rb().ldlt(side)
	}

	#[track_caller]
	/// returns the bunch-kaufman decomposition of `self`
	pub fn lblt(&self, side: Side) -> Lblt<C::Canonical> {
		self.rb().lblt(side)
	}

	#[track_caller]
	/// returns the eigendecomposition of `self`, assuming it is self-adjoint
	///
	/// eigenvalues sorted in nondecreasing order
	pub fn self_adjoint_eigen(&self, side: Side) -> Result<SelfAdjointEigen<C::Canonical>, EvdError> {
		self.rb().self_adjoint_eigen(side)
	}

	#[track_caller]
	/// returns the eigenvalues of `self`, assuming it is self-adjoint
	///
	/// eigenvalues sorted in nondecreasing order
	pub fn self_adjoint_eigenvalues(&self, side: Side) -> Result<Vec<Real<C>>, EvdError> {
		self.rb().self_adjoint_eigenvalues(side)
	}

	#[track_caller]
	/// returns the singular values of `self`
	///
	/// singular values are nonnegative and sorted in nonincreasing order
	pub fn singular_values(&self) -> Result<Vec<Real<C>>, SvdError> {
		self.rb().singular_values()
	}
}

impl<T: RealField> MatMut<'_, T> {
	#[track_caller]
	/// returns the eigendecomposition of `self`
	pub fn eigen_from_real(&self) -> Result<Eigen<T>, EvdError> {
		self.rb().eigen_from_real()
	}

	#[track_caller]
	/// returns the eigenvalues of `self`
	pub fn eigenvalues_from_real(&self) -> Result<Vec<Complex<T>>, EvdError> {
		self.rb().eigenvalues_from_real()
	}
}

impl<T: RealField> MatMut<'_, Complex<T>> {
	#[track_caller]
	/// returns the eigendecomposition of `self`
	pub fn eigen(&self) -> Result<Eigen<T>, EvdError> {
		self.rb().eigen()
	}

	#[track_caller]
	/// returns the eigenvalues of `self`
	pub fn eigenvalues(&self) -> Result<Vec<Complex<T>>, EvdError> {
		self.rb().eigenvalues()
	}
}

impl<C: Conjugate> Mat<C> {
	#[track_caller]
	/// returns the $LU$ decomposition of `self` with partial (row) pivoting
	pub fn partial_piv_lu(&self) -> PartialPivLu<C::Canonical> {
		self.rb().partial_piv_lu()
	}

	#[track_caller]
	/// returns the $LU$ decomposition of `self` with full pivoting
	pub fn full_piv_lu(&self) -> FullPivLu<C::Canonical> {
		self.rb().full_piv_lu()
	}

	#[track_caller]
	/// returns the $QR$ decomposition of `self`
	pub fn qr(&self) -> Qr<C::Canonical> {
		self.rb().qr()
	}

	#[track_caller]
	/// returns the $QR$ decomposition of `self` with column pivoting
	pub fn col_piv_qr(&self) -> ColPivQr<C::Canonical> {
		self.rb().col_piv_qr()
	}

	#[track_caller]
	/// returns the svd of `self`
	///
	/// singular values are nonnegative and sorted in nonincreasing order
	pub fn svd(&self) -> Result<Svd<C::Canonical>, SvdError> {
		self.rb().svd()
	}

	#[track_caller]
	/// returns the thin svd of `self`
	///
	/// singular values are nonnegative and sorted in nonincreasing order
	pub fn thin_svd(&self) -> Result<Svd<C::Canonical>, SvdError> {
		self.rb().thin_svd()
	}

	#[track_caller]
	/// returns the $L L^\top$ decomposition of `self`
	pub fn llt(&self, side: Side) -> Result<Llt<C::Canonical>, LltError> {
		self.rb().llt(side)
	}

	#[track_caller]
	/// returns the $L D L^\top$ decomposition of `self`
	pub fn ldlt(&self, side: Side) -> Result<Ldlt<C::Canonical>, LdltError> {
		self.rb().ldlt(side)
	}

	#[track_caller]
	/// returns the bunch-kaufman decomposition of `self`
	pub fn lblt(&self, side: Side) -> Lblt<C::Canonical> {
		self.rb().lblt(side)
	}

	#[track_caller]
	/// returns the eigendecomposition of `self`, assuming it is self-adjoint
	///
	/// eigenvalues sorted in nondecreasing order
	pub fn self_adjoint_eigen(&self, side: Side) -> Result<SelfAdjointEigen<C::Canonical>, EvdError> {
		self.rb().self_adjoint_eigen(side)
	}

	#[track_caller]
	/// returns the eigenvalues of `self`, assuming it is self-adjoint
	///
	/// eigenvalues sorted in nondecreasing order
	pub fn self_adjoint_eigenvalues(&self, side: Side) -> Result<Vec<Real<C>>, EvdError> {
		self.rb().self_adjoint_eigenvalues(side)
	}

	#[track_caller]
	/// returns the singular values of `self`
	///
	/// singular values are nonnegative and sorted in nonincreasing order
	pub fn singular_values(&self) -> Result<Vec<Real<C>>, SvdError> {
		self.rb().singular_values()
	}
}

impl<T: RealField> Mat<T> {
	#[track_caller]
	/// returns the eigendecomposition of `self`
	pub fn eigen_from_real(&self) -> Result<Eigen<T>, EvdError> {
		self.rb().eigen_from_real()
	}

	#[track_caller]
	/// returns the eigenvalues of `self`
	pub fn eigenvalues_from_real(&self) -> Result<Vec<Complex<T>>, EvdError> {
		self.rb().eigenvalues_from_real()
	}
}

impl<T: RealField> Mat<Complex<T>> {
	#[track_caller]
	/// returns the eigendecomposition of `self`
	pub fn eigen(&self) -> Result<Eigen<T>, EvdError> {
		self.rb().eigen()
	}

	#[track_caller]
	/// returns the eigenvalues of `self`
	pub fn eigenvalues(&self) -> Result<Vec<Complex<T>>, EvdError> {
		self.rb().eigenvalues()
	}
}

/// [`SolveLstsqCore`] extension trait
pub trait SolveLstsq<T: ComplexField>: SolveLstsqCore<T> {}
/// [`DenseSolveCore`] extension trait
pub trait DenseSolve<T: ComplexField>: DenseSolveCore<T> {}

impl<T: ComplexField, S: ?Sized + SolveCore<T>> Solve<T> for S {}
impl<T: ComplexField, S: ?Sized + SolveLstsqCore<T>> SolveLstsq<T> for S {}
impl<T: ComplexField, S: ?Sized + DenseSolveCore<T>> DenseSolve<T> for S {}

/// $L L^\top$ decomposition
#[derive(Clone, Debug)]
pub struct Llt<T> {
	L: Mat<T>,
}

/// $L D L^\top$ decomposition
#[derive(Clone, Debug)]
pub struct Ldlt<T> {
	L: Mat<T>,
	D: Diag<T>,
}

/// bunch-kaufman decomposition
#[derive(Clone, Debug)]
pub struct Lblt<T> {
	L: Mat<T>,
	B_diag: Diag<T>,
	B_subdiag: Diag<T>,
	P: Perm<usize>,
}

/// $LU$ decomposition with partial (row) pivoting
#[derive(Clone, Debug)]
pub struct PartialPivLu<T> {
	L: Mat<T>,
	U: Mat<T>,
	P: Perm<usize>,
}

/// $LU$ decomposition with full pivoting
#[derive(Clone, Debug)]
pub struct FullPivLu<T> {
	L: Mat<T>,
	U: Mat<T>,
	P: Perm<usize>,
	Q: Perm<usize>,
}

/// $QR$ decomposition
#[derive(Clone, Debug)]
pub struct Qr<T> {
	Q_basis: Mat<T>,
	Q_coeff: Mat<T>,
	R: Mat<T>,
}

/// $QR$ decomposition with column pivoting
#[derive(Clone, Debug)]
pub struct ColPivQr<T> {
	Q_basis: Mat<T>,
	Q_coeff: Mat<T>,
	R: Mat<T>,
	P: Perm<usize>,
}

/// svd decomposition (either full or thin)
#[derive(Clone, Debug)]
pub struct Svd<T> {
	U: Mat<T>,
	V: Mat<T>,
	S: Diag<T>,
}

/// self-adjoint eigendecomposition
#[derive(Clone, Debug)]
pub struct SelfAdjointEigen<T> {
	U: Mat<T>,
	S: Diag<T>,
}

/// eigendecomposition
#[derive(Clone, Debug)]
pub struct Eigen<T> {
	U: Mat<Complex<T>>,
	S: Diag<Complex<T>>,
}

impl<T: ComplexField> Llt<T> {
	/// returns the $L L^\top$ decomposition of $A$
	#[track_caller]
	pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>, side: Side) -> Result<Self, LltError> {
		assert!(all(A.nrows() == A.ncols()));
		let n = A.nrows();

		let mut L = Mat::zeros(n, n);
		match side {
			Side::Lower => L.copy_from_triangular_lower(A),
			Side::Upper => L.copy_from_triangular_lower(A.adjoint()),
		}

		Self::new_imp(L)
	}

	#[track_caller]
	fn new_imp(mut L: Mat<T>) -> Result<Self, LltError> {
		let par = get_global_parallelism();

		let n = L.nrows();

		let mut mem = MemBuffer::new(linalg::cholesky::llt::factor::cholesky_in_place_scratch::<T>(n, par, default()));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::llt::factor::cholesky_in_place(L.as_mut(), Default::default(), par, stack, default())?;
		z!(&mut L).for_each_triangular_upper(linalg::zip::Diag::Skip, |uz!(x)| *x = zero());

		Ok(Self { L })
	}

	/// returns the $L$ factor
	pub fn L(&self) -> MatRef<'_, T> {
		self.L.as_ref()
	}
}

impl<T: ComplexField> Ldlt<T> {
	/// returns the $L D L^\top$ decomposition of $A$
	#[track_caller]
	pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>, side: Side) -> Result<Self, LdltError> {
		assert!(all(A.nrows() == A.ncols()));
		let n = A.nrows();

		let mut L = Mat::zeros(n, n);
		match side {
			Side::Lower => L.copy_from_triangular_lower(A),
			Side::Upper => L.copy_from_triangular_lower(A.adjoint()),
		}

		Self::new_imp(L)
	}

	#[track_caller]
	fn new_imp(mut L: Mat<T>) -> Result<Self, LdltError> {
		let par = get_global_parallelism();

		let n = L.nrows();
		let mut D = Diag::zeros(n);

		let mut mem = MemBuffer::new(linalg::cholesky::ldlt::factor::cholesky_in_place_scratch::<T>(n, par, default()));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::ldlt::factor::cholesky_in_place(L.as_mut(), Default::default(), par, stack, default())?;

		D.copy_from(L.diagonal());
		L.diagonal_mut().fill(one());
		z!(&mut L).for_each_triangular_upper(linalg::zip::Diag::Skip, |uz!(x)| *x = zero());

		Ok(Self { L, D })
	}

	/// returns the $L$ factor
	pub fn L(&self) -> MatRef<'_, T> {
		self.L.as_ref()
	}

	/// returns the $D$ factor
	pub fn D(&self) -> DiagRef<'_, T> {
		self.D.as_ref()
	}
}

impl<T: ComplexField> Lblt<T> {
	/// returns the bunch-kaufman decomposition of $A$
	#[track_caller]
	pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>, side: Side) -> Self {
		assert!(all(A.nrows() == A.ncols()));
		let n = A.nrows();

		let mut L = Mat::zeros(n, n);
		match side {
			Side::Lower => L.copy_from_triangular_lower(A),
			Side::Upper => L.copy_from_triangular_lower(A.adjoint()),
		}
		Self::new_imp(L)
	}

	#[track_caller]
	fn new_imp(mut L: Mat<T>) -> Self {
		let par = get_global_parallelism();

		let n = L.nrows();

		let mut diag = Diag::zeros(n);
		let mut subdiag = Diag::zeros(n);
		let mut perm_fwd = vec![0usize; n];
		let mut perm_bwd = vec![0usize; n];

		let mut mem = MemBuffer::new(linalg::cholesky::bunch_kaufman::factor::cholesky_in_place_scratch::<usize, T>(
			n,
			par,
			default(),
		));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::bunch_kaufman::factor::cholesky_in_place(
			L.as_mut(),
			subdiag.as_mut(),
			Default::default(),
			&mut perm_fwd,
			&mut perm_bwd,
			par,
			stack,
			default(),
		);

		diag.copy_from(L.diagonal());
		L.diagonal_mut().fill(one());
		z!(&mut L).for_each_triangular_upper(linalg::zip::Diag::Skip, |uz!(x)| *x = zero());

		Self {
			L,
			B_diag: diag,
			B_subdiag: subdiag,
			P: unsafe { Perm::new_unchecked(perm_fwd.into_boxed_slice(), perm_bwd.into_boxed_slice()) },
		}
	}

	/// returns the $L$ factor
	pub fn L(&self) -> MatRef<'_, T> {
		self.L.as_ref()
	}

	/// returns the diagonal of the $B$ factor
	pub fn B_diag(&self) -> DiagRef<'_, T> {
		self.B_diag.as_ref()
	}

	/// returns the subdiagonal of the $B$ factor
	pub fn B_subdiag(&self) -> DiagRef<'_, T> {
		self.B_subdiag.as_ref()
	}

	/// returns the pivoting permutation $P$
	pub fn P(&self) -> PermRef<'_, usize> {
		self.P.as_ref()
	}
}

fn split_LU<T: ComplexField>(LU: Mat<T>) -> (Mat<T>, Mat<T>) {
	let (m, n) = LU.shape();
	let size = Ord::min(m, n);

	let (L, U) = if m >= n {
		let mut L = LU;
		let mut U = Mat::zeros(size, size);

		U.copy_from_triangular_upper(L.get(..size, ..size));

		z!(&mut L).for_each_triangular_upper(linalg::zip::Diag::Skip, |uz!(x)| *x = zero());
		L.diagonal_mut().fill(one());

		(L, U)
	} else {
		let mut U = LU;
		let mut L = Mat::zeros(size, size);

		L.copy_from_strict_triangular_lower(U.get(..size, ..size));

		z!(&mut U).for_each_triangular_lower(linalg::zip::Diag::Skip, |uz!(x)| *x = zero());
		L.diagonal_mut().fill(one());

		(L, U)
	};
	(L, U)
}

impl<T: ComplexField> PartialPivLu<T> {
	/// returns the $LU$ decomposition of $A$ with partial pivoting
	#[track_caller]
	pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>) -> Self {
		let LU = A.to_owned();
		Self::new_imp(LU)
	}

	#[track_caller]
	fn new_imp(mut LU: Mat<T>) -> Self {
		let par = get_global_parallelism();

		let (m, n) = LU.shape();
		let mut row_perm_fwd = vec![0usize; m];
		let mut row_perm_bwd = vec![0usize; m];

		linalg::lu::partial_pivoting::factor::lu_in_place(
			LU.as_mut(),
			&mut row_perm_fwd,
			&mut row_perm_bwd,
			par,
			MemStack::new(&mut MemBuffer::new(
				linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T>(m, n, par, default()),
			)),
			default(),
		);

		let (L, U) = split_LU(LU);

		Self {
			L,
			U,
			P: unsafe { Perm::new_unchecked(row_perm_fwd.into_boxed_slice(), row_perm_bwd.into_boxed_slice()) },
		}
	}

	/// returns the $L$ factor
	pub fn L(&self) -> MatRef<'_, T> {
		self.L.as_ref()
	}

	/// returns the $U$ factor
	pub fn U(&self) -> MatRef<'_, T> {
		self.U.as_ref()
	}

	/// returns the row pivoting permutation $P$
	pub fn P(&self) -> PermRef<'_, usize> {
		self.P.as_ref()
	}
}

impl<T: ComplexField> FullPivLu<T> {
	/// returns the $LU$ decomposition of $A$ with full pivoting
	#[track_caller]
	pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>) -> Self {
		let LU = A.to_owned();
		Self::new_imp(LU)
	}

	#[track_caller]
	fn new_imp(mut LU: Mat<T>) -> Self {
		let par = get_global_parallelism();

		let (m, n) = LU.shape();
		let mut row_perm_fwd = vec![0usize; m];
		let mut row_perm_bwd = vec![0usize; m];
		let mut col_perm_fwd = vec![0usize; n];
		let mut col_perm_bwd = vec![0usize; n];

		linalg::lu::full_pivoting::factor::lu_in_place(
			LU.as_mut(),
			&mut row_perm_fwd,
			&mut row_perm_bwd,
			&mut col_perm_fwd,
			&mut col_perm_bwd,
			par,
			MemStack::new(&mut MemBuffer::new(linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, T>(
				m,
				n,
				par,
				default(),
			))),
			default(),
		);

		let (L, U) = split_LU(LU);

		Self {
			L,
			U,
			P: unsafe { Perm::new_unchecked(row_perm_fwd.into_boxed_slice(), row_perm_bwd.into_boxed_slice()) },
			Q: unsafe { Perm::new_unchecked(col_perm_fwd.into_boxed_slice(), col_perm_bwd.into_boxed_slice()) },
		}
	}

	/// returns the factor $L$
	pub fn L(&self) -> MatRef<'_, T> {
		self.L.as_ref()
	}

	/// returns the factor $U$
	pub fn U(&self) -> MatRef<'_, T> {
		self.U.as_ref()
	}

	/// returns the row pivoting permutation $P$
	pub fn P(&self) -> PermRef<'_, usize> {
		self.P.as_ref()
	}

	/// returns the column pivoting permutation $P$
	pub fn Q(&self) -> PermRef<'_, usize> {
		self.Q.as_ref()
	}
}

impl<T: ComplexField> Qr<T> {
	/// returns the $QR$ decomposition of $A$
	#[track_caller]
	pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>) -> Self {
		let QR = A.to_owned();
		Self::new_imp(QR)
	}

	#[track_caller]
	fn new_imp(mut QR: Mat<T>) -> Self {
		let par = get_global_parallelism();

		let (m, n) = QR.shape();
		let size = Ord::min(m, n);

		let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(m, n);
		let mut Q_coeff = Mat::zeros(blocksize, size);

		linalg::qr::no_pivoting::factor::qr_in_place(
			QR.as_mut(),
			Q_coeff.as_mut(),
			par,
			MemStack::new(&mut MemBuffer::new(linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(
				m,
				n,
				blocksize,
				par,
				default(),
			))),
			default(),
		);

		let (Q_basis, R) = split_LU(QR);

		Self { Q_basis, Q_coeff, R }
	}

	/// returns the householder basis of $Q$
	pub fn Q_basis(&self) -> MatRef<'_, T> {
		self.Q_basis.as_ref()
	}

	/// returns the householder coefficients of $Q$
	pub fn Q_coeff(&self) -> MatRef<'_, T> {
		self.Q_coeff.as_ref()
	}

	/// returns the factor $R$
	pub fn R(&self) -> MatRef<'_, T> {
		self.R.as_ref()
	}

	/// returns the upper trapezoidal part of $R$
	pub fn thin_R(&self) -> MatRef<'_, T> {
		let size = Ord::min(self.nrows(), self.ncols());
		self.R.get(..size, ..)
	}

	/// computes the factor $Q$
	pub fn compute_Q(&self) -> Mat<T> {
		let mut Q = Mat::identity(self.nrows(), self.nrows());
		let par = get_global_parallelism();
		linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
			self.Q_basis(),
			self.Q_coeff(),
			Conj::No,
			Q.rb_mut(),
			par,
			MemStack::new(&mut MemBuffer::new(
				linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
					self.nrows(),
					self.Q_coeff.nrows(),
					self.nrows(),
				),
			)),
		);
		Q
	}

	/// computes the first $\min(\text{nrows}, \text{ncols})$ columns of the factor $Q$
	pub fn compute_thin_Q(&self) -> Mat<T> {
		let size = Ord::min(self.nrows(), self.ncols());
		let mut Q = Mat::identity(self.nrows(), size);
		let par = get_global_parallelism();
		linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
			self.Q_basis(),
			self.Q_coeff(),
			Conj::No,
			Q.rb_mut(),
			par,
			MemStack::new(&mut MemBuffer::new(
				linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(self.nrows(), self.Q_coeff.nrows(), size),
			)),
		);
		Q
	}
}

impl<T: ComplexField> ColPivQr<T> {
	/// returns the $QR$ decomposition of $A$ with column pivoting
	#[track_caller]
	pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>) -> Self {
		let QR = A.to_owned();
		Self::new_imp(QR)
	}

	#[track_caller]
	fn new_imp(mut QR: Mat<T>) -> Self {
		let par = get_global_parallelism();

		let (m, n) = QR.shape();
		let size = Ord::min(m, n);

		let mut col_perm_fwd = vec![0usize; n];
		let mut col_perm_bwd = vec![0usize; n];

		let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(m, n);
		let mut Q_coeff = Mat::zeros(blocksize, size);

		linalg::qr::col_pivoting::factor::qr_in_place(
			QR.as_mut(),
			Q_coeff.as_mut(),
			&mut col_perm_fwd,
			&mut col_perm_bwd,
			par,
			MemStack::new(&mut MemBuffer::new(linalg::qr::col_pivoting::factor::qr_in_place_scratch::<usize, T>(
				m,
				n,
				blocksize,
				par,
				default(),
			))),
			default(),
		);

		let (Q_basis, R) = split_LU(QR);

		Self {
			Q_basis,
			Q_coeff,
			R,
			P: unsafe { Perm::new_unchecked(col_perm_fwd.into_boxed_slice(), col_perm_bwd.into_boxed_slice()) },
		}
	}

	/// returns the householder basis of $Q$
	pub fn Q_basis(&self) -> MatRef<'_, T> {
		self.Q_basis.as_ref()
	}

	/// returns the householder coefficients of $Q$
	pub fn Q_coeff(&self) -> MatRef<'_, T> {
		self.Q_coeff.as_ref()
	}

	/// returns the factor $R$
	pub fn R(&self) -> MatRef<'_, T> {
		self.R.as_ref()
	}

	/// returns the upper trapezoidal part of $R$
	pub fn thin_R(&self) -> MatRef<'_, T> {
		let size = Ord::min(self.nrows(), self.ncols());
		self.R.get(..size, ..)
	}

	/// computes the factor $Q$
	pub fn compute_Q(&self) -> Mat<T> {
		let mut Q = Mat::identity(self.nrows(), self.nrows());
		let par = get_global_parallelism();
		linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
			self.Q_basis(),
			self.Q_coeff(),
			Conj::No,
			Q.rb_mut(),
			par,
			MemStack::new(&mut MemBuffer::new(
				linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
					self.nrows(),
					self.Q_coeff.nrows(),
					self.nrows(),
				),
			)),
		);
		Q
	}

	/// computes the first $\min(\text{nrows}, \text{ncols})$ columns of the factor $Q$
	pub fn compute_thin_Q(&self) -> Mat<T> {
		let size = Ord::min(self.nrows(), self.ncols());
		let mut Q = Mat::identity(self.nrows(), size);
		let par = get_global_parallelism();
		linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
			self.Q_basis(),
			self.Q_coeff(),
			Conj::No,
			Q.rb_mut(),
			par,
			MemStack::new(&mut MemBuffer::new(
				linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(self.nrows(), self.Q_coeff.nrows(), size),
			)),
		);
		Q
	}

	/// returns the column pivoting permutation $P$
	pub fn P(&self) -> PermRef<'_, usize> {
		self.P.as_ref()
	}
}

impl<T: ComplexField> Svd<T> {
	/// returns the svd of $A$
	#[track_caller]
	pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>) -> Result<Self, SvdError> {
		Self::new_imp(A.canonical(), Conj::get::<C>(), false)
	}

	/// returns the thin svd of $A$
	#[track_caller]
	pub fn new_thin<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>) -> Result<Self, SvdError> {
		Self::new_imp(A.canonical(), Conj::get::<C>(), true)
	}

	#[track_caller]
	fn new_imp(A: MatRef<'_, T>, conj: Conj, thin: bool) -> Result<Self, SvdError> {
		let par = get_global_parallelism();

		let (m, n) = A.shape();
		let size = Ord::min(m, n);

		let mut U = Mat::zeros(m, if thin { size } else { m });
		let mut V = Mat::zeros(n, if thin { size } else { n });
		let mut S = Diag::zeros(size);

		let compute = if thin { ComputeSvdVectors::Thin } else { ComputeSvdVectors::Full };

		linalg::svd::svd(
			A,
			S.as_mut(),
			Some(U.as_mut()),
			Some(V.as_mut()),
			par,
			MemStack::new(&mut MemBuffer::new(linalg::svd::svd_scratch::<T>(m, n, compute, compute, par, default()))),
			default(),
		)?;

		if conj == Conj::Yes {
			for c in U.col_iter_mut() {
				for x in c.iter_mut() {
					*x = math_utils::conj(x);
				}
			}
			for c in V.col_iter_mut() {
				for x in c.iter_mut() {
					*x = math_utils::conj(x);
				}
			}
		}

		Ok(Self { U, V, S })
	}

	/// returns the factor $U$
	pub fn U(&self) -> MatRef<'_, T> {
		self.U.as_ref()
	}

	/// returns the factor $V$
	pub fn V(&self) -> MatRef<'_, T> {
		self.V.as_ref()
	}

	/// returns the factor $S$
	pub fn S(&self) -> DiagRef<'_, T> {
		self.S.as_ref()
	}
}

impl<T: ComplexField> SelfAdjointEigen<T> {
	/// returns the eigendecomposition of $A$, assuming it is self-adjoint
	#[track_caller]
	pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>, side: Side) -> Result<Self, EvdError> {
		assert!(A.nrows() == A.ncols());

		match side {
			Side::Lower => Self::new_imp(A.canonical(), Conj::get::<C>()),
			Side::Upper => Self::new_imp(A.adjoint().canonical(), Conj::get::<C::Conj>()),
		}
	}

	#[track_caller]
	fn new_imp(A: MatRef<'_, T>, conj: Conj) -> Result<Self, EvdError> {
		let par = get_global_parallelism();

		let n = A.nrows();

		let mut U = Mat::zeros(n, n);
		let mut S = Diag::zeros(n);

		linalg::evd::self_adjoint_evd(
			A,
			S.as_mut(),
			Some(U.as_mut()),
			par,
			MemStack::new(&mut MemBuffer::new(linalg::evd::self_adjoint_evd_scratch::<T>(
				n,
				linalg::evd::ComputeEigenvectors::Yes,
				par,
				default(),
			))),
			default(),
		)?;

		if conj == Conj::Yes {
			for c in U.col_iter_mut() {
				for x in c.iter_mut() {
					*x = math_utils::conj(x);
				}
			}
		}

		Ok(Self { U, S })
	}

	/// returns the factor $U$
	pub fn U(&self) -> MatRef<'_, T> {
		self.U.as_ref()
	}

	/// returns the factor $S$
	pub fn S(&self) -> DiagRef<'_, T> {
		self.S.as_ref()
	}
}

impl<T: RealField> Eigen<T> {
	/// returns the eigendecomposition of $A$
	#[track_caller]
	pub fn new<C: Conjugate<Canonical = Complex<T>>>(A: MatRef<'_, C>) -> Result<Self, EvdError> {
		assert!(A.nrows() == A.ncols());
		Self::new_imp(A.canonical(), Conj::get::<C>())
	}

	/// returns the eigendecomposition of $A$
	#[track_caller]
	pub fn new_from_real(A: MatRef<'_, T>) -> Result<Self, EvdError> {
		assert!(A.nrows() == A.ncols());

		let par = get_global_parallelism();

		let n = A.nrows();

		let mut U_real = Mat::zeros(n, n);
		let mut S_re = Diag::zeros(n);
		let mut S_im = Diag::zeros(n);

		linalg::evd::evd_real(
			A,
			S_re.as_mut(),
			S_im.as_mut(),
			None,
			Some(U_real.as_mut()),
			par,
			MemStack::new(&mut MemBuffer::new(linalg::evd::evd_scratch::<T>(
				n,
				linalg::evd::ComputeEigenvectors::No,
				linalg::evd::ComputeEigenvectors::Yes,
				par,
				default(),
			))),
			default(),
		)?;

		let mut U = Mat::zeros(n, n);
		let mut S = Diag::zeros(n);

		let mut j = 0;
		while j < n {
			if S_im[j] == zero() {
				S[j] = Complex::new(S_re[j].clone(), zero());

				for i in 0..n {
					U[(i, j)] = Complex::new(U_real[(i, j)].clone(), zero());
				}

				j += 1;
			} else {
				S[j] = Complex::new(S_re[j].clone(), S_im[j].clone());
				S[j + 1] = Complex::new(S_re[j].clone(), neg(&S_im[j]));

				for i in 0..n {
					U[(i, j)] = Complex::new(U_real[(i, j)].clone(), U_real[(i, j + 1)].clone());
					U[(i, j + 1)] = Complex::new(U_real[(i, j)].clone(), neg(&U_real[(i, j + 1)]));
				}

				j += 2;
			}
		}

		Ok(Self { U, S })
	}

	fn new_imp(A: MatRef<'_, Complex<T>>, conj: Conj) -> Result<Self, EvdError> {
		let par = get_global_parallelism();

		let n = A.nrows();

		let mut U = Mat::zeros(n, n);
		let mut S = Diag::zeros(n);

		linalg::evd::evd_cplx(
			A,
			S.as_mut(),
			None,
			Some(U.as_mut()),
			par,
			MemStack::new(&mut MemBuffer::new(linalg::evd::evd_scratch::<Complex<T>>(
				n,
				linalg::evd::ComputeEigenvectors::No,
				linalg::evd::ComputeEigenvectors::Yes,
				par,
				default(),
			))),
			default(),
		)?;

		if conj == Conj::Yes {
			for c in U.col_iter_mut() {
				for x in c.iter_mut() {
					*x = math_utils::conj(x);
				}
			}
		}

		Ok(Self { U, S })
	}

	/// returns the factor $U$
	pub fn U(&self) -> MatRef<'_, Complex<T>> {
		self.U.as_ref()
	}

	/// returns the factor $S$
	pub fn S(&self) -> DiagRef<'_, Complex<T>> {
		self.S.as_ref()
	}
}

impl<T: ComplexField> ShapeCore for Llt<T> {
	#[inline]
	fn nrows(&self) -> usize {
		self.L().nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		self.L().ncols()
	}
}
impl<T: ComplexField> ShapeCore for Ldlt<T> {
	#[inline]
	fn nrows(&self) -> usize {
		self.L().nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		self.L().ncols()
	}
}
impl<T: ComplexField> ShapeCore for Lblt<T> {
	#[inline]
	fn nrows(&self) -> usize {
		self.L().nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		self.L().ncols()
	}
}
impl<T: ComplexField> ShapeCore for PartialPivLu<T> {
	#[inline]
	fn nrows(&self) -> usize {
		self.L().nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		self.U().ncols()
	}
}
impl<T: ComplexField> ShapeCore for FullPivLu<T> {
	#[inline]
	fn nrows(&self) -> usize {
		self.L().nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		self.U().ncols()
	}
}
impl<T: ComplexField> ShapeCore for Qr<T> {
	#[inline]
	fn nrows(&self) -> usize {
		self.Q_basis().nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		self.R().ncols()
	}
}
impl<T: ComplexField> ShapeCore for ColPivQr<T> {
	#[inline]
	fn nrows(&self) -> usize {
		self.Q_basis().nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		self.R().ncols()
	}
}
impl<T: ComplexField> ShapeCore for Svd<T> {
	#[inline]
	fn nrows(&self) -> usize {
		self.U().nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		self.V().nrows()
	}
}
impl<T: ComplexField> ShapeCore for SelfAdjointEigen<T> {
	#[inline]
	fn nrows(&self) -> usize {
		self.U().nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		self.U().nrows()
	}
}
impl<T: RealField> ShapeCore for Eigen<T> {
	#[inline]
	fn nrows(&self) -> usize {
		self.U().nrows()
	}

	#[inline]
	fn ncols(&self) -> usize {
		self.U().nrows()
	}
}

impl<T: ComplexField> SolveCore<T> for Llt<T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		let mut mem = MemBuffer::new(linalg::cholesky::llt::solve::solve_in_place_scratch::<T>(
			self.L.nrows(),
			rhs.ncols(),
			par,
		));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::llt::solve::solve_in_place_with_conj(self.L.as_ref(), conj, rhs, par, stack);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		let mut mem = MemBuffer::new(linalg::cholesky::llt::solve::solve_in_place_scratch::<T>(
			self.L.nrows(),
			rhs.ncols(),
			par,
		));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::llt::solve::solve_in_place_with_conj(self.L.as_ref(), conj.compose(Conj::Yes), rhs, par, stack);
	}
}

#[math]
fn make_self_adjoint<T: ComplexField>(mut A: MatMut<'_, T>) {
	assert!(A.nrows() == A.ncols());
	let n = A.nrows();
	for j in 0..n {
		A[(j, j)] = from_real(real(A[(j, j)]));
		for i in 0..j {
			A[(i, j)] = conj(A[(j, i)]);
		}
	}
}

impl<T: ComplexField> DenseSolveCore<T> for Llt<T> {
	#[track_caller]
	fn reconstruct(&self) -> Mat<T> {
		let par = get_global_parallelism();

		let n = self.L.nrows();
		let mut out = Mat::zeros(n, n);

		let mut mem = MemBuffer::new(linalg::cholesky::llt::reconstruct::reconstruct_scratch::<T>(n, par));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::llt::reconstruct::reconstruct(out.as_mut(), self.L(), par, stack);

		make_self_adjoint(out.as_mut());
		out
	}

	#[track_caller]
	fn inverse(&self) -> Mat<T> {
		let par = get_global_parallelism();

		let n = self.L.nrows();
		let mut out = Mat::zeros(n, n);

		let mut mem = MemBuffer::new(linalg::cholesky::llt::inverse::inverse_scratch::<T>(n, par));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::llt::inverse::inverse(out.as_mut(), self.L(), par, stack);

		make_self_adjoint(out.as_mut());
		out
	}
}

impl<T: ComplexField> SolveCore<T> for Ldlt<T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		let mut mem = MemBuffer::new(linalg::cholesky::ldlt::solve::solve_in_place_scratch::<T>(
			self.L.nrows(),
			rhs.ncols(),
			par,
		));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::ldlt::solve::solve_in_place_with_conj(self.L.as_ref(), self.D.as_ref(), conj, rhs, par, stack);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		let mut mem = MemBuffer::new(linalg::cholesky::ldlt::solve::solve_in_place_scratch::<T>(
			self.L.nrows(),
			rhs.ncols(),
			par,
		));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::ldlt::solve::solve_in_place_with_conj(self.L(), self.D(), conj.compose(Conj::Yes), rhs, par, stack);
	}
}

impl<T: ComplexField> DenseSolveCore<T> for Ldlt<T> {
	#[track_caller]
	fn reconstruct(&self) -> Mat<T> {
		let par = get_global_parallelism();

		let n = self.L.nrows();
		let mut out = Mat::zeros(n, n);

		let mut mem = MemBuffer::new(linalg::cholesky::ldlt::reconstruct::reconstruct_scratch::<T>(n, par));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::ldlt::reconstruct::reconstruct(out.as_mut(), self.L(), self.D(), par, stack);

		make_self_adjoint(out.as_mut());
		out
	}

	#[track_caller]
	fn inverse(&self) -> Mat<T> {
		let par = get_global_parallelism();

		let n = self.L.nrows();
		let mut out = Mat::zeros(n, n);

		let mut mem = MemBuffer::new(linalg::cholesky::ldlt::inverse::inverse_scratch::<T>(n, par));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::ldlt::inverse::inverse(out.as_mut(), self.L(), self.D(), par, stack);

		make_self_adjoint(out.as_mut());
		out
	}
}

impl<T: ComplexField> SolveCore<T> for Lblt<T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		let mut mem = MemBuffer::new(linalg::cholesky::bunch_kaufman::solve::solve_in_place_scratch::<usize, T>(
			self.L.nrows(),
			rhs.ncols(),
			par,
		));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::bunch_kaufman::solve::solve_in_place_with_conj(
			self.L.as_ref(),
			self.B_diag(),
			self.B_subdiag(),
			conj,
			self.P(),
			rhs,
			par,
			stack,
		);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		let mut mem = MemBuffer::new(linalg::cholesky::bunch_kaufman::solve::solve_in_place_scratch::<usize, T>(
			self.L.nrows(),
			rhs.ncols(),
			par,
		));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::bunch_kaufman::solve::solve_in_place_with_conj(
			self.L(),
			self.B_diag(),
			self.B_subdiag(),
			conj.compose(Conj::Yes),
			self.P(),
			rhs,
			par,
			stack,
		);
	}
}

impl<T: ComplexField> DenseSolveCore<T> for Lblt<T> {
	#[track_caller]
	fn reconstruct(&self) -> Mat<T> {
		let par = get_global_parallelism();

		let n = self.L.nrows();
		let mut out = Mat::zeros(n, n);

		let mut mem = MemBuffer::new(linalg::cholesky::bunch_kaufman::reconstruct::reconstruct_scratch::<usize, T>(n, par));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::bunch_kaufman::reconstruct::reconstruct(out.as_mut(), self.L(), self.B_diag(), self.B_subdiag(), self.P(), par, stack);

		make_self_adjoint(out.as_mut());
		out
	}

	#[track_caller]
	fn inverse(&self) -> Mat<T> {
		let par = get_global_parallelism();

		let n = self.L.nrows();
		let mut out = Mat::zeros(n, n);

		let mut mem = MemBuffer::new(linalg::cholesky::bunch_kaufman::inverse::inverse_scratch::<usize, T>(n, par));
		let stack = MemStack::new(&mut mem);

		linalg::cholesky::bunch_kaufman::inverse::inverse(out.as_mut(), self.L(), self.B_diag(), self.B_subdiag(), self.P(), par, stack);

		make_self_adjoint(out.as_mut());
		out
	}
}

impl<T: ComplexField> SolveCore<T> for PartialPivLu<T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.nrows() == rhs.nrows(),));

		let k = rhs.ncols();

		linalg::lu::partial_pivoting::solve::solve_in_place_with_conj(
			self.L(),
			self.U(),
			self.P(),
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(
				linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<usize, T>(self.nrows(), k, par),
			)),
		);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.ncols() == rhs.nrows(),));

		let k = rhs.ncols();

		linalg::lu::partial_pivoting::solve::solve_transpose_in_place_with_conj(
			self.L(),
			self.U(),
			self.P(),
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(
				linalg::lu::partial_pivoting::solve::solve_transpose_in_place_scratch::<usize, T>(self.nrows(), k, par),
			)),
		);
	}
}

impl<T: ComplexField> DenseSolveCore<T> for PartialPivLu<T> {
	fn reconstruct(&self) -> Mat<T> {
		let par = get_global_parallelism();
		let m = self.nrows();
		let n = self.ncols();

		let mut out = Mat::zeros(m, n);

		linalg::lu::partial_pivoting::reconstruct::reconstruct(
			out.as_mut(),
			self.L(),
			self.U(),
			self.P(),
			par,
			MemStack::new(&mut MemBuffer::new(linalg::lu::partial_pivoting::reconstruct::reconstruct_scratch::<
				usize,
				T,
			>(m, n, par))),
		);

		out
	}

	#[track_caller]
	fn inverse(&self) -> Mat<T> {
		let par = get_global_parallelism();

		assert!(self.nrows() == self.ncols());

		let n = self.ncols();

		let mut out = Mat::zeros(n, n);

		linalg::lu::partial_pivoting::inverse::inverse(
			out.as_mut(),
			self.L(),
			self.U(),
			self.P(),
			par,
			MemStack::new(&mut MemBuffer::new(linalg::lu::partial_pivoting::inverse::inverse_scratch::<usize, T>(
				n, par,
			))),
		);

		out
	}
}

impl<T: ComplexField> SolveCore<T> for FullPivLu<T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.nrows() == rhs.nrows(),));

		let k = rhs.ncols();

		linalg::lu::full_pivoting::solve::solve_in_place_with_conj(
			self.L(),
			self.U(),
			self.P(),
			self.Q(),
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, T>(
				self.nrows(),
				k,
				par,
			))),
		);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.ncols() == rhs.nrows(),));

		let k = rhs.ncols();

		linalg::lu::full_pivoting::solve::solve_transpose_in_place_with_conj(
			self.L(),
			self.U(),
			self.P(),
			self.Q(),
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(linalg::lu::full_pivoting::solve::solve_transpose_in_place_scratch::<
				usize,
				T,
			>(self.nrows(), k, par))),
		);
	}
}

impl<T: ComplexField> DenseSolveCore<T> for FullPivLu<T> {
	fn reconstruct(&self) -> Mat<T> {
		let par = get_global_parallelism();
		let m = self.nrows();
		let n = self.ncols();

		let mut out = Mat::zeros(m, n);

		linalg::lu::full_pivoting::reconstruct::reconstruct(
			out.as_mut(),
			self.L(),
			self.U(),
			self.P(),
			self.Q(),
			par,
			MemStack::new(&mut MemBuffer::new(
				linalg::lu::full_pivoting::reconstruct::reconstruct_scratch::<usize, T>(m, n, par),
			)),
		);

		out
	}

	#[track_caller]
	fn inverse(&self) -> Mat<T> {
		let par = get_global_parallelism();

		assert!(self.nrows() == self.ncols());

		let n = self.ncols();

		let mut out = Mat::zeros(n, n);

		linalg::lu::full_pivoting::inverse::inverse(
			out.as_mut(),
			self.L(),
			self.U(),
			self.P(),
			self.Q(),
			par,
			MemStack::new(&mut MemBuffer::new(linalg::lu::full_pivoting::inverse::inverse_scratch::<usize, T>(
				n, par,
			))),
		);

		out
	}
}

impl<T: ComplexField> SolveCore<T> for Qr<T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.nrows() == rhs.nrows(),));

		let n = self.nrows();
		let blocksize = self.Q_coeff().nrows();
		let k = rhs.ncols();

		linalg::qr::no_pivoting::solve::solve_in_place_with_conj(
			self.Q_basis(),
			self.Q_coeff(),
			self.R(),
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(linalg::qr::no_pivoting::solve::solve_in_place_scratch::<T>(
				n, blocksize, k, par,
			))),
		);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.ncols() == rhs.nrows(),));

		let n = self.nrows();
		let blocksize = self.Q_coeff().nrows();
		let k = rhs.ncols();

		linalg::qr::no_pivoting::solve::solve_transpose_in_place_with_conj(
			self.Q_basis(),
			self.Q_coeff(),
			self.R(),
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(
				linalg::qr::no_pivoting::solve::solve_transpose_in_place_scratch::<T>(n, blocksize, k, par),
			)),
		);
	}
}

impl<T: ComplexField> SolveLstsqCore<T> for Qr<T> {
	#[track_caller]
	fn solve_lstsq_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == rhs.nrows(), self.nrows() >= self.ncols(),));

		let m = self.nrows();
		let n = self.ncols();
		let blocksize = self.Q_coeff().nrows();
		let k = rhs.ncols();

		linalg::qr::no_pivoting::solve::solve_lstsq_in_place_with_conj(
			self.Q_basis(),
			self.Q_coeff(),
			self.R(),
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(linalg::qr::no_pivoting::solve::solve_lstsq_in_place_scratch::<T>(
				m, n, blocksize, k, par,
			))),
		);
	}
}

impl<T: ComplexField> DenseSolveCore<T> for Qr<T> {
	fn reconstruct(&self) -> Mat<T> {
		let par = get_global_parallelism();
		let m = self.nrows();
		let n = self.ncols();
		let blocksize = self.Q_coeff().nrows();

		let mut out = Mat::zeros(m, n);

		linalg::qr::no_pivoting::reconstruct::reconstruct(
			out.as_mut(),
			self.Q_basis(),
			self.Q_coeff(),
			self.R(),
			par,
			MemStack::new(&mut MemBuffer::new(linalg::qr::no_pivoting::reconstruct::reconstruct_scratch::<T>(
				m, n, blocksize, par,
			))),
		);

		out
	}

	fn inverse(&self) -> Mat<T> {
		let par = get_global_parallelism();
		assert!(self.nrows() == self.ncols());

		let n = self.ncols();
		let blocksize = self.Q_coeff().nrows();

		let mut out = Mat::zeros(n, n);

		linalg::qr::no_pivoting::inverse::inverse(
			out.as_mut(),
			self.Q_basis(),
			self.Q_coeff(),
			self.R(),
			par,
			MemStack::new(&mut MemBuffer::new(linalg::qr::no_pivoting::inverse::inverse_scratch::<T>(
				n, blocksize, par,
			))),
		);

		out
	}
}

impl<T: ComplexField> SolveCore<T> for ColPivQr<T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.nrows() == rhs.nrows(),));

		let n = self.nrows();
		let blocksize = self.Q_coeff().nrows();
		let k = rhs.ncols();

		linalg::qr::col_pivoting::solve::solve_in_place_with_conj(
			self.Q_basis(),
			self.Q_coeff(),
			self.R(),
			self.P(),
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(linalg::qr::col_pivoting::solve::solve_in_place_scratch::<usize, T>(
				n, blocksize, k, par,
			))),
		);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.ncols() == rhs.nrows(),));

		let n = self.nrows();
		let blocksize = self.Q_coeff().nrows();
		let k = rhs.ncols();

		linalg::qr::col_pivoting::solve::solve_transpose_in_place_with_conj(
			self.Q_basis(),
			self.Q_coeff(),
			self.R(),
			self.P(),
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(linalg::qr::col_pivoting::solve::solve_transpose_in_place_scratch::<
				usize,
				T,
			>(n, blocksize, k, par))),
		);
	}
}

impl<T: ComplexField> SolveLstsqCore<T> for ColPivQr<T> {
	#[track_caller]
	fn solve_lstsq_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == rhs.nrows(), self.nrows() >= self.ncols(),));

		let m = self.nrows();
		let n = self.ncols();
		let blocksize = self.Q_coeff().nrows();
		let k = rhs.ncols();

		linalg::qr::col_pivoting::solve::solve_lstsq_in_place_with_conj(
			self.Q_basis(),
			self.Q_coeff(),
			self.R(),
			self.P(),
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(linalg::qr::col_pivoting::solve::solve_lstsq_in_place_scratch::<
				usize,
				T,
			>(m, n, blocksize, k, par))),
		);
	}
}

impl<T: ComplexField> DenseSolveCore<T> for ColPivQr<T> {
	fn reconstruct(&self) -> Mat<T> {
		let par = get_global_parallelism();
		let m = self.nrows();
		let n = self.ncols();
		let blocksize = self.Q_coeff().nrows();

		let mut out = Mat::zeros(m, n);

		linalg::qr::col_pivoting::reconstruct::reconstruct(
			out.as_mut(),
			self.Q_basis(),
			self.Q_coeff(),
			self.R(),
			self.P(),
			par,
			MemStack::new(&mut MemBuffer::new(
				linalg::qr::col_pivoting::reconstruct::reconstruct_scratch::<usize, T>(m, n, blocksize, par),
			)),
		);

		out
	}

	fn inverse(&self) -> Mat<T> {
		let par = get_global_parallelism();
		assert!(self.nrows() == self.ncols());

		let n = self.ncols();
		let blocksize = self.Q_coeff().nrows();

		let mut out = Mat::zeros(n, n);

		linalg::qr::col_pivoting::inverse::inverse(
			out.as_mut(),
			self.Q_basis(),
			self.Q_coeff(),
			self.R(),
			self.P(),
			par,
			MemStack::new(&mut MemBuffer::new(linalg::qr::col_pivoting::inverse::inverse_scratch::<usize, T>(
				n, blocksize, par,
			))),
		);

		out
	}
}

impl<T: ComplexField> SolveCore<T> for Svd<T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.nrows() == rhs.nrows(),));

		let mut rhs = rhs;
		let n = self.nrows();
		let k = rhs.ncols();
		let mut tmp = Mat::zeros(n, k);

		linalg::matmul::matmul_with_conj(
			tmp.as_mut(),
			Accum::Replace,
			self.U().transpose(),
			conj.compose(Conj::Yes),
			rhs.as_ref(),
			Conj::No,
			one(),
			par,
		);

		for j in 0..k {
			for i in 0..n {
				let s = recip(&real(&self.S()[i]));
				tmp[(i, j)] = mul_real(&tmp[(i, j)], &s);
			}
		}

		linalg::matmul::matmul_with_conj(rhs.as_mut(), Accum::Replace, self.V(), conj, tmp.as_ref(), Conj::No, one(), par);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.ncols() == rhs.nrows(),));

		let mut rhs = rhs;
		let n = self.nrows();
		let k = rhs.ncols();
		let mut tmp = Mat::zeros(n, k);

		linalg::matmul::matmul_with_conj(
			tmp.as_mut(),
			Accum::Replace,
			self.V().transpose(),
			conj,
			rhs.as_ref(),
			Conj::No,
			one(),
			par,
		);

		for j in 0..k {
			for i in 0..n {
				let s = recip(&real(&self.S()[i]));
				tmp[(i, j)] = mul_real(&tmp[(i, j)], &s);
			}
		}

		linalg::matmul::matmul_with_conj(
			rhs.as_mut(),
			Accum::Replace,
			self.U(),
			conj.compose(Conj::Yes),
			tmp.as_ref(),
			Conj::No,
			one(),
			par,
		);
	}
}

impl<T: ComplexField> SolveLstsqCore<T> for Svd<T> {
	#[track_caller]
	fn solve_lstsq_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == rhs.nrows(), self.nrows() >= self.ncols(),));

		let m = self.nrows();
		let n = self.ncols();

		let size = Ord::min(m, n);

		let U = self.U().get(.., ..size);
		let V = self.V().get(.., ..size);

		let k = rhs.ncols();

		let mut tmp = Mat::zeros(size, k);

		linalg::matmul::matmul_with_conj(
			tmp.as_mut(),
			Accum::Replace,
			U.transpose(),
			conj.compose(Conj::Yes),
			rhs.as_ref(),
			Conj::No,
			one(),
			par,
		);

		for j in 0..k {
			for i in 0..size {
				let s = recip(&real(&self.S()[i]));
				tmp[(i, j)] = mul_real(&tmp[(i, j)], &s);
			}
		}

		linalg::matmul::matmul_with_conj(rhs.get_mut(..size, ..), Accum::Replace, V, conj, tmp.as_ref(), Conj::No, one(), par);
	}
}

impl<T: ComplexField> DenseSolveCore<T> for Svd<T> {
	fn reconstruct(&self) -> Mat<T> {
		let par = get_global_parallelism();
		let m = self.nrows();
		let n = self.ncols();

		let size = Ord::min(m, n);

		let U = self.U().get(.., ..size);
		let V = self.V().get(.., ..size);
		let S = self.S();

		let mut UxS = Mat::zeros(m, size);
		for j in 0..size {
			let s = real(&S[j]);
			for i in 0..m {
				UxS[(i, j)] = mul_real(&U[(i, j)], &s);
			}
		}

		let mut out = Mat::zeros(m, n);

		linalg::matmul::matmul(out.as_mut(), Accum::Replace, UxS.as_ref(), V.adjoint(), one(), par);

		out
	}

	#[track_caller]
	fn inverse(&self) -> Mat<T> {
		let par = get_global_parallelism();

		assert!(self.nrows() == self.ncols());
		let n = self.nrows();

		let U = self.U();
		let V = self.V();
		let S = self.S();

		let mut VxS = Mat::zeros(n, n);
		for j in 0..n {
			let s = recip(&real(&S[j]));

			for i in 0..n {
				VxS[(i, j)] = mul_real(&V[(i, j)], &s);
			}
		}

		let mut out = Mat::zeros(n, n);

		linalg::matmul::matmul(out.as_mut(), Accum::Replace, VxS.as_ref(), U.adjoint(), one(), par);

		out
	}
}

impl<T: ComplexField> SolveCore<T> for SelfAdjointEigen<T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.nrows() == rhs.nrows(),));

		let mut rhs = rhs;
		let n = self.nrows();
		let k = rhs.ncols();
		let mut tmp = Mat::zeros(n, k);

		linalg::matmul::matmul_with_conj(
			tmp.as_mut(),
			Accum::Replace,
			self.U().transpose(),
			conj.compose(Conj::Yes),
			rhs.as_ref(),
			Conj::No,
			one(),
			par,
		);

		for j in 0..k {
			for i in 0..n {
				let s = recip(&real(&self.S()[i]));
				tmp[(i, j)] = mul_real(&tmp[(i, j)], &s);
			}
		}

		linalg::matmul::matmul_with_conj(rhs.as_mut(), Accum::Replace, self.U(), conj, tmp.as_ref(), Conj::No, one(), par);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();

		assert!(all(self.nrows() == self.ncols(), self.ncols() == rhs.nrows(),));

		let mut rhs = rhs;
		let n = self.nrows();
		let k = rhs.ncols();
		let mut tmp = Mat::zeros(n, k);

		linalg::matmul::matmul_with_conj(
			tmp.as_mut(),
			Accum::Replace,
			self.U().transpose(),
			conj,
			rhs.as_ref(),
			Conj::No,
			one(),
			par,
		);

		for j in 0..k {
			for i in 0..n {
				let s = recip(&real(&self.S()[i]));
				tmp[(i, j)] = mul_real(&tmp[(i, j)], &s);
			}
		}

		linalg::matmul::matmul_with_conj(
			rhs.as_mut(),
			Accum::Replace,
			self.U(),
			conj.compose(Conj::Yes),
			tmp.as_ref(),
			Conj::No,
			one(),
			par,
		);
	}
}

impl<T: ComplexField> DenseSolveCore<T> for SelfAdjointEigen<T> {
	fn reconstruct(&self) -> Mat<T> {
		let par = get_global_parallelism();
		let m = self.nrows();
		let n = self.ncols();

		let size = Ord::min(m, n);

		let U = self.U().get(.., ..size);
		let V = self.U().get(.., ..size);
		let S = self.S();

		let mut UxS = Mat::zeros(m, size);
		for j in 0..size {
			let s = real(&S[j]);
			for i in 0..m {
				UxS[(i, j)] = mul_real(&U[(i, j)], &s);
			}
		}

		let mut out = Mat::zeros(m, n);

		linalg::matmul::matmul(out.as_mut(), Accum::Replace, UxS.as_ref(), V.adjoint(), one(), par);

		out
	}

	fn inverse(&self) -> Mat<T> {
		let par = get_global_parallelism();

		assert!(self.nrows() == self.ncols());
		let n = self.nrows();

		let U = self.U();
		let V = self.U();
		let S = self.S();

		let mut VxS = Mat::zeros(n, n);
		for j in 0..n {
			let s = recip(&real(&S[j]));

			for i in 0..n {
				VxS[(i, j)] = mul_real(&V[(i, j)], &s);
			}
		}

		let mut out = Mat::zeros(n, n);

		linalg::matmul::matmul(out.as_mut(), Accum::Replace, VxS.as_ref(), U.adjoint(), one(), par);

		out
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;

	#[track_caller]
	fn test_solver(A: MatRef<'_, c64>, A_dec: impl SolveCore<c64>) {
		#[track_caller]
		fn test_solver_imp(A: MatRef<'_, c64>, A_dec: &dyn SolveCore<c64>) {
			let rng = &mut StdRng::seed_from_u64(0xC0FFEE);

			let n = A.nrows();
			let approx_eq = CwiseMat(ApproxEq::eps() * 128.0 * (n as f64));

			let k = 3;

			let ref R = CwiseMatDistribution {
				nrows: n,
				ncols: k,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			let ref L = CwiseMatDistribution {
				nrows: k,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			assert!(A * A_dec.solve(R) ~ R);
			assert!(A.conjugate() * A_dec.solve_conjugate(R) ~ R);
			assert!(A.transpose() * A_dec.solve_transpose(R) ~ R);
			assert!(A.adjoint() * A_dec.solve_adjoint(R) ~ R);

			assert!(A_dec.rsolve(L) * A ~ L);
			assert!(A_dec.rsolve_conjugate(L) * A.conjugate() ~ L);
			assert!(A_dec.rsolve_transpose(L) * A.transpose() ~ L);
			assert!(A_dec.rsolve_adjoint(L) * A.adjoint() ~ L);
		}

		test_solver_imp(A, &A_dec)
	}

	#[test]
	fn test_all_solvers() {
		let rng = &mut StdRng::seed_from_u64(0);
		let n = 50;

		let ref A = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);
		let A = A.rb();

		test_solver(A, A.partial_piv_lu());
		test_solver(A, A.full_piv_lu());
		test_solver(A, A.qr());
		test_solver(A, A.col_piv_qr());
		test_solver(A, A.svd().unwrap());

		{
			let ref A = A * A.adjoint();
			let A = A.rb();
			test_solver(A, A.llt(Side::Lower).unwrap());
			test_solver(A, A.ldlt(Side::Lower).unwrap());
		}

		{
			let ref A = A + A.adjoint();
			let A = A.rb();
			test_solver(A, A.lblt(Side::Lower));
			test_solver(A, A.self_adjoint_eigen(Side::Lower).unwrap());
		}
	}

	#[test]
	fn test_eigen_cplx() {
		let rng = &mut StdRng::seed_from_u64(0);
		let n = 50;

		let A = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand::<Mat<c64>>(rng);

		let n = A.nrows();
		let approx_eq = CwiseMat(ApproxEq::eps() * 128.0 * (n as f64));

		let evd = A.eigen().unwrap();
		let e = A.eigenvalues().unwrap();
		assert!(&A * evd.U() ~ evd.U() * evd.S());
		assert!(evd.S().column_vector() ~ ColRef::from_slice(&e));
	}

	#[test]
	fn test_eigen_real() {
		let rng = &mut StdRng::seed_from_u64(0);
		let n = 50;

		let A = CwiseMatDistribution {
			nrows: n,
			ncols: n,
			dist: StandardNormal,
		}
		.rand::<Mat<f64>>(rng);

		let n = A.nrows();
		let approx_eq = CwiseMat(ApproxEq::eps() * 128.0 * (n as f64));

		let evd = A.eigen_from_real().unwrap();
		let e = A.eigenvalues_from_real().unwrap();

		let A = Mat::from_fn(A.nrows(), A.ncols(), |i, j| c64::from(A[(i, j)]));

		assert!(&A * evd.U() ~ evd.U() * evd.S());
		assert!(evd.S().column_vector() ~ ColRef::from_slice(&e));
	}

	#[test]
	fn test_svd_solver_for_rectangular_matrix() {
		#[rustfmt::skip]
    	let A = crate::mat![
    	    [4.,   5.,   7.],
    	    [8.,   8.,   2.],
    	    [4.,   0.,   9.],
    	    [2.,   6.,   2.],
    	    [0.,   6.,   0.],
    	];
		#[rustfmt::skip]
    	let B = crate::mat![
        	[105.,    49.],
        	[ 98.,    54.],
        	[113.,    35.],
        	[ 46.,    34.],
        	[ 12.,    24.],
     	];

		#[rustfmt::skip]
	    let X_true= crate::mat![
	      [8.,   2.],
	      [2.,   4.],
	      [9.,   3.],
	    ];

		let approx_eq = CwiseMat(ApproxEq::eps() * 128.0 * (A.nrows() as f64));
		let svd = A.svd().unwrap();
		let mut X = B.cloned();
		svd.solve_lstsq_in_place_with_conj(crate::Conj::No, X.as_mat_mut());
		assert!(X.get(..X_true.nrows(),..) ~ X_true);
	}
}
