use crate::get_global_parallelism;
use crate::internal_prelude_sp::*;
use crate::linalg::solvers::{ShapeCore, SolveCore, SolveLstsqCore};
use linalg_sp::{LltError, LuError};

/// Reference-counted sparse symbolic LLT factorization.
#[derive(Debug, Clone)]
pub struct SymbolicLlt<I> {
	inner: alloc::sync::Arc<linalg_sp::cholesky::SymbolicCholesky<I>>,
}

/// Sparse LLT factorization.
#[derive(Debug, Clone)]
pub struct Llt<I, T> {
	symbolic: SymbolicLlt<I>,
	numeric: alloc::vec::Vec<T>,
}

/// Reference-counted sparse symbolic QR factorization.
#[derive(Debug, Clone)]
pub struct SymbolicQr<I> {
	inner: alloc::sync::Arc<linalg_sp::qr::SymbolicQr<I>>,
}

/// Sparse Qr factorization.
#[derive(Debug, Clone)]
pub struct Qr<I, T> {
	symbolic: SymbolicQr<I>,
	indices: alloc::vec::Vec<I>,
	numeric: alloc::vec::Vec<T>,
}

/// Reference-counted sparse symbolic LU factorization.
#[derive(Debug, Clone)]
pub struct SymbolicLu<I> {
	inner: alloc::sync::Arc<linalg_sp::lu::SymbolicLu<I>>,
}

/// Sparse Qr factorization.
#[derive(Debug, Clone)]
pub struct Lu<I, T> {
	symbolic: SymbolicLu<I>,
	numeric: linalg_sp::lu::NumericLu<I, T>,
}

impl<I: Index> SymbolicLlt<I> {
	/// Returns the symbolic LLT factorization of the input matrix.
	///
	/// Only the provided side is accessed.
	#[track_caller]
	pub fn try_new(mat: SymbolicSparseColMatRef<'_, I>, side: Side) -> Result<Self, FaerError> {
		Ok(Self {
			inner: alloc::sync::Arc::new(linalg_sp::cholesky::factorize_symbolic_cholesky(
				mat,
				side,
				Default::default(),
				Default::default(),
			)?),
		})
	}
}

impl<I: Index> SymbolicQr<I> {
	/// Returns the symbolic QR factorization of the input matrix.
	#[track_caller]
	pub fn try_new(mat: SymbolicSparseColMatRef<'_, I>) -> Result<Self, FaerError> {
		Ok(Self {
			inner: alloc::sync::Arc::new(linalg_sp::qr::factorize_symbolic_qr(mat, Default::default())?),
		})
	}
}

impl<I: Index> SymbolicLu<I> {
	/// Returns the symbolic LU factorization of the input matrix.
	#[track_caller]
	pub fn try_new(mat: SymbolicSparseColMatRef<'_, I>) -> Result<Self, FaerError> {
		Ok(Self {
			inner: alloc::sync::Arc::new(linalg_sp::lu::factorize_symbolic_lu(mat, Default::default())?),
		})
	}
}

impl<I: Index, T: ComplexField> Llt<I, T> {
	/// Returns the LLT factorization of the input matrix with the same sparsity pattern as the
	/// original one used to construct the symbolic factorization.
	///
	/// Only the provided side is accessed.
	#[track_caller]
	pub fn try_new_with_symbolic(symbolic: SymbolicLlt<I>, mat: SparseColMatRef<'_, I, T>, side: Side) -> Result<Self, LltError> {
		let len_val = symbolic.inner.len_val();
		let mut numeric = alloc::vec::Vec::new();
		numeric.try_reserve_exact(len_val).map_err(|_| FaerError::OutOfMemory)?;
		numeric.resize(len_val, zero::<T>());
		let par = get_global_parallelism();
		symbolic.inner.factorize_numeric_llt::<T>(
			&mut numeric,
			mat,
			side,
			Default::default(),
			par,
			MemStack::new(&mut MemBuffer::try_new(
				symbolic.inner.factorize_numeric_llt_scratch::<T>(par, Default::default()),
			)?),
			Default::default(),
		)?;
		Ok(Self { symbolic, numeric })
	}
}

impl<I: Index, T: ComplexField> Lu<I, T> {
	/// Returns the LU factorization of the input matrix with the same sparsity pattern as the
	/// original one used to construct the symbolic factorization.
	#[track_caller]
	pub fn try_new_with_symbolic(symbolic: SymbolicLu<I>, mat: SparseColMatRef<'_, I, T>) -> Result<Self, LuError> {
		let mut numeric = linalg_sp::lu::NumericLu::new();
		let par = get_global_parallelism();
		symbolic.inner.factorize_numeric_lu::<T>(
			&mut numeric,
			mat,
			par,
			MemStack::new(&mut MemBuffer::try_new(
				symbolic.inner.factorize_numeric_lu_scratch::<T>(par, Default::default()),
			)?),
			Default::default(),
		)?;
		Ok(Self { symbolic, numeric })
	}
}

impl<I: Index, T: ComplexField> Qr<I, T> {
	/// Returns the QR factorization of the input matrix with the same sparsity pattern as the
	/// original one used to construct the symbolic factorization.
	#[track_caller]
	pub fn try_new_with_symbolic(symbolic: SymbolicQr<I>, mat: SparseColMatRef<'_, I, T>) -> Result<Self, FaerError> {
		let len_val = symbolic.inner.len_val();
		let len_idx = symbolic.inner.len_idx();

		let mut indices = alloc::vec::Vec::new();
		let mut numeric = alloc::vec::Vec::new();
		numeric.try_reserve_exact(len_val).map_err(|_| FaerError::OutOfMemory)?;
		numeric.resize(len_val, zero::<T>());

		indices.try_reserve_exact(len_idx).map_err(|_| FaerError::OutOfMemory)?;
		indices.resize(len_idx, I::truncate(0));
		let par = get_global_parallelism();

		symbolic.inner.factorize_numeric_qr::<T>(
			&mut indices,
			&mut numeric,
			mat,
			par,
			MemStack::new(&mut MemBuffer::try_new(
				symbolic.inner.factorize_numeric_qr_scratch::<T>(par, Default::default()),
			)?),
			Default::default(),
		);
		Ok(Self { symbolic, indices, numeric })
	}
}

impl<I: Index, T: ComplexField> ShapeCore for Llt<I, T> {
	#[track_caller]
	fn nrows(&self) -> usize {
		self.symbolic.inner.nrows()
	}

	#[track_caller]
	fn ncols(&self) -> usize {
		self.symbolic.inner.ncols()
	}
}

impl<I: Index, T: ComplexField> ShapeCore for Qr<I, T> {
	#[track_caller]
	fn nrows(&self) -> usize {
		self.symbolic.inner.nrows()
	}

	#[track_caller]
	fn ncols(&self) -> usize {
		self.symbolic.inner.ncols()
	}
}

impl<I: Index, T: ComplexField> ShapeCore for Lu<I, T> {
	#[track_caller]
	fn nrows(&self) -> usize {
		self.symbolic.inner.nrows()
	}

	#[track_caller]
	fn ncols(&self) -> usize {
		self.symbolic.inner.ncols()
	}
}

impl<I: Index, T: ComplexField> SolveCore<T> for Llt<I, T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();
		let rhs_ncols = rhs.ncols();
		linalg_sp::cholesky::LltRef::<'_, I, T>::new(&self.symbolic.inner, &self.numeric).solve_in_place_with_conj(
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(self.symbolic.inner.solve_in_place_scratch::<T>(rhs_ncols, par))),
		);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();
		let rhs_ncols = rhs.ncols();
		linalg_sp::cholesky::LltRef::<'_, I, T>::new(&self.symbolic.inner, &self.numeric).solve_in_place_with_conj(
			conj.compose(Conj::Yes),
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(self.symbolic.inner.solve_in_place_scratch::<T>(rhs_ncols, par))),
		);
	}
}

impl<I: Index, T: ComplexField> SolveCore<T> for Qr<I, T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();
		let rhs_ncols = rhs.ncols();
		unsafe { linalg_sp::qr::QrRef::<'_, I, T>::new_unchecked(&self.symbolic.inner, &self.indices, &self.numeric) }.solve_in_place_with_conj(
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(self.symbolic.inner.solve_in_place_scratch::<T>(rhs_ncols, par))),
		);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		_ = conj;
		_ = rhs;
		panic!("the sparse QR decomposition doesn't support solve_transpose.\nconsider using the sparse LU or Cholesky instead");
	}
}

impl<I: Index, T: ComplexField> SolveLstsqCore<T> for Qr<I, T> {
	#[track_caller]
	fn solve_lstsq_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();
		let rhs_ncols = rhs.ncols();
		unsafe { linalg_sp::qr::QrRef::<'_, I, T>::new_unchecked(&self.symbolic.inner, &self.indices, &self.numeric) }.solve_in_place_with_conj(
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(self.symbolic.inner.solve_in_place_scratch::<T>(rhs_ncols, par))),
		);
	}
}

impl<I: Index, T: ComplexField> SolveCore<T> for Lu<I, T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();
		let rhs_ncols = rhs.ncols();
		unsafe { linalg_sp::lu::LuRef::<'_, I, T>::new_unchecked(&self.symbolic.inner, &self.numeric) }.solve_in_place_with_conj(
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(self.symbolic.inner.solve_in_place_scratch::<T>(rhs_ncols, par))),
		);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();
		let rhs_ncols = rhs.ncols();
		unsafe { linalg_sp::lu::LuRef::<'_, I, T>::new_unchecked(&self.symbolic.inner, &self.numeric) }.solve_transpose_in_place_with_conj(
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(
				self.symbolic.inner.solve_transpose_in_place_scratch::<T>(rhs_ncols, par),
			)),
		);
	}
}

impl<I: Index, T: ComplexField> SparseColMatRef<'_, I, T> {
	/// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each column.
	#[track_caller]
	pub fn sp_solve_lower_triangular_in_place(&self, mut rhs: impl AsMatMut<T = T, Rows = usize>) {
		linalg_sp::triangular_solve::solve_lower_triangular_in_place(*self, Conj::No, rhs.as_mat_mut().as_dyn_cols_mut(), get_global_parallelism());
	}

	/// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each column.
	#[track_caller]
	pub fn sp_solve_upper_triangular_in_place(&self, mut rhs: impl AsMatMut<T = T, Rows = usize>) {
		linalg_sp::triangular_solve::solve_upper_triangular_in_place(*self, Conj::No, rhs.as_mat_mut().as_dyn_cols_mut(), get_global_parallelism());
	}

	/// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each column.
	#[track_caller]
	pub fn sp_solve_unit_lower_triangular_in_place(&self, mut rhs: impl AsMatMut<T = T, Rows = usize>) {
		linalg_sp::triangular_solve::solve_unit_lower_triangular_in_place(
			*self,
			Conj::No,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each column.
	#[track_caller]
	pub fn sp_solve_unit_upper_triangular_in_place(&self, mut rhs: impl AsMatMut<T = T, Rows = usize>) {
		linalg_sp::triangular_solve::solve_unit_upper_triangular_in_place(
			*self,
			Conj::No,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// Returns the LLT decomposition of `self`. Only the provided side is accessed.
	#[track_caller]
	#[doc(alias = "sp_llt")]
	pub fn sp_cholesky(&self, side: Side) -> Result<Llt<I, T>, LltError> {
		Llt::try_new_with_symbolic(SymbolicLlt::try_new(self.symbolic(), side)?, *self, side)
	}

	/// Returns the LU decomposition of `self` with partial (row) pivoting.
	#[track_caller]
	pub fn sp_lu(&self) -> Result<Lu<I, T>, LuError> {
		Lu::try_new_with_symbolic(SymbolicLu::try_new(self.symbolic())?, *self)
	}

	/// Returns the QR decomposition of `self`.
	#[track_caller]
	pub fn sp_qr(&self) -> Result<Qr<I, T>, FaerError> {
		Qr::try_new_with_symbolic(SymbolicQr::try_new(self.symbolic())?, *self)
	}
}

impl<I: Index, T: ComplexField> SparseRowMatRef<'_, I, T> {
	/// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each row.
	#[track_caller]
	pub fn sp_solve_lower_triangular_in_place(&self, mut rhs: impl AsMatMut<T = T, Rows = usize>) {
		linalg_sp::triangular_solve::solve_upper_triangular_transpose_in_place(
			self.transpose(),
			Conj::No,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each row.
	#[track_caller]
	pub fn sp_solve_upper_triangular_in_place(&self, mut rhs: impl AsMatMut<T = T, Rows = usize>) {
		linalg_sp::triangular_solve::solve_lower_triangular_transpose_in_place(
			self.transpose(),
			Conj::No,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each row.
	#[track_caller]
	pub fn sp_solve_unit_lower_triangular_in_place(&self, mut rhs: impl AsMatMut<T = T, Rows = usize>) {
		linalg_sp::triangular_solve::solve_unit_upper_triangular_transpose_in_place(
			self.transpose(),
			Conj::No,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each row.
	#[track_caller]
	pub fn sp_solve_unit_upper_triangular_in_place(&self, mut rhs: impl AsMatMut<T = T, Rows = usize>) {
		linalg_sp::triangular_solve::solve_unit_lower_triangular_transpose_in_place(
			self.transpose(),
			Conj::No,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// Returns the LLT decomposition of `self`. Only the provided side is accessed.
	#[track_caller]
	#[doc(alias = "sp_llt")]
	pub fn sp_cholesky(&self, side: Side) -> Result<Llt<I, T>, LltError> {
		let this = self.to_col_major()?;
		let this = this.rb();
		Llt::try_new_with_symbolic(SymbolicLlt::try_new(this.symbolic(), side)?, this, side)
	}

	/// Returns the LU decomposition of `self` with partial (row) pivoting.
	#[track_caller]
	pub fn sp_lu(&self) -> Result<Lu<I, T>, LuError> {
		let this = self.to_col_major()?;
		let this = this.rb();
		Lu::try_new_with_symbolic(SymbolicLu::try_new(this.symbolic())?, this)
	}

	/// Returns the QR decomposition of `self`.
	#[track_caller]
	pub fn sp_qr(&self) -> Result<Qr<I, T>, FaerError> {
		let this = self.to_col_major()?;
		let this = this.rb();
		Qr::try_new_with_symbolic(SymbolicQr::try_new(this.symbolic())?, this)
	}
}

impl<I: Index, T: ComplexField> SparseColMatMut<'_, I, T> {
	/// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each column.
	#[track_caller]
	pub fn sp_solve_lower_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_lower_triangular_in_place(rhs);
	}

	/// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each column.
	#[track_caller]
	pub fn sp_solve_upper_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_upper_triangular_in_place(rhs);
	}

	/// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each column.
	#[track_caller]
	pub fn sp_solve_unit_lower_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_unit_lower_triangular_in_place(rhs);
	}

	/// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each column.
	#[track_caller]
	pub fn sp_solve_unit_upper_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_unit_upper_triangular_in_place(rhs);
	}

	/// Returns the LLT decomposition of `self`. Only the provided side is accessed.
	#[track_caller]
	#[doc(alias = "sp_llt")]
	pub fn sp_cholesky(&self, side: Side) -> Result<Llt<I, T>, LltError> {
		self.rb().sp_cholesky(side)
	}

	/// Returns the LU decomposition of `self` with partial (row) pivoting.
	#[track_caller]
	pub fn sp_lu(&self) -> Result<Lu<I, T>, LuError> {
		self.rb().sp_lu()
	}

	/// Returns the QR decomposition of `self`.
	#[track_caller]
	pub fn sp_qr(&self) -> Result<Qr<I, T>, FaerError> {
		self.rb().sp_qr()
	}
}

impl<I: Index, T: ComplexField> SparseRowMatMut<'_, I, T> {
	/// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each row.
	#[track_caller]
	pub fn sp_solve_lower_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_lower_triangular_in_place(rhs);
	}

	/// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each row.
	#[track_caller]
	pub fn sp_solve_upper_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_upper_triangular_in_place(rhs);
	}

	/// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each row.
	#[track_caller]
	pub fn sp_solve_unit_lower_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_unit_lower_triangular_in_place(rhs);
	}

	/// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each row.
	#[track_caller]
	pub fn sp_solve_unit_upper_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_unit_upper_triangular_in_place(rhs);
	}

	/// Returns the LLT decomposition of `self`. Only the provided side is accessed.
	#[track_caller]
	#[doc(alias = "sp_llt")]
	pub fn sp_cholesky(&self, side: Side) -> Result<Llt<I, T>, LltError> {
		self.rb().sp_cholesky(side)
	}

	/// Returns the LU decomposition of `self` with partial (row) pivoting.
	#[track_caller]
	pub fn sp_lu(&self) -> Result<Lu<I, T>, LuError> {
		self.rb().sp_lu()
	}

	/// Returns the QR decomposition of `self`.
	#[track_caller]
	pub fn sp_qr(&self) -> Result<Qr<I, T>, FaerError> {
		self.rb().sp_qr()
	}
}
impl<I: Index, T: ComplexField> SparseColMat<I, T> {
	/// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each column.
	#[track_caller]
	pub fn sp_solve_lower_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_lower_triangular_in_place(rhs);
	}

	/// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each column.
	#[track_caller]
	pub fn sp_solve_upper_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_upper_triangular_in_place(rhs);
	}

	/// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each column.
	#[track_caller]
	pub fn sp_solve_unit_lower_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_unit_lower_triangular_in_place(rhs);
	}

	/// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each column.
	#[track_caller]
	pub fn sp_solve_unit_upper_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_unit_upper_triangular_in_place(rhs);
	}

	/// Returns the LLT decomposition of `self`. Only the provided side is accessed.
	#[track_caller]
	#[doc(alias = "sp_llt")]
	pub fn sp_cholesky(&self, side: Side) -> Result<Llt<I, T>, LltError> {
		self.rb().sp_cholesky(side)
	}

	/// Returns the LU decomposition of `self` with partial (row) pivoting.
	#[track_caller]
	pub fn sp_lu(&self) -> Result<Lu<I, T>, LuError> {
		self.rb().sp_lu()
	}

	/// Returns the QR decomposition of `self`.
	#[track_caller]
	pub fn sp_qr(&self) -> Result<Qr<I, T>, FaerError> {
		self.rb().sp_qr()
	}
}

impl<I: Index, T: ComplexField> SparseRowMat<I, T> {
	/// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each row.
	#[track_caller]
	pub fn sp_solve_lower_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_lower_triangular_in_place(rhs);
	}

	/// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
	/// stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each row.
	#[track_caller]
	pub fn sp_solve_upper_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_upper_triangular_in_place(rhs);
	}

	/// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each row.
	#[track_caller]
	pub fn sp_solve_unit_lower_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_unit_lower_triangular_in_place(rhs);
	}

	/// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
	/// and stores the result in `rhs`.
	///
	/// # Note
	/// The matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each row.
	#[track_caller]
	pub fn sp_solve_unit_upper_triangular_in_place(&self, rhs: impl AsMatMut<T = T, Rows = usize>) {
		self.rb().sp_solve_unit_upper_triangular_in_place(rhs);
	}

	/// Returns the LLT decomposition of `self`. Only the provided side is accessed.
	#[track_caller]
	#[doc(alias = "sp_llt")]
	pub fn sp_cholesky(&self, side: Side) -> Result<Llt<I, T>, LltError> {
		self.rb().sp_cholesky(side)
	}

	/// Returns the LU decomposition of `self` with partial (row) pivoting.
	#[track_caller]
	pub fn sp_lu(&self) -> Result<Lu<I, T>, LuError> {
		self.rb().sp_lu()
	}

	/// Returns the QR decomposition of `self`.
	#[track_caller]
	pub fn sp_qr(&self) -> Result<Qr<I, T>, FaerError> {
		self.rb().sp_qr()
	}
}
