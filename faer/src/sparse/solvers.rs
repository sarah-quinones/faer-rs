use crate::get_global_parallelism;
use crate::internal_prelude_sp::*;
use crate::linalg::solvers::{ShapeCore, SolveCore, SolveLstsqCore};
use linalg_sp::{LltError, LuError};
/// reference-counted sparse symbolic $LL^\top$ factorization
#[derive(Debug, Clone)]
pub struct SymbolicLlt<I: Index> {
	inner: alloc::sync::Arc<linalg_sp::cholesky::SymbolicCholesky<I>>,
}
/// sparse $LL^\top$ factorization
#[derive(Debug, Clone)]
pub struct Llt<I: Index, T> {
	symbolic: SymbolicLlt<I>,
	numeric: alloc::vec::Vec<T>,
}
/// reference-counted sparse symbolic $QR$ factorization
#[derive(Debug, Clone)]
pub struct SymbolicQr<I: Index> {
	inner: alloc::sync::Arc<linalg_sp::qr::SymbolicQr<I>>,
}
/// sparse $QR$ factorization
#[derive(Debug, Clone)]
pub struct Qr<I: Index, T> {
	symbolic: SymbolicQr<I>,
	indices: alloc::vec::Vec<I>,
	numeric: alloc::vec::Vec<T>,
}
/// reference-counted sparse symbolic $LU$ factorization
#[derive(Debug, Clone)]
pub struct SymbolicLu<I: Index> {
	inner: alloc::sync::Arc<linalg_sp::lu::SymbolicLu<I>>,
}
/// sparse $QR$ factorization
#[derive(Debug, Clone)]
pub struct Lu<I: Index, T> {
	symbolic: SymbolicLu<I>,
	numeric: linalg_sp::lu::NumericLu<I, T>,
}
impl<I: Index> SymbolicLlt<I> {
	/// returns the symbolic $LL^\top$ factorization of the input matrix
	///
	/// only the provided side is accessed
	#[track_caller]
	pub fn try_new(
		mat: SymbolicSparseColMatRef<'_, I>,
		side: Side,
	) -> Result<Self, FaerError> {
		Ok(Self {
			inner: alloc::sync::Arc::new(
				linalg_sp::cholesky::factorize_symbolic_cholesky(
					mat,
					side,
					Default::default(),
					Default::default(),
				)?,
			),
		})
	}
}
impl<I: Index> SymbolicQr<I> {
	/// returns the symbolic $QR$ factorization of the input matrix
	#[track_caller]
	pub fn try_new(
		mat: SymbolicSparseColMatRef<'_, I>,
	) -> Result<Self, FaerError> {
		Ok(Self {
			inner: alloc::sync::Arc::new(linalg_sp::qr::factorize_symbolic_qr(
				mat,
				Default::default(),
			)?),
		})
	}
}
impl<I: Index> SymbolicLu<I> {
	/// returns the symbolic $LU$ factorization of the input matrix
	#[track_caller]
	pub fn try_new(
		mat: SymbolicSparseColMatRef<'_, I>,
	) -> Result<Self, FaerError> {
		Ok(Self {
			inner: alloc::sync::Arc::new(linalg_sp::lu::factorize_symbolic_lu(
				mat,
				Default::default(),
			)?),
		})
	}
}
impl<I: Index, T: ComplexField> Llt<I, T> {
	/// returns the $LL^\top$ factorization of the input matrix with the same
	/// sparsity pattern as the original one used to construct the symbolic
	/// factorization
	///
	/// only the provided side is accessed
	#[track_caller]
	pub fn try_new_with_symbolic(
		symbolic: SymbolicLlt<I>,
		mat: SparseColMatRef<'_, I, T>,
		side: Side,
	) -> Result<Self, LltError> {
		let len_val = symbolic.inner.len_val();
		let mut numeric = alloc::vec::Vec::new();
		numeric
			.try_reserve_exact(len_val)
			.map_err(|_| FaerError::OutOfMemory)?;
		numeric.resize(len_val, zero::<T>());
		let par = get_global_parallelism();
		symbolic.inner.factorize_numeric_llt::<T>(
			&mut numeric,
			mat,
			side,
			Default::default(),
			par,
			MemStack::new(&mut MemBuffer::try_new(
				symbolic.inner.factorize_numeric_llt_scratch::<T>(
					par,
					Default::default(),
				),
			)?),
			Default::default(),
		)?;
		Ok(Self { symbolic, numeric })
	}
}
impl<I: Index, T: ComplexField> Lu<I, T> {
	/// returns the $LU$ factorization of the input matrix with the same
	/// sparsity pattern as the original one used to construct the symbolic
	/// factorization
	#[track_caller]
	pub fn try_new_with_symbolic(
		symbolic: SymbolicLu<I>,
		mat: SparseColMatRef<'_, I, T>,
	) -> Result<Self, LuError> {
		let mut numeric = linalg_sp::lu::NumericLu::new();
		let par = get_global_parallelism();
		symbolic.inner.factorize_numeric_lu::<T>(
			&mut numeric,
			mat,
			par,
			MemStack::new(&mut MemBuffer::try_new(
				symbolic
					.inner
					.factorize_numeric_lu_scratch::<T>(par, Default::default()),
			)?),
			Default::default(),
		)?;
		Ok(Self { symbolic, numeric })
	}
}
impl<I: Index, T: ComplexField> Qr<I, T> {
	/// returns the $QR$ factorization of the input matrix with the same
	/// sparsity pattern as the original one used to construct the symbolic
	/// factorization
	#[track_caller]
	pub fn try_new_with_symbolic(
		symbolic: SymbolicQr<I>,
		mat: SparseColMatRef<'_, I, T>,
	) -> Result<Self, FaerError> {
		let len_val = symbolic.inner.len_val();
		let len_idx = symbolic.inner.len_idx();
		let mut indices = alloc::vec::Vec::new();
		let mut numeric = alloc::vec::Vec::new();
		numeric
			.try_reserve_exact(len_val)
			.map_err(|_| FaerError::OutOfMemory)?;
		numeric.resize(len_val, zero::<T>());
		indices
			.try_reserve_exact(len_idx)
			.map_err(|_| FaerError::OutOfMemory)?;
		indices.resize(len_idx, I::truncate(0));
		let par = get_global_parallelism();
		symbolic.inner.factorize_numeric_qr::<T>(
			&mut indices,
			&mut numeric,
			mat,
			par,
			MemStack::new(&mut MemBuffer::try_new(
				symbolic
					.inner
					.factorize_numeric_qr_scratch::<T>(par, Default::default()),
			)?),
			Default::default(),
		);
		Ok(Self {
			symbolic,
			indices,
			numeric,
		})
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
		linalg_sp::cholesky::LltRef::<'_, I, T>::new(
			&self.symbolic.inner,
			&self.numeric,
		)
		.solve_in_place_with_conj(
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(
				self.symbolic
					.inner
					.solve_in_place_scratch::<T>(rhs_ncols, par),
			)),
		);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(
		&self,
		conj: Conj,
		rhs: MatMut<'_, T>,
	) {
		let par = get_global_parallelism();
		let rhs_ncols = rhs.ncols();
		linalg_sp::cholesky::LltRef::<'_, I, T>::new(
			&self.symbolic.inner,
			&self.numeric,
		)
		.solve_in_place_with_conj(
			conj.compose(Conj::Yes),
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(
				self.symbolic
					.inner
					.solve_in_place_scratch::<T>(rhs_ncols, par),
			)),
		);
	}
}
impl<I: Index, T: ComplexField> SolveCore<T> for Qr<I, T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();
		let rhs_ncols = rhs.ncols();
		unsafe {
			linalg_sp::qr::QrRef::<'_, I, T>::new_unchecked(
				&self.symbolic.inner,
				&self.indices,
				&self.numeric,
			)
		}
		.solve_in_place_with_conj(
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(
				self.symbolic
					.inner
					.solve_in_place_scratch::<T>(rhs_ncols, par),
			)),
		);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(
		&self,
		conj: Conj,
		rhs: MatMut<'_, T>,
	) {
		_ = conj;
		_ = rhs;
		panic!(
			"the sparse QR decomposition doesn't support \
			 solve_transpose.\nconsider using the sparse LU or Cholesky \
			 instead"
		);
	}
}
impl<I: Index, T: ComplexField> SolveLstsqCore<T> for Qr<I, T> {
	#[track_caller]
	fn solve_lstsq_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();
		let rhs_ncols = rhs.ncols();
		unsafe {
			linalg_sp::qr::QrRef::<'_, I, T>::new_unchecked(
				&self.symbolic.inner,
				&self.indices,
				&self.numeric,
			)
		}
		.solve_in_place_with_conj(
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(
				self.symbolic
					.inner
					.solve_in_place_scratch::<T>(rhs_ncols, par),
			)),
		);
	}
}
impl<I: Index, T: ComplexField> SolveCore<T> for Lu<I, T> {
	#[track_caller]
	fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>) {
		let par = get_global_parallelism();
		let rhs_ncols = rhs.ncols();

		linalg_sp::lu::LuRef::<'_, I, T>::new_unchecked(
			&self.symbolic.inner,
			&self.numeric,
		)
		.solve_in_place_with_conj(
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(
				self.symbolic
					.inner
					.solve_in_place_scratch::<T>(rhs_ncols, par),
			)),
		);
	}

	#[track_caller]
	fn solve_transpose_in_place_with_conj(
		&self,
		conj: Conj,
		rhs: MatMut<'_, T>,
	) {
		let par = get_global_parallelism();
		let rhs_ncols = rhs.ncols();

		linalg_sp::lu::LuRef::<'_, I, T>::new_unchecked(
			&self.symbolic.inner,
			&self.numeric,
		)
		.solve_transpose_in_place_with_conj(
			conj,
			rhs,
			par,
			MemStack::new(&mut MemBuffer::new(
				self.symbolic
					.inner
					.solve_transpose_in_place_scratch::<T>(rhs_ncols, par),
			)),
		);
	}
}
impl<
	I: Index,
	C: Conjugate,
	Inner: for<'short> Reborrow<'short, Target = csc_numeric::Ref<'short, I, C>>,
> csc_numeric::generic::SparseColMat<Inner>
{
	/// assuming `self` is a lower triangular matrix, solves the equation `self
	/// * x = rhs`, and stores the result in `rhs`
	///
	/// # note
	/// the matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each
	/// column
	#[track_caller]
	pub fn sp_solve_lower_triangular_in_place(
		&self,
		mut rhs: impl AsMatMut<T = C::Canonical, Rows = usize>,
	) {
		let this = self.rb().canonical();
		let conj = if C::IS_CANONICAL { Conj::No } else { Conj::Yes };
		linalg_sp::triangular_solve::solve_lower_triangular_in_place(
			this,
			conj,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// assuming `self` is an upper triangular matrix, solves the equation `self
	/// * x = rhs`, and stores the result in `rhs`
	///
	/// # note
	/// the matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each
	/// column
	#[track_caller]
	pub fn sp_solve_upper_triangular_in_place(
		&self,
		mut rhs: impl AsMatMut<T = C::Canonical, Rows = usize>,
	) {
		let this = self.rb().canonical();
		let conj = if C::IS_CANONICAL { Conj::No } else { Conj::Yes };
		linalg_sp::triangular_solve::solve_upper_triangular_in_place(
			this,
			conj,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// assuming `self` is a unit lower triangular matrix, solves the equation
	/// `self * x = rhs`, and stores the result in `rhs`
	///
	/// # note
	/// the matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each
	/// column
	#[track_caller]
	pub fn sp_solve_unit_lower_triangular_in_place(
		&self,
		mut rhs: impl AsMatMut<T = C::Canonical, Rows = usize>,
	) {
		let this = self.rb().canonical();
		let conj = if C::IS_CANONICAL { Conj::No } else { Conj::Yes };
		linalg_sp::triangular_solve::solve_unit_lower_triangular_in_place(
			this,
			conj,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// assuming `self` is a unit upper triangular matrix, solves the equation
	/// `self * x = rhs`, and stores the result in `rhs`
	///
	/// # note
	/// the matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each
	/// column
	#[track_caller]
	pub fn sp_solve_unit_upper_triangular_in_place(
		&self,
		mut rhs: impl AsMatMut<T = C::Canonical, Rows = usize>,
	) {
		let this = self.rb().canonical();
		let conj = if C::IS_CANONICAL { Conj::No } else { Conj::Yes };
		linalg_sp::triangular_solve::solve_unit_upper_triangular_in_place(
			this,
			conj,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}
}

impl<
	I: Index,
	T: ComplexField,
	Inner: for<'short> Reborrow<'short, Target = csc_numeric::Ref<'short, I, T>>,
> csc_numeric::generic::SparseColMat<Inner>
{
	/// returns the $LL^\top$ decomposition of `self`. only the provided side is
	/// accessed
	#[track_caller]
	#[doc(alias = "sp_llt")]
	pub fn sp_cholesky(&self, side: Side) -> Result<Llt<I, T>, LltError> {
		let this = self.rb();
		Llt::try_new_with_symbolic(
			SymbolicLlt::try_new(this.symbolic(), side)?,
			this,
			side,
		)
	}

	/// returns the $LU$ decomposition of `self` with partial (row) pivoting
	#[track_caller]
	pub fn sp_lu(&self) -> Result<Lu<I, T>, LuError> {
		let this = self.rb();
		Lu::try_new_with_symbolic(SymbolicLu::try_new(this.symbolic())?, this)
	}

	/// returns the $QR$ decomposition of `self`
	#[track_caller]
	pub fn sp_qr(&self) -> Result<Qr<I, T>, FaerError> {
		let this = self.rb();
		Qr::try_new_with_symbolic(SymbolicQr::try_new(this.symbolic())?, this)
	}
}
impl<
	I: Index,
	T: ComplexField,
	Inner: for<'short> Reborrow<'short, Target = csr_numeric::Ref<'short, I, T>>,
> csr_numeric::generic::SparseRowMat<Inner>
{
	/// assuming `self` is an upper triangular matrix, solves the equation `self
	/// * x = rhs`, and stores the result in `rhs`
	///
	/// # note
	/// the matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each
	/// row
	#[track_caller]
	pub fn sp_solve_lower_triangular_in_place(
		&self,
		mut rhs: impl AsMatMut<T = T, Rows = usize>,
	) {
		linalg_sp::triangular_solve::solve_upper_triangular_transpose_in_place(
			self.rb().transpose(),
			Conj::No,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// assuming `self` is an upper triangular matrix, solves the equation `self
	/// * x = rhs`, and stores the result in `rhs`
	///
	/// # note
	/// the matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each
	/// row
	#[track_caller]
	pub fn sp_solve_upper_triangular_in_place(
		&self,
		mut rhs: impl AsMatMut<T = T, Rows = usize>,
	) {
		linalg_sp::triangular_solve::solve_lower_triangular_transpose_in_place(
			self.rb().transpose(),
			Conj::No,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// assuming `self` is a unit lower triangular matrix, solves the equation
	/// `self * x = rhs`, and stores the result in `rhs`
	///
	/// # note
	/// the matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the last stored element in each
	/// row
	#[track_caller]
	pub fn sp_solve_unit_lower_triangular_in_place(
		&self,
		mut rhs: impl AsMatMut<T = T, Rows = usize>,
	) {
		linalg_sp::triangular_solve::solve_unit_upper_triangular_transpose_in_place(
			self.rb().transpose(),
			Conj::No,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// assuming `self` is a unit upper triangular matrix, solves the equation
	/// `self * x = rhs`, and stores the result in `rhs`
	///
	/// # note
	/// the matrix indices need not be sorted, but
	/// the diagonal element is assumed to be the first stored element in each
	/// row
	#[track_caller]
	pub fn sp_solve_unit_upper_triangular_in_place(
		&self,
		mut rhs: impl AsMatMut<T = T, Rows = usize>,
	) {
		linalg_sp::triangular_solve::solve_unit_lower_triangular_transpose_in_place(
			self.rb().transpose(),
			Conj::No,
			rhs.as_mat_mut().as_dyn_cols_mut(),
			get_global_parallelism(),
		);
	}

	/// returns the $LL^\top$ decomposition of `self`. only the provided side is
	/// accessed
	#[track_caller]
	#[doc(alias = "sp_llt")]
	pub fn sp_cholesky(&self, side: Side) -> Result<Llt<I, T>, LltError> {
		let this = self.rb().to_col_major()?;
		let this = this.rb();
		Llt::try_new_with_symbolic(
			SymbolicLlt::try_new(this.symbolic(), side)?,
			this,
			side,
		)
	}

	/// returns the $LU$ decomposition of `self` with partial (row) pivoting
	#[track_caller]
	pub fn sp_lu(&self) -> Result<Lu<I, T>, LuError> {
		let this = self.rb().to_col_major()?;
		let this = this.rb();
		Lu::try_new_with_symbolic(SymbolicLu::try_new(this.symbolic())?, this)
	}

	/// returns the $QR$ decomposition of `self`
	#[track_caller]
	pub fn sp_qr(&self) -> Result<Qr<I, T>, FaerError> {
		let this = self.rb().to_col_major()?;
		let this = this.rb();
		Qr::try_new_with_symbolic(SymbolicQr::try_new(this.symbolic())?, this)
	}
}

#[cfg(test)]
mod tests {

	use super::*;
	use crate::assert;
	use std::io::BufRead;
	use std::path::PathBuf;

	#[test]
	fn codeberg_292() {
		let mat = SparseColMat::<usize, f64>::try_new_from_triplets(
			3,
			3,
			&[
				Triplet {
					row: 2,
					col: 0,
					val: 1.0,
				},
				Triplet {
					row: 2,
					col: 1,
					val: 1.0,
				},
				Triplet {
					row: 0,
					col: 2,
					val: 1.0,
				},
				Triplet {
					row: 1,
					col: 2,
					val: 1.0,
				},
				Triplet {
					row: 2,
					col: 2,
					val: 1.0,
				},
			],
		)
		.unwrap();
		let _ = mat.sp_lu();
	}

	#[test]
	fn codeberg_301() {
		// test code by @activexray
		for path in [
			"test_data/sparse_lu/matrix_n15960.txt",
			"test_data/sparse_lu/matrix_n19168.txt",
		] {
			let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path);
			let f = std::fs::File::open(&path).expect("open matrix dump");
			let mut lines =
				std::io::BufReader::new(f).lines().map(|l| l.unwrap());

			let header = lines.next().unwrap();
			let mut it = header.split_whitespace();
			let n: usize = it.next().unwrap().parse().unwrap();
			let nnz: usize = it.next().unwrap().parse().unwrap();

			let mut triplets = Vec::with_capacity(nnz);
			for _ in 0..nnz {
				let line = lines.next().unwrap();
				let mut it = line.split_whitespace();
				let row: usize = it.next().unwrap().parse().unwrap();
				let col: usize = it.next().unwrap().parse().unwrap();
				let val: f64 = it.next().unwrap().parse().unwrap();
				triplets.push(Triplet::new(row, col, val));
			}

			let marker = lines.next().unwrap();
			assert_eq!(marker, "RHS");

			let mut b = Vec::with_capacity(n);
			for _ in 0..n {
				let line = lines.next().unwrap();
				b.push(line.trim().parse::<f64>().unwrap());
			}

			let mat = SparseColMat::<usize, f64>::try_new_from_triplets(
				n, n, &triplets,
			)
			.unwrap();
			let rhs = faer::col::Col::from_fn(n, |i| b[i]);

			let lu = mat.sp_lu().expect("sp_lu factorization");
			let x = lu.solve(&rhs);

			// Residual check: ||A x - b|| / ||b||
			let ax = &mat * &x;
			let mut res_norm = 0.0f64;
			let mut b_norm = 0.0f64;
			for i in 0..n {
				let r = ax[i] - b[i];
				res_norm += r * r;
				b_norm += b[i] * b[i];
			}
			res_norm = res_norm.sqrt();
			b_norm = b_norm.sqrt();

			assert!(res_norm / b_norm < 1e-14);
		}
	}
}
