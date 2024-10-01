use super::*;
use crate::{
    col::ColBatchMut,
    linalg::solvers::{SpSolverCore, SpSolverLstsqCore},
};

/// Reference-counted sparse symbolic Cholesky factorization.
#[derive(Debug)]
pub struct SymbolicCholesky<I: Index> {
    inner: alloc::sync::Arc<super::cholesky::SymbolicCholesky<I>>,
}
/// Sparse Cholesky factorization.
#[derive(Clone, Debug)]
pub struct Cholesky<I: Index, E: Entity> {
    symbolic: SymbolicCholesky<I>,
    values: VecGroup<E>,
}

/// Reference-counted sparse symbolic QR factorization.
#[derive(Debug)]
pub struct SymbolicQr<I: Index> {
    inner: alloc::sync::Arc<super::qr::SymbolicQr<I>>,
}
/// Sparse QR factorization.
#[derive(Clone, Debug)]
pub struct Qr<I: Index, E: Entity> {
    symbolic: SymbolicQr<I>,
    indices: alloc::vec::Vec<I>,
    values: VecGroup<E>,
}

/// Reference-counted sparse symbolic LU factorization.
#[derive(Debug)]
pub struct SymbolicLu<I: Index> {
    inner: alloc::sync::Arc<super::lu::SymbolicLu<I>>,
}
/// Sparse LU factorization.
#[derive(Clone, Debug)]
pub struct Lu<I: Index, E: Entity> {
    symbolic: SymbolicLu<I>,
    numeric: super::lu::NumericLu<I, E>,
}

impl<I: Index> Clone for SymbolicCholesky<I> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl<I: Index> Clone for SymbolicQr<I> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl<I: Index> Clone for SymbolicLu<I> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<I: Index> SymbolicCholesky<I> {
    /// Returns the symbolic Cholesky factorization of the input matrix.
    ///
    /// Only the provided side is accessed.
    #[track_caller]
    pub fn try_new(mat: SymbolicSparseColMatRef<'_, I>, side: Side) -> Result<Self, FaerError> {
        Ok(Self {
            inner: alloc::sync::Arc::new(super::cholesky::factorize_symbolic_cholesky(
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
            inner: alloc::sync::Arc::new(super::qr::factorize_symbolic_qr(
                mat,
                Default::default(),
            )?),
        })
    }
}
impl<I: Index> SymbolicLu<I> {
    /// Returns the symbolic LU factorization of the input matrix.
    #[track_caller]
    pub fn try_new(mat: SymbolicSparseColMatRef<'_, I>) -> Result<Self, FaerError> {
        Ok(Self {
            inner: alloc::sync::Arc::new(super::lu::factorize_symbolic_lu(
                mat,
                Default::default(),
            )?),
        })
    }
}

impl<I: Index, E: ComplexField> Cholesky<I, E> {
    /// Returns the Cholesky factorization of the input matrix with the same sparsity pattern as the
    /// original one used to construct the symbolic factorization.
    ///
    /// Only the provided side is accessed.
    #[track_caller]
    pub fn try_new_with_symbolic(
        symbolic: SymbolicCholesky<I>,
        mat: SparseColMatRef<'_, I, E>,
        side: Side,
    ) -> Result<Self, CholeskyError> {
        let len_values = symbolic.inner.len_values();
        let mut values = VecGroup::new();
        values
            .try_reserve_exact(len_values)
            .map_err(|_| FaerError::OutOfMemory)?;
        values.resize(len_values, E::faer_zero().faer_into_units());
        let parallelism = get_global_parallelism();
        symbolic.inner.factorize_numeric_llt::<E>(
            values.as_slice_mut().into_inner(),
            mat,
            side,
            Default::default(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                symbolic
                    .inner
                    .factorize_numeric_llt_req::<E>(parallelism)
                    .map_err(|_| FaerError::OutOfMemory)?,
            )),
        )?;
        Ok(Self { symbolic, values })
    }
}

impl<I: Index, E: ComplexField> Qr<I, E> {
    /// Returns the QR factorization of the input matrix with the same sparsity pattern as the
    /// original one used to construct the symbolic factorization.
    #[track_caller]
    pub fn try_new_with_symbolic(
        symbolic: SymbolicQr<I>,
        mat: SparseColMatRef<'_, I, E>,
    ) -> Result<Self, FaerError> {
        let len_values = symbolic.inner.len_values();
        let len_indices = symbolic.inner.len_indices();
        let mut values = VecGroup::new();
        let mut indices = alloc::vec::Vec::new();
        values
            .try_reserve_exact(len_values)
            .map_err(|_| FaerError::OutOfMemory)?;
        indices
            .try_reserve_exact(len_indices)
            .map_err(|_| FaerError::OutOfMemory)?;
        values.resize(len_values, E::faer_zero().faer_into_units());
        indices.resize(len_indices, I::truncate(0));
        let parallelism = get_global_parallelism();
        symbolic.inner.factorize_numeric_qr::<E>(
            &mut indices,
            values.as_slice_mut().into_inner(),
            mat,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                symbolic
                    .inner
                    .factorize_numeric_qr_req::<E>(parallelism)
                    .map_err(|_| FaerError::OutOfMemory)?,
            )),
        );
        Ok(Self {
            symbolic,
            indices,
            values,
        })
    }
}

impl<I: Index, E: ComplexField> Lu<I, E> {
    /// Returns the LU factorization of the input matrix with the same sparsity pattern as the
    /// original one used to construct the symbolic factorization.
    #[track_caller]
    pub fn try_new_with_symbolic(
        symbolic: SymbolicLu<I>,
        mat: SparseColMatRef<'_, I, E>,
    ) -> Result<Self, super::LuError> {
        let mut numeric = super::lu::NumericLu::new();
        let parallelism = get_global_parallelism();
        symbolic.inner.factorize_numeric_lu::<E>(
            &mut numeric,
            mat,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                symbolic
                    .inner
                    .factorize_numeric_lu_req::<E>(parallelism)
                    .map_err(|_| FaerError::OutOfMemory)?,
            )),
        )?;
        Ok(Self { symbolic, numeric })
    }
}

impl<I: Index, E: ComplexField> SpSolverCore<E> for Cholesky<I, E> {
    #[inline]
    fn nrows(&self) -> usize {
        self.symbolic.inner.nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.symbolic.inner.ncols()
    }

    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();
        super::cholesky::LltRef::<'_, I, E>::new(
            &self.symbolic.inner,
            self.values.as_slice().into_inner(),
        )
        .solve_in_place_with_conj(
            conj,
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                self.symbolic
                    .inner
                    .solve_in_place_req::<E>(rhs_ncols)
                    .unwrap(),
            )),
        );
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();
        super::cholesky::LltRef::<'_, I, E>::new(
            &self.symbolic.inner,
            self.values.as_slice().into_inner(),
        )
        .solve_in_place_with_conj(
            conj.compose(Conj::Yes),
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                self.symbolic
                    .inner
                    .solve_in_place_req::<E>(rhs_ncols)
                    .unwrap(),
            )),
        );
    }
}

impl<I: Index, E: ComplexField> SpSolverCore<E> for Qr<I, E> {
    #[inline]
    fn nrows(&self) -> usize {
        self.symbolic.inner.nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.symbolic.inner.ncols()
    }

    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        self.solve_lstsq_in_place_with_conj_impl(rhs, conj);
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let _ = (&rhs, &conj);
        unimplemented!(
            "the sparse QR decomposition doesn't support solve_transpose.\n\
                               consider using the sparse LU or Cholesky instead."
        )
    }
}

impl<I: Index, E: ComplexField> SpSolverLstsqCore<E> for Qr<I, E> {
    #[track_caller]
    fn solve_lstsq_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();
        unsafe {
            super::qr::QrRef::<'_, I, E>::new_unchecked(
                &self.symbolic.inner,
                &self.indices,
                self.values.as_slice().into_inner(),
            )
        }
        .solve_in_place_with_conj(
            conj,
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                self.symbolic
                    .inner
                    .solve_in_place_req::<E>(rhs_ncols, parallelism)
                    .unwrap(),
            )),
        );
    }
}

impl<I: Index, E: ComplexField> SpSolverCore<E> for Lu<I, E> {
    #[inline]
    fn nrows(&self) -> usize {
        self.symbolic.inner.nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.symbolic.inner.ncols()
    }

    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();
        unsafe { super::lu::LuRef::<'_, I, E>::new_unchecked(&self.symbolic.inner, &self.numeric) }
            .solve_in_place_with_conj(
                conj,
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    self.symbolic
                        .inner
                        .solve_in_place_req::<E>(rhs_ncols, parallelism)
                        .unwrap(),
                )),
            );
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();
        unsafe { super::lu::LuRef::<'_, I, E>::new_unchecked(&self.symbolic.inner, &self.numeric) }
            .solve_transpose_in_place_with_conj(
                conj,
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    self.symbolic
                        .inner
                        .solve_in_place_req::<E>(rhs_ncols, parallelism)
                        .unwrap(),
                )),
            );
    }
}

impl<I: Index, E: ComplexField> SparseColMatRef<'_, I, E> {
    /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the first stored element in each column.
    #[track_caller]
    pub fn sp_solve_lower_triangular_in_place(&self, mut rhs: impl ColBatchMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_lower_triangular_in_place(
            *self,
            Conj::No,
            rhs.as_2d_mut(),
            get_global_parallelism(),
        );
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the last stored element in each column.
    #[track_caller]
    pub fn sp_solve_upper_triangular_in_place(&self, mut rhs: impl ColBatchMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_upper_triangular_in_place(
            *self,
            Conj::No,
            rhs.as_2d_mut(),
            get_global_parallelism(),
        );
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the first stored element in each column.
    #[track_caller]
    pub fn sp_solve_unit_lower_triangular_in_place(&self, mut rhs: impl ColBatchMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
            *self,
            Conj::No,
            rhs.as_2d_mut(),
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
    pub fn sp_solve_unit_upper_triangular_in_place(&self, mut rhs: impl ColBatchMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
            *self,
            Conj::No,
            rhs.as_2d_mut(),
            get_global_parallelism(),
        );
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    #[doc(alias = "sp_llt")]
    pub fn sp_cholesky(&self, side: Side) -> Result<Cholesky<I, E>, CholeskyError> {
        Cholesky::try_new_with_symbolic(
            SymbolicCholesky::try_new(self.symbolic(), side)?,
            *self,
            side,
        )
    }

    /// Returns the LU decomposition of `self` with partial (row) pivoting.
    #[track_caller]
    pub fn sp_lu(&self) -> Result<Lu<I, E>, LuError> {
        Lu::try_new_with_symbolic(SymbolicLu::try_new(self.symbolic())?, *self)
    }

    /// Returns the QR decomposition of `self`.
    #[track_caller]
    pub fn sp_qr(&self) -> Result<Qr<I, E>, FaerError> {
        Qr::try_new_with_symbolic(SymbolicQr::try_new(self.symbolic())?, *self)
    }
}

impl<I: Index, E: ComplexField> SparseRowMatRef<'_, I, E> {
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the last stored element in each row.
    #[track_caller]
    pub fn sp_solve_lower_triangular_in_place(&self, mut rhs: impl ColBatchMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_upper_triangular_transpose_in_place(
            self.transpose(),
            Conj::No,
            rhs.as_2d_mut(),
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
    pub fn sp_solve_upper_triangular_in_place(&self, mut rhs: impl ColBatchMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_lower_triangular_transpose_in_place(
            self.transpose(),
            Conj::No,
            rhs.as_2d_mut(),
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
    pub fn sp_solve_unit_lower_triangular_in_place(&self, mut rhs: impl ColBatchMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_unit_upper_triangular_transpose_in_place(
            self.transpose(),
            Conj::No,
            rhs.as_2d_mut(),
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
    pub fn sp_solve_unit_upper_triangular_in_place(&self, mut rhs: impl ColBatchMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_unit_lower_triangular_transpose_in_place(
            self.transpose(),
            Conj::No,
            rhs.as_2d_mut(),
            get_global_parallelism(),
        );
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    #[doc(alias = "sp_llt")]
    pub fn sp_cholesky(&self, side: Side) -> Result<Cholesky<I, E>, CholeskyError> {
        let this = self.to_col_major()?;
        let this = this.as_ref();
        Cholesky::try_new_with_symbolic(
            SymbolicCholesky::try_new(this.symbolic(), side)?,
            this,
            side,
        )
    }

    /// Returns the LU decomposition of `self` with partial (row) pivoting.
    #[track_caller]
    pub fn sp_lu(&self) -> Result<Lu<I, E>, LuError> {
        let this = self.to_col_major()?;
        let this = this.as_ref();
        Lu::try_new_with_symbolic(SymbolicLu::try_new(this.symbolic())?, this)
    }

    /// Returns the QR decomposition of `self`.
    #[track_caller]
    pub fn sp_qr(&self) -> Result<Qr<I, E>, FaerError> {
        let this = self.to_col_major()?;
        let this = this.as_ref();
        Qr::try_new_with_symbolic(SymbolicQr::try_new(this.symbolic())?, this)
    }
}

impl<I: Index, E: ComplexField> SparseColMatMut<'_, I, E> {
    /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the first stored element in each column.
    #[track_caller]
    pub fn sp_solve_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_lower_triangular_in_place(rhs);
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the last stored element in each column.
    #[track_caller]
    pub fn sp_solve_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_upper_triangular_in_place(rhs);
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the first stored element in each column.
    #[track_caller]
    pub fn sp_solve_unit_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_unit_lower_triangular_in_place(rhs);
    }
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the last stored element in each column.
    #[track_caller]
    pub fn sp_solve_unit_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_unit_upper_triangular_in_place(rhs);
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    #[doc(alias = "sp_llt")]
    pub fn sp_cholesky(&self, side: Side) -> Result<Cholesky<I, E>, CholeskyError> {
        self.as_ref().sp_cholesky(side)
    }

    /// Returns the LU decomposition of `self` with partial (row) pivoting.
    #[track_caller]
    pub fn sp_lu(&self) -> Result<Lu<I, E>, LuError> {
        self.as_ref().sp_lu()
    }

    /// Returns the QR decomposition of `self`.
    #[track_caller]
    pub fn sp_qr(&self) -> Result<Qr<I, E>, FaerError> {
        self.as_ref().sp_qr()
    }
}

impl<I: Index, E: ComplexField> SparseRowMatMut<'_, I, E> {
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the last stored element in each row.
    #[track_caller]
    pub fn sp_solve_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_lower_triangular_in_place(rhs);
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the first stored element in each row.
    #[track_caller]
    pub fn sp_solve_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_upper_triangular_in_place(rhs);
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the last stored element in each row.
    #[track_caller]
    pub fn sp_solve_unit_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_unit_lower_triangular_in_place(rhs);
    }
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the first stored element in each row.
    #[track_caller]
    pub fn sp_solve_unit_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_unit_upper_triangular_in_place(rhs);
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    #[doc(alias = "sp_llt")]
    pub fn sp_cholesky(&self, side: Side) -> Result<Cholesky<I, E>, CholeskyError> {
        self.as_ref().sp_cholesky(side)
    }

    /// Returns the LU decomposition of `self` with partial (row) pivoting.
    #[track_caller]
    pub fn sp_lu(&self) -> Result<Lu<I, E>, LuError> {
        self.as_ref().sp_lu()
    }

    /// Returns the QR decomposition of `self`.
    #[track_caller]
    pub fn sp_qr(&self) -> Result<Qr<I, E>, FaerError> {
        self.as_ref().sp_qr()
    }
}
impl<I: Index, E: ComplexField> SparseColMat<I, E> {
    /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the first stored element in each column.
    #[track_caller]
    pub fn sp_solve_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_lower_triangular_in_place(rhs);
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the last stored element in each column.
    #[track_caller]
    pub fn sp_solve_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_upper_triangular_in_place(rhs);
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the first stored element in each column.
    #[track_caller]
    pub fn sp_solve_unit_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_unit_lower_triangular_in_place(rhs);
    }
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the last stored element in each column.
    #[track_caller]
    pub fn sp_solve_unit_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_unit_upper_triangular_in_place(rhs);
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    #[doc(alias = "sp_llt")]
    pub fn sp_cholesky(&self, side: Side) -> Result<Cholesky<I, E>, CholeskyError> {
        self.as_ref().sp_cholesky(side)
    }

    /// Returns the LU decomposition of `self` with partial (row) pivoting.
    #[track_caller]
    pub fn sp_lu(&self) -> Result<Lu<I, E>, LuError> {
        self.as_ref().sp_lu()
    }

    /// Returns the QR decomposition of `self`.
    #[track_caller]
    pub fn sp_qr(&self) -> Result<Qr<I, E>, FaerError> {
        self.as_ref().sp_qr()
    }
}

impl<I: Index, E: ComplexField> SparseRowMat<I, E> {
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the last stored element in each row.
    #[track_caller]
    pub fn sp_solve_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_lower_triangular_in_place(rhs);
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the first stored element in each row.
    #[track_caller]
    pub fn sp_solve_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_upper_triangular_in_place(rhs);
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the last stored element in each row.
    #[track_caller]
    pub fn sp_solve_unit_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_unit_lower_triangular_in_place(rhs);
    }
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// # Note
    /// The matrix indices need not be sorted, but
    /// the diagonal element is assumed to be the first stored element in each row.
    #[track_caller]
    pub fn sp_solve_unit_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E>) {
        self.as_ref().sp_solve_unit_upper_triangular_in_place(rhs);
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    #[doc(alias = "sp_llt")]
    pub fn sp_cholesky(&self, side: Side) -> Result<Cholesky<I, E>, CholeskyError> {
        self.as_ref().sp_cholesky(side)
    }

    /// Returns the LU decomposition of `self` with partial (row) pivoting.
    #[track_caller]
    pub fn sp_lu(&self) -> Result<Lu<I, E>, LuError> {
        self.as_ref().sp_lu()
    }

    /// Returns the QR decomposition of `self`.
    #[track_caller]
    pub fn sp_qr(&self) -> Result<Qr<I, E>, FaerError> {
        self.as_ref().sp_qr()
    }
}
