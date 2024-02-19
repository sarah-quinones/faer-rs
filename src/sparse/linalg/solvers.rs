use super::*;
use crate::mat::{AsMatMut, AsMatRef};

/// Object-safe base for [`SpSolver`]
pub trait SpSolverCore<E: Entity> {
    /// Returns the number of rows of the matrix used to construct this decomposition.
    fn nrows(&self) -> usize;
    /// Returns the number of columns of the matrix used to construct this decomposition.
    fn ncols(&self) -> usize;

    #[doc(hidden)]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj);
    #[doc(hidden)]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj);
}

pub trait SpSolverLstsqCore<E: Entity>: SpSolverCore<E> {
    #[doc(hidden)]
    fn solve_lstsq_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj);
}

pub trait SpSolver<E: ComplexField>: SpSolverCore<E> {
    /// Solves the equation `self * X = rhs`, and stores the result in `rhs`.
    fn solve_in_place(&self, rhs: impl AsMatMut<E>);
    /// Solves the equation `conjugate(self) * X = rhs`, and stores the result in `rhs`.
    fn solve_conj_in_place(&self, rhs: impl AsMatMut<E>);
    /// Solves the equation `transpose(self) * X = rhs`, and stores the result in `rhs`.
    fn solve_transpose_in_place(&self, rhs: impl AsMatMut<E>);
    /// Solves the equation `adjoint(self) * X = rhs`, and stores the result in `rhs`.
    fn solve_conj_transpose_in_place(&self, rhs: impl AsMatMut<E>);
    /// Solves the equation `self * X = rhs`, and returns the result.
    fn solve<ViewE: Conjugate<Canonical = E>>(&self, rhs: impl AsMatRef<ViewE>) -> Mat<E>;
    /// Solves the equation `conjugate(self) * X = rhs`, and returns the result.
    fn solve_conj<ViewE: Conjugate<Canonical = E>>(&self, rhs: impl AsMatRef<ViewE>) -> Mat<E>;
    /// Solves the equation `transpose(self) * X = rhs`, and returns the result.
    fn solve_transpose<ViewE: Conjugate<Canonical = E>>(&self, rhs: impl AsMatRef<ViewE>)
        -> Mat<E>;
    /// Solves the equation `adjoint(self) * X = rhs`, and returns the result.
    fn solve_conj_transpose<ViewE: Conjugate<Canonical = E>>(
        &self,
        rhs: impl AsMatRef<ViewE>,
    ) -> Mat<E>;
}

pub trait SpSolverLstsq<E: Entity>: SpSolverLstsqCore<E> {
    /// Solves the equation `self * X = rhs`, in the sense of least squares, and stores the
    /// result in the top rows of `rhs`.
    fn solve_lstsq_in_place(&self, rhs: impl AsMatMut<E>);
    /// Solves the equation `conjugate(self) * X = rhs`, in the sense of least squares, and
    /// stores the result in the top rows of `rhs`.
    fn solve_lstsq_conj_in_place(&self, rhs: impl AsMatMut<E>);
    /// Solves the equation `self * X = rhs`, and returns the result.
    fn solve_lstsq<ViewE: Conjugate<Canonical = E>>(&self, rhs: impl AsMatRef<ViewE>) -> Mat<E>;
    /// Solves the equation `conjugate(self) * X = rhs`, and returns the result.
    fn solve_lstsq_conj<ViewE: Conjugate<Canonical = E>>(
        &self,
        rhs: impl AsMatRef<ViewE>,
    ) -> Mat<E>;
}

#[track_caller]
fn solve_with_conj_impl<
    E: ComplexField,
    D: ?Sized + SpSolverCore<E>,
    ViewE: Conjugate<Canonical = E>,
>(
    d: &D,
    rhs: MatRef<'_, ViewE>,
    conj: Conj,
) -> Mat<E> {
    let mut rhs = rhs.to_owned();
    d.solve_in_place_with_conj_impl(rhs.as_mut(), conj);
    rhs
}

#[track_caller]
fn solve_transpose_with_conj_impl<
    E: ComplexField,
    D: ?Sized + SpSolverCore<E>,
    ViewE: Conjugate<Canonical = E>,
>(
    d: &D,
    rhs: MatRef<'_, ViewE>,
    conj: Conj,
) -> Mat<E> {
    let mut rhs = rhs.to_owned();
    d.solve_transpose_in_place_with_conj_impl(rhs.as_mut(), conj);
    rhs
}

#[track_caller]
fn solve_lstsq_with_conj_impl<
    E: ComplexField,
    D: ?Sized + SpSolverLstsqCore<E>,
    ViewE: Conjugate<Canonical = E>,
>(
    d: &D,
    rhs: MatRef<'_, ViewE>,
    conj: Conj,
) -> Mat<E> {
    let mut rhs = rhs.to_owned();
    let k = rhs.ncols();
    d.solve_lstsq_in_place_with_conj_impl(rhs.as_mut(), conj);
    rhs.resize_with(d.ncols(), k, |_, _| unreachable!());
    rhs
}

impl<E: ComplexField, Dec: ?Sized + SpSolverCore<E>> SpSolver<E> for Dec {
    #[track_caller]
    fn solve_in_place(&self, rhs: impl AsMatMut<E>) {
        let mut rhs = rhs;
        self.solve_in_place_with_conj_impl(rhs.as_mat_mut(), Conj::No)
    }

    #[track_caller]
    fn solve_conj_in_place(&self, rhs: impl AsMatMut<E>) {
        let mut rhs = rhs;
        self.solve_in_place_with_conj_impl(rhs.as_mat_mut(), Conj::Yes)
    }

    #[track_caller]
    fn solve_transpose_in_place(&self, rhs: impl AsMatMut<E>) {
        let mut rhs = rhs;
        self.solve_transpose_in_place_with_conj_impl(rhs.as_mat_mut(), Conj::No)
    }

    #[track_caller]
    fn solve_conj_transpose_in_place(&self, rhs: impl AsMatMut<E>) {
        let mut rhs = rhs;
        self.solve_transpose_in_place_with_conj_impl(rhs.as_mat_mut(), Conj::Yes)
    }

    #[track_caller]
    fn solve<ViewE: Conjugate<Canonical = E>>(&self, rhs: impl AsMatRef<ViewE>) -> Mat<E> {
        solve_with_conj_impl::<E, _, _>(self, rhs.as_mat_ref(), Conj::No)
    }

    #[track_caller]
    fn solve_conj<ViewE: Conjugate<Canonical = E>>(&self, rhs: impl AsMatRef<ViewE>) -> Mat<E> {
        solve_with_conj_impl::<E, _, _>(self, rhs.as_mat_ref(), Conj::Yes)
    }

    #[track_caller]
    fn solve_transpose<ViewE: Conjugate<Canonical = E>>(
        &self,
        rhs: impl AsMatRef<ViewE>,
    ) -> Mat<E> {
        solve_transpose_with_conj_impl::<E, _, _>(self, rhs.as_mat_ref(), Conj::No)
    }

    #[track_caller]
    fn solve_conj_transpose<ViewE: Conjugate<Canonical = E>>(
        &self,
        rhs: impl AsMatRef<ViewE>,
    ) -> Mat<E> {
        solve_transpose_with_conj_impl::<E, _, _>(self, rhs.as_mat_ref(), Conj::Yes)
    }
}

impl<E: ComplexField, Dec: ?Sized + SpSolverLstsqCore<E>> SpSolverLstsq<E> for Dec {
    #[track_caller]
    fn solve_lstsq_in_place(&self, rhs: impl AsMatMut<E>) {
        let mut rhs = rhs;
        self.solve_lstsq_in_place_with_conj_impl(rhs.as_mat_mut(), Conj::No)
    }

    #[track_caller]
    fn solve_lstsq_conj_in_place(&self, rhs: impl AsMatMut<E>) {
        let mut rhs = rhs;
        self.solve_lstsq_in_place_with_conj_impl(rhs.as_mat_mut(), Conj::Yes)
    }

    #[track_caller]
    fn solve_lstsq<ViewE: Conjugate<Canonical = E>>(&self, rhs: impl AsMatRef<ViewE>) -> Mat<E> {
        solve_lstsq_with_conj_impl::<E, _, _>(self, rhs.as_mat_ref(), Conj::No)
    }

    #[track_caller]
    fn solve_lstsq_conj<ViewE: Conjugate<Canonical = E>>(
        &self,
        rhs: impl AsMatRef<ViewE>,
    ) -> Mat<E> {
        solve_lstsq_with_conj_impl::<E, _, _>(self, rhs.as_mat_ref(), Conj::Yes)
    }
}

#[derive(Debug)]
pub struct SymbolicCholesky<I: Index> {
    inner: alloc::sync::Arc<super::cholesky::SymbolicCholesky<I>>,
}
#[derive(Clone, Debug)]
pub struct Cholesky<I: Index, E: Entity> {
    symbolic: SymbolicCholesky<I>,
    values: VecGroup<E>,
}

#[derive(Debug)]
pub struct SymbolicQr<I: Index> {
    inner: alloc::sync::Arc<super::qr::SymbolicQr<I>>,
}
#[derive(Clone, Debug)]
pub struct Qr<I: Index, E: Entity> {
    symbolic: SymbolicQr<I>,
    indices: alloc::vec::Vec<I>,
    values: VecGroup<E>,
}

#[derive(Debug)]
pub struct SymbolicLu<I: Index> {
    inner: alloc::sync::Arc<super::lu::SymbolicLu<I>>,
}
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
    #[track_caller]
    pub fn try_new(mat: SymbolicSparseColMatRef<'_, I>, side: Side) -> Result<Self, FaerError> {
        Ok(Self {
            inner: alloc::sync::Arc::new(super::cholesky::factorize_symbolic_cholesky(
                mat,
                side,
                Default::default(),
            )?),
        })
    }
}
impl<I: Index> SymbolicQr<I> {
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
    #[track_caller]
    pub fn sp_solve_lower_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_lower_triangular_in_place(
            *self,
            Conj::No,
            rhs.as_mat_mut(),
            get_global_parallelism(),
        );
    }
    #[track_caller]
    pub fn sp_solve_upper_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_upper_triangular_in_place(
            *self,
            Conj::No,
            rhs.as_mat_mut(),
            get_global_parallelism(),
        );
    }
    #[track_caller]
    pub fn sp_solve_unit_lower_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
            *self,
            Conj::No,
            rhs.as_mat_mut(),
            get_global_parallelism(),
        );
    }
    #[track_caller]
    pub fn sp_solve_unit_upper_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
            *self,
            Conj::No,
            rhs.as_mat_mut(),
            get_global_parallelism(),
        );
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
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
    #[track_caller]
    pub fn sp_solve_lower_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_upper_triangular_in_place(
            self.transpose(),
            Conj::No,
            rhs.as_mat_mut(),
            get_global_parallelism(),
        );
    }
    #[track_caller]
    pub fn sp_solve_upper_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_lower_triangular_in_place(
            self.transpose(),
            Conj::No,
            rhs.as_mat_mut(),
            get_global_parallelism(),
        );
    }
    #[track_caller]
    pub fn sp_solve_unit_lower_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
            self.transpose(),
            Conj::No,
            rhs.as_mat_mut(),
            get_global_parallelism(),
        );
    }
    #[track_caller]
    pub fn sp_solve_unit_upper_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
        crate::sparse::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
            self.transpose(),
            Conj::No,
            rhs.as_mat_mut(),
            get_global_parallelism(),
        );
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
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

impl<I: Index, E: ComplexField> SparseColMat<I, E> {
    #[track_caller]
    pub fn sp_solve_lower_triangular_in_place(&self, rhs: impl AsMatMut<E>) {
        self.as_ref().sp_solve_lower_triangular_in_place(rhs);
    }
    #[track_caller]
    pub fn sp_solve_upper_triangular_in_place(&self, rhs: impl AsMatMut<E>) {
        self.as_ref().sp_solve_upper_triangular_in_place(rhs);
    }
    #[track_caller]
    pub fn sp_solve_unit_lower_triangular_in_place(&self, rhs: impl AsMatMut<E>) {
        self.as_ref().sp_solve_unit_lower_triangular_in_place(rhs);
    }
    #[track_caller]
    pub fn sp_solve_unit_upper_triangular_in_place(&self, rhs: impl AsMatMut<E>) {
        self.as_ref().sp_solve_unit_upper_triangular_in_place(rhs);
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
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
    #[track_caller]
    pub fn sp_solve_lower_triangular_in_place(&self, rhs: impl AsMatMut<E>) {
        self.as_ref().sp_solve_lower_triangular_in_place(rhs);
    }
    #[track_caller]
    pub fn sp_solve_upper_triangular_in_place(&self, rhs: impl AsMatMut<E>) {
        self.as_ref().sp_solve_upper_triangular_in_place(rhs);
    }
    #[track_caller]
    pub fn sp_solve_unit_lower_triangular_in_place(&self, rhs: impl AsMatMut<E>) {
        self.as_ref().sp_solve_unit_lower_triangular_in_place(rhs);
    }
    #[track_caller]
    pub fn sp_solve_unit_upper_triangular_in_place(&self, rhs: impl AsMatMut<E>) {
        self.as_ref().sp_solve_unit_upper_triangular_in_place(rhs);
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
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
