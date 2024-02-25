use crate::{
    assert, col::*, diag::DiagRef, linalg::matmul::triangular::BlockStructure, mat::*,
    perm::PermRef, unzipped, zipped, Side, *,
};
use dyn_stack::*;
use reborrow::*;

pub use crate::{
    linalg::cholesky::llt::CholeskyError,
    sparse::linalg::solvers::{SpSolver, SpSolverCore, SpSolverLstsq, SpSolverLstsqCore},
};

/// Object-safe base for [`Solver`]
pub trait SolverCore<E: Entity>: SpSolverCore<E> {
    /// Reconstructs the original matrix using the decomposition.
    fn reconstruct(&self) -> Mat<E>;
    /// Computes the inverse of the original matrix using the decomposition.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    fn inverse(&self) -> Mat<E>;
}
/// Object-safe base for [`SolverLstsq`]
pub trait SolverLstsqCore<E: Entity>: SolverCore<E> + SpSolverLstsqCore<E> {}

/// Solver that can compute solution of a linear system.
pub trait Solver<E: ComplexField>: SolverCore<E> + SpSolver<E> {}
/// Dense solver that can compute the least squares solution of an overdetermined linear system.
pub trait SolverLstsq<E: ComplexField>: SolverLstsqCore<E> + SpSolverLstsq<E> {}

const _: () = {
    fn __assert_object_safe<E: ComplexField>() {
        let _: Option<&dyn SolverCore<E>> = None;
        let _: Option<&dyn SolverLstsqCore<E>> = None;
    }
};

impl<E: ComplexField, Dec: ?Sized + SolverLstsqCore<E>> SolverLstsq<E> for Dec {}

impl<E: ComplexField, Dec: ?Sized + SolverCore<E>> Solver<E> for Dec {}

/// Cholesky decomposition.
pub struct Cholesky<E: Entity> {
    factors: Mat<E>,
}

/// Bunch-Kaufman decomposition.
pub struct Lblt<E: Entity> {
    factors: Mat<E>,
    subdiag: Mat<E>,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
}

/// LU decomposition with partial pivoting.
pub struct PartialPivLu<E: Entity> {
    pub(crate) factors: Mat<E>,
    row_perm: Vec<usize>,
    row_perm_inv: Vec<usize>,
    n_transpositions: usize,
}
/// LU decomposition with full pivoting.
pub struct FullPivLu<E: Entity> {
    factors: Mat<E>,
    row_perm: Vec<usize>,
    row_perm_inv: Vec<usize>,
    col_perm: Vec<usize>,
    col_perm_inv: Vec<usize>,
    n_transpositions: usize,
}

/// QR decomposition.
pub struct Qr<E: Entity> {
    factors: Mat<E>,
    householder: Mat<E>,
}
/// QR decomposition with column pivoting.
pub struct ColPivQr<E: Entity> {
    factors: Mat<E>,
    householder: Mat<E>,
    col_perm: Vec<usize>,
    col_perm_inv: Vec<usize>,
}

/// Singular value decomposition.
pub struct Svd<E: Entity> {
    s: Mat<E>,
    u: Mat<E>,
    v: Mat<E>,
}
/// Thin singular value decomposition.
pub struct ThinSvd<E: Entity> {
    inner: Svd<E>,
}

/// Self-adjoint eigendecomposition.
pub struct SelfAdjointEigendecomposition<E: Entity> {
    s: Mat<E>,
    u: Mat<E>,
}

/// Complex eigendecomposition.
pub struct Eigendecomposition<E: Entity> {
    s: Col<E>,
    u: Mat<E>,
}

impl<E: ComplexField> Cholesky<E> {
    /// Returns the Cholesky factorization of the input
    /// matrix, or an error if the matrix is not positive definite.
    ///
    /// The factorization is such that $A = LL^H$, where $L$ is lower triangular.
    ///
    /// The matrix is interpreted as Hermitian, but only the provided side is accessed.
    #[track_caller]
    pub fn try_new<ViewE: Conjugate<Canonical = E>>(
        matrix: MatRef<'_, ViewE>,
        side: Side,
    ) -> Result<Self, CholeskyError> {
        assert!(matrix.nrows() == matrix.ncols());

        let dim = matrix.nrows();
        let parallelism = get_global_parallelism();

        let mut factors = Mat::<E>::zeros(dim, dim);
        match side {
            Side::Lower => {
                zipped!(factors.as_mut(), matrix).for_each_triangular_lower(
                    crate::linalg::zip::Diag::Include,
                    |unzipped!(mut dst, src)| dst.write(src.read().canonicalize()),
                );
            }
            Side::Upper => {
                zipped!(factors.as_mut(), matrix.adjoint()).for_each_triangular_lower(
                    crate::linalg::zip::Diag::Include,
                    |unzipped!(mut dst, src)| dst.write(src.read().canonicalize()),
                );
            }
        }

        let params = Default::default();

        crate::linalg::cholesky::llt::compute::cholesky_in_place(
            factors.as_mut(),
            Default::default(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::cholesky::llt::compute::cholesky_in_place_req::<E>(
                    dim,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        )?;
        Ok(Self { factors })
    }

    fn dim(&self) -> usize {
        self.factors.nrows()
    }

    /// Returns the factor $L$ of the Cholesky decomposition.
    pub fn compute_l(&self) -> Mat<E> {
        let mut factor = self.factors.to_owned();
        zipped!(factor.as_mut())
            .for_each_triangular_upper(crate::linalg::zip::Diag::Skip, |unzipped!(mut dst)| {
                dst.write(E::faer_zero())
            });
        factor
    }
}
impl<E: ComplexField> SpSolverCore<E> for Cholesky<E> {
    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();

        crate::linalg::cholesky::llt::solve::solve_in_place_with_conj(
            self.factors.as_ref(),
            conj,
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::cholesky::llt::solve::solve_in_place_req::<E>(
                    self.dim(),
                    rhs_ncols,
                    parallelism,
                )
                .unwrap(),
            )),
        );
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        self.solve_in_place_with_conj_impl(rhs, conj.compose(Conj::Yes))
    }

    fn nrows(&self) -> usize {
        self.factors.nrows()
    }

    fn ncols(&self) -> usize {
        self.factors.ncols()
    }
}
impl<E: ComplexField> SolverCore<E> for Cholesky<E> {
    fn inverse(&self) -> Mat<E> {
        let mut inv = Mat::<E>::zeros(self.dim(), self.dim());
        let parallelism = get_global_parallelism();

        crate::linalg::cholesky::llt::inverse::invert_lower(
            inv.as_mut(),
            self.factors.as_ref(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::cholesky::llt::inverse::invert_lower_req::<E>(
                    self.dim(),
                    parallelism,
                )
                .unwrap(),
            )),
        );

        for j in 0..self.dim() {
            for i in 0..j {
                inv.write(i, j, inv.read(j, i).faer_conj());
            }
        }

        inv
    }

    fn reconstruct(&self) -> Mat<E> {
        let mut rec = Mat::<E>::zeros(self.dim(), self.dim());
        let parallelism = get_global_parallelism();

        crate::linalg::cholesky::llt::reconstruct::reconstruct_lower(
            rec.as_mut(),
            self.factors.as_ref(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::cholesky::llt::reconstruct::reconstruct_lower_req::<E>(self.dim())
                    .unwrap(),
            )),
        );

        for j in 0..self.dim() {
            for i in 0..j {
                rec.write(i, j, rec.read(j, i).faer_conj());
            }
        }

        rec
    }
}

impl<E: ComplexField> Lblt<E> {
    /// Returns the Bunch-Kaufman factorization of the input matrix.
    ///
    /// The matrix is interpreted as Hermitian, but only the provided side is accessed.
    #[track_caller]
    pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>, side: Side) -> Self {
        assert!(matrix.nrows() == matrix.ncols());

        let dim = matrix.nrows();
        let parallelism = get_global_parallelism();

        let mut factors = Mat::<E>::zeros(dim, dim);
        let mut subdiag = Mat::<E>::zeros(dim, 1);
        let mut perm = vec![0; dim];
        let mut perm_inv = vec![0; dim];

        match side {
            Side::Lower => {
                zipped!(factors.as_mut(), matrix).for_each_triangular_lower(
                    crate::linalg::zip::Diag::Include,
                    |unzipped!(mut dst, src)| dst.write(src.read().canonicalize()),
                );
            }
            Side::Upper => {
                zipped!(factors.as_mut(), matrix.adjoint()).for_each_triangular_lower(
                    crate::linalg::zip::Diag::Include,
                    |unzipped!(mut dst, src)| dst.write(src.read().canonicalize()),
                );
            }
        }

        let params = Default::default();

        crate::linalg::cholesky::bunch_kaufman::compute::cholesky_in_place(
            factors.as_mut(),
            subdiag.as_mut(),
            Default::default(),
            &mut perm,
            &mut perm_inv,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::cholesky::bunch_kaufman::compute::cholesky_in_place_req::<usize, E>(
                    dim,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );
        Self {
            factors,
            subdiag,
            perm,
            perm_inv,
        }
    }

    fn dim(&self) -> usize {
        self.factors.nrows()
    }
}

impl<E: ComplexField> SpSolverCore<E> for Lblt<E> {
    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();

        crate::linalg::cholesky::bunch_kaufman::solve::solve_in_place_with_conj(
            self.factors.as_ref(),
            self.subdiag.as_ref(),
            conj,
            unsafe { PermRef::new_unchecked(&self.perm, &self.perm_inv) },
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::cholesky::bunch_kaufman::solve::solve_in_place_req::<usize, E>(
                    self.dim(),
                    rhs_ncols,
                    parallelism,
                )
                .unwrap(),
            )),
        );
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        self.solve_in_place_with_conj_impl(rhs, conj.compose(Conj::Yes))
    }

    fn nrows(&self) -> usize {
        self.factors.nrows()
    }

    fn ncols(&self) -> usize {
        self.factors.ncols()
    }
}
impl<E: ComplexField> SolverCore<E> for Lblt<E> {
    fn inverse(&self) -> Mat<E> {
        let n = self.dim();
        let mut inv = Mat::identity(n, n);
        self.solve_in_place_with_conj_impl(inv.as_mut(), Conj::No);
        inv
    }

    fn reconstruct(&self) -> Mat<E> {
        let parallelism = get_global_parallelism();
        let n = self.dim();
        let lbl = self.factors.as_ref();
        let subdiag = self.subdiag.as_ref();
        let mut mat = Mat::<E>::identity(n, n);
        let mut mat2 = Mat::<E>::identity(n, n);
        zipped!(mat.as_mut(), lbl).for_each_triangular_lower(
            crate::linalg::zip::Diag::Skip,
            |unzipped!(mut dst, src)| dst.write(src.read()),
        );

        let mut j = 0;
        while j < n {
            if subdiag.read(j, 0) == E::faer_zero() {
                let d = lbl.read(j, j).faer_real().faer_inv();
                for i in 0..n {
                    mat.write(i, j, mat.read(i, j).faer_scale_real(d));
                }
                j += 1;
            } else {
                let akp1k = subdiag.read(j, 0).faer_inv();
                let ak = akp1k.faer_scale_real(lbl.read(j, j).faer_real());
                let akp1 = akp1k
                    .faer_conj()
                    .faer_scale_real(lbl.read(j + 1, j + 1).faer_real());
                let denom = ak
                    .faer_mul(akp1)
                    .faer_sub(E::faer_one())
                    .faer_real()
                    .faer_inv();

                for i in 0..n {
                    let xk = mat.read(i, j).faer_mul(akp1k);
                    let xkp1 = mat.read(i, j + 1).faer_mul(akp1k.faer_conj());

                    mat.write(
                        i,
                        j,
                        (akp1.faer_mul(xk).faer_sub(xkp1)).faer_scale_real(denom),
                    );
                    mat.write(
                        i,
                        j + 1,
                        (ak.faer_mul(xkp1).faer_sub(xk)).faer_scale_real(denom),
                    );
                }
                j += 2;
            }
        }
        crate::linalg::matmul::triangular::matmul(
            mat2.as_mut(),
            BlockStructure::TriangularLower,
            lbl,
            BlockStructure::UnitTriangularLower,
            mat.as_ref().adjoint(),
            BlockStructure::Rectangular,
            None,
            E::faer_one(),
            parallelism,
        );

        for j in 0..n {
            let pj = self.perm_inv[j];
            for i in j..n {
                let pi = self.perm_inv[i];

                mat.write(
                    i,
                    j,
                    if pi >= pj {
                        mat2.read(pi, pj)
                    } else {
                        mat2.read(pj, pi).faer_conj()
                    },
                );
            }
        }

        for j in 0..n {
            mat.write(j, j, E::faer_from_real(mat.read(j, j).faer_real()));
            for i in 0..j {
                mat.write(i, j, mat.read(j, i).faer_conj());
            }
        }

        mat
    }
}

impl<E: ComplexField> PartialPivLu<E> {
    /// Returns the LU decomposition of the input matrix with partial (row) pivoting.
    ///
    /// The factorization is such that $PA = LU$, where $L$ is lower triangular, $U$ is unit
    /// upper triangular, and $P$ is the permutation arising from the pivoting.
    #[track_caller]
    pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
        assert!(matrix.nrows() == matrix.ncols());

        let dim = matrix.nrows();
        let parallelism = get_global_parallelism();

        let mut factors = matrix.to_owned();

        let params = Default::default();

        let mut row_perm = vec![0usize; dim];
        let mut row_perm_inv = vec![0usize; dim];

        let (n_transpositions, _) = crate::linalg::lu::partial_pivoting::compute::lu_in_place(
            factors.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::lu::partial_pivoting::compute::lu_in_place_req::<usize, E>(
                    dim,
                    dim,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        Self {
            n_transpositions: n_transpositions.transposition_count,
            factors,
            row_perm,
            row_perm_inv,
        }
    }

    fn dim(&self) -> usize {
        self.factors.nrows()
    }

    /// Returns the row permutation due to pivoting.
    pub fn row_permutation(&self) -> PermRef<'_, usize> {
        unsafe { PermRef::new_unchecked(&self.row_perm, &self.row_perm_inv) }
    }

    /// Returns the number of transpositions that consitute the permutation.
    pub fn transposition_count(&self) -> usize {
        self.n_transpositions
    }

    /// Returns the factor $L$ of the LU decomposition.
    pub fn compute_l(&self) -> Mat<E> {
        let mut factor = self.factors.to_owned();
        zipped!(factor.as_mut())
            .for_each_triangular_upper(crate::linalg::zip::Diag::Skip, |unzipped!(mut dst)| {
                dst.write(E::faer_zero())
            });
        factor
    }
    /// Returns the factor $U$ of the LU decomposition.
    pub fn compute_u(&self) -> Mat<E> {
        let mut factor = self.factors.to_owned();
        zipped!(factor.as_mut())
            .for_each_triangular_lower(crate::linalg::zip::Diag::Skip, |unzipped!(mut dst)| {
                dst.write(E::faer_zero())
            });
        factor
            .as_mut()
            .diagonal_mut()
            .column_vector_mut()
            .fill(E::faer_one());
        factor
    }
}
impl<E: ComplexField> SpSolverCore<E> for PartialPivLu<E> {
    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();

        crate::linalg::lu::partial_pivoting::solve::solve_in_place(
            self.factors.as_ref(),
            conj,
            self.row_permutation(),
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::lu::partial_pivoting::solve::solve_in_place_req::<usize, E>(
                    self.dim(),
                    self.dim(),
                    rhs_ncols,
                    parallelism,
                )
                .unwrap(),
            )),
        );
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();

        crate::linalg::lu::partial_pivoting::solve::solve_transpose_in_place(
            self.factors.as_ref(),
            conj,
            self.row_permutation(),
            rhs,
            parallelism,
            PodStack::new(
                &mut GlobalPodBuffer::new(
                    crate::linalg::lu::partial_pivoting::solve::solve_transpose_in_place_req::<
                        usize,
                        E,
                    >(self.dim(), self.dim(), rhs_ncols, parallelism)
                    .unwrap(),
                ),
            ),
        );
    }

    fn nrows(&self) -> usize {
        self.factors.nrows()
    }

    fn ncols(&self) -> usize {
        self.factors.ncols()
    }
}
impl<E: ComplexField> SolverCore<E> for PartialPivLu<E> {
    fn inverse(&self) -> Mat<E> {
        let mut inv = Mat::<E>::zeros(self.dim(), self.dim());
        let parallelism = get_global_parallelism();

        crate::linalg::lu::partial_pivoting::inverse::invert(
            inv.as_mut(),
            self.factors.as_ref(),
            self.row_permutation(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::lu::partial_pivoting::inverse::invert_req::<usize, E>(
                    self.dim(),
                    self.dim(),
                    parallelism,
                )
                .unwrap(),
            )),
        );

        inv
    }

    fn reconstruct(&self) -> Mat<E> {
        let mut rec = Mat::<E>::zeros(self.dim(), self.dim());
        let parallelism = get_global_parallelism();

        crate::linalg::lu::partial_pivoting::reconstruct::reconstruct(
            rec.as_mut(),
            self.factors.as_ref(),
            self.row_permutation(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::lu::partial_pivoting::reconstruct::reconstruct_req::<usize, E>(
                    self.dim(),
                    self.dim(),
                    parallelism,
                )
                .unwrap(),
            )),
        );

        rec
    }
}

impl<E: ComplexField> FullPivLu<E> {
    /// Returns the LU decomposition of the input matrix with row and column pivoting.
    ///
    /// The factorization is such that $PAQ^\top = LU$, where $L$ is lower triangular, $U$ is unit
    /// upper triangular, and $P$ is the permutation arising from row pivoting and $Q$ is the
    /// permutation due to column pivoting.
    #[track_caller]
    pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
        let m = matrix.nrows();
        let n = matrix.ncols();
        let parallelism = get_global_parallelism();

        let mut factors = matrix.to_owned();

        let params = Default::default();

        let mut row_perm = vec![0usize; m];
        let mut row_perm_inv = vec![0usize; m];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];

        let (n_transpositions, _, _) = crate::linalg::lu::full_pivoting::compute::lu_in_place(
            factors.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::lu::full_pivoting::compute::lu_in_place_req::<usize, E>(
                    m,
                    n,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        Self {
            factors,
            row_perm,
            row_perm_inv,
            col_perm,
            col_perm_inv,
            n_transpositions: n_transpositions.transposition_count,
        }
    }

    /// Returns the row permutation due to pivoting.
    pub fn row_permutation(&self) -> PermRef<'_, usize> {
        unsafe { PermRef::new_unchecked(&self.row_perm, &self.row_perm_inv) }
    }
    /// Returns the column permutation due to pivoting.
    pub fn col_permutation(&self) -> PermRef<'_, usize> {
        unsafe { PermRef::new_unchecked(&self.col_perm, &self.col_perm_inv) }
    }

    /// Returns the number of transpositions that consitute the two permutations.
    pub fn transposition_count(&self) -> usize {
        self.n_transpositions
    }

    /// Returns the factor $L$ of the LU decomposition.
    pub fn compute_l(&self) -> Mat<E> {
        let size = Ord::min(self.nrows(), self.ncols());
        let mut factor = self
            .factors
            .as_ref()
            .submatrix(0, 0, self.nrows(), size)
            .to_owned();
        zipped!(factor.as_mut())
            .for_each_triangular_upper(crate::linalg::zip::Diag::Skip, |unzipped!(mut dst)| {
                dst.write(E::faer_zero())
            });
        factor
    }
    /// Returns the factor $U$ of the LU decomposition.
    pub fn compute_u(&self) -> Mat<E> {
        let size = Ord::min(self.nrows(), self.ncols());
        let mut factor = self
            .factors
            .as_ref()
            .submatrix(0, 0, size, self.ncols())
            .to_owned();
        zipped!(factor.as_mut())
            .for_each_triangular_lower(crate::linalg::zip::Diag::Skip, |unzipped!(mut dst)| {
                dst.write(E::faer_zero())
            });
        factor
            .as_mut()
            .diagonal_mut()
            .column_vector_mut()
            .fill(E::faer_one());
        factor
    }
}
impl<E: ComplexField> SpSolverCore<E> for FullPivLu<E> {
    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        assert!(self.nrows() == self.ncols());

        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();

        crate::linalg::lu::full_pivoting::solve::solve_in_place(
            self.factors.as_ref(),
            conj,
            self.row_permutation(),
            self.col_permutation(),
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::lu::full_pivoting::solve::solve_in_place_req::<usize, E>(
                    self.nrows(),
                    self.ncols(),
                    rhs_ncols,
                    parallelism,
                )
                .unwrap(),
            )),
        );
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        assert!(self.nrows() == self.ncols());

        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();

        crate::linalg::lu::full_pivoting::solve::solve_transpose_in_place(
            self.factors.as_ref(),
            conj,
            self.row_permutation(),
            self.col_permutation(),
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::lu::full_pivoting::solve::solve_transpose_in_place_req::<usize, E>(
                    self.nrows(),
                    self.ncols(),
                    rhs_ncols,
                    parallelism,
                )
                .unwrap(),
            )),
        );
    }

    fn nrows(&self) -> usize {
        self.factors.nrows()
    }

    fn ncols(&self) -> usize {
        self.factors.ncols()
    }
}
impl<E: ComplexField> SolverCore<E> for FullPivLu<E> {
    #[track_caller]
    fn inverse(&self) -> Mat<E> {
        assert!(self.nrows() == self.ncols());

        let dim = self.nrows();

        let mut inv = Mat::<E>::zeros(dim, dim);
        let parallelism = get_global_parallelism();

        crate::linalg::lu::full_pivoting::inverse::invert(
            inv.as_mut(),
            self.factors.as_ref(),
            self.row_permutation(),
            self.col_permutation(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::lu::full_pivoting::inverse::invert_req::<usize, E>(
                    dim,
                    dim,
                    parallelism,
                )
                .unwrap(),
            )),
        );

        inv
    }

    fn reconstruct(&self) -> Mat<E> {
        let mut rec = Mat::<E>::zeros(self.nrows(), self.ncols());
        let parallelism = get_global_parallelism();

        crate::linalg::lu::full_pivoting::reconstruct::reconstruct(
            rec.as_mut(),
            self.factors.as_ref(),
            self.row_permutation(),
            self.col_permutation(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::lu::full_pivoting::reconstruct::reconstruct_req::<usize, E>(
                    self.nrows(),
                    self.ncols(),
                    parallelism,
                )
                .unwrap(),
            )),
        );

        rec
    }
}

impl<E: ComplexField> Qr<E> {
    /// Returns the QR decomposition of the input matrix without pivoting.
    ///
    /// The factorization is such that $A = QR$, where $R$ is upper trapezoidal and $Q$ is unitary.
    #[track_caller]
    pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
        let parallelism = get_global_parallelism();
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();

        let mut factors = matrix.to_owned();
        let size = Ord::min(nrows, ncols);
        let blocksize =
            crate::linalg::qr::no_pivoting::compute::recommended_blocksize::<E>(nrows, ncols);
        let mut householder = Mat::<E>::zeros(blocksize, size);

        let params = Default::default();

        crate::linalg::qr::no_pivoting::compute::qr_in_place(
            factors.as_mut(),
            householder.as_mut(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::qr::no_pivoting::compute::qr_in_place_req::<E>(
                    nrows,
                    ncols,
                    blocksize,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        Self {
            factors,
            householder,
        }
    }

    fn blocksize(&self) -> usize {
        self.householder.nrows()
    }

    /// Returns the factor $R$ of the QR decomposition.
    pub fn compute_r(&self) -> Mat<E> {
        let mut factor = self.factors.to_owned();
        zipped!(factor.as_mut())
            .for_each_triangular_lower(crate::linalg::zip::Diag::Skip, |unzipped!(mut dst)| {
                dst.write(E::faer_zero())
            });
        factor
    }

    /// Returns the factor $R$ of the QR decomposition.
    pub fn compute_q(&self) -> Mat<E> {
        Self::__compute_q_impl(self.factors.as_ref(), self.householder.as_ref(), false)
    }

    /// Returns the top $r$ rows of the factor $R$ of the QR decomposition, where $r =
    /// \min(\text{nrows}(A), \text{ncols}(A))$.
    pub fn compute_thin_r(&self) -> Mat<E> {
        let m = self.nrows();
        let n = self.ncols();
        let mut factor = self.factors.as_ref().subrows(0, Ord::min(m, n)).to_owned();
        zipped!(factor.as_mut())
            .for_each_triangular_lower(crate::linalg::zip::Diag::Skip, |unzipped!(mut dst)| {
                dst.write(E::faer_zero())
            });
        factor
    }

    /// Returns the leftmost $r$ columns of the factor $R$ of the QR decomposition, where $r =
    /// \min(\text{nrows}(A), \text{ncols}(A))$.
    pub fn compute_thin_q(&self) -> Mat<E> {
        Self::__compute_q_impl(self.factors.as_ref(), self.householder.as_ref(), true)
    }

    fn __compute_q_impl(factors: MatRef<'_, E>, householder: MatRef<'_, E>, thin: bool) -> Mat<E> {
        let parallelism = get_global_parallelism();
        let m = factors.nrows();
        let size = Ord::min(m, factors.ncols());

        let mut q = Mat::<E>::zeros(m, if thin { size } else { m });
        q.as_mut()
            .diagonal_mut()
            .column_vector_mut()
            .fill(E::faer_one());

        crate::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                factors,
                householder,
                Conj::No,
                q.as_mut(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    crate::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_req::<E>(
                        m,
                        householder.nrows(),
                        m,
                    )
                    .unwrap(),
                )),
            );

        q
    }
}
impl<E: ComplexField> SpSolverCore<E> for Qr<E> {
    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        assert!(self.nrows() == self.ncols());
        self.solve_lstsq_in_place_with_conj_impl(rhs, conj)
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        assert!(self.nrows() == self.ncols());

        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();

        crate::linalg::qr::no_pivoting::solve::solve_transpose_in_place(
            self.factors.as_ref(),
            self.householder.as_ref(),
            conj,
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::qr::no_pivoting::solve::solve_transpose_in_place_req::<E>(
                    self.nrows(),
                    self.blocksize(),
                    rhs_ncols,
                )
                .unwrap(),
            )),
        );
    }

    fn nrows(&self) -> usize {
        self.factors.nrows()
    }

    fn ncols(&self) -> usize {
        self.factors.ncols()
    }
}
impl<E: ComplexField> SolverCore<E> for Qr<E> {
    fn reconstruct(&self) -> Mat<E> {
        let mut rec = Mat::<E>::zeros(self.nrows(), self.ncols());
        let parallelism = get_global_parallelism();

        crate::linalg::qr::no_pivoting::reconstruct::reconstruct(
            rec.as_mut(),
            self.factors.as_ref(),
            self.householder.as_ref(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::qr::no_pivoting::reconstruct::reconstruct_req::<E>(
                    self.nrows(),
                    self.ncols(),
                    self.blocksize(),
                    parallelism,
                )
                .unwrap(),
            )),
        );

        rec
    }

    fn inverse(&self) -> Mat<E> {
        assert!(self.nrows() == self.ncols());

        let mut inv = Mat::<E>::zeros(self.nrows(), self.ncols());
        let parallelism = get_global_parallelism();

        crate::linalg::qr::no_pivoting::inverse::invert(
            inv.as_mut(),
            self.factors.as_ref(),
            self.householder.as_ref(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::qr::no_pivoting::inverse::invert_req::<E>(
                    self.nrows(),
                    self.ncols(),
                    self.blocksize(),
                    parallelism,
                )
                .unwrap(),
            )),
        );

        inv
    }
}

impl<E: ComplexField> SpSolverLstsqCore<E> for Qr<E> {
    #[track_caller]
    fn solve_lstsq_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();

        crate::linalg::qr::no_pivoting::solve::solve_in_place(
            self.factors.as_ref(),
            self.householder.as_ref(),
            conj,
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::qr::no_pivoting::solve::solve_in_place_req::<E>(
                    self.nrows(),
                    self.blocksize(),
                    rhs_ncols,
                )
                .unwrap(),
            )),
        );
    }
}
impl<E: ComplexField> SolverLstsqCore<E> for Qr<E> {}

impl<E: ComplexField> ColPivQr<E> {
    /// Returns the QR decomposition of the input matrix with column pivoting.
    ///
    /// The factorization is such that $AP^\top = QR$, where $R$ is upper trapezoidal, $Q$ is
    /// unitary, and $P$ is a permutation matrix.
    #[track_caller]
    pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
        let parallelism = get_global_parallelism();
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();

        let mut factors = matrix.to_owned();
        let size = Ord::min(nrows, ncols);
        let blocksize =
            crate::linalg::qr::col_pivoting::compute::recommended_blocksize::<E>(nrows, ncols);
        let mut householder = Mat::<E>::zeros(blocksize, size);

        let params = Default::default();

        let mut col_perm = vec![0usize; ncols];
        let mut col_perm_inv = vec![0usize; ncols];

        crate::linalg::qr::col_pivoting::compute::qr_in_place(
            factors.as_mut(),
            householder.as_mut(),
            &mut col_perm,
            &mut col_perm_inv,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::qr::col_pivoting::compute::qr_in_place_req::<usize, E>(
                    nrows,
                    ncols,
                    blocksize,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        Self {
            factors,
            householder,
            col_perm,
            col_perm_inv,
        }
    }

    /// Returns the column permutation matrix $P$ of the QR decomposition.
    pub fn col_permutation(&self) -> PermRef<'_, usize> {
        unsafe { PermRef::new_unchecked(&self.col_perm, &self.col_perm_inv) }
    }

    fn blocksize(&self) -> usize {
        self.householder.nrows()
    }

    /// Returns the factor $R$ of the QR decomposition.
    pub fn compute_r(&self) -> Mat<E> {
        let mut factor = self.factors.to_owned();
        zipped!(factor.as_mut())
            .for_each_triangular_lower(crate::linalg::zip::Diag::Skip, |unzipped!(mut dst)| {
                dst.write(E::faer_zero())
            });
        factor
    }

    /// Returns the factor $Q$ of the QR decomposition.
    pub fn compute_q(&self) -> Mat<E> {
        Qr::<E>::__compute_q_impl(self.factors.as_ref(), self.householder.as_ref(), false)
    }

    /// Returns the top $r$ rows of the factor $R$ of the QR decomposition, where $r =
    /// \min(\text{nrows}(A), \text{ncols}(A))$.
    pub fn compute_thin_r(&self) -> Mat<E> {
        let m = self.nrows();
        let n = self.ncols();
        let mut factor = self.factors.as_ref().subrows(0, Ord::min(m, n)).to_owned();
        zipped!(factor.as_mut())
            .for_each_triangular_lower(crate::linalg::zip::Diag::Skip, |unzipped!(mut dst)| {
                dst.write(E::faer_zero())
            });
        factor
    }

    /// Returns the leftmost $r$ columns of the factor $R$ of the QR decomposition, where $r =
    /// \min(\text{nrows}(A), \text{ncols}(A))$.
    pub fn compute_thin_q(&self) -> Mat<E> {
        Qr::<E>::__compute_q_impl(self.factors.as_ref(), self.householder.as_ref(), true)
    }
}
impl<E: ComplexField> SpSolverCore<E> for ColPivQr<E> {
    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        assert!(self.nrows() == self.ncols());
        self.solve_lstsq_in_place_with_conj_impl(rhs, conj);
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        assert!(self.nrows() == self.ncols());

        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();

        crate::linalg::qr::col_pivoting::solve::solve_transpose_in_place(
            self.factors.as_ref(),
            self.householder.as_ref(),
            self.col_permutation(),
            conj,
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::qr::col_pivoting::solve::solve_transpose_in_place_req::<usize, E>(
                    self.nrows(),
                    self.blocksize(),
                    rhs_ncols,
                )
                .unwrap(),
            )),
        );
    }

    fn nrows(&self) -> usize {
        self.factors.nrows()
    }

    fn ncols(&self) -> usize {
        self.factors.ncols()
    }
}
impl<E: ComplexField> SolverCore<E> for ColPivQr<E> {
    fn reconstruct(&self) -> Mat<E> {
        let mut rec = Mat::<E>::zeros(self.nrows(), self.ncols());
        let parallelism = get_global_parallelism();

        crate::linalg::qr::col_pivoting::reconstruct::reconstruct(
            rec.as_mut(),
            self.factors.as_ref(),
            self.householder.as_ref(),
            self.col_permutation(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::qr::col_pivoting::reconstruct::reconstruct_req::<usize, E>(
                    self.nrows(),
                    self.ncols(),
                    self.blocksize(),
                    parallelism,
                )
                .unwrap(),
            )),
        );

        rec
    }

    fn inverse(&self) -> Mat<E> {
        assert!(self.nrows() == self.ncols());

        let mut inv = Mat::<E>::zeros(self.nrows(), self.ncols());
        let parallelism = get_global_parallelism();

        crate::linalg::qr::col_pivoting::inverse::invert(
            inv.as_mut(),
            self.factors.as_ref(),
            self.householder.as_ref(),
            self.col_permutation(),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::qr::col_pivoting::inverse::invert_req::<usize, E>(
                    self.nrows(),
                    self.ncols(),
                    self.blocksize(),
                    parallelism,
                )
                .unwrap(),
            )),
        );

        inv
    }
}

impl<E: ComplexField> SpSolverLstsqCore<E> for ColPivQr<E> {
    #[track_caller]
    fn solve_lstsq_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        let parallelism = get_global_parallelism();
        let rhs_ncols = rhs.ncols();

        crate::linalg::qr::col_pivoting::solve::solve_in_place(
            self.factors.as_ref(),
            self.householder.as_ref(),
            self.col_permutation(),
            conj,
            rhs,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::qr::col_pivoting::solve::solve_in_place_req::<usize, E>(
                    self.nrows(),
                    self.blocksize(),
                    rhs_ncols,
                )
                .unwrap(),
            )),
        );
    }
}
impl<E: ComplexField> SolverLstsqCore<E> for ColPivQr<E> {}

impl<E: ComplexField> Svd<E> {
    #[track_caller]
    fn __new_impl((matrix, conj): (MatRef<'_, E>, Conj), thin: bool) -> Self {
        let parallelism = get_global_parallelism();
        let m = matrix.nrows();
        let n = matrix.ncols();
        let size = Ord::min(m, n);

        let mut s = Mat::<E>::zeros(size, 1);
        let mut u = Mat::<E>::zeros(m, if thin { size } else { m });
        let mut v = Mat::<E>::zeros(n, if thin { size } else { n });

        let params = Default::default();

        let compute_vecs = if thin {
            crate::linalg::svd::ComputeVectors::Thin
        } else {
            crate::linalg::svd::ComputeVectors::Full
        };

        crate::linalg::svd::compute_svd(
            matrix,
            s.as_mut(),
            Some(u.as_mut()),
            Some(v.as_mut()),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::svd::compute_svd_req::<E>(
                    m,
                    n,
                    compute_vecs,
                    compute_vecs,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        if matches!(conj, Conj::Yes) {
            zipped!(u.as_mut()).for_each(|unzipped!(mut x)| x.write(x.read().faer_conj()));
            zipped!(v.as_mut()).for_each(|unzipped!(mut x)| x.write(x.read().faer_conj()));
        }

        Self { s, u, v }
    }

    /// Returns the SVD of the input matrix.
    ///
    /// The factorization is such that $A = U S V^H$, where $U$ and $V$ are unitary and $S$ is a
    /// rectangular diagonal matrix.
    #[track_caller]
    pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
        Self::__new_impl(matrix.canonicalize(), false)
    }

    /// Returns the factor $U$ of the SVD.
    pub fn u(&self) -> MatRef<'_, E> {
        self.u.as_ref()
    }
    /// Returns the diagonal of the factor $S$ of the SVD as a column vector.
    pub fn s_diagonal(&self) -> ColRef<'_, E> {
        self.s.as_ref().col(0)
    }
    /// Returns the factor $V$ of the SVD.
    pub fn v(&self) -> MatRef<'_, E> {
        self.v.as_ref()
    }
}
fn div_by_s<E: ComplexField>(rhs: MatMut<'_, E>, s: MatRef<'_, E>) {
    let mut rhs = rhs;
    for j in 0..rhs.ncols() {
        zipped!(rhs.rb_mut().col_mut(j).as_2d_mut(), s).for_each(|unzipped!(mut rhs, s)| {
            rhs.write(rhs.read().faer_scale_real(s.read().faer_real().faer_inv()))
        });
    }
}
impl<E: ComplexField> SpSolverCore<E> for Svd<E> {
    fn nrows(&self) -> usize {
        self.u.nrows()
    }

    fn ncols(&self) -> usize {
        self.v.nrows()
    }

    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        assert!(self.nrows() == self.ncols());
        let mut rhs = rhs;

        let u = self.u.as_ref();
        let v = self.v.as_ref();
        let s = self.s.as_ref();

        match conj {
            Conj::Yes => {
                rhs.copy_from((u.transpose() * rhs.rb()).as_ref());
                div_by_s(rhs.rb_mut(), s);
                rhs.copy_from((v.conjugate() * rhs.rb()).as_ref());
            }
            Conj::No => {
                rhs.copy_from((u.adjoint() * rhs.rb()).as_ref());
                div_by_s(rhs.rb_mut(), s);
                rhs.copy_from((v * rhs.rb()).as_ref());
            }
        }
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        assert!(self.nrows() == self.ncols());
        let mut rhs = rhs;

        let u = self.u.as_ref();
        let v = self.v.as_ref();
        let s = self.s.as_ref();

        match conj {
            Conj::No => {
                rhs.copy_from((v.transpose() * rhs.rb()).as_ref());
                div_by_s(rhs.rb_mut(), s);
                rhs.copy_from((u.conjugate() * rhs.rb()).as_ref());
            }
            Conj::Yes => {
                rhs.copy_from((v.adjoint() * rhs.rb()).as_ref());
                div_by_s(rhs.rb_mut(), s);
                rhs.copy_from((u * rhs.rb()).as_ref());
            }
        }
    }
}
impl<E: ComplexField> SolverCore<E> for Svd<E> {
    fn reconstruct(&self) -> Mat<E> {
        let m = self.nrows();
        let n = self.ncols();
        let size = Ord::min(m, n);

        let thin_u = self.u.as_ref().submatrix(0, 0, m, size);
        let s = self.s.as_ref();
        let us = Mat::<E>::from_fn(m, size, |i, j| thin_u.read(i, j).faer_mul(s.read(j, 0)));

        us * self.v.adjoint()
    }

    fn inverse(&self) -> Mat<E> {
        assert!(self.nrows() == self.ncols());
        let dim = self.nrows();

        let u = self.u.as_ref();
        let v = self.v.as_ref();
        let s = self.s.as_ref();

        let vs_inv = Mat::<E>::from_fn(dim, dim, |i, j| {
            v.read(i, j).faer_mul(s.read(j, 0).faer_inv())
        });

        vs_inv * u.adjoint()
    }
}

impl<E: ComplexField> ThinSvd<E> {
    /// Returns the thin SVD of the input matrix.
    ///
    /// This is the same as the SVD except that only the leftmost $r$ columns of $U$ and $V$ are
    /// computed, where $r = \min(\text{nrows}(A), \text{ncols}(A))$.
    #[track_caller]
    pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
        Self {
            inner: Svd::__new_impl(matrix.canonicalize(), true),
        }
    }

    /// Returns the factor $U$ of the SVD.
    pub fn u(&self) -> MatRef<'_, E> {
        self.inner.u.as_ref()
    }
    /// Returns the diagonal of the factor $S$ of the SVD as a column vector.
    pub fn s_diagonal(&self) -> ColRef<'_, E> {
        self.inner.s.as_ref().col(0)
    }
    /// Returns the factor $V$ of the SVD.
    pub fn v(&self) -> MatRef<'_, E> {
        self.inner.v.as_ref()
    }
}
impl<E: ComplexField> SpSolverCore<E> for ThinSvd<E> {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        self.inner.solve_in_place_with_conj_impl(rhs, conj)
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        self.inner
            .solve_transpose_in_place_with_conj_impl(rhs, conj)
    }
}
impl<E: ComplexField> SolverCore<E> for ThinSvd<E> {
    fn reconstruct(&self) -> Mat<E> {
        self.inner.reconstruct()
    }

    fn inverse(&self) -> Mat<E> {
        self.inner.inverse()
    }
}

impl<E: ComplexField> SelfAdjointEigendecomposition<E> {
    #[track_caller]
    fn __new_impl((matrix, conj): (MatRef<'_, E>, Conj), side: Side) -> Self {
        assert!(matrix.nrows() == matrix.ncols());
        let parallelism = get_global_parallelism();

        let dim = matrix.nrows();

        let mut s = Mat::<E>::zeros(dim, 1);
        let mut u = Mat::<E>::zeros(dim, dim);

        let matrix = match side {
            Side::Lower => matrix,
            Side::Upper => matrix.transpose(),
        };
        let conj = conj.compose(match side {
            Side::Lower => Conj::No,
            Side::Upper => Conj::Yes,
        });

        let params = Default::default();
        crate::linalg::evd::compute_hermitian_evd(
            matrix,
            s.as_mut(),
            Some(u.as_mut()),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::evd::compute_hermitian_evd_req::<E>(
                    dim,
                    crate::linalg::evd::ComputeVectors::Yes,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        if matches!(conj, Conj::Yes) {
            zipped!(u.as_mut()).for_each(|unzipped!(mut x)| x.write(x.read().faer_conj()));
        }

        Self { s, u }
    }

    /// Returns the eigenvalue decomposition of the Hermitian input matrix.
    ///
    /// The factorization is such that $A = U S U^\H$, where $S$ is a diagonal matrix, and $U$ is
    /// unitary.
    ///
    /// Only the provided side is accessed.
    #[track_caller]
    pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>, side: Side) -> Self {
        Self::__new_impl(matrix.canonicalize(), side)
    }

    /// Returns the factor $U$ of the eigenvalue decomposition.
    pub fn u(&self) -> MatRef<'_, E> {
        self.u.as_ref()
    }
    /// Returns the factor $S$ of the eigenvalue decomposition.
    pub fn s(&self) -> DiagRef<'_, E> {
        self.s.as_ref().col(0).column_vector_as_diagonal()
    }
}
impl<E: ComplexField> SpSolverCore<E> for SelfAdjointEigendecomposition<E> {
    fn nrows(&self) -> usize {
        self.u.nrows()
    }

    fn ncols(&self) -> usize {
        self.u.nrows()
    }

    #[track_caller]
    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        assert!(self.nrows() == self.ncols());
        let mut rhs = rhs;

        let u = self.u.as_ref();
        let s = self.s.as_ref();

        match conj {
            Conj::Yes => {
                rhs.copy_from((u.transpose() * rhs.rb()).as_ref());
                div_by_s(rhs.rb_mut(), s);
                rhs.copy_from((u.conjugate() * rhs.rb()).as_ref());
            }
            Conj::No => {
                rhs.copy_from((u.adjoint() * rhs.rb()).as_ref());
                div_by_s(rhs.rb_mut(), s);
                rhs.copy_from((u * rhs.rb()).as_ref());
            }
        }
    }

    #[track_caller]
    fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
        assert!(self.nrows() == self.ncols());
        let mut rhs = rhs;

        let u = self.u.as_ref();
        let s = self.s.as_ref();

        match conj {
            Conj::No => {
                rhs.copy_from((u.transpose() * rhs.rb()).as_ref());
                div_by_s(rhs.rb_mut(), s);
                rhs.copy_from((u.conjugate() * rhs.rb()).as_ref());
            }
            Conj::Yes => {
                rhs.copy_from((u.adjoint() * rhs.rb()).as_ref());
                div_by_s(rhs.rb_mut(), s);
                rhs.copy_from((u * rhs.rb()).as_ref());
            }
        }
    }
}
impl<E: ComplexField> SolverCore<E> for SelfAdjointEigendecomposition<E> {
    fn reconstruct(&self) -> Mat<E> {
        let size = self.nrows();

        let u = self.u.as_ref();
        let s = self.s.as_ref();
        let us = Mat::<E>::from_fn(size, size, |i, j| u.read(i, j).faer_mul(s.read(j, 0)));

        us * u.adjoint()
    }

    fn inverse(&self) -> Mat<E> {
        let dim = self.nrows();

        let u = self.u.as_ref();
        let s = self.s.as_ref();

        let us_inv = Mat::<E>::from_fn(dim, dim, |i, j| {
            u.read(i, j).faer_mul(s.read(j, 0).faer_inv())
        });

        us_inv * u.adjoint()
    }
}

impl<E: ComplexField> Eigendecomposition<E> {
    #[track_caller]
    pub(crate) fn __values_from_real(matrix: MatRef<'_, E::Real>) -> Vec<E> {
        assert!(matrix.nrows() == matrix.ncols());
        if coe::is_same::<E, E::Real>() {
            panic!(
                "The type E ({}) must not be real-valued.",
                core::any::type_name::<E>(),
            );
        }

        let parallelism = get_global_parallelism();

        let dim = matrix.nrows();
        let mut s_re = Mat::<E::Real>::zeros(dim, 1);
        let mut s_im = Mat::<E::Real>::zeros(dim, 1);

        let params = Default::default();

        crate::linalg::evd::compute_evd_real(
            matrix,
            s_re.as_mut(),
            s_im.as_mut(),
            None,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::evd::compute_evd_req::<E::Real>(
                    dim,
                    crate::linalg::evd::ComputeVectors::Yes,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        let imag = E::faer_from_f64(-1.0).faer_sqrt();
        let cplx = |re: E::Real, im: E::Real| -> E {
            E::faer_from_real(re).faer_add(imag.faer_mul(E::faer_from_real(im)))
        };

        (0..dim)
            .map(|i| cplx(s_re.read(i, 0), s_im.read(i, 0)))
            .collect()
    }

    #[track_caller]
    pub(crate) fn __values_from_complex_impl((matrix, conj): (MatRef<'_, E>, Conj)) -> Vec<E> {
        assert!(matrix.nrows() == matrix.ncols());
        if coe::is_same::<E, E::Real>() {
            panic!(
                "The type E ({}) must not be real-valued.",
                core::any::type_name::<E>(),
            );
        }

        let parallelism = get_global_parallelism();
        let dim = matrix.nrows();

        let mut s = Mat::<E>::zeros(dim, 1);

        let params = Default::default();

        crate::linalg::evd::compute_evd_complex(
            matrix,
            s.as_mut(),
            None,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::evd::compute_evd_req::<E>(
                    dim,
                    crate::linalg::evd::ComputeVectors::Yes,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        if matches!(conj, Conj::Yes) {
            zipped!(s.as_mut()).for_each(|unzipped!(mut x)| x.write(x.read().faer_conj()));
        }

        (0..dim).map(|i| s.read(i, 0)).collect()
    }

    /// Returns the eigendecomposition of the real-valued input matrix.
    ///
    /// The factorization is such that $A = U S U^\H$, where $S$ is a diagonal matrix, and $U$ is
    /// unitary.
    #[track_caller]
    pub fn new_from_real(matrix: MatRef<'_, E::Real>) -> Self {
        assert!(matrix.nrows() == matrix.ncols());
        if coe::is_same::<E, E::Real>() {
            panic!(
                "The type E ({}) must not be real-valued.",
                core::any::type_name::<E>(),
            );
        }

        let parallelism = get_global_parallelism();

        let dim = matrix.nrows();
        let mut s_re = Col::<E::Real>::zeros(dim);
        let mut s_im = Col::<E::Real>::zeros(dim);
        let mut u_real = Mat::<E::Real>::zeros(dim, dim);

        let params = Default::default();

        crate::linalg::evd::compute_evd_real(
            matrix,
            s_re.as_mut().as_2d_mut(),
            s_im.as_mut().as_2d_mut(),
            Some(u_real.as_mut()),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::evd::compute_evd_req::<E::Real>(
                    dim,
                    crate::linalg::evd::ComputeVectors::Yes,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        let imag = E::faer_from_f64(-1.0).faer_sqrt();
        let cplx = |re: E::Real, im: E::Real| -> E {
            E::faer_from_real(re).faer_add(imag.faer_mul(E::faer_from_real(im)))
        };

        let s = Col::<E>::from_fn(dim, |i| cplx(s_re.read(i), s_im.read(i)));
        let mut u = Mat::<E>::zeros(dim, dim);
        let u_real = u_real.as_ref();

        let mut j = 0usize;
        while j < dim {
            if s_im.read(j) == E::Real::faer_zero() {
                zipped!(u.as_mut().col_mut(j).as_2d_mut(), u_real.col(j).as_2d())
                    .for_each(|unzipped!(mut dst, src)| dst.write(E::faer_from_real(src.read())));
                j += 1;
            } else {
                let (u_left, u_right) = u.as_mut().split_at_col_mut(j + 1);

                zipped!(
                    u_left.col_mut(j).as_2d_mut(),
                    u_right.col_mut(0).as_2d_mut(),
                    u_real.col(j).as_2d(),
                    u_real.col(j + 1).as_2d(),
                )
                .for_each(|unzipped!(mut dst, mut dst_conj, re, im)| {
                    let re = re.read();
                    let im = im.read();
                    dst_conj.write(cplx(re, im.faer_neg()));
                    dst.write(cplx(re, im));
                });

                j += 2;
            }
        }

        Self { s, u }
    }

    #[track_caller]
    pub(crate) fn __new_from_complex_impl((matrix, conj): (MatRef<'_, E>, Conj)) -> Self {
        assert!(matrix.nrows() == matrix.ncols());
        if coe::is_same::<E, E::Real>() {
            panic!(
                "The type E ({}) must not be real-valued.",
                core::any::type_name::<E>(),
            );
        }

        let parallelism = get_global_parallelism();
        let dim = matrix.nrows();

        let mut s = Col::<E>::zeros(dim);
        let mut u = Mat::<E>::zeros(dim, dim);

        let params = Default::default();

        crate::linalg::evd::compute_evd_complex(
            matrix,
            s.as_mut().as_2d_mut(),
            Some(u.as_mut()),
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::evd::compute_evd_req::<E>(
                    dim,
                    crate::linalg::evd::ComputeVectors::Yes,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        if matches!(conj, Conj::Yes) {
            zipped!(s.as_mut().as_2d_mut())
                .for_each(|unzipped!(mut x)| x.write(x.read().faer_conj()));
            zipped!(u.as_mut()).for_each(|unzipped!(mut x)| x.write(x.read().faer_conj()));
        }

        Self { s, u }
    }

    /// Returns the eigendecomposition of the complex-valued input matrix.
    ///
    /// The factorization is such that $A = U S U^\H$, where $S$ is a diagonal matrix, and $U$ is
    /// unitary.
    #[track_caller]
    pub fn new_from_complex<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
        Self::__new_from_complex_impl(matrix.canonicalize())
    }

    /// Returns the factor $U$ of the eigenvalue decomposition.
    pub fn u(&self) -> MatRef<'_, E> {
        self.u.as_ref()
    }
    /// Returns the factor $S$ of the eigenvalue decomposition.
    pub fn s(&self) -> DiagRef<'_, E> {
        self.s.as_ref().column_vector_as_diagonal()
    }
}

impl<E: Conjugate> MatRef<'_, E>
where
    E::Canonical: ComplexField,
{
    /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    #[track_caller]
    pub fn solve_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        let parallelism = get_global_parallelism();
        let mut rhs = rhs;
        crate::linalg::triangular_solve::solve_lower_triangular_in_place(
            *self,
            rhs.as_2d_mut(),
            parallelism,
        );
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    #[track_caller]
    pub fn solve_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        let parallelism = get_global_parallelism();
        let mut rhs = rhs;
        crate::linalg::triangular_solve::solve_upper_triangular_in_place(
            *self,
            rhs.as_2d_mut(),
            parallelism,
        );
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        let parallelism = get_global_parallelism();
        let mut rhs = rhs;
        crate::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
            *self,
            rhs.as_2d_mut(),
            parallelism,
        );
    }
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        let parallelism = get_global_parallelism();
        let mut rhs = rhs;
        crate::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
            *self,
            rhs.as_2d_mut(),
            parallelism,
        );
    }

    /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    #[track_caller]
    pub fn solve_lower_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        let mut rhs = B::new_owned_copied(&rhs);
        self.solve_lower_triangular_in_place(rhs.as_2d_mut());
        rhs
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    #[track_caller]
    pub fn solve_upper_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        let mut rhs = B::new_owned_copied(&rhs);
        self.solve_upper_triangular_in_place(rhs.as_2d_mut());
        rhs
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_lower_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        let mut rhs = B::new_owned_copied(&rhs);
        self.solve_unit_lower_triangular_in_place(rhs.as_2d_mut());
        rhs
    }
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_upper_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        let mut rhs = B::new_owned_copied(&rhs);
        self.solve_unit_upper_triangular_in_place(rhs.as_2d_mut());
        rhs
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    pub fn cholesky(&self, side: Side) -> Result<Cholesky<E::Canonical>, CholeskyError> {
        Cholesky::try_new(self.as_ref(), side)
    }
    /// Returns the Bunch-Kaufman decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    pub fn lblt(&self, side: Side) -> Lblt<E::Canonical> {
        Lblt::new(self.as_ref(), side)
    }
    /// Returns the LU decomposition of `self` with partial (row) pivoting.
    #[track_caller]
    pub fn partial_piv_lu(&self) -> PartialPivLu<E::Canonical> {
        PartialPivLu::<E::Canonical>::new(self.as_ref())
    }
    /// Returns the LU decomposition of `self` with full pivoting.
    #[track_caller]
    pub fn full_piv_lu(&self) -> FullPivLu<E::Canonical> {
        FullPivLu::<E::Canonical>::new(self.as_ref())
    }
    /// Returns the QR decomposition of `self`.
    #[track_caller]
    pub fn qr(&self) -> Qr<E::Canonical> {
        Qr::<E::Canonical>::new(self.as_ref())
    }
    /// Returns the QR decomposition of `self` with column pivoting.
    #[track_caller]
    pub fn col_piv_qr(&self) -> ColPivQr<E::Canonical> {
        ColPivQr::<E::Canonical>::new(self.as_ref())
    }
    /// Returns the SVD of `self`.
    #[track_caller]
    pub fn svd(&self) -> Svd<E::Canonical> {
        Svd::<E::Canonical>::new(self.as_ref())
    }
    /// Returns the thin SVD of `self`.
    #[track_caller]
    pub fn thin_svd(&self) -> ThinSvd<E::Canonical> {
        ThinSvd::<E::Canonical>::new(self.as_ref())
    }
    /// Returns the eigendecomposition of `self`, assuming it is self-adjoint. Only the provided
    /// side is accessed.
    #[track_caller]
    pub fn selfadjoint_eigendecomposition(
        &self,
        side: Side,
    ) -> SelfAdjointEigendecomposition<E::Canonical> {
        SelfAdjointEigendecomposition::<E::Canonical>::new(self.as_ref(), side)
    }

    /// Returns the eigendecomposition of `self`, as a complex matrix.
    #[track_caller]
    pub fn eigendecomposition<
        ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>,
    >(
        &self,
    ) -> Eigendecomposition<ComplexE> {
        if coe::is_same::<E, <E::Canonical as ComplexField>::Real>() {
            let matrix: MatRef<'_, <E::Canonical as ComplexField>::Real> =
                coe::coerce(self.as_ref());
            Eigendecomposition::<ComplexE>::new_from_real(matrix)
        } else if coe::is_same::<E::Canonical, ComplexE>() {
            let (matrix, conj) = self.as_ref().canonicalize();
            Eigendecomposition::<ComplexE>::__new_from_complex_impl((coe::coerce(matrix), conj))
        } else {
            panic!(
                "The type ComplexE must be either E::Canonical ({}) or E::Canonical::Real ({})",
                core::any::type_name::<E::Canonical>(),
                core::any::type_name::<<E::Canonical as ComplexField>::Real>(),
            );
        }
    }

    /// Returns the eigendecomposition of `self`, when `E` is in the complex domain.
    #[track_caller]
    pub fn complex_eigendecomposition(&self) -> Eigendecomposition<E::Canonical> {
        Eigendecomposition::<E::Canonical>::new_from_complex(self.as_ref())
    }

    /// Returns the determinant of `self`.
    #[track_caller]
    pub fn determinant(&self) -> E::Canonical {
        assert!(self.nrows() == self.ncols());
        let lu = self.partial_piv_lu();
        let mut det = E::Canonical::faer_one();
        for i in 0..self.nrows() {
            det = det.faer_mul(lu.factors.read(i, i));
        }
        if lu.transposition_count() % 2 == 0 {
            det
        } else {
            det.faer_neg()
        }
    }

    /// Returns the eigenvalues of `self`, assuming it is self-adjoint. Only the provided
    /// side is accessed. The order of the eigenvalues is currently unspecified.
    #[track_caller]
    pub fn selfadjoint_eigenvalues(&self, side: Side) -> Vec<<E::Canonical as ComplexField>::Real> {
        let matrix = match side {
            Side::Lower => *self,
            Side::Upper => self.transpose(),
        };

        assert!(matrix.nrows() == matrix.ncols());
        let dim = matrix.nrows();
        let parallelism = get_global_parallelism();

        let mut s = Mat::<E::Canonical>::zeros(dim, 1);
        let params = Default::default();
        crate::linalg::evd::compute_hermitian_evd(
            matrix.canonicalize().0,
            s.as_mut(),
            None,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::evd::compute_hermitian_evd_req::<E::Canonical>(
                    dim,
                    crate::linalg::evd::ComputeVectors::No,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        (0..dim).map(|i| s.read(i, 0).faer_real()).collect()
    }

    /// Returns the singular values of `self`, in nonincreasing order.
    #[track_caller]
    pub fn singular_values(&self) -> Vec<<E::Canonical as ComplexField>::Real> {
        let dim = Ord::min(self.nrows(), self.ncols());
        let parallelism = get_global_parallelism();

        let mut s = Mat::<E::Canonical>::zeros(dim, 1);
        let params = Default::default();
        crate::linalg::svd::compute_svd(
            self.canonicalize().0,
            s.as_mut(),
            None,
            None,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                crate::linalg::svd::compute_svd_req::<E::Canonical>(
                    self.nrows(),
                    self.ncols(),
                    crate::linalg::svd::ComputeVectors::No,
                    crate::linalg::svd::ComputeVectors::No,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        (0..dim).map(|i| s.read(i, 0).faer_real()).collect()
    }

    /// Returns the eigenvalues of `self`, as complex values. The order of the eigenvalues is
    /// currently unspecified.
    #[track_caller]
    pub fn eigenvalues<ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>>(
        &self,
    ) -> Vec<ComplexE> {
        if coe::is_same::<E, <E::Canonical as ComplexField>::Real>() {
            let matrix: MatRef<'_, <E::Canonical as ComplexField>::Real> =
                coe::coerce(self.as_ref());
            Eigendecomposition::<ComplexE>::__values_from_real(matrix)
        } else if coe::is_same::<E::Canonical, ComplexE>() {
            let (matrix, conj) = self.as_ref().canonicalize();
            Eigendecomposition::<ComplexE>::__values_from_complex_impl((coe::coerce(matrix), conj))
        } else {
            panic!(
                "The type ComplexE must be either E::Canonical ({}) or E::Canonical::Real ({})",
                core::any::type_name::<E::Canonical>(),
                core::any::type_name::<<E::Canonical as ComplexField>::Real>(),
            );
        }
    }

    /// Returns the eigenvalues of `self`, when `E` is in the complex domain. The order of the
    /// eigenvalues is currently unspecified.
    #[track_caller]
    pub fn complex_eigenvalues(&self) -> Vec<E::Canonical> {
        Eigendecomposition::<E::Canonical>::__values_from_complex_impl(self.canonicalize())
    }
}

impl<E: Conjugate> MatMut<'_, E>
where
    E::Canonical: ComplexField,
{
    /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    #[track_caller]
    pub fn solve_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        self.as_ref().solve_lower_triangular_in_place(rhs)
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    #[track_caller]
    pub fn solve_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        self.as_ref().solve_upper_triangular_in_place(rhs)
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        self.as_ref().solve_unit_lower_triangular_in_place(rhs)
    }
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        self.as_ref().solve_unit_upper_triangular_in_place(rhs)
    }

    /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    #[track_caller]
    pub fn solve_lower_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        self.as_ref().solve_lower_triangular(rhs)
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    #[track_caller]
    pub fn solve_upper_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        self.as_ref().solve_upper_triangular(rhs)
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_lower_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        self.as_ref().solve_unit_lower_triangular(rhs)
    }
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_upper_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        self.as_ref().solve_unit_upper_triangular(rhs)
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    pub fn cholesky(&self, side: Side) -> Result<Cholesky<E::Canonical>, CholeskyError> {
        self.as_ref().cholesky(side)
    }
    /// Returns the Bunch-Kaufman decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    pub fn lblt(&self, side: Side) -> Lblt<E::Canonical> {
        self.as_ref().lblt(side)
    }
    /// Returns the LU decomposition of `self` with partial (row) pivoting.
    #[track_caller]
    pub fn partial_piv_lu(&self) -> PartialPivLu<E::Canonical> {
        self.as_ref().partial_piv_lu()
    }
    /// Returns the LU decomposition of `self` with full pivoting.
    #[track_caller]
    pub fn full_piv_lu(&self) -> FullPivLu<E::Canonical> {
        self.as_ref().full_piv_lu()
    }
    /// Returns the QR decomposition of `self`.
    #[track_caller]
    pub fn qr(&self) -> Qr<E::Canonical> {
        self.as_ref().qr()
    }
    /// Returns the QR decomposition of `self` with column pivoting.
    #[track_caller]
    pub fn col_piv_qr(&self) -> ColPivQr<E::Canonical> {
        self.as_ref().col_piv_qr()
    }
    /// Returns the SVD of `self`.
    #[track_caller]
    pub fn svd(&self) -> Svd<E::Canonical> {
        self.as_ref().svd()
    }
    /// Returns the thin SVD of `self`.
    #[track_caller]
    pub fn thin_svd(&self) -> ThinSvd<E::Canonical> {
        self.as_ref().thin_svd()
    }
    /// Returns the eigendecomposition of `self`, assuming it is self-adjoint. Only the provided
    /// side is accessed.
    #[track_caller]
    pub fn selfadjoint_eigendecomposition(
        &self,
        side: Side,
    ) -> SelfAdjointEigendecomposition<E::Canonical> {
        self.as_ref().selfadjoint_eigendecomposition(side)
    }

    /// Returns the eigendecomposition of `self`, as a complex matrix.
    #[track_caller]
    pub fn eigendecomposition<
        ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>,
    >(
        &self,
    ) -> Eigendecomposition<ComplexE> {
        self.as_ref().eigendecomposition::<ComplexE>()
    }

    /// Returns the eigendecomposition of `self`, when `E` is in the complex domain.
    #[track_caller]
    pub fn complex_eigendecomposition(&self) -> Eigendecomposition<E::Canonical> {
        self.as_ref().complex_eigendecomposition()
    }

    /// Returns the determinant of `self`.
    #[track_caller]
    pub fn determinant(&self) -> E::Canonical {
        self.as_ref().determinant()
    }

    /// Returns the eigenvalues of `self`, assuming it is self-adjoint. Only the provided
    /// side is accessed. The order of the eigenvalues is currently unspecified.
    #[track_caller]
    pub fn selfadjoint_eigenvalues(&self, side: Side) -> Vec<<E::Canonical as ComplexField>::Real> {
        self.as_ref().selfadjoint_eigenvalues(side)
    }

    /// Returns the singular values of `self`, in nonincreasing order.
    #[track_caller]
    pub fn singular_values(&self) -> Vec<<E::Canonical as ComplexField>::Real> {
        self.as_ref().singular_values()
    }

    /// Returns the eigenvalues of `self`, as complex values. The order of the eigenvalues is
    /// currently unspecified.
    #[track_caller]
    pub fn eigenvalues<ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>>(
        &self,
    ) -> Vec<ComplexE> {
        self.as_ref().eigenvalues()
    }

    /// Returns the eigenvalues of `self`, when `E` is in the complex domain. The order of the
    /// eigenvalues is currently unspecified.
    #[track_caller]
    pub fn complex_eigenvalues(&self) -> Vec<E::Canonical> {
        self.as_ref().complex_eigenvalues()
    }
}

impl<E: Conjugate> Mat<E>
where
    E::Canonical: ComplexField,
{
    /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    #[track_caller]
    pub fn solve_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        self.as_ref().solve_lower_triangular_in_place(rhs)
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    #[track_caller]
    pub fn solve_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        self.as_ref().solve_upper_triangular_in_place(rhs)
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_lower_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        self.as_ref().solve_unit_lower_triangular_in_place(rhs)
    }
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_upper_triangular_in_place(&self, rhs: impl ColBatchMut<E::Canonical>) {
        self.as_ref().solve_unit_upper_triangular_in_place(rhs)
    }

    /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    #[track_caller]
    pub fn solve_lower_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        self.as_ref().solve_lower_triangular(rhs)
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    #[track_caller]
    pub fn solve_upper_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        self.as_ref().solve_upper_triangular(rhs)
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_lower_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        self.as_ref().solve_unit_lower_triangular(rhs)
    }
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    ///
    /// The diagonal of the matrix is not accessed.
    #[track_caller]
    pub fn solve_unit_upper_triangular<
        ViewE: Conjugate<Canonical = E::Canonical>,
        B: ColBatch<ViewE>,
    >(
        &self,
        rhs: B,
    ) -> B::Owned {
        self.as_ref().solve_unit_upper_triangular(rhs)
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    pub fn cholesky(&self, side: Side) -> Result<Cholesky<E::Canonical>, CholeskyError> {
        self.as_ref().cholesky(side)
    }
    /// Returns the Bunch-Kaufman decomposition of `self`. Only the provided side is accessed.
    #[track_caller]
    pub fn lblt(&self, side: Side) -> Lblt<E::Canonical> {
        self.as_ref().lblt(side)
    }
    /// Returns the LU decomposition of `self` with partial (row) pivoting.
    #[track_caller]
    pub fn partial_piv_lu(&self) -> PartialPivLu<E::Canonical> {
        self.as_ref().partial_piv_lu()
    }
    /// Returns the LU decomposition of `self` with full pivoting.
    #[track_caller]
    pub fn full_piv_lu(&self) -> FullPivLu<E::Canonical> {
        self.as_ref().full_piv_lu()
    }
    /// Returns the QR decomposition of `self`.
    #[track_caller]
    pub fn qr(&self) -> Qr<E::Canonical> {
        self.as_ref().qr()
    }
    /// Returns the QR decomposition of `self` with column pivoting.
    #[track_caller]
    pub fn col_piv_qr(&self) -> ColPivQr<E::Canonical> {
        self.as_ref().col_piv_qr()
    }
    /// Returns the SVD of `self`.
    #[track_caller]
    pub fn svd(&self) -> Svd<E::Canonical> {
        self.as_ref().svd()
    }
    /// Returns the thin SVD of `self`.
    #[track_caller]
    pub fn thin_svd(&self) -> ThinSvd<E::Canonical> {
        self.as_ref().thin_svd()
    }
    /// Returns the eigendecomposition of `self`, assuming it is self-adjoint. Only the provided
    /// side is accessed.
    #[track_caller]
    pub fn selfadjoint_eigendecomposition(
        &self,
        side: Side,
    ) -> SelfAdjointEigendecomposition<E::Canonical> {
        self.as_ref().selfadjoint_eigendecomposition(side)
    }

    /// Returns the eigendecomposition of `self`, as a complex matrix.
    #[track_caller]
    pub fn eigendecomposition<
        ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>,
    >(
        &self,
    ) -> Eigendecomposition<ComplexE> {
        self.as_ref().eigendecomposition::<ComplexE>()
    }

    /// Returns the eigendecomposition of `self`, when `E` is in the complex domain.
    #[track_caller]
    pub fn complex_eigendecomposition(&self) -> Eigendecomposition<E::Canonical> {
        self.as_ref().complex_eigendecomposition()
    }

    /// Returns the determinant of `self`.
    #[track_caller]
    pub fn determinant(&self) -> E::Canonical {
        self.as_ref().determinant()
    }

    /// Returns the eigenvalues of `self`, assuming it is self-adjoint. Only the provided
    /// side is accessed. The order of the eigenvalues is currently unspecified.
    #[track_caller]
    pub fn selfadjoint_eigenvalues(&self, side: Side) -> Vec<<E::Canonical as ComplexField>::Real> {
        self.as_ref().selfadjoint_eigenvalues(side)
    }

    /// Returns the singular values of `self`, in nonincreasing order.
    #[track_caller]
    pub fn singular_values(&self) -> Vec<<E::Canonical as ComplexField>::Real> {
        self.as_ref().singular_values()
    }

    /// Returns the eigenvalues of `self`, as complex values. The order of the eigenvalues is
    /// currently unspecified.
    #[track_caller]
    pub fn eigenvalues<ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>>(
        &self,
    ) -> Vec<ComplexE> {
        self.as_ref().eigenvalues()
    }

    /// Returns the eigenvalues of `self`, when `E` is in the complex domain. The order of the
    /// eigenvalues is currently unspecified.
    #[track_caller]
    pub fn complex_eigenvalues(&self) -> Vec<E::Canonical> {
        self.as_ref().complex_eigenvalues()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, RealField};
    use complex_native::*;

    #[track_caller]
    fn assert_approx_eq<E: ComplexField>(a: impl AsMatRef<E>, b: impl AsMatRef<E>) {
        let a = a.as_mat_ref();
        let b = b.as_mat_ref();
        let eps = E::Real::faer_epsilon().unwrap().faer_sqrt();

        assert!(a.nrows() == b.nrows());
        assert!(a.ncols() == b.ncols());

        let m = a.nrows();
        let n = a.ncols();

        for j in 0..n {
            for i in 0..m {
                assert!((a.read(i, j).faer_sub(b.read(i, j))).faer_abs() < eps);
            }
        }
    }

    fn test_solver_real(H: impl AsMatRef<f64>, decomp: &dyn SolverCore<f64>) {
        let H = H.as_mat_ref();
        let n = H.nrows();
        let k = 2;

        let random = |_, _| rand::random::<f64>();
        let rhs = Mat::from_fn(n, k, random);

        let I = Mat::from_fn(n, n, |i, j| {
            if i == j {
                f64::faer_one()
            } else {
                f64::faer_zero()
            }
        });

        let sol = decomp.solve(&rhs);
        assert_approx_eq(H * &sol, &rhs);

        let sol = decomp.solve_conj(&rhs);
        assert_approx_eq(H.conjugate() * &sol, &rhs);

        let sol = decomp.solve_transpose(&rhs);
        assert_approx_eq(H.transpose() * &sol, &rhs);

        let sol = decomp.solve_conj_transpose(&rhs);
        assert_approx_eq(H.adjoint() * &sol, &rhs);

        assert_approx_eq(decomp.reconstruct(), H);
        assert_approx_eq(H * decomp.inverse(), I);
    }

    fn test_solver(H: impl AsMatRef<c64>, decomp: &dyn SolverCore<c64>) {
        let H = H.as_mat_ref();
        let n = H.nrows();
        let k = 2;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let rhs = Mat::from_fn(n, k, random);

        let I = Mat::from_fn(n, n, |i, j| {
            if i == j {
                c64::faer_one()
            } else {
                c64::faer_zero()
            }
        });

        let sol = decomp.solve(&rhs);
        assert_approx_eq(H * &sol, &rhs);

        let sol = decomp.solve_conj(&rhs);
        assert_approx_eq(H.conjugate() * &sol, &rhs);

        let sol = decomp.solve_transpose(&rhs);
        assert_approx_eq(H.transpose() * &sol, &rhs);

        let sol = decomp.solve_conj_transpose(&rhs);
        assert_approx_eq(H.adjoint() * &sol, &rhs);

        assert_approx_eq(decomp.reconstruct(), H);
        assert_approx_eq(H * decomp.inverse(), I);
    }

    fn test_solver_lstsq(H: impl AsMatRef<c64>, decomp: &dyn SolverLstsqCore<c64>) {
        let H = H.as_mat_ref();

        let m = H.nrows();
        let k = 2;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let rhs = Mat::from_fn(m, k, random);

        let sol = decomp.solve_lstsq(&rhs);
        assert_approx_eq(H.adjoint() * H * &sol, H.adjoint() * &rhs);

        let sol = decomp.solve_lstsq_conj(&rhs);
        assert_approx_eq(H.transpose() * H.conjugate() * &sol, H.transpose() * &rhs);
    }

    #[test]
    fn test_lblt_real() {
        let n = 7;

        let random = |_, _| rand::random::<f64>();
        let H = Mat::from_fn(n, n, random);
        let H = &H + H.adjoint();

        test_solver_real(&H, &H.lblt(Side::Lower));
        test_solver_real(&H, &H.lblt(Side::Upper));
    }

    #[test]
    fn test_lblt() {
        let n = 7;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let H = Mat::from_fn(n, n, random);
        let H = &H + H.adjoint();

        test_solver(&H, &H.lblt(Side::Lower));
        test_solver(&H, &H.lblt(Side::Upper));
    }

    #[test]
    fn test_cholesky() {
        let n = 7;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let H = Mat::from_fn(n, n, random);
        let H = &H * H.adjoint();

        test_solver(&H, &H.cholesky(Side::Lower).unwrap());
        test_solver(&H, &H.cholesky(Side::Upper).unwrap());
    }

    #[test]
    fn test_partial_piv_lu() {
        let n = 7;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let H = Mat::from_fn(n, n, random);

        test_solver(&H, &H.partial_piv_lu());
    }

    #[test]
    fn test_full_piv_lu() {
        let n = 7;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let H = Mat::from_fn(n, n, random);

        test_solver(&H, &H.full_piv_lu());
    }

    #[test]
    fn test_qr_real() {
        let n = 7;

        let random = |_, _| rand::random::<f64>();
        let H = Mat::from_fn(n, n, random);

        let qr = H.qr();
        test_solver_real(&H, &qr);

        for (m, n) in [(7, 5), (5, 7), (7, 7)] {
            let H = Mat::from_fn(m, n, random);
            let qr = H.qr();
            assert_approx_eq(qr.compute_q() * qr.compute_r(), &H);
            assert_approx_eq(qr.compute_thin_q() * qr.compute_thin_r(), &H);
        }
    }

    #[test]
    fn test_qr() {
        let n = 7;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let H = Mat::from_fn(n, n, random);

        let qr = H.qr();
        test_solver(&H, &qr);

        for (m, n) in [(7, 5), (5, 7), (7, 7)] {
            let H = Mat::from_fn(m, n, random);
            let qr = H.qr();
            assert_approx_eq(qr.compute_q() * qr.compute_r(), &H);
            assert_approx_eq(qr.compute_thin_q() * qr.compute_thin_r(), &H);
            if m >= n {
                test_solver_lstsq(H, &qr)
            }
        }
    }

    #[test]
    fn test_col_piv_qr() {
        let n = 7;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let H = Mat::from_fn(n, n, random);

        test_solver(&H, &H.col_piv_qr());

        for (m, n) in [(7, 5), (5, 7), (7, 7)] {
            let H = Mat::from_fn(m, n, random);
            let qr = H.col_piv_qr();
            assert_approx_eq(
                qr.compute_q() * qr.compute_r(),
                &H * qr.col_permutation().inverse(),
            );
            assert_approx_eq(
                qr.compute_thin_q() * qr.compute_thin_r(),
                &H * qr.col_permutation().inverse(),
            );
            if m >= n {
                test_solver_lstsq(H, &qr)
            }
        }
    }

    #[test]
    fn test_svd() {
        let n = 7;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let H = Mat::from_fn(n, n, random);

        test_solver(&H, &H.svd());
        test_solver(H.adjoint().to_owned(), &H.adjoint().svd());

        let svd = H.svd();
        for i in 0..n - 1 {
            assert!(svd.s_diagonal()[i].re >= svd.s_diagonal()[i + 1].re);
        }
        let svd = H.singular_values();
        for i in 0..n - 1 {
            assert!(svd[i] >= svd[i + 1]);
        }
    }

    #[test]
    fn test_thin_svd() {
        let n = 7;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let H = Mat::from_fn(n, n, random);

        test_solver(&H, &H.thin_svd());
        test_solver(H.adjoint().to_owned(), &H.adjoint().thin_svd());
    }

    #[test]
    fn test_selfadjoint_eigendecomposition() {
        let n = 7;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let H = Mat::from_fn(n, n, random);
        let H = &H * H.adjoint();

        test_solver(&H, &H.selfadjoint_eigendecomposition(Side::Lower));
        test_solver(&H, &H.selfadjoint_eigendecomposition(Side::Upper));
        test_solver(
            H.adjoint().to_owned(),
            &H.adjoint().selfadjoint_eigendecomposition(Side::Lower),
        );
        test_solver(
            H.adjoint().to_owned(),
            &H.adjoint().selfadjoint_eigendecomposition(Side::Upper),
        );

        let evd = H.selfadjoint_eigendecomposition(Side::Lower);
        for i in 0..n - 1 {
            assert!(evd.s().column_vector()[i].re <= evd.s().column_vector()[i + 1].re);
        }
        let evd = H.selfadjoint_eigenvalues(Side::Lower);
        for i in 0..n - 1 {
            assert!(evd[i] <= evd[i + 1]);
        }
    }

    #[test]
    fn test_eigendecomposition() {
        let n = 7;

        let random = |_, _| c64::new(rand::random(), rand::random());
        let H = Mat::from_fn(n, n, random);

        {
            let eigen = H.eigendecomposition::<c64>();
            let s = eigen.s();
            let u = eigen.u();
            assert_approx_eq(u * s, &H * u);
        }

        {
            let eigen = H.complex_eigendecomposition();
            let s = eigen.s();
            let u = eigen.u();
            assert_approx_eq(u * &s, &H * u);
        }

        let det = H.determinant();
        let eigen_det = H
            .complex_eigenvalues()
            .into_iter()
            .fold(c64::faer_one(), |a, b| a * b);

        assert!((det - eigen_det).faer_abs() < 1e-8);
    }

    #[test]
    fn test_real_eigendecomposition() {
        let n = 7;

        let random = |_, _| rand::random::<f64>();
        let H_real = Mat::from_fn(n, n, random);
        let H = Mat::from_fn(n, n, |i, j| c64::new(H_real.read(i, j), 0.0));

        let eigen = H_real.eigendecomposition::<c64>();
        let s = eigen.s();
        let u = eigen.u();
        assert_approx_eq(u * &s, &H * u);
    }

    #[test]
    fn this_other_tree_has_correct_maximum_eigenvalue_20() {
        let edges = [
            (3, 2),
            (6, 1),
            (7, 4),
            (7, 6),
            (8, 5),
            (9, 4),
            (11, 2),
            (12, 2),
            (13, 2),
            (15, 6),
            (16, 2),
            (16, 4),
            (17, 8),
            (18, 0),
            (18, 8),
            (18, 2),
            (19, 6),
            (19, 10),
            (19, 14),
        ];
        let mut a = Mat::zeros(20, 20);
        for (v, u) in edges.iter() {
            a[(*v, *u)] = 1.0;
            a[(*u, *v)] = 1.0;
        }
        let eigs_complex: Vec<c64> = a.eigenvalues();
        println!("{eigs_complex:?}");
        let eigs_real = eigs_complex.iter().map(|e| e.re).collect::<Vec<_>>();
        println!("{eigs_real:?}");
        let lambda_1 = *eigs_real
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let correct_lamba_1 = 2.6148611139728866;
        assert!(
            (lambda_1 - correct_lamba_1).abs() < 1e-10,
            "lambda_1 = {lambda_1}, correct_lamba_1 = {correct_lamba_1}",
        );
    }

    #[test]
    fn this_other_tree_has_correct_maximum_eigenvalue_3() {
        let edges = [(1, 0), (0, 2)];
        let mut a = Mat::zeros(3, 3);
        for (v, u) in edges.iter() {
            a[(*v, *u)] = 1.0;
            a[(*u, *v)] = 1.0;
        }
        let eigs_complex: Vec<c64> = a.eigenvalues();
        let eigs_real = eigs_complex.iter().map(|e| e.re).collect::<Vec<_>>();
        let lambda_1 = *eigs_real
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let correct_lamba_1 = 1.414213562373095;
        assert!(
            (lambda_1 - correct_lamba_1).abs() < 1e-10,
            "lambda_1 = {lambda_1}, correct_lamba_1 = {correct_lamba_1}",
        );
    }
}
