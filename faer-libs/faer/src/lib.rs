//! `faer` is a general-purpose linear algebra library for Rust, with a focus on high performance
//! for algebraic operations on medium/large matrices, as well as matrix decompositions.
//!
//! Most of the high-level functionality in this library is provided through associated functions in
//! its vocabulary types: [`Mat`]/[`MatRef`]/[`MatMut`], as well as the [`FaerMat`] extension trait.
//! The parent crates (`faer-core`, `faer-cholesky`, `faer-lu`, etc.), on the other hand, offer a
//! lower-level of abstraction in exchange for more control over memory allocations and
//! multithreading behavior.
//!
//! `faer` is recommended for applications that handle medium to large dense matrices, and its
//! design is not well suited for applications that operate mostly on low dimensional vectors and
//! matrices such as computer graphics or game development. For those purposes, `nalgebra` and
//! `cgmath` may provide better tools.
//!
//! # Basic usage
//!
//! [`Mat`] is a resizable matrix type with dynamic capacity, which can be created using
//! [`Mat::new`] to produce an empty $0\times 0$ matrix, [`Mat::zeros`] to create a rectangular
//! matrix filled with zeros, [`Mat::identity`] to create an identity matrix, or [`Mat::from_fn`]
//! for the most generic case.
//!
//! Given a `&Mat<E>` (resp. `&mut Mat<E>`), a [`MatRef<'_, E>`](MatRef) (resp. [`MatMut<'_,
//! E>`](MatMut)) can be created by calling [`Mat::as_ref`] (resp. [`Mat::as_mut`]), which allow
//! for more flexibility than `Mat` in that they allow slicing ([`MatRef::get`]) and splitting
//! ([`MatRef::split_at`]).
//!
//! `MatRef` and `MatMut` are lightweight view objects. The former can be copied freely while the
//! latter has move and reborrow semantics, as described in its documentation.
//!
//! More details about the vocabulary types can be found in the `faer-core` crate-level
//! documentation. See also: [`faer_core::Entity`] and [`faer_core::complex_native`].
//!
//! Most of the matrix operations can be used through the corresponding math operators: `+` for
//! matrix addition, `-` for subtraction, `*` for either scalar or matrix multiplication depending
//! on the types of the operands.
//!
//! ## Example
//! ```
//! use faer::{mat, prelude::*, scale, Mat};
//!
//! let a = mat![
//!     [1.0, 5.0, 9.0],
//!     [2.0, 6.0, 10.0],
//!     [3.0, 7.0, 11.0],
//!     [4.0, 8.0, 12.0f64],
//! ];
//!
//! let b = Mat::<f64>::from_fn(4, 3, |i, j| (i + j) as f64);
//!
//! let add = &a + &b;
//! let sub = &a - &b;
//! let scale = scale(3.0) * &a;
//! let mul = &a * b.transpose();
//!
//! let a00 = a[(0, 0)];
//! ```
//!
//! # Matrix decompositions
//! `faer` provides a variety of matrix factorizations, each with its own advantages and drawbacks:
//!
//! ## Cholesky decomposition
//! [`FaerMat::cholesky`] decomposes a self-adjoint positive definite matrix $A$ such that
//! $$A = LL^H,$$
//! where $L$ is a lower triangular matrix. This decomposition is highly efficient and has good
//! stability properties.
//!
//! [An implementation for sparse matrices is also available.](sparse::solvers::Cholesky)
//!
//! ## Bunch-Kaufman decomposition
//! [`FaerMat::lblt`] decomposes a self-adjoint (possibly indefinite) matrix $A$ such that
//! $$P A P^\top = LBL^H,$$
//! where $P$ is a permutation matrix, $L$ is a lower triangular matrix, and $B$ is a block
//! diagonal matrix, with $1 \times 1$ or $2 \times 2$ diagonal blocks.
//! This decomposition is efficient and has good stability properties.
//! ## LU decomposition with partial pivoting
//! [`FaerMat::partial_piv_lu`] decomposes a square invertible matrix $A$ into a lower triangular
//! matrix $L$, a unit upper triangular matrix $U$, and a permutation matrix $P$, such that
//! $$PA = LU.$$
//! It is used by default for computing the determinant, and is generally the recommended method
//! for solving a square linear system or computing the inverse of a matrix (although we generally
//! recommend using a [`Solver`] instead of computing the inverse explicitly).
//!
//! [An implementation for sparse matrices is also available.](sparse::solvers::Lu)
//!
//! ## LU decomposition with full pivoting
//! [`FaerMat::full_piv_lu`] Decomposes a generic rectangular matrix $A$ into a lower triangular
//! matrix $L$, a unit upper triangular matrix $U$, and permutation matrices $P$ and $Q$, such that
//! $$PAQ^\top = LU.$$
//! It can be more stable than the LU decomposition with partial pivoting, in exchange for being
//! more computationally expensive.
//!
//! ## QR decomposition
//! The QR decomposition ([`FaerMat::qr`]) decomposes a matrix $A$ into the product
//! $$A = QR,$$
//! where $Q$ is a unitary matrix, and $R$ is an upper trapezoidal matrix. It is often used for
//! solving least squares problems.
//!
//! [An implementation for sparse matrices is also available.](sparse::solvers::Qr)
//!
//! ## QR decomposition with column pivoting
//! The QR decomposition with column pivoting ([`FaerMat::col_piv_qr`]) decomposes a matrix $A$ into
//! the product
//! $$AP^T = QR,$$
//! where $P$ is a permutation matrix, $Q$ is a unitary matrix, and $R$ is an upper trapezoidal
//! matrix.
//!
//! It is slower than the version with no pivoting, in exchange for being more numerically stable
//! for rank-deficient matrices.
//!
//! ## Singular value decomposition
//! The SVD of a matrix $M$ of shape $(m, n)$ is a decomposition into three components $U$, $S$,
//! and $V$, such that:
//!
//! - $U$ has shape $(m, m)$ and is a unitary matrix,
//! - $V$ has shape $(n, n)$ and is a unitary matrix,
//! - $S$ has shape $(m, n)$ and is zero everywhere except the main diagonal, with nonnegative
//! diagonal values in nonincreasing order,
//! - and finally:
//!
//! $$M = U S V^H.$$
//!
//! The SVD is provided in two forms: either the full matrices $U$ and $V$ are computed, using
//! [`FaerMat::svd`], or only their first $\min(m, n)$ columns are computed, using
//! [`FaerMat::thin_svd`].
//!
//! If only the singular values (elements of $S$) are desired, they can be obtained in
//! nonincreasing order using [`FaerMat::singular_values`].
//!
//! ## Eigendecomposition
//! **Note**: The order of the eigenvalues is currently unspecified and may be changed in a future
//! release.
//!
//! The eigendecomposition of a square matrix $M$ of shape $(n, n)$ is a decomposition into
//! two components $U$, $S$:
//!
//! - $U$ has shape $(n, n)$ and is invertible,
//! - $S$ has shape $(n, n)$ and is a diagonal matrix,
//! - and finally:
//!
//! $$M = U S U^{-1}.$$
//!
//! If $M$ is hermitian, then $U$ can be made unitary ($U^{-1} = U^H$), and $S$ is real valued.
//!
//! Depending on the domain of the input matrix and whether it is self-adjoint, multiple methods
//! are provided to compute the eigendecomposition:
//! * [`FaerMat::selfadjoint_eigendecomposition`] can be used with either real or complex matrices,
//! producing an eigendecomposition of the same type.
//! * [`FaerMat::eigendecomposition`] can be used with either real or complex matrices, but the
//!   output
//! complex type has to be specified.
//! * [`FaerMat::complex_eigendecomposition`] can only be used with complex matrices, with the
//!   output
//! having the same type.
//!
//! If only the eigenvalues (elements of $S$) are desired, they can be obtained in
//! nonincreasing order using [`FaerMat::selfadjoint_eigenvalues`], [`FaerMat::eigenvalues`], or
//! [`FaerMat::complex_eigenvalues`], with the same conditions described above.
//!
//! # Crate features
//!
//! - `std`: enabled by default. Links with the standard library to enable additional features such
//!   as cpu feature detection at runtime.
//! - `rayon`: enabled by default. Enables the `rayon` parallel backend and enables global
//!   parallelism by default.
//! - `matrixcompare`: enabled by default. Enables macros for approximate equality checks on
//!   matrices.
//! - `perf-warn`: Produces performance warnings when matrix operations are called with suboptimal
//! data layout.
//! - `polars`: Enables basic interoperability with the `polars` crate.
//! - `nalgebra`: Enables basic interoperability with the `nalgebra` crate.
//! - `ndarray`: Enables basic interoperability with the `ndarray` crate.
//! - `nightly`: Requires the nightly compiler. Enables experimental SIMD features such as AVX512.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

use dyn_stack::{GlobalPodBuffer, PodStack};
use faer_core::{AsMatMut, AsMatRef, ComplexField, Conj, Conjugate, Entity};
use prelude::*;
use solvers::*;

/// Similar to the [`dbg`] macro, but takes a format spec as a first parameter.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub use dbgf::dbgf;
pub use faer_cholesky::llt::CholeskyError;

/// Commonly used traits for a streamlined user experience.
pub mod prelude {
    pub use crate::{
        solvers::{Solver, SolverCore, SolverLstsq, SolverLstsqCore},
        sparse::solvers::{SpSolver, SpSolverCore, SpSolverLstsq, SpSolverLstsqCore},
        FaerMat, IntoFaer, IntoFaerComplex,
    };
    pub use reborrow::{IntoConst, Reborrow, ReborrowMut};
}

#[cfg(test)]
mod proptest_support;

pub use faer_core::{
    complex_native, get_global_parallelism, mat, scale, set_global_parallelism, unzipped, zipped,
    Col, ColMut, ColRef, Mat, MatMut, MatRef, Parallelism, Row, RowMut, RowRef, Side,
};
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub use matrixcompare::assert_matrix_eq;

extern crate alloc;
use alloc::{vec, vec::Vec};

/// Matrix solvers and decompositions.
pub mod solvers {
    use super::*;
    use faer_core::{assert, permutation::PermutationRef, zipped};
    use sparse::solvers::{SpSolverCore, SpSolverLstsqCore};

    pub trait SolverCore<E: Entity>: SpSolverCore<E> {
        /// Reconstructs the original matrix using the decomposition.
        fn reconstruct(&self) -> Mat<E>;
        /// Computes the inverse of the original matrix using the decomposition.
        ///
        /// # Panics
        /// Panics if the matrix is not square.
        fn inverse(&self) -> Mat<E>;
    }
    pub trait SolverLstsqCore<E: Entity>: SolverCore<E> + SpSolverLstsqCore<E> {}

    pub trait Solver<E: ComplexField>: SolverCore<E> + SpSolver<E> {}
    pub trait SolverLstsq<E: Entity>: SolverLstsqCore<E> + SpSolverLstsq<E> {}

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
                        faer_core::zip::Diag::Include,
                        |unzipped!(mut dst, src)| dst.write(src.read().canonicalize()),
                    );
                }
                Side::Upper => {
                    zipped!(factors.as_mut(), matrix.adjoint()).for_each_triangular_lower(
                        faer_core::zip::Diag::Include,
                        |unzipped!(mut dst, src)| dst.write(src.read().canonicalize()),
                    );
                }
            }

            let params = Default::default();

            faer_cholesky::llt::compute::cholesky_in_place(
                factors.as_mut(),
                Default::default(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_cholesky::llt::compute::cholesky_in_place_req::<E>(
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

        pub fn compute_l(&self) -> Mat<E> {
            let mut factor = self.factors.to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_upper(faer_core::zip::Diag::Skip, |unzipped!(mut dst)| {
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

            faer_cholesky::llt::solve::solve_in_place_with_conj(
                self.factors.as_ref(),
                conj,
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_cholesky::llt::solve::solve_in_place_req::<E>(
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

            faer_cholesky::llt::inverse::invert_lower(
                inv.as_mut(),
                self.factors.as_ref(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_cholesky::llt::inverse::invert_lower_req::<E>(self.dim(), parallelism)
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

            faer_cholesky::llt::reconstruct::reconstruct_lower(
                rec.as_mut(),
                self.factors.as_ref(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_cholesky::llt::reconstruct::reconstruct_lower_req::<E>(self.dim())
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
                        faer_core::zip::Diag::Include,
                        |unzipped!(mut dst, src)| dst.write(src.read().canonicalize()),
                    );
                }
                Side::Upper => {
                    zipped!(factors.as_mut(), matrix.adjoint()).for_each_triangular_lower(
                        faer_core::zip::Diag::Include,
                        |unzipped!(mut dst, src)| dst.write(src.read().canonicalize()),
                    );
                }
            }

            let params = Default::default();

            faer_cholesky::bunch_kaufman::compute::cholesky_in_place(
                factors.as_mut(),
                subdiag.as_mut(),
                Default::default(),
                &mut perm,
                &mut perm_inv,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_cholesky::bunch_kaufman::compute::cholesky_in_place_req::<usize, E>(
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

            faer_cholesky::bunch_kaufman::solve::solve_in_place_with_conj(
                self.factors.as_ref(),
                self.subdiag.as_ref(),
                conj,
                unsafe { PermutationRef::new_unchecked(&self.perm, &self.perm_inv) },
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_cholesky::bunch_kaufman::solve::solve_in_place_req::<usize, E>(
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
                faer_core::zip::Diag::Skip,
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
            faer_core::mul::triangular::matmul(
                mat2.as_mut(),
                faer_core::mul::triangular::BlockStructure::TriangularLower,
                lbl,
                faer_core::mul::triangular::BlockStructure::UnitTriangularLower,
                mat.as_ref().adjoint(),
                faer_core::mul::triangular::BlockStructure::Rectangular,
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
        #[track_caller]
        pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
            assert!(matrix.nrows() == matrix.ncols());

            let dim = matrix.nrows();
            let parallelism = get_global_parallelism();

            let mut factors = matrix.to_owned();

            let params = Default::default();

            let mut row_perm = vec![0usize; dim];
            let mut row_perm_inv = vec![0usize; dim];

            let (n_transpositions, _) = faer_lu::partial_pivoting::compute::lu_in_place(
                factors.as_mut(),
                &mut row_perm,
                &mut row_perm_inv,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_lu::partial_pivoting::compute::lu_in_place_req::<usize, E>(
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

        pub fn row_permutation(&self) -> PermutationRef<'_, usize, E> {
            unsafe { PermutationRef::new_unchecked(&self.row_perm, &self.row_perm_inv) }
        }

        pub fn transposition_count(&self) -> usize {
            self.n_transpositions
        }

        pub fn compute_l(&self) -> Mat<E> {
            let mut factor = self.factors.to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_upper(faer_core::zip::Diag::Skip, |unzipped!(mut dst)| {
                    dst.write(E::faer_zero())
                });
            factor
        }
        pub fn compute_u(&self) -> Mat<E> {
            let mut factor = self.factors.to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_lower(faer_core::zip::Diag::Skip, |unzipped!(mut dst)| {
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

            faer_lu::partial_pivoting::solve::solve_in_place(
                self.factors.as_ref(),
                conj,
                self.row_permutation(),
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_lu::partial_pivoting::solve::solve_in_place_req::<usize, E>(
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

            faer_lu::partial_pivoting::solve::solve_transpose_in_place(
                self.factors.as_ref(),
                conj,
                self.row_permutation(),
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_lu::partial_pivoting::solve::solve_transpose_in_place_req::<usize, E>(
                        self.dim(),
                        self.dim(),
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
    impl<E: ComplexField> SolverCore<E> for PartialPivLu<E> {
        fn inverse(&self) -> Mat<E> {
            let mut inv = Mat::<E>::zeros(self.dim(), self.dim());
            let parallelism = get_global_parallelism();

            faer_lu::partial_pivoting::inverse::invert(
                inv.as_mut(),
                self.factors.as_ref(),
                self.row_permutation(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_lu::partial_pivoting::inverse::invert_req::<usize, E>(
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

            faer_lu::partial_pivoting::reconstruct::reconstruct(
                rec.as_mut(),
                self.factors.as_ref(),
                self.row_permutation(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_lu::partial_pivoting::reconstruct::reconstruct_req::<usize, E>(
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

            let (n_transpositions, _, _) = faer_lu::full_pivoting::compute::lu_in_place(
                factors.as_mut(),
                &mut row_perm,
                &mut row_perm_inv,
                &mut col_perm,
                &mut col_perm_inv,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_lu::full_pivoting::compute::lu_in_place_req::<usize, E>(
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

        pub fn row_permutation(&self) -> PermutationRef<'_, usize, E> {
            unsafe { PermutationRef::new_unchecked(&self.row_perm, &self.row_perm_inv) }
        }
        pub fn col_permutation(&self) -> PermutationRef<'_, usize, E> {
            unsafe { PermutationRef::new_unchecked(&self.col_perm, &self.col_perm_inv) }
        }

        pub fn transposition_count(&self) -> usize {
            self.n_transpositions
        }

        pub fn compute_l(&self) -> Mat<E> {
            let size = Ord::min(self.nrows(), self.ncols());
            let mut factor = self
                .factors
                .as_ref()
                .submatrix(0, 0, self.nrows(), size)
                .to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_upper(faer_core::zip::Diag::Skip, |unzipped!(mut dst)| {
                    dst.write(E::faer_zero())
                });
            factor
        }
        pub fn compute_u(&self) -> Mat<E> {
            let size = Ord::min(self.nrows(), self.ncols());
            let mut factor = self
                .factors
                .as_ref()
                .submatrix(0, 0, size, self.ncols())
                .to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_lower(faer_core::zip::Diag::Skip, |unzipped!(mut dst)| {
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

            faer_lu::full_pivoting::solve::solve_in_place(
                self.factors.as_ref(),
                conj,
                self.row_permutation(),
                self.col_permutation(),
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_lu::full_pivoting::solve::solve_in_place_req::<usize, E>(
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

            faer_lu::full_pivoting::solve::solve_transpose_in_place(
                self.factors.as_ref(),
                conj,
                self.row_permutation(),
                self.col_permutation(),
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_lu::full_pivoting::solve::solve_transpose_in_place_req::<usize, E>(
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

            faer_lu::full_pivoting::inverse::invert(
                inv.as_mut(),
                self.factors.as_ref(),
                self.row_permutation(),
                self.col_permutation(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_lu::full_pivoting::inverse::invert_req::<usize, E>(dim, dim, parallelism)
                        .unwrap(),
                )),
            );

            inv
        }

        fn reconstruct(&self) -> Mat<E> {
            let mut rec = Mat::<E>::zeros(self.nrows(), self.ncols());
            let parallelism = get_global_parallelism();

            faer_lu::full_pivoting::reconstruct::reconstruct(
                rec.as_mut(),
                self.factors.as_ref(),
                self.row_permutation(),
                self.col_permutation(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_lu::full_pivoting::reconstruct::reconstruct_req::<usize, E>(
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
        #[track_caller]
        pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
            let parallelism = get_global_parallelism();
            let nrows = matrix.nrows();
            let ncols = matrix.ncols();

            let mut factors = matrix.to_owned();
            let size = Ord::min(nrows, ncols);
            let blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<E>(nrows, ncols);
            let mut householder = Mat::<E>::zeros(blocksize, size);

            let params = Default::default();

            faer_qr::no_pivoting::compute::qr_in_place(
                factors.as_mut(),
                householder.as_mut(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_qr::no_pivoting::compute::qr_in_place_req::<E>(
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

        pub fn compute_r(&self) -> Mat<E> {
            let mut factor = self.factors.to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_lower(faer_core::zip::Diag::Skip, |unzipped!(mut dst)| {
                    dst.write(E::faer_zero())
                });
            factor
        }

        pub fn compute_q(&self) -> Mat<E> {
            Self::__compute_q_impl(self.factors.as_ref(), self.householder.as_ref(), false)
        }

        pub fn compute_thin_r(&self) -> Mat<E> {
            let m = self.nrows();
            let n = self.ncols();
            let mut factor = self.factors.as_ref().subrows(0, Ord::min(m, n)).to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_lower(faer_core::zip::Diag::Skip, |unzipped!(mut dst)| {
                    dst.write(E::faer_zero())
                });
            factor
        }

        pub fn compute_thin_q(&self) -> Mat<E> {
            Self::__compute_q_impl(self.factors.as_ref(), self.householder.as_ref(), true)
        }

        fn __compute_q_impl(
            factors: MatRef<'_, E>,
            householder: MatRef<'_, E>,
            thin: bool,
        ) -> Mat<E> {
            let parallelism = get_global_parallelism();
            let m = factors.nrows();
            let size = Ord::min(m, factors.ncols());

            let mut q = Mat::<E>::zeros(m, if thin { size } else { m });
            q.as_mut()
                .diagonal_mut()
                .column_vector_mut()
                .fill(E::faer_one());

            faer_core::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                factors,
                householder,
                Conj::No,
                q.as_mut(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_core::householder::apply_block_householder_sequence_on_the_left_in_place_req::<E>(
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

            faer_qr::no_pivoting::solve::solve_transpose_in_place(
                self.factors.as_ref(),
                self.householder.as_ref(),
                conj,
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_qr::no_pivoting::solve::solve_transpose_in_place_req::<E>(
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

            faer_qr::no_pivoting::reconstruct::reconstruct(
                rec.as_mut(),
                self.factors.as_ref(),
                self.householder.as_ref(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_qr::no_pivoting::reconstruct::reconstruct_req::<E>(
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

            faer_qr::no_pivoting::inverse::invert(
                inv.as_mut(),
                self.factors.as_ref(),
                self.householder.as_ref(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_qr::no_pivoting::inverse::invert_req::<E>(
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

            faer_qr::no_pivoting::solve::solve_in_place(
                self.factors.as_ref(),
                self.householder.as_ref(),
                conj,
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_qr::no_pivoting::solve::solve_in_place_req::<E>(
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
        #[track_caller]
        pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
            let parallelism = get_global_parallelism();
            let nrows = matrix.nrows();
            let ncols = matrix.ncols();

            let mut factors = matrix.to_owned();
            let size = Ord::min(nrows, ncols);
            let blocksize =
                faer_qr::col_pivoting::compute::recommended_blocksize::<E>(nrows, ncols);
            let mut householder = Mat::<E>::zeros(blocksize, size);

            let params = Default::default();

            let mut col_perm = vec![0usize; ncols];
            let mut col_perm_inv = vec![0usize; ncols];

            faer_qr::col_pivoting::compute::qr_in_place(
                factors.as_mut(),
                householder.as_mut(),
                &mut col_perm,
                &mut col_perm_inv,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_qr::col_pivoting::compute::qr_in_place_req::<usize, E>(
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

        pub fn col_permutation(&self) -> PermutationRef<'_, usize, E> {
            unsafe { PermutationRef::new_unchecked(&self.col_perm, &self.col_perm_inv) }
        }

        fn blocksize(&self) -> usize {
            self.householder.nrows()
        }

        pub fn compute_r(&self) -> Mat<E> {
            let mut factor = self.factors.to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_lower(faer_core::zip::Diag::Skip, |unzipped!(mut dst)| {
                    dst.write(E::faer_zero())
                });
            factor
        }

        pub fn compute_q(&self) -> Mat<E> {
            Qr::<E>::__compute_q_impl(self.factors.as_ref(), self.householder.as_ref(), false)
        }

        pub fn compute_thin_r(&self) -> Mat<E> {
            let m = self.nrows();
            let n = self.ncols();
            let mut factor = self.factors.as_ref().subrows(0, Ord::min(m, n)).to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_lower(faer_core::zip::Diag::Skip, |unzipped!(mut dst)| {
                    dst.write(E::faer_zero())
                });
            factor
        }

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

            faer_qr::col_pivoting::solve::solve_transpose_in_place(
                self.factors.as_ref(),
                self.householder.as_ref(),
                self.col_permutation(),
                conj,
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_qr::col_pivoting::solve::solve_transpose_in_place_req::<usize, E>(
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

            faer_qr::col_pivoting::reconstruct::reconstruct(
                rec.as_mut(),
                self.factors.as_ref(),
                self.householder.as_ref(),
                self.col_permutation(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_qr::col_pivoting::reconstruct::reconstruct_req::<usize, E>(
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

            faer_qr::col_pivoting::inverse::invert(
                inv.as_mut(),
                self.factors.as_ref(),
                self.householder.as_ref(),
                self.col_permutation(),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_qr::col_pivoting::inverse::invert_req::<usize, E>(
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

            faer_qr::col_pivoting::solve::solve_in_place(
                self.factors.as_ref(),
                self.householder.as_ref(),
                self.col_permutation(),
                conj,
                rhs,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_qr::col_pivoting::solve::solve_in_place_req::<usize, E>(
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
                faer_svd::ComputeVectors::Thin
            } else {
                faer_svd::ComputeVectors::Full
            };

            faer_svd::compute_svd(
                matrix,
                s.as_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_svd::compute_svd_req::<E>(
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

        #[track_caller]
        pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
            Self::__new_impl(matrix.canonicalize(), false)
        }

        pub fn u(&self) -> MatRef<'_, E> {
            self.u.as_ref()
        }
        pub fn s_diagonal(&self) -> MatRef<'_, E> {
            self.s.as_ref()
        }
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
        #[track_caller]
        pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>) -> Self {
            Self {
                inner: Svd::__new_impl(matrix.canonicalize(), true),
            }
        }

        pub fn u(&self) -> MatRef<'_, E> {
            self.inner.u.as_ref()
        }
        pub fn s_diagonal(&self) -> MatRef<'_, E> {
            self.inner.s.as_ref()
        }
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
            faer_evd::compute_hermitian_evd(
                matrix,
                s.as_mut(),
                Some(u.as_mut()),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_evd::compute_hermitian_evd_req::<E>(
                        dim,
                        faer_evd::ComputeVectors::Yes,
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

        #[track_caller]
        pub fn new<ViewE: Conjugate<Canonical = E>>(matrix: MatRef<'_, ViewE>, side: Side) -> Self {
            Self::__new_impl(matrix.canonicalize(), side)
        }

        pub fn u(&self) -> MatRef<'_, E> {
            self.u.as_ref()
        }
        pub fn s_diagonal(&self) -> MatRef<'_, E> {
            self.s.as_ref()
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

            faer_evd::compute_evd_real(
                matrix,
                s_re.as_mut(),
                s_im.as_mut(),
                None,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_evd::compute_evd_req::<E::Real>(
                        dim,
                        faer_evd::ComputeVectors::Yes,
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

            faer_evd::compute_evd_complex(
                matrix,
                s.as_mut(),
                None,
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_evd::compute_evd_req::<E>(
                        dim,
                        faer_evd::ComputeVectors::Yes,
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

            faer_evd::compute_evd_real(
                matrix,
                s_re.as_mut().as_2d_mut(),
                s_im.as_mut().as_2d_mut(),
                Some(u_real.as_mut()),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_evd::compute_evd_req::<E::Real>(
                        dim,
                        faer_evd::ComputeVectors::Yes,
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
                    zipped!(u.as_mut().col_mut(j).as_2d_mut(), u_real.col(j).as_2d()).for_each(
                        |unzipped!(mut dst, src)| dst.write(E::faer_from_real(src.read())),
                    );
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

            faer_evd::compute_evd_complex(
                matrix,
                s.as_mut().as_2d_mut(),
                Some(u.as_mut()),
                parallelism,
                PodStack::new(&mut GlobalPodBuffer::new(
                    faer_evd::compute_evd_req::<E>(
                        dim,
                        faer_evd::ComputeVectors::Yes,
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

        #[track_caller]
        pub fn new_from_complex<ViewE: Conjugate<Canonical = E>>(
            matrix: MatRef<'_, ViewE>,
        ) -> Self {
            Self::__new_from_complex_impl(matrix.canonicalize())
        }

        pub fn u(&self) -> MatRef<'_, E> {
            self.u.as_ref()
        }
        pub fn s_diagonal(&self) -> ColRef<'_, E> {
            self.s.as_ref()
        }
    }
}

/// Extension trait for `faer` types.
pub trait FaerMat<E: ComplexField> {
    /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    fn solve_lower_triangular_in_place(&self, rhs: impl AsMatMut<E>);
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// stores the result in `rhs`.
    fn solve_upper_triangular_in_place(&self, rhs: impl AsMatMut<E>);
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    fn solve_unit_lower_triangular_in_place(&self, rhs: impl AsMatMut<E>);
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
    /// and stores the result in `rhs`.
    fn solve_unit_upper_triangular_in_place(&self, rhs: impl AsMatMut<E>);

    /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    #[track_caller]
    fn solve_lower_triangular<ViewE: Conjugate<Canonical = E>>(
        &self,
        rhs: impl AsMatRef<ViewE>,
    ) -> Mat<E> {
        let mut rhs = rhs.as_mat_ref().to_owned();
        self.solve_lower_triangular_in_place(rhs.as_mut());
        rhs
    }
    /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    #[track_caller]
    fn solve_upper_triangular<ViewE: Conjugate<Canonical = E>>(
        &self,
        rhs: impl AsMatRef<ViewE>,
    ) -> Mat<E> {
        let mut rhs = rhs.as_mat_ref().to_owned();
        self.solve_upper_triangular_in_place(rhs.as_mut());
        rhs
    }
    /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    #[track_caller]
    fn solve_unit_lower_triangular<ViewE: Conjugate<Canonical = E>>(
        &self,
        rhs: impl AsMatRef<ViewE>,
    ) -> Mat<E> {
        let mut rhs = rhs.as_mat_ref().to_owned();
        self.solve_unit_lower_triangular_in_place(rhs.as_mut());
        rhs
    }
    /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`, and
    /// returns the result.
    #[track_caller]
    fn solve_unit_upper_triangular<ViewE: Conjugate<Canonical = E>>(
        &self,
        rhs: impl AsMatRef<ViewE>,
    ) -> Mat<E> {
        let mut rhs = rhs.as_mat_ref().to_owned();
        self.solve_unit_upper_triangular_in_place(rhs.as_mut());
        rhs
    }

    /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
    fn cholesky(&self, side: Side) -> Result<Cholesky<E>, CholeskyError>;
    /// Returns the Bunch-Kaufman decomposition of `self`. Only the provided side is accessed.
    fn lblt(&self, side: Side) -> Lblt<E>;
    /// Returns the LU decomposition of `self` with partial (row) pivoting.
    fn partial_piv_lu(&self) -> PartialPivLu<E>;
    /// Returns the LU decomposition of `self` with full pivoting.
    fn full_piv_lu(&self) -> FullPivLu<E>;
    /// Returns the QR decomposition of `self`.
    fn qr(&self) -> Qr<E>;
    /// Returns the QR decomposition of `self`, with column pivoting.
    fn col_piv_qr(&self) -> ColPivQr<E>;
    /// Returns the SVD of `self`.
    fn svd(&self) -> Svd<E>;
    /// Returns the thin SVD of `self`.
    fn thin_svd(&self) -> ThinSvd<E>;
    /// Returns the eigendecomposition of `self`, assuming it is self-adjoint. Only the provided
    /// side is accessed.
    fn selfadjoint_eigendecomposition(&self, side: Side) -> SelfAdjointEigendecomposition<E>;
    /// Returns the eigendecomposition of `self`, as a complex matrix.
    fn eigendecomposition<ComplexE: ComplexField<Real = E::Real>>(
        &self,
    ) -> Eigendecomposition<ComplexE>;
    /// Returns the eigendecomposition of `self`, when `E` is in the complex domain.
    fn complex_eigendecomposition(&self) -> Eigendecomposition<E>;

    /// Returns the determinant of `self`.
    fn determinant(&self) -> E;
    /// Returns the singular values of `self`, in nonincreasing order.
    fn singular_values(&self) -> Vec<E::Real>;
    /// Returns the eigenvalues of `self`, assuming it is self-adjoint. Only the provided
    /// side is accessed. The order of the eigenvalues is currently unspecified.
    fn selfadjoint_eigenvalues(&self, side: Side) -> Vec<E::Real>;
    /// Returns the eigenvalues of `self`, as complex values. The order of the eigenvalues is
    /// currently unspecified.
    fn eigenvalues<ComplexE: ComplexField<Real = E::Real>>(&self) -> Vec<ComplexE>;
    /// Returns the eigenvalues of `self`, when `E` is in the complex domain. The order of the
    /// eigenvalues is currently unspecified.
    fn complex_eigenvalues(&self) -> Vec<E::Canonical>;
}

/// Sparse solvers and traits.
pub mod sparse {
    use super::*;
    use faer_core::group_helpers::VecGroup;

    pub use faer_core::{
        permutation::Index,
        sparse::{
            SparseColMatRef, SparseRowMatRef, SymbolicSparseColMatRef, SymbolicSparseRowMatRef,
        },
    };
    pub use faer_sparse::{lu::LuError, FaerError};

    /// Sparse Cholesky error.
    #[derive(Copy, Clone, Debug)]
    pub enum CholeskyError {
        Generic(FaerError),
        SymbolicSingular,
        NotPositiveDefinite,
    }

    impl core::fmt::Display for CholeskyError {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            core::fmt::Debug::fmt(self, f)
        }
    }

    #[cfg(feature = "std")]
    impl std::error::Error for CholeskyError {}

    impl From<FaerError> for CholeskyError {
        #[inline]
        fn from(value: FaerError) -> Self {
            Self::Generic(value)
        }
    }

    impl From<crate::CholeskyError> for CholeskyError {
        #[inline]
        fn from(_: crate::CholeskyError) -> Self {
            Self::NotPositiveDefinite
        }
    }

    /// Sparse solvers.
    ///
    /// Each solver satisfies the [`SpSolver`] and/or [`SpSolverLstsq`] traits, which can be used
    /// to solve linear systems.
    pub mod solvers {
        use super::*;

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
            fn solve_conj<ViewE: Conjugate<Canonical = E>>(
                &self,
                rhs: impl AsMatRef<ViewE>,
            ) -> Mat<E>;
            /// Solves the equation `transpose(self) * X = rhs`, and returns the result.
            fn solve_transpose<ViewE: Conjugate<Canonical = E>>(
                &self,
                rhs: impl AsMatRef<ViewE>,
            ) -> Mat<E>;
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
            fn solve_lstsq<ViewE: Conjugate<Canonical = E>>(
                &self,
                rhs: impl AsMatRef<ViewE>,
            ) -> Mat<E>;
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
            fn solve_conj<ViewE: Conjugate<Canonical = E>>(
                &self,
                rhs: impl AsMatRef<ViewE>,
            ) -> Mat<E> {
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
            fn solve_lstsq<ViewE: Conjugate<Canonical = E>>(
                &self,
                rhs: impl AsMatRef<ViewE>,
            ) -> Mat<E> {
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
        pub struct SymbolicCholesky<I> {
            inner: alloc::sync::Arc<faer_sparse::cholesky::SymbolicCholesky<I>>,
        }
        #[derive(Clone, Debug)]
        pub struct Cholesky<I, E: Entity> {
            symbolic: SymbolicCholesky<I>,
            values: VecGroup<E>,
        }

        #[derive(Debug)]
        pub struct SymbolicQr<I> {
            inner: alloc::sync::Arc<faer_sparse::qr::SymbolicQr<I>>,
        }
        #[derive(Clone, Debug)]
        pub struct Qr<I, E: Entity> {
            symbolic: SymbolicQr<I>,
            indices: alloc::vec::Vec<I>,
            values: VecGroup<E>,
        }

        #[derive(Debug)]
        pub struct SymbolicLu<I> {
            inner: alloc::sync::Arc<faer_sparse::lu::SymbolicLu<I>>,
        }
        #[derive(Clone, Debug)]
        pub struct Lu<I, E: Entity> {
            symbolic: SymbolicLu<I>,
            numeric: faer_sparse::lu::NumericLu<I, E>,
        }

        impl<I> Clone for SymbolicCholesky<I> {
            #[inline]
            fn clone(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }
        impl<I> Clone for SymbolicQr<I> {
            #[inline]
            fn clone(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }
        impl<I> Clone for SymbolicLu<I> {
            #[inline]
            fn clone(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }

        impl<I: Index> SymbolicCholesky<I> {
            #[track_caller]
            pub fn try_new(
                mat: SymbolicSparseColMatRef<'_, I>,
                side: Side,
            ) -> Result<Self, FaerError> {
                Ok(Self {
                    inner: alloc::sync::Arc::new(
                        faer_sparse::cholesky::factorize_symbolic_cholesky(
                            mat,
                            side,
                            Default::default(),
                        )?,
                    ),
                })
            }
        }
        impl<I: Index> SymbolicQr<I> {
            #[track_caller]
            pub fn try_new(mat: SymbolicSparseColMatRef<'_, I>) -> Result<Self, FaerError> {
                Ok(Self {
                    inner: alloc::sync::Arc::new(faer_sparse::qr::factorize_symbolic_qr(
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
                    inner: alloc::sync::Arc::new(faer_sparse::lu::factorize_symbolic_lu(
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
            ) -> Result<Self, sparse::CholeskyError> {
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
            ) -> Result<Self, LuError> {
                let mut numeric = faer_sparse::lu::NumericLu::new();
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
                faer_sparse::cholesky::LltRef::<'_, I, E>::new(
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
                faer_sparse::cholesky::LltRef::<'_, I, E>::new(
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
                    faer_sparse::qr::QrRef::<'_, I, E>::new_unchecked(
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
                unsafe {
                    faer_sparse::lu::LuRef::<'_, I, E>::new_unchecked(
                        &self.symbolic.inner,
                        &self.numeric,
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

            #[track_caller]
            fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
                let parallelism = get_global_parallelism();
                let rhs_ncols = rhs.ncols();
                unsafe {
                    faer_sparse::lu::LuRef::<'_, I, E>::new_unchecked(
                        &self.symbolic.inner,
                        &self.numeric,
                    )
                }
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
    }

    /// Extension trait for sparse `faer` types.
    pub trait FaerSparseMat<I: Index, E: ComplexField> {
        /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
        /// stores the result in `rhs`.
        ///
        /// The diagonal element is assumed to be the first stored element in each column, if the
        /// matrix is column-major, or the last stored element in each row, if it is row-major.
        fn sp_solve_lower_triangular_in_place(&self, rhs: impl AsMatMut<E>);
        /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
        /// stores the result in `rhs`.
        ///
        /// The diagonal element is assumed to be the last stored element in each column, if the
        /// matrix is column-major, or the first stored element in each row, if it is row-major.
        fn sp_solve_upper_triangular_in_place(&self, rhs: impl AsMatMut<E>);
        /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
        /// and stores the result in `rhs`.
        ///
        /// The diagonal element is assumed to be the first stored element in each column, if the
        /// matrix is column-major, or the last stored element in each row, if it is row-major.
        fn sp_solve_unit_lower_triangular_in_place(&self, rhs: impl AsMatMut<E>);
        /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
        /// and stores the result in `rhs`.
        ///
        /// The diagonal element is assumed to be the last stored element in each column, if the
        /// matrix is column-major, or the first stored element in each row, if it is row-major.
        fn sp_solve_unit_upper_triangular_in_place(&self, rhs: impl AsMatMut<E>);

        /// Assuming `self` is a lower triangular matrix, solves the equation `self * X = rhs`, and
        /// returns the result.
        ///
        /// The diagonal element is assumed to be the first stored element in each column, if the
        /// matrix is column-major, or the last stored element in each row, if it is row-major.
        #[track_caller]
        fn sp_solve_lower_triangular<ViewE: Conjugate<Canonical = E>>(
            &self,
            rhs: impl AsMatRef<ViewE>,
        ) -> Mat<E> {
            let mut rhs = rhs.as_mat_ref().to_owned();
            self.sp_solve_lower_triangular_in_place(rhs.as_mut());
            rhs
        }
        /// Assuming `self` is an upper triangular matrix, solves the equation `self * X = rhs`, and
        /// returns the result.
        ///
        /// The diagonal element is assumed to be the last stored element in each column, if the
        /// matrix is column-major, or the first stored element in each row, if it is row-major.
        #[track_caller]
        fn sp_solve_upper_triangular<ViewE: Conjugate<Canonical = E>>(
            &self,
            rhs: impl AsMatRef<ViewE>,
        ) -> Mat<E> {
            let mut rhs = rhs.as_mat_ref().to_owned();
            self.sp_solve_upper_triangular_in_place(rhs.as_mut());
            rhs
        }
        /// Assuming `self` is a unit lower triangular matrix, solves the equation `self * X = rhs`,
        /// and returns the result.
        ///
        /// The diagonal element is assumed to be the first stored element in each column, if the
        /// matrix is column-major, or the last stored element in each row, if it is row-major.
        #[track_caller]
        fn sp_solve_unit_lower_triangular<ViewE: Conjugate<Canonical = E>>(
            &self,
            rhs: impl AsMatRef<ViewE>,
        ) -> Mat<E> {
            let mut rhs = rhs.as_mat_ref().to_owned();
            self.sp_solve_unit_lower_triangular_in_place(rhs.as_mut());
            rhs
        }
        /// Assuming `self` is a unit upper triangular matrix, solves the equation `self * X = rhs`,
        /// and returns the result.
        ///
        /// The diagonal element is assumed to be the first stored element in each column, if the
        /// matrix is column-major, or the last stored element in each row, if it is row-major.
        #[track_caller]
        fn sp_solve_unit_upper_triangular<ViewE: Conjugate<Canonical = E>>(
            &self,
            rhs: impl AsMatRef<ViewE>,
        ) -> Mat<E> {
            let mut rhs = rhs.as_mat_ref().to_owned();
            self.sp_solve_unit_upper_triangular_in_place(rhs.as_mut());
            rhs
        }

        /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
        fn sp_cholesky(&self, side: Side)
            -> Result<solvers::Cholesky<I, E>, sparse::CholeskyError>;

        /// Returns the LU decomposition of `self` with partial (row) pivoting.
        fn sp_lu(&self) -> Result<solvers::Lu<I, E>, LuError>;

        /// Returns the QR decomposition of `self`.
        fn sp_qr(&self) -> Result<solvers::Qr<I, E>, FaerError>;
    }

    impl<I: Index, E: ComplexField> FaerSparseMat<I, E> for SparseColMatRef<'_, I, E> {
        #[track_caller]
        fn sp_solve_lower_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
            faer_sparse::triangular_solve::solve_lower_triangular_in_place(
                *self,
                Conj::No,
                rhs.as_mat_mut(),
                get_global_parallelism(),
            );
        }
        #[track_caller]
        fn sp_solve_upper_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
            faer_sparse::triangular_solve::solve_upper_triangular_in_place(
                *self,
                Conj::No,
                rhs.as_mat_mut(),
                get_global_parallelism(),
            );
        }
        #[track_caller]
        fn sp_solve_unit_lower_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
            faer_sparse::triangular_solve::solve_unit_lower_triangular_in_place(
                *self,
                Conj::No,
                rhs.as_mat_mut(),
                get_global_parallelism(),
            );
        }
        #[track_caller]
        fn sp_solve_unit_upper_triangular_in_place(&self, mut rhs: impl AsMatMut<E>) {
            faer_sparse::triangular_solve::solve_unit_upper_triangular_in_place(
                *self,
                Conj::No,
                rhs.as_mat_mut(),
                get_global_parallelism(),
            );
        }

        /// Returns the Cholesky decomposition of `self`. Only the provided side is accessed.
        #[track_caller]
        fn sp_cholesky(
            &self,
            side: Side,
        ) -> Result<solvers::Cholesky<I, E>, sparse::CholeskyError> {
            solvers::Cholesky::try_new_with_symbolic(
                solvers::SymbolicCholesky::try_new(self.symbolic(), side)?,
                *self,
                side,
            )
        }

        /// Returns the LU decomposition of `self` with partial (row) pivoting.
        #[track_caller]
        fn sp_lu(&self) -> Result<solvers::Lu<I, E>, LuError> {
            solvers::Lu::try_new_with_symbolic(
                solvers::SymbolicLu::try_new(self.symbolic())?,
                *self,
            )
        }

        /// Returns the QR decomposition of `self`.
        #[track_caller]
        fn sp_qr(&self) -> Result<solvers::Qr<I, E>, FaerError> {
            solvers::Qr::try_new_with_symbolic(
                solvers::SymbolicQr::try_new(self.symbolic())?,
                *self,
            )
        }
    }
}

impl<E: Conjugate> FaerMat<E::Canonical> for MatRef<'_, E>
where
    E::Canonical: ComplexField,
{
    #[track_caller]
    fn solve_lower_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        let parallelism = get_global_parallelism();
        let mut rhs = rhs;
        faer_core::solve::solve_lower_triangular_in_place(*self, rhs.as_mat_mut(), parallelism);
    }
    #[track_caller]
    fn solve_upper_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        let parallelism = get_global_parallelism();
        let mut rhs = rhs;
        faer_core::solve::solve_upper_triangular_in_place(*self, rhs.as_mat_mut(), parallelism);
    }
    #[track_caller]
    fn solve_unit_lower_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        let parallelism = get_global_parallelism();
        let mut rhs = rhs;
        faer_core::solve::solve_unit_lower_triangular_in_place(
            *self,
            rhs.as_mat_mut(),
            parallelism,
        );
    }
    #[track_caller]
    fn solve_unit_upper_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        let parallelism = get_global_parallelism();
        let mut rhs = rhs;
        faer_core::solve::solve_unit_upper_triangular_in_place(
            *self,
            rhs.as_mat_mut(),
            parallelism,
        );
    }

    #[track_caller]
    fn cholesky(&self, side: Side) -> Result<Cholesky<E::Canonical>, CholeskyError> {
        Cholesky::try_new(self.as_ref(), side)
    }
    #[track_caller]
    fn lblt(&self, side: Side) -> Lblt<E::Canonical> {
        Lblt::new(self.as_ref(), side)
    }
    #[track_caller]
    fn partial_piv_lu(&self) -> PartialPivLu<E::Canonical> {
        PartialPivLu::<E::Canonical>::new(self.as_ref())
    }
    #[track_caller]
    fn full_piv_lu(&self) -> FullPivLu<E::Canonical> {
        FullPivLu::<E::Canonical>::new(self.as_ref())
    }
    #[track_caller]
    fn qr(&self) -> Qr<E::Canonical> {
        Qr::<E::Canonical>::new(self.as_ref())
    }
    #[track_caller]
    fn col_piv_qr(&self) -> ColPivQr<E::Canonical> {
        ColPivQr::<E::Canonical>::new(self.as_ref())
    }
    #[track_caller]
    fn svd(&self) -> Svd<E::Canonical> {
        Svd::<E::Canonical>::new(self.as_ref())
    }
    #[track_caller]
    fn thin_svd(&self) -> ThinSvd<E::Canonical> {
        ThinSvd::<E::Canonical>::new(self.as_ref())
    }
    #[track_caller]
    fn selfadjoint_eigendecomposition(
        &self,
        side: Side,
    ) -> SelfAdjointEigendecomposition<E::Canonical> {
        SelfAdjointEigendecomposition::<E::Canonical>::new(self.as_ref(), side)
    }

    #[track_caller]
    fn eigendecomposition<ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>>(
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

    #[track_caller]
    fn complex_eigendecomposition(&self) -> Eigendecomposition<E::Canonical> {
        Eigendecomposition::<E::Canonical>::new_from_complex(self.as_ref())
    }

    #[track_caller]
    fn determinant(&self) -> E::Canonical {
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

    #[track_caller]
    fn selfadjoint_eigenvalues(&self, side: Side) -> Vec<<E::Canonical as ComplexField>::Real> {
        let matrix = match side {
            Side::Lower => *self,
            Side::Upper => self.transpose(),
        };

        assert!(matrix.nrows() == matrix.ncols());
        let dim = matrix.nrows();
        let parallelism = get_global_parallelism();

        let mut s = Mat::<E::Canonical>::zeros(dim, 1);
        let params = Default::default();
        faer_evd::compute_hermitian_evd(
            matrix.canonicalize().0,
            s.as_mut(),
            None,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                faer_evd::compute_hermitian_evd_req::<E::Canonical>(
                    dim,
                    faer_evd::ComputeVectors::No,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        (0..dim).map(|i| s.read(i, 0).faer_real()).collect()
    }

    #[track_caller]
    fn singular_values(&self) -> Vec<<E::Canonical as ComplexField>::Real> {
        let dim = Ord::min(self.nrows(), self.ncols());
        let parallelism = get_global_parallelism();

        let mut s = Mat::<E::Canonical>::zeros(dim, 1);
        let params = Default::default();
        faer_svd::compute_svd(
            self.canonicalize().0,
            s.as_mut(),
            None,
            None,
            parallelism,
            PodStack::new(&mut GlobalPodBuffer::new(
                faer_svd::compute_svd_req::<E::Canonical>(
                    self.nrows(),
                    self.ncols(),
                    faer_svd::ComputeVectors::No,
                    faer_svd::ComputeVectors::No,
                    parallelism,
                    params,
                )
                .unwrap(),
            )),
            params,
        );

        (0..dim).map(|i| s.read(i, 0).faer_real()).collect()
    }

    #[track_caller]
    fn eigenvalues<ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>>(
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

    #[track_caller]
    fn complex_eigenvalues(&self) -> Vec<E::Canonical> {
        Eigendecomposition::<E::Canonical>::__values_from_complex_impl(self.canonicalize())
    }
}

impl<E: Conjugate> FaerMat<E::Canonical> for MatMut<'_, E>
where
    E::Canonical: ComplexField,
{
    #[track_caller]
    fn solve_lower_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        self.as_ref().solve_lower_triangular_in_place(rhs)
    }
    #[track_caller]
    fn solve_upper_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        self.as_ref().solve_upper_triangular_in_place(rhs)
    }
    #[track_caller]
    fn solve_unit_lower_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        self.as_ref().solve_unit_lower_triangular_in_place(rhs)
    }
    #[track_caller]
    fn solve_unit_upper_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        self.as_ref().solve_unit_upper_triangular_in_place(rhs)
    }

    #[track_caller]
    fn cholesky(&self, side: Side) -> Result<Cholesky<E::Canonical>, CholeskyError> {
        self.as_ref().cholesky(side)
    }
    #[track_caller]
    fn lblt(&self, side: Side) -> Lblt<E::Canonical> {
        self.as_ref().lblt(side)
    }
    #[track_caller]
    fn partial_piv_lu(&self) -> PartialPivLu<E::Canonical> {
        self.as_ref().partial_piv_lu()
    }
    #[track_caller]
    fn full_piv_lu(&self) -> FullPivLu<E::Canonical> {
        self.as_ref().full_piv_lu()
    }
    #[track_caller]
    fn qr(&self) -> Qr<E::Canonical> {
        self.as_ref().qr()
    }
    #[track_caller]
    fn col_piv_qr(&self) -> ColPivQr<E::Canonical> {
        self.as_ref().col_piv_qr()
    }
    #[track_caller]
    fn svd(&self) -> Svd<E::Canonical> {
        self.as_ref().svd()
    }
    #[track_caller]
    fn thin_svd(&self) -> ThinSvd<E::Canonical> {
        self.as_ref().thin_svd()
    }
    #[track_caller]
    fn selfadjoint_eigendecomposition(
        &self,
        side: Side,
    ) -> SelfAdjointEigendecomposition<E::Canonical> {
        self.as_ref().selfadjoint_eigendecomposition(side)
    }

    #[track_caller]
    fn eigendecomposition<ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>>(
        &self,
    ) -> Eigendecomposition<ComplexE> {
        self.as_ref().eigendecomposition::<ComplexE>()
    }

    #[track_caller]
    fn complex_eigendecomposition(&self) -> Eigendecomposition<E::Canonical> {
        self.as_ref().complex_eigendecomposition()
    }

    #[track_caller]
    fn determinant(&self) -> E::Canonical {
        self.as_ref().determinant()
    }

    #[track_caller]
    fn selfadjoint_eigenvalues(&self, side: Side) -> Vec<<E::Canonical as ComplexField>::Real> {
        self.as_ref().selfadjoint_eigenvalues(side)
    }

    #[track_caller]
    fn singular_values(&self) -> Vec<<E::Canonical as ComplexField>::Real> {
        self.as_ref().singular_values()
    }

    #[track_caller]
    fn eigenvalues<ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>>(
        &self,
    ) -> Vec<ComplexE> {
        self.as_ref().eigenvalues()
    }

    #[track_caller]
    fn complex_eigenvalues(&self) -> Vec<E::Canonical> {
        self.as_ref().complex_eigenvalues()
    }
}

impl<E: Conjugate> FaerMat<E::Canonical> for Mat<E>
where
    E::Canonical: ComplexField,
{
    #[track_caller]
    fn solve_lower_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        self.as_ref().solve_lower_triangular_in_place(rhs)
    }
    #[track_caller]
    fn solve_upper_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        self.as_ref().solve_upper_triangular_in_place(rhs)
    }
    #[track_caller]
    fn solve_unit_lower_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        self.as_ref().solve_unit_lower_triangular_in_place(rhs)
    }
    #[track_caller]
    fn solve_unit_upper_triangular_in_place(&self, rhs: impl AsMatMut<E::Canonical>) {
        self.as_ref().solve_unit_upper_triangular_in_place(rhs)
    }

    #[track_caller]
    fn cholesky(&self, side: Side) -> Result<Cholesky<E::Canonical>, CholeskyError> {
        self.as_ref().cholesky(side)
    }
    #[track_caller]
    fn lblt(&self, side: Side) -> Lblt<E::Canonical> {
        self.as_ref().lblt(side)
    }
    #[track_caller]
    fn partial_piv_lu(&self) -> PartialPivLu<E::Canonical> {
        self.as_ref().partial_piv_lu()
    }
    #[track_caller]
    fn full_piv_lu(&self) -> FullPivLu<E::Canonical> {
        self.as_ref().full_piv_lu()
    }
    #[track_caller]
    fn qr(&self) -> Qr<E::Canonical> {
        self.as_ref().qr()
    }
    #[track_caller]
    fn col_piv_qr(&self) -> ColPivQr<E::Canonical> {
        self.as_ref().col_piv_qr()
    }
    #[track_caller]
    fn svd(&self) -> Svd<E::Canonical> {
        self.as_ref().svd()
    }
    #[track_caller]
    fn thin_svd(&self) -> ThinSvd<E::Canonical> {
        self.as_ref().thin_svd()
    }
    #[track_caller]
    fn selfadjoint_eigendecomposition(
        &self,
        side: Side,
    ) -> SelfAdjointEigendecomposition<E::Canonical> {
        self.as_ref().selfadjoint_eigendecomposition(side)
    }

    #[track_caller]
    fn eigendecomposition<ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>>(
        &self,
    ) -> Eigendecomposition<ComplexE> {
        self.as_ref().eigendecomposition::<ComplexE>()
    }

    #[track_caller]
    fn complex_eigendecomposition(&self) -> Eigendecomposition<E::Canonical> {
        self.as_ref().complex_eigendecomposition()
    }

    #[track_caller]
    fn determinant(&self) -> E::Canonical {
        self.as_ref().determinant()
    }

    #[track_caller]
    fn selfadjoint_eigenvalues(&self, side: Side) -> Vec<<E::Canonical as ComplexField>::Real> {
        self.as_ref().selfadjoint_eigenvalues(side)
    }

    #[track_caller]
    fn singular_values(&self) -> Vec<<E::Canonical as ComplexField>::Real> {
        self.as_ref().singular_values()
    }

    #[track_caller]
    fn eigenvalues<ComplexE: ComplexField<Real = <E::Canonical as ComplexField>::Real>>(
        &self,
    ) -> Vec<ComplexE> {
        self.as_ref().eigenvalues()
    }

    #[track_caller]
    fn complex_eigenvalues(&self) -> Vec<E::Canonical> {
        self.as_ref().complex_eigenvalues()
    }
}

/// Conversions from external library matrix views into `faer` types.
pub trait IntoFaer {
    type Faer;
    fn into_faer(self) -> Self::Faer;
}

#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
/// Conversions from external library matrix views into `nalgebra` types.
pub trait IntoNalgebra {
    type Nalgebra;
    fn into_nalgebra(self) -> Self::Nalgebra;
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
/// Conversions from external library matrix views into `ndarray` types.
pub trait IntoNdarray {
    type Ndarray;
    fn into_ndarray(self) -> Self::Ndarray;
}

/// Conversions from external library matrix views into complex `faer` types.
pub trait IntoFaerComplex {
    type Faer;
    fn into_faer_complex(self) -> Self::Faer;
}

#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
/// Conversions from external library matrix views into complex `nalgebra` types.
pub trait IntoNalgebraComplex {
    type Nalgebra;
    fn into_nalgebra_complex(self) -> Self::Nalgebra;
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
/// Conversions from external library matrix views into complex `ndarray` types.
pub trait IntoNdarrayComplex {
    type Ndarray;
    fn into_ndarray_complex(self) -> Self::Ndarray;
}

#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
const _: () = {
    use complex_native::*;
    use faer_core::SimpleEntity;
    use nalgebra::{Dim, Dyn, MatrixView, MatrixViewMut, ViewStorage, ViewStorageMut};
    use num_complex::{Complex32, Complex64};

    impl<'a, T: SimpleEntity, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaer
        for MatrixView<'a, T, R, C, RStride, CStride>
    {
        type Faer = MatRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = self.as_ptr();
            unsafe {
                faer_core::mat::from_raw_parts(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a, T: SimpleEntity, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaer
        for MatrixViewMut<'a, T, R, C, RStride, CStride>
    {
        type Faer = MatMut<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = { self }.as_mut_ptr();
            unsafe {
                faer_core::mat::from_raw_parts_mut::<'_, T>(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a, T: SimpleEntity> IntoNalgebra for MatRef<'a, T> {
        type Nalgebra = MatrixView<'a, T, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr();
            unsafe {
                MatrixView::<'_, T, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
                    '_,
                    T,
                    Dyn,
                    Dyn,
                    Dyn,
                    Dyn,
                >::from_raw_parts(
                    ptr,
                    (Dyn(nrows), Dyn(ncols)),
                    (
                        Dyn(row_stride.try_into().unwrap()),
                        Dyn(col_stride.try_into().unwrap()),
                    ),
                ))
            }
        }
    }

    impl<'a, T: SimpleEntity> IntoNalgebra for MatMut<'a, T> {
        type Nalgebra = MatrixViewMut<'a, T, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr_mut();
            unsafe {
                MatrixViewMut::<'_, T, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorageMut::<
                    '_,
                    T,
                    Dyn,
                    Dyn,
                    Dyn,
                    Dyn,
                >::from_raw_parts(
                    ptr,
                    (Dyn(nrows), Dyn(ncols)),
                    (
                        Dyn(row_stride.try_into().unwrap()),
                        Dyn(col_stride.try_into().unwrap()),
                    ),
                ))
            }
        }
    }

    impl<'a, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaerComplex
        for MatrixView<'a, Complex32, R, C, RStride, CStride>
    {
        type Faer = MatRef<'a, c32>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = self.as_ptr() as *const c32;
            unsafe {
                faer_core::mat::from_raw_parts(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaerComplex
        for MatrixViewMut<'a, Complex32, R, C, RStride, CStride>
    {
        type Faer = MatMut<'a, c32>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = { self }.as_mut_ptr() as *mut c32;
            unsafe {
                faer_core::mat::from_raw_parts_mut(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a> IntoNalgebraComplex for MatRef<'a, c32> {
        type Nalgebra = MatrixView<'a, Complex32, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra_complex(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr() as *const Complex32;
            unsafe {
                MatrixView::<'_, Complex32, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
                    '_,
                    Complex32,
                    Dyn,
                    Dyn,
                    Dyn,
                    Dyn,
                >::from_raw_parts(
                    ptr,
                    (Dyn(nrows), Dyn(ncols)),
                    (
                        Dyn(row_stride.try_into().unwrap()),
                        Dyn(col_stride.try_into().unwrap()),
                    ),
                ))
            }
        }
    }

    impl<'a> IntoNalgebraComplex for MatMut<'a, c32> {
        type Nalgebra = MatrixViewMut<'a, Complex32, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra_complex(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr_mut() as *mut Complex32;
            unsafe {
                MatrixViewMut::<'_, Complex32, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorageMut::<
                    '_,
                    Complex32,
                    Dyn,
                    Dyn,
                    Dyn,
                    Dyn,
                >::from_raw_parts(
                    ptr,
                    (Dyn(nrows), Dyn(ncols)),
                    (
                        Dyn(row_stride.try_into().unwrap()),
                        Dyn(col_stride.try_into().unwrap()),
                    ),
                ))
            }
        }
    }

    impl<'a, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaerComplex
        for MatrixView<'a, Complex64, R, C, RStride, CStride>
    {
        type Faer = MatRef<'a, c64>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = self.as_ptr() as *const c64;
            unsafe {
                faer_core::mat::from_raw_parts(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoFaerComplex
        for MatrixViewMut<'a, Complex64, R, C, RStride, CStride>
    {
        type Faer = MatMut<'a, c64>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides = self.strides();
            let ptr = { self }.as_mut_ptr() as *mut c64;
            unsafe {
                faer_core::mat::from_raw_parts_mut(
                    ptr,
                    nrows,
                    ncols,
                    strides.0.try_into().unwrap(),
                    strides.1.try_into().unwrap(),
                )
            }
        }
    }

    impl<'a> IntoNalgebraComplex for MatRef<'a, c64> {
        type Nalgebra = MatrixView<'a, Complex64, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra_complex(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr() as *const Complex64;
            unsafe {
                MatrixView::<'_, Complex64, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
                    '_,
                    Complex64,
                    Dyn,
                    Dyn,
                    Dyn,
                    Dyn,
                >::from_raw_parts(
                    ptr,
                    (Dyn(nrows), Dyn(ncols)),
                    (
                        Dyn(row_stride.try_into().unwrap()),
                        Dyn(col_stride.try_into().unwrap()),
                    ),
                ))
            }
        }
    }

    impl<'a> IntoNalgebraComplex for MatMut<'a, c64> {
        type Nalgebra = MatrixViewMut<'a, Complex64, Dyn, Dyn, Dyn, Dyn>;

        #[track_caller]
        fn into_nalgebra_complex(self) -> Self::Nalgebra {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let ptr = self.as_ptr_mut() as *mut Complex64;
            unsafe {
                MatrixViewMut::<'_, Complex64, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorageMut::<
                    '_,
                    Complex64,
                    Dyn,
                    Dyn,
                    Dyn,
                    Dyn,
                >::from_raw_parts(
                    ptr,
                    (Dyn(nrows), Dyn(ncols)),
                    (
                        Dyn(row_stride.try_into().unwrap()),
                        Dyn(col_stride.try_into().unwrap()),
                    ),
                ))
            }
        }
    }
};

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
const _: () = {
    use complex_native::*;
    use faer_core::SimpleEntity;
    use ndarray::{ArrayView, ArrayViewMut, IntoDimension, Ix2, ShapeBuilder};
    use num_complex::{Complex32, Complex64};

    impl<'a, T: SimpleEntity> IntoFaer for ArrayView<'a, T, Ix2> {
        type Faer = MatRef<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = self.as_ptr();
            unsafe { faer_core::mat::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a, T: SimpleEntity> IntoFaer for ArrayViewMut<'a, T, Ix2> {
        type Faer = MatMut<'a, T>;

        #[track_caller]
        fn into_faer(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = { self }.as_mut_ptr();
            unsafe {
                faer_core::mat::from_raw_parts_mut::<'_, T>(
                    ptr, nrows, ncols, strides[0], strides[1],
                )
            }
        }
    }

    impl<'a, T: SimpleEntity> IntoNdarray for MatRef<'a, T> {
        type Ndarray = ArrayView<'a, T, Ix2>;

        #[track_caller]
        fn into_ndarray(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr();
            unsafe {
                ArrayView::<'_, T, Ix2>::from_shape_ptr(
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }

    impl<'a, T: SimpleEntity> IntoNdarray for MatMut<'a, T> {
        type Ndarray = ArrayViewMut<'a, T, Ix2>;

        #[track_caller]
        fn into_ndarray(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr_mut();
            unsafe {
                ArrayViewMut::<'_, T, Ix2>::from_shape_ptr(
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }

    impl<'a> IntoFaerComplex for ArrayView<'a, Complex32, Ix2> {
        type Faer = MatRef<'a, c32>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = self.as_ptr() as *const c32;
            unsafe { faer_core::mat::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a> IntoFaerComplex for ArrayViewMut<'a, Complex32, Ix2> {
        type Faer = MatMut<'a, c32>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = { self }.as_mut_ptr() as *mut c32;
            unsafe { faer_core::mat::from_raw_parts_mut(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a> IntoNdarrayComplex for MatRef<'a, c32> {
        type Ndarray = ArrayView<'a, Complex32, Ix2>;

        #[track_caller]
        fn into_ndarray_complex(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr() as *const Complex32;
            unsafe {
                ArrayView::<'_, Complex32, Ix2>::from_shape_ptr(
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }

    impl<'a> IntoNdarrayComplex for MatMut<'a, c32> {
        type Ndarray = ArrayViewMut<'a, Complex32, Ix2>;

        #[track_caller]
        fn into_ndarray_complex(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr_mut() as *mut Complex32;
            unsafe {
                ArrayViewMut::<'_, Complex32, Ix2>::from_shape_ptr(
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }

    impl<'a> IntoFaerComplex for ArrayView<'a, Complex64, Ix2> {
        type Faer = MatRef<'a, c64>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = self.as_ptr() as *const c64;
            unsafe { faer_core::mat::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a> IntoFaerComplex for ArrayViewMut<'a, Complex64, Ix2> {
        type Faer = MatMut<'a, c64>;

        #[track_caller]
        fn into_faer_complex(self) -> Self::Faer {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let strides: [isize; 2] = self.strides().try_into().unwrap();
            let ptr = { self }.as_mut_ptr() as *mut c64;
            unsafe { faer_core::mat::from_raw_parts_mut(ptr, nrows, ncols, strides[0], strides[1]) }
        }
    }

    impl<'a> IntoNdarrayComplex for MatRef<'a, c64> {
        type Ndarray = ArrayView<'a, Complex64, Ix2>;

        #[track_caller]
        fn into_ndarray_complex(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr() as *const Complex64;
            unsafe {
                ArrayView::<'_, Complex64, Ix2>::from_shape_ptr(
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }

    impl<'a> IntoNdarrayComplex for MatMut<'a, c64> {
        type Ndarray = ArrayViewMut<'a, Complex64, Ix2>;

        #[track_caller]
        fn into_ndarray_complex(self) -> Self::Ndarray {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride: usize = self.row_stride().try_into().unwrap();
            let col_stride: usize = self.col_stride().try_into().unwrap();
            let ptr = self.as_ptr_mut() as *mut Complex64;
            unsafe {
                ArrayViewMut::<'_, Complex64, Ix2>::from_shape_ptr(
                    (nrows, ncols)
                        .into_shape()
                        .strides((row_stride, col_stride).into_dimension()),
                    ptr,
                )
            }
        }
    }
};

#[cfg(all(feature = "nalgebra", feature = "ndarray"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "ndarray"))))]
const _: () =
    {
        use nalgebra::{Dim, Dyn, MatrixView, MatrixViewMut, ViewStorage, ViewStorageMut};
        use ndarray::{ArrayView, ArrayViewMut, IntoDimension, Ix2, ShapeBuilder};
        use num_complex::Complex;

        impl<'a, T> IntoNalgebra for ArrayView<'a, T, Ix2> {
            type Nalgebra = MatrixView<'a, T, Dyn, Dyn, Dyn, Dyn>;

            #[track_caller]
            fn into_nalgebra(self) -> Self::Nalgebra {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let [row_stride, col_stride]: [isize; 2] = self.strides().try_into().unwrap();
                let ptr = self.as_ptr();

                unsafe {
                    MatrixView::<'_, T, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
                        '_,
                        T,
                        Dyn,
                        Dyn,
                        Dyn,
                        Dyn,
                    >::from_raw_parts(
                        ptr,
                        (Dyn(nrows), Dyn(ncols)),
                        (
                            Dyn(row_stride.try_into().unwrap()),
                            Dyn(col_stride.try_into().unwrap()),
                        ),
                    ))
                }
            }
        }
        impl<'a, T> IntoNalgebra for ArrayViewMut<'a, T, Ix2> {
            type Nalgebra = MatrixViewMut<'a, T, Dyn, Dyn, Dyn, Dyn>;

            #[track_caller]
            fn into_nalgebra(self) -> Self::Nalgebra {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let [row_stride, col_stride]: [isize; 2] = self.strides().try_into().unwrap();
                let ptr = { self }.as_mut_ptr();

                unsafe {
                    MatrixViewMut::<'_, T, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorageMut::<
                        '_,
                        T,
                        Dyn,
                        Dyn,
                        Dyn,
                        Dyn,
                    >::from_raw_parts(
                        ptr,
                        (Dyn(nrows), Dyn(ncols)),
                        (
                            Dyn(row_stride.try_into().unwrap()),
                            Dyn(col_stride.try_into().unwrap()),
                        ),
                    ))
                }
            }
        }

        impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoNdarray
            for MatrixView<'a, T, R, C, RStride, CStride>
        {
            type Ndarray = ArrayView<'a, T, Ix2>;

            #[track_caller]
            fn into_ndarray(self) -> Self::Ndarray {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let (row_stride, col_stride) = self.strides();
                let ptr = self.as_ptr();

                unsafe {
                    ArrayView::<'_, T, Ix2>::from_shape_ptr(
                        (nrows, ncols)
                            .into_shape()
                            .strides((row_stride, col_stride).into_dimension()),
                        ptr,
                    )
                }
            }
        }
        impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoNdarray
            for MatrixViewMut<'a, T, R, C, RStride, CStride>
        {
            type Ndarray = ArrayViewMut<'a, T, Ix2>;

            #[track_caller]
            fn into_ndarray(self) -> Self::Ndarray {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let (row_stride, col_stride) = self.strides();
                let ptr = { self }.as_mut_ptr();

                unsafe {
                    ArrayViewMut::<'_, T, Ix2>::from_shape_ptr(
                        (nrows, ncols)
                            .into_shape()
                            .strides((row_stride, col_stride).into_dimension()),
                        ptr,
                    )
                }
            }
        }

        impl<'a, T> IntoNalgebraComplex for ArrayView<'a, Complex<T>, Ix2> {
            type Nalgebra = MatrixView<'a, Complex<T>, Dyn, Dyn, Dyn, Dyn>;

            #[track_caller]
            fn into_nalgebra_complex(self) -> Self::Nalgebra {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let [row_stride, col_stride]: [isize; 2] = self.strides().try_into().unwrap();
                let ptr = self.as_ptr();

                unsafe {
                    MatrixView::<'_, Complex<T>, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
                        '_,
                        Complex<T>,
                        Dyn,
                        Dyn,
                        Dyn,
                        Dyn,
                    >::from_raw_parts(
                        ptr,
                        (Dyn(nrows), Dyn(ncols)),
                        (
                            Dyn(row_stride.try_into().unwrap()),
                            Dyn(col_stride.try_into().unwrap()),
                        ),
                    ))
                }
            }
        }
        impl<'a, T> IntoNalgebraComplex for ArrayViewMut<'a, Complex<T>, Ix2> {
            type Nalgebra = MatrixViewMut<'a, Complex<T>, Dyn, Dyn, Dyn, Dyn>;

            #[track_caller]
            fn into_nalgebra_complex(self) -> Self::Nalgebra {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let [row_stride, col_stride]: [isize; 2] = self.strides().try_into().unwrap();
                let ptr = { self }.as_mut_ptr();

                unsafe {
                    MatrixViewMut::<'_, Complex<T>, Dyn, Dyn, Dyn, Dyn>::from_data(
                        ViewStorageMut::<'_, Complex<T>, Dyn, Dyn, Dyn, Dyn>::from_raw_parts(
                            ptr,
                            (Dyn(nrows), Dyn(ncols)),
                            (
                                Dyn(row_stride.try_into().unwrap()),
                                Dyn(col_stride.try_into().unwrap()),
                            ),
                        ),
                    )
                }
            }
        }

        impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoNdarrayComplex
            for MatrixView<'a, Complex<T>, R, C, RStride, CStride>
        {
            type Ndarray = ArrayView<'a, Complex<T>, Ix2>;

            #[track_caller]
            fn into_ndarray_complex(self) -> Self::Ndarray {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let (row_stride, col_stride) = self.strides();
                let ptr = self.as_ptr();

                unsafe {
                    ArrayView::<'_, Complex<T>, Ix2>::from_shape_ptr(
                        (nrows, ncols)
                            .into_shape()
                            .strides((row_stride, col_stride).into_dimension()),
                        ptr,
                    )
                }
            }
        }
        impl<'a, T, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoNdarrayComplex
            for MatrixViewMut<'a, Complex<T>, R, C, RStride, CStride>
        {
            type Ndarray = ArrayViewMut<'a, Complex<T>, Ix2>;

            #[track_caller]
            fn into_ndarray_complex(self) -> Self::Ndarray {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let (row_stride, col_stride) = self.strides();
                let ptr = { self }.as_mut_ptr();

                unsafe {
                    ArrayViewMut::<'_, Complex<T>, Ix2>::from_shape_ptr(
                        (nrows, ncols)
                            .into_shape()
                            .strides((row_stride, col_stride).into_dimension()),
                        ptr,
                    )
                }
            }
        }
    };

#[cfg(feature = "polars")]
#[cfg_attr(docsrs, doc(cfg(feature = "polars")))]
pub mod polars {
    use super::Mat;
    use polars::prelude::*;

    pub trait Frame {
        fn is_valid(self) -> PolarsResult<LazyFrame>;
    }

    impl Frame for LazyFrame {
        fn is_valid(self) -> PolarsResult<LazyFrame> {
            let test_dtypes: bool = self
                .clone()
                .limit(0)
                .collect()
                .unwrap()
                .dtypes()
                .into_iter()
                .map(|e| {
                    matches!(
                        e,
                        DataType::UInt8
                            | DataType::UInt16
                            | DataType::UInt32
                            | DataType::UInt64
                            | DataType::Int8
                            | DataType::Int16
                            | DataType::Int32
                            | DataType::Int64
                            | DataType::Float32
                            | DataType::Float64
                    )
                })
                .all(|e| e);
            let test_no_nulls: bool = self
                .clone()
                .null_count()
                .with_column(
                    fold_exprs(lit(0u64), |acc, x| Ok(Some(acc + x)), [col("*")]).alias("sum"),
                )
                .select(&[col("sum")])
                .collect()
                .unwrap()
                .column("sum")
                .unwrap()
                .u64()
                .unwrap()
                .into_iter()
                .map(|e| e.eq(&Some(0u64)))
                .collect::<Vec<_>>()[0];
            match (test_dtypes, test_no_nulls) {
                (true, true) => Ok(self),
                (false, true) => Err(PolarsError::InvalidOperation(
                    "frame contains non-numerical data".into(),
                )),
                (true, false) => Err(PolarsError::InvalidOperation(
                    "frame contains null entries".into(),
                )),
                (false, false) => Err(PolarsError::InvalidOperation(
                    "frame contains non-numerical data and null entries".into(),
                )),
            }
        }
    }

    macro_rules! polars_impl {
        ($ty: ident, $dtype: ident, $fn_name: ident) => {
            /// Converts a `polars` lazyframe into a [`Mat`].
            ///
            /// Note that this function expects that the frame passed "looks like"
            /// a numerical array and all values will be cast to either f32 or f64
            /// prior to building [`Mat`].
            ///
            /// Passing a frame with either non-numerical column data or null
            /// entries will result in a error. Users are expected to reolve
            /// these issues in `polars` prior calling this function.
            #[cfg(feature = "polars")]
            #[cfg_attr(docsrs, doc(cfg(feature = "polars")))]
            pub fn $fn_name(
                frame: impl Frame,
            ) -> PolarsResult<Mat<$ty>> {
                use core::{iter::zip, mem::MaybeUninit};

                fn implementation(
                    lf: LazyFrame,
                ) -> PolarsResult<Mat<$ty>> {
                    let df = lf
                        .select(&[col("*").cast(DataType::$dtype)])
                        .collect()
                        .unwrap();

                    let nrows = df.height();
                    let ncols = df.get_column_names().len();

                    let mut out = Mat::<$ty>::with_capacity(df.height(), df.get_column_names().len());

                    df.get_column_names().iter()
                        .enumerate()
                        .try_for_each(|(j, col)| -> PolarsResult<()> {
                            let mut row_start = 0usize;

                            // SAFETY: this is safe since we allocated enough space for `ncols` columns and
                            // `nrows` rows
                            let out_col = unsafe {
                                core::slice::from_raw_parts_mut(
                                    out.as_mut().ptr_at_mut(0, j) as *mut MaybeUninit<$ty>,
                                    nrows,
                                )
                            };

                            df.column(col)?.$ty()?.downcast_iter().try_for_each(
                                |chunk| -> PolarsResult<()> {
                                    let len = chunk.len();
                                    if len == 0 {
                                        return Ok(());
                                    }

                                    match row_start.checked_add(len) {
                                        Some(next_row_start) => {
                                            if next_row_start <= nrows {
                                                let mut out_slice = &mut out_col[row_start..next_row_start];
                                                let mut values = chunk.values_iter().as_slice();
                                                let validity = chunk.validity();

                                                assert_eq!(values.len(), len);

                                                match validity {
                                                    Some(bitmap) => {
                                                        let (mut bytes, offset, bitmap_len) = bitmap.as_slice();
                                                        assert_eq!(bitmap_len, len);
                                                        const BITS_PER_BYTE: usize = 8;

                                                        if offset > 0 {
                                                            let first_byte_len = Ord::min(len, 8 - offset);

                                                            let (out_prefix, out_suffix) = out_slice.split_at_mut(first_byte_len);
                                                            let (values_prefix, values_suffix) = values.split_at(first_byte_len);

                                                            for (out_elem, value_elem) in zip(
                                                                out_prefix,
                                                                values_prefix,
                                                            ) {
                                                                *out_elem = MaybeUninit::new(*value_elem)
                                                            }

                                                            bytes = &bytes[1..];
                                                            values = values_suffix;
                                                            out_slice = out_suffix;
                                                        }

                                                        if bytes.len() > 0 {
                                                            for (out_slice8, values8) in zip(
                                                                out_slice.chunks_exact_mut(BITS_PER_BYTE),
                                                                values.chunks_exact(BITS_PER_BYTE),
                                                            ) {
                                                                for (out_elem, value_elem) in zip(out_slice8, values8) {
                                                                    *out_elem = MaybeUninit::new(*value_elem);
                                                                }
                                                            }

                                                            for (out_elem, value_elem) in zip(
                                                                out_slice.chunks_exact_mut(BITS_PER_BYTE).into_remainder(),
                                                                values.chunks_exact(BITS_PER_BYTE).remainder(),
                                                            ) {
                                                                *out_elem = MaybeUninit::new(*value_elem);
                                                            }
                                                        }
                                                    }
                                                    None => {
                                                        // SAFETY: T and MaybeUninit<T> have the same layout
                                                        // NOTE: This state should not be reachable
                                                        let values = unsafe {
                                                            core::slice::from_raw_parts(
                                                                values.as_ptr() as *const MaybeUninit<$ty>,
                                                                values.len(),
                                                            )
                                                        };
                                                        out_slice.copy_from_slice(values);
                                                    }
                                                }

                                                row_start = next_row_start;
                                                Ok(())
                                            } else {
                                                Err(PolarsError::ShapeMismatch(
                                                    format!("too many values in column {col}").into(),
                                                ))
                                            }
                                        }
                                        None => Err(PolarsError::ShapeMismatch(
                                            format!("too many values in column {col}").into(),
                                        )),
                                    }
                                },
                            )?;

                            if row_start < nrows {
                                Err(PolarsError::ShapeMismatch(
                                    format!("not enough values in column {col} (column has {row_start} values, while dataframe has {nrows} rows)").into(),
                                ))
                            } else {
                                Ok(())
                            }
                        })?;

                    // SAFETY: we initialized every `ncols` columns, and each one was initialized with `nrows`
                    // elements
                    unsafe { out.set_dims(nrows, ncols) };

                    Ok(out)
                }

                implementation(frame.is_valid()?)
            }
        };
    }

    polars_impl!(f32, Float32, polars_to_faer_f32);
    polars_impl!(f64, Float64, polars_to_faer_f64);
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;
    use complex_native::*;
    use faer_core::{assert, RealField};
    use matrixcompare::prop_assert_matrix_eq;
    use proptest::prelude::*;
    use proptest::strategy::Strategy;

    use crate::proptest_support::*;

    fn prop_real() -> impl Strategy<Value = f64> + Clone {
        -1e12..=1e12
    }

    #[allow(unused)]
    fn prop_complex() -> impl Strategy<Value = c64> + Clone {
        (prop_real(), prop_real()).prop_map(|(re, im)| c64::new(re, im))
    }

    #[allow(unused)]
    fn mat_rhs<E: ComplexField>(
        element: impl Strategy<Value = E> + Clone,
        mat_structure: MatrixStructure,
    ) -> impl Strategy<Value = (Mat<E>, Mat<E>)> {
        let m = square_mat_with(
            element.clone(),
            Parameters::default().structure(mat_structure),
        );
        let rhs = mat(element);
        (m, rhs).prop_map(|(mut m, mut rhs)| {
            // Force compatible dimensions.
            let dim = core::cmp::min(m.nrows(), rhs.nrows());

            m.resize_with(dim, dim, |_, _| unreachable!());
            rhs.resize_with(dim, rhs.ncols(), |_, _| unreachable!());

            (m, rhs)
        })
    }

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
        test_solver(&H.adjoint().to_owned(), &H.adjoint().svd());

        let svd = H.svd();
        for i in 0..n - 1 {
            assert!(svd.s_diagonal()[(i, 0)].re >= svd.s_diagonal()[(i + 1, 0)].re);
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
        test_solver(&H.adjoint().to_owned(), &H.adjoint().thin_svd());
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
            &H.adjoint().to_owned(),
            &H.adjoint().selfadjoint_eigendecomposition(Side::Lower),
        );
        test_solver(
            &H.adjoint().to_owned(),
            &H.adjoint().selfadjoint_eigendecomposition(Side::Upper),
        );

        let evd = H.selfadjoint_eigendecomposition(Side::Lower);
        for i in 0..n - 1 {
            assert!(evd.s_diagonal()[(i, 0)].re <= evd.s_diagonal()[(i + 1, 0)].re);
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
            let mut s = Mat::zeros(n, n);
            s.as_mut()
                .diagonal_mut()
                .column_vector_mut()
                .copy_from(eigen.s_diagonal());
            let u = eigen.u();
            assert_approx_eq(u * &s, &H * u);
        }

        {
            let eigen = H.complex_eigendecomposition();
            let mut s = Mat::zeros(n, n);
            s.as_mut()
                .diagonal_mut()
                .column_vector_mut()
                .copy_from(eigen.s_diagonal());
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
        let mut s = Mat::zeros(n, n);
        s.as_mut()
            .diagonal_mut()
            .column_vector_mut()
            .copy_from(eigen.s_diagonal());
        let u = eigen.u();
        assert_approx_eq(u * &s, &H * u);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ext_ndarray() {
        let mut I_faer = Mat::<f32>::identity(8, 7);
        let mut I_ndarray = ndarray::Array2::<f32>::zeros([8, 7]);
        I_ndarray.diag_mut().fill(1.0);

        assert_matrix_eq!(I_ndarray.view().into_faer(), I_faer, comp = exact);
        assert!(I_faer.as_ref().into_ndarray() == I_ndarray);

        assert!(I_ndarray.view_mut().into_faer() == I_faer);
        assert!(I_faer.as_mut().into_ndarray() == I_ndarray);
    }

    #[cfg(feature = "nalgebra")]
    #[test]
    fn test_ext_nalgebra() {
        let mut I_faer = Mat::<f32>::identity(8, 7);
        let mut I_nalgebra = nalgebra::DMatrix::<f32>::identity(8, 7);

        assert!(I_nalgebra.view_range(.., ..).into_faer() == I_faer);
        assert!(I_faer.as_ref().into_nalgebra() == I_nalgebra);

        assert!(I_nalgebra.view_range_mut(.., ..).into_faer() == I_faer);
        assert!(I_faer.as_mut().into_nalgebra() == I_nalgebra);
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_polars_pos() {
        use crate::polars::{polars_to_faer_f32, polars_to_faer_f64};
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", [10, 11, 12]);

        let lf = DataFrame::new(vec![s0, s1]).unwrap().lazy();

        let arr_32 = polars_to_faer_f32(lf.clone()).unwrap();
        let arr_64 = polars_to_faer_f64(lf).unwrap();

        let expected_32 = mat![[1f32, 10f32], [2f32, 11f32], [3f32, 12f32]];
        let expected_64 = mat![[1f64, 10f64], [2f64, 11f64], [3f64, 12f64]];

        assert_approx_eq(arr_32, expected_32);
        assert_approx_eq(arr_64, expected_64);
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains null entries")]
    fn test_polars_neg_32_null() {
        use crate::polars::polars_to_faer_f32;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", [Some(10), Some(11), None]);

        let lf = DataFrame::new(vec![s0, s1]).unwrap().lazy();

        polars_to_faer_f32(lf).unwrap();
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains non-numerical data")]
    fn test_polars_neg_32_strl() {
        use crate::polars::polars_to_faer_f32;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", ["fish", "dog", "crocodile"]);

        let lf = DataFrame::new(vec![s0, s1]).unwrap().lazy();

        polars_to_faer_f32(lf).unwrap();
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains non-numerical data and null entries")]
    fn test_polars_neg_32_combo() {
        use crate::polars::polars_to_faer_f32;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", [Some(10), Some(11), None]);
        let s2: Series = Series::new("c", [Some("fish"), Some("dog"), None]);

        let lf = DataFrame::new(vec![s0, s1, s2]).unwrap().lazy();

        polars_to_faer_f32(lf).unwrap();
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains null entries")]
    fn test_polars_neg_64_null() {
        use crate::polars::polars_to_faer_f64;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", [Some(10), Some(11), None]);

        let lf = DataFrame::new(vec![s0, s1]).unwrap().lazy();

        polars_to_faer_f64(lf).unwrap();
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains non-numerical data")]
    fn test_polars_neg_64_strl() {
        use crate::polars::polars_to_faer_f64;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", ["fish", "dog", "crocodile"]);

        let lf = DataFrame::new(vec![s0, s1]).unwrap().lazy();

        polars_to_faer_f64(lf).unwrap();
    }

    #[cfg(feature = "polars")]
    #[test]
    #[should_panic(expected = "frame contains non-numerical data and null entries")]
    fn test_polars_neg_64_combo() {
        use crate::polars::polars_to_faer_f64;
        #[rustfmt::skip]
        use ::polars::prelude::*;

        let s0: Series = Series::new("a", [1, 2, 3]);
        let s1: Series = Series::new("b", [Some(10), Some(11), None]);
        let s2: Series = Series::new("c", [Some("fish"), Some("dog"), None]);

        let lf = DataFrame::new(vec![s0, s1, s2]).unwrap().lazy();

        polars_to_faer_f64(lf).unwrap();
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
            .max_by(|a, b| a.partial_cmp(&b).unwrap())
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
            .max_by(|a, b| a.partial_cmp(&b).unwrap())
            .unwrap();
        let correct_lamba_1 = 1.414213562373095;
        assert!(
            (lambda_1 - correct_lamba_1).abs() < 1e-10,
            "lambda_1 = {lambda_1}, correct_lamba_1 = {correct_lamba_1}",
        );
    }

    proptest! {
        // #[test]
        // fn prop_solve_lower_triangular((m, rhs) in mat_rhs(prop_real(), structure().unit_triangular(Triangular::Lower))) {
        //     let sol = m.solve_lower_triangular(rhs.clone());
        //     let tol = relative_epsilon_cond(&m, &rhs, 10.0);
        //     let m_times_sol = m * sol;
        //     prop_assert_matrix_eq!(m_times_sol, rhs, comp = abs, tol = tol);
        // }

        #[test]
        fn prop_qr_real(H in square_mat(prop_real())) {
            let qr = H.qr();

            let q_times_r = qr.compute_q() * qr.compute_r();
            let tol = relative_epsilon_norms(&q_times_r, &H, 10.0);
            prop_assert_matrix_eq!(q_times_r, &H, comp = abs, tol = tol);

            let q_times_r_thin = qr.compute_thin_q() * qr.compute_thin_r();
            let tol = relative_epsilon_norms(&q_times_r_thin, &H, 10.0);
            prop_assert_matrix_eq!(q_times_r_thin, &H, comp = abs, tol = tol);
        }

        #[test]
        fn prop_real_eigendecomposition(H_real in square_mat(prop_real())) {
            let n = H_real.nrows();
            let H = Mat::from_fn(n, n, |i, j| c64::new(H_real.read(i, j), 0.0));

            let eigen = H_real.eigendecomposition::<c64>();
            let mut s = Mat::zeros(n, n);
            s.as_mut()
                .diagonal_mut()
                .column_vector_mut()
                .copy_from(eigen.s_diagonal());
            let u = eigen.u();

            let u_times_s = u * &s;
            let h_times_u = &H * u;
            let tol = relative_epsilon_norms(&u_times_s, &h_times_u, 10.0);

            let u_times_s_re = zipped!(u_times_s.as_ref()).map(|unzipped!(x)| x.read().faer_real());
            let h_times_u_re = zipped!(u_times_s.as_ref()).map(|unzipped!(x)| x.read().faer_real());
            prop_assert_matrix_eq!(u_times_s_re, h_times_u_re, comp = abs, tol = tol);

            let u_times_s_im = zipped!(u_times_s.as_ref()).map(|unzipped!(x)| x.read().faer_imag());
            let h_times_u_im = zipped!(u_times_s.as_ref()).map(|unzipped!(x)| x.read().faer_imag());
            prop_assert_matrix_eq!(u_times_s_im, h_times_u_im, comp = abs, tol = tol);
        }
    }
}
