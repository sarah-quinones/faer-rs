//! `faer` is a general-purpose linear algebra library for Rust, with a focus on high performance
//! for algebraic operations on medium/large matrices, as well as matrix decompositions.
//!
//! Most of the high-level functionality in this library is provided through associated functions in
//! its vocabulary types: [`Mat`]/[`MatRef`]/[`MatMut`], as well as the [`Faer`] extension trait.
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
//! use faer::{mat, prelude::*, Mat, Scale};
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
//! let scale = Scale(3.0) * &a;
//! let mul = &a * b.transpose();
//!
//! let a00 = a[(0, 0)];
//! ```
//!
//! # Matrix decompositions
//! `faer` provides a variety of matrix factorizations, each with its own advantages and drawbacks:
//!
//! ## Cholesky decomposition
//! [`Faer::cholesky`] decomposes a self-adjoint positive definite matrix $A$ such that
//! $$A = LL^H,$$
//! where $L$ is a lower triangular matrix. This decomposition is highly efficient and has good
//! stability properties.
//! ## LU decomposition with partial pivoting
//! [`Faer::partial_piv_lu`] decomposes a square invertible matrix $A$ into a lower triangular
//! matrix $L$, a unit upper triangular matrix $U$, and a permutation matrix $P$, such that
//! $$PA = LU.$$
//! It is used by default for computing the determinant, and is generally the recommended method
//! for solving a square linear system or computing the inverse of a matrix (although we generally
//! recommend using a [`Solver`] instead of computing the inverse explicitly).
//!
//! ## LU decomposition with full pivoting
//! [`Faer::full_piv_lu`] Decomposes a generic rectangular matrix $A$ into a lower triangular matrix
//! $L$, a unit upper triangular matrix $U$, and permutation matrices $P$ and $Q$, such that
//! $$PAQ^\top = LU.$$
//! It can be more stable than the LU decomposition with partial pivoting, in exchange for being
//! more computationally expensive.
//!
//! ## QR decomposition
//! The QR decomposition ([`Faer::qr`]) decomposes a matrix $A$ into the product
//! $$A = QR,$$
//! where $Q$ is a unitary matrix, and $R$ is an upper trapezoidal matrix. It is often used for
//! solving least squares problems.
//!
//! ## QR decomposition with column pivoting
//! The QR decomposition with column pivoting ([`Faer::col_piv_qr`]) decomposes a matrix $A$ into
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
//! [`Faer::svd`], or only their first $\min(m, n)$ columns are computed, using [`Faer::thin_svd`].
//!
//! If only the singular values (elements of $S$) are desired, they can be obtained in
//! nonincreasing order using [`Faer::singular_values`].
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
//! * [`Faer::selfadjoint_eigendecomposition`] can be used with either real or complex matrices,
//! producing an eigendecomposition of the same type.
//! * [`Faer::eigendecomposition`] can be used with either real or complex matrices, but the output
//! complex type has to be specified.
//! * [`Faer::complex_eigendecomposition`] can only be used with complex matrices, with the output
//! having the same type.
//!
//! If only the eigenvalues (elements of $S$) are desired, they can be obtained in
//! nonincreasing order using [`Faer::selfadjoint_eigenvalues`], [`Faer::eigenvalues`], or
//! [`Faer::complex_eigenvalues`], with the same conditions described above.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

use dyn_stack::{GlobalPodBuffer, PodStack};
use faer_cholesky::llt::CholeskyError;
use faer_core::{AsMatMut, AsMatRef, ComplexField, Conj, Conjugate, Entity};
use prelude::*;
use solvers::*;

/// Commonly used traits for a streamlined user experience.
pub mod prelude {
    pub use crate::{
        solvers::{Solver, SolverCore, SolverLstsq, SolverLstsqCore},
        Faer, IntoFaer,
    };
    pub use reborrow::{IntoConst, Reborrow, ReborrowMut};
}

pub use faer_core::{
    complex_native, get_global_parallelism, mat, set_global_parallelism, Mat, MatMut, MatRef,
    Parallelism, Scale,
};
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub use matrixcompare::assert_matrix_eq;

/// Specifies whether the triangular lower or upper part of a matrix should be accessed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    Lower,
    Upper,
}

extern crate alloc;
use alloc::{vec, vec::Vec};

/// Matrix solvers and decompositions.
pub mod solvers {
    use super::*;
    #[cfg(feature = "std")]
    use assert2::assert;
    use faer_core::{permutation::PermutationRef, zipped};

    pub trait SolverCore<E: Entity> {
        /// Returns the number of rows of the matrix used to construct this decomposition.
        fn nrows(&self) -> usize;
        /// Returns the number of columns of the matrix used to construct this decomposition.
        fn ncols(&self) -> usize;

        /// Reconstructs the original matrix using the decomposition.
        fn reconstruct(&self) -> Mat<E>;
        /// Computes the inverse of the original matrix using the decomposition.
        ///
        /// # Panics
        /// Panics if the matrix is not square.
        fn inverse(&self) -> Mat<E>;

        #[doc(hidden)]
        fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj);
        #[doc(hidden)]
        fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj);
    }

    pub trait SolverLstsqCore<E: Entity>: SolverCore<E> {
        #[doc(hidden)]
        fn solve_lstsq_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj);
    }

    pub trait SolverLstsq<E: Entity>: SolverLstsqCore<E> {
        /// Solves the equation `self * X = rhs`, in the sense of least squares, and stores the
        /// result in the top rows of `rhs`.
        fn solve_lstsq_in_place(&self, rhs: impl AsMatMut<E>);
        /// Solves the equation `conjugate(self) * X = rhs`, in the sense of least squares, and
        /// stores the result in the top rows of `rhs`.
        fn solve_lstsq_conj_in_place(&self, rhs: impl AsMatMut<E>);
        /// Solves the equation `self * X = rhs`, and returns the result.
        fn solve_lstsq<ViewE: Conjugate<Canonical = E>>(&self, rhs: impl AsMatRef<ViewE>)
            -> Mat<E>;
        /// Solves the equation `conjugate(self) * X = rhs`, and returns the result.
        fn solve_lstsq_conj<ViewE: Conjugate<Canonical = E>>(
            &self,
            rhs: impl AsMatRef<ViewE>,
        ) -> Mat<E>;
    }

    #[track_caller]
    fn solve_lstsq_with_conj_impl<
        E: ComplexField,
        D: ?Sized + SolverLstsqCore<E>,
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

    #[track_caller]
    fn solve_with_conj_impl<
        E: ComplexField,
        D: ?Sized + SolverCore<E>,
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
        D: ?Sized + SolverCore<E>,
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

    const _: () = {
        fn __assert_object_safe<E: ComplexField>() {
            let _: Option<&dyn SolverCore<E>> = None;
            let _: Option<&dyn SolverLstsqCore<E>> = None;
        }
    };

    pub trait Solver<E: ComplexField>: SolverCore<E> {
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

    impl<E: ComplexField, Dec: ?Sized + SolverLstsqCore<E>> SolverLstsq<E> for Dec {
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

    impl<E: ComplexField, Dec: ?Sized + SolverCore<E>> Solver<E> for Dec {
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

    /// Cholesky decomposition.
    pub struct Cholesky<E: Entity> {
        factors: Mat<E>,
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
        s: Mat<E>,
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
                        |mut dst, src| dst.write(src.read().canonicalize()),
                    );
                }
                Side::Upper => {
                    zipped!(factors.as_mut(), matrix.adjoint()).for_each_triangular_lower(
                        faer_core::zip::Diag::Include,
                        |mut dst, src| dst.write(src.read().canonicalize()),
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
                .for_each_triangular_upper(faer_core::zip::Diag::Skip, |mut dst| {
                    dst.write(E::faer_zero())
                });
            factor
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
                    faer_lu::partial_pivoting::compute::lu_in_place_req::<E>(
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
                n_transpositions,
                factors,
                row_perm,
                row_perm_inv,
            }
        }

        fn dim(&self) -> usize {
            self.factors.nrows()
        }

        pub fn row_permutation(&self) -> PermutationRef<'_> {
            unsafe { PermutationRef::new_unchecked(&self.row_perm, &self.row_perm_inv) }
        }

        pub fn transposition_count(&self) -> usize {
            self.n_transpositions
        }

        pub fn compute_l(&self) -> Mat<E> {
            let mut factor = self.factors.to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_upper(faer_core::zip::Diag::Skip, |mut dst| {
                    dst.write(E::faer_zero())
                });
            factor
        }
        pub fn compute_u(&self) -> Mat<E> {
            let mut factor = self.factors.to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_lower(faer_core::zip::Diag::Skip, |mut dst| {
                    dst.write(E::faer_zero())
                });
            factor.as_mut().diagonal().fill(E::faer_one());
            factor
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
                    faer_lu::partial_pivoting::inverse::invert_req::<E>(
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
                    faer_lu::partial_pivoting::reconstruct::reconstruct_req::<E>(
                        self.dim(),
                        self.dim(),
                        parallelism,
                    )
                    .unwrap(),
                )),
            );

            rec
        }

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
                    faer_lu::partial_pivoting::solve::solve_in_place_req::<E>(
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
                    faer_lu::partial_pivoting::solve::solve_transpose_in_place_req::<E>(
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
                    faer_lu::full_pivoting::compute::lu_in_place_req::<E>(
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
                n_transpositions,
            }
        }

        pub fn row_permutation(&self) -> PermutationRef<'_> {
            unsafe { PermutationRef::new_unchecked(&self.row_perm, &self.row_perm_inv) }
        }
        pub fn col_permutation(&self) -> PermutationRef<'_> {
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
                .for_each_triangular_upper(faer_core::zip::Diag::Skip, |mut dst| {
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
                .for_each_triangular_lower(faer_core::zip::Diag::Skip, |mut dst| {
                    dst.write(E::faer_zero())
                });
            factor.as_mut().diagonal().fill(E::faer_one());
            factor
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
                    faer_lu::full_pivoting::inverse::invert_req::<E>(dim, dim, parallelism)
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
                    faer_lu::full_pivoting::reconstruct::reconstruct_req::<E>(
                        self.nrows(),
                        self.ncols(),
                        parallelism,
                    )
                    .unwrap(),
                )),
            );

            rec
        }

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
                    faer_lu::full_pivoting::solve::solve_in_place_req::<E>(
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
                    faer_lu::full_pivoting::solve::solve_transpose_in_place_req::<E>(
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
                .for_each_triangular_lower(faer_core::zip::Diag::Skip, |mut dst| {
                    dst.write(E::faer_zero())
                });
            factor
        }

        pub fn compute_q(&self) -> Mat<E> {
            Self::__compute_q_impl(self.factors.as_ref(), self.householder.as_ref())
        }

        fn __compute_q_impl(factors: MatRef<'_, E>, householder: MatRef<'_, E>) -> Mat<E> {
            let parallelism = get_global_parallelism();
            let m = factors.nrows();

            let mut q = Mat::<E>::zeros(m, m);
            q.as_mut().diagonal().fill(E::faer_one());

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

    impl<E: ComplexField> SolverLstsqCore<E> for Qr<E> {
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
                    faer_qr::col_pivoting::compute::qr_in_place_req::<E>(
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

        pub fn col_permutation(&self) -> PermutationRef<'_> {
            unsafe { PermutationRef::new_unchecked(&self.col_perm, &self.col_perm_inv) }
        }

        fn blocksize(&self) -> usize {
            self.householder.nrows()
        }

        pub fn compute_r(&self) -> Mat<E> {
            let mut factor = self.factors.to_owned();
            zipped!(factor.as_mut())
                .for_each_triangular_lower(faer_core::zip::Diag::Skip, |mut dst| {
                    dst.write(E::faer_zero())
                });
            factor
        }

        pub fn compute_q(&self) -> Mat<E> {
            Qr::<E>::__compute_q_impl(self.factors.as_ref(), self.householder.as_ref())
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
                    faer_qr::col_pivoting::reconstruct::reconstruct_req::<E>(
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
                    faer_qr::col_pivoting::inverse::invert_req::<E>(
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
                    faer_qr::col_pivoting::solve::solve_transpose_in_place_req::<E>(
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

    impl<E: ComplexField> SolverLstsqCore<E> for ColPivQr<E> {
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
                    faer_qr::col_pivoting::solve::solve_in_place_req::<E>(
                        self.nrows(),
                        self.blocksize(),
                        rhs_ncols,
                    )
                    .unwrap(),
                )),
            );
        }
    }

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
                zipped!(u.as_mut()).for_each(|mut x| x.write(x.read().faer_conj()));
                zipped!(v.as_mut()).for_each(|mut x| x.write(x.read().faer_conj()));
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
            zipped!(rhs.rb_mut().col(j), s).for_each(|mut rhs, s| {
                rhs.write(rhs.read().faer_scale_real(s.read().faer_real().faer_inv()))
            });
        }
    }
    impl<E: ComplexField> SolverCore<E> for Svd<E> {
        fn nrows(&self) -> usize {
            self.u.nrows()
        }

        fn ncols(&self) -> usize {
            self.v.nrows()
        }

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

        fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
            assert!(self.nrows() == self.ncols());
            let mut rhs = rhs;

            let u = self.u.as_ref();
            let v = self.v.as_ref();
            let s = self.s.as_ref();

            match conj {
                Conj::Yes => {
                    rhs.clone_from((u.transpose() * rhs.rb()).as_ref());
                    div_by_s(rhs.rb_mut(), s);
                    rhs.clone_from((v.conjugate() * rhs.rb()).as_ref());
                }
                Conj::No => {
                    rhs.clone_from((u.adjoint() * rhs.rb()).as_ref());
                    div_by_s(rhs.rb_mut(), s);
                    rhs.clone_from((v * rhs.rb()).as_ref());
                }
            }
        }

        fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
            assert!(self.nrows() == self.ncols());
            let mut rhs = rhs;

            let u = self.u.as_ref();
            let v = self.v.as_ref();
            let s = self.s.as_ref();

            match conj {
                Conj::No => {
                    rhs.clone_from((v.transpose() * rhs.rb()).as_ref());
                    div_by_s(rhs.rb_mut(), s);
                    rhs.clone_from((u.conjugate() * rhs.rb()).as_ref());
                }
                Conj::Yes => {
                    rhs.clone_from((v.adjoint() * rhs.rb()).as_ref());
                    div_by_s(rhs.rb_mut(), s);
                    rhs.clone_from((u * rhs.rb()).as_ref());
                }
            }
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
    impl<E: ComplexField> SolverCore<E> for ThinSvd<E> {
        fn nrows(&self) -> usize {
            self.inner.nrows()
        }

        fn ncols(&self) -> usize {
            self.inner.ncols()
        }

        fn reconstruct(&self) -> Mat<E> {
            self.inner.reconstruct()
        }

        fn inverse(&self) -> Mat<E> {
            self.inner.inverse()
        }

        fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
            self.inner.solve_in_place_with_conj_impl(rhs, conj)
        }

        fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
            self.inner
                .solve_transpose_in_place_with_conj_impl(rhs, conj)
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
                zipped!(u.as_mut()).for_each(|mut x| x.write(x.read().faer_conj()));
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
    impl<E: ComplexField> SolverCore<E> for SelfAdjointEigendecomposition<E> {
        fn nrows(&self) -> usize {
            self.u.nrows()
        }

        fn ncols(&self) -> usize {
            self.u.nrows()
        }

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

        fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
            assert!(self.nrows() == self.ncols());
            let mut rhs = rhs;

            let u = self.u.as_ref();
            let s = self.s.as_ref();

            match conj {
                Conj::Yes => {
                    rhs.clone_from((u.transpose() * rhs.rb()).as_ref());
                    div_by_s(rhs.rb_mut(), s);
                    rhs.clone_from((u.conjugate() * rhs.rb()).as_ref());
                }
                Conj::No => {
                    rhs.clone_from((u.adjoint() * rhs.rb()).as_ref());
                    div_by_s(rhs.rb_mut(), s);
                    rhs.clone_from((u * rhs.rb()).as_ref());
                }
            }
        }

        fn solve_transpose_in_place_with_conj_impl(&self, rhs: MatMut<'_, E>, conj: Conj) {
            assert!(self.nrows() == self.ncols());
            let mut rhs = rhs;

            let u = self.u.as_ref();
            let s = self.s.as_ref();

            match conj {
                Conj::No => {
                    rhs.clone_from((u.transpose() * rhs.rb()).as_ref());
                    div_by_s(rhs.rb_mut(), s);
                    rhs.clone_from((u.conjugate() * rhs.rb()).as_ref());
                }
                Conj::Yes => {
                    rhs.clone_from((u.adjoint() * rhs.rb()).as_ref());
                    div_by_s(rhs.rb_mut(), s);
                    rhs.clone_from((u * rhs.rb()).as_ref());
                }
            }
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
                zipped!(s.as_mut()).for_each(|mut x| x.write(x.read().faer_conj()));
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
            let mut s_re = Mat::<E::Real>::zeros(dim, 1);
            let mut s_im = Mat::<E::Real>::zeros(dim, 1);
            let mut u_real = Mat::<E::Real>::zeros(dim, dim);

            let params = Default::default();

            faer_evd::compute_evd_real(
                matrix,
                s_re.as_mut(),
                s_im.as_mut(),
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

            let s = Mat::<E>::from_fn(dim, 1, |i, j| cplx(s_re.read(i, j), s_im.read(i, j)));
            let mut u = Mat::<E>::zeros(dim, dim);
            let u_real = u_real.as_ref();

            let mut j = 0usize;
            while j < dim {
                if s_im.read(j, 0) == E::Real::faer_zero() {
                    zipped!(u.as_mut().col(j), u_real.col(j))
                        .for_each(|mut dst, src| dst.write(E::faer_from_real(src.read())));
                    j += 1;
                } else {
                    let [u_left, u_right] = u.as_mut().split_at_col(j + 1);

                    zipped!(
                        u_left.col(j),
                        u_right.col(0),
                        u_real.col(j),
                        u_real.col(j + 1)
                    )
                    .for_each(|mut dst, mut dst_conj, re, im| {
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

            let mut s = Mat::<E>::zeros(dim, 1);
            let mut u = Mat::<E>::zeros(dim, dim);

            let params = Default::default();

            faer_evd::compute_evd_complex(
                matrix,
                s.as_mut(),
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
                zipped!(s.as_mut()).for_each(|mut x| x.write(x.read().faer_conj()));
                zipped!(u.as_mut()).for_each(|mut x| x.write(x.read().faer_conj()));
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
        pub fn s_diagonal(&self) -> MatRef<'_, E> {
            self.s.as_ref()
        }
    }
}

/// Extension trait for `faer` types.
pub trait Faer<E: ComplexField> {
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

impl<E: Conjugate> Faer<E::Canonical> for MatRef<'_, E>
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

impl<E: Conjugate> Faer<E::Canonical> for MatMut<'_, E>
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

impl<E: Conjugate> Faer<E::Canonical> for Mat<E>
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
                MatRef::<'_, T>::from_raw_parts(
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
                MatMut::<'_, T>::from_raw_parts(
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
            let ptr = self.as_ptr();
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
                MatRef::<'_, c32>::from_raw_parts(
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
                MatMut::<'_, c32>::from_raw_parts(
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
            let ptr = self.as_ptr() as *mut Complex32;
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
                MatRef::<'_, c64>::from_raw_parts(
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
                MatMut::<'_, c64>::from_raw_parts(
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
            let ptr = self.as_ptr() as *mut Complex64;
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
            unsafe { MatRef::<'_, T>::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
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
            unsafe { MatMut::<'_, T>::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
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
            let ptr = self.as_ptr();
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
            unsafe { MatRef::<'_, c32>::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
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
            unsafe { MatMut::<'_, c32>::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
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
            let ptr = self.as_ptr() as *mut Complex32;
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
            unsafe { MatRef::<'_, c64>::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
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
            unsafe { MatMut::<'_, c64>::from_raw_parts(ptr, nrows, ncols, strides[0], strides[1]) }
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
            let ptr = self.as_ptr() as *mut Complex64;
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
                .with_column(sum_horizontal(&[col("*")]))
                .select(&[col("sum")])
                .collect()
                .unwrap()
                .column("sum")
                .unwrap()
                .u32()
                .unwrap()
                .into_iter()
                .map(|e| e.eq(&Some(0u32)))
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
                                    out.as_mut().ptr_at(0, j) as *mut MaybeUninit<$ty>,
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
    use assert2::assert;
    use complex_native::*;
    use faer_core::RealField;

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

        assert_approx_eq(H, decomp.reconstruct());
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
            s.as_mut().diagonal().clone_from(eigen.s_diagonal());
            let u = eigen.u();
            assert_approx_eq(u * &s, &H * u);
        }

        {
            let eigen = H.complex_eigendecomposition();
            let mut s = Mat::zeros(n, n);
            s.as_mut().diagonal().clone_from(eigen.s_diagonal());
            let u = eigen.u();
            assert_approx_eq(u * &s, &H * u);
        }

        let det = H.determinant();
        let eigen_det = H
            .complex_eigenvalues()
            .into_iter()
            .fold(c64::faer_one(), |a, b| a * b);

        dbg!(det, eigen_det);
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
        s.as_mut().diagonal().clone_from(eigen.s_diagonal());
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
}
