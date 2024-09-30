//! `faer` is a general-purpose linear algebra library for Rust, with a focus on high performance
//! for algebraic operations on medium/large matrices, as well as matrix decompositions.
//!
//! Most of the high-level functionality in this library is provided through associated functions in
//! its vocabulary types: [`Mat`]/[`MatRef`]/[`MatMut`].
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
//! More details about the vocabulary types can be found in each one's module's
//! documentation. See also: [`faer_entity::Entity`] and [`complex_native`].
//!
//! Most of the matrix operations can be used through the corresponding math operators: `+` for
//! matrix addition, `-` for subtraction, `*` for either scalar or matrix multiplication depending
//! on the types of the operands.
//!
//! ## Example
//! ```
//! use faer::{mat, scale, Mat};
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
//! [`Mat::cholesky`] decomposes a self-adjoint positive definite matrix $A$ such that
//! $$A = LL^H,$$
//! where $L$ is a lower triangular matrix. This decomposition is highly efficient and has good
//! stability properties.
//!
//! [An implementation for sparse matrices is also available.](sparse::linalg::solvers::Cholesky)
//!
//! ## Bunch-Kaufman decomposition
//! [`Mat::lblt`] decomposes a self-adjoint (possibly indefinite) matrix $A$ such that
//! $$P A P^\top = LBL^H,$$
//! where $P$ is a permutation matrix, $L$ is a lower triangular matrix, and $B$ is a block
//! diagonal matrix, with $1 \times 1$ or $2 \times 2$ diagonal blocks.
//! This decomposition is efficient and has good stability properties.
//! ## LU decomposition with partial pivoting
//! [`Mat::partial_piv_lu`] decomposes a square invertible matrix $A$ into a lower triangular
//! matrix $L$, a unit upper triangular matrix $U$, and a permutation matrix $P$, such that
//! $$PA = LU.$$
//! It is used by default for computing the determinant, and is generally the recommended method
//! for solving a square linear system or computing the inverse of a matrix (although we generally
//! recommend using a [`faer::linalg::solvers::Solver`](crate::linalg::solvers::Solver) instead of
//! computing the inverse explicitly).
//!
//! [An implementation for sparse matrices is also available.](sparse::linalg::solvers::Lu)
//!
//! ## LU decomposition with full pivoting
//! [`Mat::full_piv_lu`] Decomposes a generic rectangular matrix $A$ into a lower triangular
//! matrix $L$, a unit upper triangular matrix $U$, and permutation matrices $P$ and $Q$, such that
//! $$PAQ^\top = LU.$$
//! It can be more stable than the LU decomposition with partial pivoting, in exchange for being
//! more computationally expensive.
//!
//! ## QR decomposition
//! The QR decomposition ([`Mat::qr`]) decomposes a matrix $A$ into the product
//! $$A = QR,$$
//! where $Q$ is a unitary matrix, and $R$ is an upper trapezoidal matrix. It is often used for
//! solving least squares problems.
//!
//! [An implementation for sparse matrices is also available.](sparse::linalg::solvers::Qr)
//!
//! ## QR decomposition with column pivoting
//! The QR decomposition with column pivoting ([`Mat::col_piv_qr`]) decomposes a matrix $A$ into
//! the product
//! $$AP^\top = QR,$$
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
//! [`Mat::svd`], or only their first $\min(m, n)$ columns are computed, using
//! [`Mat::thin_svd`].
//!
//! If only the singular values (elements of $S$) are desired, they can be obtained in
//! nonincreasing order using [`Mat::singular_values`].
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
//! * [`Mat::selfadjoint_eigendecomposition`] can be used with either real or complex matrices,
//! producing an eigendecomposition of the same type.
//! * [`Mat::eigendecomposition`] can be used with either real or complex matrices, but the output
//! complex type has to be specified.
//! * [`Mat::complex_eigendecomposition`] can only be used with complex matrices, with the output
//! having the same type.
//!
//! If only the eigenvalues (elements of $S$) are desired, they can be obtained in
//! nonincreasing order using [`Mat::selfadjoint_eigenvalues`], [`Mat::eigenvalues`], or
//! [`Mat::complex_eigenvalues`], with the same conditions described above.
//!
//! # Crate features
//!
//! - `std`: enabled by default. Links with the standard library to enable additional features such
//!   as cpu feature detection at runtime.
//! - `rayon`: enabled by default. Enables the `rayon` parallel backend and enables global
//!   parallelism by default.
//! - `serde`: Enables serialization and deserialization of [`Mat`].
//! - `npy`: Enables conversions to/from numpy's matrix file format.
//! - `perf-warn`: Produces performance warnings when matrix operations are called with suboptimal
//! data layout.
//! - `nightly`: Requires the nightly compiler. Enables experimental SIMD features such as AVX512.

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(non_snake_case)]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

macro_rules! map {
    ($ty: ty, $input: expr, |($x: pat)| $f: expr $(,)?) => {
        <$ty as ::faer_entity::Entity>::faer_map(
            $input,
            #[inline(always)]
            |$x| $f,
        )
    };
}

macro_rules! __dbg {
    ($E: ty, $(,)?) => {{
            extern crate std as __std;
            __std::eprintln!("[{}:{}:{}]", __std::file!(), __std::line!(), __std::column!())
    }};
    ($E: ty, $val:expr $(,)?) => {
        match $val {
            tmp => {{
                extern crate std as __std;
                E::faer_map(E::faer_as_ref(&tmp), |tmp| __std::eprintln!(
                    __std::concat!("[{}:{}:{}] {} = {:#?}"),
                    __std::file!(),
                    __std::line!(),
                    __std::column!(),
                    __std::stringify!($val),
                    &tmp,
                ));
                tmp
            }}
        }
    };
    ($E: ty, $($val:expr),+ $(,)?) => {
        ($(__dbg!($E, $val)),+,)
    };
}
use core::sync::atomic::AtomicUsize;
use equator::{assert, debug_assert};

extern crate alloc;

macro_rules! stack_mat {
    ([$max_nrows: expr, $max_ncols: expr$(,)?], $name: ident, $nrows: expr, $ncols: expr, $rs: expr, $cs: expr, $ty: ty) => {
        let __nrows: usize = $nrows;
        let __ncols: usize = $ncols;
        $crate::assert!(all(__nrows <= $max_nrows, __ncols <= $max_ncols));
        let __rs: isize = $rs;
        let __cs: isize = $cs;
        let mut __data = {
            #[repr(align(128))]
            struct Wrapper<E: $crate::Entity>(faer_entity::GroupFor<E, [[::core::mem::MaybeUninit<<$ty as $crate::Entity>::Unit>; $max_nrows]; $max_ncols]>);

            Wrapper::<$ty>(<$ty as $crate::Entity>::faer_map(
                <$ty as $crate::Entity>::UNIT,
                #[inline(always)]
                |()| unsafe {
                    $crate::linalg::entity::transmute_unchecked::<
                        ::core::mem::MaybeUninit<[[<$ty as $crate::Entity>::Unit; $max_nrows]; $max_ncols]>,
                        [[::core::mem::MaybeUninit<<$ty as $crate::Entity>::Unit>; $max_nrows]; $max_ncols],
                    >(::core::mem::MaybeUninit::<
                        [[<$ty as $crate::Entity>::Unit; $max_nrows]; $max_ncols],
                    >::uninit())
                },
            ))
        };

        <$ty as $crate::Entity>::faer_map(
            <$ty as $crate::Entity>::faer_zip(
                <$ty as $crate::Entity>::faer_as_mut(&mut __data.0),
                <$ty as $crate::Entity>::faer_into_units(<$ty as $crate::ComplexField>::faer_zero()),
            ),
            #[inline(always)]
            |(__data, zero)| {
                let __data: &mut _ = __data;
                for j in 0..__ncols {
                    __data[j].fill(::core::mem::MaybeUninit::new(zero));
                }
            },
        );
        let mut __data =
            <$ty as $crate::Entity>::faer_map(<$ty as $crate::Entity>::faer_as_mut(&mut __data.0), |__data: &mut _| {
                (__data as *mut [[::core::mem::MaybeUninit<<$ty as $crate::Entity>::Unit>; $max_nrows]; $max_ncols]
                    as *mut <$ty as $crate::Entity>::Unit)
            });

        let mut $name = unsafe {
            $crate::mat::from_raw_parts_mut::<'_, $ty, _, _>(__data, __nrows, __ncols, 1isize, $max_nrows as isize)
        };

        if __cs.unsigned_abs() < __rs.unsigned_abs() {
            $crate::assert!(__nrows == __ncols);
            $name = $name.transpose_mut();
        }
        if __rs == -1 {
            $name = $name.reverse_rows_mut();
        }
        if __cs == -1 {
            $name = $name.reverse_cols_mut();
        }
    };
}

pub mod linalg;

pub mod complex_native;

/// Similar to the [`dbg`] macro, but takes a format spec as a first parameter.
pub use dbgf::dbgf;
pub use dyn_stack;
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub use matrixcompare::assert_matrix_eq;
pub use reborrow;

/// Various utilities for low level implementations in generic code.
pub mod utils;

/// Iterators and related utilities.
pub mod iter;

/// Column vector type.
pub mod col;
/// Diagonal matrix type.
pub mod diag;
/// Matrix-free linear operator traits and algorithms.
#[cfg(feature = "unstable")]
#[cfg_attr(docsrs, doc(cfg(feature = "unstable")))]
pub mod linop;
/// Matrix type.
pub mod mat;
/// Permutation matrices.
pub mod perm;
/// Row vector type.
pub mod row;
/// Sparse data structures and algorithms.
pub mod sparse;

pub use col::{Col, ColMut, ColRef};
pub use mat::{Mat, MatMut, MatRef};
pub use row::{Row, RowMut, RowRef};

mod seal;
mod sort;

pub use faer_entity::{ComplexField, Conjugate, Entity, RealField, SimpleEntity};

/// Specifies whether the triangular lower or upper part of a matrix should be accessed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Side {
    /// Lower half should be accessed.
    Lower,
    /// Upper half should be accessed.
    Upper,
}

/// Whether a matrix should be implicitly conjugated when read or not.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Conj {
    /// Do conjugate.
    Yes,
    /// Do not conjugate.
    No,
}

impl Conj {
    /// Combine `self` and `other` to create a new conjugation object.
    #[inline]
    pub fn compose(self, other: Conj) -> Conj {
        if self == other {
            Conj::No
        } else {
            Conj::Yes
        }
    }
}

/// Zips together matrix of the same size, so that coefficient-wise operations can be performed on
/// their elements.
///
/// # Note
/// The order in which the matrix elements are traversed is unspecified.
///
/// # Example
/// ```
/// use faer::{mat, unzipped, zipped, Mat};
///
/// let nrows = 2;
/// let ncols = 3;
///
/// let a = mat![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
/// let b = mat![[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]];
/// let mut sum = Mat::<f64>::zeros(nrows, ncols);
///
/// zipped!(sum.as_mut(), a.as_ref(), b.as_ref()).for_each(|unzipped!(mut sum, a, b)| {
///     let a = a.read();
///     let b = b.read();
///     sum.write(a + b);
/// });
///
/// for i in 0..nrows {
///     for j in 0..ncols {
///         assert_eq!(sum.read(i, j), a.read(i, j) + b.read(i, j));
///     }
/// }
/// ```
#[macro_export]
macro_rules! zipped {
    ($head: expr $(,)?) => {
        $crate::linalg::zip::LastEq($crate::linalg::zip::ViewMut::view_mut(&mut { $head }))
    };

    ($head: expr, $($tail: expr),* $(,)?) => {
        $crate::linalg::zip::ZipEq::new($crate::linalg::zip::ViewMut::view_mut(&mut { $head }), $crate::zipped!($($tail,)*))
    };
}

/// Used to undo the zipping by the [`zipped!`] macro.
///
/// # Example
/// ```
/// use faer::{mat, unzipped, zipped, Mat};
///
/// let nrows = 2;
/// let ncols = 3;
///
/// let a = mat![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
/// let b = mat![[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]];
/// let mut sum = Mat::<f64>::zeros(nrows, ncols);
///
/// zipped!(sum.as_mut(), a.as_ref(), b.as_ref()).for_each(|unzipped!(mut sum, a, b)| {
///     let a = a.read();
///     let b = b.read();
///     sum.write(a + b);
/// });
///
/// for i in 0..nrows {
///     for j in 0..ncols {
///         assert_eq!(sum.read(i, j), a.read(i, j) + b.read(i, j));
///     }
/// }
/// ```
#[macro_export]
macro_rules! unzipped {
    ($head: pat $(,)?) => {
        $crate::linalg::zip::Last($head)
    };

    ($head: pat, $($tail: pat),* $(,)?) => {
        $crate::linalg::zip::Zip($head, $crate::unzipped!($($tail,)*))
    };
}

#[doc(hidden)]
#[inline(always)]
pub fn ref_to_ptr<T>(ptr: &T) -> *const T {
    ptr
}

#[macro_export]
#[doc(hidden)]
macro_rules! __transpose_impl {
    ([$([$($col:expr),*])*] $($v:expr;)* ) => {
        [$([$($col,)*],)* [$($v,)*]]
    };
    ([$([$($col:expr),*])*] $($v0:expr, $($v:expr),* ;)*) => {
        $crate::__transpose_impl!([$([$($col),*])* [$($v0),*]] $($($v),* ;)*)
    };
}

/// Creates a [`Mat`] containing the arguments.
///
/// ```
/// use faer::mat;
///
/// let matrix = mat![
///     [1.0, 5.0, 9.0],
///     [2.0, 6.0, 10.0],
///     [3.0, 7.0, 11.0],
///     [4.0, 8.0, 12.0f64],
/// ];
///
/// assert_eq!(matrix.read(0, 0), 1.0);
/// assert_eq!(matrix.read(1, 0), 2.0);
/// assert_eq!(matrix.read(2, 0), 3.0);
/// assert_eq!(matrix.read(3, 0), 4.0);
///
/// assert_eq!(matrix.read(0, 1), 5.0);
/// assert_eq!(matrix.read(1, 1), 6.0);
/// assert_eq!(matrix.read(2, 1), 7.0);
/// assert_eq!(matrix.read(3, 1), 8.0);
///
/// assert_eq!(matrix.read(0, 2), 9.0);
/// assert_eq!(matrix.read(1, 2), 10.0);
/// assert_eq!(matrix.read(2, 2), 11.0);
/// assert_eq!(matrix.read(3, 2), 12.0);
/// ```
#[macro_export]
macro_rules! mat {
    () => {
        {
            compile_error!("number of columns in the matrix is ambiguous");
        }
    };

    ($([$($v:expr),* $(,)?] ),* $(,)?) => {
        {
            let data = ::core::mem::ManuallyDrop::new($crate::__transpose_impl!([] $($($v),* ;)*));
            let data = &*data;
            let ncols = data.len();
            let nrows = (*data.get(0).unwrap()).len();

            #[allow(unused_unsafe)]
            unsafe {
                $crate::mat::Mat::<_>::from_fn(nrows, ncols, |i, j| $crate::ref_to_ptr(&data[j][i]).read())
            }
        }
    };
}

#[cfg(feature = "perf-warn")]
#[macro_export]
#[doc(hidden)]
macro_rules! __perf_warn {
    ($name: ident) => {{
        #[inline(always)]
        #[allow(non_snake_case)]
        fn $name() -> &'static ::core::sync::atomic::AtomicBool {
            static $name: ::core::sync::atomic::AtomicBool =
                ::core::sync::atomic::AtomicBool::new(false);
            &$name
        }
        ::core::matches!(
            $name().compare_exchange(
                false,
                true,
                ::core::sync::atomic::Ordering::Relaxed,
                ::core::sync::atomic::Ordering::Relaxed,
            ),
            Ok(_)
        )
    }};
}

/// Convenience function to concatenate a nested list of matrices into a single
/// big ['Mat']. Concatonation pattern follows the numpy.block convention that
/// each sub-list must have an equal number of columns (net) but the boundaries
/// do not need to align. In other words, this sort of thing:
/// ```notcode
///   AAAbb
///   AAAbb
///   cDDDD
/// ```
/// is perfectly acceptable.
#[doc(hidden)]
#[track_caller]
pub fn concat_impl<E: ComplexField>(blocks: &[&[(mat::MatRef<'_, E>, Conj)]]) -> mat::Mat<E> {
    #[inline(always)]
    fn count_total_columns<E: ComplexField>(block_row: &[(mat::MatRef<'_, E>, Conj)]) -> usize {
        let mut out: usize = 0;
        for (elem, _) in block_row.iter() {
            out += elem.ncols();
        }
        out
    }

    #[inline(always)]
    #[track_caller]
    fn count_rows<E: ComplexField>(block_row: &[(mat::MatRef<'_, E>, Conj)]) -> usize {
        let mut out: usize = 0;
        for (i, (e, _)) in block_row.iter().enumerate() {
            if i == 0 {
                out = e.nrows();
            } else {
                assert!(e.nrows() == out);
            }
        }
        out
    }

    // get size of result while doing checks
    let mut n: usize = 0;
    let mut m: usize = 0;
    for row in blocks.iter() {
        n += count_rows(row);
    }
    for (i, row) in blocks.iter().enumerate() {
        let cols = count_total_columns(row);
        if i == 0 {
            m = cols;
        } else {
            assert!(cols == m);
        }
    }

    let mut mat = mat::Mat::<E>::zeros(n, m);
    let mut ni: usize = 0;
    let mut mj: usize;
    for row in blocks.iter() {
        mj = 0;

        for (elem, conj) in row.iter() {
            let mut dst = mat
                .as_mut()
                .submatrix_mut(ni, mj, elem.nrows(), elem.ncols());
            if *conj == Conj::No {
                dst.copy_from(elem);
            } else {
                dst.copy_from(elem.conjugate());
            }
            mj += elem.ncols();
        }
        ni += row[0].0.nrows();
    }

    mat
}

/// Concatenates the matrices in each row horizontally,
/// then concatenates the results vertically.
///
/// `concat![[a0, a1, a2], [b1, b2]]` results in the matrix
/// ```notcode
/// [a0 | a1 | a2][b0 | b1]
/// ```
#[macro_export]
macro_rules! concat {
    () => {
        {
            compile_error!("number of columns in the matrix is ambiguous");
        }
    };

    ($([$($v:expr),* $(,)?] ),* $(,)?) => {
        {
            $crate::concat_impl(&[$(&[$(($v).as_ref().canonicalize(),)*],)*])
        }
    };
}

/// Creates a [`col::Col`] containing the arguments.
///
/// ```
/// use faer::col;
///
/// let col_vec = col![3.0, 5.0, 7.0, 9.0];
///
/// assert_eq!(col_vec.read(0), 3.0);
/// assert_eq!(col_vec.read(1), 5.0);
/// assert_eq!(col_vec.read(2), 7.0);
/// assert_eq!(col_vec.read(3), 9.0);
/// ```
#[macro_export]
macro_rules! col {
    () => {
        $crate::col::Col::<_>::new()
    };

    ($($v:expr),+ $(,)?) => {{
        let data = &[$($v),+];
        let n = data.len();

        #[allow(unused_unsafe)]
        unsafe {
            $crate::col::Col::<_>::from_fn(n, |i| $crate::ref_to_ptr(&data[i]).read())
        }
    }};
}

/// Creates a [`row::Row`] containing the arguments.
///
/// ```
/// use faer::row;
///
/// let row_vec = row![3.0, 5.0, 7.0, 9.0];
///
/// assert_eq!(row_vec.read(0), 3.0);
/// assert_eq!(row_vec.read(1), 5.0);
/// assert_eq!(row_vec.read(2), 7.0);
/// assert_eq!(row_vec.read(3), 9.0);
/// ```
#[macro_export]
macro_rules! row {
    () => {
        $crate::row::Row::<_>::new()
    };

    ($($v:expr),+ $(,)?) => {{
        let data = &[$($v),+];
        let n = data.len();

        #[allow(unused_unsafe)]
        unsafe {
            $crate::row::Row::<_>::from_fn(n, |i| $crate::ref_to_ptr(&data[i]).read())
        }
    }};
}

/// Trait for unsigned integers that can be indexed with.
///
/// Always smaller than or equal to `usize`.
pub trait Index:
    seal::Seal
    + core::fmt::Debug
    + core::ops::Not<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + bytemuck::Pod
    + Eq
    + Ord
    + Send
    + Sync
    + Ord
{
    /// Equally-sized index type with a fixed size (no `usize`).
    type FixedWidth: Index;
    /// Equally-sized signed index type.
    type Signed: SignedIndex;

    /// Truncate `value` to type [`Self`].
    #[must_use]
    #[inline(always)]
    fn truncate(value: usize) -> Self {
        Self::from_signed(<Self::Signed as SignedIndex>::truncate(value))
    }

    /// Zero extend `self`.
    #[must_use]
    #[inline(always)]
    fn zx(self) -> usize {
        self.to_signed().zx()
    }

    /// Convert a reference to a slice of [`Self`] to fixed width types.
    #[inline(always)]
    fn canonicalize(slice: &[Self]) -> &[Self::FixedWidth] {
        bytemuck::cast_slice(slice)
    }

    /// Convert a mutable reference to a slice of [`Self`] to fixed width types.
    #[inline(always)]
    fn canonicalize_mut(slice: &mut [Self]) -> &mut [Self::FixedWidth] {
        bytemuck::cast_slice_mut(slice)
    }

    /// Convert a signed value to an unsigned one.
    #[inline(always)]
    fn from_signed(value: Self::Signed) -> Self {
        bytemuck::cast(value)
    }

    /// Convert an unsigned value to a signed one.
    #[inline(always)]
    fn to_signed(self) -> Self::Signed {
        bytemuck::cast(self)
    }

    /// Sum values while checking for overflow.
    #[inline]
    fn sum_nonnegative(slice: &[Self]) -> Option<Self> {
        Self::Signed::sum_nonnegative(bytemuck::cast_slice(slice)).map(Self::from_signed)
    }
}

/// Trait for signed integers corresponding to the ones satisfying [`Index`].
///
/// Always smaller than or equal to `isize`.
pub trait SignedIndex:
    seal::Seal
    + core::fmt::Debug
    + core::ops::Neg<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + bytemuck::Pod
    + Eq
    + Ord
    + Send
    + Sync
{
    /// Maximum representable value.
    const MAX: Self;

    /// Truncate `value` to type [`Self`].
    #[must_use]
    fn truncate(value: usize) -> Self;

    /// Zero extend `self`.
    #[must_use]
    fn zx(self) -> usize;
    /// Sign extend `self`.
    #[must_use]
    fn sx(self) -> usize;

    /// Sum nonnegative values while checking for overflow.
    fn sum_nonnegative(slice: &[Self]) -> Option<Self> {
        let mut acc = Self::zeroed();
        for &i in slice {
            if Self::MAX - i < acc {
                return None;
            }
            acc += i;
        }
        Some(acc)
    }
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64",))]
impl Index for u32 {
    type FixedWidth = u32;
    type Signed = i32;
}
#[cfg(any(target_pointer_width = "64"))]
impl Index for u64 {
    type FixedWidth = u64;
    type Signed = i64;
}
impl Index for usize {
    #[cfg(target_pointer_width = "32")]
    type FixedWidth = u32;
    #[cfg(target_pointer_width = "64")]
    type FixedWidth = u64;

    type Signed = isize;
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64",))]
impl SignedIndex for i32 {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const _: () = {
            core::assert!(i32::BITS <= usize::BITS);
        };
        value as isize as Self
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as u32 as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as isize as usize
    }
}

#[cfg(any(target_pointer_width = "64"))]
impl SignedIndex for i64 {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const _: () = {
            core::assert!(i64::BITS <= usize::BITS);
        };
        value as isize as Self
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as u64 as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as isize as usize
    }
}

impl SignedIndex for isize {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        value as isize
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as usize
    }
}

/// Factor for matrix-scalar multiplication.
#[derive(Copy, Clone, Debug)]
pub struct Scale<E>(pub E);

impl<E> Scale<E> {
    /// Returns the inner value.
    #[inline]
    pub fn value(self) -> E {
        self.0
    }
}

/// Returns a factor for matrix-scalar multiplication.
#[inline]
pub fn scale<E>(val: E) -> Scale<E> {
    Scale(val)
}

/// Parallelism strategy that can be passed to most of the routines in the library.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Parallelism<'a> {
    /// No parallelism.
    ///
    /// The code is executed sequentially on the same thread that calls a function
    /// and passes this argument.
    None,
    /// Rayon parallelism. Only available with the `rayon` feature.
    ///
    /// The code is possibly executed in parallel on the current thread, as well as the currently
    /// active rayon thread pool.
    ///
    /// The contained value represents a hint about the number of threads an implementation should
    /// use, but there is no way to guarantee how many or which threads will be used.
    ///
    /// A value of `0` treated as equivalent to `rayon::current_num_threads()`.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    Rayon(usize),

    #[doc(hidden)]
    __Private(&'a ()),
}

/// 0: Disable
/// 1: None
/// n >= 2: Rayon(n - 2)
///
/// default: Rayon(0)
static GLOBAL_PARALLELISM: AtomicUsize = {
    #[cfg(feature = "rayon")]
    {
        AtomicUsize::new(2)
    }
    #[cfg(not(feature = "rayon"))]
    {
        AtomicUsize::new(1)
    }
};

/// Causes functions that access global parallelism settings to panic.
pub fn disable_global_parallelism() {
    GLOBAL_PARALLELISM.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// Sets the global parallelism settings.
pub fn set_global_parallelism(parallelism: Parallelism) {
    let value = match parallelism {
        Parallelism::None => 1,
        #[cfg(feature = "rayon")]
        Parallelism::Rayon(n) => n.saturating_add(2),
        Parallelism::__Private(_) => panic!(),
    };
    GLOBAL_PARALLELISM.store(value, core::sync::atomic::Ordering::Relaxed);
}

/// Gets the global parallelism settings.
///
/// # Panics
/// Panics if global parallelism is disabled.
#[track_caller]
pub fn get_global_parallelism() -> Parallelism<'static> {
    let value = GLOBAL_PARALLELISM.load(core::sync::atomic::Ordering::Relaxed);
    match value {
        0 => panic!("Global parallelism is disabled."),
        1 => Parallelism::None,
        #[cfg(feature = "rayon")]
        n => Parallelism::Rayon(n - 2),
        #[cfg(not(feature = "rayon"))]
        _ => unreachable!(),
    }
}

/// De-serialization from common matrix file formats.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod io;

#[cfg(feature = "serde")]
mod serde;

/// faer prelude. Includes useful types and traits for solving linear systems.
pub mod prelude {
    pub use crate::{
        col,
        complex_native::{c32, c64},
        linalg::solvers::{
            Solver, SolverCore, SolverLstsq, SolverLstsqCore, SpSolver, SpSolverCore,
            SpSolverLstsq, SpSolverLstsqCore,
        },
        mat, row, unzipped, zipped, Col, ColMut, ColRef, Mat, MatMut, MatRef, Row, RowMut, RowRef,
    };
}

/// Matrix solvers and decompositions.
#[deprecated = "moved to faer::linalg::solvers"]
pub mod solvers {
    pub use crate::linalg::solvers::*;
}

/// Statistics-related utilities.
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub mod stats;

/// Re-exports.
#[deprecated]
pub mod modules {
    /// Emulation layer for `faer_core`
    #[deprecated]
    pub mod core {
        #[deprecated = "moved to faer::col"]
        pub use crate::col;

        #[deprecated = "moved to faer::complex_native"]
        pub use crate::complex_native;

        #[deprecated = "moved to faer::utils::constrained"]
        pub use crate::utils::constrained;

        #[deprecated = "moved to faer::utils::{slice, simd, vec}"]
        pub use crate::utils as group_helpers;

        #[deprecated = "moved to faer::linalg::householder"]
        pub use crate::linalg::householder;

        #[deprecated = "moved to faer::linalg::triangular_inverse"]
        pub use crate::linalg::triangular_inverse as inverse;

        #[deprecated = "moved to faer::linalg::triangular_solve"]
        pub use crate::linalg::triangular_inverse as solve;

        #[deprecated = "moved to faer::mat"]
        pub use crate::mat;

        #[deprecated = "moved to faer::linalg::matmul"]
        pub use crate::linalg::matmul as mul;

        #[deprecated = "moved to faer::perm"]
        pub use crate::perm as permutation;

        #[deprecated = "moved to faer::row"]
        pub use crate::row;

        #[deprecated = "moved to faer::sparse"]
        pub use crate::sparse;

        #[deprecated = "moved to faer::linalg::zip"]
        pub use crate::linalg::zip;

        #[deprecated = "moved to faer::zipped"]
        pub use crate::zipped;

        #[deprecated = "moved to faer::unzipped"]
        pub use crate::unzipped;

        #[deprecated = "moved to faer::linalg::entity::IdentityGroup"]
        pub use crate::linalg::entity::IdentityGroup;

        #[deprecated = "moved to faer::Conj"]
        pub use crate::Conj;

        #[deprecated = "moved to faer::sparse::FaerError"]
        pub use crate::sparse::FaerError;

        #[deprecated = "moved to faer::Parallelism"]
        pub use crate::Parallelism;

        #[deprecated = "moved to faer::Side"]
        pub use crate::Side;

        #[deprecated = "moved to faer::col::AsColRef"]
        pub use crate::col::AsColRef;

        #[deprecated = "moved to faer::col::AsColMut"]
        pub use crate::col::AsColMut;

        #[deprecated = "moved to faer::row::AsRowRef"]
        pub use crate::row::AsRowRef;

        #[deprecated = "moved to faer::row::AsRowMut"]
        pub use crate::row::AsRowMut;

        #[deprecated = "moved to faer::mat::AsMatRef"]
        pub use crate::mat::AsMatRef;

        #[deprecated = "moved to faer::mat::AsMatMut"]
        pub use crate::mat::AsMatMut;

        #[deprecated = "moved to faer::mat::As2D"]
        pub use crate::mat::As2D;

        #[deprecated = "moved to faer::mat::As2DMut"]
        pub use crate::mat::As2DMut;

        #[deprecated = "moved to faer::ComplexField"]
        pub use crate::ComplexField;

        #[deprecated = "moved to faer::Conjugate"]
        pub use crate::Conjugate;

        #[deprecated = "moved to faer::RealField"]
        pub use crate::RealField;

        #[deprecated = "moved to faer::Entity"]
        pub use crate::Entity;

        #[deprecated = "moved to faer::SimpleEntity"]
        pub use crate::SimpleEntity;

        #[deprecated = "kronk, pull the lever"]
        pub use crate::linalg::kron;

        #[deprecated = "moved to faer::disable_global_parallelism"]
        pub use crate::disable_global_parallelism;

        #[deprecated = "moved to faer::set_global_parallelism"]
        pub use crate::set_global_parallelism;

        #[deprecated = "moved to faer::get_global_parallelism"]
        pub use crate::get_global_parallelism;

        #[deprecated = "moved to faer::linalg::temp_mat_req"]
        pub use crate::linalg::temp_mat_req;

        #[deprecated = "moved to faer::linalg::temp_mat_uninit"]
        pub use crate::linalg::temp_mat_uninit;

        #[deprecated = "moved to faer::linalg::temp_mat_constant"]
        pub use crate::linalg::temp_mat_constant;

        #[deprecated = "moved to faer::linalg::temp_mat_zeroed"]
        pub use crate::linalg::temp_mat_zeroed;

        #[deprecated = "moved to faer::Col"]
        pub use crate::Col;
        #[deprecated = "moved to faer::ColMut"]
        pub use crate::ColMut;
        #[deprecated = "moved to faer::ColRef"]
        pub use crate::ColRef;

        #[deprecated = "moved to faer::Row"]
        pub use crate::Row;
        #[deprecated = "moved to faer::RowMut"]
        pub use crate::RowMut;
        #[deprecated = "moved to faer::RowRef"]
        pub use crate::RowRef;

        #[deprecated = "moved to faer::Mat"]
        pub use crate::Mat;
        #[deprecated = "moved to faer::MatMut"]
        pub use crate::MatMut;
        #[deprecated = "moved to faer::MatRef"]
        pub use crate::MatRef;

        #[deprecated = "moved to faer::MatScale"]
        pub use crate::Scale;

        #[deprecated = "moved to faer::scale"]
        pub use crate::scale;
    }

    #[deprecated = "moved to faer::cholesky"]
    pub use crate::linalg::cholesky;
    #[deprecated = "moved to faer::evd"]
    pub use crate::linalg::evd;
    #[deprecated = "moved to faer::lu"]
    pub use crate::linalg::lu;
    #[deprecated = "moved to faer::qr"]
    pub use crate::linalg::qr;
    #[deprecated = "moved to faer::svd"]
    pub use crate::linalg::svd;
    #[deprecated = "moved to faer::sparse::linalg"]
    pub use crate::sparse::linalg as sparse;
}

#[cfg(test)]
pub(crate) use tests::ApproxEq;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert;

    #[test]
    fn basic_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let slice = unsafe { mat::from_raw_parts::<'_, f64, _, _>(data.as_ptr(), 2, 3, 3, 1) };

        assert!(slice.get(0, 0) == &1.0);
        assert!(slice.get(0, 1) == &2.0);
        assert!(slice.get(0, 2) == &3.0);

        assert!(slice.get(1, 0) == &4.0);
        assert!(slice.get(1, 1) == &5.0);
        assert!(slice.get(1, 2) == &6.0);
    }

    #[test]
    fn empty() {
        {
            let m = Mat::<f64>::new();
            assert!(m.nrows() == 0);
            assert!(m.ncols() == 0);
            assert!(m.row_capacity() == 0);
            assert!(m.col_capacity() == 0);
        }

        {
            let m = Mat::<f64>::with_capacity(100, 120);
            assert!(m.nrows() == 0);
            assert!(m.ncols() == 0);
            assert!(m.row_capacity() == 100);
            assert!(m.col_capacity() == 120);
        }
    }

    #[test]
    fn reserve() {
        let mut m = Mat::<f64>::new();

        m.reserve_exact(0, 0);
        assert!(m.row_capacity() == 0);
        assert!(m.col_capacity() == 0);

        m.reserve_exact(1, 1);
        assert!(m.row_capacity() >= 1);
        assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 0);
        assert!(m.row_capacity() >= 2);
        assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 3);
        assert!(m.row_capacity() >= 2);
        assert!(m.col_capacity() == 3);
    }

    #[test]
    fn reserve_zst() {
        let mut m = Mat::<faer_entity::Symbolic>::new();

        m.reserve_exact(0, 0);
        assert!(m.row_capacity() == 0);
        assert!(m.col_capacity() == 0);

        m.reserve_exact(1, 1);
        assert!(m.row_capacity() == 1);
        assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 0);
        assert!(m.row_capacity() == 2);
        assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 3);
        assert!(m.row_capacity() == 2);
        assert!(m.col_capacity() == 3);

        m.reserve_exact(usize::MAX, usize::MAX);
    }

    #[test]
    fn resize() {
        let mut m = Mat::new();
        let f = |i, j| i as f64 - j as f64;
        m.resize_with(2, 3, f);
        assert!(m.read(0, 0) == 0.0);
        assert!(m.read(0, 1) == -1.0);
        assert!(m.read(0, 2) == -2.0);
        assert!(m.read(1, 0) == 1.0);
        assert!(m.read(1, 1) == 0.0);
        assert!(m.read(1, 2) == -1.0);

        m.resize_with(1, 2, f);
        assert!(m.read(0, 0) == 0.0);
        assert!(m.read(0, 1) == -1.0);

        m.resize_with(2, 1, f);
        assert!(m.read(0, 0) == 0.0);
        assert!(m.read(1, 0) == 1.0);

        m.resize_with(1, 2, f);
        assert!(m.read(0, 0) == 0.0);
        assert!(m.read(0, 1) == -1.0);
    }

    #[test]
    fn resize_zst() {
        // miri test
        let mut m = Mat::new();
        let f = |_i, _j| faer_entity::Symbolic;
        m.resize_with(2, 3, f);
        m.resize_with(1, 2, f);
        m.resize_with(2, 1, f);
        m.resize_with(1, 2, f);
    }

    #[test]
    #[should_panic]
    fn cap_overflow_1() {
        let _ = Mat::<f64>::with_capacity(isize::MAX as usize, 1);
    }

    #[test]
    #[should_panic]
    fn cap_overflow_2() {
        let _ = Mat::<f64>::with_capacity(isize::MAX as usize, isize::MAX as usize);
    }

    #[test]
    fn matrix_macro() {
        let mut x = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        assert!(x[(0, 0)] == 1.0);
        assert!(x[(0, 1)] == 2.0);
        assert!(x[(0, 2)] == 3.0);

        assert!(x[(1, 0)] == 4.0);
        assert!(x[(1, 1)] == 5.0);
        assert!(x[(1, 2)] == 6.0);

        assert!(x[(2, 0)] == 7.0);
        assert!(x[(2, 1)] == 8.0);
        assert!(x[(2, 2)] == 9.0);

        x[(0, 0)] = 13.0;
        assert!(x[(0, 0)] == 13.0);

        assert!(x.get(.., ..) == x);
        assert!(x.get(.., 1..3) == x.as_ref().submatrix(0, 1, 3, 2));
    }

    #[test]
    fn matrix_macro_cplx() {
        use num_complex::Complex;
        let new = Complex::new;
        let mut x = mat![
            [new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0)],
            [new(7.0, 8.0), new(9.0, 10.0), new(11.0, 12.0)],
            [new(13.0, 14.0), new(15.0, 16.0), new(17.0, 18.0)]
        ];

        assert!(x.read(0, 0) == Complex::new(1.0, 2.0));
        assert!(x.read(0, 1) == Complex::new(3.0, 4.0));
        assert!(x.read(0, 2) == Complex::new(5.0, 6.0));

        assert!(x.read(1, 0) == Complex::new(7.0, 8.0));
        assert!(x.read(1, 1) == Complex::new(9.0, 10.0));
        assert!(x.read(1, 2) == Complex::new(11.0, 12.0));

        assert!(x.read(2, 0) == Complex::new(13.0, 14.0));
        assert!(x.read(2, 1) == Complex::new(15.0, 16.0));
        assert!(x.read(2, 2) == Complex::new(17.0, 18.0));

        x.write(1, 0, Complex::new(3.0, 2.0));
        assert!(x.read(1, 0) == Complex::new(3.0, 2.0));
    }

    #[test]
    fn matrix_macro_native_cplx() {
        use complex_native::c64 as Complex;

        let new = Complex::new;
        let mut x = mat![
            [new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0)],
            [new(7.0, 8.0), new(9.0, 10.0), new(11.0, 12.0)],
            [new(13.0, 14.0), new(15.0, 16.0), new(17.0, 18.0)]
        ];

        assert!(x.read(0, 0) == Complex::new(1.0, 2.0));
        assert!(x.read(0, 1) == Complex::new(3.0, 4.0));
        assert!(x.read(0, 2) == Complex::new(5.0, 6.0));

        assert!(x.read(1, 0) == Complex::new(7.0, 8.0));
        assert!(x.read(1, 1) == Complex::new(9.0, 10.0));
        assert!(x.read(1, 2) == Complex::new(11.0, 12.0));

        assert!(x.read(2, 0) == Complex::new(13.0, 14.0));
        assert!(x.read(2, 1) == Complex::new(15.0, 16.0));
        assert!(x.read(2, 2) == Complex::new(17.0, 18.0));

        x.write(1, 0, Complex::new(3.0, 2.0));
        assert!(x.read(1, 0) == Complex::new(3.0, 2.0));
    }

    #[test]
    fn col_macro() {
        let mut x = col![3.0, 5.0, 7.0, 9.0];

        assert!(x[0] == 3.0);
        assert!(x[1] == 5.0);
        assert!(x[2] == 7.0);
        assert!(x[3] == 9.0);

        x[0] = 13.0;
        assert!(x[0] == 13.0);

        assert!(x.get(..) == x);
    }

    #[test]
    fn col_macro_cplx() {
        use num_complex::Complex;
        let new = Complex::new;
        let mut x = col![new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0),];

        assert!(x.read(0) == Complex::new(1.0, 2.0));
        assert!(x.read(1) == Complex::new(3.0, 4.0));
        assert!(x.read(2) == Complex::new(5.0, 6.0));

        x.write(0, Complex::new(3.0, 2.0));
        assert!(x.read(0) == Complex::new(3.0, 2.0));
    }

    #[test]
    fn col_macro_native_cplx() {
        use complex_native::c64 as Complex;

        let new = Complex::new;
        let mut x = col![new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0),];

        assert!(x.read(0) == Complex::new(1.0, 2.0));
        assert!(x.read(1) == Complex::new(3.0, 4.0));
        assert!(x.read(2) == Complex::new(5.0, 6.0));

        x.write(0, Complex::new(3.0, 2.0));
        assert!(x.read(0) == Complex::new(3.0, 2.0));
    }

    #[test]
    fn row_macro() {
        let mut x = row![3.0, 5.0, 7.0, 9.0];

        assert!(x[0] == 3.0);
        assert!(x[1] == 5.0);
        assert!(x[2] == 7.0);
        assert!(x[3] == 9.0);

        x.write(0, 13.0);
        assert!(x.read(0) == 13.0);
    }

    #[test]
    fn row_macro_cplx() {
        use num_complex::Complex;

        let new = Complex::new;
        let mut x = row![new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0),];

        assert!(x.read(0) == Complex::new(1.0, 2.0));
        assert!(x.read(1) == Complex::new(3.0, 4.0));
        assert!(x.read(2) == Complex::new(5.0, 6.0));

        x.write(0, Complex::new(3.0, 2.0));
        assert!(x.read(0) == Complex::new(3.0, 2.0));
    }

    #[test]
    fn row_macro_native_cplx() {
        use complex_native::c64 as Complex;

        let new = Complex::new;
        let mut x = row![new(1.0, 2.0), new(3.0, 4.0), new(5.0, 6.0),];

        assert!(x.read(0) == new(1.0, 2.0));
        assert!(x.read(1) == new(3.0, 4.0));
        assert!(x.read(2) == new(5.0, 6.0));

        x.write(0, new(3.0, 2.0));
        assert!(x.read(0) == new(3.0, 2.0));
    }

    #[test]
    fn null_col_and_row() {
        let null_col: Col<f64> = col![];
        assert!(null_col == Col::<f64>::new());

        let null_row: Row<f64> = row![];
        assert!(null_row == Row::<f64>::new());
    }

    #[test]
    fn positive_concat_f64() {
        let a0: Mat<f64> = Mat::from_fn(2, 2, |_, _| 1f64);
        let a1: Mat<f64> = Mat::from_fn(2, 3, |_, _| 2f64);
        let a2: Mat<f64> = Mat::from_fn(2, 4, |_, _| 3f64);

        let b0: Mat<f64> = Mat::from_fn(1, 6, |_, _| 4f64);
        let b1: Mat<f64> = Mat::from_fn(1, 3, |_, _| 5f64);

        let c0: Mat<f64> = Mat::from_fn(6, 1, |_, _| 6f64);
        let c1: Mat<f64> = Mat::from_fn(6, 3, |_, _| 7f64);
        let c2: Mat<f64> = Mat::from_fn(6, 2, |_, _| 8f64);
        let c3: Mat<f64> = Mat::from_fn(6, 3, |_, _| 9f64);

        let x = concat_impl(&[
            &[
                a0.as_ref().canonicalize(),
                a1.as_ref().canonicalize(),
                a2.as_ref().canonicalize(),
            ],
            &[
                b0.as_ref().canonicalize(), //
                b1.as_ref().canonicalize(),
            ],
            &[
                c0.as_ref().canonicalize(),
                c1.as_ref().canonicalize(),
                c2.as_ref().canonicalize(),
                c3.as_ref().canonicalize(),
            ],
        ]);

        assert!(x == concat![[a0, a1, a2], [b0, b1], [c0, c1, c2, &c3]]);

        assert!(x[(0, 0)] == 1f64);
        assert!(x[(1, 1)] == 1f64);
        assert!(x[(2, 2)] == 4f64);
        assert!(x[(3, 3)] == 7f64);
        assert!(x[(4, 4)] == 8f64);
        assert!(x[(5, 5)] == 8f64);
        assert!(x[(6, 6)] == 9f64);
        assert!(x[(7, 7)] == 9f64);
        assert!(x[(8, 8)] == 9f64);
    }

    #[test]
    fn to_owned_equality() {
        use num_complex::{Complex, Complex as C};
        let mut mf32: Mat<f32> = mat![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let mut mf64: Mat<f64> = mat![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let mut mf32c: Mat<Complex<f32>> = mat![
            [C::new(1., 1.), C::new(2., 2.), C::new(3., 3.)],
            [C::new(4., 4.), C::new(5., 5.), C::new(6., 6.)],
            [C::new(7., 7.), C::new(8., 8.), C::new(9., 9.)]
        ];
        let mut mf64c: Mat<Complex<f64>> = mat![
            [C::new(1., 1.), C::new(2., 2.), C::new(3., 3.)],
            [C::new(4., 4.), C::new(5., 5.), C::new(6., 6.)],
            [C::new(7., 7.), C::new(8., 8.), C::new(9., 9.)]
        ];

        assert!(mf32.transpose().to_owned().as_ref() == mf32.transpose());
        assert!(mf64.transpose().to_owned().as_ref() == mf64.transpose());
        assert!(mf32c.transpose().to_owned().as_ref() == mf32c.transpose());
        assert!(mf64c.transpose().to_owned().as_ref() == mf64c.transpose());

        assert!(mf32.as_mut().transpose_mut().to_owned().as_ref() == mf32.transpose());
        assert!(mf64.as_mut().transpose_mut().to_owned().as_ref() == mf64.transpose());
        assert!(mf32c.as_mut().transpose_mut().to_owned().as_ref() == mf32c.transpose());
        assert!(mf64c.as_mut().transpose_mut().to_owned().as_ref() == mf64c.transpose());
    }

    #[test]
    fn conj_to_owned_equality() {
        use num_complex::{Complex, Complex as C};
        let mut mf32: Mat<f32> = mat![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let mut mf64: Mat<f64> = mat![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let mut mf32c: Mat<Complex<f32>> = mat![
            [C::new(1., 1.), C::new(2., 2.), C::new(3., 3.)],
            [C::new(4., 4.), C::new(5., 5.), C::new(6., 6.)],
            [C::new(7., 7.), C::new(8., 8.), C::new(9., 9.)]
        ];
        let mut mf64c: Mat<Complex<f64>> = mat![
            [C::new(1., 1.), C::new(2., 2.), C::new(3., 3.)],
            [C::new(4., 4.), C::new(5., 5.), C::new(6., 6.)],
            [C::new(7., 7.), C::new(8., 8.), C::new(9., 9.)]
        ];

        assert!(mf32.as_ref().adjoint().to_owned().as_ref() == mf32.adjoint());
        assert!(mf64.as_ref().adjoint().to_owned().as_ref() == mf64.adjoint());
        assert!(mf32c.as_ref().adjoint().to_owned().as_ref() == mf32c.adjoint());
        assert!(mf64c.as_ref().adjoint().to_owned().as_ref() == mf64c.adjoint());

        assert!(mf32.as_mut().adjoint_mut().to_owned().as_ref() == mf32.adjoint());
        assert!(mf64.as_mut().adjoint_mut().to_owned().as_ref() == mf64.adjoint());
        assert!(mf32c.as_mut().adjoint_mut().to_owned().as_ref() == mf32c.adjoint());
        assert!(mf64c.as_mut().adjoint_mut().to_owned().as_ref() == mf64c.adjoint());
    }

    #[test]
    fn mat_mul_assign_scalar() {
        let mut x = mat![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];

        let expected = mat![[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]];
        x *= scale(2.0);
        assert_eq!(x, expected);

        let expected = mat![[0.0, 4.0], [8.0, 12.0], [16.0, 20.0]];
        let mut x_mut = x.as_mut();
        x_mut *= scale(2.0);
        assert_eq!(x, expected);
    }

    #[test]
    fn test_col_slice() {
        let mut matrix = mat![[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0f64]];

        assert!(matrix.col_as_slice(1) == &[5.0, 6.0, 7.0]);
        assert!(matrix.col_as_slice_mut(0) == &[1.0, 2.0, 3.0]);

        matrix
            .col_as_slice_mut(0)
            .copy_from_slice(&[-1.0, -2.0, -3.0]);

        let expected = mat![[-1.0, 5.0, 9.0], [-2.0, 6.0, 10.0], [-3.0, 7.0, 11.0f64]];
        assert!(matrix == expected);
    }

    #[test]
    fn from_slice() {
        let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];

        let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        let view = mat::from_column_major_slice_generic::<'_, f64, _, _>(&slice, 3, 2);
        assert_eq!(expected, view);
        let view = mat::from_column_major_slice_generic::<'_, f64, _, _>(&mut slice, 3, 2);
        assert_eq!(expected, view);

        let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let view = mat::from_row_major_slice_generic::<'_, f64, _, _>(&slice, 3, 2);
        assert_eq!(expected, view);
        let view = mat::from_row_major_slice_generic::<'_, f64, _, _>(&mut slice, 3, 2);
        assert_eq!(expected, view);
    }

    #[test]
    #[should_panic]
    fn from_slice_too_big() {
        let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0_f64];
        mat::from_column_major_slice_generic::<'_, f64, _, _>(&slice, 3, 2);
    }

    #[test]
    #[should_panic]
    fn from_slice_too_small() {
        let slice = [1.0, 2.0, 3.0, 4.0, 5.0_f64];
        mat::from_column_major_slice_generic::<'_, f64, _, _>(&slice, 3, 2);
    }

    #[test]
    fn test_is_finite() {
        use complex_native::c32;

        let inf = f32::INFINITY;
        let nan = f32::NAN;

        {
            assert!(<f32 as ComplexField>::faer_is_finite(&1.0));
            assert!(!<f32 as ComplexField>::faer_is_finite(&inf));
            assert!(!<f32 as ComplexField>::faer_is_finite(&-inf));
            assert!(!<f32 as ComplexField>::faer_is_finite(&nan));
        }
        {
            let x = c32::new(1.0, 2.0);
            assert!(<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(inf, 2.0);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(1.0, inf);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(inf, inf);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(nan, 2.0);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(1.0, nan);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));

            let x = c32::new(nan, nan);
            assert!(!<c32 as ComplexField>::faer_is_finite(&x));
        }
    }

    #[test]
    fn test_iter() {
        let mut mat = Mat::from_fn(9, 10, |i, j| (i + j) as f64);
        let mut iter = mat.row_chunks_mut(4);

        let first = iter.next();
        let second = iter.next();
        let last = iter.next();
        let none = iter.next();

        assert!(first == Some(Mat::from_fn(4, 10, |i, j| (i + j) as f64).as_mut()));
        assert!(second == Some(Mat::from_fn(4, 10, |i, j| (i + j + 4) as f64).as_mut()));
        assert!(last == Some(Mat::from_fn(1, 10, |i, j| (i + j + 8) as f64).as_mut()));
        assert!(none == None);
    }

    #[test]
    fn test_col_index() {
        let mut col_32: Col<f32> = Col::from_fn(3, |i| i as f32);
        col_32.as_mut()[1] = 10f32;
        assert!(col_32[1] == 10f32);

        let mut col_64: Col<f64> = Col::from_fn(3, |i| i as f64);
        col_64.as_mut()[1] = 10f64;
        assert!(col_64[1] == 10f64);
    }

    #[test]
    fn test_row_index() {
        let mut row_32: Row<f32> = Row::from_fn(3, |i| i as f32);
        row_32.as_mut()[1] = 10f32;
        assert!(row_32[1] == 10f32);

        let mut row_64: Row<f64> = Row::from_fn(3, |i| i as f64);
        row_64.as_mut()[1] = 10f64;
        assert!(row_64[1] == 10f64);
    }

    #[test]
    #[should_panic]
    fn test_approx_eq() {
        let approx_eq = ApproxEq {
            abs_tol: 1e-6,
            rel_tol: 1e-6,
        };

        assert!(1.0 ~ 2.0);
    }

    #[derive(Copy, Clone, Debug)]
    pub struct ApproxEq<E> {
        pub abs_tol: E,
        pub rel_tol: E,
    }

    #[derive(Copy, Clone, Debug)]
    pub struct ApproxEqError;

    impl<E: RealField> ApproxEq<E> {
        pub fn eps() -> Self {
            Self {
                abs_tol: E::faer_epsilon() * E::faer_from_f64(128.0),
                rel_tol: E::faer_epsilon() * E::faer_from_f64(128.0),
            }
        }
    }

    impl<R: RealField, E: ComplexField<Real = R>> equator::CmpDisplay<ApproxEq<R>, E, E>
        for ApproxEqError
    {
        fn fmt(
            &self,
            cmp: &ApproxEq<R>,
            lhs: &E,
            lhs_source: &str,
            _: &dyn core::fmt::Debug,
            rhs: &E,
            rhs_source: &str,
            _: &dyn core::fmt::Debug,
            f: &mut core::fmt::Formatter,
        ) -> core::fmt::Result {
            use coe::Coerce;
            if E::IS_F64 {
                let ApproxEq { abs_tol, rel_tol }: ApproxEq<f64> = *cmp.coerce();
                writeln!(
                    f,
                    "Assertion failed: {lhs_source} ~ {rhs_source}\nwith absolute tolerance = {abs_tol:.1e}\nwith relative tolerance = {rel_tol:.1e}"
                )?;
            } else if E::IS_F32 {
                let ApproxEq { abs_tol, rel_tol }: ApproxEq<f32> = *cmp.coerce();
                writeln!(
                    f,
                    "Assertion failed: {lhs_source} ~ {rhs_source}\nwith absolute tolerance = {abs_tol:.1e}\nwith relative tolerance = {rel_tol:.1e}"
                )?;
            } else {
                let ApproxEq { abs_tol, rel_tol } = *cmp;
                writeln!(
                    f,
                    "Assertion failed: {lhs_source} ~ {rhs_source}\nwith absolute tolerance = {abs_tol:?}\nwith relative tolerance = {rel_tol:?}"
                )?;
            }

            let distance = (*lhs - *rhs).faer_abs();

            write!(f, "- {lhs_source} = {lhs:?}\n")?;
            write!(f, "- {rhs_source} = {rhs:?}\n")?;
            write!(f, "- distance = {distance:?}")
        }
    }

    impl<R: RealField, E: ComplexField<Real = R>> equator::CmpError<ApproxEq<R>, E, E> for ApproxEq<R> {
        type Error = ApproxEqError;
    }

    impl<R: RealField, E: ComplexField<Real = R>> equator::Cmp<E, E> for ApproxEq<R> {
        fn test(&self, &lhs: &E, &rhs: &E) -> Result<(), ApproxEqError> {
            let Self { abs_tol, rel_tol } = *self;
            assert!(abs_tol >= R::faer_zero());
            assert!(rel_tol >= R::faer_zero());

            let diff = (lhs - rhs).faer_abs();
            let lhs = lhs.faer_abs();
            let rhs = lhs.faer_abs();
            let max = if lhs > rhs { lhs } else { rhs };

            if (max == R::faer_zero() && diff <= abs_tol)
                || (diff <= abs_tol || diff <= rel_tol * max)
            {
                Ok(())
            } else {
                Err(ApproxEqError)
            }
        }
    }
}

// #[path = "krylov_schur.rs"]
// mod krylov_schur_prototype;

/// Sealed trait for types that can be created from "unbound" values, as long as their
/// struct preconditions are upheld.
pub trait Unbind<I = usize>: Send + Sync + Copy + core::fmt::Debug + seal::Seal {
    /// Create new value.
    /// # Safety
    /// Safety invariants must be upheld.
    unsafe fn new_unbound(idx: I) -> Self;

    /// Returns the unbound value, unconstrained by safety invariants.
    fn unbound(self) -> I;
}

/// Type that can be used to index into a range.
pub type Idx<N, I = usize> = <N as ShapeIdx>::Idx<I>;
/// Type that can be used to partition a range.
pub type IdxInc<N, I = usize> = <N as ShapeIdx>::IdxInc<I>;

/// Base trait for [`Shape`].
pub trait ShapeIdx {
    /// Type that can be used to index into a range.
    type Idx<I: Index>: Unbind<I> + Ord + Eq;
    /// Type that can be used to partition a range.
    type IdxInc<I: Index>: Unbind<I> + Ord + Eq + From<Idx<Self, I>>;
}

/// Matrix dimension.
pub trait Shape:
    Unbind
    + Ord
    + ShapeIdx<Idx<usize>: Ord + Eq + PartialOrd<Self>, IdxInc<usize>: Ord + Eq + PartialOrd<Self>>
{
    /// Whether the types involved have any safety invariants.
    const IS_BOUND: bool = true;

    /// Bind the current value using a invariant lifetime guard.
    #[inline]
    fn bind<'n>(self, guard: generativity::Guard<'n>) -> utils::bound::Dim<'n> {
        utils::bound::Dim::new(self.unbound(), guard)
    }

    /// Cast a slice of bound values to unbound values.
    #[inline]
    fn cast_idx_slice<I: Index>(slice: &[Idx<Self, I>]) -> &[I] {
        unsafe { core::slice::from_raw_parts(slice.as_ptr() as _, slice.len()) }
    }

    /// Cast a slice of bound values to unbound values.
    #[inline]
    fn cast_idx_inc_slice<I: Index>(slice: &[IdxInc<Self, I>]) -> &[I] {
        unsafe { core::slice::from_raw_parts(slice.as_ptr() as _, slice.len()) }
    }

    /// Returns the index `0`, which is always valid.
    #[inline(always)]
    fn start() -> IdxInc<Self> {
        unsafe { IdxInc::<Self>::new_unbound(0) }
    }

    /// Returns the incremented value, as an inclusive index.
    #[inline(always)]
    fn next(idx: Idx<Self>) -> IdxInc<Self> {
        unsafe { IdxInc::<Self>::new_unbound(idx.unbound() + 1) }
    }

    /// Returns the last value, equal to the dimension.
    #[inline(always)]
    fn end(self) -> IdxInc<Self> {
        unsafe { IdxInc::<Self>::new_unbound(self.unbound()) }
    }

    /// Checks if the index is valid, returning `Some(_)` in that case.
    #[inline(always)]
    fn idx(self, idx: usize) -> Option<Idx<Self>> {
        if idx < self.unbound() {
            Some(unsafe { Idx::<Self>::new_unbound(idx) })
        } else {
            None
        }
    }

    /// Checks if the index is valid, returning `Some(_)` in that case.
    #[inline(always)]
    fn idx_inc(self, idx: usize) -> Option<IdxInc<Self>> {
        if idx <= self.unbound() {
            Some(unsafe { IdxInc::<Self>::new_unbound(idx) })
        } else {
            None
        }
    }

    /// Checks if the index is valid, and panics otherwise.
    #[inline(always)]
    fn checked_idx(self, idx: usize) -> Idx<Self> {
        equator::assert!(idx < self.unbound());
        unsafe { Idx::<Self>::new_unbound(idx) }
    }

    /// Checks if the index is valid, and panics otherwise.
    #[inline(always)]
    fn checked_idx_inc(self, idx: usize) -> IdxInc<Self> {
        equator::assert!(idx <= self.unbound());
        unsafe { IdxInc::<Self>::new_unbound(idx) }
    }

    /// Assumes the index is valid.
    /// # Safety
    /// The index must be valid.
    #[inline(always)]
    unsafe fn unchecked_idx(self, idx: usize) -> Idx<Self> {
        equator::debug_assert!(idx < self.unbound());
        unsafe { Idx::<Self>::new_unbound(idx) }
    }

    /// Assumes the index is valid.
    /// # Safety
    /// The index must be valid.
    #[inline(always)]
    unsafe fn unchecked_idx_inc(self, idx: usize) -> IdxInc<Self> {
        equator::debug_assert!(idx <= self.unbound());
        unsafe { IdxInc::<Self>::new_unbound(idx) }
    }

    /// Returns an iterator over the indices between `from` and `to`.
    #[inline(always)]
    fn indices(
        from: IdxInc<Self>,
        to: IdxInc<Self>,
    ) -> impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Self>> {
        (from.unbound()..to.unbound()).map(
            #[inline(always)]
            |i| unsafe { Idx::<Self>::new_unbound(i) },
        )
    }
}

impl<T: Send + Sync + Copy + core::fmt::Debug + seal::Seal> Unbind<T> for T {
    #[inline(always)]
    unsafe fn new_unbound(idx: T) -> Self {
        idx
    }

    #[inline(always)]
    fn unbound(self) -> T {
        self
    }
}

impl ShapeIdx for usize {
    type Idx<I: Index> = I;
    type IdxInc<I: Index> = I;
}
impl Shape for usize {
    const IS_BOUND: bool = false;
}
