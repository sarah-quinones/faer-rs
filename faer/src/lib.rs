//! `faer` is a general-purpose linear algebra library for rust, with a focus on high performance
//! for algebraic operations on medium/large matrices, as well as matrix decompositions
//!
//! most of the high-level functionality in this library is provided through associated functions in
//! its vocabulary types: [`Mat`]/[`MatRef`]/[`MatMut`]
//!
//! `faer` is recommended for applications that handle medium to large dense matrices, and its
//! design is not well suited for applications that operate mostly on low dimensional vectors and
//! matrices such as computer graphics or game development. for such applications, `nalgebra` and
//! `cgmath` may be better suited
//!
//! # basic usage
//!
//! [`Mat`] is a resizable matrix type with dynamic capacity, which can be created using
//! [`Mat::new`] to produce an empty $0\times 0$ matrix, [`Mat::zeros`] to create a rectangular
//! matrix filled with zeros, [`Mat::identity`] to create an identity matrix, or [`Mat::from_fn`]
//! for the more general case
//!
//! Given a `&Mat<T>` (resp. `&mut Mat<T>`), a [`MatRef<'_, T>`](MatRef) (resp. [`MatMut<'_,
//! T>`](MatMut)) can be created by calling [`Mat::as_ref`] (resp. [`Mat::as_mut`]), which allow
//! for more flexibility than `Mat` in that they allow slicing ([`MatRef::get`]) and splitting
//! ([`MatRef::split_at`])
//!
//! `MatRef` and `MatMut` are lightweight view objects. the former can be copied freely while the
//! latter has move and reborrow semantics, as described in its documentation
//!
//! most of the matrix operations can be used through the corresponding math operators: `+` for
//! matrix addition, `-` for subtraction, `*` for either scalar or matrix multiplication depending
//! on the types of the operands.
//!
//! ## example
//! ```
//! use faer::{Mat, Scale, mat};
//!
//! let a = mat![
//! 	[1.0, 5.0, 9.0], //
//! 	[2.0, 6.0, 10.0],
//! 	[3.0, 7.0, 11.0],
//! 	[4.0, 8.0, 12.0f64],
//! ];
//!
//! let b = Mat::from_fn(4, 3, |i, j| (i + j) as f64);
//!
//! let add = &a + &b;
//! let sub = &a - &b;
//! let scale = Scale(3.0) * &a;
//! let mul = &a * b.transpose();
//!
//! let a00 = a[(0, 0)];
//! ```
//!
//! # matrix decompositions
//! `faer` provides a variety of matrix factorizations, each with its own advantages and drawbacks:
//!
//! ## $LL^\top$ decomposition
//! [`Mat::llt`] decomposes a self-adjoint positive definite matrix $A$ such that
//! $$A = LL^H,$$
//! where $L$ is a lower triangular matrix. this decomposition is highly efficient and has good
//! stability properties
//!
//! [an implementation for sparse matrices is also available](sparse::linalg::solvers::Llt)
//!
//! ## $LBL^\top$ decomposition
//! [`Mat::lblt`] decomposes a self-adjoint (possibly indefinite) matrix $A$ such that
//! $$P A P^\top = LBL^H,$$
//! where $P$ is a permutation matrix, $L$ is a lower triangular matrix, and $B$ is a block
//! diagonal matrix, with $1 \times 1$ or $2 \times 2$ diagonal blocks.
//! this decomposition is efficient and has good stability properties
//!
//! ## $LU$ decomposition with partial pivoting
//! [`Mat::partial_piv_lu`] decomposes a square invertible matrix $A$ into a lower triangular
//! matrix $L$, a unit upper triangular matrix $U$, and a permutation matrix $P$, such that
//! $$PA = LU$$
//! it is used by default for computing the determinant, and is generally the recommended method
//! for solving a square linear system or computing the inverse of a matrix (although we generally
//! recommend using a [`faer::linalg::solvers::Solve`](crate::linalg::solvers::Solve) instead of
//! computing the inverse explicitly)
//!
//! [an implementation for sparse matrices is also available](sparse::linalg::solvers::Lu)
//!
//! ## $LU$ decomposition with full pivoting
//! [`Mat::full_piv_lu`] decomposes a generic rectangular matrix $A$ into a lower triangular
//! matrix $L$, a unit upper triangular matrix $U$, and permutation matrices $P$ and $Q$, such that
//! $$PAQ^\top = LU$$
//! it can be more stable than the LU decomposition with partial pivoting, in exchange for being
//! more computationally expensive
//!
//! ## $QR$ decomposition
//! [`Mat::qr`] decomposes a matrix $A$ into the product $$A = QR,$$
//! where $Q$ is a unitary matrix, and $R$ is an upper trapezoidal matrix. it is often used for
//! solving least squares problems
//!
//! [an implementation for sparse matrices is also available](sparse::linalg::solvers::Qr)
//!
//! ## $QR$ decomposition with column pivoting
//! ([`Mat::col_piv_qr`]) decomposes a matrix $A$ into the product $$AP^\top = QR,$$
//! where $P$ is a permutation matrix, $Q$ is a unitary matrix, and $R$ is an upper trapezoidal
//! matrix
//!
//! it is slower than the version with no pivoting, in exchange for being more numerically stable
//! for rank-deficient matrices
//!
//! ## singular value decomposition
//! the SVD of a matrix $A$ of shape $(m, n)$ is a decomposition into three components $U$, $S$,
//! and $V$, such that:
//!
//! - $U$ has shape $(m, m)$ and is a unitary matrix,
//! - $V$ has shape $(n, n)$ and is a unitary matrix,
//! - $S$ has shape $(m, n)$ and is zero everywhere except the main diagonal, with nonnegative
//! diagonal values in nonincreasing order,
//! - and finally:
//!
//! $$A = U S V^H$$
//!
//! the SVD is provided in two forms: either the full matrices $U$ and $V$ are computed, using
//! [`Mat::svd`], or only their first $\min(m, n)$ columns are computed, using
//! [`Mat::thin_svd`]
//!
//! if only the singular values (elements of $S$) are desired, they can be obtained in
//! nonincreasing order using [`Mat::singular_values`]
//!
//! ## eigendecomposition
//! **note**: the order of the eigenvalues is currently unspecified and may be changed in a future
//! release
//!
//! the eigenvalue decomposition of a square matrix $A$ of shape $(n, n)$ is a decomposition into
//! two components $U$, $S$:
//!
//! - $U$ has shape $(n, n)$ and is invertible,
//! - $S$ has shape $(n, n)$ and is a diagonal matrix,
//! - and finally:
//!
//! $$A = U S U^{-1}$$
//!
//! if $A$ is self-adjoint, then $U$ can be made unitary ($U^{-1} = U^H$), and $S$ is real valued.
//! additionally, the eigenvalues are sorted in nondecreasing order
//!
//! Depending on the domain of the input matrix and whether it is self-adjoint, multiple methods
//! are provided to compute the eigendecomposition:
//! * [`Mat::self_adjoint_eigen`] can be used with either real or complex matrices,
//! producing an eigendecomposition of the same type,
//! * [`Mat::eigen`] can be used with real or complex matrices, but always produces complex values.
//!
//! if only the eigenvalues (elements of $S$) are desired, they can be obtained using
//! [`Mat::self_adjoint_eigenvalues`] (nondecreasing order), [`Mat::eigenvalues`]
//! with the same conditions described above.
//!
//! # crate features
//!
//! - `std`: enabled by default. links with the standard library to enable additional features such
//!   as cpu feature detection at runtime
//! - `rayon`: enabled by default. enables the `rayon` parallel backend and enables global
//!   parallelism by default
//! - `serde`: Enables serialization and deserialization of [`Mat`]
//! - `npy`: enables conversions to/from numpy's matrix file format
//! - `perf-warn`: produces performance warnings when matrix operations are called with suboptimal
//! data layout
//! - `nightly`: requires the nightly compiler. enables experimental simd features such as avx512

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(non_snake_case)]
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

/// see: [`generativity::make_guard`]
#[macro_export]
macro_rules! make_guard {
    ($($name:ident),* $(,)?) => {$(
        #[allow(unused_unsafe)]
        let $name = unsafe { extern crate generativity; ::generativity::Id::new() };
        #[allow(unused, unused_unsafe)]
        let lifetime_brand = unsafe { extern crate generativity; ::generativity::LifetimeBrand::new(&$name) };
        #[allow(unused_unsafe)]
        let $name = unsafe { extern crate generativity; ::generativity::Guard::new($name) };
    )*};
}

macro_rules! repeat_n {
	($e: expr, $n: expr) => {
		iter::repeat_n($e, $n)
	};
}

macro_rules! try_const {
	($e: expr) => {
		::pulp::try_const! { $e }
	};
}

use core::num::NonZeroUsize;
use core::sync::atomic::AtomicUsize;
use equator::{assert, debug_assert};
use faer_traits::*;

macro_rules! auto {
	($ty: ty) => {
		$crate::Auto::<$ty>::auto()
	};
}

macro_rules! dispatch {
	($imp: expr, $ty: ident, $T: ty $(,)?) => {
		if try_const! { <$T>::IS_NATIVE_C32 } {
			unsafe { transmute(<ComplexImpl<f32> as ComplexField>::Arch::default().dispatch(transmute::<_, $ty<ComplexImpl<f32>>>($imp))) }
		} else if try_const! { <$T>::IS_NATIVE_C64 } {
			unsafe { transmute(<ComplexImpl<f64> as ComplexField>::Arch::default().dispatch(transmute::<_, $ty<ComplexImpl<f64>>>($imp))) }
		} else {
			<$T>::Arch::default().dispatch($imp)
		}
	};
}

macro_rules! stack_mat {
	($name: ident, $m: expr, $n: expr, $A: expr, $N: expr, $T: ty $(,)?) => {
		let mut __tmp = {
			#[repr(align(64))]
			struct __Col<T, const A: usize>([T; A]);
			struct __Mat<T, const A: usize, const N: usize>([__Col<T, A>; N]);

			core::mem::MaybeUninit::<__Mat<$T, $A, $N>>::uninit()
		};
		let __stack = MemStack::new_any(core::slice::from_mut(&mut __tmp));
		let mut $name = $crate::linalg::temp_mat_zeroed::<$T, _, _>($m, $n, __stack).0;
		let mut $name = $name.as_mat_mut();
	};

	($name: ident, $m: expr, $n: expr,  $T: ty $(,)?) => {
		stack_mat!($name, $m, $n, $m, $n, $T)
	};
}

#[macro_export]
#[doc(hidden)]
macro_rules! __dbg {
    () => {
        std::eprintln!("[{}:{}:{}]", std::file!(), std::line!(), std::column!())
    };
    ($val:expr $(,)?) => {
        match $val {
            tmp => {
                std::eprintln!("[{}:{}:{}] {} = {:16.12?}",
                    std::file!(), std::line!(), std::column!(), std::stringify!($val), &tmp);
                tmp
            }
        }
    };
    ($($val:expr),+ $(,)?) => {
        ($($crate::__dbg!($val)),+,)
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
			static $name: ::core::sync::atomic::AtomicBool = ::core::sync::atomic::AtomicBool::new(false);
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

#[doc(hidden)]
#[macro_export]
macro_rules! with_dim {
	($name: ident, $value: expr $(,)?) => {
		let __val__ = $value;
		$crate::make_guard!($name);
		let $name = $crate::utils::bound::Dim::new(__val__, $name);
	};

	({$(let $name: ident = $value: expr;)*}) => {$(
		let __val__ = $value;
		$crate::make_guard!($name);
		let $name = $crate::utils::bound::Dim::new(__val__, $name);
	)*};
}

/// zips together matrix of the same size, so that coefficient-wise operations can be performed on
/// their elements.
///
/// # note
/// the order in which the matrix elements are traversed is unspecified.
///
/// # example
/// ```
/// use faer::{Mat, mat, unzip, zip};
///
/// let nrows = 2;
/// let ncols = 3;
///
/// let a = mat![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
/// let b = mat![[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]];
/// let mut sum = Mat::<f64>::zeros(nrows, ncols);
///
/// zip!(&mut sum, &a, &b).for_each(|unzip!(sum, a, b)| {
/// 	*sum = a + b;
/// });
///
/// for i in 0..nrows {
/// 	for j in 0..ncols {
/// 		assert_eq!(sum[(i, j)], a[(i, j)] + b[(i, j)]);
/// 	}
/// }
/// ```
#[macro_export]
macro_rules! zip {
    ($head: expr $(,)?) => {
        $crate::linalg::zip::LastEq($crate::linalg::zip::IntoView::into_view($head), ::core::marker::PhantomData)
    };

    ($head: expr, $($tail: expr),* $(,)?) => {
        $crate::linalg::zip::ZipEq::new($crate::linalg::zip::IntoView::into_view($head), $crate::zip!($($tail,)*))
    };
}

/// used to undo the zipping by the [`zip!`] macro.
///
/// # example
/// ```
/// use faer::{Mat, mat, unzip, zip};
///
/// let nrows = 2;
/// let ncols = 3;
///
/// let a = mat![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
/// let b = mat![[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]];
/// let mut sum = Mat::<f64>::zeros(nrows, ncols);
///
/// zip!(&mut sum, &a, &b).for_each(|unzip!(sum, a, b)| {
/// 	*sum = a + b;
/// });
///
/// for i in 0..nrows {
/// 	for j in 0..ncols {
/// 		assert_eq!(sum[(i, j)], a[(i, j)] + b[(i, j)]);
/// 	}
/// }
/// ```
#[macro_export]
macro_rules! unzip {
    ($head: pat $(,)?) => {
        $crate::linalg::zip::Last($head)
    };

    ($head: pat, $($tail: pat),* $(,)?) => {
        $crate::linalg::zip::Zip($head, $crate::unzip!($($tail,)*))
    };
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

/// creates a [`Mat`] containing the arguments.
///
/// ```
/// use faer::mat;
///
/// let matrix = mat![
/// 	[1.0, 5.0, 9.0], //
/// 	[2.0, 6.0, 10.0],
/// 	[3.0, 7.0, 11.0],
/// 	[4.0, 8.0, 12.0f64],
/// ];
///
/// assert_eq!(matrix[(0, 0)], 1.0);
/// assert_eq!(matrix[(1, 0)], 2.0);
/// assert_eq!(matrix[(2, 0)], 3.0);
/// assert_eq!(matrix[(3, 0)], 4.0);
///
/// assert_eq!(matrix[(0, 1)], 5.0);
/// assert_eq!(matrix[(1, 1)], 6.0);
/// assert_eq!(matrix[(2, 1)], 7.0);
/// assert_eq!(matrix[(3, 1)], 8.0);
///
/// assert_eq!(matrix[(0, 2)], 9.0);
/// assert_eq!(matrix[(1, 2)], 10.0);
/// assert_eq!(matrix[(2, 2)], 11.0);
/// assert_eq!(matrix[(3, 2)], 12.0);
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
            let __data = ::core::mem::ManuallyDrop::new($crate::__transpose_impl!([] $($($v),* ;)*));
            let __data = &*__data;
            let __ncols = __data.len();
            let __nrows = (*__data.get(0).unwrap()).len();

            #[allow(unused_unsafe)]
            unsafe {
                $crate::mat::Mat::from_fn(__nrows, __ncols, |i, j| ::core::ptr::from_ref(&__data[j][i]).read())
            }
        }
    };
}

/// creates a [`col::Col`] containing the arguments
///
/// ```
/// use faer::col;
///
/// let col_vec = col![3.0, 5.0, 7.0, 9.0];
///
/// assert_eq!(col_vec[0], 3.0);
/// assert_eq!(col_vec[1], 5.0);
/// assert_eq!(col_vec[2], 7.0);
/// assert_eq!(col_vec[3], 9.0);
/// ```
#[macro_export]
macro_rules! col {
    ($($v: expr),* $(,)?) => {
        {
            let __data = ::core::mem::ManuallyDrop::new([$($v,)*]);
            let __data = &*__data;
            let __len = __data.len();

            #[allow(unused_unsafe)]
            unsafe {
                $crate::col::Col::from_fn(__len, |i| ::core::ptr::from_ref(&__data[i]).read())
            }
        }
    };
}

/// creates a [`row::Row`] containing the arguments
///
/// ```
/// use faer::row;
///
/// let row_vec = row![3.0, 5.0, 7.0, 9.0];
///
/// assert_eq!(row_vec[0], 3.0);
/// assert_eq!(row_vec[1], 5.0);
/// assert_eq!(row_vec[2], 7.0);
/// assert_eq!(row_vec[3], 9.0);
/// ```
#[macro_export]
macro_rules! row {
    ($($v: expr),* $(,)?) => {
        {
            let __data = ::core::mem::ManuallyDrop::new([$($v,)*]);
            let __data = &*__data;
            let __len = __data.len();

            #[allow(unused_unsafe)]
            unsafe {
                $crate::row::Row::from_fn(__len, |i| ::core::ptr::from_ref(&__data[i]).read())
            }
        }
    };
}

/// convenience function to concatenate a nested list of matrices into a single
/// big ['Mat']. concatonation pattern follows the numpy.block convention that
/// each sub-list must have an equal number of columns (net) but the boundaries
/// do not need to align. in other words, this sort of thing:
/// ```notcode
///   AAAbb
///   AAAbb
///   cDDDD
/// ```
/// is perfectly acceptable
#[doc(hidden)]
#[track_caller]
pub fn concat_impl<T: ComplexField>(blocks: &[&[(mat::MatRef<'_, T>, Conj)]]) -> mat::Mat<T> {
	#[inline(always)]
	fn count_total_columns<T: ComplexField>(block_row: &[(mat::MatRef<'_, T>, Conj)]) -> usize {
		let mut out: usize = 0;
		for (elem, _) in block_row.iter() {
			out += elem.ncols();
		}
		out
	}

	#[inline(always)]
	#[track_caller]
	fn count_rows<T: ComplexField>(block_row: &[(mat::MatRef<'_, T>, Conj)]) -> usize {
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

	let mut mat = mat::Mat::<T>::zeros(n, m);
	let mut ni: usize = 0;
	let mut mj: usize;
	for row in blocks.iter() {
		mj = 0;

		for (elem, conj) in row.iter() {
			let mut dst = mat.as_mut().submatrix_mut(ni, mj, elem.nrows(), elem.ncols());
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

/// concatenates the matrices in each row horizontally,
/// then concatenates the results vertically
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
            $crate::concat_impl(&[$(&[$(($v).as_ref().__canonicalize(),)*],)*])
        }
    };
}

/// helper utilities
pub mod utils;

/// diagonal matrix
pub mod diag;
/// rectangular matrix
pub mod mat;
/// permutation matrix
pub mod perm;

/// column vector
pub mod col;
/// row vector
pub mod row;

pub mod linalg;
#[path = "./operator/mod.rs"]
pub mod matrix_free;
pub mod sparse;

/// de-serialization from common matrix file formats
#[cfg(feature = "std")]
pub mod io;

#[cfg(feature = "serde")]
mod serde;

/// native unsigned integer type
pub trait Index: traits::IndexCore + traits::Index + seal::Seal {}
impl<T: faer_traits::Index<Signed: seal::Seal> + seal::Seal> Index for T {}

mod seal {
	pub trait Seal {}
	impl<T: faer_traits::Seal> Seal for T {}
	impl Seal for crate::utils::bound::Dim<'_> {}
	impl<I: crate::Index> Seal for crate::utils::bound::Idx<'_, I> {}
	impl<I: crate::Index> Seal for crate::utils::bound::IdxInc<'_, I> {}
	impl<I: crate::Index> Seal for crate::utils::bound::MaybeIdx<'_, I> {}
	impl<I: crate::Index> Seal for crate::utils::bound::IdxIncOne<I> {}
	impl<I: crate::Index> Seal for crate::utils::bound::MaybeIdxOne<I> {}
	impl Seal for crate::utils::bound::One {}
	impl Seal for crate::utils::bound::Zero {}
	impl Seal for crate::ContiguousFwd {}
	impl Seal for crate::ContiguousBwd {}
}

/// sealed trait for types that can be created from "unbound" values, as long as their
/// struct preconditions are upheld
pub trait Unbind<I = usize>: Send + Sync + Copy + core::fmt::Debug + seal::Seal {
	/// creates new value
	/// # safety
	/// safety invariants must be upheld
	unsafe fn new_unbound(idx: I) -> Self;

	/// returns the unbound value, unconstrained by safety invariants
	fn unbound(self) -> I;
}

/// type that can be used to index into a range
pub type Idx<Dim, I = usize> = <Dim as ShapeIdx>::Idx<I>;
/// type that can be used to partition a range
pub type IdxInc<Dim, I = usize> = <Dim as ShapeIdx>::IdxInc<I>;
/// either an index or a negative value
pub type MaybeIdx<Dim, I = usize> = <Dim as ShapeIdx>::MaybeIdx<I>;

/// base trait for [`Shape`]
pub trait ShapeIdx {
	/// type that can be used to index into a range
	type Idx<I: Index>: Unbind<I> + Ord + Eq;
	/// type that can be used to partition a range
	type IdxInc<I: Index>: Unbind<I> + Ord + Eq + From<Idx<Self, I>>;
	/// either an index or a negative value
	type MaybeIdx<I: Index>: Unbind<I::Signed> + Ord + Eq;
}

/// matrix dimension
pub trait Shape: Unbind + Ord + ShapeIdx<Idx<usize>: Ord + Eq + PartialOrd<Self>, IdxInc<usize>: Ord + Eq + PartialOrd<Self>> {
	/// whether the types involved have any safety invariants
	const IS_BOUND: bool = true;

	/// bind the current value using a invariant lifetime guard
	#[inline]
	fn bind<'n>(self, guard: generativity::Guard<'n>) -> utils::bound::Dim<'n> {
		utils::bound::Dim::new(self.unbound(), guard)
	}

	/// cast a slice of bound values to unbound values
	#[inline]
	fn cast_idx_slice<I: Index>(slice: &[Idx<Self, I>]) -> &[I] {
		unsafe { core::slice::from_raw_parts(slice.as_ptr() as _, slice.len()) }
	}

	/// cast a slice of bound values to unbound values
	#[inline]
	fn cast_idx_inc_slice<I: Index>(slice: &[IdxInc<Self, I>]) -> &[I] {
		unsafe { core::slice::from_raw_parts(slice.as_ptr() as _, slice.len()) }
	}

	/// returns the index `0`, which is always valid
	#[inline(always)]
	fn start() -> IdxInc<Self> {
		unsafe { IdxInc::<Self>::new_unbound(0) }
	}

	/// returns the incremented value, as an inclusive index
	#[inline(always)]
	fn next(idx: Idx<Self>) -> IdxInc<Self> {
		unsafe { IdxInc::<Self>::new_unbound(idx.unbound() + 1) }
	}

	/// returns the last value, equal to the dimension
	#[inline(always)]
	fn end(self) -> IdxInc<Self> {
		unsafe { IdxInc::<Self>::new_unbound(self.unbound()) }
	}

	/// checks if the index is valid, returning `Some(_)` in that case
	#[inline(always)]
	fn idx(self, idx: usize) -> Option<Idx<Self>> {
		if idx < self.unbound() {
			Some(unsafe { Idx::<Self>::new_unbound(idx) })
		} else {
			None
		}
	}

	/// checks if the index is valid, returning `Some(_)` in that case
	#[inline(always)]
	fn idx_inc(self, idx: usize) -> Option<IdxInc<Self>> {
		if idx <= self.unbound() {
			Some(unsafe { IdxInc::<Self>::new_unbound(idx) })
		} else {
			None
		}
	}

	/// checks if the index is valid, and panics otherwise
	#[inline(always)]
	fn checked_idx(self, idx: usize) -> Idx<Self> {
		equator::assert!(idx < self.unbound());
		unsafe { Idx::<Self>::new_unbound(idx) }
	}

	/// checks if the index is valid, and panics otherwise
	#[inline(always)]
	fn checked_idx_inc(self, idx: usize) -> IdxInc<Self> {
		equator::assert!(idx <= self.unbound());
		unsafe { IdxInc::<Self>::new_unbound(idx) }
	}

	/// assumes the index is valid
	/// # safety
	/// the index must be valid
	#[inline(always)]
	unsafe fn unchecked_idx(self, idx: usize) -> Idx<Self> {
		equator::debug_assert!(idx < self.unbound());
		unsafe { Idx::<Self>::new_unbound(idx) }
	}

	/// assumes the index is valid
	/// # safety
	/// the index must be valid
	#[inline(always)]
	unsafe fn unchecked_idx_inc(self, idx: usize) -> IdxInc<Self> {
		equator::debug_assert!(idx <= self.unbound());
		unsafe { IdxInc::<Self>::new_unbound(idx) }
	}

	/// returns an iterator over the indices between `from` and `to`
	#[inline(always)]
	fn indices(from: IdxInc<Self>, to: IdxInc<Self>) -> impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Self>> {
		(from.unbound()..to.unbound()).map(
			#[inline(always)]
			|i| unsafe { Idx::<Self>::new_unbound(i) },
		)
	}
}

impl<T: Send + Sync + Copy + core::fmt::Debug + faer_traits::Seal> Unbind<T> for T {
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
	type MaybeIdx<I: Index> = I::Signed;
}
impl Shape for usize {
	const IS_BOUND: bool = false;
}

/// stride distance between two consecutive elements along a given dimension
pub trait Stride: seal::Seal + core::fmt::Debug + Copy + Send + Sync + 'static {
	/// the reversed stride type
	type Rev: Stride<Rev = Self>;
	/// returns the reversed stride
	fn rev(self) -> Self::Rev;

	/// returns the stride in elements
	fn element_stride(self) -> isize;
}

impl Stride for isize {
	type Rev = Self;

	#[inline(always)]
	fn rev(self) -> Self::Rev {
		-self
	}

	#[inline(always)]
	fn element_stride(self) -> isize {
		self
	}
}

/// contiguous stride equal to `+1`
#[derive(Copy, Clone, Debug)]
pub struct ContiguousFwd;
/// contiguous stride equal to `-1`
#[derive(Copy, Clone, Debug)]
pub struct ContiguousBwd;

impl Stride for ContiguousFwd {
	type Rev = ContiguousBwd;

	#[inline(always)]
	fn rev(self) -> Self::Rev {
		ContiguousBwd
	}

	#[inline(always)]
	fn element_stride(self) -> isize {
		1
	}
}

impl Stride for ContiguousBwd {
	type Rev = ContiguousFwd;

	#[inline(always)]
	fn rev(self) -> Self::Rev {
		ContiguousFwd
	}

	#[inline(always)]
	fn element_stride(self) -> isize {
		-1
	}
}

/// memory allocation error
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TryReserveError {
	///rRequired allocation does not fit within `isize` bytes
	CapacityOverflow,
	/// allocator could not provide an allocation with the requested layout
	AllocError {
		/// requested layout
		layout: core::alloc::Layout,
	},
}

/// determines whether the input should be implicitly conjugated or not
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Conj {
	/// no implicit conjugation
	No,
	/// implicit conjugation
	Yes,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum DiagStatus {
	Unit,
	Generic,
}

/// determines whether to replace or add to the result of a matmul operatio
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Accum {
	/// overwrites the output buffer
	Replace,
	/// adds the result to the output buffer
	Add,
}

/// determines which side of a self-adjoint matrix should be accessed
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Side {
	/// lower triangular half
	Lower,
	/// upper triangular half
	Upper,
}

impl Conj {
	/// returns `self == Conj::Yes`
	#[inline]
	pub const fn is_conj(self) -> bool {
		matches!(self, Conj::Yes)
	}

	/// returns the composition of `self` and `other`
	#[inline]
	pub const fn compose(self, other: Self) -> Self {
		match (self, other) {
			(Conj::No, Conj::No) => Conj::No,
			(Conj::Yes, Conj::Yes) => Conj::No,
			(Conj::No, Conj::Yes) => Conj::Yes,
			(Conj::Yes, Conj::No) => Conj::Yes,
		}
	}

	/// returns `Conj::No` if `T` is the canonical representation, otherwise `Conj::Yes`
	#[inline]
	pub const fn get<T: Conjugate>() -> Self {
		if T::IS_CANONICAL { Self::No } else { Self::Yes }
	}

	#[inline]
	pub(crate) fn apply<T: Conjugate>(value: &T) -> T::Canonical {
		let value = unsafe { &*(value as *const T as *const T::Canonical) };

		if try_const! { matches!(Self::get::<T>(), Conj::Yes) } {
			T::Canonical::conj_impl(value)
		} else {
			T::Canonical::copy_impl(value)
		}
	}

	#[inline]
	pub(crate) fn apply_rt<T: ComplexField>(self, value: &T) -> T {
		if self.is_conj() { T::conj_impl(value) } else { T::copy_impl(value) }
	}
}

/// determines the parallelization configuration
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Par {
	/// sequential, non portable across different platforms
	Seq,
	/// parallelized using the global rayon threadpool, non portable across different platforms
	#[cfg(feature = "rayon")]
	Rayon(NonZeroUsize),
}

impl Par {
	/// returns `Par::Rayon(nthreads)` if `nthreads` is non-zero, or
	/// `Par::Rayon(rayon::current_num_threads())` otherwise
	#[inline]
	#[cfg(feature = "rayon")]
	pub fn rayon(nthreads: usize) -> Self {
		if nthreads == 0 {
			Self::Rayon(NonZeroUsize::new(rayon::current_num_threads()).unwrap())
		} else {
			Self::Rayon(NonZeroUsize::new(nthreads).unwrap())
		}
	}

	/// the number of threads that should ideally execute an operation with the given parallelism
	#[inline]
	pub fn degree(&self) -> usize {
		utils::thread::parallelism_degree(*self)
	}
}

#[allow(non_camel_case_types)]
/// `Complex<f32>`
pub type c32 = traits::c32;
#[allow(non_camel_case_types)]
/// `Complex<f64>`
pub type c64 = traits::c64;
#[allow(non_camel_case_types)]
/// `Complex<f64>`
pub type cx128 = traits::cx128;
#[allow(non_camel_case_types)]
/// `Complex<f64>`
pub type fx128 = traits::fx128;

pub use col::{Col, ColMut, ColRef};
pub use mat::{Mat, MatMut, MatRef};
pub use row::{Row, RowMut, RowRef};

#[allow(unused_imports, dead_code)]
mod internal_prelude {
	pub trait DivCeil: Sized {
		fn msrv_div_ceil(self, rhs: Self) -> Self;
		fn msrv_next_multiple_of(self, rhs: Self) -> Self;
		fn msrv_checked_next_multiple_of(self, rhs: Self) -> Option<Self>;
	}

	impl DivCeil for usize {
		#[inline]
		fn msrv_div_ceil(self, rhs: Self) -> Self {
			let d = self / rhs;
			let r = self % rhs;
			if r > 0 { d + 1 } else { d }
		}

		#[inline]
		fn msrv_next_multiple_of(self, rhs: Self) -> Self {
			match self % rhs {
				0 => self,
				r => self + (rhs - r),
			}
		}

		#[inline]
		fn msrv_checked_next_multiple_of(self, rhs: Self) -> Option<Self> {
			{
				match self.checked_rem(rhs)? {
					0 => Some(self),
					r => self.checked_add(rhs - r),
				}
			}
		}
	}

	#[cfg(test)]
	pub(crate) use std::dbg;
	#[cfg(test)]
	pub(crate) use {alloc::boxed::Box, alloc::vec, alloc::vec::Vec};

	pub use faer_traits::{ComplexImpl, ComplexImplConj, Symbolic};

	pub(crate) use crate::col::{Col, ColMut, ColRef};
	pub(crate) use crate::diag::{Diag, DiagMut, DiagRef};
	pub(crate) use crate::hacks::transmute;
	pub(crate) use crate::linalg::{self, temp_mat_scratch, temp_mat_uninit, temp_mat_zeroed};
	pub(crate) use crate::mat::{AsMat, AsMatMut, AsMatRef, Mat, MatMut, MatRef};
	pub(crate) use crate::perm::{Perm, PermRef};
	pub(crate) use crate::prelude::*;
	pub(crate) use crate::row::{AsRowMut, AsRowRef, Row, RowMut, RowRef};
	pub(crate) use crate::utils::bound::{Array, Dim, Idx, IdxInc, MaybeIdx};
	pub(crate) use crate::utils::simd::SimdCtx;
	pub(crate) use crate::{Auto, NonExhaustive, Side, Spec};

	pub use num_complex::Complex;

	pub use faer_macros::math;
	pub use faer_traits::math_utils::*;
	pub use faer_traits::{ComplexField, Conjugate, Index, IndexCore, Real, RealField, SignedIndex, SimdArch};

	#[inline]
	pub fn simd_align(i: usize) -> usize {
		i.wrapping_neg()
	}

	pub(crate) use crate::{Accum, Conj, ContiguousBwd, ContiguousFwd, DiagStatus, Par, Shape, Stride, Unbind, unzip, zip};

	pub use {unzip as uz, zip as z};

	pub use crate::make_guard;
	pub use dyn_stack::{MemStack, StackReq};
	pub use equator::{assert, assert as Assert, debug_assert, debug_assert as DebugAssert};
	pub use reborrow::*;
}

#[allow(unused_imports)]
pub(crate) mod internal_prelude_sp {
	pub(crate) use crate::internal_prelude::*;
	pub(crate) use crate::sparse::{
		FaerError, NONE, Pair, SparseColMat, SparseColMatMut, SparseColMatRef, SparseRowMat, SparseRowMatMut, SparseRowMatRef, SymbolicSparseColMat,
		SymbolicSparseColMatRef, SymbolicSparseRowMat, SymbolicSparseRowMatRef, Triplet, csc_numeric, csc_symbolic, csr_numeric, csr_symbolic,
		linalg as linalg_sp, try_collect, try_zeroed, windows2,
	};
	pub(crate) use core::cell::Cell;
	pub(crate) use core::iter;
	pub(crate) use dyn_stack::MemBuffer;
}

/// useful imports for general usage of the library
pub mod prelude {
	pub use reborrow::{IntoConst, Reborrow, ReborrowMut};

	pub use super::{Par, Scale, c32, c64, col, mat, row, unzip, zip};
	pub use col::{Col, ColMut, ColRef};
	pub use mat::{Mat, MatMut, MatRef};
	pub use row::{Row, RowMut, RowRef};

	#[cfg(feature = "linalg")]
	pub use super::linalg::solvers::{DenseSolve, Solve, SolveLstsq};

	#[cfg(feature = "sparse")]
	pub use super::prelude_sp::*;

	/// see [`Default`]
	#[inline]
	pub fn default<T: Default>() -> T {
		Default::default()
	}
}

#[cfg(feature = "sparse")]
mod prelude_sp {
	use crate::sparse;

	pub use sparse::{SparseColMat, SparseColMatMut, SparseColMatRef, SparseRowMat, SparseRowMatMut, SparseRowMatRef};
}

/// scaling factor for multiplying matrices.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Scale<T>(pub T);
impl<T> Scale<T> {
	/// create a reference to a scaling factor from a reference to a value.
	#[inline(always)]
	pub fn from_ref(value: &T) -> &Self {
		unsafe { &*(value as *const T as *const Self) }
	}

	/// create a mutable reference to a scaling factor from a mutable reference to a value.
	#[inline(always)]
	pub fn from_mut(value: &mut T) -> &mut Self {
		unsafe { &mut *(value as *mut T as *mut Self) }
	}
}

/// 0: disabled
/// 1: `Seq`
/// n >= 2: `Rayon(n - 2)`
///
/// default: `Rayon(0)`
static GLOBAL_PARALLELISM: AtomicUsize = {
	#[cfg(all(not(miri), feature = "rayon"))]
	{
		AtomicUsize::new(2)
	}
	#[cfg(not(all(not(miri), feature = "rayon")))]
	{
		AtomicUsize::new(1)
	}
};

/// causes functions that access global parallelism settings to panic.
pub fn disable_global_parallelism() {
	GLOBAL_PARALLELISM.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// sets the global parallelism settings.
pub fn set_global_parallelism(par: Par) {
	let value = match par {
		Par::Seq => 1,
		#[cfg(feature = "rayon")]
		Par::Rayon(n) => n.get().saturating_add(2),
	};
	GLOBAL_PARALLELISM.store(value, core::sync::atomic::Ordering::Relaxed);
}

/// gets the global parallelism settings.
///
/// # panics
/// panics if global parallelism is disabled.
#[track_caller]
pub fn get_global_parallelism() -> Par {
	let value = GLOBAL_PARALLELISM.load(core::sync::atomic::Ordering::Relaxed);
	match value {
		0 => panic!("Global parallelism is disabled."),
		1 => Par::Seq,
		#[cfg(feature = "rayon")]
		n => Par::rayon(n - 2),
		#[cfg(not(feature = "rayon"))]
		_ => unreachable!(),
	}
}

#[doc(hidden)]
pub mod hacks;

/// statistics and randomness functionality
pub mod stats;

mod non_exhaustive {
	#[doc(hidden)]
	#[derive(Debug, Copy, Clone, PartialEq, Eq)]
	#[repr(transparent)]
	pub struct NonExhaustive(pub(crate) ());
}
pub(crate) use non_exhaustive::NonExhaustive;

/// like `Default`, but with an extra type parameter so that algorithm hyperparameters can be tuned
/// per scalar type.
pub trait Auto<T> {
	/// returns the default value for the type `T`
	fn auto() -> Self;
}

/// implements [`Default`] based on `Config`'s [`Auto`] implementation for the type `T`.
pub struct Spec<Config, T> {
	/// wrapped config value
	pub config: Config,
	__marker: core::marker::PhantomData<fn() -> T>,
}

impl<Config, T> core::ops::Deref for Spec<Config, T> {
	type Target = Config;

	#[inline]
	fn deref(&self) -> &Self::Target {
		&self.config
	}
}

impl<Config, T> core::ops::DerefMut for Spec<Config, T> {
	#[inline]
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.config
	}
}

impl<Config: Copy, T> Copy for Spec<Config, T> {}
impl<Config: Clone, T> Clone for Spec<Config, T> {
	#[inline]
	fn clone(&self) -> Self {
		Self::new(self.config.clone())
	}
}
impl<Config: core::fmt::Debug, T> core::fmt::Debug for Spec<Config, T> {
	#[inline]
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.config.fmt(f)
	}
}

impl<Config, T> Spec<Config, T> {
	/// wraps the given config value
	#[inline]
	pub fn new(config: Config) -> Self {
		Spec {
			config,
			__marker: core::marker::PhantomData,
		}
	}
}

impl<T, Config> From<Config> for Spec<Config, T> {
	#[inline]
	fn from(config: Config) -> Self {
		Spec {
			config,
			__marker: core::marker::PhantomData,
		}
	}
}

impl<T, Config: Auto<T>> Default for Spec<Config, T> {
	#[inline]
	fn default() -> Self {
		Spec {
			config: Auto::<T>::auto(),
			__marker: core::marker::PhantomData,
		}
	}
}

mod into_range {
	use super::*;

	pub trait IntoRange<I> {
		type Len<N: Shape>: Shape;

		fn into_range(self, min: I, max: I) -> core::ops::Range<I>;
	}

	impl<I> IntoRange<I> for core::ops::Range<I> {
		type Len<N: Shape> = usize;

		#[inline]
		fn into_range(self, _: I, _: I) -> core::ops::Range<I> {
			self
		}
	}
	impl<I> IntoRange<I> for core::ops::RangeFrom<I> {
		type Len<N: Shape> = usize;

		#[inline]
		fn into_range(self, _: I, max: I) -> core::ops::Range<I> {
			self.start..max
		}
	}
	impl<I> IntoRange<I> for core::ops::RangeTo<I> {
		type Len<N: Shape> = usize;

		#[inline]
		fn into_range(self, min: I, _: I) -> core::ops::Range<I> {
			min..self.end
		}
	}
	impl<I> IntoRange<I> for core::ops::RangeFull {
		type Len<N: Shape> = N;

		#[inline]
		fn into_range(self, min: I, max: I) -> core::ops::Range<I> {
			min..max
		}
	}
}

mod sort;

pub extern crate dyn_stack;
pub extern crate faer_traits as traits;
pub extern crate reborrow;
