//! `faer` is a linear algebra library for Rust, with a focus on high performance for
//! medium/large matrices.
//!
//! The core module contains the building blocks of linear algebra:
//! * Matrix structure definitions: [`Mat`], [`MatRef`], and [`MatMut`].
//! * Coefficient-wise matrix operations, like addition and subtraction: either using the builtin
//! `+` and `-` operators or using the low level api [`zipped!`].
//! * Matrix multiplication: either using the builtin `*` operator or the low level [`mul`] module.
//! * Triangular matrix solve: the [`solve`] module.
//! * Triangular matrix inverse: the [`inverse`] module.
//! * Householder matrix multiplication: the [`householder`] module.
//!
//! # Example
//! ```
//! use faer_core::{mat, scale, Mat};
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
//! ```
//!
//! # Entity trait
//! Matrices are built on top of the [`Entity`] trait, which describes the prefered memory storage
//! layout for a given type `E`. An entity can be decomposed into a group of units: for a natively
//! supported type ([`f32`], [`f64`], [`c32`], [`c64`]), the unit is simply the type itself, and a
//! group contains a single element. On the other hand, for a type with a more specific preferred
//! layout, like an extended precision floating point type, or a dual number type, the unit would
//! be one of the natively supported types, and the group would be a structure holding the
//! components that build up the full value.
//!
//! To take a more specific example: [`num_complex::Complex<f64>`] has a storage memory layout that
//! differs from that of [`c64`] (see [`complex_native`] for more details). Its real and complex
//! components are stored separately, so its unit type is `f64`, while its group type is `Complex`.
//! In practice, this means that for a `Mat<f64>`, methods such as [`Mat::col_as_slice`] will return
//! a `&[f64]`. Meanwhile, for a `Mat<Complex<f64>>`, [`Mat::col_as_slice`] will return
//! `Complex<&[f64]>`, which holds two slices, each pointing respectively to a view over the real
//! and the imaginary components.
//!
//! While the design of the entity trait is unconventional, it helps us achieve much higher
//! performance when targetting non native types, due to the design matching the typical preffered
//! CPU layout for SIMD operations. And for native types, since [`Group<T>` is just
//! `T`](Entity#impl-Entity-for-f64), the entity layer is a no-op, and the matrix layout is
//! compatible with the classic contiguous layout that's commonly used by other libraries.
//!
//! # Memory allocation
//! Since most `faer` crates aim to expose a low level api for optimal performance, most algorithms
//! try to defer memory allocation to the user.
//!
//! However, since a lot of algorithms need some form of temporary space for intermediate
//! computations, they may ask for a slice of memory for that purpose, by taking a [`stack:
//! PodStack`](dyn_stack::PodStack) parameter. A `PodStack` is a thin wrapper over a slice of
//! memory bytes. This memory may come from any valid source (heap allocation, fixed-size array on
//! the stack, etc.). The functions taking a `PodStack` parameter have a corresponding function
//! with a similar name ending in `_req` that returns the memory requirements of the algorithm. For
//! example:
//! [`householder::apply_block_householder_on_the_left_in_place_with_conj`] and
//! [`householder::apply_block_householder_on_the_left_in_place_req`].
//!
//! The memory stack may be reused in user-code to avoid repeated allocations, and it is also
//! possible to compute the sum ([`dyn_stack::StackReq::all_of`]) or union
//! ([`dyn_stack::StackReq::any_of`]) of multiple requirements, in order to optimally combine them
//! into a single allocation.
//!
//! After computing a [`dyn_stack::StackReq`], one can query its size and alignment to allocate the
//! required memory. The simplest way to do so is through [`dyn_stack::GlobalMemBuffer::new`].

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

use faer_entity::*;
pub use faer_entity::{
    ComplexField, Conjugate, Entity, GroupFor, IdentityGroup, RealField, SimdCtx, SimpleEntity,
};

#[doc(hidden)]
pub use equator::{assert, debug_assert};

pub use faer_entity::pulp;

use coe::Coerce;
use core::{
    fmt::Debug, marker::PhantomData, mem::ManuallyDrop, ptr::NonNull, sync::atomic::AtomicUsize,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use group_helpers::SliceGroup;
use inner::*;
use num_complex::Complex;
use pulp::Simd;
use reborrow::*;

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

#[doc(hidden)]
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
        if r > 0 {
            d + 1
        } else {
            d
        }
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

/// Specifies whether the triangular lower or upper part of a matrix should be accessed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Side {
    Lower,
    Upper,
}

extern crate alloc;

pub mod householder;
#[doc(hidden)]
pub mod jacobi;

pub mod inverse;
pub mod mul;
pub mod permutation;
pub mod solve;

pub mod matrix_ops;

/// Thin wrapper used for scalar multiplication of a matrix by a scalar value.
pub use matrix_ops::scale;

#[doc(hidden)]
pub mod simd;

#[doc(hidden)]
pub use faer_entity::transmute_unchecked;

pub mod complex_native;
pub use complex_native::*;

/// Whether a matrix should be implicitly conjugated when read or not.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Conj {
    Yes,
    No,
}

impl Conj {
    #[inline]
    pub fn compose(self, other: Conj) -> Conj {
        if self == other {
            Conj::No
        } else {
            Conj::Yes
        }
    }
}

/// Trait for types that can be converted to a row view.
pub trait AsRowRef<E: Entity> {
    fn as_row_ref(&self) -> RowRef<'_, E>;
}
/// Trait for types that can be converted to a mutable row view.
pub trait AsRowMut<E: Entity> {
    fn as_row_mut(&mut self) -> RowMut<'_, E>;
}

/// Trait for types that can be converted to a column view.
pub trait AsColRef<E: Entity> {
    fn as_col_ref(&self) -> ColRef<'_, E>;
}
/// Trait for types that can be converted to a mutable col view.
pub trait AsColMut<E: Entity> {
    fn as_col_mut(&mut self) -> ColMut<'_, E>;
}

/// Trait for types that can be converted to a matrix view.
pub trait AsMatRef<E: Entity> {
    fn as_mat_ref(&self) -> MatRef<'_, E>;
}
/// Trait for types that can be converted to a mutable matrix view.
pub trait AsMatMut<E: Entity> {
    fn as_mat_mut(&mut self) -> MatMut<'_, E>;
}

const __AS_COL: () = {
    impl<E: Entity> AsColRef<E> for ColRef<'_, E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            *self
        }
    }
    impl<E: Entity> AsColRef<E> for &'_ ColRef<'_, E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            **self
        }
    }
    impl<E: Entity> AsColRef<E> for ColMut<'_, E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            (*self).rb()
        }
    }
    impl<E: Entity> AsColRef<E> for &'_ ColMut<'_, E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            (**self).rb()
        }
    }
    impl<E: Entity> AsColRef<E> for Col<E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            (*self).as_ref()
        }
    }
    impl<E: Entity> AsColRef<E> for &'_ Col<E> {
        #[inline]
        fn as_col_ref(&self) -> ColRef<'_, E> {
            (**self).as_ref()
        }
    }

    impl<E: Entity> AsColMut<E> for ColMut<'_, E> {
        #[inline]
        fn as_col_mut(&mut self) -> ColMut<'_, E> {
            (*self).rb_mut()
        }
    }

    impl<E: Entity> AsColMut<E> for &'_ mut ColMut<'_, E> {
        #[inline]
        fn as_col_mut(&mut self) -> ColMut<'_, E> {
            (**self).rb_mut()
        }
    }

    impl<E: Entity> AsColMut<E> for Col<E> {
        #[inline]
        fn as_col_mut(&mut self) -> ColMut<'_, E> {
            (*self).as_mut()
        }
    }

    impl<E: Entity> AsColMut<E> for &'_ mut Col<E> {
        #[inline]
        fn as_col_mut(&mut self) -> ColMut<'_, E> {
            (**self).as_mut()
        }
    }
};

const __AS_ROW: () = {
    impl<E: Entity> AsRowRef<E> for RowRef<'_, E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            *self
        }
    }
    impl<E: Entity> AsRowRef<E> for &'_ RowRef<'_, E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            **self
        }
    }
    impl<E: Entity> AsRowRef<E> for RowMut<'_, E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            (*self).rb()
        }
    }
    impl<E: Entity> AsRowRef<E> for &'_ RowMut<'_, E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            (**self).rb()
        }
    }
    impl<E: Entity> AsRowRef<E> for Row<E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            (*self).as_ref()
        }
    }
    impl<E: Entity> AsRowRef<E> for &'_ Row<E> {
        #[inline]
        fn as_row_ref(&self) -> RowRef<'_, E> {
            (**self).as_ref()
        }
    }

    impl<E: Entity> AsRowMut<E> for RowMut<'_, E> {
        #[inline]
        fn as_row_mut(&mut self) -> RowMut<'_, E> {
            (*self).rb_mut()
        }
    }

    impl<E: Entity> AsRowMut<E> for &'_ mut RowMut<'_, E> {
        #[inline]
        fn as_row_mut(&mut self) -> RowMut<'_, E> {
            (**self).rb_mut()
        }
    }

    impl<E: Entity> AsRowMut<E> for Row<E> {
        #[inline]
        fn as_row_mut(&mut self) -> RowMut<'_, E> {
            (*self).as_mut()
        }
    }

    impl<E: Entity> AsRowMut<E> for &'_ mut Row<E> {
        #[inline]
        fn as_row_mut(&mut self) -> RowMut<'_, E> {
            (**self).as_mut()
        }
    }
};

const __AS_MAT: () = {
    impl<E: Entity> AsMatRef<E> for MatRef<'_, E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            *self
        }
    }
    impl<E: Entity> AsMatRef<E> for &'_ MatRef<'_, E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            **self
        }
    }
    impl<E: Entity> AsMatRef<E> for MatMut<'_, E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            (*self).rb()
        }
    }
    impl<E: Entity> AsMatRef<E> for &'_ MatMut<'_, E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            (**self).rb()
        }
    }
    impl<E: Entity> AsMatRef<E> for Mat<E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            (*self).as_ref()
        }
    }
    impl<E: Entity> AsMatRef<E> for &'_ Mat<E> {
        #[inline]
        fn as_mat_ref(&self) -> MatRef<'_, E> {
            (**self).as_ref()
        }
    }

    impl<E: Entity> AsMatMut<E> for MatMut<'_, E> {
        #[inline]
        fn as_mat_mut(&mut self) -> MatMut<'_, E> {
            (*self).rb_mut()
        }
    }

    impl<E: Entity> AsMatMut<E> for &'_ mut MatMut<'_, E> {
        #[inline]
        fn as_mat_mut(&mut self) -> MatMut<'_, E> {
            (**self).rb_mut()
        }
    }

    impl<E: Entity> AsMatMut<E> for Mat<E> {
        #[inline]
        fn as_mat_mut(&mut self) -> MatMut<'_, E> {
            (*self).as_mut()
        }
    }

    impl<E: Entity> AsMatMut<E> for &'_ mut Mat<E> {
        #[inline]
        fn as_mat_mut(&mut self) -> MatMut<'_, E> {
            (**self).as_mut()
        }
    }
};

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::Matrix<E> for MatRef<'_, E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Dense(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::DenseAccess<E> for MatRef<'_, E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::Matrix<E> for MatMut<'_, E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Dense(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::DenseAccess<E> for MatMut<'_, E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::Matrix<E> for Mat<E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Dense(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::DenseAccess<E> for Mat<E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

#[repr(C)]
struct VecImpl<E: Entity> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    len: usize,
    stride: isize,
}
#[repr(C)]
struct VecOwnImpl<E: Entity> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    len: usize,
}

#[repr(C)]
struct MatImpl<E: Entity> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
}
#[repr(C)]
struct MatOwnImpl<E: Entity> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    nrows: usize,
    ncols: usize,
}

impl<E: Entity> Copy for VecImpl<E> {}
impl<E: Entity> Clone for VecImpl<E> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Entity> Copy for MatImpl<E> {}
impl<E: Entity> Clone for MatImpl<E> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

/// Generic matrix container.
#[derive(Copy, Clone)]
pub struct Matrix<M> {
    inner: M,
}

/// Specialized containers that are used with [`Matrix`].
pub mod inner {
    use super::*;

    impl<E: Entity> Copy for DiagRef<'_, E> {}
    impl<E: Entity> Clone for DiagRef<'_, E> {
        #[inline(always)]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<E: Entity> Copy for DenseRowRef<'_, E> {}
    impl<E: Entity> Clone for DenseRowRef<'_, E> {
        #[inline(always)]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<E: Entity> Copy for DenseColRef<'_, E> {}
    impl<E: Entity> Clone for DenseColRef<'_, E> {
        #[inline(always)]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<E: Entity> Copy for DenseRef<'_, E> {}
    impl<E: Entity> Clone for DenseRef<'_, E> {
        #[inline(always)]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<I, E: Entity> Copy for PermRef<'_, I, E> {}
    impl<I, E: Entity> Clone for PermRef<'_, I, E> {
        #[inline(always)]
        fn clone(&self) -> Self {
            *self
        }
    }

    #[repr(C)]
    #[derive(Debug)]
    pub struct PermRef<'a, I, E: Entity> {
        pub(crate) forward: &'a [I],
        pub(crate) inverse: &'a [I],
        pub(crate) __marker: PhantomData<E>,
    }
    #[repr(C)]
    #[derive(Debug)]
    pub struct PermMut<'a, I, E: Entity> {
        pub(crate) forward: &'a mut [I],
        pub(crate) inverse: &'a mut [I],
        pub(crate) __marker: PhantomData<E>,
    }
    #[repr(C)]
    #[derive(Debug)]
    pub struct PermOwn<I, E: Entity> {
        pub(crate) forward: alloc::boxed::Box<[I]>,
        pub(crate) inverse: alloc::boxed::Box<[I]>,
        pub(crate) __marker: PhantomData<E>,
    }

    #[repr(C)]
    pub struct DiagRef<'a, E: Entity> {
        pub(crate) inner: ColRef<'a, E>,
    }

    #[repr(C)]
    pub struct DiagMut<'a, E: Entity> {
        pub(crate) inner: ColMut<'a, E>,
    }

    #[repr(C)]
    pub struct DiagOwn<E: Entity> {
        pub(crate) inner: Col<E>,
    }

    #[repr(C)]
    pub struct DenseColRef<'a, E: Entity> {
        pub(crate) inner: VecImpl<E>,
        pub(crate) __marker: PhantomData<&'a E>,
    }

    #[repr(C)]
    pub struct DenseColMut<'a, E: Entity> {
        pub(crate) inner: VecImpl<E>,
        pub(crate) __marker: PhantomData<&'a mut E>,
    }

    #[repr(C)]
    pub struct DenseColOwn<E: Entity> {
        pub(crate) inner: VecOwnImpl<E>,
        pub(crate) row_capacity: usize,
    }

    #[repr(C)]
    pub struct DenseRowRef<'a, E: Entity> {
        pub(crate) inner: VecImpl<E>,
        pub(crate) __marker: PhantomData<&'a E>,
    }

    #[repr(C)]
    pub struct DenseRowMut<'a, E: Entity> {
        pub(crate) inner: VecImpl<E>,
        pub(crate) __marker: PhantomData<&'a mut E>,
    }

    #[repr(C)]
    pub struct DenseRowOwn<E: Entity> {
        pub(crate) inner: VecOwnImpl<E>,
        pub(crate) col_capacity: usize,
    }

    #[repr(C)]
    pub struct DenseRef<'a, E: Entity> {
        pub(crate) inner: MatImpl<E>,
        pub(crate) __marker: PhantomData<&'a E>,
    }

    #[repr(C)]
    pub struct DenseMut<'a, E: Entity> {
        pub(crate) inner: MatImpl<E>,
        pub(crate) __marker: PhantomData<&'a mut E>,
    }

    #[repr(C)]
    pub struct DenseOwn<E: Entity> {
        pub(crate) inner: MatOwnImpl<E>,
        pub(crate) row_capacity: usize,
        pub(crate) col_capacity: usize,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    pub struct Scale<E: Entity>(pub E);

    #[derive(Debug)]
    #[doc(hidden)]
    pub struct SparseColMatRefInner<'a, I, E: Entity> {
        pub(crate) symbolic: sparse::SymbolicSparseColMatRef<'a, I>,
        pub(crate) values: SliceGroup<'a, E>,
    }

    #[derive(Debug)]
    #[doc(hidden)]
    pub struct SparseRowMatRefInner<'a, I, E: Entity> {
        pub(crate) symbolic: sparse::SymbolicSparseRowMatRef<'a, I>,
        pub(crate) values: SliceGroup<'a, E>,
    }

    impl<I, E: Entity> Copy for SparseRowMatRefInner<'_, I, E> {}
    impl<I, E: Entity> Clone for SparseRowMatRefInner<'_, I, E> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<I, E: Entity> Copy for SparseColMatRefInner<'_, I, E> {}
    impl<I, E: Entity> Clone for SparseColMatRefInner<'_, I, E> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }
}

/// Advanced: Helper types for working with [`GroupFor`] in generic contexts.
pub mod group_helpers {
    pub use pulp::{Read, Write};

    pub struct VecGroup<E: Entity, T = UnitFor<E>> {
        inner: GroupFor<E, alloc::vec::Vec<T>>,
    }

    impl<E: Entity, T: Clone> Clone for VecGroup<E, T> {
        #[inline]
        fn clone(&self) -> Self {
            Self {
                inner: E::faer_map(E::faer_as_ref(&self.inner), |v| (*v).clone()),
            }
        }
    }

    impl<E: Entity, T: Debug> Debug for VecGroup<E, T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.as_slice().fmt(f)
        }
    }

    unsafe impl<E: Entity, T: Sync> Sync for VecGroup<E, T> {}
    unsafe impl<E: Entity, T: Send> Send for VecGroup<E, T> {}

    impl<E: Entity, T> VecGroup<E, T> {
        #[inline]
        pub fn from_inner(inner: GroupFor<E, alloc::vec::Vec<T>>) -> Self {
            Self { inner }
        }

        #[inline]
        pub fn as_inner_ref(&self) -> GroupFor<E, &alloc::vec::Vec<T>> {
            E::faer_as_ref(&self.inner)
        }

        #[inline]
        pub fn as_inner_mut(&mut self) -> GroupFor<E, &mut alloc::vec::Vec<T>> {
            E::faer_as_mut(&mut self.inner)
        }

        #[inline]
        pub fn as_slice(&self) -> SliceGroup<'_, E, T> {
            SliceGroup::new(E::faer_map(
                E::faer_as_ref(&self.inner),
                #[inline]
                |slice| &**slice,
            ))
        }

        #[inline]
        pub fn as_slice_mut(&mut self) -> SliceGroupMut<'_, E, T> {
            SliceGroupMut::new(E::faer_map(
                E::faer_as_mut(&mut self.inner),
                #[inline]
                |slice| &mut **slice,
            ))
        }

        #[inline]
        pub fn new() -> Self {
            Self {
                inner: E::faer_map(E::UNIT, |()| alloc::vec::Vec::new()),
            }
        }

        #[inline]
        pub fn len(&self) -> usize {
            let mut len = usize::MAX;
            E::faer_map(
                E::faer_as_ref(&self.inner),
                #[inline(always)]
                |slice| len = Ord::min(len, slice.len()),
            );
            len
        }

        #[inline]
        pub fn capacity(&self) -> usize {
            let mut cap = usize::MAX;
            E::faer_map(
                E::faer_as_ref(&self.inner),
                #[inline(always)]
                |slice| cap = Ord::min(cap, slice.capacity()),
            );
            cap
        }

        pub fn reserve(&mut self, additional: usize) {
            E::faer_map(E::faer_as_mut(&mut self.inner), |v| v.reserve(additional));
        }

        pub fn reserve_exact(&mut self, additional: usize) {
            E::faer_map(E::faer_as_mut(&mut self.inner), |v| {
                v.reserve_exact(additional)
            });
        }

        pub fn try_reserve(
            &mut self,
            additional: usize,
        ) -> Result<(), alloc::collections::TryReserveError> {
            let mut result = Ok(());
            E::faer_map(E::faer_as_mut(&mut self.inner), |v| match &result {
                Ok(()) => result = v.try_reserve(additional),
                Err(_) => {}
            });
            result
        }

        pub fn try_reserve_exact(
            &mut self,
            additional: usize,
        ) -> Result<(), alloc::collections::TryReserveError> {
            let mut result = Ok(());
            E::faer_map(E::faer_as_mut(&mut self.inner), |v| match &result {
                Ok(()) => result = v.try_reserve_exact(additional),
                Err(_) => {}
            });
            result
        }

        pub fn truncate(&mut self, len: usize) {
            E::faer_map(E::faer_as_mut(&mut self.inner), |v| v.truncate(len));
        }

        pub fn clear(&mut self) {
            E::faer_map(E::faer_as_mut(&mut self.inner), |v| v.clear());
        }

        pub fn resize(&mut self, new_len: usize, value: GroupFor<E, T>)
        where
            T: Clone,
        {
            E::faer_map(
                E::faer_zip(E::faer_as_mut(&mut self.inner), value),
                |(v, value)| v.resize(new_len, value),
            );
        }

        pub fn resize_with(&mut self, new_len: usize, f: impl FnMut() -> GroupFor<E, T>) {
            let len = self.len();
            let mut f = f;
            if new_len <= len {
                self.truncate(new_len);
            } else {
                self.reserve(new_len - len);
                for _ in len..new_len {
                    self.push(f())
                }
            }
        }

        #[inline]
        pub fn push(&mut self, value: GroupFor<E, T>) {
            E::faer_map(
                E::faer_zip(E::faer_as_mut(&mut self.inner), value),
                #[inline]
                |(v, value)| v.push(value),
            );
        }

        #[inline]
        pub fn pop(&mut self) -> Option<GroupFor<E, T>> {
            if self.len() >= 1 {
                Some(E::faer_map(
                    E::faer_as_mut(&mut self.inner),
                    #[inline]
                    |v| v.pop().unwrap(),
                ))
            } else {
                None
            }
        }

        #[inline]
        pub fn remove(&mut self, index: usize) -> GroupFor<E, T> {
            E::faer_map(
                E::faer_as_mut(&mut self.inner),
                #[inline]
                |v| v.remove(index),
            )
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub struct YesConj;
    #[derive(Copy, Clone, Debug)]
    pub struct NoConj;

    pub trait ConjTy: Copy + Debug {
        const CONJ: Conj;
        type Flip: ConjTy;

        fn flip(self) -> Self::Flip;
    }

    impl ConjTy for YesConj {
        const CONJ: Conj = Conj::Yes;
        type Flip = NoConj;
        #[inline(always)]
        fn flip(self) -> Self::Flip {
            NoConj
        }
    }
    impl ConjTy for NoConj {
        const CONJ: Conj = Conj::No;
        type Flip = YesConj;
        #[inline(always)]
        fn flip(self) -> Self::Flip {
            YesConj
        }
    }

    use super::*;
    use crate::{assert, debug_assert};
    use core::ops::Range;

    pub struct SimdFor<E: Entity, S: pulp::Simd> {
        pub simd: S,
        __marker: PhantomData<E>,
    }

    impl<E: Entity, S: pulp::Simd> Copy for SimdFor<E, S> {}
    impl<E: Entity, S: pulp::Simd> Clone for SimdFor<E, S> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<E: ComplexField, S: pulp::Simd> SimdFor<E, S> {
        #[inline(always)]
        pub fn new(simd: S) -> Self {
            Self {
                simd,
                __marker: PhantomData,
            }
        }

        #[inline(always)]
        pub fn as_simd(
            self,
            slice: SliceGroup<'_, E>,
        ) -> (SliceGroup<'_, E, SimdUnitFor<E, S>>, SliceGroup<'_, E>) {
            let (head, tail) = slice_as_simd::<E, S>(slice.into_inner());
            (SliceGroup::new(head), SliceGroup::new(tail))
        }

        #[inline(always)]
        pub fn align_offset(self, slice: SliceGroup<'_, E>) -> pulp::Offset<E::SimdMask<S>> {
            let slice = E::faer_first(slice.into_inner());
            E::faer_align_offset(self.simd, slice.as_ptr(), slice.len())
        }

        #[inline(always)]
        pub fn align_offset_ptr(
            self,
            ptr: GroupFor<E, *const E::Unit>,
            len: usize,
        ) -> pulp::Offset<E::SimdMask<S>> {
            E::faer_align_offset(self.simd, E::faer_first(ptr), len)
        }

        #[inline(always)]
        pub fn as_simd_mut(
            self,
            slice: SliceGroupMut<'_, E>,
        ) -> (
            SliceGroupMut<'_, E, SimdUnitFor<E, S>>,
            SliceGroupMut<'_, E>,
        ) {
            let (head, tail) = slice_as_mut_simd::<E, S>(slice.into_inner());
            (SliceGroupMut::new(head), SliceGroupMut::new(tail))
        }

        #[inline(always)]
        pub fn as_aligned_simd(
            self,
            slice: SliceGroup<'_, E>,
            offset: pulp::Offset<E::SimdMask<S>>,
        ) -> (
            Prefix<'_, E, S>,
            SliceGroup<'_, E, SimdUnitFor<E, S>>,
            Suffix<'_, E, S>,
        ) {
            let (head_tail, body) = E::faer_unzip(E::faer_map(slice.into_inner(), |slice| {
                let (head, body, tail) = E::faer_slice_as_aligned_simd(self.simd, slice, offset);
                ((head, tail), body)
            }));

            let (head, tail) = E::faer_unzip(head_tail);

            unsafe {
                (
                    Prefix(
                        transmute_unchecked::<
                            GroupCopyFor<E, E::PrefixUnit<'_, S>>,
                            GroupCopyFor<E, E::PrefixUnit<'static, S>>,
                        >(into_copy::<E, _>(head)),
                        PhantomData,
                    ),
                    SliceGroup::new(body),
                    Suffix(
                        transmute_unchecked::<
                            GroupCopyFor<E, E::SuffixUnit<'_, S>>,
                            GroupCopyFor<E, E::SuffixUnit<'static, S>>,
                        >(into_copy::<E, _>(tail)),
                        PhantomData,
                    ),
                )
            }
        }

        #[inline(always)]
        pub fn as_aligned_simd_mut(
            self,
            slice: SliceGroupMut<'_, E>,
            offset: pulp::Offset<E::SimdMask<S>>,
        ) -> (
            PrefixMut<'_, E, S>,
            SliceGroupMut<'_, E, SimdUnitFor<E, S>>,
            SuffixMut<'_, E, S>,
        ) {
            let (head_tail, body) = E::faer_unzip(E::faer_map(slice.into_inner(), |slice| {
                let (head, body, tail) =
                    E::faer_slice_as_aligned_simd_mut(self.simd, slice, offset);
                ((head, tail), body)
            }));

            let (head, tail) = E::faer_unzip(head_tail);

            (
                PrefixMut(
                    unsafe {
                        transmute_unchecked::<
                            GroupFor<E, E::PrefixMutUnit<'_, S>>,
                            GroupFor<E, E::PrefixMutUnit<'static, S>>,
                        >(head)
                    },
                    PhantomData,
                ),
                SliceGroupMut::new(body),
                SuffixMut(
                    unsafe {
                        transmute_unchecked::<
                            GroupFor<E, E::SuffixMutUnit<'_, S>>,
                            GroupFor<E, E::SuffixMutUnit<'static, S>>,
                        >(tail)
                    },
                    PhantomData,
                ),
            )
        }

        #[inline(always)]
        pub fn splat(self, value: E) -> SimdGroupFor<E, S> {
            E::faer_simd_splat(self.simd, value)
        }

        #[inline(always)]
        pub fn scalar_mul(self, lhs: E, rhs: E) -> E {
            E::faer_simd_scalar_mul(self.simd, lhs, rhs)
        }
        #[inline(always)]
        pub fn scalar_conj_mul(self, lhs: E, rhs: E) -> E {
            E::faer_simd_scalar_conj_mul(self.simd, lhs, rhs)
        }
        #[inline(always)]
        pub fn scalar_mul_add_e(self, lhs: E, rhs: E, acc: E) -> E {
            E::faer_simd_scalar_mul_adde(self.simd, lhs, rhs, acc)
        }
        #[inline(always)]
        pub fn scalar_conj_mul_add_e(self, lhs: E, rhs: E, acc: E) -> E {
            E::faer_simd_scalar_conj_mul_adde(self.simd, lhs, rhs, acc)
        }

        #[inline(always)]
        pub fn scalar_conditional_conj_mul<C: ConjTy>(self, conj: C, lhs: E, rhs: E) -> E {
            let _ = conj;
            if C::CONJ == Conj::Yes {
                self.scalar_conj_mul(lhs, rhs)
            } else {
                self.scalar_mul(lhs, rhs)
            }
        }
        #[inline(always)]
        pub fn scalar_conditional_conj_mul_add_e<C: ConjTy>(
            self,
            conj: C,
            lhs: E,
            rhs: E,
            acc: E,
        ) -> E {
            let _ = conj;
            if C::CONJ == Conj::Yes {
                self.scalar_conj_mul_add_e(lhs, rhs, acc)
            } else {
                self.scalar_mul_add_e(lhs, rhs, acc)
            }
        }

        #[inline(always)]
        pub fn add(self, lhs: SimdGroupFor<E, S>, rhs: SimdGroupFor<E, S>) -> SimdGroupFor<E, S> {
            E::faer_simd_add(self.simd, lhs, rhs)
        }
        #[inline(always)]
        pub fn sub(self, lhs: SimdGroupFor<E, S>, rhs: SimdGroupFor<E, S>) -> SimdGroupFor<E, S> {
            E::faer_simd_sub(self.simd, lhs, rhs)
        }
        #[inline(always)]
        pub fn neg(self, a: SimdGroupFor<E, S>) -> SimdGroupFor<E, S> {
            E::faer_simd_neg(self.simd, a)
        }
        #[inline(always)]
        pub fn scale_real(
            self,
            lhs: SimdGroupFor<E::Real, S>,
            rhs: SimdGroupFor<E, S>,
        ) -> SimdGroupFor<E, S> {
            E::faer_simd_scale_real(self.simd, lhs, rhs)
        }
        #[inline(always)]
        pub fn mul(self, lhs: SimdGroupFor<E, S>, rhs: SimdGroupFor<E, S>) -> SimdGroupFor<E, S> {
            E::faer_simd_mul(self.simd, lhs, rhs)
        }
        #[inline(always)]
        pub fn conj_mul(
            self,
            lhs: SimdGroupFor<E, S>,
            rhs: SimdGroupFor<E, S>,
        ) -> SimdGroupFor<E, S> {
            E::faer_simd_conj_mul(self.simd, lhs, rhs)
        }
        #[inline(always)]
        pub fn conditional_conj_mul<C: ConjTy>(
            self,
            conj: C,
            lhs: SimdGroupFor<E, S>,
            rhs: SimdGroupFor<E, S>,
        ) -> SimdGroupFor<E, S> {
            let _ = conj;
            if C::CONJ == Conj::Yes {
                self.conj_mul(lhs, rhs)
            } else {
                self.mul(lhs, rhs)
            }
        }

        #[inline(always)]
        pub fn mul_add_e(
            self,
            lhs: SimdGroupFor<E, S>,
            rhs: SimdGroupFor<E, S>,
            acc: SimdGroupFor<E, S>,
        ) -> SimdGroupFor<E, S> {
            E::faer_simd_mul_adde(self.simd, lhs, rhs, acc)
        }
        #[inline(always)]
        pub fn conj_mul_add_e(
            self,
            lhs: SimdGroupFor<E, S>,
            rhs: SimdGroupFor<E, S>,
            acc: SimdGroupFor<E, S>,
        ) -> SimdGroupFor<E, S> {
            E::faer_simd_conj_mul_adde(self.simd, lhs, rhs, acc)
        }
        #[inline(always)]
        pub fn conditional_conj_mul_add_e<C: ConjTy>(
            self,
            conj: C,
            lhs: SimdGroupFor<E, S>,
            rhs: SimdGroupFor<E, S>,
            acc: SimdGroupFor<E, S>,
        ) -> SimdGroupFor<E, S> {
            let _ = conj;
            if C::CONJ == Conj::Yes {
                self.conj_mul_add_e(lhs, rhs, acc)
            } else {
                self.mul_add_e(lhs, rhs, acc)
            }
        }

        #[inline(always)]
        pub fn abs2_add_e(
            self,
            values: SimdGroupFor<E, S>,
            acc: SimdGroupFor<E::Real, S>,
        ) -> SimdGroupFor<E::Real, S> {
            E::faer_simd_abs2_adde(self.simd, values, acc)
        }
        #[inline(always)]
        pub fn abs2(self, values: SimdGroupFor<E, S>) -> SimdGroupFor<E::Real, S> {
            E::faer_simd_abs2(self.simd, values)
        }
        #[inline(always)]
        pub fn score(self, values: SimdGroupFor<E, S>) -> SimdGroupFor<E::Real, S> {
            E::faer_simd_score(self.simd, values)
        }

        #[inline(always)]
        pub fn reduce_add(self, values: SimdGroupFor<E, S>) -> E {
            E::faer_simd_reduce_add(self.simd, values)
        }

        #[inline(always)]
        pub fn rotate_left(self, values: SimdGroupFor<E, S>, amount: usize) -> SimdGroupFor<E, S> {
            E::faer_simd_rotate_left(self.simd, values, amount)
        }
    }

    impl<E: RealField, S: pulp::Simd> SimdFor<E, S> {
        #[inline(always)]
        pub fn abs(self, values: SimdGroupFor<E, S>) -> SimdGroupFor<E::Real, S> {
            E::faer_simd_abs(self.simd, values)
        }
        #[inline(always)]
        pub fn less_than(self, a: SimdGroupFor<E, S>, b: SimdGroupFor<E, S>) -> SimdMaskFor<E, S> {
            E::faer_simd_less_than(self.simd, a, b)
        }
        #[inline(always)]
        pub fn less_than_or_equal(
            self,
            a: SimdGroupFor<E, S>,
            b: SimdGroupFor<E, S>,
        ) -> SimdMaskFor<E, S> {
            E::faer_simd_less_than_or_equal(self.simd, a, b)
        }
        #[inline(always)]
        pub fn greater_than(
            self,
            a: SimdGroupFor<E, S>,
            b: SimdGroupFor<E, S>,
        ) -> SimdMaskFor<E, S> {
            E::faer_simd_greater_than(self.simd, a, b)
        }
        #[inline(always)]
        pub fn greater_than_or_equal(
            self,
            a: SimdGroupFor<E, S>,
            b: SimdGroupFor<E, S>,
        ) -> SimdMaskFor<E, S> {
            E::faer_simd_greater_than_or_equal(self.simd, a, b)
        }

        #[inline(always)]
        pub fn select(
            self,
            mask: SimdMaskFor<E, S>,
            if_true: SimdGroupFor<E, S>,
            if_false: SimdGroupFor<E, S>,
        ) -> SimdGroupFor<E, S> {
            E::faer_simd_select(self.simd, mask, if_true, if_false)
        }
        #[inline(always)]
        pub fn index_select(
            self,
            mask: SimdMaskFor<E, S>,
            if_true: SimdIndexFor<E, S>,
            if_false: SimdIndexFor<E, S>,
        ) -> SimdIndexFor<E, S> {
            E::faer_simd_index_select(self.simd, mask, if_true, if_false)
        }
        #[inline(always)]
        pub fn index_seq(self) -> SimdIndexFor<E, S> {
            E::faer_simd_index_seq(self.simd)
        }
        #[inline(always)]
        pub fn index_splat(self, value: IndexFor<E>) -> SimdIndexFor<E, S> {
            E::faer_simd_index_splat(self.simd, value)
        }
        #[inline(always)]
        pub fn index_add(self, a: SimdIndexFor<E, S>, b: SimdIndexFor<E, S>) -> SimdIndexFor<E, S> {
            E::faer_simd_index_add(self.simd, a, b)
        }
    }

    pub struct SliceGroup<'a, E: Entity, T: 'a = <E as Entity>::Unit>(
        GroupCopyFor<E, *const [T]>,
        PhantomData<&'a ()>,
    );
    pub struct SliceGroupMut<'a, E: Entity, T: 'a = <E as Entity>::Unit>(
        GroupFor<E, *mut [T]>,
        PhantomData<&'a mut ()>,
    );

    pub struct Prefix<'a, E: Entity, S: pulp::Simd>(
        GroupCopyFor<E, E::PrefixUnit<'static, S>>,
        PhantomData<&'a ()>,
    );
    pub struct Suffix<'a, E: Entity, S: pulp::Simd>(
        GroupCopyFor<E, E::SuffixUnit<'static, S>>,
        PhantomData<&'a mut ()>,
    );
    pub struct PrefixMut<'a, E: Entity, S: pulp::Simd>(
        GroupFor<E, E::PrefixMutUnit<'static, S>>,
        PhantomData<&'a ()>,
    );
    pub struct SuffixMut<'a, E: Entity, S: pulp::Simd>(
        GroupFor<E, E::SuffixMutUnit<'static, S>>,
        PhantomData<&'a mut ()>,
    );

    impl<E: Entity, T: Copy + Debug> Read for RefGroupMut<'_, E, T> {
        type Output = GroupCopyFor<E, T>;
        #[inline(always)]
        fn read_or(&self, _or: Self::Output) -> Self::Output {
            self.get()
        }
    }
    impl<E: Entity, T: Copy + Debug> Write for RefGroupMut<'_, E, T> {
        #[inline(always)]
        fn write(&mut self, values: Self::Output) {
            self.set(values)
        }
    }
    impl<E: Entity, T: Copy + Debug> Read for RefGroup<'_, E, T> {
        type Output = GroupCopyFor<E, T>;
        #[inline(always)]
        fn read_or(&self, _or: Self::Output) -> Self::Output {
            self.get()
        }
    }

    impl<E: Entity, S: pulp::Simd> Read for Prefix<'_, E, S> {
        type Output = SimdGroupFor<E, S>;
        #[inline(always)]
        fn read_or(&self, or: Self::Output) -> Self::Output {
            into_copy::<E, _>(E::faer_map(
                E::faer_zip(from_copy::<E, _>(self.0), from_copy::<E, _>(or)),
                #[inline(always)]
                |(prefix, or)| prefix.read_or(or),
            ))
        }
    }
    impl<E: Entity, S: pulp::Simd> Read for PrefixMut<'_, E, S> {
        type Output = SimdGroupFor<E, S>;
        #[inline(always)]
        fn read_or(&self, or: Self::Output) -> Self::Output {
            self.rb().read_or(or)
        }
    }
    impl<E: Entity, S: pulp::Simd> Write for PrefixMut<'_, E, S> {
        #[inline(always)]
        fn write(&mut self, values: Self::Output) {
            E::faer_map(
                E::faer_zip(self.rb_mut().0, from_copy::<E, _>(values)),
                #[inline(always)]
                |(mut prefix, values)| prefix.write(values),
            );
        }
    }

    impl<E: Entity, S: pulp::Simd> Read for Suffix<'_, E, S> {
        type Output = SimdGroupFor<E, S>;
        #[inline(always)]
        fn read_or(&self, or: Self::Output) -> Self::Output {
            into_copy::<E, _>(E::faer_map(
                E::faer_zip(from_copy::<E, _>(self.0), from_copy::<E, _>(or)),
                #[inline(always)]
                |(suffix, or)| suffix.read_or(or),
            ))
        }
    }
    impl<E: Entity, S: pulp::Simd> Read for SuffixMut<'_, E, S> {
        type Output = SimdGroupFor<E, S>;
        #[inline(always)]
        fn read_or(&self, or: Self::Output) -> Self::Output {
            self.rb().read_or(or)
        }
    }
    impl<E: Entity, S: pulp::Simd> Write for SuffixMut<'_, E, S> {
        #[inline(always)]
        fn write(&mut self, values: Self::Output) {
            E::faer_map(
                E::faer_zip(self.rb_mut().0, from_copy::<E, _>(values)),
                #[inline(always)]
                |(mut suffix, values)| suffix.write(values),
            );
        }
    }

    impl<'short, E: Entity, S: pulp::Simd> Reborrow<'short> for PrefixMut<'_, E, S> {
        type Target = Prefix<'short, E, S>;
        #[inline]
        fn rb(&'short self) -> Self::Target {
            unsafe {
                Prefix(
                    into_copy::<E, _>(transmute_unchecked::<
                        GroupFor<E, <E::PrefixMutUnit<'static, S> as Reborrow<'_>>::Target>,
                        GroupFor<E, E::PrefixUnit<'static, S>>,
                    >(E::faer_map(
                        E::faer_as_ref(&self.0),
                        |x| (*x).rb(),
                    ))),
                    PhantomData,
                )
            }
        }
    }
    impl<'short, E: Entity, S: pulp::Simd> ReborrowMut<'short> for PrefixMut<'_, E, S> {
        type Target = PrefixMut<'short, E, S>;
        #[inline]
        fn rb_mut(&'short mut self) -> Self::Target {
            unsafe {
                PrefixMut(
                    transmute_unchecked::<
                        GroupFor<E, <E::PrefixMutUnit<'static, S> as ReborrowMut<'_>>::Target>,
                        GroupFor<E, E::PrefixMutUnit<'static, S>>,
                    >(E::faer_map(E::faer_as_mut(&mut self.0), |x| {
                        (*x).rb_mut()
                    })),
                    PhantomData,
                )
            }
        }
    }
    impl<'short, E: Entity, S: pulp::Simd> Reborrow<'short> for SuffixMut<'_, E, S> {
        type Target = Suffix<'short, E, S>;
        #[inline]
        fn rb(&'short self) -> Self::Target {
            unsafe {
                Suffix(
                    into_copy::<E, _>(transmute_unchecked::<
                        GroupFor<E, <E::SuffixMutUnit<'static, S> as Reborrow<'_>>::Target>,
                        GroupFor<E, E::SuffixUnit<'static, S>>,
                    >(E::faer_map(
                        E::faer_as_ref(&self.0),
                        |x| (*x).rb(),
                    ))),
                    PhantomData,
                )
            }
        }
    }
    impl<'short, E: Entity, S: pulp::Simd> ReborrowMut<'short> for SuffixMut<'_, E, S> {
        type Target = SuffixMut<'short, E, S>;
        #[inline]
        fn rb_mut(&'short mut self) -> Self::Target {
            unsafe {
                SuffixMut(
                    transmute_unchecked::<
                        GroupFor<E, <E::SuffixMutUnit<'static, S> as ReborrowMut<'_>>::Target>,
                        GroupFor<E, E::SuffixMutUnit<'static, S>>,
                    >(E::faer_map(E::faer_as_mut(&mut self.0), |x| {
                        (*x).rb_mut()
                    })),
                    PhantomData,
                )
            }
        }
    }

    impl<'short, E: Entity, S: pulp::Simd> Reborrow<'short> for Prefix<'_, E, S> {
        type Target = Prefix<'short, E, S>;
        #[inline]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }
    impl<'short, E: Entity, S: pulp::Simd> ReborrowMut<'short> for Prefix<'_, E, S> {
        type Target = Prefix<'short, E, S>;
        #[inline]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }
    impl<'short, E: Entity, S: pulp::Simd> Reborrow<'short> for Suffix<'_, E, S> {
        type Target = Suffix<'short, E, S>;
        #[inline]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }
    impl<'short, E: Entity, S: pulp::Simd> ReborrowMut<'short> for Suffix<'_, E, S> {
        type Target = Suffix<'short, E, S>;
        #[inline]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }

    impl<E: Entity, S: pulp::Simd> Copy for Prefix<'_, E, S> {}
    impl<E: Entity, S: pulp::Simd> Clone for Prefix<'_, E, S> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<E: Entity, S: pulp::Simd> Copy for Suffix<'_, E, S> {}
    impl<E: Entity, S: pulp::Simd> Clone for Suffix<'_, E, S> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }

    pub struct RefGroup<'a, E: Entity, T: 'a = <E as Entity>::Unit>(
        GroupCopyFor<E, *const T>,
        PhantomData<&'a ()>,
    );
    pub struct RefGroupMut<'a, E: Entity, T: 'a = <E as Entity>::Unit>(
        GroupFor<E, *mut T>,
        PhantomData<&'a mut ()>,
    );

    unsafe impl<E: Entity, T: Sync> Send for SliceGroup<'_, E, T> {}
    unsafe impl<E: Entity, T: Sync> Sync for SliceGroup<'_, E, T> {}
    unsafe impl<E: Entity, T: Send> Send for SliceGroupMut<'_, E, T> {}
    unsafe impl<E: Entity, T: Sync> Sync for SliceGroupMut<'_, E, T> {}

    impl<E: Entity, T> Copy for SliceGroup<'_, E, T> {}
    impl<E: Entity, T> Copy for RefGroup<'_, E, T> {}
    impl<E: Entity, T> Clone for SliceGroup<'_, E, T> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<E: Entity, T> Clone for RefGroup<'_, E, T> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<'a, E: Entity, T> RefGroup<'a, E, T> {
        #[inline(always)]
        pub fn new(reference: GroupFor<E, &'a T>) -> Self {
            Self(
                into_copy::<E, _>(E::faer_map(
                    reference,
                    #[inline(always)]
                    |reference| reference as *const T,
                )),
                PhantomData,
            )
        }

        #[inline(always)]
        pub fn into_inner(self) -> GroupFor<E, &'a T> {
            E::faer_map(
                from_copy::<E, _>(self.0),
                #[inline(always)]
                |ptr| unsafe { &*ptr },
            )
        }

        #[inline(always)]
        pub fn get(self) -> GroupCopyFor<E, T>
        where
            T: Copy,
        {
            into_copy::<E, _>(E::faer_deref(self.into_inner()))
        }
    }

    impl<'a, E: Entity, T, const N: usize> RefGroup<'a, E, [T; N]> {
        #[inline(always)]
        pub fn unzip(self) -> [RefGroup<'a, E, T>; N] {
            unsafe {
                let mut out = transmute_unchecked::<
                    core::mem::MaybeUninit<[RefGroup<'a, E, T>; N]>,
                    [core::mem::MaybeUninit<RefGroup<'a, E, T>>; N],
                >(
                    core::mem::MaybeUninit::<[RefGroup<'a, E, T>; N]>::uninit()
                );
                for (out, inp) in
                    core::iter::zip(out.iter_mut(), E::faer_into_iter(self.into_inner()))
                {
                    out.write(RefGroup::new(inp));
                }
                transmute_unchecked::<
                    [core::mem::MaybeUninit<RefGroup<'a, E, T>>; N],
                    [RefGroup<'a, E, T>; N],
                >(out)
            }
        }
    }

    impl<'a, E: Entity, T, const N: usize> RefGroupMut<'a, E, [T; N]> {
        #[inline(always)]
        pub fn unzip(self) -> [RefGroupMut<'a, E, T>; N] {
            unsafe {
                let mut out = transmute_unchecked::<
                    core::mem::MaybeUninit<[RefGroupMut<'a, E, T>; N]>,
                    [core::mem::MaybeUninit<RefGroupMut<'a, E, T>>; N],
                >(
                    core::mem::MaybeUninit::<[RefGroupMut<'a, E, T>; N]>::uninit()
                );
                for (out, inp) in
                    core::iter::zip(out.iter_mut(), E::faer_into_iter(self.into_inner()))
                {
                    out.write(RefGroupMut::new(inp));
                }
                transmute_unchecked::<
                    [core::mem::MaybeUninit<RefGroupMut<'a, E, T>>; N],
                    [RefGroupMut<'a, E, T>; N],
                >(out)
            }
        }
    }

    impl<'a, E: Entity, T> RefGroupMut<'a, E, T> {
        #[inline(always)]
        pub fn new(reference: GroupFor<E, &'a mut T>) -> Self {
            Self(
                E::faer_map(
                    reference,
                    #[inline(always)]
                    |reference| reference as *mut T,
                ),
                PhantomData,
            )
        }

        #[inline(always)]
        pub fn into_inner(self) -> GroupFor<E, &'a mut T> {
            E::faer_map(
                self.0,
                #[inline(always)]
                |ptr| unsafe { &mut *ptr },
            )
        }

        #[inline(always)]
        pub fn get(&self) -> GroupCopyFor<E, T>
        where
            T: Copy,
        {
            self.rb().get()
        }

        #[inline(always)]
        pub fn set(&mut self, value: GroupCopyFor<E, T>)
        where
            T: Copy,
        {
            E::faer_map(
                E::faer_zip(self.rb_mut().into_inner(), from_copy::<E, _>(value)),
                #[inline(always)]
                |(r, value)| *r = value,
            );
        }
    }

    impl<'a, E: Entity, T> IntoConst for SliceGroup<'a, E, T> {
        type Target = SliceGroup<'a, E, T>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            self
        }
    }
    impl<'a, E: Entity, T> IntoConst for SliceGroupMut<'a, E, T> {
        type Target = SliceGroup<'a, E, T>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            SliceGroup::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| &*slice,
            ))
        }
    }

    impl<'a, E: Entity, T> IntoConst for RefGroup<'a, E, T> {
        type Target = RefGroup<'a, E, T>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            self
        }
    }
    impl<'a, E: Entity, T> IntoConst for RefGroupMut<'a, E, T> {
        type Target = RefGroup<'a, E, T>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            RefGroup::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| &*slice,
            ))
        }
    }

    impl<'short, 'a, E: Entity, T> ReborrowMut<'short> for RefGroup<'a, E, T> {
        type Target = RefGroup<'short, E, T>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity, T> Reborrow<'short> for RefGroup<'a, E, T> {
        type Target = RefGroup<'short, E, T>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity, T> ReborrowMut<'short> for RefGroupMut<'a, E, T> {
        type Target = RefGroupMut<'short, E, T>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            RefGroupMut::new(E::faer_map(
                E::faer_as_mut(&mut self.0),
                #[inline(always)]
                |this| unsafe { &mut **this },
            ))
        }
    }

    impl<'short, 'a, E: Entity, T> Reborrow<'short> for RefGroupMut<'a, E, T> {
        type Target = RefGroup<'short, E, T>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            RefGroup::new(E::faer_map(
                E::faer_as_ref(&self.0),
                #[inline(always)]
                |this| unsafe { &**this },
            ))
        }
    }

    impl<'a, E: Entity, T> SliceGroup<'a, E, T> {
        #[inline(always)]
        pub fn new(slice: GroupFor<E, &'a [T]>) -> Self {
            Self(
                into_copy::<E, _>(E::faer_map(slice, |slice| slice as *const [T])),
                PhantomData,
            )
        }

        #[inline(always)]
        pub fn into_inner(self) -> GroupFor<E, &'a [T]> {
            unsafe { E::faer_map(from_copy::<E, _>(self.0), |ptr| &*ptr) }
        }

        #[inline(always)]
        pub fn as_arrays<const N: usize>(
            self,
        ) -> (SliceGroup<'a, E, [T; N]>, SliceGroup<'a, E, T>) {
            let (head, tail) = E::faer_as_arrays::<N, _>(self.into_inner());
            (SliceGroup::new(head), SliceGroup::new(tail))
        }
    }

    impl<'a, E: Entity, T> SliceGroupMut<'a, E, T> {
        #[inline(always)]
        pub fn new(slice: GroupFor<E, &'a mut [T]>) -> Self {
            Self(E::faer_map(slice, |slice| slice as *mut [T]), PhantomData)
        }

        #[inline(always)]
        pub fn into_inner(self) -> GroupFor<E, &'a mut [T]> {
            unsafe { E::faer_map(self.0, |ptr| &mut *ptr) }
        }

        #[inline(always)]
        pub fn as_arrays_mut<const N: usize>(
            self,
        ) -> (SliceGroupMut<'a, E, [T; N]>, SliceGroupMut<'a, E, T>) {
            let (head, tail) = E::faer_as_arrays_mut::<N, _>(self.into_inner());
            (SliceGroupMut::new(head), SliceGroupMut::new(tail))
        }
    }

    impl<'short, 'a, E: Entity, T> ReborrowMut<'short> for SliceGroup<'a, E, T> {
        type Target = SliceGroup<'short, E, T>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity, T> Reborrow<'short> for SliceGroup<'a, E, T> {
        type Target = SliceGroup<'short, E, T>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity, T> ReborrowMut<'short> for SliceGroupMut<'a, E, T> {
        type Target = SliceGroupMut<'short, E, T>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            SliceGroupMut::new(E::faer_map(
                E::faer_as_mut(&mut self.0),
                #[inline(always)]
                |this| unsafe { &mut **this },
            ))
        }
    }

    impl<'short, 'a, E: Entity, T> Reborrow<'short> for SliceGroupMut<'a, E, T> {
        type Target = SliceGroup<'short, E, T>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            SliceGroup::new(E::faer_map(
                E::faer_as_ref(&self.0),
                #[inline(always)]
                |this| unsafe { &**this },
            ))
        }
    }

    impl<'a, E: Entity> RefGroup<'a, E> {
        #[inline(always)]
        pub fn read(&self) -> E {
            E::faer_from_units(E::faer_deref(self.into_inner()))
        }
    }

    impl<'a, E: Entity> RefGroupMut<'a, E> {
        #[inline(always)]
        pub fn read(&self) -> E {
            self.rb().read()
        }

        #[inline(always)]
        pub fn write(&mut self, value: E) {
            E::faer_map(
                E::faer_zip(self.rb_mut().into_inner(), value.faer_into_units()),
                #[inline(always)]
                |(r, value)| *r = value,
            );
        }
    }

    impl<'a, E: Entity> SliceGroup<'a, E> {
        #[inline(always)]
        #[track_caller]
        pub fn read(&self, idx: usize) -> E {
            assert!(idx < self.len());
            unsafe { self.read_unchecked(idx) }
        }

        #[inline(always)]
        #[track_caller]
        pub unsafe fn read_unchecked(&self, idx: usize) -> E {
            debug_assert!(idx < self.len());
            E::faer_from_units(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| *slice.get_unchecked(idx),
            ))
        }
    }
    impl<'a, E: Entity, T> SliceGroup<'a, E, T> {
        #[inline(always)]
        #[track_caller]
        pub fn get(self, idx: usize) -> RefGroup<'a, E, T> {
            assert!(idx < self.len());
            unsafe { self.get_unchecked(idx) }
        }

        #[inline(always)]
        #[track_caller]
        pub unsafe fn get_unchecked(self, idx: usize) -> RefGroup<'a, E, T> {
            debug_assert!(idx < self.len());
            RefGroup::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| slice.get_unchecked(idx),
            ))
        }

        #[inline]
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        #[inline]
        pub fn len(&self) -> usize {
            let mut len = usize::MAX;
            E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| len = Ord::min(len, slice.len()),
            );
            len
        }

        #[inline(always)]
        #[track_caller]
        pub fn subslice(self, range: Range<usize>) -> Self {
            assert!(all(range.start <= range.end, range.end <= self.len()));
            unsafe { self.subslice_unchecked(range) }
        }

        #[inline(always)]
        #[track_caller]
        pub fn split_at(self, idx: usize) -> (Self, Self) {
            assert!(idx <= self.len());
            let (head, tail) = E::faer_unzip(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| slice.split_at(idx),
            ));
            (Self::new(head), Self::new(tail))
        }

        #[inline(always)]
        #[track_caller]
        pub unsafe fn subslice_unchecked(self, range: Range<usize>) -> Self {
            debug_assert!(all(range.start <= range.end, range.end <= self.len()));
            Self::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| slice.get_unchecked(range.start..range.end),
            ))
        }

        #[inline(always)]
        pub fn into_ref_iter(self) -> impl Iterator<Item = RefGroup<'a, E, T>> {
            E::faer_into_iter(self.into_inner()).map(RefGroup::new)
        }

        #[inline(always)]
        pub fn into_chunks_exact(
            self,
            chunk_size: usize,
        ) -> (impl Iterator<Item = SliceGroup<'a, E, T>>, Self) {
            let len = self.len();
            let mid = len / chunk_size * chunk_size;
            let (head, tail) = E::faer_unzip(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| slice.split_at(mid),
            ));
            let head = E::faer_map(
                head,
                #[inline(always)]
                |head| head.chunks_exact(chunk_size),
            );
            (
                E::faer_into_iter(head).map(SliceGroup::new),
                SliceGroup::new(tail),
            )
        }
    }

    impl<'a, E: Entity> SliceGroupMut<'a, E> {
        #[inline(always)]
        #[track_caller]
        pub fn read(&self, idx: usize) -> E {
            self.rb().read(idx)
        }

        #[inline(always)]
        #[track_caller]
        pub unsafe fn read_unchecked(&self, idx: usize) -> E {
            self.rb().read_unchecked(idx)
        }

        #[inline(always)]
        #[track_caller]
        pub fn write(&mut self, idx: usize, value: E) {
            assert!(idx < self.len());
            unsafe { self.write_unchecked(idx, value) }
        }

        #[inline(always)]
        #[track_caller]
        pub unsafe fn write_unchecked(&mut self, idx: usize, value: E) {
            debug_assert!(idx < self.len());
            E::faer_map(
                E::faer_zip(self.rb_mut().into_inner(), value.faer_into_units()),
                #[inline(always)]
                |(slice, value)| *slice.get_unchecked_mut(idx) = value,
            );
        }

        #[inline]
        pub fn fill_zero(&mut self) {
            E::faer_map(self.rb_mut().into_inner(), |slice| unsafe {
                let len = slice.len();
                core::ptr::write_bytes(slice.as_mut_ptr(), 0u8, len);
            });
        }
    }

    impl<'a, E: Entity, T> SliceGroupMut<'a, E, T> {
        #[inline(always)]
        #[track_caller]
        pub fn get_mut(self, idx: usize) -> RefGroupMut<'a, E, T> {
            assert!(idx < self.len());
            unsafe { self.get_unchecked_mut(idx) }
        }

        #[inline(always)]
        #[track_caller]
        pub unsafe fn get_unchecked_mut(self, idx: usize) -> RefGroupMut<'a, E, T> {
            debug_assert!(idx < self.len());
            RefGroupMut::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| slice.get_unchecked_mut(idx),
            ))
        }

        #[inline(always)]
        #[track_caller]
        pub fn get(self, idx: usize) -> RefGroup<'a, E, T> {
            self.into_const().get(idx)
        }

        #[inline(always)]
        #[track_caller]
        pub unsafe fn get_unchecked(self, idx: usize) -> RefGroup<'a, E, T> {
            self.into_const().get_unchecked(idx)
        }

        #[inline]
        pub fn is_empty(&self) -> bool {
            self.rb().is_empty()
        }

        #[inline]
        pub fn len(&self) -> usize {
            self.rb().len()
        }

        #[inline(always)]
        #[track_caller]
        pub fn subslice(self, range: Range<usize>) -> Self {
            assert!(all(range.start <= range.end, range.end <= self.len()));
            unsafe { self.subslice_unchecked(range) }
        }

        #[inline(always)]
        #[track_caller]
        pub unsafe fn subslice_unchecked(self, range: Range<usize>) -> Self {
            debug_assert!(all(range.start <= range.end, range.end <= self.len()));
            Self::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| slice.get_unchecked_mut(range.start..range.end),
            ))
        }

        #[inline(always)]
        pub fn into_mut_iter(self) -> impl Iterator<Item = RefGroupMut<'a, E, T>> {
            E::faer_into_iter(self.into_inner()).map(RefGroupMut::new)
        }

        #[inline(always)]
        #[track_caller]
        pub fn split_at(self, idx: usize) -> (Self, Self) {
            assert!(idx <= self.len());
            let (head, tail) = E::faer_unzip(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| slice.split_at_mut(idx),
            ));
            (Self::new(head), Self::new(tail))
        }

        #[inline(always)]
        pub fn into_chunks_exact(
            self,
            chunk_size: usize,
        ) -> (impl Iterator<Item = SliceGroupMut<'a, E, T>>, Self) {
            let len = self.len();
            let mid = len % chunk_size * chunk_size;
            let (head, tail) = E::faer_unzip(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| slice.split_at_mut(mid),
            ));
            let head = E::faer_map(
                head,
                #[inline(always)]
                |head| head.chunks_exact_mut(chunk_size),
            );
            (
                E::faer_into_iter(head).map(SliceGroupMut::new),
                SliceGroupMut::new(tail),
            )
        }
    }

    impl<E: Entity, S: pulp::Simd> core::fmt::Debug for Prefix<'_, E, S> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            unsafe {
                transmute_unchecked::<SimdGroupFor<E, S>, GroupDebugFor<E, SimdUnitFor<E, S>>>(
                    self.read_or(core::mem::zeroed()),
                )
                .fmt(f)
            }
        }
    }
    impl<E: Entity, S: pulp::Simd> core::fmt::Debug for PrefixMut<'_, E, S> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.rb().fmt(f)
        }
    }
    impl<E: Entity, S: pulp::Simd> core::fmt::Debug for Suffix<'_, E, S> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            unsafe {
                transmute_unchecked::<SimdGroupFor<E, S>, GroupDebugFor<E, SimdUnitFor<E, S>>>(
                    self.read_or(core::mem::zeroed()),
                )
                .fmt(f)
            }
        }
    }
    impl<E: Entity, S: pulp::Simd> core::fmt::Debug for SuffixMut<'_, E, S> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.rb().fmt(f)
        }
    }
    impl<E: Entity, T: Debug> core::fmt::Debug for RefGroup<'_, E, T> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            unsafe {
                transmute_unchecked::<GroupFor<E, &T>, GroupDebugFor<E, &T>>(self.into_inner())
                    .fmt(f)
            }
        }
    }
    impl<E: Entity, T: Debug> core::fmt::Debug for RefGroupMut<'_, E, T> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.rb().fmt(f)
        }
    }
    impl<E: Entity, T: Debug> core::fmt::Debug for SliceGroup<'_, E, T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_list().entries(self.into_ref_iter()).finish()
        }
    }
    impl<E: Entity, T: Debug> core::fmt::Debug for SliceGroupMut<'_, E, T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.rb().fmt(f)
        }
    }
}

/// Sparse matrix data structures.
///
/// Most sparse matrix algorithms accept matrices in sparse column-oriented format.
/// This format represents each column of the matrix by storing the row indices of its non-zero
/// elements, as well as their values.
///
/// The indices and the values are each stored in a contiguous slice (or group of slices for
/// arbitrary values). In order to specify where each column starts and ends, a slice of size
/// `ncols + 1` stores the start of each column, with the last element being equal to the total
/// number of non-zeros (or the capacity in uncompressed mode).
///
/// # Example
///
/// Consider the 4-by-5 matrix:
/// ```notcode
/// 10.0  0.0  12.0  -1.0  13.0
///  0.0  0.0  25.0  -2.0   0.0
///  1.0  0.0   0.0   0.0   0.0
///  4.0  0.0   0.0   0.0   5.0
/// ```
///
/// The matrix is stored as follows:
/// ```notcode
/// column pointers:  0 |  3 |  3 |  5 |  7 |  9
///
/// row indices:    0 |    2 |    3 |    0 |    1 |    0 |    1 |    0 |    3
/// values     : 10.0 |  1.0 |  4.0 | 12.0 | 25.0 | -1.0 | -2.0 | 13.0 |  5.0
/// ```
pub mod sparse {
    use super::*;
    use crate::assert;
    use core::{iter::zip, ops::Range, slice::SliceIndex};
    use group_helpers::SliceGroup;
    use permutation::{Index, SignedIndex};

    #[inline(always)]
    #[track_caller]
    #[doc(hidden)]
    pub unsafe fn __get_unchecked<I, R: Clone + SliceIndex<[I]>>(slice: &[I], i: R) -> &R::Output {
        #[cfg(debug_assertions)]
        {
            let _ = &slice[i.clone()];
        }
        unsafe { slice.get_unchecked(i) }
    }
    #[inline(always)]
    #[track_caller]
    #[doc(hidden)]
    pub unsafe fn __get_unchecked_mut<I, R: Clone + SliceIndex<[I]>>(
        slice: &mut [I],
        i: R,
    ) -> &mut R::Output {
        #[cfg(debug_assertions)]
        {
            let _ = &slice[i.clone()];
        }
        unsafe { slice.get_unchecked_mut(i) }
    }

    #[inline(always)]
    #[doc(hidden)]
    pub fn windows2<I>(slice: &[I]) -> impl DoubleEndedIterator<Item = &[I; 2]> {
        slice
            .windows(2)
            .map(|window| unsafe { &*(window.as_ptr() as *const [I; 2]) })
    }

    #[inline]
    #[doc(hidden)]
    pub const fn repeat_byte(byte: u8) -> usize {
        union Union {
            bytes: [u8; 32],
            value: usize,
        }

        let data = Union { bytes: [byte; 32] };
        unsafe { data.value }
    }

    /// Symbolic structure of sparse matrix in column format, either compressed or uncompressed.
    ///
    /// Requires:
    /// * `nrows <= I::Signed::MAX` (always checked)
    /// * `ncols <= I::Signed::MAX` (always checked)
    /// * `col_ptrs` has length `ncols + 1` (always checked)
    /// * `col_ptrs` is non-decreasing
    /// * `col_ptrs[0]..col_ptrs[ncols]` is a valid range in row_indices (always checked, assuming
    ///   non-decreasing)
    /// * if `nnz_per_col` is `None`, elements of `row_indices[col_ptrs[j]..col_ptrs[j + 1]]` are
    ///   less than `nrows`
    ///
    /// * `nnz_per_col[j] <= col_ptrs[j+1] - col_ptrs[j]`
    /// * if `nnz_per_col` is `Some(_)`, elements of `row_indices[col_ptrs[j]..][..nnz_per_col[j]]`
    ///   are less than `nrows`
    #[derive(Debug)]
    pub struct SymbolicSparseColMatRef<'a, I> {
        nrows: usize,
        ncols: usize,
        col_ptr: &'a [I],
        col_nnz: Option<&'a [I]>,
        row_ind: &'a [I],
    }

    /// Symbolic structure of sparse matrix in row format, either compressed or uncompressed.
    ///
    /// Requires:
    /// * `nrows <= I::Signed::MAX` (always checked)
    /// * `ncols <= I::Signed::MAX` (always checked)
    /// * `row_ptrs` has length `nrows + 1` (always checked)
    /// * `row_ptrs` is non-decreasing
    /// * `row_ptrs[0]..row_ptrs[nrows]` is a valid range in row_indices (always checked, assuming
    ///   non-decreasing)
    /// * if `nnz_per_row` is `None`, elements of `col_indices[row_ptrs[i]..row_ptrs[i + 1]]` are
    ///   less than `ncols`
    ///
    /// * `nnz_per_row[i] <= row_ptrs[i+1] - row_ptrs[i]`
    /// * if `nnz_per_row` is `Some(_)`, elements of `col_indices[row_ptrs[i]..][..nnz_per_row[i]]`
    ///   are less than `ncols`
    #[derive(Debug)]
    pub struct SymbolicSparseRowMatRef<'a, I> {
        nrows: usize,
        ncols: usize,
        row_ptr: &'a [I],
        row_nnz: Option<&'a [I]>,
        col_ind: &'a [I],
    }

    impl<I> Copy for SymbolicSparseColMatRef<'_, I> {}
    impl<I> Clone for SymbolicSparseColMatRef<'_, I> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<I> Copy for SymbolicSparseRowMatRef<'_, I> {}
    impl<I> Clone for SymbolicSparseRowMatRef<'_, I> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<'a, I: Index> SymbolicSparseRowMatRef<'a, I> {
        /// Creates a new symbolic matrix view after asserting its invariants.
        ///
        /// # Panics
        ///
        /// See type level documentation.
        #[inline]
        #[track_caller]
        pub fn new_checked(
            nrows: usize,
            ncols: usize,
            row_ptrs: &'a [I],
            nnz_per_row: Option<&'a [I]>,
            col_indices: &'a [I],
        ) -> Self {
            assert!(all(
                ncols <= I::Signed::MAX.zx(),
                nrows <= I::Signed::MAX.zx(),
            ));
            assert!(row_ptrs.len() == nrows + 1);
            for &[c, c_next] in windows2(row_ptrs) {
                assert!(c <= c_next);
            }
            assert!(row_ptrs[ncols].zx() <= col_indices.len());

            if let Some(nnz_per_row) = nnz_per_row {
                for (&nnz_i, &[c, c_next]) in zip(nnz_per_row, windows2(row_ptrs)) {
                    assert!(nnz_i <= c_next - c);
                    for &j in &col_indices[c.zx()..c.zx() + nnz_i.zx()] {
                        assert!(j < I::truncate(ncols));
                    }
                }
            } else {
                let c0 = row_ptrs[0].zx();
                let cn = row_ptrs[ncols].zx();
                for &j in &col_indices[c0..cn] {
                    assert!(j < I::truncate(ncols));
                }
            }

            Self {
                nrows,
                ncols,
                row_ptr: row_ptrs,
                row_nnz: nnz_per_row,
                col_ind: col_indices,
            }
        }

        /// Creates a new symbolic matrix view without asserting its invariants.
        ///
        /// # Safety
        ///
        /// See type level documentation.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn new_unchecked(
            nrows: usize,
            ncols: usize,
            row_ptrs: &'a [I],
            nnz_per_row: Option<&'a [I]>,
            col_indices: &'a [I],
        ) -> Self {
            assert!(all(
                ncols <= <I::Signed as SignedIndex>::MAX.zx(),
                nrows <= <I::Signed as SignedIndex>::MAX.zx(),
            ));
            assert!(row_ptrs.len() == nrows + 1);
            assert!(row_ptrs[nrows].zx() <= col_indices.len());

            Self {
                nrows,
                ncols,
                row_ptr: row_ptrs,
                row_nnz: nnz_per_row,
                col_ind: col_indices,
            }
        }

        /// Returns the number of rows of the matrix.
        #[inline]
        pub fn nrows(&self) -> usize {
            self.nrows
        }
        /// Returns the number of columns of the matrix.
        #[inline]
        pub fn ncols(&self) -> usize {
            self.ncols
        }

        #[inline]
        pub fn transpose(self) -> SymbolicSparseColMatRef<'a, I> {
            SymbolicSparseColMatRef {
                nrows: self.ncols,
                ncols: self.nrows,
                col_ptr: self.row_ptr,
                col_nnz: self.row_nnz,
                row_ind: self.col_ind,
            }
        }

        /// Returns the number of symbolic non-zeros in the matrix.
        ///
        /// The value is guaranteed to be less than `I::Signed::MAX`.
        #[inline]
        pub fn compute_nnz(&self) -> usize {
            self.transpose().compute_nnz()
        }

        /// Returns the column pointers.
        #[inline]
        pub fn row_ptrs(&self) -> &'a [I] {
            self.row_ptr
        }

        /// Returns the count of non-zeros per column of the matrix.
        #[inline]
        pub fn nnz_per_row(&self) -> Option<&'a [I]> {
            self.row_nnz
        }

        /// Returns the column indices.
        #[inline]
        pub fn col_indices(&self) -> &'a [I] {
            self.col_ind
        }

        /// Returns the column indices of row i.
        ///
        /// # Panics
        ///
        /// Panics if `i >= self.nrows()`
        #[inline]
        #[track_caller]
        pub fn col_indices_of_row_raw(&self, i: usize) -> &'a [I] {
            &self.col_ind[self.row_range(i)]
        }

        /// Returns the column indices of row i.
        ///
        /// # Panics
        ///
        /// Panics if `i >= self.ncols()`
        #[inline]
        #[track_caller]
        pub fn col_indices_of_row(
            &self,
            i: usize,
        ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = usize> {
            self.col_indices_of_row_raw(i).iter().map(
                #[inline(always)]
                |&i| i.zx(),
            )
        }

        /// Returns the range that the row `i` occupies in `self.col_indices().
        ///
        /// # Panics
        ///
        /// Panics if `i >= self.nrows()`
        #[inline]
        #[track_caller]
        pub fn row_range(&self, i: usize) -> Range<usize> {
            let start = self.row_ptr[i].zx();
            let end = self
                .row_nnz
                .map(|row_nnz| row_nnz[i].zx() + start)
                .unwrap_or(self.row_ptr[i + 1].zx());

            start..end
        }

        /// Returns the range that the row `i` occupies in `self.col_indices().
        ///
        /// # Safety
        ///
        /// The behavior is undefined if `i >= self.nrows()`
        #[inline]
        #[track_caller]
        pub unsafe fn row_range_unchecked(&self, i: usize) -> Range<usize> {
            let start = __get_unchecked(self.row_ptr, i).zx();
            let end = self
                .row_nnz
                .map(|row_nnz| (__get_unchecked(row_nnz, i).zx() + start))
                .unwrap_or(__get_unchecked(self.row_ptr, i + 1).zx());

            start..end
        }
    }

    impl<'a, I: Index> SymbolicSparseColMatRef<'a, I> {
        /// Creates a new symbolic matrix view after asserting its invariants.
        ///
        /// # Panics
        ///
        /// See type level documentation.
        #[inline]
        #[track_caller]
        pub fn new_checked(
            nrows: usize,
            ncols: usize,
            col_ptrs: &'a [I],
            nnz_per_col: Option<&'a [I]>,
            row_indices: &'a [I],
        ) -> Self {
            assert!(all(
                ncols <= I::Signed::MAX.zx(),
                nrows <= I::Signed::MAX.zx(),
            ));
            assert!(col_ptrs.len() == ncols + 1);
            for &[c, c_next] in windows2(col_ptrs) {
                assert!(c <= c_next);
            }
            assert!(col_ptrs[ncols].zx() <= row_indices.len());

            if let Some(nnz_per_col) = nnz_per_col {
                for (&nnz_j, &[c, c_next]) in zip(nnz_per_col, windows2(col_ptrs)) {
                    assert!(nnz_j <= c_next - c);
                    for &i in &row_indices[c.zx()..c.zx() + nnz_j.zx()] {
                        assert!(i < I::truncate(nrows));
                    }
                }
            } else {
                let c0 = col_ptrs[0].zx();
                let cn = col_ptrs[ncols].zx();
                for &i in &row_indices[c0..cn] {
                    assert!(i < I::truncate(nrows));
                }
            }

            Self {
                nrows,
                ncols,
                col_ptr: col_ptrs,
                col_nnz: nnz_per_col,
                row_ind: row_indices,
            }
        }

        /// Creates a new symbolic matrix view without asserting its invariants.
        ///
        /// # Safety
        ///
        /// See type level documentation.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn new_unchecked(
            nrows: usize,
            ncols: usize,
            col_ptrs: &'a [I],
            nnz_per_col: Option<&'a [I]>,
            row_indices: &'a [I],
        ) -> Self {
            assert!(all(
                ncols <= <I::Signed as SignedIndex>::MAX.zx(),
                nrows <= <I::Signed as SignedIndex>::MAX.zx(),
            ));
            assert!(col_ptrs.len() == ncols + 1);
            assert!(col_ptrs[ncols].zx() <= row_indices.len());

            Self {
                nrows,
                ncols,
                col_ptr: col_ptrs,
                col_nnz: nnz_per_col,
                row_ind: row_indices,
            }
        }

        /// Returns the number of rows of the matrix.
        #[inline]
        pub fn nrows(&self) -> usize {
            self.nrows
        }
        /// Returns the number of columns of the matrix.
        #[inline]
        pub fn ncols(&self) -> usize {
            self.ncols
        }

        #[inline]
        pub fn transpose(self) -> SymbolicSparseRowMatRef<'a, I> {
            SymbolicSparseRowMatRef {
                nrows: self.ncols,
                ncols: self.nrows,
                row_ptr: self.col_ptr,
                row_nnz: self.col_nnz,
                col_ind: self.row_ind,
            }
        }

        /// Returns the number of symbolic non-zeros in the matrix.
        ///
        /// The value is guaranteed to be less than `I::Signed::MAX`.
        #[inline]
        pub fn compute_nnz(&self) -> usize {
            match self.col_nnz {
                Some(col_nnz) => {
                    let mut nnz = 0usize;
                    for &nnz_j in col_nnz {
                        // can't overflow
                        nnz += nnz_j.zx();
                    }
                    nnz
                }
                None => self.col_ptr[self.ncols].zx() - self.col_ptr[0].zx(),
            }
        }

        /// Returns the column pointers.
        #[inline]
        pub fn col_ptrs(&self) -> &'a [I] {
            self.col_ptr
        }

        /// Returns the count of non-zeros per column of the matrix.
        #[inline]
        pub fn nnz_per_col(&self) -> Option<&'a [I]> {
            self.col_nnz
        }

        /// Returns the row indices.
        #[inline]
        pub fn row_indices(&self) -> &'a [I] {
            self.row_ind
        }

        /// Returns the row indices of column j.
        ///
        /// # Panics
        ///
        /// Panics if `j >= self.ncols()`
        #[inline]
        #[track_caller]
        pub fn row_indices_of_col_raw(&self, j: usize) -> &'a [I] {
            &self.row_ind[self.col_range(j)]
        }

        /// Returns the row indices of column j.
        ///
        /// # Panics
        ///
        /// Panics if `j >= self.ncols()`
        #[inline]
        #[track_caller]
        pub fn row_indices_of_col(
            &self,
            j: usize,
        ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = usize> {
            self.row_indices_of_col_raw(j).iter().map(
                #[inline(always)]
                |&i| i.zx(),
            )
        }

        /// Returns the range that the column `j` occupies in `self.row_indices().
        ///
        /// # Panics
        ///
        /// Panics if `j >= self.ncols()`
        #[inline]
        #[track_caller]
        pub fn col_range(&self, j: usize) -> Range<usize> {
            let start = self.col_ptr[j].zx();
            let end = self
                .col_nnz
                .map(|col_nnz| col_nnz[j].zx() + start)
                .unwrap_or(self.col_ptr[j + 1].zx());

            start..end
        }

        /// Returns the range that the column `j` occupies in `self.row_indices().
        ///
        /// # Safety
        ///
        /// The behavior is undefined if `j >= self.ncols()`
        #[inline]
        #[track_caller]
        pub unsafe fn col_range_unchecked(&self, j: usize) -> Range<usize> {
            let start = __get_unchecked(self.col_ptr, j).zx();
            let end = self
                .col_nnz
                .map(|col_nnz| (__get_unchecked(col_nnz, j).zx() + start))
                .unwrap_or(__get_unchecked(self.col_ptr, j + 1).zx());

            start..end
        }
    }

    /// Sparse matrix in column format, either compressed or uncompressed.
    pub type SparseRowMatRef<'a, I, E> = Matrix<inner::SparseRowMatRefInner<'a, I, E>>;

    /// Sparse matrix in column format, either compressed or uncompressed.
    pub type SparseColMatRef<'a, I, E> = Matrix<inner::SparseColMatRefInner<'a, I, E>>;

    impl<'a, I: Index, E: Entity> SparseRowMatRef<'a, I, E> {
        /// Creates a new sparse matrix view.
        ///
        /// # Panics
        ///
        /// Panics if the length of `values` is not equal to the length of
        /// `symbolic.col_indices()`.
        #[inline]
        #[track_caller]
        pub fn new(
            symbolic: SymbolicSparseRowMatRef<'a, I>,
            values: GroupFor<E, &'a [E::Unit]>,
        ) -> Self {
            let values = SliceGroup::new(values);
            assert!(symbolic.col_indices().len() == values.len());
            Self {
                inner: inner::SparseRowMatRefInner { symbolic, values },
            }
        }

        /// Returns the numerical values of the matrix.
        #[inline]
        pub fn values(&self) -> GroupFor<E, &'a [E::Unit]> {
            self.inner.values.into_inner()
        }

        #[inline]
        pub fn transpose(self) -> SparseColMatRef<'a, I, E> {
            SparseColMatRef {
                inner: inner::SparseColMatRefInner {
                    symbolic: SymbolicSparseColMatRef {
                        nrows: self.inner.symbolic.ncols,
                        ncols: self.inner.symbolic.nrows,
                        col_ptr: self.inner.symbolic.row_ptr,
                        col_nnz: self.inner.symbolic.row_nnz,
                        row_ind: self.inner.symbolic.col_ind,
                    },
                    values: self.inner.values,
                },
            }
        }

        #[inline]
        pub fn conjugate(self) -> SparseRowMatRef<'a, I, E::Conj>
        where
            E: Conjugate,
        {
            SparseRowMatRef {
                inner: inner::SparseRowMatRefInner {
                    symbolic: self.inner.symbolic,
                    values: unsafe {
                        SliceGroup::<'a, E::Conj>::new(transmute_unchecked::<
                            GroupFor<E, &[UnitFor<E::Conj>]>,
                            GroupFor<E::Conj, &[UnitFor<E::Conj>]>,
                        >(E::faer_map(
                            self.inner.values.into_inner(),
                            |slice| {
                                let len = slice.len();
                                core::slice::from_raw_parts(
                                    slice.as_ptr() as *const UnitFor<E> as *const UnitFor<E::Conj>,
                                    len,
                                )
                            },
                        )))
                    },
                },
            }
        }

        #[inline]
        pub fn adjoint(self) -> SparseColMatRef<'a, I, E::Conj>
        where
            E: Conjugate,
        {
            self.transpose().conjugate()
        }

        /// Returns the numerical values of row `i` of the matrix.
        ///
        /// # Panics:
        ///
        /// Panics if `i >= nrows`.
        #[inline]
        #[track_caller]
        pub fn values_of_row(&self, i: usize) -> GroupFor<E, &'a [E::Unit]> {
            self.inner.values.subslice(self.row_range(i)).into_inner()
        }

        /// Returns the symbolic structure of the matrix.
        #[inline]
        pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'a, I> {
            self.inner.symbolic
        }
    }

    impl<'a, I: Index, E: Entity> SparseColMatRef<'a, I, E> {
        /// Creates a new sparse matrix view.
        ///
        /// # Panics
        ///
        /// Panics if the length of `values` is not equal to the length of
        /// `symbolic.row_indices()`.
        #[inline]
        #[track_caller]
        pub fn new(
            symbolic: SymbolicSparseColMatRef<'a, I>,
            values: GroupFor<E, &'a [E::Unit]>,
        ) -> Self {
            let values = SliceGroup::new(values);
            assert!(symbolic.row_indices().len() == values.len());
            Self {
                inner: inner::SparseColMatRefInner { symbolic, values },
            }
        }

        #[inline]
        pub fn transpose(self) -> SparseRowMatRef<'a, I, E> {
            SparseRowMatRef {
                inner: inner::SparseRowMatRefInner {
                    symbolic: SymbolicSparseRowMatRef {
                        nrows: self.inner.symbolic.ncols,
                        ncols: self.inner.symbolic.nrows,
                        row_ptr: self.inner.symbolic.col_ptr,
                        row_nnz: self.inner.symbolic.col_nnz,
                        col_ind: self.inner.symbolic.row_ind,
                    },
                    values: self.inner.values,
                },
            }
        }

        #[inline]
        pub fn conjugate(self) -> SparseColMatRef<'a, I, E::Conj>
        where
            E: Conjugate,
        {
            SparseColMatRef {
                inner: inner::SparseColMatRefInner {
                    symbolic: self.inner.symbolic,
                    values: unsafe {
                        SliceGroup::<'a, E::Conj>::new(transmute_unchecked::<
                            GroupFor<E, &[UnitFor<E::Conj>]>,
                            GroupFor<E::Conj, &[UnitFor<E::Conj>]>,
                        >(E::faer_map(
                            self.inner.values.into_inner(),
                            |slice| {
                                let len = slice.len();
                                core::slice::from_raw_parts(
                                    slice.as_ptr() as *const UnitFor<E> as *const UnitFor<E::Conj>,
                                    len,
                                )
                            },
                        )))
                    },
                },
            }
        }

        #[inline]
        pub fn adjoint(self) -> SparseRowMatRef<'a, I, E::Conj>
        where
            E: Conjugate,
        {
            self.transpose().conjugate()
        }

        /// Returns the numerical values of the matrix.
        #[inline]
        pub fn values(&self) -> GroupFor<E, &'a [E::Unit]> {
            self.inner.values.into_inner()
        }

        /// Returns the numerical values of column `j` of the matrix.
        ///
        /// # Panics:
        ///
        /// Panics if `j >= ncols`.
        #[inline]
        #[track_caller]
        pub fn values_of_col(&self, j: usize) -> GroupFor<E, &'a [E::Unit]> {
            self.inner.values.subslice(self.col_range(j)).into_inner()
        }

        /// Returns the symbolic structure of the matrix.
        #[inline]
        pub fn symbolic(&self) -> SymbolicSparseColMatRef<'a, I> {
            self.inner.symbolic
        }
    }

    impl<'a, I, E: Entity> core::ops::Deref for SparseRowMatRef<'a, I, E> {
        type Target = SymbolicSparseRowMatRef<'a, I>;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.inner.symbolic
        }
    }

    impl<'a, I, E: Entity> core::ops::Deref for SparseColMatRef<'a, I, E> {
        type Target = SymbolicSparseColMatRef<'a, I>;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.inner.symbolic
        }
    }

    /// Sparse matrix multiplication.
    pub mod mul {
        // TODO: sparse_sparse_matmul
        //
        // PERF: optimize matmul
        // - parallelization
        // - simd(?)

        use super::*;
        use crate::{
            assert,
            constrained::{self, Size},
        };

        /// Multiplies a sparse matrix `lhs` by a dense matrix `rhs`, and stores the result in
        /// `acc`. See [`crate::mul::matmul`] for more details.
        #[track_caller]
        pub fn sparse_dense_matmul<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            acc: MatMut<'_, E>,
            lhs: SparseColMatRef<'_, I, LhsE>,
            rhs: MatRef<'_, RhsE>,
            alpha: Option<E>,
            beta: E,
            parallelism: Parallelism,
        ) {
            assert!(all(
                acc.nrows() == lhs.nrows(),
                acc.ncols() == rhs.ncols(),
                lhs.ncols() == rhs.nrows(),
            ));

            let _ = parallelism;
            let m = acc.nrows();
            let n = acc.ncols();
            let k = lhs.ncols();

            let mut acc = acc;

            match alpha {
                Some(alpha) => {
                    if alpha != E::faer_one() {
                        zipped!(acc.rb_mut())
                            .for_each(|unzipped!(mut dst)| dst.write(dst.read().faer_mul(alpha)))
                    }
                }
                None => acc.fill_zero(),
            }

            Size::with2(m, n, |m, n| {
                Size::with(k, |k| {
                    let mut acc = constrained::MatMut::new(acc, m, n);
                    let lhs = constrained::sparse::SparseColMatRef::new(lhs, m, k);
                    let rhs = constrained::MatRef::new(rhs, k, n);

                    for j in n.indices() {
                        for depth in k.indices() {
                            let rhs_kj = rhs.read(depth, j).canonicalize().faer_mul(beta);
                            for (i, lhs_ik) in zip(
                                lhs.row_indices_of_col(depth),
                                SliceGroup::<'_, LhsE>::new(lhs.values_of_col(depth))
                                    .into_ref_iter(),
                            ) {
                                acc.write(
                                    i,
                                    j,
                                    acc.read(i, j)
                                        .faer_add(lhs_ik.read().canonicalize().faer_mul(rhs_kj)),
                                );
                            }
                        }
                    }
                });
            });
        }

        /// Multiplies a dense matrix `lhs` by a sparse matrix `rhs`, and stores the result in
        /// `acc`. See [`crate::mul::matmul`] for more details.
        #[track_caller]
        pub fn dense_sparse_matmul<
            I: Index,
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            acc: MatMut<'_, E>,
            lhs: MatRef<'_, LhsE>,
            rhs: SparseColMatRef<'_, I, RhsE>,
            alpha: Option<E>,
            beta: E,
            parallelism: Parallelism,
        ) {
            assert!(all(
                acc.nrows() == lhs.nrows(),
                acc.ncols() == rhs.ncols(),
                lhs.ncols() == rhs.nrows(),
            ));

            let _ = parallelism;
            let m = acc.nrows();
            let n = acc.ncols();
            let k = lhs.ncols();

            let mut acc = acc;

            match alpha {
                Some(alpha) => {
                    if alpha != E::faer_one() {
                        zipped!(acc.rb_mut())
                            .for_each(|unzipped!(mut dst)| dst.write(dst.read().faer_mul(alpha)))
                    }
                }
                None => acc.fill_zero(),
            }

            Size::with2(m, n, |m, n| {
                Size::with(k, |k| {
                    let mut acc = constrained::MatMut::new(acc, m, n);
                    let lhs = constrained::MatRef::new(lhs, m, k);
                    let rhs = constrained::sparse::SparseColMatRef::new(rhs, k, n);

                    for i in m.indices() {
                        for j in n.indices() {
                            let mut acc_ij = E::faer_zero();
                            for (depth, rhs_kj) in zip(
                                rhs.row_indices_of_col(j),
                                SliceGroup::<'_, RhsE>::new(rhs.values_of_col(j)).into_ref_iter(),
                            ) {
                                let lhs_ik = lhs.read(i, depth);
                                acc_ij = acc_ij.faer_add(
                                    lhs_ik.canonicalize().faer_mul(rhs_kj.read().canonicalize()),
                                );
                            }

                            acc.write(i, j, acc.read(i, j).faer_add(beta.faer_mul(acc_ij)));
                        }
                    }
                });
            });
        }
    }
}

/// Immutable view over a column vector, similar to an immutable reference to a strided
/// [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `ColRef<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`ColRef::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
pub type ColRef<'a, E> = Matrix<DenseColRef<'a, E>>;

/// Immutable view over a row vector, similar to an immutable reference to a strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `RowRef<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`RowRef::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
pub type RowRef<'a, E> = Matrix<DenseRowRef<'a, E>>;

/// Immutable view over a matrix, similar to an immutable reference to a 2D strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `MatRef<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`MatRef::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
pub type MatRef<'a, E> = Matrix<DenseRef<'a, E>>;

/// Mutable view over a column vector, similar to a mutable reference to a strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `ColMut<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`ColMut::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
pub type ColMut<'a, E> = Matrix<DenseColMut<'a, E>>;

/// Mutable view over a row vector, similar to a mutable reference to a strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `RowMut<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`RowMut::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
pub type RowMut<'a, E> = Matrix<DenseRowMut<'a, E>>;

/// Mutable view over a matrix, similar to a mutable reference to a 2D strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `MatMut<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`MatMut::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
///
/// # Move semantics
/// Since `MatMut` mutably borrows data, it cannot be [`Copy`]. This means that if we pass a
/// `MatMut` to a function that takes it by value, or use a method that consumes `self` like
/// [`MatMut::transpose`], this renders the original variable unusable.
/// ```compile_fail
/// use faer_core::{Mat, MatMut};
///
/// fn takes_matmut(view: MatMut<'_, f64>) {}
///
/// let mut matrix = Mat::new();
/// let view = matrix.as_mut();
///
/// takes_matmut(view); // `view` is moved (passed by value)
/// takes_matmut(view); // this fails to compile since `view` was moved
/// ```
/// The way to get around it is to use the [`reborrow::ReborrowMut`] trait, which allows us to
/// mutably borrow a `MatMut` to obtain another `MatMut` for the lifetime of the borrow.
/// It's also similarly possible to immutably borrow a `MatMut` to obtain a `MatRef` for the
/// lifetime of the borrow, using [`reborrow::Reborrow`].
/// ```
/// use faer_core::{Mat, MatMut, MatRef};
/// use reborrow::*;
///
/// fn takes_matmut(view: MatMut<'_, f64>) {}
/// fn takes_matref(view: MatRef<'_, f64>) {}
///
/// let mut matrix = Mat::new();
/// let mut view = matrix.as_mut();
///
/// takes_matmut(view.rb_mut());
/// takes_matmut(view.rb_mut());
/// takes_matref(view.rb());
/// // view is still usable here
/// ```
pub type MatMut<'a, E> = Matrix<DenseMut<'a, E>>;

/// Wrapper around a scalar value that allows scalar multiplication by matrices.
pub type MatScale<E> = Matrix<Scale<E>>;

impl<E: Entity> MatScale<E> {
    #[inline(always)]
    pub fn new(value: E) -> Self {
        Self {
            inner: Scale(value),
        }
    }
    #[inline(always)]
    pub fn value(self) -> E {
        self.inner.0
    }
}
const __COL_REBORROW: () = {
    impl<'a, E: Entity> IntoConst for ColMut<'a, E> {
        type Target = ColRef<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            ColRef {
                inner: DenseColRef {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for ColMut<'a, E> {
        type Target = ColRef<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            ColRef {
                inner: DenseColRef {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for ColMut<'a, E> {
        type Target = ColMut<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            ColMut {
                inner: DenseColMut {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }
    }

    impl<'a, E: Entity> IntoConst for ColRef<'a, E> {
        type Target = ColRef<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            self
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for ColRef<'a, E> {
        type Target = ColRef<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for ColRef<'a, E> {
        type Target = ColRef<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }
};

const __ROW_REBORROW: () = {
    impl<'a, E: Entity> IntoConst for RowMut<'a, E> {
        type Target = RowRef<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            RowRef {
                inner: DenseRowRef {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for RowMut<'a, E> {
        type Target = RowRef<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            RowRef {
                inner: DenseRowRef {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for RowMut<'a, E> {
        type Target = RowMut<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            RowMut {
                inner: DenseRowMut {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }
    }

    impl<'a, E: Entity> IntoConst for RowRef<'a, E> {
        type Target = RowRef<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            self
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for RowRef<'a, E> {
        type Target = RowRef<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for RowRef<'a, E> {
        type Target = RowRef<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }
};

const __MAT_REBORROW: () = {
    impl<'a, E: Entity> IntoConst for MatMut<'a, E> {
        type Target = MatRef<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            MatRef {
                inner: DenseRef {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for MatMut<'a, E> {
        type Target = MatRef<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            MatRef {
                inner: DenseRef {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for MatMut<'a, E> {
        type Target = MatMut<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            MatMut {
                inner: DenseMut {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }
    }

    impl<'a, E: Entity> IntoConst for MatRef<'a, E> {
        type Target = MatRef<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            self
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for MatRef<'a, E> {
        type Target = MatRef<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for MatRef<'a, E> {
        type Target = MatRef<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }
};

impl<'a, E: Entity> IntoConst for Matrix<DiagMut<'a, E>> {
    type Target = Matrix<DiagRef<'a, E>>;

    #[inline(always)]
    fn into_const(self) -> Self::Target {
        Matrix {
            inner: DiagRef {
                inner: self.inner.inner.into_const(),
            },
        }
    }
}

impl<'short, 'a, E: Entity> Reborrow<'short> for Matrix<DiagMut<'a, E>> {
    type Target = Matrix<DiagRef<'short, E>>;

    #[inline(always)]
    fn rb(&'short self) -> Self::Target {
        Matrix {
            inner: DiagRef {
                inner: self.inner.inner.rb(),
            },
        }
    }
}

impl<'short, 'a, E: Entity> ReborrowMut<'short> for Matrix<DiagMut<'a, E>> {
    type Target = Matrix<DiagMut<'short, E>>;

    #[inline(always)]
    fn rb_mut(&'short mut self) -> Self::Target {
        Matrix {
            inner: DiagMut {
                inner: self.inner.inner.rb_mut(),
            },
        }
    }
}

impl<'a, E: Entity> IntoConst for Matrix<DiagRef<'a, E>> {
    type Target = Matrix<DiagRef<'a, E>>;

    #[inline(always)]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'short, 'a, E: Entity> Reborrow<'short> for Matrix<DiagRef<'a, E>> {
    type Target = Matrix<DiagRef<'short, E>>;

    #[inline(always)]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, 'a, E: Entity> ReborrowMut<'short> for Matrix<DiagRef<'a, E>> {
    type Target = Matrix<DiagRef<'short, E>>;

    #[inline(always)]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

unsafe impl<E: Entity + Send + Sync> Send for VecImpl<E> {}
unsafe impl<E: Entity + Send + Sync> Sync for VecImpl<E> {}
unsafe impl<E: Entity + Send + Sync> Send for VecOwnImpl<E> {}
unsafe impl<E: Entity + Send + Sync> Sync for VecOwnImpl<E> {}

unsafe impl<E: Entity + Send + Sync> Send for MatImpl<E> {}
unsafe impl<E: Entity + Send + Sync> Sync for MatImpl<E> {}
unsafe impl<E: Entity + Send + Sync> Send for MatOwnImpl<E> {}
unsafe impl<E: Entity + Send + Sync> Sync for MatOwnImpl<E> {}

#[doc(hidden)]
#[inline]
pub fn par_split_indices(n: usize, idx: usize, chunk_count: usize) -> (usize, usize) {
    let chunk_size = n / chunk_count;
    let rem = n % chunk_count;

    let idx_to_col_start = move |idx| {
        if idx < rem {
            idx * (chunk_size + 1)
        } else {
            rem + idx * chunk_size
        }
    };

    let start = idx_to_col_start(idx);
    let end = idx_to_col_start(idx + 1);
    (start, end - start)
}

mod seal {
    pub trait Seal {}
}
impl<'a, E: Entity> seal::Seal for MatRef<'a, E> {}
impl<'a, E: Entity> seal::Seal for MatMut<'a, E> {}
impl<'a, E: Entity> seal::Seal for ColRef<'a, E> {}
impl<'a, E: Entity> seal::Seal for ColMut<'a, E> {}
impl<'a, E: Entity> seal::Seal for RowRef<'a, E> {}
impl<'a, E: Entity> seal::Seal for RowMut<'a, E> {}

/// Represents a type that can be used to slice a row, such as an index or a range of indices.
pub trait RowIndex<ColRange>: seal::Seal + Sized {
    type Target;
    #[allow(clippy::missing_safety_doc)]
    unsafe fn get_unchecked(this: Self, row: ColRange) -> Self::Target {
        <Self as RowIndex<ColRange>>::get(this, row)
    }
    fn get(this: Self, row: ColRange) -> Self::Target;
}

/// Represents a type that can be used to slice a column, such as an index or a range of indices.
pub trait ColIndex<RowRange>: seal::Seal + Sized {
    type Target;
    #[allow(clippy::missing_safety_doc)]
    unsafe fn get_unchecked(this: Self, row: RowRange) -> Self::Target {
        <Self as ColIndex<RowRange>>::get(this, row)
    }
    fn get(this: Self, row: RowRange) -> Self::Target;
}

/// Represents a type that can be used to slice a matrix, such as an index or a range of indices.
pub trait MatIndex<RowRange, ColRange>: seal::Seal + Sized {
    type Target;
    #[allow(clippy::missing_safety_doc)]
    unsafe fn get_unchecked(this: Self, row: RowRange, col: ColRange) -> Self::Target {
        <Self as MatIndex<RowRange, ColRange>>::get(this, row, col)
    }
    fn get(this: Self, row: RowRange, col: ColRange) -> Self::Target;
}

const __MAT_INDEX: () = {
    // RangeFull
    // Range
    // RangeInclusive
    // RangeTo
    // RangeToInclusive
    // usize

    use core::ops::RangeFull;
    type Range = core::ops::Range<usize>;
    type RangeInclusive = core::ops::RangeInclusive<usize>;
    type RangeFrom = core::ops::RangeFrom<usize>;
    type RangeTo = core::ops::RangeTo<usize>;
    type RangeToInclusive = core::ops::RangeToInclusive<usize>;

    impl<E: Entity, RowRange> MatIndex<RowRange, RangeFrom> for MatRef<'_, E>
    where
        Self: MatIndex<RowRange, Range>,
    {
        type Target = <Self as MatIndex<RowRange, Range>>::Target;

        #[track_caller]
        #[inline(always)]
        fn get(
            this: Self,
            row: RowRange,
            col: RangeFrom,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
            let ncols = this.ncols();
            <Self as MatIndex<RowRange, Range>>::get(this, row, col.start..ncols)
        }
    }
    impl<E: Entity, RowRange> MatIndex<RowRange, RangeTo> for MatRef<'_, E>
    where
        Self: MatIndex<RowRange, Range>,
    {
        type Target = <Self as MatIndex<RowRange, Range>>::Target;

        #[track_caller]
        #[inline(always)]
        fn get(
            this: Self,
            row: RowRange,
            col: RangeTo,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
            <Self as MatIndex<RowRange, Range>>::get(this, row, 0..col.end)
        }
    }
    impl<E: Entity, RowRange> MatIndex<RowRange, RangeToInclusive> for MatRef<'_, E>
    where
        Self: MatIndex<RowRange, Range>,
    {
        type Target = <Self as MatIndex<RowRange, Range>>::Target;

        #[track_caller]
        #[inline(always)]
        fn get(
            this: Self,
            row: RowRange,
            col: RangeToInclusive,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
            assert!(col.end != usize::MAX);
            <Self as MatIndex<RowRange, Range>>::get(this, row, 0..col.end + 1)
        }
    }
    impl<E: Entity, RowRange> MatIndex<RowRange, RangeInclusive> for MatRef<'_, E>
    where
        Self: MatIndex<RowRange, Range>,
    {
        type Target = <Self as MatIndex<RowRange, Range>>::Target;

        #[track_caller]
        #[inline(always)]
        fn get(
            this: Self,
            row: RowRange,
            col: RangeInclusive,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
            assert!(*col.end() != usize::MAX);
            <Self as MatIndex<RowRange, Range>>::get(this, row, *col.start()..*col.end() + 1)
        }
    }
    impl<E: Entity, RowRange> MatIndex<RowRange, RangeFull> for MatRef<'_, E>
    where
        Self: MatIndex<RowRange, Range>,
    {
        type Target = <Self as MatIndex<RowRange, Range>>::Target;

        #[track_caller]
        #[inline(always)]
        fn get(
            this: Self,
            row: RowRange,
            col: RangeFull,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
            let _ = col;
            let ncols = this.ncols();
            <Self as MatIndex<RowRange, Range>>::get(this, row, 0..ncols)
        }
    }

    impl<E: Entity> MatIndex<RangeFull, Range> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFull, col: Range) -> Self {
            let _ = row;
            this.subcols(col.start, col.end - col.start)
        }
    }
    impl<'a, E: Entity> MatIndex<RangeFull, usize> for MatRef<'a, E> {
        type Target = ColRef<'a, E>;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFull, col: usize) -> Self::Target {
            let _ = row;
            this.col(col)
        }
    }

    impl<E: Entity> MatIndex<Range, Range> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: Range, col: Range) -> Self {
            this.submatrix(
                row.start,
                col.start,
                row.end - row.start,
                col.end - col.start,
            )
        }
    }
    impl<E: Entity> MatIndex<Range, usize> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: Range, col: usize) -> Self {
            this.submatrix(row.start, col, row.end - row.start, 1)
        }
    }

    impl<E: Entity> MatIndex<RangeInclusive, Range> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeInclusive, col: Range) -> Self {
            assert!(*row.end() != usize::MAX);
            <Self as MatIndex<Range, Range>>::get(this, *row.start()..*row.end() + 1, col)
        }
    }
    impl<E: Entity> MatIndex<RangeInclusive, usize> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeInclusive, col: usize) -> Self {
            assert!(*row.end() != usize::MAX);
            <Self as MatIndex<Range, usize>>::get(this, *row.start()..*row.end() + 1, col)
        }
    }

    impl<E: Entity> MatIndex<RangeFrom, Range> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFrom, col: Range) -> Self {
            let nrows = this.nrows();
            <Self as MatIndex<Range, Range>>::get(this, row.start..nrows, col)
        }
    }
    impl<E: Entity> MatIndex<RangeFrom, usize> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFrom, col: usize) -> Self {
            let nrows = this.nrows();
            <Self as MatIndex<Range, usize>>::get(this, row.start..nrows, col)
        }
    }
    impl<E: Entity> MatIndex<RangeTo, Range> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeTo, col: Range) -> Self {
            <Self as MatIndex<Range, Range>>::get(this, 0..row.end, col)
        }
    }
    impl<E: Entity> MatIndex<RangeTo, usize> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeTo, col: usize) -> Self {
            <Self as MatIndex<Range, usize>>::get(this, 0..row.end, col)
        }
    }

    impl<E: Entity> MatIndex<RangeToInclusive, Range> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeToInclusive, col: Range) -> Self {
            assert!(row.end != usize::MAX);
            <Self as MatIndex<Range, Range>>::get(this, 0..row.end + 1, col)
        }
    }
    impl<E: Entity> MatIndex<RangeToInclusive, usize> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeToInclusive, col: usize) -> Self {
            assert!(row.end != usize::MAX);
            <Self as MatIndex<Range, usize>>::get(this, 0..row.end + 1, col)
        }
    }

    impl<E: Entity> MatIndex<usize, Range> for MatRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: usize, col: Range) -> Self {
            this.submatrix(row, col.start, 1, col.end - col.start)
        }
    }

    impl<E: Entity, RowRange> MatIndex<RowRange, RangeFrom> for MatMut<'_, E>
    where
        Self: MatIndex<RowRange, Range>,
    {
        type Target = <Self as MatIndex<RowRange, Range>>::Target;

        #[track_caller]
        #[inline(always)]
        fn get(
            this: Self,
            row: RowRange,
            col: RangeFrom,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
            let ncols = this.ncols();
            <Self as MatIndex<RowRange, Range>>::get(this, row, col.start..ncols)
        }
    }
    impl<E: Entity, RowRange> MatIndex<RowRange, RangeTo> for MatMut<'_, E>
    where
        Self: MatIndex<RowRange, Range>,
    {
        type Target = <Self as MatIndex<RowRange, Range>>::Target;

        #[track_caller]
        #[inline(always)]
        fn get(
            this: Self,
            row: RowRange,
            col: RangeTo,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
            <Self as MatIndex<RowRange, Range>>::get(this, row, 0..col.end)
        }
    }
    impl<E: Entity, RowRange> MatIndex<RowRange, RangeToInclusive> for MatMut<'_, E>
    where
        Self: MatIndex<RowRange, Range>,
    {
        type Target = <Self as MatIndex<RowRange, Range>>::Target;

        #[track_caller]
        #[inline(always)]
        fn get(
            this: Self,
            row: RowRange,
            col: RangeToInclusive,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
            assert!(col.end != usize::MAX);
            <Self as MatIndex<RowRange, Range>>::get(this, row, 0..col.end + 1)
        }
    }
    impl<E: Entity, RowRange> MatIndex<RowRange, RangeInclusive> for MatMut<'_, E>
    where
        Self: MatIndex<RowRange, Range>,
    {
        type Target = <Self as MatIndex<RowRange, Range>>::Target;

        #[track_caller]
        #[inline(always)]
        fn get(
            this: Self,
            row: RowRange,
            col: RangeInclusive,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
            assert!(*col.end() != usize::MAX);
            <Self as MatIndex<RowRange, Range>>::get(this, row, *col.start()..*col.end() + 1)
        }
    }
    impl<E: Entity, RowRange> MatIndex<RowRange, RangeFull> for MatMut<'_, E>
    where
        Self: MatIndex<RowRange, Range>,
    {
        type Target = <Self as MatIndex<RowRange, Range>>::Target;

        #[track_caller]
        #[inline(always)]
        fn get(
            this: Self,
            row: RowRange,
            col: RangeFull,
        ) -> <Self as MatIndex<RowRange, Range>>::Target {
            let _ = col;
            let ncols = this.ncols();
            <Self as MatIndex<RowRange, Range>>::get(this, row, 0..ncols)
        }
    }

    impl<E: Entity> MatIndex<RangeFull, Range> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFull, col: Range) -> Self {
            let _ = row;
            this.subcols_mut(col.start, col.end - col.start)
        }
    }
    impl<'a, E: Entity> MatIndex<RangeFull, usize> for MatMut<'a, E> {
        type Target = ColMut<'a, E>;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFull, col: usize) -> Self::Target {
            let _ = row;
            this.col_mut(col)
        }
    }

    impl<E: Entity> MatIndex<Range, Range> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: Range, col: Range) -> Self {
            this.submatrix_mut(
                row.start,
                col.start,
                row.end - row.start,
                col.end - col.start,
            )
        }
    }
    impl<E: Entity> MatIndex<Range, usize> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: Range, col: usize) -> Self {
            this.submatrix_mut(row.start, col, row.end - row.start, 1)
        }
    }

    impl<E: Entity> MatIndex<RangeInclusive, Range> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeInclusive, col: Range) -> Self {
            assert!(*row.end() != usize::MAX);
            <Self as MatIndex<Range, Range>>::get(this, *row.start()..*row.end() + 1, col)
        }
    }
    impl<E: Entity> MatIndex<RangeInclusive, usize> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeInclusive, col: usize) -> Self {
            assert!(*row.end() != usize::MAX);
            <Self as MatIndex<Range, usize>>::get(this, *row.start()..*row.end() + 1, col)
        }
    }

    impl<E: Entity> MatIndex<RangeFrom, Range> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFrom, col: Range) -> Self {
            let nrows = this.nrows();
            <Self as MatIndex<Range, Range>>::get(this, row.start..nrows, col)
        }
    }
    impl<E: Entity> MatIndex<RangeFrom, usize> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFrom, col: usize) -> Self {
            let nrows = this.nrows();
            <Self as MatIndex<Range, usize>>::get(this, row.start..nrows, col)
        }
    }
    impl<E: Entity> MatIndex<RangeTo, Range> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeTo, col: Range) -> Self {
            <Self as MatIndex<Range, Range>>::get(this, 0..row.end, col)
        }
    }
    impl<E: Entity> MatIndex<RangeTo, usize> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeTo, col: usize) -> Self {
            <Self as MatIndex<Range, usize>>::get(this, 0..row.end, col)
        }
    }

    impl<E: Entity> MatIndex<RangeToInclusive, Range> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeToInclusive, col: Range) -> Self {
            assert!(row.end != usize::MAX);
            <Self as MatIndex<Range, Range>>::get(this, 0..row.end + 1, col)
        }
    }
    impl<E: Entity> MatIndex<RangeToInclusive, usize> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeToInclusive, col: usize) -> Self {
            assert!(row.end != usize::MAX);
            <Self as MatIndex<Range, usize>>::get(this, 0..row.end + 1, col)
        }
    }

    impl<E: Entity> MatIndex<usize, Range> for MatMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: usize, col: Range) -> Self {
            this.submatrix_mut(row, col.start, 1, col.end - col.start)
        }
    }

    impl<'a, E: Entity> MatIndex<usize, usize> for MatRef<'a, E> {
        type Target = GroupFor<E, &'a E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, row: usize, col: usize) -> Self::Target {
            unsafe { E::faer_map(this.ptr_inbounds_at(row, col), |ptr| &*ptr) }
        }

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: usize, col: usize) -> Self::Target {
            assert!(all(row < this.nrows(), col < this.ncols()));
            unsafe { <Self as MatIndex<usize, usize>>::get_unchecked(this, row, col) }
        }
    }

    impl<'a, E: Entity> MatIndex<usize, usize> for MatMut<'a, E> {
        type Target = GroupFor<E, &'a mut E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, row: usize, col: usize) -> Self::Target {
            unsafe { E::faer_map(this.ptr_inbounds_at_mut(row, col), |ptr| &mut *ptr) }
        }

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: usize, col: usize) -> Self::Target {
            assert!(all(row < this.nrows(), col < this.ncols()));
            unsafe { <Self as MatIndex<usize, usize>>::get_unchecked(this, row, col) }
        }
    }
};

const __COL_INDEX: () = {
    // RangeFull
    // Range
    // RangeInclusive
    // RangeTo
    // RangeToInclusive
    // usize

    use core::ops::RangeFull;
    type Range = core::ops::Range<usize>;
    type RangeInclusive = core::ops::RangeInclusive<usize>;
    type RangeFrom = core::ops::RangeFrom<usize>;
    type RangeTo = core::ops::RangeTo<usize>;
    type RangeToInclusive = core::ops::RangeToInclusive<usize>;

    impl<E: Entity> ColIndex<RangeFull> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFull) -> Self {
            let _ = row;
            this
        }
    }

    impl<E: Entity> ColIndex<Range> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: Range) -> Self {
            this.subrows(row.start, row.end - row.start)
        }
    }

    impl<E: Entity> ColIndex<RangeInclusive> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeInclusive) -> Self {
            assert!(*row.end() != usize::MAX);
            <Self as ColIndex<Range>>::get(this, *row.start()..*row.end() + 1)
        }
    }

    impl<E: Entity> ColIndex<RangeFrom> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFrom) -> Self {
            let nrows = this.nrows();
            <Self as ColIndex<Range>>::get(this, row.start..nrows)
        }
    }
    impl<E: Entity> ColIndex<RangeTo> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeTo) -> Self {
            <Self as ColIndex<Range>>::get(this, 0..row.end)
        }
    }

    impl<E: Entity> ColIndex<RangeToInclusive> for ColRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeToInclusive) -> Self {
            assert!(row.end != usize::MAX);
            <Self as ColIndex<Range>>::get(this, 0..row.end + 1)
        }
    }

    impl<'a, E: Entity> ColIndex<usize> for ColRef<'a, E> {
        type Target = GroupFor<E, &'a E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, row: usize) -> Self::Target {
            unsafe { E::faer_map(this.ptr_inbounds_at(row), |ptr: *const _| &*ptr) }
        }

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: usize) -> Self::Target {
            assert!(row < this.nrows());
            unsafe { <Self as ColIndex<usize>>::get_unchecked(this, row) }
        }
    }

    impl<E: Entity> ColIndex<RangeFull> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFull) -> Self {
            let _ = row;
            this
        }
    }

    impl<E: Entity> ColIndex<Range> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: Range) -> Self {
            this.subrows_mut(row.start, row.end - row.start)
        }
    }

    impl<E: Entity> ColIndex<RangeInclusive> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeInclusive) -> Self {
            assert!(*row.end() != usize::MAX);
            <Self as ColIndex<Range>>::get(this, *row.start()..*row.end() + 1)
        }
    }

    impl<E: Entity> ColIndex<RangeFrom> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeFrom) -> Self {
            let nrows = this.nrows();
            <Self as ColIndex<Range>>::get(this, row.start..nrows)
        }
    }
    impl<E: Entity> ColIndex<RangeTo> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeTo) -> Self {
            <Self as ColIndex<Range>>::get(this, 0..row.end)
        }
    }

    impl<E: Entity> ColIndex<RangeToInclusive> for ColMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: RangeToInclusive) -> Self {
            assert!(row.end != usize::MAX);
            <Self as ColIndex<Range>>::get(this, 0..row.end + 1)
        }
    }

    impl<'a, E: Entity> ColIndex<usize> for ColMut<'a, E> {
        type Target = GroupFor<E, &'a mut E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, row: usize) -> Self::Target {
            unsafe { E::faer_map(this.ptr_inbounds_at_mut(row), |ptr: *mut _| &mut *ptr) }
        }

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: usize) -> Self::Target {
            assert!(row < this.nrows());
            unsafe { <Self as ColIndex<usize>>::get_unchecked(this, row) }
        }
    }
};

const __ROW_INDEX: () = {
    // RangeFull
    // Range
    // RangeInclusive
    // RangeTo
    // RangeToInclusive
    // usize

    use core::ops::RangeFull;
    type Range = core::ops::Range<usize>;
    type RangeInclusive = core::ops::RangeInclusive<usize>;
    type RangeFrom = core::ops::RangeFrom<usize>;
    type RangeTo = core::ops::RangeTo<usize>;
    type RangeToInclusive = core::ops::RangeToInclusive<usize>;

    impl<E: Entity> RowIndex<RangeFull> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeFull) -> Self {
            let _ = col;
            this
        }
    }

    impl<E: Entity> RowIndex<Range> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: Range) -> Self {
            this.subcols(col.start, col.end - col.start)
        }
    }

    impl<E: Entity> RowIndex<RangeInclusive> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeInclusive) -> Self {
            assert!(*col.end() != usize::MAX);
            <Self as RowIndex<Range>>::get(this, *col.start()..*col.end() + 1)
        }
    }

    impl<E: Entity> RowIndex<RangeFrom> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeFrom) -> Self {
            let ncols = this.ncols();
            <Self as RowIndex<Range>>::get(this, col.start..ncols)
        }
    }
    impl<E: Entity> RowIndex<RangeTo> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeTo) -> Self {
            <Self as RowIndex<Range>>::get(this, 0..col.end)
        }
    }

    impl<E: Entity> RowIndex<RangeToInclusive> for RowRef<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeToInclusive) -> Self {
            assert!(col.end != usize::MAX);
            <Self as RowIndex<Range>>::get(this, 0..col.end + 1)
        }
    }

    impl<'a, E: Entity> RowIndex<usize> for RowRef<'a, E> {
        type Target = GroupFor<E, &'a E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, col: usize) -> Self::Target {
            unsafe { E::faer_map(this.ptr_inbounds_at(col), |ptr: *const _| &*ptr) }
        }

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: usize) -> Self::Target {
            assert!(col < this.ncols());
            unsafe { <Self as RowIndex<usize>>::get_unchecked(this, col) }
        }
    }

    impl<E: Entity> RowIndex<RangeFull> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeFull) -> Self {
            let _ = col;
            this
        }
    }

    impl<E: Entity> RowIndex<Range> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: Range) -> Self {
            this.subcols_mut(col.start, col.end - col.start)
        }
    }

    impl<E: Entity> RowIndex<RangeInclusive> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeInclusive) -> Self {
            assert!(*col.end() != usize::MAX);
            <Self as RowIndex<Range>>::get(this, *col.start()..*col.end() + 1)
        }
    }

    impl<E: Entity> RowIndex<RangeFrom> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeFrom) -> Self {
            let ncols = this.ncols();
            <Self as RowIndex<Range>>::get(this, col.start..ncols)
        }
    }

    impl<E: Entity> RowIndex<RangeTo> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeTo) -> Self {
            <Self as RowIndex<Range>>::get(this, 0..col.end)
        }
    }

    impl<E: Entity> RowIndex<RangeToInclusive> for RowMut<'_, E> {
        type Target = Self;

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: RangeToInclusive) -> Self {
            assert!(col.end != usize::MAX);
            <Self as RowIndex<Range>>::get(this, 0..col.end + 1)
        }
    }

    impl<'a, E: Entity> RowIndex<usize> for RowMut<'a, E> {
        type Target = GroupFor<E, &'a mut E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, col: usize) -> Self::Target {
            unsafe { E::faer_map(this.ptr_inbounds_at_mut(col), |ptr: *mut _| &mut *ptr) }
        }

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, col: usize) -> Self::Target {
            assert!(col < this.ncols());
            unsafe { <Self as RowIndex<usize>>::get_unchecked(this, col) }
        }
    }
};

impl<'a, E: Entity> Matrix<DiagRef<'a, E>> {
    #[inline(always)]
    #[deprecated = "replaced by `Matrix<DiagRef<'_, E>>::column_vector`"]
    pub fn into_column_vector(self) -> ColRef<'a, E> {
        self.inner.inner
    }

    #[inline(always)]
    pub fn column_vector(self) -> ColRef<'a, E> {
        self.inner.inner
    }
}

impl<'a, E: Entity> Matrix<DiagMut<'a, E>> {
    #[inline(always)]
    #[deprecated = "replaced by `Matrix<DiagRef<'_, E>>::column_vector_mut`"]
    pub fn into_column_vector(self) -> ColMut<'a, E> {
        self.inner.inner
    }

    #[inline(always)]
    pub fn column_vector_mut(self) -> ColMut<'a, E> {
        self.inner.inner
    }
}

impl<E: Entity> Matrix<DiagOwn<E>> {
    #[inline(always)]
    pub fn into_column_vector(self) -> Col<E> {
        self.inner.inner
    }

    #[inline(always)]
    pub fn as_ref(&self) -> Matrix<DiagRef<'_, E>> {
        Matrix {
            inner: DiagRef {
                inner: self.inner.inner.as_ref(),
            },
        }
    }

    #[inline(always)]
    pub fn as_mut(&mut self) -> Matrix<DiagMut<'_, E>> {
        Matrix {
            inner: DiagMut {
                inner: self.inner.inner.as_mut(),
            },
        }
    }
}

#[track_caller]
#[inline]
fn from_slice_assert(nrows: usize, ncols: usize, len: usize) {
    // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
    // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
    // we don't care
    let size = usize::checked_mul(nrows, ncols).unwrap_or(usize::MAX);
    assert!(size == len);
}

#[track_caller]
#[inline]
fn from_strided_column_major_slice_assert(
    nrows: usize,
    ncols: usize,
    col_stride: usize,
    len: usize,
) {
    // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
    // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
    // we don't care
    let last = usize::checked_mul(col_stride, ncols - 1)
        .and_then(|last_col| last_col.checked_add(nrows - 1))
        .unwrap_or(usize::MAX);
    assert!(last < len);
}

#[track_caller]
#[inline]
fn from_strided_column_major_slice_mut_assert(
    nrows: usize,
    ncols: usize,
    col_stride: usize,
    len: usize,
) {
    // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
    // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
    // we don't care
    let last = usize::checked_mul(col_stride, ncols - 1)
        .and_then(|last_col| last_col.checked_add(nrows - 1))
        .unwrap_or(usize::MAX);
    assert!(all(col_stride >= nrows, last < len));
}

#[inline(always)]
unsafe fn unchecked_mul(a: usize, b: isize) -> isize {
    let (sum, overflow) = (a as isize).overflowing_mul(b);
    if overflow {
        core::hint::unreachable_unchecked();
    }
    sum
}

#[inline(always)]
unsafe fn unchecked_add(a: isize, b: isize) -> isize {
    let (sum, overflow) = a.overflowing_add(b);
    if overflow {
        core::hint::unreachable_unchecked();
    }
    sum
}

const __COL_IMPL: () = {
    impl<'a, E: Entity> ColRef<'a, E> {
        #[track_caller]
        #[inline(always)]
        #[doc(hidden)]
        pub fn try_get_contiguous_col(self) -> GroupFor<E, &'a [E::Unit]> {
            assert!(self.row_stride() == 1);
            let m = self.nrows();
            E::faer_map(
                self.as_ptr(),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
            )
        }

        #[inline(always)]
        pub fn nrows(&self) -> usize {
            self.inner.inner.len
        }
        #[inline(always)]
        pub fn ncols(&self) -> usize {
            1
        }

        /// Returns pointers to the matrix data.
        #[inline(always)]
        pub fn as_ptr(self) -> GroupFor<E, *const E::Unit> {
            E::faer_map(
                from_copy::<E, _>(self.inner.inner.ptr),
                #[inline(always)]
                |ptr| ptr.as_ptr() as *const E::Unit,
            )
        }

        /// Returns the row stride of the matrix, specified in number of elements, not in bytes.
        #[inline(always)]
        pub fn row_stride(&self) -> isize {
            self.inner.inner.stride
        }

        /// Returns `self` as a matrix view.
        #[inline(always)]
        pub fn as_2d(self) -> MatRef<'a, E> {
            let nrows = self.nrows();
            let row_stride = self.row_stride();
            unsafe { mat::from_raw_parts(self.as_ptr(), nrows, 1, row_stride, 0) }
        }

        /// Returns raw pointers to the element at the given index.
        #[inline(always)]
        pub fn ptr_at(self, row: usize) -> GroupFor<E, *const E::Unit> {
            let offset = (row as isize).wrapping_mul(self.inner.inner.stride);

            E::faer_map(
                self.as_ptr(),
                #[inline(always)]
                |ptr| ptr.wrapping_offset(offset),
            )
        }

        #[inline(always)]
        unsafe fn unchecked_ptr_at(self, row: usize) -> GroupFor<E, *const E::Unit> {
            let offset = unchecked_mul(row, self.inner.inner.stride);
            E::faer_map(
                self.as_ptr(),
                #[inline(always)]
                |ptr| ptr.offset(offset),
            )
        }

        #[inline(always)]
        unsafe fn overflowing_ptr_at(self, row: usize) -> GroupFor<E, *const E::Unit> {
            unsafe {
                let cond = row != self.nrows();
                let offset = (cond as usize).wrapping_neg() as isize
                    & (row as isize).wrapping_mul(self.inner.inner.stride);
                E::faer_map(
                    self.as_ptr(),
                    #[inline(always)]
                    |ptr| ptr.offset(offset),
                )
            }
        }

        /// Returns raw pointers to the element at the given index, assuming the provided index
        /// is within the size of the vector.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn ptr_inbounds_at(self, row: usize) -> GroupFor<E, *const E::Unit> {
            debug_assert!(row < self.nrows());
            self.unchecked_ptr_at(row)
        }

        /// Splits the column vector at the given index into two parts and
        /// returns an array of each subvector, in the following order:
        /// * top.
        /// * bottom.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row <= self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at_unchecked(self, row: usize) -> (Self, Self) {
            debug_assert!(row <= self.nrows());

            let row_stride = self.row_stride();

            let nrows = self.nrows();

            unsafe {
                let top = self.as_ptr();
                let bot = self.overflowing_ptr_at(row);

                (
                    col::from_raw_parts(top, row, row_stride),
                    col::from_raw_parts(bot, nrows - row, row_stride),
                )
            }
        }

        /// Splits the column vector at the given index into two parts and
        /// returns an array of each subvector, in the following order:
        /// * top.
        /// * bottom.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row <= self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at(self, row: usize) -> (Self, Self) {
            assert!(row <= self.nrows());
            unsafe { self.split_at_unchecked(row) }
        }

        /// Returns references to the element at the given index, or subvector if `row` is a
        /// range.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row` must be contained in `[0, self.nrows())`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn get_unchecked<RowRange>(
            self,
            row: RowRange,
        ) -> <Self as ColIndex<RowRange>>::Target
        where
            Self: ColIndex<RowRange>,
        {
            <Self as ColIndex<RowRange>>::get_unchecked(self, row)
        }

        /// Returns references to the element at the given index, or subvector if `row` is a
        /// range, with bound checks.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row` must be contained in `[0, self.nrows())`.
        #[inline(always)]
        #[track_caller]
        pub fn get<RowRange>(self, row: RowRange) -> <Self as ColIndex<RowRange>>::Target
        where
            Self: ColIndex<RowRange>,
        {
            <Self as ColIndex<RowRange>>::get(self, row)
        }

        /// Reads the value of the element at the given index.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn read_unchecked(&self, row: usize) -> E {
            E::faer_from_units(E::faer_map(
                self.get_unchecked(row),
                #[inline(always)]
                |ptr| *ptr,
            ))
        }

        /// Reads the value of the element at the given index, with bound checks.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub fn read(&self, row: usize) -> E {
            E::faer_from_units(E::faer_map(
                self.get(row),
                #[inline(always)]
                |ptr| *ptr,
            ))
        }

        /// Returns a view over the transpose of `self`.
        #[inline(always)]
        #[must_use]
        pub fn transpose(self) -> RowRef<'a, E> {
            unsafe { row::from_raw_parts(self.as_ptr(), self.nrows(), self.row_stride()) }
        }

        /// Returns a view over the conjugate of `self`.
        #[inline(always)]
        #[must_use]
        pub fn conjugate(self) -> ColRef<'a, E::Conj>
        where
            E: Conjugate,
        {
            unsafe {
                // SAFETY: Conjugate requires that E::Unit and E::Conj::Unit have the same layout
                // and that GroupCopyFor<E,X> == E::Conj::GroupCopy<X>
                col::from_raw_parts::<'_, E::Conj>(
                    transmute_unchecked::<
                        GroupFor<E, *const UnitFor<E>>,
                        GroupFor<E::Conj, *const UnitFor<E::Conj>>,
                    >(self.as_ptr()),
                    self.nrows(),
                    self.row_stride(),
                )
            }
        }

        /// Returns a view over the conjugate transpose of `self`.
        #[inline(always)]
        pub fn adjoint(self) -> RowRef<'a, E::Conj>
        where
            E: Conjugate,
        {
            self.conjugate().transpose()
        }

        /// Returns a view over the canonical representation of `self`, as well as a flag declaring
        /// whether `self` is implicitly conjugated or not.
        #[inline(always)]
        pub fn canonicalize(self) -> (ColRef<'a, E::Canonical>, Conj)
        where
            E: Conjugate,
        {
            (
                unsafe {
                    // SAFETY: see Self::conjugate
                    col::from_raw_parts::<'_, E::Canonical>(
                        transmute_unchecked::<
                            GroupFor<E, *const E::Unit>,
                            GroupFor<E::Canonical, *const UnitFor<E::Canonical>>,
                        >(self.as_ptr()),
                        self.nrows(),
                        self.row_stride(),
                    )
                },
                if coe::is_same::<E, E::Canonical>() {
                    Conj::No
                } else {
                    Conj::Yes
                },
            )
        }

        /// Returns a view over the `self`, with the rows in reversed order.
        #[inline(always)]
        #[must_use]
        pub fn reverse_rows(self) -> Self {
            let nrows = self.nrows();
            let row_stride = self.row_stride().wrapping_neg();

            let ptr = unsafe { self.unchecked_ptr_at(nrows.saturating_sub(1)) };
            unsafe { col::from_raw_parts(ptr, nrows, row_stride) }
        }

        /// Returns a view over the subvector starting at row `row_start`, and with number of rows
        /// `nrows`.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row_start <= self.nrows()`.
        /// * `nrows <= self.nrows() - row_start`.
        #[track_caller]
        #[inline(always)]
        pub unsafe fn subrows_unchecked(self, row_start: usize, nrows: usize) -> Self {
            debug_assert!(all(
                row_start <= self.nrows(),
                nrows <= self.nrows() - row_start
            ));
            let row_stride = self.row_stride();
            unsafe { col::from_raw_parts(self.overflowing_ptr_at(row_start), nrows, row_stride) }
        }

        /// Returns a view over the subvector starting at row `row_start`, and with number of rows
        /// `nrows`.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row_start <= self.nrows()`.
        /// * `nrows <= self.nrows() - row_start`.
        #[track_caller]
        #[inline(always)]
        pub fn subrows(self, row_start: usize, nrows: usize) -> Self {
            assert!(all(
                row_start <= self.nrows(),
                nrows <= self.nrows() - row_start
            ));
            unsafe { self.subrows_unchecked(row_start, nrows) }
        }

        /// Given a matrix with a single column, returns an object that interprets
        /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
        #[track_caller]
        #[inline(always)]
        pub fn column_vector_as_diagonal(self) -> Matrix<DiagRef<'a, E>> {
            Matrix {
                inner: DiagRef { inner: self },
            }
        }

        /// Returns an owning [`Col`] of the data.
        #[inline]
        pub fn to_owned(&self) -> Col<E::Canonical>
        where
            E: Conjugate,
        {
            let mut mat = Col::new();
            mat.resize_with(
                self.nrows(),
                #[inline(always)]
                |row| unsafe { self.read_unchecked(row).canonicalize() },
            );
            mat
        }

        /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
        #[inline]
        pub fn has_nan(&self) -> bool
        where
            E: ComplexField,
        {
            (*self).as_2d().has_nan()
        }

        /// Returns `true` if all of the elements are finite, otherwise returns `false`.
        #[inline]
        pub fn is_all_finite(&self) -> bool
        where
            E: ComplexField,
        {
            (*self).rb().as_2d().is_all_finite()
        }

        /// Returns the maximum norm of `self`.
        #[inline]
        pub fn norm_max(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_max((*self).rb().as_2d())
        }
        /// Returns the L2 norm of `self`.
        #[inline]
        pub fn norm_l2(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_l2((*self).rb().as_2d())
        }

        /// Returns a view over the matrix.
        #[inline]
        pub fn as_ref(&self) -> ColRef<'_, E> {
            *self
        }

        #[doc(hidden)]
        #[inline(always)]
        pub unsafe fn const_cast(self) -> ColMut<'a, E> {
            ColMut {
                inner: DenseColMut {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }
    }

    impl<E: SimpleEntity> core::ops::Index<usize> for ColRef<'_, E> {
        type Output = E;

        #[inline]
        #[track_caller]
        fn index(&self, row: usize) -> &E {
            self.get(row)
        }
    }

    impl<E: SimpleEntity> core::ops::Index<usize> for ColMut<'_, E> {
        type Output = E;

        #[inline]
        #[track_caller]
        fn index(&self, row: usize) -> &E {
            (*self).rb().get(row)
        }
    }

    impl<E: SimpleEntity> core::ops::IndexMut<usize> for ColMut<'_, E> {
        #[inline]
        #[track_caller]
        fn index_mut(&mut self, row: usize) -> &mut E {
            (*self).rb_mut().get_mut(row)
        }
    }

    impl<E: SimpleEntity> core::ops::Index<usize> for Col<E> {
        type Output = E;

        #[inline]
        #[track_caller]
        fn index(&self, row: usize) -> &E {
            self.as_ref().get(row)
        }
    }

    impl<E: SimpleEntity> core::ops::IndexMut<usize> for Col<E> {
        #[inline]
        #[track_caller]
        fn index_mut(&mut self, row: usize) -> &mut E {
            self.as_mut().get_mut(row)
        }
    }

    impl<'a, E: Entity> ColMut<'a, E> {
        #[track_caller]
        #[inline(always)]
        #[doc(hidden)]
        pub fn try_get_contiguous_col_mut(self) -> GroupFor<E, &'a mut [E::Unit]> {
            assert!(self.row_stride() == 1);
            let m = self.nrows();
            E::faer_map(
                self.as_ptr_mut(),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
            )
        }

        #[inline(always)]
        pub fn nrows(&self) -> usize {
            self.inner.inner.len
        }
        #[inline(always)]
        pub fn ncols(&self) -> usize {
            1
        }

        /// Returns pointers to the matrix data.
        #[inline(always)]
        pub fn as_ptr_mut(self) -> GroupFor<E, *mut E::Unit> {
            E::faer_map(
                from_copy::<E, _>(self.inner.inner.ptr),
                #[inline(always)]
                |ptr| ptr.as_ptr() as *mut E::Unit,
            )
        }

        /// Returns the row stride of the matrix, specified in number of elements, not in bytes.
        #[inline(always)]
        pub fn row_stride(&self) -> isize {
            self.inner.inner.stride
        }

        /// Returns `self` as a mutable matrix view.
        #[inline(always)]
        pub fn as_2d_mut(self) -> MatMut<'a, E> {
            let nrows = self.nrows();
            let row_stride = self.row_stride();
            unsafe { mat::from_raw_parts_mut(self.as_ptr_mut(), nrows, 1, row_stride, 0) }
        }

        /// Returns raw pointers to the element at the given index.
        #[inline(always)]
        pub fn ptr_at_mut(self, row: usize) -> GroupFor<E, *mut E::Unit> {
            let offset = (row as isize).wrapping_mul(self.inner.inner.stride);

            E::faer_map(
                self.as_ptr_mut(),
                #[inline(always)]
                |ptr| ptr.wrapping_offset(offset),
            )
        }

        #[inline(always)]
        unsafe fn ptr_at_mut_unchecked(self, row: usize) -> GroupFor<E, *mut E::Unit> {
            let offset = unchecked_mul(row, self.inner.inner.stride);
            E::faer_map(
                self.as_ptr_mut(),
                #[inline(always)]
                |ptr| ptr.offset(offset),
            )
        }

        /// Returns raw pointers to the element at the given index, assuming the provided index
        /// is within the size of the vector.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn ptr_inbounds_at_mut(self, row: usize) -> GroupFor<E, *mut E::Unit> {
            debug_assert!(row < self.nrows());
            self.ptr_at_mut_unchecked(row)
        }

        /// Splits the column vector at the given index into two parts and
        /// returns an array of each subvector, in the following order:
        /// * top.
        /// * bottom.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row <= self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at_mut_unchecked(self, row: usize) -> (Self, Self) {
            let (top, bot) = self.into_const().split_at_unchecked(row);
            unsafe { (top.const_cast(), bot.const_cast()) }
        }

        /// Splits the column vector at the given index into two parts and
        /// returns an array of each subvector, in the following order:
        /// * top.
        /// * bottom.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row <= self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub fn split_at_mut(self, row: usize) -> (Self, Self) {
            assert!(row <= self.nrows());
            unsafe { self.split_at_mut_unchecked(row) }
        }

        /// Returns references to the element at the given index, or subvector if `row` is a
        /// range.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row` must be contained in `[0, self.nrows())`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn get_unchecked_mut<RowRange>(
            self,
            row: RowRange,
        ) -> <Self as ColIndex<RowRange>>::Target
        where
            Self: ColIndex<RowRange>,
        {
            <Self as ColIndex<RowRange>>::get_unchecked(self, row)
        }

        /// Returns references to the element at the given index, or subvector if `row` is a
        /// range, with bound checks.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row` must be contained in `[0, self.nrows())`.
        #[inline(always)]
        #[track_caller]
        pub fn get_mut<RowRange>(self, row: RowRange) -> <Self as ColIndex<RowRange>>::Target
        where
            Self: ColIndex<RowRange>,
        {
            <Self as ColIndex<RowRange>>::get(self, row)
        }

        /// Reads the value of the element at the given index.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn read_unchecked(&self, row: usize) -> E {
            self.rb().read_unchecked(row)
        }

        /// Reads the value of the element at the given index, with bound checks.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub fn read(&self, row: usize) -> E {
            self.rb().read(row)
        }

        /// Writes the value to the element at the given index.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn write_unchecked(&mut self, row: usize, value: E) {
            let units = value.faer_into_units();
            let zipped = E::faer_zip(units, (*self).rb_mut().ptr_inbounds_at_mut(row));
            E::faer_map(
                zipped,
                #[inline(always)]
                |(unit, ptr)| *ptr = unit,
            );
        }

        /// Writes the value to the element at the given index, with bound checks.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub fn write(&mut self, row: usize, value: E) {
            assert!(row < self.nrows());
            unsafe { self.write_unchecked(row, value) };
        }

        /// Copies the values from `other` into `self`.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `self.nrows() == other.nrows()`.
        /// * `self.ncols() == other.ncols()`.
        #[track_caller]
        pub fn copy_from(&mut self, other: impl AsColRef<E>) {
            #[track_caller]
            #[inline(always)]
            fn implementation<E: Entity>(this: ColMut<'_, E>, other: ColRef<'_, E>) {
                zipped!(this.as_2d_mut(), other.as_2d())
                    .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
            }
            implementation(self.rb_mut(), other.as_col_ref())
        }

        /// Fills the elements of `self` with zeros.
        #[track_caller]
        pub fn fill_zero(&mut self)
        where
            E: ComplexField,
        {
            zipped!(self.rb_mut().as_2d_mut()).for_each(
                #[inline(always)]
                |unzipped!(mut x)| x.write(E::faer_zero()),
            );
        }

        /// Fills the elements of `self` with copies of `constant`.
        #[track_caller]
        pub fn fill(&mut self, constant: E) {
            zipped!((*self).rb_mut().as_2d_mut()).for_each(
                #[inline(always)]
                |unzipped!(mut x)| x.write(constant),
            );
        }

        /// Returns a view over the transpose of `self`.
        #[inline(always)]
        #[must_use]
        pub fn transpose_mut(self) -> RowMut<'a, E> {
            unsafe { self.into_const().transpose().const_cast() }
        }

        /// Returns a view over the conjugate of `self`.
        #[inline(always)]
        #[must_use]
        pub fn conjugate_mut(self) -> ColMut<'a, E::Conj>
        where
            E: Conjugate,
        {
            unsafe { self.into_const().conjugate().const_cast() }
        }

        /// Returns a view over the conjugate transpose of `self`.
        #[inline(always)]
        pub fn adjoint_mut(self) -> RowMut<'a, E::Conj>
        where
            E: Conjugate,
        {
            self.conjugate_mut().transpose_mut()
        }

        /// Returns a view over the canonical representation of `self`, as well as a flag declaring
        /// whether `self` is implicitly conjugated or not.
        #[inline(always)]
        pub fn canonicalize_mut(self) -> (ColMut<'a, E::Canonical>, Conj)
        where
            E: Conjugate,
        {
            let (canon, conj) = self.into_const().canonicalize();
            unsafe { (canon.const_cast(), conj) }
        }

        /// Returns a view over the `self`, with the rows in reversed order.
        #[inline(always)]
        #[must_use]
        pub fn reverse_rows_mut(self) -> Self {
            unsafe { self.into_const().reverse_rows().const_cast() }
        }

        /// Returns a view over the subvector starting at row `row_start`, and with number of rows
        /// `nrows`.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row_start <= self.nrows()`.
        /// * `nrows <= self.nrows() - row_start`.
        #[track_caller]
        #[inline(always)]
        pub unsafe fn subrows_mut_unchecked(self, row_start: usize, nrows: usize) -> Self {
            self.into_const()
                .subrows_unchecked(row_start, nrows)
                .const_cast()
        }

        /// Returns a view over the subvector starting at row `row_start`, and with number of rows
        /// `nrows`.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row_start <= self.nrows()`.
        /// * `nrows <= self.nrows() - row_start`.
        #[track_caller]
        #[inline(always)]
        pub fn subrows_mut(self, row_start: usize, nrows: usize) -> Self {
            unsafe { self.into_const().subrows(row_start, nrows).const_cast() }
        }

        /// Given a matrix with a single column, returns an object that interprets
        /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
        #[track_caller]
        #[inline(always)]
        pub fn column_vector_as_diagonal(self) -> Matrix<DiagMut<'a, E>> {
            Matrix {
                inner: DiagMut { inner: self },
            }
        }

        /// Returns an owning [`Col`] of the data.
        #[inline]
        pub fn to_owned(&self) -> Col<E::Canonical>
        where
            E: Conjugate,
        {
            (*self).rb().to_owned()
        }

        /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
        #[inline]
        pub fn has_nan(&self) -> bool
        where
            E: ComplexField,
        {
            (*self).rb().as_2d().has_nan()
        }

        /// Returns `true` if all of the elements are finite, otherwise returns `false`.
        #[inline]
        pub fn is_all_finite(&self) -> bool
        where
            E: ComplexField,
        {
            (*self).rb().as_2d().is_all_finite()
        }

        /// Returns the maximum norm of `self`.
        #[inline]
        pub fn norm_max(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_max((*self).rb().as_2d())
        }
        /// Returns the L2 norm of `self`.
        #[inline]
        pub fn norm_l2(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_l2((*self).rb().as_2d())
        }

        /// Returns a view over the matrix.
        #[inline]
        pub fn as_ref(&self) -> ColRef<'_, E> {
            (*self).rb()
        }
    }
};

const __ROW_IMPL: () = {
    impl<'a, E: Entity> RowRef<'a, E> {
        #[inline(always)]
        pub fn nrows(&self) -> usize {
            1
        }
        #[inline(always)]
        pub fn ncols(&self) -> usize {
            self.inner.inner.len
        }

        /// Returns pointers to the matrix data.
        #[inline(always)]
        pub fn as_ptr(self) -> GroupFor<E, *const E::Unit> {
            E::faer_map(
                from_copy::<E, _>(self.inner.inner.ptr),
                #[inline(always)]
                |ptr| ptr.as_ptr() as *const E::Unit,
            )
        }

        /// Returns the column stride of the matrix, specified in number of elements, not in bytes.
        #[inline(always)]
        pub fn col_stride(&self) -> isize {
            self.inner.inner.stride
        }

        /// Returns `self` as a matrix view.
        #[inline(always)]
        pub fn as_2d(self) -> MatRef<'a, E> {
            let ncols = self.ncols();
            let col_stride = self.col_stride();
            unsafe { mat::from_raw_parts(self.as_ptr(), 1, ncols, 0, col_stride) }
        }

        /// Returns raw pointers to the element at the given index.
        #[inline(always)]
        pub fn ptr_at(self, col: usize) -> GroupFor<E, *const E::Unit> {
            let offset = (col as isize).wrapping_mul(self.inner.inner.stride);

            E::faer_map(
                self.as_ptr(),
                #[inline(always)]
                |ptr| ptr.wrapping_offset(offset),
            )
        }

        #[inline(always)]
        unsafe fn unchecked_ptr_at(self, col: usize) -> GroupFor<E, *const E::Unit> {
            let offset = unchecked_mul(col, self.inner.inner.stride);
            E::faer_map(
                self.as_ptr(),
                #[inline(always)]
                |ptr| ptr.offset(offset),
            )
        }

        #[inline(always)]
        unsafe fn overflowing_ptr_at(self, col: usize) -> GroupFor<E, *const E::Unit> {
            unsafe {
                let cond = col != self.ncols();
                let offset = (cond as usize).wrapping_neg() as isize
                    & (col as isize).wrapping_mul(self.inner.inner.stride);
                E::faer_map(
                    self.as_ptr(),
                    #[inline(always)]
                    |ptr| ptr.offset(offset),
                )
            }
        }

        /// Returns raw pointers to the element at the given index, assuming the provided index
        /// is within the size of the vector.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn ptr_inbounds_at(self, col: usize) -> GroupFor<E, *const E::Unit> {
            debug_assert!(col < self.ncols());
            self.unchecked_ptr_at(col)
        }

        /// Splits the column vector at the given index into two parts and
        /// returns an array of each subvector, in the following order:
        /// * left.
        /// * right.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at_unchecked(self, col: usize) -> (Self, Self) {
            debug_assert!(col <= self.ncols());

            let col_stride = self.col_stride();

            let ncols = self.ncols();

            unsafe {
                let top = self.as_ptr();
                let bot = self.overflowing_ptr_at(col);

                (
                    row::from_raw_parts(top, col, col_stride),
                    row::from_raw_parts(bot, ncols - col, col_stride),
                )
            }
        }

        /// Splits the column vector at the given index into two parts and
        /// returns an array of each subvector, in the following order:
        /// * top.
        /// * bottom.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at(self, col: usize) -> (Self, Self) {
            assert!(col <= self.ncols());
            unsafe { self.split_at_unchecked(col) }
        }

        /// Returns references to the element at the given index, or subvector if `row` is a
        /// range.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col` must be contained in `[0, self.ncols())`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn get_unchecked<ColRange>(
            self,
            col: ColRange,
        ) -> <Self as RowIndex<ColRange>>::Target
        where
            Self: RowIndex<ColRange>,
        {
            <Self as RowIndex<ColRange>>::get_unchecked(self, col)
        }

        /// Returns references to the element at the given index, or subvector if `col` is a
        /// range, with bound checks.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col` must be contained in `[0, self.ncols())`.
        #[inline(always)]
        #[track_caller]
        pub fn get<ColRange>(self, col: ColRange) -> <Self as RowIndex<ColRange>>::Target
        where
            Self: RowIndex<ColRange>,
        {
            <Self as RowIndex<ColRange>>::get(self, col)
        }

        /// Reads the value of the element at the given index.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn read_unchecked(&self, col: usize) -> E {
            E::faer_from_units(E::faer_map(
                self.get_unchecked(col),
                #[inline(always)]
                |ptr| *ptr,
            ))
        }

        /// Reads the value of the element at the given index, with bound checks.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub fn read(&self, col: usize) -> E {
            E::faer_from_units(E::faer_map(
                self.get(col),
                #[inline(always)]
                |ptr| *ptr,
            ))
        }

        /// Returns a view over the transpose of `self`.
        #[inline(always)]
        #[must_use]
        pub fn transpose(self) -> ColRef<'a, E> {
            unsafe { col::from_raw_parts(self.as_ptr(), self.ncols(), self.col_stride()) }
        }

        /// Returns a view over the conjugate of `self`.
        #[inline(always)]
        #[must_use]
        pub fn conjugate(self) -> RowRef<'a, E::Conj>
        where
            E: Conjugate,
        {
            unsafe {
                // SAFETY: Conjugate requires that E::Unit and E::Conj::Unit have the same layout
                // and that GroupCopyFor<E,X> == E::Conj::GroupCopy<X>
                row::from_raw_parts::<'_, E::Conj>(
                    transmute_unchecked::<
                        GroupFor<E, *const UnitFor<E>>,
                        GroupFor<E::Conj, *const UnitFor<E::Conj>>,
                    >(self.as_ptr()),
                    self.ncols(),
                    self.col_stride(),
                )
            }
        }

        /// Returns a view over the conjugate transpose of `self`.
        #[inline(always)]
        pub fn adjoint(self) -> ColRef<'a, E::Conj>
        where
            E: Conjugate,
        {
            self.conjugate().transpose()
        }

        /// Returns a view over the canonical representation of `self`, as well as a flag declaring
        /// whether `self` is implicitly conjugated or not.
        #[inline(always)]
        pub fn canonicalize(self) -> (RowRef<'a, E::Canonical>, Conj)
        where
            E: Conjugate,
        {
            (
                unsafe {
                    // SAFETY: see Self::conjugate
                    row::from_raw_parts::<'_, E::Canonical>(
                        transmute_unchecked::<
                            GroupFor<E, *const E::Unit>,
                            GroupFor<E::Canonical, *const UnitFor<E::Canonical>>,
                        >(self.as_ptr()),
                        self.ncols(),
                        self.col_stride(),
                    )
                },
                if coe::is_same::<E, E::Canonical>() {
                    Conj::No
                } else {
                    Conj::Yes
                },
            )
        }

        /// Returns a view over the `self`, with the columnss in reversed order.
        #[inline(always)]
        #[must_use]
        pub fn reverse_cols(self) -> Self {
            let ncols = self.ncols();
            let col_stride = self.col_stride().wrapping_neg();

            let ptr = unsafe { self.unchecked_ptr_at(ncols.saturating_sub(1)) };
            unsafe { row::from_raw_parts(ptr, ncols, col_stride) }
        }

        /// Returns a view over the subvector starting at column `col_start`, and with number of
        /// columns `ncols`.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col_start <= self.ncols()`.
        /// * `ncols <= self.ncols() - col_start`.
        #[track_caller]
        #[inline(always)]
        pub unsafe fn subcols_unchecked(self, col_start: usize, ncols: usize) -> Self {
            debug_assert!(col_start <= self.ncols());
            debug_assert!(ncols <= self.ncols() - col_start);
            let col_stride = self.col_stride();
            unsafe { row::from_raw_parts(self.overflowing_ptr_at(col_start), ncols, col_stride) }
        }

        /// Returns a view over the subvector starting at col `col_start`, and with number of cols
        /// `ncols`.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col_start <= self.ncols()`.
        /// * `ncols <= self.ncols() - col_start`.
        #[track_caller]
        #[inline(always)]
        pub fn subcols(self, col_start: usize, ncols: usize) -> Self {
            assert!(col_start <= self.ncols());
            assert!(ncols <= self.ncols() - col_start);
            unsafe { self.subcols_unchecked(col_start, ncols) }
        }

        /// Returns an owning [`Row`] of the data.
        #[inline]
        pub fn to_owned(&self) -> Row<E::Canonical>
        where
            E: Conjugate,
        {
            let mut mat = Row::new();
            mat.resize_with(
                self.ncols(),
                #[inline(always)]
                |col| unsafe { self.read_unchecked(col).canonicalize() },
            );
            mat
        }

        /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
        #[inline]
        pub fn has_nan(&self) -> bool
        where
            E: ComplexField,
        {
            (*self).rb().as_2d().has_nan()
        }

        /// Returns `true` if all of the elements are finite, otherwise returns `false`.
        #[inline]
        pub fn is_all_finite(&self) -> bool
        where
            E: ComplexField,
        {
            (*self).rb().as_2d().is_all_finite()
        }

        /// Returns the maximum norm of `self`.
        #[inline]
        pub fn norm_max(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_max((*self).rb().as_2d())
        }
        /// Returns the L2 norm of `self`.
        #[inline]
        pub fn norm_l2(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_l2((*self).rb().as_2d())
        }

        /// Returns a view over the matrix.
        #[inline]
        pub fn as_ref(&self) -> RowRef<'_, E> {
            *self
        }

        #[doc(hidden)]
        #[inline(always)]
        pub unsafe fn const_cast(self) -> RowMut<'a, E> {
            RowMut {
                inner: DenseRowMut {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }
    }

    impl<E: SimpleEntity> core::ops::Index<usize> for RowRef<'_, E> {
        type Output = E;

        #[inline]
        #[track_caller]
        fn index(&self, col: usize) -> &E {
            self.get(col)
        }
    }

    impl<E: SimpleEntity> core::ops::Index<usize> for RowMut<'_, E> {
        type Output = E;

        #[inline]
        #[track_caller]
        fn index(&self, col: usize) -> &E {
            (*self).rb().get(col)
        }
    }

    impl<E: SimpleEntity> core::ops::IndexMut<usize> for RowMut<'_, E> {
        #[inline]
        #[track_caller]
        fn index_mut(&mut self, col: usize) -> &mut E {
            (*self).rb_mut().get_mut(col)
        }
    }

    impl<E: SimpleEntity> core::ops::Index<usize> for Row<E> {
        type Output = E;

        #[inline]
        #[track_caller]
        fn index(&self, col: usize) -> &E {
            self.as_ref().get(col)
        }
    }

    impl<E: SimpleEntity> core::ops::IndexMut<usize> for Row<E> {
        #[inline]
        #[track_caller]
        fn index_mut(&mut self, col: usize) -> &mut E {
            self.as_mut().get_mut(col)
        }
    }

    impl<'a, E: Entity> RowMut<'a, E> {
        #[inline(always)]
        pub fn nrows(&self) -> usize {
            1
        }
        #[inline(always)]
        pub fn ncols(&self) -> usize {
            self.inner.inner.len
        }

        /// Returns pointers to the matrix data.
        #[inline(always)]
        pub fn as_ptr_mut(self) -> GroupFor<E, *mut E::Unit> {
            E::faer_map(
                from_copy::<E, _>(self.inner.inner.ptr),
                #[inline(always)]
                |ptr| ptr.as_ptr() as *mut E::Unit,
            )
        }

        /// Returns the column stride of the matrix, specified in number of elements, not in bytes.
        #[inline(always)]
        pub fn col_stride(&self) -> isize {
            self.inner.inner.stride
        }

        /// Returns `self` as a mutable matrix view.
        #[inline(always)]
        pub fn as_2d_mut(self) -> MatMut<'a, E> {
            let ncols = self.ncols();
            let col_stride = self.col_stride();
            unsafe { mat::from_raw_parts_mut(self.as_ptr_mut(), 1, ncols, 0, col_stride) }
        }

        /// Returns raw pointers to the element at the given index.
        #[inline(always)]
        pub fn ptr_at_mut(self, col: usize) -> GroupFor<E, *mut E::Unit> {
            let offset = (col as isize).wrapping_mul(self.inner.inner.stride);

            E::faer_map(
                self.as_ptr_mut(),
                #[inline(always)]
                |ptr| ptr.wrapping_offset(offset),
            )
        }

        #[inline(always)]
        unsafe fn ptr_at_mut_unchecked(self, col: usize) -> GroupFor<E, *mut E::Unit> {
            let offset = unchecked_mul(col, self.inner.inner.stride);
            E::faer_map(
                self.as_ptr_mut(),
                #[inline(always)]
                |ptr| ptr.offset(offset),
            )
        }

        /// Returns raw pointers to the element at the given index, assuming the provided index
        /// is within the size of the vector.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn ptr_inbounds_at_mut(self, col: usize) -> GroupFor<E, *mut E::Unit> {
            debug_assert!(col < self.ncols());
            self.ptr_at_mut_unchecked(col)
        }

        /// Splits the column vector at the given index into two parts and
        /// returns an array of each subvector, in the following order:
        /// * left.
        /// * right.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at_mut_unchecked(self, col: usize) -> (Self, Self) {
            let (left, right) = self.into_const().split_at_unchecked(col);
            unsafe { (left.const_cast(), right.const_cast()) }
        }

        /// Splits the column vector at the given index into two parts and
        /// returns an array of each subvector, in the following order:
        /// * top.
        /// * bottom.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub fn split_at_mut(self, col: usize) -> (Self, Self) {
            assert!(col <= self.ncols());
            unsafe { self.split_at_mut_unchecked(col) }
        }

        /// Returns references to the element at the given index, or subvector if `col` is a
        /// range.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col` must be contained in `[0, self.ncols())`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn get_mut_unchecked<ColRange>(
            self,
            col: ColRange,
        ) -> <Self as RowIndex<ColRange>>::Target
        where
            Self: RowIndex<ColRange>,
        {
            <Self as RowIndex<ColRange>>::get_unchecked(self, col)
        }

        /// Returns references to the element at the given index, or subvector if `col` is a
        /// range, with bound checks.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col` must be contained in `[0, self.ncols())`.
        #[inline(always)]
        #[track_caller]
        pub fn get_mut<ColRange>(self, col: ColRange) -> <Self as RowIndex<ColRange>>::Target
        where
            Self: RowIndex<ColRange>,
        {
            <Self as RowIndex<ColRange>>::get(self, col)
        }

        /// Reads the value of the element at the given index.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn read_unchecked(&self, col: usize) -> E {
            self.rb().read_unchecked(col)
        }

        /// Reads the value of the element at the given index, with bound checks.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub fn read(&self, col: usize) -> E {
            self.rb().read(col)
        }

        /// Writes the value to the element at the given index.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn write_unchecked(&mut self, col: usize, value: E) {
            let units = value.faer_into_units();
            let zipped = E::faer_zip(units, (*self).rb_mut().ptr_inbounds_at_mut(col));
            E::faer_map(
                zipped,
                #[inline(always)]
                |(unit, ptr)| *ptr = unit,
            );
        }

        /// Writes the value to the element at the given index, with bound checks.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub fn write(&mut self, col: usize, value: E) {
            assert!(col < self.ncols());
            unsafe { self.write_unchecked(col, value) };
        }

        /// Copies the values from `other` into `self`.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `self.ncols() == other.ncols()`.
        #[track_caller]
        pub fn copy_from(&mut self, other: impl AsRowRef<E>) {
            #[track_caller]
            #[inline(always)]
            fn implementation<E: Entity>(this: RowMut<'_, E>, other: RowRef<'_, E>) {
                zipped!(this.as_2d_mut(), other.as_2d())
                    .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
            }
            implementation(self.rb_mut(), other.as_row_ref())
        }

        /// Fills the elements of `self` with zeros.
        #[track_caller]
        pub fn fill_zero(&mut self)
        where
            E: ComplexField,
        {
            zipped!(self.rb_mut().as_2d_mut()).for_each(
                #[inline(always)]
                |unzipped!(mut x)| x.write(E::faer_zero()),
            );
        }

        /// Fills the elements of `self` with copies of `constant`.
        #[track_caller]
        pub fn fill(&mut self, constant: E) {
            zipped!((*self).rb_mut().as_2d_mut()).for_each(
                #[inline(always)]
                |unzipped!(mut x)| x.write(constant),
            );
        }

        /// Returns a view over the transpose of `self`.
        #[inline(always)]
        #[must_use]
        pub fn transpose_mut(self) -> ColMut<'a, E> {
            unsafe { self.into_const().transpose().const_cast() }
        }

        /// Returns a view over the conjugate of `self`.
        #[inline(always)]
        #[must_use]
        pub fn conjugate_mut(self) -> RowMut<'a, E::Conj>
        where
            E: Conjugate,
        {
            unsafe { self.into_const().conjugate().const_cast() }
        }

        /// Returns a view over the conjugate transpose of `self`.
        #[inline(always)]
        pub fn adjoint_mut(self) -> ColMut<'a, E::Conj>
        where
            E: Conjugate,
        {
            self.conjugate_mut().transpose_mut()
        }

        /// Returns a view over the canonical representation of `self`, as well as a flag declaring
        /// whether `self` is implicitly conjugated or not.
        #[inline(always)]
        pub fn canonicalize_mut(self) -> (RowMut<'a, E::Canonical>, Conj)
        where
            E: Conjugate,
        {
            let (canon, conj) = self.into_const().canonicalize();
            unsafe { (canon.const_cast(), conj) }
        }

        /// Returns a view over the `self`, with the columnss in reversed order.
        #[inline(always)]
        #[must_use]
        pub fn reverse_cols_mut(self) -> Self {
            unsafe { self.into_const().reverse_cols().const_cast() }
        }

        /// Returns a view over the subvector starting at col `col_start`, and with number of
        /// columns `ncols`.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col_start <= self.ncols()`.
        /// * `ncols <= self.ncols() - col_start`.
        #[track_caller]
        #[inline(always)]
        pub unsafe fn subcols_mut_unchecked(self, col_start: usize, ncols: usize) -> Self {
            self.into_const()
                .subcols_unchecked(col_start, ncols)
                .const_cast()
        }

        /// Returns a view over the subvector starting at col `col_start`, and with number of
        /// columns `ncols`.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col_start <= self.ncols()`.
        /// * `ncols <= self.ncols() - col_start`.
        #[track_caller]
        #[inline(always)]
        pub fn subcols_mut(self, col_start: usize, ncols: usize) -> Self {
            unsafe { self.into_const().subcols(col_start, ncols).const_cast() }
        }

        /// Returns an owning [`Row`] of the data.
        #[inline]
        pub fn to_owned(&self) -> Row<E::Canonical>
        where
            E: Conjugate,
        {
            (*self).rb().to_owned()
        }

        /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
        #[inline]
        pub fn has_nan(&self) -> bool
        where
            E: ComplexField,
        {
            (*self).rb().as_2d().has_nan()
        }

        /// Returns `true` if all of the elements are finite, otherwise returns `false`.
        #[inline]
        pub fn is_all_finite(&self) -> bool
        where
            E: ComplexField,
        {
            (*self).rb().as_2d().is_all_finite()
        }

        /// Returns the maximum norm of `self`.
        #[inline]
        pub fn norm_max(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_max((*self).rb().as_2d())
        }
        /// Returns the L2 norm of `self`.
        #[inline]
        pub fn norm_l2(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_l2((*self).rb().as_2d())
        }

        /// Returns a view over the matrix.
        #[inline]
        pub fn as_ref(&self) -> RowRef<'_, E> {
            (*self).rb()
        }
    }
};

const __MAT_IMPL: () = {
    impl<'a, E: Entity> MatRef<'a, E> {
        #[track_caller]
        #[inline(always)]
        #[doc(hidden)]
        pub fn try_get_contiguous_col(self, j: usize) -> GroupFor<E, &'a [E::Unit]> {
            assert!(self.row_stride() == 1);
            let col = self.col(j);
            if col.nrows() == 0 {
                E::faer_map(
                    E::UNIT,
                    #[inline(always)]
                    |()| &[] as &[E::Unit],
                )
            } else {
                let m = col.nrows();
                E::faer_map(
                    col.as_ptr(),
                    #[inline(always)]
                    |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
                )
            }
        }

        #[track_caller]
        #[inline(always)]
        #[deprecated = "replaced by `faer_core::mat::from_column_major_slice`"]
        pub fn from_column_major_slice(
            slice: GroupFor<E, &'a [E::Unit]>,
            nrows: usize,
            ncols: usize,
        ) -> Self {
            mat::from_column_major_slice(slice, nrows, ncols)
        }

        #[inline(always)]
        #[track_caller]
        #[deprecated = "replaced by `faer_core::mat::from_row_major_slice`"]
        pub fn from_row_major_slice(
            slice: GroupFor<E, &'a [E::Unit]>,
            nrows: usize,
            ncols: usize,
        ) -> Self {
            mat::from_row_major_slice(slice, ncols, nrows)
        }

        #[inline(always)]
        #[track_caller]
        #[deprecated = "replaced by `faer_core::mat::from_raw_parts`"]
        pub unsafe fn from_raw_parts(
            ptr: GroupFor<E, *const E::Unit>,
            nrows: usize,
            ncols: usize,
            row_stride: isize,
            col_stride: isize,
        ) -> Self {
            let mut ptr_is_null = false;
            E::faer_map(
                E::faer_as_ref(&ptr),
                #[inline(always)]
                |ptr| ptr_is_null |= ptr.is_null(),
            );

            assert!(!ptr_is_null);
            mat::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride)
        }

        #[inline(always)]
        pub fn nrows(&self) -> usize {
            self.inner.inner.nrows
        }
        #[inline(always)]
        pub fn ncols(&self) -> usize {
            self.inner.inner.ncols
        }

        /// Returns pointers to the matrix data.
        #[inline(always)]
        pub fn as_ptr(self) -> GroupFor<E, *const E::Unit> {
            E::faer_map(
                from_copy::<E, _>(self.inner.inner.ptr),
                #[inline(always)]
                |ptr| ptr.as_ptr() as *const E::Unit,
            )
        }

        /// Returns the row stride of the matrix, specified in number of elements, not in bytes.
        #[inline(always)]
        pub fn row_stride(&self) -> isize {
            self.inner.inner.row_stride
        }

        /// Returns the column stride of the matrix, specified in number of elements, not in bytes.
        #[inline(always)]
        pub fn col_stride(&self) -> isize {
            self.inner.inner.col_stride
        }

        /// Returns raw pointers to the element at the given indices.
        #[inline(always)]
        pub fn ptr_at(self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
            let offset = ((row as isize).wrapping_mul(self.inner.inner.row_stride))
                .wrapping_add((col as isize).wrapping_mul(self.inner.inner.col_stride));

            E::faer_map(
                self.as_ptr(),
                #[inline(always)]
                |ptr| ptr.wrapping_offset(offset),
            )
        }

        #[inline(always)]
        unsafe fn unchecked_ptr_at(self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
            let offset = unchecked_add(
                unchecked_mul(row, self.inner.inner.row_stride),
                unchecked_mul(col, self.inner.inner.col_stride),
            );
            E::faer_map(
                self.as_ptr(),
                #[inline(always)]
                |ptr| ptr.offset(offset),
            )
        }

        #[inline(always)]
        unsafe fn overflowing_ptr_at(self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
            unsafe {
                let cond = (row != self.nrows()) & (col != self.ncols());
                let offset = (cond as usize).wrapping_neg() as isize
                    & (isize::wrapping_add(
                        (row as isize).wrapping_mul(self.inner.inner.row_stride),
                        (col as isize).wrapping_mul(self.inner.inner.col_stride),
                    ));
                E::faer_map(
                    self.as_ptr(),
                    #[inline(always)]
                    |ptr| ptr.offset(offset),
                )
            }
        }

        /// Returns raw pointers to the element at the given indices, assuming the provided indices
        /// are within the matrix dimensions.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn ptr_inbounds_at(self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
            debug_assert!(all(row < self.nrows(), col < self.ncols()));
            self.unchecked_ptr_at(row, col)
        }

        /// Splits the matrix horizontally and vertically at the given indices into four corners and
        /// returns an array of each submatrix, in the following order:
        /// * top left.
        /// * top right.
        /// * bottom left.
        /// * bottom right.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row <= self.nrows()`.
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at_unchecked(self, row: usize, col: usize) -> (Self, Self, Self, Self) {
            debug_assert!(all(row <= self.nrows(), col <= self.ncols()));

            let row_stride = self.row_stride();
            let col_stride = self.col_stride();

            let nrows = self.nrows();
            let ncols = self.ncols();

            unsafe {
                let top_left = self.overflowing_ptr_at(0, 0);
                let top_right = self.overflowing_ptr_at(0, col);
                let bot_left = self.overflowing_ptr_at(row, 0);
                let bot_right = self.overflowing_ptr_at(row, col);

                (
                    mat::from_raw_parts(top_left, row, col, row_stride, col_stride),
                    mat::from_raw_parts(top_right, row, ncols - col, row_stride, col_stride),
                    mat::from_raw_parts(bot_left, nrows - row, col, row_stride, col_stride),
                    mat::from_raw_parts(
                        bot_right,
                        nrows - row,
                        ncols - col,
                        row_stride,
                        col_stride,
                    ),
                )
            }
        }

        /// Splits the matrix horizontally and vertically at the given indices into four corners and
        /// returns an array of each submatrix, in the following order:
        /// * top left.
        /// * top right.
        /// * bottom left.
        /// * bottom right.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row <= self.nrows()`.
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub fn split_at(self, row: usize, col: usize) -> (Self, Self, Self, Self) {
            assert!(all(row <= self.nrows(), col <= self.ncols()));
            unsafe { self.split_at_unchecked(row, col) }
        }

        /// Splits the matrix horizontally at the given row into two parts and returns an array of
        /// each submatrix, in the following order:
        /// * top.
        /// * bottom.
        ///
        /// # Safety
        /// The behavior is undefined if the following condition is violated:
        /// * `row <= self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at_row_unchecked(self, row: usize) -> (Self, Self) {
            debug_assert!(row <= self.nrows());

            let row_stride = self.row_stride();
            let col_stride = self.col_stride();

            let nrows = self.nrows();
            let ncols = self.ncols();

            unsafe {
                let top_right = self.overflowing_ptr_at(0, 0);
                let bot_right = self.overflowing_ptr_at(row, 0);

                (
                    mat::from_raw_parts(top_right, row, ncols, row_stride, col_stride),
                    mat::from_raw_parts(bot_right, nrows - row, ncols, row_stride, col_stride),
                )
            }
        }

        /// Splits the matrix horizontally at the given row into two parts and returns an array of
        /// each submatrix, in the following order:
        /// * top.
        /// * bottom.
        ///
        /// # Panics
        /// The function panics if the following condition is violated:
        /// * `row <= self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub fn split_at_row(self, row: usize) -> (Self, Self) {
            assert!(row <= self.nrows());
            unsafe { self.split_at_row_unchecked(row) }
        }

        /// Splits the matrix vertically at the given row into two parts and returns an array of
        /// each submatrix, in the following order:
        /// * left.
        /// * right.
        ///
        /// # Safety
        /// The behavior is undefined if the following condition is violated:
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at_col_unchecked(self, col: usize) -> (Self, Self) {
            debug_assert!(col <= self.ncols());

            let row_stride = self.row_stride();
            let col_stride = self.col_stride();

            let nrows = self.nrows();
            let ncols = self.ncols();

            unsafe {
                let bot_left = self.overflowing_ptr_at(0, 0);
                let bot_right = self.overflowing_ptr_at(0, col);

                (
                    mat::from_raw_parts(bot_left, nrows, col, row_stride, col_stride),
                    mat::from_raw_parts(bot_right, nrows, ncols - col, row_stride, col_stride),
                )
            }
        }

        /// Splits the matrix vertically at the given row into two parts and returns an array of
        /// each submatrix, in the following order:
        /// * left.
        /// * right.
        ///
        /// # Panics
        /// The function panics if the following condition is violated:
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub fn split_at_col(self, col: usize) -> (Self, Self) {
            assert!(col <= self.ncols());
            unsafe { self.split_at_col_unchecked(col) }
        }

        /// Returns references to the element at the given indices, or submatrices if either `row`
        /// or `col` is a range.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row` must be contained in `[0, self.nrows())`.
        /// * `col` must be contained in `[0, self.ncols())`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn get_unchecked<RowRange, ColRange>(
            self,
            row: RowRange,
            col: ColRange,
        ) -> <Self as MatIndex<RowRange, ColRange>>::Target
        where
            Self: MatIndex<RowRange, ColRange>,
        {
            <Self as MatIndex<RowRange, ColRange>>::get_unchecked(self, row, col)
        }

        /// Returns references to the element at the given indices, or submatrices if either `row`
        /// or `col` is a range, with bound checks.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row` must be contained in `[0, self.nrows())`.
        /// * `col` must be contained in `[0, self.ncols())`.
        #[inline(always)]
        #[track_caller]
        pub fn get<RowRange, ColRange>(
            self,
            row: RowRange,
            col: ColRange,
        ) -> <Self as MatIndex<RowRange, ColRange>>::Target
        where
            Self: MatIndex<RowRange, ColRange>,
        {
            <Self as MatIndex<RowRange, ColRange>>::get(self, row, col)
        }

        /// Reads the value of the element at the given indices.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn read_unchecked(&self, row: usize, col: usize) -> E {
            E::faer_from_units(E::faer_map(
                self.get_unchecked(row, col),
                #[inline(always)]
                |ptr| *ptr,
            ))
        }

        /// Reads the value of the element at the given indices, with bound checks.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub fn read(&self, row: usize, col: usize) -> E {
            E::faer_from_units(E::faer_map(
                self.get(row, col),
                #[inline(always)]
                |ptr| *ptr,
            ))
        }

        /// Returns a view over the transpose of `self`.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        /// let view = matrix.as_ref();
        /// let transpose = view.transpose();
        ///
        /// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        /// assert_eq!(expected.as_ref(), transpose);
        /// ```
        #[inline(always)]
        #[must_use]
        pub fn transpose(self) -> Self {
            unsafe {
                mat::from_raw_parts(
                    self.as_ptr(),
                    self.ncols(),
                    self.nrows(),
                    self.col_stride(),
                    self.row_stride(),
                )
            }
        }

        /// Returns a view over the conjugate of `self`.
        #[inline(always)]
        #[must_use]
        pub fn conjugate(self) -> MatRef<'a, E::Conj>
        where
            E: Conjugate,
        {
            unsafe {
                // SAFETY: Conjugate requires that E::Unit and E::Conj::Unit have the same layout
                // and that GroupCopyFor<E,X> == E::Conj::GroupCopy<X>
                mat::from_raw_parts::<'_, E::Conj>(
                    transmute_unchecked::<
                        GroupFor<E, *const UnitFor<E>>,
                        GroupFor<E::Conj, *const UnitFor<E::Conj>>,
                    >(self.as_ptr()),
                    self.nrows(),
                    self.ncols(),
                    self.row_stride(),
                    self.col_stride(),
                )
            }
        }

        /// Returns a view over the conjugate transpose of `self`.
        #[inline(always)]
        pub fn adjoint(self) -> MatRef<'a, E::Conj>
        where
            E: Conjugate,
        {
            self.transpose().conjugate()
        }

        /// Returns a view over the canonical representation of `self`, as well as a flag declaring
        /// whether `self` is implicitly conjugated or not.
        #[inline(always)]
        pub fn canonicalize(self) -> (MatRef<'a, E::Canonical>, Conj)
        where
            E: Conjugate,
        {
            (
                unsafe {
                    // SAFETY: see Self::conjugate
                    mat::from_raw_parts::<'_, E::Canonical>(
                        transmute_unchecked::<
                            GroupFor<E, *const E::Unit>,
                            GroupFor<E::Canonical, *const UnitFor<E::Canonical>>,
                        >(self.as_ptr()),
                        self.nrows(),
                        self.ncols(),
                        self.row_stride(),
                        self.col_stride(),
                    )
                },
                if coe::is_same::<E, E::Canonical>() {
                    Conj::No
                } else {
                    Conj::Yes
                },
            )
        }

        /// Returns a view over the `self`, with the rows in reversed order.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        /// let view = matrix.as_ref();
        /// let reversed_rows = view.reverse_rows();
        ///
        /// let expected = mat![[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]];
        /// assert_eq!(expected.as_ref(), reversed_rows);
        /// ```
        #[inline(always)]
        #[must_use]
        pub fn reverse_rows(self) -> Self {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride().wrapping_neg();
            let col_stride = self.col_stride();

            let ptr = unsafe { self.unchecked_ptr_at(nrows.saturating_sub(1), 0) };
            unsafe { mat::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
        }

        /// Returns a view over the `self`, with the columns in reversed order.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        /// let view = matrix.as_ref();
        /// let reversed_cols = view.reverse_cols();
        ///
        /// let expected = mat![[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]];
        /// assert_eq!(expected.as_ref(), reversed_cols);
        /// ```
        #[inline(always)]
        #[must_use]
        pub fn reverse_cols(self) -> Self {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = self.row_stride();
            let col_stride = self.col_stride().wrapping_neg();
            let ptr = unsafe { self.unchecked_ptr_at(0, ncols.saturating_sub(1)) };
            unsafe { mat::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
        }

        /// Returns a view over the `self`, with the rows and the columns in reversed order.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        /// let view = matrix.as_ref();
        /// let reversed = view.reverse_rows_and_cols();
        ///
        /// let expected = mat![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
        /// assert_eq!(expected.as_ref(), reversed);
        /// ```
        #[inline(always)]
        #[must_use]
        pub fn reverse_rows_and_cols(self) -> Self {
            let nrows = self.nrows();
            let ncols = self.ncols();
            let row_stride = -self.row_stride();
            let col_stride = -self.col_stride();

            let ptr =
                unsafe { self.unchecked_ptr_at(nrows.saturating_sub(1), ncols.saturating_sub(1)) };
            unsafe { mat::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
        }

        /// Returns a view over the submatrix starting at indices `(row_start, col_start)`, and with
        /// dimensions `(nrows, ncols)`.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row_start <= self.nrows()`.
        /// * `col_start <= self.ncols()`.
        /// * `nrows <= self.nrows() - row_start`.
        /// * `ncols <= self.ncols() - col_start`.
        #[track_caller]
        #[inline(always)]
        pub unsafe fn submatrix_unchecked(
            self,
            row_start: usize,
            col_start: usize,
            nrows: usize,
            ncols: usize,
        ) -> Self {
            debug_assert!(all(row_start <= self.nrows(), col_start <= self.ncols()));
            debug_assert!(all(
                nrows <= self.nrows() - row_start,
                ncols <= self.ncols() - col_start,
            ));
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();

            unsafe {
                mat::from_raw_parts(
                    self.overflowing_ptr_at(row_start, col_start),
                    nrows,
                    ncols,
                    row_stride,
                    col_stride,
                )
            }
        }

        /// Returns a view over the submatrix starting at indices `(row_start, col_start)`, and with
        /// dimensions `(nrows, ncols)`.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row_start <= self.nrows()`.
        /// * `col_start <= self.ncols()`.
        /// * `nrows <= self.nrows() - row_start`.
        /// * `ncols <= self.ncols() - col_start`.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let matrix = mat![
        ///     [1.0, 5.0, 9.0],
        ///     [2.0, 6.0, 10.0],
        ///     [3.0, 7.0, 11.0],
        ///     [4.0, 8.0, 12.0f64],
        /// ];
        ///
        /// let view = matrix.as_ref();
        /// let submatrix = view.submatrix(2, 1, 2, 2);
        ///
        /// let expected = mat![[7.0, 11.0], [8.0, 12.0f64]];
        /// assert_eq!(expected.as_ref(), submatrix);
        /// ```
        #[track_caller]
        #[inline(always)]
        pub fn submatrix(
            self,
            row_start: usize,
            col_start: usize,
            nrows: usize,
            ncols: usize,
        ) -> Self {
            assert!(all(row_start <= self.nrows(), col_start <= self.ncols()));
            assert!(all(
                nrows <= self.nrows() - row_start,
                ncols <= self.ncols() - col_start,
            ));
            unsafe { self.submatrix_unchecked(row_start, col_start, nrows, ncols) }
        }

        /// Returns a view over the submatrix starting at row `row_start`, and with number of rows
        /// `nrows`.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row_start <= self.nrows()`.
        /// * `nrows <= self.nrows() - row_start`.
        #[track_caller]
        #[inline(always)]
        pub unsafe fn subrows_unchecked(self, row_start: usize, nrows: usize) -> Self {
            debug_assert!(row_start <= self.nrows());
            debug_assert!(nrows <= self.nrows() - row_start);
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            unsafe {
                mat::from_raw_parts(
                    self.overflowing_ptr_at(row_start, 0),
                    nrows,
                    self.ncols(),
                    row_stride,
                    col_stride,
                )
            }
        }

        /// Returns a view over the submatrix starting at row `row_start`, and with number of rows
        /// `nrows`.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row_start <= self.nrows()`.
        /// * `nrows <= self.nrows() - row_start`.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let matrix = mat![
        ///     [1.0, 5.0, 9.0],
        ///     [2.0, 6.0, 10.0],
        ///     [3.0, 7.0, 11.0],
        ///     [4.0, 8.0, 12.0f64],
        /// ];
        ///
        /// let view = matrix.as_ref();
        /// let subrows = view.subrows(1, 2);
        ///
        /// let expected = mat![[2.0, 6.0, 10.0], [3.0, 7.0, 11.0],];
        /// assert_eq!(expected.as_ref(), subrows);
        /// ```
        #[track_caller]
        #[inline(always)]
        pub fn subrows(self, row_start: usize, nrows: usize) -> Self {
            assert!(row_start <= self.nrows());
            assert!(nrows <= self.nrows() - row_start);
            unsafe { self.subrows_unchecked(row_start, nrows) }
        }

        /// Returns a view over the submatrix starting at column `col_start`, and with number of
        /// columns `ncols`.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col_start <= self.ncols()`.
        /// * `ncols <= self.ncols() - col_start`.
        #[track_caller]
        #[inline(always)]
        pub unsafe fn subcols_unchecked(self, col_start: usize, ncols: usize) -> Self {
            debug_assert!(col_start <= self.ncols());
            debug_assert!(ncols <= self.ncols() - col_start);
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            unsafe {
                mat::from_raw_parts(
                    self.overflowing_ptr_at(0, col_start),
                    self.nrows(),
                    ncols,
                    row_stride,
                    col_stride,
                )
            }
        }

        /// Returns a view over the submatrix starting at column `col_start`, and with number of
        /// columns `ncols`.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col_start <= self.ncols()`.
        /// * `ncols <= self.ncols() - col_start`.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let matrix = mat![
        ///     [1.0, 5.0, 9.0],
        ///     [2.0, 6.0, 10.0],
        ///     [3.0, 7.0, 11.0],
        ///     [4.0, 8.0, 12.0f64],
        /// ];
        ///
        /// let view = matrix.as_ref();
        /// let subcols = view.subcols(2, 1);
        ///
        /// let expected = mat![[9.0], [10.0], [11.0], [12.0f64]];
        /// assert_eq!(expected.as_ref(), subcols);
        /// ```
        #[track_caller]
        #[inline(always)]
        pub fn subcols(self, col_start: usize, ncols: usize) -> Self {
            debug_assert!(col_start <= self.ncols());
            debug_assert!(ncols <= self.ncols() - col_start);
            unsafe { self.subcols_unchecked(col_start, ncols) }
        }

        /// Returns a view over the row at the given index.
        ///
        /// # Safety
        /// The function panics if any of the following conditions are violated:
        /// * `row_idx < self.nrows()`.
        #[track_caller]
        #[inline(always)]
        pub unsafe fn row_unchecked(self, row_idx: usize) -> RowRef<'a, E> {
            debug_assert!(row_idx < self.nrows());
            unsafe {
                row::from_raw_parts(
                    self.overflowing_ptr_at(row_idx, 0),
                    self.ncols(),
                    self.col_stride(),
                )
            }
        }

        /// Returns a view over the row at the given index.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row_idx < self.nrows()`.
        #[track_caller]
        #[inline(always)]
        pub fn row(self, row_idx: usize) -> RowRef<'a, E> {
            assert!(row_idx < self.nrows());
            unsafe { self.row_unchecked(row_idx) }
        }

        /// Returns a view over the column at the given index.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `col_idx < self.ncols()`.
        #[track_caller]
        #[inline(always)]
        pub unsafe fn col_unchecked(self, col_idx: usize) -> ColRef<'a, E> {
            debug_assert!(col_idx < self.ncols());
            unsafe {
                col::from_raw_parts(
                    self.overflowing_ptr_at(0, col_idx),
                    self.nrows(),
                    self.row_stride(),
                )
            }
        }

        /// Returns a view over the column at the given index.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col_idx < self.ncols()`.
        #[track_caller]
        #[inline(always)]
        pub fn col(self, col_idx: usize) -> ColRef<'a, E> {
            assert!(col_idx < self.ncols());
            unsafe { self.col_unchecked(col_idx) }
        }

        /// Given a matrix with a single column, returns an object that interprets
        /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
        #[track_caller]
        #[inline(always)]
        pub fn column_vector_as_diagonal(self) -> Matrix<DiagRef<'a, E>> {
            assert!(self.ncols() == 1);
            Matrix {
                inner: DiagRef { inner: self.col(0) },
            }
        }

        #[inline(always)]
        pub fn diagonal(self) -> Matrix<DiagRef<'a, E>> {
            let size = self.nrows().min(self.ncols());
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            unsafe {
                Matrix {
                    inner: DiagRef {
                        inner: col::from_raw_parts(self.as_ptr(), size, row_stride + col_stride),
                    },
                }
            }
        }

        /// Returns an owning [`Mat`] of the data.
        #[inline]
        pub fn to_owned(&self) -> Mat<E::Canonical>
        where
            E: Conjugate,
        {
            let mut mat = Mat::new();
            mat.resize_with(
                self.nrows(),
                self.ncols(),
                #[inline(always)]
                |row, col| unsafe { self.read_unchecked(row, col).canonicalize() },
            );
            mat
        }

        /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
        #[inline]
        pub fn has_nan(&self) -> bool
        where
            E: ComplexField,
        {
            let mut found_nan = false;
            zipped!(*self).for_each(|unzipped!(x)| {
                found_nan |= x.read().faer_is_nan();
            });
            found_nan
        }

        /// Returns `true` if all of the elements are finite, otherwise returns `false`.
        #[inline]
        pub fn is_all_finite(&self) -> bool
        where
            E: ComplexField,
        {
            let mut all_finite = true;
            zipped!(*self).for_each(|unzipped!(x)| {
                all_finite &= x.read().faer_is_finite();
            });
            all_finite
        }

        /// Returns the maximum norm of `self`.
        #[inline]
        pub fn norm_max(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_max((*self).rb())
        }
        /// Returns the L2 norm of `self`.
        #[inline]
        pub fn norm_l2(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_l2((*self).rb())
        }

        /// Returns a view over the matrix.
        #[inline]
        pub fn as_ref(&self) -> MatRef<'_, E> {
            *self
        }

        #[doc(hidden)]
        #[inline(always)]
        pub unsafe fn const_cast(self) -> MatMut<'a, E> {
            MatMut {
                inner: DenseMut {
                    inner: self.inner.inner,
                    __marker: PhantomData,
                },
            }
        }

        /// Returns an iterator that provides successive chunks of the columns of this matrix, with
        /// each having at most `chunk_size` columns.
        ///
        /// If the number of columns is a multiple of `chunk_size`, then all chunks have
        /// `chunk_size` columns.
        #[inline]
        #[track_caller]
        pub fn col_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + DoubleEndedIterator<Item = MatRef<'a, E>> {
            assert!(chunk_size > 0);
            let chunk_count = self.ncols().msrv_div_ceil(chunk_size);
            (0..chunk_count).map(move |chunk_idx| {
                let pos = chunk_size * chunk_idx;
                self.subcols(pos, Ord::min(chunk_size, self.ncols() - pos))
            })
        }

        #[inline]
        #[track_caller]
        #[deprecated = "replaced by `MatRef::col_chunks`"]
        pub fn into_col_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + DoubleEndedIterator<Item = MatRef<'a, E>> {
            self.col_chunks(chunk_size)
        }

        /// Returns an iterator that provides successive chunks of the rows of this matrix, with
        /// each having at most `chunk_size` rows.
        ///
        /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
        /// rows.
        #[inline]
        #[track_caller]
        pub fn row_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + DoubleEndedIterator<Item = MatRef<'a, E>> {
            self.transpose()
                .col_chunks(chunk_size)
                .map(|chunk| chunk.transpose())
        }

        #[inline]
        #[track_caller]
        #[deprecated = "replaced by `MatRef::row_chunks`"]
        pub fn into_row_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + DoubleEndedIterator<Item = MatRef<'a, E>> {
            self.row_chunks(chunk_size)
        }

        /// Returns a parallel iterator that provides successive chunks of the columns of this
        /// matrix, with each having at most `chunk_size` columns.
        ///
        /// If the number of columns is a multiple of `chunk_size`, then all chunks have
        /// `chunk_size` columns.
        ///
        /// Only available with the `rayon` feature.
        #[cfg(feature = "rayon")]
        #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
        #[inline]
        #[track_caller]
        pub fn par_col_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E>> {
            use rayon::prelude::*;

            assert!(chunk_size > 0);
            let chunk_count = self.ncols().msrv_div_ceil(chunk_size);
            (0..chunk_count).into_par_iter().map(move |chunk_idx| {
                let pos = chunk_size * chunk_idx;
                self.subcols(pos, Ord::min(chunk_size, self.ncols() - pos))
            })
        }

        #[cfg(feature = "rayon")]
        #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
        #[inline]
        #[track_caller]
        #[deprecated = "replaced by `MatRef::par_col_chunks`"]
        pub fn into_par_col_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E>> {
            self.par_col_chunks(chunk_size)
        }

        /// Returns a parallel iterator that provides successive chunks of the rows of this matrix,
        /// with each having at most `chunk_size` rows.
        ///
        /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
        /// rows.
        ///
        /// Only available with the `rayon` feature.
        #[cfg(feature = "rayon")]
        #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
        #[inline]
        #[track_caller]
        pub fn par_row_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E>> {
            use rayon::prelude::*;

            self.transpose()
                .par_col_chunks(chunk_size)
                .map(|chunk| chunk.transpose())
        }

        /// Returns a parallel iterator that provides successive chunks of the rows of this matrix,
        /// with each having at most `chunk_size` rows.
        ///
        /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
        /// rows.
        ///
        /// Only available with the `rayon` feature.
        #[cfg(feature = "rayon")]
        #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
        #[inline]
        #[track_caller]
        #[deprecated = "replaced by `MatRef::par_row_chunks`"]
        pub fn into_par_row_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E>> {
            self.par_row_chunks(chunk_size)
        }
    }

    impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for MatRef<'_, E> {
        type Output = E;

        #[inline]
        #[track_caller]
        fn index(&self, (row, col): (usize, usize)) -> &E {
            self.get(row, col)
        }
    }

    impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for MatMut<'_, E> {
        type Output = E;

        #[inline]
        #[track_caller]
        fn index(&self, (row, col): (usize, usize)) -> &E {
            (*self).rb().get(row, col)
        }
    }

    impl<E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for MatMut<'_, E> {
        #[inline]
        #[track_caller]
        fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut E {
            (*self).rb_mut().get_mut(row, col)
        }
    }

    impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for Mat<E> {
        type Output = E;

        #[inline]
        #[track_caller]
        fn index(&self, (row, col): (usize, usize)) -> &E {
            self.as_ref().get(row, col)
        }
    }

    impl<E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for Mat<E> {
        #[inline]
        #[track_caller]
        fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut E {
            self.as_mut().get_mut(row, col)
        }
    }

    impl<'a, E: Entity> MatMut<'a, E> {
        #[track_caller]
        #[inline(always)]
        #[doc(hidden)]
        pub fn try_get_contiguous_col_mut(self, j: usize) -> GroupFor<E, &'a mut [E::Unit]> {
            assert!(self.row_stride() == 1);
            let col = self.col_mut(j);
            if col.nrows() == 0 {
                E::faer_map(
                    E::UNIT,
                    #[inline(always)]
                    |()| &mut [] as &mut [E::Unit],
                )
            } else {
                let m = col.nrows();
                E::faer_map(
                    col.as_ptr_mut(),
                    #[inline(always)]
                    |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
                )
            }
        }

        #[track_caller]
        #[deprecated = "replaced by `faer_core::mat::from_column_major_slice_mut`"]
        pub fn from_column_major_slice(
            slice: GroupFor<E, &'a mut [E::Unit]>,
            nrows: usize,
            ncols: usize,
        ) -> Self {
            mat::from_column_major_slice_mut(slice, nrows, ncols)
        }

        #[inline(always)]
        #[track_caller]
        #[deprecated = "replaced by `faer_core::mat::from_row_major_slice_mut`"]
        pub fn from_row_major_slice(
            slice: GroupFor<E, &'a mut [E::Unit]>,
            nrows: usize,
            ncols: usize,
        ) -> Self {
            mat::from_row_major_slice_mut(slice, ncols, nrows).transpose_mut()
        }

        #[inline(always)]
        #[track_caller]
        #[deprecated = "replaced by `faer_core::mat::from_raw_parts_mut`"]
        pub unsafe fn from_raw_parts(
            ptr: GroupFor<E, *mut E::Unit>,
            nrows: usize,
            ncols: usize,
            row_stride: isize,
            col_stride: isize,
        ) -> Self {
            let mut ptr_is_null = false;
            E::faer_map(E::faer_as_ref(&ptr), |ptr| ptr_is_null |= ptr.is_null());

            assert!(!ptr_is_null);
            mat::from_raw_parts_mut(ptr, nrows, ncols, row_stride, col_stride)
        }

        #[inline(always)]
        pub fn nrows(&self) -> usize {
            self.inner.inner.nrows
        }
        #[inline(always)]
        pub fn ncols(&self) -> usize {
            self.inner.inner.ncols
        }

        /// Returns pointers to the matrix data.
        #[inline(always)]
        pub fn as_ptr_mut(self) -> GroupFor<E, *mut E::Unit> {
            E::faer_map(
                from_copy::<E, _>(self.inner.inner.ptr),
                #[inline(always)]
                |ptr| ptr.as_ptr(),
            )
        }

        #[inline(always)]
        #[deprecated = "replaced by `MatMut::as_ptr_mut`"]
        pub fn as_ptr(self) -> GroupFor<E, *mut E::Unit> {
            self.as_ptr_mut()
        }

        /// Returns the row stride of the matrix, specified in number of elements, not in bytes.
        #[inline(always)]
        pub fn row_stride(&self) -> isize {
            self.inner.inner.row_stride
        }

        /// Returns the column stride of the matrix, specified in number of elements, not in bytes.
        #[inline(always)]
        pub fn col_stride(&self) -> isize {
            self.inner.inner.col_stride
        }

        /// Returns raw pointers to the element at the given indices.
        #[inline(always)]
        pub fn ptr_at_mut(self, row: usize, col: usize) -> GroupFor<E, *mut E::Unit> {
            let offset = ((row as isize).wrapping_mul(self.inner.inner.row_stride))
                .wrapping_add((col as isize).wrapping_mul(self.inner.inner.col_stride));
            E::faer_map(
                self.as_ptr_mut(),
                #[inline(always)]
                |ptr| ptr.wrapping_offset(offset),
            )
        }

        #[inline(always)]
        #[deprecated = "replaced by `MatMut::ptr_at_mut`"]
        pub fn ptr_at(self, row: usize, col: usize) -> GroupFor<E, *mut E::Unit> {
            self.ptr_at_mut(row, col)
        }

        #[inline(always)]
        unsafe fn ptr_at_mut_unchecked(self, row: usize, col: usize) -> GroupFor<E, *mut E::Unit> {
            let offset = unchecked_add(
                unchecked_mul(row, self.inner.inner.row_stride),
                unchecked_mul(col, self.inner.inner.col_stride),
            );
            E::faer_map(
                self.as_ptr_mut(),
                #[inline(always)]
                |ptr| ptr.offset(offset),
            )
        }

        /// Returns raw pointers to the element at the given indices, assuming the provided indices
        /// are within the matrix dimensions.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn ptr_inbounds_at_mut(
            self,
            row: usize,
            col: usize,
        ) -> GroupFor<E, *mut E::Unit> {
            debug_assert!(all(row < self.nrows(), col < self.ncols()));
            self.ptr_at_mut_unchecked(row, col)
        }

        #[inline(always)]
        #[track_caller]
        #[deprecated = "replaced by `MatMut::ptr_inbounds_at_mut`"]
        pub unsafe fn ptr_inbounds_at(self, row: usize, col: usize) -> GroupFor<E, *mut E::Unit> {
            self.ptr_inbounds_at_mut(row, col)
        }

        /// Splits the matrix horizontally and vertically at the given indices into four corners and
        /// returns an array of each submatrix, in the following order:
        /// * top left.
        /// * top right.
        /// * bottom left.
        /// * bottom right.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row <= self.nrows()`.
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at_mut_unchecked(
            self,
            row: usize,
            col: usize,
        ) -> (Self, Self, Self, Self) {
            let (top_left, top_right, bot_left, bot_right) =
                self.into_const().split_at_unchecked(row, col);
            (
                top_left.const_cast(),
                top_right.const_cast(),
                bot_left.const_cast(),
                bot_right.const_cast(),
            )
        }

        /// Splits the matrix horizontally and vertically at the given indices into four corners and
        /// returns an array of each submatrix, in the following order:
        /// * top left.
        /// * top right.
        /// * bottom left.
        /// * bottom right.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row <= self.nrows()`.
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub fn split_at_mut(self, row: usize, col: usize) -> (Self, Self, Self, Self) {
            let (top_left, top_right, bot_left, bot_right) = self.into_const().split_at(row, col);
            unsafe {
                (
                    top_left.const_cast(),
                    top_right.const_cast(),
                    bot_left.const_cast(),
                    bot_right.const_cast(),
                )
            }
        }

        /// Splits the matrix horizontally at the given row into two parts and returns an array of
        /// each submatrix, in the following order:
        /// * top.
        /// * bottom.
        ///
        /// # Safety
        /// The behavior is undefined if the following condition is violated:
        /// * `row <= self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at_row_mut_unchecked(self, row: usize) -> (Self, Self) {
            let (top, bot) = self.into_const().split_at_row_unchecked(row);
            (top.const_cast(), bot.const_cast())
        }

        /// Splits the matrix horizontally at the given row into two parts and returns an array of
        /// each submatrix, in the following order:
        /// * top.
        /// * bottom.
        ///
        /// # Panics
        /// The function panics if the following condition is violated:
        /// * `row <= self.nrows()`.
        #[inline(always)]
        #[track_caller]
        pub fn split_at_row_mut(self, row: usize) -> (Self, Self) {
            let (top, bot) = self.into_const().split_at_row(row);
            unsafe { (top.const_cast(), bot.const_cast()) }
        }

        /// Splits the matrix vertically at the given row into two parts and returns an array of
        /// each submatrix, in the following order:
        /// * left.
        /// * right.
        ///
        /// # Safety
        /// The behavior is undefined if the following condition is violated:
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn split_at_col_mut_unchecked(self, col: usize) -> (Self, Self) {
            let (left, right) = self.into_const().split_at_col_unchecked(col);
            (left.const_cast(), right.const_cast())
        }

        /// Splits the matrix vertically at the given row into two parts and returns an array of
        /// each submatrix, in the following order:
        /// * left.
        /// * right.
        ///
        /// # Panics
        /// The function panics if the following condition is violated:
        /// * `col <= self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub fn split_at_col_mut(self, col: usize) -> (Self, Self) {
            let (left, right) = self.into_const().split_at_col(col);
            unsafe { (left.const_cast(), right.const_cast()) }
        }

        #[inline(always)]
        #[track_caller]
        #[deprecated = "replaced by `MatMut::split_at_mut`"]
        pub fn split_at(self, row: usize, col: usize) -> (Self, Self, Self, Self) {
            self.split_at_mut(row, col)
        }

        #[inline(always)]
        #[track_caller]
        #[deprecated = "replaced by `MatMut::split_at_row_mut`"]
        pub fn split_at_row(self, row: usize) -> (Self, Self) {
            let (top, bot) = self.into_const().split_at_row(row);
            unsafe { (top.const_cast(), bot.const_cast()) }
        }

        #[inline(always)]
        #[track_caller]
        #[deprecated = "replaced by `MatMut::split_at_col_mut`"]
        pub fn split_at_col(self, col: usize) -> (Self, Self) {
            let (left, right) = self.into_const().split_at_col(col);
            unsafe { (left.const_cast(), right.const_cast()) }
        }

        /// Returns mutable references to the element at the given indices, or submatrices if either
        /// `row` or `col` is a range.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row` must be contained in `[0, self.nrows())`.
        /// * `col` must be contained in `[0, self.ncols())`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn get_mut_unchecked<RowRange, ColRange>(
            self,
            row: RowRange,
            col: ColRange,
        ) -> <Self as MatIndex<RowRange, ColRange>>::Target
        where
            Self: MatIndex<RowRange, ColRange>,
        {
            <Self as MatIndex<RowRange, ColRange>>::get_unchecked(self, row, col)
        }

        #[inline(always)]
        #[track_caller]
        #[deprecated = "replaced by `MatMut::get_mut_unchecked`"]
        pub unsafe fn get_unchecked<RowRange, ColRange>(
            self,
            row: RowRange,
            col: ColRange,
        ) -> <Self as MatIndex<RowRange, ColRange>>::Target
        where
            Self: MatIndex<RowRange, ColRange>,
        {
            self.get_mut_unchecked(row, col)
        }

        /// Returns mutable references to the element at the given indices, or submatrices if either
        /// `row` or `col` is a range, with bound checks.
        ///
        /// # Note
        /// The values pointed to by the references are expected to be initialized, even if the
        /// pointed-to value is not read, otherwise the behavior is undefined.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row` must be contained in `[0, self.nrows())`.
        /// * `col` must be contained in `[0, self.ncols())`.
        #[inline(always)]
        #[track_caller]
        pub fn get_mut<RowRange, ColRange>(
            self,
            row: RowRange,
            col: ColRange,
        ) -> <Self as MatIndex<RowRange, ColRange>>::Target
        where
            Self: MatIndex<RowRange, ColRange>,
        {
            <Self as MatIndex<RowRange, ColRange>>::get(self, row, col)
        }

        #[inline(always)]
        #[track_caller]
        #[deprecated = "replaced by `MatMut::get_mut`"]
        pub fn get<RowRange, ColRange>(
            self,
            row: RowRange,
            col: ColRange,
        ) -> <Self as MatIndex<RowRange, ColRange>>::Target
        where
            Self: MatIndex<RowRange, ColRange>,
        {
            self.get_mut(row, col)
        }

        /// Reads the value of the element at the given indices.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn read_unchecked(&self, row: usize, col: usize) -> E {
            self.rb().read_unchecked(row, col)
        }

        /// Reads the value of the element at the given indices, with bound checks.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub fn read(&self, row: usize, col: usize) -> E {
            self.rb().read(row, col)
        }

        /// Writes the value to the element at the given indices.
        ///
        /// # Safety
        /// The behavior is undefined if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub unsafe fn write_unchecked(&mut self, row: usize, col: usize, value: E) {
            let units = value.faer_into_units();
            let zipped = E::faer_zip(units, (*self).rb_mut().ptr_inbounds_at_mut(row, col));
            E::faer_map(
                zipped,
                #[inline(always)]
                |(unit, ptr)| *ptr = unit,
            );
        }

        /// Writes the value to the element at the given indices, with bound checks.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row < self.nrows()`.
        /// * `col < self.ncols()`.
        #[inline(always)]
        #[track_caller]
        pub fn write(&mut self, row: usize, col: usize, value: E) {
            assert!(all(row < self.nrows(), col < self.ncols()));
            unsafe { self.write_unchecked(row, col, value) };
        }

        /// Copies the values from the lower triangular part of `other` into the lower triangular
        /// part of `self`. The diagonal part is included.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `self.nrows() == other.nrows()`.
        /// * `self.ncols() == other.ncols()`.
        /// * `self.nrows() == self.ncols()`.
        #[track_caller]
        pub fn copy_from_triangular_lower(&mut self, other: impl AsMatRef<E>) {
            #[track_caller]
            #[inline(always)]
            fn implementation<E: Entity>(this: MatMut<'_, E>, other: MatRef<'_, E>) {
                zipped!(this, other).for_each_triangular_lower(
                    zip::Diag::Include,
                    #[inline(always)]
                    |unzipped!(mut dst, src)| dst.write(src.read()),
                );
            }
            implementation(self.rb_mut(), other.as_mat_ref())
        }

        /// Copies the values from the lower triangular part of `other` into the lower triangular
        /// part of `self`. The diagonal part is excluded.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `self.nrows() == other.nrows()`.
        /// * `self.ncols() == other.ncols()`.
        /// * `self.nrows() == self.ncols()`.
        #[track_caller]
        pub fn copy_from_strict_triangular_lower(&mut self, other: impl AsMatRef<E>) {
            #[track_caller]
            #[inline(always)]
            fn implementation<E: Entity>(this: MatMut<'_, E>, other: MatRef<'_, E>) {
                zipped!(this, other).for_each_triangular_lower(
                    zip::Diag::Skip,
                    #[inline(always)]
                    |unzipped!(mut dst, src)| dst.write(src.read()),
                );
            }
            implementation(self.rb_mut(), other.as_mat_ref())
        }

        /// Copies the values from the upper triangular part of `other` into the upper triangular
        /// part of `self`. The diagonal part is included.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `self.nrows() == other.nrows()`.
        /// * `self.ncols() == other.ncols()`.
        /// * `self.nrows() == self.ncols()`.
        #[track_caller]
        #[inline(always)]
        pub fn copy_from_triangular_upper(&mut self, other: impl AsMatRef<E>) {
            (*self)
                .rb_mut()
                .transpose_mut()
                .copy_from_triangular_lower(other.as_mat_ref().transpose())
        }

        /// Copies the values from the upper triangular part of `other` into the upper triangular
        /// part of `self`. The diagonal part is excluded.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `self.nrows() == other.nrows()`.
        /// * `self.ncols() == other.ncols()`.
        /// * `self.nrows() == self.ncols()`.
        #[track_caller]
        #[inline(always)]
        pub fn copy_from_strict_triangular_upper(&mut self, other: impl AsMatRef<E>) {
            (*self)
                .rb_mut()
                .transpose_mut()
                .copy_from_strict_triangular_lower(other.as_mat_ref().transpose())
        }

        /// Copies the values from `other` into `self`.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `self.nrows() == other.nrows()`.
        /// * `self.ncols() == other.ncols()`.
        #[track_caller]
        pub fn copy_from(&mut self, other: impl AsMatRef<E>) {
            #[track_caller]
            #[inline(always)]
            fn implementation<E: Entity>(this: MatMut<'_, E>, other: MatRef<'_, E>) {
                zipped!(this, other).for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
            }
            implementation(self.rb_mut(), other.as_mat_ref())
        }

        /// Fills the elements of `self` with zeros.
        #[track_caller]
        pub fn fill_zero(&mut self)
        where
            E: ComplexField,
        {
            zipped!(self.rb_mut()).for_each(
                #[inline(always)]
                |unzipped!(mut x)| x.write(E::faer_zero()),
            );
        }

        /// Fills the elements of `self` with copies of `constant`.
        #[track_caller]
        pub fn fill(&mut self, constant: E) {
            zipped!((*self).rb_mut()).for_each(
                #[inline(always)]
                |unzipped!(mut x)| x.write(constant),
            );
        }

        /// Returns a view over the transpose of `self`.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        /// let view = matrix.as_mut();
        /// let transpose = view.transpose_mut();
        ///
        /// let mut expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        /// assert_eq!(expected.as_mut(), transpose);
        /// ```
        #[inline(always)]
        #[must_use]
        pub fn transpose_mut(self) -> Self {
            unsafe {
                mat::from_raw_parts_mut(
                    E::faer_map(
                        from_copy::<E, _>(self.inner.inner.ptr),
                        #[inline(always)]
                        |ptr| ptr.as_ptr(),
                    ),
                    self.ncols(),
                    self.nrows(),
                    self.col_stride(),
                    self.row_stride(),
                )
            }
        }

        #[inline(always)]
        #[must_use]
        #[deprecated = "replaced by `MatMut::transpose_mut`"]
        pub fn transpose(self) -> Self {
            self.transpose_mut()
        }

        /// Returns a view over the conjugate of `self`.
        #[inline(always)]
        #[must_use]
        pub fn conjugate_mut(self) -> MatMut<'a, E::Conj>
        where
            E: Conjugate,
        {
            unsafe { self.into_const().conjugate().const_cast() }
        }

        #[inline(always)]
        #[must_use]
        #[deprecated = "replaced by `MatMut::conjugate_mut`"]
        pub fn conjugate(self) -> MatMut<'a, E::Conj>
        where
            E: Conjugate,
        {
            self.conjugate_mut()
        }

        /// Returns a view over the conjugate transpose of `self`.
        #[inline(always)]
        #[must_use]
        pub fn adjoint_mut(self) -> MatMut<'a, E::Conj>
        where
            E: Conjugate,
        {
            self.transpose_mut().conjugate_mut()
        }

        #[inline(always)]
        #[must_use]
        #[deprecated = "replaced by `MatMut::adjoint_mut`"]
        pub fn adjoint(self) -> MatMut<'a, E::Conj>
        where
            E: Conjugate,
        {
            self.adjoint_mut()
        }

        /// Returns a view over the canonical representation of `self`, as well as a flag declaring
        /// whether `self` is implicitly conjugated or not.
        #[inline(always)]
        #[must_use]
        pub fn canonicalize_mut(self) -> (MatMut<'a, E::Canonical>, Conj)
        where
            E: Conjugate,
        {
            let (canonical, conj) = self.into_const().canonicalize();
            unsafe { (canonical.const_cast(), conj) }
        }

        #[inline(always)]
        #[must_use]
        #[deprecated = "replaced by `MatMut::canonicalize_mut`"]
        pub fn canonicalize(self) -> (MatMut<'a, E::Canonical>, Conj)
        where
            E: Conjugate,
        {
            self.canonicalize_mut()
        }

        /// Returns a view over the `self`, with the rows in reversed order.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        /// let view = matrix.as_mut();
        /// let reversed_rows = view.reverse_rows_mut();
        ///
        /// let mut expected = mat![[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]];
        /// assert_eq!(expected.as_mut(), reversed_rows);
        /// ```
        #[inline(always)]
        #[must_use]
        pub fn reverse_rows_mut(self) -> Self {
            unsafe { self.into_const().reverse_rows().const_cast() }
        }

        #[inline(always)]
        #[must_use]
        #[deprecated = "replaced by `MatMut::reverse_rows_mut`"]
        pub fn reverse_rows(self) -> Self {
            self.reverse_rows_mut()
        }

        /// Returns a view over the `self`, with the columns in reversed order.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        /// let view = matrix.as_mut();
        /// let reversed_cols = view.reverse_cols_mut();
        ///
        /// let mut expected = mat![[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]];
        /// assert_eq!(expected.as_mut(), reversed_cols);
        /// ```
        #[inline(always)]
        #[must_use]
        pub fn reverse_cols_mut(self) -> Self {
            unsafe { self.into_const().reverse_cols().const_cast() }
        }

        #[inline(always)]
        #[must_use]
        #[deprecated = "replaced by `MatMut::reverse_cols_mut`"]
        pub fn reverse_cols(self) -> Self {
            unsafe { self.into_const().reverse_cols().const_cast() }
        }

        /// Returns a view over the `self`, with the rows and the columns in reversed order.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        /// let view = matrix.as_mut();
        /// let reversed = view.reverse_rows_and_cols_mut();
        ///
        /// let mut expected = mat![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
        /// assert_eq!(expected.as_mut(), reversed);
        /// ```
        #[inline(always)]
        #[must_use]
        pub fn reverse_rows_and_cols_mut(self) -> Self {
            unsafe { self.into_const().reverse_rows_and_cols().const_cast() }
        }

        #[inline(always)]
        #[must_use]
        #[deprecated = "replaced by `MatMut::reverse_rows_and_cols_mut`"]
        pub fn reverse_rows_and_cols(self) -> Self {
            unsafe { self.into_const().reverse_rows_and_cols().const_cast() }
        }

        /// Returns a view over the submatrix starting at indices `(row_start, col_start)`, and with
        /// dimensions `(nrows, ncols)`.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row_start <= self.nrows()`.
        /// * `col_start <= self.ncols()`.
        /// * `nrows <= self.nrows() - row_start`.
        /// * `ncols <= self.ncols() - col_start`.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let mut matrix = mat![
        ///     [1.0, 5.0, 9.0],
        ///     [2.0, 6.0, 10.0],
        ///     [3.0, 7.0, 11.0],
        ///     [4.0, 8.0, 12.0f64],
        /// ];
        ///
        /// let view = matrix.as_mut();
        /// let submatrix = view.submatrix_mut(2, 1, 2, 2);
        ///
        /// let mut expected = mat![[7.0, 11.0], [8.0, 12.0f64]];
        /// assert_eq!(expected.as_mut(), submatrix);
        /// ```
        #[track_caller]
        #[inline(always)]
        pub fn submatrix_mut(
            self,
            row_start: usize,
            col_start: usize,
            nrows: usize,
            ncols: usize,
        ) -> Self {
            unsafe {
                self.into_const()
                    .submatrix(row_start, col_start, nrows, ncols)
                    .const_cast()
            }
        }

        #[track_caller]
        #[inline(always)]
        #[deprecated = "replaced by `MatMut::submatrix_mut`"]
        pub fn submatrix(
            self,
            row_start: usize,
            col_start: usize,
            nrows: usize,
            ncols: usize,
        ) -> Self {
            self.submatrix_mut(row_start, col_start, nrows, ncols)
        }

        /// Returns a view over the submatrix starting at row `row_start`, and with number of rows
        /// `nrows`.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row_start <= self.nrows()`.
        /// * `nrows <= self.nrows() - row_start`.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let mut matrix = mat![
        ///     [1.0, 5.0, 9.0],
        ///     [2.0, 6.0, 10.0],
        ///     [3.0, 7.0, 11.0],
        ///     [4.0, 8.0, 12.0f64],
        /// ];
        ///
        /// let view = matrix.as_mut();
        /// let subrows = view.subrows_mut(1, 2);
        ///
        /// let mut expected = mat![[2.0, 6.0, 10.0], [3.0, 7.0, 11.0],];
        /// assert_eq!(expected.as_mut(), subrows);
        /// ```
        #[track_caller]
        #[inline(always)]
        pub fn subrows_mut(self, row_start: usize, nrows: usize) -> Self {
            unsafe { self.into_const().subrows(row_start, nrows).const_cast() }
        }

        #[track_caller]
        #[inline(always)]
        #[deprecated = "replaced by `MatMut::subrows_mut`"]
        pub fn subrows(self, row_start: usize, nrows: usize) -> Self {
            self.subrows_mut(row_start, nrows)
        }

        /// Returns a view over the submatrix starting at column `col_start`, and with number of
        /// columns `ncols`.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col_start <= self.ncols()`.
        /// * `ncols <= self.ncols() - col_start`.
        ///
        /// # Example
        /// ```
        /// use faer_core::mat;
        ///
        /// let mut matrix = mat![
        ///     [1.0, 5.0, 9.0],
        ///     [2.0, 6.0, 10.0],
        ///     [3.0, 7.0, 11.0],
        ///     [4.0, 8.0, 12.0f64],
        /// ];
        ///
        /// let view = matrix.as_mut();
        /// let subcols = view.subcols_mut(2, 1);
        ///
        /// let mut expected = mat![[9.0], [10.0], [11.0], [12.0f64]];
        /// assert_eq!(expected.as_mut(), subcols);
        /// ```
        #[track_caller]
        #[inline(always)]
        pub fn subcols_mut(self, col_start: usize, ncols: usize) -> Self {
            unsafe { self.into_const().subcols(col_start, ncols).const_cast() }
        }

        #[track_caller]
        #[inline(always)]
        #[deprecated = "replaced by `MatMut::subcols_mut`"]
        pub fn subcols(self, col_start: usize, ncols: usize) -> Self {
            self.subcols_mut(col_start, ncols)
        }

        /// Returns a view over the row at the given index.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `row_idx < self.nrows()`.
        #[track_caller]
        #[inline(always)]
        pub fn row_mut(self, row_idx: usize) -> RowMut<'a, E> {
            unsafe { self.into_const().row(row_idx).const_cast() }
        }

        #[track_caller]
        #[inline(always)]
        #[deprecated = "replaced by `MatMut::row_mut`"]
        pub fn row(self, row_idx: usize) -> RowMut<'a, E> {
            self.row_mut(row_idx)
        }

        /// Returns a view over the column at the given index.
        ///
        /// # Panics
        /// The function panics if any of the following conditions are violated:
        /// * `col_idx < self.ncols()`.
        #[track_caller]
        #[inline(always)]
        pub fn col_mut(self, col_idx: usize) -> ColMut<'a, E> {
            unsafe { self.into_const().col(col_idx).const_cast() }
        }

        #[track_caller]
        #[inline(always)]
        #[deprecated = "replaced by `MatMut::col_mut`"]
        pub fn col(self, col_idx: usize) -> ColMut<'a, E> {
            self.col_mut(col_idx)
        }

        /// Given a matrix with a single column, returns an object that interprets
        /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
        #[track_caller]
        #[inline(always)]
        pub fn column_vector_as_diagonal_mut(self) -> Matrix<DiagMut<'a, E>> {
            assert!(self.ncols() == 1);
            Matrix {
                inner: DiagMut {
                    inner: self.col_mut(0),
                },
            }
        }

        #[track_caller]
        #[inline(always)]
        #[deprecated = "replaced by `MatMut::column_vector_as_diagonal_mut`"]
        pub fn column_vector_as_diagonal(self) -> Matrix<DiagMut<'a, E>> {
            self.column_vector_as_diagonal_mut()
        }

        #[inline(always)]
        pub fn diagonal_mut(self) -> Matrix<DiagMut<'a, E>> {
            let size = self.nrows().min(self.ncols());
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            unsafe {
                Matrix {
                    inner: DiagMut {
                        inner: col::from_raw_parts_mut(
                            self.as_ptr_mut(),
                            size,
                            row_stride + col_stride,
                        ),
                    },
                }
            }
        }

        #[inline(always)]
        #[deprecated = "replaced by `MatMut::diagonal_mut`"]
        pub fn diagonal(self) -> Matrix<DiagMut<'a, E>> {
            self.diagonal_mut()
        }

        /// Returns an owning [`Mat`] of the data
        #[inline]
        pub fn to_owned(&self) -> Mat<E::Canonical>
        where
            E: Conjugate,
        {
            self.rb().to_owned()
        }

        /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
        #[inline]
        pub fn has_nan(&self) -> bool
        where
            E: ComplexField,
        {
            self.rb().has_nan()
        }

        /// Returns `true` if all of the elements are finite, otherwise returns `false`.
        #[inline]
        pub fn is_all_finite(&self) -> bool
        where
            E: ComplexField,
        {
            self.rb().is_all_finite()
        }

        /// Returns the maximum norm of `self`.
        #[inline]
        pub fn norm_max(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_max((*self).rb())
        }
        /// Returns the L2 norm of `self`.
        #[inline]
        pub fn norm_l2(&self) -> E::Real
        where
            E: ComplexField,
        {
            norm_l2((*self).rb())
        }

        /// Returns a view over the matrix.
        #[inline]
        pub fn as_ref(&self) -> MatRef<'_, E> {
            self.rb()
        }

        /// Returns a mutable view over the matrix.
        #[inline]
        pub fn as_mut(&mut self) -> MatMut<'_, E> {
            self.rb_mut()
        }

        /// Returns an iterator that provides successive chunks of the columns of this matrix, with
        /// each having at most `chunk_size` columns.
        ///
        /// If the number of columns is a multiple of `chunk_size`, then all chunks have
        /// `chunk_size` columns.
        #[inline]
        #[track_caller]
        pub fn col_chunks_mut(
            self,
            chunk_size: usize,
        ) -> impl 'a + DoubleEndedIterator<Item = MatMut<'a, E>> {
            self.into_const()
                .col_chunks(chunk_size)
                .map(|chunk| unsafe { chunk.const_cast() })
        }

        #[inline]
        #[track_caller]
        #[deprecated = "replaced by `MatMut::col_chunks_mut`"]
        pub fn into_col_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + DoubleEndedIterator<Item = MatMut<'a, E>> {
            self.col_chunks_mut(chunk_size)
        }

        /// Returns an iterator that provides successive chunks of the rows of this matrix,
        /// with each having at most `chunk_size` rows.
        ///
        /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
        /// rows.
        #[inline]
        #[track_caller]
        pub fn row_chunks_mut(
            self,
            chunk_size: usize,
        ) -> impl 'a + DoubleEndedIterator<Item = MatMut<'a, E>> {
            self.into_const()
                .row_chunks(chunk_size)
                .map(|chunk| unsafe { chunk.const_cast() })
        }

        #[inline]
        #[track_caller]
        #[deprecated = "replaced by `MatMut::row_chunks_mut`"]
        pub fn into_row_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + DoubleEndedIterator<Item = MatMut<'a, E>> {
            self.row_chunks_mut(chunk_size)
        }

        /// Returns a parallel iterator that provides successive chunks of the columns of this
        /// matrix, with each having at most `chunk_size` columns.
        ///
        /// If the number of columns is a multiple of `chunk_size`, then all chunks have
        /// `chunk_size` columns.
        ///
        /// Only available with the `rayon` feature.
        #[cfg(feature = "rayon")]
        #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
        #[inline]
        #[track_caller]
        pub fn par_col_chunks_mut(
            self,
            chunk_size: usize,
        ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E>> {
            use rayon::prelude::*;
            self.into_const()
                .par_col_chunks(chunk_size)
                .map(|chunk| unsafe { chunk.const_cast() })
        }

        #[cfg(feature = "rayon")]
        #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
        #[inline]
        #[track_caller]
        #[deprecated = "replaced by `MatMut::par_col_chunks_mut`"]
        pub fn into_par_col_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E>> {
            self.par_col_chunks_mut(chunk_size)
        }

        /// Returns a parallel iterator that provides successive chunks of the rows of this matrix,
        /// with each having at most `chunk_size` rows.
        ///
        /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
        /// rows.
        ///
        /// Only available with the `rayon` feature.
        #[cfg(feature = "rayon")]
        #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
        #[inline]
        #[track_caller]
        pub fn par_row_chunks_mut(
            self,
            chunk_size: usize,
        ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E>> {
            use rayon::prelude::*;
            self.into_const()
                .par_row_chunks(chunk_size)
                .map(|chunk| unsafe { chunk.const_cast() })
        }

        #[cfg(feature = "rayon")]
        #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
        #[inline]
        #[track_caller]
        #[deprecated = "replaced by `MatMut::par_row_chunks_mut`"]
        pub fn into_par_row_chunks(
            self,
            chunk_size: usize,
        ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E>> {
            self.par_row_chunks_mut(chunk_size)
        }
    }

    impl<'a, E: RealField> MatRef<'a, Complex<E>> {
        #[inline(always)]
        pub fn real_imag(self) -> Complex<MatRef<'a, E>> {
            let row_stride = self.row_stride();
            let col_stride = self.col_stride();
            let nrows = self.nrows();
            let ncols = self.ncols();
            let Complex { re, im } = self.as_ptr();
            unsafe {
                Complex {
                    re: mat::from_raw_parts(re, nrows, ncols, row_stride, col_stride),
                    im: mat::from_raw_parts(im, nrows, ncols, row_stride, col_stride),
                }
            }
        }
    }

    impl<'a, E: RealField> MatMut<'a, Complex<E>> {
        #[inline(always)]
        pub fn real_imag_mut(self) -> Complex<MatMut<'a, E>> {
            let Complex { re, im } = self.into_const().real_imag();
            unsafe {
                Complex {
                    re: re.const_cast(),
                    im: im.const_cast(),
                }
            }
        }
    }
};

#[repr(C)]
struct RawMatUnit<T: 'static> {
    ptr: NonNull<T>,
    row_capacity: usize,
    col_capacity: usize,
}

impl<T: 'static> RawMatUnit<T> {
    pub fn new(row_capacity: usize, col_capacity: usize) -> Self {
        let dangling = NonNull::<T>::dangling();
        if core::mem::size_of::<T>() == 0 {
            Self {
                ptr: dangling,
                row_capacity,
                col_capacity,
            }
        } else {
            let cap = row_capacity
                .checked_mul(col_capacity)
                .unwrap_or_else(capacity_overflow);
            let cap_bytes = cap
                .checked_mul(core::mem::size_of::<T>())
                .unwrap_or_else(capacity_overflow);
            if cap_bytes > isize::MAX as usize {
                capacity_overflow::<()>();
            }

            use alloc::alloc::{alloc, handle_alloc_error, Layout};

            let layout = Layout::from_size_align(cap_bytes, align_for::<T>())
                .ok()
                .unwrap_or_else(capacity_overflow);

            let ptr = if layout.size() == 0 {
                dangling
            } else {
                // SAFETY: we checked that layout has non zero size
                let ptr = unsafe { alloc(layout) } as *mut T;
                if ptr.is_null() {
                    handle_alloc_error(layout)
                } else {
                    // SAFETY: we checked that the pointer is not null
                    unsafe { NonNull::<T>::new_unchecked(ptr) }
                }
            };

            Self {
                ptr,
                row_capacity,
                col_capacity,
            }
        }
    }
}

impl<T: 'static> Drop for RawMatUnit<T> {
    fn drop(&mut self) {
        use alloc::alloc::{dealloc, Layout};
        // this cannot overflow because we already allocated this much memory
        // self.row_capacity.wrapping_mul(self.col_capacity) may overflow if T is a zst
        // but that's fine since we immediately multiply it by 0.
        let alloc_size =
            self.row_capacity.wrapping_mul(self.col_capacity) * core::mem::size_of::<T>();
        if alloc_size != 0 {
            // SAFETY: pointer was allocated with alloc::alloc::alloc
            unsafe {
                dealloc(
                    self.ptr.as_ptr() as *mut u8,
                    Layout::from_size_align_unchecked(alloc_size, align_for::<T>()),
                );
            }
        }
    }
}

#[repr(C)]
struct RawMat<E: Entity> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    row_capacity: usize,
    col_capacity: usize,
}

#[cold]
fn capacity_overflow_impl() -> ! {
    panic!("capacity overflow")
}

#[inline(always)]
fn capacity_overflow<T>() -> T {
    capacity_overflow_impl();
}

#[doc(hidden)]
#[inline(always)]
pub fn is_vectorizable<T: 'static>() -> bool {
    coe::is_same::<f32, T>()
        || coe::is_same::<f64, T>()
        || coe::is_same::<c32, T>()
        || coe::is_same::<c64, T>()
        || coe::is_same::<c32conj, T>()
        || coe::is_same::<c64conj, T>()
}

// https://rust-lang.github.io/hashbrown/src/crossbeam_utils/cache_padded.rs.html#128-130
#[doc(hidden)]
pub const CACHELINE_ALIGN: usize = {
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
    ))]
    {
        128
    }
    #[cfg(any(
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64",
    ))]
    {
        32
    }
    #[cfg(target_arch = "s390x")]
    {
        256
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64",
        target_arch = "s390x",
    )))]
    {
        64
    }
};

#[doc(hidden)]
#[inline(always)]
pub fn align_for<T: 'static>() -> usize {
    if is_vectorizable::<T>() {
        Ord::max(
            core::mem::size_of::<T>(),
            Ord::max(core::mem::align_of::<T>(), CACHELINE_ALIGN),
        )
    } else {
        core::mem::align_of::<T>()
    }
}

impl<E: Entity> RawMat<E> {
    pub fn new(row_capacity: usize, col_capacity: usize) -> Self {
        // allocate the unit matrices
        let group = E::faer_map(E::UNIT, |()| {
            RawMatUnit::<E::Unit>::new(row_capacity, col_capacity)
        });

        let group = E::faer_map(group, ManuallyDrop::new);

        Self {
            ptr: into_copy::<E, _>(E::faer_map(group, |mat| mat.ptr)),
            row_capacity,
            col_capacity,
        }
    }
}

impl<E: Entity> Drop for RawMat<E> {
    fn drop(&mut self) {
        drop(E::faer_map(from_copy::<E, _>(self.ptr), |ptr| RawMatUnit {
            ptr,
            row_capacity: self.row_capacity,
            col_capacity: self.col_capacity,
        }));
    }
}

/// Heap allocated resizable matrix, similar to a 2D [`Vec`].
///
/// # Note
///
/// The memory layout of `Mat` is guaranteed to be column-major, meaning that it has a row stride
/// of `1`, and an unspecified column stride that can be queried with [`Mat::col_stride`].
///
/// This implies that while each individual column is stored contiguously in memory, the matrix as
/// a whole may not necessarily be contiguous. The implementation may add padding at the end of
/// each column when overaligning each column can provide a performance gain.
///
/// Let us consider a 3×4 matrix
///
/// ```notcode
///  0 │ 3 │ 6 │  9
/// ───┼───┼───┼───
///  1 │ 4 │ 7 │ 10
/// ───┼───┼───┼───
///  2 │ 5 │ 8 │ 11
/// ```
/// The memory representation of the data held by such a matrix could look like the following:
///
/// ```notcode
/// 0 1 2 X 3 4 5 X 6 7 8 X 9 10 11 X
/// ```
///
/// where X represents padding elements.
pub type Mat<E> = Matrix<DenseOwn<E>>;

/// Heap allocated resizable column vector.
///
/// # Note
///
/// The memory layout of `Col` is guaranteed to be column-major, meaning that it has a row stride
/// of `1`.
pub type Col<E> = Matrix<DenseColOwn<E>>;

/// Heap allocated resizable row vector.
///
/// # Note
///
/// The memory layout of `Col` is guaranteed to be row-major, meaning that it has a column stride
/// of `1`.
pub type Row<E> = Matrix<DenseRowOwn<E>>;

#[repr(C)]
struct MatUnit<T: 'static> {
    raw: RawMatUnit<T>,
    nrows: usize,
    ncols: usize,
}

impl<E: Entity> Clone for Mat<E> {
    fn clone(&self) -> Self {
        let this = self.as_ref();
        unsafe {
            Self::from_fn(self.nrows(), self.ncols(), |i, j| {
                E::faer_from_units(E::faer_deref(this.get_unchecked(i, j)))
            })
        }
    }
}

impl<T> MatUnit<T> {
    #[cold]
    fn do_reserve_exact(&mut self, mut new_row_capacity: usize, mut new_col_capacity: usize) {
        new_row_capacity = self.raw.row_capacity.max(new_row_capacity);
        new_col_capacity = self.raw.col_capacity.max(new_col_capacity);

        let new_ptr = if self.raw.row_capacity == new_row_capacity
            && self.raw.row_capacity != 0
            && self.raw.col_capacity != 0
        {
            // case 1:
            // we have enough row capacity, and we've already allocated memory.
            // use realloc to get extra column memory

            use alloc::alloc::{handle_alloc_error, realloc, Layout};

            // this shouldn't overflow since we already hold this many bytes
            let old_cap = self.raw.row_capacity * self.raw.col_capacity;
            let old_cap_bytes = old_cap * core::mem::size_of::<T>();

            let new_cap = new_row_capacity
                .checked_mul(new_col_capacity)
                .unwrap_or_else(capacity_overflow);
            let new_cap_bytes = new_cap
                .checked_mul(core::mem::size_of::<T>())
                .unwrap_or_else(capacity_overflow);

            if new_cap_bytes > isize::MAX as usize {
                capacity_overflow::<()>();
            }

            // SAFETY: this shouldn't overflow since we already checked that it's valid during
            // allocation
            let old_layout =
                unsafe { Layout::from_size_align_unchecked(old_cap_bytes, align_for::<T>()) };
            let new_layout = Layout::from_size_align(new_cap_bytes, align_for::<T>())
                .ok()
                .unwrap_or_else(capacity_overflow);

            // SAFETY:
            // * old_ptr is non null and is the return value of some previous call to alloc
            // * old_layout is the same layout that was used to provide the old allocation
            // * new_cap_bytes is non zero since new_row_capacity and new_col_capacity are larger
            // than self.raw.row_capacity and self.raw.col_capacity respectively, and the computed
            // product doesn't overflow.
            // * new_cap_bytes, when rounded up to the nearest multiple of the alignment does not
            // overflow, since we checked that we can create new_layout with it.
            unsafe {
                let old_ptr = self.raw.ptr.as_ptr();
                let new_ptr = realloc(old_ptr as *mut u8, old_layout, new_cap_bytes);
                if new_ptr.is_null() {
                    handle_alloc_error(new_layout);
                }
                new_ptr as *mut T
            }
        } else {
            // case 2:
            // use alloc and move stuff manually.

            // allocate new memory region
            let new_ptr = {
                let m = ManuallyDrop::new(RawMatUnit::<T>::new(new_row_capacity, new_col_capacity));
                m.ptr.as_ptr()
            };

            let old_ptr = self.raw.ptr.as_ptr();

            // copy each column to new matrix
            for j in 0..self.ncols {
                // SAFETY:
                // * pointer offsets can't overflow since they're within an already allocated
                // memory region less than isize::MAX bytes in size.
                // * new and old allocation can't overlap, so copy_nonoverlapping is fine here.
                unsafe {
                    let old_ptr = old_ptr.add(j * self.raw.row_capacity);
                    let new_ptr = new_ptr.add(j * new_row_capacity);
                    core::ptr::copy_nonoverlapping(old_ptr, new_ptr, self.nrows);
                }
            }

            // deallocate old matrix memory
            let _ = RawMatUnit::<T> {
                // SAFETY: this ptr was checked to be non null, or was acquired from a NonNull
                // pointer.
                ptr: unsafe { NonNull::new_unchecked(old_ptr) },
                row_capacity: self.raw.row_capacity,
                col_capacity: self.raw.col_capacity,
            };

            new_ptr
        };
        self.raw.row_capacity = new_row_capacity;
        self.raw.col_capacity = new_col_capacity;
        self.raw.ptr = unsafe { NonNull::<T>::new_unchecked(new_ptr) };
    }
}

impl<E: Entity> Drop for DenseOwn<E> {
    fn drop(&mut self) {
        drop(RawMat::<E> {
            ptr: self.inner.ptr,
            row_capacity: self.row_capacity,
            col_capacity: self.col_capacity,
        });
    }
}
impl<E: Entity> Drop for DenseColOwn<E> {
    fn drop(&mut self) {
        drop(RawMat::<E> {
            ptr: self.inner.ptr,
            row_capacity: self.row_capacity,
            col_capacity: 1,
        });
    }
}
impl<E: Entity> Drop for DenseRowOwn<E> {
    fn drop(&mut self) {
        drop(RawMat::<E> {
            ptr: self.inner.ptr,
            row_capacity: self.col_capacity,
            col_capacity: 1,
        });
    }
}

impl<E: Entity> Default for Mat<E> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
impl<E: Entity> Default for Col<E> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
impl<E: Entity> Default for Row<E> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Entity> Col<E> {
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: DenseColOwn {
                inner: VecOwnImpl {
                    ptr: into_copy::<E, _>(E::faer_map(E::UNIT, |()| {
                        NonNull::<E::Unit>::dangling()
                    })),
                    len: 0,
                },
                row_capacity: 0,
            },
        }
    }

    /// Returns a new column vector with 0 rows, with enough capacity to hold a maximum of
    /// `row_capacity` rows columns without reallocating. If `row_capacity` is `0`,
    /// the function will not allocate.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn with_capacity(row_capacity: usize) -> Self {
        let raw = ManuallyDrop::new(RawMat::<E>::new(row_capacity, 1));
        Self {
            inner: DenseColOwn {
                inner: VecOwnImpl {
                    ptr: raw.ptr,
                    len: 0,
                },
                row_capacity: raw.row_capacity,
            },
        }
    }

    /// Returns a new matrix with number of rows `nrows`, filled with the provided function.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn from_fn(nrows: usize, f: impl FnMut(usize) -> E) -> Self {
        let mut this = Self::new();
        this.resize_with(nrows, f);
        this
    }

    /// Returns a new matrix with number of rows `nrows`, filled with zeros.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn zeros(nrows: usize) -> Self
    where
        E: ComplexField,
    {
        Self::from_fn(nrows, |_| E::faer_zero())
    }

    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.inner.inner.len
    }
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        1
    }

    /// Set the dimensions of the matrix.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `nrows < self.row_capacity()`.
    /// * The elements that were previously out of bounds but are now in bounds must be
    /// initialized.
    #[inline]
    pub unsafe fn set_nrows(&mut self, nrows: usize) {
        self.inner.inner.len = nrows;
    }

    /// Returns a pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr(&self) -> GroupFor<E, *const E::Unit> {
        E::faer_map(from_copy::<E, _>(self.inner.inner.ptr), |ptr| {
            ptr.as_ptr() as *const E::Unit
        })
    }

    /// Returns a mutable pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr_mut(&mut self) -> GroupFor<E, *mut E::Unit> {
        E::faer_map(from_copy::<E, _>(self.inner.inner.ptr), |ptr| ptr.as_ptr())
    }

    /// Returns the row capacity, that is, the number of rows that the matrix is able to hold
    /// without needing to reallocate, excluding column insertions.
    #[inline]
    pub fn row_capacity(&self) -> usize {
        self.inner.row_capacity
    }

    /// Returns the offset between the first elements of two successive rows in the matrix.
    /// Always returns `1` since the matrix is column major.
    #[inline]
    pub fn row_stride(&self) -> isize {
        1
    }

    #[cold]
    fn do_reserve_exact(&mut self, mut new_row_capacity: usize) {
        if is_vectorizable::<E::Unit>() {
            let align_factor = align_for::<E::Unit>() / core::mem::size_of::<E::Unit>();
            new_row_capacity = new_row_capacity
                .msrv_checked_next_multiple_of(align_factor)
                .unwrap();
        }

        let nrows = self.inner.inner.len;
        let old_row_capacity = self.inner.row_capacity;

        let mut this = ManuallyDrop::new(core::mem::take(self));
        {
            let mut this_group =
                E::faer_map(from_copy::<E, _>(this.inner.inner.ptr), |ptr| MatUnit {
                    raw: RawMatUnit {
                        ptr,
                        row_capacity: old_row_capacity,
                        col_capacity: 1,
                    },
                    nrows,
                    ncols: 1,
                });

            E::faer_map(E::faer_as_mut(&mut this_group), |mat_unit| {
                mat_unit.do_reserve_exact(new_row_capacity, 1);
            });

            let this_group = E::faer_map(this_group, ManuallyDrop::new);
            this.inner.inner.ptr =
                into_copy::<E, _>(E::faer_map(this_group, |mat_unit| mat_unit.raw.ptr));
            this.inner.row_capacity = new_row_capacity;
        }
        *self = ManuallyDrop::into_inner(this);
    }

    /// Reserves the minimum capacity for `row_capacity` rows without reallocating. Does nothing if
    /// the capacity is already sufficient.
    ///
    /// # Panics
    /// The function panics if the new total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn reserve_exact(&mut self, row_capacity: usize) {
        if self.row_capacity() >= row_capacity {
            // do nothing
        } else if core::mem::size_of::<E::Unit>() == 0 {
            self.inner.row_capacity = self.row_capacity().max(row_capacity);
        } else {
            self.do_reserve_exact(row_capacity);
        }
    }

    unsafe fn insert_block_with<F: FnMut(usize) -> E>(
        &mut self,
        f: &mut F,
        row_start: usize,
        row_end: usize,
    ) {
        debug_assert!(row_start <= row_end);

        let ptr = self.as_ptr_mut();

        for i in row_start..row_end {
            // SAFETY:
            // * pointer to element at index (i, j), which is within the
            // allocation since we reserved enough space
            // * writing to this memory region is sound since it is properly
            // aligned and valid for writes
            let ptr_ij = E::faer_map(E::faer_copy(&ptr), |ptr| ptr.add(i));
            let value = E::faer_into_units(f(i));

            E::faer_map(E::faer_zip(ptr_ij, value), |(ptr_ij, value)| {
                core::ptr::write(ptr_ij, value)
            });
        }
    }

    fn erase_last_rows(&mut self, new_nrows: usize) {
        let old_nrows = self.nrows();
        debug_assert!(new_nrows <= old_nrows);
        self.inner.inner.len = new_nrows;
    }

    unsafe fn insert_last_rows_with<F: FnMut(usize) -> E>(&mut self, f: &mut F, new_nrows: usize) {
        let old_nrows = self.nrows();

        debug_assert!(new_nrows > old_nrows);

        self.insert_block_with(f, old_nrows, new_nrows);
        self.inner.inner.len = new_nrows;
    }

    /// Resizes the vector in-place so that the new number of rows is `new_nrows`.
    /// New elements are created with the given function `f`, so that elements at index `i`
    /// are created by calling `f(i)`.
    pub fn resize_with(&mut self, new_nrows: usize, f: impl FnMut(usize) -> E) {
        let mut f = f;
        let old_nrows = self.nrows();

        if new_nrows <= old_nrows {
            self.erase_last_rows(new_nrows);
        } else {
            self.reserve_exact(new_nrows);
            unsafe {
                self.insert_last_rows_with(&mut f, new_nrows);
            }
        }
    }

    /// Returns a reference to a slice over the column.
    #[inline]
    #[track_caller]
    pub fn as_slice(&self) -> GroupFor<E, &[E::Unit]> {
        let nrows = self.nrows();
        let ptr = self.as_ref().as_ptr();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, nrows) },
        )
    }

    /// Returns a mutable reference to a slice over the column.
    #[inline]
    #[track_caller]
    pub fn as_slice_mut(&mut self) -> GroupFor<E, &mut [E::Unit]> {
        let nrows = self.nrows();
        let ptr = self.as_ptr_mut();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, nrows) },
        )
    }

    /// Returns a view over the vector.
    #[inline]
    pub fn as_ref(&self) -> ColRef<'_, E> {
        unsafe { col::from_raw_parts(self.as_ptr(), self.nrows(), 1) }
    }

    /// Returns a mutable view over the vector.
    #[inline]
    pub fn as_mut(&mut self) -> ColMut<'_, E> {
        unsafe { col::from_raw_parts_mut(self.as_ptr_mut(), self.nrows(), 1) }
    }

    /// Returns references to the element at the given index, or submatrices if `row` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline]
    pub unsafe fn get_unchecked<RowRange>(
        &self,
        row: RowRange,
    ) -> <ColRef<'_, E> as ColIndex<RowRange>>::Target
    where
        for<'a> ColRef<'a, E>: ColIndex<RowRange>,
    {
        self.as_ref().get_unchecked(row)
    }

    /// Returns references to the element at the given index, or submatrices if `row` is a range,
    /// with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline]
    pub fn get<RowRange>(&self, row: RowRange) -> <ColRef<'_, E> as ColIndex<RowRange>>::Target
    where
        for<'a> ColRef<'a, E>: ColIndex<RowRange>,
    {
        self.as_ref().get(row)
    }

    /// Returns mutable references to the element at the given index, or submatrices if
    /// `row` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline]
    pub unsafe fn get_mut_unchecked<RowRange>(
        &mut self,
        row: RowRange,
    ) -> <ColMut<'_, E> as ColIndex<RowRange>>::Target
    where
        for<'a> ColMut<'a, E>: ColIndex<RowRange>,
    {
        self.as_mut().get_unchecked_mut(row)
    }

    /// Returns mutable references to the element at the given index, or submatrices if
    /// `row` is a range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline]
    pub fn get_mut<RowRange>(
        &mut self,
        row: RowRange,
    ) -> <ColMut<'_, E> as ColIndex<RowRange>>::Target
    where
        for<'a> ColMut<'a, E>: ColIndex<RowRange>,
    {
        self.as_mut().get_mut(row)
    }

    /// Reads the value of the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, row: usize) -> E {
        self.as_ref().read_unchecked(row)
    }

    /// Reads the value of the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, row: usize) -> E {
        self.as_ref().read(row)
    }

    /// Writes the value to the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, row: usize, value: E) {
        self.as_mut().write_unchecked(row, value);
    }

    /// Writes the value to the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, row: usize, value: E) {
        self.as_mut().write(row, value);
    }

    /// Copies the values from `other` into `self`.
    #[inline(always)]
    #[track_caller]
    pub fn copy_from(&mut self, other: impl AsColRef<E>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity>(this: &mut Col<E>, other: ColRef<'_, E>) {
            let mut mat = Col::<E>::new();
            mat.resize_with(
                other.nrows(),
                #[inline(always)]
                |row| unsafe { other.read_unchecked(row) },
            );
            *this = mat;
        }
        implementation(self, other.as_col_ref());
    }

    /// Fills the elements of `self` with zeros.
    #[inline(always)]
    #[track_caller]
    pub fn fill_zero(&mut self)
    where
        E: ComplexField,
    {
        self.as_mut().fill_zero()
    }

    /// Fills the elements of `self` with copies of `constant`.
    #[inline(always)]
    #[track_caller]
    pub fn fill(&mut self, constant: E) {
        self.as_mut().fill(constant)
    }

    /// Returns a view over the transpose of `self`.
    #[inline]
    pub fn transpose(&self) -> RowRef<'_, E> {
        self.as_ref().transpose()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(&self) -> ColRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(&self) -> RowRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
    }

    /// Returns an owning [`Col`] of the data
    #[inline]
    pub fn to_owned(&self) -> Col<E::Canonical>
    where
        E: Conjugate,
    {
        self.as_ref().to_owned()
    }

    /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
    #[inline]
    pub fn has_nan(&self) -> bool
    where
        E: ComplexField,
    {
        self.as_ref().has_nan()
    }

    /// Returns `true` if all of the elements are finite, otherwise returns `false`.
    #[inline]
    pub fn is_all_finite(&self) -> bool
    where
        E: ComplexField,
    {
        self.as_ref().is_all_finite()
    }

    /// Returns the maximum norm of `self`.
    #[inline]
    pub fn norm_max(&self) -> E::Real
    where
        E: ComplexField,
    {
        norm_max((*self).as_ref().as_2d())
    }
    /// Returns the L2 norm of `self`.
    #[inline]
    pub fn norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        norm_l2((*self).as_ref().as_2d())
    }
}

impl<E: Entity> Row<E> {
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: DenseRowOwn {
                inner: VecOwnImpl {
                    ptr: into_copy::<E, _>(E::faer_map(E::UNIT, |()| {
                        NonNull::<E::Unit>::dangling()
                    })),
                    len: 0,
                },
                col_capacity: 0,
            },
        }
    }

    /// Returns a new column vector with 0 columns, with enough capacity to hold a maximum of
    /// `col_capacity` columnss columns without reallocating. If `col_capacity` is `0`,
    /// the function will not allocate.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn with_capacity(col_capacity: usize) -> Self {
        let raw = ManuallyDrop::new(RawMat::<E>::new(col_capacity, 1));
        Self {
            inner: DenseRowOwn {
                inner: VecOwnImpl {
                    ptr: raw.ptr,
                    len: 0,
                },
                col_capacity: raw.row_capacity,
            },
        }
    }

    /// Returns a new matrix with number of columns `ncols`, filled with the provided function.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn from_fn(ncols: usize, f: impl FnMut(usize) -> E) -> Self {
        let mut this = Self::new();
        this.resize_with(ncols, f);
        this
    }

    /// Returns a new matrix with number of columns `ncols`, filled with zeros.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn zeros(ncols: usize) -> Self
    where
        E: ComplexField,
    {
        Self::from_fn(ncols, |_| E::faer_zero())
    }

    #[inline(always)]
    pub fn nrows(&self) -> usize {
        1
    }
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.inner.inner.len
    }

    /// Set the dimensions of the matrix.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `ncols < self.col_capacity()`.
    /// * The elements that were previously out of bounds but are now in bounds must be
    /// initialized.
    #[inline]
    pub unsafe fn set_ncols(&mut self, ncols: usize) {
        self.inner.inner.len = ncols;
    }

    /// Returns a pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr(&self) -> GroupFor<E, *const E::Unit> {
        E::faer_map(from_copy::<E, _>(self.inner.inner.ptr), |ptr| {
            ptr.as_ptr() as *const E::Unit
        })
    }

    /// Returns a mutable pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr_mut(&mut self) -> GroupFor<E, *mut E::Unit> {
        E::faer_map(from_copy::<E, _>(self.inner.inner.ptr), |ptr| ptr.as_ptr())
    }

    /// Returns the col capacity, that is, the number of cols that the matrix is able to hold
    /// without needing to reallocate, excluding column insertions.
    #[inline]
    pub fn col_capacity(&self) -> usize {
        self.inner.col_capacity
    }

    /// Returns the offset between the first elements of two successive columns in the matrix.
    /// Always returns `1` since the matrix is column major.
    #[inline]
    pub fn col_stride(&self) -> isize {
        1
    }

    #[cold]
    fn do_reserve_exact(&mut self, mut new_col_capacity: usize) {
        if is_vectorizable::<E::Unit>() {
            let align_factor = align_for::<E::Unit>() / core::mem::size_of::<E::Unit>();
            new_col_capacity = new_col_capacity
                .msrv_checked_next_multiple_of(align_factor)
                .unwrap();
        }

        let ncols = self.inner.inner.len;
        let old_col_capacity = self.inner.col_capacity;

        let mut this = ManuallyDrop::new(core::mem::take(self));
        {
            let mut this_group =
                E::faer_map(from_copy::<E, _>(this.inner.inner.ptr), |ptr| MatUnit {
                    raw: RawMatUnit {
                        ptr,
                        row_capacity: old_col_capacity,
                        col_capacity: 1,
                    },
                    ncols,
                    nrows: 1,
                });

            E::faer_map(E::faer_as_mut(&mut this_group), |mat_unit| {
                mat_unit.do_reserve_exact(new_col_capacity, 1);
            });

            let this_group = E::faer_map(this_group, ManuallyDrop::new);
            this.inner.inner.ptr =
                into_copy::<E, _>(E::faer_map(this_group, |mat_unit| mat_unit.raw.ptr));
            this.inner.col_capacity = new_col_capacity;
        }
        *self = ManuallyDrop::into_inner(this);
    }

    /// Reserves the minimum capacity for `col_capacity` columns without reallocating. Does nothing
    /// if the capacity is already sufficient.
    ///
    /// # Panics
    /// The function panics if the new total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn reserve_exact(&mut self, col_capacity: usize) {
        if self.col_capacity() >= col_capacity {
            // do nothing
        } else if core::mem::size_of::<E::Unit>() == 0 {
            self.inner.col_capacity = self.col_capacity().max(col_capacity);
        } else {
            self.do_reserve_exact(col_capacity);
        }
    }

    unsafe fn insert_block_with<F: FnMut(usize) -> E>(
        &mut self,
        f: &mut F,
        col_start: usize,
        col_end: usize,
    ) {
        debug_assert!(col_start <= col_end);

        let ptr = self.as_ptr_mut();

        for j in col_start..col_end {
            // SAFETY:
            // * pointer to element at index (i, j), which is within the
            // allocation since we reserved enough space
            // * writing to this memory region is sound since it is properly
            // aligned and valid for writes
            let ptr_ij = E::faer_map(E::faer_copy(&ptr), |ptr| ptr.add(j));
            let value = E::faer_into_units(f(j));

            E::faer_map(E::faer_zip(ptr_ij, value), |(ptr_ij, value)| {
                core::ptr::write(ptr_ij, value)
            });
        }
    }

    fn erase_last_cols(&mut self, new_ncols: usize) {
        let old_ncols = self.ncols();
        debug_assert!(new_ncols <= old_ncols);
        self.inner.inner.len = new_ncols;
    }

    unsafe fn insert_last_cols_with<F: FnMut(usize) -> E>(&mut self, f: &mut F, new_ncols: usize) {
        let old_ncols = self.ncols();

        debug_assert!(new_ncols > old_ncols);

        self.insert_block_with(f, old_ncols, new_ncols);
        self.inner.inner.len = new_ncols;
    }

    /// Resizes the vector in-place so that the new number of columns is `new_ncols`.
    /// New elements are created with the given function `f`, so that elements at index `i`
    /// are created by calling `f(i)`.
    pub fn resize_with(&mut self, new_ncols: usize, f: impl FnMut(usize) -> E) {
        let mut f = f;
        let old_ncols = self.ncols();

        if new_ncols <= old_ncols {
            self.erase_last_cols(new_ncols);
        } else {
            self.reserve_exact(new_ncols);
            unsafe {
                self.insert_last_cols_with(&mut f, new_ncols);
            }
        }
    }

    /// Returns a reference to a slice over the row.
    #[inline]
    #[track_caller]
    pub fn as_slice(&self) -> GroupFor<E, &[E::Unit]> {
        let ncols = self.ncols();
        let ptr = self.as_ref().as_ptr();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, ncols) },
        )
    }

    /// Returns a mutable reference to a slice over the row.
    #[inline]
    #[track_caller]
    pub fn as_slice_mut(&mut self) -> GroupFor<E, &mut [E::Unit]> {
        let ncols = self.ncols();
        let ptr = self.as_ptr_mut();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, ncols) },
        )
    }

    /// Returns a view over the vector.
    #[inline]
    pub fn as_ref(&self) -> RowRef<'_, E> {
        unsafe { row::from_raw_parts(self.as_ptr(), self.ncols(), 1) }
    }

    /// Returns a mutable view over the vector.
    #[inline]
    pub fn as_mut(&mut self) -> RowMut<'_, E> {
        unsafe { row::from_raw_parts_mut(self.as_ptr_mut(), self.ncols(), 1) }
    }

    /// Returns references to the element at the given index, or submatrices if `col` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub unsafe fn get_unchecked<ColRange>(
        &self,
        col: ColRange,
    ) -> <RowRef<'_, E> as RowIndex<ColRange>>::Target
    where
        for<'a> RowRef<'a, E>: RowIndex<ColRange>,
    {
        self.as_ref().get_unchecked(col)
    }

    /// Returns references to the element at the given index, or submatrices if `col` is a range,
    /// with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub fn get<ColRange>(&self, col: ColRange) -> <RowRef<'_, E> as RowIndex<ColRange>>::Target
    where
        for<'a> RowRef<'a, E>: RowIndex<ColRange>,
    {
        self.as_ref().get(col)
    }

    /// Returns mutable references to the element at the given index, or submatrices if
    /// `col` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub unsafe fn get_mut_unchecked<ColRange>(
        &mut self,
        col: ColRange,
    ) -> <RowMut<'_, E> as RowIndex<ColRange>>::Target
    where
        for<'a> RowMut<'a, E>: RowIndex<ColRange>,
    {
        self.as_mut().get_mut_unchecked(col)
    }

    /// Returns mutable references to the element at the given index, or submatrices if
    /// `col` is a range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub fn get_mut<ColRange>(
        &mut self,
        col: ColRange,
    ) -> <RowMut<'_, E> as RowIndex<ColRange>>::Target
    where
        for<'a> RowMut<'a, E>: RowIndex<ColRange>,
    {
        self.as_mut().get_mut(col)
    }

    /// Reads the value of the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, col: usize) -> E {
        self.as_ref().read_unchecked(col)
    }

    /// Reads the value of the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, col: usize) -> E {
        self.as_ref().read(col)
    }

    /// Writes the value to the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, col: usize, value: E) {
        self.as_mut().write_unchecked(col, value);
    }

    /// Writes the value to the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, col: usize, value: E) {
        self.as_mut().write(col, value);
    }

    /// Copies the values from `other` into `self`.
    #[inline(always)]
    #[track_caller]
    pub fn copy_from(&mut self, other: impl AsRowRef<E>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity>(this: &mut Row<E>, other: RowRef<'_, E>) {
            let mut mat = Row::<E>::new();
            mat.resize_with(
                other.nrows(),
                #[inline(always)]
                |row| unsafe { other.read_unchecked(row) },
            );
            *this = mat;
        }
        implementation(self, other.as_row_ref());
    }

    /// Fills the elements of `self` with zeros.
    #[inline(always)]
    #[track_caller]
    pub fn fill_zero(&mut self)
    where
        E: ComplexField,
    {
        self.as_mut().fill_zero()
    }

    /// Fills the elements of `self` with copies of `constant`.
    #[inline(always)]
    #[track_caller]
    pub fn fill(&mut self, constant: E) {
        self.as_mut().fill(constant)
    }

    /// Returns a view over the transpose of `self`.
    #[inline]
    pub fn transpose(&self) -> ColRef<'_, E> {
        self.as_ref().transpose()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(&self) -> RowRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(&self) -> ColRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
    }

    /// Returns an owning [`Row`] of the data
    #[inline]
    pub fn to_owned(&self) -> Row<E::Canonical>
    where
        E: Conjugate,
    {
        self.as_ref().to_owned()
    }

    /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
    #[inline]
    pub fn has_nan(&self) -> bool
    where
        E: ComplexField,
    {
        self.as_ref().has_nan()
    }

    /// Returns `true` if all of the elements are finite, otherwise returns `false`.
    #[inline]
    pub fn is_all_finite(&self) -> bool
    where
        E: ComplexField,
    {
        self.as_ref().is_all_finite()
    }

    /// Returns the maximum norm of `self`.
    #[inline]
    pub fn norm_max(&self) -> E::Real
    where
        E: ComplexField,
    {
        norm_max((*self).as_ref().as_2d())
    }
    /// Returns the L2 norm of `self`.
    #[inline]
    pub fn norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        norm_l2((*self).as_ref().as_2d())
    }
}

impl<E: Entity> Mat<E> {
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: DenseOwn {
                inner: MatOwnImpl {
                    ptr: into_copy::<E, _>(E::faer_map(E::UNIT, |()| {
                        NonNull::<E::Unit>::dangling()
                    })),
                    nrows: 0,
                    ncols: 0,
                },
                row_capacity: 0,
                col_capacity: 0,
            },
        }
    }

    /// Returns a new matrix with dimensions `(0, 0)`, with enough capacity to hold a maximum of
    /// `row_capacity` rows and `col_capacity` columns without reallocating. If either is `0`,
    /// the matrix will not allocate.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn with_capacity(row_capacity: usize, col_capacity: usize) -> Self {
        let raw = ManuallyDrop::new(RawMat::<E>::new(row_capacity, col_capacity));
        Self {
            inner: DenseOwn {
                inner: MatOwnImpl {
                    ptr: raw.ptr,
                    nrows: 0,
                    ncols: 0,
                },
                row_capacity: raw.row_capacity,
                col_capacity: raw.col_capacity,
            },
        }
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with the provided function.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> E) -> Self {
        let mut this = Self::new();
        this.resize_with(nrows, ncols, f);
        this
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with zeros.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn zeros(nrows: usize, ncols: usize) -> Self
    where
        E: ComplexField,
    {
        Self::from_fn(nrows, ncols, |_, _| E::faer_zero())
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with zeros, except the main
    /// diagonal which is filled with ones.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn identity(nrows: usize, ncols: usize) -> Self
    where
        E: ComplexField,
    {
        let mut matrix = Self::zeros(nrows, ncols);
        matrix
            .as_mut()
            .diagonal_mut()
            .column_vector_mut()
            .fill(E::faer_one());
        matrix
    }

    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.inner.inner.nrows
    }
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.inner.inner.ncols
    }

    /// Set the dimensions of the matrix.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `nrows < self.row_capacity()`.
    /// * `ncols < self.col_capacity()`.
    /// * The elements that were previously out of bounds but are now in bounds must be
    /// initialized.
    #[inline]
    pub unsafe fn set_dims(&mut self, nrows: usize, ncols: usize) {
        self.inner.inner.nrows = nrows;
        self.inner.inner.ncols = ncols;
    }

    /// Returns a pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr(&self) -> GroupFor<E, *const E::Unit> {
        E::faer_map(from_copy::<E, _>(self.inner.inner.ptr), |ptr| {
            ptr.as_ptr() as *const E::Unit
        })
    }

    /// Returns a mutable pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr_mut(&mut self) -> GroupFor<E, *mut E::Unit> {
        E::faer_map(from_copy::<E, _>(self.inner.inner.ptr), |ptr| ptr.as_ptr())
    }

    /// Returns the row capacity, that is, the number of rows that the matrix is able to hold
    /// without needing to reallocate, excluding column insertions.
    #[inline]
    pub fn row_capacity(&self) -> usize {
        self.inner.row_capacity
    }

    /// Returns the column capacity, that is, the number of columns that the matrix is able to hold
    /// without needing to reallocate, excluding row insertions.
    #[inline]
    pub fn col_capacity(&self) -> usize {
        self.inner.col_capacity
    }

    /// Returns the offset between the first elements of two successive rows in the matrix.
    /// Always returns `1` since the matrix is column major.
    #[inline]
    pub fn row_stride(&self) -> isize {
        1
    }

    /// Returns the offset between the first elements of two successive columns in the matrix.
    #[inline]
    pub fn col_stride(&self) -> isize {
        self.row_capacity() as isize
    }

    #[cold]
    fn do_reserve_exact(&mut self, mut new_row_capacity: usize, new_col_capacity: usize) {
        if is_vectorizable::<E::Unit>() {
            let align_factor = align_for::<E::Unit>() / core::mem::size_of::<E::Unit>();
            new_row_capacity = new_row_capacity
                .msrv_checked_next_multiple_of(align_factor)
                .unwrap();
        }

        let nrows = self.inner.inner.nrows;
        let ncols = self.inner.inner.ncols;
        let old_row_capacity = self.inner.row_capacity;
        let old_col_capacity = self.inner.col_capacity;

        let mut this = ManuallyDrop::new(core::mem::take(self));
        {
            let mut this_group =
                E::faer_map(from_copy::<E, _>(this.inner.inner.ptr), |ptr| MatUnit {
                    raw: RawMatUnit {
                        ptr,
                        row_capacity: old_row_capacity,
                        col_capacity: old_col_capacity,
                    },
                    nrows,
                    ncols,
                });

            E::faer_map(E::faer_as_mut(&mut this_group), |mat_unit| {
                mat_unit.do_reserve_exact(new_row_capacity, new_col_capacity);
            });

            let this_group = E::faer_map(this_group, ManuallyDrop::new);
            this.inner.inner.ptr =
                into_copy::<E, _>(E::faer_map(this_group, |mat_unit| mat_unit.raw.ptr));
            this.inner.row_capacity = new_row_capacity;
            this.inner.col_capacity = new_col_capacity;
        }
        *self = ManuallyDrop::into_inner(this);
    }

    /// Reserves the minimum capacity for `row_capacity` rows and `col_capacity`
    /// columns without reallocating. Does nothing if the capacity is already sufficient.
    ///
    /// # Panics
    /// The function panics if the new total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn reserve_exact(&mut self, row_capacity: usize, col_capacity: usize) {
        if self.row_capacity() >= row_capacity && self.col_capacity() >= col_capacity {
            // do nothing
        } else if core::mem::size_of::<E::Unit>() == 0 {
            self.inner.row_capacity = self.row_capacity().max(row_capacity);
            self.inner.col_capacity = self.col_capacity().max(col_capacity);
        } else {
            self.do_reserve_exact(row_capacity, col_capacity);
        }
    }

    unsafe fn insert_block_with<F: FnMut(usize, usize) -> E>(
        &mut self,
        f: &mut F,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) {
        debug_assert!(all(row_start <= row_end, col_start <= col_end));

        let ptr = self.as_ptr_mut();

        for j in col_start..col_end {
            let ptr_j = E::faer_map(E::faer_copy(&ptr), |ptr| {
                ptr.wrapping_offset(j as isize * self.col_stride())
            });

            for i in row_start..row_end {
                // SAFETY:
                // * pointer to element at index (i, j), which is within the
                // allocation since we reserved enough space
                // * writing to this memory region is sound since it is properly
                // aligned and valid for writes
                let ptr_ij = E::faer_map(E::faer_copy(&ptr_j), |ptr_j| ptr_j.add(i));
                let value = E::faer_into_units(f(i, j));

                E::faer_map(E::faer_zip(ptr_ij, value), |(ptr_ij, value)| {
                    core::ptr::write(ptr_ij, value)
                });
            }
        }
    }

    fn erase_last_cols(&mut self, new_ncols: usize) {
        let old_ncols = self.ncols();
        debug_assert!(new_ncols <= old_ncols);
        self.inner.inner.ncols = new_ncols;
    }

    fn erase_last_rows(&mut self, new_nrows: usize) {
        let old_nrows = self.nrows();
        debug_assert!(new_nrows <= old_nrows);
        self.inner.inner.nrows = new_nrows;
    }

    unsafe fn insert_last_cols_with<F: FnMut(usize, usize) -> E>(
        &mut self,
        f: &mut F,
        new_ncols: usize,
    ) {
        let old_ncols = self.ncols();

        debug_assert!(new_ncols > old_ncols);

        self.insert_block_with(f, 0, self.nrows(), old_ncols, new_ncols);
        self.inner.inner.ncols = new_ncols;
    }

    unsafe fn insert_last_rows_with<F: FnMut(usize, usize) -> E>(
        &mut self,
        f: &mut F,
        new_nrows: usize,
    ) {
        let old_nrows = self.nrows();

        debug_assert!(new_nrows > old_nrows);

        self.insert_block_with(f, old_nrows, new_nrows, 0, self.ncols());
        self.inner.inner.nrows = new_nrows;
    }

    /// Resizes the matrix in-place so that the new dimensions are `(new_nrows, new_ncols)`.
    /// New elements are created with the given function `f`, so that elements at indices `(i, j)`
    /// are created by calling `f(i, j)`.
    pub fn resize_with(
        &mut self,
        new_nrows: usize,
        new_ncols: usize,
        f: impl FnMut(usize, usize) -> E,
    ) {
        let mut f = f;
        let old_nrows = self.nrows();
        let old_ncols = self.ncols();

        if new_ncols <= old_ncols {
            self.erase_last_cols(new_ncols);
            if new_nrows <= old_nrows {
                self.erase_last_rows(new_nrows);
            } else {
                self.reserve_exact(new_nrows, new_ncols);
                unsafe {
                    self.insert_last_rows_with(&mut f, new_nrows);
                }
            }
        } else {
            if new_nrows <= old_nrows {
                self.erase_last_rows(new_nrows);
            } else {
                self.reserve_exact(new_nrows, new_ncols);
                unsafe {
                    self.insert_last_rows_with(&mut f, new_nrows);
                }
            }
            self.reserve_exact(new_nrows, new_ncols);
            unsafe {
                self.insert_last_cols_with(&mut f, new_ncols);
            }
        }
    }

    /// Returns a reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    pub fn col_as_slice(&self, col: usize) -> GroupFor<E, &[E::Unit]> {
        assert!(col < self.ncols());
        let nrows = self.nrows();
        let ptr = self.as_ref().ptr_at(0, col);
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, nrows) },
        )
    }

    /// Returns a mutable reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    pub fn col_as_slice_mut(&mut self, col: usize) -> GroupFor<E, &mut [E::Unit]> {
        assert!(col < self.ncols());
        let nrows = self.nrows();
        let ptr = self.as_mut().ptr_at_mut(0, col);
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, nrows) },
        )
    }

    /// Returns a reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    #[deprecated = "replaced by `Mat::col_as_slice`"]
    pub fn col_ref(&self, col: usize) -> GroupFor<E, &[E::Unit]> {
        self.col_as_slice(col)
    }

    /// Returns a mutable reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    #[deprecated = "replaced by `Mat::col_as_slice_mut`"]
    pub fn col_mut(&mut self, col: usize) -> GroupFor<E, &mut [E::Unit]> {
        self.col_as_slice_mut(col)
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, E> {
        unsafe {
            mat::from_raw_parts(
                self.as_ptr(),
                self.nrows(),
                self.ncols(),
                1,
                self.col_stride(),
            )
        }
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> MatMut<'_, E> {
        unsafe {
            mat::from_raw_parts_mut(
                self.as_ptr_mut(),
                self.nrows(),
                self.ncols(),
                1,
                self.col_stride(),
            )
        }
    }

    /// Returns references to the element at the given indices, or submatrices if either `row` or
    /// `col` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub unsafe fn get_unchecked<RowRange, ColRange>(
        &self,
        row: RowRange,
        col: ColRange,
    ) -> <MatRef<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatRef<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_ref().get_unchecked(row, col)
    }

    /// Returns references to the element at the given indices, or submatrices if either `row` or
    /// `col` is a range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub fn get<RowRange, ColRange>(
        &self,
        row: RowRange,
        col: ColRange,
    ) -> <MatRef<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatRef<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_ref().get(row, col)
    }

    /// Returns mutable references to the element at the given indices, or submatrices if either
    /// `row` or `col` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub unsafe fn get_mut_unchecked<RowRange, ColRange>(
        &mut self,
        row: RowRange,
        col: ColRange,
    ) -> <MatMut<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatMut<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_mut().get_mut_unchecked(row, col)
    }

    /// Returns mutable references to the element at the given indices, or submatrices if either
    /// `row` or `col` is a range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub fn get_mut<RowRange, ColRange>(
        &mut self,
        row: RowRange,
        col: ColRange,
    ) -> <MatMut<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatMut<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_mut().get_mut(row, col)
    }

    /// Reads the value of the element at the given indices.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, row: usize, col: usize) -> E {
        self.as_ref().read_unchecked(row, col)
    }

    /// Reads the value of the element at the given indices, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, row: usize, col: usize) -> E {
        self.as_ref().read(row, col)
    }

    /// Writes the value to the element at the given indices.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, row: usize, col: usize, value: E) {
        self.as_mut().write_unchecked(row, col, value);
    }

    /// Writes the value to the element at the given indices, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, row: usize, col: usize, value: E) {
        self.as_mut().write(row, col, value);
    }

    /// Copies the values from `other` into `self`.
    #[inline(always)]
    #[track_caller]
    pub fn copy_from(&mut self, other: impl AsMatRef<E>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity>(this: &mut Mat<E>, other: MatRef<'_, E>) {
            let mut mat = Mat::<E>::new();
            mat.resize_with(
                other.nrows(),
                other.ncols(),
                #[inline(always)]
                |row, col| unsafe { other.read_unchecked(row, col) },
            );
            *this = mat;
        }
        implementation(self, other.as_mat_ref());
    }

    /// Fills the elements of `self` with zeros.
    #[inline(always)]
    #[track_caller]
    pub fn fill_zero(&mut self)
    where
        E: ComplexField,
    {
        self.as_mut().fill_zero()
    }

    /// Fills the elements of `self` with copies of `constant`.
    #[inline(always)]
    #[track_caller]
    pub fn fill(&mut self, constant: E) {
        self.as_mut().fill(constant)
    }

    /// Returns a view over the transpose of `self`.
    #[inline]
    pub fn transpose(&self) -> MatRef<'_, E> {
        self.as_ref().transpose()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(&self) -> MatRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(&self) -> MatRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
    }

    #[inline]
    pub fn diagonal(&self) -> Matrix<DiagRef<'_, E>> {
        self.as_ref().diagonal()
    }

    /// Returns an owning [`Mat`] of the data
    #[inline]
    pub fn to_owned(&self) -> Mat<E::Canonical>
    where
        E: Conjugate,
    {
        self.as_ref().to_owned()
    }

    /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
    #[inline]
    pub fn has_nan(&self) -> bool
    where
        E: ComplexField,
    {
        self.as_ref().has_nan()
    }

    /// Returns `true` if all of the elements are finite, otherwise returns `false`.
    #[inline]
    pub fn is_all_finite(&self) -> bool
    where
        E: ComplexField,
    {
        self.as_ref().is_all_finite()
    }

    /// Returns the maximum norm of `self`.
    #[inline]
    pub fn norm_max(&self) -> E::Real
    where
        E: ComplexField,
    {
        norm_max((*self).as_ref())
    }
    /// Returns the L2 norm of `self`.
    #[inline]
    pub fn norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        norm_l2((*self).as_ref())
    }

    /// Returns an iterator that provides successive chunks of the columns of a view over this
    /// matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    #[inline]
    #[track_caller]
    pub fn col_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatRef<'_, E>> {
        self.as_ref().col_chunks(chunk_size)
    }

    /// Returns an iterator that provides successive chunks of the columns of a mutable view over
    /// this matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    #[inline]
    #[track_caller]
    pub fn col_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatMut<'_, E>> {
        self.as_mut().col_chunks_mut(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the columns of a view over
    /// this matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_col_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, E>> {
        self.as_ref().par_col_chunks(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the columns of a mutable view
    /// over this matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_col_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, E>> {
        self.as_mut().par_col_chunks_mut(chunk_size)
    }

    /// Returns an iterator that provides successive chunks of the rows of a view over this
    /// matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn row_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatRef<'_, E>> {
        self.as_ref().row_chunks(chunk_size)
    }

    /// Returns an iterator that provides successive chunks of the rows of a mutable view over
    /// this matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn row_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatMut<'_, E>> {
        self.as_mut().row_chunks_mut(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the rows of a view over this
    /// matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_row_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, E>> {
        self.as_ref().par_row_chunks(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the rows of a mutable view
    /// over this matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_row_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, E>> {
        self.as_mut().par_row_chunks_mut(chunk_size)
    }
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
/// use faer_core::mat;
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
                $crate::Mat::<_>::from_fn(nrows, ncols, |i, j| $crate::ref_to_ptr(&data[j][i]).read())
            }
        }
    };
}

/// Parallelism strategy that can be passed to most of the routines in the library.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Parallelism {
    /// No parallelism.
    ///
    /// The code is executed sequentially on the same thread that calls a function
    /// and passes this argument.
    None,
    /// Rayon parallelism. Only avaialble with the `rayon` feature.
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
    };
    GLOBAL_PARALLELISM.store(value, core::sync::atomic::Ordering::Relaxed);
}

/// Gets the global parallelism settings.
///
/// # Panics
/// Panics if global parallelism is disabled.
#[track_caller]
pub fn get_global_parallelism() -> Parallelism {
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

#[inline]
#[doc(hidden)]
pub fn join_raw(
    op_a: impl Send + FnOnce(Parallelism),
    op_b: impl Send + FnOnce(Parallelism),
    parallelism: Parallelism,
) {
    fn implementation(
        op_a: &mut (dyn Send + FnMut(Parallelism)),
        op_b: &mut (dyn Send + FnMut(Parallelism)),
        parallelism: Parallelism,
    ) {
        match parallelism {
            Parallelism::None => (op_a(parallelism), op_b(parallelism)),
            #[cfg(feature = "rayon")]
            Parallelism::Rayon(n_threads) => {
                if n_threads == 1 {
                    (op_a(Parallelism::None), op_b(Parallelism::None))
                } else {
                    let n_threads = if n_threads > 0 {
                        n_threads
                    } else {
                        rayon::current_num_threads()
                    };
                    let parallelism = Parallelism::Rayon(n_threads - n_threads / 2);
                    rayon::join(|| op_a(parallelism), || op_b(parallelism))
                }
            }
        };
    }
    let mut op_a = Some(op_a);
    let mut op_b = Some(op_b);
    implementation(
        &mut |parallelism| (op_a.take().unwrap())(parallelism),
        &mut |parallelism| (op_b.take().unwrap())(parallelism),
        parallelism,
    )
}

#[inline]
#[doc(hidden)]
pub fn for_each_raw(n_tasks: usize, op: impl Send + Sync + Fn(usize), parallelism: Parallelism) {
    fn implementation(
        n_tasks: usize,
        op: &(dyn Send + Sync + Fn(usize)),
        parallelism: Parallelism,
    ) {
        if n_tasks == 1 {
            op(0);
            return;
        }

        match parallelism {
            Parallelism::None => (0..n_tasks).for_each(op),
            #[cfg(feature = "rayon")]
            Parallelism::Rayon(n_threads) => {
                let n_threads = if n_threads > 0 {
                    n_threads
                } else {
                    rayon::current_num_threads()
                };

                use rayon::prelude::*;
                let min_len = n_tasks / n_threads;
                (0..n_tasks)
                    .into_par_iter()
                    .with_min_len(min_len)
                    .for_each(op);
            }
        }
    }
    implementation(n_tasks, &op, parallelism);
}

#[doc(hidden)]
pub struct Ptr<T>(pub *mut T);
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}
impl<T> Copy for Ptr<T> {}
impl<T> Clone for Ptr<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

#[inline]
#[doc(hidden)]
pub fn parallelism_degree(parallelism: Parallelism) -> usize {
    match parallelism {
        Parallelism::None => 1,
        #[cfg(feature = "rayon")]
        Parallelism::Rayon(0) => rayon::current_num_threads(),
        #[cfg(feature = "rayon")]
        Parallelism::Rayon(n_threads) => n_threads,
    }
}

/// Creates a temporary matrix of constant values, from the given memory stack.
pub fn temp_mat_constant<E: ComplexField>(
    nrows: usize,
    ncols: usize,
    value: E,
    stack: PodStack<'_>,
) -> (MatMut<'_, E>, PodStack<'_>) {
    let (mut mat, stack) = temp_mat_uninit::<E>(nrows, ncols, stack);
    mat.as_mut().fill(value);
    (mat, stack)
}

/// Creates a temporary matrix of zero values, from the given memory stack.
pub fn temp_mat_zeroed<E: ComplexField>(
    nrows: usize,
    ncols: usize,
    stack: PodStack<'_>,
) -> (MatMut<'_, E>, PodStack<'_>) {
    let (mut mat, stack) = temp_mat_uninit::<E>(nrows, ncols, stack);
    mat.as_mut().fill_zero();
    (mat, stack)
}

/// Creates a temporary matrix of untouched values, from the given memory stack.
pub fn temp_mat_uninit<E: ComplexField>(
    nrows: usize,
    ncols: usize,
    stack: PodStack<'_>,
) -> (MatMut<'_, E>, PodStack<'_>) {
    let col_stride = col_stride::<E::Unit>(nrows);
    let alloc_size = ncols.checked_mul(col_stride).unwrap();

    let (stack, alloc) = E::faer_map_with_context(stack, E::UNIT, &mut {
        #[inline(always)]
        |stack, ()| {
            let (alloc, stack) =
                stack.make_aligned_raw::<E::Unit>(alloc_size, align_for::<E::Unit>());
            (stack, alloc)
        }
    });
    (
        unsafe {
            mat::from_raw_parts_mut(
                E::faer_map(alloc, |alloc| alloc.as_mut_ptr()),
                nrows,
                ncols,
                1,
                col_stride as isize,
            )
        },
        stack,
    )
}

#[doc(hidden)]
#[inline]
pub fn col_stride<Unit: 'static>(nrows: usize) -> usize {
    if !is_vectorizable::<Unit>() || nrows >= isize::MAX as usize {
        nrows
    } else {
        nrows
            .msrv_checked_next_multiple_of(align_for::<Unit>() / core::mem::size_of::<Unit>())
            .unwrap()
    }
}

/// Returns the stack requirements for creating a temporary matrix with the given dimensions.
#[inline]
pub fn temp_mat_req<E: Entity>(nrows: usize, ncols: usize) -> Result<StackReq, SizeOverflow> {
    let col_stride = col_stride::<E::Unit>(nrows);
    let alloc_size = ncols.checked_mul(col_stride).ok_or(SizeOverflow)?;
    let additional = StackReq::try_new_aligned::<E::Unit>(alloc_size, align_for::<E::Unit>())?;

    let req = Ok(StackReq::empty());
    let (req, _) = E::faer_map_with_context(req, E::UNIT, &mut {
        #[inline(always)]
        |req, ()| {
            let req = match req {
                Ok(req) => req.try_and(additional),
                _ => Err(SizeOverflow),
            };

            (req, ())
        }
    });

    req
}

impl<'a, FromE: Entity, ToE: Entity> Coerce<MatRef<'a, ToE>> for MatRef<'a, FromE> {
    #[inline(always)]
    fn coerce(self) -> MatRef<'a, ToE> {
        assert!(coe::is_same::<FromE, ToE>());
        unsafe { transmute_unchecked::<MatRef<'a, FromE>, MatRef<'a, ToE>>(self) }
    }
}
impl<'a, FromE: Entity, ToE: Entity> Coerce<MatMut<'a, ToE>> for MatMut<'a, FromE> {
    #[inline(always)]
    fn coerce(self) -> MatMut<'a, ToE> {
        assert!(coe::is_same::<FromE, ToE>());
        unsafe { transmute_unchecked::<MatMut<'a, FromE>, MatMut<'a, ToE>>(self) }
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
/// use faer_core::{mat, unzipped, zipped, Mat};
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
        $crate::zip::LastEq($crate::zip::ViewMut::view_mut(&mut { $head }))
    };

    ($head: expr, $($tail: expr),* $(,)?) => {
        $crate::zip::ZipEq::new($crate::zip::ViewMut::view_mut(&mut { $head }), $crate::zipped!($($tail,)*))
    };
}

#[macro_export]
macro_rules! unzipped {
    ($head: pat $(,)?) => {
        $crate::zip::Last($head)
    };

    ($head: pat, $($tail: pat),* $(,)?) => {
        $crate::zip::Zip($head, $crate::unzipped!($($tail,)*))
    };
}

impl<'a, E: Entity> Debug for RowRef<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_2d().fmt(f)
    }
}
impl<'a, E: Entity> Debug for RowMut<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<'a, E: Entity> Debug for ColRef<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_2d().fmt(f)
    }
}
impl<'a, E: Entity> Debug for ColMut<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<'a, E: Entity> Debug for MatRef<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        struct DebugRow<'a, T: Entity>(MatRef<'a, T>);

        impl<'a, T: Entity> Debug for DebugRow<'a, T> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let mut j = 0;
                f.debug_list()
                    .entries(core::iter::from_fn(|| {
                        let ret = if j < self.0.ncols() {
                            Some(T::faer_from_units(T::faer_deref(self.0.get(0, j))))
                        } else {
                            None
                        };
                        j += 1;
                        ret
                    }))
                    .finish()
            }
        }

        writeln!(f, "[")?;
        for i in 0..self.nrows() {
            let row = self.subrows(i, 1);
            DebugRow(row).fmt(f)?;
            f.write_str(",\n")?;
        }
        write!(f, "]")
    }
}

impl<'a, E: Entity> Debug for MatMut<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<E: Entity> Debug for Mat<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

/// Advanced: Module for index and matrix types with compile time checks, instead of bound checking
/// at runtime.
pub mod constrained {
    use core::ops::Range;

    use super::*;
    use crate::{
        assert, debug_assert,
        permutation::{Index, SignedIndex},
    };

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    struct Branded<'a, T: ?Sized> {
        __marker: PhantomData<fn(&'a ()) -> &'a ()>,
        inner: T,
    }

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    pub struct Size<'n>(Branded<'n, usize>);

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    pub struct Idx<'n, I>(Branded<'n, I>);

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    pub struct IdxInclusive<'n, I>(Branded<'n, I>);

    #[derive(Copy, Clone, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct MaybeIdx<'n, I: Index>(Branded<'n, I>);

    impl core::ops::Deref for Size<'_> {
        type Target = usize;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.0.inner
        }
    }
    impl<I> core::ops::Deref for Idx<'_, I> {
        type Target = I;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.0.inner
        }
    }
    impl<I: Index> core::ops::Deref for MaybeIdx<'_, I> {
        type Target = I::Signed;
        #[inline]
        fn deref(&self) -> &Self::Target {
            bytemuck::cast_ref(&self.0.inner)
        }
    }
    impl<I> core::ops::Deref for IdxInclusive<'_, I> {
        type Target = I;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.0.inner
        }
    }

    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    pub struct Array<'n, T>(Branded<'n, [T]>);

    #[repr(transparent)]
    pub struct MatRef<'nrows, 'ncols, 'a, E: Entity>(
        Branded<'ncols, Branded<'nrows, super::MatRef<'a, E>>>,
    );
    #[repr(transparent)]
    pub struct MatMut<'nrows, 'ncols, 'a, E: Entity>(
        Branded<'ncols, Branded<'nrows, super::MatMut<'a, E>>>,
    );

    pub mod permutation {
        use super::*;
        use crate::assert;

        #[repr(transparent)]
        pub struct PermutationRef<'n, 'a, I, E: Entity>(
            Branded<'n, crate::permutation::PermutationRef<'a, I, E>>,
        );

        impl<'n, 'a, I: Index, E: Entity> PermutationRef<'n, 'a, I, E> {
            #[inline]
            #[track_caller]
            pub fn new(perm: crate::permutation::PermutationRef<'a, I, E>, size: Size<'n>) -> Self {
                let (fwd, inv) = perm.into_arrays();
                assert!(all(
                    fwd.len() == size.into_inner(),
                    inv.len() == size.into_inner(),
                ));
                Self(Branded {
                    __marker: PhantomData,
                    inner: perm,
                })
            }

            #[inline]
            pub fn inverse(self) -> PermutationRef<'n, 'a, I, E> {
                PermutationRef(Branded {
                    __marker: PhantomData,
                    inner: self.0.inner.inverse(),
                })
            }

            #[inline]
            pub fn into_arrays(self) -> (&'a Array<'n, Idx<'n, I>>, &'a Array<'n, Idx<'n, I>>) {
                unsafe {
                    let (fwd, inv) = self.0.inner.into_arrays();
                    let fwd = &*(fwd as *const [I] as *const Array<'n, Idx<'n, I>>);
                    let inv = &*(inv as *const [I] as *const Array<'n, Idx<'n, I>>);
                    (fwd, inv)
                }
            }

            #[inline]
            pub fn into_inner(self) -> crate::permutation::PermutationRef<'a, I, E> {
                self.0.inner
            }

            #[inline]
            pub fn len(&self) -> Size<'n> {
                unsafe { Size::new_raw_unchecked(self.into_inner().len()) }
            }

            pub fn cast<T: Entity>(self) -> PermutationRef<'n, 'a, I, T> {
                PermutationRef(Branded {
                    __marker: PhantomData,
                    inner: self.into_inner().cast(),
                })
            }
        }

        impl<I, E: Entity> Clone for PermutationRef<'_, '_, I, E> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<I, E: Entity> Copy for PermutationRef<'_, '_, I, E> {}

        impl<I: Debug, E: Entity> Debug for PermutationRef<'_, '_, I, E> {
            #[inline]
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                self.0.inner.fmt(f)
            }
        }
    }

    pub mod sparse {
        use super::*;
        use crate::{assert, group_helpers::SliceGroup, sparse::__get_unchecked};
        use core::ops::Range;

        #[repr(transparent)]
        pub struct SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I>(
            Branded<'ncols, Branded<'nrows, crate::sparse::SymbolicSparseColMatRef<'a, I>>>,
        );
        pub struct SparseColMatRef<'nrows, 'ncols, 'a, I, E: Entity> {
            symbolic: SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I>,
            values: SliceGroup<'a, E>,
        }

        impl<'nrows, 'ncols, 'a, I: Index> SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I> {
            #[inline]
            pub fn new(
                inner: crate::sparse::SymbolicSparseColMatRef<'a, I>,
                nrows: Size<'nrows>,
                ncols: Size<'ncols>,
            ) -> Self {
                assert!(all(
                    inner.nrows() == nrows.into_inner(),
                    inner.ncols() == ncols.into_inner(),
                ));
                Self(Branded {
                    __marker: PhantomData,
                    inner: Branded {
                        __marker: PhantomData,
                        inner,
                    },
                })
            }

            #[inline]
            pub fn into_inner(self) -> crate::sparse::SymbolicSparseColMatRef<'a, I> {
                self.0.inner.inner
            }

            #[inline]
            pub fn nrows(&self) -> Size<'nrows> {
                unsafe { Size::new_raw_unchecked(self.0.inner.inner.nrows()) }
            }

            #[inline]
            pub fn ncols(&self) -> Size<'ncols> {
                unsafe { Size::new_raw_unchecked(self.0.inner.inner.ncols()) }
            }

            #[inline]
            #[track_caller]
            #[doc(hidden)]
            pub fn col_range(&self, j: Idx<'ncols, usize>) -> Range<usize> {
                unsafe { self.into_inner().col_range_unchecked(j.into_inner()) }
            }

            #[inline]
            #[track_caller]
            pub fn row_indices_of_col_raw(&self, j: Idx<'ncols, usize>) -> &'a [Idx<'nrows, I>] {
                unsafe {
                    &*(__get_unchecked(self.into_inner().row_indices(), self.col_range(j))
                        as *const [I] as *const [Idx<'_, I>])
                }
            }

            #[inline]
            #[track_caller]
            pub fn row_indices_of_col(
                &self,
                j: Idx<'ncols, usize>,
            ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = Idx<'nrows, usize>>
            {
                unsafe {
                    __get_unchecked(
                        self.into_inner().row_indices(),
                        self.into_inner().col_range_unchecked(j.into_inner()),
                    )
                    .iter()
                    .map(
                        #[inline(always)]
                        move |&row| Idx::new_raw_unchecked(row.zx()),
                    )
                }
            }
        }

        impl<'nrows, 'ncols, 'a, I: Index, E: Entity> SparseColMatRef<'nrows, 'ncols, 'a, I, E> {
            pub fn new(
                inner: crate::sparse::SparseColMatRef<'a, I, E>,
                nrows: Size<'nrows>,
                ncols: Size<'ncols>,
            ) -> Self {
                assert!(all(
                    inner.nrows() == nrows.into_inner(),
                    inner.ncols() == ncols.into_inner(),
                ));
                Self {
                    symbolic: SymbolicSparseColMatRef::new(inner.symbolic(), nrows, ncols),
                    values: SliceGroup::new(inner.values()),
                }
            }

            #[inline]
            pub fn into_inner(self) -> crate::sparse::SparseColMatRef<'a, I, E> {
                crate::sparse::SparseColMatRef::new(
                    self.symbolic.into_inner(),
                    self.values.into_inner(),
                )
            }

            #[inline]
            pub fn values_of_col(&self, j: Idx<'ncols, usize>) -> GroupFor<E, &'a [E::Unit]> {
                unsafe {
                    self.values
                        .subslice_unchecked(self.col_range(j))
                        .into_inner()
                }
            }
        }

        impl<I, E: Entity> Copy for SparseColMatRef<'_, '_, '_, I, E> {}
        impl<I, E: Entity> Clone for SparseColMatRef<'_, '_, '_, I, E> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<I> Copy for SymbolicSparseColMatRef<'_, '_, '_, I> {}
        impl<I> Clone for SymbolicSparseColMatRef<'_, '_, '_, I> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<'nrows, 'ncols, 'a, I, E: Entity> core::ops::Deref
            for SparseColMatRef<'nrows, 'ncols, 'a, I, E>
        {
            type Target = SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I>;

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.symbolic
            }
        }
    }

    pub mod group_helpers {
        use super::*;
        use crate::{
            assert,
            group_helpers::{SliceGroup, SliceGroupMut},
        };
        use core::ops::Range;

        pub struct ArrayGroup<'n, 'a, E: Entity>(Branded<'n, SliceGroup<'a, E>>);
        pub struct ArrayGroupMut<'n, 'a, E: Entity>(Branded<'n, SliceGroupMut<'a, E>>);

        impl<E: Entity> Debug for ArrayGroup<'_, '_, E> {
            #[inline]
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                self.0.inner.fmt(f)
            }
        }
        impl<E: Entity> Debug for ArrayGroupMut<'_, '_, E> {
            #[inline]
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                self.0.inner.fmt(f)
            }
        }

        impl<E: Entity> Copy for ArrayGroup<'_, '_, E> {}
        impl<E: Entity> Clone for ArrayGroup<'_, '_, E> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<'short, 'n, 'a, E: Entity> reborrow::ReborrowMut<'short> for ArrayGroup<'n, 'a, E> {
            type Target = ArrayGroup<'n, 'short, E>;

            #[inline]
            fn rb_mut(&'short mut self) -> Self::Target {
                *self
            }
        }

        impl<'short, 'n, 'a, E: Entity> reborrow::Reborrow<'short> for ArrayGroup<'n, 'a, E> {
            type Target = ArrayGroup<'n, 'short, E>;

            #[inline]
            fn rb(&'short self) -> Self::Target {
                *self
            }
        }

        impl<'short, 'n, 'a, E: Entity> reborrow::ReborrowMut<'short> for ArrayGroupMut<'n, 'a, E> {
            type Target = ArrayGroupMut<'n, 'short, E>;

            #[inline]
            fn rb_mut(&'short mut self) -> Self::Target {
                ArrayGroupMut(Branded {
                    __marker: PhantomData,
                    inner: self.0.inner.rb_mut(),
                })
            }
        }

        impl<'short, 'n, 'a, E: Entity> reborrow::Reborrow<'short> for ArrayGroupMut<'n, 'a, E> {
            type Target = ArrayGroup<'n, 'short, E>;

            #[inline]
            fn rb(&'short self) -> Self::Target {
                ArrayGroup(Branded {
                    __marker: PhantomData,
                    inner: self.0.inner.rb(),
                })
            }
        }

        impl<'n, 'a, E: Entity> ArrayGroupMut<'n, 'a, E> {
            #[inline]
            pub fn new(slice: GroupFor<E, &'a mut [E::Unit]>, len: Size<'n>) -> Self {
                let slice = SliceGroupMut::<'_, E>::new(slice);
                assert!(slice.rb().len() == len.into_inner());
                ArrayGroupMut(Branded {
                    __marker: PhantomData,
                    inner: slice,
                })
            }

            #[inline]
            pub fn into_slice(self) -> GroupFor<E, &'a mut [E::Unit]> {
                self.0.inner.into_inner()
            }

            #[inline]
            pub fn subslice(
                self,
                range: Range<IdxInclusive<'n, usize>>,
            ) -> GroupFor<E, &'a mut [E::Unit]> {
                unsafe {
                    SliceGroupMut::<'_, E>::new(self.into_slice())
                        .subslice_unchecked(range.start.into_inner()..range.end.into_inner())
                        .into_inner()
                }
            }

            #[inline]
            pub fn read(&self, j: Idx<'n, usize>) -> E {
                self.rb().read(j)
            }

            #[inline]
            pub fn write(&mut self, j: Idx<'n, usize>, value: E) {
                unsafe {
                    SliceGroupMut::new(self.rb_mut().into_slice())
                        .write_unchecked(j.into_inner(), value)
                }
            }
        }

        impl<'n, 'a, E: Entity> ArrayGroup<'n, 'a, E> {
            #[inline]
            pub fn new(slice: GroupFor<E, &'a [E::Unit]>, len: Size<'n>) -> Self {
                let slice = SliceGroup::<'_, E>::new(slice);
                assert!(slice.rb().len() == len.into_inner());
                ArrayGroup(Branded {
                    __marker: PhantomData,
                    inner: slice,
                })
            }

            #[inline]
            pub fn into_slice(self) -> GroupFor<E, &'a [E::Unit]> {
                self.0.inner.into_inner()
            }

            #[inline]
            pub fn subslice(
                self,
                range: Range<IdxInclusive<'n, usize>>,
            ) -> GroupFor<E, &'a [E::Unit]> {
                unsafe {
                    SliceGroup::<'_, E>::new(self.into_slice())
                        .subslice_unchecked(range.start.into_inner()..range.end.into_inner())
                        .into_inner()
                }
            }

            #[inline]
            pub fn read(&self, j: Idx<'n, usize>) -> E {
                unsafe { SliceGroup::new(self.into_slice()).read_unchecked(j.into_inner()) }
            }
        }
    }

    impl<'size> Size<'size> {
        #[track_caller]
        #[inline]
        pub fn with<R>(n: usize, f: impl for<'n> FnOnce(Size<'n>) -> R) -> R {
            f(Size(Branded {
                __marker: PhantomData,
                inner: n,
            }))
        }

        #[track_caller]
        #[inline]
        pub fn with2<R>(
            m: usize,
            n: usize,
            f: impl for<'m, 'n> FnOnce(Size<'m>, Size<'n>) -> R,
        ) -> R {
            f(
                Size(Branded {
                    __marker: PhantomData,
                    inner: m,
                }),
                Size(Branded {
                    __marker: PhantomData,
                    inner: n,
                }),
            )
        }

        #[inline]
        pub unsafe fn new_raw_unchecked(n: usize) -> Self {
            Size(Branded {
                __marker: PhantomData,
                inner: n,
            })
        }

        #[inline]
        pub fn into_inner(self) -> usize {
            self.0.inner
        }

        #[inline]
        pub fn indices(self) -> impl DoubleEndedIterator<Item = Idx<'size, usize>> {
            (0..self.0.inner).map(|i| unsafe { Idx::new_raw_unchecked(i) })
        }

        #[inline]
        pub fn check<I: Index>(self, idx: I) -> Idx<'size, I> {
            Idx::new_checked(idx, self)
        }

        #[inline]
        pub fn try_check<I: Index>(self, idx: I) -> Option<Idx<'size, I>> {
            if idx.zx() < self.into_inner() {
                Some(Idx(Branded {
                    __marker: PhantomData,
                    inner: idx,
                }))
            } else {
                None
            }
        }
    }

    impl<'n> Idx<'n, usize> {
        pub fn truncate<I: Index>(self) -> Idx<'n, I> {
            unsafe { Idx::new_raw_unchecked(I::truncate(self.into_inner())) }
        }
    }

    impl<'n, I: Index> Idx<'n, I> {
        #[inline]
        pub fn new_checked(idx: I, size: Size<'n>) -> Self {
            assert!(idx.zx() < size.into_inner());
            Self(Branded {
                __marker: PhantomData,
                inner: idx,
            })
        }
        #[inline]
        pub unsafe fn new_unchecked(idx: I, size: Size<'n>) -> Self {
            debug_assert!(idx.zx() < size.into_inner());
            Self(Branded {
                __marker: PhantomData,
                inner: idx,
            })
        }

        #[inline]
        pub unsafe fn new_raw_unchecked(idx: I) -> Self {
            Self(Branded {
                __marker: PhantomData,
                inner: idx,
            })
        }

        #[inline]
        pub fn into_inner(self) -> I {
            self.0.inner
        }

        #[inline]
        pub fn zx(self) -> Idx<'n, usize> {
            unsafe { Idx::new_raw_unchecked(self.0.inner.zx()) }
        }

        #[inline]
        pub fn sx(self) -> ! {
            unimplemented!()
        }

        #[inline]
        pub fn to_inclusive(self) -> IdxInclusive<'n, I> {
            unsafe { IdxInclusive::new_raw_unchecked(self.into_inner()) }
        }
        #[inline]
        pub fn next(self) -> IdxInclusive<'n, I> {
            unsafe { IdxInclusive::new_raw_unchecked(self.into_inner() + I::truncate(1)) }
        }

        #[track_caller]
        #[inline]
        pub fn from_slice_mut_checked<'a>(
            slice: &'a mut [I],
            size: Size<'n>,
        ) -> &'a mut [Idx<'n, I>] {
            Self::from_slice_ref_checked(slice, size);
            unsafe { &mut *(slice as *mut _ as *mut _) }
        }

        #[track_caller]
        #[inline]
        pub unsafe fn from_slice_mut_unchecked<'a>(slice: &'a mut [I]) -> &'a mut [Idx<'n, I>] {
            unsafe { &mut *(slice as *mut _ as *mut _) }
        }

        #[track_caller]
        pub fn from_slice_ref_checked<'a>(slice: &'a [I], size: Size<'n>) -> &'a [Idx<'n, I>] {
            for &idx in slice {
                Self::new_checked(idx, size);
            }
            unsafe { &*(slice as *const _ as *const _) }
        }

        #[track_caller]
        #[inline]
        pub unsafe fn from_slice_ref_unchecked<'a>(slice: &'a [I]) -> &'a [Idx<'n, I>] {
            unsafe { &*(slice as *const _ as *const _) }
        }
    }

    impl<'n, I: Index> MaybeIdx<'n, I> {
        #[inline]
        pub fn from_index(idx: Idx<'n, I>) -> Self {
            unsafe { Self::new_raw_unchecked(idx.into_inner()) }
        }
        #[inline]
        pub fn none() -> Self {
            unsafe { Self::new_raw_unchecked(I::truncate(usize::MAX)) }
        }

        #[inline]
        pub fn new_checked(idx: I::Signed, size: Size<'n>) -> Self {
            assert!((idx.sx() as isize) < size.into_inner() as isize);
            Self(Branded {
                __marker: PhantomData,
                inner: I::from_signed(idx),
            })
        }

        #[inline]
        pub unsafe fn new_unchecked(idx: I::Signed, size: Size<'n>) -> Self {
            debug_assert!((idx.sx() as isize) < size.into_inner() as isize);
            Self(Branded {
                __marker: PhantomData,
                inner: I::from_signed(idx),
            })
        }

        #[inline]
        pub unsafe fn new_raw_unchecked(idx: I) -> Self {
            Self(Branded {
                __marker: PhantomData,
                inner: idx,
            })
        }

        #[inline]
        pub fn into_inner(self) -> I {
            self.0.inner
        }

        #[inline]
        pub fn idx(self) -> Option<Idx<'n, I>> {
            if self.0.inner.to_signed() >= I::Signed::truncate(0) {
                Some(unsafe { Idx::new_raw_unchecked(self.into_inner()) })
            } else {
                None
            }
        }

        #[inline]
        pub fn zx(self) -> ! {
            unimplemented!()
        }

        #[inline]
        pub fn sx(self) -> MaybeIdx<'n, usize> {
            unsafe { MaybeIdx::new_raw_unchecked(self.0.inner.to_signed().sx()) }
        }

        #[track_caller]
        #[inline]
        pub fn from_slice_mut_checked<'a>(
            slice: &'a mut [I::Signed],
            size: Size<'n>,
        ) -> &'a mut [MaybeIdx<'n, I>] {
            Self::from_slice_ref_checked(slice, size);
            unsafe { &mut *(slice as *mut _ as *mut _) }
        }

        #[track_caller]
        #[inline]
        pub unsafe fn from_slice_mut_unchecked<'a>(
            slice: &'a mut [I::Signed],
        ) -> &'a mut [MaybeIdx<'n, I>] {
            unsafe { &mut *(slice as *mut _ as *mut _) }
        }

        #[track_caller]
        pub fn from_slice_ref_checked<'a>(
            slice: &'a [I::Signed],
            size: Size<'n>,
        ) -> &'a [MaybeIdx<'n, I>] {
            for &idx in slice {
                Self::new_checked(idx, size);
            }
            unsafe { &*(slice as *const _ as *const _) }
        }

        #[track_caller]
        pub fn as_slice_ref<'a>(slice: &'a [MaybeIdx<'n, I>]) -> &'a [I::Signed] {
            unsafe { &*(slice as *const _ as *const _) }
        }

        #[track_caller]
        #[inline]
        pub unsafe fn from_slice_ref_unchecked<'a>(
            slice: &'a [I::Signed],
        ) -> &'a [MaybeIdx<'n, I>] {
            unsafe { &*(slice as *const _ as *const _) }
        }
    }

    impl<'n> IdxInclusive<'n, usize> {
        #[inline]
        pub fn range_to(self, last: Self) -> impl DoubleEndedIterator<Item = Idx<'n, usize>> {
            (*self..*last).map(
                #[inline(always)]
                |idx| unsafe { Idx::new_raw_unchecked(idx) },
            )
        }
    }

    impl<'n, I: Index> IdxInclusive<'n, I> {
        #[inline]
        pub fn new_checked(idx: I, size: Size<'n>) -> Self {
            assert!(idx.zx() <= size.into_inner());
            Self(Branded {
                __marker: PhantomData,
                inner: idx,
            })
        }
        #[inline]
        pub unsafe fn new_unchecked(idx: I, size: Size<'n>) -> Self {
            debug_assert!(idx.zx() <= size.into_inner());
            Self(Branded {
                __marker: PhantomData,
                inner: idx,
            })
        }

        #[inline]
        pub unsafe fn new_raw_unchecked(idx: I) -> Self {
            Self(Branded {
                __marker: PhantomData,
                inner: idx,
            })
        }

        #[inline]
        pub fn into_inner(self) -> I {
            self.0.inner
        }

        #[inline]
        pub fn sx(self) -> ! {
            unimplemented!()
        }
        #[inline]
        pub fn zx(self) -> ! {
            unimplemented!()
        }
    }

    impl<'n, T> Array<'n, T> {
        #[inline]
        #[track_caller]
        pub fn from_ref<'a>(slice: &'a [T], size: Size<'n>) -> &'a Self {
            assert!(slice.len() == size.into_inner());
            unsafe { &*(slice as *const [T] as *const Self) }
        }

        #[inline]
        #[track_caller]
        pub fn from_mut<'a>(slice: &'a mut [T], size: Size<'n>) -> &'a mut Self {
            assert!(slice.len() == size.into_inner());
            unsafe { &mut *(slice as *mut [T] as *mut Self) }
        }

        #[inline]
        #[track_caller]
        pub fn as_ref(&self) -> &[T] {
            unsafe { &*(self as *const _ as *const _) }
        }

        #[inline]
        #[track_caller]
        pub fn as_mut<'a>(&mut self) -> &'a mut [T] {
            unsafe { &mut *(self as *mut _ as *mut _) }
        }

        #[inline]
        pub fn len(&self) -> Size<'n> {
            unsafe { Size::new_raw_unchecked(self.0.inner.len()) }
        }
    }

    impl<'nrows, 'ncols, 'a, E: Entity> MatRef<'nrows, 'ncols, 'a, E> {
        #[inline]
        #[track_caller]
        pub fn new(inner: super::MatRef<'a, E>, nrows: Size<'nrows>, ncols: Size<'ncols>) -> Self {
            assert!(all(
                inner.nrows() == nrows.into_inner(),
                inner.ncols() == ncols.into_inner(),
            ));
            Self(Branded {
                __marker: PhantomData,
                inner: Branded {
                    __marker: PhantomData,
                    inner,
                },
            })
        }

        #[inline]
        pub fn nrows(&self) -> Size<'nrows> {
            unsafe { Size::new_raw_unchecked(self.0.inner.inner.nrows()) }
        }

        #[inline]
        pub fn ncols(&self) -> Size<'ncols> {
            unsafe { Size::new_raw_unchecked(self.0.inner.inner.ncols()) }
        }

        #[inline]
        pub fn into_inner(self) -> super::MatRef<'a, E> {
            self.0.inner.inner
        }

        #[inline]
        #[track_caller]
        pub fn read(&self, i: Idx<'nrows, usize>, j: Idx<'ncols, usize>) -> E {
            unsafe {
                self.0
                    .inner
                    .inner
                    .read_unchecked(i.into_inner(), j.into_inner())
            }
        }
    }

    impl<'nrows, 'ncols, 'a, E: Entity> MatMut<'nrows, 'ncols, 'a, E> {
        #[inline]
        #[track_caller]
        pub fn new(inner: super::MatMut<'a, E>, nrows: Size<'nrows>, ncols: Size<'ncols>) -> Self {
            assert!(all(
                inner.nrows() == nrows.into_inner(),
                inner.ncols() == ncols.into_inner(),
            ));
            Self(Branded {
                __marker: PhantomData,
                inner: Branded {
                    __marker: PhantomData,
                    inner,
                },
            })
        }

        #[inline]
        pub fn nrows(&self) -> Size<'nrows> {
            unsafe { Size::new_raw_unchecked(self.0.inner.inner.nrows()) }
        }

        #[inline]
        pub fn ncols(&self) -> Size<'ncols> {
            unsafe { Size::new_raw_unchecked(self.0.inner.inner.ncols()) }
        }

        #[inline]
        pub fn into_inner(self) -> super::MatMut<'a, E> {
            self.0.inner.inner
        }

        #[inline]
        #[track_caller]
        pub fn read(&self, i: Idx<'nrows, usize>, j: Idx<'ncols, usize>) -> E {
            unsafe {
                self.0
                    .inner
                    .inner
                    .read_unchecked(i.into_inner(), j.into_inner())
            }
        }

        #[inline]
        #[track_caller]
        pub fn write(&mut self, i: Idx<'nrows, usize>, j: Idx<'ncols, usize>, value: E) {
            unsafe {
                self.0
                    .inner
                    .inner
                    .write_unchecked(i.into_inner(), j.into_inner(), value)
            };
        }
    }

    impl<E: Entity> Clone for MatRef<'_, '_, '_, E> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<E: Entity> Copy for MatRef<'_, '_, '_, E> {}

    impl<'nrows, 'ncols, 'a, E: Entity> IntoConst for MatRef<'nrows, 'ncols, 'a, E> {
        type Target = MatRef<'nrows, 'ncols, 'a, E>;
        #[inline]
        fn into_const(self) -> Self::Target {
            self
        }
    }
    impl<'nrows, 'ncols, 'a, 'short, E: Entity> Reborrow<'short> for MatRef<'nrows, 'ncols, 'a, E> {
        type Target = MatRef<'nrows, 'ncols, 'short, E>;
        #[inline]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }
    impl<'nrows, 'ncols, 'a, 'short, E: Entity> ReborrowMut<'short> for MatRef<'nrows, 'ncols, 'a, E> {
        type Target = MatRef<'nrows, 'ncols, 'short, E>;
        #[inline]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }

    impl<'nrows, 'ncols, 'a, E: Entity> IntoConst for MatMut<'nrows, 'ncols, 'a, E> {
        type Target = MatRef<'nrows, 'ncols, 'a, E>;
        #[inline]
        fn into_const(self) -> Self::Target {
            let inner = self.0.inner.inner.into_const();
            MatRef(Branded {
                __marker: PhantomData,
                inner: Branded {
                    __marker: PhantomData,
                    inner,
                },
            })
        }
    }
    impl<'nrows, 'ncols, 'a, 'short, E: Entity> Reborrow<'short> for MatMut<'nrows, 'ncols, 'a, E> {
        type Target = MatRef<'nrows, 'ncols, 'short, E>;
        #[inline]
        fn rb(&'short self) -> Self::Target {
            let inner = self.0.inner.inner.rb();
            MatRef(Branded {
                __marker: PhantomData,
                inner: Branded {
                    __marker: PhantomData,
                    inner,
                },
            })
        }
    }
    impl<'nrows, 'ncols, 'a, 'short, E: Entity> ReborrowMut<'short> for MatMut<'nrows, 'ncols, 'a, E> {
        type Target = MatMut<'nrows, 'ncols, 'short, E>;
        #[inline]
        fn rb_mut(&'short mut self) -> Self::Target {
            let inner = self.0.inner.inner.rb_mut();
            MatMut(Branded {
                __marker: PhantomData,
                inner: Branded {
                    __marker: PhantomData,
                    inner,
                },
            })
        }
    }

    impl Debug for Size<'_> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.0.inner.fmt(f)
        }
    }
    impl<I: Debug> Debug for Idx<'_, I> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.0.inner.fmt(f)
        }
    }
    impl<I: Debug> Debug for IdxInclusive<'_, I> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.0.inner.fmt(f)
        }
    }
    impl<I: Debug + Index> Debug for MaybeIdx<'_, I> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            #[derive(Debug)]
            struct None;

            match self.idx() {
                Some(idx) => idx.fmt(f),
                Option::None => None.fmt(f),
            }
        }
    }
    impl<T: Debug> Debug for Array<'_, T> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.0.inner.fmt(f)
        }
    }
    impl<E: Entity> Debug for MatRef<'_, '_, '_, E> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.0.inner.inner.fmt(f)
        }
    }
    impl<E: Entity> Debug for MatMut<'_, '_, '_, E> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.0.inner.inner.fmt(f)
        }
    }

    impl<'n, T> core::ops::Index<Range<IdxInclusive<'n, usize>>> for Array<'n, T> {
        type Output = [T];
        #[track_caller]
        fn index(&self, idx: Range<IdxInclusive<'n, usize>>) -> &Self::Output {
            #[cfg(debug_assertions)]
            {
                &self.0.inner[idx.start.into_inner()..idx.end.into_inner()]
            }
            #[cfg(not(debug_assertions))]
            unsafe {
                self.0
                    .inner
                    .get_unchecked(idx.start.into_inner()..idx.end.into_inner())
            }
        }
    }
    impl<'n, T> core::ops::IndexMut<Range<IdxInclusive<'n, usize>>> for Array<'n, T> {
        #[track_caller]
        fn index_mut(&mut self, idx: Range<IdxInclusive<'n, usize>>) -> &mut Self::Output {
            #[cfg(debug_assertions)]
            {
                &mut self.0.inner[idx.start.into_inner()..idx.end.into_inner()]
            }
            #[cfg(not(debug_assertions))]
            unsafe {
                self.0
                    .inner
                    .get_unchecked_mut(idx.start.into_inner()..idx.end.into_inner())
            }
        }
    }
    impl<'n, T> core::ops::Index<Idx<'n, usize>> for Array<'n, T> {
        type Output = T;
        #[track_caller]
        fn index(&self, idx: Idx<'n, usize>) -> &Self::Output {
            #[cfg(debug_assertions)]
            {
                &self.0.inner[idx.into_inner()]
            }
            #[cfg(not(debug_assertions))]
            unsafe {
                self.0.inner.get_unchecked(idx.into_inner())
            }
        }
    }
    impl<'n, T> core::ops::IndexMut<Idx<'n, usize>> for Array<'n, T> {
        #[track_caller]
        fn index_mut(&mut self, idx: Idx<'n, usize>) -> &mut Self::Output {
            #[cfg(debug_assertions)]
            {
                &mut self.0.inner[idx.into_inner()]
            }
            #[cfg(not(debug_assertions))]
            unsafe {
                self.0.inner.get_unchecked_mut(idx.into_inner())
            }
        }
    }
}

#[inline(always)]
fn norm_l2_with_simd_and_offset_prologue<E: ComplexField, S: pulp::Simd>(
    simd: S,
    data: SliceGroup<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> (
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
) {
    use group_helpers::*;

    let simd_real = SimdFor::<E::Real, S>::new(simd);
    let simd = SimdFor::<E, S>::new(simd);
    let half_big = simd_real.splat(E::Real::faer_min_positive_sqrt_inv());
    let half_small = simd_real.splat(E::Real::faer_min_positive_sqrt());
    let zero = simd.splat(E::faer_zero());
    let zero_real = simd_real.splat(E::Real::faer_zero());

    let (head, body, tail) = simd.as_aligned_simd(data, offset);
    let (body2, body1) = body.as_arrays::<2>();

    let mut acc0 = simd.abs2(head.read_or(zero));
    let mut acc1 = zero_real;

    let mut acc_small0 = simd.abs2(simd.scale_real(half_small, head.read_or(zero)));
    let mut acc_small1 = zero_real;

    let mut acc_big0 = simd.abs2(simd.scale_real(half_big, head.read_or(zero)));
    let mut acc_big1 = zero_real;

    for [x0, x1] in body2.into_ref_iter().map(RefGroup::unzip) {
        let x0 = x0.get();
        let x1 = x1.get();
        acc0 = simd.abs2_add_e(x0, acc0);
        acc1 = simd.abs2_add_e(x1, acc1);

        acc_small0 = simd.abs2_add_e(simd.scale_real(half_small, x0), acc_small0);
        acc_small1 = simd.abs2_add_e(simd.scale_real(half_small, x1), acc_small1);

        acc_big0 = simd.abs2_add_e(simd.scale_real(half_big, x0), acc_big0);
        acc_big1 = simd.abs2_add_e(simd.scale_real(half_big, x1), acc_big1);
    }

    for x0 in body1.into_ref_iter() {
        let x0 = x0.get();
        acc0 = simd.abs2_add_e(x0, acc0);
        acc_small0 = simd.abs2_add_e(simd.scale_real(half_small, x0), acc_small0);
        acc_big0 = simd.abs2_add_e(simd.scale_real(half_big, x0), acc_big0);
    }

    acc0 = simd.abs2_add_e(tail.read_or(zero), acc0);
    acc_small0 = simd.abs2_add_e(simd.scale_real(half_small, tail.read_or(zero)), acc_small0);
    acc_big0 = simd.abs2_add_e(simd.scale_real(half_big, tail.read_or(zero)), acc_big0);

    acc0 = simd_real.add(acc0, acc1);
    acc_small0 = simd_real.add(acc_small0, acc_small1);
    acc_big0 = simd_real.add(acc_big0, acc_big1);

    (acc_small0, acc0, acc_big0)
}

#[inline(always)]
fn norm_max_contiguous<E: RealField>(data: MatRef<'_, E>) -> E {
    struct Impl<'a, E: RealField> {
        data: MatRef<'a, E>,
    }

    impl<E: RealField> pulp::WithSimd for Impl<'_, E> {
        type Output = E;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { data } = self;
            use group_helpers::*;
            let m = data.nrows();
            let n = data.ncols();

            let offset = SimdFor::<E, S>::new(simd).align_offset_ptr(data.as_ptr(), m);

            let simd = SimdFor::<E, S>::new(simd);

            let zero = simd.splat(E::faer_zero());

            let mut acc0 = zero;
            let mut acc1 = zero;
            let mut acc2 = zero;
            let mut acc3 = zero;
            for j in 0..n {
                let col = SliceGroup::<'_, E>::new(data.try_get_contiguous_col(j));
                let (head, body, tail) = simd.as_aligned_simd(col, offset);
                let (body4, body1) = body.as_arrays::<4>();

                let head = simd.abs(head.read_or(zero));
                acc0 = simd.select(simd.greater_than(head, acc0), head, acc0);

                for [x0, x1, x2, x3] in body4.into_ref_iter().map(RefGroup::unzip) {
                    let x0 = simd.abs(x0.get());
                    let x1 = simd.abs(x1.get());
                    let x2 = simd.abs(x2.get());
                    let x3 = simd.abs(x3.get());
                    acc0 = simd.select(simd.greater_than(x0, acc0), x0, acc0);
                    acc1 = simd.select(simd.greater_than(x1, acc1), x1, acc1);
                    acc2 = simd.select(simd.greater_than(x2, acc2), x2, acc2);
                    acc3 = simd.select(simd.greater_than(x3, acc3), x3, acc3);
                }

                for x0 in body1.into_ref_iter() {
                    let x0 = simd.abs(x0.get());
                    acc0 = simd.select(simd.greater_than(x0, acc0), x0, acc0);
                }

                let tail = simd.abs(tail.read_or(zero));
                acc3 = simd.select(simd.greater_than(tail, acc3), tail, acc3);
            }
            acc0 = simd.select(simd.greater_than(acc0, acc1), acc0, acc1);
            acc2 = simd.select(simd.greater_than(acc2, acc3), acc2, acc3);
            acc0 = simd.select(simd.greater_than(acc0, acc2), acc0, acc2);

            let acc0 = from_copy::<E, _>(simd.rotate_left(acc0, offset.rotate_left_amount()));
            let acc = SliceGroup::<'_, E>::new(E::faer_map(
                E::faer_as_ref(&acc0),
                #[inline(always)]
                |acc| bytemuck::cast_slice::<_, <E as Entity>::Unit>(core::slice::from_ref(acc)),
            ));
            let mut acc_scalar = E::faer_zero();
            for x in acc.into_ref_iter() {
                let x = x.read();
                acc_scalar = if acc_scalar > x { acc_scalar } else { x };
            }
            acc_scalar
        }
    }

    E::Simd::default().dispatch(Impl { data })
}

const NORM_L2_THRESHOLD: usize = 128;

#[inline(always)]
fn norm_l2_with_simd_and_offset_pairwise_rows<E: ComplexField, S: Simd>(
    simd: S,
    data: SliceGroup<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
    last_offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> (
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
) {
    struct Impl<'a, E: ComplexField, S: Simd> {
        simd: S,
        data: SliceGroup<'a, E>,
        offset: pulp::Offset<SimdMaskFor<E, S>>,
        last_offset: pulp::Offset<SimdMaskFor<E, S>>,
    }

    impl<E: ComplexField, S: Simd> pulp::NullaryFnOnce for Impl<'_, E, S> {
        type Output = (
            SimdGroupFor<E::Real, S>,
            SimdGroupFor<E::Real, S>,
            SimdGroupFor<E::Real, S>,
        );

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                data,
                offset,
                last_offset,
            } = self;

            if data.len() == NORM_L2_THRESHOLD {
                norm_l2_with_simd_and_offset_prologue(simd, data, offset)
            } else if data.len() < NORM_L2_THRESHOLD {
                norm_l2_with_simd_and_offset_prologue(simd, data, last_offset)
            } else {
                let split_point = ((data.len() + 1) / 2).next_power_of_two();
                let (head, tail) = data.split_at(split_point);
                let (acc_small0, acc0, acc_big0) =
                    norm_l2_with_simd_and_offset_pairwise_rows(simd, head, offset, last_offset);
                let (acc_small1, acc1, acc_big1) =
                    norm_l2_with_simd_and_offset_pairwise_rows(simd, tail, offset, last_offset);

                use group_helpers::*;
                let simd = SimdFor::<E::Real, S>::new(simd);
                (
                    simd.add(acc_small0, acc_small1),
                    simd.add(acc0, acc1),
                    simd.add(acc_big0, acc_big1),
                )
            }
        }
    }

    simd.vectorize(Impl {
        simd,
        data,
        offset,
        last_offset,
    })
}

#[inline(always)]
fn norm_l2_with_simd_and_offset_pairwise_cols<E: ComplexField, S: Simd>(
    simd: S,
    data: MatRef<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
    last_offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> (
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
) {
    struct Impl<'a, E: ComplexField, S: Simd> {
        simd: S,
        data: MatRef<'a, E>,
        offset: pulp::Offset<SimdMaskFor<E, S>>,
        last_offset: pulp::Offset<SimdMaskFor<E, S>>,
    }

    impl<E: ComplexField, S: Simd> pulp::NullaryFnOnce for Impl<'_, E, S> {
        type Output = (
            SimdGroupFor<E::Real, S>,
            SimdGroupFor<E::Real, S>,
            SimdGroupFor<E::Real, S>,
        );

        #[inline(always)]
        fn call(self) -> Self::Output {
            use group_helpers::*;

            let Self {
                simd,
                data,
                offset,
                last_offset,
            } = self;
            if data.ncols() == 1 {
                norm_l2_with_simd_and_offset_pairwise_rows(
                    simd,
                    SliceGroup::<'_, E>::new(data.try_get_contiguous_col(0)),
                    offset,
                    last_offset,
                )
            } else {
                let split_point = (data.ncols() / 2).next_power_of_two();

                let (head, tail) = data.split_at_col(split_point);

                let (acc_small0, acc0, acc_big0) =
                    norm_l2_with_simd_and_offset_pairwise_cols(simd, head, offset, last_offset);
                let (acc_small1, acc1, acc_big1) =
                    norm_l2_with_simd_and_offset_pairwise_cols(simd, tail, offset, last_offset);

                let simd = SimdFor::<E::Real, S>::new(simd);
                (
                    simd.add(acc_small0, acc_small1),
                    simd.add(acc0, acc1),
                    simd.add(acc_big0, acc_big1),
                )
            }
        }
    }

    simd.vectorize(Impl {
        simd,
        data,
        offset,
        last_offset,
    })
}

fn norm_l2_contiguous<E: ComplexField>(data: MatRef<'_, E>) -> (E::Real, E::Real, E::Real) {
    struct Impl<'a, E: ComplexField> {
        data: MatRef<'a, E>,
    }

    impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
        type Output = (E::Real, E::Real, E::Real);

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { data } = self;
            use group_helpers::*;

            let offset =
                SimdFor::<E, S>::new(simd).align_offset_ptr(data.as_ptr(), NORM_L2_THRESHOLD);

            let last_offset = SimdFor::<E, S>::new(simd)
                .align_offset_ptr(data.as_ptr(), data.nrows() % NORM_L2_THRESHOLD);

            let (acc_small, acc, acc_big) =
                norm_l2_with_simd_and_offset_pairwise_cols(simd, data, offset, last_offset);

            let simd = SimdFor::<E::Real, S>::new(simd);
            (
                simd.reduce_add(simd.rotate_left(acc_small, offset.rotate_left_amount())),
                simd.reduce_add(simd.rotate_left(acc, offset.rotate_left_amount())),
                simd.reduce_add(simd.rotate_left(acc_big, offset.rotate_left_amount())),
            )
        }
    }

    E::Simd::default().dispatch(Impl { data })
}

fn norm_l2<E: ComplexField>(mut mat: MatRef<'_, E>) -> E::Real {
    if mat.ncols() > 1 && mat.col_stride().unsigned_abs() < mat.row_stride().unsigned_abs() {
        mat = mat.transpose();
    }
    if mat.row_stride() < 0 {
        mat = mat.reverse_rows();
    }

    if mat.nrows() == 0 || mat.ncols() == 0 {
        E::Real::faer_zero()
    } else {
        let m = mat.nrows();
        let n = mat.ncols();

        let half_small = E::Real::faer_min_positive_sqrt();
        let half_big = E::Real::faer_min_positive_sqrt_inv();

        let mut acc_small = E::Real::faer_zero();
        let mut acc = E::Real::faer_zero();
        let mut acc_big = E::Real::faer_zero();

        if mat.row_stride() == 1 {
            if coe::is_same::<E, c32>() {
                let mat: MatRef<'_, c32> = coe::coerce(mat);
                let mat = unsafe {
                    mat::from_raw_parts(
                        mat.as_ptr() as *const f32,
                        2 * mat.nrows(),
                        mat.ncols(),
                        1,
                        2 * mat.col_stride(),
                    )
                };
                let (acc_small_, acc_, acc_big_) = norm_l2_contiguous::<f32>(mat);
                acc_small = coe::coerce_static(acc_small_);
                acc = coe::coerce_static(acc_);
                acc_big = coe::coerce_static(acc_big_);
            } else if coe::is_same::<E, c64>() {
                let mat: MatRef<'_, c64> = coe::coerce(mat);
                let mat = unsafe {
                    mat::from_raw_parts(
                        mat.as_ptr() as *const f64,
                        2 * mat.nrows(),
                        mat.ncols(),
                        1,
                        2 * mat.col_stride(),
                    )
                };
                let (acc_small_, acc_, acc_big_) = norm_l2_contiguous::<f64>(mat);
                acc_small = coe::coerce_static(acc_small_);
                acc = coe::coerce_static(acc_);
                acc_big = coe::coerce_static(acc_big_);
            } else {
                (acc_small, acc, acc_big) = norm_l2_contiguous(mat);
            }
        } else {
            for j in 0..n {
                for i in 0..m {
                    let val = mat.read(i, j);
                    let val_small = val.faer_scale_power_of_two(half_small);
                    let val_big = val.faer_scale_power_of_two(half_big);

                    acc_small = acc_small.faer_add(val_small.faer_abs2());
                    acc = acc.faer_add(val.faer_abs2());
                    acc_big = acc_big.faer_add(val_big.faer_abs2());
                }
            }
        }

        if acc_small >= E::Real::faer_one() {
            acc_small.faer_sqrt().faer_mul(half_big)
        } else if acc_big <= E::Real::faer_one() {
            acc_big.faer_sqrt().faer_mul(half_small)
        } else {
            acc.faer_sqrt()
        }
    }
}

fn norm_max<E: ComplexField>(mut mat: MatRef<'_, E>) -> E::Real {
    if mat.ncols() > 1 && mat.col_stride().unsigned_abs() < mat.row_stride().unsigned_abs() {
        mat = mat.transpose();
    }
    if mat.row_stride() < 0 {
        mat = mat.reverse_rows();
    }

    if mat.nrows() == 0 || mat.ncols() == 0 {
        E::Real::faer_zero()
    } else {
        let m = mat.nrows();
        let n = mat.ncols();

        if mat.row_stride() == 1 {
            if coe::is_same::<E, c32>() {
                let mat: MatRef<'_, c32> = coe::coerce(mat);
                let mat = unsafe {
                    mat::from_raw_parts(
                        mat.as_ptr() as *const f32,
                        2 * mat.nrows(),
                        mat.ncols(),
                        1,
                        2 * mat.col_stride(),
                    )
                };
                return coe::coerce_static(norm_max_contiguous::<f32>(mat));
            } else if coe::is_same::<E, c64>() {
                let mat: MatRef<'_, c64> = coe::coerce(mat);
                let mat = unsafe {
                    mat::from_raw_parts(
                        mat.as_ptr() as *const f64,
                        2 * mat.nrows(),
                        mat.ncols(),
                        1,
                        2 * mat.col_stride(),
                    )
                };
                return coe::coerce_static(norm_max_contiguous::<f64>(mat));
            } else if coe::is_same::<E, num_complex::Complex<E::Real>>() {
                let mat: MatRef<'_, num_complex::Complex<E::Real>> = coe::coerce(mat);
                let num_complex::Complex { re, im } = mat.real_imag();
                let re = norm_max_contiguous(re);
                let im = norm_max_contiguous(im);
                return if re > im { re } else { im };
            } else if coe::is_same::<E, E::Real>() {
                let mat: MatRef<'_, E::Real> = coe::coerce(mat);
                return norm_max_contiguous(mat);
            }
        }

        let mut acc = E::Real::faer_zero();
        for j in 0..n {
            for i in 0..m {
                let val = mat.read(i, j);
                let re = val.faer_real();
                let im = val.faer_imag();
                acc = if re > acc { re } else { acc };
                acc = if im > acc { im } else { acc };
            }
        }
        acc
    }
}

/// Matrix view creation module.
pub mod mat {
    use super::*;

    /// Creates a `MatRef` from pointers to the matrix data, dimensions, and strides.
    ///
    /// The row (resp. column) stride is the offset from the memory address of a given matrix
    /// element at indices `(row: i, col: j)`, to the memory address of the matrix element at
    /// indices `(row: i + 1, col: 0)` (resp. `(row: 0, col: i + 1)`). This offset is specified in
    /// number of elements, not in bytes.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * For each matrix unit, the entire memory region addressed by the matrix must be contained
    /// within a single allocation, accessible in its entirety by the corresponding pointer in
    /// `ptr`.
    /// * For each matrix unit, the corresponding pointer must be properly aligned,
    /// even for a zero-sized matrix.
    /// * The values accessible by the matrix must be initialized at some point before they are
    /// read, or references to them are formed.
    /// * No mutable aliasing is allowed. In other words, none of the elements accessible by any
    /// matrix unit may be accessed for writes by any other means for the duration of the lifetime
    /// `'a`.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// // row major matrix with 2 rows, 3 columns, with a column at the end that we want to skip.
    /// // the row stride is the pointer offset from the address of 1.0 to the address of 4.0,
    /// // which is 4.
    /// // the column stride is the pointer offset from the address of 1.0 to the address of 2.0,
    /// // which is 1.
    /// let data = [[1.0, 2.0, 3.0, f64::NAN], [4.0, 5.0, 6.0, f64::NAN]];
    /// let matrix = unsafe { mat::from_raw_parts::<f64>(data.as_ptr() as *const f64, 2, 3, 4, 1) };
    ///
    /// let expected = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// assert_eq!(expected.as_ref(), matrix);
    /// ```
    #[inline(always)]
    pub unsafe fn from_raw_parts<'a, E: Entity>(
        ptr: GroupFor<E, *const E::Unit>,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> MatRef<'a, E> {
        MatRef {
            inner: DenseRef {
                inner: MatImpl {
                    ptr: into_copy::<E, _>(E::faer_map(ptr, |ptr| {
                        NonNull::new_unchecked(ptr as *mut E::Unit)
                    })),
                    nrows,
                    ncols,
                    row_stride,
                    col_stride,
                },
                __marker: PhantomData,
            },
        }
    }

    /// Creates a `MatMut` from pointers to the matrix data, dimensions, and strides.
    ///
    /// The row (resp. column) stride is the offset from the memory address of a given matrix
    /// element at indices `(row: i, col: j)`, to the memory address of the matrix element at
    /// indices `(row: i + 1, col: 0)` (resp. `(row: 0, col: i + 1)`). This offset is specified in
    /// number of elements, not in bytes.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * For each matrix unit, the entire memory region addressed by the matrix must be contained
    /// within a single allocation, accessible in its entirety by the corresponding pointer in
    /// `ptr`.
    /// * For each matrix unit, the corresponding pointer must be non null and properly aligned,
    /// even for a zero-sized matrix.
    /// * The values accessible by the matrix must be initialized at some point before they are
    ///   read, or
    /// references to them are formed.
    /// * No aliasing (including self aliasing) is allowed. In other words, none of the elements
    /// accessible by any matrix unit may be accessed for reads or writes by any other means for
    /// the duration of the lifetime `'a`. No two elements within a single matrix unit may point to
    /// the same address (such a thing can be achieved with a zero stride, for example), and no two
    /// matrix units may point to the same address.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// // row major matrix with 2 rows, 3 columns, with a column at the end that we want to skip.
    /// // the row stride is the pointer offset from the address of 1.0 to the address of 4.0,
    /// // which is 4.
    /// // the column stride is the pointer offset from the address of 1.0 to the address of 2.0,
    /// // which is 1.
    /// let mut data = [[1.0, 2.0, 3.0, f64::NAN], [4.0, 5.0, 6.0, f64::NAN]];
    /// let mut matrix =
    ///     unsafe { mat::from_raw_parts_mut::<f64>(data.as_mut_ptr() as *mut f64, 2, 3, 4, 1) };
    ///
    /// let expected = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// assert_eq!(expected.as_ref(), matrix);
    /// ```
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut<'a, E: Entity>(
        ptr: GroupFor<E, *mut E::Unit>,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> MatMut<'a, E> {
        MatMut {
            inner: DenseMut {
                inner: MatImpl {
                    ptr: into_copy::<E, _>(E::faer_map(ptr, |ptr| {
                        NonNull::new_unchecked(ptr as *mut E::Unit)
                    })),
                    nrows,
                    ncols,
                    row_stride,
                    col_stride,
                },
                __marker: PhantomData,
            },
        }
    }

    /// Creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
    /// The data is interpreted in a column-major format, so that the first chunk of `nrows`
    /// values from the slices goes in the first column of the matrix, the second chunk of `nrows`
    /// values goes in the second column, and so on.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `nrows * ncols == slice.len()`
    ///
    /// # Example
    /// ```
    /// use faer_core::mat;
    ///
    /// let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
    /// let view = mat::from_column_major_slice::<f64>(&slice, 3, 2);
    ///
    /// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    /// assert_eq!(expected, view);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn from_column_major_slice<'a, E: Entity>(
        slice: GroupFor<E, &'a [E::Unit]>,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'a, E> {
        from_slice_assert(
            nrows,
            ncols,
            SliceGroup::<'_, E>::new(E::faer_copy(&slice)).len(),
        );

        unsafe {
            from_raw_parts(
                E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.as_ptr(),
                ),
                nrows,
                ncols,
                1,
                nrows as isize,
            )
        }
    }

    /// Creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
    /// The data is interpreted in a row-major format, so that the first chunk of `ncols`
    /// values from the slices goes in the first column of the matrix, the second chunk of `ncols`
    /// values goes in the second column, and so on.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `nrows * ncols == slice.len()`
    ///
    /// # Example
    /// ```
    /// use faer_core::mat;
    ///
    /// let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
    /// let view = mat::from_row_major_slice::<f64>(&slice, 3, 2);
    ///
    /// let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// assert_eq!(expected, view);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn from_row_major_slice<'a, E: Entity>(
        slice: GroupFor<E, &'a [E::Unit]>,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'a, E> {
        from_column_major_slice(slice, ncols, nrows).transpose()
    }

    #[track_caller]
    pub fn from_column_major_slice_with_stride<'a, E: Entity>(
        slice: GroupFor<E, &'a [E::Unit]>,
        nrows: usize,
        ncols: usize,
        col_stride: usize,
    ) -> MatRef<'a, E> {
        from_strided_column_major_slice_assert(
            nrows,
            ncols,
            col_stride,
            SliceGroup::<'_, E>::new(E::faer_copy(&slice)).len(),
        );

        unsafe {
            from_raw_parts(
                E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.as_ptr(),
                ),
                nrows,
                ncols,
                1,
                col_stride as isize,
            )
        }
    }

    /// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
    /// The data is interpreted in a column-major format, so that the first chunk of `nrows`
    /// values from the slices goes in the first column of the matrix, the second chunk of `nrows`
    /// values goes in the second column, and so on.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `nrows * ncols == slice.len()`
    ///
    /// # Example
    /// ```
    /// use faer_core::mat;
    ///
    /// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
    /// let view = mat::from_column_major_slice_mut::<f64>(&mut slice, 3, 2);
    ///
    /// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    /// assert_eq!(expected, view);
    /// ```
    #[track_caller]
    pub fn from_column_major_slice_mut<'a, E: Entity>(
        slice: GroupFor<E, &'a mut [E::Unit]>,
        nrows: usize,
        ncols: usize,
    ) -> MatMut<'a, E> {
        from_slice_assert(
            nrows,
            ncols,
            SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&slice))).len(),
        );
        unsafe {
            from_raw_parts_mut(
                E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.as_mut_ptr(),
                ),
                nrows,
                ncols,
                1,
                nrows as isize,
            )
        }
    }

    /// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
    /// The data is interpreted in a row-major format, so that the first chunk of `ncols`
    /// values from the slices goes in the first column of the matrix, the second chunk of `ncols`
    /// values goes in the second column, and so on.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `nrows * ncols == slice.len()`
    ///
    /// # Example
    /// ```
    /// use faer_core::mat;
    ///
    /// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
    /// let view = mat::from_row_major_slice_mut::<f64>(&mut slice, 3, 2);
    ///
    /// let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// assert_eq!(expected, view);
    /// ```
    #[inline(always)]
    #[track_caller]
    pub fn from_row_major_slice_mut<'a, E: Entity>(
        slice: GroupFor<E, &'a mut [E::Unit]>,
        nrows: usize,
        ncols: usize,
    ) -> MatMut<'a, E> {
        from_column_major_slice_mut(slice, ncols, nrows).transpose_mut()
    }

    #[track_caller]
    pub fn from_column_major_slice_with_stride_mut<'a, E: Entity>(
        slice: GroupFor<E, &'a mut [E::Unit]>,
        nrows: usize,
        ncols: usize,
        col_stride: usize,
    ) -> MatMut<'a, E> {
        from_strided_column_major_slice_mut_assert(
            nrows,
            ncols,
            col_stride,
            SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&slice))).len(),
        );
        unsafe {
            from_raw_parts_mut(
                E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.as_mut_ptr(),
                ),
                nrows,
                ncols,
                1,
                col_stride as isize,
            )
        }
    }
}

/// Column view creation module.
pub mod col {
    use super::*;

    /// Creates a `ColRef` from pointers to the column vector data, number of rows, and row stride.
    ///
    /// # Safety:
    /// This function has the same safety requirements as
    /// [`mat::from_raw_parts(ptr, nrows, 1, row_stride, 0)`]
    #[inline(always)]
    pub unsafe fn from_raw_parts<'a, E: Entity>(
        ptr: GroupFor<E, *const E::Unit>,
        nrows: usize,
        row_stride: isize,
    ) -> ColRef<'a, E> {
        ColRef {
            inner: DenseColRef {
                inner: VecImpl {
                    ptr: into_copy::<E, _>(E::faer_map(ptr, |ptr| {
                        NonNull::new_unchecked(ptr as *mut E::Unit)
                    })),
                    len: nrows,
                    stride: row_stride,
                },
                __marker: PhantomData,
            },
        }
    }

    /// Creates a `ColMut` from pointers to the column vector data, number of rows, and row stride.
    ///
    /// # Safety:
    /// This function has the same safety requirements as
    /// [`mat::from_raw_parts_mut(ptr, nrows, 1, row_stride, 0)`]
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut<'a, E: Entity>(
        ptr: GroupFor<E, *mut E::Unit>,
        nrows: usize,
        row_stride: isize,
    ) -> ColMut<'a, E> {
        ColMut {
            inner: DenseColMut {
                inner: VecImpl {
                    ptr: into_copy::<E, _>(E::faer_map(ptr, |ptr| {
                        NonNull::new_unchecked(ptr as *mut E::Unit)
                    })),
                    len: nrows,
                    stride: row_stride,
                },
                __marker: PhantomData,
            },
        }
    }

    /// Creates a `ColRef` from slice views over the column vector data, The result has the same
    /// number of rows as the length of the input slice.
    #[inline(always)]
    pub fn from_slice<'a, E: Entity>(slice: GroupFor<E, &'a [E::Unit]>) -> ColRef<'a, E> {
        let nrows = SliceGroup::<'_, E>::new(E::faer_copy(&slice)).len();

        unsafe {
            from_raw_parts(
                E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.as_ptr(),
                ),
                nrows,
                1,
            )
        }
    }

    /// Creates a `ColMut` from slice views over the column vector data, The result has the same
    /// number of rows as the length of the input slice.
    #[inline(always)]
    pub fn from_slice_mut<'a, E: Entity>(slice: GroupFor<E, &'a mut [E::Unit]>) -> ColMut<'a, E> {
        let nrows = SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&slice))).len();

        unsafe {
            from_raw_parts_mut(
                E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.as_mut_ptr(),
                ),
                nrows,
                1,
            )
        }
    }
}

/// Row view creation module.
pub mod row {
    use super::*;

    /// Creates a `RowRef` from pointers to the row vector data, number of columns, and column
    /// stride.
    ///
    /// # Safety:
    /// This function has the same safety requirements as
    /// [`mat::from_raw_parts(ptr, 1, ncols, 0, col_stride)`]
    #[inline(always)]
    pub unsafe fn from_raw_parts<'a, E: Entity>(
        ptr: GroupFor<E, *const E::Unit>,
        ncols: usize,
        col_stride: isize,
    ) -> RowRef<'a, E> {
        RowRef {
            inner: DenseRowRef {
                inner: VecImpl {
                    ptr: into_copy::<E, _>(E::faer_map(ptr, |ptr| {
                        NonNull::new_unchecked(ptr as *mut E::Unit)
                    })),
                    len: ncols,
                    stride: col_stride,
                },
                __marker: PhantomData,
            },
        }
    }

    /// Creates a `RowMut` from pointers to the row vector data, number of columns, and column
    /// stride.
    ///
    /// # Safety:
    /// This function has the same safety requirements as
    /// [`mat::from_raw_parts_mut(ptr, 1, ncols, 0, col_stride)`]
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut<'a, E: Entity>(
        ptr: GroupFor<E, *mut E::Unit>,
        ncols: usize,
        col_stride: isize,
    ) -> RowMut<'a, E> {
        RowMut {
            inner: DenseRowMut {
                inner: VecImpl {
                    ptr: into_copy::<E, _>(E::faer_map(ptr, |ptr| {
                        NonNull::new_unchecked(ptr as *mut E::Unit)
                    })),
                    len: ncols,
                    stride: col_stride,
                },
                __marker: PhantomData,
            },
        }
    }

    /// Creates a `RowRef` from slice views over the row vector data, The result has the same
    /// number of columns as the length of the input slice.
    #[inline(always)]
    pub fn from_slice<'a, E: Entity>(slice: GroupFor<E, &'a [E::Unit]>) -> RowRef<'a, E> {
        let nrows = SliceGroup::<'_, E>::new(E::faer_copy(&slice)).len();

        unsafe {
            from_raw_parts(
                E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.as_ptr(),
                ),
                nrows,
                1,
            )
        }
    }

    /// Creates a `RowMut` from slice views over the row vector data, The result has the same
    /// number of columns as the length of the input slice.
    #[inline(always)]
    pub fn from_slice_mut<'a, E: Entity>(slice: GroupFor<E, &'a mut [E::Unit]>) -> RowMut<'a, E> {
        let nrows = SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&slice))).len();

        unsafe {
            from_raw_parts_mut(
                E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.as_mut_ptr(),
                ),
                nrows,
                1,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    macro_rules! impl_unit_entity {
        ($ty: ty) => {
            unsafe impl Entity for $ty {
                type Unit = Self;
                type Index = ();
                type SimdUnit<S: $crate::pulp::Simd> = ();
                type SimdMask<S: $crate::pulp::Simd> = ();
                type SimdIndex<S: $crate::pulp::Simd> = ();
                type Group = IdentityGroup;
                type Iter<I: Iterator> = I;

                type PrefixUnit<'a, S: Simd> = &'a [()];
                type SuffixUnit<'a, S: Simd> = &'a [()];
                type PrefixMutUnit<'a, S: Simd> = &'a mut [()];
                type SuffixMutUnit<'a, S: Simd> = &'a mut [()];

                const N_COMPONENTS: usize = 1;
                const UNIT: GroupCopyFor<Self, ()> = ();

                #[inline(always)]
                fn faer_first<T>(group: GroupFor<Self, T>) -> T {
                    group
                }

                #[inline(always)]
                fn faer_from_units(group: GroupFor<Self, Self::Unit>) -> Self {
                    group
                }

                #[inline(always)]
                fn faer_into_units(self) -> GroupFor<Self, Self::Unit> {
                    self
                }

                #[inline(always)]
                fn faer_as_ref<T>(group: &GroupFor<Self, T>) -> GroupFor<Self, &T> {
                    group
                }

                #[inline(always)]
                fn faer_as_mut<T>(group: &mut GroupFor<Self, T>) -> GroupFor<Self, &mut T> {
                    group
                }

                #[inline(always)]
                fn faer_map_impl<T, U>(
                    group: GroupFor<Self, T>,
                    f: &mut impl FnMut(T) -> U,
                ) -> GroupFor<Self, U> {
                    (*f)(group)
                }

                #[inline(always)]
                fn faer_map_with_context<Ctx, T, U>(
                    ctx: Ctx,
                    group: GroupFor<Self, T>,
                    f: &mut impl FnMut(Ctx, T) -> (Ctx, U),
                ) -> (Ctx, GroupFor<Self, U>) {
                    (*f)(ctx, group)
                }

                #[inline(always)]
                fn faer_zip<T, U>(
                    first: GroupFor<Self, T>,
                    second: GroupFor<Self, U>,
                ) -> GroupFor<Self, (T, U)> {
                    (first, second)
                }
                #[inline(always)]
                fn faer_unzip<T, U>(
                    zipped: GroupFor<Self, (T, U)>,
                ) -> (GroupFor<Self, T>, GroupFor<Self, U>) {
                    zipped
                }

                #[inline(always)]
                fn faer_into_iter<I: IntoIterator>(
                    iter: GroupFor<Self, I>,
                ) -> Self::Iter<I::IntoIter> {
                    iter.into_iter()
                }
            }
        };
    }

    use super::*;
    use crate::assert;

    #[test]
    fn basic_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let slice = unsafe { mat::from_raw_parts::<'_, f64>(data.as_ptr(), 2, 3, 3, 1) };

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

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Zst;
    unsafe impl bytemuck::Zeroable for Zst {}
    unsafe impl bytemuck::Pod for Zst {}

    #[test]
    fn reserve_zst() {
        impl_unit_entity!(Zst);

        let mut m = Mat::<Zst>::new();

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
        let f = |_i, _j| Zst;
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
    fn to_owned_equality() {
        use num_complex::Complex as C;
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
        use num_complex::Complex as C;
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

        assert_eq!(matrix.col_as_slice(1), &[5.0, 6.0, 7.0]);
        assert_eq!(matrix.col_as_slice_mut(0), &[1.0, 2.0, 3.0]);

        matrix
            .col_as_slice_mut(0)
            .copy_from_slice(&[-1.0, -2.0, -3.0]);

        let expected = mat![[-1.0, 5.0, 9.0], [-2.0, 6.0, 10.0], [-3.0, 7.0, 11.0f64]];
        assert_eq!(matrix, expected);
    }

    #[test]
    fn from_slice() {
        let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];

        let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        let view = mat::from_column_major_slice::<'_, f64>(&slice, 3, 2);
        assert_eq!(expected, view);
        let view = mat::from_column_major_slice::<'_, f64>(&mut slice, 3, 2);
        assert_eq!(expected, view);

        let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let view = mat::from_row_major_slice::<'_, f64>(&slice, 3, 2);
        assert_eq!(expected, view);
        let view = mat::from_row_major_slice::<'_, f64>(&mut slice, 3, 2);
        assert_eq!(expected, view);
    }

    #[test]
    #[should_panic]
    fn from_slice_too_big() {
        let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0_f64];
        mat::from_column_major_slice::<'_, f64>(&slice, 3, 2);
    }

    #[test]
    #[should_panic]
    fn from_slice_too_small() {
        let slice = [1.0, 2.0, 3.0, 4.0, 5.0_f64];
        mat::from_column_major_slice::<'_, f64>(&slice, 3, 2);
    }

    #[test]
    fn test_is_finite() {
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
    fn test_norm_l2() {
        let relative_err = |a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());

        for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
            for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
                let mat = Mat::from_fn(m, n, |i, j| factor * ((i + j) as f64));
                let mut target = 0.0;
                zipped!(mat.as_ref()).for_each(|unzipped!(x)| {
                    target = f64::hypot(*x, target);
                });

                if factor == 0.0 {
                    assert!(mat.norm_l2() == target);
                } else {
                    assert!(relative_err(mat.norm_l2(), target) < 1e-14);
                }
            }
        }

        let mat = Col::from_fn(10000000, |_| 0.3);
        let target = (0.3 * 0.3 * 10000000.0f64).sqrt();
        assert!(relative_err(mat.norm_l2(), target) < 1e-14);
    }

    #[test]
    fn test_col_index() {
        let mut col_32: Col<f32> = Col::from_fn(3, |i| i as f32);
        col_32.as_mut()[1] = 10f32;
        let tval: f32 = (10f32 - col_32[1]).abs();
        assert!(tval < 1e-14);

        let mut col_64: Col<f64> = Col::from_fn(3, |i| i as f64);
        col_64.as_mut()[1] = 10f64;
        let tval: f64 = (10f64 - col_64[1]).abs();
        assert!(tval < 1e-14);
    }

    #[test]
    fn test_row_index() {
        let mut row_32: Row<f32> = Row::from_fn(3, |i| i as f32);
        row_32.as_mut()[1] = 10f32;
        let tval: f32 = (10f32 - row_32[1]).abs();
        assert!(tval < 1e-14);

        let mut row_64: Row<f64> = Row::from_fn(3, |i| i as f64);
        row_64.as_mut()[1] = 10f64;
        let tval: f64 = (10f64 - row_64[1]).abs();
        assert!(tval < 1e-14);
    }
}

pub mod zip {
    use super::{assert, debug_assert, *};
    use core::mem::MaybeUninit;

    /// Read only view over a single matrix element.
    pub struct Read<'a, E: Entity> {
        ptr: GroupFor<E, &'a MaybeUninit<E::Unit>>,
    }
    /// Read-write view over a single matrix element.
    pub struct ReadWrite<'a, E: Entity> {
        ptr: GroupFor<E, &'a mut MaybeUninit<E::Unit>>,
    }

    pub trait ViewMut {
        type Target<'a>
        where
            Self: 'a;

        fn view_mut(&mut self) -> Self::Target<'_>;
    }

    impl<E: Entity> ViewMut for Row<E> {
        type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            self.as_ref()
        }
    }
    impl<E: Entity> ViewMut for &Row<E> {
        type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (*self).as_ref()
        }
    }
    impl<E: Entity> ViewMut for &mut Row<E> {
        type Target<'a> = RowMut<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (*self).as_mut()
        }
    }

    impl<E: Entity> ViewMut for RowRef<'_, E> {
        type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            *self
        }
    }
    impl<E: Entity> ViewMut for RowMut<'_, E> {
        type Target<'a> = RowMut<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (*self).rb_mut()
        }
    }
    impl<E: Entity> ViewMut for &mut RowRef<'_, E> {
        type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            **self
        }
    }
    impl<E: Entity> ViewMut for &mut RowMut<'_, E> {
        type Target<'a> = RowMut<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (**self).rb_mut()
        }
    }
    impl<E: Entity> ViewMut for &RowRef<'_, E> {
        type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            **self
        }
    }
    impl<E: Entity> ViewMut for &RowMut<'_, E> {
        type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (**self).rb()
        }
    }

    impl<E: Entity> ViewMut for Col<E> {
        type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            self.as_ref()
        }
    }
    impl<E: Entity> ViewMut for &Col<E> {
        type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (*self).as_ref()
        }
    }
    impl<E: Entity> ViewMut for &mut Col<E> {
        type Target<'a> = ColMut<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (*self).as_mut()
        }
    }

    impl<E: Entity> ViewMut for ColRef<'_, E> {
        type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            *self
        }
    }
    impl<E: Entity> ViewMut for ColMut<'_, E> {
        type Target<'a> = ColMut<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (*self).rb_mut()
        }
    }
    impl<E: Entity> ViewMut for &mut ColRef<'_, E> {
        type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            **self
        }
    }
    impl<E: Entity> ViewMut for &mut ColMut<'_, E> {
        type Target<'a> = ColMut<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (**self).rb_mut()
        }
    }
    impl<E: Entity> ViewMut for &ColRef<'_, E> {
        type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            **self
        }
    }
    impl<E: Entity> ViewMut for &ColMut<'_, E> {
        type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (**self).rb()
        }
    }

    impl<E: Entity> ViewMut for Mat<E> {
        type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            self.as_ref()
        }
    }
    impl<E: Entity> ViewMut for &Mat<E> {
        type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (*self).as_ref()
        }
    }
    impl<E: Entity> ViewMut for &mut Mat<E> {
        type Target<'a> = MatMut<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (*self).as_mut()
        }
    }

    impl<E: Entity> ViewMut for MatRef<'_, E> {
        type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            *self
        }
    }
    impl<E: Entity> ViewMut for MatMut<'_, E> {
        type Target<'a> = MatMut<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (*self).rb_mut()
        }
    }
    impl<E: Entity> ViewMut for &mut MatRef<'_, E> {
        type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            **self
        }
    }
    impl<E: Entity> ViewMut for &mut MatMut<'_, E> {
        type Target<'a> = MatMut<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (**self).rb_mut()
        }
    }
    impl<E: Entity> ViewMut for &MatRef<'_, E> {
        type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            **self
        }
    }
    impl<E: Entity> ViewMut for &MatMut<'_, E> {
        type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

        #[inline]
        fn view_mut(&mut self) -> Self::Target<'_> {
            (**self).rb()
        }
    }

    impl<E: SimpleEntity> core::ops::Deref for Read<'_, E> {
        type Target = E;
        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            unsafe { &*(self.ptr as *const _ as *const E::Unit) }
        }
    }
    impl<E: SimpleEntity> core::ops::Deref for ReadWrite<'_, E> {
        type Target = E;
        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            unsafe { &*(self.ptr as *const _ as *const E::Unit) }
        }
    }
    impl<E: SimpleEntity> core::ops::DerefMut for ReadWrite<'_, E> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut Self::Target {
            unsafe { &mut *(self.ptr as *mut _ as *mut E::Unit) }
        }
    }

    impl<E: Entity> Read<'_, E> {
        #[inline(always)]
        pub fn read(&self) -> E {
            E::faer_from_units(E::faer_map(
                E::faer_as_ref(&self.ptr),
                #[inline(always)]
                |ptr| unsafe { ptr.assume_init_read() },
            ))
        }
    }
    impl<E: Entity> ReadWrite<'_, E> {
        #[inline(always)]
        pub fn read(&self) -> E {
            E::faer_from_units(E::faer_map(
                E::faer_as_ref(&self.ptr),
                #[inline(always)]
                |ptr| unsafe { *ptr.assume_init_ref() },
            ))
        }

        #[inline(always)]
        pub fn write(&mut self, value: E) {
            let value = E::faer_into_units(value);
            E::faer_map(
                E::faer_zip(E::faer_as_mut(&mut self.ptr), value),
                #[inline(always)]
                |(ptr, value)| unsafe { *ptr.assume_init_mut() = value },
            );
        }
    }

    /// Specifies whether the main diagonal should be traversed, when iterating over a triangular
    /// chunk of the matrix.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum Diag {
        /// Do not include diagonal of matrix
        Skip,
        /// Include diagonal of matrix
        Include,
    }

    #[derive(Copy, Clone)]
    pub enum MatLayoutTransform {
        None,
        ReverseRows,
        Transpose,
        TransposeReverseRows,
    }

    #[derive(Copy, Clone)]
    pub enum VecLayoutTransform {
        None,
        Reverse,
    }

    pub trait MatShape {
        type Rows: Copy + Eq;
        type Cols: Copy + Eq;
        fn nrows(&self) -> Self::Rows;
        fn ncols(&self) -> Self::Cols;
    }

    pub unsafe trait MaybeContiguous: MatShape {
        type Index: Copy;
        type Slice;
        type LayoutTransform: Copy;
        unsafe fn get_slice_unchecked(&mut self, idx: Self::Index, n_elems: usize) -> Self::Slice;
    }

    pub unsafe trait MatIndex<'a, _Outlives = &'a Self>: MaybeContiguous {
        type Item;

        unsafe fn get_unchecked(&'a mut self, index: Self::Index) -> Self::Item;
        unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item;

        fn is_contiguous(&self) -> bool;
        fn preferred_layout(&self) -> Self::LayoutTransform;
        fn with_layout(self, layout: Self::LayoutTransform) -> Self;
    }

    #[derive(Copy, Clone, Debug)]
    pub struct Last<Mat>(pub Mat);

    #[derive(Copy, Clone, Debug)]
    pub struct Zip<Head, Tail>(pub Head, pub Tail);

    #[derive(Copy, Clone, Debug)]
    pub struct LastEq<Rows, Cols, Mat: MatShape<Rows = Rows, Cols = Cols>>(pub Mat);
    #[derive(Copy, Clone, Debug)]
    pub struct ZipEq<
        Rows,
        Cols,
        Head: MatShape<Rows = Rows, Cols = Cols>,
        Tail: MatShape<Rows = Rows, Cols = Cols>,
    >(Head, Tail);

    impl<
            Rows: Copy + Eq,
            Cols: Copy + Eq,
            Head: MatShape<Rows = Rows, Cols = Cols>,
            Tail: MatShape<Rows = Rows, Cols = Cols>,
        > ZipEq<Rows, Cols, Head, Tail>
    {
        #[inline(always)]
        pub fn new(head: Head, tail: Tail) -> Self {
            assert!((head.nrows(), head.ncols()) == (tail.nrows(), tail.ncols()));
            Self(head, tail)
        }

        #[inline(always)]
        pub fn new_unchecked(head: Head, tail: Tail) -> Self {
            debug_assert!((head.nrows(), head.ncols()) == (tail.nrows(), tail.ncols()));
            Self(head, tail)
        }
    }

    impl<Rows: Copy + Eq, Cols: Copy + Eq, Mat: MatShape<Rows = Rows, Cols = Cols>> MatShape
        for LastEq<Rows, Cols, Mat>
    {
        type Rows = Rows;
        type Cols = Cols;
        #[inline(always)]
        fn nrows(&self) -> Self::Rows {
            self.0.nrows()
        }
        #[inline(always)]
        fn ncols(&self) -> Self::Cols {
            self.0.ncols()
        }
    }

    impl<
            Rows: Copy + Eq,
            Cols: Copy + Eq,
            Head: MatShape<Rows = Rows, Cols = Cols>,
            Tail: MatShape<Rows = Rows, Cols = Cols>,
        > MatShape for ZipEq<Rows, Cols, Head, Tail>
    {
        type Rows = Rows;
        type Cols = Cols;
        #[inline(always)]
        fn nrows(&self) -> Self::Rows {
            self.0.nrows()
        }
        #[inline(always)]
        fn ncols(&self) -> Self::Cols {
            self.0.ncols()
        }
    }

    impl<E: Entity> MatShape for ColRef<'_, E> {
        type Rows = usize;
        type Cols = ();
        #[inline(always)]
        fn nrows(&self) -> Self::Rows {
            (*self).nrows()
        }
        #[inline(always)]
        fn ncols(&self) -> Self::Cols {
            ()
        }
    }

    impl<E: Entity> MatShape for ColMut<'_, E> {
        type Rows = usize;
        type Cols = ();
        #[inline(always)]
        fn nrows(&self) -> Self::Rows {
            (*self).nrows()
        }
        #[inline(always)]
        fn ncols(&self) -> Self::Cols {
            ()
        }
    }

    impl<E: Entity> MatShape for RowRef<'_, E> {
        type Rows = ();
        type Cols = usize;
        #[inline(always)]
        fn nrows(&self) -> Self::Rows {
            ()
        }
        #[inline(always)]
        fn ncols(&self) -> Self::Cols {
            (*self).ncols()
        }
    }
    impl<E: Entity> MatShape for RowMut<'_, E> {
        type Rows = ();
        type Cols = usize;
        #[inline(always)]
        fn nrows(&self) -> Self::Rows {
            ()
        }
        #[inline(always)]
        fn ncols(&self) -> Self::Cols {
            (*self).ncols()
        }
    }

    impl<E: Entity> MatShape for MatRef<'_, E> {
        type Rows = usize;
        type Cols = usize;
        #[inline(always)]
        fn nrows(&self) -> Self::Rows {
            (*self).nrows()
        }
        #[inline(always)]
        fn ncols(&self) -> Self::Cols {
            (*self).ncols()
        }
    }

    impl<E: Entity> MatShape for MatMut<'_, E> {
        type Rows = usize;
        type Cols = usize;
        #[inline(always)]
        fn nrows(&self) -> Self::Rows {
            (*self).nrows()
        }
        #[inline(always)]
        fn ncols(&self) -> Self::Cols {
            (*self).ncols()
        }
    }

    unsafe impl<Rows: Copy + Eq, Cols: Copy + Eq, Mat: MaybeContiguous<Rows = Rows, Cols = Cols>>
        MaybeContiguous for LastEq<Rows, Cols, Mat>
    {
        type Index = Mat::Index;
        type Slice = Last<Mat::Slice>;
        type LayoutTransform = Mat::LayoutTransform;
        #[inline(always)]
        unsafe fn get_slice_unchecked(&mut self, idx: Self::Index, n_elems: usize) -> Self::Slice {
            Last(self.0.get_slice_unchecked(idx, n_elems))
        }
    }

    unsafe impl<'a, Rows: Copy + Eq, Cols: Copy + Eq, Mat: MatIndex<'a, Rows = Rows, Cols = Cols>>
        MatIndex<'a> for LastEq<Rows, Cols, Mat>
    {
        type Item = Last<Mat::Item>;

        #[inline(always)]
        unsafe fn get_unchecked(&'a mut self, index: Self::Index) -> Self::Item {
            Last(self.0.get_unchecked(index))
        }

        #[inline(always)]
        unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
            Last(Mat::get_from_slice_unchecked(&mut slice.0, idx))
        }

        #[inline(always)]
        fn is_contiguous(&self) -> bool {
            self.0.is_contiguous()
        }
        #[inline(always)]
        fn preferred_layout(&self) -> Self::LayoutTransform {
            self.0.preferred_layout()
        }
        #[inline(always)]
        fn with_layout(self, layout: Self::LayoutTransform) -> Self {
            Self(self.0.with_layout(layout))
        }
    }

    unsafe impl<
            Rows: Copy + Eq,
            Cols: Copy + Eq,
            Head: MaybeContiguous<Rows = Rows, Cols = Cols>,
            Tail: MaybeContiguous<
                Rows = Rows,
                Cols = Cols,
                Index = Head::Index,
                LayoutTransform = Head::LayoutTransform,
            >,
        > MaybeContiguous for ZipEq<Rows, Cols, Head, Tail>
    {
        type Index = Head::Index;
        type Slice = Zip<Head::Slice, Tail::Slice>;
        type LayoutTransform = Head::LayoutTransform;
        #[inline(always)]
        unsafe fn get_slice_unchecked(&mut self, idx: Self::Index, n_elems: usize) -> Self::Slice {
            Zip(
                self.0.get_slice_unchecked(idx, n_elems),
                self.1.get_slice_unchecked(idx, n_elems),
            )
        }
    }

    unsafe impl<
            'a,
            Rows: Copy + Eq,
            Cols: Copy + Eq,
            Head: MatIndex<'a, Rows = Rows, Cols = Cols>,
            Tail: MatIndex<
                'a,
                Rows = Rows,
                Cols = Cols,
                Index = Head::Index,
                LayoutTransform = Head::LayoutTransform,
            >,
        > MatIndex<'a> for ZipEq<Rows, Cols, Head, Tail>
    {
        type Item = Zip<Head::Item, Tail::Item>;

        #[inline(always)]
        unsafe fn get_unchecked(&'a mut self, index: Self::Index) -> Self::Item {
            Zip(self.0.get_unchecked(index), self.1.get_unchecked(index))
        }

        #[inline(always)]
        unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
            Zip(
                Head::get_from_slice_unchecked(&mut slice.0, idx),
                Tail::get_from_slice_unchecked(&mut slice.1, idx),
            )
        }

        #[inline(always)]
        fn is_contiguous(&self) -> bool {
            self.0.is_contiguous() && self.1.is_contiguous()
        }
        #[inline(always)]
        fn preferred_layout(&self) -> Self::LayoutTransform {
            self.0.preferred_layout()
        }
        #[inline(always)]
        fn with_layout(self, layout: Self::LayoutTransform) -> Self {
            ZipEq(self.0.with_layout(layout), self.1.with_layout(layout))
        }
    }

    unsafe impl<E: Entity> MaybeContiguous for ColRef<'_, E> {
        type Index = (usize, ());
        type Slice = GroupFor<E, &'static [MaybeUninit<E::Unit>]>;
        type LayoutTransform = VecLayoutTransform;

        #[inline(always)]
        unsafe fn get_slice_unchecked(
            &mut self,
            (i, _): Self::Index,
            n_elems: usize,
        ) -> Self::Slice {
            E::faer_map(
                (*self).rb().ptr_at(i),
                #[inline(always)]
                |ptr| core::slice::from_raw_parts(ptr as *const MaybeUninit<E::Unit>, n_elems),
            )
        }
    }
    unsafe impl<'a, E: Entity> MatIndex<'a> for ColRef<'_, E> {
        type Item = Read<'a, E>;

        #[inline(always)]
        unsafe fn get_unchecked(&'a mut self, (i, _): Self::Index) -> Self::Item {
            Read {
                ptr: E::faer_map(
                    self.rb().ptr_inbounds_at(i),
                    #[inline(always)]
                    |ptr| &*(ptr as *const MaybeUninit<E::Unit>),
                ),
            }
        }

        #[inline(always)]
        unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
            let slice = E::faer_rb(E::faer_as_ref(slice));
            Read {
                ptr: E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.get_unchecked(idx),
                ),
            }
        }

        #[inline(always)]
        fn is_contiguous(&self) -> bool {
            self.row_stride() == 1
        }
        #[inline(always)]
        fn preferred_layout(&self) -> Self::LayoutTransform {
            let rs = self.row_stride();
            if self.nrows() > 1 && rs == 1 {
                VecLayoutTransform::None
            } else if self.nrows() > 1 && rs == -1 {
                VecLayoutTransform::Reverse
            } else {
                VecLayoutTransform::None
            }
        }
        #[inline(always)]
        fn with_layout(self, layout: Self::LayoutTransform) -> Self {
            use VecLayoutTransform::*;
            match layout {
                None => self,
                Reverse => self.reverse_rows(),
            }
        }
    }

    unsafe impl<E: Entity> MaybeContiguous for ColMut<'_, E> {
        type Index = (usize, ());
        type Slice = GroupFor<E, &'static mut [MaybeUninit<E::Unit>]>;
        type LayoutTransform = VecLayoutTransform;

        #[inline(always)]
        unsafe fn get_slice_unchecked(
            &mut self,
            (i, _): Self::Index,
            n_elems: usize,
        ) -> Self::Slice {
            E::faer_map(
                (*self).rb_mut().ptr_at_mut(i),
                #[inline(always)]
                |ptr| core::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<E::Unit>, n_elems),
            )
        }
    }
    unsafe impl<'a, E: Entity> MatIndex<'a> for ColMut<'_, E> {
        type Item = ReadWrite<'a, E>;

        #[inline(always)]
        unsafe fn get_unchecked(&'a mut self, (i, _): Self::Index) -> Self::Item {
            ReadWrite {
                ptr: E::faer_map(
                    self.rb_mut().ptr_inbounds_at_mut(i),
                    #[inline(always)]
                    |ptr| &mut *(ptr as *mut MaybeUninit<E::Unit>),
                ),
            }
        }

        #[inline(always)]
        unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
            let slice = E::faer_rb_mut(E::faer_as_mut(slice));
            ReadWrite {
                ptr: E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.get_unchecked_mut(idx),
                ),
            }
        }

        #[inline(always)]
        fn is_contiguous(&self) -> bool {
            self.row_stride() == 1
        }
        #[inline(always)]
        fn preferred_layout(&self) -> Self::LayoutTransform {
            let rs = self.row_stride();
            if self.nrows() > 1 && rs == 1 {
                VecLayoutTransform::None
            } else if self.nrows() > 1 && rs == -1 {
                VecLayoutTransform::Reverse
            } else {
                VecLayoutTransform::None
            }
        }
        #[inline(always)]
        fn with_layout(self, layout: Self::LayoutTransform) -> Self {
            use VecLayoutTransform::*;
            match layout {
                None => self,
                Reverse => self.reverse_rows_mut(),
            }
        }
    }

    unsafe impl<E: Entity> MaybeContiguous for RowRef<'_, E> {
        type Index = ((), usize);
        type Slice = GroupFor<E, &'static [MaybeUninit<E::Unit>]>;
        type LayoutTransform = VecLayoutTransform;

        #[inline(always)]
        unsafe fn get_slice_unchecked(
            &mut self,
            (_, j): Self::Index,
            n_elems: usize,
        ) -> Self::Slice {
            E::faer_map(
                (*self).rb().ptr_at(j),
                #[inline(always)]
                |ptr| core::slice::from_raw_parts(ptr as *const MaybeUninit<E::Unit>, n_elems),
            )
        }
    }
    unsafe impl<'a, E: Entity> MatIndex<'a> for RowRef<'_, E> {
        type Item = Read<'a, E>;

        #[inline(always)]
        unsafe fn get_unchecked(&'a mut self, (_, j): Self::Index) -> Self::Item {
            Read {
                ptr: E::faer_map(
                    self.rb().ptr_inbounds_at(j),
                    #[inline(always)]
                    |ptr| &*(ptr as *const MaybeUninit<E::Unit>),
                ),
            }
        }

        #[inline(always)]
        unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
            let slice = E::faer_rb(E::faer_as_ref(slice));
            Read {
                ptr: E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.get_unchecked(idx),
                ),
            }
        }

        #[inline(always)]
        fn is_contiguous(&self) -> bool {
            self.col_stride() == 1
        }
        #[inline(always)]
        fn preferred_layout(&self) -> Self::LayoutTransform {
            let cs = self.col_stride();
            if self.ncols() > 1 && cs == 1 {
                VecLayoutTransform::None
            } else if self.ncols() > 1 && cs == -1 {
                VecLayoutTransform::Reverse
            } else {
                VecLayoutTransform::None
            }
        }
        #[inline(always)]
        fn with_layout(self, layout: Self::LayoutTransform) -> Self {
            use VecLayoutTransform::*;
            match layout {
                None => self,
                Reverse => self.reverse_cols(),
            }
        }
    }

    unsafe impl<E: Entity> MaybeContiguous for RowMut<'_, E> {
        type Index = ((), usize);
        type Slice = GroupFor<E, &'static mut [MaybeUninit<E::Unit>]>;
        type LayoutTransform = VecLayoutTransform;

        #[inline(always)]
        unsafe fn get_slice_unchecked(
            &mut self,
            (_, j): Self::Index,
            n_elems: usize,
        ) -> Self::Slice {
            E::faer_map(
                (*self).rb_mut().ptr_at_mut(j),
                #[inline(always)]
                |ptr| core::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<E::Unit>, n_elems),
            )
        }
    }
    unsafe impl<'a, E: Entity> MatIndex<'a> for RowMut<'_, E> {
        type Item = ReadWrite<'a, E>;

        #[inline(always)]
        unsafe fn get_unchecked(&'a mut self, (_, j): Self::Index) -> Self::Item {
            ReadWrite {
                ptr: E::faer_map(
                    self.rb_mut().ptr_inbounds_at_mut(j),
                    #[inline(always)]
                    |ptr| &mut *(ptr as *mut MaybeUninit<E::Unit>),
                ),
            }
        }

        #[inline(always)]
        unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
            let slice = E::faer_rb_mut(E::faer_as_mut(slice));
            ReadWrite {
                ptr: E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.get_unchecked_mut(idx),
                ),
            }
        }

        #[inline(always)]
        fn is_contiguous(&self) -> bool {
            self.col_stride() == 1
        }
        #[inline(always)]
        fn preferred_layout(&self) -> Self::LayoutTransform {
            let cs = self.col_stride();
            if self.ncols() > 1 && cs == 1 {
                VecLayoutTransform::None
            } else if self.ncols() > 1 && cs == -1 {
                VecLayoutTransform::Reverse
            } else {
                VecLayoutTransform::None
            }
        }
        #[inline(always)]
        fn with_layout(self, layout: Self::LayoutTransform) -> Self {
            use VecLayoutTransform::*;
            match layout {
                None => self,
                Reverse => self.reverse_cols_mut(),
            }
        }
    }

    unsafe impl<E: Entity> MaybeContiguous for MatRef<'_, E> {
        type Index = (usize, usize);
        type Slice = GroupFor<E, &'static [MaybeUninit<E::Unit>]>;
        type LayoutTransform = MatLayoutTransform;

        #[inline(always)]
        unsafe fn get_slice_unchecked(
            &mut self,
            (i, j): Self::Index,
            n_elems: usize,
        ) -> Self::Slice {
            E::faer_map(
                (*self).rb().overflowing_ptr_at(i, j),
                #[inline(always)]
                |ptr| core::slice::from_raw_parts(ptr as *const MaybeUninit<E::Unit>, n_elems),
            )
        }
    }
    unsafe impl<'a, E: Entity> MatIndex<'a> for MatRef<'_, E> {
        type Item = Read<'a, E>;

        #[inline(always)]
        unsafe fn get_unchecked(&'a mut self, (i, j): Self::Index) -> Self::Item {
            Read {
                ptr: E::faer_map(
                    self.rb().ptr_inbounds_at(i, j),
                    #[inline(always)]
                    |ptr| &*(ptr as *const MaybeUninit<E::Unit>),
                ),
            }
        }

        #[inline(always)]
        unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
            let slice = E::faer_rb(E::faer_as_ref(slice));
            Read {
                ptr: E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.get_unchecked(idx),
                ),
            }
        }

        #[inline(always)]
        fn is_contiguous(&self) -> bool {
            self.row_stride() == 1
        }
        #[inline(always)]
        fn preferred_layout(&self) -> Self::LayoutTransform {
            let rs = self.row_stride();
            let cs = self.col_stride();
            if self.nrows() > 1 && rs == 1 {
                MatLayoutTransform::None
            } else if self.nrows() > 1 && rs == -1 {
                MatLayoutTransform::ReverseRows
            } else if self.ncols() > 1 && cs == 1 {
                MatLayoutTransform::Transpose
            } else if self.ncols() > 1 && cs == -1 {
                MatLayoutTransform::TransposeReverseRows
            } else {
                MatLayoutTransform::None
            }
        }
        #[inline(always)]
        fn with_layout(self, layout: Self::LayoutTransform) -> Self {
            use MatLayoutTransform::*;
            match layout {
                None => self,
                ReverseRows => self.reverse_rows(),
                Transpose => self.transpose(),
                TransposeReverseRows => self.transpose().reverse_rows(),
            }
        }
    }

    unsafe impl<E: Entity> MaybeContiguous for MatMut<'_, E> {
        type Index = (usize, usize);
        type Slice = GroupFor<E, &'static mut [MaybeUninit<E::Unit>]>;
        type LayoutTransform = MatLayoutTransform;

        #[inline(always)]
        unsafe fn get_slice_unchecked(
            &mut self,
            (i, j): Self::Index,
            n_elems: usize,
        ) -> Self::Slice {
            E::faer_map(
                (*self).rb().overflowing_ptr_at(i, j),
                #[inline(always)]
                |ptr| core::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<E::Unit>, n_elems),
            )
        }
    }

    unsafe impl<'a, E: Entity> MatIndex<'a> for MatMut<'_, E> {
        type Item = ReadWrite<'a, E>;

        #[inline(always)]
        unsafe fn get_unchecked(&'a mut self, (i, j): Self::Index) -> Self::Item {
            ReadWrite {
                ptr: E::faer_map(
                    self.rb_mut().ptr_inbounds_at_mut(i, j),
                    #[inline(always)]
                    |ptr| &mut *(ptr as *mut MaybeUninit<E::Unit>),
                ),
            }
        }

        #[inline(always)]
        unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
            let slice = E::faer_rb_mut(E::faer_as_mut(slice));
            ReadWrite {
                ptr: E::faer_map(
                    slice,
                    #[inline(always)]
                    |slice| slice.get_unchecked_mut(idx),
                ),
            }
        }

        #[inline(always)]
        fn is_contiguous(&self) -> bool {
            self.row_stride() == 1
        }
        #[inline(always)]
        fn preferred_layout(&self) -> Self::LayoutTransform {
            let rs = self.row_stride();
            let cs = self.col_stride();
            if self.nrows() > 1 && rs == 1 {
                MatLayoutTransform::None
            } else if self.nrows() > 1 && rs == -1 {
                MatLayoutTransform::ReverseRows
            } else if self.ncols() > 1 && cs == 1 {
                MatLayoutTransform::Transpose
            } else if self.ncols() > 1 && cs == -1 {
                MatLayoutTransform::TransposeReverseRows
            } else {
                MatLayoutTransform::None
            }
        }
        #[inline(always)]
        fn with_layout(self, layout: Self::LayoutTransform) -> Self {
            use MatLayoutTransform::*;
            match layout {
                None => self,
                ReverseRows => self.reverse_rows_mut(),
                Transpose => self.transpose_mut(),
                TransposeReverseRows => self.transpose_mut().reverse_rows_mut(),
            }
        }
    }

    #[inline(always)]
    fn annotate_noalias_mat<Z: for<'a> MatIndex<'a>>(
        f: &mut impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
        mut slice: Z::Slice,
        i_begin: usize,
        i_end: usize,
        _j: usize,
    ) {
        for i in i_begin..i_end {
            unsafe { f(Z::get_from_slice_unchecked(&mut slice, i - i_begin)) };
        }
    }

    #[inline(always)]
    fn annotate_noalias_col<Z: for<'a> MatIndex<'a>>(
        f: &mut impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
        mut slice: Z::Slice,
        i_begin: usize,
        i_end: usize,
    ) {
        for i in i_begin..i_end {
            unsafe { f(Z::get_from_slice_unchecked(&mut slice, i - i_begin)) };
        }
    }

    #[inline(always)]
    fn for_each_mat<Z: for<'a> MatIndex<'a, Rows = usize, Cols = usize, Index = (usize, usize)>>(
        z: Z,
        mut f: impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
    ) {
        let layout = z.preferred_layout();
        let mut z = z.with_layout(layout);

        let m = z.nrows();
        let n = z.ncols();
        if m == 0 || n == 0 {
            return;
        }

        unsafe {
            if z.is_contiguous() {
                for j in 0..n {
                    annotate_noalias_mat::<Z>(&mut f, z.get_slice_unchecked((0, j), m), 0, m, j);
                }
            } else {
                for j in 0..n {
                    for i in 0..m {
                        f(z.get_unchecked((i, j)))
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn for_each_mat_triangular_lower<
        Z: for<'a> MatIndex<
            'a,
            Rows = usize,
            Cols = usize,
            Index = (usize, usize),
            LayoutTransform = MatLayoutTransform,
        >,
    >(
        z: Z,
        diag: Diag,
        transpose: bool,
        mut f: impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
    ) {
        use MatLayoutTransform::*;

        let z = if transpose {
            z.with_layout(MatLayoutTransform::Transpose)
        } else {
            z
        };
        let layout = z.preferred_layout();
        let mut z = z.with_layout(layout);

        let m = z.nrows();
        let n = z.ncols();
        let n = match layout {
            None | ReverseRows => Ord::min(m, n),
            Transpose | TransposeReverseRows => n,
        };
        if m == 0 || n == 0 {
            return;
        }

        let strict = match diag {
            Diag::Skip => true,
            Diag::Include => false,
        };

        unsafe {
            if z.is_contiguous() {
                for j in 0..n {
                    let (start, end) = match layout {
                        None => (j + strict as usize, m),
                        ReverseRows => (0, (m - (j + strict as usize))),
                        Transpose => (0, (j + !strict as usize).min(m)),
                        TransposeReverseRows => (m - ((j + !strict as usize).min(m)), m),
                    };

                    let len = end - start;

                    annotate_noalias_mat::<Z>(
                        &mut f,
                        z.get_slice_unchecked((start, j), len),
                        start,
                        end,
                        j,
                    );
                }
            } else {
                for j in 0..n {
                    let (start, end) = match layout {
                        None => (j + strict as usize, m),
                        ReverseRows => (0, (m - (j + strict as usize))),
                        Transpose => (0, (j + !strict as usize).min(m)),
                        TransposeReverseRows => (m - ((j + !strict as usize).min(m)), m),
                    };

                    for i in start..end {
                        f(z.get_unchecked((i, j)))
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn for_each_col<Z: for<'a> MatIndex<'a, Rows = usize, Cols = (), Index = (usize, ())>>(
        z: Z,
        mut f: impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
    ) {
        let layout = z.preferred_layout();
        let mut z = z.with_layout(layout);

        let m = z.nrows();
        if m == 0 {
            return;
        }

        unsafe {
            if z.is_contiguous() {
                annotate_noalias_col::<Z>(&mut f, z.get_slice_unchecked((0, ()), m), 0, m);
            } else {
                for i in 0..m {
                    f(z.get_unchecked((i, ())))
                }
            }
        }
    }

    #[inline(always)]
    fn for_each_row<Z: for<'a> MatIndex<'a, Rows = (), Cols = usize, Index = ((), usize)>>(
        z: Z,
        mut f: impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
    ) {
        let layout = z.preferred_layout();
        let mut z = z.with_layout(layout);

        let n = z.ncols();
        if n == 0 {
            return;
        }

        unsafe {
            if z.is_contiguous() {
                annotate_noalias_col::<Z>(&mut f, z.get_slice_unchecked(((), 0), n), 0, n);
            } else {
                for j in 0..n {
                    f(z.get_unchecked(((), j)))
                }
            }
        }
    }

    impl<
            M: for<'a> MatIndex<
                'a,
                Rows = usize,
                Cols = usize,
                Index = (usize, usize),
                LayoutTransform = MatLayoutTransform,
            >,
        > LastEq<usize, usize, M>
    {
        #[inline(always)]
        pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
            for_each_mat(self, f);
        }

        #[inline(always)]
        pub fn for_each_triangular_lower(
            self,
            diag: Diag,
            f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item),
        ) {
            for_each_mat_triangular_lower(self, diag, false, f);
        }

        #[inline(always)]
        pub fn for_each_triangular_upper(
            self,
            diag: Diag,
            f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item),
        ) {
            for_each_mat_triangular_lower(self, diag, true, f);
        }

        #[inline(always)]
        pub fn map<E: Entity>(
            self,
            f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
        ) -> Mat<E> {
            let (m, n) = (self.nrows(), self.ncols());
            let mut out = Mat::<E>::with_capacity(m, n);
            let rs = 1;
            let cs = out.col_stride();
            let out_view =
                unsafe { mat::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), m, n, rs, cs) };
            let mut f = f;
            ZipEq::new(out_view, self).for_each(
                #[inline(always)]
                |Zip(mut out, item)| out.write(f(item)),
            );
            unsafe { out.set_dims(m, n) };
            out
        }
    }

    impl<
            M: for<'a> MatIndex<
                'a,
                Rows = (),
                Cols = usize,
                Index = ((), usize),
                LayoutTransform = VecLayoutTransform,
            >,
        > LastEq<(), usize, M>
    {
        #[inline(always)]
        pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
            for_each_row(self, f);
        }

        #[inline(always)]
        pub fn map<E: Entity>(
            self,
            f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
        ) -> Row<E> {
            let (_, n) = (self.nrows(), self.ncols());
            let mut out = Row::<E>::with_capacity(n);
            let out_view = unsafe { row::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), n, 1) };
            let mut f = f;
            ZipEq::new(out_view, self).for_each(
                #[inline(always)]
                |Zip(mut out, item)| out.write(f(item)),
            );
            unsafe { out.set_ncols(n) };
            out
        }
    }

    impl<
            M: for<'a> MatIndex<
                'a,
                Rows = usize,
                Cols = (),
                Index = (usize, ()),
                LayoutTransform = VecLayoutTransform,
            >,
        > LastEq<usize, (), M>
    {
        #[inline(always)]
        pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
            for_each_col(self, f);
        }

        #[inline(always)]
        pub fn map<E: Entity>(
            self,
            f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
        ) -> Col<E> {
            let (m, _) = (self.nrows(), self.ncols());
            let mut out = Col::<E>::with_capacity(m);
            let out_view = unsafe { col::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), m, 1) };
            let mut f = f;
            ZipEq::new(out_view, self).for_each(
                #[inline(always)]
                |Zip(mut out, item)| out.write(f(item)),
            );
            unsafe { out.set_nrows(m) };
            out
        }
    }

    impl<
            Head: for<'a> MatIndex<
                'a,
                Rows = (),
                Cols = usize,
                Index = ((), usize),
                LayoutTransform = VecLayoutTransform,
            >,
            Tail: for<'a> MatIndex<
                'a,
                Rows = (),
                Cols = usize,
                Index = ((), usize),
                LayoutTransform = VecLayoutTransform,
            >,
        > ZipEq<(), usize, Head, Tail>
    {
        #[inline(always)]
        pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
            for_each_row(self, f);
        }

        #[inline(always)]
        pub fn map<E: Entity>(
            self,
            f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
        ) -> Row<E> {
            let (_, n) = (self.nrows(), self.ncols());
            let mut out = Row::<E>::with_capacity(n);
            let out_view = unsafe { row::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), n, 1) };
            let mut f = f;
            ZipEq::new(out_view, self).for_each(
                #[inline(always)]
                |Zip(mut out, item)| out.write(f(item)),
            );
            unsafe { out.set_ncols(n) };
            out
        }
    }

    impl<
            Head: for<'a> MatIndex<
                'a,
                Rows = usize,
                Cols = (),
                Index = (usize, ()),
                LayoutTransform = VecLayoutTransform,
            >,
            Tail: for<'a> MatIndex<
                'a,
                Rows = usize,
                Cols = (),
                Index = (usize, ()),
                LayoutTransform = VecLayoutTransform,
            >,
        > ZipEq<usize, (), Head, Tail>
    {
        #[inline(always)]
        pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
            for_each_col(self, f);
        }

        #[inline(always)]
        pub fn map<E: Entity>(
            self,
            f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
        ) -> Col<E> {
            let (m, _) = (self.nrows(), self.ncols());
            let mut out = Col::<E>::with_capacity(m);
            let out_view = unsafe { col::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), m, 1) };
            let mut f = f;
            ZipEq::new(out_view, self).for_each(
                #[inline(always)]
                |Zip(mut out, item)| out.write(f(item)),
            );
            unsafe { out.set_nrows(m) };
            out
        }
    }

    impl<
            Head: for<'a> MatIndex<
                'a,
                Rows = usize,
                Cols = usize,
                Index = (usize, usize),
                LayoutTransform = MatLayoutTransform,
            >,
            Tail: for<'a> MatIndex<
                'a,
                Rows = usize,
                Cols = usize,
                Index = (usize, usize),
                LayoutTransform = MatLayoutTransform,
            >,
        > ZipEq<usize, usize, Head, Tail>
    {
        #[inline(always)]
        pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
            for_each_mat(self, f);
        }

        #[inline(always)]
        pub fn for_each_triangular_lower(
            self,
            diag: Diag,
            f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item),
        ) {
            for_each_mat_triangular_lower(self, diag, false, f);
        }

        #[inline(always)]
        pub fn for_each_triangular_upper(
            self,
            diag: Diag,
            f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item),
        ) {
            for_each_mat_triangular_lower(self, diag, true, f);
        }

        #[inline(always)]
        pub fn map<E: Entity>(
            self,
            f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
        ) -> Mat<E> {
            let (m, n) = (self.nrows(), self.ncols());
            let mut out = Mat::<E>::with_capacity(m, n);
            let rs = 1;
            let cs = out.col_stride();
            let out_view =
                unsafe { mat::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), m, n, rs, cs) };
            let mut f = f;
            ZipEq::new(out_view, self).for_each(
                #[inline(always)]
                |Zip(mut out, item)| out.write(f(item)),
            );
            unsafe { out.set_dims(m, n) };
            out
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{assert, unzipped, zipped, ComplexField, Mat};

        #[test]
        fn test_zip() {
            for (m, n) in [(2, 2), (4, 2), (2, 4)] {
                for rev_dst in [false, true] {
                    for rev_src in [false, true] {
                        for transpose_dst in [false, true] {
                            for transpose_src in [false, true] {
                                for diag in [Diag::Include, Diag::Skip] {
                                    let mut dst = Mat::from_fn(
                                        if transpose_dst { n } else { m },
                                        if transpose_dst { m } else { n },
                                        |_, _| f64::faer_zero(),
                                    );
                                    let src = Mat::from_fn(
                                        if transpose_src { n } else { m },
                                        if transpose_src { m } else { n },
                                        |_, _| f64::faer_one(),
                                    );

                                    let mut target = Mat::from_fn(m, n, |_, _| f64::faer_zero());
                                    let target_src = Mat::from_fn(m, n, |_, _| f64::faer_one());

                                    zipped!(target.as_mut(), target_src.as_ref())
                                        .for_each_triangular_lower(
                                            diag,
                                            |unzipped!(mut dst, src)| dst.write(src.read()),
                                        );

                                    let mut dst = dst.as_mut();
                                    let mut src = src.as_ref();

                                    if transpose_dst {
                                        dst = dst.transpose_mut();
                                    }
                                    if rev_dst {
                                        dst = dst.reverse_rows_mut();
                                    }

                                    if transpose_src {
                                        src = src.transpose();
                                    }
                                    if rev_src {
                                        src = src.reverse_rows();
                                    }

                                    zipped!(dst.rb_mut(), src).for_each_triangular_lower(
                                        diag,
                                        |unzipped!(mut dst, src)| dst.write(src.read()),
                                    );

                                    assert!(dst.rb() == target.as_ref());
                                }
                            }
                        }
                    }
                }
            }

            {
                let m = 3;
                for rev_dst in [false, true] {
                    for rev_src in [false, true] {
                        let mut dst = Col::<f64>::zeros(m);
                        let src = Col::from_fn(m, |i| (i + 1) as f64);

                        let mut target = Col::<f64>::zeros(m);
                        let target_src =
                            Col::from_fn(m, |i| if rev_src { m - i } else { i + 1 } as f64);

                        zipped!(target.as_mut(), target_src.as_ref())
                            .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));

                        let mut dst = dst.as_mut();
                        let mut src = src.as_ref();

                        if rev_dst {
                            dst = dst.reverse_rows_mut();
                        }
                        if rev_src {
                            src = src.reverse_rows();
                        }

                        zipped!(dst.rb_mut(), src)
                            .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));

                        assert!(dst.rb() == target.as_ref());
                    }
                }
            }

            {
                let m = 3;
                for rev_dst in [false, true] {
                    for rev_src in [false, true] {
                        let mut dst = Row::<f64>::zeros(m);
                        let src = Row::from_fn(m, |i| (i + 1) as f64);

                        let mut target = Row::<f64>::zeros(m);
                        let target_src =
                            Row::from_fn(m, |i| if rev_src { m - i } else { i + 1 } as f64);

                        zipped!(target.as_mut(), target_src.as_ref())
                            .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));

                        let mut dst = dst.as_mut();
                        let mut src = src.as_ref();

                        if rev_dst {
                            dst = dst.reverse_cols_mut();
                        }
                        if rev_src {
                            src = src.reverse_cols();
                        }

                        zipped!(&mut dst, src)
                            .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));

                        assert!(dst.rb() == target.as_ref());
                    }
                }
            }
        }
    }
}
