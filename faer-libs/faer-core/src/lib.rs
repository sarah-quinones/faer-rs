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
//! In practice, this means that for a `Mat<f64>`, methods such as [`Mat::col_ref`] will return a
//! `&[f64]`. Meanwhile, for a `Mat<Complex<f64>>`, [`Mat::col_ref`] will return `Complex<&[f64]>`,
//! which holds two slices, each pointing respectively to a view over the real and the imaginary
//! components.
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

#[cfg(feature = "std")]
use assert2::{assert, debug_assert};
use coe::Coerce;
use core::{
    fmt::Debug, marker::PhantomData, mem::ManuallyDrop, ptr::NonNull, sync::atomic::AtomicUsize,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use num_complex::Complex;
use pulp::{cast, Simd};
use reborrow::*;
use zip::Zip;

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
pub mod zip;

pub mod matrix_ops;

/// Thin wrapper used for scalar multiplication of a matrix by a scalar value.
pub use matrix_ops::scale;

#[doc(hidden)]
pub mod simd;

#[doc(hidden)]
pub use faer_entity::transmute_unchecked;

/// Native complex floating point types whose real and imaginary parts are stored contiguously.
///
/// The types [`c32`] and [`c64`] respectively have the same layout as [`num_complex::Complex32`]
/// and [`num_complex::Complex64`].
///
/// They differ in the way they are treated by the `faer` library: When stored in a matrix,
/// `Mat<c32>` and `Mat<c64>` internally contain a single container of contiguously stored
/// `c32` and `c64` values, whereas `Mat<num_complex::Complex32>` and
/// `Mat<num_complex::Complex64>` internally contain two containers, separately storing the real
/// and imaginary parts of the complex values.
///
/// Matrix operations using `c32` and `c64` are usually more efficient and should be preferred in
/// most cases. `num_complex::Complex` matrices have better support for generic data types.
///
/// The drawing below represents a simplified layout of the `Mat` structure for each of `c32` and
/// `num_complex::Complex32`.
///
/// ```notcode
/// ┌──────────────────┐
/// │ Mat<c32>         │
/// ├──────────────────┤
/// │ ptr: *mut c32 ─ ─│─ ─ ─ ─ ┐
/// │ nrows: usize     │   ┌─────────┐
/// │ ncols: usize     │   │ z0: c32 │
/// │        ...       │   │ z1: c32 │
/// └──────────────────┘   │ z2: c32 │
///                        │   ...   │
///                        └─────────┘
///
/// ┌───────────────────────┐
/// │ Mat<Complex32>        │
/// ├───────────────────────┤
/// │ ptr_real: *mut f32 ─ ─│─ ─ ─ ─ ┐
/// │ ptr_imag: *mut f32 ─ ─│─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ┐
/// │ nrows: usize          │   ┌──────────┐   ┌──────────┐
/// │ ncols: usize          │   │ re0: f32 │   │ im0: f32 │
/// │           ...         │   │ re1: f32 │   │ im1: f32 │
/// └───────────────────────┘   │ re2: f32 │   │ im2: f32 │
///                             │    ...   │   │    ...   │
///                             └──────────┘   └──────────┘
/// ```
pub mod complex_native {
    // 32-bit complex floating point type. See the module-level documentation for more details.
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone, PartialEq)]
    #[repr(C)]
    pub struct c32 {
        pub re: f32,
        pub im: f32,
    }

    // 64-bit complex floating point type. See the module-level documentation for more details.
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone, PartialEq)]
    #[repr(C)]
    pub struct c64 {
        pub re: f64,
        pub im: f64,
    }

    // 32-bit implicitly conjugated complex floating point type.
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone, PartialEq)]
    #[repr(C)]
    pub struct c32conj {
        pub re: f32,
        pub neg_im: f32,
    }

    // 64-bit implicitly conjugated complex floating point type.
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone, PartialEq)]
    #[repr(C)]
    pub struct c64conj {
        pub re: f64,
        pub neg_im: f64,
    }
}

pub use complex_native::*;

impl c32 {
    #[inline(always)]
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub fn abs(self) -> f32 {
        self.faer_abs()
    }
}
impl c64 {
    #[inline(always)]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub fn abs(self) -> f64 {
        self.faer_abs()
    }
}

impl core::ops::Neg for c32 {
    type Output = c32;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl core::ops::Add<f32> for c32 {
    type Output = c32;

    #[inline(always)]
    fn add(self, rhs: f32) -> Self::Output {
        Self::new(self.re + rhs, self.im)
    }
}
impl core::ops::Add<c32> for f32 {
    type Output = c32;

    #[inline(always)]
    fn add(self, rhs: c32) -> Self::Output {
        Self::Output::new(self + rhs.re, rhs.im)
    }
}
impl core::ops::Add for c32 {
    type Output = c32;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl core::ops::Sub<f32> for c32 {
    type Output = c32;

    #[inline(always)]
    fn sub(self, rhs: f32) -> Self::Output {
        Self::new(self.re - rhs, self.im)
    }
}
impl core::ops::Sub<c32> for f32 {
    type Output = c32;

    #[inline(always)]
    fn sub(self, rhs: c32) -> Self::Output {
        Self::Output::new(self - rhs.re, -rhs.im)
    }
}
impl core::ops::Sub for c32 {
    type Output = c32;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl core::ops::Mul<f32> for c32 {
    type Output = c32;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}
impl core::ops::Mul<c32> for f32 {
    type Output = c32;

    #[inline(always)]
    fn mul(self, rhs: c32) -> Self::Output {
        Self::Output::new(self * rhs.re, self * rhs.im)
    }
}
impl core::ops::Mul for c32 {
    type Output = c32;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl core::ops::Div<f32> for c32 {
    type Output = c32;

    #[inline(always)]
    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}
impl core::ops::Div<c32> for f32 {
    type Output = c32;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, rhs: c32) -> Self::Output {
        self * <c32 as ComplexField>::faer_inv(rhs)
    }
}
impl core::ops::Div for c32 {
    type Output = c32;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        self * <Self as ComplexField>::faer_inv(rhs)
    }
}

impl core::ops::Neg for c64 {
    type Output = c64;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl core::ops::Add<f64> for c64 {
    type Output = c64;

    #[inline(always)]
    fn add(self, rhs: f64) -> Self::Output {
        Self::new(self.re + rhs, self.im)
    }
}
impl core::ops::Add<c64> for f64 {
    type Output = c64;

    #[inline(always)]
    fn add(self, rhs: c64) -> Self::Output {
        Self::Output::new(self + rhs.re, rhs.im)
    }
}
impl core::ops::Add for c64 {
    type Output = c64;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl core::ops::Sub<f64> for c64 {
    type Output = c64;

    #[inline(always)]
    fn sub(self, rhs: f64) -> Self::Output {
        Self::new(self.re - rhs, self.im)
    }
}
impl core::ops::Sub<c64> for f64 {
    type Output = c64;

    #[inline(always)]
    fn sub(self, rhs: c64) -> Self::Output {
        Self::Output::new(self - rhs.re, -rhs.im)
    }
}
impl core::ops::Sub for c64 {
    type Output = c64;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl core::ops::Mul<f64> for c64 {
    type Output = c64;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}
impl core::ops::Mul<c64> for f64 {
    type Output = c64;

    #[inline(always)]
    fn mul(self, rhs: c64) -> Self::Output {
        Self::Output::new(self * rhs.re, self * rhs.im)
    }
}
impl core::ops::Mul for c64 {
    type Output = c64;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl core::ops::Div<f64> for c64 {
    type Output = c64;

    #[inline(always)]
    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}
impl core::ops::Div<c64> for f64 {
    type Output = c64;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, rhs: c64) -> Self::Output {
        self * <c64 as ComplexField>::faer_inv(rhs)
    }
}
impl core::ops::Div for c64 {
    type Output = c64;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        self * <Self as ComplexField>::faer_inv(rhs)
    }
}

impl From<c32> for num_complex::Complex32 {
    #[inline(always)]
    fn from(value: c32) -> Self {
        Self {
            re: value.re,
            im: value.im,
        }
    }
}
impl From<num_complex::Complex32> for c32 {
    #[inline(always)]
    fn from(value: num_complex::Complex32) -> Self {
        c32 {
            re: value.re,
            im: value.im,
        }
    }
}
impl From<c64> for num_complex::Complex64 {
    #[inline(always)]
    fn from(value: c64) -> Self {
        Self {
            re: value.re,
            im: value.im,
        }
    }
}
impl From<num_complex::Complex64> for c64 {
    #[inline(always)]
    fn from(value: num_complex::Complex64) -> Self {
        c64 {
            re: value.re,
            im: value.im,
        }
    }
}

unsafe impl bytemuck::Zeroable for c32 {}
unsafe impl bytemuck::Zeroable for c32conj {}
unsafe impl bytemuck::Zeroable for c64 {}
unsafe impl bytemuck::Zeroable for c64conj {}
unsafe impl bytemuck::Pod for c32 {}
unsafe impl bytemuck::Pod for c32conj {}
unsafe impl bytemuck::Pod for c64 {}
unsafe impl bytemuck::Pod for c64conj {}

impl Debug for c32 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.re.fmt(f)?;
        f.write_str(" + ")?;
        self.im.fmt(f)?;
        f.write_str(" * I")
    }
}
impl Debug for c64 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.re.fmt(f)?;
        f.write_str(" + ")?;
        self.im.fmt(f)?;
        f.write_str(" * I")
    }
}
impl Debug for c32conj {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.re.fmt(f)?;
        f.write_str(" - ")?;
        self.neg_im.fmt(f)?;
        f.write_str(" * I")
    }
}
impl Debug for c64conj {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.re.fmt(f)?;
        f.write_str(" - ")?;
        self.neg_im.fmt(f)?;
        f.write_str(" * I")
    }
}

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

/// Trait for types that can be converted to a matrix view.
pub trait AsMatRef<E: Entity> {
    fn as_mat_ref(&self) -> MatRef<'_, E>;
}
/// Trait for types that can be converted to a mutable matrix view.
pub trait AsMatMut<E: Entity> {
    fn as_mat_mut(&mut self) -> MatMut<'_, E>;
}

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

impl ComplexField for c32 {
    type Real = f32;
    type Simd = pulp::Arch;
    type ScalarSimd = NoSimd;

    #[inline(always)]
    fn faer_from_f64(value: f64) -> Self {
        Self {
            re: value as _,
            im: 0.0,
        }
    }

    #[inline(always)]
    fn faer_add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline(always)]
    fn faer_sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    #[inline(always)]
    fn faer_mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    #[inline(always)]
    fn faer_neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn faer_inv(self) -> Self {
        let inf = Self::Real::faer_zero().faer_inv();
        if self.faer_is_nan() {
            // NAN
            Self::faer_nan()
        } else if self == Self::faer_zero() {
            // zero
            Self { re: inf, im: inf }
        } else if self.re == inf || self.im == inf {
            Self::faer_zero()
        } else {
            let re = self.faer_real().faer_abs();
            let im = self.faer_imag().faer_abs();
            let max = if re > im { re } else { im };
            let max_inv = max.faer_inv();
            let x = self.faer_scale_real(max_inv);
            x.faer_conj()
                .faer_scale_real(x.faer_abs2().faer_inv().faer_mul(max_inv))
        }
    }

    #[inline(always)]
    fn faer_conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn faer_sqrt(self) -> Self {
        let this: num_complex::Complex32 = self.into();
        ComplexField::faer_sqrt(this).into()
    }

    #[inline(always)]
    fn faer_scale_real(self, rhs: Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn faer_scale_power_of_two(self, rhs: Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn faer_score(self) -> Self::Real {
        self.faer_abs2()
    }

    #[inline(always)]
    fn faer_abs(self) -> Self::Real {
        self.faer_abs2().faer_sqrt()
    }

    #[inline(always)]
    fn faer_abs2(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    fn faer_nan() -> Self {
        Self {
            re: Self::Real::NAN,
            im: Self::Real::NAN,
        }
    }

    #[inline(always)]
    fn faer_from_real(real: Self::Real) -> Self {
        Self { re: real, im: 0.0 }
    }

    #[inline(always)]
    fn faer_real(self) -> Self::Real {
        self.re
    }

    #[inline(always)]
    fn faer_imag(self) -> Self::Real {
        self.im
    }

    #[inline(always)]
    fn faer_zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    fn faer_one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    #[inline(always)]
    fn faer_slice_as_simd<S: Simd>(slice: &[Self::Unit]) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
        let (head, tail) = S::c32s_as_simd(bytemuck::cast_slice(slice));
        (bytemuck::cast_slice(head), bytemuck::cast_slice(tail))
    }

    #[inline(always)]
    fn faer_slice_as_mut_simd<S: Simd>(
        slice: &mut [Self::Unit],
    ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
        let (head, tail) = S::c32s_as_mut_simd(bytemuck::cast_slice_mut(slice));
        (
            bytemuck::cast_slice_mut(head),
            bytemuck::cast_slice_mut(tail),
        )
    }

    #[inline(always)]
    fn faer_partial_load_last_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.c32s_partial_load_last(bytemuck::cast_slice(slice))
    }

    #[inline(always)]
    fn faer_partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.c32s_partial_store_last(bytemuck::cast_slice_mut(slice), values)
    }

    #[inline(always)]
    fn faer_partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.c32s_partial_load(bytemuck::cast_slice(slice))
    }

    #[inline(always)]
    fn faer_partial_store_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.c32s_partial_store(bytemuck::cast_slice_mut(slice), values)
    }

    #[inline(always)]
    fn faer_simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
        simd.c32s_splat(pulp::cast(unit))
    }

    #[inline(always)]
    fn faer_simd_neg<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        simd.c32s_neg(values)
    }

    #[inline(always)]
    fn faer_simd_conj<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        let _ = simd;
        values
    }

    #[inline(always)]
    fn faer_simd_add<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c32s_add(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c32s_sub(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c32s_mul(lhs, rhs)
    }
    #[inline(always)]
    fn faer_simd_scale_real<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self::Real, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        if coe::is_same::<pulp::Scalar, S>() {
            let lhs: f32 = bytemuck::cast(lhs);
            let rhs: num_complex::Complex32 = bytemuck::cast(rhs);
            bytemuck::cast(lhs * rhs)
        } else {
            bytemuck::cast(simd.f32s_mul(lhs, bytemuck::cast(rhs)))
        }
    }
    #[inline(always)]
    fn faer_simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c32s_conj_mul(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c32s_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn faer_simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c32s_conj_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn faer_simd_reduce_add<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> Self {
        pulp::cast(simd.c32s_reduce_sum(values))
    }

    #[inline(always)]
    fn faer_simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self::Real, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        let _ = (simd, values, acc);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }
    #[inline(always)]
    fn faer_simd_abs2<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        let _ = (simd, values);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }
    #[inline(always)]
    fn faer_simd_score<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        let _ = (simd, values);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }

    #[inline(always)]
    fn faer_simd_scalar_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        cast(simd.c32_scalar_mul(cast(lhs), cast(rhs)))
    }
    #[inline(always)]
    fn faer_simd_scalar_conj_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        cast(simd.c32_scalar_conj_mul(cast(lhs), cast(rhs)))
    }
    #[inline(always)]
    fn faer_simd_scalar_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        cast(simd.c32_scalar_mul_add_e(cast(lhs), cast(rhs), cast(acc)))
    }
    #[inline(always)]
    fn faer_simd_scalar_conj_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        cast(simd.c32_scalar_conj_mul_add_e(cast(lhs), cast(rhs), cast(acc)))
    }
}
impl ComplexField for c64 {
    type Real = f64;
    type Simd = pulp::Arch;
    type ScalarSimd = NoSimd;

    #[inline(always)]
    fn faer_from_f64(value: f64) -> Self {
        Self {
            re: value as _,
            im: 0.0,
        }
    }

    #[inline(always)]
    fn faer_add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline(always)]
    fn faer_sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    #[inline(always)]
    fn faer_mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    #[inline(always)]
    fn faer_neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn faer_inv(self) -> Self {
        let inf = Self::Real::faer_zero().faer_inv();
        if self.faer_is_nan() {
            // NAN
            Self::faer_nan()
        } else if self == Self::faer_zero() {
            // zero
            Self { re: inf, im: inf }
        } else if self.re == inf || self.im == inf {
            Self::faer_zero()
        } else {
            let re = self.faer_real().faer_abs();
            let im = self.faer_imag().faer_abs();
            let max = if re > im { re } else { im };
            let max_inv = max.faer_inv();
            let x = self.faer_scale_real(max_inv);
            x.faer_conj()
                .faer_scale_real(x.faer_abs2().faer_inv().faer_mul(max_inv))
        }
    }

    #[inline(always)]
    fn faer_conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn faer_sqrt(self) -> Self {
        let this: num_complex::Complex64 = self.into();
        ComplexField::faer_sqrt(this).into()
    }

    #[inline(always)]
    fn faer_scale_real(self, rhs: Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn faer_scale_power_of_two(self, rhs: Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn faer_score(self) -> Self::Real {
        self.faer_abs2()
    }

    #[inline(always)]
    fn faer_abs(self) -> Self::Real {
        self.faer_abs2().faer_sqrt()
    }

    #[inline(always)]
    fn faer_abs2(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    fn faer_nan() -> Self {
        Self {
            re: Self::Real::NAN,
            im: Self::Real::NAN,
        }
    }

    #[inline(always)]
    fn faer_from_real(real: Self::Real) -> Self {
        Self { re: real, im: 0.0 }
    }

    #[inline(always)]
    fn faer_real(self) -> Self::Real {
        self.re
    }

    #[inline(always)]
    fn faer_imag(self) -> Self::Real {
        self.im
    }

    #[inline(always)]
    fn faer_zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    fn faer_one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    #[inline(always)]
    fn faer_slice_as_simd<S: Simd>(slice: &[Self::Unit]) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
        let (head, tail) = S::c64s_as_simd(bytemuck::cast_slice(slice));
        (bytemuck::cast_slice(head), bytemuck::cast_slice(tail))
    }

    #[inline(always)]
    fn faer_slice_as_mut_simd<S: Simd>(
        slice: &mut [Self::Unit],
    ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
        let (head, tail) = S::c64s_as_mut_simd(bytemuck::cast_slice_mut(slice));
        (
            bytemuck::cast_slice_mut(head),
            bytemuck::cast_slice_mut(tail),
        )
    }

    #[inline(always)]
    fn faer_partial_load_last_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.c64s_partial_load_last(bytemuck::cast_slice(slice))
    }

    #[inline(always)]
    fn faer_partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.c64s_partial_store_last(bytemuck::cast_slice_mut(slice), values)
    }

    #[inline(always)]
    fn faer_partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.c64s_partial_load(bytemuck::cast_slice(slice))
    }

    #[inline(always)]
    fn faer_partial_store_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.c64s_partial_store(bytemuck::cast_slice_mut(slice), values)
    }

    #[inline(always)]
    fn faer_simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
        simd.c64s_splat(pulp::cast(unit))
    }

    #[inline(always)]
    fn faer_simd_neg<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        simd.c64s_neg(values)
    }

    #[inline(always)]
    fn faer_simd_conj<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        let _ = simd;
        values
    }

    #[inline(always)]
    fn faer_simd_add<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_add(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_sub(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_mul(lhs, rhs)
    }
    #[inline(always)]
    fn faer_simd_scale_real<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self::Real, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        if coe::is_same::<pulp::Scalar, S>() {
            let lhs: f64 = bytemuck::cast(lhs);
            let rhs: num_complex::Complex64 = bytemuck::cast(rhs);
            bytemuck::cast(lhs * rhs)
        } else {
            bytemuck::cast(simd.f64s_mul(lhs, bytemuck::cast(rhs)))
        }
    }
    #[inline(always)]
    fn faer_simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_conj_mul(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn faer_simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_conj_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn faer_simd_reduce_add<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> Self {
        pulp::cast(simd.c64s_reduce_sum(values))
    }

    #[inline(always)]
    fn faer_simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self::Real, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        let _ = (simd, values, acc);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }
    #[inline(always)]
    fn faer_simd_abs2<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        let _ = (simd, values);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }
    #[inline(always)]
    fn faer_simd_score<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        let _ = (simd, values);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }

    #[inline(always)]
    fn faer_simd_scalar_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        cast(simd.c64_scalar_mul(cast(lhs), cast(rhs)))
    }
    #[inline(always)]
    fn faer_simd_scalar_conj_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        cast(simd.c64_scalar_conj_mul(cast(lhs), cast(rhs)))
    }
    #[inline(always)]
    fn faer_simd_scalar_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        cast(simd.c64_scalar_mul_add_e(cast(lhs), cast(rhs), cast(acc)))
    }
    #[inline(always)]
    fn faer_simd_scalar_conj_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        cast(simd.c64_scalar_conj_mul_add_e(cast(lhs), cast(rhs), cast(acc)))
    }
}

#[doc(hidden)]
pub use pulp;

unsafe impl Entity for c32 {
    type Unit = Self;
    type Index = u32;
    type SimdUnit<S: Simd> = S::c32s;
    type SimdMask<S: Simd> = S::m32s;
    type SimdIndex<S: Simd> = S::u32s;
    type Group = IdentityGroup;
    type Iter<I: Iterator> = I;

    const N_COMPONENTS: usize = 1;
    const UNIT: GroupCopyFor<Self, ()> = ();

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
    fn faer_unzip<T, U>(zipped: GroupFor<Self, (T, U)>) -> (GroupFor<Self, T>, GroupFor<Self, U>) {
        zipped
    }

    #[inline(always)]
    fn faer_into_iter<I: IntoIterator>(iter: GroupFor<Self, I>) -> Self::Iter<I::IntoIter> {
        iter.into_iter()
    }
}
unsafe impl Entity for c32conj {
    type Unit = Self;
    type Index = u32;
    type SimdUnit<S: Simd> = S::c32s;
    type SimdMask<S: Simd> = S::m32s;
    type SimdIndex<S: Simd> = S::u32s;
    type Group = IdentityGroup;
    type Iter<I: Iterator> = I;

    const N_COMPONENTS: usize = 1;
    const UNIT: GroupCopyFor<Self, ()> = ();

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
    fn faer_unzip<T, U>(zipped: GroupFor<Self, (T, U)>) -> (GroupFor<Self, T>, GroupFor<Self, U>) {
        zipped
    }

    #[inline(always)]
    fn faer_into_iter<I: IntoIterator>(iter: GroupFor<Self, I>) -> Self::Iter<I::IntoIter> {
        iter.into_iter()
    }
}

unsafe impl Entity for c64 {
    type Unit = Self;
    type Index = u64;
    type SimdUnit<S: Simd> = S::c64s;
    type SimdMask<S: Simd> = S::m64s;
    type SimdIndex<S: Simd> = S::u64s;
    type Group = IdentityGroup;
    type Iter<I: Iterator> = I;

    const N_COMPONENTS: usize = 1;
    const UNIT: GroupCopyFor<Self, ()> = ();

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
    fn faer_unzip<T, U>(zipped: GroupFor<Self, (T, U)>) -> (GroupFor<Self, T>, GroupFor<Self, U>) {
        zipped
    }

    #[inline(always)]
    fn faer_into_iter<I: IntoIterator>(iter: GroupFor<Self, I>) -> Self::Iter<I::IntoIter> {
        iter.into_iter()
    }
}
unsafe impl Entity for c64conj {
    type Unit = Self;
    type Index = u64;
    type SimdUnit<S: Simd> = S::c64s;
    type SimdMask<S: Simd> = S::m64s;
    type SimdIndex<S: Simd> = S::u64s;
    type Group = IdentityGroup;
    type Iter<I: Iterator> = I;

    const N_COMPONENTS: usize = 1;
    const UNIT: GroupCopyFor<Self, ()> = ();

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
    fn faer_unzip<T, U>(zipped: GroupFor<Self, (T, U)>) -> (GroupFor<Self, T>, GroupFor<Self, U>) {
        zipped
    }

    #[inline(always)]
    fn faer_into_iter<I: IntoIterator>(iter: GroupFor<Self, I>) -> Self::Iter<I::IntoIter> {
        iter.into_iter()
    }
}

unsafe impl Conjugate for c32 {
    type Conj = c32conj;
    type Canonical = c32;

    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        self
    }
}
unsafe impl Conjugate for c32conj {
    type Conj = c32;
    type Canonical = c32;

    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        c32 {
            re: self.re,
            im: -self.neg_im,
        }
    }
}

unsafe impl Conjugate for c64 {
    type Conj = c64conj;
    type Canonical = c64;

    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        self
    }
}
unsafe impl Conjugate for c64conj {
    type Conj = c64;
    type Canonical = c64;

    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        c64 {
            re: self.re,
            im: -self.neg_im,
        }
    }
}

#[repr(C)]
struct MatImpl<E: Entity> {
    ptr: GroupCopyFor<E, *mut E::Unit>,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
}
#[repr(C)]
struct NonNullMatImpl<E: Entity> {
    ptr: GroupCopyFor<E, NonNull<E::Unit>>,
    nrows: usize,
    ncols: usize,
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

pub mod inner {
    use super::*;

    impl<E: Entity> Copy for DiagRef<'_, E> {}
    impl<E: Entity> Clone for DiagRef<'_, E> {
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
        pub(crate) forward: Box<[I]>,
        pub(crate) inverse: Box<[I]>,
        pub(crate) __marker: PhantomData<E>,
    }

    #[repr(C)]
    pub struct DiagRef<'a, E: Entity> {
        pub(crate) inner: MatRef<'a, E>,
    }

    #[repr(C)]
    pub struct DiagMut<'a, E: Entity> {
        pub(crate) inner: MatMut<'a, E>,
    }

    #[repr(C)]
    pub struct DiagOwn<E: Entity> {
        pub(crate) inner: Mat<E>,
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
        pub(crate) inner: NonNullMatImpl<E>,
        pub(crate) row_capacity: usize,
        pub(crate) col_capacity: usize,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(transparent)]
    pub struct Scale<E: Entity>(pub E);
}
use inner::*;

pub mod group_helpers {
    use super::*;
    #[cfg(feature = "std")]
    use assert2::{assert, debug_assert};
    use core::ops::Range;

    pub struct SliceGroup<'a, E: Entity>(GroupCopyFor<E, &'static [E::Unit]>, PhantomData<&'a ()>);
    pub struct SliceGroupMut<'a, E: Entity>(
        GroupFor<E, &'static mut [E::Unit]>,
        PhantomData<&'a mut ()>,
    );

    pub struct RefGroup<'a, E: Entity>(GroupCopyFor<E, &'static E::Unit>, PhantomData<&'a ()>);
    pub struct RefGroupMut<'a, E: Entity>(
        GroupFor<E, &'static mut E::Unit>,
        PhantomData<&'a mut ()>,
    );

    impl<E: Entity> Copy for SliceGroup<'_, E> {}
    impl<E: Entity> Copy for RefGroup<'_, E> {}
    impl<E: Entity> Clone for SliceGroup<'_, E> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<E: Entity> Clone for RefGroup<'_, E> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<'a, E: Entity> RefGroup<'a, E> {
        #[inline(always)]
        pub fn new(slice: GroupFor<E, &'a E::Unit>) -> Self {
            Self(unsafe { transmute_unchecked(slice) }, PhantomData)
        }

        #[inline(always)]
        pub fn into_inner(self) -> GroupFor<E, &'a E::Unit> {
            unsafe { transmute_unchecked(self.0) }
        }
    }

    impl<'a, E: Entity> RefGroupMut<'a, E> {
        #[inline(always)]
        pub fn new(slice: GroupFor<E, &'a mut E::Unit>) -> Self {
            Self(unsafe { transmute_unchecked(slice) }, PhantomData)
        }

        #[inline(always)]
        pub fn into_inner(self) -> GroupFor<E, &'a mut E::Unit> {
            unsafe { transmute_unchecked(self.0) }
        }
    }

    impl<'a, E: Entity> IntoConst for SliceGroup<'a, E> {
        type Target = SliceGroup<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            self
        }
    }
    impl<'a, E: Entity> IntoConst for SliceGroupMut<'a, E> {
        type Target = SliceGroup<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            SliceGroup::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| &*slice,
            ))
        }
    }

    impl<'a, E: Entity> IntoConst for RefGroup<'a, E> {
        type Target = RefGroup<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            self
        }
    }
    impl<'a, E: Entity> IntoConst for RefGroupMut<'a, E> {
        type Target = RefGroup<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            RefGroup::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| &*slice,
            ))
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for RefGroup<'a, E> {
        type Target = RefGroup<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for RefGroup<'a, E> {
        type Target = RefGroup<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for RefGroupMut<'a, E> {
        type Target = RefGroupMut<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            RefGroupMut::new(E::faer_map(
                E::faer_as_mut(&mut self.0),
                #[inline(always)]
                |this| &mut **this,
            ))
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for RefGroupMut<'a, E> {
        type Target = RefGroup<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            RefGroup::new(E::faer_map(
                E::faer_as_ref(&self.0),
                #[inline(always)]
                |this| &**this,
            ))
        }
    }

    impl<'a, E: Entity> SliceGroup<'a, E> {
        #[inline(always)]
        pub fn new(slice: GroupFor<E, &'a [E::Unit]>) -> Self {
            Self(unsafe { transmute_unchecked(slice) }, PhantomData)
        }

        #[inline(always)]
        pub fn into_inner(self) -> GroupFor<E, &'a [E::Unit]> {
            unsafe { transmute_unchecked(self.0) }
        }
    }

    impl<'a, E: Entity> SliceGroupMut<'a, E> {
        #[inline(always)]
        pub fn new(slice: GroupFor<E, &'a mut [E::Unit]>) -> Self {
            Self(unsafe { transmute_unchecked(slice) }, PhantomData)
        }

        #[inline(always)]
        pub fn into_inner(self) -> GroupFor<E, &'a mut [E::Unit]> {
            unsafe { transmute_unchecked(self.0) }
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for SliceGroup<'a, E> {
        type Target = SliceGroup<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for SliceGroup<'a, E> {
        type Target = SliceGroup<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for SliceGroupMut<'a, E> {
        type Target = SliceGroupMut<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            SliceGroupMut::new(E::faer_map(
                E::faer_as_mut(&mut self.0),
                #[inline(always)]
                |this| &mut **this,
            ))
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for SliceGroupMut<'a, E> {
        type Target = SliceGroup<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            SliceGroup::new(E::faer_map(
                E::faer_as_ref(&self.0),
                #[inline(always)]
                |this| &**this,
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

        #[inline(always)]
        #[track_caller]
        pub fn subslice(self, range: Range<usize>) -> Self {
            assert!(range.start <= range.end);
            assert!(range.end <= self.len());
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
            debug_assert!(range.start <= range.end);
            debug_assert!(range.end <= self.len());
            Self::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| slice.get_unchecked(range.start..range.end),
            ))
        }

        #[inline(always)]
        pub fn into_ref_iter(self) -> impl Iterator<Item = RefGroup<'a, E>> {
            E::faer_into_iter(self.into_inner()).map(RefGroup::new)
        }

        #[inline(always)]
        pub fn into_chunks_exact(
            self,
            chunk_size: usize,
        ) -> (impl Iterator<Item = SliceGroup<'a, E>>, Self) {
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
        #[inline]
        pub fn is_empty(&self) -> bool {
            self.rb().is_empty()
        }

        #[inline]
        pub fn len(&self) -> usize {
            self.rb().len()
        }

        #[inline]
        pub fn fill_zero(&mut self) {
            E::faer_map(self.rb_mut().into_inner(), |slice| unsafe {
                let len = slice.len();
                core::ptr::write_bytes(slice.as_mut_ptr(), 0u8, len);
            });
        }

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

        #[inline(always)]
        #[track_caller]
        pub fn subslice(self, range: Range<usize>) -> Self {
            assert!(range.start <= range.end);
            assert!(range.end <= self.len());
            unsafe { self.subslice_unchecked(range) }
        }

        #[inline(always)]
        #[track_caller]
        pub unsafe fn subslice_unchecked(self, range: Range<usize>) -> Self {
            debug_assert!(range.start <= range.end);
            debug_assert!(range.end <= self.len());
            Self::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| slice.get_unchecked_mut(range.start..range.end),
            ))
        }

        #[inline(always)]
        pub fn into_mut_iter(self) -> impl Iterator<Item = RefGroupMut<'a, E>> {
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
        ) -> (impl Iterator<Item = SliceGroupMut<'a, E>>, Self) {
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

    impl<E: Entity> core::fmt::Debug for RefGroup<'_, E> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.read().fmt(f)
        }
    }
    impl<E: Entity> core::fmt::Debug for RefGroupMut<'_, E> {
        #[inline]
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.read().fmt(f)
        }
    }
    impl<E: Entity> core::fmt::Debug for SliceGroup<'_, E> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_list().entries(self.into_ref_iter()).finish()
        }
    }
    impl<E: Entity> core::fmt::Debug for SliceGroupMut<'_, E> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            self.rb().fmt(f)
        }
    }
}

pub mod sparse {
    use super::*;
    #[cfg(feature = "std")]
    use assert2::assert;
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

    #[derive(Debug)]
    pub struct SymbolicSparseColMatRef<'a, I> {
        nrows: usize,
        ncols: usize,
        col_ptr: &'a [I],
        col_nnz: Option<&'a [I]>,
        row_ind: &'a [I],
    }

    impl<I> Copy for SymbolicSparseColMatRef<'_, I> {}
    impl<I> Clone for SymbolicSparseColMatRef<'_, I> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }

    /// Requires:
    /// * `nrows <= I::MAX` (always checked)
    /// * `ncols <= I::MAX` (always checked)
    /// * `col_ptrs` has length n (always checked)
    /// * `col_ptrs` is non-decreasing
    /// * `col_ptrs[0]..col_ptrs[n]` is a valid range in row_indices (always checked, assuming
    ///   non-decreasing)
    /// * if `nnz_per_col` is `None`, elements of `row_indices[col_ptrs[j]..col_ptrs[j + 1]]` are
    ///   less than `nrows`
    ///
    /// * `nnz_per_col[j] <= col_ptrs[j+1] - col_ptrs[j]`
    /// * if `nnz_per_col` is `Some(_)`, elements of `row_indices[col_ptrs[j]..][..nnz_per_col[j]]`
    ///   are less than `nrows`
    ///
    /// Ensures:
    /// * `self.compute_nnz() <= I::MAX`
    impl<'a, I: Index> SymbolicSparseColMatRef<'a, I> {
        #[inline]
        #[track_caller]
        pub fn new_checked(
            nrows: usize,
            ncols: usize,
            col_ptrs: &'a [I],
            nnz_per_col: Option<&'a [I]>,
            row_indices: &'a [I],
        ) -> Self {
            assert!(ncols <= <I::Signed as SignedIndex>::MAX.zx());
            assert!(nrows <= <I::Signed as SignedIndex>::MAX.zx());
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

        #[inline(always)]
        #[track_caller]
        pub unsafe fn new_unchecked(
            nrows: usize,
            ncols: usize,
            col_ptrs: &'a [I],
            nnz_per_col: Option<&'a [I]>,
            row_indices: &'a [I],
        ) -> Self {
            assert!(ncols <= <I::Signed as SignedIndex>::MAX.zx());
            assert!(nrows <= <I::Signed as SignedIndex>::MAX.zx());
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

        #[inline]
        pub fn nrows(&self) -> usize {
            self.nrows
        }
        #[inline]
        pub fn ncols(&self) -> usize {
            self.ncols
        }

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

        #[inline]
        pub fn col_ptrs(&self) -> &'a [I] {
            self.col_ptr
        }

        #[inline]
        pub fn nnz_per_col(&self) -> Option<&'a [I]> {
            self.col_nnz
        }

        #[inline]
        pub fn row_indices(&self) -> &'a [I] {
            self.row_ind
        }

        #[inline]
        #[track_caller]
        pub fn row_indices_of_col_raw(&self, j: usize) -> &'a [I] {
            &self.row_ind[self.col_range(j)]
        }

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

    #[derive(Debug)]
    pub struct SparseColMatRef<'a, I, E: Entity> {
        symbolic: SymbolicSparseColMatRef<'a, I>,
        values: SliceGroup<'a, E>,
    }

    impl<I, E: Entity> Copy for SparseColMatRef<'_, I, E> {}
    impl<I, E: Entity> Clone for SparseColMatRef<'_, I, E> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<'a, I: Index, E: Entity> SparseColMatRef<'a, I, E> {
        #[inline]
        #[track_caller]
        pub fn new(
            symbolic: SymbolicSparseColMatRef<'a, I>,
            values: GroupFor<E, &'a [E::Unit]>,
        ) -> Self {
            let values = SliceGroup::new(values);
            assert!(symbolic.row_indices().len() == values.len());
            Self { symbolic, values }
        }
        #[inline]
        pub fn values(&self) -> GroupFor<E, &'a [E::Unit]> {
            self.values.into_inner()
        }

        #[inline]
        #[track_caller]
        pub fn values_of_col(&self, j: usize) -> GroupFor<E, &'a [E::Unit]> {
            self.values.subslice(self.col_range(j)).into_inner()
        }

        #[inline]
        pub fn symbolic(&self) -> SymbolicSparseColMatRef<'a, I> {
            self.symbolic
        }
    }

    impl<'a, I, E: Entity> core::ops::Deref for SparseColMatRef<'a, I, E> {
        type Target = SymbolicSparseColMatRef<'a, I>;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.symbolic
        }
    }
}

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
    pub fn value(self) -> E {
        self.inner.0
    }
}

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

unsafe impl<E: Entity + Send + Sync> Send for MatImpl<E> {}
unsafe impl<E: Entity + Send + Sync> Sync for MatImpl<E> {}
unsafe impl<E: Entity + Send + Sync> Send for NonNullMatImpl<E> {}
unsafe impl<E: Entity + Send + Sync> Sync for NonNullMatImpl<E> {}

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

/// Represents a type that can be used to slice a matrix, such as an index or a range of indices.
pub trait MatIndex<RowRange, ColRange>: seal::Seal + Sized {
    type Target;
    #[allow(clippy::missing_safety_doc)]
    unsafe fn get_unchecked(this: Self, row: RowRange, col: ColRange) -> Self::Target {
        <Self as MatIndex<RowRange, ColRange>>::get(this, row, col)
    }
    fn get(this: Self, row: RowRange, col: ColRange) -> Self::Target;
}

const _: () = {
    // RangeFull
    // Range
    // RangeInclusive
    // RangeTo
    // RangeToInclusive
    // usize

    use core::ops::RangeFull;
    type Range = core::ops::Range<usize>;
    type RangeInclusive = core::ops::RangeInclusive<usize>;
    type RangeTo = core::ops::RangeTo<usize>;
    type RangeToInclusive = core::ops::RangeToInclusive<usize>;

    macro_rules! impl_ranges {
        ($mat: ident) => {
            impl<E: Entity, RowRange> MatIndex<RowRange, RangeTo> for $mat<'_, E>
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
            impl<E: Entity, RowRange> MatIndex<RowRange, RangeToInclusive> for $mat<'_, E>
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
            impl<E: Entity, RowRange> MatIndex<RowRange, RangeInclusive> for $mat<'_, E>
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
                    <Self as MatIndex<RowRange, Range>>::get(
                        this,
                        row,
                        *col.start()..*col.end() + 1,
                    )
                }
            }
            impl<E: Entity, RowRange> MatIndex<RowRange, RangeFull> for $mat<'_, E>
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

            impl<E: Entity> MatIndex<RangeFull, Range> for $mat<'_, E> {
                type Target = Self;

                #[track_caller]
                #[inline(always)]
                fn get(this: Self, row: RangeFull, col: Range) -> Self {
                    let _ = row;
                    this.subcols(col.start, col.end - col.start)
                }
            }
            impl<E: Entity> MatIndex<RangeFull, usize> for $mat<'_, E> {
                type Target = Self;

                #[track_caller]
                #[inline(always)]
                fn get(this: Self, row: RangeFull, col: usize) -> Self {
                    let _ = row;
                    this.col(col)
                }
            }

            impl<E: Entity> MatIndex<Range, Range> for $mat<'_, E> {
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
            impl<E: Entity> MatIndex<Range, usize> for $mat<'_, E> {
                type Target = Self;

                #[track_caller]
                #[inline(always)]
                fn get(this: Self, row: Range, col: usize) -> Self {
                    this.submatrix(row.start, col, row.end - row.start, 1)
                }
            }

            impl<E: Entity> MatIndex<RangeInclusive, Range> for $mat<'_, E> {
                type Target = Self;

                #[track_caller]
                #[inline(always)]
                fn get(this: Self, row: RangeInclusive, col: Range) -> Self {
                    assert!(*row.end() != usize::MAX);
                    <Self as MatIndex<Range, Range>>::get(this, *row.start()..*row.end() + 1, col)
                }
            }
            impl<E: Entity> MatIndex<RangeInclusive, usize> for $mat<'_, E> {
                type Target = Self;

                #[track_caller]
                #[inline(always)]
                fn get(this: Self, row: RangeInclusive, col: usize) -> Self {
                    assert!(*row.end() != usize::MAX);
                    <Self as MatIndex<Range, usize>>::get(this, *row.start()..*row.end() + 1, col)
                }
            }

            impl<E: Entity> MatIndex<RangeTo, Range> for $mat<'_, E> {
                type Target = Self;

                #[track_caller]
                #[inline(always)]
                fn get(this: Self, row: RangeTo, col: Range) -> Self {
                    <Self as MatIndex<Range, Range>>::get(this, 0..row.end, col)
                }
            }
            impl<E: Entity> MatIndex<RangeTo, usize> for $mat<'_, E> {
                type Target = Self;

                #[track_caller]
                #[inline(always)]
                fn get(this: Self, row: RangeTo, col: usize) -> Self {
                    <Self as MatIndex<Range, usize>>::get(this, 0..row.end, col)
                }
            }

            impl<E: Entity> MatIndex<RangeToInclusive, Range> for $mat<'_, E> {
                type Target = Self;

                #[track_caller]
                #[inline(always)]
                fn get(this: Self, row: RangeToInclusive, col: Range) -> Self {
                    assert!(row.end != usize::MAX);
                    <Self as MatIndex<Range, Range>>::get(this, 0..row.end + 1, col)
                }
            }
            impl<E: Entity> MatIndex<RangeToInclusive, usize> for $mat<'_, E> {
                type Target = Self;

                #[track_caller]
                #[inline(always)]
                fn get(this: Self, row: RangeToInclusive, col: usize) -> Self {
                    assert!(row.end != usize::MAX);
                    <Self as MatIndex<Range, usize>>::get(this, 0..row.end + 1, col)
                }
            }

            impl<E: Entity> MatIndex<usize, Range> for $mat<'_, E> {
                type Target = Self;

                #[track_caller]
                #[inline(always)]
                fn get(this: Self, row: usize, col: Range) -> Self {
                    this.submatrix(row, col.start, 1, col.end - col.start)
                }
            }
        };
    }

    impl_ranges!(MatRef);
    impl_ranges!(MatMut);

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
            assert!(row < this.nrows());
            assert!(col < this.ncols());
            unsafe { <Self as MatIndex<usize, usize>>::get_unchecked(this, row, col) }
        }
    }

    impl<'a, E: Entity> MatIndex<usize, usize> for MatMut<'a, E> {
        type Target = GroupFor<E, &'a mut E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, row: usize, col: usize) -> Self::Target {
            unsafe { E::faer_map(this.ptr_inbounds_at(row, col), |ptr| &mut *ptr) }
        }

        #[track_caller]
        #[inline(always)]
        fn get(this: Self, row: usize, col: usize) -> Self::Target {
            assert!(row < this.nrows());
            assert!(col < this.ncols());
            unsafe { <Self as MatIndex<usize, usize>>::get_unchecked(this, row, col) }
        }
    }
};

impl<'a, E: Entity> Matrix<DiagRef<'a, E>> {
    #[inline(always)]
    pub fn into_column_vector(self) -> MatRef<'a, E> {
        self.inner.inner
    }
}

impl<'a, E: Entity> Matrix<DiagMut<'a, E>> {
    #[inline(always)]
    pub fn into_column_vector(self) -> MatMut<'a, E> {
        self.inner.inner
    }
}

impl<E: Entity> Matrix<DiagOwn<E>> {
    #[inline(always)]
    pub fn into_column_vector(self) -> Mat<E> {
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

impl<'a, E: Entity> MatRef<'a, E> {
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
    /// use faer_core::{mat, MatRef};
    ///
    /// let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
    /// let view = MatRef::<f64>::from_column_major_slice(&slice, 3, 2);
    ///
    /// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    /// assert_eq!(expected, view);
    /// ```
    #[track_caller]
    pub fn from_column_major_slice(
        slice: GroupFor<E, &'a [E::Unit]>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        let size = usize::checked_mul(nrows, ncols).unwrap_or(usize::MAX);
        // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
        // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
        // we don't care
        E::faer_map(
            E::faer_copy(&slice),
            #[inline(always)]
            |slice| assert!(size == slice.len()),
        );
        unsafe {
            Self::from_raw_parts(
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
    /// use faer_core::{mat, MatRef};
    ///
    /// let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
    /// let view = MatRef::<f64>::from_row_major_slice(&slice, 3, 2);
    ///
    /// let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// assert_eq!(expected, view);
    /// ```
    #[inline(always)]
    #[track_caller]
    pub fn from_row_major_slice(
        slice: GroupFor<E, &'a [E::Unit]>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self::from_column_major_slice(slice, ncols, nrows).transpose()
    }

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
    /// use faer_core::{mat, MatRef};
    ///
    /// // row major matrix with 2 rows, 3 columns, with a column at the end that we want to skip.
    /// // the row stride is the pointer offset from the address of 1.0 to the address of 4.0,
    /// // which is 4.
    /// // the column stride is the pointer offset from the address of 1.0 to the address of 2.0,
    /// // which is 1.
    /// let data = [[1.0, 2.0, 3.0, f64::NAN], [4.0, 5.0, 6.0, f64::NAN]];
    /// let matrix = unsafe { MatRef::<f64>::from_raw_parts(data.as_ptr() as *const f64, 2, 3, 4, 1) };
    ///
    /// let expected = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// assert_eq!(expected.as_ref(), matrix);
    /// ```
    #[inline(always)]
    #[track_caller]
    pub unsafe fn from_raw_parts(
        ptr: GroupFor<E, *const E::Unit>,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            inner: DenseRef {
                inner: MatImpl {
                    ptr: into_copy::<E, _>(E::faer_map(ptr, |ptr| ptr as *mut E::Unit)),
                    nrows,
                    ncols,
                    row_stride,
                    col_stride,
                },
                __marker: PhantomData,
            },
        }
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
        E::faer_map(from_copy::<E, _>(self.inner.inner.ptr), |ptr| {
            ptr as *const E::Unit
        })
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
        E::faer_map(self.as_ptr(), |ptr| {
            ptr.wrapping_offset(row as isize * self.inner.inner.row_stride)
                .wrapping_offset(col as isize * self.inner.inner.col_stride)
        })
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
        debug_assert!(row < self.nrows());
        debug_assert!(col < self.ncols());
        E::faer_map(self.as_ptr(), |ptr| {
            ptr.offset(row as isize * self.inner.inner.row_stride)
                .offset(col as isize * self.inner.inner.col_stride)
        })
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
    pub fn split_at(self, row: usize, col: usize) -> [Self; 4] {
        assert!(row <= self.nrows());
        assert!(col <= self.ncols());

        let row_stride = self.row_stride();
        let col_stride = self.col_stride();

        let nrows = self.nrows();
        let ncols = self.ncols();

        unsafe {
            let top_left = self.ptr_at(0, 0);
            let top_right = self.ptr_at(0, col);
            let bot_left = self.ptr_at(row, 0);
            let bot_right = self.ptr_at(row, col);

            [
                Self::from_raw_parts(top_left, row, col, row_stride, col_stride),
                Self::from_raw_parts(top_right, row, ncols - col, row_stride, col_stride),
                Self::from_raw_parts(bot_left, nrows - row, col, row_stride, col_stride),
                Self::from_raw_parts(bot_right, nrows - row, ncols - col, row_stride, col_stride),
            ]
        }
    }

    /// Splits the matrix horizontally at the given row into two parts and returns an array of each
    /// submatrix, in the following order:
    /// * top.
    /// * bottom.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_row(self, row: usize) -> [Self; 2] {
        let [_, top, _, bot] = self.split_at(row, 0);
        [top, bot]
    }

    /// Splits the matrix vertically at the given row into two parts and returns an array of each
    /// submatrix, in the following order:
    /// * left.
    /// * right.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_col(self, col: usize) -> [Self; 2] {
        let [_, _, left, right] = self.split_at(0, col);
        [left, right]
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
        E::faer_from_units(E::faer_map(self.get_unchecked(row, col), |ptr| *ptr))
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
        E::faer_from_units(E::faer_map(self.get(row, col), |ptr| *ptr))
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
            Self::from_raw_parts(
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
            MatRef::<'_, E::Conj>::from_raw_parts(
                transmute_unchecked::<
                    GroupFor<E, *const E::Unit>,
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
                MatRef::<'_, E::Canonical>::from_raw_parts(
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
        let row_stride = -self.row_stride();
        let col_stride = self.col_stride();

        let ptr = self.ptr_at(if nrows == 0 { 0 } else { nrows - 1 }, 0);
        unsafe { Self::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
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
        let col_stride = -self.col_stride();
        let ptr = self.ptr_at(0, if ncols == 0 { 0 } else { ncols - 1 });
        unsafe { Self::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
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

        let ptr = self.ptr_at(
            if nrows == 0 { 0 } else { nrows - 1 },
            if ncols == 0 { 0 } else { ncols - 1 },
        );
        unsafe { Self::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
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
    pub fn submatrix(self, row_start: usize, col_start: usize, nrows: usize, ncols: usize) -> Self {
        assert!(row_start <= self.nrows());
        assert!(col_start <= self.ncols());
        assert!(nrows <= self.nrows() - row_start);
        assert!(ncols <= self.ncols() - col_start);
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        unsafe {
            Self::from_raw_parts(
                self.ptr_at(row_start, col_start),
                nrows,
                ncols,
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
        self.submatrix(row_start, 0, nrows, self.ncols())
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
        self.submatrix(0, col_start, self.nrows(), ncols)
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub fn row(self, row_idx: usize) -> Self {
        self.subrows(row_idx, 1)
    }

    /// Returns a view over the column at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub fn col(self, col_idx: usize) -> Self {
        self.subcols(col_idx, 1)
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal(self) -> Matrix<DiagRef<'a, E>> {
        assert!(self.ncols() == 1);
        Matrix {
            inner: DiagRef { inner: self },
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
                    inner: Self::from_raw_parts(self.as_ptr(), size, 1, row_stride + col_stride, 0),
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
        mat.resize_with(self.nrows(), self.ncols(), |row, col| unsafe {
            self.read_unchecked(row, col).canonicalize()
        });
        mat
    }

    /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
    #[inline]
    pub fn has_nan(&self) -> bool
    where
        E: ComplexField,
    {
        let mut found_nan = false;
        zipped!(*self).for_each(|x| {
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
        zipped!(*self).for_each(|x| {
            all_finite &= x.read().faer_is_finite();
        });
        all_finite
    }

    /// Returns a thin wrapper that can be used to execute coefficient-wise operations on matrices.
    #[inline]
    pub fn cwise(self) -> Zip<(Self,)> {
        Zip { tuple: (self,) }
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
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    #[inline]
    #[track_caller]
    pub fn into_col_chunks(
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

    /// Returns an iterator that provides successive chunks of the rows of this matrix, with
    /// each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn into_row_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a + DoubleEndedIterator<Item = MatRef<'a, E>> {
        self.transpose()
            .into_col_chunks(chunk_size)
            .map(|chunk| chunk.transpose())
    }

    /// Returns a parallel iterator that provides successive chunks of the columns of this matrix,
    /// with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn into_par_col_chunks(
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
    pub fn into_par_row_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E>> {
        use rayon::prelude::*;

        self.transpose()
            .into_par_col_chunks(chunk_size)
            .map(|chunk| chunk.transpose())
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
        self.rb().get(row, col)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for MatMut<'_, E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut E {
        self.rb_mut().get(row, col)
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
        self.as_mut().get(row, col)
    }
}

impl<'a, E: Entity> MatMut<'a, E> {
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
    /// use faer_core::{mat, MatMut};
    ///
    /// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
    /// let view = MatMut::<f64>::from_column_major_slice(&mut slice, 3, 2);
    ///
    /// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    /// assert_eq!(expected, view);
    /// ```
    #[track_caller]
    pub fn from_column_major_slice(
        slice: GroupFor<E, &'a mut [E::Unit]>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        let size = usize::checked_mul(nrows, ncols).unwrap_or(usize::MAX);
        // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
        // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
        // we don't care
        E::faer_map(
            E::faer_as_ref(&slice),
            #[inline(always)]
            |slice| assert!(size == slice.len()),
        );
        unsafe {
            Self::from_raw_parts(
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
    /// use faer_core::{mat, MatMut};
    ///
    /// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
    /// let view = MatMut::<f64>::from_row_major_slice(&mut slice, 3, 2);
    ///
    /// let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// assert_eq!(expected, view);
    /// ```
    #[inline(always)]
    #[track_caller]
    pub fn from_row_major_slice(
        slice: GroupFor<E, &'a mut [E::Unit]>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self::from_column_major_slice(slice, ncols, nrows).transpose()
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
    /// * For each matrix unit, the corresponding pointer must be properly aligned,
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
    /// use faer_core::{mat, MatMut};
    ///
    /// // row major matrix with 2 rows, 3 columns, with a column at the end that we want to skip.
    /// // the row stride is the pointer offset from the address of 1.0 to the address of 4.0,
    /// // which is 4.
    /// // the column stride is the pointer offset from the address of 1.0 to the address of 2.0,
    /// // which is 1.
    /// let mut data = [[1.0, 2.0, 3.0, f64::NAN], [4.0, 5.0, 6.0, f64::NAN]];
    /// let mut matrix =
    ///     unsafe { MatMut::<f64>::from_raw_parts(data.as_mut_ptr() as *mut f64, 2, 3, 4, 1) };
    ///
    /// let expected = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// assert_eq!(expected.as_ref(), matrix);
    /// ```
    #[inline(always)]
    #[track_caller]
    pub unsafe fn from_raw_parts(
        ptr: GroupFor<E, *mut E::Unit>,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            inner: DenseMut {
                inner: MatImpl {
                    ptr: into_copy::<E, _>(E::faer_map(ptr, |ptr| ptr as *mut E::Unit)),
                    nrows,
                    ncols,
                    row_stride,
                    col_stride,
                },
                __marker: PhantomData,
            },
        }
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
    pub fn as_ptr(self) -> GroupFor<E, *mut E::Unit> {
        from_copy::<E, _>(self.inner.inner.ptr)
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
    pub fn ptr_at(self, row: usize, col: usize) -> GroupFor<E, *mut E::Unit> {
        let row_stride = self.inner.inner.row_stride;
        let col_stride = self.inner.inner.col_stride;
        E::faer_map(self.as_ptr(), |ptr| {
            ptr.wrapping_offset(row as isize * row_stride)
                .wrapping_offset(col as isize * col_stride)
        })
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
    pub unsafe fn ptr_inbounds_at(self, row: usize, col: usize) -> GroupFor<E, *mut E::Unit> {
        debug_assert!(row < self.nrows());
        debug_assert!(col < self.ncols());
        let row_stride = self.inner.inner.row_stride;
        let col_stride = self.inner.inner.col_stride;
        E::faer_map(self.as_ptr(), |ptr| {
            ptr.offset(row as isize * row_stride)
                .offset(col as isize * col_stride)
        })
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
    pub fn split_at(self, row: usize, col: usize) -> [Self; 4] {
        let [top_left, top_right, bot_left, bot_right] = self.into_const().split_at(row, col);
        unsafe {
            [
                top_left.const_cast(),
                top_right.const_cast(),
                bot_left.const_cast(),
                bot_right.const_cast(),
            ]
        }
    }

    /// Splits the matrix horizontally at the given row into two parts and returns an array of each
    /// submatrix, in the following order:
    /// * top.
    /// * bottom.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_row(self, row: usize) -> [Self; 2] {
        let [_, top, _, bot] = self.split_at(row, 0);
        [top, bot]
    }

    /// Splits the matrix vertically at the given row into two parts and returns an array of each
    /// submatrix, in the following order:
    /// * left.
    /// * right.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_col(self, col: usize) -> [Self; 2] {
        let [_, _, left, right] = self.split_at(0, col);
        [left, right]
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
        let zipped = E::faer_zip(units, self.rb_mut().ptr_inbounds_at(row, col));
        E::faer_map(zipped, |(unit, ptr)| *ptr = unit);
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
        assert!(row < self.nrows());
        assert!(col < self.ncols());
        unsafe { self.write_unchecked(row, col, value) };
    }

    /// Copies the values from `other` into `self`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.nrows() == other.nrows()`.
    /// * `self.ncols() == other.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn clone_from(&mut self, other: impl AsMatRef<E>) {
        #[track_caller]
        fn implementation<E: Entity>(this: MatMut<'_, E>, other: MatRef<'_, E>) {
            zipped!(this, other).for_each(|mut dst, src| dst.write(src.read()));
        }
        implementation(self.rb_mut(), other.as_mat_ref())
    }

    /// Fills the elements of `self` with zeros.
    #[inline(always)]
    #[track_caller]
    pub fn fill_zeros(&mut self)
    where
        E: ComplexField,
    {
        zipped!(self.rb_mut()).for_each(|mut x| x.write(E::faer_zero()));
    }

    /// Fills the elements of `self` with copies of `constant`.
    #[inline(always)]
    #[track_caller]
    pub fn fill(&mut self, constant: E) {
        zipped!(self.rb_mut()).for_each(|mut x| x.write(constant));
    }

    /// Returns a view over the transpose of `self`.
    ///
    /// # Example
    /// ```
    /// use faer_core::mat;
    ///
    /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_mut();
    /// let transpose = view.transpose();
    ///
    /// let mut expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    /// assert_eq!(expected.as_mut(), transpose);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn transpose(self) -> Self {
        unsafe {
            Self::from_raw_parts(
                from_copy::<E, _>(self.inner.inner.ptr),
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
    pub fn conjugate(self) -> MatMut<'a, E::Conj>
    where
        E: Conjugate,
    {
        unsafe { self.into_const().conjugate().const_cast() }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn adjoint(self) -> MatMut<'a, E::Conj>
    where
        E: Conjugate,
    {
        self.transpose().conjugate()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    #[must_use]
    pub fn canonicalize(self) -> (MatMut<'a, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        let (canonical, conj) = self.into_const().canonicalize();
        unsafe { (canonical.const_cast(), conj) }
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer_core::mat;
    ///
    /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_mut();
    /// let reversed_rows = view.reverse_rows();
    ///
    /// let mut expected = mat![[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]];
    /// assert_eq!(expected.as_mut(), reversed_rows);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows(self) -> Self {
        unsafe { self.into_const().reverse_rows().const_cast() }
    }

    /// Returns a view over the `self`, with the columns in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer_core::mat;
    ///
    /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_mut();
    /// let reversed_cols = view.reverse_cols();
    ///
    /// let mut expected = mat![[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]];
    /// assert_eq!(expected.as_mut(), reversed_cols);
    /// ```
    #[inline(always)]
    #[must_use]
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
    /// let reversed = view.reverse_rows_and_cols();
    ///
    /// let mut expected = mat![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
    /// assert_eq!(expected.as_mut(), reversed);
    /// ```
    #[inline(always)]
    #[must_use]
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
    /// let submatrix = view.submatrix(2, 1, 2, 2);
    ///
    /// let mut expected = mat![[7.0, 11.0], [8.0, 12.0f64]];
    /// assert_eq!(expected.as_mut(), submatrix);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn submatrix(self, row_start: usize, col_start: usize, nrows: usize, ncols: usize) -> Self {
        unsafe {
            self.into_const()
                .submatrix(row_start, col_start, nrows, ncols)
                .const_cast()
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
    /// let mut matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_mut();
    /// let subrows = view.subrows(1, 2);
    ///
    /// let mut expected = mat![[2.0, 6.0, 10.0], [3.0, 7.0, 11.0],];
    /// assert_eq!(expected.as_mut(), subrows);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn subrows(self, row_start: usize, nrows: usize) -> Self {
        let ncols = self.ncols();
        self.submatrix(row_start, 0, nrows, ncols)
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
    /// let subcols = view.subcols(2, 1);
    ///
    /// let mut expected = mat![[9.0], [10.0], [11.0], [12.0f64]];
    /// assert_eq!(expected.as_mut(), subcols);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn subcols(self, col_start: usize, ncols: usize) -> Self {
        let nrows = self.nrows();
        self.submatrix(0, col_start, nrows, ncols)
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub fn row(self, row_idx: usize) -> Self {
        self.subrows(row_idx, 1)
    }

    /// Returns a view over the column at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub fn col(self, col_idx: usize) -> Self {
        self.subcols(col_idx, 1)
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal(self) -> Matrix<DiagMut<'a, E>> {
        assert!(self.ncols() == 1);
        Matrix {
            inner: DiagMut { inner: self },
        }
    }

    #[inline(always)]
    pub fn diagonal(self) -> Matrix<DiagMut<'a, E>> {
        let size = self.nrows().min(self.ncols());
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        unsafe {
            Matrix {
                inner: DiagMut {
                    inner: Self::from_raw_parts(self.as_ptr(), size, 1, row_stride + col_stride, 0),
                },
            }
        }
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

    /// Returns a thin wrapper that can be used to execute coefficient-wise operations on matrices.
    #[inline]
    pub fn cwise(self) -> Zip<(Self,)> {
        Zip { tuple: (self,) }
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
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    #[inline]
    #[track_caller]
    pub fn into_col_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a + DoubleEndedIterator<Item = MatMut<'a, E>> {
        self.into_const()
            .into_col_chunks(chunk_size)
            .map(|chunk| unsafe { chunk.const_cast() })
    }

    /// Returns an iterator that provides successive chunks of the rows of this matrix,
    /// with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn into_row_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a + DoubleEndedIterator<Item = MatMut<'a, E>> {
        self.into_const()
            .into_row_chunks(chunk_size)
            .map(|chunk| unsafe { chunk.const_cast() })
    }

    /// Returns a parallel iterator that provides successive chunks of the columns of this matrix,
    /// with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn into_par_col_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E>> {
        use rayon::prelude::*;
        self.into_const()
            .into_par_col_chunks(chunk_size)
            .map(|chunk| unsafe { chunk.const_cast() })
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
    pub fn into_par_row_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E>> {
        use rayon::prelude::*;
        self.into_const()
            .into_par_row_chunks(chunk_size)
            .map(|chunk| unsafe { chunk.const_cast() })
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
                re: MatRef::from_raw_parts(re, nrows, ncols, row_stride, col_stride),
                im: MatRef::from_raw_parts(im, nrows, ncols, row_stride, col_stride),
            }
        }
    }
}

impl<'a, E: RealField> MatMut<'a, Complex<E>> {
    #[inline(always)]
    pub fn real_imag(self) -> Complex<MatMut<'a, E>> {
        let Complex { re, im } = self.into_const().real_imag();
        unsafe {
            Complex {
                re: re.const_cast(),
                im: im.const_cast(),
            }
        }
    }
}

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
const CACHELINE_ALIGN: usize = {
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

impl<E: Entity> Default for Mat<E> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Entity> Mat<E> {
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: DenseOwn {
                inner: NonNullMatImpl {
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
                inner: NonNullMatImpl {
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
            .diagonal()
            .into_column_vector()
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
    pub fn as_mut_ptr(&mut self) -> GroupFor<E, *mut E::Unit> {
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
        debug_assert!(row_start <= row_end);
        debug_assert!(col_start <= col_end);

        let ptr = self.as_mut_ptr();

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
    pub fn col_ref(&self, col: usize) -> GroupFor<E, &[E::Unit]> {
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
    pub fn col_mut(&mut self, col: usize) -> GroupFor<E, &mut [E::Unit]> {
        assert!(col < self.ncols());
        let nrows = self.nrows();
        let ptr = self.as_mut().ptr_at(0, col);
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, nrows) },
        )
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, E> {
        unsafe {
            MatRef::<'_, E>::from_raw_parts(
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
            MatMut::<'_, E>::from_raw_parts(
                self.as_mut_ptr(),
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
        self.as_mut().get_unchecked(row, col)
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
        self.as_mut().get(row, col)
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
    pub fn clone_from(&mut self, other: impl AsMatRef<E>) {
        #[track_caller]
        fn implementation<E: Entity>(this: &mut Mat<E>, other: MatRef<'_, E>) {
            let mut mat = Mat::<E>::new();
            mat.resize_with(other.nrows(), other.ncols(), |row, col| unsafe {
                other.read_unchecked(row, col)
            });
            *this = mat;
        }
        implementation(self, other.as_mat_ref());
    }

    /// Fills the elements of `self` with zeros.
    #[inline(always)]
    #[track_caller]
    pub fn fill_zeros(&mut self)
    where
        E: ComplexField,
    {
        self.as_mut().fill_zeros()
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
        self.as_ref().into_col_chunks(chunk_size)
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
        self.as_mut().into_col_chunks(chunk_size)
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
        self.as_ref().into_par_col_chunks(chunk_size)
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
        self.as_mut().into_par_col_chunks(chunk_size)
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
        self.as_ref().into_row_chunks(chunk_size)
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
        self.as_mut().into_row_chunks(chunk_size)
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
        self.as_ref().into_par_row_chunks(chunk_size)
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
        self.as_mut().into_par_row_chunks(chunk_size)
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
    mat.as_mut().fill_zeros();
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
            MatMut::from_raw_parts(
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

#[inline]
fn col_stride<Unit: 'static>(nrows: usize) -> usize {
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
        unsafe { transmute_unchecked(self) }
    }
}
impl<'a, FromE: Entity, ToE: Entity> Coerce<MatMut<'a, ToE>> for MatMut<'a, FromE> {
    #[inline(always)]
    fn coerce(self) -> MatMut<'a, ToE> {
        assert!(coe::is_same::<FromE, ToE>());
        unsafe { transmute_unchecked(self) }
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
/// use faer_core::{mat, zipped, Mat};
///
/// let nrows = 2;
/// let ncols = 3;
///
/// let a = mat![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
/// let b = mat![[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]];
/// let mut sum = Mat::<f64>::zeros(nrows, ncols);
///
/// zipped!(sum.as_mut(), a.as_ref(), b.as_ref()).for_each(|mut sum, a, b| {
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
    ($first: expr $(, $rest: expr)* $(,)?) => {
        $first.cwise()$(.zip($rest))*
    };
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

/// Module for index and matrix types with compile time checks, instead of bound checking at
/// runtime.
pub mod constrained {
    use core::ops::Range;

    use super::*;
    use crate::permutation::{Index, SignedIndex};
    #[cfg(feature = "std")]
    use assert2::{assert, debug_assert};

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
        #[cfg(feature = "std")]
        use assert2::assert;

        #[repr(transparent)]
        pub struct PermutationRef<'n, 'a, I, E: Entity>(
            Branded<'n, crate::permutation::PermutationRef<'a, I, E>>,
        );

        impl<'n, 'a, I: Index, E: Entity> PermutationRef<'n, 'a, I, E> {
            #[inline]
            #[track_caller]
            pub fn new(perm: crate::permutation::PermutationRef<'a, I, E>, size: Size<'n>) -> Self {
                let (fwd, inv) = perm.into_arrays();
                assert!(fwd.len() == size.into_inner());
                assert!(inv.len() == size.into_inner());
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
        use crate::{group_helpers::SliceGroup, sparse::__get_unchecked};
        #[cfg(feature = "std")]
        use assert2::assert;
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
                assert!((inner.nrows(), inner.ncols()) == (nrows.into_inner(), ncols.into_inner()));
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
                assert!((inner.nrows(), inner.ncols()) == (nrows.into_inner(), ncols.into_inner()));
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
        use crate::group_helpers::{SliceGroup, SliceGroupMut};
        #[cfg(feature = "std")]
        use assert2::assert;
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
            assert!((inner.nrows(), inner.ncols()) == (nrows.into_inner(), ncols.into_inner()));
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
            assert!((inner.nrows(), inner.ncols()) == (nrows.into_inner(), ncols.into_inner()));
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

                const N_COMPONENTS: usize = 1;
                const UNIT: GroupCopyFor<Self, ()> = ();

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
    #[cfg(feature = "std")]
    use assert2::assert;

    #[test]
    fn basic_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let slice = unsafe { MatRef::<'_, f64>::from_raw_parts(data.as_ptr(), 2, 3, 3, 1) };

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

        assert!(mf32.as_mut().transpose().to_owned().as_ref() == mf32.transpose());
        assert!(mf64.as_mut().transpose().to_owned().as_ref() == mf64.transpose());
        assert!(mf32c.as_mut().transpose().to_owned().as_ref() == mf32c.transpose());
        assert!(mf64c.as_mut().transpose().to_owned().as_ref() == mf64c.transpose());
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

        assert!(mf32.as_mut().adjoint().to_owned().as_ref() == mf32.adjoint());
        assert!(mf64.as_mut().adjoint().to_owned().as_ref() == mf64.adjoint());
        assert!(mf32c.as_mut().adjoint().to_owned().as_ref() == mf32c.adjoint());
        assert!(mf64c.as_mut().adjoint().to_owned().as_ref() == mf64c.adjoint());
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

        assert_eq!(matrix.col_ref(1), &[5.0, 6.0, 7.0]);
        assert_eq!(matrix.col_mut(0), &[1.0, 2.0, 3.0]);

        matrix.col_mut(0).copy_from_slice(&[-1.0, -2.0, -3.0]);

        let expected = mat![[-1.0, 5.0, 9.0], [-2.0, 6.0, 10.0], [-3.0, 7.0, 11.0f64]];
        assert_eq!(matrix, expected);
    }

    #[test]
    fn from_slice() {
        let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];

        let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        let view = MatRef::<f64>::from_column_major_slice(&slice, 3, 2);
        assert_eq!(expected, view);
        let view = MatMut::<f64>::from_column_major_slice(&mut slice, 3, 2);
        assert_eq!(expected, view);

        let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let view = MatRef::<f64>::from_row_major_slice(&slice, 3, 2);
        assert_eq!(expected, view);
        let view = MatMut::<f64>::from_row_major_slice(&mut slice, 3, 2);
        assert_eq!(expected, view);
    }

    #[test]
    #[should_panic]
    fn from_slice_too_big() {
        let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0_f64];
        MatRef::<f64>::from_column_major_slice(&slice, 3, 2);
    }

    #[test]
    #[should_panic]
    fn from_slice_too_small() {
        let slice = [1.0, 2.0, 3.0, 4.0, 5.0_f64];
        MatRef::<f64>::from_column_major_slice(&slice, 3, 2);
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
}
