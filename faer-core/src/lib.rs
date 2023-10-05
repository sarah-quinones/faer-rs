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
//! use faer_core::{mat, Mat, Scale};
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

pub use faer_entity::{ComplexField, Conjugate, Entity, RealField, SimpleEntity};

use assert2::{assert, debug_assert};
use coe::Coerce;
use core::{
    fmt::Debug, marker::PhantomData, mem::ManuallyDrop, ptr::NonNull, sync::atomic::AtomicUsize,
};
use dyn_stack::{DynArray, PodStack, SizeOverflow, StackReq};
use num_complex::Complex;
use pulp::Simd;
use reborrow::*;
use zip::Zip;

extern crate alloc;

pub mod householder;
#[doc(hidden)]
pub mod jacobi;

pub mod inverse;
pub mod mul;
pub mod permutation;
pub mod solve;
pub mod zip;

mod matrix_ops;

/// Thin wrapper used for scalar multiplication of a matrix by a scalar value.
pub use matrix_ops::Scale;

#[doc(hidden)]
pub mod simd;
type SimdGroup<E, S> = <E as Entity>::Group<<E as Entity>::SimdUnit<S>>;

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
}
impl c64 {
    #[inline(always)]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
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
        self * <c32 as ComplexField>::inv(rhs)
    }
}
impl core::ops::Div for c32 {
    type Output = c32;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        self * <Self as ComplexField>::inv(rhs)
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
        self * <c64 as ComplexField>::inv(rhs)
    }
}
impl core::ops::Div for c64 {
    type Output = c64;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        self * <Self as ComplexField>::inv(rhs)
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.re.fmt(f)?;
        f.write_str(" + ")?;
        self.im.fmt(f)?;
        f.write_str(" * I")
    }
}
impl Debug for c64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.re.fmt(f)?;
        f.write_str(" + ")?;
        self.im.fmt(f)?;
        f.write_str(" * I")
    }
}
impl Debug for c32conj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.re.fmt(f)?;
        f.write_str(" - ")?;
        self.neg_im.fmt(f)?;
        f.write_str(" * I")
    }
}
impl Debug for c64conj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.re.fmt(f)?;
        f.write_str(" - ")?;
        self.neg_im.fmt(f)?;
        f.write_str(" * I")
    }
}

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

pub trait AsMatRef<E: Entity> {
    fn as_mat_ref(&self) -> MatRef<'_, E>;
}
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

impl<E: Entity> matrixcompare_core::DenseAccess<E> for MatRef<'_, E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

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

impl<E: Entity> matrixcompare_core::DenseAccess<E> for MatMut<'_, E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

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

impl<E: Entity> matrixcompare_core::DenseAccess<E> for Mat<E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

#[inline]
fn div_ceil(a: usize, b: usize) -> usize {
    let d = a / b;
    let r = a % b;
    if r > 0 && b > 0 {
        d + 1
    } else {
        d
    }
}

impl ComplexField for c32 {
    type Real = f32;
    type Simd = pulp::Arch;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        Self {
            re: value as _,
            im: 0.0,
        }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn inv(self) -> Self {
        let inf = Self::Real::zero().inv();
        if self.is_nan() {
            // NAN
            Self::nan()
        } else if self == Self::zero() {
            // zero
            Self { re: inf, im: inf }
        } else if self.re == inf || self.im == inf {
            Self::zero()
        } else {
            let re = self.real().abs();
            let im = self.imag().abs();
            let max = if re > im { re } else { im };
            let max_inv = max.inv();
            let x = self.scale_real(max_inv);
            x.conj().scale_real(x.abs2().inv().mul(max_inv))
        }
    }

    #[inline(always)]
    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        let this: num_complex::Complex32 = self.into();
        num_complex::ComplexFloat::sqrt(this).into()
    }

    #[inline(always)]
    fn scale_real(self, rhs: Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn scale_power_of_two(self, rhs: Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn score(self) -> Self::Real {
        self.abs2()
    }

    #[inline(always)]
    fn abs(self) -> Self::Real {
        self.abs2().sqrt()
    }

    #[inline(always)]
    fn abs2(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    fn nan() -> Self {
        Self {
            re: Self::Real::NAN,
            im: Self::Real::NAN,
        }
    }

    #[inline(always)]
    fn from_real(real: Self::Real) -> Self {
        Self { re: real, im: 0.0 }
    }

    #[inline(always)]
    fn real(self) -> Self::Real {
        self.re
    }

    #[inline(always)]
    fn imag(self) -> Self::Real {
        self.im
    }

    #[inline(always)]
    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    #[inline(always)]
    fn slice_as_simd<S: Simd>(slice: &[Self::Unit]) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
        let (head, tail) = S::c32s_as_simd(bytemuck::cast_slice(slice));
        (bytemuck::cast_slice(head), bytemuck::cast_slice(tail))
    }

    #[inline(always)]
    fn slice_as_mut_simd<S: Simd>(
        slice: &mut [Self::Unit],
    ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
        let (head, tail) = S::c32s_as_mut_simd(bytemuck::cast_slice_mut(slice));
        (
            bytemuck::cast_slice_mut(head),
            bytemuck::cast_slice_mut(tail),
        )
    }

    #[inline(always)]
    fn partial_load_last_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.c32s_partial_load_last(bytemuck::cast_slice(slice))
    }

    #[inline(always)]
    fn partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.c32s_partial_store_last(bytemuck::cast_slice_mut(slice), values)
    }

    #[inline(always)]
    fn partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.c32s_partial_load(bytemuck::cast_slice(slice))
    }

    #[inline(always)]
    fn partial_store_unit<S: Simd>(simd: S, slice: &mut [Self::Unit], values: Self::SimdUnit<S>) {
        simd.c32s_partial_store(bytemuck::cast_slice_mut(slice), values)
    }

    #[inline(always)]
    fn simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
        simd.c32s_splat(pulp::cast(unit))
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
        simd.c32s_neg(values)
    }

    #[inline(always)]
    fn simd_conj<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
        let _ = simd;
        values
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c32s_add(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c32s_sub(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c32s_mul(lhs, rhs)
    }
    #[inline(always)]
    fn simd_scale_real<S: Simd>(
        simd: S,
        lhs: <Self::Real as Entity>::Group<<Self::Real as Entity>::SimdUnit<S>>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        if coe::is_same::<pulp::Scalar, S>() {
            let lhs: f32 = bytemuck::cast(lhs);
            let rhs: num_complex::Complex32 = bytemuck::cast(rhs);
            bytemuck::cast(lhs * rhs)
        } else {
            bytemuck::cast(simd.f32s_mul(lhs, bytemuck::cast(rhs)))
        }
    }
    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c32s_conj_mul(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c32s_mul_adde(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c32s_conj_mul_adde(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_reduce_add<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> Self {
        pulp::cast(simd.c32s_reduce_sum(values))
    }

    #[inline(always)]
    fn simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroup<Self, S>,
        acc: SimdGroup<Self::Real, S>,
    ) -> SimdGroup<Self::Real, S> {
        let _ = (simd, values, acc);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }
    #[inline(always)]
    fn simd_abs2<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
        let _ = (simd, values);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }
    #[inline(always)]
    fn simd_score<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
        let _ = (simd, values);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }
}
impl ComplexField for c64 {
    type Real = f64;
    type Simd = pulp::Arch;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        Self {
            re: value as _,
            im: 0.0,
        }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn inv(self) -> Self {
        let inf = Self::Real::zero().inv();
        if self.is_nan() {
            // NAN
            Self::nan()
        } else if self == Self::zero() {
            // zero
            Self { re: inf, im: inf }
        } else if self.re == inf || self.im == inf {
            Self::zero()
        } else {
            let re = self.real().abs();
            let im = self.imag().abs();
            let max = if re > im { re } else { im };
            let max_inv = max.inv();
            let x = self.scale_real(max_inv);
            x.conj().scale_real(x.abs2().inv().mul(max_inv))
        }
    }

    #[inline(always)]
    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        let this: num_complex::Complex64 = self.into();
        num_complex::ComplexFloat::sqrt(this).into()
    }

    #[inline(always)]
    fn scale_real(self, rhs: Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn scale_power_of_two(self, rhs: Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn score(self) -> Self::Real {
        self.abs2()
    }

    #[inline(always)]
    fn abs(self) -> Self::Real {
        self.abs2().sqrt()
    }

    #[inline(always)]
    fn abs2(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    fn nan() -> Self {
        Self {
            re: Self::Real::NAN,
            im: Self::Real::NAN,
        }
    }

    #[inline(always)]
    fn from_real(real: Self::Real) -> Self {
        Self { re: real, im: 0.0 }
    }

    #[inline(always)]
    fn real(self) -> Self::Real {
        self.re
    }

    #[inline(always)]
    fn imag(self) -> Self::Real {
        self.im
    }

    #[inline(always)]
    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    #[inline(always)]
    fn slice_as_simd<S: Simd>(slice: &[Self::Unit]) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
        let (head, tail) = S::c64s_as_simd(bytemuck::cast_slice(slice));
        (bytemuck::cast_slice(head), bytemuck::cast_slice(tail))
    }

    #[inline(always)]
    fn slice_as_mut_simd<S: Simd>(
        slice: &mut [Self::Unit],
    ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
        let (head, tail) = S::c64s_as_mut_simd(bytemuck::cast_slice_mut(slice));
        (
            bytemuck::cast_slice_mut(head),
            bytemuck::cast_slice_mut(tail),
        )
    }

    #[inline(always)]
    fn partial_load_last_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.c64s_partial_load_last(bytemuck::cast_slice(slice))
    }

    #[inline(always)]
    fn partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.c64s_partial_store_last(bytemuck::cast_slice_mut(slice), values)
    }

    #[inline(always)]
    fn partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.c64s_partial_load(bytemuck::cast_slice(slice))
    }

    #[inline(always)]
    fn partial_store_unit<S: Simd>(simd: S, slice: &mut [Self::Unit], values: Self::SimdUnit<S>) {
        simd.c64s_partial_store(bytemuck::cast_slice_mut(slice), values)
    }

    #[inline(always)]
    fn simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
        simd.c64s_splat(pulp::cast(unit))
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
        simd.c64s_neg(values)
    }

    #[inline(always)]
    fn simd_conj<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
        let _ = simd;
        values
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c64s_add(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c64s_sub(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c64s_mul(lhs, rhs)
    }
    #[inline(always)]
    fn simd_scale_real<S: Simd>(
        simd: S,
        lhs: <Self::Real as Entity>::Group<<Self::Real as Entity>::SimdUnit<S>>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        if coe::is_same::<pulp::Scalar, S>() {
            let lhs: f64 = bytemuck::cast(lhs);
            let rhs: num_complex::Complex64 = bytemuck::cast(rhs);
            bytemuck::cast(lhs * rhs)
        } else {
            bytemuck::cast(simd.f64s_mul(lhs, bytemuck::cast(rhs)))
        }
    }
    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c64s_conj_mul(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c64s_mul_adde(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.c64s_conj_mul_adde(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_reduce_add<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> Self {
        pulp::cast(simd.c64s_reduce_sum(values))
    }

    #[inline(always)]
    fn simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroup<Self, S>,
        acc: SimdGroup<Self::Real, S>,
    ) -> SimdGroup<Self::Real, S> {
        let _ = (simd, values, acc);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }
    #[inline(always)]
    fn simd_abs2<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
        let _ = (simd, values);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }
    #[inline(always)]
    fn simd_score<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
        let _ = (simd, values);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
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
    type Group<T> = T;
    type GroupCopy<T: Copy> = T;
    type Iter<I: Iterator> = I;

    const N_COMPONENTS: usize = 1;
    const HAS_SIMD: bool = true;
    const UNIT: Self::GroupCopy<()> = ();

    #[inline(always)]
    fn from_units(group: Self::Group<Self::Unit>) -> Self {
        group
    }

    #[inline(always)]
    fn into_units(self) -> Self::Group<Self::Unit> {
        self
    }

    #[inline(always)]
    fn as_ref<T>(group: &Self::Group<T>) -> Self::Group<&T> {
        group
    }

    #[inline(always)]
    fn as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T> {
        group
    }

    #[inline(always)]
    fn map<T, U>(group: Self::Group<T>, f: impl FnMut(T) -> U) -> Self::Group<U> {
        let mut f = f;
        f(group)
    }

    #[inline(always)]
    fn map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: Self::Group<T>,
        f: impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, Self::Group<U>) {
        let mut f = f;
        f(ctx, group)
    }

    #[inline(always)]
    fn zip<T, U>(first: Self::Group<T>, second: Self::Group<U>) -> Self::Group<(T, U)> {
        (first, second)
    }
    #[inline(always)]
    fn unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>) {
        zipped
    }

    #[inline(always)]
    fn into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter> {
        iter.into_iter()
    }

    #[inline(always)]
    fn from_copy<T: Copy>(group: Self::GroupCopy<T>) -> Self::Group<T> {
        group
    }

    #[inline(always)]
    fn into_copy<T: Copy>(group: Self::Group<T>) -> Self::GroupCopy<T> {
        group
    }
}
unsafe impl Entity for c32conj {
    type Unit = Self;
    type Index = u32;
    type SimdUnit<S: Simd> = S::c32s;
    type SimdMask<S: Simd> = S::m32s;
    type SimdIndex<S: Simd> = S::u32s;
    type Group<T> = T;
    type GroupCopy<T: Copy> = T;
    type Iter<I: Iterator> = I;

    const N_COMPONENTS: usize = 1;
    const HAS_SIMD: bool = true;
    const UNIT: Self::GroupCopy<()> = ();

    #[inline(always)]
    fn from_units(group: Self::Group<Self::Unit>) -> Self {
        group
    }

    #[inline(always)]
    fn into_units(self) -> Self::Group<Self::Unit> {
        self
    }

    #[inline(always)]
    fn as_ref<T>(group: &Self::Group<T>) -> Self::Group<&T> {
        group
    }

    #[inline(always)]
    fn as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T> {
        group
    }

    #[inline(always)]
    fn map<T, U>(group: Self::Group<T>, f: impl FnMut(T) -> U) -> Self::Group<U> {
        let mut f = f;
        f(group)
    }

    #[inline(always)]
    fn map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: Self::Group<T>,
        f: impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, Self::Group<U>) {
        let mut f = f;
        f(ctx, group)
    }

    #[inline(always)]
    fn zip<T, U>(first: Self::Group<T>, second: Self::Group<U>) -> Self::Group<(T, U)> {
        (first, second)
    }
    #[inline(always)]
    fn unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>) {
        zipped
    }

    #[inline(always)]
    fn into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter> {
        iter.into_iter()
    }

    #[inline(always)]
    fn from_copy<T: Copy>(group: Self::GroupCopy<T>) -> Self::Group<T> {
        group
    }

    #[inline(always)]
    fn into_copy<T: Copy>(group: Self::Group<T>) -> Self::GroupCopy<T> {
        group
    }
}

unsafe impl Entity for c64 {
    type Unit = Self;
    type Index = u64;
    type SimdUnit<S: Simd> = S::c64s;
    type SimdMask<S: Simd> = S::m64s;
    type SimdIndex<S: Simd> = S::u64s;
    type Group<T> = T;
    type GroupCopy<T: Copy> = T;
    type Iter<I: Iterator> = I;

    const N_COMPONENTS: usize = 1;
    const HAS_SIMD: bool = true;
    const UNIT: Self::GroupCopy<()> = ();

    #[inline(always)]
    fn from_units(group: Self::Group<Self::Unit>) -> Self {
        group
    }

    #[inline(always)]
    fn into_units(self) -> Self::Group<Self::Unit> {
        self
    }

    #[inline(always)]
    fn as_ref<T>(group: &Self::Group<T>) -> Self::Group<&T> {
        group
    }

    #[inline(always)]
    fn as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T> {
        group
    }

    #[inline(always)]
    fn map<T, U>(group: Self::Group<T>, f: impl FnMut(T) -> U) -> Self::Group<U> {
        let mut f = f;
        f(group)
    }

    #[inline(always)]
    fn map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: Self::Group<T>,
        f: impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, Self::Group<U>) {
        let mut f = f;
        f(ctx, group)
    }

    #[inline(always)]
    fn zip<T, U>(first: Self::Group<T>, second: Self::Group<U>) -> Self::Group<(T, U)> {
        (first, second)
    }
    #[inline(always)]
    fn unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>) {
        zipped
    }

    #[inline(always)]
    fn into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter> {
        iter.into_iter()
    }

    #[inline(always)]
    fn from_copy<T: Copy>(group: Self::GroupCopy<T>) -> Self::Group<T> {
        group
    }

    #[inline(always)]
    fn into_copy<T: Copy>(group: Self::Group<T>) -> Self::GroupCopy<T> {
        group
    }
}
unsafe impl Entity for c64conj {
    type Unit = Self;
    type Index = u64;
    type SimdUnit<S: Simd> = S::c64s;
    type SimdMask<S: Simd> = S::m64s;
    type SimdIndex<S: Simd> = S::u64s;
    type Group<T> = T;
    type GroupCopy<T: Copy> = T;
    type Iter<I: Iterator> = I;

    const N_COMPONENTS: usize = 1;
    const HAS_SIMD: bool = true;
    const UNIT: Self::GroupCopy<()> = ();

    #[inline(always)]
    fn from_units(group: Self::Group<Self::Unit>) -> Self {
        group
    }

    #[inline(always)]
    fn into_units(self) -> Self::Group<Self::Unit> {
        self
    }

    #[inline(always)]
    fn as_ref<T>(group: &Self::Group<T>) -> Self::Group<&T> {
        group
    }

    #[inline(always)]
    fn as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T> {
        group
    }

    #[inline(always)]
    fn map<T, U>(group: Self::Group<T>, f: impl FnMut(T) -> U) -> Self::Group<U> {
        let mut f = f;
        f(group)
    }

    #[inline(always)]
    fn map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: Self::Group<T>,
        f: impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, Self::Group<U>) {
        let mut f = f;
        f(ctx, group)
    }

    #[inline(always)]
    fn zip<T, U>(first: Self::Group<T>, second: Self::Group<U>) -> Self::Group<(T, U)> {
        (first, second)
    }
    #[inline(always)]
    fn unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>) {
        zipped
    }

    #[inline(always)]
    fn into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter> {
        iter.into_iter()
    }

    #[inline(always)]
    fn from_copy<T: Copy>(group: Self::GroupCopy<T>) -> Self::Group<T> {
        group
    }

    #[inline(always)]
    fn into_copy<T: Copy>(group: Self::Group<T>) -> Self::GroupCopy<T> {
        group
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

struct MatImpl<E: Entity> {
    ptr: E::GroupCopy<*mut E::Unit>,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
}

impl<E: Entity> Copy for MatImpl<E> {}
impl<E: Entity> Clone for MatImpl<E> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

/// Immutable view over a matrix, similar to an immutable reference to a 2D strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `MatRef<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions ([`std::mem::needs_drop::<E::Unit>()`] must be false). In
/// this case, care must be taken to not perform any operations that read the uninitialized values,
/// or form references to them, either directly through [`MatRef::read`], or indirectly through any
/// of the numerical library routines, unless it is explicitly permitted.
pub struct MatRef<'a, E: Entity> {
    inner: MatImpl<E>,
    __marker: PhantomData<&'a E>,
}

impl<E: Entity> Copy for MatRef<'_, E> {}
impl<E: Entity> Clone for MatRef<'_, E> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

/// Mutable view over a matrix, similar to a mutable reference to a 2D strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `MatMut<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions ([`std::mem::needs_drop::<E::Unit>()`] must be false). In
/// this case, care must be taken to not perform any operations that read the uninitialized values,
/// or form references to them, either directly through [`MatMut::read`], or indirectly through any
/// of the numerical library routines, unless it is explicitly permitted.
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
pub struct MatMut<'a, E: Entity> {
    inner: MatImpl<E>,
    __marker: PhantomData<&'a mut E>,
}

impl<'a, E: Entity> IntoConst for MatMut<'a, E> {
    type Target = MatRef<'a, E>;

    #[inline(always)]
    fn into_const(self) -> Self::Target {
        MatRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'short, 'a, E: Entity> Reborrow<'short> for MatMut<'a, E> {
    type Target = MatRef<'short, E>;

    #[inline(always)]
    fn rb(&'short self) -> Self::Target {
        MatRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'short, 'a, E: Entity> ReborrowMut<'short> for MatMut<'a, E> {
    type Target = MatMut<'short, E>;

    #[inline(always)]
    fn rb_mut(&'short mut self) -> Self::Target {
        MatMut {
            inner: self.inner,
            __marker: PhantomData,
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

unsafe impl<E: Entity> Send for MatRef<'_, E> {}
unsafe impl<E: Entity> Sync for MatRef<'_, E> {}
unsafe impl<E: Entity> Send for MatMut<'_, E> {}
unsafe impl<E: Entity> Sync for MatMut<'_, E> {}

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
        type Target = E::Group<&'a E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, row: usize, col: usize) -> Self::Target {
            unsafe { E::map(this.ptr_inbounds_at(row, col), |ptr| &*ptr) }
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
        type Target = E::Group<&'a mut E::Unit>;

        #[track_caller]
        #[inline(always)]
        unsafe fn get_unchecked(this: Self, row: usize, col: usize) -> Self::Target {
            unsafe { E::map(this.ptr_inbounds_at(row, col), |ptr| &mut *ptr) }
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
        slice: E::Group<&'a [E::Unit]>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        let size = usize::checked_mul(nrows, ncols).unwrap_or(usize::MAX);
        // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
        // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
        // we don't care
        E::map(
            E::copy(&slice),
            #[inline(always)]
            |slice| assert!(size == slice.len()),
        );
        unsafe {
            Self::from_raw_parts(
                E::map(
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
        slice: E::Group<&'a [E::Unit]>,
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
    /// * If [`std::mem::needs_drop::<E::Unit>()`], then all the addresses accessible by each
    /// matrix unit must point to initialized elements of type `E::Unit`. Otherwise, the values
    /// accessible by the matrix must be initialized at some point before they are read, or
    /// references to them are formed.
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
        ptr: E::Group<*const E::Unit>,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            inner: MatImpl {
                ptr: E::into_copy(E::map(ptr, |ptr| ptr as *mut E::Unit)),
                nrows,
                ncols,
                row_stride,
                col_stride,
            },
            __marker: PhantomData,
        }
    }

    /// Returns pointers to the matrix data.
    #[inline(always)]
    pub fn as_ptr(self) -> E::Group<*const E::Unit> {
        E::map(E::from_copy(self.inner.ptr), |ptr| ptr as *const E::Unit)
    }

    /// Returns the number of rows of the matrix.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.inner.nrows
    }

    /// Returns the number of columns of the matrix.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.inner.ncols
    }

    /// Returns the row stride of the matrix, specified in number of elements, not in bytes.
    #[inline(always)]
    pub fn row_stride(&self) -> isize {
        self.inner.row_stride
    }

    /// Returns the column stride of the matrix, specified in number of elements, not in bytes.
    #[inline(always)]
    pub fn col_stride(&self) -> isize {
        self.inner.col_stride
    }

    /// Returns raw pointers to the element at the given indices.
    #[inline(always)]
    pub fn ptr_at(self, row: usize, col: usize) -> E::Group<*const E::Unit> {
        E::map(self.as_ptr(), |ptr| {
            ptr.wrapping_offset(row as isize * self.inner.row_stride)
                .wrapping_offset(col as isize * self.inner.col_stride)
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
    pub unsafe fn ptr_inbounds_at(self, row: usize, col: usize) -> E::Group<*const E::Unit> {
        debug_assert!(row < self.nrows());
        debug_assert!(col < self.ncols());
        E::map(self.as_ptr(), |ptr| {
            ptr.offset(row as isize * self.inner.row_stride)
                .offset(col as isize * self.inner.col_stride)
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
        E::from_units(E::map(self.get_unchecked(row, col), |ptr| *ptr))
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
        E::from_units(E::map(self.get(row, col), |ptr| *ptr))
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
        Self {
            inner: MatImpl {
                ptr: self.inner.ptr,
                nrows: self.inner.ncols,
                ncols: self.inner.nrows,
                row_stride: self.inner.col_stride,
                col_stride: self.inner.row_stride,
            },
            __marker: PhantomData,
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
            // and that E::GroupCopy<X> == E::Conj::GroupCopy<X>
            MatRef {
                inner: MatImpl {
                    ptr: transmute_unchecked::<
                        E::GroupCopy<*mut E::Unit>,
                        <E::Conj as Entity>::GroupCopy<*mut <E::Conj as Entity>::Unit>,
                    >(self.inner.ptr),
                    nrows: self.inner.nrows,
                    ncols: self.inner.ncols,
                    row_stride: self.inner.row_stride,
                    col_stride: self.inner.col_stride,
                },
                __marker: PhantomData,
            }
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
                MatRef {
                    inner: MatImpl {
                        ptr: transmute_unchecked::<
                            E::GroupCopy<*mut E::Unit>,
                            <E::Canonical as Entity>::GroupCopy<
                                *mut <E::Canonical as Entity>::Unit,
                            >,
                        >(self.inner.ptr),
                        nrows: self.inner.nrows,
                        ncols: self.inner.ncols,
                        row_stride: self.inner.row_stride,
                        col_stride: self.inner.col_stride,
                    },
                    __marker: PhantomData,
                }
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

    /// Returns a view over the main diagonal of the matrix.
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
    /// let diagonal = view.diagonal();
    ///
    /// let expected = mat![[1.0], [6.0], [11.0]];
    /// assert_eq!(expected.as_ref(), diagonal);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn diagonal(self) -> Self {
        let size = self.nrows().min(self.ncols());
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        unsafe { Self::from_raw_parts(self.as_ptr(), size, 1, row_stride + col_stride, 0) }
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
            found_nan |= x.read().is_nan();
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
            all_finite &= x.read().is_finite();
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
            inner: self.inner,
            __marker: PhantomData,
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
        let chunk_count = div_ceil(self.ncols(), chunk_size);
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
    #[inline]
    #[track_caller]
    pub fn into_par_col_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E>> {
        use rayon::prelude::*;

        assert!(chunk_size > 0);
        let chunk_count = div_ceil(self.ncols(), chunk_size);
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
        E::from_group(self.get(row, col))
    }
}

impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for MatMut<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &E {
        E::from_group(self.rb().get(row, col))
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for MatMut<'_, E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut E {
        E::from_group(self.rb_mut().get(row, col))
    }
}

impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for Mat<E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &E {
        E::from_group(self.as_ref().get(row, col))
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for Mat<E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut E {
        E::from_group(self.as_mut().get(row, col))
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
        slice: E::Group<&'a mut [E::Unit]>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        let size = usize::checked_mul(nrows, ncols).unwrap_or(usize::MAX);
        // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
        // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
        // we don't care
        E::map(
            E::as_ref(&slice),
            #[inline(always)]
            |slice| assert!(size == slice.len()),
        );
        unsafe {
            Self::from_raw_parts(
                E::map(
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
        slice: E::Group<&'a mut [E::Unit]>,
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
    /// * If [`std::mem::needs_drop::<E::Unit>()`], then all the addresses accessible by each
    /// matrix unit must point to initialized elements of type `E::Unit`. Otherwise, the values
    /// accessible by the matrix must be initialized at some point before they are read, or
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
        ptr: E::Group<*mut E::Unit>,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            inner: MatImpl {
                ptr: E::into_copy(ptr),
                nrows,
                ncols,
                row_stride,
                col_stride,
            },
            __marker: PhantomData,
        }
    }

    /// Returns pointers to the matrix data.
    #[inline(always)]
    pub fn as_ptr(self) -> E::Group<*mut E::Unit> {
        E::from_copy(self.inner.ptr)
    }

    /// Returns the number of rows of the matrix.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.inner.nrows
    }

    /// Returns the number of columns of the matrix.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.inner.ncols
    }

    /// Returns the row stride of the matrix, specified in number of elements, not in bytes.
    #[inline(always)]
    pub fn row_stride(&self) -> isize {
        self.inner.row_stride
    }

    /// Returns the column stride of the matrix, specified in number of elements, not in bytes.
    #[inline(always)]
    pub fn col_stride(&self) -> isize {
        self.inner.col_stride
    }

    /// Returns raw pointers to the element at the given indices.
    #[inline(always)]
    pub fn ptr_at(self, row: usize, col: usize) -> E::Group<*mut E::Unit> {
        let row_stride = self.inner.row_stride;
        let col_stride = self.inner.col_stride;
        E::map(self.as_ptr(), |ptr| {
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
    pub unsafe fn ptr_inbounds_at(self, row: usize, col: usize) -> E::Group<*mut E::Unit> {
        debug_assert!(row < self.nrows());
        debug_assert!(col < self.ncols());
        let row_stride = self.inner.row_stride;
        let col_stride = self.inner.col_stride;
        E::map(self.as_ptr(), |ptr| {
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
        let units = value.into_units();
        let zipped = E::zip(units, self.rb_mut().ptr_inbounds_at(row, col));
        E::map(zipped, |(unit, ptr)| *ptr = unit);
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
        zipped!(self.rb_mut()).for_each(|mut x| x.write(E::zero()));
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
        Self {
            inner: MatImpl {
                ptr: self.inner.ptr,
                nrows: self.inner.ncols,
                ncols: self.inner.nrows,
                row_stride: self.inner.col_stride,
                col_stride: self.inner.row_stride,
            },
            __marker: PhantomData,
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

    /// Returns a view over the main diagonal of the matrix.
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
    /// let diagonal = view.diagonal();
    ///
    /// let mut expected = mat![[1.0], [6.0], [11.0]];
    /// assert_eq!(expected.as_mut(), diagonal);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn diagonal(self) -> Self {
        unsafe { self.into_const().diagonal().const_cast() }
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

impl<U: Conjugate, T: Conjugate<Canonical = U::Canonical>> PartialEq<MatRef<'_, U>>
    for MatRef<'_, T>
where
    T::Canonical: ComplexField,
{
    fn eq(&self, other: &MatRef<'_, U>) -> bool {
        let same_dims = self.nrows() == other.nrows() && self.ncols() == other.ncols();
        if !same_dims {
            false
        } else {
            let m = self.nrows();
            let n = self.ncols();

            for j in 0..n {
                for i in 0..m {
                    if !(self.read(i, j).canonicalize() == other.read(i, j).canonicalize()) {
                        return false;
                    }
                }
            }

            true
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
    ptr: E::GroupCopy<NonNull<E::Unit>>,
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

#[doc(hidden)]
#[inline(always)]
pub fn align_for<T: 'static>() -> usize {
    if is_vectorizable::<T>() {
        Ord::max(
            core::mem::size_of::<T>(),
            Ord::max(core::mem::align_of::<T>(), aligned_vec::CACHELINE_ALIGN),
        )
    } else {
        core::mem::align_of::<T>()
    }
}

impl<E: Entity> RawMat<E> {
    pub fn new(row_capacity: usize, col_capacity: usize) -> Self {
        // allocate the unit matrices
        let group = E::map(E::from_copy(E::UNIT), |()| {
            RawMatUnit::<E::Unit>::new(row_capacity, col_capacity)
        });

        let group = E::map(group, ManuallyDrop::new);

        Self {
            ptr: E::into_copy(E::map(group, |mat| mat.ptr)),
            row_capacity,
            col_capacity,
        }
    }
}

impl<E: Entity> Drop for RawMat<E> {
    fn drop(&mut self) {
        // implicitly dropped
        let _ = E::map(E::from_copy(self.ptr), |ptr| RawMatUnit {
            ptr,
            row_capacity: self.row_capacity,
            col_capacity: self.col_capacity,
        });
    }
}

struct BlockGuard<E: Entity> {
    ptr: E::GroupCopy<*mut E::Unit>,
    nrows: usize,
    ncols: usize,
    cs: isize,
}
struct ColGuard<E: Entity> {
    ptr: E::GroupCopy<*mut E::Unit>,
    nrows: usize,
}

impl<E: Entity> Drop for BlockGuard<E> {
    fn drop(&mut self) {
        for j in 0..self.ncols {
            E::map(E::from_copy(self.ptr), |ptr| {
                let ptr_j = ptr.wrapping_offset(j as isize * self.cs);
                // SAFETY: this is safe because we created these elements and need to
                // drop them
                let slice = unsafe { core::slice::from_raw_parts_mut(ptr_j, self.nrows) };
                unsafe { core::ptr::drop_in_place(slice) };
            });
        }
    }
}
impl<E: Entity> Drop for ColGuard<E> {
    fn drop(&mut self) {
        E::map(E::from_copy(self.ptr), |ptr| {
            // SAFETY: this is safe because we created these elements and need to
            // drop them
            let slice = unsafe { core::slice::from_raw_parts_mut(ptr, self.nrows) };
            unsafe { core::ptr::drop_in_place(slice) };
        });
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
#[repr(C)]
pub struct Mat<E: Entity> {
    raw: RawMat<E>,
    nrows: usize,
    ncols: usize,
}

#[repr(C)]
struct MatUnit<T: 'static> {
    raw: RawMatUnit<T>,
    nrows: usize,
    ncols: usize,
}

unsafe impl<E: Entity> Send for Mat<E> {}
unsafe impl<E: Entity> Sync for Mat<E> {}

impl<E: Entity> Clone for Mat<E> {
    fn clone(&self) -> Self {
        let this = self.as_ref();
        unsafe {
            Self::from_fn(self.nrows, self.ncols, |i, j| {
                E::from_units(E::deref(this.get_unchecked(i, j)))
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

impl<T> Drop for MatUnit<T> {
    fn drop(&mut self) {
        let mut ptr = self.raw.ptr.as_ptr();
        let nrows = self.nrows;
        let ncols = self.ncols;
        let cs = self.raw.row_capacity;

        for _ in 0..ncols {
            // SAFETY: these elements were previously created in this storage.
            unsafe {
                core::ptr::drop_in_place(core::slice::from_raw_parts_mut(ptr, nrows));
            }
            ptr = ptr.wrapping_add(cs);
        }
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
            raw: RawMat::<E> {
                ptr: E::map_copy(E::UNIT, |()| NonNull::<E::Unit>::dangling()),
                row_capacity: 0,
                col_capacity: 0,
            },
            nrows: 0,
            ncols: 0,
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
        Self {
            raw: RawMat::<E>::new(row_capacity, col_capacity),
            nrows: 0,
            ncols: 0,
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
        Self::from_fn(nrows, ncols, |_, _| E::zero())
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
        matrix.as_mut().diagonal().fill(E::one());
        matrix
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
        self.nrows = nrows;
        self.ncols = ncols;
    }

    /// Returns a pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr(&self) -> E::Group<*const E::Unit> {
        E::map(E::from_copy(self.raw.ptr), |ptr| {
            ptr.as_ptr() as *const E::Unit
        })
    }

    /// Returns a mutable pointer to the data of the matrix.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> E::Group<*mut E::Unit> {
        E::map(E::from_copy(self.raw.ptr), |ptr| ptr.as_ptr())
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

    /// Returns the row capacity, that is, the number of rows that the matrix is able to hold
    /// without needing to reallocate, excluding column insertions.
    #[inline]
    pub fn row_capacity(&self) -> usize {
        self.raw.row_capacity
    }

    /// Returns the column capacity, that is, the number of columns that the matrix is able to hold
    /// without needing to reallocate, excluding row insertions.
    #[inline]
    pub fn col_capacity(&self) -> usize {
        self.raw.col_capacity
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
            new_row_capacity =
                (new_row_capacity + (align_factor - 1)) / align_factor * align_factor;
        }

        use core::mem::swap;
        let nrows = self.nrows;
        let ncols = self.ncols;
        let old_row_capacity = self.raw.row_capacity;
        let old_col_capacity = self.raw.col_capacity;

        let mut this = Self::new();
        swap(self, &mut this);

        let mut this_group = E::map(E::from_copy(this.raw.ptr), |ptr| MatUnit {
            raw: RawMatUnit {
                ptr,
                row_capacity: old_row_capacity,
                col_capacity: old_col_capacity,
            },
            nrows,
            ncols,
        });

        E::map(E::as_mut(&mut this_group), |mat_unit| {
            mat_unit.do_reserve_exact(new_row_capacity, new_col_capacity);
        });

        let this_group = E::map(this_group, ManuallyDrop::new);
        this.raw.ptr = E::into_copy(E::map(this_group, |mat_unit| mat_unit.raw.ptr));
        this.raw.row_capacity = new_row_capacity;
        this.raw.col_capacity = new_col_capacity;
        swap(self, &mut this);
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
            self.raw.row_capacity = self.row_capacity().max(row_capacity);
            self.raw.col_capacity = self.col_capacity().max(col_capacity);
        } else {
            self.do_reserve_exact(row_capacity, col_capacity);
        }
    }

    unsafe fn erase_block(
        &mut self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) {
        debug_assert!(row_start <= row_end);
        debug_assert!(col_start <= col_end);

        E::map(self.as_mut_ptr(), |ptr| {
            for j in col_start..col_end {
                let ptr_j = ptr.wrapping_offset(j as isize * self.col_stride());

                // SAFETY: this points to a valid matrix element at index (_, j), which
                // is within bounds

                // SAFETY: we drop an object that is within its lifetime since the matrix
                // contains valid elements at each index within bounds
                core::ptr::drop_in_place(core::slice::from_raw_parts_mut(
                    ptr_j.add(row_start),
                    row_end - row_start,
                ));
            }
        });
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

        let ptr = E::into_copy(self.as_mut_ptr());

        let mut block_guard = BlockGuard::<E> {
            ptr: E::map_copy(ptr, |ptr| ptr.wrapping_add(row_start)),
            nrows: row_end - row_start,
            ncols: 0,
            cs: self.col_stride(),
        };

        for j in col_start..col_end {
            let ptr_j = E::map_copy(ptr, |ptr| {
                ptr.wrapping_offset(j as isize * self.col_stride())
            });

            // create a guard for the same purpose as the previous one
            let mut col_guard = ColGuard::<E> {
                // SAFETY: same as above
                ptr: E::map_copy(ptr_j, |ptr_j| ptr_j.wrapping_add(row_start)),
                nrows: 0,
            };

            for i in row_start..row_end {
                // SAFETY:
                // * pointer to element at index (i, j), which is within the
                // allocation since we reserved enough space
                // * writing to this memory region is sound since it is properly
                // aligned and valid for writes
                let ptr_ij = E::map(E::from_copy(ptr_j), |ptr_j| ptr_j.add(i));
                let value = E::into_units(f(i, j));

                E::map(E::zip(ptr_ij, value), |(ptr_ij, value)| {
                    core::ptr::write(ptr_ij, value)
                });
                col_guard.nrows += 1;
            }
            core::mem::forget(col_guard);
            block_guard.ncols += 1;
        }
        core::mem::forget(block_guard);
    }

    fn erase_last_cols(&mut self, new_ncols: usize) {
        let old_ncols = self.ncols();

        debug_assert!(new_ncols <= old_ncols);

        // change the size before dropping the elements, since if one of them panics the
        // matrix drop function will double drop them.
        self.ncols = new_ncols;

        unsafe {
            self.erase_block(0, self.nrows(), new_ncols, old_ncols);
        }
    }

    fn erase_last_rows(&mut self, new_nrows: usize) {
        let old_nrows = self.nrows();

        debug_assert!(new_nrows <= old_nrows);

        // see comment above
        self.nrows = new_nrows;
        unsafe {
            self.erase_block(new_nrows, old_nrows, 0, self.ncols());
        }
    }

    unsafe fn insert_last_cols_with<F: FnMut(usize, usize) -> E>(
        &mut self,
        f: &mut F,
        new_ncols: usize,
    ) {
        let old_ncols = self.ncols();

        debug_assert!(new_ncols > old_ncols);

        self.insert_block_with(f, 0, self.nrows(), old_ncols, new_ncols);
        self.ncols = new_ncols;
    }

    unsafe fn insert_last_rows_with<F: FnMut(usize, usize) -> E>(
        &mut self,
        f: &mut F,
        new_nrows: usize,
    ) {
        let old_nrows = self.nrows();

        debug_assert!(new_nrows > old_nrows);

        self.insert_block_with(f, old_nrows, new_nrows, 0, self.ncols());
        self.nrows = new_nrows;
    }

    /// Resizes the matrix in-place so that the new dimensions are `(new_nrows, new_ncols)`.
    /// Elements that are now out of bounds are dropped, while new elements are created with the
    /// given function `f`, so that elements at indices `(i, j)` are created by calling `f(i, j)`.
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
    pub fn col_ref(&self, col: usize) -> E::Group<&[E::Unit]> {
        assert!(col < self.ncols());
        let nrows = self.nrows();
        let ptr = self.as_ref().ptr_at(0, col);
        E::map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, nrows) },
        )
    }

    /// Returns a mutable reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    pub fn col_mut(&mut self, col: usize) -> E::Group<&mut [E::Unit]> {
        assert!(col < self.ncols());
        let nrows = self.nrows();
        let ptr = self.as_mut().ptr_at(0, col);
        E::map(
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
    /// Rayon parallelism.
    ///
    /// The code is possibly executed in parallel on the current thread, as well as the currently
    /// active rayon thread pool.
    ///
    /// The contained value represents a hint about the number of threads an implementation should
    /// use, but there is no way to guarantee how many or which threads will be used.
    ///
    /// A value of `0` treated as equivalent to `rayon::current_num_threads()`.
    Rayon(usize),
}

/// 0: Disable
/// 1: None
/// n >= 2: Rayon(n - 2)
///
/// default: Rayon(0)
static GLOBAL_PARALLELISM: AtomicUsize = AtomicUsize::new(2);

/// Causes functions that access global parallelism settings to panic.
pub fn disable_global_parallelism() {
    GLOBAL_PARALLELISM.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// Sets the global parallelism settings.
pub fn set_global_parallelism(parallelism: Parallelism) {
    let value = match parallelism {
        Parallelism::None => 1,
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
        n => Parallelism::Rayon(n - 2),
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
        Parallelism::Rayon(0) => rayon::current_num_threads(),
        Parallelism::Rayon(n_threads) => n_threads,
    }
}

enum DynMatUnitImpl<'a, T> {
    Init(DynArray<'a, T>),
}

/// A temporary matrix allocated from a [`PodStack`].
///
/// [`PodStack`]: dyn_stack::PodStack
pub struct DynMat<'a, E: Entity> {
    inner: E::Group<DynMatUnitImpl<'a, E::Unit>>,
    nrows: usize,
    ncols: usize,
    col_stride: usize,
}

impl<'a, E: Entity> DynMat<'a, E> {
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, E> {
        unsafe {
            MatRef::from_raw_parts(
                E::map(E::as_ref(&self.inner), |inner| match inner {
                    DynMatUnitImpl::Init(init) => init.as_ptr(),
                }),
                self.nrows,
                self.ncols,
                1,
                self.col_stride as isize,
            )
        }
    }
    #[inline]
    pub fn as_mut(&mut self) -> MatMut<'_, E> {
        unsafe {
            MatMut::from_raw_parts(
                E::map(E::as_mut(&mut self.inner), |inner| match inner {
                    DynMatUnitImpl::Init(init) => init.as_mut_ptr(),
                }),
                self.nrows,
                self.ncols,
                1,
                self.col_stride as isize,
            )
        }
    }
}

#[doc(hidden)]
#[inline]
pub fn round_up_to(n: usize, k: usize) -> usize {
    (n.checked_add(k - 1).unwrap()) / k * k
}

/// Creates a temporary matrix of constant values, from the given memory stack.
pub fn temp_mat_constant<E: ComplexField>(
    nrows: usize,
    ncols: usize,
    value: E,
    stack: PodStack<'_>,
) -> (DynMat<'_, E>, PodStack<'_>) {
    let col_stride = if is_vectorizable::<E::Unit>() {
        round_up_to(
            nrows,
            align_for::<E::Unit>() / core::mem::size_of::<E::Unit>(),
        )
    } else {
        nrows
    };

    let value = value.into_units();

    let (stack, alloc) = E::map_with_context(stack, value, |stack, value| {
        let (alloc, stack) =
            stack.make_aligned_with(ncols * col_stride, align_for::<E::Unit>(), |_| value);
        (stack, alloc)
    });

    (
        DynMat {
            inner: E::map(alloc, DynMatUnitImpl::Init),
            nrows,
            ncols,
            col_stride,
        },
        stack,
    )
}

/// Creates a temporary matrix of zero values, from the given memory stack.
pub fn temp_mat_zeroed<E: ComplexField>(
    nrows: usize,
    ncols: usize,
    stack: PodStack<'_>,
) -> (DynMat<'_, E>, PodStack<'_>) {
    let col_stride = if is_vectorizable::<E::Unit>() {
        round_up_to(
            nrows,
            align_for::<E::Unit>() / core::mem::size_of::<E::Unit>(),
        )
    } else {
        nrows
    };

    let value = E::into_units(E::zero());

    let (stack, alloc) = E::map_with_context(
        stack,
        value,
        #[inline(always)]
        |stack, value| {
            let (alloc, stack) = stack.make_aligned_with(
                ncols * col_stride,
                align_for::<E::Unit>(),
                #[inline(always)]
                |_| value,
            );
            (stack, alloc)
        },
    );

    (
        DynMat {
            inner: E::map(alloc, DynMatUnitImpl::Init),
            nrows,
            ncols,
            col_stride,
        },
        stack,
    )
}

/// Creates a temporary matrix of untouched values, from the given memory stack.
pub fn temp_mat_uninit<E: ComplexField>(
    nrows: usize,
    ncols: usize,
    stack: PodStack<'_>,
) -> (DynMat<'_, E>, PodStack<'_>) {
    let col_stride = if is_vectorizable::<E::Unit>() {
        round_up_to(
            nrows,
            align_for::<E::Unit>() / core::mem::size_of::<E::Unit>(),
        )
    } else {
        nrows
    };

    let (stack, alloc) = E::map_with_context(
        stack,
        E::from_copy(E::UNIT),
        #[inline(always)]
        |stack, ()| {
            let (alloc, stack) =
                stack.make_aligned_raw::<E::Unit>(ncols * col_stride, align_for::<E::Unit>());
            (stack, alloc)
        },
    );
    (
        DynMat {
            inner: E::map(alloc, DynMatUnitImpl::Init),
            nrows,
            ncols,
            col_stride,
        },
        stack,
    )
}

/// Returns the stack requirements for creating a temporary matrix with the given dimensions.
#[inline]
pub fn temp_mat_req<E: Entity>(nrows: usize, ncols: usize) -> Result<StackReq, SizeOverflow> {
    let col_stride = if is_vectorizable::<E::Unit>() {
        round_up_to(
            nrows,
            align_for::<E::Unit>() / core::mem::size_of::<E::Unit>(),
        )
    } else {
        nrows
    };

    let req = Ok(StackReq::empty());
    let (req, _) = E::map_with_context(req, E::from_copy(E::UNIT), |req, ()| {
        let req = match (
            req,
            StackReq::try_new_aligned::<E::Unit>(ncols * col_stride, align_for::<E::Unit>()),
        ) {
            (Ok(req), Ok(additional)) => req.try_and(additional),
            _ => Err(SizeOverflow),
        };

        (req, ())
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
                            Some(T::from_units(T::deref(self.0.get(0, j))))
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

impl<LhsE: Conjugate, RhsE: Conjugate<Canonical = LhsE::Canonical>> core::ops::Mul<MatRef<'_, RhsE>>
    for MatRef<'_, LhsE>
where
    LhsE::Canonical: ComplexField,
{
    type Output = Mat<LhsE::Canonical>;

    #[track_caller]
    fn mul(self, rhs: MatRef<'_, RhsE>) -> Self::Output {
        let mut out = Mat::zeros(self.nrows(), rhs.ncols());
        mul::matmul(
            out.as_mut(),
            self,
            rhs,
            None,
            LhsE::Canonical::one(),
            get_global_parallelism(),
        );
        out
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
                type Group<T> = T;
                type GroupCopy<T: Copy> = T;
                type Iter<I: Iterator> = I;

                const N_COMPONENTS: usize = 1;
                const HAS_SIMD: bool = false;
                const UNIT: Self::GroupCopy<()> = ();

                #[inline(always)]
                fn from_units(group: Self::Group<Self::Unit>) -> Self {
                    group
                }

                #[inline(always)]
                fn into_units(self) -> Self::Group<Self::Unit> {
                    self
                }

                #[inline(always)]
                fn as_ref<T>(group: &Self::Group<T>) -> Self::Group<&T> {
                    group
                }

                #[inline(always)]
                fn as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T> {
                    group
                }

                #[inline(always)]
                fn map<T, U>(group: Self::Group<T>, f: impl FnMut(T) -> U) -> Self::Group<U> {
                    let mut f = f;
                    f(group)
                }

                #[inline(always)]
                fn map_with_context<Ctx, T, U>(
                    ctx: Ctx,
                    group: Self::Group<T>,
                    f: impl FnMut(Ctx, T) -> (Ctx, U),
                ) -> (Ctx, Self::Group<U>) {
                    let mut f = f;
                    f(ctx, group)
                }

                #[inline(always)]
                fn zip<T, U>(first: Self::Group<T>, second: Self::Group<U>) -> Self::Group<(T, U)> {
                    (first, second)
                }
                #[inline(always)]
                fn unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>) {
                    zipped
                }

                #[inline(always)]
                fn into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter> {
                    iter.into_iter()
                }

                #[inline(always)]
                fn from_copy<T: Copy>(group: Self::GroupCopy<T>) -> Self::Group<T> {
                    group
                }

                #[inline(always)]
                fn into_copy<T: Copy>(group: Self::Group<T>) -> Self::GroupCopy<T> {
                    group
                }
            }
        };
    }

    use super::*;
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
    struct ZST;
    unsafe impl bytemuck::Zeroable for ZST {}
    unsafe impl bytemuck::Pod for ZST {}

    #[test]
    fn reserve_zst() {
        impl_unit_entity!(ZST);

        let mut m = Mat::<ZST>::new();

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
        let f = |_i, _j| ZST;
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
        x *= Scale(2.0);
        assert_eq!(x, expected);

        let expected = mat![[0.0, 4.0], [8.0, 12.0], [16.0, 20.0]];
        let mut x_mut = x.as_mut();
        x_mut *= Scale(2.0);
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
            assert!(<f32 as ComplexField>::is_finite(&1.0));
            assert!(!<f32 as ComplexField>::is_finite(&inf));
            assert!(!<f32 as ComplexField>::is_finite(&-inf));
            assert!(!<f32 as ComplexField>::is_finite(&nan));
        }
        {
            let x = c32::new(1.0, 2.0);
            assert!(<c32 as ComplexField>::is_finite(&x));

            let x = c32::new(inf, 2.0);
            assert!(!<c32 as ComplexField>::is_finite(&x));

            let x = c32::new(1.0, inf);
            assert!(!<c32 as ComplexField>::is_finite(&x));

            let x = c32::new(inf, inf);
            assert!(!<c32 as ComplexField>::is_finite(&x));

            let x = c32::new(nan, 2.0);
            assert!(!<c32 as ComplexField>::is_finite(&x));

            let x = c32::new(1.0, nan);
            assert!(!<c32 as ComplexField>::is_finite(&x));

            let x = c32::new(nan, nan);
            assert!(!<c32 as ComplexField>::is_finite(&x));
        }
    }

    #[test]
    fn test_iter() {
        let mut mat = Mat::from_fn(9, 10, |i, j| (i + j) as f64);
        let mut iter = mat.row_chunks_mut(4);

        let _0 = iter.next();
        let _1 = iter.next();
        let _2 = iter.next();
        let none = iter.next();

        assert!(_0 == Some(Mat::from_fn(4, 10, |i, j| (i + j) as f64).as_mut()));
        assert!(_1 == Some(Mat::from_fn(4, 10, |i, j| (i + j + 4) as f64).as_mut()));
        assert!(_2 == Some(Mat::from_fn(1, 10, |i, j| (i + j + 8) as f64).as_mut()));
        assert!(none == None);
    }
}
