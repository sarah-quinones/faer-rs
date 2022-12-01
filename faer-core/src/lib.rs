//! `faer` core module.
//!
//! This module contains:
//! - definitions of matrix structures ([`MatRef`], [`MatMut`], etc.),
//! - element-wise routines using multiple matrices,
//! - matrix multiplication routines,
//! - triangular matrix solve routines,
//! - etc.

#![warn(rust_2018_idioms)]
#![allow(clippy::too_many_arguments)]

use aligned_vec::CACHELINE_ALIGN;
use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use core::{
    any::TypeId,
    fmt::Debug,
    marker::PhantomData,
    mem::{size_of, MaybeUninit},
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
    ptr::NonNull,
};
use dyn_stack::{SizeOverflow, StackReq};
use rayon::prelude::IndexedParallelIterator;

use core::mem::transmute_copy;
/// Complex floating point number type, where the real and imaginary parts each occupy 32 bits.
pub use gemm::c32;
/// Complex floating point number type, where the real and imaginary parts each occupy 64 bits.
pub use gemm::c64;
use iter::*;
use num_complex::{Complex, ComplexFloat};
use reborrow::*;

extern crate alloc;

pub mod inverse;
pub mod mul;
pub mod solve;
pub mod zip;

#[doc(hidden)]
pub mod householder;
pub mod permutation;

/// Indicates whether the corresponding operand should be conjugated or not.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Conj {
    /// Do not conjugate
    No,
    /// Do conjugate
    Yes,
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

/// Trait that describes a complex number field.
///
/// Real numbers can also be seen as complex numbers, where the imaginary part is always zero.
///
/// # Note
///
/// The implementation currently implies [`Copy`], but this may be replaced by [`Clone`] in a
/// future version of this library.
pub trait ComplexField:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Send
    + Sync
    + Debug
    + 'static
{
    type Real: RealField;

    /// Returns a complex number whose real part is equal to `real`, and a zero imaginary part.
    fn from_real(real: Self::Real) -> Self;
    /// Returns the real and imaginary part.
    fn into_real_imag(self) -> (Self::Real, Self::Real);
    /// Returns the real part.
    #[inline(always)]
    fn real(self) -> Self::Real {
        self.into_real_imag().0
    }
    /// Returns the imaginary part.
    #[inline(always)]
    fn imag(self) -> Self::Real {
        self.into_real_imag().1
    }

    /// Returns the value representing `0.0`.
    fn zero() -> Self;
    /// Returns the value representing `1.0`.
    fn one() -> Self;

    /// Returns the inverse of the number.
    fn inv(self) -> Self;
    /// Returns the conjugate of the number.
    fn conj(self) -> Self;
    /// Returns the square root of the number.
    fn sqrt(self) -> Self;
    /// Returns the input, scaled by `factor`.
    #[inline(always)]
    fn scale(self, factor: Self::Real) -> Self {
        self * Self::from_real(factor)
    }
    /// Returns either the norm or squared norm of the number.
    ///
    /// An implementation may choose either, so long as it chooses consistently.
    fn score(self) -> Self::Real;

    #[inline(always)]
    fn abs(self) -> Self::Real {
        (self * self.conj()).real().sqrt()
    }
}

/// The imaginary unit. This struct is useful for copy-pasting matrix [`Debug`] output as code.
///
/// # Example
///
/// ```
/// use faer_core::{c64, I};
///
/// let z = 1.0 + 2.0 * I;
///
/// assert_eq!(z, c64::new(1.0, 2.0));
/// ```
pub struct I;
impl Mul<f32> for I {
    type Output = c32;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        c32 { re: 0.0, im: rhs }
    }
}
impl Mul<I> for f32 {
    type Output = c32;

    #[inline]
    fn mul(self, rhs: I) -> Self::Output {
        rhs * self
    }
}
impl Mul<f64> for I {
    type Output = c64;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        c64 { re: 0.0, im: rhs }
    }
}
impl Mul<I> for f64 {
    type Output = c64;

    #[inline]
    fn mul(self, rhs: I) -> Self::Output {
        rhs * self
    }
}

/// Trait that describes a real number field.
pub trait RealField: ComplexField<Real = Self> + PartialOrd {}

impl RealField for f32 {}
impl ComplexField for f32 {
    type Real = f32;

    #[inline(always)]
    fn from_real(real: Self::Real) -> Self {
        real
    }

    #[inline(always)]
    fn into_real_imag(self) -> (Self::Real, Self::Real) {
        (self, 0.0)
    }

    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn one() -> Self {
        1.0
    }

    #[inline(always)]
    fn inv(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn conj(self) -> Self {
        self
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline(always)]
    fn score(self) -> Self::Real {
        self.abs()
    }

    #[inline(always)]
    fn abs(self) -> Self::Real {
        self.abs()
    }
}

impl RealField for f64 {}
impl ComplexField for f64 {
    type Real = f64;

    #[inline(always)]
    fn from_real(real: Self::Real) -> Self {
        real
    }

    #[inline(always)]
    fn into_real_imag(self) -> (Self::Real, Self::Real) {
        (self, 0.0)
    }

    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn one() -> Self {
        1.0
    }

    #[inline(always)]
    fn inv(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn conj(self) -> Self {
        self
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline(always)]
    fn score(self) -> Self::Real {
        self.abs()
    }

    #[inline(always)]
    fn abs(self) -> Self::Real {
        self.abs()
    }
}

impl ComplexField for c32 {
    type Real = f32;

    #[inline(always)]
    fn from_real(real: Self::Real) -> Self {
        c32 { re: real, im: 0.0 }
    }

    #[inline(always)]
    fn into_real_imag(self) -> (Self::Real, Self::Real) {
        (self.re, self.im)
    }

    #[inline(always)]
    fn zero() -> Self {
        c32 { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    fn one() -> Self {
        c32 { re: 1.0, im: 0.0 }
    }

    #[inline(always)]
    fn inv(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn conj(self) -> Self {
        c32 {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        <Self as ComplexFloat>::sqrt(self)
    }

    #[inline(always)]
    fn score(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }
}

impl ComplexField for c64 {
    type Real = f64;

    #[inline(always)]
    fn from_real(real: Self::Real) -> Self {
        c64 { re: real, im: 0.0 }
    }

    #[inline(always)]
    fn into_real_imag(self) -> (Self::Real, Self::Real) {
        (self.re, self.im)
    }

    #[inline(always)]
    fn zero() -> Self {
        c64 { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    fn one() -> Self {
        c64 { re: 1.0, im: 0.0 }
    }

    #[inline(always)]
    fn inv(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn conj(self) -> Self {
        c64 {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        <Self as ComplexFloat>::sqrt(self)
    }

    #[inline(always)]
    fn score(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }
}

use zip::*;

mod seal {
    use super::*;

    pub trait Seal {}
    impl<'a, T> Seal for MatRef<'a, T> {}
    impl<'a, T> Seal for MatMut<'a, T> {}
    impl<'a, T> Seal for ColRef<'a, T> {}
    impl<'a, T> Seal for ColMut<'a, T> {}
    impl<'a, T> Seal for RowRef<'a, T> {}
    impl<'a, T> Seal for RowMut<'a, T> {}
}

#[inline]
#[doc(hidden)]
pub fn join_raw(
    op_a: impl Send + for<'a> FnOnce(Parallelism),
    op_b: impl Send + for<'a> FnOnce(Parallelism),
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

#[inline]
#[doc(hidden)]
pub fn parallelism_degree(parallelism: Parallelism) -> usize {
    match parallelism {
        Parallelism::None => 1,
        Parallelism::Rayon(0) => rayon::current_num_threads(),
        Parallelism::Rayon(n_threads) => n_threads,
    }
}

struct MatrixSliceBase<T> {
    ptr: NonNull<T>,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
}
struct VecSliceBase<T> {
    ptr: NonNull<T>,
    len: usize,
    stride: isize,
}
impl<T> Copy for MatrixSliceBase<T> {}
impl<T> Clone for MatrixSliceBase<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for VecSliceBase<T> {}
impl<T> Clone for VecSliceBase<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

/// Matrix view with general row and column strides.
pub struct MatRef<'a, T> {
    base: MatrixSliceBase<T>,
    _marker: PhantomData<&'a T>,
}

/// Mutable matrix view with general row and column strides.
///
/// For usage examples, see [`MatRef`].
pub struct MatMut<'a, T> {
    base: MatrixSliceBase<T>,
    _marker: PhantomData<&'a mut T>,
}

/// Row vector view with general column stride.
///
/// For usage examples, see [`MatRef`].
pub struct RowRef<'a, T> {
    base: VecSliceBase<T>,
    _marker: PhantomData<&'a T>,
}

/// Mutable row vector view with general column stride.
///
/// For usage examples, see [`MatRef`].
pub struct RowMut<'a, T> {
    base: VecSliceBase<T>,
    _marker: PhantomData<&'a mut T>,
}

/// Column vector view with general row stride.
///
/// For usage examples, see [`MatRef`].
pub struct ColRef<'a, T> {
    base: VecSliceBase<T>,
    _marker: PhantomData<&'a T>,
}

/// Mutable column vector view with general row stride.
///
/// For usage examples, see [`MatRef`].
pub struct ColMut<'a, T> {
    base: VecSliceBase<T>,
    _marker: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: Sync> Sync for MatRef<'a, T> {}
unsafe impl<'a, T: Sync> Send for MatRef<'a, T> {}
unsafe impl<'a, T: Sync> Sync for MatMut<'a, T> {}
unsafe impl<'a, T: Send> Send for MatMut<'a, T> {}

unsafe impl<'a, T: Sync> Sync for RowRef<'a, T> {}
unsafe impl<'a, T: Sync> Send for RowRef<'a, T> {}
unsafe impl<'a, T: Sync> Sync for RowMut<'a, T> {}
unsafe impl<'a, T: Send> Send for RowMut<'a, T> {}

unsafe impl<'a, T: Sync> Sync for ColRef<'a, T> {}
unsafe impl<'a, T: Sync> Send for ColRef<'a, T> {}
unsafe impl<'a, T: Sync> Sync for ColMut<'a, T> {}
unsafe impl<'a, T: Send> Send for ColMut<'a, T> {}

impl<'a, T> Copy for MatRef<'a, T> {}
impl<'a, T> Copy for RowRef<'a, T> {}
impl<'a, T> Copy for ColRef<'a, T> {}

impl<'a, T> Clone for MatRef<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Clone for RowRef<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Clone for ColRef<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'b, 'a, T> Reborrow<'b> for MatRef<'a, T> {
    type Target = MatRef<'b, T>;
    #[inline]
    fn rb(&'b self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for MatRef<'a, T> {
    type Target = MatRef<'b, T>;
    #[inline]
    fn rb_mut(&'b mut self) -> Self::Target {
        *self
    }
}

impl<'b, 'a, T> Reborrow<'b> for MatMut<'a, T> {
    type Target = MatRef<'b, T>;
    #[inline]
    fn rb(&'b self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for MatMut<'a, T> {
    type Target = MatMut<'b, T>;
    #[inline]
    fn rb_mut(&'b mut self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'b, 'a, T> Reborrow<'b> for RowRef<'a, T> {
    type Target = RowRef<'b, T>;
    #[inline]
    fn rb(&'b self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for RowRef<'a, T> {
    type Target = RowRef<'b, T>;
    #[inline]
    fn rb_mut(&'b mut self) -> Self::Target {
        *self
    }
}

impl<'b, 'a, T> Reborrow<'b> for RowMut<'a, T> {
    type Target = RowRef<'b, T>;
    #[inline]
    fn rb(&'b self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for RowMut<'a, T> {
    type Target = RowMut<'b, T>;
    #[inline]
    fn rb_mut(&'b mut self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'b, 'a, T> Reborrow<'b> for ColRef<'a, T> {
    type Target = ColRef<'b, T>;
    #[inline]
    fn rb(&'b self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for ColRef<'a, T> {
    type Target = ColRef<'b, T>;
    #[inline]
    fn rb_mut(&'b mut self) -> Self::Target {
        *self
    }
}

impl<'b, 'a, T> Reborrow<'b> for ColMut<'a, T> {
    type Target = ColRef<'b, T>;
    #[inline]
    fn rb(&'b self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for ColMut<'a, T> {
    type Target = ColMut<'b, T>;
    #[inline]
    fn rb_mut(&'b mut self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> IntoConst for MatRef<'a, T> {
    type Target = MatRef<'a, T>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}
impl<'a, T> IntoConst for MatMut<'a, T> {
    type Target = MatRef<'a, T>;

    #[inline]
    fn into_const(self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> IntoConst for ColRef<'a, T> {
    type Target = ColRef<'a, T>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}
impl<'a, T> IntoConst for ColMut<'a, T> {
    type Target = ColRef<'a, T>;

    #[inline]
    fn into_const(self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> IntoConst for RowRef<'a, T> {
    type Target = RowRef<'a, T>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}
impl<'a, T> IntoConst for RowMut<'a, T> {
    type Target = RowRef<'a, T>;

    #[inline]
    fn into_const(self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'a, U, T: PartialEq<U>> PartialEq<MatRef<'a, U>> for MatRef<'a, T> {
    fn eq(&self, other: &MatRef<'a, U>) -> bool {
        let same_dims = self.nrows() == other.nrows() && self.ncols() == other.ncols();
        if !same_dims {
            return false;
        } else {
            let m = self.nrows();
            let n = self.ncols();

            for j in 0..n {
                for i in 0..m {
                    unsafe {
                        if !(self.get_unchecked(i, j) == other.get_unchecked(i, j)) {
                            return false;
                        }
                    }
                }
            }

            return true;
        }
    }
}

impl<'a, U, T: PartialEq<U>> PartialEq<MatRef<'a, U>> for MatMut<'a, T> {
    #[inline]
    fn eq(&self, other: &MatRef<'a, U>) -> bool {
        self.rb() == other.rb()
    }
}

impl<'a, U, T: PartialEq<U>> PartialEq<MatMut<'a, U>> for MatRef<'a, T> {
    #[inline]
    fn eq(&self, other: &MatMut<'a, U>) -> bool {
        self.rb() == other.rb()
    }
}

impl<'a, U, T: PartialEq<U>> PartialEq<MatMut<'a, U>> for MatMut<'a, T> {
    #[inline]
    fn eq(&self, other: &MatMut<'a, U>) -> bool {
        self.rb() == other.rb()
    }
}

impl<U, T: PartialEq<U>> PartialEq<Mat<U>> for Mat<T> {
    #[inline]
    fn eq(&self, other: &Mat<U>) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl<'a, T> MatRef<'a, T> {
    /// Returns a matrix slice from the given arguments.  
    /// `ptr`: pointer to the first element of the matrix.  
    /// `nrows`: number of rows of the matrix.  
    /// `ncols`: number of columns of the matrix.  
    /// `row_stride`: offset between the first elements of two successive rows in the matrix.
    /// `col_stride`: offset between the first elements of two successive columns in the matrix.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `i < nrows` and `j < ncols`,  
    /// `ptr.offset(i as isize * row_stride + j as isize * col_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never accessed.  
    /// The referenced memory must not be mutated during the lifetime `'a`.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::MatRef;
    ///
    /// let nan = f64::NAN;
    /// let data = vec![0.0, 1.0, nan, 2.0, 3.0, nan, 4.0, 5.0];
    ///
    /// let nrows = 2;
    /// let ncols = 3;
    /// let row_stride = 1;
    /// let col_stride = 3;
    /// let m = unsafe { MatRef::from_raw_parts(data.as_ptr(), nrows, ncols, row_stride, col_stride) };
    ///
    /// assert_eq!(m.nrows(), 2);
    /// assert_eq!(m.ncols(), 3);
    ///
    /// assert_eq!(m[(0, 0)], 0.0);
    /// assert_eq!(m[(1, 0)], 1.0);
    /// assert_eq!(m[(0, 1)], 2.0);
    /// assert_eq!(m[(1, 1)], 3.0);
    /// assert_eq!(m[(0, 2)], 4.0);
    /// assert_eq!(m[(1, 2)], 5.0);
    /// ```
    #[inline]
    pub unsafe fn from_raw_parts(
        ptr: *const T,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            base: MatrixSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr as *mut T),
                nrows,
                ncols,
                row_stride,
                col_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a pointer to the first (top left) element of the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]];
    /// let m = m.as_ref();
    ///
    /// assert_eq!(unsafe { *m.as_ptr() }, 0.0);
    /// ```
    #[inline]
    pub fn as_ptr(self) -> *const T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]];
    /// let m = m.as_ref();
    ///
    /// assert_eq!(m.nrows(), 2);
    /// ```
    #[inline]
    pub fn nrows(&self) -> usize {
        self.base.nrows
    }

    /// Returns the number of columns of the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]];
    /// let m = m.as_ref();
    ///
    /// assert_eq!(m.ncols(), 3);
    /// ```
    #[inline]
    pub fn ncols(&self) -> usize {
        self.base.ncols
    }

    /// Returns the offset between the first elements of two successive rows in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]];
    /// let m = m.as_ref();
    ///
    /// assert_eq!(m.row_stride(), 1); // mat! returns a column major matrix
    /// ```
    #[inline]
    pub fn row_stride(&self) -> isize {
        self.base.row_stride
    }

    /// Returns the offset between the first elements of two successive columns in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]];
    /// let m = m.as_ref();
    ///
    /// assert!(m.col_stride() >= 2); // the stride has to be at least 2, since the matrix is
    ///                               // column major and each column has 2 elements.
    /// ```
    #[inline]
    pub fn col_stride(&self) -> isize {
        self.base.col_stride
    }

    /// Returns a pointer to the element at position (i, j) in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]];
    /// let m = m.as_ref();
    ///
    /// assert_eq!(unsafe { *m.ptr_at(1, 2) }, 5.0);
    /// ```
    #[inline]
    pub fn ptr_at(self, i: usize, j: usize) -> *const T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(i as isize * self.row_stride())
            .wrapping_offset(j as isize * self.col_stride())
    }

    /// Returns a pointer to the element at position (i, j) in the matrix, assuming it falls within
    /// its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i < self.nrows()`,
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    ///
    /// # Example
    ///
    /// See [`Self::ptr_at`].
    #[track_caller]
    #[inline]
    pub unsafe fn ptr_in_bounds_at_unchecked(self, i: usize, j: usize) -> *const T {
        fancy_debug_assert!(i < self.nrows());
        fancy_debug_assert!(j < self.ncols());
        self.base
            .ptr
            .as_ptr()
            .offset(i as isize * self.row_stride())
            .offset(j as isize * self.col_stride())
    }

    /// Returns a pointer to the element at position (i, j) in the matrix, while asserting that
    /// it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i < self.nrows()`,
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, it panics.
    ///
    /// # Example
    ///
    /// See [`Self::ptr_at`].
    #[track_caller]
    #[inline]
    pub fn ptr_in_bounds_at(self, i: usize, j: usize) -> *const T {
        fancy_assert!(i < self.nrows());
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(i, j) }
    }

    /// Splits the matrix into four corner parts in the following order: top left, top right,
    /// bottom left, bottom right.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `j <= self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    ///
    /// # Example
    ///
    /// See [`Self::split_at`].
    #[track_caller]
    #[inline]
    pub unsafe fn split_at_unchecked(self, i: usize, j: usize) -> (Self, Self, Self, Self) {
        fancy_debug_assert!(i <= self.nrows());
        fancy_debug_assert!(j <= self.ncols());
        let ptr = self.base.ptr.as_ptr();
        let cs = self.col_stride();
        let rs = self.row_stride();
        (
            Self::from_raw_parts(ptr, i, j, rs, cs),
            Self::from_raw_parts(
                ptr.wrapping_offset(j as isize * cs),
                i,
                self.ncols() - j,
                rs,
                cs,
            ),
            Self::from_raw_parts(
                ptr.wrapping_offset(i as isize * rs),
                self.nrows() - i,
                j,
                rs,
                cs,
            ),
            Self::from_raw_parts(
                ptr.wrapping_offset(i as isize * rs)
                    .wrapping_offset(j as isize * cs),
                self.nrows() - i,
                self.ncols() - j,
                rs,
                cs,
            ),
        )
    }

    /// Splits the matrix into four corner parts in the following order: top left, top right,
    /// bottom left, bottom right.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `j <= self.ncols()`.
    ///
    /// Otherwise, it panics.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]];
    /// let m = m.as_ref();
    /// let (top_left, top_right, bot_left, bot_right) = m.split_at(1, 1);
    ///
    /// assert_eq!(top_left.nrows(), 1);
    /// assert_eq!(top_left.ncols(), 1);
    /// assert_eq!(top_left[(0, 0)], 0.0);
    ///
    /// assert_eq!(top_right.nrows(), 1);
    /// assert_eq!(top_right.ncols(), 2);
    /// assert_eq!(top_right[(0, 0)], 2.0);
    /// assert_eq!(top_right[(0, 1)], 4.0);
    ///
    /// assert_eq!(bot_left.nrows(), 1);
    /// assert_eq!(bot_left.ncols(), 1);
    /// assert_eq!(bot_left[(0, 0)], 1.0);
    ///
    /// assert_eq!(bot_right.nrows(), 1);
    /// assert_eq!(bot_right.ncols(), 2);
    /// assert_eq!(bot_right[(0, 0)], 3.0);
    /// assert_eq!(bot_right[(0, 1)], 5.0);
    /// ```
    #[track_caller]
    #[inline]
    pub fn split_at(self, i: usize, j: usize) -> (Self, Self, Self, Self) {
        fancy_assert!(i <= self.nrows());
        fancy_assert!(j <= self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(i, j) }
    }

    /// Splits the matrix horizontally into two parts in the following order: top, bottom.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn split_at_row_unchecked(self, i: usize) -> (Self, Self) {
        let (_, top, _, bottom) = self.split_at_unchecked(i, 0);
        (top, bottom)
    }

    /// Splits the matrix horizontally into two parts in the following order: top, bottom.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn split_at_row(self, i: usize) -> (Self, Self) {
        let (_, top, _, bottom) = self.split_at(i, 0);
        (top, bottom)
    }

    /// Splits the matrix vertically into two parts in the following order: left, right.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j <= self.nrows()`,
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn split_at_col_unchecked(self, j: usize) -> (Self, Self) {
        let (_, _, left, right) = self.split_at_unchecked(0, j);
        (left, right)
    }

    /// Splits the matrix vertically into two parts in the following order: left, right.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j <= self.nrows()`,
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn split_at_col(self, j: usize) -> (Self, Self) {
        let (_, _, left, right) = self.split_at(0, j);
        (left, right)
    }

    /// Returns a reference to the element at position (i, j), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i < self.nrows()`,
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    ///
    /// # Example
    ///
    /// See [`Self::get`].
    #[track_caller]
    #[inline]
    pub unsafe fn get_unchecked(self, i: usize, j: usize) -> &'a T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &*self.ptr_in_bounds_at_unchecked(i, j)
    }

    /// Returns a reference to the element at position (i, j), or panics if the indices are out of
    /// bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![
    ///     [0.0, 3.0, 6.0, 9.0],
    ///     [1.0, 4.0, 7.0, 10.0],
    ///     [2.0, 5.0, 8.0, 11.0],
    /// ];
    /// let m = m.as_ref();
    ///
    /// assert_eq!(m.get(1, 2), &7.0);
    /// ```
    #[track_caller]
    #[inline]
    pub fn get(self, i: usize, j: usize) -> &'a T {
        fancy_assert!(i < self.nrows());
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked.
        unsafe { self.get_unchecked(i, j) }
    }

    /// Returns the `i`-th row of the matrix, with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i < self.nrows()`.
    ///
    /// Otherwise, the behavior is undefined.
    ///
    /// # Example
    ///
    /// See [`Self::row`].
    #[track_caller]
    #[inline]
    pub unsafe fn row_unchecked(self, i: usize) -> RowRef<'a, T> {
        fancy_debug_assert!(i < self.nrows());
        let ncols = self.ncols();
        let cs = self.col_stride();
        RowRef::from_raw_parts(self.ptr_at(i, 0), ncols, cs)
    }

    /// Returns the `i`-th row of the matrix.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i < self.nrows()`.
    ///
    /// Otherwise, it panics.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![
    ///     [0.0, 3.0, 6.0, 9.0],
    ///     [1.0, 4.0, 7.0, 10.0],
    ///     [2.0, 5.0, 8.0, 11.0],
    /// ];
    /// let m = m.as_ref();
    ///
    /// let r = m.row(1);
    ///
    /// assert_eq!(r[2], 7.0);
    /// ```
    #[track_caller]
    #[inline]
    pub fn row(self, i: usize) -> RowRef<'a, T> {
        fancy_assert!(i < self.nrows());
        // SAFETY: bounds have been checked
        unsafe { self.row_unchecked(i) }
    }

    /// Returns the `j`-th column of the matrix, with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    ///
    /// # Example
    ///
    /// See [`Self::col`].
    #[track_caller]
    #[inline]
    pub unsafe fn col_unchecked(self, j: usize) -> ColRef<'a, T> {
        fancy_debug_assert!(j < self.ncols());
        let nrows = self.nrows();
        let rs = self.row_stride();
        ColRef::from_raw_parts(self.ptr_at(0, j), nrows, rs)
    }

    /// Returns the `j`-th column of the matrix.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, it panics.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![
    ///     [0.0, 3.0, 6.0, 9.0],
    ///     [1.0, 4.0, 7.0, 10.0],
    ///     [2.0, 5.0, 8.0, 11.0],
    /// ];
    /// let m = m.as_ref();
    ///
    /// let c = m.col(1);
    ///
    /// assert_eq!(c[2], 5.0);
    /// ```
    #[track_caller]
    #[inline]
    pub fn col(self, j: usize) -> ColRef<'a, T> {
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked.
        unsafe { self.col_unchecked(j) }
    }

    /// Returns the transpose of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![
    ///     [0.0, 3.0, 6.0, 9.0],
    ///     [1.0, 4.0, 7.0, 10.0],
    ///     [2.0, 5.0, 8.0, 11.0],
    /// ];
    /// let m = m.as_ref();
    ///
    /// let t = m.transpose();
    ///
    /// assert_eq!(t.nrows(), m.ncols());
    /// assert_eq!(t.ncols(), m.nrows());
    ///
    /// assert_eq!(t[(0, 0)], m[(0, 0)]);
    /// assert_eq!(t[(1, 0)], m[(0, 1)]);
    /// assert_eq!(t[(2, 0)], m[(0, 2)]);
    /// assert_eq!(t[(3, 0)], m[(0, 3)]);
    ///
    /// assert_eq!(t[(0, 1)], m[(1, 0)]);
    /// assert_eq!(t[(1, 1)], m[(1, 1)]);
    /// assert_eq!(t[(2, 1)], m[(1, 2)]);
    /// assert_eq!(t[(3, 1)], m[(1, 3)]);
    ///
    /// assert_eq!(t[(0, 2)], m[(2, 0)]);
    /// assert_eq!(t[(1, 2)], m[(2, 1)]);
    /// assert_eq!(t[(2, 2)], m[(2, 2)]);
    /// assert_eq!(t[(3, 2)], m[(2, 3)]);
    /// ```
    #[inline]
    pub fn transpose(self) -> MatRef<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe {
            MatRef::from_raw_parts(
                ptr,
                self.ncols(),
                self.nrows(),
                self.col_stride(),
                self.row_stride(),
            )
        }
    }

    /// Returns a matrix whose rows are the the rows of the input matrix in reverse order.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![
    ///     [0.0, 3.0, 6.0, 9.0],
    ///     [1.0, 4.0, 7.0, 10.0],
    ///     [2.0, 5.0, 8.0, 11.0],
    /// ];
    /// let m = m.as_ref();
    ///
    /// let i = m.reverse_rows();
    ///
    /// assert_eq!(i.nrows(), m.nrows());
    /// assert_eq!(i.ncols(), m.ncols());
    ///
    /// assert_eq!(i[(0, 0)], m[(2, 0)]);
    /// assert_eq!(i[(1, 0)], m[(1, 0)]);
    /// assert_eq!(i[(2, 0)], m[(0, 0)]);
    ///
    /// assert_eq!(i[(0, 1)], m[(2, 1)]);
    /// assert_eq!(i[(1, 1)], m[(1, 1)]);
    /// assert_eq!(i[(2, 1)], m[(0, 1)]);
    ///
    /// assert_eq!(i[(0, 2)], m[(2, 2)]);
    /// assert_eq!(i[(1, 2)], m[(1, 2)]);
    /// assert_eq!(i[(2, 2)], m[(0, 2)]);
    ///
    /// assert_eq!(i[(0, 3)], m[(2, 3)]);
    /// assert_eq!(i[(1, 3)], m[(1, 3)]);
    /// assert_eq!(i[(2, 3)], m[(0, 3)]);
    /// ```
    #[inline]
    pub fn reverse_rows(self) -> Self {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = -self.row_stride();
        let col_stride = self.col_stride();

        let ptr = self.ptr_at(if nrows == 0 { 0 } else { nrows - 1 }, 0);
        unsafe { Self::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }

    /// Returns a matrix whose columns are the the columns of the input matrix in reverse order.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![
    ///     [0.0, 3.0, 6.0, 9.0],
    ///     [1.0, 4.0, 7.0, 10.0],
    ///     [2.0, 5.0, 8.0, 11.0],
    /// ];
    /// let m = m.as_ref();
    ///
    /// let i = m.reverse_cols();
    ///
    /// assert_eq!(i.nrows(), m.nrows());
    /// assert_eq!(i.ncols(), m.ncols());
    ///
    /// assert_eq!(i[(0, 0)], m[(0, 3)]);
    /// assert_eq!(i[(1, 0)], m[(1, 3)]);
    /// assert_eq!(i[(2, 0)], m[(2, 3)]);
    ///
    /// assert_eq!(i[(0, 1)], m[(0, 2)]);
    /// assert_eq!(i[(1, 1)], m[(1, 2)]);
    /// assert_eq!(i[(2, 1)], m[(2, 2)]);
    ///
    /// assert_eq!(i[(0, 2)], m[(0, 1)]);
    /// assert_eq!(i[(1, 2)], m[(1, 1)]);
    /// assert_eq!(i[(2, 2)], m[(2, 1)]);
    ///
    /// assert_eq!(i[(0, 3)], m[(0, 0)]);
    /// assert_eq!(i[(1, 3)], m[(1, 0)]);
    /// assert_eq!(i[(2, 3)], m[(2, 0)]);
    /// ```
    #[inline]
    pub fn reverse_cols(self) -> Self {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride();
        let col_stride = -self.col_stride();
        let ptr = self.ptr_at(0, if ncols == 0 { 0 } else { ncols - 1 });
        unsafe { Self::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }

    /// Returns a matrix whose rows and columns are the the rows and columns of the input matrix in
    /// reverse order.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![
    ///     [0.0, 3.0, 6.0, 9.0],
    ///     [1.0, 4.0, 7.0, 10.0],
    ///     [2.0, 5.0, 8.0, 11.0],
    /// ];
    /// let m = m.as_ref();
    ///
    /// let i = m.reverse_rows_and_cols();
    ///
    /// assert_eq!(i.nrows(), m.nrows());
    /// assert_eq!(i.ncols(), m.ncols());
    ///
    /// assert_eq!(i[(0, 0)], m[(2, 3)]);
    /// assert_eq!(i[(1, 0)], m[(1, 3)]);
    /// assert_eq!(i[(2, 0)], m[(0, 3)]);
    ///
    /// assert_eq!(i[(0, 1)], m[(2, 2)]);
    /// assert_eq!(i[(1, 1)], m[(1, 2)]);
    /// assert_eq!(i[(2, 1)], m[(0, 2)]);
    ///
    /// assert_eq!(i[(0, 2)], m[(2, 1)]);
    /// assert_eq!(i[(1, 2)], m[(1, 1)]);
    /// assert_eq!(i[(2, 2)], m[(0, 1)]);
    ///
    /// assert_eq!(i[(0, 3)], m[(2, 0)]);
    /// assert_eq!(i[(1, 3)], m[(1, 0)]);
    /// assert_eq!(i[(2, 3)], m[(0, 0)]);
    /// ```
    #[inline]
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

    /// Returns the diagonal of the matrix, as a column vector.
    ///
    /// # Safety
    ///
    /// Requires that the matrix be square.
    ///
    /// Otherwise, the behavior is undefined.
    ///
    /// # Example
    ///
    /// See [`Self::diagonal`].
    #[track_caller]
    #[inline]
    pub unsafe fn diagonal_unchecked(self) -> ColRef<'a, T> {
        fancy_debug_assert!(self.nrows() == self.ncols());
        ColRef::from_raw_parts(
            self.base.ptr.as_ptr(),
            self.base.nrows,
            self.base.row_stride + self.base.col_stride,
        )
    }

    /// Returns the diagonal of the matrix, as a column vector.
    ///
    /// # Panics
    ///
    /// Requires that the matrix be square.
    ///
    /// Otherwise, it panics.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]];
    /// let m = m.as_ref();
    ///
    /// let d = m.diagonal();
    ///
    /// assert_eq!(d.nrows(), 3);
    /// assert_eq!(d.ncols(), 1);
    ///
    /// assert_eq!(d[0], 0.0);
    /// assert_eq!(d[1], 4.0);
    /// assert_eq!(d[2], 8.0);
    /// ```
    #[track_caller]
    #[inline]
    pub fn diagonal(self) -> ColRef<'a, T> {
        fancy_assert!(self.nrows() == self.ncols());
        unsafe { self.diagonal_unchecked() }
    }

    /// Returns an iterator over the rows of the matrix.
    #[inline]
    pub fn into_row_iter(self) -> RowIter<'a, T> {
        RowIter(self)
    }

    /// Returns an iterator over the columns of the matrix.
    #[inline]
    pub fn into_col_iter(self) -> ColIter<'a, T> {
        ColIter(self)
    }

    /// Returns a parallel iterator over row chunks of the matrix.
    #[inline]
    #[track_caller]
    pub fn into_par_row_chunks(
        self,
        chunk_count: usize,
    ) -> impl IndexedParallelIterator<Item = (usize, MatRef<'a, T>)>
    where
        T: Sync,
    {
        use rayon::prelude::*;
        fancy_assert!(chunk_count != 0);
        let chunk_size = self.nrows() / chunk_count;
        let rem = self.nrows() % chunk_count;
        let ncols = self.ncols();

        let idx_to_col_start = move |idx| {
            if idx < rem {
                idx * (chunk_size + 1)
            } else {
                rem + idx * chunk_size
            }
        };

        (0..chunk_count).into_par_iter().map(move |idx| unsafe {
            let col_start = idx_to_col_start(idx);
            (
                col_start,
                self.submatrix_unchecked(
                    col_start,
                    0,
                    idx_to_col_start(idx + 1) - col_start,
                    ncols,
                ),
            )
        })
    }

    /// Returns a parallel iterator over column chunks of the matrix.
    #[inline]
    #[track_caller]
    pub fn into_par_col_chunks(
        self,
        chunk_count: usize,
    ) -> impl IndexedParallelIterator<Item = (usize, MatRef<'a, T>)>
    where
        T: Sync,
    {
        use rayon::prelude::*;
        self.transpose()
            .into_par_row_chunks(chunk_count)
            .map(|(idx, mat)| (idx, mat.transpose()))
    }

    /// Returns a view over a submatrix of `self`, starting at position `(i, j)`
    /// with dimensions `(nrows, ncols)`.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `j <= self.ncols()`,
    /// - `nrows <= self.nrows() - i`,
    /// - `ncols <= self.ncols() - j`.
    ///
    /// Otherwise, the behavior is undefined.
    ///
    /// # Example
    ///
    /// See [`Self::submatrix`].
    #[track_caller]
    #[inline]
    pub unsafe fn submatrix_unchecked(
        self,
        i: usize,
        j: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        fancy_debug_assert!(i <= self.nrows());
        fancy_debug_assert!(j <= self.ncols());
        fancy_debug_assert!(nrows <= self.nrows() - i);
        fancy_debug_assert!(ncols <= self.ncols() - j);
        Self::from_raw_parts(
            self.rb().ptr_at(i, j),
            nrows,
            ncols,
            self.row_stride(),
            self.col_stride(),
        )
    }

    /// Returns a view over a submatrix of `self`, starting at position `(i, j)`
    /// with dimensions `(nrows, ncols)`.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `j <= self.ncols()`,
    /// - `nrows <= self.nrows() - i`,
    /// - `ncols <= self.ncols() - j`.
    ///
    /// Otherwise, it panics.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    ///
    /// let m = mat![
    ///     [0.0, 3.0, 6.0, 9.0],
    ///     [1.0, 4.0, 7.0, 10.0],
    ///     [2.0, 5.0, 8.0, 11.0],
    /// ];
    /// let m = m.as_ref();
    ///
    /// let sub = m.submatrix(1, 2, 2, 2);
    ///
    /// assert_eq!(sub.nrows(), 2);
    /// assert_eq!(sub.ncols(), 2);
    ///
    /// assert_eq!(sub[(0, 0)], 7.0);
    /// assert_eq!(sub[(1, 0)], 8.0);
    /// assert_eq!(sub[(0, 1)], 10.0);
    /// assert_eq!(sub[(1, 1)], 11.0);
    /// ```
    #[track_caller]
    #[inline]
    pub fn submatrix(self, i: usize, j: usize, nrows: usize, ncols: usize) -> Self {
        fancy_assert!(i <= self.nrows());
        fancy_assert!(j <= self.ncols());
        fancy_assert!(nrows <= self.nrows() - i);
        fancy_assert!(ncols <= self.ncols() - j);
        unsafe { self.submatrix_unchecked(i, j, nrows, ncols) }
    }

    /// Returns a thin wrapper that can be used to execute coefficientwise operations on matrices.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::mat;
    /// use reborrow::*;
    ///
    /// let a = mat![
    ///     [0.0, 3.0, 6.0, 9.0],
    ///     [1.0, 4.0, 7.0, 10.0],
    ///     [2.0, 5.0, 8.0, 11.0],
    /// ];
    /// let b = mat![
    ///     [12.0, 15.0, 18.0, 21.0],
    ///     [13.0, 16.0, 19.0, 22.0],
    ///     [14.0, 17.0, 20.0, 23.0],
    /// ];
    /// let mut c = mat![
    ///     [0.0, 0.0, 0.0, 0.0],
    ///     [0.0, 0.0, 0.0, 0.0],
    ///     [0.0, 0.0, 0.0, 0.0],
    /// ];
    ///
    /// let a = a.as_ref();
    /// let b = b.as_ref();
    /// let mut c = c.as_mut();
    ///
    /// c.rb_mut()
    ///     .cwise()
    ///     .zip(a)
    ///     .zip(b)
    ///     .for_each(|c, a, b| *c = *a + *b);
    ///
    /// assert_eq!(c[(0, 0)], a[(0, 0)] + b[(0, 0)]);
    /// assert_eq!(c[(1, 0)], a[(1, 0)] + b[(1, 0)]);
    /// assert_eq!(c[(2, 0)], a[(2, 0)] + b[(2, 0)]);
    ///
    /// assert_eq!(c[(0, 1)], a[(0, 1)] + b[(0, 1)]);
    /// assert_eq!(c[(1, 1)], a[(1, 1)] + b[(1, 1)]);
    /// assert_eq!(c[(2, 1)], a[(2, 1)] + b[(2, 1)]);
    ///
    /// assert_eq!(c[(0, 2)], a[(0, 2)] + b[(0, 2)]);
    /// assert_eq!(c[(1, 2)], a[(1, 2)] + b[(1, 2)]);
    /// assert_eq!(c[(2, 2)], a[(2, 2)] + b[(2, 2)]);
    ///
    /// assert_eq!(c[(0, 3)], a[(0, 3)] + b[(0, 3)]);
    /// assert_eq!(c[(1, 3)], a[(1, 3)] + b[(1, 3)]);
    /// assert_eq!(c[(2, 3)], a[(2, 3)] + b[(2, 3)]);
    /// ```
    #[inline]
    pub fn cwise(self) -> ZipMat<(Self,)> {
        ZipMat { tuple: (self,) }
    }

    #[doc(hidden)]
    #[inline]
    pub unsafe fn const_cast(self) -> MatMut<'a, T> {
        MatMut {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> MatMut<'a, T> {
    /// Returns a mutable matrix slice from the given arguments.  
    /// `ptr`: pointer to the first element of the matrix.  
    /// `nrows`: number of rows of the matrix.  
    /// `ncols`: number of columns of the matrix.  
    /// `row_stride`: offset between the first elements of two successive rows in the matrix.
    /// `col_stride`: offset between the first elements of two successive columns in the matrix.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `i < nrows` and `j < ncols`,  
    /// `ptr.offset(i as isize * row_stride + j as isize * col_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never read.
    /// Additionally, when `(i, j) != (0, 0)`, this pointer is never equal to `ptr` (no self
    /// aliasing).  
    /// The referenced memory must not be accessed by another pointer which was not derived from
    /// the return value, during the lifetime `'a`.
    #[inline]
    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            base: MatrixSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr),
                nrows,
                ncols,
                row_stride,
                col_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a mutable pointer to the first (top left) element of the matrix.
    #[inline]
    pub fn as_ptr(self) -> *mut T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.base.nrows
    }

    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.base.ncols
    }

    /// Returns the offset between the first elements of two successive rows in the matrix.
    #[inline]
    pub fn row_stride(&self) -> isize {
        self.base.row_stride
    }

    /// Returns the offset between the first elements of two successive columns in the matrix.
    #[inline]
    pub fn col_stride(&self) -> isize {
        self.base.col_stride
    }

    /// Returns a mutable pointer to the element at position (i, j) in the matrix.
    #[inline]
    pub fn ptr_at(self, i: usize, j: usize) -> *mut T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(i as isize * self.row_stride())
            .wrapping_offset(j as isize * self.col_stride())
    }

    /// Returns a mutable pointer to the element at position (i, j) in the matrix, assuming it falls
    /// within its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i < self.nrows()`,
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn ptr_in_bounds_at_unchecked(self, i: usize, j: usize) -> *mut T {
        fancy_debug_assert!(i < self.nrows());
        fancy_debug_assert!(j < self.ncols());
        self.base
            .ptr
            .as_ptr()
            .offset(i as isize * self.row_stride())
            .offset(j as isize * self.col_stride())
    }

    /// Returns a mutable pointer to the element at position (i, j) in the matrix, while asserting
    /// that it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i < self.nrows()`,
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn ptr_in_bounds_at(self, i: usize, j: usize) -> *mut T {
        fancy_assert!(i < self.nrows());
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(i, j) }
    }

    /// Splits the matrix into four corner parts in the following order: top left, top right,
    /// bottom left, bottom right.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `j <= self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn split_at_unchecked(self, i: usize, j: usize) -> (Self, Self, Self, Self) {
        fancy_debug_assert!(i <= self.nrows());
        fancy_debug_assert!(j <= self.ncols());
        let ptr = self.base.ptr.as_ptr();
        let cs = self.col_stride();
        let rs = self.row_stride();
        (
            Self::from_raw_parts(ptr, i, j, rs, cs),
            Self::from_raw_parts(
                ptr.wrapping_offset(j as isize * cs),
                i,
                self.ncols() - j,
                rs,
                cs,
            ),
            Self::from_raw_parts(
                ptr.wrapping_offset(i as isize * rs),
                self.nrows() - i,
                j,
                rs,
                cs,
            ),
            Self::from_raw_parts(
                ptr.wrapping_offset(i as isize * rs)
                    .wrapping_offset(j as isize * cs),
                self.nrows() - i,
                self.ncols() - j,
                rs,
                cs,
            ),
        )
    }

    /// Splits the matrix into four corner parts in the following order: top left, top right,
    /// bottom left, bottom right.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `j <= self.ncols()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn split_at(self, i: usize, j: usize) -> (Self, Self, Self, Self) {
        fancy_assert!(i <= self.nrows());
        fancy_assert!(j <= self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(i, j) }
    }

    /// Splits the matrix horizontally into two parts in the following order: top, bottom.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn split_at_row_unchecked(self, i: usize) -> (Self, Self) {
        let (_, top, _, bottom) = self.split_at_unchecked(i, 0);
        (top, bottom)
    }

    /// Splits the matrix horizontally into two parts in the following order: top, bottom.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn split_at_row(self, i: usize) -> (Self, Self) {
        let (_, top, _, bottom) = self.split_at(i, 0);
        (top, bottom)
    }

    /// Splits the matrix vertically into two parts in the following order: left, right.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j <= self.nrows()`,
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn split_at_col_unchecked(self, j: usize) -> (Self, Self) {
        let (_, _, left, right) = self.split_at_unchecked(0, j);
        (left, right)
    }

    /// Splits the matrix vertically into two parts in the following order: left, right.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j <= self.nrows()`,
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn split_at_col(self, j: usize) -> (Self, Self) {
        let (_, _, left, right) = self.split_at(0, j);
        (left, right)
    }

    /// Returns a mutable reference to the element at position (i, j), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i < self.nrows()`,
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn get_unchecked(self, i: usize, j: usize) -> &'a mut T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &mut *self.ptr_in_bounds_at_unchecked(i, j)
    }

    /// Returns a mutable reference to the element at position (i, j), or panics if the indices are
    /// out of bounds.
    #[track_caller]
    #[inline]
    pub fn get(self, i: usize, j: usize) -> &'a mut T {
        fancy_assert!(i < self.nrows());
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked.
        unsafe { self.get_unchecked(i, j) }
    }

    /// Returns the `i`-th row of the matrix, with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i < self.nrows()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn row_unchecked(self, i: usize) -> RowMut<'a, T> {
        fancy_debug_assert!(i < self.nrows());
        let ncols = self.ncols();
        let cs = self.col_stride();
        RowMut::from_raw_parts(self.ptr_at(i, 0), ncols, cs)
    }

    /// Returns the `i`-th row of the matrix.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i < self.nrows()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn row(self, i: usize) -> RowMut<'a, T> {
        fancy_assert!(i < self.nrows());
        // SAFETY: bounds have been checked.
        unsafe { self.row_unchecked(i) }
    }

    /// Returns the `j`-th column of the matrix, with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn col_unchecked(self, j: usize) -> ColMut<'a, T> {
        fancy_debug_assert!(j < self.ncols());
        let nrows = self.nrows();
        let rs = self.row_stride();
        ColMut::from_raw_parts(self.ptr_at(0, j), nrows, rs)
    }

    /// Returns the `j`-th column of the matrix.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn col(self, j: usize) -> ColMut<'a, T> {
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked.
        unsafe { self.col_unchecked(j) }
    }

    /// Returns the transpose of `self`.
    #[inline]
    pub fn transpose(self) -> MatMut<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe {
            MatMut::from_raw_parts(
                ptr,
                self.ncols(),
                self.nrows(),
                self.col_stride(),
                self.row_stride(),
            )
        }
    }

    /// Returns a matrix whose rows are the the rows of the input matrix in reverse order.
    #[inline]
    pub fn reverse_rows(self) -> Self {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = -self.row_stride();
        let col_stride = self.col_stride();

        let ptr = self.ptr_at(if nrows == 0 { 0 } else { nrows - 1 }, 0);
        unsafe { Self::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }

    /// Returns a matrix whose columns are the the columns of the input matrix in reverse order.
    #[inline]
    pub fn reverse_cols(self) -> Self {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride();
        let col_stride = -self.col_stride();
        let ptr = self.ptr_at(0, if ncols == 0 { 0 } else { ncols - 1 });
        unsafe { Self::from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }

    /// Returns a matrix whose rows and columns are the the rows and columns of the input matrix in
    /// reverse order.
    #[inline]
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

    /// Returns the diagonal of the matrix, as a column vector.
    ///
    /// # Safety
    ///
    /// Requires that the matrix be square.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn diagonal_unchecked(self) -> ColMut<'a, T> {
        fancy_debug_assert!(self.nrows() == self.ncols());
        ColMut::from_raw_parts(
            self.base.ptr.as_ptr(),
            self.base.nrows,
            self.base.row_stride + self.base.col_stride,
        )
    }

    /// Returns the diagonal of the matrix, as a column vector.
    ///
    /// # Panics
    ///
    /// Requires that the matrix be square.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn diagonal(self) -> ColMut<'a, T> {
        fancy_assert!(self.nrows() == self.ncols());
        unsafe { self.diagonal_unchecked() }
    }

    /// Returns an iterator over the rows of the matrix.
    #[inline]
    pub fn into_row_iter(self) -> RowIterMut<'a, T> {
        RowIterMut(self)
    }

    /// Returns an iterator over the columns of the matrix.
    #[inline]
    pub fn into_col_iter(self) -> ColIterMut<'a, T> {
        ColIterMut(self)
    }

    /// Returns a parallel iterator over the rows of the matrix.
    #[inline]
    pub fn into_par_row_chunks(
        self,
        chunk_count: usize,
    ) -> impl IndexedParallelIterator<Item = (usize, MatMut<'a, T>)>
    where
        T: Sync + Send,
    {
        use rayon::prelude::*;
        self.into_const()
            .into_par_row_chunks(chunk_count)
            .map(|(idx, chunk)| (idx, unsafe { chunk.const_cast() }))
    }

    /// Returns a parallel iterator over the rows of the matrix.
    #[inline]
    pub fn into_par_col_chunks(
        self,
        chunk_count: usize,
    ) -> impl IndexedParallelIterator<Item = (usize, MatMut<'a, T>)>
    where
        T: Sync + Send,
    {
        use rayon::prelude::*;
        self.into_const()
            .into_par_col_chunks(chunk_count)
            .map(|(idx, chunk)| (idx, unsafe { chunk.const_cast() }))
    }

    /// Returns a view over a submatrix of `self`, starting at position `(i, j)`
    /// with dimensions `(nrows, ncols)`.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `j <= self.ncols()`,
    /// - `nrows <= self.nrows() - i`,
    /// - `ncols <= self.ncols() - j`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn submatrix_unchecked(
        self,
        i: usize,
        j: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        fancy_debug_assert!(i <= self.nrows());
        fancy_debug_assert!(j <= self.ncols());
        fancy_debug_assert!(nrows <= self.nrows() - i);
        fancy_debug_assert!(ncols <= self.ncols() - j);

        let mut s = self;
        Self::from_raw_parts(
            s.rb_mut().ptr_at(i, j),
            nrows,
            ncols,
            s.row_stride(),
            s.col_stride(),
        )
    }

    /// Returns a view over a submatrix of `self`, starting at position `(i, j)`
    /// with dimensions `(nrows, ncols)`.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `j <= self.ncols()`,
    /// - `nrows <= self.nrows() - i`,
    /// - `ncols <= self.ncols() - j`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn submatrix(self, i: usize, j: usize, nrows: usize, ncols: usize) -> Self {
        fancy_assert!(i <= self.nrows());
        fancy_assert!(j <= self.ncols());
        fancy_assert!(nrows <= self.nrows() - i);
        fancy_assert!(ncols <= self.ncols() - j);
        unsafe { self.submatrix_unchecked(i, j, nrows, ncols) }
    }

    /// Returns a thin wrapper that can be used to execute coefficientwise operations on matrices.
    #[inline]
    pub fn cwise(self) -> ZipMat<(Self,)> {
        ZipMat { tuple: (self,) }
    }
}

impl<'a, T> RowRef<'a, T> {
    /// Returns a row vector slice from the given arguments.  
    /// `ptr`: pointer to the first element of the row vector.  
    /// `ncols`: number of columns of the row vector.  
    /// `col_stride`: offset between the first elements of two successive columns in the row vector.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `j < ncols`,  
    /// `ptr.offset(j as isize * col_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never read.  
    /// The referenced memory must not be mutated during the lifetime `'a`.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *const T, ncols: usize, col_stride: isize) -> Self {
        Self {
            base: VecSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr as *mut T),
                len: ncols,
                stride: col_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a pointer to the first (left) element of the row vector.
    #[inline]
    pub fn as_ptr(self) -> *const T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the row vector. Always returns `1`.
    #[inline]
    pub fn nrows(&self) -> usize {
        1
    }

    /// Returns the number of columns of the row vector.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.base.len
    }

    /// Returns the offset between the first elements of two successive columns in the row vector.
    #[inline]
    pub fn col_stride(&self) -> isize {
        self.base.stride
    }

    /// Returns a pointer to the element at position (0, j) in the row vector.
    #[inline]
    pub fn ptr_at(self, j: usize) -> *const T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(j as isize * self.col_stride())
    }

    /// Returns a pointer to the element at position (0, j) in the row vector, assuming it falls
    /// within its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn ptr_in_bounds_at_unchecked(self, j: usize) -> *const T {
        fancy_debug_assert!(j < self.ncols());
        self.base
            .ptr
            .as_ptr()
            .offset(j as isize * self.col_stride())
    }

    /// Returns a pointer to the element at position (0, j) in the row vector, while asserting that
    /// it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn ptr_in_bounds_at(self, j: usize) -> *const T {
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(j) }
    }

    /// Splits the row vector into two parts in the following order: left, right.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `j <= self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn split_at_unchecked(self, j: usize) -> (Self, Self) {
        fancy_debug_assert!(j <= self.ncols());
        let ptr = self.base.ptr.as_ptr();
        let cs = self.col_stride();
        (
            Self::from_raw_parts(ptr, j, cs),
            Self::from_raw_parts(ptr.wrapping_offset(j as isize * cs), self.ncols() - j, cs),
        )
    }

    /// Splits the row vector into two parts in the following order: left, right.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j <= self.ncols()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn split_at(self, j: usize) -> (Self, Self) {
        fancy_assert!(j <= self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(j) }
    }

    /// Returns a reference to the element at position (0, j), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires `j < self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn get_unchecked(self, j: usize) -> &'a T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &*self.ptr_in_bounds_at_unchecked(j)
    }

    /// Returns a reference to the element at position (0, j), or panics if the index is out of
    /// bounds.
    #[track_caller]
    #[inline]
    pub fn get(self, j: usize) -> &'a T {
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked.
        unsafe { self.get_unchecked(j) }
    }

    /// Returns an equivalent matrix view over the same data.
    #[inline]
    pub fn as_2d(self) -> MatRef<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { MatRef::from_raw_parts(ptr, self.nrows(), self.ncols(), 0, self.col_stride()) }
    }

    /// Returns the transpose of `self`.
    #[inline]
    pub fn transpose(self) -> ColRef<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { ColRef::from_raw_parts(ptr, self.ncols(), self.col_stride()) }
    }

    /// Returns a thin wrapper that can be used to execute coefficientwise operations on matrices.
    #[inline]
    pub fn cwise(self) -> ZipRow<(Self,)> {
        ZipRow { tuple: (self,) }
    }

    /// Returns a view over subcolumns of `self`, starting at position `j`
    /// with a column count of `ncols`.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `j <= self.ncols()`,
    /// - `ncols <= self.ncols() - j`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn subcols_unchecked(self, j: usize, ncols: usize) -> Self {
        fancy_debug_assert!(j <= self.ncols());
        fancy_debug_assert!(ncols <= self.ncols() - j);

        let mut s = self;
        Self::from_raw_parts(s.rb_mut().ptr_at(j), ncols, s.col_stride())
    }

    /// Returns a view over subcolumnss of `self`, starting at position `j`
    /// with a col count of `ncols`.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j <= self.ncols()`,
    /// - `ncols <= self.ncols() - j`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn subcols(self, j: usize, ncols: usize) -> Self {
        fancy_assert!(j <= self.ncols());
        fancy_assert!(ncols <= self.ncols() - j);

        unsafe { self.subcols_unchecked(j, ncols) }
    }

    #[doc(hidden)]
    #[inline]
    pub unsafe fn const_cast(self) -> RowMut<'a, T> {
        RowMut {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> RowMut<'a, T> {
    /// Returns a mutable row vector slice from the given arguments.  
    /// `ptr`: pointer to the first element of the row vector.  
    /// `ncols`: number of columns of the row vector.  
    /// `col_stride`: offset between the first elements of two successive columns in the row vector.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `j < ncols`,  
    /// `ptr.offset(j as isize * col_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never read.  
    /// Additionally, when `j != 0`, this pointer is never equal to `ptr` (no self aliasing).  
    /// The referenced memory must not be accessed by another pointer which was not derived from
    /// the return value, during the lifetime `'a`.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, ncols: usize, col_stride: isize) -> Self {
        Self {
            base: VecSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr),
                len: ncols,
                stride: col_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a mutable pointer to the first (left) element of the row vector.
    #[inline]
    pub fn as_ptr(self) -> *mut T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the row vector. Always returns `1`.
    #[inline]
    pub fn nrows(&self) -> usize {
        1
    }

    /// Returns the number of columns of the row vector.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.base.len
    }

    /// Returns the offset between the first elements of two successive columns in the row vector.
    #[inline]
    pub fn col_stride(&self) -> isize {
        self.base.stride
    }

    /// Returns a mutable pointer to the element at position (0, j) in the row vector.
    #[inline]
    pub fn ptr_at(self, j: usize) -> *mut T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(j as isize * self.col_stride())
    }

    /// Returns a mutable pointer to the element at position (0, j) in the row vector, assuming it
    /// falls within its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn ptr_in_bounds_at_unchecked(self, j: usize) -> *mut T {
        fancy_debug_assert!(j < self.ncols());
        self.base
            .ptr
            .as_ptr()
            .offset(j as isize * self.col_stride())
    }

    /// Returns a mutable pointer to the element at position (0, j) in the row vector, while
    /// asserting that it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j < self.ncols()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn ptr_in_bounds_at(self, j: usize) -> *mut T {
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(j) }
    }

    /// Splits the row vector into two parts in the following order: left, right.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `j <= self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn split_at_unchecked(self, j: usize) -> (Self, Self) {
        fancy_debug_assert!(j <= self.ncols());
        let ptr = self.base.ptr.as_ptr();
        let cs = self.col_stride();
        (
            Self::from_raw_parts(ptr, j, cs),
            Self::from_raw_parts(ptr.wrapping_offset(j as isize * cs), self.ncols() - j, cs),
        )
    }

    /// Splits the row vector into two parts in the following order: left, right.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j <= self.ncols()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn split_at(self, j: usize) -> (Self, Self) {
        fancy_assert!(j <= self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(j) }
    }

    /// Returns a mutable reference to the element at position (0, j), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires `j < self.ncols()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn get_unchecked(self, j: usize) -> &'a mut T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &mut *self.ptr_in_bounds_at_unchecked(j)
    }

    /// Returns a mutable reference to the element at position (0, j), or panics if the index is
    /// out of bounds.
    #[track_caller]
    #[inline]
    pub fn get(self, j: usize) -> &'a mut T {
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked.
        unsafe { self.get_unchecked(j) }
    }

    /// Returns an equivalent matrix view over the same data.
    #[inline]
    pub fn as_2d(self) -> MatMut<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { MatMut::from_raw_parts(ptr, self.nrows(), self.ncols(), 0, self.col_stride()) }
    }

    /// Returns the transpose of `self`.
    #[inline]
    pub fn transpose(self) -> ColMut<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { ColMut::from_raw_parts(ptr, self.ncols(), self.col_stride()) }
    }

    /// Returns a view over subcolumns of `self`, starting at position `j`
    /// with a column count of `ncols`.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `j <= self.ncols()`,
    /// - `ncols <= self.ncols() - j`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn subcols_unchecked(self, j: usize, ncols: usize) -> Self {
        fancy_debug_assert!(j <= self.ncols());
        fancy_debug_assert!(ncols <= self.ncols() - j);

        let mut s = self;
        Self::from_raw_parts(s.rb_mut().ptr_at(j), ncols, s.col_stride())
    }

    /// Returns a view over subcolumnss of `self`, starting at position `j`
    /// with a col count of `ncols`.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `j <= self.ncols()`,
    /// - `ncols <= self.ncols() - j`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn subcols(self, j: usize, ncols: usize) -> Self {
        fancy_assert!(j <= self.ncols());
        fancy_assert!(ncols <= self.ncols() - j);

        unsafe { self.subcols_unchecked(j, ncols) }
    }

    /// Returns a thin wrapper that can be used to execute coefficientwise operations on matrices.
    #[inline]
    pub fn cwise(self) -> ZipRow<(Self,)> {
        ZipRow { tuple: (self,) }
    }
}

impl<'a, T> ColRef<'a, T> {
    /// Returns a column vector slice from the given arguments.  
    /// `ptr`: pointer to the first element of the column vector.  
    /// `ncols`: number of columns of the column vector.  
    /// `col_stride`: offset between the first elements of two successive columns in the column
    /// vector.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `i < nrows`,  
    /// `ptr.offset(i as isize * row_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never read.  
    /// The referenced memory must not be mutated during the lifetime `'a`.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *const T, nrows: usize, row_stride: isize) -> Self {
        Self {
            base: VecSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr as *mut T),
                len: nrows,
                stride: row_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a pointer to the first (top) element of the column vector.
    #[inline]
    pub fn as_ptr(self) -> *const T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the column vector.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.base.len
    }

    /// Returns the number of columns of the column vector. Always returns `1`.
    #[inline]
    pub fn ncols(&self) -> usize {
        1
    }

    /// Returns the offset between the first elements of two successive rows in the column vector.
    #[inline]
    pub fn row_stride(&self) -> isize {
        self.base.stride
    }

    /// Returns a pointer to the element at position (i, 0) in the column vector.
    #[inline]
    pub fn ptr_at(self, i: usize) -> *const T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(i as isize * self.row_stride())
    }

    /// Returns a pointer to the element at position (i, 0) in the column vector, assuming it falls
    /// within its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i < self.nrows()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn ptr_in_bounds_at_unchecked(self, i: usize) -> *const T {
        fancy_debug_assert!(i < self.nrows());
        self.base
            .ptr
            .as_ptr()
            .offset(i as isize * self.row_stride())
    }

    /// Returns a pointer to the element at position (i, 0) in the column vector, while asserting
    /// that it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i < self.nrows()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn ptr_in_bounds_at(self, i: usize) -> *const T {
        fancy_assert!(i < self.nrows());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(i) }
    }

    /// Splits the column vector into two parts in the following order: top, bottom.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i <= self.nrows()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn split_at_unchecked(self, i: usize) -> (Self, Self) {
        fancy_debug_assert!(i <= self.nrows());
        let ptr = self.base.ptr.as_ptr();
        let rs = self.row_stride();
        (
            Self::from_raw_parts(ptr, i, rs),
            Self::from_raw_parts(ptr.wrapping_offset(i as isize * rs), self.nrows() - i, rs),
        )
    }

    /// Splits the column vector into two parts in the following order: top, bottom.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn split_at(self, i: usize) -> (Self, Self) {
        fancy_assert!(i <= self.nrows());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(i) }
    }

    /// Returns a reference to the element at position (i, 0), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires `i < self.nrows()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn get_unchecked(self, i: usize) -> &'a T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &*self.ptr_in_bounds_at_unchecked(i)
    }

    /// Returns a reference to the element at position (i, 0), or panics if the index is out of
    /// bounds.
    #[track_caller]
    #[inline]
    pub fn get(self, i: usize) -> &'a T {
        fancy_assert!(i < self.nrows());
        // SAFETY: bounds have been checked.
        unsafe { self.get_unchecked(i) }
    }

    /// Returns an equivalent matrix view over the same data.
    #[inline]
    pub fn as_2d(self) -> MatRef<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { MatRef::from_raw_parts(ptr, self.nrows(), self.ncols(), self.row_stride(), 0) }
    }

    /// Returns the transpose of `self`.
    #[inline]
    pub fn transpose(self) -> RowRef<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { RowRef::from_raw_parts(ptr, self.nrows(), self.row_stride()) }
    }

    /// Returns a view over subrows of `self`, starting at position `i`
    /// with a row count of `nrows`.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `nrows <= self.nrows() - i`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn subrows_unchecked(self, i: usize, nrows: usize) -> Self {
        fancy_debug_assert!(i <= self.nrows());
        fancy_debug_assert!(nrows <= self.nrows() - i);

        let mut s = self;
        Self::from_raw_parts(s.rb_mut().ptr_at(i), nrows, s.row_stride())
    }

    /// Returns a view over subrows of `self`, starting at position `i`
    /// with a row count of `nrows`.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `nrows <= self.nrows() - i`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn subrows(self, i: usize, nrows: usize) -> Self {
        fancy_assert!(i <= self.nrows());
        fancy_assert!(nrows <= self.nrows() - i);

        unsafe { self.subrows_unchecked(i, nrows) }
    }

    /// Returns a thin wrapper that can be used to execute coefficientwise operations on matrices.
    #[inline]
    pub fn cwise(self) -> ZipCol<(Self,)> {
        ZipCol { tuple: (self,) }
    }

    #[doc(hidden)]
    #[inline]
    pub unsafe fn const_cast(self) -> ColMut<'a, T> {
        ColMut {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> ColMut<'a, T> {
    /// Returns a mutable column vector slice from the given arguments.  
    /// `ptr`: pointer to the first element of the column vector.  
    /// `ncols`: number of columns of the column vector.  
    /// `col_stride`: offset between the first elements of two successive columns in the column
    /// vector.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `i < nrows`,  
    /// `ptr.offset(i as isize * row_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never read.  
    /// Additionally, when `i != 0`, this pointer is never equal to `ptr` (no self aliasing).  
    /// The referenced memory must not be mutated during the lifetime `'a`.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, nrows: usize, row_stride: isize) -> Self {
        Self {
            base: VecSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr),
                len: nrows,
                stride: row_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a mutable pointer to the first (top) element of the column vector.
    #[inline]
    pub fn as_ptr(self) -> *mut T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the column vector.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.base.len
    }

    /// Returns the number of columns of the column vector. Always returns `1`.
    #[inline]
    pub fn ncols(&self) -> usize {
        1
    }

    /// Returns the offset between the first elements of two successive rows in the column vector.
    #[inline]
    pub fn row_stride(&self) -> isize {
        self.base.stride
    }

    /// Returns a mutable pointer to the element at position (i, 0) in the column vector.
    #[inline]
    pub fn ptr_at(self, i: usize) -> *mut T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(i as isize * self.row_stride())
    }

    /// Returns a mutable pointer to the element at position (i, 0) in the column vector,
    /// assuming it falls within its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i < self.nrows()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn ptr_in_bounds_at_unchecked(self, i: usize) -> *mut T {
        fancy_debug_assert!(i < self.nrows());
        self.base
            .ptr
            .as_ptr()
            .offset(i as isize * self.row_stride())
    }

    /// Returns a mutable pointer to the element at position (i, 0) in the column vector,
    /// while asserting that it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i < self.nrows()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn ptr_in_bounds_at(self, i: usize) -> *mut T {
        fancy_assert!(i < self.nrows());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(i) }
    }

    /// Splits the column vector into two parts in the following order: top, bottom.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i <= self.nrows()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn split_at_unchecked(self, i: usize) -> (Self, Self) {
        fancy_debug_assert!(i <= self.nrows());
        let ptr = self.base.ptr.as_ptr();
        let rs = self.row_stride();
        (
            Self::from_raw_parts(ptr, i, rs),
            Self::from_raw_parts(ptr.wrapping_offset(i as isize * rs), self.nrows() - i, rs),
        )
    }

    /// Splits the column vector into two parts in the following order: top, bottom.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn split_at(self, i: usize) -> (Self, Self) {
        fancy_assert!(i <= self.nrows());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(i) }
    }

    /// Returns a mutable reference to the element at position (i, 0), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires `i < self.nrows()`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn get_unchecked(self, i: usize) -> &'a mut T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &mut *self.ptr_in_bounds_at_unchecked(i)
    }

    /// Returns a mutable reference to the element at position (i, 0), or panics if the index is
    /// out of bounds.
    #[track_caller]
    #[inline]
    pub fn get(self, i: usize) -> &'a mut T {
        fancy_assert!(i < self.nrows());
        // SAFETY: bounds have been checked.
        unsafe { self.get_unchecked(i) }
    }

    /// Returns an equivalent matrix view over the same data.
    #[inline]
    pub fn as_2d(self) -> MatMut<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { MatMut::from_raw_parts(ptr, self.nrows(), self.ncols(), self.row_stride(), 0) }
    }

    /// Returns the transpose of `self`.
    #[inline]
    pub fn transpose(self) -> RowMut<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { RowMut::from_raw_parts(ptr, self.nrows(), self.row_stride()) }
    }

    /// Returns a view over subrows of `self`, starting at position `i`
    /// with a row count of `nrows`.
    ///
    /// # Safety
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `nrows <= self.nrows() - i`.
    ///
    /// Otherwise, the behavior is undefined.
    #[track_caller]
    #[inline]
    pub unsafe fn subrows_unchecked(self, i: usize, nrows: usize) -> Self {
        fancy_debug_assert!(i <= self.nrows());
        fancy_debug_assert!(nrows <= self.nrows() - i);

        let mut s = self;
        Self::from_raw_parts(s.rb_mut().ptr_at(i), nrows, s.row_stride())
    }

    /// Returns a view over subrows of `self`, starting at position `i`
    /// with a row count of `nrows`.
    ///
    /// # Panics
    ///
    /// Requires that
    /// - `i <= self.nrows()`,
    /// - `nrows <= self.nrows() - i`.
    ///
    /// Otherwise, it panics.
    #[track_caller]
    #[inline]
    pub fn subrows(self, i: usize, nrows: usize) -> Self {
        fancy_assert!(i <= self.nrows());
        fancy_assert!(nrows <= self.nrows() - i);

        unsafe { self.subrows_unchecked(i, nrows) }
    }

    /// Returns a thin wrapper that can be used to execute coefficientwise operations on matrices.
    #[inline]
    pub fn cwise(self) -> ZipCol<(Self,)> {
        ZipCol { tuple: (self,) }
    }
}

impl<'a, T> MatRef<'a, Complex<T>> {
    #[inline]
    pub fn into_real_imag(self) -> (MatRef<'a, T>, MatRef<'a, T>) {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let rs = self.row_stride() * 2;
        let cs = self.col_stride() * 2;
        let ptr_re = self.as_ptr() as *const T;
        let ptr_im = ptr_re.wrapping_add(1);

        unsafe {
            (
                MatRef::from_raw_parts(ptr_re, nrows, ncols, rs, cs),
                MatRef::from_raw_parts(ptr_im, nrows, ncols, rs, cs),
            )
        }
    }
}
impl<'a, T> MatMut<'a, Complex<T>> {
    #[inline]
    pub fn into_real_imag(self) -> (MatMut<'a, T>, MatMut<'a, T>) {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let rs = self.row_stride() * 2;
        let cs = self.col_stride() * 2;
        let ptr_re = self.as_ptr() as *mut T;
        let ptr_im = ptr_re.wrapping_add(1);

        unsafe {
            (
                MatMut::from_raw_parts(ptr_re, nrows, ncols, rs, cs),
                MatMut::from_raw_parts(ptr_im, nrows, ncols, rs, cs),
            )
        }
    }
}

impl<'a, T> ColRef<'a, Complex<T>> {
    #[inline]
    pub fn into_real_imag(self) -> (ColRef<'a, T>, ColRef<'a, T>) {
        let nrows = self.nrows();
        let rs = self.row_stride() * 2;
        let ptr_re = self.as_ptr() as *const T;
        let ptr_im = ptr_re.wrapping_add(1);

        unsafe {
            (
                ColRef::from_raw_parts(ptr_re, nrows, rs),
                ColRef::from_raw_parts(ptr_im, nrows, rs),
            )
        }
    }
}
impl<'a, T> ColMut<'a, Complex<T>> {
    #[inline]
    pub fn into_real_imag(self) -> (ColMut<'a, T>, ColMut<'a, T>) {
        let nrows = self.nrows();
        let rs = self.row_stride() * 2;
        let ptr_re = self.as_ptr() as *mut T;
        let ptr_im = ptr_re.wrapping_add(1);

        unsafe {
            (
                ColMut::from_raw_parts(ptr_re, nrows, rs),
                ColMut::from_raw_parts(ptr_im, nrows, rs),
            )
        }
    }
}

impl<'a, T> RowRef<'a, Complex<T>> {
    #[inline]
    pub fn into_real_imag(self) -> (RowRef<'a, T>, RowRef<'a, T>) {
        let ncols = self.ncols();
        let cs = self.col_stride() * 2;
        let ptr_re = self.as_ptr() as *const T;
        let ptr_im = ptr_re.wrapping_add(1);

        unsafe {
            (
                RowRef::from_raw_parts(ptr_re, ncols, cs),
                RowRef::from_raw_parts(ptr_im, ncols, cs),
            )
        }
    }
}
impl<'a, T> RowMut<'a, Complex<T>> {
    #[inline]
    pub fn into_real_imag(self) -> (RowMut<'a, T>, RowMut<'a, T>) {
        let ncols = self.ncols();
        let cs = self.col_stride() * 2;
        let ptr_re = self.as_ptr() as *mut T;
        let ptr_im = ptr_re.wrapping_add(1);

        unsafe {
            (
                RowMut::from_raw_parts(ptr_re, ncols, cs),
                RowMut::from_raw_parts(ptr_im, ncols, cs),
            )
        }
    }
}

impl<'a, T> Index<(usize, usize)> for MatRef<'a, T> {
    type Output = T;

    #[track_caller]
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        self.get(i, j)
    }
}
impl<'a, T> Index<(usize, usize)> for MatMut<'a, T> {
    type Output = T;

    #[track_caller]
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        self.rb().get(i, j)
    }
}
impl<'a, T> IndexMut<(usize, usize)> for MatMut<'a, T> {
    #[track_caller]
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        self.rb_mut().get(i, j)
    }
}

impl<'a, T> Index<usize> for RowRef<'a, T> {
    type Output = T;

    #[track_caller]
    #[inline]
    fn index(&self, j: usize) -> &Self::Output {
        self.get(j)
    }
}
impl<'a, T> Index<usize> for RowMut<'a, T> {
    type Output = T;

    #[track_caller]
    #[inline]
    fn index(&self, j: usize) -> &Self::Output {
        self.rb().get(j)
    }
}
impl<'a, T> IndexMut<usize> for RowMut<'a, T> {
    #[track_caller]
    #[inline]
    fn index_mut(&mut self, j: usize) -> &mut Self::Output {
        self.rb_mut().get(j)
    }
}

impl<'a, T> Index<usize> for ColRef<'a, T> {
    type Output = T;

    #[track_caller]
    #[inline]
    fn index(&self, j: usize) -> &Self::Output {
        self.get(j)
    }
}
impl<'a, T> Index<usize> for ColMut<'a, T> {
    type Output = T;

    #[track_caller]
    #[inline]
    fn index(&self, j: usize) -> &Self::Output {
        self.rb().get(j)
    }
}
impl<'a, T> IndexMut<usize> for ColMut<'a, T> {
    #[track_caller]
    #[inline]
    fn index_mut(&mut self, j: usize) -> &mut Self::Output {
        self.rb_mut().get(j)
    }
}

impl<'a, T> IntoIterator for RowRef<'a, T> {
    type Item = &'a T;
    type IntoIter = ElemIter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ElemIter(self.transpose())
    }
}
impl<'a, T> IntoIterator for RowMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = ElemIterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ElemIterMut(self.transpose())
    }
}

impl<'a, T> IntoIterator for ColRef<'a, T> {
    type Item = &'a T;
    type IntoIter = ElemIter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ElemIter(self)
    }
}
impl<'a, T> IntoIterator for ColMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = ElemIterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ElemIterMut(self)
    }
}

/// Matrix iterators.
pub mod iter {
    use crate::{ColMut, ColRef, MatMut, MatRef, RowMut, RowRef};
    use reborrow::*;

    pub struct RowIter<'a, T>(pub(crate) MatRef<'a, T>);
    pub struct ColIter<'a, T>(pub(crate) MatRef<'a, T>);
    pub struct RowIterMut<'a, T>(pub(crate) MatMut<'a, T>);
    pub struct ColIterMut<'a, T>(pub(crate) MatMut<'a, T>);
    pub struct ElemIter<'a, T>(pub(crate) ColRef<'a, T>);
    pub struct ElemIterMut<'a, T>(pub(crate) ColMut<'a, T>);

    impl<'a, T> RowIter<'a, T> {
        #[inline]
        pub fn into_matrix(self) -> MatRef<'a, T> {
            self.0
        }
    }
    impl<'a, T> RowIterMut<'a, T> {
        #[inline]
        pub fn into_matrix(self) -> MatMut<'a, T> {
            self.0
        }
    }
    impl<'a, T> ColIter<'a, T> {
        #[inline]
        pub fn into_matrix(self) -> MatRef<'a, T> {
            self.0
        }
    }
    impl<'a, T> ColIterMut<'a, T> {
        #[inline]
        pub fn into_matrix(self) -> MatMut<'a, T> {
            self.0
        }
    }
    impl<'a, T> ElemIter<'a, T> {
        #[inline]
        pub fn into_col(self) -> ColRef<'a, T> {
            self.0
        }
        #[inline]
        pub fn into_row(self) -> RowRef<'a, T> {
            self.0.transpose()
        }
    }
    impl<'a, T> ElemIterMut<'a, T> {
        #[inline]
        pub fn into_col(self) -> ColMut<'a, T> {
            self.0
        }
        #[inline]
        pub fn into_row(self) -> RowMut<'a, T> {
            self.0.transpose()
        }
    }

    impl<'a, T> Copy for RowIter<'a, T> {}
    impl<'a, T> Copy for ColIter<'a, T> {}
    impl<'a, T> Copy for ElemIter<'a, T> {}
    impl<'a, T> Clone for RowIter<'a, T> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<'a, T> Clone for ColIter<'a, T> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<'a, T> Clone for ElemIter<'a, T> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<'b, 'a, T> Reborrow<'b> for RowIter<'a, T> {
        type Target = RowIter<'b, T>;
        #[inline]
        fn rb(&'b self) -> Self::Target {
            *self
        }
    }
    impl<'b, 'a, T> ReborrowMut<'b> for RowIter<'a, T> {
        type Target = RowIter<'b, T>;
        #[inline]
        fn rb_mut(&'b mut self) -> Self::Target {
            *self
        }
    }
    impl<'b, 'a, T> Reborrow<'b> for RowIterMut<'a, T> {
        type Target = RowIter<'b, T>;
        #[inline]
        fn rb(&'b self) -> Self::Target {
            RowIter(self.0.rb())
        }
    }
    impl<'b, 'a, T> ReborrowMut<'b> for RowIterMut<'a, T> {
        type Target = RowIterMut<'b, T>;
        #[inline]
        fn rb_mut(&'b mut self) -> Self::Target {
            RowIterMut(self.0.rb_mut())
        }
    }

    impl<'b, 'a, T> Reborrow<'b> for ColIter<'a, T> {
        type Target = ColIter<'b, T>;
        #[inline]
        fn rb(&'b self) -> Self::Target {
            *self
        }
    }
    impl<'b, 'a, T> ReborrowMut<'b> for ColIter<'a, T> {
        type Target = ColIter<'b, T>;
        #[inline]
        fn rb_mut(&'b mut self) -> Self::Target {
            *self
        }
    }
    impl<'b, 'a, T> Reborrow<'b> for ColIterMut<'a, T> {
        type Target = ColIter<'b, T>;
        #[inline]
        fn rb(&'b self) -> Self::Target {
            ColIter(self.0.rb())
        }
    }
    impl<'b, 'a, T> ReborrowMut<'b> for ColIterMut<'a, T> {
        type Target = ColIterMut<'b, T>;
        #[inline]
        fn rb_mut(&'b mut self) -> Self::Target {
            ColIterMut(self.0.rb_mut())
        }
    }

    impl<'b, 'a, T> Reborrow<'b> for ElemIter<'a, T> {
        type Target = ElemIter<'b, T>;
        #[inline]
        fn rb(&'b self) -> Self::Target {
            *self
        }
    }
    impl<'b, 'a, T> ReborrowMut<'b> for ElemIter<'a, T> {
        type Target = ElemIter<'b, T>;
        #[inline]
        fn rb_mut(&'b mut self) -> Self::Target {
            *self
        }
    }
    impl<'b, 'a, T> Reborrow<'b> for ElemIterMut<'a, T> {
        type Target = ElemIter<'b, T>;
        #[inline]
        fn rb(&'b self) -> Self::Target {
            ElemIter(self.0.rb())
        }
    }
    impl<'b, 'a, T> ReborrowMut<'b> for ElemIterMut<'a, T> {
        type Target = ElemIterMut<'b, T>;
        #[inline]
        fn rb_mut(&'b mut self) -> Self::Target {
            ElemIterMut(self.0.rb_mut())
        }
    }

    impl<'a, T> Iterator for ElemIter<'a, T> {
        type Item = &'a T;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let nrows = self.0.nrows();
            if nrows == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let rs = self.0.row_stride();
                let top = unsafe { &*ptr };
                let bot = unsafe { ColRef::from_raw_parts(ptr.wrapping_offset(rs), nrows - 1, rs) };

                self.0 = bot;

                Some(top)
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.0.nrows();
            (len, Some(len))
        }
    }
    impl<'a, T> DoubleEndedIterator for ElemIter<'a, T> {
        #[inline]
        fn next_back(&mut self) -> Option<Self::Item> {
            let nrows = self.0.nrows();
            if nrows == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let rs = self.0.row_stride();
                let top = unsafe { ColRef::from_raw_parts(ptr, nrows - 1, rs) };
                let bot = unsafe { &*ptr.wrapping_offset(rs * (nrows - 1) as isize) };

                self.0 = top;

                Some(bot)
            }
        }
    }

    impl<'a, T> Iterator for ElemIterMut<'a, T> {
        type Item = &'a mut T;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let nrows = self.0.nrows();
            if nrows == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let rs = self.0.row_stride();
                let top = unsafe { &mut *ptr };
                let bot = unsafe { ColMut::from_raw_parts(ptr.wrapping_offset(rs), nrows - 1, rs) };

                self.0 = bot;

                Some(top)
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.0.nrows();
            (len, Some(len))
        }
    }
    impl<'a, T> DoubleEndedIterator for ElemIterMut<'a, T> {
        #[inline]
        fn next_back(&mut self) -> Option<Self::Item> {
            let nrows = self.0.nrows();
            if nrows == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let rs = self.0.row_stride();
                let top = unsafe { ColMut::from_raw_parts(ptr, nrows - 1, rs) };
                let bot = unsafe { &mut *ptr.wrapping_offset(rs * (nrows - 1) as isize) };

                self.0 = top;

                Some(bot)
            }
        }
    }

    impl<'a, T> Iterator for RowIter<'a, T> {
        type Item = RowRef<'a, T>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let nrows = self.0.nrows();
            if nrows == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let ncols = self.0.ncols();
                let rs = self.0.row_stride();
                let cs = self.0.col_stride();
                let top = unsafe { Self::Item::from_raw_parts(ptr, ncols, cs) };
                let bot = unsafe {
                    MatRef::from_raw_parts(ptr.wrapping_offset(rs), nrows - 1, ncols, rs, cs)
                };

                self.0 = bot;

                Some(top)
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.0.nrows();
            (len, Some(len))
        }
    }
    impl<'a, T> DoubleEndedIterator for RowIter<'a, T> {
        #[inline]
        fn next_back(&mut self) -> Option<Self::Item> {
            let nrows = self.0.nrows();
            if nrows == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let ncols = self.0.ncols();
                let rs = self.0.row_stride();
                let cs = self.0.col_stride();
                let top = unsafe { MatRef::from_raw_parts(ptr, nrows - 1, ncols, rs, cs) };
                let bot = unsafe {
                    Self::Item::from_raw_parts(
                        ptr.wrapping_offset((nrows - 1) as isize * rs),
                        ncols,
                        cs,
                    )
                };

                self.0 = top;

                Some(bot)
            }
        }
    }

    impl<'a, T> Iterator for RowIterMut<'a, T> {
        type Item = RowMut<'a, T>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let nrows = self.0.nrows();
            if nrows == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let ncols = self.0.ncols();
                let rs = self.0.row_stride();
                let cs = self.0.col_stride();
                let top = unsafe { Self::Item::from_raw_parts(ptr, ncols, cs) };
                let bot = unsafe {
                    MatMut::from_raw_parts(ptr.wrapping_offset(rs), nrows - 1, ncols, rs, cs)
                };

                self.0 = bot;

                Some(top)
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.0.nrows();
            (len, Some(len))
        }
    }
    impl<'a, T> DoubleEndedIterator for RowIterMut<'a, T> {
        #[inline]
        fn next_back(&mut self) -> Option<Self::Item> {
            let nrows = self.0.nrows();
            if nrows == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let ncols = self.0.ncols();
                let rs = self.0.row_stride();
                let cs = self.0.col_stride();
                let top = unsafe { MatMut::from_raw_parts(ptr, nrows - 1, ncols, rs, cs) };
                let bot = unsafe {
                    Self::Item::from_raw_parts(
                        ptr.wrapping_offset((nrows - 1) as isize * rs),
                        ncols,
                        cs,
                    )
                };

                self.0 = top;

                Some(bot)
            }
        }
    }

    impl<'a, T> Iterator for ColIter<'a, T> {
        type Item = ColRef<'a, T>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let ncols = self.0.ncols();
            if ncols == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let nrows = self.0.nrows();
                let rs = self.0.row_stride();
                let cs = self.0.col_stride();
                let left = unsafe { Self::Item::from_raw_parts(ptr, nrows, rs) };
                let right = unsafe {
                    MatRef::from_raw_parts(ptr.wrapping_offset(cs), nrows, ncols - 1, rs, cs)
                };

                self.0 = right;
                Some(left)
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.0.ncols();
            (len, Some(len))
        }
    }
    impl<'a, T> DoubleEndedIterator for ColIter<'a, T> {
        #[inline]
        fn next_back(&mut self) -> Option<Self::Item> {
            let ncols = self.0.ncols();
            if ncols == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let nrows = self.0.nrows();
                let rs = self.0.row_stride();
                let cs = self.0.col_stride();
                let left = unsafe { MatRef::from_raw_parts(ptr, nrows, ncols - 1, rs, cs) };
                let right = unsafe {
                    Self::Item::from_raw_parts(
                        ptr.wrapping_offset((ncols - 1) as isize * cs),
                        nrows,
                        rs,
                    )
                };

                self.0 = left;
                Some(right)
            }
        }
    }
    impl<'a, T> Iterator for ColIterMut<'a, T> {
        type Item = ColMut<'a, T>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let ncols = self.0.ncols();
            if ncols == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let nrows = self.0.nrows();
                let rs = self.0.row_stride();
                let cs = self.0.col_stride();
                let left = unsafe { Self::Item::from_raw_parts(ptr, nrows, rs) };
                let right = unsafe {
                    MatMut::from_raw_parts(ptr.wrapping_offset(cs), nrows, ncols - 1, rs, cs)
                };

                self.0 = right;
                Some(left)
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.0.ncols();
            (len, Some(len))
        }
    }
    impl<'a, T> DoubleEndedIterator for ColIterMut<'a, T> {
        #[inline]
        fn next_back(&mut self) -> Option<Self::Item> {
            let ncols = self.0.ncols();
            if ncols == 0 {
                None
            } else {
                let ptr = self.0.base.ptr.as_ptr();
                let nrows = self.0.nrows();
                let rs = self.0.row_stride();
                let cs = self.0.col_stride();
                let left = unsafe { MatMut::from_raw_parts(ptr, nrows, ncols - 1, rs, cs) };
                let right = unsafe {
                    Self::Item::from_raw_parts(
                        ptr.wrapping_offset((ncols - 1) as isize * cs),
                        nrows,
                        rs,
                    )
                };

                self.0 = left;
                Some(right)
            }
        }
    }

    impl<'a, T> ExactSizeIterator for RowIter<'a, T> {}
    impl<'a, T> ExactSizeIterator for RowIterMut<'a, T> {}
    impl<'a, T> ExactSizeIterator for ColIter<'a, T> {}
    impl<'a, T> ExactSizeIterator for ColIterMut<'a, T> {}
    impl<'a, T> ExactSizeIterator for ElemIter<'a, T> {}
    impl<'a, T> ExactSizeIterator for ElemIterMut<'a, T> {}
}

impl<'a, T: Debug + 'static> Debug for MatRef<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        struct DebugRowSlice<'a, T>(RowRef<'a, T>);
        struct ComplexDebug<'a, T>(&'a T);

        impl<'a, T: Debug + 'static> Debug for ComplexDebug<'a, T> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let id = TypeId::of::<T>();

                fn as_debug(t: impl Debug) -> impl Debug {
                    t
                }
                if id == TypeId::of::<c32>() {
                    let value: c32 = unsafe { transmute_copy(self.0) };
                    let re = as_debug(value.re);
                    let im = as_debug(value.im);
                    re.fmt(f)?;
                    f.write_str(" + ")?;
                    im.fmt(f)?;
                    f.write_str("I")
                } else if id == TypeId::of::<c64>() {
                    let value: c64 = unsafe { transmute_copy(self.0) };
                    let re = as_debug(value.re);
                    let im = as_debug(value.im);
                    re.fmt(f)?;
                    f.write_str(" + ")?;
                    im.fmt(f)?;
                    f.write_str(" * I")
                } else {
                    self.0.fmt(f)
                }
            }
        }

        impl<'a, T: Debug + 'static> Debug for DebugRowSlice<'a, T> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_list()
                    .entries(self.0.into_iter().map(|x| ComplexDebug(x)))
                    .finish()
            }
        }

        write!(f, "[\n")?;
        for elem in self.into_row_iter().map(DebugRowSlice) {
            elem.fmt(f)?;
            f.write_str(",\n")?;
        }
        write!(f, "]")
    }
}
impl<'a, T: Debug + 'static> Debug for MatMut<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}
impl<'a, T: Debug + 'static> Debug for RowRef<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().as_2d().fmt(f)
    }
}
impl<'a, T: Debug + 'static> Debug for RowMut<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().as_2d().fmt(f)
    }
}

impl<'a, T: Debug + 'static> Debug for ColRef<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().as_2d().fmt(f)
    }
}
impl<'a, T: Debug + 'static> Debug for ColMut<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().as_2d().fmt(f)
    }
}

#[doc(hidden)]
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

#[doc(hidden)]
#[inline]
pub fn round_up_to(n: usize, k: usize) -> usize {
    (n + (k - 1)) / k * k
}

#[doc(hidden)]
#[inline]
pub fn is_vectorizable<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
        || TypeId::of::<T>() == TypeId::of::<f32>()
        || TypeId::of::<T>() == TypeId::of::<c64>()
        || TypeId::of::<T>() == TypeId::of::<c32>()
}

#[doc(hidden)]
#[inline]
pub fn align_for<T: 'static>() -> usize {
    if is_vectorizable::<T>() {
        CACHELINE_ALIGN
    } else {
        core::mem::align_of::<T>()
    }
}

#[doc(hidden)]
#[inline]
pub unsafe fn as_uninit<T>(slice: &mut [T]) -> &mut [MaybeUninit<T>] {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();
    core::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<T>, len)
}

#[doc(hidden)]
#[inline]
pub unsafe fn from_mut_slice<T>(
    slice: &mut [T],
    nrows: usize,
    ncols: usize,
    row_stride: usize,
    col_stride: usize,
) -> MatMut<'_, T> {
    MatMut::from_raw_parts(
        slice.as_mut_ptr(),
        nrows,
        ncols,
        row_stride as isize,
        col_stride as isize,
    )
}

#[doc(hidden)]
#[inline]
pub unsafe fn from_uninit_mut_slice<T>(
    slice: &mut [MaybeUninit<T>],
    nrows: usize,
    ncols: usize,
    row_stride: usize,
    col_stride: usize,
) -> MatMut<'_, T> {
    MatMut::from_raw_parts(
        slice.as_mut_ptr() as *mut T,
        nrows,
        ncols,
        row_stride as isize,
        col_stride as isize,
    )
}

// https://docs.rs/itertools/0.7.8/src/itertools/lib.rs.html#247-269
#[macro_export]
#[doc(hidden)]
macro_rules! izip {
    // eg. izip!(((a, b), c) => (a, b, c) , dd , ee )
    (@ __closure @ $p:pat => $tup:expr) => {
        |$p| $tup
    };

    // The "b" identifier is a different identifier on each recursion level thanks to hygiene.
    (@ __closure @ $p:pat => ( $($tup:tt)* ) , $_iter:expr $( , $tail:expr )*) => {
        $crate::izip!(@ __closure @ ($p, b) => ( $($tup)*, b ) $( , $tail )*)
    };

    ( $first:expr $(,)?) => {
        ::core::iter::IntoIterator::into_iter($first)
    };
    ( $first:expr, $($rest:expr),+ $(,)?) => {
        {
            #[allow(unused_imports)]
            ::core::iter::IntoIterator::into_iter($first)
                $(.zip($rest))*
                .map($crate::izip!(@ __closure @ a => (a) $( , $rest )*))
        }
    };
}

/// Unsafe: Creates a temporary matrix of possibly uninitialized values, from the given memory
/// stack.
///
/// # Warning
///
/// Working with uninitialized values is trickier in Rust than other languages. In particular,
/// references must always point to initialized data, which means that using
/// [`MatMut::get`], [`MatMut::get_unchecked`], the indexing operator, etc., on the resulting matrix
/// can cause undefined behavior.
///
/// Use the pointer access API instead: e.g., [`MatMut::ptr_at`] or [`zip::MatUninit`].
#[macro_export]
macro_rules! temp_mat_uninit {
    {
        $(
            let ($id: pat, $stack_id: pat) = unsafe {
                temp_mat_uninit::<$ty: ty>(
                    $nrows: expr,
                    $ncols: expr,
                    $stack: expr$(,)?
                )
            };
        )*
    } => {
        $(
            let nrows: usize = $nrows;
            let ncols: usize = $ncols;
            let col_stride = if $crate::is_vectorizable::<$ty>() {
                $crate::round_up_to(
                    nrows,
                    $crate::align_for::<$ty>() / ::core::mem::size_of::<$ty>()
                )
            } else {
                nrows
            };

            let (mut temp_data, $stack_id) = $stack.make_aligned_uninit::<$ty>(
                ncols * col_stride,
                $crate::align_for::<$ty>(),
            );

            #[allow(unused_unsafe)]
            let $id = unsafe {
                $crate::from_uninit_mut_slice(
                    &mut (temp_data),
                    nrows,
                    ncols,
                    1,
                    col_stride,
                )
            };
        )*
    };
}

/// Creates a temporary matrix of zeroed values, from the given memory stack.
#[macro_export]
macro_rules! temp_mat_zeroed {
    {
        $(
            let ($id: pat, $stack_id: pat) = temp_mat_zeroed::<$ty: ty>(
                $nrows: expr,
                $ncols: expr,
                $stack: expr$(,)?
            );
        )*
    } => {
        $(
            let nrows: usize = $nrows;
            let ncols: usize = $ncols;
            let col_stride = if $crate::is_vectorizable::<$ty>() {
                $crate::round_up_to(
                    nrows,
                    $crate::align_for::<$ty>() / ::core::mem::size_of::<$ty>()
                )
            } else {
                nrows
            };

            let (mut temp_data, $stack_id) = $stack.make_aligned_with(
                ncols * col_stride,
                $crate::align_for::<$ty>(),
                |_| <$ty as $crate::ComplexField>::zero(),
            );

            #[allow(unused_unsafe)]
            let $id = unsafe {
                $crate::from_mut_slice(
                    &mut temp_data,
                    nrows,
                    ncols,
                    1,
                    col_stride,
                )
            };
        )*
    };
}

/// Returns the stack requirements for creating a temporary matrix with the given dimensions.
#[inline]
pub fn temp_mat_req<T: 'static>(nrows: usize, ncols: usize) -> Result<StackReq, SizeOverflow> {
    let col_stride = if is_vectorizable::<T>() {
        round_up_to(nrows, align_for::<T>() / size_of::<T>())
    } else {
        nrows
    };

    StackReq::try_new_aligned::<T>(col_stride * ncols, align_for::<T>())
}

struct RawMat<T: 'static> {
    ptr: NonNull<T>,
    row_capacity: usize,
    col_capacity: usize,
}

#[cold]
fn capacity_overflow_impl() -> ! {
    panic!("capacity overflow")
}

#[cold]
fn capacity_overflow<T>() -> T {
    capacity_overflow_impl();
}

impl<T: 'static> RawMat<T> {
    pub fn new(row_capacity: usize, col_capacity: usize) -> Self {
        if core::mem::size_of::<T>() == 0 {
            Self {
                ptr: NonNull::<T>::dangling(),
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
                core::ptr::NonNull::<T>::dangling()
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

impl<T> Drop for RawMat<T> {
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

struct BlockGuard<T> {
    ptr: *mut T,
    nrows: usize,
    ncols: usize,
    cs: isize,
}
struct ColGuard<T> {
    ptr: *mut T,
    nrows: usize,
}

impl<T> Drop for BlockGuard<T> {
    fn drop(&mut self) {
        for j in 0..self.ncols {
            let ptr_j = self.ptr.wrapping_offset(j as isize * self.cs);
            // SAFETY: this is safe because we created these elements and need to
            // drop them
            let slice = unsafe { core::slice::from_raw_parts_mut(ptr_j, self.nrows) };
            unsafe { core::ptr::drop_in_place(slice) };
        }
    }
}
impl<T> Drop for ColGuard<T> {
    fn drop(&mut self) {
        let ptr = self.ptr;
        // SAFETY: this is safe because we created these elements and need to
        // drop them
        let slice = unsafe { core::slice::from_raw_parts_mut(ptr, self.nrows) };
        unsafe { core::ptr::drop_in_place(slice) };
    }
}

/// Owning matrix structure stored in column major format.
///
/// A matrix can be thought of as a 2D array of values.
/// These values are stored in memory so that the columns are contiguous.
///
/// # Note
///
/// Note that the matrix as a whole may not necessarily be contiguous.
/// The implementation may add padding at the end of each column when
/// overaligning each column can provide a performance gain.
///
/// Let us consider a 3×4 matrix
///
/// ```notrust
///  0 │ 3 │ 6 │  9
/// ───┼───┼───┼───
///  1 │ 4 │ 7 │ 10
/// ───┼───┼───┼───
///  2 │ 5 │ 8 │ 11
/// ```
///
/// The memory representation of such a matrix could look like the following:
///
/// ```notrust
/// 0 1 2 X 3 4 5 X 6 7 8 X 9 10 11 X
/// ```
/// where `X` represents padding elements.
pub struct Mat<T: 'static> {
    raw: RawMat<T>,
    nrows: usize,
    ncols: usize,
}

impl<T: 'static> Default for Mat<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Clone for Mat<T> {
    fn clone(&self) -> Self {
        let mut other = Self::with_capacity(self.row_capacity(), self.col_capacity());
        let this = self.as_ref();
        other.resize_with(
            |i, j| unsafe { (*this.ptr_in_bounds_at_unchecked(i, j)).clone() },
            self.nrows(),
            self.ncols(),
        );
        other
    }
}

impl<T: 'static> Mat<T> {
    /// Returns a new matrix with dimensions `(0, 0)`. This does not allocate.
    #[inline]
    pub fn new() -> Self {
        Self {
            raw: RawMat::<T> {
                ptr: NonNull::<T>::dangling(),
                row_capacity: 0,
                col_capacity: 0,
            },
            nrows: 0,
            ncols: 0,
        }
    }

    /// Returns a matrix from preallocated pointer, dimensions, and capacities.
    ///
    /// # Safety
    ///
    /// The inputs to this function must be acquired from the return value of some previous call
    /// to `Self::into_raw_parts`.
    #[inline]
    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        nrows: usize,
        ncols: usize,
        row_capacity: usize,
        col_capacity: usize,
    ) -> Self {
        Self {
            raw: RawMat::<T> {
                ptr: NonNull::new_unchecked(ptr),
                row_capacity,
                col_capacity,
            },
            nrows,
            ncols,
        }
    }

    /// Consumes `self` and returns its raw parts in this order: pointer to data, number of rows,
    /// number of columns, row capacity and column capacity.
    #[inline]
    pub fn into_raw_parts(self) -> (*mut T, usize, usize, usize, usize) {
        let mut m = core::mem::ManuallyDrop::<Mat<T>>::new(self);
        (
            m.as_mut_ptr(),
            m.nrows(),
            m.ncols(),
            m.row_capacity(),
            m.col_capacity(),
        )
    }

    /// Returns a new matrix with dimensions `(0, 0)`, with enough capacity to hold a maximum of
    /// `row_capacity` rows and `col_capacity` columns without reallocating. If either is `0`,
    /// the matrix will not allocate.
    ///
    /// # Panics
    ///
    /// Panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn with_capacity(row_capacity: usize, col_capacity: usize) -> Self {
        Self {
            raw: RawMat::<T>::new(row_capacity, col_capacity),
            nrows: 0,
            ncols: 0,
        }
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with the provided function.
    ///
    /// # Panics
    ///
    /// Panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn with_dims(f: impl FnMut(usize, usize) -> T, nrows: usize, ncols: usize) -> Self {
        let mut this = Self::new();
        this.resize_with(f, nrows, ncols);
        this
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with zeros.
    ///
    /// # Panics
    ///
    /// Panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn zeros(nrows: usize, ncols: usize) -> Self
    where
        T: ComplexField,
    {
        Self::with_dims(|_, _| T::zero(), nrows, ncols)
    }

    /// Set the dimensions of the matrix.
    ///
    /// # Safety
    ///
    /// * `nrows` must be less than `self.row_capacity()`.
    /// * `ncols` must be less than `self.col_capacity()`.
    /// * The elements that were previously out of bounds but are now in bounds must be
    /// initialized.
    #[inline]
    pub unsafe fn set_dims(&mut self, nrows: usize, ncols: usize) {
        self.nrows = nrows;
        self.ncols = ncols;
    }

    /// Returns a pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.raw.ptr.as_ptr()
    }

    /// Returns a mutable pointer to the data of the matrix.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.raw.ptr.as_ptr()
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
    fn do_reserve_exact(&mut self, mut new_row_capacity: usize, mut new_col_capacity: usize) {
        use core::mem::ManuallyDrop;

        new_row_capacity = self.row_capacity().max(new_row_capacity);
        new_col_capacity = self.col_capacity().max(new_col_capacity);

        let new_ptr = if self.row_capacity() == new_row_capacity
            && self.row_capacity() != 0
            && self.col_capacity() != 0
        {
            // case 1:
            // we have enough row capacity, and we've already allocated memory.
            // use realloc to get extra column memory

            use alloc::alloc::{handle_alloc_error, realloc, Layout};

            // this shouldn't overflow since we already hold this many bytes
            let old_cap = self.row_capacity() * self.col_capacity();
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
            // than self.row_capacity() and self.col_capacity() respectively, and the computed
            // product doesn't overflow.
            // * new_cap_bytes, when rounded up to the nearest multiple of the alignment does not
            // overflow, since we checked that we can create new_layout with it.
            unsafe {
                let old_ptr = self.as_mut_ptr();
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
                let m = ManuallyDrop::new(RawMat::<T>::new(new_row_capacity, new_col_capacity));
                m.ptr.as_ptr()
            };

            let old_ptr = self.as_mut_ptr();

            // copy each column to new matrix
            for j in 0..self.ncols() {
                // SAFETY:
                // * pointer offsets can't overflow since they're within an already allocated
                // memory region less than isize::MAX bytes in size.
                // * new and old allocation can't overlap, so copy_nonoverlapping is fine here.
                unsafe {
                    let old_ptr = old_ptr.add(j * self.row_capacity());
                    let new_ptr = new_ptr.add(j * new_row_capacity);
                    core::ptr::copy_nonoverlapping(old_ptr, new_ptr, self.nrows());
                }
            }

            // deallocate old matrix memory
            let _ = RawMat::<T> {
                // SAFETY: this ptr was checked to be non null, or was acquired from a NonNull
                // pointer.
                ptr: unsafe { NonNull::new_unchecked(old_ptr) },
                row_capacity: self.row_capacity(),
                col_capacity: self.col_capacity(),
            };

            new_ptr
        };
        self.raw.row_capacity = new_row_capacity;
        self.raw.col_capacity = new_col_capacity;
        self.raw.ptr = unsafe { NonNull::<T>::new_unchecked(new_ptr) };
    }

    /// Reserves the minimum capacity for `row_capacity` rows and `col_capacity`
    /// columns without reallocating. Does nothing if the capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn reserve_exact(&mut self, row_capacity: usize, col_capacity: usize) {
        if self.row_capacity() >= row_capacity && self.col_capacity() >= col_capacity {
            // do nothing
        } else if core::mem::size_of::<T>() == 0 {
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
        fancy_debug_assert!(row_start <= row_end);
        fancy_debug_assert!(col_start <= col_end);

        let ptr = self.as_mut_ptr();

        for j in col_start..col_end {
            let ptr_j = ptr.wrapping_offset(j as isize * self.col_stride());
            for i in row_start..row_end {
                // SAFETY: this points to a valid matrix element at index (i, j), which
                // is within bounds
                let ptr_ij = ptr_j.add(i);

                // SAFETY: we drop an object that is within its lifetime since the matrix
                // contains valid elements at each index within bounds
                core::ptr::drop_in_place(ptr_ij);
            }
        }
    }

    unsafe fn insert_block_with<F: FnMut(usize, usize) -> T>(
        &mut self,
        f: &mut F,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) {
        fancy_debug_assert!(row_start <= row_end);
        fancy_debug_assert!(col_start <= col_end);

        let ptr = self.as_mut_ptr();

        let mut block_guard = BlockGuard::<T> {
            ptr: ptr.wrapping_add(row_start),
            nrows: row_end - row_start,
            ncols: 0,
            cs: self.col_stride(),
        };

        for j in col_start..col_end {
            let ptr_j = ptr.wrapping_offset(j as isize * self.col_stride());

            // create a guard for the same purpose as the previous one
            let mut col_guard = ColGuard::<T> {
                // SAFETY: same as above
                ptr: ptr_j.wrapping_add(row_start),
                nrows: 0,
            };

            for i in row_start..row_end {
                // SAFETY:
                // * pointer to element at index (i, j), which is within the
                // allocation since we reserved enough space
                // * writing to this memory region is sound since it is properly
                // aligned and valid for writes
                let ptr_ij = ptr_j.add(i);
                core::ptr::write(ptr_ij, f(i, j));
                col_guard.nrows += 1;
            }
            core::mem::forget(col_guard);
            block_guard.ncols += 1;
        }
        core::mem::forget(block_guard);
    }

    fn erase_last_cols(&mut self, new_ncols: usize) {
        let old_ncols = self.ncols();

        fancy_debug_assert!(new_ncols <= old_ncols);

        // change the size before dropping the elements, since if one of them panics the
        // matrix drop function will double drop them.
        self.ncols = new_ncols;

        unsafe {
            self.erase_block(0, self.nrows(), new_ncols, old_ncols);
        }
    }

    fn erase_last_rows(&mut self, new_nrows: usize) {
        let old_nrows = self.nrows();

        fancy_debug_assert!(new_nrows <= old_nrows);

        // see comment above
        self.nrows = new_nrows;
        unsafe {
            self.erase_block(new_nrows, old_nrows, 0, self.ncols());
        }
    }

    unsafe fn insert_last_cols_with<F: FnMut(usize, usize) -> T>(
        &mut self,
        f: &mut F,
        new_ncols: usize,
    ) {
        let old_ncols = self.ncols();

        fancy_debug_assert!(new_ncols > old_ncols);

        self.insert_block_with(f, 0, self.nrows(), old_ncols, new_ncols);
        self.ncols = new_ncols;
    }

    unsafe fn insert_last_rows_with<F: FnMut(usize, usize) -> T>(
        &mut self,
        f: &mut F,
        new_nrows: usize,
    ) {
        let old_nrows = self.nrows();

        fancy_debug_assert!(new_nrows > old_nrows);

        self.insert_block_with(f, old_nrows, new_nrows, 0, self.ncols());
        self.nrows = new_nrows;
    }

    /// Resizes the matrix in-place so that the new dimensions are `(new_nrows, new_ncols)`.
    /// Elements that are now out of bounds are dropped, while new elements are created with the
    /// given function `f`, so that elements at position `(i, j)` are created by calling `f(i, j)`.
    pub fn resize_with(
        &mut self,
        f: impl FnMut(usize, usize) -> T,
        new_nrows: usize,
        new_ncols: usize,
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

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, T> {
        unsafe {
            MatRef::<'_, T>::from_raw_parts(
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
    pub fn as_mut(&mut self) -> MatMut<'_, T> {
        unsafe {
            MatMut::<'_, T>::from_raw_parts(
                self.as_mut_ptr(),
                self.nrows(),
                self.ncols(),
                1,
                self.col_stride(),
            )
        }
    }
}

impl<T> Drop for Mat<T> {
    fn drop(&mut self) {
        let mut ptr = self.raw.ptr.as_ptr();
        let nrows = self.nrows;
        let ncols = self.ncols;
        let cs = self.raw.row_capacity;

        for _ in 0..ncols {
            for i in 0..nrows {
                // SAFETY: these elements were previously created in this storage.
                unsafe {
                    core::ptr::drop_in_place(ptr.add(i));
                }
            }
            ptr = ptr.wrapping_add(cs);
        }
    }
}

impl<T: Debug + 'static> Debug for Mat<T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T> Index<(usize, usize)> for Mat<T> {
    type Output = T;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        self.as_ref().get(i, j)
    }
}

impl<T> IndexMut<(usize, usize)> for Mat<T> {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        self.as_mut().get(i, j)
    }
}

impl<T> Add<Mat<T>> for Mat<T>
where
    T: Copy + Add<T, Output = T>,
{
    type Output = Mat<T>;

    #[inline]
    fn add(self, rhs: Mat<T>) -> Self::Output {
        fancy_assert!((self.nrows(), self.ncols()) == (rhs.nrows(), rhs.ncols()));
        Self::Output::with_dims(
            |i, j| self[(i, j)] + rhs[(i, j)],
            self.nrows(),
            self.ncols(),
        )
    }
}

impl<T> Add<&Mat<T>> for &Mat<T>
where
    T: Copy + Add<T, Output = T>,
{
    type Output = Mat<T>;

    #[inline]
    fn add(self, rhs: &Mat<T>) -> Self::Output {
        fancy_assert!((self.nrows(), self.ncols()) == (rhs.nrows(), rhs.ncols()));
        Self::Output::with_dims(
            |i, j| self[(i, j)] + rhs[(i, j)],
            self.nrows(),
            self.ncols(),
        )
    }
}

impl<T> Sub<Mat<T>> for Mat<T>
where
    T: Copy + Sub<T, Output = T>,
{
    type Output = Mat<T>;

    #[inline]
    fn sub(self, rhs: Mat<T>) -> Self::Output {
        fancy_assert!((self.nrows(), self.ncols()) == (rhs.nrows(), rhs.ncols()));
        Self::Output::with_dims(
            |i, j| self[(i, j)] - rhs[(i, j)],
            self.nrows(),
            self.ncols(),
        )
    }
}

impl<T> Sub<&Mat<T>> for &Mat<T>
where
    T: Copy + Sub<T, Output = T>,
{
    type Output = Mat<T>;

    #[inline]
    fn sub(self, rhs: &Mat<T>) -> Self::Output {
        fancy_assert!((self.nrows(), self.ncols()) == (rhs.nrows(), rhs.ncols()));
        Self::Output::with_dims(
            |i, j| self[(i, j)] - rhs[(i, j)],
            self.nrows(),
            self.ncols(),
        )
    }
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

/// Returns a [`Mat`] containing the arguments.
///
/// # Example
///
/// ```
/// use faer_core::mat;
///
/// let m = mat![
///     [0.0, 3.0, 6.0, 9.0],
///     [1.0, 4.0, 7.0, 10.0],
///     [2.0, 5.0, 8.0, 11.0],
/// ];
///
/// assert_eq!(m[(0, 0)], 0.0);
/// assert_eq!(m[(1, 0)], 1.0);
/// assert_eq!(m[(2, 0)], 2.0);
///
/// assert_eq!(m[(0, 1)], 3.0);
/// assert_eq!(m[(1, 1)], 4.0);
/// assert_eq!(m[(2, 1)], 5.0);
///
/// assert_eq!(m[(0, 2)], 6.0);
/// assert_eq!(m[(1, 2)], 7.0);
/// assert_eq!(m[(2, 2)], 8.0);
///
/// assert_eq!(m[(0, 3)], 9.0);
/// assert_eq!(m[(1, 3)], 10.0);
/// assert_eq!(m[(2, 3)], 11.0);
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

            let mut m = $crate::Mat::<_>::with_capacity(nrows, ncols);
            let dst = m.as_mut_ptr();
            let mut src = data.as_ptr() as *const _;
            let _ = || src = &data[0][0];

            #[allow(unused_unsafe)]
            unsafe {
                ::core::ptr::copy_nonoverlapping(src, dst, ncols * nrows);
                m.set_dims(nrows, ncols);
            }
            m
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let slice = unsafe { MatRef::from_raw_parts(data.as_ptr(), 2, 3, 3, 1) };

        fancy_assert!(slice.rb().get(0, 0) == &1.0);
        fancy_assert!(slice.rb().get(0, 1) == &2.0);
        fancy_assert!(slice.rb().get(0, 2) == &3.0);

        fancy_assert!(slice.rb().get(1, 0) == &4.0);
        fancy_assert!(slice.rb().get(1, 1) == &5.0);
        fancy_assert!(slice.rb().get(1, 2) == &6.0);

        // miri tests
        for r in slice.rb().into_row_iter() {
            for _ in r {}
        }
        for r in slice.rb().into_row_iter().rev() {
            for _ in r.into_iter().rev() {}
        }

        for c in slice.rb().into_col_iter() {
            for _ in c {}
        }
        for c in slice.rb().into_col_iter().rev() {
            for _ in c.into_iter().rev() {}
        }
    }

    #[test]
    fn basic_slice_mut() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut slice = unsafe { MatMut::from_raw_parts(data.as_mut_ptr(), 2, 3, 3, 1) };

        fancy_assert!(slice.rb_mut().get(0, 0) == &mut 1.0);
        fancy_assert!(slice.rb_mut().get(0, 1) == &mut 2.0);
        fancy_assert!(slice.rb_mut().get(0, 2) == &mut 3.0);

        fancy_assert!(slice.rb_mut().get(1, 0) == &mut 4.0);
        fancy_assert!(slice.rb_mut().get(1, 1) == &mut 5.0);
        fancy_assert!(slice.rb_mut().get(1, 2) == &mut 6.0);

        // miri tests
        for r in slice.rb_mut().into_row_iter() {
            for _ in r {}
        }
        for r in slice.rb_mut().into_row_iter().rev() {
            for _ in r.into_iter().rev() {}
        }

        for c in slice.rb_mut().into_col_iter() {
            for _ in c {}
        }
        for c in slice.rb_mut().into_col_iter().rev() {
            for _ in c.into_iter().rev() {}
        }
    }

    #[test]
    fn empty() {
        {
            let m = Mat::<f64>::new();
            fancy_assert!(m.nrows() == 0);
            fancy_assert!(m.ncols() == 0);
            fancy_assert!(m.row_capacity() == 0);
            fancy_assert!(m.col_capacity() == 0);
        }

        {
            let m = Mat::<f64>::with_capacity(100, 120);
            fancy_assert!(m.nrows() == 0);
            fancy_assert!(m.ncols() == 0);
            fancy_assert!(m.row_capacity() == 100);
            fancy_assert!(m.col_capacity() == 120);
        }
    }

    #[test]
    fn reserve() {
        let mut m = Mat::<f64>::new();

        m.reserve_exact(0, 0);
        fancy_assert!(m.row_capacity() == 0);
        fancy_assert!(m.col_capacity() == 0);

        m.reserve_exact(1, 1);
        fancy_assert!(m.row_capacity() == 1);
        fancy_assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 0);
        fancy_assert!(m.row_capacity() == 2);
        fancy_assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 3);
        fancy_assert!(m.row_capacity() == 2);
        fancy_assert!(m.col_capacity() == 3);
    }

    #[test]
    fn reserve_zst() {
        let mut m = Mat::<()>::new();

        m.reserve_exact(0, 0);
        fancy_assert!(m.row_capacity() == 0);
        fancy_assert!(m.col_capacity() == 0);

        m.reserve_exact(1, 1);
        fancy_assert!(m.row_capacity() == 1);
        fancy_assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 0);
        fancy_assert!(m.row_capacity() == 2);
        fancy_assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 3);
        fancy_assert!(m.row_capacity() == 2);
        fancy_assert!(m.col_capacity() == 3);

        m.reserve_exact(usize::MAX, usize::MAX);
    }

    #[test]
    fn resize() {
        let mut m = Mat::new();
        let f = |i, j| i as f64 - j as f64;
        m.resize_with(f, 2, 3);
        fancy_assert!(m[(0, 0)] == 0.0);
        fancy_assert!(m[(0, 1)] == -1.0);
        fancy_assert!(m[(0, 2)] == -2.0);
        fancy_assert!(m[(1, 0)] == 1.0);
        fancy_assert!(m[(1, 1)] == 0.0);
        fancy_assert!(m[(1, 2)] == -1.0);

        m.resize_with(f, 1, 2);
        fancy_assert!(m[(0, 0)] == 0.0);
        fancy_assert!(m[(0, 1)] == -1.0);

        m.resize_with(f, 2, 1);
        fancy_assert!(m[(0, 0)] == 0.0);
        fancy_assert!(m[(1, 0)] == 1.0);

        m.resize_with(f, 1, 2);
        fancy_assert!(m[(0, 0)] == 0.0);
        fancy_assert!(m[(0, 1)] == -1.0);
    }

    #[test]
    fn matrix_macro() {
        let x = mat![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];

        fancy_assert!(x[(0, 0)] == 1.0);
        fancy_assert!(x[(0, 1)] == 2.0);
        fancy_assert!(x[(0, 2)] == 3.0);

        fancy_assert!(x[(1, 0)] == 4.0);
        fancy_assert!(x[(1, 1)] == 5.0);
        fancy_assert!(x[(1, 2)] == 6.0);

        fancy_assert!(x[(2, 0)] == 7.0);
        fancy_assert!(x[(2, 1)] == 8.0);
        fancy_assert!(x[(2, 2)] == 9.0);

        fancy_assert!(x[(3, 0)] == 10.0);
        fancy_assert!(x[(3, 1)] == 11.0);
        fancy_assert!(x[(3, 2)] == 12.0);
    }

    #[test]
    fn resize_zst() {
        // miri test
        let mut m = Mat::new();
        let f = |_i, _j| ();
        m.resize_with(f, 2, 3);
        m.resize_with(f, 1, 2);
        m.resize_with(f, 2, 1);
        m.resize_with(f, 1, 2);
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
    fn add_mat() {
        let x = mat![[1.0, -2.0], [4.0, -8.0],] + mat![[4.0, 5.0], [-6.0, -7.0],];

        fancy_assert!(x[(0, 0)] == 5.0);
        fancy_assert!(x[(0, 1)] == 3.0);

        fancy_assert!(x[(1, 0)] == -2.0);
        fancy_assert!(x[(1, 1)] == -15.0);

        let y = &mat![
            [c64::new(1.0, 2.0), c64::new(3.0, -4.0)],
            [c64::new(-1.0, 2.0), c64::new(-3.0, -4.0)],
        ] + &mat![
            [c64::new(4.0, 5.0), c64::new(6.0, 7.0)],
            [c64::new(4.0, 5.0), c64::new(6.0, 7.0)],
        ];

        fancy_assert!(y[(0, 0)] == c64::new(5.0, 7.0));
        fancy_assert!(y[(0, 1)] == c64::new(9.0, 3.0));

        fancy_assert!(y[(1, 0)] == c64::new(3.0, 7.0));
        fancy_assert!(y[(1, 1)] == c64::new(3.0, 3.0));
    }

    #[test]
    #[should_panic]
    fn add_different_size() {
        let _ = &mat![[1.0, -2.0], [4.0, -8.0],] + &mat![[4.0, 5.0],];
    }

    #[test]
    fn sub_mat() {
        let x =
            mat![[1.0, -2.0, 3.0], [4.0, -8.0, 2.0],] - mat![[4.0, 5.0, 3.0], [-6.0, -7.0, 4.0],];

        fancy_assert!(x[(0, 0)] == -3.0);
        fancy_assert!(x[(0, 1)] == -7.0);
        fancy_assert!(x[(0, 2)] == 0.0);

        fancy_assert!(x[(1, 0)] == 10.0);
        fancy_assert!(x[(1, 1)] == -1.0);
        fancy_assert!(x[(1, 2)] == -2.0);

        let y = &mat![
            [c64::new(4.0, 5.0), c64::new(6.0, 7.0)],
            [c64::new(1.0, 2.0), c64::new(2.0, 3.0)],
        ] - &mat![
            [c64::new(1.0, 2.0), c64::new(3.0, -4.0)],
            [c64::new(-1.0, 2.0), c64::new(-3.0, -4.0)],
        ];

        fancy_assert!(y[(0, 0)] == c64::new(3.0, 3.0));
        fancy_assert!(y[(0, 1)] == c64::new(3.0, 11.0));

        fancy_assert!(y[(1, 0)] == c64::new(2.0, 0.0));
        fancy_assert!(y[(1, 1)] == c64::new(5.0, 7.0));
    }

    #[test]
    #[should_panic]
    fn sub_different_size() {
        let _ = &mat![[1.0, -2.0], [4.0, -8.0],] - &mat![[4.0, 5.0],];
    }
}
