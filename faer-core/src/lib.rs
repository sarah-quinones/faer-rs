use assert2::{assert, debug_assert};
use coe::Coerce;
use core::{
    fmt::Debug,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ptr::NonNull,
};
use dyn_stack::{DynArray, DynStack, SizeOverflow, StackReq};
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

#[doc(hidden)]
pub mod simd;

#[inline(always)]
#[doc(hidden)]
pub unsafe fn transmute_unchecked<From, To>(t: From) -> To {
    assert!(core::mem::size_of::<From>() == core::mem::size_of::<To>());
    core::mem::transmute_copy(&ManuallyDrop::new(t))
}

fn sqrt_impl<E: RealField>(re: E, im: E) -> (E, E) {
    let im_sign = if im >= E::zero() {
        E::one()
    } else {
        E::one().neg()
    };
    let half = E::from_f64(0.5);

    let abs = (re.abs2().add(&im.abs2())).sqrt();
    let sum = re.add(&abs);
    // to avoid the sum being negative with inexact floating point arithmetic (i.e., double-double)
    let sum = if sum > E::zero() { sum } else { E::zero() };
    let a = (sum.scale_power_of_two(&half)).sqrt();
    let b = ((re.neg().add(&abs)).scale_power_of_two(&half))
        .sqrt()
        .scale_power_of_two(&im_sign);

    (a, b)
}

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

/// Utilities for split complex number types whose real and imaginary parts are stored separately.
pub mod complex_split {

    /// This structure contains the real and imaginary parts of an implicity conjugated value.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[repr(C)]
    pub struct ComplexConj<T> {
        pub re: T,
        pub neg_im: T,
    }

    /// This structure contains a pair of iterators that allow simultaneous iteration over the real
    /// and imaginary parts of a collection of complex values.
    #[derive(Clone, Debug)]
    pub struct ComplexIter<I> {
        pub(crate) re: I,
        pub(crate) im: I,
    }

    /// This structure contains a pair of iterators that allow simultaneous iteration over the real
    /// and imaginary parts of a collection of implicitly conjugated complex values.
    #[derive(Clone, Debug)]
    pub struct ComplexConjIter<I> {
        pub(crate) re: I,
        pub(crate) neg_im: I,
    }
}

pub use complex_native::*;
use complex_split::*;

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
        <Self as ComplexField>::neg(&self)
    }
}
impl core::ops::Sub for c32 {
    type Output = c32;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        <Self as ComplexField>::sub(&self, &rhs)
    }
}
impl core::ops::Add for c32 {
    type Output = c32;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        <Self as ComplexField>::add(&self, &rhs)
    }
}

impl core::ops::Neg for c64 {
    type Output = c64;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        <Self as ComplexField>::neg(&self)
    }
}
impl core::ops::Sub for c64 {
    type Output = c64;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        <Self as ComplexField>::sub(&self, &rhs)
    }
}
impl core::ops::Add for c64 {
    type Output = c64;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        <Self as ComplexField>::add(&self, &rhs)
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

/// Unstable core trait for describing how a scalar value may be split up into individual
/// component.
///
/// For example, `f64` is treated as a single indivisible unit, but [`num_complex::Complex<f64>`]
/// is split up into its real and imaginary components, with each one being stored in a separate
/// container.
pub unsafe trait Entity: Clone + PartialEq + Send + Sync + Debug + 'static {
    type Unit: Clone + Send + Sync + Debug + 'static;
    type Index: Copy + Send + Sync + Debug + 'static;
    type SimdUnit<S: Simd>: Copy + Send + Sync + Debug + 'static;
    type SimdMask<S: Simd>: Copy + Send + Sync + Debug + 'static;
    type SimdIndex<S: Simd>: Copy + Send + Sync + Debug + 'static;

    type Group<T>;
    /// Must be the same as `Group<T>`.
    type GroupCopy<T: Copy>: Copy;
    /// Must be the same as `Group<T>`.
    type GroupThreadSafe<T: Send + Sync>: Send + Sync;
    type Iter<I: Iterator>: Iterator<Item = Self::Group<I::Item>>;

    const N_COMPONENTS: usize;
    const HAS_SIMD: bool;
    const UNIT: Self::GroupCopy<()>;

    fn from_units(group: Self::Group<Self::Unit>) -> Self;
    fn into_units(self) -> Self::Group<Self::Unit>;

    fn as_ref<T>(group: &Self::Group<T>) -> Self::Group<&T>;
    fn as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T>;
    fn map<T, U>(group: Self::Group<T>, f: impl FnMut(T) -> U) -> Self::Group<U>;
    fn zip<T, U>(first: Self::Group<T>, second: Self::Group<U>) -> Self::Group<(T, U)>;
    fn unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>);

    #[inline(always)]
    fn unzip2<T>(zipped: Self::Group<[T; 2]>) -> [Self::Group<T>; 2] {
        let (a, b) = Self::unzip(Self::map(
            zipped,
            #[inline(always)]
            |[a, b]| (a, b),
        ));
        [a, b]
    }

    #[inline(always)]
    fn unzip4<T>(zipped: Self::Group<[T; 4]>) -> [Self::Group<T>; 4] {
        let (ab, cd) = Self::unzip(Self::map(
            zipped,
            #[inline(always)]
            |[a, b, c, d]| ([a, b], [c, d]),
        ));
        let [a, b] = Self::unzip2(ab);
        let [c, d] = Self::unzip2(cd);
        [a, b, c, d]
    }

    #[inline(always)]
    fn unzip8<T>(zipped: Self::Group<[T; 8]>) -> [Self::Group<T>; 8] {
        let (abcd, efgh) = Self::unzip(Self::map(
            zipped,
            #[inline(always)]
            |[a, b, c, d, e, f, g, h]| ([a, b, c, d], [e, f, g, h]),
        ));
        let [a, b, c, d] = Self::unzip4(abcd);
        let [e, f, g, h] = Self::unzip4(efgh);
        [a, b, c, d, e, f, g, h]
    }

    #[inline(always)]
    fn as_arrays<const N: usize, T>(
        group: Self::Group<&[T]>,
    ) -> (Self::Group<&[[T; N]]>, Self::Group<&[T]>) {
        #[inline(always)]
        fn do_as_arrays<const N: usize, T>() -> impl Fn(&[T]) -> (&[[T; N]], &[T]) {
            #[inline(always)]
            |slice| pulp::as_arrays(slice)
        }
        Self::unzip(Self::map(group, do_as_arrays()))
    }

    #[inline(always)]
    fn as_arrays_mut<const N: usize, T>(
        group: Self::Group<&mut [T]>,
    ) -> (Self::Group<&mut [[T; N]]>, Self::Group<&mut [T]>) {
        #[inline(always)]
        fn do_as_arrays_mut<const N: usize, T>() -> impl Fn(&mut [T]) -> (&mut [[T; N]], &mut [T]) {
            #[inline(always)]
            |slice| pulp::as_arrays_mut(slice)
        }
        Self::unzip(Self::map(group, do_as_arrays_mut()))
    }

    #[inline(always)]
    fn deref<T: Clone>(group: Self::Group<&T>) -> Self::Group<T> {
        #[inline(always)]
        fn do_deref<T: Clone>() -> impl FnMut(&T) -> T {
            #[inline(always)]
            |group| group.clone()
        }
        Self::map(group, do_deref())
    }
    #[inline(always)]
    fn rb<'short, T: Reborrow<'short>>(value: Self::Group<&'short T>) -> Self::Group<T::Target> {
        Self::map(
            value,
            #[inline(always)]
            |value| value.rb(),
        )
    }

    #[inline(always)]
    fn rb_mut<'short, T: ReborrowMut<'short>>(
        value: Self::Group<&'short mut T>,
    ) -> Self::Group<T::Target> {
        Self::map(
            value,
            #[inline(always)]
            |value| value.rb_mut(),
        )
    }
    #[inline(always)]
    fn into_const<T: IntoConst>(value: Self::Group<T>) -> Self::Group<T::Target> {
        Self::map(
            value,
            #[inline(always)]
            |value| value.into_const(),
        )
    }

    fn map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: Self::Group<T>,
        f: impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, Self::Group<U>);

    fn into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter>;

    #[inline(always)]
    fn from_copy<T: Copy>(group: Self::GroupCopy<T>) -> Self::Group<T> {
        unsafe { transmute_unchecked(group) }
    }
    #[inline(always)]
    fn into_copy<T: Copy>(group: Self::Group<T>) -> Self::GroupCopy<T> {
        unsafe { transmute_unchecked(group) }
    }

    #[inline(always)]
    fn map_copy<T: Copy, U: Copy>(
        group: Self::GroupCopy<T>,
        f: impl FnMut(T) -> U,
    ) -> Self::GroupCopy<U> {
        Self::into_copy(Self::map(Self::from_copy(group), f))
    }

    #[inline(always)]
    fn copy<T: Copy>(group: &Self::Group<T>) -> Self::Group<T> {
        unsafe { core::mem::transmute_copy(group) }
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

/// Trait for types that may be implicitly conjugated.
pub unsafe trait Conjugate: Entity {
    /// Must have the same layout as `Self`, and `Conj::Unit` must have the same layout as `Unit`.
    type Conj: Entity + Conjugate<Conj = Self, Canonical = Self::Canonical>;
    /// Must have the same layout as `Self`, and `Canonical::Unit` must have the same layout as
    /// `Unit`.
    type Canonical: Entity + Conjugate;

    /// Performs the implicit conjugation operation on the given value, returning the canonical
    /// form.
    fn canonicalize(self) -> Self::Canonical;
}

type SimdGroup<E, S> = <E as Entity>::Group<<E as Entity>::SimdUnit<S>>;

/// Unstable trait containing the operations that a number type needs to implement.
pub trait ComplexField: Entity + Conjugate<Canonical = Self> {
    type Real: RealField;

    /// Converts `value` from `f64` to `Self`.  
    /// The conversion may be lossy when converting to a type with less precision.
    fn from_f64(value: f64) -> Self;

    /// Returns `self + rhs`.
    fn add(&self, rhs: &Self) -> Self;
    /// Returns `self - rhs`.
    fn sub(&self, rhs: &Self) -> Self;
    /// Returns `self * rhs`.
    fn mul(&self, rhs: &Self) -> Self;

    /// Returns an estimate of `lhs * rhs + acc`.
    #[inline(always)]
    fn mul_adde(lhs: &Self, rhs: &Self, acc: &Self) -> Self {
        acc.add(&lhs.mul(rhs))
    }
    /// Returns an estimate of `conjugate(lhs) * rhs + acc`.
    #[inline(always)]
    fn conj_mul_adde(lhs: &Self, rhs: &Self, acc: &Self) -> Self {
        acc.add(&lhs.conj().mul(rhs))
    }

    /// Returns `-self`.
    fn neg(&self) -> Self;
    /// Returns `1.0/self`.
    fn inv(&self) -> Self;
    /// Returns `conjugate(self)`.
    fn conj(&self) -> Self;
    /// Returns the square root of `self`.
    fn sqrt(&self) -> Self;

    /// Returns the input, scaled by `rhs`.
    fn scale_real(&self, rhs: &Self::Real) -> Self;

    /// Returns the input, scaled by `rhs`.
    fn scale_power_of_two(&self, rhs: &Self::Real) -> Self;

    /// Returns either the norm or squared norm of the number.
    ///
    /// An implementation may choose either, so long as it chooses consistently.
    fn score(&self) -> Self::Real;
    /// Returns the absolute value of `self`.
    fn abs(&self) -> Self::Real;
    /// Returns the squared absolute value of `self`.
    fn abs2(&self) -> Self::Real;

    /// Returns a NaN value.
    fn nan() -> Self;

    /// Returns true if `self` is a NaN value, or false otherwise.
    #[inline(always)]
    fn is_nan(&self) -> bool {
        #[allow(clippy::eq_op)]
        {
            self != self
        }
    }

    /// Returns a complex number whose real part is equal to `real`, and a zero imaginary part.
    fn from_real(real: Self::Real) -> Self;

    /// Returns the real part.
    fn real(&self) -> Self::Real;
    /// Returns the imaginary part.
    fn imag(&self) -> Self::Real;

    /// Returns `0.0`.
    fn zero() -> Self;
    /// Returns `1.0`.
    fn one() -> Self;

    fn slice_as_simd<S: Simd>(slice: &[Self::Unit]) -> (&[Self::SimdUnit<S>], &[Self::Unit]);
    fn slice_as_mut_simd<S: Simd>(
        slice: &mut [Self::Unit],
    ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]);

    fn partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S>;
    fn partial_store_unit<S: Simd>(simd: S, slice: &mut [Self::Unit], values: Self::SimdUnit<S>);
    fn partial_load_last_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S>;
    fn partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    );

    fn simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S>;

    #[inline(always)]
    fn partial_load<S: Simd>(simd: S, slice: Self::Group<&[Self::Unit]>) -> SimdGroup<Self, S> {
        Self::map(
            slice,
            #[inline(always)]
            |slice| Self::partial_load_unit(simd, slice),
        )
    }
    #[inline(always)]
    fn partial_store<S: Simd>(
        simd: S,
        slice: Self::Group<&mut [Self::Unit]>,
        values: SimdGroup<Self, S>,
    ) {
        Self::map(
            Self::zip(slice, values),
            #[inline(always)]
            |(slice, unit)| Self::partial_store_unit(simd, slice, unit),
        );
    }
    #[inline(always)]
    fn partial_load_last<S: Simd>(
        simd: S,
        slice: Self::Group<&[Self::Unit]>,
    ) -> SimdGroup<Self, S> {
        Self::map(
            slice,
            #[inline(always)]
            |slice| Self::partial_load_last_unit(simd, slice),
        )
    }
    #[inline(always)]
    fn partial_store_last<S: Simd>(
        simd: S,
        slice: Self::Group<&mut [Self::Unit]>,
        values: SimdGroup<Self, S>,
    ) {
        Self::map(
            Self::zip(slice, values),
            #[inline(always)]
            |(slice, unit)| Self::partial_store_last_unit(simd, slice, unit),
        );
    }
    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, value: Self) -> SimdGroup<Self, S> {
        Self::map(
            Self::into_units(value),
            #[inline(always)]
            |unit| Self::simd_splat_unit(simd, unit),
        )
    }

    fn simd_neg<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S>;
    fn simd_conj<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S>;

    fn simd_add<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S>;
    fn simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S>;

    fn simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S>;
    fn simd_scale_real<S: Simd>(
        simd: S,
        lhs: <Self::Real as Entity>::Group<<Self::Real as Entity>::SimdUnit<S>>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S>;
    fn simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S>;
    fn simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S>;
    fn simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S>;

    fn simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroup<Self, S>,
        acc: SimdGroup<Self::Real, S>,
    ) -> SimdGroup<Self::Real, S>;
    fn simd_abs2<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S>;
    fn simd_score<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S>;

    #[inline(always)]
    fn simd_reduce_add<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> Self {
        let _ = simd;
        let mut acc = Self::zero();

        let slice = simd::simd_as_slice::<Self, S>(Self::map(
            Self::as_ref(&values),
            #[inline(always)]
            |ptr| core::slice::from_ref(ptr),
        ));
        for units in Self::into_iter(slice) {
            let value = Self::from_units(Self::deref(units));
            acc = acc.add(&value);
        }

        acc
    }
}

/// Unstable trait containing the operations that a real number type needs to implement.
pub trait RealField: ComplexField<Real = Self> + PartialOrd {
    fn div(&self, rhs: &Self) -> Self;

    fn usize_to_index(a: usize) -> Self::Index;
    fn index_to_usize(a: Self::Index) -> usize;
    fn max_index() -> Self::Index;

    fn simd_less_than<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S>;
    fn simd_less_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S>;
    fn simd_greater_than<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S>;
    fn simd_greater_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S>;

    fn simd_select<S: Simd>(
        simd: S,
        mask: Self::SimdMask<S>,
        if_true: SimdGroup<Self, S>,
        if_false: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S>;
    fn simd_index_select<S: Simd>(
        simd: S,
        mask: Self::SimdMask<S>,
        if_true: Self::SimdIndex<S>,
        if_false: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S>;
    fn simd_index_seq<S: Simd>(simd: S) -> Self::SimdIndex<S>;
    fn simd_index_splat<S: Simd>(simd: S, value: Self::Index) -> Self::SimdIndex<S>;
    fn simd_index_add<S: Simd>(
        simd: S,
        a: Self::SimdIndex<S>,
        b: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S>;
}

impl ComplexField for f32 {
    type Real = Self;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value as _
    }

    #[inline(always)]
    fn add(&self, rhs: &Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn sub(&self, rhs: &Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn mul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        -self
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        self.recip()
    }

    #[inline(always)]
    fn conj(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        (*self).sqrt()
    }

    #[inline(always)]
    fn scale_real(&self, rhs: &Self::Real) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn scale_power_of_two(&self, rhs: &Self::Real) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn score(&self) -> Self::Real {
        (*self).abs()
    }

    #[inline(always)]
    fn abs(&self) -> Self::Real {
        (*self).abs()
    }

    #[inline(always)]
    fn abs2(&self) -> Self::Real {
        self * self
    }

    #[inline(always)]
    fn nan() -> Self {
        Self::NAN
    }

    #[inline(always)]
    fn from_real(real: Self::Real) -> Self {
        real
    }

    #[inline(always)]
    fn real(&self) -> Self::Real {
        *self
    }

    #[inline(always)]
    fn imag(&self) -> Self::Real {
        0.0
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
    fn slice_as_simd<S: Simd>(slice: &[Self::Unit]) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
        S::f32s_as_simd(slice)
    }

    #[inline(always)]
    fn slice_as_mut_simd<S: Simd>(
        slice: &mut [Self::Unit],
    ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
        S::f32s_as_mut_simd(slice)
    }

    #[inline(always)]
    fn partial_load_last_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.f32s_partial_load_last(slice)
    }

    #[inline(always)]
    fn partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.f32s_partial_store_last(slice, values)
    }

    #[inline(always)]
    fn partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.f32s_partial_load(slice)
    }

    #[inline(always)]
    fn partial_store_unit<S: Simd>(simd: S, slice: &mut [Self::Unit], values: Self::SimdUnit<S>) {
        simd.f32s_partial_store(slice, values)
    }

    #[inline(always)]
    fn simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
        simd.f32s_splat(unit)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
        simd.f32s_neg(values)
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
        simd.f32s_add(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.f32s_sub(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.f32s_mul(lhs, rhs)
    }
    #[inline(always)]
    fn simd_scale_real<S: Simd>(
        simd: S,
        lhs: <Self::Real as Entity>::Group<<Self::Real as Entity>::SimdUnit<S>>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Self::simd_mul(simd, lhs, rhs)
    }
    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.f32s_mul(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.f32s_mul_adde(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.f32s_mul_adde(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_reduce_add<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> Self {
        simd.f32s_reduce_sum(values)
    }

    #[inline(always)]
    fn simd_abs2<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
        simd.f32s_mul(values, values)
    }
    #[inline(always)]
    fn simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroup<Self, S>,
        acc: SimdGroup<Self::Real, S>,
    ) -> SimdGroup<Self::Real, S> {
        simd.f32s_mul_adde(values, values, acc)
    }
    #[inline(always)]
    fn simd_score<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
        simd.f32s_abs(values)
    }
}
impl ComplexField for f64 {
    type Real = Self;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value
    }

    #[inline(always)]
    fn add(&self, rhs: &Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn sub(&self, rhs: &Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn mul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        -self
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        self.recip()
    }

    #[inline(always)]
    fn conj(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        (*self).sqrt()
    }

    #[inline(always)]
    fn scale_real(&self, rhs: &Self::Real) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn scale_power_of_two(&self, rhs: &Self::Real) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn score(&self) -> Self::Real {
        (*self).abs()
    }

    #[inline(always)]
    fn abs(&self) -> Self::Real {
        (*self).abs()
    }

    #[inline(always)]
    fn abs2(&self) -> Self::Real {
        self * self
    }

    #[inline(always)]
    fn nan() -> Self {
        Self::NAN
    }

    #[inline(always)]
    fn from_real(real: Self::Real) -> Self {
        real
    }

    #[inline(always)]
    fn real(&self) -> Self::Real {
        *self
    }

    #[inline(always)]
    fn imag(&self) -> Self::Real {
        0.0
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
    fn slice_as_simd<S: Simd>(slice: &[Self::Unit]) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
        S::f64s_as_simd(slice)
    }

    #[inline(always)]
    fn slice_as_mut_simd<S: Simd>(
        slice: &mut [Self::Unit],
    ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
        S::f64s_as_mut_simd(slice)
    }

    #[inline(always)]
    fn partial_load_last_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.f64s_partial_load_last(slice)
    }

    #[inline(always)]
    fn partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.f64s_partial_store_last(slice, values)
    }
    #[inline(always)]
    fn partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.f64s_partial_load(slice)
    }

    #[inline(always)]
    fn partial_store_unit<S: Simd>(simd: S, slice: &mut [Self::Unit], values: Self::SimdUnit<S>) {
        simd.f64s_partial_store(slice, values)
    }

    #[inline(always)]
    fn simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
        simd.f64s_splat(unit)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
        simd.f64s_neg(values)
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
        simd.f64s_add(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.f64s_sub(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.f64s_mul(lhs, rhs)
    }
    #[inline(always)]
    fn simd_scale_real<S: Simd>(
        simd: S,
        lhs: <Self::Real as Entity>::Group<<Self::Real as Entity>::SimdUnit<S>>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Self::simd_mul(simd, lhs, rhs)
    }
    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.f64s_mul(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.f64s_mul_adde(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.f64s_mul_adde(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_reduce_add<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> Self {
        simd.f64s_reduce_sum(values)
    }

    #[inline(always)]
    fn simd_abs2<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
        simd.f64s_mul(values, values)
    }
    #[inline(always)]
    fn simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroup<Self, S>,
        acc: SimdGroup<Self::Real, S>,
    ) -> SimdGroup<Self::Real, S> {
        simd.f64s_mul_adde(values, values, acc)
    }
    #[inline(always)]
    fn simd_score<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
        simd.f64s_abs(values)
    }
}
impl RealField for f32 {
    #[inline(always)]
    fn div(&self, rhs: &Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn usize_to_index(a: usize) -> Self::Index {
        a as _
    }
    #[inline(always)]
    fn index_to_usize(a: Self::Index) -> usize {
        a as _
    }
    #[inline(always)]
    fn max_index() -> Self::Index {
        Self::Index::MAX
    }

    #[inline(always)]
    fn simd_less_than<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        simd.f32s_less_than(a, b)
    }

    #[inline(always)]
    fn simd_less_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        simd.f32s_less_than_or_equal(a, b)
    }

    #[inline(always)]
    fn simd_greater_than<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        simd.f32s_greater_than(a, b)
    }

    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        simd.f32s_greater_than_or_equal(a, b)
    }

    #[inline(always)]
    fn simd_select<S: Simd>(
        simd: S,
        mask: Self::SimdMask<S>,
        if_true: SimdGroup<Self, S>,
        if_false: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.m32s_select_f32s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn simd_index_select<S: Simd>(
        simd: S,
        mask: Self::SimdMask<S>,
        if_true: Self::SimdIndex<S>,
        if_false: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        simd.m32s_select_u32s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn simd_index_seq<S: Simd>(simd: S) -> Self::SimdIndex<S> {
        let _ = simd;
        pulp::cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u32])
    }

    #[inline(always)]
    fn simd_index_splat<S: Simd>(simd: S, value: Self::Index) -> Self::SimdIndex<S> {
        simd.u32s_splat(value)
    }

    #[inline(always)]
    fn simd_index_add<S: Simd>(
        simd: S,
        a: Self::SimdIndex<S>,
        b: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        simd.u32s_add(a, b)
    }
}
impl RealField for f64 {
    #[inline(always)]
    fn div(&self, rhs: &Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn usize_to_index(a: usize) -> Self::Index {
        a as _
    }
    #[inline(always)]
    fn index_to_usize(a: Self::Index) -> usize {
        a as _
    }
    #[inline(always)]
    fn max_index() -> Self::Index {
        Self::Index::MAX
    }

    #[inline(always)]
    fn simd_less_than<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        simd.f64s_less_than(a, b)
    }

    #[inline(always)]
    fn simd_less_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        simd.f64s_less_than_or_equal(a, b)
    }

    #[inline(always)]
    fn simd_greater_than<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        simd.f64s_greater_than(a, b)
    }

    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroup<Self, S>,
        b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        simd.f64s_greater_than_or_equal(a, b)
    }

    #[inline(always)]
    fn simd_select<S: Simd>(
        simd: S,
        mask: Self::SimdMask<S>,
        if_true: SimdGroup<Self, S>,
        if_false: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        simd.m64s_select_f64s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn simd_index_select<S: Simd>(
        simd: S,
        mask: Self::SimdMask<S>,
        if_true: Self::SimdIndex<S>,
        if_false: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        simd.m64s_select_u64s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn simd_index_seq<S: Simd>(simd: S) -> Self::SimdIndex<S> {
        let _ = simd;
        pulp::cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u64])
    }

    #[inline(always)]
    fn simd_index_splat<S: Simd>(simd: S, value: Self::Index) -> Self::SimdIndex<S> {
        simd.u64s_splat(value)
    }

    #[inline(always)]
    fn simd_index_add<S: Simd>(
        simd: S,
        a: Self::SimdIndex<S>,
        b: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        simd.u64s_add(a, b)
    }
}

impl ComplexField for c32 {
    type Real = f32;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        Self {
            re: value as _,
            im: 0.0,
        }
    }

    #[inline(always)]
    fn add(&self, rhs: &Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline(always)]
    fn sub(&self, rhs: &Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    #[inline(always)]
    fn mul(&self, rhs: &Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        let inf = Self::Real::zero().inv();
        if self.is_nan() {
            // NAN
            Self::nan()
        } else if *self == Self::zero() {
            // zero
            Self { re: inf, im: inf }
        } else if self.re == inf || self.im == inf {
            Self::zero()
        } else {
            let re = self.real().abs();
            let im = self.imag().abs();
            let max = if re > im { re } else { im };
            let max_inv = max.inv();
            let x = self.scale_real(&max_inv);
            x.conj().scale_real(&x.abs2().inv().mul(&max_inv))
        }
    }

    #[inline(always)]
    fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        let this: num_complex::Complex32 = (*self).into();
        num_complex::ComplexFloat::sqrt(this).into()
    }

    #[inline(always)]
    fn scale_real(&self, rhs: &Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn scale_power_of_two(&self, rhs: &Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn score(&self) -> Self::Real {
        self.abs2()
    }

    #[inline(always)]
    fn abs(&self) -> Self::Real {
        self.abs2().sqrt()
    }

    #[inline(always)]
    fn abs2(&self) -> Self::Real {
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
    fn real(&self) -> Self::Real {
        self.re
    }

    #[inline(always)]
    fn imag(&self) -> Self::Real {
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

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        Self {
            re: value as _,
            im: 0.0,
        }
    }

    #[inline(always)]
    fn add(&self, rhs: &Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline(always)]
    fn sub(&self, rhs: &Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    #[inline(always)]
    fn mul(&self, rhs: &Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        let inf = Self::Real::zero().inv();
        if self.is_nan() {
            // NAN
            Self::nan()
        } else if *self == Self::zero() {
            // zero
            Self { re: inf, im: inf }
        } else if self.re == inf || self.im == inf {
            Self::zero()
        } else {
            let re = self.real().abs();
            let im = self.imag().abs();
            let max = if re > im { re } else { im };
            let max_inv = max.inv();
            let x = self.scale_real(&max_inv);
            x.conj().scale_real(&x.abs2().inv().mul(&max_inv))
        }
    }

    #[inline(always)]
    fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        let this: num_complex::Complex64 = (*self).into();
        num_complex::ComplexFloat::sqrt(this).into()
    }

    #[inline(always)]
    fn scale_real(&self, rhs: &Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn scale_power_of_two(&self, rhs: &Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn score(&self) -> Self::Real {
        self.abs2()
    }

    #[inline(always)]
    fn abs(&self) -> Self::Real {
        self.abs2().sqrt()
    }

    #[inline(always)]
    fn abs2(&self) -> Self::Real {
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
    fn real(&self) -> Self::Real {
        self.re
    }

    #[inline(always)]
    fn imag(&self) -> Self::Real {
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

impl<E: RealField> ComplexField for Complex<E> {
    type Real = E;

    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        Self {
            re: Self::Real::from_f64(value),
            im: Self::Real::zero(),
        }
    }

    #[inline(always)]
    fn add(&self, rhs: &Self) -> Self {
        Self {
            re: self.re.add(&rhs.re),
            im: self.im.add(&rhs.im),
        }
    }

    #[inline(always)]
    fn sub(&self, rhs: &Self) -> Self {
        Self {
            re: self.re.sub(&rhs.re),
            im: self.im.sub(&rhs.im),
        }
    }

    #[inline(always)]
    fn mul(&self, rhs: &Self) -> Self {
        Self {
            re: Self::Real::sub(&self.re.mul(&rhs.re), &self.im.mul(&rhs.im)),
            im: Self::Real::add(&self.re.mul(&rhs.im), &self.im.mul(&rhs.re)),
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        Self {
            re: self.re.neg(),
            im: self.im.neg(),
        }
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        let inf = Self::Real::zero().inv();
        if self.is_nan() {
            // NAN
            Self::nan()
        } else if *self == Self::zero() {
            // zero
            Self {
                re: inf.clone(),
                im: inf,
            }
        } else if self.re == inf || self.im == inf {
            Self::zero()
        } else {
            let re = self.real().abs();
            let im = self.imag().abs();
            let max = if re > im { re } else { im };
            let max_inv = max.inv();
            let x = self.scale_real(&max_inv);
            x.conj().scale_real(&x.abs2().inv().mul(&max_inv))
        }
    }

    #[inline(always)]
    fn conj(&self) -> Self {
        Self {
            re: self.re.clone(),
            im: self.im.neg(),
        }
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        let (re, im) = sqrt_impl(self.re.clone(), self.im.clone());
        Self { re, im }
    }

    #[inline(always)]
    fn scale_real(&self, rhs: &Self::Real) -> Self {
        Self {
            re: self.re.scale_real(rhs),
            im: self.im.scale_real(rhs),
        }
    }

    #[inline(always)]
    fn scale_power_of_two(&self, rhs: &Self::Real) -> Self {
        Self {
            re: self.re.scale_power_of_two(rhs),
            im: self.im.scale_power_of_two(rhs),
        }
    }

    #[inline(always)]
    fn score(&self) -> Self::Real {
        self.abs2()
    }

    #[inline(always)]
    fn abs(&self) -> Self::Real {
        self.abs2().sqrt()
    }

    #[inline(always)]
    fn abs2(&self) -> Self::Real {
        Self::Real::add(&self.re.mul(&self.re), &self.im.mul(&self.im))
    }

    #[inline(always)]
    fn nan() -> Self {
        Self {
            re: Self::Real::nan(),
            im: Self::Real::nan(),
        }
    }

    #[inline(always)]
    fn from_real(real: Self::Real) -> Self {
        Self {
            re: real,
            im: Self::Real::zero(),
        }
    }

    #[inline(always)]
    fn real(&self) -> Self::Real {
        self.re.clone()
    }

    #[inline(always)]
    fn imag(&self) -> Self::Real {
        self.im.clone()
    }

    #[inline(always)]
    fn zero() -> Self {
        Self {
            re: Self::Real::zero(),
            im: Self::Real::zero(),
        }
    }

    #[inline(always)]
    fn one() -> Self {
        Self {
            re: Self::Real::one(),
            im: Self::Real::zero(),
        }
    }

    #[inline(always)]
    fn slice_as_simd<S: Simd>(slice: &[Self::Unit]) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
        E::slice_as_simd(slice)
    }

    #[inline(always)]
    fn slice_as_mut_simd<S: Simd>(
        slice: &mut [Self::Unit],
    ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
        E::slice_as_mut_simd(slice)
    }

    #[inline(always)]
    fn partial_load_last_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        E::partial_load_last_unit(simd, slice)
    }

    #[inline(always)]
    fn partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        E::partial_store_last_unit(simd, slice, values)
    }

    #[inline(always)]
    fn partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        E::partial_load_unit(simd, slice)
    }

    #[inline(always)]
    fn partial_store_unit<S: Simd>(simd: S, slice: &mut [Self::Unit], values: Self::SimdUnit<S>) {
        E::partial_store_unit(simd, slice, values)
    }

    #[inline(always)]
    fn simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
        E::simd_splat_unit(simd, unit)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
        Complex {
            re: E::simd_neg(simd, values.re),
            im: E::simd_neg(simd, values.im),
        }
    }

    #[inline(always)]
    fn simd_conj<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
        Complex {
            re: values.re,
            im: E::simd_neg(simd, values.im),
        }
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Complex {
            re: E::simd_add(simd, lhs.re, rhs.re),
            im: E::simd_add(simd, lhs.im, rhs.im),
        }
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Complex {
            re: E::simd_sub(simd, lhs.re, rhs.re),
            im: E::simd_sub(simd, lhs.im, rhs.im),
        }
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Complex {
            re: E::simd_mul_adde(
                simd,
                E::copy(&lhs.re),
                E::copy(&rhs.re),
                E::simd_mul(simd, E::simd_neg(simd, E::copy(&lhs.im)), E::copy(&rhs.im)),
            ),
            im: E::simd_mul_adde(simd, lhs.re, rhs.im, E::simd_mul(simd, lhs.im, rhs.re)),
        }
    }

    #[inline(always)]
    fn simd_scale_real<S: Simd>(
        simd: S,
        lhs: <Self::Real as Entity>::Group<<Self::Real as Entity>::SimdUnit<S>>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Complex {
            re: E::simd_mul(simd, E::copy(&lhs), rhs.re),
            im: E::simd_mul(simd, lhs, rhs.im),
        }
    }

    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Complex {
            re: E::simd_mul_adde(
                simd,
                E::copy(&lhs.re),
                E::copy(&rhs.re),
                E::simd_mul(simd, E::copy(&lhs.im), E::copy(&rhs.im)),
            ),
            im: E::simd_mul_adde(
                simd,
                lhs.re,
                rhs.im,
                E::simd_mul(simd, E::simd_neg(simd, lhs.im), rhs.re),
            ),
        }
    }

    #[inline(always)]
    fn simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Complex {
            re: E::simd_mul_adde(
                simd,
                E::copy(&lhs.re),
                E::copy(&rhs.re),
                E::simd_mul_adde(
                    simd,
                    E::simd_neg(simd, E::copy(&lhs.im)),
                    E::copy(&rhs.im),
                    acc.re,
                ),
            ),
            im: E::simd_mul_adde(
                simd,
                lhs.re,
                rhs.im,
                E::simd_mul_adde(simd, lhs.im, rhs.re, acc.im),
            ),
        }
    }

    #[inline(always)]
    fn simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroup<Self, S>,
        rhs: SimdGroup<Self, S>,
        acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Complex {
            re: E::simd_mul_adde(
                simd,
                E::copy(&lhs.re),
                E::copy(&rhs.re),
                E::simd_mul_adde(simd, E::copy(&lhs.im), E::copy(&rhs.im), acc.re),
            ),
            im: E::simd_mul_adde(
                simd,
                lhs.re,
                rhs.im,
                E::simd_mul_adde(simd, E::simd_neg(simd, lhs.im), rhs.re, acc.im),
            ),
        }
    }

    #[inline(always)]
    fn simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroup<Self, S>,
        acc: SimdGroup<Self::Real, S>,
    ) -> SimdGroup<Self::Real, S> {
        E::simd_mul_adde(
            simd,
            E::copy(&values.re),
            E::copy(&values.re),
            E::simd_mul_adde(simd, E::copy(&values.im), E::copy(&values.im), acc),
        )
    }
    #[inline(always)]
    fn simd_abs2<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
        Self::simd_score(simd, values)
    }
    #[inline(always)]
    fn simd_score<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
        E::simd_mul_adde(
            simd,
            E::copy(&values.re),
            E::copy(&values.re),
            E::simd_mul(simd, E::copy(&values.im), E::copy(&values.im)),
        )
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
    type GroupThreadSafe<T: Send + Sync> = T;
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
    type GroupThreadSafe<T: Send + Sync> = T;
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
    type GroupThreadSafe<T: Send + Sync> = T;
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
    type GroupThreadSafe<T: Send + Sync> = T;
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

unsafe impl Entity for f32 {
    type Unit = Self;
    type Index = u32;
    type SimdUnit<S: Simd> = S::f32s;
    type SimdMask<S: Simd> = S::m32s;
    type SimdIndex<S: Simd> = S::u32s;
    type Group<T> = T;
    type GroupCopy<T: Copy> = T;
    type GroupThreadSafe<T: Send + Sync> = T;
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

unsafe impl Entity for f64 {
    type Unit = Self;
    type Index = u64;
    type SimdUnit<S: Simd> = S::f64s;
    type SimdMask<S: Simd> = S::m64s;
    type SimdIndex<S: Simd> = S::u64s;
    type Group<T> = T;
    type GroupCopy<T: Copy> = T;
    type GroupThreadSafe<T: Send + Sync> = T;
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

impl<I: Iterator> Iterator for ComplexIter<I> {
    type Item = Complex<I::Item>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match (self.re.next(), self.im.next()) {
            (None, None) => None,
            (Some(re), Some(im)) => Some(Complex { re, im }),
            _ => panic!(),
        }
    }
}
impl<I: Iterator> Iterator for ComplexConjIter<I> {
    type Item = ComplexConj<I::Item>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match (self.re.next(), self.neg_im.next()) {
            (None, None) => None,
            (Some(re), Some(neg_im)) => Some(ComplexConj { re, neg_im }),
            _ => panic!(),
        }
    }
}

unsafe impl<E: Entity> Entity for Complex<E> {
    type Unit = E::Unit;
    type Index = E::Index;
    type SimdUnit<S: Simd> = E::SimdUnit<S>;
    type SimdMask<S: Simd> = E::SimdMask<S>;
    type SimdIndex<S: Simd> = E::SimdIndex<S>;
    type Group<T> = Complex<E::Group<T>>;
    type GroupCopy<T: Copy> = Complex<E::GroupCopy<T>>;
    type GroupThreadSafe<T: Send + Sync> = Complex<E::GroupThreadSafe<T>>;
    type Iter<I: Iterator> = ComplexIter<E::Iter<I>>;

    const N_COMPONENTS: usize = E::N_COMPONENTS * 2;
    const HAS_SIMD: bool = E::HAS_SIMD;
    const UNIT: Self::GroupCopy<()> = Complex {
        re: E::UNIT,
        im: E::UNIT,
    };

    #[inline(always)]
    fn from_units(group: Self::Group<Self::Unit>) -> Self {
        let re = E::from_units(group.re);
        let im = E::from_units(group.im);
        Self { re, im }
    }

    #[inline(always)]
    fn into_units(self) -> Self::Group<Self::Unit> {
        let Self { re, im } = self;
        Complex {
            re: re.into_units(),
            im: im.into_units(),
        }
    }

    #[inline(always)]
    fn as_ref<T>(group: &Self::Group<T>) -> Self::Group<&T> {
        Complex {
            re: E::as_ref(&group.re),
            im: E::as_ref(&group.im),
        }
    }

    #[inline(always)]
    fn as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T> {
        Complex {
            re: E::as_mut(&mut group.re),
            im: E::as_mut(&mut group.im),
        }
    }

    #[inline(always)]
    fn map<T, U>(group: Self::Group<T>, f: impl FnMut(T) -> U) -> Self::Group<U> {
        let mut f = f;
        Complex {
            re: E::map(group.re, &mut f),
            im: E::map(group.im, &mut f),
        }
    }
    #[inline(always)]
    fn map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: Self::Group<T>,
        f: impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, Self::Group<U>) {
        let mut f = f;
        let (ctx, re) = E::map_with_context(ctx, group.re, &mut f);
        let (ctx, im) = E::map_with_context(ctx, group.im, &mut f);
        (ctx, Complex { re, im })
    }

    #[inline(always)]
    fn zip<T, U>(first: Self::Group<T>, second: Self::Group<U>) -> Self::Group<(T, U)> {
        Complex {
            re: E::zip(first.re, second.re),
            im: E::zip(first.im, second.im),
        }
    }
    #[inline(always)]
    fn unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>) {
        let (re0, re1) = E::unzip(zipped.re);
        let (im0, im1) = E::unzip(zipped.im);
        (Complex { re: re0, im: im0 }, Complex { re: re1, im: im1 })
    }

    #[inline(always)]
    fn into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter> {
        ComplexIter {
            re: E::into_iter(iter.re),
            im: E::into_iter(iter.im),
        }
    }
}

unsafe impl<E: Entity> Entity for ComplexConj<E> {
    type Unit = E::Unit;
    type Index = E::Index;
    type SimdUnit<S: Simd> = E::SimdUnit<S>;
    type SimdMask<S: Simd> = E::SimdMask<S>;
    type SimdIndex<S: Simd> = E::SimdIndex<S>;
    type Group<T> = ComplexConj<E::Group<T>>;
    type GroupCopy<T: Copy> = ComplexConj<E::GroupCopy<T>>;
    type GroupThreadSafe<T: Send + Sync> = ComplexConj<E::GroupThreadSafe<T>>;
    type Iter<I: Iterator> = ComplexConjIter<E::Iter<I>>;

    const N_COMPONENTS: usize = E::N_COMPONENTS * 2;
    const HAS_SIMD: bool = E::HAS_SIMD;
    const UNIT: Self::GroupCopy<()> = ComplexConj {
        re: E::UNIT,
        neg_im: E::UNIT,
    };

    #[inline(always)]
    fn from_units(group: Self::Group<Self::Unit>) -> Self {
        let re = E::from_units(group.re);
        let neg_im = E::from_units(group.neg_im);
        Self { re, neg_im }
    }

    #[inline(always)]
    fn into_units(self) -> Self::Group<Self::Unit> {
        let Self { re, neg_im } = self;
        ComplexConj {
            re: re.into_units(),
            neg_im: neg_im.into_units(),
        }
    }

    #[inline(always)]
    fn as_ref<T>(group: &Self::Group<T>) -> Self::Group<&T> {
        ComplexConj {
            re: E::as_ref(&group.re),
            neg_im: E::as_ref(&group.neg_im),
        }
    }

    #[inline(always)]
    fn as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T> {
        ComplexConj {
            re: E::as_mut(&mut group.re),
            neg_im: E::as_mut(&mut group.neg_im),
        }
    }

    #[inline(always)]
    fn map<T, U>(group: Self::Group<T>, f: impl FnMut(T) -> U) -> Self::Group<U> {
        let mut f = f;
        ComplexConj {
            re: E::map(group.re, &mut f),
            neg_im: E::map(group.neg_im, &mut f),
        }
    }
    #[inline(always)]
    fn map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: Self::Group<T>,
        f: impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, Self::Group<U>) {
        let mut f = f;
        let (ctx, re) = E::map_with_context(ctx, group.re, &mut f);
        let (ctx, neg_im) = E::map_with_context(ctx, group.neg_im, &mut f);
        (ctx, ComplexConj { re, neg_im })
    }

    #[inline(always)]
    fn zip<T, U>(first: Self::Group<T>, second: Self::Group<U>) -> Self::Group<(T, U)> {
        ComplexConj {
            re: E::zip(first.re, second.re),
            neg_im: E::zip(first.neg_im, second.neg_im),
        }
    }
    #[inline(always)]
    fn unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>) {
        let (re0, re1) = E::unzip(zipped.re);
        let (neg_im0, neg_im1) = E::unzip(zipped.neg_im);
        (
            ComplexConj {
                re: re0,
                neg_im: neg_im0,
            },
            ComplexConj {
                re: re1,
                neg_im: neg_im1,
            },
        )
    }

    #[inline(always)]
    fn into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter> {
        ComplexConjIter {
            re: E::into_iter(iter.re),
            neg_im: E::into_iter(iter.neg_im),
        }
    }
}

unsafe impl Conjugate for f32 {
    type Conj = f32;
    type Canonical = f32;

    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        self
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

unsafe impl Conjugate for f64 {
    type Conj = f64;
    type Canonical = f64;

    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        self
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

unsafe impl<E: Entity + ComplexField> Conjugate for Complex<E> {
    type Conj = ComplexConj<E>;
    type Canonical = Complex<E>;

    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        self
    }
}

unsafe impl<E: Entity + ComplexField> Conjugate for ComplexConj<E> {
    type Conj = Complex<E>;
    type Canonical = Complex<E>;

    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        Complex {
            re: self.re,
            im: self.neg_im.neg(),
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
/// # Note:
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
/// # Note:
///
/// Unlike a slice, the data pointed to by `MatMut<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions ([`std::mem::needs_drop::<E::Unit>()`] must be false). In
/// this case, care must be taken to not perform any operations that read the uninitialized values,
/// or form references to them, either directly through [`MatMut::read`], or indirectly through any
/// of the numerical library routines, unless it is explicitly permitted.
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

impl<'a, E: Entity> MatRef<'a, E> {
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
        E::map(E::as_ref(&ptr), |ptr| debug_assert!(!ptr.is_null()));
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

    /// Returns references to the element at the given indices.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn get_unchecked(self, row: usize, col: usize) -> E::Group<&'a E::Unit> {
        E::map(self.ptr_inbounds_at(row, col), |ptr| &*ptr)
    }

    /// Returns references to the element at the given indices, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn get(self, row: usize, col: usize) -> E::Group<&'a E::Unit> {
        assert!(row < self.nrows());
        assert!(col < self.ncols());
        unsafe { self.get_unchecked(row, col) }
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
        E::from_units(E::map(self.get_unchecked(row, col), |ptr| (*ptr).clone()))
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
        E::from_units(E::map(self.get(row, col), |ptr| (*ptr).clone()))
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
}

impl<'a, E: Entity> MatMut<'a, E> {
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
        E::map(E::as_ref(&ptr), |ptr| debug_assert!(!ptr.is_null()));
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

    /// Returns mutable references to the element at the given indices.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn get_unchecked(self, row: usize, col: usize) -> E::Group<&'a mut E::Unit> {
        E::map(self.ptr_inbounds_at(row, col), |ptr| &mut *ptr)
    }

    /// Returns mutable references to the element at the given indices, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn get(self, row: usize, col: usize) -> E::Group<&'a mut E::Unit> {
        assert!(row < self.nrows());
        assert!(col < self.ncols());
        unsafe { self.get_unchecked(row, col) }
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
    pub fn clone_from(&mut self, other: MatRef<'_, E>) {
        zipped!(self.rb_mut(), other).for_each(|mut dst, src| dst.write(src.read()));
    }

    /// Fills the elements of `self` with zeros.
    #[inline(always)]
    #[track_caller]
    pub fn set_zeros(&mut self)
    where
        E: ComplexField,
    {
        zipped!(self.rb_mut()).for_each(|mut x| x.write(E::zero()));
    }

    /// Fills the elements of `self` with copies of `constant`.
    #[inline(always)]
    #[track_caller]
    pub fn set_constant(&mut self, constant: E) {
        zipped!(self.rb_mut()).for_each(|mut x| x.write(constant.clone()));
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
        aligned_vec::CACHELINE_ALIGN
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

        let this = Self {
            ptr: E::into_copy(E::map(group, |mat| mat.ptr)),
            row_capacity,
            col_capacity,
        };

        this
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
            Self::with_dims(self.nrows, self.ncols, |i, j| {
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
    pub fn with_dims(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> E) -> Self {
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
        Self::with_dims(nrows, ncols, |_, _| E::zero())
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
                $crate::Mat::<_>::with_dims(nrows, ncols, |i, j| $crate::ref_to_ptr(&data[j][i]).read())
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
    Uninit(DynArray<'a, MaybeUninit<T>>),
}

/// A temporary matrix allocated from a [`DynStack`].
///
/// [`DynStack`]: dyn_stack::DynStack
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
                    DynMatUnitImpl::Uninit(uninit) => uninit.as_ptr() as *const E::Unit,
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
                    DynMatUnitImpl::Uninit(uninit) => uninit.as_mut_ptr() as *mut E::Unit,
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
    stack: DynStack<'_>,
) -> (DynMat<'_, E>, DynStack<'_>) {
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
            stack.make_aligned_with(ncols * col_stride, align_for::<E::Unit>(), |_| {
                value.clone()
            });
        (stack, alloc)
    });

    (
        DynMat {
            inner: E::map(alloc, |alloc| DynMatUnitImpl::Init(alloc)),
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
    stack: DynStack<'_>,
) -> (DynMat<'_, E>, DynStack<'_>) {
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
                |_| value.clone(),
            );
            (stack, alloc)
        },
    );

    (
        DynMat {
            inner: E::map(alloc, |alloc| DynMatUnitImpl::Init(alloc)),
            nrows,
            ncols,
            col_stride,
        },
        stack,
    )
}

/// Creates a temporary matrix of zero values, from the given memory stack.
pub unsafe fn temp_mat_uninit<E: ComplexField>(
    nrows: usize,
    ncols: usize,
    stack: DynStack<'_>,
) -> (DynMat<'_, E>, DynStack<'_>) {
    if core::mem::needs_drop::<E::Unit>() || (cfg!(debug_assertions) && !cfg!(miri)) {
        temp_mat_constant(nrows, ncols, E::nan(), stack)
    } else {
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
                let (alloc, stack) = stack
                    .make_aligned_uninit::<E::Unit>(ncols * col_stride, align_for::<E::Unit>());
                (stack, alloc)
            },
        );
        (
            DynMat {
                inner: E::map(alloc, |alloc| DynMatUnitImpl::Uninit(alloc)),
                nrows,
                ncols,
                col_stride,
            },
            stack,
        )
    }
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

    fn mul(self, rhs: MatRef<'_, RhsE>) -> Self::Output {
        let mut out = Mat::zeros(self.nrows(), rhs.ncols());
        mul::matmul(
            out.as_mut(),
            self,
            rhs,
            None,
            LhsE::Canonical::one(),
            Parallelism::Rayon(0),
        );
        out
    }
}

impl<T: ComplexField> core::ops::MulAssign<T> for MatMut<'_, T> {
    fn mul_assign(&mut self, rhs: T) {
        self.rb_mut().cwise().for_each(|mut x| {
            let val = x.read();
            x.write(val.mul(&rhs));
        });
    }
}

impl<T: ComplexField> core::ops::MulAssign<T> for Mat<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.as_mut().mul_assign(rhs);
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
                type GroupThreadSafe<T: Send + Sync> = T;
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
    use assert_approx_eq::assert_approx_eq;

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

    #[derive(Debug, PartialEq, Clone)]
    struct ZST;

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
        let x = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        assert!(x.read(0, 0) == 1.0);
        assert!(x.read(0, 1) == 2.0);
        assert!(x.read(0, 2) == 3.0);

        assert!(x.read(1, 0) == 4.0);
        assert!(x.read(1, 1) == 5.0);
        assert!(x.read(1, 2) == 6.0);

        assert!(x.read(2, 0) == 7.0);
        assert!(x.read(2, 1) == 8.0);
        assert!(x.read(2, 2) == 9.0);
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
    fn test_sqrt() {
        for _ in 0..100 {
            let a = num_complex::Complex64::new(rand::random(), rand::random());
            let num_complex::Complex {
                re: target_re,
                im: target_im,
            } = a.sqrt();
            let (sqrt_re, sqrt_im) = sqrt_impl(a.re, a.im);
            assert_approx_eq!(target_re, sqrt_re);
            assert_approx_eq!(target_im, sqrt_im);
        }
    }

    #[test]
    fn mat_mul_assign_scalar() {
        let mut x = mat![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];

        let expected = mat![[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]];
        x *= 2.0;
        assert_eq!(x, expected);

        let expected = mat![[0.0, 4.0], [8.0, 12.0], [16.0, 20.0]];
        let mut x_mut = x.as_mut();
        x_mut *= 2.0;
        assert_eq!(x, expected);
    }
}
