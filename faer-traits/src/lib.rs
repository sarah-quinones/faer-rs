use bytemuck::Pod;
use core::fmt::Debug;
use num_complex::Complex;
use pulp::Simd;

use math_utils::*;

pub mod math_utils {
    use crate::{abs_impl, ByRef, ComplexField, RealField};

    #[inline(always)]
    pub fn eps<T: RealField>() -> T {
        T::Real::epsilon_impl()
    }

    #[inline(always)]
    pub fn nbits<T: ComplexField>() -> usize {
        T::Real::nbits_impl()
    }

    #[inline(always)]
    pub fn min_positive<T: RealField>() -> T {
        T::min_positive_impl()
    }
    #[inline(always)]
    pub fn max_positive<T: RealField>() -> T {
        T::max_positive_impl()
    }
    #[inline(always)]
    pub fn sqrt_min_positive<T: RealField>() -> T {
        T::sqrt_min_positive_impl()
    }
    #[inline(always)]
    pub fn sqrt_max_positive<T: RealField>() -> T {
        T::sqrt_max_positive_impl()
    }

    #[inline(always)]
    pub fn zero<T: ComplexField>() -> T {
        T::zero_impl()
    }
    #[inline(always)]
    pub fn one<T: ComplexField>() -> T {
        T::one_impl()
    }
    #[inline(always)]
    pub fn nan<T: ComplexField>() -> T {
        T::nan_impl()
    }
    #[inline(always)]
    pub fn infinity<T: ComplexField>() -> T {
        T::infinity_impl()
    }

    #[inline(always)]
    pub fn real<T: ComplexField>(value: &T) -> T::Real {
        T::real_part_impl((value).by_ref())
    }
    #[inline(always)]
    pub fn imag<T: ComplexField>(value: &T) -> T::Real {
        T::imag_part_impl((value).by_ref())
    }
    #[inline(always)]
    pub fn neg<T: NegByRef<Output = T>>(value: &T) -> T {
        value.neg_by_ref()
    }
    #[inline(always)]
    pub fn copy<T: ComplexField>(value: &T) -> T {
        T::copy_impl((value).by_ref())
    }

    #[inline(always)]
    pub fn conj<T: ComplexField>(value: &T) -> T {
        T::conj_impl((value).by_ref())
    }

    #[inline(always)]
    pub fn add<T: AddByRef<Output = T>>(lhs: &T, rhs: &T) -> T {
        lhs.by_ref().add_by_ref(rhs.by_ref())
    }
    #[inline(always)]
    pub fn sub<T: SubByRef<Output = T>>(lhs: &T, rhs: &T) -> T {
        lhs.by_ref().sub_by_ref(rhs.by_ref())
    }
    #[inline(always)]
    pub fn mul<T: MulByRef<Output = T>>(lhs: &T, rhs: &T) -> T {
        lhs.by_ref().mul_by_ref(rhs.by_ref())
    }
    #[inline(always)]
    pub fn div<T: DivByRef<Output = T>>(lhs: &T, rhs: &T) -> T {
        lhs.by_ref().div_by_ref(rhs.by_ref())
    }

    #[inline(always)]
    pub fn mul_real<T: ComplexField>(lhs: &T, rhs: &T::Real) -> T {
        T::mul_real_impl((lhs).by_ref(), (rhs).by_ref())
    }

    #[inline(always)]
    pub fn mul_pow2<T: ComplexField>(lhs: &T, rhs: &T::Real) -> T {
        T::mul_real_impl((lhs).by_ref(), (rhs).by_ref())
    }

    #[inline(always)]
    pub fn abs1<T: ComplexField>(value: &T) -> T::Real {
        T::abs1_impl((value).by_ref())
    }

    #[inline(always)]
    pub fn abs<T: ComplexField>(value: &T) -> T::Real {
        T::abs_impl((value).by_ref())
    }

    #[inline(always)]
    pub fn hypot<T: RealField>(lhs: &T, rhs: &T) -> T {
        abs_impl::<T::Real>(lhs.clone(), rhs.clone())
    }

    #[inline(always)]
    pub fn abs2<T: ComplexField>(value: &T) -> T::Real {
        T::abs2_impl((value).by_ref())
    }

    #[inline(always)]
    pub fn max<T: RealField>(lhs: &T, rhs: &T) -> T {
        if lhs > rhs {
            copy(lhs)
        } else {
            copy(rhs)
        }
    }
    #[inline(always)]
    pub fn min<T: RealField>(lhs: &T, rhs: &T) -> T {
        if lhs < rhs {
            copy(lhs)
        } else {
            copy(rhs)
        }
    }

    #[inline(always)]
    pub fn is_nan<T: ComplexField>(value: &T) -> bool {
        T::is_nan_impl((value).by_ref())
    }

    #[inline(always)]
    pub fn is_finite<T: ComplexField>(value: &T) -> bool {
        T::is_finite_impl((value).by_ref())
    }

    #[inline(always)]
    pub fn sqrt<T: ComplexField>(value: &T) -> T {
        T::sqrt_impl((value).by_ref())
    }
    #[inline(always)]
    pub fn recip<T: ComplexField>(value: &T) -> T {
        T::recip_impl((value).by_ref())
    }

    #[inline(always)]
    pub fn from_real<T: ComplexField>(value: &T::Real) -> T {
        T::from_real_impl((value).by_ref())
    }

    #[inline(always)]
    pub fn from_f64<T: ComplexField>(value: f64) -> T {
        T::from_f64_impl(value)
    }

    pub use crate::{AddByRef, DivByRef, MulByRef, NegByRef, SubByRef};
}

pub trait AddByRef<Rhs = Self> {
    type Output;
    fn add_by_ref(&self, rhs: &Rhs) -> Self::Output;
}
pub trait SubByRef<Rhs = Self> {
    type Output;
    fn sub_by_ref(&self, rhs: &Rhs) -> Self::Output;
}
pub trait NegByRef {
    type Output;
    fn neg_by_ref(&self) -> Self::Output;
}
pub trait MulByRef<Rhs = Self> {
    type Output;
    fn mul_by_ref(&self, rhs: &Rhs) -> Self::Output;
}
pub trait DivByRef<Rhs = Self> {
    type Output;
    fn div_by_ref(&self, rhs: &Rhs) -> Self::Output;
}

trait ByRefOptIn: Copy {}

impl ByRefOptIn for usize {}
impl ByRefOptIn for isize {}
impl ByRefOptIn for i8 {}
impl ByRefOptIn for f64 {}
impl ByRefOptIn for f32 {}

impl<Rhs: ByRefOptIn, Lhs: ByRefOptIn + core::ops::Add<Rhs>> AddByRef<Rhs> for Lhs {
    type Output = <Self as core::ops::Add<Rhs>>::Output;

    #[inline]
    fn add_by_ref(&self, rhs: &Rhs) -> Self::Output {
        *self + *rhs
    }
}
impl<T: ByRefOptIn + core::ops::Neg> NegByRef for T {
    type Output = <Self as core::ops::Neg>::Output;

    #[inline]
    fn neg_by_ref(&self) -> Self::Output {
        -*self
    }
}

impl<Rhs: ByRefOptIn, Lhs: ByRefOptIn + core::ops::Sub<Rhs>> SubByRef<Rhs> for Lhs {
    type Output = <Self as core::ops::Sub<Rhs>>::Output;

    #[inline]
    fn sub_by_ref(&self, rhs: &Rhs) -> Self::Output {
        *self - *rhs
    }
}
impl<Rhs: ByRefOptIn, Lhs: ByRefOptIn + core::ops::Mul<Rhs>> MulByRef<Rhs> for Lhs {
    type Output = <Self as core::ops::Mul<Rhs>>::Output;

    #[inline]
    fn mul_by_ref(&self, rhs: &Rhs) -> Self::Output {
        *self * *rhs
    }
}
impl<Rhs: ByRefOptIn, Lhs: ByRefOptIn + core::ops::Div<Rhs>> DivByRef<Rhs> for Lhs {
    type Output = <Self as core::ops::Div<Rhs>>::Output;

    #[inline]
    fn div_by_ref(&self, rhs: &Rhs) -> Self::Output {
        *self / *rhs
    }
}

#[faer_macros::math]
fn abs_impl<T: RealField>(re: T, im: T) -> T {
    let small = sqrt_min_positive();
    let big = sqrt_max_positive();
    let one = one();
    let re_abs = abs(re);
    let im_abs = abs(im);

    if re_abs > big || im_abs > big {
        sqrt(abs2(re * small) + abs2(im * small)) * big
    } else if re_abs > one || im_abs > one {
        sqrt(abs2(re) + abs2(im))
    } else {
        sqrt(abs2(re * big) + abs2(im * big)) * small
    }
}

#[faer_macros::math]
fn recip_impl<T: RealField>(re: T, im: T) -> (T, T) {
    if is_nan(re) || is_nan(im) {
        return (nan(), nan());
    }
    if re == zero() && im == zero() {
        return (infinity(), infinity());
    }
    if !is_finite(re) || !is_finite(im) {
        return (zero(), zero());
    }

    let small = sqrt_min_positive();
    let big = sqrt_max_positive();
    let one = one();
    let re_abs = abs(re);
    let im_abs = abs(im);

    if re_abs > big || im_abs > big {
        let re = re * small;
        let im = im * small;
        let inv = recip(abs2(re) + abs2(im));
        (((re * inv) * small), ((-im * inv) * small))
    } else if re_abs > one || im_abs > one {
        let inv = recip(abs2(re) + abs2(im));
        ((re * inv), (-im * inv))
    } else {
        let re = re * big;
        let im = im * big;
        let inv = recip(abs2(re) + abs2(im));
        (((re * inv) * big), ((-im * inv) * big))
    }
}

#[faer_macros::math]
fn sqrt_impl<T: RealField>(re: T, im: T) -> (T, T) {
    let im_negative = im < zero();
    let half = from_f64(0.5);
    let abs = abs_impl(re.clone(), im.clone());

    let mut sum = re + abs;
    if sum < zero() {
        sum = zero();
    }

    let out_re = sqrt(mul_pow2(sum, half));
    let mut out_im = sqrt(mul_pow2(abs - re, half));
    if im_negative {
        out_im = -out_im;
    }
    (out_re, out_im)
}

pub trait ByRef<T> {
    fn by_ref(&self) -> &T;
}
impl<T> ByRef<T> for T {
    #[inline]
    fn by_ref(&self) -> &T {
        self
    }
}
impl<T> ByRef<T> for &T {
    #[inline]
    fn by_ref(&self) -> &T {
        *self
    }
}
impl<T> ByRef<T> for &mut T {
    #[inline]
    fn by_ref(&self) -> &T {
        *self
    }
}

#[repr(transparent)]
pub struct SimdCtx<T: ComplexField, S: Simd>(pub T::SimdCtx<S>);

#[repr(transparent)]
pub struct SimdCtxCopy<T: ComplexField, S: Simd>(pub T::SimdCtx<S>);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Real<T>(pub T);

impl<T: ComplexField, S: Simd> SimdCtx<T, S> {
    #[inline(always)]
    pub fn new(ctx: &T::SimdCtx<S>) -> &Self {
        unsafe { &*(ctx as *const T::SimdCtx<S> as *const Self) }
    }

    #[inline(always)]
    pub fn zero(&self) -> T::SimdVec<S> {
        unsafe { core::mem::zeroed() }
    }

    #[inline(always)]
    pub fn splat(&self, value: &T) -> T::SimdVec<S> {
        unsafe { core::mem::transmute_copy(&T::simd_splat(&self.0, (value).by_ref())) }
    }

    #[inline(always)]
    pub fn splat_real(&self, value: &T::Real) -> Real<T::SimdVec<S>> {
        Real(unsafe { core::mem::transmute_copy(&T::simd_splat_real(&self.0, (value).by_ref())) })
    }

    #[inline(always)]
    pub fn add(&self, lhs: T::SimdVec<S>, rhs: T::SimdVec<S>) -> T::SimdVec<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_add(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn sub(&self, lhs: T::SimdVec<S>, rhs: T::SimdVec<S>) -> T::SimdVec<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_sub(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn neg(&self, value: T::SimdVec<S>) -> T::SimdVec<S> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_neg(&self.0, value)) }
    }
    #[inline(always)]
    pub fn conj(&self, value: T::SimdVec<S>) -> T::SimdVec<S> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_conj(&self.0, value)) }
    }
    #[inline(always)]
    pub fn abs1(&self, value: T::SimdVec<S>) -> Real<T::SimdVec<S>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        Real(unsafe { core::mem::transmute_copy(&T::simd_abs1(&self.0, value)) })
    }
    #[inline(always)]
    pub fn abs_max(&self, value: T::SimdVec<S>) -> Real<T::SimdVec<S>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        Real(unsafe { core::mem::transmute_copy(&T::simd_abs_max(&self.0, value)) })
    }

    #[inline(always)]
    pub fn mul_real(&self, lhs: T::SimdVec<S>, rhs: Real<T::SimdVec<S>>) -> T::SimdVec<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_mul_real(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn mul_pow2(&self, lhs: T::SimdVec<S>, rhs: Real<T::SimdVec<S>>) -> T::SimdVec<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_mul_pow2(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn mul(&self, lhs: T::SimdVec<S>, rhs: T::SimdVec<S>) -> T::SimdVec<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_mul(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn conj_mul(&self, lhs: T::SimdVec<S>, rhs: T::SimdVec<S>) -> T::SimdVec<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_conj_mul(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn mul_add(
        &self,
        lhs: T::SimdVec<S>,
        rhs: T::SimdVec<S>,
        acc: T::SimdVec<S>,
    ) -> T::SimdVec<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        let acc = unsafe { core::mem::transmute_copy(&acc) };
        unsafe { core::mem::transmute_copy(&T::simd_mul_add(&self.0, lhs, rhs, acc)) }
    }

    #[inline(always)]
    pub fn conj_mul_add(
        &self,
        lhs: T::SimdVec<S>,
        rhs: T::SimdVec<S>,
        acc: T::SimdVec<S>,
    ) -> T::SimdVec<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        let acc = unsafe { core::mem::transmute_copy(&acc) };
        unsafe { core::mem::transmute_copy(&T::simd_conj_mul_add(&self.0, lhs, rhs, acc)) }
    }

    #[inline(always)]
    pub fn abs2(&self, value: T::SimdVec<S>) -> Real<T::SimdVec<S>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        Real(unsafe { core::mem::transmute_copy(&T::simd_abs2(&self.0, value)) })
    }

    #[inline(always)]
    pub fn abs2_add(&self, value: T::SimdVec<S>, acc: Real<T::SimdVec<S>>) -> Real<T::SimdVec<S>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        let acc = unsafe { core::mem::transmute_copy(&acc) };
        Real(unsafe { core::mem::transmute_copy(&T::simd_abs2_add(&self.0, value, acc)) })
    }

    #[inline(always)]
    pub fn reduce_sum(&self, value: T::SimdVec<S>) -> T {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_reduce_sum(&self.0, value)) }
    }
    #[inline(always)]
    pub fn reduce_max(&self, value: Real<T::SimdVec<S>>) -> RealValue<T> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_reduce_max(&self.0, value)) }
    }

    #[inline(always)]
    pub fn max(&self, lhs: Real<T::SimdVec<S>>, rhs: Real<T::SimdVec<S>>) -> Real<T::SimdVec<S>> {
        let cmp = self.gt(lhs, rhs);
        Real(self.select(cmp, lhs.0, rhs.0))
    }

    #[inline(always)]
    pub fn lt(&self, lhs: Real<T::SimdVec<S>>, rhs: Real<T::SimdVec<S>>) -> T::SimdMask<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_less_than(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn gt(&self, lhs: Real<T::SimdVec<S>>, rhs: Real<T::SimdVec<S>>) -> T::SimdMask<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_greater_than(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn le(&self, lhs: Real<T::SimdVec<S>>, rhs: Real<T::SimdVec<S>>) -> T::SimdMask<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_less_than_or_equal(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn ge(&self, lhs: Real<T::SimdVec<S>>, rhs: Real<T::SimdVec<S>>) -> T::SimdMask<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_greater_than_or_equal(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn select(
        &self,
        mask: T::SimdMask<S>,
        lhs: T::SimdVec<S>,
        rhs: T::SimdVec<S>,
    ) -> T::SimdVec<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_select(&self.0, mask, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn iselect(
        &self,
        mask: T::SimdMask<S>,
        lhs: T::SimdIndex<S>,
        rhs: T::SimdIndex<S>,
    ) -> T::SimdIndex<S> {
        unsafe { core::mem::transmute_copy(&T::simd_index_select(&self.0, mask, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn isplat(&self, value: T::Index) -> T::SimdIndex<S> {
        unsafe { core::mem::transmute_copy(&T::simd_index_splat(&self.0, value)) }
    }
    #[inline(always)]
    pub fn iadd(&self, lhs: T::SimdIndex<S>, rhs: T::SimdIndex<S>) -> T::SimdIndex<S> {
        unsafe { core::mem::transmute_copy(&T::simd_index_add(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn tail_mask(&self, len: usize) -> T::SimdMask<S> {
        unsafe { core::mem::transmute_copy(&T::simd_tail_mask(&self.0, len)) }
    }
    #[inline(always)]
    pub fn head_mask(&self, len: usize) -> T::SimdMask<S> {
        unsafe { core::mem::transmute_copy(&T::simd_head_mask(&self.0, len)) }
    }
    #[inline(always)]
    pub fn and_mask(&self, lhs: T::SimdMask<S>, rhs: T::SimdMask<S>) -> T::SimdMask<S> {
        T::simd_and_mask(&self.0, lhs, rhs)
    }
    #[inline(always)]
    pub fn first_true_mask(&self, value: T::SimdMask<S>) -> usize {
        T::simd_first_true_mask(&self.0, value)
    }
    #[inline(always)]
    pub unsafe fn mask_load(
        &self,
        mask: T::SimdMask<S>,
        ptr: *const T::SimdVec<S>,
    ) -> T::SimdVec<S> {
        unsafe { core::mem::transmute_copy(&T::simd_mask_load(&self.0, mask, ptr)) }
    }
    #[inline(always)]
    pub unsafe fn mask_store(
        &self,
        mask: T::SimdMask<S>,
        ptr: *mut T::SimdVec<S>,
        value: T::SimdVec<S>,
    ) {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_mask_store(&self.0, mask, ptr, value)) }
    }

    #[inline(always)]
    pub fn load(&self, ptr: &T::SimdVec<S>) -> T::SimdVec<S> {
        unsafe { core::mem::transmute_copy(&T::simd_load(&self.0, ptr)) }
    }
    #[inline(always)]
    pub fn store(&self, ptr: &mut T::SimdVec<S>, value: T::SimdVec<S>) {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_store(&self.0, ptr, value)) }
    }
}

pub unsafe trait Conjugate {
    const IS_CANONICAL: bool;

    type Conj: Conjugate;
    type Canonical: Conjugate<Canonical = Self::Canonical> + ComplexField;
}

pub type RealValue<T> = <<T as Conjugate>::Canonical as ComplexField>::Real;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ComplexConj<T> {
    pub re: T,
    pub im_neg: T,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdCapabilities {
    None,
    Copy,
    Shuffled,
    All,
}

impl SimdCapabilities {
    #[inline]
    pub const fn is_copy(self) -> bool {
        matches!(self, Self::Copy | Self::Shuffled | Self::All)
    }

    #[inline]
    pub const fn is_simd(self) -> bool {
        matches!(self, Self::Shuffled | Self::All)
    }

    #[inline]
    pub const fn is_unshuffled_simd(self) -> bool {
        matches!(self, Self::All)
    }
}

mod seal {
    pub trait Seal {}
    impl Seal for u32 {}
    impl Seal for u64 {}
    impl Seal for usize {}
    impl Seal for i32 {}
    impl Seal for i64 {}
    impl Seal for isize {}
}

pub trait Seal: seal::Seal {}
impl<T: seal::Seal> Seal for T {}

/// Trait for signed integers corresponding to the ones satisfying [`Index`].
///
/// Always smaller than or equal to `isize`.
pub trait SignedIndex:
    Seal
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

pub trait Index:
    Seal
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

    const IOTA: &[Self; 32];
    const COMPLEX_IOTA: &[Self; 32];
}

impl Index for u32 {
    type FixedWidth = u32;
    type Signed = i32;

    const IOTA: &[Self; 32] = &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31u32,
    ];
    const COMPLEX_IOTA: &[Self; 32] = &[
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
        14, 14, 15, 15u32,
    ];
}
impl Index for u64 {
    type FixedWidth = u64;
    type Signed = i64;

    const IOTA: &[Self; 32] = &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31u64,
    ];
    const COMPLEX_IOTA: &[Self; 32] = &[
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
        14, 14, 15, 15u64,
    ];
}

impl Index for usize {
    #[cfg(target_pointer_width = "32")]
    type FixedWidth = u32;
    #[cfg(target_pointer_width = "64")]
    type FixedWidth = u64;

    const IOTA: &[Self; 32] = &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31usize,
    ];
    const COMPLEX_IOTA: &[Self; 32] = &[
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
        14, 14, 15, 15usize,
    ];

    type Signed = isize;
}

unsafe impl<T: RealField> Conjugate for T {
    const IS_CANONICAL: bool = true;
    type Conj = T;
    type Canonical = T;
}

pub trait EnableComplex: Sized + RealField + Default {
    type Arch: SimdArch;

    const COMPLEX_SIMD_CAPABILITIES: SimdCapabilities = match Self::SIMD_CAPABILITIES {
        SimdCapabilities::Copy => SimdCapabilities::Copy,
        _ => SimdCapabilities::None,
    };
    type SimdComplexUnit<S: Simd>: Pod + Debug;

    fn simd_complex_splat<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: &Complex<Self>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_splat_real<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: &Self,
    ) -> Self::SimdComplexUnit<S>;

    fn simd_complex_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_sub<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_neg<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_conj<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_abs1<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_abs_max<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_abs2<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_abs2_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
        acc: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_mul<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_conj_mul<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_mul_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
        acc: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_conj_mul_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
        acc: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;

    fn simd_complex_mul_real<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_mul_pow2<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;

    fn simd_complex_reduce_sum<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Complex<Self>;
    fn simd_complex_reduce_max<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self;

    fn simd_complex_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S>;
    fn simd_complex_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S>;
    fn simd_complex_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S>;
    fn simd_complex_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S>;

    fn simd_complex_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;

    unsafe fn simd_complex_mask_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *const Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    unsafe fn simd_complex_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *mut Self::SimdComplexUnit<S>,
        value: Self::SimdComplexUnit<S>,
    );
    fn simd_complex_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: &Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S>;
    fn simd_complex_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: &mut Self::SimdComplexUnit<S>,
        value: Self::SimdComplexUnit<S>,
    );

    fn simd_complex_iota<S: Simd>(ctx: &Self::SimdCtx<S>) -> Self::SimdIndex<S>;
}

unsafe impl<T: EnableComplex> Conjugate for Complex<T> {
    const IS_CANONICAL: bool = true;
    type Conj = ComplexConj<T>;
    type Canonical = Complex<T>;
}
unsafe impl<T: EnableComplex> Conjugate for ComplexConj<T> {
    const IS_CANONICAL: bool = false;
    type Conj = Complex<T>;
    type Canonical = Complex<T>;
}

pub trait SimdArch: Default {
    fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R;
}

impl SimdArch for pulp::Arch {
    #[inline]
    fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R {
        self.dispatch(f)
    }
}

impl SimdArch for pulp::ScalarArch {
    #[inline]
    fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R {
        self.dispatch(f)
    }
}

pub trait ComplexField:
    Debug
    + Send
    + Sync
    + Clone
    + Conjugate<Canonical = Self, Conj: Conjugate<Canonical = Self>>
    + PartialEq
    + AddByRef<Output = Self>
    + SubByRef<Output = Self>
    + MulByRef<Output = Self>
    + NegByRef<Output = Self>
{
    const IS_REAL: bool;

    type Arch: SimdArch;

    type SimdCtx<S: Simd>: Copy;
    type Index: Index;

    type Real: RealField;

    #[doc(hidden)]
    const IS_NATIVE_F32: bool = false;
    #[doc(hidden)]
    const IS_NATIVE_C32: bool = false;
    #[doc(hidden)]
    const IS_NATIVE_F64: bool = false;
    #[doc(hidden)]
    const IS_NATIVE_C64: bool = false;

    const SIMD_CAPABILITIES: SimdCapabilities;
    type SimdMask<S: Simd>: Copy + Debug;
    type SimdVec<S: Simd>: Pod + Debug;
    type SimdIndex<S: Simd>: Pod + Debug;

    fn zero_impl() -> Self;
    fn one_impl() -> Self;
    fn nan_impl() -> Self;
    fn infinity_impl() -> Self;

    fn from_real_impl(real: &Self::Real) -> Self;
    fn from_f64_impl(real: f64) -> Self;

    fn real_part_impl(value: &Self) -> Self::Real;
    fn imag_part_impl(value: &Self) -> Self::Real;

    fn copy_impl(value: &Self) -> Self;
    fn conj_impl(value: &Self) -> Self;
    fn recip_impl(value: &Self) -> Self;
    fn sqrt_impl(value: &Self) -> Self;

    fn abs_impl(value: &Self) -> Self::Real;
    fn abs1_impl(value: &Self) -> Self::Real;
    fn abs2_impl(value: &Self) -> Self::Real;

    fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self;

    fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self;

    fn is_finite_impl(value: &Self) -> bool;
    fn is_nan_impl(value: &Self) -> bool {
        value != value
    }

    fn simd_ctx<S: Simd>(simd: S) -> Self::SimdCtx<S>;
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> S;

    fn simd_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self) -> Self::SimdVec<S>;
    fn simd_splat_real<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self::Real) -> Self::SimdVec<S>;

    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;

    fn simd_neg<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S>;
    fn simd_conj<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S>;
    fn simd_abs1<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S>;
    fn simd_abs_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S>;

    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;

    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;
    fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S>;
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;

    fn simd_reduce_sum<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self;
    fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::Real;
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S>;
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S>;
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S>;
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S>;

    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;
    fn simd_index_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S>;

    fn simd_index_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::Index) -> Self::SimdIndex<S>;
    fn simd_index_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S>;

    fn simd_tail_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S>;
    fn simd_and_mask<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdMask<S>,
        rhs: Self::SimdMask<S>,
    ) -> Self::SimdMask<S>;
    fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize;

    fn simd_head_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S>;
    unsafe fn simd_mask_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *const Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;
    unsafe fn simd_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *mut Self::SimdVec<S>,
        value: Self::SimdVec<S>,
    );

    fn simd_load<S: Simd>(ctx: &Self::SimdCtx<S>, ptr: &Self::SimdVec<S>) -> Self::SimdVec<S>;
    fn simd_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: &mut Self::SimdVec<S>,
        value: Self::SimdVec<S>,
    );

    fn simd_iota<S: Simd>(ctx: &Self::SimdCtx<S>) -> Self::SimdIndex<S>;
}

pub trait RealField:
    ComplexField<Real = Self, Conj = Self> + DivByRef<Output = Self> + PartialOrd
{
    fn epsilon_impl() -> Self;
    fn nbits_impl() -> usize;

    fn min_positive_impl() -> Self;
    fn max_positive_impl() -> Self;
    fn sqrt_min_positive_impl() -> Self;
    fn sqrt_max_positive_impl() -> Self;
}

impl ComplexField for f32 {
    const IS_REAL: bool = true;

    type Index = u32;
    type SimdCtx<S: Simd> = S;
    type Real = Self;
    type Arch = pulp::Arch;

    const IS_NATIVE_F32: bool = true;

    const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::All;

    type SimdMask<S: Simd> = S::m32s;
    type SimdVec<S: Simd> = S::f32s;
    type SimdIndex<S: Simd> = S::u32s;

    #[inline(always)]
    fn zero_impl() -> Self {
        0.0
    }
    #[inline(always)]
    fn one_impl() -> Self {
        1.0
    }
    #[inline(always)]
    fn nan_impl() -> Self {
        Self::NAN
    }
    #[inline(always)]
    fn infinity_impl() -> Self {
        Self::INFINITY
    }
    #[inline(always)]
    fn from_real_impl(value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn from_f64_impl(value: f64) -> Self {
        value as _
    }

    #[inline(always)]
    fn real_part_impl(value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn imag_part_impl(_: &Self) -> Self {
        0.0
    }
    #[inline(always)]
    fn copy_impl(value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn conj_impl(value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn recip_impl(value: &Self) -> Self {
        1.0 / *value
    }
    #[inline(always)]
    fn sqrt_impl(value: &Self) -> Self {
        (*value).sqrt()
    }
    #[inline(always)]
    fn abs_impl(value: &Self) -> Self {
        (*value).abs()
    }
    #[inline(always)]
    fn abs1_impl(value: &Self) -> Self {
        (*value).abs()
    }
    #[inline(always)]
    fn abs2_impl(value: &Self) -> Self {
        (*value) * (*value)
    }
    #[inline(always)]
    fn mul_real_impl(lhs: &Self, rhs: &Self) -> Self {
        (*lhs) * (*rhs)
    }
    #[inline(always)]
    fn mul_pow2_impl(lhs: &Self, rhs: &Self) -> Self {
        (*lhs) * (*rhs)
    }

    #[inline(always)]
    fn is_finite_impl(value: &Self) -> bool {
        (*value).is_finite()
    }
    #[inline(always)]
    fn is_nan_impl(value: &Self) -> bool {
        (*value).is_nan()
    }

    #[inline(always)]
    fn simd_ctx<S: Simd>(simd: S) -> Self::SimdCtx<S> {
        simd
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self) -> Self::SimdVec<S> {
        ctx.splat_f32s(*value)
    }

    #[inline(always)]
    fn simd_splat_real<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self::Real) -> Self::SimdVec<S> {
        ctx.splat_f32s(*value)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.add_f32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.sub_f32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.neg_f32s(value)
    }
    #[inline(always)]
    fn simd_conj<S: Simd>(_: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        value
    }
    #[inline(always)]
    fn simd_abs1<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.abs_f32s(value)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_f32s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_f32s(lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_f32s(lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_f32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_add_e_f32s(lhs, rhs, acc)
    }
    #[inline(always)]
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_add_e_f32s(lhs, rhs, acc)
    }
    #[inline(always)]
    fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.mul_f32s(value, value)
    }
    #[inline(always)]
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_add_e_f32s(value, value, acc)
    }
    #[inline(always)]
    fn simd_reduce_sum<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
        ctx.reduce_sum_f32s(value)
    }
    #[inline(always)]
    fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::Real {
        ctx.reduce_max_f32s(value)
    }
    #[inline(always)]
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ctx.less_than_f32s(real_lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ctx.greater_than_f32s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ctx.less_than_or_equal_f32s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ctx.greater_than_or_equal_f32s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.select_f32s_m32s(mask, lhs, rhs)
    }
    #[inline(always)]
    fn simd_index_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        ctx.select_u32s_m32s(mask, lhs, rhs)
    }
    #[inline(always)]
    fn simd_index_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::Index) -> Self::SimdIndex<S> {
        ctx.splat_u32s(value)
    }
    #[inline(always)]
    fn simd_index_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        ctx.add_u32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_tail_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S> {
        ctx.tail_mask_f32s(len)
    }
    #[inline(always)]
    fn simd_head_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S> {
        ctx.head_mask_f32s(len)
    }
    #[inline(always)]
    unsafe fn simd_mask_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *const Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mask_load_ptr_f32s(mask, ptr as *const f32)
    }
    #[inline(always)]
    unsafe fn simd_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *mut Self::SimdVec<S>,
        value: Self::SimdVec<S>,
    ) {
        ctx.mask_store_ptr_f32s(mask, ptr as *mut f32, value);
    }

    #[inline(always)]
    fn simd_iota<S: Simd>(_: &Self::SimdCtx<S>) -> Self::SimdIndex<S> {
        const {
            core::assert!(
                size_of::<Self::Index>() * Self::Index::IOTA.len()
                    >= size_of::<Self::SimdIndex<S>>()
            )
        };
        unsafe { core::mem::transmute_copy(Self::Index::IOTA) }
    }

    #[inline(always)]
    fn simd_load<S: Simd>(_: &Self::SimdCtx<S>, ptr: &Self::SimdVec<S>) -> Self::SimdVec<S> {
        *ptr
    }
    #[inline(always)]
    fn simd_store<S: Simd>(
        _: &Self::SimdCtx<S>,
        ptr: &mut Self::SimdVec<S>,
        value: Self::SimdVec<S>,
    ) {
        *ptr = value;
    }

    #[inline(always)]
    fn simd_abs_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.abs_f32s(value)
    }

    #[inline(always)]
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> S {
        *ctx
    }

    #[inline(always)]
    fn simd_and_mask<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdMask<S>,
        rhs: Self::SimdMask<S>,
    ) -> Self::SimdMask<S> {
        simd.and_m32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
        ctx.first_true_m32s(value)
    }
}

impl RealField for f32 {
    #[inline(always)]
    fn epsilon_impl() -> Self {
        Self::EPSILON
    }
    #[inline(always)]
    fn min_positive_impl() -> Self {
        Self::MIN_POSITIVE
    }
    #[inline(always)]
    fn max_positive_impl() -> Self {
        Self::MIN_POSITIVE.recip()
    }
    #[inline(always)]
    fn sqrt_min_positive_impl() -> Self {
        Self::MIN_POSITIVE.sqrt()
    }
    #[inline(always)]
    fn sqrt_max_positive_impl() -> Self {
        Self::MIN_POSITIVE.recip().sqrt()
    }

    #[inline(always)]
    fn nbits_impl() -> usize {
        Self::MANTISSA_DIGITS as usize
    }
}

impl ComplexField for f64 {
    const IS_REAL: bool = true;

    type Index = u64;
    type SimdCtx<S: Simd> = S;
    type Real = Self;
    type Arch = pulp::Arch;

    const IS_NATIVE_F64: bool = true;

    const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::All;

    type SimdMask<S: Simd> = S::m64s;
    type SimdVec<S: Simd> = S::f64s;
    type SimdIndex<S: Simd> = S::u64s;

    #[inline(always)]
    fn zero_impl() -> Self {
        0.0
    }
    #[inline(always)]
    fn one_impl() -> Self {
        1.0
    }
    #[inline(always)]
    fn nan_impl() -> Self {
        Self::NAN
    }
    #[inline(always)]
    fn infinity_impl() -> Self {
        Self::INFINITY
    }
    #[inline(always)]
    fn from_real_impl(value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn from_f64_impl(value: f64) -> Self {
        value as _
    }

    #[inline(always)]
    fn real_part_impl(value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn imag_part_impl(_: &Self) -> Self {
        0.0
    }
    #[inline(always)]
    fn copy_impl(value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn conj_impl(value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn recip_impl(value: &Self) -> Self {
        1.0 / *value
    }
    #[inline(always)]
    fn sqrt_impl(value: &Self) -> Self {
        (*value).sqrt()
    }
    #[inline(always)]
    fn abs_impl(value: &Self) -> Self {
        (*value).abs()
    }
    #[inline(always)]
    fn abs1_impl(value: &Self) -> Self {
        (*value).abs()
    }
    #[inline(always)]
    fn abs2_impl(value: &Self) -> Self {
        (*value) * (*value)
    }
    #[inline(always)]
    fn mul_real_impl(lhs: &Self, rhs: &Self) -> Self {
        (*lhs) * (*rhs)
    }
    #[inline(always)]
    fn mul_pow2_impl(lhs: &Self, rhs: &Self) -> Self {
        (*lhs) * (*rhs)
    }

    #[inline(always)]
    fn is_nan_impl(value: &Self) -> bool {
        (*value).is_nan()
    }

    #[inline(always)]
    fn simd_ctx<S: Simd>(simd: S) -> Self::SimdCtx<S> {
        simd
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self) -> Self::SimdVec<S> {
        ctx.splat_f64s(*value)
    }

    #[inline(always)]
    fn simd_splat_real<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self::Real) -> Self::SimdVec<S> {
        ctx.splat_f64s(*value)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.add_f64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.sub_f64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.neg_f64s(value)
    }
    #[inline(always)]
    fn simd_conj<S: Simd>(_: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        value
    }
    #[inline(always)]
    fn simd_abs1<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.abs_f64s(value)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_f64s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_f64s(lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_f64s(lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_f64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_add_e_f64s(lhs, rhs, acc)
    }
    #[inline(always)]
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_add_e_f64s(lhs, rhs, acc)
    }
    #[inline(always)]
    fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.mul_f64s(value, value)
    }
    #[inline(always)]
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_add_e_f64s(value, value, acc)
    }
    #[inline(always)]
    fn simd_reduce_sum<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
        ctx.reduce_sum_f64s(value)
    }
    #[inline(always)]
    fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::Real {
        ctx.reduce_max_f64s(value)
    }
    #[inline(always)]
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ctx.less_than_f64s(real_lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ctx.greater_than_f64s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ctx.less_than_or_equal_f64s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        ctx.greater_than_or_equal_f64s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.select_f64s_m64s(mask, lhs, rhs)
    }
    #[inline(always)]
    fn simd_index_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        ctx.select_u64s_m64s(mask, lhs, rhs)
    }
    #[inline(always)]
    fn simd_index_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::Index) -> Self::SimdIndex<S> {
        ctx.splat_u64s(value)
    }
    #[inline(always)]
    fn simd_index_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        ctx.add_u64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_tail_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S> {
        ctx.tail_mask_f64s(len)
    }
    #[inline(always)]
    fn simd_head_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S> {
        ctx.head_mask_f64s(len)
    }
    #[inline(always)]
    unsafe fn simd_mask_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *const Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mask_load_ptr_f64s(mask, ptr as *const f64)
    }
    #[inline(always)]
    unsafe fn simd_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *mut Self::SimdVec<S>,
        value: Self::SimdVec<S>,
    ) {
        ctx.mask_store_ptr_f64s(mask, ptr as *mut f64, value);
    }

    #[inline(always)]
    fn simd_iota<S: Simd>(_: &Self::SimdCtx<S>) -> Self::SimdIndex<S> {
        const {
            core::assert!(
                size_of::<Self::Index>() * Self::Index::IOTA.len()
                    >= size_of::<Self::SimdIndex<S>>()
            )
        };
        unsafe { core::mem::transmute_copy(Self::Index::IOTA) }
    }
    #[inline(always)]
    fn simd_load<S: Simd>(_: &Self::SimdCtx<S>, ptr: &Self::SimdVec<S>) -> Self::SimdVec<S> {
        *ptr
    }
    #[inline(always)]
    fn simd_store<S: Simd>(
        _: &Self::SimdCtx<S>,
        ptr: &mut Self::SimdVec<S>,
        value: Self::SimdVec<S>,
    ) {
        *ptr = value;
    }

    #[inline(always)]
    fn simd_abs_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.abs_f64s(value)
    }

    #[inline(always)]
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> S {
        *ctx
    }

    #[inline(always)]
    fn simd_and_mask<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdMask<S>,
        rhs: Self::SimdMask<S>,
    ) -> Self::SimdMask<S> {
        simd.and_m64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
        ctx.first_true_m64s(value)
    }

    #[inline(always)]
    fn is_finite_impl(value: &Self) -> bool {
        (*value).is_finite()
    }
}

impl RealField for f64 {
    #[inline(always)]
    fn epsilon_impl() -> Self {
        Self::EPSILON
    }
    #[inline(always)]
    fn min_positive_impl() -> Self {
        Self::MIN_POSITIVE
    }
    #[inline(always)]
    fn max_positive_impl() -> Self {
        Self::MIN_POSITIVE.recip()
    }
    #[inline(always)]
    fn sqrt_min_positive_impl() -> Self {
        Self::MIN_POSITIVE.sqrt()
    }
    #[inline(always)]
    fn sqrt_max_positive_impl() -> Self {
        Self::MIN_POSITIVE.recip().sqrt()
    }

    #[inline(always)]
    fn nbits_impl() -> usize {
        Self::MANTISSA_DIGITS as usize
    }
}

impl<T: AddByRef<Output = T>> AddByRef for Complex<T> {
    type Output = Self;

    #[inline]
    fn add_by_ref(&self, rhs: &Self) -> Self::Output {
        Complex {
            re: self.re.add_by_ref(&rhs.re),
            im: self.im.add_by_ref(&rhs.im),
        }
    }
}
impl<T: SubByRef<Output = T>> SubByRef for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub_by_ref(&self, rhs: &Self) -> Self::Output {
        Complex {
            re: self.re.sub_by_ref(&rhs.re),
            im: self.im.sub_by_ref(&rhs.im),
        }
    }
}
impl<T: AddByRef<Output = T> + SubByRef<Output = T> + MulByRef<Output = T>> MulByRef
    for Complex<T>
{
    type Output = Self;

    #[inline]
    #[faer_macros::math]
    fn mul_by_ref(&self, rhs: &Self) -> Self::Output {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T: NegByRef<Output = T>> NegByRef for Complex<T> {
    type Output = Self;

    #[inline]
    #[faer_macros::math]
    fn neg_by_ref(&self) -> Self::Output {
        Complex {
            re: self.re.neg_by_ref(),
            im: self.im.neg_by_ref(),
        }
    }
}

impl<T: EnableComplex> ComplexField for Complex<T> {
    const IS_REAL: bool = false;
    type Arch = <T as EnableComplex>::Arch;

    type SimdCtx<S: Simd> = T::SimdCtx<S>;
    type Index = T::Index;

    type Real = T;

    const IS_NATIVE_C32: bool = T::IS_NATIVE_F32;
    const IS_NATIVE_C64: bool = T::IS_NATIVE_F64;

    const SIMD_CAPABILITIES: SimdCapabilities = T::SIMD_CAPABILITIES;
    type SimdMask<S: Simd> = T::SimdMask<S>;
    type SimdVec<S: Simd> = T::SimdComplexUnit<S>;
    type SimdIndex<S: Simd> = T::SimdIndex<S>;

    #[inline(always)]
    #[faer_macros::math]
    fn zero_impl() -> Self {
        Complex {
            re: T::zero_impl(),
            im: T::zero_impl(),
        }
    }

    #[inline(always)]
    #[faer_macros::math]
    fn mul_real_impl(lhs: &Self, rhs: &T) -> Self {
        Complex::new(mul_real(&lhs.re, &rhs), mul_real(&lhs.im, &rhs))
    }

    #[inline(always)]
    #[faer_macros::math]
    fn mul_pow2_impl(lhs: &Self, rhs: &T) -> Self {
        Complex::new(mul_pow2(&lhs.re, &rhs), mul_pow2(&lhs.im, &rhs))
    }

    #[inline(always)]
    #[faer_macros::math]
    fn one_impl() -> Self {
        Complex {
            re: T::one_impl(),
            im: T::zero_impl(),
        }
    }

    #[inline(always)]
    #[faer_macros::math]
    fn real_part_impl(value: &Self) -> T {
        T::copy_impl(&value.re)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn imag_part_impl(value: &Self) -> T {
        T::copy_impl(&value.im)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn copy_impl(value: &Self) -> Self {
        Complex::new(copy(&value.re), copy(&value.im))
    }

    #[inline(always)]
    #[faer_macros::math]
    fn conj_impl(value: &Self) -> Self {
        Complex::new(copy(&value.re), neg(&value.im))
    }

    #[inline(always)]
    #[faer_macros::math]
    fn abs1_impl(value: &Self) -> T {
        abs1(value.re) + abs1(value.im)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn abs2_impl(value: &Self) -> T {
        abs2(value.re) + abs2(value.im)
    }

    #[faer_macros::math]
    fn abs_impl(value: &Self) -> T {
        abs_impl(value.re.clone(), value.im.clone())
    }

    #[faer_macros::math]
    fn recip_impl(value: &Self) -> Self {
        let (re, im) = recip_impl(value.re.clone(), value.im.clone());
        Complex { re, im }
    }

    #[inline(always)]
    #[faer_macros::math]
    fn nan_impl() -> Self {
        Complex {
            re: T::nan_impl(),
            im: T::nan_impl(),
        }
    }

    #[inline(always)]
    #[faer_macros::math]
    fn infinity_impl() -> Self {
        Complex {
            re: T::infinity_impl(),
            im: T::infinity_impl(),
        }
    }

    #[faer_macros::math]
    fn sqrt_impl(value: &Self) -> Self {
        let (re, im) = sqrt_impl::<T>(value.re.clone(), value.im.clone());
        Complex { re, im }
    }

    #[inline(always)]
    #[faer_macros::math]
    fn from_real_impl(real: &Self::Real) -> Self {
        Complex {
            re: copy(real),
            im: zero(),
        }
    }

    #[inline(always)]
    #[faer_macros::math]
    fn from_f64_impl(real: f64) -> Self {
        Complex {
            re: from_f64(real),
            im: zero(),
        }
    }

    #[inline(always)]
    #[faer_macros::math]
    fn is_finite_impl(value: &Self) -> bool {
        is_finite(value.re) && is_finite(value.im)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn is_nan_impl(value: &Self) -> bool {
        is_nan(value.re) || is_nan(value.im)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_ctx<S: Simd>(simd: S) -> Self::SimdCtx<S> {
        T::simd_ctx(simd)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self) -> T::SimdComplexUnit<S> {
        T::simd_complex_splat(ctx, value)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_splat_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: &Self::Real,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_splat_real(ctx, value)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_add(ctx, lhs, rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_sub(ctx, lhs, rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_neg<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_neg(ctx, value)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_conj<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_conj(ctx, value)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_abs1<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_abs1(ctx, value)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_mul_real(ctx, lhs, real_rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_mul_pow2(ctx, lhs, real_rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_mul(ctx, lhs, rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_conj_mul(ctx, lhs, rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
        acc: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_mul_add(ctx, lhs, rhs, acc)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
        acc: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_conj_mul_add(ctx, lhs, rhs, acc)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_abs2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_abs2(ctx, value)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: T::SimdComplexUnit<S>,
        acc: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_abs2_add(ctx, value, acc)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_reduce_sum<S: Simd>(ctx: &Self::SimdCtx<S>, value: T::SimdComplexUnit<S>) -> Self {
        T::simd_complex_reduce_sum(ctx, value)
    }
    #[inline(always)]
    #[faer_macros::math]
    fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: T::SimdComplexUnit<S>) -> T {
        T::simd_complex_reduce_max(ctx, value)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        T::simd_complex_less_than(ctx, real_lhs, real_rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        T::simd_complex_greater_than(ctx, real_lhs, real_rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        T::simd_complex_less_than_or_equal(ctx, real_lhs, real_rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        T::simd_complex_greater_than_or_equal(ctx, real_lhs, real_rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_select(ctx, mask, lhs, rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_index_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        let ctx = SimdCtx::<T, S>::new(ctx);
        ctx.iselect(mask, lhs, rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_index_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::Index) -> Self::SimdIndex<S> {
        let ctx = SimdCtx::<T, S>::new(ctx);
        ctx.isplat(value)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_index_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        let ctx = SimdCtx::<T, S>::new(ctx);
        ctx.iadd(lhs, rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_tail_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S> {
        let ctx = SimdCtx::<T, S>::new(ctx);
        ctx.tail_mask(2 * len)
    }
    #[inline(always)]
    #[faer_macros::math]
    fn simd_head_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S> {
        let ctx = SimdCtx::<T, S>::new(ctx);
        ctx.head_mask(2 * len)
    }

    #[inline(always)]
    #[faer_macros::math]
    unsafe fn simd_mask_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *const T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_mask_load(ctx, mask, ptr)
    }
    #[inline(always)]
    #[faer_macros::math]
    unsafe fn simd_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *mut T::SimdComplexUnit<S>,
        value: T::SimdComplexUnit<S>,
    ) {
        T::simd_complex_mask_store(ctx, mask, ptr, value)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: &T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_load(ctx, ptr)
    }
    #[inline(always)]
    #[faer_macros::math]
    fn simd_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: &mut T::SimdComplexUnit<S>,
        value: T::SimdComplexUnit<S>,
    ) {
        T::simd_complex_store(ctx, ptr, value)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_iota<S: Simd>(ctx: &Self::SimdCtx<S>) -> Self::SimdIndex<S> {
        T::simd_complex_iota(ctx)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_abs_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        T::simd_complex_abs_max(ctx, value)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> S {
        T::ctx_from_simd(ctx)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_and_mask<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdMask<S>,
        rhs: Self::SimdMask<S>,
    ) -> Self::SimdMask<S> {
        T::simd_and_mask(simd, lhs, rhs)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
        T::simd_first_true_mask(ctx, value)
    }
}

impl EnableComplex for f32 {
    const COMPLEX_SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::All;
    type SimdComplexUnit<S: Simd> = S::c32s;
    type Arch = pulp::Arch;

    #[inline(always)]
    fn simd_complex_splat<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: &Complex<Self>,
    ) -> Self::SimdComplexUnit<S> {
        simd.splat_c32s(*value)
    }
    #[inline(always)]
    fn simd_complex_splat_real<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: &Self,
    ) -> Self::SimdComplexUnit<S> {
        simd.splat_c32s(Complex {
            re: *value,
            im: *value,
        })
    }

    #[inline(always)]
    fn simd_complex_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.add_c32s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_complex_sub<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.sub_c32s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_complex_neg<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.neg_c32s(value)
    }
    #[inline(always)]
    fn simd_complex_conj<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.conj_c32s(value)
    }
    #[inline(always)]
    fn simd_complex_abs1<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            bytemuck::cast(simd.abs_f32s(bytemuck::cast(value)))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            let value: Complex<f32> = bytemuck::cast(value);
            let abs = value.re.abs() + value.im.abs();
            bytemuck::cast(Complex { re: abs, im: abs })
        } else {
            panic!();
        }
    }
    #[inline(always)]
    fn simd_complex_abs2<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.abs2_c32s(value)
    }
    #[inline(always)]
    fn simd_complex_abs2_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
        acc: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.add_c32s(simd.abs2_c32s(value), acc)
    }
    #[inline(always)]
    fn simd_complex_mul<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.mul_e_c32s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_complex_conj_mul<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.conj_mul_e_c32s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_complex_mul_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
        acc: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.mul_add_e_c32s(lhs, rhs, acc)
    }
    #[inline(always)]
    fn simd_complex_conj_mul_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
        acc: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.conj_mul_add_e_c32s(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_complex_mul_real<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            bytemuck::cast(simd.mul_f32s(bytemuck::cast(lhs), bytemuck::cast(real_rhs)))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            let mut lhs: Complex<f32> = bytemuck::cast(lhs);
            let rhs: Complex<f32> = bytemuck::cast(real_rhs);
            lhs *= rhs.re;
            bytemuck::cast(lhs)
        } else {
            panic!();
        }
    }
    #[inline(always)]
    fn simd_complex_mul_pow2<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        Self::simd_complex_mul_real(simd, lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_complex_reduce_sum<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Complex<Self> {
        simd.reduce_sum_c32s(value)
    }
    #[inline(always)]
    fn simd_complex_reduce_max<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            bytemuck::cast(simd.reduce_max_f32s(bytemuck::cast(value)))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            let tmp = bytemuck::cast::<_, Complex<f32>>(value);
            if tmp.re > tmp.im {
                tmp.re
            } else {
                tmp.im
            }
        } else {
            panic!()
        }
    }

    #[inline(always)]
    fn simd_complex_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            ctx.less_than_f32s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            assert!(const { size_of::<S::m32s>() == size_of::<bool>() });

            let lhs: Complex<f32> = bytemuck::cast(real_lhs);
            let rhs: Complex<f32> = bytemuck::cast(real_rhs);
            unsafe { core::mem::transmute_copy(&(lhs.re < rhs.re)) }
        } else {
            panic!();
        }
    }
    #[inline(always)]
    fn simd_complex_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            ctx.less_than_or_equal_f32s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            assert!(const { size_of::<S::m32s>() == size_of::<bool>() });

            let lhs: Complex<f32> = bytemuck::cast(real_lhs);
            let rhs: Complex<f32> = bytemuck::cast(real_rhs);
            unsafe { core::mem::transmute_copy(&(lhs.re <= rhs.re)) }
        } else {
            panic!();
        }
    }
    #[inline(always)]
    fn simd_complex_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        Self::simd_complex_less_than(ctx, real_rhs, real_lhs)
    }
    #[inline(always)]
    fn simd_complex_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        Self::simd_complex_less_than_or_equal(ctx, real_rhs, real_lhs)
    }

    #[inline(always)]
    fn simd_complex_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            bytemuck::cast(ctx.select_f32s_m32s(mask, bytemuck::cast(lhs), bytemuck::cast(rhs)))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            assert!(const { size_of::<S::m32s>() == size_of::<bool>() });
            let mask: bool = unsafe { core::mem::transmute_copy(&mask) };
            let lhs: Complex<f32> = bytemuck::cast(lhs);
            let rhs: Complex<f32> = bytemuck::cast(rhs);
            bytemuck::cast(if mask { lhs } else { rhs })
        } else {
            panic!();
        }
    }

    #[inline(always)]
    unsafe fn simd_complex_mask_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *const Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        ctx.mask_load_ptr_c32s(mask, ptr as *const Complex<f32>)
    }
    #[inline(always)]
    unsafe fn simd_complex_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *mut Self::SimdComplexUnit<S>,
        value: Self::SimdComplexUnit<S>,
    ) {
        ctx.mask_store_ptr_c32s(mask, ptr as *mut Complex<f32>, value)
    }

    #[inline(always)]
    fn simd_complex_iota<S: Simd>(_: &Self::SimdCtx<S>) -> Self::SimdIndex<S> {
        const {
            core::assert!(
                size_of::<Self::Index>() * Self::Index::COMPLEX_IOTA.len()
                    >= size_of::<Self::SimdIndex<S>>()
            )
        };
        unsafe { core::mem::transmute_copy(Self::Index::COMPLEX_IOTA) }
    }

    #[inline(always)]
    fn simd_complex_load<S: Simd>(
        _: &Self::SimdCtx<S>,
        ptr: &Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        *ptr
    }
    #[inline(always)]
    fn simd_complex_store<S: Simd>(
        _: &Self::SimdCtx<S>,
        ptr: &mut Self::SimdComplexUnit<S>,
        value: Self::SimdComplexUnit<S>,
    ) {
        *ptr = value;
    }

    #[inline(always)]
    fn simd_complex_abs_max<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.abs_max_c32s(value)
    }
}

impl EnableComplex for f64 {
    const COMPLEX_SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::All;
    type SimdComplexUnit<S: Simd> = S::c64s;
    type Arch = pulp::Arch;

    #[inline(always)]
    fn simd_complex_splat<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: &Complex<Self>,
    ) -> Self::SimdComplexUnit<S> {
        simd.splat_c64s(*value)
    }
    #[inline(always)]
    fn simd_complex_splat_real<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: &Self,
    ) -> Self::SimdComplexUnit<S> {
        simd.splat_c64s(Complex {
            re: *value,
            im: *value,
        })
    }

    #[inline(always)]
    fn simd_complex_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.add_c64s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_complex_sub<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.sub_c64s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_complex_neg<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.neg_c64s(value)
    }
    #[inline(always)]
    fn simd_complex_conj<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.conj_c64s(value)
    }
    #[inline(always)]
    fn simd_complex_abs1<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            bytemuck::cast(simd.abs_f64s(bytemuck::cast(value)))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            let value: Complex<f64> = bytemuck::cast(value);
            let abs = value.re.abs() + value.im.abs();
            bytemuck::cast(Complex { re: abs, im: abs })
        } else {
            panic!();
        }
    }
    #[inline(always)]
    fn simd_complex_abs2<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.abs2_c64s(value)
    }
    #[inline(always)]
    fn simd_complex_abs2_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
        acc: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.add_c64s(simd.abs2_c64s(value), acc)
    }
    #[inline(always)]
    fn simd_complex_mul<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.mul_e_c64s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_complex_conj_mul<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.conj_mul_e_c64s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_complex_mul_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
        acc: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.mul_add_e_c64s(lhs, rhs, acc)
    }
    #[inline(always)]
    fn simd_complex_conj_mul_add<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
        acc: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.conj_mul_add_e_c64s(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_complex_mul_real<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            bytemuck::cast(simd.mul_f64s(bytemuck::cast(lhs), bytemuck::cast(real_rhs)))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            let mut lhs: Complex<f64> = bytemuck::cast(lhs);
            let rhs: Complex<f64> = bytemuck::cast(real_rhs);
            lhs *= rhs.re;
            bytemuck::cast(lhs)
        } else {
            panic!();
        }
    }
    #[inline(always)]
    fn simd_complex_mul_pow2<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        Self::simd_complex_mul_real(simd, lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_complex_reduce_sum<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Complex<Self> {
        simd.reduce_sum_c64s(value)
    }
    #[inline(always)]
    fn simd_complex_reduce_max<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            bytemuck::cast(simd.reduce_max_f64s(bytemuck::cast(value)))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            let tmp = bytemuck::cast::<_, Complex<f64>>(value);
            if tmp.re > tmp.im {
                tmp.re
            } else {
                tmp.im
            }
        } else {
            panic!()
        }
    }

    #[inline(always)]
    fn simd_complex_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            ctx.less_than_f64s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            assert!(const { size_of::<S::m64s>() == size_of::<bool>() });

            let lhs: Complex<f64> = bytemuck::cast(real_lhs);
            let rhs: Complex<f64> = bytemuck::cast(real_rhs);
            unsafe { core::mem::transmute_copy(&(lhs.re < rhs.re)) }
        } else {
            panic!();
        }
    }
    #[inline(always)]
    fn simd_complex_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            ctx.less_than_or_equal_f64s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            assert!(const { size_of::<S::m64s>() == size_of::<bool>() });

            let lhs: Complex<f64> = bytemuck::cast(real_lhs);
            let rhs: Complex<f64> = bytemuck::cast(real_rhs);
            unsafe { core::mem::transmute_copy(&(lhs.re <= rhs.re)) }
        } else {
            panic!();
        }
    }
    #[inline(always)]
    fn simd_complex_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        Self::simd_complex_less_than(ctx, real_rhs, real_lhs)
    }
    #[inline(always)]
    fn simd_complex_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdComplexUnit<S>,
        real_rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        Self::simd_complex_less_than_or_equal(ctx, real_rhs, real_lhs)
    }

    #[inline(always)]
    fn simd_complex_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdComplexUnit<S>,
        rhs: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            bytemuck::cast(ctx.select_f64s_m64s(mask, bytemuck::cast(lhs), bytemuck::cast(rhs)))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            assert!(const { size_of::<S::m64s>() == size_of::<bool>() });
            let mask: bool = unsafe { core::mem::transmute_copy(&mask) };
            let lhs: Complex<f64> = bytemuck::cast(lhs);
            let rhs: Complex<f64> = bytemuck::cast(rhs);
            bytemuck::cast(if mask { lhs } else { rhs })
        } else {
            panic!();
        }
    }

    #[inline(always)]
    unsafe fn simd_complex_mask_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *const Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        ctx.mask_load_ptr_c64s(mask, ptr as *const Complex<f64>)
    }
    #[inline(always)]
    unsafe fn simd_complex_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *mut Self::SimdComplexUnit<S>,
        value: Self::SimdComplexUnit<S>,
    ) {
        ctx.mask_store_ptr_c64s(mask, ptr as *mut Complex<f64>, value)
    }

    #[inline(always)]
    fn simd_complex_iota<S: Simd>(_: &Self::SimdCtx<S>) -> Self::SimdIndex<S> {
        const {
            core::assert!(
                size_of::<Self::Index>() * Self::Index::COMPLEX_IOTA.len()
                    >= size_of::<Self::SimdIndex<S>>()
            )
        };
        unsafe { core::mem::transmute_copy(Self::Index::COMPLEX_IOTA) }
    }

    #[inline(always)]
    fn simd_complex_load<S: Simd>(
        _: &Self::SimdCtx<S>,
        ptr: &Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        *ptr
    }
    #[inline(always)]
    fn simd_complex_store<S: Simd>(
        _: &Self::SimdCtx<S>,
        ptr: &mut Self::SimdComplexUnit<S>,
        value: Self::SimdComplexUnit<S>,
    ) {
        *ptr = value;
    }

    #[inline(always)]
    fn simd_complex_abs_max<S: Simd>(
        simd: &Self::SimdCtx<S>,
        value: Self::SimdComplexUnit<S>,
    ) -> Self::SimdComplexUnit<S> {
        simd.abs_max_c64s(value)
    }
}

pub mod hacks {
    use core::marker::PhantomData;

    pub use generativity::Id;

    #[doc(hidden)]
    pub struct LifetimeBrand<'id> {
        phantom: PhantomData<&'id Id<'id>>,
    }

    #[inline]
    pub unsafe fn make_guard_pair<'a>(
        id: &'a Id<'a>,
    ) -> (LifetimeBrand<'a>, generativity::Guard<'a>) {
        (
            LifetimeBrand {
                phantom: PhantomData,
            },
            generativity::Guard::new(*id),
        )
    }

    #[doc(hidden)]
    pub struct NonCopy;

    impl Drop for NonCopy {
        #[inline(always)]
        fn drop(&mut self) {}
    }

    #[doc(hidden)]
    pub struct UseLifetime<'a>(::core::marker::PhantomData<&'a fn(&'a ()) -> &'a ()>);
    impl ::core::ops::Drop for UseLifetime<'_> {
        #[inline(always)]
        fn drop(&mut self) {}
    }
    #[doc(hidden)]
    #[inline(always)]
    pub fn __with_lifetime_of(_: &mut NonCopy) -> UseLifetime<'_> {
        UseLifetime(::core::marker::PhantomData)
    }

    pub use generativity::make_guard;

    pub struct GhostNode<'scope, 'a, T> {
        pub child: T,
        marker: PhantomData<(fn(&'a ()) -> &'a (), fn(&'scope ()) -> &'scope ())>,
    }

    impl<'scope, 'a, T> GhostNode<'scope, 'a, T> {
        #[inline]
        pub fn new(inner: T, _: &generativity::Guard<'scope>, _: &UseLifetime<'a>) -> Self {
            Self {
                child: inner,
                marker: PhantomData,
            }
        }

        #[inline]
        pub unsafe fn new_unbound(inner: T) -> Self {
            Self {
                child: inner,
                marker: PhantomData,
            }
        }
    }
}
