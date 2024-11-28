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
    pub fn add<T: AddByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
        lhs.add_by_ref(rhs)
    }
    #[inline(always)]
    #[track_caller]
    pub fn sub<T: SubByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
        lhs.sub_by_ref(rhs)
    }
    #[inline(always)]
    pub fn mul<T: MulByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
        lhs.mul_by_ref(rhs)
    }
    #[inline(always)]
    pub fn div<T: DivByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
        lhs.div_by_ref(rhs)
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
    pub fn absmax<T: ComplexField>(value: &T) -> T::Real {
        if const { T::IS_REAL } {
            T::abs1_impl(value)
        } else {
            add(
                &T::Real::abs1_impl(&real(value)),
                &T::Real::abs1_impl(&imag(value)),
            )
        }
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

impl<Rhs, Lhs, Output> AddByRef<Rhs> for Lhs
where
    for<'a, 'b> &'a Lhs: core::ops::Add<&'b Rhs, Output = Output>,
{
    type Output = Output;

    #[inline]
    fn add_by_ref(&self, rhs: &Rhs) -> Self::Output {
        self + rhs
    }
}
impl<Rhs, Lhs, Output> SubByRef<Rhs> for Lhs
where
    for<'a, 'b> &'a Lhs: core::ops::Sub<&'b Rhs, Output = Output>,
{
    type Output = Output;

    #[inline]
    #[track_caller]
    fn sub_by_ref(&self, rhs: &Rhs) -> Self::Output {
        self - rhs
    }
}
impl<Rhs, Lhs, Output> MulByRef<Rhs> for Lhs
where
    for<'a, 'b> &'a Lhs: core::ops::Mul<&'b Rhs, Output = Output>,
{
    type Output = Output;

    #[inline]
    fn mul_by_ref(&self, rhs: &Rhs) -> Self::Output {
        self * rhs
    }
}
impl<Rhs, Lhs, Output> DivByRef<Rhs> for Lhs
where
    for<'a, 'b> &'a Lhs: core::ops::Div<&'b Rhs, Output = Output>,
{
    type Output = Output;

    #[inline]
    fn div_by_ref(&self, rhs: &Rhs) -> Self::Output {
        self / rhs
    }
}

impl<T, Output> NegByRef for T
where
    for<'a> &'a T: core::ops::Neg<Output = Output>,
{
    type Output = Output;

    #[inline]
    fn neg_by_ref(&self) -> Self::Output {
        -self
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
pub struct RealMarker<T>(pub T);

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
    pub fn splat_real(&self, value: &T::Real) -> RealMarker<T::SimdVec<S>> {
        RealMarker(unsafe {
            core::mem::transmute_copy(&T::simd_splat_real(&self.0, (value).by_ref()))
        })
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
    pub fn abs1(&self, value: T::SimdVec<S>) -> RealMarker<T::SimdVec<S>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        RealMarker(unsafe { core::mem::transmute_copy(&T::simd_abs1(&self.0, value)) })
    }
    #[inline(always)]
    pub fn abs_max(&self, value: T::SimdVec<S>) -> RealMarker<T::SimdVec<S>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        RealMarker(unsafe { core::mem::transmute_copy(&T::simd_abs_max(&self.0, value)) })
    }

    #[inline(always)]
    pub fn mul_real(&self, lhs: T::SimdVec<S>, rhs: RealMarker<T::SimdVec<S>>) -> T::SimdVec<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_mul_real(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn mul_pow2(&self, lhs: T::SimdVec<S>, rhs: RealMarker<T::SimdVec<S>>) -> T::SimdVec<S> {
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
    pub fn abs2(&self, value: T::SimdVec<S>) -> RealMarker<T::SimdVec<S>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        RealMarker(unsafe { core::mem::transmute_copy(&T::simd_abs2(&self.0, value)) })
    }

    #[inline(always)]
    pub fn abs2_add(
        &self,
        value: T::SimdVec<S>,
        acc: RealMarker<T::SimdVec<S>>,
    ) -> RealMarker<T::SimdVec<S>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        let acc = unsafe { core::mem::transmute_copy(&acc) };
        RealMarker(unsafe { core::mem::transmute_copy(&T::simd_abs2_add(&self.0, value, acc)) })
    }

    #[inline(always)]
    pub fn reduce_sum(&self, value: T::SimdVec<S>) -> T {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_reduce_sum(&self.0, value)) }
    }
    #[inline(always)]
    pub fn reduce_max(&self, value: RealMarker<T::SimdVec<S>>) -> T {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_reduce_max(&self.0, value)) }
    }

    #[faer_macros::math]
    #[inline(always)]
    pub fn reduce_sum_real(&self, value: RealMarker<T::SimdVec<S>>) -> Real<T> {
        let value = T::simd_reduce_sum(&self.0, value.0);
        if const { T::SIMD_ABS_SPLIT_REAL_IMAG && !S::IS_SCALAR } {
            add(real(value), imag(value))
        } else {
            real(value)
        }
    }
    #[faer_macros::math]
    #[inline(always)]
    pub fn reduce_max_real(&self, value: RealMarker<T::SimdVec<S>>) -> Real<T> {
        let value = T::simd_reduce_max(&self.0, value.0);
        if const { T::SIMD_ABS_SPLIT_REAL_IMAG && !S::IS_SCALAR } {
            max(real(value), imag(value))
        } else {
            real(value)
        }
    }

    #[inline(always)]
    pub fn max(
        &self,
        lhs: RealMarker<T::SimdVec<S>>,
        rhs: RealMarker<T::SimdVec<S>>,
    ) -> RealMarker<T::SimdVec<S>> {
        let cmp = self.gt(lhs, rhs);
        RealMarker(self.select(cmp, lhs.0, rhs.0))
    }

    #[inline(always)]
    pub fn lt(
        &self,
        lhs: RealMarker<T::SimdVec<S>>,
        rhs: RealMarker<T::SimdVec<S>>,
    ) -> T::SimdMask<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_less_than(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn gt(
        &self,
        lhs: RealMarker<T::SimdVec<S>>,
        rhs: RealMarker<T::SimdVec<S>>,
    ) -> T::SimdMask<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_greater_than(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn le(
        &self,
        lhs: RealMarker<T::SimdVec<S>>,
        rhs: RealMarker<T::SimdVec<S>>,
    ) -> T::SimdMask<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_less_than_or_equal(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn ge(
        &self,
        lhs: RealMarker<T::SimdVec<S>>,
        rhs: RealMarker<T::SimdVec<S>>,
    ) -> T::SimdMask<S> {
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
        mask: T::SimdMemMask<S>,
        ptr: *const T::SimdVec<S>,
    ) -> T::SimdVec<S> {
        unsafe { T::simd_mask_load(&self.0, mask, ptr) }
    }
    #[inline(always)]
    pub unsafe fn mask_store(
        &self,
        mask: T::SimdMemMask<S>,
        ptr: *mut T::SimdVec<S>,
        value: T::SimdVec<S>,
    ) {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { T::simd_mask_store(&self.0, mask, ptr, value) }
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

pub type Real<T> = <<T as Conjugate>::Canonical as ComplexField>::Real;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ComplexConj<T> {
    pub re: T,
    pub im_neg: T,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdCapabilities {
    None,
    Copy,
    Simd,
}

impl SimdCapabilities {
    #[inline]
    pub const fn is_copy(self) -> bool {
        matches!(self, Self::Copy | Self::Simd)
    }

    #[inline]
    pub const fn is_simd(self) -> bool {
        matches!(self, Self::Simd)
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

pub trait EnableComplex: Sized + RealField + Default {}

unsafe impl<T> Conjugate for Complex<T>
where
    Complex<T>: ComplexField,
{
    const IS_CANONICAL: bool = true;
    type Conj = ComplexConj<T>;
    type Canonical = Complex<T>;
}
unsafe impl<T> Conjugate for ComplexConj<T>
where
    Complex<T>: ComplexField,
{
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
    const SIMD_ABS_SPLIT_REAL_IMAG: bool = false;

    type Arch: SimdArch;
    type Unit: ComplexField;

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
    type SimdMemMask<S: Simd>: Copy + Debug;

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

    fn simd_mask_between<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        start: Self::Index,
        end: Self::Index,
    ) -> Self::SimdMemMask<S>;
    unsafe fn simd_mask_load_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *const Self::SimdVec<S>,
    ) -> Self::SimdVec<S>;
    unsafe fn simd_mask_store_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *mut Self::SimdVec<S>,
        values: Self::SimdVec<S>,
    );

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
    fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self;
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

    fn simd_and_mask<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdMask<S>,
        rhs: Self::SimdMask<S>,
    ) -> Self::SimdMask<S>;
    fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize;

    #[inline(always)]
    fn simd_load<S: Simd>(ctx: &Self::SimdCtx<S>, ptr: &Self::SimdVec<S>) -> Self::SimdVec<S> {
        let simd = Self::ctx_from_simd(ctx);
        if const { Self::Unit::IS_NATIVE_F32 } {
            simd.deinterleave_shfl_f32s(*ptr)
        } else if const { Self::Unit::IS_NATIVE_F64 } {
            simd.deinterleave_shfl_f64s(*ptr)
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: &mut Self::SimdVec<S>,
        value: Self::SimdVec<S>,
    ) {
        let simd = Self::ctx_from_simd(ctx);
        if const { Self::Unit::IS_NATIVE_F32 } {
            *ptr = simd.deinterleave_shfl_f32s(value)
        } else if const { Self::Unit::IS_NATIVE_F64 } {
            *ptr = simd.deinterleave_shfl_f64s(value)
        } else {
            panic!();
        }
    }

    #[inline(always)]
    unsafe fn simd_mask_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *const Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        let simd = Self::ctx_from_simd(ctx);
        let value = Self::simd_mask_load_raw(ctx, mask, ptr);
        if const { Self::Unit::IS_NATIVE_F32 } {
            simd.deinterleave_shfl_f32s(value)
        } else if const { Self::Unit::IS_NATIVE_F64 } {
            simd.deinterleave_shfl_f64s(value)
        } else {
            panic!();
        }
    }

    #[inline(always)]
    unsafe fn simd_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *mut Self::SimdVec<S>,
        value: Self::SimdVec<S>,
    ) {
        let simd = Self::ctx_from_simd(ctx);
        if const { Self::Unit::IS_NATIVE_F32 } {
            Self::simd_mask_store_raw(ctx, mask, ptr, simd.deinterleave_shfl_f32s(value))
        } else if const { Self::Unit::IS_NATIVE_F64 } {
            Self::simd_mask_store_raw(ctx, mask, ptr, simd.deinterleave_shfl_f64s(value))
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_iota<S: Simd>(ctx: &Self::SimdCtx<S>) -> Self::SimdIndex<S> {
        let simd = Self::ctx_from_simd(ctx);
        struct Interleave<T>(T);
        unsafe impl<T> pulp::Interleave for Interleave<T> {}

        unsafe {
            if const { Self::Unit::IS_NATIVE_F32 } {
                core::mem::transmute_copy::<_, Self::SimdIndex<S>>(&simd.deinterleave_shfl_f32s(
                    Interleave(core::mem::transmute_copy::<_, Self::SimdVec<S>>(
                        &<Self as pulp::Iota32>::IOTA,
                    )),
                ))
            } else if const { Self::Unit::IS_NATIVE_F64 } {
                core::mem::transmute_copy::<_, Self::SimdIndex<S>>(&simd.deinterleave_shfl_f64s(
                    core::mem::transmute_copy::<_, Self::SimdVec<S>>(&<Self as pulp::Iota64>::IOTA),
                ))
            } else {
                panic!();
            }
        }
    }
}

pub trait RealField:
    ComplexField<Real = Self, Conj = Self>
    + DivByRef<Output = Self>
    + PartialOrd
    + num_traits::NumOps
    + num_traits::Num
    + core::ops::Neg<Output = Self>
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
    type Unit = Self;
    type Arch = pulp::Arch;

    const IS_NATIVE_F32: bool = true;

    const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::Simd;

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
    fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
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

    type SimdMemMask<S: Simd> = pulp::MemMask<S::m32s>;
    #[inline(always)]
    fn simd_mask_between<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        start: u32,
        end: u32,
    ) -> Self::SimdMemMask<S> {
        ctx.mask_between_m32s(start, end)
    }
    #[inline(always)]
    unsafe fn simd_mask_load_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *const Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mask_load_ptr_f32s(mask, ptr as _)
    }
    #[inline(always)]
    unsafe fn simd_mask_store_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *mut Self::SimdVec<S>,
        values: Self::SimdVec<S>,
    ) {
        ctx.mask_store_ptr_f32s(mask, ptr as _, values);
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
    type Unit = Self;
    type Arch = pulp::Arch;

    const IS_NATIVE_F64: bool = true;

    const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::Simd;

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
    fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
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

    type SimdMemMask<S: Simd> = pulp::MemMask<S::m64s>;
    #[inline(always)]
    fn simd_mask_between<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        start: u64,
        end: u64,
    ) -> Self::SimdMemMask<S> {
        ctx.mask_between_m64s(start, end)
    }
    #[inline(always)]
    unsafe fn simd_mask_load_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *const Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mask_load_ptr_f64s(mask, ptr as _)
    }
    #[inline(always)]
    unsafe fn simd_mask_store_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *mut Self::SimdVec<S>,
        values: Self::SimdVec<S>,
    ) {
        ctx.mask_store_ptr_f64s(mask, ptr as _, values);
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

impl<T: EnableComplex> ComplexField for Complex<T> {
    const IS_REAL: bool = false;

    type Arch = T::Arch;
    type SimdCtx<S: Simd> = T::SimdCtx<S>;
    type Index = T::Index;
    type Unit = T::Unit;
    type Real = T;

    const SIMD_CAPABILITIES: SimdCapabilities = T::SIMD_CAPABILITIES;

    type SimdMask<S: Simd> = T::SimdMask<S>;
    type SimdVec<S: Simd> = Complex<T::SimdVec<S>>;
    type SimdIndex<S: Simd> = T::SimdIndex<S>;

    #[inline]
    fn zero_impl() -> Self {
        Complex {
            re: T::zero_impl(),
            im: T::zero_impl(),
        }
    }

    #[inline]
    fn one_impl() -> Self {
        Complex {
            re: T::one_impl(),
            im: T::zero_impl(),
        }
    }

    #[inline]
    fn nan_impl() -> Self {
        Complex {
            re: T::nan_impl(),
            im: T::nan_impl(),
        }
    }

    #[inline]
    fn infinity_impl() -> Self {
        Complex {
            re: T::infinity_impl(),
            im: T::infinity_impl(),
        }
    }

    #[inline]
    fn from_real_impl(real: &Self::Real) -> Self {
        Complex {
            re: real.clone(),
            im: T::zero_impl(),
        }
    }

    #[inline]
    fn from_f64_impl(real: f64) -> Self {
        Complex {
            re: T::from_f64_impl(real),
            im: T::zero_impl(),
        }
    }

    #[inline]
    fn real_part_impl(value: &Self) -> Self::Real {
        value.re.clone()
    }

    #[inline]
    fn imag_part_impl(value: &Self) -> Self::Real {
        value.im.clone()
    }

    #[inline]
    fn copy_impl(value: &Self) -> Self {
        value.clone()
    }

    #[inline]
    fn conj_impl(value: &Self) -> Self {
        Self {
            re: value.re.clone(),
            im: value.im.neg_by_ref(),
        }
    }

    #[inline]
    fn recip_impl(value: &Self) -> Self {
        let (re, im) = recip_impl(value.re.clone(), value.im.clone());
        Complex { re, im }
    }

    #[inline]
    fn sqrt_impl(value: &Self) -> Self {
        let (re, im) = sqrt_impl(value.re.clone(), value.im.clone());
        Complex { re, im }
    }

    #[inline]
    fn abs_impl(value: &Self) -> Self::Real {
        abs_impl(value.re.clone(), value.im.clone())
    }

    #[inline]
    #[faer_macros::math]
    fn abs1_impl(value: &Self) -> Self::Real {
        abs1(value.re) + abs1(value.im)
    }

    #[inline]
    #[faer_macros::math]
    fn abs2_impl(value: &Self) -> Self::Real {
        abs2(value.re) + abs2(value.im)
    }

    #[inline]
    #[faer_macros::math]
    fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self {
        Complex {
            re: lhs.re * rhs,
            im: lhs.im * rhs,
        }
    }

    #[inline]
    #[faer_macros::math]
    fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self {
        Complex {
            re: mul_pow2(lhs.re, rhs),
            im: mul_pow2(lhs.im, rhs),
        }
    }

    #[inline]
    #[faer_macros::math]
    fn is_finite_impl(value: &Self) -> bool {
        is_finite(value.re) && is_finite(value.im)
    }

    #[inline(always)]
    fn simd_ctx<S: Simd>(simd: S) -> Self::SimdCtx<S> {
        T::simd_ctx(simd)
    }

    #[inline(always)]
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> S {
        T::ctx_from_simd(ctx)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_splat(ctx, &value.re),
            im: T::simd_splat(ctx, &value.im),
        }
    }

    #[inline(always)]
    fn simd_splat_real<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self::Real) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_splat_real(ctx, value),
            im: T::simd_splat_real(ctx, value),
        }
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_add(ctx, lhs.re, rhs.re),
            im: T::simd_add(ctx, lhs.im, rhs.im),
        }
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_sub(ctx, lhs.re, rhs.re),
            im: T::simd_sub(ctx, lhs.im, rhs.im),
        }
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_neg(ctx, value.re),
            im: T::simd_neg(ctx, value.im),
        }
    }

    #[inline(always)]
    fn simd_conj<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        Complex {
            re: value.re,
            im: T::simd_neg(ctx, value.im),
        }
    }

    #[inline(always)]
    fn simd_abs1<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        let v = T::simd_add(
            ctx,
            T::simd_abs1(ctx, value.re),
            T::simd_abs1(ctx, value.im),
        );
        Complex { re: v, im: v }
    }

    #[inline(always)]
    fn simd_abs_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        let re = T::simd_abs_max(ctx, value.re);
        let im = T::simd_abs_max(ctx, value.im);

        let v = T::simd_select(ctx, T::simd_greater_than(ctx, re, im), re, im);
        Complex { re: v, im: v }
    }

    #[inline(always)]
    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_mul_real(ctx, lhs.re, real_rhs.re),
            im: T::simd_mul_real(ctx, lhs.im, real_rhs.re),
        }
    }

    #[inline(always)]
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_mul_pow2(ctx, lhs.re, real_rhs.re),
            im: T::simd_mul_pow2(ctx, lhs.im, real_rhs.re),
        }
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_mul_add(
                ctx,
                lhs.re,
                rhs.re,
                T::simd_neg(ctx, T::simd_mul(ctx, lhs.im, rhs.im)),
            ),
            im: T::simd_mul_add(ctx, lhs.re, rhs.im, T::simd_mul(ctx, lhs.im, rhs.re)),
        }
    }

    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_mul_add(ctx, lhs.re, rhs.re, T::simd_mul(ctx, lhs.im, rhs.im)),
            im: T::simd_mul_add(
                ctx,
                lhs.re,
                rhs.im,
                T::simd_neg(ctx, T::simd_mul(ctx, lhs.im, rhs.re)),
            ),
        }
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_mul_add(
                ctx,
                T::simd_neg(ctx, lhs.im),
                rhs.im,
                T::simd_mul_add(ctx, lhs.re, rhs.re, acc.re),
            ),
            im: T::simd_mul_add(
                ctx,
                lhs.re,
                rhs.im,
                T::simd_mul_add(ctx, lhs.im, rhs.re, acc.im),
            ),
        }
    }

    #[inline(always)]
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_mul_add(
                ctx,
                lhs.im,
                rhs.im,
                T::simd_mul_add(ctx, lhs.re, rhs.re, acc.re),
            ),
            im: T::simd_mul_add(
                ctx,
                lhs.re,
                rhs.im,
                T::simd_mul_add(ctx, T::simd_neg(ctx, lhs.im), rhs.re, acc.im),
            ),
        }
    }

    #[inline(always)]
    fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        let v = T::simd_abs2_add(ctx, value.re, T::simd_abs2(ctx, value.im));
        Complex { re: v, im: v }
    }

    #[inline(always)]
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        let v = T::simd_abs2_add(ctx, value.re, T::simd_abs2_add(ctx, value.im, acc.re));
        Complex { re: v, im: v }
    }

    #[inline(always)]
    fn simd_reduce_sum<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
        Complex {
            re: T::simd_reduce_sum(ctx, value.re),
            im: T::simd_reduce_sum(ctx, value.im),
        }
    }

    #[inline(always)]
    fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
        Complex {
            re: T::simd_reduce_max(ctx, value.re),
            im: T::simd_reduce_max(ctx, value.im),
        }
    }

    #[inline(always)]
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        T::simd_less_than(ctx, real_lhs.re, real_rhs.re)
    }

    #[inline(always)]
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        T::simd_less_than_or_equal(ctx, real_lhs.re, real_rhs.re)
    }

    #[inline(always)]
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        T::simd_greater_than(ctx, real_lhs.re, real_rhs.re)
    }

    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        T::simd_greater_than_or_equal(ctx, real_lhs.re, real_rhs.re)
    }

    #[inline(always)]
    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_select(ctx, mask, lhs.re, rhs.re),
            im: T::simd_select(ctx, mask, lhs.im, rhs.im),
        }
    }

    #[inline(always)]
    fn simd_index_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        T::simd_index_select(ctx, mask, lhs, rhs)
    }

    #[inline(always)]
    fn simd_index_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::Index) -> Self::SimdIndex<S> {
        T::simd_index_splat(ctx, value)
    }

    #[inline(always)]
    fn simd_index_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        T::simd_index_add(ctx, lhs, rhs)
    }

    #[inline(always)]
    fn simd_and_mask<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdMask<S>,
        rhs: Self::SimdMask<S>,
    ) -> Self::SimdMask<S> {
        T::simd_and_mask(ctx, lhs, rhs)
    }

    #[inline(always)]
    fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
        T::simd_first_true_mask(ctx, value)
    }

    type SimdMemMask<S: Simd> = Complex<T::SimdMemMask<S>>;

    #[inline(always)]
    fn simd_mask_between<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        start: Self::Index,
        end: Self::Index,
    ) -> Self::SimdMemMask<S> {
        let n = const { size_of::<Self::SimdVec<S>>() / size_of::<Self>() };
        let start = start.zx() * 2;
        let end = end.zx() * 2;

        let re = T::simd_mask_between(
            ctx,
            Self::Index::truncate(start.min(n)),
            Self::Index::truncate(end.min(n)),
        );
        let im = T::simd_mask_between(
            ctx,
            Self::Index::truncate(start.max(n) - n),
            Self::Index::truncate(end.max(n) - n),
        );
        Complex { re, im }
    }

    #[inline(always)]
    unsafe fn simd_mask_load_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *const Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Complex {
            re: T::simd_mask_load_raw(ctx, mask.re, &raw const (*ptr).re),
            im: T::simd_mask_load_raw(ctx, mask.im, &raw const (*ptr).im),
        }
    }

    #[inline(always)]
    unsafe fn simd_mask_store_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *mut Self::SimdVec<S>,
        values: Self::SimdVec<S>,
    ) {
        T::simd_mask_store_raw(ctx, mask.re, &raw mut (*ptr).re, values.re);
        T::simd_mask_store_raw(ctx, mask.im, &raw mut (*ptr).im, values.im);
    }
}

impl ComplexField for Complex<f32> {
    const IS_REAL: bool = false;
    const SIMD_ABS_SPLIT_REAL_IMAG: bool = true;

    type Arch = pulp::Arch;
    type SimdCtx<S: Simd> = S;
    type Index = u32;
    type Unit = f32;
    type Real = f32;

    const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::Simd;

    type SimdMask<S: Simd> = S::m32s;
    type SimdVec<S: Simd> = S::c32s;
    type SimdIndex<S: Simd> = S::u32s;

    #[inline]
    fn zero_impl() -> Self {
        Complex {
            re: f32::zero_impl(),
            im: f32::zero_impl(),
        }
    }

    #[inline]
    fn one_impl() -> Self {
        Complex {
            re: f32::one_impl(),
            im: f32::zero_impl(),
        }
    }

    #[inline]
    fn nan_impl() -> Self {
        Complex {
            re: f32::nan_impl(),
            im: f32::nan_impl(),
        }
    }

    #[inline]
    fn infinity_impl() -> Self {
        Complex {
            re: f32::infinity_impl(),
            im: f32::infinity_impl(),
        }
    }

    #[inline]
    fn from_real_impl(real: &Self::Real) -> Self {
        Complex {
            re: real.clone(),
            im: f32::zero_impl(),
        }
    }

    #[inline]
    fn from_f64_impl(real: f64) -> Self {
        Complex {
            re: f32::from_f64_impl(real),
            im: f32::zero_impl(),
        }
    }

    #[inline]
    fn real_part_impl(value: &Self) -> Self::Real {
        value.re.clone()
    }

    #[inline]
    fn imag_part_impl(value: &Self) -> Self::Real {
        value.im.clone()
    }

    #[inline]
    fn copy_impl(value: &Self) -> Self {
        value.clone()
    }

    #[inline]
    fn conj_impl(value: &Self) -> Self {
        Self {
            re: value.re.clone(),
            im: value.im.neg_by_ref(),
        }
    }

    #[inline]
    fn recip_impl(value: &Self) -> Self {
        let (re, im) = recip_impl(value.re.clone(), value.im.clone());
        Complex { re, im }
    }

    #[inline]
    fn sqrt_impl(value: &Self) -> Self {
        let (re, im) = sqrt_impl(value.re.clone(), value.im.clone());
        Complex { re, im }
    }

    #[inline]
    fn abs_impl(value: &Self) -> Self::Real {
        abs_impl(value.re.clone(), value.im.clone())
    }

    #[inline]
    #[faer_macros::math]
    fn abs1_impl(value: &Self) -> Self::Real {
        abs1(value.re) + abs1(value.im)
    }

    #[inline]
    #[faer_macros::math]
    fn abs2_impl(value: &Self) -> Self::Real {
        abs2(value.re) + abs2(value.im)
    }

    #[inline]
    #[faer_macros::math]
    fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self {
        Complex {
            re: lhs.re * *rhs,
            im: lhs.im * *rhs,
        }
    }

    #[inline]
    #[faer_macros::math]
    fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self {
        Complex {
            re: mul_pow2(lhs.re, rhs),
            im: mul_pow2(lhs.im, rhs),
        }
    }

    #[inline]
    #[faer_macros::math]
    fn is_finite_impl(value: &Self) -> bool {
        is_finite(value.re) && is_finite(value.im)
    }

    #[inline(always)]
    fn simd_ctx<S: Simd>(simd: S) -> Self::SimdCtx<S> {
        f32::simd_ctx(simd)
    }

    #[inline(always)]
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> S {
        f32::ctx_from_simd(ctx)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self) -> Self::SimdVec<S> {
        ctx.splat_c32s(*value)
    }

    #[inline(always)]
    fn simd_splat_real<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self::Real) -> Self::SimdVec<S> {
        ctx.splat_c32s(Complex {
            re: *value,
            im: *value,
        })
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.add_c32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.sub_c32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.neg_c32s(value)
    }

    #[inline(always)]
    fn simd_conj<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.conj_c32s(value)
    }

    #[inline(always)]
    fn simd_abs1<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            bytemuck::cast(ctx.abs_f32s(bytemuck::cast(value)))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            let value: Complex<f32> = bytemuck::cast(value);
            let v = value.re.abs() + value.im.abs();
            bytemuck::cast(Complex { re: v, im: v })
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_abs_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            bytemuck::cast(ctx.abs_f32s(bytemuck::cast(value)))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            let value: Complex<f32> = bytemuck::cast(value);
            let re = value.re.abs();
            let im = value.im.abs();
            let v = if re > im { re } else { im };
            bytemuck::cast(Complex { re: v, im: v })
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            bytemuck::cast(ctx.mul_f32s(bytemuck::cast(lhs), bytemuck::cast(real_rhs)))
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
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Self::simd_mul_real(ctx, lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_e_c32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.conj_mul_e_c32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_add_e_c32s(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.conj_mul_add_e_c32s(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            bytemuck::cast(ctx.mul_f32s(bytemuck::cast(value), bytemuck::cast(value)))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            let value: Complex<f32> = bytemuck::cast(value);
            let v = value.re * value.re + value.im * value.im;
            bytemuck::cast(Complex { re: v, im: v })
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            bytemuck::cast(ctx.mul_add_f32s(
                bytemuck::cast(value),
                bytemuck::cast(value),
                bytemuck::cast(acc),
            ))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            let value: Complex<f32> = bytemuck::cast(value);
            let acc: Complex<f32> = bytemuck::cast(acc);
            let v = value.re * value.re + value.im * value.im + acc.re;
            bytemuck::cast(Complex { re: v, im: v })
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_reduce_sum<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
        ctx.reduce_sum_c32s(value)
    }

    #[inline(always)]
    fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
        ctx.reduce_max_c32s(value)
    }

    #[inline(always)]
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
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
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
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
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            ctx.greater_than_f32s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            assert!(const { size_of::<S::m32s>() == size_of::<bool>() });

            let lhs: Complex<f32> = bytemuck::cast(real_lhs);
            let rhs: Complex<f32> = bytemuck::cast(real_rhs);
            unsafe { core::mem::transmute_copy(&(lhs.re > rhs.re)) }
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        if const { size_of::<S::c32s>() == size_of::<S::f32s>() } {
            ctx.greater_than_or_equal_f32s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
        } else if const { size_of::<S::c32s>() == size_of::<Complex<f32>>() } {
            assert!(const { size_of::<S::m32s>() == size_of::<bool>() });

            let lhs: Complex<f32> = bytemuck::cast(real_lhs);
            let rhs: Complex<f32> = bytemuck::cast(real_rhs);
            unsafe { core::mem::transmute_copy(&(lhs.re >= rhs.re)) }
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
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
    fn simd_index_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        f32::simd_index_select(ctx, mask, lhs, rhs)
    }

    #[inline(always)]
    fn simd_index_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::Index) -> Self::SimdIndex<S> {
        f32::simd_index_splat(ctx, value)
    }

    #[inline(always)]
    fn simd_index_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        f32::simd_index_add(ctx, lhs, rhs)
    }

    #[inline(always)]
    fn simd_and_mask<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdMask<S>,
        rhs: Self::SimdMask<S>,
    ) -> Self::SimdMask<S> {
        f32::simd_and_mask(ctx, lhs, rhs)
    }

    #[inline(always)]
    fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
        f32::simd_first_true_mask(ctx, value)
    }

    type SimdMemMask<S: Simd> = pulp::MemMask<S::m32s>;
    #[inline(always)]
    fn simd_mask_between<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        start: u32,
        end: u32,
    ) -> Self::SimdMemMask<S> {
        ctx.mask_between_m32s(2 * start, 2 * end)
    }
    #[inline(always)]
    unsafe fn simd_mask_load_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *const Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mask_load_ptr_c32s(mask, ptr as _)
    }
    #[inline(always)]
    unsafe fn simd_mask_store_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *mut Self::SimdVec<S>,
        values: Self::SimdVec<S>,
    ) {
        ctx.mask_store_ptr_c32s(mask, ptr as _, values);
    }
}

impl ComplexField for Complex<f64> {
    const IS_REAL: bool = false;
    const SIMD_ABS_SPLIT_REAL_IMAG: bool = true;

    type Arch = pulp::Arch;
    type SimdCtx<S: Simd> = S;
    type Index = u64;
    type Unit = f64;
    type Real = f64;

    const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::Simd;

    type SimdMask<S: Simd> = S::m64s;
    type SimdVec<S: Simd> = S::c64s;
    type SimdIndex<S: Simd> = S::u64s;

    #[inline]
    fn zero_impl() -> Self {
        Complex {
            re: f64::zero_impl(),
            im: f64::zero_impl(),
        }
    }

    #[inline]
    fn one_impl() -> Self {
        Complex {
            re: f64::one_impl(),
            im: f64::zero_impl(),
        }
    }

    #[inline]
    fn nan_impl() -> Self {
        Complex {
            re: f64::nan_impl(),
            im: f64::nan_impl(),
        }
    }

    #[inline]
    fn infinity_impl() -> Self {
        Complex {
            re: f64::infinity_impl(),
            im: f64::infinity_impl(),
        }
    }

    #[inline]
    fn from_real_impl(real: &Self::Real) -> Self {
        Complex {
            re: real.clone(),
            im: f64::zero_impl(),
        }
    }

    #[inline]
    fn from_f64_impl(real: f64) -> Self {
        Complex {
            re: f64::from_f64_impl(real),
            im: f64::zero_impl(),
        }
    }

    #[inline]
    fn real_part_impl(value: &Self) -> Self::Real {
        value.re.clone()
    }

    #[inline]
    fn imag_part_impl(value: &Self) -> Self::Real {
        value.im.clone()
    }

    #[inline]
    fn copy_impl(value: &Self) -> Self {
        value.clone()
    }

    #[inline]
    fn conj_impl(value: &Self) -> Self {
        Self {
            re: value.re.clone(),
            im: value.im.neg_by_ref(),
        }
    }

    #[inline]
    fn recip_impl(value: &Self) -> Self {
        let (re, im) = recip_impl(value.re.clone(), value.im.clone());
        Complex { re, im }
    }

    #[inline]
    fn sqrt_impl(value: &Self) -> Self {
        let (re, im) = sqrt_impl(value.re.clone(), value.im.clone());
        Complex { re, im }
    }

    #[inline]
    fn abs_impl(value: &Self) -> Self::Real {
        abs_impl(value.re.clone(), value.im.clone())
    }

    #[inline]
    #[faer_macros::math]
    fn abs1_impl(value: &Self) -> Self::Real {
        abs1(value.re) + abs1(value.im)
    }

    #[inline]
    #[faer_macros::math]
    fn abs2_impl(value: &Self) -> Self::Real {
        abs2(value.re) + abs2(value.im)
    }

    #[inline]
    #[faer_macros::math]
    fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self {
        Complex {
            re: lhs.re * *rhs,
            im: lhs.im * *rhs,
        }
    }

    #[inline]
    #[faer_macros::math]
    fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self {
        Complex {
            re: mul_pow2(lhs.re, rhs),
            im: mul_pow2(lhs.im, rhs),
        }
    }

    #[inline]
    #[faer_macros::math]
    fn is_finite_impl(value: &Self) -> bool {
        is_finite(value.re) && is_finite(value.im)
    }

    #[inline(always)]
    fn simd_ctx<S: Simd>(simd: S) -> Self::SimdCtx<S> {
        f64::simd_ctx(simd)
    }

    #[inline(always)]
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> S {
        f64::ctx_from_simd(ctx)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self) -> Self::SimdVec<S> {
        ctx.splat_c64s(*value)
    }

    #[inline(always)]
    fn simd_splat_real<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self::Real) -> Self::SimdVec<S> {
        ctx.splat_c64s(Complex {
            re: *value,
            im: *value,
        })
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.add_c64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.sub_c64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.neg_c64s(value)
    }

    #[inline(always)]
    fn simd_conj<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        ctx.conj_c64s(value)
    }

    #[inline(always)]
    fn simd_abs1<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            bytemuck::cast(ctx.abs_f64s(bytemuck::cast(value)))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            let value: Complex<f64> = bytemuck::cast(value);
            let v = value.re.abs() + value.im.abs();
            bytemuck::cast(Complex { re: v, im: v })
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_abs_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            bytemuck::cast(ctx.abs_f64s(bytemuck::cast(value)))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            let value: Complex<f64> = bytemuck::cast(value);
            let re = value.re.abs();
            let im = value.im.abs();
            let v = if re > im { re } else { im };
            bytemuck::cast(Complex { re: v, im: v })
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            bytemuck::cast(ctx.mul_f64s(bytemuck::cast(lhs), bytemuck::cast(real_rhs)))
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
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        Self::simd_mul_real(ctx, lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_e_c64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.conj_mul_e_c64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mul_add_e_c64s(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.conj_mul_add_e_c64s(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            bytemuck::cast(ctx.mul_f64s(bytemuck::cast(value), bytemuck::cast(value)))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            let value: Complex<f64> = bytemuck::cast(value);
            let v = value.re * value.re + value.im * value.im;
            bytemuck::cast(Complex { re: v, im: v })
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: Self::SimdVec<S>,
        acc: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            bytemuck::cast(ctx.mul_add_f64s(
                bytemuck::cast(value),
                bytemuck::cast(value),
                bytemuck::cast(acc),
            ))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            let value: Complex<f64> = bytemuck::cast(value);
            let acc: Complex<f64> = bytemuck::cast(acc);
            let v = value.re * value.re + value.im * value.im + acc.re;
            bytemuck::cast(Complex { re: v, im: v })
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_reduce_sum<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
        ctx.reduce_sum_c64s(value)
    }

    #[inline(always)]
    fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
        ctx.reduce_max_c64s(value)
    }

    #[inline(always)]
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
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
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
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
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            ctx.greater_than_f64s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            assert!(const { size_of::<S::m64s>() == size_of::<bool>() });

            let lhs: Complex<f64> = bytemuck::cast(real_lhs);
            let rhs: Complex<f64> = bytemuck::cast(real_rhs);
            unsafe { core::mem::transmute_copy(&(lhs.re > rhs.re)) }
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: Self::SimdVec<S>,
        real_rhs: Self::SimdVec<S>,
    ) -> Self::SimdMask<S> {
        if const { size_of::<S::c64s>() == size_of::<S::f64s>() } {
            ctx.greater_than_or_equal_f64s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
        } else if const { size_of::<S::c64s>() == size_of::<Complex<f64>>() } {
            assert!(const { size_of::<S::m64s>() == size_of::<bool>() });

            let lhs: Complex<f64> = bytemuck::cast(real_lhs);
            let rhs: Complex<f64> = bytemuck::cast(real_rhs);
            unsafe { core::mem::transmute_copy(&(lhs.re >= rhs.re)) }
        } else {
            panic!();
        }
    }

    #[inline(always)]
    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdVec<S>,
        rhs: Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
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
    fn simd_index_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        f64::simd_index_select(ctx, mask, lhs, rhs)
    }

    #[inline(always)]
    fn simd_index_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::Index) -> Self::SimdIndex<S> {
        f64::simd_index_splat(ctx, value)
    }

    #[inline(always)]
    fn simd_index_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        f64::simd_index_add(ctx, lhs, rhs)
    }

    #[inline(always)]
    fn simd_and_mask<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdMask<S>,
        rhs: Self::SimdMask<S>,
    ) -> Self::SimdMask<S> {
        f64::simd_and_mask(ctx, lhs, rhs)
    }

    #[inline(always)]
    fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
        f64::simd_first_true_mask(ctx, value)
    }

    type SimdMemMask<S: Simd> = pulp::MemMask<S::m64s>;
    #[inline(always)]
    fn simd_mask_between<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        start: u64,
        end: u64,
    ) -> Self::SimdMemMask<S> {
        ctx.mask_between_m64s(2 * start, 2 * end)
    }
    #[inline(always)]
    unsafe fn simd_mask_load_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *const Self::SimdVec<S>,
    ) -> Self::SimdVec<S> {
        ctx.mask_load_ptr_c64s(mask, ptr as _)
    }
    #[inline(always)]
    unsafe fn simd_mask_store_raw<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMemMask<S>,
        ptr: *mut Self::SimdVec<S>,
        values: Self::SimdVec<S>,
    ) {
        ctx.mask_store_ptr_c64s(mask, ptr as _, values);
    }
}
