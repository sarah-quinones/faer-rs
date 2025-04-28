#![cfg_attr(not(feature = "std"), no_std)]
#![allow(non_camel_case_types)]

use bytemuck::Pod;
use core::fmt::Debug;
use num_complex::Complex;
use pulp::Simd;
use qd::Quad;

use math_utils::*;

use pulp::try_const;

pub mod math_utils {
	use crate::{ByRef, ComplexField, RealField, abs_impl};
	use pulp::try_const;

	#[inline(always)]
	#[must_use]
	pub fn eps<T: RealField>() -> T {
		T::Real::epsilon_impl()
	}

	#[inline(always)]
	#[must_use]
	pub fn nbits<T: ComplexField>() -> usize {
		T::Real::nbits_impl()
	}

	#[inline(always)]
	#[must_use]
	pub fn min_positive<T: RealField>() -> T {
		T::min_positive_impl()
	}
	#[inline(always)]
	#[must_use]
	pub fn max_positive<T: RealField>() -> T {
		T::max_positive_impl()
	}
	#[inline(always)]
	#[must_use]
	pub fn sqrt_min_positive<T: RealField>() -> T {
		T::sqrt_min_positive_impl()
	}
	#[inline(always)]
	#[must_use]
	pub fn sqrt_max_positive<T: RealField>() -> T {
		T::sqrt_max_positive_impl()
	}

	#[inline(always)]
	#[must_use]
	pub fn zero<T: ComplexField>() -> T {
		T::zero_impl()
	}
	#[inline(always)]
	#[must_use]
	pub fn one<T: ComplexField>() -> T {
		T::one_impl()
	}
	#[inline(always)]
	#[must_use]
	pub fn nan<T: ComplexField>() -> T {
		T::nan_impl()
	}
	#[inline(always)]
	#[must_use]
	pub fn infinity<T: ComplexField>() -> T {
		T::infinity_impl()
	}

	#[inline(always)]
	#[must_use]
	pub fn real<T: ComplexField>(value: &T) -> T::Real {
		T::real_part_impl((value).by_ref())
	}
	#[inline(always)]
	#[must_use]
	pub fn imag<T: ComplexField>(value: &T) -> T::Real {
		T::imag_part_impl((value).by_ref())
	}
	#[inline(always)]
	#[track_caller]
	#[must_use]
	pub fn neg<T: NegByRef>(value: &T) -> T::Output {
		value.neg_by_ref()
	}
	#[inline(always)]
	#[must_use]
	pub fn copy<T: ComplexField>(value: &T) -> T {
		T::copy_impl((value).by_ref())
	}

	#[inline(always)]
	#[must_use]
	pub fn conj<T: ComplexField>(value: &T) -> T {
		T::conj_impl((value).by_ref())
	}

	#[inline(always)]
	#[track_caller]
	#[must_use]
	pub fn add<T: AddByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
		lhs.add_by_ref(rhs)
	}
	#[inline(always)]
	#[track_caller]
	#[must_use]
	pub fn sub<T: SubByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
		lhs.sub_by_ref(rhs)
	}
	#[inline(always)]
	#[track_caller]
	#[must_use]
	pub fn mul<T: MulByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
		lhs.mul_by_ref(rhs)
	}
	#[inline(always)]
	#[track_caller]
	#[must_use]
	pub fn div<T: DivByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
		lhs.div_by_ref(rhs)
	}

	#[inline(always)]
	#[must_use]
	pub fn mul_real<T: ComplexField>(lhs: &T, rhs: &T::Real) -> T {
		T::mul_real_impl((lhs).by_ref(), (rhs).by_ref())
	}

	#[inline(always)]
	#[must_use]
	pub fn mul_pow2<T: ComplexField>(lhs: &T, rhs: &T::Real) -> T {
		T::mul_real_impl((lhs).by_ref(), (rhs).by_ref())
	}

	#[inline(always)]
	#[must_use]
	pub fn abs1<T: ComplexField>(value: &T) -> T::Real {
		T::abs1_impl((value).by_ref())
	}

	#[inline(always)]
	#[must_use]
	pub fn absmax<T: ComplexField>(value: &T) -> T::Real {
		if try_const! { T::IS_REAL } {
			T::abs1_impl(value)
		} else {
			add(&T::Real::abs1_impl(&real(value)), &T::Real::abs1_impl(&imag(value)))
		}
	}

	#[inline(always)]
	#[must_use]
	pub fn abs<T: ComplexField>(value: &T) -> T::Real {
		T::abs_impl((value).by_ref())
	}

	#[inline(always)]
	#[must_use]
	pub fn hypot<T: RealField>(lhs: &T, rhs: &T) -> T {
		abs_impl::<T::Real>(lhs.clone(), rhs.clone())
	}

	#[inline(always)]
	#[must_use]
	pub fn abs2<T: ComplexField>(value: &T) -> T::Real {
		T::abs2_impl((value).by_ref())
	}

	#[inline(always)]
	#[must_use]
	pub fn max<T: RealField>(lhs: &T, rhs: &T) -> T {
		if lhs > rhs { copy(lhs) } else { copy(rhs) }
	}
	#[inline(always)]
	#[must_use]
	pub fn min<T: RealField>(lhs: &T, rhs: &T) -> T {
		if lhs < rhs { copy(lhs) } else { copy(rhs) }
	}

	#[inline(always)]
	#[must_use]
	pub fn is_nan<T: ComplexField>(value: &T) -> bool {
		T::is_nan_impl((value).by_ref())
	}

	#[inline(always)]
	#[must_use]
	pub fn is_finite<T: ComplexField>(value: &T) -> bool {
		T::is_finite_impl((value).by_ref())
	}

	#[inline(always)]
	#[must_use]
	pub fn sqrt<T: ComplexField>(value: &T) -> T {
		T::sqrt_impl((value).by_ref())
	}
	#[inline(always)]
	#[must_use]
	pub fn recip<T: ComplexField>(value: &T) -> T {
		T::recip_impl((value).by_ref())
	}

	#[inline(always)]
	#[must_use]
	pub fn from_real<T: ComplexField>(value: &T::Real) -> T {
		T::from_real_impl((value).by_ref())
	}

	#[inline(always)]
	#[must_use]
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
	for<'a> &'a Lhs: core::ops::Add<&'a Rhs, Output = Output>,
{
	type Output = Output;

	#[inline]
	#[track_caller]
	fn add_by_ref(&self, rhs: &Rhs) -> Self::Output {
		self + rhs
	}
}
impl<Rhs, Lhs, Output> SubByRef<Rhs> for Lhs
where
	for<'a> &'a Lhs: core::ops::Sub<&'a Rhs, Output = Output>,
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
	for<'a> &'a Lhs: core::ops::Mul<&'a Rhs, Output = Output>,
{
	type Output = Output;

	#[inline]
	#[track_caller]
	fn mul_by_ref(&self, rhs: &Rhs) -> Self::Output {
		self * rhs
	}
}
impl<Rhs, Lhs, Output> DivByRef<Rhs> for Lhs
where
	for<'a> &'a Lhs: core::ops::Div<&'a Rhs, Output = Output>,
{
	type Output = Output;

	#[inline]
	#[track_caller]
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
	#[track_caller]
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
pub struct RealReg<T>(pub T);

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
	pub fn splat_real(&self, value: &T::Real) -> RealReg<T::SimdVec<S>> {
		RealReg(unsafe { core::mem::transmute_copy(&T::simd_splat_real(&self.0, (value).by_ref())) })
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
	pub fn abs1(&self, value: T::SimdVec<S>) -> RealReg<T::SimdVec<S>> {
		let value = unsafe { core::mem::transmute_copy(&value) };
		RealReg(unsafe { core::mem::transmute_copy(&T::simd_abs1(&self.0, value)) })
	}

	#[inline(always)]
	pub fn abs_max(&self, value: T::SimdVec<S>) -> RealReg<T::SimdVec<S>> {
		let value = unsafe { core::mem::transmute_copy(&value) };
		RealReg(unsafe { core::mem::transmute_copy(&T::simd_abs_max(&self.0, value)) })
	}

	#[inline(always)]
	pub fn mul_real(&self, lhs: T::SimdVec<S>, rhs: RealReg<T::SimdVec<S>>) -> T::SimdVec<S> {
		let lhs = unsafe { core::mem::transmute_copy(&lhs) };
		let rhs = unsafe { core::mem::transmute_copy(&rhs) };
		unsafe { core::mem::transmute_copy(&T::simd_mul_real(&self.0, lhs, rhs)) }
	}

	#[inline(always)]
	pub fn mul_pow2(&self, lhs: T::SimdVec<S>, rhs: RealReg<T::SimdVec<S>>) -> T::SimdVec<S> {
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
	pub fn mul_add(&self, lhs: T::SimdVec<S>, rhs: T::SimdVec<S>, acc: T::SimdVec<S>) -> T::SimdVec<S> {
		let lhs = unsafe { core::mem::transmute_copy(&lhs) };
		let rhs = unsafe { core::mem::transmute_copy(&rhs) };
		let acc = unsafe { core::mem::transmute_copy(&acc) };
		unsafe { core::mem::transmute_copy(&T::simd_mul_add(&self.0, lhs, rhs, acc)) }
	}

	#[inline(always)]
	pub fn conj_mul_add(&self, lhs: T::SimdVec<S>, rhs: T::SimdVec<S>, acc: T::SimdVec<S>) -> T::SimdVec<S> {
		let lhs = unsafe { core::mem::transmute_copy(&lhs) };
		let rhs = unsafe { core::mem::transmute_copy(&rhs) };
		let acc = unsafe { core::mem::transmute_copy(&acc) };
		unsafe { core::mem::transmute_copy(&T::simd_conj_mul_add(&self.0, lhs, rhs, acc)) }
	}

	#[inline(always)]
	pub fn abs2(&self, value: T::SimdVec<S>) -> RealReg<T::SimdVec<S>> {
		let value = unsafe { core::mem::transmute_copy(&value) };
		RealReg(unsafe { core::mem::transmute_copy(&T::simd_abs2(&self.0, value)) })
	}

	#[inline(always)]
	pub fn abs2_add(&self, value: T::SimdVec<S>, acc: RealReg<T::SimdVec<S>>) -> RealReg<T::SimdVec<S>> {
		let value = unsafe { core::mem::transmute_copy(&value) };
		let acc = unsafe { core::mem::transmute_copy(&acc) };
		RealReg(unsafe { core::mem::transmute_copy(&T::simd_abs2_add(&self.0, value, acc)) })
	}

	#[inline(always)]
	pub fn reduce_sum(&self, value: T::SimdVec<S>) -> T {
		let value = unsafe { core::mem::transmute_copy(&value) };
		unsafe { core::mem::transmute_copy(&T::simd_reduce_sum(&self.0, value)) }
	}

	#[inline(always)]
	pub fn reduce_max(&self, value: RealReg<T::SimdVec<S>>) -> T {
		let value = unsafe { core::mem::transmute_copy(&value) };
		unsafe { core::mem::transmute_copy(&T::simd_reduce_max(&self.0, value)) }
	}

	#[faer_macros::math]
	#[inline(always)]
	pub fn reduce_sum_real(&self, value: RealReg<T::SimdVec<S>>) -> Real<T> {
		let value = T::simd_reduce_sum(&self.0, value.0);
		if try_const! { T::SIMD_ABS_SPLIT_REAL_IMAG && !S::IS_SCALAR } {
			add(real(value), imag(value))
		} else {
			real(value)
		}
	}

	#[faer_macros::math]
	#[inline(always)]
	pub fn reduce_max_real(&self, value: RealReg<T::SimdVec<S>>) -> Real<T> {
		let value = T::simd_reduce_max(&self.0, value.0);
		if try_const! { T::SIMD_ABS_SPLIT_REAL_IMAG && !S::IS_SCALAR } {
			max(real(value), imag(value))
		} else {
			real(value)
		}
	}

	#[inline(always)]
	pub fn max(&self, lhs: RealReg<T::SimdVec<S>>, rhs: RealReg<T::SimdVec<S>>) -> RealReg<T::SimdVec<S>> {
		let cmp = self.gt(lhs, rhs);
		RealReg(self.select(cmp, lhs.0, rhs.0))
	}

	#[inline(always)]
	pub fn eq(&self, lhs: T::SimdVec<S>, rhs: T::SimdVec<S>) -> T::SimdMask<S> {
		T::simd_equal(&self.0, lhs, rhs)
	}

	#[inline(always)]
	pub fn lt(&self, lhs: RealReg<T::SimdVec<S>>, rhs: RealReg<T::SimdVec<S>>) -> T::SimdMask<S> {
		let lhs = unsafe { core::mem::transmute_copy(&lhs) };
		let rhs = unsafe { core::mem::transmute_copy(&rhs) };
		unsafe { core::mem::transmute_copy(&T::simd_less_than(&self.0, lhs, rhs)) }
	}

	#[inline(always)]
	pub fn gt(&self, lhs: RealReg<T::SimdVec<S>>, rhs: RealReg<T::SimdVec<S>>) -> T::SimdMask<S> {
		let lhs = unsafe { core::mem::transmute_copy(&lhs) };
		let rhs = unsafe { core::mem::transmute_copy(&rhs) };
		unsafe { core::mem::transmute_copy(&T::simd_greater_than(&self.0, lhs, rhs)) }
	}

	#[inline(always)]
	pub fn le(&self, lhs: RealReg<T::SimdVec<S>>, rhs: RealReg<T::SimdVec<S>>) -> T::SimdMask<S> {
		let lhs = unsafe { core::mem::transmute_copy(&lhs) };
		let rhs = unsafe { core::mem::transmute_copy(&rhs) };
		unsafe { core::mem::transmute_copy(&T::simd_less_than_or_equal(&self.0, lhs, rhs)) }
	}

	#[inline(always)]
	pub fn ge(&self, lhs: RealReg<T::SimdVec<S>>, rhs: RealReg<T::SimdVec<S>>) -> T::SimdMask<S> {
		let lhs = unsafe { core::mem::transmute_copy(&lhs) };
		let rhs = unsafe { core::mem::transmute_copy(&rhs) };
		unsafe { core::mem::transmute_copy(&T::simd_greater_than_or_equal(&self.0, lhs, rhs)) }
	}

	#[inline(always)]
	pub fn select(&self, mask: T::SimdMask<S>, lhs: T::SimdVec<S>, rhs: T::SimdVec<S>) -> T::SimdVec<S> {
		let lhs = unsafe { core::mem::transmute_copy(&lhs) };
		let rhs = unsafe { core::mem::transmute_copy(&rhs) };
		unsafe { core::mem::transmute_copy(&T::simd_select(&self.0, mask, lhs, rhs)) }
	}

	#[inline(always)]
	pub fn iselect(&self, mask: T::SimdMask<S>, lhs: T::SimdIndex<S>, rhs: T::SimdIndex<S>) -> T::SimdIndex<S> {
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
	pub fn or_mask(&self, lhs: T::SimdMask<S>, rhs: T::SimdMask<S>) -> T::SimdMask<S> {
		T::simd_or_mask(&self.0, lhs, rhs)
	}

	#[inline(always)]
	pub fn and_mask(&self, lhs: T::SimdMask<S>, rhs: T::SimdMask<S>) -> T::SimdMask<S> {
		T::simd_and_mask(&self.0, lhs, rhs)
	}

	#[inline(always)]
	pub fn not_mask(&self, mask: T::SimdMask<S>) -> T::SimdMask<S> {
		T::simd_not_mask(&self.0, mask)
	}

	#[inline(always)]
	pub fn first_true_mask(&self, value: T::SimdMask<S>) -> usize {
		T::simd_first_true_mask(&self.0, value)
	}

	#[inline(always)]
	pub unsafe fn mask_load(&self, mask: T::SimdMemMask<S>, ptr: *const T::SimdVec<S>) -> T::SimdVec<S> {
		unsafe { T::simd_mask_load(&self.0, mask, ptr) }
	}

	#[inline(always)]
	pub unsafe fn mask_store(&self, mask: T::SimdMemMask<S>, ptr: *mut T::SimdVec<S>, value: T::SimdVec<S>) {
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

pub unsafe trait Conjugate: Send + Sync + core::fmt::Debug {
	const IS_CANONICAL: bool;

	type Conj: Conjugate<Conj = Self, Canonical = Self::Canonical>;
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
	impl Seal for u8 {}
	impl Seal for u16 {}
	impl Seal for u32 {}
	impl Seal for u64 {}
	impl Seal for u128 {}
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

pub trait IndexCore:
	Seal
	+ core::fmt::Debug
	+ core::ops::Not<Output = Self>
	+ core::ops::BitAnd<Output = Self>
	+ core::ops::BitOr<Output = Self>
	+ core::ops::BitXor<Output = Self>
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
	const MAX: Self;

	/// Truncate `value` to type [`Self`].
	#[must_use]
	fn truncate(value: usize) -> Self;

	/// Zero extend `self`.
	#[must_use]
	fn zx(self) -> usize;
}

pub trait Index: IndexCore {
	/// Equally-sized index type with a fixed size (no `usize`).
	type FixedWidth: Index;
	/// Equally-sized signed index type.
	type Signed: SignedIndex;

	const BITS: u32 = core::mem::size_of::<Self>() as u32 * 8;

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

impl IndexCore for u8 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as _
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as _
	}
}
impl IndexCore for u16 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as _
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as _
	}
}
impl IndexCore for u32 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as _
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as _
	}
}

impl IndexCore for u64 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as _
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as _
	}
}

impl IndexCore for u128 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as _
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as _
	}
}

impl IndexCore for usize {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self
	}
}

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

unsafe impl<T: RealField> Conjugate for T {
	type Canonical = T;
	type Conj = T;

	const IS_CANONICAL: bool = true;
}

unsafe impl<T: RealField> Conjugate for Complex<T> {
	type Canonical = Complex<T>;
	type Conj = ComplexConj<T>;

	const IS_CANONICAL: bool = true;
}
unsafe impl<T: RealField> Conjugate for ComplexConj<T> {
	type Canonical = Complex<T>;
	type Conj = Complex<T>;

	const IS_CANONICAL: bool = false;
}

pub trait SimdArch: Copy + Default + Send + Sync {
	fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R;
}

impl SimdArch for pulp::Arch {
	#[inline]
	fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R {
		self.dispatch(f)
	}
}

impl SimdArch for pulp::Scalar {
	#[inline]
	fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R {
		f.with_simd(self)
	}
}

pub trait ComplexField:
	Debug
	+ Clone
	+ Conjugate<Canonical = Self>
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
	type Index: IndexCore;

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

	fn simd_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMask<S>;
	fn simd_mem_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMemMask<S>;
	unsafe fn simd_mask_load_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *const Self::SimdVec<S>) -> Self::SimdVec<S>;
	unsafe fn simd_mask_store_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *mut Self::SimdVec<S>, values: Self::SimdVec<S>);

	fn simd_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self) -> Self::SimdVec<S>;
	fn simd_splat_real<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self::Real) -> Self::SimdVec<S>;

	fn simd_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S>;
	fn simd_sub<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S>;

	fn simd_neg<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S>;
	fn simd_conj<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S>;
	fn simd_abs1<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S>;
	fn simd_abs_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S>;

	fn simd_mul_real<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S>;
	fn simd_mul_pow2<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S>;

	fn simd_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S>;
	fn simd_conj_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S>;
	fn simd_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S>;
	fn simd_conj_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S>;
	fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S>;
	fn simd_abs2_add<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S>;

	fn simd_reduce_sum<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self;
	fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self;
	fn simd_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S>;
	fn simd_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S>;
	fn simd_less_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S>;
	fn simd_greater_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S>;
	fn simd_greater_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S>;

	fn simd_select<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S>;
	fn simd_index_select<S: Simd>(
		ctx: &Self::SimdCtx<S>,
		mask: Self::SimdMask<S>,
		lhs: Self::SimdIndex<S>,
		rhs: Self::SimdIndex<S>,
	) -> Self::SimdIndex<S>;

	fn simd_index_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::Index) -> Self::SimdIndex<S>;
	fn simd_index_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdIndex<S>;
	fn simd_index_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdMask<S>;
	#[inline(always)]
	fn simd_index_greater_than<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdMask<S> {
		Self::simd_index_less_than(ctx, rhs, lhs)
	}
	#[inline(always)]
	fn simd_index_less_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdMask<S> {
		Self::simd_not_mask(ctx, Self::simd_index_less_than(ctx, rhs, lhs))
	}
	#[inline(always)]
	fn simd_index_greater_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdMask<S> {
		Self::simd_not_mask(ctx, Self::simd_index_greater_than(ctx, rhs, lhs))
	}

	fn simd_and_mask<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S>;
	fn simd_or_mask<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S>;
	fn simd_not_mask<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>) -> Self::SimdMask<S>;
	fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize;

	#[inline(always)]
	fn simd_load<S: Simd>(ctx: &Self::SimdCtx<S>, ptr: &Self::SimdVec<S>) -> Self::SimdVec<S> {
		let simd = Self::ctx_from_simd(ctx);
		if try_const! { Self::Unit::IS_NATIVE_F32 } {
			simd.deinterleave_shfl_f32s(*ptr)
		} else if try_const! { Self::Unit::IS_NATIVE_F64 } {
			simd.deinterleave_shfl_f64s(*ptr)
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_store<S: Simd>(ctx: &Self::SimdCtx<S>, ptr: &mut Self::SimdVec<S>, value: Self::SimdVec<S>) {
		let simd = Self::ctx_from_simd(ctx);
		if try_const! { Self::Unit::IS_NATIVE_F32 } {
			*ptr = simd.interleave_shfl_f32s(value)
		} else if try_const! { Self::Unit::IS_NATIVE_F64 } {
			*ptr = simd.interleave_shfl_f64s(value)
		} else {
			panic!();
		}
	}

	#[inline(always)]
	unsafe fn simd_mask_load<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *const Self::SimdVec<S>) -> Self::SimdVec<S> {
		let simd = Self::ctx_from_simd(ctx);
		let value = Self::simd_mask_load_raw(ctx, mask, ptr);
		if try_const! { Self::Unit::IS_NATIVE_F32 } {
			simd.deinterleave_shfl_f32s(value)
		} else if try_const! { Self::Unit::IS_NATIVE_F64 } {
			simd.deinterleave_shfl_f64s(value)
		} else {
			panic!();
		}
	}

	#[inline(always)]
	unsafe fn simd_mask_store<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *mut Self::SimdVec<S>, value: Self::SimdVec<S>) {
		let simd = Self::ctx_from_simd(ctx);
		if try_const! { Self::Unit::IS_NATIVE_F32 } {
			Self::simd_mask_store_raw(ctx, mask, ptr, simd.interleave_shfl_f32s(value))
		} else if try_const! { Self::Unit::IS_NATIVE_F64 } {
			Self::simd_mask_store_raw(ctx, mask, ptr, simd.interleave_shfl_f64s(value))
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
			if try_const! { Self::Unit::IS_NATIVE_F32 } {
				core::mem::transmute_copy::<_, Self::SimdIndex<S>>(&simd.deinterleave_shfl_f32s(Interleave(core::mem::transmute_copy::<
					_,
					Self::SimdVec<S>,
				>(
					&pulp::iota_32::<Interleave<Self>>()
				))))
			} else if try_const! { Self::Unit::IS_NATIVE_F64 } {
				core::mem::transmute_copy::<_, Self::SimdIndex<S>>(
					&simd.deinterleave_shfl_f64s(core::mem::transmute_copy::<_, Self::SimdVec<S>>(&pulp::iota_64::<Interleave<Self>>())),
				)
			} else {
				panic!();
			}
		}
	}
}

pub trait RealField:
	ComplexField<Real = Self, Conj = Self> + DivByRef<Output = Self> + PartialOrd + num_traits::NumOps + num_traits::Num + core::ops::Neg<Output = Self>
{
	fn epsilon_impl() -> Self;
	fn nbits_impl() -> usize;

	fn min_positive_impl() -> Self;
	fn max_positive_impl() -> Self;
	fn sqrt_min_positive_impl() -> Self;
	fn sqrt_max_positive_impl() -> Self;
}

impl ComplexField for f32 {
	type Arch = pulp::Arch;
	type Index = u32;
	type Real = Self;
	type SimdCtx<S: Simd> = S;
	type SimdIndex<S: Simd> = S::u32s;
	type SimdMask<S: Simd> = S::m32s;
	type SimdMemMask<S: Simd> = pulp::MemMask<S::m32s>;
	type SimdVec<S: Simd> = S::f32s;
	type Unit = Self;

	const IS_NATIVE_F32: bool = true;
	const IS_REAL: bool = true;
	const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::Simd;

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
		libm::sqrtf(*value)
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
	fn simd_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.add_f32s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_sub<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
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
	fn simd_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_f32s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_mul_real<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_f32s(lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_mul_pow2<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_f32s(lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_conj_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_f32s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_add_e_f32s(lhs, rhs, acc)
	}

	#[inline(always)]
	fn simd_conj_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_add_e_f32s(lhs, rhs, acc)
	}

	#[inline(always)]
	fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_f32s(value, value)
	}

	#[inline(always)]
	fn simd_abs2_add<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
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
	fn simd_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		ctx.equal_f32s(real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		ctx.less_than_f32s(real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_greater_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		ctx.greater_than_f32s(real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_less_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		ctx.less_than_or_equal_f32s(real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_greater_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		ctx.greater_than_or_equal_f32s(real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_select<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
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
		ctx.splat_u32s(value as _)
	}

	#[inline(always)]
	fn simd_index_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdIndex<S> {
		ctx.add_u32s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_index_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdMask<S> {
		ctx.less_than_u32s(lhs, rhs)
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
	fn simd_and_mask<S: Simd>(simd: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		simd.and_m32s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_or_mask<S: Simd>(simd: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		simd.or_m32s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_not_mask<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>) -> Self::SimdMask<S> {
		ctx.not_m32s(mask)
	}

	#[inline(always)]
	fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
		ctx.first_true_m32s(value)
	}

	#[inline(always)]
	fn simd_mem_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMemMask<S> {
		ctx.mask_between_m32s(start as _, end as _)
	}

	#[inline(always)]
	fn simd_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMask<S> {
		ctx.mask_between_m32s(start as _, end as _).mask()
	}

	#[inline(always)]
	unsafe fn simd_mask_load_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *const Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mask_load_ptr_f32s(mask, ptr as _)
	}

	#[inline(always)]
	unsafe fn simd_mask_store_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *mut Self::SimdVec<S>, values: Self::SimdVec<S>) {
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
		libm::sqrtf(Self::MIN_POSITIVE)
	}

	#[inline(always)]
	fn sqrt_max_positive_impl() -> Self {
		libm::sqrtf(Self::MIN_POSITIVE.recip())
	}

	#[inline(always)]
	fn nbits_impl() -> usize {
		Self::MANTISSA_DIGITS as usize
	}
}

impl ComplexField for f64 {
	type Arch = pulp::Arch;
	type Index = u64;
	type Real = Self;
	type SimdCtx<S: Simd> = S;
	type SimdIndex<S: Simd> = S::u64s;
	type SimdMask<S: Simd> = S::m64s;
	type SimdMemMask<S: Simd> = pulp::MemMask<S::m64s>;
	type SimdVec<S: Simd> = S::f64s;
	type Unit = Self;

	const IS_NATIVE_F64: bool = true;
	const IS_REAL: bool = true;
	const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::Simd;

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
		libm::sqrt(*value)
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
	fn simd_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.add_f64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_sub<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
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
	fn simd_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_f64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_mul_real<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_f64s(lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_mul_pow2<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_f64s(lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_conj_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_f64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_add_e_f64s(lhs, rhs, acc)
	}

	#[inline(always)]
	fn simd_conj_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_add_e_f64s(lhs, rhs, acc)
	}

	#[inline(always)]
	fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_f64s(value, value)
	}

	#[inline(always)]
	fn simd_abs2_add<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
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
	fn simd_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		ctx.equal_f64s(real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		ctx.less_than_f64s(real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_greater_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		ctx.greater_than_f64s(real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_less_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		ctx.less_than_or_equal_f64s(real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_greater_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		ctx.greater_than_or_equal_f64s(real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_select<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
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
		ctx.splat_u64s(value as _)
	}

	#[inline(always)]
	fn simd_index_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdIndex<S> {
		ctx.add_u64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_index_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdMask<S> {
		ctx.less_than_u64s(lhs, rhs)
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
	fn simd_and_mask<S: Simd>(simd: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		simd.and_m64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_or_mask<S: Simd>(simd: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		simd.or_m64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_not_mask<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>) -> Self::SimdMask<S> {
		ctx.not_m64s(mask)
	}

	#[inline(always)]
	fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
		ctx.first_true_m64s(value)
	}

	#[inline(always)]
	fn is_finite_impl(value: &Self) -> bool {
		(*value).is_finite()
	}

	#[inline(always)]
	fn simd_mem_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMemMask<S> {
		ctx.mask_between_m64s(start as _, end as _)
	}

	#[inline(always)]
	fn simd_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMask<S> {
		ctx.mask_between_m64s(start as _, end as _).mask()
	}

	#[inline(always)]
	unsafe fn simd_mask_load_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *const Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mask_load_ptr_f64s(mask, ptr as _)
	}

	#[inline(always)]
	unsafe fn simd_mask_store_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *mut Self::SimdVec<S>, values: Self::SimdVec<S>) {
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
		libm::sqrt(Self::MIN_POSITIVE)
	}

	#[inline(always)]
	fn sqrt_max_positive_impl() -> Self {
		libm::sqrt(Self::MIN_POSITIVE.recip())
	}

	#[inline(always)]
	fn nbits_impl() -> usize {
		Self::MANTISSA_DIGITS as usize
	}
}

impl<T: RealField> ComplexField for Complex<T> {
	type Arch = T::Arch;
	type Index = T::Index;
	type Real = T;
	type SimdCtx<S: Simd> = T::SimdCtx<S>;
	type SimdIndex<S: Simd> = T::SimdIndex<S>;
	type SimdMask<S: Simd> = T::SimdMask<S>;
	type SimdMemMask<S: Simd> = Complex<T::SimdMemMask<S>>;
	type SimdVec<S: Simd> = Complex<T::SimdVec<S>>;
	type Unit = T::Unit;

	const IS_NATIVE_C32: bool = T::IS_NATIVE_F32;
	const IS_NATIVE_C64: bool = T::IS_NATIVE_F64;
	const IS_REAL: bool = false;
	const SIMD_CAPABILITIES: SimdCapabilities = T::SIMD_CAPABILITIES;

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
	fn simd_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		Complex {
			re: T::simd_add(ctx, lhs.re, rhs.re),
			im: T::simd_add(ctx, lhs.im, rhs.im),
		}
	}

	#[inline(always)]
	fn simd_sub<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
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
		let v = T::simd_add(ctx, T::simd_abs1(ctx, value.re), T::simd_abs1(ctx, value.im));
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
	fn simd_mul_real<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		Complex {
			re: T::simd_mul_real(ctx, lhs.re, real_rhs.re),
			im: T::simd_mul_real(ctx, lhs.im, real_rhs.re),
		}
	}

	#[inline(always)]
	fn simd_mul_pow2<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		Complex {
			re: T::simd_mul_pow2(ctx, lhs.re, real_rhs.re),
			im: T::simd_mul_pow2(ctx, lhs.im, real_rhs.re),
		}
	}

	#[inline(always)]
	fn simd_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		Complex {
			re: T::simd_mul_add(ctx, lhs.re, rhs.re, T::simd_neg(ctx, T::simd_mul(ctx, lhs.im, rhs.im))),
			im: T::simd_mul_add(ctx, lhs.re, rhs.im, T::simd_mul(ctx, lhs.im, rhs.re)),
		}
	}

	#[inline(always)]
	fn simd_conj_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		Complex {
			re: T::simd_mul_add(ctx, lhs.re, rhs.re, T::simd_mul(ctx, lhs.im, rhs.im)),
			im: T::simd_mul_add(ctx, lhs.re, rhs.im, T::simd_neg(ctx, T::simd_mul(ctx, lhs.im, rhs.re))),
		}
	}

	#[inline(always)]
	fn simd_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		Complex {
			re: T::simd_mul_add(ctx, T::simd_neg(ctx, lhs.im), rhs.im, T::simd_mul_add(ctx, lhs.re, rhs.re, acc.re)),
			im: T::simd_mul_add(ctx, lhs.re, rhs.im, T::simd_mul_add(ctx, lhs.im, rhs.re, acc.im)),
		}
	}

	#[inline(always)]
	fn simd_conj_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		Complex {
			re: T::simd_mul_add(ctx, lhs.im, rhs.im, T::simd_mul_add(ctx, lhs.re, rhs.re, acc.re)),
			im: T::simd_mul_add(ctx, lhs.re, rhs.im, T::simd_mul_add(ctx, T::simd_neg(ctx, lhs.im), rhs.re, acc.im)),
		}
	}

	#[inline(always)]
	fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		let v = T::simd_abs2_add(ctx, value.re, T::simd_abs2(ctx, value.im));
		Complex { re: v, im: v }
	}

	#[inline(always)]
	fn simd_abs2_add<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
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
	fn simd_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		T::simd_and_mask(
			ctx,
			T::simd_equal(ctx, real_lhs.re, real_rhs.re),
			T::simd_equal(ctx, real_lhs.im, real_rhs.im),
		)
	}

	#[inline(always)]
	fn simd_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		T::simd_less_than(ctx, real_lhs.re, real_rhs.re)
	}

	#[inline(always)]
	fn simd_less_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		T::simd_less_than_or_equal(ctx, real_lhs.re, real_rhs.re)
	}

	#[inline(always)]
	fn simd_greater_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		T::simd_greater_than(ctx, real_lhs.re, real_rhs.re)
	}

	#[inline(always)]
	fn simd_greater_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		T::simd_greater_than_or_equal(ctx, real_lhs.re, real_rhs.re)
	}

	#[inline(always)]
	fn simd_select<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
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
	fn simd_index_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdIndex<S> {
		T::simd_index_add(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_index_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdMask<S> {
		T::simd_index_less_than(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_and_mask<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		T::simd_and_mask(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_or_mask<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		T::simd_or_mask(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_not_mask<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>) -> Self::SimdMask<S> {
		T::simd_not_mask(ctx, mask)
	}

	#[inline(always)]
	fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
		T::simd_first_true_mask(ctx, value)
	}

	#[inline(always)]
	fn simd_mem_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMemMask<S> {
		let n = core::mem::size_of::<Self::SimdVec<S>>() / core::mem::size_of::<Self>();
		let start = start.zx() * 2;
		let end = end.zx() * 2;

		let mut sa = start.min(n);
		let mut ea = end.min(n);
		let mut sb = start.max(n) - n;
		let mut eb = end.max(n) - n;

		if sa == ea {
			sa = 0;
			ea = 0;
		}
		if sb == eb {
			sb = 0;
			eb = 0;
		}

		let re = T::simd_mem_mask_between(ctx, T::Index::truncate(sa), T::Index::truncate(ea));
		let im = T::simd_mem_mask_between(ctx, T::Index::truncate(sb), T::Index::truncate(eb));
		Complex { re, im }
	}

	#[inline(always)]
	fn simd_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMask<S> {
		T::simd_mask_between(ctx, start, end)
	}

	#[inline(always)]
	unsafe fn simd_mask_load_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *const Self::SimdVec<S>) -> Self::SimdVec<S> {
		Complex {
			re: T::simd_mask_load_raw(ctx, mask.re, core::ptr::addr_of!((*ptr).re)),
			im: T::simd_mask_load_raw(ctx, mask.im, core::ptr::addr_of!((*ptr).im)),
		}
	}

	#[inline(always)]
	unsafe fn simd_mask_store_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *mut Self::SimdVec<S>, values: Self::SimdVec<S>) {
		T::simd_mask_store_raw(ctx, mask.re, core::ptr::addr_of_mut!((*ptr).re), values.re);
		T::simd_mask_store_raw(ctx, mask.im, core::ptr::addr_of_mut!((*ptr).im), values.im);
	}
}

#[repr(transparent)]
#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ComplexImpl<T>(Complex<T>);

#[repr(transparent)]
#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ComplexImplConj<T>(Complex<T>);

unsafe impl Conjugate for ComplexImpl<f32> {
	type Canonical = ComplexImpl<f32>;
	type Conj = ComplexImplConj<f32>;

	const IS_CANONICAL: bool = true;
}
unsafe impl Conjugate for ComplexImplConj<f32> {
	type Canonical = ComplexImpl<f32>;
	type Conj = ComplexImpl<f32>;

	const IS_CANONICAL: bool = false;
}
unsafe impl Conjugate for ComplexImpl<f64> {
	type Canonical = ComplexImpl<f64>;
	type Conj = ComplexImplConj<f64>;

	const IS_CANONICAL: bool = true;
}
unsafe impl Conjugate for ComplexImplConj<f64> {
	type Canonical = ComplexImpl<f64>;
	type Conj = ComplexImpl<f64>;

	const IS_CANONICAL: bool = false;
}

impl<T: RealField> core::ops::Neg for &ComplexImpl<T> {
	type Output = ComplexImpl<T>;

	#[inline]
	fn neg(self) -> Self::Output {
		use math_utils::*;

		ComplexImpl(neg(&self.0))
	}
}
impl<T: RealField> core::ops::Add<&ComplexImpl<T>> for &ComplexImpl<T> {
	type Output = ComplexImpl<T>;

	#[inline]
	fn add(self, rhs: &ComplexImpl<T>) -> Self::Output {
		use math_utils::*;

		ComplexImpl(add(&self.0, &rhs.0))
	}
}
impl<T: RealField> core::ops::Sub<&ComplexImpl<T>> for &ComplexImpl<T> {
	type Output = ComplexImpl<T>;

	#[inline]
	fn sub(self, rhs: &ComplexImpl<T>) -> Self::Output {
		use math_utils::*;

		ComplexImpl(sub(&self.0, &rhs.0))
	}
}
impl<T: RealField> core::ops::Mul<&ComplexImpl<T>> for &ComplexImpl<T> {
	type Output = ComplexImpl<T>;

	#[inline]
	fn mul(self, rhs: &ComplexImpl<T>) -> Self::Output {
		use math_utils::*;

		ComplexImpl(mul(&self.0, &rhs.0))
	}
}

impl<T> From<Complex<T>> for ComplexImpl<T> {
	#[inline]
	fn from(value: Complex<T>) -> Self {
		Self(value)
	}
}

impl ComplexField for ComplexImpl<f32> {
	type Arch = pulp::Arch;
	type Index = u32;
	type Real = f32;
	type SimdCtx<S: Simd> = S;
	type SimdIndex<S: Simd> = S::u32s;
	type SimdMask<S: Simd> = S::m32s;
	type SimdMemMask<S: Simd> = pulp::MemMask<S::m32s>;
	type SimdVec<S: Simd> = S::c32s;
	type Unit = f32;

	const IS_NATIVE_C32: bool = true;
	const IS_REAL: bool = false;
	const SIMD_ABS_SPLIT_REAL_IMAG: bool = true;
	const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::Simd;

	#[inline]
	fn zero_impl() -> Self {
		Complex {
			re: f32::zero_impl(),
			im: f32::zero_impl(),
		}
		.into()
	}

	#[inline]
	fn one_impl() -> Self {
		Complex {
			re: f32::one_impl(),
			im: f32::zero_impl(),
		}
		.into()
	}

	#[inline]
	fn nan_impl() -> Self {
		Complex {
			re: f32::nan_impl(),
			im: f32::nan_impl(),
		}
		.into()
	}

	#[inline]
	fn infinity_impl() -> Self {
		Complex {
			re: f32::infinity_impl(),
			im: f32::infinity_impl(),
		}
		.into()
	}

	#[inline]
	fn from_real_impl(real: &Self::Real) -> Self {
		Complex {
			re: real.clone(),
			im: f32::zero_impl(),
		}
		.into()
	}

	#[inline]
	fn from_f64_impl(real: f64) -> Self {
		Complex {
			re: f32::from_f64_impl(real),
			im: f32::zero_impl(),
		}
		.into()
	}

	#[inline]
	fn real_part_impl(value: &Self) -> Self::Real {
		value.0.re.clone()
	}

	#[inline]
	fn imag_part_impl(value: &Self) -> Self::Real {
		value.0.im.clone()
	}

	#[inline]
	fn copy_impl(value: &Self) -> Self {
		value.clone()
	}

	#[inline]
	fn conj_impl(value: &Self) -> Self {
		Complex {
			re: value.0.re.clone(),
			im: value.0.im.neg_by_ref(),
		}
		.into()
	}

	#[inline]
	fn recip_impl(value: &Self) -> Self {
		let (re, im) = recip_impl(value.0.re.clone(), value.0.im.clone());
		Complex { re, im }.into()
	}

	#[inline]
	fn sqrt_impl(value: &Self) -> Self {
		let (re, im) = sqrt_impl(value.0.re.clone(), value.0.im.clone());
		Complex { re, im }.into()
	}

	#[inline]
	fn abs_impl(value: &Self) -> Self::Real {
		abs_impl(value.0.re.clone(), value.0.im.clone())
	}

	#[inline]
	#[faer_macros::math]
	fn abs1_impl(value: &Self) -> Self::Real {
		abs1(value.0.re) + abs1(value.0.im)
	}

	#[inline]
	#[faer_macros::math]
	fn abs2_impl(value: &Self) -> Self::Real {
		abs2(value.0.re) + abs2(value.0.im)
	}

	#[inline]
	#[faer_macros::math]
	fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self {
		Complex {
			re: lhs.0.re * *rhs,
			im: lhs.0.im * *rhs,
		}
		.into()
	}

	#[inline]
	#[faer_macros::math]
	fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self {
		Complex {
			re: mul_pow2(lhs.0.re, rhs),
			im: mul_pow2(lhs.0.im, rhs),
		}
		.into()
	}

	#[inline]
	#[faer_macros::math]
	fn is_finite_impl(value: &Self) -> bool {
		is_finite(value.0.re) && is_finite(value.0.im)
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
		ctx.splat_c32s(value.0)
	}

	#[inline(always)]
	fn simd_splat_real<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self::Real) -> Self::SimdVec<S> {
		ctx.splat_c32s(Complex { re: *value, im: *value })
	}

	#[inline(always)]
	fn simd_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.add_c32s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_sub<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
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
		if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<S::f32s>() } {
			bytemuck::cast(ctx.abs_f32s(bytemuck::cast(value)))
		} else if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<Complex<f32>>() } {
			let value: Complex<f32> = bytemuck::cast(value);
			let v = value.re.abs() + value.im.abs();
			bytemuck::cast(Complex { re: v, im: v })
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_abs_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<S::f32s>() } {
			bytemuck::cast(ctx.abs_f32s(bytemuck::cast(value)))
		} else if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<Complex<f32>>() } {
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
	fn simd_mul_real<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<S::f32s>() } {
			bytemuck::cast(ctx.mul_f32s(bytemuck::cast(lhs), bytemuck::cast(real_rhs)))
		} else if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<Complex<f32>>() } {
			let mut lhs: Complex<f32> = bytemuck::cast(lhs);
			let rhs: Complex<f32> = bytemuck::cast(real_rhs);
			lhs *= rhs.re;
			bytemuck::cast(lhs)
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_mul_pow2<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		Self::simd_mul_real(ctx, lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_e_c32s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_conj_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.conj_mul_e_c32s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_add_e_c32s(lhs, rhs, acc)
	}

	#[inline(always)]
	fn simd_conj_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.conj_mul_add_e_c32s(lhs, rhs, acc)
	}

	#[inline(always)]
	fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<S::f32s>() } {
			bytemuck::cast(ctx.mul_f32s(bytemuck::cast(value), bytemuck::cast(value)))
		} else if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<Complex<f32>>() } {
			let value: Complex<f32> = bytemuck::cast(value);
			let v = value.re * value.re + value.im * value.im;
			bytemuck::cast(Complex { re: v, im: v })
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_abs2_add<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<S::f32s>() } {
			bytemuck::cast(ctx.mul_add_f32s(bytemuck::cast(value), bytemuck::cast(value), bytemuck::cast(acc)))
		} else if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<Complex<f32>>() } {
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
		ctx.reduce_sum_c32s(value).into()
	}

	#[inline(always)]
	fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
		ctx.reduce_max_c32s(value).into()
	}

	#[inline(always)]
	fn simd_equal<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdMask<S> {
		panic!()
	}

	#[inline(always)]
	fn simd_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<S::f32s>() } {
			ctx.less_than_f32s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
		} else if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<Complex<f32>>() } {
			assert!(try_const! { core::mem::size_of::<S::m32s>() == core::mem::size_of::<bool>() });

			let lhs: Complex<f32> = bytemuck::cast(real_lhs);
			let rhs: Complex<f32> = bytemuck::cast(real_rhs);
			unsafe { core::mem::transmute_copy(&(lhs.re < rhs.re)) }
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_less_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<S::f32s>() } {
			ctx.less_than_or_equal_f32s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
		} else if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<Complex<f32>>() } {
			assert!(try_const! { core::mem::size_of::<S::m32s>() == core::mem::size_of::<bool>() });

			let lhs: Complex<f32> = bytemuck::cast(real_lhs);
			let rhs: Complex<f32> = bytemuck::cast(real_rhs);
			unsafe { core::mem::transmute_copy(&(lhs.re <= rhs.re)) }
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_greater_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<S::f32s>() } {
			ctx.greater_than_f32s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
		} else if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<Complex<f32>>() } {
			assert!(try_const! { core::mem::size_of::<S::m32s>() == core::mem::size_of::<bool>() });

			let lhs: Complex<f32> = bytemuck::cast(real_lhs);
			let rhs: Complex<f32> = bytemuck::cast(real_rhs);
			unsafe { core::mem::transmute_copy(&(lhs.re > rhs.re)) }
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_greater_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<S::f32s>() } {
			ctx.greater_than_or_equal_f32s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
		} else if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<Complex<f32>>() } {
			assert!(try_const! { core::mem::size_of::<S::m32s>() == core::mem::size_of::<bool>() });

			let lhs: Complex<f32> = bytemuck::cast(real_lhs);
			let rhs: Complex<f32> = bytemuck::cast(real_rhs);
			unsafe { core::mem::transmute_copy(&(lhs.re >= rhs.re)) }
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_select<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<S::f32s>() } {
			bytemuck::cast(ctx.select_f32s_m32s(mask, bytemuck::cast(lhs), bytemuck::cast(rhs)))
		} else if try_const! { core::mem::size_of::<S::c32s>() == core::mem::size_of::<Complex<f32>>() } {
			assert!(try_const! { core::mem::size_of::<S::m32s>() == core::mem::size_of::<bool>() });
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
	fn simd_index_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdIndex<S> {
		f32::simd_index_add(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_index_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdMask<S> {
		f32::simd_index_less_than(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_and_mask<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		f32::simd_and_mask(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_or_mask<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		f32::simd_or_mask(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_not_mask<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>) -> Self::SimdMask<S> {
		f32::simd_not_mask(ctx, mask)
	}

	#[inline(always)]
	fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
		f32::simd_first_true_mask(ctx, value)
	}

	#[inline(always)]
	fn simd_mem_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMemMask<S> {
		ctx.mask_between_m32s((2 * start) as _, (2 * end) as _)
	}

	#[inline(always)]
	fn simd_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMask<S> {
		ctx.mask_between_m32s((2 * start) as _, (2 * end) as _).mask()
	}

	#[inline(always)]
	unsafe fn simd_mask_load_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *const Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mask_load_ptr_c32s(mask, ptr as _)
	}

	#[inline(always)]
	unsafe fn simd_mask_store_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *mut Self::SimdVec<S>, values: Self::SimdVec<S>) {
		ctx.mask_store_ptr_c32s(mask, ptr as _, values);
	}
}

impl ComplexField for ComplexImpl<f64> {
	type Arch = pulp::Arch;
	type Index = u64;
	type Real = f64;
	type SimdCtx<S: Simd> = S;
	type SimdIndex<S: Simd> = S::u64s;
	type SimdMask<S: Simd> = S::m64s;
	type SimdMemMask<S: Simd> = pulp::MemMask<S::m64s>;
	type SimdVec<S: Simd> = S::c64s;
	type Unit = f64;

	const IS_NATIVE_C64: bool = true;
	const IS_REAL: bool = false;
	const SIMD_ABS_SPLIT_REAL_IMAG: bool = true;
	const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::Simd;

	#[inline]
	fn zero_impl() -> Self {
		Complex {
			re: f64::zero_impl(),
			im: f64::zero_impl(),
		}
		.into()
	}

	#[inline]
	fn one_impl() -> Self {
		Complex {
			re: f64::one_impl(),
			im: f64::zero_impl(),
		}
		.into()
	}

	#[inline]
	fn nan_impl() -> Self {
		Complex {
			re: f64::nan_impl(),
			im: f64::nan_impl(),
		}
		.into()
	}

	#[inline]
	fn infinity_impl() -> Self {
		Complex {
			re: f64::infinity_impl(),
			im: f64::infinity_impl(),
		}
		.into()
	}

	#[inline]
	fn from_real_impl(real: &Self::Real) -> Self {
		Complex {
			re: real.clone(),
			im: f64::zero_impl(),
		}
		.into()
	}

	#[inline]
	fn from_f64_impl(real: f64) -> Self {
		Complex {
			re: f64::from_f64_impl(real),
			im: f64::zero_impl(),
		}
		.into()
	}

	#[inline]
	fn real_part_impl(value: &Self) -> Self::Real {
		value.0.re.clone()
	}

	#[inline]
	fn imag_part_impl(value: &Self) -> Self::Real {
		value.0.im.clone()
	}

	#[inline]
	fn copy_impl(value: &Self) -> Self {
		value.clone()
	}

	#[inline]
	fn conj_impl(value: &Self) -> Self {
		Complex {
			re: value.0.re.clone(),
			im: value.0.im.neg_by_ref(),
		}
		.into()
	}

	#[inline]
	fn recip_impl(value: &Self) -> Self {
		let (re, im) = recip_impl(value.0.re.clone(), value.0.im.clone());
		Complex { re, im }.into()
	}

	#[inline]
	fn sqrt_impl(value: &Self) -> Self {
		let (re, im) = sqrt_impl(value.0.re.clone(), value.0.im.clone());
		Complex { re, im }.into()
	}

	#[inline]
	fn abs_impl(value: &Self) -> Self::Real {
		abs_impl(value.0.re.clone(), value.0.im.clone())
	}

	#[inline]
	#[faer_macros::math]
	fn abs1_impl(value: &Self) -> Self::Real {
		abs1(value.0.re) + abs1(value.0.im)
	}

	#[inline]
	#[faer_macros::math]
	fn abs2_impl(value: &Self) -> Self::Real {
		abs2(value.0.re) + abs2(value.0.im)
	}

	#[inline]
	#[faer_macros::math]
	fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self {
		Complex {
			re: lhs.0.re * *rhs,
			im: lhs.0.im * *rhs,
		}
		.into()
	}

	#[inline]
	#[faer_macros::math]
	fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self {
		Complex {
			re: mul_pow2(lhs.0.re, rhs),
			im: mul_pow2(lhs.0.im, rhs),
		}
		.into()
	}

	#[inline]
	#[faer_macros::math]
	fn is_finite_impl(value: &Self) -> bool {
		is_finite(value.0.re) && is_finite(value.0.im)
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
		ctx.splat_c64s(value.0)
	}

	#[inline(always)]
	fn simd_splat_real<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self::Real) -> Self::SimdVec<S> {
		ctx.splat_c64s(Complex { re: *value, im: *value })
	}

	#[inline(always)]
	fn simd_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.add_c64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_sub<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
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
		if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<S::f64s>() } {
			bytemuck::cast(ctx.abs_f64s(bytemuck::cast(value)))
		} else if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<Complex<f64>>() } {
			let value: Complex<f64> = bytemuck::cast(value);
			let v = value.re.abs() + value.im.abs();
			bytemuck::cast(Complex { re: v, im: v })
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_abs_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<S::f64s>() } {
			bytemuck::cast(ctx.abs_f64s(bytemuck::cast(value)))
		} else if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<Complex<f64>>() } {
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
	fn simd_mul_real<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<S::f64s>() } {
			bytemuck::cast(ctx.mul_f64s(bytemuck::cast(lhs), bytemuck::cast(real_rhs)))
		} else if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<Complex<f64>>() } {
			let mut lhs: Complex<f64> = bytemuck::cast(lhs);
			let rhs: Complex<f64> = bytemuck::cast(real_rhs);
			lhs *= rhs.re;
			bytemuck::cast(lhs)
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_mul_pow2<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		Self::simd_mul_real(ctx, lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_e_c64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_conj_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.conj_mul_e_c64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mul_add_e_c64s(lhs, rhs, acc)
	}

	#[inline(always)]
	fn simd_conj_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.conj_mul_add_e_c64s(lhs, rhs, acc)
	}

	#[inline(always)]
	fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<S::f64s>() } {
			bytemuck::cast(ctx.mul_f64s(bytemuck::cast(value), bytemuck::cast(value)))
		} else if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<Complex<f64>>() } {
			let value: Complex<f64> = bytemuck::cast(value);
			let v = value.re * value.re + value.im * value.im;
			bytemuck::cast(Complex { re: v, im: v })
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_abs2_add<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<S::f64s>() } {
			bytemuck::cast(ctx.mul_add_f64s(bytemuck::cast(value), bytemuck::cast(value), bytemuck::cast(acc)))
		} else if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<Complex<f64>>() } {
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
		ctx.reduce_sum_c64s(value).into()
	}

	#[inline(always)]
	fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
		ctx.reduce_max_c64s(value).into()
	}

	#[inline(always)]
	fn simd_equal<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdMask<S> {
		panic!()
	}

	#[inline(always)]
	fn simd_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<S::f64s>() } {
			ctx.less_than_f64s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
		} else if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<Complex<f64>>() } {
			assert!(try_const! { core::mem::size_of::<S::m64s>() == core::mem::size_of::<bool>() });

			let lhs: Complex<f64> = bytemuck::cast(real_lhs);
			let rhs: Complex<f64> = bytemuck::cast(real_rhs);
			unsafe { core::mem::transmute_copy(&(lhs.re < rhs.re)) }
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_less_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<S::f64s>() } {
			ctx.less_than_or_equal_f64s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
		} else if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<Complex<f64>>() } {
			assert!(try_const! { core::mem::size_of::<S::m64s>() == core::mem::size_of::<bool>() });

			let lhs: Complex<f64> = bytemuck::cast(real_lhs);
			let rhs: Complex<f64> = bytemuck::cast(real_rhs);
			unsafe { core::mem::transmute_copy(&(lhs.re <= rhs.re)) }
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_greater_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<S::f64s>() } {
			ctx.greater_than_f64s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
		} else if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<Complex<f64>>() } {
			assert!(try_const! { core::mem::size_of::<S::m64s>() == core::mem::size_of::<bool>() });

			let lhs: Complex<f64> = bytemuck::cast(real_lhs);
			let rhs: Complex<f64> = bytemuck::cast(real_rhs);
			unsafe { core::mem::transmute_copy(&(lhs.re > rhs.re)) }
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_greater_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<S::f64s>() } {
			ctx.greater_than_or_equal_f64s(bytemuck::cast(real_lhs), bytemuck::cast(real_rhs))
		} else if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<Complex<f64>>() } {
			assert!(try_const! { core::mem::size_of::<S::m64s>() == core::mem::size_of::<bool>() });

			let lhs: Complex<f64> = bytemuck::cast(real_lhs);
			let rhs: Complex<f64> = bytemuck::cast(real_rhs);
			unsafe { core::mem::transmute_copy(&(lhs.re >= rhs.re)) }
		} else {
			panic!();
		}
	}

	#[inline(always)]
	fn simd_select<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<S::f64s>() } {
			bytemuck::cast(ctx.select_f64s_m64s(mask, bytemuck::cast(lhs), bytemuck::cast(rhs)))
		} else if try_const! { core::mem::size_of::<S::c64s>() == core::mem::size_of::<Complex<f64>>() } {
			assert!(try_const! { core::mem::size_of::<S::m64s>() == core::mem::size_of::<bool>() });
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
	fn simd_index_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdIndex<S> {
		f64::simd_index_add(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_index_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdMask<S> {
		f64::simd_index_less_than(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_and_mask<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		f64::simd_and_mask(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_or_mask<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		f64::simd_or_mask(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_not_mask<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>) -> Self::SimdMask<S> {
		f64::simd_not_mask(ctx, mask)
	}

	#[inline(always)]
	fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
		f64::simd_first_true_mask(ctx, value)
	}

	#[inline(always)]
	fn simd_mem_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMemMask<S> {
		ctx.mask_between_m64s((2 * start) as _, (2 * end) as _)
	}

	#[inline(always)]
	fn simd_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMask<S> {
		ctx.mask_between_m64s((2 * start) as _, (2 * end) as _).mask()
	}

	#[inline(always)]
	unsafe fn simd_mask_load_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *const Self::SimdVec<S>) -> Self::SimdVec<S> {
		ctx.mask_load_ptr_c64s(mask, ptr as _)
	}

	#[inline(always)]
	unsafe fn simd_mask_store_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *mut Self::SimdVec<S>, values: Self::SimdVec<S>) {
		ctx.mask_store_ptr_c64s(mask, ptr as _, values);
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Symbolic;

impl core::ops::Add for Symbolic {
	type Output = Self;

	fn add(self, _: Self) -> Self {
		Self
	}
}
impl core::ops::Sub for Symbolic {
	type Output = Self;

	fn sub(self, _: Self) -> Self {
		Self
	}
}
impl core::ops::Mul for Symbolic {
	type Output = Self;

	fn mul(self, _: Self) -> Self {
		Self
	}
}
impl core::ops::Div for Symbolic {
	type Output = Self;

	fn div(self, _: Self) -> Self {
		Self
	}
}
impl core::ops::Neg for Symbolic {
	type Output = Self;

	fn neg(self) -> Self {
		Self
	}
}

impl core::ops::Add for &Symbolic {
	type Output = Symbolic;

	fn add(self, _: Self) -> Symbolic {
		Symbolic
	}
}
impl core::ops::Sub for &Symbolic {
	type Output = Symbolic;

	fn sub(self, _: Self) -> Symbolic {
		Symbolic
	}
}
impl core::ops::Mul for &Symbolic {
	type Output = Symbolic;

	fn mul(self, _: Self) -> Symbolic {
		Symbolic
	}
}
impl core::ops::Div for &Symbolic {
	type Output = Symbolic;

	fn div(self, _: Self) -> Symbolic {
		Symbolic
	}
}
impl core::ops::Neg for &Symbolic {
	type Output = Symbolic;

	fn neg(self) -> Symbolic {
		Symbolic
	}
}

impl core::ops::Rem for Symbolic {
	type Output = Self;

	fn rem(self, _: Self) -> Self {
		Self
	}
}
impl num_traits::Zero for Symbolic {
	fn zero() -> Self {
		Self
	}

	fn is_zero(&self) -> bool {
		true
	}
}
impl num_traits::One for Symbolic {
	fn one() -> Self {
		Self
	}

	fn is_one(&self) -> bool {
		true
	}
}
impl num_traits::Num for Symbolic {
	type FromStrRadixErr = core::convert::Infallible;

	fn from_str_radix(_: &str, _: u32) -> Result<Self, Self::FromStrRadixErr> {
		Ok(Self)
	}
}

impl Symbolic {
	#[inline]
	pub fn materialize(len: usize) -> &'static mut [Self] {
		unsafe { core::slice::from_raw_parts_mut(core::ptr::NonNull::dangling().as_ptr(), len) }
	}
}

impl RealField for Symbolic {
	fn epsilon_impl() -> Self {
		Self
	}

	fn nbits_impl() -> usize {
		0
	}

	fn min_positive_impl() -> Self {
		Self
	}

	fn max_positive_impl() -> Self {
		Self
	}

	fn sqrt_min_positive_impl() -> Self {
		Self
	}

	fn sqrt_max_positive_impl() -> Self {
		Self
	}
}

impl ComplexField for Symbolic {
	type Arch = pulp::Scalar;
	type Index = usize;
	type Real = Self;
	type SimdCtx<S: pulp::Simd> = S;
	type SimdIndex<S: pulp::Simd> = ();
	type SimdMask<S: pulp::Simd> = ();
	type SimdMemMask<S: pulp::Simd> = ();
	type SimdVec<S: pulp::Simd> = ();
	type Unit = Self;

	const IS_REAL: bool = true;
	const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::Copy;

	fn zero_impl() -> Self {
		Self
	}

	fn one_impl() -> Self {
		Self
	}

	fn nan_impl() -> Self {
		Self
	}

	fn infinity_impl() -> Self {
		Self
	}

	fn from_real_impl(_: &Self::Real) -> Self {
		Self
	}

	fn from_f64_impl(_: f64) -> Self {
		Self
	}

	fn real_part_impl(_: &Self) -> Self::Real {
		Self
	}

	fn imag_part_impl(_: &Self) -> Self::Real {
		Self
	}

	fn copy_impl(_: &Self) -> Self {
		Self
	}

	fn conj_impl(_: &Self) -> Self {
		Self
	}

	fn recip_impl(_: &Self) -> Self {
		Self
	}

	fn sqrt_impl(_: &Self) -> Self {
		Self
	}

	fn abs_impl(_: &Self) -> Self::Real {
		Self
	}

	fn abs1_impl(_: &Self) -> Self::Real {
		Self
	}

	fn abs2_impl(_: &Self) -> Self::Real {
		Self
	}

	fn mul_real_impl(_: &Self, _: &Self::Real) -> Self {
		Self
	}

	fn mul_pow2_impl(_: &Self, _: &Self::Real) -> Self {
		Self
	}

	fn is_finite_impl(_: &Self) -> bool {
		true
	}

	fn simd_ctx<S: pulp::Simd>(simd: S) -> Self::SimdCtx<S> {
		simd
	}

	fn ctx_from_simd<S: pulp::Simd>(simd: &Self::SimdCtx<S>) -> S {
		*simd
	}

	fn simd_mem_mask_between<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::Index, _: Self::Index) -> Self::SimdMemMask<S> {
		()
	}

	unsafe fn simd_mask_load_raw<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdMemMask<S>, _: *const Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	unsafe fn simd_mask_store_raw<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdMemMask<S>, _: *mut Self::SimdVec<S>, _: Self::SimdVec<S>) {
		()
	}

	fn simd_splat<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: &Self) -> Self::SimdVec<S> {
		()
	}

	fn simd_splat_real<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: &Self::Real) -> Self::SimdVec<S> {
		()
	}

	fn simd_add<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_sub<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_neg<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_conj<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_abs1<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_abs_max<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_mul_real<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_mul_pow2<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_mul<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_conj_mul<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_mul_add<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_conj_mul_add<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_abs2<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_abs2_add<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_reduce_sum<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self {
		Self
	}

	fn simd_reduce_max<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>) -> Self {
		Self
	}

	fn simd_equal<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdMask<S> {
		()
	}

	fn simd_less_than<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdMask<S> {
		()
	}

	fn simd_less_than_or_equal<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdMask<S> {
		()
	}

	fn simd_greater_than<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdMask<S> {
		()
	}

	fn simd_greater_than_or_equal<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdMask<S> {
		()
	}

	fn simd_select<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdMask<S>, _: Self::SimdVec<S>, _: Self::SimdVec<S>) -> Self::SimdVec<S> {
		()
	}

	fn simd_index_select<S: pulp::Simd>(
		_: &Self::SimdCtx<S>,
		_: Self::SimdMask<S>,
		_: Self::SimdIndex<S>,
		_: Self::SimdIndex<S>,
	) -> Self::SimdIndex<S> {
		()
	}

	fn simd_index_splat<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::Index) -> Self::SimdIndex<S> {
		()
	}

	fn simd_index_add<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdIndex<S>, _: Self::SimdIndex<S>) -> Self::SimdIndex<S> {
		()
	}

	fn simd_and_mask<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdMask<S>, _: Self::SimdMask<S>) -> Self::SimdMask<S> {
		()
	}

	fn simd_or_mask<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdMask<S>, _: Self::SimdMask<S>) -> Self::SimdMask<S> {
		()
	}

	fn simd_not_mask<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdMask<S>) -> Self::SimdMask<S> {
		()
	}

	fn simd_first_true_mask<S: pulp::Simd>(_: &Self::SimdCtx<S>, _: Self::SimdMask<S>) -> usize {
		0
	}

	fn simd_mask_between<S: Simd>(_: &Self::SimdCtx<S>, _: Self::Index, _: Self::Index) -> Self::SimdMask<S> {
		()
	}

	fn simd_index_less_than<S: Simd>(_: &Self::SimdCtx<S>, _: Self::SimdIndex<S>, _: Self::SimdIndex<S>) -> Self::SimdMask<S> {
		()
	}
}

pub type c64 = Complex<f64>;
pub type c32 = Complex<f32>;
pub type fx128 = qd::Quad;
pub type cx128 = Complex<fx128>;

pub extern crate num_traits;
pub extern crate pulp;

impl ComplexField for fx128 {
	type Arch = pulp::Arch;
	type Index = u64;
	type Real = Self;
	type SimdCtx<S: Simd> = S;
	type SimdIndex<S: Simd> = S::u64s;
	type SimdMask<S: Simd> = S::m64s;
	type SimdMemMask<S: Simd> = Quad<pulp::MemMask<S::m64s>>;
	type SimdVec<S: Simd> = Quad<S::f64s>;
	type Unit = f64;

	const IS_REAL: bool = true;
	const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::Simd;

	#[inline(always)]
	fn zero_impl() -> Self {
		Self::ZERO
	}

	#[inline(always)]
	fn one_impl() -> Self {
		Quad(1.0, 0.0)
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
	fn from_real_impl(real: &Self::Real) -> Self {
		*real
	}

	#[inline(always)]
	fn from_f64_impl(real: f64) -> Self {
		real.into()
	}

	#[inline(always)]
	fn real_part_impl(value: &Self) -> Self::Real {
		*value
	}

	#[inline(always)]
	fn imag_part_impl(_: &Self) -> Self::Real {
		Self::ZERO
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
		if value.0.abs() == f64::INFINITY {
			Quad::ZERO
		} else {
			Quad::from(1.0) / *value
		}
	}

	#[inline(always)]
	fn sqrt_impl(value: &Self) -> Self {
		value.sqrt()
	}

	#[inline(always)]
	fn abs_impl(value: &Self) -> Self::Real {
		value.abs()
	}

	#[inline(always)]
	fn abs1_impl(value: &Self) -> Self::Real {
		value.abs()
	}

	#[inline(always)]
	fn abs2_impl(value: &Self) -> Self::Real {
		value * value
	}

	#[inline(always)]
	fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self {
		lhs * rhs
	}

	#[inline(always)]
	fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self {
		lhs * rhs
	}

	#[inline(always)]
	fn is_finite_impl(value: &Self) -> bool {
		value.0.is_finite() && value.1.is_finite()
	}

	#[inline(always)]
	fn simd_ctx<S: Simd>(simd: S) -> Self::SimdCtx<S> {
		simd
	}

	#[inline(always)]
	fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> S {
		*ctx
	}

	#[inline(always)]
	fn simd_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMask<S> {
		ctx.mask_between_m64s(start as _, end as _).mask()
	}

	#[inline(always)]
	fn simd_mem_mask_between<S: Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMemMask<S> {
		let n = (core::mem::size_of::<Self::SimdVec<S>>() / core::mem::size_of::<Self>()) as u64;
		let start = start * 2;
		let end = end * 2;

		let mut sa = start.min(n);
		let mut ea = end.min(n);
		let mut sb = start.max(n) - n;
		let mut eb = end.max(n) - n;

		if sa == ea {
			sa = 0;
			ea = 0;
		}
		if sb == eb {
			sb = 0;
			eb = 0;
		}

		let a = f64::simd_mem_mask_between(ctx, sa, ea);
		let b = f64::simd_mem_mask_between(ctx, sb, eb);
		Quad(a, b)
	}

	#[inline(always)]
	unsafe fn simd_mask_load_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *const Self::SimdVec<S>) -> Self::SimdVec<S> {
		unsafe {
			Quad(
				f64::simd_mask_load_raw(ctx, mask.0, &raw const (*ptr).0),
				f64::simd_mask_load_raw(ctx, mask.1, &raw const (*ptr).1),
			)
		}
	}

	#[inline(always)]
	unsafe fn simd_mask_store_raw<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMemMask<S>, ptr: *mut Self::SimdVec<S>, values: Self::SimdVec<S>) {
		unsafe {
			Quad(
				f64::simd_mask_store_raw(ctx, mask.0, &raw mut (*ptr).0, values.0),
				f64::simd_mask_store_raw(ctx, mask.1, &raw mut (*ptr).1, values.1),
			);
		}
	}

	#[inline(always)]
	fn simd_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self) -> Self::SimdVec<S> {
		Quad(ctx.splat_f64s(value.0), ctx.splat_f64s(value.1))
	}

	#[inline(always)]
	fn simd_splat_real<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self::Real) -> Self::SimdVec<S> {
		Quad(ctx.splat_f64s(value.0), ctx.splat_f64s(value.1))
	}

	#[inline(always)]
	fn simd_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::add_estimate(*ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_sub<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::sub_estimate(*ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_neg<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::neg(*ctx, value)
	}

	#[inline(always)]
	fn simd_conj<S: Simd>(_: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		value
	}

	#[inline(always)]
	fn simd_abs1<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::abs(*ctx, value)
	}

	#[inline(always)]
	fn simd_abs_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::abs(*ctx, value)
	}

	#[inline(always)]
	fn simd_mul_real<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::mul(*ctx, lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_mul_pow2<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::mul(*ctx, lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::mul(*ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_conj_mul<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::mul(*ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::add_estimate(*ctx, qd::simd::mul(*ctx, lhs, rhs), acc)
	}

	#[inline(always)]
	fn simd_conj_mul_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::add_estimate(*ctx, qd::simd::mul(*ctx, lhs, rhs), acc)
	}

	#[inline(always)]
	fn simd_abs2<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::mul(*ctx, value, value)
	}

	#[inline(always)]
	fn simd_abs2_add<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		qd::simd::add_estimate(*ctx, qd::simd::mul(*ctx, value, value), acc)
	}

	#[inline(always)]
	fn simd_reduce_sum<S: Simd>(_: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
		let a = value.0;
		let b = value.1;
		let a: &[f64] = bytemuck::cast_slice(core::slice::from_ref(&a));
		let b: &[f64] = bytemuck::cast_slice(core::slice::from_ref(&b));
		let mut acc = Quad::ZERO;

		for (&a, &b) in core::iter::zip(a, b) {
			acc += Quad(a, b);
		}

		acc
	}

	#[inline(always)]
	fn simd_reduce_max<S: Simd>(_: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
		let a = value.0;
		let b = value.1;
		let a: &[f64] = bytemuck::cast_slice(core::slice::from_ref(&a));
		let b: &[f64] = bytemuck::cast_slice(core::slice::from_ref(&b));
		let mut acc = Quad::NEG_INFINITY;

		for (&a, &b) in core::iter::zip(a, b) {
			let val = Quad(a, b);
			if val > acc {
				acc = val
			}
		}

		acc
	}

	#[inline(always)]
	fn simd_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		qd::simd::eq(*ctx, real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		qd::simd::less_than(*ctx, real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_less_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		qd::simd::less_than_or_equal(*ctx, real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_greater_than<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		qd::simd::greater_than(*ctx, real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_greater_than_or_equal<S: Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		qd::simd::greater_than_or_equal(*ctx, real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_select<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		Quad(ctx.select_f64s_m64s(mask, lhs.0, rhs.0), ctx.select_f64s_m64s(mask, lhs.1, rhs.1))
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
		ctx.splat_u64s(value as u64)
	}

	#[inline(always)]
	fn simd_index_add<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdIndex<S> {
		ctx.add_u64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_index_less_than<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdMask<S> {
		ctx.less_than_u64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_and_mask<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		ctx.and_m64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_or_mask<S: Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		ctx.or_m64s(lhs, rhs)
	}

	#[inline(always)]
	fn simd_not_mask<S: Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>) -> Self::SimdMask<S> {
		ctx.not_m64s(mask)
	}

	#[inline(always)]
	fn simd_first_true_mask<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
		ctx.first_true_m64s(value)
	}
}

impl RealField for fx128 {
	#[inline(always)]
	fn epsilon_impl() -> Self {
		let mut x = Quad::EPSILON;
		x.0 *= 8.0;
		x.1 *= 8.0;
		x
	}

	#[inline(always)]
	fn nbits_impl() -> usize {
		100
	}

	#[inline(always)]
	fn min_positive_impl() -> Self {
		Quad::MIN_POSITIVE
	}

	#[inline(always)]
	fn max_positive_impl() -> Self {
		Quad::MIN_POSITIVE.recip()
	}

	#[inline(always)]
	fn sqrt_min_positive_impl() -> Self {
		Quad::MIN_POSITIVE.sqrt()
	}

	#[inline(always)]
	fn sqrt_max_positive_impl() -> Self {
		Quad::MIN_POSITIVE.recip().sqrt()
	}
}
