#![allow(non_snake_case, non_camel_case_types, unused_imports, dead_code)]

use std::collections::HashMap;
use std::ffi::*;
use std::ptr::*;

use aligned_vec::avec;
use diol::config::PlotMetric;
use diol::prelude::*;
use diol::result::BenchResult;
use dyn_stack::{MemBuffer, MemStack};
use equator::assert;
use num_traits::Signed;
use reborrow::*;
use toml::{Table, Value};

use ::faer::diag::Diag;
use ::faer::linalg::cholesky::lblt::factor::LbltParams;
use ::faer::prelude::*;
use ::faer::stats::prelude::*;
use ::faer::{Auto, linalg};
use ::faer_traits::math_utils::*;

use ::nalgebra as na;

#[cfg(eigen)]
use eigen_bench_setup as eig;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
struct fx128(::faer::fx128);
type cx128 = num_complex::Complex<fx128>;

impl core::fmt::Display for fx128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		core::fmt::Debug::fmt(self, f)
	}
}

impl core::ops::Neg for fx128 {
	type Output = fx128;

	#[inline(always)]
	fn neg(self) -> Self::Output {
		Self(-self.0)
	}
}

impl core::ops::Neg for &fx128 {
	type Output = fx128;

	#[inline(always)]
	fn neg(self) -> Self::Output {
		fx128(-self.0)
	}
}
impl core::ops::Add for fx128 {
	type Output = fx128;

	#[inline(always)]
	fn add(self, rhs: Self) -> fx128 {
		Self(self.0 + rhs.0)
	}
}

impl core::ops::AddAssign for fx128 {
	#[inline(always)]
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}
impl core::ops::SubAssign for fx128 {
	#[inline(always)]
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}
impl core::ops::MulAssign for fx128 {
	#[inline(always)]
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}
impl core::ops::DivAssign for fx128 {
	#[inline(always)]
	fn div_assign(&mut self, rhs: Self) {
		*self = *self / rhs;
	}
}
impl core::ops::RemAssign for fx128 {
	#[inline(always)]
	fn rem_assign(&mut self, rhs: Self) {
		*self = *self % rhs;
	}
}
impl core::ops::Sub for fx128 {
	type Output = fx128;

	#[inline(always)]
	fn sub(self, rhs: Self) -> fx128 {
		Self(self.0 - rhs.0)
	}
}

impl core::ops::Mul for fx128 {
	type Output = fx128;

	#[inline(always)]
	fn mul(self, rhs: Self) -> fx128 {
		Self(self.0 * rhs.0)
	}
}

impl core::ops::Div for fx128 {
	type Output = fx128;

	#[inline(always)]
	fn div(self, rhs: Self) -> fx128 {
		Self(self.0 / rhs.0)
	}
}

impl core::ops::Rem for fx128 {
	type Output = fx128;

	#[inline(always)]
	fn rem(self, rhs: Self) -> fx128 {
		Self(self.0 % rhs.0)
	}
}

impl core::ops::Add for &fx128 {
	type Output = fx128;

	#[inline(always)]
	fn add(self, rhs: Self) -> fx128 {
		fx128(self.0 + rhs.0)
	}
}

impl core::ops::Sub for &fx128 {
	type Output = fx128;

	#[inline(always)]
	fn sub(self, rhs: Self) -> fx128 {
		fx128(self.0 - rhs.0)
	}
}

impl core::ops::Mul for &fx128 {
	type Output = fx128;

	#[inline(always)]
	fn mul(self, rhs: Self) -> fx128 {
		fx128(self.0 * rhs.0)
	}
}

impl core::ops::Div for &fx128 {
	type Output = fx128;

	#[inline(always)]
	fn div(self, rhs: Self) -> fx128 {
		fx128(self.0 / rhs.0)
	}
}

impl core::ops::Rem for &fx128 {
	type Output = fx128;

	#[inline(always)]
	fn rem(self, rhs: Self) -> fx128 {
		fx128(self.0 % rhs.0)
	}
}

impl num_traits::Zero for fx128 {
	fn zero() -> Self {
		Self(::faer::fx128::ZERO)
	}

	fn is_zero(&self) -> bool {
		*self == Self::zero()
	}
}

impl num_traits::One for fx128 {
	fn one() -> Self {
		Self(::faer::fx128::ONE)
	}

	fn is_one(&self) -> bool {
		*self == Self::one()
	}
}

impl num_traits::Num for fx128 {
	type FromStrRadixErr = ();

	fn from_str_radix(_: &str, _: u32) -> Result<Self, Self::FromStrRadixErr> {
		Err(())
	}
}

impl faer_traits::ComplexField for fx128 {
	type Arch = <::faer::fx128 as ::faer::traits::ComplexField>::Arch;
	type Index = <::faer::fx128 as ::faer::traits::ComplexField>::Index;
	type Real = Self;
	type SimdCtx<S: pulp::Simd> = <::faer::fx128 as ::faer::traits::ComplexField>::SimdCtx<S>;
	type SimdIndex<S: pulp::Simd> = <::faer::fx128 as ::faer::traits::ComplexField>::SimdIndex<S>;
	type SimdMask<S: pulp::Simd> = <::faer::fx128 as ::faer::traits::ComplexField>::SimdMask<S>;
	type SimdMemMask<S: pulp::Simd> = <::faer::fx128 as ::faer::traits::ComplexField>::SimdMemMask<S>;
	type SimdVec<S: pulp::Simd> = <::faer::fx128 as ::faer::traits::ComplexField>::SimdVec<S>;
	type Unit = <::faer::fx128 as ::faer::traits::ComplexField>::Unit;

	const IS_REAL: bool = true;
	const SIMD_CAPABILITIES: faer_traits::SimdCapabilities = ::faer::fx128::SIMD_CAPABILITIES;

	#[inline(always)]
	fn zero_impl() -> Self {
		Self(::faer::fx128::zero_impl())
	}

	#[inline(always)]
	fn one_impl() -> Self {
		Self(::faer::fx128::one_impl())
	}

	#[inline(always)]
	fn nan_impl() -> Self {
		Self(::faer::fx128::nan_impl())
	}

	#[inline(always)]
	fn infinity_impl() -> Self {
		Self(::faer::fx128::infinity_impl())
	}

	#[inline(always)]
	fn from_real_impl(real: &Self::Real) -> Self {
		Self(::faer::fx128::from_real_impl(&real.0))
	}

	#[inline(always)]
	fn from_f64_impl(real: f64) -> Self {
		Self(::faer::fx128::from_f64_impl(real))
	}

	#[inline(always)]
	fn real_part_impl(value: &Self) -> Self::Real {
		Self(::faer::fx128::real_part_impl(&value.0))
	}

	#[inline(always)]
	fn imag_part_impl(value: &Self) -> Self::Real {
		Self(::faer::fx128::imag_part_impl(&value.0))
	}

	#[inline(always)]
	fn copy_impl(value: &Self) -> Self {
		Self(::faer::fx128::copy_impl(&value.0))
	}

	#[inline(always)]
	fn conj_impl(value: &Self) -> Self {
		Self(::faer::fx128::conj_impl(&value.0))
	}

	#[inline(always)]
	fn recip_impl(value: &Self) -> Self {
		Self(::faer::fx128::recip_impl(&value.0))
	}

	#[inline(always)]
	fn sqrt_impl(value: &Self) -> Self {
		Self(::faer::fx128::sqrt_impl(&value.0))
	}

	#[inline(always)]
	fn abs_impl(value: &Self) -> Self::Real {
		Self(::faer::fx128::abs_impl(&value.0))
	}

	#[inline(always)]
	fn abs1_impl(value: &Self) -> Self::Real {
		Self(::faer::fx128::abs1_impl(&value.0))
	}

	#[inline(always)]
	fn abs2_impl(value: &Self) -> Self::Real {
		Self(::faer::fx128::abs2_impl(&value.0))
	}

	#[inline(always)]
	fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self {
		Self(::faer::fx128::mul_real_impl(&lhs.0, &rhs.0))
	}

	#[inline(always)]
	fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self {
		Self(::faer::fx128::mul_pow2_impl(&lhs.0, &rhs.0))
	}

	#[inline(always)]
	fn is_finite_impl(value: &Self) -> bool {
		::faer::fx128::is_finite_impl(&value.0)
	}

	#[inline(always)]
	fn simd_ctx<S: pulp::Simd>(simd: S) -> Self::SimdCtx<S> {
		::faer::fx128::simd_ctx(simd)
	}

	#[inline(always)]
	fn ctx_from_simd<S: pulp::Simd>(ctx: &Self::SimdCtx<S>) -> S {
		::faer::fx128::ctx_from_simd(ctx)
	}

	#[inline(always)]
	fn simd_mask_between<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMask<S> {
		::faer::fx128::simd_mask_between(ctx, start, end)
	}

	#[inline(always)]
	fn simd_mem_mask_between<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, start: Self::Index, end: Self::Index) -> Self::SimdMemMask<S> {
		::faer::fx128::simd_mem_mask_between(ctx, start, end)
	}

	#[inline(always)]
	unsafe fn simd_mask_load_raw<S: pulp::Simd>(
		ctx: &Self::SimdCtx<S>,
		mask: Self::SimdMemMask<S>,
		ptr: *const Self::SimdVec<S>,
	) -> Self::SimdVec<S> {
		::faer::fx128::simd_mask_load_raw(ctx, mask, ptr)
	}

	#[inline(always)]
	unsafe fn simd_mask_store_raw<S: pulp::Simd>(
		ctx: &Self::SimdCtx<S>,
		mask: Self::SimdMemMask<S>,
		ptr: *mut Self::SimdVec<S>,
		values: Self::SimdVec<S>,
	) {
		::faer::fx128::simd_mask_store_raw(ctx, mask, ptr, values)
	}

	#[inline(always)]
	fn simd_splat<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: &Self) -> Self::SimdVec<S> {
		::faer::fx128::simd_splat(ctx, &value.0)
	}

	#[inline(always)]
	fn simd_splat_real<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: &Self::Real) -> Self::SimdVec<S> {
		::faer::fx128::simd_splat_real(ctx, &value.0)
	}

	#[inline(always)]
	fn simd_add<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_add(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_sub<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_sub(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_neg<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_neg(ctx, value)
	}

	#[inline(always)]
	fn simd_conj<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_conj(ctx, value)
	}

	#[inline(always)]
	fn simd_abs1<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_abs1(ctx, value)
	}

	#[inline(always)]
	fn simd_abs_max<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_abs_max(ctx, value)
	}

	#[inline(always)]
	fn simd_mul_real<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_mul_real(ctx, lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_mul_pow2<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_mul_pow2(ctx, lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_mul<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_mul(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_conj_mul<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_conj_mul(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_mul_add<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_mul_add(ctx, lhs, rhs, acc)
	}

	#[inline(always)]
	fn simd_conj_mul_add<S: pulp::Simd>(
		ctx: &Self::SimdCtx<S>,
		lhs: Self::SimdVec<S>,
		rhs: Self::SimdVec<S>,
		acc: Self::SimdVec<S>,
	) -> Self::SimdVec<S> {
		::faer::fx128::simd_conj_mul_add(ctx, lhs, rhs, acc)
	}

	#[inline(always)]
	fn simd_abs2<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_abs2(ctx, value)
	}

	#[inline(always)]
	fn simd_abs2_add<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>, acc: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_abs2_add(ctx, value, acc)
	}

	#[inline(always)]
	fn simd_reduce_sum<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
		Self(::faer::fx128::simd_reduce_sum(ctx, value))
	}

	#[inline(always)]
	fn simd_reduce_max<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdVec<S>) -> Self {
		Self(::faer::fx128::simd_reduce_max(ctx, value))
	}

	#[inline(always)]
	fn simd_equal<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		::faer::fx128::simd_equal(ctx, real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_less_than<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		::faer::fx128::simd_less_than(ctx, real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_less_than_or_equal<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		::faer::fx128::simd_less_than_or_equal(ctx, real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_greater_than<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, real_lhs: Self::SimdVec<S>, real_rhs: Self::SimdVec<S>) -> Self::SimdMask<S> {
		::faer::fx128::simd_greater_than(ctx, real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_greater_than_or_equal<S: pulp::Simd>(
		ctx: &Self::SimdCtx<S>,
		real_lhs: Self::SimdVec<S>,
		real_rhs: Self::SimdVec<S>,
	) -> Self::SimdMask<S> {
		::faer::fx128::simd_greater_than_or_equal(ctx, real_lhs, real_rhs)
	}

	#[inline(always)]
	fn simd_select<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>, lhs: Self::SimdVec<S>, rhs: Self::SimdVec<S>) -> Self::SimdVec<S> {
		::faer::fx128::simd_select(ctx, mask, lhs, rhs)
	}

	#[inline(always)]
	fn simd_index_select<S: pulp::Simd>(
		ctx: &Self::SimdCtx<S>,
		mask: Self::SimdMask<S>,
		lhs: Self::SimdIndex<S>,
		rhs: Self::SimdIndex<S>,
	) -> Self::SimdIndex<S> {
		::faer::fx128::simd_index_select(ctx, mask, lhs, rhs)
	}

	#[inline(always)]
	fn simd_index_splat<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: Self::Index) -> Self::SimdIndex<S> {
		::faer::fx128::simd_index_splat(ctx, value)
	}

	#[inline(always)]
	fn simd_index_add<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdIndex<S> {
		::faer::fx128::simd_index_add(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_index_less_than<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdIndex<S>, rhs: Self::SimdIndex<S>) -> Self::SimdMask<S> {
		::faer::fx128::simd_index_less_than(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_and_mask<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		::faer::fx128::simd_and_mask(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_or_mask<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, lhs: Self::SimdMask<S>, rhs: Self::SimdMask<S>) -> Self::SimdMask<S> {
		::faer::fx128::simd_or_mask(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn simd_not_mask<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, mask: Self::SimdMask<S>) -> Self::SimdMask<S> {
		::faer::fx128::simd_not_mask(ctx, mask)
	}

	#[inline(always)]
	fn simd_first_true_mask<S: pulp::Simd>(ctx: &Self::SimdCtx<S>, value: Self::SimdMask<S>) -> usize {
		::faer::fx128::simd_first_true_mask(ctx, value)
	}
}

impl faer_traits::RealField for fx128 {
	#[inline(always)]
	fn epsilon_impl() -> Self {
		Self(::faer::fx128::epsilon_impl())
	}

	#[inline(always)]
	fn nbits_impl() -> usize {
		::faer::fx128::nbits_impl()
	}

	#[inline(always)]
	fn min_positive_impl() -> Self {
		Self(::faer::fx128::min_positive_impl())
	}

	#[inline(always)]
	fn max_positive_impl() -> Self {
		Self(::faer::fx128::max_positive_impl())
	}

	#[inline(always)]
	fn sqrt_min_positive_impl() -> Self {
		Self(::faer::fx128::sqrt_min_positive_impl())
	}

	#[inline(always)]
	fn sqrt_max_positive_impl() -> Self {
		Self(::faer::fx128::sqrt_max_positive_impl())
	}
}

impl simba::scalar::SupersetOf<f32> for fx128 {
	#[inline(always)]
	fn is_in_subset(&self) -> bool {
		self.0.1 == 0.0 && self.0.0 as f32 as f64 == self.0.0
	}

	#[inline(always)]
	fn to_subset(&self) -> Option<f32> {
		if simba::scalar::SupersetOf::<f32>::is_in_subset(self) {
			Some(self.0.0 as f32)
		} else {
			None
		}
	}

	#[inline(always)]
	fn to_subset_unchecked(&self) -> f32 {
		self.0.0 as f32
	}

	#[inline(always)]
	fn from_subset(element: &f32) -> Self {
		Self(((*element) as f64).into())
	}
}
impl simba::scalar::SupersetOf<f64> for fx128 {
	#[inline(always)]
	fn is_in_subset(&self) -> bool {
		self.0.1 == 0.0
	}

	#[inline(always)]
	fn to_subset(&self) -> Option<f64> {
		if simba::scalar::SupersetOf::<f64>::is_in_subset(self) {
			Some(self.0.0)
		} else {
			None
		}
	}

	#[inline(always)]
	fn to_subset_unchecked(&self) -> f64 {
		self.0.0
	}

	#[inline(always)]
	fn from_subset(element: &f64) -> Self {
		Self((*element).into())
	}
}

impl simba::scalar::SubsetOf<fx128> for fx128 {
	#[inline(always)]
	fn to_superset(&self) -> fx128 {
		*self
	}

	#[inline(always)]
	fn from_superset(element: &fx128) -> Option<Self> {
		Some(*element)
	}

	#[inline(always)]
	fn from_superset_unchecked(element: &fx128) -> Self {
		*element
	}

	#[inline(always)]
	fn is_in_subset(_: &fx128) -> bool {
		true
	}
}

impl num_traits::FromPrimitive for fx128 {
	#[inline(always)]
	fn from_i64(n: i64) -> Option<Self> {
		f64::from_i64(n).map(::faer::fx128::from_f64).map(Self)
	}

	#[inline(always)]
	fn from_u64(n: u64) -> Option<Self> {
		f64::from_u64(n).map(::faer::fx128::from_f64).map(Self)
	}
}

impl ::nalgebra::SimdValue for fx128 {
	type Element = Self;
	type SimdBool = bool;

	const LANES: usize = 1;

	#[inline(always)]
	fn splat(val: Self::Element) -> Self {
		val
	}

	#[inline(always)]
	fn extract(&self, _: usize) -> Self::Element {
		*self
	}

	#[inline(always)]
	unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
		*self
	}

	#[inline(always)]
	fn replace(&mut self, _: usize, val: Self::Element) {
		*self = val;
	}

	#[inline(always)]
	unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
		*self = val;
	}

	#[inline(always)]
	fn select(self, cond: Self::SimdBool, other: Self) -> Self {
		if cond { self } else { other }
	}

	#[inline(always)]
	fn map_lanes(self, f: impl Fn(Self::Element) -> Self::Element) -> Self
	where
		Self: Clone,
	{
		f(self)
	}

	#[inline(always)]
	fn zip_map_lanes(self, b: Self, f: impl Fn(Self::Element, Self::Element) -> Self::Element) -> Self
	where
		Self: Clone,
	{
		f(self, b)
	}
}
impl ::nalgebra::Field for fx128 {}

impl approx::AbsDiffEq for fx128 {
	type Epsilon = Self;

	fn default_epsilon() -> Self::Epsilon {
		Self(::faer::fx128::EPSILON)
	}

	fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
		(self - other).0.abs() <= epsilon.0
	}
}

impl approx::UlpsEq for fx128 {
	fn default_max_ulps() -> u32 {
		0
	}

	fn ulps_eq(&self, other: &Self, _: Self::Epsilon, _: u32) -> bool {
		self == other
	}
}

impl approx::RelativeEq for fx128 {
	fn default_max_relative() -> Self::Epsilon {
		Self(::faer::fx128::EPSILON)
	}

	fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
		if self == other {
			return true;
		}
		if self.0.0.is_infinite() || other.0.0.is_infinite() {
			return false;
		}
		let abs_diff = (self - other).0.abs();

		// For when the numbers are really close together

		if abs_diff <= epsilon.0 {
			return true;
		}

		let abs_self = (*self).0.abs();

		let abs_other = (*other).0.abs();

		let largest = if abs_other > abs_self { abs_other } else { abs_self };

		// Use a relative difference comparison

		abs_diff <= largest * max_relative.0
	}
}

impl num_traits::Signed for fx128 {
	fn abs(&self) -> Self {
		Self(self.0.abs())
	}

	fn abs_sub(&self, other: &Self) -> Self {
		Self((self - other).0.abs())
	}

	fn is_negative(&self) -> bool {
		self.0.0 < 0.0
	}

	fn is_positive(&self) -> bool {
		self.0.0 > 0.0
	}

	fn signum(&self) -> Self {
		Self((self.0.0.signum()).into())
	}
}

impl ::nalgebra::ComplexField for fx128 {
	type RealField = Self;

	/// Builds a pure-real complex number from the given value.
	fn from_real(re: Self::RealField) -> Self {
		re
	}

	/// The real part of this complex number.
	fn real(self) -> Self::RealField {
		self
	}

	/// The imaginary part of this complex number.
	fn imaginary(self) -> Self::RealField {
		Self(::faer::fx128::ZERO)
	}

	/// The modulus of this complex number.
	fn modulus(self) -> Self::RealField {
		Self(self.0.abs())
	}

	/// The squared modulus of this complex number.
	fn modulus_squared(self) -> Self::RealField {
		self * self
	}

	/// The argument of this complex number.
	fn argument(self) -> Self::RealField {
		if self.0.0.is_sign_negative() {
			Self(::faer::fx128::PI)
		} else {
			Self(::faer::fx128::ZERO)
		}
	}

	/// The sum of the absolute value of this complex number's real and imaginary part.
	fn norm1(self) -> Self::RealField {
		Self(self.0.abs())
	}

	/// Multiplies this complex number by `factor`.
	fn scale(self, factor: Self::RealField) -> Self {
		self * factor
	}

	/// Divides this complex number by `factor`.
	fn unscale(self, factor: Self::RealField) -> Self {
		self / factor
	}

	fn floor(self) -> Self {
		todo!()
	}

	fn ceil(self) -> Self {
		todo!()
	}

	fn round(self) -> Self {
		todo!()
	}

	fn trunc(self) -> Self {
		todo!()
	}

	fn fract(self) -> Self {
		todo!()
	}

	fn mul_add(self, a: Self, b: Self) -> Self {
		self * a + b
	}

	/// The absolute value of this complex number: `self / self.signum()`.
	///
	/// This is equivalent to `self.modulus()`.
	fn abs(self) -> Self::RealField {
		self.modulus()
	}

	/// Computes (self.conjugate() * self + other.conjugate() * other).sqrt()
	fn hypot(self, other: Self) -> Self::RealField {
		hypot(&self, &other)
	}

	fn recip(self) -> Self {
		recip(&self)
	}

	fn conjugate(self) -> Self {
		self
	}

	fn sin(self) -> Self {
		todo!()
	}

	fn cos(self) -> Self {
		todo!()
	}

	fn sin_cos(self) -> (Self, Self) {
		todo!()
	}

	fn tan(self) -> Self {
		todo!()
	}

	fn asin(self) -> Self {
		todo!()
	}

	fn acos(self) -> Self {
		todo!()
	}

	fn atan(self) -> Self {
		todo!()
	}

	fn sinh(self) -> Self {
		todo!()
	}

	fn cosh(self) -> Self {
		todo!()
	}

	fn tanh(self) -> Self {
		todo!()
	}

	fn asinh(self) -> Self {
		todo!()
	}

	fn acosh(self) -> Self {
		todo!()
	}

	fn atanh(self) -> Self {
		todo!()
	}

	fn log(self, _: Self::RealField) -> Self {
		todo!()
	}

	fn log2(self) -> Self {
		todo!()
	}

	fn log10(self) -> Self {
		todo!()
	}

	fn ln(self) -> Self {
		todo!()
	}

	fn ln_1p(self) -> Self {
		todo!()
	}

	fn sqrt(self) -> Self {
		Self(self.0.sqrt())
	}

	fn exp(self) -> Self {
		todo!()
	}

	fn exp2(self) -> Self {
		todo!()
	}

	fn exp_m1(self) -> Self {
		todo!()
	}

	fn powi(self, _: i32) -> Self {
		todo!()
	}

	fn powf(self, _: Self::RealField) -> Self {
		todo!()
	}

	fn powc(self, _: Self) -> Self {
		todo!()
	}

	fn cbrt(self) -> Self {
		todo!()
	}

	fn is_finite(&self) -> bool {
		self.0.is_finite()
	}

	fn try_sqrt(self) -> Option<Self> {
		if !self.is_negative() { Some(Self(self.0.sqrt())) } else { None }
	}
}
impl ::nalgebra::RealField for fx128 {
	fn is_sign_positive(&self) -> bool {
		self.0.0.is_sign_positive()
	}

	fn is_sign_negative(&self) -> bool {
		self.0.0.is_sign_negative()
	}

	fn copysign(self, sign: Self) -> Self {
		self.abs() * sign
	}

	fn max(self, other: Self) -> Self {
		if self > other { self } else { other }
	}

	fn min(self, other: Self) -> Self {
		if self < other { self } else { other }
	}

	fn clamp(self, min: Self, max: Self) -> Self {
		self.min(max).max(min)
	}

	fn atan2(self, _: Self) -> Self {
		todo!()
	}

	fn min_value() -> Option<Self> {
		Some(Self(::faer::fx128::MIN_POSITIVE))
	}

	fn max_value() -> Option<Self> {
		Some(Self(::faer::fx128::MAX))
	}

	fn pi() -> Self {
		Self(::faer::fx128::PI)
	}

	fn two_pi() -> Self {
		Self(::faer::fx128::PI * 2.0.into())
	}

	fn frac_pi_2() -> Self {
		Self(::faer::fx128::PI * 0.5.into())
	}

	fn frac_pi_3() -> Self {
		Self(::faer::fx128::PI / 3.0.into())
	}

	fn frac_pi_4() -> Self {
		Self(::faer::fx128::PI * 0.25.into())
	}

	fn frac_pi_6() -> Self {
		Self(::faer::fx128::PI / 6.0.into())
	}

	fn frac_pi_8() -> Self {
		Self(::faer::fx128::PI * 0.125.into())
	}

	fn frac_1_pi() -> Self {
		recip(&Self::pi())
	}

	fn frac_2_pi() -> Self {
		Self(2.0.into()) * recip(&Self::pi())
	}

	fn frac_2_sqrt_pi() -> Self {
		Self(2.0.into()) * sqrt(&recip(&Self::pi()))
	}

	fn e() -> Self {
		todo!()
	}

	fn log2_e() -> Self {
		todo!()
	}

	fn log10_e() -> Self {
		todo!()
	}

	fn ln_2() -> Self {
		todo!()
	}

	fn ln_10() -> Self {
		todo!()
	}
}

#[cfg(eigen)]
fn eigen_dtype<T: Scalar>() -> usize {
	if T::IS_NATIVE_F32 {
		eig::F32
	} else if T::IS_NATIVE_F64 {
		eig::F64
	} else if T::IS_NATIVE_C32 {
		eig::C32
	} else if T::IS_NATIVE_C64 {
		eig::C64
	} else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<fx128>() {
		eig::FX128
	} else if core::any::TypeId::of::<T>() == core::any::TypeId::of::<cx128>() {
		eig::CX128
	} else {
		panic!()
	}
}

#[cfg(any(openblas, mkl, blis))]
use lapack_sys as la;
#[cfg(any(openblas, mkl, blis))]
extern crate lapack_src;
#[cfg(any(openblas, mkl, blis))]
extern crate openmp_sys;

#[cfg(openblas)]
extern crate openblas_src;

#[cfg(mkl)]
extern crate intel_mkl_src;

#[cfg(blis)]
extern crate blis_src;

#[cfg(any(openblas, mkl, blis))]
unsafe extern "C" {
	#[cfg(openblas)]
	fn openblas_set_num_threads(num: i32) -> c_void;
	#[cfg(openblas)]
	fn goto_set_num_threads(num: i32) -> c_void;

	#[cfg(mkl)]
	fn MKL_Set_Num_Threads(num: i32) -> c_void;

	#[cfg(blis)]
	fn bli_thread_set_num_threads(num: i64) -> c_void;

	fn omp_set_num_threads(num: i32) -> c_void;

	fn sgetc2_(n: *const c_int, A: *mut f32, lda: *const c_int, ipiv: *mut c_int, jpiv: *mut c_int, info: *mut c_int) -> c_void;
	fn dgetc2_(n: *const c_int, A: *mut f64, lda: *const c_int, ipiv: *mut c_int, jpiv: *mut c_int, info: *mut c_int) -> c_void;
	fn cgetc2_(n: *const c_int, A: *mut c32, lda: *const c_int, ipiv: *mut c_int, jpiv: *mut c_int, info: *mut c_int) -> c_void;
	fn zgetc2_(n: *const c_int, A: *mut c64, lda: *const c_int, ipiv: *mut c_int, jpiv: *mut c_int, info: *mut c_int) -> c_void;
}

fn lapack_set_num_threads(parallel: Par) {
	let _ = parallel;
	#[cfg(any(openblas, mkl, blis))]
	match parallel {
		Par::Seq => unsafe {
			#[cfg(openblas)]
			openblas_set_num_threads(1);
			#[cfg(openblas)]
			goto_set_num_threads(1);

			#[cfg(mkl)]
			MKL_Set_Num_Threads(1);

			#[cfg(blis)]
			bli_thread_set_num_threads(1);

			omp_set_num_threads(1);
		},
		Par::Rayon(nthreads) => unsafe {
			let nthreads = nthreads.get();

			#[cfg(openblas)]
			openblas_set_num_threads(nthreads as _);
			#[cfg(openblas)]
			goto_set_num_threads(nthreads as _);

			#[cfg(mkl)]
			MKL_Set_Num_Threads(nthreads as _);

			#[cfg(blis)]
			bli_thread_set_num_threads(nthreads as _);

			omp_set_num_threads(nthreads as _);
		},
	};
}

trait Scalar: Copy + faer_traits::ComplexField + na::ComplexField {
	const IS_NATIVE: bool = Self::IS_NATIVE_F32 || Self::IS_NATIVE_C32 || Self::IS_NATIVE_F64 || Self::IS_NATIVE_C64;

	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self>;
}

trait Lib {
	const LAPACK: bool = false;
	const FAER: bool = false;
	const NALGEBRA: bool = false;
	const EIGEN: bool = false;
}

trait Thread {
	const SEQ: bool = false;
	const PAR: bool = false;
}

struct eigen;

#[cfg(openblas)]
struct openblas;
#[cfg(openblas)]
type lapack = openblas;

#[cfg(mkl)]
struct mkl;
#[cfg(mkl)]
type lapack = mkl;

#[cfg(blis)]
struct blis;
#[cfg(blis)]
type lapack = blis;

#[cfg(not(any(openblas, mkl, blis)))]
struct lapack;

struct faer;
struct nalgebra;

struct seq;
struct par;

impl Thread for seq {
	const SEQ: bool = true;
}
impl Thread for par {
	const PAR: bool = true;
}

impl Lib for faer {
	const FAER: bool = true;
}
impl Lib for lapack {
	const LAPACK: bool = true;
}
impl Lib for nalgebra {
	const NALGEBRA: bool = true;
}
impl Lib for eigen {
	const EIGEN: bool = true;
}

impl Scalar for f64 {
	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self> {
		CwiseMatDistribution {
			nrows,
			ncols,
			dist: StandardNormal,
		}
		.rand(rng)
	}
}

impl Scalar for fx128 {
	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self> {
		zip!(&f64::random(rng, nrows, ncols)).map(|unzip!(&x)| Self(x.into()))
	}
}
impl Scalar for cx128 {
	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self> {
		zip!(&c64::random(rng, nrows, ncols)).map(|unzip!(&x)| Self::new(fx128(x.re.into()), fx128(x.im.into())))
	}
}
impl Scalar for f32 {
	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self> {
		CwiseMatDistribution {
			nrows,
			ncols,
			dist: StandardNormal,
		}
		.rand(rng)
	}
}
impl Scalar for c32 {
	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self> {
		CwiseMatDistribution {
			nrows,
			ncols,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand(rng)
	}
}
impl Scalar for c64 {
	fn random(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> Mat<Self> {
		CwiseMatDistribution {
			nrows,
			ncols,
			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
		}
		.rand(rng)
	}
}

#[cfg(eigen)]
fn bench_eigen<T: Scalar>(bencher: Bencher, decomp: usize, A: MatRef<'_, T>) {
	let dtype = eigen_dtype::<T>();
	let (m, n) = A.shape();
	let cs = A.col_stride() as usize;

	unsafe {
		let eig = eig::libeigen_make_decomp(decomp, dtype, m, n);
		bencher.bench(|| {
			eig::libeigen_factorize(decomp, dtype, eig, A.as_ptr() as _, m, n, cs);
		});
		eig::libeigen_free_decomp(decomp, dtype, eig);
	}

	return;
}

fn llt<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A * A.adjoint() + Scale(from_f64::<T>(m as f64)) * Mat::<T>::identity(n, n);

	let mut L = Mat::zeros(n, n);
	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::cholesky::llt::factor::cholesky_in_place_scratch::<T>(n, parallel, params));
	let stack = MemStack::new(stack);

	#[cfg(eigen)]
	if Lib::EIGEN {
		return bench_eigen(bencher, eig::LLT, A.rb());
	}

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				L.copy_from_triangular_lower(&A);
				linalg::cholesky::llt::factor::cholesky_in_place(L.rb_mut(), Default::default(), parallel, stack, params).unwrap();
			} else if Lib::LAPACK {
				L.copy_from_triangular_lower(&A);
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::spotrf_(
							&(b'L' as i8),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dpotrf_(
							&(b'L' as i8),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cpotrf_(
							&(b'L' as i8),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zpotrf_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.cholesky()
					.unwrap();
				};
			}
		})
	}
}

fn ldlt<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A * A.adjoint() + Scale(from_f64::<T>(m as f64)) * Mat::<T>::identity(n, n);
	let mut L = Mat::zeros(n, n);
	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::cholesky::ldlt::factor::cholesky_in_place_scratch::<T>(n, parallel, params));
	let stack = MemStack::new(stack);

	#[cfg(eigen)]
	if Lib::EIGEN {
		return bench_eigen(bencher, eig::LDLT, A.rb());
	}

	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::ldlt::factor::cholesky_in_place(L.rb_mut(), Default::default(), parallel, stack, params).unwrap();
		})
	}
}

fn lblt<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	#[cfg(eigen)]
	if Lib::EIGEN {
		return bencher.skip();
	}

	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *avec![0usize; n];
	let bwd = &mut *avec![0usize; n];

	let params = LbltParams {
		pivoting: linalg::cholesky::lblt::factor::PivotingStrategy::Partial,
		..Auto::<T>::auto()
	}
	.into();

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut lwork = core::mem::zeroed();
			la::ssytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_F64 {
			let mut lwork = core::mem::zeroed();
			la::dsytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_C32 {
			let mut lwork = core::mem::zeroed();
			la::csytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut lwork = core::mem::zeroed();
			la::zsytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else {
			0
		}
	};

	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let stack = &mut MemBuffer::new(linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, T>(n, parallel, params));
	let stack = MemStack::new(stack);

	if Lib::FAER || (Lib::LAPACK && T::IS_NATIVE) {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			if Lib::FAER {
				linalg::cholesky::lblt::factor::cholesky_in_place(L.rb_mut(), subdiag.rb_mut(), fwd, bwd, parallel, stack, params);
			} else if Lib::LAPACK {
				fwd.fill(0);

				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::ssytrf_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dsytrf_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::csytrf_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zsytrf_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			}
		})
	}
}

fn lblt_diag<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	#[cfg(eigen)]
	if Lib::EIGEN {
		return bencher.skip();
	}

	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *avec![0usize; n];
	let bwd = &mut *avec![0usize; n];

	let params = LbltParams {
		pivoting: linalg::cholesky::lblt::factor::PivotingStrategy::PartialDiag,
		..Auto::<T>::auto()
	}
	.into();

	let stack = &mut MemBuffer::new(linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, T>(n, parallel, params));
	let stack = MemStack::new(stack);
	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::lblt::factor::cholesky_in_place(L.rb_mut(), subdiag.rb_mut(), fwd, bwd, parallel, stack, params);
		})
	}
}

fn lblt_rook<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	#[cfg(eigen)]
	if Lib::EIGEN {
		return bencher.skip();
	}

	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *avec![0usize; n];
	let bwd = &mut *avec![0usize; n];

	let params = LbltParams {
		pivoting: linalg::cholesky::lblt::factor::PivotingStrategy::Rook,
		..Auto::<T>::auto()
	}
	.into();

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut lwork = core::mem::zeroed();
			la::ssytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_F64 {
			let mut lwork = core::mem::zeroed();
			la::dsytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_C32 {
			let mut lwork = core::mem::zeroed();
			la::csytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut lwork = core::mem::zeroed();
			la::zsytrf_(
				&(b'L' as _),
				(&n) as *const _ as *const _,
				L.as_ptr_mut() as _,
				(&L.col_stride()) as *const _ as *const _,
				fwd.as_mut_ptr() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else {
			0
		}
	};

	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let stack = &mut MemBuffer::new(linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, T>(n, parallel, params));
	let stack = MemStack::new(stack);
	if Lib::FAER || (Lib::LAPACK && T::IS_NATIVE) {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			if Lib::FAER {
				linalg::cholesky::lblt::factor::cholesky_in_place(L.rb_mut(), subdiag.rb_mut(), fwd, bwd, parallel, stack, params);
			} else if Lib::LAPACK {
				fwd.fill(0);

				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::ssytrf_rook_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dsytrf_rook_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::csytrf_rook_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zsytrf_rook_(
							&(b'L' as _),
							(&n) as *const _ as *const _,
							L.as_ptr_mut() as _,
							(&L.col_stride()) as *const _ as *const _,
							fwd.as_mut_ptr() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			}
		})
	}
}

fn lblt_rook_diag<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	#[cfg(eigen)]
	if Lib::EIGEN {
		return bencher.skip();
	}

	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *avec![0usize; n];
	let bwd = &mut *avec![0usize; n];

	let params = LbltParams {
		pivoting: linalg::cholesky::lblt::factor::PivotingStrategy::Rook,
		..Auto::<T>::auto()
	}
	.into();

	let stack = &mut MemBuffer::new(linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, T>(n, parallel, params));
	let stack = MemStack::new(stack);

	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::lblt::factor::cholesky_in_place(L.rb_mut(), subdiag.rb_mut(), fwd, bwd, parallel, stack, params);
		})
	}
}
fn lblt_full<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	#[cfg(eigen)]
	if Lib::EIGEN {
		return bencher.skip();
	}

	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();
	let mut L = Mat::zeros(n, n);
	let mut subdiag = Diag::zeros(n);
	let fwd = &mut *avec![0usize; n];
	let bwd = &mut *avec![0usize; n];

	let params = LbltParams {
		pivoting: linalg::cholesky::lblt::factor::PivotingStrategy::Full,
		..Auto::<T>::auto()
	}
	.into();

	let stack = &mut MemBuffer::new(linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, T>(n, parallel, params));
	let stack = MemStack::new(stack);

	if Lib::FAER {
		bencher.bench(|| {
			L.copy_from_triangular_lower(&A);
			linalg::cholesky::lblt::factor::cholesky_in_place(L.rb_mut(), subdiag.rb_mut(), fwd, bwd, parallel, stack, params);
		})
	}
}

fn qr<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN)) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) || (!T::IS_NATIVE && Ord::max(m, n) > 1024)
		// parallel mkl sometimes segfaults here ¯\_(ツ)_/¯
		|| (Lib::LAPACK && Thd::PAR && cfg!(mkl))
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(m, n);

	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);

	#[cfg(eigen)]
	if Lib::EIGEN {
		return bench_eigen(bencher, eig::QR, A.rb());
	}

	let mut Q = Mat::zeros(blocksize, Ord::min(m, n));
	let mut QR = Mat::zeros(m, n);

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut lwork = core::mem::zeroed();
			la::sgeqrf_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_F64 {
			let mut lwork = core::mem::zeroed();
			la::dgeqrf_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_C32 {
			let mut lwork = core::mem::zeroed();
			la::cgeqrf_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut lwork = core::mem::zeroed();
			la::zgeqrf_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else {
			0
		}
	};

	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(
		m, n, blocksize, parallel, params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				QR.copy_from(&A);
				linalg::qr::no_pivoting::factor::qr_in_place(QR.rb_mut(), Q.rb_mut(), parallel, stack, params);
			} else if Lib::LAPACK {
				QR.copy_from(&A);
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::sgeqrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							QR.as_ptr_mut() as _,
							(&QR.col_stride()) as *const _ as *const _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgeqrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							QR.as_ptr_mut() as _,
							(&QR.col_stride()) as *const _ as *const _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgeqrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							QR.as_ptr_mut() as _,
							(&QR.col_stride()) as *const _ as *const _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgeqrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							QR.as_ptr_mut() as _,
							(&QR.col_stride()) as *const _ as *const _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.qr();
				};
			}
		});
	}
}

fn col_piv_qr<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	let blocksize = linalg::qr::col_pivoting::factor::recommended_blocksize::<T>(m, n);

	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);

	#[cfg(eigen)]
	if Lib::EIGEN {
		return bench_eigen(bencher, eig::CQR, A.rb());
	}

	let mut Q = Mat::zeros(blocksize, Ord::min(m, n));
	let mut QR = Mat::zeros(m, n);

	let col_fwd = &mut *avec![0usize; n];
	let col_bwd = &mut *avec![0usize; n];

	#[cfg(any(openblas, mkl, blis))]
	let rwork = &mut *avec![zero::<T>(); 2 * n];

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut lwork = core::mem::zeroed();
			la::sgeqp3_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				col_fwd.as_mut_ptr() as _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_F64 {
			let mut lwork = core::mem::zeroed();
			la::dgeqp3_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				col_fwd.as_mut_ptr() as _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork as usize
		} else if T::IS_NATIVE_C32 {
			let mut lwork = core::mem::zeroed();
			la::cgeqp3_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				col_fwd.as_mut_ptr() as _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut lwork = core::mem::zeroed();
			la::zgeqp3_(
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				QR.as_ptr_mut() as _,
				(&QR.col_stride()) as *const _ as *const _,
				col_fwd.as_mut_ptr() as _,
				Q.as_ptr_mut() as _,
				&mut lwork,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			lwork.re as usize
		} else {
			0
		}
	};

	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::qr::col_pivoting::factor::qr_in_place_scratch::<usize, T>(
		m, n, blocksize, parallel, params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				QR.copy_from(&A);
				linalg::qr::col_pivoting::factor::qr_in_place(QR.rb_mut(), Q.rb_mut(), col_fwd, col_bwd, parallel, stack, params);
			} else if Lib::LAPACK {
				QR.copy_from(&A);
				col_fwd.fill(0);
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::sgeqp3_(
							(&mut { m } as *mut _) as *mut _,
							(&mut { n } as *mut _) as *mut _,
							QR.as_ptr_mut() as _,
							(&mut { QR.col_stride() } as *mut _) as *mut _,
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgeqp3_(
							(&mut { m } as *mut _) as *mut _,
							(&mut { n } as *mut _) as *mut _,
							QR.as_ptr_mut() as _,
							(&mut { QR.col_stride() } as *mut _) as *mut _,
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgeqp3_(
							(&mut { m } as *mut _) as *mut _,
							(&mut { n } as *mut _) as *mut _,
							QR.as_ptr_mut() as _,
							(&mut { QR.col_stride() } as *mut _) as *mut _,
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgeqp3_(
							(&mut { m } as *mut _) as *mut _,
							(&mut { n } as *mut _) as *mut _,
							QR.as_ptr_mut() as _,
							(&mut { QR.col_stride() } as *mut _) as *mut _,
							col_fwd.as_mut_ptr() as _,
							Q.as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.col_piv_qr();
				};
			}
		});
	}
}

fn partial_piv_lu<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	#[cfg(eigen)]
	if Lib::EIGEN {
		return bench_eigen(bencher, eig::PLU, A.rb());
	}

	let mut LU = Mat::zeros(n, n);
	let row_fwd = &mut *avec![0usize; n];
	let row_bwd = &mut *avec![0usize; n];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T>(
		n, n, parallel, params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				LU.copy_from(&A);
				linalg::lu::partial_pivoting::factor::lu_in_place(LU.rb_mut(), row_fwd, row_bwd, parallel, stack, params);
			} else if Lib::LAPACK {
				LU.copy_from(&A);
				row_fwd.fill(0);

				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						la::sgetrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgetrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgetrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgetrf_(
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.lu();
				};
			}
		})
	}
}

fn full_piv_lu<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	#[cfg(eigen)]
	if Lib::EIGEN {
		return bench_eigen(bencher, eig::FLU, A.rb());
	}

	let mut LU = Mat::zeros(m, n);
	let row_fwd = &mut *avec![0usize; m];
	let row_bwd = &mut *avec![0usize; m];
	let col_fwd = &mut *avec![0usize; n];
	let col_bwd = &mut *avec![0usize; n];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, T>(m, n, parallel, params));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				LU.copy_from(&A);
				linalg::lu::full_pivoting::factor::lu_in_place(LU.rb_mut(), row_fwd, row_bwd, col_fwd, col_bwd, parallel, stack, params);
			} else if Lib::LAPACK {
				LU.copy_from(&A);
				row_fwd.fill(0);
				col_fwd.fill(0);

				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					if T::IS_NATIVE_F32 {
						sgetc2_(
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							col_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						dgetc2_(
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							col_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						cgetc2_(
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							col_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						zgetc2_(
							(&n) as *const _ as *const _,
							LU.as_ptr_mut() as _,
							(&LU.col_stride()) as *const _ as *const _,
							row_fwd.as_mut_ptr() as _,
							col_fwd.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.full_piv_lu();
				};
			}
		})
	}
}

fn svd<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	#[cfg(eigen)]
	if Lib::EIGEN {
		return bench_eigen(bencher, eig::SVD, A.rb());
	}

	let mut U = Mat::zeros(m, m);
	let mut V = Mat::zeros(n, n);
	let mut S = Diag::zeros(Ord::min(m, n));
	#[cfg(any(openblas, mkl, blis))]
	let mut clone = A.cloned();

	#[cfg(any(openblas, mkl, blis))]
	let rwork = &mut *avec![zero::<T>(); m * n * 10];
	#[cfg(any(openblas, mkl, blis))]
	let iwork = &mut *avec![0usize; Ord::min(m, n) * 8];

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut work = core::mem::zeroed();
			la::sgesdd_(
				&(b'A' as _),
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				V.as_ptr_mut() as _,
				(&V.col_stride()) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				iwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work as usize
		} else if T::IS_NATIVE_F64 {
			let mut work = core::mem::zeroed();
			la::dgesdd_(
				&(b'A' as _),
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				V.as_ptr_mut() as _,
				(&V.col_stride()) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				iwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work as usize
		} else if T::IS_NATIVE_C32 {
			let mut work = core::mem::zeroed();
			la::cgesdd_(
				&(b'A' as _),
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				V.as_ptr_mut() as _,
				(&V.col_stride()) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				iwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut work = core::mem::zeroed();
			la::zgesdd_(
				&(b'A' as _),
				(&m) as *const _ as *const _,
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				V.as_ptr_mut() as _,
				(&V.col_stride()) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				iwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work.re as usize
		} else {
			0
		}
	};
	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::svd::svd_scratch::<T>(
		m,
		n,
		linalg::svd::ComputeSvdVectors::Full,
		linalg::svd::ComputeSvdVectors::Full,
		parallel,
		params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				linalg::svd::svd(A.rb(), S.rb_mut(), Some(U.rb_mut()), Some(V.rb_mut()), parallel, stack, params).unwrap();
			} else if Lib::LAPACK {
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					clone.copy_from(&A);
					if T::IS_NATIVE_F32 {
						la::sgesdd_(
							&(b'A' as _),
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							V.as_ptr_mut() as _,
							(&V.col_stride()) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgesdd_(
							&(b'A' as _),
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							V.as_ptr_mut() as _,
							(&V.col_stride()) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgesdd_(
							&(b'A' as _),
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							V.as_ptr_mut() as _,
							(&V.col_stride()) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							iwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgesdd_(
							&(b'A' as _),
							(&m) as *const _ as *const _,
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							V.as_ptr_mut() as _,
							(&V.col_stride()) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							iwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				};
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.svd(true, true);
				};
			}
		});
	}
}

fn self_adjoint_evd<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN)) || (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis)))) || (!T::IS_NATIVE && Ord::max(m, n) > 1024)
		// parallel mkl sometimes segfaults here ¯\_(ツ)_/¯
		|| (Lib::LAPACK && Thd::PAR && cfg!(mkl))
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);

	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	let A = &A + A.adjoint();

	#[cfg(eigen)]
	if Lib::EIGEN {
		return bench_eigen(bencher, eig::HEVD, A.rb());
	}

	let mut U = Mat::zeros(m, m);
	let mut S = Diag::zeros(Ord::min(m, n));
	#[cfg(any(openblas, mkl, blis))]
	let mut clone = A.cloned();

	#[cfg(any(openblas, mkl, blis))]
	let (lwork, lrwork, liwork) = unsafe {
		clone.copy_from(&A);
		if T::IS_NATIVE_F32 {
			let mut work = core::mem::zeroed();
			let mut iwork = 0usize;
			la::ssyevd_(
				&(b'V' as _),
				&(b'L' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				(&mut iwork) as *mut _ as *mut _,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			(work as usize, 0, iwork as usize)
		} else if T::IS_NATIVE_F64 {
			let mut work = core::mem::zeroed();
			let mut iwork = 0usize;
			la::dsyevd_(
				&(b'V' as _),
				&(b'L' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				(&mut iwork) as *mut _ as *mut _,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			(work as usize, 0, iwork as usize)
		} else if T::IS_NATIVE_C32 {
			let mut work = core::mem::zeroed();
			let mut rwork = core::mem::zeroed();
			let mut iwork = 0usize;
			la::cheevd_(
				&(b'V' as _),
				&(b'L' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				&mut rwork,
				(&-1isize) as *const _ as *const _,
				(&mut iwork) as *mut _ as *mut _,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			(work.re as usize, 2 * rwork as usize, iwork as usize)
		} else if T::IS_NATIVE_C64 {
			let mut work = core::mem::zeroed();
			let mut rwork = core::mem::zeroed();
			let mut iwork = 0usize;
			la::zheevd_(
				&(b'V' as _),
				&(b'L' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				&mut rwork,
				(&-1isize) as *const _ as *const _,
				(&mut iwork) as *mut _ as *mut _,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			(work.re as usize, rwork as usize, iwork as usize)
		} else {
			(0, 0, 0)
		}
	};
	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];
	#[cfg(any(openblas, mkl, blis))]
	let rwork = &mut *avec![zero::<T>(); lrwork];
	#[cfg(any(openblas, mkl, blis))]
	let iwork = &mut *avec![0usize; liwork];

	let params = Default::default();
	let stack = &mut MemBuffer::new(linalg::evd::self_adjoint_evd_scratch::<T>(
		m,
		linalg::evd::ComputeEigenvectors::Yes,
		parallel,
		params,
	));
	let stack = MemStack::new(stack);

	if !Lib::LAPACK || T::IS_NATIVE {
		bencher.bench(|| {
			if Lib::FAER {
				linalg::evd::self_adjoint_evd(A.rb(), S.rb_mut(), Some(U.rb_mut()), parallel, stack, params).unwrap();
			} else if Lib::LAPACK {
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					clone.copy_from(&A);
					if T::IS_NATIVE_F32 {
						la::ssyevd_(
							&(b'V' as _),
							&(b'L' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&liwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dsyevd_(
							&(b'V' as _),
							&(b'L' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&liwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cheevd_(
							&(b'V' as _),
							&(b'L' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&lrwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&liwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zheevd_(
							&(b'V' as _),
							&(b'L' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&lrwork) as *const _ as *const _,
							iwork.as_mut_ptr() as _,
							(&liwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					}
				}
			} else if Lib::NALGEBRA {
				unsafe {
					na::DMatrixView::from_data(na::ViewStorage::from_raw_parts(
						A.as_ptr(),
						(na::Dyn(n), na::Dyn(n)),
						(na::Const::<1>, na::Dyn(A.col_stride() as usize)),
					))
					.clone_owned()
					.symmetric_eigen();
				};
			}
		});
	}
}

fn evd<T: Scalar, Lib: self::Lib, Thd: self::Thread>(bencher: Bencher, PlotArg(n): PlotArg) {
	let m = n;
	if (Ord::max(m, n) > 2048 && (Lib::NALGEBRA || Lib::EIGEN))
		|| (Lib::LAPACK && cfg!(not(any(openblas, mkl, blis))))
		|| (!T::IS_NATIVE && Ord::max(m, n) > 1024)
	{
		bencher.skip();
		return;
	}

	let parallel = if Thd::PAR { Par::rayon(0) } else { Par::Seq };
	lapack_set_num_threads(parallel);
	assert!(m == n);
	let rng = &mut StdRng::seed_from_u64(0);
	let A = T::random(rng, m, n);
	#[cfg(eigen)]
	if Lib::EIGEN {
		return bench_eigen(bencher, eig::EVD, A.rb());
	}

	let mut U = Mat::<T>::zeros(m, m);
	let mut S = Diag::<T>::zeros(Ord::min(m, n));
	let mut S_im = Diag::<T>::zeros(Ord::min(m, n));
	#[cfg(any(openblas, mkl, blis))]
	let mut clone = A.cloned();

	#[cfg(any(openblas, mkl, blis))]
	let rwork = &mut *avec![zero::<T>(); 2 * n];

	#[cfg(any(openblas, mkl, blis))]
	let lwork = unsafe {
		if T::IS_NATIVE_F32 {
			let mut work = core::mem::zeroed();
			la::sgeev_(
				&(b'V' as _),
				&(b'N' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				S_im.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				null_mut(),
				(&1usize) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work as usize
		} else if T::IS_NATIVE_F64 {
			let mut work = core::mem::zeroed();
			la::dgeev_(
				&(b'V' as _),
				&(b'N' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				S_im.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				null_mut(),
				(&1usize) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work as usize
		} else if T::IS_NATIVE_C32 {
			let mut work = core::mem::zeroed();
			la::cgeev_(
				&(b'V' as _),
				&(b'N' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				null_mut(),
				(&1usize) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work.re as usize
		} else if T::IS_NATIVE_C64 {
			let mut work = core::mem::zeroed();
			la::zgeev_(
				&(b'V' as _),
				&(b'N' as _),
				(&n) as *const _ as *const _,
				clone.as_ptr_mut() as _,
				(&clone.col_stride()) as *const _ as *const _,
				S.column_vector_mut().as_ptr_mut() as _,
				U.as_ptr_mut() as _,
				(&U.col_stride()) as *const _ as *const _,
				null_mut(),
				(&1usize) as *const _ as *const _,
				&mut work,
				(&-1isize) as *const _ as *const _,
				rwork.as_mut_ptr() as _,
				(&mut 0usize) as *mut _ as *mut _,
			);
			work.re as usize
		} else {
			0
		}
	};

	#[cfg(any(openblas, mkl, blis))]
	let work = &mut *avec![zero::<T>(); lwork];

	let params = <linalg::evd::EvdParams as Auto<T>>::auto();
	let stack = &mut MemBuffer::new(linalg::evd::evd_scratch::<T>(
		n,
		linalg::evd::ComputeEigenvectors::Yes,
		linalg::evd::ComputeEigenvectors::Yes,
		parallel,
		params.into(),
	));
	let stack = MemStack::new(stack);

	if Lib::FAER || (Lib::LAPACK && T::IS_NATIVE) {
		bencher.bench(|| {
			if Lib::FAER {
				use core::mem::transmute;
				// SAFETY: dont worry im a pro
				unsafe {
					if T::IS_REAL {
						linalg::evd::evd_real::<T::Real>(
							transmute(A.rb()),
							transmute(S.rb_mut()),
							transmute(S_im.rb_mut()),
							Some(transmute(U.rb_mut())),
							None,
							parallel,
							stack,
							params.into(),
						)
						.unwrap();
					} else {
						linalg::evd::evd_cplx::<T::Real>(
							transmute(A.rb()),
							transmute(S.rb_mut()),
							Some(transmute(U.rb_mut())),
							None,
							parallel,
							stack,
							params.into(),
						)
						.unwrap();
					}
				}
			} else if Lib::LAPACK {
				#[cfg(any(openblas, mkl, blis))]
				unsafe {
					clone.copy_from(&A);
					if T::IS_NATIVE_F32 {
						la::sgeev_(
							&(b'V' as _),
							&(b'N' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							S_im.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							null_mut(),
							(&1usize) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_F64 {
						la::dgeev_(
							&(b'V' as _),
							&(b'N' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							S_im.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							null_mut(),
							(&1usize) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C32 {
						la::cgeev_(
							&(b'V' as _),
							&(b'N' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							null_mut(),
							(&1usize) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					} else if T::IS_NATIVE_C64 {
						la::zgeev_(
							&(b'V' as _),
							&(b'N' as _),
							(&n) as *const _ as *const _,
							clone.as_ptr_mut() as _,
							(&clone.col_stride()) as *const _ as *const _,
							S.column_vector_mut().as_ptr_mut() as _,
							U.as_ptr_mut() as _,
							(&U.col_stride()) as *const _ as *const _,
							null_mut(),
							(&1usize) as *const _ as *const _,
							work.as_mut_ptr() as _,
							(&lwork) as *const _ as *const _,
							rwork.as_mut_ptr() as _,
							(&mut 0usize) as *mut _ as *mut _,
						);
					};
				}
			}
		});
	}
}

#[derive(Clone)]
pub struct FlopsMetric;
impl diol::traits::PlotMetric for FlopsMetric {
	fn name(&self) -> &'static str {
		"∝ flops"
	}

	fn compute(&self, arg: PlotArg, time: diol::Picoseconds) -> f64 {
		(arg.0 as f64).powi(3) / time.to_secs()
	}

	fn monotonicity(&self) -> diol::traits::Monotonicity {
		diol::traits::Monotonicity::HigherIsBetter
	}
}

fn main() -> eyre::Result<()> {
	let config = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bench.toml"))?
		.parse::<Table>()
		.unwrap();

	let mut parallel = vec![];

	if config["par"]["seq"].as_bool().unwrap() {
		parallel.push(Par::Seq);
	}
	if config["par"]["rayon"].as_bool().unwrap() {
		parallel.push(Par::rayon(0));
	}

	let shapes = |name: &str| {
		config[name]["shapes"]
			.as_array()
			.unwrap()
			.iter()
			.map(|i| PlotArg(i.as_integer().unwrap() as usize))
			.collect::<Vec<_>>()
	};
	let mut bench_config = Config::from_args()?;
	bench_config.plot_metric = PlotMetric::new(FlopsMetric);

	macro_rules! register {
		($T: ident) => {{
			type T = $T;

			for &parallel in &parallel {
				macro_rules! register_one {
					($title: expr, $name: ident, $config: expr) => {
						let par_name = &format!(
							"{} parallel ({} threads) {}",
							stringify!($T),
							rayon::current_num_threads(),
							$title
						);
						let seq_name = &format!("{}{}{}", stringify!($T), " sequential ", $title);

						let name = match parallel {
							Par::Seq => seq_name,
							Par::Rayon(_) => par_name,
						};
						if bench_config.group_filter.as_ref().is_none_or(|regex| regex.is_match(name)) {
							let timings_path = format!("{}{}/timings {name}.json", env!("CARGO_MANIFEST_DIR"), "/../target");
							let timings = serde_json::de::from_str::<BenchResult>(&*std::fs::read_to_string(&timings_path).unwrap_or(String::new()))
								.unwrap_or(BenchResult { groups: HashMap::new() });

							let bench = Bench::new(&bench_config);

							match parallel {
								Par::Seq => bench.register_many(
									seq_name,
									{
										let list = diol::variadics::Nil;
										#[cfg(any(openblas, mkl))]
										let list = diol::variadics::Cons {
											head: $name::<T, self::lapack, self::seq>
												.with_name(core::any::type_name::<self::lapack>().trim_start_matches("bench::")),
											tail: list,
										};
										#[cfg(nalgebra)]
										let list = diol::variadics::Cons {
											head: $name::<T, self::nalgebra, self::seq>
												.with_name(core::any::type_name::<self::nalgebra>().trim_start_matches("bench::")),
											tail: list,
										};
										#[cfg(eigen)]
										let list = diol::variadics::Cons {
											head: $name::<T, self::eigen, self::seq>
												.with_name(core::any::type_name::<self::eigen>().trim_start_matches("bench::")),
											tail: list,
										};
										#[cfg(faer)]
										let list = diol::variadics::Cons {
											head: $name::<T, self::faer, self::seq>
												.with_name(core::any::type_name::<self::faer>().trim_start_matches("bench::")),
											tail: list,
										};

										list
									},
									shapes($config),
								),

								Par::Rayon(_) => bench.register_many(
									par_name,
									{
										let list = diol::variadics::Nil;
										#[cfg(any(openblas, mkl, blis))]
										let list = diol::variadics::Cons {
											head: $name::<T, self::lapack, self::par>
												.with_name(core::any::type_name::<self::lapack>().trim_start_matches("bench::")),
											tail: list,
										};
										#[cfg(faer)]
										let list = diol::variadics::Cons {
											head: $name::<T, self::faer, self::par>
												.with_name(core::any::type_name::<self::faer>().trim_start_matches("bench::")),
											tail: list,
										};

										list
									},
									shapes($config),
								),
							}
							let timings = bench.run()?.combine(&timings);
							timings.plot(&bench_config, bench_config.plot_dir.0.as_ref().unwrap())?;
							std::fs::write(timings_path, serde_json::to_string(&timings).unwrap())?;
						}
					};
				}

				register_one!("LLT", llt, "block_decomp");
				register_one!("LDLT", ldlt, "block_decomp");
				register_one!("LBLT", lblt, "block_decomp");
				register_one!("LBLT diagonal pivoting", lblt_diag, "block_decomp");
				register_one!("LBLT rook pivoting", lblt_rook, "block_decomp");
				register_one!("LBLT diagonal + rook pivoting", lblt_rook_diag, "block_decomp");
				register_one!("LBLT full pivoting", lblt_full, "decomp");

				register_one!("LU partial pivoting", partial_piv_lu, "block_decomp");
				register_one!("LU full pivoting", full_piv_lu, "decomp");

				register_one!("QR", qr, "block_decomp");
				register_one!("QR column pivoting", col_piv_qr, "decomp");

				register_one!("SVD", svd, "svd");
				register_one!("self adjoint EVD", self_adjoint_evd, "svd");
				register_one!("EVD", evd, "evd");
			}
		}};
	}

	spindle::with_lock(rayon::current_num_threads(), || -> eyre::Result<()> {
		register!(f32);
		register!(f64);
		register!(fx128);
		register!(c32);
		register!(c64);
		register!(cx128);
		Ok(())
	})?;

	Ok(())
}
