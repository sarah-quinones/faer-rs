#![allow(clippy::type_complexity)]
#![cfg_attr(not(feature = "std"), no_std)]

use bytemuck::Pod;
use core::{fmt::Debug, marker::PhantomData, mem::ManuallyDrop};
use num_complex::Complex;
use pulp::Simd;
use reborrow::*;

fn sqrt_impl<E: RealField>(re: E, im: E) -> (E, E) {
    let im_sign = if im >= E::faer_zero() {
        E::faer_one()
    } else {
        E::faer_one().faer_neg()
    };
    let half = E::faer_from_f64(0.5);

    let abs = (re.faer_abs2().faer_add(im.faer_abs2())).faer_sqrt();
    let sum = re.faer_add(abs);
    // to avoid the sum being negative with inexact floating point arithmetic (i.e., double-double)
    let sum = if sum > E::faer_zero() {
        sum
    } else {
        E::faer_zero()
    };
    let a = (sum.faer_scale_power_of_two(half)).faer_sqrt();
    let b = ((re.faer_neg().faer_add(abs)).faer_scale_power_of_two(half))
        .faer_sqrt()
        .faer_scale_power_of_two(im_sign);

    (a, b)
}

#[inline(always)]
pub fn slice_as_simd<E: ComplexField, S: Simd>(
    slice: GroupFor<E, &[UnitFor<E>]>,
) -> (
    GroupFor<E, &[SimdUnitFor<E, S>]>,
    GroupFor<E, &[UnitFor<E>]>,
) {
    let (a_head, a_tail) = E::faer_unzip(E::faer_map(
        slice,
        #[inline(always)]
        |slice| E::faer_slice_as_simd::<S>(slice),
    ));
    (a_head, a_tail)
}

#[inline(always)]
pub fn slice_as_mut_simd<E: ComplexField, S: Simd>(
    slice: GroupFor<E, &mut [UnitFor<E>]>,
) -> (
    GroupFor<E, &mut [SimdUnitFor<E, S>]>,
    GroupFor<E, &mut [UnitFor<E>]>,
) {
    let (a_head, a_tail) = E::faer_unzip(E::faer_map(
        slice,
        #[inline(always)]
        |slice| E::faer_slice_as_mut_simd::<S>(slice),
    ));
    (a_head, a_tail)
}

#[inline(always)]
pub fn simd_as_slice_unit<E: ComplexField, S: Simd>(values: &[SimdUnitFor<E, S>]) -> &[UnitFor<E>] {
    unsafe {
        core::slice::from_raw_parts(
            values.as_ptr() as *const UnitFor<E>,
            values.len()
                * (core::mem::size_of::<SimdUnitFor<E, S>>() / core::mem::size_of::<UnitFor<E>>()),
        )
    }
}

#[inline(always)]
pub fn simd_as_slice<E: ComplexField, S: Simd>(
    values: GroupFor<E, &[SimdUnitFor<E, S>]>,
) -> GroupFor<E, &[UnitFor<E>]> {
    E::faer_map(
        values,
        #[inline(always)]
        |values| simd_as_slice_unit::<E, S>(values),
    )
}

#[inline(always)]
pub fn one_simd_as_slice<E: ComplexField, S: Simd>(
    values: GroupFor<E, &SimdUnitFor<E, S>>,
) -> GroupFor<E, &[UnitFor<E>]> {
    E::faer_map(
        values,
        #[inline(always)]
        |values| simd_as_slice_unit::<E, S>(core::slice::from_ref(values)),
    )
}

#[inline(always)]
pub fn simd_index_as_slice<E: RealField, S: Simd>(
    indices: &[SimdIndexFor<E, S>],
) -> &[IndexFor<E>] {
    unsafe {
        core::slice::from_raw_parts(
            indices.as_ptr() as *const IndexFor<E>,
            indices.len()
                * (core::mem::size_of::<SimdIndexFor<E, S>>()
                    / core::mem::size_of::<IndexFor<E>>()),
        )
    }
}

#[inline(always)]
#[doc(hidden)]
pub unsafe fn transmute_unchecked<From, To>(t: From) -> To {
    assert!(core::mem::size_of::<From>() == core::mem::size_of::<To>());
    assert!(core::mem::align_of::<From>() == core::mem::align_of::<To>());
    core::mem::transmute_copy(&ManuallyDrop::new(t))
}

pub trait ForType {
    type FaerOf<T>;
}
pub trait ForCopyType: ForType {
    type FaerOfCopy<T: Copy>: Copy;
}

pub struct IdentityGroup {
    __private: (),
}
impl ForCopyType for IdentityGroup {
    type FaerOfCopy<T: Copy> = T;
}
impl ForType for IdentityGroup {
    type FaerOf<T> = T;
}

pub struct ComplexGroup<Group> {
    __private: PhantomData<Group>,
}
pub struct ComplexConjGroup<Group> {
    __private: PhantomData<Group>,
}
impl<Group: ForCopyType> ForCopyType for ComplexConjGroup<Group> {
    type FaerOfCopy<T: Copy> = ComplexConj<Group::FaerOfCopy<T>>;
}
impl<Group: ForType> ForType for ComplexConjGroup<Group> {
    type FaerOf<T> = ComplexConj<Group::FaerOf<T>>;
}
impl<Group: ForCopyType> ForCopyType for ComplexGroup<Group> {
    type FaerOfCopy<T: Copy> = Complex<Group::FaerOfCopy<T>>;
}
impl<Group: ForType> ForType for ComplexGroup<Group> {
    type FaerOf<T> = Complex<Group::FaerOf<T>>;
}

pub type GroupFor<E, T> = <<E as Entity>::Group as ForType>::FaerOf<T>;
pub type GroupCopyFor<E, T> = <<E as Entity>::GroupCopy as ForCopyType>::FaerOfCopy<T>;
pub type UnitFor<E> = <E as Entity>::Unit;
pub type IndexFor<E> = <E as Entity>::Index;

pub type SimdUnitFor<E, S> = <E as Entity>::SimdUnit<S>;
pub type SimdMaskFor<E, S> = <E as Entity>::SimdMask<S>;
pub type SimdIndexFor<E, S> = <E as Entity>::SimdIndex<S>;

pub type SimdGroupFor<E, S> = GroupFor<E, SimdUnitFor<E, S>>;

#[inline(always)]
pub fn into_copy<E: Entity, T: Copy>(x: GroupFor<E, T>) -> GroupCopyFor<E, T> {
    unsafe { transmute_unchecked(x) }
}
#[inline(always)]
pub fn from_copy<E: Entity, T: Copy>(x: GroupCopyFor<E, T>) -> GroupFor<E, T> {
    unsafe { transmute_unchecked(x) }
}

/// Unstable core trait for describing how a scalar value may be split up into individual
/// component.
///
/// For example, `f64` is treated as a single indivisible unit, but [`num_complex::Complex<f64>`]
/// is split up into its real and imaginary components, with each one being stored in a separate
/// container.
///
/// # Safety
/// The associated types and functions must fulfill their respective contracts.
pub unsafe trait Entity: Copy + Pod + PartialEq + Send + Sync + Debug + 'static {
    type Group: ForType;
    type GroupCopy: ForCopyType;

    type Unit: Copy + Pod + PartialEq + Send + Sync + Debug + 'static;
    type Index: Copy + Pod + Send + Sync + Debug + 'static;
    type SimdUnit<S: Simd>: Copy + Pod + Send + Sync + Debug + 'static;
    type SimdMask<S: Simd>: Copy + Send + Sync + Debug + 'static;
    type SimdIndex<S: Simd>: Copy + Send + Sync + Debug + 'static;
    type Iter<I: Iterator>: Iterator<Item = GroupFor<Self, I::Item>>;

    const N_COMPONENTS: usize;
    const UNIT: GroupFor<Self, ()>;

    fn faer_from_units(group: GroupFor<Self, UnitFor<Self>>) -> Self;
    fn faer_into_units(self) -> GroupFor<Self, UnitFor<Self>>;

    fn faer_as_ref<T>(group: &GroupFor<Self, T>) -> GroupFor<Self, &T>;
    fn faer_as_mut<T>(group: &mut GroupFor<Self, T>) -> GroupFor<Self, &mut T>;
    fn faer_map<T, U>(group: GroupFor<Self, T>, f: impl FnMut(T) -> U) -> GroupFor<Self, U>;
    fn faer_zip<T, U>(
        first: GroupFor<Self, T>,
        second: GroupFor<Self, U>,
    ) -> GroupFor<Self, (T, U)>;
    fn faer_unzip<T, U>(zipped: GroupFor<Self, (T, U)>) -> (GroupFor<Self, T>, GroupFor<Self, U>);

    #[inline(always)]
    fn faer_unzip2<T>(zipped: GroupFor<Self, [T; 2]>) -> [GroupFor<Self, T>; 2] {
        let (a, b) = Self::faer_unzip(Self::faer_map(
            zipped,
            #[inline(always)]
            |[a, b]| (a, b),
        ));
        [a, b]
    }

    #[inline(always)]
    fn faer_unzip4<T>(zipped: GroupFor<Self, [T; 4]>) -> [GroupFor<Self, T>; 4] {
        let (ab, cd) = Self::faer_unzip(Self::faer_map(
            zipped,
            #[inline(always)]
            |[a, b, c, d]| ([a, b], [c, d]),
        ));
        let [a, b] = Self::faer_unzip2(ab);
        let [c, d] = Self::faer_unzip2(cd);
        [a, b, c, d]
    }

    #[inline(always)]
    fn faer_unzip8<T>(zipped: GroupFor<Self, [T; 8]>) -> [GroupFor<Self, T>; 8] {
        let (abcd, efgh) = Self::faer_unzip(Self::faer_map(
            zipped,
            #[inline(always)]
            |[a, b, c, d, e, f, g, h]| ([a, b, c, d], [e, f, g, h]),
        ));
        let [a, b, c, d] = Self::faer_unzip4(abcd);
        let [e, f, g, h] = Self::faer_unzip4(efgh);
        [a, b, c, d, e, f, g, h]
    }

    #[inline(always)]
    fn faer_as_arrays<const N: usize, T>(
        group: GroupFor<Self, &[T]>,
    ) -> (GroupFor<Self, &[[T; N]]>, GroupFor<Self, &[T]>) {
        #[inline(always)]
        fn do_as_arrays<const N: usize, T>() -> impl Fn(&[T]) -> (&[[T; N]], &[T]) {
            #[inline(always)]
            |slice| pulp::as_arrays(slice)
        }
        Self::faer_unzip(Self::faer_map(group, do_as_arrays()))
    }

    #[inline(always)]
    fn faer_as_arrays_mut<const N: usize, T>(
        group: GroupFor<Self, &mut [T]>,
    ) -> (GroupFor<Self, &mut [[T; N]]>, GroupFor<Self, &mut [T]>) {
        #[inline(always)]
        fn do_as_arrays_mut<const N: usize, T>() -> impl Fn(&mut [T]) -> (&mut [[T; N]], &mut [T]) {
            #[inline(always)]
            |slice| pulp::as_arrays_mut(slice)
        }
        Self::faer_unzip(Self::faer_map(group, do_as_arrays_mut()))
    }

    #[inline(always)]
    fn faer_deref<T: Copy>(group: GroupFor<Self, &T>) -> GroupFor<Self, T> {
        #[inline(always)]
        fn do_deref<T: Copy>() -> impl FnMut(&T) -> T {
            #[inline(always)]
            |group| *group
        }
        Self::faer_map(group, do_deref())
    }
    #[inline(always)]
    fn faer_rb<'short, T: Reborrow<'short>>(
        value: GroupFor<Self, &'short T>,
    ) -> GroupFor<Self, T::Target> {
        Self::faer_map(
            value,
            #[inline(always)]
            |value| value.rb(),
        )
    }

    #[inline(always)]
    fn faer_rb_mut<'short, T: ReborrowMut<'short>>(
        value: GroupFor<Self, &'short mut T>,
    ) -> GroupFor<Self, T::Target> {
        Self::faer_map(
            value,
            #[inline(always)]
            |value| value.rb_mut(),
        )
    }
    #[inline(always)]
    fn faer_into_const<T: IntoConst>(value: GroupFor<Self, T>) -> GroupFor<Self, T::Target> {
        Self::faer_map(
            value,
            #[inline(always)]
            |value| value.into_const(),
        )
    }

    fn faer_map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: GroupFor<Self, T>,
        f: &mut impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, GroupFor<Self, U>);

    #[inline(always)]
    fn faer_copy<T: Copy>(x: &GroupFor<Self, T>) -> GroupFor<Self, T> {
        unsafe { core::mem::transmute_copy(x) }
    }

    fn faer_into_iter<I: IntoIterator>(iter: GroupFor<Self, I>) -> Self::Iter<I::IntoIter>;
}

/// Trait for types that may be implicitly conjugated.
///
/// # Safety
/// The associated types and functions must fulfill their respective contracts.
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

pub trait SimdCtx: core::fmt::Debug + Copy + Send + Sync + 'static + Default {
    fn dispatch<Op: pulp::WithSimd>(self, f: Op) -> Op::Output;
}

#[derive(Default, Clone, Copy, Debug)]
pub struct NoSimd;

impl SimdCtx for pulp::Arch {
    #[inline(always)]
    fn dispatch<Op: pulp::WithSimd>(self, f: Op) -> Op::Output {
        self.dispatch(f)
    }
}

impl SimdCtx for NoSimd {
    #[inline(always)]
    fn dispatch<Op: pulp::WithSimd>(self, f: Op) -> Op::Output {
        f.with_simd(pulp::Scalar::new())
    }
}

/// Unstable trait containing the operations that a number type needs to implement.
pub trait ComplexField: Entity + Conjugate<Canonical = Self> {
    type Real: RealField;
    type Simd: SimdCtx;
    type ScalarSimd: SimdCtx;

    /// Converts `value` from `f64` to `Self`.  
    /// The conversion may be lossy when converting to a type with less precision.
    fn faer_from_f64(value: f64) -> Self;

    /// Returns `self + rhs`.
    fn faer_add(self, rhs: Self) -> Self;
    /// Returns `self - rhs`.
    fn faer_sub(self, rhs: Self) -> Self;
    /// Returns `self * rhs`.
    fn faer_mul(self, rhs: Self) -> Self;

    /// Returns `-self`.
    fn faer_neg(self) -> Self;
    /// Returns `1.0/self`.
    fn faer_inv(self) -> Self;
    /// Returns `conjugate(self)`.
    fn faer_conj(self) -> Self;
    /// Returns the square root of `self`.
    fn faer_sqrt(self) -> Self;

    /// Returns the input, scaled by `rhs`.
    fn faer_scale_real(self, rhs: Self::Real) -> Self;

    /// Returns the input, scaled by `rhs`.
    fn faer_scale_power_of_two(self, rhs: Self::Real) -> Self;

    /// Returns either the norm or squared norm of the number.
    ///
    /// An implementation may choose either, so long as it chooses consistently.
    fn faer_score(self) -> Self::Real;
    /// Returns the absolute value of `self`.
    fn faer_abs(self) -> Self::Real;
    /// Returns the squared absolute value of `self`.
    fn faer_abs2(self) -> Self::Real;

    /// Returns a NaN value.
    fn faer_nan() -> Self;

    /// Returns true if `self` is a NaN value, or false otherwise.
    #[inline(always)]
    fn faer_is_nan(&self) -> bool {
        #[allow(clippy::eq_op)]
        {
            self != self
        }
    }

    /// Returns true if `self` is a NaN value, or false otherwise.
    #[inline(always)]
    fn faer_is_finite(&self) -> bool {
        let inf = Self::Real::faer_zero().faer_inv();
        if coe::is_same::<Self, Self::Real>() {
            self.faer_real().faer_abs() < inf
        } else {
            (self.faer_real().faer_abs() < inf) & (self.faer_imag().faer_abs() < inf)
        }
    }

    /// Returns a complex number whose real part is equal to `real`, and a zero imaginary part.
    fn faer_from_real(real: Self::Real) -> Self;

    /// Returns the real part.
    fn faer_real(self) -> Self::Real;
    /// Returns the imaginary part.
    fn faer_imag(self) -> Self::Real;

    /// Returns `0.0`.
    fn faer_zero() -> Self;
    /// Returns `1.0`.
    fn faer_one() -> Self;

    fn faer_slice_as_simd<S: Simd>(
        slice: &[UnitFor<Self>],
    ) -> (&[SimdUnitFor<Self, S>], &[UnitFor<Self>]);
    fn faer_slice_as_mut_simd<S: Simd>(
        slice: &mut [UnitFor<Self>],
    ) -> (&mut [SimdUnitFor<Self, S>], &mut [UnitFor<Self>]);

    fn faer_partial_load_unit<S: Simd>(simd: S, slice: &[UnitFor<Self>]) -> SimdUnitFor<Self, S>;
    fn faer_partial_store_unit<S: Simd>(
        simd: S,
        slice: &mut [UnitFor<Self>],
        values: SimdUnitFor<Self, S>,
    );
    fn faer_partial_load_last_unit<S: Simd>(
        simd: S,
        slice: &[UnitFor<Self>],
    ) -> SimdUnitFor<Self, S>;
    fn faer_partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [UnitFor<Self>],
        values: SimdUnitFor<Self, S>,
    );

    fn faer_simd_splat_unit<S: Simd>(simd: S, unit: UnitFor<Self>) -> SimdUnitFor<Self, S>;

    #[inline(always)]
    fn faer_partial_load<S: Simd>(
        simd: S,
        slice: GroupFor<Self, &[UnitFor<Self>]>,
    ) -> SimdGroupFor<Self, S> {
        Self::faer_map(
            slice,
            #[inline(always)]
            |slice| Self::faer_partial_load_unit(simd, slice),
        )
    }
    #[inline(always)]
    fn faer_partial_store<S: Simd>(
        simd: S,
        slice: GroupFor<Self, &mut [UnitFor<Self>]>,
        values: SimdGroupFor<Self, S>,
    ) {
        Self::faer_map(
            Self::faer_zip(slice, values),
            #[inline(always)]
            |(slice, unit)| Self::faer_partial_store_unit(simd, slice, unit),
        );
    }
    #[inline(always)]
    fn faer_partial_load_last<S: Simd>(
        simd: S,
        slice: GroupFor<Self, &[UnitFor<Self>]>,
    ) -> SimdGroupFor<Self, S> {
        Self::faer_map(
            slice,
            #[inline(always)]
            |slice| Self::faer_partial_load_last_unit(simd, slice),
        )
    }
    #[inline(always)]
    fn faer_partial_store_last<S: Simd>(
        simd: S,
        slice: GroupFor<Self, &mut [UnitFor<Self>]>,
        values: SimdGroupFor<Self, S>,
    ) {
        Self::faer_map(
            Self::faer_zip(slice, values),
            #[inline(always)]
            |(slice, unit)| Self::faer_partial_store_last_unit(simd, slice, unit),
        );
    }
    #[inline(always)]
    fn faer_simd_splat<S: Simd>(simd: S, value: Self) -> SimdGroupFor<Self, S> {
        Self::faer_map(
            Self::faer_into_units(value),
            #[inline(always)]
            |unit| Self::faer_simd_splat_unit(simd, unit),
        )
    }

    fn faer_simd_scalar_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self;
    fn faer_simd_scalar_conj_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self;
    fn faer_simd_scalar_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self;
    fn faer_simd_scalar_conj_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self;

    fn faer_simd_neg<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S>;
    fn faer_simd_conj<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S>;

    fn faer_simd_add<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S>;
    fn faer_simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S>;

    fn faer_simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S>;
    fn faer_simd_scale_real<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self::Real, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S>;
    fn faer_simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S>;
    fn faer_simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S>;
    fn faer_simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S>;

    fn faer_simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self::Real, S>,
    ) -> SimdGroupFor<Self::Real, S>;
    fn faer_simd_abs2<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S>;
    fn faer_simd_score<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S>;

    #[inline(always)]
    fn faer_simd_reduce_add<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> Self {
        let _ = simd;
        let mut acc = Self::faer_zero();

        let slice = simd_as_slice::<Self, S>(Self::faer_map(
            Self::faer_as_ref(&values),
            #[allow(clippy::redundant_closure)]
            #[inline(always)]
            |ptr| core::slice::from_ref(ptr),
        ));
        for units in Self::faer_into_iter(slice) {
            let value = Self::faer_from_units(Self::faer_deref(units));
            acc = acc.faer_add(value);
        }

        acc
    }
}

/// Unstable trait containing the operations that a real number type needs to implement.
pub trait RealField: ComplexField<Real = Self> + PartialOrd {
    fn faer_epsilon() -> Option<Self>;
    fn faer_zero_threshold() -> Option<Self>;

    fn faer_div(self, rhs: Self) -> Self;

    fn faer_usize_to_index(a: usize) -> IndexFor<Self>;
    fn faer_index_to_usize(a: IndexFor<Self>) -> usize;
    fn faer_max_index() -> IndexFor<Self>;

    fn faer_simd_less_than<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S>;
    fn faer_simd_less_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S>;
    fn faer_simd_greater_than<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S>;
    fn faer_simd_greater_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S>;

    fn faer_simd_select<S: Simd>(
        simd: S,
        mask: SimdMaskFor<Self, S>,
        if_true: SimdGroupFor<Self, S>,
        if_false: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S>;
    fn faer_simd_index_select<S: Simd>(
        simd: S,
        mask: SimdMaskFor<Self, S>,
        if_true: SimdIndexFor<Self, S>,
        if_false: SimdIndexFor<Self, S>,
    ) -> SimdIndexFor<Self, S>;
    fn faer_simd_index_seq<S: Simd>(simd: S) -> SimdIndexFor<Self, S>;
    fn faer_simd_index_splat<S: Simd>(simd: S, value: IndexFor<Self>) -> SimdIndexFor<Self, S>;
    fn faer_simd_index_add<S: Simd>(
        simd: S,
        a: SimdIndexFor<Self, S>,
        b: SimdIndexFor<Self, S>,
    ) -> SimdIndexFor<Self, S>;
}

impl ComplexField for f32 {
    type Real = Self;
    type Simd = pulp::Arch;
    type ScalarSimd = NoSimd;

    #[inline(always)]
    fn faer_from_f64(value: f64) -> Self {
        value as _
    }

    #[inline(always)]
    fn faer_add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn faer_sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn faer_mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn faer_neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn faer_inv(self) -> Self {
        self.recip()
    }

    #[inline(always)]
    fn faer_conj(self) -> Self {
        self
    }

    #[inline(always)]
    fn faer_sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline(always)]
    fn faer_scale_real(self, rhs: Self::Real) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn faer_scale_power_of_two(self, rhs: Self::Real) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn faer_score(self) -> Self::Real {
        self.abs()
    }

    #[inline(always)]
    fn faer_abs(self) -> Self::Real {
        self.abs()
    }

    #[inline(always)]
    fn faer_abs2(self) -> Self::Real {
        self * self
    }

    #[inline(always)]
    fn faer_nan() -> Self {
        Self::NAN
    }

    #[inline(always)]
    fn faer_from_real(real: Self::Real) -> Self {
        real
    }

    #[inline(always)]
    fn faer_real(self) -> Self::Real {
        self
    }

    #[inline(always)]
    fn faer_imag(self) -> Self::Real {
        0.0
    }

    #[inline(always)]
    fn faer_zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn faer_one() -> Self {
        1.0
    }

    #[inline(always)]
    fn faer_slice_as_simd<S: Simd>(
        slice: &[UnitFor<Self>],
    ) -> (&[SimdUnitFor<Self, S>], &[UnitFor<Self>]) {
        S::f32s_as_simd(slice)
    }

    #[inline(always)]
    fn faer_slice_as_mut_simd<S: Simd>(
        slice: &mut [UnitFor<Self>],
    ) -> (&mut [SimdUnitFor<Self, S>], &mut [UnitFor<Self>]) {
        S::f32s_as_mut_simd(slice)
    }

    #[inline(always)]
    fn faer_partial_load_last_unit<S: Simd>(
        simd: S,
        slice: &[UnitFor<Self>],
    ) -> SimdUnitFor<Self, S> {
        simd.f32s_partial_load_last(slice)
    }

    #[inline(always)]
    fn faer_partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [UnitFor<Self>],
        values: SimdUnitFor<Self, S>,
    ) {
        simd.f32s_partial_store_last(slice, values)
    }

    #[inline(always)]
    fn faer_partial_load_unit<S: Simd>(simd: S, slice: &[UnitFor<Self>]) -> SimdUnitFor<Self, S> {
        simd.f32s_partial_load(slice)
    }

    #[inline(always)]
    fn faer_partial_store_unit<S: Simd>(
        simd: S,
        slice: &mut [UnitFor<Self>],
        values: SimdUnitFor<Self, S>,
    ) {
        simd.f32s_partial_store(slice, values)
    }

    #[inline(always)]
    fn faer_simd_splat_unit<S: Simd>(simd: S, unit: UnitFor<Self>) -> SimdUnitFor<Self, S> {
        simd.f32s_splat(unit)
    }

    #[inline(always)]
    fn faer_simd_neg<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        simd.f32s_neg(values)
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
        simd.f32s_add(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.f32s_sub(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.f32s_mul(lhs, rhs)
    }
    #[inline(always)]
    fn faer_simd_scale_real<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self::Real, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        Self::faer_simd_mul(simd, lhs, rhs)
    }
    #[inline(always)]
    fn faer_simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.f32s_mul(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.f32s_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn faer_simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.f32s_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn faer_simd_reduce_add<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> Self {
        simd.f32s_reduce_sum(values)
    }

    #[inline(always)]
    fn faer_simd_abs2<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        simd.f32s_mul(values, values)
    }
    #[inline(always)]
    fn faer_simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self::Real, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        simd.f32s_mul_add_e(values, values, acc)
    }
    #[inline(always)]
    fn faer_simd_score<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        simd.f32s_abs(values)
    }

    #[inline(always)]
    fn faer_simd_scalar_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        let _ = simd;
        lhs * rhs
    }
    #[inline(always)]
    fn faer_simd_scalar_conj_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        let _ = simd;
        lhs * rhs
    }
    #[inline(always)]
    fn faer_simd_scalar_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        simd.f32_scalar_mul_add_e(lhs, rhs, acc)
    }
    #[inline(always)]
    fn faer_simd_scalar_conj_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        simd.f32_scalar_mul_add_e(lhs, rhs, acc)
    }
}
impl ComplexField for f64 {
    type Real = Self;
    type Simd = pulp::Arch;
    type ScalarSimd = NoSimd;

    #[inline(always)]
    fn faer_from_f64(value: f64) -> Self {
        value
    }

    #[inline(always)]
    fn faer_add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn faer_sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn faer_mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn faer_neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn faer_inv(self) -> Self {
        self.recip()
    }

    #[inline(always)]
    fn faer_conj(self) -> Self {
        self
    }

    #[inline(always)]
    fn faer_sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline(always)]
    fn faer_scale_real(self, rhs: Self::Real) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn faer_scale_power_of_two(self, rhs: Self::Real) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn faer_score(self) -> Self::Real {
        self.abs()
    }

    #[inline(always)]
    fn faer_abs(self) -> Self::Real {
        self.abs()
    }

    #[inline(always)]
    fn faer_abs2(self) -> Self::Real {
        self * self
    }

    #[inline(always)]
    fn faer_nan() -> Self {
        Self::NAN
    }

    #[inline(always)]
    fn faer_from_real(real: Self::Real) -> Self {
        real
    }

    #[inline(always)]
    fn faer_real(self) -> Self::Real {
        self
    }

    #[inline(always)]
    fn faer_imag(self) -> Self::Real {
        0.0
    }

    #[inline(always)]
    fn faer_zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn faer_one() -> Self {
        1.0
    }

    #[inline(always)]
    fn faer_slice_as_simd<S: Simd>(
        slice: &[UnitFor<Self>],
    ) -> (&[SimdUnitFor<Self, S>], &[UnitFor<Self>]) {
        S::f64s_as_simd(slice)
    }

    #[inline(always)]
    fn faer_slice_as_mut_simd<S: Simd>(
        slice: &mut [UnitFor<Self>],
    ) -> (&mut [SimdUnitFor<Self, S>], &mut [UnitFor<Self>]) {
        S::f64s_as_mut_simd(slice)
    }

    #[inline(always)]
    fn faer_partial_load_last_unit<S: Simd>(
        simd: S,
        slice: &[UnitFor<Self>],
    ) -> SimdUnitFor<Self, S> {
        simd.f64s_partial_load_last(slice)
    }

    #[inline(always)]
    fn faer_partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [UnitFor<Self>],
        values: SimdUnitFor<Self, S>,
    ) {
        simd.f64s_partial_store_last(slice, values)
    }
    #[inline(always)]
    fn faer_partial_load_unit<S: Simd>(simd: S, slice: &[UnitFor<Self>]) -> SimdUnitFor<Self, S> {
        simd.f64s_partial_load(slice)
    }

    #[inline(always)]
    fn faer_partial_store_unit<S: Simd>(
        simd: S,
        slice: &mut [UnitFor<Self>],
        values: SimdUnitFor<Self, S>,
    ) {
        simd.f64s_partial_store(slice, values)
    }

    #[inline(always)]
    fn faer_simd_splat_unit<S: Simd>(simd: S, unit: UnitFor<Self>) -> SimdUnitFor<Self, S> {
        simd.f64s_splat(unit)
    }

    #[inline(always)]
    fn faer_simd_neg<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        simd.f64s_neg(values)
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
        simd.f64s_add(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.f64s_sub(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.f64s_mul(lhs, rhs)
    }
    #[inline(always)]
    fn faer_simd_scale_real<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self::Real, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        Self::faer_simd_mul(simd, lhs, rhs)
    }
    #[inline(always)]
    fn faer_simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.f64s_mul(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.f64s_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn faer_simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.f64s_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn faer_simd_reduce_add<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> Self {
        simd.f64s_reduce_sum(values)
    }

    #[inline(always)]
    fn faer_simd_abs2<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        simd.f64s_mul(values, values)
    }
    #[inline(always)]
    fn faer_simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self::Real, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        simd.f64s_mul_add_e(values, values, acc)
    }
    #[inline(always)]
    fn faer_simd_score<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        simd.f64s_abs(values)
    }

    #[inline(always)]
    fn faer_simd_scalar_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        let _ = simd;
        lhs * rhs
    }
    #[inline(always)]
    fn faer_simd_scalar_conj_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        let _ = simd;
        lhs * rhs
    }
    #[inline(always)]
    fn faer_simd_scalar_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        simd.f64_scalar_mul_add_e(lhs, rhs, acc)
    }
    #[inline(always)]
    fn faer_simd_scalar_conj_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        simd.f64_scalar_mul_add_e(lhs, rhs, acc)
    }
}
impl RealField for f32 {
    #[inline(always)]
    fn faer_epsilon() -> Option<Self> {
        Some(Self::EPSILON)
    }
    #[inline(always)]
    fn faer_zero_threshold() -> Option<Self> {
        Some(Self::MIN_POSITIVE)
    }

    #[inline(always)]
    fn faer_div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn faer_usize_to_index(a: usize) -> IndexFor<Self> {
        a as _
    }
    #[inline(always)]
    fn faer_index_to_usize(a: IndexFor<Self>) -> usize {
        a as _
    }
    #[inline(always)]
    fn faer_max_index() -> IndexFor<Self> {
        IndexFor::<Self>::MAX
    }

    #[inline(always)]
    fn faer_simd_less_than<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S> {
        simd.f32s_less_than(a, b)
    }

    #[inline(always)]
    fn faer_simd_less_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S> {
        simd.f32s_less_than_or_equal(a, b)
    }

    #[inline(always)]
    fn faer_simd_greater_than<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S> {
        simd.f32s_greater_than(a, b)
    }

    #[inline(always)]
    fn faer_simd_greater_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S> {
        simd.f32s_greater_than_or_equal(a, b)
    }

    #[inline(always)]
    fn faer_simd_select<S: Simd>(
        simd: S,
        mask: SimdMaskFor<Self, S>,
        if_true: SimdGroupFor<Self, S>,
        if_false: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.m32s_select_f32s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn faer_simd_index_select<S: Simd>(
        simd: S,
        mask: SimdMaskFor<Self, S>,
        if_true: SimdIndexFor<Self, S>,
        if_false: SimdIndexFor<Self, S>,
    ) -> SimdIndexFor<Self, S> {
        simd.m32s_select_u32s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn faer_simd_index_seq<S: Simd>(simd: S) -> SimdIndexFor<Self, S> {
        let _ = simd;
        pulp::cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u32])
    }

    #[inline(always)]
    fn faer_simd_index_splat<S: Simd>(simd: S, value: IndexFor<Self>) -> SimdIndexFor<Self, S> {
        simd.u32s_splat(value)
    }

    #[inline(always)]
    fn faer_simd_index_add<S: Simd>(
        simd: S,
        a: SimdIndexFor<Self, S>,
        b: SimdIndexFor<Self, S>,
    ) -> SimdIndexFor<Self, S> {
        simd.u32s_add(a, b)
    }
}
impl RealField for f64 {
    #[inline(always)]
    fn faer_epsilon() -> Option<Self> {
        Some(Self::EPSILON)
    }
    #[inline(always)]
    fn faer_zero_threshold() -> Option<Self> {
        Some(Self::MIN_POSITIVE)
    }
    #[inline(always)]
    fn faer_div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn faer_usize_to_index(a: usize) -> IndexFor<Self> {
        a as _
    }
    #[inline(always)]
    fn faer_index_to_usize(a: IndexFor<Self>) -> usize {
        a as _
    }
    #[inline(always)]
    fn faer_max_index() -> IndexFor<Self> {
        IndexFor::<Self>::MAX
    }

    #[inline(always)]
    fn faer_simd_less_than<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S> {
        simd.f64s_less_than(a, b)
    }

    #[inline(always)]
    fn faer_simd_less_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S> {
        simd.f64s_less_than_or_equal(a, b)
    }

    #[inline(always)]
    fn faer_simd_greater_than<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S> {
        simd.f64s_greater_than(a, b)
    }

    #[inline(always)]
    fn faer_simd_greater_than_or_equal<S: Simd>(
        simd: S,
        a: SimdGroupFor<Self, S>,
        b: SimdGroupFor<Self, S>,
    ) -> SimdMaskFor<Self, S> {
        simd.f64s_greater_than_or_equal(a, b)
    }

    #[inline(always)]
    fn faer_simd_select<S: Simd>(
        simd: S,
        mask: SimdMaskFor<Self, S>,
        if_true: SimdGroupFor<Self, S>,
        if_false: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.m64s_select_f64s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn faer_simd_index_select<S: Simd>(
        simd: S,
        mask: SimdMaskFor<Self, S>,
        if_true: SimdIndexFor<Self, S>,
        if_false: SimdIndexFor<Self, S>,
    ) -> SimdIndexFor<Self, S> {
        simd.m64s_select_u64s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn faer_simd_index_seq<S: Simd>(simd: S) -> SimdIndexFor<Self, S> {
        let _ = simd;
        pulp::cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u64])
    }

    #[inline(always)]
    fn faer_simd_index_splat<S: Simd>(simd: S, value: IndexFor<Self>) -> SimdIndexFor<Self, S> {
        simd.u64s_splat(value)
    }

    #[inline(always)]
    fn faer_simd_index_add<S: Simd>(
        simd: S,
        a: SimdIndexFor<Self, S>,
        b: SimdIndexFor<Self, S>,
    ) -> SimdIndexFor<Self, S> {
        simd.u64s_add(a, b)
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

unsafe impl Conjugate for f64 {
    type Conj = f64;
    type Canonical = f64;

    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        self
    }
}

pub trait SimpleEntity:
    Entity<Group = IdentityGroup, GroupCopy = IdentityGroup, Unit = Self>
{
    #[inline(always)]
    fn faer_from_group<T>(group: GroupFor<Self, T>) -> T {
        unsafe { transmute_unchecked(group) }
    }
    #[inline(always)]
    fn faer_to_group<T>(group: T) -> GroupFor<Self, T> {
        unsafe { transmute_unchecked(group) }
    }
}
impl<E: Entity<Group = IdentityGroup, GroupCopy = IdentityGroup, Unit = Self>> SimpleEntity for E {}

const _: () = {
    const fn __assert_simple_entity<E: SimpleEntity>() {}
    __assert_simple_entity::<f32>();
};

unsafe impl Entity for f32 {
    type Unit = Self;
    type Index = u32;
    type SimdUnit<S: Simd> = S::f32s;
    type SimdMask<S: Simd> = S::m32s;
    type SimdIndex<S: Simd> = S::u32s;
    type Group = IdentityGroup;
    type GroupCopy = IdentityGroup;
    type Iter<I: Iterator> = I;

    const N_COMPONENTS: usize = 1;
    const UNIT: GroupFor<Self, ()> = ();

    #[inline(always)]
    fn faer_from_units(group: GroupFor<Self, UnitFor<Self>>) -> Self {
        group
    }

    #[inline(always)]
    fn faer_into_units(self) -> GroupFor<Self, UnitFor<Self>> {
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
    fn faer_map<T, U>(group: GroupFor<Self, T>, f: impl FnMut(T) -> U) -> GroupFor<Self, U> {
        let mut f = f;
        f(group)
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

unsafe impl Entity for f64 {
    type Unit = Self;
    type Index = u64;
    type SimdUnit<S: Simd> = S::f64s;
    type SimdMask<S: Simd> = S::m64s;
    type SimdIndex<S: Simd> = S::u64s;
    type Group = IdentityGroup;
    type GroupCopy = IdentityGroup;
    type Iter<I: Iterator> = I;

    const N_COMPONENTS: usize = 1;
    const UNIT: GroupFor<Self, ()> = ();

    #[inline(always)]
    fn faer_from_units(group: GroupFor<Self, UnitFor<Self>>) -> Self {
        group
    }

    #[inline(always)]
    fn faer_into_units(self) -> GroupFor<Self, UnitFor<Self>> {
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
    fn faer_map<T, U>(group: GroupFor<Self, T>, f: impl FnMut(T) -> U) -> GroupFor<Self, U> {
        let mut f = f;
        f(group)
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

unsafe impl<E: Entity> Entity for Complex<E> {
    type Unit = UnitFor<E>;
    type Index = IndexFor<E>;
    type SimdUnit<S: Simd> = SimdUnitFor<E, S>;
    type SimdMask<S: Simd> = E::SimdMask<S>;
    type SimdIndex<S: Simd> = SimdIndexFor<E, S>;
    type Group = ComplexGroup<E::Group>;
    type GroupCopy = ComplexGroup<E::GroupCopy>;
    type Iter<I: Iterator> = ComplexIter<E::Iter<I>>;

    const N_COMPONENTS: usize = E::N_COMPONENTS * 2;
    const UNIT: GroupFor<Self, ()> = Complex {
        re: E::UNIT,
        im: E::UNIT,
    };

    #[inline(always)]
    fn faer_from_units(group: GroupFor<Self, UnitFor<Self>>) -> Self {
        let re = E::faer_from_units(group.re);
        let im = E::faer_from_units(group.im);
        Self { re, im }
    }

    #[inline(always)]
    fn faer_into_units(self) -> GroupFor<Self, UnitFor<Self>> {
        let Self { re, im } = self;
        Complex {
            re: re.faer_into_units(),
            im: im.faer_into_units(),
        }
    }

    #[inline(always)]
    fn faer_as_ref<T>(group: &GroupFor<Self, T>) -> GroupFor<Self, &T> {
        Complex {
            re: E::faer_as_ref(&group.re),
            im: E::faer_as_ref(&group.im),
        }
    }

    #[inline(always)]
    fn faer_as_mut<T>(group: &mut GroupFor<Self, T>) -> GroupFor<Self, &mut T> {
        Complex {
            re: E::faer_as_mut(&mut group.re),
            im: E::faer_as_mut(&mut group.im),
        }
    }

    #[inline(always)]
    fn faer_map<T, U>(group: GroupFor<Self, T>, f: impl FnMut(T) -> U) -> GroupFor<Self, U> {
        let mut f = f;
        Complex {
            re: E::faer_map(group.re, &mut f),
            im: E::faer_map(group.im, &mut f),
        }
    }
    #[inline(always)]
    fn faer_map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: GroupFor<Self, T>,
        f: &mut impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, GroupFor<Self, U>) {
        let (ctx, re) = E::faer_map_with_context(ctx, group.re, f);
        let (ctx, im) = E::faer_map_with_context(ctx, group.im, f);
        (ctx, Complex { re, im })
    }

    #[inline(always)]
    fn faer_zip<T, U>(
        first: GroupFor<Self, T>,
        second: GroupFor<Self, U>,
    ) -> GroupFor<Self, (T, U)> {
        Complex {
            re: E::faer_zip(first.re, second.re),
            im: E::faer_zip(first.im, second.im),
        }
    }
    #[inline(always)]
    fn faer_unzip<T, U>(zipped: GroupFor<Self, (T, U)>) -> (GroupFor<Self, T>, GroupFor<Self, U>) {
        let (re0, re1) = E::faer_unzip(zipped.re);
        let (im0, im1) = E::faer_unzip(zipped.im);
        (Complex { re: re0, im: im0 }, Complex { re: re1, im: im1 })
    }

    #[inline(always)]
    fn faer_into_iter<I: IntoIterator>(iter: GroupFor<Self, I>) -> Self::Iter<I::IntoIter> {
        ComplexIter {
            re: E::faer_into_iter(iter.re),
            im: E::faer_into_iter(iter.im),
        }
    }
}

unsafe impl<E: Entity> Entity for ComplexConj<E> {
    type Unit = UnitFor<E>;
    type Index = IndexFor<E>;
    type SimdUnit<S: Simd> = SimdUnitFor<E, S>;
    type SimdMask<S: Simd> = E::SimdMask<S>;
    type SimdIndex<S: Simd> = SimdIndexFor<E, S>;
    type Group = ComplexConjGroup<E::Group>;
    type GroupCopy = ComplexConjGroup<E::GroupCopy>;
    type Iter<I: Iterator> = ComplexConjIter<E::Iter<I>>;

    const N_COMPONENTS: usize = E::N_COMPONENTS * 2;
    const UNIT: GroupFor<Self, ()> = ComplexConj {
        re: E::UNIT,
        neg_im: E::UNIT,
    };

    #[inline(always)]
    fn faer_from_units(group: GroupFor<Self, UnitFor<Self>>) -> Self {
        let re = E::faer_from_units(group.re);
        let neg_im = E::faer_from_units(group.neg_im);
        Self { re, neg_im }
    }

    #[inline(always)]
    fn faer_into_units(self) -> GroupFor<Self, UnitFor<Self>> {
        let Self { re, neg_im } = self;
        ComplexConj {
            re: re.faer_into_units(),
            neg_im: neg_im.faer_into_units(),
        }
    }

    #[inline(always)]
    fn faer_as_ref<T>(group: &GroupFor<Self, T>) -> GroupFor<Self, &T> {
        ComplexConj {
            re: E::faer_as_ref(&group.re),
            neg_im: E::faer_as_ref(&group.neg_im),
        }
    }

    #[inline(always)]
    fn faer_as_mut<T>(group: &mut GroupFor<Self, T>) -> GroupFor<Self, &mut T> {
        ComplexConj {
            re: E::faer_as_mut(&mut group.re),
            neg_im: E::faer_as_mut(&mut group.neg_im),
        }
    }

    #[inline(always)]
    fn faer_map<T, U>(group: GroupFor<Self, T>, f: impl FnMut(T) -> U) -> GroupFor<Self, U> {
        let mut f = f;
        ComplexConj {
            re: E::faer_map(group.re, &mut f),
            neg_im: E::faer_map(group.neg_im, &mut f),
        }
    }
    #[inline(always)]
    fn faer_map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: GroupFor<Self, T>,
        f: &mut impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, GroupFor<Self, U>) {
        let (ctx, re) = E::faer_map_with_context(ctx, group.re, f);
        let (ctx, neg_im) = E::faer_map_with_context(ctx, group.neg_im, f);
        (ctx, ComplexConj { re, neg_im })
    }

    #[inline(always)]
    fn faer_zip<T, U>(
        first: GroupFor<Self, T>,
        second: GroupFor<Self, U>,
    ) -> GroupFor<Self, (T, U)> {
        ComplexConj {
            re: E::faer_zip(first.re, second.re),
            neg_im: E::faer_zip(first.neg_im, second.neg_im),
        }
    }
    #[inline(always)]
    fn faer_unzip<T, U>(zipped: GroupFor<Self, (T, U)>) -> (GroupFor<Self, T>, GroupFor<Self, U>) {
        let (re0, re1) = E::faer_unzip(zipped.re);
        let (neg_im0, neg_im1) = E::faer_unzip(zipped.neg_im);
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
    fn faer_into_iter<I: IntoIterator>(iter: GroupFor<Self, I>) -> Self::Iter<I::IntoIter> {
        ComplexConjIter {
            re: E::faer_into_iter(iter.re),
            neg_im: E::faer_into_iter(iter.neg_im),
        }
    }
}

impl<E: RealField> ComplexField for Complex<E> {
    type Real = E;
    type Simd = <E as ComplexField>::Simd;
    type ScalarSimd = <E as ComplexField>::ScalarSimd;

    #[inline(always)]
    fn faer_from_f64(value: f64) -> Self {
        Self {
            re: Self::Real::faer_from_f64(value),
            im: Self::Real::faer_zero(),
        }
    }

    #[inline(always)]
    fn faer_add(self, rhs: Self) -> Self {
        Self {
            re: self.re.faer_add(rhs.re),
            im: self.im.faer_add(rhs.im),
        }
    }

    #[inline(always)]
    fn faer_sub(self, rhs: Self) -> Self {
        Self {
            re: self.re.faer_sub(rhs.re),
            im: self.im.faer_sub(rhs.im),
        }
    }

    #[inline(always)]
    fn faer_mul(self, rhs: Self) -> Self {
        Self {
            re: Self::Real::faer_sub(self.re.faer_mul(rhs.re), self.im.faer_mul(rhs.im)),
            im: Self::Real::faer_add(self.re.faer_mul(rhs.im), self.im.faer_mul(rhs.re)),
        }
    }

    #[inline(always)]
    fn faer_neg(self) -> Self {
        Self {
            re: self.re.faer_neg(),
            im: self.im.faer_neg(),
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
            im: self.im.faer_neg(),
        }
    }

    #[inline(always)]
    fn faer_sqrt(self) -> Self {
        let (re, im) = sqrt_impl(self.re, self.im);
        Self { re, im }
    }

    #[inline(always)]
    fn faer_scale_real(self, rhs: Self::Real) -> Self {
        Self {
            re: self.re.faer_scale_real(rhs),
            im: self.im.faer_scale_real(rhs),
        }
    }

    #[inline(always)]
    fn faer_scale_power_of_two(self, rhs: Self::Real) -> Self {
        Self {
            re: self.re.faer_scale_power_of_two(rhs),
            im: self.im.faer_scale_power_of_two(rhs),
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
        Self::Real::faer_add(self.re.faer_mul(self.re), self.im.faer_mul(self.im))
    }

    #[inline(always)]
    fn faer_nan() -> Self {
        Self {
            re: Self::Real::faer_nan(),
            im: Self::Real::faer_nan(),
        }
    }

    #[inline(always)]
    fn faer_from_real(real: Self::Real) -> Self {
        Self {
            re: real,
            im: Self::Real::faer_zero(),
        }
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
        Self {
            re: Self::Real::faer_zero(),
            im: Self::Real::faer_zero(),
        }
    }

    #[inline(always)]
    fn faer_one() -> Self {
        Self {
            re: Self::Real::faer_one(),
            im: Self::Real::faer_zero(),
        }
    }

    #[inline(always)]
    fn faer_slice_as_simd<S: Simd>(
        slice: &[UnitFor<Self>],
    ) -> (&[SimdUnitFor<Self, S>], &[UnitFor<Self>]) {
        E::faer_slice_as_simd(slice)
    }

    #[inline(always)]
    fn faer_slice_as_mut_simd<S: Simd>(
        slice: &mut [UnitFor<Self>],
    ) -> (&mut [SimdUnitFor<Self, S>], &mut [UnitFor<Self>]) {
        E::faer_slice_as_mut_simd(slice)
    }

    #[inline(always)]
    fn faer_partial_load_last_unit<S: Simd>(
        simd: S,
        slice: &[UnitFor<Self>],
    ) -> SimdUnitFor<Self, S> {
        E::faer_partial_load_last_unit(simd, slice)
    }

    #[inline(always)]
    fn faer_partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [UnitFor<Self>],
        values: SimdUnitFor<Self, S>,
    ) {
        E::faer_partial_store_last_unit(simd, slice, values)
    }

    #[inline(always)]
    fn faer_partial_load_unit<S: Simd>(simd: S, slice: &[UnitFor<Self>]) -> SimdUnitFor<Self, S> {
        E::faer_partial_load_unit(simd, slice)
    }

    #[inline(always)]
    fn faer_partial_store_unit<S: Simd>(
        simd: S,
        slice: &mut [UnitFor<Self>],
        values: SimdUnitFor<Self, S>,
    ) {
        E::faer_partial_store_unit(simd, slice, values)
    }

    #[inline(always)]
    fn faer_simd_splat_unit<S: Simd>(simd: S, unit: UnitFor<Self>) -> SimdUnitFor<Self, S> {
        E::faer_simd_splat_unit(simd, unit)
    }

    #[inline(always)]
    fn faer_simd_neg<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        Complex {
            re: E::faer_simd_neg(simd, values.re),
            im: E::faer_simd_neg(simd, values.im),
        }
    }

    #[inline(always)]
    fn faer_simd_conj<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        Complex {
            re: values.re,
            im: E::faer_simd_neg(simd, values.im),
        }
    }

    #[inline(always)]
    fn faer_simd_add<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        Complex {
            re: E::faer_simd_add(simd, lhs.re, rhs.re),
            im: E::faer_simd_add(simd, lhs.im, rhs.im),
        }
    }

    #[inline(always)]
    fn faer_simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        Complex {
            re: E::faer_simd_sub(simd, lhs.re, rhs.re),
            im: E::faer_simd_sub(simd, lhs.im, rhs.im),
        }
    }

    #[inline(always)]
    fn faer_simd_scalar_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        Complex {
            re: E::faer_simd_scalar_mul_adde(
                simd,
                lhs.re,
                rhs.re,
                E::faer_simd_scalar_mul(simd, lhs.im.faer_neg(), rhs.im),
            ),
            im: E::faer_simd_scalar_mul_adde(
                simd,
                lhs.re,
                rhs.im,
                E::faer_simd_scalar_mul(simd, lhs.im, rhs.re),
            ),
        }
    }

    #[inline(always)]
    fn faer_simd_scalar_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        Complex {
            re: E::faer_simd_scalar_mul_adde(
                simd,
                lhs.re,
                rhs.re,
                E::faer_simd_scalar_mul_adde(simd, lhs.im.faer_neg(), rhs.im, acc.re),
            ),
            im: E::faer_simd_scalar_mul_adde(
                simd,
                lhs.re,
                rhs.im,
                E::faer_simd_scalar_mul_adde(simd, lhs.im, rhs.re, acc.im),
            ),
        }
    }

    #[inline(always)]
    fn faer_simd_scalar_conj_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        Complex {
            re: E::faer_simd_scalar_mul_adde(
                simd,
                lhs.re,
                rhs.re,
                E::faer_simd_scalar_mul_adde(simd, lhs.im, rhs.im, acc.re),
            ),
            im: E::faer_simd_scalar_mul_adde(
                simd,
                lhs.re,
                rhs.im,
                E::faer_simd_scalar_mul_adde(simd, lhs.im.faer_neg(), rhs.re, acc.im),
            ),
        }
    }

    #[inline(always)]
    fn faer_simd_scalar_conj_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        Complex {
            re: E::faer_simd_scalar_mul_adde(
                simd,
                lhs.re,
                rhs.re,
                E::faer_simd_scalar_mul(simd, lhs.im, rhs.im),
            ),
            im: E::faer_simd_scalar_mul_adde(
                simd,
                lhs.re,
                rhs.im,
                E::faer_simd_scalar_mul(simd, lhs.im.faer_neg(), rhs.re),
            ),
        }
    }

    #[inline(always)]
    fn faer_simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        Complex {
            re: E::faer_simd_mul_adde(
                simd,
                E::faer_copy(&lhs.re),
                E::faer_copy(&rhs.re),
                E::faer_simd_mul(
                    simd,
                    E::faer_simd_neg(simd, E::faer_copy(&lhs.im)),
                    E::faer_copy(&rhs.im),
                ),
            ),
            im: E::faer_simd_mul_adde(simd, lhs.re, rhs.im, E::faer_simd_mul(simd, lhs.im, rhs.re)),
        }
    }

    #[inline(always)]
    fn faer_simd_scale_real<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self::Real, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        Complex {
            re: E::faer_simd_mul(simd, E::faer_copy(&lhs), rhs.re),
            im: E::faer_simd_mul(simd, lhs, rhs.im),
        }
    }

    #[inline(always)]
    fn faer_simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        Complex {
            re: E::faer_simd_mul_adde(
                simd,
                E::faer_copy(&lhs.re),
                E::faer_copy(&rhs.re),
                E::faer_simd_mul(simd, E::faer_copy(&lhs.im), E::faer_copy(&rhs.im)),
            ),
            im: E::faer_simd_mul_adde(
                simd,
                lhs.re,
                rhs.im,
                E::faer_simd_mul(simd, E::faer_simd_neg(simd, lhs.im), rhs.re),
            ),
        }
    }

    #[inline(always)]
    fn faer_simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        Complex {
            re: E::faer_simd_mul_adde(
                simd,
                E::faer_copy(&lhs.re),
                E::faer_copy(&rhs.re),
                E::faer_simd_mul_adde(
                    simd,
                    E::faer_simd_neg(simd, E::faer_copy(&lhs.im)),
                    E::faer_copy(&rhs.im),
                    acc.re,
                ),
            ),
            im: E::faer_simd_mul_adde(
                simd,
                lhs.re,
                rhs.im,
                E::faer_simd_mul_adde(simd, lhs.im, rhs.re, acc.im),
            ),
        }
    }

    #[inline(always)]
    fn faer_simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        Complex {
            re: E::faer_simd_mul_adde(
                simd,
                E::faer_copy(&lhs.re),
                E::faer_copy(&rhs.re),
                E::faer_simd_mul_adde(simd, E::faer_copy(&lhs.im), E::faer_copy(&rhs.im), acc.re),
            ),
            im: E::faer_simd_mul_adde(
                simd,
                lhs.re,
                rhs.im,
                E::faer_simd_mul_adde(simd, E::faer_simd_neg(simd, lhs.im), rhs.re, acc.im),
            ),
        }
    }

    #[inline(always)]
    fn faer_simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self::Real, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        E::faer_simd_mul_adde(
            simd,
            E::faer_copy(&values.re),
            values.re,
            E::faer_simd_mul_adde(simd, E::faer_copy(&values.im), values.im, acc),
        )
    }
    #[inline(always)]
    fn faer_simd_abs2<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        Self::faer_simd_score(simd, values)
    }
    #[inline(always)]
    fn faer_simd_score<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        E::faer_simd_mul_adde(
            simd,
            E::faer_copy(&values.re),
            values.re,
            E::faer_simd_mul(simd, E::faer_copy(&values.im), values.im),
        )
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

/// Utilities for split complex number types whose real and imaginary parts are stored separately.
pub mod complex_split {

    /// This structure contains the real and imaginary parts of an implicity conjugated value.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[repr(C)]
    pub struct ComplexConj<T> {
        pub re: T,
        pub neg_im: T,
    }

    unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for ComplexConj<T> {}
    unsafe impl<T: bytemuck::Pod> bytemuck::Pod for ComplexConj<T> {}

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

use complex_split::*;

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
            im: self.neg_im.faer_neg(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_sqrt() {
        for _ in 0..100 {
            let a = num_complex::Complex64::new(rand::random(), rand::random());
            let num_complex::Complex {
                re: target_re,
                im: target_im,
            } = a.faer_sqrt();
            let (sqrt_re, sqrt_im) = sqrt_impl(a.re, a.im);
            assert_approx_eq!(target_re, sqrt_re);
            assert_approx_eq!(target_im, sqrt_im);
        }
    }
}
