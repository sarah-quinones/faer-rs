use bytemuck::{Pod, Zeroable};
use faer_entity::{
    pulp::{Scalar, Simd},
    *,
};

// https://web.mit.edu/tabbott/Public/quaddouble-debian/qd-2.3.4-old/docs/qd.pdf
// https://gitlab.com/hodge_star/mantis

/// Value representing the implicit sum of two floating point terms, such that the absolute
/// value of the second term is less half a ULP of the first term.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C)]
pub struct Double<T>(pub T, pub T);

unsafe impl<T: Zeroable> Zeroable for Double<T> {}
unsafe impl<T: Pod> Pod for Double<T> {}

impl<I: Iterator> Iterator for Double<I> {
    type Item = Double<I::Item>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let x0 = self.0.next()?;
        let x1 = self.1.next()?;
        Some(Double(x0, x1))
    }
}

#[inline(always)]
#[allow(dead_code)]
fn two_sum_e<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let sign_bit = simd.f64s_splat(-0.0);
    let cmp = simd.u64s_greater_than(
        pulp::cast(simd.f64s_or(a, sign_bit)),
        pulp::cast(simd.f64s_or(b, sign_bit)),
    );
    let (a, b) = (
        simd.m64s_select_f64s(cmp, a, b),
        simd.m64s_select_f64s(cmp, b, a),
    );

    quick_two_sum(simd, a, b)
}

#[inline(always)]
#[allow(dead_code)]
fn quick_two_diff<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let s = simd.f64s_sub(a, b);
    let err = simd.f64s_sub(simd.f64s_sub(a, s), b);
    (s, err)
}

#[inline(always)]
#[allow(dead_code)]
fn two_diff<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let s = simd.f64s_sub(a, b);
    let bb = simd.f64s_sub(s, a);

    // (a - (s - bb)) - (b + bb)
    let err = simd.f64s_sub(simd.f64s_sub(a, simd.f64s_sub(s, bb)), simd.f64s_add(b, bb));
    (s, err)
}

#[inline(always)]
#[allow(dead_code)]
fn two_diff_e<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    two_sum_e(simd, a, simd.f64s_neg(b))
}

#[inline(always)]
fn quick_two_sum<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let s = simd.f64s_add(a, b);
    let err = simd.f64s_sub(b, simd.f64s_sub(s, a));
    (s, err)
}

#[inline(always)]
fn two_sum<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let s = simd.f64s_add(a, b);
    let bb = simd.f64s_sub(s, a);

    // (a - (s - bb)) + (b - bb)
    let err = simd.f64s_add(simd.f64s_sub(a, simd.f64s_sub(s, bb)), simd.f64s_sub(b, bb));
    (s, err)
}

#[inline(always)]
fn two_prod<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let p = simd.f64s_mul(a, b);
    let err = simd.f64s_mul_add(a, b, simd.f64s_neg(p));

    (p, err)
}

pub mod double {
    use super::*;

    #[inline(always)]
    pub fn simd_add<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> Double<S::f64s> {
        let (s, e) = two_sum_e(simd, a.0, b.0);
        let e = simd.f64s_add(e, simd.f64s_add(a.1, b.1));
        let (s, e) = quick_two_sum(simd, s, e);
        Double(s, e)
    }

    #[inline(always)]
    pub fn simd_sub<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> Double<S::f64s> {
        let (s, e) = two_diff_e(simd, a.0, b.0);
        let e = simd.f64s_add(e, a.1);
        let e = simd.f64s_sub(e, b.1);
        let (s, e) = quick_two_sum(simd, s, e);
        Double(s, e)
    }

    #[inline(always)]
    pub fn simd_neg<S: Simd>(simd: S, a: Double<S::f64s>) -> Double<S::f64s> {
        Double(simd.f64s_neg(a.0), simd.f64s_neg(a.1))
    }

    #[inline(always)]
    pub fn simd_mul<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> Double<S::f64s> {
        let (p1, p2) = two_prod(simd, a.0, b.0);
        let p2 = simd.f64s_add(
            p2,
            simd.f64s_add(simd.f64s_mul(a.0, b.1), simd.f64s_mul(a.1, b.0)),
        );
        let (p1, p2) = quick_two_sum(simd, p1, p2);
        Double(p1, p2)
    }

    #[inline(always)]
    fn simd_mul_f64<S: Simd>(simd: S, a: Double<S::f64s>, b: S::f64s) -> Double<S::f64s> {
        let (p1, p2) = two_prod(simd, a.0, b);
        let p2 = simd.f64s_add(p2, simd.f64s_mul(a.1, b));
        let (p1, p2) = quick_two_sum(simd, p1, p2);
        Double(p1, p2)
    }

    pub fn simd_select<S: Simd>(
        simd: S,
        mask: S::m64s,
        if_true: Double<S::f64s>,
        if_false: Double<S::f64s>,
    ) -> Double<S::f64s> {
        Double(
            simd.m64s_select_f64s(mask, if_true.0, if_false.0),
            simd.m64s_select_f64s(mask, if_true.1, if_false.1),
        )
    }

    #[inline]
    pub fn simd_div<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> Double<S::f64s> {
        simd.vectorize(
            #[inline(always)]
            || {
                let pos_zero = simd.f64s_splat(0.0);
                let pos_infty = simd.f64s_splat(f64::INFINITY);
                let sign_bit = simd.f64s_splat(-0.0);

                let a_sign = simd.f64s_and(a.0, sign_bit);
                let b_sign = simd.f64s_and(b.0, sign_bit);

                let combined_sign = simd.f64s_xor(a_sign, b_sign);

                let a_is_zero = simd_eq(simd, a, Double(pos_zero, pos_zero));
                let b_is_zero = simd_eq(simd, b, Double(pos_zero, pos_zero));
                let a_is_infty = simd_eq(
                    simd,
                    Double(simd.f64s_abs(a.0), simd.f64s_abs(a.1)),
                    Double(pos_infty, pos_infty),
                );
                let b_is_infty = simd_eq(
                    simd,
                    Double(simd.f64s_abs(b.0), simd.f64s_abs(b.1)),
                    Double(pos_infty, pos_infty),
                );

                let q1 = simd.f64s_div(a.0, b.0);
                let r = simd_mul_f64(simd, b, q1);

                let (s1, s2) = two_diff(simd, a.0, r.0);
                let s2 = simd.f64s_sub(s2, r.1);
                let s2 = simd.f64s_add(s2, a.1);

                let q2 = simd.f64s_div(simd.f64s_add(s1, s2), b.0);
                let (q0, q1) = quick_two_sum(simd, q1, q2);

                simd_select(
                    simd,
                    simd.m64s_and(b_is_zero, simd.m64s_not(a_is_zero)),
                    Double(
                        simd.f64s_or(combined_sign, pos_infty),
                        simd.f64s_or(combined_sign, pos_infty),
                    ),
                    simd_select(
                        simd,
                        simd.m64s_and(b_is_infty, simd.m64s_not(a_is_infty)),
                        Double(
                            simd.f64s_or(combined_sign, pos_zero),
                            simd.f64s_or(combined_sign, pos_zero),
                        ),
                        Double(q0, q1),
                    ),
                )
            },
        )
    }

    #[inline(always)]
    pub fn simd_abs<S: Simd>(simd: S, a: Double<S::f64s>) -> Double<S::f64s> {
        let is_negative = simd.f64s_less_than(a.0, simd.f64s_splat(0.0));
        Double(
            simd.m64s_select_f64s(is_negative, simd.f64s_neg(a.0), a.0),
            simd.m64s_select_f64s(is_negative, simd.f64s_neg(a.1), a.1),
        )
    }

    #[inline(always)]
    pub fn simd_less_than<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> S::m64s {
        let lt0 = simd.f64s_less_than(a.0, b.0);
        let eq0 = simd.f64s_equal(a.0, b.0);
        let lt1 = simd.f64s_less_than(a.1, b.1);
        simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
    }

    #[inline(always)]
    pub fn simd_less_than_or_equal<S: Simd>(
        simd: S,
        a: Double<S::f64s>,
        b: Double<S::f64s>,
    ) -> S::m64s {
        let lt0 = simd.f64s_less_than(a.0, b.0);
        let eq0 = simd.f64s_equal(a.0, b.0);
        let lt1 = simd.f64s_less_than_or_equal(a.1, b.1);
        simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
    }

    #[inline(always)]
    pub fn simd_greater_than<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> S::m64s {
        let lt0 = simd.f64s_greater_than(a.0, b.0);
        let eq0 = simd.f64s_equal(a.0, b.0);
        let lt1 = simd.f64s_greater_than(a.1, b.1);
        simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
    }

    #[inline(always)]
    pub fn simd_greater_than_or_equal<S: Simd>(
        simd: S,
        a: Double<S::f64s>,
        b: Double<S::f64s>,
    ) -> S::m64s {
        let lt0 = simd.f64s_greater_than(a.0, b.0);
        let eq0 = simd.f64s_equal(a.0, b.0);
        let lt1 = simd.f64s_greater_than_or_equal(a.1, b.1);
        simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
    }

    #[inline(always)]
    pub fn simd_eq<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> S::m64s {
        let eq0 = simd.f64s_equal(a.0, b.0);
        let eq1 = simd.f64s_equal(a.1, b.1);
        simd.m64s_and(eq0, eq1)
    }
}

impl core::ops::Add for Double<f64> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        double::simd_add(Scalar::new(), self, rhs)
    }
}

impl core::ops::Sub for Double<f64> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        double::simd_sub(Scalar::new(), self, rhs)
    }
}

impl core::ops::Mul for Double<f64> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        double::simd_mul(Scalar::new(), self, rhs)
    }
}

impl core::ops::Div for Double<f64> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        double::simd_div(Scalar::new(), self, rhs)
    }
}

impl core::ops::AddAssign for Double<f64> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for Double<f64> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for Double<f64> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::DivAssign for Double<f64> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl core::ops::Neg for Double<f64> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}

impl Double<f64> {
    /// 2.0^{-100}
    pub const EPSILON: Self = Self(7.888609052210118e-31, 0.0);
    /// 2.0^{-970}: precision below this value begins to degrade.
    pub const MIN_POSITIVE: Self = Self(1.0020841800044864e-292, 0.0);

    pub const ZERO: Self = Self(0.0, 0.0);
    pub const NAN: Self = Self(f64::NAN, f64::NAN);
    pub const INFINITY: Self = Self(f64::INFINITY, f64::INFINITY);

    #[inline(always)]
    pub fn abs(self) -> Self {
        double::simd_abs(Scalar::new(), self)
    }

    #[inline(always)]
    pub fn recip(self) -> Self {
        double::simd_div(Scalar::new(), Self(1.0, 0.0), self)
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        if self == Self::ZERO {
            Self::ZERO
        } else if self < Self::ZERO {
            Self::NAN
        } else if self == Self::INFINITY {
            Self::INFINITY
        } else {
            let a = self;
            let x = a.0.sqrt().recip();
            let ax = Self(a.0 * x, 0.0);

            ax + (a - ax * ax) * Double(x * 0.5, 0.0)
        }
    }
}

pub struct DoubleGroup {
    __private: (),
}

impl ForType for DoubleGroup {
    type FaerOf<T> = Double<T>;
}
impl ForCopyType for DoubleGroup {
    type FaerOfCopy<T: Copy> = Double<T>;
}
impl ForDebugType for DoubleGroup {
    type FaerOfDebug<T: core::fmt::Debug> = Double<T>;
}

mod faer_impl {
    use super::*;
    use faer::{ComplexField, Conjugate, Entity, RealField};

    unsafe impl Entity for Double<f64> {
        type Unit = f64;
        type Index = u64;

        type SimdUnit<S: Simd> = S::f64s;
        type SimdMask<S: Simd> = S::m64s;
        type SimdIndex<S: Simd> = S::u64s;

        type Group = DoubleGroup;
        type Iter<I: Iterator> = Double<I>;

        type PrefixUnit<'a, S: Simd> = pulp::Prefix<'a, f64, S, S::m64s>;
        type SuffixUnit<'a, S: Simd> = pulp::Suffix<'a, f64, S, S::m64s>;
        type PrefixMutUnit<'a, S: Simd> = pulp::PrefixMut<'a, f64, S, S::m64s>;
        type SuffixMutUnit<'a, S: Simd> = pulp::SuffixMut<'a, f64, S, S::m64s>;

        const N_COMPONENTS: usize = 2;
        const UNIT: GroupCopyFor<Self, ()> = Double((), ());

        #[inline(always)]
        fn faer_first<T>(group: GroupFor<Self, T>) -> T {
            group.0
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
            Double(&group.0, &group.1)
        }

        #[inline(always)]
        fn faer_as_mut<T>(group: &mut GroupFor<Self, T>) -> GroupFor<Self, &mut T> {
            Double(&mut group.0, &mut group.1)
        }

        #[inline(always)]
        fn faer_as_ptr<T>(group: *mut GroupFor<Self, T>) -> GroupFor<Self, *mut T> {
            unsafe {
                Double(
                    core::ptr::addr_of_mut!((*group).0),
                    core::ptr::addr_of_mut!((*group).1),
                )
            }
        }

        #[inline(always)]
        fn faer_map_impl<T, U>(
            group: GroupFor<Self, T>,
            f: &mut impl FnMut(T) -> U,
        ) -> GroupFor<Self, U> {
            Double((*f)(group.0), (*f)(group.1))
        }

        #[inline(always)]
        fn faer_zip<T, U>(
            first: GroupFor<Self, T>,
            second: GroupFor<Self, U>,
        ) -> GroupFor<Self, (T, U)> {
            Double((first.0, second.0), (first.1, second.1))
        }

        #[inline(always)]
        fn faer_unzip<T, U>(
            zipped: GroupFor<Self, (T, U)>,
        ) -> (GroupFor<Self, T>, GroupFor<Self, U>) {
            (
                Double(zipped.0 .0, zipped.1 .0),
                Double(zipped.0 .1, zipped.1 .1),
            )
        }

        #[inline(always)]
        fn faer_map_with_context<Ctx, T, U>(
            ctx: Ctx,
            group: GroupFor<Self, T>,
            f: &mut impl FnMut(Ctx, T) -> (Ctx, U),
        ) -> (Ctx, GroupFor<Self, U>) {
            let (ctx, x0) = (*f)(ctx, group.0);
            let (ctx, x1) = (*f)(ctx, group.1);
            (ctx, Double(x0, x1))
        }

        #[inline(always)]
        fn faer_into_iter<I: IntoIterator>(iter: GroupFor<Self, I>) -> Self::Iter<I::IntoIter> {
            Double(iter.0.into_iter(), iter.1.into_iter())
        }
    }

    unsafe impl Conjugate for Double<f64> {
        type Conj = Double<f64>;
        type Canonical = Double<f64>;
        #[inline(always)]
        fn canonicalize(self) -> Self::Canonical {
            self
        }
    }

    impl RealField for Double<f64> {
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
        fn faer_usize_to_index(a: usize) -> Self::Index {
            a as _
        }

        #[inline(always)]
        fn faer_index_to_usize(a: Self::Index) -> usize {
            a as _
        }

        #[inline(always)]
        fn faer_max_index() -> Self::Index {
            Self::Index::MAX
        }

        #[inline(always)]
        fn faer_simd_less_than<S: Simd>(
            simd: S,
            a: SimdGroupFor<Self, S>,
            b: SimdGroupFor<Self, S>,
        ) -> Self::SimdMask<S> {
            double::simd_less_than(simd, a, b)
        }

        #[inline(always)]
        fn faer_simd_less_than_or_equal<S: Simd>(
            simd: S,
            a: SimdGroupFor<Self, S>,
            b: SimdGroupFor<Self, S>,
        ) -> Self::SimdMask<S> {
            double::simd_less_than_or_equal(simd, a, b)
        }

        #[inline(always)]
        fn faer_simd_greater_than<S: Simd>(
            simd: S,
            a: SimdGroupFor<Self, S>,
            b: SimdGroupFor<Self, S>,
        ) -> Self::SimdMask<S> {
            double::simd_greater_than(simd, a, b)
        }

        #[inline(always)]
        fn faer_simd_greater_than_or_equal<S: Simd>(
            simd: S,
            a: SimdGroupFor<Self, S>,
            b: SimdGroupFor<Self, S>,
        ) -> Self::SimdMask<S> {
            double::simd_greater_than_or_equal(simd, a, b)
        }

        #[inline(always)]
        fn faer_simd_select<S: Simd>(
            simd: S,
            mask: Self::SimdMask<S>,
            if_true: SimdGroupFor<Self, S>,
            if_false: SimdGroupFor<Self, S>,
        ) -> SimdGroupFor<Self, S> {
            double::simd_select(simd, mask, if_true, if_false)
        }

        #[inline(always)]
        fn faer_simd_index_select<S: Simd>(
            simd: S,
            mask: Self::SimdMask<S>,
            if_true: Self::SimdIndex<S>,
            if_false: Self::SimdIndex<S>,
        ) -> Self::SimdIndex<S> {
            simd.m64s_select_u64s(mask, if_true, if_false)
        }

        #[inline(always)]
        fn faer_simd_index_seq<S: Simd>(simd: S) -> Self::SimdIndex<S> {
            let _ = simd;
            pulp::cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u64])
        }

        #[inline(always)]
        fn faer_simd_index_splat<S: Simd>(simd: S, value: Self::Index) -> Self::SimdIndex<S> {
            simd.u64s_splat(value)
        }

        #[inline(always)]
        fn faer_simd_index_add<S: Simd>(
            simd: S,
            a: Self::SimdIndex<S>,
            b: Self::SimdIndex<S>,
        ) -> Self::SimdIndex<S> {
            simd.u64s_add(a, b)
        }

        #[inline(always)]
        fn faer_min_positive() -> Self {
            Self::MIN_POSITIVE
        }

        #[inline(always)]
        fn faer_min_positive_inv() -> Self {
            Self::MIN_POSITIVE.recip()
        }

        #[inline(always)]
        fn faer_min_positive_sqrt() -> Self {
            Self::MIN_POSITIVE.sqrt()
        }

        #[inline(always)]
        fn faer_min_positive_sqrt_inv() -> Self {
            Self::MIN_POSITIVE.sqrt().recip()
        }

        #[inline(always)]
        fn faer_simd_index_rotate_left<S: Simd>(
            simd: S,
            values: SimdIndexFor<Self, S>,
            amount: usize,
        ) -> SimdIndexFor<Self, S> {
            simd.u64s_rotate_left(values, amount)
        }

        #[inline(always)]
        fn faer_simd_abs<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
            double::simd_abs(simd, values)
        }
    }

    impl ComplexField for Double<f64> {
        type Real = Double<f64>;
        type Simd = pulp::Arch;
        type ScalarSimd = pulp::Arch;
        type PortableSimd = pulp::Arch;

        #[inline(always)]
        fn faer_sqrt(self) -> Self {
            self.sqrt()
        }

        #[inline(always)]
        fn faer_from_f64(value: f64) -> Self {
            Self(value, 0.0)
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
        fn faer_scale_real(self, rhs: Self::Real) -> Self {
            self * rhs
        }

        #[inline(always)]
        fn faer_scale_power_of_two(self, rhs: Self::Real) -> Self {
            Self(self.0 * rhs.0, self.1 * rhs.0)
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
            Self::ZERO
        }

        #[inline(always)]
        fn faer_zero() -> Self {
            Self::ZERO
        }

        #[inline(always)]
        fn faer_one() -> Self {
            Self(1.0, 0.0)
        }

        #[inline(always)]
        fn faer_slice_as_simd<S: Simd>(
            slice: &[Self::Unit],
        ) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
            S::f64s_as_simd(slice)
        }

        #[inline(always)]
        fn faer_slice_as_simd_mut<S: Simd>(
            slice: &mut [Self::Unit],
        ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
            S::f64s_as_mut_simd(slice)
        }

        #[inline(always)]
        fn faer_partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
            simd.f64s_partial_load(slice)
        }

        #[inline(always)]
        fn faer_partial_store_unit<S: Simd>(
            simd: S,
            slice: &mut [Self::Unit],
            values: Self::SimdUnit<S>,
        ) {
            simd.f64s_partial_store(slice, values)
        }

        #[inline(always)]
        fn faer_partial_load_last_unit<S: Simd>(
            simd: S,
            slice: &[Self::Unit],
        ) -> Self::SimdUnit<S> {
            simd.f64s_partial_load_last(slice)
        }

        #[inline(always)]
        fn faer_partial_store_last_unit<S: Simd>(
            simd: S,
            slice: &mut [Self::Unit],
            values: Self::SimdUnit<S>,
        ) {
            simd.f64s_partial_store_last(slice, values)
        }

        #[inline(always)]
        fn faer_simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
            simd.f64s_splat(unit)
        }

        #[inline(always)]
        fn faer_simd_neg<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
            double::simd_neg(simd, values)
        }

        #[inline(always)]
        fn faer_simd_conj<S: Simd>(
            simd: S,
            values: SimdGroupFor<Self, S>,
        ) -> SimdGroupFor<Self, S> {
            let _ = simd;
            values
        }

        #[inline(always)]
        fn faer_simd_add<S: Simd>(
            simd: S,
            lhs: SimdGroupFor<Self, S>,
            rhs: SimdGroupFor<Self, S>,
        ) -> SimdGroupFor<Self, S> {
            double::simd_add(simd, lhs, rhs)
        }

        #[inline(always)]
        fn faer_simd_sub<S: Simd>(
            simd: S,
            lhs: SimdGroupFor<Self, S>,
            rhs: SimdGroupFor<Self, S>,
        ) -> SimdGroupFor<Self, S> {
            double::simd_sub(simd, lhs, rhs)
        }

        #[inline(always)]
        fn faer_simd_mul<S: Simd>(
            simd: S,
            lhs: SimdGroupFor<Self, S>,
            rhs: SimdGroupFor<Self, S>,
        ) -> SimdGroupFor<Self, S> {
            double::simd_mul(simd, lhs, rhs)
        }

        #[inline(always)]
        fn faer_simd_scale_real<S: Simd>(
            simd: S,
            lhs: SimdGroupFor<Self, S>,
            rhs: SimdGroupFor<Self, S>,
        ) -> SimdGroupFor<Self, S> {
            double::simd_mul(simd, lhs, rhs)
        }

        #[inline(always)]
        fn faer_simd_conj_mul<S: Simd>(
            simd: S,
            lhs: SimdGroupFor<Self, S>,
            rhs: SimdGroupFor<Self, S>,
        ) -> SimdGroupFor<Self, S> {
            double::simd_mul(simd, lhs, rhs)
        }

        #[inline(always)]
        fn faer_simd_mul_adde<S: Simd>(
            simd: S,
            lhs: SimdGroupFor<Self, S>,
            rhs: SimdGroupFor<Self, S>,
            acc: SimdGroupFor<Self, S>,
        ) -> SimdGroupFor<Self, S> {
            double::simd_add(simd, acc, double::simd_mul(simd, lhs, rhs))
        }

        #[inline(always)]
        fn faer_simd_conj_mul_adde<S: Simd>(
            simd: S,
            lhs: SimdGroupFor<Self, S>,
            rhs: SimdGroupFor<Self, S>,
            acc: SimdGroupFor<Self, S>,
        ) -> SimdGroupFor<Self, S> {
            double::simd_add(simd, acc, double::simd_mul(simd, lhs, rhs))
        }

        #[inline(always)]
        fn faer_simd_score<S: Simd>(
            simd: S,
            values: SimdGroupFor<Self, S>,
        ) -> SimdGroupFor<Self::Real, S> {
            double::simd_abs(simd, values)
        }

        #[inline(always)]
        fn faer_simd_abs2_adde<S: Simd>(
            simd: S,
            values: SimdGroupFor<Self, S>,
            acc: SimdGroupFor<Self::Real, S>,
        ) -> SimdGroupFor<Self::Real, S> {
            Self::faer_simd_add(simd, acc, Self::faer_simd_mul(simd, values, values))
        }

        #[inline(always)]
        fn faer_simd_abs2<S: Simd>(
            simd: S,
            values: SimdGroupFor<Self, S>,
        ) -> SimdGroupFor<Self::Real, S> {
            Self::faer_simd_mul(simd, values, values)
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
            let _ = simd;
            lhs * rhs + acc
        }

        #[inline(always)]
        fn faer_simd_scalar_conj_mul_adde<S: Simd>(
            simd: S,
            lhs: Self,
            rhs: Self,
            acc: Self,
        ) -> Self {
            let _ = simd;
            lhs * rhs + acc
        }

        #[inline(always)]
        fn faer_slice_as_aligned_simd<S: Simd>(
            simd: S,
            slice: &[UnitFor<Self>],
            offset: pulp::Offset<SimdMaskFor<Self, S>>,
        ) -> (
            pulp::Prefix<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
            &[SimdUnitFor<Self, S>],
            pulp::Suffix<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
        ) {
            simd.f64s_as_aligned_simd(slice, offset)
        }
        #[inline(always)]
        fn faer_slice_as_aligned_simd_mut<S: Simd>(
            simd: S,
            slice: &mut [UnitFor<Self>],
            offset: pulp::Offset<SimdMaskFor<Self, S>>,
        ) -> (
            pulp::PrefixMut<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
            &mut [SimdUnitFor<Self, S>],
            pulp::SuffixMut<'_, UnitFor<Self>, S, SimdMaskFor<Self, S>>,
        ) {
            simd.f64s_as_aligned_mut_simd(slice, offset)
        }

        #[inline(always)]
        fn faer_simd_rotate_left<S: Simd>(
            simd: S,
            values: SimdGroupFor<Self, S>,
            amount: usize,
        ) -> SimdGroupFor<Self, S> {
            Double(
                simd.f64s_rotate_left(values.0, amount),
                simd.f64s_rotate_left(values.1, amount),
            )
        }

        #[inline(always)]
        fn faer_align_offset<S: Simd>(
            simd: S,
            ptr: *const UnitFor<Self>,
            len: usize,
        ) -> pulp::Offset<SimdMaskFor<Self, S>> {
            simd.f64s_align_offset(ptr, len)
        }
    }
}
