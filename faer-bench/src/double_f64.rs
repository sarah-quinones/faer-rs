use pulp::{Scalar, Simd};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Double<T>(pub T, pub T);
pub type DoubleF64 = Double<f64>;

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
#[allow(dead_code)]
fn quick_two_diff<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let s = simd.f64s_sub(a, b);
    let err = simd.f64s_sub(simd.f64s_sub(a, s), b);
    (s, err)
}

#[inline(always)]
fn two_diff<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let s = simd.f64s_sub(a, b);
    let bb = simd.f64s_sub(s, a);

    // (a - (s - bb)) - (b + bb)
    let err = simd.f64s_sub(simd.f64s_sub(a, simd.f64s_sub(s, bb)), simd.f64s_add(b, bb));
    (s, err)
}

#[inline(always)]
fn two_prod<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
    let p = simd.f64s_mul(a, b);
    let err = simd.f64s_mul_add(a, b, simd.f64s_neg(p));

    (p, err)
}

pub mod simd {
    use super::*;

    #[inline(always)]
    pub fn simd_add<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> Double<S::f64s> {
        let (s, e) = two_sum(simd, a.0, b.0);
        let e = simd.f64s_add(e, simd.f64s_add(a.1, b.1));
        let (s, e) = quick_two_sum(simd, s, e);
        Double(s, e)
    }

    #[inline(always)]
    pub fn simd_sub<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> Double<S::f64s> {
        let (s, e) = two_diff(simd, a.0, b.0);
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
    pub fn simd_mul_power_of_two<S: Simd>(
        simd: S,
        a: Double<S::f64s>,
        b: S::f64s,
    ) -> Double<S::f64s> {
        Double(simd.f64s_mul(a.0, b), simd.f64s_mul(a.1, b))
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
                let (r0, r1) = quick_two_sum(simd, q1, q2);

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
                        Double(r0, r1),
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

impl core::ops::Add for DoubleF64 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        simd::simd_add(Scalar::new(), self, rhs)
    }
}

impl core::ops::Sub for DoubleF64 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        simd::simd_sub(Scalar::new(), self, rhs)
    }
}

impl core::ops::Mul for DoubleF64 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        simd::simd_mul(Scalar::new(), self, rhs)
    }
}

impl core::ops::Div for DoubleF64 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        simd::simd_div(Scalar::new(), self, rhs)
    }
}

impl core::ops::AddAssign for DoubleF64 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for DoubleF64 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for DoubleF64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::DivAssign for DoubleF64 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl core::ops::Neg for DoubleF64 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}

impl DoubleF64 {
    /// 2.0^{-100}
    pub const EPSILON: Self = Self(7.888609052210118e-31, 0.0);
    /// 2.0^{-970}: precision below this value begins to degrade.
    pub const MIN_POSITIVE: Self = Self(1.0020841800044864e-292, 0.0);

    pub const ZERO: Self = Self(0.0, 0.0);
    pub const NAN: Self = Self(f64::NAN, f64::NAN);
    pub const INFINITY: Self = Self(f64::INFINITY, f64::INFINITY);
    pub const NEG_INFINITY: Self = Self(f64::NEG_INFINITY, f64::NEG_INFINITY);

    pub const LN_2: Self = Self(core::f64::consts::LN_2, 2.3190468138462996e-17);
    pub const LN_10: Self = Self(core::f64::consts::LN_10, -2.1707562233822494e-16);

    const INV_FACT: [Self; 15] = [
        Self(1.66666666666666657e-01, 9.25185853854297066e-18),
        Self(4.16666666666666644e-02, 2.31296463463574266e-18),
        Self(8.33333333333333322e-03, 1.15648231731787138e-19),
        Self(1.38888888888888894e-03, -5.30054395437357706e-20),
        Self(1.98412698412698413e-04, 1.72095582934207053e-22),
        Self(2.48015873015873016e-05, 2.15119478667758816e-23),
        Self(2.75573192239858925e-06, -1.85839327404647208e-22),
        Self(2.75573192239858883e-07, 2.37677146222502973e-23),
        Self(2.50521083854417202e-08, -1.44881407093591197e-24),
        Self(2.08767569878681002e-09, -1.20734505911325997e-25),
        Self(1.60590438368216133e-10, 1.25852945887520981e-26),
        Self(1.14707455977297245e-11, 2.06555127528307454e-28),
        Self(7.64716373181981641e-13, 7.03872877733453001e-30),
        Self(4.77947733238738525e-14, 4.39920548583408126e-31),
        Self(2.81145725434552060e-15, 1.65088427308614326e-31),
    ];

    #[inline(always)]
    pub fn abs(self) -> Self {
        simd::simd_abs(Scalar::new(), self)
    }

    #[inline(always)]
    pub fn recip(self) -> Self {
        simd::simd_div(Scalar::new(), Self(1.0, 0.0), self)
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

    #[inline]
    pub fn exp(self) -> Self {
        const K: f64 = 512.0;
        const INV_K: f64 = 1.0 / K;

        let a = self;
        if a.0 <= -709.0 {
            return Self::ZERO;
        }
        if a.0 >= 709.0 {
            return Self::INFINITY;
        }
        if a.0 == 0.0 {
            return Self(1.0, 0.0);
        }

        let m = (a.0 / Self::LN_2.0 + 0.5).floor();
        let r = a - Self::LN_2 * Self(m, 0.0);
        let r = Self(r.0 * INV_K, r.1 * INV_K);

        let p = r * r;
        let mut s = r + Self(p.0 * 0.5, p.1 * 0.5);
        let mut p = p * r;

        let mut t = p * Self::INV_FACT[0];
        let mut i = 0;
        loop {
            s += t;
            p *= r;
            i += 1;

            t = p * Self::INV_FACT[i];
            if !(t.0.abs() > INV_K * Self::EPSILON.0 && i < 5) {
                break;
            }
        }

        s += t;

        s = Self(s.0 * 2.0, s.1 * 2.0) + s * s;
        s = Self(s.0 * 2.0, s.1 * 2.0) + s * s;
        s = Self(s.0 * 2.0, s.1 * 2.0) + s * s;
        s = Self(s.0 * 2.0, s.1 * 2.0) + s * s;
        s = Self(s.0 * 2.0, s.1 * 2.0) + s * s;
        s = Self(s.0 * 2.0, s.1 * 2.0) + s * s;
        s = Self(s.0 * 2.0, s.1 * 2.0) + s * s;
        s = Self(s.0 * 2.0, s.1 * 2.0) + s * s;
        s = Self(s.0 * 2.0, s.1 * 2.0) + s * s;

        s += Self(1.0, 0.0);

        let factor = 2.0f64.powi(m as i32);
        Self(s.0 * factor, s.1 * factor)
    }

    #[inline]
    pub fn ln(self) -> Self {
        if self == Self(1.0, 0.0) {
            Self::ZERO
        } else if self <= Self::ZERO {
            Self::NAN
        } else {
            let a = self;
            let x = Self(a.0.log(core::f64::consts::E), 0.0);

            x + a * (-x).exp() - Self(1.0, 0.0)
        }
    }

    #[inline]
    pub fn log2(self) -> Self {
        self.ln() / Self::LN_2
    }
}

mod faer_impl {
    use super::*;
    use faer_core::{ComplexField, Conjugate, Entity, RealField};

    type SimdGroup<E, S> = <E as Entity>::Group<<E as Entity>::SimdUnit<S>>;

    unsafe impl Entity for DoubleF64 {
        type Unit = f64;
        type Index = u64;

        type SimdUnit<S: Simd> = S::f64s;
        type SimdMask<S: Simd> = S::m64s;
        type SimdIndex<S: Simd> = S::u64s;

        type Group<T> = Double<T>;
        type GroupCopy<T: Copy> = Double<T>;
        type GroupThreadSafe<T: Send + Sync> = Double<T>;
        type Iter<I: Iterator> = Double<I>;

        const N_COMPONENTS: usize = 2;
        const HAS_SIMD: bool = true;
        const UNIT: Self::GroupCopy<()> = Double((), ());

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
            Double(&group.0, &group.1)
        }

        #[inline(always)]
        fn as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T> {
            Double(&mut group.0, &mut group.1)
        }

        #[inline(always)]
        fn map<T, U>(group: Self::Group<T>, mut f: impl FnMut(T) -> U) -> Self::Group<U> {
            Double(f(group.0), f(group.1))
        }

        #[inline(always)]
        fn zip<T, U>(first: Self::Group<T>, second: Self::Group<U>) -> Self::Group<(T, U)> {
            Double((first.0, second.0), (first.1, second.1))
        }

        #[inline(always)]
        fn unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>) {
            (
                Double(zipped.0 .0, zipped.1 .0),
                Double(zipped.0 .1, zipped.1 .1),
            )
        }

        #[inline(always)]
        fn map_with_context<Ctx, T, U>(
            ctx: Ctx,
            group: Self::Group<T>,
            mut f: impl FnMut(Ctx, T) -> (Ctx, U),
        ) -> (Ctx, Self::Group<U>) {
            let (ctx, x0) = f(ctx, group.0);
            let (ctx, x1) = f(ctx, group.1);
            (ctx, Double(x0, x1))
        }

        #[inline(always)]
        fn into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter> {
            Double(iter.0.into_iter(), iter.1.into_iter())
        }
    }

    unsafe impl Conjugate for DoubleF64 {
        type Conj = DoubleF64;
        type Canonical = DoubleF64;
        #[inline(always)]
        fn canonicalize(self) -> Self::Canonical {
            self
        }
    }

    impl RealField for DoubleF64 {
        #[inline(always)]
        fn sqrt(&self) -> Self {
            (*self).sqrt()
        }

        #[inline(always)]
        fn div(&self, rhs: &Self) -> Self {
            *self / *rhs
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
            simd::simd_less_than(simd, a, b)
        }

        #[inline(always)]
        fn simd_less_than_or_equal<S: Simd>(
            simd: S,
            a: SimdGroup<Self, S>,
            b: SimdGroup<Self, S>,
        ) -> Self::SimdMask<S> {
            simd::simd_less_than_or_equal(simd, a, b)
        }

        #[inline(always)]
        fn simd_greater_than<S: Simd>(
            simd: S,
            a: SimdGroup<Self, S>,
            b: SimdGroup<Self, S>,
        ) -> Self::SimdMask<S> {
            simd::simd_greater_than(simd, a, b)
        }

        #[inline(always)]
        fn simd_greater_than_or_equal<S: Simd>(
            simd: S,
            a: SimdGroup<Self, S>,
            b: SimdGroup<Self, S>,
        ) -> Self::SimdMask<S> {
            simd::simd_greater_than_or_equal(simd, a, b)
        }

        #[inline(always)]
        fn simd_select<S: Simd>(
            simd: S,
            mask: Self::SimdMask<S>,
            if_true: SimdGroup<Self, S>,
            if_false: SimdGroup<Self, S>,
        ) -> SimdGroup<Self, S> {
            simd::simd_select(simd, mask, if_true, if_false)
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

    impl ComplexField for DoubleF64 {
        type Real = DoubleF64;

        #[inline(always)]
        fn from_f64(value: f64) -> Self {
            Self(value, 0.0)
        }

        #[inline(always)]
        fn add(&self, rhs: &Self) -> Self {
            *self + *rhs
        }

        #[inline(always)]
        fn sub(&self, rhs: &Self) -> Self {
            *self - *rhs
        }

        #[inline(always)]
        fn mul(&self, rhs: &Self) -> Self {
            *self * *rhs
        }

        #[inline(always)]
        fn neg(&self) -> Self {
            -*self
        }

        #[inline(always)]
        fn inv(&self) -> Self {
            (*self).recip()
        }

        #[inline(always)]
        fn conj(&self) -> Self {
            *self
        }

        #[inline(always)]
        fn scale_real(&self, rhs: &Self::Real) -> Self {
            *self * *rhs
        }

        #[inline(always)]
        fn scale_power_of_two(&self, rhs: &Self::Real) -> Self {
            Self(self.0 * rhs.0, self.1 * rhs.0)
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
            *self * *self
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
            Self::ZERO
        }

        #[inline(always)]
        fn zero() -> Self {
            Self::ZERO
        }

        #[inline(always)]
        fn one() -> Self {
            Self(1.0, 0.0)
        }

        #[inline(always)]
        fn slice_as_simd<S: faer_core::pulp::Simd>(
            slice: &[Self::Unit],
        ) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
            S::f64s_as_simd(slice)
        }

        #[inline(always)]
        fn slice_as_mut_simd<S: faer_core::pulp::Simd>(
            slice: &mut [Self::Unit],
        ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
            S::f64s_as_mut_simd(slice)
        }

        #[inline(always)]
        fn partial_load_unit<S: faer_core::pulp::Simd>(
            simd: S,
            slice: &[Self::Unit],
        ) -> Self::SimdUnit<S> {
            simd.f64s_partial_load(slice)
        }

        #[inline(always)]
        fn partial_store_unit<S: faer_core::pulp::Simd>(
            simd: S,
            slice: &mut [Self::Unit],
            values: Self::SimdUnit<S>,
        ) {
            simd.f64s_partial_store(slice, values)
        }

        #[inline(always)]
        fn partial_load_last_unit<S: faer_core::pulp::Simd>(
            simd: S,
            slice: &[Self::Unit],
        ) -> Self::SimdUnit<S> {
            simd.f64s_partial_load_last(slice)
        }

        #[inline(always)]
        fn partial_store_last_unit<S: faer_core::pulp::Simd>(
            simd: S,
            slice: &mut [Self::Unit],
            values: Self::SimdUnit<S>,
        ) {
            simd.f64s_partial_store_last(slice, values)
        }

        #[inline(always)]
        fn simd_splat_unit<S: faer_core::pulp::Simd>(
            simd: S,
            unit: Self::Unit,
        ) -> Self::SimdUnit<S> {
            simd.f64s_splat(unit)
        }

        #[inline(always)]
        fn simd_neg<S: faer_core::pulp::Simd>(
            simd: S,
            values: SimdGroup<Self, S>,
        ) -> SimdGroup<Self, S> {
            simd::simd_neg(simd, values)
        }

        #[inline(always)]
        fn simd_conj<S: faer_core::pulp::Simd>(
            simd: S,
            values: SimdGroup<Self, S>,
        ) -> SimdGroup<Self, S> {
            let _ = simd;
            values
        }

        #[inline(always)]
        fn simd_add<S: faer_core::pulp::Simd>(
            simd: S,
            lhs: SimdGroup<Self, S>,
            rhs: SimdGroup<Self, S>,
        ) -> SimdGroup<Self, S> {
            simd::simd_add(simd, lhs, rhs)
        }

        #[inline(always)]
        fn simd_sub<S: faer_core::pulp::Simd>(
            simd: S,
            lhs: SimdGroup<Self, S>,
            rhs: SimdGroup<Self, S>,
        ) -> SimdGroup<Self, S> {
            simd::simd_sub(simd, lhs, rhs)
        }

        #[inline(always)]
        fn simd_mul<S: faer_core::pulp::Simd>(
            simd: S,
            lhs: SimdGroup<Self, S>,
            rhs: SimdGroup<Self, S>,
        ) -> SimdGroup<Self, S> {
            simd::simd_mul(simd, lhs, rhs)
        }

        #[inline(always)]
        fn simd_conj_mul<S: faer_core::pulp::Simd>(
            simd: S,
            lhs: SimdGroup<Self, S>,
            rhs: SimdGroup<Self, S>,
        ) -> SimdGroup<Self, S> {
            simd::simd_mul(simd, lhs, rhs)
        }

        #[inline(always)]
        fn simd_mul_adde<S: faer_core::pulp::Simd>(
            simd: S,
            lhs: SimdGroup<Self, S>,
            rhs: SimdGroup<Self, S>,
            acc: SimdGroup<Self, S>,
        ) -> SimdGroup<Self, S> {
            simd::simd_add(simd, acc, simd::simd_mul(simd, lhs, rhs))
        }

        #[inline(always)]
        fn simd_conj_mul_adde<S: faer_core::pulp::Simd>(
            simd: S,
            lhs: SimdGroup<Self, S>,
            rhs: SimdGroup<Self, S>,
            acc: SimdGroup<Self, S>,
        ) -> SimdGroup<Self, S> {
            simd::simd_add(simd, acc, simd::simd_mul(simd, lhs, rhs))
        }

        #[inline(always)]
        fn simd_score<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
            simd::simd_abs(simd, values)
        }
    }
}
