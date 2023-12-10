
// 64-bit complex floating point type. See the module-level documentation for more details.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct c64 {
    pub re: f64,
    pub im: f64,
}

impl c64 {
    #[inline(always)]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub fn to_num_complex(self) -> Complex<f64> {
        self.into()
    }

    #[inline(always)]
    pub fn abs(self) -> f64 {
        self.faer_abs()
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
        self * <c64 as ComplexField>::faer_inv(rhs)
    }
}
impl core::ops::Div for c64 {
    type Output = c64;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        self * <Self as ComplexField>::faer_inv(rhs)
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
unsafe impl bytemuck::Zeroable for c64 {}
unsafe impl bytemuck::Pod for c64 {}
impl Debug for c64 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.re.fmt(f)?;
        f.write_str(" + ")?;
        self.im.fmt(f)?;
        f.write_str(" * I")
    }
}

impl ComplexField for c64 {
    type Real = f64;
    type Simd = pulp::Arch;
    type ScalarSimd = NoSimd;
    type PortableSimd = pulp::Arch;

    #[inline(always)]
    fn faer_from_f64(value: f64) -> Self {
        Self {
            re: value as _,
            im: 0.0,
        }
    }

    #[inline(always)]
    fn faer_add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline(always)]
    fn faer_sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    #[inline(always)]
    fn faer_mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    #[inline(always)]
    fn faer_neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn faer_inv(self) -> Self {
        self.to_num_complex().faer_inv().into()
    }

    #[inline(always)]
    fn faer_conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    fn faer_sqrt(self) -> Self {
        self.to_num_complex().faer_sqrt().into()
    }

    #[inline(always)]
    fn faer_scale_real(self, rhs: Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn faer_scale_power_of_two(self, rhs: Self::Real) -> Self {
        Self {
            re: rhs * self.re,
            im: rhs * self.im,
        }
    }

    #[inline(always)]
    fn faer_score(self) -> Self::Real {
        self.faer_abs2()
    }

    #[inline(always)]
    fn faer_abs(self) -> Self::Real {
        self.to_num_complex().faer_abs().into()
    }

    #[inline(always)]
    fn faer_abs2(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    fn faer_nan() -> Self {
        Self {
            re: Self::Real::NAN,
            im: Self::Real::NAN,
        }
    }

    #[inline(always)]
    fn faer_from_real(real: Self::Real) -> Self {
        Self { re: real, im: 0.0 }
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
        Self { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    fn faer_one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    #[inline(always)]
    fn faer_slice_as_simd<S: Simd>(slice: &[Self::Unit]) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
        let (head, tail) = S::c64s_as_simd(bytemuck::cast_slice(slice));
        (bytemuck::cast_slice(head), bytemuck::cast_slice(tail))
    }

    #[inline(always)]
    fn faer_slice_as_simd_mut<S: Simd>(
        slice: &mut [Self::Unit],
    ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
        let (head, tail) = S::c64s_as_mut_simd(bytemuck::cast_slice_mut(slice));
        (
            bytemuck::cast_slice_mut(head),
            bytemuck::cast_slice_mut(tail),
        )
    }

    #[inline(always)]
    fn faer_partial_load_last_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.c64s_partial_load_last(bytemuck::cast_slice(slice))
    }

    #[inline(always)]
    fn faer_partial_store_last_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.c64s_partial_store_last(bytemuck::cast_slice_mut(slice), values)
    }

    #[inline(always)]
    fn faer_partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        simd.c64s_partial_load(bytemuck::cast_slice(slice))
    }

    #[inline(always)]
    fn faer_partial_store_unit<S: Simd>(
        simd: S,
        slice: &mut [Self::Unit],
        values: Self::SimdUnit<S>,
    ) {
        simd.c64s_partial_store(bytemuck::cast_slice_mut(slice), values)
    }

    #[inline(always)]
    fn faer_simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
        simd.c64s_splat(pulp::cast(unit))
    }

    #[inline(always)]
    fn faer_simd_neg<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> SimdGroupFor<Self, S> {
        simd.c64s_neg(values)
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
        simd.c64s_add(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_sub<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_sub(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_mul(lhs, rhs)
    }
    #[inline(always)]
    fn faer_simd_scale_real<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self::Real, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        if coe::is_same::<pulp::Scalar, S>() {
            let lhs: f64 = bytemuck::cast(lhs);
            let rhs: num_complex::Complex64 = bytemuck::cast(rhs);
            bytemuck::cast(lhs * rhs)
        } else {
            bytemuck::cast(simd.f64s_mul(lhs, bytemuck::cast(rhs)))
        }
    }
    #[inline(always)]
    fn faer_simd_conj_mul<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_conj_mul(lhs, rhs)
    }

    #[inline(always)]
    fn faer_simd_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn faer_simd_conj_mul_adde<S: Simd>(
        simd: S,
        lhs: SimdGroupFor<Self, S>,
        rhs: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_conj_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn faer_simd_reduce_add<S: Simd>(simd: S, values: SimdGroupFor<Self, S>) -> Self {
        pulp::cast(simd.c64s_reduce_sum(values))
    }

    #[inline(always)]
    fn faer_simd_abs2_adde<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
        acc: SimdGroupFor<Self::Real, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        let _ = (simd, values, acc);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }
    #[inline(always)]
    fn faer_simd_abs2<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        let _ = (simd, values);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }
    #[inline(always)]
    fn faer_simd_score<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
    ) -> SimdGroupFor<Self::Real, S> {
        let _ = (simd, values);
        unimplemented!("c32/c64 require special treatment when converted to their real counterparts in simd kernels");
    }

    #[inline(always)]
    fn faer_simd_scalar_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        cast(simd.c64_scalar_mul(cast(lhs), cast(rhs)))
    }
    #[inline(always)]
    fn faer_simd_scalar_conj_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
        cast(simd.c64_scalar_conj_mul(cast(lhs), cast(rhs)))
    }
    #[inline(always)]
    fn faer_simd_scalar_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        cast(simd.c64_scalar_mul_add_e(cast(lhs), cast(rhs), cast(acc)))
    }
    #[inline(always)]
    fn faer_simd_scalar_conj_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
        cast(simd.c64_scalar_conj_mul_add_e(cast(lhs), cast(rhs), cast(acc)))
    }

    #[inline(always)]
    fn faer_align_offset<S: Simd>(
        simd: S,
        ptr: *const UnitFor<Self>,
        len: usize,
    ) -> pulp::Offset<SimdMaskFor<Self, S>> {
        simd.c64s_align_offset(ptr as _, len)
    }

    #[inline(always)]
    fn faer_slice_as_aligned_simd<S: Simd>(
        simd: S,
        slice: &[UnitFor<Self>],
        offset: pulp::Offset<SimdMaskFor<Self, S>>,
    ) -> (
        Self::PrefixUnit<'_, S>,
        &[SimdUnitFor<Self, S>],
        Self::SuffixUnit<'_, S>,
    ) {
        simd.c64s_as_aligned_simd(bytemuck::cast_slice(slice), offset)
    }

    #[inline(always)]
    fn faer_slice_as_aligned_simd_mut<S: Simd>(
        simd: S,
        slice: &mut [UnitFor<Self>],
        offset: pulp::Offset<SimdMaskFor<Self, S>>,
    ) -> (
        Self::PrefixMutUnit<'_, S>,
        &mut [SimdUnitFor<Self, S>],
        Self::SuffixMutUnit<'_, S>,
    ) {
        simd.c64s_as_aligned_mut_simd(bytemuck::cast_slice_mut(slice), offset)
    }
    #[inline(always)]
    fn faer_simd_rotate_left<S: Simd>(
        simd: S,
        values: SimdGroupFor<Self, S>,
        amount: usize,
    ) -> SimdGroupFor<Self, S> {
        simd.c64s_rotate_left(values, amount)
    }
}

unsafe impl Entity for c64 {
    type Unit = Self;
    type Index = u64;
    type SimdUnit<S: Simd> = S::c64s;
    type SimdMask<S: Simd> = S::m64s;
    type SimdIndex<S: Simd> = S::u64s;
    type Group = IdentityGroup;
    type Iter<I: Iterator> = I;

    type PrefixUnit<'a, S: Simd> = pulp::Prefix<'a, num_complex::Complex64, S, S::m64s>;
    type SuffixUnit<'a, S: Simd> = pulp::Suffix<'a, num_complex::Complex64, S, S::m64s>;
    type PrefixMutUnit<'a, S: Simd> = pulp::PrefixMut<'a, num_complex::Complex64, S, S::m64s>;
    type SuffixMutUnit<'a, S: Simd> = pulp::SuffixMut<'a, num_complex::Complex64, S, S::m64s>;

    const N_COMPONENTS: usize = 1;
    const UNIT: GroupCopyFor<Self, ()> = ();

    #[inline(always)]
    fn faer_first<T>(group: GroupFor<Self, T>) -> T {
        group
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
        group
    }

    #[inline(always)]
    fn faer_as_mut<T>(group: &mut GroupFor<Self, T>) -> GroupFor<Self, &mut T> {
        group
    }

    #[inline(always)]
    fn faer_map_impl<T, U>(
        group: GroupFor<Self, T>,
        f: &mut impl FnMut(T) -> U,
    ) -> GroupFor<Self, U> {
        (*f)(group)
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
unsafe impl Conjugate for c64 {
    type Conj = c64conj;
    type Canonical = c64;

    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        self
    }
}