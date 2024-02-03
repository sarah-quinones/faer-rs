use crate::complex_native::c64_impl::c64;
use faer_entity::*;
use pulp::Simd;

// 64-bit implicitly conjugated complex floating point type.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct c64conj {
    pub re: f64,
    pub neg_im: f64,
}

unsafe impl bytemuck::Pod for c64conj {}
unsafe impl bytemuck::Zeroable for c64conj {}

impl core::fmt::Debug for c64conj {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.re.fmt(f)?;
        let im_abs = self.neg_im.faer_abs();
        if self.neg_im.is_sign_positive() {
            f.write_str(" - ")?;
            im_abs.fmt(f)?;
        } else {
            f.write_str(" + ")?;
            im_abs.fmt(f)?;
        }
        f.write_str(" * I")
    }
}

impl core::fmt::Display for c64conj {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        <Self as core::fmt::Debug>::fmt(self, f)
    }
}

unsafe impl Entity for c64conj {
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
