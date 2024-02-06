use crate::complex_native::c32_impl::c32;
use faer_entity::*;
use pulp::Simd;

/// 32-bit implicitly conjugated complex floating point type.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct c32conj {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub neg_im: f32,
}

unsafe impl bytemuck::Zeroable for c32conj {}
unsafe impl bytemuck::Pod for c32conj {}

impl core::fmt::Debug for c32conj {
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

impl core::fmt::Display for c32conj {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        <Self as core::fmt::Debug>::fmt(self, f)
    }
}

unsafe impl Entity for c32conj {
    type Unit = Self;
    type Index = u32;
    type SimdUnit<S: Simd> = S::c32s;
    type SimdMask<S: Simd> = S::m32s;
    type SimdIndex<S: Simd> = S::u32s;
    type Group = IdentityGroup;
    type Iter<I: Iterator> = I;

    type PrefixUnit<'a, S: Simd> = pulp::Prefix<'a, num_complex::Complex32, S, S::m32s>;
    type SuffixUnit<'a, S: Simd> = pulp::Suffix<'a, num_complex::Complex32, S, S::m32s>;
    type PrefixMutUnit<'a, S: Simd> = pulp::PrefixMut<'a, num_complex::Complex32, S, S::m32s>;
    type SuffixMutUnit<'a, S: Simd> = pulp::SuffixMut<'a, num_complex::Complex32, S, S::m32s>;

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
