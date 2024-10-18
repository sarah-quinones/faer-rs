use bytemuck::Pod;
use core::{cmp::Ordering, fmt::Debug, ptr::addr_of_mut};
use num_complex::Complex;
use pulp::Simd;
use reborrow::*;

#[faer_macros::math]
fn abs_impl<C: RealContainer, T: RealField<C>>(
    ctx: &T::MathCtx,
    re: C::Of<&T>,
    im: C::Of<&T>,
) -> C::Of<T> {
    help!(C);
    let ctx = Ctx::<C, T>::new(ctx);

    let small = math.sqrt_min_positive();
    let big = math.sqrt_max_positive();
    let one = math.one();
    let re_abs = math.abs(re);
    let im_abs = math.abs(im);

    if math(re_abs > big || im_abs > big) {
        math(sqrt(abs2(re * small) + abs2(im * small)) * big)
    } else if math(re_abs > one || im_abs > one) {
        math(sqrt(abs2(re) + abs2(im)))
    } else {
        math(sqrt(abs2(re * big) + abs2(im * big)) * small)
    }
}

#[faer_macros::math]
fn recip_impl<C: RealContainer, T: RealField<C>>(
    ctx: &T::MathCtx,
    re: C::Of<&T>,
    im: C::Of<&T>,
) -> (C::Of<T>, C::Of<T>) {
    help!(C);
    let ctx = Ctx::<C, T>::new(ctx);
    if math.is_nan(re) || math.is_nan(im) {
        return (math.nan(), math.nan());
    }
    if math.is_zero(re) && math.is_zero(im) {
        return (math.infinity(), math.infinity());
    }
    if math(!is_finite(re) || !is_finite(im)) {
        return (math.zero(), math.zero());
    }

    let small = math.sqrt_min_positive();
    let big = math.sqrt_max_positive();
    let one = math.one();
    let re_abs = math.abs(re);
    let im_abs = math.abs(im);

    if math(re_abs > big || im_abs > big) {
        let re = math(re * small);
        let im = math(im * small);
        let inv = math(recip(abs2(re) + abs2(im)));
        (math((re * inv) * small), math((-im * inv) * small))
    } else if math(re_abs > one || im_abs > one) {
        let inv = math(recip(abs2(re) + abs2(im)));
        (math(re * inv), math(-im * inv))
    } else {
        let re = math(re * big);
        let im = math(im * big);
        let inv = math(recip(abs2(re) + abs2(im)));
        (math((re * inv) * big), math((-im * inv) * big))
    }
}

#[faer_macros::math]
fn sqrt_impl<C: RealContainer, T: RealField<C>>(
    ctx: &T::MathCtx,
    re: C::Of<&T>,
    im: C::Of<&T>,
) -> (C::Of<T>, C::Of<T>) {
    help!(C);

    let ctx = Ctx::<C, T>::new(ctx);
    let im_negative = math.lt_zero(im);
    let half = math.from_f64(0.5);
    let abs = abs_impl::<C, _>(&ctx.0, copy!(re), copy!(im));

    let mut sum = math(re + abs);
    if math.lt_zero(sum) {
        sum = math.zero();
    }

    let out_re = math(sqrt(mul_pow2(sum, half)));
    let mut out_im = math(sqrt(mul_pow2(re - abs, half)));
    if im_negative {
        out_im = math(-out_im);
    }
    (out_re, out_im)
}

#[macro_export]
macro_rules! new {
    ($name: ident, $ctx: ident) => {
        let mut $name = $ctx.undef();
        #[allow(unused_mut)]
        let mut $name = wrap!(as_mut!($name));
    };
}

#[repr(transparent)]
pub struct Of<C: Container, T>(pub C::Of<T>);

impl<C: Container, T: Debug> Debug for Of<C, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        unsafe { &*((&self.0) as *const C::Of<T> as *const C::OfDebug<T>) }.fmt(f)
    }
}

impl<C: Container, T> Of<C, T> {
    #[doc(hidden)]
    pub fn __at<I: Copy>(&self, idx: I) -> Of<C, &T::Output>
    where
        T: core::ops::Index<I>,
    {
        help!(C);
        Of(map!(as_ref!(self.0), ptr, &ptr[idx]))
    }

    #[doc(hidden)]
    pub fn __at_mut<I: Copy>(&mut self, idx: I) -> Of<C, &mut T::Output>
    where
        T: core::ops::IndexMut<I>,
    {
        help!(C);
        Of(map!(as_mut!(self.0), ptr, &mut ptr[idx]))
    }
}

impl<'short, C: Container, T: Reborrow<'short>> Reborrow<'short> for Of<C, T> {
    type Target = Of<C, T::Target>;
    #[inline]
    fn rb(&'short self) -> Self::Target {
        help!(C);
        Of(rb!(self.0))
    }
}
impl<'short, C: Container, T: ReborrowMut<'short>> ReborrowMut<'short> for Of<C, T> {
    type Target = Of<C, T::Target>;
    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        help!(C);
        Of(rb_mut!(self.0))
    }
}

impl<C: Container, T: IntoConst> IntoConst for Of<C, T> {
    type Target = Of<C, T::Target>;
    #[inline]
    fn into_const(self) -> Self::Target {
        Of(utils::into_const::<C, _>(self.0))
    }
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

#[macro_export]
macro_rules! help {
    ($C: ty) => {
        #[allow(unused_macros)]
        macro_rules! wrap {
            ($place: expr) => {
                $crate::Of::<$C, _>($place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! debug {
            ($place: expr) => {{
                std::eprintln!(
                    "[{}:{}:{}] {} = {:#?}",
                    std::file!(),
                    std::line!(),
                    std::column!(),
                    std::stringify!($place),
                    wrap!(as_ref!($place))
                );
            }};
        }

        #[allow(unused_macros)]
        macro_rules! send {
            ($place: expr) => {
                $crate::utils::send::<$C, _>($place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! unsend {
            ($place: expr) => {
                $crate::utils::unsend::<$C, _>($place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! sync {
            ($place: expr) => {
                $crate::utils::sync::<$C, _>($place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! unsync {
            ($place: expr) => {
                $crate::utils::unsync::<$C, _>($place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! as_ref {
            ($place: expr) => {
                $crate::utils::as_ref::<$C, _>(&$place)
            };
        }
        #[allow(unused_macros)]
        macro_rules! as_mut {
            ($place: expr) => {
                $crate::utils::as_mut::<$C, _>(&mut $place)
            };
        }
        #[allow(unused_macros)]
        macro_rules! by_ref {
            ($place: expr) => {
                $crate::utils::by_ref::<$C, _>(&$place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! copy {
            ($place: expr) => {
                $crate::utils::copy::<$C, _>(&$place)
            };
        }
        #[allow(unused_macros)]
        macro_rules! map {
            ($value: expr, $pat: pat, $f: expr) => {
                $crate::utils::map::<$C, _, _>(
                    $value,
                    #[inline(always)]
                    |$pat| $f,
                )
            };
        }
        #[allow(unused_macros)]
        macro_rules! map_move {
            ($value: expr, $pat: pat, $f: expr) => {
                $crate::utils::map::<$C, _, _>(
                    $value,
                    #[inline(always)]
                    move |$pat| $f,
                )
            };
        }

        #[allow(unused_macros)]
        macro_rules! zip {
            ($a: expr, $b: expr) => {
                <$C as $crate::Container>::zip($a, $b)
            };
        }
        #[allow(unused_macros)]
        macro_rules! unzip {
            ($ab: expr) => {
                <$C as $crate::Container>::unzip($ab)
            };
        }
        #[allow(unused_macros)]
        macro_rules! rb {
            ($place: expr) => {
                $crate::utils::rb::<$C, _>(&$place)
            };
        }
        #[allow(unused_macros)]
        macro_rules! rb_mut {
            ($place: expr) => {
                $crate::utils::rb_mut::<$C, _>(&mut $place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! write1 {
            ($mat: ident[$idx: expr] = $val: expr) => {{
                let __val = $val;
                $crate::utils::write::<$C, _>(&mut $mat.rb_mut().__at_mut($idx), __val)
            }};
            ($place: expr, $val: expr) => {{
                let __val = $val;
                $crate::utils::write::<$C, _>(&mut $place, __val)
            }};
        }
    };
}

#[macro_export]
macro_rules! help2 {
    ($C: ty) => {
        #[allow(unused_macros)]
        macro_rules! wrap2 {
            ($place: expr) => {
                $crate::Of::<$C, _>($place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! debug2 {
            ($place: expr) => {{
                std::eprintln!(
                    "[{}:{}:{}] {} = {:#?}",
                    std::file!(),
                    std::line!(),
                    std::column!(),
                    std::stringify!($place),
                    wrap2!(as_ref2!($place))
                );
            }};
        }

        #[allow(unused_macros)]
        macro_rules! send2 {
            ($place: expr) => {
                $crate::utils::send::<$C, _>($place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! unsend2 {
            ($place: expr) => {
                $crate::utils::unsend::<$C, _>($place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! sync2 {
            ($place: expr) => {
                $crate::utils::sync::<$C, _>($place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! unsync2 {
            ($place: expr) => {
                $crate::utils::unsync::<$C, _>($place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! by_ref2 {
            ($place: expr) => {
                $crate::utils::by_ref::<$C, _>(&$place)
            };
        }
        #[allow(unused_macros)]
        macro_rules! as_ref2 {
            ($place: expr) => {
                $crate::utils::as_ref::<$C, _>(&$place)
            };
        }
        #[allow(unused_macros)]
        macro_rules! as_mut2 {
            ($place: expr) => {
                $crate::utils::as_mut::<$C, _>(&mut $place)
            };
        }
        #[allow(unused_macros)]
        macro_rules! copy2 {
            ($place: expr) => {
                $crate::utils::copy::<$C, _>(&$place)
            };
        }
        #[allow(unused_macros)]
        macro_rules! map2 {
            ($value: expr, $pat: pat, $f: expr) => {
                $crate::utils::map::<$C, _, _>(
                    $value,
                    #[inline(always)]
                    |$pat| $f,
                )
            };
        }

        #[allow(unused_macros)]
        macro_rules! zip2 {
            ($a: expr, $b: expr) => {
                <$C as $crate::ComplexContainer>::zip($a, $b)
            };
        }
        #[allow(unused_macros)]
        macro_rules! unzip2 {
            ($ab: expr) => {
                <$C as $crate::ComplexContainer>::unzip($ab)
            };
        }
        #[allow(unused_macros)]
        macro_rules! rb2 {
            ($place: expr) => {
                $crate::utils::rb::<$C, _>(&$place)
            };
        }
        #[allow(unused_macros)]
        macro_rules! rb_mut2 {
            ($place: expr) => {
                $crate::utils::rb_mut::<$C, _>(&mut $place)
            };
        }

        #[allow(unused_macros)]
        macro_rules! write2 {
            ($mat: ident[$idx: expr] = $val: expr) => {{
                let __val = $val;
                $crate::utils::write::<$C, _>(&mut $mat.rb_mut().__at_mut($idx), __val)
            }};
            ($place: expr, $val: expr) => {{
                let __val = $val;
                $crate::utils::write::<$C, _>(&mut $place, __val)
            }};
        }
    };
}

pub mod utils {
    use super::*;

    #[inline(always)]
    pub fn unsend<C: Container, T: Send>(value: C::OfSend<T>) -> C::Of<T> {
        unsafe { core::mem::transmute_copy(&core::mem::ManuallyDrop::new(value)) }
    }

    #[inline(always)]
    pub fn send<C: Container, T: Send>(value: C::Of<T>) -> C::OfSend<T> {
        unsafe { core::mem::transmute_copy(&core::mem::ManuallyDrop::new(value)) }
    }

    #[inline(always)]
    pub fn unsync<C: Container, T: Sync + Copy>(value: C::OfSync<T>) -> C::Of<T> {
        unsafe { core::mem::transmute_copy(&core::mem::ManuallyDrop::new(value)) }
    }

    #[inline(always)]
    pub fn sync<C: Container, T: Sync + Copy>(value: C::Of<T>) -> C::OfSync<T> {
        unsafe { core::mem::transmute_copy(&core::mem::ManuallyDrop::new(value)) }
    }

    #[inline(always)]
    pub fn map<C: Container, T, U>(value: C::Of<T>, f: impl FnMut(T) -> U) -> C::Of<U> {
        let mut f = f;
        C::map_impl(value, (), &mut |x, ()| (f(x), ())).0
    }

    #[inline(always)]
    pub fn copy<C: Container, T: Copy>(ptr: &C::Of<T>) -> C::Of<T> {
        unsafe { core::mem::transmute_copy(ptr) }
    }

    #[inline(always)]
    pub fn as_ref<C: Container, T>(ptr: &C::Of<T>) -> C::Of<&T> {
        C::map_impl(
            unsafe { C::as_ptr(ptr as *const C::Of<T> as *mut C::Of<T>) },
            (),
            &mut {
                #[inline(always)]
                |ptr, ()| unsafe { (&*ptr, ()) }
            },
        )
        .0
    }
    #[inline(always)]
    pub fn as_mut<C: Container, T>(ptr: &mut C::Of<T>) -> C::Of<&mut T> {
        C::map_impl(
            unsafe { C::as_ptr(ptr as *mut C::Of<T> as *mut C::Of<T>) },
            (),
            &mut {
                #[inline(always)]
                |ptr, ()| unsafe { (&mut *ptr, ()) }
            },
        )
        .0
    }

    #[inline(always)]
    pub fn by_ref<'a, C: Container, T>(ptr: &'a C::Of<impl 'a + ByRef<T>>) -> C::Of<&'a T> {
        C::map_impl(as_ref::<C, _>(ptr), (), &mut |ptr, ()| (ptr.by_ref(), ())).0
    }

    #[inline(always)]
    pub fn rb<'short, C: Container, T: Reborrow<'short>>(
        ptr: &'short C::Of<T>,
    ) -> C::Of<T::Target> {
        map::<C, _, _>(
            as_ref::<C, _>(ptr),
            #[inline(always)]
            |ptr| ptr.rb(),
        )
    }

    #[inline(always)]
    pub fn rb_mut<'short, C: Container, T: ReborrowMut<'short>>(
        ptr: &'short mut C::Of<T>,
    ) -> C::Of<T::Target> {
        map::<C, _, _>(
            as_mut::<C, _>(ptr),
            #[inline(always)]
            |ptr| ptr.rb_mut(),
        )
    }

    #[inline(always)]
    pub fn write<C: Container, T>(ptr: &mut C::Of<&mut T>, value: C::Of<T>) {
        map::<C, _, _>(
            C::zip(rb_mut::<C, _>(ptr), value),
            #[inline(always)]
            |(ptr, value)| *ptr = value,
        );
    }

    #[inline(always)]
    pub fn into_const<C: Container, T: IntoConst>(ptr: C::Of<T>) -> C::Of<T::Target> {
        map::<C, _, _>(
            ptr,
            #[inline(always)]
            |ptr| ptr.into_const(),
        )
    }
}

#[repr(transparent)]
pub struct Ctx<C: ComplexContainer, T: ComplexField<C>>(pub T::MathCtx);

impl<C: ComplexContainer, T: ComplexField<C, MathCtx: Default>> Default for Ctx<C, T> {
    fn default() -> Self {
        Ctx(Default::default())
    }
}
impl<C: ComplexContainer, T: ComplexField<C, MathCtx: Clone>> Clone for Ctx<C, T> {
    fn clone(&self) -> Self {
        Ctx(self.0.clone())
    }
}
impl<C: ComplexContainer, T: ComplexField<C, MathCtx: Copy>> Copy for Ctx<C, T> {}

impl<C: ComplexContainer, T: ComplexField<C>> core::ops::Deref for Ctx<C, T> {
    type Target = T::MathCtx;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[repr(transparent)]
pub struct SimdCtx<C: ComplexContainer, T: ComplexField<C>, S: Simd>(pub T::SimdCtx<S>);

#[repr(transparent)]
pub struct SimdCtxCopy<C: ComplexContainer, T: ComplexField<C>, S: Simd>(pub T::SimdCtx<S>);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Real<T>(pub T);

impl<C: ComplexContainer, T: ComplexField<C>> Ctx<C, T> {
    #[inline(always)]
    pub fn new(ctx: &T::MathCtx) -> &Self {
        unsafe { &*(ctx as *const T::MathCtx as *const Self) }
    }

    #[inline(always)]
    pub fn real_ctx(&self) -> &Ctx<C::Real, T::RealUnit> {
        Ctx::new(&self.0)
    }

    #[inline(always)]
    pub fn eps(&self) -> <C::Real as Container>::Of<T::RealUnit> {
        T::RealUnit::epsilon_impl(&self.0)
    }

    #[inline(always)]
    pub fn nbits(&self) -> usize {
        T::RealUnit::nbits_impl(&self.0)
    }

    #[inline(always)]
    pub fn min_positive(&self) -> <C::Real as Container>::Of<T::RealUnit> {
        T::RealUnit::min_positive_impl(&self.0)
    }
    #[inline(always)]
    pub fn max_positive(&self) -> <C::Real as Container>::Of<T::RealUnit> {
        T::RealUnit::max_positive_impl(&self.0)
    }
    #[inline(always)]
    pub fn sqrt_min_positive(&self) -> <C::Real as Container>::Of<T::RealUnit> {
        T::RealUnit::sqrt_min_positive_impl(&self.0)
    }
    #[inline(always)]
    pub fn sqrt_max_positive(&self) -> <C::Real as Container>::Of<T::RealUnit> {
        T::RealUnit::sqrt_max_positive_impl(&self.0)
    }

    #[inline(always)]
    pub fn zero(&self) -> C::Of<T> {
        T::zero_impl(&self.0)
    }
    #[inline(always)]
    pub fn one(&self) -> C::Of<T> {
        T::one_impl(&self.0)
    }
    #[inline(always)]
    pub fn nan(&self) -> C::Of<T> {
        T::nan_impl(&self.0)
    }
    #[inline(always)]
    pub fn infinity(&self) -> C::Of<T> {
        T::infinity_impl(&self.0)
    }

    #[inline(always)]
    pub fn real(&self, value: &C::Of<impl ByRef<T>>) -> <C::Real as Container>::Of<T::RealUnit> {
        help!(C);
        T::real_part_impl(&self.0, by_ref!(value))
    }
    #[inline(always)]
    pub fn imag(&self, value: &C::Of<impl ByRef<T>>) -> <C::Real as Container>::Of<T::RealUnit> {
        help!(C);
        T::imag_part_impl(&self.0, by_ref!(value))
    }
    #[inline(always)]
    pub fn neg(&self, value: &C::Of<impl ByRef<T>>) -> C::Of<T> {
        help!(C);
        T::neg_impl(&self.0, by_ref!(value))
    }
    #[inline(always)]
    pub fn copy(&self, value: &C::Of<impl ByRef<T>>) -> C::Of<T> {
        help!(C);
        T::copy_impl(&self.0, by_ref!(value))
    }

    #[inline(always)]
    pub fn conj(&self, value: &C::Of<impl ByRef<T>>) -> C::Of<T> {
        help!(C);
        T::conj_impl(&self.0, by_ref!(value))
    }

    #[inline(always)]
    pub fn id<'a>(&'a self, value: &'a C::Of<impl 'a + ByRef<T>>) -> C::Of<&'a T> {
        help!(C);
        by_ref!(value)
    }

    #[inline(always)]
    pub fn add(&self, lhs: &C::Of<impl ByRef<T>>, rhs: &C::Of<impl ByRef<T>>) -> C::Of<T> {
        help!(C);
        T::add_impl(&self.0, by_ref!(lhs), by_ref!(rhs))
    }
    #[inline(always)]
    pub fn sub(&self, lhs: &C::Of<impl ByRef<T>>, rhs: &C::Of<impl ByRef<T>>) -> C::Of<T> {
        help!(C);
        T::sub_impl(&self.0, by_ref!(lhs), by_ref!(rhs))
    }
    #[inline(always)]
    pub fn mul(&self, lhs: &C::Of<impl ByRef<T>>, rhs: &C::Of<impl ByRef<T>>) -> C::Of<T> {
        help!(C);
        T::mul_impl(&self.0, by_ref!(lhs), by_ref!(rhs))
    }
    #[inline(always)]
    pub fn div(&self, lhs: &C::Of<impl ByRef<T>>, rhs: &C::Of<impl ByRef<T>>) -> C::Of<T> {
        help!(C);
        T::div_impl(&self.0, by_ref!(lhs), by_ref!(rhs))
    }

    #[inline(always)]
    pub fn mul_real(
        &self,
        lhs: &C::Of<impl ByRef<T>>,
        rhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> C::Of<T> {
        help!(C);
        help2!(C::Real);
        T::mul_real_impl(&self.0, by_ref!(lhs), by_ref2!(rhs))
    }

    #[inline(always)]
    pub fn mul_pow2(
        &self,
        lhs: &C::Of<impl ByRef<T>>,
        rhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> C::Of<T> {
        help!(C);
        help2!(C::Real);
        T::mul_real_impl(&self.0, by_ref!(lhs), by_ref2!(rhs))
    }

    #[inline(always)]
    pub fn abs1(&self, value: &C::Of<impl ByRef<T>>) -> <C::Real as Container>::Of<T::RealUnit> {
        help!(C);
        T::abs1_impl(&self.0, by_ref!(value))
    }

    #[inline(always)]
    pub fn abs(&self, value: &C::Of<impl ByRef<T>>) -> <C::Real as Container>::Of<T::RealUnit> {
        help!(C);
        T::abs_impl(&self.0, by_ref!(value))
    }

    #[inline(always)]
    pub fn hypot(
        &self,
        lhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
        rhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> <C::Real as Container>::Of<T::RealUnit> {
        help!(C::Real);
        abs_impl::<C::Real, T::RealUnit>(&self.0, by_ref!(lhs), by_ref!(rhs))
    }

    #[inline(always)]
    pub fn abs2(&self, value: &C::Of<impl ByRef<T>>) -> <C::Real as Container>::Of<T::RealUnit> {
        help!(C);
        T::abs2_impl(&self.0, by_ref!(value))
    }

    #[inline(always)]
    pub fn cmp(
        &self,
        lhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
        rhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> Option<Ordering> {
        help!(C::Real);
        T::RealUnit::partial_cmp_impl(&self.0, by_ref!(lhs), by_ref!(rhs))
    }

    #[inline(always)]
    pub fn lt(
        &self,
        lhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
        rhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> bool {
        matches!(self.cmp(lhs, rhs), Some(Ordering::Less))
    }

    #[inline(always)]
    pub fn max(
        &self,
        lhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
        rhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> <C::Real as Container>::Of<T::RealUnit> {
        if self.gt(lhs, rhs) {
            Ctx::<C::Real, T::RealUnit>::new(&self.0).copy(lhs)
        } else {
            Ctx::<C::Real, T::RealUnit>::new(&self.0).copy(rhs)
        }
    }
    #[inline(always)]
    pub fn min(
        &self,
        lhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
        rhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> <C::Real as Container>::Of<T::RealUnit> {
        if self.lt(lhs, rhs) {
            Ctx::<C::Real, T::RealUnit>::new(&self.0).copy(lhs)
        } else {
            Ctx::<C::Real, T::RealUnit>::new(&self.0).copy(rhs)
        }
    }

    #[inline(always)]
    pub fn gt(
        &self,
        lhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
        rhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> bool {
        matches!(self.cmp(lhs, rhs), Some(Ordering::Greater))
    }
    #[inline(always)]
    pub fn le(
        &self,
        lhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
        rhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> bool {
        matches!(self.cmp(lhs, rhs), Some(Ordering::Less | Ordering::Equal))
    }
    #[inline(always)]
    pub fn ge(
        &self,
        lhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
        rhs: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> bool {
        matches!(
            self.cmp(lhs, rhs),
            Some(Ordering::Greater | Ordering::Equal)
        )
    }

    #[inline(always)]
    pub fn cmp_zero(
        &self,
        value: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> Option<Ordering> {
        help!(C::Real);
        T::RealUnit::partial_cmp_zero_impl(&self.0, by_ref!(value))
    }

    #[inline(always)]
    pub fn lt_zero(&self, value: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>) -> bool {
        matches!(self.cmp_zero(value), Some(Ordering::Less))
    }
    #[inline(always)]
    pub fn gt_zero(&self, value: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>) -> bool {
        matches!(self.cmp_zero(value), Some(Ordering::Greater))
    }
    #[inline(always)]
    pub fn le_zero(&self, value: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>) -> bool {
        matches!(self.cmp_zero(value), Some(Ordering::Less | Ordering::Equal))
    }
    #[inline(always)]
    pub fn ge_zero(&self, value: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>) -> bool {
        matches!(
            self.cmp_zero(value),
            Some(Ordering::Greater | Ordering::Equal)
        )
    }

    #[inline(always)]
    pub fn eq(&self, lhs: &C::Of<impl ByRef<T>>, rhs: &C::Of<impl ByRef<T>>) -> bool {
        help!(C);
        T::eq_impl(&self.0, by_ref!(lhs), by_ref!(rhs))
    }

    #[inline(always)]
    pub fn is_zero(&self, value: &C::Of<impl ByRef<T>>) -> bool {
        help!(C);
        T::is_zero_impl(&self.0, by_ref!(value))
    }

    #[inline(always)]
    pub fn is_nan(&self, value: &C::Of<impl ByRef<T>>) -> bool {
        help!(C);
        T::is_nan_impl(&self.0, by_ref!(value))
    }

    #[inline(always)]
    pub fn is_finite(&self, value: &C::Of<impl ByRef<T>>) -> bool {
        help!(C);
        T::is_finite_impl(&self.0, by_ref!(value))
    }

    #[inline(always)]
    pub fn sqrt(&self, value: &C::Of<impl ByRef<T>>) -> C::Of<T> {
        help!(C);
        T::sqrt_impl(&self.0, by_ref!(value))
    }
    #[inline(always)]
    pub fn recip(&self, value: &C::Of<impl ByRef<T>>) -> C::Of<T> {
        help!(C);
        T::recip_impl(&self.0, by_ref!(value))
    }

    #[inline(always)]
    pub fn from_real(
        &self,
        value: &<C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> C::Of<T> {
        help!(C::Real);
        T::from_real_impl(&self.0, by_ref!(value))
    }

    #[inline(always)]
    pub fn from_f64(&self, value: &f64) -> C::Of<T> {
        T::from_f64_impl(&self.0, *value)
    }
}

impl<C: ComplexContainer, T: ComplexField<C>, S: Simd> SimdCtx<C, T, S> {
    #[inline(always)]
    pub fn new(ctx: &T::SimdCtx<S>) -> &Self {
        unsafe { &*(ctx as *const T::SimdCtx<S> as *const Self) }
    }

    #[inline(always)]
    pub fn splat(&self, value: C::Of<&T>) -> C::Of<T::SimdVec<S>> {
        T::simd_splat(&self.0, value)
    }

    #[inline(always)]
    pub fn splat_real(
        &self,
        value: <C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> Real<C::Of<T::SimdVec<S>>> {
        help!(C::Real);
        Real(T::simd_splat_real(&self.0, by_ref!(value)))
    }

    #[inline(always)]
    pub fn add(
        &self,
        lhs: C::Of<T::SimdVec<S>>,
        rhs: C::Of<T::SimdVec<S>>,
    ) -> C::Of<T::SimdVec<S>> {
        T::simd_add(&self.0, lhs, rhs)
    }

    #[inline(always)]
    pub fn sub(
        &self,
        lhs: C::Of<T::SimdVec<S>>,
        rhs: C::Of<T::SimdVec<S>>,
    ) -> C::Of<T::SimdVec<S>> {
        T::simd_sub(&self.0, lhs, rhs)
    }

    #[inline(always)]
    pub fn neg(&self, value: C::Of<T::SimdVec<S>>) -> C::Of<T::SimdVec<S>> {
        T::simd_neg(&self.0, value)
    }
    #[inline(always)]
    pub fn conj(&self, value: C::Of<T::SimdVec<S>>) -> C::Of<T::SimdVec<S>> {
        T::simd_conj(&self.0, value)
    }
    #[inline(always)]
    pub fn abs1(&self, value: C::Of<T::SimdVec<S>>) -> Real<C::Of<T::SimdVec<S>>> {
        Real(T::simd_abs1(&self.0, value))
    }
    #[inline(always)]
    pub fn abs_max(&self, value: C::Of<T::SimdVec<S>>) -> Real<C::Of<T::SimdVec<S>>> {
        Real(T::simd_abs_max(&self.0, value))
    }

    #[inline(always)]
    pub fn mul_real(
        &self,
        lhs: C::Of<T::SimdVec<S>>,
        rhs: Real<C::Of<T::SimdVec<S>>>,
    ) -> C::Of<T::SimdVec<S>> {
        T::simd_mul_real(&self.0, lhs, rhs.0)
    }

    #[inline(always)]
    pub fn mul_pow2(
        &self,
        lhs: C::Of<T::SimdVec<S>>,
        rhs: Real<C::Of<T::SimdVec<S>>>,
    ) -> C::Of<T::SimdVec<S>> {
        T::simd_mul_pow2(&self.0, lhs, rhs.0)
    }

    #[inline(always)]
    pub fn mul(
        &self,
        lhs: C::Of<T::SimdVec<S>>,
        rhs: C::Of<T::SimdVec<S>>,
    ) -> C::Of<T::SimdVec<S>> {
        T::simd_mul(&self.0, lhs, rhs)
    }

    #[inline(always)]
    pub fn conj_mul(
        &self,
        lhs: C::Of<T::SimdVec<S>>,
        rhs: C::Of<T::SimdVec<S>>,
    ) -> C::Of<T::SimdVec<S>> {
        T::simd_conj_mul(&self.0, lhs, rhs)
    }

    #[inline(always)]
    pub fn mul_add(
        &self,
        lhs: C::Of<T::SimdVec<S>>,
        rhs: C::Of<T::SimdVec<S>>,
        acc: C::Of<T::SimdVec<S>>,
    ) -> C::Of<T::SimdVec<S>> {
        T::simd_mul_add(&self.0, lhs, rhs, acc)
    }

    #[inline(always)]
    pub fn conj_mul_add(
        &self,
        lhs: C::Of<T::SimdVec<S>>,
        rhs: C::Of<T::SimdVec<S>>,
        acc: C::Of<T::SimdVec<S>>,
    ) -> C::Of<T::SimdVec<S>> {
        T::simd_conj_mul_add(&self.0, lhs, rhs, acc)
    }

    #[inline(always)]
    pub fn abs2(&self, value: C::Of<T::SimdVec<S>>) -> Real<C::Of<T::SimdVec<S>>> {
        Real(T::simd_abs2(&self.0, value))
    }

    #[inline(always)]
    pub fn abs2_add(
        &self,
        value: C::Of<T::SimdVec<S>>,
        acc: Real<C::Of<T::SimdVec<S>>>,
    ) -> Real<C::Of<T::SimdVec<S>>> {
        Real(T::simd_abs2_add(&self.0, value, acc.0))
    }

    #[inline(always)]
    pub fn reduce_sum(&self, value: C::Of<T::SimdVec<S>>) -> C::Of<T> {
        T::simd_reduce_sum(&self.0, value)
    }

    #[inline(always)]
    pub fn reduce_max(
        &self,
        value: C::Of<T::SimdVec<S>>,
    ) -> <C::Real as Container>::Of<T::RealUnit> {
        T::simd_reduce_max(&self.0, value)
    }

    #[inline(always)]
    pub fn lt(
        &self,
        lhs: Real<C::Of<T::SimdVec<S>>>,
        rhs: Real<C::Of<T::SimdVec<S>>>,
    ) -> T::SimdMask<S> {
        T::simd_less_than(&self.0, lhs.0, rhs.0)
    }

    #[inline(always)]
    pub fn gt(
        &self,
        lhs: Real<C::Of<T::SimdVec<S>>>,
        rhs: Real<C::Of<T::SimdVec<S>>>,
    ) -> T::SimdMask<S> {
        T::simd_greater_than(&self.0, lhs.0, rhs.0)
    }

    #[inline(always)]
    pub fn le(
        &self,
        lhs: Real<C::Of<T::SimdVec<S>>>,
        rhs: Real<C::Of<T::SimdVec<S>>>,
    ) -> T::SimdMask<S> {
        T::simd_less_than_or_equal(&self.0, lhs.0, rhs.0)
    }

    #[inline(always)]
    pub fn ge(
        &self,
        lhs: Real<C::Of<T::SimdVec<S>>>,
        rhs: Real<C::Of<T::SimdVec<S>>>,
    ) -> T::SimdMask<S> {
        T::simd_greater_than_or_equal(&self.0, lhs.0, rhs.0)
    }

    #[inline(always)]
    pub fn select(
        &self,
        mask: T::SimdMask<S>,
        lhs: C::Of<T::SimdVec<S>>,
        rhs: C::Of<T::SimdVec<S>>,
    ) -> C::Of<T::SimdVec<S>> {
        T::simd_select(&self.0, mask, lhs, rhs)
    }

    #[inline(always)]
    pub fn iselect(
        &self,
        mask: T::SimdMask<S>,
        lhs: T::SimdIndex<S>,
        rhs: T::SimdIndex<S>,
    ) -> T::SimdIndex<S> {
        T::simd_index_select(&self.0, mask, lhs, rhs)
    }

    #[inline(always)]
    pub fn isplat(&self, value: T::Index) -> T::SimdIndex<S> {
        T::simd_index_splat(&self.0, value)
    }
    #[inline(always)]
    pub fn iadd(&self, lhs: T::SimdIndex<S>, rhs: T::SimdIndex<S>) -> T::SimdIndex<S> {
        T::simd_index_add(&self.0, lhs, rhs)
    }

    #[inline(always)]
    pub fn tail_mask(&self, len: usize) -> T::SimdMask<S> {
        T::simd_tail_mask(&self.0, len)
    }
    #[inline(always)]
    pub fn head_mask(&self, len: usize) -> T::SimdMask<S> {
        T::simd_head_mask(&self.0, len)
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
        ptr: C::Of<*const T::SimdVec<S>>,
    ) -> C::Of<T::SimdVec<S>> {
        T::simd_mask_load(&self.0, mask, ptr)
    }
    #[inline(always)]
    pub unsafe fn mask_store(
        &self,
        mask: T::SimdMask<S>,
        ptr: C::Of<*mut T::SimdVec<S>>,
        value: C::Of<T::SimdVec<S>>,
    ) {
        T::simd_mask_store(&self.0, mask, ptr, value)
    }

    #[inline(always)]
    pub fn load(&self, ptr: C::Of<&T::SimdVec<S>>) -> C::Of<T::SimdVec<S>> {
        T::simd_load(&self.0, ptr)
    }
    #[inline(always)]
    pub fn store(&self, ptr: C::Of<&mut T::SimdVec<S>>, value: C::Of<T::SimdVec<S>>) {
        T::simd_store(&self.0, ptr, value)
    }
}

impl<C: ComplexContainer, T: ComplexField<C>, S: Simd> SimdCtxCopy<C, T, S> {
    #[inline(always)]
    pub fn new(ctx: &T::SimdCtx<S>) -> &Self {
        unsafe { &*(ctx as *const T::SimdCtx<S> as *const Self) }
    }

    #[inline(always)]
    pub fn zero(&self) -> C::OfSimd<T::SimdVec<S>> {
        unsafe { core::mem::zeroed() }
    }

    #[inline(always)]
    pub fn splat(&self, value: C::Of<&T>) -> C::OfSimd<T::SimdVec<S>> {
        unsafe { core::mem::transmute_copy(&T::simd_splat(&self.0, value)) }
    }

    #[inline(always)]
    pub fn splat_real(
        &self,
        value: <C::Real as Container>::Of<impl ByRef<T::RealUnit>>,
    ) -> Real<C::OfSimd<T::SimdVec<S>>> {
        help!(C::Real);
        Real(unsafe { core::mem::transmute_copy(&T::simd_splat_real(&self.0, by_ref!(value))) })
    }

    #[inline(always)]
    pub fn add(
        &self,
        lhs: C::OfSimd<T::SimdVec<S>>,
        rhs: C::OfSimd<T::SimdVec<S>>,
    ) -> C::OfSimd<T::SimdVec<S>> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_add(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn sub(
        &self,
        lhs: C::OfSimd<T::SimdVec<S>>,
        rhs: C::OfSimd<T::SimdVec<S>>,
    ) -> C::OfSimd<T::SimdVec<S>> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_sub(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn neg(&self, value: C::OfSimd<T::SimdVec<S>>) -> C::OfSimd<T::SimdVec<S>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_neg(&self.0, value)) }
    }
    #[inline(always)]
    pub fn conj(&self, value: C::OfSimd<T::SimdVec<S>>) -> C::OfSimd<T::SimdVec<S>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_conj(&self.0, value)) }
    }
    #[inline(always)]
    pub fn abs1(&self, value: C::OfSimd<T::SimdVec<S>>) -> Real<C::OfSimd<T::SimdVec<S>>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        Real(unsafe { core::mem::transmute_copy(&T::simd_abs1(&self.0, value)) })
    }
    #[inline(always)]
    pub fn abs_max(&self, value: C::OfSimd<T::SimdVec<S>>) -> Real<C::OfSimd<T::SimdVec<S>>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        Real(unsafe { core::mem::transmute_copy(&T::simd_abs_max(&self.0, value)) })
    }

    #[inline(always)]
    pub fn mul_real(
        &self,
        lhs: C::OfSimd<T::SimdVec<S>>,
        rhs: Real<C::OfSimd<T::SimdVec<S>>>,
    ) -> C::OfSimd<T::SimdVec<S>> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_mul_real(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn mul_pow2(
        &self,
        lhs: C::OfSimd<T::SimdVec<S>>,
        rhs: Real<C::OfSimd<T::SimdVec<S>>>,
    ) -> C::OfSimd<T::SimdVec<S>> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_mul_pow2(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn mul(
        &self,
        lhs: C::OfSimd<T::SimdVec<S>>,
        rhs: C::OfSimd<T::SimdVec<S>>,
    ) -> C::OfSimd<T::SimdVec<S>> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_mul(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn conj_mul(
        &self,
        lhs: C::OfSimd<T::SimdVec<S>>,
        rhs: C::OfSimd<T::SimdVec<S>>,
    ) -> C::OfSimd<T::SimdVec<S>> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_conj_mul(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn mul_add(
        &self,
        lhs: C::OfSimd<T::SimdVec<S>>,
        rhs: C::OfSimd<T::SimdVec<S>>,
        acc: C::OfSimd<T::SimdVec<S>>,
    ) -> C::OfSimd<T::SimdVec<S>> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        let acc = unsafe { core::mem::transmute_copy(&acc) };
        unsafe { core::mem::transmute_copy(&T::simd_mul_add(&self.0, lhs, rhs, acc)) }
    }

    #[inline(always)]
    pub fn conj_mul_add(
        &self,
        lhs: C::OfSimd<T::SimdVec<S>>,
        rhs: C::OfSimd<T::SimdVec<S>>,
        acc: C::OfSimd<T::SimdVec<S>>,
    ) -> C::OfSimd<T::SimdVec<S>> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        let acc = unsafe { core::mem::transmute_copy(&acc) };
        unsafe { core::mem::transmute_copy(&T::simd_conj_mul_add(&self.0, lhs, rhs, acc)) }
    }

    #[inline(always)]
    pub fn abs2(&self, value: C::OfSimd<T::SimdVec<S>>) -> Real<C::OfSimd<T::SimdVec<S>>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        Real(unsafe { core::mem::transmute_copy(&T::simd_abs2(&self.0, value)) })
    }

    #[inline(always)]
    pub fn abs2_add(
        &self,
        value: C::OfSimd<T::SimdVec<S>>,
        acc: Real<C::OfSimd<T::SimdVec<S>>>,
    ) -> Real<C::OfSimd<T::SimdVec<S>>> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        let acc = unsafe { core::mem::transmute_copy(&acc) };
        Real(unsafe { core::mem::transmute_copy(&T::simd_abs2_add(&self.0, value, acc)) })
    }

    #[inline(always)]
    pub fn reduce_sum(&self, value: C::OfSimd<T::SimdVec<S>>) -> C::Of<T> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_reduce_sum(&self.0, value)) }
    }
    #[inline(always)]
    pub fn reduce_max(&self, value: Real<C::OfSimd<T::SimdVec<S>>>) -> RealValue<C, T> {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_reduce_max(&self.0, value)) }
    }

    #[inline(always)]
    pub fn max(
        &self,
        lhs: Real<C::OfSimd<T::SimdVec<S>>>,
        rhs: Real<C::OfSimd<T::SimdVec<S>>>,
    ) -> Real<C::OfSimd<T::SimdVec<S>>> {
        let cmp = self.gt(lhs, rhs);
        Real(self.select(cmp, lhs.0, rhs.0))
    }

    #[inline(always)]
    pub fn lt(
        &self,
        lhs: Real<C::OfSimd<T::SimdVec<S>>>,
        rhs: Real<C::OfSimd<T::SimdVec<S>>>,
    ) -> T::SimdMask<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_less_than(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn gt(
        &self,
        lhs: Real<C::OfSimd<T::SimdVec<S>>>,
        rhs: Real<C::OfSimd<T::SimdVec<S>>>,
    ) -> T::SimdMask<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_greater_than(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn le(
        &self,
        lhs: Real<C::OfSimd<T::SimdVec<S>>>,
        rhs: Real<C::OfSimd<T::SimdVec<S>>>,
    ) -> T::SimdMask<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_less_than_or_equal(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn ge(
        &self,
        lhs: Real<C::OfSimd<T::SimdVec<S>>>,
        rhs: Real<C::OfSimd<T::SimdVec<S>>>,
    ) -> T::SimdMask<S> {
        let lhs = unsafe { core::mem::transmute_copy(&lhs) };
        let rhs = unsafe { core::mem::transmute_copy(&rhs) };
        unsafe { core::mem::transmute_copy(&T::simd_greater_than_or_equal(&self.0, lhs, rhs)) }
    }

    #[inline(always)]
    pub fn select(
        &self,
        mask: T::SimdMask<S>,
        lhs: C::OfSimd<T::SimdVec<S>>,
        rhs: C::OfSimd<T::SimdVec<S>>,
    ) -> C::OfSimd<T::SimdVec<S>> {
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
        ptr: C::Of<*const T::SimdVec<S>>,
    ) -> C::OfSimd<T::SimdVec<S>> {
        unsafe { core::mem::transmute_copy(&T::simd_mask_load(&self.0, mask, ptr)) }
    }
    #[inline(always)]
    pub unsafe fn mask_store(
        &self,
        mask: T::SimdMask<S>,
        ptr: C::Of<*mut T::SimdVec<S>>,
        value: C::OfSimd<T::SimdVec<S>>,
    ) {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_mask_store(&self.0, mask, ptr, value)) }
    }

    #[inline(always)]
    pub fn load(&self, ptr: C::Of<&T::SimdVec<S>>) -> C::OfSimd<T::SimdVec<S>> {
        unsafe { core::mem::transmute_copy(&T::simd_load(&self.0, ptr)) }
    }
    #[inline(always)]
    pub fn store(&self, ptr: C::Of<&mut T::SimdVec<S>>, value: C::OfSimd<T::SimdVec<S>>) {
        let value = unsafe { core::mem::transmute_copy(&value) };
        unsafe { core::mem::transmute_copy(&T::simd_store(&self.0, ptr, value)) }
    }
}

pub trait RealUnit: ConjUnit<Conj = Self, Canonical = Self> {}
pub unsafe trait ConjUnit {
    const IS_CANONICAL: bool;

    type Conj: ConjUnit;
    type Canonical: ConjUnit<Canonical = Self::Canonical>;
}

pub unsafe trait Container: 'static + core::fmt::Debug {
    type Of<T>;
    type OfCopy<T: Copy>: Copy;
    type OfDebug<T: Debug>: Debug;
    type OfSend<T: Send>: Send;
    type OfSync<T: Sync + Copy>: Sync + Copy;

    type OfSimd<T: Copy + Debug>: Copy + Debug;

    const IS_UNIT: bool = false;
    const IS_COMPLEX: bool = false;

    const NIL: Self::Of<()>;
    const N_COMPONENTS: usize = size_of::<Self::Of<u8>>();

    const IS_CANONICAL: bool;
    type Conj: Container<Conj = Self, Canonical = Self::Canonical, Real = Self::Real>;
    type Canonical: Container<Canonical = Self::Canonical, Real = Self::Real>;
    type Real: RealContainer;

    fn map_impl<T, U, Ctx>(
        value: Self::Of<T>,
        ctx: Ctx,
        f: &mut impl FnMut(T, Ctx) -> (U, Ctx),
    ) -> (Self::Of<U>, Ctx);
    unsafe fn as_ptr<T>(ptr: *mut Self::Of<T>) -> Self::Of<*mut T>;
    fn zip<A, B>(a: Self::Of<A>, b: Self::Of<B>) -> Self::Of<(A, B)>;
    fn unzip<A, B>(ab: Self::Of<(A, B)>) -> (Self::Of<A>, Self::Of<B>);
}

pub trait ComplexContainer: Container<Canonical = Self> {}
pub trait RealContainer: ComplexContainer<Conj = Self, Canonical = Self, Real = Self> {}
impl<C: Container<Canonical = C>> ComplexContainer for C {}
impl<C: Container<Canonical = C, Conj = C, Real = Self>> RealContainer for C {}

pub type RealValue<C, T> = <<C as Container>::Real as Container>::Of<
    <<T as ConjUnit>::Canonical as ComplexField<<C as Container>::Canonical>>::RealUnit,
>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct Unit;

unsafe impl Container for Unit {
    type Of<T> = T;
    type OfCopy<T: Copy> = T;
    type OfDebug<T: Debug> = T;
    type OfSend<T: Send> = T;
    type OfSync<T: Sync + Copy> = T;
    type OfSimd<T: Copy + Debug> = T;

    const IS_UNIT: bool = true;
    const NIL: Self::Of<()> = ();

    const IS_CANONICAL: bool = true;
    type Conj = Self;
    type Canonical = Self;
    type Real = Self;

    #[inline(always)]
    fn map_impl<T, U, Ctx>(value: T, ctx: Ctx, f: &mut impl FnMut(T, Ctx) -> (U, Ctx)) -> (U, Ctx) {
        (*f)(value, ctx)
    }

    #[inline(always)]
    unsafe fn as_ptr<T>(ptr: *mut Self::Of<T>) -> Self::Of<*mut T> {
        ptr
    }

    #[inline(always)]
    fn zip<A, B>(a: Self::Of<A>, b: Self::Of<B>) -> Self::Of<(A, B)> {
        (a, b)
    }

    #[inline(always)]
    fn unzip<A, B>(ab: Self::Of<(A, B)>) -> (Self::Of<A>, Self::Of<B>) {
        (ab.0, ab.1)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ComplexConj<T> {
    pub re: T,
    pub im_neg: T,
}

unsafe impl<C: RealContainer> Container for Complex<C> {
    type Of<T> = Complex<C::Of<T>>;
    type OfCopy<T: Copy> = Complex<C::OfCopy<T>>;
    type OfDebug<T: Debug> = Complex<C::OfDebug<T>>;
    type OfSend<T: Send> = Complex<C::OfSend<T>>;
    type OfSync<T: Sync + Copy> = Complex<C::OfSync<T>>;
    type OfSimd<T: Copy + Debug> = Complex<C::OfSimd<T>>;

    const IS_COMPLEX: bool = true;
    type Real = C;

    const NIL: Self::Of<()> = Complex {
        re: C::NIL,
        im: C::NIL,
    };

    const IS_CANONICAL: bool = true;
    type Conj = ComplexConj<C>;
    type Canonical = Self;

    #[inline(always)]
    fn map_impl<T, U, Ctx>(
        value: Self::Of<T>,
        ctx: Ctx,
        f: &mut impl FnMut(T, Ctx) -> (U, Ctx),
    ) -> (Self::Of<U>, Ctx) {
        let (re, ctx) = C::map_impl(value.re, ctx, f);
        let (im, ctx) = C::map_impl(value.im, ctx, f);
        (Complex { re, im }, ctx)
    }

    #[inline(always)]
    unsafe fn as_ptr<T>(ptr: *mut Self::Of<T>) -> Self::Of<*mut T> {
        let re = addr_of_mut!((*ptr).re);
        let im = addr_of_mut!((*ptr).im);
        Complex {
            re: C::as_ptr(re),
            im: C::as_ptr(im),
        }
    }

    #[inline(always)]
    fn zip<A, B>(a: Self::Of<A>, b: Self::Of<B>) -> Self::Of<(A, B)> {
        let re = C::zip(a.re, b.re);
        let im = C::zip(a.im, b.im);
        Complex { re, im }
    }

    #[inline(always)]
    fn unzip<A, B>(ab: Self::Of<(A, B)>) -> (Self::Of<A>, Self::Of<B>) {
        let re = C::unzip(ab.re);
        let im = C::unzip(ab.im);
        (
            Complex { re: re.0, im: im.0 },
            Complex { re: re.1, im: im.1 },
        )
    }
}

unsafe impl<C: RealContainer> Container for ComplexConj<C> {
    type Of<T> = ComplexConj<C::Of<T>>;
    type OfCopy<T: Copy> = ComplexConj<C::OfCopy<T>>;
    type OfDebug<T: Debug> = ComplexConj<C::OfDebug<T>>;
    type OfSend<T: Send> = ComplexConj<C::OfSend<T>>;
    type OfSync<T: Sync + Copy> = ComplexConj<C::OfSync<T>>;
    type OfSimd<T: Copy + Debug> = ComplexConj<C::OfSimd<T>>;

    const NIL: Self::Of<()> = ComplexConj {
        re: C::NIL,
        im_neg: C::NIL,
    };

    type Real = C;

    const IS_CANONICAL: bool = false;
    type Conj = Complex<C>;
    type Canonical = Complex<C>;

    #[inline(always)]
    fn map_impl<T, U, Ctx>(
        value: Self::Of<T>,
        ctx: Ctx,
        f: &mut impl FnMut(T, Ctx) -> (U, Ctx),
    ) -> (Self::Of<U>, Ctx) {
        let (re, ctx) = C::map_impl(value.re, ctx, f);
        let (im_neg, ctx) = C::map_impl(value.im_neg, ctx, f);
        (ComplexConj { re, im_neg }, ctx)
    }

    #[inline(always)]
    unsafe fn as_ptr<T>(ptr: *mut Self::Of<T>) -> Self::Of<*mut T> {
        let re = addr_of_mut!((*ptr).re);
        let im = addr_of_mut!((*ptr).im_neg);
        ComplexConj {
            re: C::as_ptr(re),
            im_neg: C::as_ptr(im),
        }
    }

    #[inline(always)]
    fn zip<A, B>(a: Self::Of<A>, b: Self::Of<B>) -> Self::Of<(A, B)> {
        let re = C::zip(a.re, b.re);
        let im_neg = C::zip(a.im_neg, b.im_neg);
        ComplexConj { re, im_neg }
    }

    #[inline(always)]
    fn unzip<A, B>(ab: Self::Of<(A, B)>) -> (Self::Of<A>, Self::Of<B>) {
        let re = C::unzip(ab.re);
        let im = C::unzip(ab.im_neg);
        (
            ComplexConj {
                re: re.0,
                im_neg: im.0,
            },
            ComplexConj {
                re: re.1,
                im_neg: im.1,
            },
        )
    }
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

unsafe impl<T: RealUnit> ConjUnit for T {
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

unsafe impl<T: RealField> ConjUnit for Complex<T> {
    const IS_CANONICAL: bool = true;
    type Conj = ComplexConj<T>;
    type Canonical = Complex<T>;
}
unsafe impl<T: RealField> ConjUnit for ComplexConj<T> {
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

pub trait ComplexField<C: ComplexContainer = Unit>:
    Debug + Send + Sync + Clone + ConjUnit<Canonical = Self, Conj: ConjUnit<Canonical = Self>>
{
    const IS_REAL: bool;

    type Arch: SimdArch;

    type SimdCtx<S: Simd>: Copy;
    type Index: Index;

    type MathCtx: Send + Sync + Default;

    type RealUnit: RealField<C::Real, MathCtx = Self::MathCtx>;

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

    fn zero_impl(ctx: &Self::MathCtx) -> C::Of<Self>;
    fn one_impl(ctx: &Self::MathCtx) -> C::Of<Self>;
    fn nan_impl(ctx: &Self::MathCtx) -> C::Of<Self>;
    fn infinity_impl(ctx: &Self::MathCtx) -> C::Of<Self>;

    fn from_real_impl(
        ctx: &Self::MathCtx,
        real: <C::Real as Container>::Of<&Self::RealUnit>,
    ) -> C::Of<Self>;
    fn from_f64_impl(ctx: &Self::MathCtx, real: f64) -> C::Of<Self>;

    fn real_part_impl(
        ctx: &Self::MathCtx,
        value: C::Of<&Self>,
    ) -> <C::Real as Container>::Of<Self::RealUnit>;
    fn imag_part_impl(
        ctx: &Self::MathCtx,
        value: C::Of<&Self>,
    ) -> <C::Real as Container>::Of<Self::RealUnit>;

    fn copy_impl(ctx: &Self::MathCtx, value: C::Of<&Self>) -> C::Of<Self>;
    fn neg_impl(ctx: &Self::MathCtx, value: C::Of<&Self>) -> C::Of<Self>;
    fn conj_impl(ctx: &Self::MathCtx, value: C::Of<&Self>) -> C::Of<Self>;
    fn recip_impl(ctx: &Self::MathCtx, value: C::Of<&Self>) -> C::Of<Self>;
    fn sqrt_impl(ctx: &Self::MathCtx, value: C::Of<&Self>) -> C::Of<Self>;

    fn abs_impl(
        ctx: &Self::MathCtx,

        value: C::Of<&Self>,
    ) -> <C::Real as Container>::Of<Self::RealUnit>;
    fn abs1_impl(
        ctx: &Self::MathCtx,

        value: C::Of<&Self>,
    ) -> <C::Real as Container>::Of<Self::RealUnit>;
    fn abs2_impl(
        ctx: &Self::MathCtx,

        value: C::Of<&Self>,
    ) -> <C::Real as Container>::Of<Self::RealUnit>;

    fn add_impl(ctx: &Self::MathCtx, lhs: C::Of<&Self>, rhs: C::Of<&Self>) -> C::Of<Self>;

    fn sub_impl(ctx: &Self::MathCtx, lhs: C::Of<&Self>, rhs: C::Of<&Self>) -> C::Of<Self>;

    fn mul_impl(ctx: &Self::MathCtx, lhs: C::Of<&Self>, rhs: C::Of<&Self>) -> C::Of<Self>;

    fn div_impl(ctx: &Self::MathCtx, lhs: C::Of<&Self>, rhs: C::Of<&Self>) -> C::Of<Self> {
        help!(C);
        Self::mul_impl(ctx, lhs, as_ref!(Self::recip_impl(ctx, rhs)))
    }

    fn mul_real_impl(
        ctx: &Self::MathCtx,
        lhs: C::Of<&Self>,
        rhs: <C::Real as Container>::Of<&Self::RealUnit>,
    ) -> C::Of<Self>;

    fn mul_pow2_impl(
        ctx: &Self::MathCtx,
        lhs: C::Of<&Self>,
        rhs: <C::Real as Container>::Of<&Self::RealUnit>,
    ) -> C::Of<Self>;

    fn eq_impl(ctx: &Self::MathCtx, lhs: C::Of<&Self>, rhs: C::Of<&Self>) -> bool;
    fn is_finite_impl(ctx: &Self::MathCtx, value: C::Of<&Self>) -> bool;
    fn is_zero_impl(ctx: &Self::MathCtx, value: C::Of<&Self>) -> bool;
    fn is_nan_impl(ctx: &Self::MathCtx, value: C::Of<&Self>) -> bool {
        help!(C);
        !Self::eq_impl(ctx, copy!(value), value)
    }

    fn simd_ctx<S: Simd>(ctx: &Self::MathCtx, simd: S) -> Self::SimdCtx<S>;
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> (Self::MathCtx, S);

    fn simd_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: C::Of<&Self>) -> C::Of<Self::SimdVec<S>>;
    fn simd_splat_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <C::Real as Container>::Of<&Self::RealUnit>,
    ) -> C::Of<Self::SimdVec<S>>;

    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: C::Of<Self::SimdVec<S>>,
        rhs: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: C::Of<Self::SimdVec<S>>,
        rhs: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;

    fn simd_neg<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    fn simd_conj<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    fn simd_abs1<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    fn simd_abs_max<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;

    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: C::Of<Self::SimdVec<S>>,
        real_rhs: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: C::Of<Self::SimdVec<S>>,
        real_rhs: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;

    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: C::Of<Self::SimdVec<S>>,
        rhs: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: C::Of<Self::SimdVec<S>>,
        rhs: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: C::Of<Self::SimdVec<S>>,
        rhs: C::Of<Self::SimdVec<S>>,
        acc: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: C::Of<Self::SimdVec<S>>,
        rhs: C::Of<Self::SimdVec<S>>,
        acc: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    fn simd_abs2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: C::Of<Self::SimdVec<S>>,
        acc: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;

    fn simd_reduce_sum<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self>;
    fn simd_reduce_max<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: C::Of<Self::SimdVec<S>>,
    ) -> <C::Real as Container>::Of<Self::RealUnit>;
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: C::Of<Self::SimdVec<S>>,
        real_rhs: C::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S>;
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: C::Of<Self::SimdVec<S>>,
        real_rhs: C::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S>;
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: C::Of<Self::SimdVec<S>>,
        real_rhs: C::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S>;
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: C::Of<Self::SimdVec<S>>,
        real_rhs: C::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S>;

    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: C::Of<Self::SimdVec<S>>,
        rhs: C::Of<Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
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
        ptr: C::Of<*const Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    unsafe fn simd_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: C::Of<*mut Self::SimdVec<S>>,
        value: C::Of<Self::SimdVec<S>>,
    );

    fn simd_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: C::Of<&Self::SimdVec<S>>,
    ) -> C::Of<Self::SimdVec<S>>;
    fn simd_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: C::Of<&mut Self::SimdVec<S>>,
        value: C::Of<Self::SimdVec<S>>,
    );

    fn simd_iota<S: Simd>(ctx: &Self::SimdCtx<S>) -> Self::SimdIndex<S>;
}

pub trait RealField<C: RealContainer = Unit>: RealUnit + ComplexField<C, RealUnit = Self> {
    fn epsilon_impl(ctx: &Self::MathCtx) -> C::Of<Self>;
    fn nbits_impl(ctx: &Self::MathCtx) -> usize;

    fn min_positive_impl(ctx: &Self::MathCtx) -> C::Of<Self>;
    fn max_positive_impl(ctx: &Self::MathCtx) -> C::Of<Self>;
    fn sqrt_min_positive_impl(ctx: &Self::MathCtx) -> C::Of<Self>;
    fn sqrt_max_positive_impl(ctx: &Self::MathCtx) -> C::Of<Self>;

    fn partial_cmp_impl(
        ctx: &Self::MathCtx,
        lhs: C::Of<&Self>,
        rhs: C::Of<&Self>,
    ) -> Option<Ordering>;
    fn partial_cmp_zero_impl(ctx: &Self::MathCtx, value: C::Of<&Self>) -> Option<Ordering>;
}

impl<C: RealContainer, T: RealField<C>> ComplexField<Complex<C>> for T {
    const IS_REAL: bool = false;

    type SimdCtx<S: Simd> = T::SimdCtx<S>;
    type Index = T::Index;
    type MathCtx = T::MathCtx;

    type RealUnit = T;
    type Arch = T::Arch;

    const SIMD_CAPABILITIES: SimdCapabilities = T::SIMD_CAPABILITIES;
    type SimdMask<S: Simd> = T::SimdMask<S>;
    type SimdVec<S: Simd> = T::SimdVec<S>;
    type SimdIndex<S: Simd> = T::SimdIndex<S>;

    #[inline(always)]
    fn zero_impl(ctx: &Self::MathCtx) -> <Complex<C> as Container>::Of<Self> {
        Complex {
            re: T::zero_impl(ctx),
            im: T::zero_impl(ctx),
        }
    }

    #[inline(always)]
    fn add_impl(
        ctx: &Self::MathCtx,
        lhs: <Complex<C> as Container>::Of<&Self>,
        rhs: <Complex<C> as Container>::Of<&Self>,
    ) -> <Complex<C> as Container>::Of<Self> {
        let ctx = Ctx::<C, T>::new(ctx);
        Complex {
            re: ctx.add(&lhs.re, &rhs.re),
            im: ctx.add(&lhs.im, &rhs.im),
        }
    }
    #[inline(always)]
    fn sub_impl(
        ctx: &Self::MathCtx,
        lhs: <Complex<C> as Container>::Of<&Self>,
        rhs: <Complex<C> as Container>::Of<&Self>,
    ) -> <Complex<C> as Container>::Of<Self> {
        let ctx = Ctx::<C, T>::new(ctx);
        Complex {
            re: ctx.sub(&lhs.re, &rhs.re),
            im: ctx.sub(&lhs.im, &rhs.im),
        }
    }
    #[inline(always)]
    fn mul_impl(
        ctx: &Self::MathCtx,
        lhs: <Complex<C> as Container>::Of<&Self>,
        rhs: <Complex<C> as Container>::Of<&Self>,
    ) -> <Complex<C> as Container>::Of<Self> {
        help!(C);
        let ctx = Ctx::<C, T>::new(ctx);

        let left = ctx.mul(&copy!(lhs.re), &copy!(rhs.re));
        let right = ctx.mul(&copy!(lhs.im), &copy!(rhs.im));
        let re = ctx.sub(&left, &right);

        let left = ctx.mul(&copy!(lhs.re), &copy!(rhs.im));
        let right = ctx.mul(&copy!(lhs.im), &copy!(rhs.re));
        let im = ctx.add(&left, &right);
        Complex { re, im }
    }

    #[inline(always)]
    fn mul_real_impl(
        ctx: &Self::MathCtx,
        lhs: <Complex<C> as Container>::Of<&Self>,
        rhs: C::Of<&Self::RealUnit>,
    ) -> <Complex<C> as Container>::Of<Self> {
        help!(C);
        let ctx = Ctx::<C, T>::new(ctx);
        let re = ctx.mul_real(&lhs.re, &rhs);
        let im = ctx.mul_real(&lhs.im, &rhs);
        Complex { re, im }
    }

    #[inline(always)]
    fn mul_pow2_impl(
        ctx: &Self::MathCtx,
        lhs: <Complex<C> as Container>::Of<&Self>,
        rhs: C::Of<&Self::RealUnit>,
    ) -> <Complex<C> as Container>::Of<Self> {
        help!(C);
        let ctx = Ctx::<C, T>::new(ctx);
        let re = ctx.mul_pow2(&lhs.re, &rhs);
        let im = ctx.mul_pow2(&lhs.im, &rhs);
        Complex { re, im }
    }

    #[inline(always)]
    fn one_impl(ctx: &Self::MathCtx) -> <Complex<C> as Container>::Of<Self> {
        let ctx = Ctx::<C, T>::new(ctx);
        let re = ctx.one();
        let im = ctx.zero();
        Complex { re, im }
    }

    #[inline(always)]
    fn real_part_impl(
        ctx: &Self::MathCtx,
        value: <Complex<C> as Container>::Of<&Self>,
    ) -> C::Of<Self::RealUnit> {
        let ctx = Ctx::<C, T>::new(ctx);
        ctx.copy(&value.re)
    }

    #[inline(always)]
    fn imag_part_impl(
        ctx: &Self::MathCtx,
        value: <Complex<C> as Container>::Of<&Self>,
    ) -> C::Of<Self::RealUnit> {
        let ctx = Ctx::<C, T>::new(ctx);
        ctx.copy(&value.im)
    }

    #[inline(always)]
    fn copy_impl(
        ctx: &Self::MathCtx,
        value: <Complex<C> as Container>::Of<&Self>,
    ) -> <Complex<C> as Container>::Of<Self> {
        let ctx = Ctx::<C, T>::new(ctx);
        let re = ctx.copy(&value.re);
        let im = ctx.copy(&value.im);
        Complex { re, im }
    }

    #[inline(always)]
    fn neg_impl(
        ctx: &Self::MathCtx,
        value: <Complex<C> as Container>::Of<&Self>,
    ) -> <Complex<C> as Container>::Of<Self> {
        let ctx = Ctx::<C, T>::new(ctx);
        let re = ctx.neg(&value.re);
        let im = ctx.neg(&value.im);
        Complex { re, im }
    }

    #[inline(always)]
    fn conj_impl(
        ctx: &Self::MathCtx,
        value: <Complex<C> as Container>::Of<&Self>,
    ) -> <Complex<C> as Container>::Of<Self> {
        let ctx = Ctx::<C, T>::new(ctx);
        let re = ctx.copy(&value.re);
        let im = ctx.neg(&value.im);
        Complex { re, im }
    }

    #[inline(always)]
    fn abs1_impl(
        ctx: &Self::MathCtx,
        value: <Complex<C> as Container>::Of<&Self>,
    ) -> <C as Container>::Of<Self::RealUnit> {
        help!(C);
        let ctx = Ctx::<C, T>::new(ctx);
        let left = ctx.abs1(&value.re);
        let right = ctx.abs1(&value.im);
        ctx.add(&left, &right)
    }

    #[inline(always)]
    fn abs2_impl(
        ctx: &Self::MathCtx,

        value: <Complex<C> as Container>::Of<&Self>,
    ) -> <C as Container>::Of<Self::RealUnit> {
        help!(C);
        let ctx = Ctx::<C, T>::new(ctx);
        let left = ctx.abs2(&value.re);
        let right = ctx.abs2(&value.im);
        ctx.add(&left, &right)
    }

    fn recip_impl(
        ctx: &Self::MathCtx,
        value: <Complex<C> as Container>::Of<&Self>,
    ) -> <Complex<C> as Container>::Of<Self> {
        let (re, im) = recip_impl::<C, _>(ctx, value.re, value.im);
        Complex { re: re, im: im }
    }

    #[inline(always)]
    fn nan_impl(ctx: &Self::MathCtx) -> <Complex<C> as Container>::Of<Self> {
        let ctx = Ctx::<C, T>::new(ctx);
        Complex {
            re: ctx.nan(),
            im: ctx.nan(),
        }
    }

    #[inline(always)]
    fn infinity_impl(ctx: &Self::MathCtx) -> <Complex<C> as Container>::Of<Self> {
        let ctx = Ctx::<C, T>::new(ctx);
        Complex {
            re: ctx.infinity(),
            im: ctx.infinity(),
        }
    }

    #[faer_macros::math]
    #[inline(always)]
    fn eq_impl(
        ctx: &Self::MathCtx,
        lhs: <Complex<C> as Container>::Of<&Self>,
        rhs: <Complex<C> as Container>::Of<&Self>,
    ) -> bool {
        let ctx = Ctx::<C, T>::new(ctx);
        math.eq(lhs.re, rhs.re) && math.eq(lhs.im, rhs.im)
    }

    fn sqrt_impl(
        ctx: &Self::MathCtx,
        value: <Complex<C> as Container>::Of<&Self>,
    ) -> <Complex<C> as Container>::Of<Self> {
        let (re, im) = sqrt_impl::<C, _>(ctx, value.re, value.im);
        Complex { re: re, im: im }
    }

    fn abs_impl(
        ctx: &Self::MathCtx,
        value: <Complex<C> as Container>::Of<&Self>,
    ) -> <C as Container>::Of<Self::RealUnit> {
        abs_impl::<C, _>(ctx, value.re, value.im)
    }

    #[inline(always)]
    #[faer_macros::math]
    fn is_zero_impl(ctx: &Self::MathCtx, value: <Complex<C> as Container>::Of<&Self>) -> bool {
        let ctx = Ctx::<C, T>::new(ctx);
        math.is_zero(value.re) && math.is_zero(value.im)
    }

    #[inline(always)]
    fn from_real_impl(
        ctx: &Self::MathCtx,
        real: <C as Container>::Of<&Self::RealUnit>,
    ) -> <Complex<C> as Container>::Of<Self> {
        let ctx = Ctx::<C, T>::new(ctx);
        Complex {
            re: ctx.copy(&real),
            im: ctx.zero(),
        }
    }

    #[inline(always)]
    fn from_f64_impl(ctx: &Self::MathCtx, real: f64) -> <Complex<C> as Container>::Of<Self> {
        let ctx = Ctx::<C, T>::new(ctx);
        Complex {
            re: ctx.from_f64(&real),
            im: ctx.zero(),
        }
    }

    #[inline(always)]
    #[faer_macros::math]
    fn is_finite_impl(ctx: &Self::MathCtx, value: <Complex<C> as Container>::Of<&Self>) -> bool {
        let ctx = Ctx::<C, T>::new(ctx);
        math(is_finite(value.re) || is_finite(value.im))
    }

    #[inline(always)]
    #[faer_macros::math]
    fn is_nan_impl(ctx: &Self::MathCtx, value: <Complex<C> as Container>::Of<&Self>) -> bool {
        let ctx = Ctx::<C, T>::new(ctx);
        math(is_nan(value.re) || is_nan(value.im))
    }

    #[inline(always)]
    fn simd_ctx<S: Simd>(ctx: &Self::MathCtx, simd: S) -> Self::SimdCtx<S> {
        T::simd_ctx(ctx, simd)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Complex<C> as Container>::Of<&Self>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: ctx.splat(value.re),
            im: ctx.splat(value.im),
        }
    }

    #[inline(always)]
    fn simd_splat_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <C as Container>::Of<&Self::RealUnit>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        help!(C);
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: ctx.splat(copy!(value)),
            im: ctx.splat(value),
        }
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: ctx.add(lhs.re, rhs.re),
            im: ctx.add(lhs.im, rhs.im),
        }
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: ctx.sub(lhs.re, rhs.re),
            im: ctx.sub(lhs.im, rhs.im),
        }
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: ctx.neg(value.re),
            im: ctx.neg(value.im),
        }
    }

    #[inline(always)]
    fn simd_conj<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: value.re,
            im: ctx.neg(value.im),
        }
    }

    #[inline(always)]
    fn simd_abs1<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        help!(C);
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        let re = ctx.abs1(value.re).0;
        let im = ctx.abs1(value.im).0;
        let sum = ctx.add(re, im);

        Complex {
            re: copy!(sum),
            im: copy!(sum),
        }
    }

    #[inline(always)]
    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        help!(C);
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: ctx.mul_real(lhs.re, Real(real_rhs.re)),
            im: ctx.mul_real(lhs.im, Real(real_rhs.im)),
        }
    }

    #[inline(always)]
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        help!(C);
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: ctx.mul_pow2(lhs.re, Real(real_rhs.re)),
            im: ctx.mul_pow2(lhs.im, Real(real_rhs.im)),
        }
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        help!(C);
        let ctx = SimdCtx::<C, T, S>::new(ctx);

        Complex {
            re: ctx.mul_add(
                copy!(lhs.re),
                copy!(rhs.re),
                ctx.neg(ctx.mul(copy!(lhs.im), copy!(rhs.im))),
            ),
            im: ctx.mul_add(lhs.re, rhs.im, ctx.mul(lhs.im, rhs.re)),
        }
    }

    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        help!(C);
        let ctx = SimdCtx::<C, T, S>::new(ctx);

        Complex {
            re: ctx.mul_add(
                copy!(lhs.re),
                copy!(rhs.re),
                ctx.mul(copy!(lhs.im), copy!(rhs.im)),
            ),
            im: ctx.mul_add(lhs.re, rhs.im, ctx.neg(ctx.mul(lhs.im, rhs.re))),
        }
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        acc: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        help!(C);
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: ctx.mul_add(
                copy!(lhs.re),
                copy!(rhs.re),
                ctx.mul_add(ctx.neg(copy!(lhs.im)), copy!(rhs.im), acc.re),
            ),
            im: ctx.mul_add(lhs.re, rhs.im, ctx.mul_add(lhs.im, rhs.re, acc.im)),
        }
    }

    #[inline(always)]
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        acc: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        help!(C);
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: ctx.mul_add(
                copy!(lhs.re),
                copy!(rhs.re),
                ctx.mul_add(copy!(lhs.im), copy!(rhs.im), acc.re),
            ),
            im: ctx.mul_add(lhs.re, rhs.im, ctx.mul_add(ctx.neg(lhs.im), rhs.re, acc.im)),
        }
    }

    #[inline(always)]
    fn simd_abs2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        help!(C);
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        let re = ctx.abs2(value.re);
        let sum = ctx.abs2_add(value.im, re).0;

        Complex {
            re: copy!(sum),
            im: copy!(sum),
        }
    }

    #[inline(always)]
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        acc: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        help!(C);
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        let re = ctx.abs2_add(value.re, Real(acc.re));
        let sum = ctx.abs2_add(value.im, re).0;

        Complex {
            re: copy!(sum),
            im: copy!(sum),
        }
    }

    #[inline(always)]
    fn simd_reduce_sum<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        help!(C);
        Complex {
            re: ctx.reduce_sum(value.re),
            im: ctx.reduce_sum(value.im),
        }
    }

    #[inline(always)]
    fn simd_reduce_max<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> C::Of<T> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        help!(C);
        let re = ctx.reduce_max(value.re);
        let im = ctx.reduce_max(value.im);
        if T::partial_cmp_impl(&T::ctx_from_simd(&ctx.0).0, as_ref!(re), as_ref!(im))
            == Some(Ordering::Greater)
        {
            re
        } else {
            im
        }
    }

    #[inline(always)]
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        ctx.lt(Real(real_lhs.re), Real(real_rhs.re))
    }

    #[inline(always)]
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        ctx.gt(Real(real_lhs.re), Real(real_rhs.re))
    }

    #[inline(always)]
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        ctx.le(Real(real_lhs.re), Real(real_rhs.re))
    }

    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        ctx.ge(Real(real_lhs.re), Real(real_rhs.re))
    }

    #[inline(always)]
    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
        rhs: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: ctx.select(mask, lhs.re, rhs.re),
            im: ctx.select(mask, lhs.im, rhs.im),
        }
    }

    #[inline(always)]
    fn simd_index_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        ctx.iselect(mask, lhs, rhs)
    }

    #[inline(always)]
    fn simd_index_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::Index) -> Self::SimdIndex<S> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        ctx.isplat(value)
    }

    #[inline(always)]
    fn simd_index_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        ctx.iadd(lhs, rhs)
    }

    #[inline(always)]
    fn simd_tail_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        ctx.tail_mask(len)
    }
    #[inline(always)]
    fn simd_head_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        ctx.head_mask(len)
    }

    #[inline(always)]
    unsafe fn simd_mask_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: <Complex<C> as Container>::Of<*const Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        Complex {
            re: ctx.mask_load(mask, ptr.re),
            im: ctx.mask_load(mask, ptr.im),
        }
    }
    #[inline(always)]
    unsafe fn simd_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: <Complex<C> as Container>::Of<*mut Self::SimdVec<S>>,
        value: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) {
        let ctx = SimdCtx::<C, T, S>::new(ctx);
        ctx.mask_store(mask, ptr.re, value.re);
        ctx.mask_store(mask, ptr.im, value.im);
    }

    #[inline(always)]
    fn simd_iota<S: Simd>(ctx: &Self::SimdCtx<S>) -> Self::SimdIndex<S> {
        T::simd_iota(ctx)
    }

    #[inline(always)]
    fn simd_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: <Complex<C> as Container>::Of<&Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        Complex {
            re: T::simd_load(ctx, ptr.re),
            im: T::simd_load(ctx, ptr.im),
        }
    }
    #[inline(always)]
    fn simd_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: <Complex<C> as Container>::Of<&mut Self::SimdVec<S>>,
        value: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) {
        T::simd_store(ctx, ptr.re, value.re);
        T::simd_store(ctx, ptr.im, value.im);
    }

    #[inline(always)]
    fn simd_abs_max<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Complex<C> as Container>::Of<Self::SimdVec<S>>,
    ) -> <Complex<C> as Container>::Of<Self::SimdVec<S>> {
        help!(C);
        let re = T::simd_abs_max(ctx, value.re);
        let im = T::simd_abs_max(ctx, value.im);
        let cmp = T::simd_greater_than(ctx, copy!(re), copy!(im));
        let v = T::simd_select(ctx, cmp, re, im);
        Complex {
            re: copy!(v),
            im: v,
        }
    }

    #[inline(always)]
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> (Self::MathCtx, S) {
        T::ctx_from_simd(ctx)
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
}

impl AsRef<Unit> for Unit {
    #[inline(always)]
    fn as_ref(&self) -> &Unit {
        self
    }
}

impl<T: RealField> RealUnit for T {}

impl ComplexField for f32 {
    const IS_REAL: bool = true;

    type Index = u32;
    type SimdCtx<S: Simd> = S;
    type RealUnit = Self;
    type MathCtx = Unit;
    type Arch = pulp::Arch;

    const IS_NATIVE_F32: bool = true;

    const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::All;

    type SimdMask<S: Simd> = S::m32s;
    type SimdVec<S: Simd> = S::f32s;
    type SimdIndex<S: Simd> = S::u32s;

    #[inline(always)]
    fn zero_impl(_: &Self::MathCtx) -> Self {
        0.0
    }
    #[inline(always)]
    fn one_impl(_: &Self::MathCtx) -> Self {
        1.0
    }
    #[inline(always)]
    fn nan_impl(_: &Self::MathCtx) -> Self {
        Self::NAN
    }
    #[inline(always)]
    fn infinity_impl(_: &Self::MathCtx) -> Self {
        Self::INFINITY
    }
    #[inline(always)]
    fn from_real_impl(_: &Self::MathCtx, value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn from_f64_impl(_: &Self::MathCtx, value: f64) -> Self {
        value as _
    }

    #[inline(always)]
    fn real_part_impl(_: &Self::MathCtx, value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn imag_part_impl(_: &Self::MathCtx, _: &Self) -> Self {
        0.0
    }
    #[inline(always)]
    fn copy_impl(_: &Self::MathCtx, value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn neg_impl(_: &Self::MathCtx, value: &Self) -> Self {
        -*value
    }
    #[inline(always)]
    fn conj_impl(_: &Self::MathCtx, value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn recip_impl(_: &Self::MathCtx, value: &Self) -> Self {
        1.0 / *value
    }
    #[inline(always)]
    fn sqrt_impl(_: &Self::MathCtx, value: &Self) -> Self {
        (*value).sqrt()
    }
    #[inline(always)]
    fn abs_impl(_: &Self::MathCtx, value: &Self) -> Self {
        (*value).abs()
    }
    #[inline(always)]
    fn abs1_impl(_: &Self::MathCtx, value: &Self) -> Self {
        (*value).abs()
    }
    #[inline(always)]
    fn abs2_impl(_: &Self::MathCtx, value: &Self) -> Self {
        (*value) * (*value)
    }
    #[inline(always)]
    fn add_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) + (*rhs)
    }
    #[inline(always)]
    fn sub_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) - (*rhs)
    }
    #[inline(always)]
    fn mul_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) * (*rhs)
    }
    #[inline(always)]
    fn mul_real_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) * (*rhs)
    }
    #[inline(always)]
    fn mul_pow2_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) * (*rhs)
    }
    #[inline(always)]
    fn div_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) / (*rhs)
    }

    #[inline(always)]
    fn eq_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> bool {
        *lhs == *rhs
    }
    #[inline(always)]
    fn is_zero_impl(_: &Self::MathCtx, value: &Self) -> bool {
        *value == 0.0
    }
    #[inline(always)]
    fn is_finite_impl(_: &Self::MathCtx, value: &Self) -> bool {
        (*value).is_finite()
    }
    #[inline(always)]
    fn is_nan_impl(_: &Self::MathCtx, value: &Self) -> bool {
        (*value).is_nan()
    }

    #[inline(always)]
    fn simd_ctx<S: Simd>(_: &Self::MathCtx, simd: S) -> Self::SimdCtx<S> {
        simd
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<&Self>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.splat_f32s(*value)
    }

    #[inline(always)]
    fn simd_splat_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<&Self::RealUnit>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.splat_f32s(*value)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.add_f32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.sub_f32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.neg_f32s(value)
    }
    #[inline(always)]
    fn simd_conj<S: Simd>(
        _: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        value
    }
    #[inline(always)]
    fn simd_abs1<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.abs_f32s(value)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_f32s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_f32s(lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_f32s(lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_f32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        acc: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_add_e_f32s(lhs, rhs, acc)
    }
    #[inline(always)]
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        acc: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_add_e_f32s(lhs, rhs, acc)
    }
    #[inline(always)]
    fn simd_abs2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_f32s(value, value)
    }
    #[inline(always)]
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
        acc: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_add_e_f32s(value, value, acc)
    }
    #[inline(always)]
    fn simd_reduce_sum<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> Self {
        ctx.reduce_sum_f32s(value)
    }
    #[inline(always)]
    fn simd_reduce_max<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::RealUnit> {
        ctx.reduce_max_f32s(value)
    }
    #[inline(always)]
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        ctx.less_than_f32s(real_lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        ctx.greater_than_f32s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        ctx.less_than_or_equal_f32s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        ctx.greater_than_or_equal_f32s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
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
        ptr: <Unit as Container>::Of<*const Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mask_load_ptr_f32s(mask, ptr as *const f32)
    }
    #[inline(always)]
    unsafe fn simd_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: <Unit as Container>::Of<*mut Self::SimdVec<S>>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
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
    fn simd_load<S: Simd>(
        _: &Self::SimdCtx<S>,
        ptr: <Unit as Container>::Of<&Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        *ptr
    }
    #[inline(always)]
    fn simd_store<S: Simd>(
        _: &Self::SimdCtx<S>,
        ptr: <Unit as Container>::Of<&mut Self::SimdVec<S>>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) {
        *ptr = value;
    }

    #[inline(always)]
    fn simd_abs_max<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.abs_f32s(value)
    }

    #[inline(always)]
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> (Self::MathCtx, S) {
        (Unit, *ctx)
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
    fn epsilon_impl(_: &Self::MathCtx) -> Self {
        Self::EPSILON
    }
    #[inline(always)]
    fn min_positive_impl(_: &Self::MathCtx) -> Self {
        Self::MIN_POSITIVE
    }
    #[inline(always)]
    fn max_positive_impl(_: &Self::MathCtx) -> Self {
        Self::MIN_POSITIVE.recip()
    }
    #[inline(always)]
    fn sqrt_min_positive_impl(_: &Self::MathCtx) -> Self {
        Self::MIN_POSITIVE.sqrt()
    }
    #[inline(always)]
    fn sqrt_max_positive_impl(_: &Self::MathCtx) -> Self {
        Self::MIN_POSITIVE.recip().sqrt()
    }

    #[inline(always)]
    fn partial_cmp_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Option<Ordering> {
        (*lhs).partial_cmp(rhs)
    }
    #[inline(always)]
    fn partial_cmp_zero_impl(_: &Self::MathCtx, value: &Self) -> Option<Ordering> {
        (*value).partial_cmp(&0.0)
    }

    #[inline(always)]
    fn nbits_impl(_: &Self::MathCtx) -> usize {
        Self::MANTISSA_DIGITS as usize
    }
}

impl ComplexField for f64 {
    const IS_REAL: bool = true;

    type Index = u64;
    type SimdCtx<S: Simd> = S;
    type RealUnit = Self;
    type MathCtx = Unit;
    type Arch = pulp::Arch;

    const IS_NATIVE_F64: bool = true;

    const SIMD_CAPABILITIES: SimdCapabilities = SimdCapabilities::All;

    type SimdMask<S: Simd> = S::m64s;
    type SimdVec<S: Simd> = S::f64s;
    type SimdIndex<S: Simd> = S::u64s;

    #[inline(always)]
    fn zero_impl(_: &Self::MathCtx) -> Self {
        0.0
    }
    #[inline(always)]
    fn one_impl(_: &Self::MathCtx) -> Self {
        1.0
    }
    #[inline(always)]
    fn nan_impl(_: &Self::MathCtx) -> Self {
        Self::NAN
    }
    #[inline(always)]
    fn infinity_impl(_: &Self::MathCtx) -> Self {
        Self::INFINITY
    }
    #[inline(always)]
    fn from_real_impl(_: &Self::MathCtx, value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn from_f64_impl(_: &Self::MathCtx, value: f64) -> Self {
        value as _
    }

    #[inline(always)]
    fn real_part_impl(_: &Self::MathCtx, value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn imag_part_impl(_: &Self::MathCtx, _: &Self) -> Self {
        0.0
    }
    #[inline(always)]
    fn copy_impl(_: &Self::MathCtx, value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn neg_impl(_: &Self::MathCtx, value: &Self) -> Self {
        -*value
    }
    #[inline(always)]
    fn conj_impl(_: &Self::MathCtx, value: &Self) -> Self {
        *value
    }
    #[inline(always)]
    fn recip_impl(_: &Self::MathCtx, value: &Self) -> Self {
        1.0 / *value
    }
    #[inline(always)]
    fn sqrt_impl(_: &Self::MathCtx, value: &Self) -> Self {
        (*value).sqrt()
    }
    #[inline(always)]
    fn abs_impl(_: &Self::MathCtx, value: &Self) -> Self {
        (*value).abs()
    }
    #[inline(always)]
    fn abs1_impl(_: &Self::MathCtx, value: &Self) -> Self {
        (*value).abs()
    }
    #[inline(always)]
    fn abs2_impl(_: &Self::MathCtx, value: &Self) -> Self {
        (*value) * (*value)
    }
    #[inline(always)]
    fn add_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) + (*rhs)
    }
    #[inline(always)]
    fn sub_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) - (*rhs)
    }
    #[inline(always)]
    fn mul_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) * (*rhs)
    }
    #[inline(always)]
    fn mul_real_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) * (*rhs)
    }
    #[inline(always)]
    fn mul_pow2_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) * (*rhs)
    }
    #[inline(always)]
    fn div_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        (*lhs) / (*rhs)
    }

    #[inline(always)]
    fn eq_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> bool {
        *lhs == *rhs
    }
    #[inline(always)]
    fn is_zero_impl(_: &Self::MathCtx, value: &Self) -> bool {
        *value == 0.0
    }
    #[inline(always)]
    fn is_finite_impl(_: &Self::MathCtx, value: &Self) -> bool {
        (*value).is_finite()
    }
    #[inline(always)]
    fn is_nan_impl(_: &Self::MathCtx, value: &Self) -> bool {
        (*value).is_nan()
    }

    #[inline(always)]
    fn simd_ctx<S: Simd>(_: &Self::MathCtx, simd: S) -> Self::SimdCtx<S> {
        simd
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<&Self>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.splat_f64s(*value)
    }

    #[inline(always)]
    fn simd_splat_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<&Self::RealUnit>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.splat_f64s(*value)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.add_f64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.sub_f64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.neg_f64s(value)
    }
    #[inline(always)]
    fn simd_conj<S: Simd>(
        _: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        value
    }
    #[inline(always)]
    fn simd_abs1<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.abs_f64s(value)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_f64s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_f64s(lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_f64s(lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_f64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        acc: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_add_e_f64s(lhs, rhs, acc)
    }
    #[inline(always)]
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        acc: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_add_e_f64s(lhs, rhs, acc)
    }
    #[inline(always)]
    fn simd_abs2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_f64s(value, value)
    }
    #[inline(always)]
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
        acc: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mul_add_e_f64s(value, value, acc)
    }
    #[inline(always)]
    fn simd_reduce_sum<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> Self {
        ctx.reduce_sum_f64s(value)
    }
    #[inline(always)]
    fn simd_reduce_max<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::RealUnit> {
        ctx.reduce_max_f64s(value)
    }
    #[inline(always)]
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        ctx.less_than_f64s(real_lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        ctx.greater_than_f64s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        ctx.less_than_or_equal_f64s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        real_rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> Self::SimdMask<S> {
        ctx.greater_than_or_equal_f64s(real_lhs, real_rhs)
    }
    #[inline(always)]
    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: <Unit as Container>::Of<Self::SimdVec<S>>,
        rhs: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
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
        ptr: <Unit as Container>::Of<*const Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.mask_load_ptr_f64s(mask, ptr as *const f64)
    }
    #[inline(always)]
    unsafe fn simd_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: <Unit as Container>::Of<*mut Self::SimdVec<S>>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
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
    fn simd_load<S: Simd>(
        _: &Self::SimdCtx<S>,
        ptr: <Unit as Container>::Of<&Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        *ptr
    }
    #[inline(always)]
    fn simd_store<S: Simd>(
        _: &Self::SimdCtx<S>,
        ptr: <Unit as Container>::Of<&mut Self::SimdVec<S>>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) {
        *ptr = value;
    }

    #[inline(always)]
    fn simd_abs_max<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        ctx.abs_f64s(value)
    }

    #[inline(always)]
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> (Self::MathCtx, S) {
        (Unit, *ctx)
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
}

impl RealField for f64 {
    #[inline(always)]
    fn epsilon_impl(_: &Self::MathCtx) -> Self {
        Self::EPSILON
    }
    #[inline(always)]
    fn min_positive_impl(_: &Self::MathCtx) -> Self {
        Self::MIN_POSITIVE
    }
    #[inline(always)]
    fn max_positive_impl(_: &Self::MathCtx) -> Self {
        Self::MIN_POSITIVE.recip()
    }
    #[inline(always)]
    fn sqrt_min_positive_impl(_: &Self::MathCtx) -> Self {
        Self::MIN_POSITIVE.sqrt()
    }
    #[inline(always)]
    fn sqrt_max_positive_impl(_: &Self::MathCtx) -> Self {
        Self::MIN_POSITIVE.recip().sqrt()
    }

    #[inline(always)]
    fn partial_cmp_impl(_: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Option<Ordering> {
        (*lhs).partial_cmp(rhs)
    }
    #[inline(always)]
    fn partial_cmp_zero_impl(_: &Self::MathCtx, value: &Self) -> Option<Ordering> {
        (*value).partial_cmp(&0.0)
    }

    #[inline(always)]
    fn nbits_impl(_: &Self::MathCtx) -> usize {
        Self::MANTISSA_DIGITS as usize
    }
}

impl<T: EnableComplex> ComplexField for Complex<T> {
    const IS_REAL: bool = false;
    type Arch = <T as EnableComplex>::Arch;

    type SimdCtx<S: Simd> = T::SimdCtx<S>;
    type Index = T::Index;

    type RealUnit = T;
    type MathCtx = T::MathCtx;

    const IS_NATIVE_C32: bool = T::IS_NATIVE_F32;
    const IS_NATIVE_C64: bool = T::IS_NATIVE_F64;

    const SIMD_CAPABILITIES: SimdCapabilities = T::SIMD_CAPABILITIES;
    type SimdMask<S: Simd> = T::SimdMask<S>;
    type SimdVec<S: Simd> = T::SimdComplexUnit<S>;
    type SimdIndex<S: Simd> = T::SimdIndex<S>;

    #[inline(always)]
    fn zero_impl(ctx: &Self::MathCtx) -> Self {
        Complex {
            re: T::zero_impl(ctx),
            im: T::zero_impl(ctx),
        }
    }

    #[inline(always)]
    fn add_impl(ctx: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        let ctx = Ctx::<Unit, T>::new(ctx);
        Complex::new(ctx.add(&lhs.re, &rhs.re), ctx.add(&lhs.im, &rhs.im))
    }
    #[inline(always)]
    fn sub_impl(ctx: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        let ctx = Ctx::<Unit, T>::new(ctx);
        Complex::new(ctx.sub(&lhs.re, &rhs.re), ctx.sub(&lhs.im, &rhs.im))
    }
    #[inline(always)]
    fn mul_impl(ctx: &Self::MathCtx, lhs: &Self, rhs: &Self) -> Self {
        help!(Unit);
        let ctx = Ctx::<Unit, T>::new(ctx);
        let left = ctx.mul(&lhs.re, &rhs.re);
        let right = ctx.mul(&lhs.im, &rhs.im);
        let re = ctx.sub(&left, &right);

        let left = ctx.mul(&lhs.re, &rhs.im);
        let right = ctx.mul(&lhs.im, &rhs.re);
        let im = ctx.add(&left, &right);
        Complex { re, im }
    }

    #[inline(always)]
    fn mul_real_impl(ctx: &Self::MathCtx, lhs: &Self, rhs: &T) -> Self {
        let ctx = Ctx::<Unit, T>::new(ctx);
        Complex::new(ctx.mul_real(&lhs.re, &rhs), ctx.mul_real(&lhs.im, &rhs))
    }

    #[inline(always)]
    fn mul_pow2_impl(ctx: &Self::MathCtx, lhs: &Self, rhs: &T) -> Self {
        let ctx = Ctx::<Unit, T>::new(ctx);
        Complex::new(ctx.mul_pow2(&lhs.re, &rhs), ctx.mul_pow2(&lhs.im, &rhs))
    }

    #[inline(always)]
    fn one_impl(ctx: &Self::MathCtx) -> Self {
        Complex {
            re: T::one_impl(ctx),
            im: T::zero_impl(ctx),
        }
    }

    #[inline(always)]
    fn real_part_impl(ctx: &Self::MathCtx, value: &Self) -> T {
        T::copy_impl(ctx, &value.re)
    }

    #[inline(always)]
    fn imag_part_impl(ctx: &Self::MathCtx, value: &Self) -> T {
        T::copy_impl(ctx, &value.im)
    }

    #[inline(always)]
    fn copy_impl(ctx: &Self::MathCtx, value: &Self) -> Self {
        let ctx = Ctx::<Unit, T>::new(ctx);
        Complex::new(ctx.copy(&value.re), ctx.copy(&value.im))
    }

    #[inline(always)]
    fn neg_impl(ctx: &Self::MathCtx, value: &Self) -> Self {
        let ctx = Ctx::<Unit, T>::new(ctx);
        Complex::new(ctx.neg(&value.re), ctx.neg(&value.im))
    }

    #[inline(always)]
    fn conj_impl(ctx: &Self::MathCtx, value: &Self) -> Self {
        let ctx = Ctx::<Unit, T>::new(ctx);
        Complex::new(ctx.copy(&value.re), ctx.neg(&value.im))
    }

    #[inline(always)]
    fn abs1_impl(ctx: &Self::MathCtx, value: &Self) -> T {
        help!(Unit);

        let ctx = Ctx::<Unit, T>::new(ctx);

        let left = ctx.abs1(&value.re);
        let right = ctx.abs1(&value.im);
        ctx.add(&left, &right)
    }

    #[inline(always)]
    fn abs2_impl(ctx: &Self::MathCtx, value: &Self) -> T {
        help!(Unit);

        let ctx = Ctx::<Unit, T>::new(ctx);

        let left = ctx.abs2(&value.re);
        let right = ctx.abs2(&value.im);
        ctx.add(&left, &right)
    }

    fn abs_impl(ctx: &Self::MathCtx, value: &Self) -> T {
        abs_impl::<Unit, T>(ctx, &value.re, &value.im)
    }

    fn recip_impl(ctx: &Self::MathCtx, value: &Self) -> Self {
        let (re, im) = recip_impl::<Unit, T>(ctx, &value.re, &value.im);
        Complex { re, im }
    }

    #[inline(always)]
    fn nan_impl(ctx: &Self::MathCtx) -> Self {
        let ctx = Ctx::<Unit, T>::new(ctx);
        Complex {
            re: ctx.nan(),
            im: ctx.nan(),
        }
    }

    #[inline(always)]
    fn infinity_impl(ctx: &Self::MathCtx) -> Self {
        let ctx = Ctx::<Unit, T>::new(ctx);
        Complex {
            re: ctx.infinity(),
            im: ctx.infinity(),
        }
    }

    #[inline(always)]
    fn eq_impl(ctx: &Self::MathCtx, lhs: &Self, rhs: &Self) -> bool {
        let ctx = Ctx::<Unit, T>::new(ctx);
        ctx.eq(&lhs.re, &rhs.re) && ctx.eq(&lhs.im, &rhs.im)
    }

    fn sqrt_impl(ctx: &Self::MathCtx, value: &Self) -> Self {
        let (re, im) = sqrt_impl::<Unit, T>(ctx, &value.re, &value.im);
        Complex { re, im }
    }

    #[inline(always)]
    fn is_zero_impl(ctx: &Self::MathCtx, value: &Self) -> bool {
        let ctx = Ctx::<Unit, T>::new(ctx);
        ctx.is_zero(&value.re) && ctx.is_zero(&value.im)
    }

    #[inline(always)]
    fn from_real_impl(ctx: &Self::MathCtx, real: <Unit as Container>::Of<&Self::RealUnit>) -> Self {
        let ctx = Ctx::<Unit, T>::new(ctx);
        Complex {
            re: ctx.copy(real),
            im: ctx.zero(),
        }
    }

    #[inline(always)]
    fn from_f64_impl(ctx: &Self::MathCtx, real: f64) -> Self {
        let ctx = Ctx::<Unit, T>::new(ctx);
        Complex {
            re: ctx.from_f64(&real),
            im: ctx.zero(),
        }
    }

    #[inline(always)]
    fn is_finite_impl(ctx: &Self::MathCtx, value: &Self) -> bool {
        let ctx = Ctx::<Unit, T>::new(ctx);
        ctx.is_finite(&value.re) && ctx.is_finite(&value.im)
    }

    #[inline(always)]
    fn is_nan_impl(ctx: &Self::MathCtx, value: &Self) -> bool {
        let ctx = Ctx::<Unit, T>::new(ctx);
        ctx.is_nan(&value.re) || ctx.is_nan(&value.im)
    }

    #[inline(always)]
    fn simd_ctx<S: Simd>(ctx: &Self::MathCtx, simd: S) -> Self::SimdCtx<S> {
        T::simd_ctx(ctx, simd)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: &Self) -> T::SimdComplexUnit<S> {
        T::simd_complex_splat(ctx, value)
    }

    #[inline(always)]
    fn simd_splat_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<&Self::RealUnit>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_splat_real(ctx, value)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_add(ctx, lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_sub(ctx, lhs, rhs)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_neg(ctx, value)
    }

    #[inline(always)]
    fn simd_conj<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_conj(ctx, value)
    }

    #[inline(always)]
    fn simd_abs1<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_abs1(ctx, value)
    }

    #[inline(always)]
    fn simd_mul_real<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_mul_real(ctx, lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_mul_pow2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_mul_pow2(ctx, lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_mul(ctx, lhs, rhs)
    }

    #[inline(always)]
    fn simd_conj_mul<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_conj_mul(ctx, lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
        acc: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_mul_add(ctx, lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_conj_mul_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
        acc: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_conj_mul_add(ctx, lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_abs2<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_abs2(ctx, value)
    }

    #[inline(always)]
    fn simd_abs2_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: T::SimdComplexUnit<S>,
        acc: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_abs2_add(ctx, value, acc)
    }

    #[inline(always)]
    fn simd_reduce_sum<S: Simd>(ctx: &Self::SimdCtx<S>, value: T::SimdComplexUnit<S>) -> Self {
        T::simd_complex_reduce_sum(ctx, value)
    }
    #[inline(always)]
    fn simd_reduce_max<S: Simd>(ctx: &Self::SimdCtx<S>, value: T::SimdComplexUnit<S>) -> T {
        T::simd_complex_reduce_max(ctx, value)
    }

    #[inline(always)]
    fn simd_less_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        T::simd_complex_less_than(ctx, real_lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_greater_than<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        T::simd_complex_greater_than(ctx, real_lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_less_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        T::simd_complex_less_than_or_equal(ctx, real_lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_greater_than_or_equal<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        real_lhs: T::SimdComplexUnit<S>,
        real_rhs: T::SimdComplexUnit<S>,
    ) -> Self::SimdMask<S> {
        T::simd_complex_greater_than_or_equal(ctx, real_lhs, real_rhs)
    }

    #[inline(always)]
    fn simd_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: T::SimdComplexUnit<S>,
        rhs: T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_select(ctx, mask, lhs, rhs)
    }

    #[inline(always)]
    fn simd_index_select<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        let ctx = SimdCtx::<Unit, T, S>::new(ctx);
        ctx.iselect(mask, lhs, rhs)
    }

    #[inline(always)]
    fn simd_index_splat<S: Simd>(ctx: &Self::SimdCtx<S>, value: Self::Index) -> Self::SimdIndex<S> {
        let ctx = SimdCtx::<Unit, T, S>::new(ctx);
        ctx.isplat(value)
    }

    #[inline(always)]
    fn simd_index_add<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        lhs: Self::SimdIndex<S>,
        rhs: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        let ctx = SimdCtx::<Unit, T, S>::new(ctx);
        ctx.iadd(lhs, rhs)
    }

    #[inline(always)]
    fn simd_tail_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S> {
        let ctx = SimdCtx::<Unit, T, S>::new(ctx);
        ctx.tail_mask(2 * len)
    }
    #[inline(always)]
    fn simd_head_mask<S: Simd>(ctx: &Self::SimdCtx<S>, len: usize) -> Self::SimdMask<S> {
        let ctx = SimdCtx::<Unit, T, S>::new(ctx);
        ctx.head_mask(2 * len)
    }

    #[inline(always)]
    unsafe fn simd_mask_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *const T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_mask_load(ctx, mask, ptr)
    }
    #[inline(always)]
    unsafe fn simd_mask_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        mask: Self::SimdMask<S>,
        ptr: *mut T::SimdComplexUnit<S>,
        value: T::SimdComplexUnit<S>,
    ) {
        T::simd_complex_mask_store(ctx, mask, ptr, value)
    }

    #[inline(always)]
    fn simd_load<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: &T::SimdComplexUnit<S>,
    ) -> T::SimdComplexUnit<S> {
        T::simd_complex_load(ctx, ptr)
    }
    #[inline(always)]
    fn simd_store<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        ptr: &mut T::SimdComplexUnit<S>,
        value: T::SimdComplexUnit<S>,
    ) {
        T::simd_complex_store(ctx, ptr, value)
    }

    #[inline(always)]
    fn simd_iota<S: Simd>(ctx: &Self::SimdCtx<S>) -> Self::SimdIndex<S> {
        T::simd_complex_iota(ctx)
    }

    #[inline(always)]
    fn simd_abs_max<S: Simd>(
        ctx: &Self::SimdCtx<S>,
        value: <Unit as Container>::Of<Self::SimdVec<S>>,
    ) -> <Unit as Container>::Of<Self::SimdVec<S>> {
        T::simd_complex_abs_max(ctx, value)
    }

    #[inline(always)]
    fn ctx_from_simd<S: Simd>(ctx: &Self::SimdCtx<S>) -> (Self::MathCtx, S) {
        T::ctx_from_simd(ctx)
    }

    #[inline(always)]
    fn simd_and_mask<S: Simd>(
        simd: &Self::SimdCtx<S>,
        lhs: Self::SimdMask<S>,
        rhs: Self::SimdMask<S>,
    ) -> Self::SimdMask<S> {
        T::simd_and_mask(simd, lhs, rhs)
    }

    #[inline(always)]
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
