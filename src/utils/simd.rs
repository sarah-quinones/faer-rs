use super::slice::*;
use crate::Conj;
use core::fmt::Debug;
use core::marker::PhantomData;
use faer_entity::*;
use reborrow::*;

pub use faer_entity::pulp::{Read, Write};

/// Do conjugate.
#[derive(Copy, Clone, Debug)]
pub struct YesConj;
/// Do not conjugate.
#[derive(Copy, Clone, Debug)]
pub struct NoConj;

/// Similar to [`Conj`], but determined at compile time instead of runtime.
pub trait ConjTy: Copy + Debug {
    /// The corresponding [`Conj`] value.
    const CONJ: Conj;
    /// The opposing conjugation type.
    type Flip: ConjTy;

    /// Returns an instance of the corresponding conjugation type.
    fn flip(self) -> Self::Flip;
}

impl ConjTy for YesConj {
    const CONJ: Conj = Conj::Yes;
    type Flip = NoConj;
    #[inline(always)]
    fn flip(self) -> Self::Flip {
        NoConj
    }
}
impl ConjTy for NoConj {
    const CONJ: Conj = Conj::No;
    type Flip = YesConj;
    #[inline(always)]
    fn flip(self) -> Self::Flip {
        YesConj
    }
}

/// Wrapper for simd operations for type `E`.
pub struct SimdFor<E: Entity, S: pulp::Simd> {
    /// Simd token.
    pub simd: S,
    __marker: PhantomData<E>,
}

/// Simd prefix, contains the elements before the body.
pub struct Prefix<'a, E: Entity, S: pulp::Simd>(
    GroupCopyFor<E, E::PrefixUnit<'static, S>>,
    PhantomData<&'a ()>,
);
/// Simd suffix, contains the elements after the body.
pub struct Suffix<'a, E: Entity, S: pulp::Simd>(
    GroupCopyFor<E, E::SuffixUnit<'static, S>>,
    PhantomData<&'a mut ()>,
);
/// Simd prefix (mutable), contains the elements before the body.
pub struct PrefixMut<'a, E: Entity, S: pulp::Simd>(
    GroupFor<E, E::PrefixMutUnit<'static, S>>,
    PhantomData<&'a ()>,
);
/// Simd suffix (mutable), contains the elements after the body.
pub struct SuffixMut<'a, E: Entity, S: pulp::Simd>(
    GroupFor<E, E::SuffixMutUnit<'static, S>>,
    PhantomData<&'a mut ()>,
);

impl<E: Entity, S: pulp::Simd> Copy for SimdFor<E, S> {}
impl<E: Entity, S: pulp::Simd> Clone for SimdFor<E, S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: ComplexField, S: pulp::Simd> SimdFor<E, S> {
    /// Create a new wrapper from a simd token.
    #[inline(always)]
    pub fn new(simd: S) -> Self {
        Self {
            simd,
            __marker: PhantomData,
        }
    }

    /// Computes the alignment offset for subsequent aligned loads.
    #[inline(always)]
    pub fn align_offset(self, slice: SliceGroup<'_, E>) -> pulp::Offset<E::SimdMask<S>> {
        let slice = E::faer_first(slice.into_inner());
        E::faer_align_offset(self.simd, slice.as_ptr(), slice.len())
    }

    /// Computes the alignment offset for subsequent aligned loads from a pointer.
    #[inline(always)]
    pub fn align_offset_ptr(
        self,
        ptr: GroupFor<E, *const E::Unit>,
        len: usize,
    ) -> pulp::Offset<E::SimdMask<S>> {
        E::faer_align_offset(self.simd, E::faer_first(ptr), len)
    }

    /// Convert a slice to a slice over vector registers, and a scalar tail.
    #[inline(always)]
    pub fn as_simd(
        self,
        slice: SliceGroup<'_, E>,
    ) -> (SliceGroup<'_, E, SimdUnitFor<E, S>>, SliceGroup<'_, E>) {
        let (head, tail) = slice_as_simd::<E, S>(slice.into_inner());
        (SliceGroup::new(head), SliceGroup::new(tail))
    }

    /// Convert a mutable slice to a slice over vector registers, and a scalar tail.
    #[inline(always)]
    pub fn as_simd_mut(
        self,
        slice: SliceGroupMut<'_, E>,
    ) -> (
        SliceGroupMut<'_, E, SimdUnitFor<E, S>>,
        SliceGroupMut<'_, E>,
    ) {
        let (head, tail) = slice_as_mut_simd::<E, S>(slice.into_inner());
        (SliceGroupMut::new(head), SliceGroupMut::new(tail))
    }

    /// Convert a slice to a partial register prefix and suffix, and a vector register slice
    /// (body).
    #[inline(always)]
    pub fn as_aligned_simd(
        self,
        slice: SliceGroup<'_, E>,
        offset: pulp::Offset<E::SimdMask<S>>,
    ) -> (
        Prefix<'_, E, S>,
        SliceGroup<'_, E, SimdUnitFor<E, S>>,
        Suffix<'_, E, S>,
    ) {
        let (head_tail, body) = E::faer_unzip(E::faer_map(slice.into_inner(), |slice| {
            let (head, body, tail) = E::faer_slice_as_aligned_simd(self.simd, slice, offset);
            ((head, tail), body)
        }));

        let (head, tail) = E::faer_unzip(head_tail);

        unsafe {
            (
                Prefix(
                    transmute_unchecked::<
                        GroupCopyFor<E, E::PrefixUnit<'_, S>>,
                        GroupCopyFor<E, E::PrefixUnit<'static, S>>,
                    >(into_copy::<E, _>(head)),
                    PhantomData,
                ),
                SliceGroup::new(body),
                Suffix(
                    transmute_unchecked::<
                        GroupCopyFor<E, E::SuffixUnit<'_, S>>,
                        GroupCopyFor<E, E::SuffixUnit<'static, S>>,
                    >(into_copy::<E, _>(tail)),
                    PhantomData,
                ),
            )
        }
    }

    /// Convert a mutable slice to a partial register prefix and suffix, and a vector register
    /// slice (body).
    #[inline(always)]
    pub fn as_aligned_simd_mut(
        self,
        slice: SliceGroupMut<'_, E>,
        offset: pulp::Offset<E::SimdMask<S>>,
    ) -> (
        PrefixMut<'_, E, S>,
        SliceGroupMut<'_, E, SimdUnitFor<E, S>>,
        SuffixMut<'_, E, S>,
    ) {
        let (head_tail, body) = E::faer_unzip(E::faer_map(slice.into_inner(), |slice| {
            let (head, body, tail) = E::faer_slice_as_aligned_simd_mut(self.simd, slice, offset);
            ((head, tail), body)
        }));

        let (head, tail) = E::faer_unzip(head_tail);

        (
            PrefixMut(
                unsafe {
                    transmute_unchecked::<
                        GroupFor<E, E::PrefixMutUnit<'_, S>>,
                        GroupFor<E, E::PrefixMutUnit<'static, S>>,
                    >(head)
                },
                PhantomData,
            ),
            SliceGroupMut::new(body),
            SuffixMut(
                unsafe {
                    transmute_unchecked::<
                        GroupFor<E, E::SuffixMutUnit<'_, S>>,
                        GroupFor<E, E::SuffixMutUnit<'static, S>>,
                    >(tail)
                },
                PhantomData,
            ),
        )
    }

    /// Fill all the register lanes with the same value.
    #[inline(always)]
    pub fn splat(self, value: E) -> SimdGroupFor<E, S> {
        E::faer_simd_splat(self.simd, value)
    }

    /// Returns `lhs * rhs`.
    #[inline(always)]
    pub fn scalar_mul(self, lhs: E, rhs: E) -> E {
        E::faer_simd_scalar_mul(self.simd, lhs, rhs)
    }
    /// Returns `conj(lhs) * rhs`.
    #[inline(always)]
    pub fn scalar_conj_mul(self, lhs: E, rhs: E) -> E {
        E::faer_simd_scalar_conj_mul(self.simd, lhs, rhs)
    }
    /// Returns an estimate of `lhs * rhs + acc`.
    #[inline(always)]
    pub fn scalar_mul_add_e(self, lhs: E, rhs: E, acc: E) -> E {
        E::faer_simd_scalar_mul_adde(self.simd, lhs, rhs, acc)
    }
    /// Returns an estimate of `conj(lhs) * rhs + acc`.
    #[inline(always)]
    pub fn scalar_conj_mul_add_e(self, lhs: E, rhs: E, acc: E) -> E {
        E::faer_simd_scalar_conj_mul_adde(self.simd, lhs, rhs, acc)
    }

    /// Returns an estimate of `op(lhs) * rhs`, where `op` is either the conjugation
    /// or the identity operation.
    #[inline(always)]
    pub fn scalar_conditional_conj_mul<C: ConjTy>(self, conj: C, lhs: E, rhs: E) -> E {
        let _ = conj;
        if C::CONJ == Conj::Yes {
            self.scalar_conj_mul(lhs, rhs)
        } else {
            self.scalar_mul(lhs, rhs)
        }
    }
    /// Returns an estimate of `op(lhs) * rhs + acc`, where `op` is either the conjugation or
    /// the identity operation.
    #[inline(always)]
    pub fn scalar_conditional_conj_mul_add_e<C: ConjTy>(
        self,
        conj: C,
        lhs: E,
        rhs: E,
        acc: E,
    ) -> E {
        let _ = conj;
        if C::CONJ == Conj::Yes {
            self.scalar_conj_mul_add_e(lhs, rhs, acc)
        } else {
            self.scalar_mul_add_e(lhs, rhs, acc)
        }
    }

    /// Returns `lhs + rhs`.
    #[inline(always)]
    pub fn add(self, lhs: SimdGroupFor<E, S>, rhs: SimdGroupFor<E, S>) -> SimdGroupFor<E, S> {
        E::faer_simd_add(self.simd, lhs, rhs)
    }
    /// Returns `lhs - rhs`.
    #[inline(always)]
    pub fn sub(self, lhs: SimdGroupFor<E, S>, rhs: SimdGroupFor<E, S>) -> SimdGroupFor<E, S> {
        E::faer_simd_sub(self.simd, lhs, rhs)
    }
    /// Returns `-a`.
    #[inline(always)]
    pub fn neg(self, a: SimdGroupFor<E, S>) -> SimdGroupFor<E, S> {
        E::faer_simd_neg(self.simd, a)
    }
    /// Returns `lhs * rhs`.
    #[inline(always)]
    pub fn scale_real(
        self,
        lhs: SimdGroupFor<E::Real, S>,
        rhs: SimdGroupFor<E, S>,
    ) -> SimdGroupFor<E, S> {
        E::faer_simd_scale_real(self.simd, lhs, rhs)
    }
    /// Returns `lhs * rhs`.
    #[inline(always)]
    pub fn mul(self, lhs: SimdGroupFor<E, S>, rhs: SimdGroupFor<E, S>) -> SimdGroupFor<E, S> {
        E::faer_simd_mul(self.simd, lhs, rhs)
    }
    /// Returns `conj(lhs) * rhs`.
    #[inline(always)]
    pub fn conj_mul(self, lhs: SimdGroupFor<E, S>, rhs: SimdGroupFor<E, S>) -> SimdGroupFor<E, S> {
        E::faer_simd_conj_mul(self.simd, lhs, rhs)
    }
    /// Returns `op(lhs) * rhs`, where `op` is either the conjugation or the identity
    /// operation.
    #[inline(always)]
    pub fn conditional_conj_mul<C: ConjTy>(
        self,
        conj: C,
        lhs: SimdGroupFor<E, S>,
        rhs: SimdGroupFor<E, S>,
    ) -> SimdGroupFor<E, S> {
        let _ = conj;
        if C::CONJ == Conj::Yes {
            self.conj_mul(lhs, rhs)
        } else {
            self.mul(lhs, rhs)
        }
    }

    /// Returns `lhs * rhs + acc`.
    #[inline(always)]
    pub fn mul_add_e(
        self,
        lhs: SimdGroupFor<E, S>,
        rhs: SimdGroupFor<E, S>,
        acc: SimdGroupFor<E, S>,
    ) -> SimdGroupFor<E, S> {
        E::faer_simd_mul_adde(self.simd, lhs, rhs, acc)
    }
    /// Returns `conj(lhs) * rhs + acc`.
    #[inline(always)]
    pub fn conj_mul_add_e(
        self,
        lhs: SimdGroupFor<E, S>,
        rhs: SimdGroupFor<E, S>,
        acc: SimdGroupFor<E, S>,
    ) -> SimdGroupFor<E, S> {
        E::faer_simd_conj_mul_adde(self.simd, lhs, rhs, acc)
    }
    /// Returns `op(lhs) * rhs + acc`, where `op` is either the conjugation or the identity
    /// operation.
    #[inline(always)]
    pub fn conditional_conj_mul_add_e<C: ConjTy>(
        self,
        conj: C,
        lhs: SimdGroupFor<E, S>,
        rhs: SimdGroupFor<E, S>,
        acc: SimdGroupFor<E, S>,
    ) -> SimdGroupFor<E, S> {
        let _ = conj;
        if C::CONJ == Conj::Yes {
            self.conj_mul_add_e(lhs, rhs, acc)
        } else {
            self.mul_add_e(lhs, rhs, acc)
        }
    }

    /// Returns `abs(values) * abs(values) + acc`.
    #[inline(always)]
    pub fn abs2_add_e(
        self,
        values: SimdGroupFor<E, S>,
        acc: SimdGroupFor<E::Real, S>,
    ) -> SimdGroupFor<E::Real, S> {
        E::faer_simd_abs2_adde(self.simd, values, acc)
    }
    /// Returns `abs(values) * abs(values)`.
    #[inline(always)]
    pub fn abs2(self, values: SimdGroupFor<E, S>) -> SimdGroupFor<E::Real, S> {
        E::faer_simd_abs2(self.simd, values)
    }
    /// Returns `abs(values)` or `abs(values) * abs(values)`, whichever is cheaper to compute.
    #[inline(always)]
    pub fn score(self, values: SimdGroupFor<E, S>) -> SimdGroupFor<E::Real, S> {
        E::faer_simd_score(self.simd, values)
    }

    /// Sum the components of a vector register into a single accumulator.
    #[inline(always)]
    pub fn reduce_add(self, values: SimdGroupFor<E, S>) -> E {
        E::faer_simd_reduce_add(self.simd, values)
    }

    /// Rotate `values` to the left, with overflowing entries wrapping around to the right side
    /// of the register.
    #[inline(always)]
    pub fn rotate_left(self, values: SimdGroupFor<E, S>, amount: usize) -> SimdGroupFor<E, S> {
        E::faer_simd_rotate_left(self.simd, values, amount)
    }
}

impl<E: RealField, S: pulp::Simd> SimdFor<E, S> {
    /// Returns `abs(values)`.
    #[inline(always)]
    pub fn abs(self, values: SimdGroupFor<E, S>) -> SimdGroupFor<E::Real, S> {
        E::faer_simd_abs(self.simd, values)
    }
    /// Returns `a < b`.
    #[inline(always)]
    pub fn less_than(self, a: SimdGroupFor<E, S>, b: SimdGroupFor<E, S>) -> SimdMaskFor<E, S> {
        E::faer_simd_less_than(self.simd, a, b)
    }
    /// Returns `a <= b`.
    #[inline(always)]
    pub fn less_than_or_equal(
        self,
        a: SimdGroupFor<E, S>,
        b: SimdGroupFor<E, S>,
    ) -> SimdMaskFor<E, S> {
        E::faer_simd_less_than_or_equal(self.simd, a, b)
    }
    /// Returns `a > b`.
    #[inline(always)]
    pub fn greater_than(self, a: SimdGroupFor<E, S>, b: SimdGroupFor<E, S>) -> SimdMaskFor<E, S> {
        E::faer_simd_greater_than(self.simd, a, b)
    }
    /// Returns `a >= b`.
    #[inline(always)]
    pub fn greater_than_or_equal(
        self,
        a: SimdGroupFor<E, S>,
        b: SimdGroupFor<E, S>,
    ) -> SimdMaskFor<E, S> {
        E::faer_simd_greater_than_or_equal(self.simd, a, b)
    }

    /// Returns `if mask { if_true } else { if_false }`
    #[inline(always)]
    pub fn select(
        self,
        mask: SimdMaskFor<E, S>,
        if_true: SimdGroupFor<E, S>,
        if_false: SimdGroupFor<E, S>,
    ) -> SimdGroupFor<E, S> {
        E::faer_simd_select(self.simd, mask, if_true, if_false)
    }
    /// Returns `if mask { if_true } else { if_false }`
    #[inline(always)]
    pub fn index_select(
        self,
        mask: SimdMaskFor<E, S>,
        if_true: SimdIndexFor<E, S>,
        if_false: SimdIndexFor<E, S>,
    ) -> SimdIndexFor<E, S> {
        E::faer_simd_index_select(self.simd, mask, if_true, if_false)
    }
    /// Returns `[0, 1, 2, 3, ..., REGISTER_SIZE - 1]`
    #[inline(always)]
    pub fn index_seq(self) -> SimdIndexFor<E, S> {
        E::faer_simd_index_seq(self.simd)
    }
    /// Fill all the register lanes with the same value.
    #[inline(always)]
    pub fn index_splat(self, value: IndexFor<E>) -> SimdIndexFor<E, S> {
        E::faer_simd_index_splat(self.simd, value)
    }
    /// Returns `a + b`.
    #[inline(always)]
    pub fn index_add(self, a: SimdIndexFor<E, S>, b: SimdIndexFor<E, S>) -> SimdIndexFor<E, S> {
        E::faer_simd_index_add(self.simd, a, b)
    }
}
impl<E: Entity, S: pulp::Simd> Read for Prefix<'_, E, S> {
    type Output = SimdGroupFor<E, S>;
    #[inline(always)]
    fn read_or(&self, or: Self::Output) -> Self::Output {
        into_copy::<E, _>(E::faer_map(
            E::faer_zip(from_copy::<E, _>(self.0), from_copy::<E, _>(or)),
            #[inline(always)]
            |(prefix, or)| prefix.read_or(or),
        ))
    }
}
impl<E: Entity, S: pulp::Simd> Read for PrefixMut<'_, E, S> {
    type Output = SimdGroupFor<E, S>;
    #[inline(always)]
    fn read_or(&self, or: Self::Output) -> Self::Output {
        self.rb().read_or(or)
    }
}
impl<E: Entity, S: pulp::Simd> Write for PrefixMut<'_, E, S> {
    #[inline(always)]
    fn write(&mut self, values: Self::Output) {
        E::faer_map(
            E::faer_zip(self.rb_mut().0, from_copy::<E, _>(values)),
            #[inline(always)]
            |(mut prefix, values)| prefix.write(values),
        );
    }
}

impl<E: Entity, S: pulp::Simd> Read for Suffix<'_, E, S> {
    type Output = SimdGroupFor<E, S>;
    #[inline(always)]
    fn read_or(&self, or: Self::Output) -> Self::Output {
        into_copy::<E, _>(E::faer_map(
            E::faer_zip(from_copy::<E, _>(self.0), from_copy::<E, _>(or)),
            #[inline(always)]
            |(suffix, or)| suffix.read_or(or),
        ))
    }
}
impl<E: Entity, S: pulp::Simd> Read for SuffixMut<'_, E, S> {
    type Output = SimdGroupFor<E, S>;
    #[inline(always)]
    fn read_or(&self, or: Self::Output) -> Self::Output {
        self.rb().read_or(or)
    }
}
impl<E: Entity, S: pulp::Simd> Write for SuffixMut<'_, E, S> {
    #[inline(always)]
    fn write(&mut self, values: Self::Output) {
        E::faer_map(
            E::faer_zip(self.rb_mut().0, from_copy::<E, _>(values)),
            #[inline(always)]
            |(mut suffix, values)| suffix.write(values),
        );
    }
}

impl<'short, E: Entity, S: pulp::Simd> Reborrow<'short> for PrefixMut<'_, E, S> {
    type Target = Prefix<'short, E, S>;
    #[inline]
    fn rb(&'short self) -> Self::Target {
        unsafe {
            Prefix(
                into_copy::<E, _>(transmute_unchecked::<
                    GroupFor<E, <E::PrefixMutUnit<'static, S> as Reborrow<'_>>::Target>,
                    GroupFor<E, E::PrefixUnit<'static, S>>,
                >(E::faer_map(
                    E::faer_as_ref(&self.0),
                    |x| (*x).rb(),
                ))),
                PhantomData,
            )
        }
    }
}
impl<'short, E: Entity, S: pulp::Simd> ReborrowMut<'short> for PrefixMut<'_, E, S> {
    type Target = PrefixMut<'short, E, S>;
    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        unsafe {
            PrefixMut(
                transmute_unchecked::<
                    GroupFor<E, <E::PrefixMutUnit<'static, S> as ReborrowMut<'_>>::Target>,
                    GroupFor<E, E::PrefixMutUnit<'static, S>>,
                >(E::faer_map(E::faer_as_mut(&mut self.0), |x| (*x).rb_mut())),
                PhantomData,
            )
        }
    }
}
impl<'short, E: Entity, S: pulp::Simd> Reborrow<'short> for SuffixMut<'_, E, S> {
    type Target = Suffix<'short, E, S>;
    #[inline]
    fn rb(&'short self) -> Self::Target {
        unsafe {
            Suffix(
                into_copy::<E, _>(transmute_unchecked::<
                    GroupFor<E, <E::SuffixMutUnit<'static, S> as Reborrow<'_>>::Target>,
                    GroupFor<E, E::SuffixUnit<'static, S>>,
                >(E::faer_map(
                    E::faer_as_ref(&self.0),
                    |x| (*x).rb(),
                ))),
                PhantomData,
            )
        }
    }
}
impl<'short, E: Entity, S: pulp::Simd> ReborrowMut<'short> for SuffixMut<'_, E, S> {
    type Target = SuffixMut<'short, E, S>;
    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        unsafe {
            SuffixMut(
                transmute_unchecked::<
                    GroupFor<E, <E::SuffixMutUnit<'static, S> as ReborrowMut<'_>>::Target>,
                    GroupFor<E, E::SuffixMutUnit<'static, S>>,
                >(E::faer_map(E::faer_as_mut(&mut self.0), |x| (*x).rb_mut())),
                PhantomData,
            )
        }
    }
}

impl<'short, E: Entity, S: pulp::Simd> Reborrow<'short> for Prefix<'_, E, S> {
    type Target = Prefix<'short, E, S>;
    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short, E: Entity, S: pulp::Simd> ReborrowMut<'short> for Prefix<'_, E, S> {
    type Target = Prefix<'short, E, S>;
    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}
impl<'short, E: Entity, S: pulp::Simd> Reborrow<'short> for Suffix<'_, E, S> {
    type Target = Suffix<'short, E, S>;
    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short, E: Entity, S: pulp::Simd> ReborrowMut<'short> for Suffix<'_, E, S> {
    type Target = Suffix<'short, E, S>;
    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<E: Entity, S: pulp::Simd> Copy for Prefix<'_, E, S> {}
impl<E: Entity, S: pulp::Simd> Clone for Prefix<'_, E, S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Entity, S: pulp::Simd> Copy for Suffix<'_, E, S> {}
impl<E: Entity, S: pulp::Simd> Clone for Suffix<'_, E, S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Entity, S: pulp::Simd> core::fmt::Debug for Prefix<'_, E, S> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        unsafe {
            transmute_unchecked::<SimdGroupFor<E, S>, GroupDebugFor<E, SimdUnitFor<E, S>>>(
                self.read_or(core::mem::zeroed()),
            )
            .fmt(f)
        }
    }
}
impl<E: Entity, S: pulp::Simd> core::fmt::Debug for PrefixMut<'_, E, S> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}
impl<E: Entity, S: pulp::Simd> core::fmt::Debug for Suffix<'_, E, S> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        unsafe {
            transmute_unchecked::<SimdGroupFor<E, S>, GroupDebugFor<E, SimdUnitFor<E, S>>>(
                self.read_or(core::mem::zeroed()),
            )
            .fmt(f)
        }
    }
}
impl<E: Entity, S: pulp::Simd> core::fmt::Debug for SuffixMut<'_, E, S> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}
