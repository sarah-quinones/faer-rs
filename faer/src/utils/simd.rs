use core::marker::PhantomData;
use faer_traits::{help, ComplexContainer, ComplexField, SimdCapabilities};
use pulp::Simd;

use super::bound::{Array, Dim, Idx};

pub struct SimdCtx<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> {
    pub ctx: T::SimdCtx<S>,
    pub len: Dim<'N>,
    simd_len: usize,
    mask: T::SimdMask<S>,
}

impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> core::fmt::Debug
    for SimdCtx<'N, C, T, S>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SimdCtx")
            .field("len", &self.len)
            .field("simd_len", &self.simd_len)
            .field("mask", &self.mask)
            .finish()
    }
}

impl<C: ComplexContainer, T: ComplexField<C>, S: Simd> Copy for SimdCtx<'_, C, T, S> {}
impl<C: ComplexContainer, T: ComplexField<C>, S: Simd> Clone for SimdCtx<'_, C, T, S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: ComplexContainer, T: ComplexField<C>, S: Simd> core::ops::Deref for SimdCtx<'_, C, T, S> {
    type Target = faer_traits::SimdCtxCopy<C, T, S>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        Self::Target::new(&self.ctx)
    }
}

impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> SimdCtx<'N, C, T, S> {
    #[inline]
    pub fn new(simd: T::SimdCtx<S>, len: Dim<'N>) -> Self {
        let stride = const { size_of::<T::SimdVec<S>>() / size_of::<T>() };
        Self {
            ctx: simd,
            len,
            simd_len: *len / stride * stride,
            mask: T::simd_tail_mask(&simd, *len % stride),
        }
    }

    #[inline]
    pub fn new_force_mask(simd: T::SimdCtx<S>, len: Dim<'N>) -> Self {
        crate::assert!(*len != 0);
        let new_len = *len - 1;

        let stride = const { size_of::<T::SimdVec<S>>() / size_of::<T>() };
        Self {
            ctx: simd,
            len,
            simd_len: new_len / stride * stride,
            mask: T::simd_tail_mask(&simd, (new_len % stride) + 1),
        }
    }

    #[inline(always)]
    pub fn read(
        &self,
        slice: C::Of<&Array<'N, T>>,
        index: SimdIdx<'N, C, T, S>,
    ) -> C::OfSimd<T::SimdVec<S>> {
        core::assert!(
            const {
                matches!(
                    T::SIMD_CAPABILITIES,
                    SimdCapabilities::All | SimdCapabilities::Shuffled
                )
            }
        );

        help!(C);
        unsafe {
            self.load(map!(
                slice,
                slice,
                &*(slice.as_ref().as_ptr().add(index.start.unbound()) as *const T::SimdVec<S>)
            ))
        }
    }

    #[inline(always)]
    pub fn has_tail(&self) -> bool {
        self.len.unbound() > self.simd_len
    }

    #[inline(always)]
    pub fn read_tail(&self, slice: C::Of<&Array<'N, T>>) -> C::OfSimd<T::SimdVec<S>> {
        core::assert!(
            const {
                matches!(
                    T::SIMD_CAPABILITIES,
                    SimdCapabilities::All | SimdCapabilities::Shuffled
                )
            }
        );
        debug_assert!(self.has_tail());

        help!(C);
        unsafe {
            self.mask_load(
                self.mask,
                map!(
                    slice,
                    slice,
                    slice.as_ref().as_ptr().add(self.simd_len) as *const T::SimdVec<S>
                ),
            )
        }
    }

    #[inline(always)]
    pub fn write_tail(&self, slice: C::Of<&mut Array<'N, T>>, value: C::OfSimd<T::SimdVec<S>>) {
        core::assert!(
            const {
                matches!(
                    T::SIMD_CAPABILITIES,
                    SimdCapabilities::All | SimdCapabilities::Shuffled
                )
            }
        );
        debug_assert!(self.has_tail());

        help!(C);
        unsafe {
            self.mask_store(
                self.mask,
                map!(
                    slice,
                    slice,
                    slice.as_mut().as_mut_ptr().add(self.simd_len) as *mut T::SimdVec<S>
                ),
                value,
            );
        }
    }

    #[inline(always)]
    pub fn write(
        &self,
        slice: C::Of<&mut Array<'N, T>>,
        index: SimdIdx<'N, C, T, S>,
        value: C::OfSimd<T::SimdVec<S>>,
    ) {
        core::assert!(
            const {
                matches!(
                    T::SIMD_CAPABILITIES,
                    SimdCapabilities::All | SimdCapabilities::Shuffled
                )
            }
        );

        help!(C);
        unsafe {
            self.store(
                map!(
                    slice,
                    slice,
                    &mut *(slice.as_mut().as_mut_ptr().add(index.start.unbound())
                        as *mut T::SimdVec<S>)
                ),
                value,
            );
        }
    }

    #[inline]
    pub fn indices(
        &self,
    ) -> impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = SimdIdx<'N, C, T, S>> {
        macro_rules! stride {
            () => {
                const { size_of::<T::SimdVec<S>>() / size_of::<T>() }
            };
        }

        let stride = stride!();
        let len = self.simd_len;

        (0..len / stride).map(
            #[inline(always)]
            move |i| SimdIdx {
                start: unsafe { Idx::new_unbound(i * stride!()) },
                mask: PhantomData,
            },
        )
    }

    #[inline]
    pub fn batch_indices<const BATCH: usize>(
        &self,
    ) -> (
        impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = [SimdIdx<'N, C, T, S>; BATCH]>,
        impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = SimdIdx<'N, C, T, S>>,
    ) {
        const { core::assert!(BATCH.is_power_of_two()) };

        macro_rules! stride {
            () => {
                const { size_of::<T::SimdVec<S>>() / size_of::<T>() }
            };
        }

        let stride = stride!();
        let len = *self.len;

        (
            (0..len / (BATCH * stride)).map(|i| {
                core::array::from_fn(
                    #[inline(always)]
                    |k| SimdIdx {
                        start: unsafe { Idx::new_unbound((i * BATCH + k) * stride!()) },
                        mask: PhantomData,
                    },
                )
            }),
            ((len / stride) / BATCH * BATCH..len / stride).map(
                #[inline(always)]
                move |i| SimdIdx {
                    start: unsafe { Idx::new_unbound(i * stride!()) },
                    mask: PhantomData,
                },
            ),
        )
    }
}

#[repr(transparent)]
pub struct SimdIdx<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> {
    start: Idx<'N>,
    mask: PhantomData<T::SimdMask<S>>,
}

impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> Copy for SimdIdx<'N, C, T, S> {}
impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> Clone for SimdIdx<'N, C, T, S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
