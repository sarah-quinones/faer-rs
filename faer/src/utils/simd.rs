use crate::internal_prelude::*;
use core::marker::PhantomData;
use faer_traits::SimdCapabilities;
use pulp::Simd;

use super::bound::{Dim, Idx};

pub struct SimdCtx<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> {
    pub ctx: T::SimdCtx<S>,
    pub len: Dim<'N>,
    offset: usize,
    head_end: usize,
    body_end: usize,
    tail_end: usize,
    head_mask: T::SimdMask<S>,
    tail_mask: T::SimdMask<S>,
}

impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> core::fmt::Debug
    for SimdCtx<'N, C, T, S>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SimdCtx")
            .field("len", &self.len)
            .field("offset", &self.offset)
            .field("head_end", &self.head_end)
            .field("body_end", &self.body_end)
            .field("tail_end", &self.tail_end)
            .field("head_mask", &self.head_mask)
            .field("tail_mask", &self.tail_mask)
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

pub trait SimdIndex<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> {
    fn read(
        simd: &SimdCtx<'N, C, T, S>,
        slice: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
        index: Self,
    ) -> C::OfSimd<T::SimdVec<S>>;

    fn write(
        simd: &SimdCtx<'N, C, T, S>,
        slice: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
        index: Self,
        value: C::OfSimd<T::SimdVec<S>>,
    );
}

impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> SimdIndex<'N, C, T, S>
    for SimdBody<'N, C, T, S>
{
    #[inline(always)]
    fn read(
        simd: &SimdCtx<'N, C, T, S>,
        slice: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
        index: Self,
    ) -> <C>::OfSimd<T::SimdVec<S>> {
        help!(C);
        unsafe {
            simd.load(map!(
                slice.as_ptr(),
                slice,
                &*(slice.wrapping_offset(index.start) as *const T::SimdVec<S>)
            ))
        }
    }

    #[inline(always)]
    fn write(
        simd: &SimdCtx<'N, C, T, S>,
        slice: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
        index: Self,
        value: <C>::OfSimd<T::SimdVec<S>>,
    ) {
        help!(C);
        unsafe {
            simd.store(
                map!(
                    slice.as_ptr_mut(),
                    slice,
                    &mut *(slice.wrapping_offset(index.start) as *mut T::SimdVec<S>)
                ),
                value,
            );
        }
    }
}

impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> SimdIndex<'N, C, T, S>
    for SimdHead<'N, C, T, S>
{
    #[inline(always)]
    fn read(
        simd: &SimdCtx<'N, C, T, S>,
        slice: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
        index: Self,
    ) -> <C>::OfSimd<T::SimdVec<S>> {
        help!(C);
        unsafe {
            simd.mask_load(
                simd.head_mask,
                map!(
                    slice.as_ptr(),
                    slice,
                    (slice.wrapping_offset(index.start) as *const T::SimdVec<S>)
                ),
            )
        }
    }

    #[inline(always)]
    fn write(
        simd: &SimdCtx<'N, C, T, S>,
        slice: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
        index: Self,
        value: <C>::OfSimd<T::SimdVec<S>>,
    ) {
        help!(C);
        unsafe {
            simd.mask_store(
                simd.head_mask,
                map!(
                    slice.as_ptr_mut(),
                    slice,
                    slice.wrapping_offset(index.start) as *mut T::SimdVec<S>
                ),
                value,
            );
        }
    }
}

impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> SimdIndex<'N, C, T, S>
    for SimdTail<'N, C, T, S>
{
    #[inline(always)]
    fn read(
        simd: &SimdCtx<'N, C, T, S>,
        slice: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
        index: Self,
    ) -> <C>::OfSimd<T::SimdVec<S>> {
        help!(C);
        unsafe {
            simd.mask_load(
                simd.tail_mask,
                map!(
                    slice.as_ptr(),
                    slice,
                    (slice.wrapping_offset(index.start) as *const T::SimdVec<S>)
                ),
            )
        }
    }

    #[inline(always)]
    fn write(
        simd: &SimdCtx<'N, C, T, S>,
        slice: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
        index: Self,
        value: <C>::OfSimd<T::SimdVec<S>>,
    ) {
        help!(C);
        unsafe {
            simd.mask_store(
                simd.tail_mask,
                map!(
                    slice.as_ptr_mut(),
                    slice,
                    slice.wrapping_offset(index.start) as *mut T::SimdVec<S>
                ),
                value,
            );
        }
    }
}

impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> SimdCtx<'N, C, T, S> {
    #[inline]
    pub fn new(simd: T::SimdCtx<S>, len: Dim<'N>) -> Self {
        core::assert!(
            const {
                matches!(
                    T::SIMD_CAPABILITIES,
                    SimdCapabilities::All | SimdCapabilities::Shuffled
                )
            }
        );

        let stride = const { size_of::<T::SimdVec<S>>() / size_of::<T>() };
        Self {
            ctx: simd,
            len,
            offset: 0,
            head_end: 0,
            body_end: *len / stride,
            tail_end: (*len + stride - 1) / stride,
            head_mask: T::simd_head_mask(&simd, 0),
            tail_mask: T::simd_tail_mask(&simd, *len % stride),
        }
    }

    #[inline]
    pub fn new_align(simd: T::SimdCtx<S>, len: Dim<'N>, align_offset: usize) -> Self {
        core::assert!(
            const {
                matches!(
                    T::SIMD_CAPABILITIES,
                    SimdCapabilities::All | SimdCapabilities::Shuffled
                )
            }
        );

        let stride = const { size_of::<T::SimdVec<S>>() / size_of::<T>() };
        let align_offset = align_offset % stride;

        if align_offset == 0 {
            Self::new(simd, len)
        } else {
            let offset = stride - align_offset;
            let full_len = offset + *len;
            let head_mask = T::simd_head_mask(&simd, align_offset);
            let tail_mask = T::simd_tail_mask(&simd, full_len % stride);

            if align_offset <= *len {
                Self {
                    ctx: simd,
                    len,
                    offset,
                    head_end: 1,
                    body_end: full_len / stride,
                    tail_end: (full_len + stride - 1) / stride,
                    head_mask,
                    tail_mask,
                }
            } else {
                let head_mask = T::simd_and_mask(&simd, head_mask, tail_mask);
                let tail_mask = T::simd_tail_mask(&simd, 0);

                Self {
                    ctx: simd,
                    len,
                    offset,
                    head_end: 1,
                    body_end: 1,
                    tail_end: 1,
                    head_mask,
                    tail_mask,
                }
            }
        }
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn new_force_mask(simd: T::SimdCtx<S>, len: Dim<'N>) -> Self {
        core::assert!(
            const {
                matches!(
                    T::SIMD_CAPABILITIES,
                    SimdCapabilities::All | SimdCapabilities::Shuffled
                )
            }
        );

        crate::assert!(*len != 0);
        let new_len = *len - 1;

        let stride = const { size_of::<T::SimdVec<S>>() / size_of::<T>() };
        Self {
            ctx: simd,
            len,
            offset: 0,
            head_end: 0,
            body_end: new_len / stride,
            tail_end: new_len / stride + 1,
            head_mask: T::simd_head_mask(&simd, 0),
            tail_mask: T::simd_tail_mask(&simd, (new_len % stride) + 1),
        }
    }

    #[inline(always)]
    pub fn read<I: SimdIndex<'N, C, T, S>>(
        &self,
        slice: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
        index: I,
    ) -> C::OfSimd<T::SimdVec<S>> {
        I::read(self, slice, index)
    }

    #[inline(always)]
    pub fn write<I: SimdIndex<'N, C, T, S>>(
        &self,
        slice: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
        index: I,
        value: C::OfSimd<T::SimdVec<S>>,
    ) {
        I::write(self, slice, index, value)
    }

    #[inline]
    pub fn indices(
        &self,
    ) -> (
        Option<SimdHead<'N, C, T, S>>,
        impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = SimdBody<'N, C, T, S>>,
        Option<SimdTail<'N, C, T, S>>,
    ) {
        macro_rules! stride {
            () => {
                const { size_of::<T::SimdVec<S>>() / size_of::<T>() }
            };
        }

        let offset = -(self.offset as isize);
        (
            if 0 == self.head_end {
                None
            } else {
                Some(SimdHead {
                    start: offset,
                    mask: PhantomData,
                })
            },
            (self.head_end..self.body_end).map(
                #[inline(always)]
                move |i| SimdBody {
                    start: offset + (i * stride!()) as isize,
                    mask: PhantomData,
                },
            ),
            if self.body_end == self.tail_end {
                None
            } else {
                Some(SimdTail {
                    start: offset + (self.body_end * stride!()) as isize,
                    mask: PhantomData,
                })
            },
        )
    }

    #[inline]
    pub fn batch_indices<const BATCH: usize>(
        &self,
    ) -> (
        Option<SimdHead<'N, C, T, S>>,
        impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = [SimdBody<'N, C, T, S>; BATCH]>,
        impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = SimdBody<'N, C, T, S>>,
        Option<SimdTail<'N, C, T, S>>,
    ) {
        macro_rules! stride {
            () => {
                const { size_of::<T::SimdVec<S>>() / size_of::<T>() }
            };
        }

        let len = self.body_end - self.head_end;

        let offset = -(self.offset as isize);

        (
            if 0 == self.head_end {
                None
            } else {
                Some(SimdHead {
                    start: offset,
                    mask: PhantomData,
                })
            },
            (self.head_end..self.head_end + len / BATCH).map(move |i| {
                core::array::from_fn(
                    #[inline(always)]
                    |k| SimdBody {
                        start: offset + ((i * BATCH + k) * stride!()) as isize,
                        mask: PhantomData,
                    },
                )
            }),
            (self.head_end + len / BATCH * BATCH..self.body_end).map(
                #[inline(always)]
                move |i| SimdBody {
                    start: offset + (i * stride!()) as isize,
                    mask: PhantomData,
                },
            ),
            if self.body_end == self.tail_end {
                None
            } else {
                Some(SimdTail {
                    start: offset + (self.body_end * stride!()) as isize,
                    mask: PhantomData,
                })
            },
        )
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct SimdBody<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> {
    start: isize,
    mask: PhantomData<(Idx<'N>, T::SimdMask<S>)>,
}
#[repr(transparent)]
#[derive(Debug)]
pub struct SimdHead<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> {
    start: isize,
    mask: PhantomData<(Idx<'N>, T::SimdMask<S>)>,
}
#[repr(transparent)]
#[derive(Debug)]
pub struct SimdTail<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> {
    start: isize,
    mask: PhantomData<(Idx<'N>, T::SimdMask<S>)>,
}

impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> Copy for SimdBody<'N, C, T, S> {}
impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> Clone for SimdBody<'N, C, T, S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> Copy for SimdHead<'N, C, T, S> {}
impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> Clone for SimdHead<'N, C, T, S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> Copy for SimdTail<'N, C, T, S> {}
impl<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd> Clone for SimdTail<'N, C, T, S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
