use super::bound::{Dim, Idx};
use crate::internal_prelude::*;
use core::marker::PhantomData;
use faer_traits::SimdCapabilities;
use pulp::Simd;

pub struct SimdCtx<'N, T: ComplexField, S: Simd> {
	pub ctx: T::SimdCtx<S>,
	pub len: Dim<'N>,
	offset: usize,
	head_end: usize,
	body_end: usize,
	tail_end: usize,
	head_mask: T::SimdMask<S>,
	tail_mask: T::SimdMask<S>,
	head_mem_mask: T::SimdMemMask<S>,
	tail_mem_mask: T::SimdMemMask<S>,
}

impl<'N, T: ComplexField, S: Simd> core::fmt::Debug for SimdCtx<'N, T, S> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		f.debug_struct("SimdCtx")
			.field("len", &self.len)
			.field("offset", &self.offset)
			.field("head_end", &self.head_end)
			.field("body_end", &self.body_end)
			.field("tail_end", &self.tail_end)
			.field("head_mask", &self.head_mask)
			.field("tail_mask", &self.tail_mask)
			.finish_non_exhaustive()
	}
}

impl<T: ComplexField, S: Simd> Copy for SimdCtx<'_, T, S> {}
impl<T: ComplexField, S: Simd> Clone for SimdCtx<'_, T, S> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

impl<T: ComplexField, S: Simd> core::ops::Deref for SimdCtx<'_, T, S> {
	type Target = faer_traits::SimdCtx<T, S>;

	#[inline(always)]
	fn deref(&self) -> &Self::Target {
		Self::Target::new(&self.ctx)
	}
}

pub trait SimdIndex<'N, T: ComplexField, S: Simd> {
	fn read(simd: &SimdCtx<'N, T, S>, slice: ColRef<'_, T, Dim<'N>, ContiguousFwd>, index: Self) -> T::SimdVec<S>;

	fn write(simd: &SimdCtx<'N, T, S>, slice: ColMut<'_, T, Dim<'N>, ContiguousFwd>, index: Self, value: T::SimdVec<S>);
}

impl<'N, T: ComplexField, S: Simd> SimdIndex<'N, T, S> for SimdBody<'N, T, S> {
	#[inline(always)]
	fn read(simd: &SimdCtx<'N, T, S>, slice: ColRef<'_, T, Dim<'N>, ContiguousFwd>, index: Self) -> T::SimdVec<S> {
		unsafe { simd.load(&*(slice.as_ptr().wrapping_offset(index.start) as *const T::SimdVec<S>)) }
	}

	#[inline(always)]
	fn write(simd: &SimdCtx<'N, T, S>, slice: ColMut<'_, T, Dim<'N>, ContiguousFwd>, index: Self, value: T::SimdVec<S>) {
		unsafe {
			simd.store(&mut *(slice.as_ptr_mut().wrapping_offset(index.start) as *mut T::SimdVec<S>), value);
		}
	}
}

impl<'N, T: ComplexField, S: Simd> SimdIndex<'N, T, S> for SimdHead<'N, T, S> {
	#[inline(always)]
	fn read(simd: &SimdCtx<'N, T, S>, slice: ColRef<'_, T, Dim<'N>, ContiguousFwd>, index: Self) -> T::SimdVec<S> {
		unsafe { simd.mask_load(simd.head_mem_mask, slice.as_ptr().wrapping_offset(index.start) as *const T::SimdVec<S>) }
	}

	#[inline(always)]
	fn write(simd: &SimdCtx<'N, T, S>, slice: ColMut<'_, T, Dim<'N>, ContiguousFwd>, index: Self, value: T::SimdVec<S>) {
		unsafe {
			simd.mask_store(
				simd.head_mem_mask,
				slice.as_ptr_mut().wrapping_offset(index.start) as *mut T::SimdVec<S>,
				value,
			);
		}
	}
}

impl<'N, T: ComplexField, S: Simd> SimdIndex<'N, T, S> for SimdTail<'N, T, S> {
	#[inline(always)]
	fn read(simd: &SimdCtx<'N, T, S>, slice: ColRef<'_, T, Dim<'N>, ContiguousFwd>, index: Self) -> T::SimdVec<S> {
		unsafe { simd.mask_load(simd.tail_mem_mask, slice.as_ptr().wrapping_offset(index.start) as *const T::SimdVec<S>) }
	}

	#[inline(always)]
	fn write(simd: &SimdCtx<'N, T, S>, slice: ColMut<'_, T, Dim<'N>, ContiguousFwd>, index: Self, value: T::SimdVec<S>) {
		unsafe {
			simd.mask_store(
				simd.tail_mem_mask,
				slice.as_ptr_mut().wrapping_offset(index.start) as *mut T::SimdVec<S>,
				value,
			);
		}
	}
}

impl<'N, T: ComplexField, S: Simd> SimdCtx<'N, T, S> {
	#[inline(always)]
	pub fn new(simd: T::SimdCtx<S>, len: Dim<'N>) -> Self {
		core::assert!(try_const! { matches!(T::SIMD_CAPABILITIES, SimdCapabilities::Simd) });

		let stride = core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>();
		let iota = T::simd_iota(&simd);

		let head_start = T::simd_index_splat(&simd, T::Index::truncate(0));
		let head_end = T::simd_index_splat(&simd, T::Index::truncate(0));
		let tail_start = T::simd_index_splat(&simd, T::Index::truncate(0));
		let tail_end = T::simd_index_splat(&simd, T::Index::truncate(*len % stride));

		Self {
			ctx: simd,
			len,
			offset: 0,
			head_end: 0,
			body_end: *len / stride,
			tail_end: (*len + stride - 1) / stride,
			head_mask: T::simd_and_mask(
				&simd,
				T::simd_index_greater_than_or_equal(&simd, iota, head_start),
				T::simd_index_less_than(&simd, iota, head_end),
			),
			tail_mask: T::simd_and_mask(
				&simd,
				T::simd_index_greater_than_or_equal(&simd, iota, tail_start),
				T::simd_index_less_than(&simd, iota, tail_end),
			),
			head_mem_mask: T::simd_mem_mask_between(&simd, T::Index::truncate(0), T::Index::truncate(0)),
			tail_mem_mask: T::simd_mem_mask_between(&simd, T::Index::truncate(0), T::Index::truncate(*len % stride)),
		}
	}

	#[inline(always)]
	pub fn new_align(simd: T::SimdCtx<S>, len: Dim<'N>, align_offset: usize) -> Self {
		core::assert!(try_const! { matches!(T::SIMD_CAPABILITIES, SimdCapabilities::Simd) });

		let stride = core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>();
		let align_offset = align_offset % stride;
		let iota = T::simd_iota(&simd);

		if align_offset == 0 {
			Self::new(simd, len)
		} else {
			let offset = stride - align_offset;
			let full_len = offset + *len;

			let head_start = T::simd_index_splat(&simd, T::Index::truncate(offset));
			let head_end = T::simd_index_splat(&simd, T::Index::truncate(stride));
			let tail_start = T::simd_index_splat(&simd, T::Index::truncate(0));
			let tail_end = T::simd_index_splat(&simd, T::Index::truncate(full_len % stride));

			if align_offset <= *len {
				Self {
					ctx: simd,
					len,
					offset,
					head_end: 1,
					body_end: full_len / stride,
					tail_end: (full_len + stride - 1) / stride,
					head_mask: T::simd_and_mask(
						&simd,
						T::simd_index_greater_than_or_equal(&simd, iota, head_start),
						T::simd_index_less_than(&simd, iota, head_end),
					),
					tail_mask: T::simd_and_mask(
						&simd,
						T::simd_index_greater_than_or_equal(&simd, iota, tail_start),
						T::simd_index_less_than(&simd, iota, tail_end),
					),
					head_mem_mask: T::simd_mem_mask_between(&simd, T::Index::truncate(offset), T::Index::truncate(stride)),
					tail_mem_mask: T::simd_mem_mask_between(&simd, T::Index::truncate(0), T::Index::truncate(full_len % stride)),
				}
			} else {
				let head_start = T::simd_index_splat(&simd, T::Index::truncate(offset));
				let head_end = T::simd_index_splat(&simd, T::Index::truncate(full_len % stride));
				let tail_start = T::simd_index_splat(&simd, T::Index::truncate(0));
				let tail_end = T::simd_index_splat(&simd, T::Index::truncate(0));

				Self {
					ctx: simd,
					len,
					offset,
					head_end: 1,
					body_end: 1,
					tail_end: 1,
					head_mask: T::simd_and_mask(
						&simd,
						T::simd_index_greater_than_or_equal(&simd, iota, head_start),
						T::simd_index_less_than(&simd, iota, head_end),
					),
					tail_mask: T::simd_and_mask(
						&simd,
						T::simd_index_greater_than_or_equal(&simd, iota, tail_start),
						T::simd_index_less_than(&simd, iota, tail_end),
					),
					head_mem_mask: T::simd_mem_mask_between(&simd, T::Index::truncate(offset), T::Index::truncate(full_len % stride)),
					tail_mem_mask: T::simd_mem_mask_between(&simd, T::Index::truncate(0), T::Index::truncate(0)),
				}
			}
		}
	}

	#[inline]
	pub fn offset(&self) -> usize {
		self.offset
	}

	#[inline(always)]
	pub fn new_force_mask(simd: T::SimdCtx<S>, len: Dim<'N>) -> Self {
		core::assert!(try_const! { matches!(T::SIMD_CAPABILITIES, SimdCapabilities::Simd) });

		crate::assert!(*len != 0);
		let new_len = *len - 1;

		let stride = core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>();
		let iota = T::simd_iota(&simd);

		let head_start = T::simd_index_splat(&simd, T::Index::truncate(0));
		let head_end = T::simd_index_splat(&simd, T::Index::truncate(0));
		let tail_start = T::simd_index_splat(&simd, T::Index::truncate(0));
		let tail_end = T::simd_index_splat(&simd, T::Index::truncate((new_len % stride) + 1));

		Self {
			ctx: simd,
			len,
			offset: 0,
			head_end: 0,
			body_end: new_len / stride,
			tail_end: new_len / stride + 1,
			head_mask: T::simd_and_mask(
				&simd,
				T::simd_index_greater_than_or_equal(&simd, iota, head_start),
				T::simd_index_less_than(&simd, iota, head_end),
			),
			tail_mask: T::simd_and_mask(
				&simd,
				T::simd_index_greater_than_or_equal(&simd, iota, tail_start),
				T::simd_index_less_than(&simd, iota, tail_end),
			),
			head_mem_mask: T::simd_mem_mask_between(&simd, T::Index::truncate(0), T::Index::truncate(0)),
			tail_mem_mask: T::simd_mem_mask_between(&simd, T::Index::truncate(0), T::Index::truncate((new_len % stride) + 1)),
		}
	}

	#[inline(always)]
	pub fn read<I: SimdIndex<'N, T, S>>(&self, slice: ColRef<'_, T, Dim<'N>, ContiguousFwd>, index: I) -> T::SimdVec<S> {
		I::read(self, slice, index)
	}

	#[inline(always)]
	pub fn write<I: SimdIndex<'N, T, S>>(&self, slice: ColMut<'_, T, Dim<'N>, ContiguousFwd>, index: I, value: T::SimdVec<S>) {
		I::write(self, slice, index, value)
	}

	#[inline(always)]
	pub fn head_mask(&self) -> T::SimdMask<S> {
		self.head_mask
	}

	#[inline(always)]
	pub fn tail_mask(&self) -> T::SimdMask<S> {
		self.tail_mask
	}

	#[inline]
	pub fn indices(
		&self,
	) -> (
		Option<SimdHead<'N, T, S>>,
		impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = SimdBody<'N, T, S>>,
		Option<SimdTail<'N, T, S>>,
	) {
		macro_rules! stride {
			() => {
				core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>()
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
		Option<SimdHead<'N, T, S>>,
		impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = [SimdBody<'N, T, S>; BATCH]>,
		impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = SimdBody<'N, T, S>>,
		Option<SimdTail<'N, T, S>>,
	) {
		macro_rules! stride {
			() => {
				core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>()
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
			(self.head_end..self.head_end + len / BATCH * BATCH)
				.map(move |i| {
					core::array::from_fn(
						#[inline(always)]
						|k| SimdBody {
							start: offset + ((i + k) * stride!()) as isize,
							mask: PhantomData,
						},
					)
				})
				.step_by(BATCH),
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
pub struct SimdBody<'N, T: ComplexField, S: Simd> {
	start: isize,
	mask: PhantomData<(Idx<'N>, T::SimdMask<S>)>,
}

impl<T: ComplexField, S: Simd> SimdBody<'_, T, S> {
	pub fn offset(&self) -> isize {
		self.start
	}
}

#[repr(transparent)]
#[derive(Debug)]
pub struct SimdHead<'N, T: ComplexField, S: Simd> {
	start: isize,
	mask: PhantomData<(Idx<'N>, T::SimdMask<S>)>,
}
#[repr(transparent)]
#[derive(Debug)]
pub struct SimdTail<'N, T: ComplexField, S: Simd> {
	start: isize,
	mask: PhantomData<(Idx<'N>, T::SimdMask<S>)>,
}

impl<'N, T: ComplexField, S: Simd> Copy for SimdBody<'N, T, S> {}
impl<'N, T: ComplexField, S: Simd> Clone for SimdBody<'N, T, S> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
impl<'N, T: ComplexField, S: Simd> Copy for SimdHead<'N, T, S> {}
impl<'N, T: ComplexField, S: Simd> Clone for SimdHead<'N, T, S> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
impl<'N, T: ComplexField, S: Simd> Copy for SimdTail<'N, T, S> {}
impl<'N, T: ComplexField, S: Simd> Clone for SimdTail<'N, T, S> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
