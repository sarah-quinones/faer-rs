use core::slice::SliceIndex;

use crate::Index;

pub const NONE_BYTE: u8 = 0xFF;
pub const NONE: usize = repeat_byte(NONE_BYTE);

#[inline(always)]
#[track_caller]
pub fn __get_checked<I, R: Clone + SliceIndex<[I]>>(slice: &[I], i: R) -> &R::Output {
    &slice[i]
}
#[inline(always)]
#[track_caller]
pub fn __get_checked_mut<I, R: Clone + SliceIndex<[I]>>(slice: &mut [I], i: R) -> &mut R::Output {
    &mut slice[i]
}

#[inline(always)]
#[track_caller]
pub unsafe fn __get_unchecked<I, R: Clone + SliceIndex<[I]>>(slice: &[I], i: R) -> &R::Output {
    #[cfg(debug_assertions)]
    {
        let _ = &slice[i.clone()];
    }
    unsafe { slice.get_unchecked(i) }
}
#[inline(always)]
#[track_caller]
pub unsafe fn __get_unchecked_mut<I, R: Clone + SliceIndex<[I]>>(
    slice: &mut [I],
    i: R,
) -> &mut R::Output {
    #[cfg(debug_assertions)]
    {
        let _ = &slice[i.clone()];
    }
    unsafe { slice.get_unchecked_mut(i) }
}

#[inline]
pub unsafe fn transmute_slice_ref<From, To>(from: &[From]) -> &[To] {
    assert!(core::mem::size_of::<From>() == core::mem::size_of::<To>());
    assert!(core::mem::align_of::<From>() == core::mem::align_of::<To>());
    let len = from.len();
    core::slice::from_raw_parts(from.as_ptr() as *const To, len)
}

#[inline]
pub unsafe fn transmute_slice_mut<From, To>(from: &mut [From]) -> &mut [To] {
    assert!(core::mem::size_of::<From>() == core::mem::size_of::<To>());
    assert!(core::mem::align_of::<From>() == core::mem::align_of::<To>());
    let len = from.len();
    core::slice::from_raw_parts_mut(from.as_mut_ptr() as *mut To, len)
}

#[inline]
pub const fn repeat_byte(byte: u8) -> usize {
    #[repr(align(256))]
    struct Aligned([u8; 32]);

    let data = Aligned([byte; 32]);
    unsafe { *((&data) as *const _ as *const usize) }
}

#[inline]
pub fn fill_none<I: Index>(slice: &mut [I]) {
    let len = slice.len();
    unsafe { core::ptr::write_bytes(slice.as_mut_ptr(), NONE_BYTE, len) }
}
#[inline]
pub fn fill_zero<I: bytemuck::Zeroable>(slice: &mut [I]) {
    let len = slice.len();
    unsafe { core::ptr::write_bytes(slice.as_mut_ptr(), 0u8, len) }
}
