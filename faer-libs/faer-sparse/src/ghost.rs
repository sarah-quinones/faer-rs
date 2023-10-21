pub use faer_core::constrained::{group_helpers::*, permutation::*, sparse::*, *};
use faer_core::permutation::Index;

pub const NONE_BYTE: u8 = u8::MAX;

#[inline]
pub fn with_size<R>(n: usize, f: impl FnOnce(Size<'_>) -> R) -> R {
    Size::with(n, f)
}

#[inline]
pub fn fill_none<'n, 'a, I: Index>(
    slice: &'a mut [I::Signed],
    size: Size<'n>,
) -> &'a mut [MaybeIdx<'n, I>] {
    let _ = size;
    let len = slice.len();
    unsafe { core::ptr::write_bytes(slice.as_mut_ptr(), NONE_BYTE, len) };
    unsafe { &mut *(slice as *mut _ as *mut _) }
}

#[inline]
pub fn copy_slice<'n, 'a, I: Index>(dst: &'a mut [I], src: &[Idx<'n, I>]) -> &'a mut [Idx<'n, I>] {
    let dst: &mut [Idx<'_, I>] = unsafe { &mut *(dst as *mut _ as *mut _) };
    dst.copy_from_slice(src);
    dst
}
