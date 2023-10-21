use faer_core::permutation::SignedIndex;

pub const NONE_BYTE: u8 = 0xFF;
pub const NONE: usize = faer_core::sparse::repeat_byte(NONE_BYTE);

#[inline]
pub fn fill_none<I: SignedIndex>(slice: &mut [I]) {
    let len = slice.len();
    unsafe { core::ptr::write_bytes(slice.as_mut_ptr(), NONE_BYTE, len) }
}
#[inline]
pub fn fill_zero<I: bytemuck::Zeroable>(slice: &mut [I]) {
    let len = slice.len();
    unsafe { core::ptr::write_bytes(slice.as_mut_ptr(), 0u8, len) }
}
