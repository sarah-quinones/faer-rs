pub const LLT: usize = 0;
pub const LDLT: usize = 1;
pub const PLU: usize = 2;
pub const FLU: usize = 3;
pub const QR: usize = 4;
pub const CQR: usize = 5;
pub const SVD: usize = 6;
pub const HEVD: usize = 7;
pub const EVD: usize = 8;
pub const F32: usize = 0;
pub const F64: usize = 1;
pub const FX128: usize = 2;
pub const C32: usize = 3;
pub const C64: usize = 4;
pub const CX128: usize = 5;

use core::ffi::c_void;

unsafe extern "C" {
	pub fn libeigen_make_decomp(decomp: usize, dtype: usize, nrows: usize, ncols: usize) -> *mut c_void;
	pub fn libeigen_factorize(decomp: usize, dtype: usize, ptr: *mut c_void, data: *mut c_void, nrows: usize, ncols: usize, stride: usize);
	pub fn libeigen_free_decomp(decomp: usize, dtype: usize, ptr: *mut c_void);
}
