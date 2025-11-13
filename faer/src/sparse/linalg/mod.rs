use crate::internal_prelude_sp::*;
const CHOLESKY_SUPERNODAL_RATIO_FACTOR: f64 = 40.0;
const QR_SUPERNODAL_RATIO_FACTOR: f64 = 40.0;
const LU_SUPERNODAL_RATIO_FACTOR: f64 = 40.0;
/// tuning parameters for the supernodal factorizations
#[derive(Copy, Clone, Debug)]
pub struct SymbolicSupernodalParams<'a> {
	/// supernode relaxation thresholds
	///
	/// let `n` be the total number of columns in two adjacent supernodes.
	/// let `z` be the fraction of zero entries in the two supernodes if they
	/// are merged (z includes zero entries from prior amalgamations). the
	/// two supernodes are merged if:
	///
	/// `(n <= relax[0].0 && z < relax[0].1) || (n <= relax[1].0 && z < relax[1].1) || ...`
	pub relax: Option<&'a [(usize, f64)]>,
}
const DEFAULT_RELAX: &'static [(usize, f64)] = &[(4, 1.0), (16, 0.8), (48, 0.1), (usize::MAX, 0.05)];
impl Default for SymbolicSupernodalParams<'_> {
	#[inline]
	fn default() -> Self {
		Self { relax: Some(DEFAULT_RELAX) }
	}
}
/// nonnegative threshold controlling when the supernodal factorization is used
///
/// increasing it makes it more likely for the simplicial factorization to be used,
/// while decreasing it makes it more likely for the supernodal factorization to be used
///
/// `1.0` is the default value
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SupernodalThreshold(pub f64);
impl Default for SupernodalThreshold {
	#[inline]
	fn default() -> Self {
		Self(1.0)
	}
}
impl SupernodalThreshold {
	/// determine automatically which variant to select
	pub const AUTO: Self = Self(1.0);
	/// simplicial factorization is always selected
	pub const FORCE_SIMPLICIAL: Self = Self(f64::INFINITY);
	/// supernodal factorization is always selected
	pub const FORCE_SUPERNODAL: Self = Self(0.0);
}
/// sparse $ll^\top$ error
#[derive(Copy, Clone, Debug)]
pub enum LltError {
	/// numerical error
	Numeric(linalg::cholesky::llt::factor::LltError),
	/// non algorithmic error
	Generic(FaerError),
}
impl core::fmt::Display for LltError {
	#[inline]
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Debug::fmt(self, f)
	}
}
impl core::error::Error for LltError {}
impl From<linalg::cholesky::llt::factor::LltError> for LltError {
	fn from(value: linalg::cholesky::llt::factor::LltError) -> Self {
		Self::Numeric(value)
	}
}
/// sparse $lu$ error.
#[derive(Copy, Clone, Debug)]
pub enum LuError {
	/// rank deficient symbolic structure
	SymbolicSingular {
		/// iteration at which a pivot could not be found
		index: usize,
	},
	/// non algorithmic error
	Generic(FaerError),
}
impl core::fmt::Display for LuError {
	#[inline]
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Debug::fmt(self, f)
	}
}
impl core::error::Error for LuError {}
impl<T: Into<FaerError>> From<T> for LltError {
	fn from(value: T) -> Self {
		Self::Generic(value.into())
	}
}
impl<T: Into<FaerError>> From<T> for LuError {
	fn from(value: T) -> Self {
		Self::Generic(value.into())
	}
}
pub mod amd;
pub mod cholesky;
pub mod colamd;
pub mod lu;
/// sparse matrix multiplication
pub mod matmul;
pub mod qr;
/// high-level sparse matrix solvers
#[path = "../solvers.rs"]
pub mod solvers;
/// sparse matrix triangular solve
pub mod triangular_solve;
mod ghost {
	use crate::Index;
	pub use crate::utils::bound::*;
	pub const NONE_BYTE: u8 = u8::MAX;
	#[inline]
	pub fn fill_zero<'n, 'a, I: Index>(slice: &'a mut [I], size: Dim<'n>) -> &'a mut [Idx<'n, I>] {
		let len = slice.len();
		if len > 0 {
			assert!(*size > 0);
		}
		unsafe {
			core::ptr::write_bytes(slice.as_mut_ptr(), 0u8, len);
			&mut *(slice as *mut _ as *mut _)
		}
	}
	#[inline]
	pub fn fill_none<'n, 'a, I: Index>(slice: &'a mut [I::Signed], size: Dim<'n>) -> &'a mut [MaybeIdx<'n, I>] {
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
}
