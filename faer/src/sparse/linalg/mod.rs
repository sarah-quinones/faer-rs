const CHOLESKY_SUPERNODAL_RATIO_FACTOR: f64 = 40.0;
const QR_SUPERNODAL_RATIO_FACTOR: f64 = 40.0;
const LU_SUPERNODAL_RATIO_FACTOR: f64 = 40.0;

/// Tuning parameters for the supernodal factorizations.
#[derive(Copy, Clone, Debug)]
pub struct SymbolicSupernodalParams<'a> {
	/// Supernode relaxation thresholds.
	///
	/// Let `n` be the total number of columns in two adjacent supernodes.
	/// Let `z` be the fraction of zero entries in the two supernodes if they
	/// are merged (z includes zero entries from prior amalgamations). The
	/// two supernodes are merged if:
	///
	/// `(n <= relax[0].0 && z < relax[0].1) || (n <= relax[1].0 && z < relax[1].1) || ...`
	pub relax: Option<&'a [(usize, f64)]>,
}

impl Default for SymbolicSupernodalParams<'_> {
	#[inline]
	fn default() -> Self {
		Self {
			relax: Some(&const { [(4, 1.0), (16, 0.8), (48, 0.1), (usize::MAX, 0.05)] }),
		}
	}
}

/// Nonnegative threshold controlling when the supernodal factorization is used.
///
/// Increasing it makes it more likely for the simplicial factorization to be used,
/// while decreasing it makes it more likely for the supernodal factorization to be used.
///
/// A value of `1.0` is the default.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SupernodalThreshold(pub f64);

impl Default for SupernodalThreshold {
	#[inline]
	fn default() -> Self {
		Self(1.0)
	}
}

impl SupernodalThreshold {
	/// Determine automatically which variant to select.
	pub const AUTO: Self = Self(1.0);
	/// Simplicial factorization is always selected.
	pub const FORCE_SIMPLICIAL: Self = Self(f64::INFINITY);
	/// Supernodal factorization is always selected.
	pub const FORCE_SUPERNODAL: Self = Self(0.0);
}

pub mod matmul;
pub mod triangular_solve;

pub mod amd;
pub mod colamd;

pub mod cholesky;
pub mod lu;
pub mod qr;

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
