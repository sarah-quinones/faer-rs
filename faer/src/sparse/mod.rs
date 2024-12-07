mod csc;
mod csr;

pub use csc::*;
pub use csr::*;

extern crate alloc;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Pair<Row, Col> {
	pub row: Row,
	pub col: Col,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Triplet<Row, Col, T> {
	pub row: Row,
	pub col: Col,
	pub val: T,
}

/// Errors that can occur in sparse algorithms.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[non_exhaustive]
pub enum FaerError {
	/// An index exceeding the maximum value (`I::Signed::MAX` for a given index type `I`).
	IndexOverflow,
	/// Memory allocation failed.
	OutOfMemory,
}

impl From<dyn_stack::SizeOverflow> for FaerError {
	#[inline]
	fn from(value: dyn_stack::SizeOverflow) -> Self {
		_ = value;
		FaerError::OutOfMemory
	}
}

impl From<dyn_stack::mem::AllocError> for FaerError {
	#[inline]
	fn from(value: dyn_stack::mem::AllocError) -> Self {
		_ = value;
		FaerError::OutOfMemory
	}
}

impl From<alloc::collections::TryReserveError> for FaerError {
	#[inline]
	fn from(value: alloc::collections::TryReserveError) -> Self {
		_ = value;
		FaerError::OutOfMemory
	}
}

impl core::fmt::Display for FaerError {
	#[inline]
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Debug::fmt(self, f)
	}
}

impl core::error::Error for FaerError {}

/// Errors that can occur in sparse algorithms.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum CreationError {
	/// Generic error (allocation or index overflow).
	Generic(FaerError),
	/// Matrix index out-of-bounds error.
	OutOfBounds {
		/// Row of the out-of-bounds index.
		row: usize,
		/// Column of the out-of-bounds index.
		col: usize,
	},
}

#[inline(always)]
fn windows2<I>(slice: &[I]) -> impl DoubleEndedIterator<Item = &[I; 2]> {
	slice.windows(2).map(
		#[inline(always)]
		|window| unsafe { &*(window.as_ptr() as *const [I; 2]) },
	)
}

#[inline]
#[track_caller]
fn try_zeroed<I: bytemuck::Pod>(n: usize) -> Result<alloc::vec::Vec<I>, FaerError> {
	let mut v = alloc::vec::Vec::new();
	v.try_reserve_exact(n).map_err(|_| FaerError::OutOfMemory)?;
	unsafe {
		core::ptr::write_bytes::<I>(v.as_mut_ptr(), 0u8, n);
		v.set_len(n);
	}
	Ok(v)
}

#[inline]
#[track_caller]
fn try_collect<I: IntoIterator>(iter: I) -> Result<alloc::vec::Vec<I::Item>, FaerError> {
	let iter = iter.into_iter();
	let mut v = alloc::vec::Vec::new();
	v.try_reserve_exact(iter.size_hint().0).map_err(|_| FaerError::OutOfMemory)?;
	v.extend(iter);
	Ok(v)
}

pub mod utils;
