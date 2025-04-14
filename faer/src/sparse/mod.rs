//! sparse matrix data structures
//!
//! most sparse matrix algorithms accept matrices in sparse column-oriented format.
//! this format represents each column of the matrix by storing the row indices of its non-zero
//! elements, as well as their values
//!
//! the indices and the values are each stored in a contiguous slice (or group of slices for
//! arbitrary values). in order to specify where each column starts and ends, a slice of size
//! `ncols + 1` stores the start of each column, with the last element being equal to the total
//! number of non-zeros (or the capacity in uncompressed mode)
//!
//! # example
//!
//! consider the 4-by-5 matrix:
//! ```notcode
//! [[10.0, 0.0, 12.0, -1.0, 13.0]
//!  [ 0.0, 0.0, 25.0, -2.0,  0.0]
//!  [ 1.0, 0.0,  0.0,  0.0,  0.0]
//!  [ 4.0, 0.0,  0.0,  0.0,  5.0]]
//! ```
//!
//! the matrix is stored as follows:
//! ```notcode
//! column pointers: | 0                  | 3,3         | 5           | 7           | 9
//!
//! row indices    : |    0 |    2 |    3 |    0 |    1 |    0 |    1 |    0 |    3 |
//! values         : | 10.0 |  1.0 |  4.0 | 12.0 | 25.0 | -1.0 | -2.0 | 13.0 |  5.0 |
//! ```

mod csc;
mod csr;

pub(crate) const NONE: usize = usize::MAX;

/// sparse linear algebra module.
/// contains low level routines and the implementation of their corresponding high level wrappers
pub mod linalg;
/// sparse matrix binary and ternary operation implementations
pub mod ops;

use crate::internal_prelude_sp::Index;
use reborrow::*;

pub use csc::{SparseColMat, SparseColMatMut, SparseColMatRef, SymbolicSparseColMat, SymbolicSparseColMatRef};
pub use csr::{SparseRowMat, SparseRowMatMut, SparseRowMatRef, SymbolicSparseRowMat, SymbolicSparseRowMatRef};

pub use csc::symbolic as csc_symbolic;
pub use csr::symbolic as csr_symbolic;

pub use csc::numeric as csc_numeric;
pub use csr::numeric as csr_numeric;

extern crate alloc;

/// pair of indices with `C`-compatible layout
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Pair<Row, Col> {
	/// row index
	pub row: Row,
	/// column index
	pub col: Col,
}

/// triplet of indices and value with `C`-compatible layout
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Triplet<Row, Col, T> {
	/// row index
	pub row: Row,
	/// column index
	pub col: Col,
	/// value
	pub val: T,
}

impl<Row, Col> Pair<Row, Col> {
	/// creates a new pair of indices
	#[inline]
	pub const fn new(row: Row, col: Col) -> Self {
		Pair { row, col }
	}
}

impl<Row, Col, T> Triplet<Row, Col, T> {
	/// creates a new pair of indices and value
	#[inline]
	pub const fn new(row: Row, col: Col, val: T) -> Self {
		Triplet { row, col, val }
	}
}

/// errors that can occur in sparse algorithms
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[non_exhaustive]
pub enum FaerError {
	/// an index exceeding the maximum value (`I::Signed::MAX` for a given index type `I`)
	IndexOverflow,
	/// memory allocation failed
	OutOfMemory,
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

/// errors that can occur during the creation of sparse matrices from user input
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum CreationError {
	/// generic error (allocation or index overflow)
	Generic(FaerError),
	/// matrix index out-of-bounds error
	OutOfBounds {
		/// row of the out-of-bounds index
		row: usize,
		/// column of the out-of-bounds index
		col: usize,
	},
}

impl From<FaerError> for CreationError {
	#[inline]
	fn from(value: FaerError) -> Self {
		Self::Generic(value)
	}
}
impl core::fmt::Display for CreationError {
	#[inline]
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Debug::fmt(self, f)
	}
}

impl core::error::Error for CreationError {}

#[inline(always)]
pub(crate) fn windows2<I>(slice: &[I]) -> impl DoubleEndedIterator<Item = &[I; 2]> {
	slice.windows(2).map(
		#[inline(always)]
		|window| unsafe { &*(window.as_ptr() as *const [I; 2]) },
	)
}

#[inline]
#[track_caller]
pub(crate) fn try_zeroed<I: bytemuck::Pod>(n: usize) -> Result<alloc::vec::Vec<I>, FaerError> {
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
pub(crate) fn try_collect<I: IntoIterator>(iter: I) -> Result<alloc::vec::Vec<I::Item>, FaerError> {
	let iter = iter.into_iter();
	let mut v = alloc::vec::Vec::new();
	v.try_reserve_exact(iter.size_hint().0).map_err(|_| FaerError::OutOfMemory)?;
	v.extend(iter);
	Ok(v)
}

/// the order values should be read in, when constructing/filling from indices and values
///
/// allows separately creating the symbolic structure and filling the numerical values
#[derive(Debug, Clone)]
pub struct Argsort<I> {
	idx: alloc::vec::Vec<I>,
	all_nnz: usize,
	nnz: usize,
}

/// algorithmic primitives for sparse matrices
pub mod utils;

impl<I: Index, T> core::ops::Index<(usize, usize)> for SparseColMatRef<'_, I, T> {
	type Output = T;

	#[track_caller]
	fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
		self.get(row, col).unwrap()
	}
}

impl<I: Index, T> core::ops::Index<(usize, usize)> for SparseRowMatRef<'_, I, T> {
	type Output = T;

	#[track_caller]
	fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
		self.get(row, col).unwrap()
	}
}

impl<I: Index, T> core::ops::Index<(usize, usize)> for SparseColMatMut<'_, I, T> {
	type Output = T;

	#[track_caller]
	fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
		self.rb().get(row, col).unwrap()
	}
}

impl<I: Index, T> core::ops::Index<(usize, usize)> for SparseRowMatMut<'_, I, T> {
	type Output = T;

	#[track_caller]
	fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
		self.rb().get(row, col).unwrap()
	}
}

impl<I: Index, T> core::ops::IndexMut<(usize, usize)> for SparseColMatMut<'_, I, T> {
	#[track_caller]
	fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
		self.rb_mut().get_mut(row, col).unwrap()
	}
}

impl<I: Index, T> core::ops::IndexMut<(usize, usize)> for SparseRowMatMut<'_, I, T> {
	#[track_caller]
	fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
		self.rb_mut().get_mut(row, col).unwrap()
	}
}

impl<I: Index, T> core::ops::Index<(usize, usize)> for SparseColMat<I, T> {
	type Output = T;

	#[track_caller]
	fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
		self.rb().get(row, col).unwrap()
	}
}

impl<I: Index, T> core::ops::Index<(usize, usize)> for SparseRowMat<I, T> {
	type Output = T;

	#[track_caller]
	fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
		self.rb().get(row, col).unwrap()
	}
}

impl<I: Index, T> core::ops::IndexMut<(usize, usize)> for SparseColMat<I, T> {
	#[track_caller]
	fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
		self.rb_mut().get_mut(row, col).unwrap()
	}
}

impl<I: Index, T> core::ops::IndexMut<(usize, usize)> for SparseRowMat<I, T> {
	#[track_caller]
	fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
		self.rb_mut().get_mut(row, col).unwrap()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;

	#[test]
	fn test_from_indices() {
		let nrows = 5;
		let ncols = 4;

		let indices = &[
			Pair::new(0, 0),
			Pair::new(1, 2),
			Pair::new(0, 0),
			Pair::new(1, 1),
			Pair::new(0, 1),
			Pair::new(3, 3),
			Pair::new(3, 3usize),
		];
		let values = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0f64];

		let triplets = &[
			Triplet::new(0, 0, 1.0),
			Triplet::new(1, 2, 2.0),
			Triplet::new(0, 0, 3.0),
			Triplet::new(1, 1, 4.0),
			Triplet::new(0, 1, 5.0),
			Triplet::new(3, 3, 6.0),
			Triplet::new(3, 3usize, 7.0_f64),
		];

		{
			let mat = SymbolicSparseColMat::try_new_from_indices(nrows, ncols, indices);
			assert!(mat.is_ok());

			let (mat, order) = mat.unwrap();
			assert!(mat.nrows() == nrows);
			assert!(mat.ncols() == ncols);
			assert!(mat.col_ptr() == &[0, 1, 3, 4, 5]);
			assert!(mat.col_nnz() == None);
			assert!(mat.row_idx() == &[0, 0, 1, 1, 3]);

			let mat = SparseColMat::<_, f64>::new_from_argsort(mat, &order, values).unwrap();
			assert!(mat.val() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
		}

		{
			let mat = SparseColMat::try_new_from_triplets(nrows, ncols, triplets);
			assert!(mat.is_ok());
			let mat = mat.unwrap();

			assert!(mat.nrows() == nrows);
			assert!(mat.ncols() == ncols);
			assert!(mat.col_ptr() == &[0, 1, 3, 4, 5]);
			assert!(mat.col_nnz() == None);
			assert!(mat.row_idx() == &[0, 0, 1, 1, 3]);
			assert!(mat.val() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
		}

		{
			let mat = SymbolicSparseRowMat::try_new_from_indices(nrows, ncols, indices);
			assert!(mat.is_ok());

			let (mat, order) = mat.unwrap();
			assert!(mat.nrows() == nrows);
			assert!(mat.ncols() == ncols);
			assert!(mat.row_ptr() == &[0, 2, 4, 4, 5, 5]);
			assert!(mat.row_nnz() == None);
			assert!(mat.col_idx() == &[0, 1, 1, 2, 3]);

			let mat = SparseRowMat::<_, f64>::new_from_argsort(mat, &order, values).unwrap();
			assert!(mat.val() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
		}
		{
			let mat = SparseRowMat::try_new_from_triplets(nrows, ncols, triplets);
			assert!(mat.is_ok());

			let mat = mat.unwrap();
			assert!(mat.nrows() == nrows);
			assert!(mat.ncols() == ncols);
			assert!(mat.row_ptr() == &[0, 2, 4, 4, 5, 5]);
			assert!(mat.row_nnz() == None);
			assert!(mat.col_idx() == &[0, 1, 1, 2, 3]);
			assert!(mat.val() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
		}
	}

	#[test]
	fn test_from_nonnegative_indices() {
		let nrows = 5;
		let ncols = 4;

		let indices = &[
			Pair::new(0, 0),
			Pair::new(1, 2),
			Pair::new(0, 0),
			Pair::new(1, 1),
			Pair::new(0, 1),
			Pair::new(-1, 2),
			Pair::new(-2, 1),
			Pair::new(-3, -4),
			Pair::new(3, 3),
			Pair::new(3, 3isize),
		];
		let values = &[1.0, 2.0, 3.0, 4.0, 5.0, f64::NAN, f64::NAN, f64::NAN, 6.0, 7.0f64];

		let triplets = &[
			Triplet::new(0, 0, 1.0),
			Triplet::new(1, 2, 2.0),
			Triplet::new(0, 0, 3.0),
			Triplet::new(1, 1, 4.0),
			Triplet::new(0, 1, 5.0),
			Triplet::new(-1, 2, f64::NAN),
			Triplet::new(-2, 1, f64::NAN),
			Triplet::new(-3, -4, f64::NAN),
			Triplet::new(3, 3, 6.0),
			Triplet::new(3, 3isize, 7.0_f64),
		];

		{
			let mat = SymbolicSparseColMat::<usize>::try_new_from_nonnegative_indices(nrows, ncols, indices);
			assert!(mat.is_ok());

			let (mat, order) = mat.unwrap();
			assert!(mat.nrows() == nrows);
			assert!(mat.ncols() == ncols);
			assert!(mat.col_ptr() == &[0, 1, 3, 4, 5]);
			assert!(mat.col_nnz() == None);
			assert!(mat.row_idx() == &[0, 0, 1, 1, 3]);

			let mat = SparseColMat::<_, f64>::new_from_argsort(mat, &order, values).unwrap();
			assert!(mat.val() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
		}

		{
			let mat = SparseColMat::<usize, _>::try_new_from_nonnegative_triplets(nrows, ncols, triplets);
			assert!(mat.is_ok());
			let mat = mat.unwrap();

			assert!(mat.nrows() == nrows);
			assert!(mat.ncols() == ncols);
			assert!(mat.col_ptr() == &[0, 1, 3, 4, 5]);
			assert!(mat.col_nnz() == None);
			assert!(mat.row_idx() == &[0, 0, 1, 1, 3]);
			assert!(mat.val() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
		}

		{
			let mat = SymbolicSparseRowMat::<usize>::try_new_from_nonnegative_indices(nrows, ncols, indices);
			assert!(mat.is_ok());

			let (mat, order) = mat.unwrap();
			assert!(mat.nrows() == nrows);
			assert!(mat.ncols() == ncols);
			assert!(mat.row_ptr() == &[0, 2, 4, 4, 5, 5]);
			assert!(mat.row_nnz() == None);
			assert!(mat.col_idx() == &[0, 1, 1, 2, 3]);

			let mat = SparseRowMat::<_, f64>::new_from_argsort(mat, &order, values).unwrap();
			assert!(mat.val() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
		}
		{
			let mat = SparseRowMat::<usize, _>::try_new_from_nonnegative_triplets(nrows, ncols, triplets);
			assert!(mat.is_ok());

			let mat = mat.unwrap();
			assert!(mat.nrows() == nrows);
			assert!(mat.ncols() == ncols);
			assert!(mat.row_ptr() == &[0, 2, 4, 4, 5, 5]);
			assert!(mat.row_nnz() == None);
			assert!(mat.col_idx() == &[0, 1, 1, 2, 3]);
			assert!(mat.val() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
		}
	}

	#[test]
	fn test_from_indices_oob_row() {
		let nrows = 5;
		let ncols = 4;

		let indices = &[
			Pair::new(0, 0),
			Pair::new(1, 2),
			Pair::new(0, 0),
			Pair::new(1, 1),
			Pair::new(0, 1),
			Pair::new(3, 3),
			Pair::new(3, 3),
			Pair::new(5, 3usize),
		];
		let err = SymbolicSparseColMat::try_new_from_indices(nrows, ncols, indices);
		assert!(err.is_err());
		let err = err.unwrap_err();
		assert!(err == CreationError::OutOfBounds { row: 5, col: 3 });
	}

	#[test]
	fn test_from_indices_oob_col() {
		let nrows = 5;
		let ncols = 4;

		let indices = &[
			Pair::new(0, 0),
			Pair::new(1, 2),
			Pair::new(0, 0),
			Pair::new(1, 1),
			Pair::new(0, 1),
			Pair::new(3, 3),
			Pair::new(3, 3),
			Pair::new(2, 4usize),
		];
		let err = SymbolicSparseColMat::try_new_from_indices(nrows, ncols, indices);
		assert!(err.is_err());
		let err = err.unwrap_err();
		assert!(err == CreationError::OutOfBounds { row: 2, col: 4 });
	}
}
