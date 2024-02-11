//! Sparse matrix data structures.
//!
//! Most sparse matrix algorithms accept matrices in sparse column-oriented format.
//! This format represents each column of the matrix by storing the row indices of its non-zero
//! elements, as well as their values.
//!
//! The indices and the values are each stored in a contiguous slice (or group of slices for
//! arbitrary values). In order to specify where each column starts and ends, a slice of size
//! `ncols + 1` stores the start of each column, with the last element being equal to the total
//! number of non-zeros (or the capacity in uncompressed mode).
//!
//! # Example
//!
//! Consider the 4-by-5 matrix:
//! ```notcode
//! 10.0  0.0  12.0  -1.0  13.0
//!  0.0  0.0  25.0  -2.0   0.0
//!  1.0  0.0   0.0   0.0   0.0
//!  4.0  0.0   0.0   0.0   5.0
//! ```
//!
//! The matrix is stored as follows:
//! ```notcode
//! column pointers:  0 |  3 |  3 |  5 |  7 |  9
//!
//! row indices:    0 |    2 |    3 |    0 |    1 |    0 |    1 |    0 |    3
//! values     : 10.0 |  1.0 |  4.0 | 12.0 | 25.0 | -1.0 | -2.0 | 13.0 |  5.0
//! ```

use super::*;
use crate::{assert, group_helpers::VecGroup};
use core::{cell::Cell, iter::zip, ops::Range, slice::SliceIndex};
use dyn_stack::GlobalPodBuffer;
use group_helpers::SliceGroup;
use permutation::{Index, SignedIndex};

mod ghost {
    pub use crate::constrained::{group_helpers::*, permutation::*, sparse::*, *};
}

mod mem {
    #[inline]
    pub fn fill_zero<I: bytemuck::Zeroable>(slice: &mut [I]) {
        let len = slice.len();
        unsafe { core::ptr::write_bytes(slice.as_mut_ptr(), 0u8, len) }
    }
}

#[inline(always)]
#[track_caller]
#[doc(hidden)]
pub unsafe fn __get_unchecked<I, R: Clone + SliceIndex<[I]>>(slice: &[I], i: R) -> &R::Output {
    #[cfg(debug_assertions)]
    {
        let _ = &slice[i.clone()];
    }
    unsafe { slice.get_unchecked(i) }
}
#[inline(always)]
#[track_caller]
#[doc(hidden)]
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

#[inline(always)]
#[doc(hidden)]
pub fn windows2<I>(slice: &[I]) -> impl DoubleEndedIterator<Item = &[I; 2]> {
    slice
        .windows(2)
        .map(|window| unsafe { &*(window.as_ptr() as *const [I; 2]) })
}

#[inline]
#[doc(hidden)]
pub const fn repeat_byte(byte: u8) -> usize {
    union Union {
        bytes: [u8; 32],
        value: usize,
    }

    let data = Union { bytes: [byte; 32] };
    unsafe { data.value }
}

/// Symbolic view structure of sparse matrix in column format, either compressed or uncompressed.
///
/// Requires:
/// * `nrows <= I::Signed::MAX` (always checked)
/// * `ncols <= I::Signed::MAX` (always checked)
/// * `col_ptrs` has length `ncols + 1` (always checked)
/// * `col_ptrs` is non-decreasing
/// * `col_ptrs[0]..col_ptrs[ncols]` is a valid range in row_indices (always checked, assuming
///   non-decreasing)
/// * if `nnz_per_col` is `None`, elements of `row_indices[col_ptrs[j]..col_ptrs[j + 1]]` are less
///   than `nrows`
///
/// * `nnz_per_col[j] <= col_ptrs[j+1] - col_ptrs[j]`
/// * if `nnz_per_col` is `Some(_)`, elements of `row_indices[col_ptrs[j]..][..nnz_per_col[j]]` are
///   less than `nrows`
///
/// * Within each column, row indices are unique and sorted in increasing order.
///
/// # Note
/// Some algorithms allow working with matrices containing duplicate and/or unsorted row
/// indicers per column.
///
/// Passing such a matrix to an algorithm that does not explicitly permit this is unspecified
/// (though not undefined) behavior.
pub struct SymbolicSparseColMatRef<'a, I> {
    nrows: usize,
    ncols: usize,
    col_ptr: &'a [I],
    col_nnz: Option<&'a [I]>,
    row_ind: &'a [I],
}

/// Symbolic view structure of sparse matrix in row format, either compressed or uncompressed.
///
/// Requires:
/// * `nrows <= I::Signed::MAX` (always checked)
/// * `ncols <= I::Signed::MAX` (always checked)
/// * `row_ptrs` has length `nrows + 1` (always checked)
/// * `row_ptrs` is non-decreasing
/// * `row_ptrs[0]..row_ptrs[nrows]` is a valid range in row_indices (always checked, assuming
///   non-decreasing)
/// * if `nnz_per_row` is `None`, elements of `col_indices[row_ptrs[i]..row_ptrs[i + 1]]` are less
///   than `ncols`
///
/// * `nnz_per_row[i] <= row_ptrs[i+1] - row_ptrs[i]`
/// * if `nnz_per_row` is `Some(_)`, elements of `col_indices[row_ptrs[i]..][..nnz_per_row[i]]` are
///   less than `ncols`
///
/// * Within each row, column indices are unique and sorted in increasing order.
///
/// # Note
/// Some algorithms allow working with matrices containing duplicate and/or unsorted column
/// indicers per row.
///
/// Passing such a matrix to an algorithm that does not explicitly permit this is unspecified
/// (though not undefined) behavior.
pub struct SymbolicSparseRowMatRef<'a, I> {
    nrows: usize,
    ncols: usize,
    row_ptr: &'a [I],
    row_nnz: Option<&'a [I]>,
    col_ind: &'a [I],
}

/// Symbolic structure of sparse matrix in column format, either compressed or uncompressed.
///
/// Requires:
/// * `nrows <= I::Signed::MAX` (always checked)
/// * `ncols <= I::Signed::MAX` (always checked)
/// * `col_ptrs` has length `ncols + 1` (always checked)
/// * `col_ptrs` is non-decreasing
/// * `col_ptrs[0]..col_ptrs[ncols]` is a valid range in row_indices (always checked, assuming
///   non-decreasing)
/// * if `nnz_per_col` is `None`, elements of `row_indices[col_ptrs[j]..col_ptrs[j + 1]]` are less
///   than `nrows`
///
/// * `nnz_per_col[j] <= col_ptrs[j+1] - col_ptrs[j]`
/// * if `nnz_per_col` is `Some(_)`, elements of `row_indices[col_ptrs[j]..][..nnz_per_col[j]]` are
///   less than `nrows`
#[derive(Clone)]
pub struct SymbolicSparseColMat<I> {
    nrows: usize,
    ncols: usize,
    col_ptr: Vec<I>,
    col_nnz: Option<Vec<I>>,
    row_ind: Vec<I>,
}

/// Symbolic structure of sparse matrix in row format, either compressed or uncompressed.
///
/// Requires:
/// * `nrows <= I::Signed::MAX` (always checked)
/// * `ncols <= I::Signed::MAX` (always checked)
/// * `row_ptrs` has length `nrows + 1` (always checked)
/// * `row_ptrs` is non-decreasing
/// * `row_ptrs[0]..row_ptrs[nrows]` is a valid range in row_indices (always checked, assuming
///   non-decreasing)
/// * if `nnz_per_row` is `None`, elements of `col_indices[row_ptrs[i]..row_ptrs[i + 1]]` are less
///   than `ncols`
///
/// * `nnz_per_row[i] <= row_ptrs[i+1] - row_ptrs[i]`
/// * if `nnz_per_row` is `Some(_)`, elements of `col_indices[row_ptrs[i]..][..nnz_per_row[i]]` are
///   less than `ncols`
#[derive(Clone)]
pub struct SymbolicSparseRowMat<I> {
    nrows: usize,
    ncols: usize,
    row_ptr: Vec<I>,
    row_nnz: Option<Vec<I>>,
    col_ind: Vec<I>,
}

impl<I> Copy for SymbolicSparseColMatRef<'_, I> {}
impl<I> Clone for SymbolicSparseColMatRef<'_, I> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<I> Copy for SymbolicSparseRowMatRef<'_, I> {}
impl<I> Clone for SymbolicSparseRowMatRef<'_, I> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<I: Index> SymbolicSparseRowMat<I> {
    /// Creates a new symbolic matrix view after asserting its invariants.
    ///
    /// # Panics
    ///
    /// See type level documentation.
    #[inline]
    #[track_caller]
    pub fn new_checked(
        nrows: usize,
        ncols: usize,
        row_ptrs: Vec<I>,
        nnz_per_row: Option<Vec<I>>,
        col_indices: Vec<I>,
    ) -> Self {
        SymbolicSparseRowMatRef::new_checked(
            nrows,
            ncols,
            &row_ptrs,
            nnz_per_row.as_deref(),
            &col_indices,
        );

        Self {
            nrows,
            ncols,
            row_ptr: row_ptrs,
            row_nnz: nnz_per_row,
            col_ind: col_indices,
        }
    }

    /// Creates a new symbolic matrix view from data containing duplicate and/or unsorted column
    /// indices per row, after asserting its other invariants.
    ///
    /// # Panics
    ///
    /// See type level documentation.
    #[inline]
    #[track_caller]
    pub fn new_unsorted_checked(
        nrows: usize,
        ncols: usize,
        row_ptrs: Vec<I>,
        nnz_per_row: Option<Vec<I>>,
        col_indices: Vec<I>,
    ) -> Self {
        SymbolicSparseRowMatRef::new_unsorted_checked(
            nrows,
            ncols,
            &row_ptrs,
            nnz_per_row.as_deref(),
            &col_indices,
        );

        Self {
            nrows,
            ncols,
            row_ptr: row_ptrs,
            row_nnz: nnz_per_row,
            col_ind: col_indices,
        }
    }

    /// Creates a new symbolic matrix view without asserting its invariants.
    ///
    /// # Safety
    ///
    /// See type level documentation.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn new_unchecked(
        nrows: usize,
        ncols: usize,
        row_ptrs: Vec<I>,
        nnz_per_row: Option<Vec<I>>,
        col_indices: Vec<I>,
    ) -> Self {
        SymbolicSparseRowMatRef::new_unchecked(
            nrows,
            ncols,
            &row_ptrs,
            nnz_per_row.as_deref(),
            &col_indices,
        );

        Self {
            nrows,
            ncols,
            row_ptr: row_ptrs,
            row_nnz: nnz_per_row,
            col_ind: col_indices,
        }
    }

    /// Returns the components of the matrix in the order:
    /// - row count,
    /// - column count,
    /// - row pointers,
    /// - nonzeros per row,
    /// - column indices.
    #[inline]
    pub fn into_parts(self) -> (usize, usize, Vec<I>, Option<Vec<I>>, Vec<I>) {
        (
            self.nrows,
            self.ncols,
            self.row_ptr,
            self.row_nnz,
            self.col_ind,
        )
    }

    /// Returns a view over the symbolic structure of `self`.
    #[inline]
    pub fn as_ref(&self) -> SymbolicSparseRowMatRef<'_, I> {
        SymbolicSparseRowMatRef {
            nrows: self.nrows,
            ncols: self.ncols,
            row_ptr: &self.row_ptr,
            row_nnz: self.row_nnz.as_deref(),
            col_ind: &self.col_ind,
        }
    }

    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Consumes the matrix, and returns its transpose in column-major format without reallocating.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn into_transpose(self) -> SymbolicSparseColMat<I> {
        SymbolicSparseColMat {
            nrows: self.ncols,
            ncols: self.nrows,
            col_ptr: self.row_ptr,
            col_nnz: self.row_nnz,
            row_ind: self.col_ind,
        }
    }

    /// Copies the current matrix into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SymbolicSparseRowMat<I>, FaerError> {
        self.as_ref().to_owned()
    }

    /// Copies the current matrix into a newly allocated matrix, with column-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_col_major(&self) -> Result<SymbolicSparseColMat<I>, FaerError> {
        self.as_ref().to_col_major()
    }

    /// Returns the number of symbolic non-zeros in the matrix.
    ///
    /// The value is guaranteed to be less than `I::Signed::MAX`.
    ///
    /// # Note
    /// Allows unsorted matrices, but the output is a count of all the entries, including the
    /// duplicate ones.
    #[inline]
    pub fn compute_nnz(&self) -> usize {
        self.as_ref().compute_nnz()
    }

    /// Returns the column pointers.
    #[inline]
    pub fn row_ptrs(&self) -> &[I] {
        &self.row_ptr
    }

    /// Returns the count of non-zeros per row of the matrix.
    #[inline]
    pub fn nnz_per_row(&self) -> Option<&[I]> {
        self.row_nnz.as_deref()
    }

    /// Returns the column indices.
    #[inline]
    pub fn col_indices(&self) -> &[I] {
        &self.col_ind
    }

    /// Returns the column indices of row `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub fn col_indices_of_row_raw(&self, i: usize) -> &[I] {
        &self.col_ind[self.row_range(i)]
    }

    /// Returns the column indices of row `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn col_indices_of_row(
        &self,
        i: usize,
    ) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = usize> {
        self.as_ref().col_indices_of_row(i)
    }

    /// Returns the range that the row `i` occupies in `self.col_indices()`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub fn row_range(&self, i: usize) -> Range<usize> {
        self.as_ref().row_range(i)
    }

    /// Returns the range that the row `i` occupies in `self.col_indices()`.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub unsafe fn row_range_unchecked(&self, i: usize) -> Range<usize> {
        self.as_ref().row_range_unchecked(i)
    }
}

impl<I: Index> SymbolicSparseColMat<I> {
    /// Creates a new symbolic matrix view after asserting its invariants.
    ///
    /// # Panics
    ///
    /// See type level documentation.
    #[inline]
    #[track_caller]
    pub fn new_checked(
        nrows: usize,
        ncols: usize,
        col_ptrs: Vec<I>,
        nnz_per_col: Option<Vec<I>>,
        row_indices: Vec<I>,
    ) -> Self {
        SymbolicSparseColMatRef::new_checked(
            nrows,
            ncols,
            &col_ptrs,
            nnz_per_col.as_deref(),
            &row_indices,
        );

        Self {
            nrows,
            ncols,
            col_ptr: col_ptrs,
            col_nnz: nnz_per_col,
            row_ind: row_indices,
        }
    }

    /// Creates a new symbolic matrix view from data containing duplicate and/or unsorted row
    /// indices per column, after asserting its other invariants.
    ///
    /// # Panics
    ///
    /// See type level documentation.
    #[inline]
    #[track_caller]
    pub fn new_unsorted_checked(
        nrows: usize,
        ncols: usize,
        col_ptrs: Vec<I>,
        nnz_per_col: Option<Vec<I>>,
        row_indices: Vec<I>,
    ) -> Self {
        SymbolicSparseColMatRef::new_unsorted_checked(
            nrows,
            ncols,
            &col_ptrs,
            nnz_per_col.as_deref(),
            &row_indices,
        );

        Self {
            nrows,
            ncols,
            col_ptr: col_ptrs,
            col_nnz: nnz_per_col,
            row_ind: row_indices,
        }
    }

    /// Creates a new symbolic matrix view without asserting its invariants.
    ///
    /// # Safety
    ///
    /// See type level documentation.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn new_unchecked(
        nrows: usize,
        ncols: usize,
        col_ptrs: Vec<I>,
        nnz_per_col: Option<Vec<I>>,
        row_indices: Vec<I>,
    ) -> Self {
        SymbolicSparseRowMatRef::new_unchecked(
            nrows,
            ncols,
            &col_ptrs,
            nnz_per_col.as_deref(),
            &row_indices,
        );

        Self {
            nrows,
            ncols,
            col_ptr: col_ptrs,
            col_nnz: nnz_per_col,
            row_ind: row_indices,
        }
    }

    /// Returns the components of the matrix in the order:
    /// - row count,
    /// - column count,
    /// - column pointers,
    /// - nonzeros per column,
    /// - row indices.
    #[inline]
    pub fn into_parts(self) -> (usize, usize, Vec<I>, Option<Vec<I>>, Vec<I>) {
        (
            self.nrows,
            self.ncols,
            self.col_ptr,
            self.col_nnz,
            self.row_ind,
        )
    }

    /// Returns a view over the symbolic structure of `self`.
    #[inline]
    pub fn as_ref(&self) -> SymbolicSparseColMatRef<'_, I> {
        SymbolicSparseColMatRef {
            nrows: self.nrows,
            ncols: self.ncols,
            col_ptr: &self.col_ptr,
            col_nnz: self.col_nnz.as_deref(),
            row_ind: &self.row_ind,
        }
    }

    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Consumes the matrix, and returns its transpose in row-major format without reallocating.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn into_transpose(self) -> SymbolicSparseRowMat<I> {
        SymbolicSparseRowMat {
            nrows: self.ncols,
            ncols: self.nrows,
            row_ptr: self.col_ptr,
            row_nnz: self.col_nnz,
            col_ind: self.row_ind,
        }
    }

    /// Copies the current matrix into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SymbolicSparseColMat<I>, FaerError> {
        self.as_ref().to_owned()
    }

    /// Copies the current matrix into a newly allocated matrix, with row-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_row_major(&self) -> Result<SymbolicSparseRowMat<I>, FaerError> {
        self.as_ref().to_row_major()
    }

    /// Returns the number of symbolic non-zeros in the matrix.
    ///
    /// The value is guaranteed to be less than `I::Signed::MAX`.
    ///
    /// # Note
    /// Allows unsorted matrices, but the output is a count of all the entries, including the
    /// duplicate ones.
    #[inline]
    pub fn compute_nnz(&self) -> usize {
        self.as_ref().compute_nnz()
    }

    /// Returns the column pointers.
    #[inline]
    pub fn col_ptrs(&self) -> &[I] {
        &self.col_ptr
    }

    /// Returns the count of non-zeros per column of the matrix.
    #[inline]
    pub fn nnz_per_col(&self) -> Option<&[I]> {
        self.col_nnz.as_deref()
    }

    /// Returns the row indices.
    #[inline]
    pub fn row_indices(&self) -> &[I] {
        &self.row_ind
    }

    /// Returns the row indices of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn row_indices_of_col_raw(&self, j: usize) -> &[I] {
        &self.row_ind[self.col_range(j)]
    }

    /// Returns the row indices of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn row_indices_of_col(
        &self,
        j: usize,
    ) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = usize> {
        self.as_ref().row_indices_of_col(j)
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn col_range(&self, j: usize) -> Range<usize> {
        self.as_ref().col_range(j)
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub unsafe fn col_range_unchecked(&self, j: usize) -> Range<usize> {
        self.as_ref().col_range_unchecked(j)
    }
}

impl<'a, I: Index> SymbolicSparseRowMatRef<'a, I> {
    /// Creates a new symbolic matrix view after asserting its invariants.
    ///
    /// # Panics
    ///
    /// See type level documentation.
    #[inline]
    #[track_caller]
    pub fn new_checked(
        nrows: usize,
        ncols: usize,
        row_ptrs: &'a [I],
        nnz_per_row: Option<&'a [I]>,
        col_indices: &'a [I],
    ) -> Self {
        assert!(all(
            ncols <= I::Signed::MAX.zx(),
            nrows <= I::Signed::MAX.zx(),
        ));
        assert!(row_ptrs.len() == nrows + 1);
        for &[c, c_next] in windows2(row_ptrs) {
            assert!(c <= c_next);
        }
        assert!(row_ptrs[ncols].zx() <= col_indices.len());

        if let Some(nnz_per_row) = nnz_per_row {
            for (&nnz_i, &[c, c_next]) in zip(nnz_per_row, windows2(row_ptrs)) {
                assert!(nnz_i <= c_next - c);
                let col_indices = &col_indices[c.zx()..c.zx() + nnz_i.zx()];
                if !col_indices.is_empty() {
                    let mut j_prev = col_indices[0];
                    for &j in &col_indices[1..] {
                        assert!(j_prev < j);
                        j_prev = j;
                    }
                    let ncols = I::truncate(ncols);
                    assert!(j_prev < ncols);
                }
            }
        } else {
            for &[c, c_next] in windows2(row_ptrs) {
                let col_indices = &col_indices[c.zx()..c_next.zx()];
                if !col_indices.is_empty() {
                    let mut j_prev = col_indices[0];
                    for &j in &col_indices[1..] {
                        assert!(j_prev < j);
                        j_prev = j;
                    }
                    let ncols = I::truncate(ncols);
                    assert!(j_prev < ncols);
                }
            }
        }

        Self {
            nrows,
            ncols,
            row_ptr: row_ptrs,
            row_nnz: nnz_per_row,
            col_ind: col_indices,
        }
    }

    /// Creates a new symbolic matrix view from data containing duplicate and/or unsorted column
    /// indices per row, after asserting its other invariants.
    ///
    /// # Panics
    ///
    /// See type level documentation.
    #[inline]
    #[track_caller]
    pub fn new_unsorted_checked(
        nrows: usize,
        ncols: usize,
        row_ptrs: &'a [I],
        nnz_per_row: Option<&'a [I]>,
        col_indices: &'a [I],
    ) -> Self {
        assert!(all(
            ncols <= I::Signed::MAX.zx(),
            nrows <= I::Signed::MAX.zx(),
        ));
        assert!(row_ptrs.len() == nrows + 1);
        for &[c, c_next] in windows2(row_ptrs) {
            assert!(c <= c_next);
        }
        assert!(row_ptrs[ncols].zx() <= col_indices.len());

        if let Some(nnz_per_row) = nnz_per_row {
            for (&nnz_i, &[c, c_next]) in zip(nnz_per_row, windows2(row_ptrs)) {
                assert!(nnz_i <= c_next - c);
                for &j in &col_indices[c.zx()..c.zx() + nnz_i.zx()] {
                    assert!(j < I::truncate(ncols));
                }
            }
        } else {
            let c0 = row_ptrs[0].zx();
            let cn = row_ptrs[ncols].zx();
            for &j in &col_indices[c0..cn] {
                assert!(j < I::truncate(ncols));
            }
        }

        Self {
            nrows,
            ncols,
            row_ptr: row_ptrs,
            row_nnz: nnz_per_row,
            col_ind: col_indices,
        }
    }

    /// Creates a new symbolic matrix view without asserting its invariants.
    ///
    /// # Safety
    ///
    /// See type level documentation.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn new_unchecked(
        nrows: usize,
        ncols: usize,
        row_ptrs: &'a [I],
        nnz_per_row: Option<&'a [I]>,
        col_indices: &'a [I],
    ) -> Self {
        assert!(all(
            ncols <= <I::Signed as SignedIndex>::MAX.zx(),
            nrows <= <I::Signed as SignedIndex>::MAX.zx(),
        ));
        assert!(row_ptrs.len() == nrows + 1);
        assert!(row_ptrs[nrows].zx() <= col_indices.len());

        Self {
            nrows,
            ncols,
            row_ptr: row_ptrs,
            row_nnz: nnz_per_row,
            col_ind: col_indices,
        }
    }

    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns a view over the transpose of `self` in column-major format.
    #[inline]
    pub fn transpose(self) -> SymbolicSparseColMatRef<'a, I> {
        SymbolicSparseColMatRef {
            nrows: self.ncols,
            ncols: self.nrows,
            col_ptr: self.row_ptr,
            col_nnz: self.row_nnz,
            row_ind: self.col_ind,
        }
    }

    /// Copies the current matrix into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SymbolicSparseRowMat<I>, FaerError> {
        self.transpose()
            .to_owned()
            .map(SymbolicSparseColMat::into_transpose)
    }

    /// Copies the current matrix into a newly allocated matrix, with column-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_col_major(&self) -> Result<SymbolicSparseColMat<I>, FaerError> {
        self.transpose().to_row_major().map(|m| m.into_transpose())
    }

    /// Returns the number of symbolic non-zeros in the matrix.
    ///
    /// The value is guaranteed to be less than `I::Signed::MAX`.
    ///
    /// # Note
    /// Allows unsorted matrices, but the output is a count of all the entries, including the
    /// duplicate ones.
    #[inline]
    pub fn compute_nnz(&self) -> usize {
        self.transpose().compute_nnz()
    }

    /// Returns the column pointers.
    #[inline]
    pub fn row_ptrs(&self) -> &'a [I] {
        self.row_ptr
    }

    /// Returns the count of non-zeros per column of the matrix.
    #[inline]
    pub fn nnz_per_row(&self) -> Option<&'a [I]> {
        self.row_nnz
    }

    /// Returns the column indices.
    #[inline]
    pub fn col_indices(&self) -> &'a [I] {
        self.col_ind
    }

    /// Returns the column indices of row i.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub fn col_indices_of_row_raw(&self, i: usize) -> &'a [I] {
        &self.col_ind[self.row_range(i)]
    }

    /// Returns the column indices of row i.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn col_indices_of_row(
        &self,
        i: usize,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = usize> {
        self.col_indices_of_row_raw(i).iter().map(
            #[inline(always)]
            |&i| i.zx(),
        )
    }

    /// Returns the range that the row `i` occupies in `self.col_indices()`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub fn row_range(&self, i: usize) -> Range<usize> {
        let start = self.row_ptr[i].zx();
        let end = self
            .row_nnz
            .map(|row_nnz| row_nnz[i].zx() + start)
            .unwrap_or(self.row_ptr[i + 1].zx());

        start..end
    }

    /// Returns the range that the row `i` occupies in `self.col_indices()`.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub unsafe fn row_range_unchecked(&self, i: usize) -> Range<usize> {
        let start = __get_unchecked(self.row_ptr, i).zx();
        let end = self
            .row_nnz
            .map(|row_nnz| (__get_unchecked(row_nnz, i).zx() + start))
            .unwrap_or(__get_unchecked(self.row_ptr, i + 1).zx());

        start..end
    }
}

impl<'a, I: Index> SymbolicSparseColMatRef<'a, I> {
    /// Creates a new symbolic matrix view after asserting its invariants.
    ///
    /// # Panics
    ///
    /// See type level documentation.
    #[inline]
    #[track_caller]
    pub fn new_checked(
        nrows: usize,
        ncols: usize,
        col_ptrs: &'a [I],
        nnz_per_col: Option<&'a [I]>,
        row_indices: &'a [I],
    ) -> Self {
        assert!(all(
            ncols <= I::Signed::MAX.zx(),
            nrows <= I::Signed::MAX.zx(),
        ));
        assert!(col_ptrs.len() == ncols + 1);
        for &[c, c_next] in windows2(col_ptrs) {
            assert!(c <= c_next);
        }
        assert!(col_ptrs[ncols].zx() <= row_indices.len());

        if let Some(nnz_per_col) = nnz_per_col {
            for (&nnz_j, &[c, c_next]) in zip(nnz_per_col, windows2(col_ptrs)) {
                assert!(nnz_j <= c_next - c);
                let row_indices = &row_indices[c.zx()..c.zx() + nnz_j.zx()];
                if !row_indices.is_empty() {
                    let mut i_prev = row_indices[0];
                    for &i in &row_indices[1..] {
                        assert!(i_prev < i);
                        i_prev = i;
                    }
                    let nrows = I::truncate(nrows);
                    assert!(i_prev < nrows);
                }
            }
        } else {
            for &[c, c_next] in windows2(col_ptrs) {
                let row_indices = &row_indices[c.zx()..c_next.zx()];
                if !row_indices.is_empty() {
                    let mut i_prev = row_indices[0];
                    for &i in &row_indices[1..] {
                        assert!(i_prev < i);
                        i_prev = i;
                    }
                    let nrows = I::truncate(nrows);
                    assert!(i_prev < nrows);
                }
            }
        }

        Self {
            nrows,
            ncols,
            col_ptr: col_ptrs,
            col_nnz: nnz_per_col,
            row_ind: row_indices,
        }
    }

    /// Creates a new symbolic matrix view from data containing duplicate and/or unsorted row
    /// indices per column, after asserting its other invariants.
    ///
    /// # Panics
    ///
    /// See type level documentation.
    #[inline]
    #[track_caller]
    pub fn new_unsorted_checked(
        nrows: usize,
        ncols: usize,
        col_ptrs: &'a [I],
        nnz_per_col: Option<&'a [I]>,
        row_indices: &'a [I],
    ) -> Self {
        assert!(all(
            ncols <= I::Signed::MAX.zx(),
            nrows <= I::Signed::MAX.zx(),
        ));
        assert!(col_ptrs.len() == ncols + 1);
        for &[c, c_next] in windows2(col_ptrs) {
            assert!(c <= c_next);
        }
        assert!(col_ptrs[ncols].zx() <= row_indices.len());

        if let Some(nnz_per_col) = nnz_per_col {
            for (&nnz_j, &[c, c_next]) in zip(nnz_per_col, windows2(col_ptrs)) {
                assert!(nnz_j <= c_next - c);
                for &i in &row_indices[c.zx()..c.zx() + nnz_j.zx()] {
                    assert!(i < I::truncate(nrows));
                }
            }
        } else {
            let c0 = col_ptrs[0].zx();
            let cn = col_ptrs[ncols].zx();
            for &i in &row_indices[c0..cn] {
                assert!(i < I::truncate(nrows));
            }
        }

        Self {
            nrows,
            ncols,
            col_ptr: col_ptrs,
            col_nnz: nnz_per_col,
            row_ind: row_indices,
        }
    }

    /// Creates a new symbolic matrix view without asserting its invariants.
    ///
    /// # Safety
    ///
    /// See type level documentation.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn new_unchecked(
        nrows: usize,
        ncols: usize,
        col_ptrs: &'a [I],
        nnz_per_col: Option<&'a [I]>,
        row_indices: &'a [I],
    ) -> Self {
        assert!(all(
            ncols <= <I::Signed as SignedIndex>::MAX.zx(),
            nrows <= <I::Signed as SignedIndex>::MAX.zx(),
        ));
        assert!(col_ptrs.len() == ncols + 1);
        assert!(col_ptrs[ncols].zx() <= row_indices.len());

        Self {
            nrows,
            ncols,
            col_ptr: col_ptrs,
            col_nnz: nnz_per_col,
            row_ind: row_indices,
        }
    }

    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns a view over the transpose of `self` in row-major format.
    #[inline]
    pub fn transpose(self) -> SymbolicSparseRowMatRef<'a, I> {
        SymbolicSparseRowMatRef {
            nrows: self.ncols,
            ncols: self.nrows,
            row_ptr: self.col_ptr,
            row_nnz: self.col_nnz,
            col_ind: self.row_ind,
        }
    }

    /// Copies the current matrix into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SymbolicSparseColMat<I>, FaerError> {
        Ok(SymbolicSparseColMat {
            nrows: self.nrows,
            ncols: self.ncols,
            col_ptr: try_collect(self.col_ptr.iter().copied())?,
            col_nnz: self
                .col_nnz
                .map(|nnz| try_collect(nnz.iter().copied()))
                .transpose()?,
            row_ind: try_collect(self.row_ind.iter().copied())?,
        })
    }

    /// Copies the current matrix into a newly allocated matrix, with row-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_row_major(&self) -> Result<SymbolicSparseRowMat<I>, FaerError> {
        let mut col_ptr = try_zeroed::<I>(self.nrows + 1)?;
        let mut row_ind = try_zeroed::<I>(self.compute_nnz())?;

        let mut mem = GlobalPodBuffer::try_new(StackReq::new::<I>(self.nrows))
            .map_err(|_| FaerError::OutOfMemory)?;

        util::adjoint_symbolic(&mut col_ptr, &mut row_ind, *self, PodStack::new(&mut mem));

        let transpose = unsafe {
            SymbolicSparseColMat::new_unchecked(self.ncols, self.nrows, col_ptr, None, row_ind)
        };

        Ok(transpose.into_transpose())
    }

    /// Returns the number of symbolic non-zeros in the matrix.
    ///
    /// The value is guaranteed to be less than `I::Signed::MAX`.
    ///
    /// # Note
    /// Allows unsorted matrices, but the output is a count of all the entries, including the
    /// duplicate ones.
    #[inline]
    pub fn compute_nnz(&self) -> usize {
        match self.col_nnz {
            Some(col_nnz) => {
                let mut nnz = 0usize;
                for &nnz_j in col_nnz {
                    // can't overflow
                    nnz += nnz_j.zx();
                }
                nnz
            }
            None => self.col_ptr[self.ncols].zx() - self.col_ptr[0].zx(),
        }
    }

    /// Returns the column pointers.
    #[inline]
    pub fn col_ptrs(&self) -> &'a [I] {
        self.col_ptr
    }

    /// Returns the count of non-zeros per column of the matrix.
    #[inline]
    pub fn nnz_per_col(&self) -> Option<&'a [I]> {
        self.col_nnz
    }

    /// Returns the row indices.
    #[inline]
    pub fn row_indices(&self) -> &'a [I] {
        self.row_ind
    }

    /// Returns the row indices of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn row_indices_of_col_raw(&self, j: usize) -> &'a [I] {
        &self.row_ind[self.col_range(j)]
    }

    /// Returns the row indices of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn row_indices_of_col(
        &self,
        j: usize,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = usize> {
        self.row_indices_of_col_raw(j).iter().map(
            #[inline(always)]
            |&i| i.zx(),
        )
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn col_range(&self, j: usize) -> Range<usize> {
        let start = self.col_ptr[j].zx();
        let end = self
            .col_nnz
            .map(|col_nnz| col_nnz[j].zx() + start)
            .unwrap_or(self.col_ptr[j + 1].zx());

        start..end
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub unsafe fn col_range_unchecked(&self, j: usize) -> Range<usize> {
        let start = __get_unchecked(self.col_ptr, j).zx();
        let end = self
            .col_nnz
            .map(|col_nnz| (__get_unchecked(col_nnz, j).zx() + start))
            .unwrap_or(__get_unchecked(self.col_ptr, j + 1).zx());

        start..end
    }
}

impl<I: Index> core::fmt::Debug for SymbolicSparseColMatRef<'_, I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mat = *self;
        let mut iter = (0..mat.ncols()).into_iter().flat_map(move |j| {
            struct Wrapper(usize, usize);
            impl core::fmt::Debug for Wrapper {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    let row = self.0;
                    let col = self.1;
                    write!(f, "({row}, {col}")
                }
            }

            mat.row_indices_of_col(j).map(move |i| Wrapper(i, j))
        });

        f.debug_list().entries(&mut iter).finish()
    }
}

impl<I: Index> core::fmt::Debug for SymbolicSparseRowMatRef<'_, I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mat = *self;
        let mut iter = (0..mat.nrows()).into_iter().flat_map(move |i| {
            struct Wrapper(usize, usize);
            impl core::fmt::Debug for Wrapper {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    let row = self.0;
                    let col = self.1;
                    write!(f, "({row}, {col}")
                }
            }

            mat.col_indices_of_row(i).map(move |j| Wrapper(i, j))
        });

        f.debug_list().entries(&mut iter).finish()
    }
}
impl<I: Index, E: Entity> core::fmt::Debug for SparseColMatRef<'_, I, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mat = *self;
        let mut iter = (0..mat.ncols()).into_iter().flat_map(move |j| {
            struct Wrapper<E>(usize, usize, E);
            impl<E: core::fmt::Debug> core::fmt::Debug for Wrapper<E> {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    let row = self.0;
                    let col = self.1;
                    let val = &self.2;
                    write!(f, "({row}, {col}, {val:?})")
                }
            }

            mat.row_indices_of_col(j)
                .zip(SliceGroup::<E>::new(mat.values_of_col(j)).into_ref_iter())
                .map(move |(i, val)| Wrapper(i, j, val.read()))
        });

        f.debug_list().entries(&mut iter).finish()
    }
}

impl<I: Index, E: Entity> core::fmt::Debug for SparseRowMatRef<'_, I, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mat = *self;
        let mut iter = (0..mat.nrows()).into_iter().flat_map(move |i| {
            struct Wrapper<E>(usize, usize, E);
            impl<E: core::fmt::Debug> core::fmt::Debug for Wrapper<E> {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    let row = self.0;
                    let col = self.1;
                    let val = &self.2;
                    write!(f, "({row}, {col}, {val:?})")
                }
            }

            mat.col_indices_of_row(i)
                .zip(SliceGroup::<E>::new(mat.values_of_row(i)).into_ref_iter())
                .map(move |(j, val)| Wrapper(i, j, val.read()))
        });

        f.debug_list().entries(&mut iter).finish()
    }
}

impl<I: Index, E: Entity> core::fmt::Debug for SparseColMatMut<'_, I, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<I: Index, E: Entity> core::fmt::Debug for SparseColMat<I, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<I: Index> core::fmt::Debug for SymbolicSparseColMat<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<I: Index, E: Entity> core::fmt::Debug for SparseRowMatMut<'_, I, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<I: Index, E: Entity> core::fmt::Debug for SparseRowMat<I, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<I: Index> core::fmt::Debug for SymbolicSparseRowMat<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

/// Sparse matrix mutable view in row-major format, either compressed or uncompressed.
///
/// Note that only the values are modifiable through this view. The row pointers and column
/// indices are fixed.
pub type SparseRowMatMut<'a, I, E> = Matrix<inner::SparseRowMatMut<'a, I, E>>;

/// Sparse matrix mutable view in column-major format, either compressed or uncompressed.
///
/// Note that only the values are modifiable through this view. The column pointers and row indices
/// are fixed.
pub type SparseColMatMut<'a, I, E> = Matrix<inner::SparseColMatMut<'a, I, E>>;

/// Sparse matrix view in row-major format, either compressed or uncompressed.
pub type SparseRowMatRef<'a, I, E> = Matrix<inner::SparseRowMatRef<'a, I, E>>;

/// Sparse matrix view in column-major format, either compressed or uncompressed.
pub type SparseColMatRef<'a, I, E> = Matrix<inner::SparseColMatRef<'a, I, E>>;

/// Sparse matrix in row-major format, either compressed or uncompressed.
pub type SparseRowMat<I, E> = Matrix<inner::SparseRowMat<I, E>>;

/// Sparse matrix in column-major format, either compressed or uncompressed.
pub type SparseColMat<I, E> = Matrix<inner::SparseColMat<I, E>>;

impl<'a, I: Index, E: Entity> SparseRowMatMut<'a, I, E> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.col_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(
        symbolic: SymbolicSparseRowMatRef<'a, I>,
        values: GroupFor<E, &'a mut [E::Unit]>,
    ) -> Self {
        let values = SliceGroupMut::new(values);
        assert!(symbolic.col_indices().len() == values.len());
        Self {
            inner: inner::SparseRowMatMut { symbolic, values },
        }
    }

    /// Copies the current matrix into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SparseRowMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.rb().to_owned()
    }

    /// Copies the current matrix into a newly allocated matrix, with column-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_col_major(&self) -> Result<SparseColMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.rb().to_col_major()
    }

    /// Returns a view over the transpose of `self` in column-major format.
    #[inline]
    pub fn transpose_mut(self) -> SparseColMatMut<'a, I, E> {
        SparseColMatMut {
            inner: inner::SparseColMatMut {
                symbolic: SymbolicSparseColMatRef {
                    nrows: self.inner.symbolic.ncols,
                    ncols: self.inner.symbolic.nrows,
                    col_ptr: self.inner.symbolic.row_ptr,
                    col_nnz: self.inner.symbolic.row_nnz,
                    row_ind: self.inner.symbolic.col_ind,
                },
                values: self.inner.values,
            },
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn canonicalize_mut(self) -> (SparseRowMatMut<'a, I, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        (
            SparseRowMatMut {
                inner: inner::SparseRowMatMut {
                    symbolic: self.inner.symbolic,
                    values: unsafe {
                        SliceGroupMut::<'a, E::Canonical>::new(transmute_unchecked::<
                            GroupFor<E, &mut [UnitFor<E::Canonical>]>,
                            GroupFor<E::Canonical, &mut [UnitFor<E::Canonical>]>,
                        >(
                            E::faer_map(self.inner.values.into_inner(), |slice| {
                                let len = slice.len();
                                core::slice::from_raw_parts_mut(
                                    slice.as_mut_ptr() as *mut UnitFor<E>
                                        as *mut UnitFor<E::Canonical>,
                                    len,
                                )
                            }),
                        ))
                    },
                },
            },
            if coe::is_same::<E, E::Canonical>() {
                Conj::No
            } else {
                Conj::Yes
            },
        )
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate_mut(self) -> SparseRowMatMut<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        SparseRowMatMut {
            inner: inner::SparseRowMatMut {
                symbolic: self.inner.symbolic,
                values: unsafe {
                    SliceGroupMut::<'a, E::Conj>::new(transmute_unchecked::<
                        GroupFor<E, &mut [UnitFor<E::Conj>]>,
                        GroupFor<E::Conj, &mut [UnitFor<E::Conj>]>,
                    >(E::faer_map(
                        self.inner.values.into_inner(),
                        |slice| {
                            let len = slice.len();
                            core::slice::from_raw_parts_mut(
                                slice.as_mut_ptr() as *mut UnitFor<E> as *mut UnitFor<E::Conj>,
                                len,
                            )
                        },
                    )))
                },
            },
        }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint_mut(self) -> SparseColMatMut<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        self.transpose_mut().conjugate_mut()
    }

    /// Returns the numerical values of the matrix.
    #[inline]
    pub fn values_mut(self) -> GroupFor<E, &'a mut [E::Unit]> {
        self.inner.values.into_inner()
    }

    /// Returns the numerical values of row `i` of the matrix.
    ///
    /// # Panics:
    ///
    /// Panics if `i >= nrows`.
    #[inline]
    #[track_caller]
    pub fn values_of_row_mut(self, i: usize) -> GroupFor<E, &'a mut [E::Unit]> {
        let range = self.symbolic().row_range(i);
        self.inner.values.subslice(range).into_inner()
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'a, I> {
        self.inner.symbolic
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn into_parts(
        self,
    ) -> (
        SymbolicSparseRowMatRef<'a, I>,
        GroupFor<E, &'a mut [E::Unit]>,
    ) {
        (self.inner.symbolic, self.inner.values.into_inner())
    }
}

impl<'a, I: Index, E: Entity> SparseColMatMut<'a, I, E> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.row_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(
        symbolic: SymbolicSparseColMatRef<'a, I>,
        values: GroupFor<E, &'a mut [E::Unit]>,
    ) -> Self {
        let values = SliceGroupMut::new(values);
        assert!(symbolic.row_indices().len() == values.len());
        Self {
            inner: inner::SparseColMatMut { symbolic, values },
        }
    }

    /// Copies the current matrix into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SparseColMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.rb().to_owned()
    }

    /// Copies the current matrix into a newly allocated matrix, with row-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_row_major(&self) -> Result<SparseRowMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.rb().to_row_major()
    }

    /// Returns a view over the transpose of `self` in row-major format.
    #[inline]
    pub fn transpose_mut(self) -> SparseRowMatMut<'a, I, E> {
        SparseRowMatMut {
            inner: inner::SparseRowMatMut {
                symbolic: SymbolicSparseRowMatRef {
                    nrows: self.inner.symbolic.ncols,
                    ncols: self.inner.symbolic.nrows,
                    row_ptr: self.inner.symbolic.col_ptr,
                    row_nnz: self.inner.symbolic.col_nnz,
                    col_ind: self.inner.symbolic.row_ind,
                },
                values: self.inner.values,
            },
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate_mut(self) -> SparseColMatMut<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        SparseColMatMut {
            inner: inner::SparseColMatMut {
                symbolic: self.inner.symbolic,
                values: unsafe {
                    SliceGroupMut::<'a, E::Conj>::new(transmute_unchecked::<
                        GroupFor<E, &mut [UnitFor<E::Conj>]>,
                        GroupFor<E::Conj, &mut [UnitFor<E::Conj>]>,
                    >(E::faer_map(
                        self.inner.values.into_inner(),
                        |slice| {
                            let len = slice.len();
                            core::slice::from_raw_parts_mut(
                                slice.as_ptr() as *mut UnitFor<E> as *mut UnitFor<E::Conj>,
                                len,
                            )
                        },
                    )))
                },
            },
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn canonicalize_mut(self) -> (SparseColMatMut<'a, I, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        (
            SparseColMatMut {
                inner: inner::SparseColMatMut {
                    symbolic: self.inner.symbolic,
                    values: unsafe {
                        SliceGroupMut::<'a, E::Canonical>::new(transmute_unchecked::<
                            GroupFor<E, &mut [UnitFor<E::Canonical>]>,
                            GroupFor<E::Canonical, &mut [UnitFor<E::Canonical>]>,
                        >(
                            E::faer_map(self.inner.values.into_inner(), |slice| {
                                let len = slice.len();
                                core::slice::from_raw_parts_mut(
                                    slice.as_mut_ptr() as *mut UnitFor<E>
                                        as *mut UnitFor<E::Canonical>,
                                    len,
                                )
                            }),
                        ))
                    },
                },
            },
            if coe::is_same::<E, E::Canonical>() {
                Conj::No
            } else {
                Conj::Yes
            },
        )
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint_mut(self) -> SparseRowMatMut<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        self.transpose_mut().conjugate_mut()
    }

    /// Returns the numerical values of the matrix.
    #[inline]
    pub fn values_mut(self) -> GroupFor<E, &'a mut [E::Unit]> {
        self.inner.values.into_inner()
    }

    /// Returns the numerical values of column `j` of the matrix.
    ///
    /// # Panics:
    ///
    /// Panics if `j >= ncols`.
    #[inline]
    #[track_caller]
    pub fn values_of_col_mut(self, j: usize) -> GroupFor<E, &'a mut [E::Unit]> {
        let range = self.symbolic().col_range(j);
        self.inner.values.subslice(range).into_inner()
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseColMatRef<'a, I> {
        self.inner.symbolic
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn into_parts_mut(
        self,
    ) -> (
        SymbolicSparseColMatRef<'a, I>,
        GroupFor<E, &'a mut [E::Unit]>,
    ) {
        (self.inner.symbolic, self.inner.values.into_inner())
    }
}

impl<'a, I: Index, E: Entity> SparseRowMatRef<'a, I, E> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.col_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(
        symbolic: SymbolicSparseRowMatRef<'a, I>,
        values: GroupFor<E, &'a [E::Unit]>,
    ) -> Self {
        let values = SliceGroup::new(values);
        assert!(symbolic.col_indices().len() == values.len());
        Self {
            inner: inner::SparseRowMatRef { symbolic, values },
        }
    }

    /// Returns the numerical values of the matrix.
    #[inline]
    pub fn values(self) -> GroupFor<E, &'a [E::Unit]> {
        self.inner.values.into_inner()
    }

    /// Copies the current matrix into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SparseRowMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.transpose()
            .to_owned()
            .map(SparseColMat::into_transpose)
    }

    /// Copies the current matrix into a newly allocated matrix, with column-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_col_major(&self) -> Result<SparseColMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.transpose()
            .to_row_major()
            .map(SparseRowMat::into_transpose)
    }

    /// Returns a view over the transpose of `self` in column-major format.
    #[inline]
    pub fn transpose(self) -> SparseColMatRef<'a, I, E> {
        SparseColMatRef {
            inner: inner::SparseColMatRef {
                symbolic: SymbolicSparseColMatRef {
                    nrows: self.inner.symbolic.ncols,
                    ncols: self.inner.symbolic.nrows,
                    col_ptr: self.inner.symbolic.row_ptr,
                    col_nnz: self.inner.symbolic.row_nnz,
                    row_ind: self.inner.symbolic.col_ind,
                },
                values: self.inner.values,
            },
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(self) -> SparseRowMatRef<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        SparseRowMatRef {
            inner: inner::SparseRowMatRef {
                symbolic: self.inner.symbolic,
                values: unsafe {
                    SliceGroup::<'a, E::Conj>::new(transmute_unchecked::<
                        GroupFor<E, &[UnitFor<E::Conj>]>,
                        GroupFor<E::Conj, &[UnitFor<E::Conj>]>,
                    >(E::faer_map(
                        self.inner.values.into_inner(),
                        |slice| {
                            let len = slice.len();
                            core::slice::from_raw_parts(
                                slice.as_ptr() as *const UnitFor<E> as *const UnitFor<E::Conj>,
                                len,
                            )
                        },
                    )))
                },
            },
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn canonicalize(self) -> (SparseRowMatRef<'a, I, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        (
            SparseRowMatRef {
                inner: inner::SparseRowMatRef {
                    symbolic: self.inner.symbolic,
                    values: unsafe {
                        SliceGroup::<'a, E::Canonical>::new(transmute_unchecked::<
                            GroupFor<E, &[UnitFor<E::Canonical>]>,
                            GroupFor<E::Canonical, &[UnitFor<E::Canonical>]>,
                        >(E::faer_map(
                            self.inner.values.into_inner(),
                            |slice| {
                                let len = slice.len();
                                core::slice::from_raw_parts(
                                    slice.as_ptr() as *const UnitFor<E>
                                        as *const UnitFor<E::Canonical>,
                                    len,
                                )
                            },
                        )))
                    },
                },
            },
            if coe::is_same::<E, E::Canonical>() {
                Conj::No
            } else {
                Conj::Yes
            },
        )
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(self) -> SparseColMatRef<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        self.transpose().conjugate()
    }

    /// Returns the numerical values of row `i` of the matrix.
    ///
    /// # Panics:
    ///
    /// Panics if `i >= nrows`.
    #[inline]
    #[track_caller]
    pub fn values_of_row(self, i: usize) -> GroupFor<E, &'a [E::Unit]> {
        self.inner.values.subslice(self.row_range(i)).into_inner()
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'a, I> {
        self.inner.symbolic
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn into_parts(self) -> (SymbolicSparseRowMatRef<'a, I>, GroupFor<E, &'a [E::Unit]>) {
        (self.inner.symbolic, self.inner.values.into_inner())
    }
}

impl<'a, I: Index, E: Entity> SparseColMatRef<'a, I, E> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.row_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(
        symbolic: SymbolicSparseColMatRef<'a, I>,
        values: GroupFor<E, &'a [E::Unit]>,
    ) -> Self {
        let values = SliceGroup::new(values);
        assert!(symbolic.row_indices().len() == values.len());
        Self {
            inner: inner::SparseColMatRef { symbolic, values },
        }
    }

    /// Copies the current matrix into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SparseColMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        let symbolic = self.symbolic().to_owned()?;
        let mut values = VecGroup::<E::Canonical>::new();

        values
            .try_reserve_exact(self.inner.values.len())
            .map_err(|_| FaerError::OutOfMemory)?;

        values.resize(
            self.inner.values.len(),
            E::Canonical::faer_zero().faer_into_units(),
        );

        let src = self.inner.values;
        let dst = values.as_slice_mut();

        for (mut dst, src) in core::iter::zip(dst.into_mut_iter(), src.into_ref_iter()) {
            dst.write(src.read().canonicalize());
        }

        Ok(SparseColMat {
            inner: inner::SparseColMat { symbolic, values },
        })
    }

    /// Copies the current matrix into a newly allocated matrix, with row-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_row_major(&self) -> Result<SparseRowMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        let mut col_ptr = try_zeroed::<I>(self.nrows + 1)?;
        let nnz = self.compute_nnz();
        let mut row_ind = try_zeroed::<I>(nnz)?;
        let mut values = VecGroup::<E::Canonical>::new();
        values
            .try_reserve_exact(nnz)
            .map_err(|_| FaerError::OutOfMemory)?;

        let mut mem = GlobalPodBuffer::try_new(StackReq::new::<I>(self.nrows))
            .map_err(|_| FaerError::OutOfMemory)?;

        let (this, conj) = self.canonicalize();

        if conj == Conj::No {
            util::transpose::<I, E::Canonical>(
                &mut col_ptr,
                &mut row_ind,
                values.as_slice_mut().into_inner(),
                this,
                PodStack::new(&mut mem),
            );
        } else {
            util::adjoint::<I, E::Canonical>(
                &mut col_ptr,
                &mut row_ind,
                values.as_slice_mut().into_inner(),
                this,
                PodStack::new(&mut mem),
            );
        }

        let transpose = unsafe {
            SparseColMat::new(
                SymbolicSparseColMat::new_unchecked(self.ncols, self.nrows, col_ptr, None, row_ind),
                values.into_inner(),
            )
        };

        Ok(transpose.into_transpose())
    }

    /// Returns a view over the transpose of `self` in row-major format.
    #[inline]
    pub fn transpose(self) -> SparseRowMatRef<'a, I, E> {
        SparseRowMatRef {
            inner: inner::SparseRowMatRef {
                symbolic: SymbolicSparseRowMatRef {
                    nrows: self.inner.symbolic.ncols,
                    ncols: self.inner.symbolic.nrows,
                    row_ptr: self.inner.symbolic.col_ptr,
                    row_nnz: self.inner.symbolic.col_nnz,
                    col_ind: self.inner.symbolic.row_ind,
                },
                values: self.inner.values,
            },
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(self) -> SparseColMatRef<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        SparseColMatRef {
            inner: inner::SparseColMatRef {
                symbolic: self.inner.symbolic,
                values: unsafe {
                    SliceGroup::<'a, E::Conj>::new(transmute_unchecked::<
                        GroupFor<E, &[UnitFor<E::Conj>]>,
                        GroupFor<E::Conj, &[UnitFor<E::Conj>]>,
                    >(E::faer_map(
                        self.inner.values.into_inner(),
                        |slice| {
                            let len = slice.len();
                            core::slice::from_raw_parts(
                                slice.as_ptr() as *const UnitFor<E> as *const UnitFor<E::Conj>,
                                len,
                            )
                        },
                    )))
                },
            },
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn canonicalize(self) -> (SparseColMatRef<'a, I, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        (
            SparseColMatRef {
                inner: inner::SparseColMatRef {
                    symbolic: self.inner.symbolic,
                    values: unsafe {
                        SliceGroup::<'a, E::Canonical>::new(transmute_unchecked::<
                            GroupFor<E, &[UnitFor<E::Canonical>]>,
                            GroupFor<E::Canonical, &[UnitFor<E::Canonical>]>,
                        >(E::faer_map(
                            self.inner.values.into_inner(),
                            |slice| {
                                let len = slice.len();
                                core::slice::from_raw_parts(
                                    slice.as_ptr() as *const UnitFor<E>
                                        as *const UnitFor<E::Canonical>,
                                    len,
                                )
                            },
                        )))
                    },
                },
            },
            if coe::is_same::<E, E::Canonical>() {
                Conj::No
            } else {
                Conj::Yes
            },
        )
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(self) -> SparseRowMatRef<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        self.transpose().conjugate()
    }

    /// Returns the numerical values of the matrix.
    #[inline]
    pub fn values(self) -> GroupFor<E, &'a [E::Unit]> {
        self.inner.values.into_inner()
    }

    /// Returns the numerical values of column `j` of the matrix.
    ///
    /// # Panics:
    ///
    /// Panics if `j >= ncols`.
    #[inline]
    #[track_caller]
    pub fn values_of_col(self, j: usize) -> GroupFor<E, &'a [E::Unit]> {
        self.inner.values.subslice(self.col_range(j)).into_inner()
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseColMatRef<'a, I> {
        self.inner.symbolic
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn into_parts(self) -> (SymbolicSparseColMatRef<'a, I>, GroupFor<E, &'a [E::Unit]>) {
        (self.inner.symbolic, self.inner.values.into_inner())
    }
}

impl<I: Index, E: Entity> SparseColMat<I, E> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.row_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(symbolic: SymbolicSparseColMat<I>, values: GroupFor<E, Vec<E::Unit>>) -> Self {
        let values = VecGroup::from_inner(values);
        assert!(symbolic.row_indices().len() == values.len());
        Self {
            inner: inner::SparseColMat { symbolic, values },
        }
    }

    /// Copies the current matrix into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SparseColMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.as_ref().to_owned()
    }

    /// Copies the current matrix into a newly allocated matrix, with row-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_row_major(&self) -> Result<SparseRowMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.as_ref().to_row_major()
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn into_parts(self) -> (SymbolicSparseColMat<I>, GroupFor<E, Vec<E::Unit>>) {
        (self.inner.symbolic, self.inner.values.into_inner())
    }

    /// Returns a view over `self`.
    #[inline]
    pub fn as_ref(&self) -> SparseColMatRef<'_, I, E> {
        SparseColMatRef {
            inner: inner::SparseColMatRef {
                symbolic: self.inner.symbolic.as_ref(),
                values: self.inner.values.as_slice(),
            },
        }
    }

    /// Returns a mutable view over `self`.
    ///
    /// Note that the symbolic structure cannot be changed through this view.
    #[inline]
    pub fn as_mut(&mut self) -> SparseColMatMut<'_, I, E> {
        SparseColMatMut {
            inner: inner::SparseColMatMut {
                symbolic: self.inner.symbolic.as_ref(),
                values: self.inner.values.as_slice_mut(),
            },
        }
    }

    /// Returns a slice over the numerical values of the matrix.
    #[inline]
    pub fn values(&self) -> GroupFor<E, &'_ [E::Unit]> {
        self.inner.values.as_slice().into_inner()
    }

    /// Returns a mutable slice over the numerical values of the matrix.
    #[inline]
    pub fn values_mut(&mut self) -> GroupFor<E, &'_ mut [E::Unit]> {
        self.inner.values.as_slice_mut().into_inner()
    }

    /// Returns a view over the transpose of `self` in row-major format.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn into_transpose(self) -> SparseRowMat<I, E> {
        SparseRowMat {
            inner: inner::SparseRowMat {
                symbolic: SymbolicSparseRowMat {
                    nrows: self.inner.symbolic.ncols,
                    ncols: self.inner.symbolic.nrows,
                    row_ptr: self.inner.symbolic.col_ptr,
                    row_nnz: self.inner.symbolic.col_nnz,
                    col_ind: self.inner.symbolic.row_ind,
                },
                values: self.inner.values,
            },
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn into_conjugate(self) -> SparseColMat<I, E::Conj>
    where
        E: Conjugate,
    {
        SparseColMat {
            inner: inner::SparseColMat {
                symbolic: self.inner.symbolic,
                values: unsafe {
                    VecGroup::<E::Conj>::from_inner(transmute_unchecked::<
                        GroupFor<E, Vec<UnitFor<E::Conj>>>,
                        GroupFor<E::Conj, Vec<UnitFor<E::Conj>>>,
                    >(E::faer_map(
                        self.inner.values.into_inner(),
                        |mut slice| {
                            let len = slice.len();
                            let cap = slice.capacity();
                            let ptr =
                                slice.as_mut_ptr() as *mut UnitFor<E> as *mut UnitFor<E::Conj>;

                            Vec::from_raw_parts(ptr, len, cap)
                        },
                    )))
                },
            },
        }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn into_adjoint(self) -> SparseRowMat<I, E::Conj>
    where
        E: Conjugate,
    {
        self.into_transpose().into_conjugate()
    }
}

impl<I: Index, E: Entity> SparseRowMat<I, E> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.col_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(symbolic: SymbolicSparseRowMat<I>, values: GroupFor<E, Vec<E::Unit>>) -> Self {
        let values = VecGroup::from_inner(values);
        assert!(symbolic.col_indices().len() == values.len());
        Self {
            inner: inner::SparseRowMat { symbolic, values },
        }
    }

    /// Copies the current matrix into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SparseRowMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.as_ref().to_owned()
    }

    /// Copies the current matrix into a newly allocated matrix, with column-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_col_major(&self) -> Result<SparseColMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.as_ref().to_col_major()
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn into_parts(self) -> (SymbolicSparseRowMat<I>, GroupFor<E, Vec<E::Unit>>) {
        (self.inner.symbolic, self.inner.values.into_inner())
    }

    /// Returns a view over `self`.
    #[inline]
    pub fn as_ref(&self) -> SparseRowMatRef<'_, I, E> {
        SparseRowMatRef {
            inner: inner::SparseRowMatRef {
                symbolic: self.inner.symbolic.as_ref(),
                values: self.inner.values.as_slice(),
            },
        }
    }

    /// Returns a mutable view over `self`.
    ///
    /// Note that the symbolic structure cannot be changed through this view.
    #[inline]
    pub fn as_mut(&mut self) -> SparseRowMatMut<'_, I, E> {
        SparseRowMatMut {
            inner: inner::SparseRowMatMut {
                symbolic: self.inner.symbolic.as_ref(),
                values: self.inner.values.as_slice_mut(),
            },
        }
    }

    /// Returns a slice over the numerical values of the matrix.
    #[inline]
    pub fn values(&self) -> GroupFor<E, &'_ [E::Unit]> {
        self.inner.values.as_slice().into_inner()
    }

    /// Returns a mutable slice over the numerical values of the matrix.
    #[inline]
    pub fn values_mut(&mut self) -> GroupFor<E, &'_ mut [E::Unit]> {
        self.inner.values.as_slice_mut().into_inner()
    }

    /// Returns a view over the transpose of `self` in column-major format.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn into_transpose(self) -> SparseColMat<I, E> {
        SparseColMat {
            inner: inner::SparseColMat {
                symbolic: SymbolicSparseColMat {
                    nrows: self.inner.symbolic.ncols,
                    ncols: self.inner.symbolic.nrows,
                    col_ptr: self.inner.symbolic.row_ptr,
                    col_nnz: self.inner.symbolic.row_nnz,
                    row_ind: self.inner.symbolic.col_ind,
                },
                values: self.inner.values,
            },
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn into_conjugate(self) -> SparseRowMat<I, E::Conj>
    where
        E: Conjugate,
    {
        SparseRowMat {
            inner: inner::SparseRowMat {
                symbolic: self.inner.symbolic,
                values: unsafe {
                    VecGroup::<E::Conj>::from_inner(transmute_unchecked::<
                        GroupFor<E, Vec<UnitFor<E::Conj>>>,
                        GroupFor<E::Conj, Vec<UnitFor<E::Conj>>>,
                    >(E::faer_map(
                        self.inner.values.into_inner(),
                        |mut slice| {
                            let len = slice.len();
                            let cap = slice.capacity();
                            let ptr =
                                slice.as_mut_ptr() as *mut UnitFor<E> as *mut UnitFor<E::Conj>;

                            Vec::from_raw_parts(ptr, len, cap)
                        },
                    )))
                },
            },
        }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn into_adjoint(self) -> SparseColMat<I, E::Conj>
    where
        E: Conjugate,
    {
        self.into_transpose().into_conjugate()
    }
}

// DEREF/REBORROW
const _: () = {
    impl<'a, I: Index, E: Entity> core::ops::Deref for SparseRowMatMut<'a, I, E> {
        type Target = SymbolicSparseRowMatRef<'a, I>;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.inner.symbolic
        }
    }

    impl<'a, I: Index, E: Entity> core::ops::Deref for SparseColMatMut<'a, I, E> {
        type Target = SymbolicSparseColMatRef<'a, I>;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.inner.symbolic
        }
    }

    impl<'a, I: Index, E: Entity> core::ops::Deref for SparseRowMatRef<'a, I, E> {
        type Target = SymbolicSparseRowMatRef<'a, I>;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.inner.symbolic
        }
    }

    impl<'a, I: Index, E: Entity> core::ops::Deref for SparseColMatRef<'a, I, E> {
        type Target = SymbolicSparseColMatRef<'a, I>;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.inner.symbolic
        }
    }

    impl<I: Index, E: Entity> core::ops::Deref for SparseRowMat<I, E> {
        type Target = SymbolicSparseRowMat<I>;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.inner.symbolic
        }
    }

    impl<I: Index, E: Entity> core::ops::Deref for SparseColMat<I, E> {
        type Target = SymbolicSparseColMat<I>;
        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.inner.symbolic
        }
    }

    impl<'short, I: Index, E: Entity> ReborrowMut<'short> for SparseRowMatRef<'_, I, E> {
        type Target = SparseRowMatRef<'short, I, E>;

        #[inline]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }

    impl<'short, I: Index, E: Entity> Reborrow<'short> for SparseRowMatRef<'_, I, E> {
        type Target = SparseRowMatRef<'short, I, E>;

        #[inline]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }

    impl<'a, I: Index, E: Entity> IntoConst for SparseRowMatRef<'a, I, E> {
        type Target = SparseRowMatRef<'a, I, E>;

        #[inline]
        fn into_const(self) -> Self::Target {
            self
        }
    }

    impl<'short, I: Index, E: Entity> ReborrowMut<'short> for SparseColMatRef<'_, I, E> {
        type Target = SparseColMatRef<'short, I, E>;

        #[inline]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }

    impl<'short, I: Index, E: Entity> Reborrow<'short> for SparseColMatRef<'_, I, E> {
        type Target = SparseColMatRef<'short, I, E>;

        #[inline]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }

    impl<'a, I: Index, E: Entity> IntoConst for SparseColMatRef<'a, I, E> {
        type Target = SparseColMatRef<'a, I, E>;

        #[inline]
        fn into_const(self) -> Self::Target {
            self
        }
    }

    impl<'short, I: Index, E: Entity> ReborrowMut<'short> for SparseRowMatMut<'_, I, E> {
        type Target = SparseRowMatMut<'short, I, E>;

        #[inline]
        fn rb_mut(&'short mut self) -> Self::Target {
            SparseRowMatMut {
                inner: inner::SparseRowMatMut {
                    symbolic: self.inner.symbolic,
                    values: self.inner.values.rb_mut(),
                },
            }
        }
    }

    impl<'short, I: Index, E: Entity> Reborrow<'short> for SparseRowMatMut<'_, I, E> {
        type Target = SparseRowMatRef<'short, I, E>;

        #[inline]
        fn rb(&'short self) -> Self::Target {
            SparseRowMatRef {
                inner: inner::SparseRowMatRef {
                    symbolic: self.inner.symbolic,
                    values: self.inner.values.rb(),
                },
            }
        }
    }

    impl<'a, I: Index, E: Entity> IntoConst for SparseRowMatMut<'a, I, E> {
        type Target = SparseRowMatRef<'a, I, E>;

        #[inline]
        fn into_const(self) -> Self::Target {
            SparseRowMatRef {
                inner: inner::SparseRowMatRef {
                    symbolic: self.inner.symbolic,
                    values: self.inner.values.into_const(),
                },
            }
        }
    }

    impl<'short, I: Index, E: Entity> ReborrowMut<'short> for SparseColMatMut<'_, I, E> {
        type Target = SparseColMatMut<'short, I, E>;

        #[inline]
        fn rb_mut(&'short mut self) -> Self::Target {
            SparseColMatMut {
                inner: inner::SparseColMatMut {
                    symbolic: self.inner.symbolic,
                    values: self.inner.values.rb_mut(),
                },
            }
        }
    }

    impl<'short, I: Index, E: Entity> Reborrow<'short> for SparseColMatMut<'_, I, E> {
        type Target = SparseColMatRef<'short, I, E>;

        #[inline]
        fn rb(&'short self) -> Self::Target {
            SparseColMatRef {
                inner: inner::SparseColMatRef {
                    symbolic: self.inner.symbolic,
                    values: self.inner.values.rb(),
                },
            }
        }
    }

    impl<'a, I: Index, E: Entity> IntoConst for SparseColMatMut<'a, I, E> {
        type Target = SparseColMatRef<'a, I, E>;

        #[inline]
        fn into_const(self) -> Self::Target {
            SparseColMatRef {
                inner: inner::SparseColMatRef {
                    symbolic: self.inner.symbolic,
                    values: self.inner.values.into_const(),
                },
            }
        }
    }
};

/// Errors that can occur in sparse algorithms.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[non_exhaustive]
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

#[cfg(feature = "std")]
impl std::error::Error for CreationError {}

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
    v.try_reserve_exact(iter.size_hint().0)
        .map_err(|_| FaerError::OutOfMemory)?;
    v.extend(iter);
    Ok(v)
}

/// The order values should be read in, when constructing/filling from indices and values.
///
/// Allows separately creating the symbolic structure and filling the numerical values.
#[derive(Debug, Clone)]
pub struct ValuesOrder<I> {
    argsort: Vec<usize>,
    all_nnz: usize,
    nnz: usize,
    __marker: PhantomData<I>,
}

/// Whether the filled values should replace the current matrix values or be added to them.
#[derive(Debug, Copy, Clone)]
pub enum FillMode {
    /// New filled values should replace the old values.
    Replace,
    /// New filled values should be added to the old values.
    Add,
}

// FROM TRIPLETS
const _: () = {
    const TOP_BIT: usize = 1usize << (usize::BITS - 1);
    const TOP_BIT_MASK: usize = TOP_BIT - 1;

    impl<I: Index> SymbolicSparseColMat<I> {
        fn try_new_from_indices_impl(
            nrows: usize,
            ncols: usize,
            indices: impl Fn(usize) -> (I, I),
            all_nnz: usize,
        ) -> Result<(Self, ValuesOrder<I>), CreationError> {
            if nrows > I::Signed::MAX.zx() || ncols > I::Signed::MAX.zx() {
                return Err(CreationError::Generic(FaerError::IndexOverflow));
            }

            if all_nnz == 0 {
                return Ok((
                    Self {
                        nrows,
                        ncols,
                        col_ptr: try_zeroed(1)?,
                        col_nnz: None,
                        row_ind: Vec::new(),
                    },
                    ValuesOrder {
                        argsort: Vec::new(),
                        all_nnz,
                        nnz: 0,
                        __marker: PhantomData,
                    },
                ));
            }

            let mut argsort = try_collect(0..all_nnz)?;
            argsort.sort_unstable_by_key(|&i| {
                let (row, col) = indices(i);
                (col, row)
            });

            let mut n_duplicates = 0usize;
            let mut current_bit = 0usize;

            let mut prev = indices(argsort[0]);
            for i in 1..all_nnz {
                let idx = indices(argsort[i]);
                let same_as_prev = idx == prev;
                prev = idx;
                current_bit = ((current_bit == ((same_as_prev as usize) << (usize::BITS - 1)))
                    as usize)
                    << (usize::BITS - 1);
                argsort[i] |= current_bit;

                n_duplicates += same_as_prev as usize;
            }

            let nnz = all_nnz - n_duplicates;
            if nnz > I::Signed::MAX.zx() {
                return Err(CreationError::Generic(FaerError::IndexOverflow));
            }

            let mut col_ptr = try_zeroed::<I>(ncols + 1)?;
            let mut row_ind = try_zeroed::<I>(nnz)?;

            let mut original_pos = 0usize;
            let mut new_pos = 0usize;

            for j in 0..ncols {
                let mut n_unique = 0usize;

                while original_pos < all_nnz {
                    let (row, col) = indices(argsort[original_pos] & TOP_BIT_MASK);
                    if row.zx() >= nrows || col.zx() >= ncols {
                        return Err(CreationError::OutOfBounds {
                            row: row.zx(),
                            col: col.zx(),
                        });
                    }

                    if col.zx() != j {
                        break;
                    }

                    row_ind[new_pos] = row;

                    n_unique += 1;

                    new_pos += 1;
                    original_pos += 1;

                    while original_pos < all_nnz
                        && indices(argsort[original_pos] & TOP_BIT_MASK) == (row, col)
                    {
                        original_pos += 1;
                    }
                }

                col_ptr[j + 1] = col_ptr[j] + I::truncate(n_unique);
            }

            Ok((
                Self {
                    nrows,
                    ncols,
                    col_ptr,
                    col_nnz: None,
                    row_ind,
                },
                ValuesOrder {
                    argsort,
                    all_nnz,
                    nnz,
                    __marker: PhantomData,
                },
            ))
        }

        fn try_new_from_nonnegative_indices_impl(
            nrows: usize,
            ncols: usize,
            indices: impl Fn(usize) -> (I::Signed, I::Signed),
            all_nnz: usize,
        ) -> Result<(Self, ValuesOrder<I>), CreationError> {
            if nrows > I::Signed::MAX.zx() || ncols > I::Signed::MAX.zx() {
                return Err(CreationError::Generic(FaerError::IndexOverflow));
            }

            let mut argsort = try_collect(0..all_nnz)?;
            argsort.sort_unstable_by_key(|&i| {
                let (row, col) = indices(i);
                let ignore = (row < I::Signed::truncate(0)) | (col < I::Signed::truncate(0));
                (ignore, col, row)
            });

            let all_nnz = argsort.partition_point(|&i| {
                let (row, col) = indices(i);
                let ignore = (row < I::Signed::truncate(0)) | (col < I::Signed::truncate(0));
                !ignore
            });

            if all_nnz == 0 {
                return Ok((
                    Self {
                        nrows,
                        ncols,
                        col_ptr: try_zeroed(1)?,
                        col_nnz: None,
                        row_ind: Vec::new(),
                    },
                    ValuesOrder {
                        argsort: Vec::new(),
                        all_nnz,
                        nnz: 0,
                        __marker: PhantomData,
                    },
                ));
            }

            let mut n_duplicates = 0usize;
            let mut current_bit = 0usize;

            let mut prev = indices(argsort[0]);

            for i in 1..all_nnz {
                let idx = indices(argsort[i]);
                let same_as_prev = idx == prev;
                prev = idx;
                current_bit = ((current_bit == ((same_as_prev as usize) << (usize::BITS - 1)))
                    as usize)
                    << (usize::BITS - 1);
                argsort[i] |= current_bit;

                n_duplicates += same_as_prev as usize;
            }

            let nnz = all_nnz - n_duplicates;
            if nnz > I::Signed::MAX.zx() {
                return Err(CreationError::Generic(FaerError::IndexOverflow));
            }

            let mut col_ptr = try_zeroed::<I>(ncols + 1)?;
            let mut row_ind = try_zeroed::<I>(nnz)?;

            let mut original_pos = 0usize;
            let mut new_pos = 0usize;

            for j in 0..ncols {
                let mut n_unique = 0usize;

                while original_pos < all_nnz {
                    let (row, col) = indices(argsort[original_pos] & TOP_BIT_MASK);
                    if row.zx() >= nrows || col.zx() >= ncols {
                        return Err(CreationError::OutOfBounds {
                            row: row.zx(),
                            col: col.zx(),
                        });
                    }

                    if col.zx() != j {
                        break;
                    }

                    row_ind[new_pos] = I::from_signed(row);

                    n_unique += 1;

                    new_pos += 1;
                    original_pos += 1;

                    while original_pos < all_nnz
                        && indices(argsort[original_pos] & TOP_BIT_MASK) == (row, col)
                    {
                        original_pos += 1;
                    }
                }

                col_ptr[j + 1] = col_ptr[j] + I::truncate(n_unique);
            }

            Ok((
                Self {
                    nrows,
                    ncols,
                    col_ptr,
                    col_nnz: None,
                    row_ind,
                },
                ValuesOrder {
                    argsort,
                    all_nnz,
                    nnz,
                    __marker: PhantomData,
                },
            ))
        }

        /// Create a new symbolic structure, and the corresponding order for the numerical values
        /// from pairs of indices `(row, col)`.
        #[inline]
        pub fn try_new_from_indices(
            nrows: usize,
            ncols: usize,
            indices: &[(I, I)],
        ) -> Result<(Self, ValuesOrder<I>), CreationError> {
            Self::try_new_from_indices_impl(nrows, ncols, |i| indices[i], indices.len())
        }

        /// Create a new symbolic structure, and the corresponding order for the numerical values
        /// from pairs of indices `(row, col)`.
        ///
        /// Negative indices are ignored.
        #[inline]
        pub fn try_new_from_nonnegative_indices(
            nrows: usize,
            ncols: usize,
            indices: &[(I::Signed, I::Signed)],
        ) -> Result<(Self, ValuesOrder<I>), CreationError> {
            Self::try_new_from_nonnegative_indices_impl(nrows, ncols, |i| indices[i], indices.len())
        }
    }

    impl<I: Index> SymbolicSparseRowMat<I> {
        /// Create a new symbolic structure, and the corresponding order for the numerical values
        /// from pairs of indices `(row, col)`.
        #[inline]
        pub fn try_new_from_indices(
            nrows: usize,
            ncols: usize,
            indices: &[(I, I)],
        ) -> Result<(Self, ValuesOrder<I>), CreationError> {
            SymbolicSparseColMat::try_new_from_indices_impl(
                ncols,
                nrows,
                |i| {
                    let (row, col) = indices[i];
                    (col, row)
                },
                indices.len(),
            )
            .map(|(m, o)| (m.into_transpose(), o))
        }

        /// Create a new symbolic structure, and the corresponding order for the numerical values
        /// from pairs of indices `(row, col)`.
        ///
        /// Negative indices are ignored.
        #[inline]
        pub fn try_new_from_nonnegative_indices(
            nrows: usize,
            ncols: usize,
            indices: &[(I::Signed, I::Signed)],
        ) -> Result<(Self, ValuesOrder<I>), CreationError> {
            SymbolicSparseColMat::try_new_from_nonnegative_indices_impl(
                ncols,
                nrows,
                |i| {
                    let (row, col) = indices[i];
                    (col, row)
                },
                indices.len(),
            )
            .map(|(m, o)| (m.into_transpose(), o))
        }
    }

    impl<I: Index, E: ComplexField> SparseColMat<I, E> {
        #[track_caller]
        fn new_from_order_and_values_impl(
            symbolic: SymbolicSparseColMat<I>,
            order: &ValuesOrder<I>,
            all_values: impl Fn(usize) -> E,
            values_len: usize,
        ) -> Result<Self, FaerError> {
            {
                let nnz = order.argsort.len();
                assert!(values_len == nnz);
            }

            let all_nnz = order.all_nnz;

            let mut values = VecGroup::<E>::new();
            match values.try_reserve_exact(order.nnz) {
                Ok(()) => {}
                Err(_) => return Err(FaerError::OutOfMemory),
            };

            let mut pos = 0usize;
            let mut pos_unique = usize::MAX;
            let mut current_bit = TOP_BIT;

            while pos < all_nnz {
                let argsort_pos = order.argsort[pos];
                let extracted_bit = argsort_pos & TOP_BIT;
                let argsort_pos = argsort_pos & TOP_BIT_MASK;

                let val = all_values(argsort_pos);
                if extracted_bit != current_bit {
                    values.push(val.faer_into_units());
                    pos_unique = pos_unique.wrapping_add(1);
                } else {
                    let old_val = values.as_slice().read(pos_unique);
                    values
                        .as_slice_mut()
                        .write(pos_unique, old_val.faer_add(val));
                }

                current_bit = extracted_bit;

                pos += 1;
            }

            Ok(Self {
                inner: inner::SparseColMat { symbolic, values },
            })
        }

        /// Create a new matrix from a previously created symbolic structure and value order.
        /// The provided values must correspond to the same indices that were provided in the
        /// function call from which the order was created.
        #[track_caller]
        pub fn new_from_order_and_values(
            symbolic: SymbolicSparseColMat<I>,
            order: &ValuesOrder<I>,
            values: GroupFor<E, &[E::Unit]>,
        ) -> Result<Self, FaerError> {
            let values = SliceGroup::<'_, E>::new(values);
            Self::new_from_order_and_values_impl(symbolic, order, |i| values.read(i), values.len())
        }

        /// Create a new matrix from triplets `(row, col, value)`.
        #[track_caller]
        pub fn try_new_from_triplets(
            nrows: usize,
            ncols: usize,
            triplets: &[(I, I, E)],
        ) -> Result<Self, CreationError> {
            let (symbolic, order) = SymbolicSparseColMat::try_new_from_indices_impl(
                nrows,
                ncols,
                |i| {
                    let (row, col, _) = triplets[i];
                    (row, col)
                },
                triplets.len(),
            )?;
            Ok(Self::new_from_order_and_values_impl(
                symbolic,
                &order,
                |i| triplets[i].2,
                triplets.len(),
            )?)
        }

        /// Create a new matrix from triplets `(row, col, value)`. Negative indices are ignored.
        #[track_caller]
        pub fn try_new_from_nonnegative_triplets(
            nrows: usize,
            ncols: usize,
            triplets: &[(I::Signed, I::Signed, E)],
        ) -> Result<Self, CreationError> {
            let (symbolic, order) =
                SymbolicSparseColMat::<I>::try_new_from_nonnegative_indices_impl(
                    nrows,
                    ncols,
                    |i| {
                        let (row, col, _) = triplets[i];
                        (row, col)
                    },
                    triplets.len(),
                )?;
            Ok(Self::new_from_order_and_values_impl(
                symbolic,
                &order,
                |i| triplets[i].2,
                triplets.len(),
            )?)
        }
    }

    impl<I: Index, E: ComplexField> SparseRowMat<I, E> {
        /// Create a new matrix from a previously created symbolic structure and value order.
        /// The provided values must correspond to the same indices that were provided in the
        /// function call from which the order was created.
        #[track_caller]
        pub fn new_from_order_and_values(
            symbolic: SymbolicSparseRowMat<I>,
            order: &ValuesOrder<I>,
            values: GroupFor<E, &[E::Unit]>,
        ) -> Result<Self, FaerError> {
            SparseColMat::new_from_order_and_values(symbolic.into_transpose(), order, values)
                .map(SparseColMat::into_transpose)
        }

        /// Create a new matrix from triplets `(row, col, value)`.
        #[track_caller]
        pub fn try_new_from_triplets(
            nrows: usize,
            ncols: usize,
            triplets: &[(I, I, E)],
        ) -> Result<Self, CreationError> {
            let (symbolic, order) = SymbolicSparseColMat::try_new_from_indices_impl(
                ncols,
                nrows,
                |i| {
                    let (row, col, _) = triplets[i];
                    (col, row)
                },
                triplets.len(),
            )?;
            Ok(SparseColMat::new_from_order_and_values_impl(
                symbolic,
                &order,
                |i| triplets[i].2,
                triplets.len(),
            )?
            .into_transpose())
        }

        /// Create a new matrix from triplets `(row, col, value)`. Negative indices are ignored.
        #[track_caller]
        pub fn try_new_from_nonnegative_triplets(
            nrows: usize,
            ncols: usize,
            triplets: &[(I::Signed, I::Signed, E)],
        ) -> Result<Self, CreationError> {
            let (symbolic, order) =
                SymbolicSparseColMat::<I>::try_new_from_nonnegative_indices_impl(
                    ncols,
                    nrows,
                    |i| {
                        let (row, col, _) = triplets[i];
                        (col, row)
                    },
                    triplets.len(),
                )?;
            Ok(SparseColMat::new_from_order_and_values_impl(
                symbolic,
                &order,
                |i| triplets[i].2,
                triplets.len(),
            )?
            .into_transpose())
        }
    }

    impl<I: Index, E: ComplexField> SparseColMatMut<'_, I, E> {
        /// Fill the matrix from a previously created value order.
        /// The provided values must correspond to the same indices that were provided in the
        /// function call from which the order was created.
        ///
        /// # Note
        /// The symbolic structure is not changed.
        pub fn fill_from_order_and_values(
            &mut self,
            order: &ValuesOrder<I>,
            values: GroupFor<E, &[E::Unit]>,
            mode: FillMode,
        ) {
            let values = SliceGroup::<'_, E>::new(values);

            {
                let nnz = order.argsort.len();
                assert!(values.len() == nnz);
                assert!(order.nnz == self.inner.values.len());
            }
            let all_nnz = order.all_nnz;
            let mut dst = self.inner.values.rb_mut();

            let mut pos = 0usize;
            let mut pos_unique = usize::MAX;
            let mut current_bit = TOP_BIT;

            match mode {
                FillMode::Replace => {
                    while pos < all_nnz {
                        let argsort_pos = order.argsort[pos];
                        let extracted_bit = argsort_pos & TOP_BIT;
                        let argsort_pos = argsort_pos & TOP_BIT_MASK;

                        let val = values.read(argsort_pos);
                        if extracted_bit != current_bit {
                            pos_unique = pos_unique.wrapping_add(1);
                            dst.write(pos_unique, val);
                        } else {
                            let old_val = dst.read(pos_unique);
                            dst.write(pos_unique, old_val.faer_add(val));
                        }

                        current_bit = extracted_bit;

                        pos += 1;
                    }
                }
                FillMode::Add => {
                    while pos < all_nnz {
                        let argsort_pos = order.argsort[pos];
                        let extracted_bit = argsort_pos & TOP_BIT;
                        let argsort_pos = argsort_pos & TOP_BIT_MASK;

                        let val = values.read(argsort_pos);
                        if extracted_bit != current_bit {
                            pos_unique = pos_unique.wrapping_add(1);
                        }

                        let old_val = dst.read(pos_unique);
                        dst.write(pos_unique, old_val.faer_add(val));

                        current_bit = extracted_bit;

                        pos += 1;
                    }
                }
            }
        }
    }

    impl<I: Index, E: ComplexField> SparseRowMatMut<'_, I, E> {
        /// Fill the matrix from a previously created value order.
        /// The provided values must correspond to the same indices that were provided in the
        /// function call from which the order was created.
        ///
        /// # Note
        /// The symbolic structure is not changed.
        pub fn fill_from_order_and_values(
            &mut self,
            order: &ValuesOrder<I>,
            values: GroupFor<E, &[E::Unit]>,
            mode: FillMode,
        ) {
            self.rb_mut()
                .transpose_mut()
                .fill_from_order_and_values(order, values, mode);
        }
    }
};

/// Useful sparse matrix primitives.
pub mod util {
    use super::*;
    use crate::{assert, debug_assert};

    /// Sorts `row_indices` and `values` simultaneously so that `row_indices` is nonincreasing.
    pub fn sort_indices<I: Index, E: Entity>(
        col_ptrs: &[I],
        row_indices: &mut [I],
        values: GroupFor<E, &mut [E::Unit]>,
    ) {
        assert!(col_ptrs.len() >= 1);
        let mut values = SliceGroupMut::<'_, E>::new(values);

        let n = col_ptrs.len() - 1;
        for j in 0..n {
            let start = col_ptrs[j].zx();
            let end = col_ptrs[j + 1].zx();

            unsafe {
                crate::sort::sort_indices(
                    &mut row_indices[start..end],
                    values.rb_mut().subslice(start..end),
                );
            }
        }
    }

    #[doc(hidden)]
    pub unsafe fn ghost_permute_hermitian_unsorted<'n, 'out, I: Index, E: ComplexField>(
        new_values: SliceGroupMut<'out, E>,
        new_col_ptrs: &'out mut [I],
        new_row_indices: &'out mut [I],
        A: ghost::SparseColMatRef<'n, 'n, '_, I, E>,
        perm: ghost::PermutationRef<'n, '_, I, E>,
        in_side: Side,
        out_side: Side,
        sort: bool,
        stack: PodStack<'_>,
    ) -> ghost::SparseColMatMut<'n, 'n, 'out, I, E> {
        let N = A.ncols();
        let n = *A.ncols();

        // (1)
        assert!(new_col_ptrs.len() == n + 1);
        let (_, perm_inv) = perm.into_arrays();

        let (current_row_position, _) = stack.make_raw::<I>(n);
        let current_row_position = ghost::Array::from_mut(current_row_position, N);

        mem::fill_zero(current_row_position.as_mut());
        let col_counts = &mut *current_row_position;
        match (in_side, out_side) {
            (Side::Lower, Side::Lower) => {
                for old_j in N.indices() {
                    let new_j = perm_inv[old_j].zx();
                    for old_i in A.row_indices_of_col(old_j) {
                        if old_i >= old_j {
                            let new_i = perm_inv[old_i].zx();
                            let new_min = Ord::min(new_i, new_j);
                            // cannot overflow because A.compute_nnz() <= I::MAX
                            // col_counts[new_max] always >= 0
                            col_counts[new_min] += I::truncate(1);
                        }
                    }
                }
            }
            (Side::Lower, Side::Upper) => {
                for old_j in N.indices() {
                    let new_j = perm_inv[old_j].zx();
                    for old_i in A.row_indices_of_col(old_j) {
                        if old_i >= old_j {
                            let new_i = perm_inv[old_i].zx();
                            let new_max = Ord::max(new_i, new_j);
                            // cannot overflow because A.compute_nnz() <= I::MAX
                            // col_counts[new_max] always >= 0
                            col_counts[new_max] += I::truncate(1);
                        }
                    }
                }
            }
            (Side::Upper, Side::Lower) => {
                for old_j in N.indices() {
                    let new_j = perm_inv[old_j].zx();
                    for old_i in A.row_indices_of_col(old_j) {
                        if old_i <= old_j {
                            let new_i = perm_inv[old_i].zx();
                            let new_min = Ord::min(new_i, new_j);
                            // cannot overflow because A.compute_nnz() <= I::MAX
                            // col_counts[new_max] always >= 0
                            col_counts[new_min] += I::truncate(1);
                        }
                    }
                }
            }
            (Side::Upper, Side::Upper) => {
                for old_j in N.indices() {
                    let new_j = perm_inv[old_j].zx();
                    for old_i in A.row_indices_of_col(old_j) {
                        if old_i <= old_j {
                            let new_i = perm_inv[old_i].zx();
                            let new_max = Ord::max(new_i, new_j);
                            // cannot overflow because A.compute_nnz() <= I::MAX
                            // col_counts[new_max] always >= 0
                            col_counts[new_max] += I::truncate(1);
                        }
                    }
                }
            }
        }

        // col_counts[_] >= 0
        // cumulative sum cannot overflow because it is <= A.compute_nnz()

        // SAFETY: new_col_ptrs.len() == n + 1 > 0
        new_col_ptrs[0] = I::truncate(0);
        for (count, [ci0, ci1]) in zip(
            col_counts.as_mut(),
            windows2(Cell::as_slice_of_cells(Cell::from_mut(&mut *new_col_ptrs))),
        ) {
            let ci0 = ci0.get();
            ci1.set(ci0 + *count);
            *count = ci0;
        }
        // new_col_ptrs is non-decreasing

        let nnz = new_col_ptrs[n].zx();
        let new_row_indices = &mut new_row_indices[..nnz];
        let mut new_values = new_values.subslice(0..nnz);

        ghost::Size::with(
            nnz,
            #[inline(always)]
            |NNZ| {
                let mut new_values =
                    ghost::ArrayGroupMut::new(new_values.rb_mut().into_inner(), NNZ);
                let new_row_indices = ghost::Array::from_mut(new_row_indices, NNZ);

                let conj_if = |cond: bool, x: E| {
                    if !coe::is_same::<E, E::Real>() && cond {
                        x.faer_conj()
                    } else {
                        x
                    }
                };

                match (in_side, out_side) {
                    (Side::Lower, Side::Lower) => {
                        for old_j in N.indices() {
                            let new_j_ = perm_inv[old_j];
                            let new_j = new_j_.zx();

                            for (old_i, val) in zip(
                                A.row_indices_of_col(old_j),
                                SliceGroup::<'_, E>::new(A.values_of_col(old_j)).into_ref_iter(),
                            ) {
                                if old_i >= old_j {
                                    let new_i_ = perm_inv[old_i];
                                    let new_i = new_i_.zx();

                                    let new_max = Ord::max(new_i_, new_j_);
                                    let new_min = Ord::min(new_i, new_j);
                                    let current_row_pos: &mut I =
                                        &mut current_row_position[new_min];
                                    // SAFETY: current_row_pos < NNZ
                                    let row_pos = unsafe {
                                        ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ)
                                    };
                                    *current_row_pos += I::truncate(1);
                                    new_values
                                        .write(row_pos, conj_if(new_min == new_i, val.read()));
                                    // (2)
                                    new_row_indices[row_pos] = *new_max;
                                }
                            }
                        }
                    }
                    (Side::Lower, Side::Upper) => {
                        for old_j in N.indices() {
                            let new_j_ = perm_inv[old_j];
                            let new_j = new_j_.zx();

                            for (old_i, val) in zip(
                                A.row_indices_of_col(old_j),
                                SliceGroup::<'_, E>::new(A.values_of_col(old_j)).into_ref_iter(),
                            ) {
                                if old_i >= old_j {
                                    let new_i_ = perm_inv[old_i];
                                    let new_i = new_i_.zx();

                                    let new_max = Ord::max(new_i, new_j);
                                    let new_min = Ord::min(new_i_, new_j_);
                                    let current_row_pos = &mut current_row_position[new_max];
                                    // SAFETY: current_row_pos < NNZ
                                    let row_pos = unsafe {
                                        ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ)
                                    };
                                    *current_row_pos += I::truncate(1);
                                    new_values
                                        .write(row_pos, conj_if(new_max == new_i, val.read()));
                                    // (2)
                                    new_row_indices[row_pos] = *new_min;
                                }
                            }
                        }
                    }
                    (Side::Upper, Side::Lower) => {
                        for old_j in N.indices() {
                            let new_j_ = perm_inv[old_j];
                            let new_j = new_j_.zx();

                            for (old_i, val) in zip(
                                A.row_indices_of_col(old_j),
                                SliceGroup::<'_, E>::new(A.values_of_col(old_j)).into_ref_iter(),
                            ) {
                                if old_i <= old_j {
                                    let new_i_ = perm_inv[old_i];
                                    let new_i = new_i_.zx();

                                    let new_max = Ord::max(new_i_, new_j_);
                                    let new_min = Ord::min(new_i, new_j);
                                    let current_row_pos = &mut current_row_position[new_min];
                                    // SAFETY: current_row_pos < NNZ
                                    let row_pos = unsafe {
                                        ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ)
                                    };
                                    *current_row_pos += I::truncate(1);
                                    new_values
                                        .write(row_pos, conj_if(new_min == new_i, val.read()));
                                    // (2)
                                    new_row_indices[row_pos] = *new_max;
                                }
                            }
                        }
                    }
                    (Side::Upper, Side::Upper) => {
                        for old_j in N.indices() {
                            let new_j_ = perm_inv[old_j];
                            let new_j = new_j_.zx();

                            for (old_i, val) in zip(
                                A.row_indices_of_col(old_j),
                                SliceGroup::<'_, E>::new(A.values_of_col(old_j)).into_ref_iter(),
                            ) {
                                if old_i <= old_j {
                                    let new_i_ = perm_inv[old_i];
                                    let new_i = new_i_.zx();

                                    let new_max = Ord::max(new_i, new_j);
                                    let new_min = Ord::min(new_i_, new_j_);
                                    let current_row_pos = &mut current_row_position[new_max];
                                    // SAFETY: current_row_pos < NNZ
                                    let row_pos = unsafe {
                                        ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ)
                                    };
                                    *current_row_pos += I::truncate(1);
                                    new_values
                                        .write(row_pos, conj_if(new_max == new_i, val.read()));
                                    // (2)
                                    new_row_indices[row_pos] = *new_min;
                                }
                            }
                        }
                    }
                }
                debug_assert!(current_row_position.as_ref() == &new_col_ptrs[1..]);
            },
        );

        if sort {
            sort_indices::<I, E>(
                new_col_ptrs,
                new_row_indices,
                new_values.rb_mut().into_inner(),
            );
        }

        // SAFETY:
        // 0. new_col_ptrs is non-decreasing
        // 1. new_values.len() == new_row_indices.len()
        // 2. all written row indices are less than n
        unsafe {
            ghost::SparseColMatMut::new(
                SparseColMatMut::new(
                    SymbolicSparseColMatRef::new_unchecked(
                        n,
                        n,
                        new_col_ptrs,
                        None,
                        new_row_indices,
                    ),
                    new_values.into_inner(),
                ),
                N,
                N,
            )
        }
    }

    #[doc(hidden)]
    pub unsafe fn ghost_permute_hermitian_unsorted_symbolic<'n, 'out, I: Index>(
        new_col_ptrs: &'out mut [I],
        new_row_indices: &'out mut [I],
        A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
        perm: ghost::PermutationRef<'n, '_, I, Symbolic>,
        in_side: Side,
        out_side: Side,
        stack: PodStack<'_>,
    ) -> ghost::SymbolicSparseColMatRef<'n, 'n, 'out, I> {
        let old_values = &*Symbolic::materialize(A.into_inner().row_indices().len());
        let new_values = Symbolic::materialize(new_row_indices.len());
        *ghost_permute_hermitian_unsorted(
            SliceGroupMut::<'_, Symbolic>::new(new_values),
            new_col_ptrs,
            new_row_indices,
            ghost::SparseColMatRef::new(
                SparseColMatRef::new(A.into_inner(), old_values),
                A.nrows(),
                A.ncols(),
            ),
            perm,
            in_side,
            out_side,
            false,
            stack,
        )
    }

    /// Computes the self-adjoint permutation $P A P^\top$ of the matrix `A` without sorting the row
    /// indices, and returns a view over it.
    ///
    /// The result is stored in `new_col_ptrs`, `new_row_indices`.
    #[doc(hidden)]
    pub unsafe fn permute_hermitian_unsorted<'out, I: Index, E: ComplexField>(
        new_values: GroupFor<E, &'out mut [E::Unit]>,
        new_col_ptrs: &'out mut [I],
        new_row_indices: &'out mut [I],
        A: SparseColMatRef<'_, I, E>,
        perm: crate::permutation::PermutationRef<'_, I, E>,
        in_side: Side,
        out_side: Side,
        stack: PodStack<'_>,
    ) -> SparseColMatMut<'out, I, E> {
        ghost::Size::with(A.nrows(), |N| {
            assert!(A.nrows() == A.ncols());
            ghost_permute_hermitian_unsorted(
                SliceGroupMut::new(new_values),
                new_col_ptrs,
                new_row_indices,
                ghost::SparseColMatRef::new(A, N, N),
                ghost::PermutationRef::new(perm, N),
                in_side,
                out_side,
                false,
                stack,
            )
            .into_inner()
        })
    }

    /// Computes the self-adjoint permutation $P A P^\top$ of the matrix `A` and returns a view over
    /// it.
    ///
    /// The result is stored in `new_col_ptrs`, `new_row_indices`.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    pub fn permute_hermitian<'out, I: Index, E: ComplexField>(
        new_values: GroupFor<E, &'out mut [E::Unit]>,
        new_col_ptrs: &'out mut [I],
        new_row_indices: &'out mut [I],
        A: SparseColMatRef<'_, I, E>,
        perm: crate::permutation::PermutationRef<'_, I, E>,
        in_side: Side,
        out_side: Side,
        stack: PodStack<'_>,
    ) -> SparseColMatMut<'out, I, E> {
        ghost::Size::with(A.nrows(), |N| {
            assert!(A.nrows() == A.ncols());
            unsafe {
                ghost_permute_hermitian_unsorted(
                    SliceGroupMut::new(new_values),
                    new_col_ptrs,
                    new_row_indices,
                    ghost::SparseColMatRef::new(A, N, N),
                    ghost::PermutationRef::new(perm, N),
                    in_side,
                    out_side,
                    true,
                    stack,
                )
            }
            .into_inner()
        })
    }

    #[doc(hidden)]
    pub fn ghost_adjoint_symbolic<'m, 'n, 'a, I: Index>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        A: ghost::SymbolicSparseColMatRef<'m, 'n, '_, I>,
        stack: PodStack<'_>,
    ) -> ghost::SymbolicSparseColMatRef<'n, 'm, 'a, I> {
        let old_values = &*Symbolic::materialize(A.into_inner().row_indices().len());
        let new_values = Symbolic::materialize(new_row_indices.len());
        *ghost_adjoint(
            new_col_ptrs,
            new_row_indices,
            SliceGroupMut::<'_, Symbolic>::new(new_values),
            ghost::SparseColMatRef::new(
                SparseColMatRef::new(A.into_inner(), old_values),
                A.nrows(),
                A.ncols(),
            ),
            stack,
        )
    }

    #[doc(hidden)]
    pub fn ghost_adjoint<'m, 'n, 'a, I: Index, E: ComplexField>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        new_values: SliceGroupMut<'a, E>,
        A: ghost::SparseColMatRef<'m, 'n, '_, I, E>,
        stack: PodStack<'_>,
    ) -> ghost::SparseColMatMut<'n, 'm, 'a, I, E> {
        let M = A.nrows();
        let N = A.ncols();
        assert!(new_col_ptrs.len() == *M + 1);

        let (col_count, _) = stack.make_raw::<I>(*M);
        let col_count = ghost::Array::from_mut(col_count, M);
        mem::fill_zero(col_count.as_mut());

        // can't overflow because the total count is A.compute_nnz() <= I::MAX
        for j in N.indices() {
            for i in A.row_indices_of_col(j) {
                col_count[i] += I::truncate(1);
            }
        }

        new_col_ptrs[0] = I::truncate(0);
        // col_count elements are >= 0
        for (j, [pj0, pj1]) in zip(
            M.indices(),
            windows2(Cell::as_slice_of_cells(Cell::from_mut(new_col_ptrs))),
        ) {
            let cj = &mut col_count[j];
            let pj = pj0.get();
            // new_col_ptrs is non-decreasing
            pj1.set(pj + *cj);
            *cj = pj;
        }

        let new_row_indices = &mut new_row_indices[..new_col_ptrs[*M].zx()];
        let mut new_values = new_values.subslice(0..new_col_ptrs[*M].zx());
        let current_row_position = &mut *col_count;
        // current_row_position[i] == col_ptr[i]
        for j in N.indices() {
            let j_: ghost::Idx<'n, I> = j.truncate::<I>();
            for (i, val) in zip(
                A.row_indices_of_col(j),
                SliceGroup::<'_, E>::new(A.values_of_col(j)).into_ref_iter(),
            ) {
                let ci = &mut current_row_position[i];

                // SAFETY: see below
                unsafe {
                    *new_row_indices.get_unchecked_mut(ci.zx()) = *j_;
                    new_values.write_unchecked(ci.zx(), val.read().faer_conj())
                };
                *ci += I::truncate(1);
            }
        }
        // current_row_position[i] == col_ptr[i] + col_count[i] == col_ptr[i + 1] <= col_ptr[m]
        // so all the unchecked accesses were valid and non-overlapping, which means the entire
        // array is filled
        debug_assert!(current_row_position.as_ref() == &new_col_ptrs[1..]);

        // SAFETY:
        // 0. new_col_ptrs is non-decreasing
        // 1. all written row indices are less than n
        ghost::SparseColMatMut::new(
            unsafe {
                SparseColMatMut::new(
                    SymbolicSparseColMatRef::new_unchecked(
                        *N,
                        *M,
                        new_col_ptrs,
                        None,
                        new_row_indices,
                    ),
                    new_values.into_inner(),
                )
            },
            N,
            M,
        )
    }

    #[doc(hidden)]
    pub fn ghost_transpose<'m, 'n, 'a, I: Index, E: Entity>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        new_values: SliceGroupMut<'a, E>,
        A: ghost::SparseColMatRef<'m, 'n, '_, I, E>,
        stack: PodStack<'_>,
    ) -> ghost::SparseColMatMut<'n, 'm, 'a, I, E> {
        let M = A.nrows();
        let N = A.ncols();
        assert!(new_col_ptrs.len() == *M + 1);

        let (col_count, _) = stack.make_raw::<I>(*M);
        let col_count = ghost::Array::from_mut(col_count, M);
        mem::fill_zero(col_count.as_mut());

        // can't overflow because the total count is A.compute_nnz() <= I::MAX
        for j in N.indices() {
            for i in A.row_indices_of_col(j) {
                col_count[i] += I::truncate(1);
            }
        }

        new_col_ptrs[0] = I::truncate(0);
        // col_count elements are >= 0
        for (j, [pj0, pj1]) in zip(
            M.indices(),
            windows2(Cell::as_slice_of_cells(Cell::from_mut(new_col_ptrs))),
        ) {
            let cj = &mut col_count[j];
            let pj = pj0.get();
            // new_col_ptrs is non-decreasing
            pj1.set(pj + *cj);
            *cj = pj;
        }

        let new_row_indices = &mut new_row_indices[..new_col_ptrs[*M].zx()];
        let mut new_values = new_values.subslice(0..new_col_ptrs[*M].zx());
        let current_row_position = &mut *col_count;
        // current_row_position[i] == col_ptr[i]
        for j in N.indices() {
            let j_: ghost::Idx<'n, I> = j.truncate::<I>();
            for (i, val) in zip(
                A.row_indices_of_col(j),
                SliceGroup::<'_, E>::new(A.values_of_col(j)).into_ref_iter(),
            ) {
                let ci = &mut current_row_position[i];

                // SAFETY: see below
                unsafe {
                    *new_row_indices.get_unchecked_mut(ci.zx()) = *j_;
                    new_values.write_unchecked(ci.zx(), val.read())
                };
                *ci += I::truncate(1);
            }
        }
        // current_row_position[i] == col_ptr[i] + col_count[i] == col_ptr[i + 1] <= col_ptr[m]
        // so all the unchecked accesses were valid and non-overlapping, which means the entire
        // array is filled
        debug_assert!(current_row_position.as_ref() == &new_col_ptrs[1..]);

        // SAFETY:
        // 0. new_col_ptrs is non-decreasing
        // 1. all written row indices are less than n
        ghost::SparseColMatMut::new(
            unsafe {
                SparseColMatMut::new(
                    SymbolicSparseColMatRef::new_unchecked(
                        *N,
                        *M,
                        new_col_ptrs,
                        None,
                        new_row_indices,
                    ),
                    new_values.into_inner(),
                )
            },
            N,
            M,
        )
    }

    /// Computes the transpose of the matrix `A` and returns a view over it.
    ///
    /// The result is stored in `new_col_ptrs`, `new_row_indices` and `new_values`.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    pub fn transpose<'a, I: Index, E: Entity>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        new_values: GroupFor<E, &'a mut [E::Unit]>,
        A: SparseColMatRef<'_, I, E>,
        stack: PodStack<'_>,
    ) -> SparseColMatMut<'a, I, E> {
        ghost::Size::with2(A.nrows(), A.ncols(), |M, N| {
            ghost_transpose(
                new_col_ptrs,
                new_row_indices,
                SliceGroupMut::new(new_values),
                ghost::SparseColMatRef::new(A, M, N),
                stack,
            )
            .into_inner()
        })
    }

    /// Computes the adjoint of the matrix `A` and returns a view over it.
    ///
    /// The result is stored in `new_col_ptrs`, `new_row_indices` and `new_values`.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    pub fn adjoint<'a, I: Index, E: ComplexField>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        new_values: GroupFor<E, &'a mut [E::Unit]>,
        A: SparseColMatRef<'_, I, E>,
        stack: PodStack<'_>,
    ) -> SparseColMatMut<'a, I, E> {
        ghost::Size::with2(A.nrows(), A.ncols(), |M, N| {
            ghost_adjoint(
                new_col_ptrs,
                new_row_indices,
                SliceGroupMut::new(new_values),
                ghost::SparseColMatRef::new(A, M, N),
                stack,
            )
            .into_inner()
        })
    }

    /// Computes the adjoint of the symbolic matrix `A` and returns a view over it.
    ///
    /// The result is stored in `new_col_ptrs`, `new_row_indices`.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    pub fn adjoint_symbolic<'a, I: Index>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        A: SymbolicSparseColMatRef<'_, I>,
        stack: PodStack<'_>,
    ) -> SymbolicSparseColMatRef<'a, I> {
        ghost::Size::with2(A.nrows(), A.ncols(), |M, N| {
            ghost_adjoint_symbolic(
                new_col_ptrs,
                new_row_indices,
                ghost::SymbolicSparseColMatRef::new(A, M, N),
                stack,
            )
            .into_inner()
        })
    }
}

/// Arithmetic and generic binary/ternary operations.
pub mod ops {
    use super::*;
    use crate::assert;

    /// Returns the resulting matrix obtained by applying `f` to the elements from `lhs` and `rhs`,
    /// skipping entries that are unavailable in both of `lhs` and `rhs`.
    ///
    /// # Panics
    /// Panics if `lhs` and `rhs` don't have matching dimensions.  
    #[track_caller]
    pub fn binary_op<I: Index, E: Entity, LhsE: Entity, RhsE: Entity>(
        lhs: SparseColMatRef<'_, I, LhsE>,
        rhs: SparseColMatRef<'_, I, RhsE>,
        f: impl FnMut(LhsE, RhsE) -> E,
    ) -> Result<SparseColMat<I, E>, FaerError> {
        assert!(lhs.nrows() == rhs.nrows());
        assert!(lhs.ncols() == rhs.ncols());
        let mut f = f;
        let m = lhs.nrows();
        let n = lhs.ncols();

        let mut col_ptrs = try_zeroed::<I>(n + 1)?;

        let mut nnz = 0usize;
        for j in 0..n {
            let lhs = lhs.row_indices_of_col_raw(j);
            let rhs = rhs.row_indices_of_col_raw(j);

            let mut lhs_pos = 0usize;
            let mut rhs_pos = 0usize;
            while lhs_pos < lhs.len() && rhs_pos < rhs.len() {
                let lhs = lhs[lhs_pos];
                let rhs = rhs[rhs_pos];

                lhs_pos += (lhs <= rhs) as usize;
                rhs_pos += (rhs <= lhs) as usize;
                nnz += 1;
            }
            nnz += lhs.len() - lhs_pos;
            nnz += rhs.len() - rhs_pos;
            col_ptrs[j + 1] = I::truncate(nnz);
        }

        if nnz > I::Signed::MAX.zx() {
            return Err(FaerError::IndexOverflow);
        }

        let mut row_indices = try_zeroed(nnz)?;
        let mut values = VecGroup::<E>::new();
        values
            .try_reserve_exact(nnz)
            .map_err(|_| FaerError::OutOfMemory)?;
        values.resize(nnz, unsafe { core::mem::zeroed() });

        let mut nnz = 0usize;
        for j in 0..n {
            let mut values = values.as_slice_mut();
            let lhs_values = SliceGroup::<LhsE>::new(lhs.values_of_col(j));
            let rhs_values = SliceGroup::<RhsE>::new(rhs.values_of_col(j));
            let lhs = lhs.row_indices_of_col_raw(j);
            let rhs = rhs.row_indices_of_col_raw(j);

            let mut lhs_pos = 0usize;
            let mut rhs_pos = 0usize;
            while lhs_pos < lhs.len() && rhs_pos < rhs.len() {
                let lhs = lhs[lhs_pos];
                let rhs = rhs[rhs_pos];

                match lhs.cmp(&rhs) {
                    core::cmp::Ordering::Less => {
                        row_indices[nnz] = lhs;
                        values.write(
                            nnz,
                            f(lhs_values.read(lhs_pos), unsafe { core::mem::zeroed() }),
                        );
                    }
                    core::cmp::Ordering::Equal => {
                        row_indices[nnz] = lhs;
                        values.write(nnz, f(lhs_values.read(lhs_pos), rhs_values.read(rhs_pos)));
                    }
                    core::cmp::Ordering::Greater => {
                        row_indices[nnz] = rhs;
                        values.write(
                            nnz,
                            f(unsafe { core::mem::zeroed() }, rhs_values.read(rhs_pos)),
                        );
                    }
                }

                lhs_pos += (lhs <= rhs) as usize;
                rhs_pos += (rhs <= lhs) as usize;
                nnz += 1;
            }
            row_indices[nnz..nnz + lhs.len() - lhs_pos].copy_from_slice(&lhs[lhs_pos..]);
            for (mut dst, src) in values
                .rb_mut()
                .subslice(nnz..nnz + lhs.len() - lhs_pos)
                .into_mut_iter()
                .zip(lhs_values.subslice(lhs_pos..lhs.len()).into_ref_iter())
            {
                dst.write(f(src.read(), unsafe { core::mem::zeroed() }));
            }
            nnz += lhs.len() - lhs_pos;

            row_indices[nnz..nnz + rhs.len() - rhs_pos].copy_from_slice(&rhs[rhs_pos..]);
            for (mut dst, src) in values
                .rb_mut()
                .subslice(nnz..nnz + rhs.len() - rhs_pos)
                .into_mut_iter()
                .zip(rhs_values.subslice(rhs_pos..rhs.len()).into_ref_iter())
            {
                dst.write(f(unsafe { core::mem::zeroed() }, src.read()));
            }
            nnz += rhs.len() - rhs_pos;
        }

        Ok(SparseColMat::<I, E>::new(
            SymbolicSparseColMat::<I>::new_checked(m, n, col_ptrs, None, row_indices),
            values.into_inner(),
        ))
    }

    /// Returns the resulting matrix obtained by applying `f` to the elements from `dst` and `src`
    /// skipping entries that are unavailable in both of them.  
    /// The sparsity patter of `dst` is unchanged.
    ///
    /// # Panics
    /// Panics if `src` and `dst` don't have matching dimensions.  
    /// Panics if `src` contains an index that's unavailable in `dst`.  
    #[track_caller]
    pub fn binary_op_assign_into<I: Index, E: Entity, SrcE: Entity>(
        dst: SparseColMatMut<'_, I, E>,
        src: SparseColMatRef<'_, I, SrcE>,
        f: impl FnMut(E, SrcE) -> E,
    ) {
        {
            assert!(dst.nrows() == src.nrows());
            assert!(dst.ncols() == src.ncols());

            let n = dst.ncols();
            let mut dst = dst;
            let mut f = f;
            unsafe {
                assert!(f(core::mem::zeroed(), core::mem::zeroed()) == core::mem::zeroed());
            }

            for j in 0..n {
                let (dst, dst_val) = dst.rb_mut().into_parts_mut();

                let mut dst_val = SliceGroupMut::<E>::new(dst_val).subslice(dst.col_range(j));
                let src_val = SliceGroup::<SrcE>::new(src.values_of_col(j));

                let dst = dst.row_indices_of_col_raw(j);
                let src = src.row_indices_of_col_raw(j);

                let mut dst_pos = 0usize;
                let mut src_pos = 0usize;

                while src_pos < src.len() {
                    let src = src[src_pos];

                    if dst[dst_pos] < src {
                        dst_val.write(
                            dst_pos,
                            f(dst_val.read(dst_pos), unsafe { core::mem::zeroed() }),
                        );
                        dst_pos += 1;
                        continue;
                    }

                    assert!(dst[dst_pos] == src);

                    dst_val.write(dst_pos, f(dst_val.read(dst_pos), src_val.read(src_pos)));

                    src_pos += 1;
                    dst_pos += 1;
                }
                while dst_pos < dst.len() {
                    dst_val.write(
                        dst_pos,
                        f(dst_val.read(dst_pos), unsafe { core::mem::zeroed() }),
                    );
                    dst_pos += 1;
                }
            }
        }
    }

    /// Returns the resulting matrix obtained by applying `f` to the elements from `dst`, `lhs` and
    /// `rhs`, skipping entries that are unavailable in all of `dst`, `lhs` and `rhs`.  
    /// The sparsity patter of `dst` is unchanged.
    ///
    /// # Panics
    /// Panics if `lhs`, `rhs` and `dst` don't have matching dimensions.  
    /// Panics if `lhs` or `rhs` contains an index that's unavailable in `dst`.  
    #[track_caller]
    pub fn ternary_op_assign_into<I: Index, E: Entity, LhsE: Entity, RhsE: Entity>(
        dst: SparseColMatMut<'_, I, E>,
        lhs: SparseColMatRef<'_, I, LhsE>,
        rhs: SparseColMatRef<'_, I, RhsE>,
        f: impl FnMut(E, LhsE, RhsE) -> E,
    ) {
        {
            assert!(dst.nrows() == lhs.nrows());
            assert!(dst.ncols() == lhs.ncols());
            assert!(dst.nrows() == rhs.nrows());
            assert!(dst.ncols() == rhs.ncols());

            let n = dst.ncols();
            let mut dst = dst;
            let mut f = f;
            unsafe {
                assert!(
                    f(
                        core::mem::zeroed(),
                        core::mem::zeroed(),
                        core::mem::zeroed()
                    ) == core::mem::zeroed()
                );
            }

            for j in 0..n {
                let (dst, dst_val) = dst.rb_mut().into_parts_mut();

                let mut dst_val = SliceGroupMut::<E>::new(dst_val);
                let lhs_val = SliceGroup::<LhsE>::new(lhs.values_of_col(j));
                let rhs_val = SliceGroup::<RhsE>::new(rhs.values_of_col(j));

                let dst = dst.row_indices_of_col_raw(j);
                let rhs = rhs.row_indices_of_col_raw(j);
                let lhs = lhs.row_indices_of_col_raw(j);

                let mut dst_pos = 0usize;
                let mut lhs_pos = 0usize;
                let mut rhs_pos = 0usize;

                while lhs_pos < lhs.len() && rhs_pos < rhs.len() {
                    let lhs = lhs[lhs_pos];
                    let rhs = rhs[rhs_pos];

                    if dst[dst_pos] < Ord::min(lhs, rhs) {
                        dst_val.write(
                            dst_pos,
                            f(
                                dst_val.read(dst_pos),
                                unsafe { core::mem::zeroed() },
                                unsafe { core::mem::zeroed() },
                            ),
                        );
                        dst_pos += 1;
                        continue;
                    }

                    assert!(dst[dst_pos] == Ord::min(lhs, rhs));

                    match lhs.cmp(&rhs) {
                        core::cmp::Ordering::Less => {
                            dst_val.write(
                                dst_pos,
                                f(dst_val.read(dst_pos), lhs_val.read(lhs_pos), unsafe {
                                    core::mem::zeroed()
                                }),
                            );
                        }
                        core::cmp::Ordering::Equal => {
                            dst_val.write(
                                dst_pos,
                                f(
                                    dst_val.read(dst_pos),
                                    lhs_val.read(lhs_pos),
                                    rhs_val.read(rhs_pos),
                                ),
                            );
                        }
                        core::cmp::Ordering::Greater => {
                            dst_val.write(
                                dst_pos,
                                f(
                                    dst_val.read(dst_pos),
                                    unsafe { core::mem::zeroed() },
                                    rhs_val.read(rhs_pos),
                                ),
                            );
                        }
                    }

                    lhs_pos += (lhs <= rhs) as usize;
                    rhs_pos += (rhs <= lhs) as usize;
                    dst_pos += 1;
                }
                while lhs_pos < lhs.len() {
                    let lhs = lhs[lhs_pos];
                    if dst[dst_pos] < lhs {
                        dst_val.write(
                            dst_pos,
                            f(
                                dst_val.read(dst_pos),
                                unsafe { core::mem::zeroed() },
                                unsafe { core::mem::zeroed() },
                            ),
                        );
                        dst_pos += 1;
                        continue;
                    }
                    dst_val.write(
                        dst_pos,
                        f(dst_val.read(dst_pos), lhs_val.read(lhs_pos), unsafe {
                            core::mem::zeroed()
                        }),
                    );
                    lhs_pos += 1;
                    dst_pos += 1;
                }
                while rhs_pos < rhs.len() {
                    let rhs = rhs[rhs_pos];
                    if dst[dst_pos] < rhs {
                        dst_val.write(
                            dst_pos,
                            f(
                                dst_val.read(dst_pos),
                                unsafe { core::mem::zeroed() },
                                unsafe { core::mem::zeroed() },
                            ),
                        );
                        dst_pos += 1;
                        continue;
                    }
                    dst_val.write(
                        dst_pos,
                        f(
                            dst_val.read(dst_pos),
                            unsafe { core::mem::zeroed() },
                            rhs_val.read(rhs_pos),
                        ),
                    );
                    rhs_pos += 1;
                    dst_pos += 1;
                }
                while rhs_pos < rhs.len() {
                    let rhs = rhs[rhs_pos];
                    dst_pos += dst[dst_pos..].binary_search(&rhs).unwrap();
                    dst_val.write(
                        dst_pos,
                        f(
                            dst_val.read(dst_pos),
                            unsafe { core::mem::zeroed() },
                            rhs_val.read(rhs_pos),
                        ),
                    );
                    rhs_pos += 1;
                }
            }
        }
    }

    /// Returns the sparsity pattern containing the union of those of `lhs` and `rhs`.
    ///
    /// # Panics
    /// Panics if `lhs` and `rhs` don't have mathcing dimensions.  
    #[track_caller]
    #[inline]
    pub fn union_symbolic<I: Index>(
        lhs: SymbolicSparseColMatRef<'_, I>,
        rhs: SymbolicSparseColMatRef<'_, I>,
    ) -> Result<SymbolicSparseColMat<I>, FaerError> {
        Ok(binary_op(
            SparseColMatRef::<I, Symbolic>::new(lhs, Symbolic::materialize(lhs.compute_nnz())),
            SparseColMatRef::<I, Symbolic>::new(rhs, Symbolic::materialize(rhs.compute_nnz())),
            #[inline(always)]
            |_, _| Symbolic,
        )?
        .into_parts()
        .0)
    }

    /// Returns the sum of `lhs` and `rhs`.
    ///
    /// # Panics
    /// Panics if `lhs` and `rhs` don't have mathcing dimensions.  
    #[track_caller]
    #[inline]
    pub fn add<
        I: Index,
        E: ComplexField,
        LhsE: Conjugate<Canonical = E>,
        RhsE: Conjugate<Canonical = E>,
    >(
        lhs: SparseColMatRef<'_, I, LhsE>,
        rhs: SparseColMatRef<'_, I, RhsE>,
    ) -> Result<SparseColMat<I, E>, FaerError> {
        binary_op(lhs, rhs, |lhs, rhs| {
            lhs.canonicalize().faer_add(rhs.canonicalize())
        })
    }

    /// Returns the difference of `lhs` and `rhs`.
    ///
    /// # Panics
    /// Panics if `lhs` and `rhs` don't have matching dimensions.  
    #[track_caller]
    #[inline]
    pub fn sub<
        I: Index,
        LhsE: Conjugate<Canonical = E>,
        RhsE: Conjugate<Canonical = E>,
        E: ComplexField,
    >(
        lhs: SparseColMatRef<'_, I, LhsE>,
        rhs: SparseColMatRef<'_, I, RhsE>,
    ) -> Result<SparseColMat<I, E>, FaerError> {
        binary_op(lhs, rhs, |lhs, rhs| {
            lhs.canonicalize().faer_sub(rhs.canonicalize())
        })
    }

    /// Computes the sum of `dst` and `src` and stores the result in `dst` without changing its
    /// symbolic structure.
    ///
    /// # Panics
    /// Panics if `dst` and `rhs` don't have matching dimensions.  
    /// Panics if `rhs` contains an index that's unavailable in `dst`.  
    pub fn add_assign<I: Index, E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        dst: SparseColMatMut<'_, I, E>,
        rhs: SparseColMatRef<'_, I, RhsE>,
    ) {
        binary_op_assign_into(dst, rhs, |dst, rhs| dst.faer_add(rhs.canonicalize()))
    }

    /// Computes the difference of `dst` and `src` and stores the result in `dst` without changing
    /// its symbolic structure.
    ///
    /// # Panics
    /// Panics if `dst` and `rhs` don't have matching dimensions.  
    /// Panics if `rhs` contains an index that's unavailable in `dst`.  
    pub fn sub_assign<I: Index, E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        dst: SparseColMatMut<'_, I, E>,
        rhs: SparseColMatRef<'_, I, RhsE>,
    ) {
        binary_op_assign_into(dst, rhs, |dst, rhs| dst.faer_sub(rhs.canonicalize()))
    }

    /// Computes the sum of `lhs` and `rhs`, storing the result in `dst` without changing its
    /// symbolic structure.
    ///
    /// # Panics
    /// Panics if `dst`, `lhs` and `rhs` don't have matching dimensions.  
    /// Panics if `lhs` or `rhs` contains an index that's unavailable in `dst`.  
    #[track_caller]
    #[inline]
    pub fn add_into<
        I: Index,
        E: ComplexField,
        LhsE: Conjugate<Canonical = E>,
        RhsE: Conjugate<Canonical = E>,
    >(
        dst: SparseColMatMut<'_, I, E>,
        lhs: SparseColMatRef<'_, I, LhsE>,
        rhs: SparseColMatRef<'_, I, RhsE>,
    ) {
        ternary_op_assign_into(dst, lhs, rhs, |_, lhs, rhs| {
            lhs.canonicalize().faer_add(rhs.canonicalize())
        })
    }

    /// Computes the difference of `lhs` and `rhs`, storing the result in `dst` without changing its
    /// symbolic structure.
    ///
    /// # Panics
    /// Panics if `dst`, `lhs` and `rhs` don't have matching dimensions.  
    /// Panics if `lhs` or `rhs` contains an index that's unavailable in `dst`.  
    #[track_caller]
    #[inline]
    pub fn sub_into<
        I: Index,
        E: ComplexField,
        LhsE: Conjugate<Canonical = E>,
        RhsE: Conjugate<Canonical = E>,
    >(
        dst: SparseColMatMut<'_, I, E>,
        lhs: SparseColMatRef<'_, I, LhsE>,
        rhs: SparseColMatRef<'_, I, RhsE>,
    ) {
        ternary_op_assign_into(dst, lhs, rhs, |_, lhs, rhs| {
            lhs.canonicalize().faer_sub(rhs.canonicalize())
        })
    }
}

impl<'a, I: Index, E: Entity> SparseColMatRef<'a, I, E> {
    /// Returns a reference to the value at the given index, or None if the symbolic structure
    /// doesn't contain it
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get(self, row: usize, col: usize) -> Option<GroupFor<E, &'a E::Unit>> {
        assert!(row < self.nrows());
        assert!(col < self.ncols());

        let Ok(pos) = self
            .row_indices_of_col_raw(col)
            .binary_search(&I::truncate(row))
        else {
            return None;
        };

        Some(E::faer_map(self.values_of_col(col), |slice| &slice[pos]))
    }
}

impl<'a, I: Index, E: Entity> SparseColMatMut<'a, I, E> {
    /// Returns a reference to the value at the given index using a binary search, or None if the
    /// symbolic structure doesn't contain it
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get(self, row: usize, col: usize) -> Option<GroupFor<E, &'a E::Unit>> {
        self.into_const().get(row, col)
    }

    /// Returns a reference to the value at the given index using a binary search, or None if the
    /// symbolic structure doesn't contain it
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get_mut(self, row: usize, col: usize) -> Option<GroupFor<E, &'a mut E::Unit>> {
        assert!(row < self.nrows());
        assert!(col < self.ncols());

        let Ok(pos) = self
            .row_indices_of_col_raw(col)
            .binary_search(&I::truncate(row))
        else {
            return None;
        };

        Some(E::faer_map(self.values_of_col_mut(col), |slice| {
            &mut slice[pos]
        }))
    }
}

impl<I: Index, E: Entity> SparseColMat<I, E> {
    /// Returns a reference to the value at the given index using a binary search, or None if the
    /// symbolic structure doesn't contain it
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get(&self, row: usize, col: usize) -> Option<GroupFor<E, &'_ E::Unit>> {
        self.as_ref().get(row, col)
    }

    /// Returns a reference to the value at the given index using a binary search, or None if the
    /// symbolic structure doesn't contain it
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<GroupFor<E, &'_ mut E::Unit>> {
        self.as_mut().get_mut(row, col)
    }
}

impl<'a, I: Index, E: Entity> SparseRowMatRef<'a, I, E> {
    /// Returns a reference to the value at the given index using a binary search, or None if the
    /// symbolic structure doesn't contain it
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get(self, row: usize, col: usize) -> Option<GroupFor<E, &'a E::Unit>> {
        assert!(row < self.nrows());
        assert!(col < self.ncols());

        let Ok(pos) = self
            .col_indices_of_row_raw(row)
            .binary_search(&I::truncate(col))
        else {
            return None;
        };

        Some(E::faer_map(self.values_of_row(row), |slice| &slice[pos]))
    }
}

impl<'a, I: Index, E: Entity> SparseRowMatMut<'a, I, E> {
    /// Returns a reference to the value at the given index using a binary search, or None if the
    /// symbolic structure doesn't contain it
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get_mut(self, row: usize, col: usize) -> Option<GroupFor<E, &'a mut E::Unit>> {
        assert!(row < self.nrows());
        assert!(col < self.ncols());

        let Ok(pos) = self
            .col_indices_of_row_raw(row)
            .binary_search(&I::truncate(col))
        else {
            return None;
        };

        Some(E::faer_map(self.values_of_row_mut(row), |slice| {
            &mut slice[pos]
        }))
    }
}

impl<I: Index, E: Entity> SparseRowMat<I, E> {
    /// Returns a reference to the value at the given index using a binary search, or None if the
    /// symbolic structure doesn't contain it
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get(&self, row: usize, col: usize) -> Option<GroupFor<E, &'_ E::Unit>> {
        self.as_ref().get(row, col)
    }

    /// Returns a reference to the value at the given index using a binary search, or None if the
    /// symbolic structure doesn't contain it
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<GroupFor<E, &'_ mut E::Unit>> {
        self.as_mut().get_mut(row, col)
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseColMatRef<'_, I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseRowMatRef<'_, I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseColMatMut<'_, I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.rb().get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseRowMatMut<'_, I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.rb().get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for SparseColMatMut<'_, I, E> {
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.rb_mut().get_mut(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for SparseRowMatMut<'_, I, E> {
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.rb_mut().get_mut(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseColMat<I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.as_ref().get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseRowMat<I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.as_ref().get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for SparseColMat<I, E> {
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.as_mut().get_mut(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for SparseRowMat<I, E> {
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.as_mut().get_mut(row, col).unwrap()
    }
}

/// Sparse matrix multiplication.
pub mod mul {
    // TODO: sparse_sparse_matmul
    //
    // PERF: optimize matmul
    // - parallelization
    // - simd(?)

    use super::*;
    use crate::{
        assert,
        constrained::{self, Size},
    };

    /// Multiplies a sparse matrix `lhs` by a dense matrix `rhs`, and stores the result in
    /// `acc`. See [`crate::mul::matmul`] for more details.
    ///
    /// # Note
    /// Allows unsorted matrices.
    #[track_caller]
    pub fn sparse_dense_matmul<
        I: Index,
        E: ComplexField,
        LhsE: Conjugate<Canonical = E>,
        RhsE: Conjugate<Canonical = E>,
    >(
        acc: MatMut<'_, E>,
        lhs: SparseColMatRef<'_, I, LhsE>,
        rhs: MatRef<'_, RhsE>,
        alpha: Option<E>,
        beta: E,
        parallelism: Parallelism,
    ) {
        assert!(all(
            acc.nrows() == lhs.nrows(),
            acc.ncols() == rhs.ncols(),
            lhs.ncols() == rhs.nrows(),
        ));

        let _ = parallelism;
        let m = acc.nrows();
        let n = acc.ncols();
        let k = lhs.ncols();

        let mut acc = acc;

        match alpha {
            Some(alpha) => {
                if alpha != E::faer_one() {
                    zipped!(acc.rb_mut())
                        .for_each(|unzipped!(mut dst)| dst.write(dst.read().faer_mul(alpha)))
                }
            }
            None => acc.fill_zero(),
        }

        Size::with2(m, n, |m, n| {
            Size::with(k, |k| {
                let mut acc = constrained::MatMut::new(acc, m, n);
                let lhs = constrained::sparse::SparseColMatRef::new(lhs, m, k);
                let rhs = constrained::MatRef::new(rhs, k, n);

                for j in n.indices() {
                    for depth in k.indices() {
                        let rhs_kj = rhs.read(depth, j).canonicalize().faer_mul(beta);
                        for (i, lhs_ik) in zip(
                            lhs.row_indices_of_col(depth),
                            SliceGroup::<'_, LhsE>::new(lhs.values_of_col(depth)).into_ref_iter(),
                        ) {
                            acc.write(
                                i,
                                j,
                                acc.read(i, j)
                                    .faer_add(lhs_ik.read().canonicalize().faer_mul(rhs_kj)),
                            );
                        }
                    }
                }
            });
        });
    }

    /// Multiplies a dense matrix `lhs` by a sparse matrix `rhs`, and stores the result in
    /// `acc`. See [`crate::mul::matmul`] for more details.
    ///
    /// # Note
    /// Allows unsorted matrices.
    #[track_caller]
    pub fn dense_sparse_matmul<
        I: Index,
        E: ComplexField,
        LhsE: Conjugate<Canonical = E>,
        RhsE: Conjugate<Canonical = E>,
    >(
        acc: MatMut<'_, E>,
        lhs: MatRef<'_, LhsE>,
        rhs: SparseColMatRef<'_, I, RhsE>,
        alpha: Option<E>,
        beta: E,
        parallelism: Parallelism,
    ) {
        assert!(all(
            acc.nrows() == lhs.nrows(),
            acc.ncols() == rhs.ncols(),
            lhs.ncols() == rhs.nrows(),
        ));

        let _ = parallelism;
        let m = acc.nrows();
        let n = acc.ncols();
        let k = lhs.ncols();

        let mut acc = acc;

        match alpha {
            Some(alpha) => {
                if alpha != E::faer_one() {
                    zipped!(acc.rb_mut())
                        .for_each(|unzipped!(mut dst)| dst.write(dst.read().faer_mul(alpha)))
                }
            }
            None => acc.fill_zero(),
        }

        Size::with2(m, n, |m, n| {
            Size::with(k, |k| {
                let mut acc = constrained::MatMut::new(acc, m, n);
                let lhs = constrained::MatRef::new(lhs, m, k);
                let rhs = constrained::sparse::SparseColMatRef::new(rhs, k, n);

                for i in m.indices() {
                    for j in n.indices() {
                        let mut acc_ij = E::faer_zero();
                        for (depth, rhs_kj) in zip(
                            rhs.row_indices_of_col(j),
                            SliceGroup::<'_, RhsE>::new(rhs.values_of_col(j)).into_ref_iter(),
                        ) {
                            let lhs_ik = lhs.read(i, depth);
                            acc_ij = acc_ij.faer_add(
                                lhs_ik.canonicalize().faer_mul(rhs_kj.read().canonicalize()),
                            );
                        }

                        acc.write(i, j, acc.read(i, j).faer_add(beta.faer_mul(acc_ij)));
                    }
                }
            });
        });
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseColMatRef<'_, I, E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseColMatRef<'_, I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        let mut triplets = Vec::new();
        for j in 0..self.ncols() {
            for (i, val) in self
                .row_indices_of_col(j)
                .zip(SliceGroup::<'_, E>::new(self.values_of_col(j)).into_ref_iter())
            {
                triplets.push((i, j, val.read()))
            }
        }
        triplets
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseRowMatRef<'_, I, E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseRowMatRef<'_, I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        let mut triplets = Vec::new();
        for i in 0..self.nrows() {
            for (j, val) in self
                .col_indices_of_row(i)
                .zip(SliceGroup::<'_, E>::new(self.values_of_row(i)).into_ref_iter())
            {
                triplets.push((i, j, val.read()))
            }
        }
        triplets
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseColMatMut<'_, I, E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseColMatMut<'_, I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        self.rb().fetch_triplets()
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseColMat<I, E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseColMat<I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        self.as_ref().fetch_triplets()
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseRowMatMut<'_, I, E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseRowMatMut<'_, I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        self.rb().fetch_triplets()
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseRowMat<I, E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseRowMat<I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        self.as_ref().fetch_triplets()
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

        let indices = &[(0, 0), (1, 2), (0, 0), (1, 1), (0, 1), (3, 3), (3, 3usize)];
        let values = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0f64];

        let triplets = &[
            (0, 0, 1.0),
            (1, 2, 2.0),
            (0, 0, 3.0),
            (1, 1, 4.0),
            (0, 1, 5.0),
            (3, 3, 6.0),
            (3, 3usize, 7.0),
        ];

        {
            let mat = SymbolicSparseColMat::try_new_from_indices(nrows, ncols, indices);
            assert!(mat.is_ok());

            let (mat, order) = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.col_ptrs() == &[0, 1, 3, 4, 5]);
            assert!(mat.nnz_per_col() == None);
            assert!(mat.row_indices() == &[0, 0, 1, 1, 3]);

            let mat =
                SparseColMat::<_, f64>::new_from_order_and_values(mat, &order, values).unwrap();
            assert!(mat.as_ref().values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }

        {
            let mat = SparseColMat::try_new_from_triplets(nrows, ncols, triplets);
            assert!(mat.is_ok());
            let mat = mat.unwrap();

            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.col_ptrs() == &[0, 1, 3, 4, 5]);
            assert!(mat.nnz_per_col() == None);
            assert!(mat.row_indices() == &[0, 0, 1, 1, 3]);
            assert!(mat.values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }

        {
            let mat = SymbolicSparseRowMat::try_new_from_indices(nrows, ncols, indices);
            assert!(mat.is_ok());

            let (mat, order) = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.row_ptrs() == &[0, 2, 4, 4, 5, 5]);
            assert!(mat.nnz_per_row() == None);
            assert!(mat.col_indices() == &[0, 1, 1, 2, 3]);

            let mat =
                SparseRowMat::<_, f64>::new_from_order_and_values(mat, &order, values).unwrap();
            assert!(mat.values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }
        {
            let mat = SparseRowMat::try_new_from_triplets(nrows, ncols, triplets);
            assert!(mat.is_ok());

            let mat = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.row_ptrs() == &[0, 2, 4, 4, 5, 5]);
            assert!(mat.nnz_per_row() == None);
            assert!(mat.col_indices() == &[0, 1, 1, 2, 3]);
            assert!(mat.as_ref().values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }
    }

    #[test]
    fn test_from_nonnegative_indices() {
        let nrows = 5;
        let ncols = 4;

        let indices = &[
            (0, 0),
            (1, 2),
            (0, 0),
            (1, 1),
            (0, 1),
            (-1, 2),
            (-2, 1),
            (-3, -4),
            (3, 3),
            (3, 3isize),
        ];
        let values = &[
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            6.0,
            7.0f64,
        ];

        let triplets = &[
            (0, 0, 1.0),
            (1, 2, 2.0),
            (0, 0, 3.0),
            (1, 1, 4.0),
            (0, 1, 5.0),
            (-1, 2, f64::NAN),
            (-2, 1, f64::NAN),
            (-3, -4, f64::NAN),
            (3, 3, 6.0),
            (3, 3isize, 7.0),
        ];

        {
            let mat = SymbolicSparseColMat::<usize>::try_new_from_nonnegative_indices(
                nrows, ncols, indices,
            );
            assert!(mat.is_ok());

            let (mat, order) = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.col_ptrs() == &[0, 1, 3, 4, 5]);
            assert!(mat.nnz_per_col() == None);
            assert!(mat.row_indices() == &[0, 0, 1, 1, 3]);

            let mat =
                SparseColMat::<_, f64>::new_from_order_and_values(mat, &order, values).unwrap();
            assert!(mat.as_ref().values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }

        {
            let mat =
                SparseColMat::<usize, _>::try_new_from_nonnegative_triplets(nrows, ncols, triplets);
            assert!(mat.is_ok());
            let mat = mat.unwrap();

            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.col_ptrs() == &[0, 1, 3, 4, 5]);
            assert!(mat.nnz_per_col() == None);
            assert!(mat.row_indices() == &[0, 0, 1, 1, 3]);
            assert!(mat.values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }

        {
            let mat = SymbolicSparseRowMat::<usize>::try_new_from_nonnegative_indices(
                nrows, ncols, indices,
            );
            assert!(mat.is_ok());

            let (mat, order) = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.row_ptrs() == &[0, 2, 4, 4, 5, 5]);
            assert!(mat.nnz_per_row() == None);
            assert!(mat.col_indices() == &[0, 1, 1, 2, 3]);

            let mat =
                SparseRowMat::<_, f64>::new_from_order_and_values(mat, &order, values).unwrap();
            assert!(mat.values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }
        {
            let mat =
                SparseRowMat::<usize, _>::try_new_from_nonnegative_triplets(nrows, ncols, triplets);
            assert!(mat.is_ok());

            let mat = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.row_ptrs() == &[0, 2, 4, 4, 5, 5]);
            assert!(mat.nnz_per_row() == None);
            assert!(mat.col_indices() == &[0, 1, 1, 2, 3]);
            assert!(mat.as_ref().values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }
        {
            let order = SymbolicSparseRowMat::<usize>::try_new_from_nonnegative_indices(
                nrows, ncols, indices,
            )
            .unwrap()
            .1;

            let new_values = &mut [f64::NAN; 5];
            let mut mat = SparseRowMatMut::<'_, usize, f64>::new(
                SymbolicSparseRowMatRef::new_checked(
                    nrows,
                    ncols,
                    &[0, 2, 4, 4, 5, 5],
                    None,
                    &[0, 1, 1, 2, 3],
                ),
                new_values,
            );
            mat.fill_from_order_and_values(&order, values, FillMode::Replace);

            assert!(&*new_values == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }
    }

    #[test]
    fn test_from_indices_oob_row() {
        let nrows = 5;
        let ncols = 4;

        let indices = &[
            (0, 0),
            (1, 2),
            (0, 0),
            (1, 1),
            (0, 1),
            (3, 3),
            (3, 3),
            (5, 3usize),
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
            (0, 0),
            (1, 2),
            (0, 0),
            (1, 1),
            (0, 1),
            (3, 3),
            (3, 3),
            (2, 4usize),
        ];
        let err = SymbolicSparseColMat::try_new_from_indices(nrows, ncols, indices);
        assert!(err.is_err());
        let err = err.unwrap_err();
        assert!(err == CreationError::OutOfBounds { row: 2, col: 4 });
    }

    #[test]
    fn test_add_intersecting() {
        let lhs = SparseColMat::<usize, f64>::try_new_from_triplets(
            5,
            4,
            &[
                (1, 0, 1.0),
                (2, 1, 2.0),
                (3, 2, 3.0),
                (0, 0, 4.0),
                (1, 1, 5.0),
                (2, 2, 6.0),
                (3, 3, 7.0),
                (2, 0, 8.0),
                (3, 1, 9.0),
                (4, 2, 10.0),
                (0, 2, 11.0),
                (1, 3, 12.0),
                (4, 0, 13.0),
            ],
        )
        .unwrap();

        let rhs = SparseColMat::<usize, f64>::try_new_from_triplets(
            5,
            4,
            &[
                (1, 0, 10.0),
                (2, 1, 14.0),
                (3, 2, 15.0),
                (4, 3, 16.0),
                (0, 1, 17.0),
                (1, 2, 18.0),
                (2, 3, 19.0),
                (3, 0, 20.0),
                (4, 1, 21.0),
                (0, 3, 22.0),
            ],
        )
        .unwrap();

        let sum = ops::add(lhs.as_ref(), rhs.as_ref()).unwrap();
        assert!(sum.compute_nnz() == lhs.compute_nnz() + rhs.compute_nnz() - 3);

        for j in 0..4 {
            for i in 0..5 {
                assert!(sum.row_indices_of_col_raw(j)[i] == i);
            }
        }

        for j in 0..4 {
            for i in 0..5 {
                assert!(
                    sum[(i, j)] == lhs.get(i, j).unwrap_or(&0.0) + rhs.get(i, j).unwrap_or(&0.0)
                );
            }
        }
    }

    #[test]
    fn test_add_disjoint() {
        let lhs = SparseColMat::<usize, f64>::try_new_from_triplets(
            5,
            4,
            &[
                (0, 0, 1.0),
                (1, 1, 2.0),
                (2, 2, 3.0),
                (3, 3, 4.0),
                (2, 0, 5.0),
                (3, 1, 6.0),
                (4, 2, 7.0),
                (0, 2, 8.0),
                (1, 3, 9.0),
                (4, 0, 10.0),
            ],
        )
        .unwrap();

        let rhs = SparseColMat::<usize, f64>::try_new_from_triplets(
            5,
            4,
            &[
                (1, 0, 11.0),
                (2, 1, 12.0),
                (3, 2, 13.0),
                (4, 3, 14.0),
                (0, 1, 15.0),
                (1, 2, 16.0),
                (2, 3, 17.0),
                (3, 0, 18.0),
                (4, 1, 19.0),
                (0, 3, 20.0),
            ],
        )
        .unwrap();

        let sum = ops::add(lhs.as_ref(), rhs.as_ref()).unwrap();
        assert!(sum.compute_nnz() == lhs.compute_nnz() + rhs.compute_nnz());

        for j in 0..4 {
            for i in 0..5 {
                assert!(sum.row_indices_of_col_raw(j)[i] == i);
            }
        }

        for j in 0..4 {
            for i in 0..5 {
                assert!(
                    sum[(i, j)] == lhs.get(i, j).unwrap_or(&0.0) + rhs.get(i, j).unwrap_or(&0.0)
                );
            }
        }
    }
}
