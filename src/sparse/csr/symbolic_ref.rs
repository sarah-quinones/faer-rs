use super::*;
use crate::{assert, sparse::csc::*};

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
pub struct SymbolicSparseRowMatRef<'a, I: Index> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) row_ptr: &'a [I],
    pub(crate) row_nnz: Option<&'a [I]>,
    pub(crate) col_ind: &'a [I],
}

impl<I: Index> Copy for SymbolicSparseRowMatRef<'_, I> {}
impl<I: Index> Clone for SymbolicSparseRowMatRef<'_, I> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, I: Index> Reborrow<'short> for SymbolicSparseRowMatRef<'_, I> {
    type Target = SymbolicSparseRowMatRef<'short, I>;

    #[inline]
    fn rb(&self) -> Self::Target {
        *self
    }
}

impl<'short, I: Index> ReborrowMut<'short> for SymbolicSparseRowMatRef<'_, I> {
    type Target = SymbolicSparseRowMatRef<'short, I>;

    #[inline]
    fn rb_mut(&mut self) -> Self::Target {
        *self
    }
}

impl<'a, I: Index> IntoConst for SymbolicSparseRowMatRef<'a, I> {
    type Target = SymbolicSparseRowMatRef<'a, I>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
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
