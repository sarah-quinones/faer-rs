use super::*;
use crate::assert;

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
/// * Within each column, row indices are sorted in non-decreasing order.
///
/// # Note
/// Some algorithms allow working with matrices containing unsorted row indices per column.
///
/// Passing such a matrix to an algorithm that does not explicitly permit this is unspecified
/// (though not undefined) behavior.
pub struct SymbolicSparseColMatRef<'a, I: Index> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) col_ptr: &'a [I],
    pub(crate) col_nnz: Option<&'a [I]>,
    pub(crate) row_ind: &'a [I],
}

impl<I: Index> Copy for SymbolicSparseColMatRef<'_, I> {}
impl<I: Index> Clone for SymbolicSparseColMatRef<'_, I> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, I: Index> Reborrow<'short> for SymbolicSparseColMatRef<'_, I> {
    type Target = SymbolicSparseColMatRef<'short, I>;

    #[inline]
    fn rb(&self) -> Self::Target {
        *self
    }
}

impl<'short, I: Index> ReborrowMut<'short> for SymbolicSparseColMatRef<'_, I> {
    type Target = SymbolicSparseColMatRef<'short, I>;

    #[inline]
    fn rb_mut(&mut self) -> Self::Target {
        *self
    }
}

impl<'a, I: Index> IntoConst for SymbolicSparseColMatRef<'a, I> {
    type Target = SymbolicSparseColMatRef<'a, I>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
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
                        assert!(i_prev <= i);
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
                        assert!(i_prev <= i);
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

    /// Returns the number of rows and columns of the matrix.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
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

        let mut mem = GlobalPodBuffer::try_new(dyn_stack::StackReq::new::<I>(self.nrows))
            .map_err(|_| FaerError::OutOfMemory)?;

        utils::adjoint_symbolic(
            &mut col_ptr,
            &mut row_ind,
            *self,
            dyn_stack::PodStack::new(&mut mem),
        );

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
        let mut iter = (0..mat.ncols()).flat_map(move |j| {
            struct Wrapper(usize, usize);
            impl core::fmt::Debug for Wrapper {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    let row = self.0;
                    let col = self.1;
                    write!(f, "({row}, {col})")
                }
            }

            mat.row_indices_of_col(j).map(move |i| Wrapper(i, j))
        });

        f.debug_list().entries(&mut iter).finish()
    }
}
