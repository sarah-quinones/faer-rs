use super::*;
use crate::sparse::csr::*;
use core::marker::PhantomData;

/// Symbolic structure of sparse matrix in column format, either compressed or uncompressed.
///
/// Requires:
/// * `nrows <= I::Signed::MAX` (always checked)
/// * `ncols <= I::Signed::MAX` (always checked)
/// * `col_ptrs` has length `ncols + 1` (always checked)
/// * `col_ptrs` is increasing
/// * `col_ptrs[0]..col_ptrs[ncols]` is a valid range in row_indices (always checked, assuming
///   increasing)
/// * if `nnz_per_col` is `None`, elements of `row_indices[col_ptrs[j]..col_ptrs[j + 1]]` are less
///   than `nrows`
///
/// * `nnz_per_col[j] <= col_ptrs[j+1] - col_ptrs[j]`
/// * if `nnz_per_col` is `Some(_)`, elements of `row_indices[col_ptrs[j]..][..nnz_per_col[j]]` are
///   less than `nrows`
///
/// * Within each column, row indices are sorted in increasing order.
///
/// # Note
/// Some algorithms allow working with matrices containing unsorted row indices per column.
///
/// Passing such a matrix to an algorithm that does not explicitly permit this is unspecified
/// (though not undefined) behavior.
#[derive(Clone)]
pub struct SymbolicSparseColMat<I: Index, R: Shape = usize, C: Shape = usize> {
    pub(crate) nrows: R,
    pub(crate) ncols: C,
    pub(crate) col_ptr: alloc::vec::Vec<I>,
    pub(crate) col_nnz: Option<alloc::vec::Vec<I>>,
    pub(crate) row_ind: alloc::vec::Vec<I>,
}

impl<I: Index, R: Shape, C: Shape> SymbolicSparseColMat<I, R, C> {
    /// Creates a new symbolic matrix view after asserting its invariants.
    ///
    /// # Panics
    ///
    /// See type level documentation.
    #[inline]
    #[track_caller]
    pub fn new_checked(
        nrows: R,
        ncols: C,
        col_ptrs: alloc::vec::Vec<I>,
        nnz_per_col: Option<alloc::vec::Vec<I>>,
        row_indices: alloc::vec::Vec<I>,
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
        nrows: R,
        ncols: C,
        col_ptrs: alloc::vec::Vec<I>,
        nnz_per_col: Option<alloc::vec::Vec<I>>,
        row_indices: alloc::vec::Vec<I>,
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
        nrows: R,
        ncols: C,
        col_ptrs: alloc::vec::Vec<I>,
        nnz_per_col: Option<alloc::vec::Vec<I>>,
        row_indices: alloc::vec::Vec<I>,
    ) -> Self {
        SymbolicSparseColMatRef::new_unchecked(
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
    pub fn into_parts(
        self,
    ) -> (
        R,
        C,
        alloc::vec::Vec<I>,
        Option<alloc::vec::Vec<I>>,
        alloc::vec::Vec<I>,
    ) {
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
    pub fn as_ref(&self) -> SymbolicSparseColMatRef<'_, I, R, C> {
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
    pub fn nrows(&self) -> R {
        self.nrows
    }
    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> C {
        self.ncols
    }

    /// Returns the number of rows and columns of the matrix.
    #[inline]
    pub fn shape(&self) -> (R, C) {
        (self.nrows(), self.ncols())
    }

    /// Consumes the matrix, and returns its transpose in row-major format without reallocating.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn into_transpose(self) -> SymbolicSparseRowMat<I, C, R> {
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
    pub fn to_owned(&self) -> Result<SymbolicSparseColMat<I, R, C>, FaerError> {
        self.as_ref().to_owned()
    }

    /// Copies the current matrix into a newly allocated matrix, with row-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_row_major(&self) -> Result<SymbolicSparseRowMat<I, R, C>, FaerError> {
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
    pub fn row_indices_of_col_raw(&self, j: Idx<C>) -> &[Idx<R, I>] {
        self.as_ref().row_indices_of_col_raw(j)
    }

    /// Returns the row indices of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn row_indices_of_col_raw_unbound(&self, j: Idx<C>) -> &[I] {
        self.as_ref().row_indices_of_col_raw_unbound(j)
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
        j: Idx<C>,
    ) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = Idx<R>> {
        self.as_ref().row_indices_of_col(j)
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn col_range(&self, j: Idx<C>) -> Range<usize> {
        self.as_ref().col_range(j)
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub unsafe fn col_range_unchecked(&self, j: Idx<C>) -> Range<usize> {
        self.as_ref().col_range_unchecked(j)
    }

    pub(crate) fn try_new_from_indices_impl(
        nrows: R,
        ncols: C,
        indices: impl Fn(usize) -> (Idx<R, I>, Idx<C, I>),
        all_nnz: usize,
    ) -> Result<(Self, ValuesOrder<I>), CreationError> {
        if nrows.unbound() > I::Signed::MAX.zx() || ncols.unbound() > I::Signed::MAX.zx() {
            return Err(CreationError::Generic(FaerError::IndexOverflow));
        }

        if all_nnz == 0 {
            return Ok((
                Self {
                    nrows,
                    ncols,
                    col_ptr: try_zeroed(ncols.unbound() + 1)?,
                    col_nnz: None,
                    row_ind: alloc::vec::Vec::new(),
                },
                ValuesOrder {
                    argsort: alloc::vec::Vec::new(),
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

        let mut col_ptr = try_zeroed::<I>(ncols.unbound() + 1)?;
        let mut row_ind = try_zeroed::<I>(nnz)?;

        let mut original_pos = 0usize;
        let mut new_pos = 0usize;

        for j in C::indices(C::start(), ncols.end()) {
            let mut n_unique = 0usize;

            while original_pos < all_nnz {
                let (row, col) = indices(argsort[original_pos] & TOP_BIT_MASK);
                let row_x = unsafe { Idx::<R>::new_unbound(row.unbound().zx()) };
                let col_x = unsafe { Idx::<C>::new_unbound(col.unbound().zx()) };
                if row_x >= nrows || col_x >= ncols {
                    return Err(CreationError::OutOfBounds {
                        row: row.unbound().zx(),
                        col: col.unbound().zx(),
                    });
                }

                if col_x != j {
                    break;
                }

                row_ind[new_pos] = row.unbound();

                n_unique += 1;

                new_pos += 1;
                original_pos += 1;

                while original_pos < all_nnz
                    && indices(argsort[original_pos] & TOP_BIT_MASK) == (row, col)
                {
                    original_pos += 1;
                }
            }

            col_ptr[j.unbound() + 1] = col_ptr[j.unbound()] + I::truncate(n_unique);
        }

        let mut row_ind = core::mem::ManuallyDrop::new(row_ind);
        let length = row_ind.len();
        let capacity = row_ind.capacity();
        let ptr = row_ind.as_mut_ptr() as _;

        Ok((
            Self {
                nrows,
                ncols,
                col_ptr,
                col_nnz: None,
                row_ind: unsafe { alloc::vec::Vec::from_raw_parts(ptr, length, capacity) },
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
        nrows: R,
        ncols: C,
        indices: &[(Idx<R, I>, Idx<C, I>)],
    ) -> Result<(Self, ValuesOrder<I>), CreationError> {
        Self::try_new_from_indices_impl(nrows, ncols, |i| indices[i], indices.len())
    }
}

impl<I: Index> SymbolicSparseColMat<I> {
    pub(crate) fn try_new_from_nonnegative_indices_impl(
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
                    col_ptr: try_zeroed(ncols + 1)?,
                    col_nnz: None,
                    row_ind: alloc::vec::Vec::new(),
                },
                ValuesOrder {
                    argsort: alloc::vec::Vec::new(),
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
impl<I: Index> core::fmt::Debug for SymbolicSparseColMat<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}
