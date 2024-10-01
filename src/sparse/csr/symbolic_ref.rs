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
/// * Within each row, column indices are sorted in non-decreasing order.
///
/// # Note
/// Some algorithms allow working with matrices containing unsorted row indices per column.
///
/// Passing such a matrix to an algorithm that does not explicitly permit this is unspecified
/// (though not undefined) behavior.
pub struct SymbolicSparseRowMatRef<'a, I: Index, R: Shape = usize, C: Shape = usize> {
    pub(crate) nrows: R,
    pub(crate) ncols: C,
    pub(crate) row_ptr: &'a [I],
    pub(crate) row_nnz: Option<&'a [I]>,
    pub(crate) col_ind: &'a [Idx<C, I>],
}

impl<I: Index, R: Shape, C: Shape> Copy for SymbolicSparseRowMatRef<'_, I, R, C> {}
impl<I: Index, R: Shape, C: Shape> Clone for SymbolicSparseRowMatRef<'_, I, R, C> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, I: Index, R: Shape, C: Shape> Reborrow<'short>
    for SymbolicSparseRowMatRef<'_, I, R, C>
{
    type Target = SymbolicSparseRowMatRef<'short, I, R, C>;

    #[inline]
    fn rb(&self) -> Self::Target {
        *self
    }
}

impl<'short, I: Index, R: Shape, C: Shape> ReborrowMut<'short>
    for SymbolicSparseRowMatRef<'_, I, R, C>
{
    type Target = SymbolicSparseRowMatRef<'short, I, R, C>;

    #[inline]
    fn rb_mut(&mut self) -> Self::Target {
        *self
    }
}

impl<'a, I: Index, R: Shape, C: Shape> IntoConst for SymbolicSparseRowMatRef<'a, I, R, C> {
    type Target = SymbolicSparseRowMatRef<'a, I, R, C>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'a, I: Index, R: Shape, C: Shape> SymbolicSparseRowMatRef<'a, I, R, C> {
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
        row_ptrs: &'a [I],
        nnz_per_row: Option<&'a [I]>,
        col_indices: &'a [Idx<C, I>],
    ) -> Self {
        #[track_caller]
        fn check_ptrs<I: Index>(
            nrows: usize,
            ncols: usize,
            row_ptrs: &[I],
            nnz_per_row: Option<&[I]>,
        ) {
            assert!(all(
                ncols <= I::Signed::MAX.zx(),
                nrows <= I::Signed::MAX.zx(),
            ));
            assert!(row_ptrs.len() == nrows + 1);
            if let Some(nnz_per_row) = nnz_per_row {
                assert!(nnz_per_row.len() <= nrows);
                for (&nnz_i, &[c, c_next]) in zip(nnz_per_row, windows2(row_ptrs)) {
                    assert!(c <= c_next);
                    assert!(nnz_i <= c_next - c);
                }
            } else {
                for &[c, c_next] in windows2(row_ptrs) {
                    assert!(c <= c_next);
                }
            }
        }
        check_ptrs(nrows.unbound(), ncols.unbound(), row_ptrs, nnz_per_row);
        assert!(row_ptrs[nrows.unbound()].zx() <= col_indices.len());

        if const { !C::IS_BOUND } {
            if let Some(nnz_per_row) = nnz_per_row {
                for (&nnz_i, &c) in zip(nnz_per_row, row_ptrs) {
                    let col_indices = &col_indices[c.zx()..c.zx() + nnz_i.zx()];
                    if !col_indices.is_empty() {
                        let mut j_prev = col_indices[0];
                        for &j in &col_indices[1..] {
                            assert!(j_prev <= j);
                            j_prev = j;
                        }
                        let j_prev = j_prev.unbound();
                        let ncols = I::truncate(ncols.unbound());
                        assert!(j_prev < ncols);
                    }
                }
            } else {
                for &[c, c_next] in windows2(row_ptrs) {
                    let col_indices = &col_indices[c.zx()..c_next.zx()];
                    if !col_indices.is_empty() {
                        let mut j_prev = col_indices[0];
                        for &j in &col_indices[1..] {
                            assert!(j_prev <= j);
                            j_prev = j;
                        }
                        let j_prev = j_prev.unbound();
                        let ncols = I::truncate(ncols.unbound());
                        assert!(j_prev < ncols);
                    }
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
        nrows: R,
        ncols: C,
        row_ptrs: &'a [I],
        nnz_per_row: Option<&'a [I]>,
        col_indices: &'a [Idx<C, I>],
    ) -> Self {
        assert!(all(
            ncols.unbound() <= I::Signed::MAX.zx(),
            nrows.unbound() <= I::Signed::MAX.zx(),
        ));
        assert!(row_ptrs.len() == nrows.unbound() + 1);
        for &[c, c_next] in windows2(row_ptrs) {
            assert!(c <= c_next);
        }
        assert!(row_ptrs[nrows.unbound()].zx() <= col_indices.len());

        if let Some(nnz_per_row) = nnz_per_row {
            for (&nnz_i, &[c, c_next]) in zip(nnz_per_row, windows2(row_ptrs)) {
                assert!(nnz_i <= c_next - c);
                if const { !C::IS_BOUND } {
                    for &j in &col_indices[c.zx()..c.zx() + nnz_i.zx()] {
                        let j = j.unbound();
                        let ncols = ncols.unbound();
                        assert!(j < I::truncate(ncols));
                    }
                }
            }
        } else {
            if const { !C::IS_BOUND } {
                let c0 = row_ptrs[0].zx();
                let cn = row_ptrs[nrows.unbound()].zx();
                for &j in &col_indices[c0..cn] {
                    let j = j.unbound();
                    let ncols = ncols.unbound();
                    assert!(j < I::truncate(ncols));
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
        row_ptrs: &'a [I],
        nnz_per_row: Option<&'a [I]>,
        col_indices: &'a [Idx<C, I>],
    ) -> Self {
        assert!(all(
            ncols.unbound() <= <I::Signed as SignedIndex>::MAX.zx(),
            nrows.unbound() <= <I::Signed as SignedIndex>::MAX.zx(),
        ));
        assert!(row_ptrs.len() == nrows.unbound() + 1);
        assert!(row_ptrs[nrows.unbound()].zx() <= col_indices.len());

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

    /// Returns a view over the transpose of `self` in column-major format.
    #[inline]
    pub fn transpose(self) -> SymbolicSparseColMatRef<'a, I, C, R> {
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
    pub fn to_owned(&self) -> Result<SymbolicSparseRowMat<I, R, C>, FaerError> {
        self.transpose()
            .to_owned()
            .map(SymbolicSparseColMat::into_transpose)
    }

    /// Copies the current matrix into a newly allocated matrix, with column-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_col_major(&self) -> Result<SymbolicSparseColMat<I, R, C>, FaerError> {
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
    pub fn col_indices(&self) -> &'a [Idx<C, I>] {
        self.col_ind
    }

    /// Returns the column indices of row i.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub fn col_indices_of_row_raw(&self, i: Idx<R>) -> &'a [Idx<C, I>] {
        unsafe { __get_unchecked(self.col_ind, self.row_range(i)) }
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
        i: Idx<R>,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = Idx<C>> {
        self.col_indices_of_row_raw(i).iter().map(
            #[inline(always)]
            |&i| unsafe { Idx::<C>::new_unbound(i.unbound().zx()) },
        )
    }

    /// Returns the range that the row `i` occupies in `self.col_indices()`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub fn row_range(&self, i: Idx<R>) -> Range<usize> {
        assert!(i < self.nrows());

        unsafe { self.row_range_unchecked(i) }
    }

    /// Returns the range that the row `i` occupies in `self.col_indices()`.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub unsafe fn row_range_unchecked(&self, i: Idx<R>) -> Range<usize> {
        let i = i.unbound();
        let start = __get_unchecked(self.row_ptr, i).zx();
        let end = self
            .row_nnz
            .map(|row_nnz| (__get_unchecked(row_nnz, i).zx() + start))
            .unwrap_or(__get_unchecked(self.row_ptr, i + 1).zx());

        start..end
    }

    /// Returns the input matrix with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape<V: Shape, H: Shape>(
        self,
        nrows: V,
        ncols: H,
    ) -> SymbolicSparseRowMatRef<'a, I, V, H> {
        assert!(all(
            nrows.unbound() == self.nrows().unbound(),
            ncols.unbound() == self.ncols().unbound(),
        ));
        unsafe {
            SymbolicSparseRowMatRef::new_unchecked(
                nrows,
                ncols,
                self.row_ptr,
                self.row_nnz,
                core::slice::from_raw_parts(self.col_ind.as_ptr() as _, self.col_ind.len()),
            )
        }
    }

    /// Returns the input matrix with dynamic shape.
    #[inline]
    pub fn as_dyn(self) -> SymbolicSparseRowMatRef<'a, I> {
        unsafe {
            SymbolicSparseRowMatRef::new_unchecked(
                self.nrows.unbound(),
                self.ncols.unbound(),
                self.row_ptr,
                self.row_nnz,
                core::slice::from_raw_parts(self.col_ind.as_ptr() as _, self.col_ind.len()),
            )
        }
    }
}

impl<I: Index, R: Shape, C: Shape> core::fmt::Debug for SymbolicSparseRowMatRef<'_, I, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use crate::utils::bound::*;
        use generativity::make_guard;

        fn imp<I: Index>(
            mat: SymbolicSparseRowMatRef<'_, I, Dim<'_>, Dim<'_>>,
            f: &mut core::fmt::Formatter<'_>,
        ) -> core::fmt::Result {
            let mut iter = mat.nrows().indices().flat_map(move |i| {
                struct Wrapper(usize, usize);
                impl core::fmt::Debug for Wrapper {
                    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                        let row = self.0;
                        let col = self.1;
                        write!(f, "({row}, {col})")
                    }
                }

                mat.col_indices_of_row(i)
                    .map(move |j| Wrapper(i.unbound(), j.unbound()))
            });

            f.debug_list().entries(&mut iter).finish()
        }

        make_guard!(M);
        make_guard!(N);

        let M = self.nrows().bind(M);
        let N = self.ncols().bind(N);
        imp(self.as_shape(M, N), f)
    }
}
