use super::*;
use crate::{assert, debug_assert};

/// Symbolic view structure of sparse matrix in column format, either compressed or uncompressed.
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
pub struct SymbolicSparseColMatRef<'a, I: Index, R: Shape = usize, C: Shape = usize> {
    pub(crate) nrows: R,
    pub(crate) ncols: C,
    pub(crate) col_ptr: &'a [I],
    pub(crate) col_nnz: Option<&'a [I]>,
    pub(crate) row_ind: &'a [I],
}

impl<I: Index, R: Shape, C: Shape> Copy for SymbolicSparseColMatRef<'_, I, R, C> {}
impl<I: Index, R: Shape, C: Shape> Clone for SymbolicSparseColMatRef<'_, I, R, C> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, I: Index, R: Shape, C: Shape> Reborrow<'short>
    for SymbolicSparseColMatRef<'_, I, R, C>
{
    type Target = SymbolicSparseColMatRef<'short, I, R, C>;

    #[inline]
    fn rb(&self) -> Self::Target {
        *self
    }
}

impl<'short, I: Index, R: Shape, C: Shape> ReborrowMut<'short>
    for SymbolicSparseColMatRef<'_, I, R, C>
{
    type Target = SymbolicSparseColMatRef<'short, I, R, C>;

    #[inline]
    fn rb_mut(&mut self) -> Self::Target {
        *self
    }
}

impl<'a, I: Index, R: Shape, C: Shape> IntoConst for SymbolicSparseColMatRef<'a, I, R, C> {
    type Target = SymbolicSparseColMatRef<'a, I, R, C>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

#[inline(always)]
#[track_caller]
fn assume_col_ptrs<I: Index>(
    nrows: usize,
    ncols: usize,
    col_ptrs: &[I],
    nnz_per_col: Option<&[I]>,
    row_indices: &[I],
) {
    assert!(all(
        ncols <= I::Signed::MAX.zx(),
        nrows <= I::Signed::MAX.zx(),
    ));
    assert!(col_ptrs.len() == ncols + 1);
    assert!(col_ptrs[ncols].zx() <= row_indices.len());
    if let Some(nnz_per_row) = nnz_per_col {
        assert!(nnz_per_row.len() == ncols);
    }
}

#[track_caller]
fn check_col_ptrs<I: Index>(
    nrows: usize,
    ncols: usize,
    col_ptrs: &[I],
    nnz_per_col: Option<&[I]>,
    row_indices: &[I],
) {
    assert!(all(
        ncols <= I::Signed::MAX.zx(),
        nrows <= I::Signed::MAX.zx(),
    ));
    assert!(col_ptrs.len() == ncols + 1);
    if let Some(nnz_per_col) = nnz_per_col {
        assert!(nnz_per_col.len() == ncols);
        for (&nnz_j, &[col, col_next]) in zip(nnz_per_col, windows2(col_ptrs)) {
            assert!(col <= col_next);
            assert!(nnz_j <= col_next - col);
        }
    } else {
        for &[col, col_next] in windows2(col_ptrs) {
            assert!(col <= col_next);
        }
    }
    assert!(col_ptrs[ncols].zx() <= row_indices.len());
}

#[track_caller]
fn check_row_indices<I: Index>(
    nrows: usize,
    ncols: usize,
    col_ptrs: &[I],
    nnz_per_col: Option<&[I]>,
    row_indices: &[I],
) {
    _ = ncols;
    if let Some(nnz_per_col) = nnz_per_col {
        for (&nnz_j, &c) in zip(nnz_per_col, col_ptrs) {
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
}

#[track_caller]
fn check_row_indices_unsorted<I: Index>(
    nrows: usize,
    ncols: usize,
    col_ptrs: &[I],
    nnz_per_col: Option<&[I]>,
    row_indices: &[I],
) {
    _ = ncols;
    if let Some(nnz_per_col) = nnz_per_col {
        for (&nnz_i, &c) in zip(nnz_per_col, col_ptrs) {
            for &j in &row_indices[c.zx()..c.zx() + nnz_i.zx()] {
                assert!(j < I::truncate(nrows));
            }
        }
    } else {
        let c0 = col_ptrs[0].zx();
        let cn = col_ptrs[nrows].zx();
        for &j in &row_indices[c0..cn] {
            assert!(j < I::truncate(nrows));
        }
    }
}

impl<'a, I: Index, R: Shape, C: Shape> SymbolicSparseColMatRef<'a, I, R, C> {
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
        col_ptrs: &'a [I],
        nnz_per_col: Option<&'a [I]>,
        row_indices: &'a [I],
    ) -> Self {
        check_col_ptrs(
            nrows.unbound(),
            ncols.unbound(),
            col_ptrs,
            nnz_per_col,
            row_indices,
        );
        check_row_indices(
            nrows.unbound(),
            ncols.unbound(),
            col_ptrs,
            nnz_per_col,
            row_indices,
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
        col_ptrs: &'a [I],
        nnz_per_col: Option<&'a [I]>,
        row_indices: &'a [I],
    ) -> Self {
        check_col_ptrs(
            nrows.unbound(),
            ncols.unbound(),
            col_ptrs,
            nnz_per_col,
            row_indices,
        );
        check_row_indices_unsorted(
            nrows.unbound(),
            ncols.unbound(),
            col_ptrs,
            nnz_per_col,
            row_indices,
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
        col_ptrs: &'a [I],
        nnz_per_col: Option<&'a [I]>,
        row_indices: &'a [I],
    ) -> Self {
        assume_col_ptrs(
            nrows.unbound(),
            ncols.unbound(),
            col_ptrs,
            nnz_per_col,
            row_indices,
        );

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

    /// Returns a view over the transpose of `self` in row-major format.
    #[inline]
    pub fn transpose(self) -> SymbolicSparseRowMatRef<'a, I, C, R> {
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
    pub fn to_owned(&self) -> Result<SymbolicSparseColMat<I, R, C>, FaerError> {
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
    pub fn to_row_major(&self) -> Result<SymbolicSparseRowMat<I, R, C>, FaerError> {
        let mut col_ptr = try_zeroed::<I>(self.nrows.unbound() + 1)?;
        let mut row_ind = try_zeroed::<I>(self.compute_nnz())?;

        let mut mem = GlobalPodBuffer::try_new(dyn_stack::StackReq::new::<I>(self.nrows.unbound()))
            .map_err(|_| FaerError::OutOfMemory)?;

        utils::adjoint_symbolic(
            &mut col_ptr,
            &mut row_ind,
            *self,
            dyn_stack::PodStack::new(&mut mem),
        );

        let mut row_ind = core::mem::ManuallyDrop::new(row_ind);
        let length = row_ind.len();
        let capacity = row_ind.capacity();
        let ptr = row_ind.as_mut_ptr();
        let row_ind = unsafe { alloc::vec::Vec::from_raw_parts(ptr as _, length, capacity) };
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
        fn imp<I: Index>(col_nnz: Option<&[I]>, col_ptr: &[I], ncols: usize) -> usize {
            match col_nnz {
                Some(col_nnz) => {
                    let mut nnz = 0usize;
                    for &nnz_j in col_nnz {
                        // can't overflow
                        nnz += nnz_j.zx();
                    }
                    nnz
                }
                None => col_ptr[ncols].zx() - col_ptr[0].zx(),
            }
        }
        imp(self.col_nnz, self.col_ptr, self.ncols.unbound())
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
    pub fn row_indices_of_col_raw(self, j: Idx<C>) -> &'a [Idx<R, I>] {
        unsafe {
            let slice = __get_unchecked(self.row_ind, self.col_range(j));
            let len = slice.len();
            core::slice::from_raw_parts(slice.as_ptr() as *const Idx<R, I>, len)
        }
    }

    /// Returns the row indices of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn row_indices_of_col_raw_unbound(self, j: Idx<C>) -> &'a [I] {
        unsafe { __get_unchecked(self.row_ind, self.col_range(j)) }
    }

    /// Returns the row indices of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn row_indices_of_col(
        self,
        j: Idx<C>,
    ) -> impl 'a + Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<R>> {
        self.row_indices_of_col_raw_unbound(j).iter().map(
            #[inline(always)]
            |&i| unsafe { Idx::<R>::new_unbound(i.zx()) },
        )
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn col_range(&self, j: Idx<C>) -> Range<usize> {
        assert!(j < self.ncols());
        unsafe { self.col_range_unchecked(j) }
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub unsafe fn col_range_unchecked(&self, j: Idx<C>) -> Range<usize> {
        debug_assert!(j < self.ncols());
        let j = j.unbound();

        let start = __get_unchecked(self.col_ptr, j).zx();
        let end = self
            .col_nnz
            .map(|col_nnz| (__get_unchecked(col_nnz, j).zx() + start))
            .unwrap_or(__get_unchecked(self.col_ptr, j + 1).zx());

        start..end
    }

    /// Returns the input matrix with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape<V: Shape, H: Shape>(
        self,
        nrows: V,
        ncols: H,
    ) -> SymbolicSparseColMatRef<'a, I, V, H> {
        assert!(all(
            nrows.unbound() == self.nrows().unbound(),
            ncols.unbound() == self.ncols().unbound(),
        ));
        unsafe {
            SymbolicSparseColMatRef {
                nrows,
                ncols,
                col_ptr: self.col_ptr,
                col_nnz: self.col_nnz,
                row_ind: core::slice::from_raw_parts(
                    self.row_ind.as_ptr() as _,
                    self.row_ind.len(),
                ),
            }
        }
    }

    /// Returns the input matrix with dynamic shape.
    #[inline]
    pub fn as_dyn(self) -> SymbolicSparseColMatRef<'a, I> {
        unsafe {
            SymbolicSparseColMatRef {
                nrows: self.nrows.unbound(),
                ncols: self.ncols.unbound(),
                col_ptr: self.col_ptr,
                col_nnz: self.col_nnz,
                row_ind: core::slice::from_raw_parts(
                    self.row_ind.as_ptr() as _,
                    self.row_ind.len(),
                ),
            }
        }
    }
}

impl<I: Index, R: Shape, C: Shape> core::fmt::Debug for SymbolicSparseColMatRef<'_, I, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use crate::utils::bound::*;
        use generativity::make_guard;

        fn imp<I: Index>(
            mat: SymbolicSparseColMatRef<'_, I, Dim<'_>, Dim<'_>>,
            f: &mut core::fmt::Formatter<'_>,
        ) -> core::fmt::Result {
            let mut iter = mat.ncols().indices().flat_map(move |j| {
                struct Wrapper(usize, usize);
                impl core::fmt::Debug for Wrapper {
                    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                        let row = self.0;
                        let col = self.1;
                        write!(f, "({row}, {col})")
                    }
                }

                mat.row_indices_of_col(j)
                    .map(move |i| Wrapper(i.unbound(), j.unbound()))
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
