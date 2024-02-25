use super::*;
use crate::assert;

/// Sparse matrix view in column-major format, either compressed or uncompressed.
pub struct SparseRowMatRef<'a, I: Index, E: Entity> {
    pub(crate) symbolic: SymbolicSparseRowMatRef<'a, I>,
    pub(crate) values: SliceGroup<'a, E>,
}

impl<I: Index, E: Entity> Copy for SparseRowMatRef<'_, I, E> {}
impl<I: Index, E: Entity> Clone for SparseRowMatRef<'_, I, E> {
    #[inline]
    fn clone(&self) -> Self {
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

impl<'short, I: Index, E: Entity> ReborrowMut<'short> for SparseRowMatRef<'_, I, E> {
    type Target = SparseRowMatRef<'short, I, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
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
        Self { symbolic, values }
    }

    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.symbolic.nrows
    }
    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.symbolic.ncols
    }

    /// Returns a view over `self`.
    #[inline]
    pub fn as_ref(&self) -> SparseRowMatRef<'_, I, E> {
        *self
    }

    /// Returns the numerical values of the matrix.
    #[inline]
    pub fn values(self) -> GroupFor<E, &'a [E::Unit]> {
        self.values.into_inner()
    }

    /// Copies `self` into a newly allocated matrix.
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

    /// Copies `self` into a newly allocated dense matrix
    #[inline]
    pub fn to_dense(&self) -> Mat<E::Canonical>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        let mut mat = Mat::<E::Canonical>::zeros(self.nrows(), self.ncols());

        for i in 0..self.nrows() {
            for (j, val) in self.col_indices_of_row(i).zip(
                crate::utils::slice::SliceGroup::<'_, E>::new(self.values_of_row(i))
                    .into_ref_iter(),
            ) {
                mat.write(i, j, mat.read(i, j).faer_add(val.read().canonicalize()));
            }
        }

        mat
    }

    /// Copies `self` into a newly allocated matrix, with column-major order.
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
            symbolic: SymbolicSparseColMatRef {
                nrows: self.symbolic.ncols,
                ncols: self.symbolic.nrows,
                col_ptr: self.symbolic.row_ptr,
                col_nnz: self.symbolic.row_nnz,
                row_ind: self.symbolic.col_ind,
            },
            values: self.values,
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(self) -> SparseRowMatRef<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        SparseRowMatRef {
            symbolic: self.symbolic,
            values: unsafe {
                SliceGroup::<'a, E::Conj>::new(transmute_unchecked::<
                    GroupFor<E, &[UnitFor<E::Conj>]>,
                    GroupFor<E::Conj, &[UnitFor<E::Conj>]>,
                >(E::faer_map(
                    self.values.into_inner(),
                    |slice| {
                        let len = slice.len();
                        core::slice::from_raw_parts(
                            slice.as_ptr() as *const UnitFor<E> as *const UnitFor<E::Conj>,
                            len,
                        )
                    },
                )))
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
                symbolic: self.symbolic,
                values: unsafe {
                    SliceGroup::<'a, E::Canonical>::new(transmute_unchecked::<
                        GroupFor<E, &[UnitFor<E::Canonical>]>,
                        GroupFor<E::Canonical, &[UnitFor<E::Canonical>]>,
                    >(E::faer_map(
                        self.values.into_inner(),
                        |slice| {
                            let len = slice.len();
                            core::slice::from_raw_parts(
                                slice.as_ptr() as *const UnitFor<E> as *const UnitFor<E::Canonical>,
                                len,
                            )
                        },
                    )))
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
        self.values.subslice(self.row_range(i)).into_inner()
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'a, I> {
        self.symbolic
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn into_parts(self) -> (SymbolicSparseRowMatRef<'a, I>, GroupFor<E, &'a [E::Unit]>) {
        (self.symbolic, self.values.into_inner())
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
        self.symbolic.row_ptrs()
    }

    /// Returns the count of non-zeros per column of the matrix.
    #[inline]
    pub fn nnz_per_row(&self) -> Option<&'a [I]> {
        self.symbolic.nnz_per_row()
    }

    /// Returns the column indices.
    #[inline]
    pub fn col_indices(&self) -> &'a [I] {
        self.symbolic.col_indices()
    }

    /// Returns the column indices of row i.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub fn col_indices_of_row_raw(&self, i: usize) -> &'a [I] {
        self.symbolic.col_indices_of_row_raw(i)
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
        self.symbolic.col_indices_of_row(i)
    }

    /// Returns the range that the row `i` occupies in `self.col_indices()`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub fn row_range(&self, i: usize) -> Range<usize> {
        self.symbolic.row_range(i)
    }

    /// Returns the range that the row `i` occupies in `self.col_indices()`.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub unsafe fn row_range_unchecked(&self, i: usize) -> Range<usize> {
        self.symbolic.row_range_unchecked(i)
    }

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
