use super::*;
use crate::assert;

/// Sparse matrix view in column-major format, either compressed or uncompressed.
pub struct SparseRowMatMut<'a, I: Index, E: Entity> {
    pub(crate) symbolic: SymbolicSparseRowMatRef<'a, I>,
    pub(crate) values: SliceGroupMut<'a, E>,
}

impl<'short, I: Index, E: Entity> Reborrow<'short> for SparseRowMatMut<'_, I, E> {
    type Target = SparseRowMatRef<'short, I, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        SparseRowMatRef {
            symbolic: self.symbolic,
            values: self.values.rb(),
        }
    }
}

impl<'short, I: Index, E: Entity> ReborrowMut<'short> for SparseRowMatMut<'_, I, E> {
    type Target = SparseRowMatMut<'short, I, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        SparseRowMatMut {
            symbolic: self.symbolic,
            values: self.values.rb_mut(),
        }
    }
}

impl<'a, I: Index, E: Entity> IntoConst for SparseRowMatMut<'a, I, E> {
    type Target = SparseRowMatRef<'a, I, E>;

    #[inline]
    fn into_const(self) -> Self::Target {
        SparseRowMatRef {
            symbolic: self.symbolic,
            values: self.values.into_const(),
        }
    }
}

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
        (*self).rb()
    }

    /// Returns a mutable view over `self`.
    ///
    /// Note that the symbolic structure cannot be changed through this view.
    #[inline]
    pub fn as_mut(&mut self) -> SparseRowMatMut<'_, I, E> {
        (*self).rb_mut()
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
        self.rb().to_owned()
    }

    /// Copies `self` into a newly allocated dense matrix
    #[inline]
    pub fn to_dense(&self) -> Mat<E::Canonical>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.as_ref().to_dense()
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
        self.rb().to_col_major()
    }

    /// Returns a view over the transpose of `self` in column-major format.
    #[inline]
    pub fn transpose_mut(self) -> SparseColMatMut<'a, I, E> {
        SparseColMatMut {
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
    pub fn canonicalize_mut(self) -> (SparseRowMatMut<'a, I, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        (
            SparseRowMatMut {
                symbolic: self.symbolic,
                values: unsafe {
                    SliceGroupMut::<'a, E::Canonical>::new(transmute_unchecked::<
                        GroupFor<E, &mut [UnitFor<E::Canonical>]>,
                        GroupFor<E::Canonical, &mut [UnitFor<E::Canonical>]>,
                    >(E::faer_map(
                        self.values.into_inner(),
                        |slice| {
                            let len = slice.len();
                            core::slice::from_raw_parts_mut(
                                slice.as_mut_ptr() as *mut UnitFor<E> as *mut UnitFor<E::Canonical>,
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

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate_mut(self) -> SparseRowMatMut<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        SparseRowMatMut {
            symbolic: self.symbolic,
            values: unsafe {
                SliceGroupMut::<'a, E::Conj>::new(transmute_unchecked::<
                    GroupFor<E, &mut [UnitFor<E::Conj>]>,
                    GroupFor<E::Conj, &mut [UnitFor<E::Conj>]>,
                >(E::faer_map(
                    self.values.into_inner(),
                    |slice| {
                        let len = slice.len();
                        core::slice::from_raw_parts_mut(
                            slice.as_mut_ptr() as *mut UnitFor<E> as *mut UnitFor<E::Conj>,
                            len,
                        )
                    },
                )))
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
        self.values.into_inner()
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
        self.values.subslice(range).into_inner()
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'a, I> {
        self.symbolic
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn into_parts(
        self,
    ) -> (
        SymbolicSparseRowMatRef<'a, I>,
        GroupFor<E, &'a mut [E::Unit]>,
    ) {
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
        self.symbolic.compute_nnz()
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

    /// Returns a reference to the value at the given index, or None if the symbolic structure
    /// doesn't contain it, or contains multiple values with the given index.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get(self, row: usize, col: usize) -> Option<GroupFor<E, &'a E::Unit>> {
        self.into_const().get(row, col)
    }

    /// Returns a reference to the value at the given index, or None if the symbolic structure
    /// doesn't contain it, or contains multiple values with the given index.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get_mut(self, row: usize, col: usize) -> Option<GroupFor<E, &'a mut E::Unit>> {
        let values = self.get_all_mut(row, col);
        if E::faer_first(E::faer_as_ref(&values)).len() == 1 {
            Some(E::faer_map(values, |slice| &mut slice[0]))
        } else {
            None
        }
    }

    /// Returns a reference to a slice containing the values at the given index using a binary
    /// search.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get_all(self, row: usize, col: usize) -> GroupFor<E, &'a [E::Unit]> {
        self.into_const().get_all(row, col)
    }

    /// Returns a reference to a slice containing the values at the given index using a binary
    /// search.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get_all_mut(self, row: usize, col: usize) -> GroupFor<E, &'a mut [E::Unit]> {
        assert!(row < self.nrows());
        assert!(col < self.ncols());

        let col = I::truncate(col);
        let start = self
            .col_indices_of_row_raw(row)
            .partition_point(|&p| p < col);
        let end = start + self.col_indices_of_row_raw(row)[start..].partition_point(|&p| p <= col);

        E::faer_map(self.values_of_row_mut(row), |slice| &mut slice[start..end])
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
