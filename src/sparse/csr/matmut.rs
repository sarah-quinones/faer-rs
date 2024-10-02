use super::*;
use crate::assert;

/// Sparse matrix view in column-major format, either compressed or uncompressed.
pub struct SparseRowMatMut<'a, I: Index, E: Entity, R: Shape = usize, C: Shape = usize> {
    pub(crate) symbolic: SymbolicSparseRowMatRef<'a, I, R, C>,
    pub(crate) values: SliceGroupMut<'a, E>,
}

impl<'short, I: Index, E: Entity, R: Shape, C: Shape> Reborrow<'short>
    for SparseRowMatMut<'_, I, E, R, C>
{
    type Target = SparseRowMatRef<'short, I, E, R, C>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        SparseRowMatRef {
            symbolic: self.symbolic,
            values: self.values.rb(),
        }
    }
}

impl<'short, I: Index, E: Entity, R: Shape, C: Shape> ReborrowMut<'short>
    for SparseRowMatMut<'_, I, E, R, C>
{
    type Target = SparseRowMatMut<'short, I, E, R, C>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        SparseRowMatMut {
            symbolic: self.symbolic,
            values: self.values.rb_mut(),
        }
    }
}

impl<'a, I: Index, E: Entity, R: Shape, C: Shape> IntoConst for SparseRowMatMut<'a, I, E, R, C> {
    type Target = SparseRowMatRef<'a, I, E, R, C>;

    #[inline]
    fn into_const(self) -> Self::Target {
        SparseRowMatRef {
            symbolic: self.symbolic,
            values: self.values.into_const(),
        }
    }
}

impl<'a, I: Index, E: Entity, R: Shape, C: Shape> SparseRowMatMut<'a, I, E, R, C> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.col_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(symbolic: SymbolicSparseRowMatRef<'a, I, R, C>, values: SliceMut<'a, E>) -> Self {
        let values = SliceGroupMut::new(values);
        assert!(symbolic.col_indices().len() == values.len());
        Self { symbolic, values }
    }

    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> R {
        self.symbolic.nrows
    }
    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> C {
        self.symbolic.ncols
    }

    /// Returns the number of rows and columns of the matrix.
    #[inline]
    pub fn shape(&self) -> (R, C) {
        (self.nrows(), self.ncols())
    }

    /// Returns a view over `self`.
    #[inline]
    pub fn as_ref(&self) -> SparseRowMatRef<'_, I, E, R, C> {
        (*self).rb()
    }

    /// Returns a mutable view over `self`.
    ///
    /// Note that the symbolic structure cannot be changed through this view.
    #[inline]
    pub fn as_mut(&mut self) -> SparseRowMatMut<'_, I, E, R, C> {
        (*self).rb_mut()
    }

    /// Copies `self` into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SparseRowMat<I, E::Canonical, R, C>, FaerError>
    where
        E: Conjugate<Canonical: ComplexField>,
    {
        self.rb().to_owned()
    }

    /// Copies `self` into a newly allocated matrix with sorted indices.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output.
    #[inline]
    pub fn to_sorted(&self) -> Result<SparseRowMat<I, E::Canonical, R, C>, FaerError>
    where
        E: Conjugate<Canonical: ComplexField>,
    {
        self.as_ref().to_sorted()
    }

    /// Copies `self` into a newly allocated dense matrix
    #[inline]
    pub fn to_dense(&self) -> Mat<E::Canonical, R, C>
    where
        E: Conjugate<Canonical: ComplexField>,
    {
        self.as_ref().to_dense()
    }

    /// Copies `self` into a newly allocated matrix, with column-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output.
    #[inline]
    pub fn to_col_major(&self) -> Result<SparseColMat<I, E::Canonical, R, C>, FaerError>
    where
        E: Conjugate<Canonical: ComplexField>,
    {
        self.rb().to_col_major()
    }

    /// Returns a view over the transpose of `self` in column-major format.
    #[inline]
    pub fn transpose(self) -> SparseColMatRef<'a, I, E, C, R> {
        self.into_const().transpose()
    }

    /// Returns a view over the transpose of `self` in column-major format.
    #[inline]
    pub fn transpose_mut(self) -> SparseColMatMut<'a, I, E, C, R> {
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

    /// Returns a view over the canonical view of `self`, along with whether it has been conjugated
    /// or not.
    #[inline]
    pub fn canonicalize(self) -> (SparseRowMatRef<'a, I, E::Canonical, R, C>, Conj)
    where
        E: Conjugate,
    {
        self.into_const().canonicalize()
    }

    /// Returns a view over the canonical view of `self`, along with whether it has been conjugated
    /// or not.
    #[inline]
    pub fn canonicalize_mut(self) -> (SparseRowMatMut<'a, I, E::Canonical, R, C>, Conj)
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
    pub fn conjugate(self) -> SparseRowMatRef<'a, I, E::Conj, R, C>
    where
        E: Conjugate,
    {
        self.into_const().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate_mut(self) -> SparseRowMatMut<'a, I, E::Conj, R, C>
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
    pub fn adjoint(self) -> SparseColMatRef<'a, I, E::Conj, C, R>
    where
        E: Conjugate,
    {
        self.into_const().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint_mut(self) -> SparseColMatMut<'a, I, E::Conj, C, R>
    where
        E: Conjugate,
    {
        self.transpose_mut().conjugate_mut()
    }

    /// Returns the numerical values of the matrix.
    #[inline]
    pub fn values(self) -> Slice<'a, E> {
        self.into_const().values()
    }

    /// Returns the numerical values of the matrix.
    #[inline]
    pub fn values_mut(self) -> SliceMut<'a, E> {
        self.values.into_inner()
    }

    /// Returns the numerical values of row `i` of the matrix.
    ///
    /// # Panics:
    ///
    /// Panics if `i >= nrows`.
    #[inline]
    #[track_caller]
    pub fn values_of_row(self, i: Idx<R>) -> Slice<'a, E> {
        self.into_const().values_of_row(i)
    }

    /// Returns the numerical values of row `i` of the matrix.
    ///
    /// # Panics:
    ///
    /// Panics if `i >= nrows`.
    #[inline]
    #[track_caller]
    pub fn values_of_row_mut(self, i: Idx<R>) -> SliceMut<'a, E> {
        let range = self.symbolic().row_range(i);
        self.values.subslice(range).into_inner()
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'a, I, R, C> {
        self.symbolic
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn parts(self) -> (SymbolicSparseRowMatRef<'a, I, R, C>, Slice<'a, E>) {
        self.into_const().parts()
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn parts_mut(self) -> (SymbolicSparseRowMatRef<'a, I, R, C>, SliceMut<'a, E>) {
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
    pub fn col_indices_of_row_raw(&self, i: Idx<R>) -> &'a [Idx<C, I>] {
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
        i: Idx<R>,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = Idx<C>> {
        self.symbolic.col_indices_of_row(i)
    }

    /// Returns the range that the row `i` occupies in `self.col_indices()`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub fn row_range(&self, i: Idx<R>) -> Range<usize> {
        self.symbolic.row_range(i)
    }

    /// Returns the range that the row `i` occupies in `self.col_indices()`.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub unsafe fn row_range_unchecked(&self, i: Idx<R>) -> Range<usize> {
        self.symbolic.row_range_unchecked(i)
    }

    /// Returns a reference to the value at the given index, or None if the symbolic structure
    /// doesn't contain it, or contains multiple values with the given index.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get(self, row: Idx<R>, col: Idx<C>) -> Option<Ref<'a, E>> {
        self.into_const().get(row, col)
    }

    /// Returns a reference to the value at the given index, or None if the symbolic structure
    /// doesn't contain it, or contains multiple values with the given index.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get_mut(self, row: Idx<R>, col: Idx<C>) -> Option<Mut<'a, E>> {
        assert!(row < self.nrows());
        assert!(col < self.ncols());
        self.transpose_mut().get_mut(col, row)
    }

    /// Returns a reference to a slice containing the values at the given index using a binary
    /// search.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get_all(self, row: Idx<R>, col: Idx<C>) -> Slice<'a, E> {
        self.into_const().get_all(row, col)
    }

    /// Returns a reference to a slice containing the values at the given index using a binary
    /// search.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get_all_mut(self, row: Idx<R>, col: Idx<C>) -> SliceMut<'a, E> {
        assert!(row < self.nrows());
        assert!(col < self.ncols());

        self.transpose_mut().get_all_mut(col, row)
    }

    /// Fill the matrix from a previously created value order.
    /// The provided values must correspond to the same indices that were provided in the
    /// function call from which the order was created.
    ///
    /// # Note
    /// The symbolic structure is not changed.
    #[inline]
    #[track_caller]
    pub fn fill_from_order_and_values(
        &mut self,
        order: &ValuesOrder<I>,
        values: Slice<'_, E>,
        mode: FillMode,
    ) where
        E: ComplexField,
    {
        self.rb_mut()
            .transpose_mut()
            .fill_from_order_and_values(order, values, mode);
    }

    /// Returns the input matrix with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape<V: Shape, H: Shape>(
        self,
        nrows: V,
        ncols: H,
    ) -> SparseRowMatRef<'a, I, E, V, H> {
        SparseRowMatRef {
            symbolic: self.symbolic.as_shape(nrows, ncols),
            values: self.values.into_const(),
        }
    }

    /// Returns the input matrix with dynamic shape.
    #[inline]
    pub fn as_dyn(self) -> SparseRowMatRef<'a, I, E> {
        SparseRowMatRef {
            symbolic: self.symbolic.as_dyn(),
            values: self.values.into_const(),
        }
    }

    /// Returns the input matrix with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape_mut<V: Shape, H: Shape>(
        self,
        nrows: V,
        ncols: H,
    ) -> SparseRowMatMut<'a, I, E, V, H> {
        SparseRowMatMut {
            symbolic: self.symbolic.as_shape(nrows, ncols),
            values: self.values,
        }
    }

    /// Returns the input matrix with dynamic shape.
    #[inline]
    pub fn as_dyn_mut(self) -> SparseRowMatMut<'a, I, E> {
        SparseRowMatMut {
            symbolic: self.symbolic.as_dyn(),
            values: self.values,
        }
    }
}

impl<I: Index, E: Entity> core::fmt::Debug for SparseRowMatMut<'_, I, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}
