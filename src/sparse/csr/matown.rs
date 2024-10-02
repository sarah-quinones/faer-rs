use super::*;
use crate::assert;

#[derive(Clone)]
/// Sparse matrix in column-major format, either compressed or uncompressed.
pub struct SparseRowMat<I: Index, E: Entity, R: Shape = usize, C: Shape = usize> {
    pub(crate) symbolic: SymbolicSparseRowMat<I, R, C>,
    pub(crate) values: VecGroup<E>,
}

impl<I: Index, E: Entity, R: Shape, C: Shape> SparseRowMat<I, E, R, C> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.col_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(
        symbolic: SymbolicSparseRowMat<I, R, C>,
        values: GroupFor<E, alloc::vec::Vec<E::Unit>>,
    ) -> Self {
        let values = VecGroup::from_inner(values);
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

    /// Copies `self` into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SparseRowMat<I, E::Canonical, R, C>, FaerError>
    where
        E: Conjugate<Canonical: ComplexField>,
    {
        self.as_ref().to_owned()
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
        self.as_ref().to_col_major()
    }

    /// Sorts the row indices in each column of the matrix in-place.
    ///
    /// # Note
    /// Allows unsorted matrices.
    pub fn sort_indices(&mut self) {
        let len = self.symbolic.col_ind.len();
        utils::sort_indices::<I, E>(
            &self.symbolic.row_ptr,
            self.symbolic.row_nnz.as_deref(),
            unsafe {
                core::slice::from_raw_parts_mut(self.symbolic.col_ind.as_mut_ptr() as _, len)
            },
            self.values.as_slice_mut().into_inner(),
        );
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn parts(
        &self,
    ) -> (
        SymbolicSparseRowMatRef<'_, I, R, C>,
        GroupFor<E, &'_ [E::Unit]>,
    ) {
        self.as_ref().parts()
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn parts_mut(
        &mut self,
    ) -> (
        SymbolicSparseRowMatRef<'_, I, R, C>,
        GroupFor<E, &'_ mut [E::Unit]>,
    ) {
        self.as_mut().parts_mut()
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn into_parts(
        self,
    ) -> (
        SymbolicSparseRowMat<I, R, C>,
        GroupFor<E, alloc::vec::Vec<E::Unit>>,
    ) {
        (self.symbolic, self.values.into_inner())
    }

    /// Returns a view over `self`.
    #[inline]
    pub fn as_ref(&self) -> SparseRowMatRef<'_, I, E, R, C> {
        SparseRowMatRef {
            symbolic: self.symbolic.as_ref(),
            values: self.values.as_slice(),
        }
    }

    /// Returns a mutable view over `self`.
    ///
    /// Note that the symbolic structure cannot be changed through this view.
    #[inline]
    pub fn as_mut(&mut self) -> SparseRowMatMut<'_, I, E, R, C> {
        SparseRowMatMut {
            symbolic: self.symbolic.as_ref(),
            values: self.values.as_slice_mut(),
        }
    }

    /// Returns a slice over the numerical values of the matrix.
    #[inline]
    pub fn values(&self) -> GroupFor<E, &'_ [E::Unit]> {
        self.values.as_slice().into_inner()
    }

    /// Returns a mutable slice over the numerical values of the matrix.
    #[inline]
    pub fn values_mut(&mut self) -> GroupFor<E, &'_ mut [E::Unit]> {
        self.values.as_slice_mut().into_inner()
    }

    /// Returns a view over the transpose of `self` in column-major format.
    #[inline]
    pub fn transpose(&self) -> SparseColMatRef<'_, I, E, C, R> {
        self.as_ref().transpose()
    }

    /// Returns a view over the transpose of `self` in column-major format.
    #[inline]
    pub fn transpose_mut(&mut self) -> SparseColMatMut<'_, I, E, C, R> {
        self.as_mut().transpose_mut()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(&self) -> SparseRowMatRef<'_, I, E::Conj, R, C>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate_mut(&mut self) -> SparseRowMatMut<'_, I, E::Conj, R, C>
    where
        E: Conjugate,
    {
        self.as_mut().conjugate_mut()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(&self) -> SparseColMatRef<'_, I, E::Conj, C, R>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint_mut(&mut self) -> SparseColMatMut<'_, I, E::Conj, C, R>
    where
        E: Conjugate,
    {
        self.as_mut().adjoint_mut()
    }

    /// Returns a view over the canonical representation of `self`, and whether it needs to be
    /// conjugated or not.
    #[inline]
    pub fn canonicalize(&self) -> (SparseRowMatRef<'_, I, E::Canonical, R, C>, Conj)
    where
        E: Conjugate,
    {
        self.as_ref().canonicalize()
    }

    /// Returns a view over the canonical representation of `self`, and whether it needs to be
    /// conjugated or not.
    #[inline]
    pub fn canonicalize_mut(&mut self) -> (SparseRowMatMut<'_, I, E::Canonical, R, C>, Conj)
    where
        E: Conjugate,
    {
        self.as_mut().canonicalize_mut()
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
        values: GroupFor<E, &[E::Unit]>,
        mode: FillMode,
    ) where
        E: ComplexField,
    {
        self.as_mut()
            .fill_from_order_and_values(order, values, mode)
    }

    /// Returns the transpose of `self` in column-major format.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn into_transpose(self) -> SparseColMat<I, E, C, R> {
        SparseColMat {
            symbolic: SymbolicSparseColMat {
                nrows: self.symbolic.ncols,
                ncols: self.symbolic.nrows,
                col_ptr: self.symbolic.row_ptr,
                col_nnz: self.symbolic.row_nnz,
                row_ind: self.symbolic.col_ind,
            },
            values: self.values,
        }
    }

    /// Returns the conjugate of `self`.
    #[inline]
    pub fn into_conjugate(self) -> SparseRowMat<I, E::Conj, R, C>
    where
        E: Conjugate,
    {
        SparseRowMat {
            symbolic: self.symbolic,
            values: unsafe {
                VecGroup::<E::Conj>::from_inner(transmute_unchecked::<
                    GroupFor<E, alloc::vec::Vec<UnitFor<E::Conj>>>,
                    GroupFor<E::Conj, alloc::vec::Vec<UnitFor<E::Conj>>>,
                >(E::faer_map(
                    self.values.into_inner(),
                    |slice| {
                        let mut slice = core::mem::ManuallyDrop::new(slice);

                        let len = slice.len();
                        let cap = slice.capacity();
                        let ptr = slice.as_mut_ptr() as *mut UnitFor<E> as *mut UnitFor<E::Conj>;

                        alloc::vec::Vec::from_raw_parts(ptr, len, cap)
                    },
                )))
            },
        }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn into_adjoint(self) -> SparseColMat<I, E::Conj, C, R>
    where
        E: Conjugate,
    {
        self.into_transpose().into_conjugate()
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
    pub fn row_ptrs(&self) -> &'_ [I] {
        self.symbolic.row_ptrs()
    }

    /// Returns the count of non-zeros per column of the matrix.
    #[inline]
    pub fn nnz_per_row(&self) -> Option<&'_ [I]> {
        self.symbolic.nnz_per_row()
    }

    /// Returns the column indices.
    #[inline]
    pub fn col_indices(&self) -> &'_ [I] {
        self.symbolic.col_indices()
    }

    /// Returns the column indices of row i.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    #[inline]
    #[track_caller]
    pub fn col_indices_of_row_raw(&self, i: Idx<R>) -> &'_ [Idx<C, I>] {
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
    ) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = Idx<C>> {
        self.symbolic.col_indices_of_row(i)
    }

    /// Returns the numerical values of row `i` of the matrix.
    ///
    /// # Panics:
    ///
    /// Panics if `i >= nrows`.
    #[inline]
    #[track_caller]
    pub fn values_of_row(&self, i: Idx<R>) -> GroupFor<E, &[E::Unit]> {
        self.as_ref().values_of_row(i)
    }

    /// Returns the numerical values of row `i` of the matrix.
    ///
    /// # Panics:
    ///
    /// Panics if `i >= nrows`.
    #[inline]
    #[track_caller]
    pub fn values_of_row_mut(&mut self, i: Idx<R>) -> GroupFor<E, &mut [E::Unit]> {
        self.as_mut().values_of_row_mut(i)
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'_, I, R, C> {
        self.symbolic.as_ref()
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
    pub fn get(&self, row: Idx<R>, col: Idx<C>) -> Option<GroupFor<E, &'_ E::Unit>> {
        self.as_ref().get(row, col)
    }

    /// Returns a reference to the value at the given index, or None if the symbolic structure
    /// doesn't contain it, or contains multiple values with the given index.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get_mut(&mut self, row: Idx<R>, col: Idx<C>) -> Option<GroupFor<E, &'_ mut E::Unit>> {
        self.as_mut().get_mut(row, col)
    }

    /// Returns a reference to a slice containing the values at the given index using a binary
    /// search.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get_all(&self, row: Idx<R>, col: Idx<C>) -> GroupFor<E, &'_ [E::Unit]> {
        self.as_ref().get_all(row, col)
    }

    /// Returns a mutable reference to a slice containing the values at the given index using a
    /// binary search.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get_all_mut(&mut self, row: Idx<R>, col: Idx<C>) -> GroupFor<E, &'_ mut [E::Unit]> {
        self.as_mut().get_all_mut(row, col)
    }

    /// Create a new matrix from a previously created symbolic structure and value order.
    /// The provided values must correspond to the same indices that were provided in the
    /// function call from which the order was created.
    #[track_caller]
    pub fn new_from_order_and_values(
        symbolic: SymbolicSparseRowMat<I, R, C>,
        order: &ValuesOrder<I>,
        values: GroupFor<E, &[E::Unit]>,
    ) -> Result<Self, FaerError>
    where
        E: ComplexField,
    {
        SparseColMat::new_from_order_and_values(symbolic.into_transpose(), order, values)
            .map(SparseColMat::into_transpose)
    }

    /// Create a new matrix from triplets `(row, col, value)`.
    #[track_caller]
    pub fn try_new_from_triplets(
        nrows: R,
        ncols: C,
        triplets: &[(Idx<R, I>, Idx<C, I>, E)],
    ) -> Result<Self, CreationError>
    where
        E: ComplexField,
    {
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
}

impl<I: Index, E: ComplexField> SparseRowMat<I, E> {
    /// Create a new matrix from triplets `(row, col, value)`. Negative indices are ignored.
    #[track_caller]
    pub fn try_new_from_nonnegative_triplets(
        nrows: usize,
        ncols: usize,
        triplets: &[(I::Signed, I::Signed, E)],
    ) -> Result<Self, CreationError> {
        let (symbolic, order) = SymbolicSparseColMat::<I>::try_new_from_nonnegative_indices_impl(
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

impl<I: Index, E: Entity> core::fmt::Debug for SparseRowMat<I, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}
