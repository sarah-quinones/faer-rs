use super::*;
use crate::assert;

#[derive(Clone)]
/// Sparse matrix in column-major format, either compressed or uncompressed.
pub struct SparseColMat<I: Index, E: Entity, R: Shape = usize, C: Shape = usize> {
    pub(crate) symbolic: SymbolicSparseColMat<I, R, C>,
    pub(crate) values: VecGroup<E>,
}

impl<I: Index, E: Entity, R: Shape, C: Shape> SparseColMat<I, E, R, C> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.row_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(symbolic: SymbolicSparseColMat<I, R, C>, values: Vector<E>) -> Self {
        let values = VecGroup::from_inner(values);
        assert!(symbolic.row_indices().len() == values.len());
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
    pub fn to_owned(&self) -> Result<SparseColMat<I, E::Canonical, R, C>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        self.as_ref().to_owned()
    }

    /// Copies `self` into a newly allocated matrix with sorted indices.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output.
    #[inline]
    pub fn to_sorted(&self) -> Result<SparseColMat<I, E::Canonical, R, C>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
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

    /// Copies `self` into a newly allocated matrix, with row-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output.
    #[inline]
    pub fn to_row_major(&self) -> Result<SparseRowMat<I, E::Canonical, R, C>, FaerError>
    where
        E: Conjugate<Canonical: ComplexField>,
    {
        self.as_ref().to_row_major()
    }

    /// Sorts the row indices in each column of the matrix in-place.
    ///
    /// # Note
    /// Allows unsorted matrices.
    pub fn sort_indices(&mut self) {
        let len = self.symbolic.row_ind.len();
        utils::sort_indices::<I, E>(
            &self.symbolic.col_ptr,
            self.symbolic.col_nnz.as_deref(),
            unsafe {
                core::slice::from_raw_parts_mut(self.symbolic.row_ind.as_mut_ptr() as _, len)
            },
            self.values.as_slice_mut().into_inner(),
        );
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn parts(&self) -> (SymbolicSparseColMatRef<'_, I, R, C>, Slice<'_, E>) {
        self.as_ref().parts()
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn parts_mut(&mut self) -> (SymbolicSparseColMatRef<'_, I, R, C>, SliceMut<'_, E>) {
        self.as_mut().parts_mut()
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn into_parts(self) -> (SymbolicSparseColMat<I, R, C>, Vector<E>) {
        (self.symbolic, self.values.into_inner())
    }

    /// Returns a view over `self`.
    #[inline]
    pub fn as_ref(&self) -> SparseColMatRef<'_, I, E, R, C> {
        SparseColMatRef {
            symbolic: self.symbolic.as_ref(),
            values: self.values.as_slice(),
        }
    }

    /// Returns a mutable view over `self`.
    ///
    /// Note that the symbolic structure cannot be changed through this view.
    #[inline]
    pub fn as_mut(&mut self) -> SparseColMatMut<'_, I, E, R, C> {
        SparseColMatMut {
            symbolic: self.symbolic.as_ref(),
            values: self.values.as_slice_mut(),
        }
    }

    /// Returns a slice over the numerical values of the matrix.
    #[inline]
    pub fn values(&self) -> Slice<'_, E> {
        self.values.as_slice().into_inner()
    }

    /// Returns a mutable slice over the numerical values of the matrix.
    #[inline]
    pub fn values_mut(&mut self) -> SliceMut<'_, E> {
        self.values.as_slice_mut().into_inner()
    }

    /// Returns a view over the transpose of `self` in row-major format.
    #[inline]
    pub fn transpose(&self) -> SparseRowMatRef<'_, I, E, C, R> {
        self.as_ref().transpose()
    }

    /// Returns a view over the transpose of `self` in row-major format.
    #[inline]
    pub fn transpose_mut(&mut self) -> SparseRowMatMut<'_, I, E, C, R> {
        self.as_mut().transpose_mut()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(&self) -> SparseColMatRef<'_, I, E::Conj, R, C>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate_mut(&mut self) -> SparseColMatMut<'_, I, E::Conj, R, C>
    where
        E: Conjugate,
    {
        self.as_mut().conjugate_mut()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(&self) -> SparseRowMatRef<'_, I, E::Conj, C, R>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint_mut(&mut self) -> SparseRowMatMut<'_, I, E::Conj, C, R>
    where
        E: Conjugate,
    {
        self.as_mut().adjoint_mut()
    }

    /// Returns a view over the canonical representation of `self`, and whether it needs to be
    /// conjugated or not.
    #[inline]
    pub fn canonicalize(&self) -> (SparseColMatRef<'_, I, E::Canonical, R, C>, Conj)
    where
        E: Conjugate,
    {
        self.as_ref().canonicalize()
    }

    /// Returns a view over the canonical representation of `self`, and whether it needs to be
    /// conjugated or not.
    #[inline]
    pub fn canonicalize_mut(&mut self) -> (SparseColMatMut<'_, I, E::Canonical, R, C>, Conj)
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
        values: Slice<'_, E>,
        mode: FillMode,
    ) where
        E: ComplexField,
    {
        self.as_mut()
            .fill_from_order_and_values(order, values, mode)
    }

    /// Returns the transpose of `self` in row-major format.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn into_transpose(self) -> SparseRowMat<I, E, C, R> {
        SparseRowMat {
            symbolic: SymbolicSparseRowMat {
                nrows: self.symbolic.ncols,
                ncols: self.symbolic.nrows,
                row_ptr: self.symbolic.col_ptr,
                row_nnz: self.symbolic.col_nnz,
                col_ind: self.symbolic.row_ind,
            },
            values: self.values,
        }
    }

    /// Returns the conjugate of `self`.
    #[inline]
    pub fn into_conjugate(self) -> SparseColMat<I, E::Conj, R, C>
    where
        E: Conjugate,
    {
        SparseColMat {
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
    pub fn into_adjoint(self) -> SparseRowMat<I, E::Conj, C, R>
    where
        E: Conjugate,
    {
        self.into_conjugate().into_transpose()
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
    pub fn col_ptrs(&self) -> &'_ [I] {
        self.symbolic.col_ptrs()
    }

    /// Returns the count of non-zeros per column of the matrix.
    #[inline]
    pub fn nnz_per_col(&self) -> Option<&'_ [I]> {
        self.symbolic.col_nnz.as_deref()
    }

    /// Returns the row indices.
    #[inline]
    pub fn row_indices(&self) -> &'_ [I] {
        &self.symbolic.row_ind
    }

    /// Returns the row indices of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn row_indices_of_col_raw(&self, j: Idx<C>) -> &'_ [Idx<R, I>] {
        self.symbolic.row_indices_of_col_raw(j)
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
        self.symbolic.row_indices_of_col(j)
    }

    /// Returns the numerical values of column `j` of the matrix.
    ///
    /// # Panics:
    ///
    /// Panics if `j >= ncols`.
    #[inline]
    #[track_caller]
    pub fn values_of_col(&self, j: Idx<C>) -> Slice<'_, E> {
        self.as_ref().values_of_col(j)
    }

    /// Returns the numerical values of column `j` of the matrix.
    ///
    /// # Panics:
    ///
    /// Panics if `j >= ncols`.
    #[inline]
    #[track_caller]
    pub fn values_of_col_mut(&mut self, j: Idx<C>) -> SliceMut<'_, E> {
        self.as_mut().values_of_col_mut(j)
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseColMatRef<'_, I, R, C> {
        self.symbolic.as_ref()
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn col_range(&self, j: Idx<C>) -> Range<usize> {
        self.symbolic.col_range(j)
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub unsafe fn col_range_unchecked(&self, j: Idx<C>) -> Range<usize> {
        self.symbolic.col_range_unchecked(j)
    }

    /// Returns a reference to the value at the given index, or None if the symbolic structure
    /// doesn't contain it, or contains multiple values with the given index.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get(&self, row: Idx<R>, col: Idx<C>) -> Option<Ref<'_, E>> {
        self.as_ref().get(row, col)
    }

    /// Returns a reference to the value at the given index, or None if the symbolic structure
    /// doesn't contain it, or contains multiple values with the given index.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get_mut(&mut self, row: Idx<R>, col: Idx<C>) -> Option<Mut<'_, E>> {
        self.as_mut().get_mut(row, col)
    }

    /// Returns a reference to a slice containing the values at the given index using a binary
    /// search.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get_all(&self, row: Idx<R>, col: Idx<C>) -> Slice<'_, E> {
        self.as_ref().get_all(row, col)
    }

    /// Returns a mutable reference to a slice containing the values at the given index using a
    /// binary search.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get_all_mut(&mut self, row: Idx<R>, col: Idx<C>) -> SliceMut<'_, E> {
        self.as_mut().get_all_mut(row, col)
    }

    #[track_caller]
    pub(crate) fn new_from_order_and_values_impl(
        symbolic: SymbolicSparseColMat<I, R, C>,
        order: &ValuesOrder<I>,
        all_values: impl Fn(usize) -> E,
        values_len: usize,
    ) -> Result<Self, FaerError>
    where
        E: ComplexField,
    {
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

        Ok(Self { symbolic, values })
    }

    /// Create a new matrix from a previously created symbolic structure and value order.
    /// The provided values must correspond to the same indices that were provided in the
    /// function call from which the order was created.
    #[track_caller]
    pub fn new_from_order_and_values(
        symbolic: SymbolicSparseColMat<I, R, C>,
        order: &ValuesOrder<I>,
        values: Slice<'_, E>,
    ) -> Result<Self, FaerError>
    where
        E: ComplexField,
    {
        let values = SliceGroup::<'_, E>::new(values);
        Self::new_from_order_and_values_impl(symbolic, order, |i| values.read(i), values.len())
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

impl<I: Index, E: Entity> SparseColMat<I, E> {
    /// Create a new matrix from triplets `(row, col, value)`. Negative indices are ignored.
    #[track_caller]
    pub fn try_new_from_nonnegative_triplets(
        nrows: usize,
        ncols: usize,
        triplets: &[(I::Signed, I::Signed, E)],
    ) -> Result<Self, CreationError>
    where
        E: ComplexField,
    {
        let (symbolic, order) = SymbolicSparseColMat::<I>::try_new_from_nonnegative_indices_impl(
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

impl<I: Index, E: Entity> core::fmt::Debug for SparseColMat<I, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}
