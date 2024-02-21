use super::*;
use crate::assert;

/// Sparse matrix view in column-major format, either compressed or uncompressed.
pub struct SparseColMatRef<'a, I: Index, E: Entity> {
    pub(crate) symbolic: SymbolicSparseColMatRef<'a, I>,
    pub(crate) values: SliceGroup<'a, E>,
}

impl<I: Index, E: Entity> Copy for SparseColMatRef<'_, I, E> {}
impl<I: Index, E: Entity> Clone for SparseColMatRef<'_, I, E> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, I: Index, E: Entity> Reborrow<'short> for SparseColMatRef<'_, I, E> {
    type Target = SparseColMatRef<'short, I, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, I: Index, E: Entity> ReborrowMut<'short> for SparseColMatRef<'_, I, E> {
    type Target = SparseColMatRef<'short, I, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'a, I: Index, E: Entity> IntoConst for SparseColMatRef<'a, I, E> {
    type Target = SparseColMatRef<'a, I, E>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'a, I: Index, E: Entity> SparseColMatRef<'a, I, E> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.row_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(
        symbolic: SymbolicSparseColMatRef<'a, I>,
        values: GroupFor<E, &'a [E::Unit]>,
    ) -> Self {
        let values = SliceGroup::new(values);
        assert!(symbolic.row_indices().len() == values.len());
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
    pub fn as_ref(&self) -> SparseColMatRef<'_, I, E> {
        *self
    }

    /// Copies the current matrix into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SparseColMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        let symbolic = self.symbolic().to_owned()?;
        let mut values = VecGroup::<E::Canonical>::new();

        values
            .try_reserve_exact(self.values.len())
            .map_err(|_| FaerError::OutOfMemory)?;

        values.resize(
            self.values.len(),
            E::Canonical::faer_zero().faer_into_units(),
        );

        let src = self.values;
        let dst = values.as_slice_mut();

        for (mut dst, src) in core::iter::zip(dst.into_mut_iter(), src.into_ref_iter()) {
            dst.write(src.read().canonicalize());
        }

        Ok(SparseColMat { symbolic, values })
    }

    /// Copies the current matrix into a newly allocated matrix, with row-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    #[inline]
    pub fn to_row_major(&self) -> Result<SparseRowMat<I, E::Canonical>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        let mut col_ptr = try_zeroed::<I>(self.nrows() + 1)?;
        let nnz = self.compute_nnz();
        let mut row_ind = try_zeroed::<I>(nnz)?;
        let mut values = VecGroup::<E::Canonical>::new();
        values
            .try_reserve_exact(nnz)
            .map_err(|_| FaerError::OutOfMemory)?;
        values.resize(nnz, E::Canonical::faer_zero().faer_into_units());

        let mut mem = GlobalPodBuffer::try_new(StackReq::new::<I>(self.nrows()))
            .map_err(|_| FaerError::OutOfMemory)?;

        let (this, conj) = self.canonicalize();

        if conj == Conj::No {
            utils::transpose::<I, E::Canonical>(
                &mut col_ptr,
                &mut row_ind,
                values.as_slice_mut().into_inner(),
                this,
                PodStack::new(&mut mem),
            );
        } else {
            utils::adjoint::<I, E::Canonical>(
                &mut col_ptr,
                &mut row_ind,
                values.as_slice_mut().into_inner(),
                this,
                PodStack::new(&mut mem),
            );
        }

        let transpose = unsafe {
            SparseColMat::new(
                SymbolicSparseColMat::new_unchecked(
                    self.ncols(),
                    self.nrows(),
                    col_ptr,
                    None,
                    row_ind,
                ),
                values.into_inner(),
            )
        };

        Ok(transpose.into_transpose())
    }

    /// Returns a view over the transpose of `self` in row-major format.
    #[inline]
    pub fn transpose(self) -> SparseRowMatRef<'a, I, E> {
        SparseRowMatRef {
            symbolic: SymbolicSparseRowMatRef {
                nrows: self.symbolic.ncols,
                ncols: self.symbolic.nrows,
                row_ptr: self.symbolic.col_ptr,
                row_nnz: self.symbolic.col_nnz,
                col_ind: self.symbolic.row_ind,
            },
            values: self.values,
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(self) -> SparseColMatRef<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        SparseColMatRef {
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
    pub fn canonicalize(self) -> (SparseColMatRef<'a, I, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        (
            SparseColMatRef {
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
    pub fn adjoint(self) -> SparseRowMatRef<'a, I, E::Conj>
    where
        E: Conjugate,
    {
        self.transpose().conjugate()
    }

    /// Returns the numerical values of the matrix.
    #[inline]
    pub fn values(self) -> GroupFor<E, &'a [E::Unit]> {
        self.values.into_inner()
    }

    /// Returns the numerical values of column `j` of the matrix.
    ///
    /// # Panics:
    ///
    /// Panics if `j >= ncols`.
    #[inline]
    #[track_caller]
    pub fn values_of_col(self, j: usize) -> GroupFor<E, &'a [E::Unit]> {
        self.values.subslice(self.col_range(j)).into_inner()
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseColMatRef<'a, I> {
        self.symbolic
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn into_parts(self) -> (SymbolicSparseColMatRef<'a, I>, GroupFor<E, &'a [E::Unit]>) {
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
    pub fn col_ptrs(&self) -> &'a [I] {
        self.symbolic.col_ptrs()
    }

    /// Returns the count of non-zeros per column of the matrix.
    #[inline]
    pub fn nnz_per_col(&self) -> Option<&'a [I]> {
        self.symbolic.col_nnz
    }

    /// Returns the row indices.
    #[inline]
    pub fn row_indices(&self) -> &'a [I] {
        self.symbolic.row_ind
    }

    /// Returns the row indices of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn row_indices_of_col_raw(&self, j: usize) -> &'a [I] {
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
        j: usize,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = usize> {
        self.symbolic.row_indices_of_col(j)
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub fn col_range(&self, j: usize) -> Range<usize> {
        self.symbolic.col_range(j)
    }

    /// Returns the range that the column `j` occupies in `self.row_indices()`.
    ///
    /// # Safety
    ///
    /// The behavior is undefined if `j >= self.ncols()`.
    #[inline]
    #[track_caller]
    pub unsafe fn col_range_unchecked(&self, j: usize) -> Range<usize> {
        self.symbolic.col_range_unchecked(j)
    }

    /// Returns a reference to the value at the given index, or None if the symbolic structure
    /// doesn't contain it
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
    #[track_caller]
    pub fn get(self, row: usize, col: usize) -> Option<GroupFor<E, &'a E::Unit>> {
        assert!(row < self.nrows());
        assert!(col < self.ncols());

        let Ok(pos) = self
            .row_indices_of_col_raw(col)
            .binary_search(&I::truncate(row))
        else {
            return None;
        };

        Some(E::faer_map(self.values_of_col(col), |slice| &slice[pos]))
    }
}
