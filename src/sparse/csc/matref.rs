use super::*;
use crate::assert;

/// Sparse matrix view in column-major format, either compressed or uncompressed.
pub struct SparseColMatRef<'a, I: Index, E: Entity, R: Shape = usize, C: Shape = usize> {
    pub(crate) symbolic: SymbolicSparseColMatRef<'a, I, R, C>,
    pub(crate) values: SliceGroup<'a, E>,
}

impl<I: Index, E: Entity, R: Shape, C: Shape> Copy for SparseColMatRef<'_, I, E, R, C> {}
impl<I: Index, E: Entity, R: Shape, C: Shape> Clone for SparseColMatRef<'_, I, E, R, C> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, I: Index, E: Entity, R: Shape, C: Shape> Reborrow<'short>
    for SparseColMatRef<'_, I, E, R, C>
{
    type Target = SparseColMatRef<'short, I, E, R, C>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, I: Index, E: Entity, R: Shape, C: Shape> ReborrowMut<'short>
    for SparseColMatRef<'_, I, E, R, C>
{
    type Target = SparseColMatRef<'short, I, E, R, C>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'a, I: Index, E: Entity, R: Shape, C: Shape> IntoConst for SparseColMatRef<'a, I, E, R, C> {
    type Target = SparseColMatRef<'a, I, E, R, C>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'a, I: Index, E: Entity, R: Shape, C: Shape> SparseColMatRef<'a, I, E, R, C> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.row_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(symbolic: SymbolicSparseColMatRef<'a, I, R, C>, values: Slice<'a, E>) -> Self {
        let values = SliceGroup::new(values);
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

    /// Returns a view over `self`.
    #[inline]
    pub fn as_ref(&self) -> SparseColMatRef<'_, I, E, R, C> {
        *self
    }

    /// Copies `self` into a newly allocated matrix.
    ///
    /// # Note
    /// Allows unsorted matrices, producing an unsorted output.
    #[inline]
    pub fn to_owned(&self) -> Result<SparseColMat<I, E::Canonical, R, C>, FaerError>
    where
        E: Conjugate<Canonical: ComplexField>,
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
        let mut mat = (*self).to_owned()?;
        mat.sort_indices();
        Ok(mat)
    }

    /// Copies `self` into a newly allocated dense matrix
    #[inline]
    pub fn to_dense(&self) -> Mat<E::Canonical, R, C>
    where
        E: Conjugate<Canonical: ComplexField>,
    {
        use crate::utils::bound::*;
        use generativity::make_guard;

        fn imp<'M, 'N, I: Index, E: Conjugate<Canonical: ComplexField>>(
            src: SparseColMatRef<'_, I, E, Dim<'M>, Dim<'N>>,
        ) -> Mat<E::Canonical, Dim<'M>, Dim<'N>> {
            let mut mat = Mat::<E::Canonical, _, _>::zeros(src.nrows(), src.ncols());
            for j in src.ncols().indices() {
                for (i, val) in src.row_indices_of_col(j).zip(
                    crate::utils::slice::SliceGroup::<'_, E>::new(src.values_of_col(j))
                        .into_ref_iter(),
                ) {
                    mat.write(i, j, mat.read(i, j).faer_add(val.read().canonicalize()));
                }
            }

            mat
        }

        make_guard!(M);
        make_guard!(N);
        imp(self.as_shape(self.nrows().bind(M), self.ncols().bind(N)))
            .into_shape(self.nrows(), self.ncols())
    }

    /// Copies `self` into a newly allocated matrix, with row-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output.
    #[inline]
    pub fn to_row_major(&self) -> Result<SparseRowMat<I, E::Canonical, R, C>, FaerError>
    where
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        let mut col_ptr = try_zeroed::<I>(self.nrows().unbound() + 1)?;
        let nnz = self.compute_nnz();
        let mut row_ind = try_zeroed::<I>(nnz)?;
        let mut values = VecGroup::<E::Canonical>::new();
        values
            .try_reserve_exact(nnz)
            .map_err(|_| FaerError::OutOfMemory)?;
        values.resize(nnz, E::Canonical::faer_zero().faer_into_units());

        let mut mem = GlobalPodBuffer::try_new(StackReq::new::<I>(self.nrows().unbound()))
            .map_err(|_| FaerError::OutOfMemory)?;

        let (this, conj) = self.canonicalize();

        if conj == Conj::No {
            utils::transpose(
                &mut col_ptr,
                &mut row_ind,
                values.as_slice_mut().into_inner(),
                this,
                PodStack::new(&mut mem),
            );
        } else {
            utils::adjoint(
                &mut col_ptr,
                &mut row_ind,
                values.as_slice_mut().into_inner(),
                this,
                PodStack::new(&mut mem),
            );
        }

        let mut row_ind = core::mem::ManuallyDrop::new(row_ind);
        let capacity = row_ind.capacity();
        let length = row_ind.len();
        let ptr = row_ind.as_mut_ptr() as _;

        let transpose = unsafe {
            SparseColMat::new(
                SymbolicSparseColMat::new_unchecked(
                    self.ncols(),
                    self.nrows(),
                    col_ptr,
                    None,
                    alloc::vec::Vec::from_raw_parts(ptr, length, capacity),
                ),
                values.into_inner(),
            )
        };

        Ok(transpose.into_transpose())
    }

    /// Returns a view over the transpose of `self` in row-major format.
    #[inline]
    pub fn transpose(self) -> SparseRowMatRef<'a, I, E, C, R> {
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
    pub fn conjugate(self) -> SparseColMatRef<'a, I, E::Conj, R, C>
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

    /// Returns a view over the canonical view of `self`, along with whether it has been conjugated
    /// or not.
    #[inline]
    pub fn canonicalize(self) -> (SparseColMatRef<'a, I, E::Canonical, R, C>, Conj)
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
    pub fn adjoint(self) -> SparseRowMatRef<'a, I, E::Conj, C, R>
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
    pub fn values_of_col(self, j: Idx<C>) -> GroupFor<E, &'a [E::Unit]> {
        unsafe {
            self.values
                .subslice_unchecked(self.col_range(j))
                .into_inner()
        }
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseColMatRef<'a, I, R, C> {
        self.symbolic
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn parts(
        self,
    ) -> (
        SymbolicSparseColMatRef<'a, I, R, C>,
        GroupFor<E, &'a [E::Unit]>,
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
    pub fn row_indices_of_col_raw(&self, j: Idx<C>) -> &'a [Idx<R, I>] {
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
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = Idx<R>> {
        self.symbolic.row_indices_of_col(j)
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
    /// doesn't contain it, or contains multiple indices with the given index.
    ///
    /// # Panics
    /// Panics if `row >= self.nrows()`.  
    /// Panics if `col >= self.ncols()`.  
    #[track_caller]
    pub fn get(self, row: Idx<R>, col: Idx<C>) -> Option<GroupFor<E, &'a E::Unit>> {
        let values = self.get_all(row, col);
        if E::faer_first(E::faer_as_ref(&values)).len() == 1 {
            Some(E::faer_map(values, |slice| &slice[0]))
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
    pub fn get_all(self, row: Idx<R>, col: Idx<C>) -> GroupFor<E, &'a [E::Unit]> {
        assert!(row < self.nrows());
        assert!(col < self.ncols());

        let row = I::truncate(row.unbound());
        let start = self
            .row_indices_of_col_raw(col)
            .partition_point(|&p| p.unbound() < row);
        let end = start
            + self.row_indices_of_col_raw(col)[start..].partition_point(|&p| p.unbound() <= row);

        E::faer_map(self.values_of_col(col), |slice| &slice[start..end])
    }

    /// Returns the input matrix with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape<V: Shape, H: Shape>(
        self,
        nrows: V,
        ncols: H,
    ) -> SparseColMatRef<'a, I, E, V, H> {
        SparseColMatRef {
            symbolic: self.symbolic.as_shape(nrows, ncols),
            values: self.values,
        }
    }

    /// Returns the input matrix with dynamic shape.
    #[inline]
    pub fn as_dyn(self) -> SparseColMatRef<'a, I, E> {
        SparseColMatRef {
            symbolic: self.symbolic.as_dyn(),
            values: self.values,
        }
    }
}

impl<I: Index, E: Entity, R: Shape, C: Shape> core::fmt::Debug for SparseColMatRef<'_, I, E, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use crate::utils::bound::*;
        use generativity::make_guard;

        fn imp<I: Index, E: Entity>(
            mat: SparseColMatRef<'_, I, E, Dim<'_>, Dim<'_>>,
            f: &mut core::fmt::Formatter<'_>,
        ) -> core::fmt::Result {
            let mut iter = mat.ncols().indices().flat_map(move |j| {
                struct Wrapper<E: Entity>(usize, usize, E);
                impl<E: Entity> core::fmt::Debug for Wrapper<E> {
                    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                        let row = self.0;
                        let col = self.1;
                        let val = self.2;
                        write!(f, "({row}, {col}, {val:?})")
                    }
                }

                mat.row_indices_of_col(j)
                    .zip(SliceGroup::<'_, E>::new(mat.values_of_col(j)).into_ref_iter())
                    .map(move |(i, val)| Wrapper(i.unbound(), j.unbound(), val.read()))
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
