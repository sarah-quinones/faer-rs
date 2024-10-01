use super::*;
use crate::assert;

/// Sparse matrix view in column-major format, either compressed or uncompressed.
pub struct SparseRowMatRef<'a, I: Index, E: Entity, R: Shape = usize, C: Shape = usize> {
    pub(crate) symbolic: SymbolicSparseRowMatRef<'a, I, R, C>,
    pub(crate) values: SliceGroup<'a, E>,
}

impl<I: Index, E: Entity, R: Shape, C: Shape> Copy for SparseRowMatRef<'_, I, E, R, C> {}
impl<I: Index, E: Entity, R: Shape, C: Shape> Clone for SparseRowMatRef<'_, I, E, R, C> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, I: Index, E: Entity, R: Shape, C: Shape> Reborrow<'short>
    for SparseRowMatRef<'_, I, E, R, C>
{
    type Target = SparseRowMatRef<'short, I, E, R, C>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, I: Index, E: Entity, R: Shape, C: Shape> ReborrowMut<'short>
    for SparseRowMatRef<'_, I, E, R, C>
{
    type Target = SparseRowMatRef<'short, I, E, R, C>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'a, I: Index, E: Entity, R: Shape, C: Shape> IntoConst for SparseRowMatRef<'a, I, E, R, C> {
    type Target = SparseRowMatRef<'a, I, E, R, C>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'a, I: Index, E: Entity, R: Shape, C: Shape> SparseRowMatRef<'a, I, E, R, C> {
    /// Creates a new sparse matrix view.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is not equal to the length of
    /// `symbolic.col_indices()`.
    #[inline]
    #[track_caller]
    pub fn new(
        symbolic: SymbolicSparseRowMatRef<'a, I, R, C>,
        values: GroupFor<E, &'a [E::Unit]>,
    ) -> Self {
        let values = SliceGroup::new(values);
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
    pub fn to_owned(&self) -> Result<SparseRowMat<I, E::Canonical, R, C>, FaerError>
    where
        E: Conjugate<Canonical: ComplexField>,
    {
        self.transpose()
            .to_owned()
            .map(SparseColMat::into_transpose)
    }

    /// Copies `self` into a newly allocated matrix with sorted indices.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output.
    #[inline]
    pub fn to_sorted(&self) -> Result<SparseRowMat<I, E::Canonical, R, C>, FaerError>
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
        E: Conjugate,
        E::Canonical: ComplexField,
    {
        use crate::utils::bound::*;
        use generativity::make_guard;

        fn imp<'M, 'N, I: Index, E: Conjugate<Canonical: ComplexField>>(
            src: SparseRowMatRef<'_, I, E, Dim<'M>, Dim<'N>>,
        ) -> Mat<E::Canonical, Dim<'M>, Dim<'N>> {
            let mut mat = Mat::<E::Canonical, _, _>::zeros(src.nrows(), src.ncols());
            for i in src.nrows().indices() {
                for (j, val) in src.col_indices_of_row(i).zip(
                    crate::utils::slice::SliceGroup::<'_, E>::new(src.values_of_row(i))
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

    /// Copies `self` into a newly allocated matrix, with column-major order.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output.
    #[inline]
    pub fn to_col_major(&self) -> Result<SparseColMat<I, E::Canonical, R, C>, FaerError>
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
    pub fn transpose(self) -> SparseColMatRef<'a, I, E, C, R> {
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
    pub fn conjugate(self) -> SparseRowMatRef<'a, I, E::Conj, R, C>
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

    /// Returns a view over the canonical view of `self`, along with whether it has been conjugated
    /// or not.
    #[inline]
    pub fn canonicalize(self) -> (SparseRowMatRef<'a, I, E::Canonical, R, C>, Conj)
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
    pub fn adjoint(self) -> SparseColMatRef<'a, I, E::Conj, C, R>
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
    pub fn values_of_row(self, i: Idx<R>) -> GroupFor<E, &'a [E::Unit]> {
        self.values.subslice(self.row_range(i)).into_inner()
    }

    /// Returns the symbolic structure of the matrix.
    #[inline]
    pub fn symbolic(&self) -> SymbolicSparseRowMatRef<'a, I, R, C> {
        self.symbolic
    }

    /// Decomposes the matrix into the symbolic part and the numerical values.
    #[inline]
    pub fn parts(
        self,
    ) -> (
        SymbolicSparseRowMatRef<'a, I, R, C>,
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
    pub fn col_indices(&self) -> &'a [Idx<C, I>] {
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
    /// Panics if `row >= self.nrows()`  
    /// Panics if `col >= self.ncols()`  
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

        let col = I::truncate(col.unbound());
        let start = self
            .col_indices_of_row_raw(row)
            .partition_point(|&p| p.unbound() < col);
        let end = start
            + self.col_indices_of_row_raw(row)[start..].partition_point(|&p| p.unbound() <= col);

        E::faer_map(self.values_of_row(row), |slice| &slice[start..end])
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
            values: self.values,
        }
    }

    /// Returns the input matrix with dynamic shape.
    #[inline]
    pub fn as_dyn(self) -> SparseRowMatRef<'a, I, E> {
        SparseRowMatRef {
            symbolic: self.symbolic.as_dyn(),
            values: self.values,
        }
    }
}

impl<I: Index, E: Entity> core::fmt::Debug for SparseRowMatRef<'_, I, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mat = *self;
        let mut iter = (0..mat.nrows()).flat_map(move |i| {
            struct Wrapper<E>(usize, usize, E);
            impl<E: Entity> core::fmt::Debug for Wrapper<E> {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    let row = self.0;
                    let col = self.1;
                    let val = self.2;
                    write!(f, "({row}, {col}, {val:?})")
                }
            }

            mat.col_indices_of_row(i)
                .zip(SliceGroup::<'_, E>::new(mat.values_of_row(i)).into_ref_iter())
                .map(move |(j, val)| Wrapper(i, j, val.read()))
        });

        f.debug_list().entries(&mut iter).finish()
    }
}
