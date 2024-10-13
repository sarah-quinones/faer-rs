use crate::{internal_prelude::*, Idx, IdxInc, TryReserveError};
use core::ops::{Index, IndexMut};
use faer_traits::{RealValue, Unit};

pub struct Col<C: Container, T, Rows: Shape = usize> {
    column: Mat<C, T, Rows, usize>,
}

#[inline]
fn idx_to_pair<T, R>(f: impl FnMut(T) -> R) -> impl FnMut(T, usize) -> R {
    let mut f = f;
    #[inline(always)]
    move |i, _| f(i)
}

impl<C: Container, T, Rows: Shape> Col<C, T, Rows> {
    pub fn from_fn(nrows: Rows, f: impl FnMut(Idx<Rows>) -> C::Of<T>) -> Self {
        Self {
            column: Mat::from_fn(nrows, 1, idx_to_pair(f)),
        }
    }

    #[inline]
    pub fn zeros_with_ctx(ctx: &Ctx<C, T>, nrows: Rows) -> Self
    where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        Self {
            column: Mat::zeros_with_ctx(ctx, nrows, 1),
        }
    }

    pub fn try_reserve(&mut self, new_row_capacity: usize) -> Result<(), TryReserveError> {
        self.column.try_reserve(new_row_capacity, 1)
    }

    #[track_caller]
    pub fn reserve(&mut self, new_row_capacity: usize) {
        self.column.reserve(new_row_capacity, 1)
    }

    pub fn resize_with(&mut self, new_nrows: Rows, f: impl FnMut(Idx<Rows>) -> C::Of<T>) {
        self.column.resize_with(new_nrows, 1, idx_to_pair(f));
    }
    pub fn truncate(&mut self, new_nrows: Rows) {
        self.column.truncate(new_nrows, 1);
    }

    #[inline]
    pub fn into_row_shape<V: Shape>(self, nrows: V) -> Col<C, T, V> {
        Col {
            column: self.column.into_shape(nrows, 1),
        }
    }

    #[inline]
    pub fn into_diagonal(self) -> Diag<C, T, Rows> {
        Diag { inner: self }
    }
}

impl<C: Container, T, Rows: Shape> Col<C, T, Rows> {
    #[inline]
    pub fn nrows(&self) -> Rows {
        self.column.nrows()
    }
    #[inline]
    pub fn ncols(&self) -> usize {
        self.column.ncols()
    }

    #[inline]
    pub fn as_ref(&self) -> ColRef<'_, C, T, Rows> {
        self.column.as_ref().col(0)
    }

    #[inline]
    pub fn as_mut(&mut self) -> ColMut<'_, C, T, Rows> {
        self.column.as_mut().col_mut(0)
    }

    #[inline]
    pub fn norm_max_with(&self, ctx: &Ctx<C::Canonical, T::Canonical>) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical>>,
    {
        linalg::reductions::norm_max::norm_max(
            ctx,
            self.as_ref()
                .canonical()
                .as_dyn_stride()
                .as_dyn_rows()
                .as_mat(),
        )
    }

    #[inline]
    pub fn norm_max(&self) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical, MathCtx: Default>>,
    {
        self.norm_max_with(&default())
    }

    #[inline]
    pub fn norm_l2_with(&self, ctx: &Ctx<C::Canonical, T::Canonical>) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical>>,
    {
        linalg::reductions::norm_l2::norm_l2(
            ctx,
            self.as_ref()
                .canonical()
                .as_dyn_stride()
                .as_dyn_rows()
                .as_mat(),
        )
    }

    #[inline]
    pub fn norm_l2(&self) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical, MathCtx: Default>>,
    {
        self.norm_l2_with(&default())
    }
}

impl<C: Container, T: core::fmt::Debug, Rows: Shape> core::fmt::Debug for Col<C, T, Rows> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, Rows: Shape> Index<Idx<Rows>> for Col<Unit, T, Rows> {
    type Output = T;

    #[inline]
    fn index(&self, row: Idx<Rows>) -> &Self::Output {
        self.as_ref().at(row)
    }
}

impl<T, Rows: Shape> IndexMut<Idx<Rows>> for Col<Unit, T, Rows> {
    #[inline]
    fn index_mut(&mut self, row: Idx<Rows>) -> &mut Self::Output {
        self.as_mut().at_mut(row)
    }
}

impl<C: Container, T, Rows: Shape> Col<C, T, Rows> {
    #[inline(always)]
    pub fn as_ptr(&self) -> C::Of<*const T> {
        self.as_ref().as_ptr()
    }

    #[inline(always)]
    pub fn shape(&self) -> (Rows, usize) {
        (self.nrows(), self.ncols())
    }

    #[inline(always)]
    pub fn row_stride(&self) -> isize {
        1
    }

    #[inline(always)]
    pub fn ptr_at(&self, row: IdxInc<Rows>) -> C::Of<*const T> {
        self.as_ref().ptr_at(row)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>) -> C::Of<*const T> {
        self.as_ref().ptr_inbounds_at(row)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_row(
        &self,
        row: IdxInc<Rows>,
    ) -> (ColRef<'_, C, T, usize>, ColRef<'_, C, T, usize>) {
        self.as_ref().split_at_row(row)
    }

    #[inline(always)]
    pub fn transpose(&self) -> RowRef<'_, C, T, Rows> {
        self.as_ref().transpose()
    }

    #[inline(always)]
    pub fn conjugate(&self) -> ColRef<'_, C::Conj, T::Conj, Rows>
    where
        T: ConjUnit,
    {
        self.as_ref().conjugate()
    }

    #[inline(always)]
    pub fn canonical(&self) -> ColRef<'_, C::Canonical, T::Canonical, Rows>
    where
        T: ConjUnit,
    {
        self.as_ref().canonical()
    }

    #[inline(always)]
    pub fn adjoint(&self) -> RowRef<'_, C::Conj, T::Conj, Rows>
    where
        T: ConjUnit,
    {
        self.as_ref().adjoint()
    }

    #[inline(always)]
    pub fn at(&self, row: Idx<Rows>) -> C::Of<&'_ T> {
        self.as_ref().at(row)
    }

    #[inline(always)]
    pub unsafe fn at_unchecked(&self, row: Idx<Rows>) -> C::Of<&'_ T> {
        self.as_ref().at_unchecked(row)
    }

    #[inline]
    pub fn reverse_rows(&self) -> ColRef<'_, C, T, Rows> {
        self.as_ref().reverse_rows()
    }

    #[inline]
    pub fn subrows<V: Shape>(&self, row_start: IdxInc<Rows>, nrows: V) -> ColRef<'_, C, T, V> {
        self.as_ref().subrows(row_start, nrows)
    }

    #[inline]
    pub fn subrows_range(
        &self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
    ) -> ColRef<'_, C, T, usize> {
        self.as_ref().subrows_range(rows)
    }

    #[inline]
    pub fn as_row_shape<V: Shape>(&self, nrows: V) -> ColRef<'_, C, T, V> {
        self.as_ref().as_row_shape(nrows)
    }

    #[inline]
    pub fn as_dyn_rows(&self) -> ColRef<'_, C, T, usize> {
        self.as_ref().as_dyn_rows()
    }

    #[inline]
    pub fn as_dyn_stride(&self) -> ColRef<'_, C, T, Rows, isize> {
        self.as_ref().as_dyn_stride()
    }

    #[inline]
    pub fn iter(&self) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = C::Of<&'_ T>> {
        self.as_ref().iter()
    }

    #[inline]
    #[cfg(feature = "rayon")]
    pub fn par_iter(
        &self,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = C::OfSend<&'_ T>>
    where
        T: Sync,
    {
        self.as_ref().par_iter()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_partition(
        &self,
        count: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColRef<'_, C, T, usize>>
    where
        T: Sync,
    {
        self.as_ref().par_partition(count)
    }

    #[inline]
    pub fn try_as_col_major(&self) -> Option<ColRef<'_, C, T, Rows, ContiguousFwd>> {
        self.as_ref().try_as_col_major()
    }

    #[inline]
    pub fn try_as_col_major_mut(&self) -> Option<ColMut<'_, C, T, Rows, ContiguousFwd>> {
        self.as_ref()
            .try_as_col_major()
            .map(|x| unsafe { x.const_cast() })
    }

    #[inline]
    pub fn as_mat(&self) -> MatRef<'_, C, T, Rows, usize, isize> {
        self.as_ref().as_mat()
    }
    #[inline]
    pub fn as_mat_mut(&self) -> MatMut<'_, C, T, Rows, usize, isize> {
        unsafe { self.as_ref().as_mat().const_cast() }
    }

    #[inline]
    pub fn as_diagonal(&self) -> DiagRef<'_, C, T, Rows> {
        DiagRef {
            inner: self.as_ref(),
        }
    }
}

impl<C: Container, T, Rows: Shape> Col<C, T, Rows> {
    #[inline(always)]
    pub fn as_ptr_mut(&mut self) -> C::Of<*mut T> {
        self.as_mut().as_ptr_mut()
    }

    #[inline(always)]
    pub fn ptr_at_mut(&mut self, row: IdxInc<Rows>) -> C::Of<*mut T> {
        self.as_mut().ptr_at_mut(row)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(&mut self, row: Idx<Rows>) -> C::Of<*mut T> {
        self.as_mut().ptr_inbounds_at_mut(row)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_row_mut(
        &mut self,
        row: IdxInc<Rows>,
    ) -> (ColMut<'_, C, T, usize>, ColMut<'_, C, T, usize>) {
        self.as_mut().split_at_row_mut(row)
    }

    #[inline(always)]
    pub fn transpose_mut(&mut self) -> RowMut<'_, C, T, Rows> {
        self.as_mut().transpose_mut()
    }

    #[inline(always)]
    pub fn conjugate_mut(&mut self) -> ColMut<'_, C::Conj, T::Conj, Rows>
    where
        T: ConjUnit,
    {
        self.as_mut().conjugate_mut()
    }

    #[inline(always)]
    pub fn canonical_mut(&mut self) -> ColMut<'_, C::Canonical, T::Canonical, Rows>
    where
        T: ConjUnit,
    {
        self.as_mut().canonical_mut()
    }

    #[inline(always)]
    pub fn adjoint_mut(&mut self) -> RowMut<'_, C::Conj, T::Conj, Rows>
    where
        T: ConjUnit,
    {
        self.as_mut().adjoint_mut()
    }

    #[inline(always)]
    pub fn at_mut(&mut self, row: Idx<Rows>) -> C::Of<&'_ mut T> {
        self.as_mut().at_mut(row)
    }

    #[inline(always)]
    pub unsafe fn at_mut_unchecked(&mut self, row: Idx<Rows>) -> C::Of<&'_ mut T> {
        self.as_mut().at_mut_unchecked(row)
    }

    #[inline]
    pub fn reverse_rows_mut(&mut self) -> ColMut<'_, C, T, Rows> {
        self.as_mut().reverse_rows_mut()
    }

    #[inline]
    pub fn subrows_mut<V: Shape>(
        &mut self,
        row_start: IdxInc<Rows>,
        nrows: V,
    ) -> ColMut<'_, C, T, V> {
        self.as_mut().subrows_mut(row_start, nrows)
    }

    #[inline]
    pub fn subrows_range_mut(
        &mut self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
    ) -> ColMut<'_, C, T, usize> {
        self.as_mut().subrows_range_mut(rows)
    }

    #[inline]
    pub fn as_row_shape_mut<V: Shape>(&mut self, nrows: V) -> ColMut<'_, C, T, V> {
        self.as_mut().as_row_shape_mut(nrows)
    }

    #[inline]
    pub fn as_dyn_rows_mut(&mut self) -> ColMut<'_, C, T, usize> {
        self.as_mut().as_dyn_rows_mut()
    }

    #[inline]
    pub fn as_dyn_stride_mut(&mut self) -> ColMut<'_, C, T, Rows, isize> {
        self.as_mut().as_dyn_stride_mut()
    }

    #[inline]
    pub fn iter_mut(
        &mut self,
    ) -> impl '_ + ExactSizeIterator + DoubleEndedIterator<Item = C::Of<&'_ mut T>> {
        self.as_mut().iter_mut()
    }

    #[inline]
    #[cfg(feature = "rayon")]
    pub fn par_iter_mut(
        &mut self,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = C::OfSend<&'_ mut T>>
    where
        T: Send,
    {
        self.as_mut().par_iter_mut()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_partition_mut(
        &mut self,
        count: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColMut<'_, C, T, usize>>
    where
        T: Send,
    {
        self.as_mut().par_partition_mut(count)
    }

    #[inline]
    pub fn as_diagonal_mut(&mut self) -> DiagMut<'_, C, T, Rows> {
        self.as_mut().as_diagonal_mut()
    }
}
