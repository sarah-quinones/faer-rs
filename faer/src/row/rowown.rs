use crate::{internal_prelude::*, ContiguousFwd, Idx, IdxInc, TryReserveError};
use core::ops::{Index, IndexMut};
use faer_traits::{RealValue, Unit};

pub struct Row<C: Container, T, Cols: Shape = usize> {
    trans: Col<C, T, Cols>,
}

impl<C: Container, T, Cols: Shape> Row<C, T, Cols> {
    #[inline]
    pub fn from_fn(nrows: Cols, f: impl FnMut(Idx<Cols>) -> C::Of<T>) -> Self {
        Self {
            trans: Col::from_fn(nrows, f),
        }
    }

    #[inline]
    pub fn zeros_with(ctx: &Ctx<C, T>, ncols: Cols) -> Self
    where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        Self {
            trans: Col::zeros_with(ctx, ncols),
        }
    }

    #[inline]
    pub fn try_reserve(&mut self, new_row_capacity: usize) -> Result<(), TryReserveError> {
        self.trans.try_reserve(new_row_capacity)
    }

    #[track_caller]
    pub fn reserve(&mut self, new_row_capacity: usize) {
        self.trans.reserve(new_row_capacity)
    }

    #[inline]
    pub fn resize_with(&mut self, new_nrows: Cols, f: impl FnMut(Idx<Cols>) -> C::Of<T>) {
        self.trans.resize_with(new_nrows, f);
    }
    #[inline]
    pub fn truncate(&mut self, new_nrows: Cols) {
        self.trans.truncate(new_nrows);
    }

    #[inline]
    pub fn into_col_shape<V: Shape>(self, nrows: V) -> Row<C, T, V> {
        Row {
            trans: self.trans.into_row_shape(nrows),
        }
    }

    #[inline]
    pub fn into_diagonal(self) -> Diag<C, T, Cols> {
        Diag { inner: self.trans }
    }
}

impl<C: Container, T, Cols: Shape> Row<C, T, Cols> {
    #[inline]
    pub fn nrows(&self) -> usize {
        self.trans.ncols()
    }
    #[inline]
    pub fn ncols(&self) -> Cols {
        self.trans.nrows()
    }

    #[inline]
    pub fn as_ref(&self) -> RowRef<'_, C, T, Cols> {
        self.trans.as_ref().transpose()
    }

    #[inline]
    pub fn as_mut(&mut self) -> RowMut<'_, C, T, Cols> {
        self.trans.as_mut().transpose_mut()
    }

    #[inline]
    pub fn norm_max_with(&self, ctx: &Ctx<C::Canonical, T::Canonical>) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical>>,
    {
        self.as_ref().transpose().norm_max_with(ctx)
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
        self.as_ref().transpose().norm_l2_with(ctx)
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

impl<C: Container, T: core::fmt::Debug, Cols: Shape> core::fmt::Debug for Row<C, T, Cols> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, Cols: Shape> Index<Idx<Cols>> for Row<Unit, T, Cols> {
    type Output = T;

    #[inline]
    fn index(&self, col: Idx<Cols>) -> &Self::Output {
        self.as_ref().at(col)
    }
}

impl<T, Cols: Shape> IndexMut<Idx<Cols>> for Row<Unit, T, Cols> {
    #[inline]
    fn index_mut(&mut self, col: Idx<Cols>) -> &mut Self::Output {
        self.as_mut().at_mut(col)
    }
}

impl<C: Container, T, Cols: Shape> Row<C, T, Cols> {
    #[inline(always)]
    pub fn as_ptr(&self) -> C::Of<*const T> {
        self.as_ref().as_ptr()
    }

    #[inline(always)]
    pub fn shape(&self) -> (usize, Cols) {
        self.as_ref().shape()
    }

    #[inline(always)]
    pub fn col_stride(&self) -> isize {
        self.as_ref().col_stride()
    }

    #[inline(always)]
    pub fn ptr_at(&self, col: IdxInc<Cols>) -> C::Of<*const T> {
        self.as_ref().ptr_at(col)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, col: Idx<Cols>) -> C::Of<*const T> {
        self.as_ref().ptr_inbounds_at(col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col(
        &self,
        col: IdxInc<Cols>,
    ) -> (RowRef<'_, C, T, usize>, RowRef<'_, C, T, usize>) {
        self.as_ref().split_at_col(col)
    }

    #[inline(always)]
    pub fn transpose(&self) -> ColRef<'_, C, T, Cols> {
        self.as_ref().transpose()
    }

    #[inline(always)]
    pub fn conjugate(&self) -> RowRef<'_, C::Conj, T::Conj, Cols>
    where
        T: ConjUnit,
    {
        self.as_ref().conjugate()
    }

    #[inline(always)]
    pub fn canonical(&self) -> RowRef<'_, C::Canonical, T::Canonical, Cols>
    where
        T: ConjUnit,
    {
        self.as_ref().canonical()
    }

    #[inline(always)]
    pub fn adjoint(&self) -> ColRef<'_, C::Conj, T::Conj, Cols>
    where
        T: ConjUnit,
    {
        self.as_ref().adjoint()
    }

    #[inline(always)]
    pub fn at(&self, col: Idx<Cols>) -> C::Of<&'_ T> {
        self.as_ref().at(col)
    }

    #[inline(always)]
    pub unsafe fn at_unchecked(&self, col: Idx<Cols>) -> C::Of<&'_ T> {
        self.as_ref().at_unchecked(col)
    }

    #[inline]
    pub fn reverse_cols(&self) -> RowRef<'_, C, T, Cols> {
        self.as_ref().reverse_cols()
    }

    #[inline]
    pub fn subcols<V: Shape>(&self, col_start: IdxInc<Cols>, ncols: V) -> RowRef<'_, C, T, V> {
        self.as_ref().subcols(col_start, ncols)
    }

    #[inline]
    pub fn subcols_range(
        &self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> RowRef<'_, C, T, usize> {
        self.as_ref().subcols_range(cols)
    }

    #[inline]
    pub fn as_col_shape<V: Shape>(&self, ncols: V) -> RowRef<'_, C, T, V> {
        self.as_ref().as_col_shape(ncols)
    }

    #[inline]
    pub fn as_dyn_cols(&self) -> RowRef<'_, C, T, usize> {
        self.as_ref().as_dyn_cols()
    }

    #[inline]
    pub fn as_dyn_stride(&self) -> RowRef<'_, C, T, Cols, isize> {
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
    pub fn try_as_row_major(&self) -> Option<RowRef<'_, C, T, Cols, ContiguousFwd>> {
        self.as_ref().try_as_row_major()
    }

    #[inline]
    pub fn as_diagonal(&self) -> DiagRef<'_, C, T, Cols> {
        self.as_ref().as_diagonal()
    }

    #[inline(always)]
    pub unsafe fn const_cast(&self) -> RowMut<'_, C, T, Cols> {
        self.as_ref().const_cast()
    }

    #[inline]
    pub fn as_mat(&self) -> MatRef<'_, C, T, usize, Cols, isize> {
        self.as_ref().as_mat()
    }
    #[inline]
    pub fn as_mat_mut(&mut self) -> MatMut<'_, C, T, usize, Cols, isize> {
        self.as_mut().as_mat_mut()
    }
}

impl<C: Container, T, Cols: Shape> Row<C, T, Cols> {
    #[inline(always)]
    pub fn as_ptr_mut(&mut self) -> C::Of<*mut T> {
        self.as_mut().as_ptr_mut()
    }

    #[inline(always)]
    pub fn ptr_at_mut(&mut self, col: IdxInc<Cols>) -> C::Of<*mut T> {
        self.as_mut().ptr_at_mut(col)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(&mut self, col: Idx<Cols>) -> C::Of<*mut T> {
        self.as_mut().ptr_inbounds_at_mut(col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col_mut(
        &mut self,
        col: IdxInc<Cols>,
    ) -> (RowMut<'_, C, T, usize>, RowMut<'_, C, T, usize>) {
        self.as_mut().split_at_col_mut(col)
    }

    #[inline(always)]
    pub fn transpose_mut(&mut self) -> ColMut<'_, C, T, Cols> {
        self.as_mut().transpose_mut()
    }

    #[inline(always)]
    pub fn conjugate_mut(&mut self) -> RowMut<'_, C::Conj, T::Conj, Cols>
    where
        T: ConjUnit,
    {
        self.as_mut().conjugate_mut()
    }

    #[inline(always)]
    pub fn canonical_mut(&mut self) -> RowMut<'_, C::Canonical, T::Canonical, Cols>
    where
        T: ConjUnit,
    {
        self.as_mut().canonical_mut()
    }

    #[inline(always)]
    pub fn adjoint_mut(&mut self) -> ColMut<'_, C::Conj, T::Conj, Cols>
    where
        T: ConjUnit,
    {
        self.as_mut().adjoint_mut()
    }

    #[inline(always)]
    pub fn at_mut(&mut self, col: Idx<Cols>) -> C::Of<&'_ mut T> {
        self.as_mut().at_mut(col)
    }

    #[inline(always)]
    pub unsafe fn at_mut_unchecked(&mut self, col: Idx<Cols>) -> C::Of<&'_ mut T> {
        self.as_mut().at_mut_unchecked(col)
    }

    #[inline]
    pub fn reverse_cols_mut(&mut self) -> RowMut<'_, C, T, Cols> {
        self.as_mut().reverse_cols_mut()
    }

    #[inline]
    pub fn subcols_mut<V: Shape>(
        &mut self,
        col_start: IdxInc<Cols>,
        ncols: V,
    ) -> RowMut<'_, C, T, V> {
        self.as_mut().subcols_mut(col_start, ncols)
    }

    #[inline]
    pub fn subcols_range_mut(
        &mut self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> RowMut<'_, C, T, usize> {
        self.as_mut().subcols_range_mut(cols)
    }

    #[inline]
    pub fn as_col_shape_mut<V: Shape>(&mut self, ncols: V) -> RowMut<'_, C, T, V> {
        self.as_mut().as_col_shape_mut(ncols)
    }

    #[inline]
    pub fn as_dyn_cols_mut(&mut self) -> RowMut<'_, C, T, usize> {
        self.as_mut().as_dyn_cols_mut()
    }

    #[inline]
    pub fn as_dyn_stride_mut(&mut self) -> RowMut<'_, C, T, Cols, isize> {
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
    pub fn copy_from_with<RhsC: Container<Canonical = C>, RhsT: ConjUnit<Canonical = T>>(
        &mut self,
        ctx: &Ctx<C, T>,
        other: impl AsRowRef<RhsC, RhsT, Cols>,
    ) where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        self.as_mut().copy_from_with(ctx, other)
    }

    #[inline]
    pub fn try_as_row_major_mut(&mut self) -> Option<RowMut<'_, C, T, Cols, ContiguousFwd>> {
        self.as_mut().try_as_row_major_mut()
    }

    #[inline]
    pub fn as_diagonal_mut(&mut self) -> DiagMut<'_, C, T, Cols> {
        self.as_mut().as_diagonal_mut()
    }
}
