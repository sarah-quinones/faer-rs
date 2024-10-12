use super::*;
use crate::{
    internal_prelude::*,
    unzipped,
    utils::bound::{Dim, Partition},
    zipped, Conj, ContiguousFwd, Idx, IdxInc,
};
use core::ops::{Index, IndexMut};
use equator::assert;
use faer_traits::{ComplexField, Ctx, RealValue};
use generativity::{make_guard, Guard};
use matref::MatRef;

pub struct MatMut<'a, C: Container, T, Rows = usize, Cols = usize, RStride = isize, CStride = isize>
{
    pub(super) imp: MatView<C, T, Rows, Cols, RStride, CStride>,
    pub(super) __marker: PhantomData<(&'a mut T, &'a Rows, &'a Cols)>,
}

#[repr(transparent)]
pub(crate) struct SyncCell<T>(T);
unsafe impl<T> Sync for SyncCell<T> {}

impl<'short, C: Container, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Reborrow<'short>
    for MatMut<'_, C, T, Rows, Cols, RStride, CStride>
{
    type Target = MatRef<'short, C, T, Rows, Cols, RStride, CStride>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        MatRef {
            imp: self.imp,
            __marker: PhantomData,
        }
    }
}
impl<'short, C: Container, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy>
    ReborrowMut<'short> for MatMut<'_, C, T, Rows, Cols, RStride, CStride>
{
    type Target = MatMut<'short, C, T, Rows, Cols, RStride, CStride>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        MatMut {
            imp: self.imp,
            __marker: PhantomData,
        }
    }
}
impl<'a, C: Container, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> IntoConst
    for MatMut<'a, C, T, Rows, Cols, RStride, CStride>
{
    type Target = MatRef<'a, C, T, Rows, Cols, RStride, CStride>;
    #[inline]
    fn into_const(self) -> Self::Target {
        MatRef {
            imp: self.imp,
            __marker: PhantomData,
        }
    }
}

unsafe impl<C: Container, T: Sync, Rows: Sync, Cols: Sync, RStride: Sync, CStride: Sync> Sync
    for MatMut<'_, C, T, Rows, Cols, RStride, CStride>
{
}
unsafe impl<C: Container, T: Send, Rows: Send, Cols: Send, RStride: Send, CStride: Send> Send
    for MatMut<'_, C, T, Rows, Cols, RStride, CStride>
{
}

impl<'a, C: Container, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride>
    MatMut<'a, C, T, Rows, Cols, RStride, CStride>
{
    #[inline(always)]
    #[track_caller]
    pub unsafe fn from_raw_parts_mut(
        ptr: C::Of<*mut T>,
        nrows: Rows,
        ncols: Cols,
        row_stride: RStride,
        col_stride: CStride,
    ) -> Self {
        help!(C);
        Self {
            imp: MatView {
                ptr: core::mem::transmute_copy::<C::Of<NonNull<T>>, C::OfCopy<NonNull<T>>>(&map!(
                    ptr,
                    ptr,
                    NonNull::new_unchecked(ptr)
                )),
                nrows,
                ncols,
                row_stride,
                col_stride,
            },
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> C::Of<*const T> {
        help!(C);
        map!(
            unsafe {
                core::mem::transmute_copy::<C::OfCopy<NonNull<T>>, C::Of<NonNull<T>>>(&self.imp.ptr)
            },
            ptr,
            ptr.as_ptr() as *const T
        )
    }

    #[inline(always)]
    pub fn nrows(&self) -> Rows {
        self.imp.nrows
    }

    #[inline(always)]
    pub fn ncols(&self) -> Cols {
        self.imp.ncols
    }

    #[inline(always)]
    pub fn shape(&self) -> (Rows, Cols) {
        (self.nrows(), self.ncols())
    }

    #[inline(always)]
    pub fn row_stride(&self) -> RStride {
        self.imp.row_stride
    }

    #[inline(always)]
    pub fn col_stride(&self) -> CStride {
        self.imp.col_stride
    }

    #[inline(always)]
    pub fn ptr_at(&self, row: IdxInc<Rows>, col: IdxInc<Cols>) -> C::Of<*const T> {
        self.rb().ptr_at(row, col)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<*const T> {
        self.rb().ptr_inbounds_at(row, col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at(
        self,
        row: IdxInc<Rows>,
        col: IdxInc<Cols>,
    ) -> (
        MatRef<'a, C, T, usize, usize, RStride, CStride>,
        MatRef<'a, C, T, usize, usize, RStride, CStride>,
        MatRef<'a, C, T, usize, usize, RStride, CStride>,
        MatRef<'a, C, T, usize, usize, RStride, CStride>,
    ) {
        self.into_const().split_at(row, col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_row(
        self,
        row: IdxInc<Rows>,
    ) -> (
        MatRef<'a, C, T, usize, Cols, RStride, CStride>,
        MatRef<'a, C, T, usize, Cols, RStride, CStride>,
    ) {
        self.into_const().split_at_row(row)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col(
        self,
        col: IdxInc<Cols>,
    ) -> (
        MatRef<'a, C, T, Rows, usize, RStride, CStride>,
        MatRef<'a, C, T, Rows, usize, RStride, CStride>,
    ) {
        self.into_const().split_at_col(col)
    }

    #[inline(always)]
    pub fn transpose(self) -> MatRef<'a, C, T, Cols, Rows, CStride, RStride> {
        MatRef {
            imp: MatView {
                ptr: self.imp.ptr,
                nrows: self.imp.ncols,
                ncols: self.imp.nrows,
                row_stride: self.imp.col_stride,
                col_stride: self.imp.row_stride,
            },
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn conjugate(self) -> MatRef<'a, C::Conj, T::Conj, Rows, Cols, RStride, CStride>
    where
        T: ConjUnit,
    {
        self.into_const().conjugate()
    }

    #[inline(always)]
    pub fn canonical(self) -> MatRef<'a, C::Canonical, T::Canonical, Rows, Cols, RStride, CStride>
    where
        T: ConjUnit,
    {
        self.into_const().canonical()
    }

    #[inline(always)]
    pub fn adjoint(self) -> MatRef<'a, C::Conj, T::Conj, Cols, Rows, CStride, RStride>
    where
        T: ConjUnit,
    {
        self.into_const().adjoint()
    }

    #[inline(always)]
    pub fn at(self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<&'a T> {
        self.into_const().at(row, col)
    }

    #[inline(always)]
    pub unsafe fn at_unchecked(self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<&'a T> {
        self.into_const().at_unchecked(row, col)
    }

    #[inline]
    pub fn reverse_rows(self) -> MatRef<'a, C, T, Rows, Cols, RStride::Rev, CStride> {
        self.into_const().reverse_rows()
    }

    #[inline]
    pub fn reverse_cols(self) -> MatRef<'a, C, T, Rows, Cols, RStride, CStride::Rev> {
        self.into_const().reverse_cols()
    }

    #[inline]
    pub fn reverse_rows_and_cols(self) -> MatRef<'a, C, T, Rows, Cols, RStride::Rev, CStride::Rev> {
        self.into_const().reverse_rows_and_cols()
    }

    #[inline]
    pub fn submatrix<V: Shape, H: Shape>(
        self,
        row_start: IdxInc<Rows>,
        col_start: IdxInc<Cols>,
        nrows: V,
        ncols: H,
    ) -> MatRef<'a, C, T, V, H, RStride, CStride> {
        self.into_const()
            .submatrix(row_start, col_start, nrows, ncols)
    }

    #[inline]
    pub fn subrows<V: Shape>(
        self,
        row_start: IdxInc<Rows>,
        nrows: V,
    ) -> MatRef<'a, C, T, V, Cols, RStride, CStride> {
        self.into_const().subrows(row_start, nrows)
    }

    #[inline]
    pub fn subcols<H: Shape>(
        self,
        col_start: IdxInc<Cols>,
        ncols: H,
    ) -> MatRef<'a, C, T, Rows, H, RStride, CStride> {
        self.into_const().subcols(col_start, ncols)
    }

    #[inline]
    pub fn submatrix_range(
        self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> MatRef<'a, C, T, usize, usize, RStride, CStride> {
        self.into_const().submatrix_range(rows, cols)
    }

    #[inline]
    pub fn subrows_range(
        self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
    ) -> MatRef<'a, C, T, usize, Cols, RStride, CStride> {
        self.into_const().subrows_range(rows)
    }

    #[inline]
    pub fn subcols_range(
        self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> MatRef<'a, C, T, Rows, usize, RStride, CStride> {
        self.into_const().subcols_range(cols)
    }

    #[inline]
    pub fn as_shape<V: Shape, H: Shape>(
        self,
        nrows: V,
        ncols: H,
    ) -> MatRef<'a, C, T, V, H, RStride, CStride> {
        self.into_const().as_shape(nrows, ncols)
    }

    #[inline]
    pub fn as_row_shape<V: Shape>(self, nrows: V) -> MatRef<'a, C, T, V, Cols, RStride, CStride> {
        self.into_const().as_row_shape(nrows)
    }

    #[inline]
    pub fn as_col_shape<H: Shape>(self, ncols: H) -> MatRef<'a, C, T, Rows, H, RStride, CStride> {
        self.into_const().as_col_shape(ncols)
    }

    #[inline]
    pub fn as_dyn_stride(self) -> MatRef<'a, C, T, Rows, Cols, isize, isize> {
        self.into_const().as_dyn_stride()
    }

    #[inline]
    pub fn as_dyn(self) -> MatRef<'a, C, T, usize, usize, RStride, CStride> {
        self.into_const().as_dyn()
    }

    #[inline]
    pub fn as_dyn_rows(self) -> MatRef<'a, C, T, usize, Cols, RStride, CStride> {
        self.into_const().as_dyn_rows()
    }

    #[inline]
    pub fn as_dyn_cols(self) -> MatRef<'a, C, T, Rows, usize, RStride, CStride> {
        self.into_const().as_dyn_cols()
    }

    #[inline]
    pub fn row(self, i: Idx<Rows>) -> RowRef<'a, C, T, Cols, CStride> {
        self.into_const().row(i)
    }

    #[inline]
    #[track_caller]
    pub fn col(self, j: Idx<Cols>) -> ColRef<'a, C, T, Rows, RStride> {
        self.into_const().col(j)
    }

    #[inline]
    pub fn col_iter(
        self,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = ColRef<'a, C, T, Rows, RStride>>
    {
        self.into_const().col_iter()
    }

    #[inline]
    pub fn row_iter(
        self,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = RowRef<'a, C, T, Cols, CStride>>
    {
        self.into_const().row_iter()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_iter(
        self,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColRef<'a, C, T, Rows, RStride>>
    where
        T: Sync,
    {
        self.into_const().par_col_iter()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_iter(
        self,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowRef<'a, C, T, Cols, CStride>>
    where
        T: Sync,
    {
        self.into_const().par_row_iter()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a
           + rayon::iter::IndexedParallelIterator<
        Item = MatRef<'a, C, T, Rows, usize, RStride, CStride>,
    >
    where
        T: Sync,
    {
        self.into_const().par_col_chunks(chunk_size)
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_partition(
        self,
        count: usize,
    ) -> impl 'a
           + rayon::iter::IndexedParallelIterator<
        Item = MatRef<'a, C, T, Rows, usize, RStride, CStride>,
    >
    where
        T: Sync,
    {
        self.into_const().par_col_partition(count)
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a
           + rayon::iter::IndexedParallelIterator<
        Item = MatRef<'a, C, T, usize, Cols, RStride, CStride>,
    >
    where
        T: Sync,
    {
        self.into_const().par_row_chunks(chunk_size)
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_partition(
        self,
        count: usize,
    ) -> impl 'a
           + rayon::iter::IndexedParallelIterator<
        Item = MatRef<'a, C, T, usize, Cols, RStride, CStride>,
    >
    where
        T: Sync,
    {
        self.into_const().par_row_partition(count)
    }

    #[inline]
    pub fn try_as_col_major(self) -> Option<MatRef<'a, C, T, Rows, Cols, ContiguousFwd, CStride>> {
        self.into_const().try_as_col_major()
    }

    #[inline]
    pub fn try_as_row_major(self) -> Option<MatRef<'a, C, T, Rows, Cols, RStride, ContiguousFwd>> {
        self.into_const().try_as_row_major()
    }

    #[inline(always)]
    pub unsafe fn const_cast(self) -> MatMut<'a, C, T, Rows, Cols, RStride, CStride> {
        self
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, C, T, Rows, Cols, RStride, CStride> {
        self.rb()
    }
    #[inline]
    pub fn as_mut(&mut self) -> MatMut<'_, C, T, Rows, Cols, RStride, CStride> {
        self.rb_mut()
    }

    #[inline]
    pub fn bind<'M, 'N>(
        self,
        row: Guard<'M>,
        col: Guard<'N>,
    ) -> MatMut<'a, C, T, Dim<'M>, Dim<'N>, RStride, CStride> {
        unsafe {
            MatMut::from_raw_parts_mut(
                self.as_ptr_mut(),
                self.nrows().bind(row),
                self.ncols().bind(col),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    pub fn bind_r<'M>(self, row: Guard<'M>) -> MatMut<'a, C, T, Dim<'M>, Cols, RStride, CStride> {
        unsafe {
            MatMut::from_raw_parts_mut(
                self.as_ptr_mut(),
                self.nrows().bind(row),
                self.ncols(),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    pub fn bind_c<'N>(self, col: Guard<'N>) -> MatMut<'a, C, T, Rows, Dim<'N>, RStride, CStride> {
        unsafe {
            MatMut::from_raw_parts_mut(
                self.as_ptr_mut(),
                self.nrows(),
                self.ncols().bind(col),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    pub fn norm_max_with(&self, ctx: &Ctx<C::Canonical, T::Canonical>) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical>>,
    {
        self.rb().norm_max_with(ctx)
    }

    #[inline]
    pub fn norm_max(&self) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical, MathCtx: Default>>,
    {
        self.rb().norm_max()
    }
}

impl<'a, C: Container, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride>
    MatMut<'a, C, T, Rows, Cols, RStride, CStride>
{
    #[inline(always)]
    pub fn as_ptr_mut(&self) -> C::Of<*mut T> {
        help!(C);
        map!(
            unsafe {
                core::mem::transmute_copy::<C::OfCopy<NonNull<T>>, C::Of<NonNull<T>>>(&self.imp.ptr)
            },
            ptr,
            ptr.as_ptr() as *mut T
        )
    }

    #[inline(always)]
    pub fn ptr_at_mut(&self, row: IdxInc<Rows>, col: IdxInc<Cols>) -> C::Of<*mut T> {
        help!(C);
        map!(self.rb().ptr_at(row, col), ptr, ptr as *mut T)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(&self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<*mut T> {
        help!(C);
        map!(self.rb().ptr_inbounds_at(row, col), ptr, ptr as *mut T)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_mut(
        self,
        row: IdxInc<Rows>,
        col: IdxInc<Cols>,
    ) -> (
        MatMut<'a, C, T, usize, usize, RStride, CStride>,
        MatMut<'a, C, T, usize, usize, RStride, CStride>,
        MatMut<'a, C, T, usize, usize, RStride, CStride>,
        MatMut<'a, C, T, usize, usize, RStride, CStride>,
    ) {
        let (a, b, c, d) = self.into_const().split_at(row, col);
        unsafe {
            (
                a.const_cast(),
                b.const_cast(),
                c.const_cast(),
                d.const_cast(),
            )
        }
    }

    #[inline]
    #[track_caller]
    pub fn split_at_row_mut(
        self,
        row: IdxInc<Rows>,
    ) -> (
        MatMut<'a, C, T, usize, Cols, RStride, CStride>,
        MatMut<'a, C, T, usize, Cols, RStride, CStride>,
    ) {
        let (a, b) = self.into_const().split_at_row(row);
        unsafe { (a.const_cast(), b.const_cast()) }
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col_mut(
        self,
        col: IdxInc<Cols>,
    ) -> (
        MatMut<'a, C, T, Rows, usize, RStride, CStride>,
        MatMut<'a, C, T, Rows, usize, RStride, CStride>,
    ) {
        let (a, b) = self.into_const().split_at_col(col);
        unsafe { (a.const_cast(), b.const_cast()) }
    }

    #[inline(always)]
    pub fn transpose_mut(self) -> MatMut<'a, C, T, Cols, Rows, CStride, RStride> {
        MatMut {
            imp: MatView {
                ptr: self.imp.ptr,
                nrows: self.imp.ncols,
                ncols: self.imp.nrows,
                row_stride: self.imp.col_stride,
                col_stride: self.imp.row_stride,
            },
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn conjugate_mut(self) -> MatMut<'a, C::Conj, T::Conj, Rows, Cols, RStride, CStride>
    where
        T: ConjUnit,
    {
        unsafe { self.into_const().conjugate().const_cast() }
    }

    #[inline(always)]
    pub fn canonical_mut(
        self,
    ) -> MatMut<'a, C::Canonical, T::Canonical, Rows, Cols, RStride, CStride>
    where
        T: ConjUnit,
    {
        unsafe { self.into_const().canonical().const_cast() }
    }

    #[inline(always)]
    pub fn adjoint_mut(self) -> MatMut<'a, C::Conj, T::Conj, Cols, Rows, CStride, RStride>
    where
        T: ConjUnit,
    {
        unsafe { self.into_const().adjoint().const_cast() }
    }

    #[inline(always)]
    #[track_caller]
    pub fn at_mut(self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<&'a mut T> {
        assert!(all(row < self.nrows(), col < self.ncols()));
        unsafe { self.at_mut_unchecked(row, col) }
    }

    #[inline(always)]
    pub unsafe fn at_mut_unchecked(self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<&'a mut T> {
        help!(C);
        map!(self.ptr_inbounds_at_mut(row, col), ptr, &mut *ptr)
    }

    #[inline]
    pub fn reverse_rows_mut(self) -> MatMut<'a, C, T, Rows, Cols, RStride::Rev, CStride> {
        unsafe { self.into_const().reverse_rows().const_cast() }
    }

    #[inline]
    pub fn reverse_cols_mut(self) -> MatMut<'a, C, T, Rows, Cols, RStride, CStride::Rev> {
        unsafe { self.into_const().reverse_cols().const_cast() }
    }

    #[inline]
    pub fn reverse_rows_and_cols_mut(
        self,
    ) -> MatMut<'a, C, T, Rows, Cols, RStride::Rev, CStride::Rev> {
        unsafe { self.into_const().reverse_rows_and_cols().const_cast() }
    }

    #[inline]
    pub fn submatrix_mut<V: Shape, H: Shape>(
        self,
        row_start: IdxInc<Rows>,
        col_start: IdxInc<Cols>,
        nrows: V,
        ncols: H,
    ) -> MatMut<'a, C, T, V, H, RStride, CStride> {
        unsafe {
            self.into_const()
                .submatrix(row_start, col_start, nrows, ncols)
                .const_cast()
        }
    }

    #[inline]
    pub fn subrows_mut<V: Shape>(
        self,
        row_start: IdxInc<Rows>,
        nrows: V,
    ) -> MatMut<'a, C, T, V, Cols, RStride, CStride> {
        unsafe { self.into_const().subrows(row_start, nrows).const_cast() }
    }

    #[inline]
    pub fn subcols_mut<H: Shape>(
        self,
        col_start: IdxInc<Cols>,
        ncols: H,
    ) -> MatMut<'a, C, T, Rows, H, RStride, CStride> {
        unsafe { self.into_const().subcols(col_start, ncols).const_cast() }
    }

    #[inline]
    pub fn submatrix_range_mut(
        self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> MatMut<'a, C, T, usize, usize, RStride, CStride> {
        unsafe { self.into_const().submatrix_range(rows, cols).const_cast() }
    }

    #[inline]
    pub fn subrows_range_mut(
        self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
    ) -> MatMut<'a, C, T, usize, Cols, RStride, CStride> {
        unsafe { self.into_const().subrows_range(rows).const_cast() }
    }

    #[inline]
    pub fn subcols_range_mut(
        self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> MatMut<'a, C, T, Rows, usize, RStride, CStride> {
        unsafe { self.into_const().subcols_range(cols).const_cast() }
    }

    #[inline]
    #[track_caller]
    pub fn as_shape_mut<V: Shape, H: Shape>(
        self,
        nrows: V,
        ncols: H,
    ) -> MatMut<'a, C, T, V, H, RStride, CStride> {
        unsafe { self.into_const().as_shape(nrows, ncols).const_cast() }
    }

    #[inline]
    pub fn as_row_shape_mut<V: Shape>(
        self,
        nrows: V,
    ) -> MatMut<'a, C, T, V, Cols, RStride, CStride> {
        unsafe { self.into_const().as_row_shape(nrows).const_cast() }
    }

    #[inline]
    pub fn as_col_shape_mut<H: Shape>(
        self,
        ncols: H,
    ) -> MatMut<'a, C, T, Rows, H, RStride, CStride> {
        unsafe { self.into_const().as_col_shape(ncols).const_cast() }
    }

    #[inline]
    pub fn as_dyn_stride_mut(self) -> MatMut<'a, C, T, Rows, Cols, isize, isize> {
        unsafe { self.into_const().as_dyn_stride().const_cast() }
    }

    #[inline]
    pub fn as_dyn_mut(self) -> MatMut<'a, C, T, usize, usize, RStride, CStride> {
        unsafe { self.into_const().as_dyn().const_cast() }
    }

    #[inline]
    pub fn as_dyn_rows_mut(self) -> MatMut<'a, C, T, usize, Cols, RStride, CStride> {
        unsafe { self.into_const().as_dyn_rows().const_cast() }
    }

    #[inline]
    pub fn as_dyn_cols_mut(self) -> MatMut<'a, C, T, Rows, usize, RStride, CStride> {
        unsafe { self.into_const().as_dyn_cols().const_cast() }
    }

    #[inline]
    pub fn row_mut(self, i: Idx<Rows>) -> RowMut<'a, C, T, Cols, CStride> {
        unsafe { self.into_const().row(i).const_cast() }
    }

    #[inline]
    pub fn col_mut(self, j: Idx<Cols>) -> ColMut<'a, C, T, Rows, RStride> {
        unsafe { self.into_const().col(j).const_cast() }
    }

    #[inline]
    pub fn col_iter_mut(
        self,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = ColMut<'a, C, T, Rows, RStride>>
    {
        self.into_const()
            .col_iter()
            .map(|x| unsafe { x.const_cast() })
    }

    #[inline]
    pub fn row_iter_mut(
        self,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = RowMut<'a, C, T, Cols, CStride>>
    {
        self.into_const()
            .row_iter()
            .map(|x| unsafe { x.const_cast() })
    }

    pub(crate) unsafe fn as_type<U>(self) -> MatMut<'a, C, U, Rows, Cols, RStride, CStride> {
        help!(C);
        MatMut::from_raw_parts_mut(
            map!(
                self.as_ptr_mut(),
                ptr,
                core::mem::transmute_copy::<*mut T, *mut U>(&ptr)
            ),
            self.nrows(),
            self.ncols(),
            self.row_stride(),
            self.col_stride(),
        )
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_iter_mut(
        self,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColMut<'a, C, T, Rows, RStride>>
    where
        T: Send,
    {
        use rayon::prelude::*;
        unsafe {
            self.as_type::<SyncCell<T>>()
                .into_const()
                .par_col_iter()
                .map(|x| x.const_cast())
                .map(|x| x.as_type())
        }
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_iter_mut(
        self,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowMut<'a, C, T, Cols, CStride>>
    where
        T: Send,
    {
        use rayon::prelude::*;
        unsafe {
            self.as_type::<SyncCell<T>>()
                .into_const()
                .par_row_iter()
                .map(|x| x.const_cast())
                .map(|x| x.as_type())
        }
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_chunks_mut(
        self,
        chunk_size: usize,
    ) -> impl 'a
           + rayon::iter::IndexedParallelIterator<
        Item = MatMut<'a, C, T, Rows, usize, RStride, CStride>,
    >
    where
        T: Send,
    {
        use rayon::prelude::*;
        unsafe {
            self.as_type::<SyncCell<T>>()
                .into_const()
                .par_col_chunks(chunk_size)
                .map(|x| x.const_cast())
                .map(|x| x.as_type())
        }
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_col_partition_mut(
        self,
        count: usize,
    ) -> impl 'a
           + rayon::iter::IndexedParallelIterator<
        Item = MatMut<'a, C, T, Rows, usize, RStride, CStride>,
    >
    where
        T: Send,
    {
        use rayon::prelude::*;
        unsafe {
            self.as_type::<SyncCell<T>>()
                .into_const()
                .par_col_partition(count)
                .map(|x| x.const_cast())
                .map(|x| x.as_type())
        }
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_chunks_mut(
        self,
        chunk_size: usize,
    ) -> impl 'a
           + rayon::iter::IndexedParallelIterator<
        Item = MatMut<'a, C, T, usize, Cols, RStride, CStride>,
    >
    where
        T: Send,
    {
        use rayon::prelude::*;
        unsafe {
            self.as_type::<SyncCell<T>>()
                .into_const()
                .par_row_chunks(chunk_size)
                .map(|x| x.const_cast())
                .map(|x| x.as_type())
        }
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_row_partition_mut(
        self,
        count: usize,
    ) -> impl 'a
           + rayon::iter::IndexedParallelIterator<
        Item = MatMut<'a, C, T, usize, Cols, RStride, CStride>,
    >
    where
        T: Send,
    {
        use rayon::prelude::*;
        unsafe {
            self.as_type::<SyncCell<T>>()
                .into_const()
                .par_row_partition(count)
                .map(|x| x.const_cast())
                .map(|x| x.as_type())
        }
    }

    #[inline]
    pub fn split_first_row_mut(
        self,
    ) -> Option<(
        RowMut<'a, C, T, Cols, CStride>,
        MatMut<'a, C, T, usize, Cols, RStride, CStride>,
    )> {
        if let Some(i0) = self.nrows().idx(0) {
            let (head, tail) = self.split_at_row_mut(Rows::next(i0));
            Some((head.row_mut(0), tail))
        } else {
            None
        }
    }

    #[inline]
    pub fn try_as_col_major_mut(
        self,
    ) -> Option<MatMut<'a, C, T, Rows, Cols, ContiguousFwd, CStride>> {
        self.into_const()
            .try_as_col_major()
            .map(|x| unsafe { x.const_cast() })
    }

    #[inline]
    pub fn try_as_row_major_mut(
        self,
    ) -> Option<MatMut<'a, C, T, Rows, Cols, RStride, ContiguousFwd>> {
        self.into_const()
            .try_as_row_major()
            .map(|x| unsafe { x.const_cast() })
    }

    #[inline]
    #[track_caller]
    pub fn write(&mut self, i: Idx<Rows>, j: Idx<Cols>) -> C::Of<&'_ mut T> {
        self.rb_mut().at_mut(i, j)
    }

    #[inline]
    #[track_caller]
    pub fn two_cols_mut(
        self,
        i0: Idx<Cols>,
        i1: Idx<Cols>,
    ) -> (
        ColMut<'a, C, T, Rows, RStride>,
        ColMut<'a, C, T, Rows, RStride>,
    ) {
        assert!(i0 != i1);
        let this = self.into_const();
        unsafe { (this.col(i0).const_cast(), this.col(i1).const_cast()) }
    }
    #[inline]
    #[track_caller]
    pub fn two_rows_mut(
        self,
        i0: Idx<Rows>,
        i1: Idx<Rows>,
    ) -> (
        RowMut<'a, C, T, Cols, CStride>,
        RowMut<'a, C, T, Cols, CStride>,
    ) {
        assert!(i0 != i1);
        let this = self.into_const();
        unsafe { (this.row(i0).const_cast(), this.row(i1).const_cast()) }
    }

    #[inline]
    pub fn copy_from_triangular_lower_with_ctx<
        RhsC: Container<Canonical = C>,
        RhsT: ConjUnit<Canonical = T>,
    >(
        &mut self,
        ctx: &Ctx<C, T>,
        other: impl AsMatRef<C = RhsC, T = RhsT, Rows = Rows, Cols = Cols>,
    ) where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        let other = other.as_mat_ref();

        assert!(all(
            self.nrows() == other.nrows(),
            self.ncols() == other.ncols(),
        ));
        let (m, n) = self.shape();

        make_guard!(M);
        make_guard!(N);
        let M = m.bind(M);
        let N = n.bind(N);
        let this = self.rb_mut().as_shape_mut(M, N).as_dyn_stride_mut();
        let other = other.as_shape(M, N);
        imp(ctx, this, other.canonical(), Conj::get::<RhsC, RhsT>());

        pub fn imp<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
            ctx: &Ctx<C, T>,
            this: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
            other: MatRef<'_, C, T, Dim<'M>, Dim<'N>>,
            conj: Conj,
        ) {
            help!(C);

            match conj {
                Conj::No => {
                    zipped!(this, other).for_each_triangular_lower(
                        crate::linalg::zip::Diag::Include,
                        |unzipped!(mut dst, src)| write1!(dst, ctx.copy(&src)),
                    );
                }
                Conj::Yes => {
                    zipped!(this, other).for_each_triangular_lower(
                        crate::linalg::zip::Diag::Include,
                        |unzipped!(mut dst, src)| write1!(dst, ctx.conj(&src)),
                    );
                }
            }
        }
    }

    #[inline]
    pub fn copy_from_with_ctx<RhsC: Container<Canonical = C>, RhsT: ConjUnit<Canonical = T>>(
        &mut self,
        ctx: &Ctx<C, T>,
        other: impl AsMatRef<C = RhsC, T = RhsT, Rows = Rows, Cols = Cols>,
    ) where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        let other = other.as_mat_ref();

        assert!(all(
            self.nrows() == other.nrows(),
            self.ncols() == other.ncols(),
        ));
        let (m, n) = self.shape();

        make_guard!(M);
        make_guard!(N);
        let M = m.bind(M);
        let N = n.bind(N);
        let this = self.rb_mut().as_shape_mut(M, N).as_dyn_stride_mut();
        let other = other.as_shape(M, N);
        imp(ctx, this, other.canonical(), Conj::get::<RhsC, RhsT>());

        pub fn imp<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
            ctx: &Ctx<C, T>,
            this: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
            other: MatRef<'_, C, T, Dim<'M>, Dim<'N>>,
            conj: Conj,
        ) {
            help!(C);

            match conj {
                Conj::No => {
                    zipped!(this, other)
                        .for_each(|unzipped!(mut dst, src)| write1!(dst, ctx.copy(&src)));
                }
                Conj::Yes => {
                    zipped!(this, other)
                        .for_each(|unzipped!(mut dst, src)| write1!(dst, ctx.conj(&src)));
                }
            }
        }
    }

    #[inline]
    pub fn copy_from_strict_triangular_lower_with_ctx<
        RhsC: Container<Canonical = C>,
        RhsT: ConjUnit<Canonical = T>,
    >(
        &mut self,
        ctx: &Ctx<C, T>,
        other: impl AsMatRef<C = RhsC, T = RhsT, Rows = Rows, Cols = Cols>,
    ) where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        let other = other.as_mat_ref();

        assert!(all(
            self.nrows() == other.nrows(),
            self.ncols() == other.ncols(),
        ));
        let (m, n) = self.shape();

        make_guard!(M);
        make_guard!(N);
        let M = m.bind(M);
        let N = n.bind(N);
        let this = self.rb_mut().as_shape_mut(M, N).as_dyn_stride_mut();
        let other = other.as_shape(M, N);
        imp(ctx, this, other.canonical(), Conj::get::<RhsC, RhsT>());

        pub fn imp<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
            ctx: &Ctx<C, T>,
            this: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
            other: MatRef<'_, C, T, Dim<'M>, Dim<'N>>,
            conj: Conj,
        ) {
            help!(C);

            match conj {
                Conj::No => {
                    zipped!(this, other).for_each_triangular_lower(
                        crate::linalg::zip::Diag::Skip,
                        |unzipped!(mut dst, src)| write1!(dst, ctx.copy(&src)),
                    );
                }
                Conj::Yes => {
                    zipped!(this, other).for_each_triangular_lower(
                        crate::linalg::zip::Diag::Skip,
                        |unzipped!(mut dst, src)| write1!(dst, ctx.conj(&src)),
                    );
                }
            }
        }
    }

    #[inline]
    pub(crate) fn __at_mut(self, (i, j): (Idx<Rows>, Idx<Cols>)) -> C::Of<&'a mut T> {
        self.at_mut(i, j)
    }
}

impl<'a, C: Container, T, Rows: Shape, Cols: Shape> MatMut<'a, C, T, Rows, Cols> {
    #[inline(always)]
    #[track_caller]
    pub fn from_column_major_slice_mut(slice: C::Of<&'a mut [T]>, nrows: Rows, ncols: Cols) -> Self
    where
        T: Sized,
    {
        help!(C);
        from_slice_assert(nrows.unbound(), ncols.unbound(), slice_len::<C>(rb!(slice)));

        unsafe {
            Self::from_raw_parts_mut(
                map!(slice, slice, slice.as_mut_ptr()),
                nrows,
                ncols,
                1,
                nrows.unbound() as isize,
            )
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn from_column_major_slice_with_stride(
        slice: C::Of<&'a mut [T]>,
        nrows: Rows,
        ncols: Cols,
        col_stride: usize,
    ) -> Self
    where
        T: Sized,
    {
        help!(C);
        from_strided_column_major_slice_mut_assert(
            nrows.unbound(),
            ncols.unbound(),
            col_stride,
            slice_len::<C>(rb!(slice)),
        );

        unsafe {
            Self::from_raw_parts_mut(
                map!(slice, slice, slice.as_mut_ptr()),
                nrows,
                ncols,
                1,
                col_stride as isize,
            )
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn from_row_major_slice(slice: C::Of<&'a mut [T]>, nrows: Rows, ncols: Cols) -> Self
    where
        T: Sized,
    {
        MatMut::from_column_major_slice_mut(slice, ncols, nrows).transpose_mut()
    }

    #[inline(always)]
    #[track_caller]
    pub fn from_row_major_slice_with_stride(
        slice: C::Of<&'a mut [T]>,
        nrows: Rows,
        ncols: Cols,
        row_stride: usize,
    ) -> Self
    where
        T: Sized,
    {
        help!(C);
        from_strided_row_major_slice_mut_assert(
            nrows.unbound(),
            ncols.unbound(),
            row_stride,
            slice_len::<C>(rb!(slice)),
        );

        unsafe {
            Self::from_raw_parts_mut(
                map!(slice, slice, slice.as_mut_ptr()),
                nrows,
                ncols,
                1,
                row_stride as isize,
            )
        }
    }
}

impl<'ROWS, 'COLS, 'a, C: Container, T, RStride: Stride, CStride: Stride>
    MatMut<'a, C, T, Dim<'ROWS>, Dim<'COLS>, RStride, CStride>
{
    #[inline]
    pub fn split_with_mut<'TOP, 'BOT, 'LEFT, 'RIGHT>(
        self,
        row: Partition<'TOP, 'BOT, 'ROWS>,
        col: Partition<'LEFT, 'RIGHT, 'COLS>,
    ) -> (
        MatMut<'a, C, T, Dim<'TOP>, Dim<'LEFT>, RStride, CStride>,
        MatMut<'a, C, T, Dim<'TOP>, Dim<'RIGHT>, RStride, CStride>,
        MatMut<'a, C, T, Dim<'BOT>, Dim<'LEFT>, RStride, CStride>,
        MatMut<'a, C, T, Dim<'BOT>, Dim<'RIGHT>, RStride, CStride>,
    ) {
        let (a, b, c, d) = self.split_at_mut(row.midpoint(), col.midpoint());
        (
            a.as_shape_mut(row.head, col.head),
            b.as_shape_mut(row.head, col.tail),
            c.as_shape_mut(row.tail, col.head),
            d.as_shape_mut(row.tail, col.tail),
        )
    }
}

impl<'ROWS, 'a, C: Container, T, Cols: Shape, RStride: Stride, CStride: Stride>
    MatMut<'a, C, T, Dim<'ROWS>, Cols, RStride, CStride>
{
    #[inline]
    pub fn split_rows_with_mut<'TOP, 'BOT>(
        self,
        row: Partition<'TOP, 'BOT, 'ROWS>,
    ) -> (
        MatMut<'a, C, T, Dim<'TOP>, Cols, RStride, CStride>,
        MatMut<'a, C, T, Dim<'BOT>, Cols, RStride, CStride>,
    ) {
        let (a, b) = self.split_at_row_mut(row.midpoint());
        (a.as_row_shape_mut(row.head), b.as_row_shape_mut(row.tail))
    }
}

impl<'COLS, 'a, C: Container, T, Rows: Shape, RStride: Stride, CStride: Stride>
    MatMut<'a, C, T, Rows, Dim<'COLS>, RStride, CStride>
{
    #[inline]
    pub fn split_cols_with_mut<'LEFT, 'RIGHT>(
        self,
        col: Partition<'LEFT, 'RIGHT, 'COLS>,
    ) -> (
        MatMut<'a, C, T, Rows, Dim<'LEFT>, RStride, CStride>,
        MatMut<'a, C, T, Rows, Dim<'RIGHT>, RStride, CStride>,
    ) {
        let (a, b) = self.split_at_col_mut(col.midpoint());
        (a.as_col_shape_mut(col.head), b.as_col_shape_mut(col.tail))
    }
}

impl<'ROWS, 'COLS, 'a, C: Container, T, RStride: Stride, CStride: Stride>
    MatMut<'a, C, T, Dim<'ROWS>, Dim<'COLS>, RStride, CStride>
{
    #[inline]
    pub fn split_with<'TOP, 'BOT, 'LEFT, 'RIGHT>(
        self,
        row: Partition<'TOP, 'BOT, 'ROWS>,
        col: Partition<'LEFT, 'RIGHT, 'COLS>,
    ) -> (
        MatRef<'a, C, T, Dim<'TOP>, Dim<'LEFT>, RStride, CStride>,
        MatRef<'a, C, T, Dim<'TOP>, Dim<'RIGHT>, RStride, CStride>,
        MatRef<'a, C, T, Dim<'BOT>, Dim<'LEFT>, RStride, CStride>,
        MatRef<'a, C, T, Dim<'BOT>, Dim<'RIGHT>, RStride, CStride>,
    ) {
        let (a, b, c, d) = self.split_at(row.midpoint(), col.midpoint());
        (
            a.as_shape(row.head, col.head),
            b.as_shape(row.head, col.tail),
            c.as_shape(row.tail, col.head),
            d.as_shape(row.tail, col.tail),
        )
    }
}

impl<'ROWS, 'a, C: Container, T, Cols: Shape, RStride: Stride, CStride: Stride>
    MatMut<'a, C, T, Dim<'ROWS>, Cols, RStride, CStride>
{
    #[inline]
    pub fn split_rows_with<'TOP, 'BOT>(
        self,
        row: Partition<'TOP, 'BOT, 'ROWS>,
    ) -> (
        MatRef<'a, C, T, Dim<'TOP>, Cols, RStride, CStride>,
        MatRef<'a, C, T, Dim<'BOT>, Cols, RStride, CStride>,
    ) {
        let (a, b) = self.split_at_row(row.midpoint());
        (a.as_row_shape(row.head), b.as_row_shape(row.tail))
    }
}

impl<'COLS, 'a, C: Container, T, Rows: Shape, RStride: Stride, CStride: Stride>
    MatMut<'a, C, T, Rows, Dim<'COLS>, RStride, CStride>
{
    #[inline]
    pub fn split_cols_with<'LEFT, 'RIGHT>(
        self,
        col: Partition<'LEFT, 'RIGHT, 'COLS>,
    ) -> (
        MatRef<'a, C, T, Rows, Dim<'LEFT>, RStride, CStride>,
        MatRef<'a, C, T, Rows, Dim<'RIGHT>, RStride, CStride>,
    ) {
        let (a, b) = self.split_at_col(col.midpoint());
        (a.as_col_shape(col.head), b.as_col_shape(col.tail))
    }
}

impl<'a, C: Container, T, Dim: Shape, RStride: Stride, CStride: Stride>
    MatMut<'a, C, T, Dim, Dim, RStride, CStride>
{
    #[inline]
    pub fn diagonal(self) -> DiagRef<'a, C, T, Dim, isize> {
        self.into_const().diagonal()
    }
}

impl<'a, C: Container, T, Dim: Shape, RStride: Stride, CStride: Stride>
    MatMut<'a, C, T, Dim, Dim, RStride, CStride>
{
    #[inline]
    pub fn diagonal_mut(self) -> DiagMut<'a, C, T, Dim, isize> {
        unsafe {
            self.into_const()
                .diagonal()
                .column_vector()
                .const_cast()
                .as_diagonal_mut()
        }
    }
}

impl<T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> Index<(Idx<Rows>, Idx<Cols>)>
    for MatMut<'_, Unit, T, Rows, Cols, RStride, CStride>
{
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (Idx<Rows>, Idx<Cols>)) -> &Self::Output {
        self.rb().at(row, col)
    }
}

impl<T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IndexMut<(Idx<Rows>, Idx<Cols>)>
    for MatMut<'_, Unit, T, Rows, Cols, RStride, CStride>
{
    #[inline]
    fn index_mut(&mut self, (row, col): (Idx<Rows>, Idx<Cols>)) -> &mut Self::Output {
        self.rb_mut().at_mut(row, col)
    }
}

impl<
        'a,
        C: Container,
        T: core::fmt::Debug,
        Rows: Shape,
        Cols: Shape,
        RStride: Stride,
        CStride: Stride,
    > core::fmt::Debug for MatMut<'a, C, T, Rows, Cols, RStride, CStride>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

#[track_caller]
#[inline]
fn from_strided_column_major_slice_mut_assert(
    nrows: usize,
    ncols: usize,
    col_stride: usize,
    len: usize,
) {
    if nrows > 0 && ncols > 0 {
        // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
        // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
        // we don't care
        let last = usize::checked_mul(col_stride, ncols - 1)
            .and_then(|last_col| last_col.checked_add(nrows - 1));
        let Some(last) = last else {
            panic!("address computation of the last matrix element overflowed");
        };
        assert!(all(col_stride >= nrows, last < len));
    }
}

#[track_caller]
#[inline]
fn from_strided_row_major_slice_mut_assert(
    nrows: usize,
    ncols: usize,
    row_stride: usize,
    len: usize,
) {
    if nrows > 0 && ncols > 0 {
        // we don't have to worry about size == usize::MAX == slice.len(), because the length of a
        // slice can never exceed isize::MAX in bytes, unless the type is zero sized, in which case
        // we don't care
        let last = usize::checked_mul(row_stride, nrows - 1)
            .and_then(|last_row| last_row.checked_add(ncols - 1));
        let Some(last) = last else {
            panic!("address computation of the last matrix element overflowed");
        };
        assert!(all(row_stride >= ncols, last < len));
    }
}

mod bound_range {
    use super::*;
    use crate::utils::bound::{Disjoint, Segment};

    impl<'ROWS, 'a, C: Container, T, Cols: Shape, RStride: Stride, CStride: Stride>
        MatMut<'a, C, T, Dim<'ROWS>, Cols, RStride, CStride>
    {
        #[inline]
        pub fn row_segments_mut<'scope, 'TOP, 'BOT>(
            self,
            first: Segment<'scope, 'ROWS, 'TOP>,
            second: Segment<'scope, 'ROWS, 'BOT>,
            disjoint: Disjoint<'scope, 'TOP, 'BOT>,
        ) -> (
            MatMut<'a, C, T, Dim<'TOP>, Cols, RStride, CStride>,
            MatMut<'a, C, T, Dim<'BOT>, Cols, RStride, CStride>,
        ) {
            unsafe {
                _ = disjoint;
                let first = MatMut::from_raw_parts_mut(
                    self.ptr_at_mut(first.start(), Cols::start()),
                    first.len(),
                    self.ncols(),
                    self.row_stride(),
                    self.col_stride(),
                );
                let second = MatMut::from_raw_parts_mut(
                    self.ptr_at_mut(second.start(), Cols::start()),
                    second.len(),
                    self.ncols(),
                    self.row_stride(),
                    self.col_stride(),
                );
                (first, second)
            }
        }
    }

    impl<'COLS, 'a, C: Container, T, Rows: Shape, RStride: Stride, CStride: Stride>
        MatMut<'a, C, T, Rows, Dim<'COLS>, RStride, CStride>
    {
        #[inline]
        pub fn col_segments_mut<'scope, 'LEFT, 'RIGHT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
            second: Segment<'scope, 'COLS, 'RIGHT>,
            disjoint: Disjoint<'scope, 'LEFT, 'RIGHT>,
        ) -> (
            MatMut<'a, C, T, Rows, Dim<'LEFT>, RStride, CStride>,
            MatMut<'a, C, T, Rows, Dim<'RIGHT>, RStride, CStride>,
        ) {
            unsafe {
                _ = disjoint;
                let first = MatMut::from_raw_parts_mut(
                    self.ptr_at_mut(Rows::start(), first.start()),
                    self.nrows(),
                    first.len(),
                    self.row_stride(),
                    self.col_stride(),
                );
                let second = MatMut::from_raw_parts_mut(
                    self.ptr_at_mut(Rows::start(), second.start()),
                    self.nrows(),
                    second.len(),
                    self.row_stride(),
                    self.col_stride(),
                );
                (first, second)
            }
        }
    }

    impl<'ROWS, 'a, C: Container, T, Cols: Shape, RStride: Stride, CStride: Stride>
        MatRef<'a, C, T, Dim<'ROWS>, Cols, RStride, CStride>
    {
        #[inline]
        pub fn row_segments<'scope, 'TOP, 'BOT>(
            self,
            first: Segment<'scope, 'ROWS, 'TOP>,
            second: Segment<'scope, 'ROWS, 'BOT>,
        ) -> (
            MatRef<'a, C, T, Dim<'TOP>, Cols, RStride, CStride>,
            MatRef<'a, C, T, Dim<'BOT>, Cols, RStride, CStride>,
        ) {
            unsafe {
                let first = MatRef::from_raw_parts(
                    self.ptr_at(first.start(), Cols::start()),
                    first.len(),
                    self.ncols(),
                    self.row_stride(),
                    self.col_stride(),
                );
                let second = MatRef::from_raw_parts(
                    self.ptr_at(second.start(), Cols::start()),
                    second.len(),
                    self.ncols(),
                    self.row_stride(),
                    self.col_stride(),
                );
                (first, second)
            }
        }
    }

    impl<'COLS, 'a, C: Container, T, Rows: Shape, RStride: Stride, CStride: Stride>
        MatRef<'a, C, T, Rows, Dim<'COLS>, RStride, CStride>
    {
        #[inline]
        pub fn col_segments<'scope, 'LEFT, 'RIGHT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
            second: Segment<'scope, 'COLS, 'RIGHT>,
        ) -> (
            MatRef<'a, C, T, Rows, Dim<'LEFT>, RStride, CStride>,
            MatRef<'a, C, T, Rows, Dim<'RIGHT>, RStride, CStride>,
        ) {
            unsafe {
                let first = MatRef::from_raw_parts(
                    self.ptr_at(Rows::start(), first.start()),
                    self.nrows(),
                    first.len(),
                    self.row_stride(),
                    self.col_stride(),
                );
                let second = MatRef::from_raw_parts(
                    self.ptr_at(Rows::start(), second.start()),
                    self.nrows(),
                    second.len(),
                    self.row_stride(),
                    self.col_stride(),
                );
                (first, second)
            }
        }
    }

    impl<'ROWS, 'a, C: Container, T, Cols: Shape, RStride: Stride, CStride: Stride>
        MatMut<'a, C, T, Dim<'ROWS>, Cols, RStride, CStride>
    {
        #[inline]
        pub fn row_segments<'scope, 'TOP, 'BOT>(
            self,
            first: Segment<'scope, 'ROWS, 'TOP>,
            second: Segment<'scope, 'ROWS, 'BOT>,
        ) -> (
            MatRef<'a, C, T, Dim<'TOP>, Cols, RStride, CStride>,
            MatRef<'a, C, T, Dim<'BOT>, Cols, RStride, CStride>,
        ) {
            unsafe {
                let first = MatRef::from_raw_parts(
                    self.ptr_at(first.start(), Cols::start()),
                    first.len(),
                    self.ncols(),
                    self.row_stride(),
                    self.col_stride(),
                );
                let second = MatRef::from_raw_parts(
                    self.ptr_at(second.start(), Cols::start()),
                    second.len(),
                    self.ncols(),
                    self.row_stride(),
                    self.col_stride(),
                );
                (first, second)
            }
        }
    }

    impl<'COLS, 'a, C: Container, T, Rows: Shape, RStride: Stride, CStride: Stride>
        MatMut<'a, C, T, Rows, Dim<'COLS>, RStride, CStride>
    {
        #[inline]
        pub fn col_segments<'scope, 'LEFT, 'RIGHT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
            second: Segment<'scope, 'COLS, 'RIGHT>,
        ) -> (
            MatRef<'a, C, T, Rows, Dim<'LEFT>, RStride, CStride>,
            MatRef<'a, C, T, Rows, Dim<'RIGHT>, RStride, CStride>,
        ) {
            unsafe {
                let first = MatRef::from_raw_parts(
                    self.ptr_at(Rows::start(), first.start()),
                    self.nrows(),
                    first.len(),
                    self.row_stride(),
                    self.col_stride(),
                );
                let second = MatRef::from_raw_parts(
                    self.ptr_at(Rows::start(), second.start()),
                    self.nrows(),
                    second.len(),
                    self.row_stride(),
                    self.col_stride(),
                );
                (first, second)
            }
        }
    }

    // single segment

    impl<'ROWS, 'a, C: Container, T, Cols: Shape, RStride: Stride, CStride: Stride>
        MatMut<'a, C, T, Dim<'ROWS>, Cols, RStride, CStride>
    {
        #[inline]
        pub fn row_segment_mut<'scope, 'TOP>(
            self,
            first: Segment<'scope, 'ROWS, 'TOP>,
        ) -> MatMut<'a, C, T, Dim<'TOP>, Cols, RStride, CStride> {
            unsafe {
                MatMut::from_raw_parts_mut(
                    self.ptr_at_mut(first.start(), Cols::start()),
                    first.len(),
                    self.ncols(),
                    self.row_stride(),
                    self.col_stride(),
                )
            }
        }
    }

    impl<'COLS, 'a, C: Container, T, Rows: Shape, RStride: Stride, CStride: Stride>
        MatMut<'a, C, T, Rows, Dim<'COLS>, RStride, CStride>
    {
        #[inline]
        pub fn col_segment_mut<'scope, 'LEFT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
        ) -> MatMut<'a, C, T, Rows, Dim<'LEFT>, RStride, CStride> {
            unsafe {
                MatMut::from_raw_parts_mut(
                    self.ptr_at_mut(Rows::start(), first.start()),
                    self.nrows(),
                    first.len(),
                    self.row_stride(),
                    self.col_stride(),
                )
            }
        }
    }

    impl<'ROWS, 'a, C: Container, T, Cols: Shape, RStride: Stride, CStride: Stride>
        MatRef<'a, C, T, Dim<'ROWS>, Cols, RStride, CStride>
    {
        #[inline]
        pub fn row_segment<'scope, 'TOP>(
            self,
            first: Segment<'scope, 'ROWS, 'TOP>,
        ) -> MatRef<'a, C, T, Dim<'TOP>, Cols, RStride, CStride> {
            unsafe {
                MatRef::from_raw_parts(
                    self.ptr_at(first.start(), Cols::start()),
                    first.len(),
                    self.ncols(),
                    self.row_stride(),
                    self.col_stride(),
                )
            }
        }
    }

    impl<'COLS, 'a, C: Container, T, Rows: Shape, RStride: Stride, CStride: Stride>
        MatRef<'a, C, T, Rows, Dim<'COLS>, RStride, CStride>
    {
        #[inline]
        pub fn col_segment<'scope, 'LEFT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
        ) -> MatRef<'a, C, T, Rows, Dim<'LEFT>, RStride, CStride> {
            unsafe {
                MatRef::from_raw_parts(
                    self.ptr_at(Rows::start(), first.start()),
                    self.nrows(),
                    first.len(),
                    self.row_stride(),
                    self.col_stride(),
                )
            }
        }
    }

    impl<'ROWS, 'a, C: Container, T, Cols: Shape, RStride: Stride, CStride: Stride>
        MatMut<'a, C, T, Dim<'ROWS>, Cols, RStride, CStride>
    {
        #[inline]
        pub fn row_segment<'scope, 'TOP>(
            self,
            first: Segment<'scope, 'ROWS, 'TOP>,
        ) -> MatRef<'a, C, T, Dim<'TOP>, Cols, RStride, CStride> {
            unsafe {
                MatRef::from_raw_parts(
                    self.ptr_at(first.start(), Cols::start()),
                    first.len(),
                    self.ncols(),
                    self.row_stride(),
                    self.col_stride(),
                )
            }
        }
    }

    impl<'COLS, 'a, C: Container, T, Rows: Shape, RStride: Stride, CStride: Stride>
        MatMut<'a, C, T, Rows, Dim<'COLS>, RStride, CStride>
    {
        #[inline]
        pub fn col_segment<'scope, 'LEFT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
        ) -> MatRef<'a, C, T, Rows, Dim<'LEFT>, RStride, CStride> {
            unsafe {
                MatRef::from_raw_parts(
                    self.ptr_at(Rows::start(), first.start()),
                    self.nrows(),
                    first.len(),
                    self.row_stride(),
                    self.col_stride(),
                )
            }
        }
    }
}
