use super::*;
use crate::{
    col::colref::ColRef,
    internal_prelude::*,
    row::rowref::RowRef,
    utils::bound::{Dim, Partition},
    ContiguousFwd, Idx, IdxInc,
};
use core::ops::Index;
use equator::{assert, debug_assert};
use faer_traits::{ComplexField, RealValue};
use generativity::Guard;
use matmut::MatMut;
use matown::Mat;

pub struct MatRef<'a, C: Container, T, Rows = usize, Cols = usize, RStride = isize, CStride = isize>
{
    pub(super) imp: MatView<C, T, Rows, Cols, RStride, CStride>,
    pub(super) __marker: PhantomData<(&'a T, &'a Rows, &'a Cols)>,
}

impl<C: Container, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Copy
    for MatRef<'_, C, T, Rows, Cols, RStride, CStride>
{
}
impl<C: Container, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Clone
    for MatRef<'_, C, T, Rows, Cols, RStride, CStride>
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, C: Container, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> Reborrow<'short>
    for MatRef<'_, C, T, Rows, Cols, RStride, CStride>
{
    type Target = MatRef<'short, C, T, Rows, Cols, RStride, CStride>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short, C: Container, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy>
    ReborrowMut<'short> for MatRef<'_, C, T, Rows, Cols, RStride, CStride>
{
    type Target = MatRef<'short, C, T, Rows, Cols, RStride, CStride>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}
impl<'a, C: Container, T, Rows: Copy, Cols: Copy, RStride: Copy, CStride: Copy> IntoConst
    for MatRef<'a, C, T, Rows, Cols, RStride, CStride>
{
    type Target = MatRef<'a, C, T, Rows, Cols, RStride, CStride>;
    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

unsafe impl<C: Container, T: Sync, Rows: Sync, Cols: Sync, RStride: Sync, CStride: Sync> Sync
    for MatRef<'_, C, T, Rows, Cols, RStride, CStride>
{
}
unsafe impl<C: Container, T: Sync, Rows: Send, Cols: Send, RStride: Send, CStride: Send> Send
    for MatRef<'_, C, T, Rows, Cols, RStride, CStride>
{
}

#[track_caller]
#[inline]
fn from_strided_column_major_slice_assert(
    nrows: usize,
    ncols: usize,
    col_stride: usize,
    len: usize,
) {
    if nrows > 0 && ncols > 0 {
        let last = usize::checked_mul(col_stride, ncols - 1)
            .and_then(|last_col| last_col.checked_add(nrows - 1));
        let Some(last) = last else {
            panic!("address computation of the last matrix element overflowed");
        };
        assert!(last < len);
    }
}

impl<'a, C: Container, T, Rows: Shape, Cols: Shape> MatRef<'a, C, T, Rows, Cols> {
    #[inline(always)]
    #[track_caller]
    pub fn from_repeated_ref(value: C::Of<&'a T>, nrows: Rows, ncols: Cols) -> Self
    where
        T: Sized,
    {
        unsafe {
            help!(C);
            MatRef::from_raw_parts(map!(value, ptr, ptr as *const T), nrows, ncols, 0, 0)
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn from_column_major_slice(slice: C::Of<&'a [T]>, nrows: Rows, ncols: Cols) -> Self
    where
        T: Sized,
    {
        help!(C);
        from_slice_assert(
            nrows.unbound(),
            ncols.unbound(),
            slice_len::<C>(copy!(slice)),
        );

        unsafe {
            MatRef::from_raw_parts(
                map!(slice, slice, slice.as_ptr()),
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
        slice: C::Of<&'a [T]>,
        nrows: Rows,
        ncols: Cols,
        col_stride: usize,
    ) -> Self
    where
        T: Sized,
    {
        help!(C);
        from_strided_column_major_slice_assert(
            nrows.unbound(),
            ncols.unbound(),
            col_stride,
            slice_len::<C>(copy!(slice)),
        );

        unsafe {
            MatRef::from_raw_parts(
                map!(slice, slice, slice.as_ptr()),
                nrows,
                ncols,
                1,
                col_stride as isize,
            )
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn from_row_major_slice(slice: C::Of<&'a [T]>, nrows: Rows, ncols: Cols) -> Self
    where
        T: Sized,
    {
        MatRef::from_column_major_slice(slice, ncols, nrows).transpose()
    }

    #[inline(always)]
    #[track_caller]
    pub fn from_row_major_slice_with_stride(
        slice: C::Of<&'a [T]>,
        nrows: Rows,
        ncols: Cols,
        row_stride: usize,
    ) -> Self
    where
        T: Sized,
    {
        MatRef::from_column_major_slice_with_stride(slice, ncols, nrows, row_stride).transpose()
    }
}

impl<'a, C: Container, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride>
    MatRef<'a, C, T, Rows, Cols, RStride, CStride>
{
    #[inline(always)]
    #[track_caller]
    pub unsafe fn from_raw_parts(
        ptr: C::Of<*const T>,
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
                    NonNull::new_unchecked(ptr as *mut T)
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
        help!(C);
        let ptr = self.as_ptr();

        if row >= self.nrows() || col >= self.ncols() {
            ptr
        } else {
            map!(ptr, ptr, {
                ptr.wrapping_offset(row.unbound() as isize * self.row_stride().element_stride())
                    .wrapping_offset(col.unbound() as isize * self.col_stride().element_stride())
            })
        }
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<*const T> {
        help!(C);
        debug_assert!(all(row < self.nrows(), col < self.ncols()));
        map!(self.as_ptr(), ptr, {
            ptr.offset(row.unbound() as isize * self.row_stride().element_stride())
                .offset(col.unbound() as isize * self.col_stride().element_stride())
        })
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
        assert!(all(row <= self.nrows(), col <= self.ncols()));

        let rs = self.row_stride();
        let cs = self.col_stride();

        let top_left = self.ptr_at(Rows::start(), Cols::start());
        let top_right = self.ptr_at(Rows::start(), col);
        let bot_left = self.ptr_at(row, Cols::start());
        let bot_right = self.ptr_at(row, col);

        unsafe {
            (
                MatRef::from_raw_parts(top_left, row.unbound(), col.unbound(), rs, cs),
                MatRef::from_raw_parts(
                    top_right,
                    row.unbound(),
                    self.ncols().unbound() - col.unbound(),
                    rs,
                    cs,
                ),
                MatRef::from_raw_parts(
                    bot_left,
                    self.nrows().unbound() - row.unbound(),
                    col.unbound(),
                    rs,
                    cs,
                ),
                MatRef::from_raw_parts(
                    bot_right,
                    self.nrows().unbound() - row.unbound(),
                    self.ncols().unbound() - col.unbound(),
                    rs,
                    cs,
                ),
            )
        }
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
        assert!(all(row <= self.nrows()));

        let rs = self.row_stride();
        let cs = self.col_stride();

        let top = self.ptr_at(Rows::start(), Cols::start());
        let bot = self.ptr_at(row, Cols::start());

        unsafe {
            (
                MatRef::from_raw_parts(top, row.unbound(), self.ncols(), rs, cs),
                MatRef::from_raw_parts(
                    bot,
                    self.nrows().unbound() - row.unbound(),
                    self.ncols(),
                    rs,
                    cs,
                ),
            )
        }
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
        assert!(all(col <= self.ncols()));

        let rs = self.row_stride();
        let cs = self.col_stride();

        let left = self.ptr_at(Rows::start(), Cols::start());
        let right = self.ptr_at(Rows::start(), col);

        unsafe {
            (
                MatRef::from_raw_parts(left, self.nrows(), col.unbound(), rs, cs),
                MatRef::from_raw_parts(
                    right,
                    self.nrows(),
                    self.ncols().unbound() - col.unbound(),
                    rs,
                    cs,
                ),
            )
        }
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
        help!(C);
        unsafe {
            MatRef::from_raw_parts(
                core::mem::transmute_copy(&map!(
                    self.as_ptr(),
                    ptr,
                    core::mem::transmute_copy::<*const T, *const T::Conj>(&ptr)
                )),
                self.nrows(),
                self.ncols(),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline(always)]
    pub fn canonical(self) -> MatRef<'a, C::Canonical, T::Canonical, Rows, Cols, RStride, CStride>
    where
        T: ConjUnit,
    {
        help!(C);
        unsafe {
            MatRef::from_raw_parts(
                core::mem::transmute_copy(&map!(
                    self.as_ptr(),
                    ptr,
                    core::mem::transmute_copy::<*const T, *const T::Canonical>(&ptr)
                )),
                self.nrows(),
                self.ncols(),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline(always)]
    pub fn adjoint(self) -> MatRef<'a, C::Conj, T::Conj, Cols, Rows, CStride, RStride>
    where
        T: ConjUnit,
    {
        self.conjugate().transpose()
    }

    #[inline(always)]
    pub fn at(self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<&'a T> {
        assert!(all(row < self.nrows(), col < self.ncols()));
        unsafe { self.at_unchecked(row, col) }
    }

    #[inline(always)]
    pub unsafe fn at_unchecked(self, row: Idx<Rows>, col: Idx<Cols>) -> C::Of<&'a T> {
        help!(C);
        map!(self.ptr_inbounds_at(row, col), ptr, &*ptr)
    }

    #[inline]
    pub fn reverse_rows(self) -> MatRef<'a, C, T, Rows, Cols, RStride::Rev, CStride> {
        help!(C);
        let row = unsafe { IdxInc::<Rows>::new_unbound(self.nrows().unbound().saturating_sub(1)) };
        let ptr = self.ptr_at(row, Cols::start());
        unsafe {
            MatRef::from_raw_parts(
                ptr,
                self.nrows(),
                self.ncols(),
                self.row_stride().rev(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    pub fn reverse_cols(self) -> MatRef<'a, C, T, Rows, Cols, RStride, CStride::Rev> {
        help!(C);
        let col = unsafe { IdxInc::<Cols>::new_unbound(self.ncols().unbound().saturating_sub(1)) };
        let ptr = self.ptr_at(Rows::start(), col);
        unsafe {
            MatRef::from_raw_parts(
                ptr,
                self.nrows(),
                self.ncols(),
                self.row_stride(),
                self.col_stride().rev(),
            )
        }
    }

    #[inline]
    pub fn reverse_rows_and_cols(self) -> MatRef<'a, C, T, Rows, Cols, RStride::Rev, CStride::Rev> {
        self.reverse_rows().reverse_cols()
    }

    #[inline]
    pub fn submatrix<V: Shape, H: Shape>(
        self,
        row_start: IdxInc<Rows>,
        col_start: IdxInc<Cols>,
        nrows: V,
        ncols: H,
    ) -> MatRef<'a, C, T, V, H, RStride, CStride> {
        assert!(all(row_start <= self.nrows(), col_start <= self.ncols()));
        {
            let nrows = nrows.unbound();
            let full_nrows = self.nrows().unbound();
            let row_start = row_start.unbound();
            let ncols = ncols.unbound();
            let full_ncols = self.ncols().unbound();
            let col_start = col_start.unbound();
            assert!(all(
                nrows <= full_nrows - row_start,
                ncols <= full_ncols - col_start,
            ));
        }
        let rs = self.row_stride();
        let cs = self.col_stride();

        unsafe { MatRef::from_raw_parts(self.ptr_at(row_start, col_start), nrows, ncols, rs, cs) }
    }

    #[inline]
    pub fn subrows<V: Shape>(
        self,
        row_start: IdxInc<Rows>,
        nrows: V,
    ) -> MatRef<'a, C, T, V, Cols, RStride, CStride> {
        assert!(all(row_start <= self.nrows()));
        {
            let nrows = nrows.unbound();
            let full_nrows = self.nrows().unbound();
            let row_start = row_start.unbound();
            assert!(all(nrows <= full_nrows - row_start));
        }
        let rs = self.row_stride();
        let cs = self.col_stride();

        unsafe {
            MatRef::from_raw_parts(
                self.ptr_at(row_start, Cols::start()),
                nrows,
                self.ncols(),
                rs,
                cs,
            )
        }
    }

    #[inline]
    pub fn subcols<H: Shape>(
        self,
        col_start: IdxInc<Cols>,
        ncols: H,
    ) -> MatRef<'a, C, T, Rows, H, RStride, CStride> {
        assert!(all(col_start <= self.ncols()));
        {
            let ncols = ncols.unbound();
            let full_ncols = self.ncols().unbound();
            let col_start = col_start.unbound();
            assert!(all(ncols <= full_ncols - col_start));
        }
        let rs = self.row_stride();
        let cs = self.col_stride();

        unsafe {
            MatRef::from_raw_parts(
                self.ptr_at(Rows::start(), col_start),
                self.nrows(),
                ncols,
                rs,
                cs,
            )
        }
    }

    #[inline]
    pub fn submatrix_range(
        self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> MatRef<'a, C, T, usize, usize, RStride, CStride> {
        let rows = rows.0.into()..rows.1.into();
        let cols = cols.0.into()..cols.1.into();
        assert!(all(
            rows.start <= self.nrows(),
            cols.start <= self.ncols(),
            rows.end <= self.nrows(),
            cols.end <= self.ncols()
        ));

        let rs = self.row_stride();
        let cs = self.col_stride();

        unsafe {
            MatRef::from_raw_parts(
                self.ptr_at(rows.start, cols.start),
                rows.end.unbound().saturating_sub(rows.start.unbound()),
                cols.end.unbound().saturating_sub(cols.start.unbound()),
                rs,
                cs,
            )
        }
    }

    #[inline]
    #[track_caller]
    pub fn subrows_range(
        self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
    ) -> MatRef<'a, C, T, usize, Cols, RStride, CStride> {
        let rows = rows.0.into()..rows.1.into();

        assert!(all(rows.start <= self.nrows(), rows.end <= self.nrows()));

        let rs = self.row_stride();
        let cs = self.col_stride();

        unsafe {
            MatRef::from_raw_parts(
                self.ptr_at(rows.start, Cols::start()),
                rows.end.unbound().saturating_sub(rows.start.unbound()),
                self.ncols(),
                rs,
                cs,
            )
        }
    }

    #[inline]
    #[track_caller]
    pub fn subcols_range(
        self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> MatRef<'a, C, T, Rows, usize, RStride, CStride> {
        let cols = cols.0.into()..cols.1.into();
        assert!(all(cols.start <= self.ncols(), cols.end <= self.ncols()));

        let rs = self.row_stride();
        let cs = self.col_stride();

        unsafe {
            MatRef::from_raw_parts(
                self.ptr_at(Rows::start(), cols.start),
                self.nrows(),
                cols.end.unbound().saturating_sub(cols.start.unbound()),
                rs,
                cs,
            )
        }
    }

    #[inline]
    #[track_caller]
    pub fn as_shape<V: Shape, H: Shape>(
        self,
        nrows: V,
        ncols: H,
    ) -> MatRef<'a, C, T, V, H, RStride, CStride> {
        assert!(all(
            self.nrows().unbound() == nrows.unbound(),
            self.ncols().unbound() == ncols.unbound(),
        ));
        unsafe {
            MatRef::from_raw_parts(
                self.as_ptr(),
                nrows,
                ncols,
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    pub fn as_row_shape<V: Shape>(self, nrows: V) -> MatRef<'a, C, T, V, Cols, RStride, CStride> {
        assert!(all(self.nrows().unbound() == nrows.unbound()));
        unsafe {
            MatRef::from_raw_parts(
                self.as_ptr(),
                nrows,
                self.ncols(),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    pub fn as_col_shape<H: Shape>(self, ncols: H) -> MatRef<'a, C, T, Rows, H, RStride, CStride> {
        assert!(all(self.ncols().unbound() == ncols.unbound()));
        unsafe {
            MatRef::from_raw_parts(
                self.as_ptr(),
                self.nrows(),
                ncols,
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    pub fn as_dyn_stride(self) -> MatRef<'a, C, T, Rows, Cols, isize, isize> {
        unsafe {
            MatRef::from_raw_parts(
                self.as_ptr(),
                self.nrows(),
                self.ncols(),
                self.row_stride().element_stride(),
                self.col_stride().element_stride(),
            )
        }
    }

    #[inline]
    pub fn as_dyn(self) -> MatRef<'a, C, T, usize, usize, RStride, CStride> {
        unsafe {
            MatRef::from_raw_parts(
                self.as_ptr(),
                self.nrows().unbound(),
                self.ncols().unbound(),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    pub fn as_dyn_rows(self) -> MatRef<'a, C, T, usize, Cols, RStride, CStride> {
        unsafe {
            MatRef::from_raw_parts(
                self.as_ptr(),
                self.nrows().unbound(),
                self.ncols(),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    pub fn as_dyn_cols(self) -> MatRef<'a, C, T, Rows, usize, RStride, CStride> {
        unsafe {
            MatRef::from_raw_parts(
                self.as_ptr(),
                self.nrows(),
                self.ncols().unbound(),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    #[track_caller]
    pub fn row(self, i: Idx<Rows>) -> RowRef<'a, C, T, Cols, CStride> {
        assert!(i < self.nrows());

        unsafe {
            RowRef::from_raw_parts(
                self.ptr_at(i.into(), Cols::start()),
                self.ncols(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    #[track_caller]
    pub fn col(self, j: Idx<Cols>) -> ColRef<'a, C, T, Rows, RStride> {
        assert!(j < self.ncols());

        unsafe {
            ColRef::from_raw_parts(
                self.ptr_at(Rows::start(), j.into()),
                self.nrows(),
                self.row_stride(),
            )
        }
    }

    #[inline]
    pub fn col_iter(
        self,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = ColRef<'a, C, T, Rows, RStride>>
    {
        Cols::indices(Cols::start(), self.ncols().end()).map(move |j| self.col(j))
    }

    #[inline]
    pub fn row_iter(
        self,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = RowRef<'a, C, T, Cols, CStride>>
    {
        Rows::indices(Rows::start(), self.nrows().end()).map(move |i| self.row(i))
    }

    #[inline]
    #[cfg(feature = "rayon")]
    pub fn par_col_iter(
        self,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColRef<'a, C, T, Rows, RStride>>
    where
        T: Sync,
    {
        use rayon::prelude::*;

        #[inline]
        fn col_fn<C: Container, T, Rows: Shape, RStride: Stride, CStride: Stride>(
            col: MatRef<'_, C, T, Rows, usize, RStride, CStride>,
        ) -> ColRef<'_, C, T, Rows, RStride> {
            col.col(0)
        }

        self.par_col_chunks(1).map(col_fn)
    }

    #[inline]
    #[cfg(feature = "rayon")]
    pub fn par_row_iter(
        self,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowRef<'a, C, T, Cols, CStride>>
    where
        T: Sync,
    {
        use rayon::prelude::*;
        self.transpose().par_col_iter().map(ColRef::transpose)
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
        use rayon::prelude::*;

        let this = self.as_dyn_cols();

        assert!(chunk_size > 0);
        let chunk_count = this.ncols().div_ceil(chunk_size);
        (0..chunk_count).into_par_iter().map(move |chunk_idx| {
            let pos = chunk_size * chunk_idx;
            this.subcols(pos, Ord::min(chunk_size, this.ncols() - pos))
        })
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
        use rayon::prelude::*;

        let this = self.as_dyn_cols();

        assert!(count > 0);
        (0..count).into_par_iter().map(move |chunk_idx| {
            let (start, len) =
                crate::utils::thread::par_split_indices(this.ncols(), chunk_idx, count);
            this.subcols(start, len)
        })
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
        use rayon::prelude::*;
        self.transpose()
            .par_col_chunks(chunk_size)
            .map(MatRef::transpose)
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
        use rayon::prelude::*;
        self.transpose()
            .par_col_partition(count)
            .map(MatRef::transpose)
    }

    #[inline]
    pub fn cloned(self) -> Mat<C, T, Rows, Cols>
    where
        T: Clone,
    {
        fn imp<'M, 'N, C: Container, T: Clone, RStride: Stride, CStride: Stride>(
            this: MatRef<'_, C, T, Dim<'M>, Dim<'N>, RStride, CStride>,
        ) -> Mat<C, T, Dim<'M>, Dim<'N>> {
            help!(C);
            Mat::from_fn(this.nrows(), this.ncols(), |i, j| {
                map!(this.at(i, j), val, val.clone())
            })
        }

        with_dim!(M, self.nrows().unbound());
        with_dim!(N, self.ncols().unbound());
        imp(self.as_shape(M, N)).into_shape(self.nrows(), self.ncols())
    }

    #[inline]
    pub fn to_owned(self) -> Mat<C::Canonical, T::Canonical, Rows, Cols>
    where
        T: ConjUnit<Canonical: ComplexField<C::Canonical, MathCtx: Default>>,
    {
        fn imp<'M, 'N, C: Container, T, RStride: Stride, CStride: Stride>(
            this: MatRef<'_, C, T, Dim<'M>, Dim<'N>, RStride, CStride>,
        ) -> Mat<C::Canonical, T::Canonical, Dim<'M>, Dim<'N>>
        where
            T: ConjUnit<Canonical: ComplexField<C::Canonical, MathCtx: Default>>,
        {
            let ctx = &Ctx::<C::Canonical, T::Canonical>::default();
            Mat::from_fn(this.nrows(), this.ncols(), |i, j| {
                Conj::apply::<C, T>(ctx, this.at(i, j))
            })
        }

        help!(C);
        with_dim!(M, self.nrows().unbound());
        with_dim!(N, self.ncols().unbound());
        imp(self.as_shape(M, N)).into_shape(self.nrows(), self.ncols())
    }

    #[inline(always)]
    pub unsafe fn const_cast(self) -> MatMut<'a, C, T, Rows, Cols, RStride, CStride> {
        help!(C);
        MatMut::from_raw_parts_mut(
            map!(self.as_ptr(), ptr, ptr as *mut T),
            self.nrows(),
            self.ncols(),
            self.row_stride(),
            self.col_stride(),
        )
    }

    #[inline]
    pub fn try_as_col_major(self) -> Option<MatRef<'a, C, T, Rows, Cols, ContiguousFwd, CStride>> {
        if self.row_stride().element_stride() == 1 {
            Some(unsafe {
                MatRef::from_raw_parts(
                    self.as_ptr(),
                    self.nrows(),
                    self.ncols(),
                    ContiguousFwd,
                    self.col_stride(),
                )
            })
        } else {
            None
        }
    }

    #[inline]
    pub fn try_as_row_major(self) -> Option<MatRef<'a, C, T, Rows, Cols, RStride, ContiguousFwd>> {
        if self.col_stride().element_stride() == 1 {
            Some(unsafe {
                MatRef::from_raw_parts(
                    self.as_ptr(),
                    self.nrows(),
                    self.ncols(),
                    self.row_stride(),
                    ContiguousFwd,
                )
            })
        } else {
            None
        }
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, C, T, Rows, Cols, RStride, CStride> {
        *self
    }

    #[inline]
    pub fn bind<'M, 'N>(
        self,
        row: Guard<'M>,
        col: Guard<'N>,
    ) -> MatRef<'a, C, T, Dim<'M>, Dim<'N>, RStride, CStride> {
        unsafe {
            MatRef::from_raw_parts(
                self.as_ptr(),
                self.nrows().bind(row),
                self.ncols().bind(col),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    pub fn bind_r<'M>(self, row: Guard<'M>) -> MatRef<'a, C, T, Dim<'M>, Cols, RStride, CStride> {
        unsafe {
            MatRef::from_raw_parts(
                self.as_ptr(),
                self.nrows().bind(row),
                self.ncols(),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    #[inline]
    pub fn bind_c<'N>(self, col: Guard<'N>) -> MatRef<'a, C, T, Rows, Dim<'N>, RStride, CStride> {
        unsafe {
            MatRef::from_raw_parts(
                self.as_ptr(),
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
        linalg::reductions::norm_max::norm_max(ctx, self.canonical().as_dyn_stride().as_dyn())
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
    pub(crate) fn __at(self, (i, j): (Idx<Rows>, Idx<Cols>)) -> C::Of<&'a T> {
        self.at(i, j)
    }
}

impl<'a, C: Container, T, Dim: Shape, RStride: Stride, CStride: Stride>
    MatRef<'a, C, T, Dim, Dim, RStride, CStride>
{
    #[inline]
    pub fn diagonal(self) -> DiagRef<'a, C, T, Dim, isize> {
        let k = Ord::min(self.nrows(), self.ncols());
        DiagRef {
            inner: unsafe {
                ColRef::from_raw_parts(
                    self.as_ptr(),
                    k,
                    self.row_stride().element_stride() + self.col_stride().element_stride(),
                )
            },
        }
    }
}

impl<'ROWS, 'COLS, 'a, C: Container, T, RStride: Stride, CStride: Stride>
    MatRef<'a, C, T, Dim<'ROWS>, Dim<'COLS>, RStride, CStride>
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
    MatRef<'a, C, T, Dim<'ROWS>, Cols, RStride, CStride>
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
    MatRef<'a, C, T, Rows, Dim<'COLS>, RStride, CStride>
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

impl<T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> Index<(Idx<Rows>, Idx<Cols>)>
    for MatRef<'_, Unit, T, Rows, Cols, RStride, CStride>
{
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (Idx<Rows>, Idx<Cols>)) -> &Self::Output {
        self.at(row, col)
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
    > core::fmt::Debug for MatRef<'a, C, T, Rows, Cols, RStride, CStride>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fn imp<'M, 'N, C: Container, T: core::fmt::Debug>(
            this: MatRef<'_, C, T, Dim<'M>, Dim<'N>>,
            f: &mut core::fmt::Formatter<'_>,
        ) -> core::fmt::Result {
            writeln!(f, "[")?;
            for i in this.nrows().indices() {
                this.row(i).fmt(f)?;
                f.write_str(",\n")?;
            }
            write!(f, "]")
        }

        with_dim!(M, self.nrows().unbound());
        with_dim!(N, self.ncols().unbound());
        imp(self.as_shape(M, N).as_dyn_stride(), f)
    }
}
