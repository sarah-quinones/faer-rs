use crate::{
    internal_prelude::*,
    utils::bound::{Array, Dim, Partition},
    ContiguousFwd, Idx, IdxInc,
};
use equator::{assert, debug_assert};

pub struct RowRef<'a, C: Container, T, Cols = usize, CStride = isize> {
    pub(crate) trans: ColRef<'a, C, T, Cols, CStride>,
}

impl<C: Container, T, Rows: Copy, CStride: Copy> Copy for RowRef<'_, C, T, Rows, CStride> {}
impl<C: Container, T, Rows: Copy, CStride: Copy> Clone for RowRef<'_, C, T, Rows, CStride> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, C: Container, T, Rows: Copy, CStride: Copy> Reborrow<'short>
    for RowRef<'_, C, T, Rows, CStride>
{
    type Target = RowRef<'short, C, T, Rows, CStride>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short, C: Container, T, Rows: Copy, CStride: Copy> ReborrowMut<'short>
    for RowRef<'_, C, T, Rows, CStride>
{
    type Target = RowRef<'short, C, T, Rows, CStride>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}
impl<'a, C: Container, T, Rows: Copy, CStride: Copy> IntoConst for RowRef<'a, C, T, Rows, CStride> {
    type Target = RowRef<'a, C, T, Rows, CStride>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

unsafe impl<C: Container, T: Sync, Rows: Sync, CStride: Sync> Sync
    for RowRef<'_, C, T, Rows, CStride>
{
}
unsafe impl<C: Container, T: Sync, Rows: Send, CStride: Send> Send
    for RowRef<'_, C, T, Rows, CStride>
{
}

impl<'a, C: Container, T, Cols: Shape, CStride: Stride> RowRef<'a, C, T, Cols, CStride> {
    #[inline(always)]
    #[track_caller]
    pub unsafe fn from_raw_parts(ptr: C::Of<*const T>, ncols: Cols, col_stride: CStride) -> Self {
        help!(C);
        Self {
            trans: ColRef::from_raw_parts(ptr, ncols, col_stride),
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> C::Of<*const T> {
        self.trans.as_ptr()
    }

    #[inline(always)]
    pub fn nrows(&self) -> usize {
        1
    }

    #[inline(always)]
    pub fn ncols(&self) -> Cols {
        self.trans.nrows()
    }

    #[inline(always)]
    pub fn shape(&self) -> (usize, Cols) {
        (self.nrows(), self.ncols())
    }

    #[inline(always)]
    pub fn col_stride(&self) -> CStride {
        self.trans.row_stride()
    }

    #[inline(always)]
    pub fn ptr_at(&self, col: IdxInc<Cols>) -> C::Of<*const T> {
        self.trans.ptr_at(col)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, col: Idx<Cols>) -> C::Of<*const T> {
        debug_assert!(all(col < self.ncols()));
        self.trans.ptr_inbounds_at(col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col(
        self,
        col: IdxInc<Cols>,
    ) -> (
        RowRef<'a, C, T, usize, CStride>,
        RowRef<'a, C, T, usize, CStride>,
    ) {
        assert!(all(col <= self.ncols()));
        let rs = self.col_stride();

        let top = self.as_ptr();
        let bot = self.ptr_at(col);
        unsafe {
            (
                RowRef::from_raw_parts(top, col.unbound(), rs),
                RowRef::from_raw_parts(bot, self.ncols().unbound() - col.unbound(), rs),
            )
        }
    }

    #[inline(always)]
    pub fn transpose(self) -> ColRef<'a, C, T, Cols, CStride> {
        self.trans
    }

    #[inline(always)]
    pub fn conjugate(self) -> RowRef<'a, C::Conj, T::Conj, Cols, CStride>
    where
        T: ConjUnit,
    {
        RowRef {
            trans: self.trans.conjugate(),
        }
    }

    #[inline(always)]
    pub fn canonical(self) -> RowRef<'a, C::Canonical, T::Canonical, Cols, CStride>
    where
        T: ConjUnit,
    {
        RowRef {
            trans: self.trans.canonical(),
        }
    }

    #[inline(always)]
    pub fn adjoint(self) -> ColRef<'a, C::Conj, T::Conj, Cols, CStride>
    where
        T: ConjUnit,
    {
        self.conjugate().transpose()
    }

    #[inline(always)]
    pub fn at(self, col: Idx<Cols>) -> C::Of<&'a T> {
        assert!(all(col < self.ncols()));
        unsafe { self.at_unchecked(col) }
    }

    #[inline(always)]
    pub unsafe fn at_unchecked(self, col: Idx<Cols>) -> C::Of<&'a T> {
        help!(C);
        map!(self.ptr_inbounds_at(col), ptr, &*ptr)
    }

    #[inline]
    pub fn reverse_cols(self) -> RowRef<'a, C, T, Cols, CStride::Rev> {
        RowRef {
            trans: self.trans.reverse_rows(),
        }
    }

    #[inline]
    pub fn subcols<V: Shape>(
        self,
        col_start: IdxInc<Cols>,
        ncols: V,
    ) -> RowRef<'a, C, T, V, CStride> {
        assert!(all(col_start <= self.ncols()));
        {
            let ncols = ncols.unbound();
            let full_ncols = self.ncols().unbound();
            let col_start = col_start.unbound();
            assert!(all(ncols <= full_ncols - col_start));
        }
        let cs = self.col_stride();
        unsafe { RowRef::from_raw_parts(self.ptr_at(col_start), ncols, cs) }
    }

    #[inline]
    pub fn subcols_range(
        self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> RowRef<'a, C, T, usize, CStride> {
        let cols = cols.0.into()..cols.1.into();
        assert!(all(cols.start <= self.ncols()));

        let cs = self.col_stride();

        unsafe {
            RowRef::from_raw_parts(
                self.ptr_at(cols.start),
                cols.end.unbound().saturating_sub(cols.start.unbound()),
                cs,
            )
        }
    }

    #[inline]
    pub fn as_col_shape<V: Shape>(self, ncols: V) -> RowRef<'a, C, T, V, CStride> {
        assert!(all(self.ncols().unbound() == ncols.unbound()));
        unsafe { RowRef::from_raw_parts(self.as_ptr(), ncols, self.col_stride()) }
    }

    #[inline]
    pub fn as_dyn_cols(self) -> RowRef<'a, C, T, usize, CStride> {
        unsafe { RowRef::from_raw_parts(self.as_ptr(), self.ncols().unbound(), self.col_stride()) }
    }

    #[inline]
    pub fn as_dyn_stride(self) -> RowRef<'a, C, T, Cols, isize> {
        unsafe {
            RowRef::from_raw_parts(
                self.as_ptr(),
                self.ncols(),
                self.col_stride().element_stride(),
            )
        }
    }

    #[inline]
    pub fn iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = C::Of<&'a T>> {
        self.trans.iter()
    }

    #[inline]
    #[cfg(feature = "rayon")]
    pub fn par_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = C::OfSend<&'a T>>
    where
        T: Sync,
    {
        self.trans.par_iter()
    }

    #[inline]
    pub fn try_as_row_major(self) -> Option<RowRef<'a, C, T, Cols, ContiguousFwd>> {
        if self.col_stride().element_stride() == 1 {
            Some(unsafe { RowRef::from_raw_parts(self.as_ptr(), self.ncols(), ContiguousFwd) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub unsafe fn const_cast(self) -> RowMut<'a, C, T, Cols, CStride> {
        help!(C);
        RowMut {
            trans: self.trans.const_cast(),
        }
    }

    #[inline]
    pub fn as_ref(&self) -> RowRef<'_, C, T, Cols, CStride> {
        *self
    }

    #[inline]
    pub fn as_mat(self) -> MatRef<'a, C, T, usize, Cols, isize, CStride> {
        self.transpose().as_mat().transpose()
    }

    #[inline]
    pub fn as_diagonal(self) -> DiagRef<'a, C, T, Cols, CStride> {
        DiagRef { inner: self.trans }
    }

    #[inline]
    pub(crate) fn __at(self, i: Idx<Cols>) -> C::Of<&'a T> {
        self.at(i)
    }
}

impl<'a, C: Container, T, Rows: Shape> RowRef<'a, C, T, Rows, ContiguousFwd> {
    #[inline]
    pub fn as_slice(self) -> C::Of<&'a [T]> {
        self.transpose().as_slice()
    }
}

impl<'a, 'ROWS, C: Container, T> RowRef<'a, C, T, Dim<'ROWS>, ContiguousFwd> {
    #[inline]
    pub fn as_array(self) -> C::Of<&'a Array<'ROWS, T>> {
        self.transpose().as_array()
    }
}

impl<'COLS, 'a, C: Container, T, CStride: Stride> RowRef<'a, C, T, Dim<'COLS>, CStride> {
    #[inline]
    pub fn split_cols_with<'LEFT, 'RIGHT>(
        self,
        col: Partition<'LEFT, 'RIGHT, 'COLS>,
    ) -> (
        RowRef<'a, C, T, Dim<'LEFT>, CStride>,
        RowRef<'a, C, T, Dim<'RIGHT>, CStride>,
    ) {
        let (a, b) = self.split_at_col(col.midpoint());
        (a.as_col_shape(col.head), b.as_col_shape(col.tail))
    }
}

impl<C: Container, T: core::fmt::Debug, Cols: Shape, CStride: Stride> core::fmt::Debug
    for RowRef<'_, C, T, Cols, CStride>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fn imp<C: Container, T: core::fmt::Debug>(
            f: &mut core::fmt::Formatter<'_>,
            this: RowRef<'_, C, T, Dim<'_>>,
        ) -> core::fmt::Result {
            if const { C::IS_COMPLEX } {
                help!(C);
                if const { C::IS_CANONICAL } {
                    f.debug_list()
                        .entries(this.ncols().indices().map(|j| unsafe {
                            crate::hacks::coerce::<
                                C::Of<&dyn core::fmt::Debug>,
                                num_complex::Complex<
                                    <C::Real as Container>::OfDebug<&dyn core::fmt::Debug>,
                                >,
                            >(map!(
                                this.at(j),
                                dbg,
                                crate::hacks::hijack_debug(dbg)
                            ))
                        }))
                        .finish()
                } else {
                    f.debug_list()
                        .entries(this.ncols().indices().map(|j| unsafe {
                            crate::hacks::coerce::<
                                C::Of<&dyn core::fmt::Debug>,
                                faer_traits::ComplexConj<
                                    <C::Real as Container>::OfDebug<&dyn core::fmt::Debug>,
                                >,
                            >(map!(
                                this.at(j),
                                dbg,
                                crate::hacks::hijack_debug(dbg)
                            ))
                        }))
                        .finish()
                }
            } else {
                help!(C);
                f.debug_list()
                    .entries(this.ncols().indices().map(|j| unsafe {
                        crate::hacks::coerce::<
                            C::Of<&dyn core::fmt::Debug>,
                            C::OfDebug<&dyn core::fmt::Debug>,
                        >(map!(
                            this.at(j),
                            dbg,
                            crate::hacks::hijack_debug(dbg)
                        ))
                    }))
                    .finish()
            }
        }

        with_dim!(N, self.ncols().unbound());
        imp(f, self.as_col_shape(N).as_dyn_stride())
    }
}
