use super::ColView;
use crate::{
    internal_prelude::*,
    utils::bound::{Array, Dim, Partition},
    ContiguousFwd, Idx, IdxInc,
};
use core::{marker::PhantomData, ptr::NonNull};
use equator::{assert, debug_assert};
use faer_traits::RealValue;
use generativity::Guard;

pub struct ColRef<'a, C: Container, T, Rows = usize, RStride = isize> {
    pub(super) imp: ColView<C, T, Rows, RStride>,
    pub(super) __marker: PhantomData<(&'a T, &'a Rows)>,
}

impl<C: Container, T, Rows: Copy, RStride: Copy> Copy for ColRef<'_, C, T, Rows, RStride> {}
impl<C: Container, T, Rows: Copy, RStride: Copy> Clone for ColRef<'_, C, T, Rows, RStride> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, C: Container, T, Rows: Copy, RStride: Copy> Reborrow<'short>
    for ColRef<'_, C, T, Rows, RStride>
{
    type Target = ColRef<'short, C, T, Rows, RStride>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short, C: Container, T, Rows: Copy, RStride: Copy> ReborrowMut<'short>
    for ColRef<'_, C, T, Rows, RStride>
{
    type Target = ColRef<'short, C, T, Rows, RStride>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}
impl<'a, C: Container, T, Rows: Copy, RStride: Copy> IntoConst for ColRef<'a, C, T, Rows, RStride> {
    type Target = ColRef<'a, C, T, Rows, RStride>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

unsafe impl<C: Container, T: Sync, Rows: Sync, RStride: Sync> Sync
    for ColRef<'_, C, T, Rows, RStride>
{
}
unsafe impl<C: Container, T: Sync, Rows: Send, RStride: Send> Send
    for ColRef<'_, C, T, Rows, RStride>
{
}

impl<'a, C: Container, T> ColRef<'a, C, T> {
    #[inline]
    pub fn from_slice(slice: C::Of<&'a [T]>) -> Self {
        help!(C);
        let len = crate::slice_len::<C>(rb!(slice));
        unsafe { Self::from_raw_parts(map!(slice, slice, slice.as_ptr()), len, 1) }
    }
}

impl<'a, C: Container, T, Rows: Shape, RStride: Stride> ColRef<'a, C, T, Rows, RStride> {
    #[inline(always)]
    #[track_caller]
    pub unsafe fn from_raw_parts(ptr: C::Of<*const T>, nrows: Rows, row_stride: RStride) -> Self {
        help!(C);
        Self {
            imp: ColView {
                ptr: core::mem::transmute_copy::<C::Of<NonNull<T>>, C::OfCopy<NonNull<T>>>(&map!(
                    ptr,
                    ptr,
                    NonNull::new_unchecked(ptr as *mut T)
                )),
                nrows,
                row_stride,
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
    pub fn ncols(&self) -> usize {
        1
    }

    #[inline(always)]
    pub fn shape(&self) -> (Rows, usize) {
        (self.nrows(), self.ncols())
    }

    #[inline(always)]
    pub fn row_stride(&self) -> RStride {
        self.imp.row_stride
    }

    #[inline(always)]
    pub fn ptr_at(&self, row: IdxInc<Rows>) -> C::Of<*const T> {
        help!(C);
        let ptr = self.as_ptr();

        if row >= self.nrows() {
            ptr
        } else {
            map!(ptr, ptr, {
                ptr.wrapping_offset(row.unbound() as isize * self.row_stride().element_stride())
            })
        }
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>) -> C::Of<*const T> {
        help!(C);
        debug_assert!(all(row < self.nrows()));
        map!(self.as_ptr(), ptr, {
            ptr.offset(row.unbound() as isize * self.row_stride().element_stride())
        })
    }

    #[inline]
    #[track_caller]
    pub fn split_at_row(
        self,
        row: IdxInc<Rows>,
    ) -> (
        ColRef<'a, C, T, usize, RStride>,
        ColRef<'a, C, T, usize, RStride>,
    ) {
        assert!(all(row <= self.nrows()));
        let rs = self.row_stride();

        let top = self.as_ptr();
        let bot = self.ptr_at(row);
        unsafe {
            (
                ColRef::from_raw_parts(top, row.unbound(), rs),
                ColRef::from_raw_parts(bot, self.nrows().unbound() - row.unbound(), rs),
            )
        }
    }

    #[inline(always)]
    pub fn transpose(self) -> RowRef<'a, C, T, Rows, RStride> {
        RowRef { trans: self }
    }

    #[inline(always)]
    pub fn conjugate(self) -> ColRef<'a, C::Conj, T::Conj, Rows, RStride>
    where
        T: ConjUnit,
    {
        help!(C);
        unsafe {
            ColRef::from_raw_parts(
                core::mem::transmute_copy(&map!(
                    self.as_ptr(),
                    ptr,
                    core::mem::transmute_copy::<*const T, *const T::Conj>(&ptr)
                )),
                self.nrows(),
                self.row_stride(),
            )
        }
    }

    #[inline(always)]
    pub fn canonical(self) -> ColRef<'a, C::Canonical, T::Canonical, Rows, RStride>
    where
        T: ConjUnit,
    {
        help!(C);
        unsafe {
            ColRef::from_raw_parts(
                core::mem::transmute_copy(&map!(
                    self.as_ptr(),
                    ptr,
                    core::mem::transmute_copy::<*const T, *const T::Canonical>(&ptr)
                )),
                self.nrows(),
                self.row_stride(),
            )
        }
    }

    #[inline(always)]
    pub fn adjoint(self) -> RowRef<'a, C::Conj, T::Conj, Rows, RStride>
    where
        T: ConjUnit,
    {
        self.conjugate().transpose()
    }

    #[inline(always)]
    pub fn at(self, row: Idx<Rows>) -> C::Of<&'a T> {
        assert!(all(row < self.nrows()));
        unsafe { self.at_unchecked(row) }
    }

    #[inline(always)]
    pub unsafe fn at_unchecked(self, row: Idx<Rows>) -> C::Of<&'a T> {
        help!(C);
        map!(self.ptr_inbounds_at(row), ptr, &*ptr)
    }

    #[inline]
    pub fn reverse_rows(self) -> ColRef<'a, C, T, Rows, RStride::Rev> {
        help!(C);
        let row = unsafe { IdxInc::<Rows>::new_unbound(self.nrows().unbound().saturating_sub(1)) };
        let ptr = self.ptr_at(row);
        unsafe { ColRef::from_raw_parts(ptr, self.nrows(), self.row_stride().rev()) }
    }

    #[inline]
    pub fn subrows<V: Shape>(
        self,
        row_start: IdxInc<Rows>,
        nrows: V,
    ) -> ColRef<'a, C, T, V, RStride> {
        assert!(all(row_start <= self.nrows()));
        {
            let nrows = nrows.unbound();
            let full_nrows = self.nrows().unbound();
            let row_start = row_start.unbound();
            assert!(all(nrows <= full_nrows - row_start));
        }
        let rs = self.row_stride();

        unsafe { ColRef::from_raw_parts(self.ptr_at(row_start), nrows, rs) }
    }

    #[inline]
    pub fn subrows_range(
        self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
    ) -> ColRef<'a, C, T, usize, RStride> {
        let rows = rows.0.into()..rows.1.into();
        assert!(all(rows.start <= self.nrows()));

        let rs = self.row_stride();

        unsafe {
            ColRef::from_raw_parts(
                self.ptr_at(rows.start),
                rows.end.unbound().saturating_sub(rows.start.unbound()),
                rs,
            )
        }
    }

    #[inline]
    pub fn as_row_shape<V: Shape>(self, nrows: V) -> ColRef<'a, C, T, V, RStride> {
        assert!(all(self.nrows().unbound() == nrows.unbound()));
        unsafe { ColRef::from_raw_parts(self.as_ptr(), nrows, self.row_stride()) }
    }

    #[inline]
    pub fn as_dyn_rows(self) -> ColRef<'a, C, T, usize, RStride> {
        unsafe { ColRef::from_raw_parts(self.as_ptr(), self.nrows().unbound(), self.row_stride()) }
    }

    #[inline]
    pub fn as_dyn_stride(self) -> ColRef<'a, C, T, Rows, isize> {
        unsafe {
            ColRef::from_raw_parts(
                self.as_ptr(),
                self.nrows(),
                self.row_stride().element_stride(),
            )
        }
    }

    #[inline]
    pub fn iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = C::Of<&'a T>> {
        Rows::indices(Rows::start(), self.nrows().end())
            .map(move |j| unsafe { self.at_unchecked(j) })
    }

    #[inline]
    #[cfg(feature = "rayon")]
    pub fn par_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = C::OfSend<&'a T>>
    where
        T: Sync,
    {
        help!(C);
        use rayon::prelude::*;
        (0..self.nrows().unbound())
            .into_par_iter()
            .map(move |j| unsafe { send!(self.at_unchecked(Idx::<Rows>::new_unbound(j))) })
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_partition(
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColRef<'a, C, T, usize, RStride>>
    where
        T: Sync,
    {
        use rayon::prelude::*;

        let this = self.as_dyn_rows();

        assert!(count > 0);
        (0..count).into_par_iter().map(move |chunk_idx| {
            let (start, len) =
                crate::utils::thread::par_split_indices(this.nrows(), chunk_idx, count);
            this.subrows(start, len)
        })
    }

    #[inline]
    pub fn cloned(self) -> Col<C, T, Rows>
    where
        T: Clone,
    {
        fn imp<'M, C: Container, T: Clone, RStride: Stride>(
            this: ColRef<'_, C, T, Dim<'M>, RStride>,
        ) -> Col<C, T, Dim<'M>> {
            help!(C);
            Col::from_fn(this.nrows(), |i| map!(this.at(i), val, val.clone()))
        }

        with_dim!(M, self.nrows().unbound());
        imp(self.as_row_shape(M)).into_row_shape(self.nrows())
    }

    #[inline]
    pub fn to_owned(self) -> Col<C::Canonical, T::Canonical, Rows>
    where
        T: ConjUnit<Canonical: ComplexField<C::Canonical, MathCtx: Default>>,
    {
        fn imp<'M, C: Container, T, RStride: Stride>(
            this: ColRef<'_, C, T, Dim<'M>, RStride>,
        ) -> Col<C::Canonical, T::Canonical, Dim<'M>>
        where
            T: ConjUnit<Canonical: ComplexField<C::Canonical, MathCtx: Default>>,
        {
            let ctx = &Ctx::<C::Canonical, T::Canonical>::default();
            Col::from_fn(this.nrows(), |i| Conj::apply::<C, T>(ctx, this.at(i)))
        }

        with_dim!(M, self.nrows().unbound());
        imp(self.as_row_shape(M)).into_row_shape(self.nrows())
    }

    #[inline]
    pub fn try_as_col_major(self) -> Option<ColRef<'a, C, T, Rows, ContiguousFwd>> {
        if self.row_stride().element_stride() == 1 {
            Some(unsafe { ColRef::from_raw_parts(self.as_ptr(), self.nrows(), ContiguousFwd) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub unsafe fn const_cast(self) -> ColMut<'a, C, T, Rows, RStride> {
        help!(C);
        ColMut::from_raw_parts_mut(
            map!(self.as_ptr(), ptr, ptr as *mut T),
            self.nrows(),
            self.row_stride(),
        )
    }

    #[inline]
    pub fn as_mat(self) -> MatRef<'a, C, T, Rows, usize, RStride, isize> {
        unsafe {
            MatRef::from_raw_parts(
                self.as_ptr(),
                self.nrows(),
                self.ncols(),
                self.row_stride(),
                0,
            )
        }
    }

    #[inline]
    pub fn as_ref(&self) -> ColRef<'_, C, T, Rows, RStride> {
        *self
    }

    #[inline]
    pub fn bind_r<'N>(self, row: Guard<'N>) -> ColRef<'a, C, T, Dim<'N>, RStride> {
        unsafe { ColRef::from_raw_parts(self.as_ptr(), self.nrows().bind(row), self.row_stride()) }
    }

    #[inline]
    pub(crate) fn __at(self, i: Idx<Rows>) -> C::Of<&'a T> {
        self.at(i)
    }

    #[inline]
    pub fn as_diagonal(self) -> DiagRef<'a, C, T, Rows, RStride> {
        DiagRef { inner: self }
    }

    #[inline]
    pub fn norm_max_with(&self, ctx: &Ctx<C::Canonical, T::Canonical>) -> RealValue<C, T>
    where
        C: Container<Canonical: ComplexContainer>,
        T: ConjUnit<Canonical: ComplexField<C::Canonical>>,
    {
        linalg::reductions::norm_max::norm_max(
            ctx,
            self.canonical().as_dyn_stride().as_dyn_rows().as_mat(),
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
            self.canonical().as_dyn_stride().as_dyn_rows().as_mat(),
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

impl<'a, C: Container, T, Rows: Shape> ColRef<'a, C, T, Rows, ContiguousFwd> {
    #[inline]
    pub fn as_slice(self) -> C::Of<&'a [T]> {
        help!(C);
        map!(self.as_ptr(), ptr, unsafe {
            core::slice::from_raw_parts(ptr, self.nrows().unbound())
        })
    }
}

impl<'a, 'ROWS, C: Container, T> ColRef<'a, C, T, Dim<'ROWS>, ContiguousFwd> {
    #[inline]
    pub fn as_array(self) -> C::Of<&'a Array<'ROWS, T>> {
        help!(C);
        map!(self.as_ptr(), ptr, unsafe {
            &*(core::slice::from_raw_parts(ptr, self.nrows().unbound()) as *const [_]
                as *const Array<'ROWS, T>)
        })
    }
}

impl<'ROWS, 'a, C: Container, T, RStride: Stride> ColRef<'a, C, T, Dim<'ROWS>, RStride> {
    #[inline]
    pub fn split_rows_with<'TOP, 'BOT>(
        self,
        row: Partition<'TOP, 'BOT, 'ROWS>,
    ) -> (
        ColRef<'a, C, T, Dim<'TOP>, RStride>,
        ColRef<'a, C, T, Dim<'BOT>, RStride>,
    ) {
        let (a, b) = self.split_at_row(row.midpoint());
        (a.as_row_shape(row.head), b.as_row_shape(row.tail))
    }
}

impl<C: Container, T: core::fmt::Debug, Rows: Shape, RStride: Stride> core::fmt::Debug
    for ColRef<'_, C, T, Rows, RStride>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.transpose().fmt(f)
    }
}
