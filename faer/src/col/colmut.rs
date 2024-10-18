use super::ColView;
use crate::{
    internal_prelude::*,
    mat::matmut::SyncCell,
    utils::bound::{Array, Dim, Partition},
    ContiguousFwd, Idx, IdxInc,
};
use core::{
    marker::PhantomData,
    ops::{Index, IndexMut},
    ptr::NonNull,
};
use equator::assert;
use faer_traits::{RealValue, Unit};
use generativity::Guard;

pub struct ColMut<'a, C: Container, T, Rows = usize, RStride = isize> {
    pub(super) imp: ColView<C, T, Rows, RStride>,
    pub(super) __marker: PhantomData<(&'a mut T, &'a Rows)>,
}

impl<'short, C: Container, T, Rows: Copy, RStride: Copy> Reborrow<'short>
    for ColMut<'_, C, T, Rows, RStride>
{
    type Target = ColRef<'short, C, T, Rows, RStride>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        ColRef {
            imp: self.imp,
            __marker: PhantomData,
        }
    }
}
impl<'short, C: Container, T, Rows: Copy, RStride: Copy> ReborrowMut<'short>
    for ColMut<'_, C, T, Rows, RStride>
{
    type Target = ColMut<'short, C, T, Rows, RStride>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        ColMut {
            imp: self.imp,
            __marker: PhantomData,
        }
    }
}
impl<'a, C: Container, T, Rows: Copy, RStride: Copy> IntoConst for ColMut<'a, C, T, Rows, RStride> {
    type Target = ColRef<'a, C, T, Rows, RStride>;

    #[inline]
    fn into_const(self) -> Self::Target {
        ColRef {
            imp: self.imp,
            __marker: PhantomData,
        }
    }
}

unsafe impl<C: Container, T: Sync, Rows: Sync, RStride: Sync> Sync
    for ColMut<'_, C, T, Rows, RStride>
{
}
unsafe impl<C: Container, T: Send, Rows: Send, RStride: Send> Send
    for ColMut<'_, C, T, Rows, RStride>
{
}

impl<'a, C: Container, T> ColMut<'a, C, T> {
    #[inline]
    pub fn from_slice_mut(slice: C::Of<&'a mut [T]>) -> Self {
        help!(C);
        let len = crate::slice_len::<C>(rb!(slice));
        unsafe { Self::from_raw_parts_mut(map!(slice, slice, slice.as_mut_ptr()), len, 1) }
    }
}

impl<'a, C: Container, T, Rows: Shape, RStride: Stride> ColMut<'a, C, T, Rows, RStride> {
    #[inline(always)]
    #[track_caller]
    pub unsafe fn from_raw_parts_mut(ptr: C::Of<*mut T>, nrows: Rows, row_stride: RStride) -> Self {
        help!(C);
        Self {
            imp: ColView {
                ptr: core::mem::transmute_copy::<C::Of<NonNull<T>>, C::OfCopy<NonNull<T>>>(&map!(
                    ptr,
                    ptr,
                    NonNull::new_unchecked(ptr)
                )),
                nrows,
                row_stride,
            },
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> C::Of<*const T> {
        self.rb().as_ptr()
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
        self.rb().ptr_at(row)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>) -> C::Of<*const T> {
        self.rb().ptr_inbounds_at(row)
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
        self.into_const().split_at_row(row)
    }

    #[inline(always)]
    pub fn transpose(self) -> RowRef<'a, C, T, Rows, RStride> {
        self.into_const().transpose()
    }

    #[inline(always)]
    pub fn conjugate(self) -> ColRef<'a, C::Conj, T::Conj, Rows, RStride>
    where
        T: ConjUnit,
    {
        self.into_const().conjugate()
    }

    #[inline(always)]
    pub fn canonical(self) -> ColRef<'a, C::Canonical, T::Canonical, Rows, RStride>
    where
        T: ConjUnit,
    {
        self.into_const().canonical()
    }

    #[inline(always)]
    pub fn adjoint(self) -> RowRef<'a, C::Conj, T::Conj, Rows, RStride>
    where
        T: ConjUnit,
    {
        self.into_const().adjoint()
    }

    #[inline(always)]
    pub fn at(self, row: Idx<Rows>) -> C::Of<&'a T> {
        self.into_const().at(row)
    }

    #[inline(always)]
    pub unsafe fn at_unchecked(self, row: Idx<Rows>) -> C::Of<&'a T> {
        self.into_const().at_unchecked(row)
    }

    #[inline]
    pub fn reverse_rows(self) -> ColRef<'a, C, T, Rows, RStride::Rev> {
        self.into_const().reverse_rows()
    }

    #[inline]
    pub fn subrows<V: Shape>(
        self,
        row_start: IdxInc<Rows>,
        nrows: V,
    ) -> ColRef<'a, C, T, V, RStride> {
        self.into_const().subrows(row_start, nrows)
    }

    #[inline]
    pub fn subrows_range(
        self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
    ) -> ColRef<'a, C, T, usize, RStride> {
        self.into_const().subrows_range(rows)
    }

    #[inline]
    pub fn as_row_shape<V: Shape>(self, nrows: V) -> ColRef<'a, C, T, V, RStride> {
        self.into_const().as_row_shape(nrows)
    }

    #[inline]
    pub fn as_dyn_rows(self) -> ColRef<'a, C, T, usize, RStride> {
        self.into_const().as_dyn_rows()
    }

    #[inline]
    pub fn as_dyn_stride(self) -> ColRef<'a, C, T, Rows, isize> {
        self.into_const().as_dyn_stride()
    }

    #[inline]
    pub fn iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = C::Of<&'a T>> {
        self.into_const().iter()
    }

    #[inline]
    #[cfg(feature = "rayon")]
    pub fn par_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = C::OfSend<&'a T>>
    where
        T: Sync,
    {
        self.into_const().par_iter()
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
        self.into_const().par_partition(count)
    }

    #[inline]
    pub fn try_as_col_major(self) -> Option<ColRef<'a, C, T, Rows, ContiguousFwd>> {
        self.into_const().try_as_col_major()
    }

    #[inline]
    pub fn try_as_col_major_mut(self) -> Option<ColMut<'a, C, T, Rows, ContiguousFwd>> {
        self.into_const()
            .try_as_col_major()
            .map(|x| unsafe { x.const_cast() })
    }

    #[inline(always)]
    pub unsafe fn const_cast(self) -> ColMut<'a, C, T, Rows, RStride> {
        self
    }

    #[inline]
    pub fn as_ref(&self) -> ColRef<'_, C, T, Rows, RStride> {
        self.rb()
    }

    #[inline]
    pub fn as_mut(&mut self) -> ColMut<'_, C, T, Rows, RStride> {
        self.rb_mut()
    }

    #[inline]
    pub fn bind_r<'N>(self, row: Guard<'N>) -> ColMut<'a, C, T, Dim<'N>, RStride> {
        unsafe {
            ColMut::from_raw_parts_mut(self.as_ptr_mut(), self.nrows().bind(row), self.row_stride())
        }
    }

    #[inline]
    pub fn as_mat(self) -> MatRef<'a, C, T, Rows, usize, RStride, isize> {
        self.into_const().as_mat()
    }
    #[inline]
    pub fn as_mat_mut(self) -> MatMut<'a, C, T, Rows, usize, RStride, isize> {
        unsafe { self.into_const().as_mat().const_cast() }
    }

    #[inline]
    pub fn as_diagonal(self) -> DiagRef<'a, C, T, Rows, RStride> {
        DiagRef {
            inner: self.into_const(),
        }
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

impl<'a, C: Container, T, Rows: Shape, RStride: Stride> ColMut<'a, C, T, Rows, RStride> {
    #[inline(always)]
    pub fn as_ptr_mut(&self) -> C::Of<*mut T> {
        help!(C);
        map!(self.rb().as_ptr(), ptr, ptr as *mut T)
    }

    #[inline(always)]
    pub fn ptr_at_mut(&self, row: IdxInc<Rows>) -> C::Of<*mut T> {
        help!(C);
        map!(self.rb().ptr_at(row), ptr, ptr as *mut T)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(&self, row: Idx<Rows>) -> C::Of<*mut T> {
        help!(C);
        map!(self.rb().ptr_inbounds_at(row), ptr, ptr as *mut T)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_row_mut(
        self,
        row: IdxInc<Rows>,
    ) -> (
        ColMut<'a, C, T, usize, RStride>,
        ColMut<'a, C, T, usize, RStride>,
    ) {
        let (a, b) = self.into_const().split_at_row(row);
        unsafe { (a.const_cast(), b.const_cast()) }
    }

    #[inline(always)]
    pub fn transpose_mut(self) -> RowMut<'a, C, T, Rows, RStride> {
        unsafe { self.into_const().transpose().const_cast() }
    }

    #[inline(always)]
    pub fn conjugate_mut(self) -> ColMut<'a, C::Conj, T::Conj, Rows, RStride>
    where
        T: ConjUnit,
    {
        unsafe { self.into_const().conjugate().const_cast() }
    }

    #[inline(always)]
    pub fn canonical_mut(self) -> ColMut<'a, C::Canonical, T::Canonical, Rows, RStride>
    where
        T: ConjUnit,
    {
        unsafe { self.into_const().canonical().const_cast() }
    }

    #[inline(always)]
    pub fn adjoint_mut(self) -> RowMut<'a, C::Conj, T::Conj, Rows, RStride>
    where
        T: ConjUnit,
    {
        unsafe { self.into_const().adjoint().const_cast() }
    }

    #[inline(always)]
    pub fn at_mut(self, row: Idx<Rows>) -> C::Of<&'a mut T> {
        assert!(all(row < self.nrows()));
        unsafe { self.at_mut_unchecked(row) }
    }

    #[inline(always)]
    pub unsafe fn at_mut_unchecked(self, row: Idx<Rows>) -> C::Of<&'a mut T> {
        help!(C);
        map!(self.ptr_inbounds_at_mut(row), ptr, &mut *ptr)
    }

    #[inline]
    pub fn reverse_rows_mut(self) -> ColMut<'a, C, T, Rows, RStride::Rev> {
        unsafe { self.into_const().reverse_rows().const_cast() }
    }

    #[inline]
    pub fn subrows_mut<V: Shape>(
        self,
        row_start: IdxInc<Rows>,
        nrows: V,
    ) -> ColMut<'a, C, T, V, RStride> {
        unsafe { self.into_const().subrows(row_start, nrows).const_cast() }
    }

    #[inline]
    pub fn subrows_range_mut(
        self,
        rows: (impl Into<IdxInc<Rows>>, impl Into<IdxInc<Rows>>),
    ) -> ColMut<'a, C, T, usize, RStride> {
        unsafe { self.into_const().subrows_range(rows).const_cast() }
    }

    #[inline]
    pub fn as_row_shape_mut<V: Shape>(self, nrows: V) -> ColMut<'a, C, T, V, RStride> {
        unsafe { self.into_const().as_row_shape(nrows).const_cast() }
    }

    #[inline]
    pub fn as_dyn_rows_mut(self) -> ColMut<'a, C, T, usize, RStride> {
        unsafe { self.into_const().as_dyn_rows().const_cast() }
    }

    #[inline]
    pub fn as_dyn_stride_mut(self) -> ColMut<'a, C, T, Rows, isize> {
        unsafe { self.into_const().as_dyn_stride().const_cast() }
    }

    #[inline]
    pub fn iter_mut(
        self,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = C::Of<&'a mut T>> {
        let this = self.into_const();
        Rows::indices(Rows::start(), this.nrows().end())
            .map(move |j| unsafe { this.const_cast().at_mut_unchecked(j) })
    }

    #[inline]
    #[cfg(feature = "rayon")]
    pub fn par_iter_mut(
        self,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = C::OfSend<&'a mut T>>
    where
        T: Send,
    {
        unsafe {
            let this = self.as_type::<SyncCell<T>>().into_const();
            help!(C);

            use rayon::prelude::*;
            (0..this.nrows().unbound()).into_par_iter().map(move |j| {
                send!(map!(
                    this.const_cast()
                        .at_mut_unchecked(Idx::<Rows>::new_unbound(j)),
                    ptr,
                    &mut *(ptr as *mut SyncCell<T> as *mut T)
                ))
            })
        }
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_partition_mut(
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColMut<'a, C, T, usize, RStride>>
    where
        T: Send,
    {
        use rayon::prelude::*;
        unsafe {
            self.as_type::<SyncCell<T>>()
                .into_const()
                .par_partition(count)
                .map(|col| col.const_cast().as_type::<T>())
        }
    }

    pub(crate) unsafe fn as_type<U>(self) -> ColMut<'a, C, U, Rows, RStride> {
        help!(C);
        ColMut::from_raw_parts_mut(
            map!(
                self.as_ptr_mut(),
                ptr,
                core::mem::transmute_copy::<*mut T, *mut U>(&ptr)
            ),
            self.nrows(),
            self.row_stride(),
        )
    }

    #[inline]
    pub fn as_diagonal_mut(self) -> DiagMut<'a, C, T, Rows, RStride> {
        DiagMut { inner: self }
    }

    #[inline]
    pub fn copy_from_with<RhsC: Container<Canonical = C>, RhsT: ConjUnit<Canonical = T>>(
        &mut self,
        ctx: &Ctx<C, T>,
        other: impl AsColRef<RhsC, RhsT, Rows>,
    ) where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        let other = other.as_col_ref();

        assert!(all(
            self.nrows() == other.nrows(),
            self.ncols() == other.ncols(),
        ));
        let m = self.nrows();

        with_dim!(M, m.unbound());
        imp(
            ctx,
            self.rb_mut().as_row_shape_mut(M).as_dyn_stride_mut(),
            other.as_row_shape(M).canonical(),
            Conj::get::<RhsC, RhsT>(),
        );

        pub fn imp<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
            ctx: &Ctx<C, T>,
            this: ColMut<'_, C, T, Dim<'M>>,
            other: ColRef<'_, C, T, Dim<'M>>,
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
    pub fn fill(&mut self, value: C::Of<T>)
    where
        T: Clone,
    {
        fn cloner<C: Container, T: Clone>(
            value: C::Of<T>,
        ) -> impl for<'a> FnMut(crate::linalg::zip::Last<C::Of<&'a mut T>>) {
            help!(C);
            #[inline(always)]
            move |x| {
                map!(zip!(x.0, as_ref!(value)), (x, value), *x = value.clone());
            }
        }
        z!(self.rb_mut().as_dyn_rows_mut()).for_each(cloner::<C, T>(value));
    }

    #[inline]
    pub(crate) fn __at_mut(self, i: Idx<Rows>) -> C::Of<&'a mut T> {
        self.at_mut(i)
    }
}

impl<'a, C: Container, T, Rows: Shape> ColMut<'a, C, T, Rows, ContiguousFwd> {
    #[inline]
    pub fn as_slice_mut(self) -> C::Of<&'a mut [T]> {
        help!(C);
        map!(self.as_ptr_mut(), ptr, unsafe {
            core::slice::from_raw_parts_mut(ptr, self.nrows().unbound())
        })
    }
}

impl<'a, 'ROWS, C: Container, T> ColMut<'a, C, T, Dim<'ROWS>, ContiguousFwd> {
    #[inline]
    pub fn as_array_mut(self) -> C::Of<&'a mut Array<'ROWS, T>> {
        help!(C);
        map!(self.as_ptr_mut(), ptr, unsafe {
            &mut *(core::slice::from_raw_parts_mut(ptr, self.nrows().unbound()) as *mut [_]
                as *mut Array<'ROWS, T>)
        })
    }
}

impl<'ROWS, 'a, C: Container, T, RStride: Stride> ColMut<'a, C, T, Dim<'ROWS>, RStride> {
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

impl<'ROWS, 'a, C: Container, T, RStride: Stride> ColMut<'a, C, T, Dim<'ROWS>, RStride> {
    #[inline]
    pub fn split_rows_with_mut<'TOP, 'BOT>(
        self,
        row: Partition<'TOP, 'BOT, 'ROWS>,
    ) -> (
        ColMut<'a, C, T, Dim<'TOP>, RStride>,
        ColMut<'a, C, T, Dim<'BOT>, RStride>,
    ) {
        let (a, b) = self.split_at_row_mut(row.midpoint());
        (a.as_row_shape_mut(row.head), b.as_row_shape_mut(row.tail))
    }
}

impl<C: Container, T: core::fmt::Debug, Rows: Shape, RStride: Stride> core::fmt::Debug
    for ColMut<'_, C, T, Rows, RStride>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<T, Rows: Shape, CStride: Stride> Index<Idx<Rows>> for ColRef<'_, Unit, T, Rows, CStride> {
    type Output = T;

    #[inline]
    fn index(&self, row: Idx<Rows>) -> &Self::Output {
        self.at(row)
    }
}

impl<T, Rows: Shape, CStride: Stride> Index<Idx<Rows>> for ColMut<'_, Unit, T, Rows, CStride> {
    type Output = T;

    #[inline]
    fn index(&self, row: Idx<Rows>) -> &Self::Output {
        self.rb().at(row)
    }
}

impl<T, Rows: Shape, CStride: Stride> IndexMut<Idx<Rows>> for ColMut<'_, Unit, T, Rows, CStride> {
    #[inline]
    fn index_mut(&mut self, row: Idx<Rows>) -> &mut Self::Output {
        self.rb_mut().at_mut(row)
    }
}

mod bound_range {
    use super::*;
    use crate::utils::bound::Segment;

    // single segment

    impl<'ROWS, 'a, C: Container, T, RStride: Stride> ColMut<'a, C, T, Dim<'ROWS>, RStride> {
        #[inline]
        pub fn row_segment_mut<'scope, 'TOP>(
            self,
            first: Segment<'scope, 'ROWS, 'TOP>,
        ) -> ColMut<'a, C, T, Dim<'TOP>, RStride> {
            unsafe {
                ColMut::from_raw_parts_mut(
                    self.ptr_at_mut(first.start().local()),
                    first.len(),
                    self.row_stride(),
                )
            }
        }
    }
    impl<'ROWS, 'a, C: Container, T, RStride: Stride> ColRef<'a, C, T, Dim<'ROWS>, RStride> {
        #[inline]
        pub fn row_segment<'scope, 'TOP>(
            self,
            first: Segment<'scope, 'ROWS, 'TOP>,
        ) -> ColRef<'a, C, T, Dim<'TOP>, RStride> {
            unsafe {
                ColRef::from_raw_parts(
                    self.ptr_at(first.start().local()),
                    first.len(),
                    self.row_stride(),
                )
            }
        }
    }

    impl<'ROWS, 'a, C: Container, T, RStride: Stride> ColMut<'a, C, T, Dim<'ROWS>, RStride> {
        #[inline]
        pub fn row_segment<'scope, 'TOP>(
            self,
            first: Segment<'scope, 'ROWS, 'TOP>,
        ) -> ColRef<'a, C, T, Dim<'TOP>, RStride> {
            unsafe {
                ColRef::from_raw_parts(
                    self.ptr_at(first.start().local()),
                    first.len(),
                    self.row_stride(),
                )
            }
        }
    }
}

mod bound_any_range {
    use super::*;
    use crate::variadics::*;

    impl<'ROWS, 'a, C: Container, T, RStride: Stride> ColMut<'a, C, T, Dim<'ROWS>, RStride> {
        #[inline]
        pub fn row_segments_mut<'scope, S: RowSplit<'scope, 'ROWS, 'a>>(
            self,
            segments: S,
            disjoint: S::Disjoint,
        ) -> S::ColMutSegments<C, T, RStride> {
            S::col_mut_segments(segments, self, disjoint)
        }
    }

    impl<'ROWS, 'a, C: Container, T, RStride: Stride> ColRef<'a, C, T, Dim<'ROWS>, RStride> {
        #[inline]
        pub fn row_segments<'scope, S: RowSplit<'scope, 'ROWS, 'a>>(
            self,
            segments: S,
        ) -> S::ColRefSegments<C, T, RStride> {
            S::col_ref_segments(segments, self)
        }
    }

    impl<'ROWS, 'a, C: Container, T, RStride: Stride> ColMut<'a, C, T, Dim<'ROWS>, RStride> {
        #[inline]
        pub fn row_segments<'scope, S: RowSplit<'scope, 'ROWS, 'a>>(
            self,
            segments: S,
        ) -> S::ColRefSegments<C, T, RStride> {
            S::col_ref_segments(segments, self.into_const())
        }
    }
}
