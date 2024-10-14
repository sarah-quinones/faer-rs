use crate::{
    internal_prelude::*,
    utils::bound::{Array, Dim, Partition},
    ContiguousFwd, Idx, IdxInc,
};
use core::ops::{Index, IndexMut};
use equator::{assert, debug_assert};
use faer_traits::{RealValue, Unit};

pub struct RowMut<'a, C: Container, T, Cols = usize, CStride = isize> {
    pub(crate) trans: ColMut<'a, C, T, Cols, CStride>,
}

impl<'short, C: Container, T, Rows: Copy, RStride: Copy> Reborrow<'short>
    for RowMut<'_, C, T, Rows, RStride>
{
    type Target = RowRef<'short, C, T, Rows, RStride>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        RowRef {
            trans: self.trans.rb(),
        }
    }
}
impl<'short, C: Container, T, Rows: Copy, RStride: Copy> ReborrowMut<'short>
    for RowMut<'_, C, T, Rows, RStride>
{
    type Target = RowMut<'short, C, T, Rows, RStride>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        RowMut {
            trans: self.trans.rb_mut(),
        }
    }
}
impl<'a, C: Container, T, Rows: Copy, RStride: Copy> IntoConst for RowMut<'a, C, T, Rows, RStride> {
    type Target = RowRef<'a, C, T, Rows, RStride>;

    #[inline]
    fn into_const(self) -> Self::Target {
        RowRef {
            trans: self.trans.into_const(),
        }
    }
}

impl<'a, C: Container, T, Cols: Shape, CStride: Stride> RowMut<'a, C, T, Cols, CStride> {
    #[inline(always)]
    #[track_caller]
    pub unsafe fn from_raw_parts_mut(ptr: C::Of<*mut T>, ncols: Cols, col_stride: CStride) -> Self {
        help!(C);
        Self {
            trans: ColMut::from_raw_parts_mut(ptr, ncols, col_stride),
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
        self.into_const().split_at_col(col)
    }

    #[inline(always)]
    pub fn transpose(self) -> ColRef<'a, C, T, Cols, CStride> {
        self.into_const().transpose()
    }

    #[inline(always)]
    pub fn conjugate(self) -> RowRef<'a, C::Conj, T::Conj, Cols, CStride>
    where
        T: ConjUnit,
    {
        self.into_const().conjugate()
    }

    #[inline(always)]
    pub fn canonical(self) -> RowRef<'a, C::Canonical, T::Canonical, Cols, CStride>
    where
        T: ConjUnit,
    {
        self.into_const().canonical()
    }

    #[inline(always)]
    pub fn adjoint(self) -> ColRef<'a, C::Conj, T::Conj, Cols, CStride>
    where
        T: ConjUnit,
    {
        self.into_const().adjoint()
    }

    #[inline(always)]
    pub fn at(self, col: Idx<Cols>) -> C::Of<&'a T> {
        self.into_const().at(col)
    }

    #[inline(always)]
    pub unsafe fn at_unchecked(self, col: Idx<Cols>) -> C::Of<&'a T> {
        self.into_const().at_unchecked(col)
    }

    #[inline]
    pub fn reverse_cols(self) -> RowRef<'a, C, T, Cols, CStride::Rev> {
        self.into_const().reverse_cols()
    }

    #[inline]
    pub fn subcols<V: Shape>(
        self,
        col_start: IdxInc<Cols>,
        ncols: V,
    ) -> RowRef<'a, C, T, V, CStride> {
        self.into_const().subcols(col_start, ncols)
    }

    #[inline]
    pub fn subcols_range(
        self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> RowRef<'a, C, T, usize, CStride> {
        self.into_const().subcols_range(cols)
    }

    #[inline]
    pub fn as_col_shape<V: Shape>(self, ncols: V) -> RowRef<'a, C, T, V, CStride> {
        self.into_const().as_col_shape(ncols)
    }

    #[inline]
    pub fn as_dyn_cols(self) -> RowRef<'a, C, T, usize, CStride> {
        self.into_const().as_dyn_cols()
    }

    #[inline]
    pub fn as_dyn_stride(self) -> RowRef<'a, C, T, Cols, isize> {
        self.into_const().as_dyn_stride()
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
        self.into_const().try_as_row_major()
    }

    #[inline]
    pub fn as_diagonal(self) -> DiagRef<'a, C, T, Cols, CStride> {
        DiagRef {
            inner: self.trans.into_const(),
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
        self.rb()
    }
    #[inline]
    pub fn as_mut(&mut self) -> RowMut<'_, C, T, Cols, CStride> {
        self.rb_mut()
    }

    #[inline]
    pub fn as_mat(self) -> MatRef<'a, C, T, usize, Cols, isize, CStride> {
        self.into_const().as_mat()
    }
    #[inline]
    pub fn as_mat_mut(self) -> MatMut<'a, C, T, usize, Cols, isize, CStride> {
        unsafe { self.into_const().as_mat().const_cast() }
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

impl<'a, C: Container, T, Cols: Shape, CStride: Stride> RowMut<'a, C, T, Cols, CStride> {
    #[inline(always)]
    pub fn as_ptr_mut(&self) -> C::Of<*mut T> {
        self.trans.as_ptr_mut()
    }

    #[inline(always)]
    pub fn ptr_at_mut(&self, col: IdxInc<Cols>) -> C::Of<*mut T> {
        self.trans.ptr_at_mut(col)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(&self, col: Idx<Cols>) -> C::Of<*mut T> {
        debug_assert!(all(col < self.ncols()));
        self.trans.ptr_inbounds_at_mut(col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col_mut(
        self,
        col: IdxInc<Cols>,
    ) -> (
        RowMut<'a, C, T, usize, CStride>,
        RowMut<'a, C, T, usize, CStride>,
    ) {
        let (a, b) = self.into_const().split_at_col(col);
        unsafe { (a.const_cast(), b.const_cast()) }
    }

    #[inline(always)]
    pub fn transpose_mut(self) -> ColMut<'a, C, T, Cols, CStride> {
        self.trans
    }

    #[inline(always)]
    pub fn conjugate_mut(self) -> RowMut<'a, C::Conj, T::Conj, Cols, CStride>
    where
        T: ConjUnit,
    {
        unsafe { self.into_const().conjugate().const_cast() }
    }

    #[inline(always)]
    pub fn canonical_mut(self) -> RowMut<'a, C::Canonical, T::Canonical, Cols, CStride>
    where
        T: ConjUnit,
    {
        unsafe { self.into_const().canonical().const_cast() }
    }

    #[inline(always)]
    pub fn adjoint_mut(self) -> ColMut<'a, C::Conj, T::Conj, Cols, CStride>
    where
        T: ConjUnit,
    {
        unsafe { self.into_const().adjoint().const_cast() }
    }

    #[inline(always)]
    pub fn at_mut(self, col: Idx<Cols>) -> C::Of<&'a mut T> {
        assert!(all(col < self.ncols()));
        unsafe { self.at_mut_unchecked(col) }
    }

    #[inline(always)]
    pub unsafe fn at_mut_unchecked(self, col: Idx<Cols>) -> C::Of<&'a mut T> {
        help!(C);
        map!(self.ptr_inbounds_at_mut(col), ptr, &mut *ptr)
    }

    #[inline]
    pub fn reverse_cols_mut(self) -> RowMut<'a, C, T, Cols, CStride::Rev> {
        unsafe { self.into_const().reverse_cols().const_cast() }
    }

    #[inline]
    pub fn subcols_mut<V: Shape>(
        self,
        col_start: IdxInc<Cols>,
        ncols: V,
    ) -> RowMut<'a, C, T, V, CStride> {
        unsafe { self.into_const().subcols(col_start, ncols).const_cast() }
    }

    #[inline]
    pub fn subcols_range_mut(
        self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> RowMut<'a, C, T, usize, CStride> {
        unsafe { self.into_const().subcols_range(cols).const_cast() }
    }

    #[inline]
    pub fn as_col_shape_mut<V: Shape>(self, ncols: V) -> RowMut<'a, C, T, V, CStride> {
        unsafe { self.into_const().as_col_shape(ncols).const_cast() }
    }

    #[inline]
    pub fn as_dyn_cols_mut(self) -> RowMut<'a, C, T, usize, CStride> {
        unsafe { self.into_const().as_dyn_cols().const_cast() }
    }

    #[inline]
    pub fn as_dyn_stride_mut(self) -> RowMut<'a, C, T, Cols, isize> {
        unsafe { self.into_const().as_dyn_stride().const_cast() }
    }

    #[inline]
    pub fn iter_mut(
        self,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = C::Of<&'a mut T>> {
        self.trans.iter_mut()
    }

    #[inline]
    #[cfg(feature = "rayon")]
    pub fn par_iter_mut(
        self,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = C::OfSend<&'a mut T>>
    where
        T: Send,
    {
        self.trans.par_iter_mut()
    }

    pub(crate) unsafe fn as_type<U>(self) -> RowMut<'a, C, U, Cols, CStride> {
        help!(C);
        RowMut::from_raw_parts_mut(
            map!(
                self.as_ptr_mut(),
                ptr,
                core::mem::transmute_copy::<*mut T, *mut U>(&ptr)
            ),
            self.ncols(),
            self.col_stride(),
        )
    }

    #[inline]
    pub fn copy_from_with_ctx<RhsC: Container<Canonical = C>, RhsT: ConjUnit<Canonical = T>>(
        &mut self,
        ctx: &T::MathCtx,
        other: impl AsRowRef<RhsC, RhsT, Cols>,
    ) where
        C: ComplexContainer,
        T: ComplexField<C>,
    {
        let other = other.as_row_ref();

        assert!(all(
            self.nrows() == other.nrows(),
            self.ncols() == other.ncols(),
        ));
        let n = self.ncols();

        with_dim!(N, n.unbound());
        let this = self.rb_mut().as_col_shape_mut(N).as_dyn_stride_mut();
        let other = other.as_col_shape(N);
        imp(ctx, this, other.canonical(), Conj::get::<RhsC, RhsT>());

        pub fn imp<'N, C: ComplexContainer, T: ComplexField<C>>(
            ctx: &T::MathCtx,
            this: RowMut<'_, C, T, Dim<'N>>,
            other: RowRef<'_, C, T, Dim<'N>>,
            conj: Conj,
        ) {
            help!(C);
            let ctx = Ctx::<C, T>::new(ctx);
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
    pub fn try_as_row_major_mut(self) -> Option<RowMut<'a, C, T, Cols, ContiguousFwd>> {
        self.into_const()
            .try_as_row_major()
            .map(|x| unsafe { x.const_cast() })
    }

    #[inline]
    pub fn as_diagonal_mut(self) -> DiagMut<'a, C, T, Cols, CStride> {
        DiagMut { inner: self.trans }
    }

    #[inline]
    pub(crate) fn __at_mut(self, i: Idx<Cols>) -> C::Of<&'a mut T> {
        self.at_mut(i)
    }
}

impl<'a, C: Container, T, Rows: Shape> RowMut<'a, C, T, Rows, ContiguousFwd> {
    #[inline]
    pub fn as_slice(self) -> C::Of<&'a [T]> {
        self.transpose().as_slice()
    }
}

impl<'a, 'ROWS, C: Container, T> RowMut<'a, C, T, Dim<'ROWS>, ContiguousFwd> {
    #[inline]
    pub fn as_array(self) -> C::Of<&'a Array<'ROWS, T>> {
        self.transpose().as_array()
    }
}

impl<'a, C: Container, T, Cols: Shape> RowMut<'a, C, T, Cols, ContiguousFwd> {
    #[inline]
    pub fn as_slice_mut(self) -> C::Of<&'a mut [T]> {
        self.transpose_mut().as_slice_mut()
    }
}

impl<'a, 'COLS, C: Container, T> RowMut<'a, C, T, Dim<'COLS>, ContiguousFwd> {
    #[inline]
    pub fn as_array_mut(self) -> C::Of<&'a mut Array<'COLS, T>> {
        self.transpose_mut().as_array_mut()
    }
}

impl<'COLS, 'a, C: Container, T, CStride: Stride> RowMut<'a, C, T, Dim<'COLS>, CStride> {
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

impl<'COLS, 'a, C: Container, T, CStride: Stride> RowMut<'a, C, T, Dim<'COLS>, CStride> {
    #[inline]
    pub fn split_cols_with_mut<'LEFT, 'RIGHT>(
        self,
        col: Partition<'LEFT, 'RIGHT, 'COLS>,
    ) -> (
        RowMut<'a, C, T, Dim<'LEFT>, CStride>,
        RowMut<'a, C, T, Dim<'RIGHT>, CStride>,
    ) {
        let (a, b) = self.split_at_col_mut(col.midpoint());
        (a.as_col_shape_mut(col.head), b.as_col_shape_mut(col.tail))
    }
}

impl<C: Container, T: core::fmt::Debug, Cols: Shape, CStride: Stride> core::fmt::Debug
    for RowMut<'_, C, T, Cols, CStride>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<T, Cols: Shape, RStride: Stride> Index<Idx<Cols>> for RowRef<'_, Unit, T, Cols, RStride> {
    type Output = T;

    #[inline]
    fn index(&self, col: Idx<Cols>) -> &Self::Output {
        self.at(col)
    }
}

impl<T, Cols: Shape, RStride: Stride> Index<Idx<Cols>> for RowMut<'_, Unit, T, Cols, RStride> {
    type Output = T;

    #[inline]
    fn index(&self, col: Idx<Cols>) -> &Self::Output {
        self.rb().at(col)
    }
}

impl<T, Cols: Shape, RStride: Stride> IndexMut<Idx<Cols>> for RowMut<'_, Unit, T, Cols, RStride> {
    #[inline]
    fn index_mut(&mut self, col: Idx<Cols>) -> &mut Self::Output {
        self.rb_mut().at_mut(col)
    }
}

mod bound_range {
    use super::*;
    use crate::utils::bound::{Disjoint, Segment};

    impl<'COLS, 'a, C: Container, T, CStride: Stride> RowMut<'a, C, T, Dim<'COLS>, CStride> {
        #[inline]
        pub fn col_segments_mut<'scope, 'LEFT, 'RIGHT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
            second: Segment<'scope, 'COLS, 'RIGHT>,
            disjoint: Disjoint<'scope, 'LEFT, 'RIGHT>,
        ) -> (
            RowMut<'a, C, T, Dim<'LEFT>, CStride>,
            RowMut<'a, C, T, Dim<'RIGHT>, CStride>,
        ) {
            unsafe {
                _ = disjoint;
                let first = RowMut::from_raw_parts_mut(
                    self.ptr_at_mut(first.start()),
                    first.len(),
                    self.col_stride(),
                );
                let second = RowMut::from_raw_parts_mut(
                    self.ptr_at_mut(second.start()),
                    second.len(),
                    self.col_stride(),
                );
                (first, second)
            }
        }
    }

    impl<'COLS, 'a, C: Container, T, CStride: Stride> RowMut<'a, C, T, Dim<'COLS>, CStride> {
        #[inline]
        pub fn col_segments<'scope, 'LEFT, 'RIGHT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
            second: Segment<'scope, 'COLS, 'RIGHT>,
        ) -> (
            RowRef<'a, C, T, Dim<'LEFT>, CStride>,
            RowRef<'a, C, T, Dim<'RIGHT>, CStride>,
        ) {
            unsafe {
                let first = RowRef::from_raw_parts(
                    self.ptr_at(first.start()),
                    first.len(),
                    self.col_stride(),
                );
                let second = RowRef::from_raw_parts(
                    self.ptr_at(second.start()),
                    second.len(),
                    self.col_stride(),
                );
                (first, second)
            }
        }
    }

    impl<'COLS, 'a, C: Container, T, CStride: Stride> RowRef<'a, C, T, Dim<'COLS>, CStride> {
        #[inline]
        pub fn col_segments<'scope, 'LEFT, 'RIGHT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
            second: Segment<'scope, 'COLS, 'RIGHT>,
        ) -> (
            RowRef<'a, C, T, Dim<'LEFT>, CStride>,
            RowRef<'a, C, T, Dim<'RIGHT>, CStride>,
        ) {
            unsafe {
                let first = RowRef::from_raw_parts(
                    self.ptr_at(first.start()),
                    first.len(),
                    self.col_stride(),
                );
                let second = RowRef::from_raw_parts(
                    self.ptr_at(second.start()),
                    second.len(),
                    self.col_stride(),
                );
                (first, second)
            }
        }
    }

    // single segment

    impl<'COLS, 'a, C: Container, T, RStride: Stride> RowMut<'a, C, T, Dim<'COLS>, RStride> {
        #[inline]
        pub fn col_segment_mut<'scope, 'LEFT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
        ) -> RowMut<'a, C, T, Dim<'LEFT>, RStride> {
            unsafe {
                RowMut::from_raw_parts_mut(
                    self.ptr_at_mut(first.start()),
                    first.len(),
                    self.col_stride(),
                )
            }
        }
    }
    impl<'COLS, 'a, C: Container, T, RStride: Stride> RowRef<'a, C, T, Dim<'COLS>, RStride> {
        #[inline]
        pub fn col_segment<'scope, 'LEFT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
        ) -> RowRef<'a, C, T, Dim<'LEFT>, RStride> {
            unsafe {
                RowRef::from_raw_parts(self.ptr_at(first.start()), first.len(), self.col_stride())
            }
        }
    }

    impl<'COLS, 'a, C: Container, T, RStride: Stride> RowMut<'a, C, T, Dim<'COLS>, RStride> {
        #[inline]
        pub fn col_segment<'scope, 'LEFT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
        ) -> RowRef<'a, C, T, Dim<'LEFT>, RStride> {
            unsafe {
                RowRef::from_raw_parts(self.ptr_at(first.start()), first.len(), self.col_stride())
            }
        }
    }
}

mod bound_any_range {
    use super::*;
    use crate::variadics::*;

    impl<'COLS, 'a, C: Container, T, CStride: Stride> RowMut<'a, C, T, Dim<'COLS>, CStride> {
        #[inline]
        pub fn any_col_segments_mut<'scope, S: ColSplit<'scope, 'COLS, 'a>>(
            self,
            segments: S,
            disjoint: S::Disjoint,
        ) -> S::RowMutSegments<C, T, CStride> {
            S::row_mut_segments(segments, self, disjoint)
        }
    }

    impl<'COLS, 'a, C: Container, T, CStride: Stride> RowRef<'a, C, T, Dim<'COLS>, CStride> {
        #[inline]
        pub fn any_col_segments<'scope, S: ColSplit<'scope, 'COLS, 'a>>(
            self,
            segments: S,
        ) -> S::RowRefSegments<C, T, CStride> {
            S::row_ref_segments(segments, self)
        }
    }

    impl<'COLS, 'a, C: Container, T, CStride: Stride> RowMut<'a, C, T, Dim<'COLS>, CStride> {
        #[inline]
        pub fn any_col_segments<'scope, S: ColSplit<'scope, 'COLS, 'a>>(
            self,
            segments: S,
        ) -> S::RowRefSegments<C, T, CStride> {
            S::row_ref_segments(segments, self.into_const())
        }
    }
}
