use crate::{
    internal_prelude::*,
    utils::bound::{Array, Dim, Partition},
    ContiguousFwd, Idx, IdxInc,
};
use core::ops::{Index, IndexMut};
use equator::{assert, debug_assert};
use faer_traits::Real;

pub struct RowMut<'a, T, Cols = usize, CStride = isize> {
    pub(crate) trans: ColMut<'a, T, Cols, CStride>,
}

impl<'short, T, Rows: Copy, RStride: Copy> Reborrow<'short> for RowMut<'_, T, Rows, RStride> {
    type Target = RowRef<'short, T, Rows, RStride>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        RowRef {
            trans: self.trans.rb(),
        }
    }
}
impl<'short, T, Rows: Copy, RStride: Copy> ReborrowMut<'short> for RowMut<'_, T, Rows, RStride> {
    type Target = RowMut<'short, T, Rows, RStride>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        RowMut {
            trans: self.trans.rb_mut(),
        }
    }
}
impl<'a, T, Rows: Copy, RStride: Copy> IntoConst for RowMut<'a, T, Rows, RStride> {
    type Target = RowRef<'a, T, Rows, RStride>;

    #[inline]
    fn into_const(self) -> Self::Target {
        RowRef {
            trans: self.trans.into_const(),
        }
    }
}

impl<'a, T, Cols: Shape, CStride: Stride> RowMut<'a, T, Cols, CStride> {
    #[inline(always)]
    #[track_caller]
    pub unsafe fn from_raw_parts_mut(ptr: *mut T, ncols: Cols, col_stride: CStride) -> Self {
        Self {
            trans: ColMut::from_raw_parts_mut(ptr, ncols, col_stride),
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
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
    pub fn ptr_at(&self, col: IdxInc<Cols>) -> *const T {
        self.trans.ptr_at(col)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, col: Idx<Cols>) -> *const T {
        debug_assert!(all(col < self.ncols()));
        self.trans.ptr_inbounds_at(col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col(
        self,
        col: IdxInc<Cols>,
    ) -> (RowRef<'a, T, usize, CStride>, RowRef<'a, T, usize, CStride>) {
        self.into_const().split_at_col(col)
    }

    #[inline(always)]
    pub fn transpose(self) -> ColRef<'a, T, Cols, CStride> {
        self.into_const().transpose()
    }

    #[inline(always)]
    pub fn conjugate(self) -> RowRef<'a, T::Conj, Cols, CStride>
    where
        T: Conjugate,
    {
        self.into_const().conjugate()
    }

    #[inline(always)]
    pub fn canonical(self) -> RowRef<'a, T::Canonical, Cols, CStride>
    where
        T: Conjugate,
    {
        self.into_const().canonical()
    }

    #[inline(always)]
    pub fn adjoint(self) -> ColRef<'a, T::Conj, Cols, CStride>
    where
        T: Conjugate,
    {
        self.into_const().adjoint()
    }

    #[inline(always)]
    pub fn at(self, col: Idx<Cols>) -> &'a T {
        self.into_const().at(col)
    }

    #[inline(always)]
    pub unsafe fn at_unchecked(self, col: Idx<Cols>) -> &'a T {
        self.into_const().at_unchecked(col)
    }

    #[inline]
    pub fn reverse_cols(self) -> RowRef<'a, T, Cols, CStride::Rev> {
        self.into_const().reverse_cols()
    }

    #[inline]
    pub fn subcols<V: Shape>(self, col_start: IdxInc<Cols>, ncols: V) -> RowRef<'a, T, V, CStride> {
        self.into_const().subcols(col_start, ncols)
    }

    #[inline]
    pub fn subcols_range(
        self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> RowRef<'a, T, usize, CStride> {
        self.into_const().subcols_range(cols)
    }

    #[inline]
    #[track_caller]
    pub fn as_col_shape<V: Shape>(self, ncols: V) -> RowRef<'a, T, V, CStride> {
        self.into_const().as_col_shape(ncols)
    }

    #[inline]
    pub fn as_dyn_cols(self) -> RowRef<'a, T, usize, CStride> {
        self.into_const().as_dyn_cols()
    }

    #[inline]
    pub fn as_dyn_stride(self) -> RowRef<'a, T, Cols, isize> {
        self.into_const().as_dyn_stride()
    }

    #[inline]
    pub fn iter(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a T> {
        self.trans.iter()
    }

    #[inline]
    #[cfg(feature = "rayon")]
    pub fn par_iter(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a T>
    where
        T: Sync,
    {
        self.trans.par_iter()
    }

    #[inline]
    pub fn try_as_row_major(self) -> Option<RowRef<'a, T, Cols, ContiguousFwd>> {
        self.into_const().try_as_row_major()
    }

    #[inline]
    pub fn as_diagonal(self) -> DiagRef<'a, T, Cols, CStride> {
        DiagRef {
            inner: self.trans.into_const(),
        }
    }

    #[inline(always)]
    pub unsafe fn const_cast(self) -> RowMut<'a, T, Cols, CStride> {
        RowMut {
            trans: self.trans.const_cast(),
        }
    }

    #[inline]
    pub fn as_ref(&self) -> RowRef<'_, T, Cols, CStride> {
        self.rb()
    }
    #[inline]
    pub fn as_mut(&mut self) -> RowMut<'_, T, Cols, CStride> {
        self.rb_mut()
    }

    #[inline]
    pub fn as_mat(self) -> MatRef<'a, T, usize, Cols, isize, CStride> {
        self.into_const().as_mat()
    }
    #[inline]
    pub fn as_mat_mut(self) -> MatMut<'a, T, usize, Cols, isize, CStride> {
        unsafe { self.into_const().as_mat().const_cast() }
    }

    #[inline]
    pub fn norm_max(&self) -> Real<T>
    where
        T: Conjugate,
    {
        self.as_ref().transpose().norm_max()
    }

    #[inline]
    pub fn norm_l2(&self) -> Real<T>
    where
        T: Conjugate,
    {
        self.as_ref().transpose().norm_l2()
    }
}

impl<'a, T, Cols: Shape, CStride: Stride> RowMut<'a, T, Cols, CStride> {
    #[inline(always)]
    pub fn as_ptr_mut(&self) -> *mut T {
        self.trans.as_ptr_mut()
    }

    #[inline(always)]
    pub fn ptr_at_mut(&self, col: IdxInc<Cols>) -> *mut T {
        self.trans.ptr_at_mut(col)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(&self, col: Idx<Cols>) -> *mut T {
        debug_assert!(all(col < self.ncols()));
        self.trans.ptr_inbounds_at_mut(col)
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col_mut(
        self,
        col: IdxInc<Cols>,
    ) -> (RowMut<'a, T, usize, CStride>, RowMut<'a, T, usize, CStride>) {
        let (a, b) = self.into_const().split_at_col(col);
        unsafe { (a.const_cast(), b.const_cast()) }
    }

    #[inline(always)]
    pub fn transpose_mut(self) -> ColMut<'a, T, Cols, CStride> {
        self.trans
    }

    #[inline(always)]
    pub fn conjugate_mut(self) -> RowMut<'a, T::Conj, Cols, CStride>
    where
        T: Conjugate,
    {
        unsafe { self.into_const().conjugate().const_cast() }
    }

    #[inline(always)]
    pub fn canonical_mut(self) -> RowMut<'a, T::Canonical, Cols, CStride>
    where
        T: Conjugate,
    {
        unsafe { self.into_const().canonical().const_cast() }
    }

    #[inline(always)]
    pub fn adjoint_mut(self) -> ColMut<'a, T::Conj, Cols, CStride>
    where
        T: Conjugate,
    {
        unsafe { self.into_const().adjoint().const_cast() }
    }

    #[inline(always)]
    pub fn at_mut(self, col: Idx<Cols>) -> &'a mut T {
        assert!(all(col < self.ncols()));
        unsafe { self.at_mut_unchecked(col) }
    }

    #[inline(always)]
    pub unsafe fn at_mut_unchecked(self, col: Idx<Cols>) -> &'a mut T {
        &mut *self.ptr_inbounds_at_mut(col)
    }

    #[inline]
    pub fn reverse_cols_mut(self) -> RowMut<'a, T, Cols, CStride::Rev> {
        unsafe { self.into_const().reverse_cols().const_cast() }
    }

    #[inline]
    pub fn subcols_mut<V: Shape>(
        self,
        col_start: IdxInc<Cols>,
        ncols: V,
    ) -> RowMut<'a, T, V, CStride> {
        unsafe { self.into_const().subcols(col_start, ncols).const_cast() }
    }

    #[inline]
    pub fn subcols_range_mut(
        self,
        cols: (impl Into<IdxInc<Cols>>, impl Into<IdxInc<Cols>>),
    ) -> RowMut<'a, T, usize, CStride> {
        unsafe { self.into_const().subcols_range(cols).const_cast() }
    }

    #[inline]
    #[track_caller]
    pub fn as_col_shape_mut<V: Shape>(self, ncols: V) -> RowMut<'a, T, V, CStride> {
        unsafe { self.into_const().as_col_shape(ncols).const_cast() }
    }

    #[inline]
    pub fn as_dyn_cols_mut(self) -> RowMut<'a, T, usize, CStride> {
        unsafe { self.into_const().as_dyn_cols().const_cast() }
    }

    #[inline]
    pub fn as_dyn_stride_mut(self) -> RowMut<'a, T, Cols, isize> {
        unsafe { self.into_const().as_dyn_stride().const_cast() }
    }

    #[inline]
    pub fn iter_mut(self) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a mut T> {
        self.trans.iter_mut()
    }

    #[inline]
    #[cfg(feature = "rayon")]
    pub fn par_iter_mut(self) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = &'a mut T>
    where
        T: Send,
    {
        self.trans.par_iter_mut()
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_partition(
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowRef<'a, T, usize, CStride>>
    where
        T: Sync,
    {
        self.into_const().par_partition(count)
    }

    #[inline]
    #[track_caller]
    #[cfg(feature = "rayon")]
    pub fn par_partition_mut(
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowMut<'a, T, usize, CStride>>
    where
        T: Send,
    {
        use crate::mat::matmut::SyncCell;
        use rayon::prelude::*;
        unsafe {
            self.as_type::<SyncCell<T>>()
                .into_const()
                .par_partition(count)
                .map(|col| col.const_cast().as_type::<T>())
        }
    }

    pub(crate) unsafe fn as_type<U>(self) -> RowMut<'a, U, Cols, CStride> {
        RowMut::from_raw_parts_mut(self.as_ptr_mut() as *mut U, self.ncols(), self.col_stride())
    }

    #[inline]
    pub fn copy_from_with<RhsT: Conjugate<Canonical = T>>(
        &mut self,
        other: impl AsRowRef<RhsT, Cols>,
    ) where
        T: ComplexField,
    {
        self.rb_mut()
            .transpose_mut()
            .copy_from_with(other.as_row_ref().transpose());
    }

    #[inline]
    pub fn try_as_row_major_mut(self) -> Option<RowMut<'a, T, Cols, ContiguousFwd>> {
        self.into_const()
            .try_as_row_major()
            .map(|x| unsafe { x.const_cast() })
    }

    #[inline]
    pub fn as_diagonal_mut(self) -> DiagMut<'a, T, Cols, CStride> {
        DiagMut { inner: self.trans }
    }

    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.rb_mut().transpose_mut().fill(value)
    }

    #[inline]
    pub(crate) fn __at_mut(self, i: Idx<Cols>) -> &'a mut T {
        self.at_mut(i)
    }
}

impl<'a, T, Rows: Shape> RowMut<'a, T, Rows, ContiguousFwd> {
    #[inline]
    pub fn as_slice(self) -> &'a [T] {
        self.transpose().as_slice()
    }
}

impl<'a, 'ROWS, T> RowMut<'a, T, Dim<'ROWS>, ContiguousFwd> {
    #[inline]
    pub fn as_array(self) -> &'a Array<'ROWS, T> {
        self.transpose().as_array()
    }
}

impl<'a, T, Cols: Shape> RowMut<'a, T, Cols, ContiguousFwd> {
    #[inline]
    pub fn as_slice_mut(self) -> &'a mut [T] {
        self.transpose_mut().as_slice_mut()
    }
}

impl<'a, 'COLS, T> RowMut<'a, T, Dim<'COLS>, ContiguousFwd> {
    #[inline]
    pub fn as_array_mut(self) -> &'a mut Array<'COLS, T> {
        self.transpose_mut().as_array_mut()
    }
}

impl<'COLS, 'a, T, CStride: Stride> RowMut<'a, T, Dim<'COLS>, CStride> {
    #[inline]
    pub fn split_cols_with<'LEFT, 'RIGHT>(
        self,
        col: Partition<'LEFT, 'RIGHT, 'COLS>,
    ) -> (
        RowRef<'a, T, Dim<'LEFT>, CStride>,
        RowRef<'a, T, Dim<'RIGHT>, CStride>,
    ) {
        let (a, b) = self.split_at_col(col.midpoint());
        (a.as_col_shape(col.head), b.as_col_shape(col.tail))
    }
}

impl<'COLS, 'a, T, CStride: Stride> RowMut<'a, T, Dim<'COLS>, CStride> {
    #[inline]
    pub fn split_cols_with_mut<'LEFT, 'RIGHT>(
        self,
        col: Partition<'LEFT, 'RIGHT, 'COLS>,
    ) -> (
        RowMut<'a, T, Dim<'LEFT>, CStride>,
        RowMut<'a, T, Dim<'RIGHT>, CStride>,
    ) {
        let (a, b) = self.split_at_col_mut(col.midpoint());
        (a.as_col_shape_mut(col.head), b.as_col_shape_mut(col.tail))
    }
}

impl<T: core::fmt::Debug, Cols: Shape, CStride: Stride> core::fmt::Debug
    for RowMut<'_, T, Cols, CStride>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<T, Cols: Shape, RStride: Stride> Index<Idx<Cols>> for RowRef<'_, T, Cols, RStride> {
    type Output = T;

    #[inline]
    fn index(&self, col: Idx<Cols>) -> &Self::Output {
        self.at(col)
    }
}

impl<T, Cols: Shape, RStride: Stride> Index<Idx<Cols>> for RowMut<'_, T, Cols, RStride> {
    type Output = T;

    #[inline]
    fn index(&self, col: Idx<Cols>) -> &Self::Output {
        self.rb().at(col)
    }
}

impl<T, Cols: Shape, RStride: Stride> IndexMut<Idx<Cols>> for RowMut<'_, T, Cols, RStride> {
    #[inline]
    fn index_mut(&mut self, col: Idx<Cols>) -> &mut Self::Output {
        self.rb_mut().at_mut(col)
    }
}

mod bound_range {
    use super::*;
    use crate::utils::bound::Segment;

    // single segment

    impl<'COLS, 'a, T, RStride: Stride> RowMut<'a, T, Dim<'COLS>, RStride> {
        #[inline]
        pub fn col_segment_mut<'scope, 'LEFT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
        ) -> RowMut<'a, T, Dim<'LEFT>, RStride> {
            unsafe {
                RowMut::from_raw_parts_mut(
                    self.ptr_at_mut(first.start().local()),
                    first.len(),
                    self.col_stride(),
                )
            }
        }
    }
    impl<'COLS, 'a, T, RStride: Stride> RowRef<'a, T, Dim<'COLS>, RStride> {
        #[inline]
        pub fn col_segment<'scope, 'LEFT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
        ) -> RowRef<'a, T, Dim<'LEFT>, RStride> {
            unsafe {
                RowRef::from_raw_parts(
                    self.ptr_at(first.start().local()),
                    first.len(),
                    self.col_stride(),
                )
            }
        }
    }

    impl<'COLS, 'a, T, RStride: Stride> RowMut<'a, T, Dim<'COLS>, RStride> {
        #[inline]
        pub fn col_segment<'scope, 'LEFT>(
            self,
            first: Segment<'scope, 'COLS, 'LEFT>,
        ) -> RowRef<'a, T, Dim<'LEFT>, RStride> {
            unsafe {
                RowRef::from_raw_parts(
                    self.ptr_at(first.start().local()),
                    first.len(),
                    self.col_stride(),
                )
            }
        }
    }
}

mod bound_any_range {
    use super::*;
    use crate::variadics::*;

    impl<'COLS, 'a, T, CStride: Stride> RowMut<'a, T, Dim<'COLS>, CStride> {
        #[inline]
        pub fn col_segments_mut<'scope, S: ColSplit<'scope, 'COLS, 'a>>(
            self,
            segments: S,
            disjoint: S::Disjoint,
        ) -> S::RowMutSegments<T, CStride> {
            S::row_mut_segments(segments, self, disjoint)
        }
    }

    impl<'COLS, 'a, T, CStride: Stride> RowRef<'a, T, Dim<'COLS>, CStride> {
        #[inline]
        pub fn col_segments<'scope, S: ColSplit<'scope, 'COLS, 'a>>(
            self,
            segments: S,
        ) -> S::RowRefSegments<T, CStride> {
            S::row_ref_segments(segments, self)
        }
    }

    impl<'COLS, 'a, T, CStride: Stride> RowMut<'a, T, Dim<'COLS>, CStride> {
        #[inline]
        pub fn col_segments<'scope, S: ColSplit<'scope, 'COLS, 'a>>(
            self,
            segments: S,
        ) -> S::RowRefSegments<T, CStride> {
            S::row_ref_segments(segments, self.into_const())
        }
    }
}
