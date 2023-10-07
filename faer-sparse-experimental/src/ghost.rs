use crate::{mem::*, Index, __get_unchecked};
use assert2::{assert, debug_assert};
use bytemuck::Pod;
use core::{marker::PhantomData, ops::Deref};
use faer_core::Entity;
use reborrow::*;
use std::ops::Range;

pub type InvariantLifetime<'a> = PhantomData<fn(&'a ()) -> &'a ()>;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Branded<'brand, T: ?Sized> {
    __marker: InvariantLifetime<'brand>,
    unbranded: T,
}

impl<T: ?Sized + core::fmt::Debug> core::fmt::Debug for Branded<'_, T> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.unbranded.fmt(f)
    }
}

impl<'brand, T: ?Sized> Deref for Branded<'brand, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.unbranded
    }
}

impl<'brand, T: ?Sized> Branded<'brand, T> {
    #[inline]
    pub unsafe fn from_ref_unchecked(unbranded: &T) -> &Branded<'brand, T> {
        &*{ unbranded as *const T as *const Branded<'brand, T> }
    }

    #[inline]
    pub unsafe fn from_mut_unchecked(unbranded: &mut T) -> &mut Branded<'brand, T> {
        &mut *{ unbranded as *mut T as *mut Branded<'brand, T> }
    }

    #[inline]
    pub unsafe fn new_unchecked(unbranded: T) -> Branded<'brand, T>
    where
        T: Sized,
    {
        Branded {
            __marker: PhantomData,
            unbranded,
        }
    }

    #[inline]
    pub unsafe fn into_inner(self) -> T
    where
        T: Sized,
    {
        self.unbranded
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Size<'n>(Branded<'n, usize>);

impl<'n> Size<'n> {
    #[inline]
    pub fn raw_unchecked(idx: usize) -> Self {
        Self(unsafe { Branded::new_unchecked(idx) })
    }

    #[inline]
    pub fn indices(self) -> impl ExactSizeIterator + DoubleEndedIterator<Item = Idx<'n>> {
        (0..*self).map(
            #[inline(always)]
            |idx| unsafe { Idx::raw_unchecked(idx) },
        )
    }

    #[inline]
    pub fn check(self, idx: usize) -> Idx<'n> {
        Idx::new_checked(idx, self)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Idx<'n, I = usize>(Branded<'n, I>);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct MaybeIdx<'n, I = usize>(Branded<'n, I>);

impl<'n, I: Index> MaybeIdx<'n, I> {
    #[inline]
    #[track_caller]
    pub fn new_index_checked(idx: I, size: Size<'n>) -> Self {
        if idx == I::truncate(crate::NONE) {
            Self::none_index()
        } else {
            Self::from_index(Idx::new_index_checked(idx, size))
        }
    }

    #[inline]
    pub fn none_index() -> Self {
        unsafe { MaybeIdx(Branded::new_unchecked(I::truncate(crate::NONE))) }
    }

    #[inline]
    pub fn idx(self) -> Option<Idx<'n, I>> {
        if *self != I::truncate(crate::NONE) {
            unsafe { Some(Idx::raw_index_unchecked(*self)) }
        } else {
            None
        }
    }

    #[inline]
    pub fn sx(self) -> MaybeIdx<'n, usize> {
        unsafe { MaybeIdx(Branded::new_unchecked((*self).sx())) }
    }

    #[inline]
    #[track_caller]
    pub unsafe fn slice_ref_unchecked<'a>(idx: &'a [I], size: Size<'n>) -> &'a [MaybeIdx<'n, I>] {
        let _ = size;
        #[cfg(debug_assertions)]
        for &i in idx {
            Self::new_index_checked(i, size);
        }
        transmute_slice_ref(idx)
    }

    #[inline]
    #[track_caller]
    pub fn slice_ref_checked<'a>(idx: &'a [I], size: Size<'n>) -> &'a [MaybeIdx<'n, I>] {
        for &i in idx {
            Self::new_index_checked(i, size);
        }
        // SAFETY: we checked all the indices
        unsafe { transmute_slice_ref(idx) }
    }

    #[inline]
    #[track_caller]
    pub unsafe fn slice_mut_unchecked<'a>(
        idx: &'a mut [I],
        size: Size<'n>,
    ) -> &'a mut [MaybeIdx<'n, I>] {
        let _ = size;
        #[cfg(debug_assertions)]
        for &i in &*idx {
            Self::new_index_checked(i, size);
        }
        unsafe { transmute_slice_mut(idx) }
    }

    #[inline]
    #[track_caller]
    pub fn slice_mut_checked<'a>(idx: &'a mut [I], size: Size<'n>) -> &'a mut [MaybeIdx<'n, I>] {
        for &i in &*idx {
            Self::new_index_checked(i, size);
        }
        // SAFETY: we checked all the indices
        unsafe { transmute_slice_mut(idx) }
    }

    #[inline]
    pub fn as_inner_slice_ref<'a>(idx: &'a [MaybeIdx<'n, I>]) -> &'a [I] {
        unsafe { transmute_slice_ref(idx) }
    }
}
impl<'n> MaybeIdx<'n> {
    #[inline]
    pub fn none() -> Self {
        unsafe { MaybeIdx(Branded::new_unchecked(crate::NONE)) }
    }

    #[inline]
    pub fn idx(self) -> Option<Idx<'n>> {
        if *self != crate::NONE {
            unsafe { Some(Idx::raw_unchecked(*self)) }
        } else {
            None
        }
    }
}
impl<'n, I> MaybeIdx<'n, I> {
    #[inline]
    pub fn from_index(value: Idx<'n, I>) -> Self {
        Self(value.0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct IdxInclusive<'n, I = usize>(Branded<'n, I>);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Array<'n, I>(Branded<'n, [I]>);

impl<'n, I> Array<'n, I> {
    #[inline]
    pub fn len(&self) -> Size<'n> {
        Size::raw_unchecked(self.0.unbranded.len())
    }
}

impl<'n, I> Deref for Array<'n, I> {
    type Target = [I];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0.unbranded
    }
}

impl<'n, I> core::ops::DerefMut for Array<'n, I> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.unbranded
    }
}

#[inline(always)]
pub fn with_size<R>(n: usize, f: impl for<'n> FnOnce(Size<'n>) -> R) -> R {
    f(Size(unsafe { Branded::new_unchecked(n) }))
}

impl<'n, I: Index> Idx<'n, I> {
    #[inline]
    #[track_caller]
    pub unsafe fn raw_index_unchecked(idx: I) -> Self {
        debug_assert!(idx >= I::truncate(0));
        Self(unsafe { Branded::new_unchecked(idx) })
    }

    #[inline]
    #[track_caller]
    pub fn new_index_unchecked(idx: I, size: Size<'n>) -> Self {
        debug_assert!(idx >= I::truncate(0));
        debug_assert!(idx.zx() < *size.0);
        Self(unsafe { Branded::new_unchecked(idx) })
    }

    #[inline]
    #[track_caller]
    pub fn new_index_checked(idx: I, size: Size<'n>) -> Self {
        assert!(idx >= I::truncate(0));
        assert!(idx.zx() < *size.0);
        Self(unsafe { Branded::new_unchecked(idx) })
    }

    #[inline]
    pub fn zx(self) -> Idx<'n> {
        unsafe { Idx(Branded::new_unchecked(self.0.zx())) }
    }

    #[inline]
    pub fn as_inner_slice_ref<'a>(idx: &'a [Idx<'n, I>]) -> &'a [I] {
        unsafe { transmute_slice_ref(idx) }
    }

    #[inline]
    #[track_caller]
    pub unsafe fn slice_ref_unchecked<'a>(idx: &'a [I], size: Size<'n>) -> &'a [Idx<'n, I>] {
        let _ = size;
        #[cfg(debug_assertions)]
        for &i in idx {
            Self::new_index_checked(i, size);
        }
        transmute_slice_ref(idx)
    }

    #[inline]
    #[track_caller]
    pub fn slice_ref_checked<'a>(idx: &'a [I], size: Size<'n>) -> &'a [Idx<'n, I>] {
        for &i in idx {
            Self::new_index_checked(i, size);
        }
        // SAFETY: we checked all the indices
        unsafe { transmute_slice_ref(idx) }
    }

    #[inline]
    #[track_caller]
    pub unsafe fn slice_mut_unchecked<'a>(
        idx: &'a mut [I],
        size: Size<'n>,
    ) -> &'a mut [Idx<'n, I>] {
        let _ = size;
        #[cfg(debug_assertions)]
        for &i in &*idx {
            Self::new_index_checked(i, size);
        }
        unsafe { transmute_slice_mut(idx) }
    }

    #[inline]
    #[track_caller]
    pub fn slice_mut_checked<'a>(idx: &'a mut [I], size: Size<'n>) -> &'a mut [Idx<'n, I>] {
        for &i in &*idx {
            Self::new_index_checked(i, size);
        }
        // SAFETY: we checked all the indices
        unsafe { transmute_slice_mut(idx) }
    }
}

impl<'n> Idx<'n, usize> {
    #[inline]
    pub unsafe fn raw_unchecked(idx: usize) -> Self {
        Self(unsafe { Branded::new_unchecked(idx) })
    }

    #[inline]
    #[track_caller]
    pub unsafe fn new_unchecked(idx: usize, size: Size<'n>) -> Self {
        debug_assert!(idx < *size.0);
        Self::raw_unchecked(idx)
    }

    #[inline]
    #[track_caller]
    pub fn new_checked(idx: usize, size: Size<'n>) -> Self {
        assert!(idx < *size.0);
        unsafe { Self::raw_unchecked(idx) }
    }

    #[inline]
    pub fn next(self) -> IdxInclusive<'n> {
        unsafe { IdxInclusive::raw_unchecked(*self + 1) }
    }

    #[inline]
    pub fn truncate<I: Index>(self) -> Idx<'n, I> {
        unsafe { Idx::raw_index_unchecked(I::truncate(*self)) }
    }
}

impl<'n, I: Index> IdxInclusive<'n, I> {
    #[inline]
    #[track_caller]
    pub unsafe fn raw_index_unchecked(idx: I) -> Self {
        debug_assert!(idx >= I::truncate(0));
        Self(unsafe { Branded::new_unchecked(idx) })
    }

    #[inline]
    #[track_caller]
    pub unsafe fn new_index_unchecked(idx: I, size: Size<'n>) -> Self {
        debug_assert!(idx >= I::truncate(0));
        debug_assert!(idx.zx() <= *size.0);
        Self(unsafe { Branded::new_unchecked(idx) })
    }

    #[inline]
    #[track_caller]
    pub fn new_index_checked(idx: I, size: Size<'n>) -> Self {
        assert!(idx >= I::truncate(0));
        assert!(idx.zx() <= *size.0);
        Self(unsafe { Branded::new_unchecked(idx) })
    }

    #[inline]
    pub fn zx(self) -> IdxInclusive<'n> {
        unsafe { IdxInclusive(Branded::new_unchecked(self.0.zx())) }
    }

    #[inline]
    #[track_caller]
    pub unsafe fn slice_ref_unchecked<'a>(
        idx: &'a [I],
        size: Size<'n>,
    ) -> &'a [IdxInclusive<'n, I>] {
        let _ = size;
        #[cfg(debug_assertions)]
        for &i in idx {
            Self::new_index_checked(i, size);
        }
        transmute_slice_ref(idx)
    }

    #[inline]
    #[track_caller]
    pub fn slice_ref_checked<'a>(idx: &'a [I], size: Size<'n>) -> &'a [IdxInclusive<'n, I>] {
        for &i in idx {
            Self::new_index_checked(i, size);
        }
        // SAFETY: we checked all the indices
        unsafe { transmute_slice_ref(idx) }
    }

    #[inline]
    #[track_caller]
    pub unsafe fn slice_mut_unchecked<'a>(
        idx: &'a mut [I],
        size: Size<'n>,
    ) -> &'a mut [IdxInclusive<'n, I>] {
        let _ = size;
        #[cfg(debug_assertions)]
        for &i in &*idx {
            Self::new_index_checked(i, size);
        }
        unsafe { transmute_slice_mut(idx) }
    }

    #[inline]
    #[track_caller]
    pub fn slice_mut_checked<'a>(
        idx: &'a mut [I],
        size: Size<'n>,
    ) -> &'a mut [IdxInclusive<'n, I>] {
        for &i in &*idx {
            Self::new_index_checked(i, size);
        }
        // SAFETY: we checked all the indices
        unsafe { transmute_slice_mut(idx) }
    }
}

impl<'n, I> Idx<'n, I> {
    #[inline]
    pub fn to_inclusive(self) -> IdxInclusive<'n, I> {
        IdxInclusive(self.0)
    }
}

impl<'n> IdxInclusive<'n, usize> {
    #[inline]
    pub unsafe fn raw_unchecked(idx: usize) -> Self {
        Self(unsafe { Branded::new_unchecked(idx) })
    }

    #[inline]
    #[track_caller]
    pub fn new_checked(idx: usize, size: Size<'n>) -> Self {
        assert!(idx <= *size.0);
        unsafe { Self::raw_unchecked(idx) }
    }

    #[inline]
    pub fn truncate<I: Index>(self) -> IdxInclusive<'n, I> {
        unsafe { IdxInclusive::raw_index_unchecked(I::truncate(*self)) }
    }

    #[inline]
    pub fn range_to(self, last: Self) -> impl DoubleEndedIterator<Item = Idx<'n>> {
        (*self..*last).map(
            #[inline(always)]
            |idx| unsafe { Idx::raw_unchecked(idx) },
        )
    }
}

impl<'n, I> Array<'n, I> {
    #[inline]
    pub unsafe fn from_ref_unchecked(slice: &[I]) -> &Self {
        &*(Branded::from_ref_unchecked(slice) as *const _ as *const Self)
    }

    #[inline]
    pub unsafe fn from_mut_unchecked(slice: &mut [I]) -> &mut Self {
        &mut *(Branded::from_mut_unchecked(slice) as *mut _ as *mut Self)
    }

    #[inline]
    #[track_caller]
    pub fn from_ref<'a>(slice: &'a [I], size: Size<'n>) -> &'a Self {
        assert!(slice.len() == *size);
        unsafe { Self::from_ref_unchecked(slice) }
    }

    #[inline]
    #[track_caller]
    pub fn from_mut<'a>(slice: &'a mut [I], size: Size<'n>) -> &'a mut Self {
        assert!(slice.len() == *size);
        unsafe { Self::from_mut_unchecked(slice) }
    }
}

impl<'n, I> core::ops::Index<Idx<'n>> for Array<'n, I> {
    type Output = I;
    #[inline]
    #[track_caller]
    fn index(&self, index: Idx<'n>) -> &Self::Output {
        unsafe { crate::__get_unchecked(&self.0.unbranded, *index) }
    }
}
impl<'n, I> core::ops::IndexMut<Idx<'n>> for Array<'n, I> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, index: Idx<'n>) -> &mut Self::Output {
        unsafe { crate::mem::__get_unchecked_mut(&mut self.0.unbranded, *index) }
    }
}

impl<'n, I> core::ops::Index<Range<IdxInclusive<'n>>> for Array<'n, I> {
    type Output = [I];
    #[inline]
    #[track_caller]
    fn index(&self, index: Range<IdxInclusive<'n>>) -> &Self::Output {
        let Range { start, end } = index;
        unsafe { crate::__get_unchecked(&self.0.unbranded, *start..Ord::max(*start, *end)) }
    }
}
impl<'n, I> core::ops::IndexMut<Range<IdxInclusive<'n>>> for Array<'n, I> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, index: Range<IdxInclusive<'n>>) -> &mut Self::Output {
        let Range { start, end } = index;
        unsafe {
            crate::mem::__get_unchecked_mut(&mut self.0.unbranded, *start..Ord::max(*start, *end))
        }
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct PermutationRef<'n, 'a, I>(Branded<'n, crate::PermutationRef<'a, I>>);

#[derive(Debug)]
#[repr(transparent)]
pub struct SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I>(
    Branded<'nrows, crate::SymbolicSparseColMatRef<'a, I>>,
    Branded<'ncols, ()>,
);

// #[derive(Debug)]
#[repr(transparent)]
pub struct SparseColMatRef<'nrows, 'ncols, 'a, I, E: Entity>(
    Branded<'nrows, crate::SparseColMatRef<'a, I, E>>,
    Branded<'ncols, ()>,
);

impl<'n, 'a, I: Index> PermutationRef<'n, 'a, I> {
    #[inline]
    #[track_caller]
    pub fn new(perm: crate::PermutationRef<'a, I>, size: Size<'n>) -> Self {
        let (fwd, inv) = perm.fwd_inv();
        assert!(fwd.len() == *size);
        assert!(inv.len() == *size);
        Self(unsafe { Branded::new_unchecked(perm) })
    }

    #[inline]
    pub fn fwd_inv(self) -> (&'a Array<'n, Idx<'n, I>>, &'a Array<'n, Idx<'n, I>>) {
        unsafe {
            let (fwd, inv) = self.0.unbranded.fwd_inv();
            let fwd = &*(fwd as *const [I] as *const Array<'n, Idx<'n, I>>);
            let inv = &*(inv as *const [I] as *const Array<'n, Idx<'n, I>>);
            (fwd, inv)
        }
    }
}

impl<'nrows, 'ncols, 'a, I: Index> SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I> {
    #[inline]
    #[track_caller]
    pub fn new(
        mat: crate::SymbolicSparseColMatRef<'a, I>,
        nrows: Size<'nrows>,
        ncols: Size<'ncols>,
    ) -> Self {
        assert!(mat.nrows() == *nrows);
        assert!(mat.ncols() == *ncols);
        unsafe { Self::raw_unchecked(mat) }
    }

    #[inline]
    pub unsafe fn raw_unchecked(mat: crate::SymbolicSparseColMatRef<'a, I>) -> Self {
        unsafe { Self(Branded::new_unchecked(mat), Branded::new_unchecked(())) }
    }

    #[inline]
    pub fn nrows(&self) -> Size<'nrows> {
        Size::raw_unchecked(self.0.nrows())
    }
    #[inline]
    pub fn ncols(&self) -> Size<'ncols> {
        Size::raw_unchecked(self.0.ncols())
    }

    #[inline]
    pub fn row_indices_of_col_raw(&self, j: Idx<'ncols>) -> &[Idx<'nrows, I>] {
        unsafe {
            Idx::slice_ref_unchecked(
                __get_unchecked(self.0.row_indices(), self.0.col_range_unchecked(*j)),
                self.nrows(),
            )
        }
    }

    #[inline]
    pub fn row_indices_of_col(
        &self,
        j: Idx<'ncols>,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = Idx<'nrows>> {
        unsafe {
            __get_unchecked(self.0.row_indices(), self.0.col_range_unchecked(*j))
                .iter()
                .map(
                    #[inline(always)]
                    move |&row| Idx::raw_unchecked(row.zx()),
                )
        }
    }

    #[inline]
    pub fn compressed_row_indices(
        &self,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = Idx<'nrows>> {
        assert!(self.nnz_per_col().is_none());
        unsafe {
            let start = self.col_ptrs()[0].zx();
            let end = self.col_ptrs()[*self.ncols()].zx();
            self.row_indices()[start..end]
                .iter()
                .map(move |&i| Idx::raw_unchecked(i.zx()))
        }
    }
}

impl<'nrows, 'ncols, 'a, I: Index, E: Entity> SparseColMatRef<'nrows, 'ncols, 'a, I, E> {
    #[inline]
    #[track_caller]
    pub fn new(
        mat: crate::SparseColMatRef<'a, I, E>,
        nrows: Size<'nrows>,
        ncols: Size<'ncols>,
    ) -> Self {
        assert!(mat.nrows() == *nrows);
        assert!(mat.ncols() == *ncols);
        unsafe { Self(Branded::new_unchecked(mat), Branded::new_unchecked(())) }
    }

    #[inline]
    pub fn symbolic(self) -> SymbolicSparseColMatRef<'nrows, 'ncols, 'a, I> {
        unsafe { SymbolicSparseColMatRef::raw_unchecked(self.0.symbolic()) }
    }

    #[inline]
    pub fn nrows(&self) -> Size<'nrows> {
        Size::raw_unchecked(self.0.nrows())
    }
    #[inline]
    pub fn ncols(&self) -> Size<'ncols> {
        Size::raw_unchecked(self.0.ncols())
    }

    #[inline]
    pub fn row_indices_of_col(
        &self,
        j: Idx<'ncols>,
    ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = Idx<'nrows>> {
        self.symbolic().row_indices_of_col(j)
    }

    #[inline]
    pub fn values_of_col(&self, j: Idx<'ncols>) -> E::Group<&[E::Unit]> {
        let range = unsafe { self.col_range_unchecked(*j) };
        E::map(
            self.0.values(),
            #[inline(always)]
            |val| unsafe { crate::mem::__get_unchecked(val, range.clone()) },
        )
    }
}

impl_copy!(<><I> <PermutationRef<'_, '_, I>>);
impl_copy!(<><I> <SymbolicSparseColMatRef<'_, '_, '_, I>>);
impl_copy!(<><I, E: Entity> <SparseColMatRef<'_, '_, '_, I, E>>);
impl_deref!(<><><Target = usize><Size<'_>>);
impl_deref!(<><I><Target = I><Idx<'_, I>>);
impl_deref!(<><I><Target = I><MaybeIdx<'_, I>>);
impl_deref!(<><I><Target = I><IdxInclusive<'_, I>>);
impl_deref!(<'n, 'a><I><Target = crate::PermutationRef<'a, I>><PermutationRef<'n, 'a, I>>);
impl_deref!(<'m, 'n, 'a><I><Target = crate::SymbolicSparseColMatRef<'a, I>><SymbolicSparseColMatRef<'m, 'n, 'a, I>>);
impl_deref!(<'m, 'n, 'a><I, E: Entity><Target = crate::SparseColMatRef<'a, I, E>><SparseColMatRef<'m, 'n, 'a, I, E>>);

pub struct ArrayGroup<'n, 'a, E: Entity>(pub E::GroupCopy<&'a Array<'n, E::Unit>>);
pub struct ArrayGroupMut<'n, 'a, E: Entity>(pub E::Group<&'a mut Array<'n, E::Unit>>);

impl<'n, 'a, E: Entity> Copy for ArrayGroup<'n, 'a, E> {}
impl<'n, 'a, E: Entity> Clone for ArrayGroup<'n, 'a, E> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, 'n, 'a, E: Entity> reborrow::ReborrowMut<'short> for ArrayGroupMut<'n, 'a, E> {
    type Target = ArrayGroupMut<'n, 'short, E>;

    #[inline(always)]
    fn rb_mut(&'short mut self) -> Self::Target {
        ArrayGroupMut(E::map(E::as_mut(&mut self.0), |this| &mut **this))
    }
}

impl<'short, 'n, 'a, E: Entity> reborrow::Reborrow<'short> for ArrayGroupMut<'n, 'a, E> {
    type Target = ArrayGroup<'n, 'short, E>;

    #[inline(always)]
    fn rb(&'short self) -> Self::Target {
        ArrayGroup(E::into_copy(E::map(E::as_ref(&self.0), |this| &**this)))
    }
}

impl<'n, 'a, E: Entity> ArrayGroupMut<'n, 'a, E>
where
    E::Unit: Pod,
{
    #[inline(always)]
    pub fn read(&self, j: Idx<'n>) -> E {
        self.rb().read(j)
    }

    #[inline(always)]
    pub fn write(&mut self, j: Idx<'n>, value: E) {
        E::map(
            E::zip(self.rb_mut().0, value.into_units()),
            #[inline(always)]
            |(array, value)| array[j] = value,
        );
    }

    #[inline(always)]
    pub fn fill_zero(&mut self) {
        E::map(
            self.rb_mut().0,
            #[inline(always)]
            |array| crate::mem::fill_zero(array),
        );
    }
}

impl<'n, 'a, E: Entity> ArrayGroup<'n, 'a, E>
where
    E::Unit: Pod,
{
    #[inline(always)]
    pub fn read(&self, j: Idx<'n>) -> E {
        E::from_units(E::map(
            E::from_copy(self.0),
            #[inline(always)]
            |array| array[j],
        ))
    }
}

#[inline]
pub fn fill_zero<'n, 'a, I: Index>(slice: &'a mut [I], size: Size<'n>) -> &'a mut [Idx<'n, I>] {
    let len = slice.len();
    if len > 0 {
        assert!(*size > 0);
    }
    unsafe {
        core::ptr::write_bytes(slice.as_mut_ptr(), 0u8, len);
        Idx::slice_mut_unchecked(slice, size)
    }
}

#[inline]
pub fn fill_none<'n, 'a, I: Index>(
    slice: &'a mut [I],
    size: Size<'n>,
) -> &'a mut [MaybeIdx<'n, I>] {
    let _ = size;
    let len = slice.len();
    unsafe { core::ptr::write_bytes(slice.as_mut_ptr(), NONE_BYTE, len) };
    unsafe { transmute_slice_mut(slice) }
}

#[inline]
pub fn copy_slice<'n, 'a, I: Index>(dst: &'a mut [I], src: &[Idx<'n, I>]) -> &'a mut [Idx<'n, I>] {
    let size = Size::<'n>::raw_unchecked(usize::MAX);
    let dst = unsafe { Idx::slice_mut_unchecked(dst, size) };
    dst.copy_from_slice(src);
    dst
}

#[inline]
pub fn set_identity<'out, 'n, I: Index>(
    slice: &'out mut Array<'n, I>,
) -> &'out mut Array<'n, Idx<'n, I>> {
    let N = slice.len();
    for (i, pi) in slice.iter_mut().enumerate() {
        *pi = I::truncate(i);
    }
    Array::from_mut(unsafe { Idx::slice_mut_unchecked(slice, N) }, N)
}
