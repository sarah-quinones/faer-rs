use super::*;
use crate::{assert, debug_assert, Index, SignedIndex};
use core::{fmt::Debug, marker::PhantomData, ops::Range};
use faer_entity::*;
use reborrow::*;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
struct Branded<'a, T: ?Sized> {
    __marker: PhantomData<fn(&'a ()) -> &'a ()>,
    inner: T,
}

/// `usize` value tied to the lifetime `'n`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Size<'n>(Branded<'n, usize>);

/// `I` value smaller than the size corresponding to the lifetime `'n`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Idx<'n, I>(Branded<'n, I>);

/// `I` value smaller or equal to the size corresponding to the lifetime `'n`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct IdxInclusive<'n, I>(Branded<'n, I>);

/// `I` value smaller than the size corresponding to the lifetime `'n`, or `None`.
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct MaybeIdx<'n, I: Index>(Branded<'n, I>);

impl core::ops::Deref for Size<'_> {
    type Target = usize;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0.inner
    }
}
impl<I> core::ops::Deref for Idx<'_, I> {
    type Target = I;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0.inner
    }
}
impl<I: Index> core::ops::Deref for MaybeIdx<'_, I> {
    type Target = I::Signed;
    #[inline]
    fn deref(&self) -> &Self::Target {
        bytemuck::cast_ref(&self.0.inner)
    }
}
impl<I> core::ops::Deref for IdxInclusive<'_, I> {
    type Target = I;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0.inner
    }
}

/// Array of length equal to the value tied to `'n`.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Array<'n, T>(Branded<'n, [T]>);

pub mod mat;
/// Permutations with compile-time checks.
pub mod perm;

pub mod sparse;

/// Immutable array group of length equal to the value tied to `'n`.
pub struct ArrayGroup<'n, 'a, E: Entity>(Branded<'n, slice::SliceGroup<'a, E>>);
/// Mutable array group of length equal to the value tied to `'n`.
pub struct ArrayGroupMut<'n, 'a, E: Entity>(Branded<'n, slice::SliceGroupMut<'a, E>>);

impl<E: Entity> Debug for ArrayGroup<'_, '_, E> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.inner.fmt(f)
    }
}
impl<E: Entity> Debug for ArrayGroupMut<'_, '_, E> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.inner.fmt(f)
    }
}

impl<E: Entity> Copy for ArrayGroup<'_, '_, E> {}
impl<E: Entity> Clone for ArrayGroup<'_, '_, E> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, 'n, 'a, E: Entity> reborrow::ReborrowMut<'short> for ArrayGroup<'n, 'a, E> {
    type Target = ArrayGroup<'n, 'short, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'short, 'n, 'a, E: Entity> reborrow::Reborrow<'short> for ArrayGroup<'n, 'a, E> {
    type Target = ArrayGroup<'n, 'short, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, 'n, 'a, E: Entity> reborrow::ReborrowMut<'short> for ArrayGroupMut<'n, 'a, E> {
    type Target = ArrayGroupMut<'n, 'short, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        ArrayGroupMut(Branded {
            __marker: PhantomData,
            inner: self.0.inner.rb_mut(),
        })
    }
}

impl<'short, 'n, 'a, E: Entity> reborrow::Reborrow<'short> for ArrayGroupMut<'n, 'a, E> {
    type Target = ArrayGroup<'n, 'short, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        ArrayGroup(Branded {
            __marker: PhantomData,
            inner: self.0.inner.rb(),
        })
    }
}

impl<'n, 'a, E: Entity> ArrayGroupMut<'n, 'a, E> {
    /// Returns an array group with length after checking that its length matches
    /// the value tied to `'n`.
    #[inline]
    pub fn new(slice: GroupFor<E, &'a mut [E::Unit]>, len: Size<'n>) -> Self {
        let slice = slice::SliceGroupMut::<'_, E>::new(slice);
        assert!(slice.rb().len() == len.into_inner());
        ArrayGroupMut(Branded {
            __marker: PhantomData,
            inner: slice,
        })
    }

    /// Returns the unconstrained slice.
    #[inline]
    pub fn into_slice(self) -> GroupFor<E, &'a mut [E::Unit]> {
        self.0.inner.into_inner()
    }

    /// Returns a subslice at from the range start to its end.
    #[inline]
    pub fn subslice(self, range: Range<IdxInclusive<'n, usize>>) -> GroupFor<E, &'a mut [E::Unit]> {
        unsafe {
            slice::SliceGroupMut::<'_, E>::new(self.into_slice())
                .subslice_unchecked(range.start.into_inner()..range.end.into_inner())
                .into_inner()
        }
    }

    /// Read the element at position `j`.
    #[inline]
    pub fn read(&self, j: Idx<'n, usize>) -> E {
        self.rb().read(j)
    }

    /// Write `value` to the location at position `j`.
    #[inline]
    pub fn write(&mut self, j: Idx<'n, usize>, value: E) {
        unsafe {
            slice::SliceGroupMut::new(self.rb_mut().into_slice())
                .write_unchecked(j.into_inner(), value)
        }
    }
}

impl<'n, 'a, E: Entity> ArrayGroup<'n, 'a, E> {
    /// Returns an array group with length after checking that its length matches
    /// the value tied to `'n`.
    #[inline]
    pub fn new(slice: GroupFor<E, &'a [E::Unit]>, len: Size<'n>) -> Self {
        let slice = slice::SliceGroup::<'_, E>::new(slice);
        assert!(slice.rb().len() == len.into_inner());
        ArrayGroup(Branded {
            __marker: PhantomData,
            inner: slice,
        })
    }

    /// Returns the unconstrained slice.
    #[inline]
    pub fn into_slice(self) -> GroupFor<E, &'a [E::Unit]> {
        self.0.inner.into_inner()
    }

    /// Returns a subslice at from the range start to its end.
    #[inline]
    pub fn subslice(self, range: Range<IdxInclusive<'n, usize>>) -> GroupFor<E, &'a [E::Unit]> {
        unsafe {
            slice::SliceGroup::<'_, E>::new(self.into_slice())
                .subslice_unchecked(range.start.into_inner()..range.end.into_inner())
                .into_inner()
        }
    }

    /// Read the element at position `j`.
    #[inline]
    pub fn read(&self, j: Idx<'n, usize>) -> E {
        unsafe { slice::SliceGroup::new(self.into_slice()).read_unchecked(j.into_inner()) }
    }
}

impl<'size> Size<'size> {
    /// Create a new [`Size`] with a lifetime tied to `n`.
    #[track_caller]
    #[inline]
    pub fn with<R>(n: usize, f: impl for<'n> FnOnce(Size<'n>) -> R) -> R {
        f(Size(Branded {
            __marker: PhantomData,
            inner: n,
        }))
    }

    /// Create two new [`Size`] with lifetimes tied to `m` and `n`.
    #[track_caller]
    #[inline]
    pub fn with2<R>(m: usize, n: usize, f: impl for<'m, 'n> FnOnce(Size<'m>, Size<'n>) -> R) -> R {
        f(
            Size(Branded {
                __marker: PhantomData,
                inner: m,
            }),
            Size(Branded {
                __marker: PhantomData,
                inner: n,
            }),
        )
    }

    /// Create a new [`Size`] tied to the lifetime `'n`.
    #[inline]
    pub unsafe fn new_raw_unchecked(n: usize) -> Self {
        Size(Branded {
            __marker: PhantomData,
            inner: n,
        })
    }

    /// Returns the unconstrained value.
    #[inline]
    pub fn into_inner(self) -> usize {
        self.0.inner
    }

    /// Returns an iterator of the indices smaller than `self`.
    #[inline]
    pub fn indices(self) -> impl DoubleEndedIterator<Item = Idx<'size, usize>> {
        (0..self.0.inner).map(|i| unsafe { Idx::new_raw_unchecked(i) })
    }

    /// Check that the index is bounded by `self`, or panic otherwise.
    #[track_caller]
    #[inline]
    pub fn check<I: Index>(self, idx: I) -> Idx<'size, I> {
        Idx::new_checked(idx, self)
    }

    /// Check that the index is bounded by `self`, or return `None` otherwise.
    #[inline]
    pub fn try_check<I: Index>(self, idx: I) -> Option<Idx<'size, I>> {
        if idx.zx() < self.into_inner() {
            Some(Idx(Branded {
                __marker: PhantomData,
                inner: idx,
            }))
        } else {
            None
        }
    }
}

impl<'n> Idx<'n, usize> {
    /// Truncate `self` to a smaller type `I`.
    pub fn truncate<I: Index>(self) -> Idx<'n, I> {
        unsafe { Idx::new_raw_unchecked(I::truncate(self.into_inner())) }
    }
}

impl<'n, I: Index> Idx<'n, I> {
    /// Returns a new index after asserting that it's bounded by `size`.
    #[track_caller]
    #[inline]
    pub fn new_checked(idx: I, size: Size<'n>) -> Self {
        assert!(idx.zx() < size.into_inner());
        Self(Branded {
            __marker: PhantomData,
            inner: idx,
        })
    }
    /// Returns a new index without asserting that it's bounded by `size`.
    #[track_caller]
    #[inline]
    pub unsafe fn new_unchecked(idx: I, size: Size<'n>) -> Self {
        debug_assert!(idx.zx() < size.into_inner());
        Self(Branded {
            __marker: PhantomData,
            inner: idx,
        })
    }

    /// Returns a new index without asserting that it's bounded by the value tied to the
    /// lifetime `'n`.
    #[inline]
    pub unsafe fn new_raw_unchecked(idx: I) -> Self {
        Self(Branded {
            __marker: PhantomData,
            inner: idx,
        })
    }

    /// Returns the unconstrained value.
    #[inline]
    pub fn into_inner(self) -> I {
        self.0.inner
    }

    /// Zero extend the value.
    #[inline]
    pub fn zx(self) -> Idx<'n, usize> {
        unsafe { Idx::new_raw_unchecked(self.0.inner.zx()) }
    }

    /// Unimplemented: Sign extend the value.
    #[inline]
    pub fn sx(self) -> ! {
        unimplemented!()
    }

    /// Returns the index, bounded inclusively by the value tied to `'n`.
    #[inline]
    pub fn to_inclusive(self) -> IdxInclusive<'n, I> {
        unsafe { IdxInclusive::new_raw_unchecked(self.into_inner()) }
    }
    /// Returns the next index, bounded inclusively by the value tied to `'n`.
    #[inline]
    pub fn next(self) -> IdxInclusive<'n, I> {
        unsafe { IdxInclusive::new_raw_unchecked(self.into_inner() + I::truncate(1)) }
    }

    /// Assert that the values of `slice` are all bounded by `size`.
    #[track_caller]
    #[inline]
    pub fn from_slice_mut_checked<'a>(slice: &'a mut [I], size: Size<'n>) -> &'a mut [Idx<'n, I>] {
        Self::from_slice_ref_checked(slice, size);
        unsafe { &mut *(slice as *mut _ as *mut _) }
    }

    /// Assume that the values of `slice` are all bounded by the value tied to `'n`.
    #[track_caller]
    #[inline]
    pub unsafe fn from_slice_mut_unchecked<'a>(slice: &'a mut [I]) -> &'a mut [Idx<'n, I>] {
        unsafe { &mut *(slice as *mut _ as *mut _) }
    }

    /// Assert that the values of `slice` are all bounded by `size`.
    #[track_caller]
    pub fn from_slice_ref_checked<'a>(slice: &'a [I], size: Size<'n>) -> &'a [Idx<'n, I>] {
        for &idx in slice {
            Self::new_checked(idx, size);
        }
        unsafe { &*(slice as *const _ as *const _) }
    }

    /// Assume that the values of `slice` are all bounded by the value tied to `'n`.
    #[track_caller]
    #[inline]
    pub unsafe fn from_slice_ref_unchecked<'a>(slice: &'a [I]) -> &'a [Idx<'n, I>] {
        unsafe { &*(slice as *const _ as *const _) }
    }
}

impl<'n, I: Index> MaybeIdx<'n, I> {
    /// Returns an index value.
    #[inline]
    pub fn from_index(idx: Idx<'n, I>) -> Self {
        unsafe { Self::new_raw_unchecked(idx.into_inner()) }
    }
    /// Returns a `None` value.
    #[inline]
    pub fn none() -> Self {
        unsafe { Self::new_raw_unchecked(I::truncate(usize::MAX)) }
    }

    /// Returns a constrained index value if `idx` is nonnegative, `None` otherwise.
    #[inline]
    pub fn new_checked(idx: I::Signed, size: Size<'n>) -> Self {
        assert!((idx.sx() as isize) < size.into_inner() as isize);
        Self(Branded {
            __marker: PhantomData,
            inner: I::from_signed(idx),
        })
    }

    /// Returns a constrained index value if `idx` is nonnegative, `None` otherwise.
    #[inline]
    pub unsafe fn new_unchecked(idx: I::Signed, size: Size<'n>) -> Self {
        debug_assert!((idx.sx() as isize) < size.into_inner() as isize);
        Self(Branded {
            __marker: PhantomData,
            inner: I::from_signed(idx),
        })
    }

    /// Returns a constrained index value if `idx` is nonnegative, `None` otherwise.
    #[inline]
    pub unsafe fn new_raw_unchecked(idx: I) -> Self {
        Self(Branded {
            __marker: PhantomData,
            inner: idx,
        })
    }

    /// Returns the inner value.
    #[inline]
    pub fn into_inner(self) -> I {
        self.0.inner
    }

    /// Returns the index if available, or `None` otherwise.
    #[inline]
    pub fn idx(self) -> Option<Idx<'n, I>> {
        if self.0.inner.to_signed() >= I::Signed::truncate(0) {
            Some(unsafe { Idx::new_raw_unchecked(self.into_inner()) })
        } else {
            None
        }
    }

    /// Unimplemented: Zero extend the value.
    #[inline]
    pub fn zx(self) -> ! {
        unimplemented!()
    }

    /// Sign extend the value.
    #[inline]
    pub fn sx(self) -> MaybeIdx<'n, usize> {
        unsafe { MaybeIdx::new_raw_unchecked(self.0.inner.to_signed().sx()) }
    }

    /// Assert that the values of `slice` are all bounded by `size`.
    #[track_caller]
    #[inline]
    pub fn from_slice_mut_checked<'a>(
        slice: &'a mut [I::Signed],
        size: Size<'n>,
    ) -> &'a mut [MaybeIdx<'n, I>] {
        Self::from_slice_ref_checked(slice, size);
        unsafe { &mut *(slice as *mut _ as *mut _) }
    }

    /// Assume that the values of `slice` are all bounded by the value tied to `'n`.
    #[track_caller]
    #[inline]
    pub unsafe fn from_slice_mut_unchecked<'a>(
        slice: &'a mut [I::Signed],
    ) -> &'a mut [MaybeIdx<'n, I>] {
        unsafe { &mut *(slice as *mut _ as *mut _) }
    }

    /// Assert that the values of `slice` are all bounded by `size`.
    #[track_caller]
    pub fn from_slice_ref_checked<'a>(
        slice: &'a [I::Signed],
        size: Size<'n>,
    ) -> &'a [MaybeIdx<'n, I>] {
        for &idx in slice {
            Self::new_checked(idx, size);
        }
        unsafe { &*(slice as *const _ as *const _) }
    }

    /// Convert a constrained slice to an unconstrained one.
    #[track_caller]
    pub fn as_slice_ref<'a>(slice: &'a [MaybeIdx<'n, I>]) -> &'a [I::Signed] {
        unsafe { &*(slice as *const _ as *const _) }
    }

    /// Assume that the values of `slice` are all bounded by the value tied to `'n`.
    #[track_caller]
    #[inline]
    pub unsafe fn from_slice_ref_unchecked<'a>(slice: &'a [I::Signed]) -> &'a [MaybeIdx<'n, I>] {
        unsafe { &*(slice as *const _ as *const _) }
    }
}

impl<'n> IdxInclusive<'n, usize> {
    /// Returns an iterator over constrained indices from `0` to `self` (exclusive).
    #[inline]
    pub fn range_to(self, last: Self) -> impl DoubleEndedIterator<Item = Idx<'n, usize>> {
        (*self..*last).map(
            #[inline(always)]
            |idx| unsafe { Idx::new_raw_unchecked(idx) },
        )
    }
}

impl<'n, I: Index> IdxInclusive<'n, I> {
    /// Returns a constrained inclusive index after checking that it's bounded (inclusively) by
    /// `size`.
    #[inline]
    pub fn new_checked(idx: I, size: Size<'n>) -> Self {
        assert!(idx.zx() <= size.into_inner());
        Self(Branded {
            __marker: PhantomData,
            inner: idx,
        })
    }
    /// Returns a constrained inclusive index, assuming that it's bounded (inclusively) by
    /// `size`.
    #[inline]
    pub unsafe fn new_unchecked(idx: I, size: Size<'n>) -> Self {
        debug_assert!(idx.zx() <= size.into_inner());
        Self(Branded {
            __marker: PhantomData,
            inner: idx,
        })
    }

    /// Returns a constrained inclusive index, assuming that it's bounded (inclusively) by
    /// the size tied to `'n`.
    #[inline]
    pub unsafe fn new_raw_unchecked(idx: I) -> Self {
        Self(Branded {
            __marker: PhantomData,
            inner: idx,
        })
    }

    /// Returns the unconstrained value.
    #[inline]
    pub fn into_inner(self) -> I {
        self.0.inner
    }

    /// Unimplemented: Sign extend the value.
    #[inline]
    pub fn sx(self) -> ! {
        unimplemented!()
    }
    /// Unimplemented: Zero extend the value.
    #[inline]
    pub fn zx(self) -> ! {
        unimplemented!()
    }
}

impl<'n, T> Array<'n, T> {
    /// Returns a constrained array after checking that its length matches `size`.
    #[inline]
    #[track_caller]
    pub fn from_ref<'a>(slice: &'a [T], size: Size<'n>) -> &'a Self {
        assert!(slice.len() == size.into_inner());
        unsafe { &*(slice as *const [T] as *const Self) }
    }

    /// Returns a constrained array after checking that its length matches `size`.
    #[inline]
    #[track_caller]
    pub fn from_mut<'a>(slice: &'a mut [T], size: Size<'n>) -> &'a mut Self {
        assert!(slice.len() == size.into_inner());
        unsafe { &mut *(slice as *mut [T] as *mut Self) }
    }

    /// Returns the unconstrained slice.
    #[inline]
    #[track_caller]
    pub fn as_ref(&self) -> &[T] {
        unsafe { &*(self as *const _ as *const _) }
    }

    /// Returns the unconstrained slice.
    #[inline]
    #[track_caller]
    pub fn as_mut<'a>(&mut self) -> &'a mut [T] {
        unsafe { &mut *(self as *mut _ as *mut _) }
    }

    /// Returns the length of `self`.
    #[inline]
    pub fn len(&self) -> Size<'n> {
        unsafe { Size::new_raw_unchecked(self.0.inner.len()) }
    }
}

impl Debug for Size<'_> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.inner.fmt(f)
    }
}
impl<I: Debug> Debug for Idx<'_, I> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.inner.fmt(f)
    }
}
impl<I: Debug> Debug for IdxInclusive<'_, I> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.inner.fmt(f)
    }
}
impl<I: Debug + Index> Debug for MaybeIdx<'_, I> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        #[derive(Debug)]
        struct None;

        match self.idx() {
            Some(idx) => idx.fmt(f),
            Option::None => None.fmt(f),
        }
    }
}
impl<T: Debug> Debug for Array<'_, T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.inner.fmt(f)
    }
}

impl<'n, T> core::ops::Index<Range<IdxInclusive<'n, usize>>> for Array<'n, T> {
    type Output = [T];
    #[track_caller]
    fn index(&self, idx: Range<IdxInclusive<'n, usize>>) -> &Self::Output {
        #[cfg(debug_assertions)]
        {
            &self.0.inner[idx.start.into_inner()..idx.end.into_inner()]
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            self.0
                .inner
                .get_unchecked(idx.start.into_inner()..idx.end.into_inner())
        }
    }
}
impl<'n, T> core::ops::IndexMut<Range<IdxInclusive<'n, usize>>> for Array<'n, T> {
    #[track_caller]
    fn index_mut(&mut self, idx: Range<IdxInclusive<'n, usize>>) -> &mut Self::Output {
        #[cfg(debug_assertions)]
        {
            &mut self.0.inner[idx.start.into_inner()..idx.end.into_inner()]
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            self.0
                .inner
                .get_unchecked_mut(idx.start.into_inner()..idx.end.into_inner())
        }
    }
}
impl<'n, T> core::ops::Index<Idx<'n, usize>> for Array<'n, T> {
    type Output = T;
    #[track_caller]
    fn index(&self, idx: Idx<'n, usize>) -> &Self::Output {
        #[cfg(debug_assertions)]
        {
            &self.0.inner[idx.into_inner()]
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            self.0.inner.get_unchecked(idx.into_inner())
        }
    }
}
impl<'n, T> core::ops::IndexMut<Idx<'n, usize>> for Array<'n, T> {
    #[track_caller]
    fn index_mut(&mut self, idx: Idx<'n, usize>) -> &mut Self::Output {
        #[cfg(debug_assertions)]
        {
            &mut self.0.inner[idx.into_inner()]
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            self.0.inner.get_unchecked_mut(idx.into_inner())
        }
    }
}
