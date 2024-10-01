use super::*;
use crate::{assert, Index};
use core::{fmt::Debug, marker::PhantomData, ops::Range};
use faer_entity::*;
use reborrow::*;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
struct Branded<'a, T: ?Sized> {
    __marker: PhantomData<fn(&'a ()) -> &'a ()>,
    inner: T,
}

pub(crate) type Size<'n> = crate::utils::bound::Dim<'n>;
pub(crate) type Idx<'n, I> = crate::utils::bound::Idx<'n, I>;
pub(crate) type IdxInclusive<'n, I> = crate::utils::bound::IdxInc<'n, I>;
pub(crate) type MaybeIdx<'n, I> = crate::utils::bound::MaybeIdx<'n, I>;

/// Array of length equal to the value tied to `'n`.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Array<'n, T>(Branded<'n, [T]>);

/// Dense matrices with compile-time access checks.
pub mod mat;
/// Permutations with compile-time checks.
pub mod perm;
/// Sparse matrices with compile-time access checks.
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
        assert!(slice.rb().len() == len.unbound());
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
                .subslice_unchecked(range.start.unbound()..range.end.unbound())
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
                .write_unchecked(j.unbound(), value)
        }
    }
}

impl<'n, 'a, E: Entity> ArrayGroup<'n, 'a, E> {
    /// Returns an array group with length after checking that its length matches
    /// the value tied to `'n`.
    #[inline]
    pub fn new(slice: GroupFor<E, &'a [E::Unit]>, len: Size<'n>) -> Self {
        let slice = slice::SliceGroup::<'_, E>::new(slice);
        assert!(slice.rb().len() == len.unbound());
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
                .subslice_unchecked(range.start.unbound()..range.end.unbound())
                .into_inner()
        }
    }

    /// Read the element at position `j`.
    #[inline]
    pub fn read(&self, j: Idx<'n, usize>) -> E {
        unsafe { slice::SliceGroup::new(self.into_slice()).read_unchecked(j.unbound()) }
    }
}

impl<'n, T> Array<'n, T> {
    /// Returns a constrained array after checking that its length matches `size`.
    #[inline]
    #[track_caller]
    pub fn from_ref<'a>(slice: &'a [T], size: Size<'n>) -> &'a Self {
        assert!(slice.len() == size.unbound());
        unsafe { &*(slice as *const [T] as *const Self) }
    }

    /// Returns a constrained array after checking that its length matches `size`.
    #[inline]
    #[track_caller]
    pub fn from_mut<'a>(slice: &'a mut [T], size: Size<'n>) -> &'a mut Self {
        assert!(slice.len() == size.unbound());
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
        unsafe { Size::new_unbound(self.0.inner.len()) }
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
            &self.0.inner[idx.start.unbound()..idx.end.unbound()]
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            self.0
                .inner
                .get_unchecked(idx.start.unbound()..idx.end.unbound())
        }
    }
}
impl<'n, T> core::ops::IndexMut<Range<IdxInclusive<'n, usize>>> for Array<'n, T> {
    #[track_caller]
    fn index_mut(&mut self, idx: Range<IdxInclusive<'n, usize>>) -> &mut Self::Output {
        #[cfg(debug_assertions)]
        {
            &mut self.0.inner[idx.start.unbound()..idx.end.unbound()]
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            self.0
                .inner
                .get_unchecked_mut(idx.start.unbound()..idx.end.unbound())
        }
    }
}
impl<'n, T> core::ops::Index<Idx<'n, usize>> for Array<'n, T> {
    type Output = T;
    #[track_caller]
    fn index(&self, idx: Idx<'n, usize>) -> &Self::Output {
        #[cfg(debug_assertions)]
        {
            &self.0.inner[idx.unbound()]
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            self.0.inner.get_unchecked(idx.unbound())
        }
    }
}
impl<'n, T> core::ops::IndexMut<Idx<'n, usize>> for Array<'n, T> {
    #[track_caller]
    fn index_mut(&mut self, idx: Idx<'n, usize>) -> &mut Self::Output {
        #[cfg(debug_assertions)]
        {
            &mut self.0.inner[idx.unbound()]
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            self.0.inner.get_unchecked_mut(idx.unbound())
        }
    }
}
