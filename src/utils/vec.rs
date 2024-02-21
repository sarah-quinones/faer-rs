use super::slice::*;
use core::fmt::Debug;
use faer_entity::*;

/// Analogous to [`alloc::vec::Vec`] for groups.
pub struct VecGroup<E: Entity, T = UnitFor<E>> {
    inner: GroupFor<E, alloc::vec::Vec<T>>,
}

impl<E: Entity, T: Clone> Clone for VecGroup<E, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: E::faer_map(E::faer_as_ref(&self.inner), |v| (*v).clone()),
        }
    }
}

impl<E: Entity, T: Debug> Debug for VecGroup<E, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_slice().fmt(f)
    }
}

unsafe impl<E: Entity, T: Sync> Sync for VecGroup<E, T> {}
unsafe impl<E: Entity, T: Send> Send for VecGroup<E, T> {}

impl<E: Entity, T> Default for VecGroup<E, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Entity, T> VecGroup<E, T> {
    /// Create a new [`VecGroup`] from a group of [`alloc::vec::Vec`].
    #[inline]
    pub fn from_inner(inner: GroupFor<E, alloc::vec::Vec<T>>) -> Self {
        Self { inner }
    }

    /// Consume `self` to return a group of [`alloc::vec::Vec`].
    #[inline]
    pub fn into_inner(self) -> GroupFor<E, alloc::vec::Vec<T>> {
        self.inner
    }

    /// Return a reference to the inner group of [`alloc::vec::Vec`].
    #[inline]
    pub fn as_inner_ref(&self) -> GroupFor<E, &alloc::vec::Vec<T>> {
        E::faer_as_ref(&self.inner)
    }

    /// Return a mutable reference to the inner group of [`alloc::vec::Vec`].
    #[inline]
    pub fn as_inner_mut(&mut self) -> GroupFor<E, &mut alloc::vec::Vec<T>> {
        E::faer_as_mut(&mut self.inner)
    }

    /// Return a [`SliceGroup`] view over the elements of `self`.
    #[inline]
    pub fn as_slice(&self) -> SliceGroup<'_, E, T> {
        SliceGroup::new(E::faer_map(
            E::faer_as_ref(&self.inner),
            #[inline]
            |slice| &**slice,
        ))
    }

    /// Return a [`SliceGroupMut`] mutable view over the elements of `self`.
    #[inline]
    pub fn as_slice_mut(&mut self) -> SliceGroupMut<'_, E, T> {
        SliceGroupMut::new(E::faer_map(
            E::faer_as_mut(&mut self.inner),
            #[inline]
            |slice| &mut **slice,
        ))
    }

    /// Create an empty [`VecGroup`].
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: E::faer_map(E::UNIT, |()| alloc::vec::Vec::new()),
        }
    }

    /// Returns the length of the vector group.
    #[inline]
    pub fn len(&self) -> usize {
        let mut len = usize::MAX;
        E::faer_map(
            E::faer_as_ref(&self.inner),
            #[inline(always)]
            |slice| len = Ord::min(len, slice.len()),
        );
        len
    }

    /// Returns the capacity of the vector group.
    #[inline]
    pub fn capacity(&self) -> usize {
        let mut cap = usize::MAX;
        E::faer_map(
            E::faer_as_ref(&self.inner),
            #[inline(always)]
            |slice| cap = Ord::min(cap, slice.capacity()),
        );
        cap
    }

    /// Reserve enough capacity for extra `additional` elements.
    pub fn reserve(&mut self, additional: usize) {
        E::faer_map(E::faer_as_mut(&mut self.inner), |v| v.reserve(additional));
    }

    /// Reserve exactly enough capacity for extra `additional` elements.
    pub fn reserve_exact(&mut self, additional: usize) {
        E::faer_map(E::faer_as_mut(&mut self.inner), |v| {
            v.reserve_exact(additional)
        });
    }

    /// Try to reserve enough capacity for extra `additional` elements.
    pub fn try_reserve(
        &mut self,
        additional: usize,
    ) -> Result<(), alloc::collections::TryReserveError> {
        let mut result = Ok(());
        E::faer_map(E::faer_as_mut(&mut self.inner), |v| match &result {
            Ok(()) => result = v.try_reserve(additional),
            Err(_) => {}
        });
        result
    }

    /// Try to reserve exactly enough capacity for extra `additional` elements.
    pub fn try_reserve_exact(
        &mut self,
        additional: usize,
    ) -> Result<(), alloc::collections::TryReserveError> {
        let mut result = Ok(());
        E::faer_map(E::faer_as_mut(&mut self.inner), |v| match &result {
            Ok(()) => result = v.try_reserve_exact(additional),
            Err(_) => {}
        });
        result
    }

    /// Truncate the length of the vector to `len`.
    pub fn truncate(&mut self, len: usize) {
        E::faer_map(E::faer_as_mut(&mut self.inner), |v| v.truncate(len));
    }

    /// Clear the vector, making it empty.
    pub fn clear(&mut self) {
        E::faer_map(E::faer_as_mut(&mut self.inner), |v| v.clear());
    }

    /// Resize the vector to `new_len`, filling the new elements with
    /// `value`.
    pub fn resize(&mut self, new_len: usize, value: GroupFor<E, T>)
    where
        T: Clone,
    {
        E::faer_map(
            E::faer_zip(E::faer_as_mut(&mut self.inner), value),
            |(v, value)| v.resize(new_len, value),
        );
    }

    /// Resize the vector to `new_len`, filling the new elements with
    /// the output of `f`.
    pub fn resize_with(&mut self, new_len: usize, f: impl FnMut() -> GroupFor<E, T>) {
        let len = self.len();
        let mut f = f;
        if new_len <= len {
            self.truncate(new_len);
        } else {
            self.reserve(new_len - len);
            for _ in len..new_len {
                self.push(f())
            }
        }
    }

    /// Push a new element to the end of `self`.
    #[inline]
    pub fn push(&mut self, value: GroupFor<E, T>) {
        E::faer_map(
            E::faer_zip(E::faer_as_mut(&mut self.inner), value),
            #[inline]
            |(v, value)| v.push(value),
        );
    }

    /// Remove a new element from the end of `self`, and return it.
    #[inline]
    pub fn pop(&mut self) -> Option<GroupFor<E, T>> {
        if self.len() >= 1 {
            Some(E::faer_map(
                E::faer_as_mut(&mut self.inner),
                #[inline]
                |v| v.pop().unwrap(),
            ))
        } else {
            None
        }
    }

    /// Remove a new element from position `index`, and return it.
    #[inline]
    pub fn remove(&mut self, index: usize) -> GroupFor<E, T> {
        E::faer_map(
            E::faer_as_mut(&mut self.inner),
            #[inline]
            |v| v.remove(index),
        )
    }
}
