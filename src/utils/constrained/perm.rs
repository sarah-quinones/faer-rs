use super::*;
use crate::{assert, perm};

/// Permutation of length equal to the value tied to `'n`.
#[repr(transparent)]
pub struct PermRef<'n, 'a, I: Index>(Branded<'n, perm::PermRef<'a, I>>);

impl<'n, 'a, I: Index> PermRef<'n, 'a, I> {
    /// Returns a new permutation after checking that it matches the size tied to `'n`.
    #[inline]
    #[track_caller]
    pub fn new(perm: perm::PermRef<'a, I>, size: Dim<'n>) -> Self {
        let (fwd, inv) = perm.arrays();
        assert!(all(
            fwd.len() == size.unbound(),
            inv.len() == size.unbound(),
        ));
        Self(Branded {
            __marker: PhantomData,
            inner: perm,
        })
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn inverse(self) -> PermRef<'n, 'a, I> {
        PermRef(Branded {
            __marker: PhantomData,
            inner: self.0.inner.inverse(),
        })
    }

    /// Returns the forward and inverse permutation indices.
    #[inline]
    pub fn arrays(self) -> (&'a Array<'n, Idx<'n, I>>, &'a Array<'n, Idx<'n, I>>) {
        unsafe {
            let (fwd, inv) = self.0.inner.arrays();
            let fwd = &*(fwd as *const [I] as *const Array<'n, Idx<'n, I>>);
            let inv = &*(inv as *const [I] as *const Array<'n, Idx<'n, I>>);
            (fwd, inv)
        }
    }

    /// Returns the unconstrained permutation.
    #[inline]
    pub fn into_inner(self) -> perm::PermRef<'a, I> {
        self.0.inner
    }

    /// Returns the length of the permutation.
    #[inline]
    pub fn len(&self) -> Dim<'n> {
        unsafe { Dim::new_unbound(self.into_inner().len()) }
    }
}

impl<I: Index> Clone for PermRef<'_, '_, I> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<I: Index> Copy for PermRef<'_, '_, I> {}

impl<I: Index> core::fmt::Debug for PermRef<'_, '_, I> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.inner.fmt(f)
    }
}
