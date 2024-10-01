use super::*;
use crate::{assert, Idx};

/// Permutation matrix.
#[derive(Debug, Clone)]
pub struct Perm<I: Index, N: Shape = usize> {
    pub(super) forward: alloc::boxed::Box<[N::Idx<I>]>,
    pub(super) inverse: alloc::boxed::Box<[N::Idx<I>]>,
}

impl<I: Index, N: Shape> Perm<I, N> {
    /// Convert `self` to a permutation view.
    #[inline]
    pub fn as_ref(&self) -> PermRef<'_, I, N> {
        PermRef {
            forward: &self.forward,
            inverse: &self.inverse,
        }
    }

    /// Returns the input permutation with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape<M: Shape>(&self, dim: M) -> PermRef<'_, I, M> {
        self.as_ref().as_shape(dim)
    }

    /// Returns the input permutation with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn into_shape<M: Shape>(self, dim: M) -> Perm<I, M> {
        assert!(self.len().unbound() == dim.unbound());

        Perm {
            forward: unsafe { Box::from_raw(Box::into_raw(self.forward) as _) },
            inverse: unsafe { Box::from_raw(Box::into_raw(self.inverse) as _) },
        }
    }

    /// Creates a new permutation, by checking the validity of the inputs.
    ///
    /// # Panics
    ///
    /// The function panics if any of the following conditions are violated:
    /// `forward` and `inverse` must have the same length which must be less than or equal to
    /// `I::Signed::MAX`, be valid permutations, and be inverse permutations of each other.
    #[inline]
    #[track_caller]
    pub fn new_checked(
        forward: alloc::boxed::Box<[Idx<N, I>]>,
        inverse: alloc::boxed::Box<[Idx<N, I>]>,
        dim: N,
    ) -> Self {
        PermRef::<'_, I, N>::new_checked(&forward, &inverse, dim);
        Self { forward, inverse }
    }

    /// Creates a new permutation reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length which must be less than or equal to
    /// `I::Signed::MAX`, be valid permutations, and be inverse permutations of each other.
    #[inline]
    #[track_caller]
    pub unsafe fn new_unchecked(
        forward: alloc::boxed::Box<[Idx<N, I>]>,
        inverse: alloc::boxed::Box<[Idx<N, I>]>,
    ) -> Self {
        let n = forward.len();
        assert!(all(
            forward.len() == inverse.len(),
            n <= I::Signed::MAX.zx(),
        ));
        Self { forward, inverse }
    }

    /// Returns the permutation as an array.
    #[inline]
    pub fn into_arrays(
        self,
    ) -> (
        alloc::boxed::Box<[Idx<N, I>]>,
        alloc::boxed::Box<[Idx<N, I>]>,
    ) {
        (self.forward, self.inverse)
    }

    /// Returns the dimension of the permutation.
    #[inline]
    pub fn len(&self) -> N {
        unsafe { N::new_unbound(self.forward.len()) }
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn into_inverse(self) -> Self {
        Self {
            forward: self.inverse,
            inverse: self.forward,
        }
    }
}
