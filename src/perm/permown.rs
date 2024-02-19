use super::*;
use crate::assert;

/// Permutation matrix.
#[derive(Debug, Clone)]
pub struct Perm<I: Index> {
    pub(super) forward: alloc::boxed::Box<[I]>,
    pub(super) inverse: alloc::boxed::Box<[I]>,
}

impl<I: Index> Perm<I> {
    /// Convert `self` to a permutation view.
    #[inline]
    pub fn as_ref(&self) -> PermRef<'_, I> {
        PermRef {
            forward: &self.forward,
            inverse: &self.inverse,
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
    pub fn new_checked(forward: alloc::boxed::Box<[I]>, inverse: alloc::boxed::Box<[I]>) -> Self {
        PermRef::<'_, I>::new_checked(&forward, &inverse);
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
        forward: alloc::boxed::Box<[I]>,
        inverse: alloc::boxed::Box<[I]>,
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
    pub fn into_arrays(self) -> (alloc::boxed::Box<[I]>, alloc::boxed::Box<[I]>) {
        (self.forward, self.inverse)
    }

    /// Returns the dimension of the permutation.
    #[inline]
    pub fn len(&self) -> usize {
        self.forward.len()
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
