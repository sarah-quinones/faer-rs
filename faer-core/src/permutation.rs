use assert2::debug_assert as fancy_debug_assert;
use reborrow::*;

#[derive(Clone, Copy, Debug)]
pub struct PermutationIndicesRef<'a> {
    forward: &'a [usize],
    inverse: &'a [usize],
}

impl<'a> PermutationIndicesRef<'a> {
    /// Returns the permutation as an array.
    #[inline]
    pub fn into_array(self) -> &'a [usize] {
        self.forward
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            forward: self.inverse,
            inverse: self.forward,
        }
    }

    /// Creates a new permutation reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length, be valid permutations, and be inverse
    /// permutations of each other.
    #[inline]
    pub unsafe fn new_unchecked(forward: &'a [usize], inverse: &'a [usize]) -> Self {
        fancy_debug_assert!(forward.len() == inverse.len());
        Self { forward, inverse }
    }
}

impl<'a> PermutationIndicesMut<'a> {
    /// Returns the permutation as an array.
    #[inline]
    pub fn into_array(self) -> &'a mut [usize] {
        self.forward
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            forward: self.inverse,
            inverse: self.forward,
        }
    }

    /// Creates a new permutation mutable reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length, be valid permutations, and be inverse
    /// permutations of each other.
    #[inline]
    pub unsafe fn new_unchecked(forward: &'a mut [usize], inverse: &'a mut [usize]) -> Self {
        fancy_debug_assert!(forward.len() == inverse.len());
        Self { forward, inverse }
    }
}

#[derive(Debug)]
pub struct PermutationIndicesMut<'a> {
    forward: &'a mut [usize],
    inverse: &'a mut [usize],
}

impl<'short, 'a> Reborrow<'short> for PermutationIndicesRef<'a> {
    type Target = PermutationIndicesRef<'short>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, 'a> ReborrowMut<'short> for PermutationIndicesRef<'a> {
    type Target = PermutationIndicesRef<'short>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'short, 'a> Reborrow<'short> for PermutationIndicesMut<'a> {
    type Target = PermutationIndicesRef<'short>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        PermutationIndicesRef {
            forward: &*self.forward,
            inverse: &*self.inverse,
        }
    }
}

impl<'short, 'a> ReborrowMut<'short> for PermutationIndicesMut<'a> {
    type Target = PermutationIndicesMut<'short>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        PermutationIndicesMut {
            forward: &mut *self.forward,
            inverse: &mut *self.inverse,
        }
    }
}
