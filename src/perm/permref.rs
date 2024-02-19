use super::*;
use crate::assert;

/// Immutable permutation matrix view.
#[derive(Debug)]
pub struct PermRef<'a, I: Index> {
    pub(super) forward: &'a [I],
    pub(super) inverse: &'a [I],
}

impl<I: Index> Copy for PermRef<'_, I> {}
impl<I: Index> Clone for PermRef<'_, I> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, I: Index> Reborrow<'short> for PermRef<'_, I> {
    type Target = PermRef<'short, I>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short, I: Index> ReborrowMut<'short> for PermRef<'_, I> {
    type Target = PermRef<'short, I>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}
impl<'a, I: Index> IntoConst for PermRef<'a, I> {
    type Target = PermRef<'a, I>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'a, I: Index> PermRef<'a, I> {
    /// Convert `self` to a permutation view.
    #[inline]
    pub fn as_ref(&self) -> PermRef<'_, I> {
        PermRef {
            forward: self.forward,
            inverse: self.inverse,
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
    pub fn new_checked(forward: &'a [I], inverse: &'a [I]) -> Self {
        #[track_caller]
        fn check<I: Index>(forward: &[I], inverse: &[I]) {
            let n = forward.len();
            assert!(all(
                forward.len() == inverse.len(),
                n <= I::Signed::MAX.zx()
            ));
            for (i, &p) in forward.iter().enumerate() {
                let p = p.to_signed().zx();
                assert!(p < n);
                assert!(inverse[p].to_signed().zx() == i);
            }
        }

        check(I::canonicalize(forward), I::canonicalize(inverse));
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
    pub unsafe fn new_unchecked(forward: &'a [I], inverse: &'a [I]) -> Self {
        let n = forward.len();
        assert!(all(
            forward.len() == inverse.len(),
            n <= I::Signed::MAX.zx(),
        ));
        Self { forward, inverse }
    }

    /// Returns the permutation as an array.
    #[inline]
    pub fn arrays(self) -> (&'a [I], &'a [I]) {
        (self.forward, self.inverse)
    }

    /// Returns the dimension of the permutation.
    #[inline]
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    /// Returns the inverse permutation.
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            forward: self.inverse,
            inverse: self.forward,
        }
    }

    /// Cast the permutation to the fixed width index type.
    #[inline(always)]
    pub fn canonicalized(self) -> PermRef<'a, I::FixedWidth> {
        PermRef {
            forward: I::canonicalize(self.forward),
            inverse: I::canonicalize(self.inverse),
        }
    }

    /// Cast the permutation from the fixed width index type.
    #[inline(always)]
    pub fn uncanonicalized<J: Index>(self) -> PermRef<'a, J> {
        assert!(core::mem::size_of::<J>() == core::mem::size_of::<I>());
        PermRef {
            forward: bytemuck::cast_slice(self.forward),
            inverse: bytemuck::cast_slice(self.inverse),
        }
    }
}
