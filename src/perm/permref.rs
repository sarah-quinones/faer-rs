use super::*;
use crate::assert;

/// Immutable permutation matrix view.
#[derive(Debug)]
pub struct PermRef<'a, I: Index, N: Shape = usize> {
    pub(super) forward: &'a [N::Idx<I>],
    pub(super) inverse: &'a [N::Idx<I>],
}

impl<I: Index, N: Shape> Copy for PermRef<'_, I, N> {}
impl<I: Index, N: Shape> Clone for PermRef<'_, I, N> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, I: Index, N: Shape> Reborrow<'short> for PermRef<'_, I, N> {
    type Target = PermRef<'short, I, N>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short, I: Index, N: Shape> ReborrowMut<'short> for PermRef<'_, I, N> {
    type Target = PermRef<'short, I, N>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}
impl<'a, I: Index, N: Shape> IntoConst for PermRef<'a, I, N> {
    type Target = PermRef<'a, I, N>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'a, I: Index, N: Shape> PermRef<'a, I, N> {
    /// Convert `self` to a permutation view.
    #[inline]
    pub fn as_ref(&self) -> PermRef<'_, I, N> {
        PermRef {
            forward: self.forward,
            inverse: self.inverse,
        }
    }

    /// Returns the input permutation with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape<M: Shape>(self, dim: M) -> PermRef<'a, I, M> {
        assert!(self.len().unbound() == dim.unbound());

        PermRef {
            forward: unsafe {
                core::slice::from_raw_parts(self.forward.as_ptr() as _, dim.unbound())
            },
            inverse: unsafe {
                core::slice::from_raw_parts(self.inverse.as_ptr() as _, dim.unbound())
            },
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
    pub fn new_checked(forward: &'a [Idx<N, I>], inverse: &'a [Idx<N, I>], dim: N) -> Self {
        #[track_caller]
        fn check<I: Index>(forward: &[I], inverse: &[I], n: usize) {
            assert!(all(
                n <= I::Signed::MAX.zx(),
                forward.len() == n,
                inverse.len() == n,
            ));
            for (i, &p) in forward.iter().enumerate() {
                let p = p.to_signed().zx();
                assert!(p < n);
                assert!(inverse[p].to_signed().zx() == i);
            }
        }

        check(
            I::canonicalize(N::cast_idx_slice(forward)),
            I::canonicalize(N::cast_idx_slice(inverse)),
            dim.unbound(),
        );
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
        forward: &'a [Idx<N, I>],
        inverse: &'a [Idx<N, I>],
        dim: N,
    ) -> Self {
        assert!(all(
            dim.unbound() <= I::Signed::MAX.zx(),
            forward.len() == dim.unbound(),
            inverse.len() == dim.unbound(),
        ));
        Self { forward, inverse }
    }

    /// Returns the permutation as an array.
    #[inline]
    pub fn arrays(self) -> (&'a [Idx<N, I>], &'a [Idx<N, I>]) {
        (self.forward, self.inverse)
    }

    /// Returns the dimension of the permutation.
    #[inline]
    pub fn len(&self) -> N {
        unsafe { N::new_unbound(self.forward.len()) }
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
    pub fn canonicalized(self) -> PermRef<'a, I::FixedWidth, N> {
        unsafe {
            PermRef {
                forward: core::slice::from_raw_parts(
                    self.forward.as_ptr() as _,
                    self.forward.len(),
                ),
                inverse: core::slice::from_raw_parts(
                    self.inverse.as_ptr() as _,
                    self.inverse.len(),
                ),
            }
        }
    }

    /// Cast the permutation from the fixed width index type.
    #[inline(always)]
    pub fn uncanonicalized<J: Index>(self) -> PermRef<'a, J, N> {
        assert!(core::mem::size_of::<J>() == core::mem::size_of::<I>());
        unsafe {
            PermRef {
                forward: core::slice::from_raw_parts(
                    self.forward.as_ptr() as _,
                    self.forward.len(),
                ),
                inverse: core::slice::from_raw_parts(
                    self.inverse.as_ptr() as _,
                    self.inverse.len(),
                ),
            }
        }
    }
}
