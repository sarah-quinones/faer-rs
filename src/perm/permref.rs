use super::*;
use crate::assert;
use core::marker::PhantomData;

/// Immutable permutation matrix view.
#[derive(Debug)]
pub struct PermRef<'a, I: Index, N: Shape = usize> {
    pub(super) forward: &'a [N::Idx<I>],
    pub(super) inverse: &'a [N::Idx<I>],
    pub(super) __marker: PhantomData<N>,
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

impl<'a, I: Index, N: Shape> PermRef<'a, I, N> {
    /// Convert `self` to a permutation view.
    #[inline]
    pub fn as_ref(&self) -> PermRef<'_, I, N> {
        PermRef {
            forward: self.forward,
            inverse: self.inverse,
            __marker: PhantomData,
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
    pub fn new_checked(forward: &'a [Idx<N, I>], inverse: &'a [Idx<N, I>], n: N) -> Self {
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
            n.unbound(),
        );
        Self {
            forward,
            inverse,
            __marker: PhantomData,
        }
    }

    /// Creates a new permutation reference, without checking the validity of the inputs.
    ///
    /// # Safety
    ///
    /// `forward` and `inverse` must have the same length which must be less than or equal to
    /// `I::Signed::MAX`, be valid permutations, and be inverse permutations of each other.
    #[inline]
    #[track_caller]
    pub unsafe fn new_unchecked(forward: &'a [Idx<N, I>], inverse: &'a [Idx<N, I>], n: N) -> Self {
        assert!(all(
            n.unbound() <= I::Signed::MAX.zx(),
            forward.len() == n.unbound(),
            inverse.len() == n.unbound(),
        ));
        Self {
            forward,
            inverse,
            __marker: PhantomData,
        }
    }

    /// Returns the permutation as an array.
    #[inline]
    pub fn arrays(self) -> (&'a [Idx<N, I>], &'a [Idx<N, I>]) {
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
            __marker: PhantomData,
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
                __marker: PhantomData,
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
                __marker: PhantomData,
            }
        }
    }
}
