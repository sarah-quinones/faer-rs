use super::*;
use crate::{col::ColRef, Shape};

/// Diagonal matrix view.
pub struct DiagRef<'a, E: Entity, N: Shape = usize> {
    pub(crate) inner: ColRef<'a, E, N>,
}

impl<E: Entity, N: Shape> core::fmt::Debug for DiagRef<'_, E, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl<'a, E: Entity, N: Shape> DiagRef<'a, E, N> {
    /// Returns the diagonal as a column vector view.
    #[inline(always)]
    pub fn column_vector(self) -> ColRef<'a, E, N> {
        self.inner
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> DiagRef<'_, E, N> {
        *self
    }
}

impl<E: Entity, N: Shape> Clone for DiagRef<'_, E, N> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Entity, N: Shape> Copy for DiagRef<'_, E, N> {}

impl<'short, E: Entity, N: Shape> Reborrow<'short> for DiagRef<'_, E, N> {
    type Target = DiagRef<'short, E, N>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, E: Entity, N: Shape> ReborrowMut<'short> for DiagRef<'_, E, N> {
    type Target = DiagRef<'short, E, N>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<E: Entity, N: Shape> IntoConst for DiagRef<'_, E, N> {
    type Target = Self;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}
