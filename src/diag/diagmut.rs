use super::*;
use crate::{
    col::{ColMut, ColRef},
    Shape,
};

/// Diagonal mutable matrix view.
pub struct DiagMut<'a, E: Entity, N: Shape = usize> {
    pub(crate) inner: ColMut<'a, E, N>,
}

impl<E: Entity, N: Shape> core::fmt::Debug for DiagMut<'_, E, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl<'a, E: Entity, N: Shape> DiagMut<'a, E, N> {
    /// Returns the diagonal as a column vector view.
    #[inline(always)]
    pub fn column_vector(self) -> ColRef<'a, E, N> {
        self.into_const().column_vector()
    }

    /// Returns the diagonal as a mutable column vector view.
    #[inline(always)]
    pub fn column_vector_mut(self) -> ColMut<'a, E, N> {
        self.inner
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> DiagRef<'_, E, N> {
        self.rb()
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> DiagMut<'_, E, N> {
        self.rb_mut()
    }
}

impl<'short, E: Entity, N: Shape> Reborrow<'short> for DiagMut<'_, E, N> {
    type Target = DiagRef<'short, E, N>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        DiagRef {
            inner: self.inner.rb(),
        }
    }
}

impl<'short, E: Entity, N: Shape> ReborrowMut<'short> for DiagMut<'_, E, N> {
    type Target = DiagMut<'short, E, N>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        DiagMut {
            inner: self.inner.rb_mut(),
        }
    }
}

impl<'a, E: Entity, N: Shape> IntoConst for DiagMut<'a, E, N> {
    type Target = DiagRef<'a, E, N>;

    #[inline]
    fn into_const(self) -> Self::Target {
        DiagRef {
            inner: self.inner.into_const(),
        }
    }
}
