use super::*;
use crate::{
    col::{Col, ColMut, ColRef},
    Shape,
};

/// Diagonal matrix.
pub struct Diag<E: Entity, N: Shape = usize> {
    pub(crate) inner: Col<E, N>,
}

impl<E: Entity, N: Shape> core::fmt::Debug for Diag<E, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl<E: Entity, N: Shape> Diag<E, N> {
    /// Returns the diagonal as a column vector view.
    #[inline(always)]
    pub fn column_vector(&self) -> ColRef<'_, E, N> {
        self.as_ref().column_vector()
    }

    /// Returns the diagonal as a mutable column vector view.
    #[inline(always)]
    pub fn column_vector_mut(&mut self) -> ColMut<'_, E, N> {
        self.as_mut().column_vector_mut()
    }

    /// Returns the diagonal as a column vector.
    #[inline(always)]
    pub fn into_column_vector(self) -> Col<E, N> {
        self.inner
    }

    /// Returns a view over `self`.
    #[inline(always)]
    pub fn as_ref(&self) -> DiagRef<'_, E, N> {
        DiagRef {
            inner: self.inner.as_ref(),
        }
    }

    /// Returns a mutable view over `self`.
    #[inline(always)]
    pub fn as_mut(&mut self) -> DiagMut<'_, E, N> {
        DiagMut {
            inner: self.inner.as_mut(),
        }
    }
}
