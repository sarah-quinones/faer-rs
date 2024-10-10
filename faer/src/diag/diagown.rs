use super::*;
use crate::internal_prelude::{DiagMut, DiagRef};

/// Diagonal matrix.
pub struct Diag<C: Container, T, N: Shape = usize> {
    pub(crate) inner: Col<C, T, N>,
}

impl<C: Container, T, N: Shape> Diag<C, T, N> {
    /// Returns the diagonal as a column vector view.
    #[inline(always)]
    pub fn column_vector(&self) -> ColRef<'_, C, T, N> {
        self.as_ref().column_vector()
    }

    /// Returns the diagonal as a mutable column vector view.
    #[inline(always)]
    pub fn column_vector_mut(&mut self) -> ColMut<'_, C, T, N> {
        self.as_mut().column_vector_mut()
    }

    /// Returns the diagonal as a column vector.
    #[inline(always)]
    pub fn into_column_vector(self) -> Col<C, T, N> {
        self.inner
    }

    /// Returns a view over `self`.
    #[inline(always)]
    pub fn as_ref(&self) -> DiagRef<'_, C, T, N> {
        DiagRef {
            inner: self.inner.as_ref(),
        }
    }

    /// Returns a mutable view over `self`.
    #[inline(always)]
    pub fn as_mut(&mut self) -> DiagMut<'_, C, T, N> {
        DiagMut {
            inner: self.inner.as_mut(),
        }
    }
}
