use super::*;
use crate::internal_prelude::{DiagMut, DiagRef};

/// Diagonal matrix.
pub struct Diag<T, N: Shape = usize> {
    pub(crate) inner: Col<T, N>,
}

impl<T, N: Shape> Diag<T, N> {
    /// Returns the diagonal as a column vector view.
    #[inline(always)]
    pub fn column_vector(&self) -> ColRef<'_, T, N> {
        self.as_ref().column_vector()
    }

    /// Returns the diagonal as a mutable column vector view.
    #[inline(always)]
    pub fn column_vector_mut(&mut self) -> ColMut<'_, T, N> {
        self.as_mut().column_vector_mut()
    }

    /// Returns the diagonal as a column vector.
    #[inline(always)]
    pub fn into_column_vector(self) -> Col<T, N> {
        self.inner
    }

    /// Returns a view over `self`.
    #[inline(always)]
    pub fn as_ref(&self) -> DiagRef<'_, T, N> {
        DiagRef {
            inner: self.inner.as_ref(),
        }
    }

    /// Returns a mutable view over `self`.
    #[inline(always)]
    pub fn as_mut(&mut self) -> DiagMut<'_, T, N> {
        DiagMut {
            inner: self.inner.as_mut(),
        }
    }
}
