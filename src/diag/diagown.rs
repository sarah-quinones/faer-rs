use super::*;
use crate::col::{Col, ColMut, ColRef};

/// Diagonal matrix.
#[derive(Debug)]
pub struct Diag<E: Entity> {
    pub(crate) inner: Col<E>,
}

impl<E: Entity> Diag<E> {
    /// Returns the diagonal as a column vector view.
    #[inline(always)]
    pub fn column_vector(&self) -> ColRef<'_, E> {
        self.as_ref().column_vector()
    }

    /// Returns the diagonal as a mutable column vector view.
    #[inline(always)]
    pub fn column_vector_mut(&mut self) -> ColMut<'_, E> {
        self.as_mut().column_vector_mut()
    }

    /// Returns the diagonal as a column vector.
    #[inline(always)]
    pub fn into_column_vector(self) -> Col<E> {
        self.inner
    }

    /// Returns a view over `self`.
    #[inline(always)]
    pub fn as_ref(&self) -> DiagRef<'_, E> {
        DiagRef {
            inner: self.inner.as_ref(),
        }
    }

    /// Returns a mutable view over `self`.
    #[inline(always)]
    pub fn as_mut(&mut self) -> DiagMut<'_, E> {
        DiagMut {
            inner: self.inner.as_mut(),
        }
    }
}
