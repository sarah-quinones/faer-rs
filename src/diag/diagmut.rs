use super::*;
use crate::col::{ColMut, ColRef};

/// Diagonal mutable matrix view.
pub struct DiagMut<'a, E: Entity> {
    pub(crate) inner: ColMut<'a, E>,
}

impl<'a, E: Entity> DiagMut<'a, E> {
    /// Returns the diagonal as a column vector view.
    #[inline(always)]
    pub fn column_vector(self) -> ColRef<'a, E> {
        self.into_const().column_vector()
    }

    /// Returns the diagonal as a mutable column vector view.
    #[inline(always)]
    pub fn column_vector_mut(self) -> ColMut<'a, E> {
        self.inner
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> DiagRef<'_, E> {
        self.rb()
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> DiagMut<'_, E> {
        self.rb_mut()
    }
}

impl<'short, E: Entity> Reborrow<'short> for DiagMut<'_, E> {
    type Target = DiagRef<'short, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        DiagRef {
            inner: self.inner.rb(),
        }
    }
}

impl<'short, E: Entity> ReborrowMut<'short> for DiagMut<'_, E> {
    type Target = DiagMut<'short, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        DiagMut {
            inner: self.inner.rb_mut(),
        }
    }
}

impl<'a, E: Entity> IntoConst for DiagMut<'a, E> {
    type Target = DiagRef<'a, E>;

    #[inline]
    fn into_const(self) -> Self::Target {
        DiagRef {
            inner: self.inner.into_const(),
        }
    }
}
