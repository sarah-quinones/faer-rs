use super::*;
use crate::internal_prelude::DiagRef;

/// Diagonal mutable matrix view.
pub struct DiagMut<'a, T, N = usize, Stride = isize> {
    pub(crate) inner: ColMut<'a, T, N, Stride>,
}

impl<'a, T, N: Shape, Stride: crate::Stride> DiagMut<'a, T, N, Stride> {
    /// Returns the diagonal as a column vector view.
    #[inline(always)]
    pub fn column_vector(self) -> ColRef<'a, T, N, Stride> {
        self.into_const().column_vector()
    }

    /// Returns the diagonal as a mutable column vector view.
    #[inline(always)]
    pub fn column_vector_mut(self) -> ColMut<'a, T, N, Stride> {
        self.inner
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> DiagRef<'_, T, N, Stride> {
        self.rb()
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> DiagMut<'_, T, N, Stride> {
        self.rb_mut()
    }

    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.inner.fill(value)
    }
}

impl<'short, T, N: Copy, Stride: Copy> Reborrow<'short> for DiagMut<'_, T, N, Stride> {
    type Target = DiagRef<'short, T, N, Stride>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        DiagRef {
            inner: self.inner.rb(),
        }
    }
}

impl<'short, T, N: Copy, Stride: Copy> ReborrowMut<'short> for DiagMut<'_, T, N, Stride> {
    type Target = DiagMut<'short, T, N, Stride>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        DiagMut {
            inner: self.inner.rb_mut(),
        }
    }
}

impl<'a, T, N: Copy, Stride: Copy> IntoConst for DiagMut<'a, T, N, Stride> {
    type Target = DiagRef<'a, T, N, Stride>;

    #[inline]
    fn into_const(self) -> Self::Target {
        DiagRef {
            inner: self.inner.into_const(),
        }
    }
}
