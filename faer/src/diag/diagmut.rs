use super::*;
use crate::internal_prelude::DiagRef;

/// Diagonal mutable matrix view.
pub struct DiagMut<'a, C: Container, T, N = usize, Stride = isize> {
    pub(crate) inner: ColMut<'a, C, T, N, Stride>,
}

impl<'a, C: Container, T, N: Shape, Stride: crate::Stride> DiagMut<'a, C, T, N, Stride> {
    /// Returns the diagonal as a column vector view.
    #[inline(always)]
    pub fn column_vector(self) -> ColRef<'a, C, T, N, Stride> {
        self.into_const().column_vector()
    }

    /// Returns the diagonal as a mutable column vector view.
    #[inline(always)]
    pub fn column_vector_mut(self) -> ColMut<'a, C, T, N, Stride> {
        self.inner
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> DiagRef<'_, C, T, N, Stride> {
        self.rb()
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> DiagMut<'_, C, T, N, Stride> {
        self.rb_mut()
    }

    #[inline]
    pub fn fill(&mut self, value: C::Of<T>)
    where
        T: Clone,
    {
        self.inner.fill(value)
    }
}

impl<'short, C: Container, T, N: Copy, Stride: Copy> Reborrow<'short>
    for DiagMut<'_, C, T, N, Stride>
{
    type Target = DiagRef<'short, C, T, N, Stride>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        DiagRef {
            inner: self.inner.rb(),
        }
    }
}

impl<'short, C: Container, T, N: Copy, Stride: Copy> ReborrowMut<'short>
    for DiagMut<'_, C, T, N, Stride>
{
    type Target = DiagMut<'short, C, T, N, Stride>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        DiagMut {
            inner: self.inner.rb_mut(),
        }
    }
}

impl<'a, C: Container, T, N: Copy, Stride: Copy> IntoConst for DiagMut<'a, C, T, N, Stride> {
    type Target = DiagRef<'a, C, T, N, Stride>;

    #[inline]
    fn into_const(self) -> Self::Target {
        DiagRef {
            inner: self.inner.into_const(),
        }
    }
}
