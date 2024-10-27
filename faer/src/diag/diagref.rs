use super::*;

pub struct DiagRef<'a, T, Len = usize, Stride = isize> {
    pub(crate) inner: ColRef<'a, T, Len, Stride>,
}

impl<T, Len: Copy, Stride: Copy> Copy for DiagRef<'_, T, Len, Stride> {}
impl<T, Len: Copy, Stride: Copy> Clone for DiagRef<'_, T, Len, Stride> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, T, Len: Copy, Stride: Copy> Reborrow<'short> for DiagRef<'_, T, Len, Stride> {
    type Target = DiagRef<'short, T, Len, Stride>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short, T, Len: Copy, Stride: Copy> ReborrowMut<'short> for DiagRef<'_, T, Len, Stride> {
    type Target = DiagRef<'short, T, Len, Stride>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}
impl<'a, T, Len: Copy, Stride: Copy> IntoConst for DiagRef<'a, T, Len, Stride> {
    type Target = DiagRef<'a, T, Len, Stride>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

unsafe impl<T: Sync, Len: Sync, Stride: Sync> Sync for DiagRef<'_, T, Len, Stride> {}
unsafe impl<T: Sync, Len: Send, Stride: Send> Send for DiagRef<'_, T, Len, Stride> {}

impl<'a, T, N: Shape, Stride: crate::Stride> DiagRef<'a, T, N, Stride> {
    /// Returns the diagonal as a column vector view.
    #[inline(always)]
    pub fn column_vector(self) -> ColRef<'a, T, N, Stride> {
        self.inner
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> DiagRef<'_, T, N, Stride> {
        *self
    }
}
