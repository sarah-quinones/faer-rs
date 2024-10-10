use super::*;

pub struct DiagRef<'a, C: Container, T, Len = usize, Stride = isize> {
    pub(crate) inner: ColRef<'a, C, T, Len, Stride>,
}

impl<C: Container, T, Len: Copy, Stride: Copy> Copy for DiagRef<'_, C, T, Len, Stride> {}
impl<C: Container, T, Len: Copy, Stride: Copy> Clone for DiagRef<'_, C, T, Len, Stride> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'short, C: Container, T, Len: Copy, Stride: Copy> Reborrow<'short>
    for DiagRef<'_, C, T, Len, Stride>
{
    type Target = DiagRef<'short, C, T, Len, Stride>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short, C: Container, T, Len: Copy, Stride: Copy> ReborrowMut<'short>
    for DiagRef<'_, C, T, Len, Stride>
{
    type Target = DiagRef<'short, C, T, Len, Stride>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}
impl<'a, C: Container, T, Len: Copy, Stride: Copy> IntoConst for DiagRef<'a, C, T, Len, Stride> {
    type Target = DiagRef<'a, C, T, Len, Stride>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

unsafe impl<C: Container, T: Sync, Len: Sync, Stride: Sync> Sync
    for DiagRef<'_, C, T, Len, Stride>
{
}
unsafe impl<C: Container, T: Sync, Len: Send, Stride: Send> Send
    for DiagRef<'_, C, T, Len, Stride>
{
}

impl<'a, C: Container, T, N: Shape, Stride: crate::Stride> DiagRef<'a, C, T, N, Stride> {
    /// Returns the diagonal as a column vector view.
    #[inline(always)]
    pub fn column_vector(self) -> ColRef<'a, C, T, N, Stride> {
        self.inner
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> DiagRef<'_, C, T, N, Stride> {
        *self
    }
}
