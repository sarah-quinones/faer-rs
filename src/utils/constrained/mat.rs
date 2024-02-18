use super::*;
use crate::assert;
use crate::mat;

/// Immutable dense matrix view with dimensions equal to the values tied to `('nrows, 'ncols)`.
#[repr(transparent)]
pub struct MatRef<'nrows, 'ncols, 'a, E: Entity>(
    Branded<'ncols, Branded<'nrows, mat::MatRef<'a, E>>>,
);
/// Mutable dense matrix view with dimensions equal to the values tied to `('nrows, 'ncols)`.
#[repr(transparent)]
pub struct MatMut<'nrows, 'ncols, 'a, E: Entity>(
    Branded<'ncols, Branded<'nrows, mat::MatMut<'a, E>>>,
);

impl<'nrows, 'ncols, 'a, E: Entity> MatRef<'nrows, 'ncols, 'a, E> {
    /// Returns a new matrix view after checking that its dimensions match the
    /// dimensions tied to `('nrows, 'ncols)`.
    #[inline]
    #[track_caller]
    pub fn new(inner: mat::MatRef<'a, E>, nrows: Size<'nrows>, ncols: Size<'ncols>) -> Self {
        assert!(all(
            inner.nrows() == nrows.into_inner(),
            inner.ncols() == ncols.into_inner(),
        ));
        Self(Branded {
            __marker: PhantomData,
            inner: Branded {
                __marker: PhantomData,
                inner,
            },
        })
    }

    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> Size<'nrows> {
        unsafe { Size::new_raw_unchecked(self.0.inner.inner.nrows()) }
    }

    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> Size<'ncols> {
        unsafe { Size::new_raw_unchecked(self.0.inner.inner.ncols()) }
    }

    /// Returns the unconstrained matrix.
    #[inline]
    pub fn into_inner(self) -> mat::MatRef<'a, E> {
        self.0.inner.inner
    }

    /// Returns the element at position `(i, j)`.
    #[inline]
    #[track_caller]
    pub fn read(&self, i: Idx<'nrows, usize>, j: Idx<'ncols, usize>) -> E {
        unsafe {
            self.0
                .inner
                .inner
                .read_unchecked(i.into_inner(), j.into_inner())
        }
    }
}

impl<'nrows, 'ncols, 'a, E: Entity> MatMut<'nrows, 'ncols, 'a, E> {
    /// Returns a new matrix view after checking that its dimensions match the
    /// dimensions tied to `('nrows, 'ncols)`.
    #[inline]
    #[track_caller]
    pub fn new(inner: mat::MatMut<'a, E>, nrows: Size<'nrows>, ncols: Size<'ncols>) -> Self {
        assert!(all(
            inner.nrows() == nrows.into_inner(),
            inner.ncols() == ncols.into_inner(),
        ));
        Self(Branded {
            __marker: PhantomData,
            inner: Branded {
                __marker: PhantomData,
                inner,
            },
        })
    }

    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> Size<'nrows> {
        unsafe { Size::new_raw_unchecked(self.0.inner.inner.nrows()) }
    }

    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> Size<'ncols> {
        unsafe { Size::new_raw_unchecked(self.0.inner.inner.ncols()) }
    }

    /// Returns the unconstrained matrix.
    #[inline]
    pub fn into_inner(self) -> mat::MatMut<'a, E> {
        self.0.inner.inner
    }

    /// Returns the element at position `(i, j)`.
    #[inline]
    #[track_caller]
    pub fn read(&self, i: Idx<'nrows, usize>, j: Idx<'ncols, usize>) -> E {
        unsafe {
            self.0
                .inner
                .inner
                .read_unchecked(i.into_inner(), j.into_inner())
        }
    }

    /// Writes `value` to the location at position `(i, j)`.
    #[inline]
    #[track_caller]
    pub fn write(&mut self, i: Idx<'nrows, usize>, j: Idx<'ncols, usize>, value: E) {
        unsafe {
            self.0
                .inner
                .inner
                .write_unchecked(i.into_inner(), j.into_inner(), value)
        };
    }
}

impl<E: Entity> Clone for MatRef<'_, '_, '_, E> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Entity> Copy for MatRef<'_, '_, '_, E> {}

impl<'nrows, 'ncols, 'a, E: Entity> IntoConst for MatRef<'nrows, 'ncols, 'a, E> {
    type Target = MatRef<'nrows, 'ncols, 'a, E>;
    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}
impl<'nrows, 'ncols, 'a, 'short, E: Entity> Reborrow<'short> for MatRef<'nrows, 'ncols, 'a, E> {
    type Target = MatRef<'nrows, 'ncols, 'short, E>;
    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'nrows, 'ncols, 'a, 'short, E: Entity> ReborrowMut<'short> for MatRef<'nrows, 'ncols, 'a, E> {
    type Target = MatRef<'nrows, 'ncols, 'short, E>;
    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'nrows, 'ncols, 'a, E: Entity> IntoConst for MatMut<'nrows, 'ncols, 'a, E> {
    type Target = MatRef<'nrows, 'ncols, 'a, E>;
    #[inline]
    fn into_const(self) -> Self::Target {
        let inner = self.0.inner.inner.into_const();
        MatRef(Branded {
            __marker: PhantomData,
            inner: Branded {
                __marker: PhantomData,
                inner,
            },
        })
    }
}
impl<'nrows, 'ncols, 'a, 'short, E: Entity> Reborrow<'short> for MatMut<'nrows, 'ncols, 'a, E> {
    type Target = MatRef<'nrows, 'ncols, 'short, E>;
    #[inline]
    fn rb(&'short self) -> Self::Target {
        let inner = self.0.inner.inner.rb();
        MatRef(Branded {
            __marker: PhantomData,
            inner: Branded {
                __marker: PhantomData,
                inner,
            },
        })
    }
}
impl<'nrows, 'ncols, 'a, 'short, E: Entity> ReborrowMut<'short> for MatMut<'nrows, 'ncols, 'a, E> {
    type Target = MatMut<'nrows, 'ncols, 'short, E>;
    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        let inner = self.0.inner.inner.rb_mut();
        MatMut(Branded {
            __marker: PhantomData,
            inner: Branded {
                __marker: PhantomData,
                inner,
            },
        })
    }
}

impl<E: Entity> Debug for MatRef<'_, '_, '_, E> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.inner.inner.fmt(f)
    }
}
impl<E: Entity> Debug for MatMut<'_, '_, '_, E> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.inner.inner.fmt(f)
    }
}
