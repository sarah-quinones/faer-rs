use super::*;

/// diagonal matrix view
pub struct DiagRef<'a, T, Dim = usize, Stride = isize> {
	pub(crate) inner: ColRef<'a, T, Dim, Stride>,
}

impl<T: core::fmt::Debug, Dim: Shape, S: Stride> core::fmt::Debug for DiagRef<'_, T, Dim, S> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.inner.fmt(f)
	}
}

impl<T, Dim: Copy, Stride: Copy> Copy for DiagRef<'_, T, Dim, Stride> {}
impl<T, Dim: Copy, Stride: Copy> Clone for DiagRef<'_, T, Dim, Stride> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

impl<'short, T, Dim: Copy, Stride: Copy> Reborrow<'short> for DiagRef<'_, T, Dim, Stride> {
	type Target = DiagRef<'short, T, Dim, Stride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}
impl<'short, T, Dim: Copy, Stride: Copy> ReborrowMut<'short> for DiagRef<'_, T, Dim, Stride> {
	type Target = DiagRef<'short, T, Dim, Stride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}
impl<'a, T, Dim: Copy, Stride: Copy> IntoConst for DiagRef<'a, T, Dim, Stride> {
	type Target = DiagRef<'a, T, Dim, Stride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}

unsafe impl<T: Sync, Dim: Sync, Stride: Sync> Sync for DiagRef<'_, T, Dim, Stride> {}
unsafe impl<T: Sync, Dim: Send, Stride: Send> Send for DiagRef<'_, T, Dim, Stride> {}

impl<'a, T, Dim: Shape, Stride: crate::Stride> DiagRef<'a, T, Dim, Stride> {
	/// returns the diagonal as a column vector view.
	#[inline(always)]
	pub fn column_vector(self) -> ColRef<'a, T, Dim, Stride> {
		self.inner
	}

	/// returns a view over `self`
	#[inline]
	pub fn as_ref(&self) -> DiagRef<'_, T, Dim, Stride> {
		*self
	}

	/// returns the input matrix with the given shape after checking that it matches the
	/// current shape
	#[inline]
	#[track_caller]
	pub fn as_shape<D: Shape>(self, len: D) -> DiagRef<'a, T, D, Stride> {
		DiagRef {
			inner: self.inner.as_row_shape(len),
		}
	}

	/// returns the input matrix with dynamic shape
	#[inline]
	pub fn as_dyn(self) -> DiagRef<'a, T, usize, Stride> {
		DiagRef {
			inner: self.inner.as_dyn_rows(),
		}
	}

	/// returns the input matrix with dynamic stride
	#[inline]
	pub fn as_dyn_stride(self) -> DiagRef<'a, T, Dim> {
		DiagRef {
			inner: self.inner.as_dyn_stride(),
		}
	}

	/// returns a view over the conjugate of `self`
	#[inline]
	pub fn conjugate(self) -> DiagRef<'a, T::Conj, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagRef {
			inner: self.inner.conjugate(),
		}
	}

	/// returns an unconjugated view over `self`
	#[inline]
	pub fn canonical(self) -> DiagRef<'a, T::Canonical, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagRef {
			inner: self.inner.canonical(),
		}
	}

	/// returns the dimension of `self`
	#[inline]
	pub fn dim(&self) -> Dim {
		self.inner.nrows()
	}
}
