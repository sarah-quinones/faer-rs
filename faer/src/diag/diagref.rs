use super::*;

/// see [`super::DiagRef`]
pub struct Ref<'a, T, Dim = usize, Stride = isize> {
	pub(crate) inner: ColRef<'a, T, Dim, Stride>,
}

impl<T: core::fmt::Debug, Dim: Shape, S: Stride> core::fmt::Debug for Ref<'_, T, Dim, S> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.inner.fmt(f)
	}
}

impl<T, Dim: Copy, Stride: Copy> Copy for Ref<'_, T, Dim, Stride> {}
impl<T, Dim: Copy, Stride: Copy> Clone for Ref<'_, T, Dim, Stride> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

impl<'short, T, Dim: Copy, Stride: Copy> Reborrow<'short> for Ref<'_, T, Dim, Stride> {
	type Target = Ref<'short, T, Dim, Stride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}
impl<'short, T, Dim: Copy, Stride: Copy> ReborrowMut<'short> for Ref<'_, T, Dim, Stride> {
	type Target = Ref<'short, T, Dim, Stride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}
impl<'a, T, Dim: Copy, Stride: Copy> IntoConst for Ref<'a, T, Dim, Stride> {
	type Target = Ref<'a, T, Dim, Stride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}

impl<'a, T> DiagRef<'a, T> {
	/// creates a diagonal matrix view over the given element
	#[inline]
	pub fn from_ref(value: &'a T) -> Self {
		unsafe { DiagRef::from_raw_parts(value as *const T, 1, 1) }
	}

	/// creates a `DiagRef` from slice views over the diagonal data, the result has the same
	/// dimension as the length of the input slice
	#[inline]
	pub fn from_slice(slice: &'a [T]) -> Self {
		let len = slice.len();
		unsafe { Self::from_raw_parts(slice.as_ptr(), len, 1) }
	}
}

impl<'a, T, Dim: Shape, Stride: crate::Stride> DiagRef<'a, T, Dim, Stride> {
	/// creates a `DiagRef` from pointers to the diagonal data, dimension, and stride
	///
	/// # safety
	/// this function has the same safety requirements as
	/// [`MatRef::from_raw_parts(ptr, dim, 1, stride, 0)`]
	#[inline(always)]
	#[track_caller]
	pub const unsafe fn from_raw_parts(ptr: *const T, dim: Dim, stride: Stride) -> Self {
		Self {
			0: Ref {
				inner: ColRef::from_raw_parts(ptr, dim, stride),
			},
		}
	}

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
			0: Ref {
				inner: self.inner.as_row_shape(len),
			},
		}
	}

	/// returns the input matrix with dynamic shape
	#[inline]
	pub fn as_dyn(self) -> DiagRef<'a, T, usize, Stride> {
		DiagRef {
			0: Ref {
				inner: self.inner.as_dyn_rows(),
			},
		}
	}

	/// returns the input matrix with dynamic stride
	#[inline]
	pub fn as_dyn_stride(self) -> DiagRef<'a, T, Dim> {
		DiagRef {
			0: Ref {
				inner: self.inner.as_dyn_stride(),
			},
		}
	}

	/// returns a view over the conjugate of `self`
	#[inline]
	pub fn conjugate(self) -> DiagRef<'a, T::Conj, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagRef {
			0: Ref {
				inner: self.inner.conjugate(),
			},
		}
	}

	/// returns an unconjugated view over `self`
	#[inline]
	pub fn canonical(self) -> DiagRef<'a, T::Canonical, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagRef {
			0: Ref {
				inner: self.inner.canonical(),
			},
		}
	}

	/// returns the dimension of `self`
	#[inline]
	pub fn dim(&self) -> Dim {
		self.inner.nrows()
	}
}

impl<T, Dim: Shape, Stride: crate::Stride, Inner: for<'short> Reborrow<'short, Target = Ref<'short, T, Dim, Stride>>> generic::Diag<Inner> {
	/// returns `true` if all of the elements of `self` are finite.
	/// otherwise returns `false`.
	#[inline]
	pub fn is_all_finite(&self) -> bool
	where
		T: Conjugate,
	{
		self.rb().column_vector().is_all_finite()
	}

	/// returns `true` if any of the elements of `self` is `NaN`.
	/// otherwise returns `false`.
	#[inline]
	pub fn has_nan(&self) -> bool
	where
		T: Conjugate,
	{
		self.rb().column_vector().has_nan()
	}
}
