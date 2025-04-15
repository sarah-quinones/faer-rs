use super::*;
use crate::internal_prelude::DiagRef;

/// see [`super::DiagMut`]
pub struct Mut<'a, T, Dim = usize, Stride = isize> {
	pub(crate) inner: ColMut<'a, T, Dim, Stride>,
}

impl<T: core::fmt::Debug, Dim: Shape, S: Stride> core::fmt::Debug for Mut<'_, T, Dim, S> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.inner.fmt(f)
	}
}

impl<'a, T> DiagMut<'a, T> {
	/// creates a diagonal matrix view over the given element
	#[inline]
	pub fn from_mut(value: &'a mut T) -> Self {
		unsafe { DiagMut::from_raw_parts_mut(value as *mut T, 1, 1) }
	}

	/// creates a `DiagMut` from slice views over the diagonal data, the result has the same
	/// dimension as the length of the input slice
	#[inline]
	pub fn from_slice_mut(slice: &'a mut [T]) -> Self {
		let len = slice.len();
		unsafe { Self::from_raw_parts_mut(slice.as_mut_ptr(), len, 1) }
	}
}

impl<'a, T, Dim: Shape, Stride: crate::Stride> DiagMut<'a, T, Dim, Stride> {
	/// creates a `DiagMut` from pointers to the diagonal data, dimension, and stride
	///
	/// # safety
	/// this function has the same safety requirements as
	/// [`MatMut::from_raw_parts_mut(ptr, dim, 1, stride, 0)`]
	#[inline(always)]
	#[track_caller]
	pub const unsafe fn from_raw_parts_mut(ptr: *mut T, dim: Dim, stride: Stride) -> Self {
		Self {
			0: Mut {
				inner: ColMut::from_raw_parts_mut(ptr, dim, stride),
			},
		}
	}

	/// returns the diagonal as a column vector view
	#[inline(always)]
	pub fn column_vector(self) -> ColRef<'a, T, Dim, Stride> {
		self.into_const().column_vector()
	}

	/// returns the diagonal as a mutable column vector view
	#[inline(always)]
	pub fn column_vector_mut(self) -> ColMut<'a, T, Dim, Stride> {
		self.0.inner
	}

	/// returns a view over `self`
	#[inline]
	pub fn as_ref(&self) -> DiagRef<'_, T, Dim, Stride> {
		self.rb()
	}

	/// returns a view over `self`
	#[inline]
	pub fn as_mut(&mut self) -> DiagMut<'_, T, Dim, Stride> {
		self.rb_mut()
	}

	/// fills all the elements of `self` with `value`
	#[inline]
	pub fn fill(&mut self, value: T)
	where
		T: Clone,
	{
		self.0.inner.fill(value)
	}

	#[inline]
	#[track_caller]
	/// see [`DiagRef::as_shape`]
	pub fn as_shape<D: Shape>(self, len: D) -> DiagRef<'a, T, D, Stride> {
		DiagRef {
			0: Ref {
				inner: self.0.inner.as_row_shape(len),
			},
		}
	}

	#[inline]
	/// see [`DiagRef::as_dyn`]
	pub fn as_dyn(self) -> DiagRef<'a, T, usize, Stride> {
		DiagRef {
			0: Ref {
				inner: self.0.inner.as_dyn_rows(),
			},
		}
	}

	#[inline]
	/// see [`DiagRef::as_dyn_stride`]
	pub fn as_dyn_stride(self) -> DiagRef<'a, T, Dim> {
		DiagRef {
			0: Ref {
				inner: self.0.inner.as_dyn_stride(),
			},
		}
	}

	#[inline]
	/// see [`DiagRef::conjugate`]
	pub fn conjugate(self) -> DiagRef<'a, T::Conj, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagRef {
			0: Ref {
				inner: self.0.inner.conjugate(),
			},
		}
	}

	#[inline]
	/// see [`DiagRef::canonical`]
	pub fn canonical(self) -> DiagRef<'a, T::Canonical, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagRef {
			0: Ref {
				inner: self.0.inner.canonical(),
			},
		}
	}

	#[inline]
	#[track_caller]
	/// see [`DiagRef::as_shape`]
	pub fn as_shape_mut<D: Shape>(self, len: D) -> DiagMut<'a, T, D, Stride> {
		DiagMut {
			0: Mut {
				inner: self.0.inner.as_row_shape_mut(len),
			},
		}
	}

	#[inline]
	/// see [`DiagRef::as_dyn`]
	pub fn as_dyn_mut(self) -> DiagMut<'a, T, usize, Stride> {
		DiagMut {
			0: Mut {
				inner: self.0.inner.as_dyn_rows_mut(),
			},
		}
	}

	#[inline]
	/// see [`DiagRef::as_dyn_stride`]
	pub fn as_dyn_stride_mut(self) -> DiagMut<'a, T, Dim> {
		DiagMut {
			0: Mut {
				inner: self.0.inner.as_dyn_stride_mut(),
			},
		}
	}

	#[inline]
	/// see [`DiagRef::conjugate`]
	pub fn conjugate_mut(self) -> DiagMut<'a, T::Conj, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagMut {
			0: Mut {
				inner: self.0.inner.conjugate_mut(),
			},
		}
	}

	#[inline]
	/// see [`DiagRef::canonical`]
	pub fn canonical_mut(self) -> DiagMut<'a, T::Canonical, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagMut {
			0: Mut {
				inner: self.0.inner.canonical_mut(),
			},
		}
	}

	/// returns the dimension of `self`
	#[inline]
	pub fn dim(&self) -> Dim {
		self.0.inner.nrows()
	}

	/// copies `other` into `self`
	#[inline]
	#[track_caller]
	pub fn copy_from<RhsT: Conjugate<Canonical = T>>(&mut self, rhs: impl AsDiagRef<T = RhsT, Dim = Dim>)
	where
		T: ComplexField,
	{
		self.0.inner.copy_from(rhs.as_diag_ref().inner)
	}
}

impl<'short, T, N: Copy, Stride: Copy> Reborrow<'short> for Mut<'_, T, N, Stride> {
	type Target = Ref<'short, T, N, Stride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		Ref { inner: self.inner.rb() }
	}
}

impl<'short, T, N: Copy, Stride: Copy> ReborrowMut<'short> for Mut<'_, T, N, Stride> {
	type Target = Mut<'short, T, N, Stride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		Mut { inner: self.inner.rb_mut() }
	}
}

impl<'a, T, N: Copy, Stride: Copy> IntoConst for Mut<'a, T, N, Stride> {
	type Target = Ref<'a, T, N, Stride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		Ref {
			inner: self.inner.into_const(),
		}
	}
}
