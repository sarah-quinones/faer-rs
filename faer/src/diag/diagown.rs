use super::*;
use crate::internal_prelude::{DiagMut, DiagRef};

/// see [`super::Diag`]
#[derive(Clone)]
pub struct Own<T, Dim: Shape = usize> {
	pub(crate) inner: Col<T, Dim>,
}

impl<T: core::fmt::Debug, Dim: Shape> core::fmt::Debug for Own<T, Dim> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.inner.fmt(f)
	}
}

impl<T, Dim: Shape> Diag<T, Dim> {
	/// returns the diagonal as a column vector
	#[inline(always)]
	pub fn column_vector(&self) -> ColRef<'_, T, Dim> {
		self.as_ref().column_vector()
	}

	/// returns the diagonal as a column vector
	#[inline(always)]
	pub fn column_vector_mut(&mut self) -> ColMut<'_, T, Dim> {
		self.as_mut().column_vector_mut()
	}

	/// returns the diagonal as a column vector
	#[inline(always)]
	pub fn into_column_vector(self) -> Col<T, Dim> {
		self.0.inner
	}

	/// returns a view over `self`
	#[inline(always)]
	pub fn as_ref(&self) -> DiagRef<'_, T, Dim> {
		DiagRef {
			0: Ref {
				inner: self.0.inner.as_ref(),
			},
		}
	}

	/// returns a view over `self`
	#[inline(always)]
	pub fn as_mut(&mut self) -> DiagMut<'_, T, Dim> {
		DiagMut {
			0: Mut {
				inner: self.0.inner.as_mut(),
			},
		}
	}

	#[inline]
	#[track_caller]
	/// see [`DiagRef::as_shape`]
	pub fn as_shape<D: Shape>(&self, len: D) -> DiagRef<'_, T, D> {
		DiagRef {
			0: Ref {
				inner: self.0.inner.as_row_shape(len),
			},
		}
	}

	#[inline]
	/// see [`DiagRef::as_dyn`]
	pub fn as_dyn(&self) -> DiagRef<'_, T> {
		self.as_ref().as_dyn()
	}

	#[inline]
	/// see [`DiagMut::as_dyn_mut`]
	pub fn as_dyn_mut(&mut self) -> DiagMut<'_, T> {
		self.as_mut().as_dyn_mut()
	}

	#[inline]
	/// see [`DiagRef::conjugate`]
	pub fn conjugate(&self) -> DiagRef<'_, T::Conj, Dim>
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
	pub fn canonical(&self) -> DiagRef<'_, T::Canonical, Dim>
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
	/// see [`DiagMut::as_shape_mut`]
	pub fn as_shape_mut<D: Shape>(&mut self, len: D) -> DiagMut<'_, T, D> {
		DiagMut {
			0: Mut {
				inner: self.0.inner.as_row_shape_mut(len),
			},
		}
	}

	#[inline]
	/// see [`DiagMut::conjugate_mut`]
	pub fn conjugate_mut(&mut self) -> DiagMut<'_, T::Conj, Dim>
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
	/// see [`DiagMut::canonical_mut`]
	pub fn canonical_mut(&mut self) -> DiagMut<'_, T::Canonical, Dim>
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

	/// returns a new diagonal with dimension `dim`, filled with zeros
	#[inline]
	pub fn zeros(dim: Dim) -> Self
	where
		T: ComplexField,
	{
		Self {
			0: Own { inner: Col::zeros(dim) },
		}
	}

	/// returns a new diagonal with dimension `dim`, filled with ones
	#[inline]
	pub fn ones(dim: Dim) -> Self
	where
		T: ComplexField,
	{
		Self {
			0: Own { inner: Col::ones(dim) },
		}
	}

	/// returns a new diagonal with dimension `dim`, filled with `value`
	#[inline]
	pub fn full(dim: Dim, value: T) -> Self
	where
		T: Clone,
	{
		Self {
			0: Own {
				inner: Col::full(dim, value),
			},
		}
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

impl<'short, T, Dim: Shape> Reborrow<'short> for Own<T, Dim> {
	type Target = Ref<'short, T, Dim>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		Ref { inner: self.inner.rb() }
	}
}
impl<'short, T, Dim: Shape> ReborrowMut<'short> for Own<T, Dim> {
	type Target = Mut<'short, T, Dim>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		Mut { inner: self.inner.rb_mut() }
	}
}
