use super::*;
use crate::internal_prelude::{DiagMut, DiagRef};

/// Diagonal matrix.
#[derive(Clone, Debug)]
pub struct Diag<T, Dim: Shape = usize> {
	pub(crate) inner: Col<T, Dim>,
}

impl<T, Dim: Shape> Diag<T, Dim> {
	/// Returns the diagonal as a column vector view.
	#[inline(always)]
	pub fn column_vector(&self) -> ColRef<'_, T, Dim> {
		self.as_ref().column_vector()
	}

	/// Returns the diagonal as a mutable column vector view.
	#[inline(always)]
	pub fn column_vector_mut(&mut self) -> ColMut<'_, T, Dim> {
		self.as_mut().column_vector_mut()
	}

	/// Returns the diagonal as a column vector.
	#[inline(always)]
	pub fn into_column_vector(self) -> Col<T, Dim> {
		self.inner
	}

	/// Returns a view over `self`.
	#[inline(always)]
	pub fn as_ref(&self) -> DiagRef<'_, T, Dim> {
		DiagRef { inner: self.inner.as_ref() }
	}

	/// Returns a mutable view over `self`.
	#[inline(always)]
	pub fn as_mut(&mut self) -> DiagMut<'_, T, Dim> {
		DiagMut { inner: self.inner.as_mut() }
	}

	#[inline]
	#[track_caller]
	pub fn as_shape<D: Shape>(&self, len: D) -> DiagRef<'_, T, D> {
		DiagRef { inner: self.inner.as_row_shape(len) }
	}

	#[inline]
	pub fn as_dyn(&self) -> DiagRef<'_, T> {
		self.as_ref().as_dyn()
	}

	#[inline]
	pub fn as_dyn_mut(&mut self) -> DiagMut<'_, T> {
		self.as_mut().as_dyn_mut()
	}

	#[inline]
	pub fn conjugate(&self) -> DiagRef<'_, T::Conj, Dim>
	where
		T: Conjugate,
	{
		DiagRef { inner: self.inner.conjugate() }
	}

	#[inline]
	pub fn canonical(&self) -> DiagRef<'_, T::Canonical, Dim>
	where
		T: Conjugate,
	{
		DiagRef { inner: self.inner.canonical() }
	}

	#[inline]
	#[track_caller]
	pub fn as_shape_mut<D: Shape>(&mut self, len: D) -> DiagMut<'_, T, D> {
		DiagMut {
			inner: self.inner.as_row_shape_mut(len),
		}
	}

	#[inline]
	pub fn conjugate_mut(&mut self) -> DiagMut<'_, T::Conj, Dim>
	where
		T: Conjugate,
	{
		DiagMut { inner: self.inner.conjugate_mut() }
	}

	#[inline]
	pub fn canonical_mut(&mut self) -> DiagMut<'_, T::Canonical, Dim>
	where
		T: Conjugate,
	{
		DiagMut { inner: self.inner.canonical_mut() }
	}

	#[inline]
	pub fn dim(&self) -> Dim {
		self.inner.nrows()
	}

	#[inline]
	pub fn zeros(dim: Dim) -> Self
	where
		T: ComplexField,
	{
		Self { inner: Col::zeros(dim) }
	}

	#[inline]
	pub fn ones(dim: Dim) -> Self
	where
		T: ComplexField,
	{
		Self { inner: Col::ones(dim) }
	}

	#[inline]
	pub fn full(dim: Dim, value: T) -> Self
	where
		T: Clone,
	{
		Self { inner: Col::full(dim, value) }
	}

	#[inline]
	#[track_caller]
	pub fn copy_from<RhsT: Conjugate<Canonical = T>>(&mut self, rhs: impl AsDiagRef<T = RhsT, Dim = Dim>)
	where
		T: ComplexField,
	{
		self.inner.copy_from(rhs.as_diag_ref().inner)
	}
}
