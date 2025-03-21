use super::*;
use crate::Idx;
use crate::internal_prelude::DiagRef;
use core::ops::{Index, IndexMut};

/// diagonal mutable matrix view
pub struct DiagMut<'a, T, Dim = usize, Stride = isize> {
	pub(crate) inner: ColMut<'a, T, Dim, Stride>,
}

impl<T: core::fmt::Debug, Dim: Shape, S: Stride> core::fmt::Debug for DiagMut<'_, T, Dim, S> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.inner.fmt(f)
	}
}

impl<'a, T, Dim: Shape, Stride: crate::Stride> DiagMut<'a, T, Dim, Stride> {
	/// returns the diagonal as a column vector view
	#[inline(always)]
	pub fn column_vector(self) -> ColRef<'a, T, Dim, Stride> {
		self.into_const().column_vector()
	}

	/// returns the diagonal as a mutable column vector view
	#[inline(always)]
	pub fn column_vector_mut(self) -> ColMut<'a, T, Dim, Stride> {
		self.inner
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
		self.inner.fill(value)
	}

	#[inline]
	#[track_caller]
	/// see [`DiagRef::as_shape`]
	pub fn as_shape<D: Shape>(self, len: D) -> DiagRef<'a, T, D, Stride> {
		DiagRef {
			inner: self.inner.as_row_shape(len),
		}
	}

	#[inline]
	/// see [`DiagRef::as_dyn`]
	pub fn as_dyn(self) -> DiagRef<'a, T, usize, Stride> {
		DiagRef {
			inner: self.inner.as_dyn_rows(),
		}
	}

	#[inline]
	/// see [`DiagRef::as_dyn_stride`]
	pub fn as_dyn_stride(self) -> DiagRef<'a, T, Dim> {
		DiagRef {
			inner: self.inner.as_dyn_stride(),
		}
	}

	#[inline]
	/// see [`DiagRef::conjugate`]
	pub fn conjugate(self) -> DiagRef<'a, T::Conj, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagRef {
			inner: self.inner.conjugate(),
		}
	}

	#[inline]
	/// see [`DiagRef::canonical`]
	pub fn canonical(self) -> DiagRef<'a, T::Canonical, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagRef {
			inner: self.inner.canonical(),
		}
	}

	#[inline]
	#[track_caller]
	/// see [`DiagRef::as_shape`]
	pub fn as_shape_mut<D: Shape>(self, len: D) -> DiagMut<'a, T, D, Stride> {
		DiagMut {
			inner: self.inner.as_row_shape_mut(len),
		}
	}

	#[inline]
	/// see [`DiagRef::as_dyn`]
	pub fn as_dyn_mut(self) -> DiagMut<'a, T, usize, Stride> {
		DiagMut {
			inner: self.inner.as_dyn_rows_mut(),
		}
	}

	#[inline]
	/// see [`DiagRef::as_dyn_stride`]
	pub fn as_dyn_stride_mut(self) -> DiagMut<'a, T, Dim> {
		DiagMut {
			inner: self.inner.as_dyn_stride_mut(),
		}
	}

	#[inline]
	/// see [`DiagRef::conjugate`]
	pub fn conjugate_mut(self) -> DiagMut<'a, T::Conj, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagMut {
			inner: self.inner.conjugate_mut(),
		}
	}

	#[inline]
	/// see [`DiagRef::canonical`]
	pub fn canonical_mut(self) -> DiagMut<'a, T::Canonical, Dim, Stride>
	where
		T: Conjugate,
	{
		DiagMut {
			inner: self.inner.canonical_mut(),
		}
	}

	/// returns the dimension of `self`
	#[inline]
	pub fn dim(&self) -> Dim {
		self.inner.nrows()
	}

	/// copies `other` into `self`
	#[inline]
	#[track_caller]
	pub fn copy_from<RhsT: Conjugate<Canonical = T>>(&mut self, rhs: impl AsDiagRef<T = RhsT, Dim = Dim>)
	where
		T: ComplexField,
	{
		self.inner.copy_from(rhs.as_diag_ref().inner)
	}
}

impl<'short, T, N: Copy, Stride: Copy> Reborrow<'short> for DiagMut<'_, T, N, Stride> {
	type Target = DiagRef<'short, T, N, Stride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		DiagRef { inner: self.inner.rb() }
	}
}

impl<'short, T, N: Copy, Stride: Copy> ReborrowMut<'short> for DiagMut<'_, T, N, Stride> {
	type Target = DiagMut<'short, T, N, Stride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		DiagMut { inner: self.inner.rb_mut() }
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

impl<T, Dim: Shape, Stride: crate::Stride> Index<Idx<Dim>> for DiagRef<'_, T, Dim, Stride> {
	type Output = T;

	#[inline]
	#[track_caller]
	fn index(&self, idx: Idx<Dim>) -> &Self::Output {
		self.inner.at(idx)
	}
}

impl<T, Dim: Shape, Stride: crate::Stride> Index<Idx<Dim>> for DiagMut<'_, T, Dim, Stride> {
	type Output = T;

	#[inline]
	#[track_caller]
	fn index(&self, idx: Idx<Dim>) -> &Self::Output {
		self.rb().inner.at(idx)
	}
}

impl<T, Dim: Shape, Stride: crate::Stride> IndexMut<Idx<Dim>> for DiagMut<'_, T, Dim, Stride> {
	#[inline]
	#[track_caller]
	fn index_mut(&mut self, idx: Idx<Dim>) -> &mut Self::Output {
		self.inner.rb_mut().at_mut(idx)
	}
}

impl<T, Dim: Shape> Index<Idx<Dim>> for Diag<T, Dim> {
	type Output = T;

	#[inline]
	#[track_caller]
	fn index(&self, idx: Idx<Dim>) -> &Self::Output {
		&self.inner[idx]
	}
}

impl<T, Dim: Shape> IndexMut<Idx<Dim>> for Diag<T, Dim> {
	#[inline]
	#[track_caller]
	fn index_mut(&mut self, idx: Idx<Dim>) -> &mut Self::Output {
		&mut self.inner[idx]
	}
}
