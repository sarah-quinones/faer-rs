use crate::internal_prelude::*;

pub(crate) mod diagmut;
pub(crate) mod diagown;
pub(crate) mod diagref;

pub use diagmut::Mut;
pub use diagown::Own;
pub use diagref::Ref;

/// diagonal matrix view
pub type DiagRef<'a, T, Dim = usize, Stride = isize> = generic::Diag<Ref<'a, T, Dim, Stride>>;
/// diagonal mutable matrix view
pub type DiagMut<'a, T, Dim = usize, Stride = isize> = generic::Diag<Mut<'a, T, Dim, Stride>>;
/// diagonal matrix
pub type Diag<T, Dim = usize> = generic::Diag<Own<T, Dim>>;

/// generic `Diag` wrapper
pub mod generic {
	use crate::{Idx, Shape};
	use core::fmt::Debug;
	use core::ops::{Index, IndexMut};
	use reborrow::*;

	/// generic `Diag` wrapper
	#[derive(Copy, Clone)]
	#[repr(transparent)]
	pub struct Diag<Inner>(pub Inner);

	impl<Inner: Debug> Debug for Diag<Inner> {
		#[inline(always)]
		fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
			self.0.fmt(f)
		}
	}

	impl<Inner> Diag<Inner> {
		/// wrap by reference
		#[inline(always)]
		pub fn from_inner_ref(inner: &Inner) -> &Self {
			unsafe { &*(inner as *const Inner as *const Self) }
		}

		/// wrap by mutable reference
		#[inline(always)]
		pub fn from_inner_mut(inner: &mut Inner) -> &mut Self {
			unsafe { &mut *(inner as *mut Inner as *mut Self) }
		}
	}

	impl<Inner> core::ops::Deref for Diag<Inner> {
		type Target = Inner;

		#[inline(always)]
		fn deref(&self) -> &Self::Target {
			&self.0
		}
	}

	impl<Inner> core::ops::DerefMut for Diag<Inner> {
		#[inline(always)]
		fn deref_mut(&mut self) -> &mut Self::Target {
			&mut self.0
		}
	}

	impl<'short, Inner: Reborrow<'short>> Reborrow<'short> for Diag<Inner> {
		type Target = Diag<Inner::Target>;

		#[inline(always)]
		fn rb(&'short self) -> Self::Target {
			Diag(self.0.rb())
		}
	}

	impl<'short, Inner: ReborrowMut<'short>> ReborrowMut<'short> for Diag<Inner> {
		type Target = Diag<Inner::Target>;

		#[inline(always)]
		fn rb_mut(&'short mut self) -> Self::Target {
			Diag(self.0.rb_mut())
		}
	}

	impl<Inner: IntoConst> IntoConst for Diag<Inner> {
		type Target = Diag<Inner::Target>;

		#[inline(always)]
		fn into_const(self) -> Self::Target {
			Diag(self.0.into_const())
		}
	}

	impl<T, Dim: Shape, Stride: crate::Stride, Inner: for<'short> Reborrow<'short, Target = super::Ref<'short, T, Dim, Stride>>> Index<Idx<Dim>>
		for Diag<Inner>
	{
		type Output = T;

		#[inline]
		#[track_caller]
		fn index(&self, idx: Idx<Dim>) -> &Self::Output {
			self.rb().column_vector().at(idx)
		}
	}

	impl<
		T,
		Dim: Shape,
		Stride: crate::Stride,
		Inner: for<'short> Reborrow<'short, Target = super::Ref<'short, T, Dim, Stride>>
			+ for<'short> ReborrowMut<'short, Target = super::Mut<'short, T, Dim, Stride>>,
	> IndexMut<Idx<Dim>> for Diag<Inner>
	{
		#[inline]
		#[track_caller]
		fn index_mut(&mut self, idx: Idx<Dim>) -> &mut Self::Output {
			self.rb_mut().column_vector_mut().at_mut(idx)
		}
	}
}
/// trait for types that can be converted to a diagonal matrix view.
pub trait AsDiagMut: AsDiagRef {
	/// returns a view over `self`
	fn as_diag_mut(&mut self) -> DiagMut<Self::T, Self::Dim>;
}
/// trait for types that can be converted to a diagonal matrix view.
pub trait AsDiagRef {
	/// scalar type
	type T;
	/// dimension type
	type Dim: Shape;

	/// returns a view over `self`
	fn as_diag_ref(&self) -> DiagRef<Self::T, Self::Dim>;
}

impl<T, Dim: Shape, Stride: crate::Stride> AsDiagRef for DiagRef<'_, T, Dim, Stride> {
	type Dim = Dim;
	type T = T;

	#[inline]
	fn as_diag_ref(&self) -> DiagRef<Self::T, Self::Dim> {
		self.as_dyn_stride()
	}
}
