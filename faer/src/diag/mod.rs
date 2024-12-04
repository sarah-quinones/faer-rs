use crate::internal_prelude::*;

pub(crate) mod diagmut;
pub(crate) mod diagown;
pub(crate) mod diagref;

pub use diagmut::DiagMut;
pub use diagown::Diag;
pub use diagref::DiagRef;

pub trait AsDiagMut: AsDiagRef {
	fn as_diag_mut(&mut self) -> DiagMut<Self::T, Self::Dim>;
}
pub trait AsDiagRef {
	type T;
	type Dim: Shape;

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
