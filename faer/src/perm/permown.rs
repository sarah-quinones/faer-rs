use super::*;
use crate::{Idx, assert};

extern crate alloc;

/// see [`super::Perm`]
#[derive(Debug, Clone)]
pub struct Own<I: Index, N: Shape = usize> {
	pub(super) forward: alloc::boxed::Box<[N::Idx<I>]>,
	pub(super) inverse: alloc::boxed::Box<[N::Idx<I>]>,
}

impl<I: Index, N: Shape> Perm<I, N> {
	/// returns the input permutation with the given shape after checking that it matches the
	/// current shape
	#[inline]
	pub fn as_shape<M: Shape>(&self, dim: M) -> PermRef<'_, I, M> {
		self.as_ref().as_shape(dim)
	}

	/// returns the input permutation with the given shape after checking that it matches the
	/// current shape
	#[inline]
	pub fn into_shape<M: Shape>(self, dim: M) -> Perm<I, M> {
		assert!(self.len().unbound() == dim.unbound());

		Perm {
			0: Own {
				forward: unsafe { alloc::boxed::Box::from_raw(alloc::boxed::Box::into_raw(self.0.forward) as _) },
				inverse: unsafe { alloc::boxed::Box::from_raw(alloc::boxed::Box::into_raw(self.0.inverse) as _) },
			},
		}
	}

	/// creates a new permutation, by checking the validity of the inputs
	///
	/// # panics
	///
	/// the function panics if any of the following conditions are violated:
	/// `forward` and `inverse` must have the same length which must be less than or equal to
	/// `I::Signed::MAX`, be valid permutations, and be inverse permutations of each other
	#[inline]
	#[track_caller]
	pub fn new_checked(forward: alloc::boxed::Box<[Idx<N, I>]>, inverse: alloc::boxed::Box<[Idx<N, I>]>, dim: N) -> Self {
		PermRef::<'_, I, N>::new_checked(&forward, &inverse, dim);
		Self { 0: Own { forward, inverse } }
	}

	/// creates a new permutation reference, without checking the validity of the inputs
	///
	/// # safety
	///
	/// `forward` and `inverse` must have the same length which must be less than or equal to
	/// `I::Signed::MAX`, be valid permutations, and be inverse permutations of each other
	#[inline]
	#[track_caller]
	pub unsafe fn new_unchecked(forward: alloc::boxed::Box<[Idx<N, I>]>, inverse: alloc::boxed::Box<[Idx<N, I>]>) -> Self {
		let n = forward.len();
		assert!(all(forward.len() == inverse.len(), n <= I::Signed::MAX.zx(),));
		Self { 0: Own { forward, inverse } }
	}

	/// returns the permutation as an array
	#[inline]
	pub fn into_arrays(self) -> (alloc::boxed::Box<[Idx<N, I>]>, alloc::boxed::Box<[Idx<N, I>]>) {
		(self.0.forward, self.0.inverse)
	}

	/// returns the dimension of the permutation
	#[inline]
	pub fn len(&self) -> N {
		unsafe { N::new_unbound(self.0.forward.len()) }
	}

	/// returns the inverse permutation
	#[inline]
	pub fn into_inverse(self) -> Self {
		Self {
			0: Own {
				forward: self.0.inverse,
				inverse: self.0.forward,
			},
		}
	}
}

impl<'short, I: Index, N: Shape> Reborrow<'short> for Own<I, N> {
	type Target = Ref<'short, I, N>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		Ref {
			forward: &self.forward,
			inverse: &self.inverse,
		}
	}
}
