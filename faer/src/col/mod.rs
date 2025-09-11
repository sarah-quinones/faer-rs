use crate::internal_prelude::*;
use crate::utils::bound::{One, Zero};
use core::ptr::NonNull;

/// represents a type that can be used to slice a column, such as an index or a range of indices
pub trait ColIndex<RowRange> {
	/// sliced view type
	type Target;

	/// slice `this` using `row`
	fn get(this: Self, row: RowRange) -> Self::Target;
	/// slice `this` using `row`, without bound checks
	unsafe fn get_unchecked(this: Self, row: RowRange) -> Self::Target;
}

struct ColView<T: ?Sized, Rows, RStride> {
	ptr: NonNull<T>,
	nrows: Rows,
	row_stride: RStride,
}

impl<T: ?Sized, Rows: Copy, RStride: Copy> Copy for ColView<T, Rows, RStride> {}
impl<T: ?Sized, Rows: Copy, RStride: Copy> Clone for ColView<T, Rows, RStride> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}

mod col_index;

pub(crate) mod colmut;
pub(crate) mod colown;
pub(crate) mod colref;

pub use colmut::Mut;
pub use colown::Own;
pub use colref::Ref;
use mat::AsMat;

/// immutable view over a column vector, similar to an immutable reference to a strided
/// [prim@slice]
///
/// # note
///
/// unlike a slice, the data pointed to by `ColRef<'_, T>` is allowed to be partially or fully
/// uninitialized under certain conditions. in this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly or
/// indirectly through any of the numerical library routines, unless it is explicitly permitted
pub type ColRef<'a, T, Rows = usize, RStride = isize> = generic::Col<Ref<'a, T, Rows, RStride>>;

/// mutable view over a column vector, similar to a mutable reference to a strided
/// [prim@slice]
///
/// # note
///
/// unlike a slice, the data pointed to by `ColMut<'_, T>` is allowed to be partially or fully
/// uninitialized under certain conditions. in this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly or
/// indirectly through any of the numerical library routines, unless it is explicitly permitted
pub type ColMut<'a, T, Rows = usize, RStride = isize> = generic::Col<Mut<'a, T, Rows, RStride>>;

/// heap allocated resizable column vector.
///
/// # note
///
/// the memory layout of `Col` is guaranteed to be column-major, meaning that it has a row stride
/// of `1`.
pub type Col<T, Rows = usize> = generic::Col<Own<T, Rows>>;

/// generic `Col` wrapper
pub mod generic {
	use crate::{Idx, Shape, Stride};
	use core::fmt::Debug;
	use core::ops::{Index, IndexMut};
	use reborrow::*;

	/// generic `Col` wrapper
	#[derive(Copy, Clone)]
	#[repr(transparent)]
	pub struct Col<Inner>(pub Inner);

	impl<Inner: Debug> Debug for Col<Inner> {
		#[inline(always)]
		fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
			self.0.fmt(f)
		}
	}

	impl<Inner> Col<Inner> {
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

	impl<Inner> core::ops::Deref for Col<Inner> {
		type Target = Inner;

		#[inline(always)]
		fn deref(&self) -> &Self::Target {
			&self.0
		}
	}

	impl<Inner> core::ops::DerefMut for Col<Inner> {
		#[inline(always)]
		fn deref_mut(&mut self) -> &mut Self::Target {
			&mut self.0
		}
	}

	impl<'short, Inner: Reborrow<'short>> Reborrow<'short> for Col<Inner> {
		type Target = Col<Inner::Target>;

		#[inline(always)]
		fn rb(&'short self) -> Self::Target {
			Col(self.0.rb())
		}
	}

	impl<'short, Inner: ReborrowMut<'short>> ReborrowMut<'short> for Col<Inner> {
		type Target = Col<Inner::Target>;

		#[inline(always)]
		fn rb_mut(&'short mut self) -> Self::Target {
			Col(self.0.rb_mut())
		}
	}

	impl<Inner: IntoConst> IntoConst for Col<Inner> {
		type Target = Col<Inner::Target>;

		#[inline(always)]
		fn into_const(self) -> Self::Target {
			Col(self.0.into_const())
		}
	}

	impl<T, Rows: Shape, RStride: Stride, Inner: for<'short> Reborrow<'short, Target = super::Ref<'short, T, Rows, RStride>>> Index<Idx<Rows>>
		for Col<Inner>
	{
		type Output = T;

		#[inline]
		#[track_caller]
		fn index(&self, row: Idx<Rows>) -> &Self::Output {
			self.rb().at(row)
		}
	}

	impl<
		T,
		Rows: Shape,
		RStride: Stride,
		Inner: for<'short> Reborrow<'short, Target = super::Ref<'short, T, Rows, RStride>>
			+ for<'short> ReborrowMut<'short, Target = super::Mut<'short, T, Rows, RStride>>,
	> IndexMut<Idx<Rows>> for Col<Inner>
	{
		#[inline]
		#[track_caller]
		fn index_mut(&mut self, row: Idx<Rows>) -> &mut Self::Output {
			self.rb_mut().at_mut(row)
		}
	}
}

/// trait for types that can be converted to a column view
pub trait AsColMut: AsColRef {
	/// returns a view over `self`
	fn as_col_mut(&mut self) -> ColMut<'_, Self::T, Self::Rows>;
}
/// trait for types that can be converted to a column view
pub trait AsColRef: AsMatRef<Cols = One> {
	/// returns a view over `self`
	fn as_col_ref(&self) -> ColRef<'_, Self::T, Self::Rows>;
}

impl<M: AsMatRef<Cols = One>> AsColRef for M {
	#[inline]
	fn as_col_ref(&self) -> ColRef<'_, Self::T, Self::Rows> {
		self.as_mat_ref().col(Zero)
	}
}

impl<M: AsMatMut<Cols = One>> AsColMut for M {
	#[inline]
	fn as_col_mut(&mut self) -> ColMut<'_, Self::T, Self::Rows> {
		self.as_mat_mut().col_mut(Zero)
	}
}

impl<T, Rows: Shape, Rs: Stride> AsMatRef for ColRef<'_, T, Rows, Rs> {
	type Cols = One;
	type Owned = Col<T, Rows>;
	type Rows = Rows;
	type T = T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<'_, Self::T, Self::Rows, Self::Cols> {
		self.as_dyn_stride().as_mat().as_col_shape(One)
	}
}

impl<T, Rows: Shape, Rs: Stride> AsMatRef for ColMut<'_, T, Rows, Rs> {
	type Cols = One;
	type Owned = Col<T, Rows>;
	type Rows = Rows;
	type T = T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<'_, Self::T, Self::Rows, Self::Cols> {
		self.rb().as_dyn_stride().as_mat().as_col_shape(One)
	}
}

impl<T, Rows: Shape> AsMatRef for Col<T, Rows> {
	type Cols = One;
	type Owned = Col<T, Rows>;
	type Rows = Rows;
	type T = T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<'_, Self::T, Self::Rows, Self::Cols> {
		self.as_dyn_stride().as_mat().as_col_shape(One)
	}
}

impl<T, Rows: Shape, Rs: Stride> AsMatMut for ColMut<'_, T, Rows, Rs> {
	#[inline]
	fn as_mat_mut(&mut self) -> MatMut<'_, Self::T, Self::Rows, Self::Cols> {
		self.rb_mut().as_dyn_stride_mut().as_mat_mut().as_col_shape_mut(One)
	}
}

impl<T, Rows: Shape> AsMatMut for Col<T, Rows> {
	#[inline]
	fn as_mat_mut(&mut self) -> MatMut<'_, Self::T, Self::Rows, Self::Cols> {
		self.as_dyn_stride_mut().as_mat_mut().as_col_shape_mut(One)
	}
}

impl<T, Rows: Shape> AsMat<T> for Col<T, Rows> {
	#[inline]
	fn zeros(rows: Rows, _: One) -> Self
	where
		T: ComplexField,
	{
		Col::zeros(rows)
	}

	#[track_caller]
	#[inline]
	fn truncate(&mut self, rows: Self::Rows, _: One) {
		self.truncate(rows)
	}
}
