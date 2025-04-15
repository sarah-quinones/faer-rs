use crate::Shape;
use crate::internal_prelude::*;
use crate::utils::bound::{One, Zero};

/// represents a type that can be used to slice a row, such as an index or a range of indices
pub trait RowIndex<ColRange> {
	/// sliced view type
	type Target;

	/// slice `this` using `col`
	fn get(this: Self, col: ColRange) -> Self::Target;
	/// slice `this` using `col` without bound checks
	unsafe fn get_unchecked(this: Self, col: ColRange) -> Self::Target;
}

mod row_index;

pub(crate) mod rowmut;
pub(crate) mod rowown;
pub(crate) mod rowref;

use mat::AsMat;
pub use rowmut::Mut;
pub use rowown::Own;
pub use rowref::Ref;

/// immutable view over a row vector, similar to an immutable reference to a strided
/// [prim@slice]
///
/// # note
///
/// unlike a slice, the data pointed to by `RowRef<'_, T>` is allowed to be partially or fully
/// uninitialized under certain conditions. in this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly or
/// indirectly through any of the numerical library routines, unless it is explicitly permitted
pub type RowRef<'a, T, Cols = usize, CStride = isize> = generic::Row<Ref<'a, T, Cols, CStride>>;

/// mutable view over a row vector, similar to a mutable reference to a strided
/// [prim@slice]
///
/// # note
///
/// unlike a slice, the data pointed to by `RowMut<'_, T>` is allowed to be partially or fully
/// uninitialized under certain conditions. in this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly or
/// indirectly through any of the numerical library routines, unless it is explicitly permitted
pub type RowMut<'a, T, Cols = usize, CStride = isize> = generic::Row<Mut<'a, T, Cols, CStride>>;

/// heap allocated resizable row vector.
///
/// # note
///
/// the memory layout of `Row` is guaranteed to be row-major, meaning that it has a column stride
/// of `1`.
pub type Row<T, Cols = usize> = generic::Row<Own<T, Cols>>;

/// generic `Row` wrapper
pub mod generic {
	use crate::{Idx, Shape, Stride};
	use core::fmt::Debug;
	use core::ops::{Index, IndexMut};
	use reborrow::*;

	/// generic `Row` wrapper
	#[derive(Copy, Clone)]
	#[repr(transparent)]
	pub struct Row<Inner>(pub Inner);

	impl<Inner: Debug> Debug for Row<Inner> {
		#[inline(always)]
		fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
			self.0.fmt(f)
		}
	}

	impl<Inner> Row<Inner> {
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

	impl<Inner> core::ops::Deref for Row<Inner> {
		type Target = Inner;

		#[inline(always)]
		fn deref(&self) -> &Self::Target {
			&self.0
		}
	}

	impl<Inner> core::ops::DerefMut for Row<Inner> {
		#[inline(always)]
		fn deref_mut(&mut self) -> &mut Self::Target {
			&mut self.0
		}
	}

	impl<'short, Inner: Reborrow<'short>> Reborrow<'short> for Row<Inner> {
		type Target = Row<Inner::Target>;

		#[inline(always)]
		fn rb(&'short self) -> Self::Target {
			Row(self.0.rb())
		}
	}

	impl<'short, Inner: ReborrowMut<'short>> ReborrowMut<'short> for Row<Inner> {
		type Target = Row<Inner::Target>;

		#[inline(always)]
		fn rb_mut(&'short mut self) -> Self::Target {
			Row(self.0.rb_mut())
		}
	}

	impl<Inner: IntoConst> IntoConst for Row<Inner> {
		type Target = Row<Inner::Target>;

		#[inline(always)]
		fn into_const(self) -> Self::Target {
			Row(self.0.into_const())
		}
	}

	impl<T, Cols: Shape, CStride: Stride, Inner: for<'short> Reborrow<'short, Target = super::Ref<'short, T, Cols, CStride>>> Index<Idx<Cols>>
		for Row<Inner>
	{
		type Output = T;

		#[inline]
		#[track_caller]
		fn index(&self, col: Idx<Cols>) -> &Self::Output {
			self.rb().at(col)
		}
	}

	impl<
		T,
		Cols: Shape,
		CStride: Stride,
		Inner: for<'short> Reborrow<'short, Target = super::Ref<'short, T, Cols, CStride>>
			+ for<'short> ReborrowMut<'short, Target = super::Mut<'short, T, Cols, CStride>>,
	> IndexMut<Idx<Cols>> for Row<Inner>
	{
		#[inline]
		#[track_caller]
		fn index_mut(&mut self, col: Idx<Cols>) -> &mut Self::Output {
			self.rb_mut().at_mut(col)
		}
	}
}

/// trait for types that can be converted to a row view
pub trait AsRowMut: AsRowRef {
	/// returns a view over `self`
	fn as_row_mut(&mut self) -> RowMut<'_, Self::T, Self::Cols>;
}
/// trait for types that can be converted to a row view
pub trait AsRowRef: AsMatRef<Rows = One> {
	/// returns a view over `self`
	fn as_row_ref(&self) -> RowRef<'_, Self::T, Self::Cols>;
}

impl<M: AsMatRef<Rows = One>> AsRowRef for M {
	#[inline]
	fn as_row_ref(&self) -> RowRef<'_, Self::T, Self::Cols> {
		self.as_mat_ref().row(Zero)
	}
}

impl<M: AsMatMut<Rows = One>> AsRowMut for M {
	#[inline]
	fn as_row_mut(&mut self) -> RowMut<'_, Self::T, Self::Cols> {
		self.as_mat_mut().row_mut(Zero)
	}
}

impl<T, Cols: Shape, Rs: Stride> AsMatRef for RowRef<'_, T, Cols, Rs> {
	type Cols = Cols;
	type Owned = Row<T, Cols>;
	type Rows = One;
	type T = T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<Self::T, One, Self::Cols> {
		self.as_dyn_stride().as_mat().as_row_shape(One)
	}
}

impl<T, Cols: Shape, Rs: Stride> AsMatRef for RowMut<'_, T, Cols, Rs> {
	type Cols = Cols;
	type Owned = Row<T, Cols>;
	type Rows = One;
	type T = T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<Self::T, One, Self::Cols> {
		self.rb().as_dyn_stride().as_mat().as_row_shape(One)
	}
}

impl<T, Cols: Shape> AsMatRef for Row<T, Cols> {
	type Cols = Cols;
	type Owned = Row<T, Cols>;
	type Rows = One;
	type T = T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<Self::T, One, Self::Cols> {
		self.as_dyn_stride().as_mat().as_row_shape(One)
	}
}

impl<T, Cols: Shape, Rs: Stride> AsMatMut for RowMut<'_, T, Cols, Rs> {
	#[inline]
	fn as_mat_mut(&mut self) -> MatMut<Self::T, One, Self::Cols> {
		self.rb_mut().as_dyn_stride_mut().as_mat_mut().as_row_shape_mut(One)
	}
}

impl<T, Cols: Shape> AsMatMut for Row<T, Cols> {
	#[inline]
	fn as_mat_mut(&mut self) -> MatMut<Self::T, One, Self::Cols> {
		self.as_dyn_stride_mut().as_mat_mut().as_row_shape_mut(One)
	}
}

impl<T, Cols: Shape> AsMat<T> for Row<T, Cols> {
	#[inline]
	fn zeros(_: One, cols: Cols) -> Self
	where
		T: ComplexField,
	{
		Row::zeros(cols)
	}

	#[track_caller]
	#[inline]
	fn truncate(&mut self, _: One, cols: Self::Cols) {
		self.truncate(cols)
	}
}
