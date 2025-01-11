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
pub use rowmut::RowMut;
pub use rowown::Row;
pub use rowref::RowRef;

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
}
