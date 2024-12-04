use crate::internal_prelude::*;
use crate::utils::bound::{One, Zero};
use core::ptr::NonNull;

pub trait ColIndex<RowRange> {
	type Target;

	fn get(this: Self, row: RowRange) -> Self::Target;
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

pub use colmut::ColMut;
pub use colown::Col;
pub use colref::ColRef;
use mat::AsMat;

pub trait AsColMut: AsColRef {
	fn as_col_mut(&mut self) -> ColMut<'_, Self::T, Self::Rows>;
}
pub trait AsColRef: AsMatRef<Cols = One> {
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
	fn as_mat_ref(&self) -> MatRef<Self::T, Self::Rows, Self::Cols> {
		self.as_dyn_stride().as_mat().as_col_shape(One)
	}
}

impl<T, Rows: Shape, Rs: Stride> AsMatRef for ColMut<'_, T, Rows, Rs> {
	type Cols = One;
	type Owned = Col<T, Rows>;
	type Rows = Rows;
	type T = T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<Self::T, Self::Rows, Self::Cols> {
		self.rb().as_dyn_stride().as_mat().as_col_shape(One)
	}
}

impl<T, Rows: Shape> AsMatRef for Col<T, Rows> {
	type Cols = One;
	type Owned = Col<T, Rows>;
	type Rows = Rows;
	type T = T;

	#[inline]
	fn as_mat_ref(&self) -> MatRef<Self::T, Self::Rows, Self::Cols> {
		self.as_dyn_stride().as_mat().as_col_shape(One)
	}
}

impl<T, Rows: Shape, Rs: Stride> AsMatMut for ColMut<'_, T, Rows, Rs> {
	#[inline]
	fn as_mat_mut(&mut self) -> MatMut<Self::T, Self::Rows, Self::Cols> {
		self.rb_mut().as_dyn_stride_mut().as_mat_mut().as_col_shape_mut(One)
	}
}

impl<T, Rows: Shape> AsMatMut for Col<T, Rows> {
	#[inline]
	fn as_mat_mut(&mut self) -> MatMut<Self::T, Self::Rows, Self::Cols> {
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
}
