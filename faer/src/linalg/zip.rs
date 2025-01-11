#![allow(missing_docs)]

use std::marker::PhantomData;

use crate::col::{Col, ColMut, ColRef};
use crate::mat::{Mat, MatMut, MatRef};
use crate::row::{Row, RowMut, RowRef};
use crate::{ContiguousFwd, Idx, Shape, Stride, Unbind};
use equator::{assert, debug_assert};
use reborrow::*;

pub trait IntoView {
	type Target;

	fn into_view(self) -> Self::Target;
}

impl<'a, T, Rows: Shape, Cols: Shape> IntoView for &'a mut Mat<T, Rows, Cols> {
	type Target = MatMut<'a, T, Rows, Cols, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_mut().try_as_col_major_mut().unwrap()
	}
}
impl<'a, T, Rows: Shape, Cols: Shape> IntoView for &'a Mat<T, Rows, Cols> {
	type Target = MatRef<'a, T, Rows, Cols, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_ref().try_as_col_major().unwrap()
	}
}

impl<'a, T, Len: Shape> IntoView for &'a mut Col<T, Len> {
	type Target = ColMut<'a, T, Len, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_mut().try_as_col_major_mut().unwrap()
	}
}
impl<'a, T, Len: Shape> IntoView for &'a Col<T, Len> {
	type Target = ColRef<'a, T, Len, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_ref().try_as_col_major().unwrap()
	}
}

impl<'a, T, Len: Shape> IntoView for &'a mut Row<T, Len> {
	type Target = RowMut<'a, T, Len, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_mut().try_as_row_major_mut().unwrap()
	}
}
impl<'a, T, Len: Shape> IntoView for &'a Row<T, Len> {
	type Target = RowRef<'a, T, Len, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_ref().try_as_row_major().unwrap()
	}
}

impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView for MatMut<'a, T, Rows, Cols, RStride, CStride> {
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}
impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView for MatRef<'a, T, Rows, Cols, RStride, CStride> {
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}

impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView for &'a MatMut<'_, T, Rows, Cols, RStride, CStride> {
	type Target = MatRef<'a, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb()
	}
}
impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView for &'a MatRef<'_, T, Rows, Cols, RStride, CStride> {
	type Target = MatRef<'a, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}

impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView for &'a mut MatMut<'_, T, Rows, Cols, RStride, CStride> {
	type Target = MatMut<'a, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb_mut()
	}
}
impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView for &'a mut MatRef<'_, T, Rows, Cols, RStride, CStride> {
	type Target = MatRef<'a, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}

impl<'a, T, Rows: Shape, RStride: Stride> IntoView for ColMut<'a, T, Rows, RStride> {
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> IntoView for ColRef<'a, T, Rows, RStride> {
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}

impl<'a, T, Rows: Shape, RStride: Stride> IntoView for &'a ColMut<'_, T, Rows, RStride> {
	type Target = ColRef<'a, T, Rows, RStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb()
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> IntoView for &'a ColRef<'_, T, Rows, RStride> {
	type Target = ColRef<'a, T, Rows, RStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}

impl<'a, T, Rows: Shape, RStride: Stride> IntoView for &'a mut ColMut<'_, T, Rows, RStride> {
	type Target = ColMut<'a, T, Rows, RStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb_mut()
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> IntoView for &'a mut ColRef<'_, T, Rows, RStride> {
	type Target = ColRef<'a, T, Rows, RStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}

impl<'a, T, Cols: Shape, CStride: Stride> IntoView for RowMut<'a, T, Cols, CStride> {
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}
impl<'a, T, Cols: Shape, CStride: Stride> IntoView for RowRef<'a, T, Cols, CStride> {
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}

impl<'a, T, Cols: Shape, CStride: Stride> IntoView for &'a RowMut<'_, T, Cols, CStride> {
	type Target = RowRef<'a, T, Cols, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb()
	}
}
impl<'a, T, Cols: Shape, CStride: Stride> IntoView for &'a RowRef<'_, T, Cols, CStride> {
	type Target = RowRef<'a, T, Cols, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}

impl<'a, T, Cols: Shape, CStride: Stride> IntoView for &'a mut RowMut<'_, T, Cols, CStride> {
	type Target = RowMut<'a, T, Cols, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb_mut()
	}
}
impl<'a, T, Cols: Shape, CStride: Stride> IntoView for &'a mut RowRef<'_, T, Cols, CStride> {
	type Target = RowRef<'a, T, Cols, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Diag {
	Skip,
	Include,
}

/// matrix layout transformation. used for zipping optimizations
#[derive(Copy, Clone)]
pub enum MatLayoutTransform {
	/// matrix is used as-is
	None,
	/// matrix rows are reversed
	ReverseRows,
	/// matrix is transposed
	Transpose,
	/// matrix is transposed, then rows are reversed
	TransposeReverseRows,
}

/// vector layout transformation. used for zipping optimizations
#[derive(Copy, Clone)]
pub enum VecLayoutTransform {
	/// vector is used as-is
	None,
	/// vector is reversed
	Reverse,
}

/// type with a given matrix shape
pub trait MatIndex {
	/// type of rows
	type Rows: Copy + Eq + core::fmt::Debug;
	/// type of columns
	type Cols: Copy + Eq + core::fmt::Debug;
	/// returns the number of rows
	fn nrows(this: &Self) -> Self::Rows;
	/// returns the number of columns
	fn ncols(this: &Self) -> Self::Cols;

	/// indexing type
	type Index: Copy;
	/// layout transformation type
	type LayoutTransform: Copy;

	/// item produced by the zip views
	type Item;

	/// matrix type with type erased dimensions
	type Dyn: MatIndex<Dyn = Self::Dyn, LayoutTransform = Self::LayoutTransform, Item = Self::Item, Slice = Self::Slice>;

	type Slice: for<'a> SliceFamily<'a, Self::Item>;

	/// returns slice at index of length `n_elems`
	unsafe fn get_slice_unchecked<'a>(this: &'a mut Self, idx: Self::Index, n_elems: usize) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice;

	/// converts a type erased index back to its original representation
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index;

	/// get the item at the given index, skipping bound checks
	unsafe fn get_unchecked(this: &mut Self, index: Self::Index) -> Self::Item;
	/// get the item at the given slice position, skipping bound checks
	unsafe fn next_unchecked<'a>(slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice) -> Self::Item;

	/// checks if the zip matrices are contiguous
	fn is_contiguous(this: &Self) -> bool;
	/// computes the preferred iteration layout of the matrices
	fn preferred_layout(this: &Self) -> Self::LayoutTransform;
	/// applies the layout transformation to the matrices
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn;
}

pub trait SliceFamily<'a, T, Outlives = &'a Self> {
	type Slice;
}
pub struct Slice<T>(pub T);
pub struct SliceRef<'b, T>(pub &'b T);
pub struct SliceMut<'b, T>(pub &'b mut T);

impl<'a, T> SliceFamily<'a, T> for Slice<T> {
	type Slice = &'a [T];
}
impl<'a, 'b, T> SliceFamily<'a, &'b T> for SliceRef<'b, T> {
	type Slice = &'b [T];
}
impl<'a, 'b, T> SliceFamily<'a, &'b mut T> for SliceMut<'b, T> {
	type Slice = &'b mut [T];
}
impl<'a, T, F: SliceFamily<'a, T>> SliceFamily<'a, Last<T>> for Last<F> {
	type Slice = Last<F::Slice>;
}
impl<'a, T, U, F: SliceFamily<'a, T>, G: SliceFamily<'a, U>> SliceFamily<'a, Zip<T, U>> for Zip<F, G> {
	type Slice = Zip<F::Slice, G::Slice>;
}

/// single matrix
#[derive(Copy, Clone, Debug)]
pub struct LastEq<Rows, Cols, Mat>(pub Mat, pub PhantomData<(Rows, Cols)>);

/// single element
#[derive(Copy, Clone, Debug)]
pub struct Last<Mat>(pub Mat);

/// zipped matrices
#[derive(Copy, Clone, Debug)]
pub struct ZipEq<Rows, Cols, Head, Tail>(pub Head, pub Tail, PhantomData<(Rows, Cols)>);

/// zipped elements
#[derive(Copy, Clone, Debug)]
pub struct Zip<Head, Tail>(pub Head, pub Tail);

/// single matrix view
impl<
	Rows: Copy + Eq + core::fmt::Debug,
	Cols: Copy + Eq + core::fmt::Debug,
	Head: MatIndex<Rows = Rows, Cols = Cols>,
	Tail: MatIndex<Rows = Rows, Cols = Cols>,
> ZipEq<Rows, Cols, Head, Tail>
{
	/// creates a zip matrix, after asserting that the dimensions match
	#[inline(always)]
	#[track_caller]
	pub fn new(head: Head, tail: Tail) -> Self {
		assert!(all(Head::nrows(&head) == Tail::nrows(&tail), Head::ncols(&head) == Tail::ncols(&tail),));
		Self(head, tail, PhantomData)
	}

	/// creates a zip matrix, assuming that the dimensions match
	#[inline(always)]
	#[track_caller]
	pub fn new_unchecked(head: Head, tail: Tail) -> Self {
		debug_assert!(all(Head::nrows(&head) == Tail::nrows(&tail), Head::ncols(&head) == Tail::ncols(&tail),));
		Self(head, tail, PhantomData)
	}
}

impl<Rows: Copy + Eq + core::fmt::Debug, Cols: Copy + Eq + core::fmt::Debug, Mat: MatIndex<Rows = Rows, Cols = Cols>> MatIndex
	for LastEq<Rows, Cols, Mat>
{
	type Cols = Mat::Cols;
	type Dyn = LastEq<<Mat::Dyn as MatIndex>::Rows, <Mat::Dyn as MatIndex>::Cols, Mat::Dyn>;
	type Index = Mat::Index;
	type Item = Last<Mat::Item>;
	type LayoutTransform = Mat::LayoutTransform;
	type Rows = Mat::Rows;
	type Slice = Last<Mat::Slice>;

	#[inline(always)]
	fn nrows(this: &Self) -> Self::Rows {
		Mat::nrows(&this.0)
	}

	#[inline(always)]
	fn ncols(this: &Self) -> Self::Cols {
		Mat::ncols(&this.0)
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(this: &'a mut Self, idx: Self::Index, n_elems: usize) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		Last(Mat::get_slice_unchecked(&mut this.0, idx, n_elems))
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		Mat::from_dyn_idx(idx)
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, index: Self::Index) -> Self::Item {
		Last(Mat::get_unchecked(&mut this.0, index))
	}

	#[inline]
	unsafe fn next_unchecked<'a>(slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice) -> Self::Item {
		Last(Mat::next_unchecked(&mut slice.0))
	}

	#[inline]
	fn is_contiguous(this: &Self) -> bool {
		Mat::is_contiguous(&this.0)
	}

	#[inline]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		Mat::preferred_layout(&this.0)
	}

	#[inline]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		LastEq(Mat::with_layout(this.0, layout), PhantomData)
	}
}

impl<
	Rows: Copy + Eq + core::fmt::Debug,
	Cols: Copy + Eq + core::fmt::Debug,
	L: MatIndex<Rows = Rows, Cols = Cols>,
	R: MatIndex<Rows = Rows, Cols = Cols, Index = L::Index, LayoutTransform = L::LayoutTransform>,
> MatIndex for ZipEq<Rows, Cols, L, R>
where
	R::Dyn: MatIndex<Rows = <L::Dyn as MatIndex>::Rows, Cols = <L::Dyn as MatIndex>::Cols, Index = <L::Dyn as MatIndex>::Index>,
{
	type Cols = L::Cols;
	type Dyn = ZipEq<<L::Dyn as MatIndex>::Rows, <L::Dyn as MatIndex>::Cols, L::Dyn, R::Dyn>;
	type Index = L::Index;
	type Item = Zip<L::Item, R::Item>;
	type LayoutTransform = L::LayoutTransform;
	type Rows = L::Rows;
	type Slice = Zip<L::Slice, R::Slice>;

	#[inline(always)]
	fn nrows(this: &Self) -> Self::Rows {
		L::nrows(&this.0)
	}

	#[inline(always)]
	fn ncols(this: &Self) -> Self::Cols {
		L::ncols(&this.0)
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(this: &'a mut Self, idx: Self::Index, n_elems: usize) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		Zip(
			L::get_slice_unchecked(&mut this.0, idx, n_elems),
			R::get_slice_unchecked(&mut this.1, idx, n_elems),
		)
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		L::from_dyn_idx(idx)
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, index: Self::Index) -> Self::Item {
		Zip(L::get_unchecked(&mut this.0, index), R::get_unchecked(&mut this.1, index))
	}

	#[inline]
	unsafe fn next_unchecked<'a>(slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice) -> Self::Item {
		Zip(L::next_unchecked(&mut slice.0), R::next_unchecked(&mut slice.1))
	}

	#[inline]
	fn is_contiguous(this: &Self) -> bool {
		L::is_contiguous(&this.0) && R::is_contiguous(&this.1)
	}

	#[inline]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		L::preferred_layout(&this.0)
	}

	#[inline]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		ZipEq(L::with_layout(this.0, layout), R::with_layout(this.1, layout), PhantomData)
	}
}

impl<'b, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> MatIndex for MatMut<'b, T, Rows, Cols, RStride, CStride> {
	type Cols = Cols;
	type Dyn = MatMut<'b, T, usize, usize, isize, isize>;
	type Index = (Idx<Rows>, Idx<Cols>);
	type Item = &'b mut T;
	type LayoutTransform = MatLayoutTransform;
	type Rows = Rows;
	type Slice = SliceMut<'b, T>;

	#[inline]
	fn nrows(this: &Self) -> Self::Rows {
		this.nrows()
	}

	#[inline]
	fn ncols(this: &Self) -> Self::Cols {
		this.ncols()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(this: &'a mut Self, idx: Self::Index, n_elems: usize) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		let ptr = this.ptr_inbounds_at_mut(idx.0, idx.1);
		core::slice::from_raw_parts_mut(ptr, n_elems)
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		(Idx::<Rows>::new_unbound(idx.0), Idx::<Cols>::new_unbound(idx.1))
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, (i, j): Self::Index) -> Self::Item {
		let ptr = this.rb().ptr_inbounds_at_mut(i, j);
		&mut *ptr
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice) -> Self::Item {
		let (head, tail) = core::mem::take(slice).split_first_mut().unwrap_unchecked();
		*slice = tail;
		head
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.row_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let rs = this.row_stride().element_stride();
		let cs = this.col_stride().element_stride();
		let nrows = this.nrows().unbound();
		let ncols = this.ncols().unbound();

		if nrows > 1 && rs == 1 {
			MatLayoutTransform::None
		} else if nrows > 1 && rs == -1 {
			MatLayoutTransform::ReverseRows
		} else if ncols > 1 && cs == 1 {
			MatLayoutTransform::Transpose
		} else if ncols > 1 && cs == -1 {
			MatLayoutTransform::TransposeReverseRows
		} else {
			MatLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use MatLayoutTransform::*;
		let this = this.as_dyn_mut().as_dyn_stride_mut();
		match layout {
			None => this,
			ReverseRows => this.reverse_rows_mut(),
			Transpose => this.transpose_mut(),
			TransposeReverseRows => this.transpose_mut().reverse_rows_mut(),
		}
	}
}

impl<'b, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> MatIndex for MatRef<'b, T, Rows, Cols, RStride, CStride> {
	type Cols = Cols;
	type Dyn = MatRef<'b, T, usize, usize, isize, isize>;
	type Index = (Idx<Rows>, Idx<Cols>);
	type Item = &'b T;
	type LayoutTransform = MatLayoutTransform;
	type Rows = Rows;
	type Slice = SliceRef<'b, T>;

	#[inline]
	fn nrows(this: &Self) -> Self::Rows {
		this.nrows()
	}

	#[inline]
	fn ncols(this: &Self) -> Self::Cols {
		this.ncols()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(this: &'a mut Self, idx: Self::Index, n_elems: usize) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		let ptr = this.ptr_inbounds_at(idx.0, idx.1);
		core::slice::from_raw_parts(ptr, n_elems)
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		(Idx::<Rows>::new_unbound(idx.0), Idx::<Cols>::new_unbound(idx.1))
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, (i, j): Self::Index) -> Self::Item {
		let ptr = this.rb().ptr_inbounds_at(i, j);
		&*ptr
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice) -> Self::Item {
		let (head, tail) = core::mem::take(slice).split_first().unwrap_unchecked();
		*slice = tail;
		head
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.row_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let rs = this.row_stride().element_stride();
		let cs = this.col_stride().element_stride();
		let nrows = this.nrows().unbound();
		let ncols = this.ncols().unbound();

		if nrows > 1 && rs == 1 {
			MatLayoutTransform::None
		} else if nrows > 1 && rs == -1 {
			MatLayoutTransform::ReverseRows
		} else if ncols > 1 && cs == 1 {
			MatLayoutTransform::Transpose
		} else if ncols > 1 && cs == -1 {
			MatLayoutTransform::TransposeReverseRows
		} else {
			MatLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use MatLayoutTransform::*;
		let this = this.as_dyn().as_dyn_stride();
		match layout {
			None => this,
			ReverseRows => this.reverse_rows(),
			Transpose => this.transpose(),
			TransposeReverseRows => this.transpose().reverse_rows(),
		}
	}
}

impl<'b, T, Len: Shape, Strd: Stride> MatIndex for ColMut<'b, T, Len, Strd> {
	type Cols = ();
	type Dyn = ColMut<'b, T, usize, isize>;
	type Index = Idx<Len>;
	type Item = &'b mut T;
	type LayoutTransform = VecLayoutTransform;
	type Rows = Len;
	type Slice = SliceMut<'b, T>;

	#[inline]
	fn nrows(this: &Self) -> Self::Rows {
		this.nrows()
	}

	#[inline]
	fn ncols(_: &Self) -> Self::Cols {
		()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(this: &'a mut Self, idx: Self::Index, n_elems: usize) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		let ptr = this.ptr_inbounds_at_mut(idx);
		core::slice::from_raw_parts_mut(ptr, n_elems)
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		Idx::<Len>::new_unbound(idx)
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, i: Self::Index) -> Self::Item {
		let ptr = this.rb().ptr_inbounds_at_mut(i);
		&mut *ptr
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice) -> Self::Item {
		let (head, tail) = core::mem::take(slice).split_first_mut().unwrap_unchecked();
		*slice = tail;
		head
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.row_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let strd = this.row_stride().element_stride();
		let len = this.nrows().unbound();

		if len > 1 && strd == 1 {
			VecLayoutTransform::None
		} else if len > 1 && strd == -1 {
			VecLayoutTransform::Reverse
		} else {
			VecLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use VecLayoutTransform::*;
		let this = this.as_dyn_rows_mut().as_dyn_stride_mut();
		match layout {
			None => this,
			Reverse => this.reverse_rows_mut(),
		}
	}
}

impl<'b, T, Len: Shape, Strd: Stride> MatIndex for RowMut<'b, T, Len, Strd> {
	type Cols = Len;
	type Dyn = RowMut<'b, T, usize, isize>;
	type Index = Idx<Len>;
	type Item = &'b mut T;
	type LayoutTransform = VecLayoutTransform;
	type Rows = ();
	type Slice = SliceMut<'b, T>;

	#[inline]
	fn nrows(_: &Self) -> Self::Rows {
		()
	}

	#[inline]
	fn ncols(this: &Self) -> Self::Cols {
		this.ncols()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(this: &'a mut Self, idx: Self::Index, n_elems: usize) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		let ptr = this.ptr_inbounds_at_mut(idx);
		core::slice::from_raw_parts_mut(ptr, n_elems)
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		Idx::<Len>::new_unbound(idx)
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, i: Self::Index) -> Self::Item {
		let ptr = this.rb().ptr_inbounds_at_mut(i);
		&mut *ptr
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice) -> Self::Item {
		let (head, tail) = core::mem::take(slice).split_first_mut().unwrap_unchecked();
		*slice = tail;
		head
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.col_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let strd = this.col_stride().element_stride();
		let len = this.ncols().unbound();

		if len > 1 && strd == 1 {
			VecLayoutTransform::None
		} else if len > 1 && strd == -1 {
			VecLayoutTransform::Reverse
		} else {
			VecLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use VecLayoutTransform::*;
		let this = this.as_dyn_cols_mut().as_dyn_stride_mut();
		match layout {
			None => this,
			Reverse => this.reverse_cols_mut(),
		}
	}
}

impl<'b, T, Len: Shape, Strd: Stride> MatIndex for ColRef<'b, T, Len, Strd> {
	type Cols = ();
	type Dyn = ColRef<'b, T, usize, isize>;
	type Index = Idx<Len>;
	type Item = &'b T;
	type LayoutTransform = VecLayoutTransform;
	type Rows = Len;
	type Slice = SliceRef<'b, T>;

	#[inline]
	fn nrows(this: &Self) -> Self::Rows {
		this.nrows()
	}

	#[inline]
	fn ncols(_: &Self) -> Self::Cols {
		()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(this: &'a mut Self, idx: Self::Index, n_elems: usize) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		let ptr = this.ptr_inbounds_at(idx);
		core::slice::from_raw_parts(ptr, n_elems)
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		Idx::<Len>::new_unbound(idx)
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, i: Self::Index) -> Self::Item {
		let ptr = this.rb().ptr_inbounds_at(i);
		&*ptr
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice) -> Self::Item {
		let (head, tail) = core::mem::take(slice).split_first().unwrap_unchecked();
		*slice = tail;
		head
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.row_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let strd = this.row_stride().element_stride();
		let len = this.nrows().unbound();

		if len > 1 && strd == 1 {
			VecLayoutTransform::None
		} else if len > 1 && strd == -1 {
			VecLayoutTransform::Reverse
		} else {
			VecLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use VecLayoutTransform::*;
		let this = this.as_dyn_rows().as_dyn_stride();
		match layout {
			None => this,
			Reverse => this.reverse_rows(),
		}
	}
}

impl<'b, T, Len: Shape, Strd: Stride> MatIndex for RowRef<'b, T, Len, Strd> {
	type Cols = Len;
	type Dyn = RowRef<'b, T, usize, isize>;
	type Index = Idx<Len>;
	type Item = &'b T;
	type LayoutTransform = VecLayoutTransform;
	type Rows = ();
	type Slice = SliceRef<'b, T>;

	#[inline]
	fn nrows(_: &Self) -> Self::Rows {
		()
	}

	#[inline]
	fn ncols(this: &Self) -> Self::Cols {
		this.ncols()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(this: &'a mut Self, idx: Self::Index, n_elems: usize) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		let ptr = this.ptr_inbounds_at(idx);
		core::slice::from_raw_parts(ptr, n_elems)
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		Idx::<Len>::new_unbound(idx)
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, i: Self::Index) -> Self::Item {
		let ptr = this.rb().ptr_inbounds_at(i);
		&*ptr
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice) -> Self::Item {
		let (head, tail) = core::mem::take(slice).split_first().unwrap_unchecked();
		*slice = tail;
		head
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.col_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let strd = this.col_stride().element_stride();
		let len = this.ncols().unbound();

		if len > 1 && strd == 1 {
			VecLayoutTransform::None
		} else if len > 1 && strd == -1 {
			VecLayoutTransform::Reverse
		} else {
			VecLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use VecLayoutTransform::*;
		let this = this.as_dyn_cols().as_dyn_stride();
		match layout {
			None => this,
			Reverse => this.reverse_cols(),
		}
	}
}

#[inline(always)]
fn annotate_noalias_mat<Z: MatIndex>(
	f: &mut impl FnMut(<Z as MatIndex>::Item),
	mut slice: <Z::Slice as SliceFamily<'_, Z::Item>>::Slice,
	i_begin: usize,
	i_end: usize,
	_j: usize,
) {
	for _ in i_begin..i_end {
		unsafe { f(Z::next_unchecked(&mut slice)) };
	}
}

#[inline(always)]
fn annotate_noalias_mat_with_index<Z: MatIndex<Index = (RowIdx, ColIdx)>, RowIdx, ColIdx>(
	f: &mut impl FnMut(RowIdx, ColIdx, <Z as MatIndex>::Item),
	mut slice: <Z::Slice as SliceFamily<'_, Z::Item>>::Slice,
	i_begin: usize,
	i_end: usize,
	j: usize,
	transpose: bool,
	reverse_rows: bool,
) where
	Z::Dyn: MatIndex<Index = (usize, usize)>,
{
	if !transpose {
		if !reverse_rows {
			for i in i_begin..i_end {
				unsafe {
					let (ii, jj) = Z::from_dyn_idx((i, j));
					f(ii, jj, Z::next_unchecked(&mut slice))
				};
			}
		} else {
			for i in i_begin..i_end {
				unsafe {
					let (ii, jj) = Z::from_dyn_idx((i_begin + (i_end - i - 1), j));
					f(ii, jj, Z::next_unchecked(&mut slice))
				};
			}
		}
	} else {
		if !reverse_rows {
			for i in i_begin..i_end {
				unsafe {
					let (ii, jj) = Z::from_dyn_idx((j, i));
					f(ii, jj, Z::next_unchecked(&mut slice))
				};
			}
		} else {
			for i in i_begin..i_end {
				unsafe {
					let (ii, jj) = Z::from_dyn_idx((j, i_begin + (i_end - i - 1)));
					f(ii, jj, Z::next_unchecked(&mut slice))
				};
			}
		}
	}
}

#[inline(always)]
fn annotate_noalias_col<Z: MatIndex>(
	f: &mut impl FnMut(<Z as MatIndex>::Item),
	mut slice: <Z::Slice as SliceFamily<'_, Z::Item>>::Slice,
	i_begin: usize,
	i_end: usize,
) {
	for _ in i_begin..i_end {
		unsafe { f(Z::next_unchecked(&mut slice)) };
	}
}

#[inline(always)]
fn annotate_noalias_col_with_index<Z: MatIndex<Index = Idx>, Idx>(
	f: &mut impl FnMut(Idx, <Z as MatIndex>::Item),
	mut slice: <Z::Slice as SliceFamily<'_, Z::Item>>::Slice,
	i_begin: usize,
	i_end: usize,
	reverse: bool,
) where
	Z::Dyn: MatIndex<Item = Z::Item, Index = usize>,
{
	if !reverse {
		for i in i_begin..i_end {
			unsafe {
				let ii = Z::from_dyn_idx(i);
				f(ii, Z::next_unchecked(&mut slice))
			};
		}
	} else {
		for i in i_begin..i_end {
			unsafe {
				let ii = Z::from_dyn_idx(i_begin + (i_end - i - 1));
				f(ii, Z::next_unchecked(&mut slice))
			};
		}
	}
}

#[inline(always)]
fn for_each_mat<Z: MatIndex>(z: Z, mut f: impl FnMut(<Z as MatIndex>::Item))
where
	Z::Dyn: MatIndex<Item = Z::Item, Slice = Z::Slice, Rows = usize, Cols = usize, Index = (usize, usize)>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);

	let m = Z::Dyn::nrows(&z);
	let n = Z::Dyn::ncols(&z);
	if m == 0 || n == 0 {
		return;
	}

	unsafe {
		if Z::Dyn::is_contiguous(&z) {
			for j in 0..n {
				annotate_noalias_mat::<Z::Dyn>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, (0, j), m), 0, m, j);
			}
		} else {
			for j in 0..n {
				for i in 0..m {
					f(Z::Dyn::get_unchecked(&mut z, (i, j)))
				}
			}
		}
	}
}

// TODO:
// - for_each_vec_with_index

#[inline(always)]
fn for_each_mat_with_index<RowIdx, ColIdx, Z: MatIndex<Index = (RowIdx, ColIdx), LayoutTransform = MatLayoutTransform>>(
	z: Z,
	mut f: impl FnMut(RowIdx, ColIdx, <Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<Rows = usize, Cols = usize, Index = (usize, usize), Slice = Z::Slice, Item = Z::Item>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);

	let m = Z::Dyn::nrows(&z);
	let n = Z::Dyn::ncols(&z);
	if m == 0 || n == 0 {
		return;
	}

	match layout {
		MatLayoutTransform::None => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					annotate_noalias_mat_with_index::<Z, _, _>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, (0, j), m), 0, m, j, false, false);
				}
			} else {
				for j in 0..n {
					for i in 0..m {
						let (ii, jj) = Z::from_dyn_idx((i, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		MatLayoutTransform::ReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					annotate_noalias_mat_with_index::<Z, _, _>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, (0, j), m), 0, m, j, false, true);
				}
			} else {
				for j in 0..n {
					for i in 0..m {
						let (ii, jj) = Z::from_dyn_idx((m - i - 1, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		MatLayoutTransform::Transpose => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					annotate_noalias_mat_with_index::<Z, _, _>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, (0, j), m), 0, m, j, true, false);
				}
			} else {
				for j in 0..n {
					for i in 0..m {
						let (ii, jj) = Z::from_dyn_idx((j, i));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		MatLayoutTransform::TransposeReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					annotate_noalias_mat_with_index::<Z, _, _>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, (0, j), m), 0, m, j, true, true);
				}
			} else {
				for j in 0..n {
					for i in 0..m {
						let (ii, jj) = Z::from_dyn_idx((j, m - i - 1));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
	}
}

#[inline(always)]
fn for_each_mat_triangular_lower_with_index<RowIdx, ColIdx, Z: MatIndex<Index = (RowIdx, ColIdx), LayoutTransform = MatLayoutTransform>>(
	z: Z,
	diag: Diag,
	mut f: impl FnMut(RowIdx, ColIdx, <Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<Rows = usize, Cols = usize, Index = (usize, usize), Item = Z::Item, Slice = Z::Slice>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);

	let m = Z::Dyn::nrows(&z);
	let n = Z::Dyn::ncols(&z);
	let strict = match diag {
		Diag::Skip => true,
		Diag::Include => false,
	};
	let strict = strict as usize;

	if m == 0 || n == 0 {
		return;
	}

	match layout {
		MatLayoutTransform::None => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = j + strict;
					let end = m;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (start, j), end - start),
						start,
						end,
						j,
						false,
						false,
					);
				}
			} else {
				for j in 0..n {
					let start = j + strict;
					let end = m;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((i, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		MatLayoutTransform::ReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..Ord::min(m, n) {
					let start = 0;
					let end = m - j - strict;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (start, j), end - start),
						j + strict + start,
						j + strict + end,
						j,
						false,
						true,
					);
				}
			} else {
				for j in 0..Ord::min(m, n) {
					let start = 0;
					let end = m - j - strict;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((m - i - 1, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		MatLayoutTransform::Transpose => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = 0;
					let end = Ord::min(m, j + (1 - strict));
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (0, j), end - start),
						start,
						end,
						j,
						true,
						false,
					);
				}
			} else {
				for j in 0..n {
					let start = 0;
					let end = Ord::min(m, j + (1 - strict));
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((j, i));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		MatLayoutTransform::TransposeReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = m - Ord::min(j + (1 - strict) as usize, m);
					let end = m;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (start, j), end - start),
						0,
						end - start,
						j,
						true,
						true,
					);
				}
			} else {
				for j in 0..n {
					let start = m - Ord::min(j + (1 - strict) as usize, m);
					let end = m;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((j, m - i - 1));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
	}
}

#[inline(always)]
fn for_each_mat_triangular_upper_with_index<RowIdx, ColIdx, Z: MatIndex<Index = (RowIdx, ColIdx), LayoutTransform = MatLayoutTransform>>(
	z: Z,
	diag: Diag,
	mut f: impl FnMut(RowIdx, ColIdx, <Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<Rows = usize, Cols = usize, Index = (usize, usize), Item = Z::Item, Slice = Z::Slice>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);

	let m = Z::Dyn::nrows(&z);
	let n = Z::Dyn::ncols(&z);
	let strict = match diag {
		Diag::Skip => true,
		Diag::Include => false,
	};
	let strict = strict as usize;

	if m == 0 || n == 0 {
		return;
	}

	match layout {
		MatLayoutTransform::None => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = 0;
					let end = Ord::min(m, j + (1 - strict));
					if start == end {
						continue;
					}

					annotate_noalias_mat_with_index::<Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (start, j), end - start),
						start,
						end,
						j,
						false,
						false,
					);
				}
			} else {
				for j in 0..n {
					let start = 0;
					let end = Ord::min(m, j + (1 - strict));
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((i, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		MatLayoutTransform::ReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..Ord::min(m, n) {
					let start = m - Ord::min(j + (1 - strict) as usize, m);
					let end = m;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (start, j), end - start),
						0,
						end - start,
						j,
						false,
						true,
					);
				}
			} else {
				for j in 0..Ord::min(m, n) {
					let start = m - Ord::min(j + (1 - strict) as usize, m);
					let end = m;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((m - i - 1, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		MatLayoutTransform::Transpose => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = j + strict;
					let end = m;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (start, j), end - start),
						start,
						end,
						j,
						true,
						false,
					);
				}
			} else {
				for j in 0..n {
					let start = j + strict;
					let end = m;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((j, i));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		MatLayoutTransform::TransposeReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = 0;
					let end = m - j - strict;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (start, j), end - start),
						j + strict,
						j + strict + end - start,
						j,
						true,
						true,
					);
				}
			} else {
				for j in 0..n {
					let start = 0;
					let end = m - j - strict;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((j, m - i - 1));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
	}
}

#[inline(always)]
fn for_each_mat_triangular_lower<Z: MatIndex<LayoutTransform = MatLayoutTransform>>(
	z: Z,
	diag: Diag,
	transpose: bool,
	mut f: impl FnMut(<Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<
			LayoutTransform = MatLayoutTransform,
			Item = Z::Item,
			Slice = Z::Slice,
			Rows = usize,
			Cols = usize,
			Index = (usize, usize),
			Dyn = Z::Dyn,
		>,
{
	use MatLayoutTransform::*;

	let z = if transpose {
		Z::with_layout(z, MatLayoutTransform::Transpose)
	} else {
		Z::with_layout(z, MatLayoutTransform::None)
	};
	let layout = Z::Dyn::preferred_layout(&z);
	let mut z = Z::Dyn::with_layout(z, layout);

	let m = Z::Dyn::nrows(&z);
	let n = Z::Dyn::ncols(&z);
	let n = match layout {
		None | ReverseRows => Ord::min(m, n),
		Transpose | TransposeReverseRows => n,
	};
	if m == 0 || n == 0 {
		return;
	}

	let strict = match diag {
		Diag::Skip => true,
		Diag::Include => false,
	};

	unsafe {
		if Z::Dyn::is_contiguous(&z) {
			for j in 0..n {
				let (start, end) = match layout {
					None => (j + strict as usize, m),
					ReverseRows => (0, (m - (j + strict as usize))),
					Transpose => (0, (j + !strict as usize).min(m)),
					TransposeReverseRows => (m - ((j + !strict as usize).min(m)), m),
				};

				let len = end - start;
				if start == end {
					continue;
				}

				annotate_noalias_mat::<Z::Dyn>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, (start, j), len), start, end, j);
			}
		} else {
			for j in 0..n {
				let (start, end) = match layout {
					None => (j + strict as usize, m),
					ReverseRows => (0, (m - (j + strict as usize))),
					Transpose => (0, (j + !strict as usize).min(m)),
					TransposeReverseRows => (m - ((j + !strict as usize).min(m)), m),
				};
				if start == end {
					continue;
				}

				for i in start..end {
					f(Z::Dyn::get_unchecked(&mut z, (i, j)))
				}
			}
		}
	}
}

#[inline(always)]
fn for_each_col<Z: MatIndex>(z: Z, mut f: impl FnMut(<Z as MatIndex>::Item))
where
	Z::Dyn: MatIndex<Rows = usize, Cols = (), Index = usize, Item = Z::Item, Slice = Z::Slice>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);

	let m = Z::Dyn::nrows(&z);
	if m == 0 {
		return;
	}

	unsafe {
		if Z::Dyn::is_contiguous(&z) {
			annotate_noalias_col::<Z::Dyn>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, 0, m), 0, m);
		} else {
			for i in 0..m {
				f(Z::Dyn::get_unchecked(&mut z, i))
			}
		}
	}
}

#[inline(always)]
fn for_each_col_with_index<Idx, Z: MatIndex<LayoutTransform = VecLayoutTransform, Index = Idx>>(z: Z, mut f: impl FnMut(Idx, <Z as MatIndex>::Item))
where
	Z::Dyn: MatIndex<Rows = usize, Cols = (), Index = usize, Item = Z::Item, Slice = Z::Slice>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);

	let m = Z::Dyn::nrows(&z);
	if m == 0 {
		return;
	}

	unsafe {
		match layout {
			VecLayoutTransform::None => {
				if Z::Dyn::is_contiguous(&z) {
					annotate_noalias_col_with_index::<Z, _>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, 0, m), 0, m, false);
				} else {
					for i in 0..m {
						f(Z::from_dyn_idx(i), Z::Dyn::get_unchecked(&mut z, i))
					}
				}
			},
			VecLayoutTransform::Reverse => {
				if Z::Dyn::is_contiguous(&z) {
					annotate_noalias_col_with_index::<Z, _>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, 0, m), 0, m, true);
				} else {
					for i in 0..m {
						f(Z::from_dyn_idx(m - i - 1), Z::Dyn::get_unchecked(&mut z, i))
					}
				}
			},
		}
	}
}

#[inline(always)]
fn for_each_row_with_index<Idx, Z: MatIndex<LayoutTransform = VecLayoutTransform, Index = Idx>>(z: Z, mut f: impl FnMut(Idx, <Z as MatIndex>::Item))
where
	Z::Dyn: MatIndex<Rows = (), Cols = usize, Index = usize, Item = Z::Item, Slice = Z::Slice>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);

	let n = Z::Dyn::ncols(&z);
	if n == 0 {
		return;
	}

	unsafe {
		match layout {
			VecLayoutTransform::None => {
				if Z::Dyn::is_contiguous(&z) {
					annotate_noalias_col_with_index::<Z, _>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, 0, n), 0, n, false);
				} else {
					for i in 0..n {
						f(Z::from_dyn_idx(i), Z::Dyn::get_unchecked(&mut z, i))
					}
				}
			},
			VecLayoutTransform::Reverse => {
				if Z::Dyn::is_contiguous(&z) {
					annotate_noalias_col_with_index::<Z, _>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, 0, n), 0, n, true);
				} else {
					for i in 0..n {
						f(Z::from_dyn_idx(n - i - 1), Z::Dyn::get_unchecked(&mut z, i))
					}
				}
			},
		}
	}
}
#[inline(always)]
fn for_each_row<Z: MatIndex>(z: Z, mut f: impl FnMut(<Z as MatIndex>::Item))
where
	Z::Dyn: MatIndex<Rows = (), Cols = usize, Index = usize, Item = Z::Item, Slice = Z::Slice>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);

	let n = Z::Dyn::ncols(&z);
	if n == 0 {
		return;
	}

	unsafe {
		if Z::Dyn::is_contiguous(&z) {
			annotate_noalias_col::<Z::Dyn>(&mut f, Z::Dyn::get_slice_unchecked(&mut z, 0, n), 0, n);
		} else {
			for j in 0..n {
				f(Z::Dyn::get_unchecked(&mut z, j))
			}
		}
	}
}

impl<Rows: Shape, Cols: Shape, M: MatIndex<LayoutTransform = MatLayoutTransform, Rows = Rows, Cols = Cols, Index = (Idx<Rows>, Idx<Cols>)>>
	LastEq<Rows, Cols, M>
where
	M::Dyn: MatIndex<Rows = usize, Cols = usize, Index = (usize, usize)>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_mat(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the position of the
	/// current element
	#[inline(always)]
	pub fn for_each_with_index(self, f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item)) {
		for_each_mat_with_index(self, f);
	}

	/// applies `f` to each element of the lower triangular half of `self`, while passing the
	/// indices of the position of the current element
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_lower_with_index(self, diag: Diag, f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item)) {
		for_each_mat_triangular_lower_with_index(self, diag, f);
	}

	/// applies `f` to each element of the upper triangular half of `self`, while passing the
	/// indices of the position of the current element
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_upper_with_index(self, diag: Diag, f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item)) {
		for_each_mat_triangular_upper_with_index(self, diag, f);
	}

	/// applies `f` to each element of the lower triangular half of `self`
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_lower(self, diag: Diag, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_mat_triangular_lower(self, diag, false, f);
	}

	/// applies `f` to each element of the upper triangular half of `self`
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_upper(self, diag: Diag, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_mat_triangular_lower(self, diag, true, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map<T>(self, f: impl FnMut(<Self as MatIndex>::Item) -> T) -> Mat<T, Rows, Cols> {
		let (m, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Mat::from_fn(
			m,
			n,
			#[inline(always)]
			|i, j| f(unsafe { Self::get_unchecked(&mut this, (i, j)) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map_with_index<T>(self, f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item) -> T) -> Mat<T, Rows, Cols> {
		let (m, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;

		Mat::from_fn(
			m,
			n,
			#[inline(always)]
			|i, j| f(i, j, unsafe { Self::get_unchecked(&mut this, (i, j)) }),
		)
	}
}

impl<
	Rows: Shape,
	Cols: Shape,
	L: MatIndex<LayoutTransform = MatLayoutTransform, Rows = Rows, Cols = Cols, Index = (Idx<Rows>, Idx<Cols>)>,
	R: MatIndex<LayoutTransform = MatLayoutTransform, Rows = Rows, Cols = Cols, Index = (Idx<Rows>, Idx<Cols>)>,
> ZipEq<Rows, Cols, L, R>
where
	L::Dyn: MatIndex<Rows = usize, Cols = usize, Index = (usize, usize)>,
	R::Dyn: MatIndex<Rows = usize, Cols = usize, Index = (usize, usize)>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_mat(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the position of the
	/// current element
	#[inline(always)]
	pub fn for_each_with_index(self, f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item)) {
		for_each_mat_with_index(self, f);
	}

	/// applies `f` to each element of the lower triangular half of `self`, while passing the
	/// indices of the position of the current element
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_lower_with_index(self, diag: Diag, f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item)) {
		for_each_mat_triangular_lower_with_index(self, diag, f);
	}

	/// applies `f` to each element of the upper triangular half of `self`, while passing the
	/// indices of the position of the current element
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_upper_with_index(self, diag: Diag, f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item)) {
		for_each_mat_triangular_upper_with_index(self, diag, f);
	}

	/// applies `f` to each element of the lower triangular half of `self`
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_lower(self, diag: Diag, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_mat_triangular_lower(self, diag, false, f);
	}

	/// applies `f` to each element of the upper triangular half of `self`
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_upper(self, diag: Diag, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_mat_triangular_lower(self, diag, true, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map<T>(self, f: impl FnMut(<Self as MatIndex>::Item) -> T) -> Mat<T, Rows, Cols> {
		let (m, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Mat::from_fn(
			m,
			n,
			#[inline(always)]
			|i, j| f(unsafe { Self::get_unchecked(&mut this, (i, j)) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map_with_index<T>(self, f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item) -> T) -> Mat<T, Rows, Cols> {
		let (m, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;

		Mat::from_fn(
			m,
			n,
			#[inline(always)]
			|i, j| f(i, j, unsafe { Self::get_unchecked(&mut this, (i, j)) }),
		)
	}
}

impl<Rows: Shape, M: MatIndex<LayoutTransform = VecLayoutTransform, Rows = Rows, Cols = (), Index = Idx<Rows>>> LastEq<Rows, (), M>
where
	M::Dyn: MatIndex<Rows = usize, Cols = (), Index = usize>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_col(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the position of the
	/// current element
	#[inline(always)]
	pub fn for_each_with_index(self, f: impl FnMut(Idx<Rows>, <Self as MatIndex>::Item)) {
		for_each_col_with_index(self, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map<T>(self, f: impl FnMut(<Self as MatIndex>::Item) -> T) -> Col<T, Rows> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Col::from_fn(
			m,
			#[inline(always)]
			|i| f(unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map_with_index<T>(self, f: impl FnMut(Idx<Rows>, <Self as MatIndex>::Item) -> T) -> Col<T, Rows> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;

		Col::from_fn(
			m,
			#[inline(always)]
			|i| f(i, unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}
}

impl<
	Rows: Shape,
	L: MatIndex<LayoutTransform = VecLayoutTransform, Rows = Rows, Cols = (), Index = Idx<Rows>>,
	R: MatIndex<LayoutTransform = VecLayoutTransform, Rows = Rows, Cols = (), Index = Idx<Rows>>,
> ZipEq<Rows, (), L, R>
where
	L::Dyn: MatIndex<Rows = usize, Cols = (), Index = usize>,
	R::Dyn: MatIndex<Rows = usize, Cols = (), Index = usize>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_col(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the position of the
	/// current element
	#[inline(always)]
	pub fn for_each_with_index(self, f: impl FnMut(Idx<Rows>, <Self as MatIndex>::Item)) {
		for_each_col_with_index(self, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map<T>(self, f: impl FnMut(<Self as MatIndex>::Item) -> T) -> Col<T, Rows> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Col::from_fn(
			m,
			#[inline(always)]
			|i| f(unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map_with_index<T>(self, f: impl FnMut(Idx<Rows>, <Self as MatIndex>::Item) -> T) -> Col<T, Rows> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;

		Col::from_fn(
			m,
			#[inline(always)]
			|i| f(i, unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}
}

impl<Cols: Shape, M: MatIndex<LayoutTransform = VecLayoutTransform, Rows = (), Cols = Cols, Index = Idx<Cols>>> LastEq<(), Cols, M>
where
	M::Dyn: MatIndex<Rows = (), Cols = usize, Index = usize>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_row(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the position of the
	/// current element
	#[inline(always)]
	pub fn for_each_with_index(self, f: impl FnMut(Idx<Cols>, <Self as MatIndex>::Item)) {
		for_each_row_with_index(self, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map<T>(self, f: impl FnMut(<Self as MatIndex>::Item) -> T) -> Row<T, Cols> {
		let (_, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Row::from_fn(
			n,
			#[inline(always)]
			|i| f(unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map_with_index<T>(self, f: impl FnMut(Idx<Cols>, <Self as MatIndex>::Item) -> T) -> Row<T, Cols> {
		let (_, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;

		Row::from_fn(
			n,
			#[inline(always)]
			|i| f(i, unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}
}

impl<
	Cols: Shape,
	L: MatIndex<LayoutTransform = VecLayoutTransform, Rows = (), Cols = Cols, Index = Idx<Cols>>,
	R: MatIndex<LayoutTransform = VecLayoutTransform, Rows = (), Cols = Cols, Index = Idx<Cols>>,
> ZipEq<(), Cols, L, R>
where
	L::Dyn: MatIndex<Rows = (), Cols = usize, Index = usize>,
	R::Dyn: MatIndex<Rows = (), Cols = usize, Index = usize>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_row(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the position of the
	/// current element
	#[inline(always)]
	pub fn for_each_with_index(self, f: impl FnMut(Idx<Cols>, <Self as MatIndex>::Item)) {
		for_each_row_with_index(self, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map<T>(self, f: impl FnMut(<Self as MatIndex>::Item) -> T) -> Row<T, Cols> {
		let (_, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Row::from_fn(
			n,
			#[inline(always)]
			|i| f(unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new matrix
	#[inline(always)]
	pub fn map_with_index<T>(self, f: impl FnMut(Idx<Cols>, <Self as MatIndex>::Item) -> T) -> Row<T, Cols> {
		let (_, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;

		Row::from_fn(
			n,
			#[inline(always)]
			|i| f(i, unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}
}
