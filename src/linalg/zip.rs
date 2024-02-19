//! Implementation of [`zipped!`] structures.

use self::{
    col::{Col, ColMut, ColRef},
    mat::{Mat, MatMut, MatRef},
    row::{Row, RowMut, RowRef},
};
use crate::{assert, debug_assert, *};
use core::mem::MaybeUninit;
use faer_entity::*;
use reborrow::*;

/// Read only view over a single matrix element.
pub struct Read<'a, E: Entity> {
    ptr: GroupFor<E, &'a MaybeUninit<E::Unit>>,
}
/// Read-write view over a single matrix element.
pub struct ReadWrite<'a, E: Entity> {
    ptr: GroupFor<E, &'a mut MaybeUninit<E::Unit>>,
}

/// Type that can be converted to a view.
pub trait ViewMut {
    /// View type.
    type Target<'a>
    where
        Self: 'a;

    /// Returns the view over self.
    fn view_mut(&mut self) -> Self::Target<'_>;
}

impl<E: Entity> ViewMut for Row<E> {
    type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        self.as_ref()
    }
}
impl<E: Entity> ViewMut for &Row<E> {
    type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (*self).as_ref()
    }
}
impl<E: Entity> ViewMut for &mut Row<E> {
    type Target<'a> = RowMut<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (*self).as_mut()
    }
}

impl<E: Entity> ViewMut for RowRef<'_, E> {
    type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        *self
    }
}
impl<E: Entity> ViewMut for RowMut<'_, E> {
    type Target<'a> = RowMut<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (*self).rb_mut()
    }
}
impl<E: Entity> ViewMut for &mut RowRef<'_, E> {
    type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        **self
    }
}
impl<E: Entity> ViewMut for &mut RowMut<'_, E> {
    type Target<'a> = RowMut<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (**self).rb_mut()
    }
}
impl<E: Entity> ViewMut for &RowRef<'_, E> {
    type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        **self
    }
}
impl<E: Entity> ViewMut for &RowMut<'_, E> {
    type Target<'a> = RowRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (**self).rb()
    }
}

impl<E: Entity> ViewMut for Col<E> {
    type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        self.as_ref()
    }
}
impl<E: Entity> ViewMut for &Col<E> {
    type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (*self).as_ref()
    }
}
impl<E: Entity> ViewMut for &mut Col<E> {
    type Target<'a> = ColMut<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (*self).as_mut()
    }
}

impl<E: Entity> ViewMut for ColRef<'_, E> {
    type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        *self
    }
}
impl<E: Entity> ViewMut for ColMut<'_, E> {
    type Target<'a> = ColMut<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (*self).rb_mut()
    }
}
impl<E: Entity> ViewMut for &mut ColRef<'_, E> {
    type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        **self
    }
}
impl<E: Entity> ViewMut for &mut ColMut<'_, E> {
    type Target<'a> = ColMut<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (**self).rb_mut()
    }
}
impl<E: Entity> ViewMut for &ColRef<'_, E> {
    type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        **self
    }
}
impl<E: Entity> ViewMut for &ColMut<'_, E> {
    type Target<'a> = ColRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (**self).rb()
    }
}

impl<E: Entity> ViewMut for Mat<E> {
    type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        self.as_ref()
    }
}
impl<E: Entity> ViewMut for &Mat<E> {
    type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (*self).as_ref()
    }
}
impl<E: Entity> ViewMut for &mut Mat<E> {
    type Target<'a> = MatMut<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (*self).as_mut()
    }
}

impl<E: Entity> ViewMut for MatRef<'_, E> {
    type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        *self
    }
}
impl<E: Entity> ViewMut for MatMut<'_, E> {
    type Target<'a> = MatMut<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (*self).rb_mut()
    }
}
impl<E: Entity> ViewMut for &mut MatRef<'_, E> {
    type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        **self
    }
}
impl<E: Entity> ViewMut for &mut MatMut<'_, E> {
    type Target<'a> = MatMut<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (**self).rb_mut()
    }
}
impl<E: Entity> ViewMut for &MatRef<'_, E> {
    type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        **self
    }
}
impl<E: Entity> ViewMut for &MatMut<'_, E> {
    type Target<'a> = MatRef<'a, E>
        where
            Self: 'a;

    #[inline]
    fn view_mut(&mut self) -> Self::Target<'_> {
        (**self).rb()
    }
}

impl<E: SimpleEntity> core::ops::Deref for Read<'_, E> {
    type Target = E;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.ptr as *const _ as *const E::Unit) }
    }
}
impl<E: SimpleEntity> core::ops::Deref for ReadWrite<'_, E> {
    type Target = E;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.ptr as *const _ as *const E::Unit) }
    }
}
impl<E: SimpleEntity> core::ops::DerefMut for ReadWrite<'_, E> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self.ptr as *mut _ as *mut E::Unit) }
    }
}

impl<E: Entity> Read<'_, E> {
    /// Read the value of the element.
    #[inline(always)]
    pub fn read(&self) -> E {
        E::faer_from_units(E::faer_map(
            E::faer_as_ref(&self.ptr),
            #[inline(always)]
            |ptr| unsafe { ptr.assume_init_read() },
        ))
    }
}
impl<E: Entity> ReadWrite<'_, E> {
    /// Read the value of the element.
    #[inline(always)]
    pub fn read(&self) -> E {
        E::faer_from_units(E::faer_map(
            E::faer_as_ref(&self.ptr),
            #[inline(always)]
            |ptr| unsafe { *ptr.assume_init_ref() },
        ))
    }

    /// Write to the location of the element.
    #[inline(always)]
    pub fn write(&mut self, value: E) {
        let value = E::faer_into_units(value);
        E::faer_map(
            E::faer_zip(E::faer_as_mut(&mut self.ptr), value),
            #[inline(always)]
            |(ptr, value)| unsafe { *ptr.assume_init_mut() = value },
        );
    }
}

/// Specifies whether the main diagonal should be traversed, when iterating over a triangular
/// chunk of the matrix.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Diag {
    /// Do not include diagonal of matrix
    Skip,
    /// Include diagonal of matrix
    Include,
}

/// Matrix layout transformation. Used for zipping optimizations.
#[derive(Copy, Clone)]
pub enum MatLayoutTransform {
    /// Matrix is used as-is.
    None,
    /// Matrix rows are reversed.
    ReverseRows,
    /// Matrix is transposed.
    Transpose,
    /// Matrix is transposed, then rows are reversed.
    TransposeReverseRows,
}

/// Vector layout transformation. Used for zipping optimizations.
#[derive(Copy, Clone)]
pub enum VecLayoutTransform {
    /// Vector is used as-is.
    None,
    /// Vector is reversed.
    Reverse,
}

/// Type with a given matrix shape.
pub trait MatShape {
    /// Type of rows.
    type Rows: Copy + Eq;
    /// Type of columns.
    type Cols: Copy + Eq;
    /// Returns the number of rows.
    fn nrows(&self) -> Self::Rows;
    /// Returns the number of columns.
    fn ncols(&self) -> Self::Cols;
}

/// Zipped matrix views.
pub unsafe trait MaybeContiguous: MatShape {
    /// Indexing type.
    type Index: Copy;
    /// Contiguous slice type.
    type Slice;
    /// Layout transformation type.
    type LayoutTransform: Copy;
    /// Returns slice at index of length `n_elems`.
    unsafe fn get_slice_unchecked(&mut self, idx: Self::Index, n_elems: usize) -> Self::Slice;
}

/// Zipped matrix views.
pub unsafe trait MatIndex<'a, _Outlives = &'a Self>: MaybeContiguous {
    /// Item produced by the zipped views.
    type Item;

    /// Get the item at the given index, skipping bound checks.
    unsafe fn get_unchecked(&'a mut self, index: Self::Index) -> Self::Item;
    /// Get the item at the given slice position, skipping bound checks.
    unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item;

    /// Checks if the zipped matrices are contiguous.
    fn is_contiguous(&self) -> bool;
    /// Computes the preferred iteration layout of the matrices.
    fn preferred_layout(&self) -> Self::LayoutTransform;
    /// Applies the layout transformation to the matrices.
    fn with_layout(self, layout: Self::LayoutTransform) -> Self;
}

/// Single element.
#[derive(Copy, Clone, Debug)]
pub struct Last<Mat>(pub Mat);

/// Zipped elements.
#[derive(Copy, Clone, Debug)]
pub struct Zip<Head, Tail>(pub Head, pub Tail);

/// Single matrix view.
#[derive(Copy, Clone, Debug)]
pub struct LastEq<Rows, Cols, Mat: MatShape<Rows = Rows, Cols = Cols>>(pub Mat);
/// Zipped matrix views.
#[derive(Copy, Clone, Debug)]
pub struct ZipEq<
    Rows,
    Cols,
    Head: MatShape<Rows = Rows, Cols = Cols>,
    Tail: MatShape<Rows = Rows, Cols = Cols>,
>(Head, Tail);

impl<
        Rows: Copy + Eq,
        Cols: Copy + Eq,
        Head: MatShape<Rows = Rows, Cols = Cols>,
        Tail: MatShape<Rows = Rows, Cols = Cols>,
    > ZipEq<Rows, Cols, Head, Tail>
{
    /// Creates a zipped matrix, after asserting that the dimensions match.
    #[inline(always)]
    pub fn new(head: Head, tail: Tail) -> Self {
        assert!((head.nrows(), head.ncols()) == (tail.nrows(), tail.ncols()));
        Self(head, tail)
    }

    /// Creates a zipped matrix, assuming that the dimensions match.
    #[inline(always)]
    pub fn new_unchecked(head: Head, tail: Tail) -> Self {
        debug_assert!((head.nrows(), head.ncols()) == (tail.nrows(), tail.ncols()));
        Self(head, tail)
    }
}

impl<Rows: Copy + Eq, Cols: Copy + Eq, Mat: MatShape<Rows = Rows, Cols = Cols>> MatShape
    for LastEq<Rows, Cols, Mat>
{
    type Rows = Rows;
    type Cols = Cols;
    #[inline(always)]
    fn nrows(&self) -> Self::Rows {
        self.0.nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> Self::Cols {
        self.0.ncols()
    }
}

impl<
        Rows: Copy + Eq,
        Cols: Copy + Eq,
        Head: MatShape<Rows = Rows, Cols = Cols>,
        Tail: MatShape<Rows = Rows, Cols = Cols>,
    > MatShape for ZipEq<Rows, Cols, Head, Tail>
{
    type Rows = Rows;
    type Cols = Cols;
    #[inline(always)]
    fn nrows(&self) -> Self::Rows {
        self.0.nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> Self::Cols {
        self.0.ncols()
    }
}

impl<E: Entity> MatShape for ColRef<'_, E> {
    type Rows = usize;
    type Cols = ();
    #[inline(always)]
    fn nrows(&self) -> Self::Rows {
        (*self).nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> Self::Cols {
        ()
    }
}

impl<E: Entity> MatShape for ColMut<'_, E> {
    type Rows = usize;
    type Cols = ();
    #[inline(always)]
    fn nrows(&self) -> Self::Rows {
        (*self).nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> Self::Cols {
        ()
    }
}

impl<E: Entity> MatShape for RowRef<'_, E> {
    type Rows = ();
    type Cols = usize;
    #[inline(always)]
    fn nrows(&self) -> Self::Rows {
        ()
    }
    #[inline(always)]
    fn ncols(&self) -> Self::Cols {
        (*self).ncols()
    }
}
impl<E: Entity> MatShape for RowMut<'_, E> {
    type Rows = ();
    type Cols = usize;
    #[inline(always)]
    fn nrows(&self) -> Self::Rows {
        ()
    }
    #[inline(always)]
    fn ncols(&self) -> Self::Cols {
        (*self).ncols()
    }
}

impl<E: Entity> MatShape for MatRef<'_, E> {
    type Rows = usize;
    type Cols = usize;
    #[inline(always)]
    fn nrows(&self) -> Self::Rows {
        (*self).nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> Self::Cols {
        (*self).ncols()
    }
}

impl<E: Entity> MatShape for MatMut<'_, E> {
    type Rows = usize;
    type Cols = usize;
    #[inline(always)]
    fn nrows(&self) -> Self::Rows {
        (*self).nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> Self::Cols {
        (*self).ncols()
    }
}

unsafe impl<Rows: Copy + Eq, Cols: Copy + Eq, Mat: MaybeContiguous<Rows = Rows, Cols = Cols>>
    MaybeContiguous for LastEq<Rows, Cols, Mat>
{
    type Index = Mat::Index;
    type Slice = Last<Mat::Slice>;
    type LayoutTransform = Mat::LayoutTransform;
    #[inline(always)]
    unsafe fn get_slice_unchecked(&mut self, idx: Self::Index, n_elems: usize) -> Self::Slice {
        Last(self.0.get_slice_unchecked(idx, n_elems))
    }
}

unsafe impl<'a, Rows: Copy + Eq, Cols: Copy + Eq, Mat: MatIndex<'a, Rows = Rows, Cols = Cols>>
    MatIndex<'a> for LastEq<Rows, Cols, Mat>
{
    type Item = Last<Mat::Item>;

    #[inline(always)]
    unsafe fn get_unchecked(&'a mut self, index: Self::Index) -> Self::Item {
        Last(self.0.get_unchecked(index))
    }

    #[inline(always)]
    unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
        Last(Mat::get_from_slice_unchecked(&mut slice.0, idx))
    }

    #[inline(always)]
    fn is_contiguous(&self) -> bool {
        self.0.is_contiguous()
    }
    #[inline(always)]
    fn preferred_layout(&self) -> Self::LayoutTransform {
        self.0.preferred_layout()
    }
    #[inline(always)]
    fn with_layout(self, layout: Self::LayoutTransform) -> Self {
        Self(self.0.with_layout(layout))
    }
}

unsafe impl<
        Rows: Copy + Eq,
        Cols: Copy + Eq,
        Head: MaybeContiguous<Rows = Rows, Cols = Cols>,
        Tail: MaybeContiguous<
            Rows = Rows,
            Cols = Cols,
            Index = Head::Index,
            LayoutTransform = Head::LayoutTransform,
        >,
    > MaybeContiguous for ZipEq<Rows, Cols, Head, Tail>
{
    type Index = Head::Index;
    type Slice = Zip<Head::Slice, Tail::Slice>;
    type LayoutTransform = Head::LayoutTransform;
    #[inline(always)]
    unsafe fn get_slice_unchecked(&mut self, idx: Self::Index, n_elems: usize) -> Self::Slice {
        Zip(
            self.0.get_slice_unchecked(idx, n_elems),
            self.1.get_slice_unchecked(idx, n_elems),
        )
    }
}

unsafe impl<
        'a,
        Rows: Copy + Eq,
        Cols: Copy + Eq,
        Head: MatIndex<'a, Rows = Rows, Cols = Cols>,
        Tail: MatIndex<
            'a,
            Rows = Rows,
            Cols = Cols,
            Index = Head::Index,
            LayoutTransform = Head::LayoutTransform,
        >,
    > MatIndex<'a> for ZipEq<Rows, Cols, Head, Tail>
{
    type Item = Zip<Head::Item, Tail::Item>;

    #[inline(always)]
    unsafe fn get_unchecked(&'a mut self, index: Self::Index) -> Self::Item {
        Zip(self.0.get_unchecked(index), self.1.get_unchecked(index))
    }

    #[inline(always)]
    unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
        Zip(
            Head::get_from_slice_unchecked(&mut slice.0, idx),
            Tail::get_from_slice_unchecked(&mut slice.1, idx),
        )
    }

    #[inline(always)]
    fn is_contiguous(&self) -> bool {
        self.0.is_contiguous() && self.1.is_contiguous()
    }
    #[inline(always)]
    fn preferred_layout(&self) -> Self::LayoutTransform {
        self.0.preferred_layout()
    }
    #[inline(always)]
    fn with_layout(self, layout: Self::LayoutTransform) -> Self {
        ZipEq(self.0.with_layout(layout), self.1.with_layout(layout))
    }
}

unsafe impl<E: Entity> MaybeContiguous for ColRef<'_, E> {
    type Index = (usize, ());
    type Slice = GroupFor<E, &'static [MaybeUninit<E::Unit>]>;
    type LayoutTransform = VecLayoutTransform;

    #[inline(always)]
    unsafe fn get_slice_unchecked(&mut self, (i, _): Self::Index, n_elems: usize) -> Self::Slice {
        E::faer_map(
            (*self).rb().ptr_at(i),
            #[inline(always)]
            |ptr| core::slice::from_raw_parts(ptr as *const MaybeUninit<E::Unit>, n_elems),
        )
    }
}
unsafe impl<'a, E: Entity> MatIndex<'a> for ColRef<'_, E> {
    type Item = Read<'a, E>;

    #[inline(always)]
    unsafe fn get_unchecked(&'a mut self, (i, _): Self::Index) -> Self::Item {
        Read {
            ptr: E::faer_map(
                self.rb().ptr_inbounds_at(i),
                #[inline(always)]
                |ptr| &*(ptr as *const MaybeUninit<E::Unit>),
            ),
        }
    }

    #[inline(always)]
    unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
        let slice = E::faer_rb(E::faer_as_ref(slice));
        Read {
            ptr: E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.get_unchecked(idx),
            ),
        }
    }

    #[inline(always)]
    fn is_contiguous(&self) -> bool {
        self.row_stride() == 1
    }
    #[inline(always)]
    fn preferred_layout(&self) -> Self::LayoutTransform {
        let rs = self.row_stride();
        if self.nrows() > 1 && rs == 1 {
            VecLayoutTransform::None
        } else if self.nrows() > 1 && rs == -1 {
            VecLayoutTransform::Reverse
        } else {
            VecLayoutTransform::None
        }
    }
    #[inline(always)]
    fn with_layout(self, layout: Self::LayoutTransform) -> Self {
        use VecLayoutTransform::*;
        match layout {
            None => self,
            Reverse => self.reverse_rows(),
        }
    }
}

unsafe impl<E: Entity> MaybeContiguous for ColMut<'_, E> {
    type Index = (usize, ());
    type Slice = GroupFor<E, &'static mut [MaybeUninit<E::Unit>]>;
    type LayoutTransform = VecLayoutTransform;

    #[inline(always)]
    unsafe fn get_slice_unchecked(&mut self, (i, _): Self::Index, n_elems: usize) -> Self::Slice {
        E::faer_map(
            (*self).rb_mut().ptr_at_mut(i),
            #[inline(always)]
            |ptr| core::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<E::Unit>, n_elems),
        )
    }
}
unsafe impl<'a, E: Entity> MatIndex<'a> for ColMut<'_, E> {
    type Item = ReadWrite<'a, E>;

    #[inline(always)]
    unsafe fn get_unchecked(&'a mut self, (i, _): Self::Index) -> Self::Item {
        ReadWrite {
            ptr: E::faer_map(
                self.rb_mut().ptr_inbounds_at_mut(i),
                #[inline(always)]
                |ptr| &mut *(ptr as *mut MaybeUninit<E::Unit>),
            ),
        }
    }

    #[inline(always)]
    unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
        let slice = E::faer_rb_mut(E::faer_as_mut(slice));
        ReadWrite {
            ptr: E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.get_unchecked_mut(idx),
            ),
        }
    }

    #[inline(always)]
    fn is_contiguous(&self) -> bool {
        self.row_stride() == 1
    }
    #[inline(always)]
    fn preferred_layout(&self) -> Self::LayoutTransform {
        let rs = self.row_stride();
        if self.nrows() > 1 && rs == 1 {
            VecLayoutTransform::None
        } else if self.nrows() > 1 && rs == -1 {
            VecLayoutTransform::Reverse
        } else {
            VecLayoutTransform::None
        }
    }
    #[inline(always)]
    fn with_layout(self, layout: Self::LayoutTransform) -> Self {
        use VecLayoutTransform::*;
        match layout {
            None => self,
            Reverse => self.reverse_rows_mut(),
        }
    }
}

unsafe impl<E: Entity> MaybeContiguous for RowRef<'_, E> {
    type Index = ((), usize);
    type Slice = GroupFor<E, &'static [MaybeUninit<E::Unit>]>;
    type LayoutTransform = VecLayoutTransform;

    #[inline(always)]
    unsafe fn get_slice_unchecked(&mut self, (_, j): Self::Index, n_elems: usize) -> Self::Slice {
        E::faer_map(
            (*self).rb().ptr_at(j),
            #[inline(always)]
            |ptr| core::slice::from_raw_parts(ptr as *const MaybeUninit<E::Unit>, n_elems),
        )
    }
}
unsafe impl<'a, E: Entity> MatIndex<'a> for RowRef<'_, E> {
    type Item = Read<'a, E>;

    #[inline(always)]
    unsafe fn get_unchecked(&'a mut self, (_, j): Self::Index) -> Self::Item {
        Read {
            ptr: E::faer_map(
                self.rb().ptr_inbounds_at(j),
                #[inline(always)]
                |ptr| &*(ptr as *const MaybeUninit<E::Unit>),
            ),
        }
    }

    #[inline(always)]
    unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
        let slice = E::faer_rb(E::faer_as_ref(slice));
        Read {
            ptr: E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.get_unchecked(idx),
            ),
        }
    }

    #[inline(always)]
    fn is_contiguous(&self) -> bool {
        self.col_stride() == 1
    }
    #[inline(always)]
    fn preferred_layout(&self) -> Self::LayoutTransform {
        let cs = self.col_stride();
        if self.ncols() > 1 && cs == 1 {
            VecLayoutTransform::None
        } else if self.ncols() > 1 && cs == -1 {
            VecLayoutTransform::Reverse
        } else {
            VecLayoutTransform::None
        }
    }
    #[inline(always)]
    fn with_layout(self, layout: Self::LayoutTransform) -> Self {
        use VecLayoutTransform::*;
        match layout {
            None => self,
            Reverse => self.reverse_cols(),
        }
    }
}

unsafe impl<E: Entity> MaybeContiguous for RowMut<'_, E> {
    type Index = ((), usize);
    type Slice = GroupFor<E, &'static mut [MaybeUninit<E::Unit>]>;
    type LayoutTransform = VecLayoutTransform;

    #[inline(always)]
    unsafe fn get_slice_unchecked(&mut self, (_, j): Self::Index, n_elems: usize) -> Self::Slice {
        E::faer_map(
            (*self).rb_mut().ptr_at_mut(j),
            #[inline(always)]
            |ptr| core::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<E::Unit>, n_elems),
        )
    }
}
unsafe impl<'a, E: Entity> MatIndex<'a> for RowMut<'_, E> {
    type Item = ReadWrite<'a, E>;

    #[inline(always)]
    unsafe fn get_unchecked(&'a mut self, (_, j): Self::Index) -> Self::Item {
        ReadWrite {
            ptr: E::faer_map(
                self.rb_mut().ptr_inbounds_at_mut(j),
                #[inline(always)]
                |ptr| &mut *(ptr as *mut MaybeUninit<E::Unit>),
            ),
        }
    }

    #[inline(always)]
    unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
        let slice = E::faer_rb_mut(E::faer_as_mut(slice));
        ReadWrite {
            ptr: E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.get_unchecked_mut(idx),
            ),
        }
    }

    #[inline(always)]
    fn is_contiguous(&self) -> bool {
        self.col_stride() == 1
    }
    #[inline(always)]
    fn preferred_layout(&self) -> Self::LayoutTransform {
        let cs = self.col_stride();
        if self.ncols() > 1 && cs == 1 {
            VecLayoutTransform::None
        } else if self.ncols() > 1 && cs == -1 {
            VecLayoutTransform::Reverse
        } else {
            VecLayoutTransform::None
        }
    }
    #[inline(always)]
    fn with_layout(self, layout: Self::LayoutTransform) -> Self {
        use VecLayoutTransform::*;
        match layout {
            None => self,
            Reverse => self.reverse_cols_mut(),
        }
    }
}

unsafe impl<E: Entity> MaybeContiguous for MatRef<'_, E> {
    type Index = (usize, usize);
    type Slice = GroupFor<E, &'static [MaybeUninit<E::Unit>]>;
    type LayoutTransform = MatLayoutTransform;

    #[inline(always)]
    unsafe fn get_slice_unchecked(&mut self, (i, j): Self::Index, n_elems: usize) -> Self::Slice {
        E::faer_map(
            (*self).rb().overflowing_ptr_at(i, j),
            #[inline(always)]
            |ptr| core::slice::from_raw_parts(ptr as *const MaybeUninit<E::Unit>, n_elems),
        )
    }
}
unsafe impl<'a, E: Entity> MatIndex<'a> for MatRef<'_, E> {
    type Item = Read<'a, E>;

    #[inline(always)]
    unsafe fn get_unchecked(&'a mut self, (i, j): Self::Index) -> Self::Item {
        Read {
            ptr: E::faer_map(
                self.rb().ptr_inbounds_at(i, j),
                #[inline(always)]
                |ptr| &*(ptr as *const MaybeUninit<E::Unit>),
            ),
        }
    }

    #[inline(always)]
    unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
        let slice = E::faer_rb(E::faer_as_ref(slice));
        Read {
            ptr: E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.get_unchecked(idx),
            ),
        }
    }

    #[inline(always)]
    fn is_contiguous(&self) -> bool {
        self.row_stride() == 1
    }
    #[inline(always)]
    fn preferred_layout(&self) -> Self::LayoutTransform {
        let rs = self.row_stride();
        let cs = self.col_stride();
        if self.nrows() > 1 && rs == 1 {
            MatLayoutTransform::None
        } else if self.nrows() > 1 && rs == -1 {
            MatLayoutTransform::ReverseRows
        } else if self.ncols() > 1 && cs == 1 {
            MatLayoutTransform::Transpose
        } else if self.ncols() > 1 && cs == -1 {
            MatLayoutTransform::TransposeReverseRows
        } else {
            MatLayoutTransform::None
        }
    }
    #[inline(always)]
    fn with_layout(self, layout: Self::LayoutTransform) -> Self {
        use MatLayoutTransform::*;
        match layout {
            None => self,
            ReverseRows => self.reverse_rows(),
            Transpose => self.transpose(),
            TransposeReverseRows => self.transpose().reverse_rows(),
        }
    }
}

unsafe impl<E: Entity> MaybeContiguous for MatMut<'_, E> {
    type Index = (usize, usize);
    type Slice = GroupFor<E, &'static mut [MaybeUninit<E::Unit>]>;
    type LayoutTransform = MatLayoutTransform;

    #[inline(always)]
    unsafe fn get_slice_unchecked(&mut self, (i, j): Self::Index, n_elems: usize) -> Self::Slice {
        E::faer_map(
            (*self).rb().overflowing_ptr_at(i, j),
            #[inline(always)]
            |ptr| core::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<E::Unit>, n_elems),
        )
    }
}

unsafe impl<'a, E: Entity> MatIndex<'a> for MatMut<'_, E> {
    type Item = ReadWrite<'a, E>;

    #[inline(always)]
    unsafe fn get_unchecked(&'a mut self, (i, j): Self::Index) -> Self::Item {
        ReadWrite {
            ptr: E::faer_map(
                self.rb_mut().ptr_inbounds_at_mut(i, j),
                #[inline(always)]
                |ptr| &mut *(ptr as *mut MaybeUninit<E::Unit>),
            ),
        }
    }

    #[inline(always)]
    unsafe fn get_from_slice_unchecked(slice: &'a mut Self::Slice, idx: usize) -> Self::Item {
        let slice = E::faer_rb_mut(E::faer_as_mut(slice));
        ReadWrite {
            ptr: E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.get_unchecked_mut(idx),
            ),
        }
    }

    #[inline(always)]
    fn is_contiguous(&self) -> bool {
        self.row_stride() == 1
    }
    #[inline(always)]
    fn preferred_layout(&self) -> Self::LayoutTransform {
        let rs = self.row_stride();
        let cs = self.col_stride();
        if self.nrows() > 1 && rs == 1 {
            MatLayoutTransform::None
        } else if self.nrows() > 1 && rs == -1 {
            MatLayoutTransform::ReverseRows
        } else if self.ncols() > 1 && cs == 1 {
            MatLayoutTransform::Transpose
        } else if self.ncols() > 1 && cs == -1 {
            MatLayoutTransform::TransposeReverseRows
        } else {
            MatLayoutTransform::None
        }
    }
    #[inline(always)]
    fn with_layout(self, layout: Self::LayoutTransform) -> Self {
        use MatLayoutTransform::*;
        match layout {
            None => self,
            ReverseRows => self.reverse_rows_mut(),
            Transpose => self.transpose_mut(),
            TransposeReverseRows => self.transpose_mut().reverse_rows_mut(),
        }
    }
}

#[inline(always)]
fn annotate_noalias_mat<Z: for<'a> MatIndex<'a>>(
    f: &mut impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
    mut slice: Z::Slice,
    i_begin: usize,
    i_end: usize,
    _j: usize,
) {
    for i in i_begin..i_end {
        unsafe { f(Z::get_from_slice_unchecked(&mut slice, i - i_begin)) };
    }
}

#[inline(always)]
fn annotate_noalias_col<Z: for<'a> MatIndex<'a>>(
    f: &mut impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
    mut slice: Z::Slice,
    i_begin: usize,
    i_end: usize,
) {
    for i in i_begin..i_end {
        unsafe { f(Z::get_from_slice_unchecked(&mut slice, i - i_begin)) };
    }
}

#[inline(always)]
fn for_each_mat<Z: for<'a> MatIndex<'a, Rows = usize, Cols = usize, Index = (usize, usize)>>(
    z: Z,
    mut f: impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
) {
    let layout = z.preferred_layout();
    let mut z = z.with_layout(layout);

    let m = z.nrows();
    let n = z.ncols();
    if m == 0 || n == 0 {
        return;
    }

    unsafe {
        if z.is_contiguous() {
            for j in 0..n {
                annotate_noalias_mat::<Z>(&mut f, z.get_slice_unchecked((0, j), m), 0, m, j);
            }
        } else {
            for j in 0..n {
                for i in 0..m {
                    f(z.get_unchecked((i, j)))
                }
            }
        }
    }
}

#[inline(always)]
fn for_each_mat_triangular_lower<
    Z: for<'a> MatIndex<
        'a,
        Rows = usize,
        Cols = usize,
        Index = (usize, usize),
        LayoutTransform = MatLayoutTransform,
    >,
>(
    z: Z,
    diag: Diag,
    transpose: bool,
    mut f: impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
) {
    use MatLayoutTransform::*;

    let z = if transpose {
        z.with_layout(MatLayoutTransform::Transpose)
    } else {
        z
    };
    let layout = z.preferred_layout();
    let mut z = z.with_layout(layout);

    let m = z.nrows();
    let n = z.ncols();
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
        if z.is_contiguous() {
            for j in 0..n {
                let (start, end) = match layout {
                    None => (j + strict as usize, m),
                    ReverseRows => (0, (m - (j + strict as usize))),
                    Transpose => (0, (j + !strict as usize).min(m)),
                    TransposeReverseRows => (m - ((j + !strict as usize).min(m)), m),
                };

                let len = end - start;

                annotate_noalias_mat::<Z>(
                    &mut f,
                    z.get_slice_unchecked((start, j), len),
                    start,
                    end,
                    j,
                );
            }
        } else {
            for j in 0..n {
                let (start, end) = match layout {
                    None => (j + strict as usize, m),
                    ReverseRows => (0, (m - (j + strict as usize))),
                    Transpose => (0, (j + !strict as usize).min(m)),
                    TransposeReverseRows => (m - ((j + !strict as usize).min(m)), m),
                };

                for i in start..end {
                    f(z.get_unchecked((i, j)))
                }
            }
        }
    }
}

#[inline(always)]
fn for_each_col<Z: for<'a> MatIndex<'a, Rows = usize, Cols = (), Index = (usize, ())>>(
    z: Z,
    mut f: impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
) {
    let layout = z.preferred_layout();
    let mut z = z.with_layout(layout);

    let m = z.nrows();
    if m == 0 {
        return;
    }

    unsafe {
        if z.is_contiguous() {
            annotate_noalias_col::<Z>(&mut f, z.get_slice_unchecked((0, ()), m), 0, m);
        } else {
            for i in 0..m {
                f(z.get_unchecked((i, ())))
            }
        }
    }
}

#[inline(always)]
fn for_each_row<Z: for<'a> MatIndex<'a, Rows = (), Cols = usize, Index = ((), usize)>>(
    z: Z,
    mut f: impl for<'a> FnMut(<Z as MatIndex<'a>>::Item),
) {
    let layout = z.preferred_layout();
    let mut z = z.with_layout(layout);

    let n = z.ncols();
    if n == 0 {
        return;
    }

    unsafe {
        if z.is_contiguous() {
            annotate_noalias_col::<Z>(&mut f, z.get_slice_unchecked(((), 0), n), 0, n);
        } else {
            for j in 0..n {
                f(z.get_unchecked(((), j)))
            }
        }
    }
}

impl<
        M: for<'a> MatIndex<
            'a,
            Rows = usize,
            Cols = usize,
            Index = (usize, usize),
            LayoutTransform = MatLayoutTransform,
        >,
    > LastEq<usize, usize, M>
{
    /// Applies `f` to each element of `self`.
    #[inline(always)]
    pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
        for_each_mat(self, f);
    }

    /// Applies `f` to each element of the lower triangular half of `self`.
    ///
    /// `diag` specifies whether the diagonal should be included or excluded.
    #[inline(always)]
    pub fn for_each_triangular_lower(
        self,
        diag: Diag,
        f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item),
    ) {
        for_each_mat_triangular_lower(self, diag, false, f);
    }

    /// Applies `f` to each element of the upper triangular half of `self`.
    ///
    /// `diag` specifies whether the diagonal should be included or excluded.
    #[inline(always)]
    pub fn for_each_triangular_upper(
        self,
        diag: Diag,
        f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item),
    ) {
        for_each_mat_triangular_lower(self, diag, true, f);
    }

    /// Applies `f` to each element of `self` and collect its result into a new matrix.
    #[inline(always)]
    pub fn map<E: Entity>(
        self,
        f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
    ) -> Mat<E> {
        let (m, n) = (self.nrows(), self.ncols());
        let mut out = Mat::<E>::with_capacity(m, n);
        let rs = 1;
        let cs = out.col_stride();
        let out_view = unsafe { mat::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), m, n, rs, cs) };
        let mut f = f;
        ZipEq::new(out_view, self).for_each(
            #[inline(always)]
            |Zip(mut out, item)| out.write(f(item)),
        );
        unsafe { out.set_dims(m, n) };
        out
    }
}

impl<
        M: for<'a> MatIndex<
            'a,
            Rows = (),
            Cols = usize,
            Index = ((), usize),
            LayoutTransform = VecLayoutTransform,
        >,
    > LastEq<(), usize, M>
{
    /// Applies `f` to each element of `self`.
    #[inline(always)]
    pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
        for_each_row(self, f);
    }

    /// Applies `f` to each element of `self` and collect its result into a new row.
    #[inline(always)]
    pub fn map<E: Entity>(
        self,
        f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
    ) -> Row<E> {
        let (_, n) = (self.nrows(), self.ncols());
        let mut out = Row::<E>::with_capacity(n);
        let out_view = unsafe { row::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), n, 1) };
        let mut f = f;
        ZipEq::new(out_view, self).for_each(
            #[inline(always)]
            |Zip(mut out, item)| out.write(f(item)),
        );
        unsafe { out.set_ncols(n) };
        out
    }
}

impl<
        M: for<'a> MatIndex<
            'a,
            Rows = usize,
            Cols = (),
            Index = (usize, ()),
            LayoutTransform = VecLayoutTransform,
        >,
    > LastEq<usize, (), M>
{
    /// Applies `f` to each element of `self`.
    #[inline(always)]
    pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
        for_each_col(self, f);
    }

    /// Applies `f` to each element of `self` and collect its result into a new column.
    #[inline(always)]
    pub fn map<E: Entity>(
        self,
        f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
    ) -> Col<E> {
        let (m, _) = (self.nrows(), self.ncols());
        let mut out = Col::<E>::with_capacity(m);
        let out_view = unsafe { col::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), m, 1) };
        let mut f = f;
        ZipEq::new(out_view, self).for_each(
            #[inline(always)]
            |Zip(mut out, item)| out.write(f(item)),
        );
        unsafe { out.set_nrows(m) };
        out
    }
}

impl<
        Head: for<'a> MatIndex<
            'a,
            Rows = (),
            Cols = usize,
            Index = ((), usize),
            LayoutTransform = VecLayoutTransform,
        >,
        Tail: for<'a> MatIndex<
            'a,
            Rows = (),
            Cols = usize,
            Index = ((), usize),
            LayoutTransform = VecLayoutTransform,
        >,
    > ZipEq<(), usize, Head, Tail>
{
    /// Applies `f` to each element of `self`.
    #[inline(always)]
    pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
        for_each_row(self, f);
    }

    /// Applies `f` to each element of `self` and collect its result into a new row.
    #[inline(always)]
    pub fn map<E: Entity>(
        self,
        f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
    ) -> Row<E> {
        let (_, n) = (self.nrows(), self.ncols());
        let mut out = Row::<E>::with_capacity(n);
        let out_view = unsafe { row::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), n, 1) };
        let mut f = f;
        ZipEq::new(out_view, self).for_each(
            #[inline(always)]
            |Zip(mut out, item)| out.write(f(item)),
        );
        unsafe { out.set_ncols(n) };
        out
    }
}

impl<
        Head: for<'a> MatIndex<
            'a,
            Rows = usize,
            Cols = (),
            Index = (usize, ()),
            LayoutTransform = VecLayoutTransform,
        >,
        Tail: for<'a> MatIndex<
            'a,
            Rows = usize,
            Cols = (),
            Index = (usize, ()),
            LayoutTransform = VecLayoutTransform,
        >,
    > ZipEq<usize, (), Head, Tail>
{
    /// Applies `f` to each element of `self`.
    #[inline(always)]
    pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
        for_each_col(self, f);
    }

    /// Applies `f` to each element of `self` and collect its result into a new column.
    #[inline(always)]
    pub fn map<E: Entity>(
        self,
        f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
    ) -> Col<E> {
        let (m, _) = (self.nrows(), self.ncols());
        let mut out = Col::<E>::with_capacity(m);
        let out_view = unsafe { col::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), m, 1) };
        let mut f = f;
        ZipEq::new(out_view, self).for_each(
            #[inline(always)]
            |Zip(mut out, item)| out.write(f(item)),
        );
        unsafe { out.set_nrows(m) };
        out
    }
}

impl<
        Head: for<'a> MatIndex<
            'a,
            Rows = usize,
            Cols = usize,
            Index = (usize, usize),
            LayoutTransform = MatLayoutTransform,
        >,
        Tail: for<'a> MatIndex<
            'a,
            Rows = usize,
            Cols = usize,
            Index = (usize, usize),
            LayoutTransform = MatLayoutTransform,
        >,
    > ZipEq<usize, usize, Head, Tail>
{
    /// Applies `f` to each element of `self`.
    #[inline(always)]
    pub fn for_each(self, f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item)) {
        for_each_mat(self, f);
    }

    /// Applies `f` to each element of the lower triangular half of `self`.
    ///
    /// `diag` specifies whether the diagonal should be included or excluded.
    #[inline(always)]
    pub fn for_each_triangular_lower(
        self,
        diag: Diag,
        f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item),
    ) {
        for_each_mat_triangular_lower(self, diag, false, f);
    }

    /// Applies `f` to each element of the upper triangular half of `self`.
    ///
    /// `diag` specifies whether the diagonal should be included or excluded.
    #[inline(always)]
    pub fn for_each_triangular_upper(
        self,
        diag: Diag,
        f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item),
    ) {
        for_each_mat_triangular_lower(self, diag, true, f);
    }

    /// Applies `f` to each element of `self` and collect its result into a new matrix.
    #[inline(always)]
    pub fn map<E: Entity>(
        self,
        f: impl for<'a> FnMut(<Self as MatIndex<'a>>::Item) -> E,
    ) -> Mat<E> {
        let (m, n) = (self.nrows(), self.ncols());
        let mut out = Mat::<E>::with_capacity(m, n);
        let rs = 1;
        let cs = out.col_stride();
        let out_view = unsafe { mat::from_raw_parts_mut::<'_, E>(out.as_ptr_mut(), m, n, rs, cs) };
        let mut f = f;
        ZipEq::new(out_view, self).for_each(
            #[inline(always)]
            |Zip(mut out, item)| out.write(f(item)),
        );
        unsafe { out.set_dims(m, n) };
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, mat::Mat, unzipped, zipped, ComplexField};

    #[test]
    fn test_zip() {
        for (m, n) in [(2, 2), (4, 2), (2, 4)] {
            for rev_dst in [false, true] {
                for rev_src in [false, true] {
                    for transpose_dst in [false, true] {
                        for transpose_src in [false, true] {
                            for diag in [Diag::Include, Diag::Skip] {
                                let mut dst = Mat::from_fn(
                                    if transpose_dst { n } else { m },
                                    if transpose_dst { m } else { n },
                                    |_, _| f64::faer_zero(),
                                );
                                let src = Mat::from_fn(
                                    if transpose_src { n } else { m },
                                    if transpose_src { m } else { n },
                                    |_, _| f64::faer_one(),
                                );

                                let mut target = Mat::from_fn(m, n, |_, _| f64::faer_zero());
                                let target_src = Mat::from_fn(m, n, |_, _| f64::faer_one());

                                zipped!(target.as_mut(), target_src.as_ref())
                                    .for_each_triangular_lower(diag, |unzipped!(mut dst, src)| {
                                        dst.write(src.read())
                                    });

                                let mut dst = dst.as_mut();
                                let mut src = src.as_ref();

                                if transpose_dst {
                                    dst = dst.transpose_mut();
                                }
                                if rev_dst {
                                    dst = dst.reverse_rows_mut();
                                }

                                if transpose_src {
                                    src = src.transpose();
                                }
                                if rev_src {
                                    src = src.reverse_rows();
                                }

                                zipped!(dst.rb_mut(), src)
                                    .for_each_triangular_lower(diag, |unzipped!(mut dst, src)| {
                                        dst.write(src.read())
                                    });

                                assert!(dst.rb() == target.as_ref());
                            }
                        }
                    }
                }
            }
        }

        {
            let m = 3;
            for rev_dst in [false, true] {
                for rev_src in [false, true] {
                    let mut dst = Col::<f64>::zeros(m);
                    let src = Col::from_fn(m, |i| (i + 1) as f64);

                    let mut target = Col::<f64>::zeros(m);
                    let target_src =
                        Col::from_fn(m, |i| if rev_src { m - i } else { i + 1 } as f64);

                    zipped!(target.as_mut(), target_src.as_ref())
                        .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));

                    let mut dst = dst.as_mut();
                    let mut src = src.as_ref();

                    if rev_dst {
                        dst = dst.reverse_rows_mut();
                    }
                    if rev_src {
                        src = src.reverse_rows();
                    }

                    zipped!(dst.rb_mut(), src)
                        .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));

                    assert!(dst.rb() == target.as_ref());
                }
            }
        }

        {
            let m = 3;
            for rev_dst in [false, true] {
                for rev_src in [false, true] {
                    let mut dst = Row::<f64>::zeros(m);
                    let src = Row::from_fn(m, |i| (i + 1) as f64);

                    let mut target = Row::<f64>::zeros(m);
                    let target_src =
                        Row::from_fn(m, |i| if rev_src { m - i } else { i + 1 } as f64);

                    zipped!(target.as_mut(), target_src.as_ref())
                        .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));

                    let mut dst = dst.as_mut();
                    let mut src = src.as_ref();

                    if rev_dst {
                        dst = dst.reverse_cols_mut();
                    }
                    if rev_src {
                        src = src.reverse_cols();
                    }

                    zipped!(&mut dst, src)
                        .for_each(|unzipped!(mut dst, src)| dst.write(src.read()));

                    assert!(dst.rb() == target.as_ref());
                }
            }
        }
    }
}
