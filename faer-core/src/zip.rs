use crate::seal::Seal;
use crate::{ColMut, ColRef, MatMut, MatRef, RowMut, RowRef};
use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use reborrow::*;

pub trait CwiseMat<'short, Outlives = &'short Self>: Seal {
    type Item;

    fn transpose(self) -> Self;
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn is_col_major(&self) -> bool;
    fn is_row_major(&self) -> bool;
    unsafe fn get_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item;
    unsafe fn get_col_major_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item;
    unsafe fn get_row_major_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item;
}

pub trait CwiseCol<'short, Outlives = &'short Self>: Seal {
    type Item;

    fn nrows(&self) -> usize;
    fn is_contiguous(&self) -> bool;
    unsafe fn get_unchecked(&'short mut self, i: usize) -> Self::Item;
    unsafe fn get_contiguous_unchecked(&'short mut self, i: usize) -> Self::Item;
}

pub trait CwiseRow<'short, Outlives = &'short Self>: Seal {
    type Item;

    fn ncols(&self) -> usize;
    fn is_contiguous(&self) -> bool;
    unsafe fn get_unchecked(&'short mut self, i: usize) -> Self::Item;
    unsafe fn get_contiguous_unchecked(&'short mut self, i: usize) -> Self::Item;
}

// Simple wrapper indicating that values contained in this matrix may be uninitialized, and thus
// references to them shouldn't be created.
pub struct MatUninit<'a, T>(pub MatMut<'a, T>);

// Simple wrapper indicating that values contained in this column may be uninitialized, and thus
// references to them shouldn't be created.
pub struct ColUninit<'a, T>(pub ColMut<'a, T>);

// Simple wrapper indicating that values contained in this row may be uninitialized, and thus
// references to them shouldn't be created.
pub struct RowUninit<'a, T>(pub RowMut<'a, T>);

impl<'a, T> MatUninit<'a, T> {
    #[inline]
    pub fn cwise(self) -> ZipMat<(Self,)> {
        ZipMat { tuple: (self,) }
    }
}
impl<'a, T> ColUninit<'a, T> {
    #[inline]
    pub fn cwise(self) -> ZipCol<(Self,)> {
        ZipCol { tuple: (self,) }
    }
}
impl<'a, T> RowUninit<'a, T> {
    #[inline]
    pub fn cwise(self) -> ZipRow<(Self,)> {
        ZipRow { tuple: (self,) }
    }
}

impl<'a, T> crate::seal::Seal for MatUninit<'a, T> {}
impl<'a, T> crate::seal::Seal for ColUninit<'a, T> {}
impl<'a, T> crate::seal::Seal for RowUninit<'a, T> {}

impl<'short, 'a, T> CwiseMat<'short> for MatRef<'a, T> {
    type Item = &'short T;
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn is_col_major(&self) -> bool {
        self.row_stride() == 1
    }
    #[inline]
    fn is_row_major(&self) -> bool {
        self.col_stride() == 1
    }
    #[inline]
    unsafe fn get_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item {
        (*self).rb().get_unchecked(i, j)
    }
    #[inline]
    unsafe fn get_col_major_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item {
        &*(*self).rb().ptr_in_bounds_at_unchecked(0, j).add(i)
    }
    #[inline]
    unsafe fn get_row_major_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item {
        &*(*self).rb().ptr_in_bounds_at_unchecked(i, 0).add(j)
    }
    #[inline]
    fn transpose(self) -> Self {
        self.transpose()
    }
}

impl<'short, 'a, T> CwiseMat<'short> for MatMut<'a, T> {
    type Item = &'short mut T;
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn is_col_major(&self) -> bool {
        self.row_stride() == 1
    }
    #[inline]
    fn is_row_major(&self) -> bool {
        self.col_stride() == 1
    }
    #[inline]
    unsafe fn get_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item {
        (*self).rb_mut().get_unchecked(i, j)
    }
    #[inline]
    unsafe fn get_col_major_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item {
        &mut *(*self).rb_mut().ptr_in_bounds_at_unchecked(0, j).add(i)
    }
    #[inline]
    unsafe fn get_row_major_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item {
        &mut *(*self).rb_mut().ptr_in_bounds_at_unchecked(i, 0).add(j)
    }
    #[inline]
    fn transpose(self) -> Self {
        self.transpose()
    }
}

impl<'short, 'a, T> CwiseMat<'short> for MatUninit<'a, T> {
    type Item = *mut T;
    #[inline]
    fn nrows(&self) -> usize {
        self.0.nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.0.ncols()
    }
    #[inline]
    fn is_col_major(&self) -> bool {
        self.0.row_stride() == 1
    }
    #[inline]
    fn is_row_major(&self) -> bool {
        self.0.col_stride() == 1
    }
    #[inline]
    unsafe fn get_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item {
        self.0.rb_mut().ptr_in_bounds_at_unchecked(i, j)
    }
    #[inline]
    unsafe fn get_col_major_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item {
        self.0.rb_mut().ptr_in_bounds_at_unchecked(0, j).add(i)
    }
    #[inline]
    unsafe fn get_row_major_unchecked(&'short mut self, i: usize, j: usize) -> Self::Item {
        self.0.rb_mut().ptr_in_bounds_at_unchecked(i, 0).add(j)
    }
    #[inline]
    fn transpose(self) -> Self {
        MatUninit(self.0.transpose())
    }
}

impl<'short, 'a, T> CwiseCol<'short> for ColRef<'a, T> {
    type Item = &'short T;
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn is_contiguous(&self) -> bool {
        self.row_stride() == 1
    }
    #[inline]
    unsafe fn get_unchecked(&'short mut self, i: usize) -> Self::Item {
        (*self).rb().get_unchecked(i)
    }
    #[inline]
    unsafe fn get_contiguous_unchecked(&'short mut self, i: usize) -> Self::Item {
        &*(*self).rb().as_ptr().add(i)
    }
}

impl<'short, 'a, T> CwiseCol<'short> for ColMut<'a, T> {
    type Item = &'short mut T;
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn is_contiguous(&self) -> bool {
        self.row_stride() == 1
    }
    #[inline]
    unsafe fn get_unchecked(&'short mut self, i: usize) -> Self::Item {
        (*self).rb_mut().get_unchecked(i)
    }
    #[inline]
    unsafe fn get_contiguous_unchecked(&'short mut self, i: usize) -> Self::Item {
        &mut *(*self).rb_mut().as_ptr().add(i)
    }
}

impl<'short, 'a, T> CwiseCol<'short> for ColUninit<'a, T> {
    type Item = *mut T;
    #[inline]
    fn nrows(&self) -> usize {
        self.0.nrows()
    }
    #[inline]
    fn is_contiguous(&self) -> bool {
        self.0.row_stride() == 1
    }
    #[inline]
    unsafe fn get_unchecked(&'short mut self, i: usize) -> Self::Item {
        (*self).0.rb_mut().get_unchecked(i)
    }
    #[inline]
    unsafe fn get_contiguous_unchecked(&'short mut self, i: usize) -> Self::Item {
        (*self).0.rb_mut().as_ptr().add(i)
    }
}

impl<'short, 'a, T> CwiseRow<'short> for RowRef<'a, T> {
    type Item = &'short T;
    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn is_contiguous(&self) -> bool {
        self.col_stride() == 1
    }
    #[inline]
    unsafe fn get_unchecked(&'short mut self, i: usize) -> Self::Item {
        (*self).rb().get_unchecked(i)
    }
    #[inline]
    unsafe fn get_contiguous_unchecked(&'short mut self, i: usize) -> Self::Item {
        &*(*self).rb().as_ptr().add(i)
    }
}

impl<'short, 'a, T> CwiseRow<'short> for RowMut<'a, T> {
    type Item = &'short mut T;
    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn is_contiguous(&self) -> bool {
        self.col_stride() == 1
    }
    #[inline]
    unsafe fn get_unchecked(&'short mut self, i: usize) -> Self::Item {
        (*self).rb_mut().get_unchecked(i)
    }
    #[inline]
    unsafe fn get_contiguous_unchecked(&'short mut self, i: usize) -> Self::Item {
        &mut *(*self).rb_mut().as_ptr().add(i)
    }
}

impl<'short, 'a, T> CwiseRow<'short> for RowUninit<'a, T> {
    type Item = *mut T;
    #[inline]
    fn ncols(&self) -> usize {
        self.0.ncols()
    }
    #[inline]
    fn is_contiguous(&self) -> bool {
        self.0.col_stride() == 1
    }
    #[inline]
    unsafe fn get_unchecked(&'short mut self, i: usize) -> Self::Item {
        (*self).0.rb_mut().get_unchecked(i)
    }
    #[inline]
    unsafe fn get_contiguous_unchecked(&'short mut self, i: usize) -> Self::Item {
        (*self).0.rb_mut().as_ptr().add(i)
    }
}

pub struct ZipMat<Tuple> {
    pub(crate) tuple: Tuple,
}

pub struct ZipRow<Tuple> {
    pub(crate) tuple: Tuple,
}

pub struct ZipCol<Tuple> {
    pub(crate) tuple: Tuple,
}

include!(concat!(env!("OUT_DIR"), "/zip.rs"));
