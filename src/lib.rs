use assert2::{assert as fancy_assert, debug_assert as debug_fancy_assert};
use reborrow::{Reborrow, ReborrowMut};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;

struct MatrixSliceBase<T> {
    ptr: NonNull<T>,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
}
struct VecSliceBase<T> {
    ptr: NonNull<T>,
    len: usize,
    stride: isize,
}
impl<T> Copy for MatrixSliceBase<T> {}
impl<T> Clone for MatrixSliceBase<T> {
    fn clone(&self) -> Self {
        *self
    }
}
unsafe impl<T> Sync for MatrixSliceBase<T> {}
unsafe impl<T> Send for MatrixSliceBase<T> {}

impl<T> Copy for VecSliceBase<T> {}
impl<T> Clone for VecSliceBase<T> {
    fn clone(&self) -> Self {
        *self
    }
}
unsafe impl<T> Sync for VecSliceBase<T> {}
unsafe impl<T> Send for VecSliceBase<T> {}

/// 2D matrix view.
pub struct MatrixSlice<'a, T> {
    base: MatrixSliceBase<T>,
    _marker: PhantomData<&'a T>,
}

/// Mutable 2D matrix view.
pub struct MatrixSliceMut<'a, T> {
    base: MatrixSliceBase<T>,
    _marker: PhantomData<&'a mut T>,
}

/// Row vector view.
pub struct RowSlice<'a, T> {
    base: VecSliceBase<T>,
    _marker: PhantomData<&'a T>,
}

/// Mutable row vector view.
pub struct RowSliceMut<'a, T> {
    base: VecSliceBase<T>,
    _marker: PhantomData<&'a mut T>,
}

/// Column vector view.
pub struct ColSlice<'a, T> {
    base: VecSliceBase<T>,
    _marker: PhantomData<&'a T>,
}

/// Mutable column vector view.
pub struct ColSliceMut<'a, T> {
    base: VecSliceBase<T>,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> Copy for MatrixSlice<'a, T> {}
impl<'a, T> Copy for RowSlice<'a, T> {}
impl<'a, T> Copy for ColSlice<'a, T> {}

impl<'a, T> Clone for MatrixSlice<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Clone for RowSlice<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Clone for ColSlice<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'b, 'a, T> Reborrow<'b> for MatrixSlice<'a, T> {
    type Target = MatrixSlice<'b, T>;
    fn rb(&'b self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for MatrixSlice<'a, T> {
    type Target = MatrixSlice<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        *self
    }
}

impl<'b, 'a, T> Reborrow<'b> for MatrixSliceMut<'a, T> {
    type Target = MatrixSlice<'b, T>;
    fn rb(&'b self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for MatrixSliceMut<'a, T> {
    type Target = MatrixSliceMut<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'b, 'a, T> Reborrow<'b> for RowSlice<'a, T> {
    type Target = RowSlice<'b, T>;
    fn rb(&'b self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for RowSlice<'a, T> {
    type Target = RowSlice<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        *self
    }
}

impl<'b, 'a, T> Reborrow<'b> for RowSliceMut<'a, T> {
    type Target = RowSlice<'b, T>;
    fn rb(&'b self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for RowSliceMut<'a, T> {
    type Target = RowSliceMut<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'b, 'a, T> Reborrow<'b> for ColSlice<'a, T> {
    type Target = ColSlice<'b, T>;
    fn rb(&'b self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for ColSlice<'a, T> {
    type Target = ColSlice<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        *self
    }
}

impl<'b, 'a, T> Reborrow<'b> for ColSliceMut<'a, T> {
    type Target = ColSlice<'b, T>;
    fn rb(&'b self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for ColSliceMut<'a, T> {
    type Target = ColSliceMut<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        Self::Target {
            base: self.base,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> MatrixSlice<'a, T> {
    /// Returns a matrix slice from the given arguments.  
    /// `ptr`: pointer to the first element of the matrix.  
    /// `nrows`: number of rows of the matrix.  
    /// `ncols`: number of columns of the matrix.  
    /// `row_stride`: offset between the first elements of two successive rows in the matrix.
    /// `col_stride`: offset between the first elements of two successive columns in the matrix.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `i < nrows` and `j < ncols`,  
    /// `ptr.offset(i as isize * row_stride + j as isize * col_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never read.  
    /// The referenced memory must not be mutated during the lifetime `'a`.
    pub unsafe fn from_raw_parts(
        ptr: *const T,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            base: MatrixSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr as *mut T),
                nrows,
                ncols,
                row_stride,
                col_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a pointer to the first element of the matrix.
    pub fn as_ptr(self) -> *const T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the matrix.
    pub fn nrows(&self) -> usize {
        self.base.nrows
    }

    /// Returns the number of columns of the matrix.
    pub fn ncols(&self) -> usize {
        self.base.ncols
    }

    /// Returns the offset between the first elements of two successive rows in the matrix.
    pub fn row_stride(&self) -> isize {
        self.base.row_stride
    }

    /// Returns the offset between the first elements of two successive columns in the matrix.
    pub fn col_stride(&self) -> isize {
        self.base.col_stride
    }

    /// Returns a pointer to the element at position (i, j) in the matrix.
    pub fn ptr_at(self, i: usize, j: usize) -> *const T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(i as isize * self.row_stride())
            .wrapping_offset(j as isize * self.col_stride())
    }

    /// Returns a pointer to the element at position (i, j) in the matrix, assuming it falls within
    /// its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `i < self.nrows()`
    /// and `j < self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn ptr_in_bounds_at_unchecked(self, i: usize, j: usize) -> *const T {
        debug_fancy_assert!(i < self.nrows());
        debug_fancy_assert!(j < self.ncols());
        self.base
            .ptr
            .as_ptr()
            .offset(i as isize * self.row_stride())
            .offset(j as isize * self.col_stride())
    }

    /// Returns a pointer to the element at position (i, j) in the matrix, while asserting that
    /// it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that `i < self.nrows()`
    /// and `j < self.ncols()`. Otherwise, it panics.
    pub fn ptr_in_bounds_at(self, i: usize, j: usize) -> *const T {
        fancy_assert!(i < self.nrows());
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(i, j) }
    }

    /// Splits the matrix into four corner parts in the following order: top left, top right,
    /// bottom left, bottom right.
    ///
    /// # Safety
    ///
    /// Requires that `i <= self.nrows()`
    /// and `j <= self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn split_at_unchecked(self, i: usize, j: usize) -> (Self, Self, Self, Self) {
        debug_fancy_assert!(i <= self.nrows());
        debug_fancy_assert!(j <= self.ncols());
        let ptr = self.base.ptr.as_ptr();
        let cs = self.col_stride();
        let rs = self.row_stride();
        (
            Self::from_raw_parts(ptr, i, j, rs, cs),
            Self::from_raw_parts(
                ptr.wrapping_offset(j as isize * cs),
                i,
                self.ncols() - j,
                rs,
                cs,
            ),
            Self::from_raw_parts(
                ptr.wrapping_offset(i as isize * rs),
                self.nrows() - i,
                j,
                rs,
                cs,
            ),
            Self::from_raw_parts(
                ptr.wrapping_offset(i as isize * rs)
                    .wrapping_offset(j as isize * cs),
                self.nrows() - i,
                self.ncols() - j,
                rs,
                cs,
            ),
        )
    }

    /// Splits the matrix into four corner parts in the following order: top left, top right,
    /// bottom left, bottom right.
    ///
    /// # Panics
    ///
    /// Requires that `i <= self.nrows()`
    /// and `j <= self.ncols()`. Otherwise, it panics.
    pub fn split_at(self, i: usize, j: usize) -> (Self, Self, Self, Self) {
        fancy_assert!(i <= self.nrows());
        fancy_assert!(j <= self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(i, j) }
    }

    /// Returns a reference to the element at position (i, j), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `i < self.nrows()`
    /// and `j < self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn get_unchecked(self, i: usize, j: usize) -> &'a T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &*self.ptr_in_bounds_at_unchecked(i, j)
    }

    /// Returns a reference to the element at position (i, j), or `None` if the indices are out of
    /// bounds.
    pub fn get(self, i: usize, j: usize) -> Option<&'a T> {
        if i < self.nrows() && j < self.ncols() {
            // SAFETY: bounds have been checked.
            unsafe { Some(self.get_unchecked(i, j)) }
        } else {
            None
        }
    }

    /// Returns the `i`-th row of the matrix, with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `i < self.nrows()`. Otherwise, the behavior is undefined.
    pub unsafe fn row_unchecked(self, i: usize) -> RowSlice<'a, T> {
        debug_fancy_assert!(i < self.nrows());
        let ncols = self.ncols();
        let cs = self.col_stride();
        RowSlice::from_raw_parts(self.ptr_at(i, 0), ncols, cs)
    }

    /// Returns the `i`-th row of the matrix.
    ///
    /// # Panics
    ///
    /// Requires that `i < self.nrows()`. Otherwise, it panics.
    pub fn row(self, i: usize) -> RowSlice<'a, T> {
        fancy_assert!(i < self.nrows());
        // SAFETY: bounds have been checked
        unsafe { self.row_unchecked(i) }
    }

    /// Returns the `j`-th column of the matrix, with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `j < self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn col_unchecked(self, j: usize) -> ColSlice<'a, T> {
        debug_fancy_assert!(j < self.ncols());
        let nrows = self.nrows();
        let rs = self.row_stride();
        ColSlice::from_raw_parts(self.ptr_at(0, j), nrows, rs)
    }

    /// Returns the `j`-th column of the matrix.
    ///
    /// # Panics
    ///
    /// Requires that `j < self.ncols()`. Otherwise, it panics.
    pub fn col(self, j: usize) -> ColSlice<'a, T> {
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked.
        unsafe { self.col_unchecked(j) }
    }

    /// Returns the transpose of `self`.
    pub fn t(self) -> MatrixSlice<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe {
            MatrixSlice::from_raw_parts(
                ptr,
                self.ncols(),
                self.nrows(),
                self.col_stride(),
                self.row_stride(),
            )
        }
    }

    /// Returns an iterator over the rows of the matrix.
    pub fn into_row_iter(self) -> RowIter<'a, T> {
        RowIter(self)
    }

    /// Returns an iterator over the columns of the matrix.
    pub fn into_col_iter(self) -> ColIter<'a, T> {
        ColIter(self)
    }

    /// Returns a view over a submatrix of `self`, starting at position `(i, j)`
    /// with dimensions `(nrows, ncols)`.
    ///
    /// # Safety
    ///
    /// Requires that `i <= self.nrows()`,  
    /// `j <= self.ncols()`,  
    /// `nrows <= self.nrows() - i`  
    /// and `ncols <= self.ncols() - j`. Otherwise, the behavior is undefined.
    pub unsafe fn submatrix_unchecked(
        self,
        i: usize,
        j: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        debug_fancy_assert!(i <= self.nrows());
        debug_fancy_assert!(j <= self.ncols());
        debug_fancy_assert!(nrows <= self.nrows() - i);
        debug_fancy_assert!(ncols <= self.ncols() - j);
        Self::from_raw_parts(
            self.rb().ptr_at(i, j),
            nrows,
            ncols,
            self.row_stride(),
            self.col_stride(),
        )
    }

    /// Returns a view over a submatrix of `self`, starting at position `(i, j)`
    /// with dimensions `(nrows, ncols)`.
    ///
    /// # Panics
    ///
    /// Requires that `i <= self.nrows()`,  
    /// `j <= self.ncols()`,  
    /// `nrows <= self.nrows() - i`  
    /// and `ncols <= self.ncols() - j`. Otherwise, it panics.
    pub fn submatrix(self, i: usize, j: usize, nrows: usize, ncols: usize) -> Self {
        fancy_assert!(i <= self.nrows());
        fancy_assert!(j <= self.ncols());
        fancy_assert!(nrows <= self.nrows() - i);
        fancy_assert!(ncols <= self.ncols() - j);
        unsafe { self.submatrix_unchecked(i, j, nrows, ncols) }
    }
}

impl<'a, T> MatrixSliceMut<'a, T> {
    /// Returns a mutable matrix slice from the given arguments.  
    /// `ptr`: pointer to the first element of the matrix.  
    /// `nrows`: number of rows of the matrix.  
    /// `ncols`: number of columns of the matrix.  
    /// `row_stride`: offset between the first elements of two successive rows in the matrix.
    /// `col_stride`: offset between the first elements of two successive columns in the matrix.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `i < nrows` and `j < ncols`,  
    /// `ptr.offset(i as isize * row_stride + j as isize * col_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never read.
    /// Additionally, when `(i, j) != (0, 0)`, this pointer is never equal to `ptr` (no self
    /// aliasing).  
    /// The referenced memory must not be accessed by another pointer which was not derived from
    /// the return value, during the lifetime `'a`.
    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            base: MatrixSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr),
                nrows,
                ncols,
                row_stride,
                col_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a mutable pointer to the first element of the matrix.
    pub fn as_ptr(self) -> *mut T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the matrix.
    pub fn nrows(&self) -> usize {
        self.base.nrows
    }

    /// Returns the number of columns of the matrix.
    pub fn ncols(&self) -> usize {
        self.base.ncols
    }

    /// Returns the offset between the first elements of two successive rows in the matrix.
    pub fn row_stride(&self) -> isize {
        self.base.row_stride
    }

    /// Returns the offset between the first elements of two successive columns in the matrix.
    pub fn col_stride(&self) -> isize {
        self.base.col_stride
    }

    /// Returns an immutable matrix view over the same data.
    pub fn as_const(self) -> MatrixSlice<'a, T> {
        MatrixSlice::<'a, T> {
            base: self.base,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable pointer to the element at position (i, j) in the matrix.
    pub fn ptr_at(self, i: usize, j: usize) -> *mut T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(i as isize * self.row_stride())
            .wrapping_offset(j as isize * self.col_stride())
    }

    /// Returns a mutable pointer to the element at position (i, j) in the matrix, assuming it falls
    /// within its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `i < self.nrows()`
    /// and `j < self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn ptr_in_bounds_at_unchecked(self, i: usize, j: usize) -> *mut T {
        debug_fancy_assert!(i < self.nrows());
        debug_fancy_assert!(j < self.ncols());
        self.base
            .ptr
            .as_ptr()
            .offset(i as isize * self.row_stride())
            .offset(j as isize * self.col_stride())
    }

    /// Returns a mutable pointer to the element at position (i, j) in the matrix, while asserting that
    /// it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that `i < self.nrows()`
    /// and `j < self.ncols()`. Otherwise, it panics.
    pub fn ptr_in_bounds_at(self, i: usize, j: usize) -> *mut T {
        fancy_assert!(i < self.nrows());
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(i, j) }
    }

    /// Splits the matrix into four corner parts in the following order: top left, top right,
    /// bottom left, bottom right.
    ///
    /// # Safety
    ///
    /// Requires that `i <= self.nrows()`
    /// and `j <= self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn split_at_unchecked(self, i: usize, j: usize) -> (Self, Self, Self, Self) {
        debug_fancy_assert!(i <= self.nrows());
        debug_fancy_assert!(j <= self.ncols());
        let ptr = self.base.ptr.as_ptr();
        let cs = self.col_stride();
        let rs = self.row_stride();
        (
            Self::from_raw_parts(ptr, i, j, rs, cs),
            Self::from_raw_parts(
                ptr.wrapping_offset(j as isize * cs),
                i,
                self.ncols() - j,
                rs,
                cs,
            ),
            Self::from_raw_parts(
                ptr.wrapping_offset(i as isize * rs),
                self.nrows() - i,
                j,
                rs,
                cs,
            ),
            Self::from_raw_parts(
                ptr.wrapping_offset(i as isize * rs)
                    .wrapping_offset(j as isize * cs),
                self.nrows() - i,
                self.ncols() - j,
                rs,
                cs,
            ),
        )
    }

    /// Splits the matrix into four corner parts in the following order: top left, top right,
    /// bottom left, bottom right.
    ///
    /// # Panics
    ///
    /// Requires that `i <= self.nrows()`
    /// and `j <= self.ncols()`. Otherwise, it panics.
    pub fn split_at(self, i: usize, j: usize) -> (Self, Self, Self, Self) {
        fancy_assert!(i <= self.nrows());
        fancy_assert!(j <= self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(i, j) }
    }

    /// Returns a mutable reference to the element at position (i, j), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `i < self.nrows()`
    /// and `j < self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn get_unchecked(self, i: usize, j: usize) -> &'a mut T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &mut *self.ptr_in_bounds_at_unchecked(i, j)
    }

    /// Returns a mutable reference to the element at position (i, j), or `None` if the indices are
    /// out of bounds.
    pub fn get(self, i: usize, j: usize) -> Option<&'a mut T> {
        if i < self.nrows() && j < self.ncols() {
            // SAFETY: bounds have been checked.
            unsafe { Some(self.get_unchecked(i, j)) }
        } else {
            None
        }
    }

    /// Returns the `i`-th row of the matrix, with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `i < self.nrows()`. Otherwise, the behavior is undefined.
    pub unsafe fn row_unchecked(self, i: usize) -> RowSliceMut<'a, T> {
        debug_fancy_assert!(i < self.nrows());
        let ncols = self.ncols();
        let cs = self.col_stride();
        RowSliceMut::from_raw_parts(self.ptr_at(i, 0), ncols, cs)
    }

    /// Returns the `i`-th row of the matrix.
    ///
    /// # Panics
    ///
    /// Requires that `i < self.nrows()`. Otherwise, it panics.
    pub fn row(self, i: usize) -> RowSliceMut<'a, T> {
        fancy_assert!(i < self.nrows());
        // SAFETY: bounds have been checked.
        unsafe { self.row_unchecked(i) }
    }

    /// Returns the `j`-th column of the matrix, with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `j < self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn col_unchecked(self, j: usize) -> ColSliceMut<'a, T> {
        debug_fancy_assert!(j < self.ncols());
        let nrows = self.nrows();
        let rs = self.row_stride();
        ColSliceMut::from_raw_parts(self.ptr_at(0, j), nrows, rs)
    }

    /// Returns the `j`-th column of the matrix.
    ///
    /// # Panics
    ///
    /// Requires that `j < self.ncols()`. Otherwise, it panics.
    pub fn col(self, j: usize) -> ColSliceMut<'a, T> {
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked.
        unsafe { self.col_unchecked(j) }
    }

    /// Returns the transpose of `self`.
    pub fn t(self) -> MatrixSliceMut<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe {
            MatrixSliceMut::from_raw_parts(
                ptr,
                self.ncols(),
                self.nrows(),
                self.col_stride(),
                self.row_stride(),
            )
        }
    }

    /// Returns an iterator over the rows of the matrix.
    pub fn into_row_iter(self) -> RowIterMut<'a, T> {
        RowIterMut(self)
    }

    /// Returns an iterator over the columns of the matrix.
    pub fn into_col_iter(self) -> ColIterMut<'a, T> {
        ColIterMut(self)
    }

    /// Returns a view over a submatrix of `self`, starting at position `(i, j)`
    /// with dimensions `(nrows, ncols)`.
    ///
    /// # Safety
    ///
    /// Requires that `i <= self.nrows()`,  
    /// `j <= self.ncols()`,  
    /// `nrows <= self.nrows() - i`  
    /// and `ncols <= self.ncols() - j`. Otherwise, the behavior is undefined.
    pub unsafe fn submatrix_unchecked(
        self,
        i: usize,
        j: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        debug_fancy_assert!(i <= self.nrows());
        debug_fancy_assert!(j <= self.ncols());
        debug_fancy_assert!(nrows <= self.nrows() - i);
        debug_fancy_assert!(ncols <= self.ncols() - j);

        let mut s = self;
        Self::from_raw_parts(
            s.rb_mut().ptr_at(i, j),
            nrows,
            ncols,
            s.row_stride(),
            s.col_stride(),
        )
    }

    /// Returns a view over a submatrix of `self`, starting at position `(i, j)`
    /// with dimensions `(nrows, ncols)`.
    ///
    /// # Panics
    ///
    /// Requires that `i <= self.nrows()`,  
    /// `j <= self.ncols()`,  
    /// `nrows <= self.nrows() - i`  
    /// and `ncols <= self.ncols() - j`. Otherwise, it panics.
    pub fn submatrix(self, i: usize, j: usize, nrows: usize, ncols: usize) -> Self {
        fancy_assert!(i <= self.nrows());
        fancy_assert!(j <= self.ncols());
        fancy_assert!(nrows <= self.nrows() - i);
        fancy_assert!(ncols <= self.ncols() - j);
        unsafe { self.submatrix_unchecked(i, j, nrows, ncols) }
    }
}

impl<'a, T> RowSlice<'a, T> {
    /// Returns a row vector slice from the given arguments.  
    /// `ptr`: pointer to the first element of the row vector.  
    /// `ncols`: number of columns of the row vector.  
    /// `col_stride`: offset between the first elements of two successive columns in the row vector.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `j < ncols`,  
    /// `ptr.offset(j as isize * col_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never read.  
    /// The referenced memory must not be mutated during the lifetime `'a`.
    pub unsafe fn from_raw_parts(ptr: *const T, ncols: usize, col_stride: isize) -> Self {
        Self {
            base: VecSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr as *mut T),
                len: ncols,
                stride: col_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a pointer to the first element of the row vector.
    pub fn as_ptr(self) -> *const T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the row vector. Always returns `1`.
    pub fn nrows(&self) -> usize {
        1
    }

    /// Returns the number of columns of the row vector.
    pub fn ncols(&self) -> usize {
        self.base.len
    }

    /// Returns the offset between the first elements of two successive columns in the row vector.
    pub fn col_stride(&self) -> isize {
        self.base.stride
    }

    /// Returns a pointer to the element at position (0, j) in the row vector.
    pub fn ptr_at(self, j: usize) -> *const T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(j as isize * self.col_stride())
    }

    /// Returns a pointer to the element at position (0, j) in the row vector, assuming it falls within
    /// its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `j < self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn ptr_in_bounds_at_unchecked(self, j: usize) -> *const T {
        debug_fancy_assert!(j < self.ncols());
        self.base
            .ptr
            .as_ptr()
            .offset(j as isize * self.col_stride())
    }

    /// Returns a pointer to the element at position (0, j) in the row vector, while asserting that
    /// it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that `j < self.ncols()`. Otherwise, it panics.
    pub fn ptr_in_bounds_at(self, j: usize) -> *const T {
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(j) }
    }

    /// Splits the row vector into two parts in the following order: left, right.
    ///
    /// # Safety
    ///
    /// Requires that `j <= self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn split_at_unchecked(self, j: usize) -> (Self, Self) {
        debug_fancy_assert!(j <= self.ncols());
        let ptr = self.base.ptr.as_ptr();
        let cs = self.col_stride();
        (
            Self::from_raw_parts(ptr, j, cs),
            Self::from_raw_parts(ptr.wrapping_offset(j as isize * cs), self.ncols() - j, cs),
        )
    }

    /// Splits the row vector into two parts in the following order: left, right.
    ///
    /// # Panics
    ///
    /// Requires that `j <= self.ncols()`. Otherwise, it panics.
    pub fn split_at(self, j: usize) -> (Self, Self) {
        fancy_assert!(j <= self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(j) }
    }

    /// Returns a reference to the element at position (0, j), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires `j < self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn get_unchecked(self, j: usize) -> &'a T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &*self.ptr_in_bounds_at_unchecked(j)
    }

    /// Returns a reference to the element at position (0, j), or `None` if the index is out of
    /// bounds.
    pub fn get(self, j: usize) -> Option<&'a T> {
        if j < self.ncols() {
            // SAFETY: bounds have been checked.
            unsafe { Some(self.get_unchecked(j)) }
        } else {
            None
        }
    }

    /// Returns an equivalent 2D matrix view over the same data.
    pub fn as_2d(self) -> MatrixSlice<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe {
            MatrixSlice::from_raw_parts(ptr, self.nrows(), self.ncols(), 0, self.col_stride())
        }
    }

    /// Returns the transpose of `self`.
    pub fn t(self) -> ColSlice<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { ColSlice::from_raw_parts(ptr, self.ncols(), self.col_stride()) }
    }
}

impl<'a, T> RowSliceMut<'a, T> {
    /// Returns a mutable row vector slice from the given arguments.  
    /// `ptr`: pointer to the first element of the row vector.  
    /// `ncols`: number of columns of the row vector.  
    /// `col_stride`: offset between the first elements of two successive columns in the row vector.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `j < ncols`,  
    /// `ptr.offset(j as isize * col_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never read.  
    /// Additionally, when `j != 0`, this pointer is never equal to `ptr` (no self aliasing).  
    /// The referenced memory must not be accessed by another pointer which was not derived from
    /// the return value, during the lifetime `'a`.
    pub unsafe fn from_raw_parts(ptr: *mut T, ncols: usize, col_stride: isize) -> Self {
        Self {
            base: VecSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr),
                len: ncols,
                stride: col_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a mutable pointer to the first element of the row vector.
    pub fn as_ptr(self) -> *mut T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the row vector. Always returns `1`.
    pub fn nrows(&self) -> usize {
        1
    }

    /// Returns the number of columns of the row vector.
    pub fn ncols(&self) -> usize {
        self.base.len
    }

    /// Returns the offset between the first elements of two successive columns in the row vector.
    pub fn col_stride(&self) -> isize {
        self.base.stride
    }

    /// Returns an immutable row vector view over the same data.
    pub fn as_const(self) -> RowSlice<'a, T> {
        RowSlice::<'a, T> {
            base: self.base,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable pointer to the element at position (0, j) in the row vector.
    pub fn ptr_at(self, j: usize) -> *mut T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(j as isize * self.col_stride())
    }

    /// Returns a mutable pointer to the element at position (0, j) in the row vector, assuming it
    /// falls within its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `j < self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn ptr_in_bounds_at_unchecked(self, j: usize) -> *mut T {
        debug_fancy_assert!(j < self.ncols());
        self.base
            .ptr
            .as_ptr()
            .offset(j as isize * self.col_stride())
    }

    /// Returns a mutable pointer to the element at position (0, j) in the row vector, while
    /// asserting that it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that `j < self.ncols()`. Otherwise, it panics.
    pub fn ptr_in_bounds_at(self, j: usize) -> *mut T {
        fancy_assert!(j < self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(j) }
    }

    /// Splits the row vector into two parts in the following order: left, right.
    ///
    /// # Safety
    ///
    /// Requires that `j <= self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn split_at_unchecked(self, j: usize) -> (Self, Self) {
        debug_fancy_assert!(j <= self.ncols());
        let ptr = self.base.ptr.as_ptr();
        let cs = self.col_stride();
        (
            Self::from_raw_parts(ptr, j, cs),
            Self::from_raw_parts(ptr.wrapping_offset(j as isize * cs), self.ncols() - j, cs),
        )
    }

    /// Splits the row vector into two parts in the following order: left, right.
    ///
    /// # Panics
    ///
    /// Requires that `j <= self.ncols()`. Otherwise, it panics.
    pub fn split_at(self, j: usize) -> (Self, Self) {
        fancy_assert!(j <= self.ncols());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(j) }
    }

    /// Returns a mutable reference to the element at position (0, j), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires `j < self.ncols()`. Otherwise, the behavior is undefined.
    pub unsafe fn get_unchecked(self, j: usize) -> &'a mut T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &mut *self.ptr_in_bounds_at_unchecked(j)
    }

    /// Returns a mutable reference to the element at position (0, j), or `None` if the index is
    /// out of bounds.
    pub fn get(self, j: usize) -> Option<&'a mut T> {
        if j < self.ncols() {
            // SAFETY: bounds have been checked.
            unsafe { Some(self.get_unchecked(j)) }
        } else {
            None
        }
    }

    /// Returns an equivalent 2D matrix view over the same data.
    pub fn as_2d(self) -> MatrixSliceMut<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe {
            MatrixSliceMut::from_raw_parts(ptr, self.nrows(), self.ncols(), 0, self.col_stride())
        }
    }

    /// Returns the transpose of `self`.
    pub fn t(self) -> ColSliceMut<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { ColSliceMut::from_raw_parts(ptr, self.ncols(), self.col_stride()) }
    }
}

impl<'a, T> ColSlice<'a, T> {
    /// Returns a column vector slice from the given arguments.  
    /// `ptr`: pointer to the first element of the column vector.  
    /// `ncols`: number of columns of the column vector.  
    /// `col_stride`: offset between the first elements of two successive columns in the column vector.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `i < nrows`,  
    /// `ptr.offset(i as isize * row_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never read.  
    /// The referenced memory must not be mutated during the lifetime `'a`.
    pub unsafe fn from_raw_parts(ptr: *const T, nrows: usize, row_stride: isize) -> Self {
        Self {
            base: VecSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr as *mut T),
                len: nrows,
                stride: row_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a pointer to the first element of the column vector.
    pub fn as_ptr(self) -> *const T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the column vector.
    pub fn nrows(&self) -> usize {
        self.base.len
    }

    /// Returns the number of columns of the column vector. Always returns `1`.
    pub fn ncols(&self) -> usize {
        1
    }

    /// Returns the offset between the first elements of two successive rows in the column vector.
    pub fn row_stride(&self) -> isize {
        self.base.stride
    }

    /// Returns a pointer to the element at position (i, 0) in the column vector.
    pub fn ptr_at(self, i: usize) -> *const T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(i as isize * self.row_stride())
    }

    /// Returns a pointer to the element at position (i, 0) in the column vector, assuming it falls within
    /// its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `i < self.nrows()`. Otherwise, the behavior is undefined.
    pub unsafe fn ptr_in_bounds_at_unchecked(self, i: usize) -> *const T {
        debug_fancy_assert!(i < self.nrows());
        self.base
            .ptr
            .as_ptr()
            .offset(i as isize * self.row_stride())
    }

    /// Returns a pointer to the element at position (i, 0) in the column vector, while asserting that
    /// it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that `i < self.nrows()`. Otherwise, it panics.
    pub fn ptr_in_bounds_at(self, i: usize) -> *const T {
        fancy_assert!(i < self.nrows());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(i) }
    }

    /// Splits the column vector into two parts in the following order: top, bottom.
    ///
    /// # Safety
    ///
    /// Requires that `i <= self.nrows()`. Otherwise, the behavior is undefined.
    pub unsafe fn split_at_unchecked(self, i: usize) -> (Self, Self) {
        debug_fancy_assert!(i <= self.nrows());
        let ptr = self.base.ptr.as_ptr();
        let rs = self.row_stride();
        (
            Self::from_raw_parts(ptr, i, rs),
            Self::from_raw_parts(ptr.wrapping_offset(i as isize * rs), self.nrows() - i, rs),
        )
    }

    /// Splits the column vector into two parts in the following order: top, bottom.
    ///
    /// # Panics
    ///
    /// Requires that `i <= self.nrows()`. Otherwise, it panics.
    pub fn split_at(self, i: usize) -> (Self, Self) {
        fancy_assert!(i <= self.nrows());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(i) }
    }

    /// Returns a reference to the element at position (i, 0), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires `i < self.nrows()`. Otherwise, the behavior is undefined.
    pub unsafe fn get_unchecked(self, i: usize) -> &'a T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &*self.ptr_in_bounds_at_unchecked(i)
    }

    /// Returns a reference to the element at position (i, 0), or `None` if the index is out of
    /// bounds.
    pub fn get(self, i: usize) -> Option<&'a T> {
        if i < self.nrows() {
            // SAFETY: bounds have been checked.
            unsafe { Some(self.get_unchecked(i)) }
        } else {
            None
        }
    }

    /// Returns an equivalent 2D matrix view over the same data.
    pub fn as_2d(self) -> MatrixSlice<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe {
            MatrixSlice::from_raw_parts(ptr, self.nrows(), self.ncols(), self.row_stride(), 0)
        }
    }

    /// Returns the transpose of `self`.
    pub fn t(self) -> RowSlice<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { RowSlice::from_raw_parts(ptr, self.nrows(), self.row_stride()) }
    }
}

impl<'a, T> ColSliceMut<'a, T> {
    /// Returns a mutable column vector slice from the given arguments.  
    /// `ptr`: pointer to the first element of the column vector.  
    /// `ncols`: number of columns of the column vector.  
    /// `col_stride`: offset between the first elements of two successive columns in the column vector.
    ///
    /// # Safety
    ///
    /// `ptr` must be non null and properly aligned for type `T`.  
    /// For each `i < nrows`,  
    /// `ptr.offset(i as isize * row_stride)` must point to a valid
    /// initialized object of type `T`, unless memory pointing to that address is never read.  
    /// Additionally, when `i != 0`, this pointer is never equal to `ptr` (no self aliasing).  
    /// The referenced memory must not be mutated during the lifetime `'a`.
    pub unsafe fn from_raw_parts(ptr: *mut T, nrows: usize, row_stride: isize) -> Self {
        Self {
            base: VecSliceBase::<T> {
                ptr: NonNull::new_unchecked(ptr),
                len: nrows,
                stride: row_stride,
            },
            _marker: PhantomData,
        }
    }

    /// Returns a mutable pointer to the first element of the column vector.
    pub fn as_ptr(self) -> *mut T {
        self.base.ptr.as_ptr()
    }

    /// Returns the number of rows of the column vector.
    pub fn nrows(&self) -> usize {
        self.base.len
    }

    /// Returns the number of columns of the column vector. Always returns `1`.
    pub fn ncols(&self) -> usize {
        1
    }

    /// Returns the offset between the first elements of two successive rows in the column vector.
    pub fn row_stride(&self) -> isize {
        self.base.stride
    }

    /// Returns an immutable column vector view over the same data.
    pub fn as_const(self) -> ColSlice<'a, T> {
        ColSlice::<'a, T> {
            base: self.base,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable pointer to the element at position (i, 0) in the column vector.
    pub fn ptr_at(self, i: usize) -> *mut T {
        self.base
            .ptr
            .as_ptr()
            .wrapping_offset(i as isize * self.row_stride())
    }

    /// Returns a mutable pointer to the element at position (i, 0) in the column vector,
    /// assuming it falls within its bounds with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires that `i < self.nrows()`. Otherwise, the behavior is undefined.
    pub unsafe fn ptr_in_bounds_at_unchecked(self, i: usize) -> *mut T {
        debug_fancy_assert!(i < self.nrows());
        self.base
            .ptr
            .as_ptr()
            .offset(i as isize * self.row_stride())
    }

    /// Returns a mutable pointer to the element at position (i, 0) in the column vector,
    /// while asserting that it falls within its bounds.
    ///
    /// # Panics
    ///
    /// Requires that `i < self.nrows()`. Otherwise, it panics.
    pub fn ptr_in_bounds_at(self, i: usize) -> *mut T {
        fancy_assert!(i < self.nrows());
        // SAFETY: bounds have been checked
        unsafe { self.ptr_in_bounds_at_unchecked(i) }
    }

    /// Splits the column vector into two parts in the following order: top, bottom.
    ///
    /// # Safety
    ///
    /// Requires that `i <= self.nrows()`. Otherwise, the behavior is undefined.
    pub unsafe fn split_at_unchecked(self, i: usize) -> (Self, Self) {
        debug_fancy_assert!(i <= self.nrows());
        let ptr = self.base.ptr.as_ptr();
        let rs = self.row_stride();
        (
            Self::from_raw_parts(ptr, i, rs),
            Self::from_raw_parts(ptr.wrapping_offset(i as isize * rs), self.nrows() - i, rs),
        )
    }

    /// Splits the column vector into two parts in the following order: top, bottom.
    ///
    /// # Panics
    ///
    /// Requires that `i <= self.nrows()`. Otherwise, it panics.
    pub fn split_at(self, i: usize) -> (Self, Self) {
        fancy_assert!(i <= self.nrows());
        // SAFETY: bounds have been checked
        unsafe { self.split_at_unchecked(i) }
    }

    /// Returns a mutable reference to the element at position (i, 0), with no bound checks.
    ///
    /// # Safety
    ///
    /// Requires `i < self.nrows()`. Otherwise, the behavior is undefined.
    pub unsafe fn get_unchecked(self, i: usize) -> &'a mut T {
        // SAFETY: same preconditions. And we can dereference this pointer because it lives as
        // long as the underlying data.
        &mut *self.ptr_in_bounds_at_unchecked(i)
    }

    /// Returns a mutable reference to the element at position (i, 0), or `None` if the index is
    /// out of bounds.
    pub fn get(self, i: usize) -> Option<&'a mut T> {
        if i < self.nrows() {
            // SAFETY: bounds have been checked.
            unsafe { Some(self.get_unchecked(i)) }
        } else {
            None
        }
    }

    /// Returns an equivalent 2D matrix view over the same data.
    pub fn as_2d(self) -> MatrixSliceMut<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe {
            MatrixSliceMut::from_raw_parts(ptr, self.nrows(), self.ncols(), self.row_stride(), 0)
        }
    }

    /// Returns the transpose of `self`.
    pub fn t(self) -> RowSliceMut<'a, T> {
        let ptr = self.base.ptr.as_ptr();
        unsafe { RowSliceMut::from_raw_parts(ptr, self.nrows(), self.row_stride()) }
    }
}

impl<'a, T> Index<(usize, usize)> for MatrixSlice<'a, T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        self.get(i, j).unwrap()
    }
}
impl<'a, T> Index<(usize, usize)> for MatrixSliceMut<'a, T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        self.rb().get(i, j).unwrap()
    }
}
impl<'a, T> IndexMut<(usize, usize)> for MatrixSliceMut<'a, T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        self.rb_mut().get(i, j).unwrap()
    }
}

impl<'a, T> Index<usize> for RowSlice<'a, T> {
    type Output = T;

    fn index(&self, j: usize) -> &Self::Output {
        self.get(j).unwrap()
    }
}
impl<'a, T> Index<usize> for RowSliceMut<'a, T> {
    type Output = T;

    fn index(&self, j: usize) -> &Self::Output {
        self.rb().get(j).unwrap()
    }
}
impl<'a, T> IndexMut<usize> for RowSliceMut<'a, T> {
    fn index_mut(&mut self, j: usize) -> &mut Self::Output {
        self.rb_mut().get(j).unwrap()
    }
}

impl<'a, T> Index<usize> for ColSlice<'a, T> {
    type Output = T;

    fn index(&self, j: usize) -> &Self::Output {
        self.get(j).unwrap()
    }
}
impl<'a, T> Index<usize> for ColSliceMut<'a, T> {
    type Output = T;

    fn index(&self, j: usize) -> &Self::Output {
        self.rb().get(j).unwrap()
    }
}
impl<'a, T> IndexMut<usize> for ColSliceMut<'a, T> {
    fn index_mut(&mut self, j: usize) -> &mut Self::Output {
        self.rb_mut().get(j).unwrap()
    }
}

impl<'a, T> IntoIterator for RowSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = ElemIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        ElemIter(self.t())
    }
}
impl<'a, T> IntoIterator for RowSliceMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = ElemIterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        ElemIterMut(self.t())
    }
}

impl<'a, T> IntoIterator for ColSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = ElemIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        ElemIter(self)
    }
}
impl<'a, T> IntoIterator for ColSliceMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = ElemIterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        ElemIterMut(self)
    }
}

pub struct RowIter<'a, T>(MatrixSlice<'a, T>);
pub struct ColIter<'a, T>(MatrixSlice<'a, T>);
pub struct RowIterMut<'a, T>(MatrixSliceMut<'a, T>);
pub struct ColIterMut<'a, T>(MatrixSliceMut<'a, T>);
pub struct ElemIter<'a, T>(ColSlice<'a, T>);
pub struct ElemIterMut<'a, T>(ColSliceMut<'a, T>);

impl<'a, T> RowIter<'a, T> {
    pub fn into_matrix(self) -> MatrixSlice<'a, T> {
        self.0
    }
}
impl<'a, T> RowIterMut<'a, T> {
    pub fn into_matrix(self) -> MatrixSliceMut<'a, T> {
        self.0
    }
}
impl<'a, T> ColIter<'a, T> {
    pub fn into_matrix(self) -> MatrixSlice<'a, T> {
        self.0
    }
}
impl<'a, T> ColIterMut<'a, T> {
    pub fn into_matrix(self) -> MatrixSliceMut<'a, T> {
        self.0
    }
}
impl<'a, T> ElemIter<'a, T> {
    pub fn into_col(self) -> ColSlice<'a, T> {
        self.0
    }
    pub fn into_row(self) -> RowSlice<'a, T> {
        self.0.t()
    }
}
impl<'a, T> ElemIterMut<'a, T> {
    pub fn into_col(self) -> ColSliceMut<'a, T> {
        self.0
    }
    pub fn into_row(self) -> RowSliceMut<'a, T> {
        self.0.t()
    }
}

impl<'a, T> Copy for RowIter<'a, T> {}
impl<'a, T> Copy for ColIter<'a, T> {}
impl<'a, T> Copy for ElemIter<'a, T> {}
impl<'a, T> Clone for RowIter<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Clone for ColIter<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Clone for ElemIter<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'b, 'a, T> Reborrow<'b> for RowIter<'a, T> {
    type Target = RowIter<'b, T>;
    fn rb(&'b self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for RowIter<'a, T> {
    type Target = RowIter<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> Reborrow<'b> for RowIterMut<'a, T> {
    type Target = RowIter<'b, T>;
    fn rb(&'b self) -> Self::Target {
        RowIter(self.0.rb())
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for RowIterMut<'a, T> {
    type Target = RowIterMut<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        RowIterMut(self.0.rb_mut())
    }
}

impl<'b, 'a, T> Reborrow<'b> for ColIter<'a, T> {
    type Target = ColIter<'b, T>;
    fn rb(&'b self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for ColIter<'a, T> {
    type Target = ColIter<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> Reborrow<'b> for ColIterMut<'a, T> {
    type Target = ColIter<'b, T>;
    fn rb(&'b self) -> Self::Target {
        ColIter(self.0.rb())
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for ColIterMut<'a, T> {
    type Target = ColIterMut<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        ColIterMut(self.0.rb_mut())
    }
}

impl<'b, 'a, T> Reborrow<'b> for ElemIter<'a, T> {
    type Target = ElemIter<'b, T>;
    fn rb(&'b self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for ElemIter<'a, T> {
    type Target = ElemIter<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        *self
    }
}
impl<'b, 'a, T> Reborrow<'b> for ElemIterMut<'a, T> {
    type Target = ElemIter<'b, T>;
    fn rb(&'b self) -> Self::Target {
        ElemIter(self.0.rb())
    }
}
impl<'b, 'a, T> ReborrowMut<'b> for ElemIterMut<'a, T> {
    type Target = ElemIterMut<'b, T>;
    fn rb_mut(&'b mut self) -> Self::Target {
        ElemIterMut(self.0.rb_mut())
    }
}

impl<'a, T> Iterator for ElemIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let nrows = self.0.nrows();
        if nrows == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let rs = self.0.row_stride();
            let top = unsafe { &*ptr };
            let bot = unsafe { ColSlice::from_raw_parts(ptr.wrapping_offset(rs), nrows - 1, rs) };

            self.0 = bot;

            Some(top)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.nrows();
        (len, Some(len))
    }
}
impl<'a, T> DoubleEndedIterator for ElemIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let nrows = self.0.nrows();
        if nrows == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let rs = self.0.row_stride();
            let top = unsafe { ColSlice::from_raw_parts(ptr, nrows - 1, rs) };
            let bot = unsafe { &*ptr.wrapping_offset(rs * (nrows - 1) as isize) };

            self.0 = top;

            Some(bot)
        }
    }
}

impl<'a, T> Iterator for ElemIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let nrows = self.0.nrows();
        if nrows == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let rs = self.0.row_stride();
            let top = unsafe { &mut *ptr };
            let bot =
                unsafe { ColSliceMut::from_raw_parts(ptr.wrapping_offset(rs), nrows - 1, rs) };

            self.0 = bot;

            Some(top)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.nrows();
        (len, Some(len))
    }
}
impl<'a, T> DoubleEndedIterator for ElemIterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let nrows = self.0.nrows();
        if nrows == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let rs = self.0.row_stride();
            let top = unsafe { ColSliceMut::from_raw_parts(ptr, nrows - 1, rs) };
            let bot = unsafe { &mut *ptr.wrapping_offset(rs * (nrows - 1) as isize) };

            self.0 = top;

            Some(bot)
        }
    }
}

impl<'a, T> Iterator for RowIter<'a, T> {
    type Item = RowSlice<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let nrows = self.0.nrows();
        if nrows == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let ncols = self.0.ncols();
            let rs = self.0.row_stride();
            let cs = self.0.col_stride();
            let top = unsafe { Self::Item::from_raw_parts(ptr, ncols, cs) };
            let bot = unsafe {
                MatrixSlice::from_raw_parts(ptr.wrapping_offset(rs), nrows - 1, ncols, rs, cs)
            };

            self.0 = bot;

            Some(top)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.nrows();
        (len, Some(len))
    }
}
impl<'a, T> DoubleEndedIterator for RowIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let nrows = self.0.nrows();
        if nrows == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let ncols = self.0.ncols();
            let rs = self.0.row_stride();
            let cs = self.0.col_stride();
            let top = unsafe { MatrixSlice::from_raw_parts(ptr, nrows - 1, ncols, rs, cs) };
            let bot = unsafe {
                Self::Item::from_raw_parts(
                    ptr.wrapping_offset((nrows - 1) as isize * rs),
                    ncols,
                    cs,
                )
            };

            self.0 = top;

            Some(bot)
        }
    }
}

impl<'a, T> Iterator for RowIterMut<'a, T> {
    type Item = RowSliceMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let nrows = self.0.nrows();
        if nrows == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let ncols = self.0.ncols();
            let rs = self.0.row_stride();
            let cs = self.0.col_stride();
            let top = unsafe { Self::Item::from_raw_parts(ptr, ncols, cs) };
            let bot = unsafe {
                MatrixSliceMut::from_raw_parts(ptr.wrapping_offset(rs), nrows - 1, ncols, rs, cs)
            };

            self.0 = bot;

            Some(top)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.nrows();
        (len, Some(len))
    }
}
impl<'a, T> DoubleEndedIterator for RowIterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let nrows = self.0.nrows();
        if nrows == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let ncols = self.0.ncols();
            let rs = self.0.row_stride();
            let cs = self.0.col_stride();
            let top = unsafe { MatrixSliceMut::from_raw_parts(ptr, nrows - 1, ncols, rs, cs) };
            let bot = unsafe {
                Self::Item::from_raw_parts(
                    ptr.wrapping_offset((nrows - 1) as isize * rs),
                    ncols,
                    cs,
                )
            };

            self.0 = top;

            Some(bot)
        }
    }
}

impl<'a, T> Iterator for ColIter<'a, T> {
    type Item = ColSlice<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let ncols = self.0.ncols();
        if ncols == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let nrows = self.0.nrows();
            let rs = self.0.row_stride();
            let cs = self.0.col_stride();
            let left = unsafe { Self::Item::from_raw_parts(ptr, nrows, rs) };
            let right = unsafe {
                MatrixSlice::from_raw_parts(ptr.wrapping_offset(cs), nrows, ncols - 1, rs, cs)
            };

            self.0 = right;
            Some(left)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.ncols();
        (len, Some(len))
    }
}
impl<'a, T> DoubleEndedIterator for ColIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let ncols = self.0.ncols();
        if ncols == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let nrows = self.0.nrows();
            let rs = self.0.row_stride();
            let cs = self.0.col_stride();
            let left = unsafe { MatrixSlice::from_raw_parts(ptr, nrows, ncols - 1, rs, cs) };
            let right = unsafe {
                Self::Item::from_raw_parts(
                    ptr.wrapping_offset((ncols - 1) as isize * cs),
                    nrows,
                    rs,
                )
            };

            self.0 = left;
            Some(right)
        }
    }
}
impl<'a, T> Iterator for ColIterMut<'a, T> {
    type Item = ColSliceMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let ncols = self.0.ncols();
        if ncols == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let nrows = self.0.nrows();
            let rs = self.0.row_stride();
            let cs = self.0.col_stride();
            let left = unsafe { Self::Item::from_raw_parts(ptr, nrows, rs) };
            let right = unsafe {
                MatrixSliceMut::from_raw_parts(ptr.wrapping_offset(cs), nrows, ncols - 1, rs, cs)
            };

            self.0 = right;
            Some(left)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.ncols();
        (len, Some(len))
    }
}
impl<'a, T> DoubleEndedIterator for ColIterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let ncols = self.0.ncols();
        if ncols == 0 {
            None
        } else {
            let ptr = self.0.base.ptr.as_ptr();
            let nrows = self.0.nrows();
            let rs = self.0.row_stride();
            let cs = self.0.col_stride();
            let left = unsafe { MatrixSliceMut::from_raw_parts(ptr, nrows, ncols - 1, rs, cs) };
            let right = unsafe {
                Self::Item::from_raw_parts(
                    ptr.wrapping_offset((ncols - 1) as isize * cs),
                    nrows,
                    rs,
                )
            };

            self.0 = left;
            Some(right)
        }
    }
}

impl<'a, T> ExactSizeIterator for RowIter<'a, T> {}
impl<'a, T> ExactSizeIterator for RowIterMut<'a, T> {}
impl<'a, T> ExactSizeIterator for ColIter<'a, T> {}
impl<'a, T> ExactSizeIterator for ColIterMut<'a, T> {}
impl<'a, T> ExactSizeIterator for ElemIter<'a, T> {}
impl<'a, T> ExactSizeIterator for ElemIterMut<'a, T> {}

impl<'a, T: Debug> Debug for MatrixSlice<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct DebugRowSlice<'a, T>(RowSlice<'a, T>);

        impl<'a, T: Debug> Debug for DebugRowSlice<'a, T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "[")?;
                let mut iter = self.0.rb().into_iter();
                if let Some(first) = iter.next() {
                    write!(f, "{:?}", first)?;
                }
                for elem in iter {
                    write!(f, ", {:?}", elem)?;
                }
                write!(f, "]")
            }
        }

        f.debug_list()
            .entries(self.rb().into_row_iter().map(|r| DebugRowSlice(r)))
            .finish()
    }
}
impl<'a, T: Debug> Debug for MatrixSliceMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.rb().fmt(f)
    }
}
impl<'a, T: Debug> Debug for RowSlice<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.rb().as_2d().fmt(f)
    }
}
impl<'a, T: Debug> Debug for RowSliceMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.rb().as_2d().fmt(f)
    }
}

impl<'a, T: Debug> Debug for ColSlice<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.rb().as_2d().fmt(f)
    }
}
impl<'a, T: Debug> Debug for ColSliceMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.rb().as_2d().fmt(f)
    }
}

fn unique_id<T>() -> usize {
    unique_id::<T> as usize
}

lazy_static::lazy_static! {
    static ref SIMD_ALIGN: usize = {
        if is_x86_feature_detected!("avx") {
            32
        } else {
            16
        }
    };
}

fn align_for<T>() -> usize {
    if unique_id::<T>() == unique_id::<f64>() || unique_id::<T>() == unique_id::<f32>() {
        *SIMD_ALIGN
    } else {
        std::mem::align_of::<T>()
    }
}

struct RawMatrix<T> {
    ptr: NonNull<T>,
    row_capacity: usize,
    col_capacity: usize,
}

#[cold]
fn capacity_overflow() -> ! {
    panic!("capacity overflow")
}

impl<T> RawMatrix<T> {
    pub fn new(row_capacity: usize, col_capacity: usize) -> Self {
        if std::mem::size_of::<T>() == 0 {
            Self {
                ptr: NonNull::<T>::dangling(),
                row_capacity,
                col_capacity,
            }
        } else {
            let cap = row_capacity
                .checked_mul(col_capacity)
                .unwrap_or_else(|| capacity_overflow());
            let cap_bytes = cap
                .checked_mul(std::mem::size_of::<T>())
                .unwrap_or_else(|| capacity_overflow());
            if cap_bytes > isize::MAX as usize {
                capacity_overflow();
            }

            use std::alloc::{alloc, handle_alloc_error, Layout};

            let layout = Layout::from_size_align(cap_bytes, align_for::<T>())
                .ok()
                .unwrap_or_else(|| capacity_overflow());

            let ptr = if layout.size() == 0 {
                std::ptr::NonNull::<T>::dangling()
            } else {
                // SAFETY: we checked that layout has non zero size
                let ptr = unsafe { alloc(layout) } as *mut T;
                if ptr.is_null() {
                    handle_alloc_error(layout)
                } else {
                    // SAFETY: we checked that the pointer is not null
                    unsafe { NonNull::<T>::new_unchecked(ptr) }
                }
            };

            Self {
                ptr,
                row_capacity,
                col_capacity,
            }
        }
    }
}

impl<T> Drop for RawMatrix<T> {
    fn drop(&mut self) {
        use std::alloc::{dealloc, Layout};
        // this cannot overflow because we already allocated this much memory
        let alloc_size =
            self.row_capacity.wrapping_mul(self.col_capacity) * std::mem::size_of::<T>();
        if alloc_size != 0 {
            // SAFETY: pointer was allocated with std::alloc::alloc
            unsafe {
                dealloc(
                    self.ptr.as_ptr() as *mut u8,
                    Layout::from_size_align_unchecked(alloc_size, align_for::<T>()),
                );
            }
        }
    }
}

/// Owning 2D matrix stored in column major format.
pub struct Matrix<T> {
    raw: RawMatrix<T>,
    nrows: usize,
    ncols: usize,
}

impl<T> Matrix<T> {
    /// Returns a new matrix with dimensions `(0, 0)`. This does not allocate.
    #[inline]
    pub fn new() -> Self {
        Self {
            raw: RawMatrix::<T> {
                ptr: NonNull::<T>::dangling(),
                row_capacity: 0,
                col_capacity: 0,
            },
            nrows: 0,
            ncols: 0,
        }
    }

    /// Returns a matrix from preallocated pointer, dimensions, and capacities.
    ///
    /// # Safety
    ///
    /// The inputs to this function must be acquired from the return value of some previous call
    /// to `Self::into_raw_parts`.
    #[inline]
    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        nrows: usize,
        ncols: usize,
        row_capacity: usize,
        col_capacity: usize,
    ) -> Self {
        Self {
            raw: RawMatrix::<T> {
                ptr: NonNull::new_unchecked(ptr),
                row_capacity,
                col_capacity,
            },
            nrows,
            ncols,
        }
    }

    /// Consumes `self` and returns its raw parts in this order: pointer to data, number of rows,
    /// number of columns, row capacity and column capacity.
    #[inline]
    pub fn into_raw_parts(self) -> (*mut T, usize, usize, usize, usize) {
        let mut m = std::mem::ManuallyDrop::<Matrix<T>>::new(self);
        (
            m.as_mut_ptr(),
            m.nrows(),
            m.ncols(),
            m.row_capacity(),
            m.col_capacity(),
        )
    }

    /// Returns a new matrix with dimensions `(0, 0)`, with enough capacity to hold a maximum of
    /// `row_capacity` rows and `col_capacity` columns without reallocating. If either is `0`,
    /// the matrix will not allocate.
    ///
    /// # Panics
    ///
    /// Panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn with_capacity(row_capacity: usize, col_capacity: usize) -> Self {
        Self {
            raw: RawMatrix::<T>::new(row_capacity, col_capacity),
            nrows: 0,
            ncols: 0,
        }
    }

    #[cold]
    fn do_reserve_exact(&mut self, mut new_row_capacity: usize, mut new_col_capacity: usize) {
        use std::mem::ManuallyDrop;

        new_row_capacity = self.row_capacity().max(new_row_capacity);
        new_col_capacity = self.col_capacity().max(new_col_capacity);

        let new_ptr = if self.row_capacity() == new_row_capacity
            && self.row_capacity() != 0
            && self.col_capacity() != 0
        {
            // case 1:
            // we have enough row capacity, and we've already allocated memory.
            // use realloc to get extra column memory

            use std::alloc::{handle_alloc_error, realloc, Layout};

            let old_cap = self
                .row_capacity()
                .checked_mul(self.col_capacity())
                .unwrap_or_else(|| capacity_overflow());
            let old_cap_bytes = old_cap
                .checked_mul(std::mem::size_of::<T>())
                .unwrap_or_else(|| capacity_overflow());

            let new_cap = new_row_capacity
                .checked_mul(new_col_capacity)
                .unwrap_or_else(|| capacity_overflow());
            let new_cap_bytes = new_cap
                .checked_mul(std::mem::size_of::<T>())
                .unwrap_or_else(|| capacity_overflow());

            if new_cap_bytes > isize::MAX as usize {
                capacity_overflow();
            }

            let old_layout =
                unsafe { Layout::from_size_align_unchecked(old_cap_bytes, align_for::<T>()) };
            let new_layout = Layout::from_size_align(new_cap_bytes, align_for::<T>())
                .ok()
                .unwrap_or_else(|| capacity_overflow());

            unsafe {
                let old_ptr = self.as_mut_ptr();
                let new_ptr = realloc(old_ptr as *mut u8, old_layout, new_cap_bytes);
                if new_ptr.is_null() {
                    handle_alloc_error(new_layout);
                }
                new_ptr as *mut T
            }
        } else {
            // case 2:
            // use alloc and move stuff manually.

            // allocate new memory region
            let new_ptr = {
                let m = ManuallyDrop::new(RawMatrix::<T>::new(new_row_capacity, new_col_capacity));
                m.ptr.as_ptr()
            };

            let old_ptr = self.as_mut_ptr();

            // copy each column to new matrix
            for j in 0..self.ncols() {
                // SAFETY:
                // * pointer offsets can't overflow since they're within an already allocated
                // memory region less than isize::MAX bytes in size.
                // * new and old allocation can't overlap, so copy_nonoverlapping is fine here.
                unsafe {
                    let old_ptr = old_ptr.offset((j * self.row_capacity()) as isize);
                    let new_ptr = new_ptr.offset((j * new_row_capacity) as isize);
                    std::ptr::copy_nonoverlapping(old_ptr, new_ptr, self.nrows());
                }
            }

            // deallocate old matrix memory
            let _ = RawMatrix::<T> {
                ptr: unsafe { NonNull::new_unchecked(old_ptr) },
                row_capacity: self.row_capacity(),
                col_capacity: self.col_capacity(),
            };

            new_ptr
        };
        self.raw.row_capacity = new_row_capacity;
        self.raw.col_capacity = new_col_capacity;
        self.raw.ptr = unsafe { NonNull::<T>::new_unchecked(new_ptr) };
    }

    /// Reserves the minimum capacity for `row_capacity` rows and `col_capacity`
    /// columns without reallocating. Does nothing if the capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn reserve_exact(&mut self, row_capacity: usize, col_capacity: usize) {
        if self.row_capacity() >= row_capacity && self.col_capacity() >= col_capacity {
            // do nothing
        } else {
            if std::mem::size_of::<T>() == 0 {
                self.raw.row_capacity = self.row_capacity().max(row_capacity);
                self.raw.col_capacity = self.col_capacity().max(col_capacity);
            } else {
                self.do_reserve_exact(row_capacity, col_capacity);
            }
        }
    }

    /// Returns a pointer to the data of the matrix.
    pub fn as_ptr(&self) -> *const T {
        self.raw.ptr.as_ptr()
    }

    /// Returns a mutable pointer to the data of the matrix.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.raw.ptr.as_ptr()
    }

    /// Returns the number of rows of the matrix.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns of the matrix.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the row capacity, that is, the number of rows that the matrix is able to hold
    /// without needing to reallocate, excluding column insertions.
    pub fn row_capacity(&self) -> usize {
        self.raw.row_capacity
    }

    /// Returns the column capacity, that is, the number of columns that the matrix is able to hold
    /// without needing to reallocate, excluding row insertions.
    pub fn col_capacity(&self) -> usize {
        self.raw.col_capacity
    }
}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        let mut ptr = self.raw.ptr.as_ptr();
        let nrows = self.nrows;
        let ncols = self.ncols;
        let cs = self.raw.row_capacity;

        for _ in 0..ncols {
            for i in 0..nrows {
                unsafe {
                    std::ptr::drop_in_place(ptr.add(i));
                }
            }
            ptr = ptr.wrapping_add(cs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let slice = unsafe { MatrixSlice::from_raw_parts(data.as_ptr(), 2, 3, 3, 1) };

        dbg!(slice);
        dbg!(slice.t());

        assert_eq!(slice.rb().get(0, 0), Some(&1.0));
        assert_eq!(slice.rb().get(0, 1), Some(&2.0));
        assert_eq!(slice.rb().get(0, 2), Some(&3.0));
        assert_eq!(slice.rb().get(0, 3), None);

        assert_eq!(slice.rb().get(1, 0), Some(&4.0));
        assert_eq!(slice.rb().get(1, 1), Some(&5.0));
        assert_eq!(slice.rb().get(1, 2), Some(&6.0));
        assert_eq!(slice.rb().get(1, 3), None);

        assert_eq!(slice.rb().get(2, 0), None);

        for r in slice.into_row_iter() {
            dbg!(r);
        }
        for r in slice.into_row_iter().rev() {
            dbg!(r);
        }

        for c in slice.into_col_iter() {
            dbg!(c);
        }
        for c in slice.into_col_iter().rev() {
            dbg!(c);
        }
    }

    #[test]
    fn basic_slice_mut() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut slice = unsafe { MatrixSliceMut::from_raw_parts(data.as_mut_ptr(), 2, 3, 3, 1) };

        fancy_assert!(slice.rb_mut().get(0, 0) == Some(&mut 1.0));
        fancy_assert!(slice.rb_mut().get(0, 1) == Some(&mut 2.0));
        fancy_assert!(slice.rb_mut().get(0, 2) == Some(&mut 3.0));
        fancy_assert!(slice.rb_mut().get(0, 3) == None);

        fancy_assert!(slice.rb_mut().get(1, 0) == Some(&mut 4.0));
        fancy_assert!(slice.rb_mut().get(1, 1) == Some(&mut 5.0));
        fancy_assert!(slice.rb_mut().get(1, 2) == Some(&mut 6.0));
        fancy_assert!(slice.rb_mut().get(1, 3) == None);

        fancy_assert!(slice.rb_mut().get(2, 0) == None);

        for r in slice.rb_mut().into_row_iter() {
            dbg!(&r);
            for e in r {
                dbg!(e);
            }
        }
        for r in slice.rb_mut().into_row_iter().rev() {
            dbg!(&r);
            for e in r.into_iter().rev() {
                dbg!(e);
            }
        }

        for c in slice.rb_mut().into_col_iter() {
            dbg!(&c);
            for e in c {
                dbg!(e);
            }
        }
        for c in slice.rb_mut().into_col_iter().rev() {
            dbg!(&c);
            for e in c.into_iter().rev() {
                dbg!(e);
            }
        }
    }

    #[test]
    fn empty() {
        {
            let m = Matrix::<f64>::new();
            fancy_assert!(m.nrows() == 0);
            fancy_assert!(m.ncols() == 0);
            fancy_assert!(m.row_capacity() == 0);
            fancy_assert!(m.col_capacity() == 0);
        }

        {
            let m = Matrix::<f64>::with_capacity(100, 120);
            fancy_assert!(m.nrows() == 0);
            fancy_assert!(m.ncols() == 0);
            fancy_assert!(m.row_capacity() == 100);
            fancy_assert!(m.col_capacity() == 120);
        }
    }

    #[test]
    fn reserve() {
        let mut m = Matrix::<f64>::new();

        m.reserve_exact(0, 0);
        fancy_assert!(m.row_capacity() == 0);
        fancy_assert!(m.col_capacity() == 0);

        m.reserve_exact(1, 1);
        fancy_assert!(m.row_capacity() == 1);
        fancy_assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 0);
        fancy_assert!(m.row_capacity() == 2);
        fancy_assert!(m.col_capacity() == 1);

        m.reserve_exact(2, 3);
        fancy_assert!(m.row_capacity() == 2);
        fancy_assert!(m.col_capacity() == 3);
    }

    #[test]
    #[should_panic]
    fn cap_overflow_1() {
        let _ = Matrix::<f64>::with_capacity(isize::MAX as usize, 1);
    }

    #[test]
    #[should_panic]
    fn cap_overflow_2() {
        let _ = Matrix::<f64>::with_capacity(isize::MAX as usize, isize::MAX as usize);
    }

    #[test]
    fn x() {
        let mut v: Vec<()> = Vec::new();
        v.reserve(usize::MAX);
        v.reserve(1);
    }
}
