use super::*;
use crate::{
    col::{ColMut, ColRef},
    debug_assert, iter,
    mat::matalloc::{align_for, is_vectorizable, MatUnit, RawMat, RawMatUnit},
    utils::DivCeil,
};
use core::mem::{ManuallyDrop, MaybeUninit};

/// Heap allocated resizable row vector.
///
/// # Note
///
/// The memory layout of `Col` is guaranteed to be row-major, meaning that it has a column stride
/// of `1`.
#[repr(C)]
pub struct Row<E: Entity> {
    inner: VecOwnImpl<E>,
    col_capacity: usize,
    __marker: PhantomData<E>,
}

impl<E: Entity> Drop for Row<E> {
    #[inline]
    fn drop(&mut self) {
        drop(RawMat::<E> {
            ptr: self.inner.ptr,
            row_capacity: self.col_capacity,
            col_capacity: 1,
        });
    }
}

impl<E: Entity> Row<E> {
    /// Returns an empty row of dimension `0`.
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: VecOwnImpl {
                ptr: into_copy::<E, _>(E::faer_map(E::UNIT, |()| NonNull::<E::Unit>::dangling())),
                len: 0,
            },
            col_capacity: 0,
            __marker: PhantomData,
        }
    }

    /// Returns a new column vector with 0 columns, with enough capacity to hold a maximum of
    /// `col_capacity` columns without reallocating. If `col_capacity` is `0`,
    /// the function will not allocate.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn with_capacity(col_capacity: usize) -> Self {
        let raw = ManuallyDrop::new(RawMat::<E>::new(col_capacity, 1));
        Self {
            inner: VecOwnImpl {
                ptr: raw.ptr,
                len: 0,
            },
            col_capacity: raw.row_capacity,
            __marker: PhantomData,
        }
    }

    /// Returns a new matrix with number of columns `ncols`, filled with the provided function.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn from_fn(ncols: usize, f: impl FnMut(usize) -> E) -> Self {
        let mut this = Self::new();
        this.resize_with(ncols, f);
        this
    }

    /// Returns a new matrix with number of columns `ncols`, filled with zeros.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn zeros(ncols: usize) -> Self {
        Self::from_fn(ncols, |_| unsafe { core::mem::zeroed() })
    }

    /// Returns a new matrix with number of columns `ncols`, filled with ones.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn ones(ncols: usize) -> Self
    where
        E: ComplexField,
    {
        Self::full(ncols, E::faer_one())
    }

    /// Returns a new matrix with number of columns `ncols`, filled with a constant value.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn full(ncols: usize, constant: E) -> Self
    where
        E: ComplexField,
    {
        Self::from_fn(ncols, |_| constant)
    }

    /// Returns the number of rows of the row. This is always equal to `1`.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        1
    }
    /// Returns the number of columns of the row.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.inner.len
    }

    /// Set the dimensions of the matrix.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `ncols < self.col_capacity()`.
    /// * The elements that were previously out of bounds but are now in bounds must be
    /// initialized.
    #[inline]
    pub unsafe fn set_ncols(&mut self, ncols: usize) {
        self.inner.len = ncols;
    }

    /// Returns a pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr(&self) -> GroupFor<E, *const E::Unit> {
        E::faer_map(from_copy::<E, _>(self.inner.ptr), |ptr| {
            ptr.as_ptr() as *const E::Unit
        })
    }

    /// Returns a mutable pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr_mut(&mut self) -> GroupFor<E, *mut E::Unit> {
        E::faer_map(from_copy::<E, _>(self.inner.ptr), |ptr| ptr.as_ptr())
    }

    /// Returns the col capacity, that is, the number of cols that the matrix is able to hold
    /// without needing to reallocate, excluding column insertions.
    #[inline]
    pub fn col_capacity(&self) -> usize {
        self.col_capacity
    }

    /// Returns the offset between the first elements of two successive columns in the matrix.
    /// Always returns `1` since the matrix is column major.
    #[inline]
    pub fn col_stride(&self) -> isize {
        1
    }

    /// Returns `self` as a matrix view.
    #[inline(always)]
    pub fn as_2d(&self) -> MatRef<'_, E> {
        self.as_ref().as_2d()
    }

    /// Returns `self` as a mutable matrix view.
    #[inline(always)]
    pub fn as_2d_mut(&mut self) -> MatMut<'_, E> {
        self.as_mut().as_2d_mut()
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at(&self, col: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().ptr_at(col)
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at_mut(&mut self, col: usize) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().ptr_at_mut(col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(&self, col: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().ptr_at_unchecked(col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_mut_unchecked(&mut self, col: usize) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().ptr_at_mut_unchecked(col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(&self, col: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().overflowing_ptr_at(col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at_mut(&mut self, col: usize) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().overflowing_ptr_at_mut(col)
    }

    /// Returns raw pointers to the element at the given index, assuming the provided index
    /// is within the size of the vector.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, col: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().ptr_inbounds_at(col)
    }

    /// Returns raw pointers to the element at the given index, assuming the provided index
    /// is within the size of the vector.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(&mut self, col: usize) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().ptr_inbounds_at_mut(col)
    }

    /// Splits the column vector at the given index into two parts and
    /// returns an array of each subvector, in the following order:
    /// * left.
    /// * right.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_unchecked(&self, col: usize) -> (RowRef<'_, E>, RowRef<'_, E>) {
        self.as_ref().split_at_unchecked(col)
    }

    /// Splits the column vector at the given index into two parts and
    /// returns an array of each subvector, in the following order:
    /// * left.
    /// * right.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_mut_unchecked(&mut self, col: usize) -> (RowMut<'_, E>, RowMut<'_, E>) {
        self.as_mut().split_at_mut_unchecked(col)
    }

    /// Splits the column vector at the given index into two parts and
    /// returns an array of each subvector, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at(&self, col: usize) -> (RowRef<'_, E>, RowRef<'_, E>) {
        self.as_ref().split_at(col)
    }

    /// Splits the column vector at the given index into two parts and
    /// returns an array of each subvector, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_mut(&mut self, col: usize) -> (RowMut<'_, E>, RowMut<'_, E>) {
        self.as_mut().split_at_mut(col)
    }

    #[cold]
    fn do_reserve_exact(&mut self, mut new_col_capacity: usize) {
        if is_vectorizable::<E::Unit>() {
            let align_factor = align_for::<E::Unit>() / core::mem::size_of::<E::Unit>();
            new_col_capacity = new_col_capacity
                .msrv_checked_next_multiple_of(align_factor)
                .unwrap();
        }

        let ncols = self.inner.len;
        let old_col_capacity = self.col_capacity;

        let mut this = ManuallyDrop::new(core::mem::take(self));
        {
            let mut this_group = E::faer_map(from_copy::<E, _>(this.inner.ptr), |ptr| MatUnit {
                raw: RawMatUnit {
                    ptr,
                    row_capacity: old_col_capacity,
                    col_capacity: 1,
                },
                nrows: ncols,
                ncols: 1,
            });

            E::faer_map(E::faer_as_mut(&mut this_group), |mat_unit| {
                mat_unit.do_reserve_exact(new_col_capacity, 1);
            });

            let this_group = E::faer_map(this_group, ManuallyDrop::new);
            this.inner.ptr =
                into_copy::<E, _>(E::faer_map(this_group, |mat_unit| mat_unit.raw.ptr));
            this.col_capacity = new_col_capacity;
        }
        *self = ManuallyDrop::into_inner(this);
    }

    /// Reserves the minimum capacity for `col_capacity` columns without reallocating. Does nothing
    /// if the capacity is already sufficient.
    ///
    /// # Panics
    /// The function panics if the new total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn reserve_exact(&mut self, col_capacity: usize) {
        if self.col_capacity() >= col_capacity {
            // do nothing
        } else if core::mem::size_of::<E::Unit>() == 0 {
            self.col_capacity = self.col_capacity().max(col_capacity);
        } else {
            self.do_reserve_exact(col_capacity);
        }
    }

    unsafe fn insert_block_with<F: FnMut(usize) -> E>(
        &mut self,
        f: &mut F,
        col_start: usize,
        col_end: usize,
    ) {
        debug_assert!(col_start <= col_end);

        let ptr = self.as_ptr_mut();

        for j in col_start..col_end {
            // SAFETY:
            // * pointer to element at index (i, j), which is within the
            // allocation since we reserved enough space
            // * writing to this memory region is sound since it is properly
            // aligned and valid for writes
            let ptr_ij = E::faer_map(E::faer_copy(&ptr), |ptr| ptr.add(j));
            let value = E::faer_into_units(f(j));

            E::faer_map(E::faer_zip(ptr_ij, value), |(ptr_ij, value)| {
                core::ptr::write(ptr_ij, value)
            });
        }
    }

    fn erase_last_cols(&mut self, new_ncols: usize) {
        let old_ncols = self.ncols();
        debug_assert!(new_ncols <= old_ncols);
        self.inner.len = new_ncols;
    }

    unsafe fn insert_last_cols_with<F: FnMut(usize) -> E>(&mut self, f: &mut F, new_ncols: usize) {
        let old_ncols = self.ncols();

        debug_assert!(new_ncols > old_ncols);

        self.insert_block_with(f, old_ncols, new_ncols);
        self.inner.len = new_ncols;
    }

    /// Resizes the vector in-place so that the new number of columns is `new_ncols`.
    /// New elements are created with the given function `f`, so that elements at index `i`
    /// are created by calling `f(i)`.
    pub fn resize_with(&mut self, new_ncols: usize, f: impl FnMut(usize) -> E) {
        let mut f = f;
        let old_ncols = self.ncols();

        if new_ncols <= old_ncols {
            self.erase_last_cols(new_ncols);
        } else {
            self.reserve_exact(new_ncols);
            unsafe {
                self.insert_last_cols_with(&mut f, new_ncols);
            }
        }
    }

    /// Truncates the matrix so that its new number of columns is `new_ncols`.  
    /// The new dimension must be smaller than the current dimension of the vector.
    ///
    /// # Panics
    /// - Panics if `new_ncols > self.ncols()`.
    #[inline]
    pub fn truncate(&mut self, new_ncols: usize) {
        assert!(new_ncols <= self.ncols());
        self.erase_last_cols(new_ncols);
    }

    /// Returns a reference to a slice over the row.
    #[inline]
    #[track_caller]
    pub fn as_slice(&self) -> GroupFor<E, &[E::Unit]> {
        let ncols = self.ncols();
        let ptr = self.as_ref().as_ptr();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, ncols) },
        )
    }

    /// Returns a mutable reference to a slice over the row.
    #[inline]
    #[track_caller]
    pub fn as_slice_mut(&mut self) -> GroupFor<E, &mut [E::Unit]> {
        let ncols = self.ncols();
        let ptr = self.as_ptr_mut();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, ncols) },
        )
    }

    /// Returns the row as a contiguous slice if its column stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice(&self) -> Option<GroupFor<E, &[E::Unit]>> {
        Some(self.as_slice())
    }

    /// Returns the row as a contiguous slice if its column stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice_mut(&mut self) -> Option<GroupFor<E, &mut [E::Unit]>> {
        Some(self.as_slice_mut())
    }

    /// Returns the row as a contiguous potentially uninitialized slice if its column stride is
    /// equal to `1`.
    ///
    /// # Safety
    /// If uninit data is written to the slice, it must not be read at some later point.
    #[inline]
    pub unsafe fn try_as_uninit_slice_mut(
        &mut self,
    ) -> Option<GroupFor<E, &mut [MaybeUninit<E::Unit>]>> {
        Some(self.as_uninit_slice_mut())
    }

    /// Returns a mutable reference to a potentially uninitialized slice over the column.
    ///
    /// # Safety
    /// If uninit data is written to the slice, it must not be later read.
    #[inline]
    pub unsafe fn as_uninit_slice_mut(&mut self) -> GroupFor<E, &mut [MaybeUninit<E::Unit>]> {
        let nrows = self.nrows();
        let ptr = self.as_ptr_mut();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr as _, nrows) },
        )
    }

    /// Returns a view over the subvector starting at column `col_start`, and with number of
    /// columns `ncols`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col_start <= self.ncols()`.
    /// * `ncols <= self.ncols() - col_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn subcols_unchecked(&self, col_start: usize, ncols: usize) -> RowRef<'_, E> {
        self.as_ref().subcols_unchecked(col_start, ncols)
    }

    /// Returns a view over the subvector starting at col `col_start`, and with number of cols
    /// `ncols`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_start <= self.ncols()`.
    /// * `ncols <= self.ncols() - col_start`.
    #[track_caller]
    #[inline(always)]
    pub fn subcols(&self, col_start: usize, ncols: usize) -> RowRef<'_, E> {
        self.as_ref().subcols(col_start, ncols)
    }

    /// Returns a view over the subvector starting at col `col_start`, and with number of
    /// columns `ncols`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col_start <= self.ncols()`.
    /// * `ncols <= self.ncols() - col_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn subcols_mut_unchecked(
        &mut self,
        col_start: usize,
        ncols: usize,
    ) -> RowMut<'_, E> {
        self.as_mut().subcols_mut_unchecked(col_start, ncols)
    }

    /// Returns a view over the subvector starting at col `col_start`, and with number of
    /// columns `ncols`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col_start <= self.ncols()`.
    /// * `ncols <= self.ncols() - col_start`.
    #[track_caller]
    #[inline(always)]
    pub fn subcols_mut(&mut self, col_start: usize, ncols: usize) -> RowMut<'_, E> {
        self.as_mut().subcols_mut(col_start, ncols)
    }

    /// Returns a view over the vector.
    #[inline]
    pub fn as_ref(&self) -> RowRef<'_, E> {
        unsafe { super::from_raw_parts(self.as_ptr(), self.ncols(), 1) }
    }

    /// Returns a mutable view over the vector.
    #[inline]
    pub fn as_mut(&mut self) -> RowMut<'_, E> {
        unsafe { super::from_raw_parts_mut(self.as_ptr_mut(), self.ncols(), 1) }
    }

    /// Returns references to the element at the given index, or submatrices if `col` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub unsafe fn get_unchecked<ColRange>(
        &self,
        col: ColRange,
    ) -> <RowRef<'_, E> as RowIndex<ColRange>>::Target
    where
        for<'a> RowRef<'a, E>: RowIndex<ColRange>,
    {
        self.as_ref().get_unchecked(col)
    }

    /// Returns references to the element at the given index, or submatrices if `col` is a range,
    /// with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub fn get<ColRange>(&self, col: ColRange) -> <RowRef<'_, E> as RowIndex<ColRange>>::Target
    where
        for<'a> RowRef<'a, E>: RowIndex<ColRange>,
    {
        self.as_ref().get(col)
    }

    /// Returns mutable references to the element at the given index, or submatrices if
    /// `col` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub unsafe fn get_mut_unchecked<ColRange>(
        &mut self,
        col: ColRange,
    ) -> <RowMut<'_, E> as RowIndex<ColRange>>::Target
    where
        for<'a> RowMut<'a, E>: RowIndex<ColRange>,
    {
        self.as_mut().get_mut_unchecked(col)
    }

    /// Returns mutable references to the element at the given index, or submatrices if
    /// `col` is a range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    pub fn get_mut<ColRange>(
        &mut self,
        col: ColRange,
    ) -> <RowMut<'_, E> as RowIndex<ColRange>>::Target
    where
        for<'a> RowMut<'a, E>: RowIndex<ColRange>,
    {
        self.as_mut().get_mut(col)
    }

    /// Reads the value of the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, col: usize) -> E {
        self.as_ref().read_unchecked(col)
    }

    /// Reads the value of the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, col: usize) -> E {
        self.as_ref().read(col)
    }

    /// Writes the value to the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, col: usize, value: E) {
        self.as_mut().write_unchecked(col, value);
    }

    /// Writes the value to the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, col: usize, value: E) {
        self.as_mut().write(col, value);
    }

    /// Copies the values from `other` into `self`.
    #[inline(always)]
    #[track_caller]
    pub fn copy_from<ViewE: Conjugate<Canonical = E>>(&mut self, other: impl AsRowRef<ViewE>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity, ViewE: Conjugate<Canonical = E>>(
            this: &mut Row<E>,
            other: RowRef<'_, ViewE>,
        ) {
            this.resize_with(0, |_| E::zeroed());
            this.resize_with(
                other.nrows(),
                #[inline(always)]
                |row| unsafe { other.read_unchecked(row).canonicalize() },
            );
        }
        implementation(self, other.as_row_ref());
    }

    /// Fills the elements of `self` with zeros.
    #[inline(always)]
    #[track_caller]
    pub fn fill_zero(&mut self)
    where
        E: ComplexField,
    {
        self.as_mut().fill_zero()
    }

    /// Fills the elements of `self` with copies of `constant`.
    #[inline(always)]
    #[track_caller]
    pub fn fill(&mut self, constant: E) {
        self.as_mut().fill(constant)
    }

    /// Returns a view over the transpose of `self`.
    #[inline]
    pub fn transpose(&self) -> ColRef<'_, E> {
        self.as_ref().transpose()
    }

    /// Returns a view over the transpose of `self`.
    #[inline]
    pub fn transpose_mut(&mut self) -> ColMut<'_, E> {
        self.as_mut().transpose_mut()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(&self) -> RowRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate_mut(&mut self) -> RowMut<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_mut().conjugate_mut()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(&self) -> ColRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint_mut(&mut self) -> ColMut<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_mut().adjoint_mut()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize(&self) -> (RowRef<'_, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        self.as_ref().canonicalize()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize_mut(&mut self) -> (RowMut<'_, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        self.as_mut().canonicalize_mut()
    }

    /// Returns a view over the `self`, with the columns in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols(&self) -> RowRef<'_, E> {
        self.as_ref().reverse_cols()
    }

    /// Returns a view over the `self`, with the columns in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols_mut(&mut self) -> RowMut<'_, E> {
        self.as_mut().reverse_cols_mut()
    }

    /// Returns an owning [`Row`] of the data
    #[inline]
    pub fn to_owned(&self) -> Row<E::Canonical>
    where
        E: Conjugate,
    {
        self.as_ref().to_owned()
    }

    /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
    #[inline]
    pub fn has_nan(&self) -> bool
    where
        E: ComplexField,
    {
        self.as_ref().has_nan()
    }

    /// Returns `true` if all of the elements are finite, otherwise returns `false`.
    #[inline]
    pub fn is_all_finite(&self) -> bool
    where
        E: ComplexField,
    {
        self.as_ref().is_all_finite()
    }

    /// Returns the maximum norm of `self`.
    #[inline]
    pub fn norm_max(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.as_ref().as_2d().norm_max()
    }
    /// Returns the L1 norm of `self`.
    #[inline]
    pub fn norm_l1(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.as_ref().as_2d().norm_l1()
    }

    /// Returns the L2 norm of `self`.
    #[inline]
    pub fn norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.as_ref().as_2d().norm_l2()
    }

    /// Returns the squared L2 norm of `self`.
    #[inline]
    pub fn squared_norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.as_ref().as_2d().squared_norm_l2()
    }

    /// Returns the sum of `self`.
    #[inline]
    pub fn sum(&self) -> E
    where
        E: ComplexField,
    {
        self.as_ref().as_2d().sum()
    }

    /// Kronecker product of `self` and `rhs`.
    ///
    /// This is an allocating operation; see [`faer::linalg::kron`](crate::linalg::kron) for the
    /// allocation-free version or more info in general.
    #[inline]
    #[track_caller]
    pub fn kron(&self, rhs: impl As2D<E>) -> Mat<E>
    where
        E: ComplexField,
    {
        self.as_2d_ref().kron(rhs)
    }

    /// Returns a reference to the first element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_first(&self) -> Option<(GroupFor<E, &'_ E::Unit>, RowRef<'_, E>)> {
        self.as_ref().split_first()
    }

    /// Returns a reference to the last element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_last(&self) -> Option<(GroupFor<E, &'_ E::Unit>, RowRef<'_, E>)> {
        self.as_ref().split_last()
    }

    /// Returns a reference to the first element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_first_mut(&mut self) -> Option<(GroupFor<E, &'_ mut E::Unit>, RowMut<'_, E>)> {
        self.as_mut().split_first_mut()
    }

    /// Returns a reference to the last element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_last_mut(&mut self) -> Option<(GroupFor<E, &'_ mut E::Unit>, RowMut<'_, E>)> {
        self.as_mut().split_last_mut()
    }

    /// Returns an iterator over the elements of the row.
    #[inline]
    pub fn iter(&self) -> iter::ElemIter<'_, E> {
        self.as_ref().iter()
    }

    /// Returns an iterator over the elements of the row.
    #[inline]
    pub fn iter_mut(&mut self) -> iter::ElemIterMut<'_, E> {
        self.as_mut().iter_mut()
    }

    /// Returns an iterator that provides successive chunks of the elements of this row, with
    /// each having at most `chunk_size` elements.
    #[inline]
    #[track_caller]
    pub fn chunks(&self, chunk_size: usize) -> iter::RowElemChunks<'_, E> {
        self.as_ref().chunks(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// row.
    #[inline]
    #[track_caller]
    pub fn partition(&self, count: usize) -> iter::RowElemPartition<'_, E> {
        self.as_ref().partition(count)
    }

    /// Returns an iterator that provides successive chunks of the elements of this row, with
    /// each having at most `chunk_size` elements.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = RowRef<'_, E>> {
        self.as_ref().par_chunks(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// row.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_partition(
        &self,
        count: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = RowRef<'_, E>> {
        self.as_ref().par_partition(count)
    }

    /// Returns an iterator that provides successive chunks of the elements of this row, with
    /// each having at most `chunk_size` elements.
    #[inline]
    #[track_caller]
    pub fn chunks_mut(&mut self, chunk_size: usize) -> iter::RowElemChunksMut<'_, E> {
        self.as_mut().chunks_mut(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// row.
    #[inline]
    #[track_caller]
    pub fn partition_mut(&mut self, count: usize) -> iter::RowElemPartitionMut<'_, E> {
        self.as_mut().partition_mut(count)
    }

    /// Returns an iterator that provides successive chunks of the elements of this row, with
    /// each having at most `chunk_size` elements.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = RowMut<'_, E>> {
        self.as_mut().par_chunks_mut(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// row.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_partition_mut(
        &mut self,
        count: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = RowMut<'_, E>> {
        self.as_mut().par_partition_mut(count)
    }

    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn const_cast(&self) -> RowMut<'_, E> {
        self.as_ref().const_cast()
    }
}

impl<E: Entity> Default for Row<E> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Entity> Clone for Row<E> {
    fn clone(&self) -> Self {
        let this = self.as_ref();
        unsafe {
            Self::from_fn(self.ncols(), |j| {
                E::faer_from_units(E::faer_deref(this.get_unchecked(j)))
            })
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.resize_with(0, |_| E::zeroed());
        self.resize_with(
            other.nrows(),
            #[inline(always)]
            |row| unsafe { other.read_unchecked(row) },
        );
    }
}

impl<E: Entity> As2D<E> for Row<E> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (*self).as_ref().as_2d()
    }
}

impl<E: Entity> As2DMut<E> for Row<E> {
    #[inline]
    fn as_2d_mut(&mut self) -> MatMut<'_, E> {
        (*self).as_mut().as_2d_mut()
    }
}

impl<E: Entity> AsRowRef<E> for Row<E> {
    #[inline]
    fn as_row_ref(&self) -> RowRef<'_, E> {
        (*self).as_ref()
    }
}

impl<E: Entity> AsRowMut<E> for Row<E> {
    #[inline]
    fn as_row_mut(&mut self) -> RowMut<'_, E> {
        (*self).as_mut()
    }
}

impl<E: Entity> core::fmt::Debug for Row<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<E: SimpleEntity> core::ops::Index<usize> for Row<E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, col: usize) -> &E {
        self.as_ref().get(col)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<usize> for Row<E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, col: usize) -> &mut E {
        self.as_mut().get_mut(col)
    }
}

impl<E: Conjugate> RowBatch<E> for Row<E> {
    type Owned = Row<E::Canonical>;

    #[inline]
    #[track_caller]
    fn new_owned_zeros(nrows: usize, ncols: usize) -> Self::Owned {
        assert!(nrows == 1);
        Row::zeros(ncols)
    }

    #[inline]
    fn new_owned_copied(src: &Self) -> Self::Owned {
        src.to_owned()
    }

    #[inline]
    #[track_caller]
    fn resize_owned(owned: &mut Self::Owned, nrows: usize, ncols: usize) {
        assert!(ncols == 1);
        owned.resize_with(nrows, |_| unsafe { core::mem::zeroed() });
    }
}

impl<E: Conjugate> RowBatchMut<E> for Row<E> {}
