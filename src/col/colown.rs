use super::*;
use crate::{
    debug_assert,
    diag::{Diag, DiagMut, DiagRef},
    mat::matalloc::{align_for, is_vectorizable, MatUnit, RawMat, RawMatUnit},
    row::{RowMut, RowRef},
    utils::DivCeil,
};
use core::mem::{ManuallyDrop, MaybeUninit};

/// Heap allocated resizable column vector.
///
/// # Note
///
/// The memory layout of `Col` is guaranteed to be column-major, meaning that it has a row stride
/// of `1`.
#[repr(C)]
pub struct Col<E: Entity> {
    inner: VecOwnImpl<E>,
    row_capacity: usize,
    __marker: PhantomData<E>,
}

impl<E: Entity> Drop for Col<E> {
    #[inline]
    fn drop(&mut self) {
        drop(RawMat::<E> {
            ptr: self.inner.ptr,
            row_capacity: self.row_capacity,
            col_capacity: 1,
        });
    }
}

impl<E: Entity> Col<E> {
    /// Returns an empty column of dimension `0`.
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: VecOwnImpl {
                ptr: into_copy::<E, _>(E::faer_map(E::UNIT, |()| NonNull::<E::Unit>::dangling())),
                len: 0,
            },
            row_capacity: 0,
            __marker: PhantomData,
        }
    }

    /// Returns a new column vector with 0 rows, with enough capacity to hold a maximum of
    /// `row_capacity` rows columns without reallocating. If `row_capacity` is `0`,
    /// the function will not allocate.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn with_capacity(row_capacity: usize) -> Self {
        let raw = ManuallyDrop::new(RawMat::<E>::new(row_capacity, 1));
        Self {
            inner: VecOwnImpl {
                ptr: raw.ptr,
                len: 0,
            },
            row_capacity: raw.row_capacity,
            __marker: PhantomData,
        }
    }

    /// Returns a new matrix with number of rows `nrows`, filled with the provided function.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn from_fn(nrows: usize, f: impl FnMut(usize) -> E) -> Self {
        let mut this = Self::new();
        this.resize_with(nrows, f);
        this
    }

    /// Returns a new matrix with number of rows `nrows`, filled with zeros.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn zeros(nrows: usize) -> Self {
        Self::from_fn(nrows, |_| unsafe { core::mem::zeroed() })
    }

    /// Returns a new matrix with number of rows `nrows`, filled with ones.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn ones(nrows: usize) -> Self
    where
        E: ComplexField,
    {
        Self::from_fn(nrows, |_| E::faer_one())
    }

    /// Returns a new matrix with number of rows `nrows`, filled with a constant value.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn full(nrows: usize, constant: E) -> Self
    where
        E: ComplexField,
    {
        Self::from_fn(nrows, |_| constant)
    }

    /// Returns the number of rows of the column.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.inner.len
    }
    /// Returns the number of columns of the column. This is always equal to `1`.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        1
    }

    /// Set the dimensions of the matrix.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `nrows < self.row_capacity()`.
    /// * The elements that were previously out of bounds but are now in bounds must be
    /// initialized.
    #[inline]
    pub unsafe fn set_nrows(&mut self, nrows: usize) {
        self.inner.len = nrows;
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

    /// Returns the row capacity, that is, the number of rows that the matrix is able to hold
    /// without needing to reallocate, excluding column insertions.
    #[inline]
    pub fn row_capacity(&self) -> usize {
        self.row_capacity
    }

    /// Returns the offset between the first elements of two successive rows in the matrix.
    /// Always returns `1` since the matrix is column major.
    #[inline]
    pub fn row_stride(&self) -> isize {
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
    pub fn ptr_at(&self, row: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().ptr_at(row)
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at_mut(&mut self, row: usize) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().ptr_at_mut(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(&self, row: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().ptr_at_unchecked(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_mut_unchecked(&mut self, row: usize) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().ptr_at_mut_unchecked(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(&self, row: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().overflowing_ptr_at(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at_mut(&mut self, row: usize) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().overflowing_ptr_at_mut(row)
    }

    /// Returns raw pointers to the element at the given index, assuming the provided index
    /// is within the size of the vector.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(&self, row: usize) -> GroupFor<E, *const E::Unit> {
        self.as_ref().ptr_inbounds_at(row)
    }

    /// Returns raw pointers to the element at the given index, assuming the provided index
    /// is within the size of the vector.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(&mut self, row: usize) -> GroupFor<E, *mut E::Unit> {
        self.as_mut().ptr_inbounds_at_mut(row)
    }

    /// Splits the column vector at the given index into two parts and
    /// returns an array of each subvector, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row <= self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_unchecked(&self, row: usize) -> (ColRef<'_, E>, ColRef<'_, E>) {
        self.as_ref().split_at_unchecked(row)
    }

    /// Splits the column vector at the given index into two parts and
    /// returns an array of each subvector, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row <= self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_mut_unchecked(&mut self, row: usize) -> (ColMut<'_, E>, ColMut<'_, E>) {
        self.as_mut().split_at_mut_unchecked(row)
    }

    /// Splits the column vector at the given index into two parts and
    /// returns an array of each subvector, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row <= self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at(&self, row: usize) -> (ColRef<'_, E>, ColRef<'_, E>) {
        self.as_ref().split_at(row)
    }

    /// Splits the column vector at the given index into two parts and
    /// returns an array of each subvector, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row <= self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_mut(&mut self, row: usize) -> (ColMut<'_, E>, ColMut<'_, E>) {
        self.as_mut().split_at_mut(row)
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col(&self) -> GroupFor<E, &[E::Unit]> {
        self.as_ref().try_get_contiguous_col()
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col_mut(&mut self) -> GroupFor<E, &mut [E::Unit]> {
        self.as_mut().try_get_contiguous_col_mut()
    }

    #[cold]
    fn do_reserve_exact(&mut self, mut new_row_capacity: usize) {
        if is_vectorizable::<E::Unit>() {
            let align_factor = align_for::<E::Unit>() / core::mem::size_of::<E::Unit>();
            new_row_capacity = new_row_capacity
                .msrv_checked_next_multiple_of(align_factor)
                .unwrap();
        }

        let nrows = self.inner.len;
        let old_row_capacity = self.row_capacity;

        let mut this = ManuallyDrop::new(core::mem::take(self));
        {
            let mut this_group = E::faer_map(from_copy::<E, _>(this.inner.ptr), |ptr| MatUnit {
                raw: RawMatUnit {
                    ptr,
                    row_capacity: old_row_capacity,
                    col_capacity: 1,
                },
                nrows,
                ncols: 1,
            });

            E::faer_map(E::faer_as_mut(&mut this_group), |mat_unit| {
                mat_unit.do_reserve_exact(new_row_capacity, 1);
            });

            let this_group = E::faer_map(this_group, ManuallyDrop::new);
            this.inner.ptr =
                into_copy::<E, _>(E::faer_map(this_group, |mat_unit| mat_unit.raw.ptr));
            this.row_capacity = new_row_capacity;
        }
        *self = ManuallyDrop::into_inner(this);
    }

    /// Reserves the minimum capacity for `row_capacity` rows without reallocating. Does nothing if
    /// the capacity is already sufficient.
    ///
    /// # Panics
    /// The function panics if the new total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn reserve_exact(&mut self, row_capacity: usize) {
        if self.row_capacity() >= row_capacity {
            // do nothing
        } else if core::mem::size_of::<E::Unit>() == 0 {
            self.row_capacity = self.row_capacity().max(row_capacity);
        } else {
            self.do_reserve_exact(row_capacity);
        }
    }

    unsafe fn insert_block_with<F: FnMut(usize) -> E>(
        &mut self,
        f: &mut F,
        row_start: usize,
        row_end: usize,
    ) {
        debug_assert!(row_start <= row_end);

        let ptr = self.as_ptr_mut();

        for i in row_start..row_end {
            // SAFETY:
            // * pointer to element at index (i, j), which is within the
            // allocation since we reserved enough space
            // * writing to this memory region is sound since it is properly
            // aligned and valid for writes
            let ptr_ij = E::faer_map(E::faer_copy(&ptr), |ptr| ptr.add(i));
            let value = E::faer_into_units(f(i));

            E::faer_map(E::faer_zip(ptr_ij, value), |(ptr_ij, value)| {
                core::ptr::write(ptr_ij, value)
            });
        }
    }

    fn erase_last_rows(&mut self, new_nrows: usize) {
        let old_nrows = self.nrows();
        debug_assert!(new_nrows <= old_nrows);
        self.inner.len = new_nrows;
    }

    unsafe fn insert_last_rows_with<F: FnMut(usize) -> E>(&mut self, f: &mut F, new_nrows: usize) {
        let old_nrows = self.nrows();

        debug_assert!(new_nrows > old_nrows);

        self.insert_block_with(f, old_nrows, new_nrows);
        self.inner.len = new_nrows;
    }

    /// Resizes the vector in-place so that the new number of rows is `new_nrows`.
    /// New elements are created with the given function `f`, so that elements at index `i`
    /// are created by calling `f(i)`.
    pub fn resize_with(&mut self, new_nrows: usize, f: impl FnMut(usize) -> E) {
        let mut f = f;
        let old_nrows = self.nrows();

        if new_nrows <= old_nrows {
            self.erase_last_rows(new_nrows);
        } else {
            self.reserve_exact(new_nrows);
            unsafe {
                self.insert_last_rows_with(&mut f, new_nrows);
            }
        }
    }

    /// Truncates the matrix so that its new number of rows is `new_nrows`.  
    /// The new dimension must be smaller than the current dimension of the vector.
    ///
    /// # Panics
    /// - Panics if `new_nrows > self.nrows()`.
    #[inline]
    pub fn truncate(&mut self, new_nrows: usize) {
        assert!(new_nrows <= self.nrows());
        self.resize_with(new_nrows, |_| unreachable!());
    }

    /// Returns a reference to a slice over the column.
    #[inline]
    #[track_caller]
    pub fn as_slice(&self) -> GroupFor<E, &[E::Unit]> {
        let nrows = self.nrows();
        let ptr = self.as_ref().as_ptr();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, nrows) },
        )
    }

    /// Returns a mutable reference to a slice over the column.
    #[inline]
    #[track_caller]
    pub fn as_slice_mut(&mut self) -> GroupFor<E, &mut [E::Unit]> {
        let nrows = self.nrows();
        let ptr = self.as_ptr_mut();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, nrows) },
        )
    }

    /// Returns a mutable reference to a potentially uninitialized slice over the column.
    ///
    /// # Safety
    /// If uninit data is written to the slice, it must not be later read.
    #[inline]
    #[track_caller]
    pub unsafe fn as_uninit_slice_mut(&mut self) -> GroupFor<E, &mut [MaybeUninit<E::Unit>]> {
        let nrows = self.nrows();
        let ptr = self.as_ptr_mut();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr as _, nrows) },
        )
    }

    /// Returns a view over the vector.
    #[inline]
    pub fn as_ref(&self) -> ColRef<'_, E> {
        unsafe { super::from_raw_parts(self.as_ptr(), self.nrows(), 1) }
    }

    /// Returns a mutable view over the vector.
    #[inline]
    pub fn as_mut(&mut self) -> ColMut<'_, E> {
        unsafe { super::from_raw_parts_mut(self.as_ptr_mut(), self.nrows(), 1) }
    }

    /// Returns references to the element at the given index, or submatrices if `row` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline]
    pub unsafe fn get_unchecked<RowRange>(
        &self,
        row: RowRange,
    ) -> <ColRef<'_, E> as ColIndex<RowRange>>::Target
    where
        for<'a> ColRef<'a, E>: ColIndex<RowRange>,
    {
        self.as_ref().get_unchecked(row)
    }

    /// Returns references to the element at the given index, or submatrices if `row` is a range,
    /// with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline]
    pub fn get<RowRange>(&self, row: RowRange) -> <ColRef<'_, E> as ColIndex<RowRange>>::Target
    where
        for<'a> ColRef<'a, E>: ColIndex<RowRange>,
    {
        self.as_ref().get(row)
    }

    /// Returns mutable references to the element at the given index, or submatrices if
    /// `row` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline]
    pub unsafe fn get_mut_unchecked<RowRange>(
        &mut self,
        row: RowRange,
    ) -> <ColMut<'_, E> as ColIndex<RowRange>>::Target
    where
        for<'a> ColMut<'a, E>: ColIndex<RowRange>,
    {
        self.as_mut().get_mut_unchecked(row)
    }

    /// Returns mutable references to the element at the given index, or submatrices if
    /// `row` is a range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline]
    pub fn get_mut<RowRange>(
        &mut self,
        row: RowRange,
    ) -> <ColMut<'_, E> as ColIndex<RowRange>>::Target
    where
        for<'a> ColMut<'a, E>: ColIndex<RowRange>,
    {
        self.as_mut().get_mut(row)
    }

    /// Reads the value of the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, row: usize) -> E {
        self.as_ref().read_unchecked(row)
    }

    /// Reads the value of the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, row: usize) -> E {
        self.as_ref().read(row)
    }

    /// Writes the value to the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, row: usize, value: E) {
        self.as_mut().write_unchecked(row, value);
    }

    /// Writes the value to the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, row: usize, value: E) {
        self.as_mut().write(row, value);
    }

    /// Copies the values from `other` into `self`.
    #[inline(always)]
    #[track_caller]
    pub fn copy_from<ViewE: Conjugate<Canonical = E>>(&mut self, other: impl AsColRef<ViewE>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity, ViewE: Conjugate<Canonical = E>>(
            this: &mut Col<E>,
            other: ColRef<'_, ViewE>,
        ) {
            let mut mat = Col::<E>::new();
            mat.resize_with(
                other.nrows(),
                #[inline(always)]
                |row| unsafe { other.read_unchecked(row).canonicalize() },
            );
            *this = mat;
        }
        implementation(self, other.as_col_ref());
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
    pub fn transpose(&self) -> RowRef<'_, E> {
        self.as_ref().transpose()
    }

    /// Returns a view over the transpose of `self`.
    #[inline]
    pub fn transpose_mut(&mut self) -> RowMut<'_, E> {
        self.as_mut().transpose_mut()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(&self) -> ColRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate_mut(&mut self) -> ColMut<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_mut().conjugate_mut()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize(&self) -> (ColRef<'_, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        self.as_ref().canonicalize()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize_mut(&mut self) -> (ColMut<'_, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        self.as_mut().canonicalize_mut()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(&self) -> RowRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint_mut(&mut self) -> RowMut<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_mut().adjoint_mut()
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows(&self) -> ColRef<'_, E> {
        self.as_ref().reverse_rows()
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows_mut(&mut self) -> ColMut<'_, E> {
        self.as_mut().reverse_rows_mut()
    }

    /// Returns a view over the subvector starting at row `row_start`, and with number of rows
    /// `nrows`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `nrows <= self.nrows() - row_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn subrows_unchecked(&self, row_start: usize, nrows: usize) -> ColRef<'_, E> {
        self.as_ref().subrows_unchecked(row_start, nrows)
    }

    /// Returns a view over the subvector starting at row `row_start`, and with number of rows
    /// `nrows`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `nrows <= self.nrows() - row_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn subrows_mut_unchecked(
        &mut self,
        row_start: usize,
        nrows: usize,
    ) -> ColMut<'_, E> {
        self.as_mut().subrows_mut_unchecked(row_start, nrows)
    }

    /// Returns a view over the subvector starting at row `row_start`, and with number of rows
    /// `nrows`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `nrows <= self.nrows() - row_start`.
    #[track_caller]
    #[inline(always)]
    pub fn subrows(&self, row_start: usize, nrows: usize) -> ColRef<'_, E> {
        self.as_ref().subrows(row_start, nrows)
    }

    /// Returns a view over the subvector starting at row `row_start`, and with number of rows
    /// `nrows`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `nrows <= self.nrows() - row_start`.
    #[track_caller]
    #[inline(always)]
    pub fn subrows_mut(&mut self, row_start: usize, nrows: usize) -> ColMut<'_, E> {
        self.as_mut().subrows_mut(row_start, nrows)
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal(&self) -> DiagRef<'_, E> {
        self.as_ref().column_vector_as_diagonal()
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal_mut(&mut self) -> DiagMut<'_, E> {
        self.as_mut().column_vector_as_diagonal_mut()
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_into_diagonal(self) -> Diag<E> {
        Diag { inner: self }
    }

    /// Returns an owning [`Col`] of the data
    #[inline]
    pub fn to_owned(&self) -> Col<E::Canonical>
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

    /// Returns the column as a contiguous slice if its row stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice(&self) -> Option<GroupFor<E, &[E::Unit]>> {
        self.as_ref().try_as_slice()
    }

    /// Returns the column as a contiguous slice if its row stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice_mut(&mut self) -> Option<GroupFor<E, &mut [E::Unit]>> {
        self.as_mut().try_as_slice_mut()
    }

    /// Returns the column as a contiguous potentially uninitialized slice if its row stride is
    /// equal to `1`.
    ///
    /// # Safety
    /// If uninit data is written to the slice, it must not be later read.
    pub unsafe fn try_as_uninit_slice_mut(
        &mut self,
    ) -> Option<GroupFor<E, &mut [MaybeUninit<E::Unit>]>> {
        self.as_mut().try_as_uninit_slice_mut()
    }

    /// Kroneckor product of `self` and `rhs`.
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

    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn const_cast(&self) -> ColMut<'_, E> {
        self.as_ref().const_cast()
    }
}

impl<E: Entity> Default for Col<E> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Entity> Clone for Col<E> {
    fn clone(&self) -> Self {
        let this = self.as_ref();
        unsafe {
            Self::from_fn(self.nrows(), |i| {
                E::faer_from_units(E::faer_deref(this.get_unchecked(i)))
            })
        }
    }
}

impl<E: Entity> As2D<E> for Col<E> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (*self).as_ref().as_2d()
    }
}

impl<E: Entity> As2DMut<E> for Col<E> {
    #[inline]
    fn as_2d_mut(&mut self) -> MatMut<'_, E> {
        (*self).as_mut().as_2d_mut()
    }
}

impl<E: Entity> AsColRef<E> for Col<E> {
    #[inline]
    fn as_col_ref(&self) -> ColRef<'_, E> {
        (*self).as_ref()
    }
}
impl<E: Entity> AsColMut<E> for Col<E> {
    #[inline]
    fn as_col_mut(&mut self) -> ColMut<'_, E> {
        (*self).as_mut()
    }
}

impl<E: Entity> core::fmt::Debug for Col<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<E: SimpleEntity> core::ops::Index<usize> for Col<E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, row: usize) -> &E {
        self.as_ref().get(row)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<usize> for Col<E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, row: usize) -> &mut E {
        self.as_mut().get_mut(row)
    }
}

impl<E: Conjugate> ColBatch<E> for Col<E> {
    type Owned = Col<E::Canonical>;

    #[inline]
    #[track_caller]
    fn new_owned_zeros(nrows: usize, ncols: usize) -> Self::Owned {
        assert!(ncols == 1);
        Col::zeros(nrows)
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

impl<E: Conjugate> ColBatchMut<E> for Col<E> {}
