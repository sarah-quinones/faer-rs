use super::*;
use crate::{
    debug_assert,
    diag::{Diag, DiagMut, DiagRef},
    iter,
    mat::matalloc::{align_for, is_vectorizable, MatUnit, RawMat, RawMatUnit},
    row::{RowMut, RowRef},
    utils::DivCeil,
    Idx, IdxInc, Unbind,
};
use core::mem::ManuallyDrop;

/// Heap allocated resizable column vector.
///
/// # Note
///
/// The memory layout of `Col` is guaranteed to be column-major, meaning that it has a row stride
/// of `1`.
#[repr(C)]
pub struct Col<E: Entity, R: Shape = usize> {
    inner: VecOwnImpl<E, R>,
    row_capacity: usize,
    __marker: PhantomData<E>,
}

impl<E: Entity, R: Shape> Drop for Col<E, R> {
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
}

impl<E: Entity, R: Shape> Col<E, R> {
    /// Returns a new matrix with number of rows `nrows`, filled with the provided function.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn from_fn(nrows: R, f: impl FnMut(Idx<R>) -> E) -> Self {
        let mut f = f;
        let mut this = Col::<E>::new();
        this.resize_with(nrows.unbound(), |i| unsafe { f(Idx::<R>::new_unbound(i)) });
        let this = core::mem::ManuallyDrop::new(this);

        Self {
            inner: VecOwnImpl {
                ptr: this.inner.ptr,
                len: nrows,
            },
            row_capacity: this.row_capacity,
            __marker: PhantomData,
        }
    }

    /// Returns a new matrix with number of rows `nrows`, filled with zeros.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn zeros(nrows: R) -> Self {
        Self::from_fn(nrows, |_| unsafe { core::mem::zeroed() })
    }

    /// Returns a new matrix with number of rows `nrows`, filled with ones.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn ones(nrows: R) -> Self
    where
        E: ComplexField,
    {
        Self::full(nrows, E::faer_one())
    }

    /// Returns a new matrix with number of rows `nrows`, filled with a constant value.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn full(nrows: R, constant: E) -> Self
    where
        E: ComplexField,
    {
        Self::from_fn(nrows, |_| constant)
    }

    /// Returns the number of rows of the column.
    #[inline(always)]
    pub fn nrows(&self) -> R {
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
    pub unsafe fn set_nrows(&mut self, nrows: R) {
        self.inner.len = nrows;
    }

    /// Returns a pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr(&self) -> PtrConst<E> {
        E::faer_map(from_copy::<E, _>(self.inner.ptr), |ptr| {
            ptr.as_ptr() as *const E::Unit
        })
    }

    /// Returns a mutable pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr_mut(&mut self) -> PtrMut<E> {
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

    /// Returns the input column with dynamic shape.
    #[inline]
    pub fn as_dyn(&self) -> ColRef<'_, E> {
        self.as_ref().as_dyn()
    }

    /// Returns the input column with dynamic shape.
    #[inline]
    pub fn as_dyn_mut(&mut self) -> ColMut<'_, E> {
        self.as_mut().as_dyn_mut()
    }

    /// Returns the input column with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape<V: Shape>(&self, nrows: V) -> ColRef<'_, E, V> {
        self.as_ref().as_shape(nrows)
    }

    /// Returns the input column with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape_mut<V: Shape>(&mut self, nrows: V) -> ColMut<'_, E, V> {
        self.as_mut().as_shape_mut(nrows)
    }

    /// Returns the input column with the given shape after checking that it matches the
    /// current shape.
    pub fn into_shape<V: Shape>(self, nrows: V) -> Col<E, V> {
        crate::assert!(nrows.unbound() == self.nrows().unbound());
        let this = ManuallyDrop::new(self);

        Col {
            inner: VecOwnImpl {
                ptr: this.inner.ptr,
                len: nrows,
            },
            row_capacity: this.row_capacity,
            __marker: PhantomData,
        }
    }

    /// Returns `self` as a matrix view.
    #[inline(always)]
    pub fn as_2d(&self) -> MatRef<'_, E, R, usize> {
        self.as_ref().as_2d()
    }

    /// Returns `self` as a mutable matrix view.
    #[inline(always)]
    pub fn as_2d_mut(&mut self) -> MatMut<'_, E, R, usize> {
        self.as_mut().as_2d_mut()
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at(&self, row: usize) -> PtrConst<E> {
        self.as_ref().ptr_at(row)
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at_mut(&mut self, row: usize) -> PtrMut<E> {
        self.as_mut().ptr_at_mut(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(&self, row: usize) -> PtrConst<E> {
        self.as_ref().ptr_at_unchecked(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_mut_unchecked(&mut self, row: usize) -> PtrMut<E> {
        self.as_mut().ptr_at_mut_unchecked(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(&self, row: IdxInc<R>) -> PtrConst<E> {
        self.as_ref().overflowing_ptr_at(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at_mut(&mut self, row: IdxInc<R>) -> PtrMut<E> {
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
    pub unsafe fn ptr_inbounds_at(&self, row: Idx<R>) -> PtrConst<E> {
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
    pub unsafe fn ptr_inbounds_at_mut(&mut self, row: Idx<R>) -> PtrMut<E> {
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
    pub unsafe fn split_at_unchecked(&self, row: IdxInc<R>) -> (ColRef<'_, E>, ColRef<'_, E>) {
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
    pub unsafe fn split_at_mut_unchecked(
        &mut self,
        row: IdxInc<R>,
    ) -> (ColMut<'_, E>, ColMut<'_, E>) {
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
    pub fn split_at(&self, row: IdxInc<R>) -> (ColRef<'_, E>, ColRef<'_, E>) {
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
    pub fn split_at_mut(&mut self, row: IdxInc<R>) -> (ColMut<'_, E>, ColMut<'_, E>) {
        self.as_mut().split_at_mut(row)
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col(&self) -> Slice<'_, E> {
        self.as_ref().try_get_contiguous_col()
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col_mut(&mut self) -> SliceMut<'_, E> {
        self.as_mut().try_get_contiguous_col_mut()
    }

    /// Reserves the minimum capacity for `row_capacity` rows without reallocating. Does nothing if
    /// the capacity is already sufficient.
    ///
    /// # Panics
    /// The function panics if the new total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn reserve_exact(&mut self, row_capacity: usize) {
        #[cold]
        fn do_reserve_exact<E: Entity>(self_: &mut Col<E>, mut new_row_capacity: usize) {
            if is_vectorizable::<E::Unit>() {
                let align_factor = align_for::<E::Unit>() / core::mem::size_of::<E::Unit>();
                new_row_capacity = new_row_capacity
                    .msrv_checked_next_multiple_of(align_factor)
                    .unwrap();
            }

            let nrows = self_.inner.len;
            let old_row_capacity = self_.row_capacity;

            let mut this = ManuallyDrop::new(core::mem::take(self_));
            {
                let mut this_group =
                    E::faer_map(from_copy::<E, _>(this.inner.ptr), |ptr| MatUnit {
                        raw: RawMatUnit {
                            ptr,
                            row_capacity: old_row_capacity,
                            col_capacity: 1,
                        },
                        nrows,
                        ncols: 1,
                    });

                E::faer_map(E::faer_as_mut(&mut this_group), |mat_unit| {
                    mat_unit.do_reserve_exact(new_row_capacity, 1, E::N_COMPONENTS <= 1);
                });

                let this_group = E::faer_map(this_group, ManuallyDrop::new);
                this.inner.ptr =
                    into_copy::<E, _>(E::faer_map(this_group, |mat_unit| mat_unit.raw.ptr));
                this.row_capacity = new_row_capacity;
            }
            *self_ = ManuallyDrop::into_inner(this);
        }

        if self.row_capacity() >= row_capacity {
            // do nothing
        } else if core::mem::size_of::<E::Unit>() == 0 {
            self.row_capacity = self.row_capacity().max(row_capacity);
        } else {
            let mut tmp = core::mem::ManuallyDrop::new(Col::<E> {
                inner: VecOwnImpl {
                    ptr: self.inner.ptr,
                    len: self.nrows().unbound(),
                },
                row_capacity: self.row_capacity,
                __marker: PhantomData,
            });

            struct AbortOnPanic;
            impl Drop for AbortOnPanic {
                fn drop(&mut self) {
                    panic!();
                }
            }
            let guard = AbortOnPanic;
            do_reserve_exact(&mut tmp, row_capacity);
            core::mem::forget(guard);
            self.row_capacity = tmp.row_capacity;
            self.inner.ptr = tmp.inner.ptr;
        }
    }

    unsafe fn insert_block_with<F: FnMut(Idx<R>) -> E>(
        &mut self,
        f: &mut F,
        row_start: IdxInc<R>,
        row_end: R,
    ) {
        debug_assert!(row_start <= row_end);

        let ptr = self.as_ptr_mut();

        for i in R::indices(row_start, row_end.end()) {
            // SAFETY:
            // * pointer to element at index (i, j), which is within the
            // allocation since we reserved enough space
            // * writing to this memory region is sound since it is properly
            // aligned and valid for writes
            let ptr_ij = E::faer_map(E::faer_copy(&ptr), |ptr| ptr.add(i.unbound()));
            let value = E::faer_into_units(f(i));

            E::faer_map(E::faer_zip(ptr_ij, value), |(ptr_ij, value)| {
                core::ptr::write(ptr_ij, value)
            });
        }
    }

    fn erase_last_rows(&mut self, new_nrows: R) {
        let old_nrows = self.nrows();
        debug_assert!(new_nrows <= old_nrows);
        self.inner.len = new_nrows;
    }

    unsafe fn insert_last_rows_with<F: FnMut(Idx<R>) -> E>(&mut self, f: &mut F, new_nrows: R) {
        let old_nrows = self.nrows();

        debug_assert!(new_nrows > old_nrows);

        self.insert_block_with(f, old_nrows.end(), new_nrows);
        self.inner.len = new_nrows;
    }

    /// Resizes the vector in-place so that the new number of rows is `new_nrows`.
    /// New elements are created with the given function `f`, so that elements at index `i`
    /// are created by calling `f(i)`.
    pub fn resize_with(&mut self, new_nrows: R, f: impl FnMut(Idx<R>) -> E) {
        let mut f = f;
        let old_nrows = self.nrows();

        if new_nrows <= old_nrows {
            self.erase_last_rows(new_nrows);
        } else {
            self.reserve_exact(new_nrows.unbound());
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
    pub fn truncate(&mut self, new_nrows: R) {
        assert!(new_nrows <= self.nrows());
        self.erase_last_rows(new_nrows);
    }

    /// Returns a reference to a slice over the column.
    #[inline]
    #[track_caller]
    pub fn as_slice(&self) -> Slice<'_, E> {
        let nrows = self.nrows().unbound();
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
    pub fn as_slice_mut(&mut self) -> SliceMut<'_, E> {
        let nrows = self.nrows().unbound();
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
    pub unsafe fn as_uninit_slice_mut(&mut self) -> UninitSliceMut<'_, E> {
        let nrows = self.nrows().unbound();
        let ptr = self.as_ptr_mut();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr as _, nrows) },
        )
    }

    /// Returns a view over the vector.
    #[inline]
    pub fn as_ref(&self) -> ColRef<'_, E, R> {
        unsafe { super::from_raw_parts(self.as_ptr(), self.nrows(), 1) }
    }

    /// Returns a mutable view over the vector.
    #[inline]
    pub fn as_mut(&mut self) -> ColMut<'_, E, R> {
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
    ) -> <ColRef<'_, E, R> as ColIndex<RowRange>>::Target
    where
        for<'a> ColRef<'a, E, R>: ColIndex<RowRange>,
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
    pub fn get<RowRange>(&self, row: RowRange) -> <ColRef<'_, E, R> as ColIndex<RowRange>>::Target
    where
        for<'a> ColRef<'a, E, R>: ColIndex<RowRange>,
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
    ) -> <ColMut<'_, E, R> as ColIndex<RowRange>>::Target
    where
        for<'a> ColMut<'a, E, R>: ColIndex<RowRange>,
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
    ) -> <ColMut<'_, E, R> as ColIndex<RowRange>>::Target
    where
        for<'a> ColMut<'a, E, R>: ColIndex<RowRange>,
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
    pub unsafe fn read_unchecked(&self, row: Idx<R>) -> E {
        self.as_ref().read_unchecked(row)
    }

    /// Reads the value of the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, row: Idx<R>) -> E {
        self.as_ref().read(row)
    }

    /// Writes the value to the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, row: Idx<R>, value: E) {
        self.as_mut().write_unchecked(row, value);
    }

    /// Writes the value to the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, row: Idx<R>, value: E) {
        self.as_mut().write(row, value);
    }

    /// Copies the values from `other` into `self`.
    #[inline(always)]
    #[track_caller]
    pub fn copy_from<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsColRef<ViewE, R = R>,
    ) {
        #[track_caller]
        #[inline(always)]
        fn implementation<R: Shape, E: Entity, ViewE: Conjugate<Canonical = E>>(
            this: &mut Col<E, R>,
            other: ColRef<'_, ViewE, R>,
        ) {
            if this.nrows() == other.nrows() {
                this.as_mut().copy_from(other);
            } else {
                if !R::IS_BOUND {
                    this.truncate(unsafe { R::new_unbound(0) });
                } else {
                    panic!()
                }
                this.resize_with(
                    other.nrows(),
                    #[inline(always)]
                    |row| unsafe { other.read_unchecked(row).canonicalize() },
                );
            }
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
    pub fn transpose(&self) -> RowRef<'_, E, R> {
        self.as_ref().transpose()
    }

    /// Returns a view over the transpose of `self`.
    #[inline]
    pub fn transpose_mut(&mut self) -> RowMut<'_, E, R> {
        self.as_mut().transpose_mut()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(&self) -> ColRef<'_, E::Conj, R>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate_mut(&mut self) -> ColMut<'_, E::Conj, R>
    where
        E: Conjugate,
    {
        self.as_mut().conjugate_mut()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize(&self) -> (ColRef<'_, E::Canonical, R>, Conj)
    where
        E: Conjugate,
    {
        self.as_ref().canonicalize()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize_mut(&mut self) -> (ColMut<'_, E::Canonical, R>, Conj)
    where
        E: Conjugate,
    {
        self.as_mut().canonicalize_mut()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(&self) -> RowRef<'_, E::Conj, R>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint_mut(&mut self) -> RowMut<'_, E::Conj, R>
    where
        E: Conjugate,
    {
        self.as_mut().adjoint_mut()
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows(&self) -> ColRef<'_, E, R> {
        self.as_ref().reverse_rows()
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows_mut(&mut self) -> ColMut<'_, E, R> {
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
    pub unsafe fn subrows_unchecked<V: Shape>(
        &self,
        row_start: IdxInc<R>,
        nrows: V,
    ) -> ColRef<'_, E, V> {
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
    pub unsafe fn subrows_mut_unchecked<V: Shape>(
        &mut self,
        row_start: IdxInc<R>,
        nrows: V,
    ) -> ColMut<'_, E, V> {
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
    pub fn subrows<V: Shape>(&self, row_start: IdxInc<R>, nrows: V) -> ColRef<'_, E, V> {
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
    pub fn subrows_mut<V: Shape>(&mut self, row_start: IdxInc<R>, nrows: V) -> ColMut<'_, E, V> {
        self.as_mut().subrows_mut(row_start, nrows)
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whose diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal(&self) -> DiagRef<'_, E, R> {
        self.as_ref().column_vector_as_diagonal()
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whose diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal_mut(&mut self) -> DiagMut<'_, E, R> {
        self.as_mut().column_vector_as_diagonal_mut()
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whose diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_into_diagonal(self) -> Diag<E, R> {
        Diag { inner: self }
    }

    /// Returns an owning [`Col`] of the data
    #[inline]
    pub fn to_owned(&self) -> Col<E::Canonical, R>
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
    pub fn try_as_slice(&self) -> Option<Slice<'_, E>> {
        self.as_ref().try_as_slice()
    }

    /// Returns the column as a contiguous slice if its row stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice_mut(&mut self) -> Option<SliceMut<'_, E>> {
        self.as_mut().try_as_slice_mut()
    }

    /// Returns the column as a contiguous potentially uninitialized slice if its row stride is
    /// equal to `1`.
    ///
    /// # Safety
    /// If uninit data is written to the slice, it must not be later read.
    pub unsafe fn try_as_uninit_slice_mut(&mut self) -> Option<UninitSliceMut<'_, E>> {
        self.as_mut().try_as_uninit_slice_mut()
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
        self.as_ref().kron(rhs)
    }

    /// Returns a reference to the first element and a view over the remaining ones if the column is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_first(&self) -> Option<(Ref<'_, E>, ColRef<'_, E>)> {
        self.as_ref().split_first()
    }

    /// Returns a reference to the last element and a view over the remaining ones if the column is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_last(&self) -> Option<(Ref<'_, E>, ColRef<'_, E>)> {
        self.as_ref().split_last()
    }

    /// Returns a reference to the first element and a view over the remaining ones if the column is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_first_mut(&mut self) -> Option<(Mut<'_, E>, ColMut<'_, E>)> {
        self.as_mut().split_first_mut()
    }

    /// Returns a reference to the last element and a view over the remaining ones if the column is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_last_mut(&mut self) -> Option<(Mut<'_, E>, ColMut<'_, E>)> {
        self.as_mut().split_last_mut()
    }

    /// Returns an iterator over the elements of the column.
    #[inline]
    pub fn iter(&self) -> iter::ElemIter<'_, E> {
        self.as_ref().iter()
    }

    /// Returns an iterator over the elements of the column.
    #[inline]
    pub fn iter_mut(&mut self) -> iter::ElemIterMut<'_, E> {
        self.as_mut().iter_mut()
    }

    /// Returns an iterator that provides successive chunks of the elements of this column, with
    /// each having at most `chunk_size` elements.
    #[inline]
    #[track_caller]
    pub fn chunks(&self, chunk_size: usize) -> iter::ColElemChunks<'_, E> {
        self.as_ref().chunks(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// column.
    #[inline]
    #[track_caller]
    pub fn partition(&self, count: usize) -> iter::ColElemPartition<'_, E> {
        self.as_ref().partition(count)
    }

    /// Returns an iterator that provides successive chunks of the elements of this column, with
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
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColRef<'_, E>> {
        self.as_ref().par_chunks(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// column.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_partition(
        &self,
        count: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColRef<'_, E>> {
        self.as_ref().par_partition(count)
    }

    /// Returns an iterator that provides successive chunks of the elements of this column, with
    /// each having at most `chunk_size` elements.
    #[inline]
    #[track_caller]
    pub fn chunks_mut(&mut self, chunk_size: usize) -> iter::ColElemChunksMut<'_, E> {
        self.as_mut().chunks_mut(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// column.
    #[inline]
    #[track_caller]
    pub fn partition_mut(&mut self, count: usize) -> iter::ColElemPartitionMut<'_, E> {
        self.as_mut().partition_mut(count)
    }

    /// Returns an iterator that provides successive chunks of the elements of this column, with
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
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColMut<'_, E>> {
        self.as_mut().par_chunks_mut(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// column.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_partition_mut(
        &mut self,
        count: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = ColMut<'_, E>> {
        self.as_mut().par_partition_mut(count)
    }

    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn const_cast(&self) -> ColMut<'_, E, R> {
        self.as_ref().const_cast()
    }
}

impl<E: Entity> Default for Col<E> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Entity, R: Shape> Clone for Col<E, R> {
    fn clone(&self) -> Self {
        let this = self.as_ref();
        unsafe {
            Self::from_fn(self.nrows(), |i| {
                E::faer_from_units(E::faer_deref(this.at_unchecked(i)))
            })
        }
    }
    fn clone_from(&mut self, other: &Self) {
        if self.nrows() == other.nrows() {
            crate::zipped!(__rw, self, other)
                .for_each(|crate::unzipped!(mut dst, src)| dst.write(src.read()));
        } else {
            if !R::IS_BOUND {
                self.truncate(unsafe { R::new_unbound(0) });
            } else {
                panic!()
            }
            self.resize_with(
                other.nrows(),
                #[inline(always)]
                |row| unsafe { other.read_unchecked(row) },
            );
        }
    }
}

impl<E: Entity, R: Shape> As2D<E> for Col<E, R> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (*self).as_ref().as_2d().as_dyn()
    }
}

impl<E: Entity, R: Shape> As2DMut<E> for Col<E, R> {
    #[inline]
    fn as_2d_mut(&mut self) -> MatMut<'_, E> {
        (*self).as_mut().as_2d_mut().as_dyn_mut()
    }
}

impl<E: Entity, R: Shape> AsColRef<E> for Col<E, R> {
    type R = R;

    #[inline]
    fn as_col_ref(&self) -> ColRef<'_, E, R> {
        (*self).as_ref()
    }
}
impl<E: Entity, R: Shape> AsColMut<E> for Col<E, R> {
    #[inline]
    fn as_col_mut(&mut self) -> ColMut<'_, E, R> {
        (*self).as_mut()
    }
}

impl<E: Entity, R: Shape> core::fmt::Debug for Col<E, R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<E: SimpleEntity, R: Shape> core::ops::Index<Idx<R>> for Col<E, R> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, row: Idx<R>) -> &E {
        self.as_ref().at(row)
    }
}

impl<E: SimpleEntity, R: Shape> core::ops::IndexMut<Idx<R>> for Col<E, R> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, row: Idx<R>) -> &mut E {
        self.as_mut().at_mut(row)
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
