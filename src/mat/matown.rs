use super::*;
use crate::{
    assert, debug_assert,
    diag::DiagRef,
    mat::matalloc::{align_for, is_vectorizable, MatUnit, RawMat, RawMatUnit},
    utils::DivCeil,
};
use core::mem::ManuallyDrop;

/// Heap allocated resizable matrix, similar to a 2D [`Vec`].
///
/// # Note
///
/// The memory layout of `Mat` is guaranteed to be column-major, meaning that it has a row stride
/// of `1`, and an unspecified column stride that can be queried with [`Mat::col_stride`].
///
/// This implies that while each individual column is stored contiguously in memory, the matrix as
/// a whole may not necessarily be contiguous. The implementation may add padding at the end of
/// each column when overaligning each column can provide a performance gain.
///
/// Let us consider a 3×4 matrix
///
/// ```notcode
///  0 │ 3 │ 6 │  9
/// ───┼───┼───┼───
///  1 │ 4 │ 7 │ 10
/// ───┼───┼───┼───
///  2 │ 5 │ 8 │ 11
/// ```
/// The memory representation of the data held by such a matrix could look like the following:
///
/// ```notcode
/// 0 1 2 X 3 4 5 X 6 7 8 X 9 10 11 X
/// ```
///
/// where X represents padding elements.
#[repr(C)]
pub struct Mat<E: Entity> {
    inner: MatOwnImpl<E>,
    row_capacity: usize,
    col_capacity: usize,
    __marker: PhantomData<E>,
}

impl<E: Entity> Drop for Mat<E> {
    #[inline]
    fn drop(&mut self) {
        drop(RawMat::<E> {
            ptr: self.inner.ptr,
            row_capacity: self.row_capacity,
            col_capacity: self.col_capacity,
        });
    }
}

impl<E: Entity> Mat<E> {
    /// Returns an empty matrix of dimension `0×0`.
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: MatOwnImpl {
                ptr: into_copy::<E, _>(E::faer_map(E::UNIT, |()| NonNull::<E::Unit>::dangling())),
                nrows: 0,
                ncols: 0,
            },
            row_capacity: 0,
            col_capacity: 0,
            __marker: PhantomData,
        }
    }

    /// Returns a new matrix with dimensions `(0, 0)`, with enough capacity to hold a maximum of
    /// `row_capacity` rows and `col_capacity` columns without reallocating. If either is `0`,
    /// the matrix will not allocate.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn with_capacity(row_capacity: usize, col_capacity: usize) -> Self {
        let raw = ManuallyDrop::new(RawMat::<E>::new(row_capacity, col_capacity));
        Self {
            inner: MatOwnImpl {
                ptr: raw.ptr,
                nrows: 0,
                ncols: 0,
            },
            row_capacity: raw.row_capacity,
            col_capacity: raw.col_capacity,
            __marker: PhantomData,
        }
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with the provided function.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> E) -> Self {
        let mut this = Self::new();
        this.resize_with(nrows, ncols, f);
        this
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with zeros.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self::from_fn(nrows, ncols, |_, _| unsafe { core::mem::zeroed() })
    }

    /// Returns a new matrix with dimensions `(nrows, ncols)`, filled with zeros, except the main
    /// diagonal which is filled with ones.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn identity(nrows: usize, ncols: usize) -> Self
    where
        E: ComplexField,
    {
        let mut matrix = Self::zeros(nrows, ncols);
        matrix
            .as_mut()
            .diagonal_mut()
            .column_vector_mut()
            .fill(E::faer_one());
        matrix
    }

    /// Returns the number of rows of the matrix.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.inner.nrows
    }
    /// Returns the number of columns of the matrix.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.inner.ncols
    }

    /// Set the dimensions of the matrix.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `nrows < self.row_capacity()`.
    /// * `ncols < self.col_capacity()`.
    /// * The elements that were previously out of bounds but are now in bounds must be
    /// initialized.
    #[inline]
    pub unsafe fn set_dims(&mut self, nrows: usize, ncols: usize) {
        self.inner.nrows = nrows;
        self.inner.ncols = ncols;
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

    /// Returns the column capacity, that is, the number of columns that the matrix is able to hold
    /// without needing to reallocate, excluding row insertions.
    #[inline]
    pub fn col_capacity(&self) -> usize {
        self.col_capacity
    }

    /// Returns the offset between the first elements of two successive rows in the matrix.
    /// Always returns `1` since the matrix is column major.
    #[inline]
    pub fn row_stride(&self) -> isize {
        1
    }

    /// Returns the offset between the first elements of two successive columns in the matrix.
    #[inline]
    pub fn col_stride(&self) -> isize {
        self.row_capacity() as isize
    }

    #[cold]
    fn do_reserve_exact(&mut self, mut new_row_capacity: usize, new_col_capacity: usize) {
        if is_vectorizable::<E::Unit>() {
            let align_factor = align_for::<E::Unit>() / core::mem::size_of::<E::Unit>();
            new_row_capacity = new_row_capacity
                .msrv_checked_next_multiple_of(align_factor)
                .unwrap();
        }

        let nrows = self.inner.nrows;
        let ncols = self.inner.ncols;
        let old_row_capacity = self.row_capacity;
        let old_col_capacity = self.col_capacity;

        let mut this = ManuallyDrop::new(core::mem::take(self));
        {
            let mut this_group = E::faer_map(from_copy::<E, _>(this.inner.ptr), |ptr| MatUnit {
                raw: RawMatUnit {
                    ptr,
                    row_capacity: old_row_capacity,
                    col_capacity: old_col_capacity,
                },
                nrows,
                ncols,
            });

            E::faer_map(E::faer_as_mut(&mut this_group), |mat_unit| {
                mat_unit.do_reserve_exact(new_row_capacity, new_col_capacity);
            });

            let this_group = E::faer_map(this_group, ManuallyDrop::new);
            this.inner.ptr =
                into_copy::<E, _>(E::faer_map(this_group, |mat_unit| mat_unit.raw.ptr));
            this.row_capacity = new_row_capacity;
            this.col_capacity = new_col_capacity;
        }
        *self = ManuallyDrop::into_inner(this);
    }

    /// Reserves the minimum capacity for `row_capacity` rows and `col_capacity`
    /// columns without reallocating. Does nothing if the capacity is already sufficient.
    ///
    /// # Panics
    /// The function panics if the new total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn reserve_exact(&mut self, row_capacity: usize, col_capacity: usize) {
        if self.row_capacity() >= row_capacity && self.col_capacity() >= col_capacity {
            // do nothing
        } else if core::mem::size_of::<E::Unit>() == 0 {
            self.row_capacity = self.row_capacity().max(row_capacity);
            self.col_capacity = self.col_capacity().max(col_capacity);
        } else {
            self.do_reserve_exact(row_capacity, col_capacity);
        }
    }

    unsafe fn insert_block_with<F: FnMut(usize, usize) -> E>(
        &mut self,
        f: &mut F,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) {
        debug_assert!(all(row_start <= row_end, col_start <= col_end));

        let ptr = self.as_ptr_mut();

        for j in col_start..col_end {
            let ptr_j = E::faer_map(E::faer_copy(&ptr), |ptr| {
                ptr.wrapping_offset(j as isize * self.col_stride())
            });

            for i in row_start..row_end {
                // SAFETY:
                // * pointer to element at index (i, j), which is within the
                // allocation since we reserved enough space
                // * writing to this memory region is sound since it is properly
                // aligned and valid for writes
                let ptr_ij = E::faer_map(E::faer_copy(&ptr_j), |ptr_j| ptr_j.add(i));
                let value = E::faer_into_units(f(i, j));

                E::faer_map(E::faer_zip(ptr_ij, value), |(ptr_ij, value)| {
                    core::ptr::write(ptr_ij, value)
                });
            }
        }
    }

    fn erase_last_cols(&mut self, new_ncols: usize) {
        let old_ncols = self.ncols();
        debug_assert!(new_ncols <= old_ncols);
        self.inner.ncols = new_ncols;
    }

    fn erase_last_rows(&mut self, new_nrows: usize) {
        let old_nrows = self.nrows();
        debug_assert!(new_nrows <= old_nrows);
        self.inner.nrows = new_nrows;
    }

    unsafe fn insert_last_cols_with<F: FnMut(usize, usize) -> E>(
        &mut self,
        f: &mut F,
        new_ncols: usize,
    ) {
        let old_ncols = self.ncols();

        debug_assert!(new_ncols > old_ncols);

        self.insert_block_with(f, 0, self.nrows(), old_ncols, new_ncols);
        self.inner.ncols = new_ncols;
    }

    unsafe fn insert_last_rows_with<F: FnMut(usize, usize) -> E>(
        &mut self,
        f: &mut F,
        new_nrows: usize,
    ) {
        let old_nrows = self.nrows();

        debug_assert!(new_nrows > old_nrows);

        self.insert_block_with(f, old_nrows, new_nrows, 0, self.ncols());
        self.inner.nrows = new_nrows;
    }

    /// Resizes the matrix in-place so that the new dimensions are `(new_nrows, new_ncols)`.
    /// New elements are created with the given function `f`, so that elements at indices `(i, j)`
    /// are created by calling `f(i, j)`.
    pub fn resize_with(
        &mut self,
        new_nrows: usize,
        new_ncols: usize,
        f: impl FnMut(usize, usize) -> E,
    ) {
        let mut f = f;
        let old_nrows = self.nrows();
        let old_ncols = self.ncols();

        if new_ncols <= old_ncols {
            self.erase_last_cols(new_ncols);
            if new_nrows <= old_nrows {
                self.erase_last_rows(new_nrows);
            } else {
                self.reserve_exact(new_nrows, new_ncols);
                unsafe {
                    self.insert_last_rows_with(&mut f, new_nrows);
                }
            }
        } else {
            if new_nrows <= old_nrows {
                self.erase_last_rows(new_nrows);
            } else {
                self.reserve_exact(new_nrows, new_ncols);
                unsafe {
                    self.insert_last_rows_with(&mut f, new_nrows);
                }
            }
            self.reserve_exact(new_nrows, new_ncols);
            unsafe {
                self.insert_last_cols_with(&mut f, new_ncols);
            }
        }
    }

    /// Truncates the matrix so that its new dimensions are `new_nrows` and `new_ncols`.  
    /// Both of the new dimensions must be smaller than or equal to the current dimensions.
    ///
    /// # Panics
    /// - Panics if `new_nrows > self.nrows()`.
    /// - Panics if `new_ncols > self.ncols()`.
    #[inline]
    pub fn truncate(&mut self, new_nrows: usize, new_ncols: usize) {
        assert!(all(new_nrows <= self.nrows(), new_ncols <= self.ncols()));
        self.resize_with(new_nrows, new_ncols, |_, _| unreachable!());
    }

    /// Returns a reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    pub fn col_as_slice(&self, col: usize) -> GroupFor<E, &[E::Unit]> {
        assert!(col < self.ncols());
        let nrows = self.nrows();
        let ptr = self.as_ref().ptr_at(0, col);
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, nrows) },
        )
    }

    /// Returns a mutable reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    pub fn col_as_slice_mut(&mut self, col: usize) -> GroupFor<E, &mut [E::Unit]> {
        assert!(col < self.ncols());
        let nrows = self.nrows();
        let ptr = self.as_mut().ptr_at_mut(0, col);
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, nrows) },
        )
    }

    /// Returns a reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    #[deprecated = "replaced by `Mat::col_as_slice`"]
    pub fn col_ref(&self, col: usize) -> GroupFor<E, &[E::Unit]> {
        self.col_as_slice(col)
    }

    /// Returns a mutable reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    #[deprecated = "replaced by `Mat::col_as_slice_mut`"]
    pub fn col_mut(&mut self, col: usize) -> GroupFor<E, &mut [E::Unit]> {
        self.col_as_slice_mut(col)
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, E> {
        unsafe {
            super::from_raw_parts(
                self.as_ptr(),
                self.nrows(),
                self.ncols(),
                1,
                self.col_stride(),
            )
        }
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> MatMut<'_, E> {
        unsafe {
            super::from_raw_parts_mut(
                self.as_ptr_mut(),
                self.nrows(),
                self.ncols(),
                1,
                self.col_stride(),
            )
        }
    }

    /// Returns references to the element at the given indices, or submatrices if either `row` or
    /// `col` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    #[track_caller]
    pub unsafe fn get_unchecked<RowRange, ColRange>(
        &self,
        row: RowRange,
        col: ColRange,
    ) -> <MatRef<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatRef<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_ref().get_unchecked(row, col)
    }

    /// Returns references to the element at the given indices, or submatrices if either `row` or
    /// `col` is a range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    #[track_caller]
    pub fn get<RowRange, ColRange>(
        &self,
        row: RowRange,
        col: ColRange,
    ) -> <MatRef<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatRef<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_ref().get(row, col)
    }

    /// Returns mutable references to the element at the given indices, or submatrices if either
    /// `row` or `col` is a range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    #[track_caller]
    pub unsafe fn get_mut_unchecked<RowRange, ColRange>(
        &mut self,
        row: RowRange,
        col: ColRange,
    ) -> <MatMut<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatMut<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_mut().get_mut_unchecked(row, col)
    }

    /// Returns mutable references to the element at the given indices, or submatrices if either
    /// `row` or `col` is a range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline]
    #[track_caller]
    pub fn get_mut<RowRange, ColRange>(
        &mut self,
        row: RowRange,
        col: ColRange,
    ) -> <MatMut<'_, E> as MatIndex<RowRange, ColRange>>::Target
    where
        for<'a> MatMut<'a, E>: MatIndex<RowRange, ColRange>,
    {
        self.as_mut().get_mut(row, col)
    }

    /// Reads the value of the element at the given indices.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, row: usize, col: usize) -> E {
        self.as_ref().read_unchecked(row, col)
    }

    /// Reads the value of the element at the given indices, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, row: usize, col: usize) -> E {
        self.as_ref().read(row, col)
    }

    /// Writes the value to the element at the given indices.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, row: usize, col: usize, value: E) {
        self.as_mut().write_unchecked(row, col, value);
    }

    /// Writes the value to the element at the given indices, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, row: usize, col: usize, value: E) {
        self.as_mut().write(row, col, value);
    }

    /// Copies the values from `other` into `self`.
    #[inline(always)]
    #[track_caller]
    pub fn copy_from<ViewE: Conjugate<Canonical = E>>(&mut self, other: impl AsMatRef<ViewE>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity, ViewE: Conjugate<Canonical = E>>(
            this: &mut Mat<E>,
            other: MatRef<'_, ViewE>,
        ) {
            let mut mat = Mat::<E>::new();
            mat.resize_with(
                other.nrows(),
                other.ncols(),
                #[inline(always)]
                |row, col| unsafe { other.read_unchecked(row, col).canonicalize() },
            );
            *this = mat;
        }
        implementation(self, other.as_mat_ref());
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
    pub fn transpose(&self) -> MatRef<'_, E> {
        self.as_ref().transpose()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(&self) -> MatRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(&self) -> MatRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
    }

    /// Returns a view over the diagonal of the matrix.
    #[inline]
    pub fn diagonal(&self) -> DiagRef<'_, E> {
        self.as_ref().diagonal()
    }

    /// Returns an owning [`Mat`] of the data
    #[inline]
    pub fn to_owned(&self) -> Mat<E::Canonical>
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
        crate::linalg::reductions::norm_max::norm_max((*self).as_ref())
    }

    /// Returns the L1 norm of `self`.
    #[inline]
    pub fn norm_l1(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.as_ref().norm_l1()
    }

    /// Returns the L2 norm of `self`.
    #[inline]
    pub fn norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.as_ref().norm_l2()
    }

    /// Returns the squared L2 norm of `self`.
    #[inline]
    pub fn squared_norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.as_ref().squared_norm_l2()
    }

    /// Returns the sum of `self`.
    #[inline]
    pub fn sum(&self) -> E
    where
        E: ComplexField,
    {
        crate::linalg::reductions::sum::sum((*self).as_ref())
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

    /// Returns an iterator that provides successive chunks of the columns of a view over this
    /// matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    #[inline]
    #[track_caller]
    pub fn col_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatRef<'_, E>> {
        self.as_ref().col_chunks(chunk_size)
    }

    /// Returns an iterator that provides successive chunks of the columns of a mutable view over
    /// this matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    #[inline]
    #[track_caller]
    pub fn col_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatMut<'_, E>> {
        self.as_mut().col_chunks_mut(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the columns of a view over
    /// this matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_col_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, E>> {
        self.as_ref().par_col_chunks(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the columns of a mutable view
    /// over this matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// columns.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_col_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, E>> {
        self.as_mut().par_col_chunks_mut(chunk_size)
    }

    /// Returns an iterator that provides successive chunks of the rows of a view over this
    /// matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn row_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatRef<'_, E>> {
        self.as_ref().row_chunks(chunk_size)
    }

    /// Returns an iterator that provides successive chunks of the rows of a mutable view over
    /// this matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn row_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + DoubleEndedIterator<Item = MatMut<'_, E>> {
        self.as_mut().row_chunks_mut(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the rows of a view over this
    /// matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_row_chunks(
        &self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatRef<'_, E>> {
        self.as_ref().par_row_chunks(chunk_size)
    }

    /// Returns a parallel iterator that provides successive chunks of the rows of a mutable view
    /// over this matrix, with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_row_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl '_ + rayon::iter::IndexedParallelIterator<Item = MatMut<'_, E>> {
        self.as_mut().par_row_chunks_mut(chunk_size)
    }
}

impl<E: Entity> Default for Mat<E> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Entity> Clone for Mat<E> {
    fn clone(&self) -> Self {
        let this = self.as_ref();
        unsafe {
            Self::from_fn(self.nrows(), self.ncols(), |i, j| {
                E::faer_from_units(E::faer_deref(this.get_unchecked(i, j)))
            })
        }
    }
}

impl<E: Entity> AsMatRef<E> for Mat<E> {
    #[inline]
    fn as_mat_ref(&self) -> MatRef<'_, E> {
        (*self).as_ref()
    }
}

impl<E: Entity> AsMatMut<E> for Mat<E> {
    #[inline]
    fn as_mat_mut(&mut self) -> MatMut<'_, E> {
        (*self).as_mut()
    }
}

impl<E: Entity> As2D<E> for Mat<E> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (*self).as_ref()
    }
}

impl<E: Entity> As2DMut<E> for Mat<E> {
    #[inline]
    fn as_2d_mut(&mut self) -> MatMut<'_, E> {
        (*self).as_mut()
    }
}

impl<E: Entity> core::fmt::Debug for Mat<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for Mat<E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &E {
        self.as_ref().get(row, col)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for Mat<E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut E {
        self.as_mut().get_mut(row, col)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::Matrix<E> for Mat<E> {
    #[inline]
    fn rows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn cols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn access(&self) -> matrixcompare_core::Access<'_, E> {
        matrixcompare_core::Access::Dense(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::DenseAccess<E> for Mat<E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

impl<E: Conjugate> ColBatch<E> for Mat<E> {
    type Owned = Mat<E::Canonical>;

    #[inline]
    #[track_caller]
    fn new_owned_zeros(nrows: usize, ncols: usize) -> Self::Owned {
        Mat::zeros(nrows, ncols)
    }

    #[inline]
    fn new_owned_copied(src: &Self) -> Self::Owned {
        src.to_owned()
    }

    #[inline]
    #[track_caller]
    fn resize_owned(owned: &mut Self::Owned, nrows: usize, ncols: usize) {
        owned.resize_with(nrows, ncols, |_, _| unsafe { core::mem::zeroed() });
    }
}

impl<E: Conjugate> RowBatch<E> for Mat<E> {
    type Owned = Mat<E::Canonical>;

    #[inline]
    #[track_caller]
    fn new_owned_zeros(nrows: usize, ncols: usize) -> Self::Owned {
        Mat::zeros(nrows, ncols)
    }

    #[inline]
    fn new_owned_copied(src: &Self) -> Self::Owned {
        src.to_owned()
    }

    #[inline]
    #[track_caller]
    fn resize_owned(owned: &mut Self::Owned, nrows: usize, ncols: usize) {
        owned.resize_with(nrows, ncols, |_, _| unsafe { core::mem::zeroed() });
    }
}

impl<E: Conjugate> ColBatchMut<E> for Mat<E> {}
impl<E: Conjugate> RowBatchMut<E> for Mat<E> {}
