use super::*;
use crate::{
    col::ColRef,
    debug_assert,
    mat::{
        matalloc::{align_for, is_vectorizable, MatUnit, RawMat, RawMatUnit},
        As2D, As2DMut, Mat, MatMut, MatRef,
    },
    row::RowRef,
    utils::DivCeil,
};
use core::mem::ManuallyDrop;

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
    /// `col_capacity` columnss columns without reallocating. If `col_capacity` is `0`,
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
                ncols,
                nrows: 1,
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
    pub fn copy_from(&mut self, other: impl AsRowRef<E>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity>(this: &mut Row<E>, other: RowRef<'_, E>) {
            let mut mat = Row::<E>::new();
            mat.resize_with(
                other.nrows(),
                #[inline(always)]
                |row| unsafe { other.read_unchecked(row) },
            );
            *this = mat;
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

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(&self) -> RowRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(&self) -> ColRef<'_, E::Conj>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
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
}

impl<E: Conjugate> RowBatchMut<E> for Row<E> {}
