use super::*;
use crate::{
    col::{Col, ColMut, ColRef},
    iter, Idx, IdxInc,
};

/// Heap allocated resizable row vector.
///
/// # Note
///
/// The memory layout of `Col` is guaranteed to be row-major, meaning that it has a column stride
/// of `1`.
#[repr(C)]
pub struct Row<E: Entity, C: Shape = usize> {
    inner: Col<E, C>,
}

impl<E: Entity> Row<E> {
    /// Returns an empty row of dimension `0`.
    #[inline]
    pub fn new() -> Self {
        Self { inner: Col::new() }
    }

    /// Returns a new column vector with 0 columns, with enough capacity to hold a maximum of
    /// `col_capacity` columns without reallocating. If `col_capacity` is `0`,
    /// the function will not allocate.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn with_capacity(col_capacity: usize) -> Self {
        Self {
            inner: Col::with_capacity(col_capacity),
        }
    }
}
impl<E: Entity, C: Shape> Row<E, C> {
    /// Returns the input matrix with the given shape after checking that it matches the
    /// current shape.
    pub fn into_shape<H: Shape>(self, ncols: H) -> Row<E, H> {
        crate::assert!(ncols.unbound() == self.ncols().unbound());

        Row {
            inner: self.inner.into_shape(ncols),
        }
    }

    /// Returns a new matrix with number of columns `ncols`, filled with the provided function.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn from_fn(ncols: C, f: impl FnMut(Idx<C>) -> E) -> Self {
        Self {
            inner: Col::from_fn(ncols, f),
        }
    }

    /// Returns a new matrix with number of columns `ncols`, filled with zeros.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn zeros(ncols: C) -> Self {
        Self {
            inner: Col::zeros(ncols),
        }
    }

    /// Returns a new matrix with number of columns `ncols`, filled with ones.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn ones(ncols: C) -> Self
    where
        E: ComplexField,
    {
        Self {
            inner: Col::ones(ncols),
        }
    }

    /// Returns a new matrix with number of columns `ncols`, filled with a constant value.
    ///
    /// # Panics
    /// The function panics if the total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn full(ncols: C, constant: E) -> Self
    where
        E: ComplexField,
    {
        Self {
            inner: Col::full(ncols, constant),
        }
    }

    /// Returns the number of rows of the row. This is always equal to `1`.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        1
    }
    /// Returns the number of columns of the row.
    #[inline(always)]
    pub fn ncols(&self) -> C {
        self.inner.nrows()
    }

    /// Set the dimensions of the matrix.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `ncols < self.col_capacity()`.
    /// * The elements that were previously out of bounds but are now in bounds must be
    /// initialized.
    #[inline]
    pub unsafe fn set_ncols(&mut self, ncols: C) {
        self.inner.set_nrows(ncols);
    }

    /// Returns a pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr(&self) -> PtrConst<E> {
        self.inner.as_ptr()
    }

    /// Returns a mutable pointer to the data of the matrix.
    #[inline]
    pub fn as_ptr_mut(&mut self) -> PtrMut<E> {
        self.inner.as_ptr_mut()
    }

    /// Returns the col capacity, that is, the number of cols that the matrix is able to hold
    /// without needing to reallocate, excluding column insertions.
    #[inline]
    pub fn col_capacity(&self) -> usize {
        self.inner.row_capacity()
    }

    /// Returns the offset between the first elements of two successive columns in the matrix.
    /// Always returns `1` since the matrix is column major.
    #[inline]
    pub fn col_stride(&self) -> isize {
        1
    }

    /// Returns the input row with dynamic shape.
    #[inline]
    pub fn as_dyn(&self) -> RowRef<'_, E> {
        self.as_ref().as_dyn()
    }

    /// Returns the input row with dynamic shape.
    #[inline]
    pub fn as_dyn_mut(&mut self) -> RowMut<'_, E> {
        self.as_mut().as_dyn_mut()
    }

    /// Returns the input row with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape<H: Shape>(&self, ncols: H) -> RowRef<'_, E, H> {
        self.as_ref().as_shape(ncols)
    }

    /// Returns the input row with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape_mut<H: Shape>(&mut self, ncols: H) -> RowMut<'_, E, H> {
        self.as_mut().as_shape_mut(ncols)
    }

    /// Returns `self` as a matrix view.
    #[inline(always)]
    pub fn as_2d(&self) -> MatRef<'_, E, usize, C> {
        self.as_ref().as_2d()
    }

    /// Returns `self` as a mutable matrix view.
    #[inline(always)]
    pub fn as_2d_mut(&mut self) -> MatMut<'_, E, usize, C> {
        self.as_mut().as_2d_mut()
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at(&self, col: usize) -> PtrConst<E> {
        self.as_ref().ptr_at(col)
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at_mut(&mut self, col: usize) -> PtrMut<E> {
        self.as_mut().ptr_at_mut(col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(&self, col: usize) -> PtrConst<E> {
        self.as_ref().ptr_at_unchecked(col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_mut_unchecked(&mut self, col: usize) -> PtrMut<E> {
        self.as_mut().ptr_at_mut_unchecked(col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(&self, col: IdxInc<C>) -> PtrConst<E> {
        self.as_ref().overflowing_ptr_at(col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at_mut(&mut self, col: IdxInc<C>) -> PtrMut<E> {
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
    pub unsafe fn ptr_inbounds_at(&self, col: Idx<C>) -> PtrConst<E> {
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
    pub unsafe fn ptr_inbounds_at_mut(&mut self, col: Idx<C>) -> PtrMut<E> {
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
    pub unsafe fn split_at_unchecked(&self, col: IdxInc<C>) -> (RowRef<'_, E>, RowRef<'_, E>) {
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
    pub unsafe fn split_at_mut_unchecked(
        &mut self,
        col: IdxInc<C>,
    ) -> (RowMut<'_, E>, RowMut<'_, E>) {
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
    pub unsafe fn split_at(&self, col: IdxInc<C>) -> (RowRef<'_, E>, RowRef<'_, E>) {
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
    pub fn split_at_mut(&mut self, col: IdxInc<C>) -> (RowMut<'_, E>, RowMut<'_, E>) {
        self.as_mut().split_at_mut(col)
    }

    /// Reserves the minimum capacity for `col_capacity` columns without reallocating. Does nothing
    /// if the capacity is already sufficient.
    ///
    /// # Panics
    /// The function panics if the new total capacity in bytes exceeds `isize::MAX`.
    #[inline]
    pub fn reserve_exact(&mut self, col_capacity: usize) {
        self.inner.reserve_exact(col_capacity);
    }

    /// Resizes the vector in-place so that the new number of columns is `new_ncols`.
    /// New elements are created with the given function `f`, so that elements at index `i`
    /// are created by calling `f(i)`.
    pub fn resize_with(&mut self, new_ncols: C, f: impl FnMut(Idx<C>) -> E) {
        self.inner.resize_with(new_ncols, f)
    }

    /// Truncates the matrix so that its new number of columns is `new_ncols`.  
    /// The new dimension must be smaller than the current dimension of the vector.
    ///
    /// # Panics
    /// - Panics if `new_ncols > self.ncols()`.
    #[inline]
    pub fn truncate(&mut self, new_ncols: C) {
        self.inner.truncate(new_ncols)
    }

    /// Returns a reference to a slice over the row.
    #[inline]
    #[track_caller]
    pub fn as_slice(&self) -> Slice<'_, E> {
        self.inner.as_slice()
    }

    /// Returns a mutable reference to a slice over the row.
    #[inline]
    #[track_caller]
    pub fn as_slice_mut(&mut self) -> SliceMut<'_, E> {
        self.inner.as_slice_mut()
    }

    /// Returns the row as a contiguous slice if its column stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice(&self) -> Option<Slice<'_, E>> {
        Some(self.as_slice())
    }

    /// Returns the row as a contiguous slice if its column stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice_mut(&mut self) -> Option<SliceMut<'_, E>> {
        Some(self.as_slice_mut())
    }

    /// Returns the row as a contiguous potentially uninitialized slice if its column stride is
    /// equal to `1`.
    ///
    /// # Safety
    /// If uninit data is written to the slice, it must not be read at some later point.
    #[inline]
    pub unsafe fn try_as_uninit_slice_mut(&mut self) -> Option<UninitSliceMut<'_, E>> {
        Some(self.as_uninit_slice_mut())
    }

    /// Returns a mutable reference to a potentially uninitialized slice over the column.
    ///
    /// # Safety
    /// If uninit data is written to the slice, it must not be later read.
    #[inline]
    pub unsafe fn as_uninit_slice_mut(&mut self) -> UninitSliceMut<'_, E> {
        self.inner.as_uninit_slice_mut()
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
    pub unsafe fn subcols_unchecked<H: Shape>(
        &self,
        col_start: IdxInc<C>,
        ncols: H,
    ) -> RowRef<'_, E, H> {
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
    pub fn subcols<H: Shape>(&self, col_start: IdxInc<C>, ncols: H) -> RowRef<'_, E, H> {
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
    pub unsafe fn subcols_mut_unchecked<H: Shape>(
        &mut self,
        col_start: IdxInc<C>,
        ncols: H,
    ) -> RowMut<'_, E, H> {
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
    pub fn subcols_mut<H: Shape>(&mut self, col_start: IdxInc<C>, ncols: H) -> RowMut<'_, E, H> {
        self.as_mut().subcols_mut(col_start, ncols)
    }

    /// Returns a view over the vector.
    #[inline]
    pub fn as_ref(&self) -> RowRef<'_, E, C> {
        unsafe { super::from_raw_parts(self.as_ptr(), self.ncols(), 1) }
    }

    /// Returns a mutable view over the vector.
    #[inline]
    pub fn as_mut(&mut self) -> RowMut<'_, E, C> {
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
    ) -> <RowRef<'_, E, C> as RowIndex<ColRange>>::Target
    where
        for<'a> RowRef<'a, E, C>: RowIndex<ColRange>,
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
    pub fn get<ColRange>(&self, col: ColRange) -> <RowRef<'_, E, C> as RowIndex<ColRange>>::Target
    where
        for<'a> RowRef<'a, E, C>: RowIndex<ColRange>,
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
    ) -> <RowMut<'_, E, C> as RowIndex<ColRange>>::Target
    where
        for<'a> RowMut<'a, E, C>: RowIndex<ColRange>,
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
    ) -> <RowMut<'_, E, C> as RowIndex<ColRange>>::Target
    where
        for<'a> RowMut<'a, E, C>: RowIndex<ColRange>,
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
    pub unsafe fn read_unchecked(&self, col: Idx<C>) -> E {
        self.as_ref().read_unchecked(col)
    }

    /// Reads the value of the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, col: Idx<C>) -> E {
        self.as_ref().read(col)
    }

    /// Writes the value to the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, col: Idx<C>, value: E) {
        self.as_mut().write_unchecked(col, value);
    }

    /// Writes the value to the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, col: Idx<C>, value: E) {
        self.as_mut().write(col, value);
    }

    /// Copies the values from `other` into `self`.
    #[inline(always)]
    #[track_caller]
    pub fn copy_from<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsRowRef<ViewE, C = C>,
    ) {
        self.inner.copy_from(other.as_row_ref().transpose())
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
    pub fn transpose(&self) -> ColRef<'_, E, C> {
        self.as_ref().transpose()
    }

    /// Returns a view over the transpose of `self`.
    #[inline]
    pub fn transpose_mut(&mut self) -> ColMut<'_, E, C> {
        self.as_mut().transpose_mut()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate(&self) -> RowRef<'_, E::Conj, C>
    where
        E: Conjugate,
    {
        self.as_ref().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline]
    pub fn conjugate_mut(&mut self) -> RowMut<'_, E::Conj, C>
    where
        E: Conjugate,
    {
        self.as_mut().conjugate_mut()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint(&self) -> ColRef<'_, E::Conj, C>
    where
        E: Conjugate,
    {
        self.as_ref().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline]
    pub fn adjoint_mut(&mut self) -> ColMut<'_, E::Conj, C>
    where
        E: Conjugate,
    {
        self.as_mut().adjoint_mut()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize(&self) -> (RowRef<'_, E::Canonical, C>, Conj)
    where
        E: Conjugate,
    {
        self.as_ref().canonicalize()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize_mut(&mut self) -> (RowMut<'_, E::Canonical, C>, Conj)
    where
        E: Conjugate,
    {
        self.as_mut().canonicalize_mut()
    }

    /// Returns a view over the `self`, with the columns in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols(&self) -> RowRef<'_, E, C> {
        self.as_ref().reverse_cols()
    }

    /// Returns a view over the `self`, with the columns in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols_mut(&mut self) -> RowMut<'_, E, C> {
        self.as_mut().reverse_cols_mut()
    }

    /// Returns an owning [`Row`] of the data
    #[inline]
    pub fn to_owned(&self) -> Row<E::Canonical, C>
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
        self.as_ref().kron(rhs)
    }

    /// Returns a reference to the first element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_first(&self) -> Option<(Ref<'_, E>, RowRef<'_, E>)> {
        self.as_ref().split_first()
    }

    /// Returns a reference to the last element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_last(&self) -> Option<(Ref<'_, E>, RowRef<'_, E>)> {
        self.as_ref().split_last()
    }

    /// Returns a reference to the first element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_first_mut(&mut self) -> Option<(Mut<'_, E>, RowMut<'_, E>)> {
        self.as_mut().split_first_mut()
    }

    /// Returns a reference to the last element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_last_mut(&mut self) -> Option<(Mut<'_, E>, RowMut<'_, E>)> {
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
    pub unsafe fn const_cast(&self) -> RowMut<'_, E, C> {
        self.as_ref().const_cast()
    }
}

impl<E: Entity> Default for Row<E> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Entity, C: Shape> Clone for Row<E, C> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.inner.clone_from(&other.inner)
    }
}

impl<E: Entity, C: Shape> As2D<E> for Row<E, C> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (*self).as_ref().as_2d().as_dyn()
    }
}

impl<E: Entity, C: Shape> As2DMut<E> for Row<E, C> {
    #[inline]
    fn as_2d_mut(&mut self) -> MatMut<'_, E> {
        (*self).as_mut().as_2d_mut().as_dyn_mut()
    }
}

impl<E: Entity, C: Shape> AsRowRef<E> for Row<E, C> {
    type C = C;

    #[inline]
    fn as_row_ref(&self) -> RowRef<'_, E, C> {
        (*self).as_ref()
    }
}

impl<E: Entity, C: Shape> AsRowMut<E> for Row<E, C> {
    #[inline]
    fn as_row_mut(&mut self) -> RowMut<'_, E, C> {
        (*self).as_mut()
    }
}

impl<E: Entity, C: Shape> core::fmt::Debug for Row<E, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<E: SimpleEntity, C: Shape> core::ops::Index<Idx<C>> for Row<E, C> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, col: Idx<C>) -> &E {
        self.as_ref().at(col)
    }
}

impl<E: SimpleEntity, C: Shape> core::ops::IndexMut<Idx<C>> for Row<E, C> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, col: Idx<C>) -> &mut E {
        self.as_mut().at_mut(col)
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
