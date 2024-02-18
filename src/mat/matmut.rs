use super::*;
use crate::{assert, debug_assert, diag::DiagMut, linalg::zip, unzipped, zipped};

#[repr(C)]
pub struct MatMut<'a, E: Entity> {
    pub(super) inner: MatImpl<E>,
    pub(super) __marker: PhantomData<&'a E>,
}

impl<'short, E: Entity> Reborrow<'short> for MatMut<'_, E> {
    type Target = MatRef<'short, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        MatRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'short, E: Entity> ReborrowMut<'short> for MatMut<'_, E> {
    type Target = MatMut<'short, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        MatMut {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'a, E: Entity> IntoConst for MatMut<'a, E> {
    type Target = MatRef<'a, E>;

    #[inline]
    fn into_const(self) -> Self::Target {
        MatRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'a, E: Entity> MatMut<'a, E> {
    #[inline]
    pub(crate) unsafe fn __from_raw_parts(
        ptr: GroupFor<E, *mut E::Unit>,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            inner: MatImpl {
                ptr: into_copy::<E, _>(E::faer_map(
                    ptr,
                    #[inline]
                    |ptr| NonNull::new_unchecked(ptr),
                )),
                nrows,
                ncols,
                row_stride,
                col_stride,
            },
            __marker: PhantomData,
        }
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col_mut(self, j: usize) -> GroupFor<E, &'a mut [E::Unit]> {
        assert!(self.row_stride() == 1);
        let col = self.col_mut(j);
        if col.nrows() == 0 {
            E::faer_map(
                E::UNIT,
                #[inline(always)]
                |()| &mut [] as &mut [E::Unit],
            )
        } else {
            let m = col.nrows();
            E::faer_map(
                col.as_ptr_mut(),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
            )
        }
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

    /// Returns pointers to the matrix data.
    #[inline(always)]
    pub fn as_ptr_mut(self) -> GroupFor<E, *mut E::Unit> {
        E::faer_map(
            from_copy::<E, _>(self.inner.ptr),
            #[inline(always)]
            |ptr| ptr.as_ptr(),
        )
    }

    /// Returns the row stride of the matrix, specified in number of elements, not in bytes.
    #[inline(always)]
    pub fn row_stride(&self) -> isize {
        self.inner.row_stride
    }

    /// Returns the column stride of the matrix, specified in number of elements, not in bytes.
    #[inline(always)]
    pub fn col_stride(&self) -> isize {
        self.inner.col_stride
    }

    /// Returns raw pointers to the element at the given indices.
    #[inline(always)]
    pub fn ptr_at_mut(self, row: usize, col: usize) -> GroupFor<E, *mut E::Unit> {
        let offset = ((row as isize).wrapping_mul(self.inner.row_stride))
            .wrapping_add((col as isize).wrapping_mul(self.inner.col_stride));
        E::faer_map(
            self.as_ptr_mut(),
            #[inline(always)]
            |ptr| ptr.wrapping_offset(offset),
        )
    }

    #[inline(always)]
    unsafe fn ptr_at_mut_unchecked(self, row: usize, col: usize) -> GroupFor<E, *mut E::Unit> {
        let offset = crate::utils::unchecked_add(
            crate::utils::unchecked_mul(row, self.inner.row_stride),
            crate::utils::unchecked_mul(col, self.inner.col_stride),
        );
        E::faer_map(
            self.as_ptr_mut(),
            #[inline(always)]
            |ptr| ptr.offset(offset),
        )
    }

    /// Returns raw pointers to the element at the given indices, assuming the provided indices
    /// are within the matrix dimensions.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(self, row: usize, col: usize) -> GroupFor<E, *mut E::Unit> {
        debug_assert!(all(row < self.nrows(), col < self.ncols()));
        self.ptr_at_mut_unchecked(row, col)
    }

    /// Splits the matrix horizontally and vertically at the given indices into four corners and
    /// returns an array of each submatrix, in the following order:
    /// * top left.
    /// * top right.
    /// * bottom left.
    /// * bottom right.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row <= self.nrows()`.
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_mut_unchecked(self, row: usize, col: usize) -> (Self, Self, Self, Self) {
        let (top_left, top_right, bot_left, bot_right) =
            self.into_const().split_at_unchecked(row, col);
        (
            top_left.const_cast(),
            top_right.const_cast(),
            bot_left.const_cast(),
            bot_right.const_cast(),
        )
    }

    /// Splits the matrix horizontally and vertically at the given indices into four corners and
    /// returns an array of each submatrix, in the following order:
    /// * top left.
    /// * top right.
    /// * bottom left.
    /// * bottom right.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row <= self.nrows()`.
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_mut(self, row: usize, col: usize) -> (Self, Self, Self, Self) {
        let (top_left, top_right, bot_left, bot_right) = self.into_const().split_at(row, col);
        unsafe {
            (
                top_left.const_cast(),
                top_right.const_cast(),
                bot_left.const_cast(),
                bot_right.const_cast(),
            )
        }
    }

    /// Splits the matrix horizontally at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Safety
    /// The behavior is undefined if the following condition is violated:
    /// * `row <= self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_row_mut_unchecked(self, row: usize) -> (Self, Self) {
        let (top, bot) = self.into_const().split_at_row_unchecked(row);
        (top.const_cast(), bot.const_cast())
    }

    /// Splits the matrix horizontally at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * top.
    /// * bottom.
    ///
    /// # Panics
    /// The function panics if the following condition is violated:
    /// * `row <= self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_row_mut(self, row: usize) -> (Self, Self) {
        let (top, bot) = self.into_const().split_at_row(row);
        unsafe { (top.const_cast(), bot.const_cast()) }
    }

    /// Splits the matrix vertically at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * left.
    /// * right.
    ///
    /// # Safety
    /// The behavior is undefined if the following condition is violated:
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at_col_mut_unchecked(self, col: usize) -> (Self, Self) {
        let (left, right) = self.into_const().split_at_col_unchecked(col);
        (left.const_cast(), right.const_cast())
    }

    /// Splits the matrix vertically at the given row into two parts and returns an array of
    /// each submatrix, in the following order:
    /// * left.
    /// * right.
    ///
    /// # Panics
    /// The function panics if the following condition is violated:
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_at_col_mut(self, col: usize) -> (Self, Self) {
        let (left, right) = self.into_const().split_at_col(col);
        unsafe { (left.const_cast(), right.const_cast()) }
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
    #[inline(always)]
    #[track_caller]
    pub unsafe fn get_mut_unchecked<RowRange, ColRange>(
        self,
        row: RowRange,
        col: ColRange,
    ) -> <Self as MatIndex<RowRange, ColRange>>::Target
    where
        Self: MatIndex<RowRange, ColRange>,
    {
        <Self as MatIndex<RowRange, ColRange>>::get_unchecked(self, row, col)
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
    #[inline(always)]
    #[track_caller]
    pub fn get_mut<RowRange, ColRange>(
        self,
        row: RowRange,
        col: ColRange,
    ) -> <Self as MatIndex<RowRange, ColRange>>::Target
    where
        Self: MatIndex<RowRange, ColRange>,
    {
        <Self as MatIndex<RowRange, ColRange>>::get(self, row, col)
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
        self.rb().read_unchecked(row, col)
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
        self.rb().read(row, col)
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
        let units = value.faer_into_units();
        let zipped = E::faer_zip(units, (*self).rb_mut().ptr_inbounds_at_mut(row, col));
        E::faer_map(
            zipped,
            #[inline(always)]
            |(unit, ptr)| *ptr = unit,
        );
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
        assert!(all(row < self.nrows(), col < self.ncols()));
        unsafe { self.write_unchecked(row, col, value) };
    }

    /// Copies the values from the lower triangular part of `other` into the lower triangular
    /// part of `self`. The diagonal part is included.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.nrows() == other.nrows()`.
    /// * `self.ncols() == other.ncols()`.
    /// * `self.nrows() == self.ncols()`.
    #[track_caller]
    pub fn copy_from_triangular_lower(&mut self, other: impl AsMatRef<E>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity>(this: MatMut<'_, E>, other: MatRef<'_, E>) {
            zipped!(this, other).for_each_triangular_lower(
                zip::Diag::Include,
                #[inline(always)]
                |unzipped!(mut dst, src)| dst.write(src.read()),
            );
        }
        implementation(self.rb_mut(), other.as_mat_ref())
    }

    /// Copies the values from the lower triangular part of `other` into the lower triangular
    /// part of `self`. The diagonal part is excluded.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.nrows() == other.nrows()`.
    /// * `self.ncols() == other.ncols()`.
    /// * `self.nrows() == self.ncols()`.
    #[track_caller]
    pub fn copy_from_strict_triangular_lower(&mut self, other: impl AsMatRef<E>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity>(this: MatMut<'_, E>, other: MatRef<'_, E>) {
            zipped!(this, other).for_each_triangular_lower(
                zip::Diag::Skip,
                #[inline(always)]
                |unzipped!(mut dst, src)| dst.write(src.read()),
            );
        }
        implementation(self.rb_mut(), other.as_mat_ref())
    }

    /// Copies the values from the upper triangular part of `other` into the upper triangular
    /// part of `self`. The diagonal part is included.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.nrows() == other.nrows()`.
    /// * `self.ncols() == other.ncols()`.
    /// * `self.nrows() == self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub fn copy_from_triangular_upper(&mut self, other: impl AsMatRef<E>) {
        (*self)
            .rb_mut()
            .transpose_mut()
            .copy_from_triangular_lower(other.as_mat_ref().transpose())
    }

    /// Copies the values from the upper triangular part of `other` into the upper triangular
    /// part of `self`. The diagonal part is excluded.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.nrows() == other.nrows()`.
    /// * `self.ncols() == other.ncols()`.
    /// * `self.nrows() == self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub fn copy_from_strict_triangular_upper(&mut self, other: impl AsMatRef<E>) {
        (*self)
            .rb_mut()
            .transpose_mut()
            .copy_from_strict_triangular_lower(other.as_mat_ref().transpose())
    }

    /// Copies the values from `other` into `self`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.nrows() == other.nrows()`.
    /// * `self.ncols() == other.ncols()`.
    #[track_caller]
    pub fn copy_from(&mut self, other: impl AsMatRef<E>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity>(this: MatMut<'_, E>, other: MatRef<'_, E>) {
            zipped!(this, other).for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
        }
        implementation(self.rb_mut(), other.as_mat_ref())
    }

    /// Fills the elements of `self` with zeros.
    #[track_caller]
    pub fn fill_zero(&mut self)
    where
        E: ComplexField,
    {
        zipped!(self.rb_mut()).for_each(
            #[inline(always)]
            |unzipped!(mut x)| x.write(E::faer_zero()),
        );
    }

    /// Fills the elements of `self` with copies of `constant`.
    #[track_caller]
    pub fn fill(&mut self, constant: E) {
        zipped!((*self).rb_mut()).for_each(
            #[inline(always)]
            |unzipped!(mut x)| x.write(constant),
        );
    }

    /// Returns a view over the transpose of `self`.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_mut();
    /// let transpose = view.transpose_mut();
    ///
    /// let mut expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    /// assert_eq!(expected.as_mut(), transpose);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn transpose_mut(self) -> Self {
        unsafe {
            super::from_raw_parts_mut(
                E::faer_map(
                    from_copy::<E, _>(self.inner.ptr),
                    #[inline(always)]
                    |ptr| ptr.as_ptr(),
                ),
                self.ncols(),
                self.nrows(),
                self.col_stride(),
                self.row_stride(),
            )
        }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline(always)]
    #[must_use]
    pub fn conjugate_mut(self) -> MatMut<'a, E::Conj>
    where
        E: Conjugate,
    {
        unsafe { self.into_const().conjugate().const_cast() }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn adjoint_mut(self) -> MatMut<'a, E::Conj>
    where
        E: Conjugate,
    {
        self.transpose_mut().conjugate_mut()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    #[must_use]
    pub fn canonicalize_mut(self) -> (MatMut<'a, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        let (canonical, conj) = self.into_const().canonicalize();
        unsafe { (canonical.const_cast(), conj) }
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_mut();
    /// let reversed_rows = view.reverse_rows_mut();
    ///
    /// let mut expected = mat![[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]];
    /// assert_eq!(expected.as_mut(), reversed_rows);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows_mut(self) -> Self {
        unsafe { self.into_const().reverse_rows().const_cast() }
    }

    /// Returns a view over the `self`, with the columns in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_mut();
    /// let reversed_cols = view.reverse_cols_mut();
    ///
    /// let mut expected = mat![[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]];
    /// assert_eq!(expected.as_mut(), reversed_cols);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols_mut(self) -> Self {
        unsafe { self.into_const().reverse_cols().const_cast() }
    }

    /// Returns a view over the `self`, with the rows and the columns in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_mut();
    /// let reversed = view.reverse_rows_and_cols_mut();
    ///
    /// let mut expected = mat![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
    /// assert_eq!(expected.as_mut(), reversed);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows_and_cols_mut(self) -> Self {
        unsafe { self.into_const().reverse_rows_and_cols().const_cast() }
    }

    /// Returns a view over the submatrix starting at indices `(row_start, col_start)`, and with
    /// dimensions `(nrows, ncols)`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `col_start <= self.ncols()`.
    /// * `nrows <= self.nrows() - row_start`.
    /// * `ncols <= self.ncols() - col_start`.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_mut();
    /// let submatrix = view.submatrix_mut(2, 1, 2, 2);
    ///
    /// let mut expected = mat![[7.0, 11.0], [8.0, 12.0f64]];
    /// assert_eq!(expected.as_mut(), submatrix);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn submatrix_mut(
        self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        unsafe {
            self.into_const()
                .submatrix(row_start, col_start, nrows, ncols)
                .const_cast()
        }
    }

    /// Returns a view over the submatrix starting at row `row_start`, and with number of rows
    /// `nrows`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `nrows <= self.nrows() - row_start`.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_mut();
    /// let subrows = view.subrows_mut(1, 2);
    ///
    /// let mut expected = mat![[2.0, 6.0, 10.0], [3.0, 7.0, 11.0],];
    /// assert_eq!(expected.as_mut(), subrows);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn subrows_mut(self, row_start: usize, nrows: usize) -> Self {
        unsafe { self.into_const().subrows(row_start, nrows).const_cast() }
    }

    /// Returns a view over the submatrix starting at column `col_start`, and with number of
    /// columns `ncols`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_start <= self.ncols()`.
    /// * `ncols <= self.ncols() - col_start`.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let mut matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_mut();
    /// let subcols = view.subcols_mut(2, 1);
    ///
    /// let mut expected = mat![[9.0], [10.0], [11.0], [12.0f64]];
    /// assert_eq!(expected.as_mut(), subcols);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn subcols_mut(self, col_start: usize, ncols: usize) -> Self {
        unsafe { self.into_const().subcols(col_start, ncols).const_cast() }
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub fn row_mut(self, row_idx: usize) -> RowMut<'a, E> {
        unsafe { self.into_const().row(row_idx).const_cast() }
    }

    /// Returns views over the rows at the given indices.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx0 < self.nrows()`.
    /// * `row_idx1 < self.nrows()`.
    /// * `row_idx0 == row_idx1`.
    #[track_caller]
    #[inline(always)]
    pub fn two_rows_mut(self, row_idx0: usize, row_idx1: usize) -> (RowMut<'a, E>, RowMut<'a, E>) {
        assert!(row_idx0 != row_idx1);
        let this = self.into_const();
        unsafe {
            (
                this.row(row_idx0).const_cast(),
                this.row(row_idx1).const_cast(),
            )
        }
    }

    /// Returns a view over the column at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub fn col_mut(self, col_idx: usize) -> ColMut<'a, E> {
        unsafe { self.into_const().col(col_idx).const_cast() }
    }

    /// Returns views over the columns at the given indices.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_idx0 < self.ncols()`.
    /// * `col_idx1 < self.ncols()`.
    /// * `col_idx0 == col_idx1`.
    #[track_caller]
    #[inline(always)]
    pub fn two_cols_mut(self, col_idx0: usize, col_idx1: usize) -> (ColMut<'a, E>, ColMut<'a, E>) {
        assert!(col_idx0 != col_idx1);
        let this = self.into_const();
        unsafe {
            (
                this.col(col_idx0).const_cast(),
                this.col(col_idx1).const_cast(),
            )
        }
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal_mut(self) -> DiagMut<'a, E> {
        assert!(self.ncols() == 1);
        DiagMut {
            inner: self.col_mut(0),
        }
    }

    /// Returns the diagonal of the matrix.
    #[inline(always)]
    pub fn diagonal_mut(self) -> DiagMut<'a, E> {
        let size = self.nrows().min(self.ncols());
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        unsafe {
            DiagMut {
                inner: crate::col::from_raw_parts_mut(
                    self.as_ptr_mut(),
                    size,
                    row_stride + col_stride,
                ),
            }
        }
    }

    /// Returns an owning [`Mat`] of the data
    #[inline]
    pub fn to_owned(&self) -> Mat<E::Canonical>
    where
        E: Conjugate,
    {
        self.rb().to_owned()
    }

    /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
    #[inline]
    pub fn has_nan(&self) -> bool
    where
        E: ComplexField,
    {
        self.rb().has_nan()
    }

    /// Returns `true` if all of the elements are finite, otherwise returns `false`.
    #[inline]
    pub fn is_all_finite(&self) -> bool
    where
        E: ComplexField,
    {
        self.rb().is_all_finite()
    }

    /// Returns the maximum norm of `self`.
    #[inline]
    pub fn norm_max(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.rb().norm_max()
    }
    /// Returns the L2 norm of `self`.
    #[inline]
    pub fn norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.rb().norm_l2()
    }

    /// Returns the sum of `self`.
    #[inline]
    pub fn sum(&self) -> E
    where
        E: ComplexField,
    {
        self.rb().sum()
    }

    /// Kroneckor product of `self` and `rhs`.
    ///
    /// This is an allocating operation; see [`kron`] for the
    /// allocation-free version or more info in general.
    #[inline]
    #[track_caller]
    pub fn kron(&self, rhs: impl As2D<E>) -> Mat<E>
    where
        E: ComplexField,
    {
        self.as_2d_ref().kron(rhs)
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, E> {
        self.rb()
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> MatMut<'_, E> {
        self.rb_mut()
    }

    /// Returns an iterator that provides successive chunks of the columns of this matrix, with
    /// each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have
    /// `chunk_size` columns.
    #[inline]
    #[track_caller]
    pub fn col_chunks_mut(
        self,
        chunk_size: usize,
    ) -> impl 'a + DoubleEndedIterator<Item = MatMut<'a, E>> {
        self.into_const()
            .col_chunks(chunk_size)
            .map(|chunk| unsafe { chunk.const_cast() })
    }

    /// Returns an iterator that provides successive chunks of the rows of this matrix,
    /// with each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn row_chunks_mut(
        self,
        chunk_size: usize,
    ) -> impl 'a + DoubleEndedIterator<Item = MatMut<'a, E>> {
        self.into_const()
            .row_chunks(chunk_size)
            .map(|chunk| unsafe { chunk.const_cast() })
    }

    /// Returns a parallel iterator that provides successive chunks of the columns of this
    /// matrix, with each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have
    /// `chunk_size` columns.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_col_chunks_mut(
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E>> {
        use rayon::prelude::*;
        self.into_const()
            .par_col_chunks(chunk_size)
            .map(|chunk| unsafe { chunk.const_cast() })
    }

    /// Returns a parallel iterator that provides successive chunks of the rows of this matrix,
    /// with each having at most `chunk_size` rows.
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
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E>> {
        use rayon::prelude::*;
        self.into_const()
            .par_row_chunks(chunk_size)
            .map(|chunk| unsafe { chunk.const_cast() })
    }
}

impl<'a, E: RealField> MatMut<'a, num_complex::Complex<E>> {
    /// Returns the real and imaginary components of `self`.
    #[inline(always)]
    pub fn real_imag_mut(self) -> num_complex::Complex<MatMut<'a, E>> {
        let num_complex::Complex { re, im } = self.into_const().real_imag();
        unsafe {
            num_complex::Complex {
                re: re.const_cast(),
                im: im.const_cast(),
            }
        }
    }
}
