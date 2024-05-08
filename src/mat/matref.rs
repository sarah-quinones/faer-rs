use super::*;
use crate::{assert, debug_assert, diag::DiagRef, unzipped, utils::DivCeil, zipped};

/// Immutable view over a matrix, similar to an immutable reference to a 2D strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `MatRef<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`MatRef::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
#[repr(C)]
pub struct MatRef<'a, E: Entity> {
    pub(super) inner: MatImpl<E>,
    pub(super) __marker: PhantomData<&'a E>,
}

impl<E: Entity> Clone for MatRef<'_, E> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Entity> Copy for MatRef<'_, E> {}

impl<'short, E: Entity> Reborrow<'short> for MatRef<'_, E> {
    type Target = MatRef<'short, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, E: Entity> ReborrowMut<'short> for MatRef<'_, E> {
    type Target = MatRef<'short, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<E: Entity> IntoConst for MatRef<'_, E> {
    type Target = Self;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'a, E: Entity> MatRef<'a, E> {
    #[inline]
    pub(crate) unsafe fn __from_raw_parts(
        ptr: GroupFor<E, *const E::Unit>,
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
                    |ptr| NonNull::new_unchecked(ptr as *mut E::Unit),
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
    pub fn try_get_contiguous_col(self, j: usize) -> GroupFor<E, &'a [E::Unit]> {
        assert!(self.row_stride() == 1);
        let col = self.col(j);
        if col.nrows() == 0 {
            E::faer_map(
                E::UNIT,
                #[inline(always)]
                |()| &[] as &[E::Unit],
            )
        } else {
            let m = col.nrows();
            E::faer_map(
                col.as_ptr(),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
            )
        }
    }

    /// Returns pointers to the matrix data.
    #[inline(always)]
    pub fn as_ptr(self) -> GroupFor<E, *const E::Unit> {
        E::faer_map(
            from_copy::<E, _>(self.inner.ptr),
            #[inline]
            |ptr| ptr.as_ptr() as *const E::Unit,
        )
    }

    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.inner.nrows
    }

    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.inner.ncols
    }

    /// Returns the number of rows and columns of the matrix.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Returns the row stride of the matrix, specified in number of elements, not in bytes.
    #[inline]
    pub fn row_stride(&self) -> isize {
        self.inner.row_stride
    }

    /// Returns the column stride of the matrix, specified in number of elements, not in bytes.
    #[inline]
    pub fn col_stride(&self) -> isize {
        self.inner.col_stride
    }

    /// Returns raw pointers to the element at the given indices.
    #[inline(always)]
    pub fn ptr_at(self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
        let offset = ((row as isize).wrapping_mul(self.inner.row_stride))
            .wrapping_add((col as isize).wrapping_mul(self.inner.col_stride));

        E::faer_map(
            self.as_ptr(),
            #[inline(always)]
            |ptr| ptr.wrapping_offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
        let offset = crate::utils::unchecked_add(
            crate::utils::unchecked_mul(row, self.inner.row_stride),
            crate::utils::unchecked_mul(col, self.inner.col_stride),
        );
        E::faer_map(
            self.as_ptr(),
            #[inline(always)]
            |ptr| ptr.offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
        unsafe {
            let cond = (row != self.nrows()) & (col != self.ncols());
            let offset = (cond as usize).wrapping_neg() as isize
                & (isize::wrapping_add(
                    (row as isize).wrapping_mul(self.inner.row_stride),
                    (col as isize).wrapping_mul(self.inner.col_stride),
                ));
            E::faer_map(
                self.as_ptr(),
                #[inline(always)]
                |ptr| ptr.offset(offset),
            )
        }
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
    pub unsafe fn ptr_inbounds_at(self, row: usize, col: usize) -> GroupFor<E, *const E::Unit> {
        debug_assert!(all(row < self.nrows(), col < self.ncols()));
        self.ptr_at_unchecked(row, col)
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
    pub unsafe fn split_at_unchecked(self, row: usize, col: usize) -> (Self, Self, Self, Self) {
        debug_assert!(all(row <= self.nrows(), col <= self.ncols()));

        let row_stride = self.row_stride();
        let col_stride = self.col_stride();

        let nrows = self.nrows();
        let ncols = self.ncols();

        unsafe {
            let top_left = self.overflowing_ptr_at(0, 0);
            let top_right = self.overflowing_ptr_at(0, col);
            let bot_left = self.overflowing_ptr_at(row, 0);
            let bot_right = self.overflowing_ptr_at(row, col);

            (
                Self::__from_raw_parts(top_left, row, col, row_stride, col_stride),
                Self::__from_raw_parts(top_right, row, ncols - col, row_stride, col_stride),
                Self::__from_raw_parts(bot_left, nrows - row, col, row_stride, col_stride),
                Self::__from_raw_parts(bot_right, nrows - row, ncols - col, row_stride, col_stride),
            )
        }
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
    pub fn split_at(self, row: usize, col: usize) -> (Self, Self, Self, Self) {
        assert!(all(row <= self.nrows(), col <= self.ncols()));
        unsafe { self.split_at_unchecked(row, col) }
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
    pub unsafe fn split_at_row_unchecked(self, row: usize) -> (Self, Self) {
        debug_assert!(row <= self.nrows());

        let row_stride = self.row_stride();
        let col_stride = self.col_stride();

        let nrows = self.nrows();
        let ncols = self.ncols();

        unsafe {
            let top_right = self.overflowing_ptr_at(0, 0);
            let bot_right = self.overflowing_ptr_at(row, 0);

            (
                Self::__from_raw_parts(top_right, row, ncols, row_stride, col_stride),
                Self::__from_raw_parts(bot_right, nrows - row, ncols, row_stride, col_stride),
            )
        }
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
    pub fn split_at_row(self, row: usize) -> (Self, Self) {
        assert!(row <= self.nrows());
        unsafe { self.split_at_row_unchecked(row) }
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
    pub unsafe fn split_at_col_unchecked(self, col: usize) -> (Self, Self) {
        debug_assert!(col <= self.ncols());

        let row_stride = self.row_stride();
        let col_stride = self.col_stride();

        let nrows = self.nrows();
        let ncols = self.ncols();

        unsafe {
            let bot_left = self.overflowing_ptr_at(0, 0);
            let bot_right = self.overflowing_ptr_at(0, col);

            (
                Self::__from_raw_parts(bot_left, nrows, col, row_stride, col_stride),
                Self::__from_raw_parts(bot_right, nrows, ncols - col, row_stride, col_stride),
            )
        }
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
    pub fn split_at_col(self, col: usize) -> (Self, Self) {
        assert!(col <= self.ncols());
        unsafe { self.split_at_col_unchecked(col) }
    }

    /// Returns references to the element at the given indices, or submatrices if either `row`
    /// or `col` is a range.
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
    pub unsafe fn get_unchecked<RowRange, ColRange>(
        self,
        row: RowRange,
        col: ColRange,
    ) -> <Self as MatIndex<RowRange, ColRange>>::Target
    where
        Self: MatIndex<RowRange, ColRange>,
    {
        <Self as MatIndex<RowRange, ColRange>>::get_unchecked(self, row, col)
    }

    /// Returns references to the element at the given indices, or submatrices if either `row`
    /// or `col` is a range, with bound checks.
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
    pub fn get<RowRange, ColRange>(
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
        E::faer_from_units(E::faer_map(
            self.get_unchecked(row, col),
            #[inline(always)]
            |ptr| *ptr,
        ))
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
        E::faer_from_units(E::faer_map(
            self.get(row, col),
            #[inline(always)]
            |ptr| *ptr,
        ))
    }

    /// Returns a view over the transpose of `self`.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_ref();
    /// let transpose = view.transpose();
    ///
    /// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    /// assert_eq!(expected.as_ref(), transpose);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn transpose(self) -> Self {
        unsafe {
            Self::__from_raw_parts(
                self.as_ptr(),
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
    pub fn conjugate(self) -> MatRef<'a, E::Conj>
    where
        E: Conjugate,
    {
        unsafe {
            // SAFETY: Conjugate requires that E::Unit and E::Conj::Unit have the same layout
            // and that GroupCopyFor<E,X> == E::Conj::GroupCopy<X>
            MatRef::<'_, E::Conj>::__from_raw_parts(
                transmute_unchecked::<
                    GroupFor<E, *const UnitFor<E>>,
                    GroupFor<E::Conj, *const UnitFor<E::Conj>>,
                >(self.as_ptr()),
                self.nrows(),
                self.ncols(),
                self.row_stride(),
                self.col_stride(),
            )
        }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn adjoint(self) -> MatRef<'a, E::Conj>
    where
        E: Conjugate,
    {
        self.transpose().conjugate()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    #[must_use]
    pub fn canonicalize(self) -> (MatRef<'a, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        (
            unsafe {
                // SAFETY: see Self::conjugate
                MatRef::<'_, E::Canonical>::__from_raw_parts(
                    transmute_unchecked::<
                        GroupFor<E, *const E::Unit>,
                        GroupFor<E::Canonical, *const UnitFor<E::Canonical>>,
                    >(self.as_ptr()),
                    self.nrows(),
                    self.ncols(),
                    self.row_stride(),
                    self.col_stride(),
                )
            },
            if coe::is_same::<E, E::Canonical>() {
                Conj::No
            } else {
                Conj::Yes
            },
        )
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_ref();
    /// let reversed_rows = view.reverse_rows();
    ///
    /// let expected = mat![[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]];
    /// assert_eq!(expected.as_ref(), reversed_rows);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows(self) -> Self {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride().wrapping_neg();
        let col_stride = self.col_stride();

        let ptr = unsafe { self.ptr_at_unchecked(nrows.saturating_sub(1), 0) };
        unsafe { Self::__from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }

    /// Returns a view over the `self`, with the columns in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_ref();
    /// let reversed_cols = view.reverse_cols();
    ///
    /// let expected = mat![[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]];
    /// assert_eq!(expected.as_ref(), reversed_cols);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols(self) -> Self {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride();
        let col_stride = self.col_stride().wrapping_neg();
        let ptr = unsafe { self.ptr_at_unchecked(0, ncols.saturating_sub(1)) };
        unsafe { Self::__from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }

    /// Returns a view over the `self`, with the rows and the columns in reversed order.
    ///
    /// # Example
    /// ```
    /// use faer::mat;
    ///
    /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_ref();
    /// let reversed = view.reverse_rows_and_cols();
    ///
    /// let expected = mat![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
    /// assert_eq!(expected.as_ref(), reversed);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows_and_cols(self) -> Self {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = -self.row_stride();
        let col_stride = -self.col_stride();

        let ptr =
            unsafe { self.ptr_at_unchecked(nrows.saturating_sub(1), ncols.saturating_sub(1)) };
        unsafe { Self::__from_raw_parts(ptr, nrows, ncols, row_stride, col_stride) }
    }

    /// Returns a view over the submatrix starting at indices `(row_start, col_start)`, and with
    /// dimensions `(nrows, ncols)`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `col_start <= self.ncols()`.
    /// * `nrows <= self.nrows() - row_start`.
    /// * `ncols <= self.ncols() - col_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn submatrix_unchecked(
        self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        debug_assert!(all(row_start <= self.nrows(), col_start <= self.ncols()));
        debug_assert!(all(
            nrows <= self.nrows() - row_start,
            ncols <= self.ncols() - col_start,
        ));
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();

        unsafe {
            Self::__from_raw_parts(
                self.overflowing_ptr_at(row_start, col_start),
                nrows,
                ncols,
                row_stride,
                col_stride,
            )
        }
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
    /// let matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_ref();
    /// let submatrix = view.submatrix(2, 1, 2, 2);
    ///
    /// let expected = mat![[7.0, 11.0], [8.0, 12.0f64]];
    /// assert_eq!(expected.as_ref(), submatrix);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn submatrix(self, row_start: usize, col_start: usize, nrows: usize, ncols: usize) -> Self {
        assert!(all(row_start <= self.nrows(), col_start <= self.ncols()));
        assert!(all(
            nrows <= self.nrows() - row_start,
            ncols <= self.ncols() - col_start,
        ));
        unsafe { self.submatrix_unchecked(row_start, col_start, nrows, ncols) }
    }

    /// Returns a view over the submatrix starting at row `row_start`, and with number of rows
    /// `nrows`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `nrows <= self.nrows() - row_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn subrows_unchecked(self, row_start: usize, nrows: usize) -> Self {
        debug_assert!(row_start <= self.nrows());
        debug_assert!(nrows <= self.nrows() - row_start);
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        unsafe {
            Self::__from_raw_parts(
                self.overflowing_ptr_at(row_start, 0),
                nrows,
                self.ncols(),
                row_stride,
                col_stride,
            )
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
    /// let matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_ref();
    /// let subrows = view.subrows(1, 2);
    ///
    /// let expected = mat![[2.0, 6.0, 10.0], [3.0, 7.0, 11.0],];
    /// assert_eq!(expected.as_ref(), subrows);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn subrows(self, row_start: usize, nrows: usize) -> Self {
        assert!(row_start <= self.nrows());
        assert!(nrows <= self.nrows() - row_start);
        unsafe { self.subrows_unchecked(row_start, nrows) }
    }

    /// Returns a view over the submatrix starting at column `col_start`, and with number of
    /// columns `ncols`.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col_start <= self.ncols()`.
    /// * `ncols <= self.ncols() - col_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn subcols_unchecked(self, col_start: usize, ncols: usize) -> Self {
        debug_assert!(col_start <= self.ncols());
        debug_assert!(ncols <= self.ncols() - col_start);
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        unsafe {
            Self::__from_raw_parts(
                self.overflowing_ptr_at(0, col_start),
                self.nrows(),
                ncols,
                row_stride,
                col_stride,
            )
        }
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
    /// let matrix = mat![
    ///     [1.0, 5.0, 9.0],
    ///     [2.0, 6.0, 10.0],
    ///     [3.0, 7.0, 11.0],
    ///     [4.0, 8.0, 12.0f64],
    /// ];
    ///
    /// let view = matrix.as_ref();
    /// let subcols = view.subcols(2, 1);
    ///
    /// let expected = mat![[9.0], [10.0], [11.0], [12.0f64]];
    /// assert_eq!(expected.as_ref(), subcols);
    /// ```
    #[track_caller]
    #[inline(always)]
    pub fn subcols(self, col_start: usize, ncols: usize) -> Self {
        debug_assert!(col_start <= self.ncols());
        debug_assert!(ncols <= self.ncols() - col_start);
        unsafe { self.subcols_unchecked(col_start, ncols) }
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Safety
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn row_unchecked(self, row_idx: usize) -> RowRef<'a, E> {
        debug_assert!(row_idx < self.nrows());
        unsafe {
            crate::row::from_raw_parts(
                self.overflowing_ptr_at(row_idx, 0),
                self.ncols(),
                self.col_stride(),
            )
        }
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub fn row(self, row_idx: usize) -> RowRef<'a, E> {
        assert!(row_idx < self.nrows());
        unsafe { self.row_unchecked(row_idx) }
    }

    /// Returns a view over the column at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn col_unchecked(self, col_idx: usize) -> ColRef<'a, E> {
        debug_assert!(col_idx < self.ncols());
        unsafe {
            crate::col::from_raw_parts(
                self.overflowing_ptr_at(0, col_idx),
                self.nrows(),
                self.row_stride(),
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
    pub fn col(self, col_idx: usize) -> ColRef<'a, E> {
        assert!(col_idx < self.ncols());
        unsafe { self.col_unchecked(col_idx) }
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whoes diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal(self) -> DiagRef<'a, E> {
        assert!(self.ncols() == 1);
        DiagRef { inner: self.col(0) }
    }

    /// Returns the diagonal of the matrix.
    #[inline(always)]
    pub fn diagonal(self) -> DiagRef<'a, E> {
        let size = self.nrows().min(self.ncols());
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        unsafe {
            DiagRef {
                inner: crate::col::from_raw_parts(self.as_ptr(), size, row_stride + col_stride),
            }
        }
    }

    /// Returns an owning [`Mat`] of the data.
    #[inline]
    pub fn to_owned(&self) -> Mat<E::Canonical>
    where
        E: Conjugate,
    {
        let mut mat = Mat::new();
        mat.resize_with(
            self.nrows(),
            self.ncols(),
            #[inline(always)]
            |row, col| unsafe { self.read_unchecked(row, col).canonicalize() },
        );
        mat
    }

    /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
    #[inline]
    pub fn has_nan(&self) -> bool
    where
        E: ComplexField,
    {
        let mut found_nan = false;
        zipped!(*self).for_each(|unzipped!(x)| {
            found_nan |= x.read().faer_is_nan();
        });
        found_nan
    }

    /// Returns `true` if all of the elements are finite, otherwise returns `false`.
    #[inline]
    pub fn is_all_finite(&self) -> bool
    where
        E: ComplexField,
    {
        let mut all_finite = true;
        zipped!(*self).for_each(|unzipped!(x)| {
            all_finite &= x.read().faer_is_finite();
        });
        all_finite
    }

    /// Returns the maximum norm of `self`.
    #[inline]
    pub fn norm_max(&self) -> E::Real
    where
        E: ComplexField,
    {
        crate::linalg::reductions::norm_max::norm_max((*self).rb())
    }

    /// Returns the L1 norm of `self`.
    #[inline]
    pub fn norm_l1(&self) -> E::Real
    where
        E: ComplexField,
    {
        crate::linalg::reductions::norm_l1::norm_l1((*self).rb())
    }

    /// Returns the L2 norm of `self`.
    #[inline]
    pub fn norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        crate::linalg::reductions::norm_l2::norm_l2((*self).rb())
    }

    /// Returns the squared L2 norm of `self`.
    #[inline]
    pub fn squared_norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        let norm = crate::linalg::reductions::norm_l2::norm_l2((*self).rb());
        norm.faer_mul(norm)
    }

    /// Returns the sum of `self`.
    #[inline]
    pub fn sum(&self) -> E
    where
        E: ComplexField,
    {
        crate::linalg::reductions::sum::sum((*self).rb())
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
        let lhs = (*self).rb();
        let rhs = rhs.as_2d_ref();
        let mut dst = Mat::new();
        dst.resize_with(
            lhs.nrows() * rhs.nrows(),
            lhs.ncols() * rhs.ncols(),
            |_, _| E::zeroed(),
        );
        crate::linalg::kron(dst.as_mut(), lhs, rhs);
        dst
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, E> {
        *self
    }

    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn const_cast(self) -> MatMut<'a, E> {
        MatMut {
            inner: self.inner,
            __marker: PhantomData,
        }
    }

    /// Returns an iterator that provides successive chunks of the columns of this matrix, with
    /// each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have
    /// `chunk_size` columns.
    #[inline]
    #[track_caller]
    pub fn col_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a + DoubleEndedIterator<Item = MatRef<'a, E>> {
        assert!(chunk_size > 0);
        let chunk_count = self.ncols().msrv_div_ceil(chunk_size);
        (0..chunk_count).map(move |chunk_idx| {
            let pos = chunk_size * chunk_idx;
            self.subcols(pos, Ord::min(chunk_size, self.ncols() - pos))
        })
    }

    /// Returns an iterator that provides successive chunks of the rows of this matrix, with
    /// each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn row_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a + DoubleEndedIterator<Item = MatRef<'a, E>> {
        self.transpose()
            .col_chunks(chunk_size)
            .map(|chunk| chunk.transpose())
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
    pub fn par_col_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E>> {
        use rayon::prelude::*;

        assert!(chunk_size > 0);
        let chunk_count = self.ncols().msrv_div_ceil(chunk_size);
        (0..chunk_count).into_par_iter().map(move |chunk_idx| {
            let pos = chunk_size * chunk_idx;
            self.subcols(pos, Ord::min(chunk_size, self.ncols() - pos))
        })
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
    pub fn par_row_chunks(
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E>> {
        use rayon::prelude::*;

        self.transpose()
            .par_col_chunks(chunk_size)
            .map(|chunk| chunk.transpose())
    }
}

impl<'a, E: RealField> MatRef<'a, num_complex::Complex<E>> {
    /// Returns the real and imaginary components of `self`.
    #[inline(always)]
    pub fn real_imag(self) -> num_complex::Complex<MatRef<'a, E>> {
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        let nrows = self.nrows();
        let ncols = self.ncols();
        let num_complex::Complex { re, im } = self.as_ptr();
        unsafe {
            num_complex::Complex {
                re: super::from_raw_parts(re, nrows, ncols, row_stride, col_stride),
                im: super::from_raw_parts(im, nrows, ncols, row_stride, col_stride),
            }
        }
    }
}

impl<E: Entity> AsMatRef<E> for MatRef<'_, E> {
    #[inline]
    fn as_mat_ref(&self) -> MatRef<'_, E> {
        *self
    }
}

impl<E: Entity> As2D<E> for MatRef<'_, E> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        *self
    }
}

/// Creates a `MatRef` from pointers to the matrix data, dimensions, and strides.
///
/// The row (resp. column) stride is the offset from the memory address of a given matrix
/// element at indices `(row: i, col: j)`, to the memory address of the matrix element at
/// indices `(row: i + 1, col: 0)` (resp. `(row: 0, col: i + 1)`). This offset is specified in
/// number of elements, not in bytes.
///
/// # Safety
/// The behavior is undefined if any of the following conditions are violated:
/// * For each matrix unit, the entire memory region addressed by the matrix must be contained
/// within a single allocation, accessible in its entirety by the corresponding pointer in
/// `ptr`.
/// * For each matrix unit, the corresponding pointer must be properly aligned,
/// even for a zero-sized matrix.
/// * The values accessible by the matrix must be initialized at some point before they are
/// read, or references to them are formed.
/// * No mutable aliasing is allowed. In other words, none of the elements accessible by any
/// matrix unit may be accessed for writes by any other means for the duration of the lifetime
/// `'a`.
///
/// # Example
///
/// ```
/// use faer::mat;
///
/// // row major matrix with 2 rows, 3 columns, with a column at the end that we want to skip.
/// // the row stride is the pointer offset from the address of 1.0 to the address of 4.0,
/// // which is 4.
/// // the column stride is the pointer offset from the address of 1.0 to the address of 2.0,
/// // which is 1.
/// let data = [[1.0, 2.0, 3.0, f64::NAN], [4.0, 5.0, 6.0, f64::NAN]];
/// let matrix = unsafe { mat::from_raw_parts::<f64>(data.as_ptr() as *const f64, 2, 3, 4, 1) };
///
/// let expected = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// assert_eq!(expected.as_ref(), matrix);
/// ```
#[inline(always)]
pub unsafe fn from_raw_parts<'a, E: Entity>(
    ptr: GroupFor<E, *const E::Unit>,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
) -> MatRef<'a, E> {
    MatRef::__from_raw_parts(ptr, nrows, ncols, row_stride, col_stride)
}

/// Creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a column-major format, so that the first chunk of `nrows`
/// values from the slices goes in the first column of the matrix, the second chunk of `nrows`
/// values goes in the second column, and so on.
///
/// # Panics
/// The function panics if any of the following conditions are violated:
/// * `nrows * ncols == slice.len()`
///
/// # Example
/// ```
/// use faer::mat;
///
/// let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
/// let view = mat::from_column_major_slice::<f64>(&slice, 3, 2);
///
/// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
/// assert_eq!(expected, view);
/// ```
#[track_caller]
#[inline(always)]
pub fn from_column_major_slice<E: Entity>(
    slice: GroupFor<E, &[E::Unit]>,
    nrows: usize,
    ncols: usize,
) -> MatRef<'_, E> {
    from_slice_assert(
        nrows,
        ncols,
        SliceGroup::<'_, E>::new(E::faer_copy(&slice)).len(),
    );

    unsafe {
        from_raw_parts(
            E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.as_ptr(),
            ),
            nrows,
            ncols,
            1,
            nrows as isize,
        )
    }
}

/// Creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a row-major format, so that the first chunk of `ncols`
/// values from the slices goes in the first column of the matrix, the second chunk of `ncols`
/// values goes in the second column, and so on.
///
/// # Panics
/// The function panics if any of the following conditions are violated:
/// * `nrows * ncols == slice.len()`
///
/// # Example
/// ```
/// use faer::mat;
///
/// let slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
/// let view = mat::from_row_major_slice::<f64>(&slice, 3, 2);
///
/// let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// assert_eq!(expected, view);
/// ```
#[track_caller]
#[inline(always)]
pub fn from_row_major_slice<E: Entity>(
    slice: GroupFor<E, &[E::Unit]>,
    nrows: usize,
    ncols: usize,
) -> MatRef<'_, E> {
    from_column_major_slice(slice, ncols, nrows).transpose()
}

/// Creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a column-major format, where the beginnings of two consecutive
/// columns are separated by `col_stride` elements.
#[track_caller]
pub fn from_column_major_slice_with_stride<E: Entity>(
    slice: GroupFor<E, &[E::Unit]>,
    nrows: usize,
    ncols: usize,
    col_stride: usize,
) -> MatRef<'_, E> {
    from_strided_column_major_slice_assert(
        nrows,
        ncols,
        col_stride,
        SliceGroup::<'_, E>::new(E::faer_copy(&slice)).len(),
    );

    unsafe {
        from_raw_parts(
            E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.as_ptr(),
            ),
            nrows,
            ncols,
            1,
            col_stride as isize,
        )
    }
}

/// Creates a `MatRef` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a row-major format, where the beginnings of two consecutive
/// rows are separated by `row_stride` elements.
#[track_caller]
pub fn from_row_major_slice_with_stride<E: Entity>(
    slice: GroupFor<E, &[E::Unit]>,
    nrows: usize,
    ncols: usize,
    row_stride: usize,
) -> MatRef<'_, E> {
    from_column_major_slice_with_stride::<E>(slice, ncols, nrows, row_stride).transpose()
}

impl<'a, E: Entity> core::fmt::Debug for MatRef<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        struct DebugRow<'a, T: Entity>(MatRef<'a, T>);

        impl<'a, T: Entity> core::fmt::Debug for DebugRow<'a, T> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let mut j = 0;
                f.debug_list()
                    .entries(core::iter::from_fn(|| {
                        let ret = if j < self.0.ncols() {
                            Some(T::faer_from_units(T::faer_deref(self.0.get(0, j))))
                        } else {
                            None
                        };
                        j += 1;
                        ret
                    }))
                    .finish()
            }
        }

        writeln!(f, "[")?;
        for i in 0..self.nrows() {
            let row = self.subrows(i, 1);
            DebugRow(row).fmt(f)?;
            f.write_str(",\n")?;
        }
        write!(f, "]")
    }
}

impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for MatRef<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &E {
        self.get(row, col)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::Matrix<E> for MatRef<'_, E> {
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
impl<E: Entity> matrixcompare_core::DenseAccess<E> for MatRef<'_, E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

impl<E: Conjugate> ColBatch<E> for MatRef<'_, E> {
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
        <Self::Owned as ColBatch<E::Canonical>>::resize_owned(owned, nrows, ncols)
    }
}

impl<E: Conjugate> RowBatch<E> for MatRef<'_, E> {
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
        <Self::Owned as RowBatch<E::Canonical>>::resize_owned(owned, nrows, ncols)
    }
}

/// Returns a view over an `nrows×ncols` matrix containing `value` repeated for all elements.
#[doc(alias = "broadcast")]
pub fn from_repeated_ref<E: Entity>(
    value: GroupFor<E, &E::Unit>,
    nrows: usize,
    ncols: usize,
) -> MatRef<'_, E> {
    unsafe {
        from_raw_parts(
            E::faer_map(value, |ptr| ptr as *const E::Unit),
            nrows,
            ncols,
            0,
            0,
        )
    }
}

/// Returns a view over a matrix containing `col` repeated `ncols` times.
#[doc(alias = "broadcast")]
pub fn from_repeated_col<E: Entity>(col: ColRef<'_, E>, ncols: usize) -> MatRef<'_, E> {
    unsafe { from_raw_parts(col.as_ptr(), col.nrows(), ncols, col.row_stride(), 0) }
}

/// Returns a view over a matrix containing `row` repeated `nrows` times.
#[doc(alias = "broadcast")]
pub fn from_repeated_row<E: Entity>(row: RowRef<'_, E>, nrows: usize) -> MatRef<'_, E> {
    unsafe { from_raw_parts(row.as_ptr(), nrows, row.ncols(), 0, row.col_stride()) }
}

/// Returns a view over a `1×1` matrix containing value as its only element, pointing to `value`.
pub fn from_ref<E: Entity>(value: GroupFor<E, &E::Unit>) -> MatRef<'_, E> {
    from_repeated_ref(value, 1, 1)
}
