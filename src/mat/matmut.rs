use super::*;
use crate::{
    assert, debug_assert,
    diag::{DiagMut, DiagRef},
    iter,
    iter::chunks::ChunkPolicy,
    linalg::zip,
    unzipped, zipped, Idx, IdxInc, Unbind,
};
use core::ops::Range;

/// Mutable view over a matrix, similar to a mutable reference to a 2D strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `MatMut<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`MatMut::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
///
/// # Move semantics
/// Since `MatMut` mutably borrows data, it cannot be [`Copy`]. This means that if we pass a
/// `MatMut` to a function that takes it by value, or use a method that consumes `self` like
/// [`MatMut::transpose_mut`], this renders the original variable unusable.
/// ```compile_fail
/// use faer::{Mat, MatMut};
///
/// fn takes_matmut(view: MatMut<'_, f64>) {}
///
/// let mut matrix = Mat::new();
/// let view = matrix.as_mut();
///
/// takes_matmut(view); // `view` is moved (passed by value)
/// takes_matmut(view); // this fails to compile since `view` was moved
/// ```
/// The way to get around it is to use the [`reborrow::ReborrowMut`] trait, which allows us to
/// mutably borrow a `MatMut` to obtain another `MatMut` for the lifetime of the borrow.
/// It's also similarly possible to immutably borrow a `MatMut` to obtain a `MatRef` for the
/// lifetime of the borrow, using [`reborrow::Reborrow`].
/// ```
/// use faer::{Mat, MatMut, MatRef};
/// use reborrow::*;
///
/// fn takes_matmut(view: MatMut<'_, f64>) {}
/// fn takes_matref(view: MatRef<'_, f64>) {}
///
/// let mut matrix = Mat::new();
/// let mut view = matrix.as_mut();
///
/// takes_matmut(view.rb_mut());
/// takes_matmut(view.rb_mut());
/// takes_matref(view.rb());
/// // view is still usable here
/// ```
#[repr(C)]
pub struct MatMut<'a, E: Entity, R: Shape = usize, C: Shape = usize> {
    pub(super) inner: MatImpl<E, R, C>,
    pub(super) __marker: PhantomData<&'a E>,
}

impl<E: Entity> Default for MatMut<'_, E> {
    #[inline]
    fn default() -> Self {
        from_column_major_slice_mut_generic(
            map!(E, E::UNIT, |(())| &mut [] as &mut [E::Unit]),
            0,
            0,
        )
    }
}

impl<'short, E: Entity, R: Shape, C: Shape> Reborrow<'short> for MatMut<'_, E, R, C> {
    type Target = MatRef<'short, E, R, C>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        MatRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'short, E: Entity, R: Shape, C: Shape> ReborrowMut<'short> for MatMut<'_, E, R, C> {
    type Target = MatMut<'short, E, R, C>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        MatMut {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'a, E: Entity, R: Shape, C: Shape> IntoConst for MatMut<'a, E, R, C> {
    type Target = MatRef<'a, E, R, C>;

    #[inline]
    fn into_const(self) -> Self::Target {
        MatRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'a, E: Entity, R: Shape, C: Shape> MatMut<'a, E, R, C> {
    #[inline]
    pub(crate) unsafe fn __from_raw_parts(
        ptr: PtrMut<E>,
        nrows: R,
        ncols: C,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        Self {
            inner: MatImpl {
                ptr: into_copy::<E, _>(map!(E, ptr, |(ptr)| NonNull::new_unchecked(ptr),)),
                nrows,
                ncols,
                row_stride,
                col_stride,
            },
            __marker: PhantomData,
        }
    }

    /// Returns the number of rows of the matrix.
    #[inline(always)]
    pub fn nrows(&self) -> R {
        self.inner.nrows
    }
    /// Returns the number of columns of the matrix.
    #[inline(always)]
    pub fn ncols(&self) -> C {
        self.inner.ncols
    }

    /// Returns the number of rows and columns of the matrix.
    #[inline]
    pub fn shape(&self) -> (R, C) {
        (self.nrows(), self.ncols())
    }

    /// Returns pointers to the matrix data.
    #[inline(always)]
    pub fn as_ptr(self) -> PtrConst<E> {
        self.into_const().as_ptr()
    }

    /// Returns pointers to the matrix data.
    #[inline(always)]
    pub fn as_ptr_mut(self) -> PtrMut<E> {
        map!(E, from_copy::<E, _>(self.inner.ptr), |(ptr)| ptr.as_ptr(),)
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
    pub fn ptr_at(self, row: usize, col: usize) -> PtrConst<E> {
        self.into_const().ptr_at(row, col)
    }

    /// Returns raw pointers to the element at the given indices.
    #[inline(always)]
    pub fn ptr_at_mut(self, row: usize, col: usize) -> PtrMut<E> {
        let offset = ((row as isize).wrapping_mul(self.inner.row_stride))
            .wrapping_add((col as isize).wrapping_mul(self.inner.col_stride));
        map!(E, self.as_ptr_mut(), |(ptr)| ptr.wrapping_offset(offset),)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(self, row: usize, col: usize) -> PtrConst<E> {
        self.into_const().ptr_at_unchecked(row, col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_mut_unchecked(self, row: usize, col: usize) -> PtrMut<E> {
        let offset = crate::utils::unchecked_add(
            crate::utils::unchecked_mul(row, self.inner.row_stride),
            crate::utils::unchecked_mul(col, self.inner.col_stride),
        );
        map!(E, self.as_ptr_mut(), |(ptr)| ptr.offset(offset),)
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
    pub fn transpose(self) -> MatRef<'a, E, C, R> {
        self.into_const().transpose()
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
    pub fn transpose_mut(self) -> MatMut<'a, E, C, R> {
        unsafe {
            super::from_raw_parts_mut(
                map!(E, from_copy::<E, _>(self.inner.ptr), |(ptr)| ptr.as_ptr(),),
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
    pub fn conjugate(self) -> MatRef<'a, E::Conj, R, C>
    where
        E: Conjugate,
    {
        self.into_const().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline(always)]
    #[must_use]
    pub fn conjugate_mut(self) -> MatMut<'a, E::Conj, R, C>
    where
        E: Conjugate,
    {
        unsafe { self.into_const().conjugate().const_cast() }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn adjoint(self) -> MatRef<'a, E::Conj, C, R>
    where
        E: Conjugate,
    {
        self.into_const().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn adjoint_mut(self) -> MatMut<'a, E::Conj, C, R>
    where
        E: Conjugate,
    {
        self.transpose_mut().conjugate_mut()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    #[must_use]
    pub fn canonicalize(self) -> (MatRef<'a, E::Canonical, R, C>, Conj)
    where
        E: Conjugate,
    {
        self.into_const().canonicalize()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    #[must_use]
    pub fn canonicalize_mut(self) -> (MatMut<'a, E::Canonical, R, C>, Conj)
    where
        E: Conjugate,
    {
        let (canonical, conj) = self.into_const().canonicalize();
        unsafe { (canonical.const_cast(), conj) }
    }

    /// Returns the input matrix with dynamic shape.
    #[inline]
    pub fn as_dyn(self) -> MatRef<'a, E> {
        let nrows = self.nrows().unbound();
        let ncols = self.ncols().unbound();
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        unsafe { from_raw_parts(self.as_ptr(), nrows, ncols, row_stride, col_stride) }
    }

    /// Returns the input matrix with dynamic shape.
    #[inline]
    pub fn as_dyn_mut(self) -> MatMut<'a, E> {
        let nrows = self.nrows().unbound();
        let ncols = self.ncols().unbound();
        let row_stride = self.row_stride();
        let col_stride = self.col_stride();
        unsafe { from_raw_parts_mut(self.as_ptr_mut(), nrows, ncols, row_stride, col_stride) }
    }

    /// Returns the input matrix with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> MatRef<'a, E, V, H> {
        self.into_const().as_shape(nrows, ncols)
    }

    /// Returns the input matrix with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape_mut<V: Shape, H: Shape>(self, nrows: V, ncols: H) -> MatMut<'a, E, V, H> {
        unsafe { self.into_const().as_shape(nrows, ncols).const_cast() }
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(self, row: IdxInc<R>, col: IdxInc<C>) -> PtrConst<E> {
        self.into_const().overflowing_ptr_at(row, col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at_mut(self, row: IdxInc<R>, col: IdxInc<C>) -> PtrMut<E> {
        unsafe {
            let cond = (row != self.nrows()) & (col != self.ncols());
            let offset = (cond as usize).wrapping_neg() as isize
                & (isize::wrapping_add(
                    (row.unbound() as isize).wrapping_mul(self.inner.row_stride),
                    (col.unbound() as isize).wrapping_mul(self.inner.col_stride),
                ));
            map!(E, self.as_ptr_mut(), |(ptr)| ptr.offset(offset),)
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
    pub unsafe fn ptr_inbounds_at(self, row: Idx<R>, col: Idx<C>) -> PtrConst<E> {
        self.into_const().ptr_inbounds_at(row, col)
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
    pub unsafe fn ptr_inbounds_at_mut(self, row: Idx<R>, col: Idx<C>) -> PtrMut<E> {
        debug_assert!(all(row < self.nrows(), col < self.ncols()));
        self.ptr_at_mut_unchecked(row.unbound(), col.unbound())
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col(self, j: Idx<C>) -> Slice<'a, E> {
        self.into_const().try_get_contiguous_col(j)
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col_mut(self, j: Idx<C>) -> SliceMut<'a, E> {
        assert!(self.row_stride() == 1);
        let col = self.col_mut(j);
        let m = col.nrows().unbound();
        if m == 0 {
            map!(E, E::UNIT, |(())| &mut [] as &mut [E::Unit],)
        } else {
            map!(E, col.as_ptr_mut(), |(ptr)| unsafe {
                core::slice::from_raw_parts_mut(ptr, m)
            },)
        }
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
    pub unsafe fn split_at_unchecked(
        self,
        row: IdxInc<R>,
        col: IdxInc<C>,
    ) -> (
        MatRef<'a, E, usize, usize>,
        MatRef<'a, E, usize, usize>,
        MatRef<'a, E, usize, usize>,
        MatRef<'a, E, usize, usize>,
    ) {
        self.into_const().split_at_unchecked(row, col)
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
    pub fn split_at(
        self,
        row: IdxInc<R>,
        col: IdxInc<C>,
    ) -> (
        MatRef<'a, E, usize, usize>,
        MatRef<'a, E, usize, usize>,
        MatRef<'a, E, usize, usize>,
        MatRef<'a, E, usize, usize>,
    ) {
        self.into_const().split_at(row, col)
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
    pub unsafe fn split_at_mut_unchecked(
        self,
        row: IdxInc<R>,
        col: IdxInc<C>,
    ) -> (
        MatMut<'a, E, usize, usize>,
        MatMut<'a, E, usize, usize>,
        MatMut<'a, E, usize, usize>,
        MatMut<'a, E, usize, usize>,
    ) {
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
    pub fn split_at_mut(
        self,
        row: IdxInc<R>,
        col: IdxInc<C>,
    ) -> (
        MatMut<'a, E, usize, usize>,
        MatMut<'a, E, usize, usize>,
        MatMut<'a, E, usize, usize>,
        MatMut<'a, E, usize, usize>,
    ) {
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
    pub unsafe fn split_at_row_unchecked(
        self,
        row: IdxInc<R>,
    ) -> (MatRef<'a, E, usize, C>, MatRef<'a, E, usize, C>) {
        self.into_const().split_at_row_unchecked(row)
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
    pub fn split_at_row(
        self,
        row: IdxInc<R>,
    ) -> (MatRef<'a, E, usize, C>, MatRef<'a, E, usize, C>) {
        self.into_const().split_at_row(row)
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
    pub unsafe fn split_at_row_mut_unchecked(
        self,
        row: IdxInc<R>,
    ) -> (MatMut<'a, E, usize, C>, MatMut<'a, E, usize, C>) {
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
    pub fn split_at_row_mut(
        self,
        row: IdxInc<R>,
    ) -> (MatMut<'a, E, usize, C>, MatMut<'a, E, usize, C>) {
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
    pub unsafe fn split_at_col_unchecked(
        self,
        col: IdxInc<C>,
    ) -> (MatRef<'a, E, R, usize>, MatRef<'a, E, R, usize>) {
        self.into_const().split_at_col_unchecked(col)
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
    pub fn split_at_col(
        self,
        col: IdxInc<C>,
    ) -> (MatRef<'a, E, R, usize>, MatRef<'a, E, R, usize>) {
        self.into_const().split_at_col(col)
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
    pub unsafe fn split_at_col_mut_unchecked(
        self,
        col: IdxInc<C>,
    ) -> (MatMut<'a, E, R, usize>, MatMut<'a, E, R, usize>) {
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
    pub fn split_at_col_mut(
        self,
        col: IdxInc<C>,
    ) -> (MatMut<'a, E, R, usize>, MatMut<'a, E, R, usize>) {
        let (left, right) = self.into_const().split_at_col(col);
        unsafe { (left.const_cast(), right.const_cast()) }
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
    ) -> <MatRef<'a, E, R, C> as MatIndex<RowRange, ColRange>>::Target
    where
        MatRef<'a, E, R, C>: MatIndex<RowRange, ColRange>,
    {
        self.into_const().get_unchecked(row, col)
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
    ) -> <MatRef<'a, E, R, C> as MatIndex<RowRange, ColRange>>::Target
    where
        MatRef<'a, E, R, C>: MatIndex<RowRange, ColRange>,
    {
        self.into_const().get(row, col)
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

    /// Returns references to the element at the given indices.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be in `[0, self.nrows())`.
    /// * `col` must be in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn at_unchecked(self, row: Idx<R>, col: Idx<C>) -> Ref<'a, E> {
        unsafe { map!(E, self.ptr_inbounds_at(row, col), |(ptr)| &*ptr) }
    }

    /// Returns references to the element at the given indices.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be in `[0, self.nrows())`.
    /// * `col` must be in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub fn at(self, row: Idx<R>, col: Idx<C>) -> Ref<'a, E> {
        assert!(all(row < self.nrows(), col < self.ncols()));
        unsafe { map!(E, self.ptr_inbounds_at(row, col), |(ptr)| &*ptr) }
    }

    /// Returns references to the element at the given indices.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be in `[0, self.nrows())`.
    /// * `col` must be in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn at_mut_unchecked(self, row: Idx<R>, col: Idx<C>) -> Mut<'a, E> {
        unsafe { map!(E, self.ptr_inbounds_at_mut(row, col), |(ptr)| &mut *ptr) }
    }

    /// Returns references to the element at the given indices.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be in `[0, self.nrows())`.
    /// * `col` must be in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub fn at_mut(self, row: Idx<R>, col: Idx<C>) -> Mut<'a, E> {
        assert!(all(row < self.nrows(), col < self.ncols()));
        unsafe { map!(E, self.ptr_inbounds_at_mut(row, col), |(ptr)| &mut *ptr) }
    }

    /// Reads the value of the element at the given indices.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, row: Idx<R>, col: Idx<C>) -> E {
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
    pub fn read(&self, row: Idx<R>, col: Idx<C>) -> E {
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
    pub unsafe fn write_unchecked(&mut self, row: Idx<R>, col: Idx<C>, value: E) {
        let units = value.faer_into_units();
        let zipped = E::faer_zip(units, (*self).rb_mut().ptr_inbounds_at_mut(row, col));
        map!(E, zipped, |((unit, ptr))| *ptr = unit,);
    }

    /// Writes the value to the element at the given indices, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, row: Idx<R>, col: Idx<C>, value: E) {
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
    pub fn copy_from_triangular_lower<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsMatRef<ViewE, R = R, C = C>,
    ) {
        #[track_caller]
        #[inline(always)]
        fn implementation<R: Shape, C: Shape, E: Entity, ViewE: Conjugate<Canonical = E>>(
            this: MatMut<'_, E, R, C>,
            other: MatRef<'_, ViewE, R, C>,
        ) {
            zipped!(this, other).for_each_triangular_lower(
                zip::Diag::Include,
                #[inline(always)]
                |unzipped!(mut dst, src)| dst.write(src.read().canonicalize()),
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
    pub fn copy_from_strict_triangular_lower<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsMatRef<ViewE, R = R, C = C>,
    ) {
        #[track_caller]
        #[inline(always)]
        fn implementation<R: Shape, C: Shape, E: Entity, ViewE: Conjugate<Canonical = E>>(
            this: MatMut<'_, E, R, C>,
            other: MatRef<'_, ViewE, R, C>,
        ) {
            zipped!(this, other).for_each_triangular_lower(
                zip::Diag::Skip,
                #[inline(always)]
                |unzipped!(mut dst, src)| dst.write(src.read().canonicalize()),
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
    pub fn copy_from_triangular_upper<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsMatRef<ViewE, R = R, C = C>,
    ) {
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
    pub fn copy_from_strict_triangular_upper<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsMatRef<ViewE, R = R, C = C>,
    ) {
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
    pub fn copy_from<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsMatRef<ViewE, R = R, C = C>,
    ) {
        #[track_caller]
        #[inline(always)]
        fn implementation<R: Shape, C: Shape, E: Entity, ViewE: Conjugate<Canonical = E>>(
            this: MatMut<'_, E, R, C>,
            other: MatRef<'_, ViewE, R, C>,
        ) {
            zipped!(this, other)
                .for_each(|unzipped!(mut dst, src)| dst.write(src.read().canonicalize()));
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
    pub fn reverse_rows(self) -> MatRef<'a, E, R, C> {
        self.into_const().reverse_rows()
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
    /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_ref();
    /// let reversed_cols = view.reverse_cols();
    ///
    /// let expected = mat![[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]];
    /// assert_eq!(expected.as_ref(), reversed_cols);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols(self) -> MatRef<'a, E, R, C> {
        self.into_const().reverse_cols()
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
    /// let matrix = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let view = matrix.as_ref();
    /// let reversed = view.reverse_rows_and_cols();
    ///
    /// let expected = mat![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
    /// assert_eq!(expected.as_ref(), reversed);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows_and_cols(self) -> MatRef<'a, E, R, C> {
        self.into_const().reverse_rows_and_cols()
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
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row_start <= self.nrows()`.
    /// * `col_start <= self.ncols()`.
    /// * `nrows <= self.nrows() - row_start`.
    /// * `ncols <= self.ncols() - col_start`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn submatrix_unchecked<V: Shape, H: Shape>(
        self,
        row_start: IdxInc<R>,
        col_start: IdxInc<C>,
        nrows: V,
        ncols: H,
    ) -> MatRef<'a, E, V, H> {
        self.into_const()
            .submatrix_unchecked(row_start, col_start, nrows, ncols)
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
    pub unsafe fn submatrix_mut_unchecked<V: Shape, H: Shape>(
        self,
        row_start: IdxInc<R>,
        col_start: IdxInc<C>,
        nrows: V,
        ncols: H,
    ) -> MatMut<'a, E, V, H> {
        self.into_const()
            .submatrix_unchecked(row_start, col_start, nrows, ncols)
            .const_cast()
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
    pub fn submatrix<V: Shape, H: Shape>(
        self,
        row_start: IdxInc<R>,
        col_start: IdxInc<C>,
        nrows: V,
        ncols: H,
    ) -> MatRef<'a, E, V, H> {
        self.into_const()
            .submatrix(row_start, col_start, nrows, ncols)
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
    pub fn submatrix_mut<V: Shape, H: Shape>(
        self,
        row_start: IdxInc<R>,
        col_start: IdxInc<C>,
        nrows: V,
        ncols: H,
    ) -> MatMut<'a, E, V, H> {
        unsafe {
            self.into_const()
                .submatrix(row_start, col_start, nrows, ncols)
                .const_cast()
        }
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
    pub unsafe fn subrows_unchecked<V: Shape>(
        self,
        row_start: IdxInc<R>,
        nrows: V,
    ) -> MatRef<'a, E, V, C> {
        self.into_const().subrows_unchecked(row_start, nrows)
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
    pub unsafe fn subrows_mut_unchecked<V: Shape>(
        self,
        row_start: IdxInc<R>,
        nrows: V,
    ) -> MatMut<'a, E, V, C> {
        self.into_const()
            .subrows_unchecked(row_start, nrows)
            .const_cast()
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
    pub fn subrows<V: Shape>(self, row_start: IdxInc<R>, nrows: V) -> MatRef<'a, E, V, C> {
        self.into_const().subrows(row_start, nrows)
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
    pub fn subrows_mut<V: Shape>(self, row_start: IdxInc<R>, nrows: V) -> MatMut<'a, E, V, C> {
        unsafe { self.into_const().subrows(row_start, nrows).const_cast() }
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
    pub unsafe fn subcols_unchecked<H: Shape>(
        self,
        col_start: IdxInc<C>,
        ncols: H,
    ) -> MatRef<'a, E, R, H> {
        self.into_const().subcols_unchecked(col_start, ncols)
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
    pub unsafe fn subcols_mut_unchecked<H: Shape>(
        self,
        col_start: IdxInc<C>,
        ncols: H,
    ) -> MatMut<'a, E, R, H> {
        self.into_const()
            .subcols_unchecked(col_start, ncols)
            .const_cast()
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
    pub fn subcols<H: Shape>(self, col_start: IdxInc<C>, ncols: H) -> MatRef<'a, E, R, H> {
        self.into_const().subcols(col_start, ncols)
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
    pub fn subcols_mut<H: Shape>(self, col_start: IdxInc<C>, ncols: H) -> MatMut<'a, E, R, H> {
        unsafe { self.into_const().subcols(col_start, ncols).const_cast() }
    }

    #[track_caller]
    #[inline(always)]
    pub fn subcols_range(self, cols: Range<IdxInc<C>>) -> MatRef<'a, E, R, usize> {
        self.into_const().subcols_range(cols)
    }

    #[track_caller]
    #[inline(always)]
    pub fn subcols_range_mut(self, cols: Range<IdxInc<C>>) -> MatMut<'a, E, R, usize> {
        unsafe { self.into_const().subcols_range(cols).const_cast() }
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Safety
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn row_unchecked(self, row_idx: Idx<R>) -> RowRef<'a, E, C> {
        self.into_const().row_unchecked(row_idx)
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Safety
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn row_mut_unchecked(self, row_idx: Idx<R>) -> RowMut<'a, E, C> {
        self.into_const().row_unchecked(row_idx).const_cast()
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub fn row(self, row_idx: Idx<R>) -> RowRef<'a, E, C> {
        self.into_const().row(row_idx)
    }

    /// Returns a view over the row at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row_idx < self.nrows()`.
    #[track_caller]
    #[inline(always)]
    pub fn row_mut(self, row_idx: Idx<R>) -> RowMut<'a, E, C> {
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
    pub fn two_rows_mut(
        self,
        row_idx0: Idx<R>,
        row_idx1: Idx<R>,
    ) -> (RowMut<'a, E, C>, RowMut<'a, E, C>) {
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
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn col_unchecked(self, col_idx: Idx<C>) -> ColRef<'a, E, R> {
        self.into_const().col_unchecked(col_idx)
    }

    /// Returns a view over the column at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub unsafe fn col_mut_unchecked(self, col_idx: Idx<C>) -> ColMut<'a, E, R> {
        self.into_const().col_unchecked(col_idx).const_cast()
    }

    /// Returns a view over the column at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub fn col(self, col_idx: Idx<C>) -> ColRef<'a, E, R> {
        self.into_const().col(col_idx)
    }

    /// Returns a view over the column at the given index.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col_idx < self.ncols()`.
    #[track_caller]
    #[inline(always)]
    pub fn col_mut(self, col_idx: Idx<C>) -> ColMut<'a, E, R> {
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
    pub fn two_cols_mut(
        self,
        col_idx0: Idx<C>,
        col_idx1: Idx<C>,
    ) -> (ColMut<'a, E, R>, ColMut<'a, E, R>) {
        assert!(col_idx0 != col_idx1);
        let this = self.into_const();
        unsafe {
            (
                this.col(col_idx0).const_cast(),
                this.col(col_idx1).const_cast(),
            )
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

    /// Returns the L1 norm of `self`.
    #[inline]
    pub fn norm_l1(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.rb().norm_l1()
    }

    /// Returns the L2 norm of `self`.
    #[inline]
    pub fn norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.rb().norm_l2()
    }

    /// Returns the squared L2 norm of `self`.
    #[inline]
    pub fn squared_norm_l2(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.rb().squared_norm_l2()
    }

    /// Returns the sum of `self`.
    #[inline]
    pub fn sum(&self) -> E
    where
        E: ComplexField,
    {
        self.rb().sum()
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
        self.rb().kron(rhs)
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, E, R, C> {
        self.rb()
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> MatMut<'_, E, R, C> {
        self.rb_mut()
    }

    /// Returns a reference to the first column and a view over the remaining ones if the matrix has
    /// at least one column, otherwise `None`.
    #[inline]
    pub fn split_first_col(self) -> Option<(ColRef<'a, E, R>, MatRef<'a, E, R, usize>)> {
        self.into_const().split_first_col()
    }

    /// Returns a reference to the last column and a view over the remaining ones if the matrix has
    /// at least one column,  otherwise `None`.
    #[inline]
    pub fn split_last_col(self) -> Option<(ColRef<'a, E, R>, MatRef<'a, E, R, usize>)> {
        self.into_const().split_last_col()
    }

    /// Returns a reference to the first row and a view over the remaining ones if the matrix has
    /// at least one row, otherwise `None`.
    #[inline]
    pub fn split_first_row(self) -> Option<(RowRef<'a, E, C>, MatRef<'a, E, usize, C>)> {
        self.into_const().split_first_row()
    }

    /// Returns a reference to the last row and a view over the remaining ones if the matrix has
    /// at least one row,  otherwise `None`.
    #[inline]
    pub fn split_last_row(self) -> Option<(RowRef<'a, E, C>, MatRef<'a, E, usize, C>)> {
        self.into_const().split_last_row()
    }

    /// Returns a reference to the first column and a view over the remaining ones if the matrix has
    /// at least one column, otherwise `None`.
    #[inline]
    pub fn split_first_col_mut(self) -> Option<(ColMut<'a, E, R>, MatMut<'a, E, R, usize>)> {
        self.split_first_col()
            .map(|(head, tail)| unsafe { (head.const_cast(), tail.const_cast()) })
    }

    /// Returns a reference to the last column and a view over the remaining ones if the matrix has
    /// at least one column,  otherwise `None`.
    #[inline]
    pub fn split_last_col_mut(self) -> Option<(ColMut<'a, E, R>, MatMut<'a, E, R, usize>)> {
        self.split_last_col()
            .map(|(head, tail)| unsafe { (head.const_cast(), tail.const_cast()) })
    }

    /// Returns a reference to the first row and a view over the remaining ones if the matrix has
    /// at least one row, otherwise `None`.
    #[inline]
    pub fn split_first_row_mut(self) -> Option<(RowMut<'a, E, C>, MatMut<'a, E, usize, C>)> {
        self.split_first_row()
            .map(|(head, tail)| unsafe { (head.const_cast(), tail.const_cast()) })
    }

    /// Returns a reference to the last row and a view over the remaining ones if the matrix has
    /// at least one row,  otherwise `None`.
    #[inline]
    pub fn split_last_row_mut(self) -> Option<(RowMut<'a, E, C>, MatMut<'a, E, usize, C>)> {
        self.split_last_row()
            .map(|(head, tail)| unsafe { (head.const_cast(), tail.const_cast()) })
    }

    /// Returns an iterator over the columns of the matrix.
    #[inline]
    pub fn col_iter(self) -> iter::ColIter<'a, E, R> {
        self.into_const().col_iter()
    }

    /// Returns an iterator over the rows of the matrix.
    #[inline]
    pub fn row_iter(self) -> iter::RowIter<'a, E> {
        self.into_const().row_iter()
    }

    /// Returns an iterator over the columns of the matrix.
    #[inline]
    pub fn col_iter_mut(self) -> iter::ColIterMut<'a, E, R> {
        let nrows = self.nrows();
        let ncols = self.ncols();
        iter::ColIterMut {
            inner: self.as_shape_mut(nrows, ncols.unbound()),
        }
    }

    /// Returns an iterator over the rows of the matrix.
    #[inline]
    pub fn row_iter_mut(self) -> iter::RowIterMut<'a, E> {
        iter::RowIterMut {
            inner: self.as_dyn_mut(),
        }
    }

    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn const_cast(self) -> MatMut<'a, E, R, C> {
        self
    }

    /// Returns an iterator that provides successive chunks of the columns of this matrix, with
    /// each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have
    /// `chunk_size` columns.
    #[inline]
    #[track_caller]
    pub fn col_chunks(self, chunk_size: usize) -> iter::ColChunks<'a, E> {
        self.into_const().col_chunks(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the columns of this
    /// matrix.
    ///
    /// # Panics
    /// Panics if `count == 0`.
    #[inline]
    #[track_caller]
    pub fn col_partition(self, count: usize) -> iter::ColPartition<'a, E> {
        self.into_const().col_partition(count)
    }

    /// Returns an iterator that provides successive chunks of the rows of this matrix, with
    /// each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn row_chunks(self, chunk_size: usize) -> iter::RowChunks<'a, E> {
        self.into_const().row_chunks(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the rows of this
    /// matrix.
    ///
    /// # Panics
    /// Panics if `count == 0`.
    #[inline]
    #[track_caller]
    pub fn row_partition(self, count: usize) -> iter::RowPartition<'a, E> {
        self.into_const().row_partition(count)
    }

    /// Returns an iterator that provides successive chunks of the columns of this matrix, with
    /// each having at most `chunk_size` columns.
    ///
    /// If the number of columns is a multiple of `chunk_size`, then all chunks have
    /// `chunk_size` columns.
    #[inline]
    #[track_caller]
    pub fn col_chunks_mut(self, chunk_size: usize) -> iter::ColChunksMut<'a, E> {
        assert!(chunk_size > 0);
        let ncols = self.ncols().unbound();
        iter::ColChunksMut {
            inner: self.as_dyn_mut(),
            policy: iter::chunks::ChunkSizePolicy::new(ncols, iter::chunks::ChunkSize(chunk_size)),
        }
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the columns of this
    /// matrix.
    ///
    /// # Panics
    /// Panics if `count == 0`.
    #[inline]
    #[track_caller]
    pub fn col_partition_mut(self, count: usize) -> iter::ColPartitionMut<'a, E> {
        assert!(count > 0);
        let ncols = self.ncols().unbound();
        iter::ColPartitionMut {
            inner: self.as_dyn_mut(),
            policy: iter::chunks::PartitionCountPolicy::new(
                ncols,
                iter::chunks::PartitionCount(count),
            ),
        }
    }

    /// Returns an iterator that provides successive chunks of the rows of this matrix, with
    /// each having at most `chunk_size` rows.
    ///
    /// If the number of rows is a multiple of `chunk_size`, then all chunks have `chunk_size`
    /// rows.
    #[inline]
    #[track_caller]
    pub fn row_chunks_mut(self, chunk_size: usize) -> iter::RowChunksMut<'a, E> {
        assert!(chunk_size > 0);
        let nrows = self.nrows().unbound();
        iter::RowChunksMut {
            inner: self.as_dyn_mut(),
            policy: iter::chunks::ChunkSizePolicy::new(nrows, iter::chunks::ChunkSize(chunk_size)),
        }
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the rows of this
    /// matrix.
    ///
    /// # Panics
    /// Panics if `count == 0`.
    #[inline]
    #[track_caller]
    pub fn row_partition_mut(self, count: usize) -> iter::RowPartitionMut<'a, E> {
        assert!(count > 0);
        let nrows = self.nrows().unbound();
        iter::RowPartitionMut {
            inner: self.as_dyn_mut(),
            policy: iter::chunks::PartitionCountPolicy::new(
                nrows,
                iter::chunks::PartitionCount(count),
            ),
        }
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
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E, R, usize>> {
        self.into_const().par_col_chunks(chunk_size)
    }

    /// Returns a parallel iterator that provides exactly `count` successive chunks of the columns
    /// of this matrix.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_col_partition(
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E, R, usize>> {
        self.into_const().par_col_partition(count)
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
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E, usize, C>> {
        self.into_const().par_row_chunks(chunk_size)
    }

    /// Returns a parallel iterator that provides exactly `count` successive chunks of the rows
    /// of this matrix.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_row_partition(
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatRef<'a, E, usize, C>> {
        self.into_const().par_row_partition(count)
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
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E, R, usize>> {
        use rayon::prelude::*;
        self.into_const()
            .par_col_chunks(chunk_size)
            .map(|chunk| unsafe { chunk.const_cast() })
    }

    /// Returns a parallel iterator that provides exactly `count` successive chunks of the columns
    /// of this matrix.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_col_partition_mut(
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E, R, usize>> {
        use rayon::prelude::*;
        self.into_const()
            .par_col_partition(count)
            .map(|x| unsafe { x.const_cast() })
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
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E, usize, C>> {
        use rayon::prelude::*;
        self.into_const()
            .par_row_chunks(chunk_size)
            .map(|chunk| unsafe { chunk.const_cast() })
    }

    /// Returns a parallel iterator that provides exactly `count` successive chunks of the rows
    /// of this matrix.
    ///
    /// Only available with the `rayon` feature.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    #[inline]
    #[track_caller]
    pub fn par_row_partition_mut(
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = MatMut<'a, E, usize, C>> {
        use rayon::prelude::*;
        self.into_const()
            .par_row_partition(count)
            .map(|x| unsafe { x.const_cast() })
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whose diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal(self) -> DiagRef<'a, E, R> {
        self.into_const().column_vector_as_diagonal()
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whose diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal_mut(self) -> DiagMut<'a, E, R> {
        assert!(self.ncols().unbound() == 1);
        DiagMut {
            inner: self.col_mut(unsafe { Idx::<C>::new_unbound(0) }),
        }
    }
}

impl<'a, E: Entity, N: Shape> MatMut<'a, E, N, N> {
    /// Returns the diagonal of the matrix.
    #[inline(always)]
    pub fn diagonal(self) -> DiagRef<'a, E, N> {
        self.into_const().diagonal()
    }

    /// Returns the diagonal of the matrix.
    #[inline(always)]
    pub fn diagonal_mut(self) -> DiagMut<'a, E, N> {
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
}

impl<'a, E: RealField, R: Shape, C: Shape> MatMut<'a, num_complex::Complex<E>, R, C> {
    /// Returns the real and imaginary components of `self`.
    #[inline(always)]
    pub fn real_imag(self) -> num_complex::Complex<MatRef<'a, E, R, C>> {
        self.into_const().real_imag()
    }

    /// Returns the real and imaginary components of `self`.
    #[inline(always)]
    pub fn real_imag_mut(self) -> num_complex::Complex<MatMut<'a, E, R, C>> {
        let num_complex::Complex { re, im } = self.into_const().real_imag();
        unsafe {
            num_complex::Complex {
                re: re.const_cast(),
                im: im.const_cast(),
            }
        }
    }
}

impl<E: Entity, R: Shape, C: Shape> AsMatRef<E> for MatMut<'_, E, R, C> {
    type R = R;
    type C = C;

    #[inline]
    fn as_mat_ref(&self) -> MatRef<'_, E, R, C> {
        (*self).rb()
    }
}

impl<E: Entity, R: Shape, C: Shape> AsMatMut<E> for MatMut<'_, E, R, C> {
    #[inline]
    fn as_mat_mut(&mut self) -> MatMut<'_, E, R, C> {
        (*self).rb_mut()
    }
}

impl<E: Entity, R: Shape, C: Shape> As2D<E> for MatMut<'_, E, R, C> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (*self).rb().as_dyn()
    }
}

impl<E: Entity, R: Shape, C: Shape> As2DMut<E> for MatMut<'_, E, R, C> {
    #[inline]
    fn as_2d_mut(&mut self) -> MatMut<'_, E> {
        (*self).rb_mut().as_dyn_mut()
    }
}

/// Creates a `MatMut` from pointers to the matrix data, dimensions, and strides.
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
/// * For each matrix unit, the corresponding pointer must be non null and properly aligned,
/// even for a zero-sized matrix.
/// * The values accessible by the matrix must be initialized at some point before they are read, or
/// references to them are formed.
/// * No aliasing (including self aliasing) is allowed. In other words, none of the elements
/// accessible by any matrix unit may be accessed for reads or writes by any other means for
/// the duration of the lifetime `'a`. No two elements within a single matrix unit may point to
/// the same address (such a thing can be achieved with a zero stride, for example), and no two
/// matrix units may point to the same address.
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
/// let mut data = [[1.0, 2.0, 3.0, f64::NAN], [4.0, 5.0, 6.0, f64::NAN]];
/// let mut matrix =
///     unsafe { mat::from_raw_parts_mut::<f64, _, _>(data.as_mut_ptr() as *mut f64, 2, 3, 4, 1) };
///
/// let expected = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// assert_eq!(expected.as_ref(), matrix);
/// ```
#[inline(always)]
pub unsafe fn from_raw_parts_mut<'a, E: Entity, R: Shape, C: Shape>(
    ptr: PtrMut<E>,
    nrows: R,
    ncols: C,
    row_stride: isize,
    col_stride: isize,
) -> MatMut<'a, E, R, C> {
    MatMut::__from_raw_parts(ptr, nrows, ncols, row_stride, col_stride)
}

/// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
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
/// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
/// let view = mat::from_column_major_slice_mut(&mut slice, 3, 2);
///
/// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
/// assert_eq!(expected, view);
/// ```
#[track_caller]
pub fn from_column_major_slice_mut_generic<E: Entity, R: Shape, C: Shape>(
    slice: SliceMut<'_, E>,
    nrows: R,
    ncols: C,
) -> MatMut<'_, E, R, C> {
    from_slice_assert(
        nrows.unbound(),
        ncols.unbound(),
        SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&slice))).len(),
    );
    unsafe {
        from_raw_parts_mut(
            map!(E, slice, |(slice)| slice.as_mut_ptr(),),
            nrows,
            ncols,
            1,
            nrows.unbound() as isize,
        )
    }
}

/// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
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
/// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
/// let view = mat::from_row_major_slice_mut(&mut slice, 3, 2);
///
/// let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// assert_eq!(expected, view);
/// ```
#[inline(always)]
#[track_caller]
pub fn from_row_major_slice_mut_generic<E: Entity, R: Shape, C: Shape>(
    slice: SliceMut<'_, E>,
    nrows: R,
    ncols: C,
) -> MatMut<'_, E, R, C> {
    from_column_major_slice_mut_generic(slice, ncols, nrows).transpose_mut()
}

/// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a column-major format, where the beginnings of two consecutive
/// columns are separated by `col_stride` elements.
#[track_caller]
pub fn from_column_major_slice_with_stride_mut_generic<E: Entity, R: Shape, C: Shape>(
    slice: SliceMut<'_, E>,
    nrows: R,
    ncols: C,
    col_stride: usize,
) -> MatMut<'_, E, R, C> {
    from_strided_column_major_slice_mut_assert(
        nrows.unbound(),
        ncols.unbound(),
        col_stride,
        SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&slice))).len(),
    );
    unsafe {
        from_raw_parts_mut(
            map!(E, slice, |(slice)| slice.as_mut_ptr(),),
            nrows,
            ncols,
            1,
            col_stride as isize,
        )
    }
}

/// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a row-major format, where the beginnings of two consecutive
/// rows are separated by `row_stride` elements.
#[track_caller]
pub fn from_row_major_slice_with_stride_mut_generic<E: Entity, R: Shape, C: Shape>(
    slice: SliceMut<'_, E>,
    nrows: R,
    ncols: C,
    row_stride: usize,
) -> MatMut<'_, E, R, C> {
    from_column_major_slice_with_stride_mut_generic(slice, ncols, nrows, row_stride).transpose_mut()
}

/// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
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
/// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
/// let view = mat::from_column_major_slice_mut(&mut slice, 3, 2);
///
/// let expected = mat![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
/// assert_eq!(expected, view);
/// ```
#[track_caller]
pub fn from_column_major_slice_mut<E: SimpleEntity, R: Shape, C: Shape>(
    slice: &mut [E],
    nrows: R,
    ncols: C,
) -> MatMut<'_, E, R, C> {
    from_column_major_slice_mut_generic(slice, nrows, ncols)
}

/// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
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
/// let mut slice = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
/// let view = mat::from_row_major_slice_mut(&mut slice, 3, 2);
///
/// let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// assert_eq!(expected, view);
/// ```
#[inline(always)]
#[track_caller]
pub fn from_row_major_slice_mut<E: SimpleEntity, R: Shape, C: Shape>(
    slice: &mut [E],
    nrows: R,
    ncols: C,
) -> MatMut<'_, E, R, C> {
    from_row_major_slice_mut_generic(slice, nrows, ncols)
}

/// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a column-major format, where the beginnings of two consecutive
/// columns are separated by `col_stride` elements.
#[track_caller]
pub fn from_column_major_slice_with_stride_mut<E: SimpleEntity, R: Shape, C: Shape>(
    slice: &mut [E],
    nrows: R,
    ncols: C,
    col_stride: usize,
) -> MatMut<'_, E, R, C> {
    from_column_major_slice_with_stride_mut_generic(slice, nrows, ncols, col_stride)
}

/// Creates a `MatMut` from slice views over the matrix data, and the matrix dimensions.
/// The data is interpreted in a row-major format, where the beginnings of two consecutive
/// rows are separated by `row_stride` elements.
#[track_caller]
pub fn from_row_major_slice_with_stride_mut<E: SimpleEntity, R: Shape, C: Shape>(
    slice: &mut [E],
    nrows: R,
    ncols: C,
    row_stride: usize,
) -> MatMut<'_, E, R, C> {
    from_row_major_slice_with_stride_mut_generic(slice, nrows, ncols, row_stride)
}

impl<'a, E: Entity, R: Shape, C: Shape> core::fmt::Debug for MatMut<'a, E, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<E: SimpleEntity> core::ops::Index<(usize, usize)> for MatMut<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &E {
        (*self).rb().get(row, col)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for MatMut<'_, E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut E {
        (*self).rb_mut().get_mut(row, col)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<E: Entity> matrixcompare_core::Matrix<E> for MatMut<'_, E> {
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
impl<E: Entity> matrixcompare_core::DenseAccess<E> for MatMut<'_, E> {
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> E {
        self.read(row, col)
    }
}

impl<E: Conjugate> ColBatch<E> for MatMut<'_, E> {
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

impl<E: Conjugate> ColBatchMut<E> for MatMut<'_, E> {}

impl<E: Conjugate> RowBatch<E> for MatMut<'_, E> {
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

impl<E: Conjugate> RowBatchMut<E> for MatMut<'_, E> {}

/// Returns a view over a `1×1` matrix containing value as its only element, pointing to `value`.
pub fn from_mut<E: SimpleEntity>(value: &mut E) -> MatMut<'_, E> {
    from_mut_generic(value)
}

/// Returns a view over a `1×1` matrix containing value as its only element, pointing to `value`.
pub fn from_mut_generic<E: Entity>(value: Mut<'_, E>) -> MatMut<'_, E> {
    unsafe { from_raw_parts_mut(map!(E, value, |(ptr)| ptr as *mut E::Unit), 1, 1, 0, 0) }
}
