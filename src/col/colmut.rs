use super::*;
use crate::{
    diag::{DiagMut, DiagRef},
    mat,
    row::{RowMut, RowRef},
    unzipped, zipped,
};
use core::mem::MaybeUninit;

/// Mutable view over a column vector, similar to a mutable reference to a strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `ColMut<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`ColMut::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
///
/// # Move semantics
/// See [`faer::Mat`](crate::Mat) for information about reborrowing when using this type.
#[repr(C)]
pub struct ColMut<'a, E: Entity> {
    pub(super) inner: VecImpl<E>,
    pub(super) __marker: PhantomData<&'a E>,
}

impl<'short, E: Entity> Reborrow<'short> for ColMut<'_, E> {
    type Target = ColRef<'short, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        ColRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'short, E: Entity> ReborrowMut<'short> for ColMut<'_, E> {
    type Target = ColMut<'short, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        ColMut {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'a, E: Entity> IntoConst for ColMut<'a, E> {
    type Target = ColRef<'a, E>;

    #[inline]
    fn into_const(self) -> Self::Target {
        ColRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'a, E: Entity> ColMut<'a, E> {
    #[inline]
    pub(crate) unsafe fn __from_raw_parts(
        ptr: GroupFor<E, *mut E::Unit>,
        nrows: usize,
        row_stride: isize,
    ) -> Self {
        Self {
            inner: VecImpl {
                ptr: into_copy::<E, _>(E::faer_map(
                    ptr,
                    #[inline]
                    |ptr| NonNull::new_unchecked(ptr),
                )),
                len: nrows,
                stride: row_stride,
            },
            __marker: PhantomData,
        }
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col(self) -> GroupFor<E, &'a [E::Unit]> {
        self.into_const().try_get_contiguous_col()
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col_mut(self) -> GroupFor<E, &'a mut [E::Unit]> {
        assert!(self.row_stride() == 1);
        let m = self.nrows();
        E::faer_map(
            self.as_ptr_mut(),
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
        )
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

    /// Returns pointers to the matrix data.
    #[inline(always)]
    pub fn as_ptr(self) -> GroupFor<E, *const E::Unit> {
        self.into_const().as_ptr()
    }

    /// Returns pointers to the matrix data.
    #[inline(always)]
    pub fn as_ptr_mut(self) -> GroupFor<E, *mut E::Unit> {
        E::faer_map(
            from_copy::<E, _>(self.inner.ptr),
            #[inline(always)]
            |ptr| ptr.as_ptr() as *mut E::Unit,
        )
    }

    /// Returns the row stride of the matrix, specified in number of elements, not in bytes.
    #[inline(always)]
    pub fn row_stride(&self) -> isize {
        self.inner.stride
    }

    /// Returns `self` as a matrix view.
    #[inline(always)]
    pub fn as_2d(self) -> MatRef<'a, E> {
        self.into_const().as_2d()
    }

    /// Returns `self` as a mutable matrix view.
    #[inline(always)]
    pub fn as_2d_mut(self) -> MatMut<'a, E> {
        let nrows = self.nrows();
        let row_stride = self.row_stride();
        unsafe { mat::from_raw_parts_mut(self.as_ptr_mut(), nrows, 1, row_stride, isize::MAX) }
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at(self, row: usize) -> GroupFor<E, *const E::Unit> {
        self.into_const().ptr_at(row)
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at_mut(self, row: usize) -> GroupFor<E, *mut E::Unit> {
        let offset = (row as isize).wrapping_mul(self.inner.stride);

        E::faer_map(
            self.as_ptr_mut(),
            #[inline(always)]
            |ptr| ptr.wrapping_offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(self, row: usize) -> GroupFor<E, *const E::Unit> {
        self.into_const().ptr_at_unchecked(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_mut_unchecked(self, row: usize) -> GroupFor<E, *mut E::Unit> {
        let offset = crate::utils::unchecked_mul(row, self.inner.stride);
        E::faer_map(
            self.as_ptr_mut(),
            #[inline(always)]
            |ptr| ptr.offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(self, row: usize) -> GroupFor<E, *const E::Unit> {
        self.into_const().overflowing_ptr_at(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at_mut(self, row: usize) -> GroupFor<E, *mut E::Unit> {
        unsafe {
            let cond = row != self.nrows();
            let offset = (cond as usize).wrapping_neg() as isize
                & (row as isize).wrapping_mul(self.inner.stride);
            E::faer_map(
                self.as_ptr_mut(),
                #[inline(always)]
                |ptr| ptr.offset(offset),
            )
        }
    }

    /// Returns raw pointers to the element at the given index, assuming the provided index
    /// is within the size of the vector.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(self, row: usize) -> GroupFor<E, *const E::Unit> {
        self.into_const().ptr_inbounds_at(row)
    }

    /// Returns raw pointers to the element at the given index, assuming the provided index
    /// is within the size of the vector.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(self, row: usize) -> GroupFor<E, *mut E::Unit> {
        debug_assert!(row < self.nrows());
        self.ptr_at_mut_unchecked(row)
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
    pub unsafe fn split_at_unchecked(self, row: usize) -> (ColRef<'a, E>, ColRef<'a, E>) {
        self.into_const().split_at_unchecked(row)
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
    pub unsafe fn split_at_mut_unchecked(self, row: usize) -> (Self, Self) {
        let (top, bot) = self.into_const().split_at_unchecked(row);
        unsafe { (top.const_cast(), bot.const_cast()) }
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
    pub fn split_at(self, row: usize) -> (ColRef<'a, E>, ColRef<'a, E>) {
        self.into_const().split_at(row)
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
    pub fn split_at_mut(self, row: usize) -> (Self, Self) {
        assert!(row <= self.nrows());
        unsafe { self.split_at_mut_unchecked(row) }
    }

    /// Returns references to the element at the given index, or subvector if `row` is a
    /// range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn get_unchecked<RowRange>(
        self,
        row: RowRange,
    ) -> <ColRef<'a, E> as ColIndex<RowRange>>::Target
    where
        ColRef<'a, E>: ColIndex<RowRange>,
    {
        self.into_const().get_unchecked(row)
    }

    /// Returns references to the element at the given index, or subvector if `row` is a
    /// range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn get_mut_unchecked<RowRange>(
        self,
        row: RowRange,
    ) -> <Self as ColIndex<RowRange>>::Target
    where
        Self: ColIndex<RowRange>,
    {
        <Self as ColIndex<RowRange>>::get_unchecked(self, row)
    }

    /// Returns references to the element at the given index, or subvector if `row` is a
    /// range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline(always)]
    #[track_caller]
    pub fn get<RowRange>(self, row: RowRange) -> <ColRef<'a, E> as ColIndex<RowRange>>::Target
    where
        ColRef<'a, E>: ColIndex<RowRange>,
    {
        self.into_const().get(row)
    }

    /// Returns references to the element at the given index, or subvector if `row` is a
    /// range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be contained in `[0, self.nrows())`.
    #[inline(always)]
    #[track_caller]
    pub fn get_mut<RowRange>(self, row: RowRange) -> <Self as ColIndex<RowRange>>::Target
    where
        Self: ColIndex<RowRange>,
    {
        <Self as ColIndex<RowRange>>::get(self, row)
    }

    /// Reads the value of the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, row: usize) -> E {
        self.rb().read_unchecked(row)
    }

    /// Reads the value of the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, row: usize) -> E {
        self.rb().read(row)
    }

    /// Writes the value to the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, row: usize, value: E) {
        let units = value.faer_into_units();
        let zipped = E::faer_zip(units, (*self).rb_mut().ptr_inbounds_at_mut(row));
        E::faer_map(
            zipped,
            #[inline(always)]
            |(unit, ptr)| *ptr = unit,
        );
    }

    /// Writes the value to the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, row: usize, value: E) {
        assert!(row < self.nrows());
        unsafe { self.write_unchecked(row, value) };
    }

    /// Copies the values from `other` into `self`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.nrows() == other.nrows()`.
    /// * `self.ncols() == other.ncols()`.
    #[track_caller]
    pub fn copy_from<ViewE: Conjugate<Canonical = E>>(&mut self, other: impl AsColRef<ViewE>) {
        #[track_caller]
        #[inline(always)]
        fn implementation<E: Entity, ViewE: Conjugate<Canonical = E>>(
            this: ColMut<'_, E>,
            other: ColRef<'_, ViewE>,
        ) {
            zipped!(this.as_2d_mut(), other.as_2d())
                .for_each(|unzipped!(mut dst, src)| dst.write(src.read().canonicalize()));
        }
        implementation(self.rb_mut(), other.as_col_ref())
    }

    /// Fills the elements of `self` with zeros.
    #[track_caller]
    pub fn fill_zero(&mut self)
    where
        E: ComplexField,
    {
        zipped!(self.rb_mut().as_2d_mut()).for_each(
            #[inline(always)]
            |unzipped!(mut x)| x.write(E::faer_zero()),
        );
    }

    /// Fills the elements of `self` with copies of `constant`.
    #[track_caller]
    pub fn fill(&mut self, constant: E) {
        zipped!((*self).rb_mut().as_2d_mut()).for_each(
            #[inline(always)]
            |unzipped!(mut x)| x.write(constant),
        );
    }

    /// Returns a view over the transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn transpose(self) -> RowRef<'a, E> {
        self.into_const().transpose()
    }

    /// Returns a view over the transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn transpose_mut(self) -> RowMut<'a, E> {
        unsafe { self.into_const().transpose().const_cast() }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline(always)]
    #[must_use]
    pub fn conjugate(self) -> ColRef<'a, E::Conj>
    where
        E: Conjugate,
    {
        self.into_const().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline(always)]
    #[must_use]
    pub fn conjugate_mut(self) -> ColMut<'a, E::Conj>
    where
        E: Conjugate,
    {
        unsafe { self.into_const().conjugate().const_cast() }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    pub fn adjoint(self) -> RowRef<'a, E::Conj>
    where
        E: Conjugate,
    {
        self.into_const().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    pub fn adjoint_mut(self) -> RowMut<'a, E::Conj>
    where
        E: Conjugate,
    {
        self.conjugate_mut().transpose_mut()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize(self) -> (ColRef<'a, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        self.into_const().canonicalize()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize_mut(self) -> (ColMut<'a, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        let (canon, conj) = self.into_const().canonicalize();
        unsafe { (canon.const_cast(), conj) }
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows(self) -> ColRef<'a, E> {
        self.into_const().reverse_rows()
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows_mut(self) -> Self {
        unsafe { self.into_const().reverse_rows().const_cast() }
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
    pub unsafe fn subrows_unchecked(self, row_start: usize, nrows: usize) -> ColRef<'a, E> {
        self.into_const().subrows_unchecked(row_start, nrows)
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
    pub unsafe fn subrows_mut_unchecked(self, row_start: usize, nrows: usize) -> Self {
        self.into_const()
            .subrows_unchecked(row_start, nrows)
            .const_cast()
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
    pub fn subrows(self, row_start: usize, nrows: usize) -> ColRef<'a, E> {
        self.into_const().subrows(row_start, nrows)
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
    pub fn subrows_mut(self, row_start: usize, nrows: usize) -> Self {
        unsafe { self.into_const().subrows(row_start, nrows).const_cast() }
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whose diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal(self) -> DiagRef<'a, E> {
        self.into_const().column_vector_as_diagonal()
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whose diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal_mut(self) -> DiagMut<'a, E> {
        DiagMut { inner: self }
    }

    /// Returns an owning [`Col`] of the data.
    #[inline]
    pub fn to_owned(&self) -> Col<E::Canonical>
    where
        E: Conjugate,
    {
        (*self).rb().to_owned()
    }

    /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
    #[inline]
    pub fn has_nan(&self) -> bool
    where
        E: ComplexField,
    {
        (*self).rb().as_2d().has_nan()
    }

    /// Returns `true` if all of the elements are finite, otherwise returns `false`.
    #[inline]
    pub fn is_all_finite(&self) -> bool
    where
        E: ComplexField,
    {
        (*self).rb().as_2d().is_all_finite()
    }

    /// Returns the maximum norm of `self`.
    #[inline]
    pub fn norm_max(&self) -> E::Real
    where
        E: ComplexField,
    {
        self.rb().as_2d().norm_max()
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
        self.rb().as_2d().sum()
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
        self.as_ref().kron(rhs)
    }

    /// Returns the column as a contiguous slice if its row stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice(self) -> Option<GroupFor<E, &'a [E::Unit]>> {
        self.into_const().try_as_slice()
    }

    /// Returns the column as a contiguous slice if its row stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice_mut(self) -> Option<GroupFor<E, &'a mut [E::Unit]>> {
        if self.row_stride() == 1 {
            let len = self.nrows();
            Some(E::faer_map(
                self.as_ptr_mut(),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, len) },
            ))
        } else {
            None
        }
    }

    /// Returns the column as a contiguous potentially uninitialized slice if its row stride is
    /// equal to `1`.
    ///
    /// # Safety
    /// If uninit data is written to the slice, it must not be read at some later point.
    pub unsafe fn try_as_uninit_slice_mut(
        self,
    ) -> Option<GroupFor<E, &'a mut [MaybeUninit<E::Unit>]>> {
        if self.row_stride() == 1 {
            let len = self.nrows();
            Some(E::faer_map(
                self.as_ptr_mut(),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts_mut(ptr as _, len) },
            ))
        } else {
            None
        }
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> ColRef<'_, E> {
        (*self).rb()
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> ColMut<'_, E> {
        (*self).rb_mut()
    }

    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn const_cast(self) -> ColMut<'a, E> {
        self
    }
}

/// Creates a `ColMut` from pointers to the column vector data, number of rows, and row stride.
///
/// # Safety:
/// This function has the same safety requirements as
/// [`mat::from_raw_parts_mut(ptr, nrows, 1, row_stride, 0)`]
#[inline(always)]
pub unsafe fn from_raw_parts_mut<'a, E: Entity>(
    ptr: GroupFor<E, *mut E::Unit>,
    nrows: usize,
    row_stride: isize,
) -> ColMut<'a, E> {
    ColMut::__from_raw_parts(ptr, nrows, row_stride)
}

/// Creates a `ColMut` from slice views over the column vector data, The result has the same
/// number of rows as the length of the input slice.
#[inline(always)]
pub fn from_slice_mut_generic<E: Entity>(slice: GroupFor<E, &mut [E::Unit]>) -> ColMut<'_, E> {
    let nrows = SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&slice))).len();

    unsafe {
        from_raw_parts_mut(
            E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.as_mut_ptr(),
            ),
            nrows,
            1,
        )
    }
}

/// Creates a `ColMut` from slice views over the column vector data, The result has the same
/// number of rows as the length of the input slice.
#[inline(always)]
pub fn from_slice_mut<E: SimpleEntity>(slice: &mut [E]) -> ColMut<'_, E> {
    from_slice_mut_generic(slice)
}

impl<E: Entity> As2D<E> for ColMut<'_, E> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (*self).rb().as_2d()
    }
}

impl<E: Entity> As2DMut<E> for ColMut<'_, E> {
    #[inline]
    fn as_2d_mut(&mut self) -> MatMut<'_, E> {
        (*self).rb_mut().as_2d_mut()
    }
}

impl<E: Entity> AsColRef<E> for ColMut<'_, E> {
    #[inline]
    fn as_col_ref(&self) -> ColRef<'_, E> {
        (*self).rb()
    }
}
impl<E: Entity> AsColMut<E> for ColMut<'_, E> {
    #[inline]
    fn as_col_mut(&mut self) -> ColMut<'_, E> {
        (*self).rb_mut()
    }
}

impl<'a, E: Entity> core::fmt::Debug for ColMut<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<E: SimpleEntity> core::ops::Index<usize> for ColMut<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, row: usize) -> &E {
        (*self).rb().get(row)
    }
}

impl<E: SimpleEntity> core::ops::IndexMut<usize> for ColMut<'_, E> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, row: usize) -> &mut E {
        (*self).rb_mut().get_mut(row)
    }
}

impl<E: Conjugate> ColBatch<E> for ColMut<'_, E> {
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
        Self::Owned::resize_owned(owned, nrows, ncols)
    }
}
impl<E: Conjugate> ColBatchMut<E> for ColMut<'_, E> {}

/// Returns a view over a column with 1 row containing value as its only element, pointing to
/// `value`.
pub fn from_mut<E: SimpleEntity>(value: &mut E) -> ColMut<'_, E> {
    from_mut_generic(value)
}

/// Returns a view over a column with 1 row containing value as its only element, pointing to
/// `value`.
pub fn from_mut_generic<E: Entity>(value: GroupFor<E, &mut E::Unit>) -> ColMut<'_, E> {
    unsafe { from_raw_parts_mut(E::faer_map(value, |ptr| ptr as *mut E::Unit), 1, 1) }
}
