use super::*;
use crate::{
    diag::{DiagMut, DiagRef},
    iter,
    iter::chunks::ChunkPolicy,
    row::{RowMut, RowRef},
    unzipped, zipped, Idx, IdxInc, Unbind,
};

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
pub struct ColMut<'a, E: Entity, R: Shape = usize> {
    pub(super) inner: VecImpl<E, R>,
    pub(super) __marker: PhantomData<&'a E>,
}

impl<E: Entity> Default for ColMut<'_, E> {
    #[inline]
    fn default() -> Self {
        from_slice_mut_generic::<E>(E::faer_map(E::UNIT, |()| &mut [] as &mut [E::Unit]))
    }
}

impl<'short, E: Entity, R: Shape> Reborrow<'short> for ColMut<'_, E, R> {
    type Target = ColRef<'short, E, R>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        ColRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'short, E: Entity, R: Shape> ReborrowMut<'short> for ColMut<'_, E, R> {
    type Target = ColMut<'short, E, R>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        ColMut {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'a, E: Entity, R: Shape> IntoConst for ColMut<'a, E, R> {
    type Target = ColRef<'a, E, R>;

    #[inline]
    fn into_const(self) -> Self::Target {
        ColRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'a, E: Entity, R: Shape> ColMut<'a, E, R> {
    #[inline]
    pub(crate) unsafe fn __from_raw_parts(ptr: PtrMut<E>, nrows: R, row_stride: isize) -> Self {
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

    /// Returns pointers to the matrix data.
    #[inline(always)]
    pub fn as_ptr(self) -> PtrConst<E> {
        self.into_const().as_ptr()
    }

    /// Returns pointers to the matrix data.
    #[inline(always)]
    pub fn as_ptr_mut(self) -> PtrMut<E> {
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
    pub fn as_2d(self) -> MatRef<'a, E, R> {
        self.into_const().as_2d()
    }

    /// Returns `self` as a mutable matrix view.
    #[inline(always)]
    pub fn as_2d_mut(self) -> MatMut<'a, E, R> {
        unsafe { self.into_const().as_2d().const_cast() }
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at(self, row: usize) -> PtrConst<E> {
        self.into_const().ptr_at(row)
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at_mut(self, row: usize) -> PtrMut<E> {
        let offset = (row as isize).wrapping_mul(self.inner.stride);

        E::faer_map(
            self.as_ptr_mut(),
            #[inline(always)]
            |ptr| ptr.wrapping_offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(self, row: usize) -> PtrConst<E> {
        self.into_const().ptr_at_unchecked(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_mut_unchecked(self, row: usize) -> PtrMut<E> {
        let offset = crate::utils::unchecked_mul(row, self.inner.stride);
        E::faer_map(
            self.as_ptr_mut(),
            #[inline(always)]
            |ptr| ptr.offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(self, row: IdxInc<R>) -> PtrConst<E> {
        self.into_const().overflowing_ptr_at(row)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at_mut(self, row: IdxInc<R>) -> PtrMut<E> {
        unsafe {
            let cond = row != self.nrows();
            let offset = (cond as usize).wrapping_neg() as isize
                & (row.unbound() as isize).wrapping_mul(self.inner.stride);
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
    pub unsafe fn ptr_inbounds_at(self, row: Idx<R>) -> PtrConst<E> {
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
    pub unsafe fn ptr_inbounds_at_mut(self, row: Idx<R>) -> PtrMut<E> {
        debug_assert!(row < self.nrows());
        self.ptr_at_mut_unchecked(row.unbound())
    }

    /// Returns a view over the column.
    #[inline]
    pub fn as_dyn(self) -> ColRef<'a, E> {
        let nrows = self.nrows().unbound();
        let row_stride = self.row_stride();
        unsafe { from_raw_parts(self.as_ptr(), nrows, row_stride) }
    }

    /// Returns a view over the column.
    #[inline]
    pub fn as_dyn_mut(self) -> ColMut<'a, E> {
        let nrows = self.nrows().unbound();
        let row_stride = self.row_stride();
        unsafe { from_raw_parts_mut(self.as_ptr_mut(), nrows, row_stride) }
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col(self) -> Slice<'a, E> {
        self.into_const().try_get_contiguous_col()
    }

    #[track_caller]
    #[inline(always)]
    #[doc(hidden)]
    pub fn try_get_contiguous_col_mut(self) -> SliceMut<'a, E> {
        assert!(self.row_stride() == 1);
        let m = self.nrows().unbound();
        E::faer_map(
            self.as_ptr_mut(),
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
        )
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
    pub unsafe fn split_at_unchecked(self, row: IdxInc<R>) -> (ColRef<'a, E>, ColRef<'a, E>) {
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
    pub unsafe fn split_at_mut_unchecked(self, row: IdxInc<R>) -> (ColMut<'a, E>, ColMut<'a, E>) {
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
    pub fn split_at(self, row: IdxInc<R>) -> (ColRef<'a, E>, ColRef<'a, E>) {
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
    pub fn split_at_mut(self, row: IdxInc<R>) -> (ColMut<'a, E>, ColMut<'a, E>) {
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
    ) -> <ColRef<'a, E, R> as ColIndex<RowRange>>::Target
    where
        ColRef<'a, E, R>: ColIndex<RowRange>,
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
    pub fn get<RowRange>(self, row: RowRange) -> <ColRef<'a, E, R> as ColIndex<RowRange>>::Target
    where
        ColRef<'a, E, R>: ColIndex<RowRange>,
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

    /// Returns references to the element at the given index, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be in `[0, self.nrows())`.
    #[inline(always)]
    #[track_caller]
    pub fn at(self, row: Idx<R>) -> Ref<'a, E> {
        self.into_const().at(row)
    }

    /// Returns references to the element at the given index, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row` must be in `[0, self.nrows())`.
    #[inline(always)]
    #[track_caller]
    pub fn at_mut(self, row: Idx<R>) -> Mut<'a, E> {
        assert!(row < self.nrows());
        unsafe {
            E::faer_map(
                self.ptr_inbounds_at_mut(row),
                #[inline(always)]
                |ptr| &mut *ptr,
            )
        }
    }

    /// Returns references to the element at the given index.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be in `[0, self.nrows())`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn at_unchecked(self, row: Idx<R>) -> Ref<'a, E> {
        self.into_const().at_unchecked(row)
    }

    /// Returns references to the element at the given index.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row` must be in `[0, self.nrows())`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn at_mut_unchecked(self, row: Idx<R>) -> Mut<'a, E> {
        unsafe {
            E::faer_map(
                self.ptr_inbounds_at_mut(row),
                #[inline(always)]
                |ptr| &mut *ptr,
            )
        }
    }

    /// Reads the value of the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, row: Idx<R>) -> E {
        self.rb().read_unchecked(row)
    }

    /// Reads the value of the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, row: Idx<R>) -> E {
        self.rb().read(row)
    }

    /// Writes the value to the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, row: Idx<R>, value: E) {
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
    pub fn write(&mut self, row: Idx<R>, value: E) {
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
    pub fn copy_from<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsColRef<ViewE, R = R>,
    ) {
        #[track_caller]
        #[inline(always)]
        fn implementation<R: Shape, E: Entity, ViewE: Conjugate<Canonical = E>>(
            this: ColMut<'_, E, R>,
            other: ColRef<'_, ViewE, R>,
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
    pub fn transpose(self) -> RowRef<'a, E, R> {
        self.into_const().transpose()
    }

    /// Returns a view over the transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn transpose_mut(self) -> RowMut<'a, E, R> {
        unsafe { self.into_const().transpose().const_cast() }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline(always)]
    #[must_use]
    pub fn conjugate(self) -> ColRef<'a, E::Conj, R>
    where
        E: Conjugate,
    {
        self.into_const().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline(always)]
    #[must_use]
    pub fn conjugate_mut(self) -> ColMut<'a, E::Conj, R>
    where
        E: Conjugate,
    {
        unsafe { self.into_const().conjugate().const_cast() }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    pub fn adjoint(self) -> RowRef<'a, E::Conj, R>
    where
        E: Conjugate,
    {
        self.into_const().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    pub fn adjoint_mut(self) -> RowMut<'a, E::Conj, R>
    where
        E: Conjugate,
    {
        self.conjugate_mut().transpose_mut()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize(self) -> (ColRef<'a, E::Canonical, R>, Conj)
    where
        E: Conjugate,
    {
        self.into_const().canonicalize()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize_mut(self) -> (ColMut<'a, E::Canonical, R>, Conj)
    where
        E: Conjugate,
    {
        let (canon, conj) = self.into_const().canonicalize();
        unsafe { (canon.const_cast(), conj) }
    }

    /// Returns a view over the `self`, with the rows in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows(self) -> ColRef<'a, E, R> {
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
    pub unsafe fn subrows_unchecked<V: Shape>(
        self,
        row_start: IdxInc<R>,
        nrows: V,
    ) -> ColRef<'a, E, V> {
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
    pub unsafe fn subrows_mut_unchecked<V: Shape>(
        self,
        row_start: IdxInc<R>,
        nrows: V,
    ) -> ColMut<'a, E, V> {
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
    pub fn subrows<V: Shape>(self, row_start: IdxInc<R>, nrows: V) -> ColRef<'a, E, V> {
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
    pub fn subrows_mut<V: Shape>(self, row_start: IdxInc<R>, nrows: V) -> ColMut<'a, E, V> {
        unsafe { self.into_const().subrows(row_start, nrows).const_cast() }
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

    /// Returns the column as a contiguous slice if its row stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice(self) -> Option<Slice<'a, E>> {
        self.into_const().try_as_slice()
    }

    /// Returns the column as a contiguous slice if its row stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice_mut(self) -> Option<SliceMut<'a, E>> {
        if self.row_stride() == 1 {
            let len = self.nrows().unbound();
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
    pub unsafe fn try_as_uninit_slice_mut(self) -> Option<UninitSliceMut<'a, E>> {
        if self.row_stride() == 1 {
            let len = self.nrows().unbound();
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
    pub fn as_ref(&self) -> ColRef<'_, E, R> {
        (*self).rb()
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> ColMut<'_, E, R> {
        (*self).rb_mut()
    }

    /// Returns a reference to the first element and a view over the remaining ones if the column is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_first(self) -> Option<(Ref<'a, E>, ColRef<'a, E>)> {
        self.into_const().split_first()
    }

    /// Returns a reference to the last element and a view over the remaining ones if the column is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_last(self) -> Option<(Ref<'a, E>, ColRef<'a, E>)> {
        self.into_const().split_last()
    }

    /// Returns a reference to the first element and a view over the remaining ones if the column is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_first_mut(self) -> Option<(Mut<'a, E>, ColMut<'a, E>)> {
        let this = self.as_dyn_mut();
        if this.nrows() == 0 {
            None
        } else {
            unsafe {
                let (head, tail) = { this.split_at_mut_unchecked(1) };
                Some((head.get_mut_unchecked(0), tail))
            }
        }
    }

    /// Returns a reference to the last element and a view over the remaining ones if the column is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_last_mut(self) -> Option<(Mut<'a, E>, ColMut<'a, E>)> {
        let this = self.as_dyn_mut();
        if this.nrows() == 0 {
            None
        } else {
            let nrows = this.nrows();
            unsafe {
                let (head, tail) = { this.split_at_mut_unchecked(nrows - 1) };
                Some((tail.get_mut_unchecked(0), head))
            }
        }
    }

    /// Returns an iterator over the elements of the column.
    #[inline]
    pub fn iter(self) -> iter::ElemIter<'a, E> {
        iter::ElemIter {
            inner: self.as_dyn(),
        }
    }

    /// Returns an iterator over the elements of the column.
    #[inline]
    pub fn iter_mut(self) -> iter::ElemIterMut<'a, E> {
        iter::ElemIterMut {
            inner: self.as_dyn_mut(),
        }
    }

    /// Returns an iterator that provides successive chunks of the elements of this column, with
    /// each having at most `chunk_size` elements.
    #[inline]
    #[track_caller]
    pub fn chunks(self, chunk_size: usize) -> iter::ColElemChunks<'a, E> {
        self.into_const().chunks(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// column.
    #[inline]
    #[track_caller]
    pub fn partition(self, count: usize) -> iter::ColElemPartition<'a, E> {
        self.into_const().partition(count)
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
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColRef<'a, E>> {
        self.into_const().par_chunks(chunk_size)
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
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColRef<'a, E>> {
        self.into_const().par_partition(count)
    }

    /// Returns an iterator that provides successive chunks of the elements of this column, with
    /// each having at most `chunk_size` elements.
    #[inline]
    #[track_caller]
    pub fn chunks_mut(self, chunk_size: usize) -> iter::ColElemChunksMut<'a, E> {
        assert!(chunk_size > 0);
        let nrows = self.nrows().unbound();
        iter::ColElemChunksMut {
            inner: self.as_dyn_mut(),
            policy: iter::chunks::ChunkSizePolicy::new(nrows, iter::chunks::ChunkSize(chunk_size)),
        }
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// column.
    #[inline]
    #[track_caller]
    pub fn partition_mut(self, count: usize) -> iter::ColElemPartitionMut<'a, E> {
        assert!(count > 0);
        let nrows = self.nrows();
        iter::ColElemPartitionMut {
            inner: self.as_dyn_mut(),
            policy: iter::chunks::PartitionCountPolicy::new(
                nrows.unbound(),
                iter::chunks::PartitionCount(count),
            ),
        }
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
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColMut<'a, E>> {
        use rayon::prelude::*;
        self.into_const()
            .par_chunks(chunk_size)
            .map(|x| unsafe { x.const_cast() })
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
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = ColMut<'a, E>> {
        use rayon::prelude::*;
        self.into_const()
            .par_partition(count)
            .map(|x| unsafe { x.const_cast() })
    }

    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn const_cast(self) -> ColMut<'a, E, R> {
        self
    }
}

/// Creates a `ColMut` from pointers to the column vector data, number of rows, and row stride.
///
/// # Safety:
/// This function has the same safety requirements as
/// [`mat::from_raw_parts_mut(ptr, nrows, 1, row_stride, 0)`]
#[inline(always)]
pub unsafe fn from_raw_parts_mut<'a, E: Entity, R: Shape>(
    ptr: PtrMut<E>,
    nrows: R,
    row_stride: isize,
) -> ColMut<'a, E, R> {
    ColMut::__from_raw_parts(ptr, nrows, row_stride)
}

/// Creates a `ColMut` from slice views over the column vector data, The result has the same
/// number of rows as the length of the input slice.
#[inline(always)]
pub fn from_slice_mut_generic<E: Entity>(slice: SliceMut<'_, E>) -> ColMut<'_, E> {
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

impl<E: Entity, R: Shape> AsColRef<E> for ColMut<'_, E, R> {
    type R = R;

    #[inline]
    fn as_col_ref(&self) -> ColRef<'_, E, R> {
        (*self).rb()
    }
}
impl<E: Entity, R: Shape> AsColMut<E> for ColMut<'_, E, R> {
    #[inline]
    fn as_col_mut(&mut self) -> ColMut<'_, E, R> {
        (*self).rb_mut()
    }
}

impl<'a, E: Entity, R: Shape> core::fmt::Debug for ColMut<'a, E, R> {
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
pub fn from_mut_generic<E: Entity>(value: Mut<'_, E>) -> ColMut<'_, E> {
    unsafe { from_raw_parts_mut(E::faer_map(value, |ptr| ptr as *mut E::Unit), 1, 1) }
}
