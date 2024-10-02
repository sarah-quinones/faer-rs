use super::*;
use crate::{
    assert,
    col::{ColMut, ColRef},
    debug_assert, iter,
    iter::chunks::ChunkPolicy,
    mat, unzipped, zipped, Idx, IdxInc, Shape, Unbind,
};

/// Mutable view over a row vector, similar to a mutable reference to a strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `RowMut<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`RowMut::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
///
/// # Move semantics
/// See [`faer::Mat`](crate::Mat) for information about reborrowing when using this type.
#[repr(C)]
pub struct RowMut<'a, E: Entity, C: Shape = usize> {
    pub(super) inner: VecImpl<E, C>,
    pub(super) __marker: PhantomData<&'a E>,
}

impl<E: Entity> Default for RowMut<'_, E> {
    #[inline]
    fn default() -> Self {
        from_slice_mut_generic::<E>(E::faer_map(E::UNIT, |()| &mut [] as &mut [E::Unit]))
    }
}

impl<'short, E: Entity, C: Shape> Reborrow<'short> for RowMut<'_, E, C> {
    type Target = RowRef<'short, E, C>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        RowRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'short, E: Entity, C: Shape> ReborrowMut<'short> for RowMut<'_, E, C> {
    type Target = RowMut<'short, E, C>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        RowMut {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'a, E: Entity, C: Shape> IntoConst for RowMut<'a, E, C> {
    type Target = RowRef<'a, E, C>;

    #[inline]
    fn into_const(self) -> Self::Target {
        RowRef {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

impl<'a, E: Entity, C: Shape> RowMut<'a, E, C> {
    #[inline]
    pub(crate) unsafe fn __from_raw_parts(ptr: PtrMut<E>, ncols: C, col_stride: isize) -> Self {
        Self {
            inner: VecImpl {
                ptr: into_copy::<E, _>(E::faer_map(
                    ptr,
                    #[inline]
                    |ptr| NonNull::new_unchecked(ptr),
                )),
                len: ncols,
                stride: col_stride,
            },
            __marker: PhantomData,
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
        self.inner.len
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

    /// Returns the column stride of the matrix, specified in number of elements, not in bytes.
    #[inline(always)]
    pub fn col_stride(&self) -> isize {
        self.inner.stride
    }

    /// Returns `self` as a matrix view.
    #[inline(always)]
    pub fn as_2d(self) -> MatRef<'a, E, usize, C> {
        self.into_const().as_2d()
    }

    /// Returns `self` as a mutable matrix view.
    #[inline(always)]
    pub fn as_2d_mut(self) -> MatMut<'a, E, usize, C> {
        let ncols = self.ncols();
        let col_stride = self.col_stride();
        unsafe { mat::from_raw_parts_mut(self.as_ptr_mut(), 1, ncols, isize::MAX, col_stride) }
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at(self, col: usize) -> PtrConst<E> {
        self.into_const().ptr_at(col)
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at_mut(self, col: usize) -> PtrMut<E> {
        let offset = (col as isize).wrapping_mul(self.inner.stride);

        E::faer_map(
            self.as_ptr_mut(),
            #[inline(always)]
            |ptr| ptr.wrapping_offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(self, col: usize) -> PtrConst<E> {
        self.into_const().ptr_at_unchecked(col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_mut_unchecked(self, col: usize) -> PtrMut<E> {
        let offset = crate::utils::unchecked_mul(col, self.inner.stride);
        E::faer_map(
            self.as_ptr_mut(),
            #[inline(always)]
            |ptr| ptr.offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(self, col: IdxInc<C>) -> PtrConst<E> {
        self.into_const().overflowing_ptr_at(col)
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at_mut(self, col: IdxInc<C>) -> PtrMut<E> {
        unsafe {
            let cond = col != self.ncols();
            let offset = (cond as usize).wrapping_neg() as isize
                & (col.unbound() as isize).wrapping_mul(self.inner.stride);
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
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(self, col: Idx<C>) -> PtrConst<E> {
        self.into_const().ptr_inbounds_at(col)
    }

    /// Returns raw pointers to the element at the given index, assuming the provided index
    /// is within the size of the vector.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at_mut(self, col: Idx<C>) -> PtrMut<E> {
        debug_assert!(col < self.ncols());
        self.ptr_at_mut_unchecked(col.unbound())
    }

    /// Returns the input row with dynamic shape.
    #[inline]
    pub fn as_dyn(self) -> RowRef<'a, E> {
        let ncols = self.ncols().unbound();
        let col_stride = self.col_stride();
        unsafe { from_raw_parts(self.as_ptr(), ncols, col_stride) }
    }

    /// Returns the input row with dynamic shape.
    #[inline]
    pub fn as_dyn_mut(self) -> RowMut<'a, E> {
        let ncols = self.ncols().unbound();
        let col_stride = self.col_stride();
        unsafe { from_raw_parts_mut(self.as_ptr_mut(), ncols, col_stride) }
    }

    /// Returns the input row with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape<H: Shape>(self, ncols: H) -> RowRef<'a, E, H> {
        self.into_const().as_shape(ncols)
    }

    /// Returns the input row with the given shape after checking that it matches the
    /// current shape.
    #[inline]
    pub fn as_shape_mut<H: Shape>(self, ncols: H) -> RowMut<'a, E, H> {
        unsafe { self.into_const().as_shape(ncols).const_cast() }
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
    pub unsafe fn split_at_unchecked(self, col: IdxInc<C>) -> (RowRef<'a, E>, RowRef<'a, E>) {
        self.into_const().split_at_unchecked(col)
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
    pub unsafe fn split_at_mut_unchecked(self, col: IdxInc<C>) -> (RowMut<'a, E>, RowMut<'a, E>) {
        let (left, right) = self.into_const().split_at_unchecked(col);
        unsafe { (left.const_cast(), right.const_cast()) }
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
    pub fn split_at(self, col: IdxInc<C>) -> (RowRef<'a, E>, RowRef<'a, E>) {
        self.into_const().split_at(col)
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
    pub fn split_at_mut(self, col: IdxInc<C>) -> (RowMut<'a, E>, RowMut<'a, E>) {
        assert!(col <= self.ncols());
        unsafe { self.split_at_mut_unchecked(col) }
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
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn get_unchecked<ColRange>(
        self,
        col: ColRange,
    ) -> <RowRef<'a, E, C> as RowIndex<ColRange>>::Target
    where
        RowRef<'a, E, C>: RowIndex<ColRange>,
    {
        self.into_const().get_unchecked(col)
    }

    /// Returns references to the element at the given index, or subvector if `col` is a
    /// range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub fn get<ColRange>(self, col: ColRange) -> <RowRef<'a, E, C> as RowIndex<ColRange>>::Target
    where
        RowRef<'a, E, C>: RowIndex<ColRange>,
    {
        self.into_const().get(col)
    }

    /// Returns references to the element at the given index, or subvector if `col` is a
    /// range.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn get_mut_unchecked<ColRange>(
        self,
        col: ColRange,
    ) -> <Self as RowIndex<ColRange>>::Target
    where
        Self: RowIndex<ColRange>,
    {
        <Self as RowIndex<ColRange>>::get_unchecked(self, col)
    }

    /// Returns references to the element at the given index, or subvector if `col` is a
    /// range, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col` must be contained in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub fn get_mut<ColRange>(self, col: ColRange) -> <Self as RowIndex<ColRange>>::Target
    where
        Self: RowIndex<ColRange>,
    {
        <Self as RowIndex<ColRange>>::get(self, col)
    }

    /// Returns references to the element at the given index, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col` must be in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub fn at(self, col: Idx<C>) -> Ref<'a, E> {
        self.into_const().at(col)
    }

    /// Returns references to the element at the given index, with bound checks.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col` must be in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub fn at_mut(self, col: Idx<C>) -> Mut<'a, E> {
        assert!(col < self.ncols());
        unsafe {
            E::faer_map(
                self.ptr_inbounds_at_mut(col),
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
    /// * `col` must be in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn at_unchecked(self, col: Idx<C>) -> Ref<'a, E> {
        self.into_const().at_unchecked(col)
    }

    /// Returns references to the element at the given index.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col` must be in `[0, self.ncols())`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn at_mut_unchecked(self, col: Idx<C>) -> Mut<'a, E> {
        unsafe {
            E::faer_map(
                self.ptr_inbounds_at_mut(col),
                #[inline(always)]
                |ptr| &mut *ptr,
            )
        }
    }

    /// Reads the value of the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, col: Idx<C>) -> E {
        self.rb().read_unchecked(col)
    }

    /// Reads the value of the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, col: Idx<C>) -> E {
        self.rb().read(col)
    }

    /// Writes the value to the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, col: Idx<C>, value: E) {
        let units = value.faer_into_units();
        let zipped = E::faer_zip(units, (*self).rb_mut().ptr_inbounds_at_mut(col));
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
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, col: Idx<C>, value: E) {
        assert!(col < self.ncols());
        unsafe { self.write_unchecked(col, value) };
    }

    /// Copies the values from `other` into `self`.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `self.ncols() == other.ncols()`.
    #[track_caller]
    pub fn copy_from<ViewE: Conjugate<Canonical = E>>(
        &mut self,
        other: impl AsRowRef<ViewE, C = C>,
    ) {
        #[track_caller]
        #[inline(always)]
        fn implementation<C: Shape, E: Entity, ViewE: Conjugate<Canonical = E>>(
            this: RowMut<'_, E, C>,
            other: RowRef<'_, ViewE, C>,
        ) {
            zipped!(__rw, this, other)
                .for_each(|unzipped!(mut dst, src)| dst.write(src.read().canonicalize()));
        }
        implementation(self.rb_mut(), other.as_row_ref())
    }

    /// Fills the elements of `self` with zeros.
    #[track_caller]
    pub fn fill_zero(&mut self)
    where
        E: ComplexField,
    {
        zipped!(__rw, self.rb_mut().as_2d_mut()).for_each(
            #[inline(always)]
            |unzipped!(mut x)| x.write(E::faer_zero()),
        );
    }

    /// Fills the elements of `self` with copies of `constant`.
    #[track_caller]
    pub fn fill(&mut self, constant: E) {
        zipped!(__rw, (*self).rb_mut().as_2d_mut()).for_each(
            #[inline(always)]
            |unzipped!(mut x)| x.write(constant),
        );
    }

    /// Returns a view over the transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn transpose(self) -> ColRef<'a, E, C> {
        self.into_const().transpose()
    }

    /// Returns a view over the transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn transpose_mut(self) -> ColMut<'a, E, C> {
        unsafe { self.into_const().transpose().const_cast() }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline(always)]
    #[must_use]
    pub fn conjugate(self) -> RowRef<'a, E::Conj, C>
    where
        E: Conjugate,
    {
        self.into_const().conjugate()
    }

    /// Returns a view over the conjugate of `self`.
    #[inline(always)]
    #[must_use]
    pub fn conjugate_mut(self) -> RowMut<'a, E::Conj, C>
    where
        E: Conjugate,
    {
        unsafe { self.into_const().conjugate().const_cast() }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    pub fn adjoint(self) -> ColRef<'a, E::Conj, C>
    where
        E: Conjugate,
    {
        self.into_const().adjoint()
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    pub fn adjoint_mut(self) -> ColMut<'a, E::Conj, C>
    where
        E: Conjugate,
    {
        self.conjugate_mut().transpose_mut()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize(self) -> (RowRef<'a, E::Canonical, C>, Conj)
    where
        E: Conjugate,
    {
        self.into_const().canonicalize()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize_mut(self) -> (RowMut<'a, E::Canonical, C>, Conj)
    where
        E: Conjugate,
    {
        let (canon, conj) = self.into_const().canonicalize();
        unsafe { (canon.const_cast(), conj) }
    }

    /// Returns a view over the `self`, with the columns in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols(self) -> RowRef<'a, E, C> {
        self.into_const().reverse_cols()
    }

    /// Returns a view over the `self`, with the columns in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols_mut(self) -> Self {
        unsafe { self.into_const().reverse_cols().const_cast() }
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
        self,
        col_start: IdxInc<C>,
        ncols: H,
    ) -> RowRef<'a, E, H> {
        self.into_const().subcols_unchecked(col_start, ncols)
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
    pub fn subcols<H: Shape>(self, col_start: IdxInc<C>, ncols: H) -> RowRef<'a, E, H> {
        self.into_const().subcols(col_start, ncols)
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
        self,
        col_start: IdxInc<C>,
        ncols: H,
    ) -> RowMut<'a, E, H> {
        self.into_const()
            .subcols_unchecked(col_start, ncols)
            .const_cast()
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
    pub fn subcols_mut<H: Shape>(self, col_start: IdxInc<C>, ncols: H) -> RowMut<'a, E, H> {
        unsafe { self.into_const().subcols(col_start, ncols).const_cast() }
    }

    /// Returns an owning [`Row`] of the data.
    #[inline]
    pub fn to_owned(&self) -> Row<E::Canonical, C>
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
        self.rb().as_2d().kron(rhs)
    }

    /// Returns the row as a contiguous slice if its column stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice(self) -> Option<Slice<'a, E>> {
        self.into_const().try_as_slice()
    }

    /// Returns the row as a contiguous slice if its column stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice_mut(self) -> Option<SliceMut<'a, E>> {
        if self.col_stride() == 1 {
            let len = self.ncols().unbound();
            Some(E::faer_map(
                self.as_ptr_mut(),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, len) },
            ))
        } else {
            None
        }
    }

    /// Returns the row as a contiguous potentially uninitialized slice if its column stride is
    /// equal to `1`.
    ///
    /// # Safety
    /// If uninit data is written to the slice, it must not be read at some later point.
    pub unsafe fn try_as_uninit_slice_mut(self) -> Option<UninitSliceMut<'a, E>> {
        if self.col_stride() == 1 {
            let len = self.ncols().unbound();
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
    pub fn as_ref(&self) -> RowRef<'_, E, C> {
        (*self).rb()
    }

    /// Returns a mutable view over the matrix.
    #[inline]
    pub fn as_mut(&mut self) -> RowMut<'_, E, C> {
        (*self).rb_mut()
    }

    /// Returns a reference to the first element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_first(self) -> Option<(Ref<'a, E>, RowRef<'a, E>)> {
        self.into_const().split_first()
    }

    /// Returns a reference to the last element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_last(self) -> Option<(Ref<'a, E>, RowRef<'a, E>)> {
        self.into_const().split_last()
    }

    /// Returns a reference to the first element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_first_mut(self) -> Option<(Mut<'a, E>, RowMut<'a, E>)> {
        let this = self.as_dyn_mut();
        if this.ncols() == 0 {
            None
        } else {
            unsafe {
                let (head, tail) = { this.split_at_mut_unchecked(1) };
                Some((head.get_mut_unchecked(0), tail))
            }
        }
    }

    /// Returns a reference to the last element and a view over the remaining ones if the row is
    /// non-empty, otherwise `None`.
    #[inline]
    pub fn split_last_mut(self) -> Option<(Mut<'a, E>, RowMut<'a, E>)> {
        let this = self.as_dyn_mut();
        if this.ncols() == 0 {
            None
        } else {
            let ncols = this.ncols();
            unsafe {
                let (head, tail) = { this.split_at_mut_unchecked(ncols - 1) };
                Some((tail.get_mut_unchecked(0), head))
            }
        }
    }

    /// Returns an iterator over the elements of the row.
    #[inline]
    pub fn iter(self) -> iter::ElemIter<'a, E> {
        self.into_const().iter()
    }

    /// Returns an iterator over the elements of the row.
    #[inline]
    pub fn iter_mut(self) -> iter::ElemIterMut<'a, E> {
        iter::ElemIterMut {
            inner: self.transpose_mut().as_dyn_mut(),
        }
    }

    /// Returns an iterator that provides successive chunks of the elements of this row, with
    /// each having at most `chunk_size` elements.
    #[inline]
    #[track_caller]
    pub fn chunks(self, chunk_size: usize) -> iter::RowElemChunks<'a, E> {
        self.into_const().chunks(chunk_size)
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// row.
    #[inline]
    #[track_caller]
    pub fn partition(self, count: usize) -> iter::RowElemPartition<'a, E> {
        self.into_const().partition(count)
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
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowRef<'a, E>> {
        self.into_const().par_chunks(chunk_size)
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
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowRef<'a, E>> {
        self.into_const().par_partition(count)
    }

    /// Returns an iterator that provides successive chunks of the elements of this row, with
    /// each having at most `chunk_size` elements.
    #[inline]
    #[track_caller]
    pub fn chunks_mut(self, chunk_size: usize) -> iter::RowElemChunksMut<'a, E> {
        assert!(chunk_size > 0);
        let ncols = self.ncols().unbound();
        iter::RowElemChunksMut {
            inner: self.as_dyn_mut(),
            policy: iter::chunks::ChunkSizePolicy::new(ncols, iter::chunks::ChunkSize(chunk_size)),
        }
    }

    /// Returns an iterator that provides exactly `count` successive chunks of the elements of this
    /// row.
    #[inline]
    #[track_caller]
    pub fn partition_mut(self, count: usize) -> iter::RowElemPartitionMut<'a, E> {
        assert!(count > 0);
        let ncols = self.ncols().unbound();
        iter::RowElemPartitionMut {
            inner: self.as_dyn_mut(),
            policy: iter::chunks::PartitionCountPolicy::new(
                ncols,
                iter::chunks::PartitionCount(count),
            ),
        }
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
        self,
        chunk_size: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowMut<'a, E>> {
        use rayon::prelude::*;
        self.into_const()
            .par_chunks(chunk_size)
            .map(|x| unsafe { x.const_cast() })
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
        self,
        count: usize,
    ) -> impl 'a + rayon::iter::IndexedParallelIterator<Item = RowMut<'a, E>> {
        use rayon::prelude::*;
        self.into_const()
            .par_partition(count)
            .map(|x| unsafe { x.const_cast() })
    }

    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn const_cast(self) -> RowMut<'a, E, C> {
        self
    }
}

/// Creates a `RowMut` from pointers to the row vector data, number of columns, and column
/// stride.
///
/// # Safety:
/// This function has the same safety requirements as
/// [`mat::from_raw_parts_mut(ptr, 1, ncols, 0, col_stride)`]
#[inline(always)]
pub unsafe fn from_raw_parts_mut<'a, E: Entity, C: Shape>(
    ptr: PtrMut<E>,
    ncols: C,
    col_stride: isize,
) -> RowMut<'a, E, C> {
    RowMut::__from_raw_parts(ptr, ncols, col_stride)
}

/// Creates a `RowMut` from slice views over the row vector data, The result has the same
/// number of columns as the length of the input slice.
#[inline(always)]
pub fn from_slice_mut_generic<E: Entity>(slice: SliceMut<'_, E>) -> RowMut<'_, E> {
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

/// Creates a `RowMut` from slice views over the row vector data, The result has the same
/// number of columns as the length of the input slice.
#[inline(always)]
pub fn from_slice_mut<E: SimpleEntity>(slice: &mut [E]) -> RowMut<'_, E> {
    from_slice_mut_generic(slice)
}

impl<E: Entity, C: Shape> As2D<E> for RowMut<'_, E, C> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (*self).rb().as_2d().as_dyn()
    }
}

impl<E: Entity, C: Shape> As2DMut<E> for RowMut<'_, E, C> {
    #[inline]
    fn as_2d_mut(&mut self) -> MatMut<'_, E> {
        (*self).rb_mut().as_2d_mut().as_dyn_mut()
    }
}

impl<'a, E: Entity, C: Shape> core::fmt::Debug for RowMut<'a, E, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<E: SimpleEntity, C: Shape> core::ops::Index<Idx<C>> for RowMut<'_, E, C> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, col: Idx<C>) -> &E {
        (*self).rb().at(col)
    }
}

impl<E: SimpleEntity, C: Shape> core::ops::IndexMut<Idx<C>> for RowMut<'_, E, C> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, col: Idx<C>) -> &mut E {
        (*self).rb_mut().at_mut(col)
    }
}

impl<E: Entity, C: Shape> AsRowRef<E> for RowMut<'_, E, C> {
    type C = C;

    #[inline]
    fn as_row_ref(&self) -> RowRef<'_, E, C> {
        (*self).rb()
    }
}

impl<E: Entity, C: Shape> AsRowMut<E> for RowMut<'_, E, C> {
    #[inline]
    fn as_row_mut(&mut self) -> RowMut<'_, E, C> {
        (*self).rb_mut()
    }
}

impl<E: Conjugate> RowBatch<E> for RowMut<'_, E> {
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
        Self::Owned::resize_owned(owned, nrows, ncols)
    }
}

impl<E: Conjugate> RowBatchMut<E> for RowMut<'_, E> {}

/// Returns a view over a row with 1 column containing value as its only element, pointing to
/// `value`.
pub fn from_mut_generic<E: Entity>(value: Mut<'_, E>) -> RowMut<'_, E> {
    unsafe { from_raw_parts_mut(E::faer_map(value, |ptr| ptr as *mut E::Unit), 1, 1) }
}

/// Returns a view over a column with 1 row containing value as its only element, pointing to
/// `value`.
pub fn from_mut<E: SimpleEntity>(value: &mut E) -> RowMut<'_, E> {
    from_mut_generic(value)
}
