use super::*;
use crate::{assert, debug_assert, diag::DiagRef, row::RowRef};

/// Immutable view over a column vector, similar to an immutable reference to a strided
/// [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `ColRef<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`ColRef::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
#[repr(C)]
pub struct ColRef<'a, E: Entity> {
    pub(super) inner: VecImpl<E>,
    pub(super) __marker: PhantomData<&'a E>,
}

impl<E: Entity> Clone for ColRef<'_, E> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Entity> Copy for ColRef<'_, E> {}

impl<'short, E: Entity> Reborrow<'short> for ColRef<'_, E> {
    type Target = ColRef<'short, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, E: Entity> ReborrowMut<'short> for ColRef<'_, E> {
    type Target = ColRef<'short, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<E: Entity> IntoConst for ColRef<'_, E> {
    type Target = Self;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'a, E: Entity> ColRef<'a, E> {
    #[inline]
    pub(crate) unsafe fn __from_raw_parts(
        ptr: GroupFor<E, *const E::Unit>,
        nrows: usize,
        row_stride: isize,
    ) -> Self {
        Self {
            inner: VecImpl {
                ptr: into_copy::<E, _>(E::faer_map(
                    ptr,
                    #[inline]
                    |ptr| NonNull::new_unchecked(ptr as *mut E::Unit),
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
        assert!(self.row_stride() == 1);
        let m = self.nrows();
        E::faer_map(
            self.as_ptr(),
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
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
        E::faer_map(
            from_copy::<E, _>(self.inner.ptr),
            #[inline(always)]
            |ptr| ptr.as_ptr() as *const E::Unit,
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
        let nrows = self.nrows();
        let row_stride = self.row_stride();
        unsafe { crate::mat::from_raw_parts(self.as_ptr(), nrows, 1, row_stride, isize::MAX) }
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at(self, row: usize) -> GroupFor<E, *const E::Unit> {
        let offset = (row as isize).wrapping_mul(self.inner.stride);

        E::faer_map(
            self.as_ptr(),
            #[inline(always)]
            |ptr| ptr.wrapping_offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(self, row: usize) -> GroupFor<E, *const E::Unit> {
        let offset = crate::utils::unchecked_mul(row, self.inner.stride);
        E::faer_map(
            self.as_ptr(),
            #[inline(always)]
            |ptr| ptr.offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(self, row: usize) -> GroupFor<E, *const E::Unit> {
        unsafe {
            let cond = row != self.nrows();
            let offset = (cond as usize).wrapping_neg() as isize
                & (row as isize).wrapping_mul(self.inner.stride);
            E::faer_map(
                self.as_ptr(),
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
        debug_assert!(row < self.nrows());
        self.ptr_at_unchecked(row)
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
    pub unsafe fn split_at_unchecked(self, row: usize) -> (Self, Self) {
        debug_assert!(row <= self.nrows());

        let row_stride = self.row_stride();

        let nrows = self.nrows();

        unsafe {
            let top = self.as_ptr();
            let bot = self.overflowing_ptr_at(row);

            (
                Self::__from_raw_parts(top, row, row_stride),
                Self::__from_raw_parts(bot, nrows - row, row_stride),
            )
        }
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
    pub fn split_at(self, row: usize) -> (Self, Self) {
        assert!(row <= self.nrows());
        unsafe { self.split_at_unchecked(row) }
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
    pub fn get<RowRange>(self, row: RowRange) -> <Self as ColIndex<RowRange>>::Target
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
        E::faer_from_units(E::faer_map(
            self.get_unchecked(row),
            #[inline(always)]
            |ptr| *ptr,
        ))
    }

    /// Reads the value of the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `row < self.nrows()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, row: usize) -> E {
        E::faer_from_units(E::faer_map(
            self.get(row),
            #[inline(always)]
            |ptr| *ptr,
        ))
    }

    /// Returns a view over the transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn transpose(self) -> RowRef<'a, E> {
        unsafe { crate::row::from_raw_parts(self.as_ptr(), self.nrows(), self.row_stride()) }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline(always)]
    #[must_use]
    pub fn conjugate(self) -> ColRef<'a, E::Conj>
    where
        E: Conjugate,
    {
        unsafe {
            // SAFETY: Conjugate requires that E::Unit and E::Conj::Unit have the same layout
            // and that GroupCopyFor<E,X> == E::Conj::GroupCopy<X>
            super::from_raw_parts::<'_, E::Conj>(
                transmute_unchecked::<
                    GroupFor<E, *const UnitFor<E>>,
                    GroupFor<E::Conj, *const UnitFor<E::Conj>>,
                >(self.as_ptr()),
                self.nrows(),
                self.row_stride(),
            )
        }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    pub fn adjoint(self) -> RowRef<'a, E::Conj>
    where
        E: Conjugate,
    {
        self.conjugate().transpose()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize(self) -> (ColRef<'a, E::Canonical>, Conj)
    where
        E: Conjugate,
    {
        (
            unsafe {
                // SAFETY: see Self::conjugate
                super::from_raw_parts::<'_, E::Canonical>(
                    transmute_unchecked::<
                        GroupFor<E, *const E::Unit>,
                        GroupFor<E::Canonical, *const UnitFor<E::Canonical>>,
                    >(self.as_ptr()),
                    self.nrows(),
                    self.row_stride(),
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
    #[inline(always)]
    #[must_use]
    pub fn reverse_rows(self) -> Self {
        let nrows = self.nrows();
        let row_stride = self.row_stride().wrapping_neg();

        let ptr = unsafe { self.ptr_at_unchecked(nrows.saturating_sub(1)) };
        unsafe { Self::__from_raw_parts(ptr, nrows, row_stride) }
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
    pub unsafe fn subrows_unchecked(self, row_start: usize, nrows: usize) -> Self {
        debug_assert!(all(
            row_start <= self.nrows(),
            nrows <= self.nrows() - row_start
        ));
        let row_stride = self.row_stride();
        unsafe { Self::__from_raw_parts(self.overflowing_ptr_at(row_start), nrows, row_stride) }
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
    pub fn subrows(self, row_start: usize, nrows: usize) -> Self {
        assert!(all(
            row_start <= self.nrows(),
            nrows <= self.nrows() - row_start
        ));
        unsafe { self.subrows_unchecked(row_start, nrows) }
    }

    /// Given a matrix with a single column, returns an object that interprets
    /// the column as a diagonal matrix, whose diagonal elements are values in the column.
    #[track_caller]
    #[inline(always)]
    pub fn column_vector_as_diagonal(self) -> DiagRef<'a, E> {
        DiagRef { inner: self }
    }

    /// Returns an owning [`Col`] of the data.
    #[inline]
    pub fn to_owned(&self) -> Col<E::Canonical>
    where
        E: Conjugate,
    {
        let mut mat = Col::new();
        mat.resize_with(
            self.nrows(),
            #[inline(always)]
            |row| unsafe { self.read_unchecked(row).canonicalize() },
        );
        mat
    }

    /// Returns `true` if any of the elements is NaN, otherwise returns `false`.
    #[inline]
    pub fn has_nan(&self) -> bool
    where
        E: ComplexField,
    {
        (*self).as_2d().has_nan()
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
        self.as_2d().norm_max()
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
        self.as_2d().sum()
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

    /// Returns the column as a contiguous slice if its row stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice(self) -> Option<GroupFor<E, &'a [E::Unit]>> {
        if self.row_stride() == 1 {
            let len = self.nrows();
            Some(E::faer_map(
                self.as_ptr(),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts(ptr, len) },
            ))
        } else {
            None
        }
    }

    /// Returns a view over the matrix.
    #[inline]
    pub fn as_ref(&self) -> ColRef<'_, E> {
        *self
    }

    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn const_cast(self) -> ColMut<'a, E> {
        ColMut {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

/// Creates a `ColRef` from pointers to the column vector data, number of rows, and row stride.
///
/// # Safety:
/// This function has the same safety requirements as
/// [`mat::from_raw_parts(ptr, nrows, 1, row_stride, 0)`]
#[inline(always)]
pub unsafe fn from_raw_parts<'a, E: Entity>(
    ptr: GroupFor<E, *const E::Unit>,
    nrows: usize,
    row_stride: isize,
) -> ColRef<'a, E> {
    ColRef::__from_raw_parts(ptr, nrows, row_stride)
}

/// Creates a `ColRef` from slice views over the column vector data, The result has the same
/// number of rows as the length of the input slice.
#[inline(always)]
pub fn from_slice_generic<E: Entity>(slice: GroupFor<E, &[E::Unit]>) -> ColRef<'_, E> {
    let nrows = SliceGroup::<'_, E>::new(E::faer_copy(&slice)).len();

    unsafe {
        from_raw_parts(
            E::faer_map(
                slice,
                #[inline(always)]
                |slice| slice.as_ptr(),
            ),
            nrows,
            1,
        )
    }
}

/// Creates a `ColRef` from slice views over the column vector data, The result has the same
/// number of rows as the length of the input slice.
#[inline(always)]
pub fn from_slice<E: SimpleEntity>(slice: &[E]) -> ColRef<'_, E> {
    from_slice_generic(slice)
}

impl<E: Entity> As2D<E> for ColRef<'_, E> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (*self).as_2d()
    }
}

impl<E: Entity> AsColRef<E> for ColRef<'_, E> {
    #[inline]
    fn as_col_ref(&self) -> ColRef<'_, E> {
        *self
    }
}

impl<'a, E: Entity> core::fmt::Debug for ColRef<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_2d().fmt(f)
    }
}

impl<E: SimpleEntity> core::ops::Index<usize> for ColRef<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, row: usize) -> &E {
        self.get(row)
    }
}

impl<E: Conjugate> ColBatch<E> for ColRef<'_, E> {
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
        <Self::Owned as ColBatch<E::Canonical>>::resize_owned(owned, nrows, ncols)
    }
}

/// Returns a view over a column with `nrows` rows containing `value` repeated for all elements.
#[doc(alias = "broadcast")]
pub fn from_repeated_ref_generic<E: Entity>(
    value: GroupFor<E, &E::Unit>,
    nrows: usize,
) -> ColRef<'_, E> {
    unsafe { from_raw_parts(E::faer_map(value, |ptr| ptr as *const E::Unit), nrows, 0) }
}

/// Returns a view over a column with 1 row containing value as its only element, pointing to
/// `value`.
pub fn from_ref_generic<E: Entity>(value: GroupFor<E, &E::Unit>) -> ColRef<'_, E> {
    from_repeated_ref_generic(value, 1)
}

/// Returns a view over a column with `nrows` rows containing `value` repeated for all elements.
#[doc(alias = "broadcast")]
pub fn from_repeated_ref<E: SimpleEntity>(value: &E, nrows: usize) -> ColRef<'_, E> {
    from_repeated_ref_generic(value, nrows)
}

/// Returns a view over a column with 1 row containing value as its only element, pointing to
/// `value`.
pub fn from_ref<E: SimpleEntity>(value: &E) -> ColRef<'_, E> {
    from_ref_generic(value)
}
