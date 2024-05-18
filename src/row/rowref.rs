use super::*;
use crate::{assert, col::ColRef, debug_assert};

/// Immutable view over a row vector, similar to an immutable reference to a strided [prim@slice].
///
/// # Note
///
/// Unlike a slice, the data pointed to by `RowRef<'_, E>` is allowed to be partially or fully
/// uninitialized under certain conditions. In this case, care must be taken to not perform any
/// operations that read the uninitialized values, or form references to them, either directly
/// through [`RowRef::read`], or indirectly through any of the numerical library routines, unless
/// it is explicitly permitted.
#[repr(C)]
pub struct RowRef<'a, E: Entity> {
    pub(super) inner: VecImpl<E>,
    pub(super) __marker: PhantomData<&'a E>,
}

impl<E: Entity> Clone for RowRef<'_, E> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Entity> Copy for RowRef<'_, E> {}

impl<'short, E: Entity> Reborrow<'short> for RowRef<'_, E> {
    type Target = RowRef<'short, E>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, E: Entity> ReborrowMut<'short> for RowRef<'_, E> {
    type Target = RowRef<'short, E>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<E: Entity> IntoConst for RowRef<'_, E> {
    type Target = Self;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'a, E: Entity> RowRef<'a, E> {
    pub(crate) unsafe fn __from_raw_parts(
        ptr: GroupFor<E, *const E::Unit>,
        ncols: usize,
        col_stride: isize,
    ) -> Self {
        Self {
            inner: VecImpl {
                ptr: into_copy::<E, _>(E::faer_map(
                    ptr,
                    #[inline]
                    |ptr| NonNull::new_unchecked(ptr as *mut E::Unit),
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
    pub fn ncols(&self) -> usize {
        self.inner.len
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

    /// Returns the column stride of the matrix, specified in number of elements, not in bytes.
    #[inline(always)]
    pub fn col_stride(&self) -> isize {
        self.inner.stride
    }

    /// Returns `self` as a matrix view.
    #[inline(always)]
    pub fn as_2d(self) -> MatRef<'a, E> {
        let ncols = self.ncols();
        let col_stride = self.col_stride();
        unsafe { crate::mat::from_raw_parts(self.as_ptr(), 1, ncols, isize::MAX, col_stride) }
    }

    /// Returns raw pointers to the element at the given index.
    #[inline(always)]
    pub fn ptr_at(self, col: usize) -> GroupFor<E, *const E::Unit> {
        let offset = (col as isize).wrapping_mul(self.inner.stride);

        E::faer_map(
            self.as_ptr(),
            #[inline(always)]
            |ptr| ptr.wrapping_offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn ptr_at_unchecked(self, col: usize) -> GroupFor<E, *const E::Unit> {
        let offset = crate::utils::unchecked_mul(col, self.inner.stride);
        E::faer_map(
            self.as_ptr(),
            #[inline(always)]
            |ptr| ptr.offset(offset),
        )
    }

    #[inline(always)]
    #[doc(hidden)]
    pub unsafe fn overflowing_ptr_at(self, col: usize) -> GroupFor<E, *const E::Unit> {
        unsafe {
            let cond = col != self.ncols();
            let offset = (cond as usize).wrapping_neg() as isize
                & (col as isize).wrapping_mul(self.inner.stride);
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
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn ptr_inbounds_at(self, col: usize) -> GroupFor<E, *const E::Unit> {
        debug_assert!(col < self.ncols());
        self.ptr_at_unchecked(col)
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
    pub unsafe fn split_at_unchecked(self, col: usize) -> (Self, Self) {
        debug_assert!(col <= self.ncols());

        let col_stride = self.col_stride();

        let ncols = self.ncols();

        unsafe {
            let top = self.as_ptr();
            let bot = self.overflowing_ptr_at(col);

            (
                Self::__from_raw_parts(top, col, col_stride),
                Self::__from_raw_parts(bot, ncols - col, col_stride),
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
    /// * `col <= self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn split_at(self, col: usize) -> (Self, Self) {
        assert!(col <= self.ncols());
        unsafe { self.split_at_unchecked(col) }
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
    pub fn get<ColRange>(self, col: ColRange) -> <Self as RowIndex<ColRange>>::Target
    where
        Self: RowIndex<ColRange>,
    {
        <Self as RowIndex<ColRange>>::get(self, col)
    }

    /// Reads the value of the element at the given index.
    ///
    /// # Safety
    /// The behavior is undefined if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, col: usize) -> E {
        E::faer_from_units(E::faer_map(
            self.get_unchecked(col),
            #[inline(always)]
            |ptr| *ptr,
        ))
    }

    /// Reads the value of the element at the given index, with bound checks.
    ///
    /// # Panics
    /// The function panics if any of the following conditions are violated:
    /// * `col < self.ncols()`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, col: usize) -> E {
        E::faer_from_units(E::faer_map(
            self.get(col),
            #[inline(always)]
            |ptr| *ptr,
        ))
    }

    /// Returns a view over the transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub fn transpose(self) -> ColRef<'a, E> {
        unsafe { ColRef::__from_raw_parts(self.as_ptr(), self.ncols(), self.col_stride()) }
    }

    /// Returns a view over the conjugate of `self`.
    #[inline(always)]
    #[must_use]
    pub fn conjugate(self) -> RowRef<'a, E::Conj>
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
                self.ncols(),
                self.col_stride(),
            )
        }
    }

    /// Returns a view over the conjugate transpose of `self`.
    #[inline(always)]
    pub fn adjoint(self) -> ColRef<'a, E::Conj>
    where
        E: Conjugate,
    {
        self.conjugate().transpose()
    }

    /// Returns a view over the canonical representation of `self`, as well as a flag declaring
    /// whether `self` is implicitly conjugated or not.
    #[inline(always)]
    pub fn canonicalize(self) -> (RowRef<'a, E::Canonical>, Conj)
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
                    self.ncols(),
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

    /// Returns a view over the `self`, with the columns in reversed order.
    #[inline(always)]
    #[must_use]
    pub fn reverse_cols(self) -> Self {
        let ncols = self.ncols();
        let col_stride = self.col_stride().wrapping_neg();

        let ptr = unsafe { self.ptr_at_unchecked(ncols.saturating_sub(1)) };
        unsafe { Self::__from_raw_parts(ptr, ncols, col_stride) }
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
    pub unsafe fn subcols_unchecked(self, col_start: usize, ncols: usize) -> Self {
        debug_assert!(col_start <= self.ncols());
        debug_assert!(ncols <= self.ncols() - col_start);
        let col_stride = self.col_stride();
        unsafe { Self::__from_raw_parts(self.overflowing_ptr_at(col_start), ncols, col_stride) }
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
    pub fn subcols(self, col_start: usize, ncols: usize) -> Self {
        assert!(col_start <= self.ncols());
        assert!(ncols <= self.ncols() - col_start);
        unsafe { self.subcols_unchecked(col_start, ncols) }
    }

    /// Returns an owning [`Row`] of the data.
    #[inline]
    pub fn to_owned(&self) -> Row<E::Canonical>
    where
        E: Conjugate,
    {
        let mut mat = Row::new();
        mat.resize_with(
            self.ncols(),
            #[inline(always)]
            |col| unsafe { self.read_unchecked(col).canonicalize() },
        );
        mat
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

    /// Returns the row as a contiguous slice if its column stride is equal to `1`.
    ///
    /// # Note
    /// The values pointed to by the references are expected to be initialized, even if the
    /// pointed-to value is not read, otherwise the behavior is undefined.
    #[inline]
    pub fn try_as_slice(self) -> Option<GroupFor<E, &'a [E::Unit]>> {
        if self.col_stride() == 1 {
            let len = self.ncols();
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
    pub fn as_ref(&self) -> RowRef<'_, E> {
        *self
    }

    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn const_cast(self) -> RowMut<'a, E> {
        RowMut {
            inner: self.inner,
            __marker: PhantomData,
        }
    }
}

/// Creates a `RowRef` from pointers to the row vector data, number of columns, and column
/// stride.
///
/// # Safety:
/// This function has the same safety requirements as
/// [`mat::from_raw_parts(ptr, 1, ncols, 0, col_stride)`]
#[inline(always)]
pub unsafe fn from_raw_parts<'a, E: Entity>(
    ptr: GroupFor<E, *const E::Unit>,
    ncols: usize,
    col_stride: isize,
) -> RowRef<'a, E> {
    RowRef::__from_raw_parts(ptr, ncols, col_stride)
}

/// Creates a `RowRef` from slice views over the row vector data, The result has the same
/// number of columns as the length of the input slice.
#[inline(always)]
pub fn from_slice_generic<E: Entity>(slice: GroupFor<E, &[E::Unit]>) -> RowRef<'_, E> {
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

/// Creates a `RowRef` from slice views over the row vector data, The result has the same
/// number of columns as the length of the input slice.
#[inline(always)]
pub fn from_slice<E: SimpleEntity>(slice: &[E]) -> RowRef<'_, E> {
    from_slice_generic(slice)
}

impl<E: Entity> As2D<E> for RowRef<'_, E> {
    #[inline]
    fn as_2d_ref(&self) -> MatRef<'_, E> {
        (*self).as_2d()
    }
}

impl<E: Entity> AsRowRef<E> for RowRef<'_, E> {
    #[inline]
    fn as_row_ref(&self) -> RowRef<'_, E> {
        *self
    }
}

impl<'a, E: Entity> core::fmt::Debug for RowRef<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_2d().fmt(f)
    }
}

impl<E: SimpleEntity> core::ops::Index<usize> for RowRef<'_, E> {
    type Output = E;

    #[inline]
    #[track_caller]
    fn index(&self, col: usize) -> &E {
        self.get(col)
    }
}

impl<E: Conjugate> RowBatch<E> for RowRef<'_, E> {
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
    fn resize_owned(owned: &mut Self::Owned, nrows: usize, ncols: usize) {
        <Self::Owned as RowBatch<E::Canonical>>::resize_owned(owned, nrows, ncols)
    }
}

/// Returns a view over a row with `ncols` columns containing `value` repeated for all elements.
#[doc(alias = "broadcast")]
pub fn from_repeated_ref<E: SimpleEntity>(value: &E, ncols: usize) -> RowRef<'_, E> {
    unsafe { from_raw_parts(E::faer_map(value, |ptr| ptr as *const E::Unit), ncols, 0) }
}

/// Returns a view over a row with 1 column containing value as its only element, pointing to
/// `value`.
pub fn from_ref<E: SimpleEntity>(value: &E) -> RowRef<'_, E> {
    from_ref_generic(value)
}

/// Returns a view over a row with `ncols` columns containing `value` repeated for all elements.
#[doc(alias = "broadcast")]
pub fn from_repeated_ref_generic<E: Entity>(
    value: GroupFor<E, &E::Unit>,
    ncols: usize,
) -> RowRef<'_, E> {
    unsafe { from_raw_parts(E::faer_map(value, |ptr| ptr as *const E::Unit), ncols, 0) }
}

/// Returns a view over a row with 1 column containing value as its only element, pointing to
/// `value`.
pub fn from_ref_generic<E: Entity>(value: GroupFor<E, &E::Unit>) -> RowRef<'_, E> {
    from_repeated_ref_generic(value, 1)
}
