use crate::{assert, debug_assert};
use core::{marker::PhantomData, ops::Range};
use faer_entity::*;
use reborrow::*;

/// Wrapper around a group of references.
pub struct RefGroup<'a, E: Entity, T: 'a = <E as Entity>::Unit>(
    GroupCopyFor<E, *const T>,
    PhantomData<&'a ()>,
);
/// Wrapper around a group of mutable references.
pub struct RefGroupMut<'a, E: Entity, T: 'a = <E as Entity>::Unit>(
    GroupFor<E, *mut T>,
    PhantomData<&'a mut ()>,
);

/// Analogous to an immutable reference to a [prim@slice] for groups.
pub struct SliceGroup<'a, E: Entity, T: 'a = <E as Entity>::Unit>(
    GroupCopyFor<E, *const [T]>,
    PhantomData<&'a ()>,
);
/// Analogous to a mutable reference to a [prim@slice] for groups.
pub struct SliceGroupMut<'a, E: Entity, T: 'a = <E as Entity>::Unit>(
    GroupFor<E, *mut [T]>,
    PhantomData<&'a mut ()>,
);

impl<E: Entity, T: core::fmt::Debug> core::fmt::Debug for RefGroup<'_, E, T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        unsafe {
            transmute_unchecked::<GroupFor<E, &T>, GroupDebugFor<E, &T>>(self.into_inner()).fmt(f)
        }
    }
}
impl<E: Entity, T: core::fmt::Debug> core::fmt::Debug for RefGroupMut<'_, E, T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}
impl<E: Entity, T: core::fmt::Debug> core::fmt::Debug for SliceGroup<'_, E, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.into_ref_iter()).finish()
    }
}
impl<E: Entity, T: core::fmt::Debug> core::fmt::Debug for SliceGroupMut<'_, E, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

unsafe impl<E: Entity, T: Sync> Send for SliceGroup<'_, E, T> {}
unsafe impl<E: Entity, T: Sync> Sync for SliceGroup<'_, E, T> {}
unsafe impl<E: Entity, T: Send> Send for SliceGroupMut<'_, E, T> {}
unsafe impl<E: Entity, T: Sync> Sync for SliceGroupMut<'_, E, T> {}

impl<E: Entity, T> Copy for SliceGroup<'_, E, T> {}
impl<E: Entity, T> Copy for RefGroup<'_, E, T> {}
impl<E: Entity, T> Clone for SliceGroup<'_, E, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Entity, T> Clone for RefGroup<'_, E, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, E: Entity, T> RefGroup<'a, E, T> {
    /// Create a new [`RefGroup`] from a group of references.
    #[inline(always)]
    pub fn new(reference: GroupFor<E, &'a T>) -> Self {
        Self(
            into_copy::<E, _>(E::faer_map(
                reference,
                #[inline(always)]
                |reference| reference as *const T,
            )),
            PhantomData,
        )
    }

    /// Consume `self` to return the internally stored group of references.
    #[inline(always)]
    pub fn into_inner(self) -> GroupFor<E, &'a T> {
        E::faer_map(
            from_copy::<E, _>(self.0),
            #[inline(always)]
            |ptr| unsafe { &*ptr },
        )
    }

    /// Copies and returns the value pointed to by the references.
    #[inline(always)]
    pub fn get(self) -> GroupCopyFor<E, T>
    where
        T: Copy,
    {
        into_copy::<E, _>(E::faer_deref(self.into_inner()))
    }
}

impl<'a, E: Entity, T, const N: usize> RefGroup<'a, E, [T; N]> {
    /// Convert a reference to an array to an array of references.
    #[inline(always)]
    pub fn unzip(self) -> [RefGroup<'a, E, T>; N] {
        unsafe {
            let mut out = transmute_unchecked::<
                core::mem::MaybeUninit<[RefGroup<'a, E, T>; N]>,
                [core::mem::MaybeUninit<RefGroup<'a, E, T>>; N],
            >(core::mem::MaybeUninit::<[RefGroup<'a, E, T>; N]>::uninit());
            for (out, inp) in core::iter::zip(out.iter_mut(), E::faer_into_iter(self.into_inner()))
            {
                out.write(RefGroup::new(inp));
            }
            transmute_unchecked::<
                [core::mem::MaybeUninit<RefGroup<'a, E, T>>; N],
                [RefGroup<'a, E, T>; N],
            >(out)
        }
    }
}

impl<'a, E: Entity, T, const N: usize> RefGroupMut<'a, E, [T; N]> {
    /// Convert a mutable reference to an array to an array of mutable references.
    #[inline(always)]
    pub fn unzip(self) -> [RefGroupMut<'a, E, T>; N] {
        unsafe {
            let mut out =
                transmute_unchecked::<
                    core::mem::MaybeUninit<[RefGroupMut<'a, E, T>; N]>,
                    [core::mem::MaybeUninit<RefGroupMut<'a, E, T>>; N],
                >(core::mem::MaybeUninit::<[RefGroupMut<'a, E, T>; N]>::uninit());
            for (out, inp) in core::iter::zip(out.iter_mut(), E::faer_into_iter(self.into_inner()))
            {
                out.write(RefGroupMut::new(inp));
            }
            transmute_unchecked::<
                [core::mem::MaybeUninit<RefGroupMut<'a, E, T>>; N],
                [RefGroupMut<'a, E, T>; N],
            >(out)
        }
    }
}

impl<'a, E: Entity, T> RefGroupMut<'a, E, T> {
    /// Create a new [`RefGroupMut`] from a group of mutable references.
    #[inline(always)]
    pub fn new(reference: GroupFor<E, &'a mut T>) -> Self {
        Self(
            E::faer_map(
                reference,
                #[inline(always)]
                |reference| reference as *mut T,
            ),
            PhantomData,
        )
    }

    /// Consume `self` to return the internally stored group of references.
    #[inline(always)]
    pub fn into_inner(self) -> GroupFor<E, &'a mut T> {
        E::faer_map(
            self.0,
            #[inline(always)]
            |ptr| unsafe { &mut *ptr },
        )
    }

    /// Copies and returns the value pointed to by the references.
    #[inline(always)]
    pub fn get(&self) -> GroupCopyFor<E, T>
    where
        T: Copy,
    {
        self.rb().get()
    }

    /// Writes `value` to the location pointed to by the references.
    #[inline(always)]
    pub fn set(&mut self, value: GroupCopyFor<E, T>)
    where
        T: Copy,
    {
        E::faer_map(
            E::faer_zip(self.rb_mut().into_inner(), from_copy::<E, _>(value)),
            #[inline(always)]
            |(r, value)| *r = value,
        );
    }
}

impl<'a, E: Entity, T> IntoConst for SliceGroup<'a, E, T> {
    type Target = SliceGroup<'a, E, T>;

    #[inline(always)]
    fn into_const(self) -> Self::Target {
        self
    }
}
impl<'a, E: Entity, T> IntoConst for SliceGroupMut<'a, E, T> {
    type Target = SliceGroup<'a, E, T>;

    #[inline(always)]
    fn into_const(self) -> Self::Target {
        SliceGroup::new(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| &*slice,
        ))
    }
}

impl<'a, E: Entity, T> IntoConst for RefGroup<'a, E, T> {
    type Target = RefGroup<'a, E, T>;

    #[inline(always)]
    fn into_const(self) -> Self::Target {
        self
    }
}
impl<'a, E: Entity, T> IntoConst for RefGroupMut<'a, E, T> {
    type Target = RefGroup<'a, E, T>;

    #[inline(always)]
    fn into_const(self) -> Self::Target {
        RefGroup::new(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| &*slice,
        ))
    }
}

impl<'short, 'a, E: Entity, T> ReborrowMut<'short> for RefGroup<'a, E, T> {
    type Target = RefGroup<'short, E, T>;

    #[inline(always)]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'short, 'a, E: Entity, T> Reborrow<'short> for RefGroup<'a, E, T> {
    type Target = RefGroup<'short, E, T>;

    #[inline(always)]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, 'a, E: Entity, T> ReborrowMut<'short> for RefGroupMut<'a, E, T> {
    type Target = RefGroupMut<'short, E, T>;

    #[inline(always)]
    fn rb_mut(&'short mut self) -> Self::Target {
        RefGroupMut::new(E::faer_map(
            E::faer_as_mut(&mut self.0),
            #[inline(always)]
            |this| unsafe { &mut **this },
        ))
    }
}

impl<'short, 'a, E: Entity, T> Reborrow<'short> for RefGroupMut<'a, E, T> {
    type Target = RefGroup<'short, E, T>;

    #[inline(always)]
    fn rb(&'short self) -> Self::Target {
        RefGroup::new(E::faer_map(
            E::faer_as_ref(&self.0),
            #[inline(always)]
            |this| unsafe { &**this },
        ))
    }
}

impl<'a, E: Entity, T> SliceGroup<'a, E, T> {
    /// Create a new [`SliceGroup`] from a group of slice references.
    #[inline(always)]
    pub fn new(slice: GroupFor<E, &'a [T]>) -> Self {
        Self(
            into_copy::<E, _>(E::faer_map(slice, |slice| slice as *const [T])),
            PhantomData,
        )
    }

    /// Consume `self` to return the internally stored group of slice references.
    #[inline(always)]
    pub fn into_inner(self) -> GroupFor<E, &'a [T]> {
        unsafe { E::faer_map(from_copy::<E, _>(self.0), |ptr| &*ptr) }
    }

    /// Decompose `self` into a slice of arrays of size `N`, and a remainder part with length
    /// `< N`.
    #[inline(always)]
    pub fn as_arrays<const N: usize>(self) -> (SliceGroup<'a, E, [T; N]>, SliceGroup<'a, E, T>) {
        let (head, tail) = E::faer_as_arrays::<N, _>(self.into_inner());
        (SliceGroup::new(head), SliceGroup::new(tail))
    }
}

impl<'a, E: Entity, T> SliceGroupMut<'a, E, T> {
    /// Create a new [`SliceGroup`] from a group of mutable slice references.
    #[inline(always)]
    pub fn new(slice: GroupFor<E, &'a mut [T]>) -> Self {
        Self(E::faer_map(slice, |slice| slice as *mut [T]), PhantomData)
    }

    /// Consume `self` to return the internally stored group of mutable slice references.
    #[inline(always)]
    pub fn into_inner(self) -> GroupFor<E, &'a mut [T]> {
        unsafe { E::faer_map(self.0, |ptr| &mut *ptr) }
    }

    /// Decompose `self` into a mutable slice of arrays of size `N`, and a remainder part with
    /// length `< N`.
    #[inline(always)]
    pub fn as_arrays_mut<const N: usize>(
        self,
    ) -> (SliceGroupMut<'a, E, [T; N]>, SliceGroupMut<'a, E, T>) {
        let (head, tail) = E::faer_as_arrays_mut::<N, _>(self.into_inner());
        (SliceGroupMut::new(head), SliceGroupMut::new(tail))
    }
}

impl<'short, 'a, E: Entity, T> ReborrowMut<'short> for SliceGroup<'a, E, T> {
    type Target = SliceGroup<'short, E, T>;

    #[inline(always)]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}

impl<'short, 'a, E: Entity, T> Reborrow<'short> for SliceGroup<'a, E, T> {
    type Target = SliceGroup<'short, E, T>;

    #[inline(always)]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}

impl<'short, 'a, E: Entity, T> ReborrowMut<'short> for SliceGroupMut<'a, E, T> {
    type Target = SliceGroupMut<'short, E, T>;

    #[inline(always)]
    fn rb_mut(&'short mut self) -> Self::Target {
        SliceGroupMut::new(E::faer_map(
            E::faer_as_mut(&mut self.0),
            #[inline(always)]
            |this| unsafe { &mut **this },
        ))
    }
}

impl<'short, 'a, E: Entity, T> Reborrow<'short> for SliceGroupMut<'a, E, T> {
    type Target = SliceGroup<'short, E, T>;

    #[inline(always)]
    fn rb(&'short self) -> Self::Target {
        SliceGroup::new(E::faer_map(
            E::faer_as_ref(&self.0),
            #[inline(always)]
            |this| unsafe { &**this },
        ))
    }
}

impl<'a, E: Entity> RefGroup<'a, E> {
    /// Read the element pointed to by the references.
    #[inline(always)]
    pub fn read(&self) -> E {
        E::faer_from_units(E::faer_deref(self.into_inner()))
    }
}

impl<'a, E: Entity> RefGroupMut<'a, E> {
    /// Read the element pointed to by the references.
    #[inline(always)]
    pub fn read(&self) -> E {
        self.rb().read()
    }

    /// Write `value` to the location pointed to by the references.
    #[inline(always)]
    pub fn write(&mut self, value: E) {
        E::faer_map(
            E::faer_zip(self.rb_mut().into_inner(), value.faer_into_units()),
            #[inline(always)]
            |(r, value)| *r = value,
        );
    }
}

impl<'a, E: Entity> SliceGroup<'a, E> {
    /// Read the element at position `idx`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, idx: usize) -> E {
        assert!(idx < self.len());
        unsafe { self.read_unchecked(idx) }
    }

    /// Read the element at position `idx`, without bound checks.
    ///
    /// # Safety
    /// The behavior is undefined if `idx >= self.len()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, idx: usize) -> E {
        debug_assert!(idx < self.len());
        E::faer_from_units(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| *slice.get_unchecked(idx),
        ))
    }
}
impl<'a, E: Entity, T> SliceGroup<'a, E, T> {
    /// Get a [`RefGroup`] pointing to the element at position `idx`.
    #[inline(always)]
    #[track_caller]
    pub fn get(self, idx: usize) -> RefGroup<'a, E, T> {
        assert!(idx < self.len());
        unsafe { self.get_unchecked(idx) }
    }

    /// Get a [`RefGroup`] pointing to the element at position `idx`, without bound checks.
    ///
    /// # Safety
    /// The behavior is undefined if `idx >= self.len()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn get_unchecked(self, idx: usize) -> RefGroup<'a, E, T> {
        debug_assert!(idx < self.len());
        RefGroup::new(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| slice.get_unchecked(idx),
        ))
    }

    /// Checks whether the slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the length of the slice.
    #[inline]
    pub fn len(&self) -> usize {
        let mut len = usize::MAX;
        E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| len = Ord::min(len, slice.len()),
        );
        len
    }

    /// Returns the subslice of `self` from the start to the end of the provided range.
    #[inline(always)]
    #[track_caller]
    pub fn subslice(self, range: Range<usize>) -> Self {
        assert!(all(range.start <= range.end, range.end <= self.len()));
        unsafe { self.subslice_unchecked(range) }
    }

    /// Split `self` at the midpoint `idx`, and return the two parts.
    #[inline(always)]
    #[track_caller]
    pub fn split_at(self, idx: usize) -> (Self, Self) {
        assert!(idx <= self.len());
        let (head, tail) = E::faer_unzip(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| slice.split_at(idx),
        ));
        (Self::new(head), Self::new(tail))
    }

    /// Returns the subslice of `self` from the start to the end of the provided range, without
    /// bound checks.
    ///
    /// # Safety
    /// The behavior is undefined if `range.start > range.end` or `range.end > self.len()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn subslice_unchecked(self, range: Range<usize>) -> Self {
        debug_assert!(all(range.start <= range.end, range.end <= self.len()));
        Self::new(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| slice.get_unchecked(range.start..range.end),
        ))
    }

    /// Returns an iterator of [`RefGroup`] over the elements of the slice.
    #[inline(always)]
    pub fn into_ref_iter(self) -> impl Iterator<Item = RefGroup<'a, E, T>> {
        E::faer_into_iter(self.into_inner()).map(RefGroup::new)
    }

    /// Returns an iterator of slices over chunks of size `chunk_size`, and the remainder of
    /// the slice.
    #[inline(always)]
    pub fn into_chunks_exact(
        self,
        chunk_size: usize,
    ) -> (impl Iterator<Item = SliceGroup<'a, E, T>>, Self) {
        let len = self.len();
        let mid = len / chunk_size * chunk_size;
        let (head, tail) = E::faer_unzip(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| slice.split_at(mid),
        ));
        let head = E::faer_map(
            head,
            #[inline(always)]
            |head| head.chunks_exact(chunk_size),
        );
        (
            E::faer_into_iter(head).map(SliceGroup::new),
            SliceGroup::new(tail),
        )
    }
}

impl<'a, E: Entity> SliceGroupMut<'a, E> {
    /// Read the element at position `idx`.
    #[inline(always)]
    #[track_caller]
    pub fn read(&self, idx: usize) -> E {
        self.rb().read(idx)
    }

    /// Read the element at position `idx`, without bound checks.
    ///
    /// # Safety
    /// The behavior is undefined if `idx >= self.len()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, idx: usize) -> E {
        self.rb().read_unchecked(idx)
    }

    /// Write `value` to the location at position `idx`.
    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, idx: usize, value: E) {
        assert!(idx < self.len());
        unsafe { self.write_unchecked(idx, value) }
    }

    /// Write `value` to the location at position `idx`, without bound checks.
    ///
    /// # Safety
    /// The behavior is undefined if `idx >= self.len()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn write_unchecked(&mut self, idx: usize, value: E) {
        debug_assert!(idx < self.len());
        E::faer_map(
            E::faer_zip(self.rb_mut().into_inner(), value.faer_into_units()),
            #[inline(always)]
            |(slice, value)| *slice.get_unchecked_mut(idx) = value,
        );
    }

    /// Fill the slice with zeros.
    #[inline]
    pub fn fill_zero(&mut self) {
        E::faer_map(self.rb_mut().into_inner(), |slice| unsafe {
            let len = slice.len();
            core::ptr::write_bytes(slice.as_mut_ptr(), 0u8, len);
        });
    }
}

impl<'a, E: Entity, T> SliceGroupMut<'a, E, T> {
    /// Get a [`RefGroupMut`] pointing to the element at position `idx`.
    #[inline(always)]
    #[track_caller]
    pub fn get_mut(self, idx: usize) -> RefGroupMut<'a, E, T> {
        assert!(idx < self.len());
        unsafe { self.get_unchecked_mut(idx) }
    }

    /// Get a [`RefGroupMut`] pointing to the element at position `idx`.
    ///
    /// # Safety
    /// The behavior is undefined if `idx >= self.len()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn get_unchecked_mut(self, idx: usize) -> RefGroupMut<'a, E, T> {
        debug_assert!(idx < self.len());
        RefGroupMut::new(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| slice.get_unchecked_mut(idx),
        ))
    }

    /// Get a [`RefGroup`] pointing to the element at position `idx`.
    #[inline(always)]
    #[track_caller]
    pub fn get(self, idx: usize) -> RefGroup<'a, E, T> {
        self.into_const().get(idx)
    }

    /// Get a [`RefGroup`] pointing to the element at position `idx`, without bound checks.
    ///
    /// # Safety
    /// The behavior is undefined if `idx >= self.len()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn get_unchecked(self, idx: usize) -> RefGroup<'a, E, T> {
        self.into_const().get_unchecked(idx)
    }

    /// Checks whether the slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rb().is_empty()
    }

    /// Returns the length of the slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.rb().len()
    }

    /// Returns the subslice of `self` from the start to the end of the provided range.
    #[inline(always)]
    #[track_caller]
    pub fn subslice(self, range: Range<usize>) -> Self {
        assert!(all(range.start <= range.end, range.end <= self.len()));
        unsafe { self.subslice_unchecked(range) }
    }

    /// Returns the subslice of `self` from the start to the end of the provided range, without
    /// bound checks.
    ///
    /// # Safety
    /// The behavior is undefined if `range.start > range.end` or `range.end > self.len()`.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn subslice_unchecked(self, range: Range<usize>) -> Self {
        debug_assert!(all(range.start <= range.end, range.end <= self.len()));
        Self::new(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| slice.get_unchecked_mut(range.start..range.end),
        ))
    }

    /// Returns an iterator of [`RefGroupMut`] over the elements of the slice.
    #[inline(always)]
    pub fn into_mut_iter(self) -> impl Iterator<Item = RefGroupMut<'a, E, T>> {
        E::faer_into_iter(self.into_inner()).map(RefGroupMut::new)
    }

    /// Split `self` at the midpoint `idx`, and return the two parts.
    #[inline(always)]
    #[track_caller]
    pub fn split_at(self, idx: usize) -> (Self, Self) {
        assert!(idx <= self.len());
        let (head, tail) = E::faer_unzip(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| slice.split_at_mut(idx),
        ));
        (Self::new(head), Self::new(tail))
    }

    /// Returns an iterator of slices over chunks of size `chunk_size`, and the remainder of
    /// the slice.
    #[inline(always)]
    pub fn into_chunks_exact(
        self,
        chunk_size: usize,
    ) -> (impl Iterator<Item = SliceGroupMut<'a, E, T>>, Self) {
        let len = self.len();
        let mid = len % chunk_size * chunk_size;
        let (head, tail) = E::faer_unzip(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| slice.split_at_mut(mid),
        ));
        let head = E::faer_map(
            head,
            #[inline(always)]
            |head| head.chunks_exact_mut(chunk_size),
        );
        (
            E::faer_into_iter(head).map(SliceGroupMut::new),
            SliceGroupMut::new(tail),
        )
    }
}

impl<E: Entity, T: Copy + core::fmt::Debug> pulp::Read for RefGroupMut<'_, E, T> {
    type Output = GroupCopyFor<E, T>;
    #[inline(always)]
    fn read_or(&self, _or: Self::Output) -> Self::Output {
        self.get()
    }
}
impl<E: Entity, T: Copy + core::fmt::Debug> pulp::Write for RefGroupMut<'_, E, T> {
    #[inline(always)]
    fn write(&mut self, values: Self::Output) {
        self.set(values)
    }
}
impl<E: Entity, T: Copy + core::fmt::Debug> pulp::Read for RefGroup<'_, E, T> {
    type Output = GroupCopyFor<E, T>;
    #[inline(always)]
    fn read_or(&self, _or: Self::Output) -> Self::Output {
        self.get()
    }
}
