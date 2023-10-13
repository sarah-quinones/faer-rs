// TODO: document safety requirements
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![forbid(elided_lifetimes_in_paths)]
#![allow(non_snake_case)]

use crate::mem::{__get_checked, __get_unchecked};
pub use __core::*;
use assert2::assert;
use bytemuck::{Pod, Zeroable};
use core::{cell::Cell, iter::zip, ops::Range};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{transmute_unchecked, ComplexField, Entity};
use faer_entity::RealField;
use mem::NONE;
use reborrow::*;

#[allow(unused_macros)]
macro_rules! shadow {
    ($id: ident) => {
        let $id = {
            struct Shadowed;
            impl ::core::ops::Drop for Shadowed {
                fn drop(&mut self) {}
            }
            Shadowed
        };
        ::core::mem::drop($id);
    };
}

macro_rules! impl_copy {
    (< $($lt_param: lifetime),* >< $($ty_param: ident $(: $tt: tt)?),* > <$ty: ty>) => {
        impl<$($lt_param,)* $($ty_param $(: $tt)?,)*> Copy for $ty {}
        impl<$($lt_param,)* $($ty_param $(: $tt)?,)*> Clone for $ty {
            #[inline(always)]
            fn clone(&self) -> Self {
                *self
            }
        }
    };
}

macro_rules! impl_deref {
    (< $($lt_param: lifetime),* >< $($ty_param: ident $(: $tt: tt)?),* ><Target = $target:ty> <$ty: ty>) => {
        impl<$($lt_param,)* $($ty_param $(: $tt)?,)*> ::core::ops::Deref for $ty {
            type Target = $target;

            #[inline]
            fn deref(&self) -> &<Self as ::core::ops::Deref>::Target {
                &*self.0
            }
        }
    };
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[non_exhaustive]
pub enum FaerError {
    IndexOverflow,
    OutOfMemory,
}

impl core::fmt::Display for FaerError {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for FaerError {}

#[inline]
#[track_caller]
fn try_zeroed<I: Pod>(n: usize) -> Result<Vec<I>, FaerError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(nomem)?;
    unsafe {
        core::ptr::write_bytes::<I>(v.as_mut_ptr(), 0u8, n);
        v.set_len(n);
    }
    Ok(v)
}

#[inline]
#[track_caller]
fn try_collect<I: IntoIterator>(iter: I) -> Result<Vec<I::Item>, FaerError>
where
    I::IntoIter: ExactSizeIterator,
{
    let iter = iter.into_iter();
    let mut v = Vec::new();
    v.try_reserve_exact(iter.len()).map_err(nomem)?;
    v.extend(iter);
    Ok(v)
}

#[inline]
fn nomem<T>(_: T) -> FaerError {
    FaerError::OutOfMemory
}

fn make_raw_req<E: Entity>(size: usize) -> Result<StackReq, SizeOverflow> {
    let req = Ok(StackReq::empty());
    let additional = StackReq::try_new::<E::Unit>(size)?;
    let (req, _) = E::faer_map_with_context(req, E::faer_from_copy(E::UNIT), &mut {
        #[inline(always)]
        |req, ()| {
            let req = match req {
                Ok(req) => req.try_and(additional),
                _ => Err(SizeOverflow),
            };
            (req, ())
        }
    });
    req
}

fn make_raw<E: Entity>(
    size: usize,
    stack: PodStack<'_>,
) -> (E::Group<dyn_stack::DynArray<'_, E::Unit>>, PodStack<'_>) {
    let (stack, array) = E::faer_map_with_context(stack, E::faer_from_copy(E::UNIT), &mut {
        #[inline(always)]
        |stack, ()| {
            let (alloc, stack) = stack.make_raw::<E::Unit>(size);
            (stack, alloc)
        }
    });
    (array, stack)
}

pub mod amd;
pub mod cholesky;

#[doc(hidden)]
pub mod ghost;

mod mem;
mod simd;

mod seal {
    pub trait Seal {}

    impl Seal for i32 {}
    impl Seal for i64 {}
}

pub trait Index:
    seal::Seal
    + core::fmt::Debug
    + core::ops::Neg<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + Pod
    + Eq
    + Ord
    + Send
    + Sync
    + 'static
{
    #[inline]
    fn incr(&mut self) {
        *self += Self::truncate(1)
    }
    #[inline]
    fn decr(&mut self) {
        *self -= Self::truncate(1)
    }

    const MAX: Self;

    #[must_use]
    fn truncate(value: usize) -> Self;

    /// zero extend
    #[must_use]
    fn zx(self) -> usize;
    /// sign extend
    #[must_use]
    fn sx(self) -> usize;

    /// sum with overflow check
    #[must_use]
    fn sum_nonnegative(slice: &[Self]) -> Option<Self>;
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl Index for i32 {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const _: () = {
            core::assert!(i32::BITS <= usize::BITS);
        };
        value as isize as Self
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as u32 as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as isize as usize
    }

    #[inline(always)]
    fn sum_nonnegative(slice: &[Self]) -> Option<Self> {
        simd::sum_i32(slice)
    }
}

#[cfg(target_pointer_width = "64")]
impl Index for i64 {
    const MAX: Self = Self::MAX;

    #[inline(always)]
    fn truncate(value: usize) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const _: () = {
            core::assert!(i64::BITS <= usize::BITS);
        };
        value as isize as Self
    }

    #[inline(always)]
    fn zx(self) -> usize {
        self as u64 as usize
    }

    #[inline(always)]
    fn sx(self) -> usize {
        self as isize as usize
    }

    #[inline(always)]
    fn sum_nonnegative(slice: &[Self]) -> Option<Self> {
        simd::sum_i64(slice)
    }
}

mod __core {
    use super::*;
    use assert2::{assert, debug_assert};
    use core::marker::PhantomData;

    #[derive(Debug)]
    pub struct PermutationRef<'a, I> {
        fwd: &'a [I],
        inv: &'a [I],
    }

    #[derive(Debug)]
    pub struct SymbolicSparseColMatRef<'a, I> {
        nrows: usize,
        ncols: usize,
        col_ptr: &'a [I],
        col_nnz: Option<&'a [I]>,
        row_ind: &'a [I],
    }

    // #[derive(Debug)]
    pub struct SparseColMatRef<'a, I, E: Entity> {
        symbolic: SymbolicSparseColMatRef<'a, I>,
        values: SliceGroup<'a, E>,
    }

    impl<'a, I: Index> PermutationRef<'a, I> {
        #[inline]
        #[track_caller]
        pub fn new_checked(fwd: &'a [I], inv: &'a [I]) -> Self {
            let n = fwd.len();
            assert!(fwd.len() == inv.len());
            assert!(n < I::MAX.zx());
            for (i, &p) in fwd.iter().enumerate() {
                let p = p.sx();
                assert!(p < n);
                assert!(inv[p].sx() == i);
            }
            Self { fwd, inv }
        }

        #[inline]
        #[track_caller]
        pub unsafe fn new_unchecked(fwd: &'a [I], inv: &'a [I]) -> Self {
            let n = fwd.len();
            let _ = n;
            debug_assert!(fwd.len() == inv.len());
            debug_assert!(n < I::MAX.zx());
            Self { fwd, inv }
        }

        #[inline]
        pub fn inverse(self) -> PermutationRef<'a, I> {
            PermutationRef {
                fwd: self.inv,
                inv: self.fwd,
            }
        }

        #[inline]
        pub fn fwd_inv(self) -> (&'a [I], &'a [I]) {
            (self.fwd, self.inv)
        }

        #[inline]
        pub fn len(&self) -> usize {
            self.fwd.len()
        }
    }

    /// Requires:
    /// * `nrows <= I::MAX` (always checked)
    /// * `ncols <= I::MAX` (always checked)
    /// * `col_ptrs` has length n (always checked)
    /// * `col_ptrs` is non-decreasing
    /// * `col_ptrs[0]..col_ptrs[n]` is a valid range in row_indices (always checked, assuming
    ///   non-decreasing)
    /// * if `nnz_per_col` is `None`, elements of `row_indices[col_ptrs[j]..col_ptrs[j + 1]]` are
    ///   less than `nrows`
    ///
    /// * `nnz_per_col[j] <= col_ptrs[j+1] - col_ptrs[j]`
    /// * if `nnz_per_col` is `Some(_)`, elements of `row_indices[col_ptrs[j]..][..nnz_per_col[j]]`
    ///   are less than `nrows`
    ///
    /// Ensures:
    /// * `self.compute_nnz() <= I::MAX`
    impl<'a, I: Index> SymbolicSparseColMatRef<'a, I> {
        #[inline]
        #[track_caller]
        pub fn new_checked(
            nrows: usize,
            ncols: usize,
            col_ptrs: &'a [I],
            nnz_per_col: Option<&'a [I]>,
            row_indices: &'a [I],
        ) -> Self {
            assert!(ncols <= I::MAX.zx());
            assert!(nrows <= I::MAX.zx());
            assert!(col_ptrs.len() == ncols + 1);
            assert!(col_ptrs[0] >= I::truncate(0));
            for &[c, c_next] in windows2(col_ptrs) {
                assert!(c <= c_next);
            }
            assert!(col_ptrs[ncols].zx() <= row_indices.len());

            if let Some(nnz_per_col) = nnz_per_col {
                for (&nnz_j, &[c, c_next]) in zip(nnz_per_col, windows2(col_ptrs)) {
                    assert!(nnz_j <= c_next - c);
                    for &i in &row_indices[c.zx()..c.zx() + nnz_j.zx()] {
                        assert!(i.sx() < nrows);
                    }
                }
            } else {
                let c0 = col_ptrs[0].zx();
                let cn = col_ptrs[ncols].zx();
                for &i in &row_indices[c0..cn] {
                    assert!(i.sx() < nrows);
                }
            }

            Self {
                nrows,
                ncols,
                col_ptr: col_ptrs,
                col_nnz: nnz_per_col,
                row_ind: row_indices,
            }
        }

        #[inline(always)]
        #[track_caller]
        pub unsafe fn new_unchecked(
            nrows: usize,
            ncols: usize,
            col_ptrs: &'a [I],
            nnz_per_col: Option<&'a [I]>,
            row_indices: &'a [I],
        ) -> Self {
            assert!(ncols <= I::MAX.zx());
            assert!(nrows <= I::MAX.zx());
            assert!(col_ptrs.len() == ncols + 1);
            assert!(col_ptrs[0] >= I::truncate(0));
            assert!(col_ptrs[ncols].zx() <= row_indices.len());

            Self {
                nrows,
                ncols,
                col_ptr: col_ptrs,
                col_nnz: nnz_per_col,
                row_ind: row_indices,
            }
        }

        #[inline]
        pub fn nrows(&self) -> usize {
            self.nrows
        }
        #[inline]
        pub fn ncols(&self) -> usize {
            self.ncols
        }

        #[inline]
        pub fn compute_nnz(&self) -> usize {
            match self.col_nnz {
                Some(col_nnz) => {
                    let mut nnz = 0usize;
                    for &nnz_j in col_nnz {
                        // can't overflow
                        nnz += nnz_j.zx();
                    }
                    nnz
                }
                None => self.col_ptr[self.ncols].zx() - self.col_ptr[0].zx(),
            }
        }

        #[inline]
        pub fn col_ptrs(&self) -> &'a [I] {
            self.col_ptr
        }

        #[inline]
        pub fn nnz_per_col(&self) -> Option<&'a [I]> {
            self.col_nnz
        }

        #[inline]
        pub fn row_indices(&self) -> &'a [I] {
            self.row_ind
        }

        #[inline]
        #[track_caller]
        pub fn row_indices_of_col_raw(&self, j: usize) -> &'a [I] {
            __get_checked(self.row_ind, self.col_range(j))
        }

        #[inline]
        #[track_caller]
        pub fn row_indices_of_col(
            &self,
            j: usize,
        ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = usize> {
            self.row_indices_of_col_raw(j).iter().map(
                #[inline(always)]
                |&i| i.zx(),
            )
        }

        #[inline]
        #[track_caller]
        pub fn col_range(&self, j: usize) -> Range<usize> {
            let start = __get_checked(self.col_ptr, j).zx();
            let end = self
                .col_nnz
                .map(|col_nnz| (__get_checked(col_nnz, j).zx() + start))
                .unwrap_or(__get_checked(self.col_ptr, j + 1).zx());

            start..end
        }

        #[inline]
        #[track_caller]
        pub unsafe fn col_range_unchecked(&self, j: usize) -> Range<usize> {
            let start = __get_unchecked(self.col_ptr, j).zx();
            let end = self
                .col_nnz
                .map(|col_nnz| (__get_unchecked(col_nnz, j).zx() + start))
                .unwrap_or(__get_unchecked(self.col_ptr, j + 1).zx());

            start..end
        }
    }

    impl<'a, I: Index, E: Entity> SparseColMatRef<'a, I, E> {
        #[inline]
        #[track_caller]
        pub fn new(symbolic: SymbolicSparseColMatRef<'a, I>, values: SliceGroup<'a, E>) -> Self {
            assert!(symbolic.row_indices().len() == values.len());
            Self { symbolic, values }
        }

        #[inline]
        pub fn nrows(&self) -> usize {
            self.symbolic.nrows
        }
        #[inline]
        pub fn ncols(&self) -> usize {
            self.symbolic.ncols
        }

        #[inline]
        pub fn compute_nnz(&self) -> usize {
            self.symbolic.compute_nnz()
        }

        #[inline]
        pub fn col_ptrs(&self) -> &'a [I] {
            self.symbolic.col_ptr
        }

        #[inline]
        pub fn nnz_per_col(&self) -> Option<&'a [I]> {
            self.symbolic.col_nnz
        }

        #[inline]
        pub fn row_indices(&self) -> &'a [I] {
            self.symbolic.row_ind
        }

        #[inline]
        pub fn values(&self) -> SliceGroup<'a, E> {
            self.values
        }

        #[inline]
        #[track_caller]
        pub fn row_indices_of_col_raw(&self, j: usize) -> &'a [I] {
            __get_checked(self.symbolic.row_ind, self.col_range(j))
        }

        #[inline]
        #[track_caller]
        pub fn row_indices_of_col(
            &self,
            j: usize,
        ) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = usize> {
            self.symbolic.row_indices_of_col(j)
        }

        #[inline]
        #[track_caller]
        pub fn values_of_col(&self, j: usize) -> SliceGroup<'a, E> {
            self.values.subslice(self.col_range(j))
        }

        #[inline]
        pub fn symbolic(self) -> SymbolicSparseColMatRef<'a, I> {
            self.symbolic
        }

        #[inline]
        #[track_caller]
        pub fn col_range(&self, j: usize) -> Range<usize> {
            self.symbolic.col_range(j)
        }

        #[inline]
        #[track_caller]
        pub unsafe fn col_range_unchecked(&self, j: usize) -> Range<usize> {
            self.symbolic.col_range_unchecked(j)
        }
    }

    pub struct SliceGroup<'a, E: Entity>(E::GroupCopy<&'static [E::Unit]>, PhantomData<&'a ()>);
    pub struct SliceGroupMut<'a, E: Entity>(
        E::Group<&'static mut [E::Unit]>,
        PhantomData<&'a mut ()>,
    );

    pub struct RefGroup<'a, E: Entity>(E::GroupCopy<&'static E::Unit>, PhantomData<&'a ()>);
    pub struct RefGroupMut<'a, E: Entity>(E::Group<&'static mut E::Unit>, PhantomData<&'a mut ()>);

    impl_copy!(<'a><E: Entity><SliceGroup<'a, E>>);
    impl_copy!(<'a><E: Entity><RefGroup<'a, E>>);

    impl<'a, E: Entity> RefGroup<'a, E> {
        #[inline(always)]
        pub fn new(slice: E::Group<&'a E::Unit>) -> Self {
            Self(unsafe { transmute_unchecked(slice) }, PhantomData)
        }

        #[inline(always)]
        pub fn into_inner(self) -> E::Group<&'a E::Unit> {
            unsafe { transmute_unchecked(self.0) }
        }
    }

    impl<'a, E: Entity> RefGroupMut<'a, E> {
        #[inline(always)]
        pub fn new(slice: E::Group<&'a mut E::Unit>) -> Self {
            Self(unsafe { transmute_unchecked(slice) }, PhantomData)
        }

        #[inline(always)]
        pub fn into_inner(self) -> E::Group<&'a mut E::Unit> {
            unsafe { transmute_unchecked(self.0) }
        }
    }

    impl<'a, E: Entity> IntoConst for SliceGroup<'a, E> {
        type Target = SliceGroup<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            self
        }
    }
    impl<'a, E: Entity> IntoConst for SliceGroupMut<'a, E> {
        type Target = SliceGroup<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            SliceGroup::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| &*slice,
            ))
        }
    }

    impl<'a, E: Entity> IntoConst for RefGroup<'a, E> {
        type Target = RefGroup<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            self
        }
    }
    impl<'a, E: Entity> IntoConst for RefGroupMut<'a, E> {
        type Target = RefGroup<'a, E>;

        #[inline(always)]
        fn into_const(self) -> Self::Target {
            RefGroup::new(E::faer_map(
                self.into_inner(),
                #[inline(always)]
                |slice| &*slice,
            ))
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for RefGroup<'a, E> {
        type Target = RefGroup<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for RefGroup<'a, E> {
        type Target = RefGroup<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for RefGroupMut<'a, E> {
        type Target = RefGroupMut<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            RefGroupMut::new(E::faer_map(
                E::faer_as_mut(&mut self.0),
                #[inline(always)]
                |this| &mut **this,
            ))
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for RefGroupMut<'a, E> {
        type Target = RefGroup<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            RefGroup::new(E::faer_map(
                E::faer_as_ref(&self.0),
                #[inline(always)]
                |this| &**this,
            ))
        }
    }

    impl<'a, E: Entity> SliceGroup<'a, E> {
        #[inline(always)]
        pub fn new(slice: E::Group<&'a [E::Unit]>) -> Self {
            Self(unsafe { transmute_unchecked(slice) }, PhantomData)
        }

        #[inline(always)]
        pub fn into_inner(self) -> E::Group<&'a [E::Unit]> {
            unsafe { transmute_unchecked(self.0) }
        }
    }

    impl<'a, E: Entity> SliceGroupMut<'a, E> {
        #[inline(always)]
        pub fn new(slice: E::Group<&'a mut [E::Unit]>) -> Self {
            Self(unsafe { transmute_unchecked(slice) }, PhantomData)
        }

        #[inline(always)]
        pub fn into_inner(self) -> E::Group<&'a mut [E::Unit]> {
            unsafe { transmute_unchecked(self.0) }
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for SliceGroup<'a, E> {
        type Target = SliceGroup<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for SliceGroup<'a, E> {
        type Target = SliceGroup<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            *self
        }
    }

    impl<'short, 'a, E: Entity> ReborrowMut<'short> for SliceGroupMut<'a, E> {
        type Target = SliceGroupMut<'short, E>;

        #[inline(always)]
        fn rb_mut(&'short mut self) -> Self::Target {
            SliceGroupMut::new(E::faer_map(
                E::faer_as_mut(&mut self.0),
                #[inline(always)]
                |this| &mut **this,
            ))
        }
    }

    impl<'short, 'a, E: Entity> Reborrow<'short> for SliceGroupMut<'a, E> {
        type Target = SliceGroup<'short, E>;

        #[inline(always)]
        fn rb(&'short self) -> Self::Target {
            SliceGroup::new(E::faer_map(
                E::faer_as_ref(&self.0),
                #[inline(always)]
                |this| &**this,
            ))
        }
    }
}

impl<E: Entity> core::fmt::Debug for RefGroup<'_, E> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.read().fmt(f)
    }
}
impl<E: Entity> core::fmt::Debug for RefGroupMut<'_, E> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.read().fmt(f)
    }
}
impl<E: Entity> core::fmt::Debug for SliceGroup<'_, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.into_iter()).finish()
    }
}
impl<E: Entity> core::fmt::Debug for SliceGroupMut<'_, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.rb().fmt(f)
    }
}

impl<'a, E: Entity> RefGroup<'a, E> {
    #[inline(always)]
    pub fn read(&self) -> E {
        E::faer_from_units(E::faer_deref(self.into_inner()))
    }
}

impl<'a, E: Entity> RefGroupMut<'a, E> {
    #[inline(always)]
    pub fn read(&self) -> E {
        self.rb().read()
    }

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

    #[inline(always)]
    #[track_caller]
    pub fn read(&self, idx: usize) -> E {
        assert!(idx < self.len());
        unsafe { self.read_unchecked(idx) }
    }

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

    #[inline(always)]
    #[track_caller]
    pub fn subslice(self, range: Range<usize>) -> Self {
        assert!(range.start <= range.end);
        assert!(range.end <= self.len());
        unsafe { self.subslice_unchecked(range) }
    }

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

    #[inline(always)]
    #[track_caller]
    pub unsafe fn subslice_unchecked(self, range: Range<usize>) -> Self {
        debug_assert!(range.start <= range.end);
        debug_assert!(range.end <= self.len());
        Self::new(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| slice.get_unchecked(range.start..range.end),
        ))
    }

    #[inline(always)]
    pub fn into_iter(self) -> impl Iterator<Item = RefGroup<'a, E>> {
        E::faer_into_iter(self.into_inner()).map(RefGroup::new)
    }

    #[inline(always)]
    pub fn into_chunks(self, chunk_size: usize) -> (impl Iterator<Item = SliceGroup<'a, E>>, Self) {
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
    #[inline]
    pub fn len(&self) -> usize {
        self.rb().len()
    }

    #[inline(always)]
    #[track_caller]
    pub fn read(&self, idx: usize) -> E {
        self.rb().read(idx)
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn read_unchecked(&self, idx: usize) -> E {
        self.rb().read_unchecked(idx)
    }

    #[inline(always)]
    #[track_caller]
    pub fn write(&mut self, idx: usize, value: E) {
        assert!(idx < self.len());
        unsafe { self.write_unchecked(idx, value) }
    }

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

    #[inline(always)]
    pub fn fill_zero(&mut self) {
        E::faer_map(
            self.rb_mut().into_inner(),
            #[inline(always)]
            |slice| mem::fill_zero(slice),
        );
    }

    #[inline(always)]
    #[track_caller]
    pub fn subslice(self, range: Range<usize>) -> Self {
        assert!(range.start <= range.end);
        assert!(range.end <= self.len());
        unsafe { self.subslice_unchecked(range) }
    }

    #[inline(always)]
    #[track_caller]
    pub unsafe fn subslice_unchecked(self, range: Range<usize>) -> Self {
        debug_assert!(range.start <= range.end);
        debug_assert!(range.end <= self.len());
        Self::new(E::faer_map(
            self.into_inner(),
            #[inline(always)]
            |slice| slice.get_unchecked_mut(range.start..range.end),
        ))
    }

    #[inline(always)]
    pub fn into_iter(self) -> impl Iterator<Item = RefGroupMut<'a, E>> {
        E::faer_into_iter(self.into_inner()).map(RefGroupMut::new)
    }

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

    #[inline(always)]
    pub fn into_chunks(
        self,
        chunk_size: usize,
    ) -> (impl Iterator<Item = SliceGroupMut<'a, E>>, Self) {
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

impl_copy!(<><I> <PermutationRef<'_, I>>);
impl_copy!(<><I> <SymbolicSparseColMatRef<'_, I>>);
impl_copy!(<><I, E: Entity> <SparseColMatRef<'_, I, E>>);

#[inline(always)]
fn windows2<I>(slice: &[I]) -> impl DoubleEndedIterator<Item = &[I; 2]> {
    slice
        .windows(2)
        .map(|window| unsafe { &*(window.as_ptr() as *const [I; 2]) })
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Side {
    Lower,
    Upper,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Symbolic;
impl Symbolic {
    #[inline(always)]
    fn materialize(n: usize) -> &'static mut [Self] {
        unsafe {
            core::slice::from_raw_parts_mut(core::ptr::NonNull::<Symbolic>::dangling().as_ptr(), n)
        }
    }
}
unsafe impl Zeroable for Symbolic {}
unsafe impl Pod for Symbolic {}
unsafe impl faer_core::Entity for Symbolic {
    type Unit = Symbolic;
    type Index = usize;
    type SimdUnit<S: pulp::Simd> = Symbolic;
    type SimdMask<S: pulp::Simd> = bool;
    type SimdIndex<S: pulp::Simd> = usize;
    type Group<T> = T;
    type GroupCopy<T: Copy> = T;
    type Iter<I: Iterator> = I;
    const N_COMPONENTS: usize = 1;
    const UNIT: Self::GroupCopy<()> = ();

    #[inline(always)]
    fn faer_from_units(group: Self::Group<Self::Unit>) -> Self {
        group
    }

    #[inline(always)]
    fn faer_into_units(self) -> Self::Group<Self::Unit> {
        self
    }

    #[inline(always)]
    fn faer_as_ref<T>(group: &Self::Group<T>) -> Self::Group<&T> {
        group
    }

    #[inline(always)]
    fn faer_as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T> {
        group
    }

    #[inline(always)]
    fn faer_map<T, U>(group: Self::Group<T>, mut f: impl FnMut(T) -> U) -> Self::Group<U> {
        f(group)
    }

    #[inline(always)]
    fn faer_zip<T, U>(first: Self::Group<T>, second: Self::Group<U>) -> Self::Group<(T, U)> {
        (first, second)
    }

    #[inline(always)]
    fn faer_unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>) {
        zipped
    }

    #[inline(always)]
    fn faer_map_with_context<Ctx, T, U>(
        ctx: Ctx,
        group: Self::Group<T>,
        f: &mut impl FnMut(Ctx, T) -> (Ctx, U),
    ) -> (Ctx, Self::Group<U>) {
        (*f)(ctx, group)
    }

    #[inline(always)]
    fn faer_into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter> {
        iter.into_iter()
    }
}
type SimdGroup<E, S> = <E as Entity>::Group<<E as Entity>::SimdUnit<S>>;

unsafe impl faer_core::Conjugate for Symbolic {
    type Conj = Symbolic;
    type Canonical = Symbolic;
    #[inline(always)]
    fn canonicalize(self) -> Self::Canonical {
        self
    }
}

impl RealField for Symbolic {
    #[inline(always)]
    fn faer_epsilon() -> Option<Self> {
        Some(Self)
    }

    #[inline(always)]
    fn faer_zero_threshold() -> Option<Self> {
        Some(Self)
    }

    #[inline(always)]
    fn faer_div(self, _rhs: Self) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_usize_to_index(a: usize) -> Self::Index {
        a
    }

    #[inline(always)]
    fn faer_index_to_usize(a: Self::Index) -> usize {
        a
    }

    #[inline(always)]
    fn faer_max_index() -> Self::Index {
        usize::MAX
    }

    #[inline(always)]
    fn faer_simd_less_than<S: pulp::Simd>(
        _simd: S,
        _a: SimdGroup<Self, S>,
        _b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        false
    }

    #[inline(always)]
    fn faer_simd_less_than_or_equal<S: pulp::Simd>(
        _simd: S,
        _a: SimdGroup<Self, S>,
        _b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        true
    }

    #[inline(always)]
    fn faer_simd_greater_than<S: pulp::Simd>(
        _simd: S,
        _a: SimdGroup<Self, S>,
        _b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        false
    }

    #[inline(always)]
    fn faer_simd_greater_than_or_equal<S: pulp::Simd>(
        _simd: S,
        _a: SimdGroup<Self, S>,
        _b: SimdGroup<Self, S>,
    ) -> Self::SimdMask<S> {
        true
    }

    #[inline(always)]
    fn faer_simd_select<S: pulp::Simd>(
        _simd: S,
        _mask: Self::SimdMask<S>,
        _if_true: SimdGroup<Self, S>,
        _if_false: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_index_select<S: pulp::Simd>(
        _simd: S,
        mask: Self::SimdMask<S>,
        if_true: Self::SimdIndex<S>,
        if_false: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        if mask {
            if_true
        } else {
            if_false
        }
    }

    #[inline(always)]
    fn faer_simd_index_seq<S: pulp::Simd>(_simd: S) -> Self::SimdIndex<S> {
        0
    }

    #[inline(always)]
    fn faer_simd_index_splat<S: pulp::Simd>(_simd: S, value: Self::Index) -> Self::SimdIndex<S> {
        value
    }

    #[inline(always)]
    fn faer_simd_index_add<S: pulp::Simd>(
        _simd: S,
        a: Self::SimdIndex<S>,
        b: Self::SimdIndex<S>,
    ) -> Self::SimdIndex<S> {
        a.wrapping_add(b)
    }
}

impl ComplexField for Symbolic {
    type Real = Symbolic;
    type Simd = faer_entity::NoSimd;
    type ScalarSimd = faer_entity::NoSimd;

    #[inline(always)]
    fn faer_from_f64(_value: f64) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_add(self, _rhs: Self) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_sub(self, _rhs: Self) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_mul(self, _rhs: Self) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_neg(self) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_inv(self) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_conj(self) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_sqrt(self) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_scale_real(self, _rhs: Self::Real) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_scale_power_of_two(self, _rhs: Self::Real) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_score(self) -> Self::Real {
        Self
    }

    #[inline(always)]
    fn faer_abs(self) -> Self::Real {
        Self
    }

    #[inline(always)]
    fn faer_abs2(self) -> Self::Real {
        Self
    }

    #[inline(always)]
    fn faer_nan() -> Self {
        Self
    }

    #[inline(always)]
    fn faer_from_real(_real: Self::Real) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_real(self) -> Self::Real {
        Self
    }

    #[inline(always)]
    fn faer_imag(self) -> Self::Real {
        Self
    }

    #[inline(always)]
    fn faer_zero() -> Self {
        Self
    }

    #[inline(always)]
    fn faer_one() -> Self {
        Self
    }

    #[inline(always)]
    fn faer_slice_as_simd<S: pulp::Simd>(
        slice: &[Self::Unit],
    ) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
        (slice, &[])
    }

    #[inline(always)]
    fn faer_slice_as_mut_simd<S: pulp::Simd>(
        slice: &mut [Self::Unit],
    ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
        (slice, &mut [])
    }

    #[inline(always)]
    fn faer_partial_load_unit<S: pulp::Simd>(_simd: S, _slice: &[Self::Unit]) -> Self::SimdUnit<S> {
        Self
    }

    #[inline(always)]
    fn faer_partial_store_unit<S: pulp::Simd>(
        _simd: S,
        _slice: &mut [Self::Unit],
        _values: Self::SimdUnit<S>,
    ) {
    }

    #[inline(always)]
    fn faer_partial_load_last_unit<S: pulp::Simd>(
        _simd: S,
        _slice: &[Self::Unit],
    ) -> Self::SimdUnit<S> {
        Self
    }

    #[inline(always)]
    fn faer_partial_store_last_unit<S: pulp::Simd>(
        _simd: S,
        _slice: &mut [Self::Unit],
        _values: Self::SimdUnit<S>,
    ) {
    }

    #[inline(always)]
    fn faer_simd_splat_unit<S: pulp::Simd>(_simd: S, _unit: Self::Unit) -> Self::SimdUnit<S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_scalar_mul<S: pulp::Simd>(_simd: S, _lhs: Self, _rhs: Self) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_simd_scalar_conj_mul<S: pulp::Simd>(_simd: S, _lhs: Self, _rhs: Self) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_simd_scalar_mul_adde<S: pulp::Simd>(
        _simd: S,
        _lhs: Self,
        _rhs: Self,
        _acc: Self,
    ) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_simd_scalar_conj_mul_adde<S: pulp::Simd>(
        _simd: S,
        _lhs: Self,
        _rhs: Self,
        _acc: Self,
    ) -> Self {
        Self
    }

    #[inline(always)]
    fn faer_simd_neg<S: pulp::Simd>(_simd: S, _values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_conj<S: pulp::Simd>(_simd: S, _values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_add<S: pulp::Simd>(
        _simd: S,
        _lhs: SimdGroup<Self, S>,
        _rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_sub<S: pulp::Simd>(
        _simd: S,
        _lhs: SimdGroup<Self, S>,
        _rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_mul<S: pulp::Simd>(
        _simd: S,
        _lhs: SimdGroup<Self, S>,
        _rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_scale_real<S: pulp::Simd>(
        _simd: S,
        _lhs: <Self::Real as Entity>::Group<<Self::Real as Entity>::SimdUnit<S>>,
        _rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_conj_mul<S: pulp::Simd>(
        _simd: S,
        _lhs: SimdGroup<Self, S>,
        _rhs: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_mul_adde<S: pulp::Simd>(
        _simd: S,
        _lhs: SimdGroup<Self, S>,
        _rhs: SimdGroup<Self, S>,
        _acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Self
    }

    #[inline(always)]
    fn simd_conj_mul_adde<S: pulp::Simd>(
        _simd: S,
        _lhs: SimdGroup<Self, S>,
        _rhs: SimdGroup<Self, S>,
        _acc: SimdGroup<Self, S>,
    ) -> SimdGroup<Self, S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_abs2_adde<S: pulp::Simd>(
        _simd: S,
        _values: SimdGroup<Self, S>,
        _acc: SimdGroup<Self::Real, S>,
    ) -> SimdGroup<Self::Real, S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_abs2<S: pulp::Simd>(
        _simd: S,
        _values: SimdGroup<Self, S>,
    ) -> SimdGroup<Self::Real, S> {
        Self
    }

    #[inline(always)]
    fn faer_simd_score<S: pulp::Simd>(
        _simd: S,
        _values: SimdGroup<Self, S>,
    ) -> SimdGroup<Self::Real, S> {
        Self
    }
}

fn ghost_permute_hermitian<'n, 'out, I: Index, E: ComplexField>(
    new_values: SliceGroupMut<'out, E>,
    new_col_ptrs: &'out mut [I],
    new_row_indices: &'out mut [I],
    A: ghost::SparseColMatRef<'n, 'n, '_, I, E>,
    perm: ghost::PermutationRef<'n, '_, I>,
    in_side: Side,
    out_side: Side,
    stack: PodStack<'_>,
) -> ghost::SparseColMatRef<'n, 'n, 'out, I, E> {
    let N = A.ncols();
    let n = *A.ncols();

    // (1)
    assert!(new_col_ptrs.len() == n + 1);
    let (_, perm_inv) = perm.fwd_inv();

    let (mut current_row_position, _) = stack.make_raw::<I>(n);
    let current_row_position = ghost::Array::from_mut(&mut current_row_position, N);

    mem::fill_zero(current_row_position);
    let col_counts = &mut *current_row_position;
    match (in_side, out_side) {
        (Side::Lower, Side::Lower) => {
            for old_j in N.indices() {
                let new_j = perm_inv[old_j].zx();
                for old_i in A.row_indices_of_col(old_j) {
                    if old_i >= old_j {
                        let new_i = perm_inv[old_i].zx();
                        let new_min = Ord::min(new_i, new_j);
                        // cannot overflow because A.compute_nnz() <= I::MAX
                        // col_counts[new_max] always >= 0
                        col_counts[new_min].incr();
                    }
                }
            }
        }
        (Side::Lower, Side::Upper) => {
            for old_j in N.indices() {
                let new_j = perm_inv[old_j].zx();
                for old_i in A.row_indices_of_col(old_j) {
                    if old_i >= old_j {
                        let new_i = perm_inv[old_i].zx();
                        let new_max = Ord::max(new_i, new_j);
                        // cannot overflow because A.compute_nnz() <= I::MAX
                        // col_counts[new_max] always >= 0
                        col_counts[new_max].incr();
                    }
                }
            }
        }
        (Side::Upper, Side::Lower) => {
            for old_j in N.indices() {
                let new_j = perm_inv[old_j].zx();
                for old_i in A.row_indices_of_col(old_j) {
                    if old_i <= old_j {
                        let new_i = perm_inv[old_i].zx();
                        let new_min = Ord::min(new_i, new_j);
                        // cannot overflow because A.compute_nnz() <= I::MAX
                        // col_counts[new_max] always >= 0
                        col_counts[new_min].incr();
                    }
                }
            }
        }
        (Side::Upper, Side::Upper) => {
            for old_j in N.indices() {
                let new_j = perm_inv[old_j].zx();
                for old_i in A.row_indices_of_col(old_j) {
                    if old_i <= old_j {
                        let new_i = perm_inv[old_i].zx();
                        let new_max = Ord::max(new_i, new_j);
                        // cannot overflow because A.compute_nnz() <= I::MAX
                        // col_counts[new_max] always >= 0
                        col_counts[new_max].incr();
                    }
                }
            }
        }
    }

    // col_counts[_] >= 0
    // cumulative sum cannot overflow because it is <= A.compute_nnz()

    // SAFETY: new_col_ptrs.len() == n + 1 > 0
    new_col_ptrs[0] = I::truncate(0);
    for (count, [ci0, ci1]) in zip(
        &mut **col_counts,
        windows2(Cell::as_slice_of_cells(Cell::from_mut(&mut *new_col_ptrs))),
    ) {
        let ci0 = ci0.get();
        ci1.set(ci0 + *count);
        *count = ci0;
    }
    // new_col_ptrs is non-decreasing

    let nnz = new_col_ptrs[n].zx();
    let new_row_indices = &mut new_row_indices[..nnz];
    let mut new_values = new_values.subslice(0..nnz);

    ghost::with_size(
        nnz,
        #[inline(always)]
        |NNZ| {
            let mut new_values = ghost::ArrayGroupMut::new(new_values.rb_mut(), NNZ);
            let new_row_indices = ghost::Array::from_mut(new_row_indices, NNZ);

            let conj_if = |cond: bool, x: E| {
                if !coe::is_same::<E, E::Real>() && cond {
                    x.faer_conj()
                } else {
                    x
                }
            };

            match (in_side, out_side) {
                (Side::Lower, Side::Lower) => {
                    for old_j in N.indices() {
                        let new_j_ = perm_inv[old_j];
                        let new_j = new_j_.zx();

                        for (old_i, val) in zip(
                            A.row_indices_of_col(old_j),
                            A.values_of_col(old_j).into_iter(),
                        ) {
                            if old_i >= old_j {
                                let new_i_ = perm_inv[old_i];
                                let new_i = new_i_.zx();

                                let new_max = Ord::max(new_i_, new_j_);
                                let new_min = Ord::min(new_i, new_j);
                                let current_row_pos = &mut current_row_position[new_min];
                                // SAFETY: current_row_pos < NNZ
                                let row_pos =
                                    unsafe { ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ) };
                                current_row_pos.incr();
                                new_values.write(row_pos, conj_if(new_min == new_i, val.read()));
                                // (2)
                                new_row_indices[row_pos] = *new_max;
                            }
                        }
                    }
                }
                (Side::Lower, Side::Upper) => {
                    for old_j in N.indices() {
                        let new_j_ = perm_inv[old_j];
                        let new_j = new_j_.zx();

                        for (old_i, val) in zip(
                            A.row_indices_of_col(old_j),
                            A.values_of_col(old_j).into_iter(),
                        ) {
                            if old_i >= old_j {
                                let new_i_ = perm_inv[old_i];
                                let new_i = new_i_.zx();

                                let new_max = Ord::max(new_i, new_j);
                                let new_min = Ord::min(new_i_, new_j_);
                                let current_row_pos = &mut current_row_position[new_max];
                                // SAFETY: current_row_pos < NNZ
                                let row_pos =
                                    unsafe { ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ) };
                                current_row_pos.incr();
                                new_values.write(row_pos, conj_if(new_max == new_i, val.read()));
                                // (2)
                                new_row_indices[row_pos] = *new_min;
                            }
                        }
                    }
                }
                (Side::Upper, Side::Lower) => {
                    for old_j in N.indices() {
                        let new_j_ = perm_inv[old_j];
                        let new_j = new_j_.zx();

                        for (old_i, val) in zip(
                            A.row_indices_of_col(old_j),
                            A.values_of_col(old_j).into_iter(),
                        ) {
                            if old_i <= old_j {
                                let new_i_ = perm_inv[old_i];
                                let new_i = new_i_.zx();

                                let new_max = Ord::max(new_i_, new_j_);
                                let new_min = Ord::min(new_i, new_j);
                                let current_row_pos = &mut current_row_position[new_min];
                                // SAFETY: current_row_pos < NNZ
                                let row_pos =
                                    unsafe { ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ) };
                                current_row_pos.incr();
                                new_values.write(row_pos, conj_if(new_min == new_i, val.read()));
                                // (2)
                                new_row_indices[row_pos] = *new_max;
                            }
                        }
                    }
                }
                (Side::Upper, Side::Upper) => {
                    for old_j in N.indices() {
                        let new_j_ = perm_inv[old_j];
                        let new_j = new_j_.zx();

                        for (old_i, val) in zip(
                            A.row_indices_of_col(old_j),
                            A.values_of_col(old_j).into_iter(),
                        ) {
                            if old_i <= old_j {
                                let new_i_ = perm_inv[old_i];
                                let new_i = new_i_.zx();

                                let new_max = Ord::max(new_i, new_j);
                                let new_min = Ord::min(new_i_, new_j_);
                                let current_row_pos = &mut current_row_position[new_max];
                                // SAFETY: current_row_pos < NNZ
                                let row_pos =
                                    unsafe { ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ) };
                                current_row_pos.incr();
                                new_values.write(row_pos, conj_if(new_max == new_i, val.read()));
                                // (2)
                                new_row_indices[row_pos] = *new_min;
                            }
                        }
                    }
                }
            }
            debug_assert!(**current_row_position == new_col_ptrs[1..]);
        },
    );
    // SAFETY:
    // 0. new_col_ptrs is non-decreasing (see ghost_permute_symmetric_common)
    // 1. new_values.len() == new_row_indices.len()
    // 2. all written row indices are less than n
    unsafe {
        ghost::SparseColMatRef::new(
            SparseColMatRef::new(
                SymbolicSparseColMatRef::new_unchecked(n, n, new_col_ptrs, None, new_row_indices),
                new_values.into_const(),
            ),
            N,
            N,
        )
    }
}

fn ghost_permute_hermitian_symbolic<'n, 'out, I: Index>(
    new_col_ptrs: &'out mut [I],
    new_row_indices: &'out mut [I],
    A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
    perm: ghost::PermutationRef<'n, '_, I>,
    in_side: Side,
    out_side: Side,
    stack: PodStack<'_>,
) -> ghost::SymbolicSparseColMatRef<'n, 'n, 'out, I> {
    let old_values = &*Symbolic::materialize(A.row_indices().len());
    let new_values = Symbolic::materialize(new_row_indices.len());
    ghost_permute_hermitian(
        SliceGroupMut::<'_, Symbolic>::new(new_values),
        new_col_ptrs,
        new_row_indices,
        ghost::SparseColMatRef::new(
            SparseColMatRef::new(*A, SliceGroup::new(old_values)),
            A.nrows(),
            A.ncols(),
        ),
        perm,
        in_side,
        out_side,
        stack,
    )
    .symbolic()
}

pub fn permute_hermitian<'out, I: Index, E: ComplexField>(
    new_values: SliceGroupMut<'out, E>,
    new_col_ptrs: &'out mut [I],
    new_row_indices: &'out mut [I],
    A: SparseColMatRef<'_, I, E>,
    perm: PermutationRef<'_, I>,
    in_side: Side,
    out_side: Side,
    stack: PodStack<'_>,
) -> SparseColMatRef<'out, I, E> {
    ghost::with_size(A.nrows(), |N| {
        assert!(A.nrows() == A.ncols());
        assert!(A.nrows() == A.ncols());
        *ghost_permute_hermitian(
            new_values,
            new_col_ptrs,
            new_row_indices,
            ghost::SparseColMatRef::new(A, N, N),
            ghost::PermutationRef::new(perm, N),
            in_side,
            out_side,
            stack,
        )
    })
}

fn ghost_adjoint_symbolic<'m, 'n, 'a, I: Index>(
    new_col_ptrs: &'a mut [I],
    new_row_indices: &'a mut [I],
    A: ghost::SymbolicSparseColMatRef<'m, 'n, '_, I>,
    stack: PodStack<'_>,
) -> ghost::SymbolicSparseColMatRef<'n, 'm, 'a, I> {
    let old_values = &*Symbolic::materialize(A.row_indices().len());
    let new_values = Symbolic::materialize(new_row_indices.len());
    ghost_adjoint(
        new_col_ptrs,
        new_row_indices,
        SliceGroupMut::<'_, Symbolic>::new(new_values),
        ghost::SparseColMatRef::new(
            SparseColMatRef::new(*A, SliceGroup::new(old_values)),
            A.nrows(),
            A.ncols(),
        ),
        stack,
    )
    .symbolic()
}

fn ghost_adjoint<'m, 'n, 'a, I: Index, E: ComplexField>(
    new_col_ptrs: &'a mut [I],
    new_row_indices: &'a mut [I],
    new_values: SliceGroupMut<'a, E>,
    A: ghost::SparseColMatRef<'m, 'n, '_, I, E>,
    stack: PodStack<'_>,
) -> ghost::SparseColMatRef<'n, 'm, 'a, I, E> {
    let M = A.nrows();
    let N = A.ncols();
    assert!(new_col_ptrs.len() == *M + 1);

    let (mut col_count, _) = stack.make_raw::<I>(*M);
    let col_count = ghost::Array::from_mut(&mut col_count, M);
    mem::fill_zero(col_count);

    // can't overflow because the total count is A.compute_nnz() <= I::MAX
    if A.nnz_per_col().is_some() {
        for j in N.indices() {
            for i in A.row_indices_of_col(j) {
                col_count[i].incr();
            }
        }
    } else {
        for i in A.symbolic().compressed_row_indices() {
            col_count[i].incr();
        }
    }

    // col_count elements are >= 0
    for (j, [pj0, pj1]) in zip(
        M.indices(),
        windows2(Cell::as_slice_of_cells(Cell::from_mut(new_col_ptrs))),
    ) {
        let cj = &mut col_count[j];
        let pj = pj0.get();
        // new_col_ptrs is non-decreasing
        pj1.set(pj + *cj);
        *cj = pj;
    }

    let new_row_indices = &mut new_row_indices[..new_col_ptrs[*M].zx()];
    let mut new_values = new_values.subslice(0..new_col_ptrs[*M].zx());
    let current_row_position = &mut *col_count;
    // current_row_position[i] == col_ptr[i]
    for j in N.indices() {
        let j_: ghost::Idx<'n, I> = j.truncate::<I>();
        for (i, val) in zip(A.row_indices_of_col(j), A.values_of_col(j).into_iter()) {
            let ci = &mut current_row_position[i];

            // SAFETY: see below
            unsafe {
                *new_row_indices.get_unchecked_mut(ci.zx()) = *j_;
                new_values.write_unchecked(ci.zx(), val.read().faer_conj())
            };
            ci.incr();
        }
    }
    // current_row_position[i] == col_ptr[i] + col_count[i] == col_ptr[i + 1] <= col_ptr[m]
    // so all the unchecked accesses were valid and non-overlapping, which means the entire
    // array is filled
    debug_assert!(&**current_row_position == &new_col_ptrs[1..]);

    // SAFETY:
    // 0. new_col_ptrs is non-decreasing (see ghost_permute_symmetric_common)
    // 1. all written row indices are less than n
    ghost::SparseColMatRef::new(
        unsafe {
            SparseColMatRef::new(
                SymbolicSparseColMatRef::new_unchecked(*N, *M, new_col_ptrs, None, new_row_indices),
                new_values.into_const(),
            )
        },
        N,
        M,
    )
}

pub fn adjoint<'a, I: Index, E: ComplexField>(
    new_col_ptrs: &'a mut [I],
    new_row_indices: &'a mut [I],
    new_values: SliceGroupMut<'a, E>,
    A: SparseColMatRef<'_, I, E>,
    stack: PodStack<'_>,
) -> SparseColMatRef<'a, I, E> {
    ghost::with_size(A.nrows(), |M| {
        ghost::with_size(A.ncols(), |N| {
            *ghost_adjoint(
                new_col_ptrs,
                new_row_indices,
                new_values,
                ghost::SparseColMatRef::new(A, M, N),
                stack,
            )
        })
    })
}

pub fn adjoint_symbolic<'a, I: Index>(
    new_col_ptrs: &'a mut [I],
    new_row_indices: &'a mut [I],
    A: SymbolicSparseColMatRef<'_, I>,
    stack: PodStack<'_>,
) -> SymbolicSparseColMatRef<'a, I> {
    ghost::with_size(A.nrows(), |M| {
        ghost::with_size(A.ncols(), |N| {
            *ghost_adjoint_symbolic(
                new_col_ptrs,
                new_row_indices,
                ghost::SymbolicSparseColMatRef::new(A, M, N),
                stack,
            )
        })
    })
}

#[cfg(test)]
pub(crate) mod qd {
    // https://web.mit.edu/tabbott/Public/quaddouble-debian/qd-2.3.4-old/docs/qd.pdf
    // https://gitlab.com/hodge_star/mantis

    use bytemuck::{Pod, Zeroable};
    use pulp::{Scalar, Simd};

    /// Value representing the implicit sum of two floating point terms, such that the absolute
    /// value of the second term is less half a ULP of the first term.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    #[repr(C)]
    pub struct Double<T>(pub T, pub T);

    unsafe impl<T: Zeroable> Zeroable for Double<T> {}
    unsafe impl<T: Pod> Pod for Double<T> {}

    impl<I: Iterator> Iterator for Double<I> {
        type Item = Double<I::Item>;

        #[inline(always)]
        fn next(&mut self) -> Option<Self::Item> {
            let x0 = self.0.next()?;
            let x1 = self.1.next()?;
            Some(Double(x0, x1))
        }
    }

    #[inline(always)]
    fn quick_two_sum<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
        let s = simd.f64s_add(a, b);
        let err = simd.f64s_sub(b, simd.f64s_sub(s, a));
        (s, err)
    }

    #[inline(always)]
    fn two_sum<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
        let s = simd.f64s_add(a, b);
        let bb = simd.f64s_sub(s, a);

        // (a - (s - bb)) + (b - bb)
        let err = simd.f64s_add(simd.f64s_sub(a, simd.f64s_sub(s, bb)), simd.f64s_sub(b, bb));
        (s, err)
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn quick_two_diff<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
        let s = simd.f64s_sub(a, b);
        let err = simd.f64s_sub(simd.f64s_sub(a, s), b);
        (s, err)
    }

    #[inline(always)]
    fn two_diff<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
        let s = simd.f64s_sub(a, b);
        let bb = simd.f64s_sub(s, a);

        // (a - (s - bb)) - (b + bb)
        let err = simd.f64s_sub(simd.f64s_sub(a, simd.f64s_sub(s, bb)), simd.f64s_add(b, bb));
        (s, err)
    }

    #[inline(always)]
    fn two_prod<S: Simd>(simd: S, a: S::f64s, b: S::f64s) -> (S::f64s, S::f64s) {
        let p = simd.f64s_mul(a, b);
        let err = simd.f64s_mul_add(a, b, simd.f64s_neg(p));

        (p, err)
    }

    pub mod double {
        use super::*;

        #[inline(always)]
        pub fn simd_add<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> Double<S::f64s> {
            let (s, e) = two_sum(simd, a.0, b.0);
            let e = simd.f64s_add(e, simd.f64s_add(a.1, b.1));
            let (s, e) = quick_two_sum(simd, s, e);
            Double(s, e)
        }

        #[inline(always)]
        pub fn simd_sub<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> Double<S::f64s> {
            let (s, e) = two_diff(simd, a.0, b.0);
            let e = simd.f64s_add(e, a.1);
            let e = simd.f64s_sub(e, b.1);
            let (s, e) = quick_two_sum(simd, s, e);
            Double(s, e)
        }

        #[inline(always)]
        pub fn simd_neg<S: Simd>(simd: S, a: Double<S::f64s>) -> Double<S::f64s> {
            Double(simd.f64s_neg(a.0), simd.f64s_neg(a.1))
        }

        #[inline(always)]
        pub fn simd_mul<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> Double<S::f64s> {
            let (p1, p2) = two_prod(simd, a.0, b.0);
            let p2 = simd.f64s_add(
                p2,
                simd.f64s_add(simd.f64s_mul(a.0, b.1), simd.f64s_mul(a.1, b.0)),
            );
            let (p1, p2) = quick_two_sum(simd, p1, p2);
            Double(p1, p2)
        }

        #[inline(always)]
        fn simd_mul_f64<S: Simd>(simd: S, a: Double<S::f64s>, b: S::f64s) -> Double<S::f64s> {
            let (p1, p2) = two_prod(simd, a.0, b);
            let p2 = simd.f64s_add(p2, simd.f64s_mul(a.1, b));
            let (p1, p2) = quick_two_sum(simd, p1, p2);
            Double(p1, p2)
        }

        pub fn simd_select<S: Simd>(
            simd: S,
            mask: S::m64s,
            if_true: Double<S::f64s>,
            if_false: Double<S::f64s>,
        ) -> Double<S::f64s> {
            Double(
                simd.m64s_select_f64s(mask, if_true.0, if_false.0),
                simd.m64s_select_f64s(mask, if_true.1, if_false.1),
            )
        }

        #[inline]
        pub fn simd_div<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> Double<S::f64s> {
            simd.vectorize(
                #[inline(always)]
                || {
                    let pos_zero = simd.f64s_splat(0.0);
                    let pos_infty = simd.f64s_splat(f64::INFINITY);
                    let sign_bit = simd.f64s_splat(-0.0);

                    let a_sign = simd.f64s_and(a.0, sign_bit);
                    let b_sign = simd.f64s_and(b.0, sign_bit);

                    let combined_sign = simd.f64s_xor(a_sign, b_sign);

                    let a_is_zero = simd_eq(simd, a, Double(pos_zero, pos_zero));
                    let b_is_zero = simd_eq(simd, b, Double(pos_zero, pos_zero));
                    let a_is_infty = simd_eq(
                        simd,
                        Double(simd.f64s_abs(a.0), simd.f64s_abs(a.1)),
                        Double(pos_infty, pos_infty),
                    );
                    let b_is_infty = simd_eq(
                        simd,
                        Double(simd.f64s_abs(b.0), simd.f64s_abs(b.1)),
                        Double(pos_infty, pos_infty),
                    );

                    let q1 = simd.f64s_div(a.0, b.0);
                    let r = simd_mul_f64(simd, b, q1);

                    let (s1, s2) = two_diff(simd, a.0, r.0);
                    let s2 = simd.f64s_sub(s2, r.1);
                    let s2 = simd.f64s_add(s2, a.1);

                    let q2 = simd.f64s_div(simd.f64s_add(s1, s2), b.0);
                    let (q0, q1) = quick_two_sum(simd, q1, q2);

                    simd_select(
                        simd,
                        simd.m64s_and(b_is_zero, simd.m64s_not(a_is_zero)),
                        Double(
                            simd.f64s_or(combined_sign, pos_infty),
                            simd.f64s_or(combined_sign, pos_infty),
                        ),
                        simd_select(
                            simd,
                            simd.m64s_and(b_is_infty, simd.m64s_not(a_is_infty)),
                            Double(
                                simd.f64s_or(combined_sign, pos_zero),
                                simd.f64s_or(combined_sign, pos_zero),
                            ),
                            Double(q0, q1),
                        ),
                    )
                },
            )
        }

        #[inline(always)]
        pub fn simd_abs<S: Simd>(simd: S, a: Double<S::f64s>) -> Double<S::f64s> {
            let is_negative = simd.f64s_less_than(a.0, simd.f64s_splat(0.0));
            Double(
                simd.m64s_select_f64s(is_negative, simd.f64s_neg(a.0), a.0),
                simd.m64s_select_f64s(is_negative, simd.f64s_neg(a.1), a.1),
            )
        }

        #[inline(always)]
        pub fn simd_less_than<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> S::m64s {
            let lt0 = simd.f64s_less_than(a.0, b.0);
            let eq0 = simd.f64s_equal(a.0, b.0);
            let lt1 = simd.f64s_less_than(a.1, b.1);
            simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
        }

        #[inline(always)]
        pub fn simd_less_than_or_equal<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> S::m64s {
            let lt0 = simd.f64s_less_than(a.0, b.0);
            let eq0 = simd.f64s_equal(a.0, b.0);
            let lt1 = simd.f64s_less_than_or_equal(a.1, b.1);
            simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
        }

        #[inline(always)]
        pub fn simd_greater_than<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> S::m64s {
            let lt0 = simd.f64s_greater_than(a.0, b.0);
            let eq0 = simd.f64s_equal(a.0, b.0);
            let lt1 = simd.f64s_greater_than(a.1, b.1);
            simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
        }

        #[inline(always)]
        pub fn simd_greater_than_or_equal<S: Simd>(
            simd: S,
            a: Double<S::f64s>,
            b: Double<S::f64s>,
        ) -> S::m64s {
            let lt0 = simd.f64s_greater_than(a.0, b.0);
            let eq0 = simd.f64s_equal(a.0, b.0);
            let lt1 = simd.f64s_greater_than_or_equal(a.1, b.1);
            simd.m64s_or(lt0, simd.m64s_and(eq0, lt1))
        }

        #[inline(always)]
        pub fn simd_eq<S: Simd>(simd: S, a: Double<S::f64s>, b: Double<S::f64s>) -> S::m64s {
            let eq0 = simd.f64s_equal(a.0, b.0);
            let eq1 = simd.f64s_equal(a.1, b.1);
            simd.m64s_and(eq0, eq1)
        }
    }

    impl core::ops::Add for Double<f64> {
        type Output = Self;

        #[inline(always)]
        fn add(self, rhs: Self) -> Self::Output {
            double::simd_add(Scalar::new(), self, rhs)
        }
    }

    impl core::ops::Sub for Double<f64> {
        type Output = Self;

        #[inline(always)]
        fn sub(self, rhs: Self) -> Self::Output {
            double::simd_sub(Scalar::new(), self, rhs)
        }
    }

    impl core::ops::Mul for Double<f64> {
        type Output = Self;

        #[inline(always)]
        fn mul(self, rhs: Self) -> Self::Output {
            double::simd_mul(Scalar::new(), self, rhs)
        }
    }

    impl core::ops::Div for Double<f64> {
        type Output = Self;

        #[inline(always)]
        fn div(self, rhs: Self) -> Self::Output {
            double::simd_div(Scalar::new(), self, rhs)
        }
    }

    impl core::ops::AddAssign for Double<f64> {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl core::ops::SubAssign for Double<f64> {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl core::ops::MulAssign for Double<f64> {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl core::ops::DivAssign for Double<f64> {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    impl core::ops::Neg for Double<f64> {
        type Output = Self;

        #[inline(always)]
        fn neg(self) -> Self::Output {
            Self(-self.0, -self.1)
        }
    }

    impl Double<f64> {
        /// 2.0^{-100}
        pub const EPSILON: Self = Self(7.888609052210118e-31, 0.0);
        /// 2.0^{-970}: precision below this value begins to degrade.
        pub const MIN_POSITIVE: Self = Self(1.0020841800044864e-292, 0.0);

        pub const ZERO: Self = Self(0.0, 0.0);
        pub const NAN: Self = Self(f64::NAN, f64::NAN);
        pub const INFINITY: Self = Self(f64::INFINITY, f64::INFINITY);

        #[inline(always)]
        pub fn abs(self) -> Self {
            double::simd_abs(Scalar::new(), self)
        }

        #[inline(always)]
        pub fn recip(self) -> Self {
            double::simd_div(Scalar::new(), Self(1.0, 0.0), self)
        }

        #[inline]
        pub fn sqrt(self) -> Self {
            if self == Self::ZERO {
                Self::ZERO
            } else if self < Self::ZERO {
                Self::NAN
            } else if self == Self::INFINITY {
                Self::INFINITY
            } else {
                let a = self;
                let x = a.0.sqrt().recip();
                let ax = Self(a.0 * x, 0.0);

                ax + (a - ax * ax) * Double(x * 0.5, 0.0)
            }
        }
    }

    mod faer_impl {
        use super::*;
        use faer_core::{ComplexField, Conjugate, Entity, RealField};

        type SimdGroup<E, S> = <E as Entity>::Group<<E as Entity>::SimdUnit<S>>;

        unsafe impl Entity for Double<f64> {
            type Unit = f64;
            type Index = u64;

            type SimdUnit<S: Simd> = S::f64s;
            type SimdMask<S: Simd> = S::m64s;
            type SimdIndex<S: Simd> = S::u64s;

            type Group<T> = Double<T>;
            type GroupCopy<T: Copy> = Double<T>;
            type Iter<I: Iterator> = Double<I>;

            const N_COMPONENTS: usize = 2;
            const UNIT: Self::GroupCopy<()> = Double((), ());

            #[inline(always)]
            fn faer_from_units(group: Self::Group<Self::Unit>) -> Self {
                group
            }

            #[inline(always)]
            fn faer_into_units(self) -> Self::Group<Self::Unit> {
                self
            }

            #[inline(always)]
            fn faer_as_ref<T>(group: &Self::Group<T>) -> Self::Group<&T> {
                Double(&group.0, &group.1)
            }

            #[inline(always)]
            fn faer_as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T> {
                Double(&mut group.0, &mut group.1)
            }

            #[inline(always)]
            fn faer_map<T, U>(group: Self::Group<T>, mut f: impl FnMut(T) -> U) -> Self::Group<U> {
                Double(f(group.0), f(group.1))
            }

            #[inline(always)]
            fn faer_zip<T, U>(
                first: Self::Group<T>,
                second: Self::Group<U>,
            ) -> Self::Group<(T, U)> {
                Double((first.0, second.0), (first.1, second.1))
            }

            #[inline(always)]
            fn faer_unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>) {
                (
                    Double(zipped.0 .0, zipped.1 .0),
                    Double(zipped.0 .1, zipped.1 .1),
                )
            }

            #[inline(always)]
            fn faer_map_with_context<Ctx, T, U>(
                ctx: Ctx,
                group: Self::Group<T>,
                f: &mut impl FnMut(Ctx, T) -> (Ctx, U),
            ) -> (Ctx, Self::Group<U>) {
                let (ctx, x0) = (*f)(ctx, group.0);
                let (ctx, x1) = (*f)(ctx, group.1);
                (ctx, Double(x0, x1))
            }

            #[inline(always)]
            fn faer_into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter> {
                Double(iter.0.into_iter(), iter.1.into_iter())
            }
        }

        unsafe impl Conjugate for Double<f64> {
            type Conj = Double<f64>;
            type Canonical = Double<f64>;
            #[inline(always)]
            fn canonicalize(self) -> Self::Canonical {
                self
            }
        }

        impl RealField for Double<f64> {
            #[inline(always)]
            fn faer_epsilon() -> Option<Self> {
                Some(Self::EPSILON)
            }
            #[inline(always)]
            fn faer_zero_threshold() -> Option<Self> {
                Some(Self::MIN_POSITIVE)
            }

            #[inline(always)]
            fn faer_div(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn faer_usize_to_index(a: usize) -> Self::Index {
                a as _
            }

            #[inline(always)]
            fn faer_index_to_usize(a: Self::Index) -> usize {
                a as _
            }

            #[inline(always)]
            fn faer_max_index() -> Self::Index {
                Self::Index::MAX
            }

            #[inline(always)]
            fn faer_simd_less_than<S: Simd>(
                simd: S,
                a: SimdGroup<Self, S>,
                b: SimdGroup<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_less_than(simd, a, b)
            }

            #[inline(always)]
            fn faer_simd_less_than_or_equal<S: Simd>(
                simd: S,
                a: SimdGroup<Self, S>,
                b: SimdGroup<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_less_than_or_equal(simd, a, b)
            }

            #[inline(always)]
            fn faer_simd_greater_than<S: Simd>(
                simd: S,
                a: SimdGroup<Self, S>,
                b: SimdGroup<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_greater_than(simd, a, b)
            }

            #[inline(always)]
            fn faer_simd_greater_than_or_equal<S: Simd>(
                simd: S,
                a: SimdGroup<Self, S>,
                b: SimdGroup<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_greater_than_or_equal(simd, a, b)
            }

            #[inline(always)]
            fn faer_simd_select<S: Simd>(
                simd: S,
                mask: Self::SimdMask<S>,
                if_true: SimdGroup<Self, S>,
                if_false: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_select(simd, mask, if_true, if_false)
            }

            #[inline(always)]
            fn faer_simd_index_select<S: Simd>(
                simd: S,
                mask: Self::SimdMask<S>,
                if_true: Self::SimdIndex<S>,
                if_false: Self::SimdIndex<S>,
            ) -> Self::SimdIndex<S> {
                simd.m64s_select_u64s(mask, if_true, if_false)
            }

            #[inline(always)]
            fn faer_simd_index_seq<S: Simd>(simd: S) -> Self::SimdIndex<S> {
                let _ = simd;
                pulp::cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u64])
            }

            #[inline(always)]
            fn faer_simd_index_splat<S: Simd>(simd: S, value: Self::Index) -> Self::SimdIndex<S> {
                simd.u64s_splat(value)
            }

            #[inline(always)]
            fn faer_simd_index_add<S: Simd>(
                simd: S,
                a: Self::SimdIndex<S>,
                b: Self::SimdIndex<S>,
            ) -> Self::SimdIndex<S> {
                simd.u64s_add(a, b)
            }
        }

        impl ComplexField for Double<f64> {
            type Real = Double<f64>;
            type Simd = pulp::Arch;
            type ScalarSimd = pulp::Arch;

            #[inline(always)]
            fn faer_sqrt(self) -> Self {
                self.sqrt()
            }

            #[inline(always)]
            fn faer_from_f64(value: f64) -> Self {
                Self(value, 0.0)
            }

            #[inline(always)]
            fn faer_add(self, rhs: Self) -> Self {
                self + rhs
            }

            #[inline(always)]
            fn faer_sub(self, rhs: Self) -> Self {
                self - rhs
            }

            #[inline(always)]
            fn faer_mul(self, rhs: Self) -> Self {
                self * rhs
            }

            #[inline(always)]
            fn faer_neg(self) -> Self {
                -self
            }

            #[inline(always)]
            fn faer_inv(self) -> Self {
                self.recip()
            }

            #[inline(always)]
            fn faer_conj(self) -> Self {
                self
            }

            #[inline(always)]
            fn faer_scale_real(self, rhs: Self::Real) -> Self {
                self * rhs
            }

            #[inline(always)]
            fn faer_scale_power_of_two(self, rhs: Self::Real) -> Self {
                Self(self.0 * rhs.0, self.1 * rhs.0)
            }

            #[inline(always)]
            fn faer_score(self) -> Self::Real {
                self.abs()
            }

            #[inline(always)]
            fn faer_abs(self) -> Self::Real {
                self.abs()
            }

            #[inline(always)]
            fn faer_abs2(self) -> Self::Real {
                self * self
            }

            #[inline(always)]
            fn faer_nan() -> Self {
                Self::NAN
            }

            #[inline(always)]
            fn faer_from_real(real: Self::Real) -> Self {
                real
            }

            #[inline(always)]
            fn faer_real(self) -> Self::Real {
                self
            }

            #[inline(always)]
            fn faer_imag(self) -> Self::Real {
                Self::ZERO
            }

            #[inline(always)]
            fn faer_zero() -> Self {
                Self::ZERO
            }

            #[inline(always)]
            fn faer_one() -> Self {
                Self(1.0, 0.0)
            }

            #[inline(always)]
            fn faer_slice_as_simd<S: Simd>(
                slice: &[Self::Unit],
            ) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
                S::f64s_as_simd(slice)
            }

            #[inline(always)]
            fn faer_slice_as_mut_simd<S: Simd>(
                slice: &mut [Self::Unit],
            ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
                S::f64s_as_mut_simd(slice)
            }

            #[inline(always)]
            fn faer_partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
                simd.f64s_partial_load(slice)
            }

            #[inline(always)]
            fn faer_partial_store_unit<S: Simd>(
                simd: S,
                slice: &mut [Self::Unit],
                values: Self::SimdUnit<S>,
            ) {
                simd.f64s_partial_store(slice, values)
            }

            #[inline(always)]
            fn faer_partial_load_last_unit<S: Simd>(
                simd: S,
                slice: &[Self::Unit],
            ) -> Self::SimdUnit<S> {
                simd.f64s_partial_load_last(slice)
            }

            #[inline(always)]
            fn faer_partial_store_last_unit<S: Simd>(
                simd: S,
                slice: &mut [Self::Unit],
                values: Self::SimdUnit<S>,
            ) {
                simd.f64s_partial_store_last(slice, values)
            }

            #[inline(always)]
            fn faer_simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
                simd.f64s_splat(unit)
            }

            #[inline(always)]
            fn faer_simd_neg<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
                double::simd_neg(simd, values)
            }

            #[inline(always)]
            fn faer_simd_conj<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
                let _ = simd;
                values
            }

            #[inline(always)]
            fn faer_simd_add<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_add(simd, lhs, rhs)
            }

            #[inline(always)]
            fn faer_simd_sub<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_sub(simd, lhs, rhs)
            }

            #[inline(always)]
            fn faer_simd_mul<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_mul(simd, lhs, rhs)
            }

            #[inline(always)]
            fn faer_simd_scale_real<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_mul(simd, lhs, rhs)
            }

            #[inline(always)]
            fn faer_simd_conj_mul<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_mul(simd, lhs, rhs)
            }

            #[inline(always)]
            fn faer_simd_mul_adde<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
                acc: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_add(simd, acc, double::simd_mul(simd, lhs, rhs))
            }

            #[inline(always)]
            fn simd_conj_mul_adde<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
                acc: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_add(simd, acc, double::simd_mul(simd, lhs, rhs))
            }

            #[inline(always)]
            fn faer_simd_score<S: Simd>(
                simd: S,
                values: SimdGroup<Self, S>,
            ) -> SimdGroup<Self::Real, S> {
                double::simd_abs(simd, values)
            }

            #[inline(always)]
            fn faer_simd_abs2_adde<S: Simd>(
                simd: S,
                values: SimdGroup<Self, S>,
                acc: SimdGroup<Self::Real, S>,
            ) -> SimdGroup<Self::Real, S> {
                Self::faer_simd_add(simd, acc, Self::faer_simd_mul(simd, values, values))
            }

            #[inline(always)]
            fn faer_simd_abs2<S: Simd>(
                simd: S,
                values: SimdGroup<Self, S>,
            ) -> SimdGroup<Self::Real, S> {
                Self::faer_simd_mul(simd, values, values)
            }

            #[inline(always)]
            fn faer_simd_scalar_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
                let _ = simd;
                lhs * rhs
            }

            #[inline(always)]
            fn faer_simd_scalar_conj_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
                let _ = simd;
                lhs * rhs
            }

            #[inline(always)]
            fn faer_simd_scalar_mul_adde<S: Simd>(
                simd: S,
                lhs: Self,
                rhs: Self,
                acc: Self,
            ) -> Self {
                let _ = simd;
                lhs * rhs + acc
            }

            #[inline(always)]
            fn faer_simd_scalar_conj_mul_adde<S: Simd>(
                simd: S,
                lhs: Self,
                rhs: Self,
                acc: Self,
            ) -> Self {
                let _ = simd;
                lhs * rhs + acc
            }
        }
    }
}
