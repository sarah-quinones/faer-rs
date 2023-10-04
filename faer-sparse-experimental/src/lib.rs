#![forbid(elided_lifetimes_in_paths)]
#![allow(non_snake_case)]

use crate::mem::{__get_checked, __get_unchecked};
pub use __core::*;
use bytemuck::Pod;
use core::{iter::zip, ops::Range};
use dyn_stack::{SizeOverflow, StackReq};
use faer_core::{ComplexField, Entity};
use mem::NONE;

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

pub trait PodEntity: Entity + Pod {}
impl<E: Entity + Pod> PodEntity for E {}

pub trait PodComplexField: ComplexField + Pod {}
impl<E: ComplexField + Pod> PodComplexField for E {}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[non_exhaustive]
pub enum FaerSparseError {
    IndexOverflow,
    OutOfMemory,
}

#[inline]
#[track_caller]
fn try_zeroed<I: Pod>(n: usize) -> Result<Vec<I>, FaerSparseError> {
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
fn try_collect<I: IntoIterator>(iter: I) -> Result<Vec<I::Item>, FaerSparseError>
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
fn nomem<T>(_: T) -> FaerSparseError {
    FaerSparseError::OutOfMemory
}

pub mod cholesky;
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
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + Copy
    + Pod
    + Clone
    + Eq
    + Ord
    + Send
    + Sync
    + 'static
    + core::fmt::Debug
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
    pub struct SparseColMatRef<'a, I, E: PodEntity> {
        symbolic: SymbolicSparseColMatRef<'a, I>,
        val: E::GroupCopy<&'a [E::Unit]>,
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
        pub fn fwd_inv(self) -> (&'a [I], &'a [I]) {
            (self.fwd, self.inv)
        }
    }

    /// Requires:
    /// * nrows <= I::MAX (always checked)
    /// * ncols <= I::MAX (always checked)
    /// * col_ptrs has length n (always checked)
    /// * col_ptrs is non-decreasing
    /// * col_ptrs[0]..col_ptrs[n] is a valid range in row_indices (always checked, assuming
    ///   non-decreasing)
    /// * if nnz_per_col is None, elements of row_indices[col_ptrs[j]..col_ptrs[j + 1]] are less
    ///   than nrows
    ///
    /// * nnz_per_col[j] <= col_ptrs[j+1] - col_ptrs[j]
    /// * if nnz_per_col is Some(_), elements of row_indices[col_ptrs[j]..][..nnz_per_col[j]] are
    ///   less than nrows
    ///
    /// Ensures:
    /// * self.compute_nnz() is <= I::MAX
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

    /// Requires:
    /// All the requirements of [`SymbolicSparseColMatRef`], and `values` must have the same length
    /// as `row_indices`
    ///
    /// Ensures:
    /// * self.compute_nnz() is <= I::MAX
    impl<'a, I: Index, E: PodEntity> SparseColMatRef<'a, I, E> {
        #[inline]
        #[track_caller]
        pub fn new_checked(
            nrows: usize,
            ncols: usize,
            col_ptrs: &'a [I],
            nnz_per_col: Option<&'a [I]>,
            row_indices: &'a [I],
            values: E::Group<&'a [E::Unit]>,
        ) -> Self {
            E::map(
                E::copy(&values),
                #[inline(always)]
                |values| {
                    assert!(row_indices.len() == values.len());
                },
            );
            let symbolic = SymbolicSparseColMatRef::new_checked(
                nrows,
                ncols,
                col_ptrs,
                nnz_per_col,
                row_indices,
            );

            Self {
                symbolic,
                val: E::into_copy(values),
            }
        }

        #[inline]
        #[track_caller]
        pub unsafe fn new_unchecked(
            nrows: usize,
            ncols: usize,
            col_ptrs: &'a [I],
            nnz_per_col: Option<&'a [I]>,
            row_indices: &'a [I],
            values: E::Group<&'a [E::Unit]>,
        ) -> Self {
            E::map(
                E::copy(&values),
                #[inline(always)]
                |values| {
                    debug_assert!(row_indices.len() == values.len());
                },
            );
            let symbolic = SymbolicSparseColMatRef::new_unchecked(
                nrows,
                ncols,
                col_ptrs,
                nnz_per_col,
                row_indices,
            );

            Self {
                symbolic,
                val: E::into_copy(values),
            }
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
        pub fn values(&self) -> E::Group<&'a [E::Unit]> {
            E::from_copy(self.val)
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
        pub fn values_of_col(&self, j: usize) -> E::Group<&'a [E::Unit]> {
            let range = self.col_range(j);
            E::map(
                E::from_copy(self.val),
                #[inline(always)]
                |val| crate::mem::__get_checked(val, range.clone()),
            )
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
}

impl_copy!(<><I> <PermutationRef<'_, I>>);
impl_copy!(<><I> <SymbolicSparseColMatRef<'_, I>>);
impl_copy!(<><I, E: PodEntity> <SparseColMatRef<'_, I, E>>);

#[inline(always)]
pub fn windows2<I>(slice: &[I]) -> impl DoubleEndedIterator<Item = &[I; 2]> {
    slice
        .windows(2)
        .map(|window| unsafe { &*(window.as_ptr() as *const [I; 2]) })
}

/// FIXME: fix try_any_of upstream
#[inline]
fn __try_any_of(reqs: impl IntoIterator<Item = StackReq>) -> Result<StackReq, SizeOverflow> {
    fn try_any_of_impl(mut reqs: impl Iterator<Item = StackReq>) -> Result<StackReq, SizeOverflow> {
        let mut total = StackReq::empty();
        while let Some(req) = reqs.next() {
            total = total.try_or(req)?;
        }
        Ok(total)
    }
    try_any_of_impl(reqs.into_iter())
}
