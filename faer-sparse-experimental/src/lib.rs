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
    pub struct SparseColMatRef<'a, I, E: Entity> {
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
    impl<'a, I: Index, E: Entity> SparseColMatRef<'a, I, E> {
        #[inline]
        #[track_caller]
        pub fn new(
            symbolic: SymbolicSparseColMatRef<'a, I>,
            values: E::Group<&'a [E::Unit]>,
        ) -> Self {
            E::map(
                E::copy(&values),
                #[inline(always)]
                |values| {
                    assert!(symbolic.row_indices().len() == values.len());
                },
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
impl_copy!(<><I, E: Entity> <SparseColMatRef<'_, I, E>>);

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
            fn from_units(group: Self::Group<Self::Unit>) -> Self {
                group
            }

            #[inline(always)]
            fn into_units(self) -> Self::Group<Self::Unit> {
                self
            }

            #[inline(always)]
            fn as_ref<T>(group: &Self::Group<T>) -> Self::Group<&T> {
                Double(&group.0, &group.1)
            }

            #[inline(always)]
            fn as_mut<T>(group: &mut Self::Group<T>) -> Self::Group<&mut T> {
                Double(&mut group.0, &mut group.1)
            }

            #[inline(always)]
            fn map<T, U>(group: Self::Group<T>, mut f: impl FnMut(T) -> U) -> Self::Group<U> {
                Double(f(group.0), f(group.1))
            }

            #[inline(always)]
            fn zip<T, U>(first: Self::Group<T>, second: Self::Group<U>) -> Self::Group<(T, U)> {
                Double((first.0, second.0), (first.1, second.1))
            }

            #[inline(always)]
            fn unzip<T, U>(zipped: Self::Group<(T, U)>) -> (Self::Group<T>, Self::Group<U>) {
                (
                    Double(zipped.0 .0, zipped.1 .0),
                    Double(zipped.0 .1, zipped.1 .1),
                )
            }

            #[inline(always)]
            fn map_with_context<Ctx, T, U>(
                ctx: Ctx,
                group: Self::Group<T>,
                mut f: impl FnMut(Ctx, T) -> (Ctx, U),
            ) -> (Ctx, Self::Group<U>) {
                let (ctx, x0) = f(ctx, group.0);
                let (ctx, x1) = f(ctx, group.1);
                (ctx, Double(x0, x1))
            }

            #[inline(always)]
            fn into_iter<I: IntoIterator>(iter: Self::Group<I>) -> Self::Iter<I::IntoIter> {
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
            fn epsilon() -> Option<Self> {
                Some(Self::EPSILON)
            }
            #[inline(always)]
            fn zero_threshold() -> Option<Self> {
                Some(Self::MIN_POSITIVE)
            }

            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn usize_to_index(a: usize) -> Self::Index {
                a as _
            }

            #[inline(always)]
            fn index_to_usize(a: Self::Index) -> usize {
                a as _
            }

            #[inline(always)]
            fn max_index() -> Self::Index {
                Self::Index::MAX
            }

            #[inline(always)]
            fn simd_less_than<S: Simd>(
                simd: S,
                a: SimdGroup<Self, S>,
                b: SimdGroup<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_less_than(simd, a, b)
            }

            #[inline(always)]
            fn simd_less_than_or_equal<S: Simd>(
                simd: S,
                a: SimdGroup<Self, S>,
                b: SimdGroup<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_less_than_or_equal(simd, a, b)
            }

            #[inline(always)]
            fn simd_greater_than<S: Simd>(
                simd: S,
                a: SimdGroup<Self, S>,
                b: SimdGroup<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_greater_than(simd, a, b)
            }

            #[inline(always)]
            fn simd_greater_than_or_equal<S: Simd>(
                simd: S,
                a: SimdGroup<Self, S>,
                b: SimdGroup<Self, S>,
            ) -> Self::SimdMask<S> {
                double::simd_greater_than_or_equal(simd, a, b)
            }

            #[inline(always)]
            fn simd_select<S: Simd>(
                simd: S,
                mask: Self::SimdMask<S>,
                if_true: SimdGroup<Self, S>,
                if_false: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_select(simd, mask, if_true, if_false)
            }

            #[inline(always)]
            fn simd_index_select<S: Simd>(
                simd: S,
                mask: Self::SimdMask<S>,
                if_true: Self::SimdIndex<S>,
                if_false: Self::SimdIndex<S>,
            ) -> Self::SimdIndex<S> {
                simd.m64s_select_u64s(mask, if_true, if_false)
            }

            #[inline(always)]
            fn simd_index_seq<S: Simd>(simd: S) -> Self::SimdIndex<S> {
                let _ = simd;
                pulp::cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u64])
            }

            #[inline(always)]
            fn simd_index_splat<S: Simd>(simd: S, value: Self::Index) -> Self::SimdIndex<S> {
                simd.u64s_splat(value)
            }

            #[inline(always)]
            fn simd_index_add<S: Simd>(
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
            fn sqrt(self) -> Self {
                self.sqrt()
            }

            #[inline(always)]
            fn from_f64(value: f64) -> Self {
                Self(value, 0.0)
            }

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                self + rhs
            }

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                self - rhs
            }

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                self * rhs
            }

            #[inline(always)]
            fn neg(self) -> Self {
                -self
            }

            #[inline(always)]
            fn inv(self) -> Self {
                self.recip()
            }

            #[inline(always)]
            fn conj(self) -> Self {
                self
            }

            #[inline(always)]
            fn scale_real(self, rhs: Self::Real) -> Self {
                self * rhs
            }

            #[inline(always)]
            fn scale_power_of_two(self, rhs: Self::Real) -> Self {
                Self(self.0 * rhs.0, self.1 * rhs.0)
            }

            #[inline(always)]
            fn score(self) -> Self::Real {
                self.abs()
            }

            #[inline(always)]
            fn abs(self) -> Self::Real {
                self.abs()
            }

            #[inline(always)]
            fn abs2(self) -> Self::Real {
                self * self
            }

            #[inline(always)]
            fn nan() -> Self {
                Self::NAN
            }

            #[inline(always)]
            fn from_real(real: Self::Real) -> Self {
                real
            }

            #[inline(always)]
            fn real(self) -> Self::Real {
                self
            }

            #[inline(always)]
            fn imag(self) -> Self::Real {
                Self::ZERO
            }

            #[inline(always)]
            fn zero() -> Self {
                Self::ZERO
            }

            #[inline(always)]
            fn one() -> Self {
                Self(1.0, 0.0)
            }

            #[inline(always)]
            fn slice_as_simd<S: Simd>(
                slice: &[Self::Unit],
            ) -> (&[Self::SimdUnit<S>], &[Self::Unit]) {
                S::f64s_as_simd(slice)
            }

            #[inline(always)]
            fn slice_as_mut_simd<S: Simd>(
                slice: &mut [Self::Unit],
            ) -> (&mut [Self::SimdUnit<S>], &mut [Self::Unit]) {
                S::f64s_as_mut_simd(slice)
            }

            #[inline(always)]
            fn partial_load_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
                simd.f64s_partial_load(slice)
            }

            #[inline(always)]
            fn partial_store_unit<S: Simd>(
                simd: S,
                slice: &mut [Self::Unit],
                values: Self::SimdUnit<S>,
            ) {
                simd.f64s_partial_store(slice, values)
            }

            #[inline(always)]
            fn partial_load_last_unit<S: Simd>(simd: S, slice: &[Self::Unit]) -> Self::SimdUnit<S> {
                simd.f64s_partial_load_last(slice)
            }

            #[inline(always)]
            fn partial_store_last_unit<S: Simd>(
                simd: S,
                slice: &mut [Self::Unit],
                values: Self::SimdUnit<S>,
            ) {
                simd.f64s_partial_store_last(slice, values)
            }

            #[inline(always)]
            fn simd_splat_unit<S: Simd>(simd: S, unit: Self::Unit) -> Self::SimdUnit<S> {
                simd.f64s_splat(unit)
            }

            #[inline(always)]
            fn simd_neg<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
                double::simd_neg(simd, values)
            }

            #[inline(always)]
            fn simd_conj<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self, S> {
                let _ = simd;
                values
            }

            #[inline(always)]
            fn simd_add<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_add(simd, lhs, rhs)
            }

            #[inline(always)]
            fn simd_sub<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_sub(simd, lhs, rhs)
            }

            #[inline(always)]
            fn simd_mul<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_mul(simd, lhs, rhs)
            }

            #[inline(always)]
            fn simd_scale_real<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_mul(simd, lhs, rhs)
            }

            #[inline(always)]
            fn simd_conj_mul<S: Simd>(
                simd: S,
                lhs: SimdGroup<Self, S>,
                rhs: SimdGroup<Self, S>,
            ) -> SimdGroup<Self, S> {
                double::simd_mul(simd, lhs, rhs)
            }

            #[inline(always)]
            fn simd_mul_adde<S: Simd>(
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
            fn simd_score<S: Simd>(
                simd: S,
                values: SimdGroup<Self, S>,
            ) -> SimdGroup<Self::Real, S> {
                double::simd_abs(simd, values)
            }

            #[inline(always)]
            fn simd_abs2_adde<S: Simd>(
                simd: S,
                values: SimdGroup<Self, S>,
                acc: SimdGroup<Self::Real, S>,
            ) -> SimdGroup<Self::Real, S> {
                Self::simd_add(simd, acc, Self::simd_mul(simd, values, values))
            }

            #[inline(always)]
            fn simd_abs2<S: Simd>(simd: S, values: SimdGroup<Self, S>) -> SimdGroup<Self::Real, S> {
                Self::simd_mul(simd, values, values)
            }

            #[inline(always)]
            fn simd_scalar_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
                let _ = simd;
                lhs * rhs
            }

            #[inline(always)]
            fn simd_scalar_conj_mul<S: Simd>(simd: S, lhs: Self, rhs: Self) -> Self {
                let _ = simd;
                lhs * rhs
            }

            #[inline(always)]
            fn simd_scalar_mul_adde<S: Simd>(simd: S, lhs: Self, rhs: Self, acc: Self) -> Self {
                let _ = simd;
                lhs * rhs + acc
            }

            #[inline(always)]
            fn simd_scalar_conj_mul_adde<S: Simd>(
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
