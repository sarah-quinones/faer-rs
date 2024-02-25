//! Sparse matrix data structures.
//!
//! Most sparse matrix algorithms accept matrices in sparse column-oriented format.
//! This format represents each column of the matrix by storing the row indices of its non-zero
//! elements, as well as their values.
//!
//! The indices and the values are each stored in a contiguous slice (or group of slices for
//! arbitrary values). In order to specify where each column starts and ends, a slice of size
//! `ncols + 1` stores the start of each column, with the last element being equal to the total
//! number of non-zeros (or the capacity in uncompressed mode).
//!
//! # Example
//!
//! Consider the 4-by-5 matrix:
//! ```notcode
//! [[10.0, 0.0, 12.0, -1.0, 13.0]
//!  [ 0.0, 0.0, 25.0, -2.0,  0.0]
//!  [ 1.0, 0.0,  0.0,  0.0,  0.0]
//!  [ 4.0, 0.0,  0.0,  0.0,  5.0]]
//! ```
//!
//! The matrix is stored as follows:
//! ```notcode
//! column pointers:  0 |  3 |  3 |  5 |  7 |  9
//!
//! row indices:    0 |    2 |    3 |    0 |    1 |    0 |    1 |    0 |    3
//! values     : 10.0 |  1.0 |  4.0 | 12.0 | 25.0 | -1.0 | -2.0 | 13.0 |  5.0
//! ```

use super::*;
use crate::utils::{slice::*, vec::VecGroup};
use core::{cell::Cell, iter::zip, ops::Range, slice::SliceIndex};
use dyn_stack::*;
use faer_entity::*;
use reborrow::*;

mod ghost {
    pub use crate::utils::constrained::{perm::*, sparse::*, *};
}

const TOP_BIT: usize = 1usize << (usize::BITS - 1);
const TOP_BIT_MASK: usize = TOP_BIT - 1;

mod mem {
    #[inline]
    pub fn fill_zero<I: bytemuck::Zeroable>(slice: &mut [I]) {
        let len = slice.len();
        unsafe { core::ptr::write_bytes(slice.as_mut_ptr(), 0u8, len) }
    }
}

#[inline(always)]
#[track_caller]
#[doc(hidden)]
pub unsafe fn __get_unchecked<I, R: Clone + SliceIndex<[I]>>(slice: &[I], i: R) -> &R::Output {
    #[cfg(debug_assertions)]
    {
        let _ = &slice[i.clone()];
    }
    unsafe { slice.get_unchecked(i) }
}
#[inline(always)]
#[track_caller]
#[doc(hidden)]
pub unsafe fn __get_unchecked_mut<I, R: Clone + SliceIndex<[I]>>(
    slice: &mut [I],
    i: R,
) -> &mut R::Output {
    #[cfg(debug_assertions)]
    {
        let _ = &slice[i.clone()];
    }
    unsafe { slice.get_unchecked_mut(i) }
}

#[inline(always)]
#[doc(hidden)]
pub fn windows2<I>(slice: &[I]) -> impl DoubleEndedIterator<Item = &[I; 2]> {
    slice
        .windows(2)
        .map(|window| unsafe { &*(window.as_ptr() as *const [I; 2]) })
}

#[inline]
#[doc(hidden)]
pub const fn repeat_byte(byte: u8) -> usize {
    union Union {
        bytes: [u8; 32],
        value: usize,
    }

    let data = Union { bytes: [byte; 32] };
    unsafe { data.value }
}

/// Errors that can occur in sparse algorithms.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[non_exhaustive]
pub enum FaerError {
    /// An index exceeding the maximum value (`I::Signed::MAX` for a given index type `I`).
    IndexOverflow,
    /// Memory allocation failed.
    OutOfMemory,
}

impl core::fmt::Display for FaerError {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FaerError {}

/// Errors that can occur in sparse algorithms.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[non_exhaustive]
pub enum CreationError {
    /// Generic error (allocation or index overflow).
    Generic(FaerError),
    /// Matrix index out-of-bounds error.
    OutOfBounds {
        /// Row of the out-of-bounds index.
        row: usize,
        /// Column of the out-of-bounds index.
        col: usize,
    },
}

impl From<FaerError> for CreationError {
    #[inline]
    fn from(value: FaerError) -> Self {
        Self::Generic(value)
    }
}
impl core::fmt::Display for CreationError {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CreationError {}

#[inline]
#[track_caller]
fn try_zeroed<I: bytemuck::Pod>(n: usize) -> Result<alloc::vec::Vec<I>, FaerError> {
    let mut v = alloc::vec::Vec::new();
    v.try_reserve_exact(n).map_err(|_| FaerError::OutOfMemory)?;
    unsafe {
        core::ptr::write_bytes::<I>(v.as_mut_ptr(), 0u8, n);
        v.set_len(n);
    }
    Ok(v)
}

#[inline]
#[track_caller]
fn try_collect<I: IntoIterator>(iter: I) -> Result<alloc::vec::Vec<I::Item>, FaerError> {
    let iter = iter.into_iter();
    let mut v = alloc::vec::Vec::new();
    v.try_reserve_exact(iter.size_hint().0)
        .map_err(|_| FaerError::OutOfMemory)?;
    v.extend(iter);
    Ok(v)
}

/// The order values should be read in, when constructing/filling from indices and values.
///
/// Allows separately creating the symbolic structure and filling the numerical values.
#[derive(Debug, Clone)]
pub struct ValuesOrder<I> {
    argsort: Vec<usize>,
    all_nnz: usize,
    nnz: usize,
    __marker: core::marker::PhantomData<I>,
}

/// Whether the filled values should replace the current matrix values or be added to them.
#[derive(Debug, Copy, Clone)]
pub enum FillMode {
    /// New filled values should replace the old values.
    Replace,
    /// New filled values should be added to the old values.
    Add,
}

mod csc;
mod csr;

/// Sparse linear algebra module.  
/// Contains low level routines and the implementation of their corresponding high level wrappers.
pub mod linalg;

/// Sparse matrix binary and ternary operation implementations.
pub mod ops;

pub use csc::*;
pub use csr::*;

/// Useful sparse matrix primitives.
pub mod utils {
    use super::*;
    use crate::{assert, debug_assert};

    /// Sorts `row_indices` and `values` simultaneously so that `row_indices` is nonincreasing.
    pub fn sort_indices<I: Index, E: Entity>(
        col_ptrs: &[I],
        row_indices: &mut [I],
        values: GroupFor<E, &mut [E::Unit]>,
    ) {
        assert!(col_ptrs.len() >= 1);
        let mut values = SliceGroupMut::<'_, E>::new(values);

        let n = col_ptrs.len() - 1;
        for j in 0..n {
            let start = col_ptrs[j].zx();
            let end = col_ptrs[j + 1].zx();

            unsafe {
                crate::sort::sort_indices(
                    &mut row_indices[start..end],
                    values.rb_mut().subslice(start..end),
                );
            }
        }
    }

    /// Sorts and deduplicates `row_indices` and `values` simultaneously so that `row_indices` is
    /// nonincreasing and contains no duplicate indices.
    pub fn sort_dedup_indices<I: Index, E: ComplexField>(
        col_ptrs: &[I],
        nnz_per_col: &mut [I],
        row_indices: &mut [I],
        values: GroupFor<E, &mut [E::Unit]>,
    ) {
        assert!(col_ptrs.len() >= 1);
        let mut values = SliceGroupMut::<'_, E>::new(values);

        let n = col_ptrs.len() - 1;
        for j in 0..n {
            let start = col_ptrs[j].zx();
            let end = start + nnz_per_col[j].zx();

            unsafe {
                crate::sort::sort_indices(
                    &mut row_indices[start..end],
                    values.rb_mut().subslice(start..end),
                );
            }

            let mut prev = I::truncate(usize::MAX);

            let mut writer = start;
            let mut reader = start;
            while reader < end {
                if row_indices[reader] == prev {
                    let writer = writer - 1;
                    values.write(writer, values.read(writer).faer_add(values.read(reader)));
                } else {
                    values.write(writer, values.read(reader));
                    writer += 1;
                }

                prev = row_indices[reader];
                reader += 1;
            }

            nnz_per_col[j] = I::truncate(writer - start);
        }
    }

    #[doc(hidden)]
    pub unsafe fn ghost_permute_hermitian_unsorted<'n, 'out, I: Index, E: ComplexField>(
        new_values: SliceGroupMut<'out, E>,
        new_col_ptrs: &'out mut [I],
        new_row_indices: &'out mut [I],
        A: ghost::SparseColMatRef<'n, 'n, '_, I, E>,
        perm: ghost::PermRef<'n, '_, I>,
        in_side: Side,
        out_side: Side,
        sort: bool,
        stack: PodStack<'_>,
    ) -> ghost::SparseColMatMut<'n, 'n, 'out, I, E> {
        let N = A.ncols();
        let n = *A.ncols();

        // (1)
        assert!(new_col_ptrs.len() == n + 1);
        let (_, perm_inv) = perm.arrays();

        let (current_row_position, _) = stack.make_raw::<I>(n);
        let current_row_position = ghost::Array::from_mut(current_row_position, N);

        mem::fill_zero(current_row_position.as_mut());
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
                            col_counts[new_min] += I::truncate(1);
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
                            col_counts[new_max] += I::truncate(1);
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
                            col_counts[new_min] += I::truncate(1);
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
                            col_counts[new_max] += I::truncate(1);
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
            col_counts.as_mut(),
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

        ghost::Size::with(
            nnz,
            #[inline(always)]
            |NNZ| {
                let mut new_values =
                    ghost::ArrayGroupMut::new(new_values.rb_mut().into_inner(), NNZ);
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
                                SliceGroup::<'_, E>::new(A.values_of_col(old_j)).into_ref_iter(),
                            ) {
                                if old_i >= old_j {
                                    let new_i_ = perm_inv[old_i];
                                    let new_i = new_i_.zx();

                                    let new_max = Ord::max(new_i_, new_j_);
                                    let new_min = Ord::min(new_i, new_j);
                                    let current_row_pos: &mut I =
                                        &mut current_row_position[new_min];
                                    // SAFETY: current_row_pos < NNZ
                                    let row_pos = unsafe {
                                        ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ)
                                    };
                                    *current_row_pos += I::truncate(1);
                                    new_values
                                        .write(row_pos, conj_if(new_min == new_i, val.read()));
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
                                SliceGroup::<'_, E>::new(A.values_of_col(old_j)).into_ref_iter(),
                            ) {
                                if old_i >= old_j {
                                    let new_i_ = perm_inv[old_i];
                                    let new_i = new_i_.zx();

                                    let new_max = Ord::max(new_i, new_j);
                                    let new_min = Ord::min(new_i_, new_j_);
                                    let current_row_pos = &mut current_row_position[new_max];
                                    // SAFETY: current_row_pos < NNZ
                                    let row_pos = unsafe {
                                        ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ)
                                    };
                                    *current_row_pos += I::truncate(1);
                                    new_values
                                        .write(row_pos, conj_if(new_max == new_i, val.read()));
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
                                SliceGroup::<'_, E>::new(A.values_of_col(old_j)).into_ref_iter(),
                            ) {
                                if old_i <= old_j {
                                    let new_i_ = perm_inv[old_i];
                                    let new_i = new_i_.zx();

                                    let new_max = Ord::max(new_i_, new_j_);
                                    let new_min = Ord::min(new_i, new_j);
                                    let current_row_pos = &mut current_row_position[new_min];
                                    // SAFETY: current_row_pos < NNZ
                                    let row_pos = unsafe {
                                        ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ)
                                    };
                                    *current_row_pos += I::truncate(1);
                                    new_values
                                        .write(row_pos, conj_if(new_min == new_i, val.read()));
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
                                SliceGroup::<'_, E>::new(A.values_of_col(old_j)).into_ref_iter(),
                            ) {
                                if old_i <= old_j {
                                    let new_i_ = perm_inv[old_i];
                                    let new_i = new_i_.zx();

                                    let new_max = Ord::max(new_i, new_j);
                                    let new_min = Ord::min(new_i_, new_j_);
                                    let current_row_pos = &mut current_row_position[new_max];
                                    // SAFETY: current_row_pos < NNZ
                                    let row_pos = unsafe {
                                        ghost::Idx::new_unchecked(current_row_pos.zx(), NNZ)
                                    };
                                    *current_row_pos += I::truncate(1);
                                    new_values
                                        .write(row_pos, conj_if(new_max == new_i, val.read()));
                                    // (2)
                                    new_row_indices[row_pos] = *new_min;
                                }
                            }
                        }
                    }
                }
                debug_assert!(current_row_position.as_ref() == &new_col_ptrs[1..]);
            },
        );

        if sort {
            sort_indices::<I, E>(
                new_col_ptrs,
                new_row_indices,
                new_values.rb_mut().into_inner(),
            );
        }

        // SAFETY:
        // 0. new_col_ptrs is non-decreasing
        // 1. new_values.len() == new_row_indices.len()
        // 2. all written row indices are less than n
        unsafe {
            ghost::SparseColMatMut::new(
                SparseColMatMut::new(
                    SymbolicSparseColMatRef::new_unchecked(
                        n,
                        n,
                        new_col_ptrs,
                        None,
                        new_row_indices,
                    ),
                    new_values.into_inner(),
                ),
                N,
                N,
            )
        }
    }

    #[doc(hidden)]
    pub unsafe fn ghost_permute_hermitian_unsorted_symbolic<'n, 'out, I: Index>(
        new_col_ptrs: &'out mut [I],
        new_row_indices: &'out mut [I],
        A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
        perm: ghost::PermRef<'n, '_, I>,
        in_side: Side,
        out_side: Side,
        stack: PodStack<'_>,
    ) -> ghost::SymbolicSparseColMatRef<'n, 'n, 'out, I> {
        let old_values = &*Symbolic::materialize(A.into_inner().row_indices().len());
        let new_values = Symbolic::materialize(new_row_indices.len());
        *ghost_permute_hermitian_unsorted(
            SliceGroupMut::<'_, Symbolic>::new(new_values),
            new_col_ptrs,
            new_row_indices,
            ghost::SparseColMatRef::new(
                SparseColMatRef::new(A.into_inner(), old_values),
                A.nrows(),
                A.ncols(),
            ),
            perm,
            in_side,
            out_side,
            false,
            stack,
        )
    }

    /// Computes the self-adjoint permutation $P A P^\top$ of the matrix `A` without sorting the row
    /// indices, and returns a view over it.
    ///
    /// The result is stored in `new_col_ptrs`, `new_row_indices`.
    #[doc(hidden)]
    pub unsafe fn permute_hermitian_unsorted<'out, I: Index, E: ComplexField>(
        new_values: GroupFor<E, &'out mut [E::Unit]>,
        new_col_ptrs: &'out mut [I],
        new_row_indices: &'out mut [I],
        A: SparseColMatRef<'_, I, E>,
        perm: crate::perm::PermRef<'_, I>,
        in_side: Side,
        out_side: Side,
        stack: PodStack<'_>,
    ) -> SparseColMatMut<'out, I, E> {
        ghost::Size::with(A.nrows(), |N| {
            assert!(A.nrows() == A.ncols());
            ghost_permute_hermitian_unsorted(
                SliceGroupMut::new(new_values),
                new_col_ptrs,
                new_row_indices,
                ghost::SparseColMatRef::new(A, N, N),
                ghost::PermRef::new(perm, N),
                in_side,
                out_side,
                false,
                stack,
            )
            .into_inner()
        })
    }

    /// Computes the self-adjoint permutation $P A P^\top$ of the matrix `A` and returns a view over
    /// it.
    ///
    /// The result is stored in `new_col_ptrs`, `new_row_indices`.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    pub fn permute_hermitian<'out, I: Index, E: ComplexField>(
        new_values: GroupFor<E, &'out mut [E::Unit]>,
        new_col_ptrs: &'out mut [I],
        new_row_indices: &'out mut [I],
        A: SparseColMatRef<'_, I, E>,
        perm: crate::perm::PermRef<'_, I>,
        in_side: Side,
        out_side: Side,
        stack: PodStack<'_>,
    ) -> SparseColMatMut<'out, I, E> {
        ghost::Size::with(A.nrows(), |N| {
            assert!(A.nrows() == A.ncols());
            unsafe {
                ghost_permute_hermitian_unsorted(
                    SliceGroupMut::new(new_values),
                    new_col_ptrs,
                    new_row_indices,
                    ghost::SparseColMatRef::new(A, N, N),
                    ghost::PermRef::new(perm, N),
                    in_side,
                    out_side,
                    true,
                    stack,
                )
            }
            .into_inner()
        })
    }

    #[doc(hidden)]
    pub fn ghost_adjoint_symbolic<'m, 'n, 'a, I: Index>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        A: ghost::SymbolicSparseColMatRef<'m, 'n, '_, I>,
        stack: PodStack<'_>,
    ) -> ghost::SymbolicSparseColMatRef<'n, 'm, 'a, I> {
        let old_values = &*Symbolic::materialize(A.into_inner().row_indices().len());
        let new_values = Symbolic::materialize(new_row_indices.len());
        *ghost_adjoint(
            new_col_ptrs,
            new_row_indices,
            SliceGroupMut::<'_, Symbolic>::new(new_values),
            ghost::SparseColMatRef::new(
                SparseColMatRef::new(A.into_inner(), old_values),
                A.nrows(),
                A.ncols(),
            ),
            stack,
        )
    }

    #[doc(hidden)]
    pub fn ghost_adjoint<'m, 'n, 'a, I: Index, E: ComplexField>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        new_values: SliceGroupMut<'a, E>,
        A: ghost::SparseColMatRef<'m, 'n, '_, I, E>,
        stack: PodStack<'_>,
    ) -> ghost::SparseColMatMut<'n, 'm, 'a, I, E> {
        let M = A.nrows();
        let N = A.ncols();
        assert!(new_col_ptrs.len() == *M + 1);

        let (col_count, _) = stack.make_raw::<I>(*M);
        let col_count = ghost::Array::from_mut(col_count, M);
        mem::fill_zero(col_count.as_mut());

        // can't overflow because the total count is A.compute_nnz() <= I::MAX
        for j in N.indices() {
            for i in A.row_indices_of_col(j) {
                col_count[i] += I::truncate(1);
            }
        }

        new_col_ptrs[0] = I::truncate(0);
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
            for (i, val) in zip(
                A.row_indices_of_col(j),
                SliceGroup::<'_, E>::new(A.values_of_col(j)).into_ref_iter(),
            ) {
                let ci = &mut current_row_position[i];

                // SAFETY: see below
                unsafe {
                    *new_row_indices.get_unchecked_mut(ci.zx()) = *j_;
                    new_values.write_unchecked(ci.zx(), val.read().faer_conj())
                };
                *ci += I::truncate(1);
            }
        }
        // current_row_position[i] == col_ptr[i] + col_count[i] == col_ptr[i + 1] <= col_ptr[m]
        // so all the unchecked accesses were valid and non-overlapping, which means the entire
        // array is filled
        debug_assert!(current_row_position.as_ref() == &new_col_ptrs[1..]);

        // SAFETY:
        // 0. new_col_ptrs is non-decreasing
        // 1. all written row indices are less than n
        ghost::SparseColMatMut::new(
            unsafe {
                SparseColMatMut::new(
                    SymbolicSparseColMatRef::new_unchecked(
                        *N,
                        *M,
                        new_col_ptrs,
                        None,
                        new_row_indices,
                    ),
                    new_values.into_inner(),
                )
            },
            N,
            M,
        )
    }

    #[doc(hidden)]
    pub fn ghost_transpose<'m, 'n, 'a, I: Index, E: Entity>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        new_values: SliceGroupMut<'a, E>,
        A: ghost::SparseColMatRef<'m, 'n, '_, I, E>,
        stack: PodStack<'_>,
    ) -> ghost::SparseColMatMut<'n, 'm, 'a, I, E> {
        let M = A.nrows();
        let N = A.ncols();
        assert!(new_col_ptrs.len() == *M + 1);

        let (col_count, _) = stack.make_raw::<I>(*M);
        let col_count = ghost::Array::from_mut(col_count, M);
        mem::fill_zero(col_count.as_mut());

        // can't overflow because the total count is A.compute_nnz() <= I::MAX
        for j in N.indices() {
            for i in A.row_indices_of_col(j) {
                col_count[i] += I::truncate(1);
            }
        }

        new_col_ptrs[0] = I::truncate(0);
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
            for (i, val) in zip(
                A.row_indices_of_col(j),
                SliceGroup::<'_, E>::new(A.values_of_col(j)).into_ref_iter(),
            ) {
                let ci = &mut current_row_position[i];

                // SAFETY: see below
                unsafe {
                    *new_row_indices.get_unchecked_mut(ci.zx()) = *j_;
                    new_values.write_unchecked(ci.zx(), val.read())
                };
                *ci += I::truncate(1);
            }
        }
        // current_row_position[i] == col_ptr[i] + col_count[i] == col_ptr[i + 1] <= col_ptr[m]
        // so all the unchecked accesses were valid and non-overlapping, which means the entire
        // array is filled
        debug_assert!(current_row_position.as_ref() == &new_col_ptrs[1..]);

        // SAFETY:
        // 0. new_col_ptrs is non-decreasing
        // 1. all written row indices are less than n
        ghost::SparseColMatMut::new(
            unsafe {
                SparseColMatMut::new(
                    SymbolicSparseColMatRef::new_unchecked(
                        *N,
                        *M,
                        new_col_ptrs,
                        None,
                        new_row_indices,
                    ),
                    new_values.into_inner(),
                )
            },
            N,
            M,
        )
    }

    /// Computes the transpose of the matrix `A` and returns a view over it.
    ///
    /// The result is stored in `new_col_ptrs`, `new_row_indices` and `new_values`.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    pub fn transpose<'a, I: Index, E: Entity>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        new_values: GroupFor<E, &'a mut [E::Unit]>,
        A: SparseColMatRef<'_, I, E>,
        stack: PodStack<'_>,
    ) -> SparseColMatMut<'a, I, E> {
        ghost::Size::with2(A.nrows(), A.ncols(), |M, N| {
            ghost_transpose(
                new_col_ptrs,
                new_row_indices,
                SliceGroupMut::new(new_values),
                ghost::SparseColMatRef::new(A, M, N),
                stack,
            )
            .into_inner()
        })
    }

    /// Computes the adjoint of the matrix `A` and returns a view over it.
    ///
    /// The result is stored in `new_col_ptrs`, `new_row_indices` and `new_values`.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    pub fn adjoint<'a, I: Index, E: ComplexField>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        new_values: GroupFor<E, &'a mut [E::Unit]>,
        A: SparseColMatRef<'_, I, E>,
        stack: PodStack<'_>,
    ) -> SparseColMatMut<'a, I, E> {
        ghost::Size::with2(A.nrows(), A.ncols(), |M, N| {
            ghost_adjoint(
                new_col_ptrs,
                new_row_indices,
                SliceGroupMut::new(new_values),
                ghost::SparseColMatRef::new(A, M, N),
                stack,
            )
            .into_inner()
        })
    }

    /// Computes the adjoint of the symbolic matrix `A` and returns a view over it.
    ///
    /// The result is stored in `new_col_ptrs`, `new_row_indices`.
    ///
    /// # Note
    /// Allows unsorted matrices, producing a sorted output. Duplicate entries are kept, however.
    pub fn adjoint_symbolic<'a, I: Index>(
        new_col_ptrs: &'a mut [I],
        new_row_indices: &'a mut [I],
        A: SymbolicSparseColMatRef<'_, I>,
        stack: PodStack<'_>,
    ) -> SymbolicSparseColMatRef<'a, I> {
        ghost::Size::with2(A.nrows(), A.ncols(), |M, N| {
            ghost_adjoint_symbolic(
                new_col_ptrs,
                new_row_indices,
                ghost::SymbolicSparseColMatRef::new(A, M, N),
                stack,
            )
            .into_inner()
        })
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseColMatRef<'_, I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseRowMatRef<'_, I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseColMatMut<'_, I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.rb().get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseRowMatMut<'_, I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.rb().get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for SparseColMatMut<'_, I, E> {
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.rb_mut().get_mut(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for SparseRowMatMut<'_, I, E> {
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.rb_mut().get_mut(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseColMat<I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.as_ref().get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::Index<(usize, usize)> for SparseRowMat<I, E> {
    type Output = E;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.as_ref().get(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for SparseColMat<I, E> {
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.as_mut().get_mut(row, col).unwrap()
    }
}

impl<I: Index, E: SimpleEntity> core::ops::IndexMut<(usize, usize)> for SparseRowMat<I, E> {
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.as_mut().get_mut(row, col).unwrap()
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseColMatRef<'_, I, E> {
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
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseColMatRef<'_, I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        let mut triplets = Vec::new();
        for j in 0..self.ncols() {
            for (i, val) in self
                .row_indices_of_col(j)
                .zip(SliceGroup::<'_, E>::new(self.values_of_col(j)).into_ref_iter())
            {
                triplets.push((i, j, val.read()))
            }
        }
        triplets
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseRowMatRef<'_, I, E> {
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
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseRowMatRef<'_, I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        let mut triplets = Vec::new();
        for i in 0..self.nrows() {
            for (j, val) in self
                .col_indices_of_row(i)
                .zip(SliceGroup::<'_, E>::new(self.values_of_row(i)).into_ref_iter())
            {
                triplets.push((i, j, val.read()))
            }
        }
        triplets
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseColMatMut<'_, I, E> {
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
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseColMatMut<'_, I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        self.rb().fetch_triplets()
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseColMat<I, E> {
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
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseColMat<I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        self.as_ref().fetch_triplets()
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseRowMatMut<'_, I, E> {
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
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseRowMatMut<'_, I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        self.rb().fetch_triplets()
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::Matrix<E> for SparseRowMat<I, E> {
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
        matrixcompare_core::Access::Sparse(self)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<I: Index, E: Entity> matrixcompare_core::SparseAccess<E> for SparseRowMat<I, E> {
    #[inline]
    fn nnz(&self) -> usize {
        self.compute_nnz()
    }

    #[inline]
    fn fetch_triplets(&self) -> Vec<(usize, usize, E)> {
        self.as_ref().fetch_triplets()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert;

    #[test]
    fn test_from_indices() {
        let nrows = 5;
        let ncols = 4;

        let indices = &[(0, 0), (1, 2), (0, 0), (1, 1), (0, 1), (3, 3), (3, 3usize)];
        let values = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0f64];

        let triplets = &[
            (0, 0, 1.0),
            (1, 2, 2.0),
            (0, 0, 3.0),
            (1, 1, 4.0),
            (0, 1, 5.0),
            (3, 3, 6.0),
            (3, 3usize, 7.0),
        ];

        {
            let mat = SymbolicSparseColMat::try_new_from_indices(nrows, ncols, indices);
            assert!(mat.is_ok());

            let (mat, order) = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.col_ptrs() == &[0, 1, 3, 4, 5]);
            assert!(mat.nnz_per_col() == None);
            assert!(mat.row_indices() == &[0, 0, 1, 1, 3]);

            let mat =
                SparseColMat::<_, f64>::new_from_order_and_values(mat, &order, values).unwrap();
            assert!(mat.as_ref().values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }

        {
            let mat = SparseColMat::try_new_from_triplets(nrows, ncols, triplets);
            assert!(mat.is_ok());
            let mat = mat.unwrap();

            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.col_ptrs() == &[0, 1, 3, 4, 5]);
            assert!(mat.nnz_per_col() == None);
            assert!(mat.row_indices() == &[0, 0, 1, 1, 3]);
            assert!(mat.values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }

        {
            let mat = SymbolicSparseRowMat::try_new_from_indices(nrows, ncols, indices);
            assert!(mat.is_ok());

            let (mat, order) = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.row_ptrs() == &[0, 2, 4, 4, 5, 5]);
            assert!(mat.nnz_per_row() == None);
            assert!(mat.col_indices() == &[0, 1, 1, 2, 3]);

            let mat =
                SparseRowMat::<_, f64>::new_from_order_and_values(mat, &order, values).unwrap();
            assert!(mat.values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }
        {
            let mat = SparseRowMat::try_new_from_triplets(nrows, ncols, triplets);
            assert!(mat.is_ok());

            let mat = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.row_ptrs() == &[0, 2, 4, 4, 5, 5]);
            assert!(mat.nnz_per_row() == None);
            assert!(mat.col_indices() == &[0, 1, 1, 2, 3]);
            assert!(mat.as_ref().values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }
    }

    #[test]
    fn test_from_nonnegative_indices() {
        let nrows = 5;
        let ncols = 4;

        let indices = &[
            (0, 0),
            (1, 2),
            (0, 0),
            (1, 1),
            (0, 1),
            (-1, 2),
            (-2, 1),
            (-3, -4),
            (3, 3),
            (3, 3isize),
        ];
        let values = &[
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            6.0,
            7.0f64,
        ];

        let triplets = &[
            (0, 0, 1.0),
            (1, 2, 2.0),
            (0, 0, 3.0),
            (1, 1, 4.0),
            (0, 1, 5.0),
            (-1, 2, f64::NAN),
            (-2, 1, f64::NAN),
            (-3, -4, f64::NAN),
            (3, 3, 6.0),
            (3, 3isize, 7.0),
        ];

        {
            let mat = SymbolicSparseColMat::<usize>::try_new_from_nonnegative_indices(
                nrows, ncols, indices,
            );
            assert!(mat.is_ok());

            let (mat, order) = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.col_ptrs() == &[0, 1, 3, 4, 5]);
            assert!(mat.nnz_per_col() == None);
            assert!(mat.row_indices() == &[0, 0, 1, 1, 3]);

            let mat =
                SparseColMat::<_, f64>::new_from_order_and_values(mat, &order, values).unwrap();
            assert!(mat.as_ref().values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }

        {
            let mat =
                SparseColMat::<usize, _>::try_new_from_nonnegative_triplets(nrows, ncols, triplets);
            assert!(mat.is_ok());
            let mat = mat.unwrap();

            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.col_ptrs() == &[0, 1, 3, 4, 5]);
            assert!(mat.nnz_per_col() == None);
            assert!(mat.row_indices() == &[0, 0, 1, 1, 3]);
            assert!(mat.values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }

        {
            let mat = SymbolicSparseRowMat::<usize>::try_new_from_nonnegative_indices(
                nrows, ncols, indices,
            );
            assert!(mat.is_ok());

            let (mat, order) = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.row_ptrs() == &[0, 2, 4, 4, 5, 5]);
            assert!(mat.nnz_per_row() == None);
            assert!(mat.col_indices() == &[0, 1, 1, 2, 3]);

            let mat =
                SparseRowMat::<_, f64>::new_from_order_and_values(mat, &order, values).unwrap();
            assert!(mat.values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }
        {
            let mat =
                SparseRowMat::<usize, _>::try_new_from_nonnegative_triplets(nrows, ncols, triplets);
            assert!(mat.is_ok());

            let mat = mat.unwrap();
            assert!(mat.nrows() == nrows);
            assert!(mat.ncols() == ncols);
            assert!(mat.row_ptrs() == &[0, 2, 4, 4, 5, 5]);
            assert!(mat.nnz_per_row() == None);
            assert!(mat.col_indices() == &[0, 1, 1, 2, 3]);
            assert!(mat.as_ref().values() == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }
        {
            let order = SymbolicSparseRowMat::<usize>::try_new_from_nonnegative_indices(
                nrows, ncols, indices,
            )
            .unwrap()
            .1;

            let new_values = &mut [f64::NAN; 5];
            let mut mat = SparseRowMatMut::<'_, usize, f64>::new(
                SymbolicSparseRowMatRef::new_checked(
                    nrows,
                    ncols,
                    &[0, 2, 4, 4, 5, 5],
                    None,
                    &[0, 1, 1, 2, 3],
                ),
                new_values,
            );
            mat.fill_from_order_and_values(&order, values, FillMode::Replace);

            assert!(&*new_values == &[1.0 + 3.0, 5.0, 4.0, 2.0, 6.0 + 7.0]);
        }
    }

    #[test]
    fn test_from_indices_oob_row() {
        let nrows = 5;
        let ncols = 4;

        let indices = &[
            (0, 0),
            (1, 2),
            (0, 0),
            (1, 1),
            (0, 1),
            (3, 3),
            (3, 3),
            (5, 3usize),
        ];
        let err = SymbolicSparseColMat::try_new_from_indices(nrows, ncols, indices);
        assert!(err.is_err());
        let err = err.unwrap_err();
        assert!(err == CreationError::OutOfBounds { row: 5, col: 3 });
    }

    #[test]
    fn test_from_indices_oob_col() {
        let nrows = 5;
        let ncols = 4;

        let indices = &[
            (0, 0),
            (1, 2),
            (0, 0),
            (1, 1),
            (0, 1),
            (3, 3),
            (3, 3),
            (2, 4usize),
        ];
        let err = SymbolicSparseColMat::try_new_from_indices(nrows, ncols, indices);
        assert!(err.is_err());
        let err = err.unwrap_err();
        assert!(err == CreationError::OutOfBounds { row: 2, col: 4 });
    }

    #[test]
    fn test_add_intersecting() {
        let lhs = SparseColMat::<usize, f64>::try_new_from_triplets(
            5,
            4,
            &[
                (1, 0, 1.0),
                (2, 1, 2.0),
                (3, 2, 3.0),
                (0, 0, 4.0),
                (1, 1, 5.0),
                (2, 2, 6.0),
                (3, 3, 7.0),
                (2, 0, 8.0),
                (3, 1, 9.0),
                (4, 2, 10.0),
                (0, 2, 11.0),
                (1, 3, 12.0),
                (4, 0, 13.0),
            ],
        )
        .unwrap();

        let rhs = SparseColMat::<usize, f64>::try_new_from_triplets(
            5,
            4,
            &[
                (1, 0, 10.0),
                (2, 1, 14.0),
                (3, 2, 15.0),
                (4, 3, 16.0),
                (0, 1, 17.0),
                (1, 2, 18.0),
                (2, 3, 19.0),
                (3, 0, 20.0),
                (4, 1, 21.0),
                (0, 3, 22.0),
            ],
        )
        .unwrap();

        let sum = ops::add(lhs.as_ref(), rhs.as_ref()).unwrap();
        assert!(sum.compute_nnz() == lhs.compute_nnz() + rhs.compute_nnz() - 3);

        for j in 0..4 {
            for i in 0..5 {
                assert!(sum.row_indices_of_col_raw(j)[i] == i);
            }
        }

        for j in 0..4 {
            for i in 0..5 {
                assert!(
                    sum[(i, j)] == lhs.get(i, j).unwrap_or(&0.0) + rhs.get(i, j).unwrap_or(&0.0)
                );
            }
        }
    }

    #[test]
    fn test_add_duplicates() {
        let lhs = SparseColMat::<usize, f64>::new(
            SymbolicSparseColMat::<usize>::new_checked(
                5,
                4,
                vec![0, 4, 6, 9, 12],
                None,
                vec![0, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 3],
            ),
            vec![1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.5],
        );

        let rhs = SparseColMat::<usize, f64>::new(
            SymbolicSparseColMat::<usize>::new_checked(
                5,
                4,
                vec![0, 2, 6, 8, 12],
                None,
                vec![1, 3, 0, 2, 2, 4, 1, 3, 0, 2, 4, 4],
            ),
            vec![
                11.0, 12.0, 13.0, 14.0, 14.5, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
            ],
        );

        let sum = ops::add(lhs.as_ref(), rhs.as_ref()).unwrap();

        assert!(sum.compute_nnz() == lhs.compute_nnz() + rhs.compute_nnz());

        for j in 0..4 {
            for i in 0..5 {
                assert!(
                    sum.get_all(i, j).iter().sum::<f64>()
                        == lhs.get_all(i, j).iter().sum::<f64>()
                            + rhs.get_all(i, j).iter().sum::<f64>()
                );
            }
        }
    }

    #[test]
    fn test_add_disjoint() {
        let lhs = SparseColMat::<usize, f64>::try_new_from_triplets(
            5,
            4,
            &[
                (0, 0, 1.0),
                (1, 1, 2.0),
                (2, 2, 3.0),
                (3, 3, 4.0),
                (2, 0, 5.0),
                (3, 1, 6.0),
                (4, 2, 7.0),
                (0, 2, 8.0),
                (1, 3, 9.0),
                (4, 0, 10.0),
            ],
        )
        .unwrap();

        let rhs = SparseColMat::<usize, f64>::try_new_from_triplets(
            5,
            4,
            &[
                (1, 0, 11.0),
                (2, 1, 12.0),
                (3, 2, 13.0),
                (4, 3, 14.0),
                (0, 1, 15.0),
                (1, 2, 16.0),
                (2, 3, 17.0),
                (3, 0, 18.0),
                (4, 1, 19.0),
                (0, 3, 20.0),
            ],
        )
        .unwrap();

        let sum = ops::add(lhs.as_ref(), rhs.as_ref()).unwrap();
        assert!(sum.compute_nnz() == lhs.compute_nnz() + rhs.compute_nnz());

        for j in 0..4 {
            for i in 0..5 {
                assert!(sum.row_indices_of_col_raw(j)[i] == i);
            }
        }

        for j in 0..4 {
            for i in 0..5 {
                assert!(
                    sum[(i, j)] == lhs.get(i, j).unwrap_or(&0.0) + rhs.get(i, j).unwrap_or(&0.0)
                );
            }
        }
    }
}
