//! Computes the QR decomposition of a given sparse matrix. See [`crate::linalg::qr`] for more info.
//!
//! The entry point in this module is [`SymbolicQr`] and [`factorize_symbolic_qr`].
//!
//! # Warning
//! The functions in this module accept unsorted input, and always produce unsorted decomposition
//! factors.

use super::{
    cholesky::{
        ghost_postorder,
        simplicial::EliminationTreeRef,
        supernodal::{SupernodalLltRef, SymbolicSupernodalCholesky},
    },
    colamd::{self, Control},
    ghost::{self, Array, Idx, MaybeIdx},
    mem::{self, NONE},
    nomem, try_zeroed, FaerError, Index, SupernodalThreshold, SymbolicSupernodalParams,
};
use crate::{
    assert,
    linalg::{
        householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
        temp_mat_req, temp_mat_uninit,
    },
    perm::PermRef,
    sparse::{SparseColMatRef, SymbolicSparseColMatRef},
    unzipped,
    utils::{constrained::Size, slice::*},
    zipped, Conj, MatMut, Parallelism, SignedIndex,
};
use core::iter::zip;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_entity::*;
use reborrow::*;

#[inline]
pub(crate) fn ghost_col_etree<'n, I: Index>(
    A: ghost::SymbolicSparseColMatRef<'_, 'n, '_, I>,
    col_perm: Option<ghost::PermRef<'n, '_, I>>,
    etree: &mut Array<'n, I::Signed>,
    stack: &mut PodStack,
) {
    let I = I::truncate;

    let N = A.ncols();
    let M = A.nrows();

    let (ancestor, stack) = stack.make_raw::<I::Signed>(*N);
    let (prev, _) = stack.make_raw::<I::Signed>(*M);

    let ancestor = Array::from_mut(ghost::fill_none::<I>(ancestor, N), N);
    let prev = Array::from_mut(ghost::fill_none::<I>(prev, N), M);

    mem::fill_none(etree.as_mut());
    for j in N.indices() {
        let pj = col_perm.map(|perm| perm.arrays().0[j].zx()).unwrap_or(j);
        for i_ in A.row_indices_of_col(pj) {
            let mut i = prev[i_].sx();
            while let Some(i_) = i.idx() {
                if i_ == j {
                    break;
                }
                let next_i = ancestor[i_];
                ancestor[i_] = MaybeIdx::from_index(j.truncate());
                if next_i.idx().is_none() {
                    etree[i_] = I(*j).to_signed();
                    break;
                }
                i = next_i.sx();
            }
            prev[i_] = MaybeIdx::from_index(j.truncate());
        }
    }
}

/// Computes the size and alignment of the workspace required to compute the column elimination tree
/// of a matrix $A$ with dimensions `(nrows, ncols)`.
#[inline]
pub fn col_etree_req<I: Index>(nrows: usize, ncols: usize) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        StackReq::try_new::<I>(nrows)?,
        StackReq::try_new::<I>(ncols)?,
    ])
}

/// Computes the column elimination tree of $A$, which is the same as the elimination tree of $A^H
/// A$.
///
/// `etree` has length `A.ncols()`
#[inline]
pub fn col_etree<'out, I: Index>(
    A: SymbolicSparseColMatRef<'_, I>,
    col_perm: Option<PermRef<'_, I>>,
    etree: &'out mut [I],
    stack: &mut PodStack,
) -> EliminationTreeRef<'out, I> {
    with_dim!(M, A.nrows());
    with_dim!(N, A.ncols());
    ghost_col_etree(
        ghost::SymbolicSparseColMatRef::new(A, M, N),
        col_perm.map(|perm| ghost::PermRef::new(perm, N)),
        Array::from_mut(bytemuck::cast_slice_mut(etree), N),
        stack,
    );

    EliminationTreeRef {
        inner: bytemuck::cast_slice_mut(etree),
    }
}

pub(crate) fn ghost_least_common_ancestor<'n, I: Index>(
    i: Idx<'n, usize>,
    j: Idx<'n, usize>,
    first: &Array<'n, MaybeIdx<'n, I>>,
    max_first: &mut Array<'n, MaybeIdx<'n, I>>,
    prev_leaf: &mut Array<'n, MaybeIdx<'n, I>>,
    ancestor: &mut Array<'n, Idx<'n, I>>,
) -> isize {
    if i <= j || *first[j] <= *max_first[i] {
        return -2;
    }

    max_first[i] = first[j];
    let j_prev = prev_leaf[i].sx();
    prev_leaf[i] = MaybeIdx::from_index(j.truncate());
    let Some(j_prev) = j_prev.idx() else {
        return -1;
    };
    let mut lca = j_prev;
    while lca != ancestor[lca].zx() {
        lca = ancestor[lca].zx();
    }

    let mut node = j_prev;
    while node != lca {
        let next = ancestor[node].zx();
        ancestor[node] = lca.truncate();
        node = next;
    }

    *lca as isize
}

pub(crate) fn ghost_column_counts_aat<'m, 'n, I: Index>(
    col_counts: &mut Array<'m, I>,
    min_row: &mut Array<'n, I::Signed>,
    A: ghost::SymbolicSparseColMatRef<'m, 'n, '_, I>,
    row_perm: Option<ghost::PermRef<'m, '_, I>>,
    etree: &Array<'m, MaybeIdx<'m, I>>,
    post: &Array<'m, Idx<'m, I>>,
    stack: &mut PodStack,
) {
    let M: Size<'m> = A.nrows();
    let N: Size<'n> = A.ncols();
    let n = *N;
    let m = *M;

    let delta = col_counts;
    let (first, stack) = stack.make_raw::<I::Signed>(m);
    let (max_first, stack) = stack.make_raw::<I::Signed>(m);
    let (prev_leaf, stack) = stack.make_raw::<I::Signed>(m);
    let (ancestor, stack) = stack.make_raw::<I>(m);
    let (next, stack) = stack.make_raw::<I::Signed>(n);
    let (head, _) = stack.make_raw::<I::Signed>(m);

    let post_inv = &mut *first;
    let post_inv = Array::from_mut(
        ghost::fill_zero::<I>(bytemuck::cast_slice_mut(post_inv), M),
        M,
    );
    for j in M.indices() {
        post_inv[post[j].zx()] = j.truncate();
    }
    let next = Array::from_mut(ghost::fill_none::<I>(next, N), N);
    let head = Array::from_mut(ghost::fill_none::<I>(head, N), M);

    for j in N.indices() {
        if let Some(perm) = row_perm {
            let inv = perm.arrays().1;
            min_row[j] = match Iterator::min(A.row_indices_of_col(j).map(|j| inv[j].zx())) {
                Some(first_row) => I::Signed::truncate(*first_row),
                None => *MaybeIdx::<'_, I>::none(),
            };
        } else {
            min_row[j] = match Iterator::min(A.row_indices_of_col(j)) {
                Some(first_row) => I::Signed::truncate(*first_row),
                None => *MaybeIdx::<'_, I>::none(),
            };
        }

        let min_row = if let Some(perm) = row_perm {
            let inv = perm.arrays().1;
            Iterator::min(A.row_indices_of_col(j).map(|row| post_inv[inv[row].zx()]))
        } else {
            Iterator::min(A.row_indices_of_col(j).map(|row| post_inv[row]))
        };
        if let Some(min_row) = min_row {
            let min_row = min_row.zx();
            let head = &mut head[min_row];
            next[j] = *head;
            *head = MaybeIdx::from_index(j.truncate());
        };
    }

    let first = Array::from_mut(ghost::fill_none::<I>(first, M), M);
    let max_first = Array::from_mut(ghost::fill_none::<I>(max_first, M), M);
    let prev_leaf = Array::from_mut(ghost::fill_none::<I>(prev_leaf, M), M);
    for (i, p) in ancestor.iter_mut().enumerate() {
        *p = I::truncate(i);
    }
    let ancestor = Array::from_mut(unsafe { Idx::from_slice_mut_unchecked(ancestor) }, M);

    let incr = |i: &mut I| {
        *i = I::from_signed((*i).to_signed() + I::Signed::truncate(1));
    };
    let decr = |i: &mut I| {
        *i = I::from_signed((*i).to_signed() - I::Signed::truncate(1));
    };

    for k in M.indices() {
        let mut pk = post[k].zx();
        delta[pk] = I::truncate(if first[pk].idx().is_none() { 1 } else { 0 });
        loop {
            if first[pk].idx().is_some() {
                break;
            }

            first[pk] = MaybeIdx::from_index(k.truncate());
            if let Some(parent) = etree[pk].idx() {
                pk = parent.zx();
            } else {
                break;
            }
        }
    }

    for k in M.indices() {
        let pk = post[k].zx();

        if let Some(parent) = etree[pk].idx() {
            decr(&mut delta[parent.zx()]);
        }

        let head_k = &mut head[k];
        let mut j = head_k.sx();
        *head_k = MaybeIdx::none();

        while let Some(j_) = j.idx() {
            for i in A.row_indices_of_col(j_) {
                let i = row_perm.map(|perm| perm.arrays().1[i].zx()).unwrap_or(i);
                let lca =
                    ghost_least_common_ancestor::<I>(i, pk, first, max_first, prev_leaf, ancestor);

                if lca != -2 {
                    incr(&mut delta[pk]);

                    if lca != -1 {
                        decr(&mut delta[M.check(lca as usize)]);
                    }
                }
            }
            j = next[j_].sx();
        }
        if let Some(parent) = etree[pk].idx() {
            ancestor[pk] = parent;
        }
    }

    for k in M.indices() {
        if let Some(parent) = etree[k].idx() {
            let parent = parent.zx();
            delta[parent] = I::from_signed(delta[parent].to_signed() + delta[k].to_signed());
        }
    }
}

/// Computes the size and alignment of the workspace required to compute the column counts
/// of the Cholesky factor of the matrix $A A^H$, where $A$ has dimensions `(nrows, ncols)`.
#[inline]
pub fn column_counts_aat_req<I: Index>(
    nrows: usize,
    ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        StackReq::try_new::<I>(nrows)?,
        StackReq::try_new::<I>(nrows)?,
        StackReq::try_new::<I>(nrows)?,
        StackReq::try_new::<I>(nrows)?,
        StackReq::try_new::<I>(nrows)?,
        StackReq::try_new::<I>(ncols)?,
    ])
}

/// Computes the column counts of the Cholesky factor of $A\top A$.
///
/// - `col_counts` has length `A.ncols()`.
/// - `min_col` has length `A.nrows()`.
/// - `col_perm` has length `A.ncols()`: fill reducing permutation.
/// - `etree` has length `A.ncols()`: column elimination tree of $A A^H$.
/// - `post` has length `A.ncols()`: postordering of `etree`.
///
/// # Warning
/// The function takes as input `A.transpose()`, not `A`.
pub fn column_counts_ata<'m, 'n, I: Index>(
    col_counts: &mut [I],
    min_col: &mut [I],
    AT: SymbolicSparseColMatRef<'_, I>,
    col_perm: Option<PermRef<'_, I>>,
    etree: EliminationTreeRef<'_, I>,
    post: &[I],
    stack: &mut PodStack,
) {
    with_dim!(M, AT.nrows());
    with_dim!(N, AT.ncols());

    let A = ghost::SymbolicSparseColMatRef::new(AT, M, N);
    ghost_column_counts_aat(
        Array::from_mut(col_counts, M),
        Array::from_mut(bytemuck::cast_slice_mut(min_col), N),
        A,
        col_perm.map(|perm| ghost::PermRef::new(perm, M)),
        etree.ghost_inner(M),
        Array::from_ref(Idx::from_slice_ref_checked(post, M), M),
        stack,
    )
}

/// Computes the size and alignment of the workspace required to compute the postordering of an
/// elimination tree of size `n`.
#[inline]
pub fn postorder_req<I: Index>(n: usize) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([StackReq::try_new::<I>(n)?; 3])
}

/// Computes a postordering of the elimination tree of size `n`.
#[inline]
pub fn postorder<I: Index>(post: &mut [I], etree: EliminationTreeRef<'_, I>, stack: &mut PodStack) {
    with_dim!(N, etree.inner.len());
    ghost_postorder(Array::from_mut(post, N), etree.ghost_inner(N), stack)
}

/// Supernodal factorization module.
///
/// A supernodal factorization is one that processes the elements of the QR factors of the
/// input matrix by blocks, rather than by single elements. This is more efficient if the QR factors
/// are somewhat dense.
pub mod supernodal {
    use super::*;
    use crate::assert;

    /// Symbolic structure of the Householder reflections that compose $Q$,
    ///
    /// such that:
    /// $$ Q = (I - H_1 T_1^{-1} H_1^H) \cdot (I - H_2 T_2^{-1} H_2^H) \dots (I - H_k T_k^{-1}
    /// H_k^H)$$
    #[derive(Debug)]
    pub struct SymbolicSupernodalHouseholder<I> {
        col_ptrs_for_row_indices: alloc::vec::Vec<I>,
        col_ptrs_for_tau_values: alloc::vec::Vec<I>,
        col_ptrs_for_values: alloc::vec::Vec<I>,
        super_etree: alloc::vec::Vec<I>,
        max_blocksize: alloc::vec::Vec<I>,
        nrows: usize,
    }

    impl<I: Index> SymbolicSupernodalHouseholder<I> {
        /// Returns the number of rows of the Householder factors.
        #[inline]
        pub fn nrows(&self) -> usize {
            self.nrows
        }

        /// Returns the number of supernodes in the symbolic QR.
        #[inline]
        pub fn n_supernodes(&self) -> usize {
            self.super_etree.len()
        }

        /// Returns the column pointers for the numerical values of the Householder factors.
        #[inline]
        pub fn col_ptrs_for_householder_values(&self) -> &[I] {
            self.col_ptrs_for_values.as_ref()
        }

        /// Returns the column pointers for the numerical values of the $T$ factors.
        #[inline]
        pub fn col_ptrs_for_tau_values(&self) -> &[I] {
            self.col_ptrs_for_tau_values.as_ref()
        }

        /// Returns the column pointers for the row indices of the Householder factors.
        #[inline]
        pub fn col_ptrs_for_householder_row_indices(&self) -> &[I] {
            self.col_ptrs_for_row_indices.as_ref()
        }

        /// Returns the length of the slice that can be used to contain the numerical values of the
        /// Householder factors.
        #[inline]
        pub fn len_householder_values(&self) -> usize {
            self.col_ptrs_for_householder_values()[self.n_supernodes()].zx()
        }
        /// Returns the length of the slice that can be used to contain the row indices of the
        /// Householder factors.
        #[inline]
        pub fn len_householder_row_indices(&self) -> usize {
            self.col_ptrs_for_householder_row_indices()[self.n_supernodes()].zx()
        }
        /// Returns the length of the slice that can be used to contain the numerical values of the
        /// $T$ factors.
        #[inline]
        pub fn len_tau_values(&self) -> usize {
            self.col_ptrs_for_tau_values()[self.n_supernodes()].zx()
        }
    }

    /// Symbolic structure of the QR decomposition,
    #[derive(Debug)]
    pub struct SymbolicSupernodalQr<I: Index> {
        L: SymbolicSupernodalCholesky<I>,
        H: SymbolicSupernodalHouseholder<I>,
        min_col: alloc::vec::Vec<I>,
        min_col_perm: alloc::vec::Vec<I>,
        index_to_super: alloc::vec::Vec<I>,
        child_head: alloc::vec::Vec<I>,
        child_next: alloc::vec::Vec<I>,
    }

    impl<I: Index> SymbolicSupernodalQr<I> {
        /// Returns the symbolic structure of $R^H$.
        #[inline]
        pub fn r_adjoint(&self) -> &SymbolicSupernodalCholesky<I> {
            &self.L
        }

        /// Returns the symbolic structure of the Householder and $T$ factors.
        #[inline]
        pub fn householder(&self) -> &SymbolicSupernodalHouseholder<I> {
            &self.H
        }

        /// Computes the size and alignment of the workspace required to solve the linear system $A
        /// x = \text{rhs}$ in the sense of least squares.
        pub fn solve_in_place_req<E: Entity>(
            &self,
            rhs_ncols: usize,
            parallelism: Parallelism,
        ) -> Result<StackReq, SizeOverflow> {
            let _ = parallelism;
            let L_symbolic = self.r_adjoint();
            let H_symbolic = self.householder();
            let n_supernodes = L_symbolic.n_supernodes();

            let mut loop_req = StackReq::empty();
            for s in 0..n_supernodes {
                let s_h_row_begin = H_symbolic.col_ptrs_for_row_indices[s].zx();
                let s_h_row_full_end = H_symbolic.col_ptrs_for_row_indices[s + 1].zx();
                let max_blocksize = H_symbolic.max_blocksize[s].zx();

                loop_req = loop_req.try_or(crate::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_req::<E>(s_h_row_full_end - s_h_row_begin, max_blocksize, rhs_ncols)?)?;
            }

            Ok(loop_req)
        }
    }

    /// Computes the size and alignment of the workspace required to compute the symbolic QR
    /// factorization of a matrix with dimensions `(nrows, ncols)`.
    pub fn factorize_supernodal_symbolic_qr_req<I: Index>(
        nrows: usize,
        ncols: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let _ = nrows;
        crate::sparse::linalg::cholesky::supernodal::factorize_supernodal_symbolic_cholesky_req::<I>(
            ncols,
        )
    }

    /// Computes the symbolic QR factorization of a matrix $A$, given a fill-reducing column
    /// permutation, and the outputs of the pre-factorization steps.
    pub fn factorize_supernodal_symbolic_qr<I: Index>(
        A: SymbolicSparseColMatRef<'_, I>,
        col_perm: Option<PermRef<'_, I>>,
        min_col: alloc::vec::Vec<I>,
        etree: EliminationTreeRef<'_, I>,
        col_counts: &[I],
        stack: &mut PodStack,
        params: SymbolicSupernodalParams<'_>,
    ) -> Result<SymbolicSupernodalQr<I>, FaerError> {
        let m = A.nrows();
        let n = A.ncols();

        with_dim!(M, m);
        with_dim!(N, n);
        let A = ghost::SymbolicSparseColMatRef::new(A, M, N);
        let mut stack = stack;
        let (L, H) = {
            let etree = etree.ghost_inner(N);
            let min_col = Array::from_ref(
                MaybeIdx::from_slice_ref_checked(bytemuck::cast_slice(&min_col), N),
                M,
            );
            let L =
                crate::sparse::linalg::cholesky::supernodal::ghost_factorize_supernodal_symbolic(
                    A,
                    col_perm.map(|perm| ghost::PermRef::new(perm, N)),
                    Some(min_col),
                    crate::sparse::linalg::cholesky::supernodal::CholeskyInput::ATA,
                    etree,
                    Array::from_ref(col_counts, N),
                    stack.rb_mut(),
                    params,
                )?;

            let H =
                ghost_factorize_supernodal_householder_symbolic(&L, M, N, min_col, etree, stack)?;

            (L, H)
        };
        let n_supernodes = L.n_supernodes();

        let mut min_col_perm = try_zeroed::<I>(m)?;
        let mut index_to_super = try_zeroed::<I>(n)?;
        let mut child_head = try_zeroed::<I>(n_supernodes)?;
        let mut child_next = try_zeroed::<I>(n_supernodes)?;
        for i in 0..m {
            min_col_perm[i] = I::truncate(i);
        }
        min_col_perm.sort_unstable_by_key(|i| min_col[i.zx()]);
        for s in 0..n_supernodes {
            index_to_super[L.supernode_begin()[s].zx()..L.supernode_end()[s].zx()]
                .fill(I::truncate(s));
        }

        mem::fill_none::<I::Signed>(bytemuck::cast_slice_mut(&mut child_head));
        mem::fill_none::<I::Signed>(bytemuck::cast_slice_mut(&mut child_next));

        for s in 0..n_supernodes {
            let parent = H.super_etree[s];
            if parent.to_signed() >= I::Signed::truncate(0) {
                let parent = parent.zx();
                let head = child_head[parent];
                child_next[s] = head;
                child_head[parent] = I::truncate(s);
            }
        }

        Ok(SymbolicSupernodalQr {
            L,
            H,
            min_col,
            min_col_perm,
            index_to_super,
            child_head,
            child_next,
        })
    }

    fn ghost_factorize_supernodal_householder_symbolic<'m, 'n, I: Index>(
        L_symbolic: &SymbolicSupernodalCholesky<I>,
        M: Size<'m>,
        N: Size<'n>,
        min_col: &Array<'m, MaybeIdx<'n, I>>,
        etree: &Array<'n, MaybeIdx<'n, I>>,
        stack: &mut PodStack,
    ) -> Result<SymbolicSupernodalHouseholder<I>, FaerError> {
        let n_supernodes = L_symbolic.n_supernodes();

        with_dim!(N_SUPERNODES, n_supernodes);

        let mut col_ptrs_for_row_indices = try_zeroed::<I>(n_supernodes + 1)?;
        let mut col_ptrs_for_tau_values = try_zeroed::<I>(n_supernodes + 1)?;
        let mut col_ptrs_for_values = try_zeroed::<I>(n_supernodes + 1)?;
        let mut super_etree_ = try_zeroed::<I>(n_supernodes)?;
        let mut max_blocksize = try_zeroed::<I>(n_supernodes)?;
        let super_etree = bytemuck::cast_slice_mut::<I, I::Signed>(&mut super_etree_);

        let to_wide = |i: I| i.zx() as u128;
        let from_wide = |i: u128| I::truncate(i as usize);
        let from_wide_checked = |i: u128| -> Option<I> {
            (i <= to_wide(I::from_signed(I::Signed::MAX))).then_some(I::truncate(i as usize))
        };

        let supernode_begin = Array::from_ref(L_symbolic.supernode_begin(), N_SUPERNODES);
        let supernode_end = Array::from_ref(L_symbolic.supernode_end(), N_SUPERNODES);
        let L_col_ptrs_for_row_indices = L_symbolic.col_ptrs_for_row_indices();

        let (index_to_super, _) = stack.make_raw::<I>(*N);

        for s in N_SUPERNODES.indices() {
            index_to_super[supernode_begin[s].zx()..supernode_end[s].zx()].fill(*s.truncate::<I>());
        }
        let index_to_super =
            Array::from_ref(Idx::from_slice_ref_checked(index_to_super, N_SUPERNODES), N);

        let super_etree = Array::from_mut(super_etree, N_SUPERNODES);
        for s in N_SUPERNODES.indices() {
            let last = supernode_end[s].zx() - 1;
            if let Some(parent) = etree[N.check(last)].idx() {
                super_etree[s] = index_to_super[parent.zx()].to_signed();
            } else {
                super_etree[s] = I::Signed::truncate(NONE);
            }
        }
        let super_etree = Array::from_ref(
            MaybeIdx::<'_, I>::from_slice_ref_checked(super_etree.as_ref(), N_SUPERNODES),
            N_SUPERNODES,
        );

        let non_zero_count = Array::from_mut(&mut col_ptrs_for_row_indices[1..], N_SUPERNODES);
        for i in M.indices() {
            let Some(min_col) = min_col[i].idx() else {
                continue;
            };
            non_zero_count[index_to_super[min_col.zx()].zx()] += I::truncate(1);
        }

        for s in N_SUPERNODES.indices() {
            if let Some(parent) = super_etree[s].idx() {
                let s_col_count =
                    L_col_ptrs_for_row_indices[*s + 1] - L_col_ptrs_for_row_indices[*s];
                let panel_width = supernode_end[s] - supernode_begin[s];

                let s_count = non_zero_count[s];
                non_zero_count[parent.zx()] +=
                    Ord::min(Ord::max(s_count, panel_width) - panel_width, s_col_count);
            }
        }

        let mut val_count = to_wide(I::truncate(0));
        let mut tau_count = to_wide(I::truncate(0));
        let mut row_count = to_wide(I::truncate(0));
        for (s, ((next_row_ptr, next_val_ptr), next_tau_ptr)) in zip(
            N_SUPERNODES.indices(),
            zip(
                zip(
                    &mut col_ptrs_for_row_indices[1..],
                    &mut col_ptrs_for_values[1..],
                ),
                &mut col_ptrs_for_tau_values[1..],
            ),
        ) {
            let panel_width = supernode_end[s] - supernode_begin[s];
            let s_row_count = *next_row_ptr;
            let s_col_count =
                panel_width + (L_col_ptrs_for_row_indices[*s + 1] - L_col_ptrs_for_row_indices[*s]);
            val_count += to_wide(s_row_count) * to_wide(s_col_count);
            row_count += to_wide(s_row_count);
            let blocksize = crate::linalg::qr::no_pivoting::compute::recommended_blocksize::<Symbolic>(
                s_row_count.zx(),
                s_col_count.zx(),
            ) as u128;
            max_blocksize[*s] = from_wide(blocksize);
            tau_count += blocksize * to_wide(Ord::min(s_row_count, s_col_count));
            *next_val_ptr = from_wide(val_count);
            *next_row_ptr = from_wide(row_count);
            *next_tau_ptr = from_wide(tau_count);
        }
        from_wide_checked(row_count).ok_or(FaerError::IndexOverflow)?;
        from_wide_checked(tau_count).ok_or(FaerError::IndexOverflow)?;
        from_wide_checked(val_count).ok_or(FaerError::IndexOverflow)?;

        Ok(SymbolicSupernodalHouseholder {
            col_ptrs_for_row_indices,
            col_ptrs_for_values,
            super_etree: super_etree_,
            col_ptrs_for_tau_values,
            max_blocksize,
            nrows: *M,
        })
    }

    /// QR factors containing both the symbolic and numeric representations.
    #[derive(Debug)]
    pub struct SupernodalQrRef<'a, I: Index, E: Entity> {
        symbolic: &'a SymbolicSupernodalQr<I>,
        rt_values: SliceGroup<'a, E>,
        householder_values: SliceGroup<'a, E>,
        tau_values: SliceGroup<'a, E>,
        householder_row_indices: &'a [I],
        tau_blocksize: &'a [I],
        householder_nrows: &'a [I],
        householder_ncols: &'a [I],
    }

    impl<I: Index, E: Entity> Copy for SupernodalQrRef<'_, I, E> {}
    impl<I: Index, E: Entity> Clone for SupernodalQrRef<'_, I, E> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<'a, I: Index, E: Entity> SupernodalQrRef<'a, I, E> {
        /// Creates QR factors from their components.
        ///
        /// # Safety
        /// The inputs must be the outputs of [`factorize_supernodal_numeric_qr`].
        #[inline]
        pub unsafe fn new_unchecked(
            symbolic: &'a SymbolicSupernodalQr<I>,
            householder_row_indices: &'a [I],
            tau_blocksize: &'a [I],
            householder_nrows: &'a [I],
            householder_ncols: &'a [I],
            r_values: GroupFor<E, &'a [E::Unit]>,
            householder_values: GroupFor<E, &'a [E::Unit]>,
            tau_values: GroupFor<E, &'a [E::Unit]>,
        ) -> Self {
            let rt_values = SliceGroup::new(r_values);
            let householder_values = SliceGroup::new(householder_values);
            let tau_values = SliceGroup::new(tau_values);
            assert!(rt_values.len() == symbolic.r_adjoint().len_values());
            assert!(tau_values.len() == symbolic.householder().len_tau_values());
            assert!(householder_values.len() == symbolic.householder().len_householder_values());
            assert!(tau_blocksize.len() == householder_nrows.len());
            Self {
                symbolic,
                tau_blocksize,
                householder_nrows,
                householder_ncols,
                rt_values,
                householder_values,
                tau_values,
                householder_row_indices,
            }
        }

        /// Returns the symbolic structure of the QR factorization.
        #[inline]
        pub fn symbolic(self) -> &'a SymbolicSupernodalQr<I> {
            self.symbolic
        }

        /// Returns the numerical values of the factor $R$ of the QR factorization.
        #[inline]
        pub fn r_values(self) -> GroupFor<E, &'a [E::Unit]> {
            self.rt_values.into_inner()
        }

        /// Returns the numerical values of the Householder factors of the QR factorization.
        #[inline]
        pub fn householder_values(self) -> GroupFor<E, &'a [E::Unit]> {
            self.householder_values.into_inner()
        }

        /// Returns the numerical values of the $T$ factors of the QR factorization.
        #[inline]
        pub fn tau_values(self) -> GroupFor<E, &'a [E::Unit]> {
            self.tau_values.into_inner()
        }

        /// Solves the equation $\text{Op}(A) x = \text{rhs}$ in the sense of least squares, where
        /// $\text{Op}$ is either the identity or the conjugate, depending on the value of `conj`,
        /// and stores the result in the upper part of `rhs`.
        ///
        /// `work` is a temporary workspace with the same dimensions as `rhs`.
        #[track_caller]
        pub fn solve_in_place_with_conj(
            &self,
            conj: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            work: MatMut<'_, E>,
            stack: &mut PodStack,
        ) where
            E: ComplexField,
        {
            let L_symbolic = self.symbolic().r_adjoint();
            let H_symbolic = self.symbolic().householder();
            let n_supernodes = L_symbolic.n_supernodes();

            assert!(rhs.nrows() == self.symbolic().householder().nrows);

            let mut x = rhs;
            let k = x.ncols();

            let mut stack = stack;
            let mut tmp = work;
            tmp.fill_zero();

            // x <- Q^T x
            {
                let H = self.householder_values;
                let tau = self.tau_values;

                let mut block_count = 0usize;
                for s in 0..n_supernodes {
                    let tau_begin = H_symbolic.col_ptrs_for_tau_values[s].zx();
                    let tau_end = H_symbolic.col_ptrs_for_tau_values[s + 1].zx();

                    let s_h_row_begin = H_symbolic.col_ptrs_for_row_indices[s].zx();
                    let s_h_row_full_end = H_symbolic.col_ptrs_for_row_indices[s + 1].zx();

                    let s_col_begin = L_symbolic.supernode_begin()[s].zx();
                    let s_col_end = L_symbolic.supernode_end()[s].zx();
                    let s_ncols = s_col_end - s_col_begin;

                    let s_row_indices_in_panel =
                        &self.householder_row_indices[s_h_row_begin..s_h_row_full_end];

                    let mut tmp = tmp
                        .rb_mut()
                        .subrows_mut(s_col_begin, s_h_row_full_end - s_h_row_begin);
                    for j in 0..k {
                        for idx in 0..s_h_row_full_end - s_h_row_begin {
                            let i = s_row_indices_in_panel[idx].zx();
                            tmp.write(idx, j, x.read(i, j));
                        }
                    }

                    let s_H = H.subslice(
                        H_symbolic.col_ptrs_for_values[s].zx()
                            ..H_symbolic.col_ptrs_for_values[s + 1].zx(),
                    );

                    let s_H = crate::mat::from_column_major_slice_generic::<'_, E, _, _>(
                        s_H.into_inner(),
                        s_h_row_full_end - s_h_row_begin,
                        s_ncols
                            + (L_symbolic.col_ptrs_for_row_indices()[s + 1].zx()
                                - L_symbolic.col_ptrs_for_row_indices()[s].zx()),
                    );
                    let s_tau = tau.subslice(tau_begin..tau_end);
                    let max_blocksize = H_symbolic.max_blocksize[s].zx();
                    let s_tau = crate::mat::from_column_major_slice_generic::<'_, E, _, _>(
                        s_tau.into_inner(),
                        max_blocksize,
                        Ord::min(s_H.ncols(), s_h_row_full_end - s_h_row_begin),
                    );

                    let mut start = 0;
                    let end = s_H.ncols();
                    while start < end {
                        let bs = self.tau_blocksize[block_count].zx();
                        let nrows = self.householder_nrows[block_count].zx();
                        let ncols = self.householder_ncols[block_count].zx();

                        let b_H = s_H.submatrix(start, start, nrows, ncols);
                        let b_tau = s_tau.subcols(start, ncols).subrows(0, bs);

                        apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                            b_H.rb(),
                            b_tau.rb(),
                            crate::Conj::Yes.compose(conj),
                            tmp.rb_mut().subrows_mut(start, nrows),
                            parallelism,
                            stack.rb_mut(),
                        );

                        start += ncols;
                        block_count += 1;

                        if start >= s_H.nrows() {
                            break;
                        }
                    }

                    for j in 0..k {
                        for idx in 0..s_h_row_full_end - s_h_row_begin {
                            let i = s_row_indices_in_panel[idx].zx();
                            x.write(i, j, tmp.read(idx, j));
                        }
                    }
                }
            }
            let m = H_symbolic.nrows;
            let n = L_symbolic.nrows();
            x.rb_mut()
                .subrows_mut(0, n)
                .copy_from(tmp.rb().subrows(0, n));
            x.rb_mut().subrows_mut(n, m - n).fill_zero();

            // x <- R^-1 x = L^-T x
            {
                let L = SupernodalLltRef::<'_, I, E>::new(L_symbolic, self.rt_values.into_inner());

                for s in (0..n_supernodes).rev() {
                    let s = L.supernode(s);
                    let size = s.matrix().ncols();
                    let s_L = s.matrix();
                    let (s_L_top, s_L_bot) = s_L.split_at_row(size);

                    let mut tmp = tmp.rb_mut().subrows_mut(0, s.pattern().len());
                    for j in 0..k {
                        for (idx, i) in s.pattern().iter().enumerate() {
                            let i = i.zx();
                            tmp.write(idx, j, x.read(i, j));
                        }
                    }

                    let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
                    crate::linalg::matmul::matmul_with_conj(
                        x_top.rb_mut(),
                        s_L_bot.transpose(),
                        conj.compose(Conj::Yes),
                        tmp.rb(),
                        Conj::No,
                        Some(E::faer_one()),
                        E::faer_one().faer_neg(),
                        parallelism,
                    );
                    crate::linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(
                        s_L_top.transpose(),
                        conj.compose(Conj::Yes),
                        x_top.rb_mut(),
                        parallelism,
                    );
                }
            }
        }
    }

    /// Computes the size and alignment of the workspace required to compute the numerical QR
    /// factorization of the matrix whose structure was used to produce the symbolic structure.
    #[track_caller]
    pub fn factorize_supernodal_numeric_qr_req<I: Index, E: Entity>(
        symbolic: &SymbolicSupernodalQr<I>,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let n_supernodes = symbolic.L.n_supernodes();
        let n = symbolic.L.dimension;
        let m = symbolic.H.nrows;
        let init_req = StackReq::try_all_of([
            StackReq::try_new::<I>(symbolic.H.len_householder_row_indices())?,
            StackReq::try_new::<I>(n_supernodes)?,
            StackReq::try_new::<I>(n)?,
            StackReq::try_new::<I>(n)?,
            StackReq::try_new::<I>(m)?,
            StackReq::try_new::<I>(m)?,
        ])?;

        let mut loop_req = StackReq::empty();
        for s in 0..n_supernodes {
            let s_h_row_begin = symbolic.H.col_ptrs_for_row_indices[s].zx();
            let s_h_row_full_end = symbolic.H.col_ptrs_for_row_indices[s + 1].zx();
            let max_blocksize = symbolic.H.max_blocksize[s].zx();
            let s_col_begin = symbolic.L.supernode_begin()[s].zx();
            let s_col_end = symbolic.L.supernode_end()[s].zx();
            let s_ncols = s_col_end - s_col_begin;
            let s_pattern_len = symbolic.L.col_ptrs_for_row_indices()[s + 1].zx()
                - symbolic.L.col_ptrs_for_row_indices()[s].zx();

            loop_req = loop_req.try_or(
                crate::linalg::qr::no_pivoting::compute::qr_in_place_req::<E>(
                    s_h_row_full_end - s_h_row_begin,
                    s_ncols + s_pattern_len,
                    max_blocksize,
                    parallelism,
                    Default::default(),
                )?,
            )?;

            loop_req = loop_req.try_or(crate::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_req::<E>(s_h_row_full_end - s_h_row_begin, max_blocksize, s_ncols + s_pattern_len)?)?;
        }

        init_req.try_and(loop_req)
    }

    /// Computes the numerical QR factorization of $A$.
    ///
    /// - `householder_row_indices` must have length
    /// `symbolic.householder().len_householder_row_indices()`
    /// - `tau_blocksize` must have length `symbolic.householder().len_householder_row_indices() +
    ///   symbolic.householder().n_supernodes()`.
    /// - `householder_nrows` must have length `symbolic.householder().len_householder_row_indices()
    ///   + symbolic.householder().n_supernodes()`.
    /// - `householder_ncols` must have length `symbolic.householder().len_householder_row_indices()
    ///   + symbolic.householder().n_supernodes()`.
    /// - `r_values` must have length `symbolic.r_adjoint().len_values()`.
    /// - `householder_values` must have length
    ///   `symbolic.householder().length_householder_values()`.
    /// - `tau_values` must have length `symbolic.householder().len_tau_values()`.
    ///
    /// # Warning
    /// - Note that the matrix takes as input `A.transpose()`, not `A`.
    #[track_caller]
    pub fn factorize_supernodal_numeric_qr<'a, I: Index, E: ComplexField>(
        householder_row_indices: &'a mut [I],
        tau_blocksize: &'a mut [I],
        householder_nrows: &'a mut [I],
        householder_ncols: &'a mut [I],

        r_values: GroupFor<E, &'a mut [E::Unit]>,
        householder_values: GroupFor<E, &'a mut [E::Unit]>,
        tau_values: GroupFor<E, &'a mut [E::Unit]>,

        AT: SparseColMatRef<'_, I, E>,
        col_perm: Option<PermRef<'_, I>>,
        symbolic: &'a SymbolicSupernodalQr<I>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) -> SupernodalQrRef<'a, I, E> {
        {
            let L_values = SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&r_values)));
            let householder_values =
                SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&householder_values)));
            let tau_values = SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&tau_values)));
            assert!(all(
                householder_row_indices.len()
                    == symbolic.householder().len_householder_row_indices(),
                L_values.len() == symbolic.r_adjoint().len_values(),
                householder_values.len() == symbolic.householder().len_householder_values(),
                tau_values.len() == symbolic.householder().len_tau_values(),
                tau_blocksize.len()
                    == symbolic.householder().len_householder_row_indices()
                        + symbolic.householder().n_supernodes(),
                householder_nrows.len()
                    == symbolic.householder().len_householder_row_indices()
                        + symbolic.householder().n_supernodes(),
                householder_ncols.len()
                    == symbolic.householder().len_householder_row_indices()
                        + symbolic.householder().n_supernodes(),
            ));
        }
        let mut r_values = r_values;
        let mut tau_values = tau_values;
        let mut householder_values = householder_values;

        factorize_supernodal_numeric_qr_impl(
            householder_row_indices,
            tau_blocksize,
            householder_nrows,
            householder_ncols,
            E::faer_rb_mut(E::faer_as_mut(&mut r_values)),
            E::faer_rb_mut(E::faer_as_mut(&mut householder_values)),
            E::faer_rb_mut(E::faer_as_mut(&mut tau_values)),
            AT,
            col_perm,
            &symbolic.L,
            &symbolic.H,
            &symbolic.min_col,
            &symbolic.min_col_perm,
            &symbolic.index_to_super,
            bytemuck::cast_slice(&symbolic.child_head),
            bytemuck::cast_slice(&symbolic.child_next),
            parallelism,
            stack,
        );

        unsafe {
            SupernodalQrRef::<'_, I, E>::new_unchecked(
                symbolic,
                householder_row_indices,
                tau_blocksize,
                householder_nrows,
                householder_ncols,
                E::faer_into_const(r_values),
                E::faer_into_const(householder_values),
                E::faer_into_const(tau_values),
            )
        }
    }

    pub(crate) fn factorize_supernodal_numeric_qr_impl<I: Index, E: ComplexField>(
        // len: col_ptrs_for_row_indices[n_supernodes]
        householder_row_indices: &mut [I],

        tau_blocksize: &mut [I],
        householder_nrows: &mut [I],
        householder_ncols: &mut [I],

        L_values: GroupFor<E, &mut [E::Unit]>,
        householder_values: GroupFor<E, &mut [E::Unit]>,
        tau_values: GroupFor<E, &mut [E::Unit]>,

        AT: SparseColMatRef<'_, I, E>,
        col_perm: Option<PermRef<'_, I>>,
        L_symbolic: &SymbolicSupernodalCholesky<I>,
        H_symbolic: &SymbolicSupernodalHouseholder<I>,
        min_col: &[I],
        min_col_perm: &[I],
        index_to_super: &[I],
        child_head: &[I::Signed],
        child_next: &[I::Signed],

        parallelism: Parallelism,
        stack: &mut PodStack,
    ) -> usize {
        let n_supernodes = L_symbolic.n_supernodes();
        let m = AT.ncols();
        let n = AT.nrows();

        let mut L_values = SliceGroupMut::<'_, E>::new(L_values);
        let mut householder_values = SliceGroupMut::<'_, E>::new(householder_values);
        let mut tau_values = SliceGroupMut::<'_, E>::new(tau_values);
        let mut block_count = 0;

        let (min_col_in_panel, stack) =
            stack.make_raw::<I>(H_symbolic.len_householder_row_indices());
        let (min_col_in_panel_perm, stack) = stack.make_raw::<I>(m);
        let (col_end_for_row_indices_in_panel, stack) = stack.make_raw::<I>(n_supernodes);
        let (col_global_to_local, stack) = stack.make_raw::<I::Signed>(n);
        let (child_col_global_to_local, stack) = stack.make_raw::<I::Signed>(n);
        let (child_row_global_to_local, mut stack) = stack.make_raw::<I::Signed>(m);

        tau_values.fill_zero();
        L_values.fill_zero();
        householder_values.fill_zero();

        col_end_for_row_indices_in_panel
            .copy_from_slice(&H_symbolic.col_ptrs_for_row_indices[..n_supernodes]);

        for i in 0..m {
            let i = min_col_perm[i].zx();
            let min_col = min_col[i].zx();
            if min_col < n {
                let s = index_to_super[min_col].zx();
                let pos = &mut col_end_for_row_indices_in_panel[s];
                householder_row_indices[pos.zx()] = I::truncate(i);
                min_col_in_panel[pos.zx()] = I::truncate(min_col);
                *pos += I::truncate(1);
            }
        }

        mem::fill_none(col_global_to_local);
        mem::fill_none(child_col_global_to_local);
        mem::fill_none(child_row_global_to_local);

        let supernode_begin = L_symbolic.supernode_begin();
        let supernode_end = L_symbolic.supernode_end();

        let super_etree = &*H_symbolic.super_etree;

        let col_pattern = |node: usize| {
            &L_symbolic.row_indices()[L_symbolic.col_ptrs_for_row_indices()[node].zx()
                ..L_symbolic.col_ptrs_for_row_indices()[node + 1].zx()]
        };

        // assemble the parts from child supernodes
        for s in 0..n_supernodes {
            // all child nodes should be fully assembled
            let s_h_row_begin = H_symbolic.col_ptrs_for_row_indices[s].zx();
            let s_h_row_full_end = H_symbolic.col_ptrs_for_row_indices[s + 1].zx();
            let s_h_row_end = col_end_for_row_indices_in_panel[s].zx();

            let s_col_begin = supernode_begin[s].zx();
            let s_col_end = supernode_end[s].zx();
            let s_ncols = s_col_end - s_col_begin;

            let s_pattern = col_pattern(s);

            for i in 0..s_ncols {
                col_global_to_local[s_col_begin + i] = I::Signed::truncate(i);
            }
            for (i, &col) in s_pattern.iter().enumerate() {
                col_global_to_local[col.zx()] = I::Signed::truncate(i + s_ncols);
            }

            let (s_min_col_in_panel, parent_min_col_in_panel) =
                min_col_in_panel.split_at_mut(s_h_row_end);
            let parent_offset = s_h_row_end;
            let (c_min_col_in_panel, s_min_col_in_panel) =
                s_min_col_in_panel.split_at_mut(s_h_row_begin);

            let (householder_row_indices, parent_row_indices_in_panel) =
                householder_row_indices.split_at_mut(s_h_row_end);

            let (s_H, _) = householder_values
                .rb_mut()
                .split_at(H_symbolic.col_ptrs_for_values[s + 1].zx());
            let (c_H, s_H) = s_H.split_at(H_symbolic.col_ptrs_for_values[s].zx());
            let c_H = c_H.into_const();

            let mut s_H = crate::mat::from_column_major_slice_mut_generic::<'_, E, _, _>(
                s_H.into_inner(),
                s_h_row_full_end - s_h_row_begin,
                s_ncols + s_pattern.len(),
            )
            .subrows_mut(0, s_h_row_end - s_h_row_begin);

            {
                let s_min_col_in_panel_perm =
                    &mut min_col_in_panel_perm[0..s_h_row_end - s_h_row_begin];
                for (i, p) in s_min_col_in_panel_perm.iter_mut().enumerate() {
                    *p = I::truncate(i);
                }
                s_min_col_in_panel_perm.sort_unstable_by_key(|i| s_min_col_in_panel[i.zx()]);

                let s_row_indices_in_panel = &mut householder_row_indices[s_h_row_begin..];
                let tmp: &mut [I] = bytemuck::cast_slice_mut(
                    &mut child_row_global_to_local[..s_h_row_end - s_h_row_begin],
                );

                for (i, p) in s_min_col_in_panel_perm.iter().enumerate() {
                    let p = p.zx();
                    tmp[i] = s_min_col_in_panel[p];
                }
                s_min_col_in_panel.copy_from_slice(tmp);

                for (i, p) in s_min_col_in_panel_perm.iter().enumerate() {
                    let p = p.zx();
                    tmp[i] = s_row_indices_in_panel[p];
                }
                s_row_indices_in_panel.copy_from_slice(tmp);
                for (i, p) in s_min_col_in_panel_perm.iter_mut().enumerate() {
                    *p = I::truncate(i);
                }

                mem::fill_none::<I::Signed>(bytemuck::cast_slice_mut(tmp));
            }

            let s_row_indices_in_panel = &householder_row_indices[s_h_row_begin..];

            for idx in 0..s_h_row_end - s_h_row_begin {
                let i = s_row_indices_in_panel[idx].zx();
                if min_col[i].zx() >= s_col_begin {
                    for (j, value) in zip(
                        AT.row_indices_of_col(i),
                        SliceGroup::<'_, E>::new(AT.values_of_col(i)).into_ref_iter(),
                    ) {
                        let pj = col_perm.map(|perm| perm.arrays().1[j].zx()).unwrap_or(j);
                        let ix = idx;
                        let iy = col_global_to_local[pj].zx();
                        s_H.write(ix, iy, s_H.read(ix, iy).faer_add(value.read()));
                    }
                }
            }

            let mut child_ = child_head[s];
            while child_ >= I::Signed::truncate(0) {
                let child = child_.zx();
                assert!(super_etree[child].zx() == s);
                let c_pattern = col_pattern(child);
                let c_col_begin = supernode_begin[child].zx();
                let c_col_end = supernode_end[child].zx();
                let c_ncols = c_col_end - c_col_begin;

                let c_h_row_begin = H_symbolic.col_ptrs_for_row_indices[child].zx();
                let c_h_row_end = H_symbolic.col_ptrs_for_row_indices[child + 1].zx();

                let c_row_indices_in_panel = &householder_row_indices[c_h_row_begin..c_h_row_end];
                let c_min_col_in_panel = &c_min_col_in_panel[c_h_row_begin..c_h_row_end];

                let c_H = c_H.subslice(
                    H_symbolic.col_ptrs_for_values[child].zx()
                        ..H_symbolic.col_ptrs_for_values[child + 1].zx(),
                );
                let c_H = crate::mat::from_column_major_slice_generic::<'_, E, _, _>(
                    c_H.into_inner(),
                    H_symbolic.col_ptrs_for_row_indices[child + 1].zx() - c_h_row_begin,
                    c_ncols + c_pattern.len(),
                );

                for (idx, &col) in c_pattern.iter().enumerate() {
                    child_col_global_to_local[col.zx()] = I::Signed::truncate(idx + c_ncols);
                }
                for (idx, &p) in c_row_indices_in_panel.iter().enumerate() {
                    child_row_global_to_local[p.zx()] = I::Signed::truncate(idx);
                }

                for s_idx in 0..s_h_row_end - s_h_row_begin {
                    let i = s_row_indices_in_panel[s_idx].zx();
                    let c_idx = child_row_global_to_local[i];
                    if c_idx < I::Signed::truncate(0) {
                        continue;
                    }

                    let c_idx = c_idx.zx();
                    let c_min_col = c_min_col_in_panel[c_idx].zx();

                    for (j_idx_in_c, j) in c_pattern.iter().enumerate() {
                        let j_idx_in_c = j_idx_in_c + c_ncols;
                        if j.zx() >= c_min_col {
                            s_H.write(
                                s_idx,
                                col_global_to_local[j.zx()].zx(),
                                c_H.read(c_idx, j_idx_in_c),
                            );
                        }
                    }
                }

                for &row in c_row_indices_in_panel {
                    child_row_global_to_local[row.zx()] = I::Signed::truncate(NONE);
                }
                for &col in c_pattern {
                    child_col_global_to_local[col.zx()] = I::Signed::truncate(NONE);
                }
                child_ = child_next[child];
            }

            let s_col_local_to_global = |local: usize| {
                if local < s_ncols {
                    s_col_begin + local
                } else {
                    s_pattern[local - s_ncols].zx()
                }
            };

            {
                let s_h_nrows = s_h_row_end - s_h_row_begin;

                let tau_begin = H_symbolic.col_ptrs_for_tau_values[s].zx();
                let tau_end = H_symbolic.col_ptrs_for_tau_values[s + 1].zx();
                let L_begin = L_symbolic.col_ptrs_for_values()[s].zx();
                let L_end = L_symbolic.col_ptrs_for_values()[s + 1].zx();

                let s_tau = tau_values.rb_mut().subslice(tau_begin..tau_end);
                let s_L = L_values.rb_mut().subslice(L_begin..L_end);

                let max_blocksize = H_symbolic.max_blocksize[s].zx();
                let mut s_tau = crate::mat::from_column_major_slice_mut_generic::<'_, E, _, _>(
                    s_tau.into_inner(),
                    max_blocksize,
                    Ord::min(s_H.ncols(), s_h_row_full_end - s_h_row_begin),
                );

                {
                    let mut current_min_col = 0usize;
                    let mut current_start = 0usize;
                    for idx in 0..s_h_nrows + 1 {
                        let idx_global_min_col = if idx < s_h_nrows {
                            s_min_col_in_panel[idx].zx()
                        } else {
                            n
                        };

                        let idx_min_col = if idx_global_min_col < n {
                            col_global_to_local[idx_global_min_col.zx()].zx()
                        } else {
                            s_H.ncols()
                        };

                        if idx_min_col == s_H.ncols()
                            || idx_min_col
                                >= current_min_col.saturating_add(Ord::max(1, max_blocksize / 2))
                        {
                            let nrows = idx.saturating_sub(current_start);
                            let full_ncols = s_H.ncols() - current_start;
                            let ncols = Ord::min(nrows, idx_min_col - current_min_col);

                            let s_H = s_H.rb_mut().submatrix_mut(
                                current_start,
                                current_start,
                                nrows,
                                full_ncols,
                            );

                            let (mut left, mut right) = s_H.split_at_col_mut(ncols);
                            let bs = crate::linalg::qr::no_pivoting::compute::recommended_blocksize::<
                                Symbolic,
                            >(left.nrows(), left.ncols());
                            let bs = Ord::min(max_blocksize, bs);
                            tau_blocksize[block_count] = I::truncate(bs);
                            householder_nrows[block_count] = I::truncate(nrows);
                            householder_ncols[block_count] = I::truncate(ncols);
                            block_count += 1;

                            let mut s_tau = s_tau
                                .rb_mut()
                                .subrows_mut(0, bs)
                                .subcols_mut(current_start, ncols);

                            crate::linalg::qr::no_pivoting::compute::qr_in_place(
                                left.rb_mut(),
                                s_tau.rb_mut(),
                                parallelism,
                                stack.rb_mut(),
                                Default::default(),
                            );

                            if right.ncols() > 0 {
                                apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                                    left.rb(),
                                    s_tau.rb(),
                                    crate::Conj::Yes,
                                    right.rb_mut(),
                                    parallelism,
                                    stack.rb_mut(),
                                );
                            }

                            current_min_col = idx_min_col;
                            current_start += ncols;
                        }
                    }
                }

                let mut s_L = crate::mat::from_column_major_slice_mut_generic::<'_, E, _, _>(
                    s_L.into_inner(),
                    s_pattern.len() + s_ncols,
                    s_ncols,
                );
                let nrows = Ord::min(s_H.nrows(), s_L.ncols());
                zipped!(
                    s_L.rb_mut().transpose_mut().subrows_mut(0, nrows),
                    s_H.rb().subrows(0, nrows)
                )
                .for_each_triangular_upper(
                    crate::linalg::zip::Diag::Include,
                    |unzipped!(mut dst, src)| dst.write(src.read().faer_conj()),
                );
            }

            col_end_for_row_indices_in_panel[s] = Ord::min(
                I::truncate(s_h_row_begin + s_ncols + s_pattern.len()),
                col_end_for_row_indices_in_panel[s],
            );

            let s_h_row_end = col_end_for_row_indices_in_panel[s].zx();
            let s_h_nrows = s_h_row_end - s_h_row_begin;

            let mut current_min_col = 0usize;
            for idx in 0..s_h_nrows {
                let idx_global_min_col = s_min_col_in_panel[idx];
                if idx_global_min_col.zx() >= n {
                    break;
                }
                let idx_min_col = col_global_to_local[idx_global_min_col.zx()].zx();
                if current_min_col > idx_min_col {
                    s_min_col_in_panel[idx] = I::truncate(s_col_local_to_global(current_min_col));
                }
                current_min_col += 1;
            }

            let s_pivot_row_end = s_ncols;

            let parent = super_etree[s];
            if parent.to_signed() < I::Signed::truncate(0) {
                for i in 0..s_ncols {
                    col_global_to_local[s_col_begin + i] = I::Signed::truncate(NONE);
                }
                for &row in s_pattern {
                    col_global_to_local[row.zx()] = I::Signed::truncate(NONE);
                }
                continue;
            }
            let parent = parent.zx();
            let p_h_row_begin = H_symbolic.col_ptrs_for_row_indices[parent].zx();
            let mut pos = col_end_for_row_indices_in_panel[parent].zx() - p_h_row_begin;
            let parent_min_col_in_panel =
                &mut parent_min_col_in_panel[p_h_row_begin - parent_offset..];
            let parent_row_indices_in_panel =
                &mut parent_row_indices_in_panel[p_h_row_begin - parent_offset..];

            for idx in s_pivot_row_end..s_h_nrows {
                parent_row_indices_in_panel[pos] = s_row_indices_in_panel[idx];
                parent_min_col_in_panel[pos] = s_min_col_in_panel[idx];
                pos += 1;
            }
            col_end_for_row_indices_in_panel[parent] = I::truncate(pos + p_h_row_begin);

            for i in 0..s_ncols {
                col_global_to_local[s_col_begin + i] = I::Signed::truncate(NONE);
            }
            for &row in s_pattern {
                col_global_to_local[row.zx()] = I::Signed::truncate(NONE);
            }
        }
        block_count
    }
}

/// Simplicial factorization module.
///
/// A simplicial factorization is one that processes the elements of the QR factors of the
/// input matrix by single elements, rather than by blocks. This is more efficient if the QR factors
/// are very sparse.
pub mod simplicial {
    use super::*;
    use crate::{assert, sparse::linalg::triangular_solve};

    /// Symbolic structure of the QR decomposition,
    #[derive(Debug)]
    pub struct SymbolicSimplicialQr<I> {
        nrows: usize,
        ncols: usize,
        h_nnz: usize,
        l_nnz: usize,

        postorder: alloc::vec::Vec<I>,
        postorder_inv: alloc::vec::Vec<I>,
        desc_count: alloc::vec::Vec<I>,
    }

    impl<I: Index> SymbolicSimplicialQr<I> {
        /// Returns the number of rows of the matrix $A$.
        #[inline]
        pub fn nrows(&self) -> usize {
            self.nrows
        }
        /// Returns the number of columns of the matrix $A$.
        #[inline]
        pub fn ncols(&self) -> usize {
            self.ncols
        }

        /// Returns the length of the slice that can be used to contain the Householder factors.
        #[inline]
        pub fn len_householder(&self) -> usize {
            self.h_nnz
        }
        /// Returns the length of the slice that can be used to contain the $R$ factor.
        #[inline]
        pub fn len_r(&self) -> usize {
            self.l_nnz
        }
    }

    /// QR factors containing both the symbolic and numeric representations.
    #[derive(Debug)]
    pub struct SimplicialQrRef<'a, I, E: Entity> {
        symbolic: &'a SymbolicSimplicialQr<I>,
        r_col_ptrs: &'a [I],
        r_row_indices: &'a [I],
        r_values: SliceGroup<'a, E>,
        householder_col_ptrs: &'a [I],
        householder_row_indices: &'a [I],
        householder_values: SliceGroup<'a, E>,
        tau_values: SliceGroup<'a, E>,
    }

    impl<I, E: Entity> Copy for SimplicialQrRef<'_, I, E> {}
    impl<I, E: Entity> Clone for SimplicialQrRef<'_, I, E> {
        #[inline]
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<'a, I: Index, E: Entity> SimplicialQrRef<'a, I, E> {
        /// Creates QR factors from their components.
        #[inline]
        pub fn new(
            symbolic: &'a SymbolicSimplicialQr<I>,
            r: SparseColMatRef<'a, I, E>,
            householder: SparseColMatRef<'a, I, E>,
            tau_values: GroupFor<E, &'a [E::Unit]>,
        ) -> Self {
            assert!(householder.nrows() == symbolic.nrows);
            assert!(householder.ncols() == symbolic.ncols);
            assert!(r.nrows() == symbolic.ncols);
            assert!(r.ncols() == symbolic.ncols);

            let r_col_ptrs = r.col_ptrs();
            let r_row_indices = r.row_indices();
            let r_values = r.values();
            assert!(r.nnz_per_col().is_none());

            let householder_col_ptrs = householder.col_ptrs();
            let householder_row_indices = householder.row_indices();
            let householder_values = householder.values();
            assert!(householder.nnz_per_col().is_none());

            let r_values = SliceGroup::new(r_values);
            let householder_values = SliceGroup::new(householder_values);
            let tau_values = SliceGroup::new(tau_values);

            assert!(r_values.len() == symbolic.len_r());
            assert!(tau_values.len() == symbolic.ncols);
            assert!(householder_values.len() == symbolic.len_householder());
            Self {
                symbolic,
                householder_values,
                tau_values,
                r_values,
                r_col_ptrs,
                r_row_indices,
                householder_col_ptrs,
                householder_row_indices,
            }
        }

        /// Returns the symbolic structure of the QR factorization.
        #[inline]
        pub fn symbolic(&self) -> &SymbolicSimplicialQr<I> {
            self.symbolic
        }

        /// Returns the numerical values of the factor $R$ of the QR factorization.
        #[inline]
        pub fn r_values(self) -> GroupFor<E, &'a [E::Unit]> {
            self.r_values.into_inner()
        }

        /// Returns the numerical values of the Householder factors of the QR factorization.
        #[inline]
        pub fn householder_values(self) -> GroupFor<E, &'a [E::Unit]> {
            self.householder_values.into_inner()
        }

        /// Returns the numerical values of the $T$ factors of the QR factorization.
        #[inline]
        pub fn tau_values(self) -> GroupFor<E, &'a [E::Unit]> {
            self.tau_values.into_inner()
        }

        /// Solves the equation $\text{Op}(A) x = \text{rhs}$ in the sense of least squares, where
        /// $\text{Op}$ is either the identity or the conjugate, depending on the value of `conj`,
        /// and stores the result in the upper part of `rhs`.
        ///
        /// `work` is a temporary workspace with the same dimensions as `rhs`.
        #[track_caller]
        pub fn solve_in_place_with_conj(
            &self,
            conj: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            work: MatMut<'_, E>,
        ) where
            E: ComplexField,
        {
            let _ = parallelism;
            assert!(rhs.nrows() == self.symbolic.nrows);
            let mut x = rhs;

            let m = self.symbolic.nrows;
            let n = self.symbolic.ncols;

            let r = SparseColMatRef::<'_, I, E>::new(
                unsafe {
                    SymbolicSparseColMatRef::new_unchecked(
                        n,
                        n,
                        self.r_col_ptrs,
                        None,
                        self.r_row_indices,
                    )
                },
                self.r_values.into_inner(),
            );
            let h = SparseColMatRef::<'_, I, E>::new(
                unsafe {
                    SymbolicSparseColMatRef::new_unchecked(
                        m,
                        n,
                        self.householder_col_ptrs,
                        None,
                        self.householder_row_indices,
                    )
                },
                self.householder_values.into_inner(),
            );
            let tau = self.tau_values;

            let mut tmp = work;
            tmp.fill_zero();

            // x <- Q^T x
            {
                for j in 0..n {
                    let hi = h.row_indices_of_col_raw(j);
                    let hx = SliceGroup::<'_, E>::new(h.values_of_col(j));
                    let tau_inv = tau.read(j).faer_real().faer_inv();

                    if hi.is_empty() {
                        tmp.rb_mut().row_mut(j).fill_zero();
                        continue;
                    }

                    let hi0 = hi[0].zx();
                    for k in 0..x.ncols() {
                        let mut dot = E::faer_zero();
                        for (i, v) in zip(hi, hx.into_ref_iter()) {
                            let i = i.zx();
                            let v = if conj == Conj::Yes {
                                v.read()
                            } else {
                                v.read().faer_conj()
                            };
                            dot = dot.faer_add(E::faer_mul(v, x.read(i, k)));
                        }
                        dot = dot.faer_scale_real(tau_inv);
                        for (i, v) in zip(hi, hx.into_ref_iter()) {
                            let i = i.zx();
                            let v = if conj == Conj::Yes {
                                v.read().faer_conj()
                            } else {
                                v.read()
                            };
                            x.write(i, k, x.read(i, k).faer_sub(E::faer_mul(dot, v)));
                        }

                        tmp.rb_mut().row_mut(j).copy_from(x.rb().row(hi0));
                    }
                }
            }
            x.rb_mut()
                .subrows_mut(0, n)
                .copy_from(tmp.rb().subrows(0, n));
            x.rb_mut().subrows_mut(n, m - n).fill_zero();

            triangular_solve::solve_upper_triangular_in_place(
                r,
                conj,
                x.rb_mut().subrows_mut(0, n),
                parallelism,
            );
        }
    }

    /// Computes the size and alignment of the workspace required to compute the symbolic QR
    /// factorization of a matrix with dimensions `(nrows, ncols)`.
    pub fn factorize_simplicial_symbolic_qr_req<I: Index>(
        nrows: usize,
        ncols: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let _ = nrows;
        StackReq::try_all_of([StackReq::try_new::<I>(ncols)?; 3])
    }

    /// Computes the symbolic QR factorization of a matrix $A$, given the outputs of the
    /// pre-factorization steps.
    pub fn factorize_simplicial_symbolic_qr<I: Index>(
        min_col: &[I],
        etree: EliminationTreeRef<'_, I>,
        col_counts: &[I],
        stack: &mut PodStack,
    ) -> Result<SymbolicSimplicialQr<I>, FaerError> {
        let m = min_col.len();
        let n = col_counts.len();

        let mut post = try_zeroed::<I>(n)?;
        let mut post_inv = try_zeroed::<I>(n)?;
        let mut desc_count = try_zeroed::<I>(n)?;

        let h_non_zero_count = &mut *post_inv;
        for i in 0..m {
            let min_col = min_col[i];
            if min_col.to_signed() < I::Signed::truncate(0) {
                continue;
            }
            h_non_zero_count[min_col.zx()] += I::truncate(1);
        }
        for j in 0..n {
            let parent = etree.inner[j];
            if parent < I::Signed::truncate(0) || h_non_zero_count[j] == I::truncate(0) {
                continue;
            }
            h_non_zero_count[parent.zx()] += h_non_zero_count[j] - I::truncate(1);
        }

        let h_nnz = I::sum_nonnegative(h_non_zero_count)
            .ok_or(FaerError::IndexOverflow)?
            .zx();
        let l_nnz = I::sum_nonnegative(col_counts)
            .ok_or(FaerError::IndexOverflow)?
            .zx();

        postorder(&mut post, etree, stack);
        for (i, p) in post.iter().enumerate() {
            post_inv[p.zx()] = I::truncate(i);
        }
        for j in 0..n {
            let parent = etree.inner[j];
            if parent >= I::Signed::truncate(0) {
                desc_count[parent.zx()] = desc_count[parent.zx()] + desc_count[j] + I::truncate(1);
            }
        }

        Ok(SymbolicSimplicialQr {
            nrows: m,
            ncols: n,
            postorder: post,
            postorder_inv: post_inv,
            desc_count,
            h_nnz,
            l_nnz,
        })
    }

    /// Computes the size and alignment of the workspace required to compute the numerical QR
    /// factorization of the matrix whose structure was used to produce the symbolic structure.
    pub fn factorize_simplicial_numeric_qr_req<I: Index, E: Entity>(
        symbolic: &SymbolicSimplicialQr<I>,
    ) -> Result<StackReq, SizeOverflow> {
        let m = symbolic.nrows;
        StackReq::try_all_of([
            StackReq::try_new::<I>(m)?,
            StackReq::try_new::<I>(m)?,
            StackReq::try_new::<I>(m)?,
            crate::sparse::linalg::make_raw_req::<E>(m)?,
        ])
    }

    /// Computes the numerical QR factorization of $A$.
    ///
    /// - `r_col_ptrs` has length `A.ncols() + 1`.
    /// - `r_row_indices` has length `symbolic.len_r()`.
    /// - `r_values` has length `symbolic.len_r()`.
    /// - `householder_col_ptrs` has length `A.ncols() + 1`.
    /// - `householder_row_indices` has length `symbolic.len_householder()`.
    /// - `householder_values` has length `symbolic.len_householder()`.
    /// - `tau_values` has length `A.ncols()`.
    pub fn factorize_simplicial_numeric_qr_unsorted<'a, I: Index, E: ComplexField>(
        r_col_ptrs: &'a mut [I],
        r_row_indices: &'a mut [I],
        r_values: GroupFor<E, &'a mut [E::Unit]>,
        householder_col_ptrs: &'a mut [I],
        householder_row_indices: &'a mut [I],
        householder_values: GroupFor<E, &'a mut [E::Unit]>,
        tau_values: GroupFor<E, &'a mut [E::Unit]>,

        A: SparseColMatRef<'_, I, E>,
        col_perm: Option<PermRef<'_, I>>,
        symbolic: &'a SymbolicSimplicialQr<I>,
        stack: &mut PodStack,
    ) -> SimplicialQrRef<'a, I, E> {
        assert!(all(
            A.nrows() == symbolic.nrows,
            A.ncols() == symbolic.ncols,
        ));

        let I = I::truncate;
        let m = A.nrows();
        let n = A.ncols();
        let (r_idx, stack) = stack.make_raw::<I::Signed>(m);
        let (marked, stack) = stack.make_raw::<I>(m);
        let (pattern, stack) = stack.make_raw::<I>(m);
        let (mut x, _) = crate::sparse::linalg::make_raw::<E>(m, stack);
        x.fill_zero();
        super::mem::fill_zero(marked);
        super::mem::fill_none(r_idx);

        let mut r_values = SliceGroupMut::<'_, E>::new(r_values);
        let mut householder_values = SliceGroupMut::<'_, E>::new(householder_values);
        let mut tau_values = SliceGroupMut::<'_, E>::new(tau_values);

        r_col_ptrs[0] = I(0);
        let mut r_pos = 0usize;
        let mut h_pos = 0usize;
        for j in 0..n {
            let pj = col_perm.map(|perm| perm.arrays().0[j].zx()).unwrap_or(j);

            let mut pattern_len = 0usize;
            for (i, val) in zip(
                A.row_indices_of_col(pj),
                SliceGroup::<'_, E>::new(A.values_of_col(pj)).into_ref_iter(),
            ) {
                if marked[i] < I(j + 1) {
                    marked[i] = I(j + 1);
                    pattern[pattern_len] = I(i);
                    pattern_len += 1;
                }
                x.write(i, x.read(i).faer_add(val.read()));
            }

            let j_postordered = symbolic.postorder_inv[j].zx();
            let desc_count = symbolic.desc_count[j].zx();
            for d in &symbolic.postorder[j_postordered - desc_count..j_postordered] {
                let d = d.zx();

                let d_h_pattern = &householder_row_indices
                    [householder_col_ptrs[d].zx()..householder_col_ptrs[d + 1].zx()];
                let d_h_values = householder_values
                    .rb()
                    .subslice(householder_col_ptrs[d].zx()..householder_col_ptrs[d + 1].zx());

                let mut intersects = false;
                for i in d_h_pattern {
                    if marked[i.zx()] == I(j + 1) {
                        intersects = true;
                        break;
                    }
                }
                if !intersects {
                    continue;
                }

                for i in d_h_pattern {
                    let i = i.zx();
                    if marked[i] < I(j + 1) {
                        marked[i] = I(j + 1);
                        pattern[pattern_len] = I(i);
                        pattern_len += 1;
                    }
                }

                let tau_inv = tau_values.read(d).faer_real().faer_inv();
                let mut dot = E::faer_zero();
                for (i, vi) in zip(d_h_pattern, d_h_values.into_ref_iter()) {
                    let i = i.zx();
                    let vi = vi.read().faer_conj();
                    dot = dot.faer_add(E::faer_mul(vi, x.read(i)));
                }
                dot = dot.faer_scale_real(tau_inv);
                for (i, vi) in zip(d_h_pattern, d_h_values.into_ref_iter()) {
                    let i = i.zx();
                    let vi = vi.read();
                    x.write(i, x.read(i).faer_sub(E::faer_mul(dot, vi)));
                }
            }
            let pattern = &pattern[..pattern_len];

            let h_begin = h_pos;
            for i in pattern.iter() {
                let i = i.zx();
                if r_idx[i] >= I(0).to_signed() {
                    r_values.write(r_pos, x.read(i));
                    x.write(i, E::faer_zero());
                    r_row_indices[r_pos] = I::from_signed(r_idx[i]);
                    r_pos += 1;
                } else {
                    householder_values.write(h_pos, x.read(i));
                    x.write(i, E::faer_zero());
                    householder_row_indices[h_pos] = I(i);
                    h_pos += 1;
                }
            }

            householder_col_ptrs[j + 1] = I(h_pos);

            if h_begin == h_pos {
                tau_values.write(j, E::faer_zero());
                r_values.write(r_pos, E::faer_zero());
                r_row_indices[r_pos] = I(j);
                r_pos += 1;
                r_col_ptrs[j + 1] = I(r_pos);
                continue;
            }

            let mut h_col = crate::col::from_slice_mut_generic::<E>(
                householder_values
                    .rb_mut()
                    .subslice(h_begin..h_pos)
                    .into_inner(),
            );

            let (mut head, tail) = h_col.rb_mut().split_at_mut(1);
            let tail_norm = tail.norm_l2();
            let (tau, beta) = crate::linalg::householder::make_householder_in_place(
                Some(tail),
                head.read(0),
                tail_norm,
            );
            head.write(0, E::faer_one());
            tau_values.write(j, tau);
            r_values.write(r_pos, beta);
            r_row_indices[r_pos] = I(j);
            r_idx[householder_row_indices[h_begin].zx()] = I(j).to_signed();
            r_pos += 1;
            r_col_ptrs[j + 1] = I(r_pos);
        }

        unsafe {
            SimplicialQrRef::new(
                symbolic,
                SparseColMatRef::<'_, I, E>::new(
                    SymbolicSparseColMatRef::new_unchecked(n, n, r_col_ptrs, None, r_row_indices),
                    r_values.into_const().into_inner(),
                ),
                SparseColMatRef::<'_, I, E>::new(
                    SymbolicSparseColMatRef::new_unchecked(
                        m,
                        n,
                        householder_col_ptrs,
                        None,
                        householder_row_indices,
                    ),
                    householder_values.into_const().into_inner(),
                ),
                tau_values.into_const().into_inner(),
            )
        }
    }
}

/// Tuning parameters for the QR symbolic factorization.
#[derive(Copy, Clone, Debug, Default)]
pub struct QrSymbolicParams<'a> {
    /// Parameters for the fill reducing column permutation
    pub colamd_params: Control,
    /// Threshold for selecting the supernodal factorization.
    pub supernodal_flop_ratio_threshold: SupernodalThreshold,
    /// Supernodal factorization parameters.
    pub supernodal_params: SymbolicSupernodalParams<'a>,
}

/// The inner factorization used for the symbolic QR, either simplicial or symbolic.
#[derive(Debug)]
pub enum SymbolicQrRaw<I: Index> {
    /// Simplicial structure.
    Simplicial(simplicial::SymbolicSimplicialQr<I>),
    /// Supernodal structure.
    Supernodal(supernodal::SymbolicSupernodalQr<I>),
}

/// The symbolic structure of a sparse QR decomposition.
#[derive(Debug)]
pub struct SymbolicQr<I: Index> {
    raw: SymbolicQrRaw<I>,
    col_perm_fwd: alloc::vec::Vec<I>,
    col_perm_inv: alloc::vec::Vec<I>,
    A_nnz: usize,
}

/// Sparse QR factorization wrapper.
#[derive(Debug)]
pub struct QrRef<'a, I: Index, E: Entity> {
    symbolic: &'a SymbolicQr<I>,
    indices: &'a [I],
    values: SliceGroup<'a, E>,
}
impl_copy!(<'a><I: Index, E: Entity><QrRef<'a, I, E>>);

impl<'a, I: Index, E: Entity> QrRef<'a, I, E> {
    /// Creates a QR decomposition reference from its symbolic and numerical components.
    ///
    /// # Safety:
    /// The indices must be filled by a previous call to [`SymbolicQr::factorize_numeric_qr`] with
    /// the right parameters.
    #[inline]
    pub unsafe fn new_unchecked(
        symbolic: &'a SymbolicQr<I>,
        indices: &'a [I],
        values: GroupFor<E, &'a [E::Unit]>,
    ) -> Self {
        let values = SliceGroup::<'_, E>::new(values);
        assert!(all(
            symbolic.len_values() == values.len(),
            symbolic.len_indices() == indices.len(),
        ));
        Self {
            symbolic,
            values,
            indices,
        }
    }

    /// Returns the symbolic structure of the QR factorization.
    #[inline]
    pub fn symbolic(self) -> &'a SymbolicQr<I> {
        self.symbolic
    }

    /// Solves the equation $\text{Op}(A) x = \text{rhs}$ in the sense of least squares, where
    /// $\text{Op}$ is either the identity or the conjugate, depending on the value of `conj`,
    /// and stores the result in the upper part of `rhs`.
    ///
    /// `work` is a temporary workspace with the same dimensions as `rhs`.
    #[track_caller]
    pub fn solve_in_place_with_conj(
        self,
        conj: Conj,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) where
        E: ComplexField,
    {
        let k = rhs.ncols();
        let m = self.symbolic.nrows();
        let n = self.symbolic.ncols();

        assert!(all(
            rhs.nrows() == self.symbolic.nrows(),
            self.symbolic.nrows() >= self.symbolic.ncols(),
        ));
        let mut rhs = rhs;

        let (mut x, stack) = temp_mat_uninit::<E>(m, k, stack);

        let (_, inv) = self.symbolic.col_perm().arrays();
        x.copy_from(rhs.rb());

        let indices = self.indices;
        let values = self.values;

        match &self.symbolic.raw {
            SymbolicQrRaw::Simplicial(symbolic) => {
                let (r_col_ptrs, indices) = indices.split_at(n + 1);
                let (r_row_indices, indices) = indices.split_at(symbolic.len_r());
                let (householder_col_ptrs, indices) = indices.split_at(n + 1);
                let (householder_row_indices, _) = indices.split_at(symbolic.len_householder());

                let (r_values, values) = values.rb().split_at(symbolic.len_r());
                let (householder_values, values) = values.split_at(symbolic.len_householder());
                let (tau_values, _) = values.split_at(n);

                let r = SparseColMatRef::<'_, I, E>::new(
                    unsafe {
                        SymbolicSparseColMatRef::new_unchecked(
                            n,
                            n,
                            r_col_ptrs,
                            None,
                            r_row_indices,
                        )
                    },
                    r_values.into_inner(),
                );
                let h = SparseColMatRef::<'_, I, E>::new(
                    unsafe {
                        SymbolicSparseColMatRef::new_unchecked(
                            m,
                            n,
                            householder_col_ptrs,
                            None,
                            householder_row_indices,
                        )
                    },
                    householder_values.into_inner(),
                );

                let this = simplicial::SimplicialQrRef::<'_, I, E>::new(
                    symbolic,
                    r,
                    h,
                    tau_values.into_inner(),
                );
                this.solve_in_place_with_conj(conj, x.rb_mut(), parallelism, rhs.rb_mut());
            }
            SymbolicQrRaw::Supernodal(symbolic) => {
                let (householder_row_indices, indices) =
                    indices.split_at(symbolic.householder().len_householder_row_indices());
                let (tau_blocksize, indices) = indices.split_at(
                    symbolic.householder().len_householder_row_indices()
                        + symbolic.householder().n_supernodes(),
                );
                let (householder_nrows, indices) = indices.split_at(
                    symbolic.householder().len_householder_row_indices()
                        + symbolic.householder().n_supernodes(),
                );
                let (householder_ncols, _) = indices.split_at(
                    symbolic.householder().len_householder_row_indices()
                        + symbolic.householder().n_supernodes(),
                );

                let (r_values, values) = values.rb().split_at(symbolic.r_adjoint().len_values());
                let (householder_values, values) =
                    values.split_at(symbolic.householder().len_householder_values());
                let (tau_values, _) = values.split_at(symbolic.householder().len_tau_values());

                let this = unsafe {
                    supernodal::SupernodalQrRef::<'_, I, E>::new_unchecked(
                        symbolic,
                        householder_row_indices,
                        tau_blocksize,
                        householder_nrows,
                        householder_ncols,
                        r_values.into_inner(),
                        householder_values.into_inner(),
                        tau_values.into_inner(),
                    )
                };
                this.solve_in_place_with_conj(conj, x.rb_mut(), parallelism, rhs.rb_mut(), stack);
            }
        }

        for j in 0..k {
            for (i, p) in inv.iter().enumerate() {
                rhs.write(i, j, x.read(p.zx(), j));
            }
        }
    }
}

impl<I: Index> SymbolicQr<I> {
    /// Number of rows of $A$.
    #[inline]
    pub fn nrows(&self) -> usize {
        match &self.raw {
            SymbolicQrRaw::Simplicial(this) => this.nrows(),
            SymbolicQrRaw::Supernodal(this) => this.householder().nrows(),
        }
    }

    /// Number of columns of $A$.
    #[inline]
    pub fn ncols(&self) -> usize {
        match &self.raw {
            SymbolicQrRaw::Simplicial(this) => this.ncols(),
            SymbolicQrRaw::Supernodal(this) => this.r_adjoint().ncols(),
        }
    }

    /// Returns the fill-reducing column permutation that was computed during symbolic analysis.
    #[inline]
    pub fn col_perm(&self) -> PermRef<'_, I> {
        unsafe { PermRef::new_unchecked(&self.col_perm_fwd, &self.col_perm_inv, self.ncols()) }
    }

    /// Returns the length of the slice needed to store the symbolic indices of the QR
    /// decomposition.
    #[inline]
    pub fn len_indices(&self) -> usize {
        match &self.raw {
            SymbolicQrRaw::Simplicial(symbolic) => {
                symbolic.len_r() + symbolic.len_householder() + 2 * self.ncols() + 2
            }
            SymbolicQrRaw::Supernodal(symbolic) => {
                4 * symbolic.householder().len_householder_row_indices()
                    + 3 * symbolic.householder().n_supernodes()
            }
        }
    }

    /// Returns the length of the slice needed to store the numerical values of the QR
    /// decomposition.
    #[inline]
    pub fn len_values(&self) -> usize {
        match &self.raw {
            SymbolicQrRaw::Simplicial(symbolic) => {
                symbolic.len_r() + symbolic.len_householder() + self.ncols()
            }
            SymbolicQrRaw::Supernodal(symbolic) => {
                symbolic.householder().len_householder_values()
                    + symbolic.r_adjoint().len_values()
                    + symbolic.householder().len_tau_values()
            }
        }
    }

    /// Returns the size and alignment of the workspace required to solve the system $Ax =
    /// \text{rhs}$ in the sense of least squares.
    pub fn solve_in_place_req<E: Entity>(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        temp_mat_req::<E>(self.nrows(), rhs_ncols)?.try_and(match &self.raw {
            SymbolicQrRaw::Simplicial(_) => StackReq::empty(),
            SymbolicQrRaw::Supernodal(this) => {
                this.solve_in_place_req::<E>(rhs_ncols, parallelism)?
            }
        })
    }

    /// Computes the required workspace size and alignment for a numerical QR factorization.
    pub fn factorize_numeric_qr_req<E: Entity>(
        &self,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let m = self.nrows();
        let A_nnz = self.A_nnz;
        let AT_req = StackReq::try_all_of([
            crate::sparse::linalg::make_raw_req::<E>(A_nnz)?,
            StackReq::try_new::<I>(m + 1)?,
            StackReq::try_new::<I>(A_nnz)?,
        ])?;

        match &self.raw {
            SymbolicQrRaw::Simplicial(symbolic) => {
                simplicial::factorize_simplicial_numeric_qr_req::<I, E>(symbolic)
            }
            SymbolicQrRaw::Supernodal(symbolic) => StackReq::try_and(
                AT_req,
                supernodal::factorize_supernodal_numeric_qr_req::<I, E>(symbolic, parallelism)?,
            ),
        }
    }

    /// Computes a numerical QR factorization of A.
    #[track_caller]
    pub fn factorize_numeric_qr<'out, E: ComplexField>(
        &'out self,
        indices: &'out mut [I],
        values: GroupFor<E, &'out mut [E::Unit]>,
        A: SparseColMatRef<'_, I, E>,
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) -> QrRef<'out, I, E> {
        let mut values = SliceGroupMut::<'_, E>::new(values);
        assert!(all(
            values.len() == self.len_values(),
            indices.len() == self.len_indices(),
        ));
        assert!(all(A.nrows() == self.nrows(), A.ncols() == self.ncols()));

        let m = A.nrows();
        let n = A.ncols();

        match &self.raw {
            SymbolicQrRaw::Simplicial(symbolic) => {
                let (r_col_ptrs, indices) = indices.split_at_mut(n + 1);
                let (r_row_indices, indices) = indices.split_at_mut(symbolic.len_r());
                let (householder_col_ptrs, indices) = indices.split_at_mut(n + 1);
                let (householder_row_indices, _) = indices.split_at_mut(symbolic.len_householder());

                let (r_values, values) = values.rb_mut().split_at(symbolic.len_r());
                let (householder_values, values) = values.split_at(symbolic.len_householder());
                let (tau_values, _) = values.split_at(n);

                simplicial::factorize_simplicial_numeric_qr_unsorted::<I, E>(
                    r_col_ptrs,
                    r_row_indices,
                    r_values.into_inner(),
                    householder_col_ptrs,
                    householder_row_indices,
                    householder_values.into_inner(),
                    tau_values.into_inner(),
                    A,
                    Some(self.col_perm()),
                    symbolic,
                    stack,
                );
            }
            SymbolicQrRaw::Supernodal(symbolic) => {
                let (householder_row_indices, indices) =
                    indices.split_at_mut(symbolic.householder().len_householder_row_indices());
                let (tau_blocksize, indices) = indices.split_at_mut(
                    symbolic.householder().len_householder_row_indices()
                        + symbolic.householder().n_supernodes(),
                );
                let (householder_nrows, indices) = indices.split_at_mut(
                    symbolic.householder().len_householder_row_indices()
                        + symbolic.householder().n_supernodes(),
                );
                let (householder_ncols, _) = indices.split_at_mut(
                    symbolic.householder().len_householder_row_indices()
                        + symbolic.householder().n_supernodes(),
                );

                let (r_values, values) =
                    values.rb_mut().split_at(symbolic.r_adjoint().len_values());
                let (householder_values, values) =
                    values.split_at(symbolic.householder().len_householder_values());
                let (tau_values, _) = values.split_at(symbolic.householder().len_tau_values());

                let (new_col_ptr, stack) = stack.make_raw::<I>(m + 1);
                let (new_row_ind, stack) = stack.make_raw::<I>(self.A_nnz);
                let (new_values, mut stack) =
                    crate::sparse::linalg::make_raw::<E>(self.A_nnz, stack);

                let AT = crate::sparse::utils::transpose(
                    new_col_ptr,
                    new_row_ind,
                    new_values.into_inner(),
                    A,
                    stack.rb_mut(),
                )
                .into_const();

                supernodal::factorize_supernodal_numeric_qr::<I, E>(
                    householder_row_indices,
                    tau_blocksize,
                    householder_nrows,
                    householder_ncols,
                    r_values.into_inner(),
                    householder_values.into_inner(),
                    tau_values.into_inner(),
                    AT,
                    Some(self.col_perm()),
                    symbolic,
                    parallelism,
                    stack,
                );
            }
        }

        unsafe { QrRef::new_unchecked(self, indices, values.into_const().into_inner()) }
    }
}

/// Computes the symbolic QR factorization of the matrix `A`, or returns an error if the
/// operation could not be completed.
#[track_caller]
pub fn factorize_symbolic_qr<I: Index>(
    A: SymbolicSparseColMatRef<'_, I>,
    params: QrSymbolicParams<'_>,
) -> Result<SymbolicQr<I>, FaerError> {
    assert!(A.nrows() >= A.ncols());
    let m = A.nrows();
    let n = A.ncols();
    let A_nnz = A.compute_nnz();

    with_dim!(M, m);
    with_dim!(N, n);
    let A = ghost::SymbolicSparseColMatRef::new(A, M, N);

    let req = || -> Result<StackReq, SizeOverflow> {
        let n_req = StackReq::try_new::<I>(n)?;
        let m_req = StackReq::try_new::<I>(m)?;
        let AT_req = StackReq::try_and(
            // new_col_ptr
            StackReq::try_new::<I>(m + 1)?,
            // new_row_ind
            StackReq::try_new::<I>(A_nnz)?,
        )?;

        StackReq::try_or(
            colamd::order_req::<I>(m, n, A_nnz)?,
            StackReq::try_all_of([
                n_req,
                n_req,
                n_req,
                n_req,
                AT_req,
                StackReq::try_any_of([
                    StackReq::try_and(n_req, m_req)?,
                    StackReq::try_all_of([n_req; 3])?,
                    StackReq::try_all_of([n_req, n_req, n_req, n_req, n_req, m_req])?,
                    supernodal::factorize_supernodal_symbolic_qr_req::<I>(m, n)?,
                    simplicial::factorize_simplicial_symbolic_qr_req::<I>(m, n)?,
                ])?,
            ])?,
        )
    };

    let req = req().map_err(nomem)?;
    let mut mem = dyn_stack::GlobalPodBuffer::try_new(req).map_err(nomem)?;
    let mut stack = PodStack::new(&mut mem);

    let mut col_perm_fwd = try_zeroed::<I>(n)?;
    let mut col_perm_inv = try_zeroed::<I>(n)?;
    let mut min_row = try_zeroed::<I>(m)?;

    colamd::order(
        &mut col_perm_fwd,
        &mut col_perm_inv,
        A.into_inner(),
        params.colamd_params,
        stack.rb_mut(),
    )?;

    let col_perm = ghost::PermRef::new(PermRef::new_checked(&col_perm_fwd, &col_perm_inv, n), N);

    let (new_col_ptr, stack) = stack.make_raw::<I>(m + 1);
    let (new_row_ind, mut stack) = stack.make_raw::<I>(A_nnz);
    let AT =
        crate::sparse::utils::ghost_adjoint_symbolic(new_col_ptr, new_row_ind, A, stack.rb_mut());

    let (etree, stack) = stack.make_raw::<I::Signed>(n);
    let (post, stack) = stack.make_raw::<I>(n);
    let (col_counts, stack) = stack.make_raw::<I>(n);
    let (h_col_counts, mut stack) = stack.make_raw::<I>(n);

    ghost_col_etree(A, Some(col_perm), Array::from_mut(etree, N), stack.rb_mut());
    let etree_ = Array::from_ref(MaybeIdx::<'_, I>::from_slice_ref_checked(etree, N), N);
    ghost_postorder(Array::from_mut(post, N), etree_, stack.rb_mut());

    ghost_column_counts_aat(
        Array::from_mut(col_counts, N),
        Array::from_mut(bytemuck::cast_slice_mut(&mut min_row), M),
        AT,
        Some(col_perm),
        etree_,
        Array::from_ref(Idx::from_slice_ref_checked(post, N), N),
        stack.rb_mut(),
    );
    let min_col = min_row;

    let mut threshold = params.supernodal_flop_ratio_threshold;
    if threshold != SupernodalThreshold::FORCE_SIMPLICIAL
        && threshold != SupernodalThreshold::FORCE_SUPERNODAL
    {
        mem::fill_zero(h_col_counts);
        for i in 0..m {
            let min_col = min_col[i];
            if min_col.to_signed() < I::Signed::truncate(0) {
                continue;
            }
            h_col_counts[min_col.zx()] += I::truncate(1);
        }
        for j in 0..n {
            let parent = etree[j];
            if parent < I::Signed::truncate(0) || h_col_counts[j] == I::truncate(0) {
                continue;
            }
            h_col_counts[parent.zx()] += h_col_counts[j] - I::truncate(1);
        }

        let mut nnz = 0.0f64;
        let mut flops = 0.0f64;
        for j in 0..n {
            let hj = h_col_counts[j].zx() as f64;
            let rj = col_counts[j].zx() as f64;
            flops += hj + 2.0 * hj * rj;
            nnz += hj + rj;
        }

        if flops / nnz > threshold.0 * crate::sparse::linalg::QR_SUPERNODAL_RATIO_FACTOR {
            threshold = SupernodalThreshold::FORCE_SUPERNODAL;
        } else {
            threshold = SupernodalThreshold::FORCE_SIMPLICIAL;
        }
    }

    if threshold == SupernodalThreshold::FORCE_SUPERNODAL {
        let symbolic = supernodal::factorize_supernodal_symbolic_qr::<I>(
            A.into_inner(),
            Some(col_perm.into_inner()),
            min_col,
            EliminationTreeRef::<'_, I> { inner: etree },
            col_counts,
            stack.rb_mut(),
            params.supernodal_params,
        )?;
        Ok(SymbolicQr {
            raw: SymbolicQrRaw::Supernodal(symbolic),
            col_perm_fwd,
            col_perm_inv,
            A_nnz,
        })
    } else {
        let symbolic = simplicial::factorize_simplicial_symbolic_qr::<I>(
            &min_col,
            EliminationTreeRef::<'_, I> { inner: etree },
            col_counts,
            stack.rb_mut(),
        )?;
        Ok(SymbolicQr {
            raw: SymbolicQrRaw::Simplicial(symbolic),
            col_perm_fwd,
            col_perm_inv,
            A_nnz,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert,
        complex_native::c64,
        sparse::{
            linalg::{
                cholesky::supernodal::SupernodalLdltRef,
                qr::{
                    simplicial::{
                        factorize_simplicial_numeric_qr_req,
                        factorize_simplicial_numeric_qr_unsorted, factorize_simplicial_symbolic_qr,
                    },
                    supernodal::{
                        factorize_supernodal_numeric_qr, factorize_supernodal_numeric_qr_req,
                        factorize_supernodal_symbolic_qr,
                    },
                },
            },
            utils::ghost_adjoint_symbolic,
        },
        Mat,
    };
    use dyn_stack::GlobalPodBuffer;
    use matrix_market_rs::MtxData;
    use rand::{Rng, SeedableRng};

    fn reconstruct_from_supernodal_llt<I: Index, E: ComplexField>(
        symbolic: &SymbolicSupernodalCholesky<I>,
        L_values: GroupFor<E, &[E::Unit]>,
    ) -> Mat<E> {
        let L_values = SliceGroup::<'_, E>::new(L_values);
        let ldlt = SupernodalLdltRef::new(symbolic, L_values.into_inner());
        let n_supernodes = ldlt.symbolic().n_supernodes();
        let n = ldlt.symbolic().nrows();

        let mut dense = Mat::<E>::zeros(n, n);

        for s in 0..n_supernodes {
            let s = ldlt.supernode(s);
            let size = s.matrix().ncols();

            let (Ls_top, Ls_bot) = s.matrix().split_at_row(size);
            dense
                .as_mut()
                .submatrix_mut(s.start(), s.start(), size, size)
                .copy_from(Ls_top);

            for col in 0..size {
                for (i, row) in s.pattern().iter().enumerate() {
                    dense.write(row.zx(), s.start() + col, Ls_bot.read(i, col));
                }
            }
        }

        &dense * dense.adjoint()
    }

    fn test_symbolic_qr<I: Index>() {
        let I = I::truncate;

        let n = 11;
        let col_ptr = &[0, 3, 6, 10, 13, 16, 21, 24, 29, 31, 37, 43].map(I);
        let row_ind = &[
            0, 5, 6, // 0
            1, 2, 7, // 1
            1, 2, 9, 10, // 2
            3, 5, 9, // 3
            4, 7, 10, // 4
            0, 3, 5, 8, 9, // 5
            0, 6, 10, // 6
            1, 4, 7, 9, 10, // 7
            5, 8, // 8
            2, 3, 5, 7, 9, 10, // 9
            2, 4, 6, 7, 9, 10, // 10
        ]
        .map(I);

        let A = SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_ind);
        let zero = I(0);
        let mut etree = vec![zero.to_signed(); n];
        let mut post = vec![zero; n];
        let mut col_counts = vec![zero; n];

        with_dim!(N, n);
        let A = ghost::SymbolicSparseColMatRef::new(A, N, N);
        ghost_col_etree(
            A,
            None,
            Array::from_mut(&mut etree, N),
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(*N + *N))),
        );
        let etree = Array::from_ref(MaybeIdx::from_slice_ref_checked(&etree, N), N);
        ghost_postorder(
            Array::from_mut(&mut post, N),
            etree,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
        );

        let mut min_row = vec![zero.to_signed(); n];
        let mut new_col_ptrs = vec![zero; n + 1];
        let mut new_row_ind = vec![zero; 43];

        let AT = ghost_adjoint_symbolic(
            &mut new_col_ptrs,
            &mut new_row_ind,
            A,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
        );
        ghost_column_counts_aat(
            Array::from_mut(&mut col_counts, N),
            Array::from_mut(&mut min_row, N),
            AT,
            None,
            etree,
            Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
        );

        assert!(
            MaybeIdx::<'_, I>::as_slice_ref(etree.as_ref())
                == [3, 2, 3, 4, 5, 6, 7, 8, 9, 10, NONE]
                    .map(I)
                    .map(I::to_signed)
        );

        assert!(col_counts == [7, 6, 8, 8, 7, 6, 5, 4, 3, 2, 1].map(I));
    }

    fn load_mtx<I: Index>(
        data: MtxData<f64>,
    ) -> (
        usize,
        usize,
        alloc::vec::Vec<I>,
        alloc::vec::Vec<I>,
        alloc::vec::Vec<f64>,
    ) {
        let I = I::truncate;

        let MtxData::Sparse([nrows, ncols], coo_indices, coo_values, _) = data else {
            panic!()
        };

        let m = nrows;
        let n = ncols;
        let mut col_counts = vec![I(0); n];
        let mut col_ptr = vec![I(0); n + 1];

        for &[_, j] in &coo_indices {
            col_counts[j] += I(1);
        }

        for i in 0..n {
            col_ptr[i + 1] = col_ptr[i] + col_counts[i];
        }
        let nnz = col_ptr[n].zx();

        let mut row_ind = vec![I(0); nnz];
        let mut values = vec![0.0; nnz];

        col_counts.copy_from_slice(&col_ptr[..n]);

        for (&[i, j], &val) in zip(&coo_indices, &coo_values) {
            row_ind[col_counts[j].zx()] = I(i);
            values[col_counts[j].zx()] = val;
            col_counts[j] += I(1);
        }

        (m, n, col_ptr, row_ind, values)
    }

    fn sparse_to_dense<I: Index, E: ComplexField>(sparse: SparseColMatRef<'_, I, E>) -> Mat<E> {
        let m = sparse.nrows();
        let n = sparse.ncols();

        let mut dense = Mat::<E>::zeros(m, n);
        let slice_group = SliceGroup::<'_, E>::new;

        for j in 0..n {
            for (i, val) in zip(
                sparse.row_indices_of_col(j),
                slice_group(sparse.values_of_col(j)).into_ref_iter(),
            ) {
                dense.write(i, j, val.read());
            }
        }

        dense
    }

    #[test]
    fn test_numeric_qr_1_no_transpose() {
        type I = usize;
        let I = I::truncate;

        let (m, n, col_ptr, row_ind, values) =
            load_mtx::<usize>(MtxData::from_file("test_data/lp_share2b.mtx").unwrap());

        let nnz = row_ind.len();

        let A = SparseColMatRef::<'_, I, f64>::new(
            SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_ind),
            &values,
        );
        let zero = I(0);

        with_dim!(M, m);
        with_dim!(N, n);

        let A = ghost::SparseColMatRef::new(A, M, N);
        let mut new_col_ptrs = vec![zero; m + 1];
        let mut new_row_ind = vec![zero; nnz];
        let mut new_values = vec![0.0; nnz];

        let AT = crate::sparse::utils::ghost_adjoint(
            &mut new_col_ptrs,
            &mut new_row_ind,
            SliceGroupMut::<'_, f64>::new(&mut new_values),
            A,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
        )
        .into_const();

        let mut etree = vec![zero.to_signed(); n];
        let mut post = vec![zero; n];
        let mut col_counts = vec![zero; n];
        let mut min_row = vec![zero; m];

        ghost_col_etree(
            *A,
            None,
            Array::from_mut(&mut etree, N),
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(*M + *N))),
        );
        let etree_ = Array::from_ref(MaybeIdx::from_slice_ref_checked(&etree, N), N);
        ghost_postorder(
            Array::from_mut(&mut post, N),
            etree_,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
        );

        ghost_column_counts_aat(
            Array::from_mut(&mut col_counts, N),
            Array::from_mut(bytemuck::cast_slice_mut(&mut min_row), M),
            *AT,
            None,
            etree_,
            Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
        );

        let min_col = min_row;

        let symbolic = factorize_supernodal_symbolic_qr::<I>(
            A.symbolic().into_inner(),
            None,
            min_col,
            EliminationTreeRef::<'_, I> { inner: &etree },
            &col_counts,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            Default::default(),
        )
        .unwrap();

        let mut householder_row_indices =
            vec![zero; symbolic.householder().len_householder_row_indices()];

        let mut L_values = vec![0.0; symbolic.r_adjoint().len_values()];
        let mut householder_values = vec![0.0; symbolic.householder().len_householder_values()];
        let mut tau_values = vec![0.0; symbolic.householder().len_tau_values()];

        let mut tau_blocksize = vec![
            I(0);
            symbolic.householder().len_householder_row_indices()
                + symbolic.householder().n_supernodes()
        ];
        let mut householder_nrows = vec![
            I(0);
            symbolic.householder().len_householder_row_indices()
                + symbolic.householder().n_supernodes()
        ];
        let mut householder_ncols = vec![
            I(0);
            symbolic.householder().len_householder_row_indices()
                + symbolic.householder().n_supernodes()
        ];

        factorize_supernodal_numeric_qr::<I, f64>(
            &mut householder_row_indices,
            &mut tau_blocksize,
            &mut householder_nrows,
            &mut householder_ncols,
            &mut L_values,
            &mut householder_values,
            &mut tau_values,
            AT.into_inner(),
            None,
            &symbolic,
            crate::Parallelism::None,
            PodStack::new(&mut GlobalPodBuffer::new(
                factorize_supernodal_numeric_qr_req::<usize, f64>(
                    &symbolic,
                    crate::Parallelism::None,
                )
                .unwrap(),
            )),
        );
        let llt = reconstruct_from_supernodal_llt::<I, f64>(symbolic.r_adjoint(), &L_values);
        let a = sparse_to_dense(A.into_inner());
        let ata = a.adjoint() * &a;

        let llt_diff = &llt - &ata;
        assert!(llt_diff.norm_max() <= 1e-10);
    }

    #[test]
    fn test_numeric_qr_1_transpose() {
        type I = usize;
        type E = c64;

        let I = I::truncate;

        let mut gen = rand::rngs::StdRng::seed_from_u64(0);

        let (m, n, col_ptr, row_ind, values) =
            load_mtx::<usize>(MtxData::from_file("test_data/lp_share2b.mtx").unwrap());
        let values = values
            .iter()
            .map(|&x| c64::new(x, gen.gen()))
            .collect::<Vec<_>>();

        let nnz = row_ind.len();

        let A = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_ind),
            &values,
        );
        let zero = I(0);

        with_dim!(M, m);
        with_dim!(N, n);
        let A = ghost::SparseColMatRef::new(A, M, N);
        let mut new_col_ptrs = vec![zero; m + 1];
        let mut new_row_ind = vec![zero; nnz];
        let mut new_values = vec![E::faer_zero(); nnz];

        let AT = crate::sparse::utils::ghost_transpose(
            &mut new_col_ptrs,
            &mut new_row_ind,
            SliceGroupMut::<'_, E>::new(&mut new_values),
            A,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(*M))),
        )
        .into_const();

        let (A, AT) = (AT, A);
        let (M, N) = (N, M);
        let (m, n) = (n, m);

        let mut etree = vec![zero.to_signed(); n];
        let mut post = vec![zero; n];
        let mut col_counts = vec![zero; n];
        let mut min_row = vec![zero; m];

        ghost_col_etree(
            *A,
            None,
            Array::from_mut(&mut etree, N),
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(*M + *N))),
        );
        let etree_ = Array::from_ref(MaybeIdx::from_slice_ref_checked(&etree, N), N);
        ghost_postorder(
            Array::from_mut(&mut post, N),
            etree_,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(3 * *N))),
        );

        ghost_column_counts_aat(
            Array::from_mut(&mut col_counts, N),
            Array::from_mut(bytemuck::cast_slice_mut(&mut min_row), M),
            *AT,
            None,
            etree_,
            Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * *N + *M))),
        );

        let min_col = min_row;

        let symbolic = factorize_supernodal_symbolic_qr::<I>(
            A.symbolic().into_inner(),
            None,
            min_col,
            EliminationTreeRef::<'_, I> { inner: &etree },
            &col_counts,
            PodStack::new(&mut GlobalPodBuffer::new(
                supernodal::factorize_supernodal_symbolic_qr_req::<usize>(*M, *N).unwrap(),
            )),
            Default::default(),
        )
        .unwrap();

        let mut householder_row_indices =
            vec![zero; symbolic.householder().len_householder_row_indices()];

        let mut L_values = vec![E::faer_zero(); symbolic.r_adjoint().len_values()];
        let mut householder_values =
            vec![E::faer_zero(); symbolic.householder().len_householder_values()];
        let mut tau_values = vec![E::faer_zero(); symbolic.householder().len_tau_values()];

        let mut tau_blocksize = vec![
            I(0);
            symbolic.householder().len_householder_row_indices()
                + symbolic.householder().n_supernodes()
        ];
        let mut householder_nrows = vec![
            I(0);
            symbolic.householder().len_householder_row_indices()
                + symbolic.householder().n_supernodes()
        ];
        let mut householder_ncols = vec![
            I(0);
            symbolic.householder().len_householder_row_indices()
                + symbolic.householder().n_supernodes()
        ];

        let qr = factorize_supernodal_numeric_qr::<I, E>(
            &mut householder_row_indices,
            &mut tau_blocksize,
            &mut householder_nrows,
            &mut householder_ncols,
            &mut L_values,
            &mut householder_values,
            &mut tau_values,
            AT.into_inner(),
            None,
            &symbolic,
            crate::Parallelism::None,
            PodStack::new(&mut GlobalPodBuffer::new(
                factorize_supernodal_numeric_qr_req::<usize, E>(
                    &symbolic,
                    crate::Parallelism::None,
                )
                .unwrap(),
            )),
        );

        let a = sparse_to_dense(A.into_inner());

        let rhs = Mat::<E>::from_fn(m, 2, |_, _| c64::new(gen.gen(), gen.gen()));
        let mut x = rhs.clone();
        let mut work = rhs.clone();
        qr.solve_in_place_with_conj(
            crate::Conj::No,
            x.as_mut(),
            crate::Parallelism::None,
            work.as_mut(),
            PodStack::new(&mut GlobalPodBuffer::new(
                symbolic
                    .solve_in_place_req::<E>(2, crate::Parallelism::None)
                    .unwrap(),
            )),
        );
        let x = x.as_ref().subrows(0, n);

        let linsolve_diff = a.adjoint() * (&a * &x - &rhs);

        let llt = reconstruct_from_supernodal_llt::<I, E>(symbolic.r_adjoint(), &L_values);
        let ata = a.adjoint() * &a;

        let llt_diff = &llt - &ata;
        assert!(llt_diff.norm_max() <= 1e-10);
        assert!(linsolve_diff.norm_max() <= 1e-10);
    }

    #[test]
    fn test_numeric_simplicial_qr_1_transpose() {
        type I = usize;
        type E = c64;

        let I = I::truncate;
        let mut gen = rand::rngs::StdRng::seed_from_u64(0);

        let (m, n, col_ptr, row_ind, values) =
            load_mtx::<usize>(MtxData::from_file("test_data/lp_share2b.mtx").unwrap());
        let values = values
            .iter()
            .map(|&x| c64::new(x, gen.gen()))
            .collect::<Vec<_>>();

        let nnz = row_ind.len();

        let A = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_ind),
            &values,
        );
        let zero = I(0);

        with_dim!(M, m);
        with_dim!(N, n);
        let A = ghost::SparseColMatRef::new(A, M, N);
        let mut new_col_ptrs = vec![zero; m + 1];
        let mut new_row_ind = vec![zero; nnz];
        let mut new_values = vec![E::faer_zero(); nnz];

        let AT = crate::sparse::utils::ghost_transpose(
            &mut new_col_ptrs,
            &mut new_row_ind,
            SliceGroupMut::<'_, E>::new(&mut new_values),
            A,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(*M))),
        )
        .into_const();

        let (A, AT) = (AT, A);
        let (M, N) = (N, M);
        let (m, n) = (n, m);

        let mut etree = vec![zero.to_signed(); n];
        let mut post = vec![zero; n];
        let mut col_counts = vec![zero; n];
        let mut min_row = vec![zero; m];

        ghost_col_etree(
            *A,
            None,
            Array::from_mut(&mut etree, N),
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(*M + *N))),
        );
        let etree_ = Array::from_ref(MaybeIdx::from_slice_ref_checked(&etree, N), N);
        ghost_postorder(
            Array::from_mut(&mut post, N),
            etree_,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(3 * *N))),
        );

        ghost_column_counts_aat(
            Array::from_mut(&mut col_counts, N),
            Array::from_mut(bytemuck::cast_slice_mut(&mut min_row), M),
            *AT,
            None,
            etree_,
            Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * *N + *M))),
        );

        let min_col = min_row;

        let symbolic = factorize_simplicial_symbolic_qr::<I>(
            &min_col,
            EliminationTreeRef::<'_, I> { inner: &etree },
            &col_counts,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(3 * *N))),
        )
        .unwrap();

        let mut r_col_ptrs = vec![zero; n + 1];
        let mut r_row_indices = vec![zero; symbolic.len_r()];
        let mut householder_col_ptrs = vec![zero; n + 1];
        let mut householder_row_indices = vec![zero; symbolic.len_householder()];

        let mut r_values = vec![E::faer_zero(); symbolic.len_r()];
        let mut householder_values = vec![E::faer_zero(); symbolic.len_householder()];
        let mut tau_values = vec![E::faer_zero(); n];

        let qr = factorize_simplicial_numeric_qr_unsorted::<I, E>(
            &mut r_col_ptrs,
            &mut r_row_indices,
            &mut r_values,
            &mut householder_col_ptrs,
            &mut householder_row_indices,
            &mut householder_values,
            &mut tau_values,
            A.into_inner(),
            None,
            &symbolic,
            PodStack::new(&mut GlobalPodBuffer::new(
                factorize_simplicial_numeric_qr_req::<usize, E>(&symbolic).unwrap(),
            )),
        );
        let a = sparse_to_dense(A.into_inner());
        let rhs = Mat::<E>::from_fn(m, 2, |_, _| c64::new(gen.gen(), gen.gen()));
        {
            let mut x = rhs.clone();
            let mut work = rhs.clone();
            qr.solve_in_place_with_conj(
                crate::Conj::No,
                x.as_mut(),
                crate::Parallelism::None,
                work.as_mut(),
            );

            let x = x.as_ref().subrows(0, n);
            let linsolve_diff = a.adjoint() * (&a * &x - &rhs);
            assert!(linsolve_diff.norm_max() <= 1e-10);
        }
        {
            let mut x = rhs.clone();
            let mut work = rhs.clone();
            qr.solve_in_place_with_conj(
                crate::Conj::Yes,
                x.as_mut(),
                crate::Parallelism::None,
                work.as_mut(),
            );

            let x = x.as_ref().subrows(0, n);
            let a = a.conjugate();
            let linsolve_diff = a.adjoint() * (a * &x - &rhs);
            assert!(linsolve_diff.norm_max() <= 1e-10);
        }

        let R = SparseColMatRef::<'_, usize, E>::new(
            SymbolicSparseColMatRef::new_unsorted_checked(n, n, &r_col_ptrs, None, &r_row_indices),
            &r_values,
        );
        let r = sparse_to_dense(R);
        let ata = a.adjoint() * &a;
        let rtr = r.adjoint() * &r;
        assert!((&ata - &rtr).norm_max() < 1e-10);
    }

    #[test]
    fn test_solver_qr_1_transpose() {
        type I = usize;
        type E = c64;

        let I = I::truncate;
        let mut gen = rand::rngs::StdRng::seed_from_u64(0);

        let (m, n, col_ptr, row_ind, values) =
            load_mtx::<usize>(MtxData::from_file("test_data/lp_share2b.mtx").unwrap());
        let values = values
            .iter()
            .map(|&x| c64::new(x, gen.gen()))
            .collect::<Vec<_>>();
        let nnz = row_ind.len();

        let A = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_ind),
            &values,
        );

        let mut new_col_ptrs = vec![I(0); m + 1];
        let mut new_row_ind = vec![I(0); nnz];
        let mut new_values = vec![E::faer_zero(); nnz];

        let AT = crate::sparse::utils::transpose(
            &mut new_col_ptrs,
            &mut new_row_ind,
            SliceGroupMut::<'_, E>::new(&mut new_values).into_inner(),
            A,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(m))),
        )
        .into_const();
        let A = AT;
        let (m, n) = (n, m);

        let a = sparse_to_dense(A);
        let rhs = Mat::<E>::from_fn(m, 2, |_, _| c64::new(gen.gen(), gen.gen()));

        for supernodal_flop_ratio_threshold in [
            SupernodalThreshold::AUTO,
            SupernodalThreshold::FORCE_SUPERNODAL,
            SupernodalThreshold::FORCE_SIMPLICIAL,
        ] {
            let symbolic = super::factorize_symbolic_qr(
                A.symbolic(),
                QrSymbolicParams {
                    supernodal_flop_ratio_threshold,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut indices = vec![I(0); symbolic.len_indices()];
            let mut values = vec![E::faer_zero(); symbolic.len_values()];
            let qr = symbolic.factorize_numeric_qr::<E>(
                &mut indices,
                &mut values,
                A,
                Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(
                    symbolic
                        .factorize_numeric_qr_req::<E>(Parallelism::None)
                        .unwrap(),
                )),
            );

            {
                let mut x = rhs.clone();
                qr.solve_in_place_with_conj(
                    crate::Conj::No,
                    x.as_mut(),
                    crate::Parallelism::None,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        symbolic
                            .solve_in_place_req::<E>(2, Parallelism::None)
                            .unwrap(),
                    )),
                );

                let x = x.as_ref().subrows(0, n);
                let linsolve_diff = a.adjoint() * (&a * &x - &rhs);
                assert!(linsolve_diff.norm_max() <= 1e-10);
            }
            {
                let mut x = rhs.clone();
                qr.solve_in_place_with_conj(
                    crate::Conj::Yes,
                    x.as_mut(),
                    crate::Parallelism::None,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        symbolic
                            .solve_in_place_req::<E>(2, Parallelism::None)
                            .unwrap(),
                    )),
                );

                let x = x.as_ref().subrows(0, n);
                let a = a.conjugate();
                let linsolve_diff = a.adjoint() * (a * &x - &rhs);
                assert!(linsolve_diff.norm_max() <= 1e-10);
            }
        }
    }

    #[test]
    fn test_solver_qr_edge_case() {
        type I = usize;
        type E = c64;

        let I = I::truncate;
        let mut gen = rand::rngs::StdRng::seed_from_u64(0);

        let a0_col_ptr = vec![0usize; 21];
        let A0 = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_checked(40, 20, &a0_col_ptr, None, &[]),
            &[],
        );

        let a1_val = [
            c64::new(gen.gen(), gen.gen()),
            c64::new(gen.gen(), gen.gen()),
        ];
        let A1 = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_checked(40, 5, &[0, 1, 2, 2, 2, 2], None, &[0, 0]),
            &a1_val,
        );
        let A2 = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_checked(40, 5, &[0, 1, 2, 2, 2, 2], None, &[4, 4]),
            &a1_val,
        );

        for A in [A0, A1, A2] {
            for supernodal_flop_ratio_threshold in [
                SupernodalThreshold::AUTO,
                SupernodalThreshold::FORCE_SUPERNODAL,
                SupernodalThreshold::FORCE_SIMPLICIAL,
            ] {
                let symbolic = super::factorize_symbolic_qr(
                    A.symbolic(),
                    QrSymbolicParams {
                        supernodal_flop_ratio_threshold,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut indices = vec![I(0); symbolic.len_indices()];
                let mut values = vec![E::faer_zero(); symbolic.len_values()];
                symbolic.factorize_numeric_qr::<E>(
                    &mut indices,
                    &mut values,
                    A,
                    Parallelism::None,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        symbolic
                            .factorize_numeric_qr_req::<E>(Parallelism::None)
                            .unwrap(),
                    )),
                );
            }
        }
    }

    monomorphize_test!(test_symbolic_qr, u32);
}
