use crate::{
    cholesky::{ghost_postorder, simplicial::EliminationTreeRef},
    ghost::{self, Array, Idx, MaybeIdx},
    mem::{self, NONE},
    Index,
};
use dyn_stack::PodStack;
use faer_core::{
    constrained::Size,
    permutation::{PermutationRef, SignedIndex},
    sparse::SymbolicSparseColMatRef,
};
use faer_entity::*;
use reborrow::*;

#[inline]
pub(crate) fn ghost_col_etree<'m, 'n, I: Index>(
    A: ghost::SymbolicSparseColMatRef<'m, 'n, '_, I>,
    col_perm: Option<ghost::PermutationRef<'n, '_, I, Symbolic>>,
    etree: &mut Array<'n, I::Signed>,
    stack: PodStack<'_>,
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
        let pj = col_perm
            .map(|perm| perm.into_arrays().0[j].zx())
            .unwrap_or(j);
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

#[inline]
pub fn col_etree<'out, I: Index>(
    A: SymbolicSparseColMatRef<'_, I>,
    col_perm: Option<PermutationRef<'_, I, Symbolic>>,
    etree: &'out mut [I],
    stack: PodStack<'_>,
) -> EliminationTreeRef<'out, I> {
    Size::with2(A.nrows(), A.ncols(), |M, N| {
        ghost_col_etree(
            ghost::SymbolicSparseColMatRef::new(A, M, N),
            col_perm.map(|perm| ghost::PermutationRef::new(perm, N)),
            Array::from_mut(bytemuck::cast_slice_mut(etree), N),
            stack,
        );

        EliminationTreeRef {
            inner: bytemuck::cast_slice_mut(etree),
        }
    })
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
    row_perm: Option<ghost::PermutationRef<'m, '_, I, Symbolic>>,
    etree: &Array<'m, MaybeIdx<'m, I>>,
    post: &Array<'m, Idx<'m, I>>,
    stack: PodStack<'_>,
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
            let inv = perm.into_arrays().1;
            min_row[j] = match A.row_indices_of_col(j).map(|j| inv[j].zx()).min() {
                Some(first_row) => I::Signed::truncate(*first_row),
                None => *MaybeIdx::<'_, I>::none(),
            };
        } else {
            min_row[j] = match A.row_indices_of_col(j).min() {
                Some(first_row) => I::Signed::truncate(*first_row),
                None => *MaybeIdx::<'_, I>::none(),
            };
        }

        let min_row = if let Some(perm) = row_perm {
            let inv = perm.into_arrays().1;
            A.row_indices_of_col(j)
                .map(|row| post_inv[inv[row].zx()])
                .min()
        } else {
            A.row_indices_of_col(j).map(|row| post_inv[row]).min()
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
        let mut k_ = post[k].zx();
        delta[k_] = I::truncate(if first[k_].idx().is_none() { 1 } else { 0 });
        loop {
            if first[k_].idx().is_some() {
                break;
            }

            first[k_] = MaybeIdx::from_index(k.truncate());
            if let Some(parent) = etree[k_].idx() {
                k_ = parent.zx();
            } else {
                break;
            }
        }
    }

    for k in M.indices() {
        let k_ = post[k].zx();

        if let Some(parent) = etree[k_].idx() {
            let parent = parent.zx();
            decr(&mut delta[parent]);
        }

        let head_k = &mut head[k];
        let mut j = head_k.sx();
        *head_k = MaybeIdx::none();

        while let Some(j_) = j.idx() {
            for i in A.row_indices_of_col(j_) {
                let i = row_perm
                    .map(|perm| perm.into_arrays().1[i].zx())
                    .unwrap_or(i);
                let lca =
                    ghost_least_common_ancestor::<I>(i, k_, first, max_first, prev_leaf, ancestor);

                if lca != -2 {
                    incr(&mut delta[k_]);

                    if lca != -1 {
                        decr(&mut delta[M.check(lca as usize)]);
                    }
                }
            }
            j = next[j_].sx();
        }
        if let Some(parent) = etree[k_].idx() {
            ancestor[k_] = parent;
        }
    }

    for k in M.indices() {
        if let Some(parent) = etree[k].idx() {
            let parent = parent.zx();
            delta[parent] = I::from_signed(delta[parent].to_signed() + delta[k].to_signed());
        }
    }
}

pub fn column_counts_aat<'m, 'n, I: Index>(
    col_counts: &mut [I],
    min_row: &mut [I],
    A: SymbolicSparseColMatRef<'_, I>,
    row_perm: Option<PermutationRef<'_, I, Symbolic>>,
    etree: EliminationTreeRef<'_, I>,
    post: &[I],
    stack: PodStack<'_>,
) {
    Size::with2(A.nrows(), A.ncols(), |M, N| {
        let A = ghost::SymbolicSparseColMatRef::new(A, M, N);
        ghost_column_counts_aat(
            Array::from_mut(col_counts, M),
            Array::from_mut(bytemuck::cast_slice_mut(min_row), N),
            A,
            row_perm.map(|perm| ghost::PermutationRef::new(perm, M)),
            etree.ghost_inner(M),
            Array::from_ref(Idx::from_slice_ref_checked(post, M), M),
            stack,
        )
    })
}

pub fn postorder<I: Index>(post: &mut [I], etree: EliminationTreeRef<'_, I>, stack: PodStack<'_>) {
    Size::with(etree.inner.len(), |N| {
        ghost_postorder(Array::from_mut(post, N), etree.ghost_inner(N), stack)
    })
}

pub mod supernodal {
    use core::iter::zip;

    use faer_core::{
        group_helpers::{SliceGroup, SliceGroupMut},
        householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
        permutation::PermutationRef,
        sparse::{SparseColMatRef, SymbolicSparseColMatRef},
        zipped, MatMut, MatRef, Parallelism,
    };

    use super::*;
    use crate::{
        cholesky::{
            simplicial::EliminationTreeRef,
            supernodal::{CholeskySymbolicSupernodalParams, SymbolicSupernodalCholesky},
        },
        try_zeroed, FaerError,
    };

    #[derive(Debug)]
    pub struct SymbolicSupernodalHouseholder<I> {
        col_ptrs_for_row_indices: Vec<I>,
        col_ptrs_for_tau_values: Vec<I>,
        col_ptrs_for_values: Vec<I>,
        super_etree: Vec<I>,
    }

    impl<I: Index> SymbolicSupernodalHouseholder<I> {
        #[inline]
        pub fn n_supernodes(&self) -> usize {
            self.super_etree.len()
        }

        #[inline]
        pub fn col_ptrs_for_householder_values(&self) -> &[I] {
            self.col_ptrs_for_values.as_ref()
        }

        #[inline]
        pub fn col_ptrs_for_tau_values(&self) -> &[I] {
            self.col_ptrs_for_tau_values.as_ref()
        }

        #[inline]
        pub fn col_ptrs_for_householder_row_indices(&self) -> &[I] {
            self.col_ptrs_for_row_indices.as_ref()
        }

        #[inline]
        pub fn len_householder_values(&self) -> usize {
            self.col_ptrs_for_householder_values()[self.n_supernodes()].zx()
        }
        #[inline]
        pub fn len_householder_row_indices(&self) -> usize {
            self.col_ptrs_for_householder_row_indices()[self.n_supernodes()].zx()
        }
        #[inline]
        pub fn len_tau_values(&self) -> usize {
            self.col_ptrs_for_tau_values()[self.n_supernodes()].zx()
        }
    }

    #[derive(Debug)]
    pub struct SymbolicSupernodalQr<I> {
        L: SymbolicSupernodalCholesky<I>,
        H: SymbolicSupernodalHouseholder<I>,
        min_col: Vec<I>,
        min_col_perm: Vec<I>,
        index_to_super: Vec<I>,
        child_head: Vec<I>,
        child_next: Vec<I>,
    }

    impl<I> SymbolicSupernodalQr<I> {
        #[inline]
        pub fn l(&self) -> &SymbolicSupernodalCholesky<I> {
            &self.L
        }

        #[inline]
        pub fn householder(&self) -> &SymbolicSupernodalHouseholder<I> {
            &self.H
        }
    }

    pub fn factorize_supernodal_symbolic<I: Index>(
        A: SymbolicSparseColMatRef<'_, I>,
        col_perm: Option<PermutationRef<'_, I, Symbolic>>,
        min_col: Vec<I>,
        etree: EliminationTreeRef<'_, I>,
        col_counts: &[I],
        stack: PodStack<'_>,
        params: CholeskySymbolicSupernodalParams<'_>,
    ) -> Result<SymbolicSupernodalQr<I>, FaerError> {
        let m = A.nrows();
        let n = A.ncols();
        Size::with2(m, n, |M, N| {
            let A = ghost::SymbolicSparseColMatRef::new(A, M, N);
            let mut stack = stack;
            let (L, H) = {
                let etree = etree.ghost_inner(N);
                let min_col = Array::from_ref(
                    MaybeIdx::from_slice_ref_checked(bytemuck::cast_slice(&min_col), N),
                    M,
                );
                let L = crate::cholesky::supernodal::ghost_factorize_supernodal_symbolic(
                    A,
                    col_perm.map(|perm| ghost::PermutationRef::new(perm, N)),
                    Some(min_col),
                    crate::cholesky::supernodal::CholeskyInput::ATA,
                    etree,
                    Array::from_ref(&col_counts, N),
                    stack.rb_mut(),
                    params,
                )?;

                let H = ghost_factorize_supernodal_householder_symbolic(
                    &L, M, N, min_col, etree, stack,
                )?;

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
        })
    }

    pub fn ghost_factorize_supernodal_householder_symbolic<'m, 'n, I: Index>(
        L_symbolic: &SymbolicSupernodalCholesky<I>,
        M: Size<'m>,
        N: Size<'n>,
        min_col: &Array<'m, MaybeIdx<'n, I>>,
        etree: &Array<'n, MaybeIdx<'n, I>>,
        stack: PodStack<'_>,
    ) -> Result<SymbolicSupernodalHouseholder<I>, FaerError> {
        let n_supernodes = L_symbolic.n_supernodes();
        ghost::with_size(n_supernodes, |N_SUPERNODES| {
            let mut col_ptrs_for_row_indices = try_zeroed::<I>(n_supernodes + 1)?;
            let mut col_ptrs_for_tau_values = try_zeroed::<I>(n_supernodes + 1)?;
            let mut col_ptrs_for_values = try_zeroed::<I>(n_supernodes + 1)?;
            let mut super_etree_ = try_zeroed::<I>(n_supernodes)?;
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
                index_to_super.as_mut()[supernode_begin[s].zx()..supernode_end[s].zx()]
                    .fill(*s.truncate::<I>());
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
                let s_col_count = panel_width
                    + (L_col_ptrs_for_row_indices[*s + 1] - L_col_ptrs_for_row_indices[*s]);
                val_count += to_wide(s_row_count) * to_wide(s_col_count);
                row_count += to_wide(s_row_count);
                let blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<Symbolic>(
                    s_row_count.zx(),
                    s_col_count.zx(),
                ) as u128;
                // dbg!(s, blocksize, s_row_count, s_col_count);
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
            })
        })
    }

    #[track_caller]
    pub fn factorize_supernodal_numeric_qr<I: Index, E: ComplexField>(
        // len: n_supernodes
        col_end_for_row_indices_in_panel: &mut [I],
        // len: symbolic.householder().len_householder_row_indices()
        row_indices_in_panel: &mut [I],
        min_col_in_panel: &mut [I],
        min_col_in_panel_perm: &mut [I],

        L_values: GroupFor<E, &mut [E::Unit]>,
        householder_values: GroupFor<E, &mut [E::Unit]>,
        tau_values: GroupFor<E, &mut [E::Unit]>,

        AT: SparseColMatRef<'_, I, E>,
        col_perm: Option<PermutationRef<'_, I, E>>,
        symbolic: &SymbolicSupernodalQr<I>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) {
        {
            let n_supernodes = symbolic.l().n_supernodes();
            let L_values = SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&L_values)));
            let householder_values =
                SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&householder_values)));
            let tau_values = SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&tau_values)));
            assert!(col_end_for_row_indices_in_panel.len() == n_supernodes);
            assert!(
                row_indices_in_panel.len() == symbolic.householder().len_householder_row_indices()
            );
            assert!(min_col_in_panel.len() == symbolic.householder().len_householder_row_indices());
            assert!(
                min_col_in_panel_perm.len() == symbolic.householder().len_householder_row_indices()
            );
            assert!(L_values.len() == symbolic.l().len_values());
            assert!(householder_values.len() == symbolic.householder().len_householder_values());
            assert!(tau_values.len() == symbolic.householder().len_tau_values());
        }
        factorize_supernodal_numeric_qr_impl(
            col_end_for_row_indices_in_panel,
            row_indices_in_panel,
            min_col_in_panel,
            min_col_in_panel_perm,
            L_values,
            householder_values,
            tau_values,
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
    }

    pub(crate) fn factorize_supernodal_numeric_qr_impl<I: Index, E: ComplexField>(
        // len: n_supernodes
        col_end_for_row_indices_in_panel: &mut [I],
        // len: col_ptrs_for_row_indices[n_supernodes]
        row_indices_in_panel: &mut [I],
        min_col_in_panel: &mut [I],
        min_col_in_panel_perm: &mut [I],

        L_values: GroupFor<E, &mut [E::Unit]>,
        householder_values: GroupFor<E, &mut [E::Unit]>,
        tau_values: GroupFor<E, &mut [E::Unit]>,

        AT: SparseColMatRef<'_, I, E>,
        col_perm: Option<PermutationRef<'_, I, E>>,
        L_symbolic: &SymbolicSupernodalCholesky<I>,
        H_symbolic: &SymbolicSupernodalHouseholder<I>,
        min_col: &[I],
        min_col_perm: &[I],
        index_to_super: &[I],
        child_head: &[I::Signed],
        child_next: &[I::Signed],

        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) {
        let n_supernodes = L_symbolic.n_supernodes();
        let mut L_values = SliceGroupMut::<'_, E>::new(L_values);
        let mut householder_values = SliceGroupMut::<'_, E>::new(householder_values);
        let mut tau_values = SliceGroupMut::<'_, E>::new(tau_values);

        tau_values.fill_zero();
        L_values.fill_zero();
        householder_values.fill_zero();

        col_end_for_row_indices_in_panel
            .copy_from_slice(&H_symbolic.col_ptrs_for_row_indices[..n_supernodes]);

        let m = AT.ncols();
        let n = AT.nrows();

        for i in 0..m {
            let i = min_col_perm[i].zx();
            let min_col = min_col[i].zx();
            if min_col < n {
                let s = index_to_super[min_col].zx();
                let pos = &mut col_end_for_row_indices_in_panel[s];
                row_indices_in_panel[pos.zx()] = I::truncate(i);
                min_col_in_panel[pos.zx()] = I::truncate(min_col);
                *pos += I::truncate(1);
            }
        }

        let (col_global_to_local, stack) = stack.make_raw::<I::Signed>(n);
        let (child_col_global_to_local, stack) = stack.make_raw::<I::Signed>(n);
        let (child_row_global_to_local, mut stack) = stack.make_raw::<I::Signed>(m);

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

            let (row_indices_in_panel, parent_row_indices_in_panel) =
                row_indices_in_panel.split_at_mut(s_h_row_end);
            let s_row_indices_in_panel = &row_indices_in_panel[s_h_row_begin..];

            let (s_H, _) = householder_values
                .rb_mut()
                .split_at(H_symbolic.col_ptrs_for_values[s + 1].zx());
            let (c_H, s_H) = s_H.split_at(H_symbolic.col_ptrs_for_values[s].zx());
            let c_H = c_H.into_const();

            let mut s_H = MatMut::<'_, E>::from_column_major_slice(
                s_H.into_inner(),
                s_h_row_full_end - s_h_row_begin,
                s_ncols + s_pattern.len(),
            )
            .subrows(0, s_h_row_end - s_h_row_begin);

            let s_min_col_in_panel_perm = &mut min_col_in_panel_perm[s_h_row_begin..s_h_row_end];
            for (i, p) in s_min_col_in_panel_perm.iter_mut().enumerate() {
                *p = I::truncate(i);
            }
            s_min_col_in_panel_perm.sort_unstable_by_key(|i| s_min_col_in_panel[i.zx()]);
            let s_min_col_in_panel_perm = &min_col_in_panel_perm[s_h_row_begin..s_h_row_end];

            for idx in 0..s_h_row_end - s_h_row_begin {
                let i = s_row_indices_in_panel[s_min_col_in_panel_perm[idx].zx()].zx();
                if min_col[i].zx() >= s_col_begin {
                    for (j, value) in zip(
                        AT.row_indices_of_col(i),
                        SliceGroup::<'_, E>::new(AT.values_of_col(i)).into_ref_iter(),
                    ) {
                        let pj = col_perm
                            .map(|perm| perm.into_arrays().1[j].zx())
                            .unwrap_or(j);
                        s_H.write(idx, col_global_to_local[pj].zx(), value.read());
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

                let c_row_indices_in_panel = &row_indices_in_panel[c_h_row_begin..c_h_row_end];
                let c_min_col_in_panel = &c_min_col_in_panel[c_h_row_begin..c_h_row_end];

                let c_H = c_H.subslice(
                    H_symbolic.col_ptrs_for_values[child].zx()
                        ..H_symbolic.col_ptrs_for_values[child + 1].zx(),
                );
                let c_H = MatRef::<'_, E>::from_column_major_slice(
                    c_H.into_inner(),
                    H_symbolic.col_ptrs_for_row_indices[child + 1].zx() - c_h_row_begin,
                    c_ncols + c_pattern.len(),
                );

                let c_min_col_in_panel_perm = &min_col_in_panel_perm[c_h_row_begin..c_h_row_end];

                for (idx, &col) in c_pattern.iter().enumerate() {
                    child_col_global_to_local[col.zx()] = I::Signed::truncate(idx + c_ncols);
                }
                for (idx, &p) in c_min_col_in_panel_perm.iter().enumerate() {
                    child_row_global_to_local[c_row_indices_in_panel[p.zx()].zx()] =
                        I::Signed::truncate(idx);
                }

                for s_idx in 0..s_h_row_end - s_h_row_begin {
                    let i = s_row_indices_in_panel[s_min_col_in_panel_perm[s_idx].zx()].zx();
                    let c_idx = child_row_global_to_local[i];
                    if c_idx < I::Signed::truncate(0) {
                        continue;
                    }

                    let c_idx = c_idx.zx();
                    let c_min_col = c_min_col_in_panel[c_min_col_in_panel_perm[c_idx].zx()].zx();

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

                let blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<Symbolic>(
                    s_H.ncols(),
                    s_h_row_full_end - s_h_row_begin,
                );
                // dbg!(
                //     s,
                //     s_H.nrows(),
                //     s_H.ncols(),
                //     s_h_row_full_end - s_h_row_begin,
                //     blocksize
                // );
                let mut s_tau = MatMut::<'_, E>::from_column_major_slice(
                    s_tau.into_inner(),
                    blocksize,
                    Ord::min(s_H.ncols(), s_h_row_full_end - s_h_row_begin),
                );

                {
                    let mut current_min_col = 0usize;
                    let mut current_start = 0usize;
                    for idx in 0..s_h_nrows + 1 {
                        let idx_global_min_col = if idx < s_h_nrows {
                            s_min_col_in_panel[s_min_col_in_panel_perm[idx].zx()].zx()
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
                                >= current_min_col.saturating_add(Ord::max(1, blocksize / 2))
                        {
                            let nrows = idx.saturating_sub(current_start);
                            let full_ncols = s_H.ncols() - current_start;
                            let ncols = Ord::min(nrows, idx_min_col - current_min_col);

                            let s_H = s_H.rb_mut().submatrix(
                                current_start,
                                current_start,
                                nrows,
                                full_ncols,
                            );

                            let [mut left, mut right] = s_H.split_at_col(ncols);
                            let bs = faer_qr::no_pivoting::compute::recommended_blocksize::<Symbolic>(
                                left.nrows(),
                                left.ncols(),
                            );
                            let bs = Ord::min(blocksize, bs);

                            let mut s_tau =
                                s_tau.rb_mut().subrows(0, bs).subcols(current_start, ncols);

                            faer_qr::no_pivoting::compute::qr_in_place(
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
                                    faer_core::Conj::Yes,
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

                let mut s_L = MatMut::<'_, E>::from_column_major_slice(
                    s_L.into_inner(),
                    s_pattern.len() + s_ncols,
                    s_ncols,
                );
                let nrows = Ord::min(s_H.nrows(), s_L.ncols());
                zipped!(
                    s_L.rb_mut().transpose().subrows(0, nrows),
                    s_H.rb().subrows(0, nrows)
                )
                .for_each_triangular_upper(faer_core::zip::Diag::Include, |mut dst, src| {
                    dst.write(src.read().faer_conj())
                });
            }

            col_end_for_row_indices_in_panel[s] = Ord::min(
                I::truncate(s_h_row_begin + s_ncols + s_pattern.len()),
                col_end_for_row_indices_in_panel[s],
            );

            let s_h_row_end = col_end_for_row_indices_in_panel[s].zx();
            let s_h_nrows = s_h_row_end - s_h_row_begin;

            let mut current_min_col = 0usize;
            for idx in 0..s_h_nrows {
                let idx_global_min_col = s_min_col_in_panel[s_min_col_in_panel_perm[idx].zx()];
                if idx_global_min_col.zx() >= n {
                    break;
                }
                let idx_min_col = col_global_to_local[idx_global_min_col.zx()].zx();
                if current_min_col > idx_min_col {
                    s_min_col_in_panel[s_min_col_in_panel_perm[idx].zx()] =
                        I::truncate(s_col_local_to_global(current_min_col));
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
                parent_row_indices_in_panel[pos] =
                    s_row_indices_in_panel[s_min_col_in_panel_perm[idx].zx()];
                parent_min_col_in_panel[pos] =
                    s_min_col_in_panel[s_min_col_in_panel_perm[idx].zx()];
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
    }
}

#[cfg(test)]
mod tests {
    use core::iter::zip;

    use super::*;
    use crate::{
        cholesky::{
            ghost_postorder,
            simplicial::EliminationTreeRef,
            supernodal::{SupernodalLdltRef, SymbolicSupernodalCholesky},
        },
        ghost_adjoint, ghost_adjoint_symbolic,
        qr::supernodal::{
            factorize_supernodal_numeric_qr, factorize_supernodal_symbolic,
            ghost_factorize_supernodal_householder_symbolic,
        },
        SymbolicSparseColMatRef,
    };
    #[cfg(feature = "std")]
    use assert2::assert;
    use dyn_stack::{GlobalPodBuffer, StackReq};
    use faer_core::{
        group_helpers::{SliceGroup, SliceGroupMut},
        permutation::Permutation,
        sparse::SparseColMatRef,
        Mat,
    };
    use matrix_market_rs::MtxData;

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

            let [Ls_top, Ls_bot] = s.matrix().split_at_row(size);
            dense
                .as_mut()
                .submatrix(s.start(), s.start(), size, size)
                .clone_from(Ls_top);

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
        ghost::with_size(n, |N| {
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
        });
    }

    #[test]
    fn test_symbolic_qr_medium() {
        type I = usize;
        let I = I::truncate;

        let m = 32;

        let col_ptr = &[
            0, 1, 6, 7, 7, 12, 17, 20, 22, 25, 25, 27, 29, 33, 36, 40, 42usize,
        ];
        let row_ind = &[
            20, 16, 19, 24, 25, 27, 7, 12, 14, 24, 27, 28, 2, 9, 11, 12, 25, 2, 17, 26, 11, 23, 9,
            13, 25, 21, 27, 21, 28, 3, 4, 10, 11, 0, 1, 13, 3, 17, 29, 30, 4, 5,
        ];
        let n = col_ptr.len() - 1;
        let nnz = row_ind.len();

        let A = SymbolicSparseColMatRef::new_checked(m, n, col_ptr, None, row_ind);
        let zero = I(0);
        let mut etree = vec![zero.to_signed(); n];
        let mut post = vec![zero; n];
        let mut col_counts = vec![zero; n];
        Size::with2(m, n, |M, N| {
            let A = ghost::SymbolicSparseColMatRef::new(A, M, N);
            ghost_col_etree(
                A,
                None,
                Array::from_mut(&mut etree, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(2 * *N))),
            );
            let etree = Array::from_ref(MaybeIdx::from_slice_ref_checked(&etree, N), N);
            ghost_postorder(
                Array::from_mut(&mut post, N),
                etree,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );

            let mut min_row = vec![zero.to_signed(); m];
            let mut new_col_ptrs = vec![zero; m + 1];
            let mut new_row_ind = vec![zero; nnz];

            let AT = ghost_adjoint_symbolic(
                &mut new_col_ptrs,
                &mut new_row_ind,
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );
            ghost_column_counts_aat(
                Array::from_mut(&mut col_counts, N),
                Array::from_mut(&mut min_row, M),
                AT,
                None,
                etree,
                Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );

            let min_col = Array::from_ref(
                MaybeIdx::<'_, usize>::from_slice_ref_checked(&min_row, N),
                M,
            );

            let L_symbolic = crate::cholesky::supernodal::ghost_factorize_supernodal_symbolic(
                A,
                None,
                Some(min_col),
                crate::cholesky::supernodal::CholeskyInput::ATA,
                etree,
                Array::from_ref(&col_counts, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
                Default::default(),
            )
            .unwrap();

            let H_symbolic = ghost_factorize_supernodal_householder_symbolic(
                &L_symbolic,
                M,
                N,
                min_col,
                etree,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            )
            .unwrap();
            dbg!(&L_symbolic, &H_symbolic);
        });
    }

    #[test]
    fn test_numeric_qr_0_no_transpose() {
        type I = usize;
        let I = I::truncate;

        let m = 32;

        let col_ptr = &[
            0, 1, 6, 7, 7, 12, 17, 20, 22, 25, 25, 27, 29, 33, 36, 40, 42usize,
        ];
        let row_ind = &[
            20, 16, 19, 24, 25, 27, 7, 12, 14, 24, 27, 28, 2, 9, 11, 12, 25, 2, 17, 26, 11, 23, 9,
            13, 25, 21, 27, 21, 28, 3, 4, 10, 11, 0, 1, 13, 3, 17, 29, 30, 4, 5,
        ];
        let values = &vec![1.0; row_ind.len()];

        let n = col_ptr.len() - 1;
        let nnz = row_ind.len();

        let col_perm = Permutation::<I, Symbolic>::new_checked(
            Box::new([6, 2, 15, 14, 3, 4, 9, 1, 13, 10, 0, 5, 7, 11, 12, 8]),
            Box::new([10, 7, 1, 4, 5, 11, 0, 12, 15, 6, 9, 13, 14, 8, 3, 2]),
        );
        let perm = col_perm.as_ref();

        let A = SparseColMatRef::<'_, I, f64>::new(
            SymbolicSparseColMatRef::new_checked(m, n, col_ptr, None, row_ind),
            &values,
        );
        let zero = I(0);
        let mut etree = vec![zero.to_signed(); n];
        let mut post = vec![zero; n];
        let mut col_counts = vec![zero; n];
        let mut min_row = vec![zero; m];
        Size::with2(m, n, |M, N| {
            let A = ghost::SparseColMatRef::new(A, M, N);
            let mut new_col_ptrs = vec![zero; m + 1];
            let mut new_row_ind = vec![zero; nnz];
            let mut new_values = vec![0.0; nnz];

            let AT = ghost_adjoint(
                &mut new_col_ptrs,
                &mut new_row_ind,
                SliceGroupMut::<'_, f64>::new(&mut new_values),
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );

            let perm = ghost::PermutationRef::new(perm, N);

            ghost_col_etree(
                *A,
                Some(perm),
                Array::from_mut(&mut etree, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(*M + *N))),
            );
            let etree_ = Array::from_ref(MaybeIdx::from_slice_ref_checked(&etree, N), N);
            dbg!(etree_);
            ghost_postorder(
                Array::from_mut(&mut post, N),
                etree_,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );

            ghost_column_counts_aat(
                Array::from_mut(&mut col_counts, N),
                Array::from_mut(bytemuck::cast_slice_mut(&mut min_row), M),
                *AT,
                Some(perm),
                etree_,
                Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );
            dbg!(&col_counts);

            let min_col = min_row;

            let symbolic = factorize_supernodal_symbolic::<I>(
                *A.into_inner(),
                Some(perm.cast().into_inner()),
                min_col,
                EliminationTreeRef::<'_, I> { inner: &etree },
                &col_counts,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
                Default::default(),
            )
            .unwrap();

            let n_supernodes = symbolic.l().n_supernodes();

            let mut col_end_for_row_indices_in_panel = vec![zero; n_supernodes];
            let mut row_indices_in_panel =
                vec![zero; symbolic.householder().len_householder_row_indices()];
            let mut min_col_in_panel =
                vec![zero; symbolic.householder().len_householder_row_indices()];
            let mut min_col_in_panel_perm =
                vec![zero; symbolic.householder().len_householder_row_indices()];

            let mut L_values = vec![0.0; symbolic.l().len_values()];
            let mut householder_values = vec![0.0; symbolic.householder().len_householder_values()];
            let mut tau_values = vec![0.0; symbolic.householder().len_tau_values()];

            factorize_supernodal_numeric_qr::<I, f64>(
                &mut col_end_for_row_indices_in_panel,
                &mut row_indices_in_panel,
                &mut min_col_in_panel,
                &mut min_col_in_panel_perm,
                &mut L_values,
                &mut householder_values,
                &mut tau_values,
                AT.into_inner(),
                Some(perm.into_inner().cast()),
                &symbolic,
                faer_core::Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );
            let llt = reconstruct_from_supernodal_llt::<I, f64>(symbolic.l(), &L_values);
            dbgf::dbgf!("6.2?", &llt);
        });
    }

    #[test]
    fn test_numeric_qr_0_transpose() {
        type I = usize;
        let I = I::truncate;

        let m = 32;

        let col_ptr = &[
            0, 1, 6, 7, 7, 12, 17, 20, 22, 25, 25, 27, 29, 33, 36, 40, 42usize,
        ];
        let row_ind = &[
            20, 16, 19, 24, 25, 27, 7, 12, 14, 24, 27, 28, 2, 9, 11, 12, 25, 2, 17, 26, 11, 23, 9,
            13, 25, 21, 27, 21, 28, 3, 4, 10, 11, 0, 1, 13, 3, 17, 29, 30, 4, 5,
        ];
        let values = &vec![1.0; row_ind.len()];

        let n = col_ptr.len() - 1;
        let nnz = row_ind.len();

        let A = SparseColMatRef::<'_, I, f64>::new(
            SymbolicSparseColMatRef::new_checked(m, n, col_ptr, None, row_ind),
            &values,
        );

        let zero = I(0);
        Size::with2(m, n, |M, N| {
            let A = ghost::SparseColMatRef::new(A, M, N);
            let mut new_col_ptrs = vec![zero; m + 1];
            let mut new_row_ind = vec![zero; nnz];
            let mut new_values = vec![0.0; nnz];

            let AT = ghost_adjoint(
                &mut new_col_ptrs,
                &mut new_row_ind,
                SliceGroupMut::<'_, f64>::new(&mut new_values),
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );

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

            let symbolic = factorize_supernodal_symbolic::<I>(
                *A.into_inner(),
                None,
                min_col,
                EliminationTreeRef::<'_, I> { inner: &etree },
                &col_counts,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
                Default::default(),
            )
            .unwrap();

            let n_supernodes = symbolic.l().n_supernodes();

            let mut col_end_for_row_indices_in_panel = vec![zero; n_supernodes];
            let mut row_indices_in_panel =
                vec![zero; symbolic.householder().len_householder_row_indices()];
            let mut min_col_in_panel =
                vec![zero; symbolic.householder().len_householder_row_indices()];
            let mut min_col_in_panel_perm =
                vec![zero; symbolic.householder().len_householder_row_indices()];

            let mut L_values = vec![0.0; symbolic.l().len_values()];
            let mut householder_values = vec![0.0; symbolic.householder().len_householder_values()];
            let mut tau_values = vec![0.0; symbolic.householder().len_tau_values()];

            factorize_supernodal_numeric_qr::<I, f64>(
                &mut col_end_for_row_indices_in_panel,
                &mut row_indices_in_panel,
                &mut min_col_in_panel,
                &mut min_col_in_panel_perm,
                &mut L_values,
                &mut householder_values,
                &mut tau_values,
                AT.into_inner(),
                None,
                &symbolic,
                faer_core::Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );
            let llt = reconstruct_from_supernodal_llt::<I, f64>(symbolic.l(), &L_values);
            dbgf::dbgf!("6.2?", &llt);
        });
    }

    fn load_mtx<I: Index>(data: MtxData<f64>) -> (usize, usize, Vec<I>, Vec<I>, Vec<f64>) {
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
        Size::with2(m, n, |M, N| {
            let A = ghost::SparseColMatRef::new(A, M, N);
            let mut new_col_ptrs = vec![zero; m + 1];
            let mut new_row_ind = vec![zero; nnz];
            let mut new_values = vec![0.0; nnz];

            let AT = ghost_adjoint(
                &mut new_col_ptrs,
                &mut new_row_ind,
                SliceGroupMut::<'_, f64>::new(&mut new_values),
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );

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

            let symbolic = factorize_supernodal_symbolic::<I>(
                *A.into_inner(),
                None,
                min_col,
                EliminationTreeRef::<'_, I> { inner: &etree },
                &col_counts,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
                Default::default(),
            )
            .unwrap();

            let n_supernodes = symbolic.l().n_supernodes();

            let mut col_end_for_row_indices_in_panel = vec![zero; n_supernodes];
            let mut row_indices_in_panel =
                vec![zero; symbolic.householder().len_householder_row_indices()];
            let mut min_col_in_panel =
                vec![zero; symbolic.householder().len_householder_row_indices()];
            let mut min_col_in_panel_perm =
                vec![zero; symbolic.householder().len_householder_row_indices()];

            let mut L_values = vec![0.0; symbolic.l().len_values()];
            let mut householder_values = vec![0.0; symbolic.householder().len_householder_values()];
            let mut tau_values = vec![0.0; symbolic.householder().len_tau_values()];

            factorize_supernodal_numeric_qr::<I, f64>(
                &mut col_end_for_row_indices_in_panel,
                &mut row_indices_in_panel,
                &mut min_col_in_panel,
                &mut min_col_in_panel_perm,
                &mut L_values,
                &mut householder_values,
                &mut tau_values,
                AT.into_inner(),
                None,
                &symbolic,
                faer_core::Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );
            let llt = reconstruct_from_supernodal_llt::<I, f64>(symbolic.l(), &L_values);
            let a = sparse_to_dense(A.into_inner());
            let ata = a.transpose() * &a;
            dbgf::dbgf!("6.10?", &llt - &ata);
        });
    }

    #[test]
    fn test_numeric_qr_1_transpose() {
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
        Size::with2(m, n, |M, N| {
            let A = ghost::SparseColMatRef::new(A, M, N);
            let mut new_col_ptrs = vec![zero; m + 1];
            let mut new_row_ind = vec![zero; nnz];
            let mut new_values = vec![0.0; nnz];

            let AT = ghost_adjoint(
                &mut new_col_ptrs,
                &mut new_row_ind,
                SliceGroupMut::<'_, f64>::new(&mut new_values),
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );

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

            let symbolic = factorize_supernodal_symbolic::<I>(
                *A.into_inner(),
                None,
                min_col,
                EliminationTreeRef::<'_, I> { inner: &etree },
                &col_counts,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
                Default::default(),
            )
            .unwrap();

            let n_supernodes = symbolic.l().n_supernodes();

            let mut col_end_for_row_indices_in_panel = vec![zero; n_supernodes];
            let mut row_indices_in_panel =
                vec![zero; symbolic.householder().len_householder_row_indices()];
            let mut min_col_in_panel =
                vec![zero; symbolic.householder().len_householder_row_indices()];
            let mut min_col_in_panel_perm =
                vec![zero; symbolic.householder().len_householder_row_indices()];

            let mut L_values = vec![0.0; symbolic.l().len_values()];
            let mut householder_values = vec![0.0; symbolic.householder().len_householder_values()];
            let mut tau_values = vec![0.0; symbolic.householder().len_tau_values()];

            factorize_supernodal_numeric_qr::<I, f64>(
                &mut col_end_for_row_indices_in_panel,
                &mut row_indices_in_panel,
                &mut min_col_in_panel,
                &mut min_col_in_panel_perm,
                &mut L_values,
                &mut householder_values,
                &mut tau_values,
                AT.into_inner(),
                None,
                &symbolic,
                faer_core::Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * *N))),
            );
            let llt = reconstruct_from_supernodal_llt::<I, f64>(symbolic.l(), &L_values);
            let a = sparse_to_dense(A.into_inner());
            let ata = a.transpose() * &a;
            dbgf::dbgf!("6.10?", &llt - &ata);
        });
    }

    monomorphize_test!(test_symbolic_qr, u32);
}
