//! Computes the LU decomposition of a given sparse matrix. See [`faer_lu`] for more info.
//!
//! The entry point in this module is [`SymbolicLu`] and [`factorize_symbolic_lu`].

use crate::{
    cholesky::simplicial::EliminationTreeRef,
    colamd::Control,
    ghost,
    ghost::{Array, Idx, MaybeIdx},
    mem::{
        NONE, {self},
    },
    nomem, try_zeroed, FaerError, Index, SupernodalThreshold, SymbolicSparseColMatRef,
    SymbolicSupernodalParams,
};
use core::{iter::zip, mem::MaybeUninit};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    assert,
    constrained::Size,
    group_helpers::{SliceGroup, SliceGroupMut, VecGroup},
    mul,
    permutation::{PermutationRef, SignedIndex},
    solve,
    sparse::SparseColMatRef,
    temp_mat_req, temp_mat_uninit, Conj, MatMut, Parallelism,
};
use faer_entity::*;
use reborrow::*;

#[inline(never)]
fn resize_scalar<E: Entity>(
    v: &mut VecGroup<E>,
    n: usize,
    exact: bool,
    reserve_only: bool,
) -> Result<(), FaerError> {
    let reserve = if exact {
        VecGroup::try_reserve_exact
    } else {
        VecGroup::try_reserve
    };
    reserve(v, n.saturating_sub(v.len())).map_err(|_| FaerError::OutOfMemory)?;
    if !reserve_only {
        v.resize(
            Ord::max(n, v.len()),
            unsafe { core::mem::zeroed::<E>() }.faer_into_units(),
        );
    }

    Ok(())
}

#[inline(never)]
fn resize_maybe_uninit_scalar<E: Entity>(
    v: &mut GroupFor<E, alloc::vec::Vec<MaybeUninit<E::Unit>>>,
    n: usize,
) -> Result<(), FaerError> {
    let mut failed = false;
    E::faer_map(E::faer_as_mut(v), |v| {
        if !failed {
            failed = v.try_reserve(n.saturating_sub(v.len())).is_err();
            unsafe { v.set_len(n) };
        }
    });
    if failed {
        Err(FaerError::OutOfMemory)
    } else {
        Ok(())
    }
}

#[inline(never)]
fn resize_index<I: Index>(
    v: &mut alloc::vec::Vec<I>,
    n: usize,
    exact: bool,
    reserve_only: bool,
) -> Result<(), FaerError> {
    let reserve = if exact {
        alloc::vec::Vec::try_reserve_exact
    } else {
        alloc::vec::Vec::try_reserve
    };
    reserve(v, n.saturating_sub(v.len())).map_err(|_| FaerError::OutOfMemory)?;
    if !reserve_only {
        v.resize(Ord::max(n, v.len()), I::truncate(0));
    }
    Ok(())
}

/// Sparse LU error.
#[derive(Copy, Clone, Debug)]
pub enum LuError {
    Generic(FaerError),
    SymbolicSingular(usize),
}

impl core::fmt::Display for LuError {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LuError {}

impl From<FaerError> for LuError {
    #[inline]
    fn from(value: FaerError) -> Self {
        Self::Generic(value)
    }
}

pub mod supernodal {
    use super::*;
    use faer_core::assert;

    #[derive(Debug, Clone)]
    pub struct SymbolicSupernodalLu<I> {
        pub(super) supernode_ptr: alloc::vec::Vec<I>,
        pub(super) super_etree: alloc::vec::Vec<I>,
        pub(super) supernode_postorder: alloc::vec::Vec<I>,
        pub(super) supernode_postorder_inv: alloc::vec::Vec<I>,
        pub(super) descendant_count: alloc::vec::Vec<I>,
        pub(super) nrows: usize,
        pub(super) ncols: usize,
    }

    #[derive(Debug, Clone)]
    pub struct SupernodalLu<I, E: Entity> {
        nrows: usize,
        ncols: usize,
        nsupernodes: usize,

        supernode_ptr: alloc::vec::Vec<I>,

        l_col_ptr_for_row_ind: alloc::vec::Vec<I>,
        l_col_ptr_for_val: alloc::vec::Vec<I>,
        l_row_ind: alloc::vec::Vec<I>,
        l_val: VecGroup<E>,

        ut_col_ptr_for_row_ind: alloc::vec::Vec<I>,
        ut_col_ptr_for_val: alloc::vec::Vec<I>,
        ut_row_ind: alloc::vec::Vec<I>,
        ut_val: VecGroup<E>,
    }

    impl<I: Index, E: Entity> SupernodalLu<I, E> {
        #[inline]
        pub fn new() -> Self {
            Self {
                nrows: 0,
                ncols: 0,
                nsupernodes: 0,

                supernode_ptr: alloc::vec::Vec::new(),

                l_col_ptr_for_row_ind: alloc::vec::Vec::new(),
                ut_col_ptr_for_row_ind: alloc::vec::Vec::new(),

                l_col_ptr_for_val: alloc::vec::Vec::new(),
                ut_col_ptr_for_val: alloc::vec::Vec::new(),

                l_row_ind: alloc::vec::Vec::new(),
                ut_row_ind: alloc::vec::Vec::new(),

                l_val: VecGroup::new(),
                ut_val: VecGroup::new(),
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
        pub fn n_supernodes(&self) -> usize {
            self.nsupernodes
        }

        #[track_caller]
        pub fn solve_in_place_with_conj(
            &self,
            row_perm: PermutationRef<'_, I, E>,
            col_perm: PermutationRef<'_, I, E>,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            work: MatMut<'_, E>,
        ) where
            E: ComplexField,
        {
            assert!(all(
                self.nrows() == self.ncols(),
                self.nrows() == rhs.nrows()
            ));
            let mut X = rhs;
            let mut temp = work;

            faer_core::permutation::permute_rows(temp.rb_mut(), X.rb(), row_perm);
            self.l_solve_in_place_with_conj(conj_lhs, temp.rb_mut(), X.rb_mut(), parallelism);
            self.u_solve_in_place_with_conj(conj_lhs, temp.rb_mut(), X.rb_mut(), parallelism);
            faer_core::permutation::permute_rows(X.rb_mut(), temp.rb(), col_perm.inverse());
        }

        #[track_caller]
        pub fn solve_transpose_in_place_with_conj(
            &self,
            row_perm: PermutationRef<'_, I, E>,
            col_perm: PermutationRef<'_, I, E>,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            work: MatMut<'_, E>,
        ) where
            E: ComplexField,
        {
            assert!(all(
                self.nrows() == self.ncols(),
                self.nrows() == rhs.nrows()
            ));
            let mut X = rhs;
            let mut temp = work;
            faer_core::permutation::permute_rows(temp.rb_mut(), X.rb(), col_perm);
            self.u_solve_transpose_in_place_with_conj(
                conj_lhs,
                temp.rb_mut(),
                X.rb_mut(),
                parallelism,
            );
            self.l_solve_transpose_in_place_with_conj(
                conj_lhs,
                temp.rb_mut(),
                X.rb_mut(),
                parallelism,
            );
            faer_core::permutation::permute_rows(X.rb_mut(), temp.rb(), row_perm.inverse());
        }

        #[track_caller]
        pub(crate) fn l_solve_in_place_with_conj(
            &self,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            mut work: MatMut<'_, E>,
            parallelism: Parallelism,
        ) where
            E: ComplexField,
        {
            let lu = &*self;

            assert!(lu.nrows() == lu.ncols());
            assert!(lu.nrows() == rhs.nrows());

            let mut X = rhs;
            let nrhs = X.ncols();

            let supernode_ptr = &*lu.supernode_ptr;

            for s in 0..lu.nsupernodes {
                let s_begin = supernode_ptr[s].zx();
                let s_end = supernode_ptr[s + 1].zx();
                let s_size = s_end - s_begin;
                let s_row_index_count =
                    (lu.l_col_ptr_for_row_ind[s + 1] - lu.l_col_ptr_for_row_ind[s]).zx();

                let L = lu
                    .l_val
                    .as_slice()
                    .subslice(lu.l_col_ptr_for_val[s].zx()..lu.l_col_ptr_for_val[s + 1].zx());
                let L = faer_core::mat::from_column_major_slice::<'_, E>(
                    L.into_inner(),
                    s_row_index_count,
                    s_size,
                );
                let (L_top, L_bot) = L.split_at_row(s_size);
                solve::solve_unit_lower_triangular_in_place_with_conj(
                    L_top,
                    conj_lhs,
                    X.rb_mut().subrows_mut(s_begin, s_size),
                    parallelism,
                );
                mul::matmul_with_conj(
                    work.rb_mut().subrows_mut(0, s_row_index_count - s_size),
                    L_bot,
                    conj_lhs,
                    X.rb().subrows(s_begin, s_size),
                    Conj::No,
                    None,
                    E::faer_one(),
                    parallelism,
                );

                for j in 0..nrhs {
                    for (idx, &i) in lu.l_row_ind
                        [lu.l_col_ptr_for_row_ind[s].zx()..lu.l_col_ptr_for_row_ind[s + 1].zx()]
                        [s_size..]
                        .iter()
                        .enumerate()
                    {
                        let i = i.zx();
                        X.write(i, j, X.read(i, j).faer_sub(work.read(idx, j)));
                    }
                }
            }
        }

        #[track_caller]
        pub(crate) fn l_solve_transpose_in_place_with_conj(
            &self,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            mut work: MatMut<'_, E>,
            parallelism: Parallelism,
        ) where
            E: ComplexField,
        {
            let lu = &*self;

            assert!(lu.nrows() == lu.ncols());
            assert!(lu.nrows() == rhs.nrows());

            let mut X = rhs;
            let nrhs = X.ncols();

            let supernode_ptr = &*lu.supernode_ptr;

            for s in (0..lu.nsupernodes).rev() {
                let s_begin = supernode_ptr[s].zx();
                let s_end = supernode_ptr[s + 1].zx();
                let s_size = s_end - s_begin;
                let s_row_index_count =
                    (lu.l_col_ptr_for_row_ind[s + 1] - lu.l_col_ptr_for_row_ind[s]).zx();

                let L = lu
                    .l_val
                    .as_slice()
                    .subslice(lu.l_col_ptr_for_val[s].zx()..lu.l_col_ptr_for_val[s + 1].zx());
                let L = faer_core::mat::from_column_major_slice::<'_, E>(
                    L.into_inner(),
                    s_row_index_count,
                    s_size,
                );

                let (L_top, L_bot) = L.split_at_row(s_size);

                for j in 0..nrhs {
                    for (idx, &i) in lu.l_row_ind
                        [lu.l_col_ptr_for_row_ind[s].zx()..lu.l_col_ptr_for_row_ind[s + 1].zx()]
                        [s_size..]
                        .iter()
                        .enumerate()
                    {
                        let i = i.zx();
                        work.write(idx, j, X.read(i, j));
                    }
                }

                mul::matmul_with_conj(
                    X.rb_mut().subrows_mut(s_begin, s_size),
                    L_bot.transpose(),
                    conj_lhs,
                    work.rb().subrows(0, s_row_index_count - s_size),
                    Conj::No,
                    Some(E::faer_one()),
                    E::faer_one().faer_neg(),
                    parallelism,
                );
                solve::solve_unit_upper_triangular_in_place_with_conj(
                    L_top.transpose(),
                    conj_lhs,
                    X.rb_mut().subrows_mut(s_begin, s_size),
                    parallelism,
                );
            }
        }

        #[track_caller]
        pub(crate) fn u_solve_in_place_with_conj(
            &self,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            mut work: MatMut<'_, E>,
            parallelism: Parallelism,
        ) where
            E: ComplexField,
        {
            let lu = &*self;

            assert!(lu.nrows() == lu.ncols());
            assert!(lu.nrows() == rhs.nrows());

            let mut X = rhs;
            let nrhs = X.ncols();

            let supernode_ptr = &*lu.supernode_ptr;

            for s in (0..lu.nsupernodes).rev() {
                let s_begin = supernode_ptr[s].zx();
                let s_end = supernode_ptr[s + 1].zx();
                let s_size = s_end - s_begin;
                let s_row_index_count =
                    (lu.l_col_ptr_for_row_ind[s + 1] - lu.l_col_ptr_for_row_ind[s]).zx();
                let s_col_index_count =
                    (lu.ut_col_ptr_for_row_ind[s + 1] - lu.ut_col_ptr_for_row_ind[s]).zx();

                let L = lu
                    .l_val
                    .as_slice()
                    .subslice(lu.l_col_ptr_for_val[s].zx()..lu.l_col_ptr_for_val[s + 1].zx());
                let L = faer_core::mat::from_column_major_slice::<'_, E>(
                    L.into_inner(),
                    s_row_index_count,
                    s_size,
                );
                let U = lu
                    .ut_val
                    .as_slice()
                    .subslice(lu.ut_col_ptr_for_val[s].zx()..lu.ut_col_ptr_for_val[s + 1].zx());
                let U_right = faer_core::mat::from_column_major_slice::<'_, E>(
                    U.into_inner(),
                    s_col_index_count,
                    s_size,
                )
                .transpose();

                for j in 0..nrhs {
                    for (idx, &i) in lu.ut_row_ind
                        [lu.ut_col_ptr_for_row_ind[s].zx()..lu.ut_col_ptr_for_row_ind[s + 1].zx()]
                        .iter()
                        .enumerate()
                    {
                        let i = i.zx();
                        work.write(idx, j, X.read(i, j));
                    }
                }

                let (U_left, _) = L.split_at_row(s_size);
                mul::matmul_with_conj(
                    X.rb_mut().subrows_mut(s_begin, s_size),
                    U_right,
                    conj_lhs,
                    work.rb().subrows(0, s_col_index_count),
                    Conj::No,
                    Some(E::faer_one()),
                    E::faer_one().faer_neg(),
                    parallelism,
                );
                solve::solve_upper_triangular_in_place_with_conj(
                    U_left,
                    conj_lhs,
                    X.rb_mut().subrows_mut(s_begin, s_size),
                    parallelism,
                );
            }
        }

        #[track_caller]
        pub(crate) fn u_solve_transpose_in_place_with_conj(
            &self,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            mut work: MatMut<'_, E>,
            parallelism: Parallelism,
        ) where
            E: ComplexField,
        {
            let lu = &*self;

            assert!(lu.nrows() == lu.ncols());
            assert!(lu.nrows() == rhs.nrows());

            let mut X = rhs;
            let nrhs = X.ncols();

            let supernode_ptr = &*lu.supernode_ptr;

            for s in 0..lu.nsupernodes {
                let s_begin = supernode_ptr[s].zx();
                let s_end = supernode_ptr[s + 1].zx();
                let s_size = s_end - s_begin;
                let s_row_index_count =
                    (lu.l_col_ptr_for_row_ind[s + 1] - lu.l_col_ptr_for_row_ind[s]).zx();
                let s_col_index_count =
                    (lu.ut_col_ptr_for_row_ind[s + 1] - lu.ut_col_ptr_for_row_ind[s]).zx();

                let L = lu
                    .l_val
                    .as_slice()
                    .subslice(lu.l_col_ptr_for_val[s].zx()..lu.l_col_ptr_for_val[s + 1].zx());
                let L = faer_core::mat::from_column_major_slice::<'_, E>(
                    L.into_inner(),
                    s_row_index_count,
                    s_size,
                );
                let U = lu
                    .ut_val
                    .as_slice()
                    .subslice(lu.ut_col_ptr_for_val[s].zx()..lu.ut_col_ptr_for_val[s + 1].zx());
                let U_right = faer_core::mat::from_column_major_slice::<'_, E>(
                    U.into_inner(),
                    s_col_index_count,
                    s_size,
                )
                .transpose();

                let (U_left, _) = L.split_at_row(s_size);
                solve::solve_lower_triangular_in_place_with_conj(
                    U_left.transpose(),
                    conj_lhs,
                    X.rb_mut().subrows_mut(s_begin, s_size),
                    parallelism,
                );
                mul::matmul_with_conj(
                    work.rb_mut().subrows_mut(0, s_col_index_count),
                    U_right.transpose(),
                    conj_lhs,
                    X.rb().subrows(s_begin, s_size),
                    Conj::No,
                    None,
                    E::faer_one(),
                    parallelism,
                );

                for j in 0..nrhs {
                    for (idx, &i) in lu.ut_row_ind
                        [lu.ut_col_ptr_for_row_ind[s].zx()..lu.ut_col_ptr_for_row_ind[s + 1].zx()]
                        .iter()
                        .enumerate()
                    {
                        let i = i.zx();
                        X.write(i, j, X.read(i, j).faer_sub(work.read(idx, j)));
                    }
                }
            }
        }
    }

    pub fn factorize_supernodal_symbolic_lu_req<I: Index>(
        nrows: usize,
        ncols: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let _ = nrows;
        crate::cholesky::supernodal::factorize_supernodal_symbolic_cholesky_req::<I>(ncols)
    }

    #[track_caller]
    pub fn factorize_supernodal_symbolic_lu<I: Index>(
        A: SymbolicSparseColMatRef<'_, I>,
        col_perm: Option<PermutationRef<'_, I, Symbolic>>,
        min_col: &[I],
        etree: EliminationTreeRef<'_, I>,
        col_counts: &[I],
        stack: PodStack<'_>,
        params: SymbolicSupernodalParams<'_>,
    ) -> Result<SymbolicSupernodalLu<I>, FaerError> {
        let m = A.nrows();
        let n = A.ncols();
        Size::with2(m, n, |M, N| {
            let I = I::truncate;
            let A = ghost::SymbolicSparseColMatRef::new(A, M, N);
            let min_col = Array::from_ref(
                MaybeIdx::from_slice_ref_checked(bytemuck::cast_slice(&min_col), N),
                M,
            );
            let etree = etree.ghost_inner(N);
            let mut stack = stack;

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
            let n_supernodes = L.n_supernodes();
            let mut super_etree = try_zeroed::<I>(n_supernodes)?;

            let (index_to_super, _) = stack.make_raw::<I>(*N);

            for s in 0..n_supernodes {
                index_to_super.as_mut()[L.supernode_begin[s].zx()..L.supernode_begin[s + 1].zx()]
                    .fill(I(s));
            }
            for s in 0..n_supernodes {
                let last = L.supernode_begin[s + 1].zx() - 1;
                if let Some(parent) = etree[N.check(last)].idx() {
                    super_etree[s] = index_to_super[*parent.zx()];
                } else {
                    super_etree[s] = I(NONE);
                }
            }

            Ok(SymbolicSupernodalLu {
                supernode_ptr: L.supernode_begin,
                super_etree,
                supernode_postorder: L.supernode_postorder,
                supernode_postorder_inv: L.supernode_postorder_inv,
                descendant_count: L.descendant_count,
                nrows: *A.nrows(),
                ncols: *A.ncols(),
            })
        })
    }

    struct MatU8 {
        data: alloc::vec::Vec<u8>,
        nrows: usize,
    }
    impl MatU8 {
        fn new(nrows: usize, ncols: usize) -> Self {
            Self {
                data: alloc::vec![1u8; nrows * ncols],
                nrows,
            }
        }
    }
    impl core::ops::Index<(usize, usize)> for MatU8 {
        type Output = u8;
        #[inline(always)]
        fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
            &self.data[row + col * self.nrows]
        }
    }
    impl core::ops::IndexMut<(usize, usize)> for MatU8 {
        #[inline(always)]
        fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
            &mut self.data[row + col * self.nrows]
        }
    }

    struct Front;
    struct LPanel;
    struct UPanel;

    #[inline(never)]
    fn noinline<T, R>(_: T, f: impl FnOnce() -> R) -> R {
        f()
    }

    pub fn factorize_supernodal_numeric_lu_req<I: Index, E: Entity>(
        symbolic: &SymbolicSupernodalLu<I>,
    ) -> Result<StackReq, SizeOverflow> {
        let m = StackReq::try_new::<I>(symbolic.nrows)?;
        let n = StackReq::try_new::<I>(symbolic.ncols)?;
        StackReq::try_all_of([n, m, m, m, m, m])
    }

    pub fn factorize_supernodal_numeric_lu<I: Index, E: ComplexField>(
        row_perm: &mut [I],
        row_perm_inv: &mut [I],
        lu: &mut SupernodalLu<I, E>,

        A: SparseColMatRef<'_, I, E>,
        AT: SparseColMatRef<'_, I, E>,
        col_perm: PermutationRef<'_, I, E>,
        symbolic: &SymbolicSupernodalLu<I>,

        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) -> Result<(), LuError> {
        use crate::cholesky::supernodal::partition_fn;
        let SymbolicSupernodalLu {
            supernode_ptr,
            super_etree,
            supernode_postorder,
            supernode_postorder_inv,
            descendant_count,
            nrows: _,
            ncols: _,
        } = symbolic;

        let I = I::truncate;
        let I_checked = |x: usize| -> Result<I, FaerError> {
            if x > I::Signed::MAX.zx() {
                Err(FaerError::IndexOverflow)
            } else {
                Ok(I(x))
            }
        };
        let to_wide = |x: I| -> u128 { x.zx() as _ };
        let from_wide_checked = |x: u128| -> Result<I, FaerError> {
            if x > I::Signed::MAX.zx() as u128 {
                Err(FaerError::IndexOverflow)
            } else {
                Ok(I(x as _))
            }
        };

        let m = A.nrows();
        let n = A.ncols();
        assert!(m >= n);
        assert!(all(AT.nrows() == n, AT.ncols() == m));
        assert!(all(row_perm.len() == m, row_perm_inv.len() == m));
        let n_supernodes = super_etree.len();
        assert!(supernode_postorder.len() == n_supernodes);
        assert!(supernode_postorder_inv.len() == n_supernodes);
        assert!(supernode_ptr.len() == n_supernodes + 1);
        assert!(supernode_ptr[n_supernodes].zx() == n);

        lu.nrows = 0;
        lu.ncols = 0;
        lu.nsupernodes = 0;
        lu.supernode_ptr.clear();

        let (col_global_to_local, stack) = stack.make_raw::<I>(n);
        let (row_global_to_local, stack) = stack.make_raw::<I>(m);
        let (marked, stack) = stack.make_raw::<I>(m);
        let (indices, stack) = stack.make_raw::<I>(m);
        let (transpositions, stack) = stack.make_raw::<I>(m);
        let (d_active_rows, _) = stack.make_raw::<I>(m);

        mem::fill_none::<I::Signed>(bytemuck::cast_slice_mut(col_global_to_local));
        mem::fill_none::<I::Signed>(bytemuck::cast_slice_mut(row_global_to_local));

        mem::fill_zero(marked);

        resize_index(&mut lu.l_col_ptr_for_row_ind, n_supernodes + 1, true, false)?;
        resize_index(
            &mut lu.ut_col_ptr_for_row_ind,
            n_supernodes + 1,
            true,
            false,
        )?;
        resize_index(&mut lu.l_col_ptr_for_val, n_supernodes + 1, true, false)?;
        resize_index(&mut lu.ut_col_ptr_for_val, n_supernodes + 1, true, false)?;

        lu.l_col_ptr_for_row_ind[0] = I(0);
        lu.ut_col_ptr_for_row_ind[0] = I(0);
        lu.l_col_ptr_for_val[0] = I(0);
        lu.ut_col_ptr_for_val[0] = I(0);

        for i in 0..m {
            row_perm[i] = I(i);
        }
        for i in 0..m {
            row_perm_inv[i] = I(i);
        }

        let (col_perm, col_perm_inv) = col_perm.into_arrays();

        let mut contrib_work = (0..n_supernodes)
            .map(|_| {
                (
                    E::faer_map(E::UNIT, |()| alloc::vec::Vec::<MaybeUninit<E::Unit>>::new()),
                    alloc::vec::Vec::<I>::new(),
                    0usize,
                    MatU8::new(0, 0),
                )
            })
            .collect::<alloc::vec::Vec<_>>();

        let work_is_empty = |v: &GroupFor<E, alloc::vec::Vec<MaybeUninit<E::Unit>>>| {
            let mut is_empty = false;
            E::faer_map(E::faer_as_ref(v), |v| is_empty |= v.is_empty());
            is_empty
        };

        let work_make_empty = |v: &mut GroupFor<E, alloc::vec::Vec<MaybeUninit<E::Unit>>>| {
            E::faer_map(E::faer_as_mut(v), |v| *v = alloc::vec::Vec::new());
        };

        let work_to_mat_mut = |v: &mut GroupFor<E, alloc::vec::Vec<MaybeUninit<E::Unit>>>,
                               nrows: usize,
                               ncols: usize| unsafe {
            faer_core::mat::from_raw_parts_mut::<'_, E>(
                E::faer_map(E::faer_as_mut(v), |v| v.as_mut_ptr() as *mut E::Unit),
                nrows,
                ncols,
                1,
                nrows as isize,
            )
        };

        let mut A_leftover = A.compute_nnz();
        for s in 0..n_supernodes {
            let s_begin = supernode_ptr[s].zx();
            let s_end = supernode_ptr[s + 1].zx();
            let s_size = s_end - s_begin;

            let s_postordered = supernode_postorder_inv[s].zx();
            let desc_count = descendant_count[s].zx();
            let mut s_row_index_count = 0usize;
            let (left_contrib, right_contrib) = contrib_work.split_at_mut(s);

            let s_row_indices = &mut *indices;
            // add the rows from A[s_end:, s_begin:s_end]
            for j in s_begin..s_end {
                let pj = col_perm[j].zx();
                let row_ind = A.row_indices_of_col_raw(pj);
                for i in row_ind {
                    let i = i.zx();
                    let pi = row_perm_inv[i].zx();
                    if pi < s_begin {
                        continue;
                    }
                    if marked[i] < I(2 * s + 1) {
                        s_row_indices[s_row_index_count] = I(i);
                        s_row_index_count += 1;
                        marked[i] = I(2 * s + 1);
                    }
                }
            }

            // add the rows from child[s_begin:]
            for d in &supernode_postorder[s_postordered - desc_count..s_postordered] {
                let d = d.zx();
                let d_begin = supernode_ptr[d].zx();
                let d_end = supernode_ptr[d + 1].zx();
                let d_size = d_end - d_begin;
                let d_row_ind = &lu.l_row_ind
                    [lu.l_col_ptr_for_row_ind[d].zx()..lu.l_col_ptr_for_row_ind[d + 1].zx()]
                    [d_size..];
                let d_col_ind = &lu.ut_row_ind
                    [lu.ut_col_ptr_for_row_ind[d].zx()..lu.ut_col_ptr_for_row_ind[d + 1].zx()];
                let d_col_start = d_col_ind.partition_point(partition_fn(s_begin));

                if d_col_start < d_col_ind.len() && d_col_ind[d_col_start].zx() < s_end {
                    for i in d_row_ind.iter() {
                        let i = i.zx();
                        let pi = row_perm_inv[i].zx();

                        if pi < s_begin {
                            continue;
                        }

                        if marked[i] < I(2 * s + 1) {
                            s_row_indices[s_row_index_count] = I(i);
                            s_row_index_count += 1;
                            marked[i] = I(2 * s + 1);
                        }
                    }
                }
            }

            lu.l_col_ptr_for_row_ind[s + 1] =
                I_checked(lu.l_col_ptr_for_row_ind[s].zx() + s_row_index_count)?;
            lu.l_col_ptr_for_val[s + 1] = from_wide_checked(
                to_wide(lu.l_col_ptr_for_val[s]) + ((s_row_index_count) as u128 * s_size as u128),
            )?;
            resize_index(
                &mut lu.l_row_ind,
                lu.l_col_ptr_for_row_ind[s + 1].zx(),
                false,
                false,
            )?;
            resize_scalar::<E>(
                &mut lu.l_val,
                lu.l_col_ptr_for_val[s + 1].zx(),
                false,
                false,
            )?;
            lu.l_row_ind[lu.l_col_ptr_for_row_ind[s].zx()..lu.l_col_ptr_for_row_ind[s + 1].zx()]
                .copy_from_slice(&s_row_indices[..s_row_index_count]);
            lu.l_row_ind[lu.l_col_ptr_for_row_ind[s].zx()..lu.l_col_ptr_for_row_ind[s + 1].zx()]
                .sort_unstable();

            let (left_row_indices, right_row_indices) =
                lu.l_row_ind.split_at_mut(lu.l_col_ptr_for_row_ind[s].zx());

            let s_row_indices = &mut right_row_indices
                [0..lu.l_col_ptr_for_row_ind[s + 1].zx() - lu.l_col_ptr_for_row_ind[s].zx()];
            for (idx, i) in s_row_indices.iter().enumerate() {
                row_global_to_local[i.zx()] = I(idx);
            }
            let s_L = lu
                .l_val
                .as_slice_mut()
                .subslice(lu.l_col_ptr_for_val[s].zx()..lu.l_col_ptr_for_val[s + 1].zx());
            let mut s_L = faer_core::mat::from_column_major_slice_mut::<'_, E>(
                s_L.into_inner(),
                s_row_index_count,
                s_size,
            );
            s_L.fill_zero();

            for j in s_begin..s_end {
                let pj = col_perm[j].zx();
                let row_ind = A.row_indices_of_col(pj);
                let val = SliceGroup::<'_, E>::new(A.values_of_col(pj)).into_ref_iter();

                for (i, val) in zip(row_ind, val) {
                    let pi = row_perm_inv[i].zx();
                    let val = val.read();
                    if pi < s_begin {
                        continue;
                    }
                    assert!(A_leftover > 0);
                    A_leftover -= 1;
                    let ix = row_global_to_local[i].zx();
                    let iy = j - s_begin;
                    s_L.write(ix, iy, s_L.read(ix, iy).faer_add(val));
                }
            }

            noinline(LPanel, || {
                for d in &supernode_postorder[s_postordered - desc_count..s_postordered] {
                    let d = d.zx();
                    if work_is_empty(&left_contrib[d].0) {
                        continue;
                    }

                    let d_begin = supernode_ptr[d].zx();
                    let d_end = supernode_ptr[d + 1].zx();
                    let d_size = d_end - d_begin;
                    let d_row_ind = &left_row_indices
                        [lu.l_col_ptr_for_row_ind[d].zx()..lu.l_col_ptr_for_row_ind[d + 1].zx()]
                        [d_size..];
                    let d_col_ind = &lu.ut_row_ind
                        [lu.ut_col_ptr_for_row_ind[d].zx()..lu.ut_col_ptr_for_row_ind[d + 1].zx()];
                    let d_col_start = d_col_ind.partition_point(partition_fn(s_begin));

                    if d_col_start < d_col_ind.len() && d_col_ind[d_col_start].zx() < s_end {
                        let d_col_mid = d_col_start
                            + d_col_ind[d_col_start..].partition_point(partition_fn(s_end));

                        let mut d_LU_cols = work_to_mat_mut(
                            &mut left_contrib[d].0,
                            d_row_ind.len(),
                            d_col_ind.len(),
                        )
                        .subcols_mut(d_col_start, d_col_mid - d_col_start);
                        let d_active = &mut left_contrib[d].1[d_col_start..];
                        let d_active_count = &mut left_contrib[d].2;
                        let d_active_mat = &mut left_contrib[d].3;

                        for (d_j, j) in d_col_ind[d_col_start..d_col_mid].iter().enumerate() {
                            if d_active[d_j] > I(0) {
                                let mut taken_rows = 0usize;
                                let j = j.zx();
                                let s_j = j - s_begin;
                                for (d_i, i) in d_row_ind.iter().enumerate() {
                                    let i = i.zx();
                                    let pi = row_perm_inv[i].zx();
                                    if pi < s_begin {
                                        continue;
                                    }
                                    let s_i = row_global_to_local[i].zx();

                                    s_L.write(
                                        s_i,
                                        s_j,
                                        s_L.read(s_i, s_j).faer_sub(d_LU_cols.read(d_i, d_j)),
                                    );
                                    d_LU_cols.write(d_i, d_j, E::faer_zero());
                                    taken_rows += d_active_mat[(d_i, d_j + d_col_start)] as usize;
                                    d_active_mat[(d_i, d_j + d_col_start)] = 0;
                                }
                                assert!(d_active[d_j] >= I(taken_rows));
                                d_active[d_j] -= I(taken_rows);
                                if d_active[d_j] == I(0) {
                                    assert!(*d_active_count > 0);
                                    *d_active_count -= 1;
                                }
                            }
                        }
                        if *d_active_count == 0 {
                            work_make_empty(&mut left_contrib[d].0);
                            left_contrib[d].1 = alloc::vec::Vec::new();
                            left_contrib[d].2 = 0;
                            left_contrib[d].3 = MatU8::new(0, 0);
                        }
                    }
                }
            });

            if s_L.nrows() < s_L.ncols() {
                return Err(LuError::SymbolicSingular(s_begin + s_L.nrows()));
            }
            let transpositions = &mut transpositions[s_begin..s_end];
            faer_lu::partial_pivoting::compute::lu_in_place_impl(
                s_L.rb_mut(),
                0,
                s_size,
                transpositions,
                parallelism,
            );
            for (idx, t) in transpositions.iter().enumerate() {
                let i_t = s_row_indices[idx + t.zx()].zx();
                let kk = row_perm_inv[i_t].zx();
                row_perm.swap(s_begin + idx, row_perm_inv[i_t].zx());
                row_perm_inv.swap(row_perm[s_begin + idx].zx(), row_perm[kk].zx());
                s_row_indices.swap(idx, idx + t.zx());
            }
            for (idx, t) in transpositions.iter().enumerate().rev() {
                row_global_to_local.swap(s_row_indices[idx].zx(), s_row_indices[idx + t.zx()].zx());
            }
            for (idx, i) in s_row_indices.iter().enumerate() {
                assert!(row_global_to_local[i.zx()] == I(idx));
            }

            let s_col_indices = &mut indices[..n];
            let mut s_col_index_count = 0usize;
            for i in s_begin..s_end {
                let pi = row_perm[i].zx();
                for j in AT.row_indices_of_col(pi) {
                    let pj = col_perm_inv[j].zx();
                    if pj < s_end {
                        continue;
                    }
                    if marked[pj] < I(2 * s + 2) {
                        s_col_indices[s_col_index_count] = I(pj);
                        s_col_index_count += 1;
                        marked[pj] = I(2 * s + 2);
                    }
                }
            }

            for d in &supernode_postorder[s_postordered - desc_count..s_postordered] {
                let d = d.zx();

                let d_begin = supernode_ptr[d].zx();
                let d_end = supernode_ptr[d + 1].zx();
                let d_size = d_end - d_begin;

                let d_row_ind = &left_row_indices
                    [lu.l_col_ptr_for_row_ind[d].zx()..lu.l_col_ptr_for_row_ind[d + 1].zx()]
                    [d_size..];
                let d_col_ind = &lu.ut_row_ind
                    [lu.ut_col_ptr_for_row_ind[d].zx()..lu.ut_col_ptr_for_row_ind[d + 1].zx()];

                let contributes_to_u = d_row_ind.iter().any(|&i| {
                    row_perm_inv[i.zx()].zx() >= s_begin && row_perm_inv[i.zx()].zx() < s_end
                });

                if contributes_to_u {
                    let d_col_start = d_col_ind.partition_point(partition_fn(s_end));
                    for j in &d_col_ind[d_col_start..] {
                        let j = j.zx();
                        if marked[j] < I(2 * s + 2) {
                            s_col_indices[s_col_index_count] = I(j);
                            s_col_index_count += 1;
                            marked[j] = I(2 * s + 2);
                        }
                    }
                }
            }

            lu.ut_col_ptr_for_row_ind[s + 1] =
                I_checked(lu.ut_col_ptr_for_row_ind[s].zx() + s_col_index_count)?;
            lu.ut_col_ptr_for_val[s + 1] = from_wide_checked(
                to_wide(lu.ut_col_ptr_for_val[s]) + (s_col_index_count as u128 * s_size as u128),
            )?;
            resize_index(
                &mut lu.ut_row_ind,
                lu.ut_col_ptr_for_row_ind[s + 1].zx(),
                false,
                false,
            )?;
            resize_scalar::<E>(
                &mut lu.ut_val,
                lu.ut_col_ptr_for_val[s + 1].zx(),
                false,
                false,
            )?;
            lu.ut_row_ind[lu.ut_col_ptr_for_row_ind[s].zx()..lu.ut_col_ptr_for_row_ind[s + 1].zx()]
                .copy_from_slice(&s_col_indices[..s_col_index_count]);
            lu.ut_row_ind[lu.ut_col_ptr_for_row_ind[s].zx()..lu.ut_col_ptr_for_row_ind[s + 1].zx()]
                .sort_unstable();

            let s_col_indices = &lu.ut_row_ind
                [lu.ut_col_ptr_for_row_ind[s].zx()..lu.ut_col_ptr_for_row_ind[s + 1].zx()];
            for (idx, j) in s_col_indices.iter().enumerate() {
                col_global_to_local[j.zx()] = I(idx);
            }

            let s_U = lu
                .ut_val
                .as_slice_mut()
                .subslice(lu.ut_col_ptr_for_val[s].zx()..lu.ut_col_ptr_for_val[s + 1].zx());
            let mut s_U = faer_core::mat::from_column_major_slice_mut::<'_, E>(
                s_U.into_inner(),
                s_col_index_count,
                s_size,
            )
            .transpose_mut();
            s_U.fill_zero();

            for i in s_begin..s_end {
                let pi = row_perm[i].zx();
                for (j, val) in zip(
                    AT.row_indices_of_col(pi),
                    SliceGroup::<'_, E>::new(AT.values_of_col(pi)).into_ref_iter(),
                ) {
                    let pj = col_perm_inv[j].zx();
                    let val = val.read();
                    if pj < s_end {
                        continue;
                    }
                    assert!(A_leftover > 0);
                    A_leftover -= 1;
                    let ix = i - s_begin;
                    let iy = col_global_to_local[pj].zx();
                    s_U.write(ix, iy, s_U.read(ix, iy).faer_add(val));
                }
            }

            noinline(UPanel, || {
                for d in &supernode_postorder[s_postordered - desc_count..s_postordered] {
                    let d = d.zx();
                    if work_is_empty(&left_contrib[d].0) {
                        continue;
                    }

                    let d_begin = supernode_ptr[d].zx();
                    let d_end = supernode_ptr[d + 1].zx();
                    let d_size = d_end - d_begin;

                    let d_row_ind = &left_row_indices
                        [lu.l_col_ptr_for_row_ind[d].zx()..lu.l_col_ptr_for_row_ind[d + 1].zx()]
                        [d_size..];
                    let d_col_ind = &lu.ut_row_ind
                        [lu.ut_col_ptr_for_row_ind[d].zx()..lu.ut_col_ptr_for_row_ind[d + 1].zx()];

                    let contributes_to_u = d_row_ind.iter().any(|&i| {
                        row_perm_inv[i.zx()].zx() >= s_begin && row_perm_inv[i.zx()].zx() < s_end
                    });

                    if contributes_to_u {
                        let d_col_start = d_col_ind.partition_point(partition_fn(s_end));
                        let d_LU = work_to_mat_mut(
                            &mut left_contrib[d].0,
                            d_row_ind.len(),
                            d_col_ind.len(),
                        );
                        let mut d_LU = d_LU.get_mut(.., d_col_start..);
                        let d_active = &mut left_contrib[d].1[d_col_start..];
                        let d_active_count = &mut left_contrib[d].2;
                        let d_active_mat = &mut left_contrib[d].3;

                        for (d_j, j) in d_col_ind[d_col_start..].iter().enumerate() {
                            if d_active[d_j] > I(0) {
                                let mut taken_rows = 0usize;
                                let j = j.zx();
                                let s_j = col_global_to_local[j].zx();
                                for (d_i, i) in d_row_ind.iter().enumerate() {
                                    let i = i.zx();
                                    let pi = row_perm_inv[i].zx();
                                    if pi >= s_begin && pi < s_end {
                                        let s_i = row_global_to_local[i].zx();
                                        s_U.write(
                                            s_i,
                                            s_j,
                                            s_U.read(s_i, s_j).faer_sub(d_LU.read(d_i, d_j)),
                                        );
                                        d_LU.write(d_i, d_j, E::faer_zero());
                                        taken_rows +=
                                            d_active_mat[(d_i, d_j + d_col_start)] as usize;
                                        d_active_mat[(d_i, d_j + d_col_start)] = 0;
                                    }
                                }
                                assert!(d_active[d_j] >= I(taken_rows));
                                d_active[d_j] -= I(taken_rows);
                                if d_active[d_j] == I(0) {
                                    assert!(*d_active_count > 0);
                                    *d_active_count -= 1;
                                }
                            }
                        }
                        if *d_active_count == 0 {
                            work_make_empty(&mut left_contrib[d].0);
                            left_contrib[d].1 = alloc::vec::Vec::new();
                            left_contrib[d].2 = 0;
                            left_contrib[d].3 = MatU8::new(0, 0);
                        }
                    }
                }
            });
            faer_core::solve::solve_unit_lower_triangular_in_place(
                s_L.rb().subrows(0, s_size),
                s_U.rb_mut(),
                parallelism,
            );

            if s_row_index_count > s_size && s_col_index_count > 0 {
                resize_maybe_uninit_scalar::<E>(
                    &mut right_contrib[0].0,
                    from_wide_checked(
                        to_wide(I(s_row_index_count - s_size)) * to_wide(I(s_col_index_count)),
                    )?
                    .zx(),
                )?;
                right_contrib[0]
                    .1
                    .resize(s_col_index_count, I(s_row_index_count - s_size));
                right_contrib[0].2 = s_col_index_count;
                right_contrib[0].3 = MatU8::new(s_row_index_count - s_size, s_col_index_count);

                let mut s_LU = work_to_mat_mut(
                    &mut right_contrib[0].0,
                    s_row_index_count - s_size,
                    s_col_index_count,
                );
                mul::matmul(
                    s_LU.rb_mut(),
                    s_L.rb().get(s_size.., ..),
                    s_U.rb(),
                    None,
                    E::faer_one(),
                    parallelism,
                );

                noinline(Front, || {
                    for d in &supernode_postorder[s_postordered - desc_count..s_postordered] {
                        let d = d.zx();
                        if work_is_empty(&left_contrib[d].0) {
                            continue;
                        }

                        let d_begin = supernode_ptr[d].zx();
                        let d_end = supernode_ptr[d + 1].zx();
                        let d_size = d_end - d_begin;

                        let d_row_ind = &left_row_indices[lu.l_col_ptr_for_row_ind[d].zx()
                            ..lu.l_col_ptr_for_row_ind[d + 1].zx()][d_size..];
                        let d_col_ind = &lu.ut_row_ind[lu.ut_col_ptr_for_row_ind[d].zx()
                            ..lu.ut_col_ptr_for_row_ind[d + 1].zx()];

                        let contributes_to_front = d_row_ind
                            .iter()
                            .any(|&i| row_perm_inv[i.zx()].zx() >= s_end);

                        if contributes_to_front {
                            let d_col_start = d_col_ind.partition_point(partition_fn(s_end));
                            let d_LU = work_to_mat_mut(
                                &mut left_contrib[d].0,
                                d_row_ind.len(),
                                d_col_ind.len(),
                            );
                            let mut d_LU = d_LU.get_mut(.., d_col_start..);
                            let d_active = &mut left_contrib[d].1[d_col_start..];
                            let d_active_count = &mut left_contrib[d].2;
                            let d_active_mat = &mut left_contrib[d].3;

                            let mut d_active_row_count = 0usize;
                            let mut first_iter = true;

                            for (d_j, j) in d_col_ind[d_col_start..].iter().enumerate() {
                                if d_active[d_j] > I(0) {
                                    if first_iter {
                                        first_iter = false;
                                        for (d_i, i) in d_row_ind.iter().enumerate() {
                                            let i = i.zx();
                                            let pi = row_perm_inv[i].zx();
                                            if (pi < s_end) || (row_global_to_local[i] == I(NONE)) {
                                                continue;
                                            }

                                            d_active_rows[d_active_row_count] = I(d_i);
                                            d_active_row_count += 1;
                                        }
                                    }

                                    let j = j.zx();
                                    let mut taken_rows = 0usize;

                                    let s_j = col_global_to_local[j];
                                    if s_j == I(NONE) {
                                        continue;
                                    }
                                    let s_j = s_j.zx();
                                    let mut dst = s_LU.rb_mut().col_mut(s_j);
                                    let mut src = d_LU.rb_mut().col_mut(d_j);
                                    assert!(dst.row_stride() == 1);
                                    assert!(src.row_stride() == 1);

                                    for d_i in &d_active_rows[..d_active_row_count] {
                                        let d_i = d_i.zx();
                                        let i = d_row_ind[d_i].zx();
                                        let d_active_mat =
                                            &mut d_active_mat[(d_i, d_j + d_col_start)];
                                        if *d_active_mat == 0 {
                                            continue;
                                        }
                                        let s_i = row_global_to_local[i].zx() - s_size;
                                        unsafe {
                                            dst.write_unchecked(
                                                s_i,
                                                dst.read_unchecked(s_i)
                                                    .faer_add(src.read_unchecked(d_i)),
                                            );
                                            src.write_unchecked(d_i, E::faer_zero());
                                        }
                                        taken_rows += 1;
                                        *d_active_mat = 0;
                                    }

                                    d_active[d_j] -= I(taken_rows);
                                    if d_active[d_j] == I(0) {
                                        *d_active_count -= 1;
                                    }
                                }
                            }
                            if *d_active_count == 0 {
                                work_make_empty(&mut left_contrib[d].0);
                                left_contrib[d].1 = alloc::vec::Vec::new();
                                left_contrib[d].2 = 0;
                                left_contrib[d].3 = MatU8::new(0, 0);
                            }
                        }
                    }
                })
            }

            for i in s_row_indices.iter() {
                row_global_to_local[i.zx()] = I(NONE);
            }
            for j in s_col_indices.iter() {
                col_global_to_local[j.zx()] = I(NONE);
            }
        }
        assert!(A_leftover == 0);

        for idx in &mut lu.l_row_ind[..lu.l_col_ptr_for_row_ind[n_supernodes].zx()] {
            *idx = row_perm_inv[idx.zx()];
        }

        lu.nrows = m;
        lu.ncols = n;
        lu.nsupernodes = n_supernodes;
        lu.supernode_ptr.clone_from(supernode_ptr);

        Ok(())
    }
}

pub mod simplicial {
    use crate::triangular_solve;

    use super::*;
    use faer_core::assert;

    #[derive(Debug, Clone)]
    pub struct SimplicialLu<I, E: Entity> {
        nrows: usize,
        ncols: usize,

        l_col_ptr: alloc::vec::Vec<I>,
        l_row_ind: alloc::vec::Vec<I>,
        l_val: VecGroup<E>,

        u_col_ptr: alloc::vec::Vec<I>,
        u_row_ind: alloc::vec::Vec<I>,
        u_val: VecGroup<E>,
    }

    impl<I: Index, E: Entity> SimplicialLu<I, E> {
        #[inline]
        pub fn new() -> Self {
            Self {
                nrows: 0,
                ncols: 0,

                l_col_ptr: alloc::vec::Vec::new(),
                u_col_ptr: alloc::vec::Vec::new(),

                l_row_ind: alloc::vec::Vec::new(),
                u_row_ind: alloc::vec::Vec::new(),

                l_val: VecGroup::new(),
                u_val: VecGroup::new(),
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
        pub fn l_factor(&self) -> SparseColMatRef<'_, I, E> {
            SparseColMatRef::<'_, I, E>::new(
                unsafe {
                    SymbolicSparseColMatRef::new_unchecked(
                        self.nrows(),
                        self.ncols(),
                        &self.l_col_ptr,
                        None,
                        &self.l_row_ind,
                    )
                },
                self.l_val.as_slice().into_inner(),
            )
        }

        #[inline]
        pub fn u_factor(&self) -> SparseColMatRef<'_, I, E> {
            SparseColMatRef::<'_, I, E>::new(
                unsafe {
                    SymbolicSparseColMatRef::new_unchecked(
                        self.ncols(),
                        self.ncols(),
                        &self.u_col_ptr,
                        None,
                        &self.u_row_ind,
                    )
                },
                self.u_val.as_slice().into_inner(),
            )
        }

        #[track_caller]
        pub fn solve_in_place_with_conj(
            &self,
            row_perm: PermutationRef<'_, I, E>,
            col_perm: PermutationRef<'_, I, E>,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            work: MatMut<'_, E>,
        ) where
            E: ComplexField,
        {
            assert!(self.nrows() == self.ncols());
            assert!(self.nrows() == rhs.nrows());
            let mut X = rhs;
            let mut temp = work;

            let l = self.l_factor();
            let u = self.u_factor();

            faer_core::permutation::permute_rows(temp.rb_mut(), X.rb(), row_perm);
            triangular_solve::solve_unit_lower_triangular_in_place(
                l,
                conj_lhs,
                temp.rb_mut(),
                parallelism,
            );
            triangular_solve::solve_upper_triangular_in_place(
                u,
                conj_lhs,
                temp.rb_mut(),
                parallelism,
            );
            faer_core::permutation::permute_rows(X.rb_mut(), temp.rb(), col_perm.inverse());
        }

        #[track_caller]
        pub fn solve_transpose_in_place_with_conj(
            &self,
            row_perm: PermutationRef<'_, I, E>,
            col_perm: PermutationRef<'_, I, E>,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            work: MatMut<'_, E>,
        ) where
            E: ComplexField,
        {
            assert!(all(
                self.nrows() == self.ncols(),
                self.nrows() == rhs.nrows()
            ));
            let mut X = rhs;
            let mut temp = work;

            let l = self.l_factor();
            let u = self.u_factor();

            faer_core::permutation::permute_rows(temp.rb_mut(), X.rb(), col_perm);
            triangular_solve::solve_upper_triangular_transpose_in_place(
                u,
                conj_lhs,
                temp.rb_mut(),
                parallelism,
            );
            triangular_solve::solve_unit_lower_triangular_transpose_in_place(
                l,
                conj_lhs,
                temp.rb_mut(),
                parallelism,
            );
            faer_core::permutation::permute_rows(X.rb_mut(), temp.rb(), row_perm.inverse());
        }
    }

    fn depth_first_search<I: Index>(
        marked: &mut [I],
        mark: I,

        xi: &mut [I],
        l: SymbolicSparseColMatRef<'_, I>,
        row_perm_inv: &[I],
        b: usize,
        stack: &mut [I],
    ) -> usize {
        let I = I::truncate;

        let mut tail_start = xi.len();
        let mut head_len = 1usize;
        xi[0] = I(b);

        let li = l.row_indices();

        'dfs_loop: while head_len > 0 {
            let b = xi[head_len - 1].zx().zx();
            let pb = row_perm_inv[b].zx();

            let range = if pb < l.ncols() {
                l.col_range(pb)
            } else {
                0..0
            };
            if marked[b] < mark {
                marked[b] = mark;
                stack[head_len - 1] = I(range.start);
            }

            let start = stack[head_len - 1].zx();
            let end = range.end;
            for ptr in start..end {
                let i = li[ptr].zx();
                if marked[i] == mark {
                    continue;
                }
                stack[head_len - 1] = I(ptr);
                xi[head_len] = I(i);
                head_len += 1;
                continue 'dfs_loop;
            }

            head_len -= 1;
            tail_start -= 1;
            xi[tail_start] = I(b);
        }

        tail_start
    }

    fn reach<I: Index>(
        marked: &mut [I],
        mark: I,

        xi: &mut [I],
        l: SymbolicSparseColMatRef<'_, I>,
        row_perm_inv: &[I],
        bi: &[I],
        stack: &mut [I],
    ) -> usize {
        let n = l.nrows();
        let mut tail_start = n;

        for b in bi {
            let b = b.zx();
            if marked[b] < mark {
                tail_start = depth_first_search(
                    marked,
                    mark,
                    &mut xi[..tail_start],
                    l,
                    row_perm_inv,
                    b,
                    stack,
                );
            }
        }

        tail_start
    }

    fn l_incomplete_solve_sparse<I: Index, E: ComplexField>(
        marked: &mut [I],
        mark: I,

        xi: &mut [I],
        x: SliceGroupMut<'_, E>,
        l: SparseColMatRef<'_, I, E>,
        row_perm_inv: &[I],
        bi: &[I],
        bx: SliceGroup<'_, E>,
        stack: &mut [I],
    ) -> usize {
        let mut x = x;
        let tail_start = reach(marked, mark, xi, l.symbolic(), row_perm_inv, bi, stack);

        let xi = &xi[tail_start..];
        for (i, b) in zip(bi, bx.into_ref_iter()) {
            let i = i.zx();
            x.write(i, x.read(i).faer_add(b.read()));
        }

        for i in xi {
            let i = i.zx();
            let pi = row_perm_inv[i].zx();
            if pi >= l.ncols() {
                continue;
            }

            let li = l.row_indices_of_col_raw(pi);
            let lx = SliceGroup::<'_, E>::new(l.values_of_col(pi));
            let len = li.len();

            let xi = x.read(i);
            for (li, lx) in zip(&li[1..], lx.subslice(1..len).into_ref_iter()) {
                let li = li.zx();
                let lx = lx.read();
                x.write(li, x.read(li).faer_sub(E::faer_mul(lx, xi)));
            }
        }

        tail_start
    }

    pub fn factorize_simplicial_numeric_lu_req<I: Index, E: Entity>(
        nrows: usize,
        ncols: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let idx = StackReq::try_new::<I>(nrows)?;
        let val = crate::make_raw_req::<E>(nrows)?;
        let _ = ncols;
        StackReq::try_all_of([val, idx, idx, idx])
    }

    pub fn factorize_simplicial_numeric_lu<I: Index, E: ComplexField>(
        row_perm: &mut [I],
        row_perm_inv: &mut [I],
        lu: &mut SimplicialLu<I, E>,

        A: SparseColMatRef<'_, I, E>,
        col_perm: PermutationRef<'_, I, E>,
        stack: PodStack<'_>,
    ) -> Result<(), LuError> {
        let I = I::truncate;

        assert!(all(
            A.nrows() == row_perm.len(),
            A.nrows() == row_perm_inv.len(),
            A.ncols() == col_perm.len(),
            A.nrows() == A.ncols()
        ));

        lu.nrows = 0;
        lu.ncols = 0;

        let m = A.nrows();
        let n = A.ncols();

        resize_index(&mut lu.l_col_ptr, n + 1, true, false)?;
        resize_index(&mut lu.u_col_ptr, n + 1, true, false)?;

        let (mut x, stack) = crate::make_raw::<E>(m, stack);
        let (marked, stack) = stack.make_raw::<I>(m);
        let (xj, stack) = stack.make_raw::<I>(m);
        let (stack, _) = stack.make_raw::<I>(m);

        mem::fill_zero(marked);
        x.fill_zero();

        row_perm_inv.fill(I(n));

        let mut l_pos = 0usize;
        let mut u_pos = 0usize;
        lu.l_col_ptr[0] = I(0);
        lu.u_col_ptr[0] = I(0);
        for j in 0..n {
            let l = SparseColMatRef::<'_, I, E>::new(
                unsafe {
                    SymbolicSparseColMatRef::new_unchecked(
                        m,
                        j,
                        &lu.l_col_ptr[..j + 1],
                        None,
                        &lu.l_row_ind,
                    )
                },
                lu.l_val.as_slice().into_inner(),
            );

            let pj = col_perm.into_arrays().0[j].zx();
            let tail_start = l_incomplete_solve_sparse(
                marked,
                I(j + 1),
                xj,
                x.rb_mut(),
                l,
                row_perm_inv,
                A.row_indices_of_col_raw(pj),
                SliceGroup::new(A.values_of_col(pj)),
                stack,
            );
            let xj = &xj[tail_start..];

            resize_scalar::<E>(&mut lu.l_val, l_pos + xj.len() + 1, false, false)?;
            resize_index(&mut lu.l_row_ind, l_pos + xj.len() + 1, false, false)?;
            resize_scalar::<E>(&mut lu.u_val, u_pos + xj.len() + 1, false, false)?;
            resize_index(&mut lu.u_row_ind, u_pos + xj.len() + 1, false, false)?;

            let mut l_val = lu.l_val.as_slice_mut();
            let mut u_val = lu.u_val.as_slice_mut();

            let mut pivot_idx = n;
            let mut pivot_val = E::Real::faer_one().faer_neg();
            for i in xj {
                let i = i.zx();
                let xi = x.read(i);
                if row_perm_inv[i] == I(n) {
                    let val = xi.faer_abs();
                    if matches!(
                        val.partial_cmp(&pivot_val),
                        None | Some(core::cmp::Ordering::Greater)
                    ) {
                        pivot_idx = i;
                        pivot_val = val;
                    }
                } else {
                    lu.u_row_ind[u_pos] = row_perm_inv[i];
                    u_val.write(u_pos, xi);
                    u_pos += 1;
                }
            }
            if pivot_idx == n {
                return Err(LuError::SymbolicSingular(j));
            }

            let x_piv = x.read(pivot_idx);
            if x_piv == E::faer_zero() {
                panic!();
            }
            row_perm_inv[pivot_idx] = I(j);

            lu.u_row_ind[u_pos] = I(j);
            u_val.write(u_pos, x_piv);
            u_pos += 1;
            lu.u_col_ptr[j + 1] = I(u_pos);

            lu.l_row_ind[l_pos] = I(pivot_idx);
            l_val.write(l_pos, E::faer_one());
            l_pos += 1;

            let x_piv_inv = x_piv.faer_inv();
            for i in xj {
                let i = i.zx();
                let xi = x.read(i);
                if row_perm_inv[i] == I(n) {
                    lu.l_row_ind[l_pos] = I(i);
                    l_val.write(l_pos, xi.faer_mul(x_piv_inv));
                    l_pos += 1;
                }
                x.write(i, E::faer_zero());
            }
            lu.l_col_ptr[j + 1] = I(l_pos);
        }

        for i in &mut lu.l_row_ind[..l_pos] {
            *i = row_perm_inv[(*i).zx()];
        }

        for (idx, p) in row_perm_inv.iter().enumerate() {
            row_perm[p.zx()] = I(idx);
        }

        faer_core::sparse::sort_indices::<I, E>(
            &lu.l_col_ptr,
            &mut lu.l_row_ind,
            lu.l_val.as_slice_mut().into_inner(),
        );
        faer_core::sparse::sort_indices::<I, E>(
            &lu.u_col_ptr,
            &mut lu.u_row_ind,
            lu.u_val.as_slice_mut().into_inner(),
        );

        lu.nrows = m;
        lu.ncols = n;

        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct LuSymbolicParams<'a> {
    pub colamd_params: Control,
    pub supernodal_flop_ratio_threshold: SupernodalThreshold,
    pub supernodal_params: SymbolicSupernodalParams<'a>,
}

/// The inner factorization used for the symbolic LU, either simplicial or symbolic.
#[derive(Debug, Clone)]
pub enum SymbolicLuRaw<I> {
    Simplicial { nrows: usize, ncols: usize },
    Supernodal(supernodal::SymbolicSupernodalLu<I>),
}

/// The symbolic structure of a sparse LU decomposition.
#[derive(Debug, Clone)]
pub struct SymbolicLu<I> {
    raw: SymbolicLuRaw<I>,
    col_perm_fwd: alloc::vec::Vec<I>,
    col_perm_inv: alloc::vec::Vec<I>,
    A_nnz: usize,
}

#[derive(Debug, Clone)]
enum NumericLuRaw<I, E: Entity> {
    None,
    Supernodal(supernodal::SupernodalLu<I, E>),
    Simplicial(simplicial::SimplicialLu<I, E>),
}

#[derive(Debug, Clone)]
pub struct NumericLu<I, E: Entity> {
    raw: NumericLuRaw<I, E>,
    row_perm_fwd: alloc::vec::Vec<I>,
    row_perm_inv: alloc::vec::Vec<I>,
}

impl<I: Index, E: Entity> NumericLu<I, E> {
    #[inline]
    pub fn new() -> Self {
        Self {
            raw: NumericLuRaw::None,
            row_perm_fwd: alloc::vec::Vec::new(),
            row_perm_inv: alloc::vec::Vec::new(),
        }
    }
}

/// Sparse LU factorization wrapper.
#[derive(Debug)]
pub struct LuRef<'a, I: Index, E: Entity> {
    symbolic: &'a SymbolicLu<I>,
    numeric: &'a NumericLu<I, E>,
}
impl_copy!(<'a><I: Index, E: Entity><LuRef<'a, I, E>>);

impl<'a, I: Index, E: Entity> LuRef<'a, I, E> {
    #[inline]
    pub unsafe fn new_unchecked(symbolic: &'a SymbolicLu<I>, numeric: &'a NumericLu<I, E>) -> Self {
        match (&symbolic.raw, &numeric.raw) {
            (SymbolicLuRaw::Simplicial { .. }, NumericLuRaw::Simplicial(_)) => {}
            (SymbolicLuRaw::Supernodal { .. }, NumericLuRaw::Supernodal(_)) => {}
            _ => panic!("incompatible symbolic and numeric variants"),
        }
        Self { symbolic, numeric }
    }

    #[inline]
    pub fn symbolic(self) -> &'a SymbolicLu<I> {
        self.symbolic
    }

    #[inline]
    pub fn row_perm(self) -> PermutationRef<'a, I, E> {
        unsafe {
            PermutationRef::new_unchecked(&self.numeric.row_perm_fwd, &self.numeric.row_perm_inv)
        }
    }

    #[inline]
    pub fn col_perm(self) -> PermutationRef<'a, I, E> {
        self.symbolic.col_perm().cast()
    }

    #[track_caller]
    pub fn solve_in_place_with_conj(
        self,
        conj: Conj,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) where
        E: ComplexField,
    {
        let (work, _) = temp_mat_uninit(rhs.nrows(), rhs.ncols(), stack);
        match (&self.symbolic.raw, &self.numeric.raw) {
            (SymbolicLuRaw::Simplicial { .. }, NumericLuRaw::Simplicial(numeric)) => numeric
                .solve_in_place_with_conj(
                    self.row_perm(),
                    self.col_perm(),
                    conj,
                    rhs,
                    parallelism,
                    work,
                ),
            (SymbolicLuRaw::Supernodal(_), NumericLuRaw::Supernodal(numeric)) => numeric
                .solve_in_place_with_conj(
                    self.row_perm(),
                    self.col_perm(),
                    conj,
                    rhs,
                    parallelism,
                    work,
                ),
            _ => unreachable!(),
        }
    }

    #[track_caller]
    pub fn solve_transpose_in_place_with_conj(
        self,
        conj: Conj,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) where
        E: ComplexField,
    {
        let (work, _) = temp_mat_uninit(rhs.nrows(), rhs.ncols(), stack);
        match (&self.symbolic.raw, &self.numeric.raw) {
            (SymbolicLuRaw::Simplicial { .. }, NumericLuRaw::Simplicial(numeric)) => numeric
                .solve_transpose_in_place_with_conj(
                    self.row_perm(),
                    self.col_perm(),
                    conj,
                    rhs,
                    parallelism,
                    work,
                ),
            (SymbolicLuRaw::Supernodal(_), NumericLuRaw::Supernodal(numeric)) => numeric
                .solve_transpose_in_place_with_conj(
                    self.row_perm(),
                    self.col_perm(),
                    conj,
                    rhs,
                    parallelism,
                    work,
                ),
            _ => unreachable!(),
        }
    }
}

impl<I: Index> SymbolicLu<I> {
    #[inline]
    pub fn nrows(&self) -> usize {
        match &self.raw {
            SymbolicLuRaw::Simplicial { nrows, .. } => *nrows,
            SymbolicLuRaw::Supernodal(this) => this.nrows,
        }
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        match &self.raw {
            SymbolicLuRaw::Simplicial { ncols, .. } => *ncols,
            SymbolicLuRaw::Supernodal(this) => this.ncols,
        }
    }

    /// Returns the fill-reducing column permutation that was computed during symbolic analysis.
    #[inline]
    pub fn col_perm(&self) -> PermutationRef<'_, I, Symbolic> {
        unsafe { PermutationRef::new_unchecked(&self.col_perm_fwd, &self.col_perm_inv) }
    }

    pub fn factorize_numeric_lu_req<E: Entity>(
        &self,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        match &self.raw {
            SymbolicLuRaw::Simplicial { nrows, ncols } => {
                simplicial::factorize_simplicial_numeric_lu_req::<I, E>(*nrows, *ncols)
            }
            SymbolicLuRaw::Supernodal(symbolic) => {
                let _ = parallelism;
                let m = symbolic.nrows;

                let A_nnz = self.A_nnz;
                let AT_req = StackReq::try_all_of([
                    crate::make_raw_req::<E>(A_nnz)?,
                    StackReq::try_new::<I>(m + 1)?,
                    StackReq::try_new::<I>(A_nnz)?,
                ])?;
                StackReq::try_and(
                    AT_req,
                    supernodal::factorize_supernodal_numeric_lu_req::<I, E>(symbolic)?,
                )
            }
        }
    }

    pub fn solve_in_place_req<E: Entity>(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let _ = parallelism;
        temp_mat_req::<E>(self.nrows(), rhs_ncols)
    }

    pub fn solve_transpose_in_place_req<E: Entity>(
        &self,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let _ = parallelism;
        temp_mat_req::<E>(self.nrows(), rhs_ncols)
    }

    #[track_caller]
    pub fn factorize_numeric_lu<'out, E: ComplexField>(
        &'out self,
        numeric: &'out mut NumericLu<I, E>,
        A: SparseColMatRef<'_, I, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) -> Result<LuRef<'out, I, E>, LuError> {
        if matches!(self.raw, SymbolicLuRaw::Simplicial { .. })
            && !matches!(numeric.raw, NumericLuRaw::Simplicial(_))
        {
            numeric.raw = NumericLuRaw::Simplicial(simplicial::SimplicialLu::new());
        }
        if matches!(self.raw, SymbolicLuRaw::Supernodal(_))
            && !matches!(numeric.raw, NumericLuRaw::Supernodal(_))
        {
            numeric.raw = NumericLuRaw::Supernodal(supernodal::SupernodalLu::new());
        }

        let nrows = self.nrows();

        numeric
            .row_perm_fwd
            .try_reserve_exact(nrows.saturating_sub(numeric.row_perm_fwd.len()))
            .map_err(nomem)?;
        numeric
            .row_perm_inv
            .try_reserve_exact(nrows.saturating_sub(numeric.row_perm_inv.len()))
            .map_err(nomem)?;
        numeric.row_perm_fwd.resize(nrows, I::truncate(0));
        numeric.row_perm_inv.resize(nrows, I::truncate(0));

        match (&self.raw, &mut numeric.raw) {
            (SymbolicLuRaw::Simplicial { nrows, ncols }, NumericLuRaw::Simplicial(lu)) => {
                assert!(all(A.nrows() == *nrows, A.ncols() == *ncols));

                simplicial::factorize_simplicial_numeric_lu(
                    &mut numeric.row_perm_fwd,
                    &mut numeric.row_perm_inv,
                    lu,
                    A,
                    self.col_perm().cast(),
                    stack,
                )?;
            }
            (SymbolicLuRaw::Supernodal(symbolic), NumericLuRaw::Supernodal(lu)) => {
                let m = symbolic.nrows;
                let (new_col_ptr, stack) = stack.make_raw::<I>(m + 1);
                let (new_row_ind, stack) = stack.make_raw::<I>(self.A_nnz);
                let (new_values, mut stack) = crate::make_raw::<E>(self.A_nnz, stack);
                let AT = crate::transpose::<I, E>(
                    new_col_ptr,
                    new_row_ind,
                    new_values.into_inner(),
                    A,
                    stack.rb_mut(),
                )
                .into_const();

                supernodal::factorize_supernodal_numeric_lu(
                    &mut numeric.row_perm_fwd,
                    &mut numeric.row_perm_inv,
                    lu,
                    A,
                    AT,
                    self.col_perm().cast(),
                    symbolic,
                    parallelism,
                    stack,
                )?;
            }
            _ => unreachable!(),
        }

        Ok(unsafe { LuRef::new_unchecked(self, numeric) })
    }
}

/// Computes the symbolic LU factorization of the matrix `A`, or returns an error if the
/// operation could not be completed.
#[track_caller]
pub fn factorize_symbolic_lu<I: Index>(
    A: SymbolicSparseColMatRef<'_, I>,
    params: LuSymbolicParams<'_>,
) -> Result<SymbolicLu<I>, FaerError> {
    assert!(A.nrows() == A.ncols());
    let m = A.nrows();
    let n = A.ncols();
    let A_nnz = A.compute_nnz();

    Size::with2(m, n, |M, N| {
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
                crate::colamd::order_req::<I>(m, n, A_nnz)?,
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
                        supernodal::factorize_supernodal_symbolic_lu_req::<I>(m, n)?,
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

        crate::colamd::order(
            &mut col_perm_fwd,
            &mut col_perm_inv,
            A.into_inner(),
            params.colamd_params,
            stack.rb_mut(),
        )?;

        let col_perm = ghost::PermutationRef::new(
            PermutationRef::new_checked(&col_perm_fwd, &col_perm_inv),
            N,
        );

        let (new_col_ptr, stack) = stack.make_raw::<I>(m + 1);
        let (new_row_ind, mut stack) = stack.make_raw::<I>(A_nnz);
        let AT = crate::ghost_adjoint_symbolic(new_col_ptr, new_row_ind, A, stack.rb_mut());

        let (etree, stack) = stack.make_raw::<I::Signed>(n);
        let (post, stack) = stack.make_raw::<I>(n);
        let (col_counts, stack) = stack.make_raw::<I>(n);
        let (h_col_counts, mut stack) = stack.make_raw::<I>(n);

        crate::qr::ghost_col_etree(A, Some(col_perm), Array::from_mut(etree, N), stack.rb_mut());
        let etree_ = Array::from_ref(MaybeIdx::<'_, I>::from_slice_ref_checked(&etree, N), N);
        crate::cholesky::ghost_postorder(Array::from_mut(post, N), etree_, stack.rb_mut());

        crate::qr::ghost_column_counts_aat(
            Array::from_mut(col_counts, N),
            Array::from_mut(bytemuck::cast_slice_mut(&mut min_row), M),
            AT,
            Some(col_perm),
            etree_,
            Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
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
                if parent < I::Signed::truncate(0) {
                    continue;
                }
                h_col_counts[parent.zx()] += h_col_counts[j] - I::truncate(1);
            }

            let mut nnz = 0.0f64;
            let mut flops = 0.0f64;
            for j in 0..n {
                let hj = h_col_counts[j].zx() as f64;
                let rj = col_counts[j].zx() as f64;
                flops += hj + hj * rj;
                nnz += hj + rj;
            }

            if flops / nnz > threshold.0 * crate::LU_SUPERNODAL_RATIO_FACTOR {
                threshold = SupernodalThreshold::FORCE_SUPERNODAL;
            } else {
                threshold = SupernodalThreshold::FORCE_SIMPLICIAL;
            }
        }

        if threshold == SupernodalThreshold::FORCE_SUPERNODAL {
            let symbolic = supernodal::factorize_supernodal_symbolic_lu::<I>(
                A.into_inner(),
                Some(col_perm.into_inner()),
                &*min_col,
                EliminationTreeRef::<'_, I> { inner: &etree },
                &col_counts,
                stack.rb_mut(),
                params.supernodal_params,
            )?;
            Ok(SymbolicLu {
                raw: SymbolicLuRaw::Supernodal(symbolic),
                col_perm_fwd,
                col_perm_inv,
                A_nnz,
            })
        } else {
            Ok(SymbolicLu {
                raw: SymbolicLuRaw::Simplicial { nrows: m, ncols: n },
                col_perm_fwd,
                col_perm_inv,
                A_nnz,
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        lu::{
            simplicial::{
                factorize_simplicial_numeric_lu, factorize_simplicial_numeric_lu_req, SimplicialLu,
            },
            supernodal::{
                factorize_supernodal_numeric_lu, factorize_supernodal_numeric_lu_req, SupernodalLu,
            },
            LuSymbolicParams, NumericLu,
        },
        qr::col_etree,
        SupernodalThreshold, SymbolicSparseColMatRef,
    };
    use core::iter::zip;
    use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
    use faer_core::{
        assert,
        group_helpers::SliceGroup,
        permutation::{Index, PermutationRef},
        sparse::SparseColMatRef,
        Conj, Mat, Parallelism,
    };
    use faer_entity::{ComplexField, Symbolic};
    use matrix_market_rs::MtxData;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use reborrow::*;

    use super::factorize_symbolic_lu;

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

        for &[i, j] in &coo_indices {
            col_counts[j] += I(1);
            if i != j {
                col_counts[i] += I(1);
            }
        }

        for i in 0..n {
            col_ptr[i + 1] = col_ptr[i] + col_counts[i];
        }
        let nnz = col_ptr[n].zx();

        let mut row_ind = vec![I(0); nnz];
        let mut values = vec![0.0; nnz];

        col_counts.copy_from_slice(&col_ptr[..n]);

        for (&[i, j], &val) in zip(&coo_indices, &coo_values) {
            if i == j {
                values[col_counts[j].zx()] = 2.0 * val;
            } else {
                values[col_counts[i].zx()] = val;
                values[col_counts[j].zx()] = val;
            }

            row_ind[col_counts[j].zx()] = I(i);
            col_counts[j] += I(1);

            if i != j {
                row_ind[col_counts[i].zx()] = I(j);
                col_counts[i] += I(1);
            }
        }

        (m, n, col_ptr, row_ind, values)
    }

    #[test]
    fn test_numeric_lu_multifrontal() {
        type E = faer_core::c64;

        let (m, n, col_ptr, row_ind, val) =
            load_mtx::<usize>(MtxData::from_file("test_data/YAO.mtx").unwrap());

        let mut rng = StdRng::seed_from_u64(0);
        let mut gen = || E::new(rng.gen::<f64>(), rng.gen::<f64>());

        let val = val.iter().map(|_| gen()).collect::<alloc::vec::Vec<_>>();
        let A = SparseColMatRef::<'_, usize, E>::new(
            SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_ind),
            &val,
        );
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        for i in 0..n {
            col_perm[i] = i;
            col_perm_inv[i] = i;
        }
        let col_perm = PermutationRef::<'_, usize, Symbolic>::new_checked(&col_perm, &col_perm_inv);

        let mut etree = vec![0usize; n];
        let mut min_col = vec![0usize; m];
        let mut col_counts = vec![0usize; n];

        let nnz = A.compute_nnz();
        let mut new_col_ptrs = vec![0usize; m + 1];
        let mut new_row_ind = vec![0usize; nnz];
        let mut new_values = vec![E::faer_zero(); nnz];
        let AT = crate::transpose::<usize, E>(
            &mut new_col_ptrs,
            &mut new_row_ind,
            &mut new_values,
            A,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(m))),
        )
        .into_const();

        let etree = {
            let mut post = vec![0usize; n];

            let etree = col_etree(
                *A,
                Some(col_perm),
                &mut etree,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(m + n))),
            );
            crate::qr::postorder(
                &mut post,
                etree,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(3 * n))),
            );
            crate::qr::column_counts_aat(
                &mut col_counts,
                &mut min_col,
                *AT,
                Some(col_perm),
                etree,
                &post,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(5 * n + m))),
            );
            etree
        };

        let symbolic = crate::lu::supernodal::factorize_supernodal_symbolic_lu::<usize>(
            *A,
            Some(col_perm),
            &min_col,
            etree,
            &col_counts,
            PodStack::new(&mut GlobalPodBuffer::new(
                super::supernodal::factorize_supernodal_symbolic_lu_req::<usize>(m, n).unwrap(),
            )),
            crate::SymbolicSupernodalParams {
                relax: Some(&[(4, 1.0), (16, 0.8), (48, 0.1), (usize::MAX, 0.05)]),
            },
        )
        .unwrap();

        let mut lu = SupernodalLu::<usize, E>::new();
        factorize_supernodal_numeric_lu(
            &mut row_perm,
            &mut row_perm_inv,
            &mut lu,
            A,
            AT,
            col_perm.cast(),
            &symbolic,
            Parallelism::None,
            PodStack::new(&mut GlobalPodBuffer::new(
                factorize_supernodal_numeric_lu_req::<usize, E>(&symbolic).unwrap(),
            )),
        )
        .unwrap();

        let k = 2;
        let rhs = Mat::from_fn(n, k, |_, _| gen());

        let mut work = rhs.clone();
        let A_dense = sparse_to_dense(A);
        let row_perm = PermutationRef::<'_, _, Symbolic>::new_checked(&row_perm, &row_perm_inv);

        {
            let mut x = rhs.clone();

            lu.solve_in_place_with_conj(
                row_perm.cast(),
                col_perm.cast(),
                Conj::No,
                x.as_mut(),
                Parallelism::None,
                work.as_mut(),
            );
            assert!((&A_dense * &x - &rhs).norm_max() < 1e-10);
        }
        {
            let mut x = rhs.clone();

            lu.solve_in_place_with_conj(
                row_perm.cast(),
                col_perm.cast(),
                Conj::Yes,
                x.as_mut(),
                Parallelism::None,
                work.as_mut(),
            );
            assert!((A_dense.conjugate() * &x - &rhs).norm_max() < 1e-10);
        }
        {
            let mut x = rhs.clone();

            lu.solve_transpose_in_place_with_conj(
                row_perm.cast(),
                col_perm.cast(),
                Conj::No,
                x.as_mut(),
                Parallelism::None,
                work.as_mut(),
            );
            assert!((A_dense.transpose() * &x - &rhs).norm_max() < 1e-10);
        }
        {
            let mut x = rhs.clone();

            lu.solve_transpose_in_place_with_conj(
                row_perm.cast(),
                col_perm.cast(),
                Conj::Yes,
                x.as_mut(),
                Parallelism::None,
                work.as_mut(),
            );
            assert!((A_dense.adjoint() * &x - &rhs).norm_max() < 1e-10);
        }
    }

    #[test]
    fn test_numeric_lu_simplicial() {
        type E = faer_core::c64;

        let (m, n, col_ptr, row_ind, val) =
            load_mtx::<usize>(MtxData::from_file("test_data/YAO.mtx").unwrap());

        let mut rng = StdRng::seed_from_u64(0);
        let mut gen = || E::new(rng.gen::<f64>(), rng.gen::<f64>());

        let val = val.iter().map(|_| gen()).collect::<alloc::vec::Vec<_>>();
        let A = SparseColMatRef::<'_, usize, E>::new(
            SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_ind),
            &val,
        );
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        for i in 0..n {
            col_perm[i] = i;
            col_perm_inv[i] = i;
        }
        let col_perm = PermutationRef::<'_, usize, Symbolic>::new_checked(&col_perm, &col_perm_inv);

        let mut lu = SimplicialLu::<usize, E>::new();
        factorize_simplicial_numeric_lu(
            &mut row_perm,
            &mut row_perm_inv,
            &mut lu,
            A,
            col_perm.cast(),
            PodStack::new(&mut GlobalPodBuffer::new(
                factorize_simplicial_numeric_lu_req::<usize, E>(m, n).unwrap(),
            )),
        )
        .unwrap();

        let k = 1;
        let rhs = Mat::from_fn(n, k, |_, _| gen());

        let mut work = rhs.clone();
        let A_dense = sparse_to_dense(A);
        let row_perm = PermutationRef::<'_, _, Symbolic>::new_checked(&row_perm, &row_perm_inv);

        {
            let mut x = rhs.clone();

            lu.solve_in_place_with_conj(
                row_perm.cast(),
                col_perm.cast(),
                Conj::No,
                x.as_mut(),
                Parallelism::None,
                work.as_mut(),
            );
            assert!((&A_dense * &x - &rhs).norm_max() < 1e-10);
        }
        {
            let mut x = rhs.clone();

            lu.solve_in_place_with_conj(
                row_perm.cast(),
                col_perm.cast(),
                Conj::Yes,
                x.as_mut(),
                Parallelism::None,
                work.as_mut(),
            );
            assert!((A_dense.conjugate() * &x - &rhs).norm_max() < 1e-10);
        }

        {
            let mut x = rhs.clone();

            lu.solve_transpose_in_place_with_conj(
                row_perm.cast(),
                col_perm.cast(),
                Conj::No,
                x.as_mut(),
                Parallelism::None,
                work.as_mut(),
            );
            assert!((A_dense.transpose() * &x - &rhs).norm_max() < 1e-10);
        }
        {
            let mut x = rhs.clone();

            lu.solve_transpose_in_place_with_conj(
                row_perm.cast(),
                col_perm.cast(),
                Conj::Yes,
                x.as_mut(),
                Parallelism::None,
                work.as_mut(),
            );
            assert!((A_dense.adjoint() * &x - &rhs).norm_max() < 1e-10);
        }
    }

    #[test]
    fn test_solver_lu_simplicial() {
        type E = faer_core::c64;

        let (m, n, col_ptr, row_ind, val) =
            load_mtx::<usize>(MtxData::from_file("test_data/YAO.mtx").unwrap());

        let mut rng = StdRng::seed_from_u64(0);
        let mut gen = || E::new(rng.gen::<f64>(), rng.gen::<f64>());

        let val = val.iter().map(|_| gen()).collect::<alloc::vec::Vec<_>>();
        let A = SparseColMatRef::<'_, usize, E>::new(
            SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_ind),
            &val,
        );

        let rhs = Mat::<E>::from_fn(m, 6, |_, _| gen());

        for supernodal_flop_ratio_threshold in [
            SupernodalThreshold::AUTO,
            SupernodalThreshold::FORCE_SUPERNODAL,
            SupernodalThreshold::FORCE_SIMPLICIAL,
        ] {
            let symbolic = factorize_symbolic_lu(
                A.symbolic(),
                LuSymbolicParams {
                    supernodal_flop_ratio_threshold,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut numeric = NumericLu::<usize, E>::new();
            let lu = symbolic
                .factorize_numeric_lu(
                    &mut numeric,
                    A,
                    Parallelism::None,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        symbolic
                            .factorize_numeric_lu_req::<E>(Parallelism::None)
                            .unwrap(),
                    )),
                )
                .unwrap();

            {
                let mut x = rhs.clone();
                lu.solve_in_place_with_conj(
                    faer_core::Conj::No,
                    x.as_mut(),
                    Parallelism::None,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        symbolic
                            .solve_in_place_req::<E>(rhs.ncols(), Parallelism::None)
                            .unwrap(),
                    )),
                );

                let linsolve_diff = A * &x - &rhs;
                assert!(linsolve_diff.norm_max() <= 1e-10);
            }
            {
                let mut x = rhs.clone();
                lu.solve_in_place_with_conj(
                    faer_core::Conj::Yes,
                    x.as_mut(),
                    Parallelism::None,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        symbolic
                            .solve_in_place_req::<E>(rhs.ncols(), Parallelism::None)
                            .unwrap(),
                    )),
                );

                let linsolve_diff = A.conjugate() * &x - &rhs;
                assert!(linsolve_diff.norm_max() <= 1e-10);
            }

            {
                let mut x = rhs.clone();
                lu.solve_transpose_in_place_with_conj(
                    faer_core::Conj::No,
                    x.as_mut(),
                    Parallelism::None,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        symbolic
                            .solve_transpose_in_place_req::<E>(rhs.ncols(), Parallelism::None)
                            .unwrap(),
                    )),
                );

                let linsolve_diff = A.transpose() * &x - &rhs;
                assert!(linsolve_diff.norm_max() <= 1e-10);
            }
            {
                let mut x = rhs.clone();
                lu.solve_transpose_in_place_with_conj(
                    faer_core::Conj::Yes,
                    x.as_mut(),
                    Parallelism::None,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        symbolic
                            .solve_transpose_in_place_req::<E>(rhs.ncols(), Parallelism::None)
                            .unwrap(),
                    )),
                );

                let linsolve_diff = A.adjoint() * &x - &rhs;
                assert!(linsolve_diff.norm_max() <= 1e-10);
            }
        }
    }
}
