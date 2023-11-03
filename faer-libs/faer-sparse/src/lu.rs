// Copyright (c) 2003, The Regents of the University of California, through
// Lawrence Berkeley National Laboratory (subject to receipt of any required
// approvals from U.S. Dept. of Energy)
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// (2) Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// (3) Neither the name of Lawrence Berkeley National Laboratory, U.S. Dept. of
// Energy nor the names of its contributors may be used to endorse or promote
// products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::{
    ghost::{Array, Idx, MaybeIdx},
    mem::{self},
    Index,
};
use reborrow::*;

#[inline]
fn relaxed_supernodes<'n, I: Index>(
    etree: &Array<'n, MaybeIdx<'n, I>>,
    postorder: &Array<'n, Idx<'n, I>>,
    postorder_inv: &Array<'n, Idx<'n, I>>,
    relax_columns: usize,
    descendants: &mut Array<'n, I>,
    relax_end: &mut Array<'n, I::Signed>,
) {
    let I = I::truncate;

    mem::fill_none(relax_end.as_mut());
    mem::fill_zero(descendants.as_mut());

    let N = etree.len();
    let etree = |i: Idx<'n, usize>| match etree[postorder[i].zx()].idx() {
        Some(parent) => MaybeIdx::from_index(postorder_inv[parent.zx()]),
        None => MaybeIdx::none(),
    };

    for j in N.indices() {
        if let Some(parent) = etree(j.zx()).idx() {
            let parent = parent.zx();
            descendants[parent] = descendants[parent] + descendants[j] + I(1);
        }
    }

    let mut j = 0;
    while j < *N {
        let mut parent = etree(N.check(j).zx()).sx();
        let snode_start = j;
        while let Some(parent_) = parent.idx() {
            if descendants[parent_] >= I(relax_columns) {
                break;
            }
            j = *parent_;
            parent = etree(N.check(j).zx()).sx();
        }
        relax_end[N.check(snode_start)] = I(j).to_signed();

        j += 1;

        while j < *N && descendants[N.check(j)] != I(0) {
            j += 1;
        }
    }
}

pub mod supernodal {
    use super::*;
    use crate::{cholesky::simplicial::EliminationTreeRef, mem::NONE, FaerError};
    #[cfg(feature = "std")]
    use assert2::assert;
    use core::iter::zip;
    use dyn_stack::{PodStack, SizeOverflow, StackReq};
    use faer_core::{
        constrained::Size,
        group_helpers::{SliceGroup, SliceGroupMut},
        mul,
        permutation::{PermutationRef, SignedIndex},
        solve,
        sparse::SparseColMatRef,
        temp_mat_req, temp_mat_uninit, Conj, MatMut, MatRef, Parallelism,
    };
    use faer_entity::*;

    pub struct SupernodalLuT2<I, E: Entity> {
        nrows: usize,
        ncols: usize,
        nsupernodes: usize,
        xsup: Vec<I>,
        supno: Vec<I>,
        lsub: Vec<I>,
        xlusup: Vec<I>,
        xlsub: Vec<I>,
        usub: Vec<I>,
        xusub: Vec<I>,

        lusup: GroupFor<E, Vec<E::Unit>>,
        ucol: GroupFor<E, Vec<E::Unit>>,
    }

    unsafe impl<I: Index, E: Entity> Send for SupernodalLuT2<I, E> {}
    unsafe impl<I: Index, E: Entity> Sync for SupernodalLuT2<I, E> {}

    impl<I: Index, E: Entity> Default for SupernodalLuT2<I, E> {
        #[inline]
        fn default() -> Self {
            Self::new()
        }
    }

    impl<I: Index, E: Entity> SupernodalLuT2<I, E> {
        #[inline]
        pub fn new() -> Self {
            Self {
                nrows: 0,
                ncols: 0,
                nsupernodes: 0,
                xsup: Vec::new(),
                supno: Vec::new(),
                lsub: Vec::new(),
                xlusup: Vec::new(),
                xlsub: Vec::new(),
                usub: Vec::new(),
                xusub: Vec::new(),
                lusup: E::faer_map(E::UNIT, |()| Vec::<E::Unit>::new()),
                ucol: E::faer_map(E::UNIT, |()| Vec::<E::Unit>::new()),
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
        pub fn solve_transpose_in_place_with_conj(
            &self,
            row_perm: PermutationRef<'_, I, E>,
            col_perm: PermutationRef<'_, I, E>,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) where
            E: ComplexField,
        {
            assert!(self.nrows() == self.ncols());
            assert!(self.nrows() == rhs.nrows());
            let mut X = rhs;
            let (mut temp, mut stack) = temp_mat_uninit::<E>(self.nrows(), X.ncols(), stack);
            faer_core::permutation::permute_rows(temp.rb_mut(), X.rb(), col_perm);
            self.u_solve_transpose_in_place_with_conj(
                conj_lhs,
                temp.rb_mut(),
                parallelism,
                stack.rb_mut(),
            );
            self.l_solve_transpose_in_place_with_conj(
                conj_lhs,
                temp.rb_mut(),
                parallelism,
                stack.rb_mut(),
            );
            faer_core::permutation::permute_rows(X.rb_mut(), temp.rb(), row_perm.inverse());
        }

        #[track_caller]
        pub fn solve_in_place_with_conj(
            &self,
            row_perm: PermutationRef<'_, I, E>,
            col_perm: PermutationRef<'_, I, E>,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) where
            E: ComplexField,
        {
            assert!(self.nrows() == self.ncols());
            assert!(self.nrows() == rhs.nrows());
            let mut X = rhs;
            let (mut temp, mut stack) = temp_mat_uninit::<E>(self.nrows(), X.ncols(), stack);
            faer_core::permutation::permute_rows(temp.rb_mut(), X.rb(), row_perm);
            self.l_solve_in_place_with_conj(conj_lhs, temp.rb_mut(), parallelism, stack.rb_mut());
            self.u_solve_in_place_with_conj(conj_lhs, temp.rb_mut(), parallelism, stack.rb_mut());
            faer_core::permutation::permute_rows(X.rb_mut(), temp.rb(), col_perm.inverse());
        }

        #[track_caller]
        pub fn l_solve_transpose_in_place_with_conj(
            &self,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) where
            E: ComplexField,
        {
            assert!(self.nrows() == self.ncols());
            assert!(self.nrows() == rhs.nrows());

            let (mut work, _) = faer_core::temp_mat_uninit::<E>(rhs.nrows(), rhs.ncols(), stack);

            let mut X = rhs;
            let nrhs = X.ncols();

            let nzval =
                SliceGroup::<'_, E>::new(E::faer_map(E::faer_as_ref(&self.lusup), |lusup| {
                    &**lusup
                }));
            let nzval_colptr = &*self.xlusup;
            let rowind = &*self.lsub;
            let rowind_colptr = &*self.xlsub;
            let sup_to_col = &*self.xsup;

            for k in (0..self.n_supernodes()).rev() {
                let fsupc = sup_to_col[k].zx();
                let istart = rowind_colptr[fsupc].zx();
                let nsupr = rowind_colptr[fsupc + 1].zx() - istart;
                let nsupc = sup_to_col[k + 1].zx() - fsupc;
                let nrow = nsupr - nsupc;

                let luptr = nzval_colptr[fsupc].zx();
                let lda = nzval_colptr[fsupc + 1].zx() - luptr;

                let mut work = work.rb_mut().subrows(0, nrow);
                let A = MatRef::<'_, E>::from_column_major_slice_with_stride(
                    nzval.subslice(luptr..nzval.len()).into_inner(),
                    nsupr,
                    nsupc,
                    lda,
                );

                let A_top = A.subrows(0, nsupc);
                let A_bot = A.subrows(nsupc, nrow);

                for j in 0..nrhs {
                    let mut iptr = istart + nsupc;
                    for i in 0..nrow {
                        let irow = rowind[iptr].zx();
                        work.write(i, j, X.read(irow, j));
                        iptr += 1;
                    }
                }

                mul::matmul_with_conj(
                    X.rb_mut().subrows(fsupc, nsupc),
                    A_bot.transpose(),
                    conj_lhs,
                    work.rb().subrows(0, nrow),
                    Conj::No,
                    Some(E::faer_one()),
                    E::faer_one().faer_neg(),
                    parallelism,
                );
                solve::solve_unit_upper_triangular_in_place_with_conj(
                    A_top.transpose(),
                    conj_lhs,
                    X.rb_mut().subrows(fsupc, nsupc),
                    parallelism,
                );
            }
        }

        #[track_caller]
        pub fn l_solve_in_place_with_conj(
            &self,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) where
            E: ComplexField,
        {
            assert!(self.nrows() == self.ncols());
            assert!(self.nrows() == rhs.nrows());

            let (mut work, _) = faer_core::temp_mat_uninit::<E>(rhs.nrows(), rhs.ncols(), stack);

            let mut X = rhs;
            let nrhs = X.ncols();

            let nzval =
                SliceGroup::<'_, E>::new(E::faer_map(E::faer_as_ref(&self.lusup), |lusup| {
                    &**lusup
                }));
            let nzval_colptr = &*self.xlusup;
            let rowind = &*self.lsub;
            let rowind_colptr = &*self.xlsub;
            let sup_to_col = &*self.xsup;

            for k in 0..self.n_supernodes() {
                let fsupc = sup_to_col[k].zx();
                let istart = rowind_colptr[fsupc].zx();
                let nsupr = rowind_colptr[fsupc + 1].zx() - istart;
                let nsupc = sup_to_col[k + 1].zx() - fsupc;
                let nrow = nsupr - nsupc;

                let luptr = nzval_colptr[fsupc].zx();
                let lda = nzval_colptr[fsupc + 1].zx() - luptr;

                let mut work = work.rb_mut().subrows(0, nrow);
                let A = MatRef::<'_, E>::from_column_major_slice_with_stride(
                    nzval.rb().subslice(luptr..nzval.len()).into_inner(),
                    nsupr,
                    nsupc,
                    lda,
                );

                let A_top = A.subrows(0, nsupc);
                let A_bot = A.subrows(nsupc, nrow);

                solve::solve_unit_lower_triangular_in_place_with_conj(
                    A_top,
                    conj_lhs,
                    X.rb_mut().subrows(fsupc, nsupc),
                    parallelism,
                );
                mul::matmul_with_conj(
                    work.rb_mut(),
                    A_bot,
                    conj_lhs,
                    X.rb().subrows(fsupc, nsupc),
                    Conj::No,
                    None,
                    E::faer_one(),
                    parallelism,
                );

                for j in 0..nrhs {
                    let mut iptr = istart + nsupc;
                    for i in 0..nrow {
                        let irow = rowind[iptr].zx();
                        X.write(irow, j, X.read(irow, j).faer_sub(work.read(i, j)));
                        iptr += 1;
                    }
                }
            }
        }

        #[track_caller]
        pub fn u_solve_transpose_in_place_with_conj(
            &self,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) where
            E: ComplexField,
        {
            assert!(self.ncols() == rhs.nrows());

            let _ = stack;

            let mut X = rhs;
            let nrhs = X.ncols();

            let nzval =
                SliceGroup::<'_, E>::new(E::faer_map(E::faer_as_ref(&self.lusup), |lusup| {
                    &**lusup
                }));
            let nzval_colptr = &*self.xlusup;
            let sup_to_col = &*self.xsup;

            let u_col_ptr = &*self.xusub;
            let u_row_ind = &*self.usub;
            let u_val =
                SliceGroup::<'_, E>::new(E::faer_map(E::faer_as_ref(&self.ucol), |ucol| &**ucol));

            for k in 0..self.n_supernodes() {
                let fsupc = sup_to_col[k].zx();
                let nsupc = sup_to_col[k + 1].zx() - fsupc;

                let luptr = nzval_colptr[fsupc].zx();
                let lda = nzval_colptr[fsupc + 1].zx() - luptr;

                let A = MatRef::<'_, E>::from_column_major_slice_with_stride(
                    nzval.rb().subslice(luptr..nzval.len()).into_inner(),
                    nsupc,
                    nsupc,
                    lda,
                );

                let A_top = A.subrows(0, nsupc);

                // PERF(sparse-dense gemm)
                for j in 0..nrhs {
                    for jcol in fsupc..fsupc + nsupc {
                        let start = u_col_ptr[jcol].zx();
                        let end = u_col_ptr[jcol + 1].zx();
                        let mut acc = E::faer_zero();
                        for (row, val) in u_row_ind[start..end]
                            .iter()
                            .zip(u_val.subslice(start..end).into_ref_iter())
                        {
                            let val = val.read();
                            let val = if conj_lhs == Conj::Yes {
                                val.faer_conj()
                            } else {
                                val
                            };
                            acc = acc.faer_add(X.read(row.zx(), j).faer_mul(val));
                        }
                        X.write(jcol, j, X.read(jcol, j).faer_sub(acc));
                    }
                }

                solve::solve_lower_triangular_in_place_with_conj(
                    A_top.transpose(),
                    conj_lhs,
                    X.rb_mut().subrows(fsupc, nsupc),
                    parallelism,
                );
            }
        }

        #[track_caller]
        pub fn u_solve_in_place_with_conj(
            &self,
            conj_lhs: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) where
            E: ComplexField,
        {
            assert!(self.ncols() == rhs.nrows());

            let _ = stack;

            let mut X = rhs;
            let nrhs = X.ncols();

            let nzval =
                SliceGroup::<'_, E>::new(E::faer_map(E::faer_as_ref(&self.lusup), |lusup| {
                    &**lusup
                }));
            let nzval_colptr = &*self.xlusup;
            let sup_to_col = &*self.xsup;

            let u_col_ptr = &*self.xusub;
            let u_row_ind = &*self.usub;
            let u_val =
                SliceGroup::<'_, E>::new(E::faer_map(E::faer_as_ref(&self.ucol), |ucol| &**ucol));

            for k in (0..self.n_supernodes()).rev() {
                let fsupc = sup_to_col[k].zx();
                let nsupc = sup_to_col[k + 1].zx() - fsupc;

                let luptr = nzval_colptr[fsupc].zx();
                let lda = nzval_colptr[fsupc + 1].zx() - luptr;

                let A = MatRef::<'_, E>::from_column_major_slice_with_stride(
                    nzval.rb().subslice(luptr..nzval.len()).into_inner(),
                    nsupc,
                    nsupc,
                    lda,
                );

                let A_top = A.subrows(0, nsupc);

                solve::solve_upper_triangular_in_place_with_conj(
                    A_top,
                    conj_lhs,
                    X.rb_mut().subrows(fsupc, nsupc),
                    parallelism,
                );

                // PERF(sparse-dense gemm)
                for j in 0..nrhs {
                    for jcol in fsupc..fsupc + nsupc {
                        let start = u_col_ptr[jcol].zx();
                        let end = u_col_ptr[jcol + 1].zx();
                        let x_jcol = X.read(jcol, j);
                        for (row, val) in u_row_ind[start..end]
                            .iter()
                            .zip(u_val.subslice(start..end).into_ref_iter())
                        {
                            let val = val.read();
                            let val = if conj_lhs == Conj::Yes {
                                val.faer_conj()
                            } else {
                                val
                            };
                            X.write(
                                row.zx(),
                                j,
                                X.read(row.zx(), j).faer_sub(x_jcol.faer_mul(val)),
                            )
                        }
                    }
                }
            }
        }
    }

    impl<I: Index, E: Entity> core::fmt::Debug for SupernodalLuT2<I, E> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("SupernodalLu")
                .field("xsup", &self.xsup)
                .field("supno", &self.supno)
                .field("lsub", &self.lsub)
                .field("xlusup", &self.xlusup)
                .field("xlsub", &self.xlsub)
                .field("usub", &self.usub)
                .field("xusub", &self.xusub)
                .field("lusup", &to_slice_group::<E>(&self.lusup))
                .field("ucol", &to_slice_group::<E>(&self.ucol))
                .finish()
        }
    }

    #[derive(Copy, Clone)]
    pub struct SupernodalLuParams {
        pub panel_size: usize,
        pub relax: usize,
        pub max_super: usize,
        pub row_block: usize,
        pub col_block: usize,
        pub fill_factor: usize,
    }

    impl Default for SupernodalLuParams {
        fn default() -> Self {
            Self {
                panel_size: 8,
                relax: 4,
                max_super: 128,
                row_block: 16,
                col_block: 8,
                fill_factor: 20,
            }
        }
    }

    #[inline(never)]
    fn resize_scalar<E: Entity>(
        v: &mut GroupFor<E, Vec<E::Unit>>,
        n: usize,
        exact: bool,
        reserve_only: bool,
    ) -> Result<(), FaerError> {
        let mut failed = false;
        let reserve = if exact {
            Vec::try_reserve_exact
        } else {
            Vec::try_reserve
        };

        E::faer_map(E::faer_as_mut(v), |v| {
            if !failed {
                failed = reserve(v, n.saturating_sub(v.len())).is_err();
                if !reserve_only {
                    v.resize(Ord::max(n, v.len()), unsafe { core::mem::zeroed() });
                }
            }
        });
        if failed {
            Err(FaerError::OutOfMemory)
        } else {
            Ok(())
        }
    }

    #[inline(never)]
    fn resize_work(
        v: &mut Vec<u8>,
        req: Result<StackReq, SizeOverflow>,
        exact: bool,
    ) -> Result<(), FaerError> {
        let reserve = if exact {
            Vec::try_reserve_exact
        } else {
            Vec::try_reserve
        };
        let n = req
            .and_then(|req| req.try_unaligned_bytes_required())
            .map_err(|_| FaerError::OutOfMemory)?;
        reserve(v, n.saturating_sub(v.len())).map_err(|_| FaerError::OutOfMemory)?;
        v.resize(n, 0);
        unsafe { v.set_len(n) };
        Ok(())
    }

    #[inline(never)]
    fn resize_index<I: Index>(
        v: &mut Vec<I>,
        n: usize,
        exact: bool,
        reserve_only: bool,
    ) -> Result<(), FaerError> {
        let reserve = if exact {
            Vec::try_reserve_exact
        } else {
            Vec::try_reserve
        };
        reserve(v, n.saturating_sub(v.len())).map_err(|_| FaerError::OutOfMemory)?;
        if !reserve_only {
            v.resize(Ord::max(n, v.len()), I::truncate(0));
        }
        Ok(())
    }

    impl<I: Index, E: Entity> SupernodalLuT2<I, E> {
        #[inline(never)]
        fn mem_init(
            &mut self,
            m: usize,
            n: usize,
            a_nnz: usize,
            fillratio: usize,
        ) -> Result<(), FaerError> {
            use FaerError::IndexOverflow;

            let nzumax = Ord::min(
                fillratio.checked_mul(a_nnz + 1).ok_or(IndexOverflow)?,
                m.checked_mul(n).ok_or(IndexOverflow)?,
            );
            let nzlumax = nzumax;
            let nzlmax = Ord::max(4, fillratio)
                .checked_mul(a_nnz + 1)
                .ok_or(IndexOverflow)?
                / 4;

            resize_index(&mut self.xsup, n + 1, true, false)?;
            resize_index(&mut self.supno, n + 1, true, false)?;
            resize_index(&mut self.xlsub, n + 1, true, false)?;
            resize_index(&mut self.xlusup, n + 1, true, false)?;
            resize_index(&mut self.xusub, n + 1, true, false)?;

            resize_scalar::<E>(&mut self.lusup, nzlumax, true, true)?;
            resize_scalar::<E>(&mut self.ucol, nzumax, true, true)?;
            resize_index(&mut self.lsub, nzlmax, true, true)?;
            resize_index(&mut self.usub, nzumax, true, true)?;

            Ok(())
        }
    }

    #[inline(never)]
    fn panel_dfs<I: Index, E: ComplexField>(
        m: usize,
        w: usize,
        jcol: usize,
        A: SparseColMatRef<'_, I, E>,
        col_perm: &[I],
        perm_r: &[I],
        nseg: &mut usize,
        dense: SliceGroupMut<'_, E>,
        panel_lsub: &mut [I::Signed],
        segrep: &mut [I],
        repfnz: &mut [I::Signed],
        xprune: &mut [I],
        marker: &mut [I::Signed],
        parent: &mut [I],
        xplore: &mut [I],
        lu: &mut SupernodalLuT2<I, E>,
    ) {
        let I = I::truncate;
        *nseg = 0;

        let mut dense = dense;

        for jj in jcol..jcol + w {
            let mut nextl_col = (jj - jcol) * m;
            let repfnz_col = &mut repfnz[nextl_col..][..m];
            let mut dense_col = dense.rb_mut().subslice(nextl_col..nextl_col + m);

            let pjj = col_perm[jj].zx();
            for (krow, value) in zip(
                A.row_indices_of_col(pjj),
                SliceGroup::<'_, E>::new(A.values_of_col(pjj)).into_ref_iter(),
            ) {
                dense_col.write(krow, value.read());
                let kmark = marker[krow];
                if kmark == I(jj).to_signed() {
                    continue;
                }

                dfs_kernel_for_panel_dfs(
                    m,
                    jcol,
                    jj,
                    perm_r,
                    nseg,
                    panel_lsub,
                    segrep,
                    repfnz_col,
                    xprune,
                    marker,
                    parent,
                    xplore,
                    lu,
                    &mut nextl_col,
                    krow,
                );
            }
        }
    }

    #[inline(never)]
    fn dfs_kernel_for_column_dfs<I: Index, E: ComplexField>(
        jsuper: &mut usize,
        jj: usize,
        perm_r: &[I],
        nseg: &mut usize,
        segrep: &mut [I],
        repfnz_col: &mut [I::Signed],
        xprune: &mut [I],
        marker: &mut [I::Signed],
        parent: &mut [I],
        xplore: &mut [I],
        lu: &mut SupernodalLuT2<I, E>,
        nextl_col: &mut usize,
        krow: usize,
    ) -> Result<(), FaerError> {
        let I = I::truncate;
        let panel_lsub = &mut lu.lsub;

        let kmark = marker[krow];

        marker[krow] = I(jj).to_signed();
        let kperm = perm_r[krow];

        if kperm == I(NONE) {
            resize_index(panel_lsub, *nextl_col + 1, false, false)?;
            panel_lsub[*nextl_col] = I(krow);
            *nextl_col += 1;

            if kmark + I(1).to_signed() != I(jj).to_signed() {
                *jsuper = NONE;
            }
        } else {
            let mut krep = lu.xsup[lu.supno[kperm.zx()].zx() + 1]
                .zx()
                .saturating_sub(1);
            let mut myfnz = repfnz_col[krep];

            if myfnz != I(NONE).to_signed() {
                if myfnz > kperm.to_signed() {
                    repfnz_col[krep] = kperm.to_signed();
                }
            } else {
                let mut oldrep = I(NONE);
                parent[krep] = oldrep;
                repfnz_col[krep] = kperm.to_signed();
                let mut xdfs = lu.xlsub[krep].zx();
                let mut maxdfs = xprune[krep].zx();

                loop {
                    while xdfs < maxdfs {
                        let kchild = panel_lsub[xdfs].zx();
                        xdfs += 1;
                        let chmark = marker[kchild];

                        if chmark != I(jj).to_signed() {
                            marker[kchild] = I(jj).to_signed();
                            let chperm = perm_r[kchild];
                            if chperm == I(NONE) {
                                resize_index(panel_lsub, *nextl_col + 1, false, false)?;
                                panel_lsub[*nextl_col] = I(kchild);
                                *nextl_col += 1;

                                if chmark + I(1).to_signed() != I(jj).to_signed() {
                                    *jsuper = NONE;
                                }
                            } else {
                                let chrep = lu.xsup[lu.supno[chperm.zx()].zx() + 1]
                                    .zx()
                                    .saturating_sub(1);
                                myfnz = repfnz_col[chrep];

                                if myfnz != I(NONE).to_signed() {
                                    if myfnz > chperm.to_signed() {
                                        repfnz_col[chrep] = chperm.to_signed();
                                    }
                                } else {
                                    xplore[krep] = I(xdfs);
                                    oldrep = I(krep);
                                    krep = chrep;
                                    parent[krep] = oldrep;
                                    repfnz_col[krep] = chperm.to_signed();
                                    xdfs = lu.xlsub[krep].zx();
                                    maxdfs = xprune[krep].zx();
                                }
                            }
                        }
                    }

                    segrep[*nseg] = I(krep);
                    *nseg += 1;

                    let kpar = parent[krep];
                    if kpar == I(NONE) {
                        break;
                    }

                    krep = kpar.zx();
                    xdfs = xplore[krep].zx();
                    maxdfs = xprune[krep].zx();
                }
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn dfs_kernel_for_panel_dfs<I: Index, E: ComplexField>(
        m: usize,
        jcol: usize,
        jj: usize,
        perm_r: &[I],
        nseg: &mut usize,
        panel_lsub: &mut [I::Signed],
        segrep: &mut [I],
        repfnz_col: &mut [I::Signed],
        xprune: &mut [I],
        marker: &mut [I::Signed],
        parent: &mut [I],
        xplore: &mut [I],
        lu: &mut SupernodalLuT2<I, E>,
        nextl_col: &mut usize,
        krow: usize,
    ) {
        let I = I::truncate;

        marker[krow] = I(jj).to_signed();
        let kperm = perm_r[krow];

        if kperm == I(NONE) {
            panel_lsub[*nextl_col] = I(krow).to_signed();
            *nextl_col += 1;
        } else {
            let mut krep = lu.xsup[lu.supno[kperm.zx()].zx() + 1].zx() - 1;
            let mut myfnz = repfnz_col[krep];

            if myfnz != I(NONE).to_signed() {
                if myfnz > kperm.to_signed() {
                    repfnz_col[krep] = kperm.to_signed();
                }
            } else {
                let mut oldrep = I(NONE);
                parent[krep] = oldrep;
                repfnz_col[krep] = kperm.to_signed();
                let mut xdfs = lu.xlsub[krep].zx();
                let mut maxdfs = xprune[krep].zx();

                loop {
                    while xdfs < maxdfs {
                        let kchild = lu.lsub[xdfs].zx();
                        xdfs += 1;
                        let chmark = marker[kchild];

                        if chmark != I(jj).to_signed() {
                            marker[kchild] = I(jj).to_signed();
                            let chperm = perm_r[kchild];
                            if chperm == I(NONE) {
                                panel_lsub[*nextl_col] = I(kchild).to_signed();
                                *nextl_col += 1;
                            } else {
                                let chrep = lu.xsup[lu.supno[chperm.zx()].zx() + 1]
                                    .zx()
                                    .saturating_sub(1);
                                myfnz = repfnz_col[chrep];

                                if myfnz != I(NONE).to_signed() {
                                    if myfnz > chperm.to_signed() {
                                        repfnz_col[chrep] = chperm.to_signed();
                                    }
                                } else {
                                    xplore[krep] = I(xdfs);
                                    oldrep = I(krep);
                                    krep = chrep;
                                    parent[krep] = oldrep;
                                    repfnz_col[krep] = chperm.to_signed();
                                    xdfs = lu.xlsub[krep].zx();
                                    maxdfs = xprune[krep].zx();
                                }
                            }
                        }
                    }

                    if marker[m + krep] < I(jcol).to_signed() {
                        marker[m + krep] = I(jj).to_signed();
                        segrep[*nseg] = I(krep);
                        *nseg += 1;
                    }

                    let kpar = parent[krep];
                    if kpar == I(NONE) {
                        break;
                    }

                    krep = kpar.zx();
                    xdfs = xplore[krep].zx();
                    maxdfs = xprune[krep].zx();
                }
            }
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub enum LuError {
        Generic(FaerError),
        ZeroColumn(usize),
    }

    impl From<FaerError> for LuError {
        #[inline]
        fn from(value: FaerError) -> Self {
            Self::Generic(value)
        }
    }

    #[inline(never)]
    fn prune_l<I: Index, E: ComplexField>(
        jcol: usize,
        row_perm: &[I],
        pivrow: usize,
        nseg: usize,
        segrep: &[I],
        repfnz: &mut [I::Signed],
        xprune: &mut [I],
        lu: &mut SupernodalLuT2<I, E>,
    ) {
        let I = I::truncate;

        let jsupno = lu.supno[jcol].zx();
        let mut kmin = 0;
        let mut kmax = 0;

        for i in 0..nseg {
            let irep = segrep[i].zx();
            let irep1 = irep + 1;
            let mut do_prune = false;
            if repfnz[irep] == I(NONE).to_signed() {
                continue;
            }
            if lu.supno[irep] == lu.supno[irep1] {
                continue;
            }

            if lu.supno[irep] != I(jsupno) {
                if xprune[irep] >= lu.xlsub[irep1] {
                    kmin = lu.xlsub[irep].zx();
                    kmax = lu.xlsub[irep1].zx() - 1;
                    for krow in kmin..kmax + 1 {
                        if lu.lsub[krow] == I(pivrow) {
                            do_prune = true;
                            break;
                        }
                    }
                }

                if do_prune {
                    let mut movnum = false;
                    if I(irep) == lu.xsup[lu.supno[irep].zx()] {
                        movnum = true;
                    }

                    while kmin <= kmax {
                        if row_perm[lu.lsub[kmax].zx()] == I(NONE) {
                            kmax -= 1;
                        } else if row_perm[lu.lsub[kmin].zx()] != I(NONE) {
                            kmin += 1;
                        } else {
                            lu.lsub.swap(kmin, kmax);
                            if movnum {
                                let minloc = lu.xlusup[irep].zx() + (kmin - lu.xlsub[irep].zx());
                                let maxloc = lu.xlusup[irep].zx() + (kmax - lu.xlsub[irep].zx());
                                E::faer_map(E::faer_as_mut(&mut lu.lusup), |lusup| {
                                    lusup.swap(minloc, maxloc)
                                });
                            }
                            kmin += 1;
                            kmax -= 1;
                        }
                    }
                    xprune[irep] = I(kmin);
                }
            }
        }
    }

    #[inline(never)]
    fn pivot_l<I: Index, E: ComplexField>(
        jcol: usize,
        diag_pivot_thresh: E::Real,
        row_perm: &mut [I],
        col_perm_inv: PermutationRef<'_, I, E>,
        pivrow: &mut usize,
        lu: &mut SupernodalLuT2<I, E>,
    ) -> bool {
        let I = I::truncate;

        let mut values =
            SliceGroupMut::<'_, E>::new(E::faer_map(E::faer_as_mut(&mut lu.lusup), |x| &mut **x));
        let indices = &mut *lu.lsub;

        let fsupc = lu.xsup[lu.supno[jcol].zx()].zx();
        let nsupc = jcol - fsupc;
        let lptr = lu.xlsub[fsupc].zx();
        let nsupr = lu.xlsub[fsupc + 1].zx() - lptr;
        let lda = (lu.xlusup[fsupc + 1] - lu.xlusup[fsupc]).zx();

        let lu_sup_ptr = lu.xlusup[fsupc].zx();
        let lu_col_ptr = lu.xlusup[jcol].zx();
        let lsub_ptr = lptr;

        let diagind = col_perm_inv.into_arrays().0[jcol].zx();
        let mut pivmax = E::Real::faer_one().faer_neg();
        let mut pivptr = nsupc;
        let mut diag = I(NONE);
        for isub in nsupc..nsupr {
            let rtemp = values.read(lu_col_ptr + isub).faer_abs();
            if rtemp > pivmax {
                pivmax = rtemp;
                pivptr = isub;
            }
            if indices[lsub_ptr + isub].zx() == diagind {
                diag = I(isub);
            }
        }

        if pivmax <= E::Real::faer_zero() {
            *pivrow = if pivmax < E::Real::faer_zero() {
                diagind
            } else {
                indices[lsub_ptr + pivptr].zx()
            };
            row_perm[*pivrow] = I(jcol);
            return true;
        }

        let thresh = diag_pivot_thresh.faer_mul(pivmax);
        if diag.to_signed() >= I(0).to_signed() {
            let rtemp = values.read(lu_col_ptr + diag.zx()).faer_abs();
            if rtemp != E::Real::faer_zero() && rtemp >= thresh {
                pivptr = diag.zx();
            }
        }
        *pivrow = indices[lsub_ptr + pivptr].zx();
        row_perm[*pivrow] = I(jcol);

        if pivptr != nsupc {
            indices.swap(lsub_ptr + pivptr, lsub_ptr + nsupc);
            for icol in 0..nsupc + 1 {
                let itemp = pivptr + icol * lda;
                let tmp = values.read(lu_sup_ptr + itemp);
                values.write(
                    lu_sup_ptr + itemp,
                    values.read(lu_sup_ptr + nsupc + icol * lda),
                );
                values.write(lu_sup_ptr + nsupc + icol * lda, tmp);
            }
        }

        let temp = values.read(lu_col_ptr + nsupc).faer_inv();
        for k in nsupc + 1..nsupr {
            values.write(lu_col_ptr + k, values.read(lu_col_ptr + k).faer_mul(temp));
        }

        return false;
    }

    #[inline(never)]
    fn copy_to_ucol<I: Index, E: ComplexField>(
        jcol: usize,
        nseg: usize,
        segrep: &mut [I],
        repfnz: &mut [<I as Index>::Signed],
        row_perm: &[I],
        mut dense: SliceGroupMut<'_, E>,
        lu: &mut SupernodalLuT2<I, E>,
    ) -> Result<(), FaerError> {
        let I = I::truncate;

        let jsupno = lu.supno[jcol].zx();
        let mut nextu = lu.xusub[jcol].zx();

        for k in (0..nseg).rev() {
            let krep = segrep[k].zx();
            let ksupno = lu.supno[krep].zx();

            if jsupno != ksupno {
                let kfnz = repfnz[krep];
                if kfnz != I(NONE).to_signed() {
                    let fsupc = lu.xsup[ksupno].zx();
                    let mut isub = lu.xlsub[fsupc].zx() + kfnz.zx() - fsupc;
                    let segsize = krep + 1 - kfnz.zx();
                    let new_next = nextu + segsize;

                    resize_scalar::<E>(&mut lu.ucol, new_next, false, false)?;
                    resize_index(&mut lu.usub, new_next, false, false)?;

                    let mut ucol = SliceGroupMut::<'_, E>::new(E::faer_map(
                        E::faer_as_mut(&mut lu.ucol),
                        |ucol| &mut **ucol,
                    ));

                    for _ in 0..segsize {
                        let irow = lu.lsub[isub].zx();

                        lu.usub[nextu] = row_perm[irow];
                        ucol.write(nextu, dense.read(irow));
                        dense.write(irow, E::faer_zero());

                        nextu += 1;
                        isub += 1;
                    }
                }
            }
        }
        lu.xusub[jcol + 1] = I(nextu);

        Ok(())
    }

    #[inline(never)]
    fn column_bmod<I: Index, E: ComplexField>(
        jcol: usize,
        nseg: usize,
        mut dense: SliceGroupMut<'_, E>,
        work: &mut Vec<u8>,
        segrep: &mut [I],
        repfnz: &mut [I::Signed],
        fpanelc: usize,
        lu: &mut SupernodalLuT2<I, E>,
    ) -> Result<(), FaerError> {
        let I = I::truncate;

        let jsupno = lu.supno[jcol].zx();

        for k in (0..nseg).rev() {
            let krep = segrep[k].zx();
            let ksupno = lu.supno[krep];
            if I(jsupno) != ksupno {
                let ksupno = ksupno.zx();
                let fsupc = lu.xsup[ksupno].zx();
                let fst_col = Ord::max(fsupc, fpanelc);
                let d_fsupc = fst_col - fsupc;
                let mut luptr = lu.xlusup[fst_col].zx() + d_fsupc;
                let lptr = lu.xlsub[fsupc].zx() + d_fsupc;
                let kfnz = repfnz[krep];
                let kfnz = Ord::max(kfnz, I(fpanelc).to_signed()).zx();
                let segsize = krep + 1 - kfnz;
                let nsupc = krep + 1 - fst_col;
                let nsupr = (lu.xlsub[fsupc + 1] - lu.xlsub[fsupc]).zx();
                let nrow = nsupr - d_fsupc - nsupc;
                let lda = (lu.xlusup[fst_col + 1] - lu.xlusup[fst_col]).zx();
                let no_zeros = kfnz - fst_col;

                lu_kernel_bmod(
                    segsize,
                    dense.rb_mut(),
                    work,
                    SliceGroup::<'_, E>::new(E::faer_map(E::faer_as_ref(&lu.lusup), |lusup| {
                        &**lusup
                    })),
                    &mut luptr,
                    lda,
                    nrow,
                    &lu.lsub,
                    lptr,
                    no_zeros,
                )?;
            }
        }

        let mut nextlu = lu.xlusup[jcol].zx();
        let fsupc = lu.xsup[jsupno].zx();

        let new_next = nextlu + lu.xlsub[fsupc + 1].zx() - lu.xlsub[fsupc].zx();
        resize_scalar::<E>(&mut lu.lusup, new_next, false, false)?;

        let mut lusup =
            SliceGroupMut::<'_, E>::new(E::faer_map(E::faer_as_mut(&mut lu.lusup), |x| &mut **x));

        for isub in lu.xlsub[fsupc].zx()..lu.xlsub[fsupc + 1].zx() {
            let irow = lu.lsub[isub].zx();
            lusup.write(nextlu, dense.read(irow));
            dense.write(irow, E::faer_zero());
            nextlu += 1;
        }
        lu.xlusup[jcol + 1] = I(nextlu);
        let fst_col = Ord::max(fsupc, fpanelc);

        if fst_col < jcol {
            let d_fsupc = fst_col - fsupc;
            let luptr = lu.xlusup[fst_col].zx() + d_fsupc;
            let nsupr = lu.xlsub[fsupc + 1].zx() - lu.xlsub[fsupc].zx();
            let nsupc = jcol - fst_col;
            let nrow = nsupr - d_fsupc - nsupc;

            let ufirst = lu.xlusup[jcol].zx() + d_fsupc;
            let lda = lu.xlusup[jcol + 1].zx() - lu.xlusup[jcol].zx();

            let (left, right) = lusup.rb_mut().split_at(ufirst);
            let (mid, right) = right.split_at(nsupc);

            let A = MatRef::<'_, E>::from_column_major_slice_with_stride(
                left.rb().subslice(luptr..left.len()).into_inner(),
                nsupr,
                nsupc,
                lda,
            );
            let [A_top, A_bot] = A.split_at_row(nsupc);
            let A_bot = A_bot.subrows(0, nrow);

            let mut l = MatMut::<'_, E>::from_column_major_slice(
                right.subslice(0..nrow).into_inner(),
                nrow,
                1,
            );
            let mut u = MatMut::<'_, E>::from_column_major_slice(
                mid.subslice(0..nsupc).into_inner(),
                nsupc,
                1,
            );

            solve::solve_unit_lower_triangular_in_place(A_top, u.rb_mut(), Parallelism::None);

            mul::matmul(
                l.rb_mut(),
                A_bot,
                u.rb(),
                Some(E::faer_one()),
                E::faer_one().faer_neg(),
                Parallelism::None,
            );
        }

        Ok(())
    }

    #[inline(never)]
    fn lu_kernel_bmod<I: Index, E: ComplexField>(
        segsize: usize,
        mut dense: SliceGroupMut<'_, E>,
        work: &mut Vec<u8>,
        lusup: SliceGroup<'_, E>,
        luptr: &mut usize,
        lda: usize,
        nrow: usize,
        lsub: &[I],
        lptr: usize,
        no_zeros: usize,
    ) -> Result<(), FaerError> {
        if segsize == 1 {
            let f = dense.read(lsub[lptr + no_zeros].zx());
            *luptr += lda * no_zeros + no_zeros + 1;
            let a = lusup.subslice(*luptr..*luptr + nrow);
            let irow = &lsub[lptr + no_zeros + 1..][..nrow];
            let mut i = 0;
            let chunk2 = irow.chunks_exact(2);
            let rem2 = chunk2.remainder();
            for i0i1 in chunk2 {
                let i0 = i0i1[0].zx();
                let i1 = i0i1[1].zx();

                unsafe {
                    let a0 = a.read_unchecked(i);
                    let a1 = a.read_unchecked(i + 1);

                    let d0 = dense.read_unchecked(i0);
                    let d1 = dense.read_unchecked(i1);
                    dense.write_unchecked(i0, d0.faer_sub(f.faer_mul(a0)));
                    dense.write_unchecked(i1, d1.faer_sub(f.faer_mul(a1)));
                }

                i += 2;
            }
            for i0 in rem2 {
                let i0 = i0.zx();
                let a0 = a.read(i);
                let d0 = dense.read(i0);
                dense.write(i0, d0.faer_sub(f.faer_mul(a0)));
            }
        } else {
            resize_work(work, temp_mat_req::<E>(segsize + nrow, 1), false)?;
            let stack = PodStack::new(work);
            let (_, mut storage) = E::faer_map_with_context(stack, E::UNIT, &mut |stack, ()| {
                let (storage, stack) =
                    stack.make_aligned_raw::<E::Unit>(segsize + nrow, faer_core::CACHELINE_ALIGN);
                (stack, storage)
            });
            let mut tempv = unsafe {
                MatMut::<'_, E>::from_raw_parts(
                    E::faer_map(E::faer_as_mut(&mut storage), |storage| {
                        storage.as_mut_ptr() as *mut E::Unit
                    }),
                    segsize + nrow,
                    1,
                    1,
                    1,
                )
            };

            let mut isub = lptr + no_zeros;
            for i in 0..segsize {
                let irow = lsub[isub].zx();
                tempv.write(i, 0, dense.read(irow));
                isub += 1;
            }

            assert!(lda > 0);
            assert!(*luptr + (segsize + nrow - 1) + (segsize - 1) * lda < lusup.len());
            *luptr += lda * no_zeros + no_zeros;

            let A = MatRef::<'_, E>::from_column_major_slice_with_stride(
                lusup.subslice(*luptr..lusup.len()).into_inner(),
                segsize + nrow,
                segsize,
                lda,
            );
            *luptr += segsize;

            let [A, B] = A.split_at_row(segsize);
            let B = B.subrows(0, nrow);

            let [mut u, mut l] = tempv.rb_mut().split_at_row(segsize);

            solve::solve_unit_lower_triangular_in_place(A, u.rb_mut(), Parallelism::None);
            mul::matmul(
                l.rb_mut(),
                B,
                u.rb(),
                None,
                E::faer_one(),
                Parallelism::None,
            );
            let mut isub = lptr + no_zeros;
            for i in 0..segsize {
                let irow = lsub[isub].zx();
                isub += 1;
                dense.write(irow, u.read(i, 0));
            }
            for i in 0..nrow {
                let irow = lsub[isub].zx();
                isub += 1;
                dense.write(irow, dense.read(irow).faer_sub(l.read(i, 0)));
            }

            for i in 0..segsize {
                tempv.write(i, 0, E::faer_zero());
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn column_dfs<I: Index, E: ComplexField>(
        m: usize,
        jcol: usize,
        row_perm: &[I],
        maxsuper: usize,
        nseg: &mut usize,
        lsub_col: &mut [I::Signed],
        segrep: &mut [I],
        repfnz: &mut [I::Signed],
        xprune: &mut [I],
        marker: &mut [I::Signed],
        parent: &mut [I],
        xplore: &mut [I],
        lu: &mut SupernodalLuT2<I, E>,
    ) -> Result<(), FaerError> {
        let I = I::truncate;

        let mut jsuper = lu.supno[jcol].zx();
        let mut nextl = lu.xlsub[jcol].zx();
        let marker2 = &mut marker[2 * m..][..m];

        let mut k = 0;
        while if k < m {
            lsub_col[k] != I(NONE).to_signed()
        } else {
            false
        } {
            let krow = lsub_col[k].zx();
            lsub_col[k] = I(NONE).to_signed();
            let kmark = marker2[krow];

            if kmark == I(jcol).to_signed() {
                k += 1;
                continue;
            }

            dfs_kernel_for_column_dfs(
                &mut jsuper,
                jcol,
                row_perm,
                nseg,
                segrep,
                repfnz,
                xprune,
                marker2,
                parent,
                xplore,
                lu,
                &mut nextl,
                krow,
            )?;

            k += 1;
        }

        let jcolp1 = jcol + 1;

        let mut nsuper = lu.supno[jcol].zx();
        if jcol == 0 {
            nsuper = 0;
            lu.supno[0] = I(0);
        } else {
            let jcolm1 = jcol - 1;
            let fsupc = lu.xsup[nsuper].zx();
            let jptr = lu.xlsub[jcol].zx();
            let jm1ptr = lu.xlsub[jcolm1].zx();

            if nextl - jptr != jptr - jm1ptr - 1 {
                jsuper = NONE;
            }
            if jcol - fsupc >= maxsuper {
                jsuper = NONE;
            }

            if jsuper == NONE {
                if fsupc + 1 < jcolm1 {
                    let mut ito = lu.xlsub[fsupc + 1];
                    lu.xlsub[jcolm1] = ito;
                    let istop = ito + I(jptr) - I(jm1ptr);
                    xprune[jcolm1] = istop;
                    lu.xlsub[jcol] = istop;

                    for ifrom in jm1ptr..nextl {
                        lu.lsub[ito.zx()] = lu.lsub[ifrom];
                        ito += I(1);
                    }
                    nextl = ito.zx();
                }
                nsuper += 1;
                lu.supno[jcol] = I(nsuper);
            }
        }

        lu.xsup[nsuper + 1] = I(jcolp1);
        lu.supno[jcolp1] = I(nsuper);
        xprune[jcol] = I(nextl);
        lu.xlsub[jcolp1] = I(nextl);

        Ok(())
    }

    #[inline(never)]
    fn panel_bmod<I: Index, E: ComplexField>(
        m: usize,
        w: usize,
        jcol: usize,
        nseg: usize,
        mut dense: SliceGroupMut<'_, E>,
        work: &mut Vec<u8>,
        segrep: &mut [I],
        repfnz: &mut [I::Signed],
        lu: &mut SupernodalLuT2<I, E>,
        parallelism: Parallelism,
    ) -> Result<(), FaerError> {
        let I = I::truncate;
        for k in (0..nseg).rev() {
            let krep = segrep[k].zx();
            let fsupc = lu.xsup[lu.supno[krep].zx()].zx();
            let nsupc = krep + 1 - fsupc;
            let nsupr = (lu.xlsub[fsupc + 1] - lu.xlsub[fsupc]).zx();
            let nrow = nsupr - nsupc;
            let lptr = lu.xlsub[fsupc].zx();

            let mut u_rows = 0usize;
            let mut u_cols = 0usize;

            for jj in jcol..jcol + w {
                let nextl_col = (jj - jcol) * m;
                let repfnz_col = &mut repfnz[nextl_col..][..m];
                let kfnz = repfnz_col[krep];
                if kfnz == I(NONE).to_signed() {
                    continue;
                }

                let segsize = krep + 1 - kfnz.zx();
                u_cols += 1;
                u_rows = Ord::max(u_rows, segsize);
            }

            if nsupc >= 2 {
                resize_work(work, temp_mat_req::<E>(u_rows + nrow, u_cols), false)?;
                let stack = PodStack::new(work);
                let tmp_lda = faer_core::col_stride::<E::Unit>(u_rows + nrow);
                let (_, mut storage) =
                    E::faer_map_with_context(stack, E::UNIT, &mut |stack, ()| {
                        let (storage, stack) = stack.make_aligned_raw::<E::Unit>(
                            tmp_lda * u_cols,
                            faer_core::CACHELINE_ALIGN,
                        );
                        (stack, storage)
                    });
                let mut tempv = unsafe {
                    MatMut::<'_, E>::from_raw_parts(
                        E::faer_map(E::faer_as_mut(&mut storage), |storage| {
                            storage.as_mut_ptr() as *mut E::Unit
                        }),
                        u_rows + nrow,
                        u_cols,
                        1,
                        tmp_lda as isize,
                    )
                };

                let [mut U, mut L] = tempv.rb_mut().split_at_row(u_rows);

                let mut u_col = 0usize;
                for jj in jcol..jcol + w {
                    let nextl_col = (jj - jcol) * m;
                    let repfnz_col = &mut repfnz[nextl_col..][..m];
                    let dense_col = dense.rb_mut().subslice(nextl_col..nextl_col + m);

                    let kfnz = repfnz_col[krep];
                    if kfnz == I(NONE).to_signed() {
                        continue;
                    }

                    let segsize = krep + 1 - kfnz.zx();
                    let no_zeros = kfnz.zx() - fsupc;
                    let isub = lptr + no_zeros;
                    let off = u_rows - segsize;
                    assert!(off <= U.nrows());
                    assert!(u_col < U.ncols());
                    for i in 0..off {
                        U.write(i, u_col, E::faer_zero());
                    }
                    let mut U = U.rb_mut().get(off.., ..);
                    assert!(segsize <= U.nrows());
                    assert!(u_col < U.ncols());
                    for (i, irow) in lu.lsub[isub..][..segsize].iter().enumerate() {
                        U.write(i, u_col, dense_col.read(irow.zx()));
                    }
                    u_col += 1;
                }
                let mut luptr = lu.xlusup[fsupc].zx();
                let lda = (lu.xlusup[fsupc + 1] - lu.xlusup[fsupc]).zx();
                let no_zeros = (krep + 1 - u_rows) - fsupc;
                luptr += lda * no_zeros + no_zeros;
                let l_val = to_slice_group::<E>(&lu.lusup);
                let A = MatRef::<'_, E>::from_column_major_slice_with_stride(
                    l_val.subslice(luptr..l_val.len()).into_inner(),
                    u_rows + nrow,
                    u_rows,
                    lda,
                );
                let [A, B] = A.split_at_row(u_rows);
                let B = B.subrows(0, nrow);

                solve::solve_unit_lower_triangular_in_place(A, U.rb_mut(), parallelism);

                mul::matmul(L.rb_mut(), B, U.rb(), None, E::faer_one(), parallelism);
                let mut u_col = 0usize;
                for jj in jcol..jcol + w {
                    let nextl_col = (jj - jcol) * m;
                    let repfnz_col = &mut repfnz[nextl_col..][..m];
                    let mut dense_col = dense.rb_mut().subslice(nextl_col..nextl_col + m);

                    let kfnz = repfnz_col[krep];
                    if kfnz == I(NONE).to_signed() {
                        continue;
                    }

                    let segsize = krep + 1 - kfnz.zx();
                    let no_zeros = kfnz.zx() - fsupc;
                    let mut isub = lptr + no_zeros;
                    let off = u_rows - segsize;

                    let mut U = U.rb_mut().get(off.., ..);
                    assert!(segsize <= U.nrows());
                    assert!(u_col < U.ncols());
                    for (i, irow) in lu.lsub[isub..][..segsize].iter().enumerate() {
                        let irow = irow.zx();
                        unsafe {
                            dense_col.write_unchecked(irow, U.read_unchecked(i, u_col));
                            U.write_unchecked(i, u_col, E::faer_zero());
                        }
                    }
                    isub += segsize;
                    assert!(nrow <= L.nrows());
                    assert!(u_col < L.ncols());
                    for (i, irow) in lu.lsub[isub..][..nrow].iter().enumerate() {
                        let irow = irow.zx();
                        unsafe {
                            dense_col.write_unchecked(
                                irow,
                                dense_col
                                    .read_unchecked(irow)
                                    .faer_sub(L.read_unchecked(i, u_col)),
                            );
                            L.write_unchecked(i, u_col, E::faer_zero());
                        }
                    }
                    u_col += 1;
                }
            } else {
                for jj in jcol..jcol + w {
                    let nextl_col = (jj - jcol) * m;
                    let repfnz_col = &mut repfnz[nextl_col..][..m];
                    let dense_col = dense.rb_mut().subslice(nextl_col..nextl_col + m);

                    let kfnz = repfnz_col[krep];
                    if kfnz == I(NONE).to_signed() {
                        continue;
                    }
                    let kfnz = kfnz.zx();
                    let segsize = krep + 1 - kfnz;
                    let mut luptr = lu.xlusup[fsupc].zx();
                    let lda = lu.xlusup[fsupc + 1].zx() - lu.xlusup[fsupc].zx();
                    let no_zeros = kfnz - fsupc;

                    lu_kernel_bmod(
                        segsize,
                        dense_col,
                        work,
                        SliceGroup::<'_, E>::new(E::faer_map(E::faer_as_ref(&lu.lusup), |lusup| {
                            &**lusup
                        })),
                        &mut luptr,
                        lda,
                        nrow,
                        &mut lu.lsub,
                        lptr,
                        no_zeros,
                    )?;
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn to_slice_group_mut<E: Entity>(v: &mut GroupFor<E, Vec<E::Unit>>) -> SliceGroupMut<'_, E> {
        SliceGroupMut::<'_, E>::new(E::faer_map(E::faer_as_mut(v), |v| &mut **v))
    }
    #[inline]
    fn to_slice_group<E: Entity>(v: &GroupFor<E, Vec<E::Unit>>) -> SliceGroup<'_, E> {
        SliceGroup::<'_, E>::new(E::faer_map(E::faer_as_ref(v), |v| &**v))
    }

    #[inline(never)]
    fn snode_dfs<I: Index, E: ComplexField>(
        jcol: usize,
        kcol: usize,
        A: SparseColMatRef<'_, I, E>,
        row_perm: &[I],
        col_perm: &[I],
        xprune: &mut [I],
        marker: &mut [I::Signed],
        lu: &mut SupernodalLuT2<I, E>,
    ) -> Result<(), FaerError> {
        let I = I::truncate;
        let SI = I::Signed::truncate;

        // TODO: handle non leaf nodes properly
        let _ = row_perm;

        let nsuper = lu.supno[jcol].zx().wrapping_add(1);
        lu.supno[jcol] = I(nsuper);
        let mut nextl = lu.xlsub[jcol].zx();

        for i in jcol..kcol {
            let i_p = col_perm[i].zx();
            for krow in A.row_indices_of_col(i_p) {
                let kmark = marker[krow].zx();
                if kmark != kcol - 1 {
                    marker[krow] = SI(kcol - 1);
                    resize_index::<I>(&mut lu.lsub, nextl + 1, false, false)?;
                    lu.lsub[nextl] = I(krow);
                    nextl += 1;
                }
            }
            lu.supno[i] = I(nsuper);
        }

        if jcol + 1 < kcol {
            let new_next = nextl + (nextl - lu.xlsub[jcol].zx());
            resize_index::<I>(&mut lu.lsub, new_next, false, false)?;

            lu.lsub.copy_within(lu.xlsub[jcol].zx()..nextl, nextl);
            for i in jcol + 1..kcol {
                lu.xlsub[i] = I(nextl);
            }
            nextl = new_next;
        }

        lu.xsup[nsuper + 1] = I(kcol);
        lu.supno[kcol] = I(nsuper);
        xprune[kcol - 1] = I(nextl);
        lu.xlsub[kcol] = I(nextl);

        Ok(())
    }

    #[inline(never)]
    fn snode_bmod_setup<I: Index, E: ComplexField>(
        jcol: usize,
        fsupc: usize,
        mut dense: SliceGroupMut<'_, E>,
        lu: &mut SupernodalLuT2<I, E>,
    ) {
        let I = I::truncate;

        let mut nextlu = lu.xlusup[jcol].zx();
        let mut lusup = to_slice_group_mut::<E>(&mut lu.lusup);
        for isub in lu.xlsub[fsupc].zx()..lu.xlsub[fsupc + 1].zx() {
            let irow = lu.lsub[isub].zx();
            lusup.write(nextlu, dense.read(irow));
            dense.write(irow, E::faer_zero());
            nextlu += 1;
        }

        lu.xlusup[jcol + 1] = I(nextlu);
    }

    #[allow(dead_code)]
    #[inline(never)]
    fn snode_bmod<I: Index, E: ComplexField>(
        jcol: usize,
        kcol: usize,
        _jsupno: usize,
        fsupc: usize,
        lu: &mut SupernodalLuT2<I, E>,
        parallelism: Parallelism,
    ) {
        let ufirst = lu.xlusup[jcol].zx();
        let left_lda = lu.xlusup[fsupc + 1].zx() - lu.xlusup[fsupc].zx();
        let right_lda = lu.xlusup[jcol + 1].zx() - lu.xlusup[jcol].zx();
        let nsupr = (lu.xlsub[fsupc + 1] - lu.xlsub[fsupc]).zx();
        let nsupc = jcol - fsupc;
        let luptr = lu.xlusup[fsupc].zx();

        let (left, right) = to_slice_group_mut::<E>(&mut lu.lusup).split_at(ufirst);

        let [A, B] = MatRef::<'_, E>::from_column_major_slice_with_stride(
            left.rb()
                .subslice(luptr..luptr + left_lda * nsupc)
                .into_inner(),
            nsupr,
            nsupc,
            left_lda,
        )
        .subrows(0, nsupr)
        .split_at_row(nsupc);

        let [mut top, mut bot] = MatMut::<'_, E>::from_column_major_slice_with_stride(
            right.subslice(0..right_lda * (kcol - jcol)).into_inner(),
            nsupr,
            kcol - jcol,
            right_lda,
        )
        .subrows(0, nsupr)
        .split_at_row(nsupc);

        solve::solve_unit_lower_triangular_in_place(A, top.rb_mut(), parallelism);
        mul::matmul(
            bot.rb_mut(),
            B,
            top.rb(),
            Some(E::faer_one()),
            E::faer_one().faer_neg(),
            parallelism,
        );
    }

    #[inline(never)]
    fn snode_lu<I: Index, E: ComplexField>(
        kcol: usize,
        _jsupno: usize,
        fsupc: usize,
        transpositions: &mut [I],
        lu: &mut SupernodalLuT2<I, E>,
        parallelism: Parallelism,
    ) {
        let lda = lu.xlusup[fsupc + 1].zx() - lu.xlusup[fsupc].zx();
        let nsupr = (lu.xlsub[fsupc + 1] - lu.xlsub[fsupc]).zx();
        let nsupc = kcol - fsupc;
        let luptr = lu.xlusup[fsupc].zx();

        let lu_val = to_slice_group_mut::<E>(&mut lu.lusup);
        let lu_val_len = lu_val.len();
        let mut LU = MatMut::<'_, E>::from_column_major_slice_with_stride(
            lu_val.subslice(luptr..lu_val_len).into_inner(),
            nsupr,
            nsupc,
            lda,
        );
        faer_lu::partial_pivoting::compute::lu_in_place_impl(
            LU.rb_mut(),
            0,
            nsupc,
            transpositions,
            parallelism,
        );
    }

    #[track_caller]
    pub fn factorize_supernodal_numeric_dynamic_lu<I: Index, E: ComplexField>(
        row_perm: &mut [I],
        row_perm_inv: &mut [I],
        col_perm: &mut [I],
        col_perm_inv: &mut [I],
        lu: &mut SupernodalLuT2<I, E>,
        work: &mut Vec<u8>,

        A: SparseColMatRef<'_, I, E>,
        fill_reducing_col_perm: PermutationRef<'_, I, E>,
        etree: EliminationTreeRef<'_, I>,

        parallelism: Parallelism,
        stack: PodStack<'_>,
        params: SupernodalLuParams,
    ) -> Result<(), LuError> {
        let I = I::truncate;

        let m = A.nrows();
        let n = A.ncols();

        assert!(row_perm.len() == m);
        assert!(row_perm_inv.len() == m);
        assert!(fill_reducing_col_perm.len() == n);
        assert!(etree.into_inner().len() == n);

        let (row_perm, row_perm_inv) = (row_perm_inv, row_perm);

        let a_nnz = A.compute_nnz();
        lu.nrows = 0;
        lu.ncols = 0;
        lu.nsupernodes = 0;

        let maxpanel = params.panel_size.checked_mul(m).unwrap();

        let (descendants, stack) = stack.make_raw::<I>(n);
        let (relax_end, mut stack) = stack.make_raw::<I>(n);

        {
            let (post, stack) = stack.rb_mut().make_raw::<I>(n);
            let (post_inv, mut stack) = stack.make_raw::<I>(n);

            Size::with(n, |N| {
                crate::qr::postorder::<I>(post, etree, stack.rb_mut());
                for i in 0..n {
                    post_inv[post[i].zx()] = I(i);
                }
                relaxed_supernodes(
                    etree.ghost_inner(N),
                    Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
                    Array::from_ref(Idx::from_slice_ref_checked(&post_inv, N), N),
                    params.relax,
                    Array::from_mut(descendants, N),
                    Array::from_mut(bytemuck::cast_slice_mut(relax_end), N),
                );
            });

            for i in 0..n {
                col_perm[i] = fill_reducing_col_perm.into_arrays().0[post[i].zx()];
            }
        }
        for i in 0..n {
            col_perm_inv[col_perm[i].zx()] = I(i);
        }
        let col_perm = PermutationRef::new_checked(&col_perm, &col_perm_inv);

        let (repfnz, stack) = stack.make_raw::<I::Signed>(maxpanel);
        let (panel_lsub, stack) = stack.make_raw::<I::Signed>(maxpanel);
        let (marker, stack) = stack.make_raw::<I::Signed>(m.checked_mul(3).unwrap());
        let (segrep, stack) = stack.make_raw::<I>(m);
        let (parent, stack) = stack.make_raw::<I>(m);
        let (xplore, stack) = stack.make_raw::<I>(m);
        let (xprune, stack) = stack.make_raw::<I>(n);
        let (transpositions, stack) = stack.make_raw::<I>(n);

        let (mut dense, _) = crate::make_raw::<E>(maxpanel, stack);

        lu.mem_init(m, n, a_nnz, params.fill_factor)?;

        mem::fill_none::<I::Signed>(bytemuck::cast_slice_mut(row_perm));

        mem::fill_zero(segrep);
        mem::fill_zero(parent);
        mem::fill_zero(xplore);
        mem::fill_zero(xprune);

        mem::fill_none(marker);
        mem::fill_none(repfnz);
        mem::fill_none(panel_lsub);

        dense.fill_zero();

        lu.supno[0] = I(NONE);
        lu.xlsub[0] = I(0);
        lu.xusub[0] = I(0);
        lu.xlusup[0] = I(0);
        mem::fill_zero(&mut lu.xsup);
        mem::fill_none::<I::Signed>(bytemuck::cast_slice_mut(&mut lu.supno));

        let mut jcol = 0;

        let diag_pivot_thresh = E::Real::faer_one();
        while jcol < n {
            if relax_end[jcol] != I(NONE) && relax_end[jcol].zx() - jcol + 1 > 16 {
                let kcol = relax_end[jcol].zx() + 1;
                snode_dfs(
                    jcol,
                    kcol,
                    A,
                    row_perm,
                    col_perm.into_arrays().0,
                    xprune,
                    marker,
                    lu,
                )?;

                let nextu = lu.xusub[jcol].zx();
                let nextlu = lu.xlusup[jcol].zx();
                let jsupno = lu.supno[jcol].zx();
                let fsupc = lu.xsup[jsupno].zx();
                let new_next =
                    nextlu + (lu.xlsub[fsupc + 1] - lu.xlsub[fsupc]).zx() * (kcol - jcol);
                resize_scalar::<E>(&mut lu.lusup, new_next, false, false)?;

                for icol in jcol..kcol {
                    lu.xusub[icol + 1] = I(nextu);

                    let icol_p = col_perm.into_arrays().0[icol].zx();
                    for (i, val) in zip(
                        A.row_indices_of_col(icol_p),
                        SliceGroup::<'_, E>::new(A.values_of_col(icol_p)).into_ref_iter(),
                    ) {
                        dense.write(i, val.read());
                    }

                    // TODO: handle non leaf nodes properly
                    snode_bmod_setup(icol, fsupc, dense.rb_mut(), lu);
                }
                let transpositions = &mut transpositions[jcol..kcol];
                snode_lu(kcol, jsupno, fsupc, transpositions, lu, parallelism);
                for (idx, t) in transpositions.iter().enumerate() {
                    let t = t.zx();
                    let j = jcol + idx;
                    let lptr = lu.xlsub[fsupc].zx();
                    let pivrow = lu.lsub[lptr + idx + t].zx();
                    row_perm[pivrow] = I(j);
                    lu.lsub.swap(lptr + idx, lptr + idx + t);
                }

                jcol = kcol;
            } else {
                let mut panel_size = params.panel_size;

                let mut nseg1 = 0;
                let mut nseg;

                {
                    let mut k = jcol + 1;
                    while k < Ord::min(jcol.saturating_add(panel_size), n) {
                        if relax_end[k] != I(NONE) {
                            panel_size = k - jcol;
                            break;
                        }
                        k += 1;
                    }
                    if k == n {
                        panel_size = n - jcol;
                    }
                }

                panel_dfs(
                    m,
                    panel_size,
                    jcol,
                    A,
                    col_perm.into_arrays().0,
                    row_perm,
                    &mut nseg1,
                    dense.rb_mut(),
                    panel_lsub,
                    segrep,
                    repfnz,
                    xprune,
                    marker,
                    parent,
                    xplore,
                    lu,
                );
                panel_bmod(
                    m,
                    panel_size,
                    jcol,
                    nseg1,
                    dense.rb_mut(),
                    work,
                    segrep,
                    repfnz,
                    lu,
                    parallelism,
                )?;

                for jj in jcol..jcol + panel_size {
                    let k = (jj - jcol) * m;
                    nseg = nseg1;

                    let panel_lsubk = &mut panel_lsub[k..][..m];
                    let repfnz_k = &mut repfnz[k..][..m];
                    column_dfs(
                        m,
                        jj,
                        row_perm,
                        params.max_super,
                        &mut nseg,
                        panel_lsubk,
                        segrep,
                        repfnz_k,
                        xprune,
                        marker,
                        parent,
                        xplore,
                        lu,
                    )?;

                    let mut dense_k = dense.rb_mut().subslice(k..k + m);
                    let segrep_k = &mut segrep[nseg1..m];

                    column_bmod(
                        jj,
                        nseg - nseg1,
                        dense_k.rb_mut(),
                        work,
                        segrep_k,
                        repfnz_k,
                        jcol,
                        lu,
                    )?;
                    copy_to_ucol(jj, nseg, segrep, repfnz_k, row_perm, dense_k, lu)?;

                    let mut pivrow = 0usize;
                    if pivot_l(
                        jj,
                        diag_pivot_thresh,
                        row_perm,
                        col_perm.inverse(),
                        &mut pivrow,
                        lu,
                    ) {
                        return Err(LuError::ZeroColumn(jj));
                    }

                    prune_l(jj, row_perm, pivrow, nseg, segrep, repfnz_k, xprune, lu);

                    for &irep in &segrep[..nseg] {
                        repfnz_k[irep.zx()] = I(NONE).to_signed();
                    }
                }
                jcol += panel_size;
            }
        }

        for i in 0..m {
            row_perm_inv[row_perm[i].zx()] = I(i);
        }

        let mut nextl = 0usize;
        let nsuper = lu.supno[n].zx();

        for i in 0..nsuper + 1 {
            let fsupc = lu.xsup[i].zx();
            let jstart = lu.xlsub[fsupc].zx();
            lu.xlsub[fsupc] = I(nextl);
            for j in jstart..lu.xlsub[fsupc + 1].zx() {
                lu.lsub[nextl] = row_perm[lu.lsub[j].zx()];
                nextl += 1;
            }
            for k in fsupc + 1..lu.xsup[i + 1].zx() {
                lu.xlsub[k] = I(nextl);
            }
        }
        lu.xlsub[n] = I(nextl);
        lu.nrows = m;
        lu.ncols = n;
        lu.nsupernodes = nsuper + 1;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{supernodal::SupernodalLuT2, *};
    use crate::{
        cholesky::simplicial::EliminationTreeRef,
        ghost,
        lu::supernodal::factorize_supernodal_numeric_dynamic_lu,
        mem::NONE,
        qr::{col_etree, ghost_col_etree},
        SymbolicSparseColMatRef,
    };
    use assert2::assert;
    use core::iter::zip;
    use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
    use faer_core::{
        group_helpers::SliceGroup, permutation::PermutationRef, sparse::SparseColMatRef, Conj, Mat,
    };
    use faer_entity::ComplexField;
    use matrix_market_rs::MtxData;
    use rand::{Rng, SeedableRng};

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

    fn test_supernodes<I: Index>() {
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
        let mut descendants = vec![zero; n];
        let mut relax_end = vec![zero.to_signed(); n];
        ghost::with_size(n, |N| {
            let A = ghost::SymbolicSparseColMatRef::new(A, N, N);
            ghost_col_etree(
                A,
                None,
                Array::from_mut(&mut etree, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(2 * *N))),
            );
            let mut post = vec![I(0); n];
            let mut post_inv = vec![I(0); n];
            crate::qr::postorder::<I>(
                &mut post,
                EliminationTreeRef { inner: &etree },
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(3 * *N))),
            );
            for i in 0..n {
                post_inv[post[i].zx()] = I(i);
            }
            relaxed_supernodes(
                Array::from_ref(MaybeIdx::from_slice_ref_checked(&etree, N), N),
                Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
                Array::from_ref(Idx::from_slice_ref_checked(&post_inv, N), N),
                1,
                Array::from_mut(&mut descendants, N),
                Array::from_mut(&mut relax_end, N),
            );
        });
        assert!(
            etree
                == [3, 2, 3, 4, 5, 6, 7, 8, 9, 10, NONE]
                    .map(I)
                    .map(I::to_signed)
        );
        assert!(
            relax_end
                == [0, 1, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE]
                    .map(I)
                    .map(I::to_signed)
        );
    }

    #[test]
    fn test_numeric_lu_tiny() {
        let n = 8;
        let col_ptr = &[0, 4, 8, 12, 16, 21, 23, 27, 29];
        let row_ind = &[
            1, 4, 6, 7, 0, 5, 6, 7, 0, 1, 3, 6, 0, 1, 4, 5, 0, 2, 5, 6, 7, 3, 4, 0, 4, 6, 7, 2, 4,
        ];
        let val = &[
            0.783099, 0.335223, 0.55397, 0.628871, 0.513401, 0.606969, 0.242887, 0.804177,
            0.400944, 0.108809, 0.512932, 0.637552, 0.972775, 0.771358, 0.891529, 0.352458,
            0.949327, 0.192214, 0.0641713, 0.457702, 0.23828, 0.53976, 0.760249, 0.437638,
            0.738534, 0.687861, 0.440105, 0.228968, 0.68667,
        ];
        let A = SparseColMatRef::<'_, u32, f64>::new(
            SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_ind),
            val,
        );

        let mut etree = vec![0u32; n];
        let etree = col_etree(
            *A,
            None,
            &mut etree,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<u32>(2 * n))),
        );

        let mut row_perm = vec![0u32; n];
        let mut row_perm_inv = vec![0u32; n];
        let mut col_perm = vec![0u32; n];
        let mut col_perm_inv = vec![0u32; n];
        let mut fill_col_perm = vec![0u32; n];
        let mut fill_col_perm_inv = vec![0u32; n];
        for i in 0..n {
            fill_col_perm[i] = i as _;
            fill_col_perm_inv[i] = i as _;
        }
        let fill_col_perm = PermutationRef::new_checked(&fill_col_perm, &fill_col_perm_inv);

        let mut lu = SupernodalLuT2::<u32, f64>::new();

        factorize_supernodal_numeric_dynamic_lu(
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            &mut lu,
            &mut vec![],
            A,
            fill_col_perm,
            etree,
            faer_core::Parallelism::None,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<u32>(1024 * 1024))),
            Default::default(),
        )
        .unwrap();
        {
            let row_perm = PermutationRef::<'_, _, f64>::new_checked(&row_perm, &row_perm_inv);
            let col_perm = PermutationRef::<'_, _, f64>::new_checked(&col_perm, &col_perm_inv);
            let mut gen = rand::rngs::StdRng::seed_from_u64(0);
            let A_dense = sparse_to_dense(A);
            let k = 2;
            let rhs = Mat::from_fn(n, k, |_, _| gen.gen::<f64>());
            let mut x = rhs.clone();

            lu.solve_in_place_with_conj(
                row_perm,
                col_perm,
                Conj::No,
                x.as_mut(),
                faer_core::Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(
                    1024 * 1024,
                ))),
            );
            dbgf::dbgf!("?", &A_dense * &x - &rhs);
        }
    }

    #[test]
    fn test_numeric_lu_small() {
        let n = 16;
        let col_ptr = &[
            0, 2, 10, 15, 19, 20, 27, 34, 39, 44, 45, 53, 57, 61, 68, 71, 75,
        ];
        let row_ind = &[
            5, 7, 0, 1, 2, 3, 4, 6, 9, 13, 2, 5, 7, 11, 13, 1, 7, 11, 15, 3, 1, 3, 7, 10, 11, 13,
            14, 7, 9, 11, 12, 13, 14, 15, 6, 8, 9, 13, 15, 0, 1, 2, 5, 14, 6, 1, 3, 4, 5, 8, 10,
            13, 15, 6, 9, 14, 15, 4, 9, 12, 13, 0, 5, 6, 7, 8, 10, 15, 5, 7, 12, 2, 5, 9, 11,
        ];
        let val = &[
            0.335223, 0.55397, 0.606969, 0.242887, 0.804177, 0.400944, 0.108809, 0.512932,
            0.637552, 0.771358, 0.352458, 0.949327, 0.192214, 0.020023, 0.23828, 0.53976, 0.437638,
            0.738534, 0.440105, 0.893372, 0.950252, 0.881062, 0.786002, 0.187533, 0.556444,
            0.906804, 0.126075, 0.232262, 0.15239, 0.79347, 0.745071, 0.950104, 0.521563, 0.240062,
            0.134902, 0.0699064, 0.46142, 0.157807, 0.889956, 0.997799, 0.87054, 0.00416161,
            0.163131, 0.530808, 0.747803, 0.830012, 0.649707, 0.62948, 0.70062, 0.074161, 0.651132,
            0.546107, 0.471483, 0.344943, 0.675476, 0.621823, 0.413984, 0.609106, 0.920914,
            0.532441, 0.260497, 0.111276, 0.775767, 0.329642, 0.984363, 0.827391, 0.436497,
            0.685786, 0.793657, 0.904932, 0.273911, 0.180421, 0.603109, 0.221966, 0.138238,
        ];
        let A = SparseColMatRef::<'_, usize, f64>::new(
            SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_ind),
            val,
        );

        let mut etree = vec![0usize; n];
        let etree = col_etree(
            *A,
            None,
            &mut etree,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(2 * n))),
        );

        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        let fill_col_perm = vec![6, 2, 8, 7, 11, 4, 12, 14, 5, 3, 0, 9, 13, 15, 10, 1];
        let mut fill_col_perm_inv = vec![0usize; n];
        for i in 0..n {
            fill_col_perm_inv[fill_col_perm[i]] = i;
        }
        let fill_col_perm = PermutationRef::new_checked(&fill_col_perm, &fill_col_perm_inv);

        let mut lu = SupernodalLuT2::<usize, f64>::new();

        factorize_supernodal_numeric_dynamic_lu(
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            &mut lu,
            &mut vec![],
            A,
            fill_col_perm,
            etree,
            faer_core::Parallelism::None,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(
                1024 * 1024,
            ))),
            Default::default(),
        )
        .unwrap();
        let row_perm = PermutationRef::<'_, _, f64>::new_checked(&row_perm, &row_perm_inv);
        let mut gen = rand::rngs::StdRng::seed_from_u64(0);
        let A_dense = sparse_to_dense(A);
        let k = 1;
        let rhs = Mat::from_fn(n, k, |_, _| gen.gen::<f64>());
        dbg!(row_perm);
        dbgf::dbgf!("?", &rhs);

        {
            let mut x = rhs.clone();
            let col_perm = PermutationRef::<'_, _, f64>::new_checked(&col_perm, &col_perm_inv);
            lu.solve_in_place_with_conj(
                row_perm,
                col_perm,
                Conj::No,
                x.as_mut(),
                faer_core::Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(
                    1024 * 1024,
                ))),
            );
            dbgf::dbgf!("?", &A_dense * &x - &rhs);
        }

        {
            let mut x = rhs.clone();
            let col_perm = PermutationRef::<'_, _, f64>::new_checked(&col_perm, &col_perm_inv);
            lu.solve_transpose_in_place_with_conj(
                row_perm,
                col_perm,
                Conj::No,
                x.as_mut(),
                faer_core::Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(
                    1024 * 1024,
                ))),
            );
            dbgf::dbgf!("?", A_dense.transpose() * &x - &rhs);
        }
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
    fn test_numeric_lu_yao() {
        let (m, n, col_ptr, row_ind, val) =
            load_mtx::<usize>(MtxData::from_file("bench_data/rijc781.mtx").unwrap());

        let A = SparseColMatRef::<'_, usize, f64>::new(
            SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_ind),
            &val,
        );

        let mut etree = vec![0usize; n];
        let etree = col_etree(
            *A,
            None,
            &mut etree,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(2 * n))),
        );

        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        let mut fill_col_perm = vec![0usize; n];
        let mut fill_col_perm_inv = vec![0usize; n];
        for i in 0..n {
            fill_col_perm[i] = i;
            fill_col_perm_inv[i] = i;
        }
        let fill_col_perm = PermutationRef::new_checked(&fill_col_perm, &fill_col_perm_inv);

        let mut lu = SupernodalLuT2::<usize, f64>::new();

        factorize_supernodal_numeric_dynamic_lu(
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            &mut lu,
            &mut vec![],
            A,
            fill_col_perm,
            etree,
            faer_core::Parallelism::None,
            PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(
                1024 * 1024,
            ))),
            Default::default(),
        )
        .unwrap();

        {
            let row_perm = PermutationRef::<'_, _, f64>::new_checked(&row_perm, &row_perm_inv);
            let col_perm = PermutationRef::<'_, _, f64>::new_checked(&col_perm, &col_perm_inv);
            let mut gen = rand::rngs::StdRng::seed_from_u64(0);
            let A_dense = sparse_to_dense(A);
            let k = 2;
            let rhs = Mat::from_fn(n, k, |_, _| gen.gen::<f64>());
            let mut x = rhs.clone();

            lu.solve_in_place_with_conj(
                row_perm,
                col_perm,
                Conj::No,
                x.as_mut(),
                faer_core::Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<usize>(
                    1024 * 1024,
                ))),
            );
            dbgf::dbgf!("?", &A_dense * &x - &rhs);
        }
    }

    monomorphize_test!(test_supernodes);
}
