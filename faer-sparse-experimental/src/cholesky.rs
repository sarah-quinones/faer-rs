// implementation inspired by https://gitlab.com/hodge_star/catamari

use crate::{
    amd::{self, Control},
    ghost::{self, Array, Idx, MaybeIdx},
    ghost_permute_hermitian, ghost_permute_hermitian_symbolic, make_raw_req, mem,
    mem::NONE,
    nomem, try_collect, try_zeroed, windows2, FaerError, Index, PermutationRef, Side, SliceGroup,
    SliceGroupMut, SparseColMatRef, SymbolicSparseColMatRef,
};
use assert2::assert;
use core::{cell::Cell, iter::zip};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    temp_mat_req, temp_mat_uninit, zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism,
};
use reborrow::*;

pub use faer_cholesky::ldlt_diagonal::compute::LdltRegularization;

#[derive(Copy, Clone)]
#[allow(dead_code)]
enum Ordering<'a, I> {
    Identity,
    Custom(&'a [I]),
    Algorithm(
        &'a dyn Fn(
            &mut [I],                       // perm
            &mut [I],                       // perm_inv
            SymbolicSparseColMatRef<'_, I>, // A
            PodStack<'_>,
        ) -> Result<(), FaerError>,
    ),
}

pub mod simplicial {
    use super::*;
    use assert2::assert;

    fn ereach<'n, 'a, I: Index>(
        stack: &'a mut Array<'n, I>,
        A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
        etree: &Array<'n, MaybeIdx<'n, I>>,
        k: Idx<'n>,
        visited: &mut Array<'n, I>,
    ) -> &'a [Idx<'n, I>] {
        let N = A.ncols();

        // invariant: stack[top..] elements are less than or equal to k
        let mut top = *N;
        let k_: I = *k.truncate();
        visited[k] = k_;
        for mut i in A.row_indices_of_col(k) {
            // (1): after this, we know i < k
            if i >= k {
                continue;
            }
            // invariant: stack[..len] elements are less than or equal to k
            let mut len = 0usize;
            loop {
                if visited[i] == k_ {
                    break;
                }

                // inserted element is i < k, see (1)
                let pushed: Idx<'n, I> = i.truncate::<I>();
                stack[N.check(len)] = *pushed;
                // len is incremented, maintaining the invariant
                len += 1;

                visited[i] = k_;
                i = N.check(etree[i].zx());
            }

            // because stack[..len] elements are less than or equal to k
            // stack[top - len..] elements are now less than or equal to k
            stack.copy_within(..len, top - len);
            // top is decremented by len, maintaining the invariant
            top -= len;
        }

        let stack = &(**stack)[top..];

        // SAFETY: stack[top..] elements are < k < N
        unsafe { Idx::slice_ref_unchecked(stack, N) }
    }

    pub fn factorize_simplicial_symbolic_req<I: Index>(n: usize) -> Result<StackReq, SizeOverflow> {
        let n_req = StackReq::try_new::<I>(n)?;
        StackReq::try_all_of([n_req, n_req, n_req])
    }

    pub fn factorize_simplicial_symbolic<I: Index>(
        A: SymbolicSparseColMatRef<'_, I>,
        etree: EtreeRef<'_, I>,
        col_counts: &[I],
        stack: PodStack<'_>,
    ) -> Result<SymbolicSimplicialCholesky<I>, FaerError> {
        let n = A.nrows();
        assert!(A.nrows() == A.ncols());
        assert!(etree.inner.len() == n);
        assert!(col_counts.len() == n);

        ghost::with_size(n, |N| {
            ghost_factorize_simplicial_symbolic(
                ghost::SymbolicSparseColMatRef::new(A, N, N),
                etree.ghost_inner(N),
                Array::from_ref(col_counts, N),
                stack,
            )
        })
    }

    pub(crate) fn ghost_factorize_simplicial_symbolic<'n, I: Index>(
        A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
        etree: &Array<'n, MaybeIdx<'n, I>>,
        col_counts: &Array<'n, I>,
        stack: PodStack<'_>,
    ) -> Result<SymbolicSimplicialCholesky<I>, FaerError> {
        let N = A.ncols();
        let n = *N;

        let mut L_col_ptrs = try_zeroed::<I>(n + 1)?;
        for (&count, [p, p_next]) in zip(
            &**col_counts,
            windows2(Cell::as_slice_of_cells(Cell::from_mut(&mut L_col_ptrs))),
        ) {
            p_next.set(p.get() + count);
        }
        let l_nnz = L_col_ptrs[n].zx();
        let mut L_row_ind = try_zeroed::<I>(l_nnz)?;

        ghost::with_size(
            l_nnz,
            #[inline(always)]
            move |L_NNZ| {
                let (current_row_index, stack) = stack.make_raw::<I>(n);
                let (ereach_stack, stack) = stack.make_raw::<I>(n);
                let (marked, _) = stack.make_raw::<I>(n);

                let ereach_stack = Array::from_mut(ereach_stack, N);
                let etree = Array::from_ref(etree, N);
                let visited = Array::from_mut(marked, N);

                mem::fill_none(visited);
                let L_row_indices = Array::from_mut(&mut L_row_ind, L_NNZ);
                let L_col_ptrs_start =
                    Array::from_ref(Idx::slice_ref_checked(&L_col_ptrs[..n], L_NNZ), N);
                let current_row_index =
                    Array::from_mut(ghost::copy_slice(current_row_index, L_col_ptrs_start), N);

                for k in N.indices() {
                    let reach = ereach(ereach_stack, A, etree, k, visited);
                    for &j in reach {
                        let j = j.zx();
                        let cj = &mut current_row_index[j];
                        let row_idx = L_NNZ.check(*cj.zx() + 1);
                        *cj = row_idx.truncate();
                        L_row_indices[row_idx] = *k.truncate();
                    }
                    let k_start = L_col_ptrs_start[k].zx();
                    L_row_indices[k_start] = *k.truncate();
                }

                let etree = try_collect(MaybeIdx::as_inner_slice_ref(etree).iter().copied())?;

                let _ = SymbolicSparseColMatRef::new_checked(n, n, &L_col_ptrs, None, &L_row_ind);

                Ok(SymbolicSimplicialCholesky {
                    dimension: n,
                    col_ptrs: L_col_ptrs,
                    row_indices: L_row_ind,
                    etree,
                })
            },
        )
    }

    pub fn factorize_simplicial_numeric_ldlt<I: Index, E: ComplexField>(
        L_values: SliceGroupMut<'_, E>,
        A: SparseColMatRef<'_, I, E>,
        regularization: LdltRegularization<'_, E>,
        symbolic: &SymbolicSimplicialCholesky<I>,
        stack: PodStack<'_>,
    ) -> usize {
        let n = A.ncols();
        let L_row_indices = &*symbolic.row_indices;
        let L_col_ptrs = &*symbolic.col_ptrs;
        let etree = &*symbolic.etree;

        assert!(L_values.rb().len() == L_row_indices.len());
        assert!(L_col_ptrs.len() == n + 1);
        let l_nnz = L_col_ptrs[n].zx();

        ghost::with_size(
            n,
            #[inline(always)]
            |N| {
                ghost::with_size(
                    l_nnz,
                    #[inline(always)]
                    move |L_NNZ| {
                        let etree = Array::from_ref(MaybeIdx::slice_ref_checked(etree, N), N);
                        let A = ghost::SparseColMatRef::new(A, N, N);

                        let eps = regularization.dynamic_regularization_epsilon.faer_abs();
                        let delta = regularization.dynamic_regularization_delta.faer_abs();
                        let has_eps = delta > E::Real::faer_zero();
                        let mut dynamic_regularization_count = 0usize;

                        let (x, stack) = crate::make_raw::<E>(n, stack);
                        let (current_row_index, stack) = stack.make_raw::<I>(n);
                        let (ereach_stack, stack) = stack.make_raw::<I>(n);
                        let (marked, _) = stack.make_raw::<I>(n);

                        let ereach_stack = Array::from_mut(ereach_stack, N);
                        let etree = Array::from_ref(etree, N);
                        let visited = Array::from_mut(marked, N);
                        let mut x = ghost::ArrayGroupMut::new(x, N);

                        x.rb_mut().into_slice().fill_zero();
                        mem::fill_none(visited);

                        let mut L_values = ghost::ArrayGroupMut::new(L_values, L_NNZ);
                        let L_row_indices = Array::from_ref(L_row_indices, L_NNZ);

                        let L_col_ptrs_start =
                            Array::from_ref(Idx::slice_ref_checked(&L_col_ptrs[..n], L_NNZ), N);

                        let current_row_index = Array::from_mut(
                            ghost::copy_slice(current_row_index, L_col_ptrs_start),
                            N,
                        );

                        for k in N.indices() {
                            let reach = ereach(ereach_stack, A.symbolic(), etree, k, visited);

                            for (i, aik) in
                                zip(A.row_indices_of_col(k), A.values_of_col(k).into_ref_iter())
                            {
                                x.write(i, aik.read().faer_conj());
                            }

                            let mut d = x.read(k).faer_real();
                            x.write(k, E::faer_zero());

                            for &j in reach {
                                let j = j.zx();

                                let j_start = L_col_ptrs_start[j].zx();
                                let cj = &mut current_row_index[j];
                                let row_idx = L_NNZ.check(*cj.zx() + 1);
                                *cj = row_idx.truncate();

                                let xj = x.read(j);
                                x.write(j, E::faer_zero());

                                let dj = L_values.read(j_start).faer_real();
                                let lkj = xj.faer_scale_real(dj.faer_inv());

                                let range = j_start.next()..row_idx.to_inclusive();
                                for (i, lij) in zip(
                                    &L_row_indices[range.clone()],
                                    L_values.rb().subslice(range).into_ref_iter(),
                                ) {
                                    let i = N.check(i.zx());
                                    let mut xi = x.read(i);
                                    let prod = lij.read().faer_conj().faer_mul(xj);
                                    xi = xi.faer_sub(prod);
                                    x.write(i, xi);
                                }

                                d = d.faer_sub(lkj.faer_mul(xj.faer_conj()).faer_real());

                                L_values.write(row_idx, lkj);
                            }

                            let k_start = L_col_ptrs_start[k].zx();

                            if has_eps {
                                if let Some(signs) = regularization.dynamic_regularization_signs {
                                    if signs[*k] > 0 && d <= delta {
                                        d = eps;
                                        dynamic_regularization_count += 1;
                                    } else if signs[*k] < 0 && d >= delta.faer_neg() {
                                        d = eps.faer_neg();
                                        dynamic_regularization_count += 1;
                                    }
                                } else if d.faer_abs() <= delta {
                                    if d < E::Real::faer_zero() {
                                        d = eps.faer_neg();
                                        dynamic_regularization_count += 1;
                                    } else {
                                        d = eps;
                                        dynamic_regularization_count += 1;
                                    }
                                }
                            }
                            L_values.write(k_start, E::faer_from_real(d));
                        }
                        dynamic_regularization_count
                    },
                )
            },
        )
    }

    pub fn factorize_simplicial_numeric_with_row_indices<I: Index, E: ComplexField>(
        L_values: SliceGroupMut<'_, E>,
        L_row_indices: &mut [I],
        L_col_ptrs: &[I],

        etree: EtreeRef<'_, I>,
        A: SparseColMatRef<'_, I, E>,

        stack: PodStack<'_>,
    ) {
        let n = A.ncols();
        assert!(L_values.rb().len() == L_row_indices.len());
        assert!(L_col_ptrs.len() == n + 1);
        assert!(etree.inner().len() == n);
        let l_nnz = L_col_ptrs[n].zx();

        ghost::with_size(
            n,
            #[inline(always)]
            |N| {
                let etree = etree.ghost_inner(N);
                let A = ghost::SparseColMatRef::new(A, N, N);

                ghost::with_size(
                    l_nnz,
                    #[inline(always)]
                    move |L_NNZ| {
                        let (x, stack) = crate::make_raw::<E>(n, stack);
                        let (current_row_index, stack) = stack.make_raw::<I>(n);
                        let (ereach_stack, stack) = stack.make_raw::<I>(n);
                        let (marked, _) = stack.make_raw::<I>(n);

                        let ereach_stack = Array::from_mut(ereach_stack, N);
                        let etree = Array::from_ref(etree, N);
                        let visited = Array::from_mut(marked, N);
                        let mut x = ghost::ArrayGroupMut::new(x, N);

                        x.rb_mut().into_slice().fill_zero();
                        mem::fill_none(visited);

                        let mut L_values = ghost::ArrayGroupMut::new(L_values, L_NNZ);
                        let L_row_indices = Array::from_mut(L_row_indices, L_NNZ);

                        let L_col_ptrs_start =
                            Array::from_ref(Idx::slice_ref_checked(&L_col_ptrs[..n], L_NNZ), N);

                        let current_row_index = Array::from_mut(
                            ghost::copy_slice(current_row_index, L_col_ptrs_start),
                            N,
                        );

                        for k in N.indices() {
                            let reach = ereach(ereach_stack, A.symbolic(), etree, k, visited);

                            for (i, aik) in
                                zip(A.row_indices_of_col(k), A.values_of_col(k).into_ref_iter())
                            {
                                x.write(i, aik.read().faer_conj());
                            }

                            let mut d = x.read(k).faer_real();
                            x.write(k, E::faer_zero());

                            for &j in reach {
                                let j = j.zx();

                                let j_start = L_col_ptrs_start[j].zx();
                                let cj = &mut current_row_index[j];
                                let row_idx = L_NNZ.check(*cj.zx() + 1);
                                *cj = row_idx.truncate();

                                let xj = x.read(j);
                                x.write(j, E::faer_zero());

                                let dj = L_values.read(j_start).faer_real();
                                let lkj = xj.faer_scale_real(dj.faer_inv());

                                let range = j_start.next()..row_idx.to_inclusive();
                                for (i, lij) in zip(
                                    &L_row_indices[range.clone()],
                                    L_values.rb().subslice(range).into_ref_iter(),
                                ) {
                                    let i = N.check(i.zx());
                                    let mut xi = x.read(i);
                                    let prod = lij.read().faer_conj().faer_mul(xj);
                                    xi = xi.faer_sub(prod);
                                    x.write(i, xi);
                                }

                                d = d.faer_sub(lkj.faer_mul(xj.faer_conj()).faer_real());

                                L_row_indices[row_idx] = *k.truncate();
                                L_values.write(row_idx, lkj);
                            }

                            let k_start = L_col_ptrs_start[k].zx();
                            L_row_indices[k_start] = *k.truncate();
                            L_values.write(k_start, E::faer_from_real(d));
                        }
                    },
                )
            },
        )
    }

    impl<'a, I: Index, E: Entity> SimplicialLdltRef<'a, I, E> {
        #[inline]
        pub fn new(symbolic: &'a SymbolicSimplicialCholesky<I>, values: SliceGroup<'a, E>) -> Self {
            assert!(values.len() == symbolic.len_values());
            Self { symbolic, values }
        }

        #[inline]
        pub fn symbolic(self) -> &'a SymbolicSimplicialCholesky<I> {
            self.symbolic
        }

        #[inline]
        pub fn values(self) -> SliceGroup<'a, E> {
            self.values
        }

        pub fn dense_solve_in_place_with_conj(
            self,
            rhs: MatMut<'_, E>,
            conj: Conj,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) where
            E: ComplexField,
        {
            let _ = parallelism;
            let _ = stack;
            let n = self.symbolic().nrows();
            let ld = SparseColMatRef::new(self.symbolic().ld_factors(), self.values());
            assert!(rhs.nrows() == n);

            let mut x = rhs;
            for mut x in x.rb_mut().into_col_chunks(4) {
                match x.ncols() {
                    1 => {
                        for j in 0..n {
                            let xj0 = x.read(j, 0);
                            for (i, lij) in zip(
                                ld.row_indices_of_col(j),
                                ld.values_of_col(j).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                            }
                        }
                    }
                    2 => {
                        for j in 0..n {
                            let xj0 = x.read(j, 0);
                            let xj1 = x.read(j, 1);
                            for (i, lij) in zip(
                                ld.row_indices_of_col(j),
                                ld.values_of_col(j).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                                x.write(i, 1, x.read(i, 1).faer_sub(lij.faer_mul(xj1)));
                            }
                        }
                    }
                    3 => {
                        for j in 0..n {
                            let xj0 = x.read(j, 0);
                            let xj1 = x.read(j, 1);
                            let xj2 = x.read(j, 2);
                            for (i, lij) in zip(
                                ld.row_indices_of_col(j),
                                ld.values_of_col(j).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                                x.write(i, 1, x.read(i, 1).faer_sub(lij.faer_mul(xj1)));
                                x.write(i, 2, x.read(i, 2).faer_sub(lij.faer_mul(xj2)));
                            }
                        }
                    }
                    4 => {
                        for j in 0..n {
                            let xj0 = x.read(j, 0);
                            let xj1 = x.read(j, 1);
                            let xj2 = x.read(j, 2);
                            let xj3 = x.read(j, 3);
                            for (i, lij) in zip(
                                ld.row_indices_of_col(j),
                                ld.values_of_col(j).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::Yes {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                x.write(i, 0, x.read(i, 0).faer_sub(lij.faer_mul(xj0)));
                                x.write(i, 1, x.read(i, 1).faer_sub(lij.faer_mul(xj1)));
                                x.write(i, 2, x.read(i, 2).faer_sub(lij.faer_mul(xj2)));
                                x.write(i, 3, x.read(i, 3).faer_sub(lij.faer_mul(xj3)));
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }

            for mut x in x.rb_mut().into_col_chunks(1) {
                for j in 0..n {
                    let d_inv = ld.values_of_col(j).read(0).faer_real().faer_inv();
                    x.write(j, 0, x.read(j, 0).faer_scale_real(d_inv));
                }
            }

            for mut x in x.rb_mut().into_col_chunks(4) {
                match x.ncols() {
                    1 => {
                        for j in (0..n).rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc0b = E::faer_zero();
                            let mut acc0c = E::faer_zero();
                            let mut acc0d = E::faer_zero();

                            let a = 0;
                            let b = 1;
                            let c = 2;
                            let d = 3;

                            let nrows = ld.row_indices_of_col_raw(j).len();
                            let rows_head = ld.row_indices_of_col_raw(j)[1..].chunks_exact(4);
                            let rows_tail = rows_head.remainder();
                            let (values_head, values_tail) =
                                ld.values_of_col(j).subslice(1..nrows).into_chunks(4);

                            for (i, lij) in zip(rows_head, values_head) {
                                let lija = lij.read(a);
                                let lijb = lij.read(b);
                                let lijc = lij.read(c);
                                let lijd = lij.read(d);
                                let lija = if conj == Conj::No {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                let lijb = if conj == Conj::No {
                                    lijb.faer_conj()
                                } else {
                                    lijb
                                };
                                let lijc = if conj == Conj::No {
                                    lijc.faer_conj()
                                } else {
                                    lijc
                                };
                                let lijd = if conj == Conj::No {
                                    lijd.faer_conj()
                                } else {
                                    lijd
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i[a].zx(), 0)));
                                acc0b = acc0b.faer_add(lijb.faer_mul(x.read(i[b].zx(), 0)));
                                acc0c = acc0c.faer_add(lijc.faer_mul(x.read(i[c].zx(), 0)));
                                acc0d = acc0d.faer_add(lijd.faer_mul(x.read(i[d].zx(), 0)));
                            }

                            for (i, lij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let lija = lij.read();
                                let lija = if conj == Conj::No {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i.zx(), 0)));
                            }

                            x.write(
                                j,
                                0,
                                x.read(j, 0).faer_sub(
                                    acc0a.faer_add(acc0b).faer_add(acc0c.faer_add(acc0d)),
                                ),
                            );
                        }
                    }
                    2 => {
                        for j in (0..n).rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc0b = E::faer_zero();
                            let mut acc1a = E::faer_zero();
                            let mut acc1b = E::faer_zero();

                            let a = 0;
                            let b = 1;

                            let nrows = ld.row_indices_of_col_raw(j).len();
                            let rows_head = ld.row_indices_of_col_raw(j)[1..].chunks_exact(2);
                            let rows_tail = rows_head.remainder();
                            let (values_head, values_tail) =
                                ld.values_of_col(j).subslice(1..nrows).into_chunks(2);

                            for (i, lij) in zip(rows_head, values_head) {
                                let lija = lij.read(a);
                                let lijb = lij.read(b);
                                let lija = if conj == Conj::No {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                let lijb = if conj == Conj::No {
                                    lijb.faer_conj()
                                } else {
                                    lijb
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i[a].zx(), 0)));
                                acc0b = acc0b.faer_add(lijb.faer_mul(x.read(i[b].zx(), 0)));
                                acc1a = acc1a.faer_add(lija.faer_mul(x.read(i[a].zx(), 1)));
                                acc1b = acc1b.faer_add(lijb.faer_mul(x.read(i[b].zx(), 1)));
                            }

                            for (i, lij) in zip(rows_tail, values_tail.into_ref_iter()) {
                                let lija = lij.read();
                                let lija = if conj == Conj::No {
                                    lija.faer_conj()
                                } else {
                                    lija
                                };
                                acc0a = acc0a.faer_add(lija.faer_mul(x.read(i.zx(), 0)));
                                acc1a = acc1a.faer_add(lija.faer_mul(x.read(i.zx(), 1)));
                            }

                            x.write(j, 0, x.read(j, 0).faer_sub(acc0a.faer_add(acc0b)));
                            x.write(j, 1, x.read(j, 1).faer_sub(acc1a.faer_add(acc1b)));
                        }
                    }
                    3 => {
                        for j in (0..n).rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc1a = E::faer_zero();
                            let mut acc2a = E::faer_zero();

                            for (i, lij) in zip(
                                ld.row_indices_of_col(j),
                                ld.values_of_col(j).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::No {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                acc0a = acc0a.faer_add(lij.faer_mul(x.read(i, 0)));
                                acc1a = acc1a.faer_add(lij.faer_mul(x.read(i, 1)));
                                acc2a = acc2a.faer_add(lij.faer_mul(x.read(i, 2)));
                            }

                            x.write(j, 0, x.read(j, 0).faer_sub(acc0a));
                            x.write(j, 1, x.read(j, 1).faer_sub(acc1a));
                            x.write(j, 2, x.read(j, 2).faer_sub(acc2a));
                        }
                    }
                    4 => {
                        for j in (0..n).rev() {
                            let mut acc0a = E::faer_zero();
                            let mut acc1a = E::faer_zero();
                            let mut acc2a = E::faer_zero();
                            let mut acc3a = E::faer_zero();

                            for (i, lij) in zip(
                                ld.row_indices_of_col(j),
                                ld.values_of_col(j).into_ref_iter(),
                            )
                            .skip(1)
                            {
                                let lij = lij.read();
                                let lij = if conj == Conj::No {
                                    lij.faer_conj()
                                } else {
                                    lij
                                };
                                acc0a = acc0a.faer_add(lij.faer_mul(x.read(i, 0)));
                                acc1a = acc1a.faer_add(lij.faer_mul(x.read(i, 1)));
                                acc2a = acc2a.faer_add(lij.faer_mul(x.read(i, 2)));
                                acc3a = acc3a.faer_add(lij.faer_mul(x.read(i, 3)));
                            }

                            x.write(j, 0, x.read(j, 0).faer_sub(acc0a));
                            x.write(j, 1, x.read(j, 1).faer_sub(acc1a));
                            x.write(j, 2, x.read(j, 2).faer_sub(acc2a));
                            x.write(j, 3, x.read(j, 3).faer_sub(acc3a));
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    impl<I: Index> SymbolicSimplicialCholesky<I> {
        #[inline]
        pub fn nrows(&self) -> usize {
            self.dimension
        }
        #[inline]
        pub fn ncols(&self) -> usize {
            self.nrows()
        }

        #[inline]
        pub fn len_values(&self) -> usize {
            self.row_indices.len()
        }

        #[inline]
        pub fn col_ptrs(&self) -> &[I] {
            &self.col_ptrs
        }

        #[inline]
        pub fn row_indices(&self) -> &[I] {
            &self.row_indices
        }

        #[inline]
        pub fn ld_factors(&self) -> SymbolicSparseColMatRef<'_, I> {
            unsafe {
                SymbolicSparseColMatRef::new_unchecked(
                    self.dimension,
                    self.dimension,
                    &self.col_ptrs,
                    None,
                    &self.row_indices,
                )
            }
        }

        pub fn dense_solve_in_place_req<E: Entity>(
            &self,
            rhs_ncols: usize,
        ) -> Result<StackReq, SizeOverflow> {
            let _ = rhs_ncols;
            Ok(StackReq::empty())
        }
    }

    pub fn factorize_simplicial_numeric_ldlt_req<I: Index, E: Entity>(
        n: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n_req = StackReq::try_new::<I>(n)?;
        StackReq::try_all_of([make_raw_req::<E>(n)?, n_req, n_req, n_req])
    }

    #[derive(Debug)]
    pub struct SimplicialLdltRef<'a, I, E: Entity> {
        symbolic: &'a SymbolicSimplicialCholesky<I>,
        values: SliceGroup<'a, E>,
    }

    #[derive(Debug)]
    pub struct SymbolicSimplicialCholesky<I> {
        dimension: usize,
        col_ptrs: Vec<I>,
        row_indices: Vec<I>,
        etree: Vec<I>,
    }
}

pub mod supernodal {
    use super::*;
    use assert2::{assert, debug_assert};

    fn ereach_super<'n, 'nsuper, I: Index>(
        A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
        super_etree: &Array<'nsuper, MaybeIdx<'nsuper, I>>,
        index_to_super: &Array<'n, Idx<'nsuper, I>>,
        current_row_positions: &mut Array<'nsuper, I>,
        row_indices: &mut [Idx<'n, I>],
        k: Idx<'n>,
        visited: &mut Array<'nsuper, I>,
    ) {
        let k_: I = *k.truncate();
        visited[index_to_super[k].zx()] = k_;
        for i in A.row_indices_of_col(k) {
            if i >= k {
                continue;
            }
            let mut supernode_i = index_to_super[i].zx();
            loop {
                if visited[supernode_i] == k_ {
                    break;
                }

                row_indices[current_row_positions[supernode_i].zx()] = k.truncate();
                current_row_positions[supernode_i].incr();

                visited[supernode_i] = k_;
                supernode_i = super_etree[supernode_i].sx().idx().unwrap();
            }
        }
    }

    #[derive(Debug)]
    pub struct SymbolicSupernodeRef<'a, I> {
        start: usize,
        pattern: &'a [I],
    }

    impl<'a, I: Index> SymbolicSupernodeRef<'a, I> {
        #[inline]
        pub fn start(self) -> usize {
            self.start
        }

        pub fn pattern(self) -> &'a [I] {
            self.pattern
        }
    }

    impl<'a, I: Index, E: Entity> SupernodeRef<'a, I, E> {
        #[inline]
        pub fn start(self) -> usize {
            self.symbolic.start
        }

        pub fn pattern(self) -> &'a [I] {
            self.symbolic.pattern
        }

        pub fn matrix(self) -> MatRef<'a, E> {
            self.matrix
        }
    }

    #[derive(Debug)]
    pub struct SupernodeRef<'a, I, E: Entity> {
        matrix: MatRef<'a, E>,
        symbolic: SymbolicSupernodeRef<'a, I>,
    }

    impl<'a, I: Index, E: Entity> SupernodalLdltRef<'a, I, E> {
        #[inline]
        pub fn new(symbolic: &'a SymbolicSupernodalCholesky<I>, values: SliceGroup<'a, E>) -> Self {
            assert!(values.len() == symbolic.len_values());
            Self { symbolic, values }
        }

        #[inline]
        pub fn symbolic(self) -> &'a SymbolicSupernodalCholesky<I> {
            self.symbolic
        }

        #[inline]
        pub fn values(self) -> SliceGroup<'a, E> {
            self.values
        }

        #[inline]
        pub fn supernode(self, s: usize) -> SupernodeRef<'a, I, E> {
            let symbolic = self.symbolic();
            let L_values = self.values();
            let s_start = symbolic.supernode_begin[s].zx();
            let s_end = symbolic.supernode_begin[s + 1].zx();

            let s_pattern = &symbolic.row_indices()[symbolic.col_ptrs_for_row_indices()[s].zx()
                ..symbolic.col_ptrs_for_row_indices()[s + 1].zx()];
            let s_ncols = s_end - s_start;
            let s_nrows = s_pattern.len() + s_ncols;

            let Ls = MatRef::<E>::from_column_major_slice(
                L_values
                    .subslice(
                        symbolic.col_ptrs_for_values()[s].zx()
                            ..symbolic.col_ptrs_for_values()[s + 1].zx(),
                    )
                    .into_inner(),
                s_nrows,
                s_ncols,
            );

            SupernodeRef {
                matrix: Ls,
                symbolic: SymbolicSupernodeRef {
                    start: s_start,
                    pattern: s_pattern,
                },
            }
        }

        pub fn dense_solve_in_place_with_conj(
            self,
            rhs: MatMut<'_, E>,
            conj: Conj,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) where
            E: ComplexField,
        {
            let symbolic = self.symbolic();
            let n = symbolic.nrows();
            assert!(rhs.nrows() == n);

            let mut x = rhs;
            let mut stack = stack;
            let k = x.ncols();
            for s in 0..symbolic.n_supernodes() {
                let s = self.supernode(s);
                let size = s.matrix.ncols();
                let Ls = s.matrix;
                let [Ls_top, Ls_bot] = Ls.split_at_row(size);
                let mut x_top = x.rb_mut().subrows(s.start(), size);
                faer_core::solve::solve_unit_lower_triangular_in_place_with_conj(
                    Ls_top,
                    conj,
                    x_top.rb_mut(),
                    parallelism,
                );

                let (mut tmp, _) = temp_mat_uninit::<E>(s.pattern().len(), k, stack.rb_mut());
                faer_core::mul::matmul_with_conj(
                    tmp.rb_mut(),
                    Ls_bot,
                    conj,
                    x_top.rb(),
                    Conj::No,
                    None,
                    E::faer_one(),
                    parallelism,
                );

                for j in 0..k {
                    for (idx, i) in s.pattern().iter().enumerate() {
                        let i = i.zx();
                        x.write(i, j, x.read(i, j).faer_sub(tmp.read(idx, j)))
                    }
                }
            }
            for s in 0..symbolic.n_supernodes() {
                let s = self.supernode(s);
                let size = s.matrix.ncols();
                let Ds = s.matrix.diagonal();
                for j in 0..k {
                    for idx in 0..size {
                        let d_inv = Ds.read(idx, 0).faer_real().faer_inv();
                        let i = idx + s.start();
                        x.write(i, j, x.read(i, j).faer_scale_real(d_inv))
                    }
                }
            }
            for s in (0..symbolic.n_supernodes()).rev() {
                let s = self.supernode(s);
                let size = s.matrix.ncols();
                let Ls = s.matrix;
                let [Ls_top, Ls_bot] = Ls.split_at_row(size);

                let (mut tmp, _) = temp_mat_uninit::<E>(s.pattern().len(), k, stack.rb_mut());
                for j in 0..k {
                    for (idx, i) in s.pattern().iter().enumerate() {
                        let i = i.zx();
                        tmp.write(idx, j, x.read(i, j));
                    }
                }

                let mut x_top = x.rb_mut().subrows(s.start(), size);
                faer_core::mul::matmul_with_conj(
                    x_top.rb_mut(),
                    Ls_bot.transpose(),
                    conj.compose(Conj::Yes),
                    tmp.rb(),
                    Conj::No,
                    Some(E::faer_one()),
                    E::faer_one().faer_neg(),
                    parallelism,
                );
                faer_core::solve::solve_unit_upper_triangular_in_place_with_conj(
                    Ls_top.transpose(),
                    conj.compose(Conj::Yes),
                    x_top.rb_mut(),
                    parallelism,
                );
            }
        }
    }

    impl<I: Index> SymbolicSupernodalCholesky<I> {
        #[inline]
        pub fn n_supernodes(&self) -> usize {
            self.supernode_postorder.len()
        }

        #[inline]
        pub fn nrows(&self) -> usize {
            self.dimension
        }
        #[inline]
        pub fn ncols(&self) -> usize {
            self.nrows()
        }

        #[inline]
        pub fn len_values(&self) -> usize {
            self.col_ptrs_for_values()[self.n_supernodes()].zx()
        }

        #[inline]
        pub fn supernode_begin(&self) -> &[I] {
            &self.supernode_begin[..self.n_supernodes()]
        }

        #[inline]
        pub fn supernode_end(&self) -> &[I] {
            &self.supernode_begin[1..]
        }

        #[inline]
        pub fn col_ptrs_for_row_indices(&self) -> &[I] {
            &self.col_ptrs_for_row_indices
        }

        #[inline]
        pub fn col_ptrs_for_values(&self) -> &[I] {
            &self.col_ptrs_for_values
        }

        #[inline]
        pub fn row_indices(&self) -> &[I] {
            &self.row_indices
        }

        #[inline]
        pub fn supernode(&self, s: usize) -> supernodal::SymbolicSupernodeRef<'_, I> {
            let symbolic = self;
            let start = symbolic.supernode_begin[s].zx();
            let pattern = &symbolic.row_indices()[symbolic.col_ptrs_for_row_indices()[s].zx()
                ..symbolic.col_ptrs_for_row_indices()[s + 1].zx()];
            supernodal::SymbolicSupernodeRef { start, pattern }
        }

        pub fn dense_solve_in_place_req<E: Entity>(
            &self,
            rhs_ncols: usize,
        ) -> Result<StackReq, SizeOverflow> {
            let mut req = StackReq::empty();
            let symbolic = self;
            for s in 0..symbolic.n_supernodes() {
                let s = self.supernode(s);
                req = req.try_or(temp_mat_req::<E>(s.pattern.len(), rhs_ncols)?)?;
            }
            Ok(req)
        }
    }

    pub fn factorize_supernodal_symbolic_req<I: Index>(n: usize) -> Result<StackReq, SizeOverflow> {
        let n_req = StackReq::try_new::<I>(n)?;
        StackReq::try_all_of([n_req, n_req, n_req, n_req])
    }

    pub fn factorize_supernodal_symbolic<I: Index>(
        A: SymbolicSparseColMatRef<'_, I>,
        etree: EtreeRef<'_, I>,
        col_counts: &[I],
        stack: PodStack<'_>,
        params: CholeskySymbolicSupernodalParams<'_>,
    ) -> Result<SymbolicSupernodalCholesky<I>, FaerError> {
        let n = A.nrows();
        assert!(A.nrows() == A.ncols());
        assert!(etree.inner().len() == n);
        assert!(col_counts.len() == n);
        ghost::with_size(n, |N| {
            ghost_factorize_supernodal_symbolic(
                ghost::SymbolicSparseColMatRef::new(A, N, N),
                etree.ghost_inner(N),
                Array::from_ref(col_counts, N),
                stack,
                params,
            )
        })
    }

    pub(crate) fn ghost_factorize_supernodal_symbolic<'n, I: Index>(
        A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
        etree: &Array<'n, MaybeIdx<'n, I>>,
        col_counts: &Array<'n, I>,
        stack: PodStack<'_>,
        params: CholeskySymbolicSupernodalParams<'_>,
    ) -> Result<SymbolicSupernodalCholesky<I>, FaerError> {
        let to_wide = |i: I| i.zx() as u128;
        let from_wide = |i: u128| I::truncate(i as usize);
        let from_wide_checked =
            |i: u128| -> Option<I> { (i <= to_wide(I::MAX)).then_some(I::truncate(i as usize)) };

        let N = A.nrows();
        let n = *N;

        let zero = I::truncate(0);
        let one = I::truncate(1);
        let none = I::truncate(NONE);

        if n == 0 {
            // would be funny if this allocation failed
            return Ok(SymbolicSupernodalCholesky {
                dimension: n,
                supernode_postorder: Vec::new(),
                supernode_postorder_inv: Vec::new(),
                descendent_count: Vec::new(),

                supernode_begin: try_collect([zero])?,
                col_ptrs_for_row_indices: try_collect([zero])?,
                col_ptrs_for_values: try_collect([zero])?,
                row_indices: Vec::new(),
            });
        }
        let mut original_stack = stack;

        let (index_to_super__, stack) = original_stack.rb_mut().make_raw::<I>(n);
        let (super_etree__, stack) = stack.make_raw::<I>(n);
        let (supernode_sizes__, stack) = stack.make_raw::<I>(n);
        let (child_count__, _) = stack.make_raw::<I>(n);

        let child_count = Array::from_mut(child_count__, N);
        let index_to_super = Array::from_mut(index_to_super__, N);

        mem::fill_zero(child_count);
        for j in N.indices() {
            if let Some(parent) = etree[j].idx() {
                child_count[parent.zx()].incr();
            }
        }

        mem::fill_zero(supernode_sizes__);
        let mut current_supernode = 0usize;
        supernode_sizes__[0] = one;
        for (j_prev, j) in zip(N.indices().take(n - 1), N.indices().skip(1)) {
            let is_parent_of_prev = (*etree[j_prev]).sx() == *j;
            let is_parent_of_only_prev = child_count[j] == one;
            let same_pattern_as_prev = col_counts[j_prev] == col_counts[j] + one;

            if !(is_parent_of_prev && is_parent_of_only_prev && same_pattern_as_prev) {
                current_supernode += 1;
            }
            supernode_sizes__[current_supernode].incr();
        }
        let n_fundamental_supernodes = current_supernode + 1;

        // last n elements contain supernode degrees
        let supernode_begin__ = ghost::with_size(
            n_fundamental_supernodes,
            |N_FUNDAMENTAL_SUPERNODES| -> Result<Vec<I>, FaerError> {
                let supernode_sizes = Array::from_mut(
                    &mut supernode_sizes__[..n_fundamental_supernodes],
                    N_FUNDAMENTAL_SUPERNODES,
                );
                let super_etree = Array::from_mut(
                    &mut super_etree__[..n_fundamental_supernodes],
                    N_FUNDAMENTAL_SUPERNODES,
                );

                let mut supernode_begin = 0usize;
                for s in N_FUNDAMENTAL_SUPERNODES.indices() {
                    let size = supernode_sizes[s].zx();
                    (**index_to_super)[supernode_begin..][..size].fill(*s.truncate::<I>());
                    supernode_begin += size;
                }

                let index_to_super = Array::from_mut(
                    Idx::slice_mut_checked(index_to_super, N_FUNDAMENTAL_SUPERNODES),
                    N,
                );

                let mut supernode_begin = 0usize;
                for s in N_FUNDAMENTAL_SUPERNODES.indices() {
                    let size = supernode_sizes[s].zx();
                    let last = supernode_begin + size - 1;
                    let last = N.check(last);
                    if let Some(parent) = etree[last].idx() {
                        super_etree[s] = *index_to_super[parent.zx()];
                    } else {
                        super_etree[s] = none;
                    }
                    supernode_begin += size;
                }

                let super_etree = Array::from_mut(
                    MaybeIdx::slice_mut_checked(super_etree, N_FUNDAMENTAL_SUPERNODES),
                    N_FUNDAMENTAL_SUPERNODES,
                );

                if let Some(relax) = params.relax {
                    let req = || -> Result<StackReq, SizeOverflow> {
                        let req = StackReq::try_new::<I>(n_fundamental_supernodes)?;
                        StackReq::try_all_of([req; 5])
                    };
                    let mut mem = dyn_stack::GlobalPodBuffer::try_new(req().map_err(nomem)?)
                        .map_err(nomem)?;
                    let stack = PodStack::new(&mut mem);

                    let child_lists = &mut (**child_count)[..n_fundamental_supernodes];
                    let (child_list_heads, stack) = stack.make_raw::<I>(n_fundamental_supernodes);
                    let (last_merged_children, stack) =
                        stack.make_raw::<I>(n_fundamental_supernodes);
                    let (merge_parents, stack) = stack.make_raw::<I>(n_fundamental_supernodes);
                    let (fundamental_supernode_degrees, stack) =
                        stack.make_raw::<I>(n_fundamental_supernodes);
                    let (num_zeros, _) = stack.make_raw::<I>(n_fundamental_supernodes);

                    let child_lists = Array::from_mut(
                        ghost::fill_none(child_lists, N_FUNDAMENTAL_SUPERNODES),
                        N_FUNDAMENTAL_SUPERNODES,
                    );
                    let child_list_heads = Array::from_mut(
                        ghost::fill_none(child_list_heads, N_FUNDAMENTAL_SUPERNODES),
                        N_FUNDAMENTAL_SUPERNODES,
                    );
                    let last_merged_children = Array::from_mut(
                        ghost::fill_none(last_merged_children, N_FUNDAMENTAL_SUPERNODES),
                        N_FUNDAMENTAL_SUPERNODES,
                    );
                    let merge_parents = Array::from_mut(
                        ghost::fill_none(merge_parents, N_FUNDAMENTAL_SUPERNODES),
                        N_FUNDAMENTAL_SUPERNODES,
                    );
                    let fundamental_supernode_degrees =
                        Array::from_mut(fundamental_supernode_degrees, N_FUNDAMENTAL_SUPERNODES);
                    let num_zeros = Array::from_mut(num_zeros, N_FUNDAMENTAL_SUPERNODES);

                    let mut supernode_begin = 0usize;
                    for s in N_FUNDAMENTAL_SUPERNODES.indices() {
                        let size = supernode_sizes[s].zx();
                        fundamental_supernode_degrees[s] =
                            col_counts[N.check(supernode_begin + size - 1)] - one;
                        supernode_begin += size;
                    }

                    for s in N_FUNDAMENTAL_SUPERNODES.indices() {
                        if let Some(parent) = super_etree[s].idx() {
                            let parent = parent.zx();
                            child_lists[s] = child_list_heads[parent];
                            child_list_heads[parent] = MaybeIdx::from_index(s.truncate());
                        }
                    }

                    mem::fill_zero(num_zeros);
                    for parent in N_FUNDAMENTAL_SUPERNODES.indices() {
                        loop {
                            let mut merging_child = MaybeIdx::none();
                            let mut num_new_zeros = 0usize;
                            let mut num_merged_zeros = 0usize;
                            let mut largest_mergable_size = 0usize;

                            let mut child_ = child_list_heads[parent];
                            while let Some(child) = child_.idx() {
                                let child = child.zx();
                                if *child + 1 != *parent {
                                    child_ = child_lists[child];
                                    continue;
                                }

                                if merge_parents[child].idx().is_some() {
                                    child_ = child_lists[child];
                                    continue;
                                }

                                let parent_size = supernode_sizes[parent].zx();
                                let child_size = supernode_sizes[child].zx();
                                if child_size < largest_mergable_size {
                                    child_ = child_lists[child];
                                    continue;
                                }

                                let parent_degree = fundamental_supernode_degrees[parent].zx();
                                let child_degree = fundamental_supernode_degrees[child].zx();

                                let num_parent_zeros = num_zeros[parent].zx();
                                let num_child_zeros = num_zeros[child].zx();

                                let status_num_merged_zeros = {
                                    let num_new_zeros =
                                        (parent_size + parent_degree - child_degree) * child_size;

                                    if num_new_zeros == 0 {
                                        num_parent_zeros + num_child_zeros
                                    } else {
                                        let num_old_zeros = num_child_zeros + num_parent_zeros;
                                        let num_zeros = num_new_zeros + num_old_zeros;

                                        let combined_size = child_size + parent_size;
                                        let num_expanded_entries =
                                            (combined_size * (combined_size + 1)) / 2
                                                + parent_degree * combined_size;

                                        let f = || {
                                            for cutoff in relax {
                                                let num_zeros_cutoff =
                                                    num_expanded_entries as f64 * cutoff.1;
                                                if cutoff.0 >= combined_size
                                                    && num_zeros_cutoff >= num_zeros as f64
                                                {
                                                    return num_zeros;
                                                }
                                            }
                                            NONE
                                        };
                                        f()
                                    }
                                };
                                if status_num_merged_zeros == NONE {
                                    child_ = child_lists[child];
                                    continue;
                                }

                                let num_proposed_new_zeros =
                                    status_num_merged_zeros - (num_child_zeros + num_parent_zeros);
                                if child_size > largest_mergable_size
                                    || num_proposed_new_zeros < num_new_zeros
                                {
                                    merging_child = MaybeIdx::from_index(child);
                                    num_new_zeros = num_proposed_new_zeros;
                                    num_merged_zeros = status_num_merged_zeros;
                                    largest_mergable_size = child_size;
                                }

                                child_ = child_lists[child];
                            }

                            if let Some(merging_child) = merging_child.idx() {
                                supernode_sizes[parent] =
                                    supernode_sizes[parent] + supernode_sizes[merging_child];
                                supernode_sizes[merging_child] = zero;
                                num_zeros[parent] = I::truncate(num_merged_zeros);

                                merge_parents[merging_child] =
                                    if let Some(child) = last_merged_children[parent].idx() {
                                        MaybeIdx::from_index(child)
                                    } else {
                                        MaybeIdx::from_index(parent.truncate())
                                    };

                                last_merged_children[parent] = if let Some(child) =
                                    last_merged_children[merging_child].idx()
                                {
                                    MaybeIdx::from_index(child)
                                } else {
                                    MaybeIdx::from_index(merging_child.truncate())
                                };
                            } else {
                                break;
                            }
                        }
                    }

                    let original_to_relaxed = last_merged_children;
                    original_to_relaxed.fill(MaybeIdx::none_index());

                    let mut pos = 0usize;
                    for s in N_FUNDAMENTAL_SUPERNODES.indices() {
                        let idx = N_FUNDAMENTAL_SUPERNODES.check(pos);
                        let size = supernode_sizes[s];
                        let degree = fundamental_supernode_degrees[s];
                        if size > zero {
                            supernode_sizes[idx] = size;
                            fundamental_supernode_degrees[idx] = degree;
                            original_to_relaxed[s] = MaybeIdx::from_index(idx.truncate());

                            pos += 1;
                        }
                    }
                    let n_relaxed_supernodes = pos;

                    let mut supernode_begin__ = try_zeroed(n_relaxed_supernodes + 1)?;
                    supernode_begin__[1..].copy_from_slice(
                        &(**fundamental_supernode_degrees)[..n_relaxed_supernodes],
                    );

                    Ok(supernode_begin__)
                } else {
                    let mut supernode_begin__ = try_zeroed(n_fundamental_supernodes + 1)?;

                    let mut supernode_begin = 0usize;
                    for s in N_FUNDAMENTAL_SUPERNODES.indices() {
                        let size = supernode_sizes[s].zx();
                        supernode_begin__[*s + 1] =
                            col_counts[N.check(supernode_begin + size - 1)] - one;
                        supernode_begin += size;
                    }

                    Ok(supernode_begin__)
                }
            },
        )?;

        let n_supernodes = supernode_begin__.len() - 1;

        let (supernode_begin__, col_ptrs_for_row_indices__, col_ptrs_for_values__, row_indices__) =
            ghost::with_size(
                n_supernodes,
                |N_SUPERNODES| -> Result<(Vec<I>, Vec<I>, Vec<I>, Vec<I>), FaerError> {
                    let supernode_sizes =
                        Array::from_mut(&mut supernode_sizes__[..n_supernodes], N_SUPERNODES);

                    if n_supernodes != n_fundamental_supernodes {
                        let mut supernode_begin = 0usize;
                        for s in N_SUPERNODES.indices() {
                            let size = supernode_sizes[s].zx();
                            (**index_to_super)[supernode_begin..][..size].fill(*s.truncate::<I>());
                            supernode_begin += size;
                        }

                        let index_to_super = Array::from_mut(
                            Idx::slice_mut_checked(index_to_super, N_SUPERNODES),
                            N,
                        );
                        let super_etree =
                            Array::from_mut(&mut super_etree__[..n_supernodes], N_SUPERNODES);

                        let mut supernode_begin = 0usize;
                        for s in N_SUPERNODES.indices() {
                            let size = supernode_sizes[s].zx();
                            let last = supernode_begin + size - 1;
                            if let Some(parent) = etree[N.check(last)].idx() {
                                super_etree[s] = *index_to_super[parent.zx()];
                            } else {
                                super_etree[s] = none;
                            }
                            supernode_begin += size;
                        }
                    }

                    let index_to_super =
                        Array::from_mut(Idx::slice_mut_checked(index_to_super, N_SUPERNODES), N);

                    let mut supernode_begin__ = supernode_begin__;
                    let mut col_ptrs_for_row_indices__ = try_zeroed::<I>(n_supernodes + 1)?;
                    let mut col_ptrs_for_values__ = try_zeroed::<I>(n_supernodes + 1)?;

                    let mut row_ptr = zero;
                    let mut val_ptr = zero;

                    supernode_begin__[0] = zero;

                    let mut row_indices__ = {
                        let mut wide_val_count = 0u128;
                        for (s, [current, next]) in zip(
                            N_SUPERNODES.indices(),
                            windows2(Cell::as_slice_of_cells(Cell::from_mut(
                                &mut *supernode_begin__,
                            ))),
                        ) {
                            let degree = next.get();
                            let ncols = supernode_sizes[s];
                            let nrows = degree + ncols;
                            supernode_sizes[s] = row_ptr;
                            next.set(current.get() + ncols);

                            col_ptrs_for_row_indices__[*s] = row_ptr;
                            col_ptrs_for_values__[*s] = val_ptr;

                            let wide_matrix_size = to_wide(nrows) * to_wide(ncols);
                            wide_val_count += wide_matrix_size;

                            row_ptr += degree;
                            val_ptr = from_wide(to_wide(val_ptr) + wide_matrix_size);
                        }
                        col_ptrs_for_row_indices__[n_supernodes] = row_ptr;
                        col_ptrs_for_values__[n_supernodes] = val_ptr;
                        from_wide_checked(wide_val_count).ok_or(FaerError::IndexOverflow)?;

                        try_zeroed::<I>(row_ptr.zx())?
                    };

                    let super_etree = Array::from_ref(
                        MaybeIdx::slice_ref_checked(&super_etree__[..n_supernodes], N_SUPERNODES),
                        N_SUPERNODES,
                    );

                    let current_row_positions = supernode_sizes;

                    let row_indices = Idx::slice_mut_checked(&mut row_indices__, N);
                    let visited =
                        Array::from_mut(&mut (**child_count)[..n_supernodes], N_SUPERNODES);
                    mem::fill_none(visited);
                    for s in N_SUPERNODES.indices() {
                        let k1 = ghost::IdxInclusive::new_checked(supernode_begin__[*s].zx(), N);
                        let k2 =
                            ghost::IdxInclusive::new_checked(supernode_begin__[*s + 1].zx(), N);

                        for k in k1.range_to(k2) {
                            ereach_super(
                                A,
                                super_etree,
                                index_to_super,
                                current_row_positions,
                                row_indices,
                                k,
                                visited,
                            );
                        }
                    }

                    debug_assert!(**current_row_positions == col_ptrs_for_row_indices__[1..]);

                    Ok((
                        supernode_begin__,
                        col_ptrs_for_row_indices__,
                        col_ptrs_for_values__,
                        row_indices__,
                    ))
                },
            )?;

        let mut supernode_etree__ = try_collect(super_etree__[..n_supernodes].iter().copied())?;
        let mut supernode_postorder__ = try_zeroed::<I>(n_supernodes)?;

        let mut descendent_count__ = try_zeroed::<I>(n_supernodes)?;

        ghost::with_size(n_supernodes, |N_SUPERNODES| {
            let post = Array::from_mut(&mut supernode_postorder__, N_SUPERNODES);
            let desc_count = Array::from_mut(&mut descendent_count__, N_SUPERNODES);
            let etree = Array::from_ref(
                MaybeIdx::slice_ref_checked(&supernode_etree__, N_SUPERNODES),
                N_SUPERNODES,
            );

            for s in N_SUPERNODES.indices() {
                if let Some(parent) = etree[s].idx() {
                    let parent = parent.zx();
                    desc_count[parent] = desc_count[parent] + desc_count[s] + one;
                }
            }

            ghost_postorder(post, etree, original_stack);
            let post_inv = Array::from_mut(&mut supernode_etree__, N_SUPERNODES);
            for i in N_SUPERNODES.indices() {
                post_inv[N_SUPERNODES.check(post[i].zx())] = *i.truncate();
            }
        });

        Ok(SymbolicSupernodalCholesky {
            dimension: n,
            supernode_postorder: supernode_postorder__,
            supernode_postorder_inv: supernode_etree__,
            descendent_count: descendent_count__,
            supernode_begin: supernode_begin__,
            col_ptrs_for_row_indices: col_ptrs_for_row_indices__,
            col_ptrs_for_values: col_ptrs_for_values__,
            row_indices: row_indices__,
        })
    }

    #[inline]
    fn partition_fn<I: Index>(idx: usize) -> impl Fn(&I) -> bool {
        let idx = I::truncate(idx);
        move |&i| i < idx
    }

    pub fn factorize_supernodal_numeric_ldlt_req<I: Index, E: Entity>(
        symbolic: &SymbolicSupernodalCholesky<I>,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let n_supernodes = symbolic.n_supernodes();
        let n = symbolic.nrows();
        let post = &*symbolic.supernode_postorder;
        let post_inv = &*symbolic.supernode_postorder_inv;

        let desc_count = &*symbolic.descendent_count;

        let col_ptr_row = &*symbolic.col_ptrs_for_row_indices;
        let row_ind = &*symbolic.row_indices;

        let mut req = StackReq::empty();
        for s in 0..n_supernodes {
            let s_start = symbolic.supernode_begin[s].zx();
            let s_end = symbolic.supernode_begin[s + 1].zx();

            let s_ncols = s_end - s_start;

            let s_postordered = post_inv[s].zx();
            let desc_count = desc_count[s].zx();
            for d in &post[s_postordered - desc_count..s_postordered] {
                let mut d_req = StackReq::empty();

                let d = d.zx();
                let d_start = symbolic.supernode_begin[d].zx();
                let d_end = symbolic.supernode_begin[d + 1].zx();

                let d_pattern = &row_ind[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()];

                let d_ncols = d_end - d_start;

                let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
                let d_pattern_mid_len =
                    d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));

                d_req = d_req.try_and(temp_mat_req::<E>(
                    d_pattern.len() - d_pattern_start,
                    d_pattern_mid_len,
                )?)?;
                d_req = d_req.try_and(temp_mat_req::<E>(d_ncols, d_pattern_mid_len)?)?;
                req = req.try_or(d_req)?;
            }
            req = req.try_or(
                faer_cholesky::ldlt_diagonal::compute::raw_cholesky_in_place_req::<E>(
                    s_ncols,
                    parallelism,
                    Default::default(),
                )?,
            )?;
        }
        req.try_and(StackReq::try_new::<I>(n)?)
    }

    pub fn factorize_supernodal_numeric_ldlt<I: Index, E: ComplexField>(
        L_values: SliceGroupMut<'_, E>,
        A_lower: SparseColMatRef<'_, I, E>,
        regularization: LdltRegularization<'_, E>,
        symbolic: &SymbolicSupernodalCholesky<I>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) -> usize {
        let n_supernodes = symbolic.n_supernodes();
        let n = symbolic.nrows();
        let mut L_values = L_values;
        let mut dynamic_regularization_count = 0usize;

        assert!(A_lower.nrows() == n);
        assert!(A_lower.ncols() == n);
        assert!(L_values.len() == symbolic.len_values());

        let none = I::truncate(NONE);

        let post = &*symbolic.supernode_postorder;
        let post_inv = &*symbolic.supernode_postorder_inv;

        let desc_count = &*symbolic.descendent_count;

        let col_ptr_row = &*symbolic.col_ptrs_for_row_indices;
        let col_ptr_val = &*symbolic.col_ptrs_for_values;
        let row_ind = &*symbolic.row_indices;

        // mapping from global indices to local
        let (global_to_local, mut stack) = stack.make_raw::<I>(n);
        mem::fill_none(global_to_local);

        for s in 0..n_supernodes {
            let s_start = symbolic.supernode_begin[s].zx();
            let s_end = symbolic.supernode_begin[s + 1].zx();

            let s_pattern = &row_ind[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()];
            let s_ncols = s_end - s_start;
            let s_nrows = s_pattern.len() + s_ncols;

            for (i, &row) in s_pattern.iter().enumerate() {
                global_to_local[row.zx()] = I::truncate(i + s_ncols);
            }

            let (head, tail) = L_values.rb_mut().split_at(col_ptr_val[s].zx());
            let head = head.rb();
            let mut Ls = MatMut::<E>::from_column_major_slice(
                tail.subslice(0..(col_ptr_val[s + 1] - col_ptr_val[s]).zx())
                    .into_inner(),
                s_nrows,
                s_ncols,
            );

            for j in s_start..s_end {
                let j_shifted = j - s_start;
                for (i, val) in zip(
                    A_lower.row_indices_of_col(j),
                    A_lower.values_of_col(j).into_ref_iter(),
                ) {
                    let val = val.read();
                    if i >= s_end {
                        Ls.write(global_to_local[i].sx(), j_shifted, val);
                    } else if i >= j {
                        Ls.write(i - s_start, j_shifted, val);
                    }
                }
            }

            let s_postordered = post_inv[s].zx();
            let desc_count = desc_count[s].zx();
            for d in &post[s_postordered - desc_count..s_postordered] {
                let d = d.zx();
                let d_start = symbolic.supernode_begin[d].zx();
                let d_end = symbolic.supernode_begin[d + 1].zx();

                let d_pattern = &row_ind[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()];
                let d_ncols = d_end - d_start;
                let d_nrows = d_pattern.len() + d_ncols;

                let Ld = MatRef::<E>::from_column_major_slice(
                    head.subslice(col_ptr_val[d].zx()..col_ptr_val[d + 1].zx())
                        .into_inner(),
                    d_nrows,
                    d_ncols,
                );

                let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
                let d_pattern_mid_len =
                    d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));
                let d_pattern_mid = d_pattern_start + d_pattern_mid_len;

                let [Ld_top, Ld_mid_bot] = Ld.split_at_row(d_ncols);
                let [_, Ld_mid_bot] = Ld_mid_bot.split_at_row(d_pattern_start);
                let [Ld_mid, Ld_bot] = Ld_mid_bot.split_at_row(d_pattern_mid_len);
                let D = Ld_top.diagonal();

                let stack = stack.rb_mut();

                let (tmp, stack) =
                    temp_mat_uninit::<E>(Ld_mid_bot.nrows(), d_pattern_mid_len, stack);
                let (tmp2, _) = temp_mat_uninit::<E>(Ld_mid.ncols(), Ld_mid.nrows(), stack);
                let mut Ld_mid_x_D = tmp2.transpose();

                for i in 0..d_pattern_mid_len {
                    for j in 0..d_ncols {
                        Ld_mid_x_D.write(
                            i,
                            j,
                            Ld_mid.read(i, j).faer_scale_real(D.read(j, 0).faer_real()),
                        );
                    }
                }

                let [mut tmp_top, mut tmp_bot] = tmp.split_at_row(d_pattern_mid_len);

                use faer_core::{mul, mul::triangular};
                triangular::matmul(
                    tmp_top.rb_mut(),
                    triangular::BlockStructure::TriangularLower,
                    Ld_mid,
                    triangular::BlockStructure::Rectangular,
                    Ld_mid_x_D.rb().adjoint(),
                    triangular::BlockStructure::Rectangular,
                    None,
                    E::faer_one(),
                    parallelism,
                );
                mul::matmul(
                    tmp_bot.rb_mut(),
                    Ld_bot,
                    Ld_mid_x_D.rb().adjoint(),
                    None,
                    E::faer_one(),
                    parallelism,
                );
                for (j_idx, j) in d_pattern[d_pattern_start..d_pattern_mid].iter().enumerate() {
                    let j = j.zx();
                    let j_s = j - s_start;
                    for (i_idx, i) in d_pattern[d_pattern_start..d_pattern_mid][j_idx..]
                        .iter()
                        .enumerate()
                    {
                        let i_idx = i_idx + j_idx;

                        let i = i.zx();
                        let i_s = i - s_start;

                        debug_assert!(i_s >= j_s);

                        unsafe {
                            Ls.write_unchecked(
                                i_s,
                                j_s,
                                Ls.read_unchecked(i_s, j_s)
                                    .faer_sub(tmp_top.read_unchecked(i_idx, j_idx)),
                            )
                        };
                    }
                }

                for (j_idx, j) in d_pattern[d_pattern_start..d_pattern_mid].iter().enumerate() {
                    let j = j.zx();
                    let j_s = j - s_start;
                    for (i_idx, i) in d_pattern[d_pattern_mid..].iter().enumerate() {
                        let i = i.zx();
                        let i_s = global_to_local[i].zx();
                        unsafe {
                            Ls.write_unchecked(
                                i_s,
                                j_s,
                                Ls.read_unchecked(i_s, j_s)
                                    .faer_sub(tmp_bot.read_unchecked(i_idx, j_idx)),
                            )
                        };
                    }
                }
            }

            let [mut Ls_top, mut Ls_bot] = Ls.rb_mut().split_at_row(s_ncols);

            let params = Default::default();
            dynamic_regularization_count +=
                faer_cholesky::ldlt_diagonal::compute::raw_cholesky_in_place(
                    Ls_top.rb_mut(),
                    LdltRegularization {
                        dynamic_regularization_signs: regularization
                            .dynamic_regularization_signs
                            .map(|signs| &signs[s_start..s_end]),
                        ..regularization
                    },
                    parallelism,
                    stack.rb_mut(),
                    params,
                );
            zipped!(Ls_top.rb_mut())
                .for_each_triangular_upper(faer_core::zip::Diag::Skip, |mut x| {
                    x.write(E::faer_zero())
                });
            faer_core::solve::solve_unit_lower_triangular_in_place(
                Ls_top.rb().conjugate(),
                Ls_bot.rb_mut().transpose(),
                parallelism,
            );
            for j in 0..s_ncols {
                let d = Ls_top.read(j, j).faer_real().faer_inv();
                for i in 0..s_pattern.len() {
                    Ls_bot.write(i, j, Ls_bot.read(i, j).faer_scale_real(d));
                }
            }

            for &row in s_pattern {
                global_to_local[row.zx()] = none;
            }
        }
        dynamic_regularization_count
    }

    #[derive(Debug)]
    pub struct SupernodalLdltRef<'a, I, E: Entity> {
        symbolic: &'a SymbolicSupernodalCholesky<I>,
        values: SliceGroup<'a, E>,
    }

    #[derive(Debug)]
    pub struct SymbolicSupernodalCholesky<I> {
        dimension: usize,
        supernode_postorder: Vec<I>,
        supernode_postorder_inv: Vec<I>,
        descendent_count: Vec<I>,

        supernode_begin: Vec<I>,
        col_ptrs_for_row_indices: Vec<I>,
        col_ptrs_for_values: Vec<I>,
        row_indices: Vec<I>,
    }

    #[derive(Copy, Clone, Debug)]
    pub struct CholeskySymbolicSupernodalParams<'a> {
        pub relax: Option<&'a [(usize, f64)]>,
    }

    impl Default for CholeskySymbolicSupernodalParams<'_> {
        #[inline]
        fn default() -> Self {
            Self {
                relax: Some(&[(4, 1.0), (16, 0.8), (48, 0.1), (usize::MAX, 0.05)]),
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EtreeRef<'a, I> {
    inner: &'a [I],
}

impl<'a, I: Index> EtreeRef<'a, I> {
    #[inline]
    pub fn inner(self) -> &'a [I] {
        self.inner
    }

    #[inline]
    #[track_caller]
    fn ghost_inner<'n>(self, N: ghost::Size<'n>) -> &'a Array<'n, MaybeIdx<'n, I>> {
        assert!(self.inner.len() == *N);
        unsafe { Array::from_ref(MaybeIdx::slice_ref_unchecked(self.inner, N), N) }
    }
}

// workspace: I*(n)
fn ghost_prefactorize_symbolic<'n, 'out, I: Index>(
    etree: &'out mut Array<'n, I>,
    col_counts: &mut Array<'n, I>,
    A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
    stack: PodStack<'_>,
) -> &'out mut Array<'n, MaybeIdx<'n, I>> {
    let N = A.ncols();
    let etree: &mut [I] = etree;
    let (visited, _) = stack.make_raw::<I>(*N);
    let etree = Array::from_mut(ghost::fill_none(etree, N), N);
    let visited = Array::from_mut(visited, N);

    for j in N.indices() {
        let j_ = j.truncate::<I>();
        visited[j] = *j_;
        col_counts[j] = I::truncate(1);

        for mut i in A.row_indices_of_col(j) {
            if i < j {
                loop {
                    if visited[i] == *j_ {
                        break;
                    }

                    let next_i = if let Some(parent) = etree[i].idx() {
                        parent.zx()
                    } else {
                        etree[i] = MaybeIdx::from_index(j_);
                        j
                    };

                    col_counts[i].incr();
                    visited[i] = *j_;
                    i = next_i;
                }
            }
        }
    }

    etree
}

pub fn prefactorize_symbolic<'out, I: Index>(
    etree: &'out mut [I],
    col_counts: &mut [I],
    A: SymbolicSparseColMatRef<'_, I>,
    stack: PodStack<'_>,
) -> EtreeRef<'out, I> {
    let n = A.nrows();
    assert!(A.nrows() == A.ncols());
    assert!(etree.len() == n);
    assert!(col_counts.len() == n);

    ghost::with_size(n, |N| {
        ghost_prefactorize_symbolic(
            Array::from_mut(etree, N),
            Array::from_mut(col_counts, N),
            ghost::SymbolicSparseColMatRef::new(A, N, N),
            stack,
        );
    });

    EtreeRef { inner: etree }
}

#[derive(Debug, Copy, Clone)]
#[doc(hidden)]
pub struct ComputationModel {
    pub ldl: [f64; 4],
    pub triangular_solve: [f64; 6],
    pub matmul: [f64; 6],
    pub assembly: [f64; 4],
}

impl ComputationModel {
    #[allow(clippy::excessive_precision)]
    pub const OPENBLAS_I7_1185G7: Self = ComputationModel {
        ldl: [
            3.527141723946874224e-07,
            -5.382557351808083451e-08,
            4.677984682984275924e-09,
            7.384424667338682676e-12,
        ],
        triangular_solve: [
            1.101115592925888909e-06,
            6.936563076265144074e-07,
            -1.827661167503034051e-09,
            1.959826916788009885e-09,
            1.079857543323972179e-09,
            2.963338652996178598e-11,
        ],
        matmul: [
            6.14190596709488416e-07,
            -4.489948374364910256e-09,
            5.943145978912038475e-10,
            -1.201283634136652872e-08,
            1.266858215451465993e-09,
            2.624001993284897048e-11,
        ],
        assembly: [
            3.069607518266660019e-07,
            3.763778311956422235e-08,
            1.991443920635728855e-07,
            3.788938150548870089e-09,
        ],
    };

    #[inline]
    pub fn ldl_estimate(&self, n: f64) -> f64 {
        let p = self.ldl;
        p[0] + n * (p[1] + n * (p[2] + n * p[3]))
    }

    #[inline]
    pub fn triangular_solve_estimate(&self, n: f64, k: f64) -> f64 {
        let p = self.triangular_solve;
        p[0] + n * (p[1] + n * p[2]) + k * (p[3] + n * (p[4] + n * p[5]))
    }

    #[inline]
    pub fn matmul_estimate(&self, m: f64, n: f64, k: f64) -> f64 {
        let p = self.matmul;
        p[0] + (m + n) * p[1] + (m * n) * p[2] + k * (p[3] + (m + n) * p[4] + (m * n) * p[5])
    }

    #[inline]
    pub fn assembly_estimate(&self, br: f64, bc: f64) -> f64 {
        let p = self.assembly;
        p[0] + br * p[1] + bc * p[2] + br * bc * p[3]
    }
}

#[derive(Debug)]
pub enum SymbolicCholeskyRaw<I> {
    Simplicial(simplicial::SymbolicSimplicialCholesky<I>),
    Supernodal(supernodal::SymbolicSupernodalCholesky<I>),
}

#[derive(Debug)]
pub struct SymbolicCholesky<I> {
    raw: SymbolicCholeskyRaw<I>,
    perm_fwd: Vec<I>,
    perm_inv: Vec<I>,
    A_nnz: usize,
}

impl<I: Index> SymbolicCholesky<I> {
    #[inline]
    pub fn nrows(&self) -> usize {
        match &self.raw {
            SymbolicCholeskyRaw::Simplicial(this) => this.nrows(),
            SymbolicCholeskyRaw::Supernodal(this) => this.nrows(),
        }
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.nrows()
    }

    #[inline]
    pub fn raw(&self) -> &SymbolicCholeskyRaw<I> {
        &self.raw
    }

    #[inline]
    pub fn perm(&self) -> PermutationRef<'_, I> {
        unsafe { PermutationRef::new_unchecked(&self.perm_fwd, &self.perm_inv) }
    }

    #[inline]
    pub fn len_values(&self) -> usize {
        match &self.raw {
            SymbolicCholeskyRaw::Simplicial(this) => this.len_values(),
            SymbolicCholeskyRaw::Supernodal(this) => this.len_values(),
        }
    }

    #[inline]
    pub fn factorize_numeric_ldlt_req<E: Entity>(
        &self,
        with_regularization_signs: bool,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let n = self.nrows();
        let A_nnz = self.A_nnz;

        let regularization_signs = if with_regularization_signs {
            StackReq::try_new::<i8>(n)?
        } else {
            StackReq::empty()
        };

        let n_req = StackReq::try_new::<I>(n)?;
        let A_req = StackReq::try_all_of([
            make_raw_req::<E>(A_nnz)?,
            StackReq::try_new::<I>(n + 1)?,
            StackReq::try_new::<I>(A_nnz)?,
        ])?;
        let permute_req = n_req;

        let factor_req = match &self.raw {
            SymbolicCholeskyRaw::Simplicial(_) => {
                simplicial::factorize_simplicial_numeric_ldlt_req::<I, E>(n)?
            }
            SymbolicCholeskyRaw::Supernodal(this) => {
                supernodal::factorize_supernodal_numeric_ldlt_req::<I, E>(this, parallelism)?
            }
        };

        StackReq::try_all_of([
            regularization_signs,
            A_req,
            StackReq::try_or(permute_req, factor_req)?,
        ])
    }

    #[inline]
    pub fn factorize_numeric_ldlt<E: ComplexField>(
        &self,
        L_values: SliceGroupMut<'_, E>,
        A: SparseColMatRef<'_, I, E>,
        side: Side,
        regularization: LdltRegularization<'_, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) {
        assert!(A.nrows() == A.ncols());
        let n = A.nrows();

        ghost::with_size(n, |N| {
            let A_nnz = self.A_nnz;
            let A = ghost::SparseColMatRef::new(A, N, N);

            let (new_signs, stack) =
                stack.make_raw::<i8>(if regularization.dynamic_regularization_signs.is_some() {
                    n
                } else {
                    0
                });

            let perm = ghost::PermutationRef::new(self.perm(), N);
            let fwd = perm.fwd_inv().0;
            let signs = regularization.dynamic_regularization_signs.map(|signs| {
                {
                    let new_signs = Array::from_mut(new_signs, N);
                    let signs = Array::from_ref(signs, N);
                    for i in N.indices() {
                        new_signs[i] = signs[fwd[i].zx()];
                    }
                }
                &*new_signs
            });
            let regularization = LdltRegularization {
                dynamic_regularization_signs: signs,
                ..regularization
            };

            let (mut new_values, stack) = crate::make_raw::<E>(A_nnz, stack);
            let (new_col_ptr, stack) = stack.make_raw::<I>(n + 1);
            let (new_row_ind, mut stack) = stack.make_raw::<I>(A_nnz);

            let out_side = match &self.raw {
                SymbolicCholeskyRaw::Simplicial(_) => Side::Upper,
                SymbolicCholeskyRaw::Supernodal(_) => Side::Lower,
            };

            let A = ghost_permute_hermitian(
                new_values.rb_mut(),
                new_col_ptr,
                new_row_ind,
                A,
                perm,
                side,
                out_side,
                stack.rb_mut(),
            );

            match &self.raw {
                SymbolicCholeskyRaw::Simplicial(this) => {
                    simplicial::factorize_simplicial_numeric_ldlt(
                        L_values,
                        *A,
                        regularization,
                        this,
                        stack,
                    );
                }
                SymbolicCholeskyRaw::Supernodal(this) => {
                    supernodal::factorize_supernodal_numeric_ldlt(
                        L_values,
                        *A,
                        regularization,
                        this,
                        parallelism,
                        stack,
                    );
                }
            }
        });
    }

    pub fn dense_solve_in_place_req<E: Entity>(
        &self,
        rhs_ncols: usize,
    ) -> Result<StackReq, SizeOverflow> {
        temp_mat_req::<E>(self.nrows(), rhs_ncols)?.try_and(match self.raw() {
            SymbolicCholeskyRaw::Simplicial(this) => {
                this.dense_solve_in_place_req::<E>(rhs_ncols)?
            }
            SymbolicCholeskyRaw::Supernodal(this) => {
                this.dense_solve_in_place_req::<E>(rhs_ncols)?
            }
        })
    }
}

#[derive(Debug)]
pub struct LdltRef<'a, I, E: Entity> {
    symbolic: &'a SymbolicCholesky<I>,
    values: SliceGroup<'a, E>,
}

impl_copy!(<'a><I><supernodal::SymbolicSupernodeRef<'a, I>>);
impl_copy!(<'a><I, E: Entity><supernodal::SupernodalLdltRef<'a, I, E>>);
impl_copy!(<'a><I, E: Entity><simplicial::SimplicialLdltRef<'a, I, E>>);
impl_copy!(<'a><I, E: Entity><supernodal::SupernodeRef<'a, I, E>>);

impl<'a, I: Index, E: Entity> LdltRef<'a, I, E> {
    #[inline]
    pub fn new(symbolic: &'a SymbolicCholesky<I>, values: SliceGroup<'a, E>) -> Self {
        assert!(symbolic.len_values() == values.len());
        Self { symbolic, values }
    }

    #[inline]
    pub fn symbolic(self) -> &'a SymbolicCholesky<I> {
        self.symbolic
    }

    pub fn dense_solve_in_place_with_conj(
        self,
        rhs: MatMut<'_, E>,
        conj: Conj,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) where
        E: ComplexField,
    {
        let k = rhs.ncols();
        let n = self.symbolic.nrows();

        let mut rhs = rhs;

        let (mut x, stack) = temp_mat_uninit::<E>(n, k, stack);

        let (fwd, inv) = self.symbolic.perm().fwd_inv();
        for j in 0..k {
            for (i, fwd) in fwd.iter().enumerate() {
                x.write(i, j, rhs.read(fwd.zx(), j));
            }
        }

        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(symbolic) => {
                let this = simplicial::SimplicialLdltRef::new(symbolic, self.values);
                this.dense_solve_in_place_with_conj(x.rb_mut(), conj, parallelism, stack);
            }
            SymbolicCholeskyRaw::Supernodal(symbolic) => {
                let this = supernodal::SupernodalLdltRef::new(symbolic, self.values);
                this.dense_solve_in_place_with_conj(x.rb_mut(), conj, parallelism, stack);
            }
        }

        for j in 0..k {
            for (i, inv) in inv.iter().enumerate() {
                rhs.write(i, j, x.read(inv.zx(), j));
            }
        }
    }
}

fn postorder_depth_first_search<'n, I: Index>(
    post: &mut Array<'n, I>,
    root: usize,
    mut start_index: usize,
    stack: &mut Array<'n, I>,
    first_child: &mut Array<'n, MaybeIdx<'n, I>>,
    next_child: &Array<'n, I>,
) -> usize {
    let mut top = 1usize;
    let N = post.len();

    stack[N.check(0)] = I::truncate(root);
    while top != 0 {
        let current_node = stack[N.check(top - 1)].zx();
        let first_child = &mut first_child[N.check(current_node)];
        let current_child = first_child.sx();

        if let Some(current_child) = current_child.idx() {
            stack[N.check(top)] = *current_child.truncate::<I>();
            top += 1;
            *first_child = MaybeIdx::new_index_checked(next_child[current_child], N);
        } else {
            post[N.check(start_index)] = I::truncate(current_node);
            start_index += 1;
            top -= 1;
        }
    }
    start_index
}

/// workspace: I×(3*n)
fn ghost_postorder<'n, I: Index>(
    post: &mut Array<'n, I>,
    etree: &Array<'n, MaybeIdx<'n, I>>,
    stack: PodStack<'_>,
) {
    let N = post.len();
    let n = *N;

    if n == 0 {
        return;
    }

    let (stack_, stack) = stack.make_raw::<I>(n);
    let (first_child, stack) = stack.make_raw::<I>(n);
    let (next_child, _) = stack.make_raw::<I>(n);

    let stack = Array::from_mut(stack_, N);
    let next_child = Array::from_mut(next_child, N);
    let first_child = Array::from_mut(ghost::fill_none(first_child, N), N);

    for j in N.indices().rev() {
        let parent = etree[j];
        let next = &mut next_child[j];

        if let Some(parent) = parent.idx() {
            let first = &mut first_child[parent.zx()];
            *next = **first;
            *first = MaybeIdx::from_index(j.truncate::<I>());
        }
    }

    let mut start_index = 0usize;
    for (root, &parent) in etree.iter().enumerate() {
        if parent.idx().is_none() {
            start_index = postorder_depth_first_search(
                post,
                root,
                start_index,
                stack,
                first_child,
                next_child,
            );
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CholeskySymbolicParams<'a> {
    pub amd_params: Control,
    pub supernodal_flop_ratio_threshold: f64,
    pub supernodal_params: supernodal::CholeskySymbolicSupernodalParams<'a>,
}

impl Default for CholeskySymbolicParams<'_> {
    fn default() -> Self {
        Self {
            supernodal_flop_ratio_threshold: 40.0,
            amd_params: Default::default(),
            supernodal_params: Default::default(),
        }
    }
}

pub fn factorize_symbolic<I: Index>(
    A: SymbolicSparseColMatRef<'_, I>,
    side: Side,
    params: CholeskySymbolicParams<'_>,
) -> Result<SymbolicCholesky<I>, FaerError> {
    let n = A.nrows();
    let A_nnz = A.compute_nnz();

    assert!(A.nrows() == A.ncols());
    ghost::with_size(n, |N| {
        let A = ghost::SymbolicSparseColMatRef::new(A, N, N);

        let req = || -> Result<StackReq, SizeOverflow> {
            let n_req = StackReq::try_new::<I>(n)?;
            let A_req = StackReq::try_and(
                // new_col_ptr
                StackReq::try_new::<I>(n + 1)?,
                // new_row_ind
                StackReq::try_new::<I>(A_nnz)?,
            )?;

            StackReq::try_or(
                amd::order_maybe_unsorted_req::<I>(n, A_nnz)?,
                StackReq::try_all_of([
                    A_req,
                    // permute_symmetric | etree
                    n_req,
                    // col_counts
                    n_req,
                    // ghost_prefactorize_symbolic
                    n_req,
                    // ghost_factorize_*_symbolic
                    StackReq::try_or(
                        supernodal::factorize_supernodal_symbolic_req::<I>(n)?,
                        simplicial::factorize_simplicial_symbolic_req::<I>(n)?,
                    )?,
                ])?,
            )
        };

        let req = req().map_err(nomem)?;
        let mut mem = dyn_stack::GlobalPodBuffer::try_new(req).map_err(nomem)?;
        let mut stack = PodStack::new(&mut mem);

        let mut perm_fwd = try_zeroed(n)?;
        let mut perm_inv = try_zeroed(n)?;
        let flops = amd::order_maybe_unsorted(
            &mut perm_fwd,
            &mut perm_inv,
            *A,
            params.amd_params,
            stack.rb_mut(),
        )?;
        let flops = flops.n_div + flops.n_mult_subs_ldl;
        let perm_ =
            ghost::PermutationRef::new(PermutationRef::new_checked(&perm_fwd, &perm_inv), N);

        let (new_col_ptr, stack) = stack.make_raw::<I>(n + 1);
        let (new_row_ind, mut stack) = stack.make_raw::<I>(A_nnz);
        let A = ghost_permute_hermitian_symbolic(
            new_col_ptr,
            new_row_ind,
            A,
            perm_,
            side,
            Side::Upper,
            stack.rb_mut(),
        );

        let (etree, stack) = stack.make_raw::<I>(n);
        let (col_counts, mut stack) = stack.make_raw::<I>(n);
        let etree = Array::from_mut(etree, N);
        let col_counts = Array::from_mut(col_counts, N);
        let etree = &*ghost_prefactorize_symbolic(etree, col_counts, A, stack.rb_mut());
        let L_nnz = I::sum_nonnegative(col_counts).ok_or(FaerError::IndexOverflow)?;

        let raw = if (flops / L_nnz.zx() as f64) > params.supernodal_flop_ratio_threshold {
            SymbolicCholeskyRaw::Supernodal(supernodal::ghost_factorize_supernodal_symbolic(
                A,
                etree,
                col_counts,
                stack.rb_mut(),
                params.supernodal_params,
            )?)
        } else {
            SymbolicCholeskyRaw::Simplicial(simplicial::ghost_factorize_simplicial_symbolic(
                A,
                etree,
                col_counts,
                stack.rb_mut(),
            )?)
        };

        Ok(SymbolicCholesky {
            raw,
            perm_fwd,
            perm_inv,
            A_nnz,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::{supernodal::SupernodalLdltRef, *};
    use crate::qd::Double;
    use assert2::assert;
    use dyn_stack::GlobalPodBuffer;
    use faer_core::Mat;
    use rand::{Rng, SeedableRng};

    macro_rules! monomorphize_test {
        ($name: ident) => {
            monomorphize_test!($name, i32);
            monomorphize_test!($name, i64);
        };

        ($name: ident, $ty: ident) => {
            paste::paste! {
                #[test]
                fn [<$name _ $ty>]() {
                    $name::<$ty>();
                }
            }
        };
    }

    fn test_counts<I: Index>() {
        let truncate = I::truncate;

        let n = 11;
        let col_ptr = &[0, 3, 6, 10, 13, 16, 21, 24, 29, 31, 37, 43].map(truncate);
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
        .map(truncate);

        let A = SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_ind);
        let zero = truncate(0);
        let mut etree = vec![zero; n];
        let mut col_count = vec![zero; n];
        ghost::with_size(n, |N| {
            let A = ghost::SymbolicSparseColMatRef::new(A, N, N);
            let etree = ghost_prefactorize_symbolic(
                Array::from_mut(&mut etree, N),
                Array::from_mut(&mut col_count, N),
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * n))),
            );

            supernodal::ghost_factorize_supernodal_symbolic(
                A,
                etree,
                Array::from_ref(&col_count, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
                Default::default(),
            )
            .unwrap();
        });
        assert_eq!(etree, [5, 2, 7, 5, 7, 6, 8, 9, 9, 10, NONE].map(truncate));
        assert_eq!(col_count, [3, 3, 4, 3, 3, 4, 4, 3, 3, 2, 1].map(truncate));
    }

    include!("../data.rs");

    fn test_amd<I: Index>() {
        for &(_, (_, col_ptr, row_ind, _)) in ALL {
            let I = I::truncate;
            let n = col_ptr.len() - 1;

            let (amd_perm, amd_perm_inv, _) =
                ::amd::order(n, col_ptr, row_ind, &Default::default()).unwrap();
            let col_ptr = &*col_ptr.iter().copied().map(I).collect::<Vec<_>>();
            let row_ind = &*row_ind.iter().copied().map(I).collect::<Vec<_>>();
            let amd_perm = &*amd_perm.iter().copied().map(I).collect::<Vec<_>>();
            let amd_perm_inv = &*amd_perm_inv.iter().copied().map(I).collect::<Vec<_>>();
            let A = SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_ind);

            let perm = &mut vec![I(0); n];
            let perm_inv = &mut vec![I(0); n];

            crate::amd::order_maybe_unsorted(
                perm,
                perm_inv,
                A,
                Default::default(),
                PodStack::new(&mut GlobalPodBuffer::new(
                    crate::amd::order_maybe_unsorted_req::<I>(n, row_ind.len()).unwrap(),
                )),
            )
            .unwrap();

            assert!(perm == amd_perm);
            assert!(perm_inv == amd_perm_inv);
        }
    }

    fn sparse_to_dense<I: Index, E: ComplexField>(sparse: SparseColMatRef<'_, I, E>) -> Mat<E> {
        let m = sparse.nrows();
        let n = sparse.ncols();

        let mut dense = Mat::<E>::zeros(m, n);

        for j in 0..n {
            for (i, val) in zip(
                sparse.row_indices_of_col(j),
                sparse.values_of_col(j).into_ref_iter(),
            ) {
                dense.write(i, j, val.read());
            }
        }

        dense
    }

    fn reconstruct_from_supernodal<I: Index, E: ComplexField>(
        symbolic: &supernodal::SymbolicSupernodalCholesky<I>,
        L_values: SliceGroup<'_, E>,
    ) -> Mat<E> {
        let ldlt = SupernodalLdltRef::new(symbolic, L_values);
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

        let mut D = Mat::<E>::zeros(n, n);
        D.as_mut().diagonal().clone_from(dense.as_ref().diagonal());
        dense.as_mut().diagonal().fill(E::faer_one());
        &dense * D * dense.adjoint()
    }

    fn reconstruct_from_simplicial<I: Index, E: ComplexField>(
        symbolic: &simplicial::SymbolicSimplicialCholesky<I>,
        L_values: SliceGroup<'_, E>,
    ) -> Mat<E> {
        let n = symbolic.nrows();
        let mut dense = Mat::<E>::zeros(n, n);

        let L = SparseColMatRef::new(
            SymbolicSparseColMatRef::new_checked(
                n,
                n,
                symbolic.col_ptrs(),
                None,
                symbolic.row_indices(),
            ),
            L_values,
        );

        for j in 0..n {
            for (i, val) in zip(L.row_indices_of_col(j), L.values_of_col(j).into_ref_iter()) {
                dense.write(i, j, val.read());
            }
        }

        let mut D = Mat::<E>::zeros(n, n);
        D.as_mut().diagonal().clone_from(dense.as_ref().diagonal());
        dense.as_mut().diagonal().fill(E::faer_one());

        &dense * D * dense.adjoint()
    }

    fn test_supernodal<I: Index>() {
        type E = num_complex::Complex<Double<f64>>;
        let truncate = I::truncate;

        let (_, col_ptr, row_ind, values) = MEDIUM;

        let mut gen = rand::rngs::StdRng::seed_from_u64(0);
        let mut complexify = |e: E| {
            let i = E::faer_one().faer_neg().faer_sqrt();
            if e == E::faer_from_f64(1.0) {
                e.faer_add(i.faer_mul(E::faer_from_f64(gen.gen())))
            } else {
                e
            }
        };

        let n = col_ptr.len() - 1;
        let nnz = values.len();
        let col_ptr = &*col_ptr.iter().copied().map(truncate).collect::<Vec<_>>();
        let row_ind = &*row_ind.iter().copied().map(truncate).collect::<Vec<_>>();
        let values_mat =
            faer_core::Mat::<E>::from_fn(nnz, 1, |i, _| complexify(E::faer_from_f64(values[i])));
        let values = SliceGroup::new(values_mat.col_ref(0));

        let A = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_ind),
            values,
        );
        let zero = truncate(0);
        let mut etree = vec![zero; n];
        let mut col_count = vec![zero; n];
        ghost::with_size(n, |N| {
            let A = ghost::SparseColMatRef::new(A, N, N);
            let etree = ghost_prefactorize_symbolic(
                Array::from_mut(&mut etree, N),
                Array::from_mut(&mut col_count, N),
                A.symbolic(),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * n))),
            );

            let symbolic = supernodal::ghost_factorize_supernodal_symbolic(
                A.symbolic(),
                etree,
                Array::from_ref(&col_count, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
                Default::default(),
            )
            .unwrap();

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values_mat.clone();
            let mut A_lower_row_ind = row_ind.to_vec();
            let A_lower_values = SliceGroupMut::new(A_lower_values.col_mut(0));
            let A_lower = crate::ghost_adjoint(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                A_lower_values,
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            );
            let mut values = faer_core::Mat::<E>::zeros(symbolic.len_values(), 1);
            let mut values = SliceGroupMut::new(values.col_mut(0));

            supernodal::factorize_supernodal_numeric_ldlt(
                values.rb_mut(),
                *A_lower,
                Default::default(),
                &symbolic,
                Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(
                    supernodal::factorize_supernodal_numeric_ldlt_req::<I, E>(
                        &symbolic,
                        Parallelism::None,
                    )
                    .unwrap(),
                )),
            );
            let mut A = sparse_to_dense(*A);
            for j in 0..n {
                for i in j + 1..n {
                    A.write(i, j, A.read(j, i).faer_conj());
                }
            }

            let err = reconstruct_from_supernodal(&symbolic, values.rb()) - A;
            let mut max = <E as ComplexField>::Real::faer_zero();
            for j in 0..n {
                for i in 0..n {
                    let x = err.read(i, j).faer_abs();
                    max = if max > x { max } else { x }
                }
            }
            assert!(max < <E as ComplexField>::Real::faer_from_f64(1e-25));
        });
    }

    fn test_simplicial<I: Index>() {
        type E = num_complex::Complex<Double<f64>>;
        let truncate = I::truncate;

        let (_, col_ptr, row_ind, values) = SMALL;

        let mut gen = rand::rngs::StdRng::seed_from_u64(0);
        let mut complexify = |e: E| {
            let i = E::faer_one().faer_neg().faer_sqrt();
            if e == E::faer_from_f64(1.0) {
                e.faer_add(i.faer_mul(E::faer_from_f64(gen.gen())))
            } else {
                e
            }
        };

        let n = col_ptr.len() - 1;
        let nnz = values.len();
        let col_ptr = &*col_ptr.iter().copied().map(truncate).collect::<Vec<_>>();
        let row_ind = &*row_ind.iter().copied().map(truncate).collect::<Vec<_>>();
        let values_mat =
            faer_core::Mat::<E>::from_fn(nnz, 1, |i, _| complexify(E::faer_from_f64(values[i])));
        let values = SliceGroup::new(values_mat.col_ref(0));

        let A = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_ind),
            values,
        );
        let zero = truncate(0);
        let mut etree = vec![zero; n];
        let mut col_count = vec![zero; n];
        ghost::with_size(n, |N| {
            let A = ghost::SparseColMatRef::new(A, N, N);
            let etree = ghost_prefactorize_symbolic(
                Array::from_mut(&mut etree, N),
                Array::from_mut(&mut col_count, N),
                A.symbolic(),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * n))),
            );

            let symbolic = simplicial::ghost_factorize_simplicial_symbolic(
                A.symbolic(),
                etree,
                Array::from_ref(&col_count, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            )
            .unwrap();

            let mut values = faer_core::Mat::<E>::zeros(symbolic.len_values(), 1);
            let mut values = SliceGroupMut::new(values.col_mut(0));

            simplicial::factorize_simplicial_numeric_ldlt(
                values.rb_mut(),
                *A,
                Default::default(),
                &symbolic,
                PodStack::new(&mut GlobalPodBuffer::new(
                    simplicial::factorize_simplicial_numeric_ldlt_req::<I, E>(n).unwrap(),
                )),
            );
            let mut A = sparse_to_dense(*A);
            for j in 0..n {
                for i in j + 1..n {
                    A.write(i, j, A.read(j, i).faer_conj());
                }
            }

            let err = reconstruct_from_simplicial(&symbolic, values.rb()) - &A;

            let mut max = <E as ComplexField>::Real::faer_zero();
            for j in 0..n {
                for i in 0..n {
                    let x = err.read(i, j).faer_abs();
                    max = if max > x { max } else { x }
                }
            }
            assert!(max < <E as ComplexField>::Real::faer_from_f64(1e-25));
        });
    }

    fn test_solver<I: Index>() {
        type E = num_complex::Complex<Double<f64>>;
        let truncate = I::truncate;

        for (_, col_ptr, row_ind, values) in [SMALL, MEDIUM] {
            let mut gen = rand::rngs::StdRng::seed_from_u64(0);
            let i = E::faer_one().faer_neg().faer_sqrt();
            let mut complexify = |e: E| {
                if e == E::faer_from_f64(1.0) {
                    e.faer_add(i.faer_mul(E::faer_from_f64(gen.gen())))
                } else {
                    e
                }
            };

            let n = col_ptr.len() - 1;
            let nnz = values.len();
            let col_ptr = &*col_ptr.iter().copied().map(truncate).collect::<Vec<_>>();
            let row_ind = &*row_ind.iter().copied().map(truncate).collect::<Vec<_>>();
            let values_mat = faer_core::Mat::<E>::from_fn(nnz, 1, |i, _| {
                complexify(E::faer_from_f64(values[i]))
            });
            let values = SliceGroup::new(values_mat.col_ref(0));

            let A_upper = SparseColMatRef::<'_, I, E>::new(
                SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_ind),
                values,
            );

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values_mat.clone();
            let mut A_lower_row_ind = row_ind.to_vec();
            let A_lower_values = SliceGroupMut::new(A_lower_values.col_mut(0));
            let A_lower = crate::adjoint(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                A_lower_values,
                A_upper,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            );

            let mut A_dense = sparse_to_dense(A_upper);
            for j in 0..n {
                for i in j + 1..n {
                    A_dense.write(i, j, A_dense.read(j, i).faer_conj());
                }
            }

            for (A, side, supernodal_flop_ratio_threshold, parallelism) in [
                (A_upper, Side::Upper, f64::INFINITY, Parallelism::None),
                (A_upper, Side::Upper, 0.0, Parallelism::None),
                (A_lower, Side::Lower, f64::INFINITY, Parallelism::None),
                (A_lower, Side::Lower, 0.0, Parallelism::None),
            ] {
                let symbolic = factorize_symbolic(
                    A.symbolic(),
                    side,
                    CholeskySymbolicParams {
                        supernodal_flop_ratio_threshold,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut mem = GlobalPodBuffer::new(
                    symbolic
                        .factorize_numeric_ldlt_req::<E>(false, parallelism)
                        .unwrap(),
                );
                let mut L_values = Mat::<E>::zeros(symbolic.len_values(), 1);
                let mut L_values = SliceGroupMut::new(L_values.col_mut(0));

                symbolic.factorize_numeric_ldlt(
                    L_values.rb_mut(),
                    A,
                    side,
                    Default::default(),
                    parallelism,
                    PodStack::new(&mut mem),
                );
                let L_values = L_values.rb();
                let A_reconstructed = match symbolic.raw() {
                    SymbolicCholeskyRaw::Simplicial(symbolic) => {
                        reconstruct_from_simplicial(symbolic, L_values)
                    }
                    SymbolicCholeskyRaw::Supernodal(symbolic) => {
                        reconstruct_from_supernodal(symbolic, L_values)
                    }
                };

                let (perm_fwd, _) = symbolic.perm().fwd_inv();

                let mut max = <E as ComplexField>::Real::faer_zero();
                for j in 0..n {
                    for i in 0..n {
                        let x = (A_reconstructed
                            .read(i, j)
                            .faer_sub(A_dense.read(perm_fwd[i].zx(), perm_fwd[j].zx())))
                        .faer_abs();
                        max = if max > x { max } else { x }
                    }
                }
                assert!(max < <E as ComplexField>::Real::faer_from_f64(1e-25));

                for k in (0..16).chain(128..132) {
                    let rhs = Mat::<E>::from_fn(n, k, |_, _| {
                        E::faer_from_f64(gen.gen())
                            .faer_add(i.faer_mul(E::faer_from_f64(gen.gen())))
                    });
                    for conj in [Conj::Yes, Conj::No] {
                        let mut x = rhs.clone();
                        let ldlt = LdltRef::new(&symbolic, L_values);
                        ldlt.dense_solve_in_place_with_conj(
                            x.as_mut(),
                            conj,
                            parallelism,
                            PodStack::new(&mut GlobalPodBuffer::new(
                                symbolic.dense_solve_in_place_req::<E>(k).unwrap(),
                            )),
                        );

                        let rhs_reconstructed = if conj == Conj::No {
                            &A_dense * &x
                        } else {
                            A_dense.conjugate() * &x
                        };
                        let mut max = <E as ComplexField>::Real::faer_zero();
                        for j in 0..k {
                            for i in 0..n {
                                let x = rhs_reconstructed
                                    .read(i, j)
                                    .faer_sub(rhs.read(i, j))
                                    .faer_abs();
                                max = if max > x { max } else { x }
                            }
                        }
                        assert!(max < <E as ComplexField>::Real::faer_from_f64(1e-25));
                    }
                }
            }
        }
    }

    fn test_solver_regularization<I: Index>() {
        type E = f64;
        let I = I::truncate;

        for (_, col_ptr, row_ind, values) in [SMALL, MEDIUM] {
            let n = col_ptr.len() - 1;
            let nnz = values.len();
            let col_ptr = &*col_ptr.iter().copied().map(I).collect::<Vec<_>>();
            let row_ind = &*row_ind.iter().copied().map(I).collect::<Vec<_>>();
            // artificial zeros
            let values_mat = faer_core::Mat::<E>::from_fn(nnz, 1, |_, _| 0.0);
            let dynamic_regularization_epsilon = 1e-6;
            let dynamic_regularization_delta = 1e-2;

            let values = SliceGroup::new(values_mat.col_ref(0));
            let mut signs = vec![-1i8; n];
            signs[..8].fill(1);

            let A_upper = SparseColMatRef::<'_, I, E>::new(
                SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_ind),
                values,
            );

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values_mat.clone();
            let mut A_lower_row_ind = row_ind.to_vec();
            let A_lower_values = SliceGroupMut::new(A_lower_values.col_mut(0));
            let A_lower = crate::adjoint(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                A_lower_values,
                A_upper,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            );

            let mut A_dense = sparse_to_dense(A_upper);
            for (j, &sign) in signs.iter().enumerate() {
                A_dense.write(j, j, sign as f64 * dynamic_regularization_epsilon);
                for i in j + 1..n {
                    A_dense.write(i, j, A_dense.read(j, i).faer_conj());
                }
            }

            for (A, side, supernodal_flop_ratio_threshold, parallelism) in [
                (A_upper, Side::Upper, f64::INFINITY, Parallelism::None),
                (A_upper, Side::Upper, 0.0, Parallelism::None),
                (A_lower, Side::Lower, f64::INFINITY, Parallelism::None),
                (A_lower, Side::Lower, 0.0, Parallelism::None),
            ] {
                let symbolic = factorize_symbolic(
                    A.symbolic(),
                    side,
                    CholeskySymbolicParams {
                        supernodal_flop_ratio_threshold,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut mem = GlobalPodBuffer::new(
                    symbolic
                        .factorize_numeric_ldlt_req::<E>(true, parallelism)
                        .unwrap(),
                );
                let mut L_values = Mat::<E>::zeros(symbolic.len_values(), 1);
                let mut L_values = SliceGroupMut::new(L_values.col_mut(0));

                symbolic.factorize_numeric_ldlt(
                    L_values.rb_mut(),
                    A,
                    side,
                    LdltRegularization {
                        dynamic_regularization_signs: Some(&signs),
                        dynamic_regularization_delta,
                        dynamic_regularization_epsilon,
                    },
                    parallelism,
                    PodStack::new(&mut mem),
                );
                let L_values = L_values.rb();

                let A_reconstructed = match symbolic.raw() {
                    SymbolicCholeskyRaw::Simplicial(symbolic) => {
                        reconstruct_from_simplicial(symbolic, L_values)
                    }
                    SymbolicCholeskyRaw::Supernodal(symbolic) => {
                        reconstruct_from_supernodal(symbolic, L_values)
                    }
                };

                let (perm_fwd, _) = symbolic.perm().fwd_inv();
                let mut max = <E as ComplexField>::Real::faer_zero();
                for j in 0..n {
                    for i in 0..n {
                        let x = (A_reconstructed
                            .read(i, j)
                            .faer_sub(A_dense.read(perm_fwd[i].zx(), perm_fwd[j].zx())))
                        .abs();
                        max = if max > x { max } else { x }
                    }
                }
                assert!(max == 0.0);
            }
        }
    }

    monomorphize_test!(test_amd);
    monomorphize_test!(test_counts);
    monomorphize_test!(test_supernodal, i32);
    monomorphize_test!(test_simplicial, i32);
    monomorphize_test!(test_solver, i32);
    monomorphize_test!(test_solver_regularization, i32);
}
