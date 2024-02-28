//! Computes the Cholesky decomposition (either LLT, LDLT, or Bunch-Kaufman) of a given sparse
//! matrix. See [`crate::linalg::cholesky`] for more info.
//!
//! The entry point in this module is [`SymbolicCholesky`] and [`factorize_symbolic_cholesky`].
//!
//! # Note
//! The functions in this module accept unsorted input, producing a sorted decomposition factor
//! (simplicial).

// implementation inspired by https://gitlab.com/hodge_star/catamari

use super::{
    amd::{self, Control},
    ghost::{self, Array, Idx, MaybeIdx},
    ghost_permute_hermitian_unsorted, ghost_permute_hermitian_unsorted_symbolic, make_raw_req, mem,
    mem::NONE,
    nomem, triangular_solve, try_collect, try_zeroed, windows2, FaerError, Index, PermRef, Side,
    SliceGroup, SliceGroupMut, SparseColMatRef, SupernodalThreshold, SymbolicSparseColMatRef,
    SymbolicSupernodalParams,
};
pub use crate::linalg::cholesky::{
    bunch_kaufman::compute::BunchKaufmanRegularization,
    ldlt_diagonal::compute::LdltRegularization,
    llt::{compute::LltRegularization, CholeskyError},
};
use crate::{
    assert,
    linalg::{temp_mat_req, temp_mat_uninit},
    unzipped, zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism, SignedIndex,
};
use core::{cell::Cell, iter::zip};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_entity::GroupFor;
use reborrow::*;

#[derive(Copy, Clone)]
#[allow(dead_code)]
enum Ordering<'a, I: Index> {
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

/// Simplicial factorization module.
///
/// A simplicial factorization is one that processes the elements of the Cholesky factor of the
/// input matrix one by one, rather than by blocks. This is more efficient if the Cholesky factor is
/// very sparse.
pub mod simplicial {
    use super::*;
    use crate::assert;

    /// Computes the size and alignment of the workspace required to compute the elimination tree
    /// and column counts of a matrix of size `n` with `nnz` non-zero entries.
    pub fn prefactorize_symbolic_cholesky_req<I: Index>(
        n: usize,
        nnz: usize,
    ) -> Result<StackReq, SizeOverflow> {
        _ = nnz;
        StackReq::try_new::<I>(n)
    }

    /// Computes the elimination tree and column counts of the Cholesky factorization of the matrix
    /// `A`.
    ///
    /// # Note
    /// Only the upper triangular part of `A` is analyzed.
    pub fn prefactorize_symbolic_cholesky<'out, I: Index>(
        etree: &'out mut [I::Signed],
        col_counts: &mut [I],
        A: SymbolicSparseColMatRef<'_, I>,
        stack: PodStack<'_>,
    ) -> EliminationTreeRef<'out, I> {
        let n = A.nrows();
        assert!(A.nrows() == A.ncols());
        assert!(etree.len() == n);
        assert!(col_counts.len() == n);

        ghost::with_size(n, |N| {
            ghost_prefactorize_symbolic_cholesky(
                Array::from_mut(etree, N),
                Array::from_mut(col_counts, N),
                ghost::SymbolicSparseColMatRef::new(A, N, N),
                stack,
            );
        });

        simplicial::EliminationTreeRef { inner: etree }
    }

    fn ereach<'n, 'a, I: Index>(
        stack: &'a mut Array<'n, I>,
        A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
        etree: &Array<'n, MaybeIdx<'n, I>>,
        k: Idx<'n, usize>,
        visited: &mut Array<'n, I::Signed>,
    ) -> &'a [Idx<'n, I>] {
        let N = A.ncols();

        // invariant: stack[top..] elements are less than or equal to k
        let mut top = *N;
        let k_: I = *k.truncate();
        visited[k] = k_.to_signed();
        for mut i in A.row_indices_of_col(k) {
            // (1): after this, we know i < k
            if i >= k {
                continue;
            }
            // invariant: stack[..len] elements are less than or equal to k
            let mut len = 0usize;
            loop {
                if visited[i] == k_.to_signed() {
                    break;
                }

                // inserted element is i < k, see (1)
                let pushed: Idx<'n, I> = i.truncate::<I>();
                stack[N.check(len)] = *pushed;
                // len is incremented, maintaining the invariant
                len += 1;

                visited[i] = k_.to_signed();
                i = N.check(etree[i].into_inner().zx());
            }

            // because stack[..len] elements are less than or equal to k
            // stack[top - len..] elements are now less than or equal to k
            stack.as_mut().copy_within(..len, top - len);
            // top is decremented by len, maintaining the invariant
            top -= len;
        }

        let stack = &stack.as_ref()[top..];

        // SAFETY: stack[top..] elements are < k < N
        unsafe { Idx::from_slice_ref_unchecked(stack) }
    }

    /// Computes the size and alignment of the workspace required to compute the symbolic
    /// Cholesky factorization of a square matrix with size `n`.
    pub fn factorize_simplicial_symbolic_req<I: Index>(n: usize) -> Result<StackReq, SizeOverflow> {
        let n_req = StackReq::try_new::<I>(n)?;
        StackReq::try_all_of([n_req, n_req, n_req])
    }

    /// Computes the symbolic structure of the Cholesky factor of the matrix `A`.
    ///
    /// # Note
    /// Only the upper triangular part of `A` is analyzed.
    ///
    /// # Panics
    /// The elimination tree and column counts must be computed by calling
    /// [`prefactorize_symbolic_cholesky`] with the same matrix. Otherwise, the behavior is
    /// unspecified and panics may occur.
    pub fn factorize_simplicial_symbolic<I: Index>(
        A: SymbolicSparseColMatRef<'_, I>,
        etree: EliminationTreeRef<'_, I>,
        col_counts: &[I],
        stack: PodStack<'_>,
    ) -> Result<SymbolicSimplicialCholesky<I>, FaerError> {
        let n = A.nrows();
        assert!(A.nrows() == A.ncols());
        assert!(etree.inner.len() == n);
        assert!(col_counts.len() == n);

        ghost::with_size(n, |N| {
            ghost_factorize_simplicial_symbolic_cholesky(
                ghost::SymbolicSparseColMatRef::new(A, N, N),
                etree.ghost_inner(N),
                Array::from_ref(col_counts, N),
                stack,
            )
        })
    }

    pub(crate) fn ghost_factorize_simplicial_symbolic_cholesky<'n, I: Index>(
        A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
        etree: &Array<'n, MaybeIdx<'n, I>>,
        col_counts: &Array<'n, I>,
        stack: PodStack<'_>,
    ) -> Result<SymbolicSimplicialCholesky<I>, FaerError> {
        let N = A.ncols();
        let n = *N;

        let mut L_col_ptrs = try_zeroed::<I>(n + 1)?;
        for (&count, [p, p_next]) in zip(
            col_counts.as_ref(),
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
                let (visited, _) = stack.make_raw::<I::Signed>(n);

                let ereach_stack = Array::from_mut(ereach_stack, N);
                let visited = Array::from_mut(visited, N);

                mem::fill_none(visited.as_mut());
                let L_row_indices = Array::from_mut(&mut L_row_ind, L_NNZ);
                let L_col_ptrs_start =
                    Array::from_ref(Idx::from_slice_ref_checked(&L_col_ptrs[..n], L_NNZ), N);
                let current_row_index = Array::from_mut(
                    ghost::copy_slice(current_row_index, L_col_ptrs_start.as_ref()),
                    N,
                );

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

                let etree = try_collect(
                    bytemuck::cast_slice::<I::Signed, I>(MaybeIdx::as_slice_ref(etree.as_ref()))
                        .iter()
                        .copied(),
                )?;

                let _ = SymbolicSparseColMatRef::new_unsorted_checked(
                    n,
                    n,
                    &L_col_ptrs,
                    None,
                    &L_row_ind,
                );

                Ok(SymbolicSimplicialCholesky {
                    dimension: n,
                    col_ptrs: L_col_ptrs,
                    row_indices: L_row_ind,
                    etree,
                })
            },
        )
    }

    #[derive(Copy, Clone, Debug)]
    enum FactorizationKind {
        Llt,
        Ldlt,
    }

    fn factorize_simplicial_numeric_with_row_indices<I: Index, E: ComplexField>(
        L_values: GroupFor<E, &mut [E::Unit]>,
        L_row_indices: &mut [I],
        L_col_ptrs: &[I],
        kind: FactorizationKind,

        etree: EliminationTreeRef<'_, I>,
        A: SparseColMatRef<'_, I, E>,
        regularization: LdltRegularization<'_, E>,

        stack: PodStack<'_>,
    ) -> Result<usize, CholeskyError> {
        let n = A.ncols();
        {
            let L_values = SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&L_values)));
            assert!(L_values.rb().len() == L_row_indices.len());
        }
        assert!(L_col_ptrs.len() == n + 1);
        assert!(etree.into_inner().len() == n);
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
                        let eps = regularization.dynamic_regularization_epsilon.faer_abs();
                        let delta = regularization.dynamic_regularization_delta.faer_abs();
                        let has_delta = delta > E::Real::faer_zero();
                        let mut dynamic_regularization_count = 0usize;

                        let (x, stack) = crate::sparse::linalg::make_raw::<E>(n, stack);
                        let (current_row_index, stack) = stack.make_raw::<I>(n);
                        let (ereach_stack, stack) = stack.make_raw::<I>(n);
                        let (visited, _) = stack.make_raw::<I::Signed>(n);

                        let ereach_stack = Array::from_mut(ereach_stack, N);
                        let visited = Array::from_mut(visited, N);
                        let mut x = ghost::ArrayGroupMut::<'_, '_, E>::new(x.into_inner(), N);

                        SliceGroupMut::<'_, E>::new(x.rb_mut().into_slice()).fill_zero();
                        mem::fill_none(visited.as_mut());

                        let mut L_values = ghost::ArrayGroupMut::<'_, '_, E>::new(L_values, L_NNZ);
                        let L_row_indices = Array::from_mut(L_row_indices, L_NNZ);

                        let L_col_ptrs_start = Array::from_ref(
                            Idx::from_slice_ref_checked(&L_col_ptrs[..n], L_NNZ),
                            N,
                        );

                        let current_row_index = Array::from_mut(
                            ghost::copy_slice(current_row_index, L_col_ptrs_start.as_ref()),
                            N,
                        );

                        for k in N.indices() {
                            let reach = ereach(ereach_stack, *A, etree, k, visited);

                            for (i, aik) in zip(
                                A.row_indices_of_col(k),
                                SliceGroup::<'_, E>::new(A.values_of_col(k)).into_ref_iter(),
                            ) {
                                x.write(i, x.read(i).faer_add(aik.read().faer_conj()));
                            }

                            let mut d = x.read(k).faer_real();
                            x.write(k, E::faer_zero());

                            for &j in reach {
                                let j = j.zx();

                                let j_start = L_col_ptrs_start[j].zx();
                                let cj = &mut current_row_index[j];
                                let row_idx = L_NNZ.check(*cj.zx() + 1);
                                *cj = row_idx.truncate();

                                let mut xj = x.read(j);
                                x.write(j, E::faer_zero());

                                let dj = L_values.read(j_start).faer_real();
                                let lkj = xj.faer_scale_real(dj.faer_inv());
                                if matches!(kind, FactorizationKind::Llt) {
                                    xj = lkj;
                                }

                                let range = j_start.next()..row_idx.to_inclusive();
                                for (i, lij) in zip(
                                    &L_row_indices[range.clone()],
                                    SliceGroup::<'_, E>::new(L_values.rb().subslice(range))
                                        .into_ref_iter(),
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

                            if has_delta {
                                match kind {
                                    FactorizationKind::Llt => {
                                        if d <= eps {
                                            d = delta;
                                            dynamic_regularization_count += 1;
                                        }
                                    }
                                    FactorizationKind::Ldlt => {
                                        if let Some(signs) =
                                            regularization.dynamic_regularization_signs
                                        {
                                            if signs[*k] > 0 && d <= eps {
                                                d = delta;
                                                dynamic_regularization_count += 1;
                                            } else if signs[*k] < 0 && d >= eps.faer_neg() {
                                                d = delta.faer_neg();
                                                dynamic_regularization_count += 1;
                                            }
                                        } else if d.faer_abs() <= eps {
                                            if d < E::Real::faer_zero() {
                                                d = delta.faer_neg();
                                                dynamic_regularization_count += 1;
                                            } else {
                                                d = delta;
                                                dynamic_regularization_count += 1;
                                            }
                                        }
                                    }
                                }
                            }

                            match kind {
                                FactorizationKind::Llt => {
                                    if d <= E::Real::faer_zero() {
                                        return Err(CholeskyError {
                                            non_positive_definite_minor: *k + 1,
                                        });
                                    }
                                    L_values.write(k_start, E::faer_from_real(d.faer_sqrt()));
                                }
                                FactorizationKind::Ldlt => {
                                    L_values.write(k_start, E::faer_from_real(d));
                                }
                            }
                        }
                        Ok(dynamic_regularization_count)
                    },
                )
            },
        )
    }

    fn factorize_simplicial_numeric<I: Index, E: ComplexField>(
        L_values: GroupFor<E, &mut [E::Unit]>,
        kind: FactorizationKind,
        A: SparseColMatRef<'_, I, E>,
        regularization: LdltRegularization<'_, E>,
        symbolic: &SymbolicSimplicialCholesky<I>,
        stack: PodStack<'_>,
    ) -> Result<usize, CholeskyError> {
        let n = A.ncols();
        let L_row_indices = &*symbolic.row_indices;
        let L_col_ptrs = &*symbolic.col_ptrs;
        let etree = &*symbolic.etree;

        {
            let L_values = SliceGroup::<'_, E>::new(E::faer_rb(E::faer_as_ref(&L_values)));
            assert!(L_values.rb().len() == L_row_indices.len());
        }
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
                        let etree = Array::from_ref(
                            MaybeIdx::from_slice_ref_checked(
                                bytemuck::cast_slice::<I, I::Signed>(etree),
                                N,
                            ),
                            N,
                        );
                        let A = ghost::SparseColMatRef::new(A, N, N);

                        let eps = regularization.dynamic_regularization_epsilon.faer_abs();
                        let delta = regularization.dynamic_regularization_delta.faer_abs();
                        let has_delta = delta > E::Real::faer_zero();
                        let mut dynamic_regularization_count = 0usize;

                        let (x, stack) = crate::sparse::linalg::make_raw::<E>(n, stack);
                        let (current_row_index, stack) = stack.make_raw::<I>(n);
                        let (ereach_stack, stack) = stack.make_raw::<I>(n);
                        let (visited, _) = stack.make_raw::<I::Signed>(n);

                        let ereach_stack = Array::from_mut(ereach_stack, N);
                        let visited = Array::from_mut(visited, N);
                        let mut x = ghost::ArrayGroupMut::<'_, '_, E>::new(x.into_inner(), N);

                        SliceGroupMut::<'_, E>::new(x.rb_mut().into_slice()).fill_zero();
                        mem::fill_none(visited.as_mut());

                        let mut L_values = ghost::ArrayGroupMut::<'_, '_, E>::new(L_values, L_NNZ);
                        let L_row_indices = Array::from_ref(L_row_indices, L_NNZ);

                        let L_col_ptrs_start = Array::from_ref(
                            Idx::from_slice_ref_checked(&L_col_ptrs[..n], L_NNZ),
                            N,
                        );

                        let current_row_index = Array::from_mut(
                            ghost::copy_slice(current_row_index, L_col_ptrs_start.as_ref()),
                            N,
                        );

                        for k in N.indices() {
                            let reach = ereach(ereach_stack, *A, etree, k, visited);

                            for (i, aik) in zip(
                                A.row_indices_of_col(k),
                                SliceGroup::<'_, E>::new(A.values_of_col(k)).into_ref_iter(),
                            ) {
                                x.write(i, x.read(i).faer_add(aik.read().faer_conj()));
                            }

                            let mut d = x.read(k).faer_real();
                            x.write(k, E::faer_zero());

                            for &j in reach {
                                let j = j.zx();

                                let j_start = L_col_ptrs_start[j].zx();
                                let cj = &mut current_row_index[j];
                                let row_idx = L_NNZ.check(*cj.zx() + 1);
                                *cj = row_idx.truncate();

                                let mut xj = x.read(j);
                                x.write(j, E::faer_zero());

                                let dj = L_values.read(j_start).faer_real();
                                let lkj = xj.faer_scale_real(dj.faer_inv());
                                if matches!(kind, FactorizationKind::Llt) {
                                    xj = lkj;
                                }

                                let range = j_start.next()..row_idx.to_inclusive();
                                for (i, lij) in zip(
                                    &L_row_indices[range.clone()],
                                    SliceGroup::<'_, E>::new(L_values.rb().subslice(range))
                                        .into_ref_iter(),
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

                            if has_delta {
                                match kind {
                                    FactorizationKind::Llt => {
                                        if d <= eps {
                                            d = delta;
                                            dynamic_regularization_count += 1;
                                        }
                                    }
                                    FactorizationKind::Ldlt => {
                                        if let Some(signs) =
                                            regularization.dynamic_regularization_signs
                                        {
                                            if signs[*k] > 0 && d <= eps {
                                                d = delta;
                                                dynamic_regularization_count += 1;
                                            } else if signs[*k] < 0 && d >= eps.faer_neg() {
                                                d = delta.faer_neg();
                                                dynamic_regularization_count += 1;
                                            }
                                        } else if d.faer_abs() <= eps {
                                            if d < E::Real::faer_zero() {
                                                d = delta.faer_neg();
                                                dynamic_regularization_count += 1;
                                            } else {
                                                d = delta;
                                                dynamic_regularization_count += 1;
                                            }
                                        }
                                    }
                                }
                            }

                            match kind {
                                FactorizationKind::Llt => {
                                    if d <= E::Real::faer_zero() {
                                        return Err(CholeskyError {
                                            non_positive_definite_minor: *k + 1,
                                        });
                                    }
                                    L_values.write(k_start, E::faer_from_real(d.faer_sqrt()));
                                }
                                FactorizationKind::Ldlt => {
                                    L_values.write(k_start, E::faer_from_real(d));
                                }
                            }
                        }
                        Ok(dynamic_regularization_count)
                    },
                )
            },
        )
    }

    /// Computes the numeric values of the Cholesky LLT factor of the matrix `A`, and stores them in
    /// `L_values`.
    ///
    /// # Note
    /// Only the upper triangular part of `A` is accessed.
    ///
    /// # Panics
    /// The symbolic structure must be computed by calling
    /// [`factorize_simplicial_symbolic`] on a matrix with the same symbolic structure.
    /// Otherwise, the behavior is unspecified and panics may occur.
    pub fn factorize_simplicial_numeric_llt<I: Index, E: ComplexField>(
        L_values: GroupFor<E, &mut [E::Unit]>,
        A: SparseColMatRef<'_, I, E>,
        regularization: LltRegularization<E>,
        symbolic: &SymbolicSimplicialCholesky<I>,
        stack: PodStack<'_>,
    ) -> Result<usize, CholeskyError> {
        factorize_simplicial_numeric(
            L_values,
            FactorizationKind::Llt,
            A,
            LdltRegularization {
                dynamic_regularization_signs: None,
                dynamic_regularization_delta: regularization.dynamic_regularization_delta,
                dynamic_regularization_epsilon: regularization.dynamic_regularization_epsilon,
            },
            symbolic,
            stack,
        )
    }

    /// Computes the row indices and  numeric values of the Cholesky LLT factor of the matrix `A`,
    /// and stores them in `L_row_indices` and `L_values`.
    ///
    /// # Note
    /// Only the upper triangular part of `A` is accessed.
    ///
    /// # Panics
    /// The elimination tree and column counts must be computed by calling
    /// [`prefactorize_symbolic_cholesky`] with the same matrix, then the column pointers are
    /// computed from a prefix sum of the column counts. Otherwise, the behavior is unspecified
    /// and panics may occur.
    pub fn factorize_simplicial_numeric_llt_with_row_indices<I: Index, E: ComplexField>(
        L_values: GroupFor<E, &mut [E::Unit]>,
        L_row_indices: &mut [I],
        L_col_ptrs: &[I],

        etree: EliminationTreeRef<'_, I>,
        A: SparseColMatRef<'_, I, E>,
        regularization: LltRegularization<E>,

        stack: PodStack<'_>,
    ) -> Result<usize, CholeskyError> {
        factorize_simplicial_numeric_with_row_indices(
            L_values,
            L_row_indices,
            L_col_ptrs,
            FactorizationKind::Ldlt,
            etree,
            A,
            LdltRegularization {
                dynamic_regularization_signs: None,
                dynamic_regularization_delta: regularization.dynamic_regularization_delta,
                dynamic_regularization_epsilon: regularization.dynamic_regularization_epsilon,
            },
            stack,
        )
    }

    /// Computes the numeric values of the Cholesky LDLT factors of the matrix `A`, and stores them
    /// in `L_values`.
    ///
    /// # Note
    /// Only the upper triangular part of `A` is accessed.
    ///
    /// # Panics
    /// The symbolic structure must be computed by calling
    /// [`factorize_simplicial_symbolic`] on a matrix with the same symbolic structure.
    /// Otherwise, the behavior is unspecified and panics may occur.
    pub fn factorize_simplicial_numeric_ldlt<I: Index, E: ComplexField>(
        L_values: GroupFor<E, &mut [E::Unit]>,
        A: SparseColMatRef<'_, I, E>,
        regularization: LdltRegularization<'_, E>,
        symbolic: &SymbolicSimplicialCholesky<I>,
        stack: PodStack<'_>,
    ) -> usize {
        factorize_simplicial_numeric(
            L_values,
            FactorizationKind::Ldlt,
            A,
            regularization,
            symbolic,
            stack,
        )
        .unwrap()
    }

    /// Computes the row indices and  numeric values of the Cholesky LDLT factor of the matrix `A`,
    /// and stores them in `L_row_indices` and `L_values`.
    ///
    /// # Note
    /// Only the upper triangular part of `A` is accessed.
    ///
    /// # Panics
    /// The elimination tree and column counts must be computed by calling
    /// [`prefactorize_symbolic_cholesky`] with the same matrix, then the column pointers are
    /// computed from a prefix sum of the column counts. Otherwise, the behavior is unspecified
    /// and panics may occur.
    pub fn factorize_simplicial_numeric_ldlt_with_row_indices<I: Index, E: ComplexField>(
        L_values: GroupFor<E, &mut [E::Unit]>,
        L_row_indices: &mut [I],
        L_col_ptrs: &[I],

        etree: EliminationTreeRef<'_, I>,
        A: SparseColMatRef<'_, I, E>,
        regularization: LdltRegularization<'_, E>,

        stack: PodStack<'_>,
    ) -> usize {
        factorize_simplicial_numeric_with_row_indices(
            L_values,
            L_row_indices,
            L_col_ptrs,
            FactorizationKind::Ldlt,
            etree,
            A,
            regularization,
            stack,
        )
        .unwrap()
    }

    impl<'a, I: Index, E: Entity> SimplicialLltRef<'a, I, E> {
        /// Creates a new Cholesky LLT factor from the symbolic part and numerical values.
        ///
        /// # Panics
        /// Panics if `values.len() != symbolic.len_values()`>
        #[inline]
        pub fn new(
            symbolic: &'a SymbolicSimplicialCholesky<I>,
            values: GroupFor<E, &'a [E::Unit]>,
        ) -> Self {
            let values = SliceGroup::new(values);
            assert!(values.len() == symbolic.len_values());
            Self { symbolic, values }
        }

        /// Returns the symbolic part of the Cholesky factor.
        #[inline]
        pub fn symbolic(self) -> &'a SymbolicSimplicialCholesky<I> {
            self.symbolic
        }

        /// Returns the numerical values of the Cholesky LLT factor.
        #[inline]
        pub fn values(self) -> GroupFor<E, &'a [E::Unit]> {
            self.values.into_inner()
        }

        /// Solves the equation `Op(A) x = rhs` and stores the result in `rhs`, where `Op` is either
        /// the identity or the conjugate, depending on the value of `conj`.
        ///
        /// # Panics
        /// Panics if `rhs.nrows() != self.symbolic().nrows()`.
        pub fn solve_in_place_with_conj(
            &self,
            conj: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) where
            E: ComplexField,
        {
            let _ = parallelism;
            let _ = stack;
            let n = self.symbolic().nrows();
            assert!(rhs.nrows() == n);
            let l = SparseColMatRef::<'_, I, E>::new(self.symbolic().factor(), self.values());

            let mut rhs = rhs;
            triangular_solve::solve_lower_triangular_in_place(l, conj, rhs.rb_mut(), parallelism);
            triangular_solve::solve_lower_triangular_transpose_in_place(
                l,
                conj.compose(Conj::Yes),
                rhs.rb_mut(),
                parallelism,
            );
        }
    }

    impl<'a, I: Index, E: Entity> SimplicialLdltRef<'a, I, E> {
        /// Creates a new Cholesky LDLT factor from the symbolic part and numerical values.
        ///
        /// # Panics
        /// Panics if `values.len() != symbolic.len_values()`>
        #[inline]
        pub fn new(
            symbolic: &'a SymbolicSimplicialCholesky<I>,
            values: GroupFor<E, &'a [E::Unit]>,
        ) -> Self {
            let values = SliceGroup::new(values);
            assert!(values.len() == symbolic.len_values());
            Self { symbolic, values }
        }

        /// Returns the symbolic part of the Cholesky factor.
        #[inline]
        pub fn symbolic(self) -> &'a SymbolicSimplicialCholesky<I> {
            self.symbolic
        }

        /// Returns the numerical values of the Cholesky LDLT factor.
        #[inline]
        pub fn values(self) -> GroupFor<E, &'a [E::Unit]> {
            self.values.into_inner()
        }

        /// Solves the equation `Op(A)×x = rhs` and stores the result in `rhs`, where `Op` is either
        /// the identity or the conjugate, depending on the value of `conj`.
        ///
        /// # Panics
        /// Panics if `rhs.nrows() != self.symbolic().nrows()`.
        pub fn solve_in_place_with_conj(
            &self,
            conj: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) where
            E: ComplexField,
        {
            let _ = parallelism;
            let _ = stack;
            let n = self.symbolic().nrows();
            let ld = SparseColMatRef::<'_, I, E>::new(self.symbolic().factor(), self.values());
            assert!(rhs.nrows() == n);

            let slice_group = SliceGroup::<'_, E>::new;

            let mut x = rhs;
            triangular_solve::solve_unit_lower_triangular_in_place(
                ld,
                conj,
                x.rb_mut(),
                parallelism,
            );
            for mut x in x.rb_mut().col_chunks_mut(1) {
                for j in 0..n {
                    let d_inv = slice_group(ld.values_of_col(j))
                        .read(0)
                        .faer_real()
                        .faer_inv();
                    x.write(j, 0, x.read(j, 0).faer_scale_real(d_inv));
                }
            }
            triangular_solve::solve_unit_lower_triangular_transpose_in_place(
                ld,
                conj.compose(Conj::Yes),
                x.rb_mut(),
                parallelism,
            );
        }
    }

    impl<I: Index> SymbolicSimplicialCholesky<I> {
        /// Returns the number of rows of the Cholesky factor.
        #[inline]
        pub fn nrows(&self) -> usize {
            self.dimension
        }
        /// Returns the number of columns of the Cholesky factor.
        #[inline]
        pub fn ncols(&self) -> usize {
            self.nrows()
        }

        /// Returns the length of the slice that can be used to contain the numerical values of the
        /// Cholesky factor.
        #[inline]
        pub fn len_values(&self) -> usize {
            self.row_indices.len()
        }

        /// Returns the column pointers of the Cholesky factor.
        #[inline]
        pub fn col_ptrs(&self) -> &[I] {
            &self.col_ptrs
        }

        /// Returns the row indices of the Cholesky factor.
        #[inline]
        pub fn row_indices(&self) -> &[I] {
            &self.row_indices
        }

        /// Returns the Cholesky factor's symbolic structure.
        #[inline]
        pub fn factor(&self) -> SymbolicSparseColMatRef<'_, I> {
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

        /// Returns the size and alignment of the workspace required to solve the system `A×x =
        /// rhs`.
        pub fn solve_in_place_req<E: Entity>(
            &self,
            rhs_ncols: usize,
        ) -> Result<StackReq, SizeOverflow> {
            let _ = rhs_ncols;
            Ok(StackReq::empty())
        }
    }

    /// Returns the size and alignment of the workspace required to compute the numeric
    /// Cholesky LDLT factorization of a matrix `A` with dimension `n`.
    pub fn factorize_simplicial_numeric_ldlt_req<I: Index, E: Entity>(
        n: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n_req = StackReq::try_new::<I>(n)?;
        StackReq::try_all_of([make_raw_req::<E>(n)?, n_req, n_req, n_req])
    }

    /// Returns the size and alignment of the workspace required to compute the numeric
    /// Cholesky LLT factorization of a matrix `A` with dimension `n`.
    pub fn factorize_simplicial_numeric_llt_req<I: Index, E: Entity>(
        n: usize,
    ) -> Result<StackReq, SizeOverflow> {
        factorize_simplicial_numeric_ldlt_req::<I, E>(n)
    }

    /// Cholesky LLT factor containing both its symbolic and numeric representations.
    #[derive(Debug)]
    pub struct SimplicialLltRef<'a, I: Index, E: Entity> {
        symbolic: &'a SymbolicSimplicialCholesky<I>,
        values: SliceGroup<'a, E>,
    }

    /// Cholesky LDLT factors containing both the symbolic and numeric representations.
    #[derive(Debug)]
    pub struct SimplicialLdltRef<'a, I: Index, E: Entity> {
        symbolic: &'a SymbolicSimplicialCholesky<I>,
        values: SliceGroup<'a, E>,
    }

    /// Cholesky factor structure containing its symbolic structure.
    #[derive(Debug)]
    pub struct SymbolicSimplicialCholesky<I: Index> {
        dimension: usize,
        col_ptrs: alloc::vec::Vec<I>,
        row_indices: alloc::vec::Vec<I>,
        etree: alloc::vec::Vec<I>,
    }

    /// Reference to a slice containing the Cholesky factor's elimination tree.
    ///
    /// The elimination tree (or elimination forest, in the general case) is a structure
    /// representing the relationship between the columns of the Cholesky factor, and the way
    /// how earlier columns contribute their sparsity pattern to later columns of the factor.
    #[derive(Copy, Clone, Debug)]
    pub struct EliminationTreeRef<'a, I: Index> {
        pub(crate) inner: &'a [I::Signed],
    }

    impl<'a, I: Index> EliminationTreeRef<'a, I> {
        /// Returns the raw elimination tree.
        ///
        /// A value can be either nonnegative to represent the index of the parent of a given node,
        /// or `-1` to signify that it has no parent.
        #[inline]
        pub fn into_inner(self) -> &'a [I::Signed] {
            self.inner
        }

        /// Creates an elimination tree reference from the underlying array.
        ///
        /// # Safety
        /// The elimination tree must come from an array that was previously filled with
        /// [`prefactorize_symbolic_cholesky`].
        #[inline]
        pub unsafe fn from_inner(inner: &'a [I::Signed]) -> Self {
            Self { inner }
        }

        #[inline]
        #[track_caller]
        pub(crate) fn ghost_inner<'n>(self, N: ghost::Size<'n>) -> &'a Array<'n, MaybeIdx<'n, I>> {
            assert!(self.inner.len() == *N);
            unsafe { Array::from_ref(MaybeIdx::from_slice_ref_unchecked(self.inner), N) }
        }
    }
}

/// Supernodal factorization module.
///
/// A supernodal factorization is one that processes the elements of the Cholesky factor of the
/// input matrix by blocks, rather one by one. This is more efficient if the Cholesky factor is
/// somewhat dense.
pub mod supernodal {
    use super::*;
    use crate::{assert, debug_assert};

    fn ereach_super<'n, 'nsuper, I: Index>(
        A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
        super_etree: &Array<'nsuper, MaybeIdx<'nsuper, I>>,
        index_to_super: &Array<'n, Idx<'nsuper, I>>,
        current_row_positions: &mut Array<'nsuper, I>,
        row_indices: &mut [Idx<'n, I>],
        k: Idx<'n, usize>,
        visited: &mut Array<'nsuper, I::Signed>,
    ) {
        let k_: I = *k.truncate();
        visited[index_to_super[k].zx()] = k_.to_signed();
        for i in A.row_indices_of_col(k) {
            if i >= k {
                continue;
            }
            let mut supernode_i = index_to_super[i].zx();
            loop {
                if visited[supernode_i] == k_.to_signed() {
                    break;
                }

                row_indices[current_row_positions[supernode_i].zx()] = k.truncate();
                current_row_positions[supernode_i] += I::truncate(1);

                visited[supernode_i] = k_.to_signed();
                supernode_i = super_etree[supernode_i].sx().idx().unwrap();
            }
        }
    }

    fn ereach_super_ata<'m, 'n, 'nsuper, I: Index>(
        A: ghost::SymbolicSparseColMatRef<'m, 'n, '_, I>,
        perm: Option<ghost::PermRef<'n, '_, I>>,
        min_col: &Array<'m, MaybeIdx<'n, I>>,
        super_etree: &Array<'nsuper, MaybeIdx<'nsuper, I>>,
        index_to_super: &Array<'n, Idx<'nsuper, I>>,
        current_row_positions: &mut Array<'nsuper, I>,
        row_indices: &mut [Idx<'n, I>],
        k: Idx<'n, usize>,
        visited: &mut Array<'nsuper, I::Signed>,
    ) {
        let k_: I = *k.truncate();
        visited[index_to_super[k].zx()] = k_.to_signed();

        let fwd = perm.map(|perm| perm.arrays().0);
        let fwd = |i: Idx<'n, usize>| fwd.map(|fwd| fwd[k].zx()).unwrap_or(i);
        for i in A.row_indices_of_col(fwd(k)) {
            let Some(i) = min_col[i].idx() else { continue };
            let i = i.zx();

            if i >= k {
                continue;
            }
            let mut supernode_i = index_to_super[i].zx();
            loop {
                if visited[supernode_i] == k_.to_signed() {
                    break;
                }

                row_indices[current_row_positions[supernode_i].zx()] = k.truncate();
                current_row_positions[supernode_i] += I::truncate(1);

                visited[supernode_i] = k_.to_signed();
                supernode_i = super_etree[supernode_i].sx().idx().unwrap();
            }
        }
    }

    /// Symbolic structure of a single supernode from the Cholesky factor.
    #[derive(Debug)]
    pub struct SymbolicSupernodeRef<'a, I: Index> {
        start: usize,
        pattern: &'a [I],
    }

    impl<'a, I: Index> SymbolicSupernodeRef<'a, I> {
        /// Returns the starting index of the supernode.
        #[inline]
        pub fn start(self) -> usize {
            self.start
        }

        /// Returns the pattern of the row indices in the supernode, excluding those on the block
        /// diagonal.
        pub fn pattern(self) -> &'a [I] {
            self.pattern
        }
    }

    impl<'a, I: Index, E: Entity> SupernodeRef<'a, I, E> {
        /// Returns the starting index of the supernode.
        #[inline]
        pub fn start(self) -> usize {
            self.symbolic.start
        }

        /// Returns the pattern of the row indices in the supernode, excluding those on the block
        /// diagonal.
        pub fn pattern(self) -> &'a [I] {
            self.symbolic.pattern
        }

        /// Returns a view over the numerical values of the supernode.
        pub fn matrix(self) -> MatRef<'a, E> {
            self.matrix
        }
    }

    /// A single supernode from the Cholesky factor.
    #[derive(Debug)]
    pub struct SupernodeRef<'a, I: Index, E: Entity> {
        matrix: MatRef<'a, E>,
        symbolic: SymbolicSupernodeRef<'a, I>,
    }

    impl<'a, I: Index, E: Entity> SupernodalIntranodeBunchKaufmanRef<'a, I, E> {
        /// Creates a new Cholesky intranodal Bunch-Kaufman factor from the symbolic part and
        /// numerical values, as well as the pivoting permutation.
        ///
        /// # Panics
        /// - Panics if `values.len() != symbolic.len_values()`.
        /// - Panics if `subdiag.len() != symbolic.nrows()`.
        /// - Panics if `perm.len() != symbolic.nrows()`.
        #[inline]
        pub fn new(
            symbolic: &'a SymbolicSupernodalCholesky<I>,
            values: GroupFor<E, &'a [E::Unit]>,
            subdiag: GroupFor<E, &'a [E::Unit]>,
            perm: PermRef<'a, I>,
        ) -> Self {
            let values = SliceGroup::<'_, E>::new(values);
            let subdiag = SliceGroup::<'_, E>::new(subdiag);
            assert!(all(
                values.len() == symbolic.len_values(),
                subdiag.len() == symbolic.nrows(),
                perm.len() == symbolic.nrows(),
            ));
            Self {
                symbolic,
                values,
                subdiag,
                perm,
            }
        }

        /// Returns the symbolic part of the Cholesky factor.
        #[inline]
        pub fn symbolic(self) -> &'a SymbolicSupernodalCholesky<I> {
            self.symbolic
        }

        /// Returns the numerical values of the L factor.
        #[inline]
        pub fn values(self) -> SliceGroup<'a, E> {
            self.values
        }

        /// Returns the `s`'th supernode.
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

            let Ls = crate::mat::from_column_major_slice::<'_, E>(
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

        /// Solves the system $\text{Op}(L B L^H) x = \text{rhs}$, where `Op` is either the identity
        /// or the conjugate depending on the value of `conj`.
        ///
        /// # Note
        /// Note that this function doesn't apply the pivoting permutation. Users are expected to
        /// apply it manually to `rhs` before and after calling this function.
        ///
        /// # Panics
        /// Panics if `rhs.nrows() != self.symbolic().nrows()`.
        pub fn solve_in_place_no_numeric_permute_with_conj(
            self,
            conj: Conj,
            rhs: MatMut<'_, E>,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) where
            E: ComplexField,
        {
            let symbolic = self.symbolic();
            let n = symbolic.nrows();
            assert!(rhs.nrows() == n);
            let mut stack = stack;

            let mut x = rhs;

            let k = x.ncols();
            for s in 0..symbolic.n_supernodes() {
                let s = self.supernode(s);
                let size = s.matrix.ncols();
                let Ls = s.matrix;
                let (Ls_top, Ls_bot) = Ls.split_at_row(size);
                let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
                crate::linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(
                    Ls_top,
                    conj,
                    x_top.rb_mut(),
                    parallelism,
                );

                let (mut tmp, _) = temp_mat_uninit::<E>(s.pattern().len(), k, stack.rb_mut());
                crate::linalg::matmul::matmul_with_conj(
                    tmp.rb_mut(),
                    Ls_bot,
                    conj,
                    x_top.rb(),
                    Conj::No,
                    None,
                    E::faer_one(),
                    parallelism,
                );

                let inv = self.perm.arrays().1;
                for j in 0..k {
                    for (idx, i) in s.pattern().iter().enumerate() {
                        let i = i.zx();
                        let i = inv[i].zx();
                        x.write(i, j, x.read(i, j).faer_sub(tmp.read(idx, j)))
                    }
                }
            }
            for s in 0..symbolic.n_supernodes() {
                let s = self.supernode(s);
                let size = s.matrix.ncols();
                let Bs = s.matrix();
                let subdiag = self.subdiag.subslice(s.start()..s.start() + size);

                let mut idx = 0;
                while idx < size {
                    let subdiag = subdiag.read(idx);
                    let i = idx + s.start();
                    if subdiag == E::faer_zero() {
                        let d = Bs.read(idx, idx).faer_real();
                        for j in 0..k {
                            x.write(i, j, x.read(i, j).faer_scale_real(d))
                        }
                        idx += 1;
                    } else {
                        let d11 = Bs.read(idx, idx).faer_real();
                        let d22 = Bs.read(idx + 1, idx + 1).faer_real();
                        let d21 = subdiag;

                        if conj == Conj::Yes {
                            for j in 0..k {
                                let xi = x.read(i, j);
                                let xip1 = x.read(i + 1, j);

                                x.write(i, j, xi.faer_scale_real(d11).faer_add(xip1.faer_mul(d21)));
                                x.write(
                                    i + 1,
                                    j,
                                    xip1.faer_scale_real(d22)
                                        .faer_add(xi.faer_mul(d21.faer_conj())),
                                );
                            }
                        } else {
                            for j in 0..k {
                                let xi = x.read(i, j);
                                let xip1 = x.read(i + 1, j);

                                x.write(
                                    i,
                                    j,
                                    xi.faer_scale_real(d11)
                                        .faer_add(xip1.faer_mul(d21.faer_conj())),
                                );
                                x.write(
                                    i + 1,
                                    j,
                                    xip1.faer_scale_real(d22).faer_add(xi.faer_mul(d21)),
                                );
                            }
                        }
                        idx += 2;
                    }
                }
            }
            for s in (0..symbolic.n_supernodes()).rev() {
                let s = self.supernode(s);
                let size = s.matrix.ncols();
                let Ls = s.matrix;
                let (Ls_top, Ls_bot) = Ls.split_at_row(size);

                let (mut tmp, _) = temp_mat_uninit::<E>(s.pattern().len(), k, stack.rb_mut());
                let inv = self.perm.arrays().1;
                for j in 0..k {
                    for (idx, i) in s.pattern().iter().enumerate() {
                        let i = i.zx();
                        let i = inv[i].zx();
                        tmp.write(idx, j, x.read(i, j));
                    }
                }

                let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
                crate::linalg::matmul::matmul_with_conj(
                    x_top.rb_mut(),
                    Ls_bot.transpose(),
                    conj.compose(Conj::Yes),
                    tmp.rb(),
                    Conj::No,
                    Some(E::faer_one()),
                    E::faer_one().faer_neg(),
                    parallelism,
                );
                crate::linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(
                    Ls_top.transpose(),
                    conj.compose(Conj::Yes),
                    x_top.rb_mut(),
                    parallelism,
                );
            }
        }
    }

    impl<'a, I: Index, E: Entity> SupernodalLdltRef<'a, I, E> {
        /// Creates new Cholesky LDLT factors from the symbolic part and
        /// numerical values.
        ///
        /// # Panics
        /// - Panics if `values.len() != symbolic.len_values()`.
        #[inline]
        pub fn new(
            symbolic: &'a SymbolicSupernodalCholesky<I>,
            values: GroupFor<E, &'a [E::Unit]>,
        ) -> Self {
            let values = SliceGroup::new(values);
            assert!(values.len() == symbolic.len_values());
            Self { symbolic, values }
        }

        /// Returns the symbolic part of the Cholesky factor.
        #[inline]
        pub fn symbolic(self) -> &'a SymbolicSupernodalCholesky<I> {
            self.symbolic
        }

        /// Returns the numerical values of the L factor.
        #[inline]
        pub fn values(self) -> SliceGroup<'a, E> {
            self.values
        }

        /// Returns the `s`'th supernode.
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

            let Ls = crate::mat::from_column_major_slice::<'_, E>(
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

        /// Solves the equation `Op(A) x = rhs` and stores the result in `rhs`, where `Op` is either
        /// the identity or the conjugate, depending on the value of `conj`.
        ///
        /// # Panics
        /// Panics if `rhs.nrows() != self.symbolic().nrows()`.
        pub fn solve_in_place_with_conj(
            &self,
            conj: Conj,
            rhs: MatMut<'_, E>,
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
                let (Ls_top, Ls_bot) = Ls.split_at_row(size);
                let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
                crate::linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(
                    Ls_top,
                    conj,
                    x_top.rb_mut(),
                    parallelism,
                );

                let (mut tmp, _) = temp_mat_uninit::<E>(s.pattern().len(), k, stack.rb_mut());
                crate::linalg::matmul::matmul_with_conj(
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
                let Ds = s.matrix.diagonal().column_vector();
                for j in 0..k {
                    for idx in 0..size {
                        let d_inv = Ds.read(idx).faer_real();
                        let i = idx + s.start();
                        x.write(i, j, x.read(i, j).faer_scale_real(d_inv))
                    }
                }
            }
            for s in (0..symbolic.n_supernodes()).rev() {
                let s = self.supernode(s);
                let size = s.matrix.ncols();
                let Ls = s.matrix;
                let (Ls_top, Ls_bot) = Ls.split_at_row(size);

                let (mut tmp, _) = temp_mat_uninit::<E>(s.pattern().len(), k, stack.rb_mut());
                for j in 0..k {
                    for (idx, i) in s.pattern().iter().enumerate() {
                        let i = i.zx();
                        tmp.write(idx, j, x.read(i, j));
                    }
                }

                let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
                crate::linalg::matmul::matmul_with_conj(
                    x_top.rb_mut(),
                    Ls_bot.transpose(),
                    conj.compose(Conj::Yes),
                    tmp.rb(),
                    Conj::No,
                    Some(E::faer_one()),
                    E::faer_one().faer_neg(),
                    parallelism,
                );
                crate::linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(
                    Ls_top.transpose(),
                    conj.compose(Conj::Yes),
                    x_top.rb_mut(),
                    parallelism,
                );
            }
        }
    }

    impl<'a, I: Index, E: Entity> SupernodalLltRef<'a, I, E> {
        /// Creates a new Cholesky LLT factor from the symbolic part and
        /// numerical values.
        ///
        /// # Panics
        /// - Panics if `values.len() != symbolic.len_values()`.
        #[inline]
        pub fn new(
            symbolic: &'a SymbolicSupernodalCholesky<I>,
            values: GroupFor<E, &'a [E::Unit]>,
        ) -> Self {
            let values = SliceGroup::new(values);
            assert!(values.len() == symbolic.len_values());
            Self { symbolic, values }
        }

        /// Returns the symbolic part of the Cholesky factor.
        #[inline]
        pub fn symbolic(self) -> &'a SymbolicSupernodalCholesky<I> {
            self.symbolic
        }

        /// Returns the numerical values of the L factor.
        #[inline]
        pub fn values(self) -> SliceGroup<'a, E> {
            self.values
        }

        /// Returns the `s`'th supernode.
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

            let Ls = crate::mat::from_column_major_slice::<'_, E>(
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

        /// Solves the equation `Op(A) x = rhs` and stores the result in `rhs`, where `Op` is either
        /// the identity or the conjugate, depending on the value of `conj`.
        ///
        /// # Panics
        /// Panics if `rhs.nrows() != self.symbolic().nrows()`.
        pub fn solve_in_place_with_conj(
            &self,
            conj: Conj,
            rhs: MatMut<'_, E>,
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
                let (Ls_top, Ls_bot) = Ls.split_at_row(size);
                let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
                crate::linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(
                    Ls_top,
                    conj,
                    x_top.rb_mut(),
                    parallelism,
                );

                let (mut tmp, _) = temp_mat_uninit::<E>(s.pattern().len(), k, stack.rb_mut());
                crate::linalg::matmul::matmul_with_conj(
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
            for s in (0..symbolic.n_supernodes()).rev() {
                let s = self.supernode(s);
                let size = s.matrix.ncols();
                let Ls = s.matrix;
                let (Ls_top, Ls_bot) = Ls.split_at_row(size);

                let (mut tmp, _) = temp_mat_uninit::<E>(s.pattern().len(), k, stack.rb_mut());
                for j in 0..k {
                    for (idx, i) in s.pattern().iter().enumerate() {
                        let i = i.zx();
                        tmp.write(idx, j, x.read(i, j));
                    }
                }

                let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
                crate::linalg::matmul::matmul_with_conj(
                    x_top.rb_mut(),
                    Ls_bot.transpose(),
                    conj.compose(Conj::Yes),
                    tmp.rb(),
                    Conj::No,
                    Some(E::faer_one()),
                    E::faer_one().faer_neg(),
                    parallelism,
                );
                crate::linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(
                    Ls_top.transpose(),
                    conj.compose(Conj::Yes),
                    x_top.rb_mut(),
                    parallelism,
                );
            }
        }
    }

    impl<I: Index> SymbolicSupernodalCholesky<I> {
        /// Returns the number of supernodes in the Cholesky factor.
        #[inline]
        pub fn n_supernodes(&self) -> usize {
            self.supernode_postorder.len()
        }

        /// Returns the number of rows of the Cholesky factor.
        #[inline]
        pub fn nrows(&self) -> usize {
            self.dimension
        }

        /// Returns the number of columns of the Cholesky factor.
        #[inline]
        pub fn ncols(&self) -> usize {
            self.nrows()
        }

        /// Returns the length of the slice that can be used to contain the numerical values of the
        /// Cholesky factor.
        #[inline]
        pub fn len_values(&self) -> usize {
            self.col_ptrs_for_values()[self.n_supernodes()].zx()
        }

        /// Returns a slice of length `self.n_supernodes()` containing the beginning index of each
        /// supernode.
        #[inline]
        pub fn supernode_begin(&self) -> &[I] {
            &self.supernode_begin[..self.n_supernodes()]
        }

        /// Returns a slice of length `self.n_supernodes()` containing the past-the-end index of
        /// each
        #[inline]
        pub fn supernode_end(&self) -> &[I] {
            &self.supernode_begin[1..]
        }

        /// Returns the column pointers for row indices of each supernode.
        #[inline]
        pub fn col_ptrs_for_row_indices(&self) -> &[I] {
            &self.col_ptrs_for_row_indices
        }

        /// Returns the column pointers for numerical values of each supernode.
        #[inline]
        pub fn col_ptrs_for_values(&self) -> &[I] {
            &self.col_ptrs_for_values
        }

        /// Returns the row indices of the Cholesky factor.
        ///
        /// # Note
        /// Note that the row indices of each supernode do not contain those of the block diagonal
        /// part.
        #[inline]
        pub fn row_indices(&self) -> &[I] {
            &self.row_indices
        }

        /// Returns the symbolic structure of the `s`'th supernode.
        #[inline]
        pub fn supernode(&self, s: usize) -> supernodal::SymbolicSupernodeRef<'_, I> {
            let symbolic = self;
            let start = symbolic.supernode_begin[s].zx();
            let pattern = &symbolic.row_indices()[symbolic.col_ptrs_for_row_indices()[s].zx()
                ..symbolic.col_ptrs_for_row_indices()[s + 1].zx()];
            supernodal::SymbolicSupernodeRef { start, pattern }
        }

        /// Returns the size and alignment of the workspace required to solve the system `A×x =
        /// rhs`.
        pub fn solve_in_place_req<E: Entity>(
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

    /// Returns the size and alignment of the workspace required to compute the symbolic supernodal
    /// factorization of a matrix of size `n`.
    pub fn factorize_supernodal_symbolic_cholesky_req<I: Index>(
        n: usize,
    ) -> Result<StackReq, SizeOverflow> {
        let n_req = StackReq::try_new::<I>(n)?;
        StackReq::try_all_of([n_req, n_req, n_req, n_req])
    }

    /// Computes the supernodal symbolic structure of the Cholesky factor of the matrix `A`.
    ///
    /// # Note
    /// Only the upper triangular part of `A` is analyzed.
    ///
    /// # Panics
    /// The elimination tree and column counts must be computed by calling
    /// [`simplicial::prefactorize_symbolic_cholesky`] with the same matrix. Otherwise, the behavior
    /// is unspecified and panics may occur.
    pub fn factorize_supernodal_symbolic<I: Index>(
        A: SymbolicSparseColMatRef<'_, I>,
        etree: simplicial::EliminationTreeRef<'_, I>,
        col_counts: &[I],
        stack: PodStack<'_>,
        params: SymbolicSupernodalParams<'_>,
    ) -> Result<SymbolicSupernodalCholesky<I>, FaerError> {
        let n = A.nrows();
        assert!(A.nrows() == A.ncols());
        assert!(etree.into_inner().len() == n);
        assert!(col_counts.len() == n);
        ghost::with_size(n, |N| {
            ghost_factorize_supernodal_symbolic(
                ghost::SymbolicSparseColMatRef::new(A, N, N),
                None,
                None,
                CholeskyInput::A,
                etree.ghost_inner(N),
                Array::from_ref(col_counts, N),
                stack,
                params,
            )
        })
    }

    pub(crate) enum CholeskyInput {
        A,
        ATA,
    }

    pub(crate) fn ghost_factorize_supernodal_symbolic<'m, 'n, I: Index>(
        A: ghost::SymbolicSparseColMatRef<'m, 'n, '_, I>,
        col_perm: Option<ghost::PermRef<'n, '_, I>>,
        min_col: Option<&Array<'m, MaybeIdx<'n, I>>>,
        input: CholeskyInput,
        etree: &Array<'n, MaybeIdx<'n, I>>,
        col_counts: &Array<'n, I>,
        stack: PodStack<'_>,
        params: SymbolicSupernodalParams<'_>,
    ) -> Result<SymbolicSupernodalCholesky<I>, FaerError> {
        let to_wide = |i: I| i.zx() as u128;
        let from_wide = |i: u128| I::truncate(i as usize);
        let from_wide_checked = |i: u128| -> Option<I> {
            (i <= to_wide(I::from_signed(I::Signed::MAX))).then_some(I::truncate(i as usize))
        };

        let N = A.ncols();
        let n = *N;

        let zero = I::truncate(0);
        let one = I::truncate(1);
        let none = I::Signed::truncate(NONE);

        if n == 0 {
            // would be funny if this allocation failed
            return Ok(SymbolicSupernodalCholesky {
                dimension: n,
                supernode_postorder: alloc::vec::Vec::new(),
                supernode_postorder_inv: alloc::vec::Vec::new(),
                descendant_count: alloc::vec::Vec::new(),

                supernode_begin: try_collect([zero])?,
                col_ptrs_for_row_indices: try_collect([zero])?,
                col_ptrs_for_values: try_collect([zero])?,
                row_indices: alloc::vec::Vec::new(),
            });
        }
        let mut original_stack = stack;

        let (index_to_super__, stack) = original_stack.rb_mut().make_raw::<I>(n);
        let (super_etree__, stack) = stack.make_raw::<I::Signed>(n);
        let (supernode_sizes__, stack) = stack.make_raw::<I>(n);
        let (child_count__, _) = stack.make_raw::<I>(n);

        let child_count = Array::from_mut(child_count__, N);
        let index_to_super = Array::from_mut(index_to_super__, N);

        mem::fill_zero(child_count.as_mut());
        for j in N.indices() {
            if let Some(parent) = etree[j].idx() {
                child_count[parent.zx()] += one;
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
            supernode_sizes__[current_supernode] += one;
        }
        let n_fundamental_supernodes = current_supernode + 1;

        // last n elements contain supernode degrees
        let supernode_begin__ = ghost::with_size(
            n_fundamental_supernodes,
            |N_FUNDAMENTAL_SUPERNODES| -> Result<alloc::vec::Vec<I>, FaerError> {
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
                    index_to_super.as_mut()[supernode_begin..][..size].fill(*s.truncate::<I>());
                    supernode_begin += size;
                }

                let index_to_super = Array::from_mut(
                    Idx::from_slice_mut_checked(index_to_super.as_mut(), N_FUNDAMENTAL_SUPERNODES),
                    N,
                );

                let mut supernode_begin = 0usize;
                for s in N_FUNDAMENTAL_SUPERNODES.indices() {
                    let size = supernode_sizes[s].zx();
                    let last = supernode_begin + size - 1;
                    let last = N.check(last);
                    if let Some(parent) = etree[last].idx() {
                        super_etree[s] = index_to_super[parent.zx()].to_signed();
                    } else {
                        super_etree[s] = none;
                    }
                    supernode_begin += size;
                }

                let super_etree = Array::from_mut(
                    MaybeIdx::<'_, I>::from_slice_mut_checked(
                        super_etree.as_mut(),
                        N_FUNDAMENTAL_SUPERNODES,
                    ),
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

                    let child_lists = bytemuck::cast_slice_mut(
                        &mut child_count.as_mut()[..n_fundamental_supernodes],
                    );
                    let (child_list_heads, stack) =
                        stack.make_raw::<I::Signed>(n_fundamental_supernodes);
                    let (last_merged_children, stack) =
                        stack.make_raw::<I::Signed>(n_fundamental_supernodes);
                    let (merge_parents, stack) =
                        stack.make_raw::<I::Signed>(n_fundamental_supernodes);
                    let (fundamental_supernode_degrees, stack) =
                        stack.make_raw::<I>(n_fundamental_supernodes);
                    let (num_zeros, _) = stack.make_raw::<I>(n_fundamental_supernodes);

                    let child_lists = Array::from_mut(
                        ghost::fill_none::<I>(child_lists, N_FUNDAMENTAL_SUPERNODES),
                        N_FUNDAMENTAL_SUPERNODES,
                    );
                    let child_list_heads = Array::from_mut(
                        ghost::fill_none::<I>(child_list_heads, N_FUNDAMENTAL_SUPERNODES),
                        N_FUNDAMENTAL_SUPERNODES,
                    );
                    let last_merged_children = Array::from_mut(
                        ghost::fill_none::<I>(last_merged_children, N_FUNDAMENTAL_SUPERNODES),
                        N_FUNDAMENTAL_SUPERNODES,
                    );
                    let merge_parents = Array::from_mut(
                        ghost::fill_none::<I>(merge_parents, N_FUNDAMENTAL_SUPERNODES),
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

                    mem::fill_zero(num_zeros.as_mut());
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
                    original_to_relaxed.as_mut().fill(MaybeIdx::none());

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
                        &fundamental_supernode_degrees.as_ref()[..n_relaxed_supernodes],
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
                |N_SUPERNODES| -> Result<
                    (
                        alloc::vec::Vec<I>,
                        alloc::vec::Vec<I>,
                        alloc::vec::Vec<I>,
                        alloc::vec::Vec<I>,
                    ),
                    FaerError,
                > {
                    let supernode_sizes =
                        Array::from_mut(&mut supernode_sizes__[..n_supernodes], N_SUPERNODES);

                    if n_supernodes != n_fundamental_supernodes {
                        let mut supernode_begin = 0usize;
                        for s in N_SUPERNODES.indices() {
                            let size = supernode_sizes[s].zx();
                            index_to_super.as_mut()[supernode_begin..][..size]
                                .fill(*s.truncate::<I>());
                            supernode_begin += size;
                        }

                        let index_to_super = Array::from_mut(
                            Idx::<'_, I>::from_slice_mut_checked(
                                index_to_super.as_mut(),
                                N_SUPERNODES,
                            ),
                            N,
                        );
                        let super_etree =
                            Array::from_mut(&mut super_etree__[..n_supernodes], N_SUPERNODES);

                        let mut supernode_begin = 0usize;
                        for s in N_SUPERNODES.indices() {
                            let size = supernode_sizes[s].zx();
                            let last = supernode_begin + size - 1;
                            if let Some(parent) = etree[N.check(last)].idx() {
                                super_etree[s] = index_to_super[parent.zx()].to_signed();
                            } else {
                                super_etree[s] = none;
                            }
                            supernode_begin += size;
                        }
                    }

                    let index_to_super = Array::from_mut(
                        Idx::from_slice_mut_checked(index_to_super.as_mut(), N_SUPERNODES),
                        N,
                    );

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
                        MaybeIdx::from_slice_ref_checked(
                            &super_etree__[..n_supernodes],
                            N_SUPERNODES,
                        ),
                        N_SUPERNODES,
                    );

                    let current_row_positions = supernode_sizes;

                    let row_indices = Idx::from_slice_mut_checked(&mut row_indices__, N);
                    let visited = Array::from_mut(
                        bytemuck::cast_slice_mut(&mut child_count.as_mut()[..n_supernodes]),
                        N_SUPERNODES,
                    );

                    mem::fill_none::<I::Signed>(visited.as_mut());
                    if matches!(input, CholeskyInput::A) {
                        let A = ghost::SymbolicSparseColMatRef::new(A.into_inner(), N, N);
                        for s in N_SUPERNODES.indices() {
                            let k1 =
                                ghost::IdxInclusive::new_checked(supernode_begin__[*s].zx(), N);
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
                    } else {
                        let min_col = min_col.unwrap();
                        for s in N_SUPERNODES.indices() {
                            let k1 =
                                ghost::IdxInclusive::new_checked(supernode_begin__[*s].zx(), N);
                            let k2 =
                                ghost::IdxInclusive::new_checked(supernode_begin__[*s + 1].zx(), N);

                            for k in k1.range_to(k2) {
                                ereach_super_ata(
                                    A,
                                    col_perm,
                                    min_col,
                                    super_etree,
                                    index_to_super,
                                    current_row_positions,
                                    row_indices,
                                    k,
                                    visited,
                                );
                            }
                        }
                    }

                    debug_assert!(
                        current_row_positions.as_ref() == &col_ptrs_for_row_indices__[1..]
                    );

                    Ok((
                        supernode_begin__,
                        col_ptrs_for_row_indices__,
                        col_ptrs_for_values__,
                        row_indices__,
                    ))
                },
            )?;

        let mut supernode_etree__: alloc::vec::Vec<I> = try_collect(
            bytemuck::cast_slice(&super_etree__[..n_supernodes])
                .iter()
                .copied(),
        )?;
        let mut supernode_postorder__ = try_zeroed::<I>(n_supernodes)?;

        let mut descendent_count__ = try_zeroed::<I>(n_supernodes)?;

        ghost::with_size(n_supernodes, |N_SUPERNODES| {
            let post = Array::from_mut(&mut supernode_postorder__, N_SUPERNODES);
            let desc_count = Array::from_mut(&mut descendent_count__, N_SUPERNODES);
            let etree: &Array<'_, MaybeIdx<'_, I>> = Array::from_ref(
                MaybeIdx::from_slice_ref_checked(
                    bytemuck::cast_slice(&supernode_etree__),
                    N_SUPERNODES,
                ),
                N_SUPERNODES,
            );

            for s in N_SUPERNODES.indices() {
                if let Some(parent) = etree[s].idx() {
                    let parent = parent.zx();
                    desc_count[parent] = desc_count[parent] + desc_count[s] + one;
                }
            }

            ghost_postorder(post, etree, original_stack);
            let post_inv = Array::from_mut(
                bytemuck::cast_slice_mut(&mut supernode_etree__),
                N_SUPERNODES,
            );
            for i in N_SUPERNODES.indices() {
                post_inv[N_SUPERNODES.check(post[i].zx())] = I::truncate(*i);
            }
        });

        Ok(SymbolicSupernodalCholesky {
            dimension: n,
            supernode_postorder: supernode_postorder__,
            supernode_postorder_inv: supernode_etree__,
            descendant_count: descendent_count__,
            supernode_begin: supernode_begin__,
            col_ptrs_for_row_indices: col_ptrs_for_row_indices__,
            col_ptrs_for_values: col_ptrs_for_values__,
            row_indices: row_indices__,
        })
    }

    #[inline]
    pub(crate) fn partition_fn<I: Index>(idx: usize) -> impl Fn(&I) -> bool {
        let idx = I::truncate(idx);
        move |&i| i < idx
    }

    /// Returns the size and alignment of the workspace required to compute the numeric
    /// Cholesky LLT factorization of a matrix `A` with dimension `n`.
    pub fn factorize_supernodal_numeric_llt_req<I: Index, E: Entity>(
        symbolic: &SymbolicSupernodalCholesky<I>,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let n_supernodes = symbolic.n_supernodes();
        let n = symbolic.nrows();
        let post = &*symbolic.supernode_postorder;
        let post_inv = &*symbolic.supernode_postorder_inv;

        let desc_count = &*symbolic.descendant_count;

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

                let d_pattern = &row_ind[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()];
                let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
                let d_pattern_mid_len =
                    d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));

                d_req = d_req.try_and(temp_mat_req::<E>(
                    d_pattern.len() - d_pattern_start,
                    d_pattern_mid_len,
                )?)?;
                req = req.try_or(d_req)?;
            }
            req = req.try_or(
                crate::linalg::cholesky::ldlt_diagonal::compute::raw_cholesky_in_place_req::<E>(
                    s_ncols,
                    parallelism,
                    Default::default(),
                )?,
            )?;
        }
        req.try_and(StackReq::try_new::<I>(n)?)
    }

    /// Returns the size and alignment of the workspace required to compute the numeric
    /// Cholesky LDLT factorization of a matrix `A` with dimension `n`.
    pub fn factorize_supernodal_numeric_ldlt_req<I: Index, E: Entity>(
        symbolic: &SymbolicSupernodalCholesky<I>,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let n_supernodes = symbolic.n_supernodes();
        let n = symbolic.nrows();
        let post = &*symbolic.supernode_postorder;
        let post_inv = &*symbolic.supernode_postorder_inv;

        let desc_count = &*symbolic.descendant_count;

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
                crate::linalg::cholesky::ldlt_diagonal::compute::raw_cholesky_in_place_req::<E>(
                    s_ncols,
                    parallelism,
                    Default::default(),
                )?,
            )?;
        }
        req.try_and(StackReq::try_new::<I>(n)?)
    }

    /// Returns the size and alignment of the workspace required to compute the numeric
    /// Cholesky Bunch-Kaufman factorization with intranodal pivoting of a matrix `A` with dimension
    /// `n`.
    pub fn factorize_supernodal_numeric_intranode_bunch_kaufman_req<I: Index, E: Entity>(
        symbolic: &SymbolicSupernodalCholesky<I>,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let n_supernodes = symbolic.n_supernodes();
        let n = symbolic.nrows();
        let post = &*symbolic.supernode_postorder;
        let post_inv = &*symbolic.supernode_postorder_inv;

        let desc_count = &*symbolic.descendant_count;

        let col_ptr_row = &*symbolic.col_ptrs_for_row_indices;
        let row_ind = &*symbolic.row_indices;

        let mut req = StackReq::empty();
        for s in 0..n_supernodes {
            let s_start = symbolic.supernode_begin[s].zx();
            let s_end = symbolic.supernode_begin[s + 1].zx();

            let s_ncols = s_end - s_start;
            let s_pattern = &row_ind[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()];

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
            req = StackReq::try_any_of([
                req,
                crate::linalg::cholesky::bunch_kaufman::compute::cholesky_in_place_req::<I, E>(
                    s_ncols,
                    parallelism,
                    Default::default(),
                )?,
                crate::perm::permute_cols_in_place_req::<I, E>(s_pattern.len(), s_ncols)?,
            ])?;
        }
        req.try_and(StackReq::try_new::<I>(n)?)
    }

    /// Computes the numeric values of the Cholesky LLT factor of the matrix `A`, and stores them in
    /// `L_values`.
    ///
    /// # Warning
    /// Only the *lower* (not upper, unlikely the other functions) triangular part of `A` is
    /// accessed.
    ///
    /// # Panics
    /// The symbolic structure must be computed by calling
    /// [`factorize_supernodal_symbolic`] on a matrix with the same symbolic structure.
    /// Otherwise, the behavior is unspecified and panics may occur.
    pub fn factorize_supernodal_numeric_llt<I: Index, E: ComplexField>(
        L_values: GroupFor<E, &mut [E::Unit]>,
        A_lower: SparseColMatRef<'_, I, E>,
        regularization: LltRegularization<E>,
        symbolic: &SymbolicSupernodalCholesky<I>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) -> Result<usize, CholeskyError> {
        let n_supernodes = symbolic.n_supernodes();
        let n = symbolic.nrows();
        let mut dynamic_regularization_count = 0usize;
        let mut L_values = SliceGroupMut::<'_, E>::new(L_values);
        L_values.fill_zero();

        assert!(A_lower.nrows() == n);
        assert!(A_lower.ncols() == n);
        assert!(L_values.len() == symbolic.len_values());
        let slice_group = SliceGroup::<'_, E>::new;

        let none = I::Signed::truncate(NONE);

        let post = &*symbolic.supernode_postorder;
        let post_inv = &*symbolic.supernode_postorder_inv;

        let desc_count = &*symbolic.descendant_count;

        let col_ptr_row = &*symbolic.col_ptrs_for_row_indices;
        let col_ptr_val = &*symbolic.col_ptrs_for_values;
        let row_ind = &*symbolic.row_indices;

        // mapping from global indices to local
        let (global_to_local, mut stack) = stack.make_raw::<I::Signed>(n);
        mem::fill_none(global_to_local);

        for s in 0..n_supernodes {
            let s_start = symbolic.supernode_begin[s].zx();
            let s_end = symbolic.supernode_begin[s + 1].zx();

            let s_pattern = &row_ind[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()];
            let s_ncols = s_end - s_start;
            let s_nrows = s_pattern.len() + s_ncols;

            for (i, &row) in s_pattern.iter().enumerate() {
                global_to_local[row.zx()] = I::Signed::truncate(i + s_ncols);
            }

            let (head, tail) = L_values.rb_mut().split_at(col_ptr_val[s].zx());
            let head = head.rb();
            let mut Ls = crate::mat::from_column_major_slice_mut::<'_, E>(
                tail.subslice(0..(col_ptr_val[s + 1] - col_ptr_val[s]).zx())
                    .into_inner(),
                s_nrows,
                s_ncols,
            );

            for j in s_start..s_end {
                let j_shifted = j - s_start;
                for (i, val) in zip(
                    A_lower.row_indices_of_col(j),
                    slice_group(A_lower.values_of_col(j)).into_ref_iter(),
                ) {
                    let val = val.read();
                    let (ix, iy) = if i >= s_end {
                        (global_to_local[i].sx(), j_shifted)
                    } else {
                        (i - s_start, j_shifted)
                    };
                    Ls.write(ix, iy, Ls.read(ix, iy).faer_add(val));
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

                let Ld = crate::mat::from_column_major_slice::<'_, E>(
                    head.subslice(col_ptr_val[d].zx()..col_ptr_val[d + 1].zx())
                        .into_inner(),
                    d_nrows,
                    d_ncols,
                );

                let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
                let d_pattern_mid_len =
                    d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));
                let d_pattern_mid = d_pattern_start + d_pattern_mid_len;

                let (_, Ld_mid_bot) = Ld.split_at_row(d_ncols);
                let (_, Ld_mid_bot) = Ld_mid_bot.split_at_row(d_pattern_start);
                let (Ld_mid, Ld_bot) = Ld_mid_bot.split_at_row(d_pattern_mid_len);

                let stack = stack.rb_mut();

                let (tmp, _) = temp_mat_uninit::<E>(Ld_mid_bot.nrows(), d_pattern_mid_len, stack);

                let (mut tmp_top, mut tmp_bot) = tmp.split_at_row_mut(d_pattern_mid_len);

                use crate::linalg::{matmul, matmul::triangular};
                triangular::matmul(
                    tmp_top.rb_mut(),
                    triangular::BlockStructure::TriangularLower,
                    Ld_mid,
                    triangular::BlockStructure::Rectangular,
                    Ld_mid.rb().adjoint(),
                    triangular::BlockStructure::Rectangular,
                    None,
                    E::faer_one(),
                    parallelism,
                );
                matmul::matmul(
                    tmp_bot.rb_mut(),
                    Ld_bot,
                    Ld_mid.rb().adjoint(),
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

            let (mut Ls_top, mut Ls_bot) = Ls.rb_mut().split_at_row_mut(s_ncols);

            let params = Default::default();
            dynamic_regularization_count +=
                match crate::linalg::cholesky::llt::compute::cholesky_in_place(
                    Ls_top.rb_mut(),
                    regularization,
                    parallelism,
                    stack.rb_mut(),
                    params,
                ) {
                    Ok(count) => count,
                    Err(err) => {
                        return Err(CholeskyError {
                            non_positive_definite_minor: err.non_positive_definite_minor + s_start,
                        })
                    }
                }
                .dynamic_regularization_count;
            crate::linalg::triangular_solve::solve_lower_triangular_in_place(
                Ls_top.rb().conjugate(),
                Ls_bot.rb_mut().transpose_mut(),
                parallelism,
            );

            for &row in s_pattern {
                global_to_local[row.zx()] = none;
            }
        }
        Ok(dynamic_regularization_count)
    }

    /// Computes the numeric values of the Cholesky LDLT factors of the matrix `A`, and stores them
    /// in `L_values`.
    ///
    /// # Note
    /// Only the *lower* (not upper, unlikely the other functions) triangular part of `A` is
    /// accessed.
    ///
    /// # Panics
    /// The symbolic structure must be computed by calling
    /// [`factorize_supernodal_symbolic`] on a matrix with the same symbolic structure.
    /// Otherwise, the behavior is unspecified and panics may occur.
    pub fn factorize_supernodal_numeric_ldlt<I: Index, E: ComplexField>(
        L_values: GroupFor<E, &mut [E::Unit]>,
        A_lower: SparseColMatRef<'_, I, E>,
        regularization: LdltRegularization<'_, E>,
        symbolic: &SymbolicSupernodalCholesky<I>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) -> usize {
        let n_supernodes = symbolic.n_supernodes();
        let n = symbolic.nrows();
        let mut dynamic_regularization_count = 0usize;
        let mut L_values = SliceGroupMut::<'_, E>::new(L_values);
        L_values.fill_zero();

        assert!(A_lower.nrows() == n);
        assert!(A_lower.ncols() == n);
        assert!(L_values.len() == symbolic.len_values());
        let slice_group = SliceGroup::<'_, E>::new;

        let none = I::Signed::truncate(NONE);

        let post = &*symbolic.supernode_postorder;
        let post_inv = &*symbolic.supernode_postorder_inv;

        let desc_count = &*symbolic.descendant_count;

        let col_ptr_row = &*symbolic.col_ptrs_for_row_indices;
        let col_ptr_val = &*symbolic.col_ptrs_for_values;
        let row_ind = &*symbolic.row_indices;

        // mapping from global indices to local
        let (global_to_local, mut stack) = stack.make_raw::<I::Signed>(n);
        mem::fill_none(global_to_local);

        for s in 0..n_supernodes {
            let s_start = symbolic.supernode_begin[s].zx();
            let s_end = symbolic.supernode_begin[s + 1].zx();

            let s_pattern = &row_ind[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()];
            let s_ncols = s_end - s_start;
            let s_nrows = s_pattern.len() + s_ncols;

            for (i, &row) in s_pattern.iter().enumerate() {
                global_to_local[row.zx()] = I::Signed::truncate(i + s_ncols);
            }

            let (head, tail) = L_values.rb_mut().split_at(col_ptr_val[s].zx());
            let head = head.rb();
            let mut Ls = crate::mat::from_column_major_slice_mut::<'_, E>(
                tail.subslice(0..(col_ptr_val[s + 1] - col_ptr_val[s]).zx())
                    .into_inner(),
                s_nrows,
                s_ncols,
            );

            for j in s_start..s_end {
                let j_shifted = j - s_start;
                for (i, val) in zip(
                    A_lower.row_indices_of_col(j),
                    slice_group(A_lower.values_of_col(j)).into_ref_iter(),
                ) {
                    let val = val.read();
                    let (ix, iy) = if i >= s_end {
                        (global_to_local[i].sx(), j_shifted)
                    } else {
                        (i - s_start, j_shifted)
                    };
                    Ls.write(ix, iy, Ls.read(ix, iy).faer_add(val));
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

                let Ld = crate::mat::from_column_major_slice::<'_, E>(
                    head.subslice(col_ptr_val[d].zx()..col_ptr_val[d + 1].zx())
                        .into_inner(),
                    d_nrows,
                    d_ncols,
                );

                let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
                let d_pattern_mid_len =
                    d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));
                let d_pattern_mid = d_pattern_start + d_pattern_mid_len;

                let (Ld_top, Ld_mid_bot) = Ld.split_at_row(d_ncols);
                let (_, Ld_mid_bot) = Ld_mid_bot.split_at_row(d_pattern_start);
                let (Ld_mid, Ld_bot) = Ld_mid_bot.split_at_row(d_pattern_mid_len);
                let D = Ld_top.diagonal().column_vector();

                let stack = stack.rb_mut();

                let (tmp, stack) =
                    temp_mat_uninit::<E>(Ld_mid_bot.nrows(), d_pattern_mid_len, stack);
                let (tmp2, _) = temp_mat_uninit::<E>(Ld_mid.ncols(), Ld_mid.nrows(), stack);
                let mut Ld_mid_x_D = tmp2.transpose_mut();

                for i in 0..d_pattern_mid_len {
                    for j in 0..d_ncols {
                        Ld_mid_x_D.write(
                            i,
                            j,
                            Ld_mid
                                .read(i, j)
                                .faer_scale_real(D.read(j).faer_real().faer_inv()),
                        );
                    }
                }

                let (mut tmp_top, mut tmp_bot) = tmp.split_at_row_mut(d_pattern_mid_len);

                use crate::linalg::{matmul, matmul::triangular};
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
                matmul::matmul(
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

            let (mut Ls_top, mut Ls_bot) = Ls.rb_mut().split_at_row_mut(s_ncols);

            let params = Default::default();
            dynamic_regularization_count +=
                crate::linalg::cholesky::ldlt_diagonal::compute::raw_cholesky_in_place(
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
                )
                .dynamic_regularization_count;
            zipped!(Ls_top.rb_mut())
                .for_each_triangular_upper(crate::linalg::zip::Diag::Skip, |unzipped!(mut x)| {
                    x.write(E::faer_zero())
                });
            crate::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                Ls_top.rb().conjugate(),
                Ls_bot.rb_mut().transpose_mut(),
                parallelism,
            );
            for j in 0..s_ncols {
                let d = Ls_top.read(j, j).faer_real();
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

    /// Computes the numeric values of the Cholesky Bunch-Kaufman factors of the matrix `A` with
    /// intranodal pivoting, and stores them in `L_values`.
    ///
    /// # Note
    /// Only the *lower* (not upper, unlikely the other functions) triangular part of `A` is
    /// accessed.
    ///
    /// # Panics
    /// The symbolic structure must be computed by calling
    /// [`factorize_supernodal_symbolic`] on a matrix with the same symbolic structure.
    /// Otherwise, the behavior is unspecified and panics may occur.
    pub fn factorize_supernodal_numeric_intranode_bunch_kaufman<I: Index, E: ComplexField>(
        L_values: GroupFor<E, &mut [E::Unit]>,
        subdiag: GroupFor<E, &mut [E::Unit]>,
        perm_forward: &mut [I],
        perm_inverse: &mut [I],
        A_lower: SparseColMatRef<'_, I, E>,
        regularization: BunchKaufmanRegularization<'_, E>,
        symbolic: &SymbolicSupernodalCholesky<I>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) -> usize {
        let mut regularization = regularization;
        let n_supernodes = symbolic.n_supernodes();
        let n = symbolic.nrows();
        let mut dynamic_regularization_count = 0usize;
        let mut L_values = SliceGroupMut::<'_, E>::new(L_values);
        let mut subdiag = SliceGroupMut::<'_, E>::new(subdiag);
        L_values.fill_zero();

        assert!(A_lower.nrows() == n);
        assert!(A_lower.ncols() == n);
        assert!(perm_forward.len() == n);
        assert!(perm_inverse.len() == n);
        assert!(subdiag.len() == n);
        assert!(L_values.len() == symbolic.len_values());
        let slice_group = SliceGroup::<'_, E>::new;

        let none = I::Signed::truncate(NONE);

        let post = &*symbolic.supernode_postorder;
        let post_inv = &*symbolic.supernode_postorder_inv;

        let desc_count = &*symbolic.descendant_count;

        let col_ptr_row = &*symbolic.col_ptrs_for_row_indices;
        let col_ptr_val = &*symbolic.col_ptrs_for_values;
        let row_ind = &*symbolic.row_indices;

        // mapping from global indices to local
        let (global_to_local, mut stack) = stack.make_raw::<I::Signed>(n);
        mem::fill_none(global_to_local);

        for s in 0..n_supernodes {
            let s_start = symbolic.supernode_begin[s].zx();
            let s_end = symbolic.supernode_begin[s + 1].zx();

            let s_pattern = &row_ind[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()];
            let s_ncols = s_end - s_start;
            let s_nrows = s_pattern.len() + s_ncols;

            for (i, &row) in s_pattern.iter().enumerate() {
                global_to_local[row.zx()] = I::Signed::truncate(i + s_ncols);
            }

            let (head, tail) = L_values.rb_mut().split_at(col_ptr_val[s].zx());
            let head = head.rb();
            let mut Ls = crate::mat::from_column_major_slice_mut::<'_, E>(
                tail.subslice(0..(col_ptr_val[s + 1] - col_ptr_val[s]).zx())
                    .into_inner(),
                s_nrows,
                s_ncols,
            );

            for j in s_start..s_end {
                let j_shifted = j - s_start;
                for (i, val) in zip(
                    A_lower.row_indices_of_col(j),
                    slice_group(A_lower.values_of_col(j)).into_ref_iter(),
                ) {
                    let val = val.read();
                    let (ix, iy) = if i >= s_end {
                        (global_to_local[i].sx(), j_shifted)
                    } else {
                        (i - s_start, j_shifted)
                    };
                    Ls.write(ix, iy, Ls.read(ix, iy).faer_add(val));
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

                let Ld = crate::mat::from_column_major_slice::<'_, E>(
                    head.subslice(col_ptr_val[d].zx()..col_ptr_val[d + 1].zx())
                        .into_inner(),
                    d_nrows,
                    d_ncols,
                );

                let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
                let d_pattern_mid_len =
                    d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));
                let d_pattern_mid = d_pattern_start + d_pattern_mid_len;

                let (Ld_top, Ld_mid_bot) = Ld.split_at_row(d_ncols);
                let (_, Ld_mid_bot) = Ld_mid_bot.split_at_row(d_pattern_start);
                let (Ld_mid, Ld_bot) = Ld_mid_bot.split_at_row(d_pattern_mid_len);
                let d_subdiag = subdiag.rb().subslice(d_start..d_start + d_ncols);

                let stack = stack.rb_mut();

                let (tmp, stack) =
                    temp_mat_uninit::<E>(Ld_mid_bot.nrows(), d_pattern_mid_len, stack);
                let (tmp2, _) = temp_mat_uninit::<E>(Ld_mid.ncols(), Ld_mid.nrows(), stack);
                let mut Ld_mid_x_D = tmp2.transpose_mut();

                let mut j = 0;
                while j < d_ncols {
                    let subdiag = d_subdiag.read(j);
                    if subdiag == E::faer_zero() {
                        let d = Ld_top.read(j, j).faer_real().faer_inv();
                        for i in 0..d_pattern_mid_len {
                            Ld_mid_x_D.write(i, j, Ld_mid.read(i, j).faer_scale_real(d));
                        }
                        j += 1;
                    } else {
                        // 1/d21
                        let akp1k = subdiag.faer_inv();
                        // d11/d21
                        let ak = akp1k.faer_scale_real(Ld_top.read(j, j).faer_real());
                        // d22/conj(d21)
                        let akp1 = akp1k
                            .faer_conj()
                            .faer_scale_real(Ld_top.read(j + 1, j + 1).faer_real());

                        // (d11 * d21 / |d21|^2  -  1)^-1
                        // = |d21|^2 / ( d11 * d21 - |d21|^2 )
                        let denom = ak
                            .faer_mul(akp1)
                            .faer_real()
                            .faer_sub(E::Real::faer_one())
                            .faer_inv();

                        for i in 0..d_pattern_mid_len {
                            // x1 / d21
                            let xk = Ld_mid.read(i, j).faer_mul(akp1k);
                            // x2 / conj(d21)
                            let xkp1 = Ld_mid.read(i, j + 1).faer_mul(akp1k.faer_conj());

                            // d22/conj(d21) * x1/d21 * |d21|^2 / (d11 * d21 - |d21|^2)
                            // - x2/conj(d21) * |d21|^2 / (d11 * d21 - |d21|^2)
                            //
                            // =  x1 * d22/det - x2 * d21/det
                            Ld_mid_x_D.write(
                                i,
                                j,
                                (akp1.faer_mul(xk).faer_sub(xkp1)).faer_scale_real(denom),
                            );
                            Ld_mid_x_D.write(
                                i,
                                j + 1,
                                (ak.faer_mul(xkp1).faer_sub(xk)).faer_scale_real(denom),
                            );
                        }
                        j += 2;
                    }
                }

                let (mut tmp_top, mut tmp_bot) = tmp.split_at_row_mut(d_pattern_mid_len);

                use crate::linalg::{matmul, matmul::triangular};
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
                matmul::matmul(
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

            let (mut Ls_top, mut Ls_bot) = Ls.rb_mut().split_at_row_mut(s_ncols);
            let mut s_subdiag = subdiag.rb_mut().subslice(s_start..s_end);

            let params = Default::default();
            let (info, perm) = crate::linalg::cholesky::bunch_kaufman::compute::cholesky_in_place(
                Ls_top.rb_mut(),
                crate::mat::from_column_major_slice_mut::<'_, E>(
                    s_subdiag.rb_mut().into_inner(),
                    s_ncols,
                    1,
                ),
                BunchKaufmanRegularization {
                    dynamic_regularization_signs: regularization
                        .dynamic_regularization_signs
                        .rb_mut()
                        .map(|signs| &mut signs[s_start..s_end]),
                    ..regularization
                },
                &mut perm_forward[s_start..s_end],
                &mut perm_inverse[s_start..s_end],
                parallelism,
                stack.rb_mut(),
                params,
            );
            dynamic_regularization_count += info.dynamic_regularization_count;
            zipped!(Ls_top.rb_mut())
                .for_each_triangular_upper(crate::linalg::zip::Diag::Skip, |unzipped!(mut x)| {
                    x.write(E::faer_zero())
                });

            crate::perm::permute_cols_in_place(Ls_bot.rb_mut(), perm.rb(), stack.rb_mut());

            for p in &mut perm_forward[s_start..s_end] {
                *p += I::truncate(s_start);
            }
            for p in &mut perm_inverse[s_start..s_end] {
                *p += I::truncate(s_start);
            }

            crate::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                Ls_top.rb().conjugate(),
                Ls_bot.rb_mut().transpose_mut(),
                parallelism,
            );

            let mut j = 0;
            while j < s_ncols {
                if s_subdiag.read(j) == E::faer_zero() {
                    let d = Ls_top.read(j, j).faer_real();
                    for i in 0..s_pattern.len() {
                        Ls_bot.write(i, j, Ls_bot.read(i, j).faer_scale_real(d));
                    }
                    j += 1;
                } else {
                    let akp1k = s_subdiag.read(j);
                    let ak = Ls_top.read(j, j).faer_real();
                    let akp1 = Ls_top.read(j + 1, j + 1).faer_real();

                    for i in 0..s_pattern.len() {
                        let xk = Ls_bot.read(i, j);
                        let xkp1 = Ls_bot.read(i, j + 1);

                        Ls_bot.write(i, j, xk.faer_scale_real(ak).faer_add(xkp1.faer_mul(akp1k)));
                        Ls_bot.write(
                            i,
                            j + 1,
                            xkp1.faer_scale_real(akp1)
                                .faer_add(xk.faer_mul(akp1k.faer_conj())),
                        );
                    }
                    j += 2;
                }
            }

            for &row in s_pattern {
                global_to_local[row.zx()] = none;
            }
        }
        dynamic_regularization_count
    }

    /// Cholesky LLT factor containing both its symbolic and numeric representations.
    #[derive(Debug)]
    pub struct SupernodalLltRef<'a, I: Index, E: Entity> {
        symbolic: &'a SymbolicSupernodalCholesky<I>,
        values: SliceGroup<'a, E>,
    }

    /// Cholesky LDLT factors containing both the symbolic and numeric representations.
    #[derive(Debug)]
    pub struct SupernodalLdltRef<'a, I: Index, E: Entity> {
        symbolic: &'a SymbolicSupernodalCholesky<I>,
        values: SliceGroup<'a, E>,
    }

    /// Cholesky Bunch-Kaufman factors containing both the symbolic and numeric representations.
    #[derive(Debug)]
    pub struct SupernodalIntranodeBunchKaufmanRef<'a, I: Index, E: Entity> {
        symbolic: &'a SymbolicSupernodalCholesky<I>,
        values: SliceGroup<'a, E>,
        subdiag: SliceGroup<'a, E>,
        pub(super) perm: PermRef<'a, I>,
    }

    /// Cholesky factor structure containing its symbolic structure.
    #[derive(Debug)]
    pub struct SymbolicSupernodalCholesky<I: Index> {
        pub(crate) dimension: usize,
        pub(crate) supernode_postorder: alloc::vec::Vec<I>,
        pub(crate) supernode_postorder_inv: alloc::vec::Vec<I>,
        pub(crate) descendant_count: alloc::vec::Vec<I>,

        pub(crate) supernode_begin: alloc::vec::Vec<I>,
        pub(crate) col_ptrs_for_row_indices: alloc::vec::Vec<I>,
        pub(crate) col_ptrs_for_values: alloc::vec::Vec<I>,
        pub(crate) row_indices: alloc::vec::Vec<I>,
    }
}

// workspace: I×(n)
fn ghost_prefactorize_symbolic_cholesky<'n, 'out, I: Index>(
    etree: &'out mut Array<'n, I::Signed>,
    col_counts: &mut Array<'n, I>,
    A: ghost::SymbolicSparseColMatRef<'n, 'n, '_, I>,
    stack: PodStack<'_>,
) -> &'out mut Array<'n, MaybeIdx<'n, I>> {
    let N = A.ncols();
    let (visited, _) = stack.make_raw::<I>(*N);
    let etree = Array::from_mut(ghost::fill_none::<I>(etree.as_mut(), N), N);
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

                    col_counts[i] += I::truncate(1);
                    visited[i] = *j_;
                    i = next_i;
                }
            }
        }
    }

    etree
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

/// The inner factorization used for the symbolic Cholesky, either simplicial or symbolic.
#[derive(Debug)]
pub enum SymbolicCholeskyRaw<I: Index> {
    /// Simplicial structure.
    Simplicial(simplicial::SymbolicSimplicialCholesky<I>),
    /// Supernodal structure.
    Supernodal(supernodal::SymbolicSupernodalCholesky<I>),
}

/// The symbolic structure of a sparse Cholesky decomposition.
#[derive(Debug)]
pub struct SymbolicCholesky<I: Index> {
    raw: SymbolicCholeskyRaw<I>,
    perm_fwd: alloc::vec::Vec<I>,
    perm_inv: alloc::vec::Vec<I>,
    A_nnz: usize,
}

impl<I: Index> SymbolicCholesky<I> {
    /// Returns the number of rows of the matrix.
    #[inline]
    pub fn nrows(&self) -> usize {
        match &self.raw {
            SymbolicCholeskyRaw::Simplicial(this) => this.nrows(),
            SymbolicCholeskyRaw::Supernodal(this) => this.nrows(),
        }
    }

    /// Returns the number of columns of the matrix.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.nrows()
    }

    /// Returns the inner type of the factorization, either simplicial or symbolic.
    #[inline]
    pub fn raw(&self) -> &SymbolicCholeskyRaw<I> {
        &self.raw
    }

    /// Returns the permutation that was computed during symbolic analysis.
    #[inline]
    pub fn perm(&self) -> PermRef<'_, I> {
        unsafe { PermRef::new_unchecked(&self.perm_fwd, &self.perm_inv) }
    }

    /// Returns the length of the slice needed to store the numerical values of the Cholesky
    /// decomposition.
    #[inline]
    pub fn len_values(&self) -> usize {
        match &self.raw {
            SymbolicCholeskyRaw::Simplicial(this) => this.len_values(),
            SymbolicCholeskyRaw::Supernodal(this) => this.len_values(),
        }
    }

    /// Computes the required workspace size and alignment for a numerical LLT factorization.
    #[inline]
    pub fn factorize_numeric_llt_req<E: Entity>(
        &self,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let n = self.nrows();
        let A_nnz = self.A_nnz;

        let n_req = StackReq::try_new::<I>(n)?;
        let A_req = StackReq::try_all_of([
            make_raw_req::<E>(A_nnz)?,
            StackReq::try_new::<I>(n + 1)?,
            StackReq::try_new::<I>(A_nnz)?,
        ])?;
        let permute_req = n_req;

        let factor_req = match &self.raw {
            SymbolicCholeskyRaw::Simplicial(_) => {
                simplicial::factorize_simplicial_numeric_llt_req::<I, E>(n)?
            }
            SymbolicCholeskyRaw::Supernodal(this) => {
                supernodal::factorize_supernodal_numeric_llt_req::<I, E>(this, parallelism)?
            }
        };

        StackReq::try_all_of([A_req, StackReq::try_or(permute_req, factor_req)?])
    }

    /// Computes the required workspace size and alignment for a numerical LDLT factorization.
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

    /// Computes the required workspace size and alignment for a numerical intranodal Bunch-Kaufman
    /// factorization.
    #[inline]
    pub fn factorize_numeric_intranode_bunch_kaufman_req<E: Entity>(
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
                supernodal::factorize_supernodal_numeric_intranode_bunch_kaufman_req::<I, E>(
                    this,
                    parallelism,
                )?
            }
        };

        StackReq::try_all_of([
            regularization_signs,
            A_req,
            StackReq::try_or(permute_req, factor_req)?,
        ])
    }

    /// Computes a numerical LLT factorization of A, or returns a [`CholeskyError`] if the matrix
    /// is not numerically positive definite.
    #[track_caller]
    pub fn factorize_numeric_llt<'out, E: ComplexField>(
        &'out self,
        L_values: GroupFor<E, &'out mut [E::Unit]>,
        A: SparseColMatRef<'_, I, E>,
        side: Side,
        regularization: LltRegularization<E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) -> Result<LltRef<'out, I, E>, CholeskyError> {
        assert!(A.nrows() == A.ncols());
        let n = A.nrows();
        let mut L_values = L_values;

        ghost::with_size(n, |N| {
            let A_nnz = self.A_nnz;
            let A = ghost::SparseColMatRef::new(A, N, N);

            let perm = ghost::PermRef::new(self.perm(), N);

            let (mut new_values, stack) = crate::sparse::linalg::make_raw::<E>(A_nnz, stack);
            let (new_col_ptr, stack) = stack.make_raw::<I>(n + 1);
            let (new_row_ind, mut stack) = stack.make_raw::<I>(A_nnz);

            let out_side = match &self.raw {
                SymbolicCholeskyRaw::Simplicial(_) => Side::Upper,
                SymbolicCholeskyRaw::Supernodal(_) => Side::Lower,
            };

            let A = unsafe {
                ghost_permute_hermitian_unsorted(
                    new_values.rb_mut(),
                    new_col_ptr,
                    new_row_ind,
                    A,
                    perm,
                    side,
                    out_side,
                    false,
                    stack.rb_mut(),
                )
            };

            match &self.raw {
                SymbolicCholeskyRaw::Simplicial(this) => {
                    simplicial::factorize_simplicial_numeric_llt(
                        E::faer_rb_mut(E::faer_as_mut(&mut L_values)),
                        A.into_inner().into_const(),
                        regularization,
                        this,
                        stack,
                    )?;
                }
                SymbolicCholeskyRaw::Supernodal(this) => {
                    supernodal::factorize_supernodal_numeric_llt(
                        E::faer_rb_mut(E::faer_as_mut(&mut L_values)),
                        A.into_inner().into_const(),
                        regularization,
                        this,
                        parallelism,
                        stack,
                    )?;
                }
            }

            Ok(LltRef::<'out, I, E>::new(
                self,
                E::faer_into_const(L_values),
            ))
        })
    }

    /// Computes a numerical LDLT factorization of A.
    #[inline]
    pub fn factorize_numeric_ldlt<'out, E: ComplexField>(
        &'out self,
        L_values: GroupFor<E, &'out mut [E::Unit]>,
        A: SparseColMatRef<'_, I, E>,
        side: Side,
        regularization: LdltRegularization<'_, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) -> LdltRef<'out, I, E> {
        assert!(A.nrows() == A.ncols());
        let n = A.nrows();
        let mut L_values = L_values;

        ghost::with_size(n, |N| {
            let A_nnz = self.A_nnz;
            let A = ghost::SparseColMatRef::new(A, N, N);

            let (new_signs, stack) =
                stack.make_raw::<i8>(if regularization.dynamic_regularization_signs.is_some() {
                    n
                } else {
                    0
                });

            let perm = ghost::PermRef::new(self.perm(), N);
            let fwd = perm.arrays().0;
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

            let (mut new_values, stack) = crate::sparse::linalg::make_raw::<E>(A_nnz, stack);
            let (new_col_ptr, stack) = stack.make_raw::<I>(n + 1);
            let (new_row_ind, mut stack) = stack.make_raw::<I>(A_nnz);

            let out_side = match &self.raw {
                SymbolicCholeskyRaw::Simplicial(_) => Side::Upper,
                SymbolicCholeskyRaw::Supernodal(_) => Side::Lower,
            };

            let A = unsafe {
                ghost_permute_hermitian_unsorted(
                    new_values.rb_mut(),
                    new_col_ptr,
                    new_row_ind,
                    A,
                    perm,
                    side,
                    out_side,
                    false,
                    stack.rb_mut(),
                )
            };

            match &self.raw {
                SymbolicCholeskyRaw::Simplicial(this) => {
                    simplicial::factorize_simplicial_numeric_ldlt(
                        E::faer_rb_mut(E::faer_as_mut(&mut L_values)),
                        A.into_inner().into_const(),
                        regularization,
                        this,
                        stack,
                    );
                }
                SymbolicCholeskyRaw::Supernodal(this) => {
                    supernodal::factorize_supernodal_numeric_ldlt(
                        E::faer_rb_mut(E::faer_as_mut(&mut L_values)),
                        A.into_inner().into_const(),
                        regularization,
                        this,
                        parallelism,
                        stack,
                    );
                }
            }

            LdltRef::<'out, I, E>::new(self, E::faer_into_const(L_values))
        })
    }

    /// Computes a numerical intranodal Bunch-Kaufman factorization of A.
    #[inline]
    pub fn factorize_numeric_intranode_bunch_kaufman<'out, E: ComplexField>(
        &'out self,
        L_values: GroupFor<E, &'out mut [E::Unit]>,
        subdiag: GroupFor<E, &'out mut [E::Unit]>,
        perm_forward: &'out mut [I],
        perm_inverse: &'out mut [I],
        A: SparseColMatRef<'_, I, E>,
        side: Side,
        regularization: LdltRegularization<'_, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) -> IntranodeBunchKaufmanRef<'out, I, E> {
        assert!(A.nrows() == A.ncols());
        let n = A.nrows();
        let mut L_values = L_values;
        let mut subdiag = subdiag;

        ghost::with_size(n, move |N| {
            let A_nnz = self.A_nnz;
            let A = ghost::SparseColMatRef::new(A, N, N);

            let (new_signs, stack) =
                stack.make_raw::<i8>(if regularization.dynamic_regularization_signs.is_some() {
                    n
                } else {
                    0
                });

            let static_perm = ghost::PermRef::new(self.perm(), N);
            let signs = regularization.dynamic_regularization_signs.map(|signs| {
                {
                    let fwd = static_perm.arrays().0;
                    let new_signs = Array::from_mut(new_signs, N);
                    let signs = Array::from_ref(signs, N);
                    for i in N.indices() {
                        new_signs[i] = signs[fwd[i].zx()];
                    }
                }
                &mut *new_signs
            });

            let (mut new_values, stack) = crate::sparse::linalg::make_raw::<E>(A_nnz, stack);
            let (new_col_ptr, stack) = stack.make_raw::<I>(n + 1);
            let (new_row_ind, mut stack) = stack.make_raw::<I>(A_nnz);

            let out_side = match &self.raw {
                SymbolicCholeskyRaw::Simplicial(_) => Side::Upper,
                SymbolicCholeskyRaw::Supernodal(_) => Side::Lower,
            };

            let A = unsafe {
                ghost_permute_hermitian_unsorted(
                    new_values.rb_mut(),
                    new_col_ptr,
                    new_row_ind,
                    A,
                    static_perm,
                    side,
                    out_side,
                    false,
                    stack.rb_mut(),
                )
            };

            match &self.raw {
                SymbolicCholeskyRaw::Simplicial(this) => {
                    let regularization = LdltRegularization {
                        dynamic_regularization_signs: signs.rb(),
                        dynamic_regularization_delta: regularization.dynamic_regularization_delta,
                        dynamic_regularization_epsilon: regularization
                            .dynamic_regularization_epsilon,
                    };
                    for (i, p) in perm_forward.iter_mut().enumerate() {
                        *p = I::truncate(i);
                    }
                    for (i, p) in perm_inverse.iter_mut().enumerate() {
                        *p = I::truncate(i);
                    }
                    simplicial::factorize_simplicial_numeric_ldlt(
                        E::faer_rb_mut(E::faer_as_mut(&mut L_values)),
                        A.into_inner().into_const(),
                        regularization,
                        this,
                        stack,
                    );
                }
                SymbolicCholeskyRaw::Supernodal(this) => {
                    let regularization = BunchKaufmanRegularization {
                        dynamic_regularization_signs: signs,
                        dynamic_regularization_delta: regularization.dynamic_regularization_delta,
                        dynamic_regularization_epsilon: regularization
                            .dynamic_regularization_epsilon,
                    };

                    supernodal::factorize_supernodal_numeric_intranode_bunch_kaufman(
                        E::faer_rb_mut(E::faer_as_mut(&mut L_values)),
                        E::faer_rb_mut(E::faer_as_mut(&mut subdiag)),
                        perm_forward,
                        perm_inverse,
                        A.into_inner().into_const(),
                        regularization,
                        this,
                        parallelism,
                        stack,
                    );
                }
            }

            IntranodeBunchKaufmanRef::<'out, I, E>::new(
                self,
                E::faer_into_const(L_values),
                E::faer_into_const(subdiag),
                unsafe { PermRef::<'out, I>::new_unchecked(perm_forward, perm_inverse) },
            )
        })
    }

    /// Computes the required workspace size and alignment for a dense solve in place using an LLT,
    /// LDLT or intranodal Bunch-Kaufman factorization.
    pub fn solve_in_place_req<E: Entity>(
        &self,
        rhs_ncols: usize,
    ) -> Result<StackReq, SizeOverflow> {
        temp_mat_req::<E>(self.nrows(), rhs_ncols)?.try_and(match self.raw() {
            SymbolicCholeskyRaw::Simplicial(this) => this.solve_in_place_req::<E>(rhs_ncols)?,
            SymbolicCholeskyRaw::Supernodal(this) => this.solve_in_place_req::<E>(rhs_ncols)?,
        })
    }
}

/// Sparse LLT factorization wrapper.
#[derive(Debug)]
pub struct LltRef<'a, I: Index, E: Entity> {
    symbolic: &'a SymbolicCholesky<I>,
    values: SliceGroup<'a, E>,
}

/// Sparse LDLT factorization wrapper.
#[derive(Debug)]
pub struct LdltRef<'a, I: Index, E: Entity> {
    symbolic: &'a SymbolicCholesky<I>,
    values: SliceGroup<'a, E>,
}

/// Sparse intranodal Bunch-Kaufman factorization wrapper.
#[derive(Debug)]
pub struct IntranodeBunchKaufmanRef<'a, I: Index, E: Entity> {
    symbolic: &'a SymbolicCholesky<I>,
    values: SliceGroup<'a, E>,
    subdiag: SliceGroup<'a, E>,
    perm: PermRef<'a, I>,
}

impl<'a, I: Index, E: Entity> core::ops::Deref for LltRef<'a, I, E> {
    type Target = SymbolicCholesky<I>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.symbolic
    }
}
impl<'a, I: Index, E: Entity> core::ops::Deref for LdltRef<'a, I, E> {
    type Target = SymbolicCholesky<I>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.symbolic
    }
}
impl<'a, I: Index, E: Entity> core::ops::Deref for IntranodeBunchKaufmanRef<'a, I, E> {
    type Target = SymbolicCholesky<I>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.symbolic
    }
}

impl_copy!(<'a><I: Index><supernodal::SymbolicSupernodeRef<'a, I>>);
impl_copy!(<'a><I: Index, E: Entity><supernodal::SupernodeRef<'a, I, E>>);

impl_copy!(<'a><I: Index, E: Entity><simplicial::SimplicialLdltRef<'a, I, E>>);
impl_copy!(<'a><I: Index, E: Entity><simplicial::SimplicialLltRef<'a, I, E>>);

impl_copy!(<'a><I: Index, E: Entity><supernodal::SupernodalLltRef<'a, I, E>>);
impl_copy!(<'a><I: Index, E: Entity><supernodal::SupernodalLdltRef<'a, I, E>>);
impl_copy!(<'a><I: Index, E: Entity><supernodal::SupernodalIntranodeBunchKaufmanRef<'a, I, E>>);

impl_copy!(<'a><I: Index, E: Entity><IntranodeBunchKaufmanRef<'a, I, E>>);
impl_copy!(<'a><I: Index, E: Entity><LdltRef<'a, I, E>>);
impl_copy!(<'a><I: Index, E: Entity><LltRef<'a, I, E>>);

impl<'a, I: Index, E: Entity> IntranodeBunchKaufmanRef<'a, I, E> {
    /// Creates a new Cholesky intranodal Bunch-Kaufman factor from the symbolic part and
    /// numerical values, as well as the pivoting permutation.
    ///
    /// # Panics
    /// - Panics if `values.len() != symbolic.len_values()`.
    /// - Panics if `subdiag.len() != symbolic.nrows()`.
    /// - Panics if `perm.len() != symbolic.nrows()`.
    #[inline]
    pub fn new(
        symbolic: &'a SymbolicCholesky<I>,
        values: GroupFor<E, &'a [E::Unit]>,
        subdiag: GroupFor<E, &'a [E::Unit]>,
        perm: PermRef<'a, I>,
    ) -> Self {
        let values = SliceGroup::<'_, E>::new(values);
        let subdiag = SliceGroup::<'_, E>::new(subdiag);
        assert!(all(
            values.len() == symbolic.len_values(),
            subdiag.len() == symbolic.nrows(),
            perm.len() == symbolic.nrows(),
        ));
        Self {
            symbolic,
            values,
            subdiag,
            perm,
        }
    }

    /// Returns the symbolic part of the Cholesky factor.
    #[inline]
    pub fn symbolic(self) -> &'a SymbolicCholesky<I> {
        self.symbolic
    }

    /// Solves the equation `Op(A) x = rhs` and stores the result in `rhs`, where `Op` is either
    /// the identity or the conjugate, depending on the value of `conj`.
    ///
    /// # Panics
    /// Panics if `rhs.nrows() != self.symbolic().nrows()`.
    pub fn solve_in_place_with_conj(
        &self,
        conj: Conj,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) where
        E: ComplexField,
    {
        let k = rhs.ncols();
        let n = self.symbolic.nrows();

        let mut rhs = rhs;

        let (mut x, stack) = temp_mat_uninit::<E>(n, k, stack);
        let (fwd, inv) = self.symbolic.perm().arrays();

        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(symbolic) => {
                let this = simplicial::SimplicialLdltRef::new(symbolic, self.values.into_inner());

                for j in 0..k {
                    for (i, fwd) in fwd.iter().enumerate() {
                        x.write(i, j, rhs.read(fwd.zx().zx(), j));
                    }
                }
                this.solve_in_place_with_conj(conj, x.rb_mut(), parallelism, stack);
                for j in 0..k {
                    for (i, inv) in inv.iter().enumerate() {
                        rhs.write(i, j, x.read(inv.zx().zx(), j));
                    }
                }
            }
            SymbolicCholeskyRaw::Supernodal(symbolic) => {
                let (dyn_fwd, dyn_inv) = self.perm.arrays();
                for j in 0..k {
                    for (i, dyn_fwd) in dyn_fwd.iter().enumerate() {
                        x.write(i, j, rhs.read(fwd[dyn_fwd.zx()].zx(), j));
                    }
                }

                let this = supernodal::SupernodalIntranodeBunchKaufmanRef::new(
                    symbolic,
                    self.values.into_inner(),
                    self.subdiag.into_inner(),
                    self.perm,
                );
                this.solve_in_place_no_numeric_permute_with_conj(
                    conj,
                    x.rb_mut(),
                    parallelism,
                    stack,
                );

                for j in 0..k {
                    for (i, inv) in inv.iter().enumerate() {
                        rhs.write(i, j, x.read(dyn_inv[inv.zx()].zx(), j));
                    }
                }
            }
        }
    }
}

impl<'a, I: Index, E: Entity> LltRef<'a, I, E> {
    /// Creates a new Cholesky LLT factor from the symbolic part and
    /// numerical values.
    ///
    /// # Panics
    /// - Panics if `values.len() != symbolic.len_values()`.
    #[inline]
    pub fn new(symbolic: &'a SymbolicCholesky<I>, values: GroupFor<E, &'a [E::Unit]>) -> Self {
        let values = SliceGroup::<'_, E>::new(values);
        assert!(symbolic.len_values() == values.len());
        Self { symbolic, values }
    }

    /// Returns the symbolic part of the Cholesky factor.
    #[inline]
    pub fn symbolic(self) -> &'a SymbolicCholesky<I> {
        self.symbolic
    }

    /// Solves the equation `Op(A) x = rhs` and stores the result in `rhs`, where `Op` is either
    /// the identity or the conjugate, depending on the value of `conj`.
    ///
    /// # Panics
    /// Panics if `rhs.nrows() != self.symbolic().nrows()`.
    pub fn solve_in_place_with_conj(
        &self,
        conj: Conj,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) where
        E: ComplexField,
    {
        let k = rhs.ncols();
        let n = self.symbolic.nrows();

        let mut rhs = rhs;

        let (mut x, stack) = temp_mat_uninit::<E>(n, k, stack);

        let (fwd, inv) = self.symbolic.perm().arrays();
        for j in 0..k {
            for (i, fwd) in fwd.iter().enumerate() {
                x.write(i, j, rhs.read(fwd.zx(), j));
            }
        }

        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(symbolic) => {
                let this = simplicial::SimplicialLltRef::new(symbolic, self.values.into_inner());
                this.solve_in_place_with_conj(conj, x.rb_mut(), parallelism, stack);
            }
            SymbolicCholeskyRaw::Supernodal(symbolic) => {
                let this = supernodal::SupernodalLltRef::new(symbolic, self.values.into_inner());
                this.solve_in_place_with_conj(conj, x.rb_mut(), parallelism, stack);
            }
        }

        for j in 0..k {
            for (i, inv) in inv.iter().enumerate() {
                rhs.write(i, j, x.read(inv.zx(), j));
            }
        }
    }
}

impl<'a, I: Index, E: Entity> LdltRef<'a, I, E> {
    /// Creates new Cholesky LDLT factors from the symbolic part and
    /// numerical values.
    ///
    /// # Panics
    /// - Panics if `values.len() != symbolic.len_values()`.
    #[inline]
    pub fn new(symbolic: &'a SymbolicCholesky<I>, values: GroupFor<E, &'a [E::Unit]>) -> Self {
        let values = SliceGroup::<'_, E>::new(values);
        assert!(symbolic.len_values() == values.len());
        Self { symbolic, values }
    }

    /// Returns the symbolic part of the Cholesky factor.
    #[inline]
    pub fn symbolic(self) -> &'a SymbolicCholesky<I> {
        self.symbolic
    }

    /// Solves the equation `Op(A) x = rhs` and stores the result in `rhs`, where `Op` is either
    /// the identity or the conjugate, depending on the value of `conj`.
    ///
    /// # Panics
    /// Panics if `rhs.nrows() != self.symbolic().nrows()`.
    pub fn solve_in_place_with_conj(
        &self,
        conj: Conj,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) where
        E: ComplexField,
    {
        let k = rhs.ncols();
        let n = self.symbolic.nrows();

        let mut rhs = rhs;

        let (mut x, stack) = temp_mat_uninit::<E>(n, k, stack);

        let (fwd, inv) = self.symbolic.perm().arrays();
        for j in 0..k {
            for (i, fwd) in fwd.iter().enumerate() {
                x.write(i, j, rhs.read(fwd.zx(), j));
            }
        }

        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(symbolic) => {
                let this = simplicial::SimplicialLdltRef::new(symbolic, self.values.into_inner());
                this.solve_in_place_with_conj(conj, x.rb_mut(), parallelism, stack);
            }
            SymbolicCholeskyRaw::Supernodal(symbolic) => {
                let this = supernodal::SupernodalLdltRef::new(symbolic, self.values.into_inner());
                this.solve_in_place_with_conj(conj, x.rb_mut(), parallelism, stack);
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
    next_child: &Array<'n, I::Signed>,
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
            *first_child = MaybeIdx::new_checked(next_child[current_child], N);
        } else {
            post[N.check(start_index)] = I::truncate(current_node);
            start_index += 1;
            top -= 1;
        }
    }
    start_index
}

pub(crate) fn ghost_postorder<'n, I: Index>(
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
    let (first_child, stack) = stack.make_raw::<I::Signed>(n);
    let (next_child, _) = stack.make_raw::<I::Signed>(n);

    let stack = Array::from_mut(stack_, N);
    let next_child = Array::from_mut(next_child, N);
    let first_child = Array::from_mut(ghost::fill_none::<I>(first_child, N), N);

    for j in N.indices().rev() {
        let parent = etree[j];
        if let Some(parent) = parent.idx() {
            let first = &mut first_child[parent.zx()];
            next_child[j] = **first;
            *first = MaybeIdx::from_index(j.truncate::<I>());
        }
    }

    let mut start_index = 0usize;
    for (root, &parent) in etree.as_ref().iter().enumerate() {
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

/// Tuning parameters for the symbolic Cholesky factorization.
#[derive(Copy, Clone, Debug, Default)]
pub struct CholeskySymbolicParams<'a> {
    /// Parameters for computing the fill-reducing permutation.
    pub amd_params: Control,
    /// Threshold for selecting the supernodal factorization.
    pub supernodal_flop_ratio_threshold: SupernodalThreshold,
    /// Supernodal factorization parameters.
    pub supernodal_params: SymbolicSupernodalParams<'a>,
}

/// Computes the symbolic Cholesky factorization of the matrix `A`, or returns an error if the
/// operation could not be completed.
pub fn factorize_symbolic_cholesky<I: Index>(
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
                        supernodal::factorize_supernodal_symbolic_cholesky_req::<I>(n)?,
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
            A.into_inner(),
            params.amd_params,
            stack.rb_mut(),
        )?;
        let flops = flops.n_div + flops.n_mult_subs_ldl;
        let perm_ = ghost::PermRef::new(PermRef::new_checked(&perm_fwd, &perm_inv), N);

        let (new_col_ptr, stack) = stack.make_raw::<I>(n + 1);
        let (new_row_ind, mut stack) = stack.make_raw::<I>(A_nnz);
        let A = unsafe {
            ghost_permute_hermitian_unsorted_symbolic(
                new_col_ptr,
                new_row_ind,
                A,
                perm_,
                side,
                Side::Upper,
                stack.rb_mut(),
            )
        };

        let (etree, stack) = stack.make_raw::<I::Signed>(n);
        let (col_counts, mut stack) = stack.make_raw::<I>(n);
        let etree = Array::from_mut(etree, N);
        let col_counts = Array::from_mut(col_counts, N);
        let etree =
            &*ghost_prefactorize_symbolic_cholesky::<I>(etree, col_counts, A, stack.rb_mut());
        let L_nnz = I::sum_nonnegative(col_counts.as_ref()).ok_or(FaerError::IndexOverflow)?;

        let raw = if (flops / L_nnz.zx() as f64)
            > params.supernodal_flop_ratio_threshold.0
                * crate::sparse::linalg::CHOLESKY_SUPERNODAL_RATIO_FACTOR
        {
            SymbolicCholeskyRaw::Supernodal(supernodal::ghost_factorize_supernodal_symbolic(
                A,
                None,
                None,
                supernodal::CholeskyInput::A,
                etree,
                col_counts,
                stack.rb_mut(),
                params.supernodal_params,
            )?)
        } else {
            SymbolicCholeskyRaw::Simplicial(
                simplicial::ghost_factorize_simplicial_symbolic_cholesky(
                    A,
                    etree,
                    col_counts,
                    stack.rb_mut(),
                )?,
            )
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
pub(crate) mod tests {
    use super::{supernodal::SupernodalLdltRef, *};
    use crate::{
        assert,
        sparse::linalg::{
            cholesky::supernodal::{CholeskyInput, SupernodalIntranodeBunchKaufmanRef},
            qd::Double,
        },
        Mat,
    };
    use dyn_stack::GlobalPodBuffer;
    use num_complex::Complex;
    use rand::{Rng, SeedableRng};

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

        let A = SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_ind);
        let zero = truncate(0);
        let mut etree = vec![zero.to_signed(); n];
        let mut col_count = vec![zero; n];
        ghost::with_size(n, |N| {
            let A = ghost::SymbolicSparseColMatRef::new(A, N, N);
            let etree = ghost_prefactorize_symbolic_cholesky(
                Array::from_mut(&mut etree, N),
                Array::from_mut(&mut col_count, N),
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * n))),
            );

            supernodal::ghost_factorize_supernodal_symbolic(
                A,
                None,
                None,
                CholeskyInput::A,
                etree,
                Array::from_ref(&col_count, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
                Default::default(),
            )
            .unwrap();
        });
        assert_eq!(
            etree,
            [5, 2, 7, 5, 7, 6, 8, 9, 9, 10, NONE].map(I::Signed::truncate)
        );
        assert_eq!(col_count, [3, 3, 4, 3, 3, 4, 4, 3, 3, 2, 1].map(truncate));
    }

    include!("./data.rs");

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
            let A = SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_ind);

            let perm = &mut vec![I(0); n];
            let perm_inv = &mut vec![I(0); n];

            crate::sparse::linalg::amd::order_maybe_unsorted(
                perm,
                perm_inv,
                A,
                Default::default(),
                PodStack::new(&mut GlobalPodBuffer::new(
                    crate::sparse::linalg::amd::order_maybe_unsorted_req::<I>(n, row_ind.len())
                        .unwrap(),
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

    fn reconstruct_from_supernodal_llt<I: Index, E: ComplexField>(
        symbolic: &supernodal::SymbolicSupernodalCholesky<I>,
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

    fn reconstruct_from_supernodal_ldlt<I: Index, E: ComplexField>(
        symbolic: &supernodal::SymbolicSupernodalCholesky<I>,
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

        let mut D = Mat::<E>::zeros(n, n);
        zipped!(
            D.as_mut().diagonal_mut().column_vector_mut().as_2d_mut(),
            dense.as_ref().diagonal().column_vector().as_2d()
        )
        .for_each(|unzipped!(mut dst, src)| dst.write(src.read().faer_inv()));
        dense
            .as_mut()
            .diagonal_mut()
            .column_vector_mut()
            .fill(E::faer_one());
        &dense * D * dense.adjoint()
    }

    fn reconstruct_from_simplicial_llt<'a, I: Index, E: ComplexField>(
        symbolic: &'a simplicial::SymbolicSimplicialCholesky<I>,
        L_values: GroupFor<E, &'a [E::Unit]>,
    ) -> Mat<E> {
        let slice_group = SliceGroup::<'_, E>::new;
        let L_values = slice_group(L_values);
        let n = symbolic.nrows();
        let mut dense = Mat::<E>::zeros(n, n);

        let L = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_unsorted_checked(
                n,
                n,
                symbolic.col_ptrs(),
                None,
                symbolic.row_indices(),
            ),
            L_values.into_inner(),
        );

        for j in 0..n {
            for (i, val) in zip(
                L.row_indices_of_col(j),
                slice_group(L.values_of_col(j)).into_ref_iter(),
            ) {
                dense.write(i, j, val.read());
            }
        }

        &dense * dense.adjoint()
    }

    fn reconstruct_from_simplicial_ldlt<'a, I: Index, E: ComplexField>(
        symbolic: &'a simplicial::SymbolicSimplicialCholesky<I>,
        L_values: GroupFor<E, &'a [E::Unit]>,
    ) -> Mat<E> {
        let slice_group = SliceGroup::<'_, E>::new;
        let L_values = slice_group(L_values);
        let n = symbolic.nrows();
        let mut dense = Mat::<E>::zeros(n, n);

        let L = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_unsorted_checked(
                n,
                n,
                symbolic.col_ptrs(),
                None,
                symbolic.row_indices(),
            ),
            L_values.into_inner(),
        );

        for j in 0..n {
            for (i, val) in zip(
                L.row_indices_of_col(j),
                slice_group(L.values_of_col(j)).into_ref_iter(),
            ) {
                dense.write(i, j, val.read());
            }
        }

        let mut D = Mat::<E>::zeros(n, n);
        D.as_mut()
            .diagonal_mut()
            .column_vector_mut()
            .copy_from(dense.as_ref().diagonal().column_vector());
        dense
            .as_mut()
            .diagonal_mut()
            .column_vector_mut()
            .fill(E::faer_one());

        &dense * D * dense.adjoint()
    }

    fn test_supernodal<I: Index>() {
        type E = Complex<Double<f64>>;
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
            crate::Mat::<E>::from_fn(nnz, 1, |i, _| complexify(E::faer_from_f64(values[i])));
        let values = values_mat.col_as_slice(0);

        let A = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_ind),
            values,
        );
        let zero = truncate(0);
        let mut etree = vec![zero.to_signed(); n];
        let mut col_count = vec![zero; n];
        ghost::with_size(n, |N| {
            let A = ghost::SparseColMatRef::new(A, N, N);
            let etree = ghost_prefactorize_symbolic_cholesky(
                Array::from_mut(&mut etree, N),
                Array::from_mut(&mut col_count, N),
                *A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * n))),
            );

            let symbolic = supernodal::ghost_factorize_supernodal_symbolic(
                *A,
                None,
                None,
                CholeskyInput::A,
                etree,
                Array::from_ref(&col_count, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
                Default::default(),
            )
            .unwrap();

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values_mat.clone();
            let mut A_lower_row_ind = row_ind.to_vec();
            let A_lower_values = SliceGroupMut::new(A_lower_values.col_as_slice_mut(0));
            let A_lower = crate::sparse::utils::ghost_adjoint(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                A_lower_values,
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            );
            let mut values = crate::Mat::<E>::zeros(symbolic.len_values(), 1);

            supernodal::factorize_supernodal_numeric_ldlt(
                values.col_as_slice_mut(0),
                A_lower.into_inner().into_const(),
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
            let mut A = sparse_to_dense(A.into_inner());
            for j in 0..n {
                for i in j + 1..n {
                    A.write(i, j, A.read(j, i).faer_conj());
                }
            }

            let err =
                reconstruct_from_supernodal_ldlt::<I, E>(&symbolic, values.col_as_slice(0)) - A;
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

    fn test_supernodal_ldlt<I: Index>() {
        type E = Complex<Double<f64>>;
        let truncate = I::truncate;

        let (_, col_ptr, row_ind, values) = MEDIUM;

        let mut gen = rand::rngs::StdRng::seed_from_u64(0);
        let i = E::faer_one().faer_neg().faer_sqrt();
        let mut complexify = |e: E| {
            if e == E::faer_from_f64(1.0) {
                e.faer_add(i.faer_mul(E::faer_from_f64(2000.0 * gen.gen::<f64>())))
                    .faer_add(E::faer_from_f64(2000.0 * gen.gen::<f64>()))
            } else {
                e.faer_add(E::faer_from_f64(100.0 * gen.gen::<f64>()))
            }
        };

        let n = col_ptr.len() - 1;
        let nnz = values.len();
        let col_ptr = &*col_ptr.iter().copied().map(truncate).collect::<Vec<_>>();
        let row_ind = &*row_ind.iter().copied().map(truncate).collect::<Vec<_>>();
        let values_mat =
            crate::Mat::<E>::from_fn(nnz, 1, |i, _| complexify(E::faer_from_f64(values[i])));
        let values = values_mat.col_as_slice(0);

        let A = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_ind),
            values,
        );
        let mut A_dense = sparse_to_dense(A);
        for j in 0..n {
            for i in j + 1..n {
                A_dense.write(i, j, A_dense.read(j, i).faer_conj());
            }
        }

        let zero = truncate(0);
        let mut etree = vec![zero.to_signed(); n];
        let mut col_count = vec![zero; n];
        ghost::with_size(n, |N| {
            let A = ghost::SparseColMatRef::new(A, N, N);
            let etree = ghost_prefactorize_symbolic_cholesky(
                Array::from_mut(&mut etree, N),
                Array::from_mut(&mut col_count, N),
                *A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * n))),
            );

            let symbolic = supernodal::ghost_factorize_supernodal_symbolic(
                *A,
                None,
                None,
                CholeskyInput::A,
                etree,
                Array::from_ref(&col_count, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
                Default::default(),
            )
            .unwrap();

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values_mat.clone();
            let mut A_lower_row_ind = row_ind.to_vec();
            let A_lower_values = SliceGroupMut::new(A_lower_values.col_as_slice_mut(0));
            let A_lower = crate::sparse::utils::ghost_adjoint(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                A_lower_values,
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            );
            let mut values = crate::Mat::<E>::zeros(symbolic.len_values(), 1);

            supernodal::factorize_supernodal_numeric_ldlt(
                values.col_as_slice_mut(0),
                A_lower.into_inner().into_const(),
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
            let k = 2;

            let rhs = Mat::<E>::from_fn(n, k, |_, _| {
                E::faer_from_f64(gen.gen()).faer_add(i.faer_mul(E::faer_from_f64(gen.gen())))
            });
            for conj in [Conj::Yes, Conj::No] {
                let mut x = rhs.clone();
                let ldlt = SupernodalLdltRef::new(&symbolic, values.col_as_slice(0));
                ldlt.solve_in_place_with_conj(
                    conj,
                    x.as_mut(),
                    Parallelism::None,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        symbolic.solve_in_place_req::<E>(k).unwrap(),
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
        });
    }

    fn test_supernodal_intranode_bk_1<I: Index>() {
        type E = Complex<f64>;
        let truncate = I::truncate;

        let (_, col_ptr, row_ind, values) = MEDIUM;

        let mut gen = rand::rngs::StdRng::seed_from_u64(0);
        let i = E::faer_one().faer_neg().faer_sqrt();

        let n = col_ptr.len() - 1;
        let nnz = values.len();
        let col_ptr = &*col_ptr.iter().copied().map(truncate).collect::<Vec<_>>();
        let row_ind = &*row_ind.iter().copied().map(truncate).collect::<Vec<_>>();

        let mut complexify = |e: E| {
            let i = E::faer_one().faer_neg().faer_sqrt();
            if e == E::faer_from_f64(1.0) {
                e.faer_add(i.faer_mul(E::faer_from_f64(1000.0 * gen.gen::<f64>())))
            } else {
                e.faer_add(E::faer_from_f64(1000.0 * gen.gen::<f64>()))
            }
        };
        let values_mat =
            crate::Mat::<E>::from_fn(nnz, 1, |i, _| complexify(E::faer_from_f64(values[i])));
        let values = values_mat.col_as_slice(0);

        let A = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_ind),
            values,
        );
        let mut A_dense = sparse_to_dense(A);
        for j in 0..n {
            for i in j + 1..n {
                A_dense.write(i, j, A_dense.read(j, i).faer_conj());
            }
        }

        let zero = truncate(0);
        let mut etree = vec![zero.to_signed(); n];
        let mut col_count = vec![zero; n];
        ghost::with_size(n, |N| {
            let A = ghost::SparseColMatRef::new(A, N, N);
            let etree = ghost_prefactorize_symbolic_cholesky(
                Array::from_mut(&mut etree, N),
                Array::from_mut(&mut col_count, N),
                *A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * n))),
            );

            let symbolic = supernodal::ghost_factorize_supernodal_symbolic(
                *A,
                None,
                None,
                CholeskyInput::A,
                etree,
                Array::from_ref(&col_count, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
                Default::default(),
            )
            .unwrap();

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values_mat.clone();
            let mut A_lower_row_ind = row_ind.to_vec();
            let A_lower_values = SliceGroupMut::new(A_lower_values.col_as_slice_mut(0));
            let A_lower = crate::sparse::utils::ghost_adjoint(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                A_lower_values,
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            );
            let mut values = crate::Mat::<E>::zeros(symbolic.len_values(), 1);

            let mut fwd = vec![zero; n];
            let mut inv = vec![zero; n];
            let mut subdiag = Mat::<E>::zeros(n, 1);

            supernodal::factorize_supernodal_numeric_intranode_bunch_kaufman(
                values.col_as_slice_mut(0),
                subdiag.col_as_slice_mut(0),
                &mut fwd,
                &mut inv,
                A_lower.into_inner().into_const(),
                Default::default(),
                &symbolic,
                Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(
                    supernodal::factorize_supernodal_numeric_intranode_bunch_kaufman_req::<I, E>(
                        &symbolic,
                        Parallelism::None,
                    )
                    .unwrap(),
                )),
            );
            let k = 2;

            let rhs = Mat::<E>::from_fn(n, k, |_, _| {
                E::faer_from_f64(gen.gen()).faer_add(i.faer_mul(E::faer_from_f64(gen.gen())))
            });
            for conj in [Conj::Yes, Conj::No] {
                let mut x = rhs.clone();
                let lblt = SupernodalIntranodeBunchKaufmanRef::new(
                    &symbolic,
                    values.col_as_slice(0),
                    subdiag.col_as_slice(0),
                    PermRef::new_checked(&fwd, &inv),
                );
                crate::perm::permute_rows_in_place(
                    x.as_mut(),
                    lblt.perm,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        crate::perm::permute_rows_in_place_req::<I, E>(n, k).unwrap(),
                    )),
                );
                lblt.solve_in_place_no_numeric_permute_with_conj(
                    conj,
                    x.as_mut(),
                    Parallelism::None,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        symbolic.solve_in_place_req::<E>(k).unwrap(),
                    )),
                );
                crate::perm::permute_rows_in_place(
                    x.as_mut(),
                    lblt.perm.inverse(),
                    PodStack::new(&mut GlobalPodBuffer::new(
                        crate::perm::permute_rows_in_place_req::<I, E>(n, k).unwrap(),
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
                assert!(max < <E as ComplexField>::Real::faer_from_f64(1e-10));
            }
        });
    }

    fn test_supernodal_intranode_bk_2<I: Index>() {
        type E = Complex<f64>;
        let truncate = I::truncate;

        let (_, col_ptr, row_ind, values) = MEDIUM_P;

        let mut gen = rand::rngs::StdRng::seed_from_u64(0);
        let i = E::faer_one().faer_neg().faer_sqrt();

        let n = col_ptr.len() - 1;
        let nnz = values.len();
        let col_ptr = &*col_ptr.iter().copied().map(truncate).collect::<Vec<_>>();
        let row_ind = &*row_ind.iter().copied().map(truncate).collect::<Vec<_>>();
        let values_mat = crate::Mat::<E>::from_fn(nnz, 1, |i, _| values[i]);
        let values = values_mat.col_as_slice(0);

        let A = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_ind),
            values,
        );
        let mut A_dense = sparse_to_dense(A);
        for j in 0..n {
            for i in j + 1..n {
                A_dense.write(i, j, A_dense.read(j, i).faer_conj());
            }
        }

        let zero = truncate(0);
        let mut etree = vec![zero.to_signed(); n];
        let mut col_count = vec![zero; n];
        ghost::with_size(n, |N| {
            let A = ghost::SparseColMatRef::new(A, N, N);
            let etree = ghost_prefactorize_symbolic_cholesky(
                Array::from_mut(&mut etree, N),
                Array::from_mut(&mut col_count, N),
                *A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * n))),
            );

            let symbolic = supernodal::ghost_factorize_supernodal_symbolic(
                *A,
                None,
                None,
                CholeskyInput::A,
                etree,
                Array::from_ref(&col_count, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
                Default::default(),
            )
            .unwrap();

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values_mat.clone();
            let mut A_lower_row_ind = row_ind.to_vec();
            let A_lower_values = SliceGroupMut::new(A_lower_values.col_as_slice_mut(0));
            let A_lower = crate::sparse::utils::ghost_adjoint(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                A_lower_values,
                A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            );
            let mut values = crate::Mat::<E>::zeros(symbolic.len_values(), 1);

            let mut fwd = vec![zero; n];
            let mut inv = vec![zero; n];
            let mut subdiag = Mat::<E>::zeros(n, 1);

            supernodal::factorize_supernodal_numeric_intranode_bunch_kaufman(
                values.col_as_slice_mut(0),
                subdiag.col_as_slice_mut(0),
                &mut fwd,
                &mut inv,
                A_lower.into_inner().into_const(),
                Default::default(),
                &symbolic,
                Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(
                    supernodal::factorize_supernodal_numeric_intranode_bunch_kaufman_req::<I, E>(
                        &symbolic,
                        Parallelism::None,
                    )
                    .unwrap(),
                )),
            );
            let k = 2;

            let rhs = Mat::<E>::from_fn(n, k, |_, _| {
                E::faer_from_f64(gen.gen()).faer_add(i.faer_mul(E::faer_from_f64(gen.gen())))
            });
            for conj in [Conj::Yes, Conj::No] {
                let mut x = rhs.clone();
                let lblt = SupernodalIntranodeBunchKaufmanRef::new(
                    &symbolic,
                    values.col_as_slice(0),
                    subdiag.col_as_slice(0),
                    PermRef::new_checked(&fwd, &inv),
                );
                crate::perm::permute_rows_in_place(
                    x.as_mut(),
                    lblt.perm,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        crate::perm::permute_rows_in_place_req::<I, E>(n, k).unwrap(),
                    )),
                );
                lblt.solve_in_place_no_numeric_permute_with_conj(
                    conj,
                    x.as_mut(),
                    Parallelism::None,
                    PodStack::new(&mut GlobalPodBuffer::new(
                        symbolic.solve_in_place_req::<E>(k).unwrap(),
                    )),
                );
                crate::perm::permute_rows_in_place(
                    x.as_mut(),
                    lblt.perm.inverse(),
                    PodStack::new(&mut GlobalPodBuffer::new(
                        crate::perm::permute_rows_in_place_req::<I, E>(n, k).unwrap(),
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
                assert!(max < <E as ComplexField>::Real::faer_from_f64(1e-10));
            }
        });
    }

    fn test_simplicial<I: Index>() {
        type E = Complex<Double<f64>>;
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
            crate::Mat::<E>::from_fn(nnz, 1, |i, _| complexify(E::faer_from_f64(values[i])));
        let values = values_mat.col_as_slice(0);

        let A = SparseColMatRef::<'_, I, E>::new(
            SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_ind),
            values,
        );
        let zero = truncate(0);
        let mut etree = vec![zero.to_signed(); n];
        let mut col_count = vec![zero; n];
        ghost::with_size(n, |N| {
            let A = ghost::SparseColMatRef::new(A, N, N);
            let etree = ghost_prefactorize_symbolic_cholesky(
                Array::from_mut(&mut etree, N),
                Array::from_mut(&mut col_count, N),
                *A,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(5 * n))),
            );

            let symbolic = simplicial::ghost_factorize_simplicial_symbolic_cholesky(
                *A,
                etree,
                Array::from_ref(&col_count, N),
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            )
            .unwrap();

            let mut values = crate::Mat::<E>::zeros(symbolic.len_values(), 1);

            simplicial::factorize_simplicial_numeric_ldlt::<I, E>(
                values.col_as_slice_mut(0),
                A.into_inner(),
                Default::default(),
                &symbolic,
                PodStack::new(&mut GlobalPodBuffer::new(
                    simplicial::factorize_simplicial_numeric_ldlt_req::<I, E>(n).unwrap(),
                )),
            );
            let mut A = sparse_to_dense(A.into_inner());
            for j in 0..n {
                for i in j + 1..n {
                    A.write(i, j, A.read(j, i).faer_conj());
                }
            }

            let err =
                reconstruct_from_simplicial_ldlt::<I, E>(&symbolic, values.col_as_slice(0)) - &A;

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

    fn test_solver_llt<I: Index>() {
        type E = Complex<Double<f64>>;
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
            let values_mat =
                crate::Mat::<E>::from_fn(nnz, 1, |i, _| complexify(E::faer_from_f64(values[i])));
            let values = values_mat.col_as_slice(0);

            let A_upper = SparseColMatRef::<'_, I, E>::new(
                SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_ind),
                values,
            );

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values_mat.clone();
            let mut A_lower_row_ind = row_ind.to_vec();
            let A_lower = crate::sparse::utils::adjoint(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                A_lower_values.col_as_slice_mut(0),
                A_upper,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            )
            .into_const();

            let mut A_dense = sparse_to_dense(A_upper);
            for j in 0..n {
                for i in j + 1..n {
                    A_dense.write(i, j, A_dense.read(j, i).faer_conj());
                }
            }

            for (A, side, supernodal_flop_ratio_threshold, parallelism) in [
                (
                    A_upper,
                    Side::Upper,
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    Parallelism::None,
                ),
                (
                    A_upper,
                    Side::Upper,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                    Parallelism::None,
                ),
                (
                    A_lower,
                    Side::Lower,
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    Parallelism::None,
                ),
                (
                    A_lower,
                    Side::Lower,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                    Parallelism::None,
                ),
            ] {
                let symbolic = factorize_symbolic_cholesky(
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

                symbolic
                    .factorize_numeric_llt::<E>(
                        L_values.col_as_slice_mut(0),
                        A,
                        side,
                        Default::default(),
                        parallelism,
                        PodStack::new(&mut mem),
                    )
                    .unwrap();
                let L_values = L_values.col_as_slice(0);

                let A_reconstructed = match symbolic.raw() {
                    SymbolicCholeskyRaw::Simplicial(symbolic) => {
                        reconstruct_from_simplicial_llt::<I, E>(symbolic, L_values)
                    }
                    SymbolicCholeskyRaw::Supernodal(symbolic) => {
                        reconstruct_from_supernodal_llt::<I, E>(symbolic, L_values)
                    }
                };

                let (perm_fwd, _) = symbolic.perm().arrays();

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

                for k in (1..16).chain(128..132) {
                    let rhs = Mat::<E>::from_fn(n, k, |_, _| {
                        E::faer_from_f64(gen.gen())
                            .faer_add(i.faer_mul(E::faer_from_f64(gen.gen())))
                    });
                    for conj in [Conj::Yes, Conj::No] {
                        let mut x = rhs.clone();
                        let llt = LltRef::new(&symbolic, L_values);
                        llt.solve_in_place_with_conj(
                            conj,
                            x.as_mut(),
                            parallelism,
                            PodStack::new(&mut GlobalPodBuffer::new(
                                symbolic.solve_in_place_req::<E>(k).unwrap(),
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

    fn test_solver_ldlt<I: Index>() {
        type E = Complex<Double<f64>>;
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
            let values_mat =
                crate::Mat::<E>::from_fn(nnz, 1, |i, _| complexify(E::faer_from_f64(values[i])));
            let values = values_mat.col_as_slice(0);

            let A_upper = SparseColMatRef::<'_, I, E>::new(
                SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_ind),
                values,
            );

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values_mat.clone();
            let mut A_lower_row_ind = row_ind.to_vec();
            let A_lower = crate::sparse::utils::adjoint(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                A_lower_values.col_as_slice_mut(0),
                A_upper,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            )
            .into_const();

            let mut A_dense = sparse_to_dense(A_upper);
            for j in 0..n {
                for i in j + 1..n {
                    A_dense.write(i, j, A_dense.read(j, i).faer_conj());
                }
            }

            for (A, side, supernodal_flop_ratio_threshold, parallelism) in [
                (
                    A_upper,
                    Side::Upper,
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    Parallelism::None,
                ),
                (
                    A_upper,
                    Side::Upper,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                    Parallelism::None,
                ),
                (
                    A_lower,
                    Side::Lower,
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    Parallelism::None,
                ),
                (
                    A_lower,
                    Side::Lower,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                    Parallelism::None,
                ),
            ] {
                let symbolic = factorize_symbolic_cholesky(
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

                symbolic.factorize_numeric_ldlt::<E>(
                    L_values.col_as_slice_mut(0),
                    A,
                    side,
                    Default::default(),
                    parallelism,
                    PodStack::new(&mut mem),
                );
                let L_values = L_values.col_as_slice(0);
                let A_reconstructed = match symbolic.raw() {
                    SymbolicCholeskyRaw::Simplicial(symbolic) => {
                        reconstruct_from_simplicial_ldlt::<I, E>(symbolic, L_values)
                    }
                    SymbolicCholeskyRaw::Supernodal(symbolic) => {
                        reconstruct_from_supernodal_ldlt::<I, E>(symbolic, L_values)
                    }
                };

                let (perm_fwd, _) = symbolic.perm().arrays();

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
                        ldlt.solve_in_place_with_conj(
                            conj,
                            x.as_mut(),
                            parallelism,
                            PodStack::new(&mut GlobalPodBuffer::new(
                                symbolic.solve_in_place_req::<E>(k).unwrap(),
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

    fn test_solver_intranode_bk<I: Index>() {
        type E = Complex<Double<f64>>;
        let truncate = I::truncate;

        for (_, col_ptr, row_ind, values) in [MEDIUM, SMALL] {
            let mut gen = rand::rngs::StdRng::seed_from_u64(0);
            let i = E::faer_one().faer_neg().faer_sqrt();
            let mut complexify = |e: E| {
                if e == E::faer_from_f64(1.0) {
                    e.faer_add(i.faer_mul(E::faer_from_f64(2000.0 * gen.gen::<f64>())))
                        .faer_add(E::faer_from_f64(2000.0 * gen.gen::<f64>()))
                } else {
                    e.faer_add(E::faer_from_f64(100.0 * gen.gen::<f64>()))
                }
            };

            let n = col_ptr.len() - 1;
            let nnz = values.len();
            let col_ptr = &*col_ptr.iter().copied().map(truncate).collect::<Vec<_>>();
            let row_ind = &*row_ind.iter().copied().map(truncate).collect::<Vec<_>>();
            let values_mat =
                crate::Mat::<E>::from_fn(nnz, 1, |i, _| complexify(E::faer_from_f64(values[i])));
            let values = values_mat.col_as_slice(0);

            let A_upper = SparseColMatRef::<'_, I, E>::new(
                SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_ind),
                values,
            );

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values_mat.clone();
            let mut A_lower_row_ind = row_ind.to_vec();
            let A_lower = crate::sparse::utils::adjoint(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                A_lower_values.col_as_slice_mut(0),
                A_upper,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            )
            .into_const();

            let mut A_dense = sparse_to_dense(A_upper);
            for j in 0..n {
                for i in j + 1..n {
                    A_dense.write(i, j, A_dense.read(j, i).faer_conj());
                }
            }

            for (A, side, supernodal_flop_ratio_threshold, parallelism) in [
                (
                    A_upper,
                    Side::Upper,
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    Parallelism::None,
                ),
                (
                    A_upper,
                    Side::Upper,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                    Parallelism::None,
                ),
                (
                    A_lower,
                    Side::Lower,
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    Parallelism::None,
                ),
                (
                    A_lower,
                    Side::Lower,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                    Parallelism::None,
                ),
            ] {
                let symbolic = factorize_symbolic_cholesky(
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
                        .factorize_numeric_intranode_bunch_kaufman_req::<E>(false, parallelism)
                        .unwrap(),
                );
                let mut L_values = Mat::<E>::zeros(symbolic.len_values(), 1);
                let mut subdiag = Mat::<E>::zeros(n, 1);
                let mut fwd = vec![I::truncate(0); n];
                let mut inv = vec![I::truncate(0); n];

                let lblt = symbolic.factorize_numeric_intranode_bunch_kaufman::<E>(
                    L_values.col_as_slice_mut(0),
                    subdiag.col_as_slice_mut(0),
                    &mut fwd,
                    &mut inv,
                    A,
                    side,
                    Default::default(),
                    parallelism,
                    PodStack::new(&mut mem),
                );

                for k in (1..16).chain(128..132) {
                    let rhs = Mat::<E>::from_fn(n, k, |_, _| {
                        E::faer_from_f64(gen.gen())
                            .faer_add(i.faer_mul(E::faer_from_f64(gen.gen())))
                    });
                    for conj in [Conj::No, Conj::Yes] {
                        let mut x = rhs.clone();
                        lblt.solve_in_place_with_conj(
                            conj,
                            x.as_mut(),
                            parallelism,
                            PodStack::new(&mut GlobalPodBuffer::new(
                                symbolic.solve_in_place_req::<E>(k).unwrap(),
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
            let values_mat = crate::Mat::<E>::from_fn(nnz, 1, |_, _| 0.0);
            let dynamic_regularization_epsilon = 1e-6;
            let dynamic_regularization_delta = 1e-2;

            let values = values_mat.col_as_slice(0);
            let mut signs = vec![-1i8; n];
            signs[..8].fill(1);

            let A_upper = SparseColMatRef::<'_, I, E>::new(
                SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_ind),
                values,
            );

            let mut A_lower_col_ptr = col_ptr.to_vec();
            let mut A_lower_values = values_mat.clone();
            let mut A_lower_row_ind = row_ind.to_vec();
            let A_lower = crate::sparse::utils::adjoint(
                &mut A_lower_col_ptr,
                &mut A_lower_row_ind,
                A_lower_values.col_as_slice_mut(0),
                A_upper,
                PodStack::new(&mut GlobalPodBuffer::new(StackReq::new::<I>(20 * n))),
            )
            .into_const();

            let mut A_dense = sparse_to_dense(A_upper);
            for (j, &sign) in signs.iter().enumerate() {
                A_dense.write(j, j, sign as f64 * dynamic_regularization_delta);
                for i in j + 1..n {
                    A_dense.write(i, j, A_dense.read(j, i).faer_conj());
                }
            }

            for (A, side, supernodal_flop_ratio_threshold, parallelism) in [
                (
                    A_upper,
                    Side::Upper,
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    Parallelism::None,
                ),
                (
                    A_upper,
                    Side::Upper,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                    Parallelism::None,
                ),
                (
                    A_lower,
                    Side::Lower,
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    Parallelism::None,
                ),
                (
                    A_lower,
                    Side::Lower,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                    Parallelism::None,
                ),
            ] {
                let symbolic = factorize_symbolic_cholesky(
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
                let mut L_values = L_values.col_as_slice_mut(0);

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
                        reconstruct_from_simplicial_ldlt::<I, E>(symbolic, L_values)
                    }
                    SymbolicCholeskyRaw::Supernodal(symbolic) => {
                        reconstruct_from_supernodal_ldlt::<I, E>(symbolic, L_values)
                    }
                };

                let (perm_fwd, _) = symbolic.perm().arrays();
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
    monomorphize_test!(test_supernodal, u32);
    monomorphize_test!(test_supernodal_ldlt, u32);
    monomorphize_test!(test_supernodal_intranode_bk_1, u32);
    monomorphize_test!(test_supernodal_intranode_bk_2, u32);
    monomorphize_test!(test_simplicial, u32);
    monomorphize_test!(test_solver_llt, u32);
    monomorphize_test!(test_solver_ldlt, u32);
    monomorphize_test!(test_solver_intranode_bk, u32);
    monomorphize_test!(test_solver_regularization, u32);
}
