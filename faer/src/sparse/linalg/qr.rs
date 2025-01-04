use crate::internal_prelude_sp::*;
use crate::{assert, debug_assert};
use linalg_sp::cholesky::ghost_postorder;
use linalg_sp::cholesky::simplicial::EliminationTreeRef;
use linalg_sp::{SymbolicSupernodalParams, colamd, ghost};

#[inline]
pub(crate) fn ghost_col_etree<'n, I: Index>(
	A: SymbolicSparseColMatRef<'_, I, Dim<'_>, Dim<'n>>,
	col_perm: Option<PermRef<'_, I, Dim<'n>>>,
	etree: &mut Array<'n, I::Signed>,
	stack: &mut MemStack,
) {
	let I = I::truncate;

	let N = A.ncols();
	let M = A.nrows();

	let (ancestor, stack) = unsafe { stack.make_raw::<I::Signed>(*N) };
	let (prev, _) = unsafe { stack.make_raw::<I::Signed>(*M) };

	let ancestor = Array::from_mut(ghost::fill_none::<I>(ancestor, N), N);
	let prev = Array::from_mut(ghost::fill_none::<I>(prev, N), M);

	etree.as_mut().fill(I::Signed::truncate(NONE));
	for j in N.indices() {
		let pj = col_perm.map(|perm| perm.bound_arrays().0[j].zx()).unwrap_or(j);
		for i_ in A.row_idx_of_col(pj) {
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
pub fn col_etree_scratch<I: Index>(nrows: usize, ncols: usize) -> StackReq {
	StackReq::all_of(&[StackReq::new::<I>(nrows), StackReq::new::<I>(ncols)])
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
	stack: &mut MemStack,
) -> EliminationTreeRef<'out, I> {
	with_dim!(M, A.nrows());
	with_dim!(N, A.ncols());
	ghost_col_etree(
		A.as_shape(M, N),
		col_perm.map(|perm| perm.as_shape(N)),
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
	A: SymbolicSparseColMatRef<'_, I, Dim<'m>, Dim<'n>>,
	row_perm: Option<PermRef<'_, I, Dim<'m>>>,
	etree: &Array<'m, MaybeIdx<'m, I>>,
	post: &Array<'m, Idx<'m, I>>,
	stack: &mut MemStack,
) {
	let M: Dim<'m> = A.nrows();
	let N: Dim<'n> = A.ncols();
	let n = *N;
	let m = *M;

	let delta = col_counts;
	let (first, stack) = unsafe { stack.make_raw::<I::Signed>(m) };
	let (max_first, stack) = unsafe { stack.make_raw::<I::Signed>(m) };
	let (prev_leaf, stack) = unsafe { stack.make_raw::<I::Signed>(m) };
	let (ancestor, stack) = unsafe { stack.make_raw::<I>(m) };
	let (next, stack) = unsafe { stack.make_raw::<I::Signed>(n) };
	let (head, _) = unsafe { stack.make_raw::<I::Signed>(m) };

	let post_inv = &mut *first;
	let post_inv = Array::from_mut(ghost::fill_zero::<I>(bytemuck::cast_slice_mut(post_inv), M), M);
	for j in M.indices() {
		post_inv[post[j].zx()] = j.truncate();
	}
	let next = Array::from_mut(ghost::fill_none::<I>(next, N), N);
	let head = Array::from_mut(ghost::fill_none::<I>(head, N), M);

	for j in N.indices() {
		if let Some(perm) = row_perm {
			let inv = perm.bound_arrays().1;
			min_row[j] = match Iterator::min(A.row_idx_of_col(j).map(|j| inv[j].zx())) {
				Some(first_row) => I::Signed::truncate(*first_row),
				None => *MaybeIdx::<'_, I>::none(),
			};
		} else {
			min_row[j] = match Iterator::min(A.row_idx_of_col(j)) {
				Some(first_row) => I::Signed::truncate(*first_row),
				None => *MaybeIdx::<'_, I>::none(),
			};
		}

		let min_row = if let Some(perm) = row_perm {
			let inv = perm.bound_arrays().1;
			Iterator::min(A.row_idx_of_col(j).map(|row| post_inv[inv[row].zx()]))
		} else {
			Iterator::min(A.row_idx_of_col(j).map(|row| post_inv[row]))
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
			for i in A.row_idx_of_col(j_) {
				let i = row_perm.map(|perm| perm.bound_arrays().1[i].zx()).unwrap_or(i);
				let lca = ghost_least_common_ancestor::<I>(i, pk, first, max_first, prev_leaf, ancestor);

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
pub fn column_counts_aat_scrach<I: Index>(nrows: usize, ncols: usize) -> StackReq {
	StackReq::all_of(&[StackReq::new::<I>(nrows).array(5), StackReq::new::<I>(ncols)])
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
	stack: &mut MemStack,
) {
	with_dim!(M, AT.nrows());
	with_dim!(N, AT.ncols());

	let A = AT.as_shape(M, N);
	ghost_column_counts_aat(
		Array::from_mut(col_counts, M),
		Array::from_mut(bytemuck::cast_slice_mut(min_col), N),
		A,
		col_perm.map(|perm| perm.as_shape(M)),
		etree.as_bound(M),
		Array::from_ref(Idx::from_slice_ref_checked(post, M), M),
		stack,
	)
}

/// Computes the size and alignment of the workspace required to compute the postordering of an
/// elimination tree of size `n`.
#[inline]
pub fn postorder_scratch<I: Index>(n: usize) -> StackReq {
	StackReq::new::<I>(n).array(3)
}

/// Computes a postordering of the elimination tree of size `n`.
#[inline]
pub fn postorder<I: Index>(post: &mut [I], etree: EliminationTreeRef<'_, I>, stack: &mut MemStack) {
	with_dim!(N, etree.inner.len());
	ghost_postorder(Array::from_mut(post, N), etree.as_bound(N), stack)
}

/// Supernodal factorization module.
///
/// A supernodal factorization is one that processes the elements of the QR factors of the
/// input matrix by blocks, rather than by single elements. This is more efficient if the QR factors
/// are somewhat dense.
pub mod supernodal {
	use super::*;
	use crate::assert;
	pub use linalg_sp::cholesky::supernodal::SymbolicSupernodalCholesky;

	/// Symbolic structure of the Householder reflections that compose $Q$,
	///
	/// such that:
	/// $$ Q = (I - H_1 T_1^{-1} H_1^H) \cdot (I - H_2 T_2^{-1} H_2^H) \dots (I - H_k T_k^{-1}
	/// H_k^H)$$
	#[derive(Debug)]
	pub struct SymbolicSupernodalHouseholder<I> {
		col_ptr_for_row_idx: alloc::vec::Vec<I>,
		col_ptr_for_tau_val: alloc::vec::Vec<I>,
		col_ptr_for_val: alloc::vec::Vec<I>,
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

		/// Returns the column pointers for the numerical val of the Householder factors.
		#[inline]
		pub fn col_ptr_for_householder_val(&self) -> &[I] {
			self.col_ptr_for_val.as_ref()
		}

		/// Returns the column pointers for the numerical val of the $T$ factors.
		#[inline]
		pub fn col_ptr_for_tau_val(&self) -> &[I] {
			self.col_ptr_for_tau_val.as_ref()
		}

		/// Returns the column pointers for the row indices of the Householder factors.
		#[inline]
		pub fn col_ptr_for_householder_row_indices(&self) -> &[I] {
			self.col_ptr_for_row_idx.as_ref()
		}

		/// Returns the length of the slice that can be used to contain the numerical val of the
		/// Householder factors.
		#[inline]
		pub fn len_householder_val(&self) -> usize {
			self.col_ptr_for_householder_val()[self.n_supernodes()].zx()
		}

		/// Returns the length of the slice that can be used to contain the row indices of the
		/// Householder factors.
		#[inline]
		pub fn len_householder_row_indices(&self) -> usize {
			self.col_ptr_for_householder_row_indices()[self.n_supernodes()].zx()
		}

		/// Returns the length of the slice that can be used to contain the numerical val of the
		/// $T$ factors.
		#[inline]
		pub fn len_tau_val(&self) -> usize {
			self.col_ptr_for_tau_val()[self.n_supernodes()].zx()
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
		pub fn solve_in_place_scratch<T: ComplexField>(&self, rhs_ncols: usize, par: Par) -> StackReq {
			let _ = par;
			let L_symbolic = self.r_adjoint();
			let H_symbolic = self.householder();
			let n_supernodes = L_symbolic.n_supernodes();

			let mut loop_scratch = StackReq::empty();
			for s in 0..n_supernodes {
				let s_h_row_begin = H_symbolic.col_ptr_for_row_idx[s].zx();
				let s_h_row_full_end = H_symbolic.col_ptr_for_row_idx[s + 1].zx();
				let max_blocksize = H_symbolic.max_blocksize[s].zx();

				loop_scratch = loop_scratch.or(
					crate::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(
						s_h_row_full_end - s_h_row_begin,
						max_blocksize,
						rhs_ncols,
					),
				);
			}

			loop_scratch
		}
	}

	/// Computes the size and alignment of the workspace required to compute the symbolic QR
	/// factorization of a matrix with dimensions `(nrows, ncols)`.
	pub fn factorize_supernodal_symbolic_qr_scratch<I: Index>(nrows: usize, ncols: usize) -> StackReq {
		let _ = nrows;
		crate::sparse::linalg::cholesky::supernodal::factorize_supernodal_symbolic_cholesky_scratch::<I>(ncols)
	}

	/// Computes the symbolic QR factorization of a matrix $A$, given a fill-reducing column
	/// permutation, and the outputs of the pre-factorization steps.
	pub fn factorize_supernodal_symbolic_qr<I: Index>(
		A: SymbolicSparseColMatRef<'_, I>,
		col_perm: Option<PermRef<'_, I>>,
		min_col: alloc::vec::Vec<I>,
		etree: EliminationTreeRef<'_, I>,
		col_counts: &[I],
		stack: &mut MemStack,
		params: SymbolicSupernodalParams<'_>,
	) -> Result<SymbolicSupernodalQr<I>, FaerError> {
		let m = A.nrows();
		let n = A.ncols();

		with_dim!(M, m);
		with_dim!(N, n);
		let A = A.as_shape(M, N);
		let mut stack = stack;
		let (L, H) = {
			let etree = etree.as_bound(N);
			let min_col = Array::from_ref(MaybeIdx::from_slice_ref_checked(bytemuck::cast_slice(&min_col), N), M);
			let L = crate::sparse::linalg::cholesky::supernodal::ghost_factorize_supernodal_symbolic(
				A,
				col_perm.map(|perm| perm.as_shape(N)),
				Some(min_col),
				crate::sparse::linalg::cholesky::supernodal::CholeskyInput::ATA,
				etree,
				Array::from_ref(col_counts, N),
				stack.rb_mut(),
				params,
			)?;

			let H = ghost_factorize_supernodal_householder_symbolic(&L, M, N, min_col, etree, stack)?;

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
			index_to_super[L.supernode_begin()[s].zx()..L.supernode_end()[s].zx()].fill(I::truncate(s));
		}

		child_head.fill(I::truncate(NONE));
		child_next.fill(I::truncate(NONE));

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
		M: Dim<'m>,
		N: Dim<'n>,
		min_col: &Array<'m, MaybeIdx<'n, I>>,
		etree: &Array<'n, MaybeIdx<'n, I>>,
		stack: &mut MemStack,
	) -> Result<SymbolicSupernodalHouseholder<I>, FaerError> {
		let n_supernodes = L_symbolic.n_supernodes();

		with_dim!(N_SUPERNODES, n_supernodes);

		let mut col_ptr_for_row_idx = try_zeroed::<I>(n_supernodes + 1)?;
		let mut col_ptr_for_tau_val = try_zeroed::<I>(n_supernodes + 1)?;
		let mut col_ptr_for_val = try_zeroed::<I>(n_supernodes + 1)?;
		let mut super_etree_ = try_zeroed::<I>(n_supernodes)?;
		let mut max_blocksize = try_zeroed::<I>(n_supernodes)?;
		let super_etree = bytemuck::cast_slice_mut::<I, I::Signed>(&mut super_etree_);

		let to_wide = |i: I| i.zx() as u128;
		let from_wide = |i: u128| I::truncate(i as usize);
		let from_wide_checked = |i: u128| -> Option<I> { (i <= to_wide(I::from_signed(I::Signed::MAX))).then_some(I::truncate(i as usize)) };

		let supernode_begin = Array::from_ref(L_symbolic.supernode_begin(), N_SUPERNODES);
		let supernode_end = Array::from_ref(L_symbolic.supernode_end(), N_SUPERNODES);
		let L_col_ptr_for_row_idx = L_symbolic.col_ptr_for_row_idx();

		let (index_to_super, _) = unsafe { stack.make_raw::<I>(*N) };

		for s in N_SUPERNODES.indices() {
			index_to_super[supernode_begin[s].zx()..supernode_end[s].zx()].fill(*s.truncate::<I>());
		}
		let index_to_super = Array::from_ref(Idx::from_slice_ref_checked(index_to_super, N_SUPERNODES), N);

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

		let non_zero_count = Array::from_mut(&mut col_ptr_for_row_idx[1..], N_SUPERNODES);
		for i in M.indices() {
			let Some(min_col) = min_col[i].idx() else {
				continue;
			};
			non_zero_count[index_to_super[min_col.zx()].zx()] += I::truncate(1);
		}

		for s in N_SUPERNODES.indices() {
			if let Some(parent) = super_etree[s].idx() {
				let s_col_count = L_col_ptr_for_row_idx[*s + 1] - L_col_ptr_for_row_idx[*s];
				let panel_width = supernode_end[s] - supernode_begin[s];

				let s_count = non_zero_count[s];
				non_zero_count[parent.zx()] += Ord::min(Ord::max(s_count, panel_width) - panel_width, s_col_count);
			}
		}

		let mut val_count = to_wide(I::truncate(0));
		let mut tau_count = to_wide(I::truncate(0));
		let mut row_count = to_wide(I::truncate(0));
		for (s, ((next_row_ptr, next_val_ptr), next_tau_ptr)) in iter::zip(
			N_SUPERNODES.indices(),
			iter::zip(
				iter::zip(&mut col_ptr_for_row_idx[1..], &mut col_ptr_for_val[1..]),
				&mut col_ptr_for_tau_val[1..],
			),
		) {
			let panel_width = supernode_end[s] - supernode_begin[s];
			let s_row_count = *next_row_ptr;
			let s_col_count = panel_width + (L_col_ptr_for_row_idx[*s + 1] - L_col_ptr_for_row_idx[*s]);
			val_count += to_wide(s_row_count) * to_wide(s_col_count);
			row_count += to_wide(s_row_count);
			let blocksize = crate::linalg::qr::no_pivoting::factor::recommended_blocksize::<Symbolic>(s_row_count.zx(), s_col_count.zx()) as u128;
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
			col_ptr_for_row_idx,
			col_ptr_for_val,
			super_etree: super_etree_,
			col_ptr_for_tau_val,
			max_blocksize,
			nrows: *M,
		})
	}
}
