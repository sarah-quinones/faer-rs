//! computes the $QR$ decomposition of a given sparse matrix. see [`crate::linalg::qr`] for more
//! info
//!
//! the entry point in this module is [`SymbolicQr`] and [`factorize_symbolic_qr`]
//!
//! # note
//! the functions in this module accept unsorted inputs, and may produce unsorted decomposition
//! factors.

use crate::assert;
use crate::internal_prelude_sp::*;
use crate::sparse::utils;
use linalg::qr::no_pivoting::factor::QrParams;
use linalg_sp::cholesky::ghost_postorder;
use linalg_sp::cholesky::simplicial::EliminationTreeRef;
use linalg_sp::{SupernodalThreshold, SymbolicSupernodalParams, colamd, ghost};

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

/// computes the size and alignment of the workspace required to compute the column elimination tree
/// of a matrix $A$ with dimensions `(nrows, ncols)`
#[inline]
pub fn col_etree_scratch<I: Index>(nrows: usize, ncols: usize) -> StackReq {
	StackReq::all_of(&[StackReq::new::<I>(nrows), StackReq::new::<I>(ncols)])
}

/// computes the column elimination tree of $A$, which is the same as the elimination tree of
/// $A^\top A$
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
		let mut j = (*head_k).sx();
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

/// computes the size and alignment of the workspace required to compute the column counts
/// of the cholesky factor of the matrix $A A^\top$, where $A$ has dimensions `(nrows, ncols)`
#[inline]
pub fn column_counts_aat_scratch<I: Index>(nrows: usize, ncols: usize) -> StackReq {
	StackReq::all_of(&[StackReq::new::<I>(nrows).array(5), StackReq::new::<I>(ncols)])
}

/// computes the column counts of the cholesky factor of $A^\top A$
///
/// - `col_counts` has length `A.ncols()`
/// - `min_col` has length `A.nrows()`
/// - `col_perm` has length `A.ncols()`: fill reducing permutation
/// - `etree` has length `A.ncols()`: column elimination tree of $A A^\top$
/// - `post` has length `A.ncols()`: postordering of `etree`
///
/// # warning
/// the function takes as input `A.transpose()`, not `A`
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

/// computes the size and alignment of the workspace required to compute the postordering of an
/// elimination tree of size `n`
#[inline]
pub fn postorder_scratch<I: Index>(n: usize) -> StackReq {
	StackReq::new::<I>(n).array(3)
}

/// computes a postordering of the elimination tree of size `n`
#[inline]
pub fn postorder<I: Index>(post: &mut [I], etree: EliminationTreeRef<'_, I>, stack: &mut MemStack) {
	with_dim!(N, etree.inner.len());
	ghost_postorder(Array::from_mut(post, N), etree.as_bound(N), stack)
}

/// supernodal factorization module
///
/// a supernodal factorization is one that processes the elements of the $QR$ factors of the
/// input matrix by blocks, rather than by single elements. this is more efficient if the $QR$
/// factors are somewhat dense
pub mod supernodal {
	use super::*;
	use crate::assert;
	use linalg_sp::cholesky::supernodal::{SupernodalLltRef, SymbolicSupernodalCholesky};

	/// symbolic structure of the householder reflections that compose $Q$
	///
	/// such that:
	/// $$ Q = (i - H_1 t_1^{-1} H_1^H) \cdot (i - H_2 t_2^{-1} H_2^H) \dots (i - H_k t_k^{-1}
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
		/// returns the number of rows of the householder factors
		#[inline]
		pub fn nrows(&self) -> usize {
			self.nrows
		}

		/// returns the number of supernodes in the symbolic $QR$
		#[inline]
		pub fn n_supernodes(&self) -> usize {
			self.super_etree.len()
		}

		/// returns the column pointers for the numerical values of the householder factors
		#[inline]
		pub fn col_ptr_for_householder_val(&self) -> &[I] {
			self.col_ptr_for_val.as_ref()
		}

		/// returns the column pointers for the numerical values of the $t$ factors
		#[inline]
		pub fn col_ptr_for_tau_val(&self) -> &[I] {
			self.col_ptr_for_tau_val.as_ref()
		}

		/// returns the column pointers for the row indices of the householder factors
		#[inline]
		pub fn col_ptr_for_householder_row_idx(&self) -> &[I] {
			self.col_ptr_for_row_idx.as_ref()
		}

		/// returns the length of the slice that can be used to contain the numerical values of the
		/// householder factors
		#[inline]
		pub fn len_householder_val(&self) -> usize {
			self.col_ptr_for_householder_val()[self.n_supernodes()].zx()
		}

		/// returns the length of the slice that can be used to contain the row indices of the
		/// householder factors
		#[inline]
		pub fn len_householder_row_idx(&self) -> usize {
			self.col_ptr_for_householder_row_idx()[self.n_supernodes()].zx()
		}

		/// returns the length of the slice that can be used to contain the numerical values of the
		/// $t$ factors
		#[inline]
		pub fn len_tau_val(&self) -> usize {
			self.col_ptr_for_tau_val()[self.n_supernodes()].zx()
		}
	}
	/// symbolic structure of the $QR$ decomposition,
	#[derive(Debug)]
	pub struct SymbolicSupernodalQr<I> {
		L: SymbolicSupernodalCholesky<I>,
		H: SymbolicSupernodalHouseholder<I>,
		min_col: alloc::vec::Vec<I>,
		min_col_perm: alloc::vec::Vec<I>,
		index_to_super: alloc::vec::Vec<I>,
		child_head: alloc::vec::Vec<I>,
		child_next: alloc::vec::Vec<I>,
	}

	impl<I: Index> SymbolicSupernodalQr<I> {
		/// returns the symbolic structure of $R^H$
		#[inline]
		pub fn R_adjoint(&self) -> &SymbolicSupernodalCholesky<I> {
			&self.L
		}

		/// returns the symbolic structure of the householder and $t$ factors
		#[inline]
		pub fn householder(&self) -> &SymbolicSupernodalHouseholder<I> {
			&self.H
		}

		/// computes the size and alignment of the workspace required to solve the linear system
		/// $A x = \text{rhs}$ in the sense of least squares
		pub fn solve_in_place_scratch<T: ComplexField>(&self, rhs_ncols: usize, par: Par) -> StackReq {
			let _ = par;
			let L_symbolic = self.R_adjoint();
			let H_symbolic = self.householder();
			let n_supernodes = L_symbolic.n_supernodes();

			let mut loop_scratch = StackReq::empty();
			for s in 0..n_supernodes {
				let s_h_row_begin = H_symbolic.col_ptr_for_row_idx[s].zx();
				let s_h_row_full_end = H_symbolic.col_ptr_for_row_idx[s + 1].zx();
				let max_blocksize = H_symbolic.max_blocksize[s].zx();

				loop_scratch = loop_scratch.or(
					linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(
						s_h_row_full_end - s_h_row_begin,
						max_blocksize,
						rhs_ncols,
					),
				);
			}

			loop_scratch
		}
	}

	/// computes the size and alignment of the workspace required to compute the symbolic $QR$
	/// factorization of a matrix with dimensions `(nrows, ncols)`
	pub fn factorize_supernodal_symbolic_qr_scratch<I: Index>(nrows: usize, ncols: usize) -> StackReq {
		let _ = nrows;
		linalg_sp::cholesky::supernodal::factorize_supernodal_symbolic_cholesky_scratch::<I>(ncols)
	}

	/// computes the symbolic $QR$ factorization of a matrix $A$, given a fill-reducing column
	/// permutation, and the outputs of the pre-factorization steps
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
			let L = linalg_sp::cholesky::supernodal::ghost_factorize_supernodal_symbolic(
				A,
				col_perm.map(|perm| perm.as_shape(N)),
				Some(min_col),
				linalg_sp::cholesky::supernodal::CholeskyInput::ATA,
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
			let blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<Symbolic>(s_row_count.zx(), s_col_count.zx()) as u128;
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

	/// $QR$ factors containing both the symbolic and numeric representations
	#[derive(Debug)]
	pub struct SupernodalQrRef<'a, I: Index, T> {
		symbolic: &'a SymbolicSupernodalQr<I>,
		rt_val: &'a [T],
		householder_val: &'a [T],
		tau_val: &'a [T],
		householder_row_idx: &'a [I],
		tau_blocksize: &'a [I],
		householder_nrows: &'a [I],
		householder_ncols: &'a [I],
	}

	impl<I: Index, T> Copy for SupernodalQrRef<'_, I, T> {}
	impl<I: Index, T> Clone for SupernodalQrRef<'_, I, T> {
		#[inline]
		fn clone(&self) -> Self {
			*self
		}
	}

	impl<'a, I: Index, T> SupernodalQrRef<'a, I, T> {
		/// creates $QR$ factors from their components
		///
		/// # safety
		/// the inputs must be the outputs of [`factorize_supernodal_numeric_qr`]
		#[inline]
		pub unsafe fn new_unchecked(
			symbolic: &'a SymbolicSupernodalQr<I>,
			householder_row_idx: &'a [I],
			tau_blocksize: &'a [I],
			householder_nrows: &'a [I],
			householder_ncols: &'a [I],
			r_val: &'a [T],
			householder_val: &'a [T],
			tau_val: &'a [T],
		) -> Self {
			let rt_val = r_val;
			let householder_val = householder_val;
			let tau_val = tau_val;
			assert!(rt_val.len() == symbolic.R_adjoint().len_val());
			assert!(tau_val.len() == symbolic.householder().len_tau_val());
			assert!(householder_val.len() == symbolic.householder().len_householder_val());
			assert!(tau_blocksize.len() == householder_nrows.len());
			Self {
				symbolic,
				tau_blocksize,
				householder_nrows,
				householder_ncols,
				rt_val,
				householder_val,
				tau_val,
				householder_row_idx,
			}
		}

		/// returns the symbolic structure of the $QR$ factorization
		#[inline]
		pub fn symbolic(self) -> &'a SymbolicSupernodalQr<I> {
			self.symbolic
		}

		/// returns the numerical values of the factor $R$ of the $QR$ factorization
		#[inline]
		pub fn R_val(self) -> &'a [T] {
			self.rt_val
		}

		/// returns the numerical values of the householder factors of the $QR$ factorization
		#[inline]
		pub fn householder_val(self) -> &'a [T] {
			self.householder_val
		}

		/// returns the numerical values of the $t$ factors of the $QR$ factorization
		#[inline]
		pub fn tau_val(self) -> &'a [T] {
			self.tau_val
		}

		/// Applies $Q^{\top}$ to the rhs in place, implicitly conjugating $Q$ if needed
		///
		/// `work` is a temporary workspace with the same dimensions as `rhs`
		#[math]
		pub fn apply_Q_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, work: MatMut<'_, T>, stack: &mut MemStack)
		where
			T: ComplexField,
		{
			let L_symbolic = self.symbolic().R_adjoint();
			let H_symbolic = self.symbolic().householder();
			let n_supernodes = L_symbolic.n_supernodes();
			let mut stack = stack;

			assert!(rhs.nrows() == self.symbolic().householder().nrows);

			let mut x = rhs;
			let k = x.ncols();
			let mut tmp = work;
			tmp.fill(zero());

			// x <- Q^T x
			{
				let H = self.householder_val;
				let tau = self.tau_val;

				let mut block_count = 0usize;
				for s in 0..n_supernodes {
					let tau_begin = H_symbolic.col_ptr_for_tau_val[s].zx();
					let tau_end = H_symbolic.col_ptr_for_tau_val[s + 1].zx();

					let s_h_row_begin = H_symbolic.col_ptr_for_row_idx[s].zx();
					let s_h_row_full_end = H_symbolic.col_ptr_for_row_idx[s + 1].zx();

					let s_col_begin = L_symbolic.supernode_begin()[s].zx();
					let s_col_end = L_symbolic.supernode_end()[s].zx();
					let s_ncols = s_col_end - s_col_begin;

					let s_row_idx_in_panel = &self.householder_row_idx[s_h_row_begin..s_h_row_full_end];

					let mut tmp = tmp.rb_mut().subrows_mut(s_col_begin, s_h_row_full_end - s_h_row_begin);
					for j in 0..k {
						for idx in 0..s_h_row_full_end - s_h_row_begin {
							let i = s_row_idx_in_panel[idx].zx();
							tmp[(idx, j)] = copy(x[(i, j)]);
						}
					}

					let s_H = &H[H_symbolic.col_ptr_for_val[s].zx()..H_symbolic.col_ptr_for_val[s + 1].zx()];

					let s_H = MatRef::from_column_major_slice(
						s_H,
						s_h_row_full_end - s_h_row_begin,
						s_ncols + (L_symbolic.col_ptr_for_row_idx()[s + 1].zx() - L_symbolic.col_ptr_for_row_idx()[s].zx()),
					);
					let s_tau = &tau[tau_begin..tau_end];
					let max_blocksize = H_symbolic.max_blocksize[s].zx();
					let s_tau = MatRef::from_column_major_slice(s_tau, max_blocksize, Ord::min(s_H.ncols(), s_h_row_full_end - s_h_row_begin));

					let mut start = 0;
					let end = s_H.ncols();
					while start < end {
						let bs = self.tau_blocksize[block_count].zx();
						let nrows = self.householder_nrows[block_count].zx();
						let ncols = self.householder_ncols[block_count].zx();

						let b_H = s_H.submatrix(start, start, nrows, ncols);
						let b_tau = s_tau.subcols(start, ncols).subrows(0, bs);

						linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
							b_H.rb(),
							b_tau.rb(),
							conj,
							tmp.rb_mut().subrows_mut(start, nrows),
							par,
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
							let i = s_row_idx_in_panel[idx].zx();
							x[(i, j)] = copy(tmp[(idx, j)]);
						}
					}
				}
			}
			let m = H_symbolic.nrows;
			let n = L_symbolic.nrows();
			x.rb_mut().subrows_mut(0, n).copy_from(tmp.rb().subrows(0, n));
			x.rb_mut().subrows_mut(n, m - n).fill(zero());
		}

		/// solves the equation $A x = \text{rhs}$ in the sense of least squares, implicitly
		/// conjugating $A$ if needed
		///
		/// `work` is a temporary workspace with the same dimensions as `rhs`
		#[track_caller]
		#[math]
		pub fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, work: MatMut<'_, T>, stack: &mut MemStack)
		where
			T: ComplexField,
		{
			let mut work = work;
			let mut rhs = rhs;
			self.apply_Q_transpose_in_place_with_conj(conj.compose(Conj::Yes), rhs.rb_mut(), par, work.rb_mut(), stack);

			let L_symbolic = self.symbolic().R_adjoint();
			let n_supernodes = L_symbolic.n_supernodes();

			let mut tmp = work;
			let mut x = rhs;
			let k = x.ncols();

			// x <- R^-1 x = L^-T x
			{
				let L = SupernodalLltRef::<'_, I, T>::new(L_symbolic, self.rt_val);

				for s in (0..n_supernodes).rev() {
					let s = L.supernode(s);
					let size = s.val().ncols();
					let s_L = s.val();
					let (s_L_top, s_L_bot) = s_L.split_at_row(size);

					let mut tmp = tmp.rb_mut().subrows_mut(0, s.pattern().len());
					for j in 0..k {
						for (idx, i) in s.pattern().iter().enumerate() {
							let i = i.zx();
							tmp[(idx, j)] = copy(x[(i, j)]);
						}
					}

					let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
					linalg::matmul::matmul_with_conj(
						x_top.rb_mut(),
						Accum::Add,
						s_L_bot.transpose(),
						conj.compose(Conj::Yes),
						tmp.rb(),
						Conj::No,
						-one::<T>(),
						par,
					);
					linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(
						s_L_top.transpose(),
						conj.compose(Conj::Yes),
						x_top.rb_mut(),
						par,
					);
				}
			}
		}
	}

	/// computes the size and alignment of the workspace required to compute the numerical $QR$
	/// factorization of the matrix whose structure was used to produce the symbolic structure
	#[track_caller]
	pub fn factorize_supernodal_numeric_qr_scratch<I: Index, T: ComplexField>(
		symbolic: &SymbolicSupernodalQr<I>,
		par: Par,
		params: Spec<QrParams, T>,
	) -> StackReq {
		let n_supernodes = symbolic.L.n_supernodes();
		let n = symbolic.L.dimension;
		let m = symbolic.H.nrows;
		let init_scratch = StackReq::all_of(&[
			StackReq::new::<I>(symbolic.H.len_householder_row_idx()),
			StackReq::new::<I>(n_supernodes),
			StackReq::new::<I>(n),
			StackReq::new::<I>(n),
			StackReq::new::<I>(m),
			StackReq::new::<I>(m),
		]);

		let mut loop_scratch = StackReq::empty();
		for s in 0..n_supernodes {
			let s_h_row_begin = symbolic.H.col_ptr_for_row_idx[s].zx();
			let s_h_row_full_end = symbolic.H.col_ptr_for_row_idx[s + 1].zx();
			let max_blocksize = symbolic.H.max_blocksize[s].zx();
			let s_col_begin = symbolic.L.supernode_begin()[s].zx();
			let s_col_end = symbolic.L.supernode_end()[s].zx();
			let s_ncols = s_col_end - s_col_begin;
			let s_pattern_len = symbolic.L.col_ptr_for_row_idx()[s + 1].zx() - symbolic.L.col_ptr_for_row_idx()[s].zx();

			loop_scratch = loop_scratch.or(linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(
				s_h_row_full_end - s_h_row_begin,
				s_ncols + s_pattern_len,
				max_blocksize,
				par,
				params,
			));

			loop_scratch = loop_scratch.or(
				linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(
					s_h_row_full_end - s_h_row_begin,
					max_blocksize,
					s_ncols + s_pattern_len,
				),
			);
		}

		init_scratch.and(loop_scratch)
	}

	/// computes the numerical $QR$ factorization of $A$
	///
	/// - `householder_row_idx` must have length `symbolic.householder().len_householder_row_idx()`
	/// - `tau_blocksize` must have length `symbolic.householder().len_householder_row_idx() +
	///   symbolic.householder().n_supernodes()`
	/// - `householder_nrows` must have length `symbolic.householder().len_householder_row_idx()
	///   + symbolic.householder().n_supernodes()`
	/// - `householder_ncols` must have length `symbolic.householder().len_householder_row_idx()
	///   + symbolic.householder().n_supernodes()`
	/// - `r_val` must have length `symbolic.R_adjoint().len_val()`
	/// - `householder_val` must have length `symbolic.householder().length_householder_val()`.
	/// - `tau_val` must have length `symbolic.householder().len_tau_val()`
	///
	/// # warning
	/// - note that the matrix takes as input `A.transpose()`, not `A`
	#[track_caller]
	pub fn factorize_supernodal_numeric_qr<'a, I: Index, T: ComplexField>(
		householder_row_idx: &'a mut [I],
		tau_blocksize: &'a mut [I],
		householder_nrows: &'a mut [I],
		householder_ncols: &'a mut [I],

		r_val: &'a mut [T],
		householder_val: &'a mut [T],
		tau_val: &'a mut [T],

		AT: SparseColMatRef<'_, I, T>,
		col_perm: Option<PermRef<'_, I>>,
		symbolic: &'a SymbolicSupernodalQr<I>,
		par: Par,
		stack: &mut MemStack,
		params: Spec<QrParams, T>,
	) -> SupernodalQrRef<'a, I, T> {
		assert!(all(
			householder_row_idx.len() == symbolic.householder().len_householder_row_idx(),
			r_val.len() == symbolic.R_adjoint().len_val(),
			householder_val.len() == symbolic.householder().len_householder_val(),
			tau_val.len() == symbolic.householder().len_tau_val(),
			tau_blocksize.len() == symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes(),
			householder_nrows.len() == symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes(),
			householder_ncols.len() == symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes(),
		));

		factorize_supernodal_numeric_qr_impl(
			householder_row_idx,
			tau_blocksize,
			householder_nrows,
			householder_ncols,
			r_val,
			householder_val,
			tau_val,
			AT,
			col_perm,
			&symbolic.L,
			&symbolic.H,
			&symbolic.min_col,
			&symbolic.min_col_perm,
			&symbolic.index_to_super,
			bytemuck::cast_slice(&symbolic.child_head),
			bytemuck::cast_slice(&symbolic.child_next),
			par,
			stack,
			params,
		);

		unsafe {
			SupernodalQrRef::<'_, I, T>::new_unchecked(
				symbolic,
				householder_row_idx,
				tau_blocksize,
				householder_nrows,
				householder_ncols,
				r_val,
				householder_val,
				tau_val,
			)
		}
	}

	#[math]
	pub(crate) fn factorize_supernodal_numeric_qr_impl<I: Index, T: ComplexField>(
		// len: col_ptr_for_row_idx[n_supernodes]
		householder_row_idx: &mut [I],

		tau_blocksize: &mut [I],
		householder_nrows: &mut [I],
		householder_ncols: &mut [I],

		L_val: &mut [T],
		householder_val: &mut [T],
		tau_val: &mut [T],

		AT: SparseColMatRef<'_, I, T>,
		col_perm: Option<PermRef<'_, I>>,
		L_symbolic: &SymbolicSupernodalCholesky<I>,
		H_symbolic: &SymbolicSupernodalHouseholder<I>,
		min_col: &[I],
		min_col_perm: &[I],
		index_to_super: &[I],
		child_head: &[I::Signed],
		child_next: &[I::Signed],

		par: Par,
		stack: &mut MemStack,
		params: Spec<QrParams, T>,
	) -> usize {
		let n_supernodes = L_symbolic.n_supernodes();
		let m = AT.ncols();
		let n = AT.nrows();

		let mut block_count = 0;

		let (min_col_in_panel, stack) = unsafe { stack.make_raw::<I>(H_symbolic.len_householder_row_idx()) };
		let (min_col_in_panel_perm, stack) = unsafe { stack.make_raw::<I>(m) };
		let (col_end_for_row_idx_in_panel, stack) = unsafe { stack.make_raw::<I>(n_supernodes) };
		let (col_global_to_local, stack) = unsafe { stack.make_raw::<I::Signed>(n) };
		let (child_col_global_to_local, stack) = unsafe { stack.make_raw::<I::Signed>(n) };
		let (child_row_global_to_local, mut stack) = unsafe { stack.make_raw::<I::Signed>(m) };

		tau_val.fill(zero());
		L_val.fill(zero());
		householder_val.fill(zero());

		col_end_for_row_idx_in_panel.copy_from_slice(&H_symbolic.col_ptr_for_row_idx[..n_supernodes]);

		for i in 0..m {
			let i = min_col_perm[i].zx();
			let min_col = min_col[i].zx();
			if min_col < n {
				let s = index_to_super[min_col].zx();
				let pos = &mut col_end_for_row_idx_in_panel[s];
				householder_row_idx[pos.zx()] = I::truncate(i);
				min_col_in_panel[pos.zx()] = I::truncate(min_col);
				*pos += I::truncate(1);
			}
		}

		col_global_to_local.fill(I::Signed::truncate(NONE));
		child_col_global_to_local.fill(I::Signed::truncate(NONE));
		child_row_global_to_local.fill(I::Signed::truncate(NONE));

		let supernode_begin = L_symbolic.supernode_begin();
		let supernode_end = L_symbolic.supernode_end();

		let super_etree = &*H_symbolic.super_etree;

		let col_pattern =
			|node: usize| &L_symbolic.row_idx()[L_symbolic.col_ptr_for_row_idx()[node].zx()..L_symbolic.col_ptr_for_row_idx()[node + 1].zx()];

		// assemble the parts from child supernodes
		for s in 0..n_supernodes {
			// all child nodes should be fully assembled
			let s_h_row_begin = H_symbolic.col_ptr_for_row_idx[s].zx();
			let s_h_row_full_end = H_symbolic.col_ptr_for_row_idx[s + 1].zx();
			let s_h_row_end = col_end_for_row_idx_in_panel[s].zx();

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

			let (s_min_col_in_panel, parent_min_col_in_panel) = min_col_in_panel.split_at_mut(s_h_row_end);
			let parent_offset = s_h_row_end;
			let (c_min_col_in_panel, s_min_col_in_panel) = s_min_col_in_panel.split_at_mut(s_h_row_begin);

			let (householder_row_idx, parent_row_idx_in_panel) = householder_row_idx.split_at_mut(s_h_row_end);

			let (s_H, _) = householder_val.split_at_mut(H_symbolic.col_ptr_for_val[s + 1].zx());
			let (c_H, s_H) = s_H.split_at_mut(H_symbolic.col_ptr_for_val[s].zx());
			let c_H = &*c_H;

			let mut s_H = MatMut::from_column_major_slice_mut(s_H, s_h_row_full_end - s_h_row_begin, s_ncols + s_pattern.len())
				.subrows_mut(0, s_h_row_end - s_h_row_begin);

			{
				let s_min_col_in_panel_perm = &mut min_col_in_panel_perm[0..s_h_row_end - s_h_row_begin];
				for (i, p) in s_min_col_in_panel_perm.iter_mut().enumerate() {
					*p = I::truncate(i);
				}
				s_min_col_in_panel_perm.sort_unstable_by_key(|i| s_min_col_in_panel[i.zx()]);

				let s_row_idx_in_panel = &mut householder_row_idx[s_h_row_begin..];
				let tmp: &mut [I] = bytemuck::cast_slice_mut(&mut child_row_global_to_local[..s_h_row_end - s_h_row_begin]);

				for (i, p) in s_min_col_in_panel_perm.iter().enumerate() {
					let p = p.zx();
					tmp[i] = s_min_col_in_panel[p];
				}
				s_min_col_in_panel.copy_from_slice(tmp);

				for (i, p) in s_min_col_in_panel_perm.iter().enumerate() {
					let p = p.zx();
					tmp[i] = s_row_idx_in_panel[p];
				}
				s_row_idx_in_panel.copy_from_slice(tmp);
				for (i, p) in s_min_col_in_panel_perm.iter_mut().enumerate() {
					*p = I::truncate(i);
				}

				tmp.fill(I::truncate(NONE));
			}

			let s_row_idx_in_panel = &householder_row_idx[s_h_row_begin..];

			for idx in 0..s_h_row_end - s_h_row_begin {
				let i = s_row_idx_in_panel[idx].zx();
				if min_col[i].zx() >= s_col_begin {
					for (j, value) in iter::zip(AT.row_idx_of_col(i), AT.val_of_col(i)) {
						let pj = col_perm.map(|perm| perm.arrays().1[j].zx()).unwrap_or(j);
						let ix = idx;
						let iy = col_global_to_local[pj].zx();
						s_H[(ix, iy)] = s_H[(ix, iy)] + *value;
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

				let c_h_row_begin = H_symbolic.col_ptr_for_row_idx[child].zx();
				let c_h_row_end = H_symbolic.col_ptr_for_row_idx[child + 1].zx();

				let c_row_idx_in_panel = &householder_row_idx[c_h_row_begin..c_h_row_end];
				let c_min_col_in_panel = &c_min_col_in_panel[c_h_row_begin..c_h_row_end];

				let c_H = &c_H[H_symbolic.col_ptr_for_val[child].zx()..H_symbolic.col_ptr_for_val[child + 1].zx()];
				let c_H = MatRef::from_column_major_slice(
					c_H,
					H_symbolic.col_ptr_for_row_idx[child + 1].zx() - c_h_row_begin,
					c_ncols + c_pattern.len(),
				);

				for (idx, &col) in c_pattern.iter().enumerate() {
					child_col_global_to_local[col.zx()] = I::Signed::truncate(idx + c_ncols);
				}
				for (idx, &p) in c_row_idx_in_panel.iter().enumerate() {
					child_row_global_to_local[p.zx()] = I::Signed::truncate(idx);
				}

				for s_idx in 0..s_h_row_end - s_h_row_begin {
					let i = s_row_idx_in_panel[s_idx].zx();
					let c_idx = child_row_global_to_local[i];
					if c_idx < I::Signed::truncate(0) {
						continue;
					}

					let c_idx = c_idx.zx();
					let c_min_col = c_min_col_in_panel[c_idx].zx();

					for (j_idx_in_c, j) in c_pattern.iter().enumerate() {
						let j_idx_in_c = j_idx_in_c + c_ncols;
						if j.zx() >= c_min_col {
							s_H[(s_idx, col_global_to_local[j.zx()].zx())] = copy(c_H[(c_idx, j_idx_in_c)]);
						}
					}
				}

				for &row in c_row_idx_in_panel {
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

				let tau_begin = H_symbolic.col_ptr_for_tau_val[s].zx();
				let tau_end = H_symbolic.col_ptr_for_tau_val[s + 1].zx();
				let L_begin = L_symbolic.col_ptr_for_val()[s].zx();
				let L_end = L_symbolic.col_ptr_for_val()[s + 1].zx();

				let s_tau = &mut tau_val[tau_begin..tau_end];
				let s_L = &mut L_val[L_begin..L_end];

				let max_blocksize = H_symbolic.max_blocksize[s].zx();
				let mut s_tau = MatMut::from_column_major_slice_mut(s_tau, max_blocksize, Ord::min(s_H.ncols(), s_h_row_full_end - s_h_row_begin));

				{
					let mut current_min_col = 0usize;
					let mut current_start = 0usize;
					for idx in 0..s_h_nrows + 1 {
						let idx_global_min_col = if idx < s_h_nrows { s_min_col_in_panel[idx].zx() } else { n };

						let idx_min_col = if idx_global_min_col < n {
							col_global_to_local[idx_global_min_col.zx()].zx()
						} else {
							s_H.ncols()
						};

						if idx_min_col == s_H.ncols() || idx_min_col >= current_min_col.saturating_add(Ord::max(1, max_blocksize / 2)) {
							let nrows = idx.saturating_sub(current_start);
							let full_ncols = s_H.ncols() - current_start;
							let ncols = Ord::min(nrows, idx_min_col - current_min_col);

							let s_H = s_H.rb_mut().submatrix_mut(current_start, current_start, nrows, full_ncols);

							let (mut left, mut right) = s_H.split_at_col_mut(ncols);
							let bs = linalg::qr::no_pivoting::factor::recommended_blocksize::<Symbolic>(left.nrows(), left.ncols());
							let bs = Ord::min(max_blocksize, bs);
							tau_blocksize[block_count] = I::truncate(bs);
							householder_nrows[block_count] = I::truncate(nrows);
							householder_ncols[block_count] = I::truncate(ncols);
							block_count += 1;

							let mut s_tau = s_tau.rb_mut().subrows_mut(0, bs).subcols_mut(current_start, ncols);

							linalg::qr::no_pivoting::factor::qr_in_place(left.rb_mut(), s_tau.rb_mut(), par, stack.rb_mut(), params);

							if right.ncols() > 0 {
								linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
									left.rb(),
									s_tau.rb(),
									Conj::Yes,
									right.rb_mut(),
									par,
									stack.rb_mut(),
								);
							}

							current_min_col = idx_min_col;
							current_start += ncols;
						}
					}
				}

				let mut s_L = MatMut::from_column_major_slice_mut(s_L, s_pattern.len() + s_ncols, s_ncols);
				let nrows = Ord::min(s_H.nrows(), s_L.ncols());
				z!(s_L.rb_mut().transpose_mut().subrows_mut(0, nrows), s_H.rb().subrows(0, nrows))
					.for_each_triangular_upper(linalg::zip::Diag::Include, |uz!(dst, src)| *dst = conj(*src));
			}

			col_end_for_row_idx_in_panel[s] = Ord::min(I::truncate(s_h_row_begin + s_ncols + s_pattern.len()), col_end_for_row_idx_in_panel[s]);

			let s_h_row_end = col_end_for_row_idx_in_panel[s].zx();
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
			let p_h_row_begin = H_symbolic.col_ptr_for_row_idx[parent].zx();
			let mut pos = col_end_for_row_idx_in_panel[parent].zx() - p_h_row_begin;
			let parent_min_col_in_panel = &mut parent_min_col_in_panel[p_h_row_begin - parent_offset..];
			let parent_row_idx_in_panel = &mut parent_row_idx_in_panel[p_h_row_begin - parent_offset..];

			for idx in s_pivot_row_end..s_h_nrows {
				parent_row_idx_in_panel[pos] = s_row_idx_in_panel[idx];
				parent_min_col_in_panel[pos] = s_min_col_in_panel[idx];
				pos += 1;
			}
			col_end_for_row_idx_in_panel[parent] = I::truncate(pos + p_h_row_begin);

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

/// simplicial factorization module
///
/// a simplicial factorization is one that processes the elements of the $QR$ factors of the
/// input matrix by single elements, rather than by blocks. this is more efficient if the $QR$
/// factors are very sparse
pub mod simplicial {
	use super::*;
	use crate::assert;

	/// symbolic structure of the $QR$ decomposition
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
		/// returns the number of rows of the matrix $A$
		#[inline]
		pub fn nrows(&self) -> usize {
			self.nrows
		}

		/// returns the number of columns of the matrix $A$
		#[inline]
		pub fn ncols(&self) -> usize {
			self.ncols
		}

		/// returns the length of the slice that can be used to contain the householder factors
		#[inline]
		pub fn len_householder(&self) -> usize {
			self.h_nnz
		}

		/// returns the length of the slice that can be used to contain the $R$ factor
		#[inline]
		pub fn len_r(&self) -> usize {
			self.l_nnz
		}
	}

	/// $QR$ factors containing both the symbolic and numeric representations
	#[derive(Debug)]
	pub struct SimplicialQrRef<'a, I, T> {
		symbolic: &'a SymbolicSimplicialQr<I>,
		r_col_ptr: &'a [I],
		r_row_idx: &'a [I],
		r_val: &'a [T],
		householder_col_ptr: &'a [I],
		householder_row_idx: &'a [I],
		householder_val: &'a [T],
		tau_val: &'a [T],
	}

	impl<I, T> Copy for SimplicialQrRef<'_, I, T> {}
	impl<I, T> Clone for SimplicialQrRef<'_, I, T> {
		#[inline]
		fn clone(&self) -> Self {
			*self
		}
	}

	impl<'a, I: Index, T> SimplicialQrRef<'a, I, T> {
		/// creates $QR$ factors from their components
		#[inline]
		pub fn new(
			symbolic: &'a SymbolicSimplicialQr<I>,
			r: SparseColMatRef<'a, I, T>,
			householder: SparseColMatRef<'a, I, T>,
			tau_val: &'a [T],
		) -> Self {
			assert!(householder.nrows() == symbolic.nrows);
			assert!(householder.ncols() == symbolic.ncols);
			assert!(r.nrows() == symbolic.ncols);
			assert!(r.ncols() == symbolic.ncols);

			let r_col_ptr = r.col_ptr();
			let r_row_idx = r.row_idx();
			let r_val = r.val();
			assert!(r.col_nnz().is_none());

			let householder_col_ptr = householder.col_ptr();
			let householder_row_idx = householder.row_idx();
			let householder_val = householder.val();
			assert!(householder.col_nnz().is_none());

			assert!(r_val.len() == symbolic.len_r());
			assert!(tau_val.len() == symbolic.ncols);
			assert!(householder_val.len() == symbolic.len_householder());
			Self {
				symbolic,
				householder_val,
				tau_val,
				r_val,
				r_col_ptr,
				r_row_idx,
				householder_col_ptr,
				householder_row_idx,
			}
		}

		/// returns the symbolic structure of the $QR$ factorization.
		#[inline]
		pub fn symbolic(&self) -> &SymbolicSimplicialQr<I> {
			self.symbolic
		}

		/// returns the numerical values of the factor $R$ of the $QR$ factorization
		#[inline]
		pub fn R_val(self) -> &'a [T] {
			self.r_val
		}

		/// returns the factor $R$
		#[inline]
		pub fn R(self) -> SparseColMatRef<'a, I, T> {
			let n = self.symbolic().ncols();
			SparseColMatRef::<'_, I, T>::new(
				unsafe { SymbolicSparseColMatRef::new_unchecked(n, n, self.r_col_ptr, None, self.r_row_idx) },
				self.r_val,
			)
		}

		/// returns the householder coefficients $H$ in the columns of a sparse matrix
		#[inline]
		pub fn householder(self) -> SparseColMatRef<'a, I, T> {
			let m = self.symbolic.nrows;
			let n = self.symbolic.ncols;
			SparseColMatRef::<'_, I, T>::new(
				unsafe { SymbolicSparseColMatRef::new_unchecked(m, n, self.householder_col_ptr, None, self.householder_row_idx) },
				self.householder_val,
			)
		}

		/// returns the numerical values of the householder factors of the $QR$ factorization.
		#[inline]
		pub fn householder_val(self) -> &'a [T] {
			self.householder_val
		}

		/// returns the numerical values of the $t$ factors of the $QR$ factorization.
		#[inline]
		pub fn tau_val(self) -> &'a [T] {
			self.tau_val
		}

		/// Applies $Q^{\top}$ to the input matrix `rhs`, implicitly conjugating the $Q$
		/// matrix if needed
		///
		/// `work` is a temporary workspace with the same dimensions as `rhs`.
		#[math]
		pub fn apply_qt_in_place_with_conj(&self, conj_qr: Conj, rhs: MatMut<'_, T>, par: Par, work: MatMut<'_, T>)
		where
			T: ComplexField,
		{
			let _ = par;
			assert!(rhs.nrows() == self.symbolic.nrows);
			let mut x = rhs;

			let m = self.symbolic.nrows;
			let n = self.symbolic.ncols;

			let h = SparseColMatRef::<'_, I, T>::new(
				unsafe { SymbolicSparseColMatRef::new_unchecked(m, n, self.householder_col_ptr, None, self.householder_row_idx) },
				self.householder_val,
			);
			let tau = self.tau_val;

			let mut tmp = work;
			tmp.fill(zero());

			// x <- Q^T x
			{
				for j in 0..n {
					let hi = h.row_idx_of_col_raw(j);
					let hx = h.val_of_col(j);
					let tau_inv = recip(real(tau[j]));

					if hi.is_empty() {
						tmp.rb_mut().row_mut(j).fill(zero());
						continue;
					}

					let hi0 = hi[0].zx();
					for k in 0..x.ncols() {
						let mut dot = zero::<T>();
						for (i, v) in iter::zip(hi, hx) {
							let i = i.zx();
							let v = if conj_qr == Conj::Yes { copy(*v) } else { conj(*v) };
							dot = dot + v * x[(i, k)];
						}
						dot = mul_real(dot, tau_inv);
						for (i, v) in iter::zip(hi, hx) {
							let i = i.zx();
							let v = if conj_qr == Conj::Yes { conj(*v) } else { copy(*v) };
							x[(i, k)] = x[(i, k)] - dot * v;
						}

						tmp.rb_mut().row_mut(j).copy_from(x.rb().row(hi0));
					}
				}
			}
			x.rb_mut().subrows_mut(0, n).copy_from(tmp.rb().subrows(0, n));
			x.rb_mut().subrows_mut(n, m - n).fill(zero());
		}

		/// solves the equation $A x = \text{rhs}$ in the sense of least squares, implicitly
		/// conjugating $A$ if needed
		///
		/// `work` is a temporary workspace with the same dimensions as `rhs`.
		#[track_caller]
		#[math]
		pub fn solve_in_place_with_conj(&self, conj_qr: Conj, rhs: MatMut<'_, T>, par: Par, work: MatMut<'_, T>)
		where
			T: ComplexField,
		{
			let mut work = work;
			let mut rhs = rhs;
			self.apply_qt_in_place_with_conj(conj_qr, rhs.rb_mut(), par, work.rb_mut());

			let _ = par;
			assert!(rhs.nrows() == self.symbolic.nrows);
			let mut x = rhs;

			let n = self.symbolic.ncols;
			let r = SparseColMatRef::<'_, I, T>::new(
				unsafe { SymbolicSparseColMatRef::new_unchecked(n, n, self.r_col_ptr, None, self.r_row_idx) },
				self.r_val,
			);

			linalg_sp::triangular_solve::solve_upper_triangular_in_place(r, conj_qr, x.rb_mut().subrows_mut(0, n), par);
		}
	}

	/// computes the size and alignment of the workspace required to compute the symbolic $QR$
	/// factorization of a matrix with dimensions `(nrows, ncols)`
	pub fn factorize_simplicial_symbolic_qr_scratch<I: Index>(nrows: usize, ncols: usize) -> StackReq {
		let _ = nrows;
		StackReq::new::<I>(ncols).array(3)
	}

	/// computes the symbolic $QR$ factorization of a matrix $A$, given the outputs of the
	/// pre-factorization steps
	pub fn factorize_simplicial_symbolic_qr<I: Index>(
		min_col: &[I],
		etree: EliminationTreeRef<'_, I>,
		col_counts: &[I],
		stack: &mut MemStack,
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

		let h_nnz = I::sum_nonnegative(h_non_zero_count).ok_or(FaerError::IndexOverflow)?.zx();
		let l_nnz = I::sum_nonnegative(col_counts).ok_or(FaerError::IndexOverflow)?.zx();

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

	/// computes the size and alignment of the workspace required to compute the numerical $QR$
	/// factorization of the matrix whose structure was used to produce the symbolic structure
	pub fn factorize_simplicial_numeric_qr_scratch<I: Index, T: ComplexField>(symbolic: &SymbolicSimplicialQr<I>) -> StackReq {
		let m = symbolic.nrows;
		StackReq::all_of(&[
			StackReq::new::<I>(m),
			StackReq::new::<I>(m),
			StackReq::new::<I>(m),
			temp_mat_scratch::<T>(m, 1),
		])
	}

	/// computes the numerical $QR$ factorization of $A$.
	///
	/// - `r_col_ptr` has length `a.ncols() + 1`
	/// - `r_row_idx` has length `symbolic.len_r()`
	/// - `r_val` has length `symbolic.len_r()`
	/// - `householder_col_ptr` has length `a.ncols() + 1`
	/// - `householder_row_idx` has length `symbolic.len_householder()`
	/// - `householder_val` has length `symbolic.len_householder()`
	/// - `tau_val` has length `a.ncols()`
	#[math]
	pub fn factorize_simplicial_numeric_qr_unsorted<'a, I: Index, T: ComplexField>(
		r_col_ptr: &'a mut [I],
		r_row_idx: &'a mut [I],
		r_val: &'a mut [T],
		householder_col_ptr: &'a mut [I],
		householder_row_idx: &'a mut [I],
		householder_val: &'a mut [T],
		tau_val: &'a mut [T],

		A: SparseColMatRef<'_, I, T>,
		col_perm: Option<PermRef<'_, I>>,
		symbolic: &'a SymbolicSimplicialQr<I>,
		stack: &mut MemStack,
	) -> SimplicialQrRef<'a, I, T> {
		assert!(all(A.nrows() == symbolic.nrows, A.ncols() == symbolic.ncols,));

		let I = I::truncate;
		let m = A.nrows();
		let n = A.ncols();
		let (r_idx, stack) = unsafe { stack.make_raw::<I::Signed>(m) };
		let (marked, stack) = unsafe { stack.make_raw::<I>(m) };
		let (pattern, stack) = unsafe { stack.make_raw::<I>(m) };
		let (mut x, _) = temp_mat_zeroed::<T, _, _>(m, 1, stack);
		let x = x.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap().as_slice_mut();
		marked.fill(I(0));
		r_idx.fill(I::Signed::truncate(NONE));

		r_col_ptr[0] = I(0);
		let mut r_pos = 0usize;
		let mut h_pos = 0usize;
		for j in 0..n {
			let pj = col_perm.map(|perm| perm.arrays().0[j].zx()).unwrap_or(j);

			let mut pattern_len = 0usize;
			for (i, val) in iter::zip(A.row_idx_of_col(pj), A.val_of_col(pj)) {
				if marked[i] < I(j + 1) {
					marked[i] = I(j + 1);
					pattern[pattern_len] = I(i);
					pattern_len += 1;
				}
				x[i] = x[i] + *val;
			}

			let j_postordered = symbolic.postorder_inv[j].zx();
			let desc_count = symbolic.desc_count[j].zx();
			for d in &symbolic.postorder[j_postordered - desc_count..j_postordered] {
				let d = d.zx();

				let d_h_pattern = &householder_row_idx[householder_col_ptr[d].zx()..householder_col_ptr[d + 1].zx()];
				let d_h_val = &householder_val[householder_col_ptr[d].zx()..householder_col_ptr[d + 1].zx()];

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

				let tau_inv = recip(real(tau_val[d]));
				let mut dot = zero::<T>();
				for (i, vi) in iter::zip(d_h_pattern, d_h_val) {
					let i = i.zx();
					dot = dot + conj(*vi) * x[i];
				}
				dot = mul_real(dot, tau_inv);
				for (i, vi) in iter::zip(d_h_pattern, d_h_val) {
					let i = i.zx();
					x[i] = x[i] - dot * *vi;
				}
			}
			let pattern = &pattern[..pattern_len];

			let h_begin = h_pos;
			for i in pattern.iter() {
				let i = i.zx();
				if r_idx[i] >= I(0).to_signed() {
					r_val[r_pos] = copy(x[i]);
					x[i] = zero();
					r_row_idx[r_pos] = I::from_signed(r_idx[i]);
					r_pos += 1;
				} else {
					householder_val[h_pos] = copy(x[i]);
					x[i] = zero();
					householder_row_idx[h_pos] = I(i);
					h_pos += 1;
				}
			}

			householder_col_ptr[j + 1] = I(h_pos);

			if h_begin == h_pos {
				tau_val[j] = zero();
				r_val[r_pos] = zero();
				r_row_idx[r_pos] = I(j);
				r_pos += 1;
				r_col_ptr[j + 1] = I(r_pos);
				continue;
			}

			let mut h_col = ColMut::from_slice_mut(&mut householder_val[h_begin..h_pos]);

			let (mut head, tail) = h_col.rb_mut().split_at_row_mut(1);
			let head = &mut head[0];
			let crate::linalg::householder::HouseholderInfo { tau, .. } = crate::linalg::householder::make_householder_in_place(head, tail);
			tau_val[j] = from_real(tau);
			r_val[r_pos] = copy(*head);
			*head = one();

			r_row_idx[r_pos] = I(j);
			r_idx[householder_row_idx[h_begin].zx()] = I(j).to_signed();
			r_pos += 1;
			r_col_ptr[j + 1] = I(r_pos);
		}

		unsafe {
			SimplicialQrRef::new(
				symbolic,
				SparseColMatRef::<'_, I, T>::new(SymbolicSparseColMatRef::new_unchecked(n, n, r_col_ptr, None, r_row_idx), r_val),
				SparseColMatRef::<'_, I, T>::new(
					SymbolicSparseColMatRef::new_unchecked(m, n, householder_col_ptr, None, householder_row_idx),
					householder_val,
				),
				tau_val,
			)
		}
	}
}

/// tuning parameters for the $QR$ symbolic factorization
#[derive(Copy, Clone, Debug, Default)]
pub struct QrSymbolicParams<'a> {
	/// parameters for the fill reducing column permutation
	pub colamd_params: colamd::Control,
	/// threshold for selecting the supernodal factorization
	pub supernodal_flop_ratio_threshold: SupernodalThreshold,
	/// supernodal factorization parameters
	pub supernodal_params: SymbolicSupernodalParams<'a>,
}

/// the inner factorization used for the symbolic $QR$, either simplicial or symbolic
#[derive(Debug)]
pub enum SymbolicQrRaw<I> {
	/// simplicial structure
	Simplicial(simplicial::SymbolicSimplicialQr<I>),
	/// supernodal structure
	Supernodal(supernodal::SymbolicSupernodalQr<I>),
}

/// the symbolic structure of a sparse $QR$ decomposition
#[derive(Debug)]
pub struct SymbolicQr<I> {
	raw: SymbolicQrRaw<I>,
	col_perm_fwd: alloc::vec::Vec<I>,
	col_perm_inv: alloc::vec::Vec<I>,
	A_nnz: usize,
}

/// sparse $QR$ factorization wrapper
#[derive(Debug)]
pub struct QrRef<'a, I: Index, T> {
	symbolic: &'a SymbolicQr<I>,
	indices: &'a [I],
	val: &'a [T],
}

impl<I: Index, T> Copy for QrRef<'_, I, T> {}
impl<I: Index, T> Clone for QrRef<'_, I, T> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<'a, I: Index, T> QrRef<'a, I, T> {
	/// creates a $QR$ decomposition reference from its symbolic and numerical components
	///
	/// # safety
	/// the indices must be filled by a previous call to [`SymbolicQr::factorize_numeric_qr`] with
	/// the right parameters
	#[inline]
	pub unsafe fn new_unchecked(symbolic: &'a SymbolicQr<I>, indices: &'a [I], val: &'a [T]) -> Self {
		let val = val;
		assert!(all(symbolic.len_val() == val.len(), symbolic.len_idx() == indices.len(),));
		Self { symbolic, val, indices }
	}

	/// returns the symbolic structure of the $QR$ factorization.
	#[inline]
	pub fn symbolic(self) -> &'a SymbolicQr<I> {
		self.symbolic
	}

	/// solves the equation $A x = \text{rhs}$ in the sense of least squares, implicitly conjugating
	/// $A$ if needed
	///
	/// `work` is a temporary workspace with the same dimensions as `rhs`
	#[track_caller]
	pub fn solve_in_place_with_conj(self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
	where
		T: ComplexField,
	{
		let k = rhs.ncols();
		let m = self.symbolic.nrows();
		let n = self.symbolic.ncols();

		assert!(all(rhs.nrows() == self.symbolic.nrows(), self.symbolic.nrows() >= self.symbolic.ncols(),));
		let mut rhs = rhs;

		let (mut x, stack) = unsafe { temp_mat_uninit::<T, _, _>(m, k, stack) };
		let mut x = x.as_mat_mut();

		let (_, inv) = self.symbolic.col_perm().arrays();
		x.copy_from(rhs.rb());

		let indices = self.indices;
		let val = self.val;

		match &self.symbolic.raw {
			SymbolicQrRaw::Simplicial(symbolic) => {
				let (r_col_ptr, indices) = indices.split_at(n + 1);
				let (r_row_idx, indices) = indices.split_at(symbolic.len_r());
				let (householder_col_ptr, indices) = indices.split_at(n + 1);
				let (householder_row_idx, _) = indices.split_at(symbolic.len_householder());

				let (r_val, val) = val.rb().split_at(symbolic.len_r());
				let (householder_val, val) = val.split_at(symbolic.len_householder());
				let (tau_val, _) = val.split_at(n);

				let r = SparseColMatRef::<'_, I, T>::new(unsafe { SymbolicSparseColMatRef::new_unchecked(n, n, r_col_ptr, None, r_row_idx) }, r_val);
				let h = SparseColMatRef::<'_, I, T>::new(
					unsafe { SymbolicSparseColMatRef::new_unchecked(m, n, householder_col_ptr, None, householder_row_idx) },
					householder_val,
				);

				let this = simplicial::SimplicialQrRef::<'_, I, T>::new(symbolic, r, h, tau_val);
				this.solve_in_place_with_conj(conj, x.rb_mut(), par, rhs.rb_mut());
			},
			SymbolicQrRaw::Supernodal(symbolic) => {
				let (householder_row_idx, indices) = indices.split_at(symbolic.householder().len_householder_row_idx());
				let (tau_blocksize, indices) =
					indices.split_at(symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes());
				let (householder_nrows, indices) =
					indices.split_at(symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes());
				let (householder_ncols, _) =
					indices.split_at(symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes());

				let (r_val, val) = val.rb().split_at(symbolic.R_adjoint().len_val());
				let (householder_val, val) = val.split_at(symbolic.householder().len_householder_val());
				let (tau_val, _) = val.split_at(symbolic.householder().len_tau_val());

				let this = unsafe {
					supernodal::SupernodalQrRef::<'_, I, T>::new_unchecked(
						symbolic,
						householder_row_idx,
						tau_blocksize,
						householder_nrows,
						householder_ncols,
						r_val,
						householder_val,
						tau_val,
					)
				};
				this.solve_in_place_with_conj(conj, x.rb_mut(), par, rhs.rb_mut(), stack);
			},
		}

		for j in 0..k {
			for (i, p) in inv.iter().enumerate() {
				rhs[(i, j)] = copy(&x[(p.zx(), j)]);
			}
		}
	}
}

impl<I: Index> SymbolicQr<I> {
	/// number of rows of $A$
	#[inline]
	pub fn nrows(&self) -> usize {
		match &self.raw {
			SymbolicQrRaw::Simplicial(this) => this.nrows(),
			SymbolicQrRaw::Supernodal(this) => this.householder().nrows(),
		}
	}

	/// number of columns of $A$
	#[inline]
	pub fn ncols(&self) -> usize {
		match &self.raw {
			SymbolicQrRaw::Simplicial(this) => this.ncols(),
			SymbolicQrRaw::Supernodal(this) => this.R_adjoint().ncols(),
		}
	}

	/// returns the fill-reducing column permutation that was computed during symbolic analysis
	#[inline]
	pub fn col_perm(&self) -> PermRef<'_, I> {
		unsafe { PermRef::new_unchecked(&self.col_perm_fwd, &self.col_perm_inv, self.ncols()) }
	}

	/// returns the length of the slice needed to store the symbolic indices of the $QR$
	/// decomposition
	#[inline]
	pub fn len_idx(&self) -> usize {
		match &self.raw {
			SymbolicQrRaw::Simplicial(symbolic) => symbolic.len_r() + symbolic.len_householder() + 2 * self.ncols() + 2,
			SymbolicQrRaw::Supernodal(symbolic) => 4 * symbolic.householder().len_householder_row_idx() + 3 * symbolic.householder().n_supernodes(),
		}
	}

	/// returns the length of the slice needed to store the numerical values of the $QR$
	/// decomposition
	#[inline]
	pub fn len_val(&self) -> usize {
		match &self.raw {
			SymbolicQrRaw::Simplicial(symbolic) => symbolic.len_r() + symbolic.len_householder() + self.ncols(),
			SymbolicQrRaw::Supernodal(symbolic) => {
				symbolic.householder().len_householder_val() + symbolic.R_adjoint().len_val() + symbolic.householder().len_tau_val()
			},
		}
	}

	/// returns the size and alignment of the workspace required to solve the system $A x =
	/// \text{rhs}$ in the sense of least squares
	pub fn solve_in_place_scratch<T>(&self, rhs_ncols: usize, par: Par) -> StackReq
	where
		T: ComplexField,
	{
		temp_mat_scratch::<T>(self.nrows(), rhs_ncols).and(match &self.raw {
			SymbolicQrRaw::Simplicial(_) => StackReq::empty(),
			SymbolicQrRaw::Supernodal(this) => this.solve_in_place_scratch::<T>(rhs_ncols, par),
		})
	}

	/// computes the required workspace size and alignment for a numerical $QR$ factorization
	pub fn factorize_numeric_qr_scratch<T>(&self, par: Par, params: Spec<QrParams, T>) -> StackReq
	where
		T: ComplexField,
	{
		let m = self.nrows();
		let A_nnz = self.A_nnz;
		let AT_scratch = StackReq::all_of(&[temp_mat_scratch::<T>(A_nnz, 1), StackReq::new::<I>(m + 1), StackReq::new::<I>(A_nnz)]);

		match &self.raw {
			SymbolicQrRaw::Simplicial(symbolic) => simplicial::factorize_simplicial_numeric_qr_scratch::<I, T>(symbolic),
			SymbolicQrRaw::Supernodal(symbolic) => StackReq::and(
				AT_scratch,
				supernodal::factorize_supernodal_numeric_qr_scratch::<I, T>(symbolic, par, params),
			),
		}
	}

	/// computes a numerical $QR$ factorization of $A$
	#[track_caller]
	pub fn factorize_numeric_qr<'out, T: ComplexField>(
		&'out self,
		indices: &'out mut [I],
		val: &'out mut [T],
		A: SparseColMatRef<'_, I, T>,
		par: Par,
		stack: &mut MemStack,
		params: Spec<QrParams, T>,
	) -> QrRef<'out, I, T> {
		assert!(all(val.len() == self.len_val(), indices.len() == self.len_idx(),));
		assert!(all(A.nrows() == self.nrows(), A.ncols() == self.ncols()));

		let m = A.nrows();
		let n = A.ncols();

		match &self.raw {
			SymbolicQrRaw::Simplicial(symbolic) => {
				let (r_col_ptr, indices) = indices.split_at_mut(n + 1);
				let (r_row_idx, indices) = indices.split_at_mut(symbolic.len_r());
				let (householder_col_ptr, indices) = indices.split_at_mut(n + 1);
				let (householder_row_idx, _) = indices.split_at_mut(symbolic.len_householder());

				let (r_val, val) = val.split_at_mut(symbolic.len_r());
				let (householder_val, val) = val.split_at_mut(symbolic.len_householder());
				let (tau_val, _) = val.split_at_mut(n);

				simplicial::factorize_simplicial_numeric_qr_unsorted::<I, T>(
					r_col_ptr,
					r_row_idx,
					r_val,
					householder_col_ptr,
					householder_row_idx,
					householder_val,
					tau_val,
					A,
					Some(self.col_perm()),
					symbolic,
					stack,
				);
			},
			SymbolicQrRaw::Supernodal(symbolic) => {
				let (householder_row_idx, indices) = indices.split_at_mut(symbolic.householder().len_householder_row_idx());
				let (tau_blocksize, indices) =
					indices.split_at_mut(symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes());
				let (householder_nrows, indices) =
					indices.split_at_mut(symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes());
				let (householder_ncols, _) =
					indices.split_at_mut(symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes());

				let (r_val, val) = val.split_at_mut(symbolic.R_adjoint().len_val());
				let (householder_val, val) = val.split_at_mut(symbolic.householder().len_householder_val());
				let (tau_val, _) = val.split_at_mut(symbolic.householder().len_tau_val());

				let (new_col_ptr, stack) = unsafe { stack.make_raw::<I>(m + 1) };
				let (new_row_idx, stack) = unsafe { stack.make_raw::<I>(self.A_nnz) };
				let (mut new_val, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(self.A_nnz, 1, stack) };
				let new_val = new_val.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap().as_slice_mut();

				let AT = utils::transpose(new_val, new_col_ptr, new_row_idx, A, stack.rb_mut()).into_const();

				supernodal::factorize_supernodal_numeric_qr::<I, T>(
					householder_row_idx,
					tau_blocksize,
					householder_nrows,
					householder_ncols,
					r_val,
					householder_val,
					tau_val,
					AT,
					Some(self.col_perm()),
					symbolic,
					par,
					stack,
					params,
				);
			},
		}

		unsafe { QrRef::new_unchecked(self, indices, val) }
	}
}

/// computes the symbolic $QR$ factorization of the matrix $A$, or returns an error if the
/// operation could not be completed
#[track_caller]
pub fn factorize_symbolic_qr<I: Index>(A: SymbolicSparseColMatRef<'_, I>, params: QrSymbolicParams<'_>) -> Result<SymbolicQr<I>, FaerError> {
	assert!(A.nrows() >= A.ncols());
	let m = A.nrows();
	let n = A.ncols();
	let A_nnz = A.compute_nnz();

	with_dim!(M, m);
	with_dim!(N, n);
	let A = A.as_shape(M, N);

	let req = {
		let n_scratch = StackReq::new::<I>(n);
		let m_scratch = StackReq::new::<I>(m);
		let AT_scratch = StackReq::and(
			// new_col_ptr
			StackReq::new::<I>(m + 1),
			// new_row_idx
			StackReq::new::<I>(A_nnz),
		);

		StackReq::or(
			colamd::order_scratch::<I>(m, n, A_nnz),
			StackReq::all_of(&[
				n_scratch,
				n_scratch,
				n_scratch,
				n_scratch,
				AT_scratch,
				StackReq::any_of(&[
					StackReq::and(n_scratch, m_scratch),
					StackReq::all_of(&[n_scratch; 3]),
					StackReq::all_of(&[n_scratch, n_scratch, n_scratch, n_scratch, n_scratch, m_scratch]),
					supernodal::factorize_supernodal_symbolic_qr_scratch::<I>(m, n),
					simplicial::factorize_simplicial_symbolic_qr_scratch::<I>(m, n),
				]),
			]),
		)
	};

	let mut mem = dyn_stack::MemBuffer::try_new(req).ok().ok_or(FaerError::OutOfMemory)?;
	let mut stack = MemStack::new(&mut mem);

	let mut col_perm_fwd = try_zeroed::<I>(n)?;
	let mut col_perm_inv = try_zeroed::<I>(n)?;
	let mut min_row = try_zeroed::<I>(m)?;

	colamd::order(&mut col_perm_fwd, &mut col_perm_inv, A.as_dyn(), params.colamd_params, stack.rb_mut())?;

	let col_perm = PermRef::new_checked(&col_perm_fwd, &col_perm_inv, n).as_shape(N);

	let (new_col_ptr, stack) = unsafe { stack.make_raw::<I>(m + 1) };
	let (new_row_idx, mut stack) = unsafe { stack.make_raw::<I>(A_nnz) };
	let AT = utils::adjoint(
		Symbolic::materialize(new_row_idx.len()),
		new_col_ptr,
		new_row_idx,
		SparseColMatRef::new(A, Symbolic::materialize(A.row_idx().len())),
		stack.rb_mut(),
	)
	.symbolic();

	let (etree, stack) = unsafe { stack.make_raw::<I::Signed>(n) };
	let (post, stack) = unsafe { stack.make_raw::<I>(n) };
	let (col_counts, stack) = unsafe { stack.make_raw::<I>(n) };
	let (h_col_counts, mut stack) = unsafe { stack.make_raw::<I>(n) };

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
	if threshold != SupernodalThreshold::FORCE_SIMPLICIAL && threshold != SupernodalThreshold::FORCE_SUPERNODAL {
		h_col_counts.fill(I::truncate(0));
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

		if flops / nnz > threshold.0 * linalg_sp::QR_SUPERNODAL_RATIO_FACTOR {
			threshold = SupernodalThreshold::FORCE_SUPERNODAL;
		} else {
			threshold = SupernodalThreshold::FORCE_SIMPLICIAL;
		}
	}

	if threshold == SupernodalThreshold::FORCE_SUPERNODAL {
		let symbolic = supernodal::factorize_supernodal_symbolic_qr::<I>(
			A.as_dyn(),
			Some(col_perm.as_shape(n)),
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
		let symbolic =
			simplicial::factorize_simplicial_symbolic_qr::<I>(&min_col, EliminationTreeRef::<'_, I> { inner: etree }, col_counts, stack.rb_mut())?;
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
	use crate::assert;
	use crate::stats::prelude::*;
	use dyn_stack::MemBuffer;
	use linalg::solvers::SolveLstsqCore;
	use linalg_sp::cholesky::tests::{load_mtx, reconstruct_from_supernodal_llt};
	use matrix_market_rs::MtxData;
	use std::path::PathBuf;

	#[test]
	fn test_symbolic_qr() {
		let n = 11;
		let col_ptr = &[0, 3, 6, 10, 13, 16, 21, 24, 29, 31, 37, 43usize];
		let row_idx = &[
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
			2, 4, 6, 7, 9, 10usize, // 10
		];

		let A = SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_idx);
		let mut etree = vec![0isize; n];
		let mut post = vec![0usize; n];
		let mut col_counts = vec![0usize; n];

		with_dim!(N, n);
		let A = A.as_shape(N, N);
		ghost_col_etree(
			A,
			None,
			Array::from_mut(&mut etree, N),
			MemStack::new(&mut MemBuffer::new(StackReq::new::<usize>(*N + *N))),
		);
		let etree = Array::from_ref(MaybeIdx::from_slice_ref_checked(&etree, N), N);
		ghost_postorder(
			Array::from_mut(&mut post, N),
			etree,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<usize>(20 * *N))),
		);

		let mut min_row = vec![0usize.to_signed(); n];
		let mut new_col_ptr = vec![0usize; n + 1];
		let mut new_row_idx = vec![0usize; 43];

		let AT = utils::adjoint(
			Symbolic::materialize(new_row_idx.len()),
			&mut new_col_ptr,
			&mut new_row_idx,
			SparseColMatRef::new(A, Symbolic::materialize(A.row_idx().len())),
			MemStack::new(&mut MemBuffer::new(StackReq::new::<usize>(20 * *N))),
		)
		.symbolic();
		ghost_column_counts_aat(
			Array::from_mut(&mut col_counts, N),
			Array::from_mut(&mut min_row, N),
			AT,
			None,
			etree,
			Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
			MemStack::new(&mut MemBuffer::new(StackReq::new::<usize>(20 * *N))),
		);

		assert!(MaybeIdx::<'_, usize>::as_slice_ref(etree.as_ref()) == [3, 2, 3, 4, 5, 6, 7, 8, 9, 10, NONE as isize]);
		assert!(col_counts == [7, 6, 8, 8, 7, 6, 5, 4, 3, 2, 1usize]);
	}

	#[test]
	fn test_numeric_qr_1_no_transpose() {
		type I = usize;

		let (m, n, col_ptr, row_idx, val) =
			load_mtx::<usize>(MtxData::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_qr/lp_share2b.mtx")).unwrap());

		let nnz = row_idx.len();

		let A = SparseColMatRef::<'_, I, f64>::new(SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_idx), &val);

		with_dim!(M, m);
		with_dim!(N, n);

		let A = A.as_shape(M, N);
		let mut new_col_ptr = vec![0usize; m + 1];
		let mut new_row_idx = vec![0usize; nnz];
		let mut new_val = vec![0.0; nnz];

		let AT = utils::adjoint(
			&mut new_val,
			&mut new_col_ptr,
			&mut new_row_idx,
			A,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(20 * *N))),
		)
		.into_const();

		let mut etree = vec![0usize.to_signed(); n];
		let mut post = vec![0usize; n];
		let mut col_counts = vec![0usize; n];
		let mut min_row = vec![0usize; m];

		ghost_col_etree(
			A.symbolic(),
			None,
			Array::from_mut(&mut etree, N),
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(*M + *N))),
		);
		let etree_ = Array::from_ref(MaybeIdx::from_slice_ref_checked(&etree, N), N);
		ghost_postorder(
			Array::from_mut(&mut post, N),
			etree_,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(20 * *N))),
		);

		ghost_column_counts_aat(
			Array::from_mut(&mut col_counts, N),
			Array::from_mut(bytemuck::cast_slice_mut(&mut min_row), M),
			AT.symbolic(),
			None,
			etree_,
			Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(20 * *N))),
		);

		let min_col = min_row;

		let symbolic = supernodal::factorize_supernodal_symbolic_qr::<I>(
			A.symbolic().as_dyn(),
			None,
			min_col,
			EliminationTreeRef::<'_, I> { inner: &etree },
			&col_counts,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(20 * *N))),
			Default::default(),
		)
		.unwrap();

		let mut householder_row_idx = vec![0usize; symbolic.householder().len_householder_row_idx()];

		let mut L_val = vec![0.0; symbolic.R_adjoint().len_val()];
		let mut householder_val = vec![0.0; symbolic.householder().len_householder_val()];
		let mut tau_val = vec![0.0; symbolic.householder().len_tau_val()];

		let mut tau_blocksize = vec![0usize; symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes()];
		let mut householder_nrows = vec![0usize; symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes()];
		let mut householder_ncols = vec![0usize; symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes()];

		supernodal::factorize_supernodal_numeric_qr::<I, f64>(
			&mut householder_row_idx,
			&mut tau_blocksize,
			&mut householder_nrows,
			&mut householder_ncols,
			&mut L_val,
			&mut householder_val,
			&mut tau_val,
			AT.as_dyn(),
			None,
			&symbolic,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(supernodal::factorize_supernodal_numeric_qr_scratch::<usize, f64>(
				&symbolic,
				Par::Seq,
				Default::default(),
			))),
			Default::default(),
		);
		let llt = reconstruct_from_supernodal_llt::<I, f64>(symbolic.R_adjoint(), &L_val);
		let a = A.as_dyn().to_dense();
		let ata = a.adjoint() * &a;

		let llt_diff = &llt - &ata;
		assert!(llt_diff.norm_max() <= 1e-10);
	}

	#[test]
	fn test_numeric_qr_1_transpose() {
		type I = usize;
		type T = c64;

		let mut gen = rand::rngs::StdRng::seed_from_u64(0);

		let (m, n, col_ptr, row_idx, val) =
			load_mtx::<usize>(MtxData::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_qr/lp_share2b.mtx")).unwrap());
		let val = val.iter().map(|&x| c64::new(x, gen.gen())).collect::<Vec<_>>();

		let nnz = row_idx.len();

		let A = SparseColMatRef::<'_, I, T>::new(SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_idx), &val);

		with_dim!(M, m);
		with_dim!(N, n);
		let A = A.as_shape(M, N);
		let mut new_col_ptr = vec![0usize; m + 1];
		let mut new_row_idx = vec![0usize; nnz];
		let mut new_val = vec![T::ZERO; nnz];

		let AT = utils::transpose(
			&mut new_val,
			&mut new_col_ptr,
			&mut new_row_idx,
			A,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(*M))),
		)
		.into_const();

		let (A, AT) = (AT, A);
		let (M, N) = (N, M);
		let (m, n) = (n, m);

		let mut etree = vec![0usize.to_signed(); n];
		let mut post = vec![0usize; n];
		let mut col_counts = vec![0usize; n];
		let mut min_row = vec![0usize; m];

		ghost_col_etree(
			A.symbolic(),
			None,
			Array::from_mut(&mut etree, N),
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(*M + *N))),
		);
		let etree_ = Array::from_ref(MaybeIdx::from_slice_ref_checked(&etree, N), N);
		ghost_postorder(
			Array::from_mut(&mut post, N),
			etree_,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(3 * *N))),
		);

		ghost_column_counts_aat(
			Array::from_mut(&mut col_counts, N),
			Array::from_mut(bytemuck::cast_slice_mut(&mut min_row), M),
			AT.symbolic(),
			None,
			etree_,
			Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(5 * *N + *M))),
		);

		let min_col = min_row;

		let symbolic = supernodal::factorize_supernodal_symbolic_qr::<I>(
			A.symbolic().as_dyn(),
			None,
			min_col,
			EliminationTreeRef::<'_, I> { inner: &etree },
			&col_counts,
			MemStack::new(&mut MemBuffer::new(supernodal::factorize_supernodal_symbolic_qr_scratch::<usize>(*M, *N))),
			Default::default(),
		)
		.unwrap();

		let mut householder_row_idx = vec![0usize; symbolic.householder().len_householder_row_idx()];

		let mut L_val = vec![T::ZERO; symbolic.R_adjoint().len_val()];
		let mut householder_val = vec![T::ZERO; symbolic.householder().len_householder_val()];
		let mut tau_val = vec![T::ZERO; symbolic.householder().len_tau_val()];

		let mut tau_blocksize = vec![0usize; symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes()];
		let mut householder_nrows = vec![0usize; symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes()];
		let mut householder_ncols = vec![0usize; symbolic.householder().len_householder_row_idx() + symbolic.householder().n_supernodes()];

		let qr = supernodal::factorize_supernodal_numeric_qr::<I, T>(
			&mut householder_row_idx,
			&mut tau_blocksize,
			&mut householder_nrows,
			&mut householder_ncols,
			&mut L_val,
			&mut householder_val,
			&mut tau_val,
			AT.as_dyn(),
			None,
			&symbolic,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(supernodal::factorize_supernodal_numeric_qr_scratch::<usize, T>(
				&symbolic,
				Par::Seq,
				Default::default(),
			))),
			Default::default(),
		);

		let a = A.as_dyn().to_dense();

		let rhs = Mat::<T>::from_fn(m, 2, |_, _| c64::new(gen.gen(), gen.gen()));
		let mut x = rhs.clone();
		let mut work = rhs.clone();
		qr.solve_in_place_with_conj(
			Conj::No,
			x.as_mut(),
			Par::Seq,
			work.as_mut(),
			MemStack::new(&mut MemBuffer::new(symbolic.solve_in_place_scratch::<T>(2, Par::Seq))),
		);
		let x = x.as_ref().subrows(0, n);

		let linsolve_diff = a.adjoint() * (&a * &x - &rhs);

		let llt = reconstruct_from_supernodal_llt::<I, T>(symbolic.R_adjoint(), &L_val);
		let ata = a.adjoint() * &a;

		let llt_diff = &llt - &ata;
		assert!(llt_diff.norm_max() <= 1e-10);
		assert!(linsolve_diff.norm_max() <= 1e-10);
	}

	#[test]
	fn test_numeric_simplicial_qr_1_transpose() {
		type I = usize;
		type T = c64;

		let mut gen = rand::rngs::StdRng::seed_from_u64(0);

		let (m, n, col_ptr, row_idx, val) =
			load_mtx::<usize>(MtxData::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_qr/lp_share2b.mtx")).unwrap());

		let val = val.iter().map(|&x| c64::new(x, gen.gen())).collect::<Vec<_>>();

		let nnz = row_idx.len();

		let A = SparseColMatRef::<'_, I, T>::new(SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_idx), &val);

		with_dim!(M, m);
		with_dim!(N, n);
		let A = A.as_shape(M, N);
		let mut new_col_ptr = vec![0usize; m + 1];
		let mut new_row_idx = vec![0usize; nnz];
		let mut new_val = vec![T::ZERO; nnz];

		let AT = utils::transpose(
			&mut new_val,
			&mut new_col_ptr,
			&mut new_row_idx,
			A,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(*M))),
		)
		.into_const();

		let (A, AT) = (AT, A);
		let (M, N) = (N, M);
		let (m, n) = (n, m);

		let mut etree = vec![0usize.to_signed(); n];
		let mut post = vec![0usize; n];
		let mut col_counts = vec![0usize; n];
		let mut min_row = vec![0usize; m];

		ghost_col_etree(
			A.symbolic(),
			None,
			Array::from_mut(&mut etree, N),
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(*M + *N))),
		);
		let etree_ = Array::from_ref(MaybeIdx::from_slice_ref_checked(&etree, N), N);
		ghost_postorder(
			Array::from_mut(&mut post, N),
			etree_,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(3 * *N))),
		);

		ghost_column_counts_aat(
			Array::from_mut(&mut col_counts, N),
			Array::from_mut(bytemuck::cast_slice_mut(&mut min_row), M),
			AT.symbolic(),
			None,
			etree_,
			Array::from_ref(Idx::from_slice_ref_checked(&post, N), N),
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(5 * *N + *M))),
		);

		let min_col = min_row;

		let symbolic = simplicial::factorize_simplicial_symbolic_qr::<I>(
			&min_col,
			EliminationTreeRef::<'_, I> { inner: &etree },
			&col_counts,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(3 * *N))),
		)
		.unwrap();

		let mut r_col_ptr = vec![0usize; n + 1];
		let mut r_row_idx = vec![0usize; symbolic.len_r()];
		let mut householder_col_ptr = vec![0usize; n + 1];
		let mut householder_row_idx = vec![0usize; symbolic.len_householder()];

		let mut r_val = vec![T::ZERO; symbolic.len_r()];
		let mut householder_val = vec![T::ZERO; symbolic.len_householder()];
		let mut tau_val = vec![T::ZERO; n];

		let qr = simplicial::factorize_simplicial_numeric_qr_unsorted(
			&mut r_col_ptr,
			&mut r_row_idx,
			&mut r_val,
			&mut householder_col_ptr,
			&mut householder_row_idx,
			&mut householder_val,
			&mut tau_val,
			A.as_dyn(),
			None,
			&symbolic,
			MemStack::new(&mut MemBuffer::new(simplicial::factorize_simplicial_numeric_qr_scratch::<usize, T>(
				&symbolic,
			))),
		);

		let a = A.as_dyn().to_dense();
		let rhs = Mat::<T>::from_fn(m, 2, |_, _| c64::new(gen.gen(), gen.gen()));
		{
			let mut x = rhs.clone();
			let mut work = rhs.clone();
			qr.solve_in_place_with_conj(Conj::No, x.as_mut(), Par::Seq, work.as_mut());

			let mut y = rhs.clone();
			A.to_dense().as_dyn().qr().solve_lstsq_in_place_with_conj(Conj::No, y.as_mut());

			let x = x.as_ref().subrows(0, n);
			let linsolve_diff = a.adjoint() * (&a * &x - &rhs);
			assert!(linsolve_diff.norm_max() <= 1e-10);
		}
		{
			let mut x = rhs.clone();
			let mut work = rhs.clone();
			qr.solve_in_place_with_conj(Conj::Yes, x.as_mut(), Par::Seq, work.as_mut());

			let x = x.as_ref().subrows(0, n);
			let a = a.conjugate();
			let linsolve_diff = a.adjoint() * (a * &x - &rhs);
			assert!(linsolve_diff.norm_max() <= 1e-10);
		}

		let R = SparseColMatRef::<'_, usize, T>::new(SymbolicSparseColMatRef::new_unsorted_checked(n, n, &r_col_ptr, None, &r_row_idx), &r_val);
		let r = R.to_dense();
		let ata = a.adjoint() * &a;
		let rtr = r.adjoint() * &r;
		assert!((&ata - &rtr).norm_max() < 1e-10);
	}

	#[test]
	fn test_solver_qr_1_transpose() {
		type I = usize;
		type T = c64;

		let mut gen = rand::rngs::StdRng::seed_from_u64(0);

		let (m, n, col_ptr, row_idx, val) =
			load_mtx::<usize>(MtxData::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_qr/lp_share2b.mtx")).unwrap());
		let val = val.iter().map(|&x| c64::new(x, gen.gen())).collect::<Vec<_>>();
		let nnz = row_idx.len();

		let A = SparseColMatRef::<'_, I, T>::new(SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_idx), &val);

		let mut new_col_ptr = vec![0usize; m + 1];
		let mut new_row_idx = vec![0usize; nnz];
		let mut new_val = vec![T::ZERO; nnz];

		let AT = utils::transpose(
			&mut new_val,
			&mut new_col_ptr,
			&mut new_row_idx,
			A,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<I>(m))),
		)
		.into_const();
		let A = AT;
		let (m, n) = (n, m);

		let a = A.to_dense();
		let rhs = Mat::<T>::from_fn(m, 2, |_, _| c64::new(gen.gen(), gen.gen()));

		for supernodal_flop_ratio_threshold in [
			SupernodalThreshold::FORCE_SUPERNODAL,
			SupernodalThreshold::FORCE_SIMPLICIAL,
			SupernodalThreshold::AUTO,
		] {
			let symbolic = factorize_symbolic_qr(
				A.symbolic(),
				QrSymbolicParams {
					supernodal_flop_ratio_threshold,
					..Default::default()
				},
			)
			.unwrap();
			let mut indices = vec![0usize; symbolic.len_idx()];
			let mut val = vec![T::ZERO; symbolic.len_val()];
			let qr = symbolic.factorize_numeric_qr::<T>(
				&mut indices,
				&mut val,
				A,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(
					symbolic.factorize_numeric_qr_scratch::<T>(Par::Seq, Default::default()),
				)),
				Default::default(),
			);

			{
				let mut x = rhs.clone();
				qr.solve_in_place_with_conj(
					Conj::No,
					x.as_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(symbolic.solve_in_place_scratch::<T>(2, Par::Seq))),
				);

				let x = x.as_ref().subrows(0, n);
				let linsolve_diff = a.adjoint() * (&a * &x - &rhs);
				assert!(linsolve_diff.norm_max() <= 1e-10);
			}
			{
				let mut x = rhs.clone();
				qr.solve_in_place_with_conj(
					Conj::Yes,
					x.as_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(symbolic.solve_in_place_scratch::<T>(2, Par::Seq))),
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
		type T = c64;

		let mut gen = rand::rngs::StdRng::seed_from_u64(0);

		let a0_col_ptr = vec![0usize; 21];
		let A0 = SparseColMatRef::<'_, I, T>::new(SymbolicSparseColMatRef::new_checked(40, 20, &a0_col_ptr, None, &[]), &[]);

		let a1_val = [c64::new(gen.gen(), gen.gen()), c64::new(gen.gen(), gen.gen())];
		let A1 = SparseColMatRef::<'_, I, T>::new(SymbolicSparseColMatRef::new_checked(40, 5, &[0, 1, 2, 2, 2, 2], None, &[0, 0]), &a1_val);
		let A2 = SparseColMatRef::<'_, I, T>::new(SymbolicSparseColMatRef::new_checked(40, 5, &[0, 1, 2, 2, 2, 2], None, &[4, 4]), &a1_val);

		for A in [A0, A1, A2] {
			for supernodal_flop_ratio_threshold in [
				SupernodalThreshold::AUTO,
				SupernodalThreshold::FORCE_SUPERNODAL,
				SupernodalThreshold::FORCE_SIMPLICIAL,
			] {
				let symbolic = factorize_symbolic_qr(
					A.symbolic(),
					QrSymbolicParams {
						supernodal_flop_ratio_threshold,
						..Default::default()
					},
				)
				.unwrap();
				let mut indices = vec![0usize; symbolic.len_idx()];
				let mut val = vec![T::ZERO; symbolic.len_val()];
				symbolic.factorize_numeric_qr::<T>(
					&mut indices,
					&mut val,
					A,
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(
						symbolic.factorize_numeric_qr_scratch::<T>(Par::Seq, Default::default()),
					)),
					Default::default(),
				);
			}
		}
	}
}
