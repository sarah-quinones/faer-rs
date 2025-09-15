//! computes the $LU$ decomposition of a given sparse matrix. see
//! [`faer::linalg::lu`](crate::linalg::lu) for more info
//!
//! the entry point in this module is [`SymbolicLu`] and [`factorize_symbolic_lu`]
//!
//! # note
//! the functions in this module accept unsorted inputs, and may produce unsorted decomposition
//! factors.

use crate::assert;
use crate::internal_prelude_sp::*;
use crate::sparse::utils;
use linalg::lu::partial_pivoting::factor::PartialPivLuParams;
use linalg_sp::cholesky::simplicial::EliminationTreeRef;
use linalg_sp::{LuError, SupernodalThreshold, SymbolicSupernodalParams, colamd};

#[inline(never)]
fn resize_vec<T: Clone>(v: &mut alloc::vec::Vec<T>, n: usize, exact: bool, reserve_only: bool, value: T) -> Result<(), FaerError> {
	let reserve = if exact {
		alloc::vec::Vec::try_reserve_exact
	} else {
		alloc::vec::Vec::try_reserve
	};
	reserve(v, n.saturating_sub(v.len())).map_err(|_| FaerError::OutOfMemory)?;
	if !reserve_only {
		v.resize(Ord::max(n, v.len()), value);
	}
	Ok(())
}

/// supernodal factorization module
///
/// a supernodal factorization is one that processes the elements of the $LU$ factors of the
/// input matrix by blocks, rather than by single elements. this is more efficient if the lu
/// factors are somewhat dense
pub mod supernodal {
	use super::*;
	use crate::assert;

	/// $LU$ factor structure containing the symbolic structure
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

	/// $LU$ factor structure containing the symbolic and numerical representations
	#[derive(Debug, Clone)]
	pub struct SupernodalLu<I, T> {
		nrows: usize,
		ncols: usize,
		nsupernodes: usize,

		supernode_ptr: alloc::vec::Vec<I>,

		l_col_ptr_for_row_idx: alloc::vec::Vec<I>,
		l_col_ptr_for_val: alloc::vec::Vec<I>,
		l_row_idx: alloc::vec::Vec<I>,
		l_val: alloc::vec::Vec<T>,

		ut_col_ptr_for_row_idx: alloc::vec::Vec<I>,
		ut_col_ptr_for_val: alloc::vec::Vec<I>,
		ut_row_idx: alloc::vec::Vec<I>,
		ut_val: alloc::vec::Vec<T>,
	}

	impl<I: Index, T> Default for SupernodalLu<I, T> {
		fn default() -> Self {
			Self::new()
		}
	}

	impl<I: Index, T> SupernodalLu<I, T> {
		/// creates a new supernodal $LU$ of a $0 \times 0$ matrix
		#[inline]
		pub fn new() -> Self {
			Self {
				nrows: 0,
				ncols: 0,
				nsupernodes: 0,

				supernode_ptr: alloc::vec::Vec::new(),

				l_col_ptr_for_row_idx: alloc::vec::Vec::new(),
				ut_col_ptr_for_row_idx: alloc::vec::Vec::new(),

				l_col_ptr_for_val: alloc::vec::Vec::new(),
				ut_col_ptr_for_val: alloc::vec::Vec::new(),

				l_row_idx: alloc::vec::Vec::new(),
				ut_row_idx: alloc::vec::Vec::new(),

				l_val: alloc::vec::Vec::new(),
				ut_val: alloc::vec::Vec::new(),
			}
		}

		/// returns the number of rows of $A$
		#[inline]
		pub fn nrows(&self) -> usize {
			self.nrows
		}

		/// returns the number of columns of $A$
		#[inline]
		pub fn ncols(&self) -> usize {
			self.ncols
		}

		/// returns the number of supernodes
		#[inline]
		pub fn n_supernodes(&self) -> usize {
			self.nsupernodes
		}

		/// solves the equation $A x = \text{rhs}$ and stores the result in `rhs`, implicitly
		/// conjugating $A$ if needed
		///
		/// # panics
		/// - panics if `self.nrows() != self.ncols()`
		/// - panics if `rhs.nrows() != self.nrows()`
		#[track_caller]
		pub fn solve_in_place_with_conj(
			&self,
			row_perm: PermRef<'_, I>,
			col_perm: PermRef<'_, I>,
			conj_lhs: Conj,
			rhs: MatMut<'_, T>,
			par: Par,
			work: MatMut<'_, T>,
		) where
			T: ComplexField,
		{
			assert!(all(self.nrows() == self.ncols(), self.nrows() == rhs.nrows()));
			let mut X = rhs;
			let mut temp = work;

			crate::perm::permute_rows(temp.rb_mut(), X.rb(), row_perm);
			self.l_solve_in_place_with_conj(conj_lhs, temp.rb_mut(), X.rb_mut(), par);
			self.u_solve_in_place_with_conj(conj_lhs, temp.rb_mut(), X.rb_mut(), par);
			crate::perm::permute_rows(X.rb_mut(), temp.rb(), col_perm.inverse());
		}

		/// solves the equation $A^\top x = \text{rhs}$ and stores the result in `rhs`, implicitly
		/// conjugating $A$ if needed
		///
		/// # panics
		/// - panics if `self.nrows() != self.ncols()`
		/// - panics if `rhs.nrows() != self.nrows()`
		#[track_caller]
		pub fn solve_transpose_in_place_with_conj(
			&self,
			row_perm: PermRef<'_, I>,
			col_perm: PermRef<'_, I>,
			conj_lhs: Conj,
			rhs: MatMut<'_, T>,
			par: Par,
			work: MatMut<'_, T>,
		) where
			T: ComplexField,
		{
			assert!(all(self.nrows() == self.ncols(), self.nrows() == rhs.nrows()));
			let mut X = rhs;
			let mut temp = work;
			crate::perm::permute_rows(temp.rb_mut(), X.rb(), col_perm);
			self.u_solve_transpose_in_place_with_conj(conj_lhs, temp.rb_mut(), X.rb_mut(), par);
			self.l_solve_transpose_in_place_with_conj(conj_lhs, temp.rb_mut(), X.rb_mut(), par);
			crate::perm::permute_rows(X.rb_mut(), temp.rb(), row_perm.inverse());
		}

		#[track_caller]
		#[math]
		pub(crate) fn l_solve_in_place_with_conj(&self, conj_lhs: Conj, rhs: MatMut<'_, T>, mut work: MatMut<'_, T>, par: Par)
		where
			T: ComplexField,
		{
			let lu = self;

			assert!(lu.nrows() == lu.ncols());
			assert!(lu.nrows() == rhs.nrows());

			let mut X = rhs;
			let nrhs = X.ncols();

			let supernode_ptr = &*lu.supernode_ptr;

			for s in 0..lu.nsupernodes {
				let s_begin = supernode_ptr[s].zx();
				let s_end = supernode_ptr[s + 1].zx();
				let s_size = s_end - s_begin;
				let s_row_idx_count = lu.l_col_ptr_for_row_idx[s + 1].zx() - lu.l_col_ptr_for_row_idx[s].zx();

				let L = &lu.l_val[lu.l_col_ptr_for_val[s].zx()..lu.l_col_ptr_for_val[s + 1].zx()];
				let L = MatRef::from_column_major_slice(L, s_row_idx_count, s_size);
				let (L_top, L_bot) = L.split_at_row(s_size);
				linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(
					L_top,
					conj_lhs,
					X.rb_mut().subrows_mut(s_begin, s_size),
					par,
				);
				linalg::matmul::matmul_with_conj(
					work.rb_mut().subrows_mut(0, s_row_idx_count - s_size),
					Accum::Replace,
					L_bot,
					conj_lhs,
					X.rb().subrows(s_begin, s_size),
					Conj::No,
					one::<T>(),
					par,
				);

				for j in 0..nrhs {
					for (idx, &i) in lu.l_row_idx[lu.l_col_ptr_for_row_idx[s].zx()..lu.l_col_ptr_for_row_idx[s + 1].zx()][s_size..]
						.iter()
						.enumerate()
					{
						let i = i.zx();
						X[(i, j)] = X[(i, j)] - work[(idx, j)];
					}
				}
			}
		}

		#[track_caller]
		#[math]
		pub(crate) fn l_solve_transpose_in_place_with_conj(&self, conj_lhs: Conj, rhs: MatMut<'_, T>, mut work: MatMut<'_, T>, par: Par)
		where
			T: ComplexField,
		{
			let lu = self;

			assert!(lu.nrows() == lu.ncols());
			assert!(lu.nrows() == rhs.nrows());

			let mut X = rhs;
			let nrhs = X.ncols();

			let supernode_ptr = &*lu.supernode_ptr;

			for s in (0..lu.nsupernodes).rev() {
				let s_begin = supernode_ptr[s].zx();
				let s_end = supernode_ptr[s + 1].zx();
				let s_size = s_end - s_begin;
				let s_row_idx_count = lu.l_col_ptr_for_row_idx[s + 1].zx() - lu.l_col_ptr_for_row_idx[s].zx();

				let L = &lu.l_val[lu.l_col_ptr_for_val[s].zx()..lu.l_col_ptr_for_val[s + 1].zx()];
				let L = MatRef::from_column_major_slice(L, s_row_idx_count, s_size);

				let (L_top, L_bot) = L.split_at_row(s_size);

				for j in 0..nrhs {
					for (idx, &i) in lu.l_row_idx[lu.l_col_ptr_for_row_idx[s].zx()..lu.l_col_ptr_for_row_idx[s + 1].zx()][s_size..]
						.iter()
						.enumerate()
					{
						let i = i.zx();
						work[(idx, j)] = copy(X[(i, j)]);
					}
				}

				linalg::matmul::matmul_with_conj(
					X.rb_mut().subrows_mut(s_begin, s_size),
					Accum::Add,
					L_bot.transpose(),
					conj_lhs,
					work.rb().subrows(0, s_row_idx_count - s_size),
					Conj::No,
					-one::<T>(),
					par,
				);
				linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(
					L_top.transpose(),
					conj_lhs,
					X.rb_mut().subrows_mut(s_begin, s_size),
					par,
				);
			}
		}

		#[track_caller]
		#[math]
		pub(crate) fn u_solve_in_place_with_conj(&self, conj_lhs: Conj, rhs: MatMut<'_, T>, mut work: MatMut<'_, T>, par: Par)
		where
			T: ComplexField,
		{
			let lu = self;

			assert!(lu.nrows() == lu.ncols());
			assert!(lu.nrows() == rhs.nrows());

			let mut X = rhs;
			let nrhs = X.ncols();

			let supernode_ptr = &*lu.supernode_ptr;

			for s in (0..lu.nsupernodes).rev() {
				let s_begin = supernode_ptr[s].zx();
				let s_end = supernode_ptr[s + 1].zx();
				let s_size = s_end - s_begin;
				let s_row_idx_count = lu.l_col_ptr_for_row_idx[s + 1].zx() - lu.l_col_ptr_for_row_idx[s].zx();
				let s_col_index_count = lu.ut_col_ptr_for_row_idx[s + 1].zx() - lu.ut_col_ptr_for_row_idx[s].zx();

				let L = &lu.l_val[lu.l_col_ptr_for_val[s].zx()..lu.l_col_ptr_for_val[s + 1].zx()];
				let L = MatRef::from_column_major_slice(L, s_row_idx_count, s_size);
				let U = &lu.ut_val[lu.ut_col_ptr_for_val[s].zx()..lu.ut_col_ptr_for_val[s + 1].zx()];
				let U_right = MatRef::from_column_major_slice(U, s_col_index_count, s_size).transpose();

				for j in 0..nrhs {
					for (idx, &i) in lu.ut_row_idx[lu.ut_col_ptr_for_row_idx[s].zx()..lu.ut_col_ptr_for_row_idx[s + 1].zx()]
						.iter()
						.enumerate()
					{
						let i = i.zx();
						work[(idx, j)] = copy(X[(i, j)]);
					}
				}

				let (U_left, _) = L.split_at_row(s_size);
				linalg::matmul::matmul_with_conj(
					X.rb_mut().subrows_mut(s_begin, s_size),
					Accum::Add,
					U_right,
					conj_lhs,
					work.rb().subrows(0, s_col_index_count),
					Conj::No,
					-one::<T>(),
					par,
				);
				linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(U_left, conj_lhs, X.rb_mut().subrows_mut(s_begin, s_size), par);
			}
		}

		#[track_caller]
		#[math]
		pub(crate) fn u_solve_transpose_in_place_with_conj(&self, conj_lhs: Conj, rhs: MatMut<'_, T>, mut work: MatMut<'_, T>, par: Par)
		where
			T: ComplexField,
		{
			let lu = self;

			assert!(lu.nrows() == lu.ncols());
			assert!(lu.nrows() == rhs.nrows());

			let mut X = rhs;
			let nrhs = X.ncols();

			let supernode_ptr = &*lu.supernode_ptr;

			for s in 0..lu.nsupernodes {
				let s_begin = supernode_ptr[s].zx();
				let s_end = supernode_ptr[s + 1].zx();
				let s_size = s_end - s_begin;
				let s_row_idx_count = lu.l_col_ptr_for_row_idx[s + 1].zx() - lu.l_col_ptr_for_row_idx[s].zx();
				let s_col_index_count = lu.ut_col_ptr_for_row_idx[s + 1].zx() - lu.ut_col_ptr_for_row_idx[s].zx();

				let L = &lu.l_val[lu.l_col_ptr_for_val[s].zx()..lu.l_col_ptr_for_val[s + 1].zx()];
				let L = MatRef::from_column_major_slice(L, s_row_idx_count, s_size);
				let U = &lu.ut_val[lu.ut_col_ptr_for_val[s].zx()..lu.ut_col_ptr_for_val[s + 1].zx()];
				let U_right = MatRef::from_column_major_slice(U, s_col_index_count, s_size).transpose();

				let (U_left, _) = L.split_at_row(s_size);
				linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(
					U_left.transpose(),
					conj_lhs,
					X.rb_mut().subrows_mut(s_begin, s_size),
					par,
				);
				linalg::matmul::matmul_with_conj(
					work.rb_mut().subrows_mut(0, s_col_index_count),
					Accum::Replace,
					U_right.transpose(),
					conj_lhs,
					X.rb().subrows(s_begin, s_size),
					Conj::No,
					one::<T>(),
					par,
				);

				for j in 0..nrhs {
					for (idx, &i) in lu.ut_row_idx[lu.ut_col_ptr_for_row_idx[s].zx()..lu.ut_col_ptr_for_row_idx[s + 1].zx()]
						.iter()
						.enumerate()
					{
						let i = i.zx();
						X[(i, j)] = X[(i, j)] - work[(idx, j)];
					}
				}
			}
		}
	}

	/// computes the layout of the workspace required to compute the symbolic
	/// $LU$ factorization of a square matrix with size `n`.
	pub fn factorize_supernodal_symbolic_lu_scratch<I: Index>(nrows: usize, ncols: usize) -> StackReq {
		let _ = nrows;
		linalg_sp::cholesky::supernodal::factorize_supernodal_symbolic_cholesky_scratch::<I>(ncols)
	}

	/// computes the symbolic structure of the $LU$ factors of the matrix $A$
	#[track_caller]
	pub fn factorize_supernodal_symbolic_lu<I: Index>(
		A: SymbolicSparseColMatRef<'_, I>,
		col_perm: Option<PermRef<'_, I>>,
		min_col: &[I],
		etree: EliminationTreeRef<'_, I>,
		col_counts: &[I],
		stack: &mut MemStack,
		params: SymbolicSupernodalParams<'_>,
	) -> Result<SymbolicSupernodalLu<I>, FaerError> {
		let m = A.nrows();
		let n = A.ncols();

		with_dim!(M, m);
		with_dim!(N, n);

		let I = I::truncate;
		let A = A.as_shape(M, N);
		let min_col = Array::from_ref(MaybeIdx::from_slice_ref_checked(bytemuck::cast_slice(min_col), N), M);
		let etree = etree.as_bound(N);

		let L = linalg_sp::cholesky::supernodal::ghost_factorize_supernodal_symbolic(
			A,
			col_perm.map(|perm| perm.as_shape(N)),
			Some(min_col),
			linalg_sp::cholesky::supernodal::CholeskyInput::ATA,
			etree,
			Array::from_ref(col_counts, N),
			stack,
			params,
		)?;
		let n_supernodes = L.n_supernodes();
		let mut super_etree = try_zeroed::<I>(n_supernodes)?;

		let (index_to_super, _) = unsafe { stack.make_raw::<I>(*N) };

		for s in 0..n_supernodes {
			index_to_super[L.supernode_begin[s].zx()..L.supernode_begin[s + 1].zx()].fill(I(s));
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
	}

	struct MatU8 {
		data: alloc::vec::Vec<u8>,
		nrows: usize,
	}
	impl MatU8 {
		fn new() -> Self {
			Self {
				data: alloc::vec::Vec::new(),
				nrows: 0,
			}
		}

		fn with_dims(nrows: usize, ncols: usize) -> Result<Self, FaerError> {
			Ok(Self {
				data: try_collect((0..(nrows * ncols)).map(|_| 1u8))?,
				nrows,
			})
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

	/// computes the layout of the workspace required to perform a numeric $LU$
	/// factorization
	pub fn factorize_supernodal_numeric_lu_scratch<I: Index, T: ComplexField>(
		symbolic: &SymbolicSupernodalLu<I>,
		params: Spec<PartialPivLuParams, T>,
	) -> StackReq {
		let m = StackReq::new::<I>(symbolic.nrows);
		let n = StackReq::new::<I>(symbolic.ncols);
		_ = params;
		StackReq::and(n, m.array(5))
	}

	/// computes the numeric values of the $LU$ factors of the matrix $A$ as well as the row
	/// pivoting permutation, and stores them in `lu` and `row_perm`/`row_perm_inv`
	#[math]
	pub fn factorize_supernodal_numeric_lu<I: Index, T: ComplexField>(
		row_perm: &mut [I],
		row_perm_inv: &mut [I],
		lu: &mut SupernodalLu<I, T>,

		A: SparseColMatRef<'_, I, T>,
		AT: SparseColMatRef<'_, I, T>,
		col_perm: PermRef<'_, I>,
		symbolic: &SymbolicSupernodalLu<I>,

		par: Par,
		stack: &mut MemStack,
		params: Spec<PartialPivLuParams, T>,
	) -> Result<(), LuError> {
		use linalg_sp::cholesky::supernodal::partition_fn;
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

		let (col_global_to_local, stack) = unsafe { stack.make_raw::<I>(n) };
		let (row_global_to_local, stack) = unsafe { stack.make_raw::<I>(m) };
		let (marked, stack) = unsafe { stack.make_raw::<I>(m) };
		let (indices, stack) = unsafe { stack.make_raw::<I>(m) };
		let (transpositions, stack) = unsafe { stack.make_raw::<I>(m) };
		let (d_active_rows, _) = unsafe { stack.make_raw::<I>(m) };

		col_global_to_local.fill(I(NONE));
		row_global_to_local.fill(I(NONE));

		marked.fill(I(0));

		resize_vec(&mut lu.l_col_ptr_for_row_idx, n_supernodes + 1, true, false, I(0))?;
		resize_vec(&mut lu.ut_col_ptr_for_row_idx, n_supernodes + 1, true, false, I(0))?;
		resize_vec(&mut lu.l_col_ptr_for_val, n_supernodes + 1, true, false, I(0))?;
		resize_vec(&mut lu.ut_col_ptr_for_val, n_supernodes + 1, true, false, I(0))?;

		lu.l_col_ptr_for_row_idx[0] = I(0);
		lu.ut_col_ptr_for_row_idx[0] = I(0);
		lu.l_col_ptr_for_val[0] = I(0);
		lu.ut_col_ptr_for_val[0] = I(0);

		for i in 0..m {
			row_perm[i] = I(i);
		}
		for i in 0..m {
			row_perm_inv[i] = I(i);
		}

		let (col_perm, col_perm_inv) = col_perm.arrays();

		let mut contrib_work =
			try_collect((0..n_supernodes).map(|_| (alloc::vec::Vec::<T>::new(), alloc::vec::Vec::<I>::new(), 0usize, MatU8::new())))?;

		let work_to_mat_mut = |v: &mut alloc::vec::Vec<T>, nrows: usize, ncols: usize| unsafe {
			MatMut::from_raw_parts_mut(v.as_mut_ptr(), nrows, ncols, 1, nrows as isize)
		};

		let mut A_leftover = A.compute_nnz();
		for s in 0..n_supernodes {
			let s_begin = supernode_ptr[s].zx();
			let s_end = supernode_ptr[s + 1].zx();
			let s_size = s_end - s_begin;

			let s_postordered = supernode_postorder_inv[s].zx();
			let desc_count = descendant_count[s].zx();
			let mut s_row_idx_count = 0usize;
			let (left_contrib, right_contrib) = contrib_work.split_at_mut(s);

			let s_row_idxices = &mut *indices;
			// add the rows from A[s_end:, s_begin:s_end]
			for j in s_begin..s_end {
				let pj = col_perm[j].zx();
				let row_idx = A.row_idx_of_col_raw(pj);
				for i in row_idx {
					let i = i.zx();
					let pi = row_perm_inv[i].zx();
					if pi < s_begin {
						continue;
					}
					if marked[i] < I(2 * s + 1) {
						s_row_idxices[s_row_idx_count] = I(i);
						s_row_idx_count += 1;
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
				let d_row_idx = &lu.l_row_idx[lu.l_col_ptr_for_row_idx[d].zx()..lu.l_col_ptr_for_row_idx[d + 1].zx()][d_size..];
				let d_col_ind = &lu.ut_row_idx[lu.ut_col_ptr_for_row_idx[d].zx()..lu.ut_col_ptr_for_row_idx[d + 1].zx()];
				let d_col_start = d_col_ind.partition_point(partition_fn(s_begin));

				if d_col_start < d_col_ind.len() && d_col_ind[d_col_start].zx() < s_end {
					for i in d_row_idx.iter() {
						let i = i.zx();
						let pi = row_perm_inv[i].zx();

						if pi < s_begin {
							continue;
						}

						if marked[i] < I(2 * s + 1) {
							s_row_idxices[s_row_idx_count] = I(i);
							s_row_idx_count += 1;
							marked[i] = I(2 * s + 1);
						}
					}
				}
			}

			lu.l_col_ptr_for_row_idx[s + 1] = I_checked(lu.l_col_ptr_for_row_idx[s].zx() + s_row_idx_count)?;
			lu.l_col_ptr_for_val[s + 1] = from_wide_checked(to_wide(lu.l_col_ptr_for_val[s]) + ((s_row_idx_count) as u128 * s_size as u128))?;
			resize_vec(&mut lu.l_row_idx, lu.l_col_ptr_for_row_idx[s + 1].zx(), false, false, I(0))?;
			resize_vec::<T>(&mut lu.l_val, lu.l_col_ptr_for_val[s + 1].zx(), false, false, zero::<T>())?;
			lu.l_row_idx[lu.l_col_ptr_for_row_idx[s].zx()..lu.l_col_ptr_for_row_idx[s + 1].zx()].copy_from_slice(&s_row_idxices[..s_row_idx_count]);
			lu.l_row_idx[lu.l_col_ptr_for_row_idx[s].zx()..lu.l_col_ptr_for_row_idx[s + 1].zx()].sort_unstable();

			let (left_row_idxices, right_row_idxices) = lu.l_row_idx.split_at_mut(lu.l_col_ptr_for_row_idx[s].zx());

			let s_row_idxices = &mut right_row_idxices[0..lu.l_col_ptr_for_row_idx[s + 1].zx() - lu.l_col_ptr_for_row_idx[s].zx()];
			for (idx, i) in s_row_idxices.iter().enumerate() {
				row_global_to_local[i.zx()] = I(idx);
			}
			let s_L = &mut lu.l_val[lu.l_col_ptr_for_val[s].zx()..lu.l_col_ptr_for_val[s + 1].zx()];
			let mut s_L = MatMut::from_column_major_slice_mut(s_L, s_row_idx_count, s_size);
			s_L.fill(zero());

			for j in s_begin..s_end {
				let pj = col_perm[j].zx();
				let row_idx = A.row_idx_of_col(pj);
				let val = A.val_of_col(pj);

				for (i, val) in iter::zip(row_idx, val) {
					let pi = row_perm_inv[i].zx();
					if pi < s_begin {
						continue;
					}
					assert!(A_leftover > 0);
					A_leftover -= 1;
					let ix = row_global_to_local[i].zx();
					let iy = j - s_begin;
					s_L[(ix, iy)] = s_L[(ix, iy)] + *val;
				}
			}

			noinline(LPanel, || {
				for d in &supernode_postorder[s_postordered - desc_count..s_postordered] {
					let d = d.zx();
					if left_contrib[d].0.is_empty() {
						continue;
					}

					let d_begin = supernode_ptr[d].zx();
					let d_end = supernode_ptr[d + 1].zx();
					let d_size = d_end - d_begin;
					let d_row_idx = &left_row_idxices[lu.l_col_ptr_for_row_idx[d].zx()..lu.l_col_ptr_for_row_idx[d + 1].zx()][d_size..];
					let d_col_ind = &lu.ut_row_idx[lu.ut_col_ptr_for_row_idx[d].zx()..lu.ut_col_ptr_for_row_idx[d + 1].zx()];
					let d_col_start = d_col_ind.partition_point(partition_fn(s_begin));

					if d_col_start < d_col_ind.len() && d_col_ind[d_col_start].zx() < s_end {
						let d_col_mid = d_col_start + d_col_ind[d_col_start..].partition_point(partition_fn(s_end));

						let mut d_LU_cols = work_to_mat_mut(&mut left_contrib[d].0, d_row_idx.len(), d_col_ind.len())
							.subcols_mut(d_col_start, d_col_mid - d_col_start);
						let left_contrib = &mut left_contrib[d];
						let d_active = &mut left_contrib.1[d_col_start..];
						let d_active_count = &mut left_contrib.2;
						let d_active_mat = &mut left_contrib.3;

						for (d_j, j) in d_col_ind[d_col_start..d_col_mid].iter().enumerate() {
							if d_active[d_j] > I(0) {
								let mut taken_rows = 0usize;
								let j = j.zx();
								let s_j = j - s_begin;
								for (d_i, i) in d_row_idx.iter().enumerate() {
									let i = i.zx();
									let pi = row_perm_inv[i].zx();
									if pi < s_begin {
										continue;
									}
									let s_i = row_global_to_local[i].zx();

									s_L[(s_i, s_j)] = s_L[(s_i, s_j)] - d_LU_cols[(d_i, d_j)];
									d_LU_cols[(d_i, d_j)] = zero::<T>();
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
							left_contrib.0.clear();
							left_contrib.1 = alloc::vec::Vec::new();
							left_contrib.2 = 0;
							left_contrib.3 = MatU8::new();
						}
					}
				}
			});

			if s_L.nrows() < s_L.ncols() {
				return Err(LuError::SymbolicSingular {
					index: s_begin + s_L.nrows(),
				});
			}
			let transpositions = &mut transpositions[s_begin..s_end];
			crate::linalg::lu::partial_pivoting::factor::lu_in_place_recursion(s_L.rb_mut(), 0, s_size, transpositions, par, params);

			for (idx, t) in transpositions.iter().enumerate() {
				let i_t = s_row_idxices[idx + t.zx()].zx();
				let kk = row_perm_inv[i_t].zx();
				row_perm.swap(s_begin + idx, row_perm_inv[i_t].zx());
				row_perm_inv.swap(row_perm[s_begin + idx].zx(), row_perm[kk].zx());
				s_row_idxices.swap(idx, idx + t.zx());
			}
			for (idx, t) in transpositions.iter().enumerate().rev() {
				row_global_to_local.swap(s_row_idxices[idx].zx(), s_row_idxices[idx + t.zx()].zx());
			}
			for (idx, i) in s_row_idxices.iter().enumerate() {
				assert!(row_global_to_local[i.zx()] == I(idx));
			}

			let s_col_indices = &mut indices[..n];
			let mut s_col_index_count = 0usize;
			for i in s_begin..s_end {
				let pi = row_perm[i].zx();
				for j in AT.row_idx_of_col(pi) {
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

				let d_row_idx = &left_row_idxices[lu.l_col_ptr_for_row_idx[d].zx()..lu.l_col_ptr_for_row_idx[d + 1].zx()][d_size..];
				let d_col_ind = &lu.ut_row_idx[lu.ut_col_ptr_for_row_idx[d].zx()..lu.ut_col_ptr_for_row_idx[d + 1].zx()];

				let contributes_to_u = d_row_idx
					.iter()
					.any(|&i| row_perm_inv[i.zx()].zx() >= s_begin && row_perm_inv[i.zx()].zx() < s_end);

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

			lu.ut_col_ptr_for_row_idx[s + 1] = I_checked(lu.ut_col_ptr_for_row_idx[s].zx() + s_col_index_count)?;
			lu.ut_col_ptr_for_val[s + 1] = from_wide_checked(to_wide(lu.ut_col_ptr_for_val[s]) + (s_col_index_count as u128 * s_size as u128))?;
			resize_vec(&mut lu.ut_row_idx, lu.ut_col_ptr_for_row_idx[s + 1].zx(), false, false, I(0))?;
			resize_vec::<T>(&mut lu.ut_val, lu.ut_col_ptr_for_val[s + 1].zx(), false, false, zero::<T>())?;
			lu.ut_row_idx[lu.ut_col_ptr_for_row_idx[s].zx()..lu.ut_col_ptr_for_row_idx[s + 1].zx()]
				.copy_from_slice(&s_col_indices[..s_col_index_count]);
			lu.ut_row_idx[lu.ut_col_ptr_for_row_idx[s].zx()..lu.ut_col_ptr_for_row_idx[s + 1].zx()].sort_unstable();

			let s_col_indices = &lu.ut_row_idx[lu.ut_col_ptr_for_row_idx[s].zx()..lu.ut_col_ptr_for_row_idx[s + 1].zx()];
			for (idx, j) in s_col_indices.iter().enumerate() {
				col_global_to_local[j.zx()] = I(idx);
			}

			let s_U = &mut lu.ut_val[lu.ut_col_ptr_for_val[s].zx()..lu.ut_col_ptr_for_val[s + 1].zx()];
			let mut s_U = MatMut::from_column_major_slice_mut(s_U, s_col_index_count, s_size).transpose_mut();
			s_U.fill(zero());

			for i in s_begin..s_end {
				let pi = row_perm[i].zx();
				for (j, val) in iter::zip(AT.row_idx_of_col(pi), AT.val_of_col(pi)) {
					let pj = col_perm_inv[j].zx();
					if pj < s_end {
						continue;
					}
					assert!(A_leftover > 0);
					A_leftover -= 1;
					let ix = i - s_begin;
					let iy = col_global_to_local[pj].zx();
					s_U[(ix, iy)] = s_U[(ix, iy)] + *val;
				}
			}

			noinline(UPanel, || {
				for d in &supernode_postorder[s_postordered - desc_count..s_postordered] {
					let d = d.zx();
					if left_contrib[d].0.is_empty() {
						continue;
					}

					let d_begin = supernode_ptr[d].zx();
					let d_end = supernode_ptr[d + 1].zx();
					let d_size = d_end - d_begin;

					let d_row_idx = &left_row_idxices[lu.l_col_ptr_for_row_idx[d].zx()..lu.l_col_ptr_for_row_idx[d + 1].zx()][d_size..];
					let d_col_ind = &lu.ut_row_idx[lu.ut_col_ptr_for_row_idx[d].zx()..lu.ut_col_ptr_for_row_idx[d + 1].zx()];

					let contributes_to_u = d_row_idx
						.iter()
						.any(|&i| row_perm_inv[i.zx()].zx() >= s_begin && row_perm_inv[i.zx()].zx() < s_end);

					if contributes_to_u {
						let d_col_start = d_col_ind.partition_point(partition_fn(s_end));
						let d_LU = work_to_mat_mut(&mut left_contrib[d].0, d_row_idx.len(), d_col_ind.len());
						let mut d_LU = d_LU.get_mut(.., d_col_start..);
						let left_contrib = &mut left_contrib[d];
						let d_active = &mut left_contrib.1[d_col_start..];
						let d_active_count = &mut left_contrib.2;
						let d_active_mat = &mut left_contrib.3;

						for (d_j, j) in d_col_ind[d_col_start..].iter().enumerate() {
							if d_active[d_j] > I(0) {
								let mut taken_rows = 0usize;
								let j = j.zx();
								let s_j = col_global_to_local[j].zx();
								for (d_i, i) in d_row_idx.iter().enumerate() {
									let i = i.zx();
									let pi = row_perm_inv[i].zx();

									if pi >= s_begin && pi < s_end {
										let s_i = row_global_to_local[i].zx();
										s_U[(s_i, s_j)] = s_U[(s_i, s_j)] - (d_LU[(d_i, d_j)]);
										d_LU[(d_i, d_j)] = zero::<T>();
										taken_rows += d_active_mat[(d_i, d_j + d_col_start)] as usize;
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
							left_contrib.0.clear();
							left_contrib.1 = alloc::vec::Vec::new();
							left_contrib.2 = 0;
							left_contrib.3 = MatU8::new();
						}
					}
				}
			});
			linalg::triangular_solve::solve_unit_lower_triangular_in_place(s_L.rb().subrows(0, s_size), s_U.rb_mut(), par);

			if s_row_idx_count > s_size && s_col_index_count > 0 {
				resize_vec::<T>(
					&mut right_contrib[0].0,
					from_wide_checked(to_wide(I(s_row_idx_count - s_size)) * to_wide(I(s_col_index_count)))?.zx(),
					false,
					false,
					zero::<T>(),
				)?;
				right_contrib[0]
					.1
					.try_reserve_exact(s_col_index_count)
					.ok()
					.ok_or(FaerError::OutOfMemory)?;
				right_contrib[0].1.resize(s_col_index_count, I(s_row_idx_count - s_size));
				right_contrib[0].2 = s_col_index_count;
				right_contrib[0].3 = MatU8::with_dims(s_row_idx_count - s_size, s_col_index_count)?;

				let mut s_LU = work_to_mat_mut(&mut right_contrib[0].0, s_row_idx_count - s_size, s_col_index_count);
				linalg::matmul::matmul(s_LU.rb_mut(), Accum::Replace, s_L.rb().get(s_size.., ..), s_U.rb(), one::<T>(), par);

				noinline(Front, || {
					for d in &supernode_postorder[s_postordered - desc_count..s_postordered] {
						let d = d.zx();
						if left_contrib[d].0.is_empty() {
							continue;
						}

						let d_begin = supernode_ptr[d].zx();
						let d_end = supernode_ptr[d + 1].zx();
						let d_size = d_end - d_begin;

						let d_row_idx = &left_row_idxices[lu.l_col_ptr_for_row_idx[d].zx()..lu.l_col_ptr_for_row_idx[d + 1].zx()][d_size..];
						let d_col_ind = &lu.ut_row_idx[lu.ut_col_ptr_for_row_idx[d].zx()..lu.ut_col_ptr_for_row_idx[d + 1].zx()];

						let contributes_to_front = d_row_idx.iter().any(|&i| row_perm_inv[i.zx()].zx() >= s_end);

						if contributes_to_front {
							let d_col_start = d_col_ind.partition_point(partition_fn(s_end));
							let d_LU = work_to_mat_mut(&mut left_contrib[d].0, d_row_idx.len(), d_col_ind.len());
							let mut d_LU = d_LU.get_mut(.., d_col_start..);
							let left_contrib = &mut left_contrib[d];
							let d_active = &mut left_contrib.1[d_col_start..];
							let d_active_count = &mut left_contrib.2;
							let d_active_mat = &mut left_contrib.3;

							let mut d_active_row_count = 0usize;
							let mut first_iter = true;

							for (d_j, j) in d_col_ind[d_col_start..].iter().enumerate() {
								if d_active[d_j] > I(0) {
									if first_iter {
										first_iter = false;
										for (d_i, i) in d_row_idx.iter().enumerate() {
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
										let i = d_row_idx[d_i].zx();
										let d_active_mat = &mut d_active_mat[(d_i, d_j + d_col_start)];
										if *d_active_mat == 0 {
											continue;
										}
										let s_i = row_global_to_local[i].zx() - s_size;

										dst[s_i] = dst[s_i] + (src[d_i]);
										src[d_i] = zero::<T>();

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
								left_contrib.0.clear();
								left_contrib.1 = alloc::vec::Vec::new();
								left_contrib.2 = 0;
								left_contrib.3 = MatU8::new();
							}
						}
					}
				})
			}

			for i in s_row_idxices.iter() {
				row_global_to_local[i.zx()] = I(NONE);
			}
			for j in s_col_indices.iter() {
				col_global_to_local[j.zx()] = I(NONE);
			}
		}
		assert!(A_leftover == 0);

		for idx in &mut lu.l_row_idx[..lu.l_col_ptr_for_row_idx[n_supernodes].zx()] {
			*idx = row_perm_inv[idx.zx()];
		}

		lu.nrows = m;
		lu.ncols = n;
		lu.nsupernodes = n_supernodes;
		lu.supernode_ptr.clone_from(supernode_ptr);

		Ok(())
	}
}

/// simplicial factorization module
///
/// a supernodal factorization is one that processes the elements of the $LU$ factors of the
/// input matrix by single elements, rather than by blocks. this is more efficient if the lu
/// factors are very sparse
pub mod simplicial {
	use super::*;
	use crate::assert;

	/// $LU$ factor structure containing the symbolic and numerical representations
	#[derive(Debug, Clone)]
	pub struct SimplicialLu<I, T> {
		nrows: usize,
		ncols: usize,

		l_col_ptr: alloc::vec::Vec<I>,
		l_row_idx: alloc::vec::Vec<I>,
		l_val: alloc::vec::Vec<T>,

		u_col_ptr: alloc::vec::Vec<I>,
		u_row_idx: alloc::vec::Vec<I>,
		u_val: alloc::vec::Vec<T>,
	}

	impl<I: Index, T> Default for SimplicialLu<I, T> {
		fn default() -> Self {
			Self::new()
		}
	}

	impl<I: Index, T> SimplicialLu<I, T> {
		/// creates a new simplicial $LU$ of a $0 \times 0$ matrix
		#[inline]
		pub fn new() -> Self {
			Self {
				nrows: 0,
				ncols: 0,

				l_col_ptr: alloc::vec::Vec::new(),
				u_col_ptr: alloc::vec::Vec::new(),

				l_row_idx: alloc::vec::Vec::new(),
				u_row_idx: alloc::vec::Vec::new(),

				l_val: alloc::vec::Vec::new(),
				u_val: alloc::vec::Vec::new(),
			}
		}

		/// returns the number of rows of $A$
		#[inline]
		pub fn nrows(&self) -> usize {
			self.nrows
		}

		/// returns the number of columns of $A$
		#[inline]
		pub fn ncols(&self) -> usize {
			self.ncols
		}

		/// returns the $L$ factor of the $LU$ factorization. the row indices may be unsorted
		#[inline]
		pub fn l_factor_unsorted(&self) -> SparseColMatRef<'_, I, T> {
			SparseColMatRef::<'_, I, T>::new(
				unsafe { SymbolicSparseColMatRef::new_unchecked(self.nrows(), self.ncols(), &self.l_col_ptr, None, &self.l_row_idx) },
				&self.l_val,
			)
		}

		/// returns the $U$ factor of the $LU$ factorization. the row indices may be unsorted
		#[inline]
		pub fn u_factor_unsorted(&self) -> SparseColMatRef<'_, I, T> {
			SparseColMatRef::<'_, I, T>::new(
				unsafe { SymbolicSparseColMatRef::new_unchecked(self.ncols(), self.ncols(), &self.u_col_ptr, None, &self.u_row_idx) },
				&self.u_val,
			)
		}

		/// solves the equation $A x = \text{rhs}$ and stores the result in `rhs`, implicitly
		/// conjugating $A$ if needed
		///
		/// # panics
		/// - panics if `self.nrows() != self.ncols()`
		/// - panics if `rhs.nrows() != self.nrows()`
		#[track_caller]
		pub fn solve_in_place_with_conj(
			&self,
			row_perm: PermRef<'_, I>,
			col_perm: PermRef<'_, I>,
			conj_lhs: Conj,
			rhs: MatMut<'_, T>,
			par: Par,
			work: MatMut<'_, T>,
		) where
			T: ComplexField,
		{
			assert!(self.nrows() == self.ncols());
			assert!(self.nrows() == rhs.nrows());
			let mut X = rhs;
			let mut temp = work;

			let l = self.l_factor_unsorted();
			let u = self.u_factor_unsorted();

			crate::perm::permute_rows(temp.rb_mut(), X.rb(), row_perm);
			linalg_sp::triangular_solve::solve_unit_lower_triangular_in_place(l, conj_lhs, temp.rb_mut(), par);
			linalg_sp::triangular_solve::solve_upper_triangular_in_place(u, conj_lhs, temp.rb_mut(), par);
			crate::perm::permute_rows(X.rb_mut(), temp.rb(), col_perm.inverse());
		}

		/// solves the equation $A^\top x = \text{rhs}$ and stores the result in `rhs`,
		/// implicitly conjugating $A$ if needed
		///
		/// # panics
		/// - panics if `self.nrows() != self.ncols()`
		/// - panics if `rhs.nrows() != self.nrows()`
		#[track_caller]
		pub fn solve_transpose_in_place_with_conj(
			&self,
			row_perm: PermRef<'_, I>,
			col_perm: PermRef<'_, I>,
			conj_lhs: Conj,
			rhs: MatMut<'_, T>,
			par: Par,
			work: MatMut<'_, T>,
		) where
			T: ComplexField,
		{
			assert!(all(self.nrows() == self.ncols(), self.nrows() == rhs.nrows()));
			let mut X = rhs;
			let mut temp = work;

			let l = self.l_factor_unsorted();
			let u = self.u_factor_unsorted();

			crate::perm::permute_rows(temp.rb_mut(), X.rb(), col_perm);
			linalg_sp::triangular_solve::solve_upper_triangular_transpose_in_place(u, conj_lhs, temp.rb_mut(), par);
			linalg_sp::triangular_solve::solve_unit_lower_triangular_transpose_in_place(l, conj_lhs, temp.rb_mut(), par);
			crate::perm::permute_rows(X.rb_mut(), temp.rb(), row_perm.inverse());
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

		let li = l.row_idx();

		'dfs_loop: while head_len > 0 {
			let b = xi[head_len - 1].zx().zx();
			let pb = row_perm_inv[b].zx();

			let range = if pb < l.ncols() { l.col_range(pb) } else { 0..0 };
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
				tail_start = depth_first_search(marked, mark, &mut xi[..tail_start], l, row_perm_inv, b, stack);
			}
		}

		tail_start
	}

	#[math]
	fn l_incomplete_solve_sparse<I: Index, T: ComplexField>(
		marked: &mut [I],
		mark: I,

		xi: &mut [I],
		x: &mut [T],
		l: SparseColMatRef<'_, I, T>,
		row_perm_inv: &[I],
		bi: &[I],
		bx: &[T],
		stack: &mut [I],
	) -> usize {
		let tail_start = reach(marked, mark, xi, l.symbolic(), row_perm_inv, bi, stack);

		let xi = &xi[tail_start..];
		for (i, b) in iter::zip(bi, bx) {
			let i = i.zx();
			x[i] = x[i] + *b;
		}

		for i in xi {
			let i = i.zx();
			let pi = row_perm_inv[i].zx();
			if pi >= l.ncols() {
				continue;
			}

			let li = l.row_idx_of_col_raw(pi);
			let lx = l.val_of_col(pi);
			let len = li.len();

			let xi = copy(x[i]);
			for (li, lx) in iter::zip(&li[1..], &lx[1..len]) {
				let li = li.zx();
				x[li] = x[li] - *lx * xi;
			}
		}

		tail_start
	}

	/// computes the layout of the workspace required to perform a numeric $LU$
	/// factorization
	pub fn factorize_simplicial_numeric_lu_scratch<I: Index, T: ComplexField>(nrows: usize, ncols: usize) -> StackReq {
		let idx = StackReq::new::<I>(nrows);
		let val = temp_mat_scratch::<T>(nrows, 1);
		let _ = ncols;
		StackReq::all_of(&[val, idx, idx, idx])
	}

	/// computes the numeric values of the $LU$ factors of the matrix $A$ as well as the row
	/// pivoting permutation, and stores them in `lu` and `row_perm`/`row_perm_inv`
	#[math]
	pub fn factorize_simplicial_numeric_lu<I: Index, T: ComplexField>(
		row_perm: &mut [I],
		row_perm_inv: &mut [I],
		lu: &mut SimplicialLu<I, T>,

		A: SparseColMatRef<'_, I, T>,
		col_perm: PermRef<'_, I>,
		stack: &mut MemStack,
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

		resize_vec(&mut lu.l_col_ptr, n + 1, true, false, I(0))?;
		resize_vec(&mut lu.u_col_ptr, n + 1, true, false, I(0))?;

		let (mut x, stack) = temp_mat_zeroed::<T, _, _>(m, 1, stack);
		let x = x.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap().as_slice_mut();

		let (marked, stack) = unsafe { stack.make_raw::<I>(m) };
		let (xj, stack) = unsafe { stack.make_raw::<I>(m) };
		let (stack, _) = unsafe { stack.make_raw::<I>(m) };

		marked.fill(I(0));
		row_perm_inv.fill(I(n));

		let mut l_pos = 0usize;
		let mut u_pos = 0usize;
		lu.l_col_ptr[0] = I(0);
		lu.u_col_ptr[0] = I(0);
		for j in 0..n {
			let l = SparseColMatRef::<'_, I, T>::new(
				unsafe { SymbolicSparseColMatRef::new_unchecked(m, j, &lu.l_col_ptr[..j + 1], None, &lu.l_row_idx) },
				&lu.l_val,
			);

			let pj = col_perm.arrays().0[j].zx();
			let tail_start = l_incomplete_solve_sparse(
				marked,
				I(j + 1),
				xj,
				x,
				l,
				row_perm_inv,
				A.row_idx_of_col_raw(pj),
				A.val_of_col(pj),
				stack,
			);
			let xj = &xj[tail_start..];

			resize_vec::<T>(&mut lu.l_val, l_pos + xj.len() + 1, false, false, zero::<T>())?;
			resize_vec(&mut lu.l_row_idx, l_pos + xj.len() + 1, false, false, I(0))?;
			resize_vec::<T>(&mut lu.u_val, u_pos + xj.len() + 1, false, false, zero::<T>())?;
			resize_vec(&mut lu.u_row_idx, u_pos + xj.len() + 1, false, false, I(0))?;

			let l_val = &mut *lu.l_val;
			let u_val = &mut *lu.u_val;

			let mut pivot_idx = n;
			let mut pivot_val = -one::<T::Real>();
			for i in xj {
				let i = i.zx();
				let xi = copy(x[i]);
				if row_perm_inv[i] == I(n) {
					let val = abs(xi);
					if matches!(val.partial_cmp(&pivot_val), None | Some(core::cmp::Ordering::Greater)) {
						pivot_idx = i;
						pivot_val = val;
					}
				} else {
					lu.u_row_idx[u_pos] = row_perm_inv[i];
					u_val[u_pos] = xi;
					u_pos += 1;
				}
			}
			if pivot_idx == n {
				return Err(LuError::SymbolicSingular { index: j });
			}

			let x_piv = copy(x[pivot_idx]);
			let x_piv_inv = recip(x_piv);

			row_perm_inv[pivot_idx] = I(j);

			lu.u_row_idx[u_pos] = I(j);
			u_val[u_pos] = x_piv;
			u_pos += 1;
			lu.u_col_ptr[j + 1] = I(u_pos);

			lu.l_row_idx[l_pos] = I(pivot_idx);
			l_val[l_pos] = one::<T>();
			l_pos += 1;

			for i in xj {
				let i = i.zx();
				let xi = copy(x[i]);
				if row_perm_inv[i] == I(n) {
					lu.l_row_idx[l_pos] = I(i);
					l_val[l_pos] = xi * x_piv_inv;
					l_pos += 1;
				}
				x[i] = zero::<T>();
			}
			lu.l_col_ptr[j + 1] = I(l_pos);
		}

		for i in &mut lu.l_row_idx[..l_pos] {
			*i = row_perm_inv[(*i).zx()];
		}

		for (idx, p) in row_perm_inv.iter().enumerate() {
			row_perm[p.zx()] = I(idx);
		}

		lu.nrows = m;
		lu.ncols = n;

		Ok(())
	}
}

/// tuning parameters for the $LU$ symbolic factorization
#[derive(Copy, Clone, Debug, Default)]
pub struct LuSymbolicParams<'a> {
	/// parameters for the fill reducing column permutation
	pub colamd_params: colamd::Control,
	/// threshold for selecting the supernodal factorization
	pub supernodal_flop_ratio_threshold: SupernodalThreshold,
	/// supernodal factorization parameters
	pub supernodal_params: SymbolicSupernodalParams<'a>,
}

/// the inner factorization used for the symbolic $LU$, either simplicial or symbolic
#[derive(Debug, Clone)]
pub enum SymbolicLuRaw<I> {
	/// simplicial structure
	Simplicial {
		/// number of rows of $A$
		nrows: usize,
		/// number of columns of $A$
		ncols: usize,
	},
	/// supernodal structure
	Supernodal(supernodal::SymbolicSupernodalLu<I>),
}

/// the symbolic structure of a sparse $LU$ decomposition
#[derive(Debug, Clone)]
pub struct SymbolicLu<I> {
	raw: SymbolicLuRaw<I>,
	col_perm_fwd: alloc::vec::Vec<I>,
	col_perm_inv: alloc::vec::Vec<I>,
	A_nnz: usize,
}

#[derive(Debug, Clone)]
enum NumericLuRaw<I, T> {
	None,
	Supernodal(supernodal::SupernodalLu<I, T>),
	Simplicial(simplicial::SimplicialLu<I, T>),
}

/// structure that contains the numerical values and row pivoting permutation of the lu
/// decomposition
#[derive(Debug, Clone)]
pub struct NumericLu<I, T> {
	raw: NumericLuRaw<I, T>,
	row_perm_fwd: alloc::vec::Vec<I>,
	row_perm_inv: alloc::vec::Vec<I>,
}

impl<I: Index, T> Default for NumericLu<I, T> {
	fn default() -> Self {
		Self::new()
	}
}

impl<I: Index, T> NumericLu<I, T> {
	/// creates a new $LU$ of a $0\times 0$ matrix
	#[inline]
	pub fn new() -> Self {
		Self {
			raw: NumericLuRaw::None,
			row_perm_fwd: alloc::vec::Vec::new(),
			row_perm_inv: alloc::vec::Vec::new(),
		}
	}
}

/// sparse $LU$ factorization wrapper
#[derive(Debug)]
pub struct LuRef<'a, I: Index, T> {
	symbolic: &'a SymbolicLu<I>,
	numeric: &'a NumericLu<I, T>,
}
impl<I: Index, T> Copy for LuRef<'_, I, T> {}
impl<I: Index, T> Clone for LuRef<'_, I, T> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<'a, I: Index, T> LuRef<'a, I, T> {
	/// creates $LU$ factors from their components
	///
	/// # safety
	/// the numeric part must be the output of [`SymbolicLu::factorize_numeric_lu`], called with a
	/// matrix having the same symbolic structure as the one used to create `symbolic`
	#[inline]
	pub unsafe fn new_unchecked(symbolic: &'a SymbolicLu<I>, numeric: &'a NumericLu<I, T>) -> Self {
		match (&symbolic.raw, &numeric.raw) {
			(SymbolicLuRaw::Simplicial { .. }, NumericLuRaw::Simplicial(_)) => {},
			(SymbolicLuRaw::Supernodal { .. }, NumericLuRaw::Supernodal(_)) => {},
			_ => panic!("incompatible symbolic and numeric variants"),
		}
		Self { symbolic, numeric }
	}

	/// returns the symbolic structure of the $LU$ factorization
	#[inline]
	pub fn symbolic(self) -> &'a SymbolicLu<I> {
		self.symbolic
	}

	/// returns the row pivoting permutation
	#[inline]
	pub fn row_perm(self) -> PermRef<'a, I> {
		unsafe { PermRef::new_unchecked(&self.numeric.row_perm_fwd, &self.numeric.row_perm_inv, self.symbolic.nrows()) }
	}

	/// returns the fill reducing column permutation
	#[inline]
	pub fn col_perm(self) -> PermRef<'a, I> {
		self.symbolic.col_perm()
	}

	/// solves the equation $A x = \text{rhs}$ and stores the result in `rhs`, implicitly
	/// conjugating $A$ if needed
	///
	/// # panics
	/// - panics if `self.nrows() != self.ncols()`
	/// - panics if `rhs.nrows() != self.nrows()`
	#[track_caller]
	pub fn solve_in_place_with_conj(self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
	where
		T: ComplexField,
	{
		let (mut work, _) = unsafe { temp_mat_uninit(rhs.nrows(), rhs.ncols(), stack) };
		let work = work.as_mat_mut();
		match (&self.symbolic.raw, &self.numeric.raw) {
			(SymbolicLuRaw::Simplicial { .. }, NumericLuRaw::Simplicial(numeric)) => {
				numeric.solve_in_place_with_conj(self.row_perm(), self.col_perm(), conj, rhs, par, work)
			},
			(SymbolicLuRaw::Supernodal(_), NumericLuRaw::Supernodal(numeric)) => {
				numeric.solve_in_place_with_conj(self.row_perm(), self.col_perm(), conj, rhs, par, work)
			},
			_ => unreachable!(),
		}
	}

	/// solves the equation $A^\top x = \text{rhs}$ and stores the result in `rhs`,
	/// implicitly conjugating $A$ if needed
	///
	/// # panics
	/// - panics if `self.nrows() != self.ncols()`
	/// - panics if `rhs.nrows() != self.nrows()`
	#[track_caller]
	pub fn solve_transpose_in_place_with_conj(self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
	where
		T: ComplexField,
	{
		let (mut work, _) = unsafe { temp_mat_uninit(rhs.nrows(), rhs.ncols(), stack) };
		let work = work.as_mat_mut();
		match (&self.symbolic.raw, &self.numeric.raw) {
			(SymbolicLuRaw::Simplicial { .. }, NumericLuRaw::Simplicial(numeric)) => {
				numeric.solve_transpose_in_place_with_conj(self.row_perm(), self.col_perm(), conj, rhs, par, work)
			},
			(SymbolicLuRaw::Supernodal(_), NumericLuRaw::Supernodal(numeric)) => {
				numeric.solve_transpose_in_place_with_conj(self.row_perm(), self.col_perm(), conj, rhs, par, work)
			},
			_ => unreachable!(),
		}
	}
}

impl<I: Index> SymbolicLu<I> {
	/// returns the number of rows of $A$
	#[inline]
	pub fn nrows(&self) -> usize {
		match &self.raw {
			SymbolicLuRaw::Simplicial { nrows, .. } => *nrows,
			SymbolicLuRaw::Supernodal(this) => this.nrows,
		}
	}

	/// returns the number of columns of $A$
	#[inline]
	pub fn ncols(&self) -> usize {
		match &self.raw {
			SymbolicLuRaw::Simplicial { ncols, .. } => *ncols,
			SymbolicLuRaw::Supernodal(this) => this.ncols,
		}
	}

	/// returns the fill-reducing column permutation that was computed during symbolic analysis
	#[inline]
	pub fn col_perm(&self) -> PermRef<'_, I> {
		unsafe { PermRef::new_unchecked(&self.col_perm_fwd, &self.col_perm_inv, self.ncols()) }
	}

	/// computes the layout of the workspace required to compute the numerical $LU$
	/// factorization
	pub fn factorize_numeric_lu_scratch<T>(&self, par: Par, params: Spec<PartialPivLuParams, T>) -> StackReq
	where
		T: ComplexField,
	{
		match &self.raw {
			SymbolicLuRaw::Simplicial { nrows, ncols } => simplicial::factorize_simplicial_numeric_lu_scratch::<I, T>(*nrows, *ncols),
			SymbolicLuRaw::Supernodal(symbolic) => {
				let _ = par;
				let m = symbolic.nrows;

				let A_nnz = self.A_nnz;
				let AT_scratch = StackReq::all_of(&[temp_mat_scratch::<T>(A_nnz, 1), StackReq::new::<I>(m + 1), StackReq::new::<I>(A_nnz)]);
				StackReq::and(AT_scratch, supernodal::factorize_supernodal_numeric_lu_scratch::<I, T>(symbolic, params))
			},
		}
	}

	/// computes the layout of the workspace required to solve the equation $A x = b$
	pub fn solve_in_place_scratch<T>(&self, rhs_ncols: usize, par: Par) -> StackReq
	where
		T: ComplexField,
	{
		let _ = par;
		temp_mat_scratch::<T>(self.nrows(), rhs_ncols)
	}

	/// computes the layout of the workspace required to solve the equation
	/// $A^\top x = b$
	pub fn solve_transpose_in_place_scratch<T>(&self, rhs_ncols: usize, par: Par) -> StackReq
	where
		T: ComplexField,
	{
		let _ = par;
		temp_mat_scratch::<T>(self.nrows(), rhs_ncols)
	}

	/// computes a numerical $LU$ factorization of $A$
	#[track_caller]
	pub fn factorize_numeric_lu<'out, T: ComplexField>(
		&'out self,
		numeric: &'out mut NumericLu<I, T>,
		A: SparseColMatRef<'_, I, T>,
		par: Par,
		stack: &mut MemStack,
		params: Spec<PartialPivLuParams, T>,
	) -> Result<LuRef<'out, I, T>, LuError> {
		if matches!(self.raw, SymbolicLuRaw::Simplicial { .. }) && !matches!(numeric.raw, NumericLuRaw::Simplicial(_)) {
			numeric.raw = NumericLuRaw::Simplicial(simplicial::SimplicialLu::new());
		}
		if matches!(self.raw, SymbolicLuRaw::Supernodal(_)) && !matches!(numeric.raw, NumericLuRaw::Supernodal(_)) {
			numeric.raw = NumericLuRaw::Supernodal(supernodal::SupernodalLu::new());
		}

		let nrows = self.nrows();

		numeric
			.row_perm_fwd
			.try_reserve_exact(nrows.saturating_sub(numeric.row_perm_fwd.len()))
			.ok()
			.ok_or(FaerError::OutOfMemory)?;
		numeric
			.row_perm_inv
			.try_reserve_exact(nrows.saturating_sub(numeric.row_perm_inv.len()))
			.ok()
			.ok_or(FaerError::OutOfMemory)?;
		numeric.row_perm_fwd.resize(nrows, I::truncate(0));
		numeric.row_perm_inv.resize(nrows, I::truncate(0));

		match (&self.raw, &mut numeric.raw) {
			(SymbolicLuRaw::Simplicial { nrows, ncols }, NumericLuRaw::Simplicial(lu)) => {
				assert!(all(A.nrows() == *nrows, A.ncols() == *ncols));

				simplicial::factorize_simplicial_numeric_lu(&mut numeric.row_perm_fwd, &mut numeric.row_perm_inv, lu, A, self.col_perm(), stack)?;
			},
			(SymbolicLuRaw::Supernodal(symbolic), NumericLuRaw::Supernodal(lu)) => {
				let m = symbolic.nrows;
				let (new_col_ptr, stack) = unsafe { stack.make_raw::<I>(m + 1) };
				let (new_row_idx, stack) = unsafe { stack.make_raw::<I>(self.A_nnz) };
				let (mut new_values, stack) = unsafe { temp_mat_uninit::<T, _, _>(self.A_nnz, 1, stack) };
				let new_values = new_values.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap().as_slice_mut();
				let AT = utils::transpose(new_values, new_col_ptr, new_row_idx, A, stack).into_const();

				supernodal::factorize_supernodal_numeric_lu(
					&mut numeric.row_perm_fwd,
					&mut numeric.row_perm_inv,
					lu,
					A,
					AT,
					self.col_perm(),
					symbolic,
					par,
					stack,
					params,
				)?;
			},
			_ => unreachable!(),
		}

		Ok(unsafe { LuRef::new_unchecked(self, numeric) })
	}
}

/// computes the symbolic $LU$ factorization of the matrix $A$, or returns an error if the
/// operation could not be completed
#[track_caller]
pub fn factorize_symbolic_lu<I: Index>(A: SymbolicSparseColMatRef<'_, I>, params: LuSymbolicParams<'_>) -> Result<SymbolicLu<I>, FaerError> {
	assert!(A.nrows() == A.ncols());
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
			linalg_sp::colamd::order_scratch::<I>(m, n, A_nnz),
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
					supernodal::factorize_supernodal_symbolic_lu_scratch::<I>(m, n),
				]),
			]),
		)
	};

	let mut mem = dyn_stack::MemBuffer::try_new(req).ok().ok_or(FaerError::OutOfMemory)?;
	let stack = MemStack::new(&mut mem);

	let mut col_perm_fwd = try_zeroed::<I>(n)?;
	let mut col_perm_inv = try_zeroed::<I>(n)?;
	let mut min_row = try_zeroed::<I>(m)?;

	linalg_sp::colamd::order(&mut col_perm_fwd, &mut col_perm_inv, A.as_dyn(), params.colamd_params, stack)?;

	let col_perm = PermRef::new_checked(&col_perm_fwd, &col_perm_inv, n).as_shape(N);

	let (new_col_ptr, stack) = unsafe { stack.make_raw::<I>(m + 1) };
	let (new_row_idx, stack) = unsafe { stack.make_raw::<I>(A_nnz) };
	let AT = utils::adjoint(
		Symbolic::materialize(new_row_idx.len()),
		new_col_ptr,
		new_row_idx,
		SparseColMatRef::new(A, Symbolic::materialize(A.row_idx().len())),
		stack,
	)
	.symbolic();

	let (etree, stack) = unsafe { stack.make_raw::<I::Signed>(n) };
	let (post, stack) = unsafe { stack.make_raw::<I>(n) };
	let (col_counts, stack) = unsafe { stack.make_raw::<I>(n) };
	let (h_col_counts, stack) = unsafe { stack.make_raw::<I>(n) };

	linalg_sp::qr::ghost_col_etree(A, Some(col_perm), Array::from_mut(etree, N), stack);
	let etree_ = Array::from_ref(MaybeIdx::<'_, I>::from_slice_ref_checked(etree, N), N);
	linalg_sp::cholesky::ghost_postorder(Array::from_mut(post, N), etree_, stack);

	linalg_sp::qr::ghost_column_counts_aat(
		Array::from_mut(col_counts, N),
		Array::from_mut(bytemuck::cast_slice_mut(&mut min_row), M),
		AT,
		Some(col_perm),
		etree_,
		Array::from_ref(Idx::from_slice_ref_checked(post, N), N),
		stack,
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

		if flops / nnz > threshold.0 * linalg_sp::LU_SUPERNODAL_RATIO_FACTOR {
			threshold = SupernodalThreshold::FORCE_SUPERNODAL;
		} else {
			threshold = SupernodalThreshold::FORCE_SIMPLICIAL;
		}
	}

	if threshold == SupernodalThreshold::FORCE_SUPERNODAL {
		let symbolic = supernodal::factorize_supernodal_symbolic_lu::<I>(
			A.as_dyn(),
			Some(col_perm.as_shape(n)),
			&min_col,
			EliminationTreeRef::<'_, I> { inner: etree },
			col_counts,
			stack,
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
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use dyn_stack::MemBuffer;
	use linalg_sp::cholesky::tests::load_mtx;
	use matrix_market_rs::MtxData;
	use std::path::PathBuf;

	#[test]
	fn test_numeric_lu_multifrontal() {
		type T = c64;

		let (m, n, col_ptr, row_idx, val) =
			load_mtx::<usize>(MtxData::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_lu/YAO.mtx")).unwrap());

		let mut rng = StdRng::seed_from_u64(0);
		let mut gen = || T::new(rng.gen::<f64>(), rng.gen::<f64>());

		let val = val.iter().map(|_| gen()).collect::<alloc::vec::Vec<_>>();
		let A = SparseColMatRef::<'_, usize, T>::new(SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_idx), &val);
		let mut row_perm = vec![0usize; n];
		let mut row_perm_inv = vec![0usize; n];
		let mut col_perm = vec![0usize; n];
		let mut col_perm_inv = vec![0usize; n];
		for i in 0..n {
			col_perm[i] = i;
			col_perm_inv[i] = i;
		}
		let col_perm = PermRef::<'_, usize>::new_checked(&col_perm, &col_perm_inv, n);

		let mut etree = vec![0usize; n];
		let mut min_col = vec![0usize; m];
		let mut col_counts = vec![0usize; n];

		let nnz = A.compute_nnz();
		let mut new_col_ptr = vec![0usize; m + 1];
		let mut new_row_idx = vec![0usize; nnz];
		let mut new_values = vec![zero::<T>(); nnz];
		let AT = utils::transpose(
			&mut *new_values,
			&mut new_col_ptr,
			&mut new_row_idx,
			A,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<usize>(m))),
		)
		.into_const();

		let etree = {
			let mut post = vec![0usize; n];

			let etree = linalg_sp::qr::col_etree(
				A.symbolic(),
				Some(col_perm),
				&mut etree,
				MemStack::new(&mut MemBuffer::new(StackReq::new::<usize>(m + n))),
			);
			linalg_sp::qr::postorder(&mut post, etree, MemStack::new(&mut MemBuffer::new(StackReq::new::<usize>(3 * n))));
			linalg_sp::qr::column_counts_ata(
				&mut col_counts,
				&mut min_col,
				AT.symbolic(),
				Some(col_perm),
				etree,
				&post,
				MemStack::new(&mut MemBuffer::new(StackReq::new::<usize>(5 * n + m))),
			);
			etree
		};

		let symbolic = linalg_sp::lu::supernodal::factorize_supernodal_symbolic_lu::<usize>(
			A.symbolic(),
			Some(col_perm),
			&min_col,
			etree,
			&col_counts,
			MemStack::new(&mut MemBuffer::new(super::supernodal::factorize_supernodal_symbolic_lu_scratch::<usize>(
				m, n,
			))),
			linalg_sp::SymbolicSupernodalParams {
				relax: Some(&[(4, 1.0), (16, 0.8), (48, 0.1), (usize::MAX, 0.05)]),
			},
		)
		.unwrap();

		let mut lu = supernodal::SupernodalLu::<usize, T>::new();
		supernodal::factorize_supernodal_numeric_lu(
			&mut row_perm,
			&mut row_perm_inv,
			&mut lu,
			A,
			AT,
			col_perm,
			&symbolic,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(supernodal::factorize_supernodal_numeric_lu_scratch::<usize, T>(
				&symbolic,
				Default::default(),
			))),
			Default::default(),
		)
		.unwrap();

		let k = 2;
		let rhs = Mat::from_fn(n, k, |_, _| gen());

		let mut work = rhs.clone();
		let A_dense = A.to_dense();
		let row_perm = PermRef::<'_, _>::new_checked(&row_perm, &row_perm_inv, m);

		{
			let mut x = rhs.clone();

			lu.solve_in_place_with_conj(row_perm, col_perm, Conj::No, x.as_mut(), Par::Seq, work.as_mut());
			assert!((&A_dense * &x - &rhs).norm_max() < 1e-10);
		}
		{
			let mut x = rhs.clone();

			lu.solve_in_place_with_conj(row_perm, col_perm, Conj::Yes, x.as_mut(), Par::Seq, work.as_mut());
			assert!((A_dense.conjugate() * &x - &rhs).norm_max() < 1e-10);
		}
		{
			let mut x = rhs.clone();

			lu.solve_transpose_in_place_with_conj(row_perm, col_perm, Conj::No, x.as_mut(), Par::Seq, work.as_mut());
			assert!((A_dense.transpose() * &x - &rhs).norm_max() < 1e-10);
		}
		{
			let mut x = rhs.clone();

			lu.solve_transpose_in_place_with_conj(row_perm, col_perm, Conj::Yes, x.as_mut(), Par::Seq, work.as_mut());
			assert!((A_dense.adjoint() * &x - &rhs).norm_max() < 1e-10);
		}
	}

	#[test]
	fn test_numeric_lu_simplicial() {
		type T = c64;

		let (m, n, col_ptr, row_idx, val) =
			load_mtx::<usize>(MtxData::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_lu/YAO.mtx")).unwrap());

		let mut rng = StdRng::seed_from_u64(0);
		let mut gen = || T::new(rng.gen::<f64>(), rng.gen::<f64>());

		let val = val.iter().map(|_| gen()).collect::<alloc::vec::Vec<_>>();
		let A = SparseColMatRef::<'_, usize, T>::new(SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_idx), &val);
		let mut row_perm = vec![0usize; n];
		let mut row_perm_inv = vec![0usize; n];
		let mut col_perm = vec![0usize; n];
		let mut col_perm_inv = vec![0usize; n];
		for i in 0..n {
			col_perm[i] = i;
			col_perm_inv[i] = i;
		}
		let col_perm = PermRef::<'_, usize>::new_checked(&col_perm, &col_perm_inv, n);

		let mut lu = simplicial::SimplicialLu::<usize, T>::new();
		simplicial::factorize_simplicial_numeric_lu(
			&mut row_perm,
			&mut row_perm_inv,
			&mut lu,
			A,
			col_perm,
			MemStack::new(&mut MemBuffer::new(simplicial::factorize_simplicial_numeric_lu_scratch::<usize, T>(m, n))),
		)
		.unwrap();

		let k = 1;
		let rhs = Mat::from_fn(n, k, |_, _| gen());

		let mut work = rhs.clone();
		let A_dense = A.to_dense();
		let row_perm = PermRef::<'_, _>::new_checked(&row_perm, &row_perm_inv, m);

		{
			let mut x = rhs.clone();

			lu.solve_in_place_with_conj(row_perm, col_perm, Conj::No, x.as_mut(), Par::Seq, work.as_mut());
			assert!((&A_dense * &x - &rhs).norm_max() < 1e-10);
		}
		{
			let mut x = rhs.clone();

			lu.solve_in_place_with_conj(row_perm, col_perm, Conj::Yes, x.as_mut(), Par::Seq, work.as_mut());
			assert!((A_dense.conjugate() * &x - &rhs).norm_max() < 1e-10);
		}

		{
			let mut x = rhs.clone();

			lu.solve_transpose_in_place_with_conj(row_perm, col_perm, Conj::No, x.as_mut(), Par::Seq, work.as_mut());
			assert!((A_dense.transpose() * &x - &rhs).norm_max() < 1e-10);
		}
		{
			let mut x = rhs.clone();

			lu.solve_transpose_in_place_with_conj(row_perm, col_perm, Conj::Yes, x.as_mut(), Par::Seq, work.as_mut());
			assert!((A_dense.adjoint() * &x - &rhs).norm_max() < 1e-10);
		}
	}

	#[test]
	fn test_solver_lu_simplicial() {
		type T = c64;

		let (m, n, col_ptr, row_idx, val) =
			load_mtx::<usize>(MtxData::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_lu/YAO.mtx")).unwrap());

		let mut rng = StdRng::seed_from_u64(0);
		let mut gen = || T::new(rng.gen::<f64>(), rng.gen::<f64>());

		let val = val.iter().map(|_| gen()).collect::<alloc::vec::Vec<_>>();
		let A = SparseColMatRef::<'_, usize, T>::new(SymbolicSparseColMatRef::new_checked(m, n, &col_ptr, None, &row_idx), &val);

		let rhs = Mat::<T>::from_fn(m, 6, |_, _| gen());

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
			let mut numeric = NumericLu::<usize, T>::new();
			let lu = symbolic
				.factorize_numeric_lu(
					&mut numeric,
					A,
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(
						symbolic.factorize_numeric_lu_scratch::<T>(Par::Seq, Default::default()),
					)),
					Default::default(),
				)
				.unwrap();

			{
				let mut x = rhs.clone();
				lu.solve_in_place_with_conj(
					crate::Conj::No,
					x.as_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(symbolic.solve_in_place_scratch::<T>(rhs.ncols(), Par::Seq))),
				);

				let linsolve_diff = A * &x - &rhs;
				assert!(linsolve_diff.norm_max() <= 1e-10);
			}
			{
				let mut x = rhs.clone();
				lu.solve_in_place_with_conj(
					crate::Conj::Yes,
					x.as_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(symbolic.solve_in_place_scratch::<T>(rhs.ncols(), Par::Seq))),
				);

				let linsolve_diff = A.conjugate() * &x - &rhs;
				assert!(linsolve_diff.norm_max() <= 1e-10);
			}

			{
				let mut x = rhs.clone();
				lu.solve_transpose_in_place_with_conj(
					crate::Conj::No,
					x.as_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(symbolic.solve_transpose_in_place_scratch::<T>(rhs.ncols(), Par::Seq))),
				);

				let linsolve_diff = A.transpose() * &x - &rhs;
				assert!(linsolve_diff.norm_max() <= 1e-10);
			}
			{
				let mut x = rhs.clone();
				lu.solve_transpose_in_place_with_conj(
					crate::Conj::Yes,
					x.as_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(symbolic.solve_transpose_in_place_scratch::<T>(rhs.ncols(), Par::Seq))),
				);

				let linsolve_diff = A.adjoint() * &x - &rhs;
				assert!(linsolve_diff.norm_max() <= 1e-10);
			}
		}
	}
}
