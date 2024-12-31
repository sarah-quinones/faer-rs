use super::super::utils::*;
use super::ghost;
use crate::assert;
use crate::internal_prelude_sp::*;
use linalg::cholesky::bunch_kaufman::factor::BunchKaufmanRegularization;
use linalg::cholesky::ldlt::factor::{LdltError, LdltInfo, LdltRegularization};
use linalg::cholesky::llt::factor::{LltError, LltInfo, LltRegularization};
use linalg_sp::{SupernodalThreshold, SymbolicSupernodalParams, amd, triangular_solve};

#[derive(Copy, Clone, Debug, Default)]
pub enum SymmetricOrdering<'a, I: Index> {
	/// Approximate minimum degree ordering. Default option.
	#[default]
	Amd,
	/// No reordering.
	Identity,
	/// Custom reordering.
	Custom(PermRef<'a, I>),
}

/// Simplicial factorization module.
///
/// A simplicial factorization is one that processes the elements of the Cholesky factor of the
/// input matrix single elements, rather than by blocks. This is more efficient if the Cholesky
/// factor is very sparse.
pub mod simplicial {
	use super::*;
	use crate::assert;

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
		pub fn len(&self) -> usize {
			self.inner.len()
		}

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
		pub(crate) fn as_bound<'n>(self, N: ghost::Dim<'n>) -> &'a Array<'n, MaybeIdx<'n, I>> {
			assert!(self.inner.len() == *N);
			unsafe { Array::from_ref(MaybeIdx::from_slice_ref_unchecked(self.inner), N) }
		}
	}

	/// Computes the size and alignment of the workspace required to compute the elimination tree
	/// and column counts of a matrix of size `n` with `nnz` non-zero entries.
	pub fn prefactorize_symbolic_cholesky_scratch<I: Index>(n: usize, nnz: usize) -> StackReq {
		_ = nnz;
		StackReq::new::<I>(n)
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
		stack: &mut MemStack,
	) -> EliminationTreeRef<'out, I> {
		let n = A.nrows();
		assert!(A.nrows() == A.ncols());
		assert!(etree.len() == n);
		assert!(col_counts.len() == n);

		with_dim!(N, n);
		ghost_prefactorize_symbolic_cholesky(Array::from_mut(etree, N), Array::from_mut(col_counts, N), A.as_shape(N, N), stack);

		simplicial::EliminationTreeRef { inner: etree }
	}

	// workspace: I×(n)
	fn ghost_prefactorize_symbolic_cholesky<'n, 'out, I: Index>(
		etree: &'out mut Array<'n, I::Signed>,
		col_counts: &mut Array<'n, I>,
		A: SymbolicSparseColMatRef<'_, I, Dim<'n>, Dim<'n>>,
		stack: &mut MemStack,
	) -> &'out mut Array<'n, MaybeIdx<'n, I>> {
		let N = A.ncols();
		let (visited, _) = unsafe { stack.make_raw::<I>(*N) };
		let etree = Array::from_mut(ghost::fill_none::<I>(etree.as_mut(), N), N);
		let visited = Array::from_mut(visited, N);

		for j in N.indices() {
			let j_ = j.truncate::<I>();
			visited[j] = *j_;
			col_counts[j] = I::truncate(1);

			for mut i in A.row_idx_of_col(j) {
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

	fn ereach<'n, 'a, I: Index>(
		stack: &'a mut Array<'n, I>,
		A: SymbolicSparseColMatRef<'_, I, Dim<'n>, Dim<'n>>,
		etree: &Array<'n, MaybeIdx<'n, I>>,
		k: Idx<'n, usize>,
		visited: &mut Array<'n, I::Signed>,
	) -> &'a [Idx<'n, I>] {
		let N = A.ncols();

		// invariant: stack[top..] elements are less than or equal to k
		let mut top = *N;
		let k_: I = *k.truncate();
		visited[k] = k_.to_signed();
		for mut i in A.row_idx_of_col(k) {
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
				i = N.check(etree[i].unbound().zx());
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
	pub fn factorize_simplicial_symbolic_scratch<I: Index>(n: usize) -> StackReq {
		let n_scratch = StackReq::new::<I>(n);
		StackReq::all_of(&[n_scratch, n_scratch, n_scratch])
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
		stack: &mut MemStack,
	) -> Result<SymbolicSimplicialCholesky<I>, FaerError> {
		let n = A.nrows();
		assert!(A.nrows() == A.ncols());
		assert!(etree.inner.len() == n);
		assert!(col_counts.len() == n);

		with_dim!(N, n);
		ghost_factorize_simplicial_symbolic_cholesky(A.as_shape(N, N), etree.as_bound(N), Array::from_ref(col_counts, N), stack)
	}

	pub(crate) fn ghost_factorize_simplicial_symbolic_cholesky<'n, I: Index>(
		A: SymbolicSparseColMatRef<'_, I, Dim<'n>, Dim<'n>>,
		etree: &Array<'n, MaybeIdx<'n, I>>,
		col_counts: &Array<'n, I>,
		stack: &mut MemStack,
	) -> Result<SymbolicSimplicialCholesky<I>, FaerError> {
		let N = A.ncols();
		let n = *N;

		let mut L_col_ptr = try_zeroed::<I>(n + 1)?;
		for (&count, [p, p_next]) in iter::zip(col_counts.as_ref(), windows2(Cell::as_slice_of_cells(Cell::from_mut(&mut L_col_ptr)))) {
			p_next.set(p.get() + count);
		}
		let l_nnz = L_col_ptr[n].zx();
		let mut L_row_ind = try_zeroed::<I>(l_nnz)?;

		with_dim!(L_NNZ, l_nnz);
		let (current_row_index, stack) = unsafe { stack.make_raw::<I>(n) };
		let (ereach_stack, stack) = unsafe { stack.make_raw::<I>(n) };
		let (visited, _) = unsafe { stack.make_raw::<I::Signed>(n) };

		let ereach_stack = Array::from_mut(ereach_stack, N);
		let visited = Array::from_mut(visited, N);

		visited.as_mut().fill(I::Signed::truncate(NONE));
		let L_row_idx = Array::from_mut(&mut L_row_ind, L_NNZ);
		let L_col_ptr_start = Array::from_ref(Idx::from_slice_ref_checked(&L_col_ptr[..n], L_NNZ), N);
		let current_row_index = Array::from_mut(ghost::copy_slice(current_row_index, L_col_ptr_start.as_ref()), N);

		for k in N.indices() {
			let reach = ereach(ereach_stack, A, etree, k, visited);
			for &j in reach {
				let j = j.zx();
				let cj = &mut current_row_index[j];
				let row_idx = L_NNZ.check(*cj.zx() + 1);
				*cj = row_idx.truncate();
				L_row_idx[row_idx] = *k.truncate();
			}
			let k_start = L_col_ptr_start[k].zx();
			L_row_idx[k_start] = *k.truncate();
		}

		let etree = try_collect(
			bytemuck::cast_slice::<I::Signed, I>(MaybeIdx::as_slice_ref(etree.as_ref()))
				.iter()
				.copied(),
		)?;

		let _ = SymbolicSparseColMatRef::new_unsorted_checked(n, n, &L_col_ptr, None, &L_row_ind);

		Ok(SymbolicSimplicialCholesky {
			dimension: n,
			col_ptr: L_col_ptr,
			row_idx: L_row_ind,
			etree,
		})
	}

	#[derive(Copy, Clone, Debug, PartialEq, Eq)]
	enum FactorizationKind {
		Llt,
		Ldlt,
	}

	#[math]
	fn factorize_simplicial_numeric_with_row_idx<I: Index, T: ComplexField>(
		L_values: &mut [T],
		L_row_idx: &mut [I],
		L_col_ptr: &[I],
		kind: FactorizationKind,

		etree: EliminationTreeRef<'_, I>,
		A: SparseColMatRef<'_, I, T>,
		regularization: LdltRegularization<'_, T>,

		stack: &mut MemStack,
	) -> Result<LltInfo, LltError> {
		let n = A.ncols();

		assert!(L_values.len() == L_row_idx.len());
		assert!(L_col_ptr.len() == n + 1);
		assert!(etree.len() == n);
		let l_nnz = L_col_ptr[n].zx();

		with_dim!(N, n);
		with_dim!(L_NNZ, l_nnz);

		let etree = etree.as_bound(N);
		let A = A.as_shape(N, N);

		let eps = abs(regularization.dynamic_regularization_epsilon);
		let delta = abs(regularization.dynamic_regularization_delta);
		let has_delta = delta > zero::<T::Real>();
		let mut dynamic_regularization_count = 0usize;

		let (mut x, stack) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
		let mut x = x.as_mat_mut().col_mut(0).as_row_shape_mut(N);

		let (current_row_index, stack) = unsafe { stack.make_raw::<I>(n) };
		let (ereach_stack, stack) = unsafe { stack.make_raw::<I>(n) };
		let (visited, _) = unsafe { stack.make_raw::<I::Signed>(n) };

		let ereach_stack = Array::from_mut(ereach_stack, N);
		let visited = Array::from_mut(visited, N);

		visited.as_mut().fill(I::Signed::truncate(NONE));

		let L_values = Array::from_mut(L_values, L_NNZ);
		let L_row_idx = Array::from_mut(L_row_idx, L_NNZ);

		let L_col_ptr_start = Array::from_ref(Idx::from_slice_ref_checked(&L_col_ptr[..n], L_NNZ), N);

		let current_row_index = Array::from_mut(ghost::copy_slice(current_row_index, L_col_ptr_start.as_ref()), N);

		for k in N.indices() {
			let reach = ereach(ereach_stack, A.symbolic(), etree, k, visited);

			for (i, aik) in iter::zip(A.row_idx_of_col(k), A.val_of_col(k)) {
				x[i] = x[i] + conj(aik);
			}

			let mut d = real(x[k]);
			x[k] = zero::<T>();

			for &j in reach {
				let j = j.zx();

				let j_start = L_col_ptr_start[j].zx();
				let cj = &mut current_row_index[j];
				let row_idx = L_NNZ.check(*cj.zx() + 1);
				*cj = row_idx.truncate();

				let mut xj = copy(x[j]);
				x[j] = zero::<T>();

				let dj = real(L_values[j_start]);
				let lkj = mul_real(xj, recip(dj));
				if matches!(kind, FactorizationKind::Llt) {
					xj = copy(lkj);
				}

				let range = j_start.next()..row_idx.into();
				for (i, lij) in iter::zip(&L_row_idx[range.clone()], &L_values[range]) {
					let i = N.check(i.zx());
					x[i] = x[i] - conj(*lij) * xj;
				}

				d = d - real(lkj * conj(xj));

				L_row_idx[row_idx] = *k.truncate();
				L_values[row_idx] = lkj;
			}

			let k_start = L_col_ptr_start[k].zx();
			L_row_idx[k_start] = *k.truncate();

			if has_delta {
				match kind {
					FactorizationKind::Llt => {
						if d <= eps {
							d = copy(delta);
							dynamic_regularization_count += 1;
						}
					},
					FactorizationKind::Ldlt => {
						if let Some(signs) = regularization.dynamic_regularization_signs {
							if signs[*k] > 0 && d <= eps {
								d = copy(delta);
								dynamic_regularization_count += 1;
							} else if signs[*k] < 0 && d >= -eps {
								d = -delta;
								dynamic_regularization_count += 1;
							}
						} else if abs(d) <= eps {
							if d < zero::<T::Real>() {
								d = -delta;
								dynamic_regularization_count += 1;
							} else {
								d = copy(delta);
								dynamic_regularization_count += 1;
							}
						}
					},
				}
			}

			match kind {
				FactorizationKind::Llt => {
					if !(d > zero::<T::Real>()) {
						return Err(LltError::NonPositivePivot { index: *k + 1 });
					}
					L_values[k_start] = from_real(sqrt(d));
				},
				FactorizationKind::Ldlt => {
					if d == zero::<T::Real>() || !is_finite(d) {
						return Err(LltError::NonPositivePivot { index: *k + 1 });
					}
					L_values[k_start] = from_real(d);
				},
			}
		}
		Ok(LltInfo {
			dynamic_regularization_count,
		})
	}

	#[math]
	fn factorize_simplicial_numeric<I: Index, T: ComplexField>(
		L_values: &mut [T],
		kind: FactorizationKind,
		A: SparseColMatRef<'_, I, T>,
		regularization: LdltRegularization<'_, T>,
		symbolic: &SymbolicSimplicialCholesky<I>,
		stack: &mut MemStack,
	) -> Result<LltInfo, LltError> {
		let n = A.ncols();
		let L_row_idx = &*symbolic.row_idx;
		let L_col_ptr = &*symbolic.col_ptr;
		let etree = &*symbolic.etree;

		assert!(L_values.rb().len() == L_row_idx.len());
		assert!(L_col_ptr.len() == n + 1);
		let l_nnz = L_col_ptr[n].zx();

		with_dim!(N, n);
		with_dim!(L_NNZ, l_nnz);

		let etree = Array::from_ref(MaybeIdx::from_slice_ref_checked(bytemuck::cast_slice::<I, I::Signed>(etree), N), N);
		let A = A.as_shape(N, N);

		let eps = abs(regularization.dynamic_regularization_epsilon);
		let delta = abs(regularization.dynamic_regularization_delta);
		let has_delta = delta > zero::<T::Real>();
		let mut dynamic_regularization_count = 0usize;

		let (mut x, stack) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
		let mut x = x.as_mat_mut().col_mut(0).as_row_shape_mut(N);
		let (current_row_index, stack) = unsafe { stack.make_raw::<I>(n) };
		let (ereach_stack, stack) = unsafe { stack.make_raw::<I>(n) };
		let (visited, _) = unsafe { stack.make_raw::<I::Signed>(n) };

		let ereach_stack = Array::from_mut(ereach_stack, N);
		let visited = Array::from_mut(visited, N);

		visited.as_mut().fill(I::Signed::truncate(NONE));

		let L_values = Array::from_mut(L_values, L_NNZ);
		let L_row_idx = Array::from_ref(L_row_idx, L_NNZ);

		let L_col_ptr_start = Array::from_ref(Idx::from_slice_ref_checked(&L_col_ptr[..n], L_NNZ), N);

		let current_row_index = Array::from_mut(ghost::copy_slice(current_row_index, L_col_ptr_start.as_ref()), N);

		for k in N.indices() {
			let reach = ereach(ereach_stack, A.symbolic(), etree, k, visited);

			for (i, aik) in iter::zip(A.row_idx_of_col(k), A.val_of_col(k)) {
				x[i] = x[i] + conj(*aik);
			}

			let mut d = real(x[k]);
			x[k] = zero::<T>();

			for &j in reach {
				let j = j.zx();

				let j_start = L_col_ptr_start[j].zx();
				let cj = &mut current_row_index[j];
				let row_idx = L_NNZ.check(*cj.zx() + 1);
				*cj = row_idx.truncate();

				let mut xj = copy(x[j]);
				x[j] = zero::<T>();

				let dj = real(L_values[j_start]);
				let lkj = mul_real(xj, recip(dj));
				if matches!(kind, FactorizationKind::Llt) {
					xj = copy(lkj);
				}

				let range = j_start.next()..row_idx.into();
				for (i, lij) in iter::zip(&L_row_idx[range.clone()], &L_values[range]) {
					let i = i.zx();
					if i >= *N {
						panic!();
					}
					let i = unsafe { Idx::new_unchecked(i, N) };
					x[i] = x[i] - conj(*lij) * xj;
				}

				d = d - real(lkj * conj(xj));

				L_values[row_idx] = lkj;
			}

			let k_start = L_col_ptr_start[k].zx();

			if has_delta {
				match kind {
					FactorizationKind::Llt => {
						if d <= eps {
							d = copy(delta);
							dynamic_regularization_count += 1;
						}
					},
					FactorizationKind::Ldlt => {
						if let Some(signs) = regularization.dynamic_regularization_signs {
							if signs[*k] > 0 && d <= eps {
								d = copy(delta);
								dynamic_regularization_count += 1;
							} else if signs[*k] < 0 && d >= -eps {
								d = -delta;
								dynamic_regularization_count += 1;
							}
						} else if abs(d) <= eps {
							if d < zero::<T::Real>() {
								d = -delta;
								dynamic_regularization_count += 1;
							} else {
								d = copy(delta);
								dynamic_regularization_count += 1;
							}
						}
					},
				}
			}

			match kind {
				FactorizationKind::Llt => {
					if !(d > zero::<T::Real>()) {
						return Err(LltError::NonPositivePivot { index: *k + 1 });
					}
					L_values[k_start] = from_real(sqrt(d));
				},
				FactorizationKind::Ldlt => {
					if d == zero::<T::Real>() || !is_finite(d) {
						return Err(LltError::NonPositivePivot { index: *k + 1 });
					}
					L_values[k_start] = from_real(d);
				},
			}
		}
		Ok(LltInfo {
			dynamic_regularization_count,
		})
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
	pub fn factorize_simplicial_numeric_llt<I: Index, T: ComplexField>(
		L_values: &mut [T],
		A: SparseColMatRef<'_, I, T>,
		regularization: LltRegularization<T>,
		symbolic: &SymbolicSimplicialCholesky<I>,
		stack: &mut MemStack,
	) -> Result<LltInfo, LltError> {
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
	/// and stores them in `L_row_idx` and `L_values`.
	///
	/// # Note
	/// Only the upper triangular part of `A` is accessed.
	///
	/// # Panics
	/// The elimination tree and column counts must be computed by calling
	/// [`prefactorize_symbolic_cholesky`] with the same matrix, then the column pointers are
	/// computed from a prefix sum of the column counts. Otherwise, the behavior is unspecified
	/// and panics may occur.
	pub fn factorize_simplicial_numeric_llt_with_row_idx<I: Index, T: ComplexField>(
		L_values: &mut [T],
		L_row_idx: &mut [I],
		L_col_ptr: &[I],

		etree: EliminationTreeRef<'_, I>,
		A: SparseColMatRef<'_, I, T>,
		regularization: LltRegularization<T>,

		stack: &mut MemStack,
	) -> Result<LltInfo, LltError> {
		factorize_simplicial_numeric_with_row_idx(
			L_values,
			L_row_idx,
			L_col_ptr,
			FactorizationKind::Llt,
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
	pub fn factorize_simplicial_numeric_ldlt<I: Index, T: ComplexField>(
		L_values: &mut [T],
		A: SparseColMatRef<'_, I, T>,
		regularization: LdltRegularization<'_, T>,
		symbolic: &SymbolicSimplicialCholesky<I>,
		stack: &mut MemStack,
	) -> Result<LdltInfo, LdltError> {
		match factorize_simplicial_numeric(L_values, FactorizationKind::Ldlt, A, regularization, symbolic, stack) {
			Ok(info) => Ok(LdltInfo {
				dynamic_regularization_count: info.dynamic_regularization_count,
			}),
			Err(LltError::NonPositivePivot { index }) => Err(LdltError::ZeroPivot { index }),
		}
	}

	/// Computes the row indices and  numeric values of the Cholesky LDLT factor of the matrix `A`,
	/// and stores them in `L_row_idx` and `L_values`.
	///
	/// # Note
	/// Only the upper triangular part of `A` is accessed.
	///
	/// # Panics
	/// The elimination tree and column counts must be computed by calling
	/// [`prefactorize_symbolic_cholesky`] with the same matrix, then the column pointers are
	/// computed from a prefix sum of the column counts. Otherwise, the behavior is unspecified
	/// and panics may occur.
	pub fn factorize_simplicial_numeric_ldlt_with_row_idx<I: Index, T: ComplexField>(
		L_values: &mut [T],
		L_row_idx: &mut [I],
		L_col_ptr: &[I],

		etree: EliminationTreeRef<'_, I>,
		A: SparseColMatRef<'_, I, T>,
		regularization: LdltRegularization<'_, T>,

		stack: &mut MemStack,
	) -> Result<LdltInfo, LdltError> {
		match factorize_simplicial_numeric_with_row_idx(L_values, L_row_idx, L_col_ptr, FactorizationKind::Ldlt, etree, A, regularization, stack) {
			Ok(info) => Ok(LdltInfo {
				dynamic_regularization_count: info.dynamic_regularization_count,
			}),
			Err(LltError::NonPositivePivot { index }) => Err(LdltError::ZeroPivot { index }),
		}
	}

	impl<'a, I: Index, T> SimplicialLltRef<'a, I, T> {
		/// Creates a new Cholesky LLT factor from the symbolic part and numerical values.
		///
		/// # Panics
		/// Panics if `values.len() != symbolic.len_values()`>
		#[inline]
		pub fn new(symbolic: &'a SymbolicSimplicialCholesky<I>, values: &'a [T]) -> Self {
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
		pub fn values(self) -> &'a [T] {
			self.values
		}

		/// Solves the equation $\text{Op}(A) x = \text{rhs}$ and stores the result in `rhs`, where
		/// $\text{Op}$ is either the identity or the conjugate, depending on the value of `conj`.
		///
		/// # Panics
		/// Panics if `rhs.nrows() != self.symbolic().nrows()`.
		pub fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
		where
			T: ComplexField,
		{
			let _ = par;
			let _ = stack;
			let n = self.symbolic().nrows();
			assert!(rhs.nrows() == n);
			let l = SparseColMatRef::<'_, I, T>::new(self.symbolic().factor(), self.values());

			let mut rhs = rhs;
			triangular_solve::solve_lower_triangular_in_place(l, conj, DiagStatus::Generic, rhs.rb_mut(), par);
			triangular_solve::solve_lower_triangular_transpose_in_place(l, conj.compose(Conj::Yes), DiagStatus::Generic, rhs.rb_mut(), par);
		}
	}

	impl<'a, I: Index, T> SimplicialLdltRef<'a, I, T> {
		/// Creates a new Cholesky LDLT factor from the symbolic part and numerical values.
		///
		/// # Panics
		/// Panics if `values.len() != symbolic.len_values()`>
		#[inline]
		pub fn new(symbolic: &'a SymbolicSimplicialCholesky<I>, values: &'a [T]) -> Self {
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
		pub fn values(self) -> &'a [T] {
			self.values
		}

		/// Solves the equation $\text{Op}(A) x = \text{rhs}$ and stores the result in `rhs`, where
		/// $\text{Op}$ is either the identity or the conjugate, depending on the value of `conj`.
		///
		/// # Panics
		/// Panics if `rhs.nrows() != self.symbolic().nrows()`.
		pub fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
		where
			T: ComplexField,
		{
			let _ = par;
			let _ = stack;
			let n = self.symbolic().nrows();
			let ld = SparseColMatRef::<'_, I, T>::new(self.symbolic().factor(), self.values());
			assert!(rhs.nrows() == n);

			let mut x = rhs;
			triangular_solve::solve_lower_triangular_in_place(ld, conj, DiagStatus::Unit, x.rb_mut(), par);
			triangular_solve::ldlt_scale_solve_unit_lower_triangular_transpose_in_place(ld, conj.compose(Conj::Yes), x.rb_mut(), par);
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
			self.row_idx.len()
		}

		/// Returns the column pointers of the Cholesky factor.
		#[inline]
		pub fn col_ptr(&self) -> &[I] {
			&self.col_ptr
		}

		/// Returns the row indices of the Cholesky factor.
		#[inline]
		pub fn row_idx(&self) -> &[I] {
			&self.row_idx
		}

		/// Returns the Cholesky factor's symbolic structure.
		#[inline]
		pub fn factor(&self) -> SymbolicSparseColMatRef<'_, I> {
			unsafe { SymbolicSparseColMatRef::new_unchecked(self.dimension, self.dimension, &self.col_ptr, None, &self.row_idx) }
		}

		/// Returns the size and alignment of the workspace required to solve the system `A×x =
		/// rhs`.
		pub fn solve_in_place_scratch<T>(&self, rhs_ncols: usize) -> StackReq {
			let _ = rhs_ncols;
			StackReq::EMPTY
		}
	}

	/// Returns the size and alignment of the workspace required to compute the numeric
	/// Cholesky LDLT factorization of a matrix `A` with dimension `n`.
	pub fn factorize_simplicial_numeric_ldlt_scratch<I: Index, T: ComplexField>(n: usize) -> StackReq {
		let n_scratch = StackReq::new::<I>(n);
		StackReq::all_of(&[temp_mat_scratch::<T>(n, 1), n_scratch, n_scratch, n_scratch])
	}

	/// Returns the size and alignment of the workspace required to compute the numeric
	/// Cholesky LLT factorization of a matrix `A` with dimension `n`.
	pub fn factorize_simplicial_numeric_llt_scratch<I: Index, T: ComplexField>(n: usize) -> StackReq {
		factorize_simplicial_numeric_ldlt_scratch::<I, T>(n)
	}

	/// Cholesky LLT factor containing both its symbolic and numeric representations.
	#[derive(Debug)]
	pub struct SimplicialLltRef<'a, I: Index, T> {
		symbolic: &'a SymbolicSimplicialCholesky<I>,
		values: &'a [T],
	}

	/// Cholesky LDLT factors containing both the symbolic and numeric representations.
	#[derive(Debug)]
	pub struct SimplicialLdltRef<'a, I: Index, T> {
		symbolic: &'a SymbolicSimplicialCholesky<I>,
		values: &'a [T],
	}

	/// Cholesky factor structure containing its symbolic structure.
	#[derive(Debug, Clone)]
	pub struct SymbolicSimplicialCholesky<I: Index> {
		dimension: usize,
		col_ptr: alloc::vec::Vec<I>,
		row_idx: alloc::vec::Vec<I>,
		etree: alloc::vec::Vec<I>,
	}

	impl<I: Index, T> Copy for SimplicialLltRef<'_, I, T> {}
	impl<I: Index, T> Clone for SimplicialLltRef<'_, I, T> {
		fn clone(&self) -> Self {
			*self
		}
	}

	impl<I: Index, T> Copy for SimplicialLdltRef<'_, I, T> {}
	impl<I: Index, T> Clone for SimplicialLdltRef<'_, I, T> {
		fn clone(&self) -> Self {
			*self
		}
	}
}

/// Supernodal factorization module.
///
/// A supernodal factorization is one that processes the elements of the Cholesky factor of the
/// input matrix by blocks, rather than single elements. This is more efficient if the Cholesky
/// factor is somewhat dense.
pub mod supernodal {
	use super::*;
	use crate::{Shape, assert, debug_assert};

	#[doc(hidden)]
	pub fn ereach_super<'n, 'nsuper, I: Index>(
		A: SymbolicSparseColMatRef<'_, I, Dim<'n>, Dim<'n>>,
		super_etree: &Array<'nsuper, MaybeIdx<'nsuper, I>>,
		index_to_super: &Array<'n, Idx<'nsuper, I>>,
		current_row_positions: &mut Array<'nsuper, I>,
		row_idx: &mut [Idx<'n, I>],
		k: Idx<'n, usize>,
		visited: &mut Array<'nsuper, I::Signed>,
	) {
		let k_: I = *k.truncate();
		visited[index_to_super[k].zx()] = k_.to_signed();
		for i in A.row_idx_of_col(k) {
			if i >= k {
				continue;
			}
			let mut supernode_i = index_to_super[i].zx();
			loop {
				if visited[supernode_i] == k_.to_signed() {
					break;
				}

				row_idx[current_row_positions[supernode_i].zx()] = k.truncate();
				current_row_positions[supernode_i] += I::truncate(1);

				visited[supernode_i] = k_.to_signed();
				supernode_i = super_etree[supernode_i].sx().idx().unwrap();
			}
		}
	}

	fn ereach_super_ata<'m, 'n, 'nsuper, I: Index>(
		A: SymbolicSparseColMatRef<'_, I, Dim<'m>, Dim<'n>>,
		perm: Option<PermRef<'_, I, Dim<'n>>>,
		min_col: &Array<'m, MaybeIdx<'n, I>>,
		super_etree: &Array<'nsuper, MaybeIdx<'nsuper, I>>,
		index_to_super: &Array<'n, Idx<'nsuper, I>>,
		current_row_positions: &mut Array<'nsuper, I>,
		row_idx: &mut [Idx<'n, I>],
		k: Idx<'n, usize>,
		visited: &mut Array<'nsuper, I::Signed>,
	) {
		let k_: I = *k.truncate();
		visited[index_to_super[k].zx()] = k_.to_signed();

		let fwd = perm.map(|perm| perm.bound_arrays().0);
		let fwd = |i: Idx<'n, usize>| fwd.map(|fwd| fwd[k].zx()).unwrap_or(i);
		for i in A.row_idx_of_col(fwd(k)) {
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

				row_idx[current_row_positions[supernode_i].zx()] = k.truncate();
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

	/// A single supernode from the Cholesky factor.
	#[derive(Debug)]
	pub struct SupernodeRef<'a, I: Index, T> {
		matrix: MatRef<'a, T>,
		symbolic: SymbolicSupernodeRef<'a, I>,
	}

	impl<I: Index> Copy for SymbolicSupernodeRef<'_, I> {}
	impl<I: Index> Clone for SymbolicSupernodeRef<'_, I> {
		fn clone(&self) -> Self {
			*self
		}
	}

	impl<I: Index, T> Copy for SupernodeRef<'_, I, T> {}
	impl<I: Index, T> Clone for SupernodeRef<'_, I, T> {
		fn clone(&self) -> Self {
			*self
		}
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

	impl<'a, I: Index, T> SupernodeRef<'a, I, T> {
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
		pub fn val(self) -> MatRef<'a, T> {
			self.matrix
		}
	}

	/// Cholesky LLT factor containing both its symbolic and numeric representations.
	#[derive(Debug)]
	pub struct SupernodalLltRef<'a, I: Index, T> {
		symbolic: &'a SymbolicSupernodalCholesky<I>,
		values: &'a [T],
	}

	/// Cholesky LDLT factors containing both the symbolic and numeric representations.
	#[derive(Debug)]
	pub struct SupernodalLdltRef<'a, I: Index, T> {
		symbolic: &'a SymbolicSupernodalCholesky<I>,
		values: &'a [T],
	}

	/// Cholesky Bunch-Kaufman factors containing both the symbolic and numeric representations.
	#[derive(Debug)]
	pub struct SupernodalIntranodeBunchKaufmanRef<'a, I: Index, T> {
		symbolic: &'a SymbolicSupernodalCholesky<I>,
		values: &'a [T],
		subdiag: &'a [T],
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
		pub(crate) col_ptr_for_row_idx: alloc::vec::Vec<I>,
		pub(crate) col_ptr_for_val: alloc::vec::Vec<I>,
		pub(crate) row_idx: alloc::vec::Vec<I>,

		pub(crate) nnz_per_super: Option<alloc::vec::Vec<I>>,
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
			self.col_ptr_for_val()[self.n_supernodes()].zx()
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
		pub fn col_ptr_for_row_idx(&self) -> &[I] {
			&self.col_ptr_for_row_idx
		}

		/// Returns the column pointers for numerical values of each supernode.
		#[inline]
		pub fn col_ptr_for_val(&self) -> &[I] {
			&self.col_ptr_for_val
		}

		/// Returns the row indices of the Cholesky factor.
		///
		/// # Note
		/// Note that the row indices of each supernode do not contain those of the block diagonal
		/// part.
		#[inline]
		pub fn row_idx(&self) -> &[I] {
			&self.row_idx
		}

		/// Returns the symbolic structure of the `s`'th supernode.
		#[inline]
		pub fn supernode(&self, s: usize) -> supernodal::SymbolicSupernodeRef<'_, I> {
			let symbolic = self;
			let start = symbolic.supernode_begin[s].zx();
			let pattern = &symbolic.row_idx()[symbolic.col_ptr_for_row_idx()[s].zx()..symbolic.col_ptr_for_row_idx()[s + 1].zx()];
			supernodal::SymbolicSupernodeRef { start, pattern }
		}

		/// Returns the size and alignment of the workspace required to solve the system `A×x =
		/// rhs`.
		pub fn solve_in_place_scratch<T: ComplexField>(&self, rhs_ncols: usize) -> StackReq {
			let mut req = StackReq::EMPTY;
			let symbolic = self;
			for s in 0..symbolic.n_supernodes() {
				let s = self.supernode(s);
				req = req.or(temp_mat_scratch::<T>(s.pattern.len(), rhs_ncols));
			}
			req
		}

		#[doc(hidden)]
		pub fn __prepare_for_refactorize(&mut self) -> Result<(), FaerError> {
			let mut v = try_zeroed(self.n_supernodes())?;
			for s in 0..self.n_supernodes() {
				v[s] = self.col_ptr_for_row_idx[s + 1] - self.col_ptr_for_row_idx[s];
			}
			self.nnz_per_super = Some(v);
			Ok(())
		}

		#[doc(hidden)]
		#[track_caller]
		pub fn __nnz_per_super(&self) -> &[I] {
			self.nnz_per_super.as_deref().unwrap()
		}

		#[doc(hidden)]
		pub fn __refactorize(&mut self, A: SymbolicSparseColMatRef<'_, I>, etree: &[I::Signed], stack: &mut MemStack) {
			generativity::make_guard!(N);
			generativity::make_guard!(N_SUPERNODES);
			let N = self.nrows().bind(N);
			let N_SUPERNODES = self.nrows().bind(N_SUPERNODES);

			let A = A.as_shape(N, N);
			let n = *N;
			let n_supernodes = *N_SUPERNODES;
			let none = I::Signed::truncate(NONE);

			let etree = MaybeIdx::<I>::from_slice_ref_checked(etree, N);
			let etree = Array::from_ref(etree, N);

			let (index_to_super, stack) = unsafe { stack.make_raw::<I>(n) };
			let (current_row_positions, stack) = unsafe { stack.make_raw::<I>(n_supernodes) };
			let (visited, stack) = unsafe { stack.make_raw::<I::Signed>(n_supernodes) };
			let (super_etree, _) = unsafe { stack.make_raw::<I::Signed>(n_supernodes) };

			let super_etree = Array::from_mut(super_etree, N_SUPERNODES);
			let index_to_super = Array::from_mut(index_to_super, N);

			let mut supernode_begin = 0usize;
			for s in N_SUPERNODES.indices() {
				let size = self.supernode_end()[*s].zx() - self.supernode_begin()[*s].zx();
				index_to_super.as_mut()[supernode_begin..][..size].fill(*s.truncate::<I>());
				supernode_begin += size;
			}

			let index_to_super = Array::from_mut(Idx::from_slice_mut_checked(index_to_super.as_mut(), N_SUPERNODES), N);

			let mut supernode_begin = 0usize;
			for s in N_SUPERNODES.indices() {
				let size = self.supernode_end()[*s + 1].zx() - self.supernode_begin()[*s].zx();
				let last = supernode_begin + size - 1;
				if let Some(parent) = etree[N.check(last)].idx() {
					super_etree[s] = index_to_super[parent.zx()].to_signed();
				} else {
					super_etree[s] = none;
				}
				supernode_begin += size;
			}

			let super_etree = Array::from_mut(
				MaybeIdx::<'_, I>::from_slice_mut_checked(super_etree.as_mut(), N_SUPERNODES),
				N_SUPERNODES,
			);

			let visited = Array::from_mut(visited, N_SUPERNODES);
			let current_row_positions = Array::from_mut(current_row_positions, N_SUPERNODES);

			visited.as_mut().fill(I::Signed::truncate(NONE));
			current_row_positions.as_mut().fill(I::truncate(0));

			for s in N_SUPERNODES.indices() {
				let k1 = ghost::IdxInc::new_checked(self.supernode_begin()[*s].zx(), N);
				let k2 = ghost::IdxInc::new_checked(self.supernode_end()[*s].zx(), N);

				for k in k1.range_to(k2) {
					ereach_super(
						A,
						super_etree,
						index_to_super,
						current_row_positions,
						unsafe { Idx::from_slice_mut_unchecked(&mut self.row_idx) },
						k,
						visited,
					);
				}
			}

			let Some(nnz_per_super) = self.nnz_per_super.as_deref_mut() else {
				panic!()
			};

			for s in N_SUPERNODES.indices() {
				nnz_per_super[*s] = current_row_positions[s] - self.supernode_begin[*s];
			}
		}
	}

	impl<I: Index, T> Copy for SupernodalLdltRef<'_, I, T> {}
	impl<I: Index, T> Clone for SupernodalLdltRef<'_, I, T> {
		fn clone(&self) -> Self {
			*self
		}
	}
	impl<I: Index, T> Copy for SupernodalLltRef<'_, I, T> {}
	impl<I: Index, T> Clone for SupernodalLltRef<'_, I, T> {
		fn clone(&self) -> Self {
			*self
		}
	}
	impl<I: Index, T> Copy for SupernodalIntranodeBunchKaufmanRef<'_, I, T> {}
	impl<I: Index, T> Clone for SupernodalIntranodeBunchKaufmanRef<'_, I, T> {
		fn clone(&self) -> Self {
			*self
		}
	}

	impl<'a, I: Index, T> SupernodalLdltRef<'a, I, T> {
		/// Creates new Cholesky LDLT factors from the symbolic part and
		/// numerical values.
		///
		/// # Panics
		/// - Panics if `values.len() != symbolic.len_values()`.
		#[inline]
		pub fn new(symbolic: &'a SymbolicSupernodalCholesky<I>, values: &'a [T]) -> Self {
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
		pub fn values(self) -> &'a [T] {
			self.values
		}

		/// Returns the `s`'th supernode.
		#[inline]
		pub fn supernode(self, s: usize) -> SupernodeRef<'a, I, T> {
			let symbolic = self.symbolic();
			let L_values = self.values();
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();

			let s_pattern = &symbolic.row_idx()[symbolic.col_ptr_for_row_idx()[s].zx()..symbolic.col_ptr_for_row_idx()[s + 1].zx()];
			let s_ncols = s_end - s_start;
			let s_nrows = s_pattern.len() + s_ncols;

			let Ls = MatRef::from_column_major_slice(
				&L_values[symbolic.col_ptr_for_val()[s].zx()..symbolic.col_ptr_for_val()[s + 1].zx()],
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

		/// Solves the equation $\text{Op}(A) x = \text{rhs}$ and stores the result in `rhs`, where
		/// $\text{Op}$ is either the identity or the conjugate, depending on the value of `conj`.
		///
		/// # Panics
		/// Panics if `rhs.nrows() != self.symbolic().nrows()`.
		#[math]
		pub fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
		where
			T: ComplexField,
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
				linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(Ls_top, conj, x_top.rb_mut(), par);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack.rb_mut()) };
				let mut tmp = tmp.as_mat_mut();
				linalg::matmul::matmul_with_conj(tmp.rb_mut(), Accum::Replace, Ls_bot, conj, x_top.rb(), Conj::No, one::<T>(), par);

				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						x.write(i, j, x.read(i, j) - tmp.read(idx, j))
					}
				}
			}
			for s in 0..symbolic.n_supernodes() {
				let s = self.supernode(s);
				let size = s.matrix.ncols();
				let Ds = s.matrix.diagonal().column_vector();
				for j in 0..k {
					for idx in 0..size {
						let d_inv = recip(real(Ds.read(idx)));
						let i = idx + s.start();
						x.write(i, j, mul_real(x.read(i, j), d_inv))
					}
				}
			}
			for s in (0..symbolic.n_supernodes()).rev() {
				let s = self.supernode(s);
				let size = s.matrix.ncols();
				let Ls = s.matrix;
				let (Ls_top, Ls_bot) = Ls.split_at_row(size);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack.rb_mut()) };
				let mut tmp = tmp.as_mat_mut();
				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						tmp.write(idx, j, x.read(i, j));
					}
				}

				let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
				linalg::matmul::matmul_with_conj(
					x_top.rb_mut(),
					Accum::Add,
					Ls_bot.transpose(),
					conj.compose(Conj::Yes),
					tmp.rb(),
					Conj::No,
					-one::<T>(),
					par,
				);
				linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(
					Ls_top.transpose(),
					conj.compose(Conj::Yes),
					x_top.rb_mut(),
					par,
				);
			}
		}
	}

	impl<'a, I: Index, T> SupernodalLltRef<'a, I, T> {
		/// Creates a new Cholesky LLT factor from the symbolic part and
		/// numerical values.
		///
		/// # Panics
		/// - Panics if `values.len() != symbolic.len_values()`.
		#[inline]
		pub fn new(symbolic: &'a SymbolicSupernodalCholesky<I>, values: &'a [T]) -> Self {
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
		pub fn values(self) -> &'a [T] {
			self.values
		}

		/// Returns the `s`'th supernode.
		#[inline]
		pub fn supernode(self, s: usize) -> SupernodeRef<'a, I, T> {
			let symbolic = self.symbolic();
			let L_values = self.values();
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();

			let s_pattern = &symbolic.row_idx()[symbolic.col_ptr_for_row_idx()[s].zx()..symbolic.col_ptr_for_row_idx()[s + 1].zx()];
			let s_ncols = s_end - s_start;
			let s_nrows = s_pattern.len() + s_ncols;

			let Ls = MatRef::from_column_major_slice(
				&L_values[symbolic.col_ptr_for_val()[s].zx()..symbolic.col_ptr_for_val()[s + 1].zx()],
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

		/// Solves the equation $\text{Op}(L) x = \text{rhs}$ and stores the result in `rhs`, where
		/// $\text{Op}$ is either the identity or the conjugate, depending on the value of `conj`.
		///
		/// # Panics
		/// Panics if `rhs.nrows() != self.symbolic().nrows()`.
		#[math]
		pub fn l_solve_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
		where
			T: ComplexField,
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
				linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(Ls_top, conj, x_top.rb_mut(), par);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack.rb_mut()) };
				let mut tmp = tmp.as_mat_mut();
				linalg::matmul::matmul_with_conj(tmp.rb_mut(), Accum::Replace, Ls_bot, conj, x_top.rb(), Conj::No, one::<T>(), par);

				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						x.write(i, j, x.read(i, j) - tmp.read(idx, j))
					}
				}
			}
		}

		/// Solves the equation $\text{Op}(L^\top) x = \text{rhs}$ and stores the result in `rhs`,
		/// where $\text{Op}$ is either the identity or the conjugate, depending on the
		/// value of `conj`.
		///
		/// # Panics
		/// Panics if `rhs.nrows() != self.symbolic().nrows()`.
		#[inline]
		#[math]
		pub fn l_transpose_solve_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
		where
			T: ComplexField,
		{
			let symbolic = self.symbolic();
			let n = symbolic.nrows();
			assert!(rhs.nrows() == n);

			let mut x = rhs;
			let mut stack = stack;
			let k = x.ncols();
			for s in (0..symbolic.n_supernodes()).rev() {
				let s = self.supernode(s);
				let size = s.matrix.ncols();
				let Ls = s.matrix;
				let (Ls_top, Ls_bot) = Ls.split_at_row(size);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack.rb_mut()) };
				let mut tmp = tmp.as_mat_mut();
				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						tmp.write(idx, j, x.read(i, j));
					}
				}

				let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
				linalg::matmul::matmul_with_conj(x_top.rb_mut(), Accum::Add, Ls_bot.transpose(), conj, tmp.rb(), Conj::No, -one::<T>(), par);
				linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(Ls_top.transpose(), conj, x_top.rb_mut(), par);
			}
		}

		/// Solves the equation $\text{Op}(A) x = \text{rhs}$ and stores the result in `rhs`, where
		/// $\text{Op}$ is either the identity or the conjugate, depending on the value of `conj`.
		///
		/// # Panics
		/// Panics if `rhs.nrows() != self.symbolic().nrows()`.
		#[math]
		pub fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
		where
			T: ComplexField,
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
				linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(Ls_top, conj, x_top.rb_mut(), par);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack.rb_mut()) };
				let mut tmp = tmp.as_mat_mut();
				linalg::matmul::matmul_with_conj(tmp.rb_mut(), Accum::Replace, Ls_bot, conj, x_top.rb(), Conj::No, one::<T>(), par);

				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						x.write(i, j, x.read(i, j) - tmp.read(idx, j))
					}
				}
			}
			for s in (0..symbolic.n_supernodes()).rev() {
				let s = self.supernode(s);
				let size = s.matrix.ncols();
				let Ls = s.matrix;
				let (Ls_top, Ls_bot) = Ls.split_at_row(size);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack.rb_mut()) };
				let mut tmp = tmp.as_mat_mut();
				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						tmp.write(idx, j, x.read(i, j));
					}
				}

				let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
				linalg::matmul::matmul_with_conj(
					x_top.rb_mut(),
					Accum::Add,
					Ls_bot.transpose(),
					conj.compose(Conj::Yes),
					tmp.rb(),
					Conj::No,
					-one::<T>(),
					par,
				);
				linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(Ls_top.transpose(), conj.compose(Conj::Yes), x_top.rb_mut(), par);
			}
		}
	}

	impl<'a, I: Index, T> SupernodalIntranodeBunchKaufmanRef<'a, I, T> {
		/// Creates a new Cholesky intranodal Bunch-Kaufman factor from the symbolic part and
		/// numerical values, as well as the pivoting permutation.
		///
		/// # Panics
		/// - Panics if `values.len() != symbolic.len_values()`.
		/// - Panics if `subdiag.len() != symbolic.nrows()`.
		/// - Panics if `perm.len() != symbolic.nrows()`.
		#[inline]
		pub fn new(symbolic: &'a SymbolicSupernodalCholesky<I>, values: &'a [T], subdiag: &'a [T], perm: PermRef<'a, I>) -> Self {
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
		pub fn val(self) -> &'a [T] {
			self.values
		}

		/// Returns the `s`'th supernode.
		#[inline]
		pub fn supernode(self, s: usize) -> SupernodeRef<'a, I, T> {
			let symbolic = self.symbolic();
			let L_values = self.val();
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();

			let s_pattern = &symbolic.row_idx()[symbolic.col_ptr_for_row_idx()[s].zx()..symbolic.col_ptr_for_row_idx()[s + 1].zx()];
			let s_ncols = s_end - s_start;
			let s_nrows = s_pattern.len() + s_ncols;

			let Ls = MatRef::from_column_major_slice(
				&L_values[symbolic.col_ptr_for_val()[s].zx()..symbolic.col_ptr_for_val()[s + 1].zx()],
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

		/// Solves the system $\text{Op}(L B L^H) x = \text{rhs}$, where $\text{Op}$ is either the
		/// identity or the conjugate depending on the value of `conj`.
		///
		/// # Note
		/// Note that this function doesn't apply the pivoting permutation. Users are expected to
		/// apply it manually to `rhs` before and after calling this function.
		///
		/// # Panics
		/// Panics if `rhs.nrows() != self.symbolic().nrows()`.
		#[math]
		pub fn solve_in_place_no_numeric_permute_with_conj(self, conj_lb: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
		where
			T: ComplexField,
		{
			let symbolic = self.symbolic();
			let n = symbolic.nrows();
			assert!(rhs.nrows() == n);

			let mut x = rhs;

			let k = x.ncols();
			for s in 0..symbolic.n_supernodes() {
				let s = self.supernode(s);
				let size = s.matrix.ncols();
				let Ls = s.matrix;
				let (Ls_top, Ls_bot) = Ls.split_at_row(size);
				let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
				linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(Ls_top, conj_lb, x_top.rb_mut(), par);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack) };
				let mut tmp = tmp.as_mat_mut();
				linalg::matmul::matmul_with_conj(tmp.rb_mut(), Accum::Replace, Ls_bot, conj_lb, x_top.rb(), Conj::No, one::<T>(), par);

				let inv = self.perm.arrays().1;
				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						let i = inv[i].zx();
						x[(i, j)] = x[(i, j)] - tmp[(idx, j)];
					}
				}
			}
			for s in 0..symbolic.n_supernodes() {
				let s = self.supernode(s);
				let size = s.matrix.ncols();
				let Bs = s.val();
				let subdiag = &self.subdiag[s.start()..s.start() + size];

				let mut idx = 0;
				while idx < size {
					let subdiag = copy(subdiag[idx]);
					let i = idx + s.start();
					if subdiag == zero::<T>() {
						let d = recip(real(Bs[(idx, idx)]));
						for j in 0..k {
							x[(i, j)] = mul_real(x[(i, j)], d);
						}
						idx += 1;
					} else {
						let mut d21 = conj_lb.apply_rt(&subdiag);
						d21 = recip(d21);
						let d11 = mul_real(conj(d21), real(Bs[(idx, idx)]));
						let d22 = mul_real(d21, real(Bs[(idx + 1, idx + 1)]));

						let denom = recip(d11 * d22 - one::<T>());

						for j in 0..k {
							let xk = x[(i, j)] * conj(d21);
							let xkp1 = x[(i + 1, j)] * d21;

							x[(i, j)] = (d22 * xk - xkp1) * denom;
							x[(i + 1, j)] = (d11 * xkp1 - xk) * denom;
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

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack) };
				let mut tmp = tmp.as_mat_mut();
				let inv = self.perm.arrays().1;
				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						let i = inv[i].zx();
						tmp[(idx, j)] = copy(x[(i, j)]);
					}
				}

				let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
				linalg::matmul::matmul_with_conj(
					x_top.rb_mut(),
					Accum::Add,
					Ls_bot.transpose(),
					conj_lb.compose(Conj::Yes),
					tmp.rb(),
					Conj::No,
					-one::<T>(),
					par,
				);
				linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(
					Ls_top.transpose(),
					conj_lb.compose(Conj::Yes),
					x_top.rb_mut(),
					par,
				);
			}
		}
	}

	/// Returns the size and alignment of the workspace required to compute the symbolic supernodal
	/// factorization of a matrix of size `n`.
	pub fn factorize_supernodal_symbolic_cholesky_scratch<I: Index>(n: usize) -> StackReq {
		let n_scratch = StackReq::new::<I>(n);
		StackReq::all_of(&[n_scratch; 4])
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
		stack: &mut MemStack,
		params: SymbolicSupernodalParams<'_>,
	) -> Result<SymbolicSupernodalCholesky<I>, FaerError> {
		let n = A.nrows();
		assert!(A.nrows() == A.ncols());
		assert!(etree.into_inner().len() == n);
		assert!(col_counts.len() == n);
		with_dim!(N, n);
		ghost_factorize_supernodal_symbolic(
			A.as_shape(N, N),
			None,
			None,
			CholeskyInput::A,
			etree.as_bound(N),
			Array::from_ref(col_counts, N),
			stack,
			params,
		)
	}

	pub(crate) enum CholeskyInput {
		A,
		ATA,
	}

	pub(crate) fn ghost_factorize_supernodal_symbolic<'m, 'n, I: Index>(
		A: SymbolicSparseColMatRef<'_, I, Dim<'m>, Dim<'n>>,
		col_perm: Option<PermRef<'_, I, Dim<'n>>>,
		min_col: Option<&Array<'m, MaybeIdx<'n, I>>>,
		input: CholeskyInput,
		etree: &Array<'n, MaybeIdx<'n, I>>,
		col_counts: &Array<'n, I>,
		stack: &mut MemStack,
		params: SymbolicSupernodalParams<'_>,
	) -> Result<SymbolicSupernodalCholesky<I>, FaerError> {
		let to_wide = |i: I| i.zx() as u128;
		let from_wide = |i: u128| I::truncate(i as usize);
		let from_wide_checked = |i: u128| -> Option<I> { (i <= to_wide(I::from_signed(I::Signed::MAX))).then_some(I::truncate(i as usize)) };

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
				col_ptr_for_row_idx: try_collect([zero])?,
				col_ptr_for_val: try_collect([zero])?,
				row_idx: alloc::vec::Vec::new(),
				nnz_per_super: None,
			});
		}
		let mut original_stack = stack;

		let (index_to_super__, stack) = unsafe { original_stack.rb_mut().make_raw::<I>(n) };
		let (super_etree__, stack) = unsafe { stack.make_raw::<I::Signed>(n) };
		let (supernode_sizes__, stack) = unsafe { stack.make_raw::<I>(n) };
		let (child_count__, _) = unsafe { stack.make_raw::<I>(n) };

		let child_count = Array::from_mut(child_count__, N);
		let index_to_super = Array::from_mut(index_to_super__, N);

		child_count.as_mut().fill(zero);
		for j in N.indices() {
			if let Some(parent) = etree[j].idx() {
				child_count[parent.zx()] += one;
			}
		}

		supernode_sizes__.fill(zero);
		let mut current_supernode = 0usize;
		supernode_sizes__[0] = one;
		for (j_prev, j) in iter::zip(N.indices().take(n - 1), N.indices().skip(1)) {
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
		let supernode_begin__ = {
			with_dim!(N_FUNDAMENTAL_SUPERNODES, n_fundamental_supernodes);
			let supernode_sizes = Array::from_mut(&mut supernode_sizes__[..n_fundamental_supernodes], N_FUNDAMENTAL_SUPERNODES);
			let super_etree = Array::from_mut(&mut super_etree__[..n_fundamental_supernodes], N_FUNDAMENTAL_SUPERNODES);

			let mut supernode_begin = 0usize;
			for s in N_FUNDAMENTAL_SUPERNODES.indices() {
				let size = supernode_sizes[s].zx();
				index_to_super.as_mut()[supernode_begin..][..size].fill(*s.truncate::<I>());
				supernode_begin += size;
			}

			let index_to_super = Array::from_mut(Idx::from_slice_mut_checked(index_to_super.as_mut(), N_FUNDAMENTAL_SUPERNODES), N);

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
				MaybeIdx::<'_, I>::from_slice_mut_checked(super_etree.as_mut(), N_FUNDAMENTAL_SUPERNODES),
				N_FUNDAMENTAL_SUPERNODES,
			);

			if let Some(relax) = params.relax {
				let mut mem = dyn_stack::MemBuffer::try_new(StackReq::all_of(&[StackReq::new::<I>(n_fundamental_supernodes); 5]))
					.ok()
					.ok_or(FaerError::OutOfMemory)?;
				let stack = MemStack::new(&mut mem);

				let child_lists = bytemuck::cast_slice_mut(&mut child_count.as_mut()[..n_fundamental_supernodes]);
				let (child_list_heads, stack) = unsafe { stack.make_raw::<I::Signed>(n_fundamental_supernodes) };
				let (last_merged_children, stack) = unsafe { stack.make_raw::<I::Signed>(n_fundamental_supernodes) };
				let (merge_parents, stack) = unsafe { stack.make_raw::<I::Signed>(n_fundamental_supernodes) };
				let (fundamental_supernode_degrees, stack) = unsafe { stack.make_raw::<I>(n_fundamental_supernodes) };
				let (num_zeros, _) = unsafe { stack.make_raw::<I>(n_fundamental_supernodes) };

				let child_lists = Array::from_mut(ghost::fill_none::<I>(child_lists, N_FUNDAMENTAL_SUPERNODES), N_FUNDAMENTAL_SUPERNODES);
				let child_list_heads = Array::from_mut(
					ghost::fill_none::<I>(child_list_heads, N_FUNDAMENTAL_SUPERNODES),
					N_FUNDAMENTAL_SUPERNODES,
				);
				let last_merged_children = Array::from_mut(
					ghost::fill_none::<I>(last_merged_children, N_FUNDAMENTAL_SUPERNODES),
					N_FUNDAMENTAL_SUPERNODES,
				);
				let merge_parents = Array::from_mut(ghost::fill_none::<I>(merge_parents, N_FUNDAMENTAL_SUPERNODES), N_FUNDAMENTAL_SUPERNODES);
				let fundamental_supernode_degrees = Array::from_mut(fundamental_supernode_degrees, N_FUNDAMENTAL_SUPERNODES);
				let num_zeros = Array::from_mut(num_zeros, N_FUNDAMENTAL_SUPERNODES);

				let mut supernode_begin = 0usize;
				for s in N_FUNDAMENTAL_SUPERNODES.indices() {
					let size = supernode_sizes[s].zx();
					fundamental_supernode_degrees[s] = col_counts[N.check(supernode_begin + size - 1)] - one;
					supernode_begin += size;
				}

				for s in N_FUNDAMENTAL_SUPERNODES.indices() {
					if let Some(parent) = super_etree[s].idx() {
						let parent = parent.zx();
						child_lists[s] = child_list_heads[parent];
						child_list_heads[parent] = MaybeIdx::from_index(s.truncate());
					}
				}

				num_zeros.as_mut().fill(I::truncate(0));
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
								let num_new_zeros = (parent_size + parent_degree - child_degree) * child_size;

								if num_new_zeros == 0 {
									num_parent_zeros + num_child_zeros
								} else {
									let num_old_zeros = num_child_zeros + num_parent_zeros;
									let num_zeros = num_new_zeros + num_old_zeros;

									let combined_size = child_size + parent_size;
									let num_expanded_entries = (combined_size * (combined_size + 1)) / 2 + parent_degree * combined_size;

									let f = || {
										for cutoff in relax {
											let num_zeros_cutoff = num_expanded_entries as f64 * cutoff.1;
											if cutoff.0 >= combined_size && num_zeros_cutoff >= num_zeros as f64 {
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

							let num_proposed_new_zeros = status_num_merged_zeros - (num_child_zeros + num_parent_zeros);
							if child_size > largest_mergable_size || num_proposed_new_zeros < num_new_zeros {
								merging_child = MaybeIdx::from_index(child);
								num_new_zeros = num_proposed_new_zeros;
								num_merged_zeros = status_num_merged_zeros;
								largest_mergable_size = child_size;
							}

							child_ = child_lists[child];
						}

						if let Some(merging_child) = merging_child.idx() {
							supernode_sizes[parent] = supernode_sizes[parent] + supernode_sizes[merging_child];
							supernode_sizes[merging_child] = zero;
							num_zeros[parent] = I::truncate(num_merged_zeros);

							merge_parents[merging_child] = if let Some(child) = last_merged_children[parent].idx() {
								MaybeIdx::from_index(child)
							} else {
								MaybeIdx::from_index(parent.truncate())
							};

							last_merged_children[parent] = if let Some(child) = last_merged_children[merging_child].idx() {
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
				supernode_begin__[1..].copy_from_slice(&fundamental_supernode_degrees.as_ref()[..n_relaxed_supernodes]);

				supernode_begin__
			} else {
				let mut supernode_begin__ = try_zeroed(n_fundamental_supernodes + 1)?;

				let mut supernode_begin = 0usize;
				for s in N_FUNDAMENTAL_SUPERNODES.indices() {
					let size = supernode_sizes[s].zx();
					supernode_begin__[*s + 1] = col_counts[N.check(supernode_begin + size - 1)] - one;
					supernode_begin += size;
				}

				supernode_begin__
			}
		};

		let n_supernodes = supernode_begin__.len() - 1;

		let (supernode_begin__, col_ptr_for_row_idx__, col_ptr_for_val__, row_idx__) = {
			with_dim!(N_SUPERNODES, n_supernodes);
			let supernode_sizes = Array::from_mut(&mut supernode_sizes__[..n_supernodes], N_SUPERNODES);

			if n_supernodes != n_fundamental_supernodes {
				let mut supernode_begin = 0usize;
				for s in N_SUPERNODES.indices() {
					let size = supernode_sizes[s].zx();
					index_to_super.as_mut()[supernode_begin..][..size].fill(*s.truncate::<I>());
					supernode_begin += size;
				}

				let index_to_super = Array::from_mut(Idx::<'_, I>::from_slice_mut_checked(index_to_super.as_mut(), N_SUPERNODES), N);
				let super_etree = Array::from_mut(&mut super_etree__[..n_supernodes], N_SUPERNODES);

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

			let index_to_super = Array::from_mut(Idx::from_slice_mut_checked(index_to_super.as_mut(), N_SUPERNODES), N);

			let mut supernode_begin__ = supernode_begin__;
			let mut col_ptr_for_row_idx__ = try_zeroed::<I>(n_supernodes + 1)?;
			let mut col_ptr_for_val__ = try_zeroed::<I>(n_supernodes + 1)?;

			let mut row_ptr = zero;
			let mut val_ptr = zero;

			supernode_begin__[0] = zero;

			let mut row_idx__ = {
				let mut wide_val_count = 0u128;
				for (s, [current, next]) in iter::zip(
					N_SUPERNODES.indices(),
					windows2(Cell::as_slice_of_cells(Cell::from_mut(&mut *supernode_begin__))),
				) {
					let degree = next.get();
					let ncols = supernode_sizes[s];
					let nrows = degree + ncols;
					supernode_sizes[s] = row_ptr;
					next.set(current.get() + ncols);

					col_ptr_for_row_idx__[*s] = row_ptr;
					col_ptr_for_val__[*s] = val_ptr;

					let wide_matrix_size = to_wide(nrows) * to_wide(ncols);
					wide_val_count += wide_matrix_size;

					row_ptr += degree;
					val_ptr = from_wide(to_wide(val_ptr) + wide_matrix_size);
				}
				col_ptr_for_row_idx__[n_supernodes] = row_ptr;
				col_ptr_for_val__[n_supernodes] = val_ptr;
				from_wide_checked(wide_val_count).ok_or(FaerError::IndexOverflow)?;

				try_zeroed::<I>(row_ptr.zx())?
			};

			let super_etree = Array::from_ref(
				MaybeIdx::from_slice_ref_checked(&super_etree__[..n_supernodes], N_SUPERNODES),
				N_SUPERNODES,
			);

			let current_row_positions = supernode_sizes;

			let row_idx = Idx::from_slice_mut_checked(&mut row_idx__, N);
			let visited = Array::from_mut(bytemuck::cast_slice_mut(&mut child_count.as_mut()[..n_supernodes]), N_SUPERNODES);

			visited.as_mut().fill(I::Signed::truncate(NONE));
			if matches!(input, CholeskyInput::A) {
				let A = A.as_shape(N, N);
				for s in N_SUPERNODES.indices() {
					let k1 = ghost::IdxInc::new_checked(supernode_begin__[*s].zx(), N);
					let k2 = ghost::IdxInc::new_checked(supernode_begin__[*s + 1].zx(), N);

					for k in k1.range_to(k2) {
						ereach_super(A, super_etree, index_to_super, current_row_positions, row_idx, k, visited);
					}
				}
			} else {
				let min_col = min_col.unwrap();
				for s in N_SUPERNODES.indices() {
					let k1 = ghost::IdxInc::new_checked(supernode_begin__[*s].zx(), N);
					let k2 = ghost::IdxInc::new_checked(supernode_begin__[*s + 1].zx(), N);

					for k in k1.range_to(k2) {
						ereach_super_ata(
							A,
							col_perm,
							min_col,
							super_etree,
							index_to_super,
							current_row_positions,
							row_idx,
							k,
							visited,
						);
					}
				}
			}

			debug_assert!(current_row_positions.as_ref() == &col_ptr_for_row_idx__[1..]);

			(supernode_begin__, col_ptr_for_row_idx__, col_ptr_for_val__, row_idx__)
		};

		let mut supernode_etree__: alloc::vec::Vec<I> = try_collect(bytemuck::cast_slice(&super_etree__[..n_supernodes]).iter().copied())?;
		let mut supernode_postorder__ = try_zeroed::<I>(n_supernodes)?;

		let mut descendent_count__ = try_zeroed::<I>(n_supernodes)?;

		{
			with_dim!(N_SUPERNODES, n_supernodes);
			let post = Array::from_mut(&mut supernode_postorder__, N_SUPERNODES);
			let desc_count = Array::from_mut(&mut descendent_count__, N_SUPERNODES);
			let etree: &Array<'_, MaybeIdx<'_, I>> = Array::from_ref(
				MaybeIdx::from_slice_ref_checked(bytemuck::cast_slice(&supernode_etree__), N_SUPERNODES),
				N_SUPERNODES,
			);

			for s in N_SUPERNODES.indices() {
				if let Some(parent) = etree[s].idx() {
					let parent = parent.zx();
					desc_count[parent] = desc_count[parent] + desc_count[s] + one;
				}
			}

			ghost_postorder(post, etree, original_stack);
			let post_inv = Array::from_mut(bytemuck::cast_slice_mut(&mut supernode_etree__), N_SUPERNODES);
			for i in N_SUPERNODES.indices() {
				post_inv[N_SUPERNODES.check(post[i].zx())] = I::truncate(*i);
			}
		};

		Ok(SymbolicSupernodalCholesky {
			dimension: n,
			supernode_postorder: supernode_postorder__,
			supernode_postorder_inv: supernode_etree__,
			descendant_count: descendent_count__,
			supernode_begin: supernode_begin__,
			col_ptr_for_row_idx: col_ptr_for_row_idx__,
			col_ptr_for_val: col_ptr_for_val__,
			row_idx: row_idx__,
			nnz_per_super: None,
		})
	}

	#[inline]
	pub(crate) fn partition_fn<I: Index>(idx: usize) -> impl Fn(&I) -> bool {
		let idx = I::truncate(idx);
		move |&i| i < idx
	}

	/// Returns the size and alignment of the workspace required to compute the numeric
	/// Cholesky LLT factorization of a matrix `A` with dimension `n`.
	pub fn factorize_supernodal_numeric_llt_scratch<I: Index, T: ComplexField>(symbolic: &SymbolicSupernodalCholesky<I>, par: Par) -> StackReq {
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let row_ind = &*symbolic.row_idx;

		let mut req = StackReq::empty();
		for s in 0..n_supernodes {
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();

			let s_ncols = s_end - s_start;

			let s_postordered = post_inv[s].zx();
			let desc_count = desc_count[s].zx();
			for d in &post[s_postordered - desc_count..s_postordered] {
				let mut d_scratch = StackReq::empty();
				let d = d.zx();

				let d_pattern = &row_ind[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()];
				let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
				let d_pattern_mid_len = d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));

				d_scratch = d_scratch.and(temp_mat_scratch::<T>(d_pattern.len() - d_pattern_start, d_pattern_mid_len));
				req = req.or(d_scratch);
			}
			req = req.or(linalg::cholesky::llt::factor::cholesky_in_place_scratch::<T>(
				s_ncols,
				par,
				Default::default(),
			));
		}
		req.and(StackReq::new::<I>(n))
	}

	/// Returns the size and alignment of the workspace required to compute the numeric
	/// Cholesky LDLT factorization of a matrix `A` with dimension `n`.
	pub fn factorize_supernodal_numeric_ldlt_scratch<I: Index, T: ComplexField>(symbolic: &SymbolicSupernodalCholesky<I>, par: Par) -> StackReq {
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let row_ind = &*symbolic.row_idx;

		let mut req = StackReq::empty();
		for s in 0..n_supernodes {
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();

			let s_ncols = s_end - s_start;

			let s_postordered = post_inv[s].zx();
			let desc_count = desc_count[s].zx();
			for d in &post[s_postordered - desc_count..s_postordered] {
				let mut d_scratch = StackReq::empty();

				let d = d.zx();
				let d_start = symbolic.supernode_begin[d].zx();
				let d_end = symbolic.supernode_begin[d + 1].zx();

				let d_pattern = &row_ind[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()];

				let d_ncols = d_end - d_start;

				let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
				let d_pattern_mid_len = d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));

				d_scratch = d_scratch.and(temp_mat_scratch::<T>(d_pattern.len() - d_pattern_start, d_pattern_mid_len));
				d_scratch = d_scratch.and(temp_mat_scratch::<T>(d_ncols, d_pattern_mid_len));
				req = req.or(d_scratch);
			}
			req = req.or(linalg::cholesky::ldlt::factor::cholesky_in_place_scratch::<T>(
				s_ncols,
				par,
				Default::default(),
			));
		}
		req.and(StackReq::new::<I>(n))
	}

	/// Returns the size and alignment of the workspace required to compute the numeric
	/// Cholesky Bunch-Kaufman factorization with intranodal pivoting of a matrix `A` with dimension
	/// `n`.
	pub fn factorize_supernodal_numeric_intranode_bunch_kaufman_scratch<I: Index, T: ComplexField>(
		symbolic: &SymbolicSupernodalCholesky<I>,
		par: Par,
	) -> StackReq {
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let row_ind = &*symbolic.row_idx;

		let mut req = StackReq::empty();
		for s in 0..n_supernodes {
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();

			let s_ncols = s_end - s_start;
			let s_pattern = &row_ind[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()];

			let s_postordered = post_inv[s].zx();
			let desc_count = desc_count[s].zx();
			for d in &post[s_postordered - desc_count..s_postordered] {
				let mut d_scratch = StackReq::empty();

				let d = d.zx();
				let d_start = symbolic.supernode_begin[d].zx();
				let d_end = symbolic.supernode_begin[d + 1].zx();

				let d_pattern = &row_ind[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()];

				let d_ncols = d_end - d_start;

				let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
				let d_pattern_mid_len = d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));

				d_scratch = d_scratch.and(temp_mat_scratch::<T>(d_pattern.len() - d_pattern_start, d_pattern_mid_len));
				d_scratch = d_scratch.and(temp_mat_scratch::<T>(d_ncols, d_pattern_mid_len));
				req = req.or(d_scratch);
			}
			req = StackReq::any_of(&[
				req,
				linalg::cholesky::bunch_kaufman::factor::cholesky_in_place_scratch::<I, T>(s_ncols, par, Default::default()),
				crate::perm::permute_cols_in_place_scratch::<I, T>(s_pattern.len(), s_ncols),
			]);
		}
		req.and(StackReq::new::<I>(n))
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
	#[math]
	pub fn factorize_supernodal_numeric_llt<I: Index, T: ComplexField>(
		L_values: &mut [T],
		A_lower: SparseColMatRef<'_, I, T>,
		regularization: LltRegularization<T>,
		symbolic: &SymbolicSupernodalCholesky<I>,
		par: Par,
		stack: &mut MemStack,
	) -> Result<LltInfo, LltError> {
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let mut dynamic_regularization_count = 0usize;
		L_values.fill(zero::<T>());

		assert!(A_lower.nrows() == n);
		assert!(A_lower.ncols() == n);
		assert!(L_values.len() == symbolic.len_values());

		let none = I::Signed::truncate(NONE);

		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let col_ptr_val = &*symbolic.col_ptr_for_val;
		let row_ind = &*symbolic.row_idx;

		// mapping from global indices to local
		let (global_to_local, mut stack) = unsafe { stack.make_raw::<I::Signed>(n) };
		global_to_local.fill(I::Signed::truncate(NONE));

		for s in 0..n_supernodes {
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();

			let s_pattern = &row_ind[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()];
			let s_ncols = s_end - s_start;
			let s_nrows = s_pattern.len() + s_ncols;

			for (i, &row) in s_pattern.iter().enumerate() {
				global_to_local[row.zx()] = I::Signed::truncate(i + s_ncols);
			}

			let (head, tail) = L_values.split_at_mut(col_ptr_val[s].zx());
			let head = head.rb();
			let mut Ls = MatMut::from_column_major_slice_mut(&mut tail[..col_ptr_val[s + 1].zx() - col_ptr_val[s].zx()], s_nrows, s_ncols);

			for j in s_start..s_end {
				let j_shifted = j - s_start;
				for (i, val) in iter::zip(A_lower.row_idx_of_col(j), A_lower.val_of_col(j)) {
					if i < j {
						continue;
					}

					let (ix, iy) = if i >= s_end {
						(global_to_local[i].sx(), j_shifted)
					} else {
						(i - s_start, j_shifted)
					};
					Ls[(ix, iy)] = Ls[(ix, iy)] + *val;
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

				let Ld = MatRef::from_column_major_slice(&head[col_ptr_val[d].zx()..col_ptr_val[d + 1].zx()], d_nrows, d_ncols);

				let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
				let d_pattern_mid_len = d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));
				let d_pattern_mid = d_pattern_start + d_pattern_mid_len;

				let (_, Ld_mid_bot) = Ld.split_at_row(d_ncols);
				let (_, Ld_mid_bot) = Ld_mid_bot.split_at_row(d_pattern_start);
				let (Ld_mid, Ld_bot) = Ld_mid_bot.split_at_row(d_pattern_mid_len);

				let stack = stack.rb_mut();

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(Ld_mid_bot.nrows(), d_pattern_mid_len, stack) };
				let tmp = tmp.as_mat_mut();

				let (mut tmp_top, mut tmp_bot) = tmp.split_at_row_mut(d_pattern_mid_len);

				use linalg::matmul;
				use linalg::matmul::triangular;
				triangular::matmul(
					tmp_top.rb_mut(),
					triangular::BlockStructure::TriangularLower,
					Accum::Replace,
					Ld_mid,
					triangular::BlockStructure::Rectangular,
					Ld_mid.rb().adjoint(),
					triangular::BlockStructure::Rectangular,
					one::<T>(),
					par,
				);
				matmul::matmul(tmp_bot.rb_mut(), Accum::Replace, Ld_bot, Ld_mid.rb().adjoint(), one::<T>(), par);
				for (j_idx, j) in d_pattern[d_pattern_start..d_pattern_mid].iter().enumerate() {
					let j = j.zx();
					let j_s = j - s_start;
					for (i_idx, i) in d_pattern[d_pattern_start..d_pattern_mid][j_idx..].iter().enumerate() {
						let i_idx = i_idx + j_idx;

						let i = i.zx();
						let i_s = i - s_start;

						debug_assert!(i_s >= j_s);

						Ls[(i_s, j_s)] = Ls[(i_s, j_s)] - tmp_top[(i_idx, j_idx)];
					}
				}

				for (j_idx, j) in d_pattern[d_pattern_start..d_pattern_mid].iter().enumerate() {
					let j = j.zx();
					let j_s = j - s_start;
					for (i_idx, i) in d_pattern[d_pattern_mid..].iter().enumerate() {
						let i = i.zx();
						let i_s = global_to_local[i].zx();
						Ls[(i_s, j_s)] = Ls[(i_s, j_s)] - tmp_bot[(i_idx, j_idx)];
					}
				}
			}

			let (mut Ls_top, mut Ls_bot) = Ls.rb_mut().split_at_row_mut(s_ncols);

			let params = Default::default();
			dynamic_regularization_count +=
				match linalg::cholesky::llt::factor::cholesky_in_place(Ls_top.rb_mut(), regularization.clone(), par, stack.rb_mut(), params) {
					Ok(count) => count,
					Err(LltError::NonPositivePivot { index }) => {
						return Err(LltError::NonPositivePivot { index: index + s_start });
					},
				}
				.dynamic_regularization_count;
			linalg::triangular_solve::solve_lower_triangular_in_place(Ls_top.rb().conjugate(), Ls_bot.rb_mut().transpose_mut(), par);

			for &row in s_pattern {
				global_to_local[row.zx()] = none;
			}
		}
		Ok(LltInfo {
			dynamic_regularization_count,
		})
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
	#[math]
	pub fn factorize_supernodal_numeric_ldlt<I: Index, T: ComplexField>(
		L_values: &mut [T],
		A_lower: SparseColMatRef<'_, I, T>,
		regularization: LdltRegularization<'_, T>,
		symbolic: &SymbolicSupernodalCholesky<I>,
		par: Par,
		stack: &mut MemStack,
	) -> Result<LdltInfo, LdltError> {
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let mut dynamic_regularization_count = 0usize;
		L_values.fill(zero());

		assert!(A_lower.nrows() == n);
		assert!(A_lower.ncols() == n);
		assert!(L_values.len() == symbolic.len_values());

		let none = I::Signed::truncate(NONE);

		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let col_ptr_val = &*symbolic.col_ptr_for_val;
		let row_ind = &*symbolic.row_idx;

		// mapping from global indices to local
		let (global_to_local, mut stack) = unsafe { stack.make_raw::<I::Signed>(n) };
		global_to_local.fill(I::Signed::truncate(NONE));

		for s in 0..n_supernodes {
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();
			let s_pattern = if let Some(nnz_per_super) = symbolic.nnz_per_super.as_deref() {
				&row_ind[col_ptr_row[s].zx()..][..nnz_per_super[s].zx()]
			} else {
				&row_ind[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()]
			};

			let s_ncols = s_end - s_start;
			let s_nrows = s_pattern.len() + s_ncols;

			for (i, &row) in s_pattern.iter().enumerate() {
				global_to_local[row.zx()] = I::Signed::truncate(i + s_ncols);
			}

			let (head, tail) = L_values.split_at_mut(col_ptr_val[s].zx());
			let head = head.rb();
			let mut Ls = MatMut::from_column_major_slice_mut(&mut tail[..col_ptr_val[s + 1].zx() - col_ptr_val[s].zx()], s_nrows, s_ncols);

			for j in s_start..s_end {
				let j_shifted = j - s_start;
				for (i, val) in iter::zip(A_lower.row_idx_of_col(j), A_lower.val_of_col(j)) {
					if i < j {
						continue;
					}

					let (ix, iy) = if i >= s_end {
						(global_to_local[i].sx(), j_shifted)
					} else {
						(i - s_start, j_shifted)
					};
					Ls.write(ix, iy, Ls.read(ix, iy) + *val);
				}
			}

			let s_postordered = post_inv[s].zx();
			let desc_count = desc_count[s].zx();
			for d in &post[s_postordered - desc_count..s_postordered] {
				let d = d.zx();
				let d_start = symbolic.supernode_begin[d].zx();
				let d_end = symbolic.supernode_begin[d + 1].zx();
				let d_pattern = if let Some(nnz_per_super) = symbolic.nnz_per_super.as_deref() {
					&row_ind[col_ptr_row[d].zx()..][..nnz_per_super[d].zx()]
				} else {
					&row_ind[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()]
				};

				let d_ncols = d_end - d_start;
				let d_nrows = d_pattern.len() + d_ncols;

				let Ld = MatRef::from_column_major_slice(&head[col_ptr_val[d].zx()..col_ptr_val[d + 1].zx()], d_nrows, d_ncols);

				let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
				let d_pattern_mid_len = d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));
				let d_pattern_mid = d_pattern_start + d_pattern_mid_len;

				let (Ld_top, Ld_mid_bot) = Ld.split_at_row(d_ncols);
				let (_, Ld_mid_bot) = Ld_mid_bot.split_at_row(d_pattern_start);
				let (Ld_mid, Ld_bot) = Ld_mid_bot.split_at_row(d_pattern_mid_len);
				let D = Ld_top.diagonal().column_vector();

				let stack = stack.rb_mut();

				let (mut tmp, stack) = unsafe { temp_mat_uninit::<T, _, _>(Ld_mid_bot.nrows(), d_pattern_mid_len, stack) };
				let tmp = tmp.as_mat_mut();
				let (mut tmp2, _) = unsafe { temp_mat_uninit::<T, _, _>(Ld_mid.ncols(), Ld_mid.nrows(), stack) };
				let tmp2 = tmp2.as_mat_mut();
				let mut Ld_mid_x_D = tmp2.transpose_mut();

				for i in 0..d_pattern_mid_len {
					for j in 0..d_ncols {
						Ld_mid_x_D.write(i, j, mul_real(Ld_mid.read(i, j), real(D.read(j))));
					}
				}

				let (mut tmp_top, mut tmp_bot) = tmp.split_at_row_mut(d_pattern_mid_len);

				use linalg::matmul;
				use linalg::matmul::triangular;
				triangular::matmul(
					tmp_top.rb_mut(),
					triangular::BlockStructure::TriangularLower,
					Accum::Replace,
					Ld_mid,
					triangular::BlockStructure::Rectangular,
					Ld_mid_x_D.rb().adjoint(),
					triangular::BlockStructure::Rectangular,
					one::<T>(),
					par,
				);
				matmul::matmul(tmp_bot.rb_mut(), Accum::Replace, Ld_bot, Ld_mid_x_D.rb().adjoint(), one::<T>(), par);
				for (j_idx, j) in d_pattern[d_pattern_start..d_pattern_mid].iter().enumerate() {
					let j = j.zx();
					let j_s = j - s_start;
					for (i_idx, i) in d_pattern[d_pattern_start..d_pattern_mid][j_idx..].iter().enumerate() {
						let i_idx = i_idx + j_idx;

						let i = i.zx();
						let i_s = i - s_start;

						debug_assert!(i_s >= j_s);

						Ls.write(i_s, j_s, Ls.read(i_s, j_s) - tmp_top.read(i_idx, j_idx));
					}
				}

				for (j_idx, j) in d_pattern[d_pattern_start..d_pattern_mid].iter().enumerate() {
					let j = j.zx();
					let j_s = j - s_start;
					for (i_idx, i) in d_pattern[d_pattern_mid..].iter().enumerate() {
						let i = i.zx();
						let i_s = global_to_local[i].zx();
						Ls.write(i_s, j_s, Ls.read(i_s, j_s) - tmp_bot.read(i_idx, j_idx));
					}
				}
			}

			let (mut Ls_top, mut Ls_bot) = Ls.rb_mut().split_at_row_mut(s_ncols);

			let params = Default::default();
			dynamic_regularization_count += match linalg::cholesky::ldlt::factor::cholesky_in_place(
				Ls_top.rb_mut(),
				LdltRegularization {
					dynamic_regularization_signs: regularization.dynamic_regularization_signs.map(|signs| &signs[s_start..s_end]),
					..regularization.clone()
				},
				par,
				stack.rb_mut(),
				params,
			) {
				Ok(count) => count.dynamic_regularization_count,
				Err(LdltError::ZeroPivot { index }) => {
					return Err(LdltError::ZeroPivot { index: index + s_start });
				},
			};
			z!(Ls_top.rb_mut()).for_each_triangular_upper(linalg::zip::Diag::Skip, |uz!(x)| *x = zero::<T>());
			linalg::triangular_solve::solve_unit_lower_triangular_in_place(Ls_top.rb().conjugate(), Ls_bot.rb_mut().transpose_mut(), par);
			for j in 0..s_ncols {
				let d = recip(real(Ls_top.read(j, j)));
				for i in 0..s_pattern.len() {
					Ls_bot.write(i, j, mul_real(Ls_bot.read(i, j), d));
				}
			}

			for &row in s_pattern {
				global_to_local[row.zx()] = none;
			}
		}
		Ok(LdltInfo {
			dynamic_regularization_count,
		})
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
	#[math]
	pub fn factorize_supernodal_numeric_intranode_bunch_kaufman<I: Index, T: ComplexField>(
		L_values: &mut [T],
		subdiag: &mut [T],
		perm_forward: &mut [I],
		perm_inverse: &mut [I],
		A_lower: SparseColMatRef<'_, I, T>,
		regularization: BunchKaufmanRegularization<'_, T>,
		symbolic: &SymbolicSupernodalCholesky<I>,
		par: Par,
		stack: &mut MemStack,
	) -> usize {
		let mut regularization = regularization;
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let mut dynamic_regularization_count = 0usize;
		L_values.fill(zero());

		assert!(A_lower.nrows() == n);
		assert!(A_lower.ncols() == n);
		assert!(perm_forward.len() == n);
		assert!(perm_inverse.len() == n);
		assert!(subdiag.len() == n);
		assert!(L_values.len() == symbolic.len_values());

		let none = I::Signed::truncate(NONE);

		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let col_ptr_val = &*symbolic.col_ptr_for_val;
		let row_ind = &*symbolic.row_idx;

		// mapping from global indices to local
		let (global_to_local, mut stack) = unsafe { stack.make_raw::<I::Signed>(n) };
		global_to_local.fill(I::Signed::truncate(NONE));

		for s in 0..n_supernodes {
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();

			let s_pattern = &row_ind[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()];
			let s_ncols = s_end - s_start;
			let s_nrows = s_pattern.len() + s_ncols;

			for (i, &row) in s_pattern.iter().enumerate() {
				global_to_local[row.zx()] = I::Signed::truncate(i + s_ncols);
			}

			let (head, tail) = L_values.split_at_mut(col_ptr_val[s].zx());
			let head = head.rb();
			let mut Ls = MatMut::from_column_major_slice_mut(&mut tail[..col_ptr_val[s + 1].zx() - col_ptr_val[s].zx()], s_nrows, s_ncols);

			for j in s_start..s_end {
				let j_shifted = j - s_start;
				for (i, val) in iter::zip(A_lower.row_idx_of_col(j), A_lower.val_of_col(j)) {
					if i < j {
						continue;
					}

					let (ix, iy) = if i >= s_end {
						(global_to_local[i].sx(), j_shifted)
					} else {
						(i - s_start, j_shifted)
					};
					Ls.write(ix, iy, Ls.read(ix, iy) + *val);
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

				let Ld = MatRef::from_column_major_slice(&head[col_ptr_val[d].zx()..col_ptr_val[d + 1].zx()], d_nrows, d_ncols);

				let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
				let d_pattern_mid_len = d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));
				let d_pattern_mid = d_pattern_start + d_pattern_mid_len;

				let (Ld_top, Ld_mid_bot) = Ld.split_at_row(d_ncols);
				let (_, Ld_mid_bot) = Ld_mid_bot.split_at_row(d_pattern_start);
				let (Ld_mid, Ld_bot) = Ld_mid_bot.split_at_row(d_pattern_mid_len);
				let d_subdiag = &subdiag[d_start..d_start + d_ncols];

				let stack = stack.rb_mut();

				let (mut tmp, stack) = unsafe { temp_mat_uninit::<T, _, _>(Ld_mid_bot.nrows(), d_pattern_mid_len, stack) };
				let (mut tmp2, _) = unsafe { temp_mat_uninit::<T, _, _>(Ld_mid.ncols(), Ld_mid.nrows(), stack) };
				let tmp = tmp.as_mat_mut();
				let mut Ld_mid_x_D = tmp2.as_mat_mut().transpose_mut();

				let mut j = 0;
				while j < d_ncols {
					let subdiag = copy(d_subdiag[j]);
					if subdiag == zero::<T>() {
						let d = real(Ld_top.read(j, j));
						for i in 0..d_pattern_mid_len {
							Ld_mid_x_D.write(i, j, mul_real(Ld_mid.read(i, j), d));
						}
						j += 1;
					} else {
						let akp1k = subdiag;
						let ak = real(Ld_top.read(j, j));
						let akp1 = real(Ld_top.read(j + 1, j + 1));

						for i in 0..d_pattern_mid_len {
							let xk = Ld_mid.read(i, j);
							let xkp1 = Ld_mid.read(i, j + 1);

							Ld_mid_x_D.write(i, j, mul_real(xk, ak) + xkp1 * akp1k);
							Ld_mid_x_D.write(i, j + 1, mul_real(xkp1, akp1) + xk * conj(akp1k));
						}
						j += 2;
					}
				}

				let (mut tmp_top, mut tmp_bot) = tmp.split_at_row_mut(d_pattern_mid_len);

				use linalg::matmul;
				use linalg::matmul::triangular;
				triangular::matmul(
					tmp_top.rb_mut(),
					triangular::BlockStructure::TriangularLower,
					Accum::Replace,
					Ld_mid,
					triangular::BlockStructure::Rectangular,
					Ld_mid_x_D.rb().adjoint(),
					triangular::BlockStructure::Rectangular,
					one::<T>(),
					par,
				);
				matmul::matmul(tmp_bot.rb_mut(), Accum::Replace, Ld_bot, Ld_mid_x_D.rb().adjoint(), one::<T>(), par);

				for (j_idx, j) in d_pattern[d_pattern_start..d_pattern_mid].iter().enumerate() {
					let j = j.zx();
					let j_s = j - s_start;
					for (i_idx, i) in d_pattern[d_pattern_start..d_pattern_mid][j_idx..].iter().enumerate() {
						let i_idx = i_idx + j_idx;

						let i = i.zx();
						let i_s = i - s_start;

						debug_assert!(i_s >= j_s);
						Ls.write(i_s, j_s, Ls.read(i_s, j_s) - tmp_top.read(i_idx, j_idx));
					}
				}

				for (j_idx, j) in d_pattern[d_pattern_start..d_pattern_mid].iter().enumerate() {
					let j = j.zx();
					let j_s = j - s_start;
					for (i_idx, i) in d_pattern[d_pattern_mid..].iter().enumerate() {
						let i = i.zx();
						let i_s = global_to_local[i].zx();
						Ls.write(i_s, j_s, Ls.read(i_s, j_s) - tmp_bot.read(i_idx, j_idx));
					}
				}
			}

			let (mut Ls_top, mut Ls_bot) = Ls.rb_mut().split_at_row_mut(s_ncols);
			let s_subdiag = &mut subdiag[s_start..s_end];

			let params = Default::default();
			let (info, perm) = linalg::cholesky::bunch_kaufman::factor::cholesky_in_place(
				Ls_top.rb_mut(),
				ColMut::from_slice_mut(s_subdiag).as_diagonal_mut(),
				BunchKaufmanRegularization {
					dynamic_regularization_signs: regularization
						.dynamic_regularization_signs
						.rb_mut()
						.map(|signs| &mut signs[s_start..s_end]),

					dynamic_regularization_delta: copy(regularization.dynamic_regularization_delta),
					dynamic_regularization_epsilon: copy(regularization.dynamic_regularization_epsilon),
				},
				&mut perm_forward[s_start..s_end],
				&mut perm_inverse[s_start..s_end],
				par,
				stack.rb_mut(),
				params,
			);
			dynamic_regularization_count += info.dynamic_regularization_count;
			z!(Ls_top.rb_mut()).for_each_triangular_upper(linalg::zip::Diag::Skip, |uz!(x)| *x = zero::<T>());

			crate::perm::permute_cols_in_place(Ls_bot.rb_mut(), perm.rb(), stack.rb_mut());

			for p in &mut perm_forward[s_start..s_end] {
				*p += I::truncate(s_start);
			}
			for p in &mut perm_inverse[s_start..s_end] {
				*p += I::truncate(s_start);
			}

			linalg::triangular_solve::solve_unit_lower_triangular_in_place(Ls_top.rb().conjugate(), Ls_bot.rb_mut().transpose_mut(), par);

			let mut j = 0;
			while j < s_ncols {
				if s_subdiag[j] == zero::<T>() {
					let d = recip(real(Ls_top.read(j, j)));
					for i in 0..s_pattern.len() {
						Ls_bot.write(i, j, mul_real(Ls_bot.read(i, j), d));
					}
					j += 1;
				} else {
					let akp1k = recip(conj(s_subdiag[j]));
					let ak = mul_real(conj(akp1k), real(Ls_top.read(j, j)));
					let akp1 = mul_real(akp1k, real(Ls_top.read(j + 1, j + 1)));

					let denom = recip(ak * akp1 - one::<T>());

					for i in 0..s_pattern.len() {
						let xk = Ls_bot.read(i, j) * conj(akp1k);
						let xkp1 = Ls_bot.read(i, j + 1) * akp1k;

						Ls_bot.write(i, j, (akp1 * xk - xkp1) * denom);
						Ls_bot.write(i, j + 1, (ak * xkp1 - xk) * denom);
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

pub(crate) fn ghost_postorder<'n, I: Index>(post: &mut Array<'n, I>, etree: &Array<'n, MaybeIdx<'n, I>>, stack: &mut MemStack) {
	let N = post.len();
	let n = *N;

	if n == 0 {
		return;
	}

	let (stack_, stack) = unsafe { stack.make_raw::<I>(n) };
	let (first_child, stack) = unsafe { stack.make_raw::<I::Signed>(n) };
	let (next_child, _) = unsafe { stack.make_raw::<I::Signed>(n) };

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
			start_index = postorder_depth_first_search(post, root, start_index, stack, first_child, next_child);
		}
	}
}

/// Tuning parameters for the symbolic Cholesky factorization.
#[derive(Copy, Clone, Debug, Default)]
pub struct CholeskySymbolicParams<'a> {
	/// Parameters for computing the fill-reducing permutation.
	pub amd_params: amd::Control,
	/// Threshold for selecting the supernodal factorization.
	pub supernodal_flop_ratio_threshold: SupernodalThreshold,
	/// Supernodal factorization parameters.
	pub supernodal_params: SymbolicSupernodalParams<'a>,
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
	perm_fwd: Option<alloc::vec::Vec<I>>,
	perm_inv: Option<alloc::vec::Vec<I>>,
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
	pub fn perm(&self) -> Option<PermRef<'_, I>> {
		match (&self.perm_fwd, &self.perm_inv) {
			(Some(perm_fwd), Some(perm_inv)) => unsafe { Some(PermRef::new_unchecked(perm_fwd, perm_inv, self.ncols())) },
			_ => None,
		}
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
	pub fn factorize_numeric_llt_scratch<T: ComplexField>(&self, par: Par) -> StackReq {
		let n = self.nrows();
		let A_nnz = self.A_nnz;

		let n_scratch = StackReq::new::<I>(n);
		let A_scratch = StackReq::all_of(&[temp_mat_scratch::<T>(A_nnz, 1), StackReq::new::<I>(n + 1), StackReq::new::<I>(A_nnz)]);
		let permute_scratch = n_scratch;

		let factor_scratch = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => simplicial::factorize_simplicial_numeric_llt_scratch::<I, T>(n),
			SymbolicCholeskyRaw::Supernodal(this) => supernodal::factorize_supernodal_numeric_llt_scratch::<I, T>(this, par),
		};

		StackReq::all_of(&[A_scratch, StackReq::or(permute_scratch, factor_scratch)])
	}

	/// Computes the required workspace size and alignment for a numerical LDLT factorization.
	#[inline]
	pub fn factorize_numeric_ldlt_scratch<T: ComplexField>(&self, with_regularization_signs: bool, par: Par) -> StackReq {
		let n = self.nrows();
		let A_nnz = self.A_nnz;

		let regularization_signs = if with_regularization_signs {
			StackReq::new::<i8>(n)
		} else {
			StackReq::empty()
		};

		let n_scratch = StackReq::new::<I>(n);
		let A_scratch = StackReq::all_of(&[temp_mat_scratch::<T>(A_nnz, 1), StackReq::new::<I>(n + 1), StackReq::new::<I>(A_nnz)]);
		let permute_scratch = n_scratch;

		let factor_scratch = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => simplicial::factorize_simplicial_numeric_ldlt_scratch::<I, T>(n),
			SymbolicCholeskyRaw::Supernodal(this) => supernodal::factorize_supernodal_numeric_ldlt_scratch::<I, T>(this, par),
		};

		StackReq::all_of(&[regularization_signs, A_scratch, StackReq::or(permute_scratch, factor_scratch)])
	}

	/// Computes the required workspace size and alignment for a numerical intranodal Bunch-Kaufman
	/// factorization.
	#[inline]
	pub fn factorize_numeric_intranode_bunch_kaufman_scratch<T: ComplexField>(&self, with_regularization_signs: bool, par: Par) -> StackReq {
		let n = self.nrows();
		let A_nnz = self.A_nnz;

		let regularization_signs = if with_regularization_signs {
			StackReq::new::<i8>(n)
		} else {
			StackReq::empty()
		};

		let n_scratch = StackReq::new::<I>(n);
		let A_scratch = StackReq::all_of(&[temp_mat_scratch::<T>(A_nnz, 1), StackReq::new::<I>(n + 1), StackReq::new::<I>(A_nnz)]);
		let permute_scratch = n_scratch;

		let factor_scratch = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => simplicial::factorize_simplicial_numeric_ldlt_scratch::<I, T>(n),
			SymbolicCholeskyRaw::Supernodal(this) => supernodal::factorize_supernodal_numeric_intranode_bunch_kaufman_scratch::<I, T>(this, par),
		};

		StackReq::all_of(&[regularization_signs, A_scratch, StackReq::or(permute_scratch, factor_scratch)])
	}

	/// Computes a numerical LLT factorization of A, or returns a [`LltError`] if the matrix
	/// is not numerically positive definite.
	#[track_caller]
	pub fn factorize_numeric_llt<'out, T: ComplexField>(
		&'out self,
		L_values: &'out mut [T],
		A: SparseColMatRef<'_, I, T>,
		side: Side,
		regularization: LltRegularization<T>,
		par: Par,
		stack: &mut MemStack,
	) -> Result<LltRef<'out, I, T>, LltError> {
		assert!(A.nrows() == A.ncols());
		let n = A.nrows();
		with_dim!(N, n);

		let A_nnz = self.A_nnz;
		let A = A.as_shape(N, N);

		let (mut new_values, stack) = unsafe { temp_mat_uninit::<T, _, _>(A_nnz, 1, stack) };
		let new_values = new_values.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap().as_slice_mut();
		let (new_col_ptr, stack) = unsafe { stack.make_raw::<I>(n + 1) };
		let (new_row_ind, mut stack) = unsafe { stack.make_raw::<I>(A_nnz) };

		let out_side = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => Side::Upper,
			SymbolicCholeskyRaw::Supernodal(_) => Side::Lower,
		};

		let A = match self.perm() {
			Some(perm) => {
				let perm = perm.as_shape(N);
				permute_self_adjoint_to_unsorted(new_values, new_col_ptr, new_row_ind, A, perm, side, out_side, stack.rb_mut()).into_const()
			},
			None => {
				if side == out_side {
					A
				} else {
					adjoint(new_values, new_col_ptr, new_row_ind, A, stack.rb_mut()).into_const()
				}
			},
		};

		match &self.raw {
			SymbolicCholeskyRaw::Simplicial(this) => {
				simplicial::factorize_simplicial_numeric_llt(L_values, A.as_dyn().into_const(), regularization, this, stack)?;
			},
			SymbolicCholeskyRaw::Supernodal(this) => {
				supernodal::factorize_supernodal_numeric_llt(L_values, A.as_dyn().into_const(), regularization, this, par, stack)?;
			},
		}
		Ok(LltRef::<'out, I, T>::new(self, L_values))
	}

	/// Computes a numerical LDLT factorization of A.
	#[inline]
	pub fn factorize_numeric_ldlt<'out, T: ComplexField>(
		&'out self,
		L_values: &'out mut [T],
		A: SparseColMatRef<'_, I, T>,
		side: Side,
		regularization: LdltRegularization<'_, T>,
		par: Par,
		stack: &mut MemStack,
	) -> Result<LdltRef<'out, I, T>, LdltError> {
		assert!(A.nrows() == A.ncols());
		let n = A.nrows();

		with_dim!(N, n);
		let A_nnz = self.A_nnz;
		let A = A.as_shape(N, N);

		let (new_signs, stack) = unsafe {
			stack.make_raw::<i8>(if regularization.dynamic_regularization_signs.is_some() && self.perm().is_some() {
				n
			} else {
				0
			})
		};

		let (mut new_values, stack) = unsafe { temp_mat_uninit::<T, _, _>(A_nnz, 1, stack) };
		let new_values = new_values.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap().as_slice_mut();
		let (new_col_ptr, stack) = unsafe { stack.make_raw::<I>(n + 1) };
		let (new_row_ind, mut stack) = unsafe { stack.make_raw::<I>(A_nnz) };

		let out_side = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => Side::Upper,
			SymbolicCholeskyRaw::Supernodal(_) => Side::Lower,
		};

		let (A, signs) = match self.perm() {
			Some(perm) => {
				let perm = perm.as_shape(N);
				let A = permute_self_adjoint_to_unsorted(new_values, new_col_ptr, new_row_ind, A, perm, side, out_side, stack.rb_mut()).into_const();
				let fwd = perm.bound_arrays().0;
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

				(A, signs)
			},
			None => {
				if side == out_side {
					(A, regularization.dynamic_regularization_signs)
				} else {
					(
						adjoint(new_values, new_col_ptr, new_row_ind, A, stack.rb_mut()).into_const(),
						regularization.dynamic_regularization_signs,
					)
				}
			},
		};

		let regularization = LdltRegularization {
			dynamic_regularization_signs: signs,
			..regularization
		};

		match &self.raw {
			SymbolicCholeskyRaw::Simplicial(this) => {
				simplicial::factorize_simplicial_numeric_ldlt(L_values, A.as_dyn().into_const(), regularization, this, stack)?;
			},
			SymbolicCholeskyRaw::Supernodal(this) => {
				supernodal::factorize_supernodal_numeric_ldlt(L_values, A.as_dyn().into_const(), regularization, this, par, stack)?;
			},
		}

		Ok(LdltRef::<'out, I, T>::new(self, L_values))
	}

	/// Computes a numerical intranodal Bunch-Kaufman factorization of A.
	#[inline]
	pub fn factorize_numeric_intranode_bunch_kaufman<'out, T: ComplexField>(
		&'out self,
		L_values: &'out mut [T],
		subdiag: &'out mut [T],
		perm_forward: &'out mut [I],
		perm_inverse: &'out mut [I],
		A: SparseColMatRef<'_, I, T>,
		side: Side,
		regularization: BunchKaufmanRegularization<'_, T>,
		par: Par,
		stack: &mut MemStack,
	) -> IntranodeBunchKaufmanRef<'out, I, T> {
		assert!(A.nrows() == A.ncols());
		let n = A.nrows();

		with_dim!(N, n);
		let A_nnz = self.A_nnz;
		let A = A.as_shape(N, N);

		let (new_signs, stack) = unsafe { stack.make_raw::<i8>(if regularization.dynamic_regularization_signs.is_some() { n } else { 0 }) };

		let (mut new_values, stack) = unsafe { temp_mat_uninit::<T, _, _>(A_nnz, 1, stack) };
		let new_values = new_values.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap().as_slice_mut();
		let (new_col_ptr, stack) = unsafe { stack.make_raw::<I>(n + 1) };
		let (new_row_ind, mut stack) = unsafe { stack.make_raw::<I>(A_nnz) };

		let out_side = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => Side::Upper,
			SymbolicCholeskyRaw::Supernodal(_) => Side::Lower,
		};

		let (A, signs) = match self.perm() {
			Some(perm) => {
				let perm = perm.as_shape(N);
				let fwd = perm.bound_arrays().0;
				let signs = regularization.dynamic_regularization_signs.map(|signs| {
					{
						let new_signs = Array::from_mut(new_signs, N);
						let signs = Array::from_ref(signs, N);
						for i in N.indices() {
							new_signs[i] = signs[fwd[i].zx()];
						}
					}
					&mut *new_signs
				});

				let A = permute_self_adjoint_to_unsorted(new_values, new_col_ptr, new_row_ind, A, perm, side, out_side, stack.rb_mut()).into_const();

				(A, signs)
			},
			None => {
				if side == out_side {
					(A, regularization.dynamic_regularization_signs)
				} else {
					(
						adjoint(new_values, new_col_ptr, new_row_ind, A, stack.rb_mut()).into_const(),
						regularization.dynamic_regularization_signs,
					)
				}
			},
		};

		match &self.raw {
			SymbolicCholeskyRaw::Simplicial(this) => {
				let regularization = LdltRegularization {
					dynamic_regularization_signs: signs.rb(),
					dynamic_regularization_delta: regularization.dynamic_regularization_delta,
					dynamic_regularization_epsilon: regularization.dynamic_regularization_epsilon,
				};
				for (i, p) in perm_forward.iter_mut().enumerate() {
					*p = I::truncate(i);
				}
				for (i, p) in perm_inverse.iter_mut().enumerate() {
					*p = I::truncate(i);
				}
				let _ = simplicial::factorize_simplicial_numeric_ldlt(L_values, A.as_dyn().into_const(), regularization, this, stack);
			},
			SymbolicCholeskyRaw::Supernodal(this) => {
				let regularization = BunchKaufmanRegularization {
					dynamic_regularization_signs: signs,
					dynamic_regularization_delta: regularization.dynamic_regularization_delta,
					dynamic_regularization_epsilon: regularization.dynamic_regularization_epsilon,
				};

				supernodal::factorize_supernodal_numeric_intranode_bunch_kaufman(
					L_values,
					subdiag,
					perm_forward,
					perm_inverse,
					A.as_dyn().into_const(),
					regularization,
					this,
					par,
					stack,
				);
			},
		}

		IntranodeBunchKaufmanRef::<'out, I, T>::new(self, L_values, subdiag, unsafe {
			PermRef::<'out, I>::new_unchecked(perm_forward, perm_inverse, n)
		})
	}

	/// Computes the required workspace size and alignment for a dense solve in place using an LLT,
	/// LDLT or intranodal Bunch-Kaufman factorization.
	pub fn solve_in_place_scratch<T: ComplexField>(&self, rhs_ncols: usize) -> StackReq {
		temp_mat_scratch::<T>(self.nrows(), rhs_ncols).and(match self.raw() {
			SymbolicCholeskyRaw::Simplicial(this) => this.solve_in_place_scratch::<T>(rhs_ncols),
			SymbolicCholeskyRaw::Supernodal(this) => this.solve_in_place_scratch::<T>(rhs_ncols),
		})
	}
}

/// Sparse LLT factorization wrapper.
#[derive(Debug)]
pub struct LltRef<'a, I: Index, T> {
	symbolic: &'a SymbolicCholesky<I>,
	values: &'a [T],
}

/// Sparse LDLT factorization wrapper.
#[derive(Debug)]
pub struct LdltRef<'a, I: Index, T> {
	symbolic: &'a SymbolicCholesky<I>,
	values: &'a [T],
}

/// Sparse intranodal Bunch-Kaufman factorization wrapper.
#[derive(Debug)]
pub struct IntranodeBunchKaufmanRef<'a, I: Index, T> {
	symbolic: &'a SymbolicCholesky<I>,
	values: &'a [T],
	subdiag: &'a [T],
	perm: PermRef<'a, I>,
}

impl<'a, I: Index, T> core::ops::Deref for LltRef<'a, I, T> {
	type Target = SymbolicCholesky<I>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		self.symbolic
	}
}
impl<'a, I: Index, T> core::ops::Deref for LdltRef<'a, I, T> {
	type Target = SymbolicCholesky<I>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		self.symbolic
	}
}
impl<'a, I: Index, T> core::ops::Deref for IntranodeBunchKaufmanRef<'a, I, T> {
	type Target = SymbolicCholesky<I>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		self.symbolic
	}
}

impl<'a, I: Index, T> Copy for LltRef<'a, I, T> {}
impl<'a, I: Index, T> Copy for LdltRef<'a, I, T> {}
impl<'a, I: Index, T> Copy for IntranodeBunchKaufmanRef<'a, I, T> {}

impl<'a, I: Index, T> Clone for LltRef<'a, I, T> {
	fn clone(&self) -> Self {
		*self
	}
}
impl<'a, I: Index, T> Clone for LdltRef<'a, I, T> {
	fn clone(&self) -> Self {
		*self
	}
}
impl<'a, I: Index, T> Clone for IntranodeBunchKaufmanRef<'a, I, T> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<'a, I: Index, T> IntranodeBunchKaufmanRef<'a, I, T> {
	/// Creates a new Cholesky intranodal Bunch-Kaufman factor from the symbolic part and
	/// numerical values, as well as the pivoting permutation.
	///
	/// # Panics
	/// - Panics if `values.len() != symbolic.len_values()`.
	/// - Panics if `subdiag.len() != symbolic.nrows()`.
	/// - Panics if `perm.len() != symbolic.nrows()`.
	#[inline]
	pub fn new(symbolic: &'a SymbolicCholesky<I>, values: &'a [T], subdiag: &'a [T], perm: PermRef<'a, I>) -> Self {
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

	/// Solves the equation $\text{Op}(A) x = \text{rhs}$ and stores the result in `rhs`, where
	/// $\text{Op}$ is either the identity or the conjugate, depending on the value of `conj`.
	///
	/// # Panics
	/// Panics if `rhs.nrows() != self.symbolic().nrows()`.
	pub fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
	where
		T: ComplexField,
	{
		let k = rhs.ncols();
		let n = self.symbolic.nrows();

		let mut rhs = rhs;

		let (mut x, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
		let mut x = x.as_mat_mut();

		match self.symbolic.raw() {
			SymbolicCholeskyRaw::Simplicial(symbolic) => {
				let this = simplicial::SimplicialLdltRef::new(symbolic, self.values);

				if let Some(perm) = self.symbolic.perm() {
					for j in 0..k {
						for (i, fwd) in perm.arrays().0.iter().enumerate() {
							x.write(i, j, rhs.read(fwd.zx(), j));
						}
					}
				}
				this.solve_in_place_with_conj(conj, if self.symbolic.perm().is_some() { x.rb_mut() } else { rhs.rb_mut() }, par, stack);
				if let Some(perm) = self.symbolic.perm() {
					for j in 0..k {
						for (i, inv) in perm.arrays().1.iter().enumerate() {
							rhs.write(i, j, x.read(inv.zx(), j));
						}
					}
				}
			},
			SymbolicCholeskyRaw::Supernodal(symbolic) => {
				let (dyn_fwd, dyn_inv) = self.perm.arrays();
				let (fwd, inv) = match self.symbolic.perm() {
					Some(perm) => {
						let (fwd, inv) = perm.arrays();
						(Some(fwd), Some(inv))
					},
					None => (None, None),
				};

				if let Some(fwd) = fwd {
					for j in 0..k {
						for (i, dyn_fwd) in dyn_fwd.iter().enumerate() {
							x.write(i, j, rhs.read(fwd[dyn_fwd.zx()].zx(), j));
						}
					}
				} else {
					for j in 0..k {
						for (i, dyn_fwd) in dyn_fwd.iter().enumerate() {
							x.write(i, j, rhs.read(dyn_fwd.zx(), j));
						}
					}
				}

				let this = supernodal::SupernodalIntranodeBunchKaufmanRef::new(symbolic, self.values, self.subdiag, self.perm);
				this.solve_in_place_no_numeric_permute_with_conj(conj, x.rb_mut(), par, stack);

				if let Some(inv) = inv {
					for j in 0..k {
						for (i, inv) in inv.iter().enumerate() {
							rhs.write(i, j, x.read(dyn_inv[inv.zx()].zx(), j));
						}
					}
				} else {
					for j in 0..k {
						for (i, dyn_inv) in dyn_inv.iter().enumerate() {
							rhs.write(i, j, x.read(dyn_inv.zx(), j));
						}
					}
				}
			},
		}
	}
}

impl<'a, I: Index, T> LltRef<'a, I, T> {
	/// Creates a new Cholesky LLT factor from the symbolic part and
	/// numerical values.
	///
	/// # Panics
	/// - Panics if `values.len() != symbolic.len_values()`.
	#[inline]
	pub fn new(symbolic: &'a SymbolicCholesky<I>, values: &'a [T]) -> Self {
		assert!(symbolic.len_values() == values.len());
		Self { symbolic, values }
	}

	/// Returns the symbolic part of the Cholesky factor.
	#[inline]
	pub fn symbolic(self) -> &'a SymbolicCholesky<I> {
		self.symbolic
	}

	/// Solves the equation $\text{Op}(A) x = \text{rhs}$ and stores the result in `rhs`, where
	/// $\text{Op}$ is either the identity or the conjugate, depending on the value of `conj`.
	///
	/// # Panics
	/// Panics if `rhs.nrows() != self.symbolic().nrows()`.
	pub fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
	where
		T: ComplexField,
	{
		let k = rhs.ncols();
		let n = self.symbolic.nrows();

		let mut rhs = rhs;

		let (mut x, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
		let mut x = x.as_mat_mut();

		if let Some(perm) = self.symbolic.perm() {
			for j in 0..k {
				for (i, fwd) in perm.arrays().0.iter().enumerate() {
					x.write(i, j, rhs.read(fwd.zx(), j));
				}
			}
		}

		match self.symbolic.raw() {
			SymbolicCholeskyRaw::Simplicial(symbolic) => {
				let this = simplicial::SimplicialLltRef::new(symbolic, self.values);
				this.solve_in_place_with_conj(conj, if self.symbolic.perm().is_some() { x.rb_mut() } else { rhs.rb_mut() }, par, stack);
			},
			SymbolicCholeskyRaw::Supernodal(symbolic) => {
				let this = supernodal::SupernodalLltRef::new(symbolic, self.values);
				this.solve_in_place_with_conj(conj, if self.symbolic.perm().is_some() { x.rb_mut() } else { rhs.rb_mut() }, par, stack);
			},
		}

		if let Some(perm) = self.symbolic.perm() {
			for j in 0..k {
				for (i, inv) in perm.arrays().1.iter().enumerate() {
					rhs.write(i, j, x.read(inv.zx(), j));
				}
			}
		}
	}
}

impl<'a, I: Index, T> LdltRef<'a, I, T> {
	/// Creates new Cholesky LDLT factors from the symbolic part and
	/// numerical values.
	///
	/// # Panics
	/// - Panics if `values.len() != symbolic.len_values()`.
	#[inline]
	pub fn new(symbolic: &'a SymbolicCholesky<I>, values: &'a [T]) -> Self {
		assert!(symbolic.len_values() == values.len());
		Self { symbolic, values }
	}

	/// Returns the symbolic part of the Cholesky factor.
	#[inline]
	pub fn symbolic(self) -> &'a SymbolicCholesky<I> {
		self.symbolic
	}

	/// Solves the equation $\text{Op}(A) x = \text{rhs}$ and stores the result in `rhs`, where
	/// $\text{Op}$ is either the identity or the conjugate, depending on the value of `conj`.
	///
	/// # Panics
	/// Panics if `rhs.nrows() != self.symbolic().nrows()`.
	pub fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
	where
		T: ComplexField,
	{
		let k = rhs.ncols();
		let n = self.symbolic.nrows();

		let mut rhs = rhs;

		let (mut x, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack) };
		let mut x = x.as_mat_mut();

		if let Some(perm) = self.symbolic.perm() {
			for j in 0..k {
				for (i, fwd) in perm.arrays().0.iter().enumerate() {
					x.write(i, j, rhs.read(fwd.zx(), j));
				}
			}
		}

		match self.symbolic.raw() {
			SymbolicCholeskyRaw::Simplicial(symbolic) => {
				let this = simplicial::SimplicialLdltRef::new(symbolic, self.values);
				this.solve_in_place_with_conj(conj, if self.symbolic.perm().is_some() { x.rb_mut() } else { rhs.rb_mut() }, par, stack);
			},
			SymbolicCholeskyRaw::Supernodal(symbolic) => {
				let this = supernodal::SupernodalLdltRef::new(symbolic, self.values);
				this.solve_in_place_with_conj(conj, if self.symbolic.perm().is_some() { x.rb_mut() } else { rhs.rb_mut() }, par, stack);
			},
		}

		if let Some(perm) = self.symbolic.perm() {
			for j in 0..k {
				for (i, inv) in perm.arrays().1.iter().enumerate() {
					rhs.write(i, j, x.read(inv.zx(), j));
				}
			}
		}
	}
}
