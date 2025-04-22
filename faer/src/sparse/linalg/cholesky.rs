//! computes the Cholesky decomposition (either $LL^\top$, $LTL^\top$, or $LBL^\top$) of a given
//! sparse matrix. see [`crate::linalg::cholesky`] for more info.
//!
//! the entry point in this module is [`SymbolicCholesky`] and [`factorize_symbolic_cholesky`].
//!
//! # note
//! the functions in this module accept unsorted input, producing a sorted decomposition factor
//! (simplicial).
//!
//! # example (low level api)
//! simplicial:
//! ```
//! fn simplicial_cholesky() -> Result<(), faer::sparse::FaerError> {
//! 	use faer::Par;
//! 	use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
//! 	use faer::reborrow::*;
//!
//! 	use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
//! 	use faer::linalg::cholesky::llt::factor::LltRegularization;
//!
//! 	use faer::sparse::linalg::amd;
//! 	use faer::sparse::linalg::cholesky::simplicial;
//! 	use faer::sparse::{CreationError, SparseColMat, SymbolicSparseColMat, Triplet};
//!
//! 	use rand::prelude::*;
//!
//! 	// the simplicial cholesky api takes an upper triangular matrix as input, to be
//! 	// interpreted as self-adjoint.
//! 	let dim = 4;
//! 	let A_upper = match SparseColMat::<usize, f64>::try_new_from_triplets(
//! 		dim,
//! 		dim,
//! 		&[
//! 			// diagonal entries
//! 			Triplet::new(0, 0, 10.0),
//! 			Triplet::new(1, 1, 11.0),
//! 			Triplet::new(2, 2, 12.0),
//! 			Triplet::new(3, 3, 13.0),
//! 			// non diagonal entries
//! 			Triplet::new(0, 1, 1.0),
//! 			Triplet::new(0, 3, 1.5),
//! 			Triplet::new(1, 3, -3.2),
//! 		],
//! 	) {
//! 		Ok(A) => Ok(A),
//! 		Err(CreationError::Generic(err)) => Err(err),
//! 		Err(CreationError::OutOfBounds { .. }) => panic!(),
//! 	}?;
//!
//! 	let mut A = faer::sparse::ops::add(A_upper.rb(), A_upper.to_row_major()?.rb().transpose())?;
//! 	for i in 0..dim {
//! 		A[(i, i)] /= 2.0;
//! 	}
//!
//! 	let A_nnz = A_upper.compute_nnz();
//! 	let mut rhs = Vec::new();
//! 	let mut rng = StdRng::seed_from_u64(0);
//! 	rhs.try_reserve_exact(dim)?;
//! 	rhs.resize_with(dim, || rng.gen::<f64>());
//!
//! 	let mut sol = Vec::new();
//! 	sol.try_reserve_exact(dim)?;
//! 	sol.resize(dim, 0.0f64);
//!
//! 	let rhs = faer::MatRef::from_column_major_slice(&rhs, dim, 1);
//! 	let mut sol = faer::MatMut::from_column_major_slice_mut(&mut sol, dim, 1);
//!
//! 	// optional: fill reducing permutation
//! 	let (perm, perm_inv) = {
//! 		let mut perm = Vec::new();
//! 		let mut perm_inv = Vec::new();
//! 		perm.try_reserve_exact(dim)?;
//! 		perm_inv.try_reserve_exact(dim)?;
//! 		perm.resize(dim, 0usize);
//! 		perm_inv.resize(dim, 0usize);
//!
//! 		let mut mem = MemBuffer::try_new(amd::order_scratch::<usize>(dim, A_nnz))?;
//! 		amd::order(
//! 			&mut perm,
//! 			&mut perm_inv,
//! 			A_upper.symbolic(),
//! 			amd::Control::default(),
//! 			MemStack::new(&mut mem),
//! 		)?;
//!
//! 		(perm, perm_inv)
//! 	};
//!
//! 	let perm = unsafe { faer::perm::PermRef::new_unchecked(&perm, &perm_inv, dim) };
//!
//! 	let A_perm_upper = {
//! 		let mut A_perm_col_ptrs = Vec::new();
//! 		let mut A_perm_row_indices = Vec::new();
//! 		let mut A_perm_values = Vec::new();
//!
//! 		A_perm_col_ptrs.try_reserve_exact(dim + 1)?;
//! 		A_perm_col_ptrs.resize(dim + 1, 0usize);
//! 		A_perm_row_indices.try_reserve_exact(A_nnz)?;
//! 		A_perm_row_indices.resize(A_nnz, 0usize);
//! 		A_perm_values.try_reserve_exact(A_nnz)?;
//! 		A_perm_values.resize(A_nnz, 0.0f64);
//!
//! 		let mut mem = MemBuffer::try_new(faer::sparse::utils::permute_self_adjoint_scratch::<
//! 			usize,
//! 		>(dim))?;
//! 		faer::sparse::utils::permute_self_adjoint_to_unsorted(
//! 			&mut A_perm_values,
//! 			&mut A_perm_col_ptrs,
//! 			&mut A_perm_row_indices,
//! 			A_upper.rb(),
//! 			perm,
//! 			faer::Side::Upper,
//! 			faer::Side::Upper,
//! 			MemStack::new(&mut mem),
//! 		);
//!
//! 		SparseColMat::<usize, f64>::new(
//! 			unsafe {
//! 				SymbolicSparseColMat::new_unchecked(
//! 					dim,
//! 					dim,
//! 					A_perm_col_ptrs,
//! 					None,
//! 					A_perm_row_indices,
//! 				)
//! 			},
//! 			A_perm_values,
//! 		)
//! 	};
//!
//! 	// symbolic analysis
//! 	let symbolic = {
//! 		let mut mem = MemBuffer::try_new(StackReq::any_of(&[
//! 			simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(dim, A_nnz),
//! 			simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(dim),
//! 		]))?;
//! 		let stack = MemStack::new(&mut mem);
//!
//! 		let mut etree = Vec::new();
//! 		let mut col_counts = Vec::new();
//! 		etree.try_reserve_exact(dim)?;
//! 		etree.resize(dim, 0isize);
//! 		col_counts.try_reserve_exact(dim)?;
//! 		col_counts.resize(dim, 0usize);
//!
//! 		simplicial::prefactorize_symbolic_cholesky(
//! 			&mut etree,
//! 			&mut col_counts,
//! 			A_perm_upper.symbolic(),
//! 			stack,
//! 		);
//! 		simplicial::factorize_simplicial_symbolic_cholesky(
//! 			A_perm_upper.symbolic(),
//! 			// SAFETY: `etree` was filled correctly by
//! 			// `simplicial::prefactorize_symbolic_cholesky`.
//! 			unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
//! 			&col_counts,
//! 			stack,
//! 		)?
//! 	};
//!
//! 	// numerical factorization
//! 	let mut mem = MemBuffer::try_new(StackReq::all_of(&[
//! 		simplicial::factorize_simplicial_numeric_llt_scratch::<usize, f64>(dim),
//! 		simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, f64>(dim),
//! 		faer::perm::permute_rows_in_place_scratch::<usize, f64>(dim, 1),
//! 		symbolic.solve_in_place_scratch::<f64>(dim),
//! 	]))?;
//! 	let stack = MemStack::new(&mut mem);
//!
//! 	// numerical llt factorization
//! 	{
//! 		let mut L_values = Vec::new();
//! 		L_values.try_reserve_exact(symbolic.len_val())?;
//! 		L_values.resize(symbolic.len_val(), 0.0f64);
//!
//! 		match simplicial::factorize_simplicial_numeric_llt::<usize, f64>(
//! 			&mut L_values,
//! 			A_perm_upper.rb(),
//! 			LltRegularization::default(),
//! 			&symbolic,
//! 			stack,
//! 		) {
//! 			Ok(_) => {},
//! 			Err(err) => panic!("matrix is not positive definite: {err}"),
//! 		};
//!
//! 		let llt = simplicial::SimplicialLltRef::<'_, usize, f64>::new(&symbolic, &L_values);
//!
//! 		sol.copy_from(rhs);
//! 		faer::perm::permute_rows_in_place(sol.rb_mut(), perm, stack);
//! 		llt.solve_in_place_with_conj(faer::Conj::No, sol.rb_mut(), faer::Par::Seq, stack);
//! 		faer::perm::permute_rows_in_place(sol.rb_mut(), perm.inverse(), stack);
//!
//! 		assert!((&A * &sol - &rhs).norm_max() <= 1e-14);
//! 	}
//!
//! 	// numerical ldlt factorization
//! 	{
//! 		let mut L_values = Vec::new();
//! 		L_values.try_reserve_exact(symbolic.len_val())?;
//! 		L_values.resize(symbolic.len_val(), 0.0f64);
//!
//! 		simplicial::factorize_simplicial_numeric_ldlt::<usize, f64>(
//! 			&mut L_values,
//! 			A_perm_upper.rb(),
//! 			LdltRegularization::default(),
//! 			&symbolic,
//! 			stack,
//! 		);
//!
//! 		let ldlt = simplicial::SimplicialLdltRef::<'_, usize, f64>::new(&symbolic, &L_values);
//!
//! 		sol.copy_from(rhs);
//! 		faer::perm::permute_rows_in_place(sol.rb_mut(), perm, stack);
//! 		ldlt.solve_in_place_with_conj(faer::Conj::No, sol.rb_mut(), faer::Par::Seq, stack);
//! 		faer::perm::permute_rows_in_place(sol.rb_mut(), perm.inverse(), stack);
//!
//! 		assert!((&A * &sol - &rhs).norm_max() <= 1e-14);
//! 	}
//! 	Ok(())
//! }
//! simplicial_cholesky().unwrap()
//! ```
//! supernodal:
//! ```
//! fn supernodal_cholesky() -> Result<(), faer::sparse::FaerError> {
//! 	use faer::Par;
//! 	use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
//! 	use faer::reborrow::*;
//!
//! 	use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
//!
//! 	use faer::sparse::linalg::amd;
//! 	use faer::sparse::linalg::cholesky::{simplicial, supernodal};
//! 	use faer::sparse::{CreationError, SparseColMat, SymbolicSparseColMat, Triplet};
//! 	use rand::prelude::*;
//!
//! 	// the supernodal cholesky api takes a lower triangular matrix as input, to be
//! 	// interpreted as self-adjoint.
//! 	let dim = 8;
//! 	let A_lower = match SparseColMat::<usize, f64>::try_new_from_triplets(
//! 		dim,
//! 		dim,
//! 		&[
//! 			// diagonal entries
//! 			Triplet::new(0, 0, 11.0),
//! 			Triplet::new(1, 1, 11.0),
//! 			Triplet::new(2, 2, 12.0),
//! 			Triplet::new(3, 3, 13.0),
//! 			Triplet::new(4, 4, 14.0),
//! 			Triplet::new(5, 5, 16.0),
//! 			Triplet::new(6, 6, 16.0),
//! 			Triplet::new(7, 7, 16.0),
//! 			// non diagonal entries
//! 			Triplet::new(1, 0, 10.0),
//! 			Triplet::new(3, 0, 10.5),
//! 			Triplet::new(4, 0, 10.0),
//! 			Triplet::new(7, 0, 10.5),
//! 			Triplet::new(3, 1, 10.5),
//! 			Triplet::new(4, 1, 10.0),
//! 			Triplet::new(7, 1, 10.5),
//! 			Triplet::new(3, 2, 10.5),
//! 			Triplet::new(4, 2, 10.0),
//! 			Triplet::new(7, 2, 10.0),
//! 		],
//! 	) {
//! 		Ok(A) => Ok(A),
//! 		Err(CreationError::Generic(err)) => Err(err),
//! 		Err(CreationError::OutOfBounds { .. }) => panic!(),
//! 	}?;
//!
//! 	let mut A = faer::sparse::ops::add(A_lower.rb(), A_lower.to_row_major()?.rb().transpose())?;
//! 	for i in 0..dim {
//! 		A[(i, i)] /= 2.0;
//! 	}
//!
//! 	let A_nnz = A_lower.compute_nnz();
//! 	let mut rhs = Vec::new();
//! 	let mut rng = StdRng::seed_from_u64(0);
//! 	rhs.try_reserve_exact(dim)?;
//! 	rhs.resize_with(dim, || rng.gen::<f64>());
//!
//! 	let mut sol = Vec::new();
//! 	sol.try_reserve_exact(dim)?;
//! 	sol.resize(dim, 0.0f64);
//!
//! 	let rhs = faer::MatRef::from_column_major_slice(&rhs, dim, 1);
//! 	let mut sol = faer::MatMut::from_column_major_slice_mut(&mut sol, dim, 1);
//!
//! 	// optional: fill reducing permutation
//! 	let (perm, perm_inv) = {
//! 		let mut perm = Vec::new();
//! 		let mut perm_inv = Vec::new();
//! 		perm.try_reserve_exact(dim)?;
//! 		perm_inv.try_reserve_exact(dim)?;
//! 		perm.resize(dim, 0usize);
//! 		perm_inv.resize(dim, 0usize);
//!
//! 		let mut mem = MemBuffer::try_new(amd::order_scratch::<usize>(dim, A_nnz))?;
//! 		amd::order(
//! 			&mut perm,
//! 			&mut perm_inv,
//! 			A_lower.symbolic(),
//! 			amd::Control::default(),
//! 			MemStack::new(&mut mem),
//! 		)?;
//!
//! 		(perm, perm_inv)
//! 	};
//!
//! 	let perm = unsafe { faer::perm::PermRef::new_unchecked(&perm, &perm_inv, dim) };
//!
//! 	let A_perm_lower = {
//! 		let mut A_perm_col_ptrs = Vec::new();
//! 		let mut A_perm_row_indices = Vec::new();
//! 		let mut A_perm_values = Vec::new();
//!
//! 		A_perm_col_ptrs.try_reserve_exact(dim + 1)?;
//! 		A_perm_col_ptrs.resize(dim + 1, 0usize);
//! 		A_perm_row_indices.try_reserve_exact(A_nnz)?;
//! 		A_perm_row_indices.resize(A_nnz, 0usize);
//! 		A_perm_values.try_reserve_exact(A_nnz)?;
//! 		A_perm_values.resize(A_nnz, 0.0f64);
//!
//! 		let mut mem = MemBuffer::try_new(faer::sparse::utils::permute_self_adjoint_scratch::<
//! 			usize,
//! 		>(dim))?;
//! 		faer::sparse::utils::permute_self_adjoint_to_unsorted(
//! 			&mut A_perm_values,
//! 			&mut A_perm_col_ptrs,
//! 			&mut A_perm_row_indices,
//! 			A_lower.rb(),
//! 			perm,
//! 			faer::Side::Lower,
//! 			faer::Side::Lower,
//! 			MemStack::new(&mut mem),
//! 		);
//!
//! 		SparseColMat::<usize, f64>::new(
//! 			unsafe {
//! 				SymbolicSparseColMat::new_unchecked(
//! 					dim,
//! 					dim,
//! 					A_perm_col_ptrs,
//! 					None,
//! 					A_perm_row_indices,
//! 				)
//! 			},
//! 			A_perm_values,
//! 		)
//! 	};
//!
//! 	let A_perm_upper = A_perm_lower.rb().transpose().symbolic().to_col_major()?;
//!
//! 	// symbolic analysis
//! 	let symbolic = {
//! 		let mut mem = MemBuffer::try_new(StackReq::any_of(&[
//! 			simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(dim, A_nnz),
//! 			supernodal::factorize_supernodal_symbolic_cholesky_scratch::<usize>(dim),
//! 		]))?;
//! 		let stack = MemStack::new(&mut mem);
//!
//! 		let mut etree = Vec::new();
//! 		let mut col_counts = Vec::new();
//! 		etree.try_reserve_exact(dim)?;
//! 		etree.resize(dim, 0isize);
//! 		col_counts.try_reserve_exact(dim)?;
//! 		col_counts.resize(dim, 0usize);
//!
//! 		simplicial::prefactorize_symbolic_cholesky(
//! 			&mut etree,
//! 			&mut col_counts,
//! 			A_perm_upper.rb(),
//! 			stack,
//! 		);
//! 		supernodal::factorize_supernodal_symbolic_cholesky(
//! 			A_perm_upper.rb(),
//! 			// SAFETY: `etree` was filled correctly by
//! 			// `simplicial::prefactorize_symbolic_cholesky`.
//! 			unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
//! 			&col_counts,
//! 			stack,
//! 			faer::sparse::linalg::SymbolicSupernodalParams {
//! 				relax: Some(&[(usize::MAX, 1.0)]),
//! 			},
//! 		)?
//! 	};
//!
//! 	// numerical factorization
//! 	let mut mem = MemBuffer::try_new(StackReq::any_of(&[
//! 		supernodal::factorize_supernodal_numeric_llt_scratch::<usize, f64>(
//! 			&symbolic,
//! 			faer::Par::Seq,
//! 			Default::default(),
//! 		),
//! 		supernodal::factorize_supernodal_numeric_ldlt_scratch::<usize, f64>(
//! 			&symbolic,
//! 			faer::Par::Seq,
//! 			Default::default(),
//! 		),
//! 		supernodal::factorize_supernodal_numeric_intranode_lblt_scratch::<usize, f64>(
//! 			&symbolic,
//! 			faer::Par::Seq,
//! 			Default::default(),
//! 		),
//! 		faer::perm::permute_rows_in_place_scratch::<usize, f64>(dim, 1),
//! 		symbolic.solve_in_place_scratch::<f64>(dim, Par::Seq),
//! 	]))?;
//! 	let stack = MemStack::new(&mut mem);
//!
//! 	// llt skipped since a is not positive-definite
//!
//! 	// numerical ldlt factorization
//! 	{
//! 		let mut L_values = Vec::new();
//! 		L_values.try_reserve_exact(symbolic.len_val())?;
//! 		L_values.resize(symbolic.len_val(), 0.0f64);
//!
//! 		supernodal::factorize_supernodal_numeric_ldlt::<usize, f64>(
//! 			&mut L_values,
//! 			A_perm_lower.rb(),
//! 			LdltRegularization::default(),
//! 			&symbolic,
//! 			faer::Par::Seq,
//! 			stack,
//! 			Default::default(),
//! 		);
//!
//! 		let ldlt = supernodal::SupernodalLdltRef::<'_, usize, f64>::new(&symbolic, &L_values);
//!
//! 		sol.copy_from(rhs);
//! 		faer::perm::permute_rows_in_place(sol.rb_mut(), perm, stack);
//! 		ldlt.solve_in_place_with_conj(faer::Conj::No, sol.rb_mut(), faer::Par::Seq, stack);
//! 		faer::perm::permute_rows_in_place(sol.rb_mut(), perm.inverse(), stack);
//!
//! 		assert!((&A * &sol - &rhs).norm_max() <= 1e-14);
//! 	}
//!
//! 	// numerical intranodal LBLT factorization
//! 	{
//! 		let mut L_values = Vec::new();
//! 		let mut subdiag = Vec::new();
//! 		let mut pivot_perm = Vec::new();
//! 		let mut pivot_perm_inv = Vec::new();
//!
//! 		L_values.try_reserve_exact(symbolic.len_val())?;
//! 		L_values.resize(symbolic.len_val(), 0.0f64);
//! 		subdiag.try_reserve_exact(dim)?;
//! 		subdiag.resize(dim, 0.0f64);
//! 		pivot_perm.try_reserve(dim)?;
//! 		pivot_perm.resize(dim, 0usize);
//! 		pivot_perm_inv.try_reserve(dim)?;
//! 		pivot_perm_inv.resize(dim, 0usize);
//!
//! 		supernodal::factorize_supernodal_numeric_intranode_lblt::<usize, f64>(
//! 			&mut L_values,
//! 			&mut subdiag,
//! 			&mut pivot_perm,
//! 			&mut pivot_perm_inv,
//! 			A_perm_lower.rb(),
//! 			&symbolic,
//! 			faer::Par::Seq,
//! 			stack,
//! 			Default::default(),
//! 		);
//!
//! 		let piv_perm =
//! 			unsafe { faer::perm::PermRef::new_unchecked(&pivot_perm, &pivot_perm_inv, dim) };
//! 		let lblt = supernodal::SupernodalIntranodeLbltRef::<'_, usize, f64>::new(
//! 			&symbolic, &L_values, &subdiag, piv_perm,
//! 		);
//!
//! 		sol.copy_from(rhs);
//! 		// we can merge these two permutations if we want to be optimal
//! 		faer::perm::permute_rows_in_place(sol.rb_mut(), perm, stack);
//! 		faer::perm::permute_rows_in_place(sol.rb_mut(), piv_perm, stack);
//!
//! 		lblt.solve_in_place_no_numeric_permute_with_conj(
//! 			faer::Conj::No,
//! 			sol.rb_mut(),
//! 			faer::Par::Seq,
//! 			stack,
//! 		);
//!
//! 		// we can also merge these two permutations if we want to be optimal
//! 		faer::perm::permute_rows_in_place(sol.rb_mut(), piv_perm.inverse(), stack);
//! 		faer::perm::permute_rows_in_place(sol.rb_mut(), perm.inverse(), stack);
//!
//! 		assert!((&A * &sol - &rhs).norm_max() <= 1e-14);
//! 	}
//! 	Ok(())
//! }
//!
//! supernodal_cholesky().unwrap()
//! ```
// implementation inspired by https://gitlab.com/hodge_star/catamari

use super::super::utils::*;
use super::ghost;
use crate::assert;
use crate::internal_prelude_sp::*;
use linalg::cholesky::lblt::factor::{LbltInfo, LbltParams};
use linalg::cholesky::ldlt::factor::{LdltError, LdltInfo, LdltParams, LdltRegularization};
use linalg::cholesky::llt::factor::{LltError, LltInfo, LltParams, LltRegularization};
use linalg_sp::{SupernodalThreshold, SymbolicSupernodalParams, amd, triangular_solve};

/// fill reducing ordering to use for the cholesky factorization
#[derive(Copy, Clone, Debug, Default)]
pub enum SymmetricOrdering<'a, I: Index> {
	/// approximate minimum degree ordering. default option
	#[default]
	Amd,
	/// no reordering
	Identity,
	/// custom reordering
	Custom(PermRef<'a, I>),
}

/// simplicial factorization module
///
/// a simplicial factorization is one that processes the elements of the cholesky factor of the
/// input matrix single elements, rather than by blocks. this is more efficient if the cholesky
/// factor is very sparse
pub mod simplicial {
	use super::*;
	use crate::assert;

	/// reference to a slice containing the cholesky factor's elimination tree
	///
	/// the elimination tree (or elimination forest, in the general case) is a structure
	/// representing the relationship between the columns of the cholesky factor, and the way
	/// how earlier columns contribute their sparsity pattern to later columns of the factor
	#[derive(Copy, Clone, Debug)]
	pub struct EliminationTreeRef<'a, I: Index> {
		pub(crate) inner: &'a [I::Signed],
	}

	impl<'a, I: Index> EliminationTreeRef<'a, I> {
		/// dimension of the original matrix
		pub fn len(&self) -> usize {
			self.inner.len()
		}

		/// returns the raw elimination tree
		///
		/// a value can be either nonnegative to represent the index of the parent of a given node,
		/// or `-1` to signify that it has no parent
		#[inline]
		pub fn into_inner(self) -> &'a [I::Signed] {
			self.inner
		}

		/// creates an elimination tree reference from the underlying array
		///
		/// # safety
		/// the elimination tree must come from an array that was previously filled with
		/// [`prefactorize_symbolic_cholesky`]
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

	/// computes the size and alignment of the workspace required to compute the elimination tree
	/// and column counts of a matrix of size `n` with `nnz` non-zero entries
	pub fn prefactorize_symbolic_cholesky_scratch<I: Index>(n: usize, nnz: usize) -> StackReq {
		_ = nnz;
		StackReq::new::<I>(n)
	}

	/// computes the elimination tree and column counts of the cholesky factorization of the matrix
	/// $A$
	///
	/// # note
	/// only the upper triangular part of $A$ is analyzed
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

	/// computes the size and alignment of the workspace required to compute the symbolic
	/// cholesky factorization of a square matrix with size `n`
	pub fn factorize_simplicial_symbolic_cholesky_scratch<I: Index>(n: usize) -> StackReq {
		let n_scratch = StackReq::new::<I>(n);
		StackReq::all_of(&[n_scratch, n_scratch, n_scratch])
	}

	/// computes the symbolic structure of the cholesky factor of the matrix $A$
	///
	/// # note
	/// only the upper triangular part of $A$ is analyzed
	///
	/// # panics
	/// the elimination tree and column counts must be computed by calling
	/// [`prefactorize_symbolic_cholesky`] with the same matrix. otherwise, the behavior is
	/// unspecified and panics may occur
	pub fn factorize_simplicial_symbolic_cholesky<I: Index>(
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
		let mut L_row_idx = try_zeroed::<I>(l_nnz)?;

		with_dim!(L_NNZ, l_nnz);
		let (current_row_idxex, stack) = unsafe { stack.make_raw::<I>(n) };
		let (ereach_stack, stack) = unsafe { stack.make_raw::<I>(n) };
		let (visited, _) = unsafe { stack.make_raw::<I::Signed>(n) };

		let ereach_stack = Array::from_mut(ereach_stack, N);
		let visited = Array::from_mut(visited, N);

		visited.as_mut().fill(I::Signed::truncate(NONE));
		{
			let L_row_idx = Array::from_mut(&mut L_row_idx, L_NNZ);
			let L_col_ptr_start = Array::from_ref(Idx::from_slice_ref_checked(&L_col_ptr[..n], L_NNZ), N);
			let current_row_idxex = Array::from_mut(ghost::copy_slice(current_row_idxex, L_col_ptr_start.as_ref()), N);

			for k in N.indices() {
				let reach = ereach(ereach_stack, A, etree, k, visited);
				for &j in reach {
					let j = j.zx();
					let cj = &mut current_row_idxex[j];
					let row_idx = L_NNZ.check(*cj.zx() + 1);
					*cj = row_idx.truncate();
					L_row_idx[row_idx] = *k.truncate();
				}
				let k_start = L_col_ptr_start[k].zx();
				L_row_idx[k_start] = *k.truncate();
			}
		}

		let etree = try_collect(
			bytemuck::cast_slice::<I::Signed, I>(MaybeIdx::as_slice_ref(etree.as_ref()))
				.iter()
				.copied(),
		)?;

		let _ = SymbolicSparseColMatRef::new_unsorted_checked(n, n, &L_col_ptr, None, &L_row_idx);
		Ok(SymbolicSimplicialCholesky {
			dimension: n,
			col_ptr: L_col_ptr,
			row_idx: L_row_idx,
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
		regularization: LdltRegularization<'_, T::Real>,

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

		let (current_row_idxex, stack) = unsafe { stack.make_raw::<I>(n) };
		let (ereach_stack, stack) = unsafe { stack.make_raw::<I>(n) };
		let (visited, _) = unsafe { stack.make_raw::<I::Signed>(n) };

		let ereach_stack = Array::from_mut(ereach_stack, N);
		let visited = Array::from_mut(visited, N);

		visited.as_mut().fill(I::Signed::truncate(NONE));

		let L_values = Array::from_mut(L_values, L_NNZ);
		let L_row_idx = Array::from_mut(L_row_idx, L_NNZ);

		let L_col_ptr_start = Array::from_ref(Idx::from_slice_ref_checked(&L_col_ptr[..n], L_NNZ), N);

		let current_row_idxex = Array::from_mut(ghost::copy_slice(current_row_idxex, L_col_ptr_start.as_ref()), N);

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
				let cj = &mut current_row_idxex[j];
				let row_idx = L_NNZ.check(*cj.zx() + 1);
				*cj = row_idx.truncate();

				let mut xj = copy(x[j]);
				x[j] = zero::<T>();

				let dj = recip(real(L_values[j_start]));
				let lkj = mul_real(xj, dj);
				if kind == FactorizationKind::Llt {
					xj = copy(lkj);
				}

				let range = j_start.next()..row_idx.into();
				for (i, lij) in iter::zip(&L_row_idx[range.clone()], &L_values[range]) {
					let i = N.check(i.zx());
					x[i] = x[i] - conj(*lij) * xj;
				}

				d = d - real(lkj * conj(xj));

				L_values[row_idx] = lkj;
				L_row_idx[row_idx] = *k.truncate();
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
	fn factorize_simplicial_numeric_cholesky<I: Index, T: ComplexField>(
		L_values: &mut [T],
		kind: FactorizationKind,
		A: SparseColMatRef<'_, I, T>,
		regularization: LdltRegularization<'_, T::Real>,
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
		let (current_row_idxex, stack) = unsafe { stack.make_raw::<I>(n) };
		let (ereach_stack, stack) = unsafe { stack.make_raw::<I>(n) };
		let (visited, _) = unsafe { stack.make_raw::<I::Signed>(n) };

		let ereach_stack = Array::from_mut(ereach_stack, N);
		let visited = Array::from_mut(visited, N);

		visited.as_mut().fill(I::Signed::truncate(NONE));

		let L_values = Array::from_mut(L_values, L_NNZ);
		let L_row_idx = Array::from_ref(L_row_idx, L_NNZ);

		let L_col_ptr_start = Array::from_ref(Idx::from_slice_ref_checked(&L_col_ptr[..n], L_NNZ), N);

		let current_row_idxex = Array::from_mut(ghost::copy_slice(current_row_idxex, L_col_ptr_start.as_ref()), N);

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
				let cj = &mut current_row_idxex[j];
				let row_idx = L_NNZ.check(*cj.zx() + 1);
				*cj = row_idx.truncate();

				let mut xj = copy(x[j]);
				x[j] = zero::<T>();

				let dj = recip(real(L_values[j_start]));
				let lkj = mul_real(xj, dj);
				if kind == FactorizationKind::Llt {
					xj = copy(lkj);
				}

				let range = j_start.next()..row_idx.into();
				for (i, lij) in iter::zip(&L_row_idx[range.clone()], &L_values[range]) {
					let i = N.check(i.zx());
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

	/// computes the numeric values of the cholesky $LL^H$ factor of the matrix $A$, and stores them
	/// in `l_values`
	///
	/// # note
	/// only the upper triangular part of $A$ is accessed
	///
	/// # panics
	/// the symbolic structure must be computed by calling
	/// [`factorize_simplicial_symbolic_cholesky`] on a matrix with the same symbolic structure
	/// otherwise, the behavior is unspecified and panics may occur
	pub fn factorize_simplicial_numeric_llt<I: Index, T: ComplexField>(
		L_values: &mut [T],
		A: SparseColMatRef<'_, I, T>,
		regularization: LltRegularization<T::Real>,
		symbolic: &SymbolicSimplicialCholesky<I>,
		stack: &mut MemStack,
	) -> Result<LltInfo, LltError> {
		factorize_simplicial_numeric_cholesky(
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

	/// computes the row indices and  numeric values of the cholesky $LL^H$ factor of the matrix
	/// $A$, and stores them in `l_row_idx` and `l_values`
	///
	/// # note
	/// only the upper triangular part of $A$ is accessed
	///
	/// # panics
	/// the elimination tree and column counts must be computed by calling
	/// [`prefactorize_symbolic_cholesky`] with the same matrix, then the column pointers are
	/// computed from a prefix sum of the column counts. otherwise, the behavior is unspecified
	/// and panics may occur
	pub fn factorize_simplicial_numeric_llt_with_row_idx<I: Index, T: ComplexField>(
		L_values: &mut [T],
		L_row_idx: &mut [I],
		L_col_ptr: &[I],

		etree: EliminationTreeRef<'_, I>,
		A: SparseColMatRef<'_, I, T>,
		regularization: LltRegularization<T::Real>,

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

	/// computes the numeric values of the cholesky $LDL^H$ factors of the matrix $A$, and stores
	/// them in `l_values`
	///
	/// # note
	/// only the upper triangular part of $A$ is accessed
	///
	/// # panics
	/// the symbolic structure must be computed by calling
	/// [`factorize_simplicial_symbolic_cholesky`] on a matrix with the same symbolic structure
	/// otherwise, the behavior is unspecified and panics may occur
	pub fn factorize_simplicial_numeric_ldlt<I: Index, T: ComplexField>(
		L_values: &mut [T],
		A: SparseColMatRef<'_, I, T>,
		regularization: LdltRegularization<'_, T::Real>,
		symbolic: &SymbolicSimplicialCholesky<I>,
		stack: &mut MemStack,
	) -> Result<LdltInfo, LdltError> {
		match factorize_simplicial_numeric_cholesky(L_values, FactorizationKind::Ldlt, A, regularization, symbolic, stack) {
			Ok(info) => Ok(LdltInfo {
				dynamic_regularization_count: info.dynamic_regularization_count,
			}),
			Err(LltError::NonPositivePivot { index }) => Err(LdltError::ZeroPivot { index }),
		}
	}

	/// computes the row indices and  numeric values of the cholesky $LDL^H$ factor of the matrix
	/// $A$, and stores them in `l_row_idx` and `l_values`
	///
	/// # note
	/// only the upper triangular part of $A$ is accessed
	///
	/// # panics
	/// the elimination tree and column counts must be computed by calling
	/// [`prefactorize_symbolic_cholesky`] with the same matrix, then the column pointers are
	/// computed from a prefix sum of the column counts. otherwise, the behavior is unspecified
	/// and panics may occur
	pub fn factorize_simplicial_numeric_ldlt_with_row_idx<I: Index, T: ComplexField>(
		L_values: &mut [T],
		L_row_idx: &mut [I],
		L_col_ptr: &[I],

		etree: EliminationTreeRef<'_, I>,
		A: SparseColMatRef<'_, I, T>,
		regularization: LdltRegularization<'_, T::Real>,

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
		/// creates a new cholesky $LL^H$ factor from the symbolic part and numerical values
		///
		/// # panics
		/// panics if `values.len() != symbolic.len_val()`>
		#[inline]
		pub fn new(symbolic: &'a SymbolicSimplicialCholesky<I>, values: &'a [T]) -> Self {
			assert!(values.len() == symbolic.len_val());
			Self { symbolic, values }
		}

		/// returns the symbolic part of the cholesky factor
		#[inline]
		pub fn symbolic(self) -> &'a SymbolicSimplicialCholesky<I> {
			self.symbolic
		}

		/// returns the numerical values of the cholesky $LL^H$ factor
		#[inline]
		pub fn values(self) -> &'a [T] {
			self.values
		}

		/// solves the equation $A x = \text{rhs}$ and stores the result in `rhs`, implicitly
		/// conjugating $A$ if needed
		///
		/// # panics
		/// panics if `rhs.nrows() != self.symbolic().nrows()`
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
			triangular_solve::solve_lower_triangular_in_place(l, conj, rhs.rb_mut(), par);
			triangular_solve::solve_lower_triangular_transpose_in_place(l, conj.compose(Conj::Yes), rhs.rb_mut(), par);
		}
	}

	impl<'a, I: Index, T> SimplicialLdltRef<'a, I, T> {
		/// creates a new cholesky $LDL^H$ factor from the symbolic part and numerical values
		///
		/// # panics
		/// panics if `values.len() != symbolic.len_val()`>
		#[inline]
		pub fn new(symbolic: &'a SymbolicSimplicialCholesky<I>, values: &'a [T]) -> Self {
			assert!(values.len() == symbolic.len_val());
			Self { symbolic, values }
		}

		/// returns the symbolic part of the cholesky factor
		#[inline]
		pub fn symbolic(self) -> &'a SymbolicSimplicialCholesky<I> {
			self.symbolic
		}

		/// returns the numerical values of the cholesky $LDL^H$ factor
		#[inline]
		pub fn values(self) -> &'a [T] {
			self.values
		}

		/// solves the equation $A x = \text{rhs}$ and stores the result in `rhs`, implicitly
		/// conjugating $A$ if needed
		///
		/// # panics
		/// panics if `rhs.nrows() != self.symbolic().nrows()`
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
			triangular_solve::solve_unit_lower_triangular_in_place(ld, conj, x.rb_mut(), par);
			triangular_solve::ldlt_scale_solve_unit_lower_triangular_transpose_in_place_impl(ld, conj.compose(Conj::Yes), x.rb_mut(), par);
		}
	}

	impl<I: Index> SymbolicSimplicialCholesky<I> {
		/// returns the number of rows of the cholesky factor
		#[inline]
		pub fn nrows(&self) -> usize {
			self.dimension
		}

		/// returns the number of columns of the cholesky factor
		#[inline]
		pub fn ncols(&self) -> usize {
			self.nrows()
		}

		/// returns the length of the slice that can be used to contain the numerical values of the
		/// cholesky factor
		#[inline]
		pub fn len_val(&self) -> usize {
			self.row_idx.len()
		}

		/// returns the column pointers of the cholesky factor
		#[inline]
		pub fn col_ptr(&self) -> &[I] {
			&self.col_ptr
		}

		/// returns the row indices of the cholesky factor
		#[inline]
		pub fn row_idx(&self) -> &[I] {
			&self.row_idx
		}

		/// returns the cholesky factor's symbolic structure
		#[inline]
		pub fn factor(&self) -> SymbolicSparseColMatRef<'_, I> {
			unsafe { SymbolicSparseColMatRef::new_unchecked(self.dimension, self.dimension, &self.col_ptr, None, &self.row_idx) }
		}

		/// returns the size and alignment of the workspace required to solve the system
		/// $A x = rhs$
		pub fn solve_in_place_scratch<T>(&self, rhs_ncols: usize) -> StackReq {
			let _ = rhs_ncols;
			StackReq::EMPTY
		}
	}

	/// returns the size and alignment of the workspace required to compute the numeric
	/// cholesky $LDL^H$ factorization of a matrix $A$ with dimension `n`
	pub fn factorize_simplicial_numeric_ldlt_scratch<I: Index, T: ComplexField>(n: usize) -> StackReq {
		let n_scratch = StackReq::new::<I>(n);
		StackReq::all_of(&[temp_mat_scratch::<T>(n, 1), n_scratch, n_scratch, n_scratch])
	}

	/// returns the size and alignment of the workspace required to compute the numeric
	/// cholesky $LL^H$ factorization of a matrix $A$ with dimension `n`
	pub fn factorize_simplicial_numeric_llt_scratch<I: Index, T: ComplexField>(n: usize) -> StackReq {
		factorize_simplicial_numeric_ldlt_scratch::<I, T>(n)
	}

	/// cholesky $LL^H$ factor containing both its symbolic and numeric representations
	#[derive(Debug)]
	pub struct SimplicialLltRef<'a, I: Index, T> {
		symbolic: &'a SymbolicSimplicialCholesky<I>,
		values: &'a [T],
	}

	/// cholesky $LDL^H$ factors containing both the symbolic and numeric representations
	#[derive(Debug)]
	pub struct SimplicialLdltRef<'a, I: Index, T> {
		symbolic: &'a SymbolicSimplicialCholesky<I>,
		values: &'a [T],
	}

	/// cholesky factor structure containing its symbolic structure
	#[derive(Debug, Clone)]
	pub struct SymbolicSimplicialCholesky<I> {
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

/// supernodal factorization module
///
/// a supernodal factorization is one that processes the elements of the cholesky factor of the
/// input matrix by blocks, rather than single elements. this is more efficient if the cholesky
/// factor is somewhat dense
pub mod supernodal {
	use super::*;
	use crate::linalg::matmul::internal::{spicy_matmul, spicy_matmul_scratch};
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

	/// symbolic structure of a single supernode from the cholesky factor
	#[derive(Debug)]
	pub struct SymbolicSupernodeRef<'a, I> {
		start: usize,
		pattern: &'a [I],
	}

	/// a single supernode from the cholesky factor
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
		/// returns the starting index of the supernode
		#[inline]
		pub fn start(self) -> usize {
			self.start
		}

		/// returns the pattern of the row indices in the supernode, excluding those on the block
		/// diagonal
		pub fn pattern(self) -> &'a [I] {
			self.pattern
		}
	}

	impl<'a, I: Index, T> SupernodeRef<'a, I, T> {
		/// returns the starting index of the supernode
		#[inline]
		pub fn start(self) -> usize {
			self.symbolic.start
		}

		/// returns the pattern of the row indices in the supernode, excluding those on the block
		/// diagonal
		pub fn pattern(self) -> &'a [I] {
			self.symbolic.pattern
		}

		/// returns a view over the numerical values of the supernode
		pub fn val(self) -> MatRef<'a, T> {
			self.matrix
		}
	}

	/// cholesky $LL^H$ factor containing both its symbolic and numeric representations
	#[derive(Debug)]
	pub struct SupernodalLltRef<'a, I: Index, T> {
		symbolic: &'a SymbolicSupernodalCholesky<I>,
		values: &'a [T],
	}

	/// cholesky $LDL^H$ factors containing both the symbolic and numeric representations
	#[derive(Debug)]
	pub struct SupernodalLdltRef<'a, I: Index, T> {
		symbolic: &'a SymbolicSupernodalCholesky<I>,
		values: &'a [T],
	}

	/// cholesky $LBL^\top$ factors containing both the symbolic and numeric representations
	#[derive(Debug)]
	pub struct SupernodalIntranodeLbltRef<'a, I: Index, T> {
		symbolic: &'a SymbolicSupernodalCholesky<I>,
		values: &'a [T],
		subdiag: &'a [T],
		pub(super) perm: PermRef<'a, I>,
	}

	/// cholesky factor structure containing its symbolic structure
	#[derive(Debug)]
	pub struct SymbolicSupernodalCholesky<I> {
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
		/// returns the number of supernodes in the cholesky factor
		#[inline]
		pub fn n_supernodes(&self) -> usize {
			self.supernode_postorder.len()
		}

		/// returns the number of rows of the cholesky factor
		#[inline]
		pub fn nrows(&self) -> usize {
			self.dimension
		}

		/// returns the number of columns of the cholesky factor
		#[inline]
		pub fn ncols(&self) -> usize {
			self.nrows()
		}

		/// returns the length of the slice that can be used to contain the numerical values of the
		/// cholesky factor
		#[inline]
		pub fn len_val(&self) -> usize {
			self.col_ptr_for_val()[self.n_supernodes()].zx()
		}

		/// returns a slice of length `self.n_supernodes()` containing the beginning index of each
		/// supernode
		#[inline]
		pub fn supernode_begin(&self) -> &[I] {
			&self.supernode_begin[..self.n_supernodes()]
		}

		/// returns a slice of length `self.n_supernodes()` containing the past-the-end index of
		/// each
		#[inline]
		pub fn supernode_end(&self) -> &[I] {
			&self.supernode_begin[1..]
		}

		/// returns the column pointers for row indices of each supernode
		#[inline]
		pub fn col_ptr_for_row_idx(&self) -> &[I] {
			&self.col_ptr_for_row_idx
		}

		/// returns the column pointers for numerical values of each supernode
		#[inline]
		pub fn col_ptr_for_val(&self) -> &[I] {
			&self.col_ptr_for_val
		}

		/// returns the row indices of the cholesky factor
		///
		/// # note
		/// note that the row indices of each supernode do not contain those of the block diagonal
		/// part
		#[inline]
		pub fn row_idx(&self) -> &[I] {
			&self.row_idx
		}

		/// returns the symbolic structure of the `s`'th supernode
		#[inline]
		pub fn supernode(&self, s: usize) -> supernodal::SymbolicSupernodeRef<'_, I> {
			let symbolic = self;
			let start = symbolic.supernode_begin[s].zx();
			let pattern = &symbolic.row_idx()[symbolic.col_ptr_for_row_idx()[s].zx()..symbolic.col_ptr_for_row_idx()[s + 1].zx()];
			supernodal::SymbolicSupernodeRef { start, pattern }
		}

		/// returns the size and alignment of the workspace required to solve the system
		/// $A x = rhs$
		pub fn solve_in_place_scratch<T: ComplexField>(&self, rhs_ncols: usize, par: Par) -> StackReq {
			_ = par;
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
	impl<I: Index, T> Copy for SupernodalIntranodeLbltRef<'_, I, T> {}
	impl<I: Index, T> Clone for SupernodalIntranodeLbltRef<'_, I, T> {
		fn clone(&self) -> Self {
			*self
		}
	}

	impl<'a, I: Index, T> SupernodalLdltRef<'a, I, T> {
		/// creates new cholesky $LDL^H$ factors from the symbolic part and
		/// numerical values
		///
		/// # panics
		/// - panics if `values.len() != symbolic.len_val()`
		#[inline]
		pub fn new(symbolic: &'a SymbolicSupernodalCholesky<I>, values: &'a [T]) -> Self {
			assert!(values.len() == symbolic.len_val());
			Self { symbolic, values }
		}

		/// returns the symbolic part of the cholesky factor
		#[inline]
		pub fn symbolic(self) -> &'a SymbolicSupernodalCholesky<I> {
			self.symbolic
		}

		/// returns the numerical values of the l factor
		#[inline]
		pub fn values(self) -> &'a [T] {
			self.values
		}

		/// returns the `s`'th supernode
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

		/// solves the equation $A x = \text{rhs}$ and stores the result in `rhs`, implicitly
		/// conjugating $A$ if needed
		///
		/// # panics
		/// panics if `rhs.nrows() != self.symbolic().nrows()`
		#[math]
		pub fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
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
				linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(Ls_top, conj, x_top.rb_mut(), par);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack) };
				let mut tmp = tmp.as_mat_mut();
				linalg::matmul::matmul_with_conj(tmp.rb_mut(), Accum::Replace, Ls_bot, conj, x_top.rb(), Conj::No, one::<T>(), par);

				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						x[(i, j)] = x[(i, j)] - tmp[(idx, j)]
					}
				}
			}
			for s in 0..symbolic.n_supernodes() {
				let s = self.supernode(s);
				let size = s.matrix.ncols();
				let Ds = s.matrix.diagonal().column_vector();
				for j in 0..k {
					for idx in 0..size {
						let d_inv = recip(real(Ds[idx]));
						let i = idx + s.start();
						x[(i, j)] = mul_real(x[(i, j)], d_inv)
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
		/// creates a new cholesky $LL^H$ factor from the symbolic part and
		/// numerical values
		///
		/// # panics
		/// - panics if `values.len() != symbolic.len_val()`
		#[inline]
		pub fn new(symbolic: &'a SymbolicSupernodalCholesky<I>, values: &'a [T]) -> Self {
			assert!(values.len() == symbolic.len_val());
			Self { symbolic, values }
		}

		/// returns the symbolic part of the cholesky factor
		#[inline]
		pub fn symbolic(self) -> &'a SymbolicSupernodalCholesky<I> {
			self.symbolic
		}

		/// returns the numerical values of the l factor
		#[inline]
		pub fn values(self) -> &'a [T] {
			self.values
		}

		/// returns the `s`'th supernode
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

		/// solves the equation $L x = \text{rhs}$ and stores the result in `rhs`,
		/// implicitly conjugating $L$ if needed
		///
		/// # panics
		/// panics if `rhs.nrows() != self.symbolic().nrows()`
		#[math]
		pub fn l_solve_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
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
				linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(Ls_top, conj, x_top.rb_mut(), par);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack) };
				let mut tmp = tmp.as_mat_mut();
				linalg::matmul::matmul_with_conj(tmp.rb_mut(), Accum::Replace, Ls_bot, conj, x_top.rb(), Conj::No, one::<T>(), par);

				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						x[(i, j)] = x[(i, j)] - tmp[(idx, j)]
					}
				}
			}
		}

		/// solves the equation $L^\top x = \text{rhs}$ and stores the result in `rhs`,
		/// implicitly conjugating $L$ if needed
		///
		/// # panics
		/// panics if `rhs.nrows() != self.symbolic().nrows()`
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
			let k = x.ncols();
			for s in (0..symbolic.n_supernodes()).rev() {
				let s = self.supernode(s);
				let size = s.matrix.ncols();
				let Ls = s.matrix;
				let (Ls_top, Ls_bot) = Ls.split_at_row(size);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack) };
				let mut tmp = tmp.as_mat_mut();
				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						tmp[(idx, j)] = copy(x[(i, j)]);
					}
				}

				let mut x_top = x.rb_mut().subrows_mut(s.start(), size);
				linalg::matmul::matmul_with_conj(x_top.rb_mut(), Accum::Add, Ls_bot.transpose(), conj, tmp.rb(), Conj::No, -one::<T>(), par);
				linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(Ls_top.transpose(), conj, x_top.rb_mut(), par);
			}
		}

		/// solves the equation $A x = \text{rhs}$ and stores the result in `rhs`, implicitly
		/// conjugating $A$ if needed
		///
		/// # panics
		/// panics if `rhs.nrows() != self.symbolic().nrows()`
		#[math]
		pub fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack)
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
				linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(Ls_top, conj, x_top.rb_mut(), par);

				let (mut tmp, _) = unsafe { temp_mat_uninit::<T, _, _>(s.pattern().len(), k, stack) };
				let mut tmp = tmp.as_mat_mut();
				linalg::matmul::matmul_with_conj(tmp.rb_mut(), Accum::Replace, Ls_bot, conj, x_top.rb(), Conj::No, one::<T>(), par);

				for j in 0..k {
					for (idx, i) in s.pattern().iter().enumerate() {
						let i = i.zx();
						x[(i, j)] = x[(i, j)] - tmp[(idx, j)]
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

	impl<'a, I: Index, T> SupernodalIntranodeLbltRef<'a, I, T> {
		/// creates a new cholesky intranodal $LBL^\top$ factor from the symbolic part and
		/// numerical values, as well as the pivoting permutation
		///
		/// # panics
		/// - panics if `values.len() != symbolic.len_val()`
		/// - panics if `subdiag.len() != symbolic.nrows()`
		/// - panics if `perm.len() != symbolic.nrows()`
		#[inline]
		pub fn new(symbolic: &'a SymbolicSupernodalCholesky<I>, values: &'a [T], subdiag: &'a [T], perm: PermRef<'a, I>) -> Self {
			assert!(all(
				values.len() == symbolic.len_val(),
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

		/// returns the symbolic part of the cholesky factor
		#[inline]
		pub fn symbolic(self) -> &'a SymbolicSupernodalCholesky<I> {
			self.symbolic
		}

		/// returns the numerical values of the l factor
		#[inline]
		pub fn val(self) -> &'a [T] {
			self.values
		}

		/// returns the `s`'th supernode
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

		/// returns the pivoting permutation
		#[inline]
		pub fn perm(&self) -> PermRef<'a, I> {
			self.perm
		}

		/// solves the system $L B L^H x = \text{rhs}$, implicitly conjugating $L$ and
		/// $B$ if needed
		///
		/// # note
		/// note that this function doesn't apply the pivoting permutation. users are expected to
		/// apply it manually to `rhs` before and after calling this function
		///
		/// # panics
		/// panics if `rhs.nrows() != self.symbolic().nrows()`
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

	/// returns the size and alignment of the workspace required to compute the symbolic supernodal
	/// factorization of a matrix of size `n`
	pub fn factorize_supernodal_symbolic_cholesky_scratch<I: Index>(n: usize) -> StackReq {
		StackReq::new::<I>(n).array(4)
	}

	/// computes the supernodal symbolic structure of the cholesky factor of the matrix $A$
	///
	/// # note
	/// only the upper triangular part of $A$ is analyzed
	///
	/// # panics
	/// the elimination tree and column counts must be computed by calling
	/// [`simplicial::prefactorize_symbolic_cholesky`] with the same matrix. otherwise, the behavior
	/// is unspecified and panics may occur
	pub fn factorize_supernodal_symbolic_cholesky<I: Index>(
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
		let original_stack = stack;

		let (index_to_super__, stack) = unsafe { original_stack.make_raw::<I>(n) };
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

	/// returns the size and alignment of the workspace required to compute the numeric
	/// cholesky $LL^H$ factorization of a matrix $A$ with dimension `n`
	pub fn factorize_supernodal_numeric_llt_scratch<I: Index, T: ComplexField>(
		symbolic: &SymbolicSupernodalCholesky<I>,
		par: Par,
		params: Spec<LltParams, T>,
	) -> StackReq {
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let row_idx = &*symbolic.row_idx;

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

				let d_pattern = &row_idx[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()];
				let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
				let d_pattern_mid_len = d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));

				d_scratch = d_scratch
					.and(StackReq::new::<I>(d_pattern.len() - d_pattern_start))
					.and(StackReq::new::<I>(d_pattern_mid_len));

				let d_ncols = d_end - d_start;

				d_scratch = d_scratch.and(spicy_matmul_scratch::<T>(
					d_pattern.len() - d_pattern_start,
					d_pattern_mid_len,
					d_ncols,
					true,
					false,
				));
				req = req.or(d_scratch);
			}
			req = req.or(linalg::cholesky::llt::factor::cholesky_in_place_scratch::<T>(s_ncols, par, params));
		}
		req.and(StackReq::new::<I>(n))
	}

	/// returns the size and alignment of the workspace required to compute the numeric
	/// cholesky $LDL^H$ factorization of a matrix $A$ with dimension `n`
	pub fn factorize_supernodal_numeric_ldlt_scratch<I: Index, T: ComplexField>(
		symbolic: &SymbolicSupernodalCholesky<I>,
		par: Par,
		params: Spec<LdltParams, T>,
	) -> StackReq {
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let row_idx = &*symbolic.row_idx;

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

				let d_pattern = &row_idx[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()];

				let d_ncols = d_end - d_start;

				let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
				let d_pattern_mid_len = d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));

				d_scratch = d_scratch
					.and(StackReq::new::<I>(d_pattern.len() - d_pattern_start))
					.and(StackReq::new::<I>(d_pattern_mid_len));

				d_scratch = d_scratch.and(spicy_matmul_scratch::<T>(
					d_pattern.len() - d_pattern_start,
					d_pattern_mid_len,
					d_ncols,
					true,
					true,
				));
				req = req.or(d_scratch);
			}
			req = req.or(linalg::cholesky::ldlt::factor::cholesky_in_place_scratch::<T>(s_ncols, par, params));
		}
		req.and(StackReq::new::<I>(n))
	}

	/// returns the size and alignment of the workspace required to compute the numeric
	/// cholesky $LBL^\top$ factorization with intranodal pivoting of a matrix $A$ with dimension
	/// `n`
	pub fn factorize_supernodal_numeric_intranode_lblt_scratch<I: Index, T: ComplexField>(
		symbolic: &SymbolicSupernodalCholesky<I>,
		par: Par,
		params: Spec<LbltParams, T>,
	) -> StackReq {
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let row_idx = &*symbolic.row_idx;

		let mut req = StackReq::empty();
		for s in 0..n_supernodes {
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();

			let s_ncols = s_end - s_start;
			let s_pattern = &row_idx[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()];

			let s_postordered = post_inv[s].zx();
			let desc_count = desc_count[s].zx();
			for d in &post[s_postordered - desc_count..s_postordered] {
				let mut d_scratch = StackReq::empty();

				let d = d.zx();
				let d_start = symbolic.supernode_begin[d].zx();
				let d_end = symbolic.supernode_begin[d + 1].zx();

				let d_pattern = &row_idx[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()];

				let d_ncols = d_end - d_start;

				let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
				let d_pattern_mid_len = d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));

				d_scratch = d_scratch.and(temp_mat_scratch::<T>(d_pattern.len() - d_pattern_start, d_pattern_mid_len));
				d_scratch = d_scratch.and(temp_mat_scratch::<T>(d_ncols, d_pattern_mid_len));
				req = req.or(d_scratch);
			}
			req = StackReq::any_of(&[
				req,
				linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<I, T>(s_ncols, par, params),
				crate::perm::permute_cols_in_place_scratch::<I, T>(s_pattern.len(), s_ncols),
			]);
		}
		req.and(StackReq::new::<I>(n))
	}

	/// computes the numeric values of the cholesky $LL^H$ factor of the matrix $A$, and stores them
	/// in `l_values`
	///
	/// # warning
	/// only the *lower* (not upper, unlike the other functions) triangular part of $A$ is
	/// accessed
	///
	/// # panics
	/// the symbolic structure must be computed by calling
	/// [`factorize_supernodal_symbolic_cholesky`] on a matrix with the same symbolic structure
	/// otherwise, the behavior is unspecified and panics may occur
	#[math]
	pub fn factorize_supernodal_numeric_llt<I: Index, T: ComplexField>(
		L_values: &mut [T],
		A_lower: SparseColMatRef<'_, I, T>,
		regularization: LltRegularization<T::Real>,
		symbolic: &SymbolicSupernodalCholesky<I>,
		par: Par,
		stack: &mut MemStack,
		params: Spec<LltParams, T>,
	) -> Result<LltInfo, LltError> {
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let mut dynamic_regularization_count = 0usize;
		L_values.fill(zero::<T>());

		assert!(A_lower.nrows() == n);
		assert!(A_lower.ncols() == n);
		assert!(L_values.len() == symbolic.len_val());

		let none = I::Signed::truncate(NONE);

		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let col_ptr_val = &*symbolic.col_ptr_for_val;
		let row_idx = &*symbolic.row_idx;

		// mapping from global indices to local
		let (global_to_local, stack) = unsafe { stack.make_raw::<I::Signed>(n) };
		global_to_local.fill(I::Signed::truncate(NONE));

		for s in 0..n_supernodes {
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();

			let s_pattern = &row_idx[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()];
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

				let d_pattern = &row_idx[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()];
				let d_ncols = d_end - d_start;
				let d_nrows = d_pattern.len() + d_ncols;

				let Ld = MatRef::from_column_major_slice(&head[col_ptr_val[d].zx()..col_ptr_val[d + 1].zx()], d_nrows, d_ncols);

				let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
				let d_pattern_mid_len = d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));

				let (_, Ld_mid_bot) = Ld.split_at_row(d_ncols);
				let (_, Ld_mid_bot) = Ld_mid_bot.split_at_row(d_pattern_start);
				let (Ld_mid, _) = Ld_mid_bot.split_at_row(d_pattern_mid_len);

				use linalg::matmul::triangular;
				let (row_idx, stack) = stack.make_with(Ld_mid_bot.nrows(), |i| {
					if i < d_pattern_mid_len {
						I::truncate(d_pattern[d_pattern_start + i].zx() - s_start)
					} else {
						I::from_signed(global_to_local[d_pattern[d_pattern_start + i].zx()])
					}
				});
				let (col_idx, stack) = stack.make_with(d_pattern_mid_len, |j| I::truncate(d_pattern[d_pattern_start + j].zx() - s_start));

				spicy_matmul(
					Ls.rb_mut(),
					triangular::BlockStructure::TriangularLower,
					Some(&row_idx),
					Some(&col_idx),
					Accum::Add,
					Ld_mid_bot,
					Conj::No,
					Ld_mid.transpose(),
					Conj::Yes,
					None,
					-one::<T>(),
					par,
					stack,
				);
			}

			let (mut Ls_top, mut Ls_bot) = Ls.rb_mut().split_at_row_mut(s_ncols);

			dynamic_regularization_count +=
				match linalg::cholesky::llt::factor::cholesky_in_place(Ls_top.rb_mut(), regularization.clone(), par, stack, params) {
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

	/// computes the numeric values of the cholesky $LDL^H$ factors of the matrix $A$, and stores
	/// them in `l_values`
	///
	/// # note
	/// only the *lower* (not upper, unlike the other functions) triangular part of $A$ is
	/// accessed
	///
	/// # panics
	/// the symbolic structure must be computed by calling
	/// [`factorize_supernodal_symbolic_cholesky`] on a matrix with the same symbolic structure
	/// otherwise, the behavior is unspecified and panics may occur
	#[math]
	pub fn factorize_supernodal_numeric_ldlt<I: Index, T: ComplexField>(
		L_values: &mut [T],
		A_lower: SparseColMatRef<'_, I, T>,
		regularization: LdltRegularization<'_, T::Real>,
		symbolic: &SymbolicSupernodalCholesky<I>,
		par: Par,
		stack: &mut MemStack,
		params: Spec<LdltParams, T>,
	) -> Result<LdltInfo, LdltError> {
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let mut dynamic_regularization_count = 0usize;
		L_values.fill(zero());

		assert!(A_lower.nrows() == n);
		assert!(A_lower.ncols() == n);
		assert!(L_values.len() == symbolic.len_val());

		let none = I::Signed::truncate(NONE);

		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let col_ptr_val = &*symbolic.col_ptr_for_val;
		let row_idx = &*symbolic.row_idx;

		// mapping from global indices to local
		let (global_to_local, stack) = unsafe { stack.make_raw::<I::Signed>(n) };
		global_to_local.fill(I::Signed::truncate(NONE));

		for s in 0..n_supernodes {
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();
			let s_pattern = if let Some(nnz_per_super) = symbolic.nnz_per_super.as_deref() {
				&row_idx[col_ptr_row[s].zx()..][..nnz_per_super[s].zx()]
			} else {
				&row_idx[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()]
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
					Ls[(ix, iy)] = Ls[(ix, iy)] + *val;
				}
			}

			let s_postordered = post_inv[s].zx();
			let desc_count = desc_count[s].zx();
			for d in &post[s_postordered - desc_count..s_postordered] {
				let d = d.zx();
				let d_start = symbolic.supernode_begin[d].zx();
				let d_end = symbolic.supernode_begin[d + 1].zx();
				let d_pattern = if let Some(nnz_per_super) = symbolic.nnz_per_super.as_deref() {
					&row_idx[col_ptr_row[d].zx()..][..nnz_per_super[d].zx()]
				} else {
					&row_idx[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()]
				};

				let d_ncols = d_end - d_start;
				let d_nrows = d_pattern.len() + d_ncols;

				let Ld = MatRef::from_column_major_slice(&head[col_ptr_val[d].zx()..col_ptr_val[d + 1].zx()], d_nrows, d_ncols);

				let d_pattern_start = d_pattern.partition_point(partition_fn(s_start));
				let d_pattern_mid_len = d_pattern[d_pattern_start..].partition_point(partition_fn(s_end));

				let (Ld_top, Ld_mid_bot) = Ld.split_at_row(d_ncols);
				let (_, Ld_mid_bot) = Ld_mid_bot.split_at_row(d_pattern_start);
				let (Ld_mid, _) = Ld_mid_bot.split_at_row(d_pattern_mid_len);
				let D = Ld_top.diagonal().column_vector();

				use linalg::matmul::triangular;
				let (row_idx, stack) = stack.make_with(Ld_mid_bot.nrows(), |i| {
					if i < d_pattern_mid_len {
						I::truncate(d_pattern[d_pattern_start + i].zx() - s_start)
					} else {
						I::from_signed(global_to_local[d_pattern[d_pattern_start + i].zx()])
					}
				});
				let (col_idx, stack) = stack.make_with(d_pattern_mid_len, |j| I::truncate(d_pattern[d_pattern_start + j].zx() - s_start));

				spicy_matmul(
					Ls.rb_mut(),
					triangular::BlockStructure::TriangularLower,
					Some(&row_idx),
					Some(&col_idx),
					Accum::Add,
					Ld_mid_bot,
					Conj::No,
					Ld_mid.transpose(),
					Conj::Yes,
					Some(D.as_diagonal()),
					-one::<T>(),
					par,
					stack,
				);
			}

			let (mut Ls_top, mut Ls_bot) = Ls.rb_mut().split_at_row_mut(s_ncols);

			dynamic_regularization_count += match linalg::cholesky::ldlt::factor::cholesky_in_place(
				Ls_top.rb_mut(),
				LdltRegularization {
					dynamic_regularization_signs: regularization.dynamic_regularization_signs.map(|signs| &signs[s_start..s_end]),
					..regularization.clone()
				},
				par,
				stack,
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
				let d = recip(real(Ls_top[(j, j)]));
				for i in 0..s_pattern.len() {
					Ls_bot[(i, j)] = mul_real(Ls_bot[(i, j)], d);
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

	/// computes the numeric values of the cholesky $LBL^\top$ factors of the matrix $A$ with
	/// intranodal pivoting, and stores them in `l_values`
	///
	/// # note
	/// only the *lower* (not upper, unlike the other functions) triangular part of $A$ is
	/// accessed
	///
	/// # panics
	/// the symbolic structure must be computed by calling
	/// [`factorize_supernodal_symbolic_cholesky`] on a matrix with the same symbolic structure
	/// otherwise, the behavior is unspecified and panics may occur
	#[math]
	pub fn factorize_supernodal_numeric_intranode_lblt<I: Index, T: ComplexField>(
		L_values: &mut [T],
		subdiag: &mut [T],
		perm_forward: &mut [I],
		perm_inverse: &mut [I],
		A_lower: SparseColMatRef<'_, I, T>,
		symbolic: &SymbolicSupernodalCholesky<I>,
		par: Par,
		stack: &mut MemStack,
		params: Spec<LbltParams, T>,
	) -> LbltInfo {
		let n_supernodes = symbolic.n_supernodes();
		let n = symbolic.nrows();
		let mut transposition_count = 0usize;
		L_values.fill(zero());

		assert!(A_lower.nrows() == n);
		assert!(A_lower.ncols() == n);
		assert!(perm_forward.len() == n);
		assert!(perm_inverse.len() == n);
		assert!(subdiag.len() == n);
		assert!(L_values.len() == symbolic.len_val());

		let none = I::Signed::truncate(NONE);

		let post = &*symbolic.supernode_postorder;
		let post_inv = &*symbolic.supernode_postorder_inv;

		let desc_count = &*symbolic.descendant_count;

		let col_ptr_row = &*symbolic.col_ptr_for_row_idx;
		let col_ptr_val = &*symbolic.col_ptr_for_val;
		let row_idx = &*symbolic.row_idx;

		// mapping from global indices to local
		let (global_to_local, stack) = unsafe { stack.make_raw::<I::Signed>(n) };
		global_to_local.fill(I::Signed::truncate(NONE));

		for s in 0..n_supernodes {
			let s_start = symbolic.supernode_begin[s].zx();
			let s_end = symbolic.supernode_begin[s + 1].zx();

			let s_pattern = &row_idx[col_ptr_row[s].zx()..col_ptr_row[s + 1].zx()];
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

				let d_pattern = &row_idx[col_ptr_row[d].zx()..col_ptr_row[d + 1].zx()];
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

				let (mut tmp, stack) = unsafe { temp_mat_uninit::<T, _, _>(Ld_mid_bot.nrows(), d_pattern_mid_len, stack) };
				let (mut tmp2, _) = unsafe { temp_mat_uninit::<T, _, _>(Ld_mid.ncols(), Ld_mid.nrows(), stack) };
				let tmp = tmp.as_mat_mut();
				let mut Ld_mid_x_D = tmp2.as_mat_mut().transpose_mut();

				let mut j = 0;
				while j < d_ncols {
					let subdiag = copy(d_subdiag[j]);
					if subdiag == zero::<T>() {
						let d = real(Ld_top[(j, j)]);
						for i in 0..d_pattern_mid_len {
							Ld_mid_x_D[(i, j)] = mul_real(Ld_mid[(i, j)], d);
						}
						j += 1;
					} else {
						let akp1k = subdiag;
						let ak = real(Ld_top[(j, j)]);
						let akp1 = real(Ld_top[(j + 1, j + 1)]);

						for i in 0..d_pattern_mid_len {
							let xk = copy(Ld_mid[(i, j)]);
							let xkp1 = copy(Ld_mid[(i, j + 1)]);

							Ld_mid_x_D[(i, j)] = mul_real(xk, ak) + xkp1 * akp1k;
							Ld_mid_x_D[(i, j + 1)] = mul_real(xkp1, akp1) + xk * conj(akp1k);
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
			let s_subdiag = &mut subdiag[s_start..s_end];

			let (info, perm) = linalg::cholesky::lblt::factor::cholesky_in_place(
				Ls_top.rb_mut(),
				ColMut::from_slice_mut(s_subdiag).as_diagonal_mut(),
				&mut perm_forward[s_start..s_end],
				&mut perm_inverse[s_start..s_end],
				par,
				stack,
				params,
			);
			transposition_count += info.transposition_count;
			z!(Ls_top.rb_mut()).for_each_triangular_upper(linalg::zip::Diag::Skip, |uz!(x)| *x = zero::<T>());

			crate::perm::permute_cols_in_place(Ls_bot.rb_mut(), perm.rb(), stack);

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
					let d = recip(real(Ls_top[(j, j)]));
					for i in 0..s_pattern.len() {
						Ls_bot[(i, j)] = mul_real(Ls_bot[(i, j)], d);
					}
					j += 1;
				} else {
					let akp1k = recip(conj(s_subdiag[j]));
					let ak = mul_real(conj(akp1k), real(Ls_top[(j, j)]));
					let akp1 = mul_real(akp1k, real(Ls_top[(j + 1, j + 1)]));

					let denom = recip(ak * akp1 - one::<T>());

					for i in 0..s_pattern.len() {
						let xk = Ls_bot[(i, j)] * conj(akp1k);
						let xkp1 = Ls_bot[(i, j + 1)] * akp1k;

						Ls_bot[(i, j)] = (akp1 * xk - xkp1) * denom;
						Ls_bot[(i, j + 1)] = (ak * xkp1 - xk) * denom;
					}
					j += 2;
				}
			}

			for &row in s_pattern {
				global_to_local[row.zx()] = none;
			}
		}
		LbltInfo { transposition_count }
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
		let current_child = (*first_child).sx();

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

/// tuning parameters for the symbolic cholesky factorization
#[derive(Copy, Clone, Debug, Default)]
pub struct CholeskySymbolicParams<'a> {
	/// parameters for computing the fill-reducing permutation
	pub amd_params: amd::Control,
	/// threshold for selecting the supernodal factorization
	pub supernodal_flop_ratio_threshold: SupernodalThreshold,
	/// supernodal factorization parameters
	pub supernodal_params: SymbolicSupernodalParams<'a>,
}

/// the inner factorization used for the symbolic cholesky, either simplicial or symbolic
#[derive(Debug)]
pub enum SymbolicCholeskyRaw<I> {
	/// simplicial structure
	Simplicial(simplicial::SymbolicSimplicialCholesky<I>),
	/// supernodal structure
	Supernodal(supernodal::SymbolicSupernodalCholesky<I>),
}

/// the symbolic structure of a sparse cholesky decomposition
#[derive(Debug)]
pub struct SymbolicCholesky<I> {
	raw: SymbolicCholeskyRaw<I>,
	perm_fwd: Option<alloc::vec::Vec<I>>,
	perm_inv: Option<alloc::vec::Vec<I>>,
	A_nnz: usize,
}

impl<I: Index> SymbolicCholesky<I> {
	/// returns the number of rows of the matrix
	#[inline]
	pub fn nrows(&self) -> usize {
		match &self.raw {
			SymbolicCholeskyRaw::Simplicial(this) => this.nrows(),
			SymbolicCholeskyRaw::Supernodal(this) => this.nrows(),
		}
	}

	/// returns the number of columns of the matrix
	#[inline]
	pub fn ncols(&self) -> usize {
		self.nrows()
	}

	/// returns the inner type of the factorization, either simplicial or symbolic
	#[inline]
	pub fn raw(&self) -> &SymbolicCholeskyRaw<I> {
		&self.raw
	}

	/// returns the permutation that was computed during symbolic analysis
	#[inline]
	pub fn perm(&self) -> Option<PermRef<'_, I>> {
		match (&self.perm_fwd, &self.perm_inv) {
			(Some(perm_fwd), Some(perm_inv)) => unsafe { Some(PermRef::new_unchecked(perm_fwd, perm_inv, self.ncols())) },
			_ => None,
		}
	}

	/// returns the length of the slice needed to store the numerical values of the cholesky
	/// decomposition
	#[inline]
	pub fn len_val(&self) -> usize {
		match &self.raw {
			SymbolicCholeskyRaw::Simplicial(this) => this.len_val(),
			SymbolicCholeskyRaw::Supernodal(this) => this.len_val(),
		}
	}

	/// computes the required workspace size and alignment for a numerical $LL^H$ factorization
	#[inline]
	pub fn factorize_numeric_llt_scratch<T: ComplexField>(&self, par: Par, params: Spec<LltParams, T>) -> StackReq {
		let n = self.nrows();
		let A_nnz = self.A_nnz;

		let n_scratch = StackReq::new::<I>(n);
		let A_scratch = StackReq::all_of(&[temp_mat_scratch::<T>(A_nnz, 1), StackReq::new::<I>(n + 1), StackReq::new::<I>(A_nnz)]);
		let permute_scratch = n_scratch;

		let factor_scratch = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => simplicial::factorize_simplicial_numeric_llt_scratch::<I, T>(n),
			SymbolicCholeskyRaw::Supernodal(this) => supernodal::factorize_supernodal_numeric_llt_scratch::<I, T>(this, par, params),
		};

		StackReq::all_of(&[A_scratch, StackReq::or(permute_scratch, factor_scratch)])
	}

	/// computes the required workspace size and alignment for a numerical $LDL^H$ factorization
	#[inline]
	pub fn factorize_numeric_ldlt_scratch<T: ComplexField>(&self, par: Par, params: Spec<LdltParams, T>) -> StackReq {
		let n = self.nrows();
		let A_nnz = self.A_nnz;

		let regularization_signs = StackReq::new::<i8>(n);

		let n_scratch = StackReq::new::<I>(n);
		let A_scratch = StackReq::all_of(&[temp_mat_scratch::<T>(A_nnz, 1), StackReq::new::<I>(n + 1), StackReq::new::<I>(A_nnz)]);
		let permute_scratch = n_scratch;

		let factor_scratch = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => simplicial::factorize_simplicial_numeric_ldlt_scratch::<I, T>(n),
			SymbolicCholeskyRaw::Supernodal(this) => supernodal::factorize_supernodal_numeric_ldlt_scratch::<I, T>(this, par, params),
		};

		StackReq::all_of(&[regularization_signs, A_scratch, StackReq::or(permute_scratch, factor_scratch)])
	}

	/// computes the required workspace size and alignment for a numerical intranodal $LBL^\top$
	/// factorization
	#[inline]
	pub fn factorize_numeric_intranode_lblt_scratch<T: ComplexField>(&self, par: Par, params: Spec<LbltParams, T>) -> StackReq {
		let n = self.nrows();
		let A_nnz = self.A_nnz;

		let regularization_signs = StackReq::new::<i8>(n);

		let n_scratch = StackReq::new::<I>(n);
		let A_scratch = StackReq::all_of(&[temp_mat_scratch::<T>(A_nnz, 1), StackReq::new::<I>(n + 1), StackReq::new::<I>(A_nnz)]);
		let permute_scratch = n_scratch;

		let factor_scratch = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => simplicial::factorize_simplicial_numeric_ldlt_scratch::<I, T>(n),
			SymbolicCholeskyRaw::Supernodal(this) => supernodal::factorize_supernodal_numeric_intranode_lblt_scratch::<I, T>(this, par, params),
		};

		StackReq::all_of(&[regularization_signs, A_scratch, StackReq::or(permute_scratch, factor_scratch)])
	}

	/// computes a numerical llt factorization of a, or returns a [`LltError`] if the matrix
	/// is not numerically positive definite
	#[track_caller]
	pub fn factorize_numeric_llt<'out, T: ComplexField>(
		&'out self,
		L_values: &'out mut [T],
		A: SparseColMatRef<'_, I, T>,
		side: Side,
		regularization: LltRegularization<T::Real>,
		par: Par,
		stack: &mut MemStack,
		params: Spec<LltParams, T>,
	) -> Result<LltRef<'out, I, T>, LltError> {
		assert!(A.nrows() == A.ncols());
		let n = A.nrows();
		with_dim!(N, n);

		let A_nnz = self.A_nnz;
		let A = A.as_shape(N, N);

		let (mut new_values, stack) = unsafe { temp_mat_uninit::<T, _, _>(A_nnz, 1, stack) };
		let new_values = new_values.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap().as_slice_mut();
		let (new_col_ptr, stack) = unsafe { stack.make_raw::<I>(n + 1) };
		let (new_row_idx, stack) = unsafe { stack.make_raw::<I>(A_nnz) };

		let out_side = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => Side::Upper,
			SymbolicCholeskyRaw::Supernodal(_) => Side::Lower,
		};

		let A = match self.perm() {
			Some(perm) => {
				let perm = perm.as_shape(N);
				permute_self_adjoint_to_unsorted(new_values, new_col_ptr, new_row_idx, A, perm, side, out_side, stack).into_const()
			},
			None => {
				if side == out_side {
					A
				} else {
					adjoint(new_values, new_col_ptr, new_row_idx, A, stack).into_const()
				}
			},
		};

		match &self.raw {
			SymbolicCholeskyRaw::Simplicial(this) => {
				simplicial::factorize_simplicial_numeric_llt(L_values, A.as_dyn().into_const(), regularization, this, stack)?;
			},
			SymbolicCholeskyRaw::Supernodal(this) => {
				supernodal::factorize_supernodal_numeric_llt(L_values, A.as_dyn().into_const(), regularization, this, par, stack, params)?;
			},
		}
		Ok(LltRef::<'out, I, T>::new(self, L_values))
	}

	/// computes a numerical $LDL^H$ factorization of a
	#[inline]
	pub fn factorize_numeric_ldlt<'out, T: ComplexField>(
		&'out self,
		L_values: &'out mut [T],
		A: SparseColMatRef<'_, I, T>,
		side: Side,
		regularization: LdltRegularization<'_, T::Real>,
		par: Par,
		stack: &mut MemStack,
		params: Spec<LdltParams, T>,
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
		let (new_row_idx, stack) = unsafe { stack.make_raw::<I>(A_nnz) };

		let out_side = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => Side::Upper,
			SymbolicCholeskyRaw::Supernodal(_) => Side::Lower,
		};

		let (A, signs) = match self.perm() {
			Some(perm) => {
				let perm = perm.as_shape(N);
				let A = permute_self_adjoint_to_unsorted(new_values, new_col_ptr, new_row_idx, A, perm, side, out_side, stack).into_const();
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
						adjoint(new_values, new_col_ptr, new_row_idx, A, stack).into_const(),
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
				supernodal::factorize_supernodal_numeric_ldlt(L_values, A.as_dyn().into_const(), regularization, this, par, stack, params)?;
			},
		}

		Ok(LdltRef::<'out, I, T>::new(self, L_values))
	}

	/// computes a numerical intranodal $LBL^\top$ factorization of a
	#[inline]
	pub fn factorize_numeric_intranode_lblt<'out, T: ComplexField>(
		&'out self,
		L_values: &'out mut [T],
		subdiag: &'out mut [T],
		perm_forward: &'out mut [I],
		perm_inverse: &'out mut [I],
		A: SparseColMatRef<'_, I, T>,
		side: Side,
		par: Par,
		stack: &mut MemStack,
		params: Spec<LbltParams, T>,
	) -> IntranodeLbltRef<'out, I, T> {
		assert!(A.nrows() == A.ncols());
		let n = A.nrows();

		with_dim!(N, n);
		let A_nnz = self.A_nnz;
		let A = A.as_shape(N, N);

		let (mut new_values, stack) = unsafe { temp_mat_uninit::<T, _, _>(A_nnz, 1, stack) };
		let new_values = new_values.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap().as_slice_mut();
		let (new_col_ptr, stack) = unsafe { stack.make_raw::<I>(n + 1) };
		let (new_row_idx, stack) = unsafe { stack.make_raw::<I>(A_nnz) };

		let out_side = match &self.raw {
			SymbolicCholeskyRaw::Simplicial(_) => Side::Upper,
			SymbolicCholeskyRaw::Supernodal(_) => Side::Lower,
		};

		let A = match self.perm() {
			Some(perm) => {
				let perm = perm.as_shape(N);
				let A = permute_self_adjoint_to_unsorted(new_values, new_col_ptr, new_row_idx, A, perm, side, out_side, stack).into_const();

				A
			},
			None => {
				if side == out_side {
					A
				} else {
					adjoint(new_values, new_col_ptr, new_row_idx, A, stack).into_const()
				}
			},
		};

		match &self.raw {
			SymbolicCholeskyRaw::Simplicial(this) => {
				let regularization = LdltRegularization::default();
				for (i, p) in perm_forward.iter_mut().enumerate() {
					*p = I::truncate(i);
				}
				for (i, p) in perm_inverse.iter_mut().enumerate() {
					*p = I::truncate(i);
				}
				let _ = simplicial::factorize_simplicial_numeric_ldlt(L_values, A.as_dyn().into_const(), regularization, this, stack);
			},
			SymbolicCholeskyRaw::Supernodal(this) => {
				supernodal::factorize_supernodal_numeric_intranode_lblt(
					L_values,
					subdiag,
					perm_forward,
					perm_inverse,
					A.as_dyn().into_const(),
					this,
					par,
					stack,
					params,
				);
			},
		}

		IntranodeLbltRef::<'out, I, T>::new(self, L_values, subdiag, unsafe {
			PermRef::<'out, I>::new_unchecked(perm_forward, perm_inverse, n)
		})
	}

	/// computes the required workspace size and alignment for a dense solve in place using an
	/// $LL^H$, $LDL^H$ or intranodal $LBL^\top$ factorization
	pub fn solve_in_place_scratch<T: ComplexField>(&self, rhs_ncols: usize, par: Par) -> StackReq {
		temp_mat_scratch::<T>(self.nrows(), rhs_ncols).and(match self.raw() {
			SymbolicCholeskyRaw::Simplicial(this) => this.solve_in_place_scratch::<T>(rhs_ncols),
			SymbolicCholeskyRaw::Supernodal(this) => this.solve_in_place_scratch::<T>(rhs_ncols, par),
		})
	}
}

/// sparse $LL^H$ factorization wrapper
#[derive(Debug)]
pub struct LltRef<'a, I: Index, T> {
	symbolic: &'a SymbolicCholesky<I>,
	values: &'a [T],
}

/// sparse $LDL^H$ factorization wrapper
#[derive(Debug)]
pub struct LdltRef<'a, I: Index, T> {
	symbolic: &'a SymbolicCholesky<I>,
	values: &'a [T],
}

/// sparse intranodal $LBL^\top$ factorization wrapper
#[derive(Debug)]
pub struct IntranodeLbltRef<'a, I: Index, T> {
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
impl<'a, I: Index, T> core::ops::Deref for IntranodeLbltRef<'a, I, T> {
	type Target = SymbolicCholesky<I>;

	#[inline]
	fn deref(&self) -> &Self::Target {
		self.symbolic
	}
}

impl<'a, I: Index, T> Copy for LltRef<'a, I, T> {}
impl<'a, I: Index, T> Copy for LdltRef<'a, I, T> {}
impl<'a, I: Index, T> Copy for IntranodeLbltRef<'a, I, T> {}

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
impl<'a, I: Index, T> Clone for IntranodeLbltRef<'a, I, T> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<'a, I: Index, T> IntranodeLbltRef<'a, I, T> {
	/// creates a new cholesky intranodal $LBL^\top$ factor from the symbolic part and
	/// numerical values, as well as the pivoting permutation
	///
	/// # panics
	/// - panics if `values.len() != symbolic.len_val()`
	/// - panics if `subdiag.len() != symbolic.nrows()`
	/// - panics if `perm.len() != symbolic.nrows()`
	#[inline]
	pub fn new(symbolic: &'a SymbolicCholesky<I>, values: &'a [T], subdiag: &'a [T], perm: PermRef<'a, I>) -> Self {
		assert!(all(
			values.len() == symbolic.len_val(),
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

	/// returns the symbolic part of the cholesky factor
	#[inline]
	pub fn symbolic(self) -> &'a SymbolicCholesky<I> {
		self.symbolic
	}

	/// solves the equation $A x = \text{rhs}$ and stores the result in `rhs`, implicitly
	/// conjugating $A$ if needed
	///
	/// # panics
	/// panics if `rhs.nrows() != self.symbolic().nrows()`
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
							x[(i, j)] = copy(&rhs[(fwd.zx(), j)]);
						}
					}
				}
				this.solve_in_place_with_conj(conj, if self.symbolic.perm().is_some() { x.rb_mut() } else { rhs.rb_mut() }, par, stack);
				if let Some(perm) = self.symbolic.perm() {
					for j in 0..k {
						for (i, inv) in perm.arrays().1.iter().enumerate() {
							rhs[(i, j)] = copy(&x[(inv.zx(), j)]);
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
							x[(i, j)] = copy(&rhs[(fwd[dyn_fwd.zx()].zx(), j)]);
						}
					}
				} else {
					for j in 0..k {
						for (i, dyn_fwd) in dyn_fwd.iter().enumerate() {
							x[(i, j)] = copy(&rhs[(dyn_fwd.zx(), j)]);
						}
					}
				}

				let this = supernodal::SupernodalIntranodeLbltRef::new(symbolic, self.values, self.subdiag, self.perm);
				this.solve_in_place_no_numeric_permute_with_conj(conj, x.rb_mut(), par, stack);

				if let Some(inv) = inv {
					for j in 0..k {
						for (i, inv) in inv.iter().enumerate() {
							rhs[(i, j)] = copy(&x[(dyn_inv[inv.zx()].zx(), j)]);
						}
					}
				} else {
					for j in 0..k {
						for (i, dyn_inv) in dyn_inv.iter().enumerate() {
							rhs[(i, j)] = copy(&x[(dyn_inv.zx(), j)]);
						}
					}
				}
			},
		}
	}
}

impl<'a, I: Index, T> LltRef<'a, I, T> {
	/// creates a new cholesky $LL^H$ factor from the symbolic part and
	/// numerical values
	///
	/// # panics
	/// - panics if `values.len() != symbolic.len_val()`
	#[inline]
	pub fn new(symbolic: &'a SymbolicCholesky<I>, values: &'a [T]) -> Self {
		assert!(symbolic.len_val() == values.len());
		Self { symbolic, values }
	}

	/// returns the symbolic part of the cholesky factor
	#[inline]
	pub fn symbolic(self) -> &'a SymbolicCholesky<I> {
		self.symbolic
	}

	/// solves the equation $A x = \text{rhs}$ and stores the result in `rhs`, implicitly
	/// conjugating $A$ if needed
	///
	/// # panics
	/// panics if `rhs.nrows() != self.symbolic().nrows()`
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
					x[(i, j)] = copy(&rhs[(fwd.zx(), j)]);
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
					rhs[(i, j)] = copy(&x[(inv.zx(), j)]);
				}
			}
		}
	}
}

impl<'a, I: Index, T> LdltRef<'a, I, T> {
	/// creates new cholesky $LDL^H$ factors from the symbolic part and
	/// numerical values
	///
	/// # panics
	/// - panics if `values.len() != symbolic.len_val()`
	#[inline]
	pub fn new(symbolic: &'a SymbolicCholesky<I>, values: &'a [T]) -> Self {
		assert!(symbolic.len_val() == values.len());
		Self { symbolic, values }
	}

	/// returns the symbolic part of the cholesky factor
	#[inline]
	pub fn symbolic(self) -> &'a SymbolicCholesky<I> {
		self.symbolic
	}

	/// solves the equation $A x = \text{rhs}$ and stores the result in `rhs`, implicitly
	/// conjugating $A$ if needed
	///
	/// # panics
	/// panics if `rhs.nrows() != self.symbolic().nrows()`
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
					x[(i, j)] = copy(&rhs[(fwd.zx(), j)]);
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
					rhs[(i, j)] = copy(&x[(inv.zx(), j)]);
				}
			}
		}
	}
}

/// computes the symbolic cholesky factorization of the matrix $A$, or returns an error if the
/// operation could not be completed
pub fn factorize_symbolic_cholesky<I: Index>(
	A: SymbolicSparseColMatRef<'_, I>,
	side: Side,
	ord: SymmetricOrdering<'_, I>,
	params: CholeskySymbolicParams<'_>,
) -> Result<SymbolicCholesky<I>, FaerError> {
	let n = A.nrows();
	let A_nnz = A.compute_nnz();

	assert!(A.nrows() == A.ncols());

	with_dim!(N, n);
	let A = A.as_shape(N, N);

	let req = {
		let n_scratch = StackReq::new::<I>(n);
		let A_scratch = StackReq::and(
			// new_col_ptr
			StackReq::new::<I>(n + 1),
			// new_row_idx
			StackReq::new::<I>(A_nnz),
		);

		StackReq::or(
			match ord {
				SymmetricOrdering::Amd => amd::order_maybe_unsorted_scratch::<I>(n, A_nnz),
				_ => StackReq::empty(),
			},
			StackReq::all_of(&[
				A_scratch,
				// permute_symmetric | etree
				n_scratch,
				// col_counts
				n_scratch,
				// ghost_prefactorize_symbolic
				n_scratch,
				// ghost_factorize_*_symbolic
				StackReq::or(
					supernodal::factorize_supernodal_symbolic_cholesky_scratch::<I>(n),
					simplicial::factorize_simplicial_symbolic_cholesky_scratch::<I>(n),
				),
			]),
		)
	};

	let mut mem = dyn_stack::MemBuffer::try_new(req).ok().ok_or(FaerError::OutOfMemory)?;
	let stack = MemStack::new(&mut mem);

	let mut perm_fwd = match ord {
		SymmetricOrdering::Identity => None,
		_ => Some(try_zeroed(n)?),
	};
	let mut perm_inv = match ord {
		SymmetricOrdering::Identity => None,
		_ => Some(try_zeroed(n)?),
	};
	let flops = match ord {
		SymmetricOrdering::Amd => Some(amd::order_maybe_unsorted(
			perm_fwd.as_mut().unwrap(),
			perm_inv.as_mut().unwrap(),
			A.as_dyn(),
			params.amd_params,
			stack,
		)?),
		SymmetricOrdering::Identity => None,
		SymmetricOrdering::Custom(perm) => {
			let (fwd, inv) = perm.arrays();
			perm_fwd.as_mut().unwrap().copy_from_slice(fwd);
			perm_inv.as_mut().unwrap().copy_from_slice(inv);
			None
		},
	};

	let (new_col_ptr, stack) = unsafe { stack.make_raw::<I>(n + 1) };
	let (new_row_idx, stack) = unsafe { stack.make_raw::<I>(A_nnz) };
	let A = match ord {
		SymmetricOrdering::Identity => A,
		_ => permute_self_adjoint_to_unsorted(
			Symbolic::materialize(A_nnz),
			new_col_ptr,
			new_row_idx,
			SparseColMatRef::new(A, Symbolic::materialize(A.row_idx().len())),
			PermRef::new_checked(perm_fwd.as_ref().unwrap(), perm_inv.as_ref().unwrap(), n).as_shape(N),
			side,
			Side::Upper,
			stack,
		)
		.symbolic(),
	};

	let (etree, stack) = unsafe { stack.make_raw::<I::Signed>(n) };
	let (col_counts, stack) = unsafe { stack.make_raw::<I>(n) };
	let etree = simplicial::prefactorize_symbolic_cholesky::<I>(etree, col_counts, A.as_shape(n, n), stack);
	let L_nnz = I::sum_nonnegative(col_counts.as_ref()).ok_or(FaerError::IndexOverflow)?;

	let col_counts = Array::from_mut(col_counts, N);
	let flops = match flops {
		Some(flops) => flops,
		None => {
			let mut n_div = 0u128;
			let mut n_mult_subs_ldl = 0u128;
			for i in N.indices() {
				let c = col_counts[i].zx();
				n_div += c as u128;
				n_mult_subs_ldl += (c as u128 * (c as u128 + 1)) / 2;
			}
			amd::FlopCount {
				n_div: n_div as f64,
				n_mult_subs_ldl: n_mult_subs_ldl as f64,
				n_mult_subs_lu: 0.0,
			}
		},
	};

	let flops = flops.n_div + flops.n_mult_subs_ldl;
	let raw = if (flops / L_nnz.zx() as f64) > params.supernodal_flop_ratio_threshold.0 * crate::sparse::linalg::CHOLESKY_SUPERNODAL_RATIO_FACTOR {
		SymbolicCholeskyRaw::Supernodal(supernodal::ghost_factorize_supernodal_symbolic(
			A,
			None,
			None,
			supernodal::CholeskyInput::A,
			etree.as_bound(N),
			col_counts,
			stack,
			params.supernodal_params,
		)?)
	} else {
		SymbolicCholeskyRaw::Simplicial(simplicial::ghost_factorize_simplicial_symbolic_cholesky(
			A,
			etree.as_bound(N),
			col_counts,
			stack,
		)?)
	};

	Ok(SymbolicCholesky {
		raw,
		perm_fwd,
		perm_inv,
		A_nnz,
	})
}

#[cfg(test)]
pub(super) mod tests {
	use super::*;
	use crate::assert;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use dyn_stack::MemBuffer;
	use matrix_market_rs::MtxData;
	use std::path::PathBuf;
	use std::str::FromStr;

	type Error = Box<dyn std::error::Error>;
	type Result<T = (), E = Error> = core::result::Result<T, E>;

	pub(crate) fn load_mtx<I: Index>(data: MtxData<f64>) -> (usize, usize, Vec<I>, Vec<I>, Vec<f64>) {
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

		let mut row_idx = vec![I(0); nnz];
		let mut values = vec![0.0; nnz];

		col_counts.copy_from_slice(&col_ptr[..n]);

		for (&[i, j], &val) in iter::zip(&coo_indices, &coo_values) {
			values[col_counts[j].zx()] = val;
			row_idx[col_counts[j].zx()] = I(i);
			col_counts[j] += I(1);
		}

		(m, n, col_ptr, row_idx, values)
	}

	#[track_caller]
	pub(crate) fn parse_vec<F: FromStr>(text: &str) -> (Vec<F>, &str) {
		let mut text = text;
		let mut out = Vec::new();

		assert!(text.trim().starts_with('['));
		text = &text.trim()[1..];
		while !text.trim().starts_with(']') {
			let i = text.find(',').unwrap();
			let num = &text[..i];

			let num = num.trim().parse::<F>().ok().unwrap();
			out.push(num);
			text = &text[i + 1..];
		}

		assert!(text.trim().starts_with("],"));
		text = &text.trim()[2..];

		(out, text)
	}

	pub(crate) fn parse_csc_symbolic(text: &str) -> (SymbolicSparseColMat<usize>, &str) {
		let (col_ptr, text) = parse_vec::<usize>(text);
		let (row_idx, text) = parse_vec::<usize>(text);
		let n = col_ptr.len() - 1;

		(SymbolicSparseColMat::new_unsorted_checked(n, n, col_ptr, None, row_idx), text)
	}

	pub(crate) fn parse_csc<T: FromStr>(text: &str) -> (SparseColMat<usize, T>, &str) {
		let (symbolic, text) = parse_csc_symbolic(text);
		let (numeric, text) = parse_vec::<T>(text);
		(SparseColMat::new(symbolic, numeric), text)
	}

	#[test]
	fn test_counts() {
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

		let A = SymbolicSparseColMatRef::new_unsorted_checked(n, n, col_ptr, None, row_idx);
		let mut etree = vec![0isize; n];
		let mut col_count = vec![0usize; n];

		simplicial::prefactorize_symbolic_cholesky(
			&mut etree,
			&mut col_count,
			A,
			MemStack::new(&mut MemBuffer::new(StackReq::new::<usize>(n))),
		);

		assert!(etree == [5, 2, 7, 5, 7, 6, 8, 9, 9, 10, NONE as isize]);
		assert!(col_count == [3, 3, 4, 3, 3, 4, 4, 3, 3, 2, 1usize]);
	}

	#[test]
	fn test_amd() -> Result {
		for file in [
			PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_cholesky/small.txt"),
			PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_cholesky/medium-0.txt"),
			PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_cholesky/medium-1.txt"),
		] {
			let (A, _) = parse_csc_symbolic(&std::fs::read_to_string(&file)?);
			let n = A.nrows();

			let (target_fwd, target_bwd, _) = ::amd::order(A.nrows(), A.col_ptr(), A.row_idx(), &::amd::Control::default()).unwrap();

			let fwd = &mut *vec![0usize; n];
			let bwd = &mut *vec![0usize; n];
			amd::order_maybe_unsorted(
				fwd,
				bwd,
				A.rb(),
				amd::Control::default(),
				MemStack::new(&mut MemBuffer::new(amd::order_maybe_unsorted_scratch::<usize>(n, A.compute_nnz()))),
			)?;

			assert!(fwd == &target_fwd);
			assert!(bwd == &target_bwd);
		}
		Ok(())
	}

	fn reconstruct_from_supernodal_ldlt<I: Index, T: ComplexField>(symbolic: &supernodal::SymbolicSupernodalCholesky<I>, L_values: &[T]) -> Mat<T> {
		let ldlt = supernodal::SupernodalLdltRef::new(symbolic, L_values);
		let n_supernodes = ldlt.symbolic().n_supernodes();
		let n = ldlt.symbolic().nrows();

		let mut dense = Mat::<T>::zeros(n, n);

		for s in 0..n_supernodes {
			let s = ldlt.supernode(s);
			let node = s.val();
			let size = node.ncols();

			let (Ls_top, Ls_bot) = node.split_at_row(size);
			dense
				.rb_mut()
				.submatrix_mut(s.start(), s.start(), size, size)
				.copy_from_triangular_lower(Ls_top);

			for col in 0..size {
				for (i, &row) in s.pattern().iter().enumerate() {
					dense[(row.zx(), s.start() + col)] = Ls_bot[(i, col)].clone();
				}
			}
		}
		let mut D = Col::<T>::zeros(n);
		D.copy_from(dense.rb().diagonal().column_vector());
		dense.rb_mut().diagonal_mut().fill(one::<T>());

		&dense * D.as_diagonal() * dense.adjoint()
	}

	pub(crate) fn reconstruct_from_supernodal_llt<I: Index, T: ComplexField>(
		symbolic: &supernodal::SymbolicSupernodalCholesky<I>,
		L_values: &[T],
	) -> Mat<T> {
		let llt = supernodal::SupernodalLltRef::new(symbolic, L_values);
		let n_supernodes = llt.symbolic().n_supernodes();
		let n = llt.symbolic().nrows();

		let mut dense = Mat::<T>::zeros(n, n);

		for s in 0..n_supernodes {
			let s = llt.supernode(s);
			let node = s.val();
			let size = node.ncols();

			let (Ls_top, Ls_bot) = node.split_at_row(size);
			dense
				.rb_mut()
				.submatrix_mut(s.start(), s.start(), size, size)
				.copy_from_triangular_lower(Ls_top);

			for col in 0..size {
				for (i, &row) in s.pattern().iter().enumerate() {
					dense[(row.zx(), s.start() + col)] = Ls_bot[(i, col)].clone();
				}
			}
		}
		&dense * dense.adjoint()
	}
	fn reconstruct_from_simplicial_ldlt<I: Index, T: ComplexField>(symbolic: &simplicial::SymbolicSimplicialCholesky<I>, L_values: &[T]) -> Mat<T> {
		let n = symbolic.nrows();

		let mut dense = SparseColMatRef::new(symbolic.factor(), L_values).to_dense();
		let mut D = Col::<T>::zeros(n);
		D.copy_from(dense.rb().diagonal().column_vector());
		dense.rb_mut().diagonal_mut().fill(one::<T>());

		&dense * D.as_diagonal() * dense.adjoint()
	}

	fn reconstruct_from_simplicial_llt<I: Index, T: ComplexField>(symbolic: &simplicial::SymbolicSimplicialCholesky<I>, L_values: &[T]) -> Mat<T> {
		let dense = SparseColMatRef::new(symbolic.factor(), L_values).to_dense();
		&dense * dense.adjoint()
	}

	#[test]
	fn test_supernodal() -> Result {
		let file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_cholesky/medium-1.txt");
		let A_upper = parse_csc::<c64>(&std::fs::read_to_string(&file)?).0;
		let mut A_lower = A_upper.adjoint().to_col_major()?;
		let A_upper = A_upper.rb();

		let n = A_upper.nrows();
		let etree = &mut *vec![0isize; n];
		let col_counts = &mut *vec![0usize; n];
		let etree = simplicial::prefactorize_symbolic_cholesky(
			etree,
			col_counts,
			A_upper.symbolic(),
			MemStack::new(&mut MemBuffer::new(simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(
				n,
				A_upper.compute_nnz(),
			))),
		);

		let symbolic = &supernodal::factorize_supernodal_symbolic_cholesky(
			A_upper.symbolic(),
			etree,
			col_counts,
			MemStack::new(&mut MemBuffer::new(supernodal::factorize_supernodal_symbolic_cholesky_scratch::<usize>(
				n,
			))),
			Default::default(),
		)?;

		{
			let A_lower = A_lower.rb();
			let approx_eq = CwiseMat(ApproxEq::eps() * 1e5);
			let L_val = &mut *vec![zero::<c64>(); symbolic.len_val()];
			supernodal::factorize_supernodal_numeric_ldlt(
				L_val,
				A_lower,
				Default::default(),
				symbolic,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(supernodal::factorize_supernodal_numeric_ldlt_scratch::<usize, c64>(
					symbolic,
					Par::Seq,
					Default::default(),
				))),
				Default::default(),
			)?;

			let mut target = A_lower.to_dense();
			let adjoint = target.adjoint().to_owned();
			target.copy_from_strict_triangular_upper(adjoint);
			let A = reconstruct_from_supernodal_ldlt(symbolic, L_val);

			assert!(A ~ target);

			let k = 3;
			let rng = &mut StdRng::seed_from_u64(0);

			let rhs = CwiseMatDistribution {
				nrows: n,
				ncols: k,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			let supernodal = supernodal::SupernodalLdltRef::new(symbolic, L_val);
			for conj in [Conj::No, Conj::Yes] {
				let mut x = rhs.clone();
				supernodal.solve_in_place_with_conj(
					conj,
					x.rb_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(symbolic.solve_in_place_scratch::<c64>(k, Par::Seq))),
				);

				let target = rhs.rb();
				let rhs = match conj {
					Conj::No => &A * &x,
					Conj::Yes => A.conjugate() * &x,
				};

				assert!(rhs ~ target);
			}
		}

		{
			let A_lower = A_lower.rb();
			let approx_eq = CwiseMat(ApproxEq::eps() * 1e2);
			let L_val = &mut *vec![zero::<c64>(); symbolic.len_val()];
			let fwd = &mut *vec![0usize; n];
			let bwd = &mut *vec![0usize; n];
			let subdiag = &mut *vec![zero::<c64>(); n];

			supernodal::factorize_supernodal_numeric_intranode_lblt(
				L_val,
				subdiag,
				fwd,
				bwd,
				A_lower,
				symbolic,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(supernodal::factorize_supernodal_numeric_intranode_lblt_scratch::<
					usize,
					c64,
				>(symbolic, Par::Seq, Default::default()))),
				Default::default(),
			);

			let mut A = A_lower.to_dense();
			let adjoint = A.adjoint().to_owned();
			A.copy_from_strict_triangular_upper(adjoint);

			let k = 3;
			let rng = &mut StdRng::seed_from_u64(0);

			let rhs = CwiseMatDistribution {
				nrows: n,
				ncols: k,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			let supernodal = supernodal::SupernodalIntranodeLbltRef::new(symbolic, L_val, subdiag, PermRef::new_checked(fwd, bwd, n));
			for conj in [Conj::No, Conj::Yes] {
				let mut x = rhs.clone();
				let mut tmp = x.clone();

				for j in 0..k {
					for (i, &fwd) in fwd.iter().enumerate() {
						tmp[(i, j)] = x[(fwd, j)];
					}
				}

				supernodal.solve_in_place_no_numeric_permute_with_conj(
					conj,
					tmp.rb_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(symbolic.solve_in_place_scratch::<c64>(k, Par::Seq))),
				);

				for j in 0..k {
					for (i, &bwd) in bwd.iter().enumerate() {
						x[(i, j)] = tmp[(bwd, j)];
					}
				}

				let target = rhs.rb();
				let rhs = match conj {
					Conj::No => &A * &x,
					Conj::Yes => A.conjugate() * &x,
				};

				assert!(rhs ~ target);
			}
		}

		{
			for j in 0..n {
				*A_lower.val_of_col_mut(j).first_mut().unwrap() *= 1e3;
			}
			let A_lower = A_lower.rb();

			let approx_eq = CwiseMat(ApproxEq::eps() * 1e5);
			let L_val = &mut *vec![zero::<c64>(); symbolic.len_val()];
			supernodal::factorize_supernodal_numeric_llt(
				L_val,
				A_lower,
				Default::default(),
				symbolic,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(supernodal::factorize_supernodal_numeric_llt_scratch::<usize, c64>(
					symbolic,
					Par::Seq,
					Default::default(),
				))),
				Default::default(),
			)?;

			let mut target = A_lower.to_dense();
			let adjoint = target.adjoint().to_owned();
			target.copy_from_strict_triangular_upper(adjoint);
			let A = reconstruct_from_supernodal_llt(symbolic, L_val);

			assert!(A ~ target);

			let k = 3;
			let rng = &mut StdRng::seed_from_u64(0);

			let rhs = CwiseMatDistribution {
				nrows: n,
				ncols: k,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			let supernodal = supernodal::SupernodalLltRef::new(symbolic, L_val);
			for conj in [Conj::No, Conj::Yes] {
				let mut x = rhs.clone();
				supernodal.solve_in_place_with_conj(
					conj,
					x.rb_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(symbolic.solve_in_place_scratch::<c64>(k, Par::Seq))),
				);

				let target = rhs.rb();
				let rhs = match conj {
					Conj::No => &A * &x,
					Conj::Yes => A.conjugate() * &x,
				};

				assert!(rhs ~ target);
			}
		}
		Ok(())
	}

	#[test]
	fn test_simplicial() -> Result {
		let file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_cholesky/medium-1.txt");
		let mut A_upper = parse_csc::<c64>(&std::fs::read_to_string(&file)?).0;

		let n = A_upper.nrows();
		let etree = &mut *vec![0isize; n];
		let col_counts = &mut *vec![0usize; n];
		let etree = simplicial::prefactorize_symbolic_cholesky(
			etree,
			col_counts,
			A_upper.symbolic(),
			MemStack::new(&mut MemBuffer::new(simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(
				n,
				A_upper.compute_nnz(),
			))),
		);

		let symbolic = &simplicial::factorize_simplicial_symbolic_cholesky(
			A_upper.symbolic(),
			etree,
			col_counts,
			MemStack::new(&mut MemBuffer::new(simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(
				n,
			))),
		)?;

		{
			let approx_eq = CwiseMat(ApproxEq::eps() * 1e5);
			let L_val = &mut *vec![zero::<c64>(); symbolic.len_val()];
			let A_upper = A_upper.rb();
			simplicial::factorize_simplicial_numeric_ldlt(
				L_val,
				A_upper,
				Default::default(),
				symbolic,
				MemStack::new(&mut MemBuffer::new(simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, c64>(
					n,
				))),
			)?;

			let mut target = A_upper.to_dense();
			let adjoint = target.adjoint().to_owned();
			target.copy_from_strict_triangular_lower(adjoint);
			let A = reconstruct_from_simplicial_ldlt(symbolic, L_val);

			assert!(A ~ target);

			let k = 3;
			let rng = &mut StdRng::seed_from_u64(0);

			let rhs = CwiseMatDistribution {
				nrows: n,
				ncols: k,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			let simplicial = simplicial::SimplicialLdltRef::new(symbolic, L_val);
			for conj in [Conj::No, Conj::Yes] {
				let mut x = rhs.clone();
				simplicial.solve_in_place_with_conj(
					conj,
					x.rb_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(symbolic.solve_in_place_scratch::<c64>(k))),
				);

				let target = rhs.rb();
				let rhs = match conj {
					Conj::No => &A * &x,
					Conj::Yes => A.conjugate() * &x,
				};

				assert!(rhs ~ target);
			}
		}

		{
			for j in 0..n {
				let (i, x) = A_upper.rb_mut().idx_val_of_col_mut(j);
				for (i, x) in iter::zip(i, x) {
					if i == j {
						*x *= 1e3;
					}
				}
			}
			let A_upper = A_upper.rb();

			let approx_eq = CwiseMat(ApproxEq::eps() * 1e5);
			let L_val = &mut *vec![zero::<c64>(); symbolic.len_val()];
			simplicial::factorize_simplicial_numeric_llt(
				L_val,
				A_upper,
				Default::default(),
				symbolic,
				MemStack::new(&mut MemBuffer::new(simplicial::factorize_simplicial_numeric_llt_scratch::<usize, c64>(n))),
			)?;

			let mut target = A_upper.to_dense();
			let adjoint = target.adjoint().to_owned();
			target.copy_from_strict_triangular_lower(adjoint);
			let A = reconstruct_from_simplicial_llt(symbolic, L_val);

			assert!(A ~ target);

			let k = 3;
			let rng = &mut StdRng::seed_from_u64(0);

			let rhs = CwiseMatDistribution {
				nrows: n,
				ncols: k,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);

			let simplicial = simplicial::SimplicialLltRef::new(symbolic, L_val);
			for conj in [Conj::No, Conj::Yes] {
				let mut x = rhs.clone();
				simplicial.solve_in_place_with_conj(
					conj,
					x.rb_mut(),
					Par::Seq,
					MemStack::new(&mut MemBuffer::new(symbolic.solve_in_place_scratch::<c64>(k))),
				);

				let target = rhs.rb();
				let rhs = match conj {
					Conj::No => &A * &x,
					Conj::Yes => A.conjugate() * &x,
				};

				assert!(rhs ~ target);
			}
		}
		Ok(())
	}

	#[test]
	fn test_solver_llt() -> Result {
		let file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_cholesky/medium-1.txt");
		let mut A_upper = parse_csc::<c64>(&std::fs::read_to_string(&file)?).0;
		let n = A_upper.nrows();
		for j in 0..n {
			let (i, x) = A_upper.rb_mut().idx_val_of_col_mut(j);
			for (i, x) in iter::zip(i, x) {
				if i == j {
					*x *= 1e3;
				}
			}
		}
		let A_upper = A_upper.rb();
		let A_lower = A_upper.adjoint().to_col_major()?;
		let A_lower = A_lower.rb();

		let mut A_full = A_lower.to_dense();
		let adjoint = A_full.adjoint().to_owned();
		A_full.copy_from_triangular_upper(adjoint);
		let A_full = A_full.rb();

		let rng = &mut StdRng::seed_from_u64(0);
		let approx_eq = CwiseMat(ApproxEq::eps() * 1e4);

		for (A, side) in [(A_lower, Side::Lower), (A_upper, Side::Upper)] {
			for supernodal_flop_ratio_threshold in [SupernodalThreshold::FORCE_SIMPLICIAL, SupernodalThreshold::FORCE_SUPERNODAL] {
				for par in [Par::Seq, Par::rayon(4)] {
					let symbolic = &factorize_symbolic_cholesky(
						A.symbolic(),
						side,
						SymmetricOrdering::Amd,
						CholeskySymbolicParams {
							supernodal_flop_ratio_threshold,
							..Default::default()
						},
					)?;

					let L_val = &mut *vec![zero::<c64>(); symbolic.len_val()];
					let llt = symbolic.factorize_numeric_llt(
						L_val,
						A,
						side,
						Default::default(),
						par,
						MemStack::new(&mut MemBuffer::new(
							symbolic.factorize_numeric_llt_scratch::<c64>(par, Default::default()),
						)),
						Default::default(),
					)?;

					for k in (1..16).chain(128..132) {
						let rhs = CwiseMatDistribution {
							nrows: n,
							ncols: k,
							dist: ComplexDistribution::new(StandardNormal, StandardNormal),
						}
						.rand::<Mat<c64>>(rng);

						for conj in [Conj::No, Conj::Yes] {
							let mut x = rhs.clone();
							llt.solve_in_place_with_conj(
								conj,
								x.rb_mut(),
								par,
								MemStack::new(&mut MemBuffer::new(llt.solve_in_place_scratch::<c64>(k, Par::Seq))),
							);

							let target = rhs.as_ref();
							let rhs = match conj {
								Conj::No => A_full * &x,
								Conj::Yes => A_full.conjugate() * &x,
							};
							assert!(rhs ~ target);
						}
					}
				}
			}
		}

		Ok(())
	}

	#[test]
	fn test_solver_ldlt() -> Result {
		let file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_cholesky/medium-1.txt");
		let A_upper = parse_csc::<c64>(&std::fs::read_to_string(&file)?).0;
		let n = A_upper.nrows();

		let A_upper = A_upper.rb();
		let A_lower = A_upper.adjoint().to_col_major()?;
		let A_lower = A_lower.rb();

		let mut A_full = A_lower.to_dense();
		let adjoint = A_full.adjoint().to_owned();
		A_full.copy_from_triangular_upper(adjoint);
		let A_full = A_full.rb();

		let rng = &mut StdRng::seed_from_u64(0);
		let approx_eq = CwiseMat(ApproxEq::eps() * 1e5);

		for (A, side) in [(A_lower, Side::Lower), (A_upper, Side::Upper)] {
			for supernodal_flop_ratio_threshold in [SupernodalThreshold::FORCE_SIMPLICIAL, SupernodalThreshold::FORCE_SUPERNODAL] {
				for par in [Par::Seq, Par::rayon(4)] {
					let symbolic = &factorize_symbolic_cholesky(
						A.symbolic(),
						side,
						SymmetricOrdering::Amd,
						CholeskySymbolicParams {
							supernodal_flop_ratio_threshold,
							..Default::default()
						},
					)?;

					let L_val = &mut *vec![zero::<c64>(); symbolic.len_val()];
					let ldlt = symbolic.factorize_numeric_ldlt(
						L_val,
						A,
						side,
						Default::default(),
						par,
						MemStack::new(&mut MemBuffer::new(
							symbolic.factorize_numeric_ldlt_scratch::<c64>(par, Default::default()),
						)),
						Default::default(),
					)?;

					for k in (1..16).chain(128..132) {
						let rhs = CwiseMatDistribution {
							nrows: n,
							ncols: k,
							dist: ComplexDistribution::new(StandardNormal, StandardNormal),
						}
						.rand::<Mat<c64>>(rng);

						for conj in [Conj::No, Conj::Yes] {
							let mut x = rhs.clone();
							ldlt.solve_in_place_with_conj(
								conj,
								x.rb_mut(),
								par,
								MemStack::new(&mut MemBuffer::new(ldlt.solve_in_place_scratch::<c64>(k, Par::Seq))),
							);

							let target = rhs.as_ref();
							let rhs = match conj {
								Conj::No => A_full * &x,
								Conj::Yes => A_full.conjugate() * &x,
							};
							assert!(rhs ~ target);
						}
					}
				}
			}
		}

		Ok(())
	}

	#[test]
	fn test_solver_bk() -> Result {
		let file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/sparse_cholesky/medium-1.txt");
		let A_upper = parse_csc::<c64>(&std::fs::read_to_string(&file)?).0;
		let n = A_upper.nrows();

		let A_upper = A_upper.rb();
		let A_lower = A_upper.adjoint().to_col_major()?;
		let A_lower = A_lower.rb();

		let mut A_full = A_lower.to_dense();
		let adjoint = A_full.adjoint().to_owned();
		A_full.copy_from_triangular_upper(adjoint);
		let A_full = A_full.rb();

		let rng = &mut StdRng::seed_from_u64(0);
		let approx_eq = CwiseMat(ApproxEq::eps() * 1e4);

		for (A, side) in [(A_lower, Side::Lower), (A_upper, Side::Upper)] {
			for supernodal_flop_ratio_threshold in [SupernodalThreshold::FORCE_SIMPLICIAL, SupernodalThreshold::FORCE_SUPERNODAL] {
				for par in [Par::Seq, Par::rayon(4)] {
					let symbolic = &factorize_symbolic_cholesky(
						A.symbolic(),
						side,
						SymmetricOrdering::Amd,
						CholeskySymbolicParams {
							supernodal_flop_ratio_threshold,
							..Default::default()
						},
					)?;
					let fwd = &mut *vec![0usize; n];
					let bwd = &mut *vec![0usize; n];
					let subdiag = &mut *vec![zero::<c64>(); n];

					let L_val = &mut *vec![zero::<c64>(); symbolic.len_val()];
					let lblt = symbolic.factorize_numeric_intranode_lblt(
						L_val,
						subdiag,
						fwd,
						bwd,
						A,
						side,
						par,
						MemStack::new(&mut MemBuffer::new(
							symbolic.factorize_numeric_intranode_lblt_scratch::<c64>(par, Default::default()),
						)),
						Default::default(),
					);

					for k in (1..16).chain(128..132) {
						let rhs = CwiseMatDistribution {
							nrows: n,
							ncols: k,
							dist: ComplexDistribution::new(StandardNormal, StandardNormal),
						}
						.rand::<Mat<c64>>(rng);

						for conj in [Conj::No, Conj::Yes] {
							let mut x = rhs.clone();
							lblt.solve_in_place_with_conj(
								conj,
								x.rb_mut(),
								par,
								MemStack::new(&mut MemBuffer::new(lblt.solve_in_place_scratch::<c64>(k, Par::Seq))),
							);

							let target = rhs.as_ref();
							let rhs = match conj {
								Conj::No => A_full * &x,
								Conj::Yes => A_full.conjugate() * &x,
							};
							assert!(rhs ~ target);
						}
					}
				}
			}
		}

		Ok(())
	}
}
