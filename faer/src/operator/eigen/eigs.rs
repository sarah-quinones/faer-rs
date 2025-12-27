use crate::{
	dyn_stack::{MemBuffer, MemStack, StackReq},
	mat::AsMatRef,
	matrix_free::{
		LinOp,
		eigen::{partial_eigen, partial_eigen_scratch},
	},
	prelude::*,
	sparse::linalg::{LuError, solvers::Lu},
	traits::{ComplexField, Index},
};
use num_complex::Complex;
use std::fmt::Debug;
use std::ops::Mul;

/// A trait that enables eigenvalue decomposition using the shift-invert method for a matrix type.
pub trait EigsSolvable: Debug + Clone + Sync + for<'a> Mul<MatRef<'a, Self::T>, Output: AsMatRef<T = Self::T, Rows = usize, Cols = usize>> {
	/// The type of error that can occur during the setup of the solver.
	type Error;

	/// The solver that will be used during the shift-invert process.
	type Solver: Debug + Sync;

	/// The type of the matrix elements.
	type T: ComplexField;

	/// Prepares the solver for shift-invert iterations.
	fn setup_solver(mat_a: Self, mat_b: Self, shift: &Self::T) -> Result<Self::Solver, Self::Error>;

	/// Use an appropriate solver to find x in (A - σB)x = b.
	fn solve_x<Rhs: AsMatRef<T = Self::T, Rows = usize, Cols = usize>>(solver: &Self::Solver, b: Rhs) -> Rhs::Owned;

	/// Number of rows in the A (and B) matrices.
	fn n_rows(&self) -> usize;

	/// Number of columns in the A (and B) matrices.
	fn n_cols(&self) -> usize;
}

#[derive(Debug)]
struct ShiftInvertOperator<M: EigsSolvable> {
	mat_a: M,
	mat_b: M,
	solver: M::Solver,
}

impl<M: EigsSolvable> ShiftInvertOperator<M> {
	fn new(mat_a: M, mat_b: M, shift: &M::T) -> Result<Self, M::Error> {
		assert_eq!(mat_a.n_rows(), mat_b.n_rows());
		assert_eq!(mat_a.n_cols(), mat_b.n_cols());

		// Assuming M is usually of the "view" type, cloning should be cheap.
		Ok(Self {
			mat_a: mat_a.clone(),
			mat_b: mat_b.clone(),
			solver: M::setup_solver(mat_a.clone(), mat_b.clone(), shift)?,
		})
	}
}

impl<M: EigsSolvable> LinOp<M::T> for ShiftInvertOperator<M> {
	fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
		StackReq::default()
	}

	fn nrows(&self) -> usize {
		self.mat_a.n_rows()
	}

	fn ncols(&self) -> usize {
		self.mat_a.n_cols()
	}

	fn apply(&self, out: MatMut<'_, M::T>, rhs: MatRef<'_, M::T>, _par: Par, _stack: &mut MemStack) {
		// For generalized problems y = (A - σB)⁻¹B * v

		// Compute w = B * v
		let w = self.mat_b.clone() * rhs;

		// Solve (A - σB)y = w
		let res = M::solve_x(&self.solver, w);
		let res = res.as_mat_ref();

		// TODO: is there a more efficient solution than to copy here?
		assert_eq!(out.shape(), res.shape());
		out.col_iter_mut().zip(res.col_iter()).for_each(|(mut out_col, res_col)| {
			out_col.copy_from(res_col);
		});
	}

	fn conj_apply(&self, _out: MatMut<'_, M::T>, _rhs: MatRef<'_, M::T>, _par: Par, _stack: &mut MemStack) {
		// We assume this will not be called during the shift-invert process
		unimplemented!()
	}
}

fn solve_largest_eigenvalues<T: ComplexField>(
	matrix: &dyn LinOp<T>,
	n_eigval: usize,
	par: Par,
	tolerance: T::Real,
) -> (Vec<Complex<T::Real>>, Mat<Complex<T::Real>>) {
	let nrows = matrix.nrows();
	let mut mem = MemBuffer::new(partial_eigen_scratch(matrix, n_eigval, par, default()));
	let mut eigvecs = Mat::zeros(nrows, n_eigval);
	let mut eigvals = vec![Complex::<T::Real>::zero_impl(); n_eigval];

	// TODO: are zeros really the best solution here?
	let v0: Col<T> = Col::zeros(nrows);

	partial_eigen(
		eigvecs.rb_mut(),
		&mut eigvals,
		matrix,
		v0.as_ref(),
		tolerance,
		par,
		MemStack::new(&mut mem),
		default(),
	);
	(eigvals, eigvecs)
}

/// Solves the generalized eigenvalue problem (Av=λBv) to find a specified number of eigenvalues
/// (and their corresponding eigenvectors) closest to a given eigenvalue.
///
/// Parameters:
/// - `mat_a`: The matrix A in the generalized eigenvalue problem.
/// - `mat_b`: The matrix B in the generalized eigenvalue problem.
/// - `sigma`: The eigenvalue around which to search for eigenvalues.
/// - `n_eigval`: The number of eigenvalues to compute.
/// - `par`: The parallelization strategy to use.
/// - `tolerance`: The tolerance for convergence to use.
///
/// Returns:
/// - A tuple containing the eigenvalues and eigenvectors.
pub fn eigs<M: EigsSolvable>(
	mat_a: M,
	mat_b: M,
	sigma: &M::T,
	n_eigval: usize,
	par: Par,
	tolerance: <<M as EigsSolvable>::T as ComplexField>::Real,
) -> Result<
	(
		Vec<Complex<<<M as EigsSolvable>::T as ComplexField>::Real>>,
		Mat<Complex<<<M as EigsSolvable>::T as ComplexField>::Real>>,
	),
	M::Error,
> {
	let shift_invert_op = ShiftInvertOperator::new(mat_a, mat_b, sigma)?;
	let (shifted_eigenvalues, eigenvectors) = solve_largest_eigenvalues(&shift_invert_op, n_eigval, par, tolerance);

	// Since we always get back complex eigenvalues, we need to ensure we have complex values for one and sigma
	let one = Complex::<<<M as EigsSolvable>::T as ComplexField>::Real>::one_impl();
	let sigma = Complex::<<<M as EigsSolvable>::T as ComplexField>::Real>::new(
		<M as EigsSolvable>::T::real_part_impl(&sigma),
		<M as EigsSolvable>::T::imag_part_impl(&sigma),
	);

	// Transform back to original spectrum using μ = 1/(λ - σ)
	let original_eigenvalues = shifted_eigenvalues.iter().map(|mu| sigma.clone() + one.clone() / mu).collect();
	Ok((original_eigenvalues, eigenvectors))
}

/// Uses the LU solver to implement the shift-invert method for Sparse Column Matrices.
#[derive(Debug)]
pub struct SparseColMatSolver<I, T> {
	lu: Lu<I, T>,
}

impl<I: Index, T: ComplexField> SparseColMatSolver<I, T> {
	fn new(mat_a: SparseColMatRef<I, T>, mat_b: SparseColMatRef<I, T>, shift: &T) -> Result<Self, LuError> {
		// Form (A - σB)
		let m = mat_a - (Scale::from_ref(shift) * mat_b);

		// Pre-compute factorization to solve iterations quickly
		Ok(Self { lu: m.sp_lu()? })
	}

	fn solve<Rhs: AsMatRef<T = T, Rows = usize>>(&self, b: Rhs) -> Rhs::Owned {
		self.lu.solve(b)
	}
}

impl<'a, I: Index, T: ComplexField> EigsSolvable for SparseColMatRef<'a, I, T> {
	type Error = LuError;
	type Solver = SparseColMatSolver<I, T>;
	type T = T;

	fn setup_solver(mat_a: Self, mat_b: Self, shift: &Self::T) -> Result<Self::Solver, Self::Error> {
		SparseColMatSolver::new(mat_a, mat_b, shift)
	}

	fn solve_x<Rhs: AsMatRef<T = T, Rows = usize>>(solver: &Self::Solver, b: Rhs) -> Rhs::Owned {
		solver.solve(b)
	}

	fn n_rows(&self) -> usize {
		self.nrows()
	}

	fn n_cols(&self) -> usize {
		self.ncols()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::sparse::Triplet;
	use num_complex::Complex64;

	#[test]
	fn test_eigs_sparse_cplx() {
		// A = [2 1 0 0]    B = [1 0 0 0]
		//     [1 2 1 0]        [0 1 0 0]
		//     [0 1 2 1]        [0 0 1 0]
		//     [0 0 1 2]        [0 0 0 1]
		// The eigenvalues are approximately 3.6180, 2.6180, 1.3820, 0.3820

		let mat_a = SparseColMat::<usize, Complex64>::try_new_from_triplets(
			4,
			4,
			&[
				Triplet::new(0, 0, Complex64::new(2.0, 0.0)),
				Triplet::new(0, 1, Complex64::new(1.0, 0.0)),
				Triplet::new(1, 0, Complex64::new(1.0, 0.0)),
				Triplet::new(1, 1, Complex64::new(2.0, 0.0)),
				Triplet::new(1, 2, Complex64::new(1.0, 0.0)),
				Triplet::new(2, 1, Complex64::new(1.0, 0.0)),
				Triplet::new(2, 2, Complex64::new(2.0, 0.0)),
				Triplet::new(2, 3, Complex64::new(1.0, 0.0)),
				Triplet::new(3, 2, Complex64::new(1.0, 0.0)),
				Triplet::new(3, 3, Complex64::new(2.0, 0.0)),
			],
		)
		.unwrap();

		let mat_b = SparseColMat::<usize, Complex64>::try_new_from_triplets(
			4,
			4,
			&[
				Triplet::new(0, 0, Complex64::new(1.0, 0.0)),
				Triplet::new(1, 1, Complex64::new(1.0, 0.0)),
				Triplet::new(2, 2, Complex64::new(1.0, 0.0)),
				Triplet::new(3, 3, Complex64::new(1.0, 0.0)),
			],
		)
		.unwrap();

		// Choose a shift near 2.0 to find the two eigenvalues closest to it
		let sigma = Complex64::new(2.0, 0.0);
		let n_eigval = 2;

		// Compute eigenvalues and eigenvectors
		let (eigenvalues, eigenvectors) = eigs(mat_a.as_ref(), mat_b.as_ref(), &sigma, n_eigval, Par::Seq, f64::EPSILON * 128.0).unwrap();
		assert_eq!(eigenvalues.len(), n_eigval);

		// Verify eigenvectors satisfy Ax = λBx
		for (i, &lambda) in eigenvalues.iter().enumerate() {
			let x = eigenvectors.col(i);
			let ax = &mat_a * x;
			let bx = &mat_b * x;
			let residual = ax - Scale(lambda) * bx;
			assert!(residual.norm_l2() < 1e-10);
		}

		// Sort eigenvalues by distance from sigma
		let mut sorted_eigenvalues = eigenvalues;
		sorted_eigenvalues.sort_by(|a, b| (a - sigma).norm().partial_cmp(&(b - sigma).norm()).unwrap());

		// Check that we found the two eigenvalues closest to sigma (2.0)
		// These should be approximately 1.3820 and 2.6180
		assert!((sorted_eigenvalues[0] - Complex64::new(1.3820, 0.0)).norm() < 1e-3);
		assert!((sorted_eigenvalues[1] - Complex64::new(2.6180, 0.0)).norm() < 1e-3);
	}
}
