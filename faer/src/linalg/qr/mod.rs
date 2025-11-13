//! `faer` provides utilities for computing and manipulating the $QR$ factorization with and
//! without pivoting. the $QR$ factorization decomposes a matrix into a product of a unitary matrix
//! $Q$ (represented using block householder sequences), and an upper trapezoidal matrix $R$, such
//! that their product is equal to the original matrix (or a column permutation of it in the case
//! where column pivoting is used)
//!
//! # example
//!
//! assume we have an overdetermined system $Ax = b$ with full rank, and that we wish to find the
//! solution that minimizes the 2-norm
//!
//! this is equivalent to computing a matrix $x$ that minimizes the value $||Ax - b||^2$,
//! which is given by the solution $$x = (A^H A)^{-1} A^H b$$
//!
//! if we compute the $QR$ decomposition of $A$, such that $A = QR = Q_{\text{thin}}
//! R_{\text{thin}}$, then we get $$x = R_{\text{thin}}^{-1} Q_{\text{thin}}^H b$$
//!
//! to translate this to code, we can proceed as follows:
//!
//! ```
//! use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
//! use faer::linalg::qr::no_pivoting::factor;
//! use faer::linalg::{householder, triangular_solve};
//! use faer::reborrow::*;
//! use faer::{Conj, Mat, Par, mat};
//! // we start by defining matrices A and B that define our least-squares problem.
//! let a = mat![
//! 	[-1.14920683, -1.67950492],
//! 	[-0.93009756, -0.03885086],
//! 	[1.22579735, 0.88489976],
//! 	[0.70698973, 0.38928314],
//! 	[-1.66293762, 0.38123281],
//! 	[0.27639595, -0.32559289],
//! 	[-0.37506387, -0.13180778],
//! 	[-1.20774962, -0.38635657],
//! 	[0.44373549, 0.84397648],
//! 	[-1.96779374, -1.42751757_f64],
//! ];
//! let b = mat![
//! 	[-0.14689786, -0.52845138, -2.26975669],
//! 	[-1.00844774, -1.38550214, 0.50329459],
//! 	[1.07941646, 0.71514245, -0.73987761],
//! 	[0.1281168, -0.23999022, 1.58776697],
//! 	[-0.49385283, 1.17875407, 2.01019076],
//! 	[0.65117811, -0.60339895, 0.27217694],
//! 	[0.85599951, -0.00699227, 0.93607199],
//! 	[-0.12635444, 0.94945626, 0.86565968],
//! 	[0.02383305, 0.41515805, -1.2816278],
//! 	[0.34158312, -0.07552168, 0.56724015_f64],
//! ];
//! // approximate solution computed with numpy
//! let expected_solution = mat![
//! 	[0.33960324, -0.33812452, -0.8458301], //
//! 	[-0.25718351, 0.6281214, 1.07071764_f64],
//! ];
//! let rank = Ord::min(a.nrows(), a.ncols());
//! // we choose the recommended block size for the householder factors of our problem.
//! let block_size = factor::recommended_block_size::<f64>(a.nrows(), a.ncols());
//! // we allocate the memory for the operations that we perform
//! let mut mem =
//! 	MemBuffer::new(StackReq::any_of(&[
//! 		factor::qr_in_place_scratch::<f64>(
//! 			a.nrows(),
//! 			a.ncols(),
//! 			block_size,
//! 			Par::Seq,
//! 			Default::default(),
//! 		),
//! 		householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<
//! 			f64,
//! 		>(a.nrows(), block_size, b.ncols()),
//! 	]));
//! let mut stack = MemStack::new(&mut mem);
//! let mut qr = a;
//! let mut h_factor = Mat::zeros(block_size, rank);
//! factor::qr_in_place(
//! 	qr.as_mut(),
//! 	h_factor.as_mut(),
//! 	Par::Seq,
//! 	stack.rb_mut(),
//! 	Default::default(),
//! );
//! // now the householder bases are in the strictly lower trapezoidal part of `a`, and the
//! // matrix R is in the upper triangular part of `a`.
//! let mut solution = b.clone();
//! // compute Q^H×B
//! householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
//! 	qr.as_ref(),
//! 	h_factor.as_ref(),
//! 	Conj::Yes,
//! 	solution.as_mut(),
//! 	Par::Seq,
//! 	stack.rb_mut(),
//! );
//! solution.truncate(rank, b.ncols());
//! // compute R_thin^{-1} Q_thin^H×B
//! triangular_solve::solve_upper_triangular_in_place(
//! 	qr.as_ref().split_at_row(rank).0,
//! 	solution.as_mut(),
//! 	Par::Seq,
//! );
//! for i in 0..rank {
//! 	for j in 0..b.ncols() {
//! 		assert!((solution[(i, j)] - expected_solution[(i, j)]).abs() <= 1e-6);
//! 	}
//! }
//! ```
pub mod col_pivoting;
pub mod no_pivoting;
#[cfg(test)]
mod tests {
	use crate as faer;
	use equator::assert;
	#[test]
	fn test_example() {
		use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
		use faer::linalg::qr::no_pivoting::factor;
		use faer::linalg::{householder, triangular_solve};
		use faer::reborrow::*;
		use faer::{Conj, Mat, Par, mat};
		let a = mat![
			[-1.14920683, -1.67950492],
			[-0.93009756, -0.03885086],
			[1.22579735, 0.88489976],
			[0.70698973, 0.38928314],
			[-1.66293762, 0.38123281],
			[0.27639595, -0.32559289],
			[-0.37506387, -0.13180778],
			[-1.20774962, -0.38635657],
			[0.44373549, 0.84397648],
			[-1.96779374, -1.42751757_f64],
		];
		let b = mat![
			[-0.14689786, -0.52845138, -2.26975669],
			[-1.00844774, -1.38550214, 0.50329459],
			[1.07941646, 0.71514245, -0.73987761],
			[0.1281168, -0.23999022, 1.58776697],
			[-0.49385283, 1.17875407, 2.01019076],
			[0.65117811, -0.60339895, 0.27217694],
			[0.85599951, -0.00699227, 0.93607199],
			[-0.12635444, 0.94945626, 0.86565968],
			[0.02383305, 0.41515805, -1.2816278],
			[0.34158312, -0.07552168, 0.56724015_f64],
		];
		let expected_solution = mat![[0.33960324, -0.33812452, -0.8458301], [-0.25718351, 0.6281214, 1.07071764_f64],];
		let rank = Ord::min(a.nrows(), a.ncols());
		let block_size = factor::recommended_block_size::<f64>(a.nrows(), a.ncols());
		let mut mem = MemBuffer::new(StackReq::any_of(&[
			factor::qr_in_place_scratch::<f64>(a.nrows(), a.ncols(), block_size, Par::Seq, Default::default()),
			householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(a.nrows(), block_size, b.ncols()),
		]));
		let mut stack = MemStack::new(&mut mem);
		let mut qr = a;
		let mut h_factor = Mat::zeros(block_size, rank);
		factor::qr_in_place(qr.as_mut(), h_factor.as_mut(), Par::Seq, stack.rb_mut(), Default::default());
		let mut solution = b.clone();
		householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
			qr.as_ref(),
			h_factor.as_ref(),
			Conj::Yes,
			solution.as_mut(),
			Par::Seq,
			stack.rb_mut(),
		);
		solution.truncate(rank, b.ncols());
		triangular_solve::solve_upper_triangular_in_place(qr.as_ref().split_at_row(rank).0, solution.as_mut(), Par::Seq);
		for i in 0..rank {
			for j in 0..b.ncols() {
				assert!((solution[(i, j)] - expected_solution[(i, j)]).abs() <= 1e-6);
			}
		}
	}
}
