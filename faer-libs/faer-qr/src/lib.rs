//! This crate provides utilities for computing and manipulating the QR factorization with and
//! without pivoting. The QR factorization decomposes a matrix into a product of a unitary matrix
//! $Q$ (represented using block Householder sequences), and an upper trapezoidal matrix $R$, such
//! that their product is equal to the original matrix (or a column permutation of it in the case
//! where column pivoting is used).
//!
//! # Example
//!
//! Assume we have an overdetermined system $AX = B$ with full rank, and that we wish to find the
//! solution that minimizes the 2-norm.
//!
//! This is equivalent to computing a matrix $X$ that minimizes the value $||AX - B||^2$,
//! which is given by the solution $$X = (A^H A)^{-1} A^H B.$$
//!
//! If we compute the QR decomposition of $A$, such that $A = QR = Q_{\text{thin}} R_{\text{rect}}$,
//! then we get $$X = R_{\text{rect}}^{-1} Q_{\text{thin}}^H B.$$
//!
//! To translate this to code, we can proceed as follows:
//!
//! ```
//! use assert_approx_eq::assert_approx_eq;
//! use dyn_stack::{PodStack, GlobalPodBuffer, StackReq};
//! use faer_core::{mat, solve, Conj, Mat, Parallelism};
//! use reborrow::*;
//!
//! // we start by defining matrices A and B that define our least-squares problem.
//! let a = mat![
//!     [-1.14920683, -1.67950492],
//!     [-0.93009756, -0.03885086],
//!     [1.22579735, 0.88489976],
//!     [0.70698973, 0.38928314],
//!     [-1.66293762, 0.38123281],
//!     [0.27639595, -0.32559289],
//!     [-0.37506387, -0.13180778],
//!     [-1.20774962, -0.38635657],
//!     [0.44373549, 0.84397648],
//!     [-1.96779374, -1.42751757_f64],
//! ];
//!
//! let b = mat![
//!     [-0.14689786, -0.52845138, -2.26975669],
//!     [-1.00844774, -1.38550214, 0.50329459],
//!     [1.07941646, 0.71514245, -0.73987761],
//!     [0.1281168, -0.23999022, 1.58776697],
//!     [-0.49385283, 1.17875407, 2.01019076],
//!     [0.65117811, -0.60339895, 0.27217694],
//!     [0.85599951, -0.00699227, 0.93607199],
//!     [-0.12635444, 0.94945626, 0.86565968],
//!     [0.02383305, 0.41515805, -1.2816278],
//!     [0.34158312, -0.07552168, 0.56724015_f64],
//! ];
//!
//! // computed with numpy
//! let expected_solution = mat![
//!     [0.33960324, -0.33812452, -0.8458301],
//!     [-0.25718351, 0.6281214, 1.07071764_f64],
//! ];
//!
//! let rank = a.nrows().min(a.ncols());
//!
//! // we choose the recommended block size for the householder factors of our problem.
//! let blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<f64>(a.nrows(), a.ncols());
//!
//! // we allocate the memory for the operations that we perform
//! let mut mem =
//!     GlobalPodBuffer::new(StackReq::any_of(
//!         [
//!             faer_qr::no_pivoting::compute::qr_in_place_req::<f64>(
//!                 a.nrows(),
//!                 a.ncols(),
//!                 blocksize,
//!                 Parallelism::None,
//!                 Default::default(),
//!             )
//!             .unwrap(),
//!             faer_core::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_req::<
//!                 f64,
//!             >(a.nrows(), blocksize, b.ncols())
//!             .unwrap(),
//!         ],
//!     ));
//! let mut stack = PodStack::new(&mut mem);
//!
//! let mut qr = a.clone();
//! let mut h_factor = Mat::zeros(blocksize, rank);
//! faer_qr::no_pivoting::compute::qr_in_place(
//!     qr.as_mut(),
//!     h_factor.as_mut(),
//!     Parallelism::None,
//!     stack.rb_mut(),
//!     Default::default(),
//! );
//!
//! // now the Householder bases are in the strictly lower trapezoidal part of `a`, and the
//! // matrix R is in the upper triangular part of `a`.
//!
//! let mut solution = b.clone();
//!
//! // compute Q^H×B
//! faer_core::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
//!     qr.as_ref(),
//!     h_factor.as_ref(),
//!     Conj::Yes,
//!     solution.as_mut(),
//!     Parallelism::None,
//!     stack.rb_mut(),
//! );
//!
//! solution.resize_with(rank, b.ncols(), |_, _| unreachable!());
//!
//! // compute R_rect^{-1} Q_thin^H×B
//! solve::solve_upper_triangular_in_place(
//!     qr.as_ref().split_at_row(rank).0,
//!     solution.as_mut(),
//!     Parallelism::None,
//! );
//!
//! for i in 0..rank {
//!     for j in 0..b.ncols() {
//!         assert_approx_eq!(solution.read(i, j), expected_solution.read(i, j));
//!     }
//! }
//! ```

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(feature = "std"), no_std)]

pub mod col_pivoting;
pub mod no_pivoting;

#[cfg(test)]
mod tests {

    #[test]
    fn test_example() {
        use crate::no_pivoting::compute;
        use assert_approx_eq::assert_approx_eq;
        use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
        use faer_core::{householder, mat, solve, Conj, Mat, Parallelism};
        use reborrow::*;

        // we start by defining matrices A and B that define our least-squares problem.
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

        // computed with numpy
        let expected_solution = mat![
            [0.33960324, -0.33812452, -0.8458301],
            [-0.25718351, 0.6281214, 1.07071764_f64],
        ];

        let rank = a.nrows().min(a.ncols());

        // we choose the recommended block size for the householder factors of our problem.
        let blocksize = compute::recommended_blocksize::<f64>(a.nrows(), a.ncols());

        // we allocate the memory for the operations that we perform
        let mut mem =
            GlobalPodBuffer::new(StackReq::any_of([
                compute::qr_in_place_req::<f64>(
                    a.nrows(),
                    a.ncols(),
                    blocksize,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
                faer_core::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_req::<
                    f64,
                >(a.nrows(), blocksize, b.ncols())
                .unwrap(),
            ]));
        let mut stack = PodStack::new(&mut mem);

        let mut qr = a;
        let mut h_factor = Mat::zeros(blocksize, rank);
        compute::qr_in_place(
            qr.as_mut(),
            h_factor.as_mut(),
            Parallelism::None,
            stack.rb_mut(),
            Default::default(),
        );

        // now the Householder bases are in the strictly lower trapezoidal part of `a`, and the
        // matrix R is in the upper triangular part of `a`.

        let mut solution = b.clone();

        // compute Q^H×B
        householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            qr.as_ref(),
            h_factor.as_ref(),
            Conj::Yes,
            solution.as_mut(),
            Parallelism::None,
            stack.rb_mut(),
        );

        solution.resize_with(rank, b.ncols(), |_, _| unreachable!());

        // compute R_rect^{-1} Q_thin^H×B
        solve::solve_upper_triangular_in_place(
            qr.as_ref().split_at_row(rank).0,
            solution.as_mut(),
            Parallelism::None,
        );

        for i in 0..rank {
            for j in 0..b.ncols() {
                assert_approx_eq!(solution.read(i, j), expected_solution.read(i, j));
            }
        }
    }
}
