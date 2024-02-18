use crate::assert;
use crate::mat::*;
use crate::*;
use reborrow::*;

/// Kronecker product of two matrices.
///
/// The Kronecker product of two matrices `A` and `B` is a block matrix
/// `C` with the following structure:
///
/// ```text
/// C = [ a[(0, 0)] * B    , a[(0, 1)] * B    , ... , a[(0, n-1)] * B    ]
///     [ a[(1, 0)] * B    , a[(1, 1)] * B    , ... , a[(1, n-1)] * B    ]
///     [ ...              , ...              , ... , ...              ]
///     [ a[(m-1, 0)] * B  , a[(m-1, 1)] * B  , ... , a[(m-1, n-1)] * B  ]
/// ```
///
/// # Panics
///
/// Panics if `dst` does not have the correct dimensions. The dimensions
/// of `dst` must be `nrows(A) * nrows(B)` by `ncols(A) * ncols(B)`.
///
/// # Example
///
/// ```
/// use faer::{linalg::kron, mat, Mat};
///
/// let a = mat![[1.0, 2.0], [3.0, 4.0]];
/// let b = mat![[0.0, 5.0], [6.0, 7.0]];
/// let c = mat![
///     [0.0, 5.0, 0.0, 10.0],
///     [6.0, 7.0, 12.0, 14.0],
///     [0.0, 15.0, 0.0, 20.0],
///     [18.0, 21.0, 24.0, 28.0],
/// ];
/// let mut dst = Mat::new();
/// dst.resize_with(4, 4, |_, _| 0f64);
/// kron(dst.as_mut(), a.as_ref(), b.as_ref());
/// assert_eq!(dst, c);
/// ```
#[track_caller]
pub fn kron<E: ComplexField>(dst: MatMut<E>, lhs: MatRef<E>, rhs: MatRef<E>) {
    let mut dst = dst;
    let mut lhs = lhs;
    let mut rhs = rhs;
    if dst.col_stride().unsigned_abs() < dst.row_stride().unsigned_abs() {
        dst = dst.transpose_mut();
        lhs = lhs.transpose();
        rhs = rhs.transpose();
    }

    assert!(Some(dst.nrows()) == lhs.nrows().checked_mul(rhs.nrows()));
    assert!(Some(dst.ncols()) == lhs.ncols().checked_mul(rhs.ncols()));

    for lhs_j in 0..lhs.ncols() {
        for lhs_i in 0..lhs.nrows() {
            let lhs_val = lhs.read(lhs_i, lhs_j);
            let mut dst = dst.rb_mut().submatrix_mut(
                lhs_i * rhs.nrows(),
                lhs_j * rhs.ncols(),
                rhs.nrows(),
                rhs.ncols(),
            );

            for rhs_j in 0..rhs.ncols() {
                for rhs_i in 0..rhs.nrows() {
                    // SAFETY: Bounds have been checked.
                    unsafe {
                        let rhs_val = rhs.read_unchecked(rhs_i, rhs_j);
                        dst.write_unchecked(rhs_i, rhs_j, lhs_val.faer_mul(rhs_val));
                    }
                }
            }
        }
    }
}
