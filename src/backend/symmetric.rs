use crate::{backend::permutation::PermutationIndicesRef, MatMut, MatRef};
use assert2::assert as fancy_assert;
use reborrow::*;

/// Computes a symmetric permutation of the source matrix using the given permutation, and stores
/// the result in the destination matrix.
///
/// Both the source and the destination are interpreted as symmetric matrices, and only their lower
/// triangular part is accessed.
pub fn apply_symmetric_permutation<T: Clone>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    perm_indices: PermutationIndicesRef<'_>,
) {
    let mut dst = dst;
    let n = src.nrows();
    fancy_assert!(src.nrows() == src.ncols(), "source matrix must be square",);
    fancy_assert!(
        dst.nrows() == dst.ncols(),
        "destination matrix must be square",
    );
    fancy_assert!(
        src.nrows() == dst.nrows(),
        "source and destination matrices must have the same shape",
    );
    fancy_assert!(
        perm_indices.into_array().len() == n,
        "permutation must have the same length as the dimension of the matrices"
    );

    let perm = perm_indices.into_array();
    let src_tril = |i, j| unsafe {
        if i > j {
            src.get_unchecked(i, j)
        } else {
            src.get_unchecked(j, i)
        }
    };
    for j in 0..n {
        for i in j..n {
            unsafe {
                *dst.rb_mut().get_unchecked(i, j) =
                    src_tril(*perm.get_unchecked(i), *perm.get_unchecked(j)).clone();
            }
        }
    }
}
