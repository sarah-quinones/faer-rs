#[cfg(feature = "std")]
use assert2::assert;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    mul::triangular, permutation::PermutationRef, temp_mat_req, temp_mat_uninit, ComplexField,
    Entity, MatMut, MatRef, Parallelism,
};
use reborrow::*;
use triangular::BlockStructure;

#[track_caller]
fn reconstruct_impl<T: ComplexField>(
    mut dst: MatMut<'_, T>,
    lu_factors: Option<MatRef<'_, T>>,
    row_perm: PermutationRef<'_>,
    col_perm: PermutationRef<'_>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let lu_factors = match lu_factors {
        Some(lu_factors) => lu_factors,
        None => dst.rb(),
    };

    let m = lu_factors.nrows();
    let n = lu_factors.ncols();
    let size = Ord::min(m, n);

    let (mut lu, _) = temp_mat_uninit::<T>(m, n, stack);
    let mut lu = lu.as_mut();

    let [l_top, _, l_bot, _] = lu_factors.split_at(size, size);
    let [u_left, u_right, _, _] = lu_factors.split_at(size, size);

    let [lu_topleft, lu_topright, lu_botleft, _] = lu.rb_mut().split_at(size, size);

    triangular::matmul(
        lu_topleft,
        BlockStructure::Rectangular,
        l_top,
        BlockStructure::UnitTriangularLower,
        u_left,
        BlockStructure::TriangularUpper,
        None,
        T::one(),
        parallelism,
    );
    triangular::matmul(
        lu_topright,
        BlockStructure::Rectangular,
        l_top,
        BlockStructure::UnitTriangularLower,
        u_right,
        BlockStructure::Rectangular,
        None,
        T::one(),
        parallelism,
    );
    triangular::matmul(
        lu_botleft,
        BlockStructure::Rectangular,
        l_bot,
        BlockStructure::Rectangular,
        u_left,
        BlockStructure::TriangularUpper,
        None,
        T::one(),
        parallelism,
    );

    let row_inv = row_perm.into_arrays().1;
    let col_inv = col_perm.into_arrays().1;
    assert!(row_inv.len() == m);
    assert!(col_inv.len() == n);
    unsafe {
        if dst.row_stride().unsigned_abs() <= dst.col_stride().unsigned_abs() {
            for j in 0..n {
                let jj = *col_inv.get_unchecked(j);
                for i in 0..m {
                    let ii = *row_inv.get_unchecked(i);
                    dst.write_unchecked(i, j, lu.read_unchecked(ii, jj));
                }
            }
        } else {
            for i in 0..m {
                let ii = *row_inv.get_unchecked(i);
                for j in 0..n {
                    let jj = *col_inv.get_unchecked(j);
                    dst.write_unchecked(i, j, lu.read_unchecked(ii, jj));
                }
            }
        }
    }
}

/// Computes the reconstructed matrix, given its full pivoting LU decomposition,
/// and stores the result in `dst`.
///
/// # Panics
///
/// - Panics if the row permutation doesn't have the same dimension as the number of rows of the
///   matrix.
/// - Panics if the column permutation doesn't have the same dimension as the number of columns of
///   the matrix.
/// - Panics if the destination shape doesn't match the shape of the matrix.
/// - Panics if the provided memory in `stack` is insufficient (see [`reconstruct_req`]).
#[track_caller]
pub fn reconstruct<T: ComplexField>(
    dst: MatMut<'_, T>,
    lu_factors: MatRef<'_, T>,
    row_perm: PermutationRef<'_>,
    col_perm: PermutationRef<'_>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    assert!((dst.nrows(), dst.ncols()) == (lu_factors.nrows(), lu_factors.ncols()));
    assert!((row_perm.len(), col_perm.len()) == (lu_factors.nrows(), lu_factors.ncols()));
    reconstruct_impl(
        dst,
        Some(lu_factors),
        row_perm,
        col_perm,
        parallelism,
        stack,
    )
}

/// Computes the reconstructed matrix, given its full pivoting LU decomposition,
/// and stores the result in `lu_factors`.
///
/// # Panics
///
/// - Panics if the row permutation doesn't have the same dimension as the number of rows of the
///   matrix.
/// - Panics if the column permutation doesn't have the same dimension as the number of columns of
///   the matrix.
/// - Panics if the provided memory in `stack` is insufficient (see [`reconstruct_in_place_req`]).
#[track_caller]
pub fn reconstruct_in_place<T: ComplexField>(
    lu_factors: MatMut<'_, T>,
    row_perm: PermutationRef<'_>,
    col_perm: PermutationRef<'_>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    assert!((row_perm.len(), col_perm.len()) == (lu_factors.nrows(), lu_factors.ncols()));
    reconstruct_impl(lu_factors, None, row_perm, col_perm, parallelism, stack)
}

/// Computes the size and alignment of required workspace for reconstructing a matrix in place,
/// given its full pivoting LU decomposition.
pub fn reconstruct_in_place_req<T: Entity>(
    nrows: usize,
    ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    temp_mat_req::<T>(nrows, ncols)
}

/// Computes the size and alignment of required workspace for reconstructing a matrix out of place,
/// given its full pivoting LU decomposition.
pub fn reconstruct_req<T: Entity>(
    nrows: usize,
    ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    reconstruct_in_place_req::<T>(nrows, ncols, parallelism)
}
