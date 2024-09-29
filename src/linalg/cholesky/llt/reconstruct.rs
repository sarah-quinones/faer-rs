use crate::{
    assert,
    linalg::{
        matmul::triangular::{self, BlockStructure},
        temp_mat_req, temp_mat_uninit,
        zip::Diag,
    },
    unzipped, zipped, ComplexField, Entity, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

/// Computes the size and alignment of required workspace for reconstructing the lower triangular
/// part of the a matrix out of place, given its Cholesky decomposition.
pub fn reconstruct_lower_req<E: Entity>(dimension: usize) -> Result<StackReq, SizeOverflow> {
    let _ = dimension;
    Ok(StackReq::default())
}

/// Computes the size and alignment of required workspace for reconstructing the lower triangular
/// part of the a matrix in place, given its Cholesky decomposition.
pub fn reconstruct_lower_in_place_req<E: Entity>(
    dimension: usize,
) -> Result<StackReq, SizeOverflow> {
    temp_mat_req::<E>(dimension, dimension)
}

/// Computes the lower triangular part of the reconstructed matrix, given its Cholesky
/// decomposition, and stores the result in `dst`.
///
/// # Panics
///
/// - Panics if `cholesky_factor` is not a square matrix.
/// - Panics if the destination shape doesn't match the shape of the matrix.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`reconstruct_lower_req`]).
#[track_caller]
pub fn reconstruct_lower<E: ComplexField>(
    dst: MatMut<'_, E>,
    cholesky_factor: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    assert!(cholesky_factor.nrows() == cholesky_factor.ncols());
    assert!((dst.nrows(), dst.ncols()) == (cholesky_factor.nrows(), cholesky_factor.ncols()));

    let _ = &stack;
    triangular::matmul(
        dst,
        BlockStructure::TriangularLower,
        cholesky_factor,
        BlockStructure::TriangularLower,
        cholesky_factor.adjoint(),
        BlockStructure::TriangularUpper,
        None,
        E::faer_one(),
        parallelism,
    );
}

/// Computes the lower triangular part of the reconstructed matrix, given its Cholesky
/// decomposition, and stores the result in `cholesky_factor`.
///
/// # Panics
///
/// - Panics if `cholesky_factor` is not a square matrix.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`reconstruct_lower_in_place_req`]).
#[track_caller]
pub fn reconstruct_lower_in_place<E: ComplexField>(
    cholesky_factor: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    let n = cholesky_factor.nrows();
    let (mut tmp, stack) = temp_mat_uninit::<E>(n, n, stack);
    let mut tmp = tmp.as_mut();
    reconstruct_lower(tmp.rb_mut(), cholesky_factor.rb(), parallelism, stack);
    zipped!(cholesky_factor, tmp.rb())
        .for_each_triangular_lower(Diag::Include, |unzipped!(mut dst, src)| {
            dst.write(src.read())
        });
}
