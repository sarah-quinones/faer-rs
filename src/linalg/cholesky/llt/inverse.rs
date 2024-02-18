use crate::{
    assert,
    linalg::matmul::triangular::{self, BlockStructure},
    linalg::triangular_inverse::invert_lower_triangular,
    linalg::{temp_mat_req, temp_mat_uninit},
    ComplexField, Entity, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

fn invert_lower_impl<E: ComplexField>(
    dst: MatMut<'_, E>,
    cholesky_factor: Option<MatRef<'_, E>>,
    parallelism: Parallelism,
    stack: PodStack,
) {
    // (L L.*).inv() = L.inv().* L.inv()
    //
    let cholesky_factor = match cholesky_factor {
        Some(cholesky_factor) => cholesky_factor,
        None => dst.rb(),
    };
    let n = cholesky_factor.nrows();

    let (mut tmp, _) = temp_mat_uninit::<E>(n, n, stack);
    let mut tmp = tmp.as_mut();

    invert_lower_triangular(tmp.rb_mut(), cholesky_factor, parallelism);

    triangular::matmul(
        dst,
        BlockStructure::TriangularLower,
        tmp.rb().adjoint(),
        BlockStructure::TriangularUpper,
        tmp.rb(),
        BlockStructure::TriangularLower,
        None,
        E::faer_one(),
        parallelism,
    );
}

/// Computes the size and alignment of required workspace for computing the lower triangular part
/// of the inverse of a matrix out of place, given the Cholesky
/// decomposition.
pub fn invert_lower_req<E: Entity>(
    dimension: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    temp_mat_req::<E>(dimension, dimension)
}

/// Computes the size and alignment of required workspace for computing the lower triangular part
/// of the inverse of a matrix, given its Cholesky decomposition.
pub fn invert_lower_in_place_req<E: Entity>(
    dimension: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    invert_lower_req::<E>(dimension, parallelism)
}

/// Computes the lower triangular part of the inverse of a matrix, given its Cholesky
/// decomposition, and stores the result in `cholesky_factor`.
///
/// # Panics
///
/// - Panics if `cholesky_factor` is not a square matrix.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`invert_lower_req`]).
#[track_caller]
pub fn invert_lower_in_place<E: ComplexField>(
    cholesky_factor: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack,
) {
    assert!(cholesky_factor.nrows() == cholesky_factor.ncols());
    invert_lower_impl(cholesky_factor, None, parallelism, stack);
}

/// Computes the lower triangular part of the inverse of a matrix, given its Cholesky
/// decomposition, and stores the result in `dst`.
///
/// # Panics
///
/// - Panics if `cholesky_factor` is not a square matrix.
/// - Panics if the destination shape doesn't match the shape of the matrix.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`invert_lower_in_place_req`]).
#[track_caller]
pub fn invert_lower<E: ComplexField>(
    dst: MatMut<'_, E>,
    cholesky_factor: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: PodStack,
) {
    assert!(all(
        cholesky_factor.nrows() == cholesky_factor.ncols(),
        dst.nrows() == cholesky_factor.nrows(),
        dst.ncols() == cholesky_factor.ncols(),
    ));
    invert_lower_impl(dst, Some(cholesky_factor), parallelism, stack);
}
