use assert2::assert as fancy_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    mul::triangular::{self, BlockStructure},
    temp_mat_req, temp_mat_uninit, ComplexField, Conj, MatMut, MatRef, Parallelism,
};
use reborrow::*;

/// Computes the size and alignment of required workspace for reconstructing the lower triangular
/// part of the a matrix out of place, given its Cholesky decomposition.
pub fn reconstruct_lower_to_req<T: 'static>(dimension: usize) -> Result<StackReq, SizeOverflow> {
    let _ = dimension;
    Ok(StackReq::default())
}

/// Computes the size and alignment of required workspace for reconstructing the lower triangular
/// part of the a matrix in place, given its Cholesky decomposition.
pub fn reconstruct_lower_in_place_req<T: 'static>(
    dimension: usize,
) -> Result<StackReq, SizeOverflow> {
    temp_mat_req::<T>(dimension, dimension)
}

/// Computes the lower triangular part of the reconstructed matrix, given its Cholesky
/// decomposition, and stores the result in `dst`.
///
/// # Panics
///
/// - Panics if `cholesky_factor` is not a square matrix.
/// - Panics if the destination shape doesn't match the shape of the matrix.
#[track_caller]
pub fn reconstruct_lower_to<T: ComplexField>(
    dst: MatMut<'_, T>,
    cholesky_factor: MatRef<'_, T>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    fancy_assert!(cholesky_factor.nrows() == cholesky_factor.ncols());
    fancy_assert!((dst.nrows(), dst.ncols()) == (cholesky_factor.nrows(), cholesky_factor.ncols()));

    let _ = &stack;
    triangular::matmul(
        dst,
        BlockStructure::TriangularLower,
        Conj::No,
        cholesky_factor,
        BlockStructure::TriangularLower,
        Conj::No,
        cholesky_factor.transpose(),
        BlockStructure::TriangularUpper,
        Conj::Yes,
        None,
        T::one(),
        parallelism,
    );
}

/// Computes the lower triangular part of the reconstructed matrix, given its Cholesky
/// decomposition, and stores the result in `cholesky_factor`.
///
/// # Panics
///
/// - Panics if `cholesky_factor` is not a square matrix.
#[track_caller]
pub fn reconstruct_lower_in_place<T: ComplexField>(
    cholesky_factor: MatMut<'_, T>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let n = cholesky_factor.nrows();
    temp_mat_uninit! {
        let (mut tmp, stack) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
    }
    reconstruct_lower_to(tmp.rb_mut(), cholesky_factor.rb(), parallelism, stack);
    cholesky_factor
        .cwise()
        .zip(tmp.rb())
        .for_each_triangular_lower(faer_core::zip::Diag::Include, |dst, src| *dst = *src);
}
