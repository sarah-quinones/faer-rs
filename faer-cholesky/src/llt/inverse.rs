use assert2::assert as fancy_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    inverse::invert_lower_triangular_to,
    mul::triangular::{self, BlockStructure},
    temp_mat_req, temp_mat_uninit, ComplexField, Conj, MatMut, MatRef, Parallelism,
};
use reborrow::*;

fn invert_lower_impl<T: ComplexField>(
    dst: MatMut<'_, T>,
    cholesky_factor: Option<MatRef<'_, T>>,
    parallelism: Parallelism,
    stack: DynStack,
) {
    // (L L.*).inv() = L.inv().* L.inv()
    //
    let cholesky_factor = match cholesky_factor {
        Some(cholesky_factor) => cholesky_factor,
        None => dst.rb(),
    };
    let n = cholesky_factor.nrows();

    temp_mat_uninit! {
        let (mut tmp, _) = unsafe { temp_mat_uninit::<T>(n, n, stack) };
    }

    invert_lower_triangular_to(tmp.rb_mut(), cholesky_factor, Conj::No, parallelism);

    triangular::matmul(
        dst,
        BlockStructure::TriangularLower,
        Conj::No,
        tmp.rb().transpose(),
        BlockStructure::TriangularUpper,
        Conj::Yes,
        tmp.rb(),
        BlockStructure::TriangularLower,
        Conj::No,
        None,
        T::one(),
        parallelism,
    );
}

/// Computes the size and alignment of required workspace for computing the lower triangular part
/// of the inverse of a matrix out of place, given the Cholesky
/// decomposition.
pub fn invert_lower_to_req<T: 'static>(
    dimension: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    temp_mat_req::<T>(dimension, dimension)
}

/// Computes the size and alignment of required workspace for computing the lower triangular part
/// of the inverse of a matrix, given its Cholesky decomposition.
pub fn invert_lower_in_place_req<T: 'static>(
    dimension: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    invert_lower_to_req::<T>(dimension, parallelism)
}

/// Computes the lower triangular part of the inverse of a matrix, given its Cholesky
/// decomposition, and stores the result in `cholesky_factor`.
///
/// # Panics
///
/// - Panics if `cholesky_factor` is not a square matrix.
/// - Panics if the provided memory in `stack` is insufficient.
#[track_caller]
pub fn invert_lower_in_place<T: ComplexField>(
    cholesky_factor: MatMut<'_, T>,
    parallelism: Parallelism,
    stack: DynStack,
) {
    fancy_assert!(cholesky_factor.nrows() == cholesky_factor.ncols());
    invert_lower_impl(cholesky_factor, None, parallelism, stack);
}

/// Computes the inverse of a matrix, given its partial pivoting LU decomposition,
/// and stores the result in `lu_factors`.
///
/// # Panics
///
/// - Panics if `cholesky_factor` is not a square matrix.
/// - Panics if the destination shape doesn't match the shape of the matrix.
/// - Panics if the provided memory in `stack` is insufficient.
#[track_caller]
pub fn invert_lower_to<T: ComplexField>(
    dst: MatMut<'_, T>,
    cholesky_factor: MatRef<'_, T>,
    parallelism: Parallelism,
    stack: DynStack,
) {
    fancy_assert!(cholesky_factor.nrows() == cholesky_factor.ncols());
    fancy_assert!((dst.nrows(), dst.ncols()) == (cholesky_factor.nrows(), cholesky_factor.ncols()));
    invert_lower_impl(dst, Some(cholesky_factor), parallelism, stack);
}
