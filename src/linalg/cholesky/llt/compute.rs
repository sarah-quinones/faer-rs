use super::CholeskyError;
use crate::{
    assert, debug_assert,
    linalg::{
        cholesky::ldlt_diagonal::compute::new_cholesky,
        entity::{self, *},
        matmul::triangular::BlockStructure,
        triangular_solve,
    },
    utils::DivCeil,
    ComplexField, Entity, MatMut, Parallelism,
};
use core::marker::PhantomData;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use pulp::Simd;
use reborrow::*;

fn cholesky_in_place_left_looking_impl<E: ComplexField>(
    offset: usize,
    matrix: MatMut<'_, E>,
    regularization: LltRegularization<E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: LltParams,
) -> Result<usize, CholeskyError> {
    _ = params;
    _ = parallelism;
    match new_cholesky(matrix, &regularization, stack) {
        Ok(dyn_reg_count) => Ok(dyn_reg_count),
        Err(mut e) => {
            e.non_positive_definite_minor += offset;
            Err(e)
        }
    }
}

/// LLT factorization tuning parameters.
#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct LltParams {}

/// Dynamic LLT regularization.
/// Values below `epsilon` in absolute value, or with a negative sign are set to `delta` with
/// a positive sign.
#[derive(Copy, Clone, Debug)]
pub struct LltRegularization<E: ComplexField> {
    /// Regularized value.
    pub dynamic_regularization_delta: E::Real,
    /// Regularization threshold.
    pub dynamic_regularization_epsilon: E::Real,
}

impl<E: ComplexField> Default for LltRegularization<E> {
    fn default() -> Self {
        Self {
            dynamic_regularization_delta: E::Real::faer_zero(),
            dynamic_regularization_epsilon: E::Real::faer_zero(),
        }
    }
}

/// Computes the size and alignment of required workspace for performing a Cholesky
/// decomposition.
pub fn cholesky_in_place_req<E: Entity>(
    dim: usize,
    parallelism: Parallelism,
    params: LltParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    let _ = params;
    // only the left looking impl allocates
    let dim = Ord::min(dim, 64);
    crate::linalg::temp_mat_req::<E>(dim, dim)?.try_and(StackReq::try_new::<E>(dim)?)
}

// uses an out parameter for tail recursion
fn cholesky_in_place_impl<E: ComplexField>(
    offset: usize,
    count: &mut usize,
    matrix: MatMut<'_, E>,
    regularization: LltRegularization<E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: LltParams,
) -> Result<(), CholeskyError> {
    // right looking cholesky

    debug_assert!(matrix.nrows() == matrix.ncols());
    let mut matrix = matrix;
    let mut stack = stack;

    struct Lanes<E> {
        __marker: PhantomData<E>,
    }
    impl<E: ComplexField> pulp::WithSimd for Lanes<E> {
        type Output = usize;
        fn with_simd<S: Simd>(self, _: S) -> Self::Output {
            core::mem::size_of::<entity::SimdUnitFor<E, S>>() / core::mem::size_of::<E::Unit>()
        }
    }

    let lanes = E::Simd::default().dispatch(Lanes {
        __marker: PhantomData::<E>,
    });
    let stride = matrix.nrows().msrv_div_ceil(lanes);

    let n = matrix.nrows();
    if stride <= 4 && n <= 64 {
        *count += cholesky_in_place_left_looking_impl(
            offset,
            matrix,
            regularization,
            parallelism,
            stack,
            params,
        )?;
        Ok(())
    } else {
        let block_size = n / 2;
        let (mut l00, _, mut a10, mut a11) = matrix.rb_mut().split_at_mut(block_size, block_size);

        cholesky_in_place_impl(
            offset,
            count,
            l00.rb_mut(),
            regularization,
            parallelism,
            stack.rb_mut(),
            params,
        )?;

        let l00 = l00.into_const();

        triangular_solve::solve_lower_triangular_in_place(
            l00.conjugate(),
            a10.rb_mut().transpose_mut(),
            parallelism,
        );

        crate::linalg::matmul::triangular::matmul(
            a11.rb_mut(),
            BlockStructure::TriangularLower,
            a10.rb(),
            BlockStructure::Rectangular,
            a10.rb().adjoint(),
            BlockStructure::Rectangular,
            Some(E::faer_one()),
            E::faer_one().faer_neg(),
            parallelism,
        );

        cholesky_in_place_impl(
            offset + block_size,
            count,
            a11,
            regularization,
            parallelism,
            stack,
            params,
        )
    }
}

/// Info about the result of the LLT factorization.
#[derive(Copy, Clone, Debug)]
pub struct LltInfo {
    /// Number of pivots whose value or sign had to be corrected.
    pub dynamic_regularization_count: usize,
}

/// Computes the Cholesky factor $L$ of a Hermitian positive definite input matrix $A$ such that
/// $L$ is lower triangular, and
/// $$LL^H == A.$$
///
/// The result is stored back in the lower half of the same matrix, or an error is returned if the
/// matrix is not positive definite.
///
/// The input matrix is interpreted as Hermitian with the values being extracted from the lower
/// part, but the entire matrix is required to be initialized.
///
/// The strictly upper triangular part of the matrix is clobbered and may be filled with garbage
/// values.
///
/// # Panics
///
/// Panics if the input matrix is not square.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`cholesky_in_place_req`]).
#[track_caller]
#[inline]
pub fn cholesky_in_place<E: ComplexField>(
    matrix: MatMut<'_, E>,
    regularization: LltRegularization<E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: LltParams,
) -> Result<LltInfo, CholeskyError> {
    let _ = params;
    assert!(matrix.ncols() == matrix.nrows());
    #[cfg(feature = "perf-warn")]
    if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(CHOLESKY_WARN) {
        if matrix.col_stride().unsigned_abs() == 1 {
            log::warn!(target: "faer_perf", "LLT prefers column-major matrix. Found row-major matrix.");
        } else {
            log::warn!(target: "faer_perf", "LLT prefers column-major matrix. Found matrix with generic strides.");
        }
    }

    let mut count = 0;
    cholesky_in_place_impl(
        0,
        &mut count,
        matrix,
        regularization,
        parallelism,
        stack,
        params,
    )?;
    Ok(LltInfo {
        dynamic_regularization_count: count,
    })
}
