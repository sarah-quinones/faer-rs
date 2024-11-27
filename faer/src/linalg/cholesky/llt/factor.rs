use crate::{internal_prelude::*, linalg::cholesky::ldlt::factor::cholesky_recursion, Real};

/// Dynamic LDLT regularization.
/// Values below `epsilon` in absolute value, or with the wrong sign are set to `delta` with
/// their corrected sign.
pub struct LltRegularization<T: ComplexField> {
    /// Regularized value.
    pub dynamic_regularization_delta: Real<T>,
    /// Regularization threshold.
    pub dynamic_regularization_epsilon: Real<T>,
}

/// Info about the result of the LDLT factorization.
#[derive(Copy, Clone, Debug)]
pub struct LltInfo {
    /// Number of pivots whose value or sign had to be corrected.
    pub dynamic_regularization_count: usize,
}

/// Error in the LDLT factorization.
#[derive(Copy, Clone, Debug)]
pub enum LltError {
    NonPositivePivot { index: usize },
}

impl<T: ComplexField> Default for LltRegularization<T> {
    fn default() -> Self {
        Self {
            dynamic_regularization_delta: zero(),
            dynamic_regularization_epsilon: zero(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LltParams {
    pub recursion_threshold: usize,
    pub blocksize: usize,

    pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for LltParams {
    #[inline]
    fn auto() -> Self {
        Self {
            recursion_threshold: 64,
            blocksize: 128,
            non_exhaustive: NonExhaustive(()),
        }
    }
}

#[inline]
pub fn cholesky_in_place_scratch<T: ComplexField>(
    dim: usize,
    par: Par,
    params: LltParams,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    _ = params;
    temp_mat_scratch::<T>(dim, 1)
}

#[math]
pub fn cholesky_in_place<T: ComplexField>(
    A: MatMut<'_, T>,
    regularization: LltRegularization<T>,
    par: Par,
    stack: &mut DynStack,
    params: LltParams,
) -> Result<LltInfo, LltError> {
    let N = A.nrows();
    let mut D = unsafe { temp_mat_uninit(N, 1, stack).0 };
    let D = D.as_mat_mut();

    match cholesky_recursion(
        A,
        D.col_mut(0).transpose_mut(),
        params.recursion_threshold,
        params.blocksize,
        true,
        regularization.dynamic_regularization_delta > zero()
            && regularization.dynamic_regularization_epsilon > zero(),
        &regularization.dynamic_regularization_epsilon,
        &regularization.dynamic_regularization_delta,
        None,
        par,
    ) {
        Ok(count) => Ok(LltInfo {
            dynamic_regularization_count: count,
        }),
        Err(index) => Err(LltError::NonPositivePivot { index }),
    }
}
