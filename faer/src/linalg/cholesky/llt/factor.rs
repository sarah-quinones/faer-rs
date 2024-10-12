use crate::{internal_prelude::*, linalg::cholesky::ldlt::factor::cholesky_recursion, RealValue};
use core::num::NonZero;

/// Dynamic LDLT regularization.
/// Values below `epsilon` in absolute value, or with the wrong sign are set to `delta` with
/// their corrected sign.
pub struct LltRegularization<C: ComplexContainer, T: ComplexField<C>> {
    /// Regularized value.
    pub dynamic_regularization_delta: RealValue<C, T>,
    /// Regularization threshold.
    pub dynamic_regularization_epsilon: RealValue<C, T>,
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

impl<C: ComplexContainer, T: ComplexField<C>> LltRegularization<C, T> {
    #[math]
    pub fn default_with(ctx: &Ctx<C, T>) -> Self {
        Self {
            dynamic_regularization_delta: math.re(zero()),
            dynamic_regularization_epsilon: math.re(zero()),
        }
    }
}

impl<C: ComplexContainer, T: ComplexField<C, MathCtx: Default>> Default
    for LltRegularization<C, T>
{
    fn default() -> Self {
        Self::default_with(&ctx())
    }
}

#[non_exhaustive]
pub struct LltParams {
    pub blocksize: NonZero<usize>,
}

impl Default for LltParams {
    #[inline]
    fn default() -> Self {
        Self {
            blocksize: NonZero::new(64).unwrap(),
        }
    }
}

#[inline]
pub fn cholesky_in_place_scratch<C: ComplexContainer, T: ComplexField<C>>(
    dim: usize,
    par: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    temp_mat_scratch::<C, T>(dim, 1)
}

#[math]
pub fn cholesky_in_place<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    regularization: LltRegularization<C, T>,
    par: Parallelism,
    stack: &mut DynStack,
    params: LltParams,
) -> Result<LltInfo, LltError> {
    let N = A.nrows();
    let mut D = unsafe { temp_mat_uninit(ctx, N, 1, stack).0 };
    let D = D.as_mat_mut();

    help!(C::Real);
    match cholesky_recursion(
        ctx,
        A,
        D.col_mut(0).transpose_mut(),
        params.blocksize.get(),
        true,
        math.gt_zero(regularization.dynamic_regularization_delta)
            && math.gt_zero(regularization.dynamic_regularization_epsilon),
        as_ref!(regularization.dynamic_regularization_epsilon),
        as_ref!(regularization.dynamic_regularization_delta),
        None,
        par,
    ) {
        Ok(count) => Ok(LltInfo {
            dynamic_regularization_count: count,
        }),
        Err(index) => Err(LltError::NonPositivePivot { index }),
    }
}
