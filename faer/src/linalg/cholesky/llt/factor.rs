use crate::internal_prelude::*;
use crate::linalg::cholesky::ldlt::factor::LdltParams;
/// dynamic $LL^\top$ regularization.
///
/// values below `epsilon` in absolute value, or with the wrong sign are set to `delta` with
/// their corrected sign
#[derive(Copy, Clone, Debug)]
pub struct LltRegularization<T> {
	/// regularized value
	pub dynamic_regularization_delta: T,
	/// regularization threshold
	pub dynamic_regularization_epsilon: T,
}
/// info about the result of the $LL^\top$ factorization
#[derive(Copy, Clone, Debug)]
pub struct LltInfo {
	/// number of pivots whose value or sign had to be corrected
	pub dynamic_regularization_count: usize,
}
/// error in the $LL^\top$ factorization
#[derive(Copy, Clone, Debug)]
pub enum LltError {
	NonPositivePivot { index: usize },
}
impl core::fmt::Display for LltError {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Debug::fmt(self, f)
	}
}
impl core::error::Error for LltError {}
impl<T: RealField> Default for LltRegularization<T> {
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
	pub block_size: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
impl<T: ComplexField> Auto<T> for LltParams {
	#[inline]
	fn auto() -> Self {
		let ldlt = <LdltParams as Auto<T>>::auto();
		Self {
			recursion_threshold: ldlt.recursion_threshold,
			block_size: ldlt.block_size,
			non_exhaustive: NonExhaustive(()),
		}
	}
}
#[inline]
pub fn cholesky_in_place_scratch<T: ComplexField>(dim: usize, par: Par, params: Spec<LltParams, T>) -> StackReq {
	_ = par;
	_ = params;
	temp_mat_scratch::<T>(dim, 1)
}
pub fn cholesky_in_place<T: ComplexField>(
	A: MatMut<'_, T>,
	regularization: LltRegularization<T::Real>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<LltParams, T>,
) -> Result<LltInfo, LltError> {
	let params = params.config;
	let N = A.nrows();
	let mut D = unsafe { temp_mat_uninit(N, 1, stack).0 };
	let D = D.as_mat_mut();
	match linalg::cholesky::ldlt::factor::cholesky_block_left_looking(
		A,
		D.col_mut(0).transpose_mut(),
		params.block_size,
		params.recursion_threshold,
		params.block_size,
		true,
		regularization.dynamic_regularization_delta > zero() && regularization.dynamic_regularization_epsilon > zero(),
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
