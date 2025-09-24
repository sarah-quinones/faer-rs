use super::*;

pub(crate) mod complex_schur;
pub(crate) mod real_schur;

#[derive(Clone, Copy, Debug)]
pub struct SchurParams {
	/// function that returns the number of shifts to use for a given matrix size
	pub recommended_shift_count: extern "C" fn(matrix_dimension: usize, active_block_dimension: usize) -> usize,
	/// function that returns the deflation window to use for a given matrix size
	pub recommended_deflation_window: extern "C" fn(matrix_dimension: usize, active_block_dimension: usize) -> usize,
	/// threshold to switch between blocked and unblocked code
	pub blocking_threshold: usize,
	/// threshold of percent of aggressive-early-deflation window that must converge to skip a
	/// sweep
	pub nibble_threshold: usize,

	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for SchurParams {
	fn auto() -> Self {
		Self {
			recommended_shift_count: default_recommended_shift_count,
			recommended_deflation_window: default_recommended_deflation_window,
			blocking_threshold: 75,
			nibble_threshold: 50,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

pub fn multishift_qr_scratch<T: ComplexField>(n: usize, nh: usize, want_t: bool, want_z: bool, parallelism: Par, params: SchurParams) -> StackReq {
	let nsr = (params.recommended_shift_count)(n, nh);

	let _ = want_t;
	let _ = want_z;

	if n <= 3 {
		return StackReq::EMPTY;
	}

	let nw_max = (n - 3) / 3;

	StackReq::any_of(&[
		hessenberg::hessenberg_in_place_scratch::<T>(nw_max, 1, parallelism, Default::default()),
		linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(nw_max, nw_max, nw_max),
		linalg::householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<T>(nw_max, nw_max, nw_max),
		temp_mat_scratch::<T>(3, nsr),
	])
}

extern "C" fn default_recommended_shift_count(dim: usize, _active_block_dim: usize) -> usize {
	let n = dim;
	if n < 30 {
		2
	} else if n < 60 {
		4
	} else if n < 150 {
		12
	} else if n < 590 {
		32
	} else if n < 3000 {
		64
	} else if n < 6000 {
		128
	} else {
		256
	}
}

extern "C" fn default_recommended_deflation_window(dim: usize, _active_block_dim: usize) -> usize {
	let n = dim;
	if n < 30 {
		2
	} else if n < 60 {
		4
	} else if n < 150 {
		10
	} else if n < 590 {
		#[cfg(feature = "std")]
		{
			(n as f64 / (n as f64).log2()) as usize
		}
		#[cfg(not(feature = "std"))]
		{
			libm::log2(n as f64 / (n as f64)) as usize
		}
	} else if n < 3000 {
		96
	} else if n < 6000 {
		192
	} else {
		384
	}
}
