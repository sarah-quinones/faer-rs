use faer_traits::RealReg;
use num_complex::Complex;

use super::LINEAR_IMPL_THRESHOLD;
use crate::internal_prelude::*;

#[inline(always)]
#[math]
fn norm_l1_simd<'N, T: ComplexField>(data: ColRef<'_, T, Dim<'N>, ContiguousFwd>) -> T::Real {
	struct Impl<'a, 'N, T: ComplexField> {
		data: ColRef<'a, T, Dim<'N>, ContiguousFwd>,
	}

	impl<'N, T: ComplexField> pulp::WithSimd for Impl<'_, 'N, T> {
		type Output = T::Real;

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self { data } = self;
			let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), data.nrows());

			let zero = simd.splat(&zero());

			let mut acc0 = RealReg(zero);
			let mut acc1 = RealReg(zero);
			let mut acc2 = RealReg(zero);
			let mut acc3 = RealReg(zero);

			let (head, body4, body1, tail) = simd.batch_indices::<4>();

			if let Some(i0) = head {
				let x0 = simd.abs1(simd.read(data, i0));
				acc0 = RealReg(simd.add(acc0.0, x0.0));
			}
			for [i0, i1, i2, i3] in body4 {
				let x0 = simd.abs1(simd.read(data, i0));
				let x1 = simd.abs1(simd.read(data, i1));
				let x2 = simd.abs1(simd.read(data, i2));
				let x3 = simd.abs1(simd.read(data, i3));

				acc0 = RealReg(simd.add(acc0.0, x0.0));
				acc1 = RealReg(simd.add(acc1.0, x1.0));
				acc2 = RealReg(simd.add(acc2.0, x2.0));
				acc3 = RealReg(simd.add(acc3.0, x3.0));
			}
			for i0 in body1 {
				let x0 = simd.abs1(simd.read(data, i0));
				acc0 = RealReg(simd.add(acc0.0, x0.0));
			}
			if let Some(i0) = tail {
				let x0 = simd.abs1(simd.read(data, i0));
				acc0 = RealReg(simd.add(acc0.0, x0.0));
			}

			acc0 = RealReg(simd.add(acc0.0, acc1.0));
			acc2 = RealReg(simd.add(acc2.0, acc3.0));
			acc0 = RealReg(simd.add(acc0.0, acc2.0));

			simd.reduce_sum_real(acc0)
		}
	}

	dispatch!(Impl { data }, Impl, T)
}

#[math]
fn norm_l1_simd_pairwise_rows<T: ComplexField>(data: ColRef<'_, T, usize, ContiguousFwd>) -> T::Real {
	if data.nrows() <= LINEAR_IMPL_THRESHOLD {
		with_dim!(N, data.nrows());

		norm_l1_simd(data.as_row_shape(N))
	} else {
		let split_point = ((data.nrows() + 1) / 2).next_power_of_two();
		let (head, tail) = data.split_at_row(split_point);
		let acc0 = norm_l1_simd_pairwise_rows(head);
		let acc1 = norm_l1_simd_pairwise_rows(tail);

		acc0 + acc1
	}
}

#[math]
fn norm_l1_simd_pairwise_cols<T: ComplexField>(data: MatRef<'_, T, usize, usize, ContiguousFwd>) -> T::Real {
	if data.ncols() == 1 {
		norm_l1_simd_pairwise_rows(data.col(0))
	} else {
		let split_point = ((data.ncols() + 1) / 2).next_power_of_two();
		let (head, tail) = data.split_at_col(split_point);
		let acc0 = norm_l1_simd_pairwise_cols(head);
		let acc1 = norm_l1_simd_pairwise_cols(tail);

		acc0 + acc1
	}
}

#[math]
pub fn norm_l1<T: ComplexField>(mut mat: MatRef<'_, T>) -> T::Real {
	if mat.ncols() > 1 && mat.col_stride().unsigned_abs() == 1 {
		mat = mat.transpose();
	}
	if mat.row_stride() < 0 {
		mat = mat.reverse_rows();
	}

	if mat.nrows() == 0 || mat.ncols() == 0 {
		zero()
	} else {
		let m = mat.nrows();
		let n = mat.ncols();

		if const { T::SIMD_CAPABILITIES.is_simd() } {
			if let Some(mat) = mat.try_as_col_major() {
				if const { T::IS_NATIVE_C32 } {
					let mat: MatRef<'_, Complex<f32>, usize, usize, ContiguousFwd> = unsafe { crate::hacks::coerce(mat) };
					let mat = unsafe {
						MatRef::<'_, f32, usize, usize, ContiguousFwd>::from_raw_parts(mat.as_ptr() as *const f32, 2 * mat.nrows(), mat.ncols(), ContiguousFwd, mat.col_stride().wrapping_mul(2))
					};
					return unsafe { crate::hacks::coerce(norm_l1_simd_pairwise_cols::<f32>(mat)) };
				} else if const { T::IS_NATIVE_C64 } {
					let mat: MatRef<'_, Complex<f64>, usize, usize, ContiguousFwd> = unsafe { crate::hacks::coerce(mat) };
					let mat = unsafe {
						MatRef::<'_, f64, usize, usize, ContiguousFwd>::from_raw_parts(mat.as_ptr() as *const f64, 2 * mat.nrows(), mat.ncols(), ContiguousFwd, mat.col_stride().wrapping_mul(2))
					};
					return unsafe { crate::hacks::coerce(norm_l1_simd_pairwise_cols::<f64>(mat)) };
				} else {
					return norm_l1_simd_pairwise_cols(mat);
				}
			}
		}

		let mut acc = zero();
		for j in 0..n {
			for i in 0..m {
				acc = acc + abs1(mat[(i, j)]);
			}
		}
		acc
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Col, Mat, assert, unzipped, zipped};

	#[test]
	fn test_norm_l1() {
		let relative_err = |a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());

		for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
			for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
				let mat = Mat::from_fn(m, n, |i, j| factor * ((i + j) as f64));
				let mut target = 0.0;
				zipped!(mat.as_ref()).for_each(|unzipped!(x)| {
					target += x.abs();
				});

				if factor == 0.0 {
					assert!(norm_l1(mat.as_ref()) == target);
				} else {
					assert!(relative_err(norm_l1(mat.as_ref()), target) < 1e-14);
				}
			}
		}

		let mat = Col::from_fn(10000000, |_| 0.3);
		let target = 0.3 * 10000000.0f64;
		assert!(relative_err(norm_l1(mat.as_ref().as_mat()), target) < 1e-14);
	}
}
