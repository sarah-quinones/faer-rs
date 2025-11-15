use super::LINEAR_IMPL_THRESHOLD;
use crate::internal_prelude::*;
use faer_traits::RealReg;
use num_complex::Complex;
#[inline(always)]
fn norm_max_simd<'N, T: ComplexField>(
	data: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
) -> T::Real {
	struct Impl<'a, 'N, T: ComplexField> {
		data: ColRef<'a, T, Dim<'N>, ContiguousFwd>,
	}
	impl<'N, T: ComplexField> pulp::WithSimd for Impl<'_, 'N, T> {
		type Output = T::Real;

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self { data } = self;
			let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), data.nrows());
			let mut acc = [RealReg(simd.zero()); 4];
			simd_iter!(for (IDX, i) in [simd.batch_indices(); 4] {
				let x = simd.abs_max(simd.read(data, i));
				acc[IDX] = (*simd).max(acc[IDX], x);
			});
			let acc0 = simd.max(acc[0], acc[1]);
			let acc2 = simd.max(acc[2], acc[3]);
			let acc0 = simd.max(acc0, acc2);
			simd.reduce_max_real(acc0)
		}
	}
	dispatch!(Impl { data }, Impl, T)
}
fn norm_max_simd_pairwise_rows<T: ComplexField>(
	data: ColRef<'_, T, usize, ContiguousFwd>,
) -> T::Real {
	if data.nrows() <= LINEAR_IMPL_THRESHOLD {
		with_dim!(N, data.nrows());
		norm_max_simd(data.as_row_shape(N))
	} else {
		let split_point = ((data.nrows() + 1) / 2).next_power_of_two();
		let (head, tail) = data.split_at_row(split_point);
		let acc0 = norm_max_simd_pairwise_rows(head);
		let acc1 = norm_max_simd_pairwise_rows(tail);
		acc0.fmax(acc1)
	}
}
fn norm_max_simd_pairwise_cols<T: ComplexField>(
	data: MatRef<'_, T, usize, usize, ContiguousFwd>,
) -> T::Real {
	if data.ncols() == 1 {
		norm_max_simd_pairwise_rows(data.col(0))
	} else {
		let split_point = ((data.ncols() + 1) / 2).next_power_of_two();
		let (head, tail) = data.split_at_col(split_point);
		let acc0 = norm_max_simd_pairwise_cols(head);
		let acc1 = norm_max_simd_pairwise_cols(tail);
		acc0.fmax(acc1)
	}
}
pub fn norm_max<T: ComplexField>(mut mat: MatRef<'_, T>) -> T::Real {
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
					let mat: MatRef<
						'_,
						Complex<f32>,
						usize,
						usize,
						ContiguousFwd,
					> = unsafe { crate::hacks::coerce(mat) };
					let mat = unsafe {
						MatRef::<'_, f32, usize, usize, ContiguousFwd>::from_raw_parts(
							mat.as_ptr() as *const f32,
							2 * mat.nrows(),
							mat.ncols(),
							ContiguousFwd,
							mat.col_stride().wrapping_mul(2),
						)
					};
					return unsafe {
						crate::hacks::coerce(
							norm_max_simd_pairwise_cols::<f32>(mat),
						)
					};
				} else if const { T::IS_NATIVE_C64 } {
					let mat: MatRef<
						'_,
						Complex<f64>,
						usize,
						usize,
						ContiguousFwd,
					> = unsafe { crate::hacks::coerce(mat) };
					let mat = unsafe {
						MatRef::<'_, f64, usize, usize, ContiguousFwd>::from_raw_parts(
							mat.as_ptr() as *const f64,
							2 * mat.nrows(),
							mat.ncols(),
							ContiguousFwd,
							mat.col_stride().wrapping_mul(2),
						)
					};
					return unsafe {
						crate::hacks::coerce(
							norm_max_simd_pairwise_cols::<f64>(mat),
						)
					};
				} else {
					return norm_max_simd_pairwise_cols(mat);
				}
			}
		}
		let mut acc = zero::<T::Real>();
		for j in 0..n {
			for i in 0..m {
				let val = &mat[(i, j)];
				acc = acc.fmax(val.real().abs().fmax(val.imag().abs()));
			}
		}
		acc
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Mat, assert, unzip, zip};
	#[test]
	fn test_norm_max() {
		let relative_err =
			|a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());
		for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
			for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
				let mat = Mat::from_fn(m, n, |i, j| factor * ((i + j) as f64));
				let mut target = 0.0;
				zip!(mat.as_ref()).for_each(|unzip!(x)| {
					target = f64::max(target, x.abs());
				});
				if factor == 0.0 {
					assert!(norm_max(mat.as_ref()) == target);
				} else {
					assert!(
						relative_err(norm_max(mat.as_ref()), target) < 1e-14
					);
				}
			}
		}
	}
}
