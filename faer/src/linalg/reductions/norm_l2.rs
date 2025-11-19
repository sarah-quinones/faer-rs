use super::LINEAR_IMPL_THRESHOLD;
use crate::internal_prelude::*;
use faer_traits::RealReg;
use num_complex::Complex;
#[inline(always)]
fn norm_l2_simd<'N, T: ComplexField>(
	data: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
) -> [T::Real; 3] {
	struct Impl<'a, 'N, T: ComplexField> {
		data: ColRef<'a, T, Dim<'N>, ContiguousFwd>,
	}
	impl<'N, T: ComplexField> pulp::WithSimd for Impl<'_, 'N, T> {
		type Output = [T::Real; 3];

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self { data } = self;
			let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), data.nrows());
			let sml = simd.splat_real(&sqrt_min_positive());
			let big = simd.splat_real(&sqrt_max_positive());
			let mut acc_sml = [RealReg(simd.zero()); 2];
			let mut acc_med = [RealReg(simd.zero()); 2];
			let mut acc_big = [RealReg(simd.zero()); 2];

			simd_iter!(for (IDX, i) in [simd.batch_indices::<2>(); 2] {
				let x = simd.read(data, i);
				acc_sml[IDX] =
					simd.abs2_add(simd.mul_real(x, sml), acc_sml[IDX]);
				acc_med[IDX] = simd.abs2_add(x, acc_med[IDX]);
				acc_big[IDX] =
					simd.abs2_add(simd.mul_real(x, big), acc_big[IDX]);
			});

			let acc0_sml = RealReg(simd.add(acc_sml[0].0, acc_sml[1].0));
			let acc0_big = RealReg(simd.add(acc_big[0].0, acc_big[1].0));
			let acc0_med = RealReg(simd.add(acc_med[0].0, acc_med[1].0));
			[
				simd.reduce_sum_real(acc0_sml),
				simd.reduce_sum_real(acc0_med),
				simd.reduce_sum_real(acc0_big),
			]
		}
	}
	dispatch!(Impl { data }, Impl, T)
}
fn norm_l2_simd_pairwise_rows<T: ComplexField>(
	data: ColRef<'_, T, usize, ContiguousFwd>,
) -> [T::Real; 3] {
	if data.nrows() <= LINEAR_IMPL_THRESHOLD {
		with_dim!(N, data.nrows());
		norm_l2_simd(data.as_row_shape(N))
	} else {
		let split_point = ((data.nrows() + 1) / 2).next_power_of_two();
		let (head, tail) = data.split_at_row(split_point);
		let acc0 = norm_l2_simd_pairwise_rows(head);
		let acc1 = norm_l2_simd_pairwise_rows(tail);
		[
			&acc0[0] + &acc1[0],
			&acc0[1] + &acc1[1],
			&acc0[2] + &acc1[2],
		]
	}
}
fn norm_l2_simd_pairwise_cols<T: ComplexField>(
	data: MatRef<'_, T, usize, usize, ContiguousFwd>,
) -> [T::Real; 3] {
	if data.ncols() == 1 {
		norm_l2_simd_pairwise_rows(data.col(0))
	} else {
		let split_point = ((data.ncols() + 1) / 2).next_power_of_two();
		let (head, tail) = data.split_at_col(split_point);
		let acc0 = norm_l2_simd_pairwise_cols(head);
		let acc1 = norm_l2_simd_pairwise_cols(tail);
		[
			&acc0[0] + &acc1[0],
			&acc0[1] + &acc1[1],
			&acc0[2] + &acc1[2],
		]
	}
}
pub fn norm_l2_x3<T: ComplexField>(mut mat: MatRef<'_, T>) -> [T::Real; 3] {
	if mat.ncols() > 1 && mat.col_stride().unsigned_abs() == 1 {
		mat = mat.transpose();
	}
	if mat.row_stride() < 0 {
		mat = mat.reverse_rows();
	}
	if mat.nrows() == 0 || mat.ncols() == 0 {
		[zero(), zero(), zero()]
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
						crate::hacks::coerce(norm_l2_simd_pairwise_cols::<f32>(
							mat,
						))
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
						crate::hacks::coerce(norm_l2_simd_pairwise_cols::<f64>(
							mat,
						))
					};
				} else {
					return norm_l2_simd_pairwise_cols(mat);
				}
			}
		}
		let ref sml = min_positive::<T::Real>();
		let ref big = max_positive::<T::Real>();
		let mut acc = zero::<T::Real>();
		for j in 0..n {
			for i in 0..m {
				acc = acc.hypot(mat[(i, j)].abs());
			}
		}
		acc = acc.abs2();
		[sml * &acc, acc.copy(), big * &acc]
	}
}
pub fn norm_l2<T: ComplexField>(mat: MatRef<'_, T>) -> T::Real {
	let [acc_sml, acc_med, acc_big] = norm_l2_x3(mat);
	let sml = sqrt_min_positive::<T::Real>();
	let big = sqrt_max_positive::<T::Real>();
	if acc_sml >= one() {
		acc_sml.sqrt() * big
	} else if acc_med >= one() {
		acc_med.sqrt()
	} else {
		acc_big.sqrt() * sml
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Col, Mat, assert, c64};

	#[test]
	fn test_norm_l2() {
		let relative_err =
			|a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());
		for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
			for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
				let mat = Mat::from_fn(m, n, |i, j| factor * ((i + j) as f64));
				let mut target = 0.0;
				zip!(mat.as_ref()).for_each(|unzip!(x)| {
					target = f64::hypot(*x, target);
				});
				if factor == 0.0 {
					assert!(norm_l2(mat.as_ref()) == target);
				} else {
					assert!(
						relative_err(norm_l2(mat.as_ref()), target) < 1e-14
					);
				}
			}
		}
		let mat = Col::from_fn(10000000, |_| 0.3);
		let target = (0.3 * 0.3 * 10000000.0f64).sqrt();
		assert!(relative_err(norm_l2(mat.as_ref().as_mat()), target) < 1e-14);
	}
	#[test]
	fn test_norm_l2_cplx() {
		let relative_err =
			|a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());
		for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
			for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
				let mat = Mat::from_fn(m, n, |i, j| {
					factor * c64::new((i + j) as f64, i.wrapping_sub(j) as f64)
				});
				let mut target = 0.0;
				zip!(mat.as_ref()).for_each(|unzip!(x)| {
					target = f64::hypot(x.abs(), target);
				});
				if factor == 0.0 {
					assert!(norm_l2(mat.as_ref()) == target);
				} else {
					assert!(
						relative_err(norm_l2(mat.as_ref()), target) < 1e-14
					);
				}
			}
		}
	}
}
