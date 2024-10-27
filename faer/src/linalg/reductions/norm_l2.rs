use faer_traits::Real;
use num_complex::Complex;

use super::LINEAR_IMPL_THRESHOLD;
use crate::internal_prelude::*;

#[inline(always)]
#[math]
fn norm_l2_simd<'N, T: ComplexField>(data: ColRef<'_, T, Dim<'N>, ContiguousFwd>) -> [T::Real; 3] {
    struct Impl<'a, 'N, T: ComplexField> {
        data: ColRef<'a, T, Dim<'N>, ContiguousFwd>,
    }

    impl<'N, T: ComplexField> pulp::WithSimd for Impl<'_, 'N, T> {
        type Output = [T::Real; 3];
        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { data } = self;
            let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), data.nrows());

            let zero = simd.splat(&zero());

            let sml = simd.splat_real(&sqrt_min_positive());
            let big = simd.splat_real(&sqrt_max_positive());

            let mut acc0_sml = Real(zero);
            let mut acc1_sml = Real(zero);
            let mut acc0_med = Real(zero);
            let mut acc1_med = Real(zero);
            let mut acc0_big = Real(zero);
            let mut acc1_big = Real(zero);

            let (head, body2, body1, tail) = simd.batch_indices::<2>();

            if let Some(i0) = head {
                let x0 = simd.abs1(simd.read(data, i0));

                acc0_sml = simd.abs2_add(simd.mul_real(x0.0, sml), acc0_sml);
                acc0_med = simd.abs2_add(x0.0, acc0_med);
                acc0_big = simd.abs2_add(simd.mul_real(x0.0, big), acc0_big);
            }
            for [i0, i1] in body2 {
                let x0 = simd.abs1(simd.read(data, i0));
                let x1 = simd.abs1(simd.read(data, i1));

                acc0_sml = simd.abs2_add(simd.mul_real(x0.0, sml), acc0_sml);
                acc1_sml = simd.abs2_add(simd.mul_real(x1.0, sml), acc1_sml);

                acc0_med = simd.abs2_add(x0.0, acc0_med);
                acc1_med = simd.abs2_add(x1.0, acc1_med);

                acc0_big = simd.abs2_add(simd.mul_real(x0.0, big), acc0_big);
                acc1_big = simd.abs2_add(simd.mul_real(x1.0, big), acc1_big);
            }
            for i0 in body1 {
                let x0 = simd.abs1(simd.read(data, i0));

                acc0_sml = simd.abs2_add(simd.mul_real(x0.0, sml), acc0_sml);
                acc0_med = simd.abs2_add(x0.0, acc0_med);
                acc0_big = simd.abs2_add(simd.mul_real(x0.0, big), acc0_big);
            }
            if let Some(i0) = tail {
                let x0 = simd.abs1(simd.read(data, i0));

                acc0_sml = simd.abs2_add(simd.mul_real(x0.0, sml), acc0_sml);
                acc0_med = simd.abs2_add(x0.0, acc0_med);
                acc0_big = simd.abs2_add(simd.mul_real(x0.0, big), acc0_big);
            }

            acc0_sml = Real(simd.add(acc0_sml.0, acc1_sml.0));
            acc0_big = Real(simd.add(acc0_big.0, acc1_big.0));
            acc0_med = Real(simd.add(acc0_med.0, acc1_med.0));
            [
                real(simd.reduce_sum(acc0_sml.0)),
                real(simd.reduce_sum(acc0_med.0)),
                real(simd.reduce_sum(acc0_big.0)),
            ]
        }
    }

    T::Arch::default().dispatch(Impl { data })
}

#[math]
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
            add(acc0[0], acc1[0]),
            add(acc0[1], acc1[1]),
            add(acc0[2], acc1[2]),
        ]
    }
}

#[math]
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
            add(acc0[0], acc1[0]),
            add(acc0[1], acc1[1]),
            add(acc0[2], acc1[2]),
        ]
    }
}

#[math]
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
                    let mat: MatRef<'_, Complex<f32>, usize, usize, ContiguousFwd> =
                        unsafe { crate::hacks::coerce(mat) };
                    let mat = unsafe {
                        MatRef::<'_, f32, usize, usize, ContiguousFwd>::from_raw_parts(
                            mat.as_ptr() as *const f32,
                            2 * mat.nrows(),
                            mat.ncols(),
                            ContiguousFwd,
                            mat.col_stride().wrapping_mul(2),
                        )
                    };
                    return unsafe { crate::hacks::coerce(norm_l2_simd_pairwise_cols::<f32>(mat)) };
                } else if const { T::IS_NATIVE_C64 } {
                    let mat: MatRef<'_, Complex<f64>, usize, usize, ContiguousFwd> =
                        unsafe { crate::hacks::coerce(mat) };
                    let mat = unsafe {
                        MatRef::<'_, f64, usize, usize, ContiguousFwd>::from_raw_parts(
                            mat.as_ptr() as *const f64,
                            2 * mat.nrows(),
                            mat.ncols(),
                            ContiguousFwd,
                            mat.col_stride().wrapping_mul(2),
                        )
                    };
                    return unsafe { crate::hacks::coerce(norm_l2_simd_pairwise_cols::<f64>(mat)) };
                } else {
                    return norm_l2_simd_pairwise_cols(mat);
                }
            }
        }

        let sml = min_positive();
        let big = max_positive();
        let mut acc = zero();
        for j in 0..n {
            for i in 0..m {
                acc = hypot(acc, abs(mat[(i, j)]));
            }
        }
        acc = abs2(acc);
        [sml * acc, copy(acc), big * acc]
    }
}

#[math]
pub fn norm_l2<T: ComplexField>(mat: MatRef<'_, T>) -> T::Real {
    let [acc_sml, acc_med, acc_big] = norm_l2_x3(mat);

    let sml = sqrt_min_positive();
    let big = sqrt_max_positive();

    if acc_sml >= one() {
        sqrt(acc_sml) * big
    } else if acc_med >= one() {
        sqrt(acc_med)
    } else {
        sqrt(acc_big) * sml
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, unzipped, zipped, Col, Mat};

    #[test]
    fn test_norm_l2() {
        let relative_err = |a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());

        for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
            for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
                let mat = Mat::from_fn(m, n, |i, j| factor * ((i + j) as f64));
                let mut target = 0.0;
                zipped!(mat.as_ref()).for_each(|unzipped!(x)| {
                    target = f64::hypot(*x, target);
                });

                if factor == 0.0 {
                    assert!(norm_l2(mat.as_ref()) == target);
                } else {
                    assert!(relative_err(norm_l2(mat.as_ref()), target) < 1e-14);
                }
            }
        }

        let mat = Col::from_fn(10000000, |_| 0.3);
        let target = (0.3 * 0.3 * 10000000.0f64).sqrt();
        assert!(relative_err(norm_l2(mat.as_ref().as_mat()), target) < 1e-14);
    }
}
