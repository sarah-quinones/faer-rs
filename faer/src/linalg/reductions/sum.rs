use faer_traits::Unit;
use num_complex::Complex;

use super::LINEAR_IMPL_THRESHOLD;
use crate::internal_prelude::*;

#[inline(always)]
#[math]
fn sum_simd<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    data: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
) -> C::Of<T> {
    struct Impl<'a, 'N, C: ComplexContainer, T: ComplexField<C>> {
        ctx: &'a Ctx<C, T>,
        data: ColRef<'a, C, T, Dim<'N>, ContiguousFwd>,
    }

    impl<'N, C: ComplexContainer, T: ComplexField<C>> pulp::WithSimd for Impl<'_, 'N, C, T> {
        type Output = C::Of<T>;
        #[inline(always)]
        #[math]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { ctx, data } = self;
            let simd = SimdCtx::<C, T, S>::new(T::simd_ctx(ctx, simd), data.nrows());

            help!(C);
            let zero = simd.splat(as_ref!(math.zero()));

            let mut acc0 = zero;
            let mut acc1 = zero;
            let mut acc2 = zero;
            let mut acc3 = zero;

            let (head, body4, body1, tail) = simd.batch_indices::<4>();
            if let Some(i0) = head {
                let x0 = simd.read(data, i0);
                acc0 = simd.add(acc0, x0);
            }
            for [i0, i1, i2, i3] in body4 {
                let x0 = simd.read(data, i0);
                let x1 = simd.read(data, i1);
                let x2 = simd.read(data, i2);
                let x3 = simd.read(data, i3);

                acc0 = simd.add(acc0, x0);
                acc1 = simd.add(acc1, x1);
                acc2 = simd.add(acc2, x2);
                acc3 = simd.add(acc3, x3);
            }
            for i0 in body1 {
                let x0 = simd.read(data, i0);
                acc0 = simd.add(acc0, x0);
            }
            if let Some(i0) = tail {
                let x0 = simd.read(data, i0);
                acc0 = simd.add(acc0, x0);
            }
            acc0 = simd.add(acc0, acc1);
            acc2 = simd.add(acc2, acc3);
            acc0 = simd.add(acc0, acc2);

            simd.reduce_sum(acc0)
        }
    }

    T::Arch::default().dispatch(Impl { ctx, data })
}

#[math]
fn sum_simd_pairwise_rows<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    data: ColRef<'_, C, T, usize, ContiguousFwd>,
) -> C::Of<T> {
    if data.nrows() <= LINEAR_IMPL_THRESHOLD {
        with_dim!(N, data.nrows());

        sum_simd(ctx, data.as_row_shape(N))
    } else {
        let split_point = ((data.nrows() + 1) / 2).next_power_of_two();
        let (head, tail) = data.split_at_row(split_point);
        let acc0 = sum_simd_pairwise_rows(ctx, head);
        let acc1 = sum_simd_pairwise_rows(ctx, tail);

        math(acc0 + acc1)
    }
}

#[math]
fn sum_simd_pairwise_cols<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    data: MatRef<'_, C, T, usize, usize, ContiguousFwd>,
) -> C::Of<T> {
    if data.ncols() == 1 {
        sum_simd_pairwise_rows(ctx, data.col(0))
    } else {
        let split_point = ((data.ncols() + 1) / 2).next_power_of_two();
        let (head, tail) = data.split_at_col(split_point);
        let acc0 = sum_simd_pairwise_cols(ctx, head);
        let acc1 = sum_simd_pairwise_cols(ctx, tail);

        math(acc0 + acc1)
    }
}

#[math]
pub fn sum<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    mut mat: MatRef<'_, C, T>,
) -> C::Of<T> {
    if mat.ncols() > 1 && mat.col_stride().unsigned_abs() == 1 {
        mat = mat.transpose();
    }
    if mat.row_stride() < 0 {
        mat = mat.reverse_rows();
    }

    if mat.nrows() == 0 || mat.ncols() == 0 {
        math.zero()
    } else {
        let m = mat.nrows();
        let n = mat.ncols();

        if const { T::SIMD_CAPABILITIES.is_simd() } {
            if let Some(mat) = mat.try_as_col_major() {
                if const { T::IS_NATIVE_C32 } {
                    let mat: MatRef<'_, Unit, Complex<f32>, usize, usize, ContiguousFwd> =
                        unsafe { crate::hacks::coerce(mat) };
                    let mat = unsafe {
                        MatRef::<'_, Unit, f32, usize, usize, ContiguousFwd>::from_raw_parts(
                            mat.as_ptr() as *const f32,
                            2 * mat.nrows(),
                            mat.ncols(),
                            ContiguousFwd,
                            mat.col_stride().wrapping_mul(2),
                        )
                    };
                    return unsafe {
                        crate::hacks::coerce(sum_simd_pairwise_cols::<Unit, f32>(&Ctx(Unit), mat))
                    };
                } else if const { T::IS_NATIVE_C64 } {
                    let mat: MatRef<'_, Unit, Complex<f64>, usize, usize, ContiguousFwd> =
                        unsafe { crate::hacks::coerce(mat) };
                    let mat = unsafe {
                        MatRef::<'_, Unit, f64, usize, usize, ContiguousFwd>::from_raw_parts(
                            mat.as_ptr() as *const f64,
                            2 * mat.nrows(),
                            mat.ncols(),
                            ContiguousFwd,
                            mat.col_stride().wrapping_mul(2),
                        )
                    };
                    return unsafe {
                        crate::hacks::coerce(sum_simd_pairwise_cols::<Unit, f64>(&Ctx(Unit), mat))
                    };
                } else if const { C::IS_COMPLEX } {
                    let mat: MatRef<
                        num_complex::Complex<C::Real>,
                        T::RealUnit,
                        usize,
                        usize,
                        ContiguousFwd,
                    > = unsafe { crate::hacks::coerce(mat) };
                    let (re, im) = super::real_imag(mat);
                    return unsafe {
                        crate::hacks::coerce(core::mem::ManuallyDrop::new(
                            num_complex::Complex::new(
                                sum_simd_pairwise_cols::<C::Real, T::RealUnit>(
                                    Ctx::new(&**ctx),
                                    re,
                                ),
                                sum_simd_pairwise_cols::<C::Real, T::RealUnit>(
                                    Ctx::new(&**ctx),
                                    im,
                                ),
                            ),
                        ))
                    };
                } else {
                    return sum_simd_pairwise_cols(ctx, mat);
                }
            }
        }

        let mut acc = math.zero();
        for j in 0..n {
            for i in 0..m {
                let val = mat.at(i, j);

                acc = math(acc + val);
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, unzipped, zipped, Col, Mat};

    #[test]
    fn test_sum() {
        let relative_err = |a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());

        for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
            for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
                let mat = Mat::from_fn(m, n, |i, j| factor * ((i + j) as f64));
                let mut target = 0.0;
                zipped!(mat.as_ref()).for_each(|unzipped!(x)| {
                    target += x;
                });

                if factor == 0.0 {
                    assert!(sum(&default(), mat.as_ref()) == target);
                } else {
                    assert!(relative_err(sum(&default(), mat.as_ref()), target) < 1e-14);
                }
            }
        }

        let mat = Col::from_fn(10000000, |_| 0.3);
        let target = 0.3 * 10000000.0f64;
        assert!(relative_err(sum(&default(), mat.as_ref().as_mat()), target) < 1e-14);
    }
}
