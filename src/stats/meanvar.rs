use crate::{
    linalg::entity::{pulp, SimdCtx, SimdGroupFor},
    prelude::*,
    utils::slice::{RefGroup, SliceGroup},
    ComplexField,
};
use equator::assert;
use pulp::Read;

/// Computes the mean of the columns of `mat` and stores the result in `out`.
#[track_caller]
pub fn col_mean<E: ComplexField>(out: ColMut<'_, E>, mat: MatRef<'_, E>) {
    assert!(all(out.nrows() == mat.nrows(), mat.ncols() > 0));

    fn col_mean_row_major<E: ComplexField>(out: ColMut<'_, E>, mat: MatRef<'_, E>) {
        struct Impl<'a, E: ComplexField> {
            out: ColMut<'a, E>,
            mat: MatRef<'a, E>,
        }

        impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self { mut out, mat } = self;
                let simd = crate::utils::simd::SimdFor::<E, S>::new(simd);

                let m = mat.nrows();
                let n = mat.ncols();
                let one_n = E::Real::faer_from_f64(n as f64).faer_inv();

                let offset = simd.align_offset_ptr(mat.as_ptr(), mat.ncols());
                for i in 0..m {
                    let row = SliceGroup::<'_, E>::new(mat.row(i).try_as_slice().unwrap());
                    let (head, body, tail) = simd.as_aligned_simd(row, offset);
                    let mut sum0 = head.read_or(simd.splat(E::faer_zero()));
                    let mut sum1 = simd.splat(E::faer_zero());
                    let mut sum2 = simd.splat(E::faer_zero());
                    let mut sum3 = simd.splat(E::faer_zero());

                    let (body4, body1) = body.as_arrays::<4>();
                    for [x0, x1, x2, x3] in body4.into_ref_iter().map(RefGroup::unzip) {
                        sum0 = simd.add(sum0, x0.get());
                        sum1 = simd.add(sum1, x1.get());
                        sum2 = simd.add(sum2, x2.get());
                        sum3 = simd.add(sum3, x3.get());
                    }
                    for x0 in body1.into_ref_iter() {
                        sum0 = simd.add(sum0, x0.get());
                    }
                    sum0 = simd.add(sum0, tail.read_or(simd.splat(E::faer_zero())));

                    sum0 = simd.add(sum0, sum1);
                    sum2 = simd.add(sum2, sum3);
                    sum0 = simd.add(sum0, sum2);

                    sum0 = simd.rotate_left(sum0, offset.rotate_left_amount());
                    let sum = simd.reduce_add(sum0);

                    out.write(i, sum.faer_scale_real(one_n));
                }
            }
        }

        E::Simd::default().dispatch(Impl { out, mat });
    }

    let mat = if mat.col_stride() >= 0 {
        mat
    } else {
        mat.reverse_cols()
    };
    if mat.col_stride() == 1 {
        col_mean_row_major(out, mat)
    } else {
        let mut out = out;

        let n = mat.ncols();
        let one_n = E::Real::faer_from_f64(n as f64).faer_inv();

        out.fill_zero();
        for j in 0..n {
            out += mat.col(j);
        }
        zipped!(out).for_each(|unzipped!(mut x)| x.write(x.read().faer_scale_real(one_n)));
    }
}

/// Computes the mean of the rows of `mat` and stores the result in `out`.
#[track_caller]
pub fn row_mean<E: ComplexField>(out: RowMut<'_, E>, mat: MatRef<'_, E>) {
    assert!(all(out.ncols() == mat.ncols(), mat.nrows() > 0));
    col_mean(out.transpose_mut(), mat.transpose());
}

/// Computes the variance of the columns of `mat` given their mean, and stores the result in `out`.
#[track_caller]
pub fn col_varm<E: ComplexField>(
    out: ColMut<'_, E::Real>,
    mat: MatRef<'_, E>,
    col_mean: ColRef<'_, E>,
) {
    assert!(all(
        mat.ncols() > 0,
        out.nrows() == mat.nrows(),
        col_mean.nrows() == mat.nrows(),
    ));

    fn col_varm_row_major<E: ComplexField>(
        out: ColMut<'_, E::Real>,
        mat: MatRef<'_, E>,
        col_mean: ColRef<'_, E>,
    ) {
        struct Impl<'a, E: ComplexField> {
            out: ColMut<'a, E::Real>,
            mat: MatRef<'a, E>,
            col_mean: ColRef<'a, E>,
        }

        impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self {
                    mut out,
                    mat,
                    col_mean,
                } = self;

                let simd_real = crate::utils::simd::SimdFor::<E::Real, S>::new(simd);
                let simd = crate::utils::simd::SimdFor::<E, S>::new(simd);

                let m = mat.nrows();
                let n = mat.ncols();
                let one_n1 = E::Real::faer_from_f64((n - 1) as f64).faer_inv();

                let offset = simd.align_offset_ptr(mat.as_ptr(), mat.ncols());
                for i in 0..m {
                    let mean = simd.splat(col_mean.read(i));
                    let row = SliceGroup::<'_, E>::new(mat.row(i).try_as_slice().unwrap());
                    let (head, body, tail) = simd.as_aligned_simd(row, offset);

                    #[inline(always)]
                    fn process<E: ComplexField, S: pulp::Simd>(
                        simd: crate::utils::simd::SimdFor<E, S>,
                        acc: SimdGroupFor<E::Real, S>,
                        mean: SimdGroupFor<E, S>,
                        val: impl Read<Output = SimdGroupFor<E, S>>,
                    ) -> SimdGroupFor<E::Real, S> {
                        let diff = simd.sub(val.read_or(mean), mean);
                        if coe::is_same::<E, c32>() {
                            let diff = coe::coerce_static::<SimdGroupFor<E, S>, SimdGroupFor<c32, S>>(
                                diff,
                            );
                            let acc = coe::coerce_static::<
                                SimdGroupFor<E::Real, S>,
                                SimdGroupFor<f32, S>,
                            >(acc);

                            if coe::is_same::<S, pulp::Scalar>() {
                                let diff: c32 = bytemuck::cast(diff);
                                let acc: f32 = bytemuck::cast(acc);

                                coe::coerce_static::<
                                    SimdGroupFor<f32, pulp::Scalar>,
                                    SimdGroupFor<E::Real, S>,
                                >(diff.faer_abs2() + acc)
                            } else {
                                let diff: S::f32s = bytemuck::cast(diff);
                                coe::coerce_static::<SimdGroupFor<f32, S>, SimdGroupFor<E::Real, S>>(
                                    simd.simd.f32s_mul_add_e(diff, diff, bytemuck::cast(acc)),
                                )
                            }
                        } else if coe::is_same::<E, c64>() {
                            let diff = coe::coerce_static::<SimdGroupFor<E, S>, SimdGroupFor<c64, S>>(
                                diff,
                            );
                            let acc = coe::coerce_static::<
                                SimdGroupFor<E::Real, S>,
                                SimdGroupFor<f64, S>,
                            >(acc);

                            if coe::is_same::<S, pulp::Scalar>() {
                                let diff: c64 = bytemuck::cast(diff);
                                let acc: f64 = bytemuck::cast(acc);

                                coe::coerce_static::<
                                    SimdGroupFor<f64, pulp::Scalar>,
                                    SimdGroupFor<E::Real, S>,
                                >(diff.faer_abs2() + acc)
                            } else {
                                let diff: S::f64s = bytemuck::cast(diff);
                                simd.simd.f64s_mul_add_e(diff, diff, bytemuck::cast(acc));
                                coe::coerce_static::<SimdGroupFor<f64, S>, SimdGroupFor<E::Real, S>>(
                                    simd.simd.f64s_mul_add_e(diff, diff, bytemuck::cast(acc)),
                                )
                            }
                        } else {
                            simd.abs2_add_e(diff, acc)
                        }
                    }

                    let mut sum0 = simd_real.splat(E::Real::faer_zero());
                    let mut sum1 = simd_real.splat(E::Real::faer_zero());
                    let mut sum2 = simd_real.splat(E::Real::faer_zero());
                    let mut sum3 = simd_real.splat(E::Real::faer_zero());

                    sum0 = process(simd, sum0, mean, head);
                    let (body4, body1) = body.as_arrays::<4>();
                    for [x0, x1, x2, x3] in body4.into_ref_iter().map(RefGroup::unzip) {
                        sum0 = process(simd, sum0, mean, x0);
                        sum1 = process(simd, sum1, mean, x1);
                        sum2 = process(simd, sum2, mean, x2);
                        sum3 = process(simd, sum3, mean, x3);
                    }
                    for x0 in body1.into_ref_iter() {
                        sum0 = process(simd, sum0, mean, x0);
                    }
                    sum0 = process(simd, sum0, mean, tail);

                    sum0 = simd_real.add(sum0, sum1);
                    sum2 = simd_real.add(sum2, sum3);
                    sum0 = simd_real.add(sum0, sum2);

                    sum0 = simd_real.rotate_left(sum0, offset.rotate_left_amount());
                    let sum = simd_real.reduce_add(sum0);

                    out.write(i, sum.faer_scale_real(one_n1));
                }
            }
        }

        E::Simd::default().dispatch(Impl { out, mat, col_mean });
    }

    let mut out = out;
    if mat.ncols() == 1 {
        out.fill_zero();
        return;
    }

    let mat = if mat.col_stride() >= 0 {
        mat
    } else {
        mat.reverse_cols()
    };
    if mat.col_stride() == 1 {
        col_varm_row_major(out, mat, col_mean)
    } else {
        let n = mat.ncols();
        let one_n1 = E::Real::faer_from_f64((n - 1) as f64).faer_inv();

        out.fill_zero();
        for j in 0..n {
            zipped!(&mut out, col_mean, mat.col(j)).for_each(|unzipped!(mut out, mean, x)| {
                let diff = x.read().faer_sub(mean.read());
                out.write(out.read().faer_add(diff.faer_abs2()))
            });
        }
        zipped!(out).for_each(|unzipped!(mut x)| x.write(x.read().faer_scale_real(one_n1)));
    }
}

/// Computes the variance of the rows of `mat` given their mean, and stores the result in `out`.
#[track_caller]
pub fn row_varm<E: ComplexField>(
    out: RowMut<'_, E::Real>,
    mat: MatRef<'_, E>,
    row_mean: RowRef<'_, E>,
) {
    assert!(all(
        mat.nrows() > 0,
        out.ncols() == mat.ncols(),
        row_mean.ncols() == mat.ncols(),
    ));

    col_varm(out.transpose_mut(), mat.transpose(), row_mean.transpose());
}

#[cfg(test)]
mod tests {
    use super::*;
    use equator::assert;

    #[test]
    fn test_meanvar() {
        let c32 = c32::new;
        let A = mat![
            [c32(1.2, 2.3), c32(3.4, 1.2)],
            [c32(1.7, -1.0), c32(-3.8, 1.95)],
        ];

        let mut row_mean = Row::zeros(A.ncols());
        let mut row_var = Row::zeros(A.ncols());
        super::row_mean(row_mean.as_mut(), A.as_ref());
        super::row_varm(row_var.as_mut(), A.as_ref(), row_mean.as_ref());

        let mut col_mean = Col::zeros(A.nrows());
        let mut col_var = Col::zeros(A.nrows());
        super::col_mean(col_mean.as_mut(), A.as_ref());
        super::col_varm(col_var.as_mut(), A.as_ref(), col_mean.as_ref());

        assert!(row_mean == row![(A[(0, 0)] + A[(1, 0)]) / 2.0, (A[(0, 1)] + A[(1, 1)]) / 2.0,]);
        assert!(
            row_var
                == row![
                    (A[(0, 0)] - row_mean[0]).faer_abs2() + (A[(1, 0)] - row_mean[0]).faer_abs2(),
                    (A[(0, 1)] - row_mean[1]).faer_abs2() + (A[(1, 1)] - row_mean[1]).faer_abs2(),
                ]
        );

        assert!(col_mean == col![(A[(0, 0)] + A[(0, 1)]) / 2.0, (A[(1, 0)] + A[(1, 1)]) / 2.0,]);
        assert!(
            col_var
                == col![
                    (A[(0, 0)] - col_mean[0]).faer_abs2() + (A[(0, 1)] - col_mean[0]).faer_abs2(),
                    (A[(1, 0)] - col_mean[1]).faer_abs2() + (A[(1, 1)] - col_mean[1]).faer_abs2(),
                ]
        );
    }
}
