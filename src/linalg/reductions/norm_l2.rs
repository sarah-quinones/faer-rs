use super::LINEAR_IMPL_THRESHOLD;
use crate::{
    complex_native::*,
    mat::MatRef,
    utils::{simd::*, slice::*},
};
use faer_entity::*;
use pulp::Simd;

#[inline(always)]
fn norm_l2_with_simd_and_offset_prologue<E: ComplexField, S: pulp::Simd>(
    simd: S,
    data: SliceGroup<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> (
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
) {
    let simd_real = SimdFor::<E::Real, S>::new(simd);
    let simd = SimdFor::<E, S>::new(simd);
    let half_big = simd_real.splat(E::Real::faer_min_positive_sqrt_inv());
    let half_small = simd_real.splat(E::Real::faer_min_positive_sqrt());
    let zero = simd.splat(E::faer_zero());
    let zero_real = simd_real.splat(E::Real::faer_zero());

    let (head, body, tail) = simd.as_aligned_simd(data, offset);
    let (body2, body1) = body.as_arrays::<2>();

    let mut acc0 = simd.abs2(head.read_or(zero));
    let mut acc1 = zero_real;

    let mut acc_small0 = simd.abs2(simd.scale_real(half_small, head.read_or(zero)));
    let mut acc_small1 = zero_real;

    let mut acc_big0 = simd.abs2(simd.scale_real(half_big, head.read_or(zero)));
    let mut acc_big1 = zero_real;

    for [x0, x1] in body2.into_ref_iter().map(RefGroup::unzip) {
        let x0 = x0.get();
        let x1 = x1.get();
        acc0 = simd.abs2_add_e(x0, acc0);
        acc1 = simd.abs2_add_e(x1, acc1);

        acc_small0 = simd.abs2_add_e(simd.scale_real(half_small, x0), acc_small0);
        acc_small1 = simd.abs2_add_e(simd.scale_real(half_small, x1), acc_small1);

        acc_big0 = simd.abs2_add_e(simd.scale_real(half_big, x0), acc_big0);
        acc_big1 = simd.abs2_add_e(simd.scale_real(half_big, x1), acc_big1);
    }

    if body1.len() == 1 {
        let x0 = body1.get(0).get();
        acc0 = simd.abs2_add_e(x0, acc0);
        acc_small0 = simd.abs2_add_e(simd.scale_real(half_small, x0), acc_small0);
        acc_big0 = simd.abs2_add_e(simd.scale_real(half_big, x0), acc_big0);

        acc1 = simd.abs2_add_e(tail.read_or(zero), acc1);
        acc_small1 = simd.abs2_add_e(simd.scale_real(half_small, tail.read_or(zero)), acc_small1);
        acc_big1 = simd.abs2_add_e(simd.scale_real(half_big, tail.read_or(zero)), acc_big1);
    } else {
        acc0 = simd.abs2_add_e(tail.read_or(zero), acc0);
        acc_small0 = simd.abs2_add_e(simd.scale_real(half_small, tail.read_or(zero)), acc_small0);
        acc_big0 = simd.abs2_add_e(simd.scale_real(half_big, tail.read_or(zero)), acc_big0);
    }

    acc0 = simd_real.add(acc0, acc1);
    acc_small0 = simd_real.add(acc_small0, acc_small1);
    acc_big0 = simd_real.add(acc_big0, acc_big1);

    (acc_small0, acc0, acc_big0)
}

#[inline(always)]
fn norm_l2_with_simd_and_offset_pairwise_rows<E: ComplexField, S: Simd>(
    simd: S,
    data: SliceGroup<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
    last_offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> (
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
) {
    struct Impl<'a, E: ComplexField, S: Simd> {
        simd: S,
        data: SliceGroup<'a, E>,
        offset: pulp::Offset<SimdMaskFor<E, S>>,
        last_offset: pulp::Offset<SimdMaskFor<E, S>>,
    }

    impl<E: ComplexField, S: Simd> pulp::NullaryFnOnce for Impl<'_, E, S> {
        type Output = (
            SimdGroupFor<E::Real, S>,
            SimdGroupFor<E::Real, S>,
            SimdGroupFor<E::Real, S>,
        );

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                data,
                offset,
                last_offset,
            } = self;

            if data.len() == LINEAR_IMPL_THRESHOLD {
                norm_l2_with_simd_and_offset_prologue(simd, data, offset)
            } else if data.len() < LINEAR_IMPL_THRESHOLD {
                norm_l2_with_simd_and_offset_prologue(simd, data, last_offset)
            } else {
                let split_point = ((data.len() + 1) / 2).next_power_of_two();
                let (head, tail) = data.split_at(split_point);
                let (acc_small0, acc0, acc_big0) =
                    norm_l2_with_simd_and_offset_pairwise_rows(simd, head, offset, last_offset);
                let (acc_small1, acc1, acc_big1) =
                    norm_l2_with_simd_and_offset_pairwise_rows(simd, tail, offset, last_offset);

                let simd = SimdFor::<E::Real, S>::new(simd);
                (
                    simd.add(acc_small0, acc_small1),
                    simd.add(acc0, acc1),
                    simd.add(acc_big0, acc_big1),
                )
            }
        }
    }

    simd.vectorize(Impl {
        simd,
        data,
        offset,
        last_offset,
    })
}

#[inline(always)]
fn norm_l2_with_simd_and_offset_pairwise_cols<E: ComplexField, S: Simd>(
    simd: S,
    data: MatRef<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
    last_offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> (
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
    SimdGroupFor<E::Real, S>,
) {
    struct Impl<'a, E: ComplexField, S: Simd> {
        simd: S,
        data: MatRef<'a, E>,
        offset: pulp::Offset<SimdMaskFor<E, S>>,
        last_offset: pulp::Offset<SimdMaskFor<E, S>>,
    }

    impl<E: ComplexField, S: Simd> pulp::NullaryFnOnce for Impl<'_, E, S> {
        type Output = (
            SimdGroupFor<E::Real, S>,
            SimdGroupFor<E::Real, S>,
            SimdGroupFor<E::Real, S>,
        );

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                data,
                offset,
                last_offset,
            } = self;
            if data.ncols() == 1 {
                norm_l2_with_simd_and_offset_pairwise_rows(
                    simd,
                    SliceGroup::<'_, E>::new(data.try_get_contiguous_col(0)),
                    offset,
                    last_offset,
                )
            } else {
                let split_point = (data.ncols() / 2).next_power_of_two();

                let (head, tail) = data.split_at_col(split_point);

                let (acc_small0, acc0, acc_big0) =
                    norm_l2_with_simd_and_offset_pairwise_cols(simd, head, offset, last_offset);
                let (acc_small1, acc1, acc_big1) =
                    norm_l2_with_simd_and_offset_pairwise_cols(simd, tail, offset, last_offset);

                let simd = SimdFor::<E::Real, S>::new(simd);
                (
                    simd.add(acc_small0, acc_small1),
                    simd.add(acc0, acc1),
                    simd.add(acc_big0, acc_big1),
                )
            }
        }
    }

    simd.vectorize(Impl {
        simd,
        data,
        offset,
        last_offset,
    })
}

#[inline(always)]

fn norm_l2_contiguous<E: ComplexField>(data: MatRef<'_, E>) -> (E::Real, E::Real, E::Real) {
    struct Impl<'a, E: ComplexField> {
        data: MatRef<'a, E>,
    }

    impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
        type Output = (E::Real, E::Real, E::Real);

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { data } = self;

            let offset =
                SimdFor::<E, S>::new(simd).align_offset_ptr(data.as_ptr(), LINEAR_IMPL_THRESHOLD);

            let last_offset = SimdFor::<E, S>::new(simd)
                .align_offset_ptr(data.as_ptr(), data.nrows() % LINEAR_IMPL_THRESHOLD);

            let (acc_small, acc, acc_big) =
                norm_l2_with_simd_and_offset_pairwise_cols(simd, data, offset, last_offset);

            let simd = SimdFor::<E::Real, S>::new(simd);
            (
                simd.reduce_add(acc_small),
                simd.reduce_add(acc),
                simd.reduce_add(acc_big),
            )
        }
    }

    E::Simd::default().dispatch(Impl { data })
}

pub fn norm_l2<E: ComplexField>(mut mat: MatRef<'_, E>) -> E::Real {
    if mat.ncols() > 1 && mat.col_stride().unsigned_abs() < mat.row_stride().unsigned_abs() {
        mat = mat.transpose();
    }
    if mat.row_stride() < 0 {
        mat = mat.reverse_rows();
    }

    if mat.nrows() == 0 || mat.ncols() == 0 {
        E::Real::faer_zero()
    } else {
        let m = mat.nrows();
        let n = mat.ncols();

        let half_small = E::Real::faer_min_positive_sqrt();
        let half_big = E::Real::faer_min_positive_sqrt_inv();

        let mut acc_small = E::Real::faer_zero();
        let mut acc = E::Real::faer_zero();
        let mut acc_big = E::Real::faer_zero();

        if mat.row_stride() == 1 {
            if coe::is_same::<E, c32>() {
                let mat: MatRef<'_, c32> = coe::coerce(mat);
                let mat = unsafe {
                    crate::mat::from_raw_parts(
                        mat.as_ptr() as *const f32,
                        2 * mat.nrows(),
                        mat.ncols(),
                        1,
                        mat.col_stride().wrapping_mul(2),
                    )
                };
                let (acc_small_, acc_, acc_big_) = norm_l2_contiguous::<f32>(mat);
                acc_small = coe::coerce_static(acc_small_);
                acc = coe::coerce_static(acc_);
                acc_big = coe::coerce_static(acc_big_);
            } else if coe::is_same::<E, c64>() {
                let mat: MatRef<'_, c64> = coe::coerce(mat);
                let mat = unsafe {
                    crate::mat::from_raw_parts(
                        mat.as_ptr() as *const f64,
                        2 * mat.nrows(),
                        mat.ncols(),
                        1,
                        mat.col_stride().wrapping_mul(2),
                    )
                };
                let (acc_small_, acc_, acc_big_) = norm_l2_contiguous::<f64>(mat);
                acc_small = coe::coerce_static(acc_small_);
                acc = coe::coerce_static(acc_);
                acc_big = coe::coerce_static(acc_big_);
            } else {
                (acc_small, acc, acc_big) = norm_l2_contiguous(mat);
            }
        } else {
            for j in 0..n {
                for i in 0..m {
                    let val = mat.read(i, j);
                    let val_small = val.faer_scale_power_of_two(half_small);
                    let val_big = val.faer_scale_power_of_two(half_big);

                    acc_small = acc_small.faer_add(val_small.faer_abs2());
                    acc = acc.faer_add(val.faer_abs2());
                    acc_big = acc_big.faer_add(val_big.faer_abs2());
                }
            }
        }

        if acc_small >= E::Real::faer_one() {
            acc_small.faer_sqrt().faer_mul(half_big)
        } else if acc_big <= E::Real::faer_one() {
            acc_big.faer_sqrt().faer_mul(half_small)
        } else {
            acc.faer_sqrt()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert, prelude::*, unzipped, zipped};

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
                    assert!(mat.norm_l2() == target);
                } else {
                    assert!(relative_err(mat.norm_l2(), target) < 1e-14);
                }
            }
        }

        let mat = Col::from_fn(10000000, |_| 0.3);
        let target = (0.3 * 0.3 * 10000000.0f64).sqrt();
        assert!(relative_err(mat.norm_l2(), target) < 1e-14);
    }
}
