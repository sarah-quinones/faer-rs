use super::LINEAR_IMPL_THRESHOLD;
use crate::{
    complex_native::*,
    mat::MatRef,
    utils::{simd::*, slice::*},
};
use coe::Coerce;
use faer_entity::*;
use pulp::Simd;

#[inline(always)]
fn norm_l1_with_simd_and_offset_prologue<E: RealField, S: pulp::Simd>(
    simd: S,
    data: SliceGroup<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> SimdGroupFor<E, S> {
    let simd = SimdFor::<E, S>::new(simd);
    let zero = simd.splat(E::faer_zero());

    let (head, body, tail) = simd.as_aligned_simd(data, offset);
    let (body2, body1) = body.as_arrays::<4>();

    let mut acc0 = simd.abs(head.read_or(zero));
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;

    for [x0, x1, x2, x3] in body2.into_ref_iter().map(RefGroup::unzip) {
        let x0 = simd.abs(x0.get());
        let x1 = simd.abs(x1.get());
        let x2 = simd.abs(x2.get());
        let x3 = simd.abs(x3.get());
        acc1 = simd.add(x0, acc1);
        acc2 = simd.add(x1, acc2);
        acc3 = simd.add(x2, acc3);
        acc0 = simd.add(x3, acc0);
    }

    match body1.len() {
        0 => {
            acc1 = simd.add(simd.abs(tail.read_or(zero)), acc1);
        }
        1 => {
            acc1 = simd.add(simd.abs(body1.get(0).get()), acc1);
            acc2 = simd.add(simd.abs(tail.read_or(zero)), acc2);
        }
        2 => {
            acc1 = simd.add(simd.abs(body1.get(0).get()), acc1);
            acc2 = simd.add(simd.abs(body1.get(1).get()), acc2);
            acc3 = simd.add(simd.abs(tail.read_or(zero)), acc3);
        }
        3 => {
            acc1 = simd.add(simd.abs(body1.get(0).get()), acc1);
            acc2 = simd.add(simd.abs(body1.get(1).get()), acc2);
            acc3 = simd.add(simd.abs(body1.get(2).get()), acc3);
            acc0 = simd.add(simd.abs(tail.read_or(zero)), acc0);
        }
        _ => unreachable!(),
    }

    acc0 = simd.add(acc0, acc2);
    acc1 = simd.add(acc1, acc3);
    acc0 = simd.add(acc0, acc1);

    acc0
}

#[inline(always)]
fn norm_l1_with_simd_and_offset_pairwise_rows<E: RealField, S: Simd>(
    simd: S,
    data: SliceGroup<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
    last_offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> SimdGroupFor<E, S> {
    struct Impl<'a, E: RealField, S: Simd> {
        simd: S,
        data: SliceGroup<'a, E>,
        offset: pulp::Offset<SimdMaskFor<E, S>>,
        last_offset: pulp::Offset<SimdMaskFor<E, S>>,
    }

    impl<E: RealField, S: Simd> pulp::NullaryFnOnce for Impl<'_, E, S> {
        type Output = SimdGroupFor<E, S>;

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                data,
                offset,
                last_offset,
            } = self;

            if data.len() == LINEAR_IMPL_THRESHOLD {
                norm_l1_with_simd_and_offset_prologue(simd, data, offset)
            } else if data.len() < LINEAR_IMPL_THRESHOLD {
                norm_l1_with_simd_and_offset_prologue(simd, data, last_offset)
            } else {
                let split_point = ((data.len() + 1) / 2).next_power_of_two();
                let (head, tail) = data.split_at(split_point);
                let acc0 =
                    norm_l1_with_simd_and_offset_pairwise_rows(simd, head, offset, last_offset);
                let acc1 =
                    norm_l1_with_simd_and_offset_pairwise_rows(simd, tail, offset, last_offset);

                let simd = SimdFor::<E, S>::new(simd);

                simd.add(acc0, acc1)
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
fn norm_l1_with_simd_and_offset_pairwise_cols<E: RealField, S: Simd>(
    simd: S,
    data: MatRef<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
    last_offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> SimdGroupFor<E, S> {
    struct Impl<'a, E: RealField, S: Simd> {
        simd: S,
        data: MatRef<'a, E>,
        offset: pulp::Offset<SimdMaskFor<E, S>>,
        last_offset: pulp::Offset<SimdMaskFor<E, S>>,
    }

    impl<E: RealField, S: Simd> pulp::NullaryFnOnce for Impl<'_, E, S> {
        type Output = SimdGroupFor<E, S>;

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                data,
                offset,
                last_offset,
            } = self;
            if data.ncols() == 1 {
                norm_l1_with_simd_and_offset_pairwise_rows(
                    simd,
                    SliceGroup::<'_, E>::new(data.try_get_contiguous_col(0)),
                    offset,
                    last_offset,
                )
            } else {
                let split_point = (data.ncols() / 2).next_power_of_two();

                let (head, tail) = data.split_at_col(split_point);

                let acc0 =
                    norm_l1_with_simd_and_offset_pairwise_cols(simd, head, offset, last_offset);
                let acc1 =
                    norm_l1_with_simd_and_offset_pairwise_cols(simd, tail, offset, last_offset);

                let simd = SimdFor::<E, S>::new(simd);
                simd.add(acc0, acc1)
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

fn norm_l1_contiguous<E: RealField>(data: MatRef<'_, E>) -> E {
    struct Impl<'a, E: RealField> {
        data: MatRef<'a, E>,
    }

    impl<E: RealField> pulp::WithSimd for Impl<'_, E> {
        type Output = E;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { data } = self;

            let offset =
                SimdFor::<E, S>::new(simd).align_offset_ptr(data.as_ptr(), LINEAR_IMPL_THRESHOLD);

            let last_offset = SimdFor::<E, S>::new(simd)
                .align_offset_ptr(data.as_ptr(), data.nrows() % LINEAR_IMPL_THRESHOLD);

            let acc = norm_l1_with_simd_and_offset_pairwise_cols(simd, data, offset, last_offset);
            let simd = SimdFor::<E, S>::new(simd);

            simd.reduce_add(acc)
        }
    }

    E::Simd::default().dispatch(Impl { data })
}

pub fn norm_l1<E: ComplexField>(mut mat: MatRef<'_, E>) -> E::Real {
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

        if mat.row_stride() == 1 {
            if const { E::IS_C32 } {
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
                return coe::coerce_static(norm_l1_contiguous::<f32>(mat));
            } else if const { E::IS_C64 } {
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
                return coe::coerce_static(norm_l1_contiguous::<f64>(mat));
            } else if const { E::IS_NUM_COMPLEX } {
                let mat: MatRef<num_complex::Complex<E::Real>> = mat.coerce();
                let num_complex::Complex { re, im } = mat.real_imag();
                return norm_l1_contiguous(re).faer_add(norm_l1_contiguous(im));
            } else if const { E::IS_REAL } {
                return norm_l1_contiguous::<E::Real>(mat.coerce());
            }
        }

        let mut acc = E::Real::faer_zero();
        for j in 0..n {
            for i in 0..m {
                let val = mat.read(i, j);

                acc = acc.faer_add(
                    val.faer_real()
                        .faer_abs()
                        .faer_add(val.faer_imag().faer_abs()),
                );
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert, prelude::*, unzipped, zipped_rw};

    #[test]
    fn test_norm_l1() {
        let relative_err = |a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());

        for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
            for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
                let mat = Mat::from_fn(m, n, |i, j| factor * ((i + j) as f64));
                let mut target = 0.0;
                zipped_rw!(mat.as_ref()).for_each(|unzipped!(x)| {
                    target += (*x).abs();
                });

                if factor == 0.0 {
                    assert!(mat.norm_l1() == target);
                } else {
                    assert!(relative_err(mat.norm_l1(), target) < 1e-14);
                }
            }
        }

        let mat = Col::from_fn(10000000, |_| 0.3);
        let target = 0.3 * 10000000.0f64;
        assert!(relative_err(mat.norm_l1(), target) < 1e-14);
    }
}
