use super::LINEAR_IMPL_THRESHOLD;
use crate::{
    mat::MatRef,
    utils::{simd::*, slice::*},
};
use faer_entity::*;
use pulp::{Read, Simd};

#[inline(always)]
fn sum_with_simd_and_offset_prologue<E: ComplexField, S: pulp::Simd>(
    simd: S,
    data: SliceGroup<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> SimdGroupFor<E, S> {
    let simd = SimdFor::<E, S>::new(simd);

    let zero = simd.splat(E::faer_zero());

    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;
    let (head, body, tail) = simd.as_aligned_simd(data, offset);
    let (body4, body1) = body.as_arrays::<4>();
    let head = head.read_or(zero);
    acc0 = simd.add(acc0, head);

    for [x0, x1, x2, x3] in body4.into_ref_iter().map(RefGroup::unzip) {
        let x0 = x0.get();
        let x1 = x1.get();
        let x2 = x2.get();
        let x3 = x3.get();
        acc0 = simd.add(acc0, x0);
        acc1 = simd.add(acc1, x1);
        acc2 = simd.add(acc2, x2);
        acc3 = simd.add(acc3, x3);
    }

    for x0 in body1.into_ref_iter() {
        let x0 = x0.get();
        acc0 = simd.add(acc0, x0);
    }

    let tail = tail.read_or(zero);
    acc3 = simd.add(acc3, tail);

    acc0 = simd.add(acc0, acc1);
    acc2 = simd.add(acc2, acc3);
    simd.add(acc0, acc2)
}

#[inline(always)]
fn sum_with_simd_and_offset_pairwise_rows<E: ComplexField, S: Simd>(
    simd: S,
    data: SliceGroup<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
    last_offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> SimdGroupFor<E, S> {
    struct Impl<'a, E: ComplexField, S: Simd> {
        simd: S,
        data: SliceGroup<'a, E>,
        offset: pulp::Offset<SimdMaskFor<E, S>>,
        last_offset: pulp::Offset<SimdMaskFor<E, S>>,
    }

    impl<E: ComplexField, S: Simd> pulp::NullaryFnOnce for Impl<'_, E, S> {
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
                sum_with_simd_and_offset_prologue(simd, data, offset)
            } else if data.len() < LINEAR_IMPL_THRESHOLD {
                sum_with_simd_and_offset_prologue(simd, data, last_offset)
            } else {
                let split_point = ((data.len() + 1) / 2).next_power_of_two();
                let (head, tail) = data.split_at(split_point);
                let acc0 = sum_with_simd_and_offset_pairwise_rows(simd, head, offset, last_offset);
                let acc1 = sum_with_simd_and_offset_pairwise_rows(simd, tail, offset, last_offset);

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
fn sum_with_simd_and_offset_pairwise_cols<E: ComplexField, S: Simd>(
    simd: S,
    data: MatRef<'_, E>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
    last_offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> SimdGroupFor<E, S> {
    struct Impl<'a, E: ComplexField, S: Simd> {
        simd: S,
        data: MatRef<'a, E>,
        offset: pulp::Offset<SimdMaskFor<E, S>>,
        last_offset: pulp::Offset<SimdMaskFor<E, S>>,
    }

    impl<E: ComplexField, S: Simd> pulp::NullaryFnOnce for Impl<'_, E, S> {
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
                sum_with_simd_and_offset_pairwise_rows(
                    simd,
                    SliceGroup::<'_, E>::new(data.try_get_contiguous_col(0)),
                    offset,
                    last_offset,
                )
            } else {
                let split_point = (data.ncols() / 2).next_power_of_two();

                let (head, tail) = data.split_at_col(split_point);

                let acc0 = sum_with_simd_and_offset_pairwise_cols(simd, head, offset, last_offset);
                let acc1 = sum_with_simd_and_offset_pairwise_cols(simd, tail, offset, last_offset);

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

fn sum_contiguous<E: ComplexField>(data: MatRef<'_, E>) -> E {
    struct Impl<'a, E: ComplexField> {
        data: MatRef<'a, E>,
    }

    impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
        type Output = E;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { data } = self;

            let offset =
                SimdFor::<E, S>::new(simd).align_offset_ptr(data.as_ptr(), LINEAR_IMPL_THRESHOLD);

            let last_offset = SimdFor::<E, S>::new(simd)
                .align_offset_ptr(data.as_ptr(), data.nrows() % LINEAR_IMPL_THRESHOLD);

            let acc = sum_with_simd_and_offset_pairwise_cols(simd, data, offset, last_offset);

            let simd = SimdFor::<E, S>::new(simd);
            simd.reduce_add(simd.rotate_left(acc, offset.rotate_left_amount()))
        }
    }

    E::Simd::default().dispatch(Impl { data })
}

pub fn sum<E: ComplexField>(mut mat: MatRef<'_, E>) -> E {
    if mat.ncols() > 1 && mat.col_stride().unsigned_abs() < mat.row_stride().unsigned_abs() {
        mat = mat.transpose();
    }
    if mat.row_stride() < 0 {
        mat = mat.reverse_rows();
    }

    if mat.nrows() == 0 || mat.ncols() == 0 {
        E::faer_zero()
    } else {
        let m = mat.nrows();
        let n = mat.ncols();

        let mut acc = E::faer_zero();

        if mat.row_stride() == 1 {
            acc = sum_contiguous(mat);
        } else {
            for j in 0..n {
                for i in 0..m {
                    acc = acc.faer_add(mat.read(i, j));
                }
            }
        }

        acc
    }
}
