#[cfg(feature = "std")]
use assert2::{assert, debug_assert};
use bytemuck::cast;
use coe::Coerce;
use core::slice;
use dyn_stack::{PodStack, StackReq};
use faer_core::{
    c32, c64,
    mul::matmul,
    permutation::{swap_cols, swap_rows, PermutationMut},
    simd, ComplexField, Entity, MatMut, MatRef, Parallelism, RealField, SimdCtx,
};
use faer_entity::*;
use paste::paste;
use pulp::{cast_lossy, Simd};
use reborrow::*;

#[inline(always)]
fn best_f64<S: Simd>(
    simd: S,
    best_value: S::f64s,
    best_indices: S::u64s,
    data: S::f64s,
    indices: S::u64s,
) -> (S::f64s, S::u64s) {
    let value = simd.f64s_abs(data);
    let is_better = simd.f64s_greater_than(value, best_value);
    (
        simd.m64s_select_f64s(is_better, value, best_value),
        simd.m64s_select_u64s(is_better, indices, best_indices),
    )
}

#[inline(always)]
fn best_f32<S: Simd>(
    simd: S,
    best_value: S::f32s,
    best_indices: S::u32s,
    data: S::f32s,
    indices: S::u32s,
) -> (S::f32s, S::u32s) {
    let value = simd.f32s_abs(data);
    let is_better = simd.f32s_greater_than(value, best_value);
    (
        simd.m32s_select_f32s(is_better, value, best_value),
        simd.m32s_select_u32s(is_better, indices, best_indices),
    )
}

#[inline(always)]
fn best_c64_scalar(best_value: f64, best_indices: u64, data: c64, indices: u64) -> (f64, u64) {
    let simd = pulp::Scalar::new();
    let value = simd.c64s_abs2(pulp::cast(data)).re;
    let is_better = simd.f64s_greater_than(value, best_value);
    (
        simd.m64s_select_f64s(is_better, value, best_value),
        simd.m64s_select_u64s(is_better, indices, best_indices),
    )
}

#[inline(always)]
fn best_c64<S: Simd>(
    simd: S,
    best_value: S::f64s,
    best_indices: S::u64s,
    data: S::c64s,
    indices: S::u64s,
) -> (S::f64s, S::u64s) {
    if coe::is_same::<pulp::Scalar, S>() {
        coe::coerce_static(best_c64_scalar(
            coe::coerce_static(best_value),
            coe::coerce_static(best_indices),
            bytemuck::cast(data),
            coe::coerce_static(indices),
        ))
    } else {
        let value = cast(simd.c64s_abs2(data));
        let is_better = simd.f64s_greater_than(value, best_value);
        (
            simd.m64s_select_f64s(is_better, value, best_value),
            simd.m64s_select_u64s(is_better, indices, best_indices),
        )
    }
}

#[inline(always)]
fn best_c32_scalar(best_value: f32, best_indices: u32, data: c32, indices: u32) -> (f32, u32) {
    let simd = pulp::Scalar::new();
    let value = simd.c32s_abs2(pulp::cast(data)).re;
    let is_better = simd.f32s_greater_than(value, best_value);
    (
        simd.m32s_select_f32s(is_better, value, best_value),
        simd.m32s_select_u32s(is_better, indices, best_indices),
    )
}

#[inline(always)]
fn best_c32<S: Simd>(
    simd: S,
    best_value: S::f32s,
    best_indices: S::u32s,
    data: S::c32s,
    indices: S::u32s,
) -> (S::f32s, S::u32s) {
    if coe::is_same::<pulp::Scalar, S>() {
        coe::coerce_static(best_c32_scalar(
            coe::coerce_static(best_value),
            coe::coerce_static(best_indices),
            bytemuck::cast(data),
            coe::coerce_static(indices),
        ))
    } else {
        let value = cast(simd.c32s_abs2(data));
        let is_better = simd.f32s_greater_than(value, best_value);
        (
            simd.m32s_select_f32s(is_better, value, best_value),
            simd.m32s_select_u32s(is_better, indices, best_indices),
        )
    }
}

#[inline(always)]
fn best2d_f64<S: Simd>(
    simd: S,
    best_value: S::f64s,
    best_row_indices: S::u64s,
    best_col_indices: S::u64s,
    value: S::f64s,
    row_indices: S::u64s,
    col_indices: S::u64s,
) -> (S::f64s, S::u64s, S::u64s) {
    let is_better = simd.f64s_greater_than(value, best_value);
    (
        simd.m64s_select_f64s(is_better, value, best_value),
        simd.m64s_select_u64s(is_better, row_indices, best_row_indices),
        simd.m64s_select_u64s(is_better, col_indices, best_col_indices),
    )
}

#[inline(always)]
fn best2d_f32<S: Simd>(
    simd: S,
    best_value: S::f32s,
    best_row_indices: S::u32s,
    best_col_indices: S::u32s,
    value: S::f32s,
    row_indices: S::u32s,
    col_indices: S::u32s,
) -> (S::f32s, S::u32s, S::u32s) {
    let is_better = simd.f32s_greater_than(value, best_value);
    (
        simd.m32s_select_f32s(is_better, value, best_value),
        simd.m32s_select_u32s(is_better, row_indices, best_row_indices),
        simd.m32s_select_u32s(is_better, col_indices, best_col_indices),
    )
}

#[inline(always)]
fn best2d<E: RealField, S: Simd>(
    simd: S,
    best_value: GroupFor<E, SimdUnitFor<E, S>>,
    best_row_indices: SimdIndexFor<E, S>,
    best_col_indices: SimdIndexFor<E, S>,
    value: GroupFor<E, SimdUnitFor<E, S>>,
    row_indices: SimdIndexFor<E, S>,
    col_indices: SimdIndexFor<E, S>,
) -> (
    GroupFor<E, SimdUnitFor<E, S>>,
    SimdIndexFor<E, S>,
    SimdIndexFor<E, S>,
) {
    let is_better =
        E::faer_simd_greater_than(simd, E::faer_copy(&value), E::faer_copy(&best_value));
    (
        E::faer_simd_select(simd, is_better, value, best_value),
        E::faer_simd_index_select(simd, is_better, row_indices, best_row_indices),
        E::faer_simd_index_select(simd, is_better, col_indices, best_col_indices),
    )
}

#[inline(always)]
fn best_value<E: ComplexField, S: Simd>(
    simd: S,
    best_value: SimdGroupFor<E::Real, S>,
    best_indices: <E::Real as Entity>::SimdIndex<S>,
    data: GroupFor<E, SimdUnitFor<E, S>>,
    indices: <E::Real as Entity>::SimdIndex<S>,
) -> (SimdGroupFor<E::Real, S>, <E::Real as Entity>::SimdIndex<S>) {
    let value = E::faer_simd_score(simd, data);
    let is_better = E::Real::faer_simd_greater_than(
        simd,
        E::Real::faer_copy(&value),
        E::Real::faer_copy(&best_value),
    );
    (
        E::Real::faer_simd_select(simd, is_better, value, best_value),
        E::Real::faer_simd_index_select(simd, is_better, indices, best_indices),
    )
}

#[inline(always)]
fn best_score<E: RealField, S: Simd>(
    simd: S,
    best_value: GroupFor<E, SimdUnitFor<E, S>>,
    best_indices: SimdIndexFor<E, S>,
    value: GroupFor<E, SimdUnitFor<E, S>>,
    indices: SimdIndexFor<E, S>,
) -> (GroupFor<E, SimdUnitFor<E, S>>, SimdIndexFor<E, S>) {
    let is_better = E::Real::faer_simd_greater_than(
        simd,
        E::Real::faer_copy(&value),
        E::Real::faer_copy(&best_value),
    );
    (
        E::Real::faer_simd_select(simd, is_better, value, best_value),
        E::Real::faer_simd_index_select(simd, is_better, indices, best_indices),
    )
}

#[inline(always)]
fn best_in_col<E: ComplexField, S: Simd>(
    simd: S,
    data: GroupFor<E, &[UnitFor<E>]>,
) -> (SimdGroupFor<E::Real, S>, SimdIndexFor<E::Real, S>) {
    let (head, tail) = simd::slice_as_simd::<E, S>(data);

    let iota = E::Real::faer_simd_index_seq(simd);
    let lane_count = core::mem::size_of::<SimdUnitFor<E, S>>() / core::mem::size_of::<UnitFor<E>>();
    let increment1 = E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(lane_count));
    let increment4 =
        E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(4 * lane_count));

    let mut best_value0 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());
    let mut best_value1 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());
    let mut best_value2 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());
    let mut best_value3 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());

    let mut best_indices0 = E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(0));
    let mut best_indices1 = E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(0));
    let mut best_indices2 = E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(0));
    let mut best_indices3 = E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(0));

    let mut indices0 = iota;
    let mut indices1 = E::Real::faer_simd_index_add(simd, indices0, increment1);
    let mut indices2 = E::Real::faer_simd_index_add(simd, indices1, increment1);
    let mut indices3 = E::Real::faer_simd_index_add(simd, indices2, increment1);

    let (head4, tail4) = E::faer_as_arrays::<4, _>(head);
    for data in E::faer_into_iter(head4) {
        let [data0, data1, data2, data3] = E::faer_unzip4(E::faer_deref(data));
        (best_value0, best_indices0) =
            best_value::<E, S>(simd, best_value0, best_indices0, data0, indices0);
        (best_value1, best_indices1) =
            best_value::<E, S>(simd, best_value1, best_indices1, data1, indices1);
        (best_value2, best_indices2) =
            best_value::<E, S>(simd, best_value2, best_indices2, data2, indices2);
        (best_value3, best_indices3) =
            best_value::<E, S>(simd, best_value3, best_indices3, data3, indices3);

        indices0 = E::Real::faer_simd_index_add(simd, indices0, increment4);
        indices1 = E::Real::faer_simd_index_add(simd, indices1, increment4);
        indices2 = E::Real::faer_simd_index_add(simd, indices2, increment4);
        indices3 = E::Real::faer_simd_index_add(simd, indices3, increment4);
    }

    (best_value0, best_indices0) =
        best_score::<E::Real, S>(simd, best_value0, best_indices0, best_value1, best_indices1);
    (best_value2, best_indices2) =
        best_score::<E::Real, S>(simd, best_value2, best_indices2, best_value3, best_indices3);

    (best_value0, best_indices0) =
        best_score::<E::Real, S>(simd, best_value0, best_indices0, best_value2, best_indices2);

    for data in E::faer_into_iter(tail4) {
        let data0 = E::faer_deref(data);
        (best_value0, best_indices0) =
            best_value::<E, S>(simd, best_value0, best_indices0, data0, indices0);
        indices0 = E::Real::faer_simd_index_add(simd, indices0, increment1);
    }

    best_value::<E, S>(
        simd,
        best_value0,
        best_indices0,
        E::faer_partial_load(simd, tail),
        indices0,
    )
}

#[inline(always)]
fn update_and_best_in_col<E: ComplexField, S: Simd>(
    simd: S,
    data: GroupFor<E, &mut [UnitFor<E>]>,
    lhs: GroupFor<E, &[UnitFor<E>]>,
    rhs: E,
) -> (SimdGroupFor<E::Real, S>, <E::Real as Entity>::SimdIndex<S>) {
    let mut len = 0;
    E::faer_map(
        E::faer_as_ref(&data),
        #[inline(always)]
        |slice| len = slice.len(),
    );

    let iota = E::Real::faer_simd_index_seq(simd);
    let lane_count = core::mem::size_of::<SimdUnitFor<E, S>>() / core::mem::size_of::<UnitFor<E>>();

    let rhs = E::faer_simd_splat(simd, rhs);

    if len <= lane_count {
        let dst0 = E::faer_partial_load(simd, E::faer_rb(E::faer_as_ref(&data)));
        let lhs0 = E::faer_partial_load(simd, lhs);
        let dst0 = E::faer_simd_mul_adde(simd, lhs0, E::faer_copy(&rhs), dst0);
        E::faer_partial_store(simd, data, E::faer_copy(&dst0));

        return (E::faer_simd_score(simd, dst0), iota);
    }

    let (dst_head, dst_tail) = simd::slice_as_mut_simd::<E, S>(data);
    let (lhs_head, lhs_tail) = simd::slice_as_simd::<E, S>(lhs);

    let increment1 = E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(lane_count));
    let increment2 =
        E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(2 * lane_count));

    let mut best_value0 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());
    let mut best_value1 = E::Real::faer_simd_splat(simd, E::Real::faer_zero());

    let mut best_indices0 = E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(0));
    let mut best_indices1 = E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(0));

    let mut indices0 = iota;
    let mut indices1 = E::Real::faer_simd_index_add(simd, indices0, increment1);

    let (dst_head2, dst_tail2) = E::faer_as_arrays_mut::<2, _>(dst_head);
    let (lhs_head2, lhs_tail2) = E::faer_as_arrays::<2, _>(lhs_head);

    for (dst, lhs) in E::faer_into_iter(dst_head2).zip(E::faer_into_iter(lhs_head2)) {
        let [dst0, dst1] = E::faer_unzip2(E::faer_deref(E::faer_rb(E::faer_as_ref(&dst))));
        let [lhs0, lhs1] = E::faer_unzip2(E::faer_deref(lhs));

        let dst0 = E::faer_simd_mul_adde(simd, lhs0, E::faer_copy(&rhs), dst0);
        let dst1 = E::faer_simd_mul_adde(simd, lhs1, E::faer_copy(&rhs), dst1);

        E::faer_map(
            E::faer_zip(dst, E::faer_zip(E::faer_copy(&dst0), E::faer_copy(&dst1))),
            #[inline(always)]
            |(dst, (dst0, dst1))| {
                dst[0] = dst0;
                dst[1] = dst1;
            },
        );

        (best_value0, best_indices0) =
            best_value::<E, S>(simd, best_value0, best_indices0, dst0, indices0);
        (best_value1, best_indices1) =
            best_value::<E, S>(simd, best_value1, best_indices1, dst1, indices1);

        indices0 = E::Real::faer_simd_index_add(simd, indices0, increment2);
        indices1 = E::Real::faer_simd_index_add(simd, indices1, increment2);
    }

    (best_value0, best_indices0) =
        best_score::<E::Real, S>(simd, best_value0, best_indices0, best_value1, best_indices1);

    for (dst, lhs) in E::faer_into_iter(dst_tail2).zip(E::faer_into_iter(lhs_tail2)) {
        let dst0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&dst)));
        let lhs0 = E::faer_deref(lhs);

        let dst0 = E::faer_simd_mul_adde(simd, lhs0, E::faer_copy(&rhs), dst0);

        E::faer_map(
            E::faer_zip(dst, E::faer_copy(&dst0)),
            #[inline(always)]
            |(dst, dst0)| *dst = dst0,
        );

        (best_value0, best_indices0) =
            best_value::<E, S>(simd, best_value0, best_indices0, dst0, indices0);

        indices0 = E::Real::faer_simd_index_add(simd, indices0, increment1);
    }

    let dst0 = E::faer_partial_load(simd, E::faer_rb(E::faer_as_ref(&dst_tail)));
    let lhs0 = E::faer_partial_load(simd, lhs_tail);
    let dst0 = E::faer_simd_mul_adde(simd, lhs0, E::faer_copy(&rhs), dst0);
    E::faer_partial_store(simd, dst_tail, E::faer_copy(&dst0));

    best_value::<E, S>(simd, best_value0, best_indices0, dst0, indices0)
}

macro_rules! best_in_col_simd {
    ($scalar: ident, $real_scalar: ident, $uint: ident) => {
        paste! {
            #[inline(always)]
            fn [<best_in_col_ $scalar _generic>]<S: pulp::Simd>(
                simd: S,
                iota: S::[<$uint s>],
                data: &[$scalar],
            ) -> (S::[<$real_scalar s>], S::[<$uint s>]) {
                let (head, tail) = S::[<$scalar s_as_simd>](bytemuck::cast_slice(data));

                let lane_count = core::mem::size_of::<S::[<$scalar s>]>() / core::mem::size_of::<$scalar>();
                let increment1 = simd.[<$uint s_splat>](lane_count as $uint);
                let increment3 = simd.[<$uint s_splat>](3 * lane_count as $uint);

                let mut best_value0 = simd.[<$real_scalar s_splat>](0.0);
                let mut best_value1 = simd.[<$real_scalar s_splat>](0.0);
                let mut best_value2 = simd.[<$real_scalar s_splat>](0.0);
                let mut best_indices0 = simd.[<$uint s_splat>](0);
                let mut best_indices1 = simd.[<$uint s_splat>](0);
                let mut best_indices2 = simd.[<$uint s_splat>](0);
                let mut indices0 = iota;
                let mut indices1 = simd.[<$uint s_add>](indices0, increment1);
                let mut indices2 = simd.[<$uint s_add>](indices1, increment1);

                let (head3, tail3) = pulp::as_arrays::<3, _>(head);

                for &data in head3.clone() {
                    (best_value0, best_indices0) =
                        [<best_ $scalar>](simd, best_value0, best_indices0, data[0], indices0);
                    (best_value1, best_indices1) =
                        [<best_ $scalar>](simd, best_value1, best_indices1, data[1], indices1);
                    (best_value2, best_indices2) =
                        [<best_ $scalar>](simd, best_value2, best_indices2, data[2], indices2);
                    indices0 = simd.[<$uint s_add>](indices0, increment3);
                    indices1 = simd.[<$uint s_add>](indices1, increment3);
                    indices2 = simd.[<$uint s_add>](indices2, increment3);
                }

                (best_value0, best_indices0) =
                    [<best_ $real_scalar>](simd, best_value0, best_indices0, best_value1, best_indices1);
                (best_value0, best_indices0) =
                    [<best_ $real_scalar>](simd, best_value0, best_indices0, best_value2, best_indices2);

                for &data in tail3 {
                    (best_value0, best_indices0) = [<best_ $scalar>](simd, best_value0, best_indices0, data, indices0);
                    indices0 = simd.[<$uint s_add>](indices0, increment1);
                }

                (best_value0, best_indices0) = [<best_ $scalar>](
                    simd,
                    best_value0,
                    best_indices0,
                    simd.[<$scalar s_partial_load>](tail),
                    indices0
                );

                (
                    best_value0,
                    best_indices0,
                )
            }

            #[inline(always)]
            fn [<update_and_best_in_col_ $scalar _generic>]<S: pulp::Simd>(
                simd: S,
                iota: S::[<$uint s>],
                dst: &mut [$scalar],
                lhs: &[$scalar],
                rhs: $scalar,
            ) -> (S::[<$real_scalar s>], S::[<$uint s>]) {
                let lane_count = core::mem::size_of::<S::[<$scalar s>]>() / core::mem::size_of::<$scalar>();

                let (dst_head, dst_tail) = S::[<$scalar s_as_mut_simd>](bytemuck::cast_slice_mut(dst));
                let (lhs_head, lhs_tail) = S::[<$scalar s_as_simd>](bytemuck::cast_slice(lhs));

                let increment1 = simd.[<$uint s_splat>](lane_count as $uint);
                let increment2 = simd.[<$uint s_splat>](2 * lane_count as $uint);

                let mut best_value0 = simd.[<$real_scalar s_splat>](0.0);
                let mut best_value1 = simd.[<$real_scalar s_splat>](0.0);
                let mut best_indices0 = simd.[<$uint s_splat>](0);
                let mut best_indices1 = simd.[<$uint s_splat>](0);
                let mut indices0 = simd.[<$uint s_add>](iota, simd.[<$uint s_splat>](0));
                let mut indices1 = simd.[<$uint s_add>](indices0, increment1);

                let (dst_head2, dst_tail2) = pulp::as_arrays_mut::<2, _>(dst_head);
                let (lhs_head2, lhs_tail2) = pulp::as_arrays::<2, _>(lhs_head);

                let rhs_v = simd.[<$scalar s_splat>](pulp::cast(rhs));
                for ([dst0, dst1], [lhs0, lhs1]) in dst_head2.iter_mut().zip(lhs_head2) {
                    let new_dst0 = simd.[<$scalar s_mul_add_e>](*lhs0, rhs_v, *dst0);
                    let new_dst1 = simd.[<$scalar s_mul_add_e>](*lhs1, rhs_v, *dst1);
                    *dst0 = new_dst0;
                    *dst1 = new_dst1;

                    (best_value0, best_indices0) =
                        [<best_ $scalar>](simd, best_value0, best_indices0, new_dst0, indices0);
                    (best_value1, best_indices1) =
                        [<best_ $scalar>](simd, best_value1, best_indices1, new_dst1, indices1);
                    indices0 = simd.[<$uint s_add>](indices0, increment2);
                    indices1 = simd.[<$uint s_add>](indices1, increment2);
                }

                (best_value0, best_indices0) =
                    [<best_ $real_scalar>](simd, best_value0, best_indices0, best_value1, best_indices1);

                for (dst, lhs) in dst_tail2
                    .iter_mut()
                    .zip(lhs_tail2)
                {
                    let new_dst = simd.[<$scalar s_mul_add_e>](*lhs, rhs_v, *dst);
                    *dst = new_dst;
                    (best_value0, best_indices0) =
                        [<best_ $scalar>](simd, best_value0, best_indices0, new_dst, indices0);
                    indices0 = simd.[<$uint s_add>](indices0, increment1);
                }

                {
                    let new_dst = simd.[<$scalar s_mul_add_e>](
                        simd.[<$scalar s_partial_load>](lhs_tail),
                        rhs_v,
                        simd.[<$scalar s_partial_load>](dst_tail),
                    );
                    simd.[<$scalar s_partial_store>](dst_tail, new_dst);
                    (best_value0, best_indices0) =
                        [<best_ $scalar>](simd, best_value0, best_indices0, new_dst, indices0);
                }

                (
                    best_value0,
                    best_indices0,
                )
            }
        }
    };
}
best_in_col_simd!(c64, f64, u64);
best_in_col_simd!(c32, f32, u32);

#[inline(always)]
fn reduce2d<E: RealField>(
    len: usize,
    best_value: GroupFor<E, &[UnitFor<E>]>,
    best_row: &[IndexFor<E>],
    best_col: &[IndexFor<E>],
) -> (E, IndexFor<E>, IndexFor<E>) {
    let (mut best_value_scalar, mut best_row_scalar, mut best_col_scalar) = (
        E::faer_zero(),
        E::faer_usize_to_index(0),
        E::faer_usize_to_index(0),
    );
    for ((value, &row), &col) in E::faer_into_iter(best_value)
        .zip(best_row)
        .zip(best_col)
        .take(len)
    {
        let value = E::faer_from_units(E::faer_deref(value));
        (best_value_scalar, best_row_scalar, best_col_scalar) = {
            if value > best_value_scalar {
                (value, row, col)
            } else {
                (best_value_scalar, best_row_scalar, best_col_scalar)
            }
        };
    }
    (best_value_scalar, best_row_scalar, best_col_scalar)
}

#[inline(always)]
fn reduce2d_f64(best_value: &[f64], best_row: &[u64], best_col: &[u64]) -> (f64, u64, u64) {
    let (mut best_value_scalar, mut best_row_scalar, mut best_col_scalar) = (0.0, 0, 0);
    for ((data, &row), &col) in best_value.iter().copied().zip(best_row).zip(best_col) {
        (best_value_scalar, best_row_scalar, best_col_scalar) = best2d_f64(
            pulp::Scalar::new(),
            best_value_scalar,
            best_row_scalar,
            best_col_scalar,
            data,
            row,
            col,
        );
    }
    (best_value_scalar, best_row_scalar, best_col_scalar)
}

#[inline(always)]
fn reduce2d_f32(best_value: &[f32], best_row: &[u32], best_col: &[u32]) -> (f32, u32, u32) {
    let (mut best_value_scalar, mut best_row_scalar, mut best_col_scalar) = (0.0, 0, 0);
    for ((data, &row), &col) in best_value.iter().copied().zip(best_row).zip(best_col) {
        (best_value_scalar, best_row_scalar, best_col_scalar) = best2d_f32(
            pulp::Scalar::new(),
            best_value_scalar,
            best_row_scalar,
            best_col_scalar,
            data,
            row,
            col,
        );
    }
    (best_value_scalar, best_row_scalar, best_col_scalar)
}

#[inline(always)]
fn best_in_col_c64<S: Simd>(simd: S, data: &[c64]) -> (S::f64s, S::u64s) {
    best_in_col_c64_generic(simd, cast_lossy([0, 0, 1, 1, 2, 2, 3, 3_u64]), data)
}

#[inline(always)]
fn update_and_best_in_col_c64<S: Simd>(
    simd: S,
    dst: &mut [c64],
    lhs: &[c64],
    rhs: c64,
) -> (S::f64s, S::u64s) {
    update_and_best_in_col_c64_generic(
        simd,
        cast_lossy([0, 0, 1, 1, 2, 2, 3, 3_u64]),
        dst,
        lhs,
        rhs,
    )
}

#[inline(always)]
fn best_in_col_c32<S: Simd>(simd: S, data: &[c32]) -> (S::f32s, S::u32s) {
    best_in_col_c32_generic(
        simd,
        cast_lossy([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7_u32]),
        data,
    )
}

#[inline(always)]
fn update_and_best_in_col_c32<S: Simd>(
    simd: S,
    dst: &mut [c32],
    lhs: &[c32],
    rhs: c32,
) -> (S::f32s, S::u32s) {
    update_and_best_in_col_c32_generic(
        simd,
        cast_lossy([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7_u32]),
        dst,
        lhs,
        rhs,
    )
}

fn best_in_matrix_simd<E: ComplexField>(matrix: MatRef<'_, E>) -> (usize, usize, E::Real) {
    struct BestInMat<'a, E: ComplexField>(MatRef<'a, E>);
    impl<E: ComplexField> pulp::WithSimd for BestInMat<'_, E> {
        type Output = (usize, usize, E::Real);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let matrix = self.0;
            debug_assert!(matrix.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row =
                E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(0));
            let mut best_col =
                E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(0));
            let mut best_value = E::Real::faer_simd_splat(simd, E::Real::faer_zero());

            for j in 0..n {
                let ptr = matrix.col(j).as_ptr();
                let col = E::faer_map(
                    ptr,
                    #[inline(always)]
                    |ptr| unsafe { slice::from_raw_parts(ptr, m) },
                );
                let (best_value_in_col, best_index_in_col) = best_in_col::<E, S>(simd, col);
                (best_value, best_row, best_col) = best2d::<E::Real, S>(
                    simd,
                    best_value,
                    best_row,
                    best_col,
                    best_value_in_col,
                    best_index_in_col,
                    E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(j)),
                );
            }

            let len = Ord::min(
                m,
                core::mem::size_of::<SimdIndexFor<E, S>>() / core::mem::size_of::<IndexFor<E>>(),
            );
            let (best_value, best_row, best_col) = reduce2d::<E::Real>(
                len,
                simd::one_simd_as_slice::<E::Real, S>(E::Real::faer_as_ref(&best_value)),
                simd::simd_index_as_slice::<E::Real, S>(&[best_row]),
                simd::simd_index_as_slice::<E::Real, S>(&[best_col]),
            );

            (
                E::Real::faer_index_to_usize(best_row),
                E::Real::faer_index_to_usize(best_col),
                best_value,
            )
        }
    }
    E::Simd::default().dispatch(BestInMat(matrix))
}

fn update_and_best_in_matrix_simd<E: ComplexField>(
    matrix: MatMut<'_, E>,
    lhs: MatRef<'_, E>,
    rhs: MatRef<'_, E>,
) -> (usize, usize, E::Real) {
    struct UpdateAndBestInMat<'a, E: ComplexField>(MatMut<'a, E>, MatRef<'a, E>, MatRef<'a, E>);
    impl<E: ComplexField> pulp::WithSimd for UpdateAndBestInMat<'_, E> {
        type Output = (usize, usize, E::Real);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let UpdateAndBestInMat(mut matrix, lhs, rhs) = self;
            assert!(matrix.row_stride() == 1);
            assert!(lhs.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row =
                E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(0));
            let mut best_col =
                E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(0));
            let mut best_value = E::Real::faer_simd_splat(simd, E::Real::faer_zero());

            let lhs = E::faer_map(
                lhs.as_ptr(),
                #[inline(always)]
                |ptr| unsafe { slice::from_raw_parts(ptr, m) },
            );

            for j in 0..n {
                let rhs = rhs.read(0, j).faer_neg();

                let ptr = matrix.rb_mut().col(j).as_ptr();
                let dst = E::faer_map(
                    ptr,
                    #[inline(always)]
                    |ptr| unsafe { slice::from_raw_parts_mut(ptr, m) },
                );

                let (best_value_in_col, best_index_in_col) =
                    update_and_best_in_col(simd, dst, E::faer_copy(&lhs), rhs);

                (best_value, best_row, best_col) = best2d::<E::Real, S>(
                    simd,
                    best_value,
                    best_row,
                    best_col,
                    best_value_in_col,
                    best_index_in_col,
                    E::Real::faer_simd_index_splat(simd, E::Real::faer_usize_to_index(j)),
                );
            }

            let len = Ord::min(
                m,
                core::mem::size_of::<SimdIndexFor<E, S>>() / core::mem::size_of::<IndexFor<E>>(),
            );
            let (best_value, best_row, best_col) = reduce2d::<E::Real>(
                len,
                simd::one_simd_as_slice::<E::Real, S>(E::Real::faer_as_ref(&best_value)),
                simd::simd_index_as_slice::<E::Real, S>(&[best_row]),
                simd::simd_index_as_slice::<E::Real, S>(&[best_col]),
            );

            (
                E::Real::faer_index_to_usize(best_row),
                E::Real::faer_index_to_usize(best_col),
                best_value,
            )
        }
    }
    E::Simd::default().dispatch(UpdateAndBestInMat(matrix, lhs, rhs))
}

fn best_in_matrix_c64(matrix: MatRef<'_, c64>) -> (usize, usize, f64) {
    struct BestInMat<'a>(MatRef<'a, c64>);
    impl pulp::WithSimd for BestInMat<'_> {
        type Output = (usize, usize, f64);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let matrix = self.0;
            debug_assert!(matrix.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row = simd.u64s_splat(0);
            let mut best_col = simd.u64s_splat(0);
            let mut best_value = simd.f64s_splat(0.0);

            for j in 0..n {
                let ptr = matrix.col(j).as_ptr();
                let col = unsafe { slice::from_raw_parts(ptr, m) };
                let (best_value_in_col, best_index_in_col) = best_in_col_c64(simd, col);
                (best_value, best_row, best_col) = best2d_f64(
                    simd,
                    best_value,
                    best_row,
                    best_col,
                    best_value_in_col,
                    best_index_in_col,
                    simd.u64s_splat(j as u64),
                );
            }

            let (best_value, best_row, best_col) = reduce2d_f64(
                bytemuck::cast_slice(&[best_value]),
                bytemuck::cast_slice(&[best_row]),
                bytemuck::cast_slice(&[best_col]),
            );

            (best_row as usize, best_col as usize, best_value)
        }
    }
    <c64 as ComplexField>::Simd::default().dispatch(BestInMat(matrix))
}

fn update_and_best_in_matrix_c64(
    matrix: MatMut<'_, c64>,
    lhs: MatRef<'_, c64>,
    rhs: MatRef<'_, c64>,
) -> (usize, usize, f64) {
    struct UpdateAndBestInMat<'a>(MatMut<'a, c64>, MatRef<'a, c64>, MatRef<'a, c64>);
    impl pulp::WithSimd for UpdateAndBestInMat<'_> {
        type Output = (usize, usize, f64);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let UpdateAndBestInMat(mut matrix, lhs, rhs) = self;
            assert!(matrix.row_stride() == 1);
            assert!(lhs.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row = simd.u64s_splat(0);
            let mut best_col = simd.u64s_splat(0);
            let mut best_value = simd.f64s_splat(0.0);

            let lhs = unsafe { slice::from_raw_parts(lhs.as_ptr(), m) };

            for j in 0..n {
                let rhs = -rhs.read(0, j);

                let ptr = matrix.rb_mut().col(j).as_ptr();
                let dst = unsafe { slice::from_raw_parts_mut(ptr, m) };

                let (best_value_in_col, best_index_in_col) =
                    update_and_best_in_col_c64(simd, dst, lhs, rhs);
                (best_value, best_row, best_col) = best2d_f64(
                    simd,
                    best_value,
                    best_row,
                    best_col,
                    best_value_in_col,
                    best_index_in_col,
                    simd.u64s_splat(j as u64),
                );
            }

            let (best_value, best_row, best_col) = reduce2d_f64(
                bytemuck::cast_slice(&[best_value]),
                bytemuck::cast_slice(&[best_row]),
                bytemuck::cast_slice(&[best_col]),
            );

            (best_row as usize, best_col as usize, best_value)
        }
    }
    <c64 as ComplexField>::Simd::default().dispatch(UpdateAndBestInMat(matrix, lhs, rhs))
}

fn best_in_matrix_c32(matrix: MatRef<'_, c32>) -> (usize, usize, f32) {
    struct BestInMat<'a>(MatRef<'a, c32>);
    impl pulp::WithSimd for BestInMat<'_> {
        type Output = (usize, usize, f32);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let matrix = self.0;
            debug_assert!(matrix.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row = simd.u32s_splat(0);
            let mut best_col = simd.u32s_splat(0);
            let mut best_value = simd.f32s_splat(0.0);

            for j in 0..n {
                let ptr = matrix.col(j).as_ptr();
                let col = unsafe { slice::from_raw_parts(ptr, m) };
                let (best_value_in_col, best_index_in_col) = best_in_col_c32(simd, col);
                (best_value, best_row, best_col) = best2d_f32(
                    simd,
                    best_value,
                    best_row,
                    best_col,
                    best_value_in_col,
                    best_index_in_col,
                    simd.u32s_splat(j as u32),
                );
            }

            let (best_value, best_row, best_col) = reduce2d_f32(
                bytemuck::cast_slice(&[best_value]),
                bytemuck::cast_slice(&[best_row]),
                bytemuck::cast_slice(&[best_col]),
            );

            (best_row as usize, best_col as usize, best_value)
        }
    }
    <c32 as ComplexField>::Simd::default().dispatch(BestInMat(matrix))
}

fn update_and_best_in_matrix_c32(
    matrix: MatMut<'_, c32>,
    lhs: MatRef<'_, c32>,
    rhs: MatRef<'_, c32>,
) -> (usize, usize, f32) {
    struct UpdateAndBestInMat<'a>(MatMut<'a, c32>, MatRef<'a, c32>, MatRef<'a, c32>);
    impl pulp::WithSimd for UpdateAndBestInMat<'_> {
        type Output = (usize, usize, f32);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let UpdateAndBestInMat(mut matrix, lhs, rhs) = self;
            debug_assert!(matrix.row_stride() == 1);
            debug_assert!(lhs.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row = simd.u32s_splat(0);
            let mut best_col = simd.u32s_splat(0);
            let mut best_value = simd.f32s_splat(0.0);

            let lhs = unsafe { slice::from_raw_parts(lhs.as_ptr(), m) };

            for j in 0..n {
                let rhs = -rhs.read(0, j);

                let ptr = matrix.rb_mut().col(j).as_ptr();
                let dst = unsafe { slice::from_raw_parts_mut(ptr, m) };

                let (best_value_in_col, best_index_in_col) =
                    update_and_best_in_col_c32(simd, dst, lhs, rhs);
                (best_value, best_row, best_col) = best2d_f32(
                    simd,
                    best_value,
                    best_row,
                    best_col,
                    best_value_in_col,
                    best_index_in_col,
                    simd.u32s_splat(j as u32),
                );
            }

            let (best_value, best_row, best_col) = reduce2d_f32(
                bytemuck::cast_slice(&[best_value]),
                bytemuck::cast_slice(&[best_row]),
                bytemuck::cast_slice(&[best_col]),
            );

            (best_row as usize, best_col as usize, best_value)
        }
    }
    <c32 as ComplexField>::Simd::default().dispatch(UpdateAndBestInMat(matrix, lhs, rhs))
}

#[inline]
fn best_in_matrix<E: ComplexField>(matrix: MatRef<'_, E>) -> (usize, usize, E::Real) {
    let is_c64 = coe::is_same::<c64, E>();
    let is_c32 = coe::is_same::<c32, E>();

    let is_col_major = matrix.row_stride() == 1;

    if is_col_major && is_c64 {
        coe::coerce_static(best_in_matrix_c64(matrix.coerce()))
    } else if is_col_major && is_c32 {
        coe::coerce_static(best_in_matrix_c32(matrix.coerce()))
    } else if is_col_major {
        best_in_matrix_simd(matrix)
    } else {
        let m = matrix.nrows();
        let n = matrix.ncols();

        let mut max = E::Real::faer_zero();
        let mut max_row = 0;
        let mut max_col = 0;

        for j in 0..n {
            for i in 0..m {
                let abs = matrix.read(i, j).faer_score();
                if abs > max {
                    max_row = i;
                    max_col = j;
                    max = abs;
                }
            }
        }

        (max_row, max_col, max)
    }
}

#[inline]
fn rank_one_update_and_best_in_matrix<E: ComplexField>(
    mut dst: MatMut<'_, E>,
    lhs: MatRef<'_, E>,
    rhs: MatRef<'_, E>,
) -> (usize, usize, E::Real) {
    let is_c64 = coe::is_same::<c64, E>();
    let is_c32 = coe::is_same::<c32, E>();

    let is_col_major = dst.row_stride() == 1 && lhs.row_stride() == 1;

    if is_c64 && is_col_major {
        coe::coerce_static(update_and_best_in_matrix_c64(
            dst.coerce(),
            lhs.coerce(),
            rhs.coerce(),
        ))
    } else if is_c32 && is_col_major {
        coe::coerce_static(update_and_best_in_matrix_c32(
            dst.coerce(),
            lhs.coerce(),
            rhs.coerce(),
        ))
    } else if is_col_major {
        update_and_best_in_matrix_simd(dst, lhs, rhs)
    } else {
        matmul(
            dst.rb_mut(),
            lhs,
            rhs,
            Some(E::faer_one()),
            E::faer_one().faer_neg(),
            Parallelism::None,
        );
        best_in_matrix(dst.rb())
    }
}

#[inline]
fn lu_in_place_unblocked<E: ComplexField>(
    mut matrix: MatMut<'_, E>,
    row_transpositions: &mut [usize],
    col_transpositions: &mut [usize],
    parallelism: Parallelism,
    transposed: bool,
    disable_parallelism: fn(usize, usize) -> bool,
) -> usize {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = Ord::min(m, n);

    debug_assert!(row_transpositions.len() == size);
    debug_assert!(col_transpositions.len() == size);

    if n == 0 || m == 0 {
        return 0;
    }

    let mut n_transpositions = 0;

    let (mut max_row, mut max_col, mut biggest) = best_in_matrix(matrix.rb());

    for k in 0..size {
        if biggest < E::Real::faer_zero_threshold().unwrap() {
            for idx in k..size {
                row_transpositions[idx] = idx;
                col_transpositions[idx] = idx;
            }
            break;
        }

        row_transpositions[k] = max_row;
        col_transpositions[k] = max_col;

        if max_row != k {
            n_transpositions += 1;
            swap_rows(matrix.rb_mut(), k, max_row);
        }

        if max_col != k {
            n_transpositions += 1;
            swap_cols(matrix.rb_mut(), k, max_col);
        }

        let inv = matrix.read(k, k).faer_inv();
        if !transposed {
            for i in k + 1..m {
                let elem = matrix.read(i, k);
                matrix.write(i, k, elem.faer_mul(inv));
            }
        } else {
            for i in k + 1..n {
                let elem = matrix.read(k, i);
                matrix.write(k, i, elem.faer_mul(inv));
            }
        }

        if k + 1 == size {
            break;
        }

        let [_, top_right, bottom_left, bottom_right] = matrix.rb_mut().split_at(k + 1, k + 1);

        let parallelism = if disable_parallelism(m - k, n - k) {
            Parallelism::None
        } else {
            parallelism
        };

        match parallelism {
            Parallelism::None => {
                (max_row, max_col, biggest) = rank_one_update_and_best_in_matrix(
                    bottom_right,
                    bottom_left.col(k).rb(),
                    top_right.row(k).rb(),
                );
            }
            #[cfg(feature = "rayon")]
            _ => {
                use faer_core::{for_each_raw, par_split_indices, parallelism_degree, Ptr};

                let n_threads = parallelism_degree(parallelism);

                let mut biggest_vec = vec![(0_usize, 0_usize, E::Real::faer_zero()); n_threads];

                let lhs = bottom_left.col(k).into_const();
                let rhs = top_right.row(k).into_const();

                {
                    let biggest = Ptr(biggest_vec.as_mut_ptr());

                    for_each_raw(
                        n_threads,
                        |idx| {
                            let (col_start, ncols) =
                                par_split_indices(bottom_right.ncols(), idx, n_threads);
                            let matrix =
                                unsafe { bottom_right.rb().subcols(col_start, ncols).const_cast() };
                            let rhs = rhs.subcols(col_start, matrix.ncols());
                            let biggest = unsafe { &mut *{ biggest }.0.add(idx) };
                            *biggest = rank_one_update_and_best_in_matrix(matrix, lhs, rhs);
                            biggest.1 += col_start;
                        },
                        parallelism,
                    );
                }

                max_row = 0;
                max_col = 0;
                let mut biggest_value = E::Real::faer_zero();
                for (row, col, value) in biggest_vec {
                    if value > biggest_value {
                        max_row = row;
                        max_col = col;
                        biggest_value = value;
                    }
                }
                biggest = biggest_value;
            }
        };
        max_row += k + 1;
        max_col += k + 1;
    }

    n_transpositions
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct FullPivLuComputeParams {
    pub disable_parallelism: Option<fn(nrows: usize, ncols: usize) -> bool>,
}

/// Computes the size and alignment of required workspace for performing an LU
/// decomposition with full pivoting.
pub fn lu_in_place_req<E: 'static>(
    m: usize,
    n: usize,
    parallelism: Parallelism,
    params: FullPivLuComputeParams,
) -> Result<StackReq, dyn_stack::SizeOverflow> {
    let _ = parallelism;
    let _ = params;
    StackReq::try_all_of([
        StackReq::try_new::<usize>(m)?,
        StackReq::try_new::<usize>(n)?,
    ])
}

fn default_disable_parallelism(m: usize, n: usize) -> bool {
    let prod = m * n;
    prod < 512 * 256
}

/// Computes the LU decomposition of the given matrix with partial pivoting, replacing the matrix
/// with its factors in place.
///
/// The decomposition is such that:
/// $$PAQ^\top = LU,$$
/// where $P$ and $Q$ are permutation matrices, $L$ is a unit lower triangular matrix, and $U$ is
/// an upper triangular matrix.
///
/// - $L$ is stored in the strictly lower triangular half of `matrix`, with an implicit unit
///   diagonal,
/// - $U$ is stored in the upper triangular half of `matrix`,
/// - the permutation representing $P$, as well as its inverse, are stored in `row_perm` and
///   `row_perm_inv` respectively,
/// - the permutation representing $Q$, as well as its inverse, are stored in `col_perm` and
///   `col_perm_inv` respectively.
///
/// After the function returns, `row_perm` (resp. `col_perm`) contains the order of the rows (resp.
/// columns) after pivoting, i.e. the result is the same as computing the non-pivoted LU
/// decomposition of the matrix `matrix[row_perm, col_perm]`. `row_perm_inv` (resp. `col_perm_inv`)
/// contains its inverse permutation.
///
/// # Output
///
/// - The number of transpositions that constitute the permutation,
/// - a structure representing the permutation $P$.
/// - a structure representing the permutation $Q$.
///
/// # Panics
///
/// - Panics if the length of the row permutation slices is not equal to the number of rows of the
///   matrix.
/// - Panics if the length of the column permutation slices is not equal to the number of columns of
///   the matrix.
/// - Panics if the provided memory in `stack` is insufficient (see [`lu_in_place_req`]).
pub fn lu_in_place<'out, E: ComplexField>(
    matrix: MatMut<'_, E>,
    row_perm: &'out mut [usize],
    row_perm_inv: &'out mut [usize],
    col_perm: &'out mut [usize],
    col_perm_inv: &'out mut [usize],
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: FullPivLuComputeParams,
) -> (usize, PermutationMut<'out>, PermutationMut<'out>) {
    let disable_parallelism = params
        .disable_parallelism
        .unwrap_or(default_disable_parallelism);

    let _ = parallelism;
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = Ord::min(m, n);

    assert!(row_perm.len() == m);
    assert!(row_perm_inv.len() == m);
    assert!(col_perm.len() == n);
    assert!(col_perm_inv.len() == n);

    #[cfg(feature = "perf-warn")]
    if (matrix.col_stride().unsigned_abs() == 1 || matrix.row_stride().unsigned_abs() != 1)
        && faer_core::__perf_warn!(LU_WARN)
    {
        log::warn!(target: "faer_perf", "LU with full pivoting prefers column-major or row-major matrix. Found matrix with generic strides.");
    }

    let (row_transpositions, stack) = stack.make_with(size, |_| 0usize);
    let (col_transpositions, _) = stack.make_with(size, |_| 0usize);

    let n_transpositions = if matrix.row_stride().abs() < matrix.col_stride().abs() {
        lu_in_place_unblocked(
            matrix,
            row_transpositions,
            col_transpositions,
            parallelism,
            false,
            disable_parallelism,
        )
    } else {
        lu_in_place_unblocked(
            matrix.transpose(),
            col_transpositions,
            row_transpositions,
            parallelism,
            true,
            disable_parallelism,
        )
    };

    row_perm.iter_mut().enumerate().for_each(|(i, e)| *e = i);
    for (i, t) in row_transpositions.iter().copied().enumerate() {
        row_perm.swap(i, t);
    }

    col_perm.iter_mut().enumerate().for_each(|(i, e)| *e = i);
    for (i, t) in col_transpositions.iter().copied().enumerate() {
        col_perm.swap(i, t);
    }

    for (i, p) in row_perm.iter().copied().enumerate() {
        row_perm_inv[p] = i;
    }
    for (i, p) in col_perm.iter().copied().enumerate() {
        col_perm_inv[p] = i;
    }

    unsafe {
        (
            n_transpositions,
            PermutationMut::new_unchecked(row_perm, row_perm_inv),
            PermutationMut::new_unchecked(col_perm, col_perm_inv),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::full_pivoting::reconstruct;
    use assert2::assert;
    use faer_core::{c32, c64, permutation::PermutationRef, Mat};
    use rand::random;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    fn reconstruct_matrix<E: ComplexField>(
        lu_factors: MatRef<'_, E>,
        row_perm: PermutationRef<'_>,
        col_perm: PermutationRef<'_>,
    ) -> Mat<E> {
        let m = lu_factors.nrows();
        let n = lu_factors.ncols();
        let mut dst = Mat::zeros(m, n);
        reconstruct::reconstruct(
            dst.as_mut(),
            lu_factors,
            row_perm,
            col_perm,
            Parallelism::Rayon(0),
            make_stack!(reconstruct::reconstruct_req::<E>(
                m,
                n,
                Parallelism::Rayon(0)
            )),
        );
        dst
    }

    fn compute_lu_col_major_generic<E: ComplexField>(random: fn() -> E, epsilon: E::Real) {
        for (m, n) in [
            (2, 4),
            (2, 2),
            (3, 3),
            (4, 2),
            (2, 1),
            (4, 4),
            (20, 20),
            (20, 2),
            (2, 20),
            (40, 20),
            (20, 40),
            (1024, 1023),
        ] {
            let random_mat = Mat::from_fn(m, n, |_i, _j| random());
            for parallelism in [Parallelism::None, Parallelism::Rayon(0)] {
                let mut mat = random_mat.clone();
                let mat_orig = mat.clone();
                let mut row_perm = vec![0; m];
                let mut row_perm_inv = vec![0; m];
                let mut col_perm = vec![0; n];
                let mut col_perm_inv = vec![0; n];

                let (_, row_perm, col_perm) = lu_in_place(
                    mat.as_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    &mut col_perm,
                    &mut col_perm_inv,
                    parallelism,
                    make_stack!(lu_in_place_req::<f64>(
                        m,
                        n,
                        Parallelism::None,
                        Default::default()
                    )),
                    Default::default(),
                );
                let reconstructed = reconstruct_matrix(mat.as_ref(), row_perm.rb(), col_perm.rb());

                for i in 0..m {
                    for j in 0..n {
                        assert!(
                            (mat_orig.read(i, j).faer_sub(reconstructed.read(i, j))).faer_abs()
                                < epsilon
                        );
                    }
                }
            }
        }
    }

    fn compute_lu_row_major_generic<E: ComplexField>(random: fn() -> E, epsilon: E::Real) {
        for (m, n) in [
            (2, 4),
            (2, 2),
            (3, 3),
            (4, 2),
            (2, 1),
            (4, 4),
            (20, 20),
            (20, 2),
            (2, 20),
            (40, 20),
            (20, 40),
            (1024, 1023),
        ] {
            let random_mat = Mat::from_fn(n, m, |_i, _j| random());
            for parallelism in [Parallelism::None, Parallelism::Rayon(0)] {
                let mut mat = random_mat.clone();
                let mat_orig = mat.clone();

                let mut mat = mat.as_mut().transpose();
                let mat_orig = mat_orig.as_ref().transpose();

                let mut row_perm = vec![0; m];
                let mut row_perm_inv = vec![0; m];
                let mut col_perm = vec![0; n];
                let mut col_perm_inv = vec![0; n];

                let (_, row_perm, col_perm) = lu_in_place(
                    mat.rb_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    &mut col_perm,
                    &mut col_perm_inv,
                    parallelism,
                    make_stack!(lu_in_place_req::<f64>(
                        m,
                        n,
                        Parallelism::None,
                        Default::default()
                    )),
                    Default::default(),
                );
                let reconstructed = reconstruct_matrix(mat.rb(), row_perm.rb(), col_perm.rb());

                for i in 0..m {
                    for j in 0..n {
                        assert!(
                            (mat_orig.read(i, j).faer_sub(reconstructed.read(i, j))).faer_abs()
                                < epsilon
                        );
                    }
                }
            }
        }
    }

    fn random_c64() -> c64 {
        c64 {
            re: random(),
            im: random(),
        }
    }
    fn random_c32() -> c32 {
        c32 {
            re: random(),
            im: random(),
        }
    }

    #[test]
    fn test_compute_lu_row_major() {
        compute_lu_col_major_generic::<f64>(random, 1e-6);
        compute_lu_col_major_generic::<f32>(random, 1e-2);
        compute_lu_row_major_generic::<f64>(random, 1e-6);
        compute_lu_row_major_generic::<f32>(random, 1e-2);
        compute_lu_col_major_generic::<c64>(random_c64, 1e-6);
        compute_lu_col_major_generic::<c32>(random_c32, 1e-2);
        compute_lu_row_major_generic::<c64>(random_c64, 1e-6);
        compute_lu_row_major_generic::<c32>(random_c32, 1e-2);
    }

    #[test]
    fn test_lu_c32_zeros() {
        for (m, n) in [
            (2, 4),
            (2, 2),
            (3, 3),
            (4, 2),
            (2, 1),
            (4, 4),
            (20, 20),
            (20, 2),
            (2, 20),
            (40, 20),
            (20, 40),
        ] {
            let mat = Mat::from_fn(m, n, |_i, _j| c32::faer_zero());
            for parallelism in [Parallelism::None, Parallelism::Rayon(0)] {
                let mut mat = mat.clone();
                let mat_orig = mat.clone();

                let mut mat = mat.as_mut();
                let mat_orig = mat_orig.as_ref();

                let mut row_perm = vec![0; m];
                let mut row_perm_inv = vec![0; m];
                let mut col_perm = vec![0; n];
                let mut col_perm_inv = vec![0; n];

                let (_, row_perm, col_perm) = lu_in_place(
                    mat.rb_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    &mut col_perm,
                    &mut col_perm_inv,
                    parallelism,
                    make_stack!(lu_in_place_req::<f64>(
                        m,
                        n,
                        Parallelism::None,
                        Default::default()
                    )),
                    Default::default(),
                );
                let reconstructed = reconstruct_matrix(mat.rb(), row_perm.rb(), col_perm.rb());

                for i in 0..m {
                    for j in 0..n {
                        assert!(
                            (mat_orig.read(i, j).faer_sub(reconstructed.read(i, j))).faer_abs()
                                < 1e-4
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_lu_c32_ones() {
        for (m, n) in [
            (2, 4),
            (2, 2),
            (3, 3),
            (4, 2),
            (2, 1),
            (4, 4),
            (20, 20),
            (20, 2),
            (2, 20),
            (40, 20),
            (20, 40),
        ] {
            let mat = Mat::from_fn(m, n, |_i, _j| c32::faer_one());
            for parallelism in [Parallelism::None, Parallelism::Rayon(0)] {
                let mut mat = mat.clone();
                let mat_orig = mat.clone();

                let mut mat = mat.as_mut();
                let mat_orig = mat_orig.as_ref();

                let mut row_perm = vec![0; m];
                let mut row_perm_inv = vec![0; m];
                let mut col_perm = vec![0; n];
                let mut col_perm_inv = vec![0; n];

                let (_, row_perm, col_perm) = lu_in_place(
                    mat.rb_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    &mut col_perm,
                    &mut col_perm_inv,
                    parallelism,
                    make_stack!(lu_in_place_req::<f64>(
                        m,
                        n,
                        Parallelism::None,
                        Default::default()
                    )),
                    Default::default(),
                );
                let reconstructed = reconstruct_matrix(mat.rb(), row_perm.rb(), col_perm.rb());

                for i in 0..m {
                    for j in 0..n {
                        assert!(
                            (mat_orig.read(i, j).faer_sub(reconstructed.read(i, j))).faer_abs()
                                < 1e-4
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_lu_c32_rank_2() {
        for (m, n) in [
            (2, 4),
            (2, 2),
            (3, 3),
            (4, 2),
            (2, 1),
            (4, 4),
            (20, 20),
            (20, 2),
            (2, 20),
            (40, 20),
            (20, 40),
        ] {
            let u0 = Mat::from_fn(m, 1, |_, _| c32::new(random(), random()));
            let v0 = Mat::from_fn(1, n, |_, _| c32::new(random(), random()));
            let u1 = Mat::from_fn(m, 1, |_, _| c32::new(random(), random()));
            let v1 = Mat::from_fn(1, n, |_, _| c32::new(random(), random()));

            let mat = u0 * v0 + u1 * v1;
            for parallelism in [Parallelism::None, Parallelism::Rayon(0)] {
                let mut mat = mat.clone();
                let mat_orig = mat.clone();

                let mut mat = mat.as_mut();
                let mat_orig = mat_orig.as_ref();

                let mut row_perm = vec![0; m];
                let mut row_perm_inv = vec![0; m];
                let mut col_perm = vec![0; n];
                let mut col_perm_inv = vec![0; n];

                let (_, row_perm, col_perm) = lu_in_place(
                    mat.rb_mut(),
                    &mut row_perm,
                    &mut row_perm_inv,
                    &mut col_perm,
                    &mut col_perm_inv,
                    parallelism,
                    make_stack!(lu_in_place_req::<f64>(
                        m,
                        n,
                        Parallelism::None,
                        Default::default()
                    )),
                    Default::default(),
                );
                let reconstructed = reconstruct_matrix(mat.rb(), row_perm.rb(), col_perm.rb());

                for i in 0..m {
                    for j in 0..n {
                        assert!(
                            (mat_orig.read(i, j).faer_sub(reconstructed.read(i, j))).faer_abs()
                                < 1e-4
                        );
                    }
                }
            }
        }
    }
}
