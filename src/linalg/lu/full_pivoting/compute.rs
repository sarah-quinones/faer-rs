use crate::{
    assert,
    complex_native::{c32, c64},
    debug_assert,
    linalg::matmul::matmul,
    perm::{swap_cols_idx as swap_cols, swap_rows_idx as swap_rows, PermRef},
    utils::{simd::*, slice::*},
    ColRef, Index, MatMut, MatRef, Parallelism, RowMut, SignedIndex,
};
use bytemuck::cast;
use coe::Coerce;
use core::slice;
use dyn_stack::{PodStack, StackReq};
use faer_entity::*;
use pulp::{cast_lossy, Simd};
use reborrow::*;

// doesn't seem like we benefit from vectorization on aarch64 here
#[cfg(target_arch = "aarch64")]
fn aarch64_nodispatch<E: ComplexField, F: pulp::WithSimd>(op: F) -> F::Output {
    pulp::Scalar::new().vectorize(op)
}
#[cfg(not(target_arch = "aarch64"))]
fn aarch64_nodispatch<E: ComplexField, F: pulp::WithSimd>(op: F) -> F::Output {
    E::Simd::default().dispatch(op)
}

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
    simd: SimdFor<E, S>,
    best_value: SimdGroupFor<E, S>,
    best_row_indices: SimdIndexFor<E, S>,
    best_col_indices: SimdIndexFor<E, S>,
    value: SimdGroupFor<E, S>,
    row_indices: SimdIndexFor<E, S>,
    col_indices: SimdIndexFor<E, S>,
) -> (SimdGroupFor<E, S>, SimdIndexFor<E, S>, SimdIndexFor<E, S>) {
    let is_better = simd.greater_than(value, best_value);
    (
        simd.select(is_better, value, best_value),
        simd.index_select(is_better, row_indices, best_row_indices),
        simd.index_select(is_better, col_indices, best_col_indices),
    )
}

#[inline(always)]
fn best_value<E: ComplexField, S: Simd>(
    simd: SimdFor<E, S>,
    best_value: SimdGroupFor<E::Real, S>,
    best_indices: <E::Real as Entity>::SimdIndex<S>,
    data: SimdGroupFor<E, S>,
    indices: <E::Real as Entity>::SimdIndex<S>,
) -> (SimdGroupFor<E::Real, S>, <E::Real as Entity>::SimdIndex<S>) {
    let simd_real = SimdFor::<E::Real, S>::new(simd.simd);
    let value = simd.score(data);
    let is_better = simd_real.greater_than(value, best_value);
    (
        simd_real.select(is_better, value, best_value),
        simd_real.index_select(is_better, indices, best_indices),
    )
}

#[inline(always)]
fn best_score<E: RealField, S: Simd>(
    simd: SimdFor<E, S>,
    best_value: SimdGroupFor<E, S>,
    best_indices: SimdIndexFor<E, S>,
    value: SimdGroupFor<E, S>,
    indices: SimdIndexFor<E, S>,
) -> (SimdGroupFor<E, S>, SimdIndexFor<E, S>) {
    let is_better = simd.greater_than(value, best_value);
    (
        simd.select(is_better, value, best_value),
        simd.index_select(is_better, indices, best_indices),
    )
}

#[inline(always)]
fn best_in_col<E: ComplexField, S: Simd>(
    simd: SimdFor<E, S>,
    data: GroupFor<E, &[UnitFor<E>]>,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> (SimdGroupFor<E::Real, S>, SimdIndexFor<E::Real, S>) {
    let simd_real = SimdFor::<E::Real, S>::new(simd.simd);

    let (head, body, tail) = simd.as_aligned_simd(SliceGroup::<'_, E>::new(data), offset);
    let (body4, body1) = body.as_arrays::<4>();

    let iota = simd_real.index_seq();
    let lane_count = core::mem::size_of::<SimdUnitFor<E, S>>() / core::mem::size_of::<UnitFor<E>>();
    let increment1 = simd_real.index_splat(E::Real::faer_usize_to_index(lane_count));
    let increment4 = simd_real.index_splat(E::Real::faer_usize_to_index(4 * lane_count));

    let mut best_value0 = simd_real.splat(E::Real::faer_one().faer_neg());
    let mut best_value1 = simd_real.splat(E::Real::faer_one().faer_neg());
    let mut best_value2 = simd_real.splat(E::Real::faer_one().faer_neg());
    let mut best_value3 = simd_real.splat(E::Real::faer_one().faer_neg());

    let mut best_indices0 = simd_real.index_splat(E::Real::faer_usize_to_index(0));
    let mut best_indices1 = simd_real.index_splat(E::Real::faer_usize_to_index(0));
    let mut best_indices2 = simd_real.index_splat(E::Real::faer_usize_to_index(0));
    let mut best_indices3 = simd_real.index_splat(E::Real::faer_usize_to_index(0));

    #[inline(always)]
    fn process<E: ComplexField, S: Simd>(
        simd: SimdFor<E, S>,
        data: impl Read<Output = SimdGroupFor<E, S>>,
        indices: SimdIndexFor<E::Real, S>,
        best_values: SimdGroupFor<E::Real, S>,
        best_indices: SimdIndexFor<E::Real, S>,
    ) -> (SimdGroupFor<E::Real, S>, SimdIndexFor<E::Real, S>) {
        best_value::<E, S>(
            simd,
            best_values,
            best_indices,
            data.read_or(simd.splat(E::faer_zero())),
            indices,
        )
    }

    let mut indices0 = simd_real.index_add(
        iota,
        simd_real.index_splat(E::Real::faer_usize_to_index(
            offset.rotate_left_amount().wrapping_neg(),
        )),
    );
    (best_value0, best_indices0) = process(simd, head, indices0, best_value0, best_indices0);

    indices0 = simd_real.index_add(indices0, increment1);
    let mut indices1 = simd_real.index_add(indices0, increment1);
    let mut indices2 = simd_real.index_add(indices1, increment1);
    let mut indices3 = simd_real.index_add(indices2, increment1);
    for [data0, data1, data2, data3] in body4.into_ref_iter().map(RefGroup::unzip) {
        (best_value0, best_indices0) = process(simd, data0, indices0, best_value0, best_indices0);
        (best_value1, best_indices1) = process(simd, data1, indices1, best_value1, best_indices1);
        (best_value2, best_indices2) = process(simd, data2, indices2, best_value2, best_indices2);
        (best_value3, best_indices3) = process(simd, data3, indices3, best_value3, best_indices3);

        indices0 = simd_real.index_add(indices0, increment4);
        indices1 = simd_real.index_add(indices1, increment4);
        indices2 = simd_real.index_add(indices2, increment4);
        indices3 = simd_real.index_add(indices3, increment4);
    }
    for data0 in body1.into_ref_iter() {
        (best_value0, best_indices0) = process(simd, data0, indices0, best_value0, best_indices0);
        indices0 = simd_real.index_add(indices0, increment1);
    }
    (best_value0, best_indices0) = process(simd, tail, indices0, best_value0, best_indices0);

    (best_value0, best_indices0) = best_score::<E::Real, S>(
        simd_real,
        best_value0,
        best_indices0,
        best_value1,
        best_indices1,
    );
    (best_value2, best_indices2) = best_score::<E::Real, S>(
        simd_real,
        best_value2,
        best_indices2,
        best_value3,
        best_indices3,
    );

    best_score::<E::Real, S>(
        simd_real,
        best_value0,
        best_indices0,
        best_value2,
        best_indices2,
    )
}

#[inline(always)]
fn update_and_best_in_col<E: ComplexField, S: Simd>(
    simd: SimdFor<E, S>,
    data: GroupFor<E, &mut [UnitFor<E>]>,
    lhs: GroupFor<E, &[UnitFor<E>]>,
    rhs: E,
    offset: pulp::Offset<SimdMaskFor<E, S>>,
) -> (SimdGroupFor<E::Real, S>, <E::Real as Entity>::SimdIndex<S>) {
    let simd_real = SimdFor::<E::Real, S>::new(simd.simd);

    let rhs = simd.splat(rhs);
    let data = SliceGroupMut::<'_, E>::new(data);
    let len = data.len();
    let lane_count = core::mem::size_of::<SimdUnitFor<E, S>>() / core::mem::size_of::<UnitFor<E>>();

    let (dst_head, dst_body, dst_tail) = simd.as_aligned_simd_mut(data, offset);

    let (dst_body3, dst_body1) = dst_body.as_arrays_mut::<3>();
    let (lhs_head, lhs_body, lhs_tail) =
        simd.as_aligned_simd(SliceGroup::<'_, E>::new(lhs), offset);
    let (lhs_body3, lhs_body1) = lhs_body.as_arrays::<3>();

    let iota = simd_real.index_seq();
    let increment1 = simd_real.index_splat(E::Real::faer_usize_to_index(lane_count));
    let increment3 = simd_real.index_splat(E::Real::faer_usize_to_index(3 * lane_count));

    let mut best_value0 = simd_real.splat(E::Real::faer_zero());
    let mut best_value1 = simd_real.splat(E::Real::faer_zero());
    let mut best_value2 = simd_real.splat(E::Real::faer_zero());

    let mut best_indices0 = simd_real.index_splat(E::Real::faer_usize_to_index(0));
    let mut best_indices1 = simd_real.index_splat(E::Real::faer_usize_to_index(0));
    let mut best_indices2 = simd_real.index_splat(E::Real::faer_usize_to_index(0));

    #[inline(always)]
    fn process<E: ComplexField, S: Simd>(
        simd: SimdFor<E, S>,
        mut dst: impl Write<Output = SimdGroupFor<E, S>>,
        lhs: impl Read<Output = SimdGroupFor<E, S>>,
        rhs: SimdGroupFor<E, S>,
        indices: SimdIndexFor<E::Real, S>,
        best_values: SimdGroupFor<E::Real, S>,
        best_indices: SimdIndexFor<E::Real, S>,
    ) -> (SimdGroupFor<E::Real, S>, SimdIndexFor<E::Real, S>) {
        let zero = simd.splat(E::faer_zero());
        let mut dst_ = dst.read_or(zero);
        let lhs = lhs.read_or(zero);
        dst_ = simd.mul_add_e(rhs, lhs, dst_);
        dst.write(dst_);
        best_value::<E, S>(simd, best_values, best_indices, dst_, indices)
    }

    let mut indices0 = simd_real.index_add(
        iota,
        simd_real.index_splat(E::Real::faer_usize_to_index(
            offset.rotate_left_amount().wrapping_neg(),
        )),
    );
    (best_value0, best_indices0) = process(
        simd,
        dst_head,
        lhs_head,
        rhs,
        indices0,
        best_value0,
        best_indices0,
    );

    if len + offset.rotate_left_amount() == lane_count {
        return (best_value0, best_indices0);
    }

    indices0 = simd_real.index_add(indices0, increment1);
    let mut indices1 = simd_real.index_add(indices0, increment1);
    let mut indices2 = simd_real.index_add(indices1, increment1);
    for ([dst0, dst1, dst2], [lhs0, lhs1, lhs2]) in dst_body3
        .into_mut_iter()
        .map(RefGroupMut::unzip)
        .zip(lhs_body3.into_ref_iter().map(RefGroup::unzip))
    {
        (best_value0, best_indices0) =
            process(simd, dst0, lhs0, rhs, indices0, best_value0, best_indices0);
        (best_value1, best_indices1) =
            process(simd, dst1, lhs1, rhs, indices1, best_value1, best_indices1);
        (best_value2, best_indices2) =
            process(simd, dst2, lhs2, rhs, indices2, best_value2, best_indices2);
        indices0 = simd_real.index_add(indices0, increment3);
        indices1 = simd_real.index_add(indices1, increment3);
        indices2 = simd_real.index_add(indices2, increment3);
    }
    for (dst0, lhs0) in dst_body1.into_mut_iter().zip(lhs_body1.into_ref_iter()) {
        (best_value0, best_indices0) =
            process(simd, dst0, lhs0, rhs, indices0, best_value0, best_indices0);
        indices0 = simd_real.index_add(indices0, increment1);
    }
    (best_value0, best_indices0) = process(
        simd,
        dst_tail,
        lhs_tail,
        rhs,
        indices0,
        best_value0,
        best_indices0,
    );

    (best_value0, best_indices0) = best_score::<E::Real, S>(
        simd_real,
        best_value0,
        best_indices0,
        best_value1,
        best_indices1,
    );
    best_score::<E::Real, S>(
        simd_real,
        best_value0,
        best_indices0,
        best_value2,
        best_indices2,
    )
}

#[inline(always)]
fn best_in_col_c64_generic<S: pulp::Simd>(
    simd: S,
    iota: S::u64s,
    data: &[c64],
) -> (S::f64s, S::u64s) {
    let (head, tail) = S::c64s_as_simd(bytemuck::cast_slice(data));
    let lane_count = core::mem::size_of::<S::c64s>() / core::mem::size_of::<c64>();
    let increment1 = simd.u64s_splat(lane_count as u64);
    let increment3 = simd.u64s_splat(3 * lane_count as u64);
    let mut best_value0 = simd.f64s_splat(0.0);
    let mut best_value1 = simd.f64s_splat(0.0);
    let mut best_value2 = simd.f64s_splat(0.0);
    let mut best_indices0 = simd.u64s_splat(0);
    let mut best_indices1 = simd.u64s_splat(0);
    let mut best_indices2 = simd.u64s_splat(0);
    let mut indices0 = iota;
    let mut indices1 = simd.u64s_add(indices0, increment1);
    let mut indices2 = simd.u64s_add(indices1, increment1);
    let (head3, tail3) = pulp::as_arrays::<3, _>(head);
    for &data in head3 {
        (best_value0, best_indices0) =
            best_c64(simd, best_value0, best_indices0, data[0], indices0);
        (best_value1, best_indices1) =
            best_c64(simd, best_value1, best_indices1, data[1], indices1);
        (best_value2, best_indices2) =
            best_c64(simd, best_value2, best_indices2, data[2], indices2);
        indices0 = simd.u64s_add(indices0, increment3);
        indices1 = simd.u64s_add(indices1, increment3);
        indices2 = simd.u64s_add(indices2, increment3);
    }
    (best_value0, best_indices0) =
        best_f64(simd, best_value0, best_indices0, best_value1, best_indices1);
    (best_value0, best_indices0) =
        best_f64(simd, best_value0, best_indices0, best_value2, best_indices2);
    for &data in tail3 {
        (best_value0, best_indices0) = best_c64(simd, best_value0, best_indices0, data, indices0);
        indices0 = simd.u64s_add(indices0, increment1);
    }
    (best_value0, best_indices0) = best_c64(
        simd,
        best_value0,
        best_indices0,
        simd.c64s_partial_load(tail),
        indices0,
    );
    (best_value0, best_indices0)
}
#[inline(always)]
fn update_and_best_in_col_c64_generic<S: pulp::Simd>(
    simd: S,
    iota: S::u64s,
    dst: &mut [c64],
    lhs: &[c64],
    rhs: c64,
) -> (S::f64s, S::u64s) {
    let lane_count = core::mem::size_of::<S::c64s>() / core::mem::size_of::<c64>();
    let (dst_head, dst_tail) = S::c64s_as_mut_simd(bytemuck::cast_slice_mut(dst));
    let (lhs_head, lhs_tail) = S::c64s_as_simd(bytemuck::cast_slice(lhs));
    let increment1 = simd.u64s_splat(lane_count as u64);
    let increment2 = simd.u64s_splat(2 * lane_count as u64);
    let mut best_value0 = simd.f64s_splat(0.0);
    let mut best_value1 = simd.f64s_splat(0.0);
    let mut best_indices0 = simd.u64s_splat(0);
    let mut best_indices1 = simd.u64s_splat(0);
    let mut indices0 = simd.u64s_add(iota, simd.u64s_splat(0));
    let mut indices1 = simd.u64s_add(indices0, increment1);
    let (dst_head2, dst_tail2) = pulp::as_arrays_mut::<2, _>(dst_head);
    let (lhs_head2, lhs_tail2) = pulp::as_arrays::<2, _>(lhs_head);
    let rhs_v = simd.c64s_splat(pulp::cast(rhs));
    for ([dst0, dst1], [lhs0, lhs1]) in dst_head2.iter_mut().zip(lhs_head2) {
        let new_dst0 = simd.c64s_mul_add_e(*lhs0, rhs_v, *dst0);
        let new_dst1 = simd.c64s_mul_add_e(*lhs1, rhs_v, *dst1);
        *dst0 = new_dst0;
        *dst1 = new_dst1;
        (best_value0, best_indices0) =
            best_c64(simd, best_value0, best_indices0, new_dst0, indices0);
        (best_value1, best_indices1) =
            best_c64(simd, best_value1, best_indices1, new_dst1, indices1);
        indices0 = simd.u64s_add(indices0, increment2);
        indices1 = simd.u64s_add(indices1, increment2);
    }
    (best_value0, best_indices0) =
        best_f64(simd, best_value0, best_indices0, best_value1, best_indices1);
    for (dst, lhs) in dst_tail2.iter_mut().zip(lhs_tail2) {
        let new_dst = simd.c64s_mul_add_e(*lhs, rhs_v, *dst);
        *dst = new_dst;
        (best_value0, best_indices0) =
            best_c64(simd, best_value0, best_indices0, new_dst, indices0);
        indices0 = simd.u64s_add(indices0, increment1);
    }
    {
        let new_dst = simd.c64s_mul_add_e(
            simd.c64s_partial_load(lhs_tail),
            rhs_v,
            simd.c64s_partial_load(dst_tail),
        );
        simd.c64s_partial_store(dst_tail, new_dst);
        (best_value0, best_indices0) =
            best_c64(simd, best_value0, best_indices0, new_dst, indices0);
    }
    (best_value0, best_indices0)
}
#[inline(always)]
fn best_in_col_c32_generic<S: pulp::Simd>(
    simd: S,
    iota: S::u32s,
    data: &[c32],
) -> (S::f32s, S::u32s) {
    let (head, tail) = S::c32s_as_simd(bytemuck::cast_slice(data));
    let lane_count = core::mem::size_of::<S::c32s>() / core::mem::size_of::<c32>();
    let increment1 = simd.u32s_splat(lane_count as u32);
    let increment3 = simd.u32s_splat(3 * lane_count as u32);
    let mut best_value0 = simd.f32s_splat(0.0);
    let mut best_value1 = simd.f32s_splat(0.0);
    let mut best_value2 = simd.f32s_splat(0.0);
    let mut best_indices0 = simd.u32s_splat(0);
    let mut best_indices1 = simd.u32s_splat(0);
    let mut best_indices2 = simd.u32s_splat(0);
    let mut indices0 = iota;
    let mut indices1 = simd.u32s_add(indices0, increment1);
    let mut indices2 = simd.u32s_add(indices1, increment1);
    let (head3, tail3) = pulp::as_arrays::<3, _>(head);
    for &data in head3 {
        (best_value0, best_indices0) =
            best_c32(simd, best_value0, best_indices0, data[0], indices0);
        (best_value1, best_indices1) =
            best_c32(simd, best_value1, best_indices1, data[1], indices1);
        (best_value2, best_indices2) =
            best_c32(simd, best_value2, best_indices2, data[2], indices2);
        indices0 = simd.u32s_add(indices0, increment3);
        indices1 = simd.u32s_add(indices1, increment3);
        indices2 = simd.u32s_add(indices2, increment3);
    }
    (best_value0, best_indices0) =
        best_f32(simd, best_value0, best_indices0, best_value1, best_indices1);
    (best_value0, best_indices0) =
        best_f32(simd, best_value0, best_indices0, best_value2, best_indices2);
    for &data in tail3 {
        (best_value0, best_indices0) = best_c32(simd, best_value0, best_indices0, data, indices0);
        indices0 = simd.u32s_add(indices0, increment1);
    }
    (best_value0, best_indices0) = best_c32(
        simd,
        best_value0,
        best_indices0,
        simd.c32s_partial_load(tail),
        indices0,
    );
    (best_value0, best_indices0)
}
#[inline(always)]
fn update_and_best_in_col_c32_generic<S: pulp::Simd>(
    simd: S,
    iota: S::u32s,
    dst: &mut [c32],
    lhs: &[c32],
    rhs: c32,
) -> (S::f32s, S::u32s) {
    let lane_count = core::mem::size_of::<S::c32s>() / core::mem::size_of::<c32>();
    let (dst_head, dst_tail) = S::c32s_as_mut_simd(bytemuck::cast_slice_mut(dst));
    let (lhs_head, lhs_tail) = S::c32s_as_simd(bytemuck::cast_slice(lhs));
    let increment1 = simd.u32s_splat(lane_count as u32);
    let increment2 = simd.u32s_splat(2 * lane_count as u32);
    let mut best_value0 = simd.f32s_splat(0.0);
    let mut best_value1 = simd.f32s_splat(0.0);
    let mut best_indices0 = simd.u32s_splat(0);
    let mut best_indices1 = simd.u32s_splat(0);
    let mut indices0 = simd.u32s_add(iota, simd.u32s_splat(0));
    let mut indices1 = simd.u32s_add(indices0, increment1);
    let (dst_head2, dst_tail2) = pulp::as_arrays_mut::<2, _>(dst_head);
    let (lhs_head2, lhs_tail2) = pulp::as_arrays::<2, _>(lhs_head);
    let rhs_v = simd.c32s_splat(pulp::cast(rhs));
    for ([dst0, dst1], [lhs0, lhs1]) in dst_head2.iter_mut().zip(lhs_head2) {
        let new_dst0 = simd.c32s_mul_add_e(*lhs0, rhs_v, *dst0);
        let new_dst1 = simd.c32s_mul_add_e(*lhs1, rhs_v, *dst1);
        *dst0 = new_dst0;
        *dst1 = new_dst1;
        (best_value0, best_indices0) =
            best_c32(simd, best_value0, best_indices0, new_dst0, indices0);
        (best_value1, best_indices1) =
            best_c32(simd, best_value1, best_indices1, new_dst1, indices1);
        indices0 = simd.u32s_add(indices0, increment2);
        indices1 = simd.u32s_add(indices1, increment2);
    }
    (best_value0, best_indices0) =
        best_f32(simd, best_value0, best_indices0, best_value1, best_indices1);
    for (dst, lhs) in dst_tail2.iter_mut().zip(lhs_tail2) {
        let new_dst = simd.c32s_mul_add_e(*lhs, rhs_v, *dst);
        *dst = new_dst;
        (best_value0, best_indices0) =
            best_c32(simd, best_value0, best_indices0, new_dst, indices0);
        indices0 = simd.u32s_add(indices0, increment1);
    }
    {
        let new_dst = simd.c32s_mul_add_e(
            simd.c32s_partial_load(lhs_tail),
            rhs_v,
            simd.c32s_partial_load(dst_tail),
        );
        simd.c32s_partial_store(dst_tail, new_dst);
        (best_value0, best_indices0) =
            best_c32(simd, best_value0, best_indices0, new_dst, indices0);
    }
    (best_value0, best_indices0)
}

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
            let simd_real = SimdFor::<E::Real, S>::new(simd);
            let simd = SimdFor::<E, S>::new(simd);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row = simd_real.index_splat(E::Real::faer_usize_to_index(0));
            let mut best_col = simd_real.index_splat(E::Real::faer_usize_to_index(0));
            let mut best_value = simd_real.splat(E::Real::faer_one().faer_neg());
            let offset = simd.align_offset(SliceGroup::<'_, E>::new(
                matrix.rb().try_get_contiguous_col(0),
            ));

            for j in 0..n {
                let col = matrix.try_get_contiguous_col(j);
                let (best_value_in_col, best_index_in_col) = best_in_col::<E, S>(simd, col, offset);
                (best_value, best_row, best_col) = best2d::<E::Real, S>(
                    simd_real,
                    best_value,
                    best_row,
                    best_col,
                    best_value_in_col,
                    best_index_in_col,
                    simd_real.index_splat(E::Real::faer_usize_to_index(j)),
                );
            }

            let len = Ord::min(
                m + offset.rotate_left_amount(),
                core::mem::size_of::<SimdIndexFor<E, S>>() / core::mem::size_of::<IndexFor<E>>(),
            );
            let (best_value, best_row, best_col) =
                reduce2d::<E::Real>(
                    len,
                    faer_entity::one_simd_as_slice::<E::Real, S>(E::Real::faer_as_ref(
                        &from_copy::<E::Real, _>(best_value),
                    )),
                    faer_entity::simd_index_as_slice::<E::Real, S>(&[best_row]),
                    faer_entity::simd_index_as_slice::<E::Real, S>(&[best_col]),
                );

            (
                E::Real::faer_index_to_usize(best_row),
                E::Real::faer_index_to_usize(best_col),
                best_value,
            )
        }
    }
    aarch64_nodispatch::<E, _>(BestInMat(matrix))
}

fn update_and_best_in_matrix_simd<E: ComplexField>(
    mut matrix: MatMut<'_, E>,
    lhs: ColRef<'_, E>,
    mut rhs: RowMut<'_, E>,
    max_row: usize,
) -> (usize, usize, E::Real) {
    struct UpdateAndBestInMatSwap<'a, E: ComplexField>(
        MatMut<'a, E>,
        ColRef<'a, E>,
        RowMut<'a, E>,
        usize,
    );

    struct UpdateAndBestInMat<'a, E: ComplexField>(MatMut<'a, E>, ColRef<'a, E>, RowMut<'a, E>);
    impl<E: ComplexField> pulp::WithSimd for UpdateAndBestInMat<'_, E> {
        type Output = (usize, usize, E::Real);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let UpdateAndBestInMat(mut matrix, lhs, rhs) = self;
            assert!(matrix.row_stride() == 1);
            assert!(lhs.row_stride() == 1);
            let simd_real = SimdFor::<E::Real, S>::new(simd);
            let simd = SimdFor::<E, S>::new(simd);

            let m = matrix.nrows();
            let n = matrix.ncols();
            assert!(matrix.ncols() == n);
            assert!(rhs.ncols() == n);
            let mut best_row = simd_real.index_splat(E::Real::faer_usize_to_index(0));
            let mut best_col = simd_real.index_splat(E::Real::faer_usize_to_index(0));
            let mut best_value = simd_real.splat(E::Real::faer_one().faer_neg());

            let offset = simd.align_offset(SliceGroup::<'_, E>::new(
                matrix.rb().try_get_contiguous_col(0),
            ));

            let lhs = SliceGroup::<'_, E>::new(lhs.try_get_contiguous_col());

            for j in 0..n {
                let rhs = rhs.read(j).faer_neg();

                let dst =
                    SliceGroupMut::<'_, E>::new(matrix.rb_mut().try_get_contiguous_col_mut(j));

                let (best_value_in_col, best_index_in_col) =
                    update_and_best_in_col(simd, dst.into_inner(), lhs.into_inner(), rhs, offset);

                (best_value, best_row, best_col) = best2d::<E::Real, S>(
                    simd_real,
                    best_value,
                    best_row,
                    best_col,
                    best_value_in_col,
                    best_index_in_col,
                    simd_real.index_splat(E::Real::faer_usize_to_index(j)),
                );
            }

            let len = Ord::min(
                m + offset.rotate_left_amount(),
                core::mem::size_of::<SimdIndexFor<E, S>>() / core::mem::size_of::<IndexFor<E>>(),
            );
            let (best_value, best_row, best_col) =
                reduce2d::<E::Real>(
                    len,
                    faer_entity::one_simd_as_slice::<E::Real, S>(E::Real::faer_as_ref(
                        &from_copy::<E::Real, _>(best_value),
                    )),
                    faer_entity::simd_index_as_slice::<E::Real, S>(&[best_row]),
                    faer_entity::simd_index_as_slice::<E::Real, S>(&[best_col]),
                );

            (
                E::Real::faer_index_to_usize(best_row),
                E::Real::faer_index_to_usize(best_col),
                best_value,
            )
        }
    }

    impl<E: ComplexField> pulp::WithSimd for UpdateAndBestInMatSwap<'_, E> {
        type Output = (usize, usize, E::Real);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let UpdateAndBestInMatSwap(mut matrix, lhs, mut rhs, max_row) = self;
            assert!(matrix.row_stride() == 1);
            assert!(lhs.row_stride() == 1);
            let simd_real = SimdFor::<E::Real, S>::new(simd);
            let simd = SimdFor::<E, S>::new(simd);

            let m = matrix.nrows();
            let n = matrix.ncols();
            assert!(matrix.ncols() == n);
            assert!(rhs.ncols() == n);
            let mut best_row = simd_real.index_splat(E::Real::faer_usize_to_index(0));
            let mut best_col = simd_real.index_splat(E::Real::faer_usize_to_index(0));
            let mut best_value = simd_real.splat(E::Real::faer_one().faer_neg());

            let offset = simd.align_offset(SliceGroup::<'_, E>::new(
                matrix.rb().try_get_contiguous_col(0),
            ));

            let lhs = SliceGroup::<'_, E>::new(lhs.try_get_contiguous_col());

            for j in 0..n {
                let a = rhs.read(j);
                let b = matrix.read(max_row, j);
                rhs.write(j, b);
                matrix.write(max_row, j, a);

                let rhs = rhs.read(j).faer_neg();

                let dst =
                    SliceGroupMut::<'_, E>::new(matrix.rb_mut().try_get_contiguous_col_mut(j));

                let (best_value_in_col, best_index_in_col) =
                    update_and_best_in_col(simd, dst.into_inner(), lhs.into_inner(), rhs, offset);

                (best_value, best_row, best_col) = best2d::<E::Real, S>(
                    simd_real,
                    best_value,
                    best_row,
                    best_col,
                    best_value_in_col,
                    best_index_in_col,
                    simd_real.index_splat(E::Real::faer_usize_to_index(j)),
                );
            }

            let len = Ord::min(
                m + offset.rotate_left_amount(),
                core::mem::size_of::<SimdIndexFor<E, S>>() / core::mem::size_of::<IndexFor<E>>(),
            );
            let (best_value, best_row, best_col) =
                reduce2d::<E::Real>(
                    len,
                    faer_entity::one_simd_as_slice::<E::Real, S>(E::Real::faer_as_ref(
                        &from_copy::<E::Real, _>(best_value),
                    )),
                    faer_entity::simd_index_as_slice::<E::Real, S>(&[best_row]),
                    faer_entity::simd_index_as_slice::<E::Real, S>(&[best_col]),
                );

            (
                E::Real::faer_index_to_usize(best_row),
                E::Real::faer_index_to_usize(best_col),
                best_value,
            )
        }
    }

    if max_row == 0 {
        aarch64_nodispatch::<E, _>(UpdateAndBestInMat(matrix, lhs, rhs))
    } else {
        let max_row = max_row - 1;
        let cs = matrix.col_stride().unsigned_abs();
        let n = matrix.ncols();
        let span = cs.saturating_mul(n);
        if span >= 128 * 128 {
            aarch64_nodispatch::<E, _>(UpdateAndBestInMatSwap(matrix, lhs, rhs, max_row))
        } else {
            for j in 0..n {
                let a = rhs.read(j);
                let b = matrix.read(max_row, j);
                rhs.write(j, b);
                matrix.write(max_row, j, a);
            }
            aarch64_nodispatch::<E, _>(UpdateAndBestInMat(matrix, lhs, rhs))
        }
    }
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
    aarch64_nodispatch::<c64, _>(BestInMat(matrix))
}

fn update_and_best_in_matrix_c64(
    matrix: MatMut<'_, c64>,
    lhs: ColRef<'_, c64>,
    rhs: RowMut<'_, c64>,
    max_row: usize,
) -> (usize, usize, f64) {
    struct UpdateAndBestInMat<'a>(MatMut<'a, c64>, ColRef<'a, c64>, RowMut<'a, c64>, usize);
    impl pulp::WithSimd for UpdateAndBestInMat<'_> {
        type Output = (usize, usize, f64);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let UpdateAndBestInMat(mut matrix, lhs, mut rhs, max_row) = self;
            assert!(matrix.row_stride() == 1);
            assert!(lhs.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row = simd.u64s_splat(0);
            let mut best_col = simd.u64s_splat(0);
            let mut best_value = simd.f64s_splat(0.0);

            let lhs = unsafe { slice::from_raw_parts(lhs.as_ptr(), m) };

            for j in 0..n {
                if max_row > 0 {
                    let a = rhs.read(j);
                    let b = matrix.read(max_row - 1, j);
                    rhs.write(j, b);
                    matrix.write(max_row - 1, j, a);
                }

                let rhs = -rhs.read(j);

                let ptr = matrix.rb_mut().col_mut(j).as_ptr_mut();
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
    aarch64_nodispatch::<c64, _>(UpdateAndBestInMat(matrix, lhs, rhs, max_row))
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
    aarch64_nodispatch::<c32, _>(BestInMat(matrix))
}

fn update_and_best_in_matrix_c32(
    matrix: MatMut<'_, c32>,
    lhs: ColRef<'_, c32>,
    rhs: RowMut<'_, c32>,
    max_row: usize,
) -> (usize, usize, f32) {
    struct UpdateAndBestInMat<'a>(MatMut<'a, c32>, ColRef<'a, c32>, RowMut<'a, c32>, usize);
    impl pulp::WithSimd for UpdateAndBestInMat<'_> {
        type Output = (usize, usize, f32);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let UpdateAndBestInMat(mut matrix, lhs, mut rhs, max_row) = self;
            debug_assert!(matrix.row_stride() == 1);
            debug_assert!(lhs.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row = simd.u32s_splat(0);
            let mut best_col = simd.u32s_splat(0);
            let mut best_value = simd.f32s_splat(0.0);

            let lhs = unsafe { slice::from_raw_parts(lhs.as_ptr(), m) };

            for j in 0..n {
                if max_row > 0 {
                    let a = rhs.read(j);
                    let b = matrix.read(max_row - 1, j);
                    rhs.write(j, b);
                    matrix.write(max_row - 1, j, a);
                }

                let rhs = -rhs.read(j);

                let ptr = matrix.rb_mut().col_mut(j).as_ptr_mut();
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
    aarch64_nodispatch::<c32, _>(UpdateAndBestInMat(matrix, lhs, rhs, max_row))
}

#[inline]
fn best_in_matrix<E: ComplexField>(matrix: MatRef<'_, E>) -> (usize, usize, E::Real) {
    let is_col_major = matrix.row_stride() == 1;

    if is_col_major {
        if const { E::IS_C64 } {
            coe::coerce_static(best_in_matrix_c64(matrix.coerce()))
        } else if const { E::IS_C32 } {
            coe::coerce_static(best_in_matrix_c32(matrix.coerce()))
        } else {
            best_in_matrix_simd(matrix)
        }
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
    lhs: ColRef<'_, E>,
    rhs: RowMut<'_, E>,
    max_row: usize,
) -> (usize, usize, E::Real) {
    let is_col_major = dst.row_stride() == 1 && lhs.row_stride() == 1;
    if is_col_major {
        if const { E::IS_C64 } {
            coe::coerce_static(update_and_best_in_matrix_c64(
                dst.coerce(),
                lhs.coerce(),
                rhs.coerce(),
                max_row,
            ))
        } else if const { E::IS_C32 } {
            coe::coerce_static(update_and_best_in_matrix_c32(
                dst.coerce(),
                lhs.coerce(),
                rhs.coerce(),
                max_row,
            ))
        } else {
            update_and_best_in_matrix_simd(dst, lhs, rhs, max_row)
        }
    } else {
        matmul(
            dst.rb_mut(),
            lhs,
            rhs.rb(),
            Some(E::faer_one()),
            E::faer_one().faer_neg(),
            Parallelism::None,
        );
        best_in_matrix(dst.rb())
    }
}

#[inline]
fn lu_in_place_unblocked<I: Index, E: ComplexField>(
    mut matrix: MatMut<'_, E>,
    row_transpositions: &mut [I],
    col_transpositions: &mut [I],
    parallelism: Parallelism,
    transposed: bool,
    disable_parallelism: fn(usize, usize) -> bool,
) -> usize {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = Ord::min(m, n);

    let truncate = <I::Signed as SignedIndex>::truncate;

    debug_assert!(row_transpositions.len() == size);
    debug_assert!(col_transpositions.len() == size);

    if n == 0 || m == 0 {
        return 0;
    }

    let mut n_transpositions = 0;

    let (mut max_row, mut max_col, mut biggest) = best_in_matrix(matrix.rb());

    for k in 0..size {
        if biggest < E::Real::faer_zero_threshold() {
            for idx in k..size {
                row_transpositions[idx] = I::from_signed(truncate(idx));
                col_transpositions[idx] = I::from_signed(truncate(idx));
            }
            break;
        }

        row_transpositions[k] = I::from_signed(truncate(max_row));
        col_transpositions[k] = I::from_signed(truncate(max_col));

        if max_col != k {
            n_transpositions += 1;
            swap_cols(matrix.rb_mut(), k, max_col);
        }

        if max_row != k {
            n_transpositions += 1;
            swap_rows(matrix.rb_mut().subcols_mut(0, k + 1), k, max_row);
        }

        if !transposed {
            let inv = matrix.read(k, k).faer_inv();

            if matrix.row_stride() == 1 {
                let slice = SliceGroupMut::<'_, E>::new(
                    matrix
                        .rb_mut()
                        .subrows_mut(k + 1, m - k - 1)
                        .try_get_contiguous_col_mut(k),
                );

                struct Div<'a, E: Entity>(SliceGroupMut<'a, E>, E);
                impl<E: ComplexField> pulp::WithSimd for Div<'_, E> {
                    type Output = ();

                    #[inline(always)]
                    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                        let slice = self.0;
                        let simd = SimdFor::<E, S>::new(simd);
                        let inv = simd.splat(self.1);
                        let offset = simd.align_offset(slice.rb());
                        let (head, body, tail) = simd.as_aligned_simd_mut(slice, offset);
                        #[inline(always)]
                        fn process<E: ComplexField, S: Simd>(
                            simd: SimdFor<E, S>,
                            mut a: impl Write<Output = SimdGroupFor<E, S>>,
                            inv: SimdGroupFor<E, S>,
                        ) {
                            a.write(simd.mul(inv, a.read_or(simd.splat(E::faer_zero()))));
                        }

                        process(simd, head, inv);
                        for a in body.into_mut_iter() {
                            process(simd, a, inv);
                        }
                        process(simd, tail, inv);
                    }
                }

                aarch64_nodispatch::<E, _>(Div(slice, inv));
            } else {
                for i in k + 1..m {
                    let elem = matrix.read(i, k);
                    matrix.write(i, k, elem.faer_mul(inv));
                }
            }
        } else {
            let inv = matrix.read(k, k).faer_inv();
            for j in k + 1..n {
                let elem = matrix.read(max_row, j);
                matrix.write(max_row, j, elem.faer_mul(inv));
            }
        }

        if k + 1 == size {
            break;
        }

        let (_, top_right, bottom_left, bottom_right) = matrix.rb_mut().split_at_mut(k + 1, k + 1);

        let parallelism = if disable_parallelism(m - k, n - k) {
            Parallelism::None
        } else {
            parallelism
        };

        match parallelism {
            Parallelism::None => {
                (max_row, max_col, biggest) = rank_one_update_and_best_in_matrix(
                    bottom_right,
                    bottom_left.rb().col(k),
                    top_right.row_mut(k),
                    max_row - k,
                );
            }
            Parallelism::__Private(_) => panic!(),
            #[cfg(feature = "rayon")]
            _ => {
                use crate::utils::thread::{
                    for_each_raw, par_split_indices, parallelism_degree, Ptr,
                };

                let n_threads = parallelism_degree(parallelism);

                let mut biggest_vec = vec![(0_usize, 0_usize, E::Real::faer_zero()); n_threads];

                let lhs = bottom_left.rb().col(k);
                let rhs = top_right.rb().row(k);

                {
                    let biggest = Ptr(biggest_vec.as_mut_ptr());

                    for_each_raw(
                        n_threads,
                        |idx| {
                            let (col_start, ncols) =
                                par_split_indices(bottom_right.ncols(), idx, n_threads);
                            let matrix =
                                unsafe { bottom_right.rb().subcols(col_start, ncols).const_cast() };
                            let rhs =
                                unsafe { rhs.subcols(col_start, matrix.ncols()).const_cast() };
                            let biggest = unsafe { &mut *{ biggest }.0.add(idx) };
                            *biggest =
                                rank_one_update_and_best_in_matrix(matrix, lhs, rhs, max_row - k);
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

/// LU factorization tuning parameters.
#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct FullPivLuComputeParams {
    /// At which size the parallelism should be disabled. `None` to automatically determine this
    /// threshold.
    pub disable_parallelism: Option<fn(nrows: usize, ncols: usize) -> bool>,
}

/// Computes the size and alignment of required workspace for performing an LU
/// decomposition with full pivoting.
pub fn lu_in_place_req<I: Index, E: Entity>(
    m: usize,
    n: usize,
    parallelism: Parallelism,
    params: FullPivLuComputeParams,
) -> Result<StackReq, dyn_stack::SizeOverflow> {
    let _ = parallelism;
    let _ = params;
    StackReq::try_all_of([StackReq::try_new::<I>(m)?, StackReq::try_new::<I>(n)?])
}

fn default_disable_parallelism(m: usize, n: usize) -> bool {
    let prod = m * n;
    prod < 512 * 256
}

/// Information about the resulting LU factorization.
#[derive(Copy, Clone, Debug)]
pub struct FullPivLuInfo {
    /// Number of transpositions that were performed, can be used to compute the determinant of
    /// $PQ$.
    pub transposition_count: usize,
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
pub fn lu_in_place<'out, I: Index, E: ComplexField>(
    matrix: MatMut<'_, E>,
    row_perm: &'out mut [I],
    row_perm_inv: &'out mut [I],
    col_perm: &'out mut [I],
    col_perm_inv: &'out mut [I],
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: FullPivLuComputeParams,
) -> (FullPivLuInfo, PermRef<'out, I>, PermRef<'out, I>) {
    let disable_parallelism = params
        .disable_parallelism
        .unwrap_or(default_disable_parallelism);

    let truncate = <I::Signed as SignedIndex>::truncate;

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
        && crate::__perf_warn!(LU_WARN)
    {
        log::warn!(target: "faer_perf", "LU with full pivoting prefers column-major or row-major matrix. Found matrix with generic strides.");
    }

    let (row_transpositions, stack) = stack.make_with(size, |_| I::from_signed(truncate(0)));
    let (col_transpositions, _) = stack.make_with(size, |_| I::from_signed(truncate(0)));

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
            matrix.transpose_mut(),
            col_transpositions,
            row_transpositions,
            parallelism,
            true,
            disable_parallelism,
        )
    };

    row_perm
        .iter_mut()
        .enumerate()
        .for_each(|(i, e)| *e = I::from_signed(truncate(i)));
    for (i, t) in row_transpositions.iter().copied().enumerate() {
        row_perm.swap(i, t.to_signed().zx());
    }

    col_perm
        .iter_mut()
        .enumerate()
        .for_each(|(i, e)| *e = I::from_signed(truncate(i)));
    for (i, t) in col_transpositions.iter().copied().enumerate() {
        col_perm.swap(i, t.to_signed().zx());
    }

    for (i, p) in row_perm.iter().copied().enumerate() {
        row_perm_inv[p.to_signed().zx()] = I::from_signed(truncate(i));
    }
    for (i, p) in col_perm.iter().copied().enumerate() {
        col_perm_inv[p.to_signed().zx()] = I::from_signed(truncate(i));
    }

    unsafe {
        (
            FullPivLuInfo {
                transposition_count: n_transpositions,
            },
            PermRef::new_unchecked(row_perm, row_perm_inv),
            PermRef::new_unchecked(col_perm, col_perm_inv),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, linalg::lu::full_pivoting::reconstruct, Mat};
    use rand::random;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    fn reconstruct_matrix<I: Index, E: ComplexField>(
        lu_factors: MatRef<'_, E>,
        row_perm: PermRef<'_, I>,
        col_perm: PermRef<'_, I>,
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
            make_stack!(reconstruct::reconstruct_req::<I, E>(
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
            for parallelism in [Parallelism::None, Parallelism::Rayon(4)] {
                let mut mat = random_mat.clone();
                let mat_orig = mat.clone();
                let mut row_perm = vec![0usize; m];
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
                    make_stack!(lu_in_place_req::<usize, f64>(
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
            for parallelism in [Parallelism::None, Parallelism::Rayon(4)] {
                let mut mat = random_mat.clone();
                let mat_orig = mat.clone();

                let mut mat = mat.as_mut().transpose_mut();
                let mat_orig = mat_orig.as_ref().transpose();

                let mut row_perm = vec![0usize; m];
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
                    make_stack!(lu_in_place_req::<usize, f64>(
                        m,
                        n,
                        Parallelism::None,
                        Default::default()
                    )),
                    Default::default(),
                );
                let reconstructed = reconstruct_matrix(mat.rb(), row_perm.rb(), col_perm.rb());
                assert!((&mat_orig - &reconstructed).norm_max() < epsilon);
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

                let mut row_perm = vec![0usize; m];
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
                    make_stack!(lu_in_place_req::<usize, f64>(
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

                let mut row_perm = vec![0usize; m];
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
                    make_stack!(lu_in_place_req::<usize, f64>(
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

                let mut row_perm = vec![0usize; m];
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
                    make_stack!(lu_in_place_req::<usize, f64>(
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
