use core::{
    any::TypeId,
    mem::{size_of, transmute_copy},
};
use std::mem::transmute;

use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use bytemuck::cast;
use dyn_stack::{DynStack, StackReq};
use faer_core::{
    mul::matmul,
    permutation::{swap_cols, swap_rows, PermutationMut},
    ColRef, ComplexField, Conj, MatMut, MatRef, Parallelism, RowRef,
};
use pulp::Simd;
use reborrow::*;

#[inline]
fn coerce<T: 'static, U: 'static>(t: T) -> U {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
    let no_drop = core::mem::MaybeUninit::new(t);
    unsafe { transmute_copy(&no_drop) }
}

#[inline]
fn coerce_col_ref<T: 'static, U: 'static>(t: ColRef<'_, T>) -> ColRef<'_, U> {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
    unsafe { transmute(t) }
}

#[inline]
fn coerce_row_ref<T: 'static, U: 'static>(t: RowRef<'_, T>) -> RowRef<'_, U> {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
    unsafe { transmute(t) }
}

#[inline]
fn coerce_mat_ref<T: 'static, U: 'static>(t: MatRef<'_, T>) -> MatRef<'_, U> {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
    unsafe { transmute(t) }
}

#[inline]
fn coerce_mat_mut<T: 'static, U: 'static>(t: MatMut<'_, T>) -> MatMut<'_, U> {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
    unsafe { transmute(t) }
}

#[inline(always)]
fn best_f64<S: pulp::Simd>(
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
fn best_f32<S: pulp::Simd>(
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
fn best_in_col_f64_generic<S: pulp::Simd>(
    simd: S,
    iota: S::u64s,
    data: &[f64],
) -> (S::f64s, S::u64s, f64, u64) {
    let (head, tail) = S::f64s_as_simd(data);

    let lane_count = core::mem::size_of::<S::u64s>() / core::mem::size_of::<u64>();
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

    let head_chunks = head.chunks_exact(3);

    for data in head_chunks.clone() {
        let d0 = data[0];
        let d1 = data[1];
        let d2 = data[2];
        (best_value0, best_indices0) = best_f64(simd, best_value0, best_indices0, d0, indices0);
        (best_value1, best_indices1) = best_f64(simd, best_value1, best_indices1, d1, indices1);
        (best_value2, best_indices2) = best_f64(simd, best_value2, best_indices2, d2, indices2);
        indices0 = simd.u64s_add(indices0, increment3);
        indices1 = simd.u64s_add(indices1, increment3);
        indices2 = simd.u64s_add(indices2, increment3);
    }

    (best_value0, best_indices0) =
        best_f64(simd, best_value0, best_indices0, best_value1, best_indices1);
    (best_value0, best_indices0) =
        best_f64(simd, best_value0, best_indices0, best_value2, best_indices2);

    for data in head_chunks.remainder().iter().copied() {
        (best_value0, best_indices0) = best_f64(simd, best_value0, best_indices0, data, indices0);
        indices0 = simd.u64s_add(indices0, increment1);
    }

    let mut best_value_scalar = 0.0;
    let mut best_index_scalar = 0;
    let mut index = (head.len() * lane_count) as u64;
    for data in tail.iter().copied() {
        (best_value_scalar, best_index_scalar) = best_f64(
            pulp::Scalar::new(),
            best_value_scalar,
            best_index_scalar,
            data,
            index,
        );
        index += 1;
    }

    (
        best_value0,
        best_indices0,
        best_value_scalar,
        best_index_scalar,
    )
}

#[inline(always)]
fn best_in_col_f32_generic<S: pulp::Simd>(
    simd: S,
    iota: S::u32s,
    data: &[f32],
) -> (S::f32s, S::u32s, f32, u32) {
    let (head, tail) = S::f32s_as_simd(data);

    let lane_count = core::mem::size_of::<S::u32s>() / core::mem::size_of::<u32>();
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

    let head_chunks = head.chunks_exact(3);

    for data in head_chunks.clone() {
        let d0 = data[0];
        let d1 = data[1];
        let d2 = data[2];
        (best_value0, best_indices0) = best_f32(simd, best_value0, best_indices0, d0, indices0);
        (best_value1, best_indices1) = best_f32(simd, best_value1, best_indices1, d1, indices1);
        (best_value2, best_indices2) = best_f32(simd, best_value2, best_indices2, d2, indices2);
        indices0 = simd.u32s_add(indices0, increment3);
        indices1 = simd.u32s_add(indices1, increment3);
        indices2 = simd.u32s_add(indices2, increment3);
    }

    (best_value0, best_indices0) =
        best_f32(simd, best_value0, best_indices0, best_value1, best_indices1);
    (best_value0, best_indices0) =
        best_f32(simd, best_value0, best_indices0, best_value2, best_indices2);

    for data in head_chunks.remainder().iter().copied() {
        (best_value0, best_indices0) = best_f32(simd, best_value0, best_indices0, data, indices0);
        indices0 = simd.u32s_add(indices0, increment1);
    }

    let mut best_value_scalar = 0.0;
    let mut best_index_scalar = 0;
    let mut index = (head.len() * lane_count) as u32;
    for data in tail.iter().copied() {
        (best_value_scalar, best_index_scalar) = best_f32(
            pulp::Scalar::new(),
            best_value_scalar,
            best_index_scalar,
            data,
            index,
        );
        index += 1;
    }

    (
        best_value0,
        best_indices0,
        best_value_scalar,
        best_index_scalar,
    )
}

#[inline(always)]
fn update_and_best_in_col_f64_generic<S: pulp::Simd>(
    simd: S,
    iota: S::u64s,
    dst: &mut [f64],
    lhs: &[f64],
    rhs: f64,
) -> (S::f64s, S::u64s, f64, u64) {
    let lane_count = core::mem::size_of::<S::u64s>() / core::mem::size_of::<u64>();
    let len = dst.len();

    let offset = dst.as_ptr().align_offset(size_of::<S::f64s>());
    let ((dst_prefix, dst_suffix), (lhs_prefix, lhs_suffix)) = (
        dst.split_at_mut(offset.min(len)),
        lhs.split_at(offset.min(len)),
    );

    let mut best_value_scalar = 0.0;
    let mut best_index_scalar = 0;
    let mut index = 0_u64;

    for (dst, lhs) in dst_prefix.iter_mut().zip(lhs_prefix) {
        let new_dst = f64::mul_add(*lhs, rhs, *dst);
        *dst = new_dst;
        (best_value_scalar, best_index_scalar) = best_f64(
            pulp::Scalar::new(),
            best_value_scalar,
            best_index_scalar,
            new_dst,
            index,
        );
        index += 1;
    }

    let (dst_head, dst_tail) = S::f64s_as_mut_simd(dst_suffix);
    let (lhs_head, lhs_tail) = S::f64s_as_simd(lhs_suffix);

    let increment1 = simd.u64s_splat(1 * lane_count as u64);
    let increment2 = simd.u64s_splat(2 * lane_count as u64);

    let mut best_value0 = simd.f64s_splat(0.0);
    let mut best_value1 = simd.f64s_splat(0.0);
    let mut best_indices0 = simd.u64s_splat(0);
    let mut best_indices1 = simd.u64s_splat(0);
    let mut indices0 = simd.u64s_add(iota, simd.u64s_splat(offset as u64));
    let mut indices1 = simd.u64s_add(indices0, increment1);

    let mut dst_head_chunks = dst_head.chunks_exact_mut(2);
    let lhs_head_chunks = lhs_head.chunks_exact(2);

    let rhs_v = simd.f64s_splat(rhs);
    for (dst, lhs) in (&mut dst_head_chunks).zip(lhs_head_chunks.clone()) {
        let (dst0, dst1) = dst.split_at_mut(1);
        let dst0 = &mut dst0[0];
        let dst1 = &mut dst1[0];
        let lhs0 = lhs[0];
        let lhs1 = lhs[1];

        let new_dst0 = simd.f64s_mul_adde(lhs0, rhs_v, *dst0);
        let new_dst1 = simd.f64s_mul_adde(lhs1, rhs_v, *dst1);
        *dst0 = new_dst0;
        *dst1 = new_dst1;

        (best_value0, best_indices0) =
            best_f64(simd, best_value0, best_indices0, new_dst0, indices0);
        (best_value1, best_indices1) =
            best_f64(simd, best_value1, best_indices1, new_dst1, indices1);
        indices0 = simd.u64s_add(indices0, increment2);
        indices1 = simd.u64s_add(indices1, increment2);
    }

    (best_value0, best_indices0) =
        best_f64(simd, best_value0, best_indices0, best_value1, best_indices1);

    for (dst, lhs) in dst_head_chunks
        .into_remainder()
        .iter_mut()
        .zip(lhs_head_chunks.remainder().iter().copied())
    {
        let new_dst = simd.f64s_mul_adde(lhs, rhs_v, *dst);
        *dst = new_dst;
        (best_value0, best_indices0) =
            best_f64(simd, best_value0, best_indices0, new_dst, indices0);
    }

    index = (offset + dst_head.len() * lane_count) as u64;
    for (dst, lhs) in dst_tail.iter_mut().zip(lhs_tail) {
        let new_dst = f64::mul_add(*lhs, rhs, *dst);
        *dst = new_dst;
        (best_value_scalar, best_index_scalar) = best_f64(
            pulp::Scalar::new(),
            best_value_scalar,
            best_index_scalar,
            new_dst,
            index,
        );
        index += 1;
    }

    (
        best_value0,
        best_indices0,
        best_value_scalar,
        best_index_scalar,
    )
}

#[inline(always)]
fn update_and_best_in_col_f32_generic<S: pulp::Simd>(
    simd: S,
    iota: S::u32s,
    dst: &mut [f32],
    lhs: &[f32],
    rhs: f32,
) -> (S::f32s, S::u32s, f32, u32) {
    let lane_count = core::mem::size_of::<S::u32s>() / core::mem::size_of::<u32>();
    let len = dst.len();

    let offset = dst.as_ptr().align_offset(size_of::<S::f32s>());
    let ((dst_prefix, dst_suffix), (lhs_prefix, lhs_suffix)) = (
        dst.split_at_mut(offset.min(len)),
        lhs.split_at(offset.min(len)),
    );

    let mut best_value_scalar = 0.0;
    let mut best_index_scalar = 0;
    let mut index = 0_u32;

    for (dst, lhs) in dst_prefix.iter_mut().zip(lhs_prefix) {
        let new_dst = f32::mul_add(*lhs, rhs, *dst);
        *dst = new_dst;
        (best_value_scalar, best_index_scalar) = best_f32(
            pulp::Scalar::new(),
            best_value_scalar,
            best_index_scalar,
            new_dst,
            index,
        );
        index += 1;
    }

    let (dst_head, dst_tail) = S::f32s_as_mut_simd(dst_suffix);
    let (lhs_head, lhs_tail) = S::f32s_as_simd(lhs_suffix);

    let increment1 = simd.u32s_splat(1 * lane_count as u32);
    let increment2 = simd.u32s_splat(2 * lane_count as u32);

    let mut best_value0 = simd.f32s_splat(0.0);
    let mut best_value1 = simd.f32s_splat(0.0);
    let mut best_indices0 = simd.u32s_splat(0);
    let mut best_indices1 = simd.u32s_splat(0);
    let mut indices0 = simd.u32s_add(iota, simd.u32s_splat(offset as u32));
    let mut indices1 = simd.u32s_add(indices0, increment1);

    let mut dst_head_chunks = dst_head.chunks_exact_mut(2);
    let lhs_head_chunks = lhs_head.chunks_exact(2);

    let rhs_v = simd.f32s_splat(rhs);
    for (dst, lhs) in (&mut dst_head_chunks).zip(lhs_head_chunks.clone()) {
        let (dst0, dst1) = dst.split_at_mut(1);
        let dst0 = &mut dst0[0];
        let dst1 = &mut dst1[0];
        let lhs0 = lhs[0];
        let lhs1 = lhs[1];

        let new_dst0 = simd.f32s_mul_adde(lhs0, rhs_v, *dst0);
        let new_dst1 = simd.f32s_mul_adde(lhs1, rhs_v, *dst1);
        *dst0 = new_dst0;
        *dst1 = new_dst1;

        (best_value0, best_indices0) =
            best_f32(simd, best_value0, best_indices0, new_dst0, indices0);
        (best_value1, best_indices1) =
            best_f32(simd, best_value1, best_indices1, new_dst1, indices1);
        indices0 = simd.u32s_add(indices0, increment2);
        indices1 = simd.u32s_add(indices1, increment2);
    }

    (best_value0, best_indices0) =
        best_f32(simd, best_value0, best_indices0, best_value1, best_indices1);

    for (dst, lhs) in dst_head_chunks
        .into_remainder()
        .iter_mut()
        .zip(lhs_head_chunks.remainder().iter().copied())
    {
        let new_dst = simd.f32s_mul_adde(lhs, rhs_v, *dst);
        *dst = new_dst;
        (best_value0, best_indices0) =
            best_f32(simd, best_value0, best_indices0, new_dst, indices0);
    }

    index = (offset + dst_head.len() * lane_count) as u32;
    for (dst, lhs) in dst_tail.iter_mut().zip(lhs_tail) {
        let new_dst = f32::mul_add(*lhs, rhs, *dst);
        *dst = new_dst;
        (best_value_scalar, best_index_scalar) = best_f32(
            pulp::Scalar::new(),
            best_value_scalar,
            best_index_scalar,
            new_dst,
            index,
        );
        index += 1;
    }

    (
        best_value0,
        best_indices0,
        best_value_scalar,
        best_index_scalar,
    )
}

#[inline(always)]
fn best_in_col_f64x2<S: Simd>(
    simd: S,
    data: &[f64],
    reduce: impl Fn(f64, u64, &[f64], &[u64]) -> (f64, u64),
) -> (f64, u64) {
    let (best_value, best_indices, best_value_s, best_index_s) =
        best_in_col_f64_generic(simd, cast([0, 1_u64]), data);
    let best_value_v: [f64; 2] = cast(best_value);
    let best_index_v: [u64; 2] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}
#[inline(always)]
fn best_in_col_f64x4<S: Simd>(
    simd: S,
    data: &[f64],
    reduce: impl Fn(f64, u64, &[f64], &[u64]) -> (f64, u64),
) -> (f64, u64) {
    let (best_value, best_indices, best_value_s, best_index_s) =
        best_in_col_f64_generic(simd, cast([0, 1, 2, 3_u64]), data);
    let best_value_v: [f64; 4] = cast(best_value);
    let best_index_v: [u64; 4] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}
#[inline(always)]
fn best_in_col_f64x8<S: Simd>(
    simd: S,
    data: &[f64],
    reduce: impl Fn(f64, u64, &[f64], &[u64]) -> (f64, u64),
) -> (f64, u64) {
    let (best_value, best_indices, best_value_s, best_index_s) =
        best_in_col_f64_generic(simd, cast([0, 1, 2, 3, 4, 5, 6, 7_u64]), data);
    let best_value_v: [f64; 8] = cast(best_value);
    let best_index_v: [u64; 8] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}

#[inline(always)]
fn update_and_best_in_col_f64x2<S: Simd>(
    simd: S,
    dst: &mut [f64],
    lhs: &[f64],
    rhs: f64,
    reduce: impl Fn(f64, u64, &[f64], &[u64]) -> (f64, u64),
) -> (f64, u64) {
    let (best_value, best_indices, best_value_s, best_index_s) =
        update_and_best_in_col_f64_generic(simd, cast([0, 1_u64]), dst, lhs, rhs);
    let best_value_v: [f64; 2] = cast(best_value);
    let best_index_v: [u64; 2] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}
#[inline(always)]
fn update_and_best_in_col_f64x4<S: Simd>(
    simd: S,
    dst: &mut [f64],
    lhs: &[f64],
    rhs: f64,
    reduce: impl Fn(f64, u64, &[f64], &[u64]) -> (f64, u64),
) -> (f64, u64) {
    let (best_value, best_indices, best_value_s, best_index_s) =
        update_and_best_in_col_f64_generic(simd, cast([0, 1, 2, 3_u64]), dst, lhs, rhs);
    let best_value_v: [f64; 4] = cast(best_value);
    let best_index_v: [u64; 4] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}
#[inline(always)]
fn update_and_best_in_col_f64x8<S: Simd>(
    simd: S,
    dst: &mut [f64],
    lhs: &[f64],
    rhs: f64,
    reduce: impl Fn(f64, u64, &[f64], &[u64]) -> (f64, u64),
) -> (f64, u64) {
    let (best_value, best_indices, best_value_s, best_index_s) =
        update_and_best_in_col_f64_generic(simd, cast([0, 1, 2, 3, 4, 5, 6, 7_u64]), dst, lhs, rhs);
    let best_value_v: [f64; 8] = cast(best_value);
    let best_index_v: [u64; 8] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}

#[inline(always)]
fn best_in_col_f32x4<S: Simd>(
    simd: S,
    data: &[f32],
    reduce: impl Fn(f32, u32, &[f32], &[u32]) -> (f32, u32),
) -> (f32, u32) {
    let (best_value, best_indices, best_value_s, best_index_s) =
        best_in_col_f32_generic(simd, cast([0, 1, 2, 3_u32]), data);
    let best_value_v: [f32; 4] = cast(best_value);
    let best_index_v: [u32; 4] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}
#[inline(always)]
fn best_in_col_f32x8<S: Simd>(
    simd: S,
    data: &[f32],
    reduce: impl Fn(f32, u32, &[f32], &[u32]) -> (f32, u32),
) -> (f32, u32) {
    let (best_value, best_indices, best_value_s, best_index_s) =
        best_in_col_f32_generic(simd, cast([0, 1, 2, 3, 4, 5, 6, 7_u32]), data);
    let best_value_v: [f32; 8] = cast(best_value);
    let best_index_v: [u32; 8] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}

#[inline(always)]
fn best_in_col_f32x16<S: Simd>(
    simd: S,
    data: &[f32],
    reduce: impl Fn(f32, u32, &[f32], &[u32]) -> (f32, u32),
) -> (f32, u32) {
    let (best_value, best_indices, best_value_s, best_index_s) = best_in_col_f32_generic(
        simd,
        cast([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u32]),
        data,
    );
    let best_value_v: [f32; 16] = cast(best_value);
    let best_index_v: [u32; 16] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}

#[inline(always)]
fn update_and_best_in_col_f32x4<S: Simd>(
    simd: S,
    dst: &mut [f32],
    lhs: &[f32],
    rhs: f32,
    reduce: impl Fn(f32, u32, &[f32], &[u32]) -> (f32, u32),
) -> (f32, u32) {
    let (best_value, best_indices, best_value_s, best_index_s) =
        update_and_best_in_col_f32_generic(simd, cast([0, 1, 2, 3_u32]), dst, lhs, rhs);
    let best_value_v: [f32; 4] = cast(best_value);
    let best_index_v: [u32; 4] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}
#[inline(always)]
fn update_and_best_in_col_f32x8<S: Simd>(
    simd: S,
    dst: &mut [f32],
    lhs: &[f32],
    rhs: f32,
    reduce: impl Fn(f32, u32, &[f32], &[u32]) -> (f32, u32),
) -> (f32, u32) {
    let (best_value, best_indices, best_value_s, best_index_s) =
        update_and_best_in_col_f32_generic(simd, cast([0, 1, 2, 3, 4, 5, 6, 7_u32]), dst, lhs, rhs);
    let best_value_v: [f32; 8] = cast(best_value);
    let best_index_v: [u32; 8] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}

#[inline(always)]
fn update_and_best_in_col_f32x16<S: Simd>(
    simd: S,
    dst: &mut [f32],
    lhs: &[f32],
    rhs: f32,
    reduce: impl Fn(f32, u32, &[f32], &[u32]) -> (f32, u32),
) -> (f32, u32) {
    let (best_value, best_indices, best_value_s, best_index_s) = update_and_best_in_col_f32_generic(
        simd,
        cast([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u32]),
        dst,
        lhs,
        rhs,
    );
    let best_value_v: [f32; 16] = cast(best_value);
    let best_index_v: [u32; 16] = cast(best_indices);
    reduce(best_value_s, best_index_s, &best_value_v, &best_index_v)
}

#[inline(always)]
fn best_in_col_f64<S: Simd>(simd: S, data: &[f64]) -> (f64, u64) {
    let lane_count = core::mem::size_of::<S::u64s>() / core::mem::size_of::<u64>();
    let reduce =
        |mut best_value_scalar, mut best_index_scalar, best_value: &[f64], best_indices: &[u64]| {
            for (data, index) in best_value.iter().copied().zip(best_indices.iter().copied()) {
                (best_value_scalar, best_index_scalar) = best_f64(
                    pulp::Scalar::new(),
                    best_value_scalar,
                    best_index_scalar,
                    data,
                    index,
                );
            }
            (best_value_scalar, best_index_scalar)
        };
    if lane_count == 8 {
        best_in_col_f64x8(simd, data, reduce)
    } else if lane_count == 4 {
        best_in_col_f64x4(simd, data, reduce)
    } else if lane_count == 2 {
        best_in_col_f64x2(simd, data, reduce)
    } else {
        let (best_values, best_indices, _, _) = best_in_col_f64_generic(simd, cast(0_u64), data);
        (cast(best_values), cast(best_indices))
    }
}

#[inline(always)]
fn update_and_best_in_col_f64<S: Simd>(
    simd: S,
    dst: &mut [f64],
    lhs: &[f64],
    rhs: f64,
) -> (f64, u64) {
    let lane_count = core::mem::size_of::<S::u64s>() / core::mem::size_of::<u64>();
    let reduce =
        |mut best_value_scalar, mut best_index_scalar, best_value: &[f64], best_indices: &[u64]| {
            for (data, index) in best_value.iter().copied().zip(best_indices.iter().copied()) {
                (best_value_scalar, best_index_scalar) = best_f64(
                    pulp::Scalar::new(),
                    best_value_scalar,
                    best_index_scalar,
                    data,
                    index,
                );
            }
            (best_value_scalar, best_index_scalar)
        };
    if lane_count == 8 {
        update_and_best_in_col_f64x8(simd, dst, lhs, rhs, reduce)
    } else if lane_count == 4 {
        update_and_best_in_col_f64x4(simd, dst, lhs, rhs, reduce)
    } else if lane_count == 2 {
        update_and_best_in_col_f64x2(simd, dst, lhs, rhs, reduce)
    } else {
        let (best_values, best_indices, _, _) =
            update_and_best_in_col_f64_generic(simd, cast(0_u64), dst, lhs, rhs);
        (cast(best_values), cast(best_indices))
    }
}

#[inline(always)]
fn best_in_col_f32<S: Simd>(simd: S, data: &[f32]) -> (f32, u32) {
    let lane_count = core::mem::size_of::<S::u32s>() / core::mem::size_of::<u32>();
    let reduce =
        |mut best_value_scalar, mut best_index_scalar, best_value: &[f32], best_indices: &[u32]| {
            for (data, index) in best_value.iter().copied().zip(best_indices.iter().copied()) {
                (best_value_scalar, best_index_scalar) = best_f32(
                    pulp::Scalar::new(),
                    best_value_scalar,
                    best_index_scalar,
                    data,
                    index,
                );
            }
            (best_value_scalar, best_index_scalar)
        };
    if lane_count == 16 {
        best_in_col_f32x16(simd, data, reduce)
    } else if lane_count == 8 {
        best_in_col_f32x8(simd, data, reduce)
    } else if lane_count == 4 {
        best_in_col_f32x4(simd, data, reduce)
    } else {
        let (best_values, best_indices, _, _) = best_in_col_f32_generic(simd, cast(0_u32), data);
        (cast(best_values), cast(best_indices))
    }
}

#[inline(always)]
fn update_and_best_in_col_f32<S: Simd>(
    simd: S,
    dst: &mut [f32],
    lhs: &[f32],
    rhs: f32,
) -> (f32, u32) {
    let lane_count = core::mem::size_of::<S::u32s>() / core::mem::size_of::<u32>();
    let reduce =
        |mut best_value_scalar, mut best_index_scalar, best_value: &[f32], best_indices: &[u32]| {
            for (data, index) in best_value.iter().copied().zip(best_indices.iter().copied()) {
                (best_value_scalar, best_index_scalar) = best_f32(
                    pulp::Scalar::new(),
                    best_value_scalar,
                    best_index_scalar,
                    data,
                    index,
                );
            }
            (best_value_scalar, best_index_scalar)
        };
    if lane_count == 16 {
        update_and_best_in_col_f32x16(simd, dst, lhs, rhs, reduce)
    } else if lane_count == 8 {
        update_and_best_in_col_f32x8(simd, dst, lhs, rhs, reduce)
    } else if lane_count == 4 {
        update_and_best_in_col_f32x4(simd, dst, lhs, rhs, reduce)
    } else {
        let (best_values, best_indices, _, _) =
            update_and_best_in_col_f32_generic(simd, cast(0_u32), dst, lhs, rhs);
        (cast(best_values), cast(best_indices))
    }
}

#[inline(always)]
fn best_in_matrix_f64(matrix: MatRef<'_, f64>) -> (usize, usize, f64) {
    struct BestInMat<'a>(MatRef<'a, f64>);
    impl<'a> pulp::WithSimd for BestInMat<'a> {
        type Output = (usize, usize, f64);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let matrix = self.0;
            fancy_debug_assert!(matrix.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row = 0;
            let mut best_col = 0;
            let mut best_value = 0.0;

            for j in 0..n {
                let ptr = matrix.col(j).as_ptr();
                let col = unsafe { core::slice::from_raw_parts(ptr, m) };
                let (best_value_in_col, best_index_in_col) = best_in_col_f64(simd, col);
                if best_value_in_col > best_value {
                    best_value = best_value_in_col;
                    best_row = best_index_in_col as usize;
                    best_col = j;
                }
            }

            (best_row, best_col, best_value)
        }
    }

    pulp::Arch::new().dispatch(BestInMat(matrix))
}

#[inline(always)]
fn update_and_best_in_matrix_f64(
    matrix: MatMut<'_, f64>,
    lhs: ColRef<'_, f64>,
    rhs: RowRef<'_, f64>,
) -> (usize, usize, f64) {
    struct UpdateAndBestInMat<'a>(MatMut<'a, f64>, ColRef<'a, f64>, RowRef<'a, f64>);
    impl<'a> pulp::WithSimd for UpdateAndBestInMat<'a> {
        type Output = (usize, usize, f64);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let UpdateAndBestInMat(mut matrix, lhs, rhs) = self;
            fancy_assert!(matrix.row_stride() == 1);
            fancy_assert!(lhs.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row = 0;
            let mut best_col = 0;
            let mut best_value = 0.0;

            let lhs = unsafe { core::slice::from_raw_parts(lhs.as_ptr(), m) };

            for j in 0..n {
                let rhs = -*rhs.get(j);

                let ptr = matrix.rb_mut().col(j).as_ptr();
                let dst = unsafe { core::slice::from_raw_parts_mut(ptr, m) };

                let (best_value_in_col, best_index_in_col) =
                    update_and_best_in_col_f64(simd, dst, lhs, rhs);
                if best_value_in_col > best_value {
                    best_value = best_value_in_col;
                    best_row = best_index_in_col as usize;
                    best_col = j;
                }
            }

            (best_row, best_col, best_value)
        }
    }

    pulp::Arch::new().dispatch(UpdateAndBestInMat(matrix, lhs, rhs))
}

#[inline(always)]
fn best_in_matrix_f32(matrix: MatRef<'_, f32>) -> (usize, usize, f32) {
    struct BestInMat<'a>(MatRef<'a, f32>);
    impl<'a> pulp::WithSimd for BestInMat<'a> {
        type Output = (usize, usize, f32);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let matrix = self.0;
            fancy_debug_assert!(matrix.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row = 0;
            let mut best_col = 0;
            let mut best_value = 0.0;

            for j in 0..n {
                let ptr = matrix.col(j).as_ptr();
                let col = unsafe { core::slice::from_raw_parts(ptr, m) };
                let (best_value_in_col, best_index_in_col) = best_in_col_f32(simd, col);
                if best_value_in_col > best_value {
                    best_value = best_value_in_col;
                    best_row = best_index_in_col as usize;
                    best_col = j;
                }
            }

            (best_row, best_col, best_value)
        }
    }

    pulp::Arch::new().dispatch(BestInMat(matrix))
}

#[inline(always)]
fn update_and_best_in_matrix_f32(
    matrix: MatMut<'_, f32>,
    lhs: ColRef<'_, f32>,
    rhs: RowRef<'_, f32>,
) -> (usize, usize, f32) {
    struct UpdateAndBestInMat<'a>(MatMut<'a, f32>, ColRef<'a, f32>, RowRef<'a, f32>);
    impl<'a> pulp::WithSimd for UpdateAndBestInMat<'a> {
        type Output = (usize, usize, f32);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let UpdateAndBestInMat(mut matrix, lhs, rhs) = self;
            fancy_debug_assert!(matrix.row_stride() == 1);
            fancy_debug_assert!(lhs.row_stride() == 1);

            let m = matrix.nrows();
            let n = matrix.ncols();
            let mut best_row = 0;
            let mut best_col = 0;
            let mut best_value = 0.0;

            let lhs = unsafe { core::slice::from_raw_parts(lhs.as_ptr(), m) };

            for j in 0..n {
                let rhs = -*rhs.get(j);

                let ptr = matrix.rb_mut().col(j).as_ptr();
                let dst = unsafe { core::slice::from_raw_parts_mut(ptr, m) };

                let (best_value_in_col, best_index_in_col) =
                    update_and_best_in_col_f32(simd, dst, lhs, rhs);
                if best_value_in_col > best_value {
                    best_value = best_value_in_col;
                    best_row = best_index_in_col as usize;
                    best_col = j;
                }
            }

            (best_row, best_col, best_value)
        }
    }

    pulp::Arch::new().dispatch(UpdateAndBestInMat(matrix, lhs, rhs))
}

#[inline]
fn best_in_matrix<T: ComplexField>(matrix: MatRef<'_, T>) -> (usize, usize, T::Real) {
    // let is_f32 = core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>();
    let is_f64 = core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>();
    let is_f32 = core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>();

    let is_col_major = matrix.row_stride() == 1;

    if is_col_major && is_f64 {
        coerce(best_in_matrix_f64(coerce_mat_ref(matrix)))
    } else if is_col_major && is_f32 {
        coerce(best_in_matrix_f32(coerce_mat_ref(matrix)))
    } else {
        let m = matrix.nrows();
        let n = matrix.ncols();

        let mut max = T::Real::zero();
        let mut max_row = 0;
        let mut max_col = 0;

        for j in 0..n {
            for i in 0..m {
                let abs = (*matrix.get(i, j)).score();
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
fn rank_one_update_and_best_in_matrix<T: ComplexField>(
    mut dst: MatMut<'_, T>,
    lhs: ColRef<'_, T>,
    rhs: RowRef<'_, T>,
) -> (usize, usize, T::Real) {
    let is_f64 = core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>();
    let is_f32 = core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>();

    let is_col_major = dst.row_stride() == 1 && lhs.row_stride() == 1;

    if is_f64 && is_col_major {
        coerce(update_and_best_in_matrix_f64(
            coerce_mat_mut(dst),
            coerce_col_ref(lhs),
            coerce_row_ref(rhs),
        ))
    } else if is_f32 && is_col_major {
        coerce(update_and_best_in_matrix_f32(
            coerce_mat_mut(dst),
            coerce_col_ref(lhs),
            coerce_row_ref(rhs),
        ))
    } else {
        matmul(
            dst.rb_mut(),
            Conj::No,
            lhs.as_2d(),
            Conj::No,
            rhs.as_2d(),
            Conj::No,
            Some(T::one()),
            -T::one(),
            Parallelism::None,
        );
        best_in_matrix(dst.rb())
    }
}

#[inline]
fn lu_in_place_unblocked<T: ComplexField>(
    mut matrix: MatMut<'_, T>,
    row_transpositions: &mut [usize],
    col_transpositions: &mut [usize],
    parallelism: Parallelism,
    transposed: bool,
    disable_parallelism: fn(usize, usize) -> bool,
) -> usize {
    let m = matrix.nrows();
    let n = matrix.ncols();

    fancy_debug_assert!(row_transpositions.len() == m);
    fancy_debug_assert!(col_transpositions.len() == n);

    if n == 0 || m == 0 {
        return 0;
    }

    let size = m.min(n);

    let mut n_transpositions = 0;

    let (mut max_row, mut max_col, _) = best_in_matrix(matrix.rb());

    for k in 0..size {
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

        let inv = matrix.rb().get(k, k).inv();
        if !transposed {
            for i in k + 1..m {
                let elem = matrix.rb_mut().get(i, k);
                *elem = *elem * inv;
            }
        } else {
            for i in k + 1..n {
                let elem = matrix.rb_mut().get(k, i);
                *elem = *elem * inv;
            }
        }

        if k + 1 == size {
            break;
        }

        let (_, top_right, bottom_left, bottom_right) = matrix.rb_mut().split_at(k + 1, k + 1);

        let parallelism = if disable_parallelism(m - k, n - k) {
            Parallelism::None
        } else {
            parallelism
        };

        match parallelism {
            Parallelism::None => {
                (max_row, max_col, _) = rank_one_update_and_best_in_matrix(
                    bottom_right,
                    bottom_left.col(k).rb(),
                    top_right.row(k).rb(),
                );
            }
            Parallelism::Rayon(n_threads) => {
                use rayon::prelude::*;
                let n_threads = if n_threads > 0 {
                    n_threads
                } else {
                    rayon::current_num_threads()
                };

                let mut biggest = vec![(0_usize, 0_usize, T::Real::zero()); n_threads];

                let lhs = bottom_left.col(k).into_const();
                let rhs = top_right.row(k).into_const();

                bottom_right
                    .into_par_col_chunks(n_threads)
                    .zip(biggest.par_iter_mut())
                    .for_each(|((col_start, matrix), biggest)| {
                        let rhs = rhs.subcols(col_start, matrix.ncols());
                        *biggest = rank_one_update_and_best_in_matrix(matrix, lhs, rhs);
                        biggest.1 += col_start;
                    });

                max_row = 0;
                max_col = 0;
                let mut biggest_value = T::Real::zero();
                for (row, col, value) in biggest {
                    if value > biggest_value {
                        max_row = row;
                        max_col = col;
                        biggest_value = value;
                    }
                }
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
pub fn lu_in_place_req<T: 'static>(
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
/// # Output
///
/// - The number of transpositions that constitute the permutation,
/// - a structure representing the permutation $P$.
/// - a structure representing the permutation $Q$.
///
/// # Panics
///
/// - Panics if the length of the row permutation slices is not equal to the number of rows of the
///   matrix
/// - Panics if the length of the column permutation slices is not equal to the number of columns of
///   the matrix
/// - Panics if the provided memory in `stack` is insufficient.
pub fn lu_in_place<'out, T: ComplexField>(
    matrix: MatMut<'_, T>,
    row_perm: &'out mut [usize],
    row_perm_inv: &'out mut [usize],
    col_perm: &'out mut [usize],
    col_perm_inv: &'out mut [usize],
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: FullPivLuComputeParams,
) -> (usize, PermutationMut<'out>, PermutationMut<'out>) {
    let disable_parallelism = params
        .disable_parallelism
        .unwrap_or(default_disable_parallelism);

    let _ = parallelism;
    let m = matrix.nrows();
    let n = matrix.ncols();
    fancy_assert!(row_perm.len() == m);
    fancy_assert!(row_perm_inv.len() == m);
    fancy_assert!(col_perm.len() == n);
    fancy_assert!(col_perm_inv.len() == n);

    let (mut row_transpositions, stack) = stack.make_with(m, |i| i);
    let (mut col_transpositions, _) = stack.make_with(n, |i| i);

    let n_transpositions = if matrix.row_stride().abs() < matrix.col_stride().abs() {
        lu_in_place_unblocked(
            matrix,
            &mut row_transpositions,
            &mut col_transpositions,
            parallelism,
            false,
            disable_parallelism,
        )
    } else {
        lu_in_place_unblocked(
            matrix.transpose(),
            &mut col_transpositions,
            &mut row_transpositions,
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
    use faer_core::{permutation::PermutationRef, Mat};
    use rand::random;

    use crate::full_pivoting::reconstruct;

    use super::*;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req))
        };
    }

    fn reconstruct_matrix<T: ComplexField>(
        lu_factors: MatRef<'_, T>,
        row_perm: PermutationRef<'_>,
        col_perm: PermutationRef<'_>,
    ) -> Mat<T> {
        let m = lu_factors.nrows();
        let n = lu_factors.ncols();
        let mut dst = Mat::zeros(m, n);
        reconstruct::reconstruct_to(
            dst.as_mut(),
            lu_factors,
            row_perm,
            col_perm,
            Parallelism::Rayon(0),
            make_stack!(reconstruct::reconstruct_to_req::<T>(m, n, Parallelism::Rayon(0)).unwrap()),
        );
        dst
    }

    fn compute_lu_col_major_generic<T: ComplexField>(random: fn() -> T, epsilon: T::Real) {
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
            let random_mat = Mat::with_dims(|_i, _j| random(), m, n);
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
                    )
                    .unwrap()),
                    Default::default(),
                );
                let reconstructed = reconstruct_matrix(mat.as_ref(), row_perm.rb(), col_perm.rb());

                for i in 0..m {
                    for j in 0..n {
                        fancy_assert!((mat_orig[(i, j)] - reconstructed[(i, j)]).abs() < epsilon);
                    }
                }
            }
        }
    }

    fn compute_lu_row_major_generic<T: ComplexField>(random: fn() -> T, epsilon: T::Real) {
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
            let random_mat = Mat::with_dims(|_i, _j| random(), n, m);
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
                    )
                    .unwrap()),
                    Default::default(),
                );
                let reconstructed = reconstruct_matrix(mat.rb(), row_perm.rb(), col_perm.rb());
                println!("target:{mat_orig:5.3?}");
                println!("actual:{reconstructed:5.3?}");

                for i in 0..m {
                    for j in 0..n {
                        fancy_assert!((mat_orig[(i, j)] - reconstructed[(i, j)]).abs() < epsilon);
                    }
                }
            }
        }
    }

    #[test]
    fn test_compute_lu_row_major() {
        compute_lu_col_major_generic::<f64>(random, 1e-6);
        compute_lu_col_major_generic::<f32>(random, 1e-2);
        compute_lu_row_major_generic::<f64>(random, 1e-6);
        compute_lu_row_major_generic::<f32>(random, 1e-2);
    }
}
