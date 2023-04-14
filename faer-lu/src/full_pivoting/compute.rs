use assert2::{assert, debug_assert};
use bytemuck::cast;
use coe::Coerce;
use core::slice;
use dyn_stack::{DynStack, StackReq};
use faer_core::{
    c32, c64, for_each_raw,
    mul::matmul,
    par_split_indices, parallelism_degree,
    permutation::{swap_cols, swap_rows, PermutationMut},
    ComplexField, MatMut, MatRef, Parallelism, Ptr,
};
use paste::paste;
use pulp::{cast_lossy, Simd};
use reborrow::*;

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
fn best_c64<S: pulp::Simd>(
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
fn best_c32<S: pulp::Simd>(
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
fn best2d_f64<S: pulp::Simd>(
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
fn best2d_f32<S: pulp::Simd>(
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
                    simd.[<$scalar s_partial_load>](tail, simd.[<$scalar s_splat>](pulp::cast(<$scalar>::zero()))),
                    indices0
                );

                let mut best_value_scalar = 0.0;
                let mut best_index_scalar = 0;
                let mut index = (head.len() * lane_count) as $uint;
                for data in tail.iter().copied() {
                    (best_value_scalar, best_index_scalar) = [<best_ $scalar>](
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
                    let new_dst0 = simd.[<$scalar s_mul_adde>](*lhs0, rhs_v, *dst0);
                    let new_dst1 = simd.[<$scalar s_mul_adde>](*lhs1, rhs_v, *dst1);
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
                    let new_dst = simd.[<$scalar s_mul_adde>](*lhs, rhs_v, *dst);
                    *dst = new_dst;
                    (best_value0, best_indices0) =
                        [<best_ $scalar>](simd, best_value0, best_indices0, new_dst, indices0);
                    indices0 = simd.[<$uint s_add>](indices0, increment1);
                }

                {
                    let new_dst = simd.[<$scalar s_mul_adde>](
                        simd.[<$scalar s_partial_load>](lhs_tail, simd.[<$scalar s_splat>](pulp::cast(<$scalar>::zero()))),
                        rhs_v,
                        simd.[<$scalar s_partial_load>](dst_tail, simd.[<$scalar s_splat>](pulp::cast(<$scalar>::zero()))),
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
best_in_col_simd!(f64, f64, u64);
best_in_col_simd!(f32, f32, u32);
best_in_col_simd!(c64, f64, u64);
best_in_col_simd!(c32, f32, u32);

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
fn best_in_col_f64<S: Simd>(simd: S, data: &[f64]) -> (S::f64s, S::u64s) {
    best_in_col_f64_generic(simd, cast_lossy([0, 1, 2, 3, 4, 5, 6, 7_u64]), data)
}

#[inline(always)]
fn update_and_best_in_col_f64<S: Simd>(
    simd: S,
    dst: &mut [f64],
    lhs: &[f64],
    rhs: f64,
) -> (S::f64s, S::u64s) {
    update_and_best_in_col_f64_generic(
        simd,
        cast_lossy([0, 1, 2, 3, 4, 5, 6, 7_u64]),
        dst,
        lhs,
        rhs,
    )
}

#[inline(always)]
fn best_in_col_f32<S: Simd>(simd: S, data: &[f32]) -> (S::f32s, S::u32s) {
    best_in_col_f32_generic(
        simd,
        cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u32]),
        data,
    )
}

#[inline(always)]
fn update_and_best_in_col_f32<S: Simd>(
    simd: S,
    dst: &mut [f32],
    lhs: &[f32],
    rhs: f32,
) -> (S::f32s, S::u32s) {
    update_and_best_in_col_f32_generic(
        simd,
        cast_lossy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15_u32]),
        dst,
        lhs,
        rhs,
    )
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

fn best_in_matrix_f64(matrix: MatRef<'_, f64>) -> (usize, usize, f64) {
    struct BestInMat<'a>(MatRef<'a, f64>);
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
                let (best_value_in_col, best_index_in_col) = best_in_col_f64(simd, col);
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
    pulp::Arch::new().dispatch(BestInMat(matrix))
}

fn update_and_best_in_matrix_f64(
    matrix: MatMut<'_, f64>,
    lhs: MatRef<'_, f64>,
    rhs: MatRef<'_, f64>,
) -> (usize, usize, f64) {
    struct UpdateAndBestInMat<'a>(MatMut<'a, f64>, MatRef<'a, f64>, MatRef<'a, f64>);
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
                    update_and_best_in_col_f64(simd, dst, lhs, rhs);

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
    pulp::Arch::new().dispatch(UpdateAndBestInMat(matrix, lhs, rhs))
}

fn best_in_matrix_f32(matrix: MatRef<'_, f32>) -> (usize, usize, f32) {
    struct BestInMat<'a>(MatRef<'a, f32>);
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
                let (best_value_in_col, best_index_in_col) = best_in_col_f32(simd, col);
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
    pulp::Arch::new().dispatch(BestInMat(matrix))
}

fn update_and_best_in_matrix_f32(
    matrix: MatMut<'_, f32>,
    lhs: MatRef<'_, f32>,
    rhs: MatRef<'_, f32>,
) -> (usize, usize, f32) {
    struct UpdateAndBestInMat<'a>(MatMut<'a, f32>, MatRef<'a, f32>, MatRef<'a, f32>);
    impl pulp::WithSimd for UpdateAndBestInMat<'_> {
        type Output = (usize, usize, f32);

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let UpdateAndBestInMat(mut matrix, lhs, rhs) = self;
            assert!(matrix.row_stride() == 1);
            assert!(lhs.row_stride() == 1);

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
                    update_and_best_in_col_f32(simd, dst, lhs, rhs);

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
    pulp::Arch::new().dispatch(UpdateAndBestInMat(matrix, lhs, rhs))
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
    pulp::Arch::new().dispatch(BestInMat(matrix))
}

pub fn update_and_best_in_matrix_c64(
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
    pulp::Arch::new().dispatch(UpdateAndBestInMat(matrix, lhs, rhs))
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
    pulp::Arch::new().dispatch(BestInMat(matrix))
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
    pulp::Arch::new().dispatch(UpdateAndBestInMat(matrix, lhs, rhs))
}

#[inline]
fn best_in_matrix<T: ComplexField>(matrix: MatRef<'_, T>) -> (usize, usize, T::Real) {
    let is_f64 = coe::is_same::<f64, T>();
    let is_f32 = coe::is_same::<f32, T>();
    let is_c64 = coe::is_same::<c64, T>();
    let is_c32 = coe::is_same::<c32, T>();

    let is_col_major = matrix.row_stride() == 1;

    if is_col_major && is_f64 {
        coe::coerce_static(best_in_matrix_f64(matrix.coerce()))
    } else if is_col_major && is_f32 {
        coe::coerce_static(best_in_matrix_f32(matrix.coerce()))
    } else if is_col_major && is_c64 {
        coe::coerce_static(best_in_matrix_c64(matrix.coerce()))
    } else if is_col_major && is_c32 {
        coe::coerce_static(best_in_matrix_c32(matrix.coerce()))
    } else {
        let m = matrix.nrows();
        let n = matrix.ncols();

        let mut max = T::Real::zero();
        let mut max_row = 0;
        let mut max_col = 0;

        for j in 0..n {
            for i in 0..m {
                let abs = matrix.read(i, j).score();
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
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
) -> (usize, usize, T::Real) {
    let is_f64 = coe::is_same::<f64, T>();
    let is_f32 = coe::is_same::<f32, T>();
    let is_c64 = coe::is_same::<c64, T>();
    let is_c32 = coe::is_same::<c32, T>();

    let is_col_major = dst.row_stride() == 1 && lhs.row_stride() == 1;

    if is_f64 && is_col_major {
        coe::coerce_static(update_and_best_in_matrix_f64(
            dst.coerce(),
            lhs.coerce(),
            rhs.coerce(),
        ))
    } else if is_f32 && is_col_major {
        coe::coerce_static(update_and_best_in_matrix_f32(
            dst.coerce(),
            lhs.coerce(),
            rhs.coerce(),
        ))
    } else if is_c64 && is_col_major {
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
    } else {
        matmul(
            dst.rb_mut(),
            lhs,
            rhs,
            Some(T::one()),
            T::one().neg(),
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

    debug_assert!(row_transpositions.len() == m);
    debug_assert!(col_transpositions.len() == n);

    if n == 0 || m == 0 {
        return 0;
    }

    let size = <usize as Ord>::min(m, n);

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

        let inv = matrix.read(k, k).inv();
        if !transposed {
            for i in k + 1..m {
                let elem = matrix.read(i, k);
                matrix.write(i, k, elem.mul(&inv));
            }
        } else {
            for i in k + 1..n {
                let elem = matrix.read(k, i);
                matrix.write(k, i, elem.mul(&inv));
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
                (max_row, max_col, _) = rank_one_update_and_best_in_matrix(
                    bottom_right,
                    bottom_left.col(k).rb(),
                    top_right.row(k).rb(),
                );
            }
            _ => {
                let n_threads = parallelism_degree(parallelism);

                let mut biggest = vec![(0_usize, 0_usize, T::Real::zero()); n_threads];

                let lhs = bottom_left.col(k).into_const();
                let rhs = top_right.row(k).into_const();

                {
                    let biggest = Ptr(biggest.as_mut_ptr());

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

                // bottom_right
                //     .into_par_col_chunks(n_threads)
                //     .zip(biggest.par_iter_mut())
                //     .for_each(|((col_start, matrix), biggest)| {
                //         let rhs = rhs.subcols(col_start, matrix.ncols());
                //         *biggest = rank_one_update_and_best_in_matrix(matrix, lhs, rhs);
                //         biggest.1 += col_start;
                //     });

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
    assert!(row_perm.len() == m);
    assert!(row_perm_inv.len() == m);
    assert!(col_perm.len() == n);
    assert!(col_perm_inv.len() == n);

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
    use super::*;
    use crate::full_pivoting::reconstruct;
    use assert2::assert;
    use faer_core::{c32, c64, permutation::PermutationRef, Mat};
    use rand::random;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req.unwrap()))
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
        reconstruct::reconstruct(
            dst.as_mut(),
            lu_factors,
            row_perm,
            col_perm,
            Parallelism::Rayon(0),
            make_stack!(reconstruct::reconstruct_req::<T>(
                m,
                n,
                Parallelism::Rayon(0)
            )),
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
            let random_mat = Mat::with_dims(m, n, |_i, _j| random());
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
                            (mat_orig.read(i, j).sub(&reconstructed.read(i, j))).abs() < epsilon
                        );
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
            let random_mat = Mat::with_dims(n, m, |_i, _j| random());
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
                            (mat_orig.read(i, j).sub(&reconstructed.read(i, j))).abs() < epsilon
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
}
