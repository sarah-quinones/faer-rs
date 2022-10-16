use core::ops::{Add, Mul, Neg};
use std::mem::transmute_copy;

use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use bytemuck::cast;
use dyn_stack::DynStack;
use faer_core::mul::matmul;
use faer_core::permutation::PermutationIndicesMut;
use faer_core::{MatMut, MatRef};
use num_traits::{Inv, One, Signed, Zero};
use pulp::Simd;
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

    let head = head.chunks_exact(3);

    for data in head.clone() {
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

    for data in head.remainder().iter().copied() {
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
fn best_in_matrix_f64(matrix: MatRef<'_, f64>) -> (usize, usize) {
    struct BestInMat<'a>(MatRef<'a, f64>);
    impl<'a> pulp::WithSimd for BestInMat<'a> {
        type Output = (usize, usize);

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
                unsafe {
                    let ptr = matrix.col_unchecked(j).as_ptr();
                    let col = core::slice::from_raw_parts(ptr, m);
                    let (best_value_in_col, best_index_in_col) = best_in_col_f64(simd, col);
                    if best_value_in_col > best_value {
                        best_value = best_value_in_col;
                        best_row = best_index_in_col as usize;
                        best_col = j;
                    }
                }
            }

            (best_row, best_col)
        }
    }

    pulp::Arch::new().dispatch(BestInMat(matrix))
}

#[inline]
fn best_in_matrix<T>(matrix: MatRef<'_, T>) -> (usize, usize)
where
    T: Zero + Signed + PartialOrd + 'static,
{
    // let is_f32 = core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>();
    let is_f64 = core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>();

    let is_col_major = matrix.row_stride() == 1;
    let is_row_major = matrix.col_stride() == 1;

    if is_col_major && is_f64 {
        best_in_matrix_f64(unsafe { transmute_copy(&matrix) })
    } else if is_row_major && is_f64 {
        best_in_matrix_f64(unsafe { transmute_copy(&matrix.transpose()) })
    } else {
        let m = matrix.nrows();
        let n = matrix.ncols();

        let mut max = T::zero();
        let mut max_row = 0;
        let mut max_col = 0;

        for j in 0..n {
            for i in 0..m {
                let abs = unsafe { matrix.get_unchecked(i, j).abs() };
                if abs > max {
                    max_row = i;
                    max_col = j;
                    max = abs;
                }
            }
        }

        (max_row, max_col)
    }
}

#[inline]
unsafe fn lu_in_place_unblocked<T>(
    mut matrix: MatMut<'_, T>,
    row_transpositions: &mut [usize],
    col_transpositions: &mut [usize],
    n_threads: usize,
    mut stack: DynStack<'_>,
) -> usize
where
    T: Zero + One + Clone + Send + Sync + Signed + PartialOrd + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    let m = matrix.nrows();
    let n = matrix.ncols();

    fancy_debug_assert!(row_transpositions.len() == m);
    fancy_debug_assert!(col_transpositions.len() == n);

    if n == 0 || m == 0 {
        return 0;
    }

    let size = m.min(n);

    let mut n_transpositions = 0;

    for k in 0..size {
        let (max_row, max_col) =
            best_in_matrix(matrix.rb().submatrix_unchecked(k, k, m - k, n - k));
        let (max_row, max_col) = (max_row + k, max_col + k);

        if max_row != k {
            n_transpositions += 1;
            row_transpositions[k] = max_row;

            let (_, top, _, bot) = matrix.rb_mut().split_at_unchecked(k + 1, 0);
            let row_j = top.row_unchecked(k);
            let row_max = bot.row_unchecked(max_row - k - 1);
            swap_cols(row_j.transpose(), row_max.transpose());
        }

        if max_col != k {
            n_transpositions += 1;
            col_transpositions[k] = max_col;

            let (_, _, left, right) = matrix.rb_mut().split_at_unchecked(0, k + 1);
            let col_j = left.col_unchecked(k);
            let col_max = right.col_unchecked(max_col - k - 1);
            swap_cols(col_j, col_max);
        }

        let inv = matrix.rb().get_unchecked(k, k).inv();
        for i in k + 1..m {
            let elem = matrix.rb_mut().get_unchecked(i, k);
            *elem = &*elem * &inv;
        }

        let (_, top_right, bottom_left, bottom_right) =
            matrix.rb_mut().split_at_unchecked(k + 1, k + 1);

        matmul(
            bottom_right,
            bottom_left.rb().col(k).as_2d(),
            top_right.rb().row(k).as_2d(),
            Some(&T::one()),
            &-&T::one(),
            n_threads,
            stack.rb_mut(),
        )
    }

    n_transpositions
}

#[inline]
unsafe fn swap_cols<T>(mut col_j: faer_core::ColMut<T>, mut col_max: faer_core::ColMut<T>) {
    let m = col_j.nrows();
    for k in 0..m {
        core::mem::swap(
            col_j.rb_mut().get_unchecked(k),
            col_max.rb_mut().get_unchecked(k),
        );
    }
}

pub fn lu_in_place<'out, T>(
    matrix: MatMut<'_, T>,
    row_perm: &'out mut [usize],
    row_perm_inv: &'out mut [usize],
    col_perm: &'out mut [usize],
    col_perm_inv: &'out mut [usize],
    n_threads: usize,
    stack: DynStack<'_>,
) -> (
    usize,
    PermutationIndicesMut<'out>,
    PermutationIndicesMut<'out>,
)
where
    T: Zero + One + Clone + Send + Sync + Signed + PartialOrd + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    let m = matrix.nrows();
    let n = matrix.ncols();
    fancy_assert!(row_perm.len() == m);
    fancy_assert!(row_perm_inv.len() == m);
    fancy_assert!(col_perm.len() == n);
    fancy_assert!(col_perm_inv.len() == n);

    let (mut row_transpositions, stack) = stack.make_with(m, |_| 0);
    let (mut col_transpositions, stack) = stack.make_with(m, |_| 0);

    let n_transpositions = unsafe {
        lu_in_place_unblocked(
            matrix,
            &mut row_transpositions,
            &mut col_transpositions,
            n_threads,
            stack,
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
            PermutationIndicesMut::new_unchecked(row_perm, row_perm_inv),
            PermutationIndicesMut::new_unchecked(col_perm, col_perm_inv),
        )
    }
}
