use core::{
    any::TypeId,
    mem::transmute_copy,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use faer_core::{ColMut, ColRef, ComplexField, MatMut, Parallelism};
use pulp::{as_arrays, as_arrays_mut, Simd};
use reborrow::*;

#[inline(always)]
fn dot_f64<S: Simd>(simd: S, a: &[f64], b: &[f64]) -> f64 {
    let mut acc0 = simd.f64s_splat(0.0);
    let mut acc1 = simd.f64s_splat(0.0);
    let mut acc2 = simd.f64s_splat(0.0);
    let mut acc3 = simd.f64s_splat(0.0);
    let mut acc4 = simd.f64s_splat(0.0);
    let mut acc5 = simd.f64s_splat(0.0);
    let mut acc6 = simd.f64s_splat(0.0);
    let mut acc7 = simd.f64s_splat(0.0);

    let (a, a_rem) = S::f64s_as_simd(a);
    let (b, b_rem) = S::f64s_as_simd(b);

    let (a, a_remv) = as_arrays::<8, _>(a);
    let (b, b_remv) = as_arrays::<8, _>(b);

    for (a, b) in a.iter().zip(b.iter()) {
        acc0 = simd.f64s_mul_adde(a[0], b[0], acc0);
        acc1 = simd.f64s_mul_adde(a[1], b[1], acc1);
        acc2 = simd.f64s_mul_adde(a[2], b[2], acc2);
        acc3 = simd.f64s_mul_adde(a[3], b[3], acc3);
        acc4 = simd.f64s_mul_adde(a[4], b[4], acc4);
        acc5 = simd.f64s_mul_adde(a[5], b[5], acc5);
        acc6 = simd.f64s_mul_adde(a[6], b[6], acc6);
        acc7 = simd.f64s_mul_adde(a[7], b[7], acc7);
    }

    acc2 = simd.f64s_add(acc2, acc3);
    acc4 = simd.f64s_add(acc4, acc5);
    acc6 = simd.f64s_add(acc6, acc7);
    acc4 = simd.f64s_add(acc4, acc6);

    for (a, b) in a_remv.iter().zip(b_remv.iter()) {
        acc0 = simd.f64s_mul_adde(*a, *b, acc0);
    }

    acc0 = simd.f64s_add(acc0, acc1);
    acc0 = simd.f64s_add(acc0, acc2);
    acc0 = simd.f64s_add(acc0, acc4);

    let mut acc = simd.f64s_reduce_sum(acc0);

    for (a, b) in a_rem.iter().zip(b_rem.iter()) {
        acc = f64::mul_add(*a, *b, acc);
    }

    acc
}

#[inline(always)]
fn dot_generic<S: Simd, T: ComplexField>(_simd: S, a: &[T], b: &[T]) -> T {
    let mut acc = T::zero();
    for (a, b) in a.iter().zip(b.iter()) {
        acc = acc + (*a).conj() * *b;
    }
    acc
}

// a += k * b
//
// returns ||a||²
#[inline(always)]
fn update_and_norm2_f64<S: Simd>(simd: S, a: &mut [f64], b: &[f64], k: f64) -> f64 {
    let mut acc0 = simd.f64s_splat(0.0);
    let mut acc1 = simd.f64s_splat(0.0);
    let mut acc2 = simd.f64s_splat(0.0);
    let mut acc3 = simd.f64s_splat(0.0);
    let mut acc4 = simd.f64s_splat(0.0);
    let mut acc5 = simd.f64s_splat(0.0);
    let mut acc6 = simd.f64s_splat(0.0);
    let mut acc7 = simd.f64s_splat(0.0);

    let (a, a_rem) = S::f64s_as_mut_simd(a);
    let (b, b_rem) = S::f64s_as_simd(b);

    let (a, a_remv) = as_arrays_mut::<8, _>(a);
    let (b, b_remv) = as_arrays::<8, _>(b);

    let vk = simd.f64s_splat(k);

    for (a, b) in a.iter_mut().zip(b.iter()) {
        a[0] = simd.f64s_mul_adde(vk, b[0], a[0]);
        acc0 = simd.f64s_mul_adde(a[0], a[0], acc0);

        a[1] = simd.f64s_mul_adde(vk, b[1], a[1]);
        acc1 = simd.f64s_mul_adde(a[1], a[1], acc1);

        a[2] = simd.f64s_mul_adde(vk, b[2], a[2]);
        acc2 = simd.f64s_mul_adde(a[2], a[2], acc2);

        a[3] = simd.f64s_mul_adde(vk, b[3], a[3]);
        acc3 = simd.f64s_mul_adde(a[3], a[3], acc3);

        a[4] = simd.f64s_mul_adde(vk, b[4], a[4]);
        acc4 = simd.f64s_mul_adde(a[4], a[4], acc4);

        a[5] = simd.f64s_mul_adde(vk, b[5], a[5]);
        acc5 = simd.f64s_mul_adde(a[5], a[5], acc5);

        a[6] = simd.f64s_mul_adde(vk, b[6], a[6]);
        acc6 = simd.f64s_mul_adde(a[6], a[6], acc6);

        a[7] = simd.f64s_mul_adde(vk, b[7], a[7]);
        acc7 = simd.f64s_mul_adde(a[7], a[7], acc7);
    }

    acc2 = simd.f64s_add(acc2, acc3);
    acc4 = simd.f64s_add(acc4, acc5);
    acc6 = simd.f64s_add(acc6, acc7);
    acc4 = simd.f64s_add(acc4, acc6);

    for (a, b) in a_remv.iter_mut().zip(b_remv.iter()) {
        *a = simd.f64s_mul_adde(vk, *b, *a);
        acc0 = simd.f64s_mul_adde(*a, *a, acc0);
    }

    acc0 = simd.f64s_add(acc0, acc1);
    acc0 = simd.f64s_add(acc0, acc2);
    acc0 = simd.f64s_add(acc0, acc4);

    let mut acc = simd.f64s_reduce_sum(acc0);

    for (a, b) in a_rem.iter_mut().zip(b_rem.iter()) {
        *a = f64::mul_add(k, *b, *a);
        acc = f64::mul_add(*a, *a, acc);
    }

    acc
}

#[inline(always)]
fn update_and_norm2_generic<S: Simd, T: ComplexField>(
    _simd: S,
    a: &mut [T],
    b: &[T],
    k: T,
) -> T::Real {
    let mut acc = T::Real::zero();

    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a = *a + k * *b;
        acc = acc + ((*a).conj() * *a).real();
    }

    acc
}

// a^* b
#[inline(always)]
fn dot<S: Simd, T: ComplexField>(simd: S, a: ColRef<'_, T>, b: ColRef<'_, T>) -> T {
    let colmajor = a.row_stride() == 1 && b.row_stride() == 1;
    let id = TypeId::of::<T>();
    if colmajor {
        let a_len = a.nrows();
        let b_len = b.nrows();

        if id == TypeId::of::<f64>() {
            unsafe {
                transmute_copy(&dot_f64(
                    simd,
                    from_raw_parts(a.as_ptr() as _, a_len),
                    from_raw_parts(b.as_ptr() as _, b_len),
                ))
            }
        } else {
            unsafe {
                dot_generic::<S, T>(
                    simd,
                    from_raw_parts(a.as_ptr(), a_len),
                    from_raw_parts(b.as_ptr(), b_len),
                )
            }
        }
    } else {
        let mut acc = T::zero();
        for (a, b) in a.into_iter().zip(b.into_iter()) {
            acc = acc + (*a).conj() * *b;
        }
        acc
    }
}

#[inline(always)]
fn norm2<S: Simd, T: ComplexField>(simd: S, a: ColRef<'_, T>) -> T::Real {
    dot(simd, a, a).real()
}

#[inline(always)]
fn update_and_norm2<S: Simd, T: ComplexField>(
    simd: S,
    a: ColMut<'_, T>,
    b: ColRef<'_, T>,
    k: T,
) -> T::Real {
    let colmajor = a.row_stride() == 1 && b.row_stride() == 1;
    let id = TypeId::of::<T>();
    if colmajor {
        let a_len = a.nrows();
        let b_len = b.nrows();

        if id == TypeId::of::<f64>() {
            unsafe {
                transmute_copy(&update_and_norm2_f64(
                    simd,
                    from_raw_parts_mut(a.as_ptr() as _, a_len),
                    from_raw_parts(b.as_ptr() as _, b_len),
                    transmute_copy(&k),
                ))
            }
        } else {
            unsafe {
                update_and_norm2_generic(
                    simd,
                    from_raw_parts_mut(a.as_ptr(), a_len),
                    from_raw_parts(b.as_ptr(), b_len),
                    k,
                )
            }
        }
    } else {
        let mut acc = T::Real::zero();

        for (a, b) in a.into_iter().zip(b.into_iter()) {
            *a = *a + k * *b;
            acc = acc + ((*a).conj() * *a).real();
        }

        acc
    }
}

#[inline(always)]
unsafe fn qr_in_place_colmajor<S: Simd, T: ComplexField>(
    simd: S,
    mut matrix: MatMut<'_, T>,
    mut householder_coeffs: ColMut<'_, T>,
    col_transpositions: &mut [usize],
    parallelism: Parallelism,
) -> usize {
    fancy_debug_assert!(matrix.row_stride() == 1);

    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = m.min(n);

    fancy_debug_assert!(householder_coeffs.nrows() == size);

    let mut n_transpositions = 0;

    if size == 0 {
        return n_transpositions;
    }

    let mut biggest_col_idx = 0;
    let mut biggest_col_value = T::Real::zero();
    for j in 0..n {
        let col_value = norm2(simd, matrix.rb().col_unchecked(j));
        if col_value > biggest_col_value {
            biggest_col_value = col_value;
            biggest_col_idx = j;
        }
    }

    for k in 0..size {
        let mut matrix = matrix.rb_mut().submatrix_unchecked(k, k, m - k, n - k);
        let householder_coeffs = householder_coeffs.rb_mut().split_at_unchecked(k).1;
        let m = matrix.nrows();
        let n = matrix.ncols();

        let (_, _, first_col, mut last_cols) = matrix.rb_mut().split_at_unchecked(0, 1);
        let mut first_col = first_col.col_unchecked(0);

        col_transpositions[k] = k + biggest_col_idx;
        if biggest_col_idx > 0 {
            n_transpositions += 1;

            if first_col.row_stride() == 1 {
                core::ptr::swap_nonoverlapping(
                    first_col.rb_mut().as_ptr(),
                    last_cols.rb_mut().ptr_at(0, biggest_col_idx - 1),
                    m,
                );
            } else {
                for i in 0..m {
                    core::mem::swap(
                        &mut *first_col.rb_mut().ptr_at(i),
                        &mut *last_cols.rb_mut().ptr_at(i, biggest_col_idx - 1),
                    );
                }
            }
        }

        let (first_head, mut first_tail) = first_col.split_at_unchecked(1);
        let tail_squared_norm = norm2(simd, first_tail.rb());
        let (tau, beta) = faer_core::householder::make_householder_in_place_unchecked(
            first_tail.rb_mut(),
            *first_head.rb().as_ptr(),
            tail_squared_norm,
        );
        *first_head.as_ptr() = beta;
        *householder_coeffs.ptr_in_bounds_at_unchecked(0) = tau;

        let first_tail = first_tail.rb();

        if n == 0 {
            return n_transpositions;
        }

        let rs = last_cols.row_stride();
        let cs = last_cols.col_stride();

        match parallelism {
            Parallelism::Rayon(n_threads) => {
                use rayon::prelude::*;

                struct Ptr<T>(*mut T);
                unsafe impl<T> Send for Ptr<T> {}
                unsafe impl<T> Sync for Ptr<T> {}
                impl<T> Copy for Ptr<T> {}
                impl<T> Clone for Ptr<T> {
                    fn clone(&self) -> Self {
                        *self
                    }
                }

                let mut biggest_col = vec![(T::Real::zero(), 0_usize); n_threads];

                let base_ptr = Ptr(last_cols.as_ptr());

                let cols_per_thread = (n - 1) / n_threads;
                let rem = (n - 1) % n_threads;

                (0..n_threads)
                    .into_par_iter()
                    .zip(biggest_col.par_iter_mut())
                    .for_each(|(tid, (biggest_col_value, biggest_col_idx))| {
                        let ptr = { base_ptr }.0;
                        let tid_to_col_start = |tid| {
                            if tid < rem {
                                tid * (cols_per_thread + 1)
                            } else {
                                rem * (cols_per_thread + 1) + (tid - rem) * cols_per_thread
                            }
                        };

                        let col_start = tid_to_col_start(tid);
                        let col_end = tid_to_col_start(tid + 1);

                        simd.vectorize(|| {
                            process_cols(
                                simd,
                                col_start,
                                col_end,
                                ptr,
                                rs,
                                cs,
                                m,
                                first_tail,
                                tau,
                                biggest_col_value,
                                biggest_col_idx,
                            );
                        })
                    });

                biggest_col_value = T::Real::zero();
                biggest_col_idx = 0;

                for (col_value, col_idx) in biggest_col {
                    if col_value > biggest_col_value {
                        biggest_col_value = col_value;
                        biggest_col_idx = col_idx;
                    }
                }
            }
            _ => {
                biggest_col_value = T::Real::zero();
                biggest_col_idx = 0;

                let ptr = last_cols.as_ptr();
                let col_start = 0;
                let col_end = n - 1;

                process_cols(
                    simd,
                    col_start,
                    col_end,
                    ptr,
                    rs,
                    cs,
                    m,
                    first_tail,
                    tau,
                    &mut biggest_col_value,
                    &mut biggest_col_idx,
                );
            }
        }
    }

    n_transpositions
}

unsafe fn process_cols<S: Simd, T: ComplexField>(
    simd: S,
    col_start: usize,
    col_end: usize,
    ptr: *mut T,
    rs: isize,
    cs: isize,
    m: usize,
    first_tail: ColRef<'_, T>,
    tau: T,
    biggest_col_value: &mut T::Real,
    biggest_col_idx: &mut usize,
) {
    simd.vectorize(|| {
        for j in col_start..col_end {
            let (col_head, col_tail) =
                ColMut::from_raw_parts(ptr.wrapping_offset(j as isize * cs), m, rs)
                    .split_at_unchecked(1);
            let col_head = col_head.get_unchecked(0);

            let dot = *col_head + dot(simd, col_tail.rb(), first_tail);
            let k = -tau * dot;
            *col_head = *col_head + k;

            let col_value = update_and_norm2(simd, col_tail, first_tail, k);
            if col_value > *biggest_col_value {
                *biggest_col_value = col_value;
                *biggest_col_idx = j;
            }
        }
    });
}

pub fn qr_in_place<T: ComplexField>(
    matrix: MatMut<'_, T>,
    householder_coeffs: ColMut<'_, T>,
    col_transpositions: &mut [usize],
    parallelism: Parallelism,
) -> usize {
    fancy_assert!(matrix.row_stride() == 1);
    struct QrInPlaceColMajor<'a, T> {
        matrix: MatMut<'a, T>,
        householder_coeffs: ColMut<'a, T>,
        col_transpositions: &'a mut [usize],
        parallelism: Parallelism,
    }

    impl<'a, T: ComplexField> pulp::WithSimd for QrInPlaceColMajor<'a, T> {
        type Output = usize;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            unsafe {
                qr_in_place_colmajor(
                    simd,
                    self.matrix,
                    self.householder_coeffs,
                    self.col_transpositions,
                    self.parallelism,
                )
            }
        }
    }

    pulp::Arch::new().dispatch(QrInPlaceColMajor {
        matrix,
        householder_coeffs,
        col_transpositions,
        parallelism,
    })
}
