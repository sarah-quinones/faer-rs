use core::{
    any::TypeId,
    mem::transmute_copy,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    mul::dot, permutation::swap_cols, ColMut, ColRef, ComplexField, MatMut, Parallelism,
};
use pulp::{as_arrays, as_arrays_mut, Simd};
use reborrow::*;

#[inline]
fn coerce<T: 'static, U: 'static>(t: T) -> U {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
    let no_drop = core::mem::MaybeUninit::new(t);
    unsafe { transmute_copy(&no_drop) }
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

    for (a, b) in a_remv.iter_mut().zip(b_remv.iter()) {
        *a = simd.f64s_mul_adde(vk, *b, *a);
        acc0 = simd.f64s_mul_adde(*a, *a, acc0);
    }

    acc0 = simd.f64s_add(acc0, acc1);
    acc2 = simd.f64s_add(acc2, acc3);
    acc4 = simd.f64s_add(acc4, acc5);
    acc6 = simd.f64s_add(acc6, acc7);

    acc0 = simd.f64s_add(acc0, acc2);
    acc4 = simd.f64s_add(acc4, acc6);

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
            coerce(update_and_norm2_f64(
                simd,
                unsafe { from_raw_parts_mut(a.as_ptr() as _, a_len) },
                unsafe { from_raw_parts(b.as_ptr() as _, b_len) },
                coerce(k),
            ))
        } else {
            update_and_norm2_generic(
                simd,
                unsafe { from_raw_parts_mut(a.as_ptr(), a_len) },
                unsafe { from_raw_parts(b.as_ptr(), b_len) },
                k,
            )
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
fn qr_in_place_colmajor<S: Simd, T: ComplexField>(
    simd: S,
    mut matrix: MatMut<'_, T>,
    mut householder_coeffs: ColMut<'_, T>,
    col_transpositions: &mut [usize],
    parallelism: Parallelism,
    disable_parallelism: fn(usize, usize) -> bool,
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
        let col_value = norm2(simd, matrix.rb().col(j));
        if col_value > biggest_col_value {
            biggest_col_value = col_value;
            biggest_col_idx = j;
        }
    }

    for k in 0..size {
        let mut matrix_right = matrix.rb_mut().submatrix(0, k, m, n - k);

        col_transpositions[k] = k + biggest_col_idx;
        if biggest_col_idx > 0 {
            n_transpositions += 1;
            swap_cols(matrix_right.rb_mut(), 0, biggest_col_idx);
        }

        let mut matrix = matrix.rb_mut().submatrix(k, k, m - k, n - k);
        let m = matrix.nrows();
        let n = matrix.ncols();

        let (_, _, first_col, last_cols) = matrix.rb_mut().split_at(0, 1);
        let first_col = first_col.col(0);

        let (mut first_head, mut first_tail) = first_col.split_at(1);
        let tail_squared_norm = norm2(simd, first_tail.rb());
        let (tau, beta) = faer_core::householder::make_householder_in_place(
            first_tail.rb_mut(),
            first_head[0],
            tail_squared_norm,
        );
        first_head[0] = beta;
        unsafe { *householder_coeffs.rb_mut().ptr_in_bounds_at(k) = tau };

        let first_tail = first_tail.rb();

        if n == 0 {
            return n_transpositions;
        }

        let extra_parallelism = if disable_parallelism(m, n) {
            Parallelism::None
        } else {
            parallelism
        };

        match extra_parallelism {
            Parallelism::Rayon(n_threads) => {
                use rayon::prelude::*;
                let n_threads = if n_threads > 0 {
                    n_threads
                } else {
                    rayon::current_num_threads()
                };

                let mut biggest_col = vec![(T::Real::zero(), 0_usize); n_threads];

                last_cols
                    .into_par_col_chunks(n_threads)
                    .zip(biggest_col.par_iter_mut())
                    .for_each(
                        |((col_start, matrix), (biggest_col_value, biggest_col_idx))| {
                            process_cols(
                                simd,
                                matrix,
                                col_start,
                                first_tail,
                                tau,
                                biggest_col_value,
                                biggest_col_idx,
                            );
                        },
                    );

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

                process_cols(
                    simd,
                    last_cols,
                    0,
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

fn process_cols<S: Simd, T: ComplexField>(
    simd: S,
    mut matrix: MatMut<'_, T>,
    offset: usize,
    first_tail: ColRef<'_, T>,
    tau: T,
    biggest_col_value: &mut T::Real,
    biggest_col_idx: &mut usize,
) {
    simd.vectorize(|| {
        for j in 0..matrix.ncols() {
            let (col_head, col_tail) = matrix.rb_mut().col(j).split_at(1);
            let col_head = col_head.get(0);

            let dot = *col_head + dot(simd, first_tail, col_tail.rb());
            let k = -tau * dot;
            *col_head = *col_head + k;

            let col_value = update_and_norm2(simd, col_tail, first_tail, k);
            if col_value > *biggest_col_value {
                *biggest_col_value = col_value;
                *biggest_col_idx = j + offset;
            }
        }
    });
}

fn default_disable_parallelism(m: usize, n: usize) -> bool {
    let prod = m * n;
    prod < 192 * 256
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct ColPivQrComputeParams {
    pub disable_parallelism: Option<fn(nrows: usize, ncols: usize) -> bool>,
}

impl ColPivQrComputeParams {
    fn normalize(self) -> fn(usize, usize) -> bool {
        self.disable_parallelism
            .unwrap_or(default_disable_parallelism)
    }
}

pub fn qr_in_place_req<T: 'static>(
    nrows: usize,
    ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = nrows;
    let _ = ncols;
    let _ = parallelism;
    Ok(StackReq::default())
}

pub fn qr_in_place<T: ComplexField>(
    matrix: MatMut<'_, T>,
    householder_coeffs: ColMut<'_, T>,
    col_transpositions: &mut [usize],
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: ColPivQrComputeParams,
) -> usize {
    let _ = &stack;
    let disable_parallelism = params.normalize();

    fancy_assert!(matrix.row_stride() == 1);
    struct QrInPlaceColMajor<'a, T> {
        matrix: MatMut<'a, T>,
        householder_coeffs: ColMut<'a, T>,
        col_transpositions: &'a mut [usize],
        parallelism: Parallelism,
        disable_parallelism: fn(usize, usize) -> bool,
    }

    impl<'a, T: ComplexField> pulp::WithSimd for QrInPlaceColMajor<'a, T> {
        type Output = usize;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            qr_in_place_colmajor(
                simd,
                self.matrix,
                self.householder_coeffs,
                self.col_transpositions,
                self.parallelism,
                self.disable_parallelism,
            )
        }
    }

    pulp::Arch::new().dispatch(QrInPlaceColMajor {
        matrix,
        householder_coeffs,
        col_transpositions,
        parallelism,
        disable_parallelism,
    })
}

#[cfg(test)]
mod tests {

    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::{DynStack, GlobalMemBuffer, StackReq};
    use faer_core::{
        c64, householder::apply_householder_on_the_left, mul::matmul, zip::Diag, Conj, Mat, MatRef,
    };
    use rand::random;

    macro_rules! placeholder_stack {
        () => {
            DynStack::new(&mut GlobalMemBuffer::new(StackReq::new::<T>(1024 * 1024)))
        };
    }

    type T = c64;

    fn reconstruct_factors<T: ComplexField>(
        qr_factors: MatRef<'_, T>,
        householder: ColRef<'_, T>,
    ) -> (Mat<T>, Mat<T>) {
        let m = qr_factors.nrows();
        let n = qr_factors.ncols();
        let size = m.min(n);

        let mut q = Mat::zeros(m, m);
        let mut r = Mat::zeros(m, n);

        r.as_mut()
            .cwise()
            .zip(qr_factors)
            .for_each_triangular_upper(Diag::Include, |a, b| *a = *b);

        q.as_mut().diagonal().cwise().for_each(|a| *a = T::one());

        for k in (0..size).rev() {
            let tau = householder[k];
            let essential = qr_factors.col(k).split_at(k + 1).1;
            apply_householder_on_the_left(
                q.as_mut().submatrix(k, k, m - k, m - k),
                essential,
                tau,
                placeholder_stack!(),
            );
        }

        (q, r)
    }

    #[test]
    fn test_qr_f64() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            for (m, n) in [(2, 2), (2, 4), (4, 2), (4, 4), (63, 63), (1024, 1024)] {
                let mut mat = Mat::<f64>::with_dims(|_, _| random(), m, n);
                let mat_orig = mat.clone();
                let size = m.min(n);
                let mut householder = Mat::zeros(size, 1);
                let mut transpositions = vec![0; size];
                let mut perm = (0..n).collect::<Vec<_>>();

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut().col(0),
                    &mut transpositions,
                    parallelism,
                    placeholder_stack!(),
                    Default::default(),
                );

                for k in 0..size {
                    perm.swap(k, transpositions[k]);
                }

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref().col(0));
                let mut qr = Mat::zeros(m, n);
                matmul(
                    qr.as_mut(),
                    Conj::No,
                    q.as_ref(),
                    Conj::No,
                    r.as_ref(),
                    Conj::No,
                    None,
                    1.0,
                    Parallelism::Rayon(8),
                );

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(qr[(i, j)], mat_orig[(i, perm[j])]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_qr_c64() {
        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            for (m, n) in [(2, 2), (2, 4), (4, 2), (4, 4), (63, 63)] {
                let mut mat = Mat::<c64>::with_dims(|_, _| c64::new(random(), random()), m, n);
                let mat_orig = mat.clone();
                let size = m.min(n);
                let mut householder = Mat::zeros(size, 1);
                let mut transpositions = vec![0; size];
                let mut perm = (0..n).collect::<Vec<_>>();

                qr_in_place(
                    mat.as_mut(),
                    householder.as_mut().col(0),
                    &mut transpositions,
                    parallelism,
                    placeholder_stack!(),
                    Default::default(),
                );

                for k in 0..size {
                    perm.swap(k, transpositions[k]);
                }

                let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref().col(0));
                let mut qr = Mat::zeros(m, n);
                let mut qhq = Mat::zeros(m, m);
                matmul(
                    qr.as_mut(),
                    Conj::No,
                    q.as_ref(),
                    Conj::No,
                    r.as_ref(),
                    Conj::No,
                    None,
                    c64::one(),
                    Parallelism::Rayon(8),
                );

                matmul(
                    qhq.as_mut(),
                    Conj::No,
                    q.as_ref().transpose(),
                    Conj::Yes,
                    q.as_ref(),
                    Conj::No,
                    None,
                    c64::one(),
                    Parallelism::Rayon(8),
                );

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(qr[(i, j)], mat_orig[(i, perm[j])]);
                    }
                }
            }
        }
    }
}
