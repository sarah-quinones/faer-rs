//! Matrix multiplication module.

use core::any::TypeId;

use crate::{ColRef, ComplexField, Conj, MatMut, MatRef, Parallelism};
use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use gemm::gemm;
use pulp::{as_arrays, Simd};
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

    for (a, b) in a_remv.iter().zip(b_remv.iter()) {
        acc0 = simd.f64s_mul_adde(*a, *b, acc0);
    }

    acc0 = simd.f64s_add(acc0, acc1);
    acc2 = simd.f64s_add(acc2, acc3);
    acc4 = simd.f64s_add(acc4, acc5);
    acc6 = simd.f64s_add(acc6, acc7);

    acc0 = simd.f64s_add(acc0, acc2);
    acc4 = simd.f64s_add(acc4, acc6);

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

#[inline]
fn coerce<T: 'static, U: 'static>(t: T) -> U {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
    let no_drop = core::mem::MaybeUninit::new(t);
    unsafe { core::mem::transmute_copy(&no_drop) }
}

// a^* b
#[doc(hidden)]
#[inline(always)]
pub fn dot<S: Simd, T: ComplexField>(simd: S, a: ColRef<'_, T>, b: ColRef<'_, T>) -> T {
    let colmajor = a.row_stride() == 1 && b.row_stride() == 1;
    let id = TypeId::of::<T>();
    if colmajor {
        let a_len = a.nrows();
        let b_len = b.nrows();

        if id == TypeId::of::<f64>() {
            coerce(dot_f64(
                simd,
                unsafe { core::slice::from_raw_parts(a.as_ptr() as _, a_len) },
                unsafe { core::slice::from_raw_parts(b.as_ptr() as _, b_len) },
            ))
        } else {
            dot_generic::<S, T>(
                simd,
                unsafe { core::slice::from_raw_parts(a.as_ptr(), a_len) },
                unsafe { core::slice::from_raw_parts(b.as_ptr(), b_len) },
            )
        }
    } else {
        let mut acc = T::zero();
        for (a, b) in a.into_iter().zip(b.into_iter()) {
            acc = acc + (*a).conj() * *b;
        }
        acc
    }
}

/// Same as [`matmul`], except that panics become undefined behavior.
unsafe fn gemm_wrapper_unchecked<T: ComplexField>(
    acc: MatMut<'_, T>,
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    alpha: Option<T>,
    beta: T,
    conj_acc: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    parallelism: Parallelism,
) {
    fancy_debug_assert!(acc.nrows() == lhs.nrows());
    fancy_debug_assert!(acc.ncols() == rhs.ncols());
    fancy_debug_assert!(lhs.ncols() == rhs.nrows());

    let m = acc.nrows();
    let n = acc.ncols();
    let k = lhs.ncols();

    let acc_col_stride = acc.col_stride();
    let acc_row_stride = acc.row_stride();

    // SAFETY:
    // * matching operand/destination dimensions.
    // * strides were verified during creation of matrix views.
    match alpha {
        Some(alpha) => gemm(
            m,
            n,
            k,
            acc.as_ptr(),
            acc_col_stride,
            acc_row_stride,
            true,
            lhs.as_ptr(),
            lhs.col_stride(),
            lhs.row_stride(),
            rhs.as_ptr(),
            rhs.col_stride(),
            rhs.row_stride(),
            alpha,
            beta,
            conj_acc,
            conj_lhs,
            conj_rhs,
            match parallelism {
                Parallelism::None => ::gemm::Parallelism::None,
                Parallelism::Rayon(n_threads) => ::gemm::Parallelism::Rayon(n_threads),
            },
        ),
        None => gemm(
            m,
            n,
            k,
            acc.as_ptr(),
            acc_col_stride,
            acc_row_stride,
            false,
            lhs.as_ptr(),
            lhs.col_stride(),
            lhs.row_stride(),
            rhs.as_ptr(),
            rhs.col_stride(),
            rhs.row_stride(),
            T::zero(),
            beta,
            conj_acc,
            conj_lhs,
            conj_rhs,
            match parallelism {
                Parallelism::None => ::gemm::Parallelism::None,
                Parallelism::Rayon(n_threads) => ::gemm::Parallelism::Rayon(n_threads),
            },
        ),
    }
}

/// # Safety
///
/// Same as [`matmul`], except that panics become undefined behavior.
#[inline]
pub unsafe fn matmul_unchecked<T: ComplexField>(
    acc: MatMut<'_, T>,
    conj_acc: Conj,
    lhs: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatRef<'_, T>,
    conj_rhs: Conj,
    alpha: Option<T>,
    beta: T,
    parallelism: Parallelism,
) {
    gemm_wrapper_unchecked(
        acc,
        lhs,
        rhs,
        alpha,
        beta,
        conj_acc == Conj::Yes,
        conj_lhs == Conj::Yes,
        conj_rhs == Conj::Yes,
        parallelism,
    )
}

/// Computes the matrix product `[alpha * Op_acc(acc)] + beta * Op_lhs(lhs) * Op_rhs(rhs)` and
/// stores the result in `acc`.
///
/// If `alpha` is not provided, he preexisting values in `acc` are not read so it is allowed to be
/// a view over uninitialized values if `T: Copy`.
///
/// `Op_acc` is the identity if `conj_acc` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
/// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
/// `Op_rhs` is the identity if `conj_rhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
///
/// # Panics
///
/// Panics if the matrix dimensions are not compatible for matrix multiplication.  
/// i.e.  
///  - `acc.nrows() == lhs.nrows()`
///  - `acc.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
///
/// # Example
///
/// ```
/// use faer_core::{mat, mul::matmul, Conj, Mat, Parallelism};
///
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
///
/// let mut acc = Mat::<f64>::zeros(2, 2);
/// let target = mat![
///     [
///         2.5 * (lhs[(0, 0)] * rhs[(0, 0)] + lhs[(0, 1)] * rhs[(1, 0)]),
///         2.5 * (lhs[(0, 0)] * rhs[(0, 1)] + lhs[(0, 1)] * rhs[(1, 1)]),
///     ],
///     [
///         2.5 * (lhs[(1, 0)] * rhs[(0, 0)] + lhs[(1, 1)] * rhs[(1, 0)]),
///         2.5 * (lhs[(1, 0)] * rhs[(0, 1)] + lhs[(1, 1)] * rhs[(1, 1)]),
///     ],
/// ];
///
/// matmul(
///     acc.as_mut(),
///     Conj::No,
///     lhs.as_ref(),
///     Conj::No,
///     rhs.as_ref(),
///     Conj::No,
///     None,
///     2.5,
///     Parallelism::None,
/// );
///
/// acc.as_ref()
///     .cwise()
///     .zip(target.as_ref())
///     .for_each(|acc, target| assert!((acc - target).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn matmul<T: ComplexField>(
    acc: MatMut<'_, T>,
    conj_acc: Conj,
    lhs: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatRef<'_, T>,
    conj_rhs: Conj,
    alpha: Option<T>,
    beta: T,
    parallelism: Parallelism,
) {
    fancy_assert!(acc.nrows() == lhs.nrows());
    fancy_assert!(acc.ncols() == rhs.ncols());
    fancy_assert!(lhs.ncols() == rhs.nrows());
    unsafe {
        matmul_unchecked(
            acc,
            conj_acc,
            lhs,
            conj_lhs,
            rhs,
            conj_rhs,
            alpha,
            beta,
            parallelism,
        )
    }
}

/// Triangular matrix multiplication module, where some of the operands are treated as triangular
/// matrices.
pub mod triangular {
    use core::mem::MaybeUninit;

    use super::*;
    use crate::{
        join_raw,
        zip::{ColUninit, Diag, MatUninit},
    };

    #[repr(u8)]
    #[derive(Copy, Clone, Debug)]
    pub(crate) enum DiagonalKind {
        Zero,
        Unit,
        Generic,
    }

    unsafe fn copy_lower<T: ComplexField>(
        mut dst: MatMut<'_, T>,
        src: MatRef<'_, T>,
        src_diag: DiagonalKind,
    ) {
        let n = dst.nrows();
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());
        fancy_debug_assert!(n == src.nrows());
        fancy_debug_assert!(n == src.ncols());

        let strict = match src_diag {
            DiagonalKind::Zero => {
                for j in 0..n {
                    *dst.rb_mut().ptr_in_bounds_at(j, j) = T::zero();
                }
                true
            }
            DiagonalKind::Unit => {
                for j in 0..n {
                    *dst.rb_mut().ptr_in_bounds_at(j, j) = T::one();
                }
                true
            }
            DiagonalKind::Generic => false,
        };

        MatUninit(dst.rb_mut())
            .cwise()
            .for_each_triangular_upper(Diag::Skip, |dst| {
                *dst = T::zero();
            });
        MatUninit(dst)
            .cwise()
            .zip_unchecked(src)
            .for_each_triangular_lower(
                if strict { Diag::Skip } else { Diag::Include },
                |dst, src| {
                    *dst = *src;
                },
            );
    }

    unsafe fn accum_lower<T: ComplexField>(
        dst: MatMut<'_, T>,
        src: MatRef<'_, T>,
        skip_diag: bool,
        alpha: Option<T>,
    ) {
        let n = dst.nrows();
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());
        fancy_debug_assert!(n == src.nrows());
        fancy_debug_assert!(n == src.ncols());

        match alpha {
            Some(alpha) => {
                dst.cwise().zip_unchecked(src).for_each_triangular_lower(
                    if skip_diag { Diag::Skip } else { Diag::Include },
                    |dst, src| {
                        *dst = alpha * *dst + *src;
                    },
                );
            }
            None => {
                MatUninit(dst)
                    .cwise()
                    .zip_unchecked(src)
                    .for_each_triangular_lower(
                        if skip_diag { Diag::Skip } else { Diag::Include },
                        |dst, src| {
                            *dst = *src;
                        },
                    );
            }
        }
    }

    #[inline]
    unsafe fn copy_upper<T: ComplexField>(
        dst: MatMut<'_, T>,
        src: MatRef<'_, T>,
        src_diag: DiagonalKind,
    ) {
        copy_lower(dst.transpose(), src.transpose(), src_diag)
    }

    #[inline]
    unsafe fn mul<T: ComplexField>(
        dst: MatMut<'_, T>,
        lhs: MatRef<'_, T>,
        rhs: MatRef<'_, T>,
        alpha: Option<T>,
        beta: T,
        conj_dst: Conj,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        super::matmul_unchecked(
            dst,
            conj_dst,
            lhs,
            conj_lhs,
            rhs,
            conj_rhs,
            alpha,
            beta,
            parallelism,
        );
    }

    unsafe fn mat_x_lower_into_lower_impl_unchecked<T: ComplexField>(
        dst: MatMut<'_, T>,
        skip_diag: bool,
        lhs: MatRef<'_, T>,
        rhs: MatRef<'_, T>,
        rhs_diag: DiagonalKind,
        alpha: Option<T>,
        beta: T,
        conj_dst: Conj,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        let n = dst.nrows();
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());
        fancy_debug_assert!(n == lhs.nrows());
        fancy_debug_assert!(n == lhs.ncols());
        fancy_debug_assert!(n == rhs.nrows());
        fancy_debug_assert!(n == rhs.ncols());

        if n <= 16 {
            let mut dst_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
            let mut temp_dst =
                MatMut::from_raw_parts(dst_buffer.as_mut_ptr() as _, n, n, 1, n as isize);
            let mut rhs_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
            let mut temp_rhs =
                MatMut::from_raw_parts(rhs_buffer.as_mut_ptr() as _, n, n, 1, n as isize);
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);
            mul(
                temp_dst.rb_mut(),
                lhs,
                temp_rhs.into_const(),
                None,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            accum_lower(dst, temp_dst.into_const(), skip_diag, alpha);
        } else {
            let bs = n / 2;

            let (mut dst_top_left, _, mut dst_bot_left, dst_bot_right) =
                dst.split_at_unchecked(bs, bs);
            let (lhs_top_left, lhs_top_right, lhs_bot_left, lhs_bot_right) =
                lhs.split_at_unchecked(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at_unchecked(bs, bs);

            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | mat × mat => mat |   1
            // lhs_bot_right × rhs_bot_right => dst_bot_right | mat × low => low |   X
            //
            // lhs_top_left  × rhs_top_left  => dst_top_left  | mat × low => low |   X
            // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
            // lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2

            mul(
                dst_bot_left.rb_mut(),
                lhs_bot_right,
                rhs_bot_left,
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_lower_into_lower_impl_unchecked(
                dst_bot_right,
                skip_diag,
                lhs_bot_right,
                rhs_bot_right,
                rhs_diag,
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );

            mat_x_lower_into_lower_impl_unchecked(
                dst_top_left.rb_mut(),
                skip_diag,
                lhs_top_left,
                rhs_top_left,
                rhs_diag,
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_mat_into_lower_impl_unchecked(
                dst_top_left,
                skip_diag,
                lhs_top_right,
                rhs_bot_left,
                Some(T::one()),
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_lower_impl_unchecked(
                dst_bot_left,
                lhs_bot_left,
                rhs_top_left,
                rhs_diag,
                Some(T::one()),
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
        }
    }

    unsafe fn mat_x_lower_impl_unchecked<T: ComplexField>(
        dst: MatMut<'_, T>,
        lhs: MatRef<'_, T>,
        rhs: MatRef<'_, T>,
        rhs_diag: DiagonalKind,
        alpha: Option<T>,
        beta: T,
        conj_dst: Conj,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        let n = rhs.nrows();
        let m = lhs.nrows();
        fancy_debug_assert!(m == lhs.nrows());
        fancy_debug_assert!(n == lhs.ncols());
        fancy_debug_assert!(n == rhs.nrows());
        fancy_debug_assert!(n == rhs.ncols());
        fancy_debug_assert!(m == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());

        if n <= 16 {
            let op = {
                #[inline(never)]
                || {
                    let mut rhs_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
                    let mut temp_rhs =
                        MatMut::from_raw_parts(rhs_buffer.as_mut_ptr() as _, n, n, 1, 16);

                    copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);
                    let temp_rhs = temp_rhs.into_const();

                    mul(
                        dst,
                        lhs,
                        temp_rhs,
                        alpha,
                        beta,
                        conj_dst,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                }
            };

            op();
        } else {
            // split rhs into 3 sections
            // split lhs and dst into 2 sections

            let bs = n / 2;

            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at_unchecked(bs, bs);
            let (lhs_left, lhs_right) = lhs.split_at_col(bs);
            let (mut dst_left, mut dst_right) = dst.split_at_col(bs);

            join_raw(
                |parallelism| {
                    mat_x_lower_impl_unchecked(
                        dst_left.rb_mut(),
                        lhs_left,
                        rhs_top_left,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_dst,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                |parallelism| {
                    mat_x_lower_impl_unchecked(
                        dst_right.rb_mut(),
                        lhs_right,
                        rhs_bot_right,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_dst,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                parallelism,
            );
            mul(
                dst_left,
                lhs_right,
                rhs_bot_left,
                Some(T::one()),
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
        }
    }

    unsafe fn lower_x_lower_into_lower_impl_unchecked<T: ComplexField>(
        dst: MatMut<'_, T>,
        skip_diag: bool,
        lhs: MatRef<'_, T>,
        lhs_diag: DiagonalKind,
        rhs: MatRef<'_, T>,
        rhs_diag: DiagonalKind,
        alpha: Option<T>,
        beta: T,
        conj_dst: Conj,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        let n = dst.nrows();
        fancy_debug_assert!(n == lhs.nrows());
        fancy_debug_assert!(n == lhs.ncols());
        fancy_debug_assert!(n == rhs.nrows());
        fancy_debug_assert!(n == rhs.ncols());
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());

        if n <= 16 {
            let mut dst_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
            let mut temp_dst =
                MatMut::from_raw_parts(dst_buffer.as_mut_ptr() as _, n, n, 1, n as isize);
            let mut lhs_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
            let mut temp_lhs =
                MatMut::from_raw_parts(lhs_buffer.as_mut_ptr() as _, n, n, 1, n as isize);
            let mut rhs_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
            let mut temp_rhs =
                MatMut::from_raw_parts(rhs_buffer.as_mut_ptr() as _, n, n, 1, n as isize);

            copy_lower(temp_lhs.rb_mut(), lhs, lhs_diag);
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

            let temp_lhs = temp_lhs.into_const();
            let temp_rhs = temp_rhs.into_const();
            mul(
                temp_dst.rb_mut(),
                temp_lhs,
                temp_rhs,
                None,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            accum_lower(dst, temp_dst.into_const(), skip_diag, alpha);
        } else {
            let bs = n / 2;

            let (dst_top_left, _, mut dst_bot_left, dst_bot_right) = dst.split_at_unchecked(bs, bs);
            let (lhs_top_left, _, lhs_bot_left, lhs_bot_right) = lhs.split_at_unchecked(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at_unchecked(bs, bs);

            // lhs_top_left  × rhs_top_left  => dst_top_left  | low × low => low |   X
            // lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2
            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | low × mat => mat | 1/2
            // lhs_bot_right × rhs_bot_right => dst_bot_right | low × low => low |   X

            lower_x_lower_into_lower_impl_unchecked(
                dst_top_left,
                skip_diag,
                lhs_top_left,
                lhs_diag,
                rhs_top_left,
                rhs_diag,
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_lower_impl_unchecked(
                dst_bot_left.rb_mut(),
                lhs_bot_left,
                rhs_top_left,
                rhs_diag,
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_lower_impl_unchecked(
                dst_bot_left.reverse_rows_and_cols().transpose(),
                rhs_bot_left.reverse_rows_and_cols().transpose(),
                lhs_bot_right.reverse_rows_and_cols().transpose(),
                lhs_diag,
                Some(T::one()),
                beta,
                conj_dst,
                conj_rhs,
                conj_lhs,
                parallelism,
            );
            lower_x_lower_into_lower_impl_unchecked(
                dst_bot_right,
                skip_diag,
                lhs_bot_right,
                lhs_diag,
                rhs_bot_right,
                rhs_diag,
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            )
        }
    }

    unsafe fn upper_x_lower_impl_unchecked<T: ComplexField>(
        dst: MatMut<'_, T>,
        lhs: MatRef<'_, T>,
        lhs_diag: DiagonalKind,
        rhs: MatRef<'_, T>,
        rhs_diag: DiagonalKind,
        alpha: Option<T>,
        beta: T,
        conj_dst: Conj,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        let n = dst.nrows();
        fancy_debug_assert!(n == lhs.nrows());
        fancy_debug_assert!(n == lhs.ncols());
        fancy_debug_assert!(n == rhs.nrows());
        fancy_debug_assert!(n == rhs.ncols());
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());

        if n <= 16 {
            let mut lhs_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
            let mut temp_lhs =
                MatMut::from_raw_parts(lhs_buffer.as_mut_ptr() as _, n, n, 1, n as isize);
            let mut rhs_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
            let mut temp_rhs =
                MatMut::from_raw_parts(rhs_buffer.as_mut_ptr() as _, n, n, 1, n as isize);

            copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

            let temp_lhs = temp_lhs.into_const();
            let temp_rhs = temp_rhs.into_const();
            mul(
                dst,
                temp_lhs,
                temp_rhs,
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
        } else {
            let bs = n / 2;

            let (mut dst_top_left, dst_top_right, dst_bot_left, dst_bot_right) =
                dst.split_at_unchecked(bs, bs);
            let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_at_unchecked(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at_unchecked(bs, bs);

            // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => mat |   1
            // lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => mat |   X
            //
            // lhs_top_right × rhs_bot_right => dst_top_right | mat × low => mat | 1/2
            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
            // lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => mat |   X

            join_raw(
                |_| {
                    mul(
                        dst_top_left.rb_mut(),
                        lhs_top_right,
                        rhs_bot_left,
                        alpha,
                        beta,
                        conj_dst,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                    upper_x_lower_impl_unchecked(
                        dst_top_left,
                        lhs_top_left,
                        lhs_diag,
                        rhs_top_left,
                        rhs_diag,
                        Some(T::one()),
                        beta,
                        conj_dst,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                |_| {
                    join_raw(
                        |_| {
                            mat_x_lower_impl_unchecked(
                                dst_top_right,
                                lhs_top_right,
                                rhs_bot_right,
                                rhs_diag,
                                alpha,
                                beta,
                                conj_dst,
                                conj_lhs,
                                conj_rhs,
                                parallelism,
                            )
                        },
                        |_| {
                            mat_x_lower_impl_unchecked(
                                dst_bot_left.transpose(),
                                rhs_bot_left.transpose(),
                                lhs_bot_right.transpose(),
                                lhs_diag,
                                alpha,
                                beta,
                                conj_dst,
                                conj_rhs,
                                conj_lhs,
                                parallelism,
                            )
                        },
                        parallelism,
                    );

                    upper_x_lower_impl_unchecked(
                        dst_bot_right,
                        lhs_bot_right,
                        lhs_diag,
                        rhs_bot_right,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_dst,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                parallelism,
            );
        }
    }

    unsafe fn upper_x_lower_into_lower_impl_unchecked<T: ComplexField>(
        mut dst: MatMut<'_, T>,
        skip_diag: bool,
        lhs: MatRef<'_, T>,
        lhs_diag: DiagonalKind,
        rhs: MatRef<'_, T>,
        rhs_diag: DiagonalKind,
        alpha: Option<T>,
        beta: T,
        conj_dst: Conj,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        let n = dst.nrows();
        fancy_debug_assert!(n == lhs.nrows());
        fancy_debug_assert!(n == lhs.ncols());
        fancy_debug_assert!(n == rhs.nrows());
        fancy_debug_assert!(n == rhs.ncols());
        fancy_debug_assert!(n == dst.nrows());
        fancy_debug_assert!(n == dst.ncols());

        if n <= 16 {
            let mut dst_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
            let mut temp_dst =
                MatMut::from_raw_parts(dst_buffer.as_mut_ptr() as _, n, n, 1, n as isize);
            let mut lhs_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
            let mut temp_lhs =
                MatMut::from_raw_parts(lhs_buffer.as_mut_ptr() as _, n, n, 1, n as isize);
            let mut rhs_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
            let mut temp_rhs =
                MatMut::from_raw_parts(rhs_buffer.as_mut_ptr() as _, n, n, 1, n as isize);

            copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

            let temp_lhs = temp_lhs.into_const();
            let temp_rhs = temp_rhs.into_const();
            mul(
                temp_dst.rb_mut(),
                temp_lhs,
                temp_rhs,
                None,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );

            accum_lower(dst.rb_mut(), temp_dst.into_const(), skip_diag, alpha);
        } else {
            let bs = n / 2;

            let (mut dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_at_unchecked(bs, bs);
            let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_at_unchecked(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at_unchecked(bs, bs);

            // lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => low |   X
            // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
            //
            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
            // lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => low |   X

            join_raw(
                |_| {
                    mat_x_mat_into_lower_impl_unchecked(
                        dst_top_left.rb_mut(),
                        skip_diag,
                        lhs_top_right,
                        rhs_bot_left,
                        alpha,
                        beta,
                        conj_dst,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                    upper_x_lower_into_lower_impl_unchecked(
                        dst_top_left,
                        skip_diag,
                        lhs_top_left,
                        lhs_diag,
                        rhs_top_left,
                        rhs_diag,
                        Some(T::one()),
                        beta,
                        conj_dst,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                |_| {
                    mat_x_lower_impl_unchecked(
                        dst_bot_left.transpose(),
                        rhs_bot_left.transpose(),
                        lhs_bot_right.transpose(),
                        lhs_diag,
                        alpha,
                        beta,
                        conj_dst,
                        conj_rhs,
                        conj_lhs,
                        parallelism,
                    );
                    upper_x_lower_into_lower_impl_unchecked(
                        dst_bot_right,
                        skip_diag,
                        lhs_bot_right,
                        lhs_diag,
                        rhs_bot_right,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_dst,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                parallelism,
            );
        }
    }

    unsafe fn mat_x_mat_into_lower_impl_unchecked<T: ComplexField>(
        dst: MatMut<'_, T>,
        skip_diag: bool,
        lhs: MatRef<'_, T>,
        rhs: MatRef<'_, T>,
        alpha: Option<T>,
        beta: T,
        conj_dst: Conj,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        fancy_debug_assert!(dst.nrows() == dst.ncols());
        fancy_debug_assert!(dst.nrows() == lhs.nrows());
        fancy_debug_assert!(dst.ncols() == rhs.ncols());
        fancy_debug_assert!(lhs.ncols() == rhs.nrows());

        let n = dst.nrows();
        let k = lhs.ncols();

        let join_parallelism = if n * n * k < 128 * 128 * 128 {
            Parallelism::None
        } else {
            parallelism
        };

        if n <= 16 {
            let mut dst_buffer = [MaybeUninit::<T>::uninit(); 16 * 16];
            let mut temp_dst =
                MatMut::from_raw_parts(dst_buffer.as_mut_ptr() as _, n, n, 1, n as isize);
            mul(
                temp_dst.rb_mut(),
                lhs,
                rhs,
                None,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            accum_lower(dst, temp_dst.rb(), skip_diag, alpha)
        } else {
            let bs = n / 2;
            let (dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_at_unchecked(bs, bs);
            let (lhs_top, lhs_bot) = lhs.split_at_row(bs);
            let (rhs_left, rhs_right) = rhs.split_at_col(bs);

            join_raw(
                |_| {
                    mul(
                        dst_bot_left,
                        lhs_bot,
                        rhs_left,
                        alpha,
                        beta,
                        conj_dst,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                |_| {
                    join_raw(
                        |_| {
                            mat_x_mat_into_lower_impl_unchecked(
                                dst_top_left,
                                skip_diag,
                                lhs_top,
                                rhs_left,
                                alpha,
                                beta,
                                conj_dst,
                                conj_lhs,
                                conj_rhs,
                                parallelism,
                            )
                        },
                        |_| {
                            mat_x_mat_into_lower_impl_unchecked(
                                dst_bot_right,
                                skip_diag,
                                lhs_bot,
                                rhs_right,
                                alpha,
                                beta,
                                conj_dst,
                                conj_lhs,
                                conj_rhs,
                                parallelism,
                            )
                        },
                        join_parallelism,
                    )
                },
                join_parallelism,
            );
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub enum BlockStructure {
        Rectangular,
        TriangularLower,
        StrictTriangularLower,
        UnitTriangularLower,
        TriangularUpper,
        StrictTriangularUpper,
        UnitTriangularUpper,
    }

    impl BlockStructure {
        #[inline]
        pub fn is_dense(self) -> bool {
            matches!(self, BlockStructure::Rectangular)
        }

        #[inline]
        pub fn is_lower(self) -> bool {
            use BlockStructure::*;
            matches!(
                self,
                TriangularLower | StrictTriangularLower | UnitTriangularLower
            )
        }

        #[inline]
        pub fn is_upper(self) -> bool {
            use BlockStructure::*;
            matches!(
                self,
                TriangularUpper | StrictTriangularUpper | UnitTriangularUpper
            )
        }

        #[inline]
        pub fn transpose(self) -> Self {
            use BlockStructure::*;
            match self {
                Rectangular => Rectangular,
                TriangularLower => TriangularUpper,
                StrictTriangularLower => StrictTriangularUpper,
                UnitTriangularLower => UnitTriangularUpper,
                TriangularUpper => TriangularLower,
                StrictTriangularUpper => StrictTriangularLower,
                UnitTriangularUpper => UnitTriangularLower,
            }
        }

        #[inline]
        pub(crate) fn diag_kind(self) -> DiagonalKind {
            use BlockStructure::*;
            match self {
                Rectangular | TriangularLower | TriangularUpper => DiagonalKind::Generic,
                StrictTriangularLower | StrictTriangularUpper => DiagonalKind::Zero,
                UnitTriangularLower | UnitTriangularUpper => DiagonalKind::Unit,
            }
        }
    }

    /// Computes the matrix product `[alpha * Op_acc(acc)] + beta * Op_lhs(lhs) * Op_rhs(rhs)` and
    /// stores the result in `acc`.
    ///
    /// The left hand side and right hand side may be interpreted as triangular depending on the
    /// given corresponding matrix structure.  
    ///
    /// For the destination matrix, the result is:
    /// - fully computed if the structure is rectangular,
    /// - only the triangular half (including the diagonal) is computed if the structure is
    /// triangular,
    /// - only the strict triangular half (excluding the diagonal) is computed if the structure is
    /// strictly triangular or unit triangular.
    ///
    /// If `alpha` is not provided, he preexisting values in `acc` are not read so it is allowed to
    /// be a view over uninitialized values if `T: Copy`.
    ///
    /// `Op_acc` is the identity if `conj_acc` is `Conj::No`, and the conjugation operation if it is
    /// `Conj::Yes`.  
    /// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
    /// `Conj::Yes`.  
    /// `Op_rhs` is the identity if `conj_rhs` is `Conj::No`, and the conjugation operation if it is
    /// `Conj::Yes`.  
    ///
    /// # Panics
    ///
    /// Panics if the matrix dimensions are not compatible for matrix multiplication.  
    /// i.e.  
    ///  - `acc.nrows() == lhs.nrows()`
    ///  - `acc.ncols() == rhs.ncols()`
    ///  - `lhs.ncols() == rhs.nrows()`
    ///
    ///  Additionally, matrices that are marked as triangular must be square, i.e., they must have
    ///  the same number of rows and columns.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::{
    ///     mat,
    ///     mul::triangular::{matmul, BlockStructure},
    ///     Conj, Mat, Parallelism,
    /// };
    ///
    /// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
    /// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
    ///
    /// let mut acc = Mat::<f64>::zeros(2, 2);
    /// let target = mat![
    ///     [
    ///         2.5 * (lhs[(0, 0)] * rhs[(0, 0)] + lhs[(0, 1)] * rhs[(1, 0)]),
    ///         0.0,
    ///     ],
    ///     [
    ///         2.5 * (lhs[(1, 0)] * rhs[(0, 0)] + lhs[(1, 1)] * rhs[(1, 0)]),
    ///         2.5 * (lhs[(1, 0)] * rhs[(0, 1)] + lhs[(1, 1)] * rhs[(1, 1)]),
    ///     ],
    /// ];
    ///
    /// matmul(
    ///     acc.as_mut(),
    ///     BlockStructure::TriangularLower,
    ///     Conj::No,
    ///     lhs.as_ref(),
    ///     BlockStructure::Rectangular,
    ///     Conj::No,
    ///     rhs.as_ref(),
    ///     BlockStructure::Rectangular,
    ///     Conj::No,
    ///     None,
    ///     2.5,
    ///     Parallelism::None,
    /// );
    ///
    /// acc.as_ref()
    ///     .cwise()
    ///     .zip(target.as_ref())
    ///     .for_each(|acc, target| assert!((acc - target).abs() < 1e-10));
    /// ```
    #[track_caller]
    #[inline]
    pub fn matmul<T: ComplexField>(
        acc: MatMut<'_, T>,
        acc_structure: BlockStructure,
        conj_acc: Conj,
        lhs: MatRef<'_, T>,
        lhs_structure: BlockStructure,
        conj_lhs: Conj,
        rhs: MatRef<'_, T>,
        rhs_structure: BlockStructure,
        conj_rhs: Conj,
        alpha: Option<T>,
        beta: T,
        parallelism: Parallelism,
    ) {
        fancy_assert!(acc.nrows() == lhs.nrows());
        fancy_assert!(acc.ncols() == rhs.ncols());
        fancy_assert!(lhs.ncols() == rhs.nrows());

        if !acc_structure.is_dense() {
            fancy_assert!(acc.nrows() == acc.ncols());
        }
        if !lhs_structure.is_dense() {
            fancy_assert!(lhs.nrows() == lhs.ncols());
        }
        if !rhs_structure.is_dense() {
            fancy_assert!(rhs.nrows() == rhs.ncols());
        }

        unsafe {
            matmul_unchecked(
                acc,
                acc_structure,
                conj_acc,
                lhs,
                lhs_structure,
                conj_lhs,
                rhs,
                rhs_structure,
                conj_rhs,
                alpha,
                beta,
                parallelism,
            )
        }
    }

    /// # Safety
    ///
    /// Same as [`matmul`], except that panics become undefined behavior.
    #[inline]
    pub unsafe fn matmul_unchecked<T: ComplexField>(
        acc: MatMut<'_, T>,
        acc_structure: BlockStructure,
        conj_acc: Conj,
        lhs: MatRef<'_, T>,
        lhs_structure: BlockStructure,
        conj_lhs: Conj,
        rhs: MatRef<'_, T>,
        rhs_structure: BlockStructure,
        conj_rhs: Conj,
        alpha: Option<T>,
        beta: T,
        parallelism: Parallelism,
    ) {
        fancy_debug_assert!(acc.nrows() == lhs.nrows());
        fancy_debug_assert!(acc.ncols() == rhs.ncols());
        fancy_debug_assert!(lhs.ncols() == rhs.nrows());

        if !acc_structure.is_dense() {
            fancy_debug_assert!(acc.nrows() == acc.ncols());
        }
        if !lhs_structure.is_dense() {
            fancy_debug_assert!(lhs.nrows() == lhs.ncols());
        }
        if !rhs_structure.is_dense() {
            fancy_debug_assert!(rhs.nrows() == rhs.ncols());
        }

        let mut acc = acc;
        let mut lhs = lhs;
        let mut rhs = rhs;

        let mut acc_structure = acc_structure;
        let mut lhs_structure = lhs_structure;
        let mut rhs_structure = rhs_structure;

        let mut conj_lhs = conj_lhs;
        let mut conj_rhs = conj_rhs;

        // if either the lhs or the rhs is triangular
        if rhs_structure.is_lower() {
            // do nothing
            false
        } else if rhs_structure.is_upper() {
            // invert acc, lhs and rhs
            acc = acc.reverse_rows_and_cols();
            lhs = lhs.reverse_rows_and_cols();
            rhs = rhs.reverse_rows_and_cols();
            acc_structure = acc_structure.transpose();
            lhs_structure = lhs_structure.transpose();
            rhs_structure = rhs_structure.transpose();
            false
        } else if lhs_structure.is_lower() {
            // invert and transpose
            acc = acc.reverse_rows_and_cols().transpose();
            (lhs, rhs) = (
                rhs.reverse_rows_and_cols().transpose(),
                lhs.reverse_rows_and_cols().transpose(),
            );
            (conj_lhs, conj_rhs) = (conj_rhs, conj_lhs);
            (lhs_structure, rhs_structure) = (rhs_structure, lhs_structure);
            true
        } else if lhs_structure.is_upper() {
            // transpose
            acc_structure = acc_structure.transpose();
            acc = acc.transpose();
            (lhs, rhs) = (rhs.transpose(), lhs.transpose());
            (conj_lhs, conj_rhs) = (conj_rhs, conj_lhs);
            (lhs_structure, rhs_structure) = (rhs_structure.transpose(), lhs_structure.transpose());
            true
        } else {
            // do nothing
            false
        };

        let clear_upper = |acc: MatMut<'_, T>, skip_diag: bool| match alpha {
            Some(alpha) => acc.cwise().for_each_triangular_upper(
                if skip_diag { Diag::Skip } else { Diag::Include },
                |acc| *acc = alpha * *acc,
            ),

            None => MatUninit(acc).cwise().for_each_triangular_upper(
                if skip_diag { Diag::Skip } else { Diag::Include },
                |acc| *acc = T::zero(),
            ),
        };

        let skip_diag = matches!(
            acc_structure,
            BlockStructure::StrictTriangularLower
                | BlockStructure::StrictTriangularUpper
                | BlockStructure::UnitTriangularLower
                | BlockStructure::UnitTriangularUpper
        );
        let lhs_diag = lhs_structure.diag_kind();
        let rhs_diag = rhs_structure.diag_kind();

        if acc_structure.is_dense() {
            if lhs_structure.is_dense() && rhs_structure.is_dense() {
                mul(
                    acc,
                    lhs,
                    rhs,
                    alpha,
                    beta,
                    conj_acc,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                );
            } else {
                fancy_debug_assert!(rhs_structure.is_lower());

                if lhs_structure.is_dense() {
                    mat_x_lower_impl_unchecked(
                        acc,
                        lhs,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_acc,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                } else if lhs_structure.is_lower() {
                    clear_upper(acc.rb_mut(), true);
                    lower_x_lower_into_lower_impl_unchecked(
                        acc,
                        false,
                        lhs,
                        lhs_diag,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_acc,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                } else {
                    fancy_debug_assert!(lhs_structure.is_upper());
                    upper_x_lower_impl_unchecked(
                        acc,
                        lhs,
                        lhs_diag,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_acc,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                }
            }
        } else if acc_structure.is_lower() {
            if lhs_structure.is_dense() && rhs_structure.is_dense() {
                mat_x_mat_into_lower_impl_unchecked(
                    acc,
                    skip_diag,
                    lhs,
                    rhs,
                    alpha,
                    beta,
                    conj_acc,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            } else {
                fancy_debug_assert!(rhs_structure.is_lower());
                if lhs_structure.is_dense() {
                    mat_x_lower_into_lower_impl_unchecked(
                        acc,
                        skip_diag,
                        lhs,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_acc,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                } else if lhs_structure.is_lower() {
                    lower_x_lower_into_lower_impl_unchecked(
                        acc,
                        skip_diag,
                        lhs,
                        lhs_diag,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_acc,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                } else {
                    upper_x_lower_into_lower_impl_unchecked(
                        acc,
                        skip_diag,
                        lhs,
                        lhs_diag,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_acc,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                }
            }
        } else if lhs_structure.is_dense() && rhs_structure.is_dense() {
            mat_x_mat_into_lower_impl_unchecked(
                acc.transpose(),
                skip_diag,
                rhs.transpose(),
                lhs.transpose(),
                alpha,
                beta,
                conj_acc,
                conj_rhs,
                conj_lhs,
                parallelism,
            )
        } else {
            fancy_debug_assert!(rhs_structure.is_lower());
            if lhs_structure.is_dense() {
                // lower part of lhs does not contribute to result
                upper_x_lower_into_lower_impl_unchecked(
                    acc.transpose(),
                    skip_diag,
                    rhs.transpose(),
                    rhs_diag,
                    lhs.transpose(),
                    lhs_diag,
                    alpha,
                    beta,
                    conj_acc,
                    conj_rhs,
                    conj_lhs,
                    parallelism,
                )
            } else if lhs_structure.is_lower() {
                if !skip_diag {
                    match alpha {
                        Some(alpha) => {
                            acc.rb_mut()
                                .diagonal_unchecked()
                                .cwise()
                                .zip(lhs.diagonal_unchecked())
                                .zip(rhs.diagonal_unchecked())
                                .for_each(|acc, lhs, rhs| {
                                    *acc = alpha * *acc + beta * (*lhs * *rhs)
                                });
                        }
                        None => {
                            ColUninit(acc.rb_mut().diagonal_unchecked())
                                .cwise()
                                .zip(lhs.diagonal_unchecked())
                                .zip(rhs.diagonal_unchecked())
                                .for_each(|acc, lhs, rhs| *acc = beta * (*lhs * *rhs));
                        }
                    }
                }
                clear_upper(acc.rb_mut(), true);
            } else {
                fancy_debug_assert!(lhs_structure.is_upper());
                upper_x_lower_into_lower_impl_unchecked(
                    acc.transpose(),
                    skip_diag,
                    rhs.transpose(),
                    rhs_diag,
                    lhs.transpose(),
                    lhs_diag,
                    alpha,
                    beta,
                    conj_acc,
                    conj_rhs,
                    conj_lhs,
                    parallelism,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use rand::random;

    use super::{
        triangular::{BlockStructure, DiagonalKind},
        *,
    };
    use crate::{mat, Mat};

    #[test]
    fn rectangular() {
        let lhs = mat![[1.0, 2.0], [3.0, 4.0]];
        let rhs = mat![[5.0], [7.0]];
        let mut dst = mat![[0.0], [0.0]];

        super::matmul(
            dst.as_mut(),
            Conj::No,
            lhs.as_ref(),
            Conj::No,
            rhs.as_ref(),
            Conj::No,
            None,
            2.0,
            Parallelism::None,
        );

        fancy_assert!(dst[(0, 0)] == 38.0);
        fancy_assert!(dst[(1, 0)] == 86.0);
    }

    #[test]
    fn rectangular_accum() {
        let lhs = mat![[1.0, 2.0], [3.0, 4.0]];
        let rhs = mat![[5.0], [7.0]];
        let mut dst = mat![[4.0], [3.0]];

        super::matmul(
            dst.as_mut(),
            Conj::No,
            lhs.as_ref(),
            Conj::No,
            rhs.as_ref(),
            Conj::No,
            Some(-2.0),
            2.0,
            Parallelism::None,
        );

        fancy_assert!(dst[(0, 0)] == 30.0);
        fancy_assert!(dst[(1, 0)] == 80.0);
    }

    fn generate_structured_matrix(
        is_dst: bool,
        nrows: usize,
        ncols: usize,
        structure: BlockStructure,
    ) -> Mat<f64> {
        let mut mat = Mat::new();
        mat.resize_with(|_, _| random(), nrows, ncols);

        if !is_dst {
            let kind = structure.diag_kind();
            if structure.is_lower() {
                for j in 0..ncols {
                    for i in 0..j {
                        mat[(i, j)] = 0.0;
                    }
                }
            } else if structure.is_upper() {
                for j in 0..ncols {
                    for i in j + 1..nrows {
                        mat[(i, j)] = 0.0;
                    }
                }
            }

            match kind {
                triangular::DiagonalKind::Zero => {
                    for i in 0..nrows {
                        mat[(i, i)] = 0.0;
                    }
                }
                triangular::DiagonalKind::Unit => {
                    for i in 0..nrows {
                        mat[(i, i)] = 1.0;
                    }
                }
                triangular::DiagonalKind::Generic => (),
            }
        }
        mat
    }

    fn run_test_problem(
        m: usize,
        n: usize,
        k: usize,
        dst_structure: BlockStructure,
        lhs_structure: BlockStructure,
        rhs_structure: BlockStructure,
    ) {
        let mut dst = generate_structured_matrix(true, m, n, dst_structure);
        let mut dst_target = dst.clone();
        let dst_orig = dst.clone();
        let lhs = generate_structured_matrix(false, m, k, lhs_structure);
        let rhs = generate_structured_matrix(false, k, n, rhs_structure);

        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            triangular::matmul(
                dst.as_mut(),
                dst_structure,
                Conj::No,
                lhs.as_ref(),
                lhs_structure,
                Conj::No,
                rhs.as_ref(),
                rhs_structure,
                Conj::No,
                None,
                2.5,
                parallelism,
            );

            matmul(
                dst_target.as_mut(),
                Conj::No,
                lhs.as_ref(),
                Conj::No,
                rhs.as_ref(),
                Conj::No,
                None,
                2.5,
                parallelism,
            );

            if dst_structure.is_dense() {
                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(dst[(i, j)], dst_target[(i, j)]);
                    }
                }
            } else if dst_structure.is_lower() {
                for j in 0..n {
                    if matches!(dst_structure.diag_kind(), DiagonalKind::Generic) {
                        for i in 0..j {
                            assert_eq!(dst[(i, j)], dst_orig[(i, j)]);
                        }
                        for i in j..n {
                            assert_approx_eq!(dst[(i, j)], dst_target[(i, j)]);
                        }
                    } else {
                        for i in 0..=j {
                            assert_eq!(dst[(i, j)], dst_orig[(i, j)]);
                        }
                        for i in j + 1..n {
                            assert_approx_eq!(dst[(i, j)], dst_target[(i, j)]);
                        }
                    }
                }
            } else {
                for j in 0..n {
                    if matches!(dst_structure.diag_kind(), DiagonalKind::Generic) {
                        for i in 0..=j {
                            assert_approx_eq!(dst[(i, j)], dst_target[(i, j)]);
                        }
                        for i in j + 1..n {
                            assert_eq!(dst[(i, j)], dst_orig[(i, j)]);
                        }
                    } else {
                        for i in 0..j {
                            assert_approx_eq!(dst[(i, j)], dst_target[(i, j)]);
                        }
                        for i in j..n {
                            assert_eq!(dst[(i, j)], dst_orig[(i, j)]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn triangular() {
        use BlockStructure::*;
        let structures = [
            Rectangular,
            TriangularLower,
            TriangularUpper,
            StrictTriangularLower,
            StrictTriangularUpper,
            UnitTriangularLower,
            UnitTriangularUpper,
        ];

        for dst in structures {
            for lhs in structures {
                for rhs in structures {
                    for _ in 0..3 {
                        let m = random::<usize>() % 100;
                        let mut n = random::<usize>() % 100;
                        let mut k = random::<usize>() % 100;

                        match (!dst.is_dense(), !lhs.is_dense(), !rhs.is_dense()) {
                            (true, true, _) | (true, _, true) | (_, true, true) => {
                                n = m;
                                k = m;
                            }
                            _ => (),
                        }

                        if !dst.is_dense() {
                            n = m;
                        }

                        if !lhs.is_dense() {
                            k = m;
                        }

                        if !rhs.is_dense() {
                            k = n;
                        }

                        run_test_problem(m, n, k, dst, lhs, rhs);
                    }
                }
            }
        }
    }
}
