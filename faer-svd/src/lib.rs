//! The SVD of a matrix $M$ of shape $(m, n)$ is a decomposition into three components $U$, $S$,
//! and $V$, such that:
//!
//! - $U$ has shape $(m, m)$ and is an orthogonal matrix,
//! - $V$ has shape $(n, n)$ and is an orthogonal matrix,
//! - $S$ has shape $(m, n)$ and is zero everywhere except the main diagonal,
//! - and finally:
//!
//! $$M = U S V^H.$$

use assert2::assert as fancy_assert;
use coe::Coerce;
use core::mem::swap;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    householder::{
        apply_block_householder_sequence_on_the_left_in_place, upgrade_householder_factor,
    },
    temp_mat_req, temp_mat_uninit, zip, ColMut, ComplexField, Conj, MatMut, MatRef, Parallelism,
    RealField,
};
use num_complex::Complex;
use reborrow::*;

#[doc(hidden)]
pub mod bidiag;
#[doc(hidden)]
pub mod bidiag_real_svd;
#[doc(hidden)]
pub mod jacobi;

const JACOBI_FALLBACK_THRESHOLD: usize = 4;

/// Indicates whether the singular vectors are fully computed, partially computed, or skipped.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComputeVectors {
    No,
    Thin,
    Full,
}

fn compute_real_svd_small_req<T: 'static>(
    mut m: usize,
    mut n: usize,
    mut compute_u: ComputeVectors,
    mut compute_v: ComputeVectors,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    if n > m {
        swap(&mut m, &mut n);
        swap(&mut compute_u, &mut compute_v);
    }

    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<T>(m, n);

    let qr = temp_mat_req::<T>(m, n)?;
    let householder = temp_mat_req::<T>(householder_blocksize, n)?;
    let r = temp_mat_req::<T>(n, n)?;

    let col_perm = StackReq::try_new::<usize>(n)?;
    let col_perm_inv = col_perm;

    let compute_qr = faer_qr::col_pivoting::compute::qr_in_place_req::<T>(
        m,
        n,
        householder_blocksize,
        parallelism,
        faer_qr::col_pivoting::compute::ColPivQrComputeParams::default(),
    )?;
    let permute_cols = faer_core::permutation::permute_rows_in_place_req::<T>(n, n)?;

    let apply_householder =
        faer_core::householder::apply_block_householder_sequence_on_the_left_in_place_req::<T>(
            m,
            householder_blocksize,
            match compute_u {
                ComputeVectors::No => 0,
                ComputeVectors::Thin => n,
                ComputeVectors::Full => m,
            },
        )?;

    StackReq::try_all_of([
        qr,
        householder,
        StackReq::try_any_of([
            StackReq::try_all_of([
                r,
                StackReq::try_any_of([
                    StackReq::try_all_of([col_perm, col_perm_inv, compute_qr])?,
                    permute_cols,
                ])?,
            ])?,
            apply_householder,
        ])?,
    ])
}

fn compute_real_svd_big_req<T: 'static>(
    mut m: usize,
    mut n: usize,
    mut compute_u: ComputeVectors,
    mut compute_v: ComputeVectors,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    if n > m {
        swap(&mut m, &mut n);
        swap(&mut compute_u, &mut compute_v);
    }

    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<T>(m, n);

    let bid = temp_mat_req::<T>(m, n)?;
    let householder_left = temp_mat_req::<T>(householder_blocksize, n)?;
    let householder_right = temp_mat_req::<T>(householder_blocksize, n - 1)?;

    let compute_bidiag = bidiag::bidiagonalize_in_place_req::<T>(m, n, parallelism)?;

    let diag = StackReq::try_new::<T>(n)?;
    let subdiag = diag;
    let compute_ub = compute_v != ComputeVectors::No;
    let compute_vb = compute_u != ComputeVectors::No;
    let u_b = temp_mat_req::<T>(if compute_ub { n + 1 } else { 2 }, n + 1)?;
    let v_b = temp_mat_req::<T>(n, if compute_vb { n } else { 0 })?;

    let compute_bidiag_svd =
        bidiag_real_svd::bidiag_svd_req::<T>(n, compute_ub, compute_vb, parallelism)?;
    let apply_householder_u =
        faer_core::householder::apply_block_householder_sequence_on_the_left_in_place_req::<T>(
            m,
            householder_blocksize,
            match compute_u {
                ComputeVectors::No => 0,
                ComputeVectors::Thin => n,
                ComputeVectors::Full => m,
            },
        )?;
    let apply_householder_v =
        faer_core::householder::apply_block_householder_sequence_on_the_left_in_place_req::<T>(
            n - 1,
            householder_blocksize,
            match compute_u {
                ComputeVectors::No => 0,
                _ => n,
            },
        )?;

    StackReq::try_all_of([
        bid,
        householder_left,
        householder_right,
        StackReq::try_any_of([
            compute_bidiag,
            StackReq::try_all_of([
                diag,
                subdiag,
                u_b,
                v_b,
                StackReq::try_any_of([
                    compute_bidiag_svd,
                    StackReq::try_all_of([apply_householder_u, apply_householder_v])?,
                ])?,
            ])?,
        ])?,
    ])
}

/// does qr -> jacobi svd
///
/// question: should we have a version for square matrices that skips the QR?
fn compute_real_svd_small<T: RealField>(
    matrix: MatRef<'_, T>,
    s: ColMut<'_, T>,
    u: Option<MatMut<'_, T>>,
    v: Option<MatMut<'_, T>>,
    epsilon: T,
    zero_threshold: T,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut u = u;
    let mut v = v;

    let do_transpose = matrix.ncols() > matrix.nrows();

    let matrix = if do_transpose {
        matrix.transpose()
    } else {
        matrix
    };

    if do_transpose {
        swap(&mut u, &mut v);
    }

    let m = matrix.nrows();
    let n = matrix.ncols();

    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<T>(m, n);

    temp_mat_uninit! {
        let (mut qr, stack) = unsafe { temp_mat_uninit::<T>(m, n, stack) };
        let (mut householder, mut stack) = unsafe { temp_mat_uninit::<T>(householder_blocksize, n, stack) };
    }

    {
        temp_mat_uninit! {
            let (mut r, mut stack) = unsafe { temp_mat_uninit::<T>(n, n, stack.rb_mut()) };
        }

        zip!(qr.rb_mut(), matrix).for_each(|dst, src| *dst = *src);

        {
            let (mut col_perm, stack) = stack.rb_mut().make_with(n, |_| 0usize);
            let (mut col_perm_inv, mut stack) = stack.make_with(n, |_| 0usize);

            // matrix = q * r * P
            let (_, col_perm) = faer_qr::col_pivoting::compute::qr_in_place(
                qr.rb_mut(),
                householder.rb_mut(),
                &mut col_perm,
                &mut col_perm_inv,
                parallelism,
                stack.rb_mut(),
                faer_qr::col_pivoting::compute::ColPivQrComputeParams::default(),
            );
            zip!(r.rb_mut()).for_each_triangular_lower(zip::Diag::Skip, |dst| *dst = T::zero());
            zip!(r.rb_mut(), qr.rb().submatrix(0, 0, n, n))
                .for_each_triangular_upper(zip::Diag::Include, |dst, src| *dst = *src);
            faer_core::permutation::permute_cols_in_place(
                r.rb_mut(),
                col_perm.rb().inverse(),
                stack,
            );
        }

        // r * P = u s v
        jacobi::jacobi_svd(
            r.rb_mut(),
            u.rb_mut().map(|u| u.submatrix(0, 0, n, n)),
            v.rb_mut(),
            jacobi::Skip::None,
            epsilon,
            zero_threshold,
        );
        zip!(s, r.rb().diagonal()).for_each(|dst, src| *dst = *src);
    }

    // matrix = q u s v
    if let Some(mut u) = u.rb_mut() {
        let ncols = u.ncols();
        zip!(u.rb_mut().submatrix(n, 0, m - n, n)).for_each(|dst| *dst = T::zero());
        zip!(u.rb_mut().submatrix(0, n, m, ncols - n)).for_each(|dst| *dst = T::zero());
        if ncols == m {
            zip!(u.rb_mut().submatrix(n, n, m - n, m - n).diagonal())
                .for_each(|dst| *dst = T::one());
        }

        faer_core::householder::apply_block_householder_sequence_on_the_left_in_place(
            qr.rb(),
            householder.rb(),
            Conj::No,
            u,
            Conj::No,
            parallelism,
            stack.rb_mut(),
        );
    }
}

/// does bidiagonilization -> divide conquer svd
fn compute_real_svd_big<T: RealField>(
    matrix: MatRef<'_, T>,
    s: ColMut<'_, T>,
    u: Option<MatMut<'_, T>>,
    v: Option<MatMut<'_, T>>,
    epsilon: T,
    zero_threshold: T,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut stack = stack;
    let mut u = u;
    let mut v = v;

    let do_transpose = matrix.ncols() > matrix.nrows();

    let matrix = if do_transpose {
        matrix.transpose()
    } else {
        matrix
    };

    if do_transpose {
        swap(&mut u, &mut v);
    }

    let m = matrix.nrows();
    let n = matrix.ncols();
    if n == 0 {
        if let Some(mut u) = u {
            zip!(u.rb_mut()).for_each(|dst| *dst = T::zero());
            zip!(u.submatrix(0, 0, n, n).diagonal()).for_each(|dst| *dst = T::one());
        }

        return;
    }

    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<T>(m, n);

    temp_mat_uninit! {
        let (mut bid, stack) = unsafe { temp_mat_uninit::<T>(m, n, stack.rb_mut()) };
        let (mut householder_left, stack) = unsafe { temp_mat_uninit::<T>(householder_blocksize, n, stack) };
        let (mut householder_right, mut stack) = unsafe { temp_mat_uninit::<T>(householder_blocksize, n - 1, stack) };
    }

    zip!(bid.rb_mut(), matrix).for_each(|dst, src| *dst = *src);

    bidiag::bidiagonalize_in_place(
        bid.rb_mut(),
        householder_left.rb_mut().row(0).transpose(),
        householder_right.rb_mut().row(0).transpose(),
        parallelism,
        stack.rb_mut(),
    );

    let bid = bid.into_const();

    let (mut diag, stack) = stack.make_with(n, |i| bid[(i, i)]);
    let (mut subdiag, stack) = stack.make_with(n, |i| {
        if i < n - 1 {
            bid[(i, i + 1)]
        } else {
            T::zero()
        }
    });

    temp_mat_uninit! {
        let (mut u_b, stack) = unsafe { temp_mat_uninit::<T>(if v.is_some() { n + 1 } else { 2 }, n + 1, stack) };
        let (mut v_b, mut stack) = unsafe { temp_mat_uninit::<T>(n, if u.is_some() { n } else { 0 }, stack) };
    }

    let mut j_base = 0;
    while j_base < n {
        let bs = householder_blocksize.min(n - j_base);
        let mut householder = householder_left.rb_mut().submatrix(0, j_base, bs, bs);
        let essentials = bid.submatrix(j_base, j_base, m - j_base, bs);
        for j in 0..bs {
            householder[(j, j)] = householder[(0, j)];
        }
        upgrade_householder_factor(householder, essentials, bs, 1, parallelism);
        j_base += bs;
    }
    let mut j_base = 0;
    while j_base < n - 1 {
        let bs = householder_blocksize.min(n - 1 - j_base);
        let mut householder = householder_right.rb_mut().submatrix(0, j_base, bs, bs);
        let full_essentials = bid.submatrix(0, 1, m, n - 1).transpose();
        let essentials = full_essentials.submatrix(j_base, j_base, n - 1 - j_base, bs);
        for j in 0..bs {
            householder[(j, j)] = householder[(0, j)];
        }
        upgrade_householder_factor(householder, essentials, bs, 1, parallelism);
        j_base += bs;
    }

    bidiag_real_svd::bidiag_svd(
        &mut diag,
        &mut subdiag,
        u_b.rb_mut(),
        u.is_some().then_some(v_b.rb_mut()),
        v.is_some(),
        JACOBI_FALLBACK_THRESHOLD,
        epsilon,
        zero_threshold,
        parallelism,
        stack.rb_mut(),
    );

    for (s, val) in s.into_iter().zip(&*diag) {
        *s = *val;
    }

    if let Some(mut u) = u {
        let ncols = u.ncols();
        zip!(
            u.rb_mut().submatrix(0, 0, n, n),
            v_b.rb().submatrix(0, 0, n, n),
        )
        .for_each(|dst, src| *dst = *src);

        zip!(u.rb_mut().submatrix(n, 0, m - n, ncols)).for_each(|x| *x = T::zero());
        zip!(u.rb_mut().submatrix(0, n, n, ncols - n)).for_each(|x| *x = T::zero());
        zip!(u.rb_mut().submatrix(n, n, ncols - n, ncols - n).diagonal())
            .for_each(|x| *x = T::one());

        apply_block_householder_sequence_on_the_left_in_place(
            bid,
            householder_left.rb(),
            Conj::No,
            u,
            Conj::No,
            parallelism,
            stack.rb_mut(),
        );
    };
    if let Some(mut v) = v {
        zip!(
            v.rb_mut().submatrix(0, 0, n, n),
            u_b.rb().submatrix(0, 0, n, n),
        )
        .for_each(|dst, src| *dst = *src);

        apply_block_householder_sequence_on_the_left_in_place(
            bid.submatrix(0, 1, m, n - 1).transpose(),
            householder_right.rb(),
            Conj::No,
            v.submatrix(1, 0, n - 1, n),
            Conj::No,
            parallelism,
            stack.rb_mut(),
        );
    }
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct SvdParams {}

/// Computes the size and alignment of required workspace for performing a singular value
/// decomposition. $U$ and $V$ may be computed fully, partially, or not computed at all.
pub fn compute_svd_req<T: ComplexField>(
    nrows: usize,
    ncols: usize,
    compute_u: ComputeVectors,
    compute_v: ComputeVectors,
    parallelism: Parallelism,
    params: SvdParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = params;
    if coe::is_same::<T, T::Real>() {
        let size = usize::min(nrows, ncols);
        if size <= JACOBI_FALLBACK_THRESHOLD {
            compute_real_svd_small_req::<T>(nrows, ncols, compute_u, compute_v, parallelism)
        } else {
            compute_real_svd_big_req::<T>(nrows, ncols, compute_u, compute_v, parallelism)
        }
    } else if coe::is_same::<T, Complex<T::Real>>() {
        todo!("complex values are not yet supported in the svd")
    } else {
        unimplemented!("only real and complex values are supported in the svd")
    }
}

/// Computes the singular value decomposition of `matrix`.
///
/// `s` represents the main diagonal of the matrix $S$, and must have size equal to the minimum of
/// `matrix.nrows()` and `matrix.ncols()`.
///
/// For each of `u` and `v`:
/// - If the argument is `None`, then the corresponding singular vector matrix is not computed.
/// - If it is `Some(..)`, then it must have a number of rows equal to `matrix.nrows()` for `u`,
/// and `matrix.ncols()` for `v`.
/// - The number of columns may be either equal to the number of rows, or it may be equal to the
/// minimum of `matrix.nrows()` and `matrix.ncols()`, in which case only the singular vectors
/// corresponding to the provided column storage are computed.
pub fn compute_svd<T: ComplexField>(
    matrix: MatRef<'_, T>,
    s: ColMut<'_, T>,
    u: Option<MatMut<'_, T>>,
    v: Option<MatMut<'_, T>>,
    epsilon: T,
    zero_threshold: T,
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: SvdParams,
) {
    let size = usize::min(matrix.nrows(), matrix.ncols());
    fancy_assert!(s.nrows() == size);
    if let Some(u) = u.rb() {
        fancy_assert!(u.nrows() == matrix.nrows());
        fancy_assert!(u.ncols() == matrix.nrows() || u.ncols() == size);
    }
    if let Some(v) = v.rb() {
        fancy_assert!(v.nrows() == matrix.ncols());
        fancy_assert!(v.ncols() == matrix.ncols() || v.ncols() == size);
    }

    let _ = params;
    if coe::is_same::<T, T::Real>() {
        if size <= JACOBI_FALLBACK_THRESHOLD {
            compute_real_svd_small::<T::Real>(
                matrix.coerce(),
                s.coerce(),
                u.map(coe::Coerce::coerce),
                v.map(coe::Coerce::coerce),
                coe::coerce_static(epsilon),
                coe::coerce_static(zero_threshold),
                parallelism,
                stack,
            );
        } else {
            compute_real_svd_big::<T::Real>(
                matrix.coerce(),
                s.coerce(),
                u.map(coe::Coerce::coerce),
                v.map(coe::Coerce::coerce),
                coe::coerce_static(epsilon),
                coe::coerce_static(zero_threshold),
                parallelism,
                stack,
            );
        }
    } else if coe::is_same::<T, Complex<T::Real>>() {
        todo!("complex values are not yet supported in the svd")
    } else {
        unimplemented!("only real and complex values are supported in the svd")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::Mat;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req))
        };
    }

    #[test]
    fn test_real_big() {
        for (m, n) in [(15, 10), (10, 15), (10, 10), (15, 15)] {
            let mat = Mat::with_dims(|_, _| rand::random::<f64>(), m, n);
            let size = m.min(n);

            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            compute_real_svd_big(
                mat.as_ref(),
                s.as_mut().submatrix(0, 0, size, size).diagonal(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
                Parallelism::None,
                make_stack!(compute_real_svd_big_req::<f64>(
                    m,
                    n,
                    ComputeVectors::Full,
                    ComputeVectors::Full,
                    Parallelism::None,
                )
                .unwrap()),
            );

            let reconstructed = &u * &s * v.transpose();

            for j in 0..n {
                for i in 0..m {
                    assert_approx_eq!(reconstructed[(i, j)], mat[(i, j)], 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_identity() {
        for (m, n) in [(15, 10), (10, 15), (10, 10), (15, 15)] {
            let mut mat = Mat::zeros(m, n);
            let size = m.min(n);
            for i in 0..size {
                mat[(i, i)] = 1.0;
            }

            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            compute_real_svd_big(
                mat.as_ref(),
                s.as_mut().submatrix(0, 0, size, size).diagonal(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
                Parallelism::None,
                make_stack!(compute_real_svd_big_req::<f64>(
                    m,
                    n,
                    ComputeVectors::Full,
                    ComputeVectors::Full,
                    Parallelism::None,
                )
                .unwrap()),
            );

            let reconstructed = &u * &s * v.transpose();

            for j in 0..n {
                for i in 0..m {
                    assert_approx_eq!(reconstructed[(i, j)], mat[(i, j)], 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_zero() {
        for (m, n) in [(15, 10), (10, 15), (10, 10), (15, 15)] {
            let mat = Mat::zeros(m, n);
            let size = m.min(n);

            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            compute_real_svd_big(
                mat.as_ref(),
                s.as_mut().submatrix(0, 0, size, size).diagonal(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
                Parallelism::None,
                make_stack!(compute_real_svd_big_req::<f64>(
                    m,
                    n,
                    ComputeVectors::Full,
                    ComputeVectors::Full,
                    Parallelism::None,
                )
                .unwrap()),
            );

            let reconstructed = &u * &s * v.transpose();

            for j in 0..n {
                for i in 0..m {
                    assert_approx_eq!(reconstructed[(i, j)], mat[(i, j)], 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_real_small() {
        for (m, n) in [(4, 4), (5, 5), (15, 10), (10, 15), (10, 10), (15, 15)] {
            let mat = Mat::with_dims(|_, _| rand::random::<f64>(), m, n);
            let size = m.min(n);

            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            compute_real_svd_small(
                mat.as_ref(),
                s.as_mut().submatrix(0, 0, size, size).diagonal(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
                Parallelism::None,
                make_stack!(compute_real_svd_small_req::<f64>(
                    m,
                    n,
                    ComputeVectors::Full,
                    ComputeVectors::Full,
                    Parallelism::None,
                )
                .unwrap()),
            );

            let reconstructed = &u * &s * v.transpose();

            for j in 0..n {
                for i in 0..m {
                    assert_approx_eq!(reconstructed[(i, j)], mat[(i, j)], 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_real() {
        for m in 0..10 {
            for n in 0..10 {
                let mat = Mat::with_dims(|_, _| rand::random::<f64>(), m, n);
                let size = m.min(n);

                let mut s = Mat::zeros(m, n);
                let mut u = Mat::zeros(m, m);
                let mut v = Mat::zeros(n, n);

                compute_svd(
                    mat.as_ref(),
                    s.as_mut().submatrix(0, 0, size, size).diagonal(),
                    Some(u.as_mut()),
                    Some(v.as_mut()),
                    f64::EPSILON,
                    f64::MIN_POSITIVE,
                    Parallelism::None,
                    make_stack!(compute_svd_req::<f64>(
                        m,
                        n,
                        ComputeVectors::Full,
                        ComputeVectors::Full,
                        Parallelism::None,
                        SvdParams::default(),
                    )
                    .unwrap()),
                    SvdParams::default(),
                );

                let reconstructed = &u * &s * v.transpose();

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(reconstructed[(i, j)], mat[(i, j)], 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_real_thin() {
        for m in 0..20 {
            for n in 0..20 {
                use ComputeVectors::*;
                for compute_u in [No, Thin, Full] {
                    for compute_v in [No, Thin, Full] {
                        dbg!(m, n, compute_u, compute_v);
                        let mat = Mat::with_dims(|_, _| rand::random::<f64>(), m, n);
                        let size = m.min(n);

                        let mut s = Mat::zeros(m, n);
                        let mut u = Mat::zeros(
                            m,
                            match compute_u {
                                No => 0,
                                Thin => size,
                                Full => m,
                            },
                        );
                        let mut v = Mat::zeros(
                            n,
                            match compute_v {
                                No => 0,
                                Thin => size,
                                Full => n,
                            },
                        );

                        compute_svd(
                            mat.as_ref(),
                            s.as_mut().submatrix(0, 0, size, size).diagonal(),
                            if compute_u == No {
                                None
                            } else {
                                Some(u.as_mut())
                            },
                            if compute_v == No {
                                None
                            } else {
                                Some(v.as_mut())
                            },
                            f64::EPSILON,
                            f64::MIN_POSITIVE,
                            Parallelism::None,
                            make_stack!(compute_svd_req::<f64>(
                                m,
                                n,
                                compute_u,
                                compute_v,
                                Parallelism::None,
                                SvdParams::default(),
                            )
                            .unwrap()),
                            SvdParams::default(),
                        );

                        let mut s_target = Mat::zeros(m, n);
                        let mut u_target = Mat::zeros(m, m);
                        let mut v_target = Mat::zeros(n, n);

                        compute_svd(
                            mat.as_ref(),
                            s_target.as_mut().submatrix(0, 0, size, size).diagonal(),
                            Some(u_target.as_mut()),
                            Some(v_target.as_mut()),
                            f64::EPSILON,
                            f64::MIN_POSITIVE,
                            Parallelism::None,
                            make_stack!(compute_svd_req::<f64>(
                                m,
                                n,
                                ComputeVectors::Full,
                                ComputeVectors::Full,
                                Parallelism::None,
                                SvdParams::default(),
                            )
                            .unwrap()),
                            SvdParams::default(),
                        );

                        for j in 0..u.ncols() {
                            for i in 0..u.nrows() {
                                assert_approx_eq!(u[(i, j)], u_target[(i, j)], 1e-10);
                            }
                        }
                        for j in 0..v.ncols() {
                            for i in 0..v.nrows() {
                                assert_approx_eq!(v[(i, j)], v_target[(i, j)], 1e-10);
                            }
                        }
                        for j in 0..s.ncols() {
                            for i in 0..s.nrows() {
                                assert_approx_eq!(s[(i, j)], s_target[(i, j)], 1e-10);
                            }
                        }
                    }
                }
            }
        }
    }
}
