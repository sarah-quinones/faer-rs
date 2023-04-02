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
use bidiag_real_svd::bidiag_real_svd_req;
use coe::Coerce;
use core::{iter::zip, mem::swap};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    householder::{
        apply_block_householder_sequence_on_the_left_in_place,
        apply_block_householder_sequence_on_the_left_in_place_req, upgrade_householder_factor,
    },
    temp_mat_req, temp_mat_uninit, zip,
    zip::RowUninit,
    ColMut, ComplexField, Conj, MatMut, MatRef, Parallelism, RealField,
};
use num_complex::Complex;
use reborrow::*;

use crate::bidiag_real_svd::compute_bidiag_real_svd;

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
    m: usize,
    n: usize,
    compute_u: ComputeVectors,
    compute_v: ComputeVectors,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    fancy_assert!(m >= n);

    if m == n {
        return temp_mat_req::<T>(m, n);
    }

    let _ = compute_v;
    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<T>(m, n);

    let qr = temp_mat_req::<T>(m, n)?;
    let householder = temp_mat_req::<T>(householder_blocksize, n)?;
    let r = temp_mat_req::<T>(n, n)?;

    let compute_qr = faer_qr::no_pivoting::compute::qr_in_place_req::<T>(
        m,
        n,
        householder_blocksize,
        parallelism,
        Default::default(),
    )?;

    let apply_householder = apply_block_householder_sequence_on_the_left_in_place_req::<T>(
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
        StackReq::try_any_of([StackReq::try_all_of([r, compute_qr])?, apply_householder])?,
    ])
}

fn compute_svd_big_req<T: 'static>(
    m: usize,
    n: usize,
    compute_u: ComputeVectors,
    compute_v: ComputeVectors,
    bidiag_svd_req: fn(
        n: usize,
        jacobi_fallback_threshold: usize,
        compute_u: bool,
        compute_v: bool,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow>,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    fancy_assert!(m >= n);
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

    let compute_bidiag_svd = bidiag_svd_req(
        n,
        JACOBI_FALLBACK_THRESHOLD,
        compute_ub,
        compute_vb,
        parallelism,
    )?;
    let apply_householder_u = apply_block_householder_sequence_on_the_left_in_place_req::<T>(
        m,
        householder_blocksize,
        match compute_u {
            ComputeVectors::No => 0,
            ComputeVectors::Thin => n,
            ComputeVectors::Full => m,
        },
    )?;
    let apply_householder_v = apply_block_householder_sequence_on_the_left_in_place_req::<T>(
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

    assert!(matrix.nrows() >= matrix.ncols());

    let m = matrix.nrows();
    let n = matrix.ncols();

    // if the matrix is square, skip the QR
    if m == n {
        temp_mat_uninit! {
            let (mut jacobi_mat, _) = unsafe { temp_mat_uninit::<T>(m, n, stack) };
        }
        zip!(jacobi_mat.rb_mut(), matrix).for_each(|dst, src| *dst = src.clone());

        jacobi::jacobi_svd(
            jacobi_mat.rb_mut(),
            u,
            v,
            jacobi::Skip::None,
            epsilon,
            zero_threshold,
        );
        zip!(s, jacobi_mat.rb().diagonal()).for_each(|dst, src| *dst = src.clone());
        return;
    }

    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<T>(m, n);

    temp_mat_uninit! {
        let (mut qr, stack) = unsafe { temp_mat_uninit::<T>(m, n, stack) };
        let (mut householder, mut stack) = unsafe { temp_mat_uninit::<T>(householder_blocksize, n, stack) };
    }

    {
        temp_mat_uninit! {
            let (mut r, mut stack) = unsafe { temp_mat_uninit::<T>(n, n, stack.rb_mut()) };
        }

        zip!(qr.rb_mut(), matrix).for_each(|dst, src| *dst = src.clone());

        // matrix = q * r
        faer_qr::no_pivoting::compute::qr_in_place(
            qr.rb_mut(),
            householder.rb_mut(),
            parallelism,
            stack.rb_mut(),
            Default::default(),
        );
        zip!(r.rb_mut()).for_each_triangular_lower(zip::Diag::Skip, |dst| *dst = T::zero());
        zip!(r.rb_mut(), qr.rb().submatrix(0, 0, n, n))
            .for_each_triangular_upper(zip::Diag::Include, |dst, src| *dst = src.clone());

        // r = u s v
        jacobi::jacobi_svd(
            r.rb_mut(),
            u.rb_mut().map(|u| u.submatrix(0, 0, n, n)),
            v.rb_mut(),
            jacobi::Skip::None,
            epsilon,
            zero_threshold,
        );
        zip!(s, r.rb().diagonal()).for_each(|dst, src| *dst = src.clone());
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

fn compute_bidiag_cplx_svd<T: RealField>(
    diag: &mut [Complex<T>],
    subdiag: &mut [Complex<T>],
    mut u: Option<MatMut<'_, Complex<T>>>,
    mut v: Option<MatMut<'_, Complex<T>>>,
    jacobi_fallback_threshold: usize,
    epsilon: T,
    consider_zero_threshold: T,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let n = diag.len();
    temp_mat_uninit! {
        let (mut u_real, stack) = unsafe { temp_mat_uninit::<T>(n + 1, if u.is_some() { n + 1 } else { 0 }, stack) };
        let (mut v_real, stack) = unsafe { temp_mat_uninit::<T>(n, if v.is_some() { n } else { 0 }, stack) };
    }
    let (mut diag_real, stack) = stack.collect(diag.iter().map(|x| x.abs()));
    let (mut subdiag_real, stack) = stack.collect(subdiag.iter().map(|x| x.abs()));

    let (mut col_mul, stack) = stack.make_with(n, |_| Complex::<T>::zero());
    let (mut row_mul, stack) = stack.make_with(n - 1, |_| Complex::<T>::zero());

    let normalized = |x: Complex<T>| {
        if x == Complex::<T>::zero() {
            Complex::<T>::one()
        } else {
            x.scale_real(&x.abs().inv())
        }
    };

    let mut col_normalized = normalized(diag[0].clone()).conj();
    col_mul[0] = col_normalized.clone();
    for i in 1..n {
        let row_normalized = normalized(subdiag[i - 1].mul(&col_normalized)).conj();
        row_mul[i - 1] = row_normalized.conj();
        col_normalized = normalized(diag[i].mul(&row_normalized)).conj();
        col_mul[i] = col_normalized.clone();
    }

    compute_bidiag_real_svd(
        &mut diag_real,
        &mut subdiag_real,
        u.is_some().then_some(u_real.rb_mut()),
        v.is_some().then_some(v_real.rb_mut()),
        jacobi_fallback_threshold,
        epsilon,
        consider_zero_threshold,
        parallelism,
        stack,
    );

    for i in 0..n {
        diag[i] = Complex::<T>::from_real(diag_real[i].clone());
    }

    let u_real = u_real.rb();
    let v_real = v_real.rb();

    if let Some(mut u) = u.rb_mut() {
        zip!(RowUninit(u.rb_mut().row(0)), u_real.row(0))
            .for_each(|u, u_real| unsafe { *u = Complex::<T>::from_real(u_real.clone()) });
        zip!(RowUninit(u.rb_mut().row(n)), u_real.row(n))
            .for_each(|u, u_real| unsafe { *u = Complex::<T>::from_real(u_real.clone()) });

        for (u, u_real) in zip(u.into_col_iter(), u_real.into_col_iter()) {
            let u = u.subrows(1, n - 1);
            let u_real = u_real.subrows(1, n - 1);
            for ((u, u_real), k) in zip(zip(u, u_real), &*row_mul) {
                *u = k.scale_real(u_real);
            }
        }
    }
    if let Some(v) = v.rb_mut() {
        for (v, v_real) in zip(v.into_col_iter(), v_real.into_col_iter()) {
            for ((v, v_real), k) in zip(zip(v, v_real), &*col_mul) {
                *v = k.scale_real(v_real);
            }
        }
    }
}

pub fn bidiag_cplx_svd_req<T: 'static>(
    n: usize,
    jacobi_fallback_threshold: usize,
    compute_u: bool,
    compute_v: bool,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        temp_mat_req::<T>(n + 1, if compute_u { n + 1 } else { 0 })?,
        temp_mat_req::<T>(n, if compute_u { n } else { 0 })?,
        StackReq::try_new::<T>(n)?,
        StackReq::try_new::<T>(n)?,
        StackReq::try_new::<Complex<T>>(n)?,
        StackReq::try_new::<Complex<T>>(n - 1)?,
        bidiag_real_svd_req::<T>(
            n,
            jacobi_fallback_threshold,
            compute_u,
            compute_v,
            parallelism,
        )?,
    ])
}

/// does bidiagonilization -> divide conquer svd
fn compute_svd_big<T: ComplexField>(
    matrix: MatRef<'_, T>,
    s: ColMut<'_, T>,
    u: Option<MatMut<'_, T>>,
    v: Option<MatMut<'_, T>>,
    bidiag_svd: fn(
        diag: &mut [T],
        subdiag: &mut [T],
        u: Option<MatMut<'_, T>>,
        v: Option<MatMut<'_, T>>,
        jacobi_fallback_threshold: usize,
        epsilon: T::Real,
        consider_zero_threshold: T::Real,
        parallelism: Parallelism,
        stack: DynStack<'_>,
    ),
    epsilon: T::Real,
    zero_threshold: T::Real,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut stack = stack;

    assert!(matrix.nrows() >= matrix.ncols());

    let m = matrix.nrows();
    let n = matrix.ncols();
    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<T>(m, n);

    temp_mat_uninit! {
        let (mut bid, stack) = unsafe { temp_mat_uninit::<T>(m, n, stack.rb_mut()) };
        let (mut householder_left, stack) = unsafe { temp_mat_uninit::<T>(householder_blocksize, n, stack) };
        let (mut householder_right, mut stack) = unsafe { temp_mat_uninit::<T>(householder_blocksize, n - 1, stack) };
    }

    zip!(bid.rb_mut(), matrix).for_each(|dst, src| *dst = src.clone());

    bidiag::bidiagonalize_in_place(
        bid.rb_mut(),
        householder_left.rb_mut().row(0).transpose(),
        householder_right.rb_mut().row(0).transpose(),
        parallelism,
        stack.rb_mut(),
    );

    let bid = bid.into_const();

    let (mut diag, stack) = stack.make_with(n, |i| bid[(i, i)].conj());
    let (mut subdiag, stack) = stack.make_with(n, |i| {
        if i < n - 1 {
            bid[(i, i + 1)].conj()
        } else {
            T::zero()
        }
    });

    let mut j_base = 0;
    while j_base < n {
        let bs = householder_blocksize.min(n - j_base);
        let mut householder = householder_left.rb_mut().submatrix(0, j_base, bs, bs);
        let essentials = bid.submatrix(j_base, j_base, m - j_base, bs);
        for j in 0..bs {
            householder[(j, j)] = householder[(0, j)].clone();
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
            householder[(j, j)] = householder[(0, j)].clone();
        }
        upgrade_householder_factor(householder, essentials, bs, 1, parallelism);
        j_base += bs;
    }

    temp_mat_uninit! {
        let (mut u_b, stack) = unsafe { temp_mat_uninit::<T>(if v.is_some() { n + 1 } else { 0 }, n + 1, stack) };
        let (mut v_b, mut stack) = unsafe { temp_mat_uninit::<T>(n, if u.is_some() { n } else { 0 }, stack) };
    }

    bidiag_svd(
        &mut diag,
        &mut subdiag,
        v.is_some().then_some(u_b.rb_mut()),
        u.is_some().then_some(v_b.rb_mut()),
        JACOBI_FALLBACK_THRESHOLD,
        epsilon,
        zero_threshold,
        parallelism,
        stack.rb_mut(),
    );

    for (s, val) in s.into_iter().zip(&*diag) {
        *s = val.clone();
    }

    if let Some(mut u) = u {
        let ncols = u.ncols();
        zip!(
            u.rb_mut().submatrix(0, 0, n, n),
            v_b.rb().submatrix(0, 0, n, n),
        )
        .for_each(|dst, src| *dst = src.clone());

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
        .for_each(|dst, src| *dst = src.clone());

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
    let mut nrows = nrows;
    let mut ncols = ncols;
    let mut compute_u = compute_u;
    let mut compute_v = compute_v;
    let do_transpose = ncols > nrows;
    if do_transpose {
        swap(&mut nrows, &mut ncols);
        swap(&mut compute_u, &mut compute_v);
    }

    if ncols == 0 {
        return Ok(StackReq::default());
    }

    let size = usize::min(nrows, ncols);
    let skip_qr = nrows as f64 / ncols as f64 <= 11.0 / 6.0;
    let (svd_nrows, svd_ncols) = if skip_qr {
        (nrows, ncols)
    } else {
        (size, size)
    };

    let _ = params;
    let squareish_svd = if coe::is_same::<T, T::Real>() {
        if size <= JACOBI_FALLBACK_THRESHOLD {
            compute_real_svd_small_req::<T>(svd_nrows, svd_ncols, compute_u, compute_v, parallelism)
        } else {
            compute_svd_big_req::<T::Real>(
                svd_nrows,
                svd_ncols,
                compute_u,
                compute_v,
                bidiag_real_svd_req::<T::Real>,
                parallelism,
            )
        }
    } else if coe::is_same::<T, Complex<T::Real>>() {
        compute_svd_big_req::<Complex<T::Real>>(
            svd_nrows,
            svd_ncols,
            compute_u,
            compute_v,
            bidiag_cplx_svd_req::<T::Real>,
            parallelism,
        )
    } else {
        unimplemented!("only real and complex values are supported in the svd")
    }?;

    if skip_qr {
        Ok(squareish_svd)
    } else {
        let householder_blocksize =
            faer_qr::no_pivoting::compute::recommended_blocksize::<T>(nrows, ncols);

        StackReq::try_all_of([
            temp_mat_req::<T>(nrows, ncols)?,
            temp_mat_req::<T>(householder_blocksize, ncols)?,
            StackReq::try_any_of([
                StackReq::try_all_of([
                    temp_mat_req::<T>(size, size)?,
                    StackReq::try_any_of([
                        faer_qr::no_pivoting::compute::qr_in_place_req::<T>(
                            nrows,
                            ncols,
                            householder_blocksize,
                            parallelism,
                            Default::default(),
                        )?,
                        squareish_svd,
                    ])?,
                ])?,
                apply_block_householder_sequence_on_the_left_in_place_req::<T>(
                    nrows,
                    householder_blocksize,
                    nrows,
                )?,
            ])?,
        ])
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
    epsilon: T::Real,
    zero_threshold: T::Real,
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

    let mut u = u;
    let mut v = v;
    let mut matrix = matrix;
    let do_transpose = matrix.ncols() > matrix.nrows();
    if do_transpose {
        matrix = matrix.transpose();
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

    let _ = params;

    if m as f64 / n as f64 <= 11.0 / 6.0 {
        squareish_svd(
            matrix,
            s,
            u.rb_mut(),
            v.rb_mut(),
            epsilon,
            zero_threshold,
            parallelism,
            stack,
        );
    } else {
        // do a qr first, then do the svd
        let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<T>(m, n);

        temp_mat_uninit! {
            let (mut qr, stack) = unsafe { temp_mat_uninit::<T>(m, n, stack) };
            let (mut householder, mut stack) = unsafe { temp_mat_uninit::<T>(householder_blocksize, n, stack) };
        }

        {
            temp_mat_uninit! {
                let (mut r, mut stack) = unsafe { temp_mat_uninit::<T>(n, n, stack.rb_mut()) };
            }

            zip!(qr.rb_mut(), matrix).for_each(|dst, src| *dst = src.clone());

            // matrix = q * r
            faer_qr::no_pivoting::compute::qr_in_place(
                qr.rb_mut(),
                householder.rb_mut(),
                parallelism,
                stack.rb_mut(),
                Default::default(),
            );
            zip!(r.rb_mut()).for_each_triangular_lower(zip::Diag::Skip, |dst| *dst = T::zero());
            zip!(r.rb_mut(), qr.rb().submatrix(0, 0, n, n))
                .for_each_triangular_upper(zip::Diag::Include, |dst, src| *dst = src.clone());

            // r = u s v
            squareish_svd(
                r.rb(),
                s,
                u.rb_mut().map(|u| u.submatrix(0, 0, n, n)),
                v.rb_mut(),
                epsilon,
                zero_threshold,
                parallelism,
                stack,
            );
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

    if do_transpose {
        // conjugate u and v
        if let Some(u) = u {
            zip!(u).for_each(|x| *x = (*x).conj())
        }
        if let Some(v) = v {
            zip!(v).for_each(|x| *x = (*x).conj())
        }
    }
}

fn squareish_svd<T: ComplexField>(
    matrix: MatRef<T>,
    s: ColMut<T>,
    mut u: Option<MatMut<T>>,
    mut v: Option<MatMut<T>>,
    epsilon: T::Real,
    zero_threshold: T::Real,
    parallelism: Parallelism,
    stack: DynStack,
) {
    let size = matrix.ncols();
    if coe::is_same::<T, T::Real>() {
        if size <= JACOBI_FALLBACK_THRESHOLD {
            compute_real_svd_small::<T::Real>(
                matrix.coerce(),
                s.coerce(),
                u.rb_mut().map(coe::Coerce::coerce),
                v.rb_mut().map(coe::Coerce::coerce),
                coe::coerce_static(epsilon),
                coe::coerce_static(zero_threshold),
                parallelism,
                stack,
            );
        } else {
            compute_svd_big::<T::Real>(
                matrix.coerce(),
                s.coerce(),
                u.rb_mut().map(coe::Coerce::coerce),
                v.rb_mut().map(coe::Coerce::coerce),
                compute_bidiag_real_svd::<T::Real>,
                coe::coerce_static(epsilon),
                coe::coerce_static(zero_threshold),
                parallelism,
                stack,
            );
        }
    } else if coe::is_same::<T, Complex<T::Real>>() {
        compute_svd_big::<Complex<T::Real>>(
            matrix.coerce(),
            s.coerce(),
            u.rb_mut().map(coe::Coerce::coerce),
            v.rb_mut().map(coe::Coerce::coerce),
            compute_bidiag_cplx_svd::<T::Real>,
            coe::coerce_static(epsilon),
            coe::coerce_static(zero_threshold),
            parallelism,
            stack,
        );
    } else {
        unimplemented!("only real and complex values are supported in the svd")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{c64, Mat};

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn test_real_big() {
        for (m, n) in [(3, 2), (2, 2), (4, 4), (15, 10), (10, 10), (15, 15)] {
            let mat = Mat::with_dims(|_, _| rand::random::<f64>(), m, n);
            let size = m.min(n);

            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            compute_svd_big(
                mat.as_ref(),
                s.as_mut().submatrix(0, 0, size, size).diagonal(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                compute_bidiag_real_svd::<f64>,
                f64::EPSILON,
                f64::MIN_POSITIVE,
                Parallelism::None,
                make_stack!(compute_svd_big_req::<f64>(
                    m,
                    n,
                    ComputeVectors::Full,
                    ComputeVectors::Full,
                    bidiag_real_svd_req::<f64>,
                    Parallelism::None,
                )),
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
    fn test_real_identity() {
        for (m, n) in [(15, 10), (10, 10), (15, 15)] {
            let mut mat = Mat::zeros(m, n);
            let size = m.min(n);
            for i in 0..size {
                mat[(i, i)] = 1.0;
            }

            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            compute_svd_big(
                mat.as_ref(),
                s.as_mut().submatrix(0, 0, size, size).diagonal(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                compute_bidiag_real_svd::<f64>,
                f64::EPSILON,
                f64::MIN_POSITIVE,
                Parallelism::None,
                make_stack!(compute_svd_big_req::<f64>(
                    m,
                    n,
                    ComputeVectors::Full,
                    ComputeVectors::Full,
                    bidiag_real_svd_req::<f64>,
                    Parallelism::None,
                )),
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
    fn test_real_zero() {
        for (m, n) in [(15, 10), (10, 10), (15, 15)] {
            let mat = Mat::zeros(m, n);
            let size = m.min(n);

            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            compute_svd_big(
                mat.as_ref(),
                s.as_mut().submatrix(0, 0, size, size).diagonal(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                compute_bidiag_real_svd::<f64>,
                f64::EPSILON,
                f64::MIN_POSITIVE,
                Parallelism::None,
                make_stack!(compute_svd_big_req::<f64>(
                    m,
                    n,
                    ComputeVectors::Full,
                    ComputeVectors::Full,
                    bidiag_real_svd_req::<f64>,
                    Parallelism::None,
                )),
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
        for (m, n) in [(4, 4), (5, 5), (15, 10), (10, 10), (15, 15)] {
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
                )),
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
        for m in 0..20 {
            for n in 0..20 {
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
                    )),
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
                            )),
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
                            )),
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

    #[test]
    fn test_cplx() {
        for m in 0..20 {
            for n in 0..20 {
                let mat = Mat::with_dims(|_, _| c64::new(rand::random(), rand::random()), m, n);
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
                    make_stack!(compute_svd_req::<c64>(
                        m,
                        n,
                        ComputeVectors::Full,
                        ComputeVectors::Full,
                        Parallelism::None,
                        SvdParams::default(),
                    )),
                    SvdParams::default(),
                );

                let reconstructed = &u * &s * v.adjoint();

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(reconstructed[(i, j)], mat[(i, j)], 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_cplx_thin() {
        for m in 0..20 {
            for n in 0..20 {
                use ComputeVectors::*;
                for compute_u in [No, Thin, Full] {
                    for compute_v in [No, Thin, Full] {
                        dbg!(m, n, compute_u, compute_v);
                        let mat =
                            Mat::with_dims(|_, _| c64::new(rand::random(), rand::random()), m, n);
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
                            make_stack!(compute_svd_req::<c64>(
                                m,
                                n,
                                compute_u,
                                compute_v,
                                Parallelism::None,
                                SvdParams::default(),
                            )),
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
                            make_stack!(compute_svd_req::<c64>(
                                m,
                                n,
                                ComputeVectors::Full,
                                ComputeVectors::Full,
                                Parallelism::None,
                                SvdParams::default(),
                            )),
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

    #[test]
    fn test_cplx_identity() {
        for (m, n) in [(15, 10), (10, 10), (15, 15)] {
            let mut mat = Mat::zeros(m, n);
            let size = m.min(n);
            for i in 0..size {
                mat[(i, i)] = c64::one();
            }

            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            compute_svd_big(
                mat.as_ref(),
                s.as_mut().submatrix(0, 0, size, size).diagonal(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                compute_bidiag_cplx_svd::<f64>,
                f64::EPSILON,
                f64::MIN_POSITIVE,
                Parallelism::None,
                make_stack!(compute_svd_big_req::<c64>(
                    m,
                    n,
                    ComputeVectors::Full,
                    ComputeVectors::Full,
                    bidiag_cplx_svd_req::<f64>,
                    Parallelism::None,
                )),
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
    fn test_cplx_zero() {
        for (m, n) in [(15, 10), (10, 10), (15, 15)] {
            let mat = Mat::zeros(m, n);
            let size = m.min(n);

            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            compute_svd_big(
                mat.as_ref(),
                s.as_mut().submatrix(0, 0, size, size).diagonal(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                compute_bidiag_cplx_svd::<f64>,
                f64::EPSILON,
                f64::MIN_POSITIVE,
                Parallelism::None,
                make_stack!(compute_svd_big_req::<c64>(
                    m,
                    n,
                    ComputeVectors::Full,
                    ComputeVectors::Full,
                    bidiag_cplx_svd_req::<f64>,
                    Parallelism::None,
                )),
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
