//! The SVD of a matrix $M$ of shape $(m, n)$ is a decomposition into three components $U$, $S$,
//! and $V$, such that:
//!
//! - $U$ has shape $(m, m)$ and is a unitary matrix,
//! - $V$ has shape $(n, n)$ and is a unitary matrix,
//! - $S$ has shape $(m, n)$ and is zero everywhere except the main diagonal,
//! - and finally:
//!
//! $$M = U S V^H.$$

use assert2::assert;
use bidiag_real_svd::bidiag_real_svd_req;
use coe::Coerce;
use core::mem::swap;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    householder::{
        apply_block_householder_sequence_on_the_left_in_place_req,
        apply_block_householder_sequence_on_the_left_in_place_with_conj,
        upgrade_householder_factor,
    },
    temp_mat_req, temp_mat_uninit,
    zip::Diag,
    zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism, RealField,
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

fn compute_real_svd_small_req<E: Entity>(
    m: usize,
    n: usize,
    compute_u: ComputeVectors,
    compute_v: ComputeVectors,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    assert!(m >= n);

    if m == n {
        return temp_mat_req::<E>(m, n);
    }

    let _ = compute_v;
    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<E>(m, n);

    let qr = temp_mat_req::<E>(m, n)?;
    let householder = temp_mat_req::<E>(householder_blocksize, n)?;
    let r = temp_mat_req::<E>(n, n)?;

    let compute_qr = faer_qr::no_pivoting::compute::qr_in_place_req::<E>(
        m,
        n,
        householder_blocksize,
        parallelism,
        Default::default(),
    )?;

    let apply_householder = apply_block_householder_sequence_on_the_left_in_place_req::<E>(
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

fn compute_svd_big_req<E: Entity>(
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
    assert!(m >= n);
    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<E>(m, n);

    let bid = temp_mat_req::<E>(m, n)?;
    let householder_left = temp_mat_req::<E>(householder_blocksize, n)?;
    let householder_right = temp_mat_req::<E>(householder_blocksize, n - 1)?;

    let compute_bidiag = bidiag::bidiagonalize_in_place_req::<E>(m, n, parallelism)?;

    let diag = StackReq::try_new::<E>(n)?;
    let subdiag = diag;
    let compute_ub = compute_v != ComputeVectors::No;
    let compute_vb = compute_u != ComputeVectors::No;
    let u_b = temp_mat_req::<E>(if compute_ub { n + 1 } else { 2 }, n + 1)?;
    let v_b = temp_mat_req::<E>(n, if compute_vb { n } else { 0 })?;

    let compute_bidiag_svd = bidiag_svd_req(
        n,
        JACOBI_FALLBACK_THRESHOLD,
        compute_ub,
        compute_vb,
        parallelism,
    )?;
    let apply_householder_u = apply_block_householder_sequence_on_the_left_in_place_req::<E>(
        m,
        householder_blocksize,
        match compute_u {
            ComputeVectors::No => 0,
            ComputeVectors::Thin => n,
            ComputeVectors::Full => m,
        },
    )?;
    let apply_householder_v = apply_block_householder_sequence_on_the_left_in_place_req::<E>(
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
fn compute_real_svd_small<E: RealField>(
    matrix: MatRef<'_, E>,
    s: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    v: Option<MatMut<'_, E>>,
    epsilon: E,
    zero_threshold: E,
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
        let (mut jacobi_mat, _) = unsafe { temp_mat_uninit::<E>(m, n, stack) };
        let mut jacobi_mat = jacobi_mat.as_mut();
        zipped!(jacobi_mat.rb_mut(), matrix).for_each(|mut dst, src| dst.write(src.read()));

        jacobi::jacobi_svd(
            jacobi_mat.rb_mut(),
            u,
            v,
            jacobi::Skip::None,
            epsilon,
            zero_threshold,
        );
        zipped!(s, jacobi_mat.rb().diagonal()).for_each(|mut dst, src| dst.write(src.read()));
        return;
    }

    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<E>(m, n);

    let (mut qr, stack) = unsafe { temp_mat_uninit::<E>(m, n, stack) };
    let (mut householder, mut stack) =
        unsafe { temp_mat_uninit::<E>(householder_blocksize, n, stack) };
    let mut qr = qr.as_mut();
    let mut householder = householder.as_mut();

    {
        let (mut r, mut stack) = unsafe { temp_mat_uninit::<E>(n, n, stack.rb_mut()) };
        let mut r = r.as_mut();

        zipped!(qr.rb_mut(), matrix).for_each(|mut dst, src| dst.write(src.read()));

        // matrix = q * r
        faer_qr::no_pivoting::compute::qr_in_place(
            qr.rb_mut(),
            householder.rb_mut(),
            parallelism,
            stack.rb_mut(),
            Default::default(),
        );
        zipped!(r.rb_mut()).for_each_triangular_lower(Diag::Skip, |mut dst| dst.write(E::zero()));
        zipped!(r.rb_mut(), qr.rb().submatrix(0, 0, n, n))
            .for_each_triangular_upper(Diag::Include, |mut dst, src| dst.write(src.read()));

        // r = u s v
        jacobi::jacobi_svd(
            r.rb_mut(),
            u.rb_mut().map(|u| u.submatrix(0, 0, n, n)),
            v.rb_mut(),
            jacobi::Skip::None,
            epsilon,
            zero_threshold,
        );
        zipped!(s, r.rb().diagonal()).for_each(|mut dst, src| dst.write(src.read()));
    }

    // matrix = q u s v
    if let Some(mut u) = u.rb_mut() {
        let ncols = u.ncols();
        zipped!(u.rb_mut().submatrix(n, 0, m - n, n)).for_each(|mut dst| dst.write(E::zero()));
        zipped!(u.rb_mut().submatrix(0, n, m, ncols - n)).for_each(|mut dst| dst.write(E::zero()));
        if ncols == m {
            zipped!(u.rb_mut().submatrix(n, n, m - n, m - n).diagonal())
                .for_each(|mut dst| dst.write(E::one()));
        }

        faer_core::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr.rb(),
            householder.rb(),
            Conj::No,
            u,
            parallelism,
            stack.rb_mut(),
        );
    }
}

fn compute_bidiag_cplx_svd<E: ComplexField>(
    diag: &mut [E],
    subdiag: &mut [E],
    mut u: Option<MatMut<'_, E>>,
    mut v: Option<MatMut<'_, E>>,
    jacobi_fallback_threshold: usize,
    epsilon: E::Real,
    consider_zero_threshold: E::Real,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let n = diag.len();
    let (mut u_real, stack) =
        unsafe { temp_mat_uninit::<E::Real>(n + 1, if u.is_some() { n + 1 } else { 0 }, stack) };
    let mut u_real = u_real.as_mut();
    let (mut v_real, stack) =
        unsafe { temp_mat_uninit::<E::Real>(n, if v.is_some() { n } else { 0 }, stack) };
    let mut v_real = v_real.as_mut();
    let (mut diag_real, stack) = stack.collect(diag.iter().map(|x| x.abs()));
    let (mut subdiag_real, stack) = stack.collect(subdiag.iter().map(|x| x.abs()));

    let (mut col_mul, stack) = stack.make_with(n, |_| E::zero());
    let (mut row_mul, stack) = stack.make_with(n - 1, |_| E::zero());

    let normalized = |x: E| {
        if x == E::zero() {
            E::one()
        } else {
            let re = x.real().abs();
            let im = x.imag().abs();
            let max = if re > im { re } else { im };
            let x = x.scale_real(&max.inv());
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
        diag[i] = E::from_real(diag_real[i].clone());
    }

    let u_real = u_real.rb();
    let v_real = v_real.rb();

    if let Some(mut u) = u.rb_mut() {
        zipped!(u.rb_mut().row(0), u_real.row(0))
            .for_each(|mut u, u_real| u.write(E::from_real(u_real.read())));
        zipped!(u.rb_mut().row(n), u_real.row(n))
            .for_each(|mut u, u_real| u.write(E::from_real(u_real.read())));

        for col_idx in 0..u.ncols() {
            let mut u = u.rb_mut().col(col_idx).subrows(1, n - 1);
            let u_real = u_real.col(col_idx).subrows(1, n - 1);

            assert!(row_mul.len() == n - 1);
            unsafe {
                for i in 0..n - 1 {
                    u.write_unchecked(i, 0, row_mul[i].scale_real(&u_real.read_unchecked(i, 0)));
                }
            }
        }
    }
    if let Some(mut v) = v.rb_mut() {
        for col_idx in 0..v.ncols() {
            let mut v = v.rb_mut().col(col_idx);
            let v_real = v_real.col(col_idx);

            assert!(col_mul.len() == n);
            unsafe {
                for i in 0..n {
                    v.write_unchecked(i, 0, col_mul[i].scale_real(&v_real.read_unchecked(i, 0)));
                }
            }
        }
    }
}

fn bidiag_cplx_svd_req<E: Entity>(
    n: usize,
    jacobi_fallback_threshold: usize,
    compute_u: bool,
    compute_v: bool,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        temp_mat_req::<E>(n + 1, if compute_u { n + 1 } else { 0 })?,
        temp_mat_req::<E>(n, if compute_u { n } else { 0 })?,
        StackReq::try_new::<E>(n)?,
        StackReq::try_new::<E>(n)?,
        StackReq::try_new::<Complex<E>>(n)?,
        StackReq::try_new::<Complex<E>>(n - 1)?,
        bidiag_real_svd_req::<E>(
            n,
            jacobi_fallback_threshold,
            compute_u,
            compute_v,
            parallelism,
        )?,
    ])
}

/// does bidiagonilization -> divide conquer svd
fn compute_svd_big<E: ComplexField>(
    matrix: MatRef<'_, E>,
    mut s: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    v: Option<MatMut<'_, E>>,
    bidiag_svd: fn(
        diag: &mut [E],
        subdiag: &mut [E],
        u: Option<MatMut<'_, E>>,
        v: Option<MatMut<'_, E>>,
        jacobi_fallback_threshold: usize,
        epsilon: E::Real,
        consider_zero_threshold: E::Real,
        parallelism: Parallelism,
        stack: DynStack<'_>,
    ),
    epsilon: E::Real,
    zero_threshold: E::Real,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut stack = stack;

    assert!(matrix.nrows() >= matrix.ncols());

    let m = matrix.nrows();
    let n = matrix.ncols();
    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<E>(m, n);

    let (mut bid, stack) = unsafe { temp_mat_uninit::<E>(m, n, stack.rb_mut()) };
    let mut bid = bid.as_mut();
    let (mut householder_left, stack) =
        unsafe { temp_mat_uninit::<E>(householder_blocksize, n, stack) };
    let mut householder_left = householder_left.as_mut();
    let (mut householder_right, mut stack) =
        unsafe { temp_mat_uninit::<E>(householder_blocksize, n - 1, stack) };
    let mut householder_right = householder_right.as_mut();

    zipped!(bid.rb_mut(), matrix).for_each(|mut dst, src| dst.write(src.read()));

    bidiag::bidiagonalize_in_place(
        bid.rb_mut(),
        householder_left.rb_mut().row(0).transpose(),
        householder_right.rb_mut().row(0).transpose(),
        parallelism,
        stack.rb_mut(),
    );

    let bid = bid.into_const();

    let (mut diag, stack) = stack.make_with(n, |i| bid.read(i, i).conj());
    let (mut subdiag, stack) = stack.make_with(n, |i| {
        if i < n - 1 {
            bid.read(i, i + 1).conj()
        } else {
            E::zero()
        }
    });

    let mut j_base = 0;
    while j_base < n {
        let bs = Ord::min(householder_blocksize, n - j_base);
        let mut householder = householder_left.rb_mut().submatrix(0, j_base, bs, bs);
        let essentials = bid.submatrix(j_base, j_base, m - j_base, bs);
        for j in 0..bs {
            householder.write(j, j, householder.read(0, j));
        }
        upgrade_householder_factor(householder, essentials, bs, 1, parallelism);
        j_base += bs;
    }
    let mut j_base = 0;
    while j_base < n - 1 {
        let bs = Ord::min(householder_blocksize, n - 1 - j_base);
        let mut householder = householder_right.rb_mut().submatrix(0, j_base, bs, bs);
        let full_essentials = bid.submatrix(0, 1, m, n - 1).transpose();
        let essentials = full_essentials.submatrix(j_base, j_base, n - 1 - j_base, bs);
        for j in 0..bs {
            householder.write(j, j, householder.read(0, j));
        }
        upgrade_householder_factor(householder, essentials, bs, 1, parallelism);
        j_base += bs;
    }

    let (mut u_b, stack) =
        unsafe { temp_mat_uninit::<E>(if v.is_some() { n + 1 } else { 0 }, n + 1, stack) };
    let mut u_b = u_b.as_mut();
    let (mut v_b, mut stack) =
        unsafe { temp_mat_uninit::<E>(n, if u.is_some() { n } else { 0 }, stack) };
    let mut v_b = v_b.as_mut();

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

    for idx in 0..s.nrows() {
        s.write(idx, 0, diag[idx].clone());
    }

    if let Some(mut u) = u {
        let ncols = u.ncols();
        zipped!(
            u.rb_mut().submatrix(0, 0, n, n),
            v_b.rb().submatrix(0, 0, n, n),
        )
        .for_each(|mut dst, src| dst.write(src.read()));

        zipped!(u.rb_mut().submatrix(n, 0, m - n, ncols)).for_each(|mut x| x.write(E::zero()));
        zipped!(u.rb_mut().submatrix(0, n, n, ncols - n)).for_each(|mut x| x.write(E::zero()));
        zipped!(u.rb_mut().submatrix(n, n, ncols - n, ncols - n).diagonal())
            .for_each(|mut x| x.write(E::one()));

        apply_block_householder_sequence_on_the_left_in_place_with_conj(
            bid,
            householder_left.rb(),
            Conj::No,
            u,
            parallelism,
            stack.rb_mut(),
        );
    };
    if let Some(mut v) = v {
        zipped!(
            v.rb_mut().submatrix(0, 0, n, n),
            u_b.rb().submatrix(0, 0, n, n),
        )
        .for_each(|mut dst, src| dst.write(src.read()));

        let (mut bid_col_major, mut stack) =
            unsafe { faer_core::temp_mat_uninit::<E>(n - 1, m, stack.rb_mut()) };
        let mut bid_col_major = bid_col_major.as_mut();
        zipped!(
            bid_col_major.rb_mut(),
            bid.submatrix(0, 1, m, n - 1).transpose()
        )
        .for_each_triangular_lower(faer_core::zip::Diag::Skip, |mut dst, src| {
            dst.write(src.read())
        });

        apply_block_householder_sequence_on_the_left_in_place_with_conj(
            bid_col_major.rb(),
            householder_right.rb(),
            Conj::No,
            v.submatrix(1, 0, n - 1, n),
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
pub fn compute_svd_req<E: ComplexField>(
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

    let size = Ord::min(nrows, ncols);
    let skip_qr = nrows as f64 / ncols as f64 <= 11.0 / 6.0;
    let (svd_nrows, svd_ncols) = if skip_qr {
        (nrows, ncols)
    } else {
        (size, size)
    };

    let _ = params;
    let squareish_svd = if coe::is_same::<E, E::Real>() {
        if size <= JACOBI_FALLBACK_THRESHOLD {
            compute_real_svd_small_req::<E>(svd_nrows, svd_ncols, compute_u, compute_v, parallelism)
        } else {
            compute_svd_big_req::<E::Real>(
                svd_nrows,
                svd_ncols,
                compute_u,
                compute_v,
                bidiag_real_svd_req::<E::Real>,
                parallelism,
            )
        }
    } else {
        compute_svd_big_req::<E>(
            svd_nrows,
            svd_ncols,
            compute_u,
            compute_v,
            bidiag_cplx_svd_req::<E>,
            parallelism,
        )
    }?;

    if skip_qr {
        Ok(squareish_svd)
    } else {
        let householder_blocksize =
            faer_qr::no_pivoting::compute::recommended_blocksize::<E>(nrows, ncols);

        StackReq::try_all_of([
            temp_mat_req::<E>(nrows, ncols)?,
            temp_mat_req::<E>(householder_blocksize, ncols)?,
            StackReq::try_any_of([
                StackReq::try_all_of([
                    temp_mat_req::<E>(size, size)?,
                    StackReq::try_any_of([
                        faer_qr::no_pivoting::compute::qr_in_place_req::<E>(
                            nrows,
                            ncols,
                            householder_blocksize,
                            parallelism,
                            Default::default(),
                        )?,
                        squareish_svd,
                    ])?,
                ])?,
                apply_block_householder_sequence_on_the_left_in_place_req::<E>(
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
///
/// # Panics
/// Panics if any of the conditions described above is violated, or if the type `E` does not have a
/// fixed precision at compile time, e.g. a dynamic multiprecision floating point type.
///
/// This can also panic if the provided memory in `stack` is insufficient (see [`compute_svd_req`]).
#[track_caller]
pub fn compute_svd<E: ComplexField>(
    matrix: MatRef<'_, E>,
    s: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    v: Option<MatMut<'_, E>>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: SvdParams,
) {
    compute_svd_custom_epsilon(
        matrix,
        s,
        u,
        v,
        E::Real::epsilon().unwrap(),
        E::Real::zero_threshold().unwrap(),
        parallelism,
        stack,
        params,
    );
}

/// See [`compute_svd`].
///
/// This function takes an additional `epsilon` and `zero_threshold` parameters. `epsilon`
/// represents the precision of the values in the matrix, and `zero_threshold` is the value below
/// which the precision starts to deteriorate, e.g. due to denormalized numbers.
///
/// These values need to be provided manually for types that do not have a known precision at
/// compile time, e.g. a dynamic multiprecision floating point type.
#[track_caller]
pub fn compute_svd_custom_epsilon<E: ComplexField>(
    matrix: MatRef<'_, E>,
    s: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    v: Option<MatMut<'_, E>>,
    epsilon: E::Real,
    zero_threshold: E::Real,
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: SvdParams,
) {
    let size = Ord::min(matrix.nrows(), matrix.ncols());
    assert!(s.nrows() == size);
    assert!(s.ncols() == 1);
    if let Some(u) = u.rb() {
        assert!(u.nrows() == matrix.nrows());
        assert!(u.ncols() == matrix.nrows() || u.ncols() == size);
    }
    if let Some(v) = v.rb() {
        assert!(v.nrows() == matrix.ncols());
        assert!(v.ncols() == matrix.ncols() || v.ncols() == size);
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
            zipped!(u.rb_mut()).for_each(|mut dst| dst.write(E::zero()));
            zipped!(u.submatrix(0, 0, n, n).diagonal()).for_each(|mut dst| dst.write(E::one()));
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
        let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<E>(m, n);

        let (mut qr, stack) = unsafe { temp_mat_uninit::<E>(m, n, stack) };
        let mut qr = qr.as_mut();
        let (mut householder, mut stack) =
            unsafe { temp_mat_uninit::<E>(householder_blocksize, n, stack) };
        let mut householder = householder.as_mut();

        {
            let (mut r, mut stack) = unsafe { temp_mat_uninit::<E>(n, n, stack.rb_mut()) };
            let mut r = r.as_mut();

            zipped!(qr.rb_mut(), matrix).for_each(|mut dst, src| dst.write(src.read()));

            // matrix = q * r
            faer_qr::no_pivoting::compute::qr_in_place(
                qr.rb_mut(),
                householder.rb_mut(),
                parallelism,
                stack.rb_mut(),
                Default::default(),
            );
            zipped!(r.rb_mut())
                .for_each_triangular_lower(Diag::Skip, |mut dst| dst.write(E::zero()));
            zipped!(r.rb_mut(), qr.rb().submatrix(0, 0, n, n))
                .for_each_triangular_upper(Diag::Include, |mut dst, src| dst.write(src.read()));

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
            zipped!(u.rb_mut().submatrix(n, 0, m - n, n)).for_each(|mut dst| dst.write(E::zero()));
            zipped!(u.rb_mut().submatrix(0, n, m, ncols - n))
                .for_each(|mut dst| dst.write(E::zero()));
            if ncols == m {
                zipped!(u.rb_mut().submatrix(n, n, m - n, m - n).diagonal())
                    .for_each(|mut dst| dst.write(E::one()));
            }

            faer_core::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                qr.rb(),
                householder.rb(),
                Conj::No,
                u,
                parallelism,
                stack.rb_mut(),
            );
        }
    }

    if do_transpose {
        // conjugate u and v
        if let Some(u) = u {
            zipped!(u).for_each(|mut x| x.write(x.read().conj()))
        }
        if let Some(v) = v {
            zipped!(v).for_each(|mut x| x.write(x.read().conj()))
        }
    }
}

fn squareish_svd<E: ComplexField>(
    matrix: MatRef<E>,
    s: MatMut<E>,
    mut u: Option<MatMut<E>>,
    mut v: Option<MatMut<E>>,
    epsilon: E::Real,
    zero_threshold: E::Real,
    parallelism: Parallelism,
    stack: DynStack,
) {
    let size = matrix.ncols();
    if coe::is_same::<E, E::Real>() {
        if size <= JACOBI_FALLBACK_THRESHOLD {
            compute_real_svd_small::<E::Real>(
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
            compute_svd_big::<E::Real>(
                matrix.coerce(),
                s.coerce(),
                u.rb_mut().map(coe::Coerce::coerce),
                v.rb_mut().map(coe::Coerce::coerce),
                compute_bidiag_real_svd::<E::Real>,
                coe::coerce_static(epsilon),
                coe::coerce_static(zero_threshold),
                parallelism,
                stack,
            );
        }
    } else {
        compute_svd_big::<E>(
            matrix.coerce(),
            s,
            u,
            v,
            compute_bidiag_cplx_svd::<E>,
            coe::coerce_static(epsilon),
            coe::coerce_static(zero_threshold),
            parallelism,
            stack,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{c32, c64, Mat};

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn test_real_big() {
        for (m, n) in [(3, 2), (2, 2), (4, 4), (15, 10), (10, 10), (15, 15)] {
            let mat = Mat::with_dims(m, n, |_, _| rand::random::<f64>());
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
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
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
                mat.write(i, i, 1.0);
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
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
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
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_real_small() {
        for (m, n) in [(4, 4), (5, 5), (15, 10), (10, 10), (15, 15)] {
            let mat = Mat::with_dims(m, n, |_, _| rand::random::<f64>());
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
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_real() {
        for m in 0..20 {
            for n in 0..20 {
                let mat = Mat::with_dims(m, n, |_, _| rand::random::<f64>());
                let size = m.min(n);

                let mut s = Mat::zeros(m, n);
                let mut u = Mat::zeros(m, m);
                let mut v = Mat::zeros(n, n);

                compute_svd(
                    mat.as_ref(),
                    s.as_mut().submatrix(0, 0, size, size).diagonal(),
                    Some(u.as_mut()),
                    Some(v.as_mut()),
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
                        assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_real_f32() {
        for m in 0..20 {
            for n in 0..20 {
                let mat = Mat::with_dims(m, n, |_, _| rand::random::<f32>());
                let size = m.min(n);

                let mut s = Mat::zeros(m, n);
                let mut u = Mat::zeros(m, m);
                let mut v = Mat::zeros(n, n);

                compute_svd(
                    mat.as_ref(),
                    s.as_mut().submatrix(0, 0, size, size).diagonal(),
                    Some(u.as_mut()),
                    Some(v.as_mut()),
                    Parallelism::None,
                    make_stack!(compute_svd_req::<f32>(
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
                        assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-3);
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
                        let mat = Mat::with_dims(m, n, |_, _| rand::random::<f64>());
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
                                assert_approx_eq!(u.read(i, j), u_target.read(i, j), 1e-10);
                            }
                        }
                        for j in 0..v.ncols() {
                            for i in 0..v.nrows() {
                                assert_approx_eq!(v.read(i, j), v_target.read(i, j), 1e-10);
                            }
                        }
                        for j in 0..s.ncols() {
                            for i in 0..s.nrows() {
                                assert_approx_eq!(s.read(i, j), s_target.read(i, j), 1e-10);
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
                let mat = Mat::with_dims(m, n, |_, _| c64::new(rand::random(), rand::random()));
                let size = m.min(n);

                let mut s = Mat::zeros(m, n);
                let mut u = Mat::zeros(m, m);
                let mut v = Mat::zeros(n, n);

                compute_svd(
                    mat.as_ref(),
                    s.as_mut().submatrix(0, 0, size, size).diagonal(),
                    Some(u.as_mut()),
                    Some(v.as_mut()),
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
                        assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_cplx_f32() {
        for m in 0..20 {
            for n in 0..20 {
                let mat = Mat::with_dims(m, n, |_, _| c32::new(rand::random(), rand::random()));
                let size = m.min(n);

                let mut s = Mat::zeros(m, n);
                let mut u = Mat::zeros(m, m);
                let mut v = Mat::zeros(n, n);

                compute_svd(
                    mat.as_ref(),
                    s.as_mut().submatrix(0, 0, size, size).diagonal(),
                    Some(u.as_mut()),
                    Some(v.as_mut()),
                    Parallelism::None,
                    make_stack!(compute_svd_req::<c32>(
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
                        assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-3);
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
                            Mat::with_dims(m, n, |_, _| c64::new(rand::random(), rand::random()));
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
                                assert_approx_eq!(u.read(i, j), u_target.read(i, j), 1e-10);
                            }
                        }
                        for j in 0..v.ncols() {
                            for i in 0..v.nrows() {
                                assert_approx_eq!(v.read(i, j), v_target.read(i, j), 1e-10);
                            }
                        }
                        for j in 0..s.ncols() {
                            for i in 0..s.nrows() {
                                assert_approx_eq!(s.read(i, j), s_target.read(i, j), 1e-10);
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
                mat.write(i, i, c64::one());
            }

            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            compute_svd_big(
                mat.as_ref(),
                s.as_mut().submatrix(0, 0, size, size).diagonal(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                compute_bidiag_cplx_svd::<c64>,
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
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
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
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_real_ones() {
        for n in [1, 2, 4, 8, 64, 512] {
            for m in [1, 2, 4, 8, 64, 512] {
                let f = |_, _| 1f64;
                let mat = Mat::with_dims(m, n, f);
                let mut s = Mat::zeros(m, n);
                let mut u = Mat::zeros(m, m);
                let mut v = Mat::zeros(n, n);

                compute_svd(
                    mat.as_ref(),
                    s.as_mut().diagonal(),
                    Some(u.as_mut()),
                    Some(v.as_mut()),
                    faer_core::Parallelism::None,
                    make_stack!(compute_svd_req::<f64>(
                        m,
                        n,
                        ComputeVectors::Full,
                        ComputeVectors::Full,
                        faer_core::Parallelism::None,
                        Default::default(),
                    )),
                    Default::default(),
                );

                let reconstructed = &u * &s * v.transpose();

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_cplx_ones() {
        for n in [1, 2, 4, 8, 32, 64, 512] {
            for m in [1, 2, 4, 8, 32, 64, 512] {
                let f = |_, _| c64::new(1.0, 0.0);
                let mat = Mat::with_dims(m, n, f);
                let mut s = Mat::zeros(m, n);
                let mut u = Mat::zeros(m, m);
                let mut v = Mat::zeros(n, n);

                compute_svd(
                    mat.as_ref(),
                    s.as_mut().diagonal(),
                    Some(u.as_mut()),
                    Some(v.as_mut()),
                    faer_core::Parallelism::None,
                    make_stack!(compute_svd_req::<c64>(
                        m,
                        n,
                        ComputeVectors::Full,
                        ComputeVectors::Full,
                        faer_core::Parallelism::None,
                        Default::default(),
                    )),
                    Default::default(),
                );

                let reconstructed = &u * &s * v.transpose();

                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                    }
                }
            }
        }
    }
}
