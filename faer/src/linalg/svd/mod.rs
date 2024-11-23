//! Low level implementation of the SVD of a matrix.
//!
//! The SVD of a matrix $M$ of shape $(m, n)$ is a decomposition into three components $U$, $S$,
//! and $V$, such that:
//!
//! - $U$ has shape $(m, m)$ and is a unitary matrix,
//! - $V$ has shape $(n, n)$ and is a unitary matrix,
//! - $S$ has shape $(m, n)$ and is zero everywhere except the main diagonal,
//! - and finally:
//!
//! $$M = U S V^H.$$

use bidiag::BidiagParams;

use crate::{assert, internal_prelude::*};

pub mod bidiag;
pub(crate) mod bidiag_svd;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComputeSvdVectors {
    No,
    Thin,
    Full,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SvdError {
    NoConvergence,
}

#[derive(Debug, Copy, Clone)]
pub struct SvdParams {
    pub bidiag: BidiagParams,
    pub recursion_threshold: usize,
    pub qr_ratio_threshold: f64,

    #[doc(hidden)]
    pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for SvdParams {
    fn auto() -> Self {
        Self {
            recursion_threshold: 128,
            qr_ratio_threshold: 11.0 / 6.0,

            bidiag: Auto::<T>::auto(),
            non_exhaustive: NonExhaustive(()),
        }
    }
}

fn compute_svd_imp_scratch<T: ComplexField>(
    m: usize,
    n: usize,
    compute_u: ComputeSvdVectors,
    compute_v: ComputeSvdVectors,

    bidiag_svd_scratch: fn(
        n: usize,
        recursion_threshold: usize,
        compute_u: bool,
        compute_v: bool,
        par: Par,
    ) -> Result<StackReq, SizeOverflow>,

    params: SvdParams,

    par: Par,
) -> Result<StackReq, SizeOverflow> {
    assert!(m >= n);

    let householder_blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(m, n);
    let bid = temp_mat_scratch::<T>(m, n)?;
    let householder_left = temp_mat_scratch::<T>(householder_blocksize, n)?;
    let householder_right = temp_mat_scratch::<T>(householder_blocksize, n - 1)?;

    let compute_bidiag = bidiag::bidiag_in_place_scratch::<T>(m, n, par, params.bidiag)?;
    let diag = temp_mat_scratch::<T>(n, 1)?;
    let subdiag = diag;
    let compute_ub = compute_v != ComputeSvdVectors::No;
    let compute_vb = compute_u != ComputeSvdVectors::No;
    let u_b = temp_mat_scratch::<T>(if compute_ub { n + 1 } else { 2 }, n + 1)?;
    let v_b = temp_mat_scratch::<T>(n, if compute_vb { n } else { 0 })?;

    let compute_bidiag_svd =
        bidiag_svd_scratch(n, params.recursion_threshold, compute_ub, compute_vb, par)?;

    let apply_householder_u =
        linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
            m,
            householder_blocksize,
            match compute_u {
                ComputeSvdVectors::No => 0,
                ComputeSvdVectors::Thin => n,
                ComputeSvdVectors::Full => m,
            },
        )?;
    let apply_householder_v =
        linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
            n - 1,
            householder_blocksize,
            match compute_v {
                ComputeSvdVectors::No => 0,
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

#[math]
fn compute_bidiag_cplx_svd<T: ComplexField>(
    mut diag: ColMut<'_, T, usize, ContiguousFwd>,
    mut subdiag: ColMut<'_, T, usize, ContiguousFwd>,
    mut u: Option<MatMut<'_, T>>,
    mut v: Option<MatMut<'_, T>>,
    params: SvdParams,
    par: Par,
    stack: &mut DynStack,
) -> Result<(), SvdError> {
    let n = diag.nrows();

    let (mut diag_real, stack) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
    let (mut subdiag_real, stack) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
    let (mut u_real, stack) = unsafe {
        temp_mat_uninit::<T::Real, _, _>(n + 1, if u.is_some() { n + 1 } else { 0 }, stack)
    };
    let (mut v_real, stack) =
        unsafe { temp_mat_uninit::<T::Real, _, _>(n, if v.is_some() { n } else { 0 }, stack) };

    let (mut col_mul, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
    let (mut row_mul, stack) = unsafe { temp_mat_uninit::<T, _, _>(n - 1, 1, stack) };

    let mut u_real = u.rb().map(|_| u_real.as_mat_mut());
    let mut v_real = v.rb().map(|_| v_real.as_mat_mut());
    let mut diag_real = diag_real
        .as_mat_mut()
        .col_mut(0)
        .try_as_col_major_mut()
        .unwrap();
    let mut subdiag_real = subdiag_real
        .as_mat_mut()
        .col_mut(0)
        .try_as_col_major_mut()
        .unwrap();

    let mut col_mul = col_mul.as_mat_mut().col_mut(0);
    let mut row_mul = row_mul.as_mat_mut().col_mut(0);

    let normalized = |x: T| {
        if x == zero() {
            one()
        } else {
            let norm1 = max(abs(real(x)), abs(imag(x)));
            let y = x * from_real(recip(norm1));
            y * from_real(recip(abs(y)))
        }
    };

    let mut col_normalized = normalized(conj(diag[0]));
    col_mul[0] = copy(col_normalized);
    diag_real[0] = abs(diag[0]);
    subdiag_real[n - 1] = zero();
    for i in 1..n {
        let row_normalized = normalized(conj(subdiag[i - 1] * col_normalized));
        subdiag_real[i - 1] = abs(subdiag[i - 1]);
        row_mul[i - 1] = conj(row_normalized);

        col_normalized = normalized(conj(diag[i] * row_normalized));
        diag_real[i] = abs(diag[i]);
        col_mul[i] = copy(col_normalized);
    }

    compute_bidiag_real_svd(
        diag_real.rb_mut(),
        subdiag_real.rb_mut(),
        u_real.rb_mut(),
        v_real.rb_mut(),
        params,
        par,
        stack,
    )?;

    for i in 0..n {
        diag[i] = from_real(diag_real[i]);
    }

    let u_real = u_real.rb();
    let v_real = v_real.rb();

    if let (Some(mut u), Some(u_real)) = (u.rb_mut(), u_real) {
        z!(u.rb_mut().row_mut(0), u_real.row(0)).for_each(|uz!(u, r)| *u = from_real(*r));
        z!(u.rb_mut().row_mut(n), u_real.row(n)).for_each(|uz!(u, r)| *u = from_real(*r));

        for j in 0..u.ncols() {
            let mut u = u.rb_mut().col_mut(j).subrows_mut(1, n - 1);
            let u_real = u_real.rb().col(j).subrows(1, n - 1);
            z!(u.rb_mut(), u_real, row_mul.rb()).for_each(|uz!(u, re, f)| *u = mul_real(*f, *re));
        }
    }
    if let (Some(mut v), Some(v_real)) = (v.rb_mut(), v_real) {
        for j in 0..v.ncols() {
            let mut v = v.rb_mut().col_mut(j);
            let v_real = v_real.rb().col(j);
            z!(v.rb_mut(), v_real, col_mul.rb()).for_each(|uz!(v, re, f)| *v = mul_real(*f, *re));
        }
    }

    Ok(())
}

#[math]
fn compute_bidiag_real_svd<T: RealField>(
    mut diag: ColMut<'_, T, usize, ContiguousFwd>,
    mut subdiag: ColMut<'_, T, usize, ContiguousFwd>,
    mut u: Option<MatMut<'_, T, usize, usize>>,
    mut v: Option<MatMut<'_, T, usize, usize>>,
    params: SvdParams,
    par: Par,
    stack: &mut DynStack,
) -> Result<(), SvdError> {
    with_dim!(N, diag.nrows());
    for i in 0..*N {
        if !(is_finite(diag[i]) && is_finite(subdiag[i])) {
            return Err(SvdError::NoConvergence);
        }
    }

    if *N < params.recursion_threshold {
        bidiag_svd::qr_algorithm(
            diag.rb_mut().as_row_shape_mut(N),
            subdiag.rb_mut().as_row_shape_mut(N),
            u.rb_mut().map(|u| u.submatrix_mut(0, 0, N, N)),
            v.rb_mut().map(|v| v.as_shape_mut(N, N)),
        )?;

        if let Some(mut u) = u.rb_mut() {
            for i in N.indices() {
                u[(*N, *i)] = zero();
                u[(*i, *N)] = zero();
            }
            u[(*N, *N)] = one();
        }
        return Ok(());
    } else {
        let (mut u2, stack) = unsafe {
            temp_mat_uninit::<T::Real, _, _>(2, if u.is_some() { 0 } else { *N + 1 }, stack)
        };

        bidiag_svd::divide_and_conquer(
            diag.as_row_shape_mut(N),
            subdiag.as_row_shape_mut(N),
            match u {
                Some(u) => bidiag_svd::MatU::Full(u),
                None => bidiag_svd::MatU::TwoRows(u2.as_mat_mut()),
            },
            v.map(|m| m.as_shape_mut(N, N)),
            par,
            stack,
            params.recursion_threshold,
        )
    }
}

/// bidiag -> divide conquer svd / qr algo
#[math]
fn compute_svd_imp<T: ComplexField>(
    matrix: MatRef<'_, T>,
    s: ColMut<'_, T>,
    u: Option<MatMut<'_, T>>,
    v: Option<MatMut<'_, T>>,
    bidiag_svd: fn(
        diag: ColMut<'_, T, usize, ContiguousFwd>,
        subdiag: ColMut<'_, T, usize, ContiguousFwd>,
        u: Option<MatMut<'_, T, usize, usize>>,
        v: Option<MatMut<'_, T, usize, usize>>,
        params: SvdParams,
        par: Par,
        stack: &mut DynStack,
    ) -> Result<(), SvdError>,
    par: Par,
    stack: &mut DynStack,
    params: SvdParams,
) -> Result<(), SvdError> {
    assert!(matrix.nrows() >= matrix.ncols());
    let m = matrix.nrows();
    let n = matrix.ncols();

    let bs = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(m, n);

    let (mut bid, stack) = unsafe { temp_mat_uninit::<T, _, _>(m, n, stack) };
    let mut bid = bid.as_mat_mut();

    let (mut Hl, stack) = unsafe { temp_mat_uninit::<T, _, _>(bs, n, stack) };
    let (mut Hr, stack) = unsafe { temp_mat_uninit::<T, _, _>(bs, n, stack) };

    let mut Hl = Hl.as_mat_mut();
    let mut Hr = Hr.as_mat_mut();

    bid.copy_from(matrix);

    {
        with_dim!(M, m);
        with_dim!(N, n);
        with_dim!(BS, bs);

        bidiag::bidiag_in_place(
            bid.rb_mut().as_shape_mut(M, N),
            Hl.rb_mut().as_shape_mut(BS, N),
            Hr.rb_mut().as_shape_mut(BS, N),
            par,
            stack,
            params.bidiag,
        );
    }

    let (mut diag, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
    let (mut subdiag, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
    let mut diag = diag.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();
    let mut subdiag = subdiag
        .as_mat_mut()
        .col_mut(0)
        .try_as_col_major_mut()
        .unwrap();

    let (mut ub, stack) =
        unsafe { temp_mat_uninit::<T, _, _>(n + 1, if u.is_some() { n + 1 } else { 0 }, stack) };
    let (mut vb, stack) =
        unsafe { temp_mat_uninit::<T, _, _>(n, if v.is_some() { n } else { 0 }, stack) };

    let mut ub = ub.as_mat_mut();
    let mut vb = vb.as_mat_mut();

    for i in 0..n {
        diag[i] = copy(bid[(i, i)]);
        if i + 1 < n {
            subdiag[i] = conj(bid[(i, i + 1)]);
        } else {
            subdiag[i] = zero();
        }
    }

    bidiag_svd(
        diag.rb_mut(),
        subdiag.rb_mut(),
        v.rb().map(|_| ub.rb_mut()),
        u.rb().map(|_| vb.rb_mut()),
        params,
        par,
        stack,
    )?;

    { s }.copy_from(diag);

    if let Some(mut u) = u {
        let ncols = u.ncols();
        u.rb_mut().submatrix_mut(0, 0, n, n).copy_from(vb.rb());
        u.rb_mut().submatrix_mut(n, 0, m - n, ncols).fill(zero());
        u.rb_mut().submatrix_mut(0, n, n, ncols - n).fill(zero());
        u.rb_mut()
            .submatrix_mut(n, n, ncols - n, ncols - n)
            .diagonal_mut()
            .fill(one());

        linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            bid.rb(),
            Hl.rb(),
            Conj::No,
            u,
            par,
            stack,
        );
    }
    if let Some(mut v) = v {
        v.copy_from(ub.rb().submatrix(0, 0, n, n));

        for j in 1..n {
            for i in 0..j {
                bid[(j, i)] = copy(bid[(i, j)]);
            }
        }

        linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            bid.rb().subrows(1, n - 1).subcols(0, n - 1),
            Hr.rb().subcols(0, n - 1),
            Conj::No,
            v.subrows_mut(1, n - 1),
            par,
            stack,
        );
    }

    Ok(())
}

fn compute_squareish_svd<T: ComplexField>(
    matrix: MatRef<'_, T>,
    s: ColMut<'_, T>,
    u: Option<MatMut<'_, T>>,
    v: Option<MatMut<'_, T>>,
    par: Par,
    stack: &mut DynStack,
    params: SvdParams,
) -> Result<(), SvdError> {
    if const { T::IS_REAL } {
        compute_svd_imp::<T::Real>(
            unsafe { core::mem::transmute(matrix) },
            unsafe { core::mem::transmute(s) },
            unsafe { core::mem::transmute(u) },
            unsafe { core::mem::transmute(v) },
            compute_bidiag_real_svd::<T::Real>,
            par,
            stack,
            params,
        )
    } else {
        compute_svd_imp::<T>(
            matrix,
            s,
            u,
            v,
            compute_bidiag_cplx_svd::<T>,
            par,
            stack,
            params,
        )
    }
}

#[math]
pub fn compute_svd<T: ComplexField>(
    matrix: MatRef<'_, T>,
    s: ColMut<'_, T>,
    u: Option<MatMut<'_, T>>,
    v: Option<MatMut<'_, T>>,
    par: Par,
    stack: &mut DynStack,
    params: SvdParams,
) -> Result<(), SvdError> {
    let (m, n) = matrix.shape();
    let size = Ord::min(m, n);
    assert!(s.nrows() == size);

    if let Some(u) = u.rb() {
        assert!(all(
            u.nrows() == matrix.nrows(),
            any(u.ncols() == matrix.nrows(), u.ncols() == size),
        ));
    }
    if let Some(v) = v.rb() {
        assert!(all(
            v.nrows() == matrix.ncols(),
            any(v.ncols() == matrix.ncols(), v.ncols() == size),
        ));
    }

    #[cfg(feature = "perf-warn")]
    match (u.rb(), v.rb()) {
        (Some(matrix), _) | (_, Some(matrix)) => {
            if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(QR_WARN) {
                if matrix.col_stride().unsigned_abs() == 1 {
                    log::warn!(target: "faer_perf", "SVD prefers column-major singular vector matrices. Found row-major matrix.");
                } else {
                    log::warn!(target: "faer_perf", "SVD prefers column-major singular vector matrices. Found matrix with generic strides.");
                }
            }
        }
        _ => {}
    }

    let mut u = u;
    let mut v = v;
    let mut matrix = matrix;
    if n > m {
        matrix = matrix.transpose();
        core::mem::swap(&mut u, &mut v)
    }

    let (m, n) = matrix.shape();
    if n == 0 {
        if let Some(mut u) = u {
            u.fill(zero());
            u.rb_mut().diagonal_mut().fill(one());
        }
        return Ok(());
    }

    if m as f64 / n as f64 <= params.qr_ratio_threshold {
        compute_squareish_svd(matrix, s, u.rb_mut(), v.rb_mut(), par, stack, params)
    } else {
        compute_squareish_svd(matrix, s, u.rb_mut(), v.rb_mut(), par, stack, params)?;

        let bs = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(m, n);

        Ok(())
    }
}
