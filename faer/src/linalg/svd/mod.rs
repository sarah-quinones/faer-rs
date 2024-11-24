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
use linalg::qr::no_pivoting::factor::QrParams;

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
    pub qr: QrParams,
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

            bidiag: auto!(T),
            qr: auto!(T),
            non_exhaustive: NonExhaustive(()),
        }
    }
}

fn svd_imp_scratch<T: ComplexField>(
    m: usize,
    n: usize,
    compute_u: ComputeSvdVectors,
    compute_v: ComputeSvdVectors,

    bidiag_svd_scratch: fn(
        n: usize,
        compute_u: bool,
        compute_v: bool,
        par: Par,
        params: SvdParams,
    ) -> Result<StackReq, SizeOverflow>,

    params: SvdParams,

    par: Par,
) -> Result<StackReq, SizeOverflow> {
    assert!(m >= n);

    let householder_blocksize = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(m, n);
    let bid = temp_mat_scratch::<T>(m, n)?;
    let householder_left = temp_mat_scratch::<T>(householder_blocksize, n)?;
    let householder_right = temp_mat_scratch::<T>(householder_blocksize, n)?;

    let compute_bidiag = bidiag::bidiag_in_place_scratch::<T>(m, n, par, params.bidiag)?;
    let diag = temp_mat_scratch::<T>(n, 1)?;
    let subdiag = diag;
    let compute_ub = compute_v != ComputeSvdVectors::No;
    let compute_vb = compute_u != ComputeSvdVectors::No;
    let u_b = temp_mat_scratch::<T>(if compute_ub { n + 1 } else { 2 }, n + 1)?;
    let v_b = temp_mat_scratch::<T>(n, if compute_vb { n } else { 0 })?;

    let compute_bidiag_svd = bidiag_svd_scratch(n, compute_ub, compute_vb, par, params)?;

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

fn bidiag_cplx_svd_scratch<T: ComplexField>(
    n: usize,
    compute_u: bool,
    compute_v: bool,
    par: Par,
    params: SvdParams,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        temp_mat_scratch::<T>(n, 1)?.try_array(4)?,
        temp_mat_scratch::<T::Real>(n + 1, if compute_u { n + 1 } else { 0 })?,
        temp_mat_scratch::<T::Real>(n, if compute_v { n } else { 0 })?,
        bidiag_real_svd_scratch::<T::Real>(n, compute_u, compute_v, par, params)?,
    ])
}

fn bidiag_real_svd_scratch<T: RealField>(
    n: usize,
    compute_u: bool,
    compute_v: bool,
    par: Par,
    params: SvdParams,
) -> Result<StackReq, SizeOverflow> {
    if n < params.recursion_threshold {
        Ok(StackReq::empty())
    } else {
        StackReq::try_all_of([
            temp_mat_scratch::<T>(2, if compute_u { 0 } else { n + 1 })?,
            bidiag_svd::divide_and_conquer_scratch::<T>(
                n,
                params.recursion_threshold,
                compute_u,
                compute_v,
                par,
            )?,
        ])
    }
}

#[math]
fn compute_bidiag_cplx_svd<T: ComplexField>(
    mut diag: ColMut<'_, T, usize, ContiguousFwd>,
    subdiag: ColMut<'_, T, usize, ContiguousFwd>,
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
                None => bidiag_svd::MatU::TwoRowsStorage(u2.as_mat_mut()),
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
fn svd_imp<T: ComplexField>(
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
        __dbg!(&bid);
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
        diag[i] = conj(bid[(i, i)]);
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
            Conj::Yes,
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
        svd_imp::<T::Real>(
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
        svd_imp::<T>(
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

pub fn svd_scratch<T: ComplexField>(
    nrows: usize,
    ncols: usize,
    compute_u: ComputeSvdVectors,
    compute_v: ComputeSvdVectors,
    par: Par,
    params: SvdParams,
) -> Result<StackReq, SizeOverflow> {
    let mut m = nrows;
    let mut n = ncols;
    let mut compute_u = compute_u;
    let mut compute_v = compute_v;

    if n > m {
        core::mem::swap(&mut m, &mut n);
        core::mem::swap(&mut compute_u, &mut compute_v);
    }

    if n == 0 {
        return Ok(StackReq::empty());
    }

    let bidiag_svd_scratch = if const { T::IS_REAL } {
        bidiag_real_svd_scratch::<T::Real>
    } else {
        bidiag_cplx_svd_scratch::<T>
    };

    if m as f64 / n as f64 <= params.qr_ratio_threshold {
        svd_imp_scratch::<T>(m, n, compute_u, compute_v, bidiag_svd_scratch, params, par)
    } else {
        let bs = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(m, n);
        StackReq::try_all_of([
            temp_mat_scratch::<T>(m, n)?,
            temp_mat_scratch::<T>(bs, n)?,
            StackReq::try_any_of([
                StackReq::try_all_of([
                    temp_mat_scratch::<T>(n, n)?,
                    svd_imp_scratch::<T>(
                        n,
                        n,
                        compute_u,
                        compute_v,
                        bidiag_svd_scratch,
                        params,
                        par,
                    )?,
                ])?,
                linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<
                    T,
                >(
                    m,
                    bs,
                    match compute_u {
                        ComputeSvdVectors::No => 0,
                        ComputeSvdVectors::Thin => n,
                        ComputeSvdVectors::Full => m,
                    },
                )?,
            ])?,
        ])
    }
}

#[math]
pub fn svd<T: ComplexField>(
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
    let do_transpose = n > m;
    if do_transpose {
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
        compute_squareish_svd(matrix, s, u.rb_mut(), v.rb_mut(), par, stack, params)?;
    } else {
        let bs = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(m, n);
        let (mut qr, stack) = unsafe { temp_mat_uninit::<T, _, _>(m, n, stack) };
        let mut qr = qr.as_mat_mut();
        let (mut householder, stack) = unsafe { temp_mat_uninit::<T, _, _>(bs, n, stack) };
        let mut householder = householder.as_mat_mut();

        {
            qr.copy_from(matrix.rb());
            linalg::qr::no_pivoting::factor::qr_in_place(
                qr.rb_mut(),
                householder.rb_mut(),
                par,
                stack,
                params.qr,
            );
        }

        {
            let (mut r, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
            let mut r = r.as_mat_mut();
            z!(r.rb_mut())
                .for_each_triangular_lower(linalg::zip::Diag::Skip, |uz!(dst)| *dst = zero());
            z!(r.rb_mut(), qr.rb().submatrix(0, 0, n, n))
                .for_each_triangular_upper(linalg::zip::Diag::Include, |uz!(dst, src)| {
                    *dst = copy(*src)
                });

            // r = u s v
            compute_squareish_svd(
                r.rb(),
                s,
                u.rb_mut().map(|u| u.submatrix_mut(0, 0, n, n)),
                v.rb_mut(),
                par,
                stack,
                params,
            )?;
        }

        // matrix = q u s v
        if let Some(mut u) = u.rb_mut() {
            u.rb_mut().subrows_mut(n, m - n).fill(zero());
            if u.ncols() == m {
                u.rb_mut()
                    .submatrix_mut(n, n, m - n, m - n)
                    .diagonal_mut()
                    .fill(one());
            }

            linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                qr.rb(),
                householder.rb(),
                Conj::No,
                u.rb_mut(),
                par,
                stack,
            );
        }
    }

    if do_transpose {
        // conjugate u and v
        if let Some(u) = u.rb_mut() {
            z!(u).for_each(|uz!(u)| *u = conj(*u))
        }
        if let Some(v) = v.rb_mut() {
            z!(v).for_each(|uz!(v)| *v = conj(*v))
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, stats::prelude::*, utils::approx::*};
    use dyn_stack::GlobalMemBuffer;

    #[track_caller]
    fn test_svd<T: ComplexField>(mat: MatRef<'_, T>) {
        let (m, n) = mat.shape();
        let params = SvdParams {
            recursion_threshold: 8,
            qr_ratio_threshold: 1.0,
            ..auto!(T)
        };
        use faer_traits::math_utils::*;
        let approx_eq =
            CwiseMat(ApproxEq::<T>::eps() * sqrt(&from_f64(8.0 * Ord::max(m, n) as f64)));

        {
            let mut s = Mat::zeros(m, n);
            let mut u = Mat::zeros(m, m);
            let mut v = Mat::zeros(n, n);

            svd(
                mat.as_ref(),
                s.as_mut().diagonal_mut().column_vector_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    svd_scratch::<T>(
                        m,
                        n,
                        ComputeSvdVectors::Full,
                        ComputeSvdVectors::Full,
                        Par::Seq,
                        params,
                    )
                    .unwrap(),
                )),
                params,
            )
            .unwrap();

            let reconstructed = &u * &s * v.adjoint();
            assert!(reconstructed ~ mat);
        }

        let size = Ord::min(m, n);
        let mut s = Mat::zeros(size, size);
        {
            let mut u = Mat::zeros(m, size);
            let mut v = Mat::zeros(n, size);

            svd(
                mat.as_ref(),
                s.as_mut().diagonal_mut().column_vector_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    svd_scratch::<T>(
                        m,
                        n,
                        ComputeSvdVectors::Thin,
                        ComputeSvdVectors::Thin,
                        Par::Seq,
                        params,
                    )
                    .unwrap(),
                )),
                params,
            )
            .unwrap();

            let reconstructed = &u * &s * v.adjoint();
            assert!(reconstructed ~ mat);
        }
        {
            let mut s2 = Mat::zeros(size, size);

            svd(
                mat.as_ref(),
                s2.as_mut().diagonal_mut().column_vector_mut(),
                None,
                None,
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    svd_scratch::<T>(
                        m,
                        n,
                        ComputeSvdVectors::No,
                        ComputeSvdVectors::No,
                        Par::Seq,
                        params,
                    )
                    .unwrap(),
                )),
                params,
            )
            .unwrap();

            assert!(s2 ~ s);
        }
    }

    #[test]
    fn test_real() {
        let rng = &mut StdRng::seed_from_u64(1);

        for (m, n) in [
            (3, 2),
            (2, 2),
            (4, 4),
            (15, 10),
            (10, 10),
            (15, 15),
            (50, 50),
            (100, 100),
            (150, 150),
            (150, 20),
            (20, 150),
        ] {
            let mat = CwiseMatDistribution {
                nrows: m,
                ncols: n,
                dist: StandardNormal,
            }
            .rand::<Mat<f64>>(rng);

            test_svd(mat.as_ref());
        }
    }

    #[test]
    fn test_cplx() {
        let rng = &mut StdRng::seed_from_u64(1);

        for (m, n) in [
            (1, 1),
            (2, 2),
            (3, 2),
            (2, 2),
            (3, 3),
            (4, 4),
            (15, 10),
            (10, 10),
            (15, 15),
            (16, 16),
            (17, 17),
            (18, 18),
            (19, 19),
            (20, 20),
            (30, 30),
            (50, 50),
            (100, 100),
            (150, 150),
            (150, 20),
            (20, 150),
        ] {
            let mat = CwiseMatDistribution {
                nrows: m,
                ncols: n,
                dist: ComplexDistribution::new(StandardNormal, StandardNormal),
            }
            .rand::<Mat<c64>>(rng);

            test_svd(mat.as_ref());
        }
    }

    #[test]
    fn test_special() {
        for (m, n) in [
            (3, 2),
            (2, 2),
            (4, 4),
            (15, 10),
            (10, 10),
            (15, 15),
            (50, 50),
            (100, 100),
            (150, 150),
            (150, 20),
            (20, 150),
        ] {
            dbg!(m, n);
            test_svd(Mat::<f64>::zeros(m, n).as_ref());
            test_svd(Mat::<c64>::zeros(m, n).as_ref());
            test_svd(Mat::<f64>::full(m, n, 1.0).as_ref());
            test_svd(Mat::<c64>::full(m, n, c64::ONE).as_ref());
            test_svd(Mat::<f64>::identity(m, n).as_ref());
            test_svd(Mat::<c64>::identity(m, n).as_ref());
        }
    }
}
