//! The eigenvalue decomposition of a square matrix $M$ of shape $(n, n)$ is a decomposition into
//! two components $U$, $S$:
//!
//! - $U$ has shape $(n, n)$ and is invertible,
//! - $S$ has shape $(n, n)$ and is a diagonal matrix,
//! - and finally:
//!
//! $$M = U S U^{-1}.$$
//!
//! If $M$ is hermitian, then $U$ can be made unitary ($U^{-1} = U^H$), and $S$ is real valued.

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
use assert2::assert;
use coe::Coerce;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    householder::{
        apply_block_householder_sequence_on_the_right_in_place_req,
        apply_block_householder_sequence_on_the_right_in_place_with_conj,
        upgrade_householder_factor,
    },
    mul::{
        inner_prod::inner_prod_with_conj,
        triangular::{self, BlockStructure},
    },
    temp_mat_req, temp_mat_uninit, temp_mat_zeroed, zipped, ComplexField, Conj, MatMut, MatRef,
    Parallelism, RealField,
};
use faer_qr::no_pivoting::compute::recommended_blocksize;
pub use hessenberg_cplx_evd::EvdParams;
use reborrow::*;

#[doc(hidden)]
pub mod tridiag_qr_algorithm;

#[doc(hidden)]
pub mod tridiag_real_evd;

#[doc(hidden)]
pub mod tridiag;

#[doc(hidden)]
pub mod hessenberg;

#[doc(hidden)]
pub mod hessenberg_cplx_evd;
#[doc(hidden)]
pub mod hessenberg_real_evd;

/// Indicates whether the eigenvectors are fully computed, partially computed, or skipped.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComputeVectors {
    No,
    Yes,
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct SymmetricEvdParams {}

/// Computes the size and alignment of required workspace for performing a hermitian eigenvalue
/// decomposition. The eigenvectors may be optionally computed.
pub fn compute_hermitian_evd_req<E: ComplexField>(
    n: usize,
    compute_eigenvectors: ComputeVectors,
    parallelism: Parallelism,
    params: SymmetricEvdParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = params;
    let _ = compute_eigenvectors;
    let householder_blocksize = faer_qr::no_pivoting::compute::recommended_blocksize::<E>(n, n);

    let cplx_storage = if coe::is_same::<E::Real, E>() {
        StackReq::empty()
    } else {
        StackReq::try_all_of([
            temp_mat_req::<E::Real>(n, n)?,
            StackReq::try_new::<E::Real>(n)?,
        ])?
    };

    StackReq::try_all_of([
        temp_mat_req::<E>(n, n)?,
        temp_mat_req::<E>(householder_blocksize, n - 1)?,
        StackReq::try_any_of([
            tridiag::tridiagonalize_in_place_req::<E>(n, parallelism)?,
            StackReq::try_all_of([
                StackReq::try_new::<E::Real>(n)?,
                StackReq::try_new::<E::Real>(n - 1)?,
                tridiag_real_evd::compute_tridiag_real_evd_req::<E>(n, parallelism)?,
                cplx_storage,
            ])?,
            faer_core::householder::apply_block_householder_sequence_on_the_left_in_place_req::<E>(
                n - 1,
                householder_blocksize,
                n,
            )?,
        ])?,
    ])
}

/// Computes the eigenvalue decomposition of a square hermitian `matrix`. Only the lower triangular
/// half of the matrix is accessed.
///
/// `s` represents the diagonal of the matrix $S$, and must have size equal to the dimension of the
/// matrix.
///
/// If `u` is `None`, then only the eigenvalues are computed. Otherwise, the eigenvectors are
/// computed and stored in `u`.
///
/// # Panics
/// Panics if any of the conditions described above is violated, or if the type `E` does not have a
/// fixed precision at compile time, e.g. a dynamic multiprecision floating point type.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`compute_hermitian_evd_req`]).
pub fn compute_hermitian_evd<E: ComplexField>(
    matrix: MatRef<'_, E>,
    s: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: SymmetricEvdParams,
) {
    compute_hermitian_evd_custom_epsilon(
        matrix,
        s,
        u,
        E::Real::faer_epsilon().unwrap(),
        E::Real::faer_zero_threshold().unwrap(),
        parallelism,
        stack,
        params,
    );
}

/// See [`compute_hermitian_evd`].
///
/// This function takes an additional `epsilon` and `zero_threshold` parameters. `epsilon`
/// represents the precision of the values in the matrix, and `zero_threshold` is the value below
/// which the precision starts to deteriorate, e.g. due to denormalized numbers.
///
/// These values need to be provided manually for types that do not have a known precision at
/// compile time, e.g. a dynamic multiprecision floating point type.
pub fn compute_hermitian_evd_custom_epsilon<E: ComplexField>(
    matrix: MatRef<'_, E>,
    s: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    epsilon: E::Real,
    zero_threshold: E::Real,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: SymmetricEvdParams,
) {
    let _ = params;
    assert!(matrix.nrows() == matrix.ncols());
    let n = matrix.nrows();

    assert!(s.nrows() == n);
    assert!(s.ncols() == 1);
    if let Some(u) = u.rb() {
        assert!(u.nrows() == n);
        assert!(u.ncols() == n);
    }

    let mut all_finite = true;
    zipped!(matrix).for_each_triangular_lower(faer_core::zip::Diag::Include, |x| {
        all_finite &= x.read().faer_is_finite();
    });

    if !all_finite {
        { s }.fill(E::faer_nan());
        if let Some(mut u) = u {
            u.fill(E::faer_nan());
        }
        return;
    }

    let (mut trid, stack) = temp_mat_uninit::<E>(n, n, stack);
    let householder_blocksize =
        faer_qr::no_pivoting::compute::recommended_blocksize::<E>(n - 1, n - 1);

    let (mut householder, mut stack) = temp_mat_uninit::<E>(householder_blocksize, n - 1, stack);
    let mut householder = householder.as_mut();

    let mut trid = trid.as_mut();

    zipped!(trid.rb_mut(), matrix)
        .for_each_triangular_lower(faer_core::zip::Diag::Include, |mut dst, src| {
            dst.write(src.read())
        });

    tridiag::tridiagonalize_in_place(
        trid.rb_mut(),
        householder.rb_mut().transpose(),
        parallelism,
        stack.rb_mut(),
    );

    let trid = trid.into_const();
    let mut s = s;

    let mut u = match u {
        Some(u) => u,
        None => {
            let (mut diag, stack) = stack.rb_mut().make_with(n, |i| trid.read(i, i).faer_real());
            let (mut offdiag, _) = stack.make_with(n - 1, |i| trid.read(i + 1, i).faer_abs());
            tridiag_qr_algorithm::compute_tridiag_real_evd_qr_algorithm(
                &mut diag,
                &mut offdiag,
                None,
                epsilon,
                zero_threshold,
            );
            for i in 0..n {
                s.write(i, 0, E::faer_from_real(diag[i]));
            }

            return;
        }
    };

    let mut j_base = 0;
    while j_base < n - 1 {
        let bs = Ord::min(householder_blocksize, n - 1 - j_base);
        let mut householder = householder.rb_mut().submatrix(0, j_base, bs, bs);
        let full_essentials = trid.submatrix(1, 0, n - 1, n);
        let essentials = full_essentials.submatrix(j_base, j_base, n - 1 - j_base, bs);
        for j in 0..bs {
            householder.write(j, j, householder.read(0, j));
        }
        upgrade_householder_factor(householder, essentials, bs, 1, parallelism);
        j_base += bs;
    }

    {
        let (mut diag, stack) = stack.rb_mut().make_with(n, |i| trid.read(i, i).faer_real());

        if coe::is_same::<E::Real, E>() {
            let (mut offdiag, stack) = stack.make_with(n - 1, |i| trid.read(i + 1, i).faer_real());

            tridiag_real_evd::compute_tridiag_real_evd::<E::Real>(
                &mut diag,
                &mut offdiag,
                u.rb_mut().coerce(),
                epsilon,
                zero_threshold,
                parallelism,
                stack,
            );
        } else {
            let (mut offdiag, stack) = stack.make_with(n - 1, |i| trid.read(i + 1, i).faer_abs());

            let (mut u_real, stack) = temp_mat_uninit::<E::Real>(n, n, stack);
            let (mut mul, stack) = stack.make_with(n, |_| E::faer_zero());

            let normalized = |x: E| {
                if x == E::faer_zero() {
                    E::faer_one()
                } else {
                    x.faer_scale_real(x.faer_abs().faer_inv())
                }
            };

            mul[0] = E::faer_one();

            let mut x = E::faer_one();
            for i in 1..n {
                x = normalized(trid.read(i, i - 1).faer_mul(x.faer_conj())).faer_conj();
                mul[i] = x.faer_conj();
            }

            let mut u_real = u_real.as_mut();

            tridiag_real_evd::compute_tridiag_real_evd::<E::Real>(
                &mut diag,
                &mut offdiag,
                u_real.rb_mut(),
                epsilon,
                zero_threshold,
                parallelism,
                stack,
            );

            for j in 0..n {
                for i in 0..n {
                    unsafe {
                        u.write_unchecked(i, j, mul[i].faer_scale_real(u_real.read_unchecked(i, j)))
                    };
                }
            }
        }

        for i in 0..n {
            s.write(i, 0, E::faer_from_real(diag[i]));
        }
    }

    let mut m = faer_core::Mat::<E>::zeros(n, n);
    for i in 0..n {
        m.write(i, i, s.read(i, 0));
    }

    faer_core::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        trid.submatrix(1, 0, n - 1, n - 1),
        householder.rb(),
        Conj::No,
        u.rb_mut().subrows(1, n - 1),
        parallelism,
        stack.rb_mut(),
    );
}

/// Computes the eigenvalue decomposition of a square real `matrix`.
///
/// `s_re` and `s_im` respectively represent the real and imaginary parts of the diagonal of the
/// matrix $S$, and must have size equal to the dimension of the matrix.
///
/// If `u` is `None`, then only the eigenvalues are computed. Otherwise, the eigenvectors are
/// computed and stored in `u`.
///
/// The eigenvectors are stored as follows, for each real eigenvalue, the corresponding column of
/// the eigenvector matrix is the corresponding eigenvector.
///
/// For each complex eigenvalue pair $a + ib$ and $a - ib$ at indices `k` and `k + 1`, the
/// eigenvalues are stored consecutively. And the real and imaginary parts of the eigenvector
/// corresponding to the eigenvalue $a + ib$ are stored at indices `k` and `k+1`. The eigenvector
/// corresponding to $a - ib$ can be computed as the conjugate of that vector.
///
/// # Panics
/// Panics if any of the conditions described above is violated, or if the type `E` does not have a
/// fixed precision at compile time, e.g. a dynamic multiprecision floating point type.
///
/// This can also panic if the provided memory in `stack` is insufficient (see [`compute_evd_req`]).
pub fn compute_evd_real<E: RealField>(
    matrix: MatRef<'_, E>,
    s_re: MatMut<'_, E>,
    s_im: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: EvdParams,
) {
    compute_evd_real_custom_epsilon(
        matrix,
        s_re,
        s_im,
        u,
        E::faer_epsilon().unwrap(),
        E::faer_zero_threshold().unwrap(),
        parallelism,
        stack,
        params,
    );
}

/// See [`compute_evd_real`].
///
/// This function takes an additional `epsilon` and `zero_threshold` parameters. `epsilon`
/// represents the precision of the values in the matrix, and `zero_threshold` is the value below
/// which the precision starts to deteriorate, e.g. due to denormalized numbers.
///
/// These values need to be provided manually for types that do not have a known precision at
/// compile time, e.g. a dynamic multiprecision floating point type.
pub fn compute_evd_real_custom_epsilon<E: RealField>(
    matrix: MatRef<'_, E>,
    s_re: MatMut<'_, E>,
    s_im: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    epsilon: E,
    zero_threshold: E,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: EvdParams,
) {
    assert!(matrix.nrows() == matrix.ncols());
    let n = matrix.nrows();

    assert!(s_re.nrows() == n);
    assert!(s_re.ncols() == 1);
    assert!(s_im.nrows() == n);
    assert!(s_im.ncols() == 1);
    if let Some(u) = u.rb() {
        assert!(u.nrows() == n);
        assert!(u.ncols() == n);
    }

    if n == 0 {
        return;
    }

    if !matrix.is_all_finite() {
        { s_re }.fill(E::faer_nan());
        { s_im }.fill(E::faer_nan());
        if let Some(mut u) = u {
            u.fill(E::faer_nan());
        }
        return;
    }

    let householder_blocksize = recommended_blocksize::<E>(n - 1, n - 1);

    let mut u = u;
    let mut s_re = s_re;
    let mut s_im = s_im;

    let (mut h, stack) = temp_mat_uninit(n, n, stack);
    let mut h = h.as_mut();

    h.clone_from(matrix);

    let (mut z, mut stack) = temp_mat_zeroed::<E>(n, if u.is_some() { n } else { 0 }, stack);
    let mut z = z.as_mut();
    z.rb_mut().diagonal().fill(E::faer_one());

    {
        let (mut householder, mut stack) =
            temp_mat_uninit(householder_blocksize, n - 1, stack.rb_mut());
        let mut householder = householder.as_mut();

        hessenberg::make_hessenberg_in_place(
            h.rb_mut(),
            householder.rb_mut().transpose(),
            parallelism,
            stack.rb_mut(),
        );
        if u.is_some() {
            apply_block_householder_sequence_on_the_right_in_place_with_conj(
                h.rb().submatrix(1, 0, n - 1, n - 1),
                householder.rb(),
                Conj::No,
                z.rb_mut().submatrix(1, 1, n - 1, n - 1),
                parallelism,
                stack,
            );
        }

        for j in 0..n {
            for i in j + 2..n {
                h.write(i, j, E::faer_zero());
            }
        }
    }

    if let Some(mut u) = u.rb_mut() {
        hessenberg_real_evd::multishift_qr(
            true,
            h.rb_mut(),
            Some(z.rb_mut()),
            s_re.rb_mut(),
            s_im.rb_mut(),
            0,
            n,
            epsilon,
            zero_threshold,
            parallelism,
            stack.rb_mut(),
            params,
        );

        let (mut x, _) = temp_mat_zeroed::<E>(n, n, stack);
        let mut x = x.as_mut();

        let mut norm = zero_threshold;
        zipped!(h.rb()).for_each_triangular_upper(faer_core::zip::Diag::Include, |x| {
            norm = norm.faer_add(x.read().faer_abs());
        });
        // subdiagonal
        zipped!(h.rb().submatrix(1, 0, n - 1, n - 1).diagonal()).for_each(|x| {
            norm = norm.faer_add(x.read().faer_abs());
        });

        {
            let mut k = n;
            loop {
                if k == 0 {
                    break;
                }
                k -= 1;

                if k == 0 || h.read(k, k - 1) == E::faer_zero() {
                    // real eigenvalue
                    let p = h.read(k, k);

                    x.write(k, k, E::faer_one());

                    // solve (h[:k, :k] - p I) X = -h[:i, i]
                    // form RHS
                    for i in 0..k {
                        x.write(i, k, h.read(i, k).faer_neg());
                    }

                    // solve in place
                    let mut i = k;
                    loop {
                        if i == 0 {
                            break;
                        }
                        i -= 1;

                        if i == 0 || h.read(i, i - 1) == E::faer_zero() {
                            // 1x1 block
                            let dot = inner_prod_with_conj(
                                h.rb().row(i).subcols(i + 1, k - i - 1).transpose(),
                                Conj::No,
                                x.rb().col(k).subrows(i + 1, k - i - 1),
                                Conj::No,
                            );

                            x.write(i, k, x.read(i, k).faer_sub(dot));
                            let mut z = h.read(i, i).faer_sub(p);
                            if z == E::faer_zero() {
                                z = epsilon.faer_mul(norm);
                            }
                            let z_inv = z.faer_inv();
                            let x_ = x.read(i, k);
                            if x_ != E::faer_zero() {
                                x.write(i, k, x.read(i, k).faer_mul(z_inv));
                            }
                        } else {
                            // 2x2 block
                            let dot0 = inner_prod_with_conj(
                                h.rb().row(i - 1).subcols(i + 1, k - i - 1).transpose(),
                                Conj::No,
                                x.rb().col(k).subrows(i + 1, k - i - 1),
                                Conj::No,
                            );
                            let dot1 = inner_prod_with_conj(
                                h.rb().row(i).subcols(i + 1, k - i - 1).transpose(),
                                Conj::No,
                                x.rb().col(k).subrows(i + 1, k - i - 1),
                                Conj::No,
                            );

                            x.write(i - 1, k, x.read(i - 1, k).faer_sub(dot0));
                            x.write(i, k, x.read(i, k).faer_sub(dot1));

                            // solve
                            // [a b  [x0    [r0
                            //  c a]× x1] =  r1]
                            //
                            //  [x0    [a  -b  [r0
                            //   x1] =  -c  a]× r1] / det
                            let a = h.read(i, i).faer_sub(p);
                            let b = h.read(i - 1, i);
                            let c = h.read(i, i - 1);

                            let r0 = x.read(i - 1, k);
                            let r1 = x.read(i, k);

                            let inv_det = (a.faer_mul(a).faer_sub(b.faer_mul(c))).faer_inv();

                            let x0 = a.faer_mul(r0).faer_sub(b.faer_mul(r1)).faer_mul(inv_det);
                            let x1 = a.faer_mul(r1).faer_sub(c.faer_mul(r0)).faer_mul(inv_det);

                            x.write(i - 1, k, x0);
                            x.write(i, k, x1);

                            i -= 1;
                        }
                    }
                } else {
                    // complex eigenvalue pair
                    let p = h.read(k, k);
                    let q = h
                        .read(k, k - 1)
                        .faer_abs()
                        .faer_sqrt()
                        .faer_mul(h.read(k - 1, k).faer_abs().faer_sqrt());

                    if h.read(k - 1, k).faer_abs() >= h.read(k, k - 1) {
                        x.write(k - 1, k - 1, E::faer_one());
                        x.write(k, k, q.faer_div(h.read(k - 1, k)));
                    } else {
                        x.write(k - 1, k - 1, q.faer_neg().faer_div(h.read(k, k - 1)));
                        x.write(k, k, E::faer_one());
                    }
                    x.write(k - 1, k, E::faer_zero());
                    x.write(k, k - 1, E::faer_zero());

                    // solve (h[:k-1, :k-1] - (p + iq) I) X = RHS
                    // form RHS
                    for i in 0..k - 1 {
                        x.write(
                            i,
                            k - 1,
                            x.read(k - 1, k - 1).faer_neg().faer_mul(h.read(i, k - 1)),
                        );
                        x.write(i, k, x.read(k, k).faer_neg().faer_mul(h.read(i, k)));
                    }

                    // solve in place
                    let mut i = k - 1;
                    loop {
                        use num_complex::Complex;

                        if i == 0 {
                            break;
                        }
                        i -= 1;

                        if i == 0 || h.read(i, i - 1) == E::faer_zero() {
                            // 1x1 block
                            let mut dot = Complex::<E>::faer_zero();
                            for j in i + 1..k - 1 {
                                dot = dot.faer_add(
                                    Complex {
                                        re: x.read(j, k - 1),
                                        im: x.read(j, k),
                                    }
                                    .faer_scale_real(h.read(i, j)),
                                );
                            }

                            x.write(i, k - 1, x.read(i, k - 1).faer_sub(dot.re));
                            x.write(i, k, x.read(i, k).faer_sub(dot.im));

                            let z = Complex {
                                re: h.read(i, i).faer_sub(p),
                                im: q.faer_neg(),
                            };
                            let z_inv = z.faer_inv();
                            let x_ = Complex {
                                re: x.read(i, k - 1),
                                im: x.read(i, k),
                            };
                            if x_ != Complex::<E>::faer_zero() {
                                let x_ = z_inv.faer_mul(x_);
                                x.write(i, k - 1, x_.re);
                                x.write(i, k, x_.im);
                            }
                        } else {
                            // 2x2 block
                            let mut dot0 = Complex::<E>::faer_zero();
                            let mut dot1 = Complex::<E>::faer_zero();
                            for j in i + 1..k - 1 {
                                dot0 = dot0.faer_add(
                                    Complex {
                                        re: x.read(j, k - 1),
                                        im: x.read(j, k),
                                    }
                                    .faer_scale_real(h.read(i - 1, j)),
                                );
                                dot1 = dot1.faer_add(
                                    Complex {
                                        re: x.read(j, k - 1),
                                        im: x.read(j, k),
                                    }
                                    .faer_scale_real(h.read(i, j)),
                                );
                            }

                            x.write(i - 1, k - 1, x.read(i - 1, k - 1).faer_sub(dot0.re));
                            x.write(i - 1, k, x.read(i - 1, k).faer_sub(dot0.im));
                            x.write(i, k - 1, x.read(i, k - 1).faer_sub(dot1.re));
                            x.write(i, k, x.read(i, k).faer_sub(dot1.im));

                            let a = Complex {
                                re: h.read(i, i).faer_sub(p),
                                im: q.faer_neg(),
                            };
                            let b = h.read(i - 1, i);
                            let c = h.read(i, i - 1);

                            let r0 = Complex {
                                re: x.read(i - 1, k - 1),
                                im: x.read(i - 1, k),
                            };
                            let r1 = Complex {
                                re: x.read(i, k - 1),
                                im: x.read(i, k),
                            };

                            let inv_det = (a
                                .faer_mul(a)
                                .faer_sub(Complex::<E>::faer_from_real(b.faer_mul(c))))
                            .faer_inv();

                            let x0 = a
                                .faer_mul(r0)
                                .faer_sub(r1.faer_scale_real(b))
                                .faer_mul(inv_det);
                            let x1 = a
                                .faer_mul(r1)
                                .faer_sub(r0.faer_scale_real(c))
                                .faer_mul(inv_det);

                            x.write(i - 1, k - 1, x0.re);
                            x.write(i - 1, k, x0.im);
                            x.write(i, k - 1, x1.re);
                            x.write(i, k, x1.im);

                            i -= 1;
                        }
                    }

                    k -= 1;
                }
            }
        }

        triangular::matmul(
            u.rb_mut(),
            BlockStructure::Rectangular,
            z.rb(),
            BlockStructure::Rectangular,
            x.rb(),
            BlockStructure::TriangularUpper,
            None,
            E::faer_one(),
            parallelism,
        );
    } else {
        hessenberg_real_evd::multishift_qr(
            false,
            h.rb_mut(),
            None,
            s_re.rb_mut(),
            s_im.rb_mut(),
            0,
            n,
            epsilon,
            zero_threshold,
            parallelism,
            stack.rb_mut(),
            params,
        );
    }
}

/// Computes the size and alignment of required workspace for performing an eigenvalue
/// decomposition. The eigenvectors may be optionally computed.
pub fn compute_evd_req<E: ComplexField>(
    n: usize,
    compute_eigenvectors: ComputeVectors,
    parallelism: Parallelism,
    params: EvdParams,
) -> Result<StackReq, SizeOverflow> {
    if n == 0 {
        return Ok(StackReq::empty());
    }
    let householder_blocksize = recommended_blocksize::<E>(n - 1, n - 1);
    let compute_vecs = matches!(compute_eigenvectors, ComputeVectors::Yes);
    StackReq::try_all_of([
        // h
        temp_mat_req::<E>(n, n)?,
        // z
        temp_mat_req::<E>(n, if compute_vecs { n } else { 0 })?,
        StackReq::try_any_of([
            StackReq::try_all_of([
                temp_mat_req::<E>(householder_blocksize, n - 1)?,
                StackReq::try_any_of([
                    hessenberg::make_hessenberg_in_place_req::<E>(
                        n,
                        householder_blocksize,
                        parallelism,
                    )?,
                    apply_block_householder_sequence_on_the_right_in_place_req::<E>(
                        n - 1,
                        householder_blocksize,
                        n,
                    )?,
                ])?,
            ])?,
            StackReq::try_any_of([
                hessenberg_cplx_evd::multishift_qr_req::<E>(
                    n,
                    n,
                    compute_vecs,
                    compute_vecs,
                    parallelism,
                    params,
                )?,
                temp_mat_req::<E>(n, n)?,
            ])?,
        ])?,
    ])
}

/// Computes the eigenvalue decomposition of a square complex `matrix`.
///
/// `s` represents the diagonal of the matrix $S$, and must have size equal to the dimension of the
/// matrix.
///
/// If `u` is `None`, then only the eigenvalues are computed. Otherwise, the eigenvectors are
/// computed and stored in `u`.
///
/// # Panics
/// Panics if any of the conditions described above is violated, or if the type `E` does not have a
/// fixed precision at compile time, e.g. a dynamic multiprecision floating point type.
///
/// This can also panic if the provided memory in `stack` is insufficient (see [`compute_evd_req`]).
pub fn compute_evd_complex<E: ComplexField>(
    matrix: MatRef<'_, E>,
    s: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: EvdParams,
) {
    compute_evd_complex_custom_epsilon(
        matrix,
        s,
        u,
        E::Real::faer_epsilon().unwrap(),
        E::Real::faer_zero_threshold().unwrap(),
        parallelism,
        stack,
        params,
    );
}

/// See [`compute_evd_complex`].
///
/// This function takes an additional `epsilon` and `zero_threshold` parameters. `epsilon`
/// represents the precision of the values in the matrix, and `zero_threshold` is the value below
/// which the precision starts to deteriorate, e.g. due to denormalized numbers.
///
/// These values need to be provided manually for types that do not have a known precision at
/// compile time, e.g. a dynamic multiprecision floating point type.
pub fn compute_evd_complex_custom_epsilon<E: ComplexField>(
    matrix: MatRef<'_, E>,
    s: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    epsilon: E::Real,
    zero_threshold: E::Real,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: EvdParams,
) {
    assert!(!coe::is_same::<E, E::Real>());
    assert!(matrix.nrows() == matrix.ncols());
    let n = matrix.nrows();

    assert!(s.nrows() == n);
    assert!(s.ncols() == 1);
    if let Some(u) = u.rb() {
        assert!(u.nrows() == n);
        assert!(u.ncols() == n);
    }

    if n == 0 {
        return;
    }

    if !matrix.is_all_finite() {
        { s }.fill(E::faer_nan());
        if let Some(mut u) = u {
            u.fill(E::faer_nan());
        }
        return;
    }

    let householder_blocksize = recommended_blocksize::<E>(n - 1, n - 1);

    let mut u = u;
    let mut s = s;

    let (mut h, stack) = temp_mat_uninit(n, n, stack);
    let mut h = h.as_mut();

    h.clone_from(matrix);

    let (mut z, mut stack) = temp_mat_zeroed::<E>(n, if u.is_some() { n } else { 0 }, stack);
    let mut z = z.as_mut();
    z.rb_mut().diagonal().fill(E::faer_one());

    {
        let (mut householder, mut stack) =
            temp_mat_uninit(n - 1, householder_blocksize, stack.rb_mut());
        let mut householder = householder.as_mut();

        hessenberg::make_hessenberg_in_place(
            h.rb_mut(),
            householder.rb_mut(),
            parallelism,
            stack.rb_mut(),
        );
        if u.is_some() {
            apply_block_householder_sequence_on_the_right_in_place_with_conj(
                h.rb().submatrix(1, 0, n - 1, n - 1),
                householder.rb().transpose(),
                Conj::No,
                z.rb_mut().submatrix(1, 1, n - 1, n - 1),
                parallelism,
                stack,
            );
        }

        for j in 0..n {
            for i in j + 2..n {
                h.write(i, j, E::faer_zero());
            }
        }
    }

    if let Some(mut u) = u.rb_mut() {
        hessenberg_cplx_evd::multishift_qr(
            true,
            h.rb_mut(),
            Some(z.rb_mut()),
            s.rb_mut(),
            0,
            n,
            epsilon,
            zero_threshold,
            parallelism,
            stack.rb_mut(),
            params,
        );

        let (mut x, _) = temp_mat_zeroed::<E>(n, n, stack);
        let mut x = x.as_mut();

        let mut norm = zero_threshold;
        zipped!(h.rb()).for_each_triangular_upper(faer_core::zip::Diag::Include, |x| {
            norm = norm.faer_add(x.read().faer_abs2());
        });
        let norm = norm.faer_sqrt();

        for k in (0..n).rev() {
            x.write(k, k, E::faer_zero());
            for i in (0..k).rev() {
                x.write(i, k, h.read(i, k).faer_neg());
                if k > i + 1 {
                    let dot = inner_prod_with_conj(
                        h.rb().row(i).subcols(i + 1, k - i - 1).transpose(),
                        Conj::No,
                        x.rb().col(k).subrows(i + 1, k - i - 1),
                        Conj::No,
                    );
                    x.write(i, k, x.read(i, k).faer_sub(dot));
                }

                let mut z = h.read(i, i).faer_sub(h.read(k, k));
                if z == E::faer_zero() {
                    z = E::faer_from_real(epsilon.faer_mul(norm));
                }
                let z_inv = z.faer_inv();
                let x_ = x.read(i, k);
                if x_ != E::faer_zero() {
                    x.write(i, k, x.read(i, k).faer_mul(z_inv));
                }
            }
        }

        triangular::matmul(
            u.rb_mut(),
            BlockStructure::Rectangular,
            z.rb(),
            BlockStructure::Rectangular,
            x.rb(),
            BlockStructure::UnitTriangularUpper,
            None,
            E::faer_one(),
            parallelism,
        );
    } else {
        hessenberg_cplx_evd::multishift_qr(
            false,
            h.rb_mut(),
            None,
            s.rb_mut(),
            0,
            n,
            epsilon,
            zero_threshold,
            parallelism,
            stack.rb_mut(),
            params,
        );
    }
}

#[cfg(test)]
mod herm_tests {
    use super::*;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{c64, Mat};

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn test_real() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |_, _| rand::random::<f64>());

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                Parallelism::None,
                make_stack!(compute_hermitian_evd_req::<f64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )),
                Default::default(),
            );

            let reconstructed = &u * &s * u.transpose();

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_cplx() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |i, j| {
                c64::new(rand::random(), if i == j { 0.0 } else { rand::random() })
            });

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                Parallelism::None,
                make_stack!(compute_hermitian_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )),
                Default::default(),
            );

            let reconstructed = &u * &s * u.adjoint();
            dbgf::dbgf!("6.2?", &u, &reconstructed, &mat);

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_real_identity() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    f64::faer_one()
                } else {
                    f64::faer_zero()
                }
            });

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                Parallelism::None,
                make_stack!(compute_hermitian_evd_req::<f64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )),
                Default::default(),
            );

            let reconstructed = &u * &s * u.transpose();

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_cplx_identity() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    c64::faer_one()
                } else {
                    c64::faer_zero()
                }
            });

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                Parallelism::None,
                make_stack!(compute_hermitian_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )),
                Default::default(),
            );

            let reconstructed = &u * &s * u.adjoint();
            dbgf::dbgf!("6.2?", &u, &reconstructed, &mat);

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_real_zero() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |_, _| f64::faer_zero());

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                Parallelism::None,
                make_stack!(compute_hermitian_evd_req::<f64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )),
                Default::default(),
            );

            let reconstructed = &u * &s * u.transpose();

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_cplx_zero() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |_, _| c64::faer_zero());

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                Parallelism::None,
                make_stack!(compute_hermitian_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )),
                Default::default(),
            );

            let reconstructed = &u * &s * u.adjoint();
            dbgf::dbgf!("6.2?", &u, &reconstructed, &mat);

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{c64, Mat};
    use num_complex::Complex;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn test_real() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            for _ in 0..10 {
                let mat = Mat::from_fn(n, n, |_, _| rand::random::<f64>());

                let n = mat.nrows();

                let mut s_re = Mat::zeros(n, n);
                let mut s_im = Mat::zeros(n, n);
                let mut u_re = Mat::zeros(n, n);
                let mut u_im = Mat::zeros(n, n);

                compute_evd_real(
                    mat.as_ref(),
                    s_re.as_mut().diagonal(),
                    s_im.as_mut().diagonal(),
                    Some(u_re.as_mut()),
                    Parallelism::None,
                    make_stack!(compute_evd_req::<c64>(
                        n,
                        ComputeVectors::Yes,
                        Parallelism::None,
                        Default::default(),
                    )),
                    Default::default(),
                );

                let mut j = 0;
                loop {
                    if j == n {
                        break;
                    }

                    if s_im.read(j, j) != 0.0 {
                        for i in 0..n {
                            u_im.write(i, j, u_re.read(i, j + 1));
                            u_im.write(i, j + 1, -u_re.read(i, j + 1));
                            u_re.write(i, j + 1, u_re.read(i, j));
                        }

                        j += 1;
                    }

                    j += 1;
                }

                let u = Mat::from_fn(n, n, |i, j| Complex::new(u_re.read(i, j), u_im.read(i, j)));
                let s = Mat::from_fn(n, n, |i, j| Complex::new(s_re.read(i, j), s_im.read(i, j)));
                let mat = Mat::from_fn(n, n, |i, j| Complex::new(mat.read(i, j), 0.0));

                let left = &mat * &u;
                let right = &u * &s;

                for j in 0..n {
                    for i in 0..n {
                        assert_approx_eq!(left.read(i, j).re, right.read(i, j).re, 1e-10);
                        assert_approx_eq!(left.read(i, j).im, right.read(i, j).im, 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_real_identity() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    f64::faer_one()
                } else {
                    f64::faer_zero()
                }
            });

            let n = mat.nrows();

            let mut s_re = Mat::zeros(n, n);
            let mut s_im = Mat::zeros(n, n);
            let mut u_re = Mat::zeros(n, n);
            let mut u_im = Mat::zeros(n, n);

            compute_evd_real(
                mat.as_ref(),
                s_re.as_mut().diagonal(),
                s_im.as_mut().diagonal(),
                Some(u_re.as_mut()),
                Parallelism::None,
                make_stack!(compute_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )),
                Default::default(),
            );

            let mut j = 0;
            loop {
                if j == n {
                    break;
                }

                if s_im.read(j, j) != 0.0 {
                    for i in 0..n {
                        u_im.write(i, j, u_re.read(i, j + 1));
                        u_im.write(i, j + 1, -u_re.read(i, j + 1));
                        u_re.write(i, j + 1, u_re.read(i, j));
                    }

                    j += 1;
                }

                j += 1;
            }

            let u = Mat::from_fn(n, n, |i, j| Complex::new(u_re.read(i, j), u_im.read(i, j)));
            let s = Mat::from_fn(n, n, |i, j| Complex::new(s_re.read(i, j), s_im.read(i, j)));
            let mat = Mat::from_fn(n, n, |i, j| Complex::new(mat.read(i, j), 0.0));

            let left = &mat * &u;
            let right = &u * &s;

            for j in 0..n {
                for i in 0..n {
                    assert_approx_eq!(left.read(i, j).re, right.read(i, j).re, 1e-10);
                    assert_approx_eq!(left.read(i, j).im, right.read(i, j).im, 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_real_zero() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::<f64>::zeros(n, n);

            let n = mat.nrows();

            let mut s_re = Mat::zeros(n, n);
            let mut s_im = Mat::zeros(n, n);
            let mut u_re = Mat::zeros(n, n);
            let mut u_im = Mat::zeros(n, n);

            compute_evd_real(
                mat.as_ref(),
                s_re.as_mut().diagonal(),
                s_im.as_mut().diagonal(),
                Some(u_re.as_mut()),
                Parallelism::None,
                make_stack!(compute_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )),
                Default::default(),
            );

            let mut j = 0;
            loop {
                if j == n {
                    break;
                }

                if s_im.read(j, j) != 0.0 {
                    for i in 0..n {
                        u_im.write(i, j, u_re.read(i, j + 1));
                        u_im.write(i, j + 1, -u_re.read(i, j + 1));
                        u_re.write(i, j + 1, u_re.read(i, j));
                    }

                    j += 1;
                }

                j += 1;
            }

            let u = Mat::from_fn(n, n, |i, j| Complex::new(u_re.read(i, j), u_im.read(i, j)));
            let s = Mat::from_fn(n, n, |i, j| Complex::new(s_re.read(i, j), s_im.read(i, j)));
            let mat = Mat::from_fn(n, n, |i, j| Complex::new(mat.read(i, j), 0.0));

            let left = &mat * &u;
            let right = &u * &s;

            for j in 0..n {
                for i in 0..n {
                    assert_approx_eq!(left.read(i, j).re, right.read(i, j).re, 1e-10);
                    assert_approx_eq!(left.read(i, j).im, right.read(i, j).im, 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_cplx() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |_, _| c64::new(rand::random(), rand::random()));

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_evd_complex(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                Parallelism::None,
                make_stack!(compute_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )),
                Default::default(),
            );

            let left = &mat * &u;
            let right = &u * &s;

            dbgf::dbgf!("6.2?", &mat, &left, &right);

            for j in 0..n {
                for i in 0..n {
                    assert_approx_eq!(left.read(i, j), right.read(i, j), 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_cplx_identity() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    c64::faer_one()
                } else {
                    c64::faer_zero()
                }
            });

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_evd_complex(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                Parallelism::None,
                make_stack!(compute_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )),
                Default::default(),
            );

            let left = &mat * &u;
            let right = &u * &s;

            for j in 0..n {
                for i in 0..n {
                    assert_approx_eq!(left.read(i, j), right.read(i, j), 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_cplx_zero() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |_, _| c64::faer_zero());

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_evd_complex(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                Parallelism::None,
                make_stack!(compute_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )),
                Default::default(),
            );

            let left = &mat * &u;
            let right = &u * &s;

            for j in 0..n {
                for i in 0..n {
                    assert_approx_eq!(left.read(i, j), right.read(i, j), 1e-10);
                }
            }
        }
    }
}
