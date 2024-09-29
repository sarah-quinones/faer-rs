//! Low level implementation of the eigenvalue decomposition of a square diagonalizable matrix.
//!
//! The eigenvalue decomposition of a square matrix $M$ of shape $(n, n)$ is a decomposition into
//! two components $U$, $S$:
//!
//! - $U$ has shape $(n, n)$ and is invertible,
//! - $S$ has shape $(n, n)$ and is a diagonal matrix,
//! - and finally:
//!
//! $$M = U S U^{-1}.$$
//!
//! If $M$ is Hermitian, then $U$ can be made unitary ($U^{-1} = U^H$), and $S$ is real valued.

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

use crate::{
    assert,
    linalg::{
        householder::{
            apply_block_householder_sequence_on_the_right_in_place_req,
            apply_block_householder_sequence_on_the_right_in_place_with_conj,
            upgrade_householder_factor,
        },
        matmul::{
            inner_prod::inner_prod_with_conj,
            triangular::{self, BlockStructure},
        },
        qr::no_pivoting::compute::recommended_blocksize,
        temp_mat_req, temp_mat_uninit, temp_mat_zeroed,
    },
    unzipped,
    utils::thread::parallelism_degree,
    zipped, ColMut, ColRef, ComplexField, Conj, MatMut, MatRef, Parallelism, RealField,
};
use coe::Coerce;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
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
    /// Do not compute the eigenvectors.
    No,
    /// Do compute the eigenvectors.
    Yes,
}

/// Hermitian eigendecomposition tuning parameters.
#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct HermitianEvdParams {}

/// Computes the size and alignment of required workspace for performing a Hermitian eigenvalue
/// decomposition. The eigenvectors may be optionally computed.
pub fn compute_hermitian_evd_req<E: ComplexField>(
    n: usize,
    compute_eigenvectors: ComputeVectors,
    parallelism: Parallelism,
    params: HermitianEvdParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = params;
    let _ = compute_eigenvectors;
    let householder_blocksize =
        crate::linalg::qr::no_pivoting::compute::recommended_blocksize::<E>(n, n);

    let cplx_storage = if const { E::IS_REAL } {
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
            crate::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_req::<
                E,
            >(n - 1, householder_blocksize, n)?,
        ])?,
    ])
}

/// Computes the eigenvalue decomposition of a square Hermitian `matrix`. Only the lower triangular
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
    s: ColMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    parallelism: Parallelism,
    stack: &mut PodStack,
    params: HermitianEvdParams,
) {
    compute_hermitian_evd_custom_epsilon(
        matrix,
        s,
        u,
        E::Real::faer_epsilon(),
        E::Real::faer_zero_threshold(),
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
    s: ColMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    epsilon: E::Real,
    zero_threshold: E::Real,
    parallelism: Parallelism,
    stack: &mut PodStack,
    params: HermitianEvdParams,
) {
    let _ = params;
    let n = matrix.nrows();

    assert!(all(matrix.nrows() == matrix.ncols(), s.nrows() == n));
    if let Some(u) = u.rb() {
        assert!(all(u.nrows() == n, u.ncols() == n));
    }

    if n == 0 {
        return;
    }

    #[cfg(feature = "perf-warn")]
    if let Some(matrix) = u.rb() {
        if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(QR_WARN) {
            if matrix.col_stride().unsigned_abs() == 1 {
                log::warn!(target: "faer_perf", "EVD prefers column-major eigenvector matrix. Found row-major matrix.");
            } else {
                log::warn!(target: "faer_perf", "EVD prefers column-major eigenvector matrix. Found matrix with generic strides.");
            }
        }
    }

    let mut all_finite = true;
    zipped!(matrix).for_each_triangular_lower(crate::linalg::zip::Diag::Include, |unzipped!(x)| {
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
        crate::linalg::qr::no_pivoting::compute::recommended_blocksize::<E>(n - 1, n - 1);

    let (mut householder, mut stack) = temp_mat_uninit::<E>(householder_blocksize, n - 1, stack);
    let mut householder = householder.as_mut();

    let mut trid = trid.as_mut();

    zipped!(trid.rb_mut(), matrix).for_each_triangular_lower(
        crate::linalg::zip::Diag::Include,
        |unzipped!(mut dst, src)| dst.write(src.read()),
    );

    tridiag::tridiagonalize_in_place(
        trid.rb_mut(),
        householder.rb_mut().transpose_mut(),
        parallelism,
        stack.rb_mut(),
    );

    let trid = trid.into_const();
    let mut s = s;

    let mut u = match u {
        Some(u) => u,
        None => {
            let (diag, stack) = stack.rb_mut().make_with(n, |i| trid.read(i, i).faer_real());
            let (offdiag, _) = stack.make_with(n - 1, |i| trid.read(i + 1, i).faer_abs());
            tridiag_qr_algorithm::compute_tridiag_real_evd_qr_algorithm(
                diag,
                offdiag,
                None,
                epsilon,
                zero_threshold,
            );
            for (i, &diag) in diag.iter().enumerate() {
                s.write(i, E::faer_from_real(diag));
            }

            return;
        }
    };

    let mut j_base = 0;
    while j_base < n - 1 {
        let bs = Ord::min(householder_blocksize, n - 1 - j_base);
        let mut householder = householder.rb_mut().submatrix_mut(0, j_base, bs, bs);
        let full_essentials = trid.submatrix(1, 0, n - 1, n);
        let essentials = full_essentials.submatrix(j_base, j_base, n - 1 - j_base, bs);
        for j in 0..bs {
            householder.write(j, j, householder.read(0, j));
        }
        upgrade_householder_factor(householder, essentials, bs, 1, parallelism);
        j_base += bs;
    }

    {
        let (diag, stack) = stack.rb_mut().make_with(n, |i| trid.read(i, i).faer_real());

        if const { E::IS_REAL } {
            let (offdiag, stack) = stack.make_with(n - 1, |i| trid.read(i + 1, i).faer_real());

            tridiag_real_evd::compute_tridiag_real_evd::<E::Real>(
                diag,
                offdiag,
                u.rb_mut().coerce(),
                epsilon,
                zero_threshold,
                parallelism,
                stack,
            );
        } else {
            let (offdiag, stack) = stack.make_with(n - 1, |i| trid.read(i + 1, i).faer_abs());

            let (mut u_real, stack) = temp_mat_uninit::<E::Real>(n, n, stack);
            let (mul, stack) = stack.make_with(n, |_| E::faer_zero());

            let normalized = |x: E| {
                if x == E::faer_zero() {
                    E::faer_one()
                } else {
                    x.faer_scale_real(x.faer_abs().faer_inv())
                }
            };

            mul[0] = E::faer_one();

            let mut x = E::faer_one();
            for (i, mul) in mul.iter_mut().enumerate().skip(1) {
                x = normalized(trid.read(i, i - 1).faer_mul(x.faer_conj())).faer_conj();
                *mul = x.faer_conj();
            }

            tridiag_real_evd::compute_tridiag_real_evd::<E::Real>(
                diag,
                offdiag,
                u_real.rb_mut(),
                epsilon,
                zero_threshold,
                parallelism,
                stack,
            );

            for j in 0..n {
                for (i, &mul) in mul.iter().enumerate() {
                    unsafe {
                        u.write_unchecked(i, j, mul.faer_scale_real(u_real.read_unchecked(i, j)))
                    };
                }
            }
        }

        for (i, &diag) in diag.iter().enumerate() {
            s.write(i, E::faer_from_real(diag));
        }
    }

    let mut m = crate::Mat::<E>::zeros(n, n);
    for i in 0..n {
        m.write(i, i, s.read(i));
    }

    crate::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        trid.submatrix(1, 0, n - 1, n - 1),
        householder.rb(),
        Conj::No,
        u.rb_mut().subrows_mut(1, n - 1),
        parallelism,
        stack.rb_mut(),
    );
}

/// Computes the size and alignment of required workspace for performing a decomposed
/// Hermitian matrix pseudoi inverse
pub fn compute_hermitian_pseudoinverse_req<E: ComplexField>(
    n: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    StackReq::try_all_of([
        temp_mat_req::<E>(n, n)?,
        temp_mat_req::<E>(n, n)?,
        temp_mat_req::<E::Real>(1, n)?,
        StackReq::try_new::<usize>(n)?,
    ])
}

/// Computes the pseudo inverse of a decomposed square Hermitian matrix.  
/// `s` represents the diagonal of the matrix $S$, and must have size equal to the dimension of the
/// matrix.  
/// `u` represents the eigenvectors.  
/// The result is stored into `pinv`.  
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`compute_hermitian_pseudoinverse_req`]).
pub fn compute_hermitian_pseudoinverse<E: ComplexField>(
    pinv: MatMut<'_, E>,
    s: ColRef<'_, E>,
    u: MatRef<'_, E>,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    compute_hermitian_pseudoinverse_custom_epsilon(pinv, s, u, None, None, parallelism, stack);
}

/// See [`compute_hermitian_pseudoinverse`].
///
/// This function takes an additional `atol` and `rtol` parameters. `atol`
/// represents the absolute threshold term, if `None` provided value is 0, and `rtol` represents the
/// relative threshold term, if `None` provided value is `N * eps` where
/// `eps` is the machine precision value of the `E::Real`
///
/// # Panics
/// Panics if `atol` or `rtol` equal or less than zero
pub fn compute_hermitian_pseudoinverse_custom_epsilon<E: ComplexField>(
    pinv: MatMut<'_, E>,
    s: ColRef<'_, E>,
    u: MatRef<'_, E>,
    atol: Option<E::Real>,
    rtol: Option<E::Real>,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    let mut pinv = pinv;
    let n = u.ncols();
    let mut max_s = E::Real::faer_zero();
    for i in 0..n {
        let v = s.read(i).faer_abs();
        if v > max_s {
            max_s = v;
        }
    }

    let atol = atol.unwrap_or(E::Real::faer_zero());
    let rtol = rtol.unwrap_or(E::Real::faer_epsilon() * E::Real::faer_from_f64(n as f64));
    assert!(
        rtol >= E::Real::faer_zero() && atol >= E::Real::faer_zero(),
        "atol and rtol values must be positive.",
    );
    let (above_cutoff, stack) = stack.make_raw::<usize>(n);
    let val = atol + max_s * rtol;

    let mut r_take = 0;
    for i in 0..n {
        if s.read(i).faer_abs() > val {
            above_cutoff[r_take] = i;
            r_take += 1;
        }
    }

    let (mut psigma_diag, stack) = temp_mat_uninit::<E::Real>(1, r_take, stack);
    for i in 0..r_take {
        psigma_diag.write(
            0,
            i,
            E::Real::faer_one() / s.read(above_cutoff[i]).faer_real(),
        );
    }
    let (mut ru, stack) = temp_mat_uninit::<E>(n, r_take, stack);
    for i in 0..r_take {
        ru.as_mut().col_mut(i).copy_from(u.col(above_cutoff[i]));
    }

    let (mut up, _stack) = temp_mat_uninit::<E>(n, r_take, stack);
    up.as_mut().copy_from(ru.as_ref());
    for i in 0..n {
        zipped!(up.as_mut().row_mut(i), psigma_diag.as_ref().row(0)).for_each(
            |unzipped!(mut dst, src)| dst.write(dst.read() * E::faer_from_real(src.read())),
        );
    }
    crate::linalg::matmul::matmul_with_conj(
        pinv.as_mut(),
        up.as_ref(),
        crate::Conj::No,
        ru.transpose(),
        crate::Conj::Yes,
        None,
        E::faer_one(),
        parallelism,
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
    s_re: ColMut<'_, E>,
    s_im: ColMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    parallelism: Parallelism,
    stack: &mut PodStack,
    params: EvdParams,
) {
    compute_evd_real_custom_epsilon(
        matrix,
        s_re,
        s_im,
        u,
        E::faer_epsilon(),
        E::faer_zero_threshold(),
        parallelism,
        stack,
        params,
    );
}

fn dot2<E: RealField>(lhs0: ColRef<'_, E>, lhs1: ColRef<'_, E>, rhs: ColRef<'_, E>) -> (E, E) {
    let n = rhs.nrows();
    assert!(lhs0.nrows() == n);
    assert!(lhs1.nrows() == n);

    let mut acc00 = E::faer_zero();
    let mut acc01 = E::faer_zero();
    let mut acc02 = E::faer_zero();
    let mut acc03 = E::faer_zero();

    let mut acc10 = E::faer_zero();
    let mut acc11 = E::faer_zero();
    let mut acc12 = E::faer_zero();
    let mut acc13 = E::faer_zero();

    let n4 = n / 4 * 4;

    let mut i = 0;
    unsafe {
        while i < n4 {
            acc00 = acc00.faer_add(E::faer_mul(lhs0.read_unchecked(i), rhs.read_unchecked(i)));
            acc01 = acc01.faer_add(E::faer_mul(
                lhs0.read_unchecked(i + 1),
                rhs.read_unchecked(i + 1),
            ));
            acc02 = acc02.faer_add(E::faer_mul(
                lhs0.read_unchecked(i + 2),
                rhs.read_unchecked(i + 2),
            ));
            acc03 = acc03.faer_add(E::faer_mul(
                lhs0.read_unchecked(i + 3),
                rhs.read_unchecked(i + 3),
            ));

            acc10 = acc10.faer_add(E::faer_mul(lhs1.read_unchecked(i), rhs.read_unchecked(i)));
            acc11 = acc11.faer_add(E::faer_mul(
                lhs1.read_unchecked(i + 1),
                rhs.read_unchecked(i + 1),
            ));
            acc12 = acc12.faer_add(E::faer_mul(
                lhs1.read_unchecked(i + 2),
                rhs.read_unchecked(i + 2),
            ));
            acc13 = acc13.faer_add(E::faer_mul(
                lhs1.read_unchecked(i + 3),
                rhs.read_unchecked(i + 3),
            ));

            i += 4;
        }
        while i < n {
            acc00 = acc00.faer_add(E::faer_mul(lhs0.read_unchecked(i), rhs.read_unchecked(i)));
            acc10 = acc10.faer_add(E::faer_mul(lhs1.read_unchecked(i), rhs.read_unchecked(i)));

            i += 1;
        }
    }

    (
        E::faer_add(acc00.faer_add(acc01), acc02.faer_add(acc03)),
        E::faer_add(acc10.faer_add(acc11), acc12.faer_add(acc13)),
    )
}

fn dot4<E: RealField>(
    lhs0: ColRef<'_, E>,
    lhs1: ColRef<'_, E>,
    rhs0: ColRef<'_, E>,
    rhs1: ColRef<'_, E>,
) -> (E, E, E, E) {
    let n = rhs0.nrows();
    assert!(lhs0.nrows() == n);
    assert!(lhs1.nrows() == n);
    assert!(rhs0.nrows() == n);
    assert!(rhs1.nrows() == n);

    let mut acc00 = E::faer_zero();
    let mut acc01 = E::faer_zero();

    let mut acc10 = E::faer_zero();
    let mut acc11 = E::faer_zero();

    let mut acc20 = E::faer_zero();
    let mut acc21 = E::faer_zero();

    let mut acc30 = E::faer_zero();
    let mut acc31 = E::faer_zero();

    let n2 = n / 2 * 2;

    let mut i = 0;
    unsafe {
        while i < n2 {
            acc00 = acc00.faer_add(E::faer_mul(lhs0.read_unchecked(i), rhs0.read_unchecked(i)));
            acc01 = acc01.faer_add(E::faer_mul(
                lhs0.read_unchecked(i + 1),
                rhs0.read_unchecked(i + 1),
            ));

            acc10 = acc10.faer_add(E::faer_mul(lhs1.read_unchecked(i), rhs0.read_unchecked(i)));
            acc11 = acc11.faer_add(E::faer_mul(
                lhs1.read_unchecked(i + 1),
                rhs0.read_unchecked(i + 1),
            ));

            acc20 = acc20.faer_add(E::faer_mul(lhs0.read_unchecked(i), rhs1.read_unchecked(i)));
            acc21 = acc21.faer_add(E::faer_mul(
                lhs0.read_unchecked(i + 1),
                rhs1.read_unchecked(i + 1),
            ));

            acc30 = acc30.faer_add(E::faer_mul(lhs1.read_unchecked(i), rhs1.read_unchecked(i)));
            acc31 = acc31.faer_add(E::faer_mul(
                lhs1.read_unchecked(i + 1),
                rhs1.read_unchecked(i + 1),
            ));

            i += 2;
        }
        while i < n {
            acc00 = acc00.faer_add(E::faer_mul(lhs0.read_unchecked(i), rhs0.read_unchecked(i)));
            acc10 = acc10.faer_add(E::faer_mul(lhs1.read_unchecked(i), rhs0.read_unchecked(i)));
            acc20 = acc20.faer_add(E::faer_mul(lhs0.read_unchecked(i), rhs1.read_unchecked(i)));
            acc30 = acc30.faer_add(E::faer_mul(lhs1.read_unchecked(i), rhs1.read_unchecked(i)));

            i += 1;
        }
    }

    (
        acc00.faer_add(acc01),
        acc10.faer_add(acc11),
        acc20.faer_add(acc21),
        acc30.faer_add(acc31),
    )
}

fn solve_shifted_upper_quasi_triangular_system<E: RealField>(
    h: MatRef<'_, E>,
    shift: E,
    x: ColMut<'_, E>,
    epsilon: E,
    norm: E,
    parallelism: Parallelism,
) {
    let mut x = x;
    let k = h.nrows();
    let p = shift;

    if parallelism_degree(parallelism) == 1 || k < 512 {
        let mut i = k;
        loop {
            if i == 0 {
                break;
            }
            i -= 1;

            if i == 0 || h.read(i, i - 1) == E::faer_zero() {
                // 1x1 block
                let dot = inner_prod_with_conj(
                    h.row(i).subcols(i + 1, k - i - 1).transpose(),
                    Conj::No,
                    x.rb().subrows(i + 1, k - i - 1),
                    Conj::No,
                );

                x.write(i, x.read(i).faer_sub(dot));
                let mut z = h.read(i, i).faer_sub(p);
                if z == E::faer_zero() {
                    z = epsilon.faer_mul(norm);
                }
                let z_inv = z.faer_inv();
                let x_ = x.read(i);
                if x_ != E::faer_zero() {
                    x.write(i, x.read(i).faer_mul(z_inv));
                }
            } else {
                // 2x2 block
                let dot0 = inner_prod_with_conj(
                    h.row(i - 1).subcols(i + 1, k - i - 1).transpose(),
                    Conj::No,
                    x.rb().subrows(i + 1, k - i - 1),
                    Conj::No,
                );
                let dot1 = inner_prod_with_conj(
                    h.row(i).subcols(i + 1, k - i - 1).transpose(),
                    Conj::No,
                    x.rb().subrows(i + 1, k - i - 1),
                    Conj::No,
                );

                x.write(i - 1, x.read(i - 1).faer_sub(dot0));
                x.write(i, x.read(i).faer_sub(dot1));

                // solve
                // [a b] [x0]   [r0]
                // [c a]×[x1] = [r1]
                //
                //  [x0]   [a  -b] [r0]
                //  [x1] = [-c  a]×[r1] / det
                let a = h.read(i, i).faer_sub(p);
                let b = h.read(i - 1, i);
                let c = h.read(i, i - 1);

                let r0 = x.read(i - 1);
                let r1 = x.read(i);

                let inv_det = (a.faer_mul(a).faer_sub(b.faer_mul(c))).faer_inv();

                let x0 = a.faer_mul(r0).faer_sub(b.faer_mul(r1)).faer_mul(inv_det);
                let x1 = a.faer_mul(r1).faer_sub(c.faer_mul(r0)).faer_mul(inv_det);

                x.write(i - 1, x0);
                x.write(i, x1);

                i -= 1;
            }
        }
    } else {
        let mut mid = k / 2;
        if h.read(mid, mid - 1) != E::faer_zero() {
            mid -= 1;
        }

        solve_shifted_upper_quasi_triangular_system(
            h.get(mid.., mid..),
            shift,
            x.rb_mut().get_mut(mid..),
            epsilon,
            norm,
            parallelism,
        );

        // h is row major
        crate::utils::thread::for_each_raw(
            mid,
            |i| {
                let mut x = unsafe { x.rb().const_cast() };
                let dot = inner_prod_with_conj(
                    h.get(i, mid..).transpose(),
                    Conj::No,
                    x.rb().get(mid..),
                    Conj::No,
                );
                x.write(i, x.read(i).faer_sub(dot));
            },
            parallelism,
        );

        solve_shifted_upper_quasi_triangular_system(
            h.get(..mid, ..mid),
            shift,
            x.rb_mut().get_mut(..mid),
            epsilon,
            norm,
            parallelism,
        );
    }
}

fn solve_shifted_upper_triangular_system<E: ComplexField>(
    h: MatRef<'_, E>,
    shift: E,
    x: ColMut<'_, E>,
    epsilon: E::Real,
    norm: E::Real,
    parallelism: Parallelism,
) {
    let mut x = x;
    let k = h.nrows();
    let p = shift;

    if parallelism_degree(parallelism) == 1 || k < 512 {
        for i in (0..k).rev() {
            if k > i + 1 {
                let dot = inner_prod_with_conj(
                    h.rb().row(i).subcols(i + 1, k - i - 1).transpose(),
                    Conj::No,
                    x.rb().subrows(i + 1, k - i - 1),
                    Conj::No,
                );
                x.write(i, x.read(i).faer_sub(dot));
            }

            let mut z = h.read(i, i).faer_sub(p);
            if z == E::faer_zero() {
                z = E::faer_from_real(epsilon.faer_mul(norm));
            }
            let z_inv = z.faer_inv();
            let x_ = x.read(i);
            if x_ != E::faer_zero() {
                x.write(i, x.read(i).faer_mul(z_inv));
            }
        }
    } else {
        let mid = k / 2;
        solve_shifted_upper_triangular_system(
            h.get(mid.., mid..),
            shift,
            x.rb_mut().get_mut(mid..),
            epsilon,
            norm,
            parallelism,
        );

        // h is row major
        crate::utils::thread::for_each_raw(
            mid,
            |i| {
                let mut x = unsafe { x.rb().const_cast() };
                let dot = inner_prod_with_conj(
                    h.get(i, mid..).transpose(),
                    Conj::No,
                    x.rb().get(mid..),
                    Conj::No,
                );
                x.write(i, x.read(i).faer_sub(dot));
            },
            parallelism,
        );

        solve_shifted_upper_triangular_system(
            h.get(..mid, ..mid),
            shift,
            x.rb_mut().get_mut(..mid),
            epsilon,
            norm,
            parallelism,
        );
    }
}
fn solve_complex_shifted_upper_quasi_triangular_system<E: RealField>(
    h: MatRef<'_, E>,
    shift_re: E,
    shift_im: E,
    x: MatMut<'_, E>,
    epsilon: E,
    norm: E,
    parallelism: Parallelism,
) {
    // solve in place
    let k = h.nrows();
    let p = shift_re;
    let q = shift_im;
    let mut x = x;
    _ = epsilon;
    _ = norm;

    if parallelism_degree(parallelism) == 1 || k < 512 {
        let mut i = k;
        loop {
            use num_complex::Complex;

            if i == 0 {
                break;
            }
            i -= 1;

            if i == 0 || h.read(i, i - 1) == E::faer_zero() {
                // 1x1 block
                let start = i + 1;
                let len = k - (i + 1);
                let (dot_re, dot_im) = dot2(
                    x.rb().col(0).subrows(start, len),
                    x.rb().col(1).subrows(start, len),
                    h.transpose().col(i).subrows(start, len),
                );

                x.write(i, 0, x.read(i, 0).faer_sub(dot_re));
                x.write(i, 1, x.read(i, 1).faer_sub(dot_im));

                let z = Complex {
                    re: h.read(i, i).faer_sub(p),
                    im: q.faer_neg(),
                };
                let z_inv = z.faer_inv();
                let x_ = Complex {
                    re: x.read(i, 0),
                    im: x.read(i, 1),
                };
                if x_ != Complex::<E>::faer_zero() {
                    let x_ = z_inv.faer_mul(x_);
                    x.write(i, 0, x_.re);
                    x.write(i, 1, x_.im);
                }
            } else {
                // 2x2 block
                let start = i + 1;
                let len = k - (i + 1);
                let (dot0_re, dot0_im, dot1_re, dot1_im) = dot4(
                    x.rb().col(0).subrows(start, len),
                    x.rb().col(1).subrows(start, len),
                    h.transpose().col(i - 1).subrows(start, len),
                    h.transpose().col(i).subrows(start, len),
                );
                let mut dot0 = Complex::<E>::faer_zero();
                let mut dot1 = Complex::<E>::faer_zero();
                for j in i + 1..k {
                    dot0 = dot0.faer_add(
                        Complex {
                            re: x.read(j, 0),
                            im: x.read(j, 1),
                        }
                        .faer_scale_real(h.read(i - 1, j)),
                    );
                    dot1 = dot1.faer_add(
                        Complex {
                            re: x.read(j, 0),
                            im: x.read(j, 1),
                        }
                        .faer_scale_real(h.read(i, j)),
                    );
                }

                x.write(i - 1, 0, x.read(i - 1, 0).faer_sub(dot0_re));
                x.write(i - 1, 1, x.read(i - 1, 1).faer_sub(dot0_im));
                x.write(i, 0, x.read(i, 0).faer_sub(dot1_re));
                x.write(i, 1, x.read(i, 1).faer_sub(dot1_im));

                let a = Complex {
                    re: h.read(i, i).faer_sub(p),
                    im: q.faer_neg(),
                };
                let b = h.read(i - 1, i);
                let c = h.read(i, i - 1);

                let r0 = Complex {
                    re: x.read(i - 1, 0),
                    im: x.read(i - 1, 1),
                };
                let r1 = Complex {
                    re: x.read(i, 0),
                    im: x.read(i, 1),
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

                x.write(i - 1, 0, x0.re);
                x.write(i - 1, 1, x0.im);
                x.write(i, 0, x1.re);
                x.write(i, 1, x1.im);

                i -= 1;
            }
        }
    } else {
        let mut mid = k / 2;
        if h.read(mid, mid - 1) != E::faer_zero() {
            mid -= 1;
        }

        solve_complex_shifted_upper_quasi_triangular_system(
            h.get(mid.., mid..),
            shift_re,
            shift_im,
            x.rb_mut().get_mut(mid.., ..),
            epsilon,
            norm,
            parallelism,
        );

        // h is row major
        crate::utils::thread::for_each_raw(
            mid,
            |i| {
                let mut x = unsafe { x.rb().const_cast() };
                let (dot0, dot1) = dot2(
                    x.rb().get(mid.., 0),
                    x.rb().get(mid.., 1),
                    h.get(i, mid..).transpose(),
                );
                x.write(i, 0, x.read(i, 0).faer_sub(dot0));
                x.write(i, 1, x.read(i, 1).faer_sub(dot1));
            },
            parallelism,
        );

        solve_complex_shifted_upper_quasi_triangular_system(
            h.get(..mid, ..mid),
            shift_re,
            shift_im,
            x.rb_mut().get_mut(..mid, ..),
            epsilon,
            norm,
            parallelism,
        );
    }
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
    s_re: ColMut<'_, E>,
    s_im: ColMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    epsilon: E,
    zero_threshold: E,
    parallelism: Parallelism,
    stack: &mut PodStack,
    params: EvdParams,
) {
    let n = matrix.nrows();

    assert!(all(
        matrix.nrows() == matrix.ncols(),
        s_re.nrows() == n,
        s_re.ncols() == 1,
        s_im.nrows() == n,
        s_im.ncols() == 1,
    ));
    if let Some(u) = u.rb() {
        assert!(all(u.nrows() == n, u.ncols() == n));
    }

    if n == 0 {
        return;
    }

    #[cfg(feature = "perf-warn")]
    if let Some(matrix) = u.rb() {
        if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(QR_WARN) {
            if matrix.col_stride().unsigned_abs() == 1 {
                log::warn!(target: "faer_perf", "EVD prefers column-major eigenvector matrix. Found row-major matrix.");
            } else {
                log::warn!(target: "faer_perf", "EVD prefers column-major eigenvector matrix. Found matrix with generic strides.");
            }
        }
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

    h.copy_from(matrix);

    let (mut z, mut stack) = temp_mat_zeroed::<E>(n, if u.is_some() { n } else { 0 }, stack);
    let mut z = z.as_mut();
    z.rb_mut()
        .diagonal_mut()
        .column_vector_mut()
        .fill(E::faer_one());

    {
        let (mut householder, mut stack) =
            temp_mat_uninit(householder_blocksize, n - 1, stack.rb_mut());
        let mut householder = householder.as_mut();

        hessenberg::make_hessenberg_in_place(
            h.rb_mut(),
            householder.rb_mut().transpose_mut(),
            parallelism,
            stack.rb_mut(),
        );
        if u.is_some() {
            apply_block_householder_sequence_on_the_right_in_place_with_conj(
                h.rb().submatrix(1, 0, n - 1, n - 1),
                householder.rb(),
                Conj::No,
                z.rb_mut().submatrix_mut(1, 1, n - 1, n - 1),
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

        let (mut x, _) = temp_mat_uninit::<E>(n, n, stack);

        let mut h = h.transpose_mut();

        for j in 1..n {
            let upper = h.read(j - 1, j);
            let lower = h.read(j, j - 1);

            h.write(j - 1, j, lower);
            h.write(j, j - 1, upper);
        }

        for j in 2..n {
            for i in 0..j - 1 {
                h.write(i, j, h.read(j, i));
            }
        }
        let h = h.rb();

        real_schur_to_eigen(h, x.rb_mut(), parallelism);

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

#[doc(hidden)]
pub fn real_schur_to_eigen<E: RealField>(S: MatRef<'_, E>, Q: MatMut<E>, parallelism: Parallelism) {
    let epsilon = E::faer_epsilon();
    let zero_threshold = E::Real::faer_zero_threshold();
    let n = S.nrows();

    let mut Q = Q;
    Q.fill_zero();

    let mut norm = zero_threshold;
    zipped!(S.rb()).for_each_triangular_upper(crate::linalg::zip::Diag::Include, |unzipped!(x)| {
        norm = norm.faer_add(x.read().faer_abs());
    });
    // subdiagonal
    zipped!(S
        .rb()
        .submatrix(1, 0, n - 1, n - 1)
        .diagonal()
        .column_vector())
    .for_each(|unzipped!(x)| {
        norm = norm.faer_add(x.read().faer_abs());
    });

    let mut k = n;
    loop {
        if k == 0 {
            break;
        }
        k -= 1;

        if k == 0 || S.read(k, k - 1) == E::faer_zero() {
            // real eigenvalue
            let p = S.read(k, k);

            Q.write(k, k, E::faer_one());

            // solve (h[:k, :k] - p I) X = -h[:k, k]
            // form RHS
            for i in 0..k {
                Q.write(i, k, S.read(i, k).faer_neg());
            }

            solve_shifted_upper_quasi_triangular_system(
                S.get(..k, ..k),
                p,
                Q.rb_mut().get_mut(..k, k),
                epsilon,
                norm,
                parallelism,
            );
        } else {
            // complex eigenvalue pair

            let p = S.read(k, k);
            let q = S
                .read(k, k - 1)
                .faer_abs()
                .faer_sqrt()
                .faer_mul(S.read(k - 1, k).faer_abs().faer_sqrt());

            if S.read(k - 1, k).faer_abs() >= S.read(k, k - 1) {
                Q.write(k - 1, k - 1, E::faer_one());
                Q.write(k, k, q.faer_div(S.read(k - 1, k)));
            } else {
                Q.write(k - 1, k - 1, q.faer_neg().faer_div(S.read(k, k - 1)));
                Q.write(k, k, E::faer_one());
            }
            Q.write(k - 1, k, E::faer_zero());
            Q.write(k, k - 1, E::faer_zero());

            // solve (h[:k-1, :k-1] - (p + iq) I) X = RHS
            // form RHS
            for i in 0..k - 1 {
                Q.write(
                    i,
                    k - 1,
                    Q.read(k - 1, k - 1).faer_neg().faer_mul(S.read(i, k - 1)),
                );
                Q.write(i, k, Q.read(k, k).faer_neg().faer_mul(S.read(i, k)));
            }

            solve_complex_shifted_upper_quasi_triangular_system(
                S.get(..k - 1, ..k - 1),
                p,
                q,
                Q.rb_mut().get_mut(..k - 1, k - 1..k + 1),
                epsilon,
                norm,
                parallelism,
            );

            k -= 1;
        }
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
    s: ColMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    parallelism: Parallelism,
    stack: &mut PodStack,
    params: EvdParams,
) {
    compute_evd_complex_custom_epsilon(
        matrix,
        s,
        u,
        E::Real::faer_epsilon(),
        E::Real::faer_zero_threshold(),
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
    s: ColMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    epsilon: E::Real,
    zero_threshold: E::Real,
    parallelism: Parallelism,
    stack: &mut PodStack,
    params: EvdParams,
) {
    assert!(!coe::is_same::<E, E::Real>());
    let n = matrix.nrows();

    assert!(all(
        matrix.nrows() == matrix.ncols(),
        s.nrows() == n,
        s.ncols() == 1,
    ));
    if let Some(u) = u.rb() {
        assert!(all(u.nrows() == n, u.ncols() == n));
    }

    if n == 0 {
        return;
    }

    #[cfg(feature = "perf-warn")]
    if let Some(matrix) = u.rb() {
        if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(QR_WARN) {
            if matrix.col_stride().unsigned_abs() == 1 {
                log::warn!(target: "faer_perf", "EVD prefers column-major eigenvector matrix. Found row-major matrix.");
            } else {
                log::warn!(target: "faer_perf", "EVD prefers column-major eigenvector matrix. Found matrix with generic strides.");
            }
        }
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

    h.copy_from(matrix);

    let (mut z, mut stack) = temp_mat_zeroed::<E>(n, if u.is_some() { n } else { 0 }, stack);
    let mut z = z.as_mut();
    z.rb_mut()
        .diagonal_mut()
        .column_vector_mut()
        .fill(E::faer_one());

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
                z.rb_mut().submatrix_mut(1, 1, n - 1, n - 1),
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
        zipped!(h.rb()).for_each_triangular_upper(
            crate::linalg::zip::Diag::Include,
            |unzipped!(x)| {
                norm = norm.faer_add(x.read().faer_abs2());
            },
        );
        let norm = norm.faer_sqrt();

        let mut h = h.transpose_mut();

        for j in 1..n {
            for i in 0..j {
                h.write(i, j, h.read(j, i));
            }
        }

        for k in (0..n).rev() {
            x.write(k, k, E::faer_zero());
            for i in 0..k {
                x.write(i, k, h.read(i, k).faer_neg());
            }

            solve_shifted_upper_triangular_system(
                h.rb().get(..k, ..k),
                h.read(k, k),
                x.rb_mut().get_mut(..k, k),
                epsilon,
                norm,
                parallelism,
            );
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
    use crate::{assert, complex_native::c64, Mat};
    use assert_approx_eq::assert_approx_eq;

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
                s.as_mut().diagonal_mut().column_vector_mut(),
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
    fn test_real_pseudoinverse() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |_, _| rand::random::<f64>());
            let mut mat = &mat * mat.transpose();
            for i in 0..n / 2 {
                for j in 0..n / 2 {
                    mat.as_mut().write(i, j, 0.0);
                }
            }

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);
            let mut pinv = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal_mut().column_vector_mut(),
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

            compute_hermitian_pseudoinverse(
                pinv.as_mut(),
                s.as_ref().diagonal().column_vector(),
                u.as_ref(),
                Parallelism::None,
                make_stack!(compute_hermitian_pseudoinverse_req::<f64>(
                    n,
                    Parallelism::None,
                )),
            );
            let reconstructed = &mat * &pinv * &mat;
            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-9);
                }
            }

            let reconstructed_pinv = &pinv * &mat * &pinv;
            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(
                        reconstructed_pinv.read(i, j),
                        pinv.read(i, j),
                        f64::max(1e-9, 1e-9 * pinv.read(i, j).abs())
                    );
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
                s.as_mut().diagonal_mut().column_vector_mut(),
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

            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_cplx_pseudoinverse() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::from_fn(n, n, |i, j| {
                c64::new(rand::random(), if i == j { 0.0 } else { rand::random() })
            });

            let mut mat = &mat * mat.adjoint();
            for i in 0..n / 2 {
                for j in 0..n / 2 {
                    mat.as_mut().write(i, j, c64::new(0.0, 0.0));
                }
            }

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);
            let mut pinv = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal_mut().column_vector_mut(),
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

            compute_hermitian_pseudoinverse(
                pinv.as_mut(),
                s.as_ref().diagonal().column_vector(),
                u.as_ref(),
                Parallelism::None,
                make_stack!(compute_hermitian_pseudoinverse_req::<c64>(
                    n,
                    Parallelism::None,
                )),
            );
            let reconstructed = &mat * &pinv * &mat;
            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j), 1e-9);
                }
            }

            let reconstructed_pinv = &pinv * &mat * &pinv;
            for j in 0..n {
                for i in j..n {
                    assert_approx_eq!(
                        reconstructed_pinv.read(i, j),
                        pinv.read(i, j),
                        f64::max(1e-9, 1e-9 * pinv.read(i, j).abs())
                    );
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
                s.as_mut().diagonal_mut().column_vector_mut(),
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
                s.as_mut().diagonal_mut().column_vector_mut(),
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
                s.as_mut().diagonal_mut().column_vector_mut(),
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
                s.as_mut().diagonal_mut().column_vector_mut(),
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
    use crate::{assert, complex_native::c64, Mat};
    use assert_approx_eq::assert_approx_eq;
    use num_complex::Complex;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn test_real_3() {
        let mat = crate::mat![
            [0.03498524449256035, 0.5246466104879548, 0.20804192188707582,],
            [0.007467248113335545, 0.1723793560841066, 0.2677423170633869,],
            [
                0.5907508388039022,
                0.11540612644030279,
                0.2624452803216497f64,
            ],
        ];

        let n = mat.nrows();

        let mut s_re = Mat::zeros(n, n);
        let mut s_im = Mat::zeros(n, n);
        let mut u_re = Mat::zeros(n, n);
        let mut u_im = Mat::zeros(n, n);

        compute_evd_real(
            mat.as_ref(),
            s_re.as_mut().diagonal_mut().column_vector_mut(),
            s_im.as_mut().diagonal_mut().column_vector_mut(),
            Some(u_re.as_mut()),
            Parallelism::Rayon(0),
            make_stack!(compute_evd_req::<c64>(
                n,
                ComputeVectors::Yes,
                Parallelism::Rayon(0),
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

    #[test]
    fn test_real() {
        for n in [3, 2, 4, 5, 6, 7, 10, 15, 25, 600] {
            for _ in 0..10 {
                let mat = Mat::from_fn(n, n, |_, _| rand::random::<f64>());

                let n = mat.nrows();

                let mut s_re = Mat::zeros(n, n);
                let mut s_im = Mat::zeros(n, n);
                let mut u_re = Mat::zeros(n, n);
                let mut u_im = Mat::zeros(n, n);

                compute_evd_real(
                    mat.as_ref(),
                    s_re.as_mut().diagonal_mut().column_vector_mut(),
                    s_im.as_mut().diagonal_mut().column_vector_mut(),
                    Some(u_re.as_mut()),
                    Parallelism::Rayon(0),
                    make_stack!(compute_evd_req::<c64>(
                        n,
                        ComputeVectors::Yes,
                        Parallelism::Rayon(0),
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
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25, 600] {
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
                s_re.as_mut().diagonal_mut().column_vector_mut(),
                s_im.as_mut().diagonal_mut().column_vector_mut(),
                Some(u_re.as_mut()),
                Parallelism::Rayon(0),
                make_stack!(compute_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::Rayon(0),
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
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25, 600] {
            let mat = Mat::<f64>::zeros(n, n);

            let n = mat.nrows();

            let mut s_re = Mat::zeros(n, n);
            let mut s_im = Mat::zeros(n, n);
            let mut u_re = Mat::zeros(n, n);
            let mut u_im = Mat::zeros(n, n);

            compute_evd_real(
                mat.as_ref(),
                s_re.as_mut().diagonal_mut().column_vector_mut(),
                s_im.as_mut().diagonal_mut().column_vector_mut(),
                Some(u_re.as_mut()),
                Parallelism::Rayon(0),
                make_stack!(compute_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::Rayon(0),
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
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25, 600] {
            let mat = Mat::from_fn(n, n, |_, _| c64::new(rand::random(), rand::random()));

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_evd_complex(
                mat.as_ref(),
                s.as_mut().diagonal_mut().column_vector_mut(),
                Some(u.as_mut()),
                Parallelism::Rayon(0),
                make_stack!(compute_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::Rayon(0),
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
    fn test_cplx_identity() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25, 600] {
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
                s.as_mut().diagonal_mut().column_vector_mut(),
                Some(u.as_mut()),
                Parallelism::Rayon(0),
                make_stack!(compute_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::Rayon(0),
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
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25, 600] {
            let mat = Mat::from_fn(n, n, |_, _| c64::faer_zero());

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_evd_complex(
                mat.as_ref(),
                s.as_mut().diagonal_mut().column_vector_mut(),
                Some(u.as_mut()),
                Parallelism::Rayon(0),
                make_stack!(compute_evd_req::<c64>(
                    n,
                    ComputeVectors::Yes,
                    Parallelism::Rayon(0),
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

    // https://github.com/sarah-ek/faer-rs/issues/78
    #[test]
    fn test_cplx_gh78() {
        let i = c64::new(0.0, 1.0);

        let mat = crate::mat![
            [
                0.0 + 0.0 * i,
                0.0 + 0.0 * i,
                0.0 + 0.0 * i,
                2.220446049250313e-16 + -1.0000000000000002 * i
            ],
            [
                0.0 + 0.0 * i,
                0.0 + 0.0 * i,
                2.220446049250313e-16 + -1.0000000000000002 * i,
                0.0 + 0.0 * i
            ],
            [
                0.0 + 0.0 * i,
                2.220446049250313e-16 + -1.0000000000000002 * i,
                0.0 + 0.0 * i,
                0.0 + 0.0 * i
            ],
            [
                2.220446049250313e-16 + -1.0000000000000002 * i,
                0.0 + 0.0 * i,
                0.0 + 0.0 * i,
                0.0 + 0.0 * i
            ],
        ];
        let n = mat.nrows();

        let mut s = Mat::zeros(n, n);
        let mut u = Mat::zeros(n, n);

        compute_evd_complex(
            mat.as_ref(),
            s.as_mut().diagonal_mut().column_vector_mut(),
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
