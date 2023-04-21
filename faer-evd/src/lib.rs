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

use assert2::assert;
use coe::Coerce;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    householder::upgrade_householder_factor, temp_mat_req, temp_mat_uninit, zipped, ComplexField,
    Conj, MatMut, MatRef, Parallelism,
};
use reborrow::*;

#[doc(hidden)]
pub mod tridiag_qr_algorithm;

#[doc(hidden)]
pub mod tridiag_real_evd;

#[doc(hidden)]
pub mod tridiag;

/// Indicates whether the eigenvectors are fully computed, partially computed, or skipped.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComputeVectors {
    No,
    Yes,
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct SymmetricEvdParams {}

/// Computes the size and alignment of required workspace for performing an eigenvalue
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

/// Computes the eigenvalue decomposition of square hermitian `matrix`. Only the lower triangular
/// half of the matrix is accessed.
///
/// `s` represents the diagonal of the matrix $S$, and must have size equal to the dimension of the
/// matrix.
///
/// If `u` is `None`, then only the eigenvalues are computed. Otherwise, the eigenvectors are
/// computed and stored in `u`.
pub fn compute_hermitian_evd<E: ComplexField>(
    matrix: MatRef<'_, E>,
    s: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    epsilon: E::Real,
    zero_threshold: E::Real,
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: SymmetricEvdParams,
) {
    let _ = params;
    assert!(matrix.nrows() == matrix.ncols());
    let n = matrix.nrows();

    assert!(s.nrows() == n);
    if let Some(u) = u.rb() {
        assert!(u.nrows() == n);
        assert!(u.ncols() == n);
    }

    let (mut trid, stack) = unsafe { temp_mat_uninit::<E>(n, n, stack) };
    let householder_blocksize =
        faer_qr::no_pivoting::compute::recommended_blocksize::<E>(n - 1, n - 1);

    let (mut householder, mut stack) =
        unsafe { temp_mat_uninit::<E>(householder_blocksize, n - 1, stack) };
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
            let (mut diag, stack) = stack.rb_mut().make_with(n, |i| trid.read(i, i).real());
            let (mut offdiag, _) = stack.make_with(n - 1, |i| trid.read(i + 1, i).abs());
            tridiag_qr_algorithm::compute_tridiag_real_evd_qr_algorithm(
                &mut diag,
                &mut offdiag,
                None,
                epsilon,
                zero_threshold,
            );
            for i in 0..n {
                s.write(i, 0, E::from_real(diag[i].clone()));
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
        let (mut diag, stack) = stack.rb_mut().make_with(n, |i| trid.read(i, i).real());

        if coe::is_same::<E::Real, E>() {
            let (mut offdiag, stack) = stack.make_with(n - 1, |i| trid.read(i + 1, i).real());

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
            let (mut offdiag, stack) = stack.make_with(n - 1, |i| trid.read(i + 1, i).abs());

            let (mut u_real, stack) = unsafe { temp_mat_uninit::<E::Real>(n, n, stack) };
            let (mut mul, stack) = stack.make_with(n, |_| E::zero());

            let normalized = |x: E| {
                if x == E::zero() {
                    E::one()
                } else {
                    x.scale_real(&x.abs().inv())
                }
            };

            mul[0] = E::one();

            let mut x = E::one();
            for i in 1..n {
                x = normalized(trid.read(i, i - 1).mul(&x.conj())).conj();
                mul[i] = x.conj();
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
                        u.write_unchecked(i, j, mul[i].scale_real(&u_real.read_unchecked(i, j)))
                    };
                }
            }
        }

        for i in 0..n {
            s.write(i, 0, E::from_real(diag[i].clone()));
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

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{c64, Mat};

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn test_real() {
        for n in [2, 3, 4, 5, 6, 7, 10, 15, 25] {
            let mat = Mat::with_dims(n, n, |_, _| rand::random::<f64>());

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
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
            let mat = Mat::with_dims(n, n, |i, j| {
                c64::new(rand::random(), if i == j { 0.0 } else { rand::random() })
            });

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
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
            let mat = Mat::with_dims(n, n, |i, j| if i == j { f64::one() } else { f64::zero() });

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
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
            let mat = Mat::with_dims(n, n, |i, j| if i == j { c64::one() } else { c64::zero() });

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
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
            let mat = Mat::with_dims(n, n, |_, _| f64::zero());

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
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
            let mat = Mat::with_dims(n, n, |_, _| c64::zero());

            let mut s = Mat::zeros(n, n);
            let mut u = Mat::zeros(n, n);

            compute_hermitian_evd(
                mat.as_ref(),
                s.as_mut().diagonal(),
                Some(u.as_mut()),
                f64::EPSILON,
                f64::MIN_POSITIVE,
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
