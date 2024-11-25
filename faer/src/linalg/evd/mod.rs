pub mod hessenberg;
pub mod schur;

pub mod tridiag;
pub(crate) mod tridiag_evd;

use crate::{assert, internal_prelude::*};
use hessenberg::HessenbergParams;
use linalg::matmul::triangular::BlockStructure;
use schur::SchurParams;
use tridiag::TridiagParams;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvdError {
    NoConvergence,
}

#[derive(Clone, Copy, Debug)]
pub struct EvdParams {
    pub hessenberg: HessenbergParams,
    pub schur: SchurParams,

    pub non_exhaustive: NonExhaustive,
}

#[derive(Clone, Copy, Debug)]
pub struct SelfAdjointEvdParams {
    pub tridiag: TridiagParams,
    pub recursion_threshold: usize,

    pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for EvdParams {
    fn auto() -> Self {
        Self {
            hessenberg: auto!(T),
            schur: auto!(T),
            non_exhaustive: NonExhaustive(()),
        }
    }
}

impl<T: ComplexField> Auto<T> for SelfAdjointEvdParams {
    fn auto() -> Self {
        Self {
            tridiag: auto!(T),
            recursion_threshold: 128,
            non_exhaustive: NonExhaustive(()),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComputeEigenvectors {
    No,
    Yes,
}

#[math]
pub fn self_adjoint_evd_scratch<T: ComplexField>(
    dim: usize,
    compute_u: ComputeEigenvectors,
    par: Par,
    params: SelfAdjointEvdParams,
) -> Result<StackReq, SizeOverflow> {
    let n = dim;
    let bs = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n, n);

    let prologue = StackReq::try_all_of([
        temp_mat_scratch::<T>(n, n)?,
        temp_mat_scratch::<T>(bs, n)?,
        StackReq::try_any_of([tridiag::tridiag_in_place_scratch::<T>(
            n,
            par,
            params.tridiag,
        )?])?,
        temp_mat_scratch::<T::Real>(n, 1)?.try_array(2)?,
    ])?;
    if compute_u == ComputeEigenvectors::No {
        return Ok(prologue);
    }

    StackReq::try_all_of([
        prologue,
        temp_mat_scratch::<T::Real>(n, if const { T::IS_REAL } { 0 } else { n })?.try_array(2)?,
        StackReq::try_any_of([
            if n < params.recursion_threshold {
                StackReq::empty()
            } else {
                tridiag_evd::divide_and_conquer_scratch::<T>(n, par)?
            },
            temp_mat_scratch::<T>(n, 1)?,
            linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                n - 1,
                bs,
                n,
            )?,
        ])?,
    ])
}

#[math]
pub fn self_adjoint_evd<T: ComplexField>(
    matrix: MatRef<'_, T>,
    s: ColMut<'_, T>,
    u: Option<MatMut<'_, T>>,
    par: Par,
    stack: &mut DynStack,
    params: SelfAdjointEvdParams,
) -> Result<(), EvdError> {
    let n = matrix.nrows();
    assert!(all(matrix.nrows() == matrix.ncols(), s.nrows() == n));
    if let Some(u) = u.rb() {
        assert!(all(u.nrows() == n, u.ncols() == n));
    }

    if n == 0 {
        return Ok(());
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

    let (mut trid, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
    let mut trid = trid.as_mat_mut();

    trid.copy_from_triangular_lower(matrix);

    let bs = linalg::qr::no_pivoting::factor::recommended_blocksize::<T>(n, n);
    let (mut householder, stack) = unsafe { temp_mat_uninit::<T, _, _>(bs, n, stack) };
    let mut householder = householder.as_mat_mut();

    {
        with_dim!(N, n);
        with_dim!(B, bs);

        tridiag::tridiag_in_place(
            trid.rb_mut().as_shape_mut(N, N),
            householder.rb_mut().as_shape_mut(B, N),
            par,
            stack,
            params.tridiag,
        );
    }

    let trid = trid.rb();

    let (mut diag, stack) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };
    let (mut offdiag, stack) = unsafe { temp_mat_uninit::<T::Real, _, _>(n, 1, stack) };

    let mut diag = diag.as_mat_mut().col_mut(0).try_as_col_major_mut().unwrap();
    let mut offdiag = offdiag
        .as_mat_mut()
        .col_mut(0)
        .try_as_col_major_mut()
        .unwrap();

    for i in 0..n {
        diag[i] = real(trid[(i, i)]);

        if i + 1 < n {
            if const { T::IS_REAL } {
                offdiag[i] = real(trid[(i + 1, i)]);
            } else {
                offdiag[i] = abs(trid[(i + 1, i)]);
            }
        } else {
            offdiag[i] = zero();
        }
    }

    let mut s = s;
    let mut u = match u {
        Some(u) => u,
        None => {
            tridiag_evd::qr_algorithm(diag.rb_mut(), offdiag.rb_mut(), None)?;
            for i in 0..n {
                s[i] = from_real(diag[i]);
            }

            return Ok(());
        }
    };

    let (mut u_real, stack) =
        unsafe { temp_mat_uninit::<T::Real, _, _>(n, if T::IS_REAL { 0 } else { n }, stack) };
    let mut u_real = u_real.as_mat_mut();
    let mut u_evd = if const { T::IS_REAL } {
        unsafe { core::mem::transmute(u.rb_mut()) }
    } else {
        u_real.rb_mut()
    };

    if n < params.recursion_threshold {
        tridiag_evd::qr_algorithm(diag.rb_mut(), offdiag.rb_mut(), Some(u_evd.rb_mut()))?;
    } else {
        tridiag_evd::divide_and_conquer::<T::Real>(
            diag.rb_mut(),
            offdiag.rb_mut(),
            u_evd.rb_mut(),
            par,
            stack,
            params.recursion_threshold,
        )?;
    }

    if const { !T::IS_REAL } {
        let normalized = |x: T| {
            if x == zero() {
                one()
            } else {
                mul_real(x, recip(abs(x)))
            }
        };

        let (mut scale, _) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
        let mut scale = scale
            .as_mat_mut()
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();

        let mut x = one::<T>();
        scale[0] = one();

        for i in 1..n {
            x = normalized(trid[(i, i - 1)] * x);
            scale[i] = copy(x);
        }
        for j in 0..n {
            z!(u.rb_mut().col_mut(j), u_real.rb().col(j), scale.rb()).for_each(
                |uz!(u, real, scale)| {
                    *u = mul_real(*scale, *real);
                },
            );
        }
    }

    linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        trid.submatrix(1, 0, n - 1, n - 1),
        householder.rb().subcols(0, n - 1),
        Conj::No,
        u.rb_mut().subrows_mut(1, n - 1),
        par,
        stack,
    );

    for i in 0..n {
        s[i] = from_real(diag[i]);
    }

    Ok(())
}

pub fn self_adjoint_pseudoinverse_scratch<T: ComplexField>(
    dim: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    temp_mat_scratch::<T>(dim, dim)?.try_array(2)
}

#[math]
#[track_caller]
pub fn self_adjoint_pseudoinverse<T: ComplexField>(
    pinv: MatMut<'_, T>,
    s: ColRef<'_, T>,
    u: MatRef<'_, T>,
    par: Par,
    stack: &mut DynStack,
) {
    self_adjoint_pseudoinverse_with_tolerance(
        pinv,
        s,
        u,
        zero(),
        eps() * from_f64(u.ncols() as f64),
        par,
        stack,
    );
}

#[math]
#[track_caller]
pub fn self_adjoint_pseudoinverse_with_tolerance<T: ComplexField>(
    pinv: MatMut<'_, T>,
    s: ColRef<'_, T>,
    u: MatRef<'_, T>,
    abs_tol: T::Real,
    rel_tol: T::Real,
    par: Par,
    stack: &mut DynStack,
) {
    let mut pinv = pinv;
    let n = u.ncols();

    assert!(all(u.nrows() == n, u.ncols() == n, s.nrows() == n));

    let smax = s.norm_max();
    let tol = max(abs_tol, rel_tol * smax);

    let (mut u_trunc, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
    let (mut up_trunc, _) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };

    let mut u_trunc = u_trunc.as_mat_mut();
    let mut up_trunc = up_trunc.as_mat_mut();
    let mut len = 0;

    for j in 0..n {
        let x = absmax(s[j]);
        if x > tol {
            let p = recip(real(s[j]));
            u_trunc.rb_mut().col_mut(len).copy_from(u.col(j));
            z!(up_trunc.rb_mut().col_mut(len), u.col(j))
                .for_each(|uz!(dst, src)| *dst = mul_real(*src, p));

            len += 1;
        }
    }

    linalg::matmul::triangular::matmul(
        pinv.rb_mut(),
        BlockStructure::TriangularLower,
        Accum::Replace,
        up_trunc.rb(),
        BlockStructure::Rectangular,
        u_trunc.rb().adjoint(),
        BlockStructure::Rectangular,
        one(),
        par,
    );

    for j in 0..n {
        for i in 0..j {
            pinv[(i, j)] = conj(pinv[(j, i)]);
        }
    }
}

#[cfg(test)]
mod self_adjoint_tests {
    use super::*;
    use crate::{assert, stats::prelude::*, utils::approx::*};
    use dyn_stack::GlobalMemBuffer;

    fn test_self_adjoint_evd<T: ComplexField>(mat: MatRef<'_, T>) {
        let n = mat.nrows();
        let params = SelfAdjointEvdParams {
            recursion_threshold: 8,
            ..auto!(T)
        };
        use faer_traits::math_utils::*;
        let approx_eq = CwiseMat(ApproxEq::<T>::eps() * sqrt(&from_f64(8.0 * n as f64)));

        let mut s = Mat::zeros(n, n);
        {
            let mut u = Mat::zeros(n, n);

            self_adjoint_evd(
                mat.as_ref(),
                s.as_mut().diagonal_mut().column_vector_mut(),
                Some(u.as_mut()),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    self_adjoint_evd_scratch::<T>(n, ComputeEigenvectors::Yes, Par::Seq, params)
                        .unwrap(),
                )),
                params,
            )
            .unwrap();

            let reconstructed = &u * &s * u.adjoint();
            assert!(reconstructed ~ mat);
        }

        {
            let mut s2 = Mat::zeros(n, n);

            self_adjoint_evd(
                mat.as_ref(),
                s2.as_mut().diagonal_mut().column_vector_mut(),
                None,
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    self_adjoint_evd_scratch::<T>(n, ComputeEigenvectors::No, Par::Seq, params)
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

        for n in [1, 2, 4, 10, 15, 20, 50, 100, 150] {
            let mat = CwiseMatDistribution {
                nrows: n,
                ncols: n,
                dist: StandardNormal,
            }
            .rand::<Mat<f64>>(rng);
            let mat = &mat + mat.adjoint();

            test_self_adjoint_evd(mat.as_ref());
        }
    }

    #[test]
    fn test_cplx() {
        let rng = &mut StdRng::seed_from_u64(1);

        for n in [2, 4, 10, 15, 20, 50, 100, 150] {
            let mat = CwiseMatDistribution {
                nrows: n,
                ncols: n,
                dist: ComplexDistribution::new(StandardNormal, StandardNormal),
            }
            .rand::<Mat<c64>>(rng);
            let mat = &mat + mat.adjoint();

            test_self_adjoint_evd(mat.as_ref());
        }
    }

    #[test]
    fn test_special() {
        for n in [1, 2, 4, 10, 15, 20, 50, 100, 150] {
            test_self_adjoint_evd(Mat::full(n, n, 0.0).as_ref());
            test_self_adjoint_evd(Mat::full(n, n, c64::ZERO).as_ref());
            test_self_adjoint_evd(Mat::full(n, n, 1.0).as_ref());
            test_self_adjoint_evd(Mat::full(n, n, c64::ONE).as_ref());
            test_self_adjoint_evd(Mat::<f64>::identity(n, n).as_ref());
            test_self_adjoint_evd(Mat::<c64>::identity(n, n).as_ref());
        }
    }
}
