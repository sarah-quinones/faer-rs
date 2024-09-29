use crate::{
    linalg::{
        cholesky::piv_llt::compute as piv_llt, matmul::triangular::BlockStructure, temp_mat_req,
        temp_mat_uninit,
    },
    linop::{InitialGuessStatus, LinOp, Precond},
    prelude::*,
    ComplexField, Parallelism, RealField,
};
use core::marker::PhantomData;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use equator::assert;
use reborrow::*;

/// Algorithm parameters.
#[derive(Copy, Clone, Debug)]
pub struct CgParams<E: ComplexField> {
    /// Whether the initial guess is implicitly zero or not.
    pub initial_guess: InitialGuessStatus,
    /// Absolute tolerance for convergence testing.
    pub abs_tolerance: E::Real,
    /// Relative tolerance for convergence testing.
    pub rel_tolerance: E::Real,
    /// Maximum number of iterations.
    pub max_iters: usize,

    #[doc(hidden)]
    pub __private: PhantomData<()>,
}

/// Algorithm result.
#[derive(Copy, Clone, Debug)]
pub struct CgInfo<E: ComplexField> {
    /// Absolute residual at the final step.
    pub abs_residual: E::Real,
    /// Relative residual at the final step.
    pub rel_residual: E::Real,
    /// Number of iterations executed by the algorithm.
    pub iter_count: usize,

    #[doc(hidden)]
    pub __private: PhantomData<()>,
}

/// Algorithm error.
#[derive(Copy, Clone, Debug)]
pub enum CgError<E: ComplexField> {
    /// Operator was detected to not be positive definite.
    NonPositiveDefiniteOperator,
    /// Preconditioner was detected to not be positive definite.
    NonPositiveDefinitePreconditioner,
    /// Convergence failure.
    NoConvergence {
        /// Absolute residual at the final step.
        abs_residual: E::Real,
        /// Relative residual at the final step.
        rel_residual: E::Real,
    },
}

impl<E: ComplexField> Default for CgParams<E> {
    #[inline]
    fn default() -> Self {
        Self {
            initial_guess: InitialGuessStatus::MaybeNonZero,
            abs_tolerance: E::Real::faer_zero(),
            rel_tolerance: E::Real::faer_epsilon().faer_mul(E::Real::faer_from_f64(128.0)),
            max_iters: usize::MAX,
            __private: PhantomData,
        }
    }
}

/// Computes the size and alignment of required workspace for executing the conjugate gradient
/// algorithm.
pub fn conjugate_gradient_req<E: ComplexField>(
    precond: impl Precond<E>,
    mat: impl LinOp<E>,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    fn implementation<E: ComplexField>(
        M: &dyn Precond<E>,
        A: &dyn LinOp<E>,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let n = A.nrows();
        let k = rhs_ncols;

        let nk = temp_mat_req::<E>(n, k)?;
        let kk = temp_mat_req::<E>(k, k)?;
        let k_usize = StackReq::try_new::<usize>(k)?;
        let chol = piv_llt::cholesky_in_place_req::<usize, E>(k, parallelism)?;
        StackReq::try_all_of([
            nk,      // residual
            nk,      // p
            nk,      // z
            kk,      // rtz
            k_usize, // perm
            k_usize, // perm_inv
            StackReq::try_any_of([
                StackReq::try_all_of([
                    nk, // Ap
                    kk, // ptAp | rtz_new
                    StackReq::try_any_of([
                        A.apply_req(k, parallelism)?,
                        chol, // ptAp | rtz
                        StackReq::try_all_of([
                            kk, // alpha | beta
                            kk, // alpha_perm | beta_perm
                        ])?,
                    ])?,
                ])?,
                M.apply_req(k, parallelism)?,
            ])?,
        ])
    }
    implementation(&precond, &mat, rhs_ncols, parallelism)
}

/// Executes the conjugate gradient using the provided preconditioner.
///
/// # Note
/// This function is also optimized for a RHS with multiple columns.
#[inline]
#[track_caller]
pub fn conjugate_gradient<E: ComplexField>(
    out: MatMut<'_, E>,
    precond: impl Precond<E>,
    mat: impl LinOp<E>,
    rhs: MatRef<'_, E>,
    params: CgParams<E>,
    callback: impl FnMut(MatRef<'_, E>),
    parallelism: Parallelism,
    stack: &mut PodStack,
) -> Result<CgInfo<E>, CgError<E>> {
    #[track_caller]
    fn implementation<E: ComplexField>(
        mut x: MatMut<'_, E>,
        M: &dyn Precond<E>,
        A: &dyn LinOp<E>,
        b: MatRef<'_, E>,

        params: CgParams<E>,
        callback: &mut dyn FnMut(MatRef<'_, E>),
        parallelism: Parallelism,
        mut stack: &mut PodStack,
    ) -> Result<CgInfo<E>, CgError<E>> {
        assert!(A.nrows() == A.ncols());

        let n = A.nrows();
        let k = b.ncols();
        let b_norm = b.norm_l2();
        if b_norm == E::Real::faer_zero() {
            x.fill_zero();
            return Ok(CgInfo {
                abs_residual: E::Real::faer_zero(),
                rel_residual: E::Real::faer_zero(),
                iter_count: 0,
                __private: PhantomData,
            });
        }

        let rel_threshold = params.rel_tolerance.faer_mul(b_norm);
        let abs_threshold = params.abs_tolerance;

        let threshold = if abs_threshold > rel_threshold {
            abs_threshold
        } else {
            rel_threshold
        };

        let (mut r, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
        let (mut p, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
        let (mut z, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());

        let (mut rtz, mut stack) = temp_mat_uninit::<E>(k, k, stack.rb_mut());
        let (perm, mut stack) = stack.rb_mut().make_raw::<usize>(k);
        let (perm_inv, mut stack) = stack.rb_mut().make_raw::<usize>(k);

        let abs_residual = if params.initial_guess == InitialGuessStatus::MaybeNonZero {
            A.apply(r.rb_mut(), x.rb(), parallelism, stack.rb_mut());
            zipped!(&mut r, &b)
                .for_each(|unzipped!(mut res, rhs)| res.write(rhs.read().faer_sub(res.read())));
            r.norm_l2()
        } else {
            b_norm
        };

        if abs_residual < threshold {
            return Ok(CgInfo {
                abs_residual,
                rel_residual: abs_residual.faer_div(b_norm),
                iter_count: 0,
                __private: PhantomData,
            });
        }

        let tril = BlockStructure::TriangularLower;

        {
            M.apply(p.rb_mut(), r.rb(), parallelism, stack.rb_mut());

            crate::linalg::matmul::triangular::matmul(
                rtz.rb_mut(),
                tril,
                r.rb().adjoint(),
                BlockStructure::Rectangular,
                p.rb(),
                BlockStructure::Rectangular,
                None,
                E::faer_one(),
                parallelism,
            );
        }
        for iter in 0..params.max_iters {
            {
                let (mut Ap, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
                let (mut ptAp, mut stack) = temp_mat_uninit::<E>(k, k, stack.rb_mut());

                A.apply(Ap.rb_mut(), p.rb(), parallelism, stack.rb_mut());
                crate::linalg::matmul::triangular::matmul(
                    ptAp.rb_mut(),
                    tril,
                    p.rb().adjoint(),
                    BlockStructure::Rectangular,
                    Ap.rb(),
                    BlockStructure::Rectangular,
                    None,
                    E::faer_one(),
                    parallelism,
                );

                let (info, llt_perm) = match piv_llt::cholesky_in_place(
                    ptAp.rb_mut(),
                    perm,
                    perm_inv,
                    parallelism,
                    stack.rb_mut(),
                    Default::default(),
                ) {
                    Ok(ok) => ok,
                    Err(_) => return Err(CgError::NonPositiveDefiniteOperator),
                };

                let (mut alpha, mut stack) = temp_mat_uninit::<E>(k, k, stack.rb_mut());
                let (mut alpha_perm, _) = temp_mat_uninit::<E>(k, k, stack.rb_mut());
                alpha.copy_from(&rtz);
                for j in 0..k {
                    for i in 0..j {
                        alpha.write(i, j, alpha.read(j, i).faer_conj());
                    }
                }
                crate::perm::permute_rows(alpha_perm.rb_mut(), alpha.rb(), llt_perm);
                crate::linalg::triangular_solve::solve_lower_triangular_in_place(
                    ptAp.rb().get(..info.rank, ..info.rank),
                    alpha_perm.rb_mut().get_mut(..info.rank, ..),
                    parallelism,
                );
                crate::linalg::triangular_solve::solve_upper_triangular_in_place(
                    ptAp.rb().get(..info.rank, ..info.rank).adjoint(),
                    alpha_perm.rb_mut().get_mut(..info.rank, ..),
                    parallelism,
                );
                alpha_perm.rb_mut().get_mut(info.rank.., ..).fill_zero();
                crate::perm::permute_rows(alpha.rb_mut(), alpha_perm.rb(), llt_perm.inverse());

                crate::linalg::matmul::matmul(
                    x.rb_mut(),
                    p.rb(),
                    alpha.rb(),
                    if iter == 0 && params.initial_guess == InitialGuessStatus::Zero {
                        None
                    } else {
                        Some(E::faer_one())
                    },
                    E::faer_one(),
                    parallelism,
                );
                crate::linalg::matmul::matmul(
                    r.rb_mut(),
                    Ap.rb(),
                    alpha.rb(),
                    Some(E::faer_one()),
                    E::faer_one().faer_neg(),
                    parallelism,
                );
                callback(x.rb());
            }

            let abs_residual = r.norm_l2();
            if abs_residual < threshold {
                return Ok(CgInfo {
                    abs_residual,
                    rel_residual: abs_residual.faer_div(b_norm),
                    iter_count: iter + 1,
                    __private: PhantomData,
                });
            }

            M.apply(z.rb_mut(), r.rb(), parallelism, stack.rb_mut());

            let (mut rtz_new, mut stack) = temp_mat_uninit::<E>(k, k, stack.rb_mut());
            crate::linalg::matmul::triangular::matmul(
                rtz_new.rb_mut(),
                tril,
                r.rb().adjoint(),
                BlockStructure::Rectangular,
                z.rb(),
                BlockStructure::Rectangular,
                None,
                E::faer_one(),
                parallelism,
            );

            {
                let (info, llt_perm) = match piv_llt::cholesky_in_place(
                    rtz.rb_mut(),
                    perm,
                    perm_inv,
                    parallelism,
                    stack.rb_mut(),
                    Default::default(),
                ) {
                    Ok(ok) => ok,
                    Err(_) => return Err(CgError::NonPositiveDefiniteOperator),
                };
                let (mut beta, mut stack) = temp_mat_uninit::<E>(k, k, stack.rb_mut());
                let (mut beta_perm, _) = temp_mat_uninit::<E>(k, k, stack.rb_mut());
                beta.copy_from(&rtz_new);
                for j in 0..k {
                    for i in 0..j {
                        beta.write(i, j, beta.read(j, i).faer_conj());
                    }
                }
                crate::perm::permute_rows(beta_perm.rb_mut(), beta.rb(), llt_perm);
                crate::linalg::triangular_solve::solve_lower_triangular_in_place(
                    rtz.rb().get(..info.rank, ..info.rank),
                    beta_perm.rb_mut().get_mut(..info.rank, ..),
                    parallelism,
                );
                crate::linalg::triangular_solve::solve_upper_triangular_in_place(
                    rtz.rb().get(..info.rank, ..info.rank).adjoint(),
                    beta_perm.rb_mut().get_mut(..info.rank, ..),
                    parallelism,
                );
                beta_perm.rb_mut().get_mut(info.rank.., ..).fill_zero();
                crate::perm::permute_rows(beta.rb_mut(), beta_perm.rb(), llt_perm.inverse());
                rtz.copy_from(&rtz_new);

                crate::linalg::matmul::matmul(
                    z.rb_mut(),
                    p.rb(),
                    beta.rb(),
                    Some(E::faer_one()),
                    E::faer_one(),
                    parallelism,
                );
                p.copy_from(&z);
            }
        }

        Err(CgError::NoConvergence {
            abs_residual,
            rel_residual: abs_residual.faer_div(b_norm),
        })
    }

    implementation(
        out,
        &precond,
        &mat,
        rhs,
        params,
        &mut { callback },
        parallelism,
        stack,
    )
}

#[cfg(test)]
mod tests {
    use crate::linop;

    use super::*;
    use crate::mat;
    use dyn_stack::GlobalPodBuffer;
    use equator::assert;
    use rand::prelude::*;

    #[test]
    fn test_cg() {
        let ref A = mat![[2.5, -1.0], [-1.0, 3.1]];
        let ref sol = mat![[2.1, 2.4], [4.1, 4.0]];
        let ref rhs = A * sol;
        let ref mut out = Mat::<f64>::zeros(2, sol.ncols());
        let mut params = CgParams::default();
        params.max_iters = 10;
        let precond = linop::IdentityPrecond { dim: 2 };
        let result = conjugate_gradient(
            out.as_mut(),
            precond,
            A.as_ref(),
            rhs.as_ref(),
            params,
            |_| {},
            Parallelism::None,
            PodStack::new(&mut GlobalPodBuffer::new(
                conjugate_gradient_req(precond, A.as_ref(), 2, Parallelism::None).unwrap(),
            )),
        );
        let ref out = *out;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!((A * out - rhs).norm_l2() <= params.rel_tolerance * rhs.norm_l2());
        assert!(result.iter_count <= 1);
    }

    #[test]
    fn test_cg_breakdown() {
        let ref mut rng = StdRng::seed_from_u64(0);
        let n = 10;
        let k = 15;
        let ref Q: Mat<c64> = crate::stats::UnitaryMat { dimension: n }.sample(rng);
        let mut d = Col::zeros(n);
        for i in 0..n {
            d[i] = c64::new(
                f64::exp(rand::distributions::Standard.sample(rng)).recip(),
                0.0,
            );
        }
        let ref A = Q * d.as_ref().column_vector_as_diagonal() * Q.adjoint();
        let ref mut diag = Mat::<c64>::identity(n, n);
        for i in 0..n {
            diag[(i, i)] = c64::new(
                f64::exp(rand::distributions::Standard.sample(rng)).recip(),
                0.0,
            );
        }
        let ref diag = *diag;
        let ref mut sol = crate::stats::NormalMat {
            nrows: n,
            ncols: k,
            normal: crate::stats::Normal::new(c64::new(0.0, 0.0), 1.0).unwrap(),
        }
        .sample(rng);

        for i in 0..n {
            sol[(i, k - 1)] = c64::new(0.0, 0.0);
            for j in 0..k - 1 {
                let val = sol[(i, j)];
                sol[(i, k - 1)] += val;
            }
        }

        let ref sol = *sol;
        let ref rhs = A * sol;
        let ref mut out = Mat::<c64>::zeros(n, k);
        let params = CgParams::default();
        let result = conjugate_gradient(
            out.as_mut(),
            diag.as_ref(),
            A.as_ref(),
            rhs.as_ref(),
            params,
            |_| {},
            Parallelism::None,
            PodStack::new(&mut GlobalPodBuffer::new(
                conjugate_gradient_req::<c64>(diag.as_ref(), A.as_ref(), k, Parallelism::None)
                    .unwrap(),
            )),
        );
        let ref out = *out;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!((A * out - rhs).norm_l2() <= params.rel_tolerance * rhs.norm_l2());
        assert!(result.iter_count <= 1);
    }
}
