use crate::{
    linalg::{temp_mat_req, temp_mat_uninit},
    linop::{InitialGuessStatus, LinOp, Precond},
    prelude::*,
    ComplexField, Parallelism, RealField,
};
use core::marker::PhantomData;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use equator::assert;
use reborrow::*;

/// Computes the size and alignment of required workspace for executing the BiCGSTAB algorithm.
pub fn bicgstab_req<E: ComplexField>(
    left_precond: impl Precond<E>,
    right_precond: impl Precond<E>,
    mat: impl LinOp<E>,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    fn implementation<E: ComplexField>(
        K1: &dyn Precond<E>,
        K2: &dyn Precond<E>,
        A: &dyn LinOp<E>,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let n = A.nrows();
        let k = rhs_ncols;

        let nk = temp_mat_req::<E>(n, k)?;
        let kk = temp_mat_req::<E>(k, k)?;
        let k_usize = StackReq::try_new::<usize>(k)?;
        let lu = crate::linalg::lu::full_pivoting::compute::lu_in_place_req::<usize, E>(
            k,
            k,
            parallelism,
            Default::default(),
        )?;
        StackReq::try_all_of([
            k_usize, // row_perm
            k_usize, // row_perm_inv
            k_usize, // col_perm
            k_usize, // col_perm_inv
            kk,      // rtv
            nk,      // r
            nk,      // p
            nk,      // r_tilde
            nk,      // v
            nk,      // y
            nk,      // s
            nk,      // t
            nk,      // z
            StackReq::try_any_of([
                lu,
                A.apply_req(k, parallelism)?,
                StackReq::try_all_of([
                    nk, // y0 | z0 | ks
                    K1.apply_req(k, parallelism)?,
                    K2.apply_req(k, parallelism)?,
                ])?,
                StackReq::try_all_of([
                    kk, // rtr | rtt
                    kk, // temp
                ])?,
                kk, // rtr | rtt
            ])?,
        ])
    }
    implementation(&left_precond, &right_precond, &mat, rhs_ncols, parallelism)
}

/// Algorithm parameters.
#[derive(Copy, Clone, Debug)]
pub struct BicgParams<E: ComplexField> {
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

impl<E: ComplexField> Default for BicgParams<E> {
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

/// Algorithm result.
#[derive(Copy, Clone, Debug)]
pub struct BicgInfo<E: ComplexField> {
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
pub enum BicgError<E: ComplexField> {
    /// Convergence failure.
    NoConvergence {
        /// Absolute residual at the final step.
        abs_residual: E::Real,
        /// Relative residual at the final step.
        rel_residual: E::Real,
    },
}

/// Executes BiCGSTAB using the provided preconditioners.
///
/// # Note
/// This function is also optimized for a RHS with multiple columns.
#[track_caller]
pub fn bicgstab<E: ComplexField>(
    out: MatMut<'_, E>,
    left_precond: impl Precond<E>,
    right_precond: impl Precond<E>,
    mat: impl LinOp<E>,
    rhs: MatRef<'_, E>,
    params: BicgParams<E>,
    callback: impl FnMut(MatRef<'_, E>),
    parallelism: Parallelism,
    stack: &mut PodStack,
) -> Result<BicgInfo<E>, BicgError<E>> {
    #[track_caller]
    fn implementation<E: ComplexField>(
        out: MatMut<'_, E>,
        left_precond: &dyn Precond<E>,
        right_precond: &dyn Precond<E>,
        mat: &dyn LinOp<E>,
        rhs: MatRef<'_, E>,
        params: BicgParams<E>,
        callback: &mut dyn FnMut(MatRef<'_, E>),
        parallelism: Parallelism,
        stack: &mut PodStack,
    ) -> Result<BicgInfo<E>, BicgError<E>> {
        let mut x = out;
        let A = mat;
        let K1 = left_precond;
        let K2 = right_precond;
        let b = rhs;

        assert!(A.nrows() == A.ncols());
        let n = A.nrows();
        let k = x.ncols();

        let b_norm = b.norm_l2();
        if b_norm == E::Real::faer_zero() {
            x.fill_zero();
            return Ok(BicgInfo {
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

        let mut stack = stack;
        let (row_perm, mut stack) = stack.rb_mut().make_raw::<usize>(k);
        let (row_perm_inv, mut stack) = stack.rb_mut().make_raw::<usize>(k);
        let (col_perm, mut stack) = stack.rb_mut().make_raw::<usize>(k);
        let (col_perm_inv, mut stack) = stack.rb_mut().make_raw::<usize>(k);
        let (mut rtv, mut stack) = temp_mat_uninit::<E>(k, k, stack.rb_mut());
        let (mut r, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
        let (mut p, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
        let (mut r_tilde, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());

        let abs_residual = if params.initial_guess == InitialGuessStatus::MaybeNonZero {
            A.apply(r.rb_mut(), x.rb(), parallelism, stack.rb_mut());
            zipped_rw!(&mut r, &b)
                .for_each(|unzipped!(mut r, b)| r.write(b.read().faer_sub(r.read())));

            r.norm_l2()
        } else {
            b_norm
        };

        if abs_residual < threshold {
            return Ok(BicgInfo {
                abs_residual,
                rel_residual: abs_residual.faer_div(b_norm),
                iter_count: 0,
                __private: PhantomData,
            });
        }

        p.copy_from(&r);
        r_tilde.copy_from(&r);

        for iter in 0..params.max_iters {
            let (mut v, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
            let (mut y, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
            {
                let (mut y0, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
                K1.apply(y0.rb_mut(), p.rb(), parallelism, stack.rb_mut());
                K2.apply(y.rb_mut(), y0.rb(), parallelism, stack.rb_mut());
            }
            A.apply(v.rb_mut(), y.rb(), parallelism, stack.rb_mut());

            crate::linalg::matmul::matmul(
                rtv.rb_mut(),
                r_tilde.rb().transpose(),
                v.rb(),
                None,
                E::faer_one(),
                parallelism,
            );
            let (_, row_perm, col_perm) = crate::linalg::lu::full_pivoting::compute::lu_in_place(
                rtv.rb_mut(),
                row_perm,
                row_perm_inv,
                col_perm,
                col_perm_inv,
                parallelism,
                stack.rb_mut(),
                Default::default(),
            );
            let mut rank = k;
            let tol = E::Real::faer_epsilon()
                .faer_mul(E::Real::faer_from_f64(k as f64))
                .faer_mul(rtv.read(0, 0).faer_abs());
            for i in 0..k {
                if rtv.read(i, i).faer_abs() < tol {
                    rank = i;
                    break;
                }
            }

            let (mut s, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
            {
                let (mut rtr, mut stack) = temp_mat_uninit::<E>(k, k, stack.rb_mut());
                crate::linalg::matmul::matmul(
                    rtr.rb_mut(),
                    r_tilde.rb().transpose(),
                    r.rb(),
                    None,
                    E::faer_one(),
                    parallelism,
                );
                let (mut temp, _) = temp_mat_uninit::<E>(k, k, stack.rb_mut());
                crate::perm::permute_rows(temp.rb_mut(), rtr.rb(), row_perm);
                crate::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                    rtv.rb().get(..rank, ..rank),
                    temp.rb_mut().get_mut(..rank, ..),
                    parallelism,
                );
                crate::linalg::triangular_solve::solve_upper_triangular_in_place(
                    rtv.rb().get(..rank, ..rank),
                    temp.rb_mut().get_mut(..rank, ..),
                    parallelism,
                );
                temp.rb_mut().get_mut(rank.., ..).fill_zero();
                crate::perm::permute_rows(rtr.rb_mut(), temp.rb(), col_perm.inverse());
                let alpha = rtr.rb();

                s.copy_from(&r);
                crate::linalg::matmul::matmul(
                    s.rb_mut(),
                    v.rb(),
                    alpha.rb(),
                    Some(E::faer_one()),
                    E::faer_one().faer_neg(),
                    parallelism,
                );
                crate::linalg::matmul::matmul(
                    x.rb_mut(),
                    y.rb(),
                    alpha.rb(),
                    if iter == 0 && params.initial_guess == InitialGuessStatus::Zero {
                        None
                    } else {
                        Some(E::faer_one())
                    },
                    E::faer_one(),
                    parallelism,
                );
            }
            let norm = s.norm_l2();
            if norm < threshold {
                return Ok(BicgInfo {
                    abs_residual: norm,
                    rel_residual: norm.faer_div(b_norm),
                    iter_count: iter + 1,
                    __private: PhantomData,
                });
            }

            let (mut t, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
            let (mut z, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
            {
                let (mut z0, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
                K1.apply(z0.rb_mut(), s.rb(), parallelism, stack.rb_mut());
                K2.apply(z.rb_mut(), z0.rb(), parallelism, stack.rb_mut());
            }
            A.apply(t.rb_mut(), z.rb(), parallelism, stack.rb_mut());

            let compute_w = |kt: MatRef<'_, E>, ks: MatRef<'_, E>| {
                let mut wt = E::faer_zero();
                let mut ws = E::faer_zero();
                for j in 0..k {
                    let kt = kt.rb().col(j);
                    let ks = ks.rb().col(j);
                    ws = ws.faer_add(kt.transpose() * ks);
                    wt = wt.faer_add(kt.transpose() * kt);
                }
                wt.faer_inv().faer_mul(ws)
            };

            let w = {
                let mut kt = y;
                let (mut ks, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
                K1.apply(kt.rb_mut(), t.rb(), parallelism, stack.rb_mut());
                K1.apply(ks.rb_mut(), s.rb(), parallelism, stack.rb_mut());
                compute_w(kt.rb(), ks.rb())
            };

            zipped_rw!(&mut r, &s, &t).for_each(|unzipped!(mut r, s, t)| {
                r.write(s.read().faer_sub(w.faer_mul(t.read())))
            });
            zipped_rw!(&mut x, &z)
                .for_each(|unzipped!(mut x, z)| x.write(x.read().faer_add(w.faer_mul(z.read()))));
            zipped_rw!(&mut p, &v)
                .for_each(|unzipped!(mut p, v)| p.write(p.read().faer_sub(w.faer_mul(v.read()))));

            callback(x.rb());

            let norm = r.norm_l2();
            if norm < threshold {
                return Ok(BicgInfo {
                    abs_residual: norm,
                    rel_residual: norm.faer_div(b_norm),
                    iter_count: iter + 1,
                    __private: PhantomData,
                });
            }

            let (mut rtt, mut stack) = temp_mat_uninit::<E>(k, k, stack.rb_mut());
            {
                crate::linalg::matmul::matmul(
                    rtt.rb_mut(),
                    r_tilde.rb().transpose(),
                    t.rb(),
                    None,
                    E::faer_one(),
                    parallelism,
                );
                let (mut temp, _) = temp_mat_uninit::<E>(k, k, stack.rb_mut());
                crate::perm::permute_rows(temp.rb_mut(), rtt.rb(), row_perm);
                crate::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                    rtv.rb().get(..rank, ..rank),
                    temp.rb_mut().get_mut(..rank, ..),
                    parallelism,
                );
                crate::linalg::triangular_solve::solve_upper_triangular_in_place(
                    rtv.rb().get(..rank, ..rank),
                    temp.rb_mut().get_mut(..rank, ..),
                    parallelism,
                );
                temp.rb_mut().get_mut(rank.., ..).fill_zero();
                crate::perm::permute_rows(rtt.rb_mut(), temp.rb(), col_perm.inverse());
            }

            let beta = rtt.rb();
            let mut tmp = v;
            crate::linalg::matmul::matmul(
                tmp.rb_mut(),
                p.rb(),
                beta.rb(),
                None,
                E::faer_one(),
                parallelism,
            );
            zipped_rw!(&mut p, &r, &tmp)
                .for_each(|unzipped!(mut p, r, tmp)| p.write(r.read().faer_sub(tmp.read())));
        }
        Err(BicgError::NoConvergence {
            abs_residual,
            rel_residual: abs_residual.faer_div(b_norm),
        })
    }
    implementation(
        out,
        &left_precond,
        &right_precond,
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
    use super::*;
    use crate::mat;
    use dyn_stack::GlobalPodBuffer;
    use equator::assert;
    use rand::prelude::*;

    #[test]
    fn test_bicgstab() {
        let ref mut rng = StdRng::seed_from_u64(0);

        let ref A = mat![[2.5, -1.0], [1.0, 3.1]];
        let ref sol = mat![[2.1, 2.1], [4.1, 3.2]];
        let ref rhs = A * sol;
        let ref mut diag = Mat::<f64>::identity(2, 2);
        for i in 0..2 {
            diag[(i, i)] = f64::exp(rand::distributions::Standard.sample(rng));
        }
        let ref diag = *diag;

        let ref mut out = Mat::<f64>::zeros(2, sol.ncols());
        let mut params = BicgParams::default();
        params.max_iters = 10;
        let result = bicgstab(
            out.as_mut(),
            diag.as_ref(),
            diag.as_ref(),
            A.as_ref(),
            rhs.as_ref(),
            params,
            |_| {},
            Parallelism::None,
            PodStack::new(&mut GlobalPodBuffer::new(
                bicgstab_req(
                    diag.as_ref(),
                    diag.as_ref(),
                    A.as_ref(),
                    sol.ncols(),
                    Parallelism::None,
                )
                .unwrap(),
            )),
        );
        let ref out = *out;
        dbg!(&result);

        assert!(result.is_ok());
        assert!((A * out - rhs).norm_l2() <= params.rel_tolerance * rhs.norm_l2());
    }
}
