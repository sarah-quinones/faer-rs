use crate::{
    linalg::{householder, matmul::matmul, qr, temp_mat_req, temp_mat_uninit},
    linop::{BiLinOp, BiPrecond, InitialGuessStatus},
    prelude::*,
    utils::DivCeil,
    ComplexField, Conj, Parallelism, RealField,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use equator::{assert, debug_assert};
use reborrow::*;

#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct LsmrParams<E: ComplexField> {
    pub initial_guess: InitialGuessStatus,
    pub abs_tolerance: E::Real,
    pub rel_tolerance: E::Real,
    pub max_iters: usize,
}

impl<E: ComplexField> Default for LsmrParams<E> {
    #[inline]
    fn default() -> Self {
        Self {
            initial_guess: InitialGuessStatus::MaybeNonZero,
            abs_tolerance: E::Real::faer_zero(),
            rel_tolerance: E::Real::faer_epsilon().faer_mul(E::Real::faer_from_f64(128.0)),
            max_iters: usize::MAX,
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct LsmrInfo<E: ComplexField> {
    pub abs_residual: E::Real,
    pub rel_residual: E::Real,
    pub iter_count: usize,
}

#[derive(Copy, Clone, Debug)]
pub enum LsmrError<E: ComplexField> {
    NoConvergence {
        abs_residual: E::Real,
        rel_residual: E::Real,
    },
}

#[allow(dead_code)]
#[cfg(test)]
fn lsmr_prototype_v2<E: ComplexField>(
    mut x: MatMut<'_, E>,
    _M: MatRef<'_, E>,
    A: MatRef<'_, E>,
    b: MatRef<'_, E>,
    params: LsmrParams<E>,
) {
    let mut u = vec![];
    let mut v = vec![];
    let mut alpha = vec![];
    let mut beta = vec![];

    // let m = A.nrows();
    // let n = A.ncols();
    let s = b.ncols();

    let qr = (b - A * &x).qr();
    u.push(qr.compute_thin_q());
    beta.push(qr.compute_thin_r());

    let qr = (A.adjoint() * &u[0]).qr();
    v.push(qr.compute_thin_q());
    alpha.push(qr.compute_thin_r());

    let mut L_kn = Mat::<E>::zeros((params.max_iters + 1) * s, (params.max_iters + 1) * s);
    L_kn.as_mut()
        .submatrix_mut(0, 0, s, s)
        .copy_from(alpha[0].adjoint().to_owned());

    for k in 0..params.max_iters {
        let qr = (A * &v[k] - &u[k] * alpha[k].adjoint()).qr();
        u.push(qr.compute_thin_q());
        beta.push(qr.compute_thin_r());

        let qr = (A.adjoint() * &u[k + 1] - &v[k] * beta[k + 1].adjoint()).qr();
        v.push(qr.compute_thin_q());
        alpha.push(qr.compute_thin_r());

        L_kn.as_mut()
            .submatrix_mut((k + 1) * s, (k + 1) * s, s, s)
            .copy_from(alpha[k + 1].adjoint().to_owned());
        L_kn.as_mut()
            .submatrix_mut((k + 1) * s, k * s, s, s)
            .copy_from(&beta[k + 1]);

        let L_kn = L_kn.as_ref().get(..(k + 2) * s, ..(k + 2) * s);
        let B_k = L_kn.get(.., ..(k + 1) * s);
        let ref R_k = B_k.qr().compute_thin_r();
        let q_k = R_k.adjoint().solve_lower_triangular(crate::concat![
            [Mat::<E>::zeros(k * s, s)], //
            [(&alpha[k + 1] * &beta[k + 1]).adjoint()],
        ]);

        let qr = crate::concat![
            [R_k.adjoint()], //
            [q_k.adjoint()]
        ]
        .qr();
        let Rbar_k = qr.compute_thin_r();
        let Qbar = qr.compute_q().adjoint().to_owned();

        let rhs = &Qbar
            * crate::concat![
                [(&alpha[0] * &beta[0])], //
                [Mat::<E>::zeros((k + 1) * s, s)]
            ];

        let z_k = rhs.get(..(k + 1) * s, ..).to_owned();
        let zetabar = rhs.get((k + 1) * s.., ..).to_owned();
        _ = zetabar;

        let t_k = Rbar_k.solve_upper_triangular(&z_k);
        let y_k = R_k.solve_upper_triangular(&t_k);

        x.fill(E::faer_zero());
        for kk in 0..k + 1 {
            x += &v[kk] * y_k.get(kk * s..(kk + 1) * s, ..);
        }

        let mut V = Mat::<E>::zeros(A.ncols(), (k + 1) * s);
        for kk in 0..(k + 1) {
            V.get_mut(.., kk * s..(kk + 1) * s).copy_from(&v[kk]);
        }
    }
}

#[allow(dead_code)]
#[cfg(test)]
fn lsmr_prototype_v3<E: ComplexField>(
    mut x: MatMut<'_, E>,
    _M: MatRef<'_, E>,
    A: MatRef<'_, E>,
    b: MatRef<'_, E>,
    params: LsmrParams<E>,
) {
    let s = b.ncols();

    let qr = (b - A * &x).qr();
    let mut u = qr.compute_thin_q();
    let mut beta = qr.compute_thin_r();

    let qr = (A.adjoint() * &u).qr();
    let mut v = qr.compute_thin_q();
    let mut alpha = qr.compute_thin_r();

    let mut zetabar = &alpha * &beta;
    let mut alphabar = alpha.clone();
    let mut pbar = Mat::<E>::identity(2 * s, 2 * s);
    let mut theta = Mat::<E>::zeros(s, s);

    let mut w = Mat::<E>::zeros(s, A.ncols());
    let mut wbar = Mat::<E>::zeros(s, A.ncols());

    for _ in 0..params.max_iters {
        let qr = (A * &v - &u * alpha.adjoint()).qr();
        u = qr.compute_thin_q();
        beta = qr.compute_thin_r();

        let v_old = v.clone();
        let qr = (A.adjoint() * &u - &v * beta.adjoint()).qr();
        v = qr.compute_thin_q();
        alpha = qr.compute_thin_r();

        let qr = (crate::concat![[alphabar.adjoint()], [beta]]).qr();
        let p = qr.compute_q().adjoint().to_owned();
        let rho = qr.compute_thin_r();

        let tmp = &alpha * p.get(.., s..).adjoint();
        let thetaold = theta.to_owned();
        theta = tmp.get(.., ..s).to_owned();
        alphabar = tmp.get(.., s..).to_owned();

        let tmp = pbar.get(.., s..) * rho.adjoint();
        let thetabar = tmp.get(..s, ..).adjoint();
        let qr = (crate::concat![[tmp.get(s.., ..)], [theta]]).qr();
        pbar = qr.compute_q().adjoint().to_owned();
        let rhobar = qr.compute_thin_r();

        let tmp = pbar.get(.., ..s) * &zetabar;
        let zeta = tmp.get(..s, ..);
        zetabar.copy_from(tmp.get(s.., ..));

        w = rho
            .adjoint()
            .solve_lower_triangular(v_old.adjoint() - &thetaold * &w);
        wbar = rhobar
            .adjoint()
            .solve_lower_triangular(&w - thetabar * &wbar);
        x += &wbar.adjoint() * &zeta;
    }
}

pub fn lsmr_req<E: ComplexField>(
    right_precond: impl BiPrecond<E>,
    mat: impl BiLinOp<E>,
    rhs_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    fn implementation<E: ComplexField>(
        M: &dyn BiPrecond<E>,
        A: &dyn BiLinOp<E>,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let m = A.nrows();
        let n = A.ncols();
        let mut k = rhs_ncols;

        assert!(k < isize::MAX as usize);
        if k > n {
            k = k.msrv_checked_next_multiple_of(n).unwrap();
        }
        assert!(k < isize::MAX as usize);

        let s = Ord::min(k, Ord::min(n, m));

        let mk = temp_mat_req::<E>(m, k)?;
        let nk = temp_mat_req::<E>(n, k)?;
        let ss = temp_mat_req::<E>(s, s)?;
        let ss2 = temp_mat_req::<E>(2 * s, 2 * s)?;
        let sk = temp_mat_req::<E>(s, k)?;
        let sk2 = temp_mat_req::<E>(2 * s, 2 * k)?;

        let ms_bs = qr::no_pivoting::compute::recommended_blocksize::<E>(m, s);
        let ns_bs = qr::no_pivoting::compute::recommended_blocksize::<E>(n, s);
        let ss_bs = qr::no_pivoting::compute::recommended_blocksize::<E>(2 * s, 2 * s);

        let AT = A.transpose_apply_req(k, parallelism)?;
        let A = A.apply_req(k, parallelism)?;
        let MT = M.transpose_apply_in_place_req(k, parallelism)?;
        let M = M.apply_in_place_req(k, parallelism)?;

        let m_qr = StackReq::try_any_of([
            temp_mat_req::<E>(ms_bs, s)?,
            qr::no_pivoting::compute::qr_in_place_req::<E>(
                m,
                s,
                ms_bs,
                parallelism,
                Default::default(),
            )?,
            householder::apply_block_householder_sequence_on_the_left_in_place_req::<E>(
                m, ms_bs, s,
            )?,
        ])?;

        let n_qr = StackReq::try_any_of([
            temp_mat_req::<E>(ns_bs, s)?,
            qr::no_pivoting::compute::qr_in_place_req::<E>(
                n,
                s,
                ns_bs,
                parallelism,
                Default::default(),
            )?,
            householder::apply_block_householder_sequence_on_the_left_in_place_req::<E>(
                n, ns_bs, s,
            )?,
        ])?;

        let s_qr = StackReq::try_any_of([
            temp_mat_req::<E>(ss_bs, s)?,
            qr::no_pivoting::compute::qr_in_place_req::<E>(
                2 * s,
                2 * s,
                ss_bs,
                parallelism,
                Default::default(),
            )?,
            householder::apply_block_householder_sequence_on_the_left_in_place_req::<E>(
                2 * s,
                ss_bs,
                2 * s,
            )?,
        ])?;

        StackReq::try_all_of([
            mk,  // u
            nk,  // v
            sk,  // beta
            sk,  // alpha
            sk,  // zetabar
            sk,  // alphabar
            sk,  // theta
            sk2, // pbar_adjoint
            nk,  // vold
            StackReq::try_any_of([StackReq::try_all_of([
                mk,
                StackReq::try_any_of([A, M, m_qr])?,
            ])?])?,
            StackReq::try_any_of([StackReq::try_all_of([
                nk,
                StackReq::try_any_of([AT, MT, n_qr])?,
            ])?])?,
            ss2, // p_adjoint
            ss,  // rho
            ss,  // thetaold
            ss,  // rhobar
            ss,  // thetabar
            ss,  // zeta
            ss,  // zetabar
            StackReq::try_all_of([temp_mat_req::<E>(2 * s, 2 * s)?, s_qr])?,
        ])
    }

    implementation(&right_precond, &mat, rhs_ncols, parallelism)
}

#[track_caller]
pub fn lsmr<E: ComplexField>(
    out: MatMut<'_, E>,
    right_precond: impl BiPrecond<E>,
    mat: impl BiLinOp<E>,
    rhs: MatRef<'_, E>,
    params: LsmrParams<E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) -> Result<LsmrInfo<E>, LsmrError<E>> {
    #[track_caller]
    fn implementation<E: ComplexField>(
        mut x: MatMut<'_, E>,
        M: &impl BiPrecond<E>,
        A: &impl BiLinOp<E>,
        b: MatRef<'_, E>,
        params: LsmrParams<E>,
        par: Parallelism,
        stack: PodStack<'_>,
    ) -> Result<LsmrInfo<E>, LsmrError<E>> {
        fn thin_qr<E: ComplexField>(
            mut Q: MatMut<'_, E>,
            mut R: MatMut<'_, E>,
            mut mat: MatMut<'_, E>,
            parallelism: Parallelism,
            stack: PodStack<'_>,
        ) {
            let k = R.nrows();
            let bs = qr::no_pivoting::compute::recommended_blocksize::<E>(mat.nrows(), mat.ncols());
            let (mut house, mut stack) =
                temp_mat_uninit::<E>(bs, Ord::min(mat.nrows(), mat.ncols()), stack);

            qr::no_pivoting::compute::qr_in_place(
                mat.rb_mut(),
                house.rb_mut(),
                parallelism,
                stack.rb_mut(),
                Default::default(),
            );

            R.fill_zero();
            R.copy_from_triangular_upper(mat.rb().get(..k, ..k));
            Q.fill_zero();
            Q.rb_mut()
                .diagonal_mut()
                .column_vector_mut()
                .fill(E::faer_one());
            householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                mat.rb(),
                house.rb(),
                Conj::No,
                Q.rb_mut(),
                parallelism,
                stack.rb_mut(),
            );
        }

        let m = A.nrows();
        let n = A.ncols();
        let mut k = b.ncols();
        {
            let out = x.rb();
            let mat = A;
            let right_precond = M;
            let rhs = b;
            assert!(all(
                right_precond.nrows() == mat.ncols(),
                right_precond.ncols() == mat.ncols(),
                rhs.nrows() == mat.nrows(),
                out.nrows() == mat.ncols(),
                out.ncols() == rhs.ncols(),
            ));
        }

        if m == 0 || n == 0 || k == 0 || core::mem::size_of::<E::Unit>() == 0 {
            x.fill_zero();
            return Ok(LsmrInfo {
                abs_residual: E::Real::faer_zero(),
                rel_residual: E::Real::faer_zero(),
                iter_count: 0,
            });
        }

        debug_assert!(all(
            m < isize::MAX as usize,
            n < isize::MAX as usize,
            k < isize::MAX as usize,
        ));
        let actual_k = k;
        if k > n {
            // pad to avoid last block slowing down the rest
            k = k.msrv_checked_next_multiple_of(n).unwrap();
        }
        debug_assert!(k < isize::MAX as usize);

        let s = Ord::min(k, Ord::min(n, m));

        let mut stack = stack;

        let one = E::faer_one();

        let (mut u, mut stack) = temp_mat_uninit::<E>(m, k, stack.rb_mut());
        let (mut beta, mut stack) = temp_mat_uninit::<E>(s, k, stack.rb_mut());

        let (mut v, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
        let (mut alpha, mut stack) = temp_mat_uninit::<E>(s, k, stack.rb_mut());

        let (mut zetabar, mut stack) = temp_mat_uninit::<E>(s, k, stack.rb_mut());
        let (mut alphabar, mut stack) = temp_mat_uninit::<E>(s, k, stack.rb_mut());
        let (mut theta, mut stack) = temp_mat_uninit::<E>(s, k, stack.rb_mut());
        let (mut pbar_adjoint, mut stack) = temp_mat_uninit::<E>(2 * s, 2 * k, stack.rb_mut());

        let (mut w, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
        let (mut wbar, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());

        {
            let (mut qr, mut stack) = temp_mat_uninit::<E>(m, k, stack.rb_mut());
            if params.initial_guess == InitialGuessStatus::Zero {
                qr.rb_mut().get_mut(.., ..actual_k).copy_from(b);
                qr.rb_mut().get_mut(.., actual_k..).fill_zero();
            } else {
                A.apply(
                    qr.rb_mut().rb_mut().get_mut(.., ..actual_k),
                    x.rb(),
                    par,
                    stack.rb_mut(),
                );
                zipped!(qr.rb_mut().get_mut(.., ..actual_k), &b).for_each(
                    |unzipped!(mut ax, b)| ax.write(b.read().canonicalize().faer_sub(ax.read())),
                );
                qr.rb_mut().get_mut(.., actual_k..).fill_zero();
            }
            let mut start = 0;
            while start < k {
                let end = Ord::min(k - start, s) + start;
                let len = end - start;
                thin_qr(
                    u.rb_mut().get_mut(.., start..end),
                    beta.rb_mut().get_mut(..len, start..end),
                    qr.rb_mut().get_mut(.., start..end),
                    par,
                    stack.rb_mut(),
                );
                start = end;
            }
        }

        {
            let (mut qr, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
            A.adjoint_apply(qr.rb_mut(), u.rb(), par, stack.rb_mut());
            M.adjoint_apply_in_place(qr.rb_mut(), par, stack.rb_mut());
            let mut start = 0;
            while start < k {
                let end = Ord::min(k - start, s) + start;
                let len = end - start;
                thin_qr(
                    v.rb_mut().get_mut(.., start..end),
                    alpha.rb_mut().get_mut(..len, start..end),
                    qr.rb_mut().get_mut(.., start..end),
                    par,
                    stack.rb_mut(),
                );
                start = end;
            }
        }

        zetabar.fill_zero();
        let mut start = 0;
        while start < k {
            let end = Ord::min(k - start, s) + start;
            let len = end - start;
            matmul(
                zetabar.rb_mut().get_mut(..len, start..end),
                alpha.rb().get(..len, start..end),
                beta.rb().get(..len, start..end),
                None,
                one,
                par,
            );
            start = end;
        }
        alphabar.copy_from(&alpha);
        pbar_adjoint.fill_zero();
        let mut start = 0;
        while start < k {
            let end = Ord::min(k - start, s) + start;
            let len = end - start;
            pbar_adjoint
                .rb_mut()
                .get_mut(..2 * len, 2 * start..2 * end)
                .diagonal_mut()
                .column_vector_mut()
                .fill(one);
            start = end;
        }
        theta.fill_zero();
        w.fill_zero();
        wbar.fill_zero();

        let mut norm;
        let norm_ref = if params.initial_guess == InitialGuessStatus::Zero {
            norm = zetabar.norm_l2();
            norm
        } else {
            norm = zetabar.norm_l2();
            let (mut tmp, mut stack) = temp_mat_uninit::<E>(n, actual_k, stack.rb_mut());
            A.adjoint_apply(tmp.rb_mut(), b, par, stack.rb_mut());
            M.adjoint_apply_in_place(tmp.rb_mut(), par, stack.rb_mut());
            tmp.norm_l2()
        };
        let threshold = norm_ref.faer_mul(params.rel_tolerance);

        if norm_ref == E::Real::faer_zero() {
            x.fill_zero();
            return Ok(LsmrInfo {
                abs_residual: E::Real::faer_zero(),
                rel_residual: E::Real::faer_zero(),
                iter_count: 0,
            });
        }

        if norm <= threshold {
            return Ok(LsmrInfo {
                abs_residual: E::Real::faer_zero(),
                rel_residual: E::Real::faer_zero(),
                iter_count: 0,
            });
        }

        for iter in 0..params.max_iters {
            let (mut vold, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
            {
                let (mut qr, mut stack) = temp_mat_uninit::<E>(m, k, stack.rb_mut());
                vold.copy_from(&v);
                M.apply_in_place(v.rb_mut(), par, stack.rb_mut());
                A.apply(qr.rb_mut(), v.rb(), par, stack.rb_mut());

                let mut start = 0;
                while start < k {
                    let s = Ord::min(k - start, s);
                    let end = start + s;
                    matmul(
                        qr.rb_mut().get_mut(.., start..end),
                        u.rb().get(.., start..end),
                        alpha.rb().get(..s, start..end).adjoint(),
                        Some(one),
                        one.faer_neg(),
                        par,
                    );
                    thin_qr(
                        u.rb_mut().get_mut(.., start..end),
                        beta.rb_mut().get_mut(..s, start..end),
                        qr.rb_mut().get_mut(.., start..end),
                        par,
                        stack.rb_mut(),
                    );
                    start = end;
                }
            }

            {
                let (mut qr, mut stack) = temp_mat_uninit::<E>(n, k, stack.rb_mut());
                A.adjoint_apply(qr.rb_mut(), u.rb(), par, stack.rb_mut());
                M.adjoint_apply_in_place(qr.rb_mut(), par, stack.rb_mut());

                let mut start = 0;
                while start < k {
                    let s = Ord::min(k - start, s);
                    let end = start + s;
                    matmul(
                        qr.rb_mut().get_mut(.., start..end),
                        vold.rb().get(.., start..end),
                        beta.rb().get(..s, start..end).adjoint(),
                        Some(one),
                        one.faer_neg(),
                        par,
                    );

                    // now contains M v_old
                    vold.rb_mut()
                        .get_mut(.., start..end)
                        .copy_from(v.rb().get(.., start..end));

                    thin_qr(
                        v.rb_mut().get_mut(.., start..end),
                        alpha.rb_mut().get_mut(..s, start..end),
                        qr.rb_mut().get_mut(.., start..end),
                        par,
                        stack.rb_mut(),
                    );
                    start = end;
                }
            }

            let mut Mvold = vold;

            let mut start = 0;
            while start < k {
                let s = Ord::min(k - start, s);
                let end = start + s;

                let mut x = x.rb_mut().get_mut(.., start..Ord::min(actual_k, end));
                let mut Mvold = Mvold.rb_mut().get_mut(.., start..end);
                let mut w = w.rb_mut().get_mut(.., start..end);
                let mut wbar = wbar.rb_mut().get_mut(.., start..end);

                let alpha = alpha.rb_mut().get_mut(..s, start..end);
                let beta = beta.rb_mut().get_mut(..s, start..end);
                let mut zetabar = zetabar.rb_mut().get_mut(..s, start..end);
                let mut alphabar = alphabar.rb_mut().get_mut(..s, start..end);
                let mut theta = theta.rb_mut().get_mut(..s, start..end);
                let mut pbar_adjoint = pbar_adjoint.rb_mut().get_mut(..2 * s, 2 * start..2 * end);

                let (mut p_adjoint, mut stack) = temp_mat_uninit::<E>(2 * s, 2 * s, stack.rb_mut());

                let (mut rho, mut stack) = temp_mat_uninit::<E>(s, s, stack.rb_mut());
                let (mut thetaold, mut stack) = temp_mat_uninit::<E>(s, s, stack.rb_mut());
                let (mut rhobar, mut stack) = temp_mat_uninit::<E>(s, s, stack.rb_mut());
                let (mut thetabar, mut stack) = temp_mat_uninit::<E>(s, s, stack.rb_mut());
                let (mut zeta, mut stack) = temp_mat_uninit::<E>(s, s, stack.rb_mut());
                let (mut zetabar_tmp, mut stack) = temp_mat_uninit::<E>(s, s, stack.rb_mut());

                {
                    let (mut qr, mut stack) = temp_mat_uninit::<E>(2 * s, s, stack.rb_mut());
                    qr.rb_mut()
                        .get_mut(..s, ..)
                        .copy_from(alphabar.rb().adjoint());
                    qr.rb_mut().get_mut(s.., ..).copy_from(&beta);
                    thin_qr(
                        p_adjoint.rb_mut(),
                        rho.rb_mut(),
                        qr.rb_mut(),
                        par,
                        stack.rb_mut(),
                    );
                }

                thetaold.copy_from(&theta);
                matmul(
                    theta.rb_mut(),
                    alpha.rb(),
                    p_adjoint.rb().get(s.., ..s),
                    None,
                    one,
                    par,
                );
                matmul(
                    alphabar.rb_mut(),
                    alpha.rb(),
                    p_adjoint.rb().get(s.., s..),
                    None,
                    one,
                    par,
                );

                matmul(
                    thetabar.rb_mut(),
                    rho.rb(),
                    pbar_adjoint.rb().get(s.., ..s),
                    None,
                    one,
                    par,
                );
                {
                    let (mut qr, mut stack) = temp_mat_uninit::<E>(2 * s, s, stack.rb_mut());
                    matmul(
                        qr.rb_mut().get_mut(..s, ..),
                        pbar_adjoint.rb().adjoint().get(s.., s..),
                        rho.rb().adjoint(),
                        None,
                        one,
                        par,
                    );
                    qr.rb_mut().get_mut(s.., ..).copy_from(&theta);
                    thin_qr(
                        pbar_adjoint.rb_mut(),
                        rhobar.rb_mut(),
                        qr.rb_mut(),
                        par,
                        stack.rb_mut(),
                    );
                }

                matmul(
                    zeta.rb_mut(),
                    pbar_adjoint.rb().adjoint().get(..s, ..s),
                    zetabar.rb(),
                    None,
                    one,
                    par,
                );
                matmul(
                    zetabar_tmp.rb_mut(),
                    pbar_adjoint.rb().adjoint().get(s.., ..s),
                    zetabar.rb(),
                    None,
                    one,
                    par,
                );
                zetabar.copy_from(&zetabar_tmp);

                matmul(
                    Mvold.rb_mut(),
                    w.rb(),
                    thetaold.rb().adjoint(),
                    Some(one),
                    one.faer_neg(),
                    par,
                );
                crate::linalg::triangular_solve::solve_lower_triangular_in_place(
                    rho.rb().transpose(),
                    Mvold.rb_mut().transpose_mut(),
                    par,
                );
                w.copy_from(&Mvold);

                matmul(
                    Mvold.rb_mut(),
                    wbar.rb(),
                    thetabar.rb().adjoint(),
                    Some(one),
                    one.faer_neg(),
                    par,
                );
                crate::linalg::triangular_solve::solve_lower_triangular_in_place(
                    rhobar.rb().transpose(),
                    Mvold.rb_mut().transpose_mut(),
                    par,
                );
                wbar.copy_from(&Mvold);

                let actual_s = x.ncols();
                matmul(
                    x.rb_mut(),
                    wbar.rb(),
                    zeta.rb().get(.., ..actual_s),
                    Some(one),
                    one,
                    par,
                );
                start = end;
            }
            norm = zetabar.norm_l2();
            if norm <= threshold {
                return Ok(LsmrInfo {
                    abs_residual: norm,
                    rel_residual: norm.faer_div(norm_ref),
                    iter_count: iter + 1,
                });
            }
        }

        Err(LsmrError::NoConvergence {
            abs_residual: norm,
            rel_residual: norm.faer_div(norm_ref),
        })
    }
    implementation(out, &right_precond, &mat, rhs, params, parallelism, stack)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dyn_stack::GlobalPodBuffer;
    use equator::assert;
    use rand::prelude::*;

    #[test]
    fn test_lsmr() {
        let ref mut rng = StdRng::seed_from_u64(0);
        let m = 100;
        let n = 80;
        for k in [1, 2, 4, 7, 10, 40, 80, 100] {
            let A: Mat<c64> = crate::stats::StandardNormalMat { nrows: m, ncols: n }.sample(rng);
            let b: Mat<c64> = crate::stats::StandardNormalMat { nrows: m, ncols: k }.sample(rng);
            let k = b.ncols();

            let ref mut diag = crate::scale(c64::new(2.0, 0.0)) * Mat::<c64>::identity(n, n);
            for i in 0..n {
                diag[(i, i)] = (128.0 * f64::exp(rand::distributions::Standard.sample(rng))).into();
            }
            for i in 0..n - 1 {
                diag[(i + 1, i)] = f64::exp(rand::distributions::Standard.sample(rng)).into();
            }

            let params = LsmrParams::default();

            let rand = crate::stats::StandardNormalMat { nrows: n, ncols: k };
            let mut out = rand.sample(rng);

            let result = lsmr(
                out.as_mut(),
                diag.as_ref(),
                A.as_ref(),
                b.as_ref(),
                params,
                Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(
                    lsmr_req(diag.as_ref(), A.as_ref(), k, Parallelism::None).unwrap(),
                )),
            );
            assert!(result.is_ok());
            let result = result.unwrap();
            assert!(result.iter_count <= (4 * n).msrv_div_ceil(Ord::min(k, n)));
        }
    }

    #[test]
    fn test_lsmr_breakdown() {
        let ref mut rng = StdRng::seed_from_u64(0);
        let m = 100;
        let n = 80;
        for k in [1, 2, 4, 7, 10, 40, 80, 100] {
            let A: Mat<c64> = crate::stats::StandardNormalMat { nrows: m, ncols: n }.sample(rng);
            let b: Mat<c64> = crate::stats::StandardNormalMat { nrows: m, ncols: k }.sample(rng);
            let b = crate::concat![[b, b]];
            let k = b.ncols();

            let ref mut diag = crate::scale(c64::new(2.0, 0.0)) * Mat::<c64>::identity(n, n);
            for i in 0..n {
                diag[(i, i)] = (128.0 * f64::exp(rand::distributions::Standard.sample(rng))).into();
            }
            for i in 0..n - 1 {
                diag[(i + 1, i)] = f64::exp(rand::distributions::Standard.sample(rng)).into();
            }

            let params = LsmrParams::default();

            let rand = crate::stats::StandardNormalMat { nrows: n, ncols: k };
            let mut out = rand.sample(rng);
            let result = lsmr(
                out.as_mut(),
                diag.as_ref(),
                A.as_ref(),
                b.as_ref(),
                params,
                Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(
                    lsmr_req(diag.as_ref(), A.as_ref(), k, Parallelism::None).unwrap(),
                )),
            );
            assert!(result.is_ok());
            let result = result.unwrap();
            assert!(result.iter_count <= (4 * n).msrv_div_ceil(Ord::min(k, n)));
        }
    }
}
