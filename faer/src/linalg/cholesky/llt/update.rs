use crate::internal_prelude::*;
use pulp::Simd;

use super::factor::LltError;

#[math]
fn rank_update_step_simd<'N, 'R, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    L: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    W: MatMut<'_, C, T, Dim<'N>, Dim<'R>, ContiguousFwd>,
    p: ColRef<'_, C, T, Dim<'R>>,
    beta: ColRef<'_, C, T, Dim<'R>>,
    gamma: ColRef<'_, C, T, Dim<'R>>,
    align_offset: usize,
) {
    struct Impl<'a, 'N, 'R, C: ComplexContainer, T: ComplexField<C>> {
        ctx: &'a Ctx<C, T>,
        L: ColMut<'a, C, T, Dim<'N>, ContiguousFwd>,
        W: MatMut<'a, C, T, Dim<'N>, Dim<'R>, ContiguousFwd>,
        p: ColRef<'a, C, T, Dim<'R>>,
        beta: ColRef<'a, C, T, Dim<'R>>,
        gamma: ColRef<'a, C, T, Dim<'R>>,
        align_offset: usize,
    }

    impl<'a, 'N, 'R, C: ComplexContainer, T: ComplexField<C>> pulp::WithSimd
        for Impl<'a, 'N, 'R, C, T>
    {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let Self {
                ctx,
                L,
                W,
                p,
                beta,
                gamma,
                align_offset,
            } = self;

            let mut L = L;
            let mut W = W;
            let N = W.nrows();
            let R = W.ncols();

            let simd = SimdCtx::<C, T, S>::new_align(T::simd_ctx(ctx, simd), N, align_offset);
            let (head, body, tail) = simd.indices();

            let mut iter = R.indices();
            let (i0, i1, i2, i3) = (iter.next(), iter.next(), iter.next(), iter.next());

            match (i0, i1, i2, i3) {
                (Some(i0), None, None, None) => {
                    let p0 = math(simd.splat(p[i0]));
                    let beta0 = math(simd.splat(beta[i0]));
                    let gamma0 = math(simd.splat_real(real(gamma[i0])));

                    macro_rules! simd {
                        ($i: expr) => {{
                            let i = $i;
                            let mut l = simd.read(L.rb(), i);
                            let mut w0 = simd.read(W.rb().col(i0), i);

                            w0 = simd.mul_add(p0, l, w0);
                            l = simd.mul_add(beta0, w0, simd.mul_real(l, gamma0));

                            simd.write(L.rb_mut(), i, l);
                            simd.write(W.rb_mut().col_mut(i0), i, w0);
                        }};
                    }

                    if let Some(i) = head {
                        simd!(i);
                    }
                    for i in body {
                        simd!(i);
                    }
                    if let Some(i) = tail {
                        simd!(i);
                    }
                }
                (Some(i0), Some(i1), None, None) => {
                    let (p0, p1) = math((simd.splat(p[i0]), simd.splat(p[i1])));
                    let (beta0, beta1) = math((simd.splat(beta[i0]), simd.splat(beta[i1])));
                    let (gamma0, gamma1) = math((
                        simd.splat_real(real(gamma[i0])),
                        simd.splat_real(real(gamma[i1])),
                    ));

                    macro_rules! simd {
                        ($i: expr) => {{
                            let i = $i;
                            let mut l = simd.read(L.rb(), i);
                            let mut w0 = simd.read(W.rb().col(i0), i);
                            let mut w1 = simd.read(W.rb().col(i1), i);

                            w0 = simd.mul_add(p0, l, w0);
                            l = simd.mul_add(beta0, w0, simd.mul_real(l, gamma0));
                            w1 = simd.mul_add(p1, l, w1);
                            l = simd.mul_add(beta1, w1, simd.mul_real(l, gamma1));

                            simd.write(L.rb_mut(), i, l);
                            simd.write(W.rb_mut().col_mut(i0), i, w0);
                            simd.write(W.rb_mut().col_mut(i1), i, w1);
                        }};
                    }

                    if let Some(i) = head {
                        simd!(i);
                    }
                    for i in body {
                        simd!(i);
                    }
                    if let Some(i) = tail {
                        simd!(i);
                    }
                }
                (Some(i0), Some(i1), Some(i2), None) => {
                    let (p0, p1, p2) =
                        math((simd.splat(p[i0]), simd.splat(p[i1]), simd.splat(p[i2])));
                    let (beta0, beta1, beta2) = math((
                        simd.splat(beta[i0]),
                        simd.splat(beta[i1]),
                        simd.splat(beta[i2]),
                    ));
                    let (gamma0, gamma1, gamma2) = math((
                        simd.splat_real(real(gamma[i0])),
                        simd.splat_real(real(gamma[i1])),
                        simd.splat_real(real(gamma[i2])),
                    ));

                    macro_rules! simd {
                        ($i: expr) => {{
                            let i = $i;
                            let mut l = simd.read(L.rb(), i);
                            let mut w0 = simd.read(W.rb().col(i0), i);
                            let mut w1 = simd.read(W.rb().col(i1), i);
                            let mut w2 = simd.read(W.rb().col(i2), i);

                            w0 = simd.mul_add(p0, l, w0);
                            l = simd.mul_add(beta0, w0, simd.mul_real(l, gamma0));
                            w1 = simd.mul_add(p1, l, w1);
                            l = simd.mul_add(beta1, w1, simd.mul_real(l, gamma1));
                            w2 = simd.mul_add(p2, l, w2);
                            l = simd.mul_add(beta2, w2, simd.mul_real(l, gamma2));

                            simd.write(L.rb_mut(), i, l);
                            simd.write(W.rb_mut().col_mut(i0), i, w0);
                            simd.write(W.rb_mut().col_mut(i1), i, w1);
                            simd.write(W.rb_mut().col_mut(i2), i, w2);
                        }};
                    }

                    if let Some(i) = head {
                        simd!(i);
                    }
                    for i in body {
                        simd!(i);
                    }
                    if let Some(i) = tail {
                        simd!(i);
                    }
                }
                (Some(i0), Some(i1), Some(i2), Some(i3)) => {
                    let (p0, p1, p2, p3) = math((
                        simd.splat(p[i0]),
                        simd.splat(p[i1]),
                        simd.splat(p[i2]),
                        simd.splat(p[i3]),
                    ));
                    let (beta0, beta1, beta2, beta3) = math((
                        simd.splat(beta[i0]),
                        simd.splat(beta[i1]),
                        simd.splat(beta[i2]),
                        simd.splat(beta[i3]),
                    ));
                    let (gamma0, gamma1, gamma2, gamma3) = math((
                        simd.splat_real(real(gamma[i0])),
                        simd.splat_real(real(gamma[i1])),
                        simd.splat_real(real(gamma[i2])),
                        simd.splat_real(real(gamma[i3])),
                    ));

                    macro_rules! simd {
                        ($i: expr) => {{
                            let i = $i;
                            let mut l = simd.read(L.rb(), i);
                            let mut w0 = simd.read(W.rb().col(i0), i);
                            let mut w1 = simd.read(W.rb().col(i1), i);
                            let mut w2 = simd.read(W.rb().col(i2), i);
                            let mut w3 = simd.read(W.rb().col(i3), i);

                            w0 = simd.mul_add(p0, l, w0);
                            l = simd.mul_add(beta0, w0, simd.mul_real(l, gamma0));
                            w1 = simd.mul_add(p1, l, w1);
                            l = simd.mul_add(beta1, w1, simd.mul_real(l, gamma1));
                            w2 = simd.mul_add(p2, l, w2);
                            l = simd.mul_add(beta2, w2, simd.mul_real(l, gamma2));
                            w3 = simd.mul_add(p3, l, w3);
                            l = simd.mul_add(beta3, w3, simd.mul_real(l, gamma3));

                            simd.write(L.rb_mut(), i, l);
                            simd.write(W.rb_mut().col_mut(i0), i, w0);
                            simd.write(W.rb_mut().col_mut(i1), i, w1);
                            simd.write(W.rb_mut().col_mut(i2), i, w2);
                            simd.write(W.rb_mut().col_mut(i3), i, w3);
                        }};
                    }

                    if let Some(i) = head {
                        simd!(i);
                    }
                    for i in body {
                        simd!(i);
                    }
                    if let Some(i) = tail {
                        simd!(i);
                    }
                }
                _ => panic!(),
            }
        }
    }

    T::Arch::default().dispatch(Impl {
        ctx,
        L,
        W,
        p,
        beta,
        gamma,
        align_offset,
    })
}

#[math]
fn rank_update_step_fallback<'N, 'R, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    L: ColMut<'_, C, T, Dim<'N>>,
    W: MatMut<'_, C, T, Dim<'N>, Dim<'R>>,
    p: ColRef<'_, C, T, Dim<'R>>,
    beta: ColRef<'_, C, T, Dim<'R>>,
    gamma: ColRef<'_, C, T, Dim<'R>>,
) {
    let mut L = L;
    let mut W = W;
    let N = W.nrows();
    let R = W.ncols();

    let body = N.indices();

    let mut iter = R.indices();
    let (i0, i1, i2, i3) = (iter.next(), iter.next(), iter.next(), iter.next());
    help!(C);

    match (i0, i1, i2, i3) {
        (Some(i0), None, None, None) => {
            let p0 = math(p[i0]);
            let beta0 = math(beta[i0]);
            let gamma0 = math(gamma[i0]);

            for i in body {
                {
                    let mut l = math(copy(L[i]));
                    let mut w0 = math(copy(W[(i, i0)]));

                    w0 = math(p0 * l + w0);
                    l = math(beta0 * w0 + l * gamma0);

                    write1!(L[i] = l);
                    write1!(W[(i, i0,)] = w0);
                }
            }
        }
        (Some(i0), Some(i1), None, None) => {
            let (p0, p1) = math((p[i0], p[i1]));
            let (beta0, beta1) = math((beta[i0], beta[i1]));
            let (gamma0, gamma1) = math((gamma[i0], gamma[i1]));

            for i in body {
                {
                    let mut l = math(copy(L[i]));
                    let mut w0 = math(copy(W[(i, i0)]));
                    let mut w1 = math(copy(W[(i, i1)]));

                    w0 = math(p0 * l + w0);
                    l = math(beta0 * w0 + l * gamma0);
                    w1 = math(p1 * l + w1);
                    l = math(beta1 * w1 + l * gamma1);

                    write1!(L[i] = l);
                    write1!(W[(i, i0,)] = w0);
                    write1!(W[(i, i1,)] = w1);
                }
            }
        }
        (Some(i0), Some(i1), Some(i2), None) => {
            let (p0, p1, p2) = math((p[i0], p[i1], p[i2]));
            let (beta0, beta1, beta2) = math((beta[i0], beta[i1], beta[i2]));
            let (gamma0, gamma1, gamma2) = math((gamma[i0], gamma[i1], gamma[i2]));

            for i in body {
                {
                    let mut l = math(copy(L[i]));
                    let mut w0 = math(copy(W[(i, i0)]));
                    let mut w1 = math(copy(W[(i, i1)]));
                    let mut w2 = math(copy(W[(i, i2)]));

                    w0 = math(p0 * l + w0);
                    l = math(beta0 * w0 + l * gamma0);
                    w1 = math(p1 * l + w1);
                    l = math(beta1 * w1 + l * gamma1);
                    w2 = math(p2 * l + w2);
                    l = math(beta2 * w2 + l * gamma2);

                    write1!(L[i] = l);
                    write1!(W[(i, i0,)] = w0);
                    write1!(W[(i, i1,)] = w1);
                    write1!(W[(i, i2,)] = w2);
                }
            }
        }
        (Some(i0), Some(i1), Some(i2), Some(i3)) => {
            let (p0, p1, p2, p3) = math((p[i0], p[i1], p[i2], p[i3]));
            let (beta0, beta1, beta2, beta3) = math((beta[i0], beta[i1], beta[i2], beta[i3]));
            let (gamma0, gamma1, gamma2, gamma3) =
                math((gamma[i0], gamma[i1], gamma[i2], gamma[i3]));

            for i in body {
                {
                    let mut l = math(copy(L[i]));
                    let mut w0 = math(copy(W[(i, i0)]));
                    let mut w1 = math(copy(W[(i, i1)]));
                    let mut w2 = math(copy(W[(i, i2)]));
                    let mut w3 = math(copy(W[(i, i3)]));

                    w0 = math(p0 * l + w0);
                    l = math(beta0 * w0 + l * gamma0);
                    w1 = math(p1 * l + w1);
                    l = math(beta1 * w1 + l * gamma1);
                    w2 = math(p2 * l + w2);
                    l = math(beta2 * w2 + l * gamma2);
                    w3 = math(p3 * l + w3);
                    l = math(beta3 * w3 + l * gamma3);

                    write1!(L[i] = l);
                    write1!(W[(i, i0,)] = w0);
                    write1!(W[(i, i1,)] = w1);
                    write1!(W[(i, i2,)] = w2);
                    write1!(W[(i, i3,)] = w3);
                }
            }
        }
        _ => panic!(),
    }
}

struct RankRUpdate<'a, 'N, 'R, C: ComplexContainer, T: ComplexField<C>> {
    ctx: &'a Ctx<C, T>,
    ld: MatMut<'a, C, T, Dim<'N>, Dim<'N>>,
    w: MatMut<'a, C, T, Dim<'N>, Dim<'R>>,
    alpha: ColMut<'a, C, T, Dim<'R>>,
    r: &'a mut dyn FnMut() -> IdxInc<'R>,
}

impl<'N, 'R, C: ComplexContainer, T: ComplexField<C>> RankRUpdate<'_, 'N, 'R, C, T> {
    // On the Modification of LDLT Factorizations
    // By R. Fletcher and M. J. D. Powell
    // https://www.ams.org/journals/mcom/1974-28-128/S0025-5718-1974-0359297-1/S0025-5718-1974-0359297-1.pdf

    #[math]
    fn run(self) -> Result<(), LltError> {
        let Self {
            ctx,
            mut ld,
            mut w,
            mut alpha,
            r,
        } = self;

        let N = w.nrows();
        let K = w.ncols();
        help!(C);

        for j in N.indices() {
            ghost_tree!(FULL(J, TAIL), {
                let (l![col, tail], _) = N.split(l![j, ..], FULL);
                let mut L_col = ld.rb_mut().col_mut(col.local());

                let r = Ord::min((*r)(), K.end());
                ghost_tree!(W_FULL(R), {
                    let (l![R_segment], _) = K.split(l![..r], W_FULL);
                    let R = R_segment.len();
                    let mut W = w.rb_mut().col_segment_mut(R_segment);
                    let mut alpha = alpha.rb_mut().row_segment_mut(R_segment);

                    const BLOCKSIZE: usize = 4;

                    let mut r_next = zero();
                    while let Some(r) = R.try_check(*r_next) {
                        r_next = R.advance(r, BLOCKSIZE);

                        ghost_tree!(W_FULL(R0), {
                            let (l![r0], _) = R.split(l![r.into()..r_next], W_FULL);

                            stack_mat!(ctx, p, r0.len(), 1, BLOCKSIZE, 1, C, T);
                            stack_mat!(ctx, beta, r0.len(), 1, BLOCKSIZE, 1, C, T);
                            stack_mat!(ctx, gamma, r0.len(), 1, BLOCKSIZE, 1, C, T);

                            let mut p = p.rb_mut().col_mut(0);
                            let mut beta = beta.rb_mut().col_mut(0);
                            let mut gamma = gamma.rb_mut().col_mut(0);

                            for k in r0 {
                                let mut p = p.rb_mut().at_mut(r0.from_global(k));
                                let mut beta = beta.rb_mut().at_mut(r0.from_global(k));
                                let mut gamma = gamma.rb_mut().at_mut(r0.from_global(k));

                                let mut alpha = alpha.rb_mut().at_mut(k.local());
                                let mut d = L_col.rb_mut().at_mut(j);

                                let w = W.rb().col(k.local());

                                write1!(p, math(copy(w[j])));

                                let alpha_conj_p = math(alpha * conj(p));
                                let new_d =
                                    math.re(abs2(cx.real(d)) + cx.real(cx.mul(alpha_conj_p, p)));

                                if math.le_zero(new_d) {
                                    return Err(LltError::NonPositivePivot { index: *j });
                                }

                                let new_d = math.re.sqrt(new_d);
                                let d_inv = math.re(recip(cx.real(d)));
                                let new_d_inv = math.re.recip(new_d);

                                write1!(gamma, math.from_real(math.re(new_d * d_inv)));
                                write1!(beta, math(mul_real(alpha_conj_p, new_d_inv)));
                                write1!(p, math(mul_real(-p, d_inv)));

                                write1!(
                                    alpha,
                                    math.re(cx.from_real(cx.real(alpha) - cx.abs2(beta)))
                                );
                                write1!(d, math.from_real(new_d));
                            }

                            let mut L_col = L_col.rb_mut().row_segment_mut(tail);
                            let mut W_col = W.rb_mut().col_segment_mut(r0).row_segment_mut(tail);

                            if const { T::SIMD_CAPABILITIES.is_simd() } {
                                if let (Some(L_col), Some(W_col)) = (
                                    L_col.rb_mut().try_as_col_major_mut(),
                                    W_col.rb_mut().try_as_col_major_mut(),
                                ) {
                                    rank_update_step_simd(
                                        ctx,
                                        L_col,
                                        W_col,
                                        p.rb(),
                                        beta.rb(),
                                        gamma.rb(),
                                        N.next_power_of_two() - *j.next(),
                                    );
                                } else {
                                    rank_update_step_fallback(
                                        ctx,
                                        L_col,
                                        W_col,
                                        p.rb(),
                                        beta.rb(),
                                        gamma.rb(),
                                    );
                                }
                            } else {
                                rank_update_step_fallback(
                                    ctx,
                                    L_col,
                                    W_col,
                                    p.rb(),
                                    beta.rb(),
                                    gamma.rb(),
                                );
                            }
                        });
                    }
                });
            });
        }
        Ok(())
    }
}

#[track_caller]
pub fn rank_r_update_clobber<'N, 'R, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    cholesky_factors: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    w: MatMut<'_, C, T, Dim<'N>, Dim<'R>>,
    alpha: DiagMut<'_, C, T, Dim<'R>>,
) -> Result<(), LltError> {
    let N = cholesky_factors.nrows();
    let R = w.ncols();

    if *N == 0 {
        return Ok(());
    }

    RankRUpdate {
        ctx,
        ld: cholesky_factors,
        w,
        alpha: alpha.column_vector_mut(),
        r: &mut || R.end(),
    }
    .run()
}

#[cfg(test)]
mod tests {
    use dyn_stack::GlobalMemBuffer;
    use faer_traits::Unit;
    use num_complex::ComplexFloat;

    use super::*;
    use crate::{assert, c64, stats::prelude::*, utils::approx::*, Col, Mat};

    #[test]
    fn test_rank_update() {
        let rng = &mut StdRng::seed_from_u64(0);

        let approx_eq = CwiseMat(ApproxEq {
            ctx: ctx::<Ctx<Unit, c64>>(),
            abs_tol: 1e-12,
            rel_tol: 1e-12,
        });

        for r in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10] {
            for n in [2, 4, 8, 15] {
                with_dim!(N, n);
                with_dim!(R, r);

                let A = CwiseMatDistribution {
                    nrows: N,
                    ncols: N,
                    dist: ComplexDistribution::new(StandardNormal, StandardNormal),
                }
                .rand::<Mat<c64, Dim, Dim>>(rng);
                let mut W = CwiseMatDistribution {
                    nrows: N,
                    ncols: R,
                    dist: ComplexDistribution::new(StandardNormal, StandardNormal),
                }
                .rand::<Mat<c64, Dim, Dim>>(rng);
                let mut alpha = CwiseColDistribution {
                    nrows: R,
                    dist: ComplexDistribution::new(StandardNormal, StandardNormal),
                }
                .rand::<Col<c64, Dim>>(rng)
                .into_diagonal();

                for j in R.indices() {
                    alpha.column_vector_mut()[j].re = alpha.column_vector_mut()[j].abs();
                    alpha.column_vector_mut()[j].im = 0.0;
                }

                let A = &A * &A.adjoint();
                let A_new = &A + &W * &alpha * &W.adjoint();

                let A = A.as_ref();
                let A_new = A_new.as_ref();

                let mut L = A.cloned();
                let mut L = L.as_mut();

                linalg::cholesky::llt::factor::cholesky_in_place(
                    &ctx(),
                    L.rb_mut(),
                    default(),
                    Par::Seq,
                    DynStack::new(&mut GlobalMemBuffer::new(
                        linalg::cholesky::llt::factor::cholesky_in_place_scratch::<Unit, c64>(
                            *N,
                            Par::Seq,
                        )
                        .unwrap(),
                    )),
                    Default::default(),
                )
                .unwrap();

                linalg::cholesky::llt::update::rank_r_update_clobber(
                    &ctx(),
                    L.rb_mut(),
                    W.as_mut(),
                    alpha.as_mut(),
                )
                .unwrap();

                for j in N.indices() {
                    for i in zero().to(j.excl()) {
                        L[(i, j)] = c64::ZERO;
                    }
                }
                let L = L.as_ref();

                assert!(A_new ~ L * L.adjoint());
            }
        }
    }
}
