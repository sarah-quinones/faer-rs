use crate::internal_prelude::*;
use pulp::Simd;

#[math]
fn rank_update_step_simd<'N, 'R, T: ComplexField>(
    L: ColMut<'_, T, Dim<'N>, ContiguousFwd>,
    W: MatMut<'_, T, Dim<'N>, Dim<'R>, ContiguousFwd>,
    p: ColRef<'_, T, Dim<'R>>,
    beta: ColRef<'_, T, Dim<'R>>,
    align_offset: usize,
) {
    struct Impl<'a, 'N, 'R, T: ComplexField> {
        L: ColMut<'a, T, Dim<'N>, ContiguousFwd>,
        W: MatMut<'a, T, Dim<'N>, Dim<'R>, ContiguousFwd>,
        p: ColRef<'a, T, Dim<'R>>,
        beta: ColRef<'a, T, Dim<'R>>,
        align_offset: usize,
    }

    impl<'a, 'N, 'R, T: ComplexField> pulp::WithSimd for Impl<'a, 'N, 'R, T> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let Self {
                L,
                W,
                p,
                beta,
                align_offset,
            } = self;

            let mut L = L;
            let mut W = W;
            let N = W.nrows();
            let R = W.ncols();

            let simd = SimdCtx::<T, S>::new_align(T::simd_ctx(simd), N, align_offset);
            let (head, body, tail) = simd.indices();

            let mut iter = R.indices();
            let (i0, i1, i2, i3) = (iter.next(), iter.next(), iter.next(), iter.next());

            match (i0, i1, i2, i3) {
                (Some(i0), None, None, None) => {
                    let p0 = simd.splat(&p[i0]);
                    let beta0 = simd.splat(&beta[i0]);

                    macro_rules! simd {
                        ($i: expr) => {{
                            let i = $i;
                            let mut l = simd.read(L.rb(), i);
                            let mut w0 = simd.read(W.rb().col(i0), i);

                            w0 = simd.mul_add(p0, l, w0);
                            l = simd.mul_add(beta0, w0, l);

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
                    let (p0, p1) = (simd.splat(&p[i0]), simd.splat(&p[i1]));
                    let (beta0, beta1) = (simd.splat(&beta[i0]), simd.splat(&beta[i1]));

                    macro_rules! simd {
                        ($i: expr) => {{
                            let i = $i;
                            let mut l = simd.read(L.rb(), i);
                            let mut w0 = simd.read(W.rb().col(i0), i);
                            let mut w1 = simd.read(W.rb().col(i1), i);

                            w0 = simd.mul_add(p0, l, w0);
                            l = simd.mul_add(beta0, w0, l);
                            w1 = simd.mul_add(p1, l, w1);
                            l = simd.mul_add(beta1, w1, l);

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
                    let (p0, p1, p2) = (simd.splat(&p[i0]), simd.splat(&p[i1]), simd.splat(&p[i2]));
                    let (beta0, beta1, beta2) = (
                        simd.splat(&beta[i0]),
                        simd.splat(&beta[i1]),
                        simd.splat(&beta[i2]),
                    );

                    macro_rules! simd {
                        ($i: expr) => {{
                            let i = $i;
                            let mut l = simd.read(L.rb(), i);
                            let mut w0 = simd.read(W.rb().col(i0), i);
                            let mut w1 = simd.read(W.rb().col(i1), i);
                            let mut w2 = simd.read(W.rb().col(i2), i);

                            w0 = simd.mul_add(p0, l, w0);
                            l = simd.mul_add(beta0, w0, l);
                            w1 = simd.mul_add(p1, l, w1);
                            l = simd.mul_add(beta1, w1, l);
                            w2 = simd.mul_add(p2, l, w2);
                            l = simd.mul_add(beta2, w2, l);

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
                    let (p0, p1, p2, p3) = (
                        simd.splat(&p[i0]),
                        simd.splat(&p[i1]),
                        simd.splat(&p[i2]),
                        simd.splat(&p[i3]),
                    );
                    let (beta0, beta1, beta2, beta3) = (
                        simd.splat(&beta[i0]),
                        simd.splat(&beta[i1]),
                        simd.splat(&beta[i2]),
                        simd.splat(&beta[i3]),
                    );

                    macro_rules! simd {
                        ($i: expr) => {{
                            let i = $i;
                            let mut l = simd.read(L.rb(), i);
                            let mut w0 = simd.read(W.rb().col(i0), i);
                            let mut w1 = simd.read(W.rb().col(i1), i);
                            let mut w2 = simd.read(W.rb().col(i2), i);
                            let mut w3 = simd.read(W.rb().col(i3), i);

                            w0 = simd.mul_add(p0, l, w0);
                            l = simd.mul_add(beta0, w0, l);
                            w1 = simd.mul_add(p1, l, w1);
                            l = simd.mul_add(beta1, w1, l);
                            w2 = simd.mul_add(p2, l, w2);
                            l = simd.mul_add(beta2, w2, l);
                            w3 = simd.mul_add(p3, l, w3);
                            l = simd.mul_add(beta3, w3, l);

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
        L,
        W,
        p,
        beta,
        align_offset,
    })
}

#[math]
fn rank_update_step_fallback<'N, 'R, T: ComplexField>(
    L: ColMut<'_, T, Dim<'N>>,
    W: MatMut<'_, T, Dim<'N>, Dim<'R>>,
    p: ColRef<'_, T, Dim<'R>>,
    beta: ColRef<'_, T, Dim<'R>>,
) {
    let mut L = L;
    let mut W = W;
    let N = W.nrows();
    let R = W.ncols();

    let body = N.indices();

    let mut iter = R.indices();
    let (i0, i1, i2, i3) = (iter.next(), iter.next(), iter.next(), iter.next());

    match (i0, i1, i2, i3) {
        (Some(i0), None, None, None) => {
            let p0 = &p[i0];
            let beta0 = &beta[i0];

            for i in body {
                {
                    let mut l = copy(L[i]);
                    let mut w0 = copy(W[(i, i0)]);

                    w0 = p0 * l + w0;
                    l = beta0 * w0 + l;

                    L[i] = l;
                    W[(i, i0)] = w0;
                }
            }
        }
        (Some(i0), Some(i1), None, None) => {
            let (p0, p1) = (&p[i0], &p[i1]);
            let (beta0, beta1) = (&beta[i0], &beta[i1]);

            for i in body {
                {
                    let mut l = copy(L[i]);
                    let mut w0 = copy(W[(i, i0)]);
                    let mut w1 = copy(W[(i, i1)]);

                    w0 = p0 * l + w0;
                    l = beta0 * w0 + l;
                    w1 = p1 * l + w1;
                    l = beta1 * w1 + l;

                    L[i] = l;
                    W[(i, i0)] = w0;
                    W[(i, i1)] = w1;
                }
            }
        }
        (Some(i0), Some(i1), Some(i2), None) => {
            let (p0, p1, p2) = (&p[i0], &p[i1], &p[i2]);
            let (beta0, beta1, beta2) = (&beta[i0], &beta[i1], &beta[i2]);

            for i in body {
                {
                    let mut l = copy(L[i]);
                    let mut w0 = copy(W[(i, i0)]);
                    let mut w1 = copy(W[(i, i1)]);
                    let mut w2 = copy(W[(i, i2)]);

                    w0 = p0 * l + w0;
                    l = beta0 * w0 + l;
                    w1 = p1 * l + w1;
                    l = beta1 * w1 + l;
                    w2 = p2 * l + w2;
                    l = beta2 * w2 + l;

                    L[i] = l;
                    W[(i, i0)] = w0;
                    W[(i, i1)] = w1;
                    W[(i, i2)] = w2;
                }
            }
        }
        (Some(i0), Some(i1), Some(i2), Some(i3)) => {
            let (p0, p1, p2, p3) = (&p[i0], &p[i1], &p[i2], &p[i3]);
            let (beta0, beta1, beta2, beta3) = (&beta[i0], &beta[i1], &beta[i2], &beta[i3]);

            for i in body {
                {
                    let mut l = copy(L[i]);
                    let mut w0 = copy(W[(i, i0)]);
                    let mut w1 = copy(W[(i, i1)]);
                    let mut w2 = copy(W[(i, i2)]);
                    let mut w3 = copy(W[(i, i3)]);

                    w0 = p0 * l + w0;
                    l = beta0 * w0 + l;
                    w1 = p1 * l + w1;
                    l = beta1 * w1 + l;
                    w2 = p2 * l + w2;
                    l = beta2 * w2 + l;
                    w3 = p3 * l + w3;
                    l = beta3 * w3 + l;

                    L[i] = l;
                    W[(i, i0)] = w0;
                    W[(i, i1)] = w1;
                    W[(i, i2)] = w2;
                    W[(i, i3)] = w3;
                }
            }
        }
        _ => panic!(),
    }
}

struct RankRUpdate<'a, 'N, 'R, T: ComplexField> {
    ld: MatMut<'a, T, Dim<'N>, Dim<'N>>,
    w: MatMut<'a, T, Dim<'N>, Dim<'R>>,
    alpha: ColMut<'a, T, Dim<'R>>,
    r: &'a mut dyn FnMut() -> IdxInc<'R>,
}

impl<'N, 'R, T: ComplexField> RankRUpdate<'_, 'N, 'R, T> {
    // On the Modification of LDLT Factorizations
    // By R. Fletcher and M. J. D. Powell
    // https://www.ams.org/journals/mcom/1974-28-128/S0025-5718-1974-0359297-1/S0025-5718-1974-0359297-1.pdf

    #[math]
    fn run(self) {
        let Self {
            mut ld,
            mut w,
            mut alpha,
            r,
        } = self;

        let N = w.nrows();
        let K = w.ncols();

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

                    let mut r_next = IdxInc::ZERO;
                    while let Some(r) = R.try_check(*r_next) {
                        r_next = R.advance(r, BLOCKSIZE);

                        ghost_tree!(W_FULL(R0), {
                            let (l![r0], _) = R.split(l![r.into()..r_next], W_FULL);

                            stack_mat!(p, r0.len(), 1, BLOCKSIZE, 1, T);
                            stack_mat!(beta, r0.len(), 1, BLOCKSIZE, 1, T);

                            let mut p = p.rb_mut().col_mut(0);
                            let mut beta = beta.rb_mut().col_mut(0);

                            for k in r0 {
                                let p = p.rb_mut().at_mut(r0.from_global(k));
                                let beta = beta.rb_mut().at_mut(r0.from_global(k));
                                let alpha = alpha.rb_mut().at_mut(k.local());
                                let d = L_col.rb_mut().at_mut(j);

                                let w = W.rb().col(k.local());

                                *p = copy(w[j]);

                                let alpha_conj_p = *alpha * conj(*p);
                                let new_d = real(*d) + real(mul(alpha_conj_p, *p));
                                *beta = mul_real(alpha_conj_p, recip(new_d));

                                *alpha = from_real(real(*alpha) - new_d * abs2(*beta));
                                *d = from_real(new_d);
                                *p = -p;
                            }

                            let mut L_col = L_col.rb_mut().row_segment_mut(tail);
                            let mut W_col = W.rb_mut().col_segment_mut(r0).row_segment_mut(tail);

                            if const { T::SIMD_CAPABILITIES.is_simd() } {
                                if let (Some(L_col), Some(W_col)) = (
                                    L_col.rb_mut().try_as_col_major_mut(),
                                    W_col.rb_mut().try_as_col_major_mut(),
                                ) {
                                    rank_update_step_simd(
                                        L_col,
                                        W_col,
                                        p.rb(),
                                        beta.rb(),
                                        simd_align(*j.next()),
                                    );
                                } else {
                                    rank_update_step_fallback(L_col, W_col, p.rb(), beta.rb());
                                }
                            } else {
                                rank_update_step_fallback(L_col, W_col, p.rb(), beta.rb());
                            }
                        });
                    }
                });
            });
        }
    }
}

#[track_caller]
pub fn rank_r_update_clobber<'N, 'R, T: ComplexField>(
    cholesky_factors: MatMut<'_, T, Dim<'N>, Dim<'N>>,
    w: MatMut<'_, T, Dim<'N>, Dim<'R>>,
    alpha: DiagMut<'_, T, Dim<'R>>,
) {
    let N = cholesky_factors.nrows();
    let R = w.ncols();

    if *N == 0 {
        return;
    }

    RankRUpdate {
        ld: cholesky_factors,
        w,
        alpha: alpha.column_vector_mut(),
        r: &mut || R.end(),
    }
    .run();
}

#[cfg(test)]
mod tests {
    use dyn_stack::GlobalMemBuffer;

    use super::*;
    use crate::{assert, c64, stats::prelude::*, utils::approx::*, Col, Mat};

    #[test]
    fn test_rank_update() {
        let rng = &mut StdRng::seed_from_u64(0);

        let approx_eq = CwiseMat(ApproxEq {
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
                    alpha.column_vector_mut()[j].im = 0.0;
                }

                let A = &A * &A.adjoint();
                let A_new = &A + &W * &alpha * &W.adjoint();

                let A = A.as_ref();
                let A_new = A_new.as_ref();

                let mut L = A.cloned();
                let mut L = L.as_mut();

                linalg::cholesky::ldlt::factor::cholesky_in_place(
                    L.rb_mut(),
                    default(),
                    Par::Seq,
                    DynStack::new(&mut GlobalMemBuffer::new(
                        linalg::cholesky::ldlt::factor::cholesky_in_place_scratch::<c64>(
                            *N,
                            Par::Seq,
                            auto!(c64),
                        )
                        .unwrap(),
                    )),
                    auto!(c64),
                )
                .unwrap();

                linalg::cholesky::ldlt::update::rank_r_update_clobber(
                    L.rb_mut(),
                    W.as_mut(),
                    alpha.as_mut(),
                );
                let D = L.as_mut().diagonal().column_vector().as_mat().cloned();
                let D = D.col(0).as_diagonal();

                for j in N.indices() {
                    for i in IdxInc::ZERO.to(j.excl()) {
                        L[(i, j)] = c64::ZERO;
                    }
                    L[(j, j)] = c64::ONE;
                }
                let L = L.as_ref();

                assert!(A_new ~ L * D * L.adjoint());
            }
        }
    }
}
