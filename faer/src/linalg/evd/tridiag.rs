use crate::internal_prelude::*;
use linalg::{
    householder,
    matmul::{self, dot, triangular::BlockStructure},
};

/// QR factorization tuning parameters.
#[derive(Copy, Clone)]
#[non_exhaustive]
pub struct TridiagParams {
    /// At which size the parallelism should be disabled.
    pub par_threshold: usize,
}

impl Default for TridiagParams {
    fn default() -> Self {
        Self {
            par_threshold: 192 * 256,
        }
    }
}

#[math]
pub fn tridiag_in_place<'N, 'B, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    H: MatMut<'_, C, T, Dim<'B>, Dim<'N>>,
    par: Par,
    stack: &mut DynStack,
    params: TridiagParams,
) {
    let mut A = A;
    let mut H = H;
    let mut par = par;
    let n = A.nrows();
    let b = H.nrows();

    if *n == 0 {
        return;
    }

    let (mut y, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
    let (mut w, _) = unsafe { temp_mat_uninit(ctx, n, par.degree(), stack) };
    let mut y = y.as_mat_mut().col_mut(0);
    let mut w = w.as_mat_mut();
    help!(C);

    {
        let mut H = H.rb_mut().row_mut(b.idx(0));
        for k in n.indices() {
            ghost_tree!(MAT(HEAD, K, TAIL(K1, END)), {
                let (split @ l![head, _, tail], (mat_x, _, l![_, _, TAIL])) =
                    n.split(l![..k.into(), k, ..], MAT);

                let l![A0, A1, A2] = A.rb_mut().row_segments_mut(split, mat_x);

                let l![_, _, _] = A0.col_segments_mut(split, mat_x);
                let l![_, mut a11, _] = A1.col_segments_mut(split, mat_x);
                let l![A20, mut A21, mut A22] = A2.col_segments_mut(split, mat_x);

                let l![_, y1, mut y2] = y.rb_mut().row_segments_mut(split, mat_x);

                let split = if k.next() != n.end() {
                    let n = tail.len();
                    let (split, (tail_x, _, _)) = n.split(l![n.idx(0), ..], TAIL);
                    Some((split, tail_x))
                } else {
                    None
                };

                if *k > 0 {
                    let p = head.idx(*k - 1);

                    let u2 = A20.rb().col(head.local(p));

                    write1!(a11, math(a11 - y1 - conj(y1)));

                    z!(A21.rb_mut(), u2, y2.rb()).for_each(|uz!(mut a, u, y)| {
                        write1!(a, math(a - conj(y1) * u - y));
                    });
                }

                if k.next() == n.end() {
                    break;
                }

                let rem = *n - *k.next();
                if rem * rem / 2 < params.par_threshold {
                    par = Par::Seq;
                }

                let k1 = tail.idx(*k + 1);
                let (split, tail_x) = split.unwrap();

                let tau_inv;
                {
                    let l![mut a11, mut u2] = A21.rb_mut().row_segments_mut(split, tail_x);
                    let (tau, _) =
                        householder::make_householder_in_place(ctx, rb_mut!(a11), u2.rb_mut());

                    tau_inv = math(re.recip(real(tau)));
                    write1!(H[k1.local()] = tau);

                    let l![mut y1, mut y2] = y2.rb_mut().row_segments_mut(split, tail_x);
                    let l![A1, A2] = A22.rb_mut().row_segments_mut(split, tail_x);
                    let l![mut a11, _] = A1.col_segments_mut(split, tail_x);
                    let l![mut A21, mut A22] = A2.col_segments_mut(split, tail_x);

                    if *k > 0 {
                        let p = head.idx(*k - 1);

                        let l![u1, u2] = A20.rb().col(head.local(p)).row_segments(split);

                        write1!(a11, math(a11 - u1 * conj(y1) - y1 * conj(u1)));

                        z!(A21.rb_mut(), u2.rb(), y2.rb()).for_each(|uz!(mut a, u, y)| {
                            write1!(a, math(a - u * conj(y1) - y * conj(u1)));
                        });

                        let block_struct = BlockStructure::TriangularLower;

                        matmul::triangular::matmul(
                            ctx,
                            A22.rb_mut(),
                            block_struct,
                            Accum::Add,
                            u2.rb().as_mat(),
                            BlockStructure::Rectangular,
                            y2.rb().adjoint().as_mat(),
                            BlockStructure::Rectangular,
                            math(-one()),
                            par,
                        );

                        matmul::triangular::matmul(
                            ctx,
                            A22.rb_mut(),
                            block_struct,
                            Accum::Add,
                            y2.rb().as_mat(),
                            BlockStructure::Rectangular,
                            u2.rb().adjoint().as_mat(),
                            BlockStructure::Rectangular,
                            math(-one()),
                            par,
                        );
                    }

                    write1!(
                        y1,
                        math(
                            a11 + dot::inner_prod(
                                ctx,
                                A21.rb().transpose(),
                                Conj::Yes,
                                u2.rb(),
                                Conj::No
                            )
                        )
                    );

                    z!(y2.rb_mut(), A21.rb()).for_each(|uz!(mut y, a)| write1!(y, math(copy(a))));
                    matmul::triangular::matmul(
                        ctx,
                        y2.rb_mut().as_mat_mut(),
                        BlockStructure::Rectangular,
                        Accum::Add,
                        A22.rb(),
                        BlockStructure::TriangularLower,
                        u2.rb().as_mat(),
                        BlockStructure::Rectangular,
                        math(one()),
                        par,
                    );
                    matmul::triangular::matmul(
                        ctx,
                        y2.rb_mut().as_mat_mut(),
                        BlockStructure::Rectangular,
                        Accum::Add,
                        A22.rb().adjoint(),
                        BlockStructure::StrictTriangularUpper,
                        u2.rb().as_mat(),
                        BlockStructure::Rectangular,
                        math(one()),
                        par,
                    );

                    let b = math(mul_real(
                        mul_pow2(
                            y1 + dot::inner_prod(
                                ctx,
                                u2.rb().transpose(),
                                Conj::Yes,
                                y2.rb(),
                                Conj::No,
                            ),
                            re.from_f64(0.5),
                        ),
                        tau_inv,
                    ));
                    write1!(y1, math(mul_real(y1 - b, tau_inv)));
                    z!(y2.rb_mut(), u2.rb()).for_each(|uz!(mut y, u)| {
                        write1!(y, math(mul_real(y - b * u, tau_inv)));
                    });
                }
            });
        }
    }

    if *n > 0 {
        ghost_tree!(BLOCK, {
            let (block, _) = n.split(n.idx_inc(1).., BLOCK);
            let n = block.len();
            let A = A.rb().subcols(zero(), n).row_segment(block);
            let mut Hr = H.rb_mut().col_segment_mut(block);

            let mut j_next = zero();
            while let Some(j) = n.try_check(*j_next) {
                j_next = n.advance(j, *b);

                ghost_tree!(BLOCK, ROWS, {
                    let (block, _) = n.split(j.into()..j_next, BLOCK);
                    let (rows, _) = n.split(n.idx_inc(*j).., ROWS);

                    let mut Hr = Hr
                        .rb_mut()
                        .col_segment_mut(block)
                        .subrows_mut(zero(), block.len());

                    let zero = block.len().idx(0);
                    for k in block.len().indices() {
                        write1!(Hr[(k, k)] = math(copy(Hr[(zero, k)])));
                    }

                    householder::upgrade_householder_factor(
                        ctx,
                        Hr.rb_mut(),
                        A.col_segment(block).row_segment(rows),
                        *b,
                        1,
                        par,
                    );
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use dyn_stack::GlobalMemBuffer;
    use faer_traits::Unit;

    use super::*;
    use crate::{assert, c64, stats::prelude::*, utils::approx::*, Mat};

    #[test]
    fn test_tridiag_real() {
        let rng = &mut StdRng::seed_from_u64(0);

        for n in [2, 3, 4, 8, 16] {
            with_dim!(n, n);

            let A = CwiseMatDistribution {
                nrows: n,
                ncols: n,
                dist: StandardNormal,
            }
            .rand::<Mat<f64, _, _>>(rng);

            let A = &A + A.adjoint();

            with_dim!(br, 3);
            let mut Hr = Mat::zeros_with(&ctx(), br, n);

            let mut V = A.clone();
            let mut V = V.as_mut();
            tridiag_in_place(
                &ctx(),
                V.rb_mut(),
                Hr.as_mut(),
                Par::Seq,
                DynStack::new(&mut [MaybeUninit::uninit(); 1024]),
                Default::default(),
            );

            let mut A = A.clone();
            let mut A = A.as_mut();

            for iter in 0..2 {
                let mut A = if iter == 0 {
                    A.rb_mut()
                } else {
                    A.rb_mut().transpose_mut()
                };

                ghost_tree!(BLOCK(K0, REST), {
                    let (l![_, rest], _) = n.split(l![n.idx(0), ..], BLOCK);

                    let V = V.rb().row_segment(rest);
                    let V = V.rb().subcols(zero(), rest.len());
                    let mut A = A.rb_mut().row_segment_mut(rest);
                    let Hr = Hr.as_ref().col_segment(rest);

                    householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                        &ctx(),
                        V,
                        Hr.as_ref(),
                        Conj::No,
                        A.rb_mut(),
                        Par::Seq,
                        DynStack::new(&mut GlobalMemBuffer::new(
                            householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<
                                Unit,
                                f64,
                            >(*n - 1, 1, *n)
                            .unwrap(),
                        )),
                    );
                });
            }

            let approx_eq = CwiseMat(ApproxEq::<Unit, f64>::eps());
            for j in n.indices() {
                for i in n.indices() {
                    if *i > *j + 1 || *j > *i + 1 {
                        V[(i, j)] = 0.0;
                    }
                }
            }
            for i in n.indices() {
                if let Some(i1) = n.try_check(*i + 1) {
                    V[(i, i1)] = V[(i1, i)];
                }
            }

            assert!(V ~ A);
        }
    }

    #[test]
    fn test_tridiag_cplx() {
        let rng = &mut StdRng::seed_from_u64(0);

        for n in [2, 3, 4, 8, 16] {
            with_dim!(n, n);
            let A = CwiseMatDistribution {
                nrows: n,
                ncols: n,
                dist: ComplexDistribution::new(StandardNormal, StandardNormal),
            }
            .rand::<Mat<c64, _, _>>(rng);

            let A = &A + A.adjoint();

            with_dim!(br, 1);
            let mut Hr = Mat::zeros_with(&ctx(), br, n);

            let mut V = A.clone();
            let mut V = V.as_mut();
            tridiag_in_place(
                &ctx(),
                V.rb_mut(),
                Hr.as_mut(),
                Par::Seq,
                DynStack::new(&mut [MaybeUninit::uninit(); 1024]),
                Default::default(),
            );

            let mut A = A.clone();
            let mut A = A.as_mut();

            for iter in 0..2 {
                let mut A = if iter == 0 {
                    A.rb_mut()
                } else {
                    A.rb_mut().transpose_mut()
                };

                ghost_tree!(BLOCK(K0, REST), {
                    let (l![_, rest], _) = n.split(l![n.idx(0), ..], BLOCK);

                    let V = V.rb().row_segment(rest);
                    let V = V.rb().subcols(zero(), rest.len());
                    let mut A = A.rb_mut().row_segment_mut(rest);
                    let Hr = Hr.as_ref().col_segment(rest);

                    householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                        &ctx(),
                        V,
                        Hr.as_ref(),
                        if iter == 0{Conj::Yes} else {Conj::No},
                        A.rb_mut(),
                        Par::Seq,
                        DynStack::new(&mut GlobalMemBuffer::new(
                            householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<
                                Unit,
                                c64,
                            >(*n - 1, 1, *n)
                            .unwrap(),
                        )),
                    );
                });
            }

            let approx_eq = CwiseMat(ApproxEq::<Unit, c64>::eps());
            for j in n.indices() {
                for i in n.indices() {
                    if *i > *j + 1 || *j > *i + 1 {
                        V[(i, j)] = c64::ZERO;
                    }
                }
            }
            for i in n.indices() {
                if let Some(i1) = n.try_check(*i + 1) {
                    V[(i, i1)] = V[(i1, i)].conj();
                }
            }

            assert!(V ~ A);
        }
    }
}
