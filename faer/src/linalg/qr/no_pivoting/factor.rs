use crate::{assert, internal_prelude::*};
use linalg::householder;

#[math]
fn qr_in_place_unblocked<'M, 'N, 'H, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    H: RowMut<'_, C, T, Dim<'H>>,
) {
    let mut A = A;
    let mut H = H;

    help!(C);
    help2!(C::Real);

    let M = A.nrows();
    let N = A.ncols();
    let size = H.ncols();

    for k in size.indices() {
        let ki = M.check(*k);
        let kj = N.check(*k);

        ghost_tree!(ROWS(TOP, BOT), COLS(LEFT, RIGHT), {
            let (rows, ROWS) = M.full(ROWS);
            let (cols, COLS) = N.full(COLS);

            let (disjoint_rows, top, bot, _, _) = rows.split_inc(ki.next(), ROWS.TOP, ROWS.BOT);
            let (disjoint_cols, left, right, _, _) =
                cols.split_inc(kj.next(), COLS.LEFT, COLS.RIGHT);

            let ki = top.idx(*ki);
            let kj = left.idx(*kj);

            let (A0, A1) = A.rb_mut().row_segments_mut(top, bot, disjoint_rows);
            let (A00, A01) = A0.col_segments_mut(left, right, disjoint_cols);
            let (A10, mut A11) = A1.col_segments_mut(left, right, disjoint_cols);

            let mut A00 = A00.at_mut(top.local(ki), left.local(kj));
            let mut A01 = A01.row_mut(top.local(ki));
            let mut A10 = A10.col_mut(left.local(kj));

            let norm = A10.rb().norm_l2_with(ctx);

            let (tau, beta) = householder::make_householder_in_place(
                ctx,
                Some(A10.rb_mut()),
                rb!(A00),
                as_ref2!(norm),
            );

            let tau_inv = math.re(recip(cx.real(tau)));
            write1!(H[k] = tau);
            write1!(A00, beta);

            for j in right {
                let mut head = A01.rb_mut().at_mut(right.local(j));
                let tail = A11.rb_mut().col_mut(right.local(j));

                let dot = math(
                    head + linalg::matmul::dot::inner_prod(
                        ctx,
                        A10.rb().transpose(),
                        Conj::Yes,
                        tail.rb(),
                        Conj::No,
                    ),
                );
                let k = math(-mul_real(dot, tau_inv));
                write1!(head, math(head + k));
                zipped!(tail, A10.rb()).for_each(|unzipped!(mut dst, src)| {
                    write1!(dst, math(dst + k * src));
                });
            }
        });
    }
}

/// The recommended block size to use for a QR decomposition of a matrix with the given shape.
#[inline]
pub fn recommended_blocksize<C: ComplexContainer, T: ComplexField<C>>(
    nrows: usize,
    ncols: usize,
) -> usize {
    let prod = nrows * ncols;
    let size = nrows.min(ncols);

    (if prod > 8192 * 8192 {
        256
    } else if prod > 2048 * 2048 {
        128
    } else if prod > 1024 * 1024 {
        64
    } else if prod > 512 * 512 {
        48
    } else if prod > 128 * 128 {
        32
    } else if prod > 32 * 32 {
        8
    } else if prod > 16 * 16 {
        4
    } else {
        1
    })
    .min(size)
    .max(1)
}

/// QR factorization tuning parameters.
#[derive(Copy, Clone)]
#[non_exhaustive]
pub struct QrParams {
    /// At which size blocking algorithms should be disabled.
    pub blocking_threshold: usize,
    /// At which size the parallelism should be disabled.
    pub par_threshold: usize,
}

impl Default for QrParams {
    #[inline]
    fn default() -> Self {
        Self {
            blocking_threshold: 48 * 48,
            par_threshold: 192 * 256,
        }
    }
}

#[math]
fn qr_in_place_blocked<'M, 'N, 'B, 'H, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    H: MatMut<'_, C, T, Dim<'B>, Dim<'H>>,
    par: Parallelism,
    stack: &mut DynStack,
    params: QrParams,
) {
    let m = A.nrows();
    let n = A.ncols();
    let size = H.ncols();
    let blocksize = H.nrows();

    assert!(*blocksize > 0);

    if *blocksize == 1 {
        return qr_in_place_unblocked(ctx, A, H.row_mut(blocksize.idx(0)));
    }
    let sub_blocksize = if *m * *n < params.blocking_threshold {
        blocksize.idx(1)
    } else {
        blocksize.idx(*blocksize / 2)
    };

    let mut A = A;
    let mut H = H;

    let mut j_next = zero();
    while let Some(j) = size.try_check(*j_next) {
        j_next = size.advance(j, *blocksize);

        guards!(REM_ROWS);
        let mut A = A
            .rb_mut()
            .subrows_range_mut((m.idx(*j), m.end()))
            .bind_r(REM_ROWS);

        ghost_tree!(COLS(HEAD(BLOCK), TAIL), {
            let (cols, COLS) = size.full(COLS);
            let (disjoint_cols, head, tail, COLS, _) = cols.split_inc(j_next, COLS.HEAD, COLS.TAIL);

            let (block, _) = head.segment(
                head.idx_inc(*j.to_incl()),
                head.idx_inc(*j_next),
                COLS.BLOCK,
            );
            qr_in_place_blocked(
                ctx,
                A.rb_mut()
                    .subcols_range_mut((n.idx_inc(*block.start()), n.idx_inc(*block.end())))
                    .bind_c(unique!()),
                H.rb_mut()
                    .subrows_mut(zero(), *sub_blocksize)
                    .col_segment_mut(block)
                    .bind_r(unique!()),
                par,
                stack,
                params,
            );

            let blocksize = block.len();

            let mut H = H
                .rb_mut()
                .submatrix_mut(zero(), j.to_incl(), blocksize, blocksize);

            let mut k_next = zero();
            while let Some(k) = blocksize.try_check(*k_next) {
                k_next = blocksize.advance(k, *sub_blocksize);

                if *k == 0 {
                    continue;
                }

                ghost_tree!(ROWS(TOP, BOT), BLOCK(SUBCOLS), {
                    let (rows, ROWS) = blocksize.full(ROWS);
                    let (block, BLOCK) = blocksize.full(BLOCK);

                    let (subcols, _) = block.segment(k.to_incl(), k_next, BLOCK.SUBCOLS);
                    let (disjoint, top, bot, _, _) =
                        rows.split_inc(k.to_incl(), ROWS.TOP, ROWS.BOT);

                    let mut H = H.rb_mut().col_segment_mut(subcols);

                    let (H0, mut H1) = H.rb_mut().row_segments_mut(top, bot, disjoint);
                    let H0 = H0.rb().subrows(zero(), subcols.len());
                    let H1 = H1.rb_mut().subrows_mut(zero(), subcols.len());

                    H1.transpose_mut()
                        .copy_from_triangular_lower_with_ctx(ctx, H0.transpose());
                });
            }
            let (A0, A2) = A.rb_mut().split_at_col_mut(n.idx_inc(*size));
            guards!(REM);
            let A2 = A2.bind_c(REM);

            let A0 = A0.subcols_mut(0, size);
            let (A0, A1) = A0.col_segments_mut(block, tail, disjoint_cols);
            let A0 = A0.rb();

            householder::upgrade_householder_factor(
                ctx,
                H.rb_mut(),
                A0,
                *blocksize,
                *sub_blocksize,
                par,
            );
            if *A1.ncols() > 0 {
                householder::apply_block_householder_transpose_on_the_left_in_place_with_conj(
                    ctx,
                    A0,
                    H.rb(),
                    Conj::Yes,
                    A1,
                    par,
                    stack,
                )
            };
            if *A2.ncols() > 0 {
                householder::apply_block_householder_transpose_on_the_left_in_place_with_conj(
                    ctx,
                    A0,
                    H.rb(),
                    Conj::Yes,
                    A2,
                    par,
                    stack,
                );
            }
        });
    }
}

#[track_caller]
#[math]
pub fn qr_in_place<'M, 'N, 'B, 'H, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    H: MatMut<'_, C, T, Dim<'B>, Dim<'H>>,
    par: Parallelism,
    stack: &mut DynStack,
    params: QrParams,
) {
    let blocksize = H.nrows();
    assert!(all(
        *blocksize > 0,
        *H.ncols() == Ord::min(*A.nrows(), *A.ncols()),
    ));

    #[cfg(feature = "perf-warn")]
    if A.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(QR_WARN) {
        if A.col_stride().unsigned_abs() == 1 {
            log::warn!(target: "faer_perf", "QR prefers column-major matrix. Found row-major matrix.");
        } else {
            log::warn!(target: "faer_perf", "QR prefers column-major matrix. Found matrix with generic strides.");
        }
    }

    qr_in_place_blocked(ctx, A, H, par, stack, params);
}

/// Computes the size and alignment of required workspace for performing a QR
/// decomposition with no pivoting.
#[inline]
pub fn qr_in_place_scratch<C: ComplexContainer, T: ComplexField<C>>(
    nrows: usize,
    ncols: usize,
    blocksize: usize,
    par: Parallelism,
    params: QrParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = par;
    let _ = nrows;
    let _ = &params;
    temp_mat_scratch::<C, T>(blocksize, ncols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, c64, stats::prelude::*, utils::approx::*, Mat, Row};
    use dyn_stack::GlobalMemBuffer;
    use faer_traits::Unit;

    #[test]
    fn test_unblocked_qr() {
        let rng = &mut StdRng::seed_from_u64(0);

        for par in [Parallelism::None, Parallelism::rayon(8)] {
            for n in [2, 4, 8, 16, 24, 32, 128, 255, 256, 257] {
                with_dim!(N, n);
                let approx_eq = CwiseMat(ApproxEq {
                    ctx: ctx::<Ctx<Unit, f64>>(),
                    abs_tol: 1e-10,
                    rel_tol: 1e-10,
                });

                let A = CwiseMatDistribution {
                    nrows: N,
                    ncols: N,
                    dist: StandardNormal,
                }
                .rand::<Mat<f64, Dim, Dim>>(rng);
                let A = A.as_ref();

                let mut QR = A.cloned();
                let mut H = Row::zeros_with_ctx(&ctx(), N);

                qr_in_place_unblocked(&ctx(), QR.as_mut(), H.as_mut());

                let mut Q = Mat::<f64, _, _>::zeros_with_ctx(&ctx(), N, N);
                let mut R = QR.as_ref().cloned();

                for j in N.indices() {
                    Q[(j, j)] = 1.0;
                }

                householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                    &ctx(),
                    QR.as_ref(),
                    H.as_mat(),
                    Conj::No,
                    Q.as_mut(),
                    Parallelism::None,
                    DynStack::new(
                        &mut GlobalMemBuffer::new(
                            householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<Unit, f64>(
                                *N,
                                1,
                                *N,
                            ).unwrap()
                        )
                    )
                );

                for j in N.indices() {
                    for i in j.next().to(N.end()) {
                        R[(i, j)] = 0.0;
                    }
                }

                assert!(Q * R ~ A);
            }

            for n in [2, 3, 4, 8, 16, 24, 32, 128, 255, 256, 257] {
                with_dim!(N, n);
                with_dim!(B, 2);

                let approx_eq = CwiseMat(ApproxEq {
                    ctx: ctx::<Ctx<Unit, f64>>(),
                    abs_tol: 1e-10,
                    rel_tol: 1e-10,
                });

                let A = CwiseMatDistribution {
                    nrows: N,
                    ncols: N,
                    dist: StandardNormal,
                }
                .rand::<Mat<f64, Dim, Dim>>(rng);
                let A = A.as_ref();
                let mut QR = A.cloned();
                let mut H = Mat::zeros_with_ctx(&ctx(), B, N);

                qr_in_place_blocked(
                    &ctx(),
                    QR.as_mut(),
                    H.as_mut(),
                    par,
                    DynStack::new(&mut GlobalMemBuffer::new(
                        qr_in_place_scratch::<Unit, f64>(*N, *N, *B, par, Default::default())
                            .unwrap(),
                    )),
                    QrParams {
                        blocking_threshold: 2,
                        ..Default::default()
                    },
                );

                let mut Q = Mat::<f64, _, _>::zeros_with_ctx(&ctx(), N, N);
                let mut R = QR.as_ref().cloned();

                for j in N.indices() {
                    Q[(j, j)] = 1.0;
                }

                householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                    &ctx(),
                    QR.as_ref(),
                    H.as_ref(),
                    Conj::No,
                    Q.as_mut(),
                    Parallelism::None,
                    DynStack::new(
                        &mut GlobalMemBuffer::new(
                            householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<Unit, f64>(
                                *N,
                                *B,
                                *N,
                            ).unwrap()
                        )
                    )
                );

                for j in N.indices() {
                    for i in j.next().to(N.end()) {
                        R[(i, j)] = 0.0;
                    }
                }

                assert!(Q * R ~ A);
            }
        }
    }
}
