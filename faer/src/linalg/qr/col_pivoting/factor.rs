use crate::{assert, internal_prelude::*, perm::swap_cols_idx, utils::thread::par_split_indices};
use faer_traits::{Real, RealValue};
use linalg::{householder, matmul::dot};
use pulp::Simd;

#[inline(always)]
#[math]
fn update_col_and_norm2_simd<'M, C: ComplexContainer, T: ComplexField<C>, S: Simd>(
    simd: SimdCtx<'M, C, T, S>,
    A: ColMut<'_, C, T, Dim<'M>, ContiguousFwd>,
    lhs: ColRef<'_, C, T, Dim<'M>, ContiguousFwd>,
    rhs: C::Of<T>,
) -> RealValue<C, T> {
    let mut A = A;
    let ctx = &Ctx::<C, T>(T::ctx_from_simd(&simd.0).0);

    let mut sml0 = Real(simd.zero());
    let mut sml1 = Real(simd.zero());
    let mut sml2 = Real(simd.zero());

    let mut med0 = Real(simd.zero());
    let mut med1 = Real(simd.zero());
    let mut med2 = Real(simd.zero());

    let mut big0 = Real(simd.zero());
    let mut big1 = Real(simd.zero());
    let mut big2 = Real(simd.zero());

    let (head, body3, body1, tail) = simd.batch_indices::<3>();
    help!(C);

    let sml = simd.splat_real(math(sqrt_min_positive()));
    let big = simd.splat_real(math(sqrt_max_positive()));

    let rhs = simd.splat(as_ref!(rhs));

    if let Some(i0) = head {
        let mut a0 = simd.read(A.rb(), i0);
        let l0 = simd.read(lhs, i0);
        a0 = simd.mul_add(rhs, l0, a0);
        simd.write(A.rb_mut(), i0, a0);

        sml0 = simd.abs2_add(simd.mul_real(a0, sml), sml0);
        med0 = simd.abs2_add(a0, med0);
        big0 = simd.abs2_add(simd.mul_real(a0, big), big0);
    }

    for [i0, i1, i2] in body3 {
        {
            let mut a0 = simd.read(A.rb(), i0);
            let l0 = simd.read(lhs, i0);
            a0 = simd.mul_add(rhs, l0, a0);
            simd.write(A.rb_mut(), i0, a0);

            sml0 = simd.abs2_add(simd.mul_real(a0, sml), sml0);
            med0 = simd.abs2_add(a0, med0);
            big0 = simd.abs2_add(simd.mul_real(a0, big), big0);
        }
        {
            let mut a1 = simd.read(A.rb(), i1);
            let l1 = simd.read(lhs, i1);
            a1 = simd.mul_add(rhs, l1, a1);
            simd.write(A.rb_mut(), i1, a1);

            sml1 = simd.abs2_add(simd.mul_real(a1, sml), sml1);
            med1 = simd.abs2_add(a1, med1);
            big1 = simd.abs2_add(simd.mul_real(a1, big), big1);
        }
        {
            let mut a2 = simd.read(A.rb(), i2);
            let l2 = simd.read(lhs, i2);
            a2 = simd.mul_add(rhs, l2, a2);
            simd.write(A.rb_mut(), i2, a2);

            sml2 = simd.abs2_add(simd.mul_real(a2, sml), sml2);
            med2 = simd.abs2_add(a2, med2);
            big2 = simd.abs2_add(simd.mul_real(a2, big), big2);
        }
    }
    for i0 in body1 {
        let mut a0 = simd.read(A.rb(), i0);
        let l0 = simd.read(lhs, i0);
        a0 = simd.mul_add(rhs, l0, a0);
        simd.write(A.rb_mut(), i0, a0);

        sml0 = simd.abs2_add(simd.mul_real(a0, sml), sml0);
        med0 = simd.abs2_add(a0, med0);
        big0 = simd.abs2_add(simd.mul_real(a0, big), big0);
    }

    if let Some(i0) = tail {
        let mut a0 = simd.read(A.rb(), i0);
        let l0 = simd.read(lhs, i0);
        a0 = simd.mul_add(rhs, l0, a0);
        simd.write(A.rb_mut(), i0, a0);

        sml0 = simd.abs2_add(simd.mul_real(a0, sml), sml0);
        med0 = simd.abs2_add(a0, med0);
        big0 = simd.abs2_add(simd.mul_real(a0, big), big0);
    }

    sml0.0 = simd.add(sml0.0, sml1.0);
    sml0.0 = simd.add(sml0.0, sml2.0);
    med0.0 = simd.add(med0.0, med1.0);
    med0.0 = simd.add(med0.0, med2.0);
    big0.0 = simd.add(big0.0, big1.0);
    big0.0 = simd.add(big0.0, big2.0);

    let sml0 = math(real(simd.reduce_sum(sml0.0)));
    let med0 = math(real(simd.reduce_sum(med0.0)));
    let big0 = math(real(simd.reduce_sum(big0.0)));

    let sml = math(sqrt_min_positive());
    let big = math(sqrt_max_positive());

    math.re(if sml0 >= one() {
        sml0 * big
    } else if med0 >= one() {
        med0
    } else {
        big0 * sml
    })
}

#[math]
fn update_mat_and_best_norm2_simd<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
    lhs: ColRef<'_, C, T, Dim<'M>, ContiguousFwd>,
    rhs: RowMut<'_, C, T, Dim<'N>>,
    tau_inv: RealValue<C, T>,
    align: usize,
) -> (Idx<'N>, RealValue<C, T>) {
    struct Impl<'a, 'M, 'N, C: ComplexContainer, T: ComplexField<C>> {
        ctx: &'a Ctx<C, T>,
        A: MatMut<'a, C, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
        lhs: ColRef<'a, C, T, Dim<'M>, ContiguousFwd>,
        rhs: RowMut<'a, C, T, Dim<'N>>,
        tau_inv: RealValue<C, T>,
        align: usize,
    }

    impl<'M, 'N, C: ComplexContainer, T: ComplexField<C>> pulp::WithSimd for Impl<'_, 'M, 'N, C, T> {
        type Output = (Idx<'N>, RealValue<C, T>);
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self {
                ctx,
                mut A,
                lhs,
                mut rhs,
                tau_inv,
                align,
            } = self;

            let m = A.nrows();
            let n = A.ncols();

            let simd = SimdCtx::<'_, C, T, S>::new_align(T::simd_ctx(ctx, simd), m, align);
            help!(C);

            let mut best = math.re(zero());
            let mut best_col = n.idx(0);
            for j in n.indices() {
                let dot = math(dot::inner_prod_conj_lhs_simd(simd, lhs, A.rb().col(j)) + rhs[j]);
                let k = math(mul_real(-dot, tau_inv));
                write1!(rhs[j] = math(rhs[j] + k));

                let val = update_col_and_norm2_simd(simd, A.rb_mut().col_mut(j), lhs, k);

                if math(val > best) {
                    best = val;
                    best_col = j;
                }
            }

            (best_col, best)
        }
    }

    T::Arch::default().dispatch(Impl {
        ctx,
        A,
        lhs,
        rhs,
        tau_inv,
        align,
    })
}

#[math]
fn update_mat_and_best_norm2_fallback<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    lhs: ColRef<'_, C, T, Dim<'M>>,
    rhs: RowMut<'_, C, T, Dim<'N>>,
    tau_inv: RealValue<C, T>,
) -> (Idx<'N>, RealValue<C, T>) {
    let mut A = A;
    let mut rhs = rhs;
    help!(C);

    let n = A.ncols();

    let mut best = math.re(zero());
    let mut best_col = n.idx(0);
    for j in n.indices() {
        let dot = math(
            dot::inner_prod(ctx, lhs.transpose(), Conj::Yes, A.rb().col(j), Conj::No) + rhs[j],
        );

        let k = math(mul_real(-dot, tau_inv));
        write1!(rhs[j] = math(rhs[j] + k));
        zipped!(A.rb_mut().col_mut(j), lhs).for_each(|unzipped!(mut dst, src)| {
            write1!(dst, math(dst + k * src));
        });

        let val = A.rb().col(j).norm_l2();
        if math(val > best) {
            best = val;
            best_col = j;
        }
    }
    (best_col, best)
}

#[math]
fn update_mat_and_best_norm2<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    lhs: ColRef<'_, C, T, Dim<'M>>,
    rhs: RowMut<'_, C, T, Dim<'N>>,
    tau_inv: RealValue<C, T>,
    align: usize,
) -> (Idx<'N>, RealValue<C, T>) {
    if const { T::SIMD_CAPABILITIES.is_simd() } {
        let mut A = A;

        if let (Some(A), Some(lhs)) = (A.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major()) {
            update_mat_and_best_norm2_simd(ctx, A, lhs, rhs, tau_inv, align)
        } else {
            update_mat_and_best_norm2_fallback(ctx, A, lhs, rhs, tau_inv)
        }
    } else {
        update_mat_and_best_norm2_fallback(ctx, A, lhs, rhs, tau_inv)
    }
}

/// QR factorization with column pivoting tuning parameters.
#[derive(Copy, Clone)]
#[non_exhaustive]
pub struct ColPivQrParams {
    /// At which size blocking algorithms should be disabled.
    pub blocking_threshold: usize,
    /// At which size the parallelism should be disabled.
    pub par_threshold: usize,
}

impl Default for ColPivQrParams {
    #[inline]
    fn default() -> Self {
        Self {
            blocking_threshold: 48 * 48,
            par_threshold: 192 * 256,
        }
    }
}

#[track_caller]
#[math]
fn qr_in_place_unblocked<'out, 'M, 'N, 'H, I: Index, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    H: RowMut<'_, C, T, Dim<'H>>,
    col_perm: &'out mut Array<'N, I>,
    col_perm_inv: &'out mut Array<'N, I>,
    par: Par,
    stack: &mut DynStack,
    params: ColPivQrParams,
) -> (ColPivQrInfo, PermRef<'out, I, Dim<'N>>) {
    let mut A = A;
    let mut H = H;
    let mut par = par;
    _ = stack;

    let m = A.nrows();
    let n = A.ncols();
    let size = H.ncols();

    assert!(*size == Ord::min(*m, *n));
    for j in n.indices() {
        col_perm[j] = I::truncate(*j);
    }

    let mut n_trans = 0;

    'main: {
        if *size == 0 {
            break 'main;
        }

        help!(C);
        help2!(C::Real);

        let mut best = math.re(zero());
        let mut best_col = n.idx(0);
        for j in n.indices() {
            let val = A.rb().col(j).norm_l2();
            if math(val > best) {
                best = val;
                best_col = j;
            }
        }

        for k in size.indices() {
            let ki = m.idx(*k);
            let kj = n.idx(*k);

            if best_col != kj {
                n_trans += 1;
                col_perm.as_mut().swap(*best_col, *kj);
                swap_cols_idx(ctx, A.rb_mut(), best_col, kj);
            }

            ghost_tree!(ROWS(TOP, BOT), COLS(LEFT, RIGHT), {
                let (rows @ l![top, _], (disjoint_rows, ..)) =
                    m.split(l![..ki.next(), ..], ROWS);
                let (cols @ l![left, right], (disjoint_cols, ..)) =
                    n.split(l![..kj.next(), ..], COLS);

                let ki = top.idx(*ki);
                let kj = left.idx(*kj);

                let l![A0, A1] = A.rb_mut().row_segments_mut(rows, disjoint_rows);
                let l![A00, A01] = A0.col_segments_mut(cols, disjoint_cols);
                let l![A10, mut A11] = A1.col_segments_mut(cols, disjoint_cols);

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

                if k.next() == size.end() {
                    break 'main;
                }

                if (*m - *ki.next()) * (*n - *kj.next()) < params.par_threshold {
                    par = Par::Seq;
                }

                let best_right;
                (best_right, _) = match par {
                    Par::Seq => update_mat_and_best_norm2(
                        ctx,
                        A11.rb_mut(),
                        A10.rb(),
                        A01.rb_mut(),
                        tau_inv,
                        m.next_power_of_two() - *ki.next(),
                    ),
                    Par::Rayon(nthreads) => {
                        use rayon::prelude::*;
                        let nthreads = nthreads.get();

                        let mut best = core::iter::repeat_with(|| (0, send2!(math.re(zero()))))
                            .take(nthreads)
                            .collect::<Vec<_>>();
                        let full_cols = *A11.ncols();

                        let tau_inv = sync2!(as_ref2!(tau_inv));

                        best.par_iter_mut()
                            .zip_eq(A11.rb_mut().par_col_partition_mut(nthreads))
                            .zip_eq(A01.rb_mut().transpose_mut().par_partition_mut(nthreads))
                            .enumerate()
                            .for_each(|(idx, (((max_col, max_score), A11), A01))| {
                                with_dim!(N, A11.ncols());

                                let (col, score) = update_mat_and_best_norm2(
                                    ctx,
                                    A11.as_col_shape_mut(N),
                                    A10.rb(),
                                    A01.transpose_mut().as_col_shape_mut(N),
                                    math.re(copy(unsync2!(tau_inv))),
                                    m.next_power_of_two() - *ki.next(),
                                );

                                *max_col = *col + par_split_indices(full_cols, idx, nthreads).0;
                                *max_score = send2!(score);
                            });

                        let mut best_col = right.len().idx(0);
                        let mut best_val = math.re.zero();

                        for (col, val) in best {
                            let val = unsend2!(val);
                            if math(val > best_val) {
                                best_col = right.len().idx(col);
                                best_val = val;
                            }
                        }

                        (best_col, best_val)
                    }
                };
                best_col = right.global(best_right).local();
            });
        }
    }

    for j in n.indices() {
        col_perm_inv[n.idx(col_perm[j].zx())] = I::truncate(*j);
    }

    (
        ColPivQrInfo {
            transposition_count: n_trans,
        },
        unsafe {
            PermRef::new_unchecked(
                Idx::from_slice_ref_unchecked(col_perm.as_ref()),
                Idx::from_slice_ref_unchecked(col_perm_inv.as_ref()),
                n,
            )
        },
    )
}

/// Computes the size and alignment of required workspace for performing a QR decomposition
/// with column pivoting.
pub fn qr_in_place_scratch<I: Index, C: ComplexContainer, T: ComplexField<C>>(
    nrows: usize,
    ncols: usize,
    blocksize: usize,
    parallelism: Par,
    params: ColPivQrParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = nrows;
    let _ = ncols;
    let _ = parallelism;
    let _ = blocksize;
    let _ = &params;
    Ok(StackReq::default())
}

/// Information about the resulting QR factorization.
#[derive(Copy, Clone, Debug)]
pub struct ColPivQrInfo {
    /// Number of transpositions that were performed, can be used to compute the determinant of
    /// $P$.
    pub transposition_count: usize,
}

#[track_caller]
#[math]
pub fn qr_in_place<'out, 'M, 'N, 'B, 'H, I: Index, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    H: MatMut<'_, C, T, Dim<'B>, Dim<'H>>,
    col_perm: &'out mut Array<'N, I>,
    col_perm_inv: &'out mut Array<'N, I>,
    par: Par,
    stack: &mut DynStack,
    params: ColPivQrParams,
) -> (ColPivQrInfo, PermRef<'out, I, Dim<'N>>) {
    let mut A = A;
    let mut H = H;
    let size = H.ncols();
    let blocksize = H.nrows();

    let ret = qr_in_place_unblocked(
        ctx,
        A.rb_mut(),
        H.rb_mut().row_mut(blocksize.idx(0)),
        col_perm,
        col_perm_inv,
        par,
        stack,
        params,
    );

    let mut j_next = zero();
    while let Some(j) = size.try_check(*j_next) {
        j_next = size.advance(j, *blocksize);
        let ji = A.nrows().idx_inc(*j);
        let jj = A.ncols().idx_inc(*j);

        with_dim!(blocksize, *j_next - *j);

        let mut H = H
            .rb_mut()
            .subcols_mut(j.to_incl(), blocksize)
            .subrows_mut(zero(), blocksize);

        help!(C);
        let i = blocksize.idx(0);
        for j in blocksize.indices() {
            write1!(H[(j, j)] = math(copy(H[(i, j)])));
        }

        let A = A
            .rb()
            .subcols(jj, blocksize)
            .subrows_range((ji, A.nrows().end()));

        householder::upgrade_householder_factor(
            ctx,
            H.rb_mut(),
            A.bind_r(unique!()),
            *blocksize,
            1,
            par,
        );
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, c64, stats::prelude::*, utils::approx::*, Mat};
    use dyn_stack::GlobalMemBuffer;
    use faer_traits::Unit;

    #[test]
    fn test_unblocked_qr() {
        let rng = &mut StdRng::seed_from_u64(0);

        for par in [Par::Seq, Par::rayon(8)] {
            for n in [2, 3, 4, 8, 16, 24, 32, 128, 255] {
                with_dim!(N, n);
                with_dim!(B, 15);

                let approx_eq = CwiseMat(ApproxEq {
                    ctx: ctx::<Ctx<Unit, c64>>(),
                    abs_tol: 1e-10,
                    rel_tol: 1e-10,
                });

                let A = CwiseMatDistribution {
                    nrows: N,
                    ncols: N,
                    dist: ComplexDistribution::new(StandardNormal, StandardNormal),
                }
                .rand::<Mat<c64, Dim, Dim>>(rng);
                let A = A.as_ref();
                let mut QR = A.cloned();
                let mut H = Mat::zeros_with(&ctx(), B, N);

                let col_perm = &mut *vec![0usize; *N];
                let col_perm_inv = &mut *vec![0usize; *N];
                let col_perm = Array::from_mut(col_perm, N);
                let col_perm_inv = Array::from_mut(col_perm_inv, N);

                let q = qr_in_place(
                    &ctx(),
                    QR.as_mut(),
                    H.as_mut(),
                    col_perm,
                    col_perm_inv,
                    par,
                    DynStack::new(&mut GlobalMemBuffer::new(
                        qr_in_place_scratch::<usize, Unit, c64>(
                            *N,
                            *N,
                            *B,
                            par,
                            Default::default(),
                        )
                        .unwrap(),
                    )),
                    Default::default(),
                )
                .1;

                let mut Q = Mat::<c64, _, _>::zeros_with(&ctx(), N, N);
                let mut R = QR.as_ref().cloned();

                for j in N.indices() {
                    Q[(j, j)] = c64::ONE;
                }

                householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                    &ctx(),
                    QR.as_ref(),
                    H.as_ref(),
                    Conj::No,
                    Q.as_mut(),
                    Par::Seq,
                    DynStack::new(
                        &mut GlobalMemBuffer::new(
                            householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<Unit, c64>(
                                *N,
                                *B,
                                *N,
                            ).unwrap()
                        )
                    )
                );

                for j in N.indices() {
                    for i in j.next().to(N.end()) {
                        R[(i, j)] = c64::ZERO;
                    }
                }

                assert!(Q * R * q ~ A);
            }

            with_dim!(N, 20);
            for m in [2, 3, 4, 8, 16, 24, 32, 128, 255] {
                with_dim!(M, m);
                with_dim!(B, 15);
                with_dim!(H, Ord::min(*M, *N));

                let approx_eq = CwiseMat(ApproxEq {
                    ctx: ctx::<Ctx<Unit, c64>>(),
                    abs_tol: 1e-10,
                    rel_tol: 1e-10,
                });

                let A = CwiseMatDistribution {
                    nrows: M,
                    ncols: N,
                    dist: ComplexDistribution::new(StandardNormal, StandardNormal),
                }
                .rand::<Mat<c64, Dim, Dim>>(rng);
                let A = A.as_ref();
                let mut QR = A.cloned();
                let mut H = Mat::zeros_with(&ctx(), B, H);

                let col_perm = &mut *vec![0usize; *N];
                let col_perm_inv = &mut *vec![0usize; *N];
                let col_perm = Array::from_mut(col_perm, N);
                let col_perm_inv = Array::from_mut(col_perm_inv, N);

                let q = qr_in_place(
                    &ctx(),
                    QR.as_mut(),
                    H.as_mut(),
                    col_perm,
                    col_perm_inv,
                    par,
                    DynStack::new(&mut GlobalMemBuffer::new(
                        qr_in_place_scratch::<usize, Unit, c64>(
                            *M,
                            *N,
                            *B,
                            par,
                            Default::default(),
                        )
                        .unwrap(),
                    )),
                    Default::default(),
                )
                .1;

                let mut Q = Mat::<c64, _, _>::zeros_with(&ctx(), M, M);
                let mut R = QR.as_ref().cloned();

                for j in M.indices() {
                    Q[(j, j)] = c64::ONE;
                }

                householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                    &ctx(),
                    QR.as_ref(),
                    H.as_ref(),
                    Conj::No,
                    Q.as_mut(),
                    Par::Seq,
                    DynStack::new(
                        &mut GlobalMemBuffer::new(
                            householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<Unit, c64>(
                                *M,
                                *B,
                                *M,
                            ).unwrap()
                        )
                    )
                );

                for j in N.indices() {
                    for i in M.indices().skip(*j + 1) {
                        R[(i, j)] = c64::ZERO;
                    }
                }

                assert!(Q * R * q ~ A);
            }
        }
    }
}
