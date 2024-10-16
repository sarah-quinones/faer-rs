use crate::{assert, internal_prelude::*};
use linalg::{
    householder,
    matmul::{dot, matmul},
};

pub fn bidiag_in_place_scratch<C: ComplexContainer, T: ComplexField<C>>(
    nrows: usize,
    ncols: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    StackReq::try_all_of([
        temp_mat_scratch::<C, T>(nrows, 1)?,
        temp_mat_scratch::<C, T>(ncols, 1)?,
    ])
}

/// QR factorization tuning parameters.
#[derive(Copy, Clone)]
#[non_exhaustive]
pub struct BidiagParams {
    /// At which size the parallelism should be disabled.
    pub par_threshold: usize,
}

impl Default for BidiagParams {
    fn default() -> Self {
        Self {
            par_threshold: 192 * 256,
        }
    }
}

#[math]
pub fn bidiag_in_place<'M, 'N, 'BL, 'BR, 'H, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    H_left: MatMut<'_, C, T, Dim<'BL>, Dim<'H>>,
    H_right: MatMut<'_, C, T, Dim<'BR>, Dim<'H>>,
    mut par: Par,
    stack: &mut DynStack,
    params: BidiagParams,
) {
    help!(C);
    help2!(C::Real);

    let m = A.nrows();
    let n = A.ncols();
    let mn = H_left.ncols();
    let bl = H_left.nrows();
    let br = H_right.nrows();
    assert!(*H_left.ncols() == Ord::min(*m, *n));

    let (mut y, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
    let (mut z, _) = unsafe { temp_mat_uninit(ctx, m, 1, stack) };

    let mut y = y.as_mat_mut().col_mut(0).transpose_mut();
    let mut z = z.as_mat_mut().col_mut(0);

    let mut A = A;
    let mut Hl = H_left;
    let mut Hr = H_right;

    {
        let mut Hl = Hl.rb_mut().row_mut(bl.idx(0));
        let mut Hr = Hr.rb_mut().row_mut(br.idx(0));

        for kj in mn.indices() {
            write1!(Hr[kj] = math(infinity()));
        }

        for k in mn.indices() {
            let kj = n.idx(*k);
            let ki = m.idx(*k);

            let mut A = A.rb_mut();
            ghost_tree!(ROWS(TOP, I, BOT), COLS(LEFT, J, RIGHT(J1, NEXT)), {
                let (row_split @ l![top, _, _], (rows_x, ..)) =
                    m.split(l![..ki.into(), ki, ..], ROWS);
                let (col_split @ l![left, _, right], (cols_x, _, l![_, _, RIGHT])) =
                    n.split(l![..kj.into(), kj, ..], COLS);

                let l![A0, A1, A2] = A.rb_mut().row_segments_mut(row_split, rows_x);

                let l![_, _, A02] = A0.col_segments(col_split);
                let l![A10, mut a11, mut A12] = A1.col_segments_mut(col_split, cols_x);
                let l![A20, mut A21, mut A22] = A2.col_segments_mut(col_split, cols_x);

                let l![_, y1, mut y2] = y.rb_mut().col_segments_mut(col_split, cols_x);
                let l![_, z1, mut z2] = z.rb_mut().row_segments_mut(row_split, rows_x);

                if *kj > 0 {
                    let kj1 = left.idx(*kj - 1);
                    let ki1 = top.idx(*ki - 1);

                    let up0 = A10.rb().at(left.local(kj1));
                    let up = A20.rb().col(left.local(kj1));
                    let vp = A02.row(top.local(ki1));

                    write1!(a11, math(a11 - up0 * y1 - z1));
                    z!(A21.rb_mut(), up.rb(), z2.rb())
                        .for_each(|uz!(mut a, u, z)| write1!(a, math(a - u * y1 - z)));
                    z!(A12.rb_mut(), y2.rb(), vp.rb())
                        .for_each(|uz!(mut a, y, v)| write1!(a, math(a - up0 * y - z1 * v)));
                }

                let (tl, beta, _) =
                    householder::make_householder_in_place(ctx, A21.rb_mut(), rb!(a11));
                let tl_inv = math(re.recip(real(tl)));
                write1!(Hl[k] = tl);
                write1!(a11, beta);

                if (*m - *ki.next()) * (*n - *kj.next()) < params.par_threshold {
                    par = Par::Seq;
                }

                if *kj > 0 {
                    let kj1 = left.idx(*kj - 1);
                    let ki1 = top.idx(*ki - 1);

                    let up = A20.rb().col(left.local(kj1));
                    let vp = A02.row(top.local(ki1));

                    match par {
                        Par::Seq => bidiag_fused_op(
                            ctx,
                            A22.rb_mut(),
                            A21.rb(),
                            up.rb(),
                            z2.rb(),
                            y2.rb_mut(),
                            vp.rb(),
                            m.next_power_of_two() - *ki.next(),
                        ),
                        #[cfg(feature = "rayon")]
                        Par::Rayon(nthreads) => {
                            use rayon::prelude::*;
                            let nthreads = nthreads.get();

                            A22.rb_mut()
                                .par_col_partition_mut(nthreads)
                                .zip_eq(y2.rb_mut().par_partition_mut(nthreads))
                                .zip_eq(vp.par_partition(nthreads))
                                .for_each(|((A22, y2), vp)| {
                                    with_dim!(N, A22.ncols());

                                    bidiag_fused_op(
                                        ctx,
                                        A22.as_col_shape_mut(N),
                                        A21.rb(),
                                        up.rb(),
                                        z2.rb(),
                                        y2.as_col_shape_mut(N),
                                        vp.rb().as_col_shape(N),
                                        m.next_power_of_two() - *ki.next(),
                                    );
                                });
                        }
                    }
                } else {
                    matmul(
                        ctx,
                        y2.rb_mut().as_mat_mut(),
                        Accum::Replace,
                        A21.rb().adjoint().as_mat(),
                        A22.rb(),
                        math(one()),
                        par,
                    );
                }

                z!(y2.rb_mut(), A12.rb_mut()).for_each(|uz!(mut y, mut a)| {
                    write1!(y, math(mul_real(y + a, tl_inv)));
                    write1!(a, math(a - y));
                });
                let norm = A12.rb().norm_l2_with(ctx);
                let norm_inv = math.re(recip(norm));
                if !math.re.is_zero(norm) {
                    z!(A12.rb_mut()).for_each(|uz!(mut a)| write1!(a, math(mul_real(a, norm_inv))));
                }
                matmul(
                    ctx,
                    z2.rb_mut().as_mat_mut(),
                    Accum::Replace,
                    A22.rb(),
                    A12.rb().adjoint().as_mat(),
                    math(one()),
                    par,
                );

                if k.next() == mn.end() {
                    break;
                }

                let k1 = mn.idx(*k + 1);
                let kj1 = right.global(right.idx(*kj + 1));

                let (l![j1, next], (rows_x2, ..)) =
                    right.len().split(l![right.local(kj1), ..], RIGHT);
                let l![mut a12_a, mut A12_b] = A12.rb_mut().col_segments_mut(l![j1, next], rows_x2);
                let l![A22_a, _] = A22.rb().col_segments(l![j1, next]);
                let l![y2_a, y2_b] = y2.rb().col_segments(l![j1, next]);

                let (tr, beta, mul) = householder::make_householder_in_place(
                    ctx,
                    A12_b.rb_mut().transpose_mut(),
                    rb!(a12_a),
                );
                let tr_inv = math(re.recip(real(tr)));
                write1!(Hr[k1] = tr);
                write1!(a12_a, math(mul_real(beta, norm)));

                let b = math(
                    y2_a + dot::inner_prod(ctx, y2_b, Conj::No, A12_b.rb().transpose(), Conj::Yes),
                );

                if let Some(mul) = mul {
                    z!(z2.rb_mut(), A21.rb(), A22_a.rb()).for_each(|uz!(mut z, u, a)| {
                        let w = math(z - a * conj(beta));
                        let w = math(w * conj(mul));
                        let w = math(w - u * b);
                        write1!(z, math(mul_real(w, tr_inv)));
                    });
                } else {
                    z!(z2.rb_mut(), A21.rb(), A22_a.rb()).for_each(|uz!(mut z, u, a)| {
                        let w = math(a - u * b);
                        write1!(z, math(mul_real(w, tr_inv)));
                    });
                }
            });
        }
    }

    let mut j_next = zero();
    while let Some(j) = mn.try_check(*j_next) {
        j_next = mn.advance(j, *bl);

        ghost_tree!(BLOCK, COLS, ROWS, {
            let (block, _) = mn.split(j.into()..j_next, BLOCK);
            let (cols, _) = n.split(block, COLS);
            let (rows, _) = m.split(m.idx_inc(*j).., ROWS);

            let mut Hl = Hl
                .rb_mut()
                .col_segment_mut(block)
                .subrows_mut(zero(), block.len());

            let zero = cols.len().idx(0);
            for k in cols.len().indices() {
                write1!(Hl[(k, k)] = math(copy(Hl[(zero, k)])));
            }

            householder::upgrade_householder_factor(
                ctx,
                Hl.rb_mut(),
                A.rb().col_segment(cols).row_segment(rows),
                *bl,
                1,
                par,
            );
        });
    }

    if *n > 0 {
        ghost_tree!(BLOCK, COLS, {
            let (block, _) = mn.split(mn.idx_inc(1).., BLOCK);
            let (cols, _) = n.split(n.idx_inc(1).., COLS);
            let mn = block.len();
            let n = cols.len();
            let A = A.rb().col_segment(cols).subrows(zero(), mn);
            let mut Hr = Hr.rb_mut().col_segment_mut(block);

            let mut j_next = zero();
            while let Some(j) = mn.try_check(*j_next) {
                j_next = mn.advance(j, *br);

                ghost_tree!(BLOCK, ROWS, {
                    let (block, _) = mn.split(j.into()..j_next, BLOCK);
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
                        A.transpose().col_segment(block).row_segment(rows),
                        *bl,
                        1,
                        par,
                    );
                });
            }
        });
    }
}

#[math]
fn bidiag_fused_op<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A22: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,

    u: ColRef<'_, C, T, Dim<'M>>,

    up: ColRef<'_, C, T, Dim<'M>>,
    z: ColRef<'_, C, T, Dim<'M>>,

    y: RowMut<'_, C, T, Dim<'N>>,
    vp: RowRef<'_, C, T, Dim<'N>>,

    align: usize,
) {
    let mut A22 = A22;

    if const { T::SIMD_CAPABILITIES.is_simd() } {
        if let (Some(A22), Some(u), Some(up), Some(z)) = (
            A22.rb_mut().try_as_col_major_mut(),
            u.try_as_col_major(),
            up.try_as_col_major(),
            z.try_as_col_major(),
        ) {
            bidiag_fused_op_simd(ctx, A22, u, up, z, y, vp, align);
        } else {
            bidiag_fused_op_fallback(ctx, A22, u, up, z, y, vp);
        }
    } else {
        bidiag_fused_op_fallback(ctx, A22, u, up, z, y, vp);
    }
}

#[math]
fn bidiag_fused_op_fallback<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A22: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,

    u: ColRef<'_, C, T, Dim<'M>>,

    up: ColRef<'_, C, T, Dim<'M>>,
    z: ColRef<'_, C, T, Dim<'M>>,

    y: RowMut<'_, C, T, Dim<'N>>,
    vp: RowRef<'_, C, T, Dim<'N>>,
) {
    let mut A22 = A22;
    let mut y = y;

    matmul(
        ctx,
        A22.rb_mut(),
        Accum::Add,
        up.as_mat(),
        y.rb().as_mat(),
        math(-one()),
        Par::Seq,
    );
    matmul(
        ctx,
        A22.rb_mut(),
        Accum::Add,
        z.as_mat(),
        vp.as_mat(),
        math(-one()),
        Par::Seq,
    );
    matmul(
        ctx,
        y.rb_mut().as_mat_mut(),
        Accum::Replace,
        u.adjoint().as_mat(),
        A22.rb(),
        math(one()),
        Par::Seq,
    );
}

#[math]
fn bidiag_fused_op_simd<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A22: MatMut<'_, C, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
    u: ColRef<'_, C, T, Dim<'M>, ContiguousFwd>,
    up: ColRef<'_, C, T, Dim<'M>, ContiguousFwd>,
    z: ColRef<'_, C, T, Dim<'M>, ContiguousFwd>,

    y: RowMut<'_, C, T, Dim<'N>>,
    vp: RowRef<'_, C, T, Dim<'N>>,

    align: usize,
) {
    struct Impl<'a, 'M, 'N, C: ComplexContainer, T: ComplexField<C>> {
        ctx: &'a Ctx<C, T>,
        A22: MatMut<'a, C, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
        u: ColRef<'a, C, T, Dim<'M>, ContiguousFwd>,
        up: ColRef<'a, C, T, Dim<'M>, ContiguousFwd>,
        z: ColRef<'a, C, T, Dim<'M>, ContiguousFwd>,

        y: RowMut<'a, C, T, Dim<'N>>,
        vp: RowRef<'a, C, T, Dim<'N>>,

        align: usize,
    }

    impl<'a, 'M, 'N, C: ComplexContainer, T: ComplexField<C>> pulp::WithSimd
        for Impl<'a, 'M, 'N, C, T>
    {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self {
                ctx,
                mut A22,
                u,
                up,
                z,
                mut y,
                vp,
                align,
            } = self;

            let m = A22.nrows();
            let n = A22.ncols();
            let simd = SimdCtx::<C, T, S>::new_align(T::simd_ctx(ctx, simd), m, align);
            let (head, body4, body1, tail) = simd.batch_indices::<4>();
            help!(C);

            for j in n.indices() {
                let mut a = A22.rb_mut().col_mut(j);

                let mut acc0 = simd.zero();
                let mut acc1 = simd.zero();
                let mut acc2 = simd.zero();
                let mut acc3 = simd.zero();

                let yj = simd.splat(as_ref!(math(-y[j])));
                let vj = simd.splat(as_ref!(math(-vp[j])));

                if let Some(i0) = head {
                    let mut a0 = simd.read(a.rb(), i0);
                    a0 = simd.mul_add(simd.read(up, i0), yj, a0);
                    a0 = simd.mul_add(simd.read(z, i0), vj, a0);
                    simd.write(a.rb_mut(), i0, a0);

                    acc0 = simd.conj_mul_add(simd.read(u, i0), a0, acc0);
                }

                for [i0, i1, i2, i3] in body4.clone() {
                    {
                        let mut a0 = simd.read(a.rb(), i0);
                        a0 = simd.mul_add(simd.read(up, i0), yj, a0);
                        a0 = simd.mul_add(simd.read(z, i0), vj, a0);
                        simd.write(a.rb_mut(), i0, a0);

                        acc0 = simd.conj_mul_add(simd.read(u, i0), a0, acc0);
                    }
                    {
                        let mut a1 = simd.read(a.rb(), i1);
                        a1 = simd.mul_add(simd.read(up, i1), yj, a1);
                        a1 = simd.mul_add(simd.read(z, i1), vj, a1);
                        simd.write(a.rb_mut(), i1, a1);

                        acc1 = simd.conj_mul_add(simd.read(u, i1), a1, acc1);
                    }
                    {
                        let mut a2 = simd.read(a.rb(), i2);
                        a2 = simd.mul_add(simd.read(up, i2), yj, a2);
                        a2 = simd.mul_add(simd.read(z, i2), vj, a2);
                        simd.write(a.rb_mut(), i2, a2);

                        acc2 = simd.conj_mul_add(simd.read(u, i2), a2, acc2);
                    }
                    {
                        let mut a3 = simd.read(a.rb(), i3);
                        a3 = simd.mul_add(simd.read(up, i3), yj, a3);
                        a3 = simd.mul_add(simd.read(z, i3), vj, a3);
                        simd.write(a.rb_mut(), i3, a3);

                        acc3 = simd.conj_mul_add(simd.read(u, i3), a3, acc3);
                    }
                }

                for i0 in body1.clone() {
                    let mut a0 = simd.read(a.rb(), i0);
                    a0 = simd.mul_add(simd.read(up, i0), yj, a0);
                    a0 = simd.mul_add(simd.read(z, i0), vj, a0);
                    simd.write(a.rb_mut(), i0, a0);

                    acc0 = simd.conj_mul_add(simd.read(u, i0), a0, acc0);
                }
                if let Some(i0) = tail {
                    let mut a0 = simd.read(a.rb(), i0);
                    a0 = simd.mul_add(simd.read(up, i0), yj, a0);
                    a0 = simd.mul_add(simd.read(z, i0), vj, a0);
                    simd.write(a.rb_mut(), i0, a0);

                    acc0 = simd.conj_mul_add(simd.read(u, i0), a0, acc0);
                }

                acc0 = simd.add(acc0, acc1);
                acc2 = simd.add(acc2, acc3);
                acc0 = simd.add(acc0, acc2);

                write1!(y[j] = simd.reduce_sum(acc0));
            }
        }
    }

    T::Arch::default().dispatch(Impl {
        ctx,
        A22,
        u,
        up,
        z,
        y,
        vp,
        align,
    })
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use dyn_stack::GlobalMemBuffer;
    use faer_traits::Unit;

    use super::*;
    use crate::{assert, c64, stats::prelude::*, utils::approx::*, Mat};

    #[test]
    fn test_bidiag_real() {
        let rng = &mut StdRng::seed_from_u64(0);

        for (m, n) in [(8, 4), (8, 8)] {
            with_dim!(m, m);
            with_dim!(n, n);
            with_dim!(mn, Ord::min(*m, *n));

            let A = CwiseMatDistribution {
                nrows: m,
                ncols: n,
                dist: StandardNormal,
            }
            .rand::<Mat<f64, _, _>>(rng);

            with_dim!(bl, 4);
            with_dim!(br, 3);
            let mut Hl = Mat::zeros_with(&ctx(), bl, mn);
            let mut Hr = Mat::zeros_with(&ctx(), br, mn);

            let mut UV = A.clone();
            let mut UV = UV.as_mut();
            bidiag_in_place(
                &ctx(),
                UV.rb_mut(),
                Hl.as_mut(),
                Hr.as_mut(),
                Par::Seq,
                DynStack::new(&mut [MaybeUninit::uninit(); 1024]),
                Default::default(),
            );

            let mut A = A.clone();
            let mut A = A.as_mut();

            householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                &ctx(),
                UV.rb().subcols(zero(), mn),
                Hl.as_ref(),
                Conj::Yes,
                A.rb_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<
                    Unit,
                    f64,
                >(*n - 1, 1, *m)
                    .unwrap(),
                )),
            );

            ghost_tree!(BLOCK(K0, REST), COLS(J0, RIGHT), {
                let (block_split @ l![_, rest], _) = mn.split(l![mn.idx(0), ..], BLOCK);
                let (col_split, (col_x, ..)) = n.split(l![n.idx(0), ..], COLS);
                let UV = UV.rb().subrows(zero(), rest.len());

                let l![_, mut A1] = A.rb_mut().col_segments_mut(col_split, col_x);
                let l![_, V] = UV.rb().col_segments(col_split);
                let l![_, Hr] = Hr.as_ref().col_segments(block_split);

                householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
                    &ctx(),
                    V.transpose(),
                    Hr.as_ref(),
                    Conj::Yes,
                    A1.rb_mut(),
                    Par::Seq,
                    DynStack::new(&mut GlobalMemBuffer::new(
                        householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<
                            Unit,
                            f64,
                        >(*n - 1, 1, *m)
                        .unwrap(),
                    )),
                );
            });

            let approx_eq = CwiseMat(ApproxEq::<Unit, f64>::eps());
            for j in n.indices() {
                for i in m.indices() {
                    if *i > *j || *j > *i + 1 {
                        UV[(i, j)] = 0.0;
                    }
                }
            }

            assert!(UV ~ A);
        }
    }

    #[test]
    fn test_bidiag_cplx() {
        let rng = &mut StdRng::seed_from_u64(0);

        for (m, n) in [(8, 4), (8, 8)] {
            with_dim!(m, m);
            with_dim!(n, n);
            with_dim!(mn, Ord::min(*m, *n));
            let A = CwiseMatDistribution {
                nrows: m,
                ncols: n,
                dist: ComplexDistribution::new(StandardNormal, StandardNormal),
            }
            .rand::<Mat<c64, _, _>>(rng);

            with_dim!(bl, 4);
            with_dim!(br, 3);
            let mut Hl = Mat::zeros_with(&ctx(), bl, mn);
            let mut Hr = Mat::zeros_with(&ctx(), br, mn);

            let mut UV = A.clone();
            let mut UV = UV.as_mut();
            bidiag_in_place(
                &ctx(),
                UV.rb_mut(),
                Hl.as_mut(),
                Hr.as_mut(),
                Par::Seq,
                DynStack::new(&mut [MaybeUninit::uninit(); 1024]),
                Default::default(),
            );

            let mut A = A.clone();
            let mut A = A.as_mut();

            householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                &ctx(),
                UV.rb().subcols(zero(), mn),
                Hl.as_ref(),
                Conj::Yes,
                A.rb_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<
                    Unit,
                    c64,
                >(*n - 1, 1, *m)
                    .unwrap(),
                )),
            );

            ghost_tree!(BLOCK(K0, REST), COLS(J0, RIGHT), {
                let (block_split @ l![_, rest], _) = mn.split(l![mn.idx(0), ..], BLOCK);
                let (col_split, (col_x, ..)) = n.split(l![n.idx(0), ..], COLS);
                let UV = UV.rb().subrows(zero(), rest.len());

                let l![_, mut A1] = A.rb_mut().col_segments_mut(col_split, col_x);
                let l![_, V] = UV.rb().col_segments(col_split);
                let l![_, Hr] = Hr.as_ref().col_segments(block_split);

                householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
                    &ctx(),
                    V.transpose(),
                    Hr.as_ref(),
                    Conj::Yes,
                    A1.rb_mut(),
                    Par::Seq,
                    DynStack::new(&mut GlobalMemBuffer::new(
                        householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<
                            Unit,
                            c64,
                        >(*n - 1, 1, *m)
                        .unwrap(),
                    )),
                );
            });

            let approx_eq = CwiseMat(ApproxEq::<Unit, c64>::eps());
            for j in n.indices() {
                for i in m.indices() {
                    if *i > *j || *j > *i + 1 {
                        UV[(i, j)] = c64::ZERO;
                    }
                }
            }

            assert!(UV ~ A);
        }
    }
}
