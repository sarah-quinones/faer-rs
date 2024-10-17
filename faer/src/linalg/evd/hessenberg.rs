use linalg::{
    householder,
    matmul::{self, dot, matmul, triangular::BlockStructure},
    triangular_solve,
};

use crate::internal_prelude::*;

/// QR factorization tuning parameters.
#[derive(Copy, Clone)]
#[non_exhaustive]
pub struct HessenbergParams {
    /// At which size the parallelism should be disabled.
    pub par_threshold: usize,
    /// At which size the parallelism should be disabled.
    pub blocking_threshold: usize,
}

impl Default for HessenbergParams {
    fn default() -> Self {
        Self {
            par_threshold: 192 * 256,
            blocking_threshold: 256 * 256,
        }
    }
}

pub fn hessenberg_in_place_scratch<C: ComplexContainer, T: ComplexField<C>>(
    dim: usize,
    blocksize: usize,
    par: Par,
    params: HessenbergParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = par;
    let n = dim;
    if n * n < params.blocking_threshold {
        StackReq::try_any_of([
            StackReq::try_all_of([
                temp_mat_scratch::<C, T>(n, 1)?.try_array(3)?,
                temp_mat_scratch::<C, T>(n, par.degree())?.try_array(2)?,
            ])?,
            householder::apply_block_householder_on_the_right_in_place_scratch::<C, T>(
                n, blocksize, n,
            )?,
        ])
    } else {
        StackReq::try_all_of([
            temp_mat_scratch::<C, T>(n, blocksize)?,
            StackReq::try_any_of([
                StackReq::try_all_of([
                    temp_mat_scratch::<C, T>(n, 1)?,
                    temp_mat_scratch::<C, T>(n, par.degree())?,
                ])?,
                temp_mat_scratch::<C, T>(n, blocksize)?,
            ])?,
        ])
    }
}

#[math]
fn hessenberg_fused_op_simd<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>, ContiguousFwd>,

    l_out: RowMut<'_, C, T, Dim<'N>>,
    r_out: ColMut<'_, C, T, Dim<'M>, ContiguousFwd>,
    l_in: RowRef<'_, C, T, Dim<'M>, ContiguousFwd>,
    r_in: ColRef<'_, C, T, Dim<'N>>,

    l0: ColRef<'_, C, T, Dim<'M>, ContiguousFwd>,
    l1: ColRef<'_, C, T, Dim<'M>, ContiguousFwd>,
    r0: RowRef<'_, C, T, Dim<'N>>,
    r1: RowRef<'_, C, T, Dim<'N>>,
    align: usize,
) {
    struct Impl<'a, 'M, 'N, C: ComplexContainer, T: ComplexField<C>> {
        ctx: &'a Ctx<C, T>,
        A: MatMut<'a, C, T, Dim<'M>, Dim<'N>, ContiguousFwd>,

        l_out: RowMut<'a, C, T, Dim<'N>>,
        r_out: ColMut<'a, C, T, Dim<'M>, ContiguousFwd>,
        l_in: RowRef<'a, C, T, Dim<'M>, ContiguousFwd>,
        r_in: ColRef<'a, C, T, Dim<'N>>,

        l0: ColRef<'a, C, T, Dim<'M>, ContiguousFwd>,
        l1: ColRef<'a, C, T, Dim<'M>, ContiguousFwd>,
        r0: RowRef<'a, C, T, Dim<'N>>,
        r1: RowRef<'a, C, T, Dim<'N>>,
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
                mut A,
                mut l_out,
                mut r_out,
                l_in,
                r_in,
                l0,
                l1,
                r0,
                r1,
                align,
            } = self;

            let (m, n) = A.shape();

            let simd = SimdCtx::<C, T, S>::new_align(T::simd_ctx(ctx, simd), m, align);

            {
                let (head, body, tail) = simd.indices();
                if let Some(i) = head {
                    simd.write(r_out.rb_mut(), i, simd.zero());
                }
                for i in body {
                    simd.write(r_out.rb_mut(), i, simd.zero());
                }
                if let Some(i) = tail {
                    simd.write(r_out.rb_mut(), i, simd.zero());
                }
            }

            let (head, body4, body1, tail) = simd.batch_indices::<4>();

            let l_in = l_in.transpose();

            help!(C);
            for j in n.indices() {
                let mut A = A.rb_mut().col_mut(j);
                let r_in = simd.splat(r_in.at(j));
                let r0 = simd.splat(as_ref!(math(-r0[j])));
                let r1 = simd.splat(as_ref!(math(-r1[j])));

                let mut acc0 = simd.zero();
                let mut acc1 = simd.zero();
                let mut acc2 = simd.zero();
                let mut acc3 = simd.zero();

                if let Some(i0) = head {
                    let mut a0 = simd.read(A.rb(), i0);
                    a0 = simd.mul_add(simd.read(l0, i0), r0, a0);
                    a0 = simd.conj_mul_add(r1, simd.read(l1, i0), a0);
                    simd.write(A.rb_mut(), i0, a0);
                    acc0 = simd.conj_mul_add(simd.read(l_in, i0), a0, acc0);
                    let tmp = simd.read(r_out.rb(), i0);
                    simd.write(r_out.rb_mut(), i0, simd.mul_add(a0, r_in, tmp));
                }
                for [i0, i1, i2, i3] in body4.clone() {
                    {
                        let mut a0 = simd.read(A.rb(), i0);
                        a0 = simd.mul_add(simd.read(l0, i0), r0, a0);
                        a0 = simd.conj_mul_add(r1, simd.read(l1, i0), a0);
                        simd.write(A.rb_mut(), i0, a0);
                        acc0 = simd.conj_mul_add(simd.read(l_in, i0), a0, acc0);
                        let tmp = simd.read(r_out.rb(), i0);
                        simd.write(r_out.rb_mut(), i0, simd.mul_add(a0, r_in, tmp));
                    }
                    {
                        let mut a1 = simd.read(A.rb(), i1);
                        a1 = simd.mul_add(simd.read(l0, i1), r0, a1);
                        a1 = simd.conj_mul_add(r1, simd.read(l1, i1), a1);
                        simd.write(A.rb_mut(), i1, a1);
                        acc1 = simd.conj_mul_add(simd.read(l_in, i1), a1, acc1);
                        let tmp = simd.read(r_out.rb(), i1);
                        simd.write(r_out.rb_mut(), i1, simd.mul_add(a1, r_in, tmp));
                    }
                    {
                        let mut a2 = simd.read(A.rb(), i2);
                        a2 = simd.mul_add(simd.read(l0, i2), r0, a2);
                        a2 = simd.conj_mul_add(r1, simd.read(l1, i2), a2);
                        simd.write(A.rb_mut(), i2, a2);
                        acc2 = simd.conj_mul_add(simd.read(l_in, i2), a2, acc2);
                        let tmp = simd.read(r_out.rb(), i2);
                        simd.write(r_out.rb_mut(), i2, simd.mul_add(a2, r_in, tmp));
                    }
                    {
                        let mut a3 = simd.read(A.rb(), i3);
                        a3 = simd.mul_add(simd.read(l0, i3), r0, a3);
                        a3 = simd.conj_mul_add(r1, simd.read(l1, i3), a3);
                        simd.write(A.rb_mut(), i3, a3);
                        acc3 = simd.conj_mul_add(simd.read(l_in, i3), a3, acc3);
                        let tmp = simd.read(r_out.rb(), i3);
                        simd.write(r_out.rb_mut(), i3, simd.mul_add(a3, r_in, tmp));
                    }
                }
                for i0 in body1.clone() {
                    let mut a0 = simd.read(A.rb(), i0);
                    a0 = simd.mul_add(simd.read(l0, i0), r0, a0);
                    a0 = simd.conj_mul_add(r1, simd.read(l1, i0), a0);
                    simd.write(A.rb_mut(), i0, a0);
                    acc0 = simd.conj_mul_add(simd.read(l_in, i0), a0, acc0);
                    let tmp = simd.read(r_out.rb(), i0);
                    simd.write(r_out.rb_mut(), i0, simd.mul_add(a0, r_in, tmp));
                }
                if let Some(i0) = tail {
                    let mut a0 = simd.read(A.rb(), i0);
                    a0 = simd.mul_add(simd.read(l0, i0), r0, a0);
                    a0 = simd.conj_mul_add(r1, simd.read(l1, i0), a0);
                    simd.write(A.rb_mut(), i0, a0);
                    acc0 = simd.conj_mul_add(simd.read(l_in, i0), a0, acc0);
                    let tmp = simd.read(r_out.rb(), i0);
                    simd.write(r_out.rb_mut(), i0, simd.mul_add(a0, r_in, tmp));
                }

                acc0 = simd.add(acc0, acc1);
                acc2 = simd.add(acc2, acc3);
                acc0 = simd.add(acc0, acc2);

                let mut l_out = l_out.rb_mut().at_mut(j);
                write1!(l_out, simd.reduce_sum(acc0));
            }
        }
    }

    T::Arch::default().dispatch(Impl {
        ctx,
        A,
        l_out,
        r_out,
        l_in,
        r_in,
        l0,
        l1,
        r0,
        r1,
        align,
    })
}

#[math]
fn hessenberg_fused_op_fallback<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,

    l_out: RowMut<'_, C, T, Dim<'N>>,
    r_out: ColMut<'_, C, T, Dim<'M>>,
    l_in: RowRef<'_, C, T, Dim<'M>>,
    r_in: ColRef<'_, C, T, Dim<'N>>,

    l0: ColRef<'_, C, T, Dim<'M>>,
    l1: ColRef<'_, C, T, Dim<'M>>,
    r0: RowRef<'_, C, T, Dim<'N>>,
    r1: RowRef<'_, C, T, Dim<'N>>,
) {
    let mut A = A;

    matmul(
        ctx,
        A.rb_mut(),
        Accum::Add,
        l0.as_mat(),
        r0.as_mat(),
        math(-one()),
        Par::Seq,
    );
    matmul(
        ctx,
        A.rb_mut(),
        Accum::Add,
        l1.as_mat(),
        r1.as_mat().conjugate(),
        math(-one()),
        Par::Seq,
    );

    matmul(
        ctx,
        r_out.as_mat_mut(),
        Accum::Replace,
        A.rb(),
        r_in.as_mat(),
        math(one()),
        Par::Seq,
    );
    matmul(
        ctx,
        l_out.as_mat_mut(),
        Accum::Replace,
        l_in.as_mat().conjugate(),
        A.rb(),
        math(one()),
        Par::Seq,
    );
}

fn hessenberg_fused_op<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,

    l_out: RowMut<'_, C, T, Dim<'N>>,
    r_out: ColMut<'_, C, T, Dim<'M>>,
    l_in: RowRef<'_, C, T, Dim<'M>>,
    r_in: ColRef<'_, C, T, Dim<'N>>,

    l0: ColRef<'_, C, T, Dim<'M>>,
    l1: ColRef<'_, C, T, Dim<'M>>,
    r0: RowRef<'_, C, T, Dim<'N>>,
    r1: RowRef<'_, C, T, Dim<'N>>,
    align: usize,
) {
    let mut A = A;
    let mut r_out = r_out;

    if const { T::SIMD_CAPABILITIES.is_simd() } {
        if let (Some(A), Some(r_out), Some(l_in), Some(l0), Some(l1)) = (
            A.rb_mut().try_as_col_major_mut(),
            r_out.rb_mut().try_as_col_major_mut(),
            l_in.try_as_row_major(),
            l0.try_as_col_major(),
            l1.try_as_col_major(),
        ) {
            hessenberg_fused_op_simd(ctx, A, l_out, r_out, l_in, r_in, l0, l1, r0, r1, align);
        } else {
            hessenberg_fused_op_fallback(ctx, A, l_out, r_out, l_in, r_in, l0, l1, r0, r1);
        }
    } else {
        hessenberg_fused_op_fallback(ctx, A, l_out, r_out, l_in, r_in, l0, l1, r0, r1);
    }
}

#[math]
fn hessenberg_rearranged_unblocked<'N, 'B, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    H: MatMut<'_, C, T, Dim<'B>, Dim<'N>>,
    par: Par,
    stack: &mut DynStack,
    params: HessenbergParams,
) {
    let n = A.nrows();
    let b = H.nrows();
    let mut A = A;
    let mut H = H;
    let mut par = par;

    help!(C);
    help2!(C::Real);
    {
        let mut H = H.rb_mut().row_mut(b.idx(0));
        let (mut y, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
        let (mut z, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
        let (mut v, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
        let (mut w, _) = unsafe { temp_mat_uninit(ctx, n, par.degree(), stack) };

        let mut y = y.as_mat_mut().col_mut(0).transpose_mut();
        let mut z = z.as_mat_mut().col_mut(0);
        let mut v = v.as_mat_mut().col_mut(0).transpose_mut();
        let mut w = w.as_mat_mut();

        for k in n.indices() {
            ghost_tree!(MAT(HEAD, K, TAIL(K1, END)), {
                let (split @ l![head, _, tail], (mat_x, _, l![_, _, TAIL])) =
                    n.split(l![..k.into(), k, ..], MAT);

                let tail_split = if k.next() == n.end() {
                    None
                } else {
                    Some(tail.len().split(l![tail.len().idx(0), ..], TAIL))
                };

                let l![A0, A1, A2] = A.rb_mut().row_segments_mut(split, mat_x);
                let l![_, _, mut A02] = A0.col_segments_mut(split, mat_x);
                let l![_, mut A11, mut A12] = A1.col_segments_mut(split, mat_x);
                let l![A20, mut A21, mut A22] = A2.col_segments_mut(split, mat_x);

                let l![_, y1, mut y2] = y.rb_mut().col_segments_mut(split, mat_x);
                let l![_, z1, mut z2] = z.rb_mut().row_segments_mut(split, mat_x);
                let l![_, _, mut v2] = v.rb_mut().col_segments_mut(split, mat_x);
                let l![mut w0, _, mut w2] = w.rb_mut().row_segments_mut(split, mat_x);

                if *k > 0 {
                    let p = head.len().idx(*k - 1);
                    let u2 = A20.rb().col(p);

                    write1!(A11, math(A11 - y1 - z1));
                    z!(&mut A12, &y2, u2.rb().transpose())
                        .for_each(|uz!(mut a, y, u)| write1!(a, math(a - y - z1 * conj(u))));
                    z!(&mut A21, &u2, &z2)
                        .for_each(|uz!(mut a, u, z)| write1!(a, math(a - u * y1 - z)));
                }

                {
                    let n = *tail.len();
                    if n * n < params.par_threshold {
                        par = Par::Seq;
                    }
                }

                if let Some((split @ l![k1, _], (tail_x, _, _))) = tail_split {
                    let beta;
                    let tau_inv;
                    {
                        let l![mut A11, mut A21] = A21.rb_mut().row_segments_mut(split, tail_x);

                        let (tau, _) =
                            householder::make_householder_in_place(ctx, rb_mut!(A11), A21.rb_mut());
                        tau_inv = math(re.recip(real(tau)));
                        beta = math(copy(A11));
                        write1!(A11, math(one()));

                        write1!(H[k] = tau);
                    }

                    let x2 = A21.rb();

                    if *k > 0 {
                        let p = head.len().idx(*k - 1);
                        let u2 = A20.rb().col(p);
                        hessenberg_fused_op(
                            ctx,
                            A22.rb_mut(),
                            v2.rb_mut(),
                            w2.rb_mut().col_mut(0),
                            x2.transpose(),
                            x2,
                            u2,
                            z2.rb(),
                            y2.rb(),
                            u2.transpose(),
                            simd_align(*k.next()),
                        );
                        y2.copy_from_with(ctx, v2.rb());
                        z2.copy_from_with(ctx, w2.rb().col(0));
                    } else {
                        matmul(
                            ctx,
                            z2.rb_mut().as_mat_mut(),
                            Accum::Replace,
                            A22.rb(),
                            x2.as_mat(),
                            math(one()),
                            par,
                        );
                        matmul(
                            ctx,
                            y2.rb_mut().as_mat_mut(),
                            Accum::Replace,
                            x2.adjoint().as_mat(),
                            A22.rb(),
                            math(one()),
                            par,
                        );
                    }

                    let u2 = x2;

                    let b = math(mul_real(
                        mul_pow2(
                            dot::inner_prod(ctx, u2.rb().transpose(), Conj::Yes, z2.rb(), Conj::No),
                            re.from_f64(0.5),
                        ),
                        tau_inv,
                    ));
                    z!(&mut y2, u2.transpose()).for_each(|uz!(mut y, u)| {
                        write1!(y, math(mul_real(y - b * conj(u), tau_inv)))
                    });
                    z!(&mut z2, u2)
                        .for_each(|uz!(mut z, u)| write1!(z, math(mul_real(z - b * u, tau_inv))));

                    let dot = math(mul_real(
                        dot::inner_prod(ctx, A12.rb(), Conj::No, u2.rb(), Conj::No),
                        tau_inv,
                    ));
                    z!(&mut A12, u2.transpose())
                        .for_each(|uz!(mut a, u)| write1!(a, math(a - dot * conj(u))));

                    matmul(
                        ctx,
                        w0.rb_mut().col_mut(0).as_mat_mut(),
                        Accum::Replace,
                        A02.rb(),
                        u2.as_mat(),
                        math(one()),
                        par,
                    );
                    matmul(
                        ctx,
                        A02.rb_mut(),
                        Accum::Add,
                        w0.rb().col(0).as_mat(),
                        u2.adjoint().as_mat(),
                        math(-from_real(tau_inv)),
                        par,
                    );

                    write1!(A21[k1.local()] = beta);
                }
            });
        }
    }

    if *n > 0 {
        ghost_tree!(BLOCK, {
            let (block, _) = n.split(n.idx_inc(1).., BLOCK);
            let n = block.len();
            let A = A.rb().subcols(zero(), n).row_segment(block);
            let mut H = H.rb_mut().subcols_mut(zero(), block.len());

            let mut j_next = zero();
            while let Some(j) = n.try_check(*j_next) {
                j_next = n.advance(j, *b);

                ghost_tree!(BLOCK, ROWS, {
                    let (block, _) = n.split(j.into()..j_next, BLOCK);
                    let (rows, _) = n.split(n.idx_inc(*j).., ROWS);

                    let mut H = H
                        .rb_mut()
                        .col_segment_mut(block)
                        .subrows_mut(zero(), block.len());

                    let zero = block.len().idx(0);
                    for k in block.len().indices() {
                        write1!(H[(k, k)] = math(copy(H[(zero, k)])));
                    }

                    householder::upgrade_householder_factor(
                        ctx,
                        H.rb_mut(),
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

#[math]
fn hessenberg_gqvdg_unblocked<'N, 'B, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    Z: MatMut<'_, C, T, Dim<'N>, Dim<'B>>,
    H: MatMut<'_, C, T, Dim<'B>, Dim<'B>>,
    beta: ColMut<'_, C, T, Dim<'B>>,
    par: Par,
    stack: &mut DynStack,
    params: HessenbergParams,
) {
    let n = A.nrows();
    let b = H.nrows();
    let mut A = A;
    let mut H = H;
    let mut Z = Z;
    _ = params;

    let (mut x, _) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
    let mut x = x.as_mat_mut().col_mut(0);
    let mut beta = beta;

    help!(C);
    help2!(C::Real);

    for k in b.indices() {
        let ki = n.idx(*k);

        ghost_tree!(COL(LEFT, J, RIGHT), ROW(TOP, I, BOT(I1, END)), {
            let (col_split @ l![left, _, _], (col_x, _, _)) = b.split(l![..k.into(), k, ..], COL);
            let (row_split @ l![_, _, bot], (row_x, _, l![_, _, BOT])) =
                n.split(l![left, ki, ..], ROW);
            let bot_split = if ki.next() == n.end() {
                None
            } else {
                Some(bot.len().split(l![bot.len().idx(0), ..], BOT))
            };

            let l![mut x0, _, _] = x.rb_mut().row_segments_mut(row_split, row_x);

            let l![T0, T1, _] = H.rb_mut().row_segments_mut(col_split, col_x);
            let l![T00, mut T01, _] = T0.col_segments_mut(col_split, col_x);
            let l![_, mut T11, _] = T1.col_segments_mut(col_split, col_x);

            let l![U0, mut A1, A2] = A.rb_mut().col_segments_mut(row_split, row_x);
            let l![Z0, mut Z1, _] = Z.rb_mut().col_segments_mut(col_split, col_x);

            let U0 = U0.rb();
            let Z0 = Z0.rb();
            let T00 = T00.rb();
            let l![U00, U10, U20] = U0.row_segments(row_split);

            x0.copy_from_with(ctx, U10.adjoint());
            triangular_solve::solve_upper_triangular_in_place(
                ctx,
                T00,
                x0.rb_mut().as_mat_mut(),
                par,
            );
            matmul::matmul(
                ctx,
                A1.rb_mut().as_mat_mut(),
                Accum::Add,
                Z0,
                x0.rb().as_mat(),
                math(-one()),
                par,
            );

            let l![mut A01, mut A11, mut A21] = A1.rb_mut().row_segments_mut(row_split, row_x);

            {
                matmul::triangular::matmul(
                    ctx,
                    x0.rb_mut().as_mat_mut(),
                    BlockStructure::Rectangular,
                    Accum::Replace,
                    U00.adjoint(),
                    BlockStructure::StrictTriangularUpper,
                    A01.rb().as_mat(),
                    BlockStructure::Rectangular,
                    math(one()),
                    par,
                );
                z!(x0.rb_mut(), U10.transpose())
                    .for_each(|uz!(mut x, u)| write1!(x, math(x + A11 * conj(u))));
                matmul::matmul(
                    ctx,
                    x0.rb_mut().as_mat_mut(),
                    Accum::Add,
                    U20.adjoint(),
                    A21.rb().as_mat(),
                    math(one()),
                    par,
                );
            }
            {
                triangular_solve::solve_lower_triangular_in_place(
                    ctx,
                    T00.adjoint(),
                    x0.rb_mut().as_mat_mut(),
                    par,
                );
            }
            {
                matmul::triangular::matmul(
                    ctx,
                    A01.rb_mut().as_mat_mut(),
                    BlockStructure::Rectangular,
                    Accum::Add,
                    U00,
                    BlockStructure::StrictTriangularLower,
                    x0.rb().as_mat(),
                    BlockStructure::Rectangular,
                    math(-one()),
                    par,
                );
                write1!(
                    A11,
                    math(A11 - dot::inner_prod(ctx, U10, Conj::No, x0.rb(), Conj::No))
                );
                matmul::matmul(
                    ctx,
                    A21.rb_mut().as_mat_mut(),
                    Accum::Add,
                    U20,
                    x0.rb().as_mat(),
                    math(-one()),
                    par,
                );
            }
            if let Some((bot_split, (bot_x, _, _))) = bot_split {
                let l![mut A11, mut A21] = A21.rb_mut().row_segments_mut(bot_split, bot_x);

                let (tau, _) =
                    householder::make_householder_in_place(ctx, rb_mut!(A11), A21.rb_mut());

                write1!(beta[k] = math(copy(A11)));
                write1!(A11, math(one()));
                write1!(T11, tau);
            } else {
                write1!(T11, math(infinity()));
            }

            matmul::matmul(
                ctx,
                Z1.rb_mut().as_mat_mut(),
                Accum::Replace,
                A2.rb(),
                A21.rb().as_mat(),
                math(one()),
                par,
            );

            matmul::matmul(
                ctx,
                T01.rb_mut().as_mat_mut(),
                Accum::Replace,
                U20.adjoint(),
                A21.rb().as_mat(),
                math(one()),
                par,
            );
        });
    }
}

#[math]
fn hessenberg_gqvdg_blocked<'N, 'B, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    H: MatMut<'_, C, T, Dim<'B>, Dim<'N>>,
    par: Par,
    stack: &mut DynStack,
    params: HessenbergParams,
) {
    let n = A.nrows();
    let b = H.nrows();
    let mut A = A;
    let mut H = H;
    let (mut Z, stack) = unsafe { temp_mat_uninit(ctx, n, b, stack) };
    let mut Z = Z.as_mat_mut();
    let (mut X, stack) = unsafe { temp_mat_uninit(ctx, n, b, stack) };
    let mut X = X.as_mat_mut();

    help!(C);
    help2!(C::Real);

    let mut j_next = zero();
    while let Some(j) = n.try_check(*j_next) {
        j_next = n.advance(j, *b);

        ghost_tree!(BLOCK_TAIL, MAT(HEAD, BLOCK, TAIL), {
            let (tail, _) = n.split(j.into().., BLOCK_TAIL);
            let (split @ l![_, b, _], (mat_x, _, _)) =
                n.split(l![..j.into(), j.into()..j_next, ..], MAT);

            let b = b.len();
            let (mut beta, stack) = unsafe { temp_mat_uninit(ctx, b, 1, stack) };
            let mut beta = beta.as_mat_mut().col_mut(0);

            {
                let mut T11 = H
                    .rb_mut()
                    .col_segment_mut(tail)
                    .subcols_mut(zero(), b)
                    .subrows_mut(zero(), b);
                {
                    let A1 = A.rb_mut().row_segment_mut(tail);
                    let A11 = A1.col_segment_mut(tail);
                    let Z1 = Z.rb_mut().row_segment_mut(tail).subcols_mut(zero(), b);

                    hessenberg_gqvdg_unblocked(
                        ctx,
                        A11,
                        Z1,
                        T11.rb_mut(),
                        beta.rb_mut(),
                        par,
                        stack,
                        params,
                    );
                }

                let l![mut X0, _, mut X2] = X
                    .rb_mut()
                    .subcols_mut(zero(), b)
                    .row_segments_mut(split, mat_x);
                let l![A0, A1, A2] = A.rb_mut().row_segments_mut(split, mat_x);

                let l![_, mut A01, mut A02] = A0.col_segments_mut(split, mat_x);
                let l![_, A11, mut A12] = A1.col_segments_mut(split, mat_x);
                let l![_, A21, mut A22] = A2.col_segments_mut(split, mat_x);

                let U1 = A11.rb();
                let U2 = A21.rb();

                let l![_, mut Z1, mut Z2] = Z
                    .rb_mut()
                    .subcols_mut(zero(), b)
                    .row_segments_mut(split, mat_x);
                let T1 = T11.rb();

                matmul::triangular::matmul(
                    ctx,
                    X0.rb_mut(),
                    BlockStructure::Rectangular,
                    Accum::Replace,
                    A01.rb(),
                    BlockStructure::Rectangular,
                    U1,
                    BlockStructure::StrictTriangularLower,
                    math(one()),
                    par,
                );
                matmul::matmul(ctx, X0.rb_mut(), Accum::Add, A02.rb(), U2, math(one()), par);

                triangular_solve::solve_lower_triangular_in_place(
                    ctx,
                    T1.transpose(),
                    X0.rb_mut().transpose_mut(),
                    par,
                );

                matmul::triangular::matmul(
                    ctx,
                    A01.rb_mut(),
                    BlockStructure::Rectangular,
                    Accum::Add,
                    X0.rb(),
                    BlockStructure::Rectangular,
                    U1.adjoint(),
                    BlockStructure::StrictTriangularUpper,
                    math(-one()),
                    par,
                );
                matmul::matmul(
                    ctx,
                    A02.rb_mut(),
                    Accum::Add,
                    X0.rb(),
                    U2.adjoint(),
                    math(-one()),
                    par,
                );

                triangular_solve::solve_lower_triangular_in_place(
                    ctx,
                    T1.transpose(),
                    Z1.rb_mut().transpose_mut(),
                    par,
                );
                triangular_solve::solve_lower_triangular_in_place(
                    ctx,
                    T1.transpose(),
                    Z2.rb_mut().transpose_mut(),
                    par,
                );

                matmul::matmul(
                    ctx,
                    A12.rb_mut(),
                    Accum::Add,
                    Z1.rb(),
                    U2.adjoint(),
                    math(-one()),
                    par,
                );
                matmul::matmul(
                    ctx,
                    A22.rb_mut(),
                    Accum::Add,
                    Z2.rb(),
                    U2.adjoint(),
                    math(-one()),
                    par,
                );

                let mut X = X2.rb_mut().transpose_mut();

                matmul::triangular::matmul(
                    ctx,
                    X.rb_mut(),
                    BlockStructure::Rectangular,
                    Accum::Replace,
                    U1.adjoint(),
                    BlockStructure::StrictTriangularUpper,
                    A12.rb(),
                    BlockStructure::Rectangular,
                    math(one()),
                    par,
                );
                matmul::matmul(
                    ctx,
                    X.rb_mut(),
                    Accum::Add,
                    U2.adjoint(),
                    A22.rb(),
                    math(one()),
                    par,
                );

                triangular_solve::solve_lower_triangular_in_place(
                    ctx,
                    T1.adjoint(),
                    X.rb_mut(),
                    par,
                );

                matmul::triangular::matmul(
                    ctx,
                    A12.rb_mut(),
                    BlockStructure::Rectangular,
                    Accum::Add,
                    U1,
                    BlockStructure::StrictTriangularLower,
                    X.rb(),
                    BlockStructure::Rectangular,
                    math(-one()),
                    par,
                );
                matmul::matmul(ctx, A22.rb_mut(), Accum::Add, U2, X.rb(), math(-one()), par);
            }

            {
                let l![_, mut A, _] = A
                    .rb_mut()
                    .row_segment_mut(tail)
                    .col_segments_mut(split, mat_x);
                let n = tail.len();
                for k in b.indices() {
                    let ki = n.idx(*k);

                    ghost_tree!(COL(LEFT, J, RIGHT), ROW(TOP, I, BOT(I1, END)), {
                        let (col_split @ l![left, _, _], (col_x, _, _)) =
                            b.split(l![..k.into(), k, ..], COL);
                        let (row_split @ l![_, _, bot], (row_x, _, l![_, _, BOT])) =
                            n.split(l![left, ki, ..], ROW);
                        let bot_split = if ki.next() == n.end() {
                            None
                        } else {
                            Some(bot.len().split(l![bot.len().idx(0), ..], BOT))
                        };

                        let l![_, mut A1, _] = A.rb_mut().col_segments_mut(col_split, col_x);
                        let l![_, _, mut A21] = A1.rb_mut().row_segments_mut(row_split, row_x);
                        if let Some((bot_split, (bot_x, _, _))) = bot_split {
                            let l![mut A11, _] = A21.rb_mut().row_segments_mut(bot_split, bot_x);
                            write1!(A11, math(copy(beta[k])));
                        }
                    });
                }
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
    fn test_hessenberg_real() {
        let rng = &mut StdRng::seed_from_u64(0);

        for n in [3, 4, 8, 16] {
            with_dim!(n, n);

            let A = CwiseMatDistribution {
                nrows: n,
                ncols: n,
                dist: StandardNormal,
            }
            .rand::<Mat<f64, _, _>>(rng);

            with_dim!(br, 3);
            let mut H = Mat::zeros_with(&ctx(), br, n);

            let mut V = A.clone();
            let mut V = V.as_mut();
            hessenberg_rearranged_unblocked(
                &ctx(),
                V.rb_mut(),
                H.as_mut(),
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
                    let H = H.as_ref().subcols(zero(), rest.len());

                    householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                        &ctx(),
                        V,
                        H.as_ref(),
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
                    if *i > *j + 1 {
                        V[(i, j)] = 0.0;
                    }
                }
            }

            assert!(V ~ A);
        }
    }

    #[test]
    fn test_hessenberg_cplx() {
        let rng = &mut StdRng::seed_from_u64(0);

        for n in [1, 2, 3, 4, 8, 16] {
            for par in [Par::Seq, Par::rayon(4)] {
                with_dim!(n, n);
                let A = CwiseMatDistribution {
                    nrows: n,
                    ncols: n,
                    dist: ComplexDistribution::new(StandardNormal, StandardNormal),
                }
                .rand::<Mat<c64, _, _>>(rng);

                with_dim!(br, 1);
                let mut H = Mat::zeros_with(&ctx(), br, n);

                let mut V = A.clone();
                let mut V = V.as_mut();
                hessenberg_rearranged_unblocked(
                    &ctx(),
                    V.rb_mut(),
                    H.as_mut(),
                    par,
                    DynStack::new(&mut [MaybeUninit::uninit(); 8 * 1024]),
                    HessenbergParams {
                        par_threshold: 0,
                        ..Default::default()
                    },
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
                        let H = H.as_ref().subcols(zero(), rest.len());

                        householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                        &ctx(),
                        V,
                        H.as_ref(),
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
                        if *i > *j + 1 {
                            V[(i, j)] = c64::ZERO;
                        }
                    }
                }

                assert!(V ~ A);
            }
        }
    }

    #[test]
    fn test_hessenberg_cplx_gqvdg() {
        let rng = &mut StdRng::seed_from_u64(0);

        for n in [2, 3, 4, 8, 16, 21] {
            for par in [Par::Seq, Par::rayon(4)] {
                with_dim!(n, n);
                with_dim!(b, 4);

                let A = CwiseMatDistribution {
                    nrows: n,
                    ncols: n,
                    dist: ComplexDistribution::new(StandardNormal, StandardNormal),
                }
                .rand::<Mat<c64, _, _>>(rng);

                let mut H = Mat::zeros_with(&ctx(), b, n);

                let mut V = A.clone();
                let mut V = V.as_mut();
                hessenberg_gqvdg_blocked(
                    &ctx(),
                    V.rb_mut(),
                    H.as_mut(),
                    par,
                    DynStack::new(&mut [MaybeUninit::uninit(); 16 * 1024]),
                    HessenbergParams {
                        par_threshold: 0,
                        ..Default::default()
                    },
                );

                {
                    let mut V = A.clone();
                    let mut V = V.as_mut();
                    let mut H = Mat::zeros_with(&ctx(), n, n);
                    hessenberg_rearranged_unblocked(
                        &ctx(),
                        V.rb_mut(),
                        H.as_mut(),
                        par,
                        DynStack::new(&mut [MaybeUninit::uninit(); 8 * 1024]),
                        HessenbergParams {
                            par_threshold: 0,
                            ..Default::default()
                        },
                    );
                }

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
                        let H = H.as_ref().subcols(zero(), rest.len());

                        householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                        &ctx(),
                        V,
                        H.as_ref(),
                        if iter == 0{Conj::Yes} else {Conj::No},
                        A.rb_mut(),
                        Par::Seq,
                        DynStack::new(&mut GlobalMemBuffer::new(
                            householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<
                                Unit,
                                c64,
                            >(*n - 1, *n, *n)
                            .unwrap(),
                        )),
                    );
                    });
                }

                let approx_eq = CwiseMat(ApproxEq::<Unit, c64>::eps());
                for j in n.indices() {
                    for i in n.indices() {
                        if *i > *j + 1 {
                            V[(i, j)] = c64::ZERO;
                        }
                    }
                }

                assert!(V ~ A);
            }
        }
    }
}
