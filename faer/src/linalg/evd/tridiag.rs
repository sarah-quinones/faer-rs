use crate::{assert, internal_prelude::*};
use linalg::{
    householder,
    matmul::{self, dot, triangular::BlockStructure},
};

pub fn tridiag_in_place_scratch<C: ComplexContainer, T: ComplexField<C>>(
    dim: usize,
    par: Par,
    params: TridiagParams,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    _ = params;
    StackReq::try_all_of([
        temp_mat_scratch::<C, T>(dim, 1)?.try_array(2)?,
        temp_mat_scratch::<C, T>(dim, par.degree())?,
    ])
}

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
pub fn tridiag_fused_op_simd<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
    y2: ColMut<'_, C, T, Dim<'N>>,
    z2: ColMut<'_, C, T, Dim<'M>, ContiguousFwd>,

    ry2: ColRef<'_, C, T, Dim<'N>>,
    rz2: ColRef<'_, C, T, Dim<'M>, ContiguousFwd>,

    u0: ColRef<'_, C, T, Dim<'M>, ContiguousFwd>,
    u1: ColRef<'_, C, T, Dim<'N>>,
    u2: ColRef<'_, C, T, Dim<'N>>,
    v2: ColRef<'_, C, T, Dim<'M>, ContiguousFwd>,

    f: C::Of<T>,
    align: usize,
) {
    struct Impl<'a, 'M, 'N, C: ComplexContainer, T: ComplexField<C>> {
        ctx: &'a Ctx<C, T>,
        A: MatMut<'a, C, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
        y2: ColMut<'a, C, T, Dim<'N>>,
        z2: ColMut<'a, C, T, Dim<'M>, ContiguousFwd>,

        ry2: ColRef<'a, C, T, Dim<'N>>,
        rz2: ColRef<'a, C, T, Dim<'M>, ContiguousFwd>,

        u0: ColRef<'a, C, T, Dim<'M>, ContiguousFwd>,
        u1: ColRef<'a, C, T, Dim<'N>>,
        u2: ColRef<'a, C, T, Dim<'N>>,
        v2: ColRef<'a, C, T, Dim<'M>, ContiguousFwd>,

        f: C::Of<T>,
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
                mut y2,
                mut z2,
                ry2,
                rz2,
                u0,
                u1,
                u2,
                v2,
                f,
                mut align,
            } = self;

            help!(C);

            let simd = T::simd_ctx(ctx, simd);
            let (m, n) = A.shape();
            {
                let simd = SimdCtx::<C, T, S>::new_align(simd, m, align);
                let (head, body, tail) = simd.indices();

                if let Some(i0) = head {
                    simd.write(z2.rb_mut(), i0, simd.zero());
                }
                for i0 in body {
                    simd.write(z2.rb_mut(), i0, simd.zero());
                }
                if let Some(i0) = tail {
                    simd.write(z2.rb_mut(), i0, simd.zero());
                }
            }

            for j in n.indices() {
                ghost_tree!(ROW, {
                    let (tail, _) = m.split(m.idx_inc(*j).., ROW);
                    let m = tail.len();

                    let simd = SimdCtx::<C, T, S>::new_align(simd, tail.len(), align);
                    align -= 1;

                    let mut A = A.rb_mut().col_mut(j).row_segment_mut(tail);

                    let mut z = z2.rb_mut().row_segment_mut(tail);
                    let rz = rz2.row_segment(tail);
                    let ua = u0.row_segment(tail);
                    let v = v2.row_segment(tail);

                    let mut y = y2.rb_mut().at_mut(j);
                    let ry = simd.splat(as_ref!(math(-ry2[j])));
                    let ub = simd.splat(as_ref!(math(-u1[j])));
                    let uc = simd.splat(as_ref!(math(f * u2[j])));

                    let mut acc0 = simd.zero();
                    let mut acc1 = simd.zero();
                    let mut acc2 = simd.zero();
                    let mut acc3 = simd.zero();

                    let (head, body4, body1, tail) = simd.batch_indices::<4>();
                    if let Some(i0) = head {
                        let mut a = simd.read(A.rb(), i0);
                        a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
                        a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
                        simd.write(A.rb_mut(), i0, a);

                        let tmp = simd.read(z.rb(), i0);
                        simd.write(z.rb_mut(), i0, simd.mul_add(a, uc, tmp));

                        acc0 = simd.conj_mul_add(a, simd.read(v, i0), acc0);
                    }

                    for [i0, i1, i2, i3] in body4 {
                        {
                            let mut a = simd.read(A.rb(), i0);
                            a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
                            a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
                            simd.write(A.rb_mut(), i0, a);

                            let tmp = simd.read(z.rb(), i0);
                            simd.write(z.rb_mut(), i0, simd.mul_add(a, uc, tmp));

                            acc0 = simd.conj_mul_add(a, simd.read(v, i0), acc0);
                        }
                        {
                            let mut a = simd.read(A.rb(), i1);
                            a = simd.conj_mul_add(ry, simd.read(ua, i1), a);
                            a = simd.conj_mul_add(ub, simd.read(rz, i1), a);
                            simd.write(A.rb_mut(), i1, a);

                            let tmp = simd.read(z.rb(), i1);
                            simd.write(z.rb_mut(), i1, simd.mul_add(a, uc, tmp));

                            acc1 = simd.conj_mul_add(a, simd.read(v, i1), acc1);
                        }
                        {
                            let mut a = simd.read(A.rb(), i2);
                            a = simd.conj_mul_add(ry, simd.read(ua, i2), a);
                            a = simd.conj_mul_add(ub, simd.read(rz, i2), a);
                            simd.write(A.rb_mut(), i2, a);

                            let tmp = simd.read(z.rb(), i2);
                            simd.write(z.rb_mut(), i2, simd.mul_add(a, uc, tmp));

                            acc2 = simd.conj_mul_add(a, simd.read(v, i2), acc2);
                        }
                        {
                            let mut a = simd.read(A.rb(), i3);
                            a = simd.conj_mul_add(ry, simd.read(ua, i3), a);
                            a = simd.conj_mul_add(ub, simd.read(rz, i3), a);
                            simd.write(A.rb_mut(), i3, a);

                            let tmp = simd.read(z.rb(), i3);
                            simd.write(z.rb_mut(), i3, simd.mul_add(a, uc, tmp));

                            acc3 = simd.conj_mul_add(a, simd.read(v, i3), acc3);
                        }
                    }
                    for i0 in body1 {
                        let mut a = simd.read(A.rb(), i0);
                        a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
                        a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
                        simd.write(A.rb_mut(), i0, a);

                        let tmp = simd.read(z.rb(), i0);
                        simd.write(z.rb_mut(), i0, simd.mul_add(a, uc, tmp));

                        acc0 = simd.conj_mul_add(a, simd.read(v, i0), acc0);
                    }
                    if let Some(i0) = tail {
                        let mut a = simd.read(A.rb(), i0);
                        a = simd.conj_mul_add(ry, simd.read(ua, i0), a);
                        a = simd.conj_mul_add(ub, simd.read(rz, i0), a);
                        simd.write(A.rb_mut(), i0, a);

                        let tmp = simd.read(z.rb(), i0);
                        simd.write(z.rb_mut(), i0, simd.mul_add(a, uc, tmp));

                        acc0 = simd.conj_mul_add(a, simd.read(v, i0), acc0);
                    }

                    acc0 = simd.add(acc0, acc1);
                    acc2 = simd.add(acc2, acc3);
                    acc0 = simd.add(acc0, acc2);

                    let acc0 = simd.reduce_sum(acc0);
                    let i0 = m.idx(0);
                    write1!(y, math(f * (acc0 - A[i0] * v[i0])));
                });
            }
        }
    }

    T::Arch::default().dispatch(Impl {
        ctx,
        A,
        y2,
        z2,
        ry2,
        rz2,
        u0,
        u1,
        u2,
        v2,
        f,
        align,
    })
}

#[math]
pub fn tridiag_fused_op<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    y2: ColMut<'_, C, T, Dim<'N>>,
    z2: ColMut<'_, C, T, Dim<'M>>,

    ry2: ColRef<'_, C, T, Dim<'N>>,
    rz2: ColRef<'_, C, T, Dim<'M>>,

    u0: ColRef<'_, C, T, Dim<'M>>,
    u1: ColRef<'_, C, T, Dim<'N>>,
    u2: ColRef<'_, C, T, Dim<'N>>,
    v2: ColRef<'_, C, T, Dim<'M>>,

    f: C::Of<T>,
    align: usize,
) {
    let mut A = A;
    let mut z2 = z2;

    if const { T::SIMD_CAPABILITIES.is_simd() } {
        if let (Some(A), Some(z2), Some(rz2), Some(u0), Some(v2)) = (
            A.rb_mut().try_as_col_major_mut(),
            z2.rb_mut().try_as_col_major_mut(),
            rz2.try_as_col_major(),
            u0.try_as_col_major(),
            v2.try_as_col_major(),
        ) {
            tridiag_fused_op_simd(ctx, A, y2, z2, ry2, rz2, u0, u1, u2, v2, f, align);
        } else {
            tridiag_fused_op_fallback(ctx, A, y2, z2, ry2, rz2, u0, u1, u2, v2, f);
        }
    } else {
        tridiag_fused_op_fallback(ctx, A, y2, z2, ry2, rz2, u0, u1, u2, v2, f);
    }
}

#[math]
pub fn tridiag_fused_op_fallback<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    y2: ColMut<'_, C, T, Dim<'N>>,
    z2: ColMut<'_, C, T, Dim<'M>>,

    ry2: ColRef<'_, C, T, Dim<'N>>,
    rz2: ColRef<'_, C, T, Dim<'M>>,

    u0: ColRef<'_, C, T, Dim<'M>>,
    u1: ColRef<'_, C, T, Dim<'N>>,
    u2: ColRef<'_, C, T, Dim<'N>>,
    v2: ColRef<'_, C, T, Dim<'M>>,

    f: C::Of<T>,
) {
    let par = Par::Seq;

    let mut A = A;
    let mut y2 = y2;

    ghost_tree!(ROWS(TOP, BOT), COLS, {
        let (cols, _) = A.ncols().split(.., COLS);
        let (rows, (row_x, _, _)) = A.nrows().split(l![cols, ..], ROWS);

        let l![mut A0, mut A1] = A.rb_mut().row_segments_mut(rows, row_x);
        let l![u00, u01] = u0.row_segments(rows);
        let l![v20, v21] = v2.row_segments(rows);
        let l![mut z20, mut z21] = z2.row_segments_mut(rows, row_x);

        let l![rz20, rz21] = rz2.row_segments(rows);

        matmul::triangular::matmul(
            ctx,
            A0.rb_mut(),
            BlockStructure::TriangularLower,
            Accum::Add,
            u00.rb().as_mat(),
            BlockStructure::Rectangular,
            ry2.rb().adjoint().as_mat(),
            BlockStructure::Rectangular,
            math(-one()),
            par,
        );
        matmul::triangular::matmul(
            ctx,
            A0.rb_mut(),
            BlockStructure::TriangularLower,
            Accum::Add,
            rz20.rb().as_mat(),
            BlockStructure::Rectangular,
            u1.rb().adjoint().as_mat(),
            BlockStructure::Rectangular,
            math(-one()),
            par,
        );
        matmul::matmul(
            ctx,
            A1.rb_mut(),
            Accum::Add,
            u01.as_mat(),
            ry2.rb().adjoint().as_mat(),
            math(-one()),
            par,
        );
        matmul::matmul(
            ctx,
            A1.rb_mut(),
            Accum::Add,
            rz21.rb().as_mat(),
            u1.adjoint().as_mat(),
            math(-one()),
            par,
        );

        help!(C);
        matmul::triangular::matmul(
            ctx,
            z20.rb_mut().as_mat_mut(),
            BlockStructure::Rectangular,
            Accum::Replace,
            A0.rb(),
            BlockStructure::TriangularLower,
            u2.rb().as_mat(),
            BlockStructure::Rectangular,
            as_ref!(f),
            par,
        );
        matmul::triangular::matmul(
            ctx,
            y2.rb_mut().as_mat_mut(),
            BlockStructure::Rectangular,
            Accum::Replace,
            A0.rb().adjoint(),
            BlockStructure::StrictTriangularUpper,
            v20.rb().as_mat(),
            BlockStructure::Rectangular,
            as_ref!(f),
            par,
        );

        matmul::matmul(
            ctx,
            z21.rb_mut().as_mat_mut(),
            Accum::Replace,
            A1.rb(),
            u2.rb().as_mat(),
            as_ref!(f),
            par,
        );
        matmul::matmul(
            ctx,
            y2.rb_mut().as_mat_mut(),
            Accum::Add,
            A1.rb().adjoint(),
            v21.rb().as_mat(),
            as_ref!(f),
            par,
        );
    });
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
    let (mut w, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
    let (mut z, _) = unsafe { temp_mat_uninit(ctx, n, par.degree(), stack) };
    let mut y = y.as_mat_mut().col_mut(0);
    let mut w = w.as_mat_mut().col_mut(0);
    let mut z = z.as_mat_mut();
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
                    let l![mut a11, mut x2] = A21.rb_mut().row_segments_mut(split, tail_x);
                    let (tau, _) =
                        householder::make_householder_in_place(ctx, rb_mut!(a11), x2.rb_mut());

                    tau_inv = math(re.recip(real(tau)));
                    write1!(H[k1.local()] = tau);

                    let l![_, mut z2] = z
                        .rb_mut()
                        .row_segment_mut(tail)
                        .row_segments_mut(split, tail_x);
                    let l![_, mut w2] = w
                        .rb_mut()
                        .row_segment_mut(tail)
                        .row_segments_mut(split, tail_x);
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

                        w2.copy_from_with(ctx, y2.rb());

                        match par {
                            Par::Seq => {
                                let mut z2 = z2.rb_mut().col_mut(0);
                                tridiag_fused_op(
                                    ctx,
                                    A22.rb_mut(),
                                    y2.rb_mut(),
                                    z2.rb_mut(),
                                    w2.rb(),
                                    w2.rb(),
                                    u2.rb(),
                                    u2.rb(),
                                    x2.rb(),
                                    x2.rb(),
                                    math(from_real(tau_inv)),
                                    n.next_power_of_two() - *k1.next(),
                                );
                                z!(y2.rb_mut(), z2.rb_mut())
                                    .for_each(|uz!(mut y, z)| write1!(y, math(y + z)));
                            }
                            #[cfg(feature = "rayon")]
                            Par::Rayon(nthreads) => {
                                use rayon::prelude::*;
                                let nthreads = nthreads.get();
                                let mut z2 = z2.rb_mut().subcols_mut(0, nthreads);

                                let n2 = A22.ncols();
                                assert!((*n2 as u64) < (1u64 << 50)); // to check that integers can be
                                                                      // represented exactly as floats

                                let idx_to_col_start = |idx: usize| {
                                    let idx_as_percent = idx as f64 / nthreads as f64;
                                    let col_start_percent =
                                        1.0f64 - libm::sqrt(1.0f64 - idx_as_percent);
                                    (col_start_percent * (*n2) as f64) as usize
                                };

                                {
                                    let A22 = A22.rb();
                                    let y2 = y2.rb();

                                    let f = math(from_real(tau_inv));
                                    let f = sync!(as_ref!(f));
                                    z2.rb_mut().par_col_iter_mut().enumerate().for_each(
                                        |(idx, mut z2)| {
                                            let first = idx_to_col_start(idx);
                                            let last_col = idx_to_col_start(idx + 1);
                                            with_dim!(nrows, *n2 - first);
                                            with_dim!(ncols, last_col - first);

                                            let first = n2.idx_inc(first);

                                            let mut A = unsafe {
                                                A22.rb()
                                                    .subcols(first, ncols)
                                                    .subrows(first, nrows)
                                                    .const_cast()
                                            };

                                            {
                                                let y2 = unsafe {
                                                    y2.subrows(first, ncols).const_cast()
                                                };
                                                let mut z2 = z2.rb_mut().subrows_mut(first, nrows);

                                                let ry2 = w2.rb().subrows(first, ncols);
                                                let rz2 = w2.rb().subrows(first, nrows);

                                                let u0 = u2.subrows(first, nrows);
                                                let u1 = u2.subrows(first, ncols);
                                                let u2 = x2.rb().subrows(first, ncols);
                                                let v2 = x2.rb().subrows(first, nrows);

                                                tridiag_fused_op(
                                                    ctx,
                                                    A.rb_mut(),
                                                    y2,
                                                    z2.rb_mut(),
                                                    ry2,
                                                    rz2,
                                                    u0,
                                                    u1,
                                                    u2,
                                                    v2,
                                                    math.copy(unsync!(f)),
                                                    n.next_power_of_two() - *k1.next() - *first,
                                                );
                                            }

                                            let z2 = z2
                                                .rb_mut()
                                                .subrows_range_mut((n2.idx_inc(0), first));
                                            z!(z2).for_each(|uz!(mut z)| write1!(z, math(zero())));
                                        },
                                    );
                                }

                                for mut z2 in z2.rb_mut().col_iter_mut() {
                                    z!(y2.rb_mut(), z2.rb_mut())
                                        .for_each(|uz!(mut y, z)| write1!(y, math(y + z)));
                                }
                            }
                        }
                    } else {
                        matmul::triangular::matmul(
                            ctx,
                            y2.rb_mut().as_mat_mut(),
                            BlockStructure::Rectangular,
                            Accum::Replace,
                            A22.rb(),
                            BlockStructure::TriangularLower,
                            x2.rb().as_mat(),
                            BlockStructure::Rectangular,
                            math(from_real(tau_inv)),
                            par,
                        );
                        matmul::triangular::matmul(
                            ctx,
                            y2.rb_mut().as_mat_mut(),
                            BlockStructure::Rectangular,
                            Accum::Add,
                            A22.rb().adjoint(),
                            BlockStructure::StrictTriangularUpper,
                            x2.rb().as_mat(),
                            BlockStructure::Rectangular,
                            math(from_real(tau_inv)),
                            par,
                        );
                    }

                    z!(y2.rb_mut(), A21.rb())
                        .for_each(|uz!(mut y, a)| write1!(y, math(y + mul_real(a, tau_inv))));

                    write1!(
                        y1,
                        math(mul_real(
                            a11 + dot::inner_prod(
                                ctx,
                                A21.rb().transpose(),
                                Conj::Yes,
                                x2.rb(),
                                Conj::No
                            ),
                            tau_inv
                        ))
                    );

                    let b = math(mul_real(
                        mul_pow2(
                            y1 + dot::inner_prod(
                                ctx,
                                x2.rb().transpose(),
                                Conj::Yes,
                                y2.rb(),
                                Conj::No,
                            ),
                            re.from_f64(0.5),
                        ),
                        tau_inv,
                    ));
                    write1!(y1, math(y1 - b));
                    z!(y2.rb_mut(), x2.rb()).for_each(|uz!(mut y, u)| {
                        write1!(y, math(y - b * u));
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

        for n in [1, 2, 3, 4, 8, 16] {
            for par in [Par::Seq, Par::rayon(4)] {
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
                    par,
                    DynStack::new(&mut [MaybeUninit::uninit(); 8 * 1024]),
                    TridiagParams { par_threshold: 0 },
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
}
