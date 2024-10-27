use crate::{assert, internal_prelude::*};
use linalg::{
    householder,
    matmul::{dot, matmul},
};

pub fn bidiag_in_place_scratch<T: ComplexField>(
    nrows: usize,
    ncols: usize,
    par: Par,
    params: BidiagParams,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    _ = params;
    StackReq::try_all_of([
        temp_mat_scratch::<T>(nrows, 1)?,
        temp_mat_scratch::<T>(ncols, 1)?,
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
pub fn bidiag_in_place<'M, 'N, 'BL, 'BR, 'H, T: ComplexField>(
    A: MatMut<'_, T, Dim<'M>, Dim<'N>>,
    H_left: MatMut<'_, T, Dim<'BL>, Dim<'H>>,
    H_right: MatMut<'_, T, Dim<'BR>, Dim<'H>>,
    par: Par,
    stack: &mut DynStack,
    params: BidiagParams,
) {
    let m = A.nrows();
    let n = A.ncols();
    let mn = H_left.ncols();
    let bl = H_left.nrows();
    let br = H_right.nrows();
    assert!(*H_left.ncols() == Ord::min(*m, *n));

    let (mut y, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
    let (mut z, _) = unsafe { temp_mat_uninit(m, 1, stack) };

    let mut y = y.as_mat_mut().col_mut(0).transpose_mut();
    let mut z = z.as_mat_mut().col_mut(0);

    let mut A = A;
    let mut Hl = H_left;
    let mut Hr = H_right;
    let mut par = par;

    {
        let mut Hl = Hl.rb_mut().row_mut(bl.idx(0));
        let mut Hr = Hr.rb_mut().row_mut(br.idx(0));

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
                let l![A10, a11, mut A12] = A1.col_segments_mut(col_split, cols_x);
                let l![A20, mut A21, mut A22] = A2.col_segments_mut(col_split, cols_x);

                let l![_, y1, mut y2] = y.rb_mut().col_segments_mut(col_split, cols_x);
                let l![_, z1, mut z2] = z.rb_mut().row_segments_mut(row_split, rows_x);

                if *kj > 0 {
                    let kj1 = left.idx(*kj - 1);
                    let ki1 = top.idx(*ki - 1);

                    let up0 = A10.rb().at(left.local(kj1));
                    let up = A20.rb().col(left.local(kj1));
                    let vp = A02.row(top.local(ki1));

                    *a11 = a11 - up0 * y1 - z1;
                    z!(A21.rb_mut(), up.rb(), z2.rb()).for_each(|uz!(a, u, z)| *a = a - u * y1 - z);
                    z!(A12.rb_mut(), y2.rb(), vp.rb())
                        .for_each(|uz!(a, y, v)| *a = a - up0 * y - z1 * v);
                }

                let (tl, _) = householder::make_householder_in_place(a11, A21.rb_mut());
                let tl_inv = recip(real(tl));
                Hl[k] = tl;

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
                            A22.rb_mut(),
                            A21.rb(),
                            up.rb(),
                            z2.rb(),
                            y2.rb_mut(),
                            vp.rb(),
                            simd_align(*ki.next()),
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
                                        A22.as_col_shape_mut(N),
                                        A21.rb(),
                                        up.rb(),
                                        z2.rb(),
                                        y2.as_col_shape_mut(N),
                                        vp.rb().as_col_shape(N),
                                        simd_align(*ki.next()),
                                    );
                                });
                        }
                    }
                } else {
                    matmul(
                        y2.rb_mut().as_mat_mut(),
                        Accum::Replace,
                        A21.rb().adjoint().as_mat(),
                        A22.rb(),
                        one(),
                        par,
                    );
                }

                z!(y2.rb_mut(), A12.rb_mut()).for_each(|uz!(y, a)| {
                    *y = mul_real(y + a, tl_inv);
                    *a = a - y;
                });
                let norm = A12.rb().norm_l2();
                let norm_inv = recip(norm);
                if norm != zero() {
                    z!(A12.rb_mut()).for_each(|uz!(a)| *a = mul_real(a, norm_inv));
                }
                matmul(
                    z2.rb_mut().as_mat_mut(),
                    Accum::Replace,
                    A22.rb(),
                    A12.rb().adjoint().as_mat(),
                    one(),
                    par,
                );

                if k.next() == mn.end() {
                    break;
                }

                let kj1 = right.global(right.idx(*kj + 1));

                let (l![j1, next], (rows_x2, ..)) =
                    right.len().split(l![right.local(kj1), ..], RIGHT);
                let l![a12_a, mut A12_b] = A12.rb_mut().col_segments_mut(l![j1, next], rows_x2);
                let l![A22_a, _] = A22.rb().col_segments(l![j1, next]);
                let l![y2_a, y2_b] = y2.rb().col_segments(l![j1, next]);

                let (tr, m) =
                    householder::make_householder_in_place(a12_a, A12_b.rb_mut().transpose_mut());
                let tr_inv = recip(real(tr));
                Hr[k] = tr;
                let beta = copy(*a12_a);
                *a12_a = mul_real(*a12_a, norm);

                let b = *y2_a + dot::inner_prod(y2_b, Conj::No, A12_b.rb().transpose(), Conj::Yes);

                if let Some(m) = m {
                    z!(z2.rb_mut(), A21.rb(), A22_a.rb()).for_each(|uz!(z, u, a)| {
                        let w = *z - *a * conj(beta);
                        let w = w * conj(m);
                        let w = w - *u * b;
                        *z = mul_real(w, tr_inv);
                    });
                } else {
                    z!(z2.rb_mut(), A21.rb(), A22_a.rb()).for_each(|uz!(z, u, a)| {
                        let w = *a - *u * b;
                        *z = mul_real(w, tr_inv);
                    });
                }
            });
        }
    }

    let mut j_next = IdxInc::ZERO;
    while let Some(j) = mn.try_check(*j_next) {
        j_next = mn.advance(j, *bl);

        ghost_tree!(BLOCK, COLS, ROWS, {
            let (block, _) = mn.split(j.into()..j_next, BLOCK);
            let (cols, _) = n.split(block, COLS);
            let (rows, _) = m.split(m.idx_inc(*j).., ROWS);

            let mut Hl = Hl
                .rb_mut()
                .col_segment_mut(block)
                .subrows_mut(IdxInc::ZERO, block.len());

            let zero = cols.len().idx(0);
            for k in cols.len().indices() {
                Hl[(k, k)] = copy(Hl[(zero, k)]);
            }

            householder::upgrade_householder_factor(
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
            let A = A.rb().col_segment(cols).subrows(IdxInc::ZERO, mn);
            let mut Hr = Hr.rb_mut().subcols_mut(IdxInc::ZERO, mn);

            let mut j_next = IdxInc::ZERO;
            while let Some(j) = mn.try_check(*j_next) {
                j_next = mn.advance(j, *br);

                ghost_tree!(BLOCK, ROWS, {
                    let (block, _) = mn.split(j.into()..j_next, BLOCK);
                    let (rows, _) = n.split(n.idx_inc(*j).., ROWS);

                    let mut Hr = Hr
                        .rb_mut()
                        .col_segment_mut(block)
                        .subrows_mut(IdxInc::ZERO, block.len());

                    let zero = block.len().idx(0);
                    for k in block.len().indices() {
                        Hr[(k, k)] = copy(Hr[(zero, k)]);
                    }

                    householder::upgrade_householder_factor(
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
fn bidiag_fused_op<'M, 'N, T: ComplexField>(
    A22: MatMut<'_, T, Dim<'M>, Dim<'N>>,

    u: ColRef<'_, T, Dim<'M>>,

    up: ColRef<'_, T, Dim<'M>>,
    z: ColRef<'_, T, Dim<'M>>,

    y: RowMut<'_, T, Dim<'N>>,
    vp: RowRef<'_, T, Dim<'N>>,

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
            bidiag_fused_op_simd(A22, u, up, z, y, vp, align);
        } else {
            bidiag_fused_op_fallback(A22, u, up, z, y, vp);
        }
    } else {
        bidiag_fused_op_fallback(A22, u, up, z, y, vp);
    }
}

#[math]
fn bidiag_fused_op_fallback<'M, 'N, T: ComplexField>(
    A22: MatMut<'_, T, Dim<'M>, Dim<'N>>,

    u: ColRef<'_, T, Dim<'M>>,

    up: ColRef<'_, T, Dim<'M>>,
    z: ColRef<'_, T, Dim<'M>>,

    y: RowMut<'_, T, Dim<'N>>,
    vp: RowRef<'_, T, Dim<'N>>,
) {
    let mut A22 = A22;
    let mut y = y;

    matmul(
        A22.rb_mut(),
        Accum::Add,
        up.as_mat(),
        y.rb().as_mat(),
        -one(),
        Par::Seq,
    );
    matmul(
        A22.rb_mut(),
        Accum::Add,
        z.as_mat(),
        vp.as_mat(),
        -one(),
        Par::Seq,
    );
    matmul(
        y.rb_mut().as_mat_mut(),
        Accum::Replace,
        u.adjoint().as_mat(),
        A22.rb(),
        one(),
        Par::Seq,
    );
}

#[math]
fn bidiag_fused_op_simd<'M, 'N, T: ComplexField>(
    A22: MatMut<'_, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
    u: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
    up: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
    z: ColRef<'_, T, Dim<'M>, ContiguousFwd>,

    y: RowMut<'_, T, Dim<'N>>,
    vp: RowRef<'_, T, Dim<'N>>,

    align: usize,
) {
    struct Impl<'a, 'M, 'N, T: ComplexField> {
        A22: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
        u: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
        up: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
        z: ColRef<'a, T, Dim<'M>, ContiguousFwd>,

        y: RowMut<'a, T, Dim<'N>>,
        vp: RowRef<'a, T, Dim<'N>>,

        align: usize,
    }

    impl<'a, 'M, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'M, 'N, T> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self {
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
            let simd = SimdCtx::<T, S>::new_align(T::simd_ctx(simd), m, align);
            let (head, body4, body1, tail) = simd.batch_indices::<4>();

            for j in n.indices() {
                let mut a = A22.rb_mut().col_mut(j);

                let mut acc0 = simd.zero();
                let mut acc1 = simd.zero();
                let mut acc2 = simd.zero();
                let mut acc3 = simd.zero();

                let yj = simd.splat(&-y[j]);
                let vj = simd.splat(&-vp[j]);

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

                y[j] = simd.reduce_sum(acc0);
            }
        }
    }

    T::Arch::default().dispatch(Impl {
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
            let mut Hl = Mat::zeros(bl, mn);
            let mut Hr = Mat::zeros(br, mn);

            let mut UV = A.clone();
            let mut UV = UV.as_mut();
            bidiag_in_place(
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
                UV.rb().subcols(IdxInc::ZERO, mn),
                Hl.as_ref(),
                Conj::Yes,
                A.rb_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<
                    f64,
                >(*n - 1, 1, *m)
                    .unwrap(),
                )),
            );

            ghost_tree!(BLOCK(K0, REST), COLS(J0, RIGHT), {
                let (l![_, rest], _) = mn.split(l![mn.idx(0), ..], BLOCK);
                let (col_split, (col_x, ..)) = n.split(l![n.idx(0), ..], COLS);
                let UV = UV.rb().subrows(IdxInc::ZERO, rest.len());

                let l![_, mut A1] = A.rb_mut().col_segments_mut(col_split, col_x);
                let l![_, V] = UV.rb().col_segments(col_split);
                let Hr = Hr.as_ref().subcols(IdxInc::ZERO, rest.len());

                householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
                    V.transpose(),
                    Hr.as_ref(),
                    Conj::Yes,
                    A1.rb_mut(),
                    Par::Seq,
                    DynStack::new(&mut GlobalMemBuffer::new(
                        householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<
                            f64,
                        >(*n - 1, 1, *m)
                        .unwrap(),
                    )),
                );
            });

            let approx_eq = CwiseMat(ApproxEq::<f64>::eps());
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
            let mut Hl = Mat::zeros(bl, mn);
            let mut Hr = Mat::zeros(br, mn);

            let mut UV = A.clone();
            let mut UV = UV.as_mut();
            bidiag_in_place(
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
                UV.rb().subcols(IdxInc::ZERO, mn),
                Hl.as_ref(),
                Conj::Yes,
                A.rb_mut(),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<
                    c64,
                >(*n - 1, 1, *m)
                    .unwrap(),
                )),
            );

            ghost_tree!(BLOCK(K0, REST), COLS(J0, RIGHT), {
                let (l![_, rest], _) = mn.split(l![mn.idx(0), ..], BLOCK);
                let (col_split, (col_x, ..)) = n.split(l![n.idx(0), ..], COLS);
                let UV = UV.rb().subrows(IdxInc::ZERO, rest.len());

                let l![_, mut A1] = A.rb_mut().col_segments_mut(col_split, col_x);
                let l![_, V] = UV.rb().col_segments(col_split);
                let Hr = Hr.as_ref().subcols(IdxInc::ZERO, rest.len());

                householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
                    V.transpose(),
                    Hr.as_ref(),
                    Conj::Yes,
                    A1.rb_mut(),
                    Par::Seq,
                    DynStack::new(&mut GlobalMemBuffer::new(
                        householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<
                            c64,
                        >(*n - 1, 1, *m)
                        .unwrap(),
                    )),
                );
            });

            let approx_eq = CwiseMat(ApproxEq::<c64>::eps());
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
