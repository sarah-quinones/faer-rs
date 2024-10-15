use crate::{assert, internal_prelude::*};
use faer_traits::RealValue;
use linalg::{
    householder,
    matmul::{dot, matmul},
};

pub fn bidiag_in_place_scratch<C: ComplexContainer, T: ComplexField<C>>(
    nrows: usize,
    ncols: usize,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        temp_mat_scratch::<C, T>(nrows, 1)?,
        temp_mat_scratch::<C, T>(nrows, par.degree())?,
        temp_mat_scratch::<C, T>(ncols, 1)?,
    ])
}

#[math]
pub fn bidiag_in_place<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    H_left: RowMut<'_, C, T, Dim<'N>>,
    H_right: RowMut<'_, C, T, Dim<'N>>,
    par: Par,
    stack: &mut DynStack,
) {
    help!(C);
    help2!(C::Real);

    let nthreads = par.degree();
    let m = A.nrows();
    let n = A.ncols();
    assert!(*m >= *n);

    let (mut y, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
    let (mut z, stack) = unsafe { temp_mat_uninit(ctx, m, 1, stack) };
    let (mut z_tmp, _) = unsafe { temp_mat_uninit(ctx, m, nthreads, stack) };

    let mut y = y.as_mat_mut().col_mut(0).transpose_mut();
    let mut z = z.as_mat_mut().col_mut(0);
    let mut z_tmp = z_tmp.as_mat_mut();
    let mut tl_inv = math.re(zero());
    let mut tr_inv = math.re(zero());
    let mut a01 = math(zero());

    let mut A = A;
    let mut Hl = H_left;
    let mut Hr = H_right;

    for kj in n.indices() {
        write1!(Hl[kj] = math(infinity()));
        write1!(Hr[kj] = math(infinity()));
    }

    for kj in n.indices() {
        let ki = m.idx(*kj);

        let A = A.rb_mut();
        ghost_tree!(ROWS(TOP, I, BOT), COLS(LEFT, J, RIGHT(J1, NEXT)), {
            let (row_split @ l![top, _, _], (rows_x, ..)) = m.split(l![..ki.into(), ki, ..], ROWS);
            let (col_split @ l![left, _, right], (cols_x, _, l![_, _, RIGHT])) =
                n.split(l![top, kj, ..], COLS);

            let l![A0, A1, A2] = A.row_segments_mut(row_split, rows_x);

            let l![_, mut A01, A02] = A0.col_segments_mut(col_split, cols_x);
            let l![A10, mut a11, mut A12] = A1.col_segments_mut(col_split, cols_x);
            let l![A20, mut A21, mut A22] = A2.col_segments_mut(col_split, cols_x);

            let l![_, y1, mut y2] = y.rb_mut().col_segments_mut(col_split, cols_x);
            let l![_, z1, mut z2] = z.rb_mut().row_segments_mut(row_split, rows_x);
            let l![_, _, mut z_tmp] = z_tmp.rb_mut().row_segments_mut(row_split, rows_x);
            _ = &mut z_tmp;

            let (u2_p, v2_p) = if *kj > 0 {
                let p = left.local(left.idx(*kj - 1));
                let u1 = A10.rb().at(p);
                let u2 = A20.rb().col(p);

                let mut v1 = A01.rb_mut().at_mut(p);
                let v2 = A02.rb().row(p);

                let f0 = math(mul_real(y1, tl_inv));
                let f1 = math(mul_real(v1, tr_inv));

                write1!(a11, math(a11 - f0 * u1 - f1 * z1));
                z!(A21.rb_mut(), u2, z2.rb()).for_each(|uz!(mut a, u, z)| {
                    write1!(a, math(a - f0 * u - f1 * z));
                });

                let f0 = math(mul_real(u1, tl_inv));
                let f1 = math(mul_real(z1, tr_inv));
                z!(A12.rb_mut(), y2.rb(), v2)
                    .for_each(|uz!(mut a, y, v)| write1!(a, math(a - f0 * y - f1 * v)));

                write1!(v1, math.copy(a01));
                (Some(u2), Some(v2))
            } else {
                (None, None)
            };

            let tl_inv_prev = tl_inv;
            let norm = A21.norm_l2_with(ctx);
            let (tl, beta) = householder::make_householder_in_place(
                ctx,
                Some(A21.rb_mut()),
                rb!(a11),
                as_ref2!(norm),
            );
            tl_inv = math(re.recip(real(tl)));
            write1!(a11, beta);
            write1!(Hl[kj] = tl);

            if kj.next() == n.end() {
                break;
            }

            let u2 = A21.rb();

            bidiag_fused_op(
                ctx,
                A12.rb_mut(),
                A22.rb_mut(),
                z2.rb_mut(),
                y2.rb_mut(),
                u2,
                u2_p,
                v2_p,
                &tl_inv_prev,
                &tl_inv,
                &tr_inv,
                par,
            );

            let kj1 = right.global(right.idx(*kj + 1));

            let (l![j1, next], (rows_x2, ..)) = right.len().split(l![right.local(kj1), ..], RIGHT);
            let l![mut a12, mut A12] = A12.rb_mut().col_segments_mut(l![j1, next], rows_x2);
            let l![a22, _] = A22.rb().col_segments(l![j1, next]);

            let (tr, beta) = householder::make_householder_in_place(
                ctx,
                None,
                rb!(a12),
                as_ref2!(A12.norm_l2_with(ctx)),
            );
            tr_inv = math(re.recip(real(tr)));
            a01 = beta;
            write1!(Hr[kj1.local()] = tr);

            let diff = math(a12 - a01);

            if !math.is_zero(diff) {
                let f = math.recip(diff);
                z!(A12.rb_mut()).for_each(|uz!(mut a)| write1!(a, math(a * f)));
                z!(z2.rb_mut(), a22.rb()).for_each(|uz!(mut z, a)| write1!(z, math(z - a01 * a)));
            }
            write1!(a12, math(one()));

            let l![yy1, yy2] = y2.rb().col_segments(l![j1, next]);
            let b =
                math(yy1 + dot::inner_prod(ctx, yy2, Conj::No, A12.rb().transpose(), Conj::Yes));

            let f = math(mul_real(-b, tl_inv));
            z!(z2.rb_mut(), u2).for_each(|uz!(mut z, u)| write1!(z, math(z + u * f)));
        });
    }
}

#[math]
fn bidiag_fused_op<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A12: RowMut<'_, C, T, Dim<'N>>,
    A22: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    z2: ColMut<'_, C, T, Dim<'M>>,
    y2: RowMut<'_, C, T, Dim<'N>>,
    u2: ColRef<'_, C, T, Dim<'M>>,
    u2_p: Option<ColRef<'_, C, T, Dim<'M>>>,
    v2_p: Option<RowRef<'_, C, T, Dim<'N>>>,

    tl_inv_prev: &RealValue<C, T>,
    tl_inv: &RealValue<C, T>,
    tr_inv: &RealValue<C, T>,

    par: Par,
) {
    help!(C);

    let mut A12 = A12;
    let mut A22 = A22;
    let mut z2 = z2;
    let mut y2 = y2;

    if let (Some(u2_p), Some(v2_p)) = (u2_p, v2_p) {
        matmul(
            ctx,
            A22.rb_mut(),
            Accum::Add,
            u2_p.as_mat(),
            y2.rb().as_mat(),
            math.from_real(math.re(-tl_inv_prev)),
            par,
        );
        matmul(
            ctx,
            A22.rb_mut(),
            Accum::Add,
            z2.rb().as_mat(),
            v2_p.as_mat(),
            math.from_real(math.re(-tr_inv)),
            par,
        );
    }

    matmul(
        ctx,
        y2.rb_mut().as_mat_mut(),
        Accum::Replace,
        u2.adjoint().as_mat(),
        A22.rb(),
        math(one()),
        par,
    );

    z!(y2.rb_mut(), A12.rb_mut()).for_each(|uz!(mut y, mut a)| {
        write1!(y, math(y + a));
        write1!(a, math(a - mul_real(y, tl_inv)));
    });

    matmul(
        ctx,
        z2.rb_mut().as_mat_mut(),
        Accum::Replace,
        A22.rb(),
        A12.rb().adjoint().as_mat(),
        math(id(one())),
        par,
    );
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use dyn_stack::GlobalMemBuffer;
    use faer_traits::Unit;

    use super::*;
    use crate::{assert, stats::prelude::*, utils::approx::*, Mat, Row};

    #[test]
    fn test_bidiag_real() {
        let rng = &mut StdRng::seed_from_u64(0);

        with_dim!(m, 4);
        with_dim!(n, 4);

        let A = CwiseMatDistribution {
            nrows: m,
            ncols: n,
            dist: StandardNormal,
        }
        .rand::<Mat<f64, _, _>>(rng);

        let mut Hl = Row::zeros_with(&ctx(), n);
        let mut Hr = Row::zeros_with(&ctx(), n);

        let mut UV = A.clone();
        let mut UV = UV.as_mut();
        bidiag_in_place(
            &ctx(),
            UV.rb_mut(),
            Hl.as_mut(),
            Hr.as_mut(),
            Par::Seq,
            DynStack::new(&mut [MaybeUninit::uninit(); 1024]),
        );
        let UV = UV.rb();

        let mut A = A.clone();
        let mut A = A.as_mut();

        householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            &ctx(),
            UV,
            Hl.as_mat(),
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

        ghost_tree!(COLS(J0, RIGHT), {
            let (col_split @ l![_, bot], (col_x, ..)) = n.split(l![n.idx(0), ..], COLS);
            let l![_, mut A1] = A.rb_mut().col_segments_mut(col_split, col_x);
            let l![_, V] = UV.col_segments(col_split);
            let V = V.subrows(zero(), bot.len());
            let l![_, Hr] = Hr.as_ref().col_segments(col_split);

            householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
                &ctx(),
                V.transpose(),
                Hr.as_mat(),
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

        __dbg!(&A, &Hl, &Hr);
    }
}
