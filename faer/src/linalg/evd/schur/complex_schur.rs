// adapted from <T>LAPACK implementation
//
// https://github.com/tlapack/tlapack
// https://github.com/tlapack/tlapack/blob/master/include/tlapack/lapack/lahqr.hpp

use super::super::*;
use crate::{assert, debug_assert};
use linalg::{householder::*, jacobi::JacobiRotation, matmul::matmul};

#[faer_macros::migrate]
fn lahqr_shiftcolumn<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    h: MatRef<'_, C, T>,
    v: ColMut<'_, C, T>,
    s1: C::Of<T>,
    s2: C::Of<T>,
) {
    help!(C);
    let mut v = v;

    let n = h.nrows();

    if n == 2 {
        let s = math.re(cx.abs1(h.read(0, 0).faer_sub(s2)) + cx.abs1(h.read(1, 0)));
        if math.re(s == zero()) {
            v.write(0, math.zero());
            v.write(1, math.zero());
        } else {
            let s_inv = math.re.recip(s);
            let h10s = h.read(1, 0).faer_scale_real(s_inv);
            v.write(
                0,
                (h10s.faer_mul(h.read(0, 1))).faer_add(
                    h.read(0, 0)
                        .faer_sub(s1)
                        .faer_mul((h.read(0, 0).faer_sub(s2)).faer_scale_real(s_inv)),
                ),
            );
            v.write(
                1,
                h10s.faer_mul(
                    h.read(0, 0)
                        .faer_add(h.read(1, 1))
                        .faer_sub(s1)
                        .faer_sub(s2),
                ),
            );
        }
    } else {
        let s = math
            .re(cx.abs1(h.read(0, 0).faer_sub(s2)) + cx.abs1(h.read(1, 0)) + cx.abs1(h.read(2, 0)));
        if math.re(s == zero()) {
            v.write(0, math.zero());
            v.write(1, math.zero());
            v.write(2, math.zero());
        } else {
            let s_inv = math.re.recip(s);
            let h10s = h.read(1, 0).faer_scale_real(s_inv);
            let h20s = h.read(2, 0).faer_scale_real(s_inv);
            v.write(
                0,
                ((h.read(0, 0).faer_sub(s1))
                    .faer_mul((h.read(0, 0).faer_sub(s2)).faer_scale_real(s_inv)))
                .faer_add(h.read(0, 1).faer_mul(h10s))
                .faer_add(h.read(0, 2).faer_mul(h20s)),
            );
            v.write(
                1,
                (h10s.faer_mul(
                    h.read(0, 0)
                        .faer_add(h.read(1, 1).faer_sub(s1).faer_sub(s2)),
                ))
                .faer_add(h.read(1, 2).faer_mul(h20s)),
            );
            v.write(
                2,
                (h20s.faer_mul(
                    h.read(0, 0)
                        .faer_add(h.read(2, 2).faer_sub(s1).faer_sub(s2)),
                ))
                .faer_add(h10s.faer_mul(h.read(2, 1))),
            );
        }
    }
}

#[math]
fn lahqr_eig22<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    a00: C::Of<T>,
    a01: C::Of<T>,
    a10: C::Of<T>,
    a11: C::Of<T>,
) -> (C::Of<T>, C::Of<T>) {
    let zero = math.re.zero();

    let s = math.re(cx.abs1(a00) + cx.abs1(a01) + cx.abs1(a10) + cx.abs1(a11));
    if math.re(s == zero) {
        return math((zero(), zero()));
    }

    let half = math.re.from_f64(0.5);
    let s_inv = math.re.recip(s);
    let a00 = math.mul_real(a00, s_inv);
    let a01 = math.mul_real(a01, s_inv);
    let a10 = math.mul_real(a10, s_inv);
    let a11 = math.mul_real(a11, s_inv);

    let tr = math(mul_real(a00 + a11, half));
    let det = math((a00 - tr) * (a00 - tr) + (a01 * a10));

    let rtdisc = math.sqrt(det);
    math((mul_real(tr + rtdisc, s), mul_real(tr - rtdisc, s)))
}

#[math]
fn lahqr<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    want_t: bool,
    A: MatMut<'_, C, T>,
    Z: Option<MatMut<'_, C, T>>,
    w: ColMut<'_, C, T>,
    ilo: usize,
    ihi: usize,
) -> isize {
    help!(C);
    let n = A.nrows();
    let nh = ihi - ilo;

    let mut A = A;
    let mut Z = Z;
    let mut w = w;

    let zero = math.re.zero();
    let eps = math.eps();
    let smlnum = math.re(min_positive() / eps);
    let non_convergence_limit = 10usize;
    let dat1 = math.re.from_f64(0.75);
    let dat2 = math.re.from_f64(-0.4375);

    if nh == 0 {
        return 0;
    }
    if nh == 1 {
        write1!(w[ilo] = math(A[(ilo, ilo)]));
    }

    let itmax = Ord::max(30, ctx.nbits() / 2)
        .saturating_mul(Ord::max(10, nh))
        .saturating_mul(nh);

    // k_defl counts the number of iterations since a deflation
    let mut k_defl = 0usize;

    // istop is the end of the active subblock.
    // As more and more eigenvalues converge, it eventually
    // becomes ilo+1 and the loop ends.
    let mut istop = ihi;

    // istart is the start of the active subblock. Either
    // istart = ilo, or H(istart, istart-1) = 0. This means
    // that we can treat this subblock separately.
    let mut istart = ilo;

    for iter in 0..itmax {
        if iter + 1 == itmax {
            return istop as isize;
        }

        if istart + 1 >= istop {
            if istart + 1 == istop {
                write1!(w[istart] = math(A[(istart, istart)]));
            }
            break;
        }

        let istart_m;
        let istop_m;

        if want_t {
            istart_m = 0;
            istop_m = n;
        } else {
            istart_m = istart;
            istop_m = istop;
        }

        for i in (istart + 1..istop).rev() {
            if math(abs1(A[(i, i - 1)]) < smlnum) {
                write1!(A[(i, i - 1)] = math.zero());
                istart = i;
                break;
            }

            let mut tst = math.re(cx.abs1(A[(i - 1, i - 1)]) + cx.abs1(A[(i, i)]));

            if math.re(tst == zero) {
                if i >= ilo + 2 {
                    tst = math.re(tst + cx.abs1(A[(i - 1, i - 2)]));
                }
                if i + 1 < ihi {
                    tst = math.re(tst + cx.abs(A[(i + 1, i)]));
                }
            }

            if math.re(cx.abs1(A[(i, i - 1)]) <= eps * tst) {
                // The elementwise deflation test has passed
                // The following performs second deflation test due
                // to Ahues & Tisseur (LAWN 122, 1997). It has better
                // mathematical foundation and improves accuracy in some
                // examples.
                //
                // The test is |A(i,i-1)|*|A(i-1,i)| <=
                // eps*|A(i,i)|*|A(i-1,i-1)| The multiplications might overflow
                // so we do some scaling first.

                let ab = math(max(abs1(A[(i, i - 1)]), abs1(A[(i - 1, i)])));
                let ba = math(min(abs1(A[(i, i - 1)]), abs1(A[(i - 1, i)])));
                let aa = math(max(abs1(A[(i, i)]), abs1(A[(i, i)] - A[(i - 1, i - 1)])));
                let bb = math(min(abs1(A[(i, i)]), abs1(A[(i, i)] - A[(i - 1, i - 1)])));
                let s = math.re(aa + ab);
                if math.re(ba * (ab / s) <= max(smlnum, eps * (bb * (aa / s)))) {
                    write1!(A[(i, i - 1)] = math.zero());
                    istart = i;
                    break;
                }
            }
        }

        if istart + 1 >= istop {
            k_defl = 0;
            write1!(w[istart] = math(A[(istart, istart)]));
            istop = istart;
            istart = ilo;
            continue;
        }

        // determine shift
        let (a00, a01, a10, a11);
        k_defl += 1;

        if k_defl % non_convergence_limit == 0 {
            // exceptional shift
            let mut s = math(abs(A[(istop - 1, istop - 2)]));

            if istop > ilo + 2 {
                s = math.re(s + cx.abs(A[(istop - 2, istop - 3)]));
            }

            a00 = math(from_real(re.mul(dat1, s)), A[(istop - 1, istop - 1)]);
            a10 = math(from_real(re.mul(dat2, s)));
            a01 = math(from_real(s));
            a11 = math(copy(a00));
        } else {
            // wilkinson shift
            a00 = math(A[(istop - 2, istop - 2)]);
            a10 = math(A[(istop - 1, istop - 2)]);
            a01 = math(A[(istop - 2, istop - 1)]);
            a11 = math(A[(istop - 1, istop - 1)]);
        }

        let (mut s1, s2) = lahqr_eig22(ctx, a00, a01, a10, a11);
        if math(abs1(s1 - A[(istop - 1, istop - 1)]) > abs1(s2 - A[(istop - 1, istop - 1)])) {
            s1 = math.copy(s2);
        }

        // We have already checked whether the subblock has split.
        // If it has split, we can introduce any shift at the top of the new
        // subblock. Now that we know the specific shift, we can also check
        // whether we can introduce that shift somewhere else in the subblock.
        let mut istart2 = istart;
        if istart + 2 < istop {
            for i in (istart + 1..istop - 1).rev() {
                let h00 = math(A[(i, i)] - (s1));
                let h10 = math(A[(i + 1, i)]);

                let JacobiRotation { c: _, s: sn } = JacobiRotation::rotg(ctx, h00, h10).0;
                if math(
                    abs1(conj(sn) * A[(i, i - 1)])
                        <= re.mul(
                            eps, //
                            re.add(abs1(A[(i, i - 1)]), abs1(A[(i, i + 1)])),
                        ),
                ) {
                    istart2 = i;
                    break;
                }
            }
        }
        for i in istart2..istop - 1 {
            let (rot, r);

            if i == istart2 {
                let h00 = math(A[(i, i)] - s1);
                let h10 = math(A[(i + 1, i)]);

                (rot, r) = JacobiRotation::rotg(ctx, h00, h10);
                if i > istart {
                    write1!(A[(i, i - 1)] = math(A[(i, i - 1)] * rot.c));
                }
            } else {
                (rot, r) = math(JacobiRotation::rotg(ctx, A[(i, i - 1)], A[(i + 1, i - 1)]));

                write1!(A[(i, i - 1)] = math.copy(r));
                write1!(A[(i + 1, i - 1)] = math.zero());
            }
            drop(r);

            rot.adjoint(ctx).apply_on_the_left_in_place(
                ctx,
                A.rb_mut()
                    .subcols_mut(i, istop_m - i)
                    .two_rows_mut(i, i + 1),
            );

            rot.apply_on_the_right_in_place(
                ctx,
                A.rb_mut()
                    .subrows_range_mut((istart_m, Ord::min(i + 3, istop)))
                    .two_cols_mut(i, i + 1),
            );

            if let Some(Z) = Z.rb_mut() {
                rot.apply_on_the_right_in_place(ctx, Z.two_cols_mut(i, i + 1));
            }
        }
    }

    0
}

#[faer_macros::migrate]
fn aggressive_early_deflation<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    want_t: bool,
    mut a: MatMut<'_, C, T>,
    mut z: Option<MatMut<'_, C, T>>,
    mut s: ColMut<'_, C, T>,
    ilo: usize,
    ihi: usize,
    nw: usize,
    par: Par,
    stack: &mut DynStack,
    params: EvdParams,
) -> (usize, usize) {
    let n = a.nrows();

    // Because we will use the lower triangular part of A as workspace,
    // We have a maximum window size
    let nw_max = (n - 3) / 3;
    let eps = math.eps();
    let small_num = math.re(min_positive() / eps * from_f64(n as f64));

    // Size of the deflation window
    let jw = Ord::min(Ord::min(nw, ihi - ilo), nw_max);
    // First row index in the deflation window
    let kwtop = ihi - jw;

    // s is the value just outside the window. It determines the spike
    // together with the orthogonal schur factors.
    let mut s_spike = if kwtop == ilo {
        math.zero()
    } else {
        a.read(kwtop, kwtop - 1)
    };

    if kwtop + 1 == ihi {
        // 1x1 deflation window, not much to do
        s.write(kwtop, a.read(kwtop, kwtop));
        let mut ns = 1;
        let mut nd = 0;
        if math.re(cx.abs1(s_spike) <= max(small_num, eps * cx.abs1(a.read(kwtop, kwtop)))) {
            ns = 0;
            nd = 1;
            if kwtop > ilo {
                a.write(kwtop, kwtop - 1, math.zero());
            }
        }
        return (ns, nd);
    }

    // Define workspace matrices
    // We use the lower triangular part of A as workspace
    // TW and WH overlap, but WH is only used after we no longer need
    // TW so it is ok.
    let mut v = unsafe { a.rb().submatrix(n - jw, 0, jw, jw).const_cast() };
    let mut tw = unsafe { a.rb().submatrix(n - jw, jw, jw, jw).const_cast() };
    let mut wh = unsafe {
        a.rb()
            .submatrix(n - jw, jw, jw, n - 2 * jw - 3)
            .const_cast()
    };
    let mut wv = unsafe { a.rb().submatrix(jw + 3, 0, n - 2 * jw - 3, jw).const_cast() };
    let mut a = unsafe { a.rb().const_cast() };

    // Convert the window to spike-triangular form. i.e. calculate the
    // Schur form of the deflation window.
    // If the QR algorithm fails to convergence, it can still be
    // partially in Schur form. In that case we continue on a smaller
    // window (note the use of infqr later in the code).
    let a_window = a.rb().submatrix(kwtop, kwtop, ihi - kwtop, ihi - kwtop);
    let mut s_window = unsafe { s.rb().subrows(kwtop, ihi - kwtop).const_cast() };
    tw.fill(math.zero());
    for j in 0..jw {
        for i in 0..Ord::min(j + 2, jw) {
            tw.write(i, j, a_window.read(i, j));
        }
    }
    v.fill(math.zero());
    v.rb_mut().diagonal_mut().fill(math.one());

    let infqr = if jw
        < params
            .blocking_threshold
            .unwrap_or(default_blocking_threshold())
    {
        lahqr(
            ctx,
            true,
            tw.rb_mut(),
            Some(v.rb_mut()),
            s_window.rb_mut(),
            0,
            jw,
        )
    } else {
        let infqr = multishift_qr(
            ctx,
            true,
            tw.rb_mut(),
            Some(v.rb_mut()),
            s_window.rb_mut(),
            0,
            jw,
            par,
            stack,
            params,
        )
        .0;
        for j in 0..jw {
            for i in j + 2..jw {
                tw.write(i, j, math.zero());
            }
        }
        infqr
    };
    let infqr = infqr as usize;

    // Deflation detection loop
    // one eigenvalue block at a time, we will check if it is deflatable
    // by checking the bottom spike element. If it is not deflatable,
    // we move the block up. This moves other blocks down to check.
    let mut ns = jw;
    let nd;
    let mut ilst = infqr;
    while ilst < ns {
        // 1x1 eigenvalue block
        #[allow(clippy::disallowed_names)]
        let mut foo = math(abs1(tw[(ns - 1, ns - 1)]));
        if math.re(foo == zero()) {
            foo = math.abs1(s_spike);
        }
        if math.re(cx.abs1(s_spike) * cx.abs1(v[(0, ns - 1)]) <= max(small_num, eps * foo)) {
            // Eigenvalue is deflatable
            ns -= 1;
        } else {
            // Eigenvalue is not deflatable.
            // Move it up out of the way.
            let ifst = ns - 1;
            schur_move(ctx, tw.rb_mut(), Some(v.rb_mut()), ifst, &mut ilst);
            ilst += 1;
        }
    }

    if ns == 0 {
        s_spike = math.zero();
    }

    if ns == jw {
        // Aggressive early deflation didn't deflate any eigenvalues
        // We don't need to apply the update to the rest of the matrix
        nd = jw - ns;
        ns -= infqr;

        return (ns, nd);
    }

    // sorting diagonal blocks of T improves accuracy for graded matrices.
    // Bubble sort deals well with exchange failures.
    let mut sorted = false;
    // Window to be checked (other eigenvalue are sorted)
    let mut sorting_window_size = jw;
    while !sorted {
        sorted = true;

        // Index of last eigenvalue that was swapped
        let mut ilst = 0;

        // Index of the first block
        let mut i1 = ns;

        while i1 + 1 < sorting_window_size {
            // Check if there is a next block
            if i1 + 1 == jw {
                ilst -= 1;
                break;
            }

            // Index of the second block
            let i2 = i1 + 1;

            // Size of the second block

            let ev1 = math.abs1(tw.read(i1, i1));
            let ev2 = math.abs1(tw.read(i2, i2));

            if math(ev1 > ev2) {
                i1 = i2;
            } else {
                sorted = false;
                let ierr = schur_swap(ctx, tw.rb_mut(), Some(v.rb_mut()), i1);
                if ierr == 0 {
                    i1 += 1;
                } else {
                    i1 = i2;
                }
                ilst = i1;
            }
        }
        sorting_window_size = ilst;
    }

    // Recalculate the eigenvalues
    let mut i = 0;
    while i < jw {
        s.write(kwtop + i, tw.read(i, i));
        i += 1;
    }
    help!(C);

    // Reduce A back to Hessenberg form (if necessary)
    if math(s_spike != zero()) {
        // Reflect spike back
        {
            let mut vv = wv.rb_mut().col_mut(0).subrows_mut(0, ns);
            for i in 0..ns {
                vv.write(i, v.read(0, i).faer_conj());
            }
            let mut head = vv.read(0);
            let tail = vv.rb_mut().subrows_mut(1, ns - 1);
            let (tau, _) = make_householder_in_place(ctx, as_mut!(head), tail);
            vv.write(0, math.one());
            let tau = tau.faer_inv();

            {
                let mut tw_slice = tw.rb_mut().submatrix_mut(0, 0, ns, jw);
                let tmp = vv.rb().adjoint() * tw_slice.rb();
                matmul(
                    ctx,
                    tw_slice.rb_mut(),
                    Accum::Add,
                    vv.rb().as_mat(),
                    tmp.as_ref().as_mat(),
                    tau.faer_neg(),
                    par,
                );
            }

            {
                let mut tw_slice2 = tw.rb_mut().submatrix_mut(0, 0, jw, ns);
                let tmp = tw_slice2.rb() * vv.rb();
                matmul(
                    ctx,
                    tw_slice2.rb_mut(),
                    Accum::Add,
                    tmp.as_ref().as_mat(),
                    vv.rb().adjoint().as_mat(),
                    tau.faer_neg(),
                    par,
                );
            }

            {
                let mut v_slice = v.rb_mut().submatrix_mut(0, 0, jw, ns);
                let tmp = v_slice.rb() * vv.rb();
                matmul(
                    ctx,
                    v_slice.rb_mut(),
                    Accum::Add,
                    tmp.as_ref().as_mat(),
                    vv.rb().adjoint().as_mat(),
                    tau.faer_neg(),
                    par,
                );
            }
            vv.write(0, head);
        }

        // Hessenberg reduction
        {
            let mut householder = wv.rb_mut().col_mut(0).subrows_mut(0, ns);
            hessenberg::hessenberg_in_place(
                ctx,
                tw.rb_mut().submatrix_mut(0, 0, ns, ns),
                householder.rb_mut().as_mat_mut().transpose_mut(),
                par,
                stack,
                Default::default(),
            );

            let householder = wv.rb_mut().col_mut(0).subrows_mut(0, ns - 1);
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                ctx,
                tw.rb().submatrix(1, 0, ns - 1, ns - 1),
                householder.rb().transpose().as_mat(),
                Conj::Yes,
                unsafe { tw.rb().submatrix(1, ns, ns - 1, jw - ns).const_cast() },
                par,
                stack,
            );
            apply_block_householder_sequence_on_the_right_in_place_with_conj(
                ctx,
                tw.rb().submatrix(1, 0, ns - 1, ns - 1),
                householder.rb().transpose().as_mat(),
                Conj::No,
                v.rb_mut().submatrix_mut(0, 1, jw, ns - 1),
                par,
                stack,
            );
        }
    }

    // Copy the deflation window back into place
    if kwtop > 0 {
        a.write(kwtop, kwtop - 1, s_spike.faer_mul(v.read(0, 0).faer_conj()));
    }
    for j in 0..jw {
        for i in 0..Ord::min(j + 2, jw) {
            a.write(kwtop + i, kwtop + j, tw.read(i, j));
        }
    }

    // Store number of deflated eigenvalues
    nd = jw - ns;
    ns -= infqr;

    //
    // Update rest of the matrix using matrix matrix multiplication
    //
    let (istart_m, istop_m);
    if want_t {
        istart_m = 0;
        istop_m = n;
    } else {
        istart_m = ilo;
        istop_m = ihi;
    }

    // Horizontal multiply
    if ihi < istop_m {
        let mut i = ihi;
        while i < istop_m {
            let iblock = Ord::min(istop_m - i, wh.ncols());
            let mut a_slice = a.rb_mut().submatrix_mut(kwtop, i, ihi - kwtop, iblock);
            let mut wh_slice = wh
                .rb_mut()
                .submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
            matmul(
                ctx,
                wh_slice.rb_mut(),
                Accum::Replace,
                v.rb().adjoint(),
                a_slice.rb(),
                math.one(),
                par,
            );
            a_slice.copy_from_with(ctx, wh_slice.rb());
            i += iblock;
        }
    }

    // Vertical multiply
    if istart_m < kwtop {
        let mut i = istart_m;
        while i < kwtop {
            let iblock = Ord::min(kwtop - i, wv.nrows());
            let mut a_slice = a.rb_mut().submatrix_mut(i, kwtop, iblock, ihi - kwtop);
            let mut wv_slice = wv
                .rb_mut()
                .submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
            matmul(
                ctx,
                wv_slice.rb_mut(),
                Accum::Replace,
                a_slice.rb(),
                v.rb(),
                math.one(),
                par,
            );
            a_slice.copy_from_with(ctx, wv_slice.rb());
            i += iblock;
        }
    }
    // Update Z (also a vertical multiplication)
    if let Some(mut z) = z.rb_mut() {
        let mut i = 0;
        while i < n {
            let iblock = Ord::min(n - i, wv.nrows());
            let mut z_slice = z.rb_mut().submatrix_mut(i, kwtop, iblock, ihi - kwtop);
            let mut wv_slice = wv
                .rb_mut()
                .submatrix_mut(0, 0, z_slice.nrows(), z_slice.ncols());
            matmul(
                ctx,
                wv_slice.rb_mut(),
                Accum::Replace,
                z_slice.rb(),
                v.rb(),
                math.one(),
                par,
            );
            z_slice.copy_from_with(ctx, wv_slice.rb());
            i += iblock;
        }
    }

    (ns, nd)
}

fn schur_move<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    mut a: MatMut<'_, C, T>,
    mut q: Option<MatMut<'_, C, T>>,
    ifst: usize,
    ilst: &mut usize,
) -> isize {
    let n = a.nrows();

    // Quick return
    if n == 0 {
        return 0;
    }

    let mut here = ifst;
    if ifst < *ilst {
        while here != *ilst {
            // Size of the next eigenvalue block
            let ierr = schur_swap(ctx, a.rb_mut(), q.rb_mut(), here);
            if ierr != 0 {
                // The swap failed, return with error
                *ilst = here;
                return 1;
            }
            here += 1;
        }
    } else {
        while here != *ilst {
            // Size of the next eigenvalue block
            let ierr = schur_swap(ctx, a.rb_mut(), q.rb_mut(), here - 1);
            if ierr != 0 {
                // The swap failed, return with error
                *ilst = here;
                return 1;
            }
            here -= 1;
        }
    }

    0
}

#[faer_macros::migrate]
fn schur_swap<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    mut a: MatMut<'_, C, T>,
    q: Option<MatMut<'_, C, T>>,
    j0: usize,
) -> isize {
    let n = a.nrows();

    let j1 = j0 + 1;
    let j2 = j0 + 2;

    //
    // In the complex case, there can only be 1x1 blocks to swap
    //
    let t00 = a.read(j0, j0);
    let t11 = a.read(j1, j1);
    //
    // Determine the transformation to perform the interchange
    //
    let (rot, _) = JacobiRotation::<C, T>::rotg(ctx, a.read(j0, j1), t11.faer_sub(t00));

    a.write(j1, j1, t00);
    a.write(j0, j0, t11);

    // Apply transformation from the left
    if j2 < n {
        let row1 = unsafe { a.rb().row(j0).subcols(j2, n - j2).const_cast() };
        let row2 = unsafe { a.rb().row(j1).subcols(j2, n - j2).const_cast() };
        rot.adjoint(ctx)
            .apply_on_the_left_in_place(ctx, (row1, row2));
    }
    // Apply transformation from the right
    if j0 > 0 {
        let col1 = unsafe { a.rb().col(j0).subrows(0, j0).const_cast() };
        let col2 = unsafe { a.rb().col(j1).subrows(0, j0).const_cast() };

        rot.apply_on_the_right_in_place(ctx, (col1, col2));
    }
    if let Some(q) = q {
        let col1 = unsafe { q.rb().col(j0).const_cast() };
        let col2 = unsafe { q.rb().col(j1).const_cast() };
        rot.apply_on_the_right_in_place(ctx, (col1, col2));
    }

    0
}

pub fn multishift_qr_scratch<C: ComplexContainer, T: ComplexField<C>>(
    n: usize,
    nh: usize,
    want_t: bool,
    want_z: bool,
    par: Par,
    params: EvdParams,
) -> Result<StackReq, SizeOverflow> {
    let nsr = (params
        .recommended_shift_count
        .unwrap_or(default_recommended_shift_count))(n, nh);

    let _ = want_t;
    let _ = want_z;

    if n <= 3 {
        return Ok(StackReq::empty());
    }

    let nw_max = (n - 3) / 3;

    StackReq::try_any_of([
        hessenberg::hessenberg_in_place_scratch::<C, T>(nw_max, 1, par, Default::default())?,
        apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<C, T>(
            nw_max, nw_max, nw_max,
        )?,
        apply_block_householder_sequence_on_the_right_in_place_scratch::<C, T>(
            nw_max, nw_max, nw_max,
        )?,
        temp_mat_scratch::<C, T>(3, nsr)?,
    ])
}

/// returns err code, number of aggressive early deflations, number of qr sweeps
#[faer_macros::migrate]
pub fn multishift_qr<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    want_t: bool,
    a: MatMut<'_, C, T>,
    z: Option<MatMut<'_, C, T>>,
    w: ColMut<'_, C, T>,
    ilo: usize,
    ihi: usize,
    par: Par,
    stack: &mut DynStack,
    params: EvdParams,
) -> (isize, usize, usize) {
    assert!(a.nrows() == a.ncols());
    assert!(ilo <= ihi);

    let n = a.nrows();
    let nh = ihi - ilo;

    assert!(w.nrows() == n);
    assert!(w.ncols() == 1);

    if let Some(z) = z.rb() {
        assert!(z.nrows() == n);
        assert!(z.ncols() == n);
    }

    let mut a = a;
    let mut z = z;
    let mut w = w;
    let mut stack = stack;

    let non_convergence_limit_window = 5;
    let non_convergence_limit_shift = 6;
    let dat1 = math.re.from_f64(0.75);
    let dat2 = math.re.from_f64(-0.4375);

    // This routine uses the space below the subdiagonal as workspace
    // For small matrices, this is not enough
    // if n < nmin, the matrix will be passed to lahqr
    let nmin = Ord::max(
        15,
        params
            .blocking_threshold
            .unwrap_or(default_blocking_threshold()),
    );
    let nibble = params
        .nibble_threshold
        .unwrap_or(default_nibble_threshold());

    // Recommended number of shifts
    let nsr = (params
        .recommended_shift_count
        .unwrap_or(default_recommended_shift_count))(n, nh);
    let nsr = Ord::min(Ord::min(nsr, (n.saturating_sub(3)) / 6), ihi - ilo - 1);
    let nsr = Ord::max(nsr / 2 * 2, 2);

    // Recommended deflation window size
    let nwr = (params
        .recommended_deflation_window
        .unwrap_or(default_recommended_deflation_window))(n, nh);
    let nwr = Ord::max(nwr, 2);
    let nwr = Ord::min(Ord::min(nwr, (n.saturating_sub(1)) / 3), ihi - ilo);

    // Tiny matrices must use lahqr
    if n < nmin {
        let err = lahqr(ctx, want_t, a, z, w, ilo, ihi);
        return (err, 0, 0);
    }
    if nh == 0 {
        return (0, 0, 0);
    }

    let nw_max = (n - 3) / 3;

    // itmax is the total number of QR iterations allowed.
    // For most matrices, 3 shifts per eigenvalue is enough, so
    // we set itmax to 30 times nh as a safe limit.
    let itmax = 30 * Ord::max(10, nh);

    // k_defl counts the number of iterations since a deflation
    let mut k_defl = 0;

    // istop is the end of the active subblock.
    // As more and more eigenvalues converge, it eventually
    // becomes ilo+1 and the loop ends.
    let mut istop = ihi;

    let mut info = 0;
    let mut nw = 0;

    let mut count_aed = 0;
    let mut count_sweep = 0;

    help!(C::Real);
    let orig;
    if let Some(z) = z.rb() {
        let mut a = a.rb().cloned();

        for j in 0..n {
            for i in 0..n {
                if i > j + 1 {
                    a.as_mut().write(i, j, math.zero());
                }
            }
        }
        orig = &z * &a * z.adjoint();
    } else {
        panic!();
    }

    for iter in 0..itmax + 1 {
        if iter == itmax {
            // The QR algorithm failed to converge, return with error.
            info = istop as isize;
            break;
        }

        if ilo + 1 >= istop {
            if ilo + 1 == istop {
                w.write(ilo, a.read(ilo, ilo));
            }
            // All eigenvalues have been found, exit and return 0.
            break;
        }

        // istart is the start of the active subblock. Either
        // istart = ilo, or H(istart, istart-1) = 0. This means
        // that we can treat this subblock separately.
        let mut istart = ilo;

        // Find active block
        for i in (ilo + 1..istop).rev() {
            if math(a[(i, i - 1)] == zero()) {
                istart = i;
                break;
            }
        }

        //
        // Aggressive early deflation
        //
        let nh = istop - istart;
        let nwupbd = Ord::min(nh, nw_max);
        if k_defl < non_convergence_limit_window {
            nw = Ord::min(nwupbd, nwr);
        } else {
            // There have been no deflations in many iterations
            // Try to vary the deflation window size.
            nw = Ord::min(nwupbd, 2 * nw);
        }
        if nh <= 4 {
            // n >= nmin, so there is always enough space for a 4x4 window
            nw = nh;
        }
        if nw < nw_max {
            if nw + 1 >= nh {
                nw = nh
            };
            let kwtop = istop - nw;
            if (kwtop > istart + 2)
                && math(abs1(a[(kwtop, kwtop - 1)]) > abs1(a[(kwtop - 1, kwtop - 2)]))
            {
                nw += 1;
            }
        }

        let (ls, ld) = aggressive_early_deflation(
            ctx,
            want_t,
            a.rb_mut(),
            z.rb_mut(),
            w.rb_mut(),
            istart,
            istop,
            nw,
            par,
            stack.rb_mut(),
            params,
        );

        count_aed += 1;

        istop -= ld;

        if ld > 0 {
            k_defl = 0;
        }

        // Skip an expensive QR sweep if there is a (partly heuristic)
        // reason to expect that many eigenvalues will deflate without it.
        // Here, the QR sweep is skipped if many eigenvalues have just been
        // deflated or if the remaining active block is small.
        if ld > 0 && (100 * ld > nwr * nibble || (istop - istart) <= Ord::min(nmin, nw_max)) {
            continue;
        }

        k_defl += 1;
        let mut ns = Ord::min(nh - 1, Ord::min(Ord::max(2, ls), nsr));
        ns = ns / 2 * 2;
        let mut i_shifts = istop - ns;

        if k_defl % non_convergence_limit_shift == 0 {
            for i in (i_shifts + 1..istop).rev().step_by(2) {
                if i >= ilo + 2 {
                    let ss = math.re(cx.abs1(a[(i, i - 1)]) + cx.abs1(a[(i - 1, i - 2)]));
                    let aa = math.from_real(math.re(dat1 * ss)).faer_add(a.read(i, i));
                    let bb = math.from_real(ss);
                    let cc = math.from_real(math.re(dat2 * ss));
                    let dd = math.copy(aa);
                    let (s1, s2) = lahqr_eig22(ctx, aa, bb, cc, dd);
                    w.write(i - 1, s1);
                    w.write(i, s2);
                } else {
                    w.write(i - 1, a.read(i, i));
                    w.write(i, a.read(i, i));
                }
            }
        } else {
            if ls <= ns / 2 {
                // Got ns/2 or fewer shifts? Then use multi/double shift qr to
                // get more
                let mut temp = a.rb_mut().submatrix_mut(n - ns, 0, ns, ns);
                let mut shifts = w.rb_mut().subrows_mut(istop - ns, ns);
                let ierr = lahqr(ctx, false, temp.rb_mut(), None, shifts.rb_mut(), 0, ns) as usize;

                ns = ns - ierr;

                if ns < 2 {
                    // In case of a rare QR failure, use eigenvalues
                    // of the trailing 2x2 submatrix
                    let aa = a.read(istop - 2, istop - 2);
                    let bb = a.read(istop - 2, istop - 1);
                    let cc = a.read(istop - 1, istop - 2);
                    let dd = a.read(istop - 1, istop - 1);
                    let (s1, s2) = lahqr_eig22(ctx, aa, bb, cc, dd);
                    w.write(istop - 2, s1);
                    w.write(istop - 1, s2);
                    ns = 2;
                }

                i_shifts = istop - ns;
            }

            // Sort the shifts (helps a little)
            // Bubble sort keeps complex conjugate pairs together
            let mut sorted = false;
            let mut k = istop;
            while !sorted && k > i_shifts {
                sorted = true;
                for i in i_shifts..k - 1 {
                    if math(abs1(w[i]) < abs1(w[i + 1])) {
                        sorted = false;
                        let wi = w.read(i);
                        let wip1 = w.read(i + 1);
                        w.write(i, wip1);
                        w.write(i + 1, wi);
                    }
                }
                k -= 1;
            }

            // Shuffle shifts into pairs of real shifts
            // and pairs of complex conjugate shifts
            // assuming complex conjugate shifts are
            // already adjacent to one another. (Yes,
            // they are.)
            for i in (i_shifts + 2..istop).rev().step_by(2) {
                if math.re(cx.imag(w[i]) != -cx.imag(w[i - 1])) {
                    let tmp = w.read(i);
                    w.write(i, w.read(i - 1));
                    w.write(i - 1, w.read(i - 2));
                    w.write(i - 2, tmp);
                }
            }

            // Since we shuffled the shifts, we will only drop
            // Real shifts
            if ns % 2 == 1 {
                ns -= 1;
            }
            i_shifts = istop - ns;
        }

        let mut shifts = w.rb_mut().subrows_mut(i_shifts, ns);

        multishift_qr_sweep(
            ctx,
            want_t,
            a.rb_mut(),
            z.rb_mut(),
            shifts.rb_mut(),
            istart,
            istop,
            par,
            stack,
        );

        count_sweep += 1;
    }

    (info, count_aed, count_sweep)
}

#[faer_macros::migrate]
fn move_bulge<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    mut h: MatMut<'_, C, T>,
    mut v: ColMut<'_, C, T>,
    s1: C::Of<T>,
    s2: C::Of<T>,
) {
    help!(C);

    // Perform delayed update of row below the bulge
    // Assumes the first two elements of the row are zero
    let v0 = math.real(v.read(0));
    let v1 = v.read(1);
    let v2 = v.read(2);
    let refsum = v2.faer_scale_real(v0).faer_mul(h.read(3, 2));

    let epsilon = math.eps();

    h.write(3, 0, refsum.faer_neg());
    h.write(3, 1, refsum.faer_neg().faer_mul(v1.faer_conj()));
    h.write(3, 2, h.read(3, 2).faer_sub(refsum.faer_mul(v2.faer_conj())));

    // Generate reflector to move bulge down
    v.write(0, h.read(1, 0));
    v.write(1, h.read(2, 0));
    v.write(2, h.read(3, 0));

    let mut beta = v.read(0);
    let tail = v.rb_mut().subrows_mut(1, 2);
    let (tau, _) = make_householder_in_place(ctx, as_mut!(beta), tail);
    v.write(0, tau.faer_inv());

    // Check for bulge collapse
    if math(h[(3, 0)] != zero() || h[(3, 1)] != zero() || h[(3, 2)] != zero()) {
        // The bulge hasn't collapsed, typical case
        h.write(1, 0, beta);
        h.write(2, 0, math.zero());
        h.write(3, 0, math.zero());
    } else {
        // The bulge has collapsed, attempt to reintroduce using
        // 2-small-subdiagonals trick
        stack_mat!(ctx, vt, 3, 1, C, T);
        let mut vt = vt.rb_mut().col_mut(0);

        let h2 = h.rb().submatrix(1, 1, 3, 3);
        lahqr_shiftcolumn(ctx, h2, vt.rb_mut(), s1, s2);

        let mut beta_unused = vt.read(0);
        let tail = vt.rb_mut().subrows_mut(1, 2);
        let (tau, _) = make_householder_in_place(ctx, as_mut!(beta_unused), tail);
        vt.write(0, tau.faer_inv());
        let vt0 = vt.read(0);
        let vt1 = vt.read(1);
        let vt2 = vt.read(2);

        let refsum = (vt0.faer_conj().faer_mul(h.read(1, 0)))
            .faer_add(vt1.faer_conj().faer_mul(h.read(2, 0)));

        if math.re(cx.abs1(
            //
            cx.sub(h[(2, 0)], cx.mul(refsum, vt1)),
        ) + cx.abs1(cx.mul(refsum, vt2))
            > epsilon * (cx.abs1(h[(0, 0)]) + cx.abs1(h[(1, 1)]) + cx.abs1(h[(2, 2)])))
        {
            // Starting a new bulge here would create non-negligible fill. Use
            // the old one.
            h.write(1, 0, beta);
            h.write(2, 0, math.zero());
            h.write(3, 0, math.zero());
        } else {
            // Fill-in is negligible, use the new reflector.
            h.write(1, 0, h.read(1, 0).faer_sub(refsum));
            h.write(2, 0, math.zero());
            h.write(3, 0, math.zero());
            v.write(0, vt.read(0));
            v.write(1, vt.read(1));
            v.write(2, vt.read(2));
        }
    }
}

#[faer_macros::migrate]
fn multishift_qr_sweep<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    want_t: bool,
    a: MatMut<'_, C, T>,
    mut z: Option<MatMut<'_, C, T>>,
    s: ColMut<'_, C, T>,
    ilo: usize,
    ihi: usize,
    par: Par,
    stack: &mut DynStack,
) {
    let n = a.nrows();

    assert!(n >= 12);

    let (mut v, _stack) = temp_mat_zeroed(ctx, 3, s.nrows() / 2, stack);
    let mut v = v.as_mat_mut();

    let n_block_max = (n - 3) / 3;
    let n_shifts_max = Ord::min(ihi - ilo - 1, Ord::max(2, 3 * (n_block_max / 4)));

    let mut n_shifts = Ord::min(s.nrows(), n_shifts_max);
    if n_shifts % 2 == 1 {
        n_shifts -= 1;
    }
    let n_bulges = n_shifts / 2;

    let n_block_desired = Ord::min(2 * n_shifts, n_block_max);

    // Define workspace matrices
    // We use the lower triangular part of A as workspace

    // U stores the orthogonal transformations
    let mut u = unsafe {
        a.rb()
            .submatrix(n - n_block_desired, 0, n_block_desired, n_block_desired)
            .const_cast()
    };
    // Workspace for horizontal multiplications
    let mut wh = unsafe {
        a.rb()
            .submatrix(
                n - n_block_desired,
                n_block_desired,
                n_block_desired,
                n - 2 * n_block_desired - 3,
            )
            .const_cast()
    };
    // Workspace for vertical multiplications
    let mut wv = unsafe {
        a.rb()
            .submatrix(
                n_block_desired + 3,
                0,
                n - 2 * n_block_desired - 3,
                n_block_desired,
            )
            .const_cast()
    };
    let mut a = unsafe { a.rb().const_cast() };

    // i_pos_block points to the start of the block of bulges
    let mut i_pos_block = 0;

    help!(C::Real);
    let orig;
    if let Some(z) = z.rb() {
        let a = a.rb().cloned();
        orig = &z * &a * z.adjoint();
    } else {
        panic!();
    }
    introduce_bulges(
        ctx,
        ilo,
        ihi,
        n_block_desired,
        n_bulges,
        n_shifts,
        want_t,
        a.rb_mut(),
        z.rb_mut(),
        u.rb_mut(),
        v.rb_mut(),
        wh.rb_mut(),
        wv.rb_mut(),
        s.rb(),
        &mut i_pos_block,
        par,
    );
    move_bulges_down(
        ctx,
        ilo,
        ihi,
        n_block_desired,
        n_bulges,
        n_shifts,
        want_t,
        a.rb_mut(),
        z.rb_mut(),
        u.rb_mut(),
        v.rb_mut(),
        wh.rb_mut(),
        wv.rb_mut(),
        s.rb(),
        &mut i_pos_block,
        par,
    );
    remove_bulges(
        ctx,
        ilo,
        ihi,
        n_bulges,
        n_shifts,
        want_t,
        a.rb_mut(),
        z.rb_mut(),
        u.rb_mut(),
        v.rb_mut(),
        wh.rb_mut(),
        wv.rb_mut(),
        s.rb(),
        &mut i_pos_block,
        par,
    );
}

#[inline(never)]
#[faer_macros::migrate]
fn introduce_bulges<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    ilo: usize,
    ihi: usize,
    n_block_desired: usize,
    n_bulges: usize,
    n_shifts: usize,
    want_t: bool,
    mut a: MatMut<'_, C, T>,
    mut z: Option<MatMut<'_, C, T>>,

    mut u: MatMut<'_, C, T>,
    mut v: MatMut<'_, C, T>,
    mut wh: MatMut<'_, C, T>,
    mut wv: MatMut<'_, C, T>,
    s: ColRef<'_, C, T>,

    i_pos_block: &mut usize,
    par: Par,
) {
    help!(C);

    let n = a.nrows();

    let eps = math.eps();
    let small_num = math.re(min_positive() / eps * from_f64(n as f64));

    // Near-the-diagonal bulge introduction
    // The calculations are initially limited to the window:
    // A(ilo:ilo+n_block,ilo:ilo+n_block) The rest is updated later via
    // level 3 BLAS
    let n_block = Ord::min(n_block_desired, ihi - ilo);
    let mut istart_m = ilo;
    let mut istop_m = ilo + n_block;
    let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
    u2.fill(math.zero());
    u2.rb_mut().diagonal_mut().fill(math.one());
    for i_pos_last in ilo..ilo + n_block - 2 {
        // The number of bulges that are in the pencil
        let n_active_bulges = Ord::min(n_bulges, ((i_pos_last - ilo) / 2) + 1);

        for i_bulge in 0..n_active_bulges {
            let i_pos = i_pos_last - 2 * i_bulge;
            let mut v = v.rb_mut().col_mut(i_bulge);
            if i_pos == ilo {
                // Introduce bulge
                let h = a.rb().submatrix(ilo, ilo, 3, 3);

                let s1 = s.read(s.nrows() - 1 - 2 * i_bulge);
                let s2 = s.read(s.nrows() - 1 - 2 * i_bulge - 1);
                lahqr_shiftcolumn(ctx, h, v.rb_mut(), s1, s2);

                debug_assert!(v.nrows() == 3);
                let mut beta = v.read(0);
                let tail = v.rb_mut().subrows_mut(1, 2);
                let (tau, _) = make_householder_in_place(ctx, as_mut!(beta), tail);
                v.write(0, tau.faer_inv());
            } else {
                // Chase bulge down
                let mut h = a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
                let s1 = s.read(s.nrows() - 1 - 2 * i_bulge);
                let s2 = s.read(s.nrows() - 1 - 2 * i_bulge - 1);
                move_bulge(ctx, h.rb_mut(), v.rb_mut(), s1, s2);
            }

            // Apply the reflector we just calculated from the right
            // We leave the last row for later (it interferes with the
            // optimally packed bulges)

            let v0 = v.read(0).faer_real();
            let v1 = v.read(1);
            let v2 = v.read(2);

            for j in istart_m..i_pos + 3 {
                let sum = a
                    .read(j, i_pos)
                    .faer_add(v1.faer_mul(a.read(j, i_pos + 1)))
                    .faer_add(v2.faer_mul(a.read(j, i_pos + 2)));
                a.write(j, i_pos, a.read(j, i_pos).faer_sub(sum.faer_scale_real(v0)));
                a.write(
                    j,
                    i_pos + 1,
                    a.read(j, i_pos + 1)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v1.faer_conj())),
                );
                a.write(
                    j,
                    i_pos + 2,
                    a.read(j, i_pos + 2)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v2.faer_conj())),
                );
            }

            // Apply the reflector we just calculated from the left
            // We only update a single column, the rest is updated later
            let sum = a
                .read(i_pos, i_pos)
                .faer_add(v1.faer_conj().faer_mul(a.read(i_pos + 1, i_pos)))
                .faer_add(v2.faer_conj().faer_mul(a.read(i_pos + 2, i_pos)));
            a.write(
                i_pos,
                i_pos,
                a.read(i_pos, i_pos).faer_sub(sum.faer_scale_real(v0)),
            );
            a.write(
                i_pos + 1,
                i_pos,
                a.read(i_pos + 1, i_pos)
                    .faer_sub(sum.faer_scale_real(v0).faer_mul(v1)),
            );
            a.write(
                i_pos + 2,
                i_pos,
                a.read(i_pos + 2, i_pos)
                    .faer_sub(sum.faer_scale_real(v0).faer_mul(v2)),
            );

            // Test for deflation.
            if (i_pos > ilo) && math(a[(i_pos, i_pos - 1)] != zero()) {
                let mut tst1 =
                    math.re(cx.abs1(a[(i_pos - 1, i_pos - 1)]) + cx.abs1(a[(i_pos, i_pos)]));
                if math.re(tst1 == zero()) {
                    if i_pos > ilo + 1 {
                        tst1 = math.re(tst1 + cx.abs1(a[(i_pos - 1, i_pos - 2)]));
                    }
                    if i_pos > ilo + 2 {
                        tst1 = math.re(tst1 + cx.abs1(a[(i_pos - 1, i_pos - 3)]));
                    }
                    if i_pos > ilo + 3 {
                        tst1 = math.re(tst1 + cx.abs1(a[(i_pos - 1, i_pos - 4)]));
                    }
                    if i_pos < ihi - 1 {
                        tst1 = math.re(tst1 + cx.abs1(a[(i_pos + 1, i_pos)]));
                    }
                    if i_pos < ihi - 2 {
                        tst1 = math.re(tst1 + cx.abs1(a[(i_pos + 2, i_pos)]));
                    }
                    if i_pos < ihi - 3 {
                        tst1 = math.re(tst1 + cx.abs1(a[(i_pos + 3, i_pos)]));
                    }
                }
                if math.re(cx.abs1(a[(i_pos, i_pos - 1)]) < max(small_num, eps * tst1)) {
                    let ab = math.max(
                        math.abs1(a.read(i_pos, i_pos - 1)),
                        math.abs1(a.read(i_pos - 1, i_pos)),
                    );
                    let ba = math.min(
                        math.abs1(a.read(i_pos, i_pos - 1)),
                        math.abs1(a.read(i_pos - 1, i_pos)),
                    );
                    let aa = math.max(
                        math.abs1(a.read(i_pos, i_pos)),
                        math.abs1(a.read(i_pos, i_pos).faer_sub(a.read(i_pos - 1, i_pos - 1))),
                    );
                    let bb = math.min(
                        math.abs1(a.read(i_pos, i_pos)),
                        math.abs1(a.read(i_pos, i_pos).faer_sub(a.read(i_pos - 1, i_pos - 1))),
                    );
                    let s = math.re(aa + ab);
                    if math.re(ba * (ab / s) <= max(small_num, eps * (bb * (aa / s)))) {
                        a.write(i_pos, i_pos - 1, math.zero());
                    }
                }
            }
        }

        // Delayed update from the left
        for i_bulge in 0..n_active_bulges {
            let i_pos = i_pos_last - 2 * i_bulge;
            let v = v.rb_mut().col_mut(i_bulge);

            let v0 = v.read(0).faer_real();
            let v1 = v.read(1);
            let v2 = v.read(2);

            for j in i_pos + 1..istop_m {
                let sum = a
                    .read(i_pos, j)
                    .faer_add(v1.faer_conj().faer_mul(a.read(i_pos + 1, j)))
                    .faer_add(v2.faer_conj().faer_mul(a.read(i_pos + 2, j)));
                a.write(i_pos, j, a.read(i_pos, j).faer_sub(sum.faer_scale_real(v0)));
                a.write(
                    i_pos + 1,
                    j,
                    a.read(i_pos + 1, j)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v1)),
                );
                a.write(
                    i_pos + 2,
                    j,
                    a.read(i_pos + 2, j)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v2)),
                );
            }
        }

        // Accumulate the reflectors into U
        for i_bulge in 0..n_active_bulges {
            let i_pos = i_pos_last - 2 * i_bulge;
            let v = v.rb_mut().col_mut(i_bulge);

            let v0 = v.read(0).faer_real();
            let v1 = v.read(1);
            let v2 = v.read(2);

            let i1 = 0;
            let i2 = Ord::min(u2.nrows(), (i_pos_last - ilo) + (i_pos_last - ilo) + 3);

            for j in i1..i2 {
                let sum = u2
                    .read(j, i_pos - ilo)
                    .faer_add(v1.faer_mul(u2.read(j, i_pos - ilo + 1)))
                    .faer_add(v2.faer_mul(u2.read(j, i_pos - ilo + 2)));

                u2.write(
                    j,
                    i_pos - ilo,
                    u2.read(j, i_pos - ilo).faer_sub(sum.faer_scale_real(v0)),
                );
                u2.write(
                    j,
                    i_pos - ilo + 1,
                    u2.read(j, i_pos - ilo + 1)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v1.faer_conj())),
                );
                u2.write(
                    j,
                    i_pos - ilo + 2,
                    u2.read(j, i_pos - ilo + 2)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v2.faer_conj())),
                );
            }
        }
    }
    // Update rest of the matrix
    if want_t {
        istart_m = 0;
        istop_m = n;
    } else {
        istart_m = ilo;
        istop_m = ihi;
    }
    // Horizontal multiply
    if ilo + n_shifts + 1 < istop_m {
        let mut i = ilo + n_block;
        while i < istop_m {
            let iblock = Ord::min(istop_m - i, wh.ncols());
            let mut a_slice = a.rb_mut().submatrix_mut(ilo, i, n_block, iblock);
            let mut wh_slice = wh
                .rb_mut()
                .submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
            matmul(
                ctx,
                wh_slice.rb_mut(),
                Accum::Replace,
                u2.rb().adjoint(),
                a_slice.rb(),
                math.one(),
                par,
            );
            a_slice.copy_from_with(ctx, wh_slice.rb());
            i += iblock;
        }
    }
    // Vertical multiply
    if istart_m < ilo {
        let mut i = istart_m;
        while i < ilo {
            let iblock = Ord::min(ilo - i, wv.nrows());
            let mut a_slice = a.rb_mut().submatrix_mut(i, ilo, iblock, n_block);
            let mut wv_slice = wv
                .rb_mut()
                .submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
            matmul(
                ctx,
                wv_slice.rb_mut(),
                Accum::Replace,
                a_slice.rb(),
                u2.rb(),
                math.one(),
                par,
            );
            a_slice.copy_from_with(ctx, wv_slice.rb());
            i += iblock;
        }
    }
    // Update Z (also a vertical multiplication)
    if let Some(mut z) = z.rb_mut() {
        let mut i = 0;
        while i < n {
            let iblock = Ord::min(n - i, wv.nrows());
            let mut z_slice = z.rb_mut().submatrix_mut(i, ilo, iblock, n_block);
            let mut wv_slice = wv
                .rb_mut()
                .submatrix_mut(0, 0, z_slice.nrows(), z_slice.ncols());
            matmul(
                ctx,
                wv_slice.rb_mut(),
                Accum::Replace,
                z_slice.rb(),
                u2.rb(),
                math.one(),
                par,
            );
            z_slice.copy_from_with(ctx, wv_slice.rb());
            i += iblock;
        }
    }
    *i_pos_block = ilo + n_block - n_shifts;
}

#[inline(never)]
#[faer_macros::migrate]
fn move_bulges_down<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    ilo: usize,
    ihi: usize,
    n_block_desired: usize,
    n_bulges: usize,
    n_shifts: usize,
    want_t: bool,
    mut a: MatMut<'_, C, T>,
    mut z: Option<MatMut<'_, C, T>>,
    mut u: MatMut<'_, C, T>,
    mut v: MatMut<'_, C, T>,
    mut wh: MatMut<'_, C, T>,
    mut wv: MatMut<'_, C, T>,
    s: ColRef<'_, C, T>,
    i_pos_block: &mut usize,
    par: Par,
) {
    help!(C);

    let n = a.nrows();

    let eps = math.eps();
    let small_num = math.re(min_positive() / eps * from_f64(n as f64));

    while *i_pos_block + n_block_desired < ihi {
        // Number of positions each bulge will be moved down
        let n_pos = Ord::min(
            n_block_desired - n_shifts,
            ihi - n_shifts - 1 - *i_pos_block,
        );
        // Actual blocksize
        let n_block = n_shifts + n_pos;

        let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
        u2.fill(math.zero());
        u2.rb_mut().diagonal_mut().fill(math.one());

        // Near-the-diagonal bulge chase
        // The calculations are initially limited to the window:
        // A(i_pos_block-1:i_pos_block+n_block,i_pos_block:i_pos_block+n_block)
        // The rest is updated later via level 3 BLAS
        let mut istart_m = *i_pos_block;
        let mut istop_m = *i_pos_block + n_block;

        for i_pos_last in *i_pos_block + n_shifts - 2..*i_pos_block + n_shifts - 2 + n_pos {
            for i_bulge in 0..n_bulges {
                let i_pos = i_pos_last - 2 * i_bulge;
                let mut v = v.rb_mut().col_mut(i_bulge);

                // Chase bulge down
                let mut h = a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
                let s1 = s.read(s.nrows() - 1 - 2 * i_bulge);
                let s2 = s.read(s.nrows() - 1 - 2 * i_bulge - 1);
                move_bulge(ctx, h.rb_mut(), v.rb_mut(), s1, s2);

                // Apply the reflector we just calculated from the right
                // We leave the last row for later (it interferes with the
                // optimally packed bulges)

                let v0 = v.read(0).faer_real();
                let v1 = v.read(1);
                let v2 = v.read(2);

                for j in istart_m..i_pos + 3 {
                    let sum = a
                        .read(j, i_pos)
                        .faer_add(v1.faer_mul(a.read(j, i_pos + 1)))
                        .faer_add(v2.faer_mul(a.read(j, i_pos + 2)));
                    a.write(j, i_pos, a.read(j, i_pos).faer_sub(sum.faer_scale_real(v0)));
                    a.write(
                        j,
                        i_pos + 1,
                        a.read(j, i_pos + 1)
                            .faer_sub(sum.faer_scale_real(v0).faer_mul(v1.faer_conj())),
                    );
                    a.write(
                        j,
                        i_pos + 2,
                        a.read(j, i_pos + 2)
                            .faer_sub(sum.faer_scale_real(v0).faer_mul(v2.faer_conj())),
                    );
                }

                // Apply the reflector we just calculated from the left
                // We only update a single column, the rest is updated later
                let sum = a
                    .read(i_pos, i_pos)
                    .faer_add(v1.faer_conj().faer_mul(a.read(i_pos + 1, i_pos)))
                    .faer_add(v2.faer_conj().faer_mul(a.read(i_pos + 2, i_pos)));
                a.write(
                    i_pos,
                    i_pos,
                    a.read(i_pos, i_pos).faer_sub(sum.faer_scale_real(v0)),
                );
                a.write(
                    i_pos + 1,
                    i_pos,
                    a.read(i_pos + 1, i_pos)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v1)),
                );
                a.write(
                    i_pos + 2,
                    i_pos,
                    a.read(i_pos + 2, i_pos)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v2)),
                );

                // Test for deflation.
                if (i_pos > ilo) && math(a[(i_pos, i_pos - 1)] != zero()) {
                    let mut tst1 =
                        math.re(cx.abs1(a[(i_pos - 1, i_pos - 1)]) + cx.abs1(a[(i_pos, i_pos)]));
                    if math.re(tst1 == zero()) {
                        if i_pos > ilo + 1 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos - 1, i_pos - 2)]));
                        }
                        if i_pos > ilo + 2 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos - 1, i_pos - 3)]));
                        }
                        if i_pos > ilo + 3 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos - 1, i_pos - 4)]));
                        }
                        if i_pos < ihi - 1 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos + 1, i_pos)]));
                        }
                        if i_pos < ihi - 2 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos + 2, i_pos)]));
                        }
                        if i_pos < ihi - 3 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos + 3, i_pos)]));
                        }
                    }
                    if math.re(cx.abs1(a[(i_pos, i_pos - 1)]) < max(small_num, eps * tst1)) {
                        let ab = math.max(
                            math.abs1(a.read(i_pos, i_pos - 1)),
                            math.abs1(a.read(i_pos - 1, i_pos)),
                        );
                        let ba = math.min(
                            math.abs1(a.read(i_pos, i_pos - 1)),
                            math.abs1(a.read(i_pos - 1, i_pos)),
                        );
                        let aa = math.max(
                            math.abs1(a.read(i_pos, i_pos)),
                            math.abs1(a.read(i_pos, i_pos).faer_sub(a.read(i_pos - 1, i_pos - 1))),
                        );
                        let bb = math.min(
                            math.abs1(a.read(i_pos, i_pos)),
                            math.abs1(a.read(i_pos, i_pos).faer_sub(a.read(i_pos - 1, i_pos - 1))),
                        );
                        let s = math.re(aa + ab);
                        if math.re(ba * (ab / s) <= max(small_num, eps * (bb * (aa / s)))) {
                            a.write(i_pos, i_pos - 1, math.zero());
                        }
                    }
                }
            }

            // Delayed update from the left
            for i_bulge in 0..n_bulges {
                let i_pos = i_pos_last - 2 * i_bulge;
                let v = v.rb_mut().col_mut(i_bulge);

                let v0 = v.read(0).faer_real();
                let v1 = v.read(1);
                let v2 = v.read(2);

                for j in i_pos + 1..istop_m {
                    let sum = a
                        .read(i_pos, j)
                        .faer_add(v1.faer_conj().faer_mul(a.read(i_pos + 1, j)))
                        .faer_add(v2.faer_conj().faer_mul(a.read(i_pos + 2, j)));
                    a.write(i_pos, j, a.read(i_pos, j).faer_sub(sum.faer_scale_real(v0)));
                    a.write(
                        i_pos + 1,
                        j,
                        a.read(i_pos + 1, j)
                            .faer_sub(sum.faer_scale_real(v0).faer_mul(v1)),
                    );
                    a.write(
                        i_pos + 2,
                        j,
                        a.read(i_pos + 2, j)
                            .faer_sub(sum.faer_scale_real(v0).faer_mul(v2)),
                    );
                }
            }

            // Accumulate the reflectors into U
            for i_bulge in 0..n_bulges {
                let i_pos = i_pos_last - 2 * i_bulge;
                let v = v.rb_mut().col_mut(i_bulge);

                let v0 = v.read(0).faer_real();
                let v1 = v.read(1);
                let v2 = v.read(2);

                let i1 = (i_pos - *i_pos_block) - (i_pos_last + 2 - *i_pos_block - n_shifts);
                let i2 = Ord::min(
                    u2.nrows(),
                    (i_pos_last - *i_pos_block) + (i_pos_last + 2 - *i_pos_block - n_shifts) + 3,
                );

                for j in i1..i2 {
                    let sum = u2
                        .read(j, i_pos - *i_pos_block)
                        .faer_add(v1.faer_mul(u2.read(j, i_pos - *i_pos_block + 1)))
                        .faer_add(v2.faer_mul(u2.read(j, i_pos - *i_pos_block + 2)));

                    u2.write(
                        j,
                        i_pos - *i_pos_block,
                        u2.read(j, i_pos - *i_pos_block)
                            .faer_sub(sum.faer_scale_real(v0)),
                    );
                    u2.write(
                        j,
                        i_pos - *i_pos_block + 1,
                        u2.read(j, i_pos - *i_pos_block + 1)
                            .faer_sub(sum.faer_scale_real(v0).faer_mul(v1.faer_conj())),
                    );
                    u2.write(
                        j,
                        i_pos - *i_pos_block + 2,
                        u2.read(j, i_pos - *i_pos_block + 2)
                            .faer_sub(sum.faer_scale_real(v0).faer_mul(v2.faer_conj())),
                    );
                }
            }
        }

        // Update rest of the matrix
        if want_t {
            istart_m = 0;
            istop_m = n;
        } else {
            istart_m = ilo;
            istop_m = ihi;
        }

        // Horizontal multiply
        if *i_pos_block + n_block < istop_m {
            let mut i = *i_pos_block + n_block;
            while i < istop_m {
                let iblock = Ord::min(istop_m - i, wh.ncols());
                let mut a_slice = a.rb_mut().submatrix_mut(*i_pos_block, i, n_block, iblock);
                let mut wh_slice =
                    wh.rb_mut()
                        .submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
                matmul(
                    ctx,
                    wh_slice.rb_mut(),
                    Accum::Replace,
                    u2.rb().adjoint(),
                    a_slice.rb(),
                    math.one(),
                    par,
                );
                a_slice.copy_from_with(ctx, wh_slice.rb());
                i += iblock;
            }
        }

        // Vertical multiply
        if istart_m < *i_pos_block {
            let mut i = istart_m;
            while i < *i_pos_block {
                let iblock = Ord::min(*i_pos_block - i, wv.nrows());
                let mut a_slice = a.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
                let mut wv_slice =
                    wv.rb_mut()
                        .submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
                matmul(
                    ctx,
                    wv_slice.rb_mut(),
                    Accum::Replace,
                    a_slice.rb(),
                    u2.rb(),
                    math.one(),
                    par,
                );
                a_slice.copy_from_with(ctx, wv_slice.rb());
                i += iblock;
            }
        }
        // Update Z (also a vertical multiplication)
        if let Some(mut z) = z.rb_mut() {
            let mut i = 0;
            while i < n {
                let iblock = Ord::min(n - i, wv.nrows());
                let mut z_slice = z.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
                let mut wv_slice =
                    wv.rb_mut()
                        .submatrix_mut(0, 0, z_slice.nrows(), z_slice.ncols());
                matmul(
                    ctx,
                    wv_slice.rb_mut(),
                    Accum::Replace,
                    z_slice.rb(),
                    u2.rb(),
                    math.one(),
                    par,
                );
                z_slice.copy_from_with(ctx, wv_slice.rb());
                i += iblock;
            }
        }

        *i_pos_block += n_pos;
    }
}

#[inline(never)]
#[faer_macros::migrate]
fn remove_bulges<C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    ilo: usize,
    ihi: usize,
    n_bulges: usize,
    n_shifts: usize,
    want_t: bool,
    mut a: MatMut<'_, C, T>,
    mut z: Option<MatMut<'_, C, T>>,
    mut u: MatMut<'_, C, T>,
    mut v: MatMut<'_, C, T>,
    mut wh: MatMut<'_, C, T>,
    mut wv: MatMut<'_, C, T>,
    s: ColRef<'_, C, T>,
    i_pos_block: &mut usize,
    par: Par,
) {
    help!(C);

    let n = a.nrows();

    let eps = math.eps();
    let small_num = math.re(min_positive() / eps * from_f64(n as f64));
    let n_block = ihi - *i_pos_block;

    let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
    u2.fill(math.zero());
    u2.rb_mut().diagonal_mut().fill(math.one());

    // Near-the-diagonal bulge chase
    // The calculations are initially limited to the window:
    // A(i_pos_block-1:ihi,i_pos_block:ihi) The rest is updated later via
    // level 3 BLAS
    let mut istart_m = *i_pos_block;
    let mut istop_m = ihi;

    for i_pos_last in *i_pos_block + n_shifts - 2..ihi + n_shifts - 1 {
        let mut i_bulge_start = if i_pos_last + 3 > ihi {
            (i_pos_last + 3 - ihi) / 2
        } else {
            0
        };

        for i_bulge in i_bulge_start..n_bulges {
            let i_pos = i_pos_last - 2 * i_bulge;
            if i_pos == ihi - 2 {
                // Special case, the bulge is at the bottom, needs a smaller
                // reflector (order 2)
                let mut v = v.rb_mut().subrows_mut(0, 2).col_mut(i_bulge);
                let mut h = a.rb_mut().subrows_mut(i_pos, 2).col_mut(i_pos - 1);
                let mut beta = h.read(0);
                let tail = h.rb_mut().subrows_mut(1, 1);
                let (tau, _) = make_householder_in_place(ctx, as_mut!(beta), tail);
                v.write(0, tau.faer_inv());
                v.write(1, h.read(1));
                h.write(0, beta);
                h.write(1, math.zero());

                let t0 = v.read(0).faer_conj();
                let v1 = v.read(1);
                let t1 = t0.faer_mul(v1);
                // Apply the reflector we just calculated from the right
                for j in istart_m..i_pos + 2 {
                    let sum = a.read(j, i_pos).faer_add(v1.faer_mul(a.read(j, i_pos + 1)));
                    a.write(
                        j,
                        i_pos,
                        a.read(j, i_pos).faer_sub(sum.faer_mul(t0.faer_conj())),
                    );
                    a.write(
                        j,
                        i_pos + 1,
                        a.read(j, i_pos + 1).faer_sub(sum.faer_mul(t1.faer_conj())),
                    );
                }
                // Apply the reflector we just calculated from the left
                for j in i_pos..istop_m {
                    let sum = a
                        .read(i_pos, j)
                        .faer_add(v1.faer_conj().faer_mul(a.read(i_pos + 1, j)));
                    a.write(i_pos, j, a.read(i_pos, j).faer_sub(sum.faer_mul(t0)));
                    a.write(
                        i_pos + 1,
                        j,
                        a.read(i_pos + 1, j).faer_sub(sum.faer_mul(t1)),
                    );
                }
                // Accumulate the reflector into U
                // The loop bounds should be changed to reflect the fact
                // that U2 starts off as diagonal
                for j in 0..u2.nrows() {
                    let sum = u2
                        .read(j, i_pos - *i_pos_block)
                        .faer_add(v1.faer_mul(u2.read(j, i_pos - *i_pos_block + 1)));
                    u2.write(
                        j,
                        i_pos - *i_pos_block,
                        u2.read(j, i_pos - *i_pos_block)
                            .faer_sub(sum.faer_mul(t0.faer_conj())),
                    );
                    u2.write(
                        j,
                        i_pos - *i_pos_block + 1,
                        u2.read(j, i_pos - *i_pos_block + 1)
                            .faer_sub(sum.faer_mul(t1.faer_conj())),
                    );
                }
            } else {
                let mut v = v.rb_mut().col_mut(i_bulge);
                let mut h = a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
                let s1 = s.read(s.nrows() - 1 - 2 * i_bulge);
                let s2 = s.read(s.nrows() - 1 - 2 * i_bulge - 1);
                move_bulge(ctx, h.rb_mut(), v.rb_mut(), s1, s2);

                {
                    let t0 = v.read(0).faer_conj();
                    let v1 = v.read(1);
                    let t1 = t0.faer_mul(v1);
                    let v2 = v.read(2);
                    let t2 = t0.faer_mul(v2);
                    // Apply the reflector we just calculated from the right
                    // (but leave the last row for later)
                    for j in istart_m..i_pos + 3 {
                        let sum = a
                            .read(j, i_pos)
                            .faer_add(v1.faer_mul(a.read(j, i_pos + 1)))
                            .faer_add(v2.faer_mul(a.read(j, i_pos + 2)));
                        a.write(
                            j,
                            i_pos,
                            a.read(j, i_pos).faer_sub(sum.faer_mul(t0.faer_conj())),
                        );
                        a.write(
                            j,
                            i_pos + 1,
                            a.read(j, i_pos + 1).faer_sub(sum.faer_mul(t1.faer_conj())),
                        );
                        a.write(
                            j,
                            i_pos + 2,
                            a.read(j, i_pos + 2).faer_sub(sum.faer_mul(t2.faer_conj())),
                        );
                    }
                }

                let v0 = v.read(0).faer_real();
                let v1 = v.read(1);
                let v2 = v.read(2);
                // Apply the reflector we just calculated from the left
                // We only update a single column, the rest is updated later
                let sum = a
                    .read(i_pos, i_pos)
                    .faer_add(v1.faer_conj().faer_mul(a.read(i_pos + 1, i_pos)))
                    .faer_add(v2.faer_conj().faer_mul(a.read(i_pos + 2, i_pos)));
                a.write(
                    i_pos,
                    i_pos,
                    a.read(i_pos, i_pos).faer_sub(sum.faer_scale_real(v0)),
                );
                a.write(
                    i_pos + 1,
                    i_pos,
                    a.read(i_pos + 1, i_pos)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v1)),
                );
                a.write(
                    i_pos + 2,
                    i_pos,
                    a.read(i_pos + 2, i_pos)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v2)),
                );

                // Test for deflation.
                if (i_pos > ilo) && math(a[(i_pos, i_pos - 1)] != zero()) {
                    let mut tst1 =
                        math.re(cx.abs1(a[(i_pos - 1, i_pos - 1)]) + cx.abs1(a[(i_pos, i_pos)]));
                    if math.re(tst1 == zero()) {
                        if i_pos > ilo + 1 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos - 1, i_pos - 2)]));
                        }
                        if i_pos > ilo + 2 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos - 1, i_pos - 3)]));
                        }
                        if i_pos > ilo + 3 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos - 1, i_pos - 4)]));
                        }
                        if i_pos < ihi - 1 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos + 1, i_pos)]));
                        }
                        if i_pos < ihi - 2 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos + 2, i_pos)]));
                        }
                        if i_pos < ihi - 3 {
                            tst1 = math.re(tst1 + cx.abs1(a[(i_pos + 3, i_pos)]));
                        }
                    }
                    if math.re(cx.abs1(a[(i_pos, i_pos - 1)]) < max(small_num, eps * tst1)) {
                        let ab = math.max(
                            math.abs1(a.read(i_pos, i_pos - 1)),
                            math.abs1(a.read(i_pos - 1, i_pos)),
                        );
                        let ba = math.min(
                            math.abs1(a.read(i_pos, i_pos - 1)),
                            math.abs1(a.read(i_pos - 1, i_pos)),
                        );
                        let aa = math.max(
                            math.abs1(a.read(i_pos, i_pos)),
                            math.abs1(a.read(i_pos, i_pos).faer_sub(a.read(i_pos - 1, i_pos - 1))),
                        );
                        let bb = math.min(
                            math.abs1(a.read(i_pos, i_pos)),
                            math.abs1(a.read(i_pos, i_pos).faer_sub(a.read(i_pos - 1, i_pos - 1))),
                        );
                        let s = math.re(aa + ab);
                        if math.re(ba * (ab / s) <= max(small_num, eps * (bb * (aa / s)))) {
                            a.write(i_pos, i_pos - 1, math.zero());
                        }
                    }
                }
            }
        }

        i_bulge_start = if i_pos_last + 4 > ihi {
            (i_pos_last + 4 - ihi) / 2
        } else {
            0
        };

        // Delayed update from the left
        for i_bulge in i_bulge_start..n_bulges {
            let i_pos = i_pos_last - 2 * i_bulge;
            let v = v.rb_mut().col_mut(i_bulge);

            let v0 = v.read(0).faer_real();
            let v1 = v.read(1);
            let v2 = v.read(2);

            for j in i_pos + 1..istop_m {
                let sum = a
                    .read(i_pos, j)
                    .faer_add(v1.faer_conj().faer_mul(a.read(i_pos + 1, j)))
                    .faer_add(v2.faer_conj().faer_mul(a.read(i_pos + 2, j)));
                a.write(i_pos, j, a.read(i_pos, j).faer_sub(sum.faer_scale_real(v0)));
                a.write(
                    i_pos + 1,
                    j,
                    a.read(i_pos + 1, j)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v1)),
                );
                a.write(
                    i_pos + 2,
                    j,
                    a.read(i_pos + 2, j)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v2)),
                );
            }
        }

        // Accumulate the reflectors into U
        for i_bulge in i_bulge_start..n_bulges {
            let i_pos = i_pos_last - 2 * i_bulge;
            let v = v.rb_mut().col_mut(i_bulge);

            let v0 = v.read(0).faer_real();
            let v1 = v.read(1);
            let v2 = v.read(2);

            let i1 = (i_pos - *i_pos_block) - (i_pos_last + 2 - *i_pos_block - n_shifts);
            let i2 = Ord::min(
                u2.nrows(),
                (i_pos_last - *i_pos_block) + (i_pos_last + 2 - *i_pos_block - n_shifts) + 3,
            );

            for j in i1..i2 {
                let sum = u2
                    .read(j, i_pos - *i_pos_block)
                    .faer_add(v1.faer_mul(u2.read(j, i_pos - *i_pos_block + 1)))
                    .faer_add(v2.faer_mul(u2.read(j, i_pos - *i_pos_block + 2)));

                u2.write(
                    j,
                    i_pos - *i_pos_block,
                    u2.read(j, i_pos - *i_pos_block)
                        .faer_sub(sum.faer_scale_real(v0)),
                );
                u2.write(
                    j,
                    i_pos - *i_pos_block + 1,
                    u2.read(j, i_pos - *i_pos_block + 1)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v1.faer_conj())),
                );
                u2.write(
                    j,
                    i_pos - *i_pos_block + 2,
                    u2.read(j, i_pos - *i_pos_block + 2)
                        .faer_sub(sum.faer_scale_real(v0).faer_mul(v2.faer_conj())),
                );
            }
        }
    }

    // Update rest of the matrix
    if want_t {
        istart_m = 0;
        istop_m = n;
    } else {
        istart_m = ilo;
        istop_m = ihi;
    }

    debug_assert!(*i_pos_block + n_block == ihi);

    // Horizontal multiply
    if ihi < istop_m {
        let mut i = ihi;
        while i < istop_m {
            let iblock = Ord::min(istop_m - i, wh.ncols());
            let mut a_slice = a.rb_mut().submatrix_mut(*i_pos_block, i, n_block, iblock);
            let mut wh_slice = wh
                .rb_mut()
                .submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
            matmul(
                ctx,
                wh_slice.rb_mut(),
                Accum::Replace,
                u2.rb().adjoint(),
                a_slice.rb(),
                math.one(),
                par,
            );
            a_slice.copy_from_with(ctx, wh_slice.rb());
            i += iblock;
        }
    }

    // Vertical multiply
    if istart_m < *i_pos_block {
        let mut i = istart_m;
        while i < *i_pos_block {
            let iblock = Ord::min(*i_pos_block - i, wv.nrows());
            let mut a_slice = a.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
            let mut wv_slice = wv
                .rb_mut()
                .submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
            matmul(
                ctx,
                wv_slice.rb_mut(),
                Accum::Replace,
                a_slice.rb(),
                u2.rb(),
                math.one(),
                par,
            );
            a_slice.copy_from_with(ctx, wv_slice.rb());
            i += iblock;
        }
    }
    // Update Z (also a vertical multiplication)
    if let Some(mut z) = z.rb_mut() {
        let mut i = 0;
        while i < n {
            let iblock = Ord::min(n - i, wv.nrows());
            let mut z_slice = z.rb_mut().submatrix_mut(i, *i_pos_block, iblock, n_block);
            let mut wv_slice = wv
                .rb_mut()
                .submatrix_mut(0, 0, z_slice.nrows(), z_slice.ncols());
            matmul(
                ctx,
                wv_slice.rb_mut(),
                Accum::Replace,
                z_slice.rb(),
                u2.rb(),
                math.one(),
                par,
            );
            z_slice.copy_from_with(ctx, wv_slice.rb());
            i += iblock;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert, c64, linalg::evd::multishift_qr_scratch, prelude::*, utils::approx::*};
    use dyn_stack::{DynStack, GlobalMemBuffer};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn test_3() {
        let n = 3;
        let h = MatRef::from_row_major_array(
            const {
                &[
                    [
                        c64::new(0.997386, 0.677592),
                        c64::new(0.646064, 0.936948),
                        c64::new(0.090948, 0.674011),
                    ],
                    [
                        c64::new(0.212396, 0.976794),
                        c64::new(0.460270, 0.926436),
                        c64::new(0.494441, 0.888187),
                    ],
                    [
                        c64::new(0.000000, 0.000000),
                        c64::new(0.616652, 0.840012),
                        c64::new(0.768245, 0.349193),
                    ],
                ]
            },
        );

        let mut q = Mat::from_fn(n, n, |i, j| if i == j { c64::ONE } else { c64::ZERO });
        let mut w = Col::zeros(n);
        let mut t = h.cloned();
        super::lahqr(&ctx(), true, t.as_mut(), Some(q.as_mut()), w.as_mut(), 0, n);

        let h_reconstructed = &q * &t * q.adjoint();

        let approx_eq = CwiseMat(ApproxEq::<Unit, c64>::eps());
        assert!(h_reconstructed ~ h);
    }

    #[test]
    fn test_n() {
        let rng = &mut StdRng::seed_from_u64(4);

        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32, 128, 256] {
            for _ in 0..10 {
                let mut h = Mat::<c64>::zeros(n, n);
                for j in 0..n {
                    for i in 0..n {
                        if i <= j + 1 {
                            h[(i, j)] = c64::new(rng.gen(), rng.gen());
                        }
                    }
                }

                if n <= 128 {
                    let mut q =
                        Mat::from_fn(n, n, |i, j| if i == j { c64::ONE } else { c64::ZERO });

                    let mut w = Col::zeros(n);

                    let mut t = h.clone();
                    super::lahqr(&ctx(), true, t.as_mut(), Some(q.as_mut()), w.as_mut(), 0, n);

                    let h_reconstructed = &q * &t * q.adjoint();

                    let mut approx_eq = CwiseMat(ApproxEq::<Unit, c64>::eps());
                    approx_eq.0.abs_tol *= 10.0 * (n as f64).sqrt();
                    approx_eq.0.rel_tol *= 10.0 * (n as f64).sqrt();

                    assert!(h_reconstructed ~ h);
                }

                {
                    let mut q = Mat::zeros(n, n);
                    for i in 0..n {
                        q[(i, i)] = c64::ONE;
                    }

                    let mut w = Col::zeros(n);

                    let mut t = h.as_ref().cloned();
                    super::multishift_qr(
                        &ctx(),
                        true,
                        t.as_mut(),
                        Some(q.as_mut()),
                        w.as_mut(),
                        0,
                        n,
                        Par::Seq,
                        DynStack::new(&mut GlobalMemBuffer::new(
                            multishift_qr_scratch::<Unit, c64>(
                                n,
                                n,
                                true,
                                true,
                                Par::Seq,
                                Default::default(),
                            )
                            .unwrap(),
                        )),
                        crate::linalg::evd::EvdParams::default(),
                    );

                    for j in 0..n {
                        for i in 0..n {
                            if i > j + 1 {
                                t[(i, j)] = c64::ZERO;
                            }
                        }
                    }

                    let h_reconstructed = &q * &t * q.adjoint();

                    let mut approx_eq = CwiseMat(ApproxEq::<Unit, c64>::eps());
                    approx_eq.0.abs_tol *= 10.0 * (n as f64);
                    approx_eq.0.rel_tol *= 10.0 * (n as f64);

                    assert!(h ~ h_reconstructed);
                }
            }
        }
    }
}
