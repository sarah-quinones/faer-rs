// adapted from <T>LAPACK implementation
//
// https://github.com/tlapack/tlapack
// https://github.com/tlapack/tlapack/blob/master/include/tlapack/lapack/lahqr.hpp

use linalg::jacobi::JacobiRotation;

use crate::internal_prelude::*;

#[math]
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
        let s = math(re.add(
            //
            abs1(h[(0, 0)] - s2),
            abs1(h[(1, 0)]),
        ));
        if math.re(s == zero()) {
            write1!(v[0] = math.zero());
            write1!(v[1] = math.zero());
        } else {
            let s_inv = math.re.recip(s);
            let h10s = math(mul_real(h[(1, 0)], s));
            write1!(
                v[0] = math(
                    mul_real(
                        //
                        (h[(0, 0)] - s1) * (h[(0, 0)] - s2),
                        s_inv
                    ) + h10s * h[(0, 1)]
                )
            );
            write1!(v[1] = math(h10s * (h[(0, 0)] + h[(1, 1)] - s1 - s2)));
        }
    } else {
        let s = math(re.add(
            re.add(
                //
                abs1(h[(0, 0)] - s2),
                abs1(h[(1, 0)]),
            ),
            abs1(h[(2, 0)]),
        ));

        if math.re(s == zero()) {
            write1!(v[0] = math.zero());
            write1!(v[1] = math.zero());
            write1!(v[2] = math.zero());
        } else {
            let s_inv = math.re.recip(s);
            let h10s = math(mul_real(h[(1, 0)], s));
            let h20s = math(mul_real(h[(2, 0)], s));

            write1!(
                v[0] = math(
                    mul_real((h[(0, 0)] - s1) * (h[(0, 0)] - s2), s_inv)
                        + h10s * h[(0, 1)]
                        + h20s * h[(0, 2)]
                )
            );
            write1!(v[1] = math(h10s * (h[(0, 0)] + h[(1, 1)] - s1 - s2) + h20s * h[(1, 2)]));
            write1!(v[2] = math(h20s * (h[(0, 0)] + h[(2, 2)] - s1 - s2) + h10s * h[(2, 1)]));
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

                let JacobiRotation { c: _, s: sn } = JacobiRotation::make_givens(ctx, h00, h10);
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

#[cfg(test)]
mod tests {
    use super::lahqr;
    use crate::{assert, c64, prelude::*, utils::approx::*};

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
        lahqr(&ctx(), true, t.as_mut(), Some(q.as_mut()), w.as_mut(), 0, n);

        let h_reconstructed = &q * &t * q.adjoint();

        let approx_eq = CwiseMat(ApproxEq::<Unit, c64>::eps());
        assert!(h_reconstructed ~ h);
    }

    #[test]
    fn test_n() {
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32, 128] {
            for _ in 0..10 {
                let mut h = Mat::<c64>::zeros(n, n);
                for j in 0..n {
                    for i in 0..n {
                        if i <= j + 1 {
                            h[(i, j)] = c64::new(rand::random(), rand::random());
                        }
                    }
                }

                let mut q = Mat::from_fn(n, n, |i, j| if i == j { c64::ONE } else { c64::ZERO });

                let mut w = Col::zeros(n);

                let mut t = h.clone();
                lahqr(&ctx(), true, t.as_mut(), Some(q.as_mut()), w.as_mut(), 0, n);

                let h_reconstructed = &q * &t * q.adjoint();

                let mut approx_eq = CwiseMat(ApproxEq::<Unit, c64>::eps());
                approx_eq.0.abs_tol *= 10.0 * (n as f64).sqrt();
                approx_eq.0.rel_tol *= 10.0 * (n as f64).sqrt();

                assert!(h_reconstructed ~ h);
            }
        }
    }
}
