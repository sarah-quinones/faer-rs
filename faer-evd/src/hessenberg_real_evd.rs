// adapted from <T>LAPACK implementation
//
// https://github.com/tlapack/tlapack
// https://github.com/tlapack/tlapack/blob/master/include/tlapack/lapack/lahqr.hpp

use assert2::{assert, debug_assert};
use faer_core::{
    householder::make_householder_in_place, mul::inner_prod::inner_prod_with_conj, zipped, Conj,
    MatMut, MatRef, RealField,
};
use reborrow::*;

fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

fn min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

fn sign<E: RealField>(a: &E) -> E {
    let zero = E::zero();
    if *a == zero {
        zero
    } else if *a > zero {
        E::one()
    } else {
        E::one().neg()
    }
}

fn rot<E: RealField>(x: MatMut<'_, E>, y: MatMut<'_, E>, c: E, s: E) {
    zipped!(x, y).for_each(|mut x, mut y| {
        let mut x_ = x.read();
        let mut y_ = y.read();

        (x_, y_) = (c.mul(&x_).add(&s.mul(&y_)), c.mul(&y_).sub(&s.mul(&x_)));

        x.write(x_);
        y.write(y_);
    });
}

// ret: (a b c d) (eig1_re eig1_im) (eig2_re eig2_im) (cs sn)
fn lahqr_schur22<E: RealField>(
    mut a: E,
    mut b: E,
    mut c: E,
    mut d: E,
    eps: E,
    zero_threshold: E,
) -> ((E, E, E, E), (E, E), (E, E), (E, E)) {
    let zero = E::zero();

    let half = E::from_f64(0.5);
    let one = E::one();
    let multpl = E::from_f64(4.0);

    let safmin = zero_threshold;
    let safmn2 = safmin.div(&eps).sqrt();
    let safmx2 = safmn2.inv();

    let mut cs;
    let mut sn;

    if c == zero {
        // c is zero, the matrix is already in Schur form.
        cs = one.clone();
        sn = zero.clone();
    } else if b == zero {
        // b is zero, swapping rows and columns results in Schur form.
        cs = zero.clone();
        sn = one.clone();

        let temp = d;

        d = a;
        a = temp;
        b = c.neg();
        c = zero.clone();
    } else if a.sub(&d) == zero && sign(&b) != sign(&c) {
        cs = one.clone();
        sn = zero.clone();
    } else {
        let mut temp = a.sub(&d);
        let mut p = temp.scale_power_of_two(&half);

        let bcmax = max(b.abs(), c.abs());
        let bcmin = min(b.abs(), c.abs()).mul(&sign(&b).mul(&sign(&c)));

        let mut scale = max(p.abs(), bcmax.clone());

        let mut z = ((p.div(&scale)).mul(&p)).add(&bcmax.div(&scale).mul(&bcmin));

        // if z is positive, we should have real eigenvalues
        // however, is z is very small, but positive, we postpone the decision
        if z >= multpl.mul(&eps) {
            // Real eigenvalues.

            // Compute a and d.

            z = p.add(&sign(&p).mul(&scale.sqrt().mul(&z.sqrt())));
            a = d.add(&z);
            d = d.sub(&(bcmax.div(&z)).mul(&bcmin));
            // Compute b and the rotation matrix
            let tau = (c.abs2().add(&z.abs2())).sqrt();
            cs = z.div(&tau);
            sn = c.div(&tau);
            b = b.sub(&c);
            c = zero.clone();
        } else {
            // Complex eigenvalues, or real (almost) equal eigenvalues.

            // Make diagonal elements equal.

            let mut sigma = b.add(&c);
            for _ in 0..20 {
                scale = max(temp.abs(), sigma.abs());
                if scale >= safmx2 {
                    sigma = sigma.mul(&safmn2);
                    temp = temp.mul(&safmn2);
                    continue;
                }
                if scale <= safmn2 {
                    sigma = sigma.mul(&safmx2);
                    temp = temp.mul(&safmx2);
                    continue;
                }
                break;
            }

            p = temp.scale_power_of_two(&half);
            let mut tau = (sigma.abs2().add(&temp.abs2())).sqrt();
            cs = ((one.add(&sigma.abs().div(&tau))).scale_power_of_two(&half)).sqrt();
            sn = (p.div(&tau.mul(&cs))).neg().mul(&sign(&sigma));
            //
            // Compute [aa bb] = [a b][cs -sn]
            //         [cc dd] = [c d][sn  cs]
            //
            let aa = a.mul(&cs).add(&b.mul(&sn));
            let bb = a.neg().mul(&sn).add(&b.mul(&cs));
            let cc = c.mul(&cs).add(&d.mul(&sn));
            let dd = c.neg().mul(&sn).add(&d.mul(&cs));
            //
            // Compute [a b] = [ cs sn][aa bb]
            //         [c d] = [-sn cs][cc dd]
            //
            a = aa.mul(&cs).add(&cc.mul(&sn));
            b = bb.mul(&cs).add(&dd.mul(&sn));
            c = aa.neg().mul(&sn).add(&cc.mul(&cs));
            d = bb.neg().mul(&sn).add(&dd.mul(&cs));

            temp = (a.add(&d)).scale_power_of_two(&half);
            a = temp.clone();
            d = temp.clone();

            if c != zero {
                if b != zero {
                    if sign(&b) == sign(&c) {
                        // Real eigenvalues: reduce to upper triangular form
                        let sab = b.abs().sqrt();
                        let sac = c.abs().sqrt();
                        p = if c > zero {
                            sab.mul(&sac)
                        } else {
                            sab.neg().mul(&sac)
                        };
                        tau = (b.add(&c)).abs().sqrt().inv();
                        a = temp.add(&p);
                        d = temp.sub(&p);
                        b = b.sub(&c);
                        c = zero.clone();
                        let cs1 = sab.mul(&tau);
                        let sn1 = sac.mul(&tau);
                        temp = cs.mul(&cs1).sub(&sn.mul(&sn1));
                        sn = cs.mul(&sn1).add(&sn.mul(&cs1));
                        cs = temp;
                    }
                }
            }
        }
    }

    let (s1, s2) = if c != zero {
        let temp = b.abs().sqrt().mul(&c.abs().sqrt());
        ((a.clone(), temp.clone()), (d.clone(), temp.neg()))
    } else {
        ((a.clone(), E::zero()), (d.clone(), E::zero()))
    };

    ((a, b, c, d), s1, s2, (cs, sn))
}

// ret: (eig1_re eig1_im) (eig2_re eig2_im)
fn lahqr_eig22<E: RealField>(mut a00: E, mut a01: E, mut a10: E, mut a11: E) -> ((E, E), (E, E)) {
    let zero = E::zero();
    let half = E::from_f64(0.5);

    let s = a00.abs().add(&a01.abs()).add(&a10.abs()).add(&a11.abs());
    if s == zero {
        return ((zero.clone(), zero.clone()), (zero.clone(), zero));
    }

    a00 = a00.div(&s);
    a01 = a01.div(&s);
    a10 = a10.div(&s);
    a11 = a11.div(&s);

    let tr = (a00.add(&a11)).scale_power_of_two(&half);
    let det = (a00.sub(&tr)).abs2().add(&a01.mul(&a10));

    if det >= zero {
        let rtdisc = det.sqrt();
        (
            (s.mul(&tr.add(&rtdisc)), zero.clone()),
            (s.mul(&tr.sub(&rtdisc)), zero.clone()),
        )
    } else {
        let rtdisc = det.neg().sqrt();
        let re = s.mul(&tr);
        let im = s.mul(&rtdisc);
        ((re.clone(), im.clone()), (re, im.neg()))
    }
}

fn lahqr_shiftcolumn<E: RealField>(h: MatRef<'_, E>, mut v: MatMut<'_, E>, s1: (E, E), s2: (E, E)) {
    debug_assert!(h.nrows() == h.ncols());
    let n = h.nrows();

    debug_assert!(v.nrows() == n);
    debug_assert!(v.ncols() == 1);

    if n == 2 {
        let s = (h.read(0, 0).sub(&s2.0))
            .abs()
            .add(&s2.1.abs())
            .add(&h.read(1, 0).abs());

        if s == E::zero() {
            v.write(0, 0, E::zero());
            v.write(1, 0, E::zero());
        } else {
            let h10s = h.read(1, 0).div(&s);

            let v0 = (h10s.mul(&h.read(0, 1)))
                .add(&(h.read(0, 0).sub(&s1.0)).mul(&(h.read(0, 0).sub(&s2.0)).div(&s)))
                .sub(&s1.1.mul(&s2.1.div(&s)));
            let v1 = h10s.mul(&h.read(0, 0).add(&h.read(1, 1)).sub(&s1.0).sub(&s2.0));
            v.write(0, 0, v0);
            v.write(1, 0, v1);
        }
    } else {
        let s = (h.read(0, 0).sub(&s2.0))
            .abs()
            .add(&s2.1.abs())
            .add(&h.read(1, 0).abs())
            .add(&h.read(2, 0).abs());

        if s == E::zero() {
            v.write(0, 0, E::zero());
            v.write(1, 0, E::zero());
            v.write(2, 0, E::zero());
        } else {
            let h10s = h.read(1, 0).div(&s);
            let h20s = h.read(2, 0).div(&s);
            let v0 = ((h.read(0, 0).sub(&s1.0)).mul(&(h.read(0, 0).sub(&s2.0)).div(&s)))
                .sub(&s1.1.mul(&s2.1.div(&s)))
                .add(&h.read(0, 1).mul(&h10s))
                .add(&h.read(0, 2).mul(&h20s));
            let v1 = (h10s.mul(&h.read(0, 0).add(&h.read(1, 1)).sub(&s1.0).sub(&s2.0)))
                .add(&h.read(1, 2).mul(&h20s));
            let v2 = (h20s.mul(&h.read(0, 0).add(&h.read(2, 2)).sub(&s1.0).sub(&s2.0)))
                .add(&h10s.mul(&h.read(2, 1)));

            v.write(0, 0, v0);
            v.write(1, 0, v1);
            v.write(2, 0, v2);
        }
    }
}

pub fn lahqr<E: RealField>(
    want_t: bool,
    a: MatMut<'_, E>,
    z: Option<MatMut<'_, E>>,
    w_re: MatMut<'_, E>,
    w_im: MatMut<'_, E>,
    ilo: usize,
    ihi: usize,
    epsilon: E,
    zero_threshold: E,
) -> isize {
    assert!(a.nrows() == a.ncols());
    assert!(ilo <= ihi);

    let n = a.nrows();
    let nh = ihi - ilo;

    assert!(w_re.nrows() == n);
    assert!(w_im.nrows() == n);
    assert!(w_re.ncols() == 1);
    assert!(w_im.ncols() == 1);

    if let Some(z) = z.rb() {
        assert!(z.nrows() == n);
        assert!(z.ncols() == n);
    }

    let mut a = a;
    let mut z = z;
    let mut w_re = w_re;
    let mut w_im = w_im;

    let zero = E::zero();
    let one = E::one();
    let eps = epsilon;
    let small_num = zero_threshold.div(&eps);
    let non_convergence_limit = 10;
    let dat1 = E::from_f64(0.75);
    let dat2 = E::from_f64(-0.4375);

    if nh == 0 {
        return 0;
    }

    if nh == 1 {
        w_re.write(ilo, 0, a.read(ilo, ilo));
        w_im.write(ilo, 0, E::zero());
    }

    // itmax is the total number of QR iterations allowed.
    // For most matrices, 3 shifts per eigenvalue is enough, so
    // we set itmax to 30 times nh as a safe limit.
    let itmax = 30 * Ord::max(10, nh);

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

    let mut v = faer_core::Mat::<E>::zeros(3, 1);
    let mut v = v.as_mut();
    for iter in 0..itmax + 1 {
        if iter == itmax {
            return istop as isize;
        }

        if istart + 1 >= istop {
            if istart + 1 == istop {
                w_re.write(istart, 0, a.read(istart, istart));
                w_im.write(istart, 0, E::zero());
            }
            // All eigenvalues have been found, exit and return 0.
            break;
        }

        // Determine range to apply rotations
        let istart_m;
        let istop_m;
        if !want_t {
            istart_m = istart;
            istop_m = istop;
        } else {
            istart_m = 0;
            istop_m = n;
        }

        // Check if active subblock has split
        for i in (istart + 1..istop).rev() {
            if a.read(i, i - 1).abs() < small_num {
                // A(i,i-1) is negligible, take i as new istart.
                a.write(i, i - 1, zero.clone());
                istart = i;
                break;
            }

            let mut tst = a.read(i - 1, i - 1).abs().add(&a.read(i, i).abs());
            if tst == zero {
                if i >= ilo + 2 {
                    tst = tst.add(&a.read(i - 1, i - 2).abs());
                }
                if i < ihi {
                    tst = tst.add(&a.read(i + 1, i).abs());
                }
            }

            if a.read(i, i - 1).abs() <= eps.mul(&tst) {
                //
                // The elementwise deflation test has passed
                // The following performs second deflation test due
                // to Ahues & Tisseur (LAWN 122, 1997). It has better
                // mathematical foundation and improves accuracy in some
                // examples.
                //
                // The test is |A(i,i-1)|*|A(i-1,i)| <=
                // eps*|A(i,i)|*|A(i-1,i-1)| The multiplications might overflow
                // so we do some scaling first.
                //

                let ab = max(a.read(i, i - 1).abs(), a.read(i - 1, i).abs());
                let ba = min(a.read(i, i - 1).abs(), a.read(i - 1, i).abs());
                let aa = max(
                    a.read(i, i).abs(),
                    (a.read(i, i).sub(&a.read(i - 1, i - 1))).abs(),
                );
                let bb = min(
                    a.read(i, i).abs(),
                    (a.read(i, i).sub(&a.read(i - 1, i - 1))).abs(),
                );
                let s = aa.add(&ab);
                if ba.mul(&ab.div(&s)) <= max(small_num.clone(), eps.mul(&bb.mul(&aa.div(&s)))) {
                    // A(i,i-1) is negligible, take i as new istart.
                    a.write(i, i - 1, zero.clone());
                    istart = i;
                    break;
                }
            }
        }

        if istart + 2 >= istop {
            if istart + 1 == istop {
                // 1x1 block
                k_defl = 0;
                w_re.write(istart, 0, a.read(istart, istart));
                w_im.write(istart, 0, zero.clone());
                istop = istart;
                istart = ilo;
                continue;
            }

            if istart + 2 == istop {
                // 2x2 block, normalize the block

                let ((a00, a01, a10, a11), (s1_re, s1_im), (s2_re, s2_im), (cs, sn)) =
                    lahqr_schur22(
                        a.read(istart, istart),
                        a.read(istart, istart + 1),
                        a.read(istart + 1, istart),
                        a.read(istart + 1, istart + 1),
                        eps.clone(),
                        zero_threshold.clone(),
                    );

                a.write(istart, istart, a00);
                a.write(istart, istart + 1, a01);
                a.write(istart + 1, istart, a10);
                a.write(istart + 1, istart + 1, a11);

                w_re.write(istart, 0, s1_re);
                w_im.write(istart, 0, s1_im);
                w_re.write(istart + 1, 0, s2_re);
                w_im.write(istart + 1, 0, s2_im);

                // Apply the rotations from the normalization to the rest of the
                // matrix.
                if want_t {
                    if istart + 2 < istop_m {
                        let x = unsafe {
                            a.rb()
                                .row(istart)
                                .subcols(istart + 2, istop_m - (istart + 2))
                                .const_cast()
                                .transpose()
                        };
                        let y = unsafe {
                            a.rb()
                                .row(istart + 1)
                                .subcols(istart + 2, istop_m - (istart + 2))
                                .const_cast()
                                .transpose()
                        };

                        rot(x, y, cs.clone(), sn.clone());
                    }

                    let x = unsafe {
                        a.rb()
                            .col(istart)
                            .subrows(istart_m, istart - istart_m)
                            .const_cast()
                    };
                    let y = unsafe {
                        a.rb()
                            .col(istart + 1)
                            .subrows(istart_m, istart - istart_m)
                            .const_cast()
                    };

                    rot(x, y, cs.clone(), sn.clone());
                }
                if let Some(z) = z.rb_mut() {
                    let x = unsafe { z.rb().col(istart).const_cast() };
                    let y = unsafe { z.rb().col(istart + 1).const_cast() };

                    rot(x, y, cs.clone(), sn.clone());
                }

                k_defl = 0;
                istop = istart;
                istart = ilo;
                continue;
            }
        }

        // Determine shift
        let (a00, a01, a10, a11);
        k_defl += 1;

        if k_defl % non_convergence_limit == 0 {
            // Exceptional shift
            let mut s = a.read(istop - 1, istop - 2).abs();
            if istop > ilo + 2 {
                s = s.add(&a.read(istop - 2, istop - 3).abs());
            };
            a00 = dat1.mul(&s).add(&a.read(istop - 1, istop - 1));
            a01 = dat2.mul(&s);
            a10 = s;
            a11 = a00.clone();
        } else {
            // Wilkinson shift
            a00 = a.read(istop - 2, istop - 2);
            a10 = a.read(istop - 1, istop - 2);
            a01 = a.read(istop - 2, istop - 1);
            a11 = a.read(istop - 1, istop - 1);
        }

        let (mut s1, mut s2) = lahqr_eig22(a00, a01, a10, a11);
        if s1.1 == zero && s2.1 == zero {
            // The eigenvalues are not complex conjugate, keep only the one
            // closest to A(istop-1, istop-1)
            if (s1.0.sub(&a.read(istop - 1, istop - 1))).abs()
                <= (s2.0.sub(&a.read(istop - 1, istop - 1))).abs()
            {
                s2 = s1.clone();
            } else {
                s1 = s2.clone();
            }
        }

        // We have already checked whether the subblock has split.
        // If it has split, we can introduce any shift at the top of the new
        // subblock. Now that we know the specific shift, we can also check
        // whether we can introduce that shift somewhere else in the subblock.
        let mut istart2 = istart;
        if istart + 3 < istop {
            for i in (istart + 1..istop - 2).rev() {
                let h = a.rb().submatrix(i, i, 3, 3);
                lahqr_shiftcolumn(h, v.rb_mut(), s1.clone(), s2.clone());
                let head = v.read(0, 0);
                let tail_sqr_norm = v.read(1, 0).abs2().add(&v.read(2, 0).abs2());
                let (tau, _) =
                    make_householder_in_place(Some(v.rb_mut().subrows(1, 2)), head, tail_sqr_norm);
                let tau = tau.inv();

                let v0 = tau;
                let v1 = v.read(1, 0);
                let v2 = v.read(2, 0);

                let refsum = (v0.mul(&a.read(i, i - 1))).add(&v1.mul(&a.read(i + 1, i - 1)));
                if (a.read(i + 1, i - 1).sub(&refsum.mul(&v1)))
                    .abs()
                    .add(&refsum.mul(&v2).abs())
                    <= eps.mul(
                        &a.read(i, i - 1)
                            .abs()
                            .add(&a.read(i, i + 1).abs())
                            .add(&a.read(i + 1, i + 2).abs()),
                    )
                {
                    istart2 = i;
                    break;
                }
            }
        }

        for i in istart2..istop - 1 {
            let nr = Ord::min(3, istop - i);
            let mut t1;
            if i == istart2 {
                let h = a.rb().submatrix(i, i, nr, nr);
                let mut x = v.rb_mut().subrows(0, nr);
                lahqr_shiftcolumn(h, x.rb_mut(), s1.clone(), s2.clone());
                let head = x.read(0, 0);
                let tail = x.rb_mut().subrows(1, nr - 1);
                let tail_sqr_norm = inner_prod_with_conj(tail.rb(), Conj::No, tail.rb(), Conj::No);
                let beta;
                (t1, beta) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
                v.write(0, 0, beta.clone());
                t1 = t1.inv();
                if i > istart {
                    a.write(i, i - 1, a.read(i, i - 1).mul(&one.sub(&t1)));
                }
            } else {
                v.write(0, 0, a.read(i, i - 1));
                v.write(1, 0, a.read(i + 1, i - 1));
                if nr == 3 {
                    v.write(2, 0, a.read(i + 2, i - 1));
                }
                let head = v.read(0, 0);
                let tail = v.rb_mut().subrows(1, nr - 1);
                let tail_sqr_norm = inner_prod_with_conj(tail.rb(), Conj::No, tail.rb(), Conj::No);
                let beta;
                (t1, beta) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
                t1 = t1.inv();
                v.write(0, 0, beta.clone());
                a.write(i, i - 1, beta);
                a.write(i + 1, i - 1, E::zero());
                if nr == 3 {
                    a.write(i + 2, i - 1, E::zero());
                }
            }

            // The following code applies the reflector we have just calculated.
            // We write this out instead of using larf because a direct loop is
            // more efficient for small reflectors.
            let v2 = v.read(1, 0);
            let t2 = t1.mul(&v2);

            if nr == 3 {
                let v3 = v.read(2, 0);
                let t3 = t1.mul(&v.read(2, 0));

                // Apply G from the left to A
                for j in i..istop_m {
                    let sum = a
                        .read(i, j)
                        .add(&v2.mul(&a.read(i + 1, j)))
                        .add(&v3.mul(&a.read(i + 2, j)));
                    a.write(i, j, a.read(i, j).sub(&sum.mul(&t1)));
                    a.write(i + 1, j, a.read(i + 1, j).sub(&sum.mul(&t2)));
                    a.write(i + 2, j, a.read(i + 2, j).sub(&sum.mul(&t3)));
                }
                // Apply G from the right to A
                for j in istart_m..Ord::min(i + 4, istop) {
                    let sum = a
                        .read(j, i)
                        .add(&v2.mul(&a.read(j, i + 1)))
                        .add(&v3.mul(&a.read(j, i + 2)));
                    a.write(j, i, a.read(j, i).sub(&sum.mul(&t1)));
                    a.write(j, i + 1, a.read(j, i + 1).sub(&sum.mul(&t2)));
                    a.write(j, i + 2, a.read(j, i + 2).sub(&sum.mul(&t3)));
                }
                if let Some(mut z) = z.rb_mut() {
                    // Apply G to Z from the right
                    for j in 0..n {
                        let sum = z
                            .read(j, i)
                            .add(&v2.mul(&z.read(j, i + 1)))
                            .add(&v3.mul(&z.read(j, i + 2)));
                        z.write(j, i, z.read(j, i).sub(&sum.mul(&t1)));
                        z.write(j, i + 1, z.read(j, i + 1).sub(&sum.mul(&t2)));
                        z.write(j, i + 2, z.read(j, i + 2).sub(&sum.mul(&t3)));
                    }
                }
            } else {
                // Apply G from the left to A
                for j in i..istop_m {
                    let sum = a.read(i, j).add(&v2.mul(&a.read(i + 1, j)));
                    a.write(i, j, a.read(i, j).sub(&sum.mul(&t1)));
                    a.write(i + 1, j, a.read(i + 1, j).sub(&sum.mul(&t2)));
                }
                // Apply G from the right to A
                for j in istart_m..Ord::min(i + 3, istop) {
                    let sum = a.read(j, i).add(&v2.mul(&a.read(j, i + 1)));
                    a.write(j, i, a.read(j, i).sub(&sum.mul(&t1)));
                    a.write(j, i + 1, a.read(j, i + 1).sub(&sum.mul(&t2)));
                }
                if let Some(mut z) = z.rb_mut() {
                    // Apply G to Z from the right
                    for j in 0..n {
                        let sum = z.read(j, i).add(&v2.mul(&z.read(j, i + 1)));
                        z.write(j, i, z.read(j, i).sub(&sum.mul(&t1)));
                        z.write(j, i + 1, z.read(j, i + 1).sub(&sum.mul(&t2)));
                    }
                }
            }
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{mat, Mat};

    #[test]
    fn test_5() {
        let h = mat![
            [-0.417, -0.056, -2.136, 1.64, -1.793],
            [-0.842, 0.503, -1.245, -1.058, -0.909],
            [0., 2.292, 0.042, -1.118, 0.539],
            [0., 0., 1.175, -0.748, 0.009],
            [0., 0., 0., -0.989, -0.339],
        ];
        let mut q = mat![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ];

        let mut w_re = Mat::zeros(5, 1);
        let mut w_im = Mat::zeros(5, 1);

        let mut t = h.clone();
        lahqr(
            true,
            t.as_mut(),
            Some(q.as_mut()),
            w_re.as_mut(),
            w_im.as_mut(),
            0,
            5,
            f64::EPSILON,
            f64::MIN_POSITIVE,
        );

        let h_reconstructed = &q * &t * q.transpose();

        for i in 0..5 {
            for j in 0..5 {
                assert_approx_eq!(h_reconstructed.read(i, j), h.read(i, j));
            }
        }
    }

    #[test]
    fn test_5_2() {
        let h = mat![
            [0.10, 0.97, 0.19, 0.21, 0.84],
            [0.19, 0.21, 0.05, 0.83, 0.15],
            [0.00, 0.13, 0.05, 0.20, 0.14],
            [0.00, 0.00, 0.45, 0.44, 0.67],
            [0.00, 0.00, 0.00, 0.78, 0.27],
        ];
        let mut q = mat![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ];

        let mut w_re = Mat::zeros(5, 1);
        let mut w_im = Mat::zeros(5, 1);

        let mut t = h.clone();
        lahqr(
            true,
            t.as_mut(),
            Some(q.as_mut()),
            w_re.as_mut(),
            w_im.as_mut(),
            0,
            5,
            f64::EPSILON,
            f64::MIN_POSITIVE,
        );

        let h_reconstructed = &q * &t * q.transpose();

        for i in 0..5 {
            for j in 0..5 {
                assert_approx_eq!(h_reconstructed.read(i, j), h.read(i, j));
            }
        }
    }

    #[test]
    fn test_n() {
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32, 128] {
            for _ in 0..10 {
                let mut h = Mat::<f64>::zeros(n, n);
                for j in 0..n {
                    for i in 0..n {
                        if i <= j + 1 {
                            h.write(i, j, rand::random());
                        }
                    }
                }
                let mut q = Mat::with_dims(n, n, |i, j| if i == j { 1.0 } else { 0.0 });

                let mut w_re = Mat::zeros(n, 1);
                let mut w_im = Mat::zeros(n, 1);

                let mut t = h.clone();
                lahqr(
                    true,
                    t.as_mut(),
                    Some(q.as_mut()),
                    w_re.as_mut(),
                    w_im.as_mut(),
                    0,
                    n,
                    f64::EPSILON,
                    f64::MIN_POSITIVE,
                );
                dbgf::dbgf!("6.6?", &t, &h);

                let h_reconstructed = &q * &t * q.transpose();

                for i in 0..n {
                    for j in 0..n {
                        assert_approx_eq!(h_reconstructed.read(i, j), h.read(i, j));
                    }
                }
            }
        }
    }
}
