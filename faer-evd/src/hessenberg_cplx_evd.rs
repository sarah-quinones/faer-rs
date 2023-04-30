use faer_core::{ComplexField, MatMut, RealField};
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

// ret: (eig1_re eig1_im) (eig2_re eig2_im)
fn lahqr_eig22<E: ComplexField>(mut a00: E, mut a01: E, mut a10: E, mut a11: E) -> (E, E) {
    let zero = E::Real::zero();
    let half = E::Real::from_f64(0.5);

    let s = a00.abs().add(&a01.abs()).add(&a10.abs()).add(&a11.abs());
    if s == zero {
        return (E::zero(), E::zero());
    }

    let s_inv = s.inv();

    a00 = a00.scale_real(&s_inv);
    a01 = a01.scale_real(&s_inv);
    a10 = a10.scale_real(&s_inv);
    a11 = a11.scale_real(&s_inv);

    let tr = (a00.add(&a11)).scale_power_of_two(&half);
    let det = ((a00.sub(&tr)).mul(&a00.sub(&tr))).add(&a01.mul(&a10));

    let rtdisc = det.sqrt();
    (
        (tr.add(&rtdisc)).scale_real(&s),
        (tr.sub(&rtdisc)).scale_real(&s),
    )
}

fn rotg<E: ComplexField>(a: E, b: E, epsilon: E::Real, zero_threshold: E::Real) -> (E::Real, E, E) {
    let safmin = zero_threshold.clone();
    let safmax = zero_threshold.inv();
    let rtmin = zero_threshold.div(&epsilon).sqrt();
    let rtmax = rtmin.inv();

    // quick return
    if b == E::zero() {
        return (E::Real::one(), E::zero(), E::one());
    }

    let (c, s, r);
    if a == E::zero() {
        c = E::Real::zero();
        let g1 = max(b.real().abs(), b.imag().abs());
        if g1 > rtmin && g1 < rtmax {
            // Use unscaled algorithm
            let g2 = b.real().abs2().add(&b.imag().abs2());
            let d = g2.sqrt();
            s = b.conj().scale_real(&d.inv());
            r = E::from_real(d);
        } else {
            // Use scaled algorithm
            let u = min(safmax, max(safmin, g1));
            let uu = u.inv();
            let gs = b.scale_real(&uu);
            let g2 = gs.real().abs2().add(&gs.imag().abs2());
            let d = g2.sqrt();
            s = gs.conj().scale_real(&d.inv());
            r = E::from_real(d.mul(&u));
        }
    } else {
        let f1 = max(E::real(&a).abs(), E::imag(&a).abs());
        let g1 = max(E::real(&b).abs(), E::imag(&b).abs());
        if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
            // Use unscaled algorithm
            let f2 = a.real().abs2().add(&a.imag().abs2());
            let g2 = b.real().abs2().add(&b.imag().abs2());
            let h2 = f2.add(&g2);
            let d = if f2 > rtmin && h2 < rtmax {
                f2.mul(&h2).sqrt()
            } else {
                f2.sqrt().mul(&h2.sqrt())
            };
            let p = d.inv();
            c = f2.scale_real(&p);
            s = b.conj().mul(&a.scale_real(&p));

            r = a.scale_real(&h2.mul(&p));
        } else {
            // Use scaled algorithm
            let u = min(safmax.clone(), max(safmin.clone(), max(f1.clone(), g1)));
            let uu = u.inv();
            let gs = b.scale_real(&uu);
            let g2 = gs.real().abs2().add(&gs.imag().abs2());
            let (f2, h2, w);
            let fs;
            if f1.scale_real(&uu) < rtmin {
                // a is not well-scaled when scaled by g1.
                let v = min(safmax, max(safmin, f1));
                let vv = v.inv();
                w = v.mul(&uu);
                fs = a.scale_real(&vv);
                f2 = fs.real().abs2().add(&fs.imag().abs2());
                h2 = (f2.mul(&w).mul(&w)).add(&g2);
            } else {
                // Otherwise use the same scaling for a and b.
                w = E::Real::one();
                fs = a.scale_real(&uu);
                f2 = fs.real().abs2().add(&fs.imag().abs2());
                h2 = f2.add(&g2);
            }
            let d = if f2 > rtmin && h2 < rtmax {
                f2.mul(&h2).sqrt()
            } else {
                f2.sqrt().mul(&h2.sqrt())
            };
            let p = d.inv();
            c = (f2.mul(&p)).mul(&w);
            s = gs.conj().mul(&fs.scale_real(&p));
            r = (fs.scale_real(&h2.scale_real(&p))).scale_real(&u);
        }
    }

    (c, s, r)
}

pub fn lahqr<E: ComplexField>(
    want_t: bool,
    a: MatMut<'_, E>,
    z: Option<MatMut<'_, E>>,
    w: MatMut<'_, E>,
    ilo: usize,
    ihi: usize,
    epsilon: E::Real,
    zero_threshold: E::Real,
) -> isize {
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

    let zero = E::Real::zero();
    let eps = epsilon;
    let small_num = zero_threshold.div(&eps);
    let non_convergence_limit = 10;
    let dat1 = E::Real::from_f64(0.75);
    let dat2 = E::Real::from_f64(-0.4375);

    if nh == 0 {
        return 0;
    }

    if nh == 1 {
        w.write(ilo, 0, a.read(ilo, ilo));
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

    for iter in 0..itmax {
        if iter == itmax {
            return istop as isize;
        }

        if istart + 1 >= istop {
            if istart + 1 == istop {
                w.write(istart, 0, a.read(istart, istart));
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
                a.write(i, i - 1, E::zero());
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
                    a.write(i, i - 1, E::zero());
                    istart = i;
                    break;
                }
            }
        }

        if istart + 1 >= istop {
            k_defl = 0;
            w.write(istart, 0, a.read(istart, istart));
            istop = istart;
            istart = ilo;
            continue;
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
            a00 = E::from_real(dat1.mul(&s)).add(&a.read(istop - 1, istop - 1));
            a01 = E::from_real(dat2.mul(&s));
            a10 = E::from_real(s);
            a11 = a00.clone();
        } else {
            // Wilkinson shift
            a00 = a.read(istop - 2, istop - 2);
            a10 = a.read(istop - 1, istop - 2);
            a01 = a.read(istop - 2, istop - 1);
            a11 = a.read(istop - 1, istop - 1);
        }

        let (mut s1, s2) = lahqr_eig22(a00, a01, a10, a11);

        if (s1.sub(&a.read(istop - 1, istop - 1))).abs()
            > (s2.sub(&a.read(istop - 1, istop - 1))).abs()
        {
            s1 = s2;
        }

        // We have already checked whether the subblock has split.
        // If it has split, we can introduce any shift at the top of the new
        // subblock. Now that we know the specific shift, we can also check
        // whether we can introduce that shift somewhere else in the subblock.
        let mut istart2 = istart;
        if istart + 2 < istop {
            for i in (istart + 1..istop - 1).rev() {
                let h00 = a.read(i, i).sub(&s1);
                let h10 = a.read(i + 1, i);

                let (_cs, sn, _r) = rotg(h00, h10, eps.clone(), zero_threshold.clone());
                if (sn.conj().mul(&a.read(i, i - 1))).abs()
                    <= eps.mul(&(a.read(i, i - 1)).abs().add(&(a.read(i, i + 1)).abs()))
                {
                    istart2 = i;
                    break;
                }
            }
        }

        for i in istart2..istop - 1 {
            let (cs, sn, r);
            if i == istart2 {
                let h00 = a.read(i, i).sub(&s1);
                let h10 = a.read(i + 1, i);
                (cs, sn, _) = rotg(h00, h10, eps.clone(), zero_threshold.clone());
                if i > istart {
                    a.write(i, i - 1, a.read(i, i - 1).scale_real(&cs));
                }
            } else {
                (cs, sn, r) = rotg(
                    a.read(i, i - 1),
                    a.read(i + 1, i - 1),
                    eps.clone(),
                    zero_threshold.clone(),
                );
                a.write(i, i - 1, r);
                a.write(i + 1, i - 1, E::zero());
            }

            // Apply G from the left to A
            for j in i..istop_m {
                let tmp = (a.read(i, j).scale_real(&cs)).add(&a.read(i + 1, j).mul(&sn));
                a.write(
                    i + 1,
                    j,
                    (a.read(i, j).mul(&sn.conj().neg())).add(&a.read(i + 1, j).scale_real(&cs)),
                );
                a.write(i, j, tmp);
            }
            // Apply G**H from the right to A
            for j in istart_m..Ord::min(i + 3, istop) {
                let tmp = (a.read(j, i).scale_real(&cs)).add(&a.read(j, i + 1).mul(&sn.conj()));
                a.write(
                    j,
                    i + 1,
                    (a.read(j, i).mul(&sn.neg())).add(&a.read(j, i + 1).scale_real(&cs)),
                );
                a.write(j, i, tmp);
            }
            if let Some(mut z) = z.rb_mut() {
                // Apply G**H to Z from the right
                for j in 0..n {
                    let tmp = (z.read(j, i).scale_real(&cs)).add(&z.read(j, i + 1).mul(&sn.conj()));
                    z.write(
                        j,
                        i + 1,
                        (z.read(j, i).mul(&sn.neg())).add(&z.read(j, i + 1).scale_real(&cs)),
                    );
                    z.write(j, i, tmp);
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
    use faer_core::{c64, mat, Mat};

    #[test]
    fn test_3() {
        let n = 3;
        let h = mat![
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
        ];

        let mut q = Mat::with_dims(n, n, |i, j| if i == j { c64::one() } else { c64::zero() });
        let mut w = Mat::zeros(n, 1);
        let mut t = h.clone();
        lahqr(
            true,
            t.as_mut(),
            Some(q.as_mut()),
            w.as_mut(),
            0,
            n,
            f64::EPSILON,
            f64::MIN_POSITIVE,
        );

        let h_reconstructed = &q * &t * q.adjoint();

        for i in 0..n {
            for j in 0..n {
                assert_approx_eq!(h_reconstructed.read(i, j), h.read(i, j));
            }
        }
    }

    #[test]
    fn test_n() {
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32, 128] {
            for _ in 0..10 {
                let mut h = Mat::<c64>::zeros(n, n);
                for j in 0..n {
                    for i in 0..n {
                        if i <= j + 1 {
                            h.write(i, j, c64::new(rand::random(), rand::random()));
                        }
                    }
                }

                let mut q =
                    Mat::with_dims(n, n, |i, j| if i == j { c64::one() } else { c64::zero() });

                let mut w = Mat::zeros(n, 1);

                let mut t = h.clone();
                lahqr(
                    true,
                    t.as_mut(),
                    Some(q.as_mut()),
                    w.as_mut(),
                    0,
                    n,
                    f64::EPSILON,
                    f64::MIN_POSITIVE,
                );
                dbgf::dbgf!("6.6?", &t, &h);

                let h_reconstructed = &q * &t * q.adjoint();

                for i in 0..n {
                    for j in 0..n {
                        assert_approx_eq!(h_reconstructed.read(i, j), h.read(i, j));
                    }
                }
            }
        }
    }
}
