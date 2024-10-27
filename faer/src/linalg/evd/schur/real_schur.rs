use super::super::*;
use crate::{assert, debug_assert};
use linalg::{householder::*, jacobi::JacobiRotation, matmul::matmul};

// ret: (a b c d) (eig1_re eig1_im) (eig2_re eig2_im) (cs sn)
#[faer_macros::migrate]
fn lahqr_schur22<T: RealField>(
    mut a: T,
    mut b: T,
    mut c: T,
    mut d: T,
) -> ((T, T, T, T), (T, T), (T, T), (T, T)) {
    let half = from_f64(0.5);
    let one = one();
    let multpl = from_f64(4.0);

    let eps = eps();
    let safmin = min_positive();
    let safmn2 = safmin.faer_div(eps).faer_sqrt();
    let safmx2 = safmn2.faer_inv();

    let mut cs;
    let mut sn;

    if c == zero() {
        // c is zero, the matrix is already in Schur form.
        cs = one;
        sn = zero();
    } else if b == zero() {
        // b is zero, swapping rows and columns results in Schur form.
        cs = zero();
        sn = one;

        core::mem::swap(&mut d, &mut a);
        b = c.faer_neg();
        c = zero();
    } else if (a.faer_sub(d) == zero()) && (b > zero()) != (c > zero()) {
        cs = one;
        sn = zero();
    } else {
        let mut temp = a.faer_sub(d);
        let mut p = temp.faer_scale_power_of_two(half);

        let bcmax = max(b.faer_abs(), c.faer_abs());
        let mut bcmin = min(b.faer_abs(), c.faer_abs());
        if (b > zero()) != (c > zero()) {
            bcmin = -bcmin;
        }

        let mut scale = max(p.faer_abs(), bcmax);

        let mut z =
            ((p.faer_div(scale)).faer_mul(p)).faer_add(bcmax.faer_div(scale).faer_mul(bcmin));

        // if z is positive, we should have real eigenvalues
        // however, is z is very small, but positive, we postpone the decision
        if z >= multpl.faer_mul(eps) {
            // Real eigenvalues.

            // Compute a and d.

            let mut __tmp = scale.faer_sqrt().faer_mul(z.faer_sqrt());
            if p < zero() {
                __tmp = -__tmp;
            }
            z = p.faer_add(__tmp);

            a = d.faer_add(z);
            d = d.faer_sub((bcmax.faer_div(z)).faer_mul(bcmin));
            // Compute b and the rotation matrix
            let tau = (c.faer_abs2().faer_add(z.faer_abs2())).faer_sqrt();
            cs = z.faer_div(tau);
            sn = c.faer_div(tau);
            b = b.faer_sub(c);
            c = zero();
        } else {
            // Complex eigenvalues, or real (almost) equal eigenvalues.

            // Make diagonal elements equal.

            let mut sigma = b.faer_add(c);
            for _ in 0..20 {
                scale = max(temp.faer_abs(), sigma.faer_abs());
                if scale >= safmx2 {
                    sigma = sigma.faer_mul(safmn2);
                    temp = temp.faer_mul(safmn2);
                    continue;
                }
                if scale <= safmn2 {
                    sigma = sigma.faer_mul(safmx2);
                    temp = temp.faer_mul(safmx2);
                    continue;
                }
                break;
            }

            p = temp.faer_scale_power_of_two(half);
            let mut tau = (sigma.faer_abs2().faer_add(temp.faer_abs2())).faer_sqrt();
            cs = ((one.faer_add(sigma.faer_abs().faer_div(tau))).faer_scale_power_of_two(half))
                .faer_sqrt();

            sn = (p.faer_div(tau.faer_mul(cs))).faer_neg();
            if sigma < zero() {
                sn = -sn;
            }
            //
            // Compute [aa bb] = [a b][cs -sn]
            //         [cc dd] = [c d][sn  cs]
            //
            let aa = a.faer_mul(cs).faer_add(b.faer_mul(sn));
            let bb = a.faer_neg().faer_mul(sn).faer_add(b.faer_mul(cs));
            let cc = c.faer_mul(cs).faer_add(d.faer_mul(sn));
            let dd = c.faer_neg().faer_mul(sn).faer_add(d.faer_mul(cs));
            //
            // Compute [a b] = [ cs sn][aa bb]
            //         [c d] = [-sn cs][cc dd]
            //
            a = aa.faer_mul(cs).faer_add(cc.faer_mul(sn));
            b = bb.faer_mul(cs).faer_add(dd.faer_mul(sn));
            c = aa.faer_neg().faer_mul(sn).faer_add(cc.faer_mul(cs));
            d = bb.faer_neg().faer_mul(sn).faer_add(dd.faer_mul(cs));

            temp = (a.faer_add(d)).faer_scale_power_of_two(half);
            a = copy(temp);
            d = copy(temp);

            if c != zero() && b != zero() && (b > zero()) == (c > zero()) {
                // Real eigenvalues: reduce to upper triangular form
                let sab = b.faer_abs().faer_sqrt();
                let sac = c.faer_abs().faer_sqrt();
                p = if c > zero() {
                    sab.faer_mul(sac)
                } else {
                    sab.faer_neg().faer_mul(sac)
                };
                tau = (b.faer_add(c)).faer_abs().faer_sqrt().faer_inv();
                a = temp.faer_add(p);
                d = temp.faer_sub(p);
                b = b.faer_sub(c);
                c = zero();
                let cs1 = sab.faer_mul(tau);
                let sn1 = sac.faer_mul(tau);
                temp = cs.faer_mul(cs1).faer_sub(sn.faer_mul(sn1));
                sn = cs.faer_mul(sn1).faer_add(sn.faer_mul(cs1));
                cs = temp;
            }
        }
    }

    let (s1, s2) = if c != zero() {
        let temp = b.faer_abs().faer_sqrt().faer_mul(c.faer_abs().faer_sqrt());
        ((copy(a), copy(temp)), (copy(d), temp.faer_neg()))
    } else {
        ((copy(a), zero()), (copy(d), zero()))
    };

    ((a, b, c, d), s1, s2, (cs, sn))
}

// ret: (a b c d) (eig1_re eig1_im) (eig2_re eig2_im) (cs sn)
#[math]
fn lahqr_eig22<T: RealField>(mut a00: T, mut a01: T, mut a10: T, mut a11: T) -> ((T, T), (T, T)) {
    let half = from_f64(0.5);

    let s = abs(a00) + abs(a01) + abs(a10) + abs(a11);
    if s == zero() {
        return ((zero(), zero()), (zero(), zero()));
    }

    a00 = a00 / s;
    a01 = a01 / s;
    a10 = a10 / s;
    a11 = a11 / s;

    let tr = (a00 + a11) * half;
    let det = abs2(a00 - tr) + a01 * a10;

    if det >= zero() {
        let rtdisc = sqrt(det);
        ((s * (tr + rtdisc), zero()), (s * (tr - rtdisc), zero()))
    } else {
        let rtdisc = sqrt(-det);
        let re = s * tr;
        let im = s * rtdisc;
        ((copy(re), copy(im)), (re, -im))
    }
}

#[faer_macros::migrate]
fn lahqr_shiftcolumn<T: RealField>(h: MatRef<'_, T>, mut v: ColMut<'_, T>, s1: (T, T), s2: (T, T)) {
    debug_assert!(h.nrows() == h.ncols());
    let n = h.nrows();

    debug_assert!(v.nrows() == n);

    if n == 2 {
        let s = (h.read(0, 0).faer_sub(s2.0))
            .faer_abs()
            .faer_add(s2.1.faer_abs())
            .faer_add(h.read(1, 0).faer_abs());

        if s == zero() {
            v.write(0, zero());
            v.write(1, zero());
        } else {
            let h10s = h.read(1, 0).faer_div(s);

            let v0 = (h10s.faer_mul(h.read(0, 1)))
                .faer_add(
                    (h.read(0, 0).faer_sub(s1.0))
                        .faer_mul((h.read(0, 0).faer_sub(s2.0)).faer_div(s)),
                )
                .faer_sub(s1.1.faer_mul(s2.1.faer_div(s)));
            let v1 = h10s.faer_mul(
                h.read(0, 0)
                    .faer_add(h.read(1, 1))
                    .faer_sub(s1.0)
                    .faer_sub(s2.0),
            );
            v.write(0, v0);
            v.write(1, v1);
        }
    } else {
        let s = (h.read(0, 0).faer_sub(s2.0))
            .faer_abs()
            .faer_add(s2.1.faer_abs())
            .faer_add(h.read(1, 0).faer_abs())
            .faer_add(h.read(2, 0).faer_abs());

        if s == zero() {
            v.write(0, zero());
            v.write(1, zero());
            v.write(2, zero());
        } else {
            let h10s = h.read(1, 0).faer_div(s);
            let h20s = h.read(2, 0).faer_div(s);
            let v0 = ((h.read(0, 0).faer_sub(s1.0))
                .faer_mul((h.read(0, 0).faer_sub(s2.0)).faer_div(s)))
            .faer_sub(s1.1.faer_mul(s2.1.faer_div(s)))
            .faer_add(h.read(0, 1).faer_mul(h10s))
            .faer_add(h.read(0, 2).faer_mul(h20s));
            let v1 = (h10s.faer_mul(
                h.read(0, 0)
                    .faer_add(h.read(1, 1))
                    .faer_sub(s1.0)
                    .faer_sub(s2.0),
            ))
            .faer_add(h.read(1, 2).faer_mul(h20s));
            let v2 = (h20s.faer_mul(
                h.read(0, 0)
                    .faer_add(h.read(2, 2))
                    .faer_sub(s1.0)
                    .faer_sub(s2.0),
            ))
            .faer_add(h10s.faer_mul(h.read(2, 1)));

            v.write(0, v0);
            v.write(1, v1);
            v.write(2, v2);
        }
    }
}

#[math]
fn lasy2<T: RealField>(
    tl: MatRef<'_, T>,
    tr: MatRef<'_, T>,
    b: MatRef<'_, T>,
    x: MatMut<'_, T>,
) -> T {
    let mut x = x;
    let mut info = 0;

    assert!(all(
        tl.nrows() == 2,
        tr.nrows() == 2,
        tl.ncols() == 2,
        tr.ncols() == 2,
    ));

    let eps = eps();
    let smlnum = min_positive() / eps;

    stack_mat!(btmp, 4, 1, T);
    stack_mat!(tmp, 4, 1, T);

    stack_mat!(t16, 4, 4, T);

    let mut jpiv = [0usize; 4];

    let mut smin = max(
        max(abs1(tr[(0, 0)]), abs1(tr[(0, 1)])),
        max(abs1(tr[(1, 0)]), abs1(tr[(1, 1)])),
    );
    smin = max(
        smin,
        max(
            max(abs1(tl[(0, 0)]), abs1(tl[(0, 1)])),
            max(abs1(tl[(1, 0)]), abs1(tl[(1, 1)])),
        ),
    );
    smin = max(eps * smin, smlnum);

    t16.write(0, 0, tl[(0, 0)] - tr[(0, 0)]);
    t16.write(1, 1, tl[(1, 1)] - tr[(0, 0)]);
    t16.write(2, 2, tl[(0, 0)] - tr[(1, 1)]);
    t16.write(3, 3, tl[(1, 1)] - tr[(1, 1)]);

    t16.write(0, 1, copy(tl[(0, 1)]));
    t16.write(1, 0, copy(tl[(1, 0)]));
    t16.write(2, 3, copy(tl[(0, 1)]));
    t16.write(3, 2, copy(tl[(1, 0)]));

    t16.write(0, 2, -tr[(1, 0)]);
    t16.write(1, 3, -tr[(1, 0)]);
    t16.write(2, 0, -tr[(0, 1)]);
    t16.write(3, 1, -tr[(0, 1)]);

    btmp.write(0, 0, copy(b[(0, 0)]));
    btmp.write(1, 0, copy(b[(1, 0)]));
    btmp.write(2, 0, copy(b[(0, 1)]));
    btmp.write(3, 0, copy(b[(1, 1)]));

    let (mut ipsv, mut jpsv);
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        ipsv = i;
        jpsv = i;
        // Do pivoting to get largest pivot element
        let mut xmax = zero();
        for ip in i..4 {
            for jp in i..4 {
                if abs1(t16.read(ip, jp)) >= xmax {
                    xmax = abs1(t16.read(ip, jp));
                    ipsv = ip;
                    jpsv = jp;
                }
            }
        }
        if ipsv != i {
            crate::perm::swap_rows_idx(t16.rb_mut(), ipsv, i);

            let temp = btmp.read(i, 0);
            btmp.write(i, 0, btmp.read(ipsv, 0));
            btmp.write(ipsv, 0, temp);
        }
        if jpsv != i {
            crate::perm::swap_cols_idx(t16.rb_mut(), jpsv, i);
        }
        jpiv[i] = jpsv;
        if abs1(t16.read(i, i)) < smin {
            info = 1;
            t16.write(i, i, copy(smin));
        }
        for j in i + 1..4 {
            t16.write(j, i, t16.read(j, i) / t16.read(i, i));
            btmp.write(j, 0, btmp.read(j, 0) - t16.read(j, i) * btmp.read(i, 0));
            for k in i + 1..4 {
                t16.write(j, k, t16.read(j, k) - t16.read(j, i) * t16.read(i, k));
            }
        }
    }

    if abs1(t16.read(3, 3)) < smin {
        info = 1;
        t16.write(3, 3, copy(smin));
    }
    let mut scale = one();
    let eight = from_f64(8.0);

    if (eight * smlnum) * abs1(btmp[(0, 0)]) > abs1(t16[(0, 0)])
        || (eight * smlnum) * abs1(btmp[(1, 0)]) > abs1(t16[(1, 1)])
        || (eight * smlnum) * abs1(btmp[(2, 0)]) > abs1(t16[(2, 2)])
        || (eight * smlnum) * abs1(btmp[(3, 0)]) > abs1(t16[(3, 3)])
    {
        scale = from_f64(0.125)
            / max(
                max(abs1(btmp[(0, 0)]), abs1(btmp[(1, 0)])),
                max(abs1(btmp[(2, 0)]), abs1(btmp[(3, 0)])),
            );
        btmp.write(0, 0, btmp.read(0, 0) * scale);
        btmp.write(1, 0, btmp.read(1, 0) * scale);
        btmp.write(2, 0, btmp.read(2, 0) * scale);
        btmp.write(3, 0, btmp.read(3, 0) * scale);
    }

    for i in 0..4 {
        let k = 3 - i;
        let temp = recip(t16.read(k, k));
        tmp.write(k, 0, btmp.read(k, 0) * temp);
        for j in k + 1..4 {
            tmp.write(
                k,
                0,
                tmp.read(k, 0) - temp * t16.read(k, j) * tmp.read(j, 0),
            );
        }
    }
    for i in 0..3 {
        if jpiv[2 - i] != 2 - i {
            let temp = tmp.read(2 - i, 0);
            tmp.write(2 - i, 0, tmp.read(jpiv[2 - i], 0));
            tmp.write(jpiv[2 - i], 0, temp);
        }
    }
    x.write(0, 0, tmp.read(0, 0));
    x.write(1, 0, tmp.read(1, 0));
    x.write(0, 1, tmp.read(2, 0));
    x.write(1, 1, tmp.read(3, 0));

    let _ = info;
    scale
}

#[faer_macros::migrate]
fn schur_move<T: RealField>(
    mut a: MatMut<T>,
    mut q: Option<MatMut<T>>,
    mut ifst: usize,
    ilst: &mut usize,
) -> isize {
    let n = a.nrows();

    // Quick return
    if n == 0 {
        return 0;
    }

    // Check if ifst points to the middle of a 2x2 block
    if ifst > 0 && (a[(ifst, ifst - 1)] != zero()) {
        ifst -= 1;
    }

    // Size of the current block, can be either 1, 2
    let mut nbf = 1;
    if ifst < n - 1 && (a[(ifst + 1, ifst)] != zero()) {
        nbf = 2;
    }

    // Check if ilst points to the middle of a 2x2 block
    if *ilst > 0 && (a[(*ilst, *ilst - 1)] != zero()) {
        *ilst -= 1;
    }

    // Size of the final block, can be either 1, 2
    let mut nbl = 1;
    if (*ilst < n - 1) && (a[(*ilst + 1, *ilst)] != zero()) {
        nbl = 2;
    }

    let mut here = ifst;
    if ifst < *ilst {
        if nbf == 2 && nbl == 1 {
            *ilst -= 1;
        }
        if nbf == 1 && nbl == 2 {
            *ilst += 1;
        }

        while here != *ilst {
            // Size of the next eigenvalue block
            let mut nbnext = 1;
            if (here + nbf + 1 < n) && (a[(here + nbf + 1, here + nbf)] != zero()) {
                nbnext = 2;
            }

            let ierr = schur_swap(a.rb_mut(), q.rb_mut(), here, nbf, nbnext);
            if ierr != 0 {
                // The swap failed, return with error
                *ilst = here;
                return 1;
            }
            here += nbnext;
        }
    } else {
        while here != *ilst {
            // Size of the next eigenvalue block
            let mut nbnext = 1;
            if here > 1 && (a[(here - 1, here - 2)] != zero()) {
                nbnext = 2;
            }

            let ierr = schur_swap(a.rb_mut(), q.rb_mut(), here - nbnext, nbnext, nbf);
            if ierr != 0 {
                // The swap failed, return with error
                *ilst = here;
                return 1;
            }
            here -= nbnext;
        }
    }

    0
}

#[faer_macros::migrate]
pub fn schur_swap<T: RealField>(
    mut a: MatMut<T>,
    mut q: Option<MatMut<T>>,
    j0: usize,
    n1: usize,
    n2: usize,
) -> isize {
    let n = a.nrows();
    let epsilon = eps();
    let zero_threshold = min_positive();

    let j1 = j0 + 1;
    let j2 = j0 + 2;
    let j3 = j0 + 3;

    // Check if the 2x2 eigenvalue blocks consist of 2 1x1 blocks
    // If so, treat them separately
    if n1 == 2 && (a.read(j1, j0) == zero()) {
        // only 2x2 swaps can fail, so we don't need to check for error
        schur_swap(a.rb_mut(), q.rb_mut(), j1, 1, n2);
        schur_swap(a.rb_mut(), q.rb_mut(), j0, 1, n2);
        return 0;
    }
    if n2 == 2 && (a[(j0 + n1 + 1, j0 + n1)] == zero()) {
        // only 2x2 swaps can fail, so we don't need to check for error
        schur_swap(a.rb_mut(), q.rb_mut(), j0, n1, 1);
        schur_swap(a.rb_mut(), q.rb_mut(), j1, n1, 1);
        return 0;
    }

    if n1 == 1 && n2 == 1 {
        //
        // Swap two 1-by-1 blocks.
        //
        let t00 = a.read(j0, j0);
        let t11 = a.read(j1, j1);
        //
        // Determine the transformation to perform the interchange
        //
        let temp = a.read(j0, j1);
        let temp2 = t11.faer_sub(t00);
        let (rot, _) = JacobiRotation::rotg(temp, temp2);

        a.write(j1, j1, t00);
        a.write(j0, j0, t11);

        // Apply transformation from the left
        if j2 < n {
            let row1 = unsafe { a.rb().row(j0).subcols(j2, n - j2).const_cast() };
            let row2 = unsafe { a.rb().row(j1).subcols(j2, n - j2).const_cast() };
            rot.apply_on_the_right_in_place((row1.transpose_mut(), row2.transpose_mut()));
        }
        // Apply transformation from the right
        if j0 > 0 {
            let col1 = unsafe { a.rb().col(j0).subrows(0, j0).const_cast() };
            let col2 = unsafe { a.rb().col(j1).subrows(0, j0).const_cast() };
            rot.apply_on_the_right_in_place((col1, col2));
        }
        if let Some(q) = q.rb_mut() {
            let col1 = unsafe { q.rb().col(j0).const_cast() };
            let col2 = unsafe { q.rb().col(j1).const_cast() };
            rot.apply_on_the_right_in_place((col1, col2));
        }
    }
    if n1 == 1 && n2 == 2 {
        //
        // Swap 1-by-1 block with 2-by-2 block
        //
        stack_mat!(b, 3, 2, T);

        b.write(0, 0, a.read(j0, j1));
        b.write(1, 0, a.read(j1, j1).faer_sub(a.read(j0, j0)));
        b.write(2, 0, a.read(j2, j1));
        b.write(0, 1, a.read(j0, j2));
        b.write(1, 1, a.read(j1, j2));
        b.write(2, 1, a.read(j2, j2).faer_sub(a.read(j0, j0)));

        // Make B upper triangular
        let mut v1 = b.rb_mut().col_mut(0);
        let (head, tail) = v1.rb_mut().split_at_row_mut(1);
        let (tau1, _) = make_householder_in_place(head.at_mut(0), tail);
        let tau1 = tau1.faer_inv();
        let v11 = b.read(1, 0);
        let v12 = b.read(2, 0);

        let sum = b
            .read(0, 1)
            .faer_add(v11.faer_mul(b.read(1, 1)))
            .faer_add(v12.faer_mul(b.read(2, 1)));

        b.write(0, 1, b.read(0, 1).faer_sub(sum.faer_mul(tau1)));
        b.write(
            1,
            1,
            b.read(1, 1).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
        );
        b.write(
            2,
            1,
            b.read(2, 1).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
        );

        let mut v2 = b.rb_mut().col_mut(1).subrows_mut(1, 2);
        let (head, tail) = v2.rb_mut().split_at_row_mut(1);
        let (tau2, _) = make_householder_in_place(head.at_mut(0), tail);
        let tau2 = tau2.faer_inv();
        let v21 = v2.read(1);

        //
        // Apply reflections to A and Q
        //

        // Reflections from the left
        for j in j0..n {
            let sum = a
                .read(j0, j)
                .faer_add(v11.faer_mul(a.read(j1, j)))
                .faer_add(v12.faer_mul(a.read(j2, j)));
            a.write(j0, j, a.read(j0, j).faer_sub(sum.faer_mul(tau1)));
            a.write(
                j1,
                j,
                a.read(j1, j).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
            );
            a.write(
                j2,
                j,
                a.read(j2, j).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
            );

            let sum = a.read(j1, j).faer_add(v21.faer_mul(a.read(j2, j)));
            a.write(j1, j, a.read(j1, j).faer_sub(sum.faer_mul(tau2)));
            a.write(
                j2,
                j,
                a.read(j2, j).faer_sub(sum.faer_mul(tau2).faer_mul(v21)),
            );
        }
        // Reflections from the right
        for j in 0..j3 {
            let sum = a
                .read(j, j0)
                .faer_add(v11.faer_mul(a.read(j, j1)))
                .faer_add(v12.faer_mul(a.read(j, j2)));
            a.write(j, j0, a.read(j, j0).faer_sub(sum.faer_mul(tau1)));
            a.write(
                j,
                j1,
                a.read(j, j1).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
            );
            a.write(
                j,
                j2,
                a.read(j, j2).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
            );

            let sum = a.read(j, j1).faer_add(v21.faer_mul(a.read(j, j2)));
            a.write(j, j1, a.read(j, j1).faer_sub(sum.faer_mul(tau2)));
            a.write(
                j,
                j2,
                a.read(j, j2).faer_sub(sum.faer_mul(tau2).faer_mul(v21)),
            );
        }

        if let Some(mut q) = q.rb_mut() {
            for j in 0..n {
                let sum = q
                    .read(j, j0)
                    .faer_add(v11.faer_mul(q.read(j, j1)))
                    .faer_add(v12.faer_mul(q.read(j, j2)));
                q.write(j, j0, q.read(j, j0).faer_sub(sum.faer_mul(tau1)));
                q.write(
                    j,
                    j1,
                    q.read(j, j1).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
                );
                q.write(
                    j,
                    j2,
                    q.read(j, j2).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
                );

                let sum = q.read(j, j1).faer_add(v21.faer_mul(q.read(j, j2)));
                q.write(j, j1, q.read(j, j1).faer_sub(sum.faer_mul(tau2)));
                q.write(
                    j,
                    j2,
                    q.read(j, j2).faer_sub(sum.faer_mul(tau2).faer_mul(v21)),
                );
            }
        }

        a.write(j2, j0, zero());
        a.write(j2, j1, zero());
    }

    if n1 == 2 && n2 == 1 {
        //
        // Swap 2-by-2 block with 1-by-1 block
        //

        stack_mat!(b, 3, 2, T);

        b.write(0, 0, a.read(j1, j2));
        b.write(1, 0, a.read(j1, j1).faer_sub(a.read(j2, j2)));
        b.write(2, 0, a.read(j1, j0));
        b.write(0, 1, a.read(j0, j2));
        b.write(1, 1, a.read(j0, j1));
        b.write(2, 1, a.read(j0, j0).faer_sub(a.read(j2, j2)));

        // Make B upper triangular
        let mut v1 = b.rb_mut().col_mut(0);
        let (head, tail) = v1.rb_mut().split_at_row_mut(1);
        let (tau1, _) = make_householder_in_place(head.at_mut(0), tail);
        let tau1 = tau1.faer_inv();
        let v11 = v1.read(1);
        let v12 = v1.read(2);

        let sum = b
            .read(0, 1)
            .faer_add(v11.faer_mul(b.read(1, 1)))
            .faer_add(v12.faer_mul(b.read(2, 1)));

        b.write(0, 1, b.read(0, 1).faer_sub(sum.faer_mul(tau1)));
        b.write(
            1,
            1,
            b.read(1, 1).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
        );
        b.write(
            2,
            1,
            b.read(2, 1).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
        );

        let mut v2 = b.rb_mut().col_mut(1).subrows_mut(1, 2);
        let (head, tail) = v2.rb_mut().split_at_row_mut(1);
        let (tau2, _) = make_householder_in_place(head.at_mut(0), tail);
        let tau2 = tau2.faer_inv();
        let v21 = v2.read(1);

        //
        // Apply reflections to A and Q
        //

        // Reflections from the left
        for j in j0..n {
            let sum = a
                .read(j2, j)
                .faer_add(v11.faer_mul(a.read(j1, j)))
                .faer_add(v12.faer_mul(a.read(j0, j)));
            a.write(j2, j, a.read(j2, j).faer_sub(sum.faer_mul(tau1)));
            a.write(
                j1,
                j,
                a.read(j1, j).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
            );
            a.write(
                j0,
                j,
                a.read(j0, j).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
            );

            let sum = a.read(j1, j).faer_add(v21.faer_mul(a.read(j0, j)));
            a.write(j1, j, a.read(j1, j).faer_sub(sum.faer_mul(tau2)));
            a.write(
                j0,
                j,
                a.read(j0, j).faer_sub(sum.faer_mul(tau2).faer_mul(v21)),
            );
        }
        // Reflections from the right
        for j in 0..j3 {
            let sum = a
                .read(j, j2)
                .faer_add(v11.faer_mul(a.read(j, j1)))
                .faer_add(v12.faer_mul(a.read(j, j0)));
            a.write(j, j2, a.read(j, j2).faer_sub(sum.faer_mul(tau1)));
            a.write(
                j,
                j1,
                a.read(j, j1).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
            );
            a.write(
                j,
                j0,
                a.read(j, j0).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
            );

            let sum = a.read(j, j1).faer_add(v21.faer_mul(a.read(j, j0)));
            a.write(j, j1, a.read(j, j1).faer_sub(sum.faer_mul(tau2)));
            a.write(
                j,
                j0,
                a.read(j, j0).faer_sub(sum.faer_mul(tau2).faer_mul(v21)),
            );
        }

        if let Some(mut q) = q.rb_mut() {
            for j in 0..n {
                let sum = q
                    .read(j, j2)
                    .faer_add(v11.faer_mul(q.read(j, j1)))
                    .faer_add(v12.faer_mul(q.read(j, j0)));
                q.write(j, j2, q.read(j, j2).faer_sub(sum.faer_mul(tau1)));
                q.write(
                    j,
                    j1,
                    q.read(j, j1).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
                );
                q.write(
                    j,
                    j0,
                    q.read(j, j0).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
                );

                let sum = q.read(j, j1).faer_add(v21.faer_mul(q.read(j, j0)));
                q.write(j, j1, q.read(j, j1).faer_sub(sum.faer_mul(tau2)));
                q.write(
                    j,
                    j0,
                    q.read(j, j0).faer_sub(sum.faer_mul(tau2).faer_mul(v21)),
                );
            }
        }

        a.write(j1, j0, zero());
        a.write(j2, j0, zero());
    }

    if n1 == 2 && n2 == 2 {
        stack_mat!(d, 4, 4, T);

        let ad_slice = a.rb().submatrix(j0, j0, 4, 4);
        d.copy_from_with(ad_slice);
        let mut dnorm = zero();
        z!(d.rb()).for_each(|unzipped!(d)| dnorm = max(dnorm, (*d).faer_abs()));

        let eps = epsilon;
        let small_num = zero_threshold.faer_div(eps);
        let thresh = max(from_f64(10.0).faer_mul(eps).faer_mul(dnorm), small_num);

        stack_mat!(v, 4, 2, T);

        let mut x = v.rb_mut().submatrix_mut(0, 0, 2, 2);
        let (tl, b, _, tr) = d.rb().split_at(2, 2);

        let scale = lasy2(tl, tr, b, x.rb_mut());

        v.write(2, 0, scale.faer_neg());
        v.write(2, 1, zero());
        v.write(3, 0, zero());
        v.write(3, 1, scale.faer_neg());

        // Make V upper triangular
        let mut v1 = v.rb_mut().col_mut(0);
        let (head, tail) = v1.rb_mut().split_at_row_mut(1);
        let (tau1, _) = make_householder_in_place(head.at_mut(0), tail);
        let tau1 = tau1.faer_inv();
        let v11 = v1.read(1);
        let v12 = v1.read(2);
        let v13 = v1.read(3);

        let sum = v
            .read(0, 1)
            .faer_add(v11.faer_mul(v.read(1, 1)))
            .faer_add(v12.faer_mul(v.read(2, 1)))
            .faer_add(v13.faer_mul(v.read(3, 1)));

        v.write(0, 1, v.read(0, 1).faer_sub(sum.faer_mul(tau1)));
        v.write(
            1,
            1,
            v.read(1, 1).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
        );
        v.write(
            2,
            1,
            v.read(2, 1).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
        );
        v.write(
            3,
            1,
            v.read(3, 1).faer_sub(sum.faer_mul(tau1).faer_mul(v13)),
        );

        let mut v2 = v.rb_mut().col_mut(1).subrows_mut(1, 3);
        let (head, tail) = v2.rb_mut().split_at_row_mut(1);
        let (tau2, _) = make_householder_in_place(head.at_mut(0), tail);
        let tau2 = tau2.faer_inv();

        let v21 = v2.read(1);
        let v22 = v2.read(2);

        // Apply reflections to D to check error
        for j in 0..4 {
            let sum = d
                .read(0, j)
                .faer_add(v11.faer_mul(d.read(1, j)))
                .faer_add(v12.faer_mul(d.read(2, j)))
                .faer_add(v13.faer_mul(d.read(3, j)));
            d.write(0, j, d.read(0, j).faer_sub(sum.faer_mul(tau1)));
            d.write(
                1,
                j,
                d.read(1, j).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
            );
            d.write(
                2,
                j,
                d.read(2, j).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
            );
            d.write(
                3,
                j,
                d.read(3, j).faer_sub(sum.faer_mul(tau1).faer_mul(v13)),
            );

            let sum = d
                .read(1, j)
                .faer_add(v21.faer_mul(d.read(2, j)))
                .faer_add(v22.faer_mul(d.read(3, j)));

            d.write(1, j, d.read(1, j).faer_sub(sum.faer_mul(tau2)));
            d.write(
                2,
                j,
                d.read(2, j).faer_sub(sum.faer_mul(tau2).faer_mul(v21)),
            );
            d.write(
                3,
                j,
                d.read(3, j).faer_sub(sum.faer_mul(tau2).faer_mul(v22)),
            );
        }
        for j in 0..4 {
            let sum = d
                .read(j, 0)
                .faer_add(v11.faer_mul(d.read(j, 1)))
                .faer_add(v12.faer_mul(d.read(j, 2)))
                .faer_add(v13.faer_mul(d.read(j, 3)));
            d.write(j, 0, d.read(j, 0).faer_sub(sum.faer_mul(tau1)));
            d.write(
                j,
                1,
                d.read(j, 1).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
            );
            d.write(
                j,
                2,
                d.read(j, 2).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
            );
            d.write(
                j,
                3,
                d.read(j, 3).faer_sub(sum.faer_mul(tau1).faer_mul(v13)),
            );

            let sum = d
                .read(j, 1)
                .faer_add(v21.faer_mul(d.read(j, 2)))
                .faer_add(v22.faer_mul(d.read(j, 3)));
            d.write(j, 1, d.read(j, 1).faer_sub(sum.faer_mul(tau2)));
            d.write(
                j,
                2,
                d.read(j, 2).faer_sub(sum.faer_mul(tau2).faer_mul(v21)),
            );
            d.write(
                j,
                3,
                d.read(j, 3).faer_sub(sum.faer_mul(tau2).faer_mul(v22)),
            );
        }

        if max(
            max(d.read(2, 0).faer_abs(), d.read(2, 1).faer_abs()),
            max(d.read(3, 0).faer_abs(), d.read(3, 1).faer_abs()),
        ) > thresh
        {
            return 1;
        }

        // Reflections from the left
        for j in j0..n {
            let sum = a
                .read(j0, j)
                .faer_add(v11.faer_mul(a.read(j1, j)))
                .faer_add(v12.faer_mul(a.read(j2, j)))
                .faer_add(v13.faer_mul(a.read(j3, j)));

            a.write(j0, j, a.read(j0, j).faer_sub(sum.faer_mul(tau1)));
            a.write(
                j1,
                j,
                a.read(j1, j).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
            );
            a.write(
                j2,
                j,
                a.read(j2, j).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
            );
            a.write(
                j3,
                j,
                a.read(j3, j).faer_sub(sum.faer_mul(tau1).faer_mul(v13)),
            );

            let sum = a
                .read(j1, j)
                .faer_add(v21.faer_mul(a.read(j2, j)))
                .faer_add(v22.faer_mul(a.read(j3, j)));

            a.write(j1, j, a.read(j1, j).faer_sub(sum.faer_mul(tau2)));
            a.write(
                j2,
                j,
                a.read(j2, j).faer_sub(sum.faer_mul(tau2).faer_mul(v21)),
            );
            a.write(
                j3,
                j,
                a.read(j3, j).faer_sub(sum.faer_mul(tau2).faer_mul(v22)),
            );
        }
        // Reflections from the right
        for j in 0..j0 + 4 {
            let sum = a
                .read(j, j0)
                .faer_add(v11.faer_mul(a.read(j, j1)))
                .faer_add(v12.faer_mul(a.read(j, j2)))
                .faer_add(v13.faer_mul(a.read(j, j3)));
            a.write(j, j0, a.read(j, j0).faer_sub(sum.faer_mul(tau1)));
            a.write(
                j,
                j1,
                a.read(j, j1).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
            );
            a.write(
                j,
                j2,
                a.read(j, j2).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
            );
            a.write(
                j,
                j3,
                a.read(j, j3).faer_sub(sum.faer_mul(tau1).faer_mul(v13)),
            );

            let sum = a
                .read(j, j1)
                .faer_add(v21.faer_mul(a.read(j, j2)))
                .faer_add(v22.faer_mul(a.read(j, j3)));
            a.write(j, j1, a.read(j, j1).faer_sub(sum.faer_mul(tau2)));
            a.write(
                j,
                j2,
                a.read(j, j2).faer_sub(sum.faer_mul(tau2).faer_mul(v21)),
            );
            a.write(
                j,
                j3,
                a.read(j, j3).faer_sub(sum.faer_mul(tau2).faer_mul(v22)),
            );
        }

        if let Some(mut q) = q.rb_mut() {
            for j in 0..n {
                let sum = q
                    .read(j, j0)
                    .faer_add(v11.faer_mul(q.read(j, j1)))
                    .faer_add(v12.faer_mul(q.read(j, j2)))
                    .faer_add(v13.faer_mul(q.read(j, j3)));
                q.write(j, j0, q.read(j, j0).faer_sub(sum.faer_mul(tau1)));
                q.write(
                    j,
                    j1,
                    q.read(j, j1).faer_sub(sum.faer_mul(tau1).faer_mul(v11)),
                );
                q.write(
                    j,
                    j2,
                    q.read(j, j2).faer_sub(sum.faer_mul(tau1).faer_mul(v12)),
                );
                q.write(
                    j,
                    j3,
                    q.read(j, j3).faer_sub(sum.faer_mul(tau1).faer_mul(v13)),
                );

                let sum = q
                    .read(j, j1)
                    .faer_add(v21.faer_mul(q.read(j, j2)))
                    .faer_add(v22.faer_mul(q.read(j, j3)));
                q.write(j, j1, q.read(j, j1).faer_sub(sum.faer_mul(tau2)));
                q.write(
                    j,
                    j2,
                    q.read(j, j2).faer_sub(sum.faer_mul(tau2).faer_mul(v21)),
                );
                q.write(
                    j,
                    j3,
                    q.read(j, j3).faer_sub(sum.faer_mul(tau2).faer_mul(v22)),
                );
            }
        }

        a.write(j2, j0, zero());
        a.write(j2, j1, zero());
        a.write(j3, j0, zero());
        a.write(j3, j1, zero());
    }

    // Standardize the 2x2 Schur blocks (if any)
    if n2 == 2 {
        let ((a00, a01, a10, a11), _, _, (cs, sn)) = lahqr_schur22(
            a.read(j0, j0),
            a.read(j0, j1),
            a.read(j1, j0),
            a.read(j1, j1),
        ); // Apply transformation from the left
        let rot = JacobiRotation { c: cs, s: sn };

        a.write(j0, j0, a00);
        a.write(j0, j1, a01);
        a.write(j1, j0, a10);
        a.write(j1, j1, a11);

        if j2 < n {
            let row1 = unsafe { a.rb().row(j0).subcols(j2, n - j2).const_cast() };
            let row2 = unsafe { a.rb().row(j1).subcols(j2, n - j2).const_cast() };

            rot.apply_on_the_right_in_place((row1.transpose_mut(), row2.transpose_mut()));
        }
        // Apply transformation from the right
        if j0 > 0 {
            let col1 = unsafe { a.rb().col(j0).subrows(0, j0).const_cast() };
            let col2 = unsafe { a.rb().col(j1).subrows(0, j0).const_cast() };
            rot.apply_on_the_right_in_place((col1, col2));
        }
        if let Some(q) = q.rb_mut() {
            let col1 = unsafe { q.rb().col(j0).const_cast() };
            let col2 = unsafe { q.rb().col(j1).const_cast() };
            rot.apply_on_the_right_in_place((col1, col2));
        }
    }

    if n1 == 2 {
        let j0 = j0 + n2;
        let j1 = j1 + n2;
        let j2 = j2 + n2;

        let ((a00, a01, a10, a11), _, _, (cs, sn)) = lahqr_schur22(
            a.read(j0, j0),
            a.read(j0, j1),
            a.read(j1, j0),
            a.read(j1, j1),
        ); // Apply transformation from the left
        let rot = JacobiRotation { c: cs, s: sn };

        a.write(j0, j0, a00);
        a.write(j0, j1, a01);
        a.write(j1, j0, a10);
        a.write(j1, j1, a11);

        if j2 < n {
            let row1 = unsafe { a.rb().row(j0).subcols(j2, n - j2).const_cast() };
            let row2 = unsafe { a.rb().row(j1).subcols(j2, n - j2).const_cast() };

            rot.apply_on_the_right_in_place((row1.transpose_mut(), row2.transpose_mut()));
        }
        // Apply transformation from the right
        if j0 > 0 {
            let col1 = unsafe { a.rb().col(j0).subrows(0, j0).const_cast() };
            let col2 = unsafe { a.rb().col(j1).subrows(0, j0).const_cast() };
            rot.apply_on_the_right_in_place((col1, col2));
        }
        if let Some(q) = q.rb_mut() {
            let col1 = unsafe { q.rb().col(j0).const_cast() };
            let col2 = unsafe { q.rb().col(j1).const_cast() };
            rot.apply_on_the_right_in_place((col1, col2));
        }
    }

    0
}

#[faer_macros::migrate]
fn aggressive_early_deflation<T: RealField>(
    want_t: bool,
    mut a: MatMut<'_, T>,
    mut z: Option<MatMut<'_, T>>,
    mut s_re: ColMut<'_, T>,
    mut s_im: ColMut<'_, T>,
    ilo: usize,
    ihi: usize,
    nw: usize,
    par: Par,
    mut stack: &mut DynStack,
    params: EvdParams,
) -> (usize, usize) {
    let n = a.nrows();

    let epsilon = eps();
    let zero_threshold = min_positive();

    // Because we will use the lower triangular part of A as workspace,
    // We have a maximum window size
    let nw_max = (n - 3) / 3;
    let eps = epsilon;
    let small_num = zero_threshold.faer_div(eps).faer_mul(from_f64(n as f64));

    // Size of the deflation window
    let jw = Ord::min(Ord::min(nw, ihi - ilo), nw_max);
    // First row index in the deflation window
    let kwtop = ihi - jw;

    // s is the value just outside the window. It determines the spike
    // together with the orthogonal schur factors.
    let mut s_spike = if kwtop == ilo {
        zero()
    } else {
        a.read(kwtop, kwtop - 1)
    };

    if kwtop + 1 == ihi {
        // 1x1 deflation window, not much to do
        s_re.write(kwtop, a.read(kwtop, kwtop));
        s_im.write(kwtop, zero());
        let mut ns = 1;
        let mut nd = 0;
        if s_spike.faer_abs() <= max(small_num, eps.faer_mul(a.read(kwtop, kwtop).faer_abs())) {
            ns = 0;
            nd = 1;
            if kwtop > ilo {
                a.write(kwtop, kwtop - 1, zero());
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
    let mut s_re_window = unsafe { s_re.rb().subrows(kwtop, ihi - kwtop).const_cast() };
    let mut s_im_window = unsafe { s_im.rb().subrows(kwtop, ihi - kwtop).const_cast() };
    z!(tw.rb_mut())
        .for_each_triangular_lower(linalg::zip::Diag::Include, |unzipped!(mut x)| *x = zero());
    for j in 0..jw {
        for i in 0..Ord::min(j + 2, jw) {
            tw.write(i, j, a_window.read(i, j));
        }
    }
    v.fill(zero());
    v.rb_mut().diagonal_mut().fill(one());

    let infqr = if true
        || jw
            < params
                .blocking_threshold
                .unwrap_or(default_blocking_threshold())
    {
        lahqr(
            true,
            tw.rb_mut(),
            Some(v.rb_mut()),
            s_re_window.rb_mut(),
            s_im_window.rb_mut(),
            0,
            jw,
        )
    } else {
        let infqr = multishift_qr(
            true,
            tw.rb_mut(),
            Some(v.rb_mut()),
            s_re_window.rb_mut(),
            s_im_window.rb_mut(),
            0,
            jw,
            par,
            stack.rb_mut(),
            params,
        )
        .0;
        for j in 0..jw {
            for i in j + 2..jw {
                tw.write(i, j, zero());
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
        let mut bulge = false;
        if ns > 1 && (tw[(ns - 1, ns - 2)] != zero()) {
            bulge = true;
        }

        if !bulge {
            // 1x1 eigenvalue block
            #[allow(clippy::disallowed_names)]
            let mut foo = tw.read(ns - 1, ns - 1).faer_abs();
            if foo == zero() {
                foo = s_spike.faer_abs();
            }
            if s_spike.faer_abs().faer_mul(v[(0, ns - 1)].faer_abs())
                <= max(small_num, eps.faer_mul(foo))
            {
                // Eigenvalue is deflatable
                ns -= 1;
            } else {
                // Eigenvalue is not deflatable.
                // Move it up out of the way.
                let ifst = ns - 1;
                schur_move(tw.rb_mut(), Some(v.rb_mut()), ifst, &mut ilst);
                ilst += 1;
            }
        } else {
            // 2x2 eigenvalue block
            #[allow(clippy::disallowed_names)]
            let mut foo = tw.read(ns - 1, ns - 1).faer_abs().faer_add(
                tw.read(ns - 1, ns - 2)
                    .faer_abs()
                    .faer_sqrt()
                    .faer_mul(tw.read(ns - 2, ns - 1).faer_abs().faer_sqrt()),
            );
            if foo == zero() {
                foo = s_spike.faer_abs();
            }
            if max(
                (s_spike.faer_mul(v[(0, ns - 1)])).faer_abs(),
                (s_spike.faer_mul(v[(0, ns - 2)])).faer_abs(),
            ) <= max(small_num, eps.faer_mul(foo))
            {
                // Eigenvalue pair is deflatable
                ns -= 2;
            } else {
                // Eigenvalue pair is not deflatable.
                // Move it up out of the way.
                let ifst = ns - 2;
                schur_move(tw.rb_mut(), Some(v.rb_mut()), ifst, &mut ilst);
                ilst += 2;
            }
        }
    }

    if ns == 0 {
        s_spike = zero();
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
    let mut sorting_window_size = jw as isize;
    while !sorted {
        sorted = true;

        // Index of last eigenvalue that was swapped
        let mut ilst = 0isize;

        // Index of the first block
        let mut i1 = ns;

        while i1 as isize + 1 < sorting_window_size {
            // Size of the first block
            let mut n1 = 1;
            if tw[(i1 + 1, i1)] != zero() {
                n1 = 2;
            }

            // Check if there is a next block
            if i1 + n1 == jw {
                ilst -= n1 as isize;
                break;
            }

            // Index of the second block
            let i2 = i1 + n1;

            // Size of the second block
            let mut n2 = 1;
            if i2 + 1 < jw && (tw[(i2 + 1, i2)] != zero()) {
                n2 = 2;
            }

            let (ev1, ev2);
            if n1 == 1 {
                ev1 = tw.read(i1, i1).faer_abs();
            } else {
                ev1 = tw.read(i1, i1).faer_abs().faer_add(
                    (tw.read(i1 + 1, i1).faer_abs().faer_sqrt())
                        .faer_mul(tw.read(i1, i1 + 1).faer_abs().faer_sqrt()),
                );
            }
            if n2 == 1 {
                ev2 = tw.read(i2, i2).faer_abs();
            } else {
                ev2 = tw.read(i2, i2).faer_abs().faer_add(
                    (tw.read(i2 + 1, i2).faer_abs().faer_sqrt())
                        .faer_mul(tw.read(i2, i2 + 1).faer_abs().faer_sqrt()),
                );
            }

            if ev1 >= ev2 {
                i1 = i2;
            } else {
                sorted = false;
                let ierr = schur_swap(tw.rb_mut(), Some(v.rb_mut()), i1, n1, n2);
                if ierr == 0 {
                    i1 += n2;
                } else {
                    i1 = i2;
                }
                ilst = i1 as isize;
            }
        }
        sorting_window_size = ilst;
    }

    // Recalculate the eigenvalues
    let mut i = 0;
    while i < jw {
        let mut n1 = 1;
        if i + 1 < jw && (tw[(i + 1, i)] != zero()) {
            n1 = 2;
        }

        if n1 == 1 {
            s_re.write(kwtop + i, tw.read(i, i));
            s_im.write(kwtop + i, zero());
        } else {
            let ((s1_re, s1_im), (s2_re, s2_im)) = lahqr_eig22(
                tw.read(i, i),
                tw.read(i, i + 1),
                tw.read(i + 1, i),
                tw.read(i + 1, i + 1),
            );

            s_re.write(kwtop + i, s1_re);
            s_im.write(kwtop + i, s1_im);
            s_re.write(kwtop + i + 1, s2_re);
            s_im.write(kwtop + i + 1, s2_im);
        }
        i += n1;
    }

    // Reduce A back to Hessenberg form (if necessary)
    if s_spike != zero() {
        // Reflect spike back
        {
            let mut vv = wv.rb_mut().col_mut(0).subrows_mut(0, ns);
            for i in 0..ns {
                vv.write(i, v.read(0, i).faer_conj());
            }
            let mut head = vv.read(0);
            let tail = vv.rb_mut().subrows_mut(1, ns - 1);
            let (tau, _) = make_householder_in_place(&mut head, tail);
            let beta = copy(head);
            vv.write(0, one());
            let tau = tau.faer_inv();

            {
                let mut tw_slice = tw.rb_mut().submatrix_mut(0, 0, ns, jw);
                let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
                let mut tmp = tmp.as_mat_mut().transpose_mut();
                matmul(
                    tmp.rb_mut(),
                    Accum::Replace,
                    vv.rb().adjoint().as_mat(),
                    tw_slice.rb(),
                    one(),
                    par,
                );
                matmul(
                    tw_slice.rb_mut(),
                    Accum::Add,
                    vv.rb().as_mat(),
                    tmp.as_ref(),
                    tau.faer_neg(),
                    par,
                );
            }

            {
                let mut tw_slice2 = tw.rb_mut().submatrix_mut(0, 0, jw, ns);
                let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
                let mut tmp = tmp.as_mat_mut();
                matmul(
                    tmp.rb_mut(),
                    Accum::Replace,
                    tw_slice2.rb(),
                    vv.rb().as_mat(),
                    one(),
                    par,
                );
                matmul(
                    tw_slice2.rb_mut(),
                    Accum::Add,
                    tmp.as_ref(),
                    vv.rb().adjoint().as_mat(),
                    tau.faer_neg(),
                    par,
                );
            }

            {
                let mut v_slice = v.rb_mut().submatrix_mut(0, 0, jw, ns);
                let (mut tmp, _) = unsafe { temp_mat_uninit(jw, 1, stack) };
                let mut tmp = tmp.as_mat_mut();
                matmul(
                    tmp.rb_mut(),
                    Accum::Replace,
                    v_slice.rb(),
                    vv.rb().as_mat(),
                    one(),
                    par,
                );
                matmul(
                    v_slice.rb_mut(),
                    Accum::Add,
                    tmp.as_ref(),
                    vv.rb().adjoint().as_mat(),
                    tau.faer_neg(),
                    par,
                );
            }
            vv.write(0, beta);
        }

        // Hessenberg reduction
        {
            let mut householder = wv.rb_mut().col_mut(0).subrows_mut(0, ns);
            hessenberg::hessenberg_in_place(
                tw.rb_mut().submatrix_mut(0, 0, ns, ns),
                householder.rb_mut().as_mat_mut().transpose_mut(),
                par,
                stack.rb_mut(),
                Default::default(),
            );
            let householder = wv.rb_mut().col_mut(0).subrows_mut(0, ns - 1);

            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                tw.rb().submatrix(1, 0, ns - 1, ns - 1),
                householder.rb().transpose().as_mat(),
                Conj::Yes,
                unsafe { tw.rb().submatrix(1, ns, ns - 1, jw - ns).const_cast() },
                par,
                stack.rb_mut(),
            );
            apply_block_householder_sequence_on_the_right_in_place_with_conj(
                tw.rb().submatrix(1, 0, ns - 1, ns - 1),
                householder.rb().transpose().as_mat(),
                Conj::No,
                v.rb_mut().submatrix_mut(0, 1, jw, ns - 1),
                par,
                stack.rb_mut(),
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
                wh_slice.rb_mut(),
                Accum::Replace,
                v.rb().adjoint(),
                a_slice.rb(),
                one(),
                par,
            );
            a_slice.copy_from_with(wh_slice.rb());
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
                wv_slice.rb_mut(),
                Accum::Replace,
                a_slice.rb(),
                v.rb(),
                one(),
                par,
            );
            a_slice.copy_from_with(wv_slice.rb());
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
                wv_slice.rb_mut(),
                Accum::Replace,
                z_slice.rb(),
                v.rb(),
                one(),
                par,
            );
            z_slice.copy_from_with(wv_slice.rb());
            i += iblock;
        }
    }

    (ns, nd)
}

#[faer_macros::migrate]
fn move_bulge<T: RealField>(mut h: MatMut<'_, T>, mut v: ColMut<'_, T>, s1: (T, T), s2: (T, T)) {
    let epsilon = eps();

    // Perform delayed update of row below the bulge
    // Assumes the first two elements of the row are zero
    let v0 = v.read(0).faer_real();
    let v1 = v.read(1);
    let v2 = v.read(2);
    let refsum = v2.faer_scale_real(v0).faer_mul(h.read(3, 2));

    h.write(3, 0, refsum.faer_neg());
    h.write(3, 1, refsum.faer_neg().faer_mul(v1.faer_conj()));
    h.write(3, 2, h.read(3, 2).faer_sub(refsum.faer_mul(v2.faer_conj())));

    // Generate reflector to move bulge down
    v.write(0, h.read(1, 0));
    v.write(1, h.read(2, 0));
    v.write(2, h.read(3, 0));

    let mut beta = v.read(0);
    let tail = v.rb_mut().subrows_mut(1, 2);
    let (tau, _) = make_householder_in_place(&mut beta, tail);
    v.write(0, tau.faer_inv());

    // Check for bulge collapse
    if h[(3, 0)] != zero() || h[(3, 1)] != zero() || h[(3, 2)] != zero() {
        // The bulge hasn't collapsed, typical case
        h.write(1, 0, beta);
        h.write(2, 0, zero());
        h.write(3, 0, zero());
    } else {
        // The bulge has collapsed, attempt to reintroduce using
        // 2-small-subdiagonals trick

        stack_mat!(vt, 3, 1, T);
        let mut vt = vt.rb_mut().col_mut(0);

        let h2 = h.rb().submatrix(1, 1, 3, 3);
        lahqr_shiftcolumn(h2, vt.rb_mut(), s1, s2);

        let mut beta_unused = vt.read(0);
        let tail = vt.rb_mut().subrows_mut(1, 2);
        let (tau, _) = make_householder_in_place(&mut beta_unused, tail);
        vt.write(0, tau.faer_inv());
        let vt0 = vt.read(0);
        let vt1 = vt.read(1);
        let vt2 = vt.read(2);

        let refsum = (vt0.faer_conj().faer_mul(h.read(1, 0)))
            .faer_add(vt1.faer_conj().faer_mul(h.read(2, 0)));

        if abs1(h[(2, 0)].faer_sub(refsum.faer_mul(vt1))).faer_add(abs1(refsum.faer_mul(vt2)))
            > epsilon.faer_mul(
                abs1(h[(0, 0)])
                    .faer_add(abs1(h[(1, 1)]))
                    .faer_add(abs1(h[(2, 2)])),
            )
        {
            // Starting a new bulge here would create non-negligible fill. Use
            // the old one.
            h.write(1, 0, beta);
            h.write(2, 0, zero());
            h.write(3, 0, zero());
        } else {
            // Fill-in is negligible, use the new reflector.
            h.write(1, 0, h.read(1, 0).faer_sub(refsum));
            h.write(2, 0, zero());
            h.write(3, 0, zero());
            v.write(0, vt.read(0));
            v.write(1, vt.read(1));
            v.write(2, vt.read(2));
        }
    }
}

#[faer_macros::migrate]
fn multishift_qr_sweep<T: RealField>(
    want_t: bool,
    a: MatMut<T>,
    mut z: Option<MatMut<T>>,
    s_re: ColMut<T>,
    s_im: ColMut<T>,
    ilo: usize,
    ihi: usize,
    par: Par,
    stack: &mut DynStack,
) {
    let n = a.nrows();

    assert!(n >= 12);

    let (mut v, _stack) = crate::linalg::temp_mat_zeroed(3, s_re.nrows() / 2, stack);
    let mut v = v.as_mat_mut();

    let n_block_max = (n - 3) / 3;
    let n_shifts_max = Ord::min(ihi - ilo - 1, Ord::max(2, 3 * (n_block_max / 4)));

    let mut n_shifts = Ord::min(s_re.nrows(), n_shifts_max);
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

    introduce_bulges(
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
        s_re.rb(),
        s_im.rb(),
        &mut i_pos_block,
        par,
    );

    move_bulges_down(
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
        s_re.rb(),
        s_im.rb(),
        &mut i_pos_block,
        par,
    );

    remove_bulges(
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
        s_re.rb(),
        s_im.rb(),
        &mut i_pos_block,
        par,
    );
}

#[inline(never)]
#[faer_macros::migrate]
fn introduce_bulges<T: RealField>(
    ilo: usize,
    ihi: usize,
    n_block_desired: usize,
    n_bulges: usize,
    n_shifts: usize,
    want_t: bool,
    mut a: MatMut<'_, T>,
    mut z: Option<MatMut<'_, T>>,

    mut u: MatMut<'_, T>,
    mut v: MatMut<'_, T>,
    mut wh: MatMut<'_, T>,
    mut wv: MatMut<'_, T>,
    s_re: ColRef<'_, T>,
    s_im: ColRef<'_, T>,

    i_pos_block: &mut usize,
    parallelism: Par,
) {
    let n = a.nrows();

    let eps = eps();
    let small_num = min_positive().faer_div(eps).faer_mul(from_f64(n as f64));

    // Near-the-diagonal bulge introduction
    // The calculations are initially limited to the window:
    // A(ilo:ilo+n_block,ilo:ilo+n_block) The rest is updated later via
    // level 3 BLAS

    let n_block = Ord::min(n_block_desired, ihi - ilo);
    let mut istart_m = ilo;
    let mut istop_m = ilo + n_block;
    let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
    u2.fill(zero());
    u2.rb_mut().diagonal_mut().fill(one());

    for i_pos_last in ilo..ilo + n_block - 2 {
        // The number of bulges that are in the pencil
        let n_active_bulges = Ord::min(n_bulges, ((i_pos_last - ilo) / 2) + 1);

        for i_bulge in 0..n_active_bulges {
            let i_pos = i_pos_last - 2 * i_bulge;
            let mut v = v.rb_mut().col_mut(i_bulge);
            if i_pos == ilo {
                // Introduce bulge
                let h = a.rb().submatrix(ilo, ilo, 3, 3);

                let s1_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge);
                let s1_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge);
                let s2_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge - 1);
                let s2_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge - 1);
                lahqr_shiftcolumn(h, v.rb_mut(), (s1_re, s1_im), (s2_re, s2_im));

                debug_assert!(v.nrows() == 3);
                let mut head = v.read(0);
                let tail = v.rb_mut().subrows_mut(1, 2);
                let (tau, _) = make_householder_in_place(&mut head, tail);
                v.write(0, tau.faer_inv());
            } else {
                // Chase bulge down
                let mut h = a.rb_mut().submatrix_mut(i_pos - 1, i_pos - 1, 4, 4);
                let s1_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge);
                let s1_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge);
                let s2_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge - 1);
                let s2_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge - 1);
                move_bulge(h.rb_mut(), v.rb_mut(), (s1_re, s1_im), (s2_re, s2_im));
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
            if i_pos > ilo && (a[(i_pos, i_pos - 1)] != zero()) {
                let mut tst1 = abs1(a[(i_pos - 1, i_pos - 1)]).faer_add(abs1(a[(i_pos, i_pos)]));
                if tst1 == zero() {
                    if i_pos > ilo + 1 {
                        tst1 = tst1.faer_add(abs1(a.read(i_pos - 1, i_pos - 2)));
                    }
                    if i_pos > ilo + 2 {
                        tst1 = tst1.faer_add(abs1(a.read(i_pos - 1, i_pos - 3)));
                    }
                    if i_pos > ilo + 3 {
                        tst1 = tst1.faer_add(abs1(a.read(i_pos - 1, i_pos - 4)));
                    }
                    if i_pos < ihi - 1 {
                        tst1 = tst1.faer_add(abs1(a.read(i_pos + 1, i_pos)));
                    }
                    if i_pos < ihi - 2 {
                        tst1 = tst1.faer_add(abs1(a.read(i_pos + 2, i_pos)));
                    }
                    if i_pos < ihi - 3 {
                        tst1 = tst1.faer_add(abs1(a.read(i_pos + 3, i_pos)));
                    }
                }
                if abs1(a[(i_pos, i_pos - 1)]) < max(small_num, eps.faer_mul(tst1)) {
                    let ab = max(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
                    let ba = min(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
                    let aa = max(
                        abs1(a[(i_pos, i_pos)]),
                        abs1(a[(i_pos, i_pos)].faer_sub(a[(i_pos - 1, i_pos - 1)])),
                    );
                    let bb = min(
                        abs1(a[(i_pos, i_pos)]),
                        abs1(a[(i_pos, i_pos)].faer_sub(a[(i_pos - 1, i_pos - 1)])),
                    );
                    let s = aa.faer_add(ab);
                    if ba.faer_mul(ab.faer_div(s))
                        <= max(small_num, eps.faer_mul(bb.faer_mul(aa.faer_div(s))))
                    {
                        a.write(i_pos, i_pos - 1, zero());
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
                wh_slice.rb_mut(),
                Accum::Replace,
                u2.rb().adjoint(),
                a_slice.rb(),
                one(),
                parallelism,
            );
            a_slice.copy_from_with(wh_slice.rb());
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
                wv_slice.rb_mut(),
                Accum::Replace,
                a_slice.rb(),
                u2.rb(),
                one(),
                parallelism,
            );
            a_slice.copy_from_with(wv_slice.rb());
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
                wv_slice.rb_mut(),
                Accum::Replace,
                z_slice.rb(),
                u2.rb(),
                one(),
                parallelism,
            );
            z_slice.copy_from_with(wv_slice.rb());
            i += iblock;
        }
    }

    *i_pos_block = ilo + n_block - n_shifts;
}

#[inline(never)]
#[faer_macros::migrate]
fn move_bulges_down<T: RealField>(
    ilo: usize,
    ihi: usize,
    n_block_desired: usize,
    n_bulges: usize,
    n_shifts: usize,
    want_t: bool,
    mut a: MatMut<'_, T>,
    mut z: Option<MatMut<'_, T>>,
    mut u: MatMut<'_, T>,
    mut v: MatMut<'_, T>,
    mut wh: MatMut<'_, T>,
    mut wv: MatMut<'_, T>,
    s_re: ColRef<'_, T>,
    s_im: ColRef<'_, T>,

    i_pos_block: &mut usize,
    parallelism: Par,
) {
    let n = a.nrows();
    let eps = eps();
    let small_num = min_positive().faer_div(eps).faer_mul(from_f64(n as f64));

    while *i_pos_block + n_block_desired < ihi {
        // Number of positions each bulge will be moved down
        let n_pos = Ord::min(
            n_block_desired - n_shifts,
            ihi - n_shifts - 1 - *i_pos_block,
        );
        // Actual blocksize
        let n_block = n_shifts + n_pos;

        let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
        u2.fill(zero());
        u2.rb_mut().diagonal_mut().fill(one());

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
                let s1_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge);
                let s1_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge);
                let s2_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge - 1);
                let s2_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge - 1);
                move_bulge(h.rb_mut(), v.rb_mut(), (s1_re, s1_im), (s2_re, s2_im));

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
                if i_pos > ilo && (a[(i_pos, i_pos - 1)] != zero()) {
                    let mut tst1 =
                        abs1(a[(i_pos - 1, i_pos - 1)]).faer_add(abs1(a[(i_pos, i_pos)]));
                    if tst1 == zero() {
                        if i_pos > ilo + 1 {
                            tst1 = tst1.faer_add(abs1(a[(i_pos - 1, i_pos - 2)]));
                        }
                        if i_pos > ilo + 2 {
                            tst1 = tst1.faer_add(abs1(a[(i_pos - 1, i_pos - 3)]));
                        }
                        if i_pos > ilo + 3 {
                            tst1 = tst1.faer_add(abs1(a[(i_pos - 1, i_pos - 4)]));
                        }
                        if i_pos < ihi - 1 {
                            tst1 = tst1.faer_add(abs1(a[(i_pos + 1, i_pos)]));
                        }
                        if i_pos < ihi - 2 {
                            tst1 = tst1.faer_add(abs1(a[(i_pos + 2, i_pos)]));
                        }
                        if i_pos < ihi - 3 {
                            tst1 = tst1.faer_add(abs1(a[(i_pos + 3, i_pos)]));
                        }
                    }
                    if abs1(a[(i_pos, i_pos - 1)]) < max(small_num, eps.faer_mul(tst1)) {
                        let ab = max(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
                        let ba = min(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
                        let aa = max(
                            abs1(a[(i_pos, i_pos)]),
                            abs1(a[(i_pos, i_pos)].faer_sub(a[(i_pos - 1, i_pos - 1)])),
                        );
                        let bb = min(
                            abs1(a[(i_pos, i_pos)]),
                            abs1(a[(i_pos, i_pos)].faer_sub(a[(i_pos - 1, i_pos - 1)])),
                        );
                        let s = aa.faer_add(ab);
                        if ba.faer_mul(ab.faer_div(s))
                            <= max(small_num, eps.faer_mul(bb.faer_mul(aa.faer_div(s))))
                        {
                            a.write(i_pos, i_pos - 1, zero());
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
                    wh_slice.rb_mut(),
                    Accum::Replace,
                    u2.rb().adjoint(),
                    a_slice.rb(),
                    one(),
                    parallelism,
                );
                a_slice.copy_from_with(wh_slice.rb());
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
                    wv_slice.rb_mut(),
                    Accum::Replace,
                    a_slice.rb(),
                    u2.rb(),
                    one(),
                    parallelism,
                );
                a_slice.copy_from_with(wv_slice.rb());
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
                    wv_slice.rb_mut(),
                    Accum::Replace,
                    z_slice.rb(),
                    u2.rb(),
                    one(),
                    parallelism,
                );
                z_slice.copy_from_with(wv_slice.rb());
                i += iblock;
            }
        }

        *i_pos_block += n_pos;
    }
}

#[inline(never)]
#[faer_macros::migrate]
fn remove_bulges<T: RealField>(
    ilo: usize,
    ihi: usize,
    n_bulges: usize,
    n_shifts: usize,
    want_t: bool,
    mut a: MatMut<'_, T>,
    mut z: Option<MatMut<'_, T>>,
    mut u: MatMut<'_, T>,
    mut v: MatMut<'_, T>,
    mut wh: MatMut<'_, T>,
    mut wv: MatMut<'_, T>,
    s_re: ColRef<'_, T>,
    s_im: ColRef<'_, T>,

    i_pos_block: &mut usize,
    parallelism: Par,
) {
    let n = a.nrows();
    let eps = eps();
    let small_num = min_positive().faer_div(eps).faer_mul(from_f64(n as f64));

    {
        let n_block = ihi - *i_pos_block;

        let mut u2 = u.rb_mut().submatrix_mut(0, 0, n_block, n_block);
        u2.fill(zero());
        u2.rb_mut().diagonal_mut().fill(one());

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
                    let (tau, _) = make_householder_in_place(&mut beta, tail);
                    v.write(0, tau.faer_inv());
                    v.write(1, h.read(1));
                    h.write(0, beta);
                    h.write(1, zero());

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
                    let s1_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge);
                    let s1_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge);
                    let s2_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge - 1);
                    let s2_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge - 1);
                    move_bulge(h.rb_mut(), v.rb_mut(), (s1_re, s1_im), (s2_re, s2_im));

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
                    if i_pos > ilo && (a[(i_pos, i_pos - 1)] != zero()) {
                        let mut tst1 =
                            abs1(a[(i_pos - 1, i_pos - 1)]).faer_add(abs1(a[(i_pos, i_pos)]));
                        if tst1 == zero() {
                            if i_pos > ilo + 1 {
                                tst1 = tst1.faer_add(abs1(a[(i_pos - 1, i_pos - 2)]));
                            }
                            if i_pos > ilo + 2 {
                                tst1 = tst1.faer_add(abs1(a[(i_pos - 1, i_pos - 3)]));
                            }
                            if i_pos > ilo + 3 {
                                tst1 = tst1.faer_add(abs1(a[(i_pos - 1, i_pos - 4)]));
                            }
                            if i_pos < ihi - 1 {
                                tst1 = tst1.faer_add(abs1(a[(i_pos + 1, i_pos)]));
                            }
                            if i_pos < ihi - 2 {
                                tst1 = tst1.faer_add(abs1(a[(i_pos + 2, i_pos)]));
                            }
                            if i_pos < ihi - 3 {
                                tst1 = tst1.faer_add(abs1(a[(i_pos + 3, i_pos)]));
                            }
                        }
                        if abs1(a[(i_pos, i_pos - 1)]) < max(small_num, eps.faer_mul(tst1)) {
                            let ab = max(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
                            let ba = min(abs1(a[(i_pos, i_pos - 1)]), abs1(a[(i_pos - 1, i_pos)]));
                            let aa = max(
                                abs1(a[(i_pos, i_pos)]),
                                abs1(a[(i_pos, i_pos)].faer_sub(a[(i_pos - 1, i_pos - 1)])),
                            );
                            let bb = min(
                                abs1(a[(i_pos, i_pos)]),
                                abs1(a[(i_pos, i_pos)].faer_sub(a[(i_pos - 1, i_pos - 1)])),
                            );
                            let s = aa.faer_add(ab);
                            if ba.faer_mul(ab.faer_div(s))
                                <= max(small_num, eps.faer_mul(bb.faer_mul(aa.faer_div(s))))
                            {
                                a.write(i_pos, i_pos - 1, zero());
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
                let mut wh_slice =
                    wh.rb_mut()
                        .submatrix_mut(0, 0, a_slice.nrows(), a_slice.ncols());
                matmul(
                    wh_slice.rb_mut(),
                    Accum::Replace,
                    u2.rb().adjoint(),
                    a_slice.rb(),
                    one(),
                    parallelism,
                );
                a_slice.copy_from_with(wh_slice.rb());
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
                    wv_slice.rb_mut(),
                    Accum::Replace,
                    a_slice.rb(),
                    u2.rb(),
                    one(),
                    parallelism,
                );
                a_slice.copy_from_with(wv_slice.rb());
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
                    wv_slice.rb_mut(),
                    Accum::Replace,
                    z_slice.rb(),
                    u2.rb(),
                    one(),
                    parallelism,
                );
                z_slice.copy_from_with(wv_slice.rb());
                i += iblock;
            }
        }
    }
}

#[faer_macros::migrate]
pub fn multishift_qr<T: RealField>(
    want_t: bool,
    a: MatMut<'_, T>,
    z: Option<MatMut<'_, T>>,
    w_re: ColMut<'_, T>,
    w_im: ColMut<'_, T>,
    ilo: usize,
    ihi: usize,
    parallelism: Par,
    stack: &mut DynStack,
    params: EvdParams,
) -> (isize, usize, usize) {
    assert!(a.nrows() == a.ncols());
    assert!(ilo <= ihi);

    let n = a.nrows();
    let nh = ihi - ilo;

    assert!(w_re.nrows() == n);
    assert!(w_im.nrows() == n);

    if let Some(z) = z.rb() {
        assert!(z.nrows() == n);
        assert!(z.ncols() == n);
    }

    let mut a = a;
    let mut z = z;
    let mut w_re = w_re;
    let mut w_im = w_im;
    let mut stack = stack;

    let non_convergence_limit_window = 5;
    let non_convergence_limit_shift = 6;
    let dat1 = from_f64(0.75);
    let dat2 = from_f64(-0.4375);

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
        let err = lahqr(want_t, a, z, w_re, w_im, ilo, ihi);
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

    for iter in 0..itmax + 1 {
        if iter == itmax {
            // The QR algorithm failed to converge, return with error.
            info = istop as isize;
            break;
        }

        if ilo + 1 >= istop {
            if ilo + 1 == istop {
                w_re.write(ilo, a.read(ilo, ilo));
                w_im.write(ilo, zero());
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
            if a[(i, i - 1)] == zero() {
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
            if kwtop > istart + 2 && (abs1(a[(kwtop, kwtop - 1)]) > abs1(a[(kwtop - 1, kwtop - 2)]))
            {
                nw += 1;
            }
        }

        let (ls, ld) = aggressive_early_deflation(
            want_t,
            a.rb_mut(),
            z.rb_mut(),
            w_re.rb_mut(),
            w_im.rb_mut(),
            istart,
            istop,
            nw,
            parallelism,
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
        let mut i_shifts = istop - ls;

        if k_defl % non_convergence_limit_shift == 0 {
            for i in (i_shifts + 1..istop).rev().step_by(2) {
                if i >= ilo + 2 {
                    let ss = abs1(a[(i, i - 1)]).faer_add(abs1(a[(i - 1, i - 2)]));
                    let aa = from_real(dat1.faer_mul(ss)).faer_add(a.read(i, i));
                    let bb = from_real(ss);
                    let cc = from_real(dat2.faer_mul(ss));
                    let dd = copy(aa);
                    let (s1, s2) = lahqr_eig22(aa, bb, cc, dd);
                    w_re.write(i - 1, s1.0);
                    w_im.write(i - 1, s1.1);
                    w_re.write(i, s2.0);
                    w_im.write(i, s2.1);
                } else {
                    w_re.write(i, a.read(i, i));
                    w_re.write(i - 1, a.read(i, i));
                    w_im.write(i, zero());
                    w_im.write(i - 1, zero());
                }
            }
        } else {
            if ls <= ns / 2 {
                // Got ns/2 or fewer shifts? Then use multi/double shift qr to
                // get more
                let mut temp = a.rb_mut().submatrix_mut(n - ns, 0, ns, ns);
                let mut shifts_re = w_re.rb_mut().subrows_mut(istop - ns, ns);
                let mut shifts_im = w_im.rb_mut().subrows_mut(istop - ns, ns);
                let ierr = lahqr(
                    false,
                    temp.rb_mut(),
                    None,
                    shifts_re.rb_mut(),
                    shifts_im.rb_mut(),
                    0,
                    ns,
                ) as usize;

                ns = ns - ierr;

                if ns < 2 {
                    // In case of a rare QR failure, use eigenvalues
                    // of the trailing 2x2 submatrix
                    let aa = a.read(istop - 2, istop - 2);
                    let bb = a.read(istop - 2, istop - 1);
                    let cc = a.read(istop - 1, istop - 2);
                    let dd = a.read(istop - 1, istop - 1);
                    let (s1, s2) = lahqr_eig22(aa, bb, cc, dd);
                    w_re.write(istop - 2, s1.0);
                    w_im.write(istop - 2, s1.1);
                    w_re.write(istop - 1, s2.0);
                    w_im.write(istop - 1, s2.1);
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
                    if w_re[i].faer_abs().faer_add(w_im[i].faer_abs())
                        < w_re[i + 1].faer_abs().faer_add(w_im[i + 1].faer_abs())
                    {
                        sorted = false;
                        let wi = (w_re.read(i), w_im.read(i));
                        let wip1 = (w_re.read(i + 1), w_im.read(i + 1));
                        w_re.write(i, wip1.0);
                        w_im.write(i, wip1.1);
                        w_re.write(i + 1, wi.0);
                        w_im.write(i + 1, wi.1);
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
                if w_im[i] != w_im[i - 1].faer_neg() {
                    let tmp = (w_re.read(i), w_im.read(i));
                    w_re.write(i, w_re.read(i - 1));
                    w_im.write(i, w_im.read(i - 1));
                    w_re.write(i - 1, w_re.read(i - 2));
                    w_im.write(i - 1, w_im.read(i - 2));
                    w_re.write(i - 2, tmp.0);
                    w_im.write(i - 2, tmp.1);
                }
            }

            // Since we shuffled the shifts, we will only drop
            // Real shifts
            if ns > 1 && ns % 2 == 1 {
                ns -= 1;
            }
            i_shifts = istop - ns;
        }

        // If there are only two shifts and both are real
        // then use only one (helps avoid interference)
        if ns == 2 && (w_im[i_shifts] == zero()) {
            if (w_re[i_shifts].faer_sub(a[(istop - 1, istop - 1)])).faer_abs()
                < (w_re[i_shifts + 1].faer_sub(a[(istop - 1, istop - 1)])).faer_abs()
            {
                w_re.write(i_shifts + 1, w_re.read(i_shifts));
                w_im.write(i_shifts + 1, w_im.read(i_shifts));
            } else {
                w_re.write(i_shifts, w_re.read(i_shifts + 1));
                w_im.write(i_shifts, w_im.read(i_shifts + 1));
            }
        }

        let mut shifts_re = w_re.rb_mut().subrows_mut(i_shifts, ns);
        let mut shifts_im = w_im.rb_mut().subrows_mut(i_shifts, ns);

        multishift_qr_sweep(
            want_t,
            a.rb_mut(),
            z.rb_mut(),
            shifts_re.rb_mut(),
            shifts_im.rb_mut(),
            istart,
            istop,
            parallelism,
            stack.rb_mut(),
        );

        count_sweep += 1;
    }

    (info, count_aed, count_sweep)
}

#[faer_macros::migrate]
pub fn lahqr<T: RealField>(
    want_t: bool,
    a: MatMut<'_, T>,
    z: Option<MatMut<'_, T>>,
    w_re: ColMut<'_, T>,
    w_im: ColMut<'_, T>,
    ilo: usize,
    ihi: usize,
) -> isize {
    let epsilon = eps();
    let zero_threshold = min_positive();

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

    let one = one();
    let eps = epsilon;
    let small_num = zero_threshold.faer_div(eps);
    let non_convergence_limit = 10;
    let dat1 = from_f64(0.75);
    let dat2 = from_f64(-0.4375);

    if nh == 0 {
        return 0;
    }

    if nh == 1 {
        w_re.write(ilo, a.read(ilo, ilo));
        w_im.write(ilo, zero());
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

    stack_mat!(v, 3, 1, T);
    let mut v = v.rb_mut().col_mut(0);
    for iter in 0..itmax + 1 {
        if iter == itmax {
            return istop as isize;
        }

        if istart + 1 >= istop {
            if istart + 1 == istop {
                w_re.write(istart, a.read(istart, istart));
                w_im.write(istart, zero());
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
            if a[(i, i - 1)].faer_abs() < small_num {
                // A(i,i-1) is negligible, take i as new istart.
                a.write(i, i - 1, zero());
                istart = i;
                break;
            }

            let mut tst = a
                .read(i - 1, i - 1)
                .faer_abs()
                .faer_add(a.read(i, i).faer_abs());
            if tst == zero() {
                if i >= ilo + 2 {
                    tst = tst.faer_add(a.read(i - 1, i - 2).faer_abs());
                }
                if i + 1 < ihi {
                    tst = tst.faer_add(a.read(i + 1, i).faer_abs());
                }
            }

            if a[(i, i - 1)].faer_abs() <= eps.faer_mul(tst) {
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

                let ab = max(a[(i, i - 1)].faer_abs(), a[(i - 1, i)].faer_abs());
                let ba = min(a[(i, i - 1)].faer_abs(), a[(i - 1, i)].faer_abs());
                let aa = max(
                    a[(i, i)].faer_abs(),
                    (a[(i, i)].faer_sub(a[(i - 1, i - 1)])).faer_abs(),
                );
                let bb = min(
                    a[(i, i)].faer_abs(),
                    (a[(i, i)].faer_sub(a[(i - 1, i - 1)])).faer_abs(),
                );
                let s = aa.faer_add(ab);
                if ba.faer_mul(ab.faer_div(s))
                    <= max(small_num, eps.faer_mul(bb.faer_mul(aa.faer_div(s))))
                {
                    // A(i,i-1) is negligible, take i as new istart.
                    a.write(i, i - 1, zero());
                    istart = i;
                    break;
                }
            }
        }

        if istart + 2 >= istop {
            if istart + 1 == istop {
                // 1x1 block
                k_defl = 0;
                w_re.write(istart, a.read(istart, istart));
                w_im.write(istart, zero());
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
                    );
                let rot = JacobiRotation { c: cs, s: sn };

                a.write(istart, istart, a00);
                a.write(istart, istart + 1, a01);
                a.write(istart + 1, istart, a10);
                a.write(istart + 1, istart + 1, a11);

                w_re.write(istart, s1_re);
                w_im.write(istart, s1_im);
                w_re.write(istart + 1, s2_re);
                w_im.write(istart + 1, s2_im);

                // Apply the rotations from the normalization to the rest of the
                // matrix.
                if want_t {
                    if istart + 2 < istop_m {
                        let x = unsafe {
                            a.rb()
                                .row(istart)
                                .subcols(istart + 2, istop_m - (istart + 2))
                                .const_cast()
                                .transpose_mut()
                        };
                        let y = unsafe {
                            a.rb()
                                .row(istart + 1)
                                .subcols(istart + 2, istop_m - (istart + 2))
                                .const_cast()
                                .transpose_mut()
                        };

                        rot.apply_on_the_right_in_place((x, y));
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

                    rot.apply_on_the_right_in_place((x, y));
                }
                if let Some(z) = z.rb_mut() {
                    let x = unsafe { z.rb().col(istart).const_cast() };
                    let y = unsafe { z.rb().col(istart + 1).const_cast() };

                    rot.apply_on_the_right_in_place((x, y));
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
            let mut s = a.read(istop - 1, istop - 2).faer_abs();
            if istop > ilo + 2 {
                s = s.faer_add(a.read(istop - 2, istop - 3).faer_abs());
            };
            a00 = dat1.faer_mul(s).faer_add(a.read(istop - 1, istop - 1));
            a01 = dat2.faer_mul(s);
            a10 = s;
            a11 = copy(a00);
        } else {
            // Wilkinson shift
            a00 = a.read(istop - 2, istop - 2);
            a10 = a.read(istop - 1, istop - 2);
            a01 = a.read(istop - 2, istop - 1);
            a11 = a.read(istop - 1, istop - 1);
        }

        let (mut s1, mut s2) = lahqr_eig22(a00, a01, a10, a11);
        if s1.1 == zero() && s2.1 == zero() {
            // The eigenvalues are not complex conjugate, keep only the one
            // closest to A(istop-1, istop-1)
            if (s1.0.faer_sub(a[(istop - 1, istop - 1)])).faer_abs()
                <= (s2.0.faer_sub(a[(istop - 1, istop - 1)])).faer_abs()
            {
                s2 = (copy(s1.0), copy(s1.1));
            } else {
                s1 = (copy(s2.0), copy(s2.1));
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
                lahqr_shiftcolumn(
                    h,
                    v.rb_mut(),
                    (copy(s1.0), copy(s1.1)),
                    (copy(s2.0), copy(s2.1)),
                );
                let mut head = v.read(0);
                let (tau, _) = make_householder_in_place(&mut head, v.rb_mut().subrows_mut(1, 2));
                let tau = tau.faer_inv();

                let v0 = tau;
                let v1 = v.read(1);
                let v2 = v.read(2);

                let refsum =
                    (v0.faer_mul(a.read(i, i - 1))).faer_add(v1.faer_mul(a.read(i + 1, i - 1)));
                if (a[(i + 1, i - 1)].faer_sub(refsum.faer_mul(v1)))
                    .faer_abs()
                    .faer_add(refsum.faer_mul(v2).faer_abs())
                    <= eps.faer_mul(
                        a[(i, i - 1)]
                            .faer_abs()
                            .faer_add(a[(i, i + 1)].faer_abs())
                            .faer_add(a[(i + 1, i + 2)].faer_abs()),
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
                let mut x = v.rb_mut().subrows_mut(0, nr);
                lahqr_shiftcolumn(
                    h,
                    x.rb_mut(),
                    (copy(s1.0), copy(s1.1)),
                    (copy(s2.0), copy(s2.1)),
                );
                let mut beta = x.read(0);
                let tail = x.rb_mut().subrows_mut(1, nr - 1);
                (t1, _) = make_householder_in_place(&mut beta, tail);
                v.write(0, beta);
                t1 = t1.faer_inv();
                if i > istart {
                    a.write(i, i - 1, a.read(i, i - 1).faer_mul(one.faer_sub(t1)));
                }
            } else {
                v.write(0, a.read(i, i - 1));
                v.write(1, a.read(i + 1, i - 1));
                if nr == 3 {
                    v.write(2, a.read(i + 2, i - 1));
                }
                let mut beta = v.read(0);
                let tail = v.rb_mut().subrows_mut(1, nr - 1);
                (t1, _) = make_householder_in_place(&mut beta, tail);
                t1 = t1.faer_inv();
                v.write(0, copy(beta));
                a.write(i, i - 1, copy(beta));
                a.write(i + 1, i - 1, zero());
                if nr == 3 {
                    a.write(i + 2, i - 1, zero());
                }
            }

            // The following code applies the reflector we have just calculated.
            // We write this out instead of using larf because a direct loop is
            // more efficient for small reflectors.
            let v2 = v.read(1);
            let t2 = t1.faer_mul(v2);

            if nr == 3 {
                let v3 = v.read(2);
                let t3 = t1.faer_mul(v.read(2));

                // Apply G from the left to A
                for j in i..istop_m {
                    let sum = a
                        .read(i, j)
                        .faer_add(v2.faer_mul(a.read(i + 1, j)))
                        .faer_add(v3.faer_mul(a.read(i + 2, j)));
                    a.write(i, j, a.read(i, j).faer_sub(sum.faer_mul(t1)));
                    a.write(i + 1, j, a.read(i + 1, j).faer_sub(sum.faer_mul(t2)));
                    a.write(i + 2, j, a.read(i + 2, j).faer_sub(sum.faer_mul(t3)));
                }
                // Apply G from the right to A
                for j in istart_m..Ord::min(i + 4, istop) {
                    let sum = a
                        .read(j, i)
                        .faer_add(v2.faer_mul(a.read(j, i + 1)))
                        .faer_add(v3.faer_mul(a.read(j, i + 2)));
                    a.write(j, i, a.read(j, i).faer_sub(sum.faer_mul(t1)));
                    a.write(j, i + 1, a.read(j, i + 1).faer_sub(sum.faer_mul(t2)));
                    a.write(j, i + 2, a.read(j, i + 2).faer_sub(sum.faer_mul(t3)));
                }
                if let Some(mut z) = z.rb_mut() {
                    // Apply G to Z from the right
                    for j in 0..n {
                        let sum = z
                            .read(j, i)
                            .faer_add(v2.faer_mul(z.read(j, i + 1)))
                            .faer_add(v3.faer_mul(z.read(j, i + 2)));
                        z.write(j, i, z.read(j, i).faer_sub(sum.faer_mul(t1)));
                        z.write(j, i + 1, z.read(j, i + 1).faer_sub(sum.faer_mul(t2)));
                        z.write(j, i + 2, z.read(j, i + 2).faer_sub(sum.faer_mul(t3)));
                    }
                }
            } else {
                // Apply G from the left to A
                for j in i..istop_m {
                    let sum = a.read(i, j).faer_add(v2.faer_mul(a.read(i + 1, j)));
                    a.write(i, j, a.read(i, j).faer_sub(sum.faer_mul(t1)));
                    a.write(i + 1, j, a.read(i + 1, j).faer_sub(sum.faer_mul(t2)));
                }
                // Apply G from the right to A
                for j in istart_m..Ord::min(i + 3, istop) {
                    let sum = a.read(j, i).faer_add(v2.faer_mul(a.read(j, i + 1)));
                    a.write(j, i, a.read(j, i).faer_sub(sum.faer_mul(t1)));
                    a.write(j, i + 1, a.read(j, i + 1).faer_sub(sum.faer_mul(t2)));
                }
                if let Some(mut z) = z.rb_mut() {
                    // Apply G to Z from the right
                    for j in 0..n {
                        let sum = z.read(j, i).faer_add(v2.faer_mul(z.read(j, i + 1)));
                        z.write(j, i, z.read(j, i).faer_sub(sum.faer_mul(t1)));
                        z.write(j, i + 1, z.read(j, i + 1).faer_sub(sum.faer_mul(t2)));
                    }
                }
            }
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use dyn_stack::{DynStack, GlobalMemBuffer};

    use super::{lahqr, multishift_qr};
    use crate::{assert, linalg::evd::multishift_qr_scratch, prelude::*, utils::approx::*};

    #[test]
    fn test_5() {
        let h = [
            [-0.417, -0.056, -2.136, 1.64, -1.793],
            [-0.842, 0.503, -1.245, -1.058, -0.909],
            [0., 2.292, 0.042, -1.118, 0.539],
            [0., 0., 1.175, -0.748, 0.009],
            [0., 0., 0., -0.989, -0.339],
        ];
        let h = MatRef::from_row_major_array(&h);
        let mut q = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let mut q = MatMut::from_row_major_array_mut(&mut q);

        let mut w_re = Col::zeros(5);
        let mut w_im = Col::zeros(5);

        let mut t = h.cloned();
        lahqr(
            true,
            t.as_mut(),
            Some(q.as_mut()),
            w_re.as_mut(),
            w_im.as_mut(),
            0,
            5,
        );

        let h_reconstructed = &q * &t * q.transpose();

        let approx_eq = CwiseMat(ApproxEq::<f64>::eps());
        assert!(h ~ h_reconstructed);
    }

    #[test]
    fn test_5_2() {
        let h = [
            [0.10, 0.97, 0.19, 0.21, 0.84],
            [0.19, 0.21, 0.05, 0.83, 0.15],
            [0.00, 0.13, 0.05, 0.20, 0.14],
            [0.00, 0.00, 0.45, 0.44, 0.67],
            [0.00, 0.00, 0.00, 0.78, 0.27],
        ];
        let h = MatRef::from_row_major_array(&h);
        let mut q = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let mut q = MatMut::from_row_major_array_mut(&mut q);

        let mut w_re = Col::zeros(5);
        let mut w_im = Col::zeros(5);

        let mut t = h.cloned();
        lahqr(
            true,
            t.as_mut(),
            Some(q.as_mut()),
            w_re.as_mut(),
            w_im.as_mut(),
            0,
            5,
        );

        let h_reconstructed = &q * &t * q.transpose();

        let approx_eq = CwiseMat(ApproxEq::<f64>::eps());
        assert!(h ~ h_reconstructed);
    }

    #[test]
    fn test_n() {
        use rand::prelude::*;
        let rng = &mut StdRng::seed_from_u64(0);

        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32, 128, 256] {
            let mut h = Mat::zeros(n, n);
            for j in 0..n {
                for i in 0..n {
                    if i <= j + 1 {
                        h[(i, j)] = rng.gen::<f64>();
                    }
                }
            }
            if n <= 128 {
                let mut q = Mat::zeros(n, n);
                for i in 0..n {
                    q[(i, i)] = 1.0;
                }

                let mut w_re = Col::zeros(n);
                let mut w_im = Col::zeros(n);

                let mut t = h.as_ref().cloned();
                lahqr(
                    true,
                    t.as_mut(),
                    Some(q.as_mut()),
                    w_re.as_mut(),
                    w_im.as_mut(),
                    0,
                    n,
                );

                let h_reconstructed = &q * &t * q.transpose();

                let approx_eq = CwiseMat(ApproxEq::<f64>::eps());
                assert!(h ~ h_reconstructed);
            }
            {
                let mut q = Mat::zeros(n, n);
                for i in 0..n {
                    q[(i, i)] = 1.0;
                }

                let mut w_re = Col::zeros(n);
                let mut w_im = Col::zeros(n);

                let mut t = h.as_ref().cloned();
                multishift_qr(
                    true,
                    t.as_mut(),
                    Some(q.as_mut()),
                    w_re.as_mut(),
                    w_im.as_mut(),
                    0,
                    n,
                    Par::Seq,
                    DynStack::new(&mut GlobalMemBuffer::new(
                        multishift_qr_scratch::<f64>(
                            n,
                            n,
                            true,
                            true,
                            Par::Seq,
                            Default::default(),
                        )
                        .unwrap(),
                    )),
                    Default::default(),
                );

                for j in 0..n {
                    for i in 0..n {
                        if i > j + 1 {
                            t[(i, j)] = 0.0;
                        }
                    }
                }

                let h_reconstructed = &q * &t * q.transpose();

                let approx_eq = CwiseMat(ApproxEq::<f64>::eps());
                assert!(h ~ h_reconstructed);
            }
        }
    }
}
