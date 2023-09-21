// adapted from <T>LAPACK implementation
//
// https://github.com/tlapack/tlapack
// https://github.com/tlapack/tlapack/blob/master/include/tlapack/lapack/lahqr.hpp

use crate::{
    hessenberg::make_hessenberg_in_place,
    hessenberg_cplx_evd::{
        default_blocking_threshold, default_nibble_threshold, default_recommended_deflation_window,
        sqr_norm,
    },
};
use assert2::{assert, debug_assert};
use dyn_stack::DynStack;
use faer_core::{
    householder::{
        apply_block_householder_sequence_on_the_right_in_place_with_conj,
        apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
        make_householder_in_place,
    },
    mul::{inner_prod::inner_prod_with_conj, matmul},
    zip::Diag,
    zipped, Conj, MatMut, MatRef, Parallelism, RealField,
};
use reborrow::*;

pub use crate::hessenberg_cplx_evd::{multishift_qr_req, EvdParams};

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

fn abs1<E: RealField>(a: &E) -> E::Real {
    a.abs()
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

        std::mem::swap(&mut d, &mut a);
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

            if c != zero && b != zero && sign(&b) == sign(&c) {
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

fn rotg<E: RealField>(a: E, b: E, zero_threshold: E::Real) -> (E::Real, E, E) {
    let safmin = zero_threshold.clone();
    let safmax = zero_threshold.inv();

    let (c, s, r);

    let anorm = a.abs();
    let bnorm = b.abs();

    // quick return
    if bnorm == E::zero() {
        c = E::one();
        s = E::zero();
        r = E::one();
    } else if anorm == E::zero() {
        c = E::zero();
        s = E::one();
        r = b;
    } else {
        let scl = min(safmax, max(safmin, max(anorm.clone(), bnorm.clone())));
        let sigma = if anorm > bnorm { sign(&a) } else { sign(&b) };
        r = sigma
            .mul(&scl)
            .mul(&E::sqrt(&(a.div(&scl)).abs2().add(&(b.div(&scl)).abs2())));
        c = a.div(&r);
        s = b.div(&r);
    }

    (c, s, r)
}

fn lasy2<E: RealField>(
    tl: MatRef<'_, E>,
    tr: MatRef<'_, E>,
    b: MatRef<'_, E>,
    mut x: MatMut<'_, E>,
    epsilon: E,
    zero_threshold: E,
) -> E {
    let mut info = 0;
    let n1 = tl.nrows();
    let n2 = tr.nrows();

    let eps = epsilon.clone();
    let small_num = zero_threshold.div(&eps);

    debug_assert!(n1 == 2);
    debug_assert!(n2 == 2);

    let mut btmp = E::map(E::zero().into_units(), |zero| {
        [zero.clone(), zero.clone(), zero.clone(), zero]
    });
    let mut btmp = unsafe {
        MatMut::<E>::from_raw_parts(
            E::map(E::as_mut(&mut btmp), |array| array.as_mut_ptr()),
            4,
            1,
            1,
            4,
        )
    };

    let mut tmp = E::map(E::zero().into_units(), |zero| {
        [zero.clone(), zero.clone(), zero.clone(), zero]
    });
    let mut tmp = unsafe {
        MatMut::<E>::from_raw_parts(
            E::map(E::as_mut(&mut tmp), |array| array.as_mut_ptr()),
            4,
            1,
            1,
            4,
        )
    };

    let mut t16 = E::map(E::zero().into_units(), |zero| {
        [
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
        ]
    });
    let mut t16 = unsafe {
        MatMut::<E>::from_raw_parts(
            E::map(E::as_mut(&mut t16), |array| array.as_mut_ptr()),
            4,
            4,
            1,
            4,
        )
    };

    let mut jpiv = [0usize; 4];

    let mut smin = max(
        max(abs1(&tr.read(0, 0)), abs1(&tr.read(0, 1))),
        max(abs1(&tr.read(1, 0)), abs1(&tr.read(1, 1))),
    );
    smin = max(
        smin,
        max(
            max(abs1(&tl.read(0, 0)), abs1(&tl.read(0, 1))),
            max(abs1(&tl.read(1, 0)), abs1(&tl.read(1, 1))),
        ),
    );
    smin = max(eps.mul(&smin), small_num.clone());

    t16.write(0, 0, tl.read(0, 0).sub(&tr.read(0, 0)));
    t16.write(1, 1, tl.read(1, 1).sub(&tr.read(0, 0)));
    t16.write(2, 2, tl.read(0, 0).sub(&tr.read(1, 1)));
    t16.write(3, 3, tl.read(1, 1).sub(&tr.read(1, 1)));

    t16.write(0, 1, tl.read(0, 1));
    t16.write(1, 0, tl.read(1, 0));
    t16.write(2, 3, tl.read(0, 1));
    t16.write(3, 2, tl.read(1, 0));

    t16.write(0, 2, tr.read(1, 0).neg());
    t16.write(1, 3, tr.read(1, 0).neg());
    t16.write(2, 0, tr.read(0, 1).neg());
    t16.write(3, 1, tr.read(0, 1).neg());

    btmp.write(0, 0, b.read(0, 0));
    btmp.write(1, 0, b.read(1, 0));
    btmp.write(2, 0, b.read(0, 1));
    btmp.write(3, 0, b.read(1, 1));

    // Perform elimination with pivoting to solve 4x4 system
    let (mut ipsv, mut jpsv);
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        ipsv = i;
        jpsv = i;
        // Do pivoting to get largest pivot element
        let mut xmax = E::zero();
        for ip in i..4 {
            for jp in i..4 {
                if abs1(&t16.read(ip, jp)) >= xmax {
                    xmax = abs1(&t16.read(ip, jp));
                    ipsv = ip;
                    jpsv = jp;
                }
            }
        }
        if ipsv != i {
            faer_core::permutation::swap_rows(t16.rb_mut(), ipsv, i);

            let temp = btmp.read(i, 0);
            btmp.write(i, 0, btmp.read(ipsv, 0));
            btmp.write(ipsv, 0, temp);
        }
        if jpsv != i {
            faer_core::permutation::swap_cols(t16.rb_mut(), jpsv, i);
        }
        jpiv[i] = jpsv;
        if abs1(&t16.read(i, i)) < smin {
            info = 1;
            t16.write(i, i, smin.clone());
        }
        for j in i + 1..4 {
            t16.write(j, i, t16.read(j, i).div(&t16.read(i, i)));
            btmp.write(
                j,
                0,
                btmp.read(j, 0).sub(&t16.read(j, i).mul(&btmp.read(i, 0))),
            );
            for k in i + 1..4 {
                t16.write(
                    j,
                    k,
                    t16.read(j, k).sub(&t16.read(j, i).mul(&t16.read(i, k))),
                );
            }
        }
    }

    if abs1(&t16.read(3, 3)) < smin {
        info = 1;
        t16.write(3, 3, smin);
    }
    let mut scale = E::one();
    let eight = E::from_f64(8.0);

    if (eight.mul(&small_num)).mul(&abs1(&btmp.read(0, 0))) > abs1(&t16.read(0, 0))
        || (eight.mul(&small_num)).mul(&abs1(&btmp.read(1, 0))) > abs1(&t16.read(1, 1))
        || (eight.mul(&small_num)).mul(&abs1(&btmp.read(2, 0))) > abs1(&t16.read(2, 2))
        || (eight.mul(&small_num)).mul(&abs1(&btmp.read(3, 0))) > abs1(&t16.read(3, 3))
    {
        scale = eight.inv().div(&max(
            max(abs1(&btmp.read(0, 0)), abs1(&btmp.read(1, 0))),
            max(abs1(&btmp.read(2, 0)), abs1(&btmp.read(3, 0))),
        ));
        btmp.write(0, 0, btmp.read(0, 0).mul(&scale));
        btmp.write(1, 0, btmp.read(1, 0).mul(&scale));
        btmp.write(2, 0, btmp.read(2, 0).mul(&scale));
        btmp.write(3, 0, btmp.read(3, 0).mul(&scale));
    }
    for i in 0..4 {
        let k = 3 - i;
        let temp = t16.read(k, k).inv();
        tmp.write(k, 0, btmp.read(k, 0).mul(&temp));
        for j in k + 1..4 {
            tmp.write(
                k,
                0,
                tmp.read(k, 0)
                    .sub(&temp.mul(&t16.read(k, j)).mul(&tmp.read(j, 0))),
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

fn schur_move<E: RealField>(
    mut a: MatMut<E>,
    mut q: Option<MatMut<E>>,
    mut ifst: usize,
    ilst: &mut usize,
    epsilon: E::Real,
    zero_threshold: E::Real,
) -> isize {
    let n = a.nrows();

    // Quick return
    if n == 0 {
        return 0;
    }

    // Check if ifst points to the middle of a 2x2 block
    if ifst > 0 && (a.read(ifst, ifst - 1) != E::zero()) {
        ifst -= 1;
    }

    // Size of the current block, can be either 1, 2
    let mut nbf = 1;
    if ifst < n - 1 && (a.read(ifst + 1, ifst) != E::zero()) {
        nbf = 2;
    }

    // Check if ilst points to the middle of a 2x2 block
    if *ilst > 0 && (a.read(*ilst, *ilst - 1) != E::zero()) {
        *ilst -= 1;
    }

    // Size of the final block, can be either 1, 2
    let mut nbl = 1;
    if (*ilst < n - 1) && (a.read(*ilst + 1, *ilst) != E::zero()) {
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
            if (here + nbf + 1 < n) && (a.read(here + nbf + 1, here + nbf) != E::zero()) {
                nbnext = 2;
            }

            let ierr = schur_swap(
                a.rb_mut(),
                q.rb_mut(),
                here,
                nbf,
                nbnext,
                epsilon.clone(),
                zero_threshold.clone(),
            );
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
            if here > 1 && (a.read(here - 1, here - 2) != E::zero()) {
                nbnext = 2;
            }

            let ierr = schur_swap(
                a.rb_mut(),
                q.rb_mut(),
                here - nbnext,
                nbnext,
                nbf,
                epsilon.clone(),
                zero_threshold.clone(),
            );
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

fn schur_swap<E: RealField>(
    mut a: MatMut<E>,
    mut q: Option<MatMut<E>>,
    j0: usize,
    n1: usize,
    n2: usize,
    epsilon: E,
    zero_threshold: E,
) -> isize {
    let n = a.nrows();

    let j1 = j0 + 1;
    let j2 = j0 + 2;
    let j3 = j0 + 3;

    // Check if the 2x2 eigenvalue blocks consist of 2 1x1 blocks
    // If so, treat them separately
    if n1 == 2 && (a.read(j1, j0) == E::zero()) {
        // only 2x2 swaps can fail, so we don't need to check for error
        schur_swap(
            a.rb_mut(),
            q.rb_mut(),
            j1,
            1,
            n2,
            epsilon.clone(),
            zero_threshold.clone(),
        );
        schur_swap(
            a.rb_mut(),
            q.rb_mut(),
            j0,
            1,
            n2,
            epsilon.clone(),
            zero_threshold.clone(),
        );
        return 0;
    }
    if n2 == 2 && a.read(j0 + n1 + 1, j0 + n1) == E::zero() {
        // only 2x2 swaps can fail, so we don't need to check for error
        schur_swap(
            a.rb_mut(),
            q.rb_mut(),
            j0,
            n1,
            1,
            epsilon.clone(),
            zero_threshold.clone(),
        );
        schur_swap(
            a.rb_mut(),
            q.rb_mut(),
            j1,
            n1,
            1,
            epsilon.clone(),
            zero_threshold.clone(),
        );
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
        let temp2 = t11.sub(&t00);
        let (cs, sn, _) = rotg(temp, temp2, zero_threshold.clone());

        a.write(j1, j1, t00);
        a.write(j0, j0, t11);

        // Apply transformation from the left
        if j2 < n {
            let row1 = unsafe { a.rb().row(j0).subcols(j2, n - j2).const_cast() };
            let row2 = unsafe { a.rb().row(j1).subcols(j2, n - j2).const_cast() };
            rot(row1.transpose(), row2.transpose(), cs.clone(), sn.clone());
        }
        // Apply transformation from the right
        if j0 > 0 {
            let col1 = unsafe { a.rb().col(j0).subrows(0, j0).const_cast() };
            let col2 = unsafe { a.rb().col(j1).subrows(0, j0).const_cast() };
            rot(col1, col2, cs.clone(), sn.clone());
        }
        if let Some(q) = q.rb_mut() {
            let col1 = unsafe { q.rb().col(j0).const_cast() };
            let col2 = unsafe { q.rb().col(j1).const_cast() };
            rot(col1, col2, cs, sn);
        }
    }
    if n1 == 1 && n2 == 2 {
        //
        // Swap 1-by-1 block with 2-by-2 block
        //
        let mut b_storage = E::map(E::zero().into_units(), |zero| {
            [
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero,
            ]
        });
        let b_ptr = E::map(E::as_mut(&mut b_storage), |array| array.as_mut_ptr());
        let mut b = unsafe { MatMut::from_raw_parts(b_ptr, 3, 2, 1, 3) };

        b.write(0, 0, a.read(j0, j1));
        b.write(1, 0, a.read(j1, j1).sub(&a.read(j0, j0)));
        b.write(2, 0, a.read(j2, j1));
        b.write(0, 1, a.read(j0, j2));
        b.write(1, 1, a.read(j1, j2));
        b.write(2, 1, a.read(j2, j2).sub(&a.read(j0, j0)));

        // Make B upper triangular
        let mut v1 = b.rb_mut().col(0);
        let head = v1.read(0, 0);
        let tail = v1.rb_mut().subrows(1, 2);
        let tail_sqr_norm = tail.read(0, 0).abs2().add(&tail.read(1, 0).abs2());
        let (tau1, beta1) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
        let tau1 = tau1.inv();
        v1.write(0, 0, beta1);
        let v11 = b.read(1, 0);
        let v12 = b.read(2, 0);

        let sum = b
            .read(0, 1)
            .add(&v11.mul(&b.read(1, 1)))
            .add(&v12.mul(&b.read(2, 1)));

        b.write(0, 1, b.read(0, 1).sub(&sum.mul(&tau1)));
        b.write(1, 1, b.read(1, 1).sub(&sum.mul(&tau1).mul(&v11)));
        b.write(2, 1, b.read(2, 1).sub(&sum.mul(&tau1).mul(&v12)));

        let mut v2 = b.rb_mut().col(1).subrows(1, 2);
        let head = v2.read(0, 0);
        let tail = v2.rb_mut().subrows(1, 1);
        let tail_sqr_norm = tail.read(0, 0).abs2();
        let (tau2, beta2) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
        let tau2 = tau2.inv();
        v2.write(0, 0, beta2);
        let v21 = v2.read(1, 0);

        //
        // Apply reflections to A and Q
        //

        // Reflections from the left
        for j in j0..n {
            let sum = a
                .read(j0, j)
                .add(&v11.mul(&a.read(j1, j)))
                .add(&v12.mul(&a.read(j2, j)));
            a.write(j0, j, a.read(j0, j).sub(&sum.mul(&tau1)));
            a.write(j1, j, a.read(j1, j).sub(&sum.mul(&tau1).mul(&v11)));
            a.write(j2, j, a.read(j2, j).sub(&sum.mul(&tau1).mul(&v12)));

            let sum = a.read(j1, j).add(&v21.mul(&a.read(j2, j)));
            a.write(j1, j, a.read(j1, j).sub(&sum.mul(&tau2)));
            a.write(j2, j, a.read(j2, j).sub(&sum.mul(&tau2).mul(&v21)));
        }
        // Reflections from the right
        for j in 0..j3 {
            let sum = a
                .read(j, j0)
                .add(&v11.mul(&a.read(j, j1)))
                .add(&v12.mul(&a.read(j, j2)));
            a.write(j, j0, a.read(j, j0).sub(&sum.mul(&tau1)));
            a.write(j, j1, a.read(j, j1).sub(&sum.mul(&tau1).mul(&v11)));
            a.write(j, j2, a.read(j, j2).sub(&sum.mul(&tau1).mul(&v12)));

            let sum = a.read(j, j1).add(&v21.mul(&a.read(j, j2)));
            a.write(j, j1, a.read(j, j1).sub(&sum.mul(&tau2)));
            a.write(j, j2, a.read(j, j2).sub(&sum.mul(&tau2).mul(&v21)));
        }

        if let Some(mut q) = q.rb_mut() {
            for j in 0..n {
                let sum = q
                    .read(j, j0)
                    .add(&v11.mul(&q.read(j, j1)))
                    .add(&v12.mul(&q.read(j, j2)));
                q.write(j, j0, q.read(j, j0).sub(&sum.mul(&tau1)));
                q.write(j, j1, q.read(j, j1).sub(&sum.mul(&tau1).mul(&v11)));
                q.write(j, j2, q.read(j, j2).sub(&sum.mul(&tau1).mul(&v12)));

                let sum = q.read(j, j1).add(&v21.mul(&q.read(j, j2)));
                q.write(j, j1, q.read(j, j1).sub(&sum.mul(&tau2)));
                q.write(j, j2, q.read(j, j2).sub(&sum.mul(&tau2).mul(&v21)));
            }
        }

        a.write(j2, j0, E::zero());
        a.write(j2, j1, E::zero());
    }

    if n1 == 2 && n2 == 1 {
        //
        // Swap 2-by-2 block with 1-by-1 block
        //

        let mut b_storage = E::map(E::zero().into_units(), |zero| {
            [
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero,
            ]
        });
        let b_ptr = E::map(E::as_mut(&mut b_storage), |array| array.as_mut_ptr());
        let mut b = unsafe { MatMut::from_raw_parts(b_ptr, 3, 2, 1, 3) };

        b.write(0, 0, a.read(j1, j2));
        b.write(1, 0, a.read(j1, j1).sub(&a.read(j2, j2)));
        b.write(2, 0, a.read(j1, j0));
        b.write(0, 1, a.read(j0, j2));
        b.write(1, 1, a.read(j0, j1));
        b.write(2, 1, a.read(j0, j0).sub(&a.read(j2, j2)));

        // Make B upper triangular
        let mut v1 = b.rb_mut().col(0);
        let head = v1.read(0, 0);
        let tail = v1.rb_mut().subrows(1, 2);
        let tail_sqr_norm = tail.read(0, 0).abs2().add(&tail.read(1, 0).abs2());
        let (tau1, beta1) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
        let tau1 = tau1.inv();
        v1.write(0, 0, beta1);
        let v11 = v1.read(1, 0);
        let v12 = v1.read(2, 0);

        let sum = b
            .read(0, 1)
            .add(&v11.mul(&b.read(1, 1)))
            .add(&v12.mul(&b.read(2, 1)));

        b.write(0, 1, b.read(0, 1).sub(&sum.mul(&tau1)));
        b.write(1, 1, b.read(1, 1).sub(&sum.mul(&tau1).mul(&v11)));
        b.write(2, 1, b.read(2, 1).sub(&sum.mul(&tau1).mul(&v12)));

        let mut v2 = b.rb_mut().col(1).subrows(1, 2);
        let head = v2.read(0, 0);
        let tail = v2.rb_mut().subrows(1, 1);
        let tail_sqr_norm = tail.read(0, 0).abs2();
        let (tau2, beta2) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
        let tau2 = tau2.inv();
        v2.write(0, 0, beta2);
        let v21 = v2.read(1, 0);

        //
        // Apply reflections to A and Q
        //

        // Reflections from the left
        for j in j0..n {
            let sum = a
                .read(j2, j)
                .add(&v11.mul(&a.read(j1, j)))
                .add(&v12.mul(&a.read(j0, j)));
            a.write(j2, j, a.read(j2, j).sub(&sum.mul(&tau1)));
            a.write(j1, j, a.read(j1, j).sub(&sum.mul(&tau1).mul(&v11)));
            a.write(j0, j, a.read(j0, j).sub(&sum.mul(&tau1).mul(&v12)));

            let sum = a.read(j1, j).add(&v21.mul(&a.read(j0, j)));
            a.write(j1, j, a.read(j1, j).sub(&sum.mul(&tau2)));
            a.write(j0, j, a.read(j0, j).sub(&sum.mul(&tau2).mul(&v21)));
        }
        // Reflections from the right
        for j in 0..j3 {
            let sum = a
                .read(j, j2)
                .add(&v11.mul(&a.read(j, j1)))
                .add(&v12.mul(&a.read(j, j0)));
            a.write(j, j2, a.read(j, j2).sub(&sum.mul(&tau1)));
            a.write(j, j1, a.read(j, j1).sub(&sum.mul(&tau1).mul(&v11)));
            a.write(j, j0, a.read(j, j0).sub(&sum.mul(&tau1).mul(&v12)));

            let sum = a.read(j, j1).add(&v21.mul(&a.read(j, j0)));
            a.write(j, j1, a.read(j, j1).sub(&sum.mul(&tau2)));
            a.write(j, j0, a.read(j, j0).sub(&sum.mul(&tau2).mul(&v21)));
        }

        if let Some(mut q) = q.rb_mut() {
            for j in 0..n {
                let sum = q
                    .read(j, j2)
                    .add(&v11.mul(&q.read(j, j1)))
                    .add(&v12.mul(&q.read(j, j0)));
                q.write(j, j2, q.read(j, j2).sub(&sum.mul(&tau1)));
                q.write(j, j1, q.read(j, j1).sub(&sum.mul(&tau1).mul(&v11)));
                q.write(j, j0, q.read(j, j0).sub(&sum.mul(&tau1).mul(&v12)));

                let sum = q.read(j, j1).add(&v21.mul(&q.read(j, j0)));
                q.write(j, j1, q.read(j, j1).sub(&sum.mul(&tau2)));
                q.write(j, j0, q.read(j, j0).sub(&sum.mul(&tau2).mul(&v21)));
            }
        }

        a.write(j1, j0, E::zero());
        a.write(j2, j0, E::zero());
    }

    if n1 == 2 && n2 == 2 {
        let mut d_storage = E::map(E::zero().into_units(), |zero| {
            [
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero,
            ]
        });
        let d_ptr = E::map(E::as_mut(&mut d_storage), |array| array.as_mut_ptr());
        let mut d = unsafe { MatMut::from_raw_parts(d_ptr, 4, 4, 1, 4) };

        let ad_slice = a.rb().submatrix(j0, j0, 4, 4);
        d.clone_from(ad_slice);
        let mut dnorm = E::zero();
        zipped!(d.rb()).for_each(|d| dnorm = max(dnorm.clone(), d.read().abs()));

        let eps = epsilon.clone();
        let small_num = zero_threshold.div(&eps);
        let thresh = max(E::from_f64(10.0).mul(&eps).mul(&dnorm), small_num);

        let mut v_storage = E::map(E::zero().into_units(), |zero| {
            [
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero,
            ]
        });
        let v_ptr = E::map(E::as_mut(&mut v_storage), |array| array.as_mut_ptr());
        let mut v = unsafe { MatMut::<E>::from_raw_parts(v_ptr, 4, 2, 1, 4) };

        let mut x = v.rb_mut().submatrix(0, 0, 2, 2);
        let [tl, b, _, tr] = d.rb().split_at(2, 2);

        let scale = lasy2(
            tl,
            tr,
            b,
            x.rb_mut(),
            epsilon.clone(),
            zero_threshold.clone(),
        );

        v.write(2, 0, scale.neg());
        v.write(2, 1, E::zero());
        v.write(3, 0, E::zero());
        v.write(3, 1, scale.neg());

        // Make V upper triangular
        let mut v1 = v.rb_mut().col(0);
        let head = v1.read(0, 0);
        let tail = v1.rb_mut().subrows(1, 3);
        let tail_sqr_norm = tail
            .read(0, 0)
            .abs2()
            .add(&tail.read(1, 0).abs2())
            .add(&tail.read(2, 0).abs2());
        let (tau1, beta1) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
        let tau1 = tau1.inv();
        v1.write(0, 0, beta1);
        let v11 = v1.read(1, 0);
        let v12 = v1.read(2, 0);
        let v13 = v1.read(3, 0);

        let sum = v
            .read(0, 1)
            .add(&v11.mul(&v.read(1, 1)))
            .add(&v12.mul(&v.read(2, 1)))
            .add(&v13.mul(&v.read(3, 1)));

        v.write(0, 1, v.read(0, 1).sub(&sum.mul(&tau1)));
        v.write(1, 1, v.read(1, 1).sub(&sum.mul(&tau1).mul(&v11)));
        v.write(2, 1, v.read(2, 1).sub(&sum.mul(&tau1).mul(&v12)));
        v.write(3, 1, v.read(3, 1).sub(&sum.mul(&tau1).mul(&v13)));

        let mut v2 = v.rb_mut().col(1).subrows(1, 3);
        let head = v2.read(0, 0);
        let tail = v2.rb_mut().subrows(1, 2);
        let tail_sqr_norm = tail.read(0, 0).abs2().add(&tail.read(1, 0).abs2());
        let (tau2, beta2) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
        let tau2 = tau2.inv();
        v2.write(0, 0, beta2);

        let v21 = v2.read(1, 0);
        let v22 = v2.read(2, 0);

        // Apply reflections to D to check error
        for j in 0..4 {
            let sum = d
                .read(0, j)
                .add(&v11.mul(&d.read(1, j)))
                .add(&v12.mul(&d.read(2, j)))
                .add(&v13.mul(&d.read(3, j)));
            d.write(0, j, d.read(0, j).sub(&sum.mul(&tau1)));
            d.write(1, j, d.read(1, j).sub(&sum.mul(&tau1).mul(&v11)));
            d.write(2, j, d.read(2, j).sub(&sum.mul(&tau1).mul(&v12)));
            d.write(3, j, d.read(3, j).sub(&sum.mul(&tau1).mul(&v13)));

            let sum = d
                .read(1, j)
                .add(&v21.mul(&d.read(2, j)))
                .add(&v22.mul(&d.read(3, j)));

            d.write(1, j, d.read(1, j).sub(&sum.mul(&tau2)));
            d.write(2, j, d.read(2, j).sub(&sum.mul(&tau2).mul(&v21)));
            d.write(3, j, d.read(3, j).sub(&sum.mul(&tau2).mul(&v22)));
        }
        for j in 0..4 {
            let sum = d
                .read(j, 0)
                .add(&v11.mul(&d.read(j, 1)))
                .add(&v12.mul(&d.read(j, 2)))
                .add(&v13.mul(&d.read(j, 3)));
            d.write(j, 0, d.read(j, 0).sub(&sum.mul(&tau1)));
            d.write(j, 1, d.read(j, 1).sub(&sum.mul(&tau1).mul(&v11)));
            d.write(j, 2, d.read(j, 2).sub(&sum.mul(&tau1).mul(&v12)));
            d.write(j, 3, d.read(j, 3).sub(&sum.mul(&tau1).mul(&v13)));

            let sum = d
                .read(j, 1)
                .add(&v21.mul(&d.read(j, 2)))
                .add(&v22.mul(&d.read(j, 3)));
            d.write(j, 1, d.read(j, 1).sub(&sum.mul(&tau2)));
            d.write(j, 2, d.read(j, 2).sub(&sum.mul(&tau2).mul(&v21)));
            d.write(j, 3, d.read(j, 3).sub(&sum.mul(&tau2).mul(&v22)));
        }

        if max(
            max(d.read(2, 0).abs(), d.read(2, 1).abs()),
            max(d.read(3, 0).abs(), d.read(3, 1).abs()),
        ) > thresh
        {
            return 1;
        }

        // Reflections from the left
        for j in j0..n {
            let sum = a
                .read(j0, j)
                .add(&v11.mul(&a.read(j1, j)))
                .add(&v12.mul(&a.read(j2, j)))
                .add(&v13.mul(&a.read(j3, j)));

            a.write(j0, j, a.read(j0, j).sub(&sum.mul(&tau1)));
            a.write(j1, j, a.read(j1, j).sub(&sum.mul(&tau1).mul(&v11)));
            a.write(j2, j, a.read(j2, j).sub(&sum.mul(&tau1).mul(&v12)));
            a.write(j3, j, a.read(j3, j).sub(&sum.mul(&tau1).mul(&v13)));

            let sum = a
                .read(j1, j)
                .add(&v21.mul(&a.read(j2, j)))
                .add(&v22.mul(&a.read(j3, j)));

            a.write(j1, j, a.read(j1, j).sub(&sum.mul(&tau2)));
            a.write(j2, j, a.read(j2, j).sub(&sum.mul(&tau2).mul(&v21)));
            a.write(j3, j, a.read(j3, j).sub(&sum.mul(&tau2).mul(&v22)));
        }
        // Reflections from the right
        for j in 0..j0 + 4 {
            let sum = a
                .read(j, j0)
                .add(&v11.mul(&a.read(j, j1)))
                .add(&v12.mul(&a.read(j, j2)))
                .add(&v13.mul(&a.read(j, j3)));
            a.write(j, j0, a.read(j, j0).sub(&sum.mul(&tau1)));
            a.write(j, j1, a.read(j, j1).sub(&sum.mul(&tau1).mul(&v11)));
            a.write(j, j2, a.read(j, j2).sub(&sum.mul(&tau1).mul(&v12)));
            a.write(j, j3, a.read(j, j3).sub(&sum.mul(&tau1).mul(&v13)));

            let sum = a
                .read(j, j1)
                .add(&v21.mul(&a.read(j, j2)))
                .add(&v22.mul(&a.read(j, j3)));
            a.write(j, j1, a.read(j, j1).sub(&sum.mul(&tau2)));
            a.write(j, j2, a.read(j, j2).sub(&sum.mul(&tau2).mul(&v21)));
            a.write(j, j3, a.read(j, j3).sub(&sum.mul(&tau2).mul(&v22)));
        }

        if let Some(mut q) = q.rb_mut() {
            for j in 0..n {
                let sum = q
                    .read(j, j0)
                    .add(&v11.mul(&q.read(j, j1)))
                    .add(&v12.mul(&q.read(j, j2)))
                    .add(&v13.mul(&q.read(j, j3)));
                q.write(j, j0, q.read(j, j0).sub(&sum.mul(&tau1)));
                q.write(j, j1, q.read(j, j1).sub(&sum.mul(&tau1).mul(&v11)));
                q.write(j, j2, q.read(j, j2).sub(&sum.mul(&tau1).mul(&v12)));
                q.write(j, j3, q.read(j, j3).sub(&sum.mul(&tau1).mul(&v13)));

                let sum = q
                    .read(j, j1)
                    .add(&v21.mul(&q.read(j, j2)))
                    .add(&v22.mul(&q.read(j, j3)));
                q.write(j, j1, q.read(j, j1).sub(&sum.mul(&tau2)));
                q.write(j, j2, q.read(j, j2).sub(&sum.mul(&tau2).mul(&v21)));
                q.write(j, j3, q.read(j, j3).sub(&sum.mul(&tau2).mul(&v22)));
            }
        }

        a.write(j2, j0, E::zero());
        a.write(j2, j1, E::zero());
        a.write(j3, j0, E::zero());
        a.write(j3, j1, E::zero());
    }

    // Standardize the 2x2 Schur blocks (if any)
    if n2 == 2 {
        let ((a00, a01, a10, a11), _, _, (cs, sn)) = lahqr_schur22(
            a.read(j0, j0),
            a.read(j0, j1),
            a.read(j1, j0),
            a.read(j1, j1),
            epsilon.clone(),
            zero_threshold.clone(),
        ); // Apply transformation from the left

        a.write(j0, j0, a00);
        a.write(j0, j1, a01);
        a.write(j1, j0, a10);
        a.write(j1, j1, a11);

        if j2 < n {
            let row1 = unsafe { a.rb().row(j0).subcols(j2, n - j2).const_cast() };
            let row2 = unsafe { a.rb().row(j1).subcols(j2, n - j2).const_cast() };

            rot(row1.transpose(), row2.transpose(), cs.clone(), sn.clone());
        }
        // Apply transformation from the right
        if j0 > 0 {
            let col1 = unsafe { a.rb().col(j0).subrows(0, j0).const_cast() };
            let col2 = unsafe { a.rb().col(j1).subrows(0, j0).const_cast() };
            rot(col1, col2, cs.clone(), sn.clone());
        }
        if let Some(q) = q.rb_mut() {
            let col1 = unsafe { q.rb().col(j0).const_cast() };
            let col2 = unsafe { q.rb().col(j1).const_cast() };
            rot(col1, col2, cs, sn);
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
            epsilon.clone(),
            zero_threshold.clone(),
        ); // Apply transformation from the left

        a.write(j0, j0, a00);
        a.write(j0, j1, a01);
        a.write(j1, j0, a10);
        a.write(j1, j1, a11);

        if j2 < n {
            let row1 = unsafe { a.rb().row(j0).subcols(j2, n - j2).const_cast() };
            let row2 = unsafe { a.rb().row(j1).subcols(j2, n - j2).const_cast() };

            rot(row1.transpose(), row2.transpose(), cs.clone(), sn.clone());
        }
        // Apply transformation from the right
        if j0 > 0 {
            let col1 = unsafe { a.rb().col(j0).subrows(0, j0).const_cast() };
            let col2 = unsafe { a.rb().col(j1).subrows(0, j0).const_cast() };
            rot(col1, col2, cs.clone(), sn.clone());
        }
        if let Some(q) = q.rb_mut() {
            let col1 = unsafe { q.rb().col(j0).const_cast() };
            let col2 = unsafe { q.rb().col(j1).const_cast() };
            rot(col1, col2, cs, sn);
        }
    }

    0
}

fn aggressive_early_deflation<E: RealField>(
    want_t: bool,
    mut a: MatMut<'_, E>,
    mut z: Option<MatMut<'_, E>>,
    mut s_re: MatMut<'_, E>,
    mut s_im: MatMut<'_, E>,
    ilo: usize,
    ihi: usize,
    nw: usize,
    epsilon: E::Real,
    zero_threshold: E::Real,
    parallelism: Parallelism,
    mut stack: DynStack<'_>,
    params: EvdParams,
) -> (usize, usize) {
    let n = a.nrows();

    // Because we will use the lower triangular part of A as workspace,
    // We have a maximum window size
    let nw_max = (n - 3) / 3;
    let eps = epsilon.clone();
    let small_num = zero_threshold.div(&eps).mul(&E::Real::from_f64(n as f64));

    // Size of the deflation window
    let jw = Ord::min(Ord::min(nw, ihi - ilo), nw_max);
    // First row index in the deflation window
    let kwtop = ihi - jw;

    // s is the value just outside the window. It determines the spike
    // together with the orthogonal schur factors.
    let mut s_spike = if kwtop == ilo {
        E::zero()
    } else {
        a.read(kwtop, kwtop - 1)
    };

    if kwtop + 1 == ihi {
        // 1x1 deflation window, not much to do
        s_re.write(kwtop, 0, a.read(kwtop, kwtop));
        s_im.write(kwtop, 0, E::zero());
        let mut ns = 1;
        let mut nd = 0;
        if s_spike.abs() <= max(small_num, eps.mul(&a.read(kwtop, kwtop).abs())) {
            ns = 0;
            nd = 1;
            if kwtop > ilo {
                a.write(kwtop, kwtop - 1, E::zero());
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
    zipped!(tw.rb_mut()).for_each_triangular_lower(Diag::Include, |mut x| x.write(E::zero()));
    for j in 0..jw {
        for i in 0..Ord::min(j + 2, jw) {
            tw.write(i, j, a_window.read(i, j));
        }
    }
    v.fill_with_zero();
    v.rb_mut().diagonal().fill_with_constant(E::one());

    let infqr = if jw
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
            epsilon.clone(),
            zero_threshold.clone(),
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
            epsilon.clone(),
            zero_threshold.clone(),
            parallelism,
            stack.rb_mut(),
            params,
        )
        .0;
        for j in 0..jw {
            for i in j + 2..jw {
                tw.write(i, j, E::zero());
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
        if ns > 1 && tw.read(ns - 1, ns - 2) != E::zero() {
            bulge = true;
        }

        if !bulge {
            // 1x1 eigenvalue block
            #[allow(clippy::disallowed_names)]
            let mut foo = tw.read(ns - 1, ns - 1).abs();
            if foo == E::zero() {
                foo = s_spike.abs();
            }
            if s_spike.abs().mul(&v.read(0, ns - 1).abs()) <= max(small_num.clone(), eps.mul(&foo))
            {
                // Eigenvalue is deflatable
                ns -= 1;
            } else {
                // Eigenvalue is not deflatable.
                // Move it up out of the way.
                let ifst = ns - 1;
                schur_move(
                    tw.rb_mut(),
                    Some(v.rb_mut()),
                    ifst,
                    &mut ilst,
                    epsilon.clone(),
                    zero_threshold.clone(),
                );
                ilst += 1;
            }
        } else {
            // 2x2 eigenvalue block
            #[allow(clippy::disallowed_names)]
            let mut foo = tw.read(ns - 1, ns - 1).abs().add(
                &tw.read(ns - 1, ns - 2)
                    .abs()
                    .sqrt()
                    .mul(&tw.read(ns - 2, ns - 1).abs().sqrt()),
            );
            if foo == E::zero() {
                foo = s_spike.abs();
            }
            if max(
                (s_spike.mul(&v.read(0, ns - 1))).abs(),
                (s_spike.mul(&v.read(0, ns - 2))).abs(),
            ) <= max(small_num.clone(), eps.mul(&foo))
            {
                // Eigenvalue pair is deflatable
                ns -= 2;
            } else {
                // Eigenvalue pair is not deflatable.
                // Move it up out of the way.
                let ifst = ns - 2;
                schur_move(
                    tw.rb_mut(),
                    Some(v.rb_mut()),
                    ifst,
                    &mut ilst,
                    epsilon.clone(),
                    zero_threshold.clone(),
                );
                ilst += 2;
            }
        }
    }

    if ns == 0 {
        s_spike = E::zero();
    }

    if ns == jw {
        // Agressive early deflation didn't deflate any eigenvalues
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
            if tw.read(i1 + 1, i1) != E::zero() {
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
            if i2 + 1 < jw && tw.read(i2 + 1, i2) != E::zero() {
                n2 = 2;
            }

            let (ev1, ev2);
            if n1 == 1 {
                ev1 = tw.read(i1, i1).abs();
            } else {
                ev1 = tw.read(i1, i1).abs().add(
                    &(tw.read(i1 + 1, i1).abs().sqrt()).mul(&tw.read(i1, i1 + 1).abs().sqrt()),
                );
            }
            if n2 == 1 {
                ev2 = tw.read(i2, i2).abs();
            } else {
                ev2 = tw.read(i2, i2).abs().add(
                    &(tw.read(i2 + 1, i2).abs().sqrt()).mul(&tw.read(i2, i2 + 1).abs().sqrt()),
                );
            }

            if ev1 > ev2 {
                i1 = i2;
            } else {
                sorted = false;
                let ierr = schur_swap(
                    tw.rb_mut(),
                    Some(v.rb_mut()),
                    i1,
                    n1,
                    n2,
                    epsilon.clone(),
                    zero_threshold.clone(),
                );
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
        if i + 1 < jw && tw.read(i + 1, i) != E::zero() {
            n1 = 2;
        }

        if n1 == 1 {
            s_re.write(kwtop + i, 0, tw.read(i, i));
            s_im.write(kwtop + i, 0, E::zero());
        } else {
            let ((s1_re, s1_im), (s2_re, s2_im)) = lahqr_eig22(
                tw.read(i, i),
                tw.read(i, i + 1),
                tw.read(i + 1, i),
                tw.read(i + 1, i + 1),
            );

            s_re.write(kwtop + i, 0, s1_re);
            s_im.write(kwtop + i, 0, s1_im);
            s_re.write(kwtop + i + 1, 0, s2_re);
            s_im.write(kwtop + i + 1, 0, s2_im);
        }
        i += n1;
    }

    // Reduce A back to Hessenberg form (if neccesary)
    if s_spike != E::zero() {
        // Reflect spike back
        {
            let mut vv = wv.rb_mut().col(0).subrows(0, ns);
            for i in 0..ns {
                vv.write(i, 0, v.read(0, i).conj());
            }
            let head = vv.read(0, 0);
            let tail = vv.rb_mut().subrows(1, ns - 1);
            let tail_sqr_norm = sqr_norm(tail.rb());
            let (tau, beta) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
            vv.write(0, 0, E::one());
            let tau = tau.inv();

            {
                let mut tw_slice = tw.rb_mut().submatrix(0, 0, ns, jw);
                let tmp = vv.rb().adjoint() * tw_slice.rb();
                matmul(
                    tw_slice.rb_mut(),
                    vv.rb(),
                    tmp.as_ref(),
                    Some(E::one()),
                    tau.neg(),
                    parallelism,
                );
            }

            {
                let mut tw_slice2 = tw.rb_mut().submatrix(0, 0, jw, ns);
                let tmp = tw_slice2.rb() * vv.rb();
                matmul(
                    tw_slice2.rb_mut(),
                    tmp.as_ref(),
                    vv.rb().adjoint(),
                    Some(E::one()),
                    tau.neg(),
                    parallelism,
                );
            }

            {
                let mut v_slice = v.rb_mut().submatrix(0, 0, jw, ns);
                let tmp = v_slice.rb() * vv.rb();
                matmul(
                    v_slice.rb_mut(),
                    tmp.as_ref(),
                    vv.rb().adjoint(),
                    Some(E::one()),
                    tau.neg(),
                    parallelism,
                );
            }
            vv.write(0, 0, beta);
        }

        // Hessenberg reduction
        {
            let mut householder = wv.rb_mut().col(0).subrows(0, ns - 1);
            make_hessenberg_in_place(
                tw.rb_mut().submatrix(0, 0, ns, ns),
                householder.rb_mut(),
                parallelism,
                stack.rb_mut(),
            );
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                tw.rb().submatrix(1, 0, ns - 1, ns - 1),
                householder.rb().transpose(),
                Conj::Yes,
                unsafe { tw.rb().submatrix(1, ns, ns - 1, jw - ns).const_cast() },
                parallelism,
                stack.rb_mut(),
            );
            apply_block_householder_sequence_on_the_right_in_place_with_conj(
                tw.rb().submatrix(1, 0, ns - 1, ns - 1),
                householder.rb().transpose(),
                Conj::No,
                v.rb_mut().submatrix(0, 1, jw, ns - 1),
                parallelism,
                stack.rb_mut(),
            );
        }
    }

    // Copy the deflation window back into place
    if kwtop > 0 {
        a.write(kwtop, kwtop - 1, s_spike.mul(&v.read(0, 0).conj()));
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
            let mut a_slice = a.rb_mut().submatrix(kwtop, i, ihi - kwtop, iblock);
            let mut wh_slice = wh
                .rb_mut()
                .submatrix(0, 0, a_slice.nrows(), a_slice.ncols());
            matmul(
                wh_slice.rb_mut(),
                v.rb().adjoint(),
                a_slice.rb(),
                None,
                E::one(),
                parallelism,
            );
            a_slice.clone_from(wh_slice.rb());
            i += iblock;
        }
    }

    // Vertical multiply
    if istart_m < kwtop {
        let mut i = istart_m;
        while i < kwtop {
            let iblock = Ord::min(kwtop - i, wv.nrows());
            let mut a_slice = a.rb_mut().submatrix(i, kwtop, iblock, ihi - kwtop);
            let mut wv_slice = wv
                .rb_mut()
                .submatrix(0, 0, a_slice.nrows(), a_slice.ncols());
            matmul(
                wv_slice.rb_mut(),
                a_slice.rb(),
                v.rb(),
                None,
                E::one(),
                parallelism,
            );
            a_slice.clone_from(wv_slice.rb());
            i += iblock;
        }
    }
    // Update Z (also a vertical multiplication)
    if let Some(mut z) = z.rb_mut() {
        let mut i = 0;
        while i < n {
            let iblock = Ord::min(n - i, wv.nrows());
            let mut z_slice = z.rb_mut().submatrix(i, kwtop, iblock, ihi - kwtop);
            let mut wv_slice = wv
                .rb_mut()
                .submatrix(0, 0, z_slice.nrows(), z_slice.ncols());
            matmul(
                wv_slice.rb_mut(),
                z_slice.rb(),
                v.rb(),
                None,
                E::one(),
                parallelism,
            );
            z_slice.clone_from(wv_slice.rb());
            i += iblock;
        }
    }

    (ns, nd)
}

fn move_bulge<E: RealField>(
    mut h: MatMut<'_, E>,
    mut v: MatMut<'_, E>,
    s1: (E, E),
    s2: (E, E),
    epsilon: E::Real,
) {
    // Perform delayed update of row below the bulge
    // Assumes the first two elements of the row are zero
    let v0 = v.read(0, 0).real();
    let v1 = v.read(1, 0);
    let v2 = v.read(2, 0);
    let refsum = v2.scale_real(&v0).mul(&h.read(3, 2));

    h.write(3, 0, refsum.neg());
    h.write(3, 1, refsum.neg().mul(&v1.conj()));
    h.write(3, 2, h.read(3, 2).sub(&refsum.mul(&v2.conj())));

    // Generate reflector to move bulge down
    v.write(0, 0, h.read(1, 0));
    v.write(1, 0, h.read(2, 0));
    v.write(2, 0, h.read(3, 0));

    let head = v.read(0, 0);
    let tail = v.rb_mut().subrows(1, 2);
    let tail_sqr_norm = sqr_norm(tail.rb());
    let (tau, beta) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
    v.write(0, 0, tau.inv());

    // Check for bulge collapse
    if h.read(3, 0) != E::zero() || h.read(3, 1) != E::zero() || h.read(3, 2) != E::zero() {
        // The bulge hasn't collapsed, typical case
        h.write(1, 0, beta);
        h.write(2, 0, E::zero());
        h.write(3, 0, E::zero());
    } else {
        // The bulge has collapsed, attempt to reintroduce using
        // 2-small-subdiagonals trick
        let mut vt_storage = E::map(E::zero().into_units(), |zero_unit| {
            [zero_unit.clone(), zero_unit.clone(), zero_unit]
        });
        let vt_ptr = E::map(E::as_mut(&mut vt_storage), |array| array.as_mut_ptr());
        let mut vt = unsafe { MatMut::<E>::from_raw_parts(vt_ptr, 3, 1, 1, 3) };

        let h2 = h.rb().submatrix(1, 1, 3, 3);
        lahqr_shiftcolumn(h2, vt.rb_mut(), s1, s2);

        let head = vt.read(0, 0);
        let tail = vt.rb_mut().subrows(1, 2);
        let tail_sqr_norm = sqr_norm(tail.rb());
        let (tau, _) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
        vt.write(0, 0, tau.inv());
        let vt0 = vt.read(0, 0);
        let vt1 = vt.read(1, 0);
        let vt2 = vt.read(2, 0);

        let refsum = (vt0.conj().mul(&h.read(1, 0))).add(&vt1.conj().mul(&h.read(2, 0)));

        if abs1(&h.read(2, 0).sub(&refsum.mul(&vt1))).add(&abs1(&refsum.mul(&vt2)))
            > epsilon.mul(
                &abs1(&h.read(0, 0))
                    .add(&abs1(&h.read(1, 1)))
                    .add(&abs1(&h.read(2, 2))),
            )
        {
            // Starting a new bulge here would create non-negligible fill. Use
            // the old one.
            h.write(1, 0, beta);
            h.write(2, 0, E::zero());
            h.write(3, 0, E::zero());
        } else {
            // Fill-in is negligible, use the new reflector.
            h.write(1, 0, h.read(1, 0).sub(&refsum));
            h.write(2, 0, E::zero());
            h.write(3, 0, E::zero());
            v.write(0, 0, vt.read(0, 0));
            v.write(1, 0, vt.read(1, 0));
            v.write(2, 0, vt.read(2, 0));
        }
    }
}

fn multishift_qr_sweep<E: RealField>(
    want_t: bool,
    a: MatMut<E>,
    mut z: Option<MatMut<E>>,
    s_re: MatMut<E>,
    s_im: MatMut<E>,
    ilo: usize,
    ihi: usize,
    epsilon: E::Real,
    zero_threshold: E::Real,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let n = a.nrows();

    let eps = epsilon.clone();
    let small_num = zero_threshold.div(&eps).mul(&E::Real::from_f64(n as f64));
    assert!(n >= 12);

    let (mut v, _stack) = faer_core::temp_mat_zeroed::<E>(3, s_re.nrows() / 2, stack);
    let mut v = v.as_mut();

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
    let mut i_pos_block;

    //
    // The following code block introduces the bulges into the matrix
    //
    {
        // Near-the-diagonal bulge introduction
        // The calculations are initially limited to the window:
        // A(ilo:ilo+n_block,ilo:ilo+n_block) The rest is updated later via
        // level 3 BLAS

        let n_block = Ord::min(n_block_desired, ihi - ilo);
        let mut istart_m = ilo;
        let mut istop_m = ilo + n_block;
        let mut u2 = u.rb_mut().submatrix(0, 0, n_block, n_block);
        u2.fill_with_zero();
        u2.rb_mut().diagonal().fill_with_constant(E::one());

        for i_pos_last in ilo..ilo + n_block - 2 {
            // The number of bulges that are in the pencil
            let n_active_bulges = Ord::min(n_bulges, ((i_pos_last - ilo) / 2) + 1);

            for i_bulge in 0..n_active_bulges {
                let i_pos = i_pos_last - 2 * i_bulge;
                let mut v = v.rb_mut().col(i_bulge);
                if i_pos == ilo {
                    // Introduce bulge
                    let h = a.rb().submatrix(ilo, ilo, 3, 3);

                    let s1_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge, 0);
                    let s1_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge, 0);
                    let s2_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge - 1, 0);
                    let s2_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge - 1, 0);
                    lahqr_shiftcolumn(h, v.rb_mut(), (s1_re, s1_im), (s2_re, s2_im));

                    debug_assert!(v.nrows() == 3);
                    let head = v.read(0, 0);
                    let tail = v.rb_mut().subrows(1, 2);
                    let tail_sqr_norm = sqr_norm(tail.rb());
                    let (tau, _) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
                    v.write(0, 0, tau.inv());
                } else {
                    // Chase bulge down
                    let mut h = a.rb_mut().submatrix(i_pos - 1, i_pos - 1, 4, 4);
                    let s1_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge, 0);
                    let s1_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge, 0);
                    let s2_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge - 1, 0);
                    let s2_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge - 1, 0);
                    move_bulge(
                        h.rb_mut(),
                        v.rb_mut(),
                        (s1_re, s1_im),
                        (s2_re, s2_im),
                        epsilon.clone(),
                    );
                }

                // Apply the reflector we just calculated from the right
                // We leave the last row for later (it interferes with the
                // optimally packed bulges)

                let v0 = v.read(0, 0).real();
                let v1 = v.read(1, 0);
                let v2 = v.read(2, 0);

                for j in istart_m..i_pos + 3 {
                    let sum = a
                        .read(j, i_pos)
                        .add(&v1.mul(&a.read(j, i_pos + 1)))
                        .add(&v2.mul(&a.read(j, i_pos + 2)));
                    a.write(j, i_pos, a.read(j, i_pos).sub(&sum.scale_real(&v0)));
                    a.write(
                        j,
                        i_pos + 1,
                        a.read(j, i_pos + 1)
                            .sub(&sum.scale_real(&v0).mul(&v1.conj())),
                    );
                    a.write(
                        j,
                        i_pos + 2,
                        a.read(j, i_pos + 2)
                            .sub(&sum.scale_real(&v0).mul(&v2.conj())),
                    );
                }

                // Apply the reflector we just calculated from the left
                // We only update a single column, the rest is updated later
                let sum = a
                    .read(i_pos, i_pos)
                    .add(&v1.conj().mul(&a.read(i_pos + 1, i_pos)))
                    .add(&v2.conj().mul(&a.read(i_pos + 2, i_pos)));
                a.write(i_pos, i_pos, a.read(i_pos, i_pos).sub(&sum.scale_real(&v0)));
                a.write(
                    i_pos + 1,
                    i_pos,
                    a.read(i_pos + 1, i_pos).sub(&sum.scale_real(&v0).mul(&v1)),
                );
                a.write(
                    i_pos + 2,
                    i_pos,
                    a.read(i_pos + 2, i_pos).sub(&sum.scale_real(&v0).mul(&v2)),
                );

                // Test for deflation.
                if i_pos > ilo && a.read(i_pos, i_pos - 1) != E::zero() {
                    let mut tst1 =
                        abs1(&a.read(i_pos - 1, i_pos - 1)).add(&abs1(&a.read(i_pos, i_pos)));
                    if tst1 == E::Real::zero() {
                        if i_pos > ilo + 1 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos - 1, i_pos - 2)));
                        }
                        if i_pos > ilo + 2 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos - 1, i_pos - 3)));
                        }
                        if i_pos > ilo + 3 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos - 1, i_pos - 4)));
                        }
                        if i_pos < ihi - 1 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos + 1, i_pos)));
                        }
                        if i_pos < ihi - 2 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos + 2, i_pos)));
                        }
                        if i_pos < ihi - 3 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos + 3, i_pos)));
                        }
                    }
                    if abs1(&a.read(i_pos, i_pos - 1)) < max(small_num.clone(), eps.mul(&tst1)) {
                        let ab = max(
                            abs1(&a.read(i_pos, i_pos - 1)),
                            abs1(&a.read(i_pos - 1, i_pos)),
                        );
                        let ba = min(
                            abs1(&a.read(i_pos, i_pos - 1)),
                            abs1(&a.read(i_pos - 1, i_pos)),
                        );
                        let aa = max(
                            abs1(&a.read(i_pos, i_pos)),
                            abs1(&a.read(i_pos, i_pos).sub(&a.read(i_pos - 1, i_pos - 1))),
                        );
                        let bb = min(
                            abs1(&a.read(i_pos, i_pos)),
                            abs1(&a.read(i_pos, i_pos).sub(&a.read(i_pos - 1, i_pos - 1))),
                        );
                        let s = aa.add(&ab);
                        if ba.mul(&ab.div(&s))
                            <= max(small_num.clone(), eps.mul(&bb.mul(&aa.div(&s))))
                        {
                            a.write(i_pos, i_pos - 1, E::zero());
                        }
                    }
                }
            }

            // Delayed update from the left
            for i_bulge in 0..n_active_bulges {
                let i_pos = i_pos_last - 2 * i_bulge;
                let v = v.rb_mut().col(i_bulge);

                let v0 = v.read(0, 0).real();
                let v1 = v.read(1, 0);
                let v2 = v.read(2, 0);

                for j in i_pos + 1..istop_m {
                    let sum = a
                        .read(i_pos, j)
                        .add(&v1.conj().mul(&a.read(i_pos + 1, j)))
                        .add(&v2.conj().mul(&a.read(i_pos + 2, j)));
                    a.write(i_pos, j, a.read(i_pos, j).sub(&sum.scale_real(&v0)));
                    a.write(
                        i_pos + 1,
                        j,
                        a.read(i_pos + 1, j).sub(&sum.scale_real(&v0).mul(&v1)),
                    );
                    a.write(
                        i_pos + 2,
                        j,
                        a.read(i_pos + 2, j).sub(&sum.scale_real(&v0).mul(&v2)),
                    );
                }
            }

            // Accumulate the reflectors into U
            for i_bulge in 0..n_active_bulges {
                let i_pos = i_pos_last - 2 * i_bulge;
                let v = v.rb_mut().col(i_bulge);

                let v0 = v.read(0, 0).real();
                let v1 = v.read(1, 0);
                let v2 = v.read(2, 0);

                let i1 = 0;
                let i2 = Ord::min(u2.nrows(), (i_pos_last - ilo) + (i_pos_last - ilo) + 3);

                for j in i1..i2 {
                    let sum = u2
                        .read(j, i_pos - ilo)
                        .add(&v1.mul(&u2.read(j, i_pos - ilo + 1)))
                        .add(&v2.mul(&u2.read(j, i_pos - ilo + 2)));

                    u2.write(
                        j,
                        i_pos - ilo,
                        u2.read(j, i_pos - ilo).sub(&sum.scale_real(&v0)),
                    );
                    u2.write(
                        j,
                        i_pos - ilo + 1,
                        u2.read(j, i_pos - ilo + 1)
                            .sub(&sum.scale_real(&v0).mul(&v1.conj())),
                    );
                    u2.write(
                        j,
                        i_pos - ilo + 2,
                        u2.read(j, i_pos - ilo + 2)
                            .sub(&sum.scale_real(&v0).mul(&v2.conj())),
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
                let mut a_slice = a.rb_mut().submatrix(ilo, i, n_block, iblock);
                let mut wh_slice = wh
                    .rb_mut()
                    .submatrix(0, 0, a_slice.nrows(), a_slice.ncols());
                matmul(
                    wh_slice.rb_mut(),
                    u2.rb().adjoint(),
                    a_slice.rb(),
                    None,
                    E::one(),
                    parallelism,
                );
                a_slice.clone_from(wh_slice.rb());
                i += iblock;
            }
        }
        // Vertical multiply
        if istart_m < ilo {
            let mut i = istart_m;
            while i < ilo {
                let iblock = Ord::min(ilo - i, wv.nrows());
                let mut a_slice = a.rb_mut().submatrix(i, ilo, iblock, n_block);
                let mut wv_slice = wv
                    .rb_mut()
                    .submatrix(0, 0, a_slice.nrows(), a_slice.ncols());
                matmul(
                    wv_slice.rb_mut(),
                    a_slice.rb(),
                    u2.rb(),
                    None,
                    E::one(),
                    parallelism,
                );
                a_slice.clone_from(wv_slice.rb());
                i += iblock;
            }
        }
        // Update Z (also a vertical multiplication)
        if let Some(mut z) = z.rb_mut() {
            let mut i = 0;
            while i < n {
                let iblock = Ord::min(n - i, wv.nrows());
                let mut z_slice = z.rb_mut().submatrix(i, ilo, iblock, n_block);
                let mut wv_slice = wv
                    .rb_mut()
                    .submatrix(0, 0, z_slice.nrows(), z_slice.ncols());
                matmul(
                    wv_slice.rb_mut(),
                    z_slice.rb(),
                    u2.rb(),
                    None,
                    E::one(),
                    parallelism,
                );
                z_slice.clone_from(wv_slice.rb());
                i += iblock;
            }
        }

        i_pos_block = ilo + n_block - n_shifts;
    }

    //
    // The following code block moves the bulges down untill they are low enough
    // to be removed
    //
    while i_pos_block + n_block_desired < ihi {
        // Number of positions each bulge will be moved down
        let n_pos = Ord::min(n_block_desired - n_shifts, ihi - n_shifts - 1 - i_pos_block);
        // Actual blocksize
        let n_block = n_shifts + n_pos;

        let mut u2 = u.rb_mut().submatrix(0, 0, n_block, n_block);
        u2.fill_with_zero();
        u2.rb_mut().diagonal().fill_with_constant(E::one());

        // Near-the-diagonal bulge chase
        // The calculations are initially limited to the window:
        // A(i_pos_block-1:i_pos_block+n_block,i_pos_block:i_pos_block+n_block)
        // The rest is updated later via level 3 BLAS
        let mut istart_m = i_pos_block;
        let mut istop_m = i_pos_block + n_block;

        for i_pos_last in i_pos_block + n_shifts - 2..i_pos_block + n_shifts - 2 + n_pos {
            for i_bulge in 0..n_bulges {
                let i_pos = i_pos_last - 2 * i_bulge;
                let mut v = v.rb_mut().col(i_bulge);

                // Chase bulge down
                let mut h = a.rb_mut().submatrix(i_pos - 1, i_pos - 1, 4, 4);
                let s1_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge, 0);
                let s1_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge, 0);
                let s2_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge - 1, 0);
                let s2_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge - 1, 0);
                move_bulge(
                    h.rb_mut(),
                    v.rb_mut(),
                    (s1_re, s1_im),
                    (s2_re, s2_im),
                    epsilon.clone(),
                );

                // Apply the reflector we just calculated from the right
                // We leave the last row for later (it interferes with the
                // optimally packed bulges)

                let v0 = v.read(0, 0).real();
                let v1 = v.read(1, 0);
                let v2 = v.read(2, 0);

                for j in istart_m..i_pos + 3 {
                    let sum = a
                        .read(j, i_pos)
                        .add(&v1.mul(&a.read(j, i_pos + 1)))
                        .add(&v2.mul(&a.read(j, i_pos + 2)));
                    a.write(j, i_pos, a.read(j, i_pos).sub(&sum.scale_real(&v0)));
                    a.write(
                        j,
                        i_pos + 1,
                        a.read(j, i_pos + 1)
                            .sub(&sum.scale_real(&v0).mul(&v1.conj())),
                    );
                    a.write(
                        j,
                        i_pos + 2,
                        a.read(j, i_pos + 2)
                            .sub(&sum.scale_real(&v0).mul(&v2.conj())),
                    );
                }

                // Apply the reflector we just calculated from the left
                // We only update a single column, the rest is updated later
                let sum = a
                    .read(i_pos, i_pos)
                    .add(&v1.conj().mul(&a.read(i_pos + 1, i_pos)))
                    .add(&v2.conj().mul(&a.read(i_pos + 2, i_pos)));
                a.write(i_pos, i_pos, a.read(i_pos, i_pos).sub(&sum.scale_real(&v0)));
                a.write(
                    i_pos + 1,
                    i_pos,
                    a.read(i_pos + 1, i_pos).sub(&sum.scale_real(&v0).mul(&v1)),
                );
                a.write(
                    i_pos + 2,
                    i_pos,
                    a.read(i_pos + 2, i_pos).sub(&sum.scale_real(&v0).mul(&v2)),
                );

                // Test for deflation.
                if i_pos > ilo && a.read(i_pos, i_pos - 1) != E::zero() {
                    let mut tst1 =
                        abs1(&a.read(i_pos - 1, i_pos - 1)).add(&abs1(&a.read(i_pos, i_pos)));
                    if tst1 == E::Real::zero() {
                        if i_pos > ilo + 1 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos - 1, i_pos - 2)));
                        }
                        if i_pos > ilo + 2 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos - 1, i_pos - 3)));
                        }
                        if i_pos > ilo + 3 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos - 1, i_pos - 4)));
                        }
                        if i_pos < ihi - 1 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos + 1, i_pos)));
                        }
                        if i_pos < ihi - 2 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos + 2, i_pos)));
                        }
                        if i_pos < ihi - 3 {
                            tst1 = tst1.add(&abs1(&a.read(i_pos + 3, i_pos)));
                        }
                    }
                    if abs1(&a.read(i_pos, i_pos - 1)) < max(small_num.clone(), eps.mul(&tst1)) {
                        let ab = max(
                            abs1(&a.read(i_pos, i_pos - 1)),
                            abs1(&a.read(i_pos - 1, i_pos)),
                        );
                        let ba = min(
                            abs1(&a.read(i_pos, i_pos - 1)),
                            abs1(&a.read(i_pos - 1, i_pos)),
                        );
                        let aa = max(
                            abs1(&a.read(i_pos, i_pos)),
                            abs1(&a.read(i_pos, i_pos).sub(&a.read(i_pos - 1, i_pos - 1))),
                        );
                        let bb = min(
                            abs1(&a.read(i_pos, i_pos)),
                            abs1(&a.read(i_pos, i_pos).sub(&a.read(i_pos - 1, i_pos - 1))),
                        );
                        let s = aa.add(&ab);
                        if ba.mul(&ab.div(&s))
                            <= max(small_num.clone(), eps.mul(&bb.mul(&aa.div(&s))))
                        {
                            a.write(i_pos, i_pos - 1, E::zero());
                        }
                    }
                }
            }

            // Delayed update from the left
            for i_bulge in 0..n_bulges {
                let i_pos = i_pos_last - 2 * i_bulge;
                let v = v.rb_mut().col(i_bulge);

                let v0 = v.read(0, 0).real();
                let v1 = v.read(1, 0);
                let v2 = v.read(2, 0);

                for j in i_pos + 1..istop_m {
                    let sum = a
                        .read(i_pos, j)
                        .add(&v1.conj().mul(&a.read(i_pos + 1, j)))
                        .add(&v2.conj().mul(&a.read(i_pos + 2, j)));
                    a.write(i_pos, j, a.read(i_pos, j).sub(&sum.scale_real(&v0)));
                    a.write(
                        i_pos + 1,
                        j,
                        a.read(i_pos + 1, j).sub(&sum.scale_real(&v0).mul(&v1)),
                    );
                    a.write(
                        i_pos + 2,
                        j,
                        a.read(i_pos + 2, j).sub(&sum.scale_real(&v0).mul(&v2)),
                    );
                }
            }

            // Accumulate the reflectors into U
            for i_bulge in 0..n_bulges {
                let i_pos = i_pos_last - 2 * i_bulge;
                let v = v.rb_mut().col(i_bulge);

                let v0 = v.read(0, 0).real();
                let v1 = v.read(1, 0);
                let v2 = v.read(2, 0);

                let i1 = (i_pos - i_pos_block) - (i_pos_last + 2 - i_pos_block - n_shifts);
                let i2 = Ord::min(
                    u2.nrows(),
                    (i_pos_last - i_pos_block) + (i_pos_last + 2 - i_pos_block - n_shifts) + 3,
                );

                for j in i1..i2 {
                    let sum = u2
                        .read(j, i_pos - i_pos_block)
                        .add(&v1.mul(&u2.read(j, i_pos - i_pos_block + 1)))
                        .add(&v2.mul(&u2.read(j, i_pos - i_pos_block + 2)));

                    u2.write(
                        j,
                        i_pos - i_pos_block,
                        u2.read(j, i_pos - i_pos_block).sub(&sum.scale_real(&v0)),
                    );
                    u2.write(
                        j,
                        i_pos - i_pos_block + 1,
                        u2.read(j, i_pos - i_pos_block + 1)
                            .sub(&sum.scale_real(&v0).mul(&v1.conj())),
                    );
                    u2.write(
                        j,
                        i_pos - i_pos_block + 2,
                        u2.read(j, i_pos - i_pos_block + 2)
                            .sub(&sum.scale_real(&v0).mul(&v2.conj())),
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
        if i_pos_block + n_block < istop_m {
            let mut i = i_pos_block + n_block;
            while i < istop_m {
                let iblock = Ord::min(istop_m - i, wh.ncols());
                let mut a_slice = a.rb_mut().submatrix(i_pos_block, i, n_block, iblock);
                let mut wh_slice = wh
                    .rb_mut()
                    .submatrix(0, 0, a_slice.nrows(), a_slice.ncols());
                matmul(
                    wh_slice.rb_mut(),
                    u2.rb().adjoint(),
                    a_slice.rb(),
                    None,
                    E::one(),
                    parallelism,
                );
                a_slice.clone_from(wh_slice.rb());
                i += iblock;
            }
        }

        // Vertical multiply
        if istart_m < i_pos_block {
            let mut i = istart_m;
            while i < i_pos_block {
                let iblock = Ord::min(i_pos_block - i, wv.nrows());
                let mut a_slice = a.rb_mut().submatrix(i, i_pos_block, iblock, n_block);
                let mut wv_slice = wv
                    .rb_mut()
                    .submatrix(0, 0, a_slice.nrows(), a_slice.ncols());
                matmul(
                    wv_slice.rb_mut(),
                    a_slice.rb(),
                    u2.rb(),
                    None,
                    E::one(),
                    parallelism,
                );
                a_slice.clone_from(wv_slice.rb());
                i += iblock;
            }
        }
        // Update Z (also a vertical multiplication)
        if let Some(mut z) = z.rb_mut() {
            let mut i = 0;
            while i < n {
                let iblock = Ord::min(n - i, wv.nrows());
                let mut z_slice = z.rb_mut().submatrix(i, i_pos_block, iblock, n_block);
                let mut wv_slice = wv
                    .rb_mut()
                    .submatrix(0, 0, z_slice.nrows(), z_slice.ncols());
                matmul(
                    wv_slice.rb_mut(),
                    z_slice.rb(),
                    u2.rb(),
                    None,
                    E::one(),
                    parallelism,
                );
                z_slice.clone_from(wv_slice.rb());
                i += iblock;
            }
        }

        i_pos_block += n_pos;
    }

    //
    // The following code removes the bulges from the matrix
    //
    {
        let n_block = ihi - i_pos_block;

        let mut u2 = u.rb_mut().submatrix(0, 0, n_block, n_block);
        u2.fill_with_zero();
        u2.rb_mut().diagonal().fill_with_constant(E::one());

        // Near-the-diagonal bulge chase
        // The calculations are initially limited to the window:
        // A(i_pos_block-1:ihi,i_pos_block:ihi) The rest is updated later via
        // level 3 BLAS
        let mut istart_m = i_pos_block;
        let mut istop_m = ihi;

        for i_pos_last in i_pos_block + n_shifts - 2..ihi + n_shifts - 1 {
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
                    let mut v = v.rb_mut().subrows(0, 2).col(i_bulge);
                    let mut h = a.rb_mut().subrows(i_pos, 2).col(i_pos - 1);
                    let head = h.read(0, 0);
                    let tail = h.rb_mut().subrows(1, 1);
                    let tail_sqr_norm = sqr_norm(tail.rb());
                    let (tau, beta) = make_householder_in_place(Some(tail), head, tail_sqr_norm);
                    v.write(0, 0, tau.inv());
                    v.write(1, 0, h.read(1, 0));
                    h.write(0, 0, beta);
                    h.write(1, 0, E::zero());

                    let t0 = v.read(0, 0).conj();
                    let v1 = v.read(1, 0);
                    let t1 = t0.mul(&v1);
                    // Apply the reflector we just calculated from the right
                    for j in istart_m..i_pos + 2 {
                        let sum = a.read(j, i_pos).add(&v1.mul(&a.read(j, i_pos + 1)));
                        a.write(j, i_pos, a.read(j, i_pos).sub(&sum.mul(&t0.conj())));
                        a.write(j, i_pos + 1, a.read(j, i_pos + 1).sub(&sum.mul(&t1.conj())));
                    }
                    // Apply the reflector we just calculated from the left
                    for j in i_pos..istop_m {
                        let sum = a.read(i_pos, j).add(&v1.conj().mul(&a.read(i_pos + 1, j)));
                        a.write(i_pos, j, a.read(i_pos, j).sub(&sum.mul(&t0)));
                        a.write(i_pos + 1, j, a.read(i_pos + 1, j).sub(&sum.mul(&t1)));
                    }
                    // Accumulate the reflector into U
                    // The loop bounds should be changed to reflect the fact
                    // that U2 starts off as diagonal
                    for j in 0..u2.nrows() {
                        let sum = u2
                            .read(j, i_pos - i_pos_block)
                            .add(&v1.mul(&u2.read(j, i_pos - i_pos_block + 1)));
                        u2.write(
                            j,
                            i_pos - i_pos_block,
                            u2.read(j, i_pos - i_pos_block).sub(&sum.mul(&t0.conj())),
                        );
                        u2.write(
                            j,
                            i_pos - i_pos_block + 1,
                            u2.read(j, i_pos - i_pos_block + 1)
                                .sub(&sum.mul(&t1.conj())),
                        );
                    }
                } else {
                    let mut v = v.rb_mut().col(i_bulge);
                    let mut h = a.rb_mut().submatrix(i_pos - 1, i_pos - 1, 4, 4);
                    let s1_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge, 0);
                    let s1_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge, 0);
                    let s2_re = s_re.read(s_re.nrows() - 1 - 2 * i_bulge - 1, 0);
                    let s2_im = s_im.read(s_im.nrows() - 1 - 2 * i_bulge - 1, 0);
                    move_bulge(
                        h.rb_mut(),
                        v.rb_mut(),
                        (s1_re, s1_im),
                        (s2_re, s2_im),
                        epsilon.clone(),
                    );

                    {
                        let t0 = v.read(0, 0).conj();
                        let v1 = v.read(1, 0);
                        let t1 = t0.mul(&v1);
                        let v2 = v.read(2, 0);
                        let t2 = t0.mul(&v2);
                        // Apply the reflector we just calculated from the right
                        // (but leave the last row for later)
                        for j in istart_m..i_pos + 3 {
                            let sum = a
                                .read(j, i_pos)
                                .add(&v1.mul(&a.read(j, i_pos + 1)))
                                .add(&v2.mul(&a.read(j, i_pos + 2)));
                            a.write(j, i_pos, a.read(j, i_pos).sub(&sum.mul(&t0.conj())));
                            a.write(j, i_pos + 1, a.read(j, i_pos + 1).sub(&sum.mul(&t1.conj())));
                            a.write(j, i_pos + 2, a.read(j, i_pos + 2).sub(&sum.mul(&t2.conj())));
                        }
                    }

                    let v0 = v.read(0, 0).real();
                    let v1 = v.read(1, 0);
                    let v2 = v.read(2, 0);
                    // Apply the reflector we just calculated from the left
                    // We only update a single column, the rest is updated later
                    let sum = a
                        .read(i_pos, i_pos)
                        .add(&v1.conj().mul(&a.read(i_pos + 1, i_pos)))
                        .add(&v2.conj().mul(&a.read(i_pos + 2, i_pos)));
                    a.write(i_pos, i_pos, a.read(i_pos, i_pos).sub(&sum.scale_real(&v0)));
                    a.write(
                        i_pos + 1,
                        i_pos,
                        a.read(i_pos + 1, i_pos).sub(&sum.scale_real(&v0).mul(&v1)),
                    );
                    a.write(
                        i_pos + 2,
                        i_pos,
                        a.read(i_pos + 2, i_pos).sub(&sum.scale_real(&v0).mul(&v2)),
                    );

                    // Test for deflation.
                    if i_pos > ilo && a.read(i_pos, i_pos - 1) != E::zero() {
                        let mut tst1 =
                            abs1(&a.read(i_pos - 1, i_pos - 1)).add(&abs1(&a.read(i_pos, i_pos)));
                        if tst1 == E::Real::zero() {
                            if i_pos > ilo + 1 {
                                tst1 = tst1.add(&abs1(&a.read(i_pos - 1, i_pos - 2)));
                            }
                            if i_pos > ilo + 2 {
                                tst1 = tst1.add(&abs1(&a.read(i_pos - 1, i_pos - 3)));
                            }
                            if i_pos > ilo + 3 {
                                tst1 = tst1.add(&abs1(&a.read(i_pos - 1, i_pos - 4)));
                            }
                            if i_pos < ihi - 1 {
                                tst1 = tst1.add(&abs1(&a.read(i_pos + 1, i_pos)));
                            }
                            if i_pos < ihi - 2 {
                                tst1 = tst1.add(&abs1(&a.read(i_pos + 2, i_pos)));
                            }
                            if i_pos < ihi - 3 {
                                tst1 = tst1.add(&abs1(&a.read(i_pos + 3, i_pos)));
                            }
                        }
                        if abs1(&a.read(i_pos, i_pos - 1)) < max(small_num.clone(), eps.mul(&tst1))
                        {
                            let ab = max(
                                abs1(&a.read(i_pos, i_pos - 1)),
                                abs1(&a.read(i_pos - 1, i_pos)),
                            );
                            let ba = min(
                                abs1(&a.read(i_pos, i_pos - 1)),
                                abs1(&a.read(i_pos - 1, i_pos)),
                            );
                            let aa = max(
                                abs1(&a.read(i_pos, i_pos)),
                                abs1(&a.read(i_pos, i_pos).sub(&a.read(i_pos - 1, i_pos - 1))),
                            );
                            let bb = min(
                                abs1(&a.read(i_pos, i_pos)),
                                abs1(&a.read(i_pos, i_pos).sub(&a.read(i_pos - 1, i_pos - 1))),
                            );
                            let s = aa.add(&ab);
                            if ba.mul(&ab.div(&s))
                                <= max(small_num.clone(), eps.mul(&bb.mul(&aa.div(&s))))
                            {
                                a.write(i_pos, i_pos - 1, E::zero());
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
                let v = v.rb_mut().col(i_bulge);

                let v0 = v.read(0, 0).real();
                let v1 = v.read(1, 0);
                let v2 = v.read(2, 0);

                for j in i_pos + 1..istop_m {
                    let sum = a
                        .read(i_pos, j)
                        .add(&v1.conj().mul(&a.read(i_pos + 1, j)))
                        .add(&v2.conj().mul(&a.read(i_pos + 2, j)));
                    a.write(i_pos, j, a.read(i_pos, j).sub(&sum.scale_real(&v0)));
                    a.write(
                        i_pos + 1,
                        j,
                        a.read(i_pos + 1, j).sub(&sum.scale_real(&v0).mul(&v1)),
                    );
                    a.write(
                        i_pos + 2,
                        j,
                        a.read(i_pos + 2, j).sub(&sum.scale_real(&v0).mul(&v2)),
                    );
                }
            }

            // Accumulate the reflectors into U
            for i_bulge in i_bulge_start..n_bulges {
                let i_pos = i_pos_last - 2 * i_bulge;
                let v = v.rb_mut().col(i_bulge);

                let v0 = v.read(0, 0).real();
                let v1 = v.read(1, 0);
                let v2 = v.read(2, 0);

                let i1 = (i_pos - i_pos_block) - (i_pos_last + 2 - i_pos_block - n_shifts);
                let i2 = Ord::min(
                    u2.nrows(),
                    (i_pos_last - i_pos_block) + (i_pos_last + 2 - i_pos_block - n_shifts) + 3,
                );

                for j in i1..i2 {
                    let sum = u2
                        .read(j, i_pos - i_pos_block)
                        .add(&v1.mul(&u2.read(j, i_pos - i_pos_block + 1)))
                        .add(&v2.mul(&u2.read(j, i_pos - i_pos_block + 2)));

                    u2.write(
                        j,
                        i_pos - i_pos_block,
                        u2.read(j, i_pos - i_pos_block).sub(&sum.scale_real(&v0)),
                    );
                    u2.write(
                        j,
                        i_pos - i_pos_block + 1,
                        u2.read(j, i_pos - i_pos_block + 1)
                            .sub(&sum.scale_real(&v0).mul(&v1.conj())),
                    );
                    u2.write(
                        j,
                        i_pos - i_pos_block + 2,
                        u2.read(j, i_pos - i_pos_block + 2)
                            .sub(&sum.scale_real(&v0).mul(&v2.conj())),
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

        debug_assert!(i_pos_block + n_block == ihi);

        // Horizontal multiply
        if ihi < istop_m {
            let mut i = ihi;
            while i < istop_m {
                let iblock = Ord::min(istop_m - i, wh.ncols());
                let mut a_slice = a.rb_mut().submatrix(i_pos_block, i, n_block, iblock);
                let mut wh_slice = wh
                    .rb_mut()
                    .submatrix(0, 0, a_slice.nrows(), a_slice.ncols());
                matmul(
                    wh_slice.rb_mut(),
                    u2.rb().adjoint(),
                    a_slice.rb(),
                    None,
                    E::one(),
                    parallelism,
                );
                a_slice.clone_from(wh_slice.rb());
                i += iblock;
            }
        }

        // Vertical multiply
        if istart_m < i_pos_block {
            let mut i = istart_m;
            while i < i_pos_block {
                let iblock = Ord::min(i_pos_block - i, wv.nrows());
                let mut a_slice = a.rb_mut().submatrix(i, i_pos_block, iblock, n_block);
                let mut wv_slice = wv
                    .rb_mut()
                    .submatrix(0, 0, a_slice.nrows(), a_slice.ncols());
                matmul(
                    wv_slice.rb_mut(),
                    a_slice.rb(),
                    u2.rb(),
                    None,
                    E::one(),
                    parallelism,
                );
                a_slice.clone_from(wv_slice.rb());
                i += iblock;
            }
        }
        // Update Z (also a vertical multiplication)
        if let Some(mut z) = z.rb_mut() {
            let mut i = 0;
            while i < n {
                let iblock = Ord::min(n - i, wv.nrows());
                let mut z_slice = z.rb_mut().submatrix(i, i_pos_block, iblock, n_block);
                let mut wv_slice = wv
                    .rb_mut()
                    .submatrix(0, 0, z_slice.nrows(), z_slice.ncols());
                matmul(
                    wv_slice.rb_mut(),
                    z_slice.rb(),
                    u2.rb(),
                    None,
                    E::one(),
                    parallelism,
                );
                z_slice.clone_from(wv_slice.rb());
                i += iblock;
            }
        }
    }
}

pub fn default_recommended_shift_count(dim: usize, _active_block_dim: usize) -> usize {
    let n = dim;
    if n < 30 {
        2
    } else if n < 60 {
        4
    } else if n < 150 {
        12
    } else if n < 590 {
        32
    } else if n < 1500 {
        48
    } else if n < 3000 {
        64
    } else if n < 6000 {
        128
    } else {
        256
    }
}

pub fn multishift_qr<E: RealField>(
    want_t: bool,
    a: MatMut<'_, E>,
    z: Option<MatMut<'_, E>>,
    w_re: MatMut<'_, E>,
    w_im: MatMut<'_, E>,
    ilo: usize,
    ihi: usize,
    epsilon: E::Real,
    zero_threshold: E::Real,
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: EvdParams,
) -> (isize, usize, usize) {
    assert!(a.nrows() == a.ncols());
    assert!(ilo <= ihi);

    let n = a.nrows();
    let nh = ihi - ilo;

    assert!(w_re.nrows() == n);
    assert!(w_re.ncols() == 1);
    assert!(w_im.nrows() == n);
    assert!(w_im.ncols() == 1);

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
    let dat1 = E::Real::from_f64(0.75);
    let dat2 = E::Real::from_f64(-0.4375);

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

    // Recommended deflation window size
    let nwr = (params
        .recommended_deflation_window
        .unwrap_or(default_recommended_deflation_window))(n, nh);

    // Tiny matrices must use lahqr
    if n < nmin {
        let err = lahqr(want_t, a, z, w_re, w_im, ilo, ihi, epsilon, zero_threshold);
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
                w_re.write(ilo, 0, a.read(ilo, ilo));
                w_im.write(ilo, 0, E::zero());
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
            if a.read(i, i - 1) == E::zero() {
                istart = i;
                break;
            }
        }

        //
        // Agressive early deflation
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
            if kwtop > istart + 2
                && abs1(&a.read(kwtop, kwtop - 1)) > abs1(&a.read(kwtop - 1, kwtop - 2))
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
            epsilon.clone(),
            zero_threshold.clone(),
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
        let mut ns = Ord::min(nh - 1, Ord::min(ls, nsr));
        let mut i_shifts = istop - ls;

        if k_defl % non_convergence_limit_shift == 0 {
            ns = nsr;
            for i in (i_shifts..istop - 1).step_by(2) {
                let ss = abs1(&a.read(i, i - 1)).add(&abs1(&a.read(i - 1, i - 2)));
                let aa = E::from_real(dat1.mul(&ss)).add(&a.read(i, i));
                let bb = E::from_real(ss.clone());
                let cc = E::from_real(dat2.mul(&ss));
                let dd = aa.clone();
                let (s1, s2) = lahqr_eig22(aa, bb, cc, dd);
                w_re.write(i, 0, s1.0);
                w_im.write(i, 0, s1.1);
                w_re.write(i + 1, 0, s2.0);
                w_im.write(i + 1, 0, s2.1);
            }
        } else {
            if ls <= nsr / 2 {
                // Got nsr/2 or fewer shifts? Then use multi/double shift qr to
                // get more
                let mut temp = a.rb_mut().submatrix(n - nsr, 0, nsr, nsr);
                let mut shifts_re = w_re.rb_mut().subrows(istop - nsr, nsr);
                let mut shifts_im = w_im.rb_mut().subrows(istop - nsr, nsr);
                let ierr = lahqr(
                    false,
                    temp.rb_mut(),
                    None,
                    shifts_re.rb_mut(),
                    shifts_im.rb_mut(),
                    0,
                    nsr,
                    epsilon.clone(),
                    zero_threshold.clone(),
                ) as usize;

                ns = nsr - ierr;

                if ns < 2 {
                    // In case of a rare QR failure, use eigenvalues
                    // of the trailing 2x2 submatrix
                    let aa = a.read(istop - 2, istop - 2);
                    let bb = a.read(istop - 2, istop - 1);
                    let cc = a.read(istop - 1, istop - 2);
                    let dd = a.read(istop - 1, istop - 1);
                    let (s1, s2) = lahqr_eig22(aa, bb, cc, dd);
                    w_re.write(istop - 2, 0, s1.0);
                    w_im.write(istop - 2, 0, s1.1);
                    w_re.write(istop - 1, 0, s2.0);
                    w_im.write(istop - 1, 0, s2.1);
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
                    if w_re.read(i, 0).abs().add(&w_im.read(i, 0).abs())
                        < w_re.read(i + 1, 0).abs().add(&w_im.read(i + 1, 0).abs())
                    {
                        sorted = false;
                        let wi = (w_re.read(i, 0), w_im.read(i, 0));
                        let wip1 = (w_re.read(i + 1, 0), w_im.read(i + 1, 0));
                        w_re.write(i, 0, wip1.0);
                        w_im.write(i, 0, wip1.1);
                        w_re.write(i + 1, 0, wi.0);
                        w_im.write(i + 1, 0, wi.1);
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
                if w_im.read(i, 0) != w_im.read(i - 1, 0).neg() {
                    let tmp = (w_re.read(i, 0), w_im.read(i, 0));
                    w_re.write(i, 0, w_re.read(i - 1, 0));
                    w_im.write(i, 0, w_im.read(i - 1, 0));
                    w_re.write(i - 1, 0, w_re.read(i - 2, 0));
                    w_im.write(i - 1, 0, w_im.read(i - 2, 0));
                    w_re.write(i - 2, 0, tmp.0);
                    w_im.write(i - 2, 0, tmp.1);
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
        if ns == 2 && w_im.read(i_shifts, 0) == E::zero() {
            if (w_re.read(i_shifts, 0).sub(&a.read(istop - 1, istop - 1))).abs()
                < (w_re
                    .read(i_shifts + 1, 0)
                    .sub(&a.read(istop - 1, istop - 1)))
                .abs()
            {
                w_re.write(i_shifts + 1, 0, w_re.read(i_shifts, 0));
                w_im.write(i_shifts + 1, 0, w_im.read(i_shifts, 0));
            } else {
                w_re.write(i_shifts, 0, w_re.read(i_shifts + 1, 0));
                w_im.write(i_shifts, 0, w_im.read(i_shifts + 1, 0));
            }
        }

        let mut shifts_re = w_re.rb_mut().subrows(i_shifts, ns);
        let mut shifts_im = w_im.rb_mut().subrows(i_shifts, ns);

        multishift_qr_sweep(
            want_t,
            a.rb_mut(),
            z.rb_mut(),
            shifts_re.rb_mut(),
            shifts_im.rb_mut(),
            istart,
            istop,
            epsilon.clone(),
            zero_threshold.clone(),
            parallelism,
            stack.rb_mut(),
        );
        count_sweep += 1;
    }

    (info, count_aed, count_sweep)
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

    let mut v_storage = E::map(E::zero().into_units(), |zero_unit| {
        [zero_unit.clone(), zero_unit.clone(), zero_unit]
    });
    let v_ptr = E::map(E::as_mut(&mut v_storage), |array| array.as_mut_ptr());
    let mut v = unsafe { MatMut::<E>::from_raw_parts(v_ptr, 3, 1, 1, 3) };
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
                if i + 1 < ihi {
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

                        rot(x.transpose(), y.transpose(), cs.clone(), sn.clone());
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
    use faer_core::{mat, ComplexField, Mat};

    macro_rules! make_stack {
        ($req: expr $(,)?) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req.unwrap()))
        };
    }

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
                let mut q = Mat::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });

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

    #[test]
    fn test_multi_n() {
        for n in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32, 63, 64, 65, 128, 256,
        ] {
            for _ in 0..10 {
                let mut h = Mat::<f64>::zeros(n, n);
                for j in 0..n {
                    for i in 0..n {
                        if i <= j + 1 {
                            h.write(i, j, rand::random());
                        }
                    }
                }

                let mut q =
                    Mat::from_fn(n, n, |i, j| if i == j { f64::one() } else { f64::zero() });

                let mut w_re = Mat::zeros(n, 1);
                let mut w_im = Mat::zeros(n, 1);

                let mut t = h.clone();
                let params = EvdParams {
                    recommended_shift_count: None,
                    recommended_deflation_window: None,
                    blocking_threshold: Some(15),
                    nibble_threshold: Some(14),
                };
                dbgf::dbgf!("6.6?", &h);
                multishift_qr(
                    true,
                    t.as_mut(),
                    Some(q.as_mut()),
                    w_re.as_mut(),
                    w_im.as_mut(),
                    0,
                    n,
                    f64::EPSILON,
                    f64::MIN_POSITIVE,
                    Parallelism::None,
                    make_stack!(multishift_qr_req::<f64>(
                        n,
                        n,
                        true,
                        true,
                        Parallelism::None,
                        params,
                    )),
                    params,
                );
                for j in 0..n {
                    for i in j + 2..n {
                        t.write(i, j, f64::zero());
                    }
                }

                dbgf::dbgf!("13.10?", &t, q.as_ref() * q.transpose());

                let h_reconstructed = &q * &t * q.adjoint();

                for i in 0..n {
                    for j in 0..n {
                        assert_approx_eq!(h_reconstructed.read(i, j), h.read(i, j));
                    }
                }
            }
        }
    }

    #[test]
    fn test_multi_16() {
        let n = 16;
        let h = mat![
            [
                0.238434, 0.698001, 0.885770, 0.690614, 0.379370, 0.920268, 0.413072, 0.651627,
                0.928669, 0.535102, 0.693188, 0.817634, 0.936823, 0.110140, 0.726488, 0.913997
            ],
            [
                0.224096, 0.657603, 0.074754, 0.348251, 0.178755, 0.050091, 0.997528, 0.482081,
                0.428036, 0.567365, 0.991769, 0.618078, 0.650064, 0.106555, 0.676259, 0.322415
            ],
            [
                0.000000, 0.442482, 0.215858, 0.775816, 0.641317, 0.382869, 0.118101, 0.746954,
                0.759515, 0.133278, 0.418937, 0.841150, 0.853753, 0.248821, 0.671283, 0.785202
            ],
            [
                0.000000, 0.000000, 0.074608, 0.546737, 0.432506, 0.555511, 0.572336, 0.825026,
                0.285712, 0.062708, 0.201395, 0.887187, 0.855668, 0.380182, 0.817253, 0.729050
            ],
            [
                0.000000, 0.000000, 0.000000, 0.433472, 0.191306, 0.323110, 0.912452, 0.189640,
                0.752605, 0.268832, 0.264967, 0.862273, 0.473225, 0.229077, 0.477724, 0.843642
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.050402, 0.878767, 0.520969, 0.154328,
                0.338822, 0.466055, 0.780303, 0.338676, 0.902086, 0.896993, 0.879068, 0.652666
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.240152, 0.452439, 0.783997,
                0.728769, 0.972107, 0.368986, 0.886526, 0.469381, 0.061919, 0.825741, 0.929793
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.353673, 0.340159,
                0.625835, 0.969770, 0.771019, 0.639253, 0.460149, 0.890654, 0.287737, 0.036852
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.077547,
                0.367661, 0.015909, 0.744168, 0.383175, 0.779238, 0.431544, 0.558823, 0.423483
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.699894, 0.794555, 0.846859, 0.863141, 0.198618, 0.866309, 0.889694, 0.699332
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.063769, 0.376254, 0.413182, 0.047518, 0.213909, 0.617082, 0.592612
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.115594, 0.322987, 0.279308, 0.339607, 0.280421, 0.187590
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.317097, 0.292163, 0.798380, 0.683442, 0.926154
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.297301, 0.477071, 0.656784, 0.572734
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.579016, 0.368096, 0.618694
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.220369, 0.470277
            ],
        ];

        let mut q = Mat::from_fn(n, n, |i, j| if i == j { f64::one() } else { f64::zero() });

        let mut w_re = Mat::zeros(n, 1);
        let mut w_im = Mat::zeros(n, 1);

        let mut t = h.clone();
        let params = EvdParams {
            blocking_threshold: Some(15),
            ..Default::default()
        };
        multishift_qr(
            true,
            t.as_mut(),
            Some(q.as_mut()),
            w_re.as_mut(),
            w_im.as_mut(),
            0,
            n,
            f64::EPSILON,
            f64::MIN_POSITIVE,
            Parallelism::None,
            make_stack!(multishift_qr_req::<f64>(
                n,
                n,
                true,
                true,
                Parallelism::None,
                params,
            )),
            params,
        );
        for j in 0..n {
            for i in j + 2..n {
                t.write(i, j, f64::zero());
            }
        }

        let h_reconstructed = &q * &t * q.adjoint();

        for i in 0..n {
            for j in 0..n {
                assert_approx_eq!(h_reconstructed.read(i, j), h.read(i, j));
            }
        }
    }

    #[test]
    fn test_multi_32() {
        let n = 32;
        let h = mat![
            [
                0.895840, 0.492184, 0.249037, 0.385869, 0.415107, 0.284650, 0.748490, 0.309102,
                0.261545, 0.747919, 0.635451, 0.370437, 0.541976, 0.414454, 0.033241, 0.223196,
                0.209735, 0.634490, 0.477402, 0.530707, 0.448061, 0.134809, 0.151881, 0.019189,
                0.419580, 0.489912, 0.757299, 0.964512, 0.403594, 0.790115, 0.470007, 0.754676
            ],
            [
                0.584245, 0.581423, 0.122522, 0.206506, 0.065297, 0.233078, 0.364388, 0.632608,
                0.490819, 0.684385, 0.453680, 0.897429, 0.801071, 0.841607, 0.697677, 0.507270,
                0.209635, 0.274525, 0.088630, 0.048420, 0.608626, 0.023219, 0.363352, 0.229030,
                0.314430, 0.958402, 0.090706, 0.383109, 0.705000, 0.674160, 0.005444, 0.379302
            ],
            [
                0.000000, 0.343457, 0.476776, 0.323714, 0.354512, 0.924569, 0.597751, 0.042292,
                0.232223, 0.463596, 0.755038, 0.004536, 0.906102, 0.427557, 0.635330, 0.796263,
                0.899874, 0.808427, 0.677221, 0.594849, 0.909910, 0.724250, 0.823062, 0.049812,
                0.166154, 0.659315, 0.854872, 0.441198, 0.631351, 0.879576, 0.267541, 0.176600
            ],
            [
                0.000000, 0.000000, 0.447571, 0.908237, 0.094083, 0.548680, 0.026379, 0.273365,
                0.974865, 0.734554, 0.387500, 0.421985, 0.958553, 0.440056, 0.774227, 0.248423,
                0.814815, 0.111030, 0.506024, 0.851596, 0.114510, 0.605463, 0.626837, 0.729912,
                0.232655, 0.647627, 0.302694, 0.338882, 0.842796, 0.642724, 0.707455, 0.872217
            ],
            [
                0.000000, 0.000000, 0.000000, 0.509472, 0.123015, 0.861445, 0.039835, 0.682465,
                0.213290, 0.236134, 0.853770, 0.378738, 0.165466, 0.600211, 0.542581, 0.604116,
                0.294857, 0.958330, 0.849626, 0.006263, 0.036329, 0.484993, 0.385435, 0.286740,
                0.702705, 0.555659, 0.935187, 0.810887, 0.547764, 0.062826, 0.941677, 0.364113
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.739291, 0.953674, 0.716995, 0.284982,
                0.263362, 0.615112, 0.694771, 0.737425, 0.512893, 0.363257, 0.691461, 0.525420,
                0.471937, 0.997745, 0.890505, 0.623372, 0.970136, 0.660782, 0.252220, 0.746361,
                0.876281, 0.180030, 0.696275, 0.524590, 0.971057, 0.369345, 0.934049, 0.970015
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.792937, 0.246874, 0.453803,
                0.206570, 0.791085, 0.370493, 0.779450, 0.016356, 0.904046, 0.298112, 0.114365,
                0.168019, 0.642973, 0.729209, 0.644699, 0.673161, 0.540481, 0.040544, 0.925033,
                0.685922, 0.305938, 0.926575, 0.162295, 0.878931, 0.447073, 0.388841, 0.137859
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.324264, 0.579979,
                0.234875, 0.265304, 0.444844, 0.773396, 0.842338, 0.972723, 0.019593, 0.549465,
                0.154628, 0.468435, 0.780542, 0.606772, 0.613991, 0.399483, 0.966505, 0.195359,
                0.031727, 0.599870, 0.701595, 0.019191, 0.700599, 0.754616, 0.530737, 0.417080
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.864317,
                0.941316, 0.030946, 0.767162, 0.367848, 0.159554, 0.184048, 0.091384, 0.496371,
                0.404363, 0.207715, 0.758445, 0.719842, 0.952013, 0.742002, 0.744598, 0.585577,
                0.067824, 0.397561, 0.027169, 0.806925, 0.629409, 0.320005, 0.374283, 0.850164
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.761267, 0.369822, 0.034261, 0.801222, 0.284250, 0.693414, 0.963132, 0.073582,
                0.853656, 0.538012, 0.787006, 0.263560, 0.116782, 0.223476, 0.090279, 0.544467,
                0.826400, 0.136176, 0.937127, 0.140393, 0.487297, 0.578832, 0.838340, 0.748319
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.715471, 0.167920, 0.905939, 0.023061, 0.942515, 0.383672, 0.559407,
                0.625033, 0.469114, 0.773845, 0.017133, 0.735326, 0.056741, 0.368998, 0.887253,
                0.176089, 0.516393, 0.448292, 0.801151, 0.006668, 0.315267, 0.333553, 0.599671
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.719893, 0.474976, 0.625507, 0.414931, 0.252818, 0.065984,
                0.615539, 0.041632, 0.064238, 0.450241, 0.360354, 0.251508, 0.872240, 0.202806,
                0.054714, 0.447498, 0.324232, 0.745859, 0.601976, 0.621448, 0.895272, 0.836070
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.992355, 0.086135, 0.307423, 0.840290, 0.969172,
                0.968561, 0.772006, 0.005932, 0.496436, 0.545797, 0.623991, 0.217294, 0.940454,
                0.894360, 0.950231, 0.459334, 0.625646, 0.352789, 0.103258, 0.063887, 0.834253
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.622028, 0.796841, 0.639503, 0.851427,
                0.133254, 0.273298, 0.941060, 0.512052, 0.848992, 0.824055, 0.609770, 0.342289,
                0.684608, 0.332425, 0.331361, 0.356139, 0.505000, 0.571246, 0.020497, 0.609717
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.528769, 0.976225, 0.816522,
                0.194512, 0.056273, 0.304916, 0.487897, 0.190118, 0.268621, 0.888859, 0.830107,
                0.722662, 0.951409, 0.153546, 0.324512, 0.412234, 0.608329, 0.875272, 0.073587
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.244978, 0.603447,
                0.234747, 0.215849, 0.695299, 0.425020, 0.091163, 0.085074, 0.446307, 0.644296,
                0.103132, 0.239552, 0.644311, 0.701797, 0.288408, 0.838756, 0.026208, 0.870627
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.867901,
                0.808224, 0.329383, 0.367983, 0.945933, 0.298928, 0.135797, 0.870544, 0.746703,
                0.491968, 0.041548, 0.856434, 0.265530, 0.001704, 0.351426, 0.597663, 0.663115
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.418417, 0.263897, 0.790961, 0.225846, 0.512225, 0.753096, 0.804326, 0.280598,
                0.243005, 0.612386, 0.942294, 0.617088, 0.497914, 0.927840, 0.424336, 0.632917
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.222989, 0.046276, 0.343741, 0.541652, 0.629739, 0.744535, 0.287013,
                0.340066, 0.918148, 0.446662, 0.435167, 0.964071, 0.712034, 0.630611, 0.720468
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.812013, 0.785432, 0.807885, 0.386583, 0.586813, 0.671112,
                0.353614, 0.612113, 0.348961, 0.002511, 0.102003, 0.574472, 0.587863, 0.584312
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.951638, 0.565235, 0.531977, 0.961445, 0.153953,
                0.592585, 0.455941, 0.055360, 0.479133, 0.661549, 0.923702, 0.353505, 0.581660
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.962089, 0.216366, 0.207517, 0.097300,
                0.674224, 0.273980, 0.420738, 0.325519, 0.965438, 0.428374, 0.303777, 0.205935
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.112642, 0.482438, 0.932636,
                0.902607, 0.740841, 0.573974, 0.063617, 0.344776, 0.543607, 0.669433, 0.900494
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.699461, 0.828957,
                0.512710, 0.326704, 0.730331, 0.272372, 0.889724, 0.600470, 0.960022, 0.865939
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.708931,
                0.892664, 0.887752, 0.490966, 0.779740, 0.315851, 0.396350, 0.647934, 0.858041
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.758454, 0.449067, 0.038270, 0.609523, 0.818255, 0.036108, 0.541736, 0.150148
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.030850, 0.201847, 0.709798, 0.942495, 0.362055, 0.009254, 0.583685
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.881405, 0.198124, 0.080659, 0.252470, 0.328455, 0.774040
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.086240, 0.814067, 0.241303, 0.191704, 0.672216
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.056984, 0.640654, 0.885082, 0.282523
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.545698, 0.321230, 0.783794
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.506647, 0.868296
            ],
        ];

        let mut q = Mat::from_fn(n, n, |i, j| if i == j { f64::one() } else { f64::zero() });

        let mut w_re = Mat::zeros(n, 1);
        let mut w_im = Mat::zeros(n, 1);

        let mut t = h.clone();
        let params = EvdParams {
            blocking_threshold: Some(15),
            ..Default::default()
        };
        multishift_qr(
            true,
            t.as_mut(),
            Some(q.as_mut()),
            w_re.as_mut(),
            w_im.as_mut(),
            0,
            n,
            f64::EPSILON,
            f64::MIN_POSITIVE,
            Parallelism::None,
            make_stack!(multishift_qr_req::<f64>(
                n,
                n,
                true,
                true,
                Parallelism::None,
                params,
            )),
            params,
        );
        for j in 0..n {
            for i in j + 2..n {
                t.write(i, j, f64::zero());
            }
        }

        let h_reconstructed = &q * &t * q.adjoint();

        for i in 0..n {
            for j in 0..n {
                assert_approx_eq!(h_reconstructed.read(i, j), h.read(i, j));
            }
        }
    }

    #[test]
    fn test_multi_63() {
        let h = mat![
            [
                0.471329, 0.707497, 0.352221, 0.945142, 0.731018, 0.343537, 0.771213, 0.810808,
                0.793544, 0.095497, 0.527420, 0.687952, 0.065130, 0.619127, 0.173395, 0.243835,
                0.036623, 0.542363, 0.670536, 0.212648, 0.391606, 0.428096, 0.546485, 0.749324,
                0.530726, 0.396340, 0.703546, 0.702380, 0.256754, 0.703589, 0.221020, 0.803774,
                0.945693, 0.810279, 0.918896, 0.414914, 0.083609, 0.155719, 0.619805, 0.634431,
                0.809856, 0.937686, 0.549730, 0.180061, 0.317235, 0.443740, 0.401900, 0.949929,
                0.691917, 0.074387, 0.934330, 0.153808, 0.842668, 0.167294, 0.050434, 0.943059,
                0.522410, 0.769138, 0.527005, 0.547074, 0.873159, 0.331555, 0.661714
            ],
            [
                0.390657, 0.670213, 0.982454, 0.204643, 0.099564, 0.862055, 0.162598, 0.927341,
                0.922960, 0.088886, 0.380536, 0.706517, 0.538582, 0.781954, 0.321461, 0.693731,
                0.939401, 0.936015, 0.300382, 0.521475, 0.525539, 0.385316, 0.585309, 0.476847,
                0.771530, 0.914799, 0.980300, 0.269330, 0.106700, 0.890915, 0.104459, 0.800256,
                0.951560, 0.653287, 0.854888, 0.433286, 0.518838, 0.616261, 0.824597, 0.593296,
                0.642487, 0.204391, 0.080193, 0.837078, 0.423710, 0.906396, 0.148474, 0.940165,
                0.922029, 0.562307, 0.520542, 0.758005, 0.627034, 0.976541, 0.822384, 0.608522,
                0.690644, 0.039345, 0.498192, 0.812126, 0.094682, 0.779217, 0.716666
            ],
            [
                0.000000, 0.249998, 0.385551, 0.900568, 0.361887, 0.379649, 0.881810, 0.399990,
                0.955748, 0.748852, 0.767547, 0.709336, 0.791408, 0.687191, 0.050613, 0.177541,
                0.318357, 0.610998, 0.289791, 0.349633, 0.377180, 0.533078, 0.869872, 0.037299,
                0.893271, 0.107826, 0.324060, 0.458382, 0.609544, 0.772459, 0.022543, 0.382789,
                0.097098, 0.944717, 0.966179, 0.608728, 0.292472, 0.946522, 0.878866, 0.136685,
                0.490803, 0.442319, 0.325728, 0.693296, 0.820098, 0.715850, 0.493264, 0.702384,
                0.480486, 0.346145, 0.795214, 0.621722, 0.113321, 0.576704, 0.210346, 0.445004,
                0.860162, 0.895683, 0.878616, 0.748717, 0.113194, 0.799646, 0.563749
            ],
            [
                0.000000, 0.000000, 0.451211, 0.713567, 0.728568, 0.432708, 0.431312, 0.033492,
                0.412753, 0.425699, 0.325236, 0.769875, 0.071937, 0.774085, 0.965014, 0.828533,
                0.954922, 0.788160, 0.672507, 0.805224, 0.554230, 0.328563, 0.859627, 0.035464,
                0.943124, 0.719711, 0.618427, 0.448262, 0.846814, 0.032690, 0.033196, 0.403312,
                0.652728, 0.518446, 0.537742, 0.292541, 0.435778, 0.006386, 0.300163, 0.279516,
                0.751122, 0.625826, 0.755909, 0.039926, 0.785631, 0.552131, 0.047674, 0.688207,
                0.750476, 0.681643, 0.813769, 0.137407, 0.978287, 0.900028, 0.638241, 0.383902,
                0.219723, 0.423036, 0.096022, 0.099336, 0.048419, 0.869296, 0.060274
            ],
            [
                0.000000, 0.000000, 0.000000, 0.971923, 0.639964, 0.741056, 0.626094, 0.405235,
                0.263972, 0.029055, 0.224554, 0.243604, 0.855985, 0.464387, 0.626211, 0.507577,
                0.341594, 0.934964, 0.157615, 0.239638, 0.250005, 0.537806, 0.549801, 0.825308,
                0.365004, 0.540100, 0.576966, 0.960500, 0.748158, 0.897963, 0.405714, 0.772973,
                0.617550, 0.363556, 0.793695, 0.461776, 0.686680, 0.939718, 0.326538, 0.416722,
                0.480510, 0.967141, 0.426313, 0.657213, 0.307889, 0.435038, 0.746696, 0.023467,
                0.167188, 0.868658, 0.841313, 0.198250, 0.903554, 0.585333, 0.870929, 0.722007,
                0.638906, 0.017168, 0.560623, 0.535360, 0.919469, 0.446904, 0.095656
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.252333, 0.327256, 0.042604, 0.746568,
                0.962798, 0.373143, 0.884695, 0.114759, 0.750273, 0.008323, 0.992342, 0.577739,
                0.123647, 0.388389, 0.724006, 0.128864, 0.743630, 0.981433, 0.426792, 0.940222,
                0.780615, 0.491701, 0.934418, 0.640474, 0.195516, 0.395932, 0.571322, 0.128187,
                0.471027, 0.229502, 0.363000, 0.380086, 0.868336, 0.878267, 0.781680, 0.903118,
                0.579789, 0.140763, 0.743334, 0.259173, 0.368718, 0.298506, 0.880891, 0.668640,
                0.695002, 0.556733, 0.714782, 0.539351, 0.049052, 0.430708, 0.495645, 0.055857,
                0.016414, 0.287261, 0.149517, 0.715162, 0.019451, 0.759066, 0.248017
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.101068, 0.182314, 0.333623,
                0.465323, 0.678302, 0.691266, 0.382368, 0.711563, 0.422348, 0.450919, 0.687702,
                0.850823, 0.505436, 0.831148, 0.622939, 0.042053, 0.335923, 0.833452, 0.604722,
                0.763554, 0.033969, 0.784093, 0.587345, 0.126119, 0.534263, 0.791144, 0.530774,
                0.402204, 0.636342, 0.055213, 0.349921, 0.429979, 0.623230, 0.943297, 0.639653,
                0.861724, 0.318577, 0.097105, 0.096352, 0.657220, 0.566792, 0.352498, 0.339411,
                0.650463, 0.933192, 0.717553, 0.182471, 0.173043, 0.841590, 0.290777, 0.298263,
                0.229432, 0.668146, 0.376169, 0.118658, 0.074191, 0.364562, 0.750235
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.652071, 0.530793,
                0.927036, 0.034230, 0.167494, 0.902349, 0.590741, 0.805017, 0.283426, 0.566659,
                0.654040, 0.301307, 0.481946, 0.002044, 0.111174, 0.325773, 0.079815, 0.656998,
                0.679171, 0.176470, 0.526729, 0.510844, 0.791339, 0.317950, 0.153202, 0.768102,
                0.813195, 0.182797, 0.557353, 0.816305, 0.752258, 0.976863, 0.773404, 0.756378,
                0.127835, 0.316817, 0.337825, 0.712814, 0.332875, 0.439860, 0.767020, 0.250887,
                0.124324, 0.718531, 0.241141, 0.868258, 0.201957, 0.124954, 0.644152, 0.741508,
                0.149396, 0.776884, 0.533407, 0.005019, 0.587207, 0.085723, 0.757688
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.930331,
                0.010867, 0.197635, 0.084432, 0.725446, 0.121400, 0.114763, 0.154413, 0.519925,
                0.202541, 0.922663, 0.905313, 0.583953, 0.961928, 0.671621, 0.295900, 0.315302,
                0.285507, 0.464995, 0.707526, 0.566137, 0.511331, 0.218164, 0.110146, 0.943963,
                0.663649, 0.652564, 0.652022, 0.009590, 0.801449, 0.049297, 0.182070, 0.298892,
                0.709786, 0.539233, 0.804601, 0.432649, 0.002864, 0.281380, 0.926649, 0.584708,
                0.357428, 0.790069, 0.165310, 0.090105, 0.707378, 0.849492, 0.453500, 0.397462,
                0.486707, 0.830573, 0.616181, 0.727474, 0.919983, 0.095947, 0.262766
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.806076, 0.458292, 0.046170, 0.919739, 0.697977, 0.988237, 0.511846, 0.475719,
                0.847145, 0.704390, 0.655839, 0.153343, 0.429450, 0.937956, 0.974258, 0.932827,
                0.197146, 0.871771, 0.203858, 0.363547, 0.053243, 0.298077, 0.791229, 0.449783,
                0.694805, 0.563696, 0.317210, 0.080866, 0.551051, 0.431791, 0.858560, 0.097850,
                0.288632, 0.958346, 0.970960, 0.443346, 0.775688, 0.416729, 0.939468, 0.993247,
                0.015829, 0.874304, 0.749902, 0.739098, 0.292871, 0.871213, 0.837197, 0.522220,
                0.701806, 0.986479, 0.268497, 0.270136, 0.263265, 0.672888, 0.220938
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.057361, 0.731795, 0.466344, 0.983472, 0.408559, 0.134386, 0.782429,
                0.070605, 0.509286, 0.169110, 0.210032, 0.184445, 0.827286, 0.466636, 0.097771,
                0.735163, 0.013467, 0.655605, 0.395358, 0.821475, 0.027649, 0.289450, 0.728915,
                0.622923, 0.749030, 0.167692, 0.238058, 0.644268, 0.956126, 0.179054, 0.467438,
                0.556385, 0.575229, 0.152773, 0.954230, 0.551599, 0.647023, 0.211323, 0.494855,
                0.760563, 0.478946, 0.890622, 0.091748, 0.296879, 0.805926, 0.997492, 0.162544,
                0.220510, 0.464210, 0.179808, 0.649383, 0.513834, 0.649821, 0.691381
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.333651, 0.182264, 0.738327, 0.899536, 0.044006, 0.545597,
                0.584323, 0.918004, 0.126205, 0.378106, 0.000638, 0.986749, 0.248798, 0.561642,
                0.478128, 0.959126, 0.925218, 0.278698, 0.042217, 0.405885, 0.470239, 0.346746,
                0.858529, 0.059967, 0.278196, 0.523302, 0.123482, 0.483097, 0.878119, 0.096125,
                0.230490, 0.413070, 0.480987, 0.147695, 0.972841, 0.915872, 0.509918, 0.438945,
                0.202974, 0.071056, 0.982675, 0.234982, 0.895803, 0.013092, 0.189989, 0.228932,
                0.835576, 0.969768, 0.371293, 0.910171, 0.668608, 0.864307, 0.814922
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.632243, 0.520019, 0.487075, 0.647965, 0.067971,
                0.738721, 0.952760, 0.542778, 0.720878, 0.176673, 0.710916, 0.063673, 0.167211,
                0.927520, 0.575541, 0.878695, 0.496095, 0.299147, 0.549855, 0.179045, 0.133649,
                0.636806, 0.060261, 0.692491, 0.550958, 0.245698, 0.316341, 0.919609, 0.103931,
                0.524301, 0.416250, 0.560819, 0.202214, 0.717274, 0.543218, 0.722470, 0.430136,
                0.871626, 0.471557, 0.890412, 0.385671, 0.583099, 0.834099, 0.028127, 0.196494,
                0.294347, 0.341889, 0.158749, 0.897989, 0.195440, 0.603711, 0.121908
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.617842, 0.771751, 0.955641, 0.973338,
                0.060793, 0.933017, 0.217330, 0.428476, 0.960418, 0.804578, 0.246311, 0.720828,
                0.582253, 0.383689, 0.111359, 0.431775, 0.738233, 0.900730, 0.940150, 0.055358,
                0.656581, 0.596321, 0.820989, 0.800243, 0.051147, 0.872542, 0.935633, 0.704173,
                0.335839, 0.848765, 0.627017, 0.535353, 0.515942, 0.776423, 0.357953, 0.428258,
                0.671906, 0.001047, 0.257457, 0.931851, 0.174954, 0.917053, 0.362174, 0.076865,
                0.152247, 0.095362, 0.543238, 0.967612, 0.365605, 0.956459, 0.743537
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.825939, 0.571120, 0.552922,
                0.135300, 0.132294, 0.678044, 0.726895, 0.739711, 0.036668, 0.313992, 0.482755,
                0.207549, 0.298782, 0.207065, 0.144913, 0.147549, 0.580983, 0.864411, 0.051965,
                0.127675, 0.525005, 0.430779, 0.748795, 0.770718, 0.109832, 0.937897, 0.513030,
                0.040881, 0.217129, 0.487057, 0.899769, 0.336370, 0.351659, 0.699575, 0.269033,
                0.517315, 0.692714, 0.334091, 0.404520, 0.045924, 0.967382, 0.356084, 0.645442,
                0.200997, 0.727590, 0.347395, 0.937974, 0.432052, 0.609547, 0.949460
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.103464, 0.910034,
                0.395068, 0.626080, 0.836573, 0.569551, 0.579654, 0.188389, 0.708035, 0.236670,
                0.507742, 0.077670, 0.476730, 0.142077, 0.768759, 0.961693, 0.564986, 0.407837,
                0.773780, 0.940672, 0.410894, 0.793928, 0.545650, 0.646820, 0.426645, 0.959100,
                0.643719, 0.914480, 0.573200, 0.829013, 0.548565, 0.756338, 0.941820, 0.661630,
                0.214981, 0.524893, 0.300833, 0.910478, 0.280354, 0.045218, 0.612717, 0.788497,
                0.547293, 0.614951, 0.934190, 0.954700, 0.147926, 0.646601, 0.890367
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.128334,
                0.670745, 0.255184, 0.487234, 0.319756, 0.330827, 0.883481, 0.179967, 0.915516,
                0.588025, 0.703003, 0.304631, 0.488980, 0.613056, 0.355503, 0.165319, 0.569403,
                0.187109, 0.007294, 0.766319, 0.116012, 0.378333, 0.150446, 0.039042, 0.758017,
                0.040064, 0.421488, 0.012347, 0.805034, 0.509113, 0.567890, 0.229175, 0.151718,
                0.856004, 0.342705, 0.370705, 0.944854, 0.238169, 0.165361, 0.155724, 0.532457,
                0.906388, 0.664469, 0.142961, 0.252622, 0.752598, 0.696146, 0.799966
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.050322, 0.265062, 0.093633, 0.434402, 0.695266, 0.398954, 0.367148, 0.429999,
                0.530175, 0.327977, 0.809410, 0.784776, 0.760322, 0.985646, 0.771230, 0.359149,
                0.945376, 0.022893, 0.591423, 0.056510, 0.007163, 0.841015, 0.926796, 0.329611,
                0.357482, 0.946781, 0.959423, 0.521878, 0.970083, 0.777624, 0.402829, 0.352146,
                0.249312, 0.524888, 0.300729, 0.505039, 0.968562, 0.298689, 0.417337, 0.278229,
                0.164533, 0.479338, 0.695559, 0.325852, 0.729625, 0.440588, 0.268737
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.321619, 0.954314, 0.780782, 0.297053, 0.055923, 0.591197, 0.146459,
                0.700695, 0.851798, 0.744687, 0.165057, 0.608443, 0.955879, 0.857050, 0.249295,
                0.066488, 0.231500, 0.472161, 0.557250, 0.137957, 0.049279, 0.883183, 0.673387,
                0.060654, 0.301115, 0.365485, 0.784362, 0.538310, 0.117218, 0.749519, 0.568648,
                0.789537, 0.246896, 0.029047, 0.220633, 0.741582, 0.731372, 0.948782, 0.833137,
                0.734425, 0.630732, 0.433372, 0.200971, 0.862391, 0.724083, 0.870948
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.993755, 0.669979, 0.292359, 0.942558, 0.196782, 0.960420,
                0.128945, 0.695161, 0.680078, 0.898443, 0.528033, 0.073173, 0.812920, 0.747343,
                0.872205, 0.234288, 0.081215, 0.505880, 0.374385, 0.413327, 0.802354, 0.279630,
                0.820162, 0.190095, 0.057743, 0.122577, 0.204983, 0.864067, 0.775379, 0.166969,
                0.685553, 0.067123, 0.723600, 0.650825, 0.077226, 0.206058, 0.098741, 0.735729,
                0.019222, 0.619724, 0.882233, 0.095233, 0.622359, 0.230179, 0.162153
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.914219, 0.292127, 0.819216, 0.475440, 0.419489,
                0.049556, 0.140818, 0.780565, 0.034798, 0.881872, 0.542852, 0.321052, 0.636182,
                0.158786, 0.411837, 0.009036, 0.728390, 0.043966, 0.157989, 0.997226, 0.463796,
                0.522904, 0.873463, 0.553241, 0.220606, 0.088968, 0.364752, 0.760449, 0.936882,
                0.471582, 0.967434, 0.983092, 0.856739, 0.821424, 0.849692, 0.188478, 0.234940,
                0.670112, 0.676833, 0.655943, 0.371985, 0.581804, 0.302006, 0.014608
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.150121, 0.430199, 0.509366, 0.420602,
                0.237205, 0.460560, 0.563018, 0.841479, 0.869233, 0.065867, 0.567763, 0.735803,
                0.124032, 0.374551, 0.086493, 0.041802, 0.249332, 0.604022, 0.499945, 0.331346,
                0.741456, 0.496159, 0.525313, 0.168191, 0.823061, 0.064389, 0.807584, 0.502211,
                0.683453, 0.897813, 0.984829, 0.471068, 0.109265, 0.896220, 0.350877, 0.119979,
                0.192104, 0.449967, 0.383662, 0.803322, 0.952073, 0.327173, 0.896819
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.345980, 0.291262, 0.451772,
                0.879859, 0.578493, 0.181668, 0.413203, 0.298116, 0.355098, 0.917939, 0.408550,
                0.469510, 0.137953, 0.846158, 0.992262, 0.852144, 0.891884, 0.752652, 0.128656,
                0.134262, 0.166518, 0.829518, 0.694012, 0.704149, 0.179452, 0.604025, 0.081876,
                0.620042, 0.381227, 0.592020, 0.394564, 0.152763, 0.986690, 0.857370, 0.419942,
                0.700065, 0.810717, 0.548788, 0.962071, 0.371389, 0.345329, 0.621041
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.538458, 0.493781,
                0.579949, 0.774404, 0.811160, 0.031164, 0.875810, 0.867280, 0.519693, 0.301846,
                0.130705, 0.810443, 0.237980, 0.521364, 0.725355, 0.611043, 0.677615, 0.418368,
                0.534078, 0.032398, 0.265383, 0.750549, 0.422733, 0.389686, 0.739241, 0.352608,
                0.894875, 0.775730, 0.964292, 0.855536, 0.862128, 0.102291, 0.427782, 0.395151,
                0.726327, 0.731716, 0.069448, 0.702388, 0.882323, 0.561503, 0.017183
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.787960,
                0.167387, 0.825669, 0.106668, 0.566647, 0.642244, 0.066923, 0.067144, 0.294421,
                0.738481, 0.762524, 0.772121, 0.196861, 0.225197, 0.519602, 0.767302, 0.147066,
                0.679314, 0.024191, 0.648860, 0.294886, 0.573572, 0.042879, 0.160595, 0.131667,
                0.438811, 0.096806, 0.718749, 0.353918, 0.896721, 0.736696, 0.536587, 0.434771,
                0.224876, 0.565288, 0.222983, 0.687854, 0.315662, 0.844507, 0.158599
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.639668, 0.614831, 0.088017, 0.552701, 0.537497, 0.332657, 0.415742, 0.543572,
                0.602150, 0.634064, 0.547921, 0.211351, 0.747884, 0.791033, 0.812799, 0.710241,
                0.363352, 0.021586, 0.012039, 0.445813, 0.985272, 0.124986, 0.149151, 0.085474,
                0.504163, 0.807524, 0.491996, 0.789674, 0.067727, 0.574634, 0.340954, 0.082003,
                0.022856, 0.787410, 0.189319, 0.717430, 0.917578, 0.613973, 0.472661
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.566107, 0.502870, 0.496664, 0.332109, 0.189717, 0.538124, 0.048149,
                0.088833, 0.863326, 0.023427, 0.566033, 0.518604, 0.455140, 0.866196, 0.684874,
                0.728038, 0.121254, 0.254995, 0.715422, 0.420717, 0.146540, 0.907117, 0.550555,
                0.902458, 0.823986, 0.890149, 0.770903, 0.744895, 0.683412, 0.876796, 0.169095,
                0.173063, 0.006777, 0.116902, 0.013995, 0.484944, 0.077156, 0.375577
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.320441, 0.562477, 0.080510, 0.164586, 0.857731, 0.655256,
                0.890089, 0.748000, 0.961405, 0.402961, 0.343957, 0.015382, 0.250814, 0.364652,
                0.215188, 0.065695, 0.869749, 0.375092, 0.780361, 0.210538, 0.102848, 0.101306,
                0.713029, 0.297260, 0.792159, 0.140317, 0.798725, 0.805319, 0.688061, 0.327894,
                0.714547, 0.759421, 0.643988, 0.626784, 0.253879, 0.042110, 0.561077
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.525519, 0.124259, 0.691961, 0.268966, 0.152779,
                0.011801, 0.365655, 0.759321, 0.330548, 0.745904, 0.564996, 0.359044, 0.571254,
                0.602305, 0.774709, 0.639320, 0.956428, 0.518505, 0.597369, 0.209782, 0.402660,
                0.838692, 0.883218, 0.817463, 0.304741, 0.291167, 0.568009, 0.899216, 0.589436,
                0.065252, 0.275585, 0.286562, 0.258983, 0.475170, 0.578467, 0.639263
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.871484, 0.569782, 0.923951, 0.682707,
                0.685499, 0.370055, 0.709012, 0.308717, 0.898987, 0.881012, 0.783547, 0.186491,
                0.062373, 0.890061, 0.452075, 0.195765, 0.507293, 0.022504, 0.388442, 0.146899,
                0.234462, 0.718260, 0.132596, 0.014555, 0.264358, 0.073670, 0.886954, 0.788218,
                0.816820, 0.181396, 0.408453, 0.585111, 0.664769, 0.093276, 0.644903
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.862883, 0.301375, 0.259260,
                0.646717, 0.664225, 0.073917, 0.283041, 0.242954, 0.414422, 0.289215, 0.991763,
                0.949328, 0.032650, 0.192436, 0.481743, 0.303482, 0.564919, 0.872959, 0.154532,
                0.140791, 0.985868, 0.445961, 0.796440, 0.933053, 0.619375, 0.666891, 0.300755,
                0.518233, 0.601653, 0.676000, 0.757847, 0.702324, 0.533810, 0.386234
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.560684, 0.411427,
                0.225155, 0.257363, 0.421568, 0.275746, 0.306913, 0.314126, 0.054726, 0.321248,
                0.036699, 0.322176, 0.596407, 0.637525, 0.898528, 0.210334, 0.751982, 0.773484,
                0.486409, 0.506715, 0.930019, 0.933749, 0.168900, 0.888197, 0.841166, 0.775251,
                0.409127, 0.544451, 0.945227, 0.201067, 0.230058, 0.011978, 0.683173
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.843707,
                0.473753, 0.884039, 0.103654, 0.242211, 0.285426, 0.814021, 0.039819, 0.563370,
                0.549792, 0.079171, 0.819496, 0.007372, 0.346101, 0.000770, 0.960185, 0.284079,
                0.099634, 0.305986, 0.323563, 0.577051, 0.414000, 0.736568, 0.143752, 0.810491,
                0.304641, 0.924444, 0.650471, 0.441430, 0.711409, 0.071921, 0.788928
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.360859, 0.693067, 0.693169, 0.868613, 0.236832, 0.474167, 0.457248, 0.849678,
                0.052820, 0.054816, 0.570999, 0.534548, 0.949570, 0.664881, 0.432254, 0.910584,
                0.886445, 0.871505, 0.493898, 0.623795, 0.181700, 0.138376, 0.021610, 0.855862,
                0.953979, 0.927132, 0.488138, 0.498304, 0.371985, 0.888913, 0.270117
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.286591, 0.995506, 0.032365, 0.202751, 0.180217, 0.518968, 0.711249,
                0.692286, 0.165984, 0.621816, 0.941068, 0.501897, 0.130162, 0.103924, 0.296868,
                0.508031, 0.402614, 0.473505, 0.934292, 0.727884, 0.241396, 0.656143, 0.082136,
                0.879273, 0.486917, 0.248886, 0.802905, 0.172131, 0.366586, 0.822351
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.118585, 0.228560, 0.892964, 0.942608, 0.997551, 0.995657,
                0.750532, 0.437387, 0.204915, 0.291216, 0.905010, 0.377968, 0.234637, 0.468398,
                0.068751, 0.844538, 0.993301, 0.675911, 0.639793, 0.189963, 0.072921, 0.513233,
                0.837688, 0.329283, 0.429525, 0.898185, 0.668677, 0.139443, 0.829080
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.188208, 0.783814, 0.165250, 0.368781, 0.207734,
                0.066240, 0.535570, 0.339133, 0.048907, 0.676866, 0.658358, 0.611406, 0.488035,
                0.654296, 0.167999, 0.228813, 0.457179, 0.944747, 0.649755, 0.695383, 0.991380,
                0.072842, 0.525342, 0.973807, 0.639318, 0.388186, 0.542923, 0.170159
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.874923, 0.447731, 0.289869, 0.744775,
                0.170462, 0.577232, 0.529011, 0.269001, 0.825693, 0.850720, 0.800630, 0.965913,
                0.426271, 0.595582, 0.363931, 0.200617, 0.520631, 0.186107, 0.504003, 0.508111,
                0.331448, 0.631017, 0.892951, 0.557269, 0.927148, 0.117911, 0.996670
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.145245, 0.765040, 0.023057,
                0.414437, 0.442364, 0.647694, 0.697366, 0.643305, 0.639494, 0.002926, 0.620336,
                0.390856, 0.222545, 0.489109, 0.178915, 0.026394, 0.783177, 0.595012, 0.199271,
                0.471820, 0.471403, 0.415482, 0.019270, 0.099026, 0.876464, 0.476354
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.308339, 0.939196,
                0.053888, 0.675363, 0.062049, 0.847298, 0.309829, 0.079971, 0.712394, 0.587305,
                0.198209, 0.769633, 0.146552, 0.183823, 0.713434, 0.867346, 0.617666, 0.393632,
                0.474708, 0.253089, 0.370446, 0.321414, 0.987991, 0.193431, 0.538840
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.001628,
                0.736132, 0.956614, 0.565938, 0.689599, 0.139013, 0.607261, 0.922666, 0.703021,
                0.623029, 0.625977, 0.288470, 0.594142, 0.054956, 0.402750, 0.044795, 0.752144,
                0.737346, 0.079213, 0.953601, 0.228317, 0.950017, 0.559359, 0.296047
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.291474, 0.893327, 0.842537, 0.063992, 0.409020, 0.810916, 0.377763, 0.322520,
                0.402443, 0.905683, 0.048850, 0.098189, 0.314053, 0.783679, 0.723088, 0.423404,
                0.535251, 0.819774, 0.634137, 0.643748, 0.824785, 0.358144, 0.245043
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.057630, 0.949845, 0.158038, 0.601297, 0.824650, 0.660627, 0.113690,
                0.843997, 0.372026, 0.188104, 0.655164, 0.025428, 0.391168, 0.060325, 0.700706,
                0.160417, 0.127880, 0.607506, 0.197686, 0.876882, 0.279937, 0.726450
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.033275, 0.212286, 0.569210, 0.185610, 0.026777, 0.235483,
                0.277490, 0.894238, 0.509124, 0.180807, 0.289290, 0.895419, 0.085453, 0.556812,
                0.256709, 0.757390, 0.179171, 0.894325, 0.752170, 0.219809, 0.628171
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.539717, 0.542037, 0.719869, 0.220899, 0.316028,
                0.344909, 0.005820, 0.003484, 0.436912, 0.719304, 0.874256, 0.548070, 0.421794,
                0.614683, 0.472670, 0.267402, 0.297171, 0.306715, 0.308022, 0.262279
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.552993, 0.239458, 0.218459, 0.341668,
                0.191117, 0.096201, 0.921410, 0.924509, 0.379986, 0.459116, 0.535478, 0.227914,
                0.048228, 0.617268, 0.392331, 0.533026, 0.747950, 0.362611, 0.988167
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.433886, 0.958143, 0.102487,
                0.738938, 0.636731, 0.442812, 0.746198, 0.664181, 0.164605, 0.273800, 0.957371,
                0.608375, 0.700961, 0.183773, 0.363996, 0.537675, 0.583714, 0.836833
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.479099, 0.076537,
                0.416699, 0.958339, 0.987406, 0.157750, 0.971100, 0.985621, 0.590406, 0.311220,
                0.950718, 0.614852, 0.166385, 0.834438, 0.181470, 0.016776, 0.374244
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.118618,
                0.169020, 0.393729, 0.308834, 0.793447, 0.337371, 0.754953, 0.316461, 0.625055,
                0.764049, 0.528747, 0.521148, 0.270870, 0.563530, 0.814726, 0.576660
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.800041, 0.264832, 0.808932, 0.345071, 0.305018, 0.238888, 0.924637, 0.807370,
                0.735855, 0.639500, 0.394329, 0.046158, 0.652393, 0.007147, 0.371862
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.917810, 0.879081, 0.236476, 0.845774, 0.682276, 0.925104, 0.511553,
                0.959155, 0.096447, 0.700227, 0.228581, 0.578347, 0.121326, 0.420843
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.267984, 0.361054, 0.197019, 0.493756, 0.253383, 0.568701,
                0.593510, 0.960850, 0.879049, 0.467828, 0.150496, 0.693201, 0.517899
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.677260, 0.533689, 0.490560, 0.229967, 0.839060,
                0.689383, 0.201250, 0.188357, 0.110341, 0.132560, 0.001984, 0.568096
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.572511, 0.712151, 0.384552, 0.080542,
                0.835457, 0.137682, 0.777503, 0.597384, 0.354857, 0.932363, 0.718881
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.485678, 0.250686, 0.374568,
                0.800058, 0.943406, 0.290507, 0.733454, 0.380244, 0.377067, 0.799318
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.141217, 0.985572,
                0.211096, 0.366295, 0.753331, 0.100392, 0.887683, 0.440172, 0.936085
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.650734,
                0.110166, 0.601598, 0.596148, 0.993499, 0.596591, 0.655336, 0.297806
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.561178, 0.479528, 0.371016, 0.881205, 0.672228, 0.492059, 0.697566
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.648892, 0.921668, 0.059269, 0.614716, 0.864352, 0.643155
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.849166, 0.303690, 0.622688, 0.155835, 0.668703
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.775990, 0.058993, 0.156178, 0.155920
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.259461, 0.714638, 0.662105
            ],
            [
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.032925, 0.397687
            ],
        ];
        let n = h.nrows();

        let mut q = Mat::from_fn(n, n, |i, j| if i == j { f64::one() } else { f64::zero() });

        let mut w_re = Mat::zeros(n, 1);
        let mut w_im = Mat::zeros(n, 1);

        let mut t = h.clone();
        let params = EvdParams {
            blocking_threshold: Some(15),
            ..Default::default()
        };
        let (_, n_aed, n_sweep) = multishift_qr(
            true,
            t.as_mut(),
            Some(q.as_mut()),
            w_re.as_mut(),
            w_im.as_mut(),
            0,
            n,
            f64::EPSILON,
            f64::MIN_POSITIVE,
            Parallelism::None,
            make_stack!(multishift_qr_req::<f64>(
                n,
                n,
                true,
                true,
                Parallelism::None,
                params,
            )),
            params,
        );
        // asserts to ensure that we don't mess up the shift computation and slow down convergence
        assert!(n_aed <= 30);
        assert!(n_sweep <= 16);

        for j in 0..n {
            for i in j + 2..n {
                t.write(i, j, f64::zero());
            }
        }

        let h_reconstructed = &q * &t * q.adjoint();

        for i in 0..n {
            for j in 0..n {
                assert_approx_eq!(h_reconstructed.read(i, j), h.read(i, j));
            }
        }
    }
}
