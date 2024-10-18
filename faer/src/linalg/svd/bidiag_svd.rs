// Algorithm ported from Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
// Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
// Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
// Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>
// Copyright (C) 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2014-2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::{internal_prelude::*, perm::swap_cols_idx};
use core::mem::swap;
use linalg::jacobi::JacobiRotation;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SvdError {
    NoConvergence,
}

#[math]
#[allow(dead_code)]
fn bidiag_to_mat<'N, C: RealContainer, T: RealField<C, MathCtx: Default>>(
    diag: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    subdiag: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
) -> Mat<C, T, Dim<'N>, Dim<'N>> {
    let n = diag.nrows();
    let ctx: &Ctx<C, T> = &ctx();
    let mut m = Mat::zeros_with(ctx, n, n);

    help!(C);
    {
        let mut m = m.as_mut();
        for i in n.indices() {
            write1!(m[(i, i)] = math(copy(diag[i])));
            if let Some(i1) = n.try_check(*i + 1) {
                write1!(m[(i1, i)] = math(copy(subdiag[i])));
            }
        }
    }
    m
}

#[math]
#[allow(dead_code)]
fn bidiag_to_mat2<'N, C: RealContainer, T: RealField<C, MathCtx: Default>>(
    diag: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    subdiag: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
) -> Mat<C, T, usize, Dim<'N>> {
    let n = diag.nrows();
    let ctx: &Ctx<C, T> = &ctx();
    let mut m = Mat::zeros_with(ctx, *n + 1, n);

    help!(C);
    {
        let mut m = m.as_mut();
        for i in n.indices() {
            write1!(m[(*i, i)] = math(copy(diag[i])));
            write1!(m[(*i + 1, i)] = math(copy(subdiag[i])));
        }
    }
    m
}

#[math]
#[allow(dead_code)]
fn arrow_to_mat<'N, C: RealContainer, T: RealField<C, MathCtx: Default>>(
    diag: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    col0: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
) -> Mat<C, T, usize, Dim<'N>> {
    let n = diag.nrows();
    let ctx: &Ctx<C, T> = &ctx();
    let mut m = Mat::zeros_with(ctx, *n + 1, n);

    help!(C);
    {
        let mut m = m.as_mut();
        for i in n.indices() {
            write1!(m[(*i, n.idx(0))] = math(copy(col0[i])));
            write1!(m[(*i, i)] = math(copy(diag[i])));
        }
    }
    m
}

#[math]
fn qr_algorithm<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,

    diag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    subdiag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    u: Option<MatMut<'_, C, T, Dim<'N>, Dim<'N>>>,
    v: Option<MatMut<'_, C, T, Dim<'N>, Dim<'N>>>,
) -> Result<(), SvdError> {
    let n = diag.nrows();
    let eps = math(eps());
    let sml = math(min_positive());

    if *n == 0 {
        return Ok(());
    }

    let max_iters = Ord::max(30, ctx.nbits())
        .saturating_mul(*n)
        .saturating_mul(*n);

    let last = n.idx(*n - 1);

    let max = math.max(diag.norm_max_with(ctx), subdiag.norm_max_with(ctx));
    if math.is_zero(max) {
        return Ok(());
    }
    help!(C);

    let max_inv = math(recip(max));

    let mut diag = diag;
    let mut subdiag = subdiag;
    let mut u = u;
    let mut v = v;

    for mut x in diag.rb_mut().iter_mut() {
        write1!(x, math(x * max_inv));
    }
    for mut x in subdiag.rb_mut().iter_mut() {
        write1!(x, math(x * max_inv));
    }

    {
        let eps2 = math(eps * eps);

        for iter in 0..max_iters {
            for i0 in zero().to(last.excl()) {
                let i1 = n.idx(*i0 + 1);
                if math(abs2(subdiag[i0]) <= eps2 * abs(diag[i0] * diag[i1]) + sml) {
                    write1!(subdiag[i0] = math.zero());
                }
            }

            for i in n.indices() {
                if math(abs(diag[i]) <= eps) {
                    write1!(diag[i] = math.zero());
                }
            }

            let mut end = n.end();
            while *end >= 2 && math(abs2(subdiag[n.idx(*end - 2)]) <= sml) {
                end = n.idx_inc(*end - 1)
            }

            if *end == 1 {
                break;
            }

            let mut start = n.idx(*end - 1);
            while *start >= 1 && !math(is_zero(subdiag[n.idx(*start - 1)])) {
                start = n.idx(*start - 1);
            }

            let mut found_zero_diag = false;

            for i in start.to_incl().to(n.idx_inc(*end - 1)) {
                if math(is_zero(diag[i])) {
                    found_zero_diag = true;

                    let mut val = math(copy(subdiag[i]));
                    write1!(subdiag[i] = math(zero()));
                    for j in i.next().to(end) {
                        let rot = math(JacobiRotation::make_givens(ctx, copy(diag[j]), copy(val)));
                        write1!(diag[j] = math(abs(rot.c * diag[j] - rot.s * val)));
                        if j.next() < end {
                            val = math(-rot.s * subdiag[j]);
                            write1!(subdiag[j] = math(rot.c * subdiag[j]));
                        }
                        if let Some(v) = v.rb_mut() {
                            let (i, j) = v.two_cols_mut(i, j);
                            rot.apply_on_the_right_in_place(ctx, i, j);
                        }
                    }
                }
            }

            if found_zero_diag {
                continue;
            }

            let end2 = n.idx(*end - 2);
            let end1 = n.idx(*end - 1);

            let t00 = if *end - *start == 2 {
                math(abs2(diag[end2]))
            } else {
                math(abs2(diag[end2] + abs2(subdiag[n.idx(*end - 3)])))
            };
            let t11 = math(abs2(diag[end1]) + abs2(subdiag[end2]));
            let t01 = math(diag[end2] * subdiag[end2]);

            let t01_2 = math(abs2(t01));

            let mu;
            if math(t01_2 > sml) {
                let d = math(mul_pow2((t00 - t11), from_f64(0.5)));
                let mut delta = math(sqrt(abs2(d) + t01_2));
                if math(lt_zero(d)) {
                    delta = math(-delta);
                }
                mu = math(t11 - (t01_2 / (d + delta)));
            } else {
                mu = math(copy(t11));
            }

            let mut y = math(abs2(diag[start]) - mu);
            let mut z = math(diag[start] * subdiag[start]);

            for k in start.to_incl().to(end1.excl()) {
                let rot = JacobiRotation::make_givens(ctx, math(copy(y)), math(copy(z)));
                if k > start {
                    write1!(subdiag[n.idx(*k - 1)] = math(abs(rot.c * y - rot.s * z)));
                }

                let mut diag_k = math(copy(diag[k]));

                let tmp = math((
                    rot.c * diag_k - rot.s * subdiag[k],
                    rot.s * diag_k + rot.c * subdiag[k],
                ));

                diag_k = tmp.0;
                write1!(subdiag[k] = tmp.1);

                let k1 = n.idx(*k + 1);

                y = math(copy(diag_k));
                z = math(-rot.s * diag[k1]);
                write1!(diag[k1] = math(rot.c * diag[k1]));

                if let Some(u) = u.rb_mut() {
                    let (k, k1) = u.two_cols_mut(k, k1);
                    rot.apply_on_the_right_in_place(ctx, k, k1);
                }

                let rot = JacobiRotation::make_givens(ctx, math(copy(y)), math(copy(z)));

                diag_k = math(abs(rot.c * y - rot.s * z));
                write1!(diag[k] = diag_k);

                let tmp = math((
                    rot.c * subdiag[k] - rot.s * diag[k1],
                    rot.s * subdiag[k] + rot.c * diag[k1],
                ));
                write1!(subdiag[k] = tmp.0);
                write1!(diag[k1] = tmp.1);

                if *k < *end - 2 {
                    y = math(copy(subdiag[k]));
                    z = math(-rot.s * subdiag[k1]);
                    write1!(subdiag[k1] = math(rot.c * subdiag[k1]));
                }

                if let Some(v) = v.rb_mut() {
                    let (k, k1) = v.two_cols_mut(k, k1);
                    rot.apply_on_the_right_in_place(ctx, k, k1);
                }
            }

            if iter + 1 == max_iters {
                for mut x in diag.rb_mut().iter_mut() {
                    write1!(x, math(x * max));
                }
                for mut x in subdiag.rb_mut().iter_mut() {
                    write1!(x, math(x * max));
                }

                return Err(SvdError::NoConvergence);
            }
        }
    }

    for j in n.indices() {
        let mut d = diag.rb_mut().at_mut(j);

        if math(lt_zero(d)) {
            write1!(d, math(-d));
            if let Some(mut v) = v.rb_mut() {
                for i in n.indices() {
                    write1!(v[(i, j)] = math(-v[(i, j)]));
                }
            }
        }
    }

    for k in n.indices() {
        let mut max = math(zero());
        let mut idx = k;

        for kk in k.to_incl().to(n.end()) {
            if math(diag[kk] > max) {
                max = math(copy(diag[kk]));
                idx = kk;
            }
        }

        if k != idx {
            let dk = math(copy(diag[k]));
            let di = math(copy(diag[idx]));

            write1!(diag[idx] = dk);
            write1!(diag[k] = di);

            if let Some(u) = u.rb_mut() {
                swap_cols_idx(ctx, u, k, idx);
            }
            if let Some(v) = v.rb_mut() {
                swap_cols_idx(ctx, v, k, idx);
            }
        }
    }

    for mut x in diag.rb_mut().iter_mut() {
        write1!(x, math(x * max));
    }
    for mut x in subdiag.rb_mut().iter_mut() {
        write1!(x, math(x * max));
    }

    Ok(())
}

#[math]
fn compute_svd_of_m<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    um: Option<MatMut<'_, C, T, usize, usize, ContiguousFwd>>,
    vm: Option<MatMut<'_, C, T, Dim<'N>, Dim<'N>, ContiguousFwd>>,
    diag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    col0: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    outer_perm: &Array<'N, Idx<'N>>,
    stack: &mut DynStack,
) {
    help!(C);

    let mut diag = diag;
    let mut um = um;
    let mut vm = vm;
    let n = diag.nrows();

    write1!(diag[n.idx(0)] = math.zero());
    let mut actual_n = n.end();
    while *actual_n > 1 && math(is_zero(diag[n.idx(*actual_n - 1)])) {
        actual_n = n.idx_inc(*actual_n - 1);
    }

    let (perm, stack) = stack.collect(
        col0.iter()
            .take(*actual_n)
            .map(|x| math.abs(x))
            .enumerate()
            .filter(|(_, x)| !math.is_zero(*x))
            .map(|(i, _)| n.idx(i)),
    );
    let perm = &*perm;
    with_dim!(o, perm.len());

    let (mut col0_perm, stack) = unsafe { temp_mat_uninit(ctx, o, 1, stack) };
    let (mut diag_perm, stack) = unsafe { temp_mat_uninit(ctx, o, 1, stack) };

    let mut col0_perm = col0_perm
        .as_mat_mut()
        .col_mut(0)
        .try_as_col_major_mut()
        .unwrap();
    let mut diag_perm = diag_perm
        .as_mat_mut()
        .col_mut(0)
        .try_as_col_major_mut()
        .unwrap();

    for (k, &p) in perm.iter().enumerate() {
        let k = o.idx(k);
        write1!(col0_perm[k] = math(copy(col0[p])));
        write1!(diag_perm[k] = math(copy(diag[p])));
    }

    let (mut shifts, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
    let shifts = shifts.as_mat_mut();
    let (mut mus, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
    let mus = mus.as_mat_mut();
    let (mut singular_vals, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
    let singular_vals = singular_vals.as_mat_mut();
    let (mut zhat, stack) = unsafe { temp_mat_uninit(ctx, n, 1, stack) };
    let zhat = zhat.as_mat_mut();

    let mut shifts = shifts.col_mut(0).try_as_col_major_mut().unwrap();
    let mut mus = mus.col_mut(0).try_as_col_major_mut().unwrap();
    let mut s = singular_vals.col_mut(0).try_as_col_major_mut().unwrap();
    let mut zhat = zhat.col_mut(0).try_as_col_major_mut().unwrap();

    compute_singular_values(
        ctx,
        shifts.rb_mut(),
        mus.rb_mut(),
        s.rb_mut(),
        diag.rb(),
        diag_perm.rb(),
        col0,
        col0_perm.rb(),
    );

    perturb_col0(
        ctx,
        zhat.rb_mut(),
        col0,
        diag.rb(),
        perm,
        s.rb(),
        shifts.rb(),
        mus.rb(),
    );

    let (mut col_perm, stack) = stack.make_with(*actual_n, |i| i);
    let (mut col_perm_inv, _) = stack.make_with(*actual_n, |i| i);

    for i0 in zero().to(n.idx_inc(*actual_n - 1)) {
        let i1 = n.idx(*i0 + 1);
        math(if s[i0] > s[i1] {
            let si = copy(s[i0]);
            let sj = copy(s[i1]);
            write1!(s.rb_mut().at_mut(i0), sj);
            write1!(s.rb_mut().at_mut(i1), si);

            col_perm.swap(*i0, *i1);
        })
    }
    for (i, p) in col_perm.iter().copied().enumerate() {
        col_perm_inv[p] = i;
    }

    compute_singular_vectors(
        ctx,
        um.rb_mut(),
        vm.rb_mut(),
        zhat.rb(),
        diag.rb(),
        perm,
        outer_perm,
        &col_perm_inv,
        *actual_n,
        shifts.rb(),
        mus.rb(),
    );

    for (idx, mut diag) in diag
        .rb_mut()
        .subrows_range_mut((zero(), actual_n))
        .iter_mut()
        .enumerate()
    {
        write1!(diag, math(copy(s[n.idx(*actual_n - idx - 1)])));
    }

    for (idx, mut diag) in diag
        .rb_mut()
        .subrows_range_mut((actual_n, n.end()))
        .iter_mut()
        .enumerate()
    {
        write1!(diag, math(copy(s[n.idx(*actual_n + idx)])));
    }
}

#[math]
#[inline(never)]
fn compute_singular_vectors<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    mut um: Option<MatMut<C, T, usize, usize, ContiguousFwd>>,
    mut vm: Option<MatMut<C, T, Dim<'N>, Dim<'N>, ContiguousFwd>>,
    zhat: ColRef<C, T, Dim<'N>, ContiguousFwd>,
    diag: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    perm: &[Idx<'N>],
    outer_perm: &Array<'N, Idx<'N>>,
    col_perm_inv: &[usize],
    actual_n: usize,
    shifts: ColRef<C, T, Dim<'N>, ContiguousFwd>,
    mus: ColRef<C, T, Dim<'N>, ContiguousFwd>,
) {
    let n = diag.nrows();

    help!(C);
    for k in n.indices() {
        let actual_k = if *k >= actual_n {
            k
        } else {
            n.idx(actual_n - col_perm_inv[*k] - 1)
        };

        if let Some(mut u) = um.rb_mut() {
            write1!(u[(*n, *actual_k)] = math.zero());
        }

        let mut u = um
            .rb_mut()
            .map(|u| u.submatrix_mut(0, 0, n, n).col_mut(actual_k));
        let mut v = vm
            .rb_mut()
            .map(|v| v.submatrix_mut(zero(), zero(), n, n).col_mut(actual_k));

        if math(is_zero(zhat[k])) {
            if let Some(mut u) = u.rb_mut() {
                write1!(u[outer_perm[k]] = math.one());
            }
            if let Some(mut v) = v.rb_mut() {
                write1!(v[outer_perm[k]] = math.one());
            }
            continue;
        }

        let mu = math(mus[k]);
        let shift = math(shifts[k]);

        if let Some(mut u) = u.rb_mut() {
            for &i in perm {
                write1!(
                    u[outer_perm[i]] =
                        math((zhat[i] / ((diag[i] - shift) - mu)) / (diag[i] + (shift + mu)))
                );
            }
            let norm_inv = math.recip(u.norm_l2_with(ctx));
            z!(u.rb_mut()).for_each(|uz!(mut x)| write1!(x, math(x * norm_inv)));
        }

        if let Some(mut v) = v {
            write1!(v[outer_perm[n.idx(0)]] = math(-one()));
            for &i in &perm[1..] {
                write1!(
                    v[outer_perm[i]] = math(
                        ((diag[i] * zhat[i]) / ((diag[i] - shift) - mu)) / (diag[i] + (shift + mu))
                    )
                );
            }
            let norm_inv = math.recip(v.norm_l2_with(ctx));
            z!(v.rb_mut()).for_each(|uz!(mut x)| write1!(x, math(x * norm_inv)));
        }
    }
    if let Some(mut u) = um.rb_mut() {
        write1!(u[(*n, *n)] = math.one());
    }
}

#[math]
fn perturb_col0<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    zhat: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    col0: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    diag: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    perm: &[Idx<'N>],
    s: ColRef<C, T, Dim<'N>, ContiguousFwd>,
    shifts: ColRef<C, T, Dim<'N>, ContiguousFwd>,
    mus: ColRef<C, T, Dim<'N>, ContiguousFwd>,
) {
    let mut zhat = zhat;

    help!(C);
    let n = diag.nrows();
    let m = perm.len();
    if m == 0 {
        z!(zhat).for_each(|uz!(mut x)| write1!(x, math(zero())));
        return;
    }

    let last_idx = perm[m - 1];
    for k in n.indices() {
        if math(is_zero(col0[k])) {
            write1!(zhat[k] = math(zero()));
            continue;
        }

        let dk = math(diag[k]);
        // NOTE: the order of operations is crucial here
        let mut prod = math((s[last_idx] + dk) * (mus[last_idx] + (shifts[last_idx] - dk)));

        for (l, &i) in perm.iter().enumerate() {
            if i == k {
                continue;
            }
            if i >= k && l == 0 {
                prod = math.zero();
                break;
            }

            let j = if i < k {
                i
            } else if l > 0 {
                perm[l - 1]
            } else {
                i
            };

            let term = math(
                ((s[j] + dk) / (diag[i] + dk)) * ((mus[j] + (shifts[j] - dk)) / (diag[i] - dk)),
            );
            prod = math(prod * term);
        }

        let tmp = math.sqrt(prod);

        if math(gt_zero(col0[k])) {
            write1!(zhat[k] = tmp);
        } else {
            write1!(zhat[k] = math(-tmp));
        }
    }
}

#[math]
fn compute_singular_values<'N, 'O, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    mut shifts: ColMut<C, T, Dim<'N>, ContiguousFwd>,
    mut mus: ColMut<C, T, Dim<'N>, ContiguousFwd>,
    mut s: ColMut<C, T, Dim<'N>, ContiguousFwd>,
    diag: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    diag_perm: ColRef<'_, C, T, Dim<'O>, ContiguousFwd>,
    col0: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    col0_perm: ColRef<'_, C, T, Dim<'O>, ContiguousFwd>,
) {
    let n = diag.nrows();
    let mut actual_n = *n;
    while actual_n > 1 && math(is_zero(col0[n.idx(actual_n - 1)])) {
        actual_n -= 1;
    }
    let actual_n = actual_n;

    let two = math(from_f64(2.0));
    let eight = math(from_f64(8.0));
    let one_half = math(from_f64(0.5));

    help!(C);
    let first = n.idx(0);
    let epsilon = math.eps();

    'kth_iter: for k in n.indices() {
        write1!(s[k] = math(zero()));
        write1!(shifts[k] = math(zero()));
        write1!(mus[k] = math(zero()));

        if math(is_zero(col0[k])) || actual_n == 1 {
            write1!(
                s[k] = if *k == 0 {
                    math(copy(col0[first]))
                } else {
                    math(copy(diag[k]))
                }
            );
            write1!(shifts[k] = math(copy(s[k])));
            write1!(mus[k] = math(zero()));

            continue 'kth_iter;
        }

        let last_k = *k == actual_n - 1;
        let left = math(diag[k]);
        let right = if last_k {
            math(diag[n.idx(actual_n - 1)] + (col0.norm_l2_with(ctx)))
        } else {
            let mut l = n.idx(*k + 1);
            while math(is_zero(col0[l])) {
                l = n.idx(*l + 1);
            }
            math(copy(diag[l]))
        };

        let mid = math(left + mul_pow2((right - left), one_half));
        let [mut f_mid, f_max, f_mid_left_shift, f_mid_right_shift] = secular_eq_multi_fast(
            ctx,
            math(&[
                mid,
                if last_k {
                    right - left
                } else {
                    mul_pow2(right - left, one_half)
                },
                mul_pow2(right - left, one_half),
                mul_pow2(left - right, one_half),
            ]),
            col0_perm,
            diag_perm,
            math([zero(), copy(left), copy(left), copy(right)]),
        );

        let (mut shift, mu) = if last_k || math(gt_zero(f_mid)) {
            math((copy(left), mul_pow2(right - left, one_half)))
        } else {
            math((copy(right), mul_pow2(left - right, one_half)))
        };

        if math(le_zero(f_mid_left_shift) && gt_zero(f_mid_right_shift)) {
            write1!(s[k] = math(shift + mu));
            write1!(shifts[k] = shift);
            write1!(mus[k] = mu);

            continue 'kth_iter;
        }

        if !last_k {
            if math(shift == left) {
                if math(lt_zero(f_mid_left_shift)) {
                    shift = math.copy(right);
                    f_mid = f_mid_right_shift;
                }
            } else if math(gt_zero(f_mid_right_shift)) {
                shift = math.copy(left);
                f_mid = f_mid_left_shift;
            }
        }

        enum SecantError {
            OutOfBounds,
            PrecisionLimitReached,
        }

        let secant = {
            |mut mu_cur: C::Of<T>,
             mut mu_prev: C::Of<T>,
             mut f_cur: C::Of<T>,
             mut f_prev: C::Of<T>| {
                if math(abs(f_prev) < abs(f_cur)) {
                    swap(&mut f_prev, &mut f_cur);
                    swap(&mut mu_prev, &mut mu_cur);
                }

                let mut left_candidate = None;
                let mut right_candidate = None;

                let mut use_bisection = false;
                let same_sign = math.gt_zero(f_prev) == math.gt_zero(f_cur);
                if !same_sign {
                    let (min, max) = if math(mu_cur < mu_prev) {
                        math((copy(mu_cur), copy(mu_prev)))
                    } else {
                        math((copy(mu_prev), copy(mu_cur)))
                    };
                    left_candidate = Some(min);
                    right_candidate = Some(max);
                }

                let mut err = SecantError::PrecisionLimitReached;

                while !math.is_zero(f_cur)
                    && math(
                        abs(mu_cur - mu_prev) > eight * epsilon * max(abs(mu_cur), abs(mu_prev)),
                    )
                    && math(abs(f_cur - f_prev) > epsilon)
                    && !use_bisection
                {
                    // rational interpolation: fit a function of the form a / mu + b through
                    // the two previous iterates and use its
                    // zero to compute the next iterate
                    let a = math((f_cur - f_prev) * (mu_prev * mu_cur) / (mu_prev - mu_cur));
                    let b = math(f_cur - a / mu_cur);
                    let mu_zero = math(-a / b);
                    let f_zero = secular_eq(
                        ctx,
                        math.copy(mu_zero),
                        col0_perm,
                        diag_perm,
                        math.copy(shift),
                    );

                    if math.lt_zero(f_zero) {
                        left_candidate = Some(math.copy(mu_zero));
                    } else {
                        right_candidate = Some(math.copy(mu_zero));
                    }

                    mu_prev = math.copy(mu_cur);
                    f_prev = math.copy(f_cur);
                    mu_cur = math.copy(mu_zero);
                    f_cur = math.copy(f_zero);

                    if math(shift == left && (lt_zero(mu_cur) || mu_cur > right - left)) {
                        err = SecantError::OutOfBounds;
                        use_bisection = true;
                    }
                    if math(shift == right && (gt_zero(mu_cur) || mu_cur < left - right)) {
                        err = SecantError::OutOfBounds;
                        use_bisection = true;
                    }
                    if math(abs(f_cur) > abs(f_prev)) {
                        // find mu such that a / mu + b = -k * f_zero
                        // a / mu = -f_zero - b
                        // mu = -a / (f_zero + b)
                        let mut k = math(one());
                        for _ in 0..4 {
                            let mu_opposite = math(-a / (k * f_zero + b));
                            let f_opposite = secular_eq(
                                ctx,
                                math.copy(mu_opposite),
                                col0_perm,
                                diag_perm,
                                math.copy(shift),
                            );
                            if math(lt_zero(f_zero) && ge_zero(f_opposite)) {
                                // this will be our right candidate
                                right_candidate = Some(mu_opposite);
                                break;
                            }
                            if math(gt_zero(f_zero) && le_zero(f_opposite)) {
                                // this will be our left candidate
                                left_candidate = Some(mu_opposite);
                                break;
                            }
                            k = math.mul_pow2(k, two);
                        }
                        use_bisection = true;
                    }
                }
                (use_bisection, mu_cur, left_candidate, right_candidate, err)
            }
        };

        let (mut left_shifted, mut f_left, mut right_shifted, mut f_right) = if math(shift == left)
        {
            math((
                zero(),
                -infinity(),
                if last_k {
                    right - left
                } else {
                    (right - left) * one_half
                },
                copy(*if last_k { &f_max } else { &f_mid }),
            ))
        } else {
            math(((left - right) * one_half, copy(f_mid), zero(), infinity()))
        };

        let mut iteration_count = 0;
        let mut f_prev = math.copy(f_mid);
        // try to find non zero starting bounds

        let half0 = math.copy(one_half);
        let half1 = math(mul_pow2(half0, half0));
        let half2 = math(mul_pow2(half1, half1));
        let half3 = math(mul_pow2(half2, half2));
        let half4 = math(mul_pow2(half3, half3));
        let half5 = math(mul_pow2(half4, half4));
        let half6 = math(mul_pow2(half5, half5));
        let half7 = math(mul_pow2(half6, half6));

        let mu_values = math(if shift == left {
            [
                mul_pow2(right_shifted, half7),
                mul_pow2(right_shifted, half6),
                mul_pow2(right_shifted, half5),
                mul_pow2(right_shifted, half4),
                mul_pow2(right_shifted, half3),
                mul_pow2(right_shifted, half2),
                mul_pow2(right_shifted, half1),
                mul_pow2(right_shifted, half0),
            ]
        } else {
            [
                mul_pow2(left_shifted, half7),
                mul_pow2(left_shifted, half6),
                mul_pow2(left_shifted, half5),
                mul_pow2(left_shifted, half4),
                mul_pow2(left_shifted, half3),
                mul_pow2(left_shifted, half2),
                mul_pow2(left_shifted, half1),
                mul_pow2(left_shifted, half0),
            ]
        });

        let f_values = secular_eq_multi_fast(
            ctx,
            &mu_values,
            col0_perm,
            diag_perm,
            [(); 8].map(|_| math.copy(shift)),
        );

        if math(shift == left) {
            let mut i = 0;
            for (idx, (mu, f)) in core::iter::zip(&mu_values, &f_values).enumerate() {
                if math.lt_zero(*f) {
                    left_shifted = math.copy(*mu);
                    f_left = math.copy(*f);
                    i = idx + 1;
                }
            }
            if i < f_values.len() {
                right_shifted = math.copy(mu_values[i]);
                f_right = math.copy(f_values[i]);
            }
        } else {
            let mut i = 0;
            for (idx, (mu, f)) in core::iter::zip(&mu_values, &f_values).enumerate() {
                if math.gt_zero(f) {
                    right_shifted = math.copy(*mu);
                    f_right = math.copy(*f);
                    i = idx + 1;
                }
            }
            if i < f_values.len() {
                left_shifted = math.copy(mu_values[i]);
                f_left = math.copy(f_values[i]);
            }
        }

        // try bisection just to get a good guess for secant
        while math(
            right_shifted - left_shifted
                > two * epsilon * (max(abs(left_shifted), abs(right_shifted))),
        ) {
            let mid_shifted_arithmetic = math((left_shifted + right_shifted) * one_half);
            let mut mid_shifted_geometric =
                math(sqrt(abs(left_shifted)) * sqrt(abs(right_shifted)));
            if math.lt_zero(left_shifted) {
                mid_shifted_geometric = math(-mid_shifted_geometric);
            }
            let mid_shifted = if math.is_zero(mid_shifted_geometric) {
                mid_shifted_arithmetic
            } else {
                mid_shifted_geometric
            };
            let f_mid = secular_eq(
                ctx,
                math.copy(mid_shifted),
                col0_perm,
                diag_perm,
                math.copy(shift),
            );

            if math.is_zero(f_mid) {
                write1!(s[k] = math(shift + mid_shifted));
                write1!(shifts[k] = shift);
                write1!(mus[k] = mid_shifted);
                continue 'kth_iter;
            } else if math.gt_zero(f_mid) {
                right_shifted = mid_shifted;
                f_prev = f_right;
                f_right = f_mid;
            } else {
                left_shifted = mid_shifted;
                f_prev = f_left;
                f_left = f_mid;
            }

            if iteration_count == 4 {
                break;
            }

            iteration_count += 1;
        }

        // try secant with the guess from bisection
        let args = if math.is_zero(left_shifted) {
            (
                math(mul_pow2(right_shifted, two)),
                math.copy(right_shifted),
                f_prev,
                f_right,
            )
        } else if math.is_zero(right_shifted) {
            (
                math(mul_pow2(left_shifted, two)),
                math.copy(left_shifted),
                f_prev,
                f_left,
            )
        } else {
            (
                math.copy(left_shifted),
                math.copy(right_shifted),
                f_left,
                f_right,
            )
        };

        let (use_bisection, mut mu_cur, left_candidate, right_candidate, _err) =
            secant(args.0, args.1, args.2, args.3);

        match (left_candidate, right_candidate) {
            (Some(left), Some(right)) if math(left < right) => math({
                if left > left_shifted {
                    left_shifted = left;
                }
                if right < right_shifted {
                    right_shifted = right;
                }
            }),
            _ => (),
        }

        // secant failed, use bisection again
        if use_bisection {
            while math(
                right_shifted - left_shifted
                    > two * epsilon * (max(abs(left_shifted), abs(right_shifted))),
            ) {
                let mid_shifted = math((left_shifted + right_shifted) * one_half);
                let f_mid = secular_eq(
                    ctx,
                    math.copy(mid_shifted),
                    col0_perm,
                    diag_perm,
                    math.copy(shift),
                );

                if math.is_zero(f_mid) {
                    break;
                } else if math.gt_zero(f_mid) {
                    right_shifted = mid_shifted;
                } else {
                    left_shifted = mid_shifted;
                }
            }

            mu_cur = math((left_shifted + right_shifted) * one_half);
        }

        write1!(s[k] = math(shift + mu_cur));
        write1!(shifts[k] = shift);
        write1!(mus[k] = mu_cur);
    }
}

#[math]
fn secular_eq<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    mu: C::Of<T>,
    col0_perm: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    diag_perm: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    shift: C::Of<T>,
) -> C::Of<T> {
    let mut res = math(one());

    let n = diag_perm.nrows();
    for i in n.indices() {
        let c = math(col0_perm[i]);
        let d = math(diag_perm[i]);
        res = math(res + (c / ((d - shift) - mu)) * (c / ((d + shift) + mu)));
    }

    res
}

#[math]
fn secular_eq_multi_fast<'N, const N: usize, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    mu: &[C::Of<T>; N],
    col0_perm: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    diag_perm: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    shift: [C::Of<T>; N],
) -> [C::Of<T>; N] {
    let n = col0_perm.nrows();
    let mut res0 = [(); N].map(|_| math(one()));
    for i in n.indices() {
        let c = math(col0_perm[i]);
        let d = math(diag_perm[i]);

        for ((res0, mu), shift) in res0.iter_mut().zip(mu.iter()).zip(shift.iter()) {
            *res0 = math((*res0) + (abs2(c) / (((d - *shift) - *mu) * ((d + *shift) + *mu))));
        }
    }
    res0
}

#[math]
fn deflate<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    diag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    col0: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    jacobi_coeff: &mut [JacobiRotation<C, T>],
    jacobi_idx: &mut [Idx<'N>],
    transpositions: &mut [Idx<'N>],
    perm: &mut Array<'N, Idx<'N>>,
    k: usize,
    stack: &mut DynStack,
) -> (usize, usize) {
    help!(C);
    let mut diag = diag;
    let mut col0 = col0;

    let n = diag.nrows();
    let first = n.idx(0);

    let mut jacobi_0i = 0;
    let mut jacobi_ij = 0;

    let (max_diag, max_col0) = (
        diag.rb()
            .subrows_range((first.next(), n.end()))
            .norm_max_with(ctx),
        col0.norm_max_with(ctx),
    );
    let max = math(max(max_diag, max_col0));

    let eps = math(eps());
    let sml = math(min_positive());

    let eps_strict = math(max(eps * max_diag, sml));

    let eps_coarse = math(from_f64(8.0) * eps * max);

    // condition 4.1
    if math(diag[first] < eps_coarse) {
        write1!(diag[first] = math.copy(eps_coarse));
        write1!(col0[first] = math.copy(eps_coarse));
    }

    // condition 4.2
    for i in first.next().to(n.end()) {
        if math(abs(col0[i]) < eps_strict) {
            write1!(col0[i] = math(zero()));
        }
    }

    // condition 4.3
    for i in first.next().to(n.end()) {
        if math(diag[i] < eps_coarse) {
            if let Some(rot) = deflation_43(ctx, diag.rb_mut(), col0.rb_mut(), i) {
                jacobi_coeff[jacobi_0i] = rot;
                jacobi_idx[jacobi_0i] = i;
                jacobi_0i += 1;
            }
        }
    }

    let mut total_deflation = true;
    for i in first.next().to(n.end()) {
        if math(!(abs(col0[i]) < sml)) {
            total_deflation = false;
            break;
        }
    }

    perm[first] = first;
    let mut p = 1;
    for i in first.next().to(n.end()) {
        if math(abs(diag[i]) < sml) {
            perm[n.idx(p)] = i;
            p += 1;
        }
    }

    let mut i = 1;
    let mut j = k + 1;

    for p in &mut perm.as_mut()[p..] {
        if i >= k + 1 {
            *p = n.idx(j);
            j += 1;
        } else if j >= *n {
            *p = n.idx(i);
            i += 1;
        } else if math(diag[n.idx(i)] < diag[n.idx(j)]) {
            *p = n.idx(j);
            j += 1;
        } else {
            *p = n.idx(i);
            i += 1;
        }
    }

    if total_deflation {
        for i in first.next().to(n.end()) {
            let i1 = n.idx(*i - 1);

            let pi = perm[i];
            if math(abs(diag[pi]) < sml || diag[pi] > diag[first]) {
                perm[i1] = perm[i];
            } else {
                perm[i1] = first;
                break;
            }
        }
    }

    let (mut real_ind, stack) = stack.collect(n.indices());
    let (mut real_col, _) = stack.collect(n.indices());

    for i in (if total_deflation {
        zero()
    } else {
        first.next()
    })
    .to(n.end())
    {
        let pi = perm[n.idx(*n - if total_deflation { *i + 1 } else { *i })];
        let j = real_col[*pi];

        let (a, b) = math((copy(diag[i]), copy(diag[j])));
        write1!(diag[i] = b);
        write1!(diag[j] = a);

        if *i != 0 && *j != 0 {
            let (a, b) = math((copy(col0[i]), copy(col0[j])));
            write1!(col0[i] = b);
            write1!(col0[j] = a);
        }

        transpositions[*i] = j;
        let real_i = real_ind[*i];
        real_col[*real_i] = j;
        real_col[*pi] = i;
        real_ind[*j] = real_i;
        real_ind[*i] = pi;
    }

    write1!(col0[first] = math(copy(diag[first])));

    for i in n.indices() {
        perm[i] = i;
    }

    for (i, &j) in transpositions.iter().enumerate() {
        perm.as_mut().swap(i, *j);
    }

    // condition 4.4
    let mut i = n.idx(*n - 1);
    while *i > 0 && math((abs(diag[i]) < sml || abs(col0[i]) < sml)) {
        i = n.idx(*i - 1);
    }

    while *i > 1 {
        let i1 = n.idx(*i - 1);
        if math(diag[i] - diag[i1] < eps_strict) {
            if let Some(rot) = deflation_44(ctx, diag.rb_mut(), col0.rb_mut(), i1, i) {
                jacobi_coeff[jacobi_0i + jacobi_ij] = rot;
                jacobi_idx[jacobi_0i + jacobi_ij] = i;
                jacobi_ij += 1;
            }
        }

        i = i1;
    }

    (jacobi_0i, jacobi_ij)
}

#[math]
fn deflation_43<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    diag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    col0: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    i: Idx<'N>,
) -> Option<JacobiRotation<C, T>> {
    let mut diag = diag;
    let mut col0 = col0;

    let n = diag.nrows();
    let first = n.idx(0);
    help!(C);

    let p = math(copy(col0[first]));
    let q = math(copy(col0[i]));

    if math(is_zero(p) && is_zero(q)) {
        write1!(diag[i] = math(zero()));
        return None;
    }

    let rot = JacobiRotation::make_givens(ctx, math.copy(p), math.copy(q));

    let r = math(rot.c * p - rot.s * q);
    write1!(col0[first] = math.copy(r));
    write1!(diag[first] = math.copy(r));
    write1!(col0[i] = math.zero());
    write1!(diag[i] = math.zero());

    Some(rot)
}

#[math]
fn deflation_44<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    diag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    col0: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    i: Idx<'N>,
    j: Idx<'N>,
) -> Option<JacobiRotation<C, T>> {
    let mut diag = diag;
    let mut col0 = col0;

    help!(C);

    let p = math(copy(col0[i]));
    let q = math(copy(col0[j]));

    if math(is_zero(p) && is_zero(q)) {
        write1!(diag[i] = math(copy(diag[j])));
        return None;
    }

    let rot = JacobiRotation::make_givens(ctx, math.copy(p), math.copy(q));

    let r = math(rot.c * p - rot.s * q);
    write1!(col0[i] = math.copy(r));
    write1!(col0[j] = math.zero());
    write1!(diag[i] = math(copy(diag[j])));

    Some(rot)
}

#[derive(Debug)]
enum MatU<'a, C: RealContainer, T: RealField<C>> {
    Full(MatMut<'a, C, T>),
    TwoRows(MatMut<'a, C, T>),
}

impl<'short, C: RealContainer, T: RealField<C>> ReborrowMut<'short> for MatU<'_, C, T> {
    type Target = MatU<'short, C, T>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        match self {
            Self::Full(u) => MatU::Full(u.rb_mut()),
            Self::TwoRows(u) => MatU::TwoRows(u.rb_mut()),
        }
    }
}

fn divide_and_conquer_scratch<C: ComplexContainer, T: ComplexField<C>>(
    n: usize,
    qr_fallback_threshold: usize,
    compute_u: bool,
    compute_v: bool,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    if n <= qr_fallback_threshold {
        temp_mat_scratch::<C, T>(n + 1, n + 1)?.try_array(2)
    } else {
        let _ = par;
        let perm = StackReq::try_new::<usize>(n)?;
        let jacobi_coeffs = StackReq::try_new::<JacobiRotation<C, T>>(n)?;
        let jacobi_indices = perm;
        let transpositions = perm;
        let real_ind = perm;
        let real_col = perm;

        let um = temp_mat_scratch::<C, T>(n + 1, n + 1)?;
        let vm = temp_mat_scratch::<C, T>(n, if compute_v { n } else { 0 })?;

        let combined_u = temp_mat_scratch::<C, T>(if compute_u { n + 1 } else { 2 }, n + 1)?;
        let combined_v = vm;

        let prologue = StackReq::try_all_of([perm, jacobi_coeffs, jacobi_indices])?;

        StackReq::try_all_of([
            prologue,
            um,
            vm,
            combined_u,
            combined_v,
            transpositions,
            real_ind,
            real_col,
        ])
    }
}

#[math]
fn divide_and_conquer<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    diag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    subdiag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    u: MatU<'_, C, T>,
    v: Option<MatMut<'_, C, T, Dim<'N>, Dim<'N>>>,
    par: Par,
    stack: &mut DynStack,
    qr_fallback_threshold: usize,
) -> Result<(), SvdError> {
    help!(C);

    let qr_fallback_threshold = Ord::max(qr_fallback_threshold, 4);

    let n = diag.nrows();

    let mut diag = diag;
    let mut subdiag = subdiag;
    let mut u = u;
    let mut v = v;

    if *n < qr_fallback_threshold {
        with_dim!(n1, *n + 1);

        let (mut diag_alloc, stack) = unsafe { temp_mat_uninit(ctx, n1, 1, stack) };
        let (mut subdiag_alloc, stack) = unsafe { temp_mat_uninit(ctx, n1, 1, stack) };

        let mut diag_alloc = diag_alloc
            .as_mat_mut()
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();
        let mut subdiag_alloc = subdiag_alloc
            .as_mat_mut()
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();

        for i in n.indices() {
            let i1 = n1.idx(*i);

            write1!(diag_alloc[i1] = math(copy(diag[i])));
            write1!(subdiag_alloc[i1] = math(copy(subdiag[i])));
        }
        let i1 = n1.idx(*n);
        write1!(diag_alloc[i1] = math(zero()));
        write1!(subdiag_alloc[i1] = math(zero()));

        let (mut u_alloc, stack) = temp_mat_zeroed(ctx, n1, n1, stack);
        let (mut v_alloc, _) = temp_mat_zeroed(ctx, n1, n1, stack);
        let mut u_alloc = u_alloc.as_mat_mut();
        let mut v_alloc = v_alloc.as_mat_mut();
        for i in n1.indices() {
            write1!(u_alloc[(i, i)] = math.one());
            write1!(v_alloc[(i, i)] = math.one());
        }

        qr_algorithm(
            ctx,
            diag_alloc.rb_mut(),
            subdiag_alloc.rb_mut(),
            Some(u_alloc.rb_mut()),
            Some(v_alloc.rb_mut()),
        )?;

        if math(lt_zero(v_alloc[(i1, i1)])) {
            for i in n1.indices() {
                write1!(u_alloc[(i, i1)] = math(-u_alloc[(i, i1)]));
            }
            for i in n1.indices() {
                write1!(v_alloc[(i, i1)] = math(-v_alloc[(i, i1)]));
            }
        }

        if math(!(v_alloc[(i1, i1)] == one())) {
            panic!("svd bottom right corner should be one");
        }

        for i in n.indices() {
            let i1 = n1.idx(*i);
            write1!(diag[i] = math(copy(diag_alloc[i1])));
            write1!(subdiag[i] = math(copy(subdiag_alloc[i1])));
        }

        if let Some(mut v) = v.rb_mut() {
            v.copy_from_with(ctx, v_alloc.rb().submatrix(zero(), zero(), n, n));
        }
        match u.rb_mut() {
            MatU::Full(u) => u
                .submatrix_mut(0, 0, n1, n1)
                .copy_from_with(ctx, u_alloc.rb()),
            MatU::TwoRows(u) => {
                let (mut top, mut bot) = u.subcols_mut(0, n1).two_rows_mut(0, 1);

                let first = n1.idx(0);
                let last = n1.idx(*n);

                top.copy_from_with(ctx, u_alloc.rb().row(first));
                bot.copy_from_with(ctx, u_alloc.rb().row(last));
            }
        }

        return Ok(());
    }

    let max = math.max(diag.norm_max_with(ctx), subdiag.norm_max_with(ctx));

    if math(is_zero(max)) {
        return Ok(());
    }

    let max_inv = math(recip(max));

    for i in n.indices() {
        write1!(diag[i] = math(diag[i] * max_inv));
        write1!(subdiag[i] = math(subdiag[i] * max_inv));
    }

    let k = n.idx(*n / 2);
    let rem = n.idx(*n - 1 - *k);

    let (alpha, beta) = ghost_tree!(N0(HEAD, K, TAIL), {
        let (split @ l![head, _, tail], (split_x, ..)) = n.split(l![..k.into(), k, ..], N0);

        let l![mut d1, alpha, mut d2] = diag.rb_mut().row_segments_mut(split, split_x);
        let l![mut subd1, beta, mut subd2] = subdiag.rb_mut().row_segments_mut(split, split_x);

        let (mut u1, mut u2) = match u.rb_mut() {
            MatU::Full(u) => {
                let (u1, u2) = u.split_at_row_mut(*k + 1);

                (
                    MatU::Full(u1.submatrix_mut(0, 1, *k + 1, *k + 1)),
                    MatU::Full(u2.submatrix_mut(0, *k + 1, *rem + 1, *rem + 1)),
                )
            }
            MatU::TwoRows(u) => {
                let (u1, u2) = u.split_at_col_mut(*k + 1);
                (MatU::TwoRows(u1), MatU::TwoRows(u2))
            }
        };

        let (mut v1, mut v2) = match v.rb_mut() {
            Some(v) => {
                let l![v1, _, v2] = v.row_segments_mut(split, split_x);
                (
                    Some(v1.subcols_mut(n.idx_inc(1), head.len())),
                    Some(v2.subcols_mut(n.idx_inc(*k + 1), tail.len())),
                )
            }
            None => (None, None),
        };

        let stack_bytes = stack.len_bytes();
        let (mut stack1, stack2) = stack.make_uninit::<u8>(stack_bytes / 2);
        let stack1 = DynStack::new(&mut stack1);

        let mut r1 = Ok(());
        let mut r2 = Ok(());

        crate::utils::thread::join_raw(
            |par| {
                r1 = divide_and_conquer(
                    ctx,
                    d1.rb_mut(),
                    subd1.rb_mut(),
                    u1.rb_mut(),
                    v1.rb_mut(),
                    par,
                    stack1,
                    qr_fallback_threshold,
                );
            },
            |par| {
                r2 = divide_and_conquer(
                    ctx,
                    d2.rb_mut(),
                    subd2.rb_mut(),
                    u2.rb_mut(),
                    v2.rb_mut(),
                    par,
                    stack2,
                    qr_fallback_threshold,
                );
            },
            par,
        );
        r1?;
        r2?;

        match u1.rb_mut() {
            MatU::TwoRows(mut u1) => {
                // rotation of Q1, q1
                for i in (0..*k).rev() {
                    with_dim!(ncols, u1.ncols());

                    crate::perm::swap_cols_idx(
                        ctx,
                        u1.rb_mut().as_shape_mut(ncols, ncols),
                        ncols.idx(i),
                        ncols.idx(i + 1),
                    );
                }
            }
            _ => {}
        }

        (math(copy(alpha)), math(copy(beta)))
    });
    if let Some(mut v) = v.rb_mut() {
        write1!(v[(k, n.idx(0))] = math.one());
    }

    for i in zero().to(k.excl()).rev() {
        let i1 = n.idx(*i + 1);
        write1!(diag[i1] = math(copy(diag[i])));
    }

    let lambda = match u.rb_mut() {
        MatU::Full(u) => math(copy(u[(*k, *k + 1)])),
        MatU::TwoRows(u) => math(copy(u[(1, 0)])),
    };
    let phi = match u.rb_mut() {
        MatU::Full(u) => math(copy(u[(*k + 1, *n)])),
        MatU::TwoRows(u) => math(copy(u[(0, *n)])),
    };

    let al = math(alpha * lambda);
    let bp = math(beta * phi);
    let r0 = math(sqrt(abs2(al) + abs2(bp)));
    let (c0, s0) = math(if is_zero(r0) {
        (one(), zero())
    } else {
        (al / r0, bp / r0)
    });

    let mut col0 = subdiag;

    write1!(diag[n.idx(0)] = math.copy(r0));
    write1!(col0[n.idx(0)] = math.copy(r0));

    match u.rb_mut() {
        MatU::Full(u) => {
            let (u1, u2) = u.split_at_row_mut(*k + 1);

            let (mut u0_top, u1) = u1.split_at_col_mut(1);
            let (u1, mut un_top) = u1.split_at_col_mut(*n - 1);
            let (mut u0_bot, u2) = u2.split_at_col_mut(1);
            let (u2, mut un_bot) = u2.split_at_col_mut(*n - 1);

            for j in n.idx_inc(1).to(k.next()) {
                write1!(col0[j] = math(alpha * u1[(*k, *j - 1)]));
            }

            for j in k.next().to(n.end()) {
                write1!(col0[j] = math(beta * u2[(0, *j - 1)]));
            }

            z!(
                u0_top.rb_mut().col_mut(0),
                un_top.rb_mut().col_mut(0),
                u1.col_mut(*k)
            )
            .for_each(|uz!(mut x0, mut xn, mut a)| {
                write1!(x0, math(c0 * a));
                write1!(xn, math(-s0 * a));
                if cfg!(debug_assertions) {
                    write1!(a, math.zero());
                }
            });

            z!(u0_bot.rb_mut().col_mut(0), un_bot.rb_mut().col_mut(0)).for_each(
                |uz!(mut x0, mut xn)| {
                    write1!(x0, math(s0 * xn));
                    write1!(xn, math(c0 * xn));
                },
            );
        }
        MatU::TwoRows(mut u) => {
            for j in n.idx_inc(1).to(k.next()) {
                write1!(col0[j] = math(alpha * u[(1, *j)]));
                write1!(u[(1, *j)] = math.zero());
            }

            for j in k.next().to(n.end()) {
                write1!(col0[j] = math(beta * u[(0, *j)]));
                write1!(u[(0, *j)] = math.zero());
            }

            let q10 = math(copy(u[(0, 0)]));
            let q21 = math(copy(u[(1, *n)]));

            write1!(u[(0, 0)] = math(c0 * q10));
            write1!(u[(0, *n)] = math(-s0 * q10));
            write1!(u[(1, 0)] = math(s0 * q21));
            write1!(u[(1, *n)] = math(c0 * q21));
        }
    }

    let (mut perm, stack) = stack.collect(n.indices());
    let perm = Array::from_mut(&mut *perm, n);

    let (mut jacobi_coeff, stack) = stack.make_with(*n, |_| JacobiRotation::<C, T> {
        c: ctx.zero(),
        s: ctx.zero(),
    });
    let jacobi_coeff = &mut *jacobi_coeff;
    let (mut jacobi_idx, stack) = stack.collect(n.indices());
    let jacobi_idx = &mut *jacobi_idx;

    let (jacobi_0i, jacobi_ij) = {
        let (mut transpositions, stack) = stack.collect(n.indices());
        deflate(
            ctx,
            diag.rb_mut(),
            col0.rb_mut(),
            jacobi_coeff,
            jacobi_idx,
            &mut transpositions,
            perm,
            *k,
            stack,
        )
    };

    let allocate_vm = v.is_some() as usize;
    let (mut um, stack) = temp_mat_zeroed(ctx, *n + 1, *n + 1, stack);
    let (mut vm, stack) = temp_mat_zeroed(ctx, n, allocate_vm * *n, stack);
    let mut um = um.as_mat_mut().try_as_col_major_mut().unwrap();
    let mut vm = if v.is_some() {
        Some(
            vm.as_mat_mut()
                .subcols_mut(0, n)
                .try_as_col_major_mut()
                .unwrap(),
        )
    } else {
        None
    };

    compute_svd_of_m(
        ctx,
        Some(um.rb_mut()),
        vm.rb_mut(),
        diag.rb_mut(),
        col0.rb(),
        perm,
        stack,
    );
    for i in n.indices() {
        write1!(col0[i] = math.zero());
    }

    for (rot, &i) in core::iter::zip(
        &jacobi_coeff[jacobi_0i..][..jacobi_ij],
        &jacobi_idx[jacobi_0i..][..jacobi_ij],
    )
    .rev()
    {
        let (i, j) = (n.idx(*i - 1), i);

        let actual_i = perm[i];
        let actual_j = perm[j];

        {
            let (i, j) = um
                .rb_mut()
                .subcols_mut(0, n)
                .two_rows_mut(*actual_i, *actual_j);
            rot.apply_on_the_left_in_place(ctx, i, j);
        }
        if let Some(mut vm) = vm.rb_mut() {
            let (i, j) = vm.rb_mut().two_rows_mut(actual_i, actual_j);
            rot.apply_on_the_left_in_place(ctx, i, j);
        }
    }

    for (rot, &i) in core::iter::zip(&jacobi_coeff[..jacobi_0i], &jacobi_idx[..jacobi_0i]).rev() {
        let (i, j) = um.rb_mut().subcols_mut(0, n).two_rows_mut(0, *i);
        rot.apply_on_the_left_in_place(ctx, i, j);
    }

    let _v_is_none = v.is_none();

    let update_v = |mut v: Option<MatMut<'_, C, T, Dim<'N>, Dim<'N>>>,
                    par: Par,
                    stack: &mut DynStack| {
        let (mut combined_v, _) = unsafe { temp_mat_uninit(ctx, n, allocate_vm * *n, stack) };
        let mut combined_v = if v.is_some() {
            Some(
                combined_v
                    .as_mat_mut()
                    .subcols_mut(0, n)
                    .try_as_col_major_mut()
                    .unwrap(),
            )
        } else {
            None
        };

        if let (Some(mut v), Some(v_rhs), Some(mut combined_v)) =
            (v.rb_mut(), vm.rb(), combined_v.rb_mut())
        {
            let mut combined_v = combined_v.rb_mut();
            let (mut combined_v1, combined_v2) = combined_v.rb_mut().split_at_row_mut(k.to_incl());
            let mut combined_v2 = combined_v2.submatrix_mut(1, zero(), *rem, n);

            let v_lhs = v.rb();
            let v_lhs1 = v_lhs.submatrix(zero(), n.idx_inc(1), *k, *k);
            let v_lhs2 = v_lhs.submatrix(k.next(), k.next(), *rem, *rem);
            let (v_rhs1, v_rhs2) = v_rhs.split_at_row(n.idx_inc(1)).1.split_at_row(*k);

            crate::utils::thread::join_raw(
                |par| {
                    crate::linalg::matmul::matmul(
                        ctx,
                        combined_v1.rb_mut(),
                        Accum::Replace,
                        v_lhs1,
                        v_rhs1,
                        math(one()),
                        par,
                    )
                },
                |par| {
                    crate::linalg::matmul::matmul(
                        ctx,
                        combined_v2.rb_mut(),
                        Accum::Replace,
                        v_lhs2,
                        v_rhs2,
                        math(one()),
                        par,
                    )
                },
                par,
            );

            crate::linalg::matmul::matmul(
                ctx,
                combined_v.rb_mut().submatrix_mut(k.to_incl(), zero(), 1, n),
                Accum::Replace,
                v_lhs.submatrix(k.to_incl(), zero(), 1, 1),
                v_rhs.submatrix(zero(), zero(), 1, n),
                math.one(),
                par,
            );

            v.copy_from_with(ctx, combined_v.rb());
        }
    };

    let update_u = |mut u: MatMut<'_, C, T>, par: Par, stack: &mut DynStack| {
        let n = *n;
        let k = *k;
        let rem = *rem;

        let (mut combined_u, _) = unsafe { temp_mat_uninit(ctx, n + 1, n + 1, stack) };
        let mut combined_u = combined_u.as_mat_mut();

        let (mut combined_u1, mut combined_u2) = combined_u.rb_mut().split_at_row_mut(k + 1);
        let u_lhs = u.rb();
        let u_rhs = um.rb();
        let (u_lhs1, u_lhs2) = (
            u_lhs.submatrix(0, 0, k + 1, k + 1),
            u_lhs.submatrix(k + 1, k + 1, rem + 1, rem + 1),
        );
        let (u_rhs1, u_rhs2) = u_rhs.split_at_row(k + 1);

        crate::utils::thread::join_raw(
            |par| {
                // matrix matrix
                crate::linalg::matmul::matmul(
                    ctx,
                    combined_u1.rb_mut(),
                    Accum::Replace,
                    u_lhs1,
                    u_rhs1,
                    math.one(),
                    par,
                );
                // rank 1 update
                crate::linalg::matmul::matmul(
                    ctx,
                    combined_u1.rb_mut(),
                    Accum::Add,
                    u_lhs.col(n).subrows(0, k + 1).as_mat(),
                    u_rhs2.row(rem).as_mat(),
                    math.one(),
                    par,
                );
            },
            |par| {
                // matrix matrix
                crate::linalg::matmul::matmul(
                    ctx,
                    combined_u2.rb_mut(),
                    Accum::Replace,
                    u_lhs2,
                    u_rhs2,
                    math.one(),
                    par,
                );
                // rank 1 update
                crate::linalg::matmul::matmul(
                    ctx,
                    combined_u2.rb_mut(),
                    Accum::Add,
                    u_lhs.col(0).subrows(k + 1, rem + 1).as_mat(),
                    u_rhs1.row(0).as_mat(),
                    math.one(),
                    par,
                );
            },
            par,
        );

        u.copy_from_with(ctx, combined_u.rb());
    };

    match u.rb_mut() {
        MatU::TwoRows(mut u) => {
            update_v(v.rb_mut(), par, stack);

            let (mut combined_u, _) = unsafe { temp_mat_uninit(ctx, 2, *n + 1, stack) };
            let mut combined_u = combined_u.as_mat_mut();
            crate::linalg::matmul::matmul(
                ctx,
                combined_u.rb_mut(),
                Accum::Replace,
                u.rb(),
                um.rb(),
                math.one(),
                par,
            );

            u.copy_from_with(ctx, combined_u.rb());
        }
        MatU::Full(u) => match par {
            #[cfg(feature = "rayon")]
            Par::Rayon(_) if !_v_is_none => {
                let req_v = crate::linalg::temp_mat_scratch::<C, T>(*n, *n).unwrap();
                let (mem_v, stack_u) =
                    stack.make_aligned_uninit::<u8>(req_v.size_bytes(), req_v.align_bytes());
                let stack_v = DynStack::new(mem_v);
                crate::utils::thread::join_raw(
                    |par| update_v(v.rb_mut(), par, stack_v),
                    |par| update_u(u, par, stack_u),
                    par,
                );
            }
            _ => {
                update_v(v.rb_mut(), par, stack);
                update_u(u, par, stack);
            }
        },
    }

    for i in n.indices() {
        write1!(diag[i] = math(diag[i] * max));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use dyn_stack::GlobalMemBuffer;
    use faer_traits::Unit;

    use super::*;
    use crate::{assert, utils::approx::*, Col, Mat, MatMut};
    use std::{
        mem::MaybeUninit,
        path::{Path, PathBuf},
    };

    fn parse_bidiag(path: &Path) -> (Col<f64>, Col<f64>) {
        let file = &*std::fs::read_to_string(path).unwrap();
        let mut diag = vec![];
        let mut subdiag = vec![];

        let mut iter = file.lines();
        for line in &mut iter {
            if line.starts_with("diag") {
                continue;
            }
            if line.starts_with("subdiag") {
                break;
            }

            let line = line.trim();
            let line = line.strip_suffix(",").unwrap_or(line);

            if line.is_empty() {
                continue;
            }

            diag.push(line.parse::<f64>().unwrap());
        }
        for line in iter {
            let line = line.trim();
            let line = line.strip_suffix(",").unwrap_or(line);

            if line.is_empty() {
                continue;
            }
            subdiag.push(line.parse::<f64>().unwrap());
        }

        assert!(diag.len() == subdiag.len());

        (
            Col::from_fn(diag.len(), |i| diag[i]),
            Col::from_fn(diag.len(), |i| subdiag[i]),
        )
    }

    #[test]
    fn test_qr_algorithm() {
        for file in
            std::fs::read_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/svd/"))
                .unwrap()
        {
            let (diag, mut subdiag) = parse_bidiag(&file.unwrap().path());
            subdiag[diag.nrows() - 1] = 0.0;

            with_dim!(n, diag.nrows());
            if *n > 512 {
                continue;
            }

            let diag = diag.as_ref().as_row_shape(n).try_as_col_major().unwrap();
            let subdiag = subdiag.as_ref().as_row_shape(n).try_as_col_major().unwrap();

            let mut d = diag.to_owned();
            let mut subd = subdiag.to_owned();

            let mut u = Mat::zeros_with(&ctx(), n, n);
            let mut v = Mat::zeros_with(&ctx(), n, n);

            for i in n.indices() {
                u[(i, i)] = 1.0;
                v[(i, i)] = 1.0;
            }

            qr_algorithm(
                &ctx(),
                d.as_mut().try_as_col_major_mut().unwrap(),
                subd.as_mut().try_as_col_major_mut().unwrap(),
                Some(u.as_mut()),
                Some(v.as_mut()),
            )
            .unwrap();

            for &x in subd.iter() {
                assert!(x == 0.0);
            }

            let mut approx_eq = CwiseMat(ApproxEq::<Unit, f64>::eps());
            approx_eq.0.abs_tol *=
                f64::max(diag.norm_max(), subdiag.norm_max()) * (*n as f64).sqrt();
            approx_eq.0.rel_tol *=
                f64::max(diag.norm_max(), subdiag.norm_max()) * (*n as f64).sqrt();
            let reconstructed = &u * d.as_diagonal() * v.adjoint();

            assert!(reconstructed ~ bidiag_to_mat(diag, subdiag));
        }
    }

    #[test]
    fn test_divide_and_conquer() {
        for file in
            std::fs::read_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/svd/"))
                .unwrap()
        {
            let (diag, subdiag) = parse_bidiag(&file.unwrap().path());
            with_dim!(n, diag.nrows());

            if *n > 1024 {
                continue;
            }

            let diag = diag.as_ref().as_row_shape(n).try_as_col_major().unwrap();
            let subdiag = subdiag.as_ref().as_row_shape(n).try_as_col_major().unwrap();

            let mut d = diag.to_owned();
            let mut subd = subdiag.to_owned();

            let mut u = Mat::zeros_with(&ctx(), *n + 1, *n + 1);
            let mut v = Mat::zeros_with(&ctx(), n, n);

            for i in n.indices() {
                u[(*i, *i)] = 1.0;
                v[(i, i)] = 1.0;
            }
            u[(*n, *n)] = 1.0;

            help!(Unit);
            let ctx = &Ctx::<Unit, f64>(Default::default());

            divide_and_conquer(
                &ctx,
                d.as_mut().try_as_col_major_mut().unwrap(),
                subd.as_mut().try_as_col_major_mut().unwrap(),
                MatU::Full(u.as_mut()),
                Some(v.as_mut()),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    divide_and_conquer_scratch::<Unit, f64>(*n, 4, true, true, Par::Seq).unwrap(),
                )),
                4,
            )
            .unwrap();

            for x in subd.iter_mut() {
                *x = 0.0;
            }

            let mut d2 = Mat::zeros_with(&ctx, *n + 1, n);
            for i in n.indices() {
                d2[(*i, i)] = d[i];
            }

            let mut approx_eq = CwiseMat(ApproxEq::<Unit, f64>::eps());
            approx_eq.0.abs_tol *=
                f64::max(diag.norm_max(), subdiag.norm_max()) * (*n as f64).sqrt() * 10.0;
            approx_eq.0.rel_tol *=
                f64::max(diag.norm_max(), subdiag.norm_max()) * (*n as f64).sqrt() * 10.0;
            let reconstructed = &u * &d2 * v.adjoint();

            assert!(reconstructed ~ bidiag_to_mat2(diag, subdiag));
        }
    }

    #[test]
    #[ignore]
    fn test_josef() {
        for file in
            std::fs::read_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/svd/"))
                .unwrap()
        {
            let (diag, subdiag) = parse_bidiag(&file.unwrap().path());
            let diag = z!(&diag).map(|uz!(x)| *x as f32);
            let subdiag = z!(&subdiag).map(|uz!(x)| *x as f32);

            with_dim!(n, diag.nrows());

            if *n <= 1024 {
                continue;
            }

            let diag = diag.as_ref().as_row_shape(n).try_as_col_major().unwrap();
            let subdiag = subdiag.as_ref().as_row_shape(n).try_as_col_major().unwrap();

            let mut d = diag.to_owned();
            let mut subd = subdiag.to_owned();

            let mut u = Mat::zeros_with(&ctx(), *n + 1, *n + 1);
            let mut v = Mat::zeros_with(&ctx(), n, n);

            for i in n.indices() {
                u[(*i, *i)] = 1.0;
                v[(i, i)] = 1.0;
            }
            u[(*n, *n)] = 1.0;

            help!(Unit);
            let ctx = &Ctx::<Unit, f32>(Default::default());

            divide_and_conquer(
                &ctx,
                d.as_mut().try_as_col_major_mut().unwrap(),
                subd.as_mut().try_as_col_major_mut().unwrap(),
                MatU::Full(u.as_mut()),
                Some(v.as_mut()),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    divide_and_conquer_scratch::<Unit, f32>(*n, 4, true, true, Par::Seq).unwrap(),
                )),
                4,
            )
            .unwrap();

            for x in subd.iter_mut() {
                *x = 0.0;
            }

            let mut d2 = Mat::zeros_with(&ctx, *n + 1, n);
            for i in n.indices() {
                d2[(*i, i)] = d[i];
            }

            let mut approx_eq = CwiseMat(ApproxEq::<Unit, f32>::eps());
            approx_eq.0.abs_tol *=
                f32::max(diag.norm_max(), subdiag.norm_max()) * (*n as f32).sqrt() * 10.0;
            approx_eq.0.rel_tol *=
                f32::max(diag.norm_max(), subdiag.norm_max()) * (*n as f32).sqrt() * 10.0;
            let reconstructed = &u * &d2 * v.adjoint();

            assert!(reconstructed ~ bidiag_to_mat2(diag, subdiag));
        }
    }

    #[test]
    fn test_deflation43() {
        let approx_eq = CwiseMat(ApproxEq::<Unit, f32>::eps());

        let n = 7;
        let k = 3;
        let diag_orig = &*vec![1.0, 5.0, 3.0, 1e-7, 4.0, 2.0, 2e-7_f32];
        let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];

        with_dim!(n, n);

        let n_jacobi = 2;
        let jacobi_coeffs = &mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
        let jacobi_indices = &mut *vec![n.idx(0); n_jacobi];
        let perm = &mut *vec![n.idx(0); *n];
        let diag = &mut *(diag_orig.to_vec());
        let col = &mut *(col_orig.to_vec());

        let mut diag = MatMut::from_column_major_slice_mut(diag, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();
        let mut col = MatMut::from_column_major_slice_mut(col, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();

        let (jacobi_0i, jacobi_ij) = deflate(
            &ctx(),
            diag.rb_mut(),
            col.rb_mut(),
            jacobi_coeffs,
            jacobi_indices,
            &mut *vec![n.idx(0); *n],
            Array::from_mut(perm, n),
            k,
            DynStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * *n]),
        );
        assert!(all(jacobi_0i == 2, jacobi_ij == 0));

        let perm_inv = &mut *vec![n.idx(0); *n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[*p] = n.idx(i);
        }

        let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);

        let mut M_orig = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M_orig[(i, i)] = diag_orig[*i];
            M_orig[(i, n.idx(0))] = col_orig[*i];
        }

        let mut M = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M[(i, i)] = diag[i];
            M[(i, n.idx(0))] = col[i];
        }

        M = P.inverse() * M * P;

        for (&rot, &i) in core::iter::zip(&*jacobi_coeffs, &*jacobi_indices).rev() {
            let (x, y) = M.two_rows_mut(n.idx(0), i);
            rot.apply_on_the_left_in_place(&ctx(), x, y);
        }

        assert!(M ~ M_orig);
    }

    #[test]
    fn test_deflation44() {
        let approx_eq = CwiseMat(ApproxEq::<Unit, f32>::eps());

        let n = 7;
        let k = 3;
        let diag_orig = &*vec![1.0, 5.0, 3.0, 1.0, 4.0, 2.0, 1.0];
        let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];

        with_dim!(n, n);

        let n_jacobi = 1;
        let jacobi_coeffs = &mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
        let jacobi_indices = &mut *vec![n.idx(0); n_jacobi];
        let perm = &mut *vec![n.idx(0); *n];
        let diag = &mut *(diag_orig.to_vec());
        let col = &mut *(col_orig.to_vec());

        let mut diag = MatMut::from_column_major_slice_mut(diag, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();
        let mut col = MatMut::from_column_major_slice_mut(col, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();

        let (jacobi_0i, jacobi_ij) = deflate(
            &ctx(),
            diag.rb_mut(),
            col.rb_mut(),
            jacobi_coeffs,
            jacobi_indices,
            &mut *vec![n.idx(0); *n],
            Array::from_mut(perm, n),
            k,
            DynStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * *n]),
        );
        assert!(all(jacobi_0i == 0, jacobi_ij == 1));

        let perm_inv = &mut *vec![n.idx(0); *n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[*p] = n.idx(i);
        }

        let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);

        let mut M_orig = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M_orig[(i, i)] = diag_orig[*i];
            M_orig[(i, n.idx(0))] = col_orig[*i];
        }

        let mut M = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M[(i, i)] = diag[i];
            M[(i, n.idx(0))] = col[i];
        }

        M = P.inverse() * M * P;

        for (&rot, &i) in core::iter::zip(&*jacobi_coeffs, &*jacobi_indices).rev() {
            let (i, j) = (n.idx(*i - 1), i);
            let (pi, pj) = (perm[*i], perm[*j]);

            let (x, y) = M.two_rows_mut(pi, pj);
            rot.apply_on_the_left_in_place(&ctx(), x, y);

            let (x, y) = M.two_cols_mut(pi, pj);
            rot.transpose(&ctx())
                .apply_on_the_right_in_place(&ctx(), x, y);
        }

        assert!(M ~ M_orig);
    }

    #[test]
    fn test_both_deflation() {
        let approx_eq = CwiseMat(ApproxEq::<Unit, f32>::eps());

        let n = 7;
        let k = 3;
        let diag_orig = &*vec![1.0, 5.0, 3.0, 2.0, 4.0, 2.0, 0.0];
        let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];

        with_dim!(n, n);

        let n_jacobi = 2;
        let jacobi_coeffs = &mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
        let jacobi_indices = &mut *vec![n.idx(0); n_jacobi];
        let perm = &mut *vec![n.idx(0); *n];
        let diag = &mut *(diag_orig.to_vec());
        let col = &mut *(col_orig.to_vec());

        let mut diag = MatMut::from_column_major_slice_mut(diag, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();
        let mut col = MatMut::from_column_major_slice_mut(col, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();

        let (jacobi_0i, jacobi_ij) = deflate(
            &ctx(),
            diag.rb_mut(),
            col.rb_mut(),
            jacobi_coeffs,
            jacobi_indices,
            &mut *vec![n.idx(0); *n],
            Array::from_mut(perm, n),
            k,
            DynStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * *n]),
        );
        assert!(all(jacobi_0i == 1, jacobi_ij == 1));

        let perm_inv = &mut *vec![n.idx(0); *n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[*p] = n.idx(i);
        }

        let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);

        let mut M_orig = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M_orig[(i, i)] = diag_orig[*i];
            M_orig[(i, n.idx(0))] = col_orig[*i];
        }

        let mut M = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M[(i, i)] = diag[i];
            M[(i, n.idx(0))] = col[i];
        }

        M = P.inverse() * M * P;

        for (&rot, &i) in core::iter::zip(&jacobi_coeffs[1..], &jacobi_indices[1..]).rev() {
            let (i, j) = (n.idx(*i - 1), i);
            let (pi, pj) = (perm[*i], perm[*j]);

            let (x, y) = M.two_rows_mut(pi, pj);
            rot.apply_on_the_left_in_place(&ctx(), x, y);

            let (x, y) = M.two_cols_mut(pi, pj);
            rot.transpose(&ctx())
                .apply_on_the_right_in_place(&ctx(), x, y);
        }

        for (&rot, &i) in core::iter::zip(&jacobi_coeffs[..1], &jacobi_indices[..1]).rev() {
            let (x, y) = M.two_rows_mut(n.idx(0), i);
            rot.apply_on_the_left_in_place(&ctx(), x, y);
        }

        assert!(M ~ M_orig);
    }
}
