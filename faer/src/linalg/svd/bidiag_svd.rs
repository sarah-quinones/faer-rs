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

use super::SvdError;
use crate::{assert, internal_prelude::*, perm::swap_cols_idx};
use core::mem::swap;
use linalg::jacobi::JacobiRotation;

#[math]
#[allow(dead_code)]
fn bidiag_to_mat<T: RealField>(
    diag: ColRef<'_, T, usize, ContiguousFwd>,
    subdiag: ColRef<'_, T, usize, ContiguousFwd>,
) -> Mat<T> {
    let n = diag.nrows();
    let mut m = Mat::zeros(n, n);

    for i in 0..n {
        m[(i, i)] = copy(diag[i]);
        if i + 1 < n {
            m[(i + 1, i)] = copy(subdiag[i]);
        }
    }

    m
}

#[math]
#[allow(dead_code)]
fn bidiag_to_mat2<T: RealField>(
    diag: ColRef<'_, T, usize, ContiguousFwd>,
    subdiag: ColRef<'_, T, usize, ContiguousFwd>,
) -> Mat<T> {
    let n = diag.nrows();
    let mut m = Mat::zeros(n + 1, n);

    for i in 0..n {
        m[(i, i)] = copy(diag[i]);
        m[(i + 1, i)] = copy(subdiag[i]);
    }

    m
}

#[math]
#[allow(dead_code)]
fn arrow_to_mat<T: RealField>(
    diag: ColRef<'_, T, usize, ContiguousFwd>,
    col0: ColRef<'_, T, usize, ContiguousFwd>,
) -> Mat<T> {
    let n = diag.nrows();
    let mut m = Mat::zeros(n + 1, n);

    for i in 0..n {
        m[(i, 0)] = copy(col0[i]);
        m[(i, i)] = copy(diag[i]);
    }

    m
}

/// secular eq must be increasing
#[math]
pub(crate) fn secular_eq_root_finder<T: RealField>(
    secular_eq: &dyn Fn(T, T) -> T,
    batch_secular_eq: &dyn Fn(&[T; 4], &[T; 4]) -> [T; 4],
    left: T,
    right: T,
    last: bool,
) -> (T, T) {
    let two = from_f64(2.0);
    let eight = from_f64(8.0);
    let one_half = from_f64(0.5);
    let epsilon = eps();

    let mid = left + mul_pow2(right - left, one_half);
    let [mut f_mid, f_max, f_mid_left_shift, f_mid_right_shift] = batch_secular_eq(
        &[zero(), copy(left), copy(left), copy(right)],
        &[
            mid,
            if last {
                right - left
            } else {
                mul_pow2(right - left, one_half)
            },
            mul_pow2(right - left, one_half),
            mul_pow2(left - right, one_half),
        ],
    );

    let (mut shift, mu) = if last || (f_mid > zero()) {
        (copy(left), mul_pow2(right - left, one_half))
    } else {
        (copy(right), mul_pow2(left - right, one_half))
    };

    if (f_mid_left_shift <= zero()) && (f_mid_right_shift > zero()) {
        return (shift, mu);
    }

    if !last {
        if shift == left {
            if f_mid_left_shift < zero() {
                shift = copy(right);
                f_mid = f_mid_right_shift;
            }
        } else if f_mid_right_shift > zero() {
            shift = copy(left);
            f_mid = f_mid_left_shift;
        }
    }

    enum SecantError {
        OutOfBounds,
        PrecisionLimitReached,
    }

    let secant = {
        |mut mu_cur: T, mut mu_prev: T, mut f_cur: T, mut f_prev: T| {
            if abs(f_prev) < abs(f_cur) {
                swap(&mut f_prev, &mut f_cur);
                swap(&mut mu_prev, &mut mu_cur);
            }

            let mut left_candidate = None;
            let mut right_candidate = None;

            let mut use_bisection = false;
            let same_sign = (f_prev > zero()) == (f_cur > zero());
            if !same_sign {
                let (min, max) = if mu_cur < mu_prev {
                    (copy(mu_cur), copy(mu_prev))
                } else {
                    (copy(mu_prev), copy(mu_cur))
                };
                left_candidate = Some(min);
                right_candidate = Some(max);
            }

            let mut err = SecantError::PrecisionLimitReached;

            while !(f_cur == zero())
                && abs(mu_cur - mu_prev) > eight * epsilon * max(abs(mu_cur), abs(mu_prev))
                && abs(f_cur - f_prev) > epsilon
                && !use_bisection
            {
                // rational interpolation: fit a function of the form a / mu + b through
                // the two previous iterates and use its
                // zero to compute the next iterate
                let a = (f_cur - f_prev) * (mu_prev * mu_cur) / (mu_prev - mu_cur);
                let b = f_cur - a / mu_cur;
                let mu_zero = -a / b;
                let f_zero = secular_eq(copy(shift), copy(mu_zero));

                if f_zero < zero() {
                    left_candidate = Some(copy(mu_zero));
                } else {
                    right_candidate = Some(copy(mu_zero));
                }

                mu_prev = copy(mu_cur);
                f_prev = copy(f_cur);
                mu_cur = copy(mu_zero);
                f_cur = copy(f_zero);

                if shift == left && ((mu_cur < zero()) || mu_cur > right - left) {
                    err = SecantError::OutOfBounds;
                    use_bisection = true;
                }
                if shift == right && ((mu_cur > zero()) || mu_cur < left - right) {
                    err = SecantError::OutOfBounds;
                    use_bisection = true;
                }
                if abs(f_cur) > abs(f_prev) {
                    // find mu such that a / mu + b = -k * f_zero
                    // a / mu = -f_zero - b
                    // mu = -a / (f_zero + b)
                    let mut k = one();
                    for _ in 0..4 {
                        let mu_opposite = -a / (k * f_zero + b);
                        let f_opposite = secular_eq(copy(shift), copy(mu_opposite));
                        if (f_zero < zero()) && (f_opposite >= zero()) {
                            // this will be our right candidate
                            right_candidate = Some(mu_opposite);
                            break;
                        }
                        if (f_zero > zero()) && (f_opposite <= zero()) {
                            // this will be our left candidate
                            left_candidate = Some(mu_opposite);
                            break;
                        }
                        k = mul_pow2(k, two);
                    }
                    use_bisection = true;
                }
            }
            (use_bisection, mu_cur, left_candidate, right_candidate, err)
        }
    };

    let (mut left_shifted, mut f_left, mut right_shifted, mut f_right) = if shift == left {
        (
            zero(),
            -infinity(),
            if last {
                right - left
            } else {
                (right - left) * one_half
            },
            copy(*if last { &f_max } else { &f_mid }),
        )
    } else {
        ((left - right) * one_half, copy(f_mid), zero(), infinity())
    };

    let mut iteration_count = 0;
    let mut f_prev = copy(f_mid);
    // try to find non zero starting bounds

    let half0 = copy(one_half);
    let half1 = mul_pow2(half0, half0);
    let half2 = mul_pow2(half1, half1);
    let half3 = mul_pow2(half2, half2);

    let mu_values = if shift == left {
        [
            mul_pow2(right_shifted, half3),
            mul_pow2(right_shifted, half2),
            mul_pow2(right_shifted, half1),
            mul_pow2(right_shifted, half0),
        ]
    } else {
        [
            mul_pow2(left_shifted, half3),
            mul_pow2(left_shifted, half2),
            mul_pow2(left_shifted, half1),
            mul_pow2(left_shifted, half0),
        ]
    };

    let f_values = batch_secular_eq(&[(); 4].map(|_| copy(shift)), &mu_values);

    if shift == left {
        let mut i = 0;
        for (idx, (mu, f)) in core::iter::zip(&mu_values, &f_values).enumerate() {
            if *f < zero() {
                left_shifted = copy(*mu);
                f_left = copy(*f);
                i = idx + 1;
            }
        }
        if i < f_values.len() {
            right_shifted = copy(mu_values[i]);
            f_right = copy(f_values[i]);
        }
    } else {
        let mut i = 0;
        for (idx, (mu, f)) in core::iter::zip(&mu_values, &f_values).enumerate() {
            if *f > zero() {
                right_shifted = copy(*mu);
                f_right = copy(*f);
                i = idx + 1;
            }
        }
        if i < f_values.len() {
            left_shifted = copy(mu_values[i]);
            f_left = copy(f_values[i]);
        }
    }

    // try bisection just to get a good guess for secant
    while right_shifted - left_shifted
        > two * epsilon * (max(abs(left_shifted), abs(right_shifted)))
    {
        let mid_shifted_arithmetic = (left_shifted + right_shifted) * one_half;
        let mut mid_shifted_geometric = sqrt(abs(left_shifted)) * sqrt(abs(right_shifted));
        if left_shifted < zero() {
            mid_shifted_geometric = -mid_shifted_geometric;
        }
        let mid_shifted = if mid_shifted_geometric == zero() {
            mid_shifted_arithmetic
        } else {
            mid_shifted_geometric
        };
        let f_mid = secular_eq(copy(shift), copy(mid_shifted));

        if f_mid == zero() {
            return (shift, mid_shifted);
        } else if f_mid > zero() {
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
    let args = if left_shifted == zero() {
        (
            mul_pow2(right_shifted, two),
            copy(right_shifted),
            f_prev,
            f_right,
        )
    } else if right_shifted == zero() {
        (
            mul_pow2(left_shifted, two),
            copy(left_shifted),
            f_prev,
            f_left,
        )
    } else {
        (copy(left_shifted), copy(right_shifted), f_left, f_right)
    };

    let (use_bisection, mut mu_cur, left_candidate, right_candidate, _err) =
        secant(args.0, args.1, args.2, args.3);

    match (left_candidate, right_candidate) {
        (Some(left), Some(right)) if left < right => {
            if left > left_shifted {
                left_shifted = left;
            }
            if right < right_shifted {
                right_shifted = right;
            }
        }
        _ => (),
    }

    // secant failed, use bisection again
    if use_bisection {
        while right_shifted - left_shifted
            > two * epsilon * (max(abs(left_shifted), abs(right_shifted)))
        {
            let mid_shifted = (left_shifted + right_shifted) * one_half;
            let f_mid = secular_eq(copy(shift), copy(mid_shifted));

            if f_mid == zero() {
                break;
            } else if f_mid > zero() {
                right_shifted = mid_shifted;
            } else {
                left_shifted = mid_shifted;
            }
        }

        mu_cur = (left_shifted + right_shifted) * one_half;
    }

    (shift, mu_cur)
}

#[math]
pub(super) fn qr_algorithm<T: RealField>(
    diag: ColMut<'_, T, usize, ContiguousFwd>,
    subdiag: ColMut<'_, T, usize, ContiguousFwd>,
    u: Option<MatMut<'_, T>>,
    v: Option<MatMut<'_, T>>,
) -> Result<(), SvdError> {
    let n = diag.nrows();
    let eps = eps();
    let sml = min_positive();

    let mut u = u;
    let mut v = v;

    if let Some(mut u) = u.rb_mut() {
        u.fill(zero());
        u.diagonal_mut().fill(one());
    }
    if let Some(mut u) = v.rb_mut() {
        u.fill(zero());
        u.diagonal_mut().fill(one());
    }

    if n == 0 {
        return Ok(());
    }

    let max_iters = Ord::max(30, nbits::<T>() / 2)
        .saturating_mul(n)
        .saturating_mul(n);

    let last = n - 1;

    let max = max(diag.norm_max(), subdiag.norm_max());
    if max == zero() {
        return Ok(());
    }

    let max_inv = recip(max);

    let mut diag = diag;
    let mut subdiag = subdiag;
    let mut u = u;
    let mut v = v;

    for x in diag.rb_mut().iter_mut() {
        *x = x * max_inv;
    }
    for x in subdiag.rb_mut().iter_mut() {
        *x = x * max_inv;
    }

    {
        let eps2 = eps * eps;

        for iter in 0..max_iters {
            for i0 in 0..last {
                let i1 = i0 + 1;
                if abs2(subdiag[i0]) <= eps2 * abs(diag[i0] * diag[i1]) + sml {
                    subdiag[i0] = zero();
                }
            }

            for i in 0..n {
                if abs(diag[i]) <= eps {
                    diag[i] = zero();
                }
            }

            let mut end = n;
            while end >= 2 && abs2(subdiag[end - 2]) <= sml {
                end -= 1
            }

            if end == 1 {
                break;
            }

            let mut start = end - 1;
            while start >= 1 && !(subdiag[start - 1] == zero()) {
                start -= 1;
            }

            let mut found_zero_diag = false;

            for i in start..end - 1 {
                if diag[i] == zero() {
                    found_zero_diag = true;

                    let mut val = copy(subdiag[i]);
                    subdiag[i] = zero();
                    for j in i + 1..end {
                        let rot = JacobiRotation::make_givens(copy(diag[j]), copy(val));
                        diag[j] = abs(rot.c * diag[j] - rot.s * val);
                        if j + 1 < end {
                            val = -rot.s * subdiag[j];
                            subdiag[j] = rot.c * subdiag[j];
                        }
                        if let Some(v) = v.rb_mut() {
                            let (i, j) = v.two_cols_mut(i, j);
                            rot.apply_on_the_right_in_place((j, i));
                        }
                    }
                }
            }

            if found_zero_diag {
                continue;
            }

            let end2 = end - 2;
            let end1 = end - 1;

            let t00 = if end - start == 2 {
                abs2(diag[end2])
            } else {
                abs2(diag[end2] + abs2(subdiag[end - 3]))
            };
            let t11 = abs2(diag[end1]) + abs2(subdiag[end2]);
            let t01 = diag[end2] * subdiag[end2];

            let t01_2 = abs2(t01);

            let mu;
            if t01_2 > sml {
                let d = mul_pow2((t00 - t11), from_f64(0.5));
                let mut delta = sqrt(abs2(d) + t01_2);
                if d < zero() {
                    delta = -delta;
                }
                mu = t11 - (t01_2 / (d + delta));
            } else {
                mu = copy(t11);
            }

            let mut y = abs2(diag[start]) - mu;
            let mut z = diag[start] * subdiag[start];

            for k in start..end1 {
                let rot = JacobiRotation::make_givens(copy(y), copy(z));
                if k > start {
                    subdiag[k - 1] = abs(rot.c * y - rot.s * z);
                }

                let mut diag_k = copy(diag[k]);

                let tmp = (
                    rot.c * diag_k - rot.s * subdiag[k],
                    rot.s * diag_k + rot.c * subdiag[k],
                );

                diag_k = tmp.0;
                subdiag[k] = tmp.1;

                let k1 = k + 1;

                y = copy(diag_k);
                z = -rot.s * diag[k1];
                diag[k1] = rot.c * diag[k1];

                if let Some(u) = u.rb_mut() {
                    let (k, k1) = u.two_cols_mut(k, k1);
                    rot.apply_on_the_right_in_place((k1, k));
                }

                let rot = JacobiRotation::make_givens(copy(y), copy(z));

                diag_k = abs(rot.c * y - rot.s * z);
                diag[k] = diag_k;

                let tmp = (
                    rot.c * subdiag[k] - rot.s * diag[k1],
                    rot.s * subdiag[k] + rot.c * diag[k1],
                );
                subdiag[k] = tmp.0;
                diag[k1] = tmp.1;

                if k < end - 2 {
                    y = copy(subdiag[k]);
                    z = -rot.s * subdiag[k1];
                    subdiag[k1] = rot.c * subdiag[k1];
                }

                if let Some(v) = v.rb_mut() {
                    let (k, k1) = v.two_cols_mut(k, k1);
                    rot.apply_on_the_right_in_place((k1, k));
                }
            }

            if iter + 1 == max_iters {
                for x in diag.rb_mut().iter_mut() {
                    *x = x * max;
                }
                for x in subdiag.rb_mut().iter_mut() {
                    *x = x * max;
                }

                return Err(SvdError::NoConvergence);
            }
        }
    }

    for j in 0..n {
        let d = diag.rb_mut().at_mut(j);

        if *d < zero() {
            *d = -d;
            if let Some(mut v) = v.rb_mut() {
                for i in 0..n {
                    v[(i, j)] = -v[(i, j)];
                }
            }
        }
    }

    for k in 0..n {
        let mut max = zero();
        let mut idx = k;

        for kk in k..n {
            if diag[kk] > max {
                max = copy(diag[kk]);
                idx = kk;
            }
        }

        if k != idx {
            let dk = copy(diag[k]);
            let di = copy(diag[idx]);

            diag[idx] = dk;
            diag[k] = di;

            if let Some(u) = u.rb_mut() {
                swap_cols_idx(u, k, idx);
            }
            if let Some(v) = v.rb_mut() {
                swap_cols_idx(v, k, idx);
            }
        }
    }

    for x in diag.rb_mut().iter_mut() {
        *x = x * max;
    }
    for x in subdiag.rb_mut().iter_mut() {
        *x = x * max;
    }

    Ok(())
}

#[math]
fn compute_svd_of_m<T: RealField>(
    um: Option<MatMut<'_, T, usize, usize, ContiguousFwd>>,
    vm: Option<MatMut<'_, T, usize, usize, ContiguousFwd>>,
    diag: ColMut<'_, T, usize, ContiguousFwd>,
    col0: ColRef<'_, T, usize, ContiguousFwd>,
    outer_perm: &[usize],
    stack: &mut DynStack,
) {
    let mut diag = diag;
    let mut um = um;
    let mut vm = vm;
    let n = diag.nrows();

    diag[0] = zero();
    let mut actual_n = n;
    while actual_n > 1 && (diag[actual_n - 1] == zero()) {
        actual_n -= 1;
    }

    let (perm, stack) = stack.collect(
        col0.iter()
            .take(actual_n)
            .map(|x| abs(*x))
            .enumerate()
            .filter(|(_, x)| !(*x == zero()))
            .map(|(i, _)| i),
    );
    let perm = &*perm;
    with_dim!(o, perm.len());

    let (mut col0_perm, stack) = unsafe { temp_mat_uninit(o, 1, stack) };
    let (mut diag_perm, stack) = unsafe { temp_mat_uninit(o, 1, stack) };

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
        col0_perm[k] = copy(col0[p]);
        diag_perm[k] = copy(diag[p]);
    }

    let (mut shifts, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
    let shifts = shifts.as_mat_mut();
    let (mut mus, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
    let mus = mus.as_mat_mut();
    let (mut singular_vals, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
    let singular_vals = singular_vals.as_mat_mut();
    let (mut zhat, stack) = unsafe { temp_mat_uninit(n, 1, stack) };
    let zhat = zhat.as_mat_mut();

    let mut shifts = shifts.col_mut(0).try_as_col_major_mut().unwrap();
    let mut mus = mus.col_mut(0).try_as_col_major_mut().unwrap();
    let mut s = singular_vals.col_mut(0).try_as_col_major_mut().unwrap();
    let mut zhat = zhat.col_mut(0).try_as_col_major_mut().unwrap();

    with_dim!(N, diag.nrows());
    compute_singular_values(
        shifts.rb_mut().as_row_shape_mut(N),
        mus.rb_mut().as_row_shape_mut(N),
        s.rb_mut().as_row_shape_mut(N),
        diag.rb().as_row_shape(N),
        diag_perm.rb(),
        col0.as_row_shape(N),
        col0_perm.rb(),
    );

    let perm = Idx::from_slice_ref_checked(perm, N);
    let outer_perm = Array::from_ref(Idx::from_slice_ref_checked(outer_perm, N), N);

    perturb_col0(
        zhat.rb_mut().as_row_shape_mut(N),
        col0.as_row_shape(N),
        diag.rb().as_row_shape(N),
        perm,
        s.rb().as_row_shape(N),
        shifts.rb().as_row_shape(N),
        mus.rb().as_row_shape(N),
    );

    let (mut col_perm, stack) = stack.make_with(actual_n, |i| i);
    let (mut col_perm_inv, _) = stack.make_with(actual_n, |i| i);

    for i0 in 0..actual_n - 1 {
        let i1 = i0 + 1;
        if s[i0] > s[i1] {
            let si = copy(s[i0]);
            let sj = copy(s[i1]);
            s[i0] = sj;
            s[i1] = si;

            col_perm.swap(i0, i1);
        }
    }
    for (i, p) in col_perm.iter().copied().enumerate() {
        col_perm_inv[p] = i;
    }

    compute_singular_vectors(
        um.rb_mut(),
        vm.rb_mut().map(|v| v.as_shape_mut(N, N)),
        zhat.rb().as_row_shape(N),
        diag.rb().as_row_shape(N),
        perm,
        outer_perm,
        &col_perm_inv,
        actual_n,
        shifts.rb().as_row_shape(N),
        mus.rb().as_row_shape(N),
    );

    for (idx, diag) in diag
        .rb_mut()
        .subrows_range_mut((0usize, actual_n))
        .iter_mut()
        .enumerate()
    {
        *diag = copy(s[actual_n - idx - 1]);
    }

    for (idx, diag) in diag
        .rb_mut()
        .subrows_range_mut((actual_n, n.end()))
        .iter_mut()
        .enumerate()
    {
        *diag = copy(s[actual_n + idx]);
    }
}

#[math]
#[inline(never)]
fn compute_singular_vectors<'N, T: RealField>(
    mut um: Option<MatMut<T, usize, usize, ContiguousFwd>>,
    mut vm: Option<MatMut<T, Dim<'N>, Dim<'N>, ContiguousFwd>>,
    zhat: ColRef<T, Dim<'N>, ContiguousFwd>,
    diag: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
    perm: &[Idx<'N>],
    outer_perm: &Array<'N, Idx<'N>>,
    col_perm_inv: &[usize],
    actual_n: usize,
    shifts: ColRef<T, Dim<'N>, ContiguousFwd>,
    mus: ColRef<T, Dim<'N>, ContiguousFwd>,
) {
    let n = diag.nrows();

    for k in n.indices() {
        let actual_k = if *k >= actual_n {
            k
        } else {
            n.idx(actual_n - col_perm_inv[*k] - 1)
        };

        if let Some(mut u) = um.rb_mut() {
            u[(*n, *actual_k)] = zero();
        }

        let mut u = um
            .rb_mut()
            .map(|u| u.submatrix_mut(0, 0, n, n).col_mut(actual_k));
        let mut v = vm.rb_mut().map(|v| {
            v.submatrix_mut(IdxInc::ZERO, IdxInc::ZERO, n, n)
                .col_mut(actual_k)
        });

        if zhat[k] == zero() {
            if let Some(mut u) = u.rb_mut() {
                u[outer_perm[k]] = one();
            }
            if let Some(mut v) = v.rb_mut() {
                v[outer_perm[k]] = one();
            }
            continue;
        }

        let mu = copy(mus[k]);
        let shift = copy(shifts[k]);

        if let Some(mut u) = u.rb_mut() {
            for &i in perm {
                u[outer_perm[i]] = (zhat[i] / ((diag[i] - shift) - mu)) / (diag[i] + (shift + mu));
            }
            let norm_inv = recip(u.norm_l2());
            z!(u.rb_mut()).for_each(|uz!(x)| *x = x * norm_inv);
        }

        if let Some(mut v) = v {
            v[outer_perm[n.idx(0)]] = -one();
            for &i in &perm[1..] {
                v[outer_perm[i]] =
                    ((diag[i] * zhat[i]) / ((diag[i] - shift) - mu)) / (diag[i] + (shift + mu));
            }
            let norm_inv = recip(v.norm_l2());
            z!(v.rb_mut()).for_each(|uz!(x)| *x = x * norm_inv);
        }
    }
    if let Some(mut u) = um.rb_mut() {
        u[(*n, *n)] = one();
    }
}

#[math]
fn perturb_col0<'N, T: RealField>(
    zhat: ColMut<'_, T, Dim<'N>, ContiguousFwd>,
    col0: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
    diag: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
    perm: &[Idx<'N>],
    s: ColRef<T, Dim<'N>, ContiguousFwd>,
    shifts: ColRef<T, Dim<'N>, ContiguousFwd>,
    mus: ColRef<T, Dim<'N>, ContiguousFwd>,
) {
    let mut zhat = zhat;

    let n = diag.nrows();
    let m = perm.len();
    if m == 0 {
        z!(zhat).for_each(|uz!(x)| *x = zero());
        return;
    }

    let last_idx = perm[m - 1];
    for k in n.indices() {
        if col0[k] == zero() {
            zhat[k] = zero();
            continue;
        }

        let dk = &diag[k];
        // NOTE: the order of operations is crucial here
        let mut prod = (s[last_idx] + *dk) * (mus[last_idx] + (shifts[last_idx] - *dk));

        for (l, &i) in perm.iter().enumerate() {
            if i == k {
                continue;
            }
            if i >= k && l == 0 {
                prod = zero();
                break;
            }

            let j = if i < k {
                i
            } else if l > 0 {
                perm[l - 1]
            } else {
                i
            };

            let term =
                ((s[j] + *dk) / (diag[i] + *dk)) * ((mus[j] + (shifts[j] - *dk)) / (diag[i] - *dk));
            prod = prod * term;
        }

        let tmp = sqrt(prod);

        if col0[k] > zero() {
            zhat[k] = tmp;
        } else {
            zhat[k] = -tmp;
        }
    }
}

#[math]
fn compute_singular_values<'N, 'O, T: RealField>(
    mut shifts: ColMut<T, Dim<'N>, ContiguousFwd>,
    mut mus: ColMut<T, Dim<'N>, ContiguousFwd>,
    mut s: ColMut<T, Dim<'N>, ContiguousFwd>,
    diag: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
    diag_perm: ColRef<'_, T, Dim<'O>, ContiguousFwd>,
    col0: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
    col0_perm: ColRef<'_, T, Dim<'O>, ContiguousFwd>,
) {
    let n = diag.nrows();
    let mut actual_n = *n;
    while actual_n > 1 && (col0[n.idx(actual_n - 1)] == zero()) {
        actual_n -= 1;
    }
    let actual_n = actual_n;

    let first = n.idx(0);

    'kth_iter: for k in n.indices() {
        s[k] = zero();
        shifts[k] = zero();
        mus[k] = zero();

        if (col0[k] == zero()) || actual_n == 1 {
            s[k] = if *k == 0 {
                copy(col0[first])
            } else {
                copy(diag[k])
            };
            shifts[k] = copy(s[k]);
            mus[k] = zero();

            continue 'kth_iter;
        }

        let last_k = *k == actual_n - 1;
        let left = copy(diag[k]);
        let right = if last_k {
            diag[n.idx(actual_n - 1)] + col0.norm_l2()
        } else {
            let mut l = n.idx(*k + 1);
            while col0[l] == zero() {
                l = n.idx(*l + 1);
            }
            copy(diag[l])
        };

        let (shift, mu) = secular_eq_root_finder(
            &|shift, mu| secular_eq(shift, mu, col0_perm, diag_perm),
            &|shift, mu| batch_secular_eq(shift, mu, col0_perm, diag_perm),
            left,
            right,
            last_k,
        );

        s[k] = shift + mu;
        shifts[k] = shift;
        mus[k] = mu;
    }
}

#[math]
fn secular_eq<'N, T: RealField>(
    shift: T,
    mu: T,
    col0_perm: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
    diag_perm: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
) -> T {
    let mut res = one();

    let n = diag_perm.nrows();
    for i in n.indices() {
        let c = copy(col0_perm[i]);
        let d = copy(diag_perm[i]);
        res = res + (c / ((d - shift) - mu)) * (c / ((d + shift) + mu));
    }

    res
}

#[math]
fn batch_secular_eq<'N, const N: usize, T: RealField>(
    shift: &[T; N],
    mu: &[T; N],
    col0_perm: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
    diag_perm: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
) -> [T; N] {
    let n = col0_perm.nrows();
    let mut res = [(); N].map(|_| one());
    for i in n.indices() {
        let c = copy(col0_perm[i]);
        let d = copy(diag_perm[i]);

        for ((res, mu), shift) in res.iter_mut().zip(mu.iter()).zip(shift.iter()) {
            *res = (*res) + (abs2(c) / (((d - *shift) - *mu) * ((d + *shift) + *mu)));
        }
    }
    res
}

#[math]
fn deflate<T: RealField>(
    diag: ColMut<'_, T, usize, ContiguousFwd>,
    col0: ColMut<'_, T, usize, ContiguousFwd>,
    jacobi_coeff: &mut [JacobiRotation<T>],
    jacobi_idx: &mut [usize],
    transpositions: &mut [usize],
    perm: &mut [usize],
    k: usize,
    stack: &mut DynStack,
) -> (usize, usize) {
    let mut diag = diag;
    let mut col0 = col0;

    let n = diag.nrows();
    let first = 0;

    let mut jacobi_0i = 0;
    let mut jacobi_ij = 0;

    let (max_diag, max_col0) = (
        diag.rb().subrows_range((first + 1, n)).norm_max(),
        col0.norm_max(),
    );
    let mx = max(max_diag, max_col0);

    let eps = eps();
    let sml = min_positive();

    let eps_strict = max(eps * max_diag, sml);

    let eps_coarse = from_f64(8.0) * eps * mx;

    // condition 4.1
    if diag[first] < eps_coarse {
        diag[first] = copy(eps_coarse);
        col0[first] = copy(eps_coarse);
    }

    // condition 4.2
    for i in first + 1..n {
        if abs(col0[i]) < eps_strict {
            col0[i] = zero();
        }
    }

    // condition 4.3
    for i in first + 1..n {
        if diag[i] < eps_coarse {
            if let Some(rot) = deflation_43(diag.rb_mut(), col0.rb_mut(), i) {
                jacobi_coeff[jacobi_0i] = rot;
                jacobi_idx[jacobi_0i] = i;
                jacobi_0i += 1;
            }
        }
    }

    let mut total_deflation = true;
    for i in first + 1..n {
        if !(abs(col0[i]) < sml) {
            total_deflation = false;
            break;
        }
    }

    perm[first] = first;
    let mut p = 1;
    for i in first + 1..n {
        if abs(diag[i]) < sml {
            perm[p] = i;
            p += 1;
        }
    }

    let mut i = 1;
    let mut j = k + 1;

    for p in &mut perm.as_mut()[p..] {
        if i >= k + 1 {
            *p = j;
            j += 1;
        } else if j >= n {
            *p = i;
            i += 1;
        } else if diag[i] < diag[j] {
            *p = j;
            j += 1;
        } else {
            *p = i;
            i += 1;
        }
    }

    if total_deflation {
        for i in first + 1..n {
            let i1 = i - 1;

            let pi = perm[i];
            if abs(diag[pi]) < sml || diag[pi] > diag[first] {
                perm[i1] = perm[i];
            } else {
                perm[i1] = first;
                break;
            }
        }
    }

    let (mut real_ind, stack) = stack.collect(0..n);
    let (mut real_col, _) = stack.collect(0..n);

    for i in (if total_deflation { first } else { first + 1 })..n {
        let pi = perm[n - if total_deflation { i + 1 } else { i }];
        let j = real_col[pi];

        let (a, b) = (copy(diag[i]), copy(diag[j]));
        diag[i] = b;
        diag[j] = a;

        if i != 0 && j != 0 {
            let (a, b) = (copy(col0[i]), copy(col0[j]));
            col0[i] = b;
            col0[j] = a;
        }

        transpositions[i] = j;
        let real_i = real_ind[i];
        real_col[real_i] = j;
        real_col[pi] = i;
        real_ind[j] = real_i;
        real_ind[i] = pi;
    }

    col0[first] = copy(diag[first]);

    for i in 0..n {
        perm[i] = i;
    }

    for (i, &j) in transpositions.iter().enumerate() {
        perm.as_mut().swap(i, j);
    }

    // condition 4.4
    let mut i = n - 1;
    while i > 0 && (abs(diag[i]) < sml || abs(col0[i]) < sml) {
        i -= 1;
    }

    while i > 1 {
        let i1 = i - 1;
        if diag[i] - diag[i1] < eps_strict {
            if let Some(rot) = deflation_44(diag.rb_mut(), col0.rb_mut(), i1, i) {
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
fn deflation_43<T: RealField>(
    diag: ColMut<'_, T, usize, ContiguousFwd>,
    col0: ColMut<'_, T, usize, ContiguousFwd>,
    i: usize,
) -> Option<JacobiRotation<T>> {
    let mut diag = diag;
    let mut col0 = col0;

    let first = 0;

    let p = copy(col0[first]);
    let q = copy(col0[i]);

    if (p == zero()) && (q == zero()) {
        diag[i] = zero();
        return None;
    }

    let rot = JacobiRotation::make_givens(copy(p), copy(q));

    let r = rot.c * p - rot.s * q;
    col0[first] = copy(r);
    diag[first] = copy(r);
    col0[i] = zero();
    diag[i] = zero();

    Some(rot)
}

#[math]
fn deflation_44<T: RealField>(
    diag: ColMut<'_, T, usize, ContiguousFwd>,
    col0: ColMut<'_, T, usize, ContiguousFwd>,
    i: usize,
    j: usize,
) -> Option<JacobiRotation<T>> {
    let mut diag = diag;
    let mut col0 = col0;

    let p = copy(col0[i]);
    let q = copy(col0[j]);

    if (p == zero()) && (q == zero()) {
        diag[i] = copy(diag[j]);
        return None;
    }

    let rot = JacobiRotation::make_givens(copy(p), copy(q));

    let r = rot.c * p - rot.s * q;
    col0[i] = copy(r);
    col0[j] = zero();
    diag[i] = copy(diag[j]);

    Some(rot)
}

#[derive(Debug)]
pub(super) enum MatU<'a, T: RealField> {
    Full(MatMut<'a, T>),
    TwoRows(MatMut<'a, T>),
    TwoRowsStorage(MatMut<'a, T>),
}

impl<'short, T: RealField> ReborrowMut<'short> for MatU<'_, T> {
    type Target = MatU<'short, T>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        match self {
            Self::Full(u) => MatU::Full(u.rb_mut()),
            Self::TwoRows(u) => MatU::TwoRows(u.rb_mut()),
            Self::TwoRowsStorage(u) => MatU::TwoRowsStorage(u.rb_mut()),
        }
    }
}

pub(super) fn divide_and_conquer_scratch<T: ComplexField>(
    n: usize,
    qr_fallback_threshold: usize,
    compute_u: bool,
    compute_v: bool,
    par: Par,
) -> Result<StackReq, SizeOverflow> {
    if n < qr_fallback_threshold {
        temp_mat_scratch::<T>(n + 1, n + 1)?.try_array(2)
    } else {
        let _ = par;
        let perm = StackReq::try_new::<usize>(n)?.try_array(8)?;
        let jacobi_coeffs = StackReq::try_new::<JacobiRotation<T>>(n)?;

        let um = temp_mat_scratch::<T>(n + 1, n + 1)?;
        let vm = temp_mat_scratch::<T>(n, if compute_v { n } else { 0 })?;

        let combined_u = temp_mat_scratch::<T>(if compute_u { n + 1 } else { 2 }, n + 1)?;
        let combined_v = vm;

        StackReq::try_all_of([perm, jacobi_coeffs, um, vm, combined_u, combined_v])
    }
}

#[math]
pub(super) fn divide_and_conquer<T: RealField>(
    diag: ColMut<'_, T, usize, ContiguousFwd>,
    subdiag: ColMut<'_, T, usize, ContiguousFwd>,
    u: MatU<'_, T>,
    v: Option<MatMut<'_, T>>,
    par: Par,
    stack: &mut DynStack,
    qr_fallback_threshold: usize,
) -> Result<(), SvdError> {
    let qr_fallback_threshold = Ord::max(qr_fallback_threshold, 4);

    let n = diag.nrows();

    let mut diag = diag;
    let mut subdiag = subdiag;
    let mut u = u;
    let mut v = v;

    if n < qr_fallback_threshold {
        let n1 = n + 1;

        let (mut diag_alloc, stack) = unsafe { temp_mat_uninit(n1, 1, stack) };
        let (mut subdiag_alloc, stack) = unsafe { temp_mat_uninit(n1, 1, stack) };

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

        for i in 0..n {
            diag_alloc[i] = copy(diag[i]);
            subdiag_alloc[i] = copy(subdiag[i]);
        }
        diag_alloc[n] = zero();
        subdiag_alloc[n] = zero();

        let (mut u_alloc, stack) = unsafe { temp_mat_uninit(n1, n1, stack) };
        let (mut v_alloc, _) = unsafe { temp_mat_uninit(n1, n1, stack) };
        let mut u_alloc = u_alloc.as_mat_mut();
        let mut v_alloc = v_alloc.as_mat_mut();

        qr_algorithm(
            diag_alloc.rb_mut(),
            subdiag_alloc.rb_mut(),
            Some(u_alloc.rb_mut()),
            Some(v_alloc.rb_mut()),
        )?;

        let mut first_zero = n;
        for i in 0..n1 {
            if diag_alloc[i] == zero() {
                first_zero = i;
                break;
            }
        }

        let mut last = n;
        for i in first_zero..n1 {
            if abs(v_alloc[(n, i)]) == one() {
                last = i;
            }
        }

        if last != n {
            swap_cols_idx(v_alloc.rb_mut(), last, n);
        }

        if v_alloc[(n, n)] < zero() {
            for i in 0..n1 {
                u_alloc[(i, n)] = -u_alloc[(i, n)];
            }
            for i in 0..n1 {
                v_alloc[(i, n)] = -v_alloc[(i, n)];
            }
        }

        assert!(v_alloc[(n, n)] == one());

        for i in 0..n {
            diag[i] = copy(diag_alloc[i]);
            subdiag[i] = copy(subdiag_alloc[i]);
        }

        if let Some(mut v) = v.rb_mut() {
            v.copy_from(v_alloc.rb().submatrix(0, 0, n, n));
        }
        match u.rb_mut() {
            MatU::Full(u) => u.submatrix_mut(0, 0, n1, n1).copy_from(u_alloc.rb()),
            MatU::TwoRows(u) => {
                let (mut top, mut bot) = u.subcols_mut(0, n1).two_rows_mut(0, 1);

                top.copy_from(u_alloc.rb().row(0));
                bot.copy_from(u_alloc.rb().row(n));
            }
            MatU::TwoRowsStorage(_) => {}
        }

        return Ok(());
    }

    let max = max(diag.norm_max(), subdiag.norm_max());

    if max == zero() {
        match u {
            MatU::Full(mut u) => {
                u.fill(zero());
                u.diagonal_mut().fill(one());
            }
            MatU::TwoRows(mut u) => {
                u.fill(zero());
                u[(0, 0)] = one();
                u[(1, n)] = zero();
            }
            MatU::TwoRowsStorage(_) => {}
        }
        if let Some(mut u) = v.rb_mut() {
            u.fill(zero());
            u.diagonal_mut().fill(one());
        }
        return Ok(());
    }

    let max_inv = recip(max);

    for i in 0..n {
        diag[i] = diag[i] * max_inv;
        subdiag[i] = subdiag[i] * max_inv;
    }

    let k = n / 2;
    let rem = n - k - 1;

    let (alpha, beta) = {
        let (d1, d2) = diag.rb_mut().split_at_row_mut(k);
        let (subd1, subd2) = subdiag.rb_mut().split_at_row_mut(k);
        let (alpha, d2) = d2.split_at_row_mut(1);
        let (beta, subd2) = subd2.split_at_row_mut(1);

        let mut d1 = d1;
        let mut d2 = d2;
        let mut subd1 = subd1;
        let mut subd2 = subd2;

        let alpha = copy(alpha[0]);
        let beta = copy(beta[0]);

        let (mut u1, mut u2) = match u.rb_mut() {
            MatU::Full(u) => {
                let (u1, u2) = u.split_at_row_mut(k + 1);

                (
                    MatU::Full(u1.submatrix_mut(0, 1, k + 1, k + 1)),
                    MatU::Full(u2.submatrix_mut(0, k + 1, rem + 1, rem + 1)),
                )
            }
            MatU::TwoRows(u) | MatU::TwoRowsStorage(u) => {
                let (u1, u2) = u.split_at_col_mut(k + 1);
                (MatU::TwoRows(u1), MatU::TwoRows(u2))
            }
        };

        let (mut v1, mut v2) = match v.rb_mut() {
            Some(v) => {
                let (v1, v2) = v.split_at_row_mut(k);
                let v2 = v2.split_at_row_mut(1).1;
                (Some(v1.subcols_mut(1, k)), Some(v2.subcols_mut(k + 1, rem)))
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
                for i in (0..k).rev() {
                    with_dim!(ncols, u1.ncols());

                    swap_cols_idx(
                        u1.rb_mut().as_col_shape_mut(ncols),
                        ncols.idx(i),
                        ncols.idx(i + 1),
                    );
                }
            }
            _ => {}
        }

        (alpha, beta)
    };
    if let Some(mut v) = v.rb_mut() {
        v[(k, 0)] = one();
    }

    for i in (0..k).rev() {
        diag[i + 1] = copy(diag[i]);
    }

    let lambda = match u.rb_mut() {
        MatU::Full(u) => copy(u[(k, k + 1)]),
        MatU::TwoRows(u) | MatU::TwoRowsStorage(u) => copy(u[(1, 0)]),
    };
    let phi = match u.rb_mut() {
        MatU::Full(u) => copy(u[(k + 1, n)]),
        MatU::TwoRows(u) | MatU::TwoRowsStorage(u) => copy(u[(0, n)]),
    };

    let al = alpha * lambda;
    let bp = beta * phi;
    let r0 = sqrt(abs2(al) + abs2(bp));
    let (c0, s0) = if r0 == zero() {
        (one(), zero())
    } else {
        (al / r0, bp / r0)
    };

    let mut col0 = subdiag;

    diag[0] = copy(r0);
    col0[0] = copy(r0);

    match u.rb_mut() {
        MatU::Full(u) => {
            let (u1, u2) = u.split_at_row_mut(k + 1);

            let (mut u0_top, u1) = u1.split_at_col_mut(1);
            let (u1, mut un_top) = u1.split_at_col_mut(n - 1);
            let (mut u0_bot, u2) = u2.split_at_col_mut(1);
            let (u2, mut un_bot) = u2.split_at_col_mut(n - 1);

            for j in 1..k + 1 {
                col0[j] = alpha * u1[(k, j - 1)];
            }

            for j in k + 1..n {
                col0[j] = beta * u2[(0, j - 1)];
            }

            z!(
                u0_top.rb_mut().col_mut(0),
                un_top.rb_mut().col_mut(0),
                u1.col_mut(k)
            )
            .for_each(|uz!(x0, xn, a)| {
                *x0 = c0 * a;
                *xn = -s0 * a;
                if cfg!(debug_assertions) {
                    *a = zero();
                }
            });

            z!(u0_bot.rb_mut().col_mut(0), un_bot.rb_mut().col_mut(0)).for_each(|uz!(x0, xn)| {
                *x0 = s0 * xn;
                *xn = c0 * xn;
            });
        }
        MatU::TwoRows(mut u) => {
            for j in 1..k + 1 {
                col0[j] = alpha * u[(1, j)];
                u[(1, j)] = zero();
            }

            for j in k + 1..n {
                col0[j] = beta * u[(0, j)];
                u[(0, j)] = zero();
            }

            let q10 = copy(u[(0, 0)]);
            let q21 = copy(u[(1, n)]);

            u[(0, 0)] = c0 * q10;
            u[(0, n)] = -s0 * q10;
            u[(1, 0)] = s0 * q21;
            u[(1, n)] = c0 * q21;
        }
        MatU::TwoRowsStorage(u) => {
            for j in 1..k + 1 {
                col0[j] = alpha * u[(1, j)];
            }

            for j in k + 1..n {
                col0[j] = beta * u[(0, j)];
            }
        }
    }

    let (mut perm, stack) = stack.collect(0..n);
    let perm = &mut *perm;

    let (mut jacobi_coeff, stack) = stack.make_with(n, |_| JacobiRotation::<T> {
        c: zero(),
        s: zero(),
    });
    let jacobi_coeff = &mut *jacobi_coeff;
    let (mut jacobi_idx, stack) = stack.collect(0..n);
    let jacobi_idx = &mut *jacobi_idx;

    let (jacobi_0i, jacobi_ij) = {
        let (mut transpositions, stack) = stack.collect(0..n);
        deflate(
            diag.rb_mut(),
            col0.rb_mut(),
            jacobi_coeff,
            jacobi_idx,
            &mut transpositions,
            perm,
            k,
            stack,
        )
    };

    let allocate_vm = v.is_some() as usize;
    let (mut um, stack) = temp_mat_zeroed(n + 1, n + 1, stack);
    let (mut vm, stack) = temp_mat_zeroed(n, allocate_vm * n, stack);
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
        Some(um.rb_mut()),
        vm.rb_mut(),
        diag.rb_mut(),
        col0.rb(),
        perm,
        stack,
    );
    for i in 0..n {
        col0[i] = zero();
    }

    for (rot, &i) in core::iter::zip(
        &jacobi_coeff[jacobi_0i..][..jacobi_ij],
        &jacobi_idx[jacobi_0i..][..jacobi_ij],
    )
    .rev()
    {
        let (i, j) = (i - 1, i);

        let actual_i = perm[i];
        let actual_j = perm[j];

        {
            let (i, j) = um
                .rb_mut()
                .subcols_mut(0, n)
                .two_rows_mut(actual_i, actual_j);
            rot.apply_on_the_left_in_place((j, i));
        }
        if let Some(mut vm) = vm.rb_mut() {
            let (i, j) = vm.rb_mut().two_rows_mut(actual_i, actual_j);
            rot.apply_on_the_left_in_place((j, i));
        }
    }

    for (rot, &i) in core::iter::zip(&jacobi_coeff[..jacobi_0i], &jacobi_idx[..jacobi_0i]).rev() {
        let (i, j) = um.rb_mut().subcols_mut(0, n).two_rows_mut(0, i);
        rot.apply_on_the_left_in_place((j, i));
    }

    let _v_is_none = v.is_none();

    let update_v = |mut v: Option<MatMut<'_, T>>, par: Par, stack: &mut DynStack| {
        let (mut combined_v, _) = unsafe { temp_mat_uninit(n, allocate_vm * n, stack) };
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
            let (mut combined_v1, combined_v2) = combined_v.rb_mut().split_at_row_mut(k);
            let mut combined_v2 = combined_v2.submatrix_mut(1, 0, rem, n);

            let v_lhs = v.rb();
            let v_lhs1 = v_lhs.submatrix(0, 1, k, k);
            let v_lhs2 = v_lhs.submatrix(k + 1, k + 1, rem, rem);
            let (v_rhs1, v_rhs2) = v_rhs.split_at_row(1).1.split_at_row(k);

            crate::utils::thread::join_raw(
                |par| {
                    crate::linalg::matmul::matmul(
                        combined_v1.rb_mut(),
                        Accum::Replace,
                        v_lhs1,
                        v_rhs1,
                        one(),
                        par,
                    )
                },
                |par| {
                    crate::linalg::matmul::matmul(
                        combined_v2.rb_mut(),
                        Accum::Replace,
                        v_lhs2,
                        v_rhs2,
                        one(),
                        par,
                    )
                },
                par,
            );

            crate::linalg::matmul::matmul(
                combined_v.rb_mut().submatrix_mut(k, 0, 1, n),
                Accum::Replace,
                v_lhs.submatrix(k, 0, 1, 1),
                v_rhs.submatrix(0, 0, 1, n),
                one(),
                par,
            );

            v.copy_from(combined_v.rb());
        }
    };

    let update_u = |mut u: MatMut<'_, T>, par: Par, stack: &mut DynStack| {
        let n = n;
        let k = k;
        let rem = rem;

        let (mut combined_u, _) = unsafe { temp_mat_uninit(n + 1, n + 1, stack) };
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
                    combined_u1.rb_mut(),
                    Accum::Replace,
                    u_lhs1,
                    u_rhs1,
                    one(),
                    par,
                );
                // rank 1 update
                crate::linalg::matmul::matmul(
                    combined_u1.rb_mut(),
                    Accum::Add,
                    u_lhs.col(n).subrows(0, k + 1).as_mat(),
                    u_rhs2.row(rem).as_mat(),
                    one(),
                    par,
                );
            },
            |par| {
                // matrix matrix
                crate::linalg::matmul::matmul(
                    combined_u2.rb_mut(),
                    Accum::Replace,
                    u_lhs2,
                    u_rhs2,
                    one(),
                    par,
                );
                // rank 1 update
                crate::linalg::matmul::matmul(
                    combined_u2.rb_mut(),
                    Accum::Add,
                    u_lhs.col(0).subrows(k + 1, rem + 1).as_mat(),
                    u_rhs1.row(0).as_mat(),
                    one(),
                    par,
                );
            },
            par,
        );

        u.copy_from(combined_u.rb());
    };

    match u.rb_mut() {
        MatU::TwoRowsStorage(_) => {
            update_v(v.rb_mut(), par, stack);
        }
        MatU::TwoRows(mut u) => {
            update_v(v.rb_mut(), par, stack);

            let (mut combined_u, _) = unsafe { temp_mat_uninit(2, n + 1, stack) };
            let mut combined_u = combined_u.as_mat_mut();
            crate::linalg::matmul::matmul(
                combined_u.rb_mut(),
                Accum::Replace,
                u.rb(),
                um.rb(),
                one(),
                par,
            );

            u.copy_from(combined_u.rb());
        }
        MatU::Full(u) => match par {
            #[cfg(feature = "rayon")]
            Par::Rayon(_) if !_v_is_none => {
                let req_v = crate::linalg::temp_mat_scratch::<T>(n, n).unwrap();
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

    for i in 0..n {
        diag[i] = diag[i] * max;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use dyn_stack::GlobalMemBuffer;

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
            let n = diag.nrows();

            if n > 512 {
                continue;
            }

            let diag = diag.as_ref().try_as_col_major().unwrap();
            let subdiag = subdiag.as_ref().try_as_col_major().unwrap();

            let mut d = diag.to_owned();
            let mut subd = subdiag.to_owned();

            let mut u = Mat::zeros(n, n);
            let mut v = Mat::zeros(n, n);

            for i in 0..n {
                u[(i, i)] = 1.0;
                v[(i, i)] = 1.0;
            }

            qr_algorithm(
                d.as_mut().try_as_col_major_mut().unwrap(),
                subd.as_mut().try_as_col_major_mut().unwrap(),
                Some(u.as_mut()),
                Some(v.as_mut()),
            )
            .unwrap();

            for &x in subd.iter() {
                assert!(x == 0.0);
            }

            let mut approx_eq = CwiseMat(ApproxEq::<f64>::eps());
            approx_eq.0.abs_tol *=
                f64::max(diag.norm_max(), subdiag.norm_max()) * (n as f64).sqrt();
            approx_eq.0.rel_tol *=
                f64::max(diag.norm_max(), subdiag.norm_max()) * (n as f64).sqrt();
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
            let n = diag.nrows();

            if n > 1024 {
                continue;
            }

            let diag = diag.as_ref().try_as_col_major().unwrap();
            let subdiag = subdiag.as_ref().try_as_col_major().unwrap();

            let mut d = diag.to_owned();
            let mut subd = subdiag.to_owned();

            let mut u = Mat::zeros(n + 1, n + 1);
            let mut v = Mat::zeros(n, n);

            for i in 0..n {
                u[(i, i)] = 1.0;
                v[(i, i)] = 1.0;
            }
            u[(n, n)] = 1.0;

            divide_and_conquer(
                d.as_mut().try_as_col_major_mut().unwrap(),
                subd.as_mut().try_as_col_major_mut().unwrap(),
                MatU::Full(u.as_mut()),
                Some(v.as_mut()),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    divide_and_conquer_scratch::<f64>(n, 4, true, true, Par::Seq).unwrap(),
                )),
                4,
            )
            .unwrap();

            for x in subd.iter_mut() {
                *x = 0.0;
            }

            let mut d2 = Mat::zeros(n + 1, n);
            for i in 0..n {
                d2[(i, i)] = d[i];
            }

            let mut approx_eq = CwiseMat(ApproxEq::<f64>::eps());
            approx_eq.0.abs_tol *=
                f64::max(diag.norm_max(), subdiag.norm_max()) * (n as f64).sqrt() * 10.0;
            approx_eq.0.rel_tol *=
                f64::max(diag.norm_max(), subdiag.norm_max()) * (n as f64).sqrt() * 10.0;
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

            let n = diag.nrows();

            if n <= 1024 {
                continue;
            }

            let diag = diag.as_ref().try_as_col_major().unwrap();
            let subdiag = subdiag.as_ref().try_as_col_major().unwrap();

            let mut d = diag.to_owned();
            let mut subd = subdiag.to_owned();

            let mut u = Mat::zeros(n + 1, n + 1);
            let mut v = Mat::zeros(n, n);

            for i in 0..n {
                u[(i, i)] = 1.0;
                v[(i, i)] = 1.0;
            }
            u[(n, n)] = 1.0;

            divide_and_conquer(
                d.as_mut().try_as_col_major_mut().unwrap(),
                subd.as_mut().try_as_col_major_mut().unwrap(),
                MatU::Full(u.as_mut()),
                Some(v.as_mut()),
                Par::Seq,
                DynStack::new(&mut GlobalMemBuffer::new(
                    divide_and_conquer_scratch::<f32>(n, 4, true, true, Par::Seq).unwrap(),
                )),
                4,
            )
            .unwrap();

            for x in subd.iter_mut() {
                *x = 0.0;
            }

            let mut d2 = Mat::zeros(n + 1, n);
            for i in 0..n {
                d2[(i, i)] = d[i];
            }

            let mut approx_eq = CwiseMat(ApproxEq::<f32>::eps());
            approx_eq.0.abs_tol *=
                f32::max(diag.norm_max(), subdiag.norm_max()) * (n as f32).sqrt() * 10.0;
            approx_eq.0.rel_tol *=
                f32::max(diag.norm_max(), subdiag.norm_max()) * (n as f32).sqrt() * 10.0;
            let reconstructed = &u * &d2 * v.adjoint();

            assert!(reconstructed ~ bidiag_to_mat2(diag, subdiag));
        }
    }

    #[test]
    fn test_deflation43() {
        let approx_eq = CwiseMat(ApproxEq::<f32>::eps());

        let n = 7;
        let k = 3;
        let diag_orig = &*vec![1.0, 5.0, 3.0, 1e-7, 4.0, 2.0, 2e-7_f32];
        let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];

        let n_jacobi = 2;
        let jacobi_coeffs = &mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
        let jacobi_indices = &mut *vec![0; n_jacobi];
        let perm = &mut *vec![0; n];
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
            diag.rb_mut(),
            col.rb_mut(),
            jacobi_coeffs,
            jacobi_indices,
            &mut *vec![0; n],
            perm,
            k,
            DynStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * n]),
        );
        assert!(all(jacobi_0i == 2, jacobi_ij == 0));

        let perm_inv = &mut *vec![0; n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[p] = i;
        }

        let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);

        let mut M_orig = Mat::zeros(n, n);
        for i in 0..n {
            M_orig[(i, i)] = diag_orig[i];
            M_orig[(i, 0)] = col_orig[i];
        }

        let mut M = Mat::zeros(n, n);
        for i in 0..n {
            M[(i, i)] = diag[i];
            M[(i, 0)] = col[i];
        }

        M = P.inverse() * M * P;

        for (&rot, &i) in core::iter::zip(&*jacobi_coeffs, &*jacobi_indices).rev() {
            let (x, y) = M.two_rows_mut(0, i);
            rot.apply_on_the_left_in_place((y, x));
        }

        assert!(M ~ M_orig);
    }

    #[test]
    fn test_deflation44() {
        let approx_eq = CwiseMat(ApproxEq::<f32>::eps());

        let n = 7;
        let k = 3;
        let diag_orig = &*vec![1.0, 5.0, 3.0, 1.0, 4.0, 2.0, 1.0];
        let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];

        let n_jacobi = 1;
        let jacobi_coeffs = &mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
        let jacobi_indices = &mut *vec![0; n_jacobi];
        let perm = &mut *vec![0; n];
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
            diag.rb_mut(),
            col.rb_mut(),
            jacobi_coeffs,
            jacobi_indices,
            &mut *vec![0; n],
            perm,
            k,
            DynStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * n]),
        );
        assert!(all(jacobi_0i == 0, jacobi_ij == 1));

        let perm_inv = &mut *vec![0; n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[p] = i;
        }

        let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);

        let mut M_orig = Mat::zeros(n, n);
        for i in 0..n {
            M_orig[(i, i)] = diag_orig[i];
            M_orig[(i, 0)] = col_orig[i];
        }

        let mut M = Mat::zeros(n, n);
        for i in 0..n {
            M[(i, i)] = diag[i];
            M[(i, 0)] = col[i];
        }

        M = P.inverse() * M * P;

        for (&rot, &i) in core::iter::zip(&*jacobi_coeffs, &*jacobi_indices).rev() {
            let (i, j) = (i - 1, i);
            let (pi, pj) = (perm[i], perm[j]);

            let (x, y) = M.two_rows_mut(pi, pj);
            rot.apply_on_the_left_in_place((y, x));

            let (x, y) = M.two_cols_mut(pi, pj);
            rot.adjoint().apply_on_the_right_in_place((y, x));
        }

        assert!(M ~ M_orig);
    }

    #[test]
    fn test_both_deflation() {
        let approx_eq = CwiseMat(ApproxEq::<f32>::eps());

        let n = 7;
        let k = 3;
        let diag_orig = &*vec![1.0, 5.0, 3.0, 2.0, 4.0, 2.0, 0.0];
        let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];

        let n_jacobi = 2;
        let jacobi_coeffs = &mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
        let jacobi_indices = &mut *vec![0; n_jacobi];
        let perm = &mut *vec![0; n];
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
            diag.rb_mut(),
            col.rb_mut(),
            jacobi_coeffs,
            jacobi_indices,
            &mut *vec![0; n],
            perm,
            k,
            DynStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * n]),
        );
        assert!(all(jacobi_0i == 1, jacobi_ij == 1));

        let perm_inv = &mut *vec![0; n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[p] = i;
        }

        let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);

        let mut M_orig = Mat::zeros(n, n);
        for i in 0..n {
            M_orig[(i, i)] = diag_orig[i];
            M_orig[(i, 0)] = col_orig[i];
        }

        let mut M = Mat::zeros(n, n);
        for i in 0..n {
            M[(i, i)] = diag[i];
            M[(i, 0)] = col[i];
        }

        M = P.inverse() * M * P;

        for (&rot, &i) in core::iter::zip(&jacobi_coeffs[1..], &jacobi_indices[1..]).rev() {
            let (i, j) = (i - 1, i);
            let (pi, pj) = (perm[i], perm[j]);

            let (x, y) = M.two_rows_mut(pi, pj);
            rot.apply_on_the_left_in_place((y, x));

            let (x, y) = M.two_cols_mut(pi, pj);
            rot.adjoint().apply_on_the_right_in_place((y, x));
        }

        for (&rot, &i) in core::iter::zip(&jacobi_coeffs[..1], &jacobi_indices[..1]).rev() {
            let (x, y) = M.two_rows_mut(0, i);
            rot.apply_on_the_left_in_place((y, x));
        }

        assert!(M ~ M_orig);
    }
}
