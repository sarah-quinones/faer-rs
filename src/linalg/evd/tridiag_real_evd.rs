use crate::{
    debug_assert,
    linalg::{
        matmul::{inner_prod::inner_prod_with_conj, matmul},
        temp_mat_req, temp_mat_uninit,
    },
    unzipped, zipped, ColMut, ColRef, Conj, Entity, MatMut, Parallelism, RealField,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use reborrow::*;

fn max<T: PartialOrd>(x: T, y: T) -> T {
    if x > y {
        x
    } else {
        y
    }
}

/// the argument of the secular eq is mu + shift
#[inline(always)]
fn secular_eq<E: RealField>(
    d: ColRef<'_, E>,
    z: ColRef<'_, E>,
    rho: E,
    mu: E,
    shift: E,
    n: usize,
) -> E {
    assert!(d.row_stride() == 1);
    assert!(z.row_stride() == 1);
    debug_assert!(d.nrows() == z.nrows());
    debug_assert!(d.nrows() >= n);
    debug_assert!(z.nrows() >= n);

    let mut res0 = rho.faer_inv();
    let mut res1 = E::faer_zero();
    let mut res2 = E::faer_zero();
    let mut res3 = E::faer_zero();
    let mut res4 = E::faer_zero();
    let mut res5 = E::faer_zero();
    let mut res6 = E::faer_zero();
    let mut res7 = E::faer_zero();

    unsafe {
        let mut i = 0;

        while i < n / 8 * 8 {
            let z0 = z.read_unchecked(i);
            let d0 = d.read_unchecked(i);
            let z1 = z.read_unchecked(i + 1);
            let d1 = d.read_unchecked(i + 1);
            let z2 = z.read_unchecked(i + 2);
            let d2 = d.read_unchecked(i + 2);
            let z3 = z.read_unchecked(i + 3);
            let d3 = d.read_unchecked(i + 3);
            let z4 = z.read_unchecked(i + 4);
            let d4 = d.read_unchecked(i + 4);
            let z5 = z.read_unchecked(i + 5);
            let d5 = d.read_unchecked(i + 5);
            let z6 = z.read_unchecked(i + 6);
            let d6 = d.read_unchecked(i + 6);
            let z7 = z.read_unchecked(i + 7);
            let d7 = d.read_unchecked(i + 7);

            // res = res + z * (z / (d - arg))
            res0 = res0.faer_add(z0.faer_mul(z0.faer_div(d0.faer_sub(shift).faer_sub(mu))));
            res1 = res1.faer_add(z1.faer_mul(z1.faer_div(d1.faer_sub(shift).faer_sub(mu))));
            res2 = res2.faer_add(z2.faer_mul(z2.faer_div(d2.faer_sub(shift).faer_sub(mu))));
            res3 = res3.faer_add(z3.faer_mul(z3.faer_div(d3.faer_sub(shift).faer_sub(mu))));
            res4 = res4.faer_add(z4.faer_mul(z4.faer_div(d4.faer_sub(shift).faer_sub(mu))));
            res5 = res5.faer_add(z5.faer_mul(z5.faer_div(d5.faer_sub(shift).faer_sub(mu))));
            res6 = res6.faer_add(z6.faer_mul(z6.faer_div(d6.faer_sub(shift).faer_sub(mu))));
            res7 = res7.faer_add(z7.faer_mul(z7.faer_div(d7.faer_sub(shift).faer_sub(mu))));

            i += 8;
        }

        while i < n {
            let z0 = z.read_unchecked(i);
            let d0 = d.read_unchecked(i);

            // res = res + z * (z / (d - arg))
            res0 = res0.faer_add(z0.faer_mul(z0.faer_div(d0.faer_sub(shift).faer_sub(mu))));

            i += 1;
        }
    }

    E::faer_add(
        E::faer_add(E::faer_add(res0, res1), E::faer_add(res2, res3)),
        E::faer_add(E::faer_add(res4, res5), E::faer_add(res6, res7)),
    )
}

fn compute_eigenvalues<E: RealField>(
    mut mus: ColMut<'_, E>,
    mut shifts: ColMut<'_, E>,
    d: ColRef<'_, E>,
    z: ColRef<'_, E>,
    rho: E,
    epsilon: E,
    non_deflated: usize,
    consider_zero_threshold: E,
) {
    debug_assert!(d.row_stride() == 1);
    debug_assert!(z.row_stride() == 1);
    debug_assert!(d.nrows() == z.nrows());

    let n = non_deflated;
    let full_n = d.nrows();

    let one_half = E::faer_from_f64(0.5);
    let two = E::faer_from_f64(2.0);
    let eight = E::faer_from_f64(8.0);

    'kth_iter: for i in 0..n {
        // left = d_i
        let left = d.read(i);

        let last_i = i == n - 1;

        let right = if last_i {
            // right = d_i + rho z.T z
            d.read(i).faer_add(rho.faer_mul(z.squared_norm_l2()))
        } else {
            // right = d_{i+1}
            d.read(i + 1)
        };

        // find the root between left and right
        let mid_left_shift = right.faer_sub(left).faer_scale_power_of_two(one_half);
        let f_mid_left_shift = secular_eq(d, z, rho, mid_left_shift, left, n);
        let mid_right_shift = mid_left_shift.faer_neg();
        let f_mid_right_shift = secular_eq(d, z, rho, mid_right_shift, right, n);

        let (shift, mu) = if last_i || f_mid_left_shift > E::faer_zero() {
            (left, mid_left_shift)
        } else {
            (right, mid_right_shift)
        };

        if f_mid_left_shift <= E::faer_zero() && f_mid_right_shift >= E::faer_zero() {
            shifts.write(i, shift);
            mus.write(i, mu);
            continue 'kth_iter;
        }

        enum SecantError {
            OutOfBounds,
            PrecisionLimitReached,
        }

        let secant = {
            #[inline(always)]
            |mut mu_cur: E, mut mu_prev: E, mut f_cur: E, mut f_prev: E| {
                if f_prev.faer_abs() < f_cur.faer_abs() {
                    core::mem::swap(&mut f_prev, &mut f_cur);
                    core::mem::swap(&mut mu_prev, &mut mu_cur);
                }

                let mut left_candidate = None;
                let mut right_candidate = None;

                let mut use_bisection = false;
                let same_sign = f_prev.faer_mul(f_cur) > E::faer_zero();
                if !same_sign {
                    let (min, max) = if mu_cur < mu_prev {
                        (mu_cur, mu_prev)
                    } else {
                        (mu_prev, mu_cur)
                    };
                    left_candidate = Some(min);
                    right_candidate = Some(max);
                }

                let mut err = SecantError::PrecisionLimitReached;

                while f_cur != E::faer_zero()
                    && ((mu_cur.faer_sub(mu_prev)).faer_abs()
                        > eight.faer_mul(epsilon).faer_mul(
                            if mu_cur.faer_abs() > mu_prev.faer_abs() {
                                mu_cur.faer_abs()
                            } else {
                                mu_prev.faer_abs()
                            },
                        ))
                    && ((f_cur.faer_sub(f_prev)).faer_abs() > epsilon)
                    && !use_bisection
                {
                    // rational interpolation: fit a function of the form a / mu + b through
                    // the two previous iterates and use its
                    // zero to compute the next iterate
                    let a = (f_cur.faer_sub(f_prev))
                        .faer_mul(mu_prev.faer_mul(mu_cur))
                        .faer_div(mu_prev.faer_sub(mu_cur));
                    let b = f_cur.faer_sub(a.faer_div(mu_cur));
                    let mu_zero = a.faer_div(b).faer_neg();
                    let f_zero = secular_eq(d, z, rho, mu_zero, shift, n);

                    if f_zero < E::faer_zero() {
                        left_candidate = Some(mu_zero);
                    } else {
                        right_candidate = Some(mu_zero);
                    }

                    mu_prev = mu_cur;
                    f_prev = f_cur;
                    mu_cur = mu_zero;
                    f_cur = f_zero;

                    if shift == left && (mu_cur < E::faer_zero() || mu_cur > (right.faer_sub(left)))
                    {
                        err = SecantError::OutOfBounds;
                        use_bisection = true;
                    }
                    if shift == right
                        && (mu_cur < (right.faer_sub(left)).faer_neg() || mu_cur > E::faer_zero())
                    {
                        err = SecantError::OutOfBounds;
                        use_bisection = true;
                    }
                    if f_cur.faer_abs() > f_prev.faer_abs() {
                        // find mu such that a / mu + b = -k * f_zero
                        // a / mu = -f_zero - b
                        // mu = -a / (f_zero + b)
                        let mut k = E::faer_one();
                        for _ in 0..4 {
                            let mu_opposite = a.faer_neg().faer_div(k.faer_mul(f_zero).faer_add(b));
                            let f_opposite = secular_eq(d, z, rho, mu_opposite, shift, n);
                            if f_zero < E::faer_zero() && f_opposite >= E::faer_zero() {
                                // this will be our right candidate
                                right_candidate = Some(mu_opposite);
                                break;
                            } else if f_zero > E::faer_zero() && f_opposite <= E::faer_zero() {
                                // this will be our left candidate
                                left_candidate = Some(mu_opposite);
                                break;
                            }
                            k = k.faer_scale_power_of_two(two);
                        }
                        use_bisection = true;
                    }
                }
                (use_bisection, mu_cur, left_candidate, right_candidate, err)
            }
        };

        let (mut left_shifted, mut f_left, mut right_shifted, mut f_right) = if shift == left {
            let (right_shifted, f_right) = if last_i {
                (
                    right.faer_sub(left),
                    secular_eq(d, z, rho, right.faer_sub(left), shift, n),
                )
            } else {
                (
                    right.faer_sub(left).faer_scale_power_of_two(one_half),
                    f_mid_left_shift,
                )
            };
            (
                E::faer_zero(),
                E::faer_zero().faer_inv().faer_neg(),
                right_shifted,
                f_right,
            )
        } else {
            (
                right
                    .faer_sub(left)
                    .faer_neg()
                    .faer_scale_power_of_two(one_half),
                f_mid_left_shift,
                E::faer_zero(),
                E::faer_zero().faer_inv(),
            )
        };

        let mut iteration_count = 0;
        let mut f_prev = f_mid_left_shift;

        let half0 = one_half;
        let half1 = half0.faer_scale_power_of_two(half0);
        let half2 = half1.faer_scale_power_of_two(half1);
        let half3 = half2.faer_scale_power_of_two(half2);
        let half4 = half3.faer_scale_power_of_two(half3);
        let half5 = half4.faer_scale_power_of_two(half4);
        let half6 = half5.faer_scale_power_of_two(half5);
        let half7 = half6.faer_scale_power_of_two(half6);

        let mu_values = if shift == left {
            [
                right_shifted.faer_scale_power_of_two(half7),
                right_shifted.faer_scale_power_of_two(half6),
                right_shifted.faer_scale_power_of_two(half5),
                right_shifted.faer_scale_power_of_two(half4),
                right_shifted.faer_scale_power_of_two(half3),
                right_shifted.faer_scale_power_of_two(half2),
                right_shifted.faer_scale_power_of_two(half1),
                right_shifted.faer_scale_power_of_two(half0),
            ]
        } else {
            [
                left_shifted.faer_scale_power_of_two(half7),
                left_shifted.faer_scale_power_of_two(half6),
                left_shifted.faer_scale_power_of_two(half5),
                left_shifted.faer_scale_power_of_two(half4),
                left_shifted.faer_scale_power_of_two(half3),
                left_shifted.faer_scale_power_of_two(half2),
                left_shifted.faer_scale_power_of_two(half1),
                left_shifted.faer_scale_power_of_two(half0),
            ]
        };

        let f_values = [
            secular_eq(d, z, rho, mu_values[0], shift, n),
            secular_eq(d, z, rho, mu_values[1], shift, n),
            secular_eq(d, z, rho, mu_values[2], shift, n),
            secular_eq(d, z, rho, mu_values[3], shift, n),
            secular_eq(d, z, rho, mu_values[4], shift, n),
            secular_eq(d, z, rho, mu_values[5], shift, n),
            secular_eq(d, z, rho, mu_values[6], shift, n),
            secular_eq(d, z, rho, mu_values[7], shift, n),
        ];

        if shift == left {
            let mut i = 0;
            for (idx, (mu, f)) in core::iter::zip(mu_values, f_values).enumerate() {
                if f < E::faer_zero() {
                    left_shifted = mu;
                    f_left = f;
                    i = idx + 1;
                }
            }
            if i < f_values.len() {
                right_shifted = mu_values[i];
                f_right = f_values[i];
            }
        } else {
            let mut i = 0;
            for (idx, (mu, f)) in core::iter::zip(mu_values, f_values).enumerate() {
                if f > E::faer_zero() {
                    right_shifted = mu;
                    f_right = f;
                    i = idx + 1;
                }
            }
            if i < f_values.len() {
                left_shifted = mu_values[i];
                f_left = f_values[i];
            }
        }

        assert!(
            PartialOrd::partial_cmp(&f_left, &E::faer_zero()) != Some(core::cmp::Ordering::Greater)
        );
        assert!(
            PartialOrd::partial_cmp(&f_right, &E::faer_zero()) != Some(core::cmp::Ordering::Less)
        );

        // try bisection just to get a good guess for secant
        while right_shifted.faer_sub(left_shifted)
            > two.faer_mul(epsilon).faer_mul(
                if left_shifted.faer_abs() > right_shifted.faer_abs() {
                    left_shifted.faer_abs()
                } else {
                    right_shifted.faer_abs()
                },
            )
        {
            let mid_shifted_arithmetic =
                (left_shifted.faer_add(right_shifted)).faer_scale_power_of_two(one_half);
            let mut mid_shifted_geometric = left_shifted
                .faer_abs()
                .faer_sqrt()
                .faer_mul(right_shifted.faer_abs().faer_sqrt());
            if left_shifted < E::faer_zero() {
                mid_shifted_geometric = mid_shifted_geometric.faer_neg();
            }
            let mid_shifted = if mid_shifted_geometric == E::faer_zero() {
                mid_shifted_arithmetic
            } else {
                mid_shifted_geometric
            };
            let f_mid = secular_eq(d, z, rho, mid_shifted, shift, n);

            if f_mid == E::faer_zero() {
                shifts.write(i, shift);
                mus.write(i, mid_shifted);
                continue 'kth_iter;
            } else if f_mid > E::faer_zero() {
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
        let args = if left_shifted == E::faer_zero() {
            (
                right_shifted.faer_add(right_shifted),
                right_shifted,
                f_prev,
                f_right,
            )
        } else if right_shifted == E::faer_zero() {
            (
                left_shifted.faer_add(left_shifted),
                left_shifted,
                f_prev,
                f_left,
            )
        } else {
            (left_shifted, right_shifted, f_left, f_right)
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
            while right_shifted.faer_sub(left_shifted)
                > max(
                    two.faer_mul(epsilon)
                        .faer_mul(max(left_shifted.faer_abs(), right_shifted.faer_abs())),
                    consider_zero_threshold,
                )
            {
                assert!(right_shifted > left_shifted);
                let mid_shifted = left_shifted.faer_add(
                    right_shifted
                        .faer_sub(left_shifted)
                        .faer_scale_power_of_two(one_half),
                );
                let f_mid = secular_eq(d, z, rho, mid_shifted, shift, n);
                if f_mid == E::faer_zero() {
                    break;
                } else if f_mid > E::faer_zero() {
                    right_shifted = mid_shifted;
                } else {
                    left_shifted = mid_shifted;
                }

                mu_cur = left_shifted.faer_add(right_shifted).faer_mul(one_half);
            }
        }

        mus.write(i, mu_cur);
        shifts.write(i, shift);
    }
    for i in n..full_n {
        mus.write(i, E::faer_zero());
        shifts.write(i, d.read(i));
    }
}

pub fn compute_tridiag_real_evd<E: RealField>(
    diag: &mut [E],
    offdiag: &mut [E],
    u: MatMut<'_, E>,
    epsilon: E,
    consider_zero_threshold: E,
    parallelism: Parallelism,
    stack: &mut PodStack,
) {
    let n = diag.len();
    let (pl_before, stack) = stack.make_with(n, |_| 0usize);
    let (pl_after, stack) = stack.make_with(n, |_| 0usize);
    let (pr, stack) = stack.make_with(n, |_| 0usize);
    let (run_info, stack) = stack.make_with(n, |_| 0usize);
    let (z, stack) = temp_mat_uninit::<E>(n, 1, stack);
    let (permuted_diag, stack) = temp_mat_uninit::<E>(n, 1, stack);
    let (permuted_z, stack) = temp_mat_uninit::<E>(n, 1, stack);
    let (householder, stack) = temp_mat_uninit::<E>(n, 1, stack);
    let (mus, stack) = temp_mat_uninit::<E>(n, 1, stack);
    let (shifts, stack) = temp_mat_uninit::<E>(n, 1, stack);
    let (repaired_u, stack) = temp_mat_uninit::<E>(n, n, stack);
    let (tmp, _) = temp_mat_uninit::<E>(n, n, stack);

    compute_tridiag_real_evd_impl(
        diag,
        offdiag,
        u,
        epsilon,
        consider_zero_threshold,
        parallelism,
        pl_before,
        pl_after,
        pr,
        run_info,
        z.col_mut(0),
        permuted_diag.col_mut(0),
        permuted_z.col_mut(0),
        householder.col_mut(0),
        mus.col_mut(0),
        shifts.col_mut(0),
        repaired_u,
        tmp,
    );
}

fn compute_tridiag_real_evd_impl<E: RealField>(
    diag: &mut [E],
    offdiag: &mut [E],
    mut u: MatMut<'_, E>,
    epsilon: E,
    consider_zero_threshold: E,
    parallelism: Parallelism,
    pl_before: &mut [usize],
    pl_after: &mut [usize],
    pr: &mut [usize],
    run_info: &mut [usize],
    mut z: ColMut<'_, E>,
    mut permuted_diag: ColMut<'_, E>,
    mut permuted_z: ColMut<'_, E>,
    mut householder: ColMut<'_, E>,
    mut mus: ColMut<'_, E>,
    mut shifts: ColMut<'_, E>,
    mut repaired_u: MatMut<'_, E>,
    mut tmp: MatMut<'_, E>,
) {
    let n = diag.len();

    if n <= 1 {
        u.rb_mut()
            .diagonal_mut()
            .column_vector_mut()
            .fill(E::faer_one());
        return;
    }

    if n == 2 {
        let a = diag[0];
        let d = diag[1];
        let b = offdiag[0];

        let half = E::Real::faer_from_f64(0.5);

        let t0 = ((a.faer_sub(d)).faer_abs2().faer_add(
            b.faer_abs2()
                .faer_scale_power_of_two(E::Real::faer_from_f64(4.0)),
        ))
        .faer_sqrt()
        .faer_scale_power_of_two(half);
        let t1 = (a.faer_add(d)).faer_scale_power_of_two(half);

        let r0 = t1.faer_sub(t0);
        let r1 = t1.faer_add(t0);

        let tol = max(r1.faer_abs(), r0.faer_abs()).faer_mul(epsilon);

        if r1.faer_sub(r0) <= tol {
            u.write(0, 0, E::faer_one());
            u.write(1, 0, E::faer_zero());
            u.write(0, 1, E::faer_zero());
            u.write(1, 1, E::faer_one());
        } else if b.faer_abs() <= tol {
            if diag[0] < diag[1] {
                u.write(0, 0, E::faer_one());
                u.write(1, 0, E::faer_zero());
                u.write(0, 1, E::faer_zero());
                u.write(1, 1, E::faer_one());
            } else {
                u.write(0, 0, E::faer_zero());
                u.write(1, 0, E::faer_one());
                u.write(0, 1, E::faer_one());
                u.write(1, 1, E::faer_zero());
            }
        } else {
            let tau = (d.faer_sub(a)).faer_div(b).faer_scale_power_of_two(half);
            let mut t = tau
                .faer_abs()
                .faer_add((E::faer_one().faer_add(tau.faer_abs2())).faer_sqrt())
                .faer_inv();
            if tau < E::faer_zero() {
                t = t.faer_neg();
            }

            let c = (E::faer_one().faer_add(t.faer_abs2())).faer_sqrt();
            let s = c.faer_mul(t);

            let r = c.faer_abs2().faer_add(s.faer_abs2()).faer_sqrt();
            let c = c.faer_div(r);
            let s = s.faer_div(r);

            let r0_reconstructed = c.faer_mul(a).faer_sub(s.faer_mul(b)).faer_div(c);
            if r0.faer_sub(r0_reconstructed).faer_abs() < r1.faer_sub(r0_reconstructed) {
                u.write(0, 0, c);
                u.write(1, 0, s.faer_neg());
                u.write(0, 1, s);
                u.write(1, 1, c);
            } else {
                u.write(0, 1, c);
                u.write(1, 1, s.faer_neg());
                u.write(0, 0, s);
                u.write(1, 0, c);
            }
        }

        diag[0] = r0;
        diag[1] = r1;
        return;
    }
    if n <= 32 {
        super::tridiag_qr_algorithm::compute_tridiag_real_evd_qr_algorithm(
            diag,
            offdiag,
            Some(u),
            epsilon,
            consider_zero_threshold,
        );
        return;
    }

    let n1 = n / 2;
    let mut rho = offdiag[n1 - 1];

    let (diag0, diag1) = diag.split_at_mut(n1);
    let (offdiag0, mut offdiag1) = offdiag.split_at_mut(n1 - 1);
    offdiag1 = &mut offdiag1[1..];

    diag0[n1 - 1] = diag0[n1 - 1].faer_sub(rho.faer_abs());
    diag1[0] = diag1[0].faer_sub(rho.faer_abs());

    let (mut u0, _, _, mut u1) = u.rb_mut().split_at_mut(n1, n1);
    {
        let (pl_before0, pl_before1) = pl_before.split_at_mut(n1);
        let (pl_after0, pl_after1) = pl_after.split_at_mut(n1);
        let (pr0, pr1) = pr.split_at_mut(n1);
        let (run_info0, run_info1) = run_info.split_at_mut(n1);
        let (z0, z1) = z.rb_mut().split_at_mut(n1);
        let (permuted_diag0, permuted_diag1) = permuted_diag.rb_mut().split_at_mut(n1);
        let (permuted_z0, permuted_z1) = permuted_z.rb_mut().split_at_mut(n1);
        let (householder0, householder1) = householder.rb_mut().split_at_mut(n1);
        let (mus0, mus1) = mus.rb_mut().split_at_mut(n1);
        let (shift0, shift1) = shifts.rb_mut().split_at_mut(n1);
        let (repaired_u0, repaired_u1) = repaired_u.rb_mut().split_at_col_mut(n1);
        let (tmp0, tmp1) = tmp.rb_mut().split_at_col_mut(n1);

        crate::utils::thread::join_raw(
            |parallelism| {
                compute_tridiag_real_evd_impl(
                    diag0,
                    offdiag0,
                    u0.rb_mut(),
                    epsilon,
                    consider_zero_threshold,
                    parallelism,
                    pl_before0,
                    pl_after0,
                    pr0,
                    run_info0,
                    z0,
                    permuted_diag0,
                    permuted_z0,
                    householder0,
                    mus0,
                    shift0,
                    repaired_u0,
                    tmp0,
                )
            },
            |parallelism| {
                compute_tridiag_real_evd_impl(
                    diag1,
                    offdiag1,
                    u1.rb_mut(),
                    epsilon,
                    consider_zero_threshold,
                    parallelism,
                    pl_before1,
                    pl_after1,
                    pr1,
                    run_info1,
                    z1,
                    permuted_diag1,
                    permuted_z1,
                    householder1,
                    mus1,
                    shift1,
                    repaired_u1,
                    tmp1,
                )
            },
            parallelism,
        );
    }
    let mut repaired_u = repaired_u.subrows_mut(0, n);
    let mut tmp = tmp.subrows_mut(0, n);

    //     [Q0   0] ([D0   0]            ) [Q0.T     0]
    // T = [ 0  Q1]×([ 0  D1] + rho×z×z.T)×[   0  Q1.T]
    //
    // we compute the permutation Pl_before that sorts diag([D0 D1])
    // we apply householder transformations to segments of permuted_z that correspond to
    // consecutive almost-equal eigenvalues
    // we compute the permutation Pl_after that places the deflated eigenvalues at the end
    //
    // we compute the EVD of:
    // Pl_after × H × Pl_before × (diag([D0 D1]) + rho×z×z.T) × Pl_before.T × H.T × Pl_after.T =
    // Q×W×Q.T
    //
    // we sort the eigenvalues in W
    // Pr × W × Pr.T = E
    //
    // diag([D0 D1]) + rho×z×z.T = Pl_before^-1×H^-1×Pl_after^-1×Q×Pr × E × ...

    let (mut z0, mut z1) = z.rb_mut().split_at_mut(n1);
    z0.rb_mut().copy_from(u0.rb().row(n1 - 1).transpose());
    if rho < E::faer_zero() {
        zipped!(__rw, z1.rb_mut(), u1.rb().row(0).transpose())
            .for_each(|unzipped!(mut z, u)| z.write(u.read().faer_neg()));
    } else {
        z1.rb_mut().copy_from(u1.rb().row(0).transpose());
    }

    let inv_sqrt2 = E::faer_from_f64(2.0).faer_sqrt().faer_inv();
    zipped!(__rw, z.rb_mut()).for_each(|unzipped!(mut x)| x.write(x.read().faer_mul(inv_sqrt2)));

    rho = rho
        .faer_scale_power_of_two(E::faer_from_f64(2.0))
        .faer_abs();

    // merge two sorted diagonals
    {
        let mut i = 0usize;
        let mut j = 0usize;
        for p in &mut *pl_before {
            if i == n1 {
                *p = n1 + j;
                j += 1;
            } else if (j == n - n1) || (diag[i] < diag[n1 + j]) {
                *p = i;
                i += 1;
            } else {
                *p = n1 + j;
                j += 1;
            }
        }
    }

    // permuted_diag = Pl * diag * Pl.T
    // permuted_z = Pl * diag
    for (i, &pl_before) in pl_before.iter().enumerate() {
        permuted_diag.write(i, diag[pl_before]);
    }
    for (i, &pl_before) in pl_before.iter().enumerate() {
        permuted_z.write(i, z.read(pl_before));
    }

    let mut dmax = E::faer_zero();
    let mut zmax = E::faer_zero();

    zipped!(__rw, permuted_diag.rb()).for_each(|unzipped!(x)| {
        let x = x.read().faer_abs();
        if x > dmax {
            dmax = x
        }
    });
    zipped!(__rw, permuted_z.rb()).for_each(|unzipped!(x)| {
        let x = x.read().faer_abs();
        if x > zmax {
            zmax = x
        }
    });

    let tol = E::faer_from_f64(8.0)
        .faer_mul(epsilon)
        .faer_mul(if dmax > zmax { dmax } else { zmax });

    if rho.faer_mul(zmax) <= tol {
        // fill uninitialized values of u with zeros
        // apply Pl to u on the right
        // copy permuted_diag to diag
        // return

        let (mut tmp_tl, mut tmp_tr, mut tmp_bl, mut tmp_br) = tmp.rb_mut().split_at_mut(n1, n1);
        tmp_tl.rb_mut().copy_from(u0.rb());
        tmp_br.rb_mut().copy_from(u1.rb());
        tmp_tr.fill_zero();
        tmp_bl.fill_zero();

        for (j, &pl_before) in pl_before.iter().enumerate() {
            u.rb_mut().col_mut(j).copy_from(tmp.rb().col(pl_before));
        }

        for (j, diag) in diag.iter_mut().enumerate() {
            *diag = permuted_diag.read(j);
        }

        return;
    }

    for i in 0..n {
        let zi = permuted_z.read(i);
        if rho.faer_mul(zi).faer_abs() <= tol {
            permuted_z.write(i, E::faer_zero());
        }
    }

    let mut applied_householder = false;

    let mut idx = 0usize;
    while idx < n {
        let mut run_len = 1;

        let d_prev = permuted_diag.read(idx);

        while idx + run_len < n {
            let d_next = permuted_diag.read(idx + run_len);
            if d_next.faer_sub(d_prev) < tol {
                permuted_diag.write(idx + run_len, d_prev);
                run_len += 1;
            } else {
                break;
            }
        }

        run_info[idx..][..run_len].fill(run_len);

        if run_len > 1 {
            applied_householder = true;

            let mut householder = householder.rb_mut().subrows_mut(idx, run_len);
            let mut permuted_z = permuted_z.rb_mut().subrows_mut(idx, run_len);

            householder.rb_mut().copy_from(permuted_z.rb());

            let head = householder.read(run_len - 1);
            let tail_norm = householder.rb().subrows(0, run_len - 1).norm_l2();

            let (tau, beta) = crate::linalg::householder::make_householder_in_place(
                Some(
                    householder
                        .rb_mut()
                        .subrows_mut(0, run_len - 1)
                        .reverse_rows_mut(),
                ),
                head,
                tail_norm,
            );

            householder.write(run_len - 1, tau);
            permuted_z.fill_zero();
            permuted_z.write(run_len - 1, beta);
        }

        idx += run_len;
    }

    // move deflated eigenvalues to the end
    let mut writer_deflated = 0;
    let mut writer_non_deflated = 0;
    for reader in 0..n {
        let z = permuted_z.read(reader);
        let d = permuted_diag.read(reader);

        if z == E::faer_zero() {
            // deflated value, store in diag
            diag[writer_deflated] = d;
            pr[writer_deflated] = reader;
            writer_deflated += 1;
        } else {
            permuted_z.write(writer_non_deflated, z);
            permuted_diag.write(writer_non_deflated, d);
            pl_after[writer_non_deflated] = reader;
            writer_non_deflated += 1;
        }
    }

    let non_deflated = writer_non_deflated;
    let deflated = writer_deflated;

    for i in 0..deflated {
        permuted_diag.write(non_deflated + i, diag[i]);
        pl_after[non_deflated + i] = pr[i];
    }

    // compute eigenvalues
    let mut mus = mus.subrows_mut(0, n);
    let mut shifts = shifts.subrows_mut(0, n);

    compute_eigenvalues(
        mus.rb_mut(),
        shifts.rb_mut(),
        permuted_diag.rb(),
        permuted_z.rb(),
        rho,
        epsilon,
        non_deflated,
        consider_zero_threshold,
    );

    // perturb z and rho
    // we don't actually need rho for computing the eigenvectors so we're not going to perturb it

    // new_zi^2 = prod(wk - di) / prod_{k != i} (dk - di)
    for i in 0..non_deflated {
        let di = permuted_diag.read(i);
        let mu_i = mus.read(i);
        let shift_i = shifts.read(i);
        let mut prod = mu_i.faer_add(shift_i.faer_sub(di));

        (0..i).chain(i + 1..non_deflated).for_each(|k| {
            let dk = permuted_diag.read(k);
            let mu_k = mus.read(k);
            let shift_k = shifts.read(k);

            let numerator = mu_k.faer_add(shift_k.faer_sub(di));
            let denominator = dk.faer_sub(di);
            prod = prod.faer_mul(numerator.faer_div(denominator));
        });

        let prod = prod.faer_abs().faer_sqrt();
        let old_zi = permuted_z.read(i);
        let new_zi = if old_zi < E::faer_zero() {
            prod.faer_neg()
        } else {
            prod
        };

        permuted_z.write(i, new_zi);
    }

    // reuse z to store computed eigenvalues, since it's not used anymore
    let mut eigenvals = z;
    for i in 0..n {
        eigenvals.write(i, mus.read(i).faer_add(shifts.read(i)));
    }

    for (i, p) in pr.iter_mut().enumerate() {
        *p = i;
    }

    pr.sort_unstable_by(
        |&i, &j| match eigenvals.read(i).partial_cmp(&eigenvals.read(j)) {
            Some(ord) => ord,
            None => core::cmp::Ordering::Equal,
        },
    );

    if !applied_householder {
        for p in pl_after.iter_mut() {
            *p = pl_before[*p];
        }
    }

    // compute singular vectors
    for (j, &pj) in pr.iter().enumerate() {
        if pj >= non_deflated {
            repaired_u.rb_mut().col_mut(j).fill_zero();
            repaired_u.write(pl_after[pj], j, E::faer_one());
        } else {
            let mu_j = mus.read(pj);
            let shift_j = shifts.read(pj);

            for (i, &pl_after) in pl_after[..non_deflated].iter().enumerate() {
                let zi = permuted_z.read(i);
                let di = permuted_diag.read(i);

                repaired_u.write(
                    pl_after,
                    j,
                    zi.faer_div(di.faer_sub(shift_j).faer_sub(mu_j)),
                );
            }
            for &pl_after in &pl_after[non_deflated..non_deflated + deflated] {
                repaired_u.write(pl_after, j, E::faer_zero());
            }

            let inv_norm = repaired_u.rb().col(j).norm_l2().faer_inv();
            zipped!(__rw, repaired_u.rb_mut().col_mut(j))
                .for_each(|unzipped!(mut x)| x.write(x.read().faer_mul(inv_norm)));
        }
    }

    if applied_householder {
        let mut idx = 0;
        while idx < n {
            let run_len = run_info[idx];

            if run_len > 1 {
                let mut householder = householder.rb_mut().subrows_mut(idx, run_len);
                let tau = householder.read(run_len - 1);
                householder.write(run_len - 1, E::faer_one());
                let householder = householder.rb();

                let mut repaired_u = repaired_u.rb_mut().subrows_mut(idx, run_len);

                let tau_inv = tau.faer_inv();

                for j in 0..n {
                    let mut col = repaired_u.rb_mut().col_mut(j);
                    let dot = tau_inv.faer_mul(inner_prod_with_conj(
                        householder,
                        Conj::No,
                        col.rb(),
                        Conj::No,
                    ));
                    zipped!(__rw, col.rb_mut(), householder).for_each(|unzipped!(mut u, h)| {
                        u.write(u.read().faer_sub(dot.faer_mul(h.read())))
                    });
                }
            }
            idx += run_len;
        }

        for j in 0..n {
            for (i, &pl_before) in pl_before.iter().enumerate() {
                tmp.write(pl_before, j, repaired_u.read(i, j));
            }
        }

        core::mem::swap(&mut repaired_u, &mut tmp);
    }

    // multiply u by repaired_u (taking into account uninitialized values and sparsity structure)
    //     [u0   0]
    // u = [ 0  u1], u0 is n1×n1, u1 is (n-n1)×(n-n1)

    // compute: u×repaired_u
    // partition repaired_u
    //
    //              [u'_0]
    // repaired_u = [u'_1], u'_0 is n1×n, u'_1 is (n-n1)×n
    //
    //                  [u0×u'_0]
    // u × repaired_u = [u1×u'_1]

    let (repaired_u_top, repaired_u_bot) = repaired_u.rb().split_at_row(n1);
    let (tmp_top, tmp_bot) = tmp.rb_mut().split_at_row_mut(n1);

    crate::utils::thread::join_raw(
        |parallelism| {
            matmul(
                tmp_top,
                u0.rb(),
                repaired_u_top,
                None,
                E::faer_one(),
                parallelism,
            )
        },
        |parallelism| {
            matmul(
                tmp_bot,
                u1.rb(),
                repaired_u_bot,
                None,
                E::faer_one(),
                parallelism,
            )
        },
        parallelism,
    );

    u.copy_from(tmp.rb());
    for i in 0..n {
        let mu_i = mus.read(pr[i]);
        let shift_i = shifts.read(pr[i]);
        diag[i] = mu_i.faer_add(shift_i);
    }
}

pub fn compute_tridiag_real_evd_req<E: Entity>(
    n: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    StackReq::try_all_of([
        StackReq::try_new::<usize>(n)?, // pl_before
        StackReq::try_new::<usize>(n)?, // pl_after
        StackReq::try_new::<usize>(n)?, // pr
        StackReq::try_new::<usize>(n)?, // run_info
        temp_mat_req::<E>(n, 1)?,       // z
        temp_mat_req::<E>(n, 1)?,       // permuted_diag
        temp_mat_req::<E>(n, 1)?,       // permuted_z
        temp_mat_req::<E>(n, 1)?,       // householder
        temp_mat_req::<E>(n, 1)?,       // mus
        temp_mat_req::<E>(n, 1)?,       // shifts
        temp_mat_req::<E>(n, n)?,       // repaired_u
        temp_mat_req::<E>(n, n)?,       // tmp
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, Mat};
    use assert_approx_eq::assert_approx_eq;

    macro_rules! make_stack {
        ($req: expr) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    #[track_caller]
    fn test_evd(diag: &[f64], offdiag: &[f64]) {
        let n = diag.len();
        let mut u = Mat::from_fn(n, n, |_, _| f64::NAN);

        let s = {
            let mut diag = diag.to_vec();
            let mut offdiag = offdiag.to_vec();

            compute_tridiag_real_evd(
                &mut diag,
                &mut offdiag,
                u.as_mut(),
                f64::EPSILON,
                f64::MIN_POSITIVE,
                Parallelism::None,
                make_stack!(compute_tridiag_real_evd_req::<f64>(n, Parallelism::None)),
            );

            Mat::from_fn(n, n, |i, j| if i == j { diag[i] } else { 0.0 })
        };

        let reconstructed = &u * &s * u.transpose();
        for j in 0..n {
            for i in 0..n {
                let target = if i == j {
                    diag[j]
                } else if i == j + 1 {
                    offdiag[j]
                } else if j == i + 1 {
                    offdiag[i]
                } else {
                    0.0
                };

                assert_approx_eq!(reconstructed.read(i, j), target, 1e-13);
            }
        }
    }

    #[test]
    fn test_evd_n() {
        for n in [2, 9, 16, 32, 64, 128, 256, 512, 1024] {
            let diag = (0..n).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
            let offdiag = (0..n - 1)
                .map(|_| rand::random::<f64>())
                .collect::<Vec<_>>();
            test_evd(&diag, &offdiag);
        }
    }

    // rho deflation
    #[test]
    fn test_evd_2_0() {
        let diag = [1.0, 1.0];
        let offdiag = [0.0];
        test_evd(&diag, &offdiag);
    }

    // identical eigenvalues deflation
    #[test]
    fn test_evd_2_1() {
        let diag = [1.0, 1.0];
        let offdiag = [0.5213289];
        test_evd(&diag, &offdiag);
    }

    #[test]
    fn test_evd_3() {
        let diag = [1.79069356, 1.20930644, 1.0];
        let offdiag = [-4.06813537e-01, 0.0];

        test_evd(&diag, &offdiag);
    }

    // zero z component deflation
    #[test]
    fn test_evd_5() {
        let diag = [1.95069537, 2.44845332, 2.56957029, 3.03128102, 1.0];
        let offdiag = [-7.02200909e-01, -1.11661820e+00, -6.81418803e-01, 0.0];
        test_evd(&diag, &offdiag);
    }

    #[test]
    fn test_evd_wilkinson() {
        let diag = [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let offdiag = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        test_evd(&diag, &offdiag);
    }

    #[test]
    fn test_glued_wilkinson() {
        let diag = [
            3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0,
        ];
        let x = 1e-6;
        let offdiag = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, x, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        test_evd(&diag, &offdiag);
    }

    #[test]
    fn test_gh124() {
        use crate::prelude::*;

        let n = 33;

        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = (n - i) as f64;
        }

        let eig = a.selfadjoint_eigendecomposition(crate::Side::Upper);
        let a_reconstructed = eig.reconstruct();

        for i in 0..n {
            equator::assert!(a[(i, i)] == a_reconstructed[(i, i)]);
        }
    }

    #[test]
    fn test_1024() {
        let diag = [
            0.7053659153484068,
            0.16283837518845234,
            0.7820798653289281,
            0.12327594505170891,
            0.9184790549897094,
            0.08504875982857252,
            0.8433580996803264,
            0.4495997809361497,
            0.4169173310797408,
            0.2941457530872156,
            0.8916687736944768,
            0.5464498289412515,
            0.22181724347884602,
            0.7046698747597693,
            0.804830409959474,
            0.7153500404913045,
            0.0413729411446182,
            0.49325981311289013,
            0.34275540892877954,
            0.009953898299149944,
            0.2959545578157238,
            0.47543203459078887,
            0.18254434919371,
            0.9427249874118044,
            0.7895775035583353,
            0.17817879506813616,
            0.9196496683865195,
            0.30994589833471775,
            0.5135664644806912,
            0.3043960588594925,
            0.7484527064686266,
            0.45363420455362935,
            0.47159453863918877,
            0.6576734995383898,
            0.3240203552772424,
            0.26100766016114363,
            0.5707274805294293,
            0.8817639540901481,
            0.3952597393345666,
            0.3326713991937268,
            0.5602396006075736,
            0.0851036328319078,
            0.5294544833169335,
            0.2464980981636684,
            0.47536879325008097,
            0.07671066869411924,
            0.031247822552202487,
            0.610971136298494,
            0.786166907069149,
            0.8451994231783562,
            0.3867020699351741,
            0.17737106906530786,
            0.5016169161228928,
            0.6523142237610623,
            0.18593931139718667,
            0.28401134646853754,
            0.9379110983967619,
            0.06920531452676826,
            0.06997324454394804,
            0.4212681264714916,
            0.6550328027647301,
            0.8437427944983875,
            0.6945545862017923,
            0.029033566324969895,
            0.4663773010678842,
            0.5139251452497144,
            0.003866057084260799,
            0.8811899459865746,
            0.3027648548646934,
            0.04641703732062541,
            0.9613084207506807,
            0.9682773878908925,
            0.22330658466040576,
            0.775411790252933,
            0.6350266147402651,
            0.09565252761838927,
            0.9173875739143477,
            0.6885608023586024,
            0.5669781313302036,
            0.08217215465926608,
            0.16400204786986838,
            0.49596689497923085,
            0.27829911253481476,
            0.3594685333442279,
            0.0939816615160396,
            0.42391724355031546,
            0.1694406386709273,
            0.2409094605961677,
            0.5629851941333569,
            0.9418968688935828,
            0.06954907117154718,
            0.19948279633195443,
            0.6068386632194414,
            0.7978444044292887,
            0.4131218079115442,
            0.339991426062735,
            0.46965089164536766,
            0.5836914185959515,
            0.14240588747830596,
            0.7197652260723726,
            0.03211560727048146,
            0.17179238073084868,
            0.0003002946073034085,
            0.027523425179731498,
            0.19204597313571004,
            0.11915198074487865,
            0.3813395228235744,
            0.9785659783520002,
            0.02966544502267965,
            0.24639140611015187,
            0.7315819260626224,
            0.31969270672097194,
            0.39531777426389525,
            0.6105082373173502,
            0.9844327696659598,
            0.21016439298552236,
            0.6675614477145732,
            0.8897575309381546,
            0.43687525372615177,
            0.40865704199511343,
            0.9943455956320695,
            0.577354831787885,
            0.15617243671141634,
            0.5586795905970019,
            0.7101644725909908,
            0.9527262607955335,
            0.6383337407145285,
            0.7591240256315894,
            0.5599866917399552,
            0.4310290368494408,
            0.42886453089694687,
            0.7767967771446133,
            0.8001163447018743,
            0.23785592222520802,
            0.7944207384675835,
            0.06820773920646483,
            0.15864591091988578,
            0.5674664197037173,
            0.6640261407484389,
            0.36638808614916285,
            0.5163278355829153,
            0.29464914473368187,
            0.7567500149518955,
            0.06983567800525325,
            0.5065181103830178,
            0.009825372298084334,
            0.3407341787906496,
            0.3010767744360012,
            0.40696284547015815,
            0.29419459707927753,
            0.4192200654241548,
            0.9000745560433367,
            0.7222856301839715,
            0.5599675366326251,
            0.23682164559839536,
            0.9430635050036061,
            0.01167698556625052,
            0.012109574772764975,
            0.8994249966764583,
            0.3731513105310449,
            0.53981979118163,
            0.41239845453955315,
            0.6175667791652835,
            0.6714710698737263,
            0.28139225504064813,
            0.7650315706446593,
            0.7074268641880941,
            0.5678617063549468,
            0.40422212478354713,
            0.2122183462187004,
            0.18194499171169265,
            0.6262207991308603,
            0.7997611072135911,
            0.4958876658338275,
            0.7352969937285756,
            0.8039232882040559,
            0.3218508715580266,
            0.29655993607821374,
            0.08631918044757136,
            0.9167751955595608,
            0.4425304410434312,
            0.12708612013809673,
            0.31268371587299126,
            0.7621013459877313,
            0.9241321945944686,
            0.68516590493807,
            0.7888072614467434,
            0.2887474009501818,
            0.23379756256405493,
            0.5025992416683314,
            0.7738106827914786,
            0.20481038351321645,
            0.6115154347825773,
            0.6287048527383301,
            0.5961539818327658,
            0.40149785431378704,
            0.46333832644395556,
            0.4334890153660115,
            0.723556257952246,
            0.26601994391670003,
            0.6085892660690332,
            0.9191553998844725,
            0.09145358845582341,
            0.7900501026123641,
            0.38455316100251524,
            0.5211790963613835,
            0.26380924656745,
            0.4862913960569659,
            0.2738602523606971,
            0.9053353575858042,
            0.6025330237647762,
            0.48084911599337365,
            0.9236885636932398,
            0.027813778126360655,
            0.3436215898476468,
            0.07717933193699533,
            0.1042158535164327,
            0.3548179887327152,
            0.4451790853070148,
            0.3737904692531021,
            0.3580086945449227,
            0.09141079962484444,
            0.5839391018440223,
            0.7198089731653043,
            0.3415727758593279,
            0.7037331090445025,
            0.04890703041124167,
            0.6546736697005364,
            0.7470707160955582,
            0.7183926330303295,
            0.23556026319442358,
            0.8196736191337265,
            0.20690767971141344,
            0.8000353218640882,
            0.6292543974180758,
            0.014100172215330553,
            0.14864701190621643,
            0.12161751231478313,
            0.1976187267648538,
            0.11303939706888622,
            0.2811384026174176,
            0.9455913730032509,
            0.6623683334485828,
            0.29693407399820304,
            0.42631827625568663,
            0.9291993170182012,
            0.693467156906286,
            0.04067233805964898,
            0.8359730772972829,
            0.7365735455283606,
            0.7493388449855591,
            0.5354056240469942,
            0.5151749147610983,
            0.2961572132638135,
            0.6021076409448446,
            0.5217854971295433,
            0.6168026465104337,
            0.9035525790400402,
            0.7531927074976361,
            0.7597181426525309,
            0.19213052837650324,
            0.13276238606842616,
            0.10433008987408188,
            0.5297877954610236,
            0.2982625151090724,
            0.6079013066393296,
            0.9731750162216378,
            0.7343676234755467,
            0.6844438734117643,
            0.37009243288447513,
            0.44984293824744437,
            0.8610640589689661,
            0.2547920670590679,
            0.07269827226800996,
            0.2642860061807133,
            0.7522140744124413,
            0.9727797578076334,
            0.8598385410986861,
            0.0012466633356610357,
            0.05409498517072786,
            0.979790468735558,
            0.055726556914588166,
            0.38835676607782976,
            0.12243670741592039,
            0.9869565550999067,
            0.47248238308478807,
            0.5192104376982419,
            0.07850002185316551,
            0.5109257281727329,
            0.5244384507638432,
            0.5579871046228895,
            0.728734507493948,
            0.09372647712962123,
            0.6901850436728805,
            0.6962144721107824,
            0.3663491307698101,
            0.05883780475477607,
            0.25964380589122016,
            0.7563893904796716,
            0.40299499094686186,
            0.1881532714017874,
            0.29070562017414403,
            0.5863989094649348,
            0.5952144817768622,
            0.8596927327467544,
            0.06249275804507248,
            0.21004710410109295,
            0.8653567780473038,
            0.32586991468196047,
            0.26206836226571484,
            0.854854150397743,
            0.15230952051162627,
            0.857423819690177,
            0.30110599614405165,
            0.8078050270101736,
            0.6562176228981037,
            0.014520397447552513,
            0.46460502029080475,
            0.12152606519132614,
            0.3373362881588685,
            0.5267582508925237,
            0.3418659767282023,
            0.9136710070969981,
            0.5006560127985242,
            0.9029891992204855,
            0.6014063028884823,
            0.7563272677026414,
            0.7831338044515186,
            0.826265278078875,
            0.6967244818311983,
            0.047495521680801955,
            0.6148761115097102,
            0.7154563684395757,
            0.5109959604823276,
            0.04542660046563163,
            0.3657821916026933,
            0.9401380932449741,
            0.574968081244015,
            0.8567967726919206,
            0.18484974893098116,
            0.5513208422685556,
            0.49408773397875305,
            0.2760145631666918,
            0.49251819801559316,
            0.043985002641317994,
            0.5757597866443834,
            0.640169900604829,
            0.8868573246809022,
            0.5181775605724595,
            0.9498723894345027,
            0.5071772359724047,
            0.5921262676979373,
            0.5251856008411668,
            0.977614897360506,
            0.7279069160180331,
            0.7242801698970204,
            0.4238093706647964,
            0.2897246211113851,
            0.005156044742290367,
            0.3939678543700925,
            0.6553263287062726,
            0.3249698941519896,
            0.7653817421025535,
            0.6007240856969494,
            0.6716392966568111,
            0.3581214391269818,
            0.234194878766529,
            0.5579673782865352,
            0.5294028330730169,
            0.3935378814128405,
            0.1769404078427963,
            0.1436081801961412,
            0.7770522784825877,
            0.8664024279790287,
            0.43752472424818034,
            0.4276096978796803,
            0.4085333038399108,
            0.6404321204508902,
            0.18649692348066516,
            0.5488805061777449,
            0.1776235771346376,
            0.8339051954562745,
            0.7846129617014548,
            0.13459842758629936,
            0.8154047177707373,
            0.8812635161872573,
            0.6504463508353618,
            0.6133891102550515,
            0.39696038147833146,
            0.773915748922372,
            0.7043923140705611,
            0.024457008322290807,
            0.6176221573341643,
            0.20105957250917417,
            0.5073226215511174,
            0.12452221746892345,
            0.4537823799084868,
            0.00916512338971387,
            0.21206325820424132,
            0.02604783828328272,
            0.7142974808477487,
            0.21784189880856353,
            0.6841926135301823,
            0.24963464545978453,
            0.804857584403722,
            0.6928669927641435,
            0.8495193685423005,
            0.09813389886571033,
            0.49950991419627744,
            0.26278111261611925,
            0.32354475329519505,
            0.03195758226001999,
            0.8685332729908947,
            0.08299517021909375,
            0.6498137400764834,
            0.483585582788037,
            0.4072480797578164,
            0.013196636632030079,
            0.7592789269929946,
            0.005856384501688017,
            0.6095679889168549,
            0.06320790476969262,
            0.6339610788041369,
            0.20930584638530203,
            0.4283418666816454,
            0.21153857918173613,
            0.77558906888917,
            0.770010997389785,
            0.651631806096506,
            0.421946324390993,
            0.6718648307646292,
            0.02760677095442321,
            0.7917380720057272,
            0.26381829203302853,
            0.9297815649959602,
            0.9154361040176282,
            0.8326165748356265,
            0.14889603202698098,
            0.9095179997576232,
            0.9306162637706332,
            0.3760328387374612,
            0.34331854848775223,
            0.4511762221527058,
            0.7609502538821007,
            0.5333510077423558,
            0.6836714940883674,
            0.1757353350667491,
            0.49665814823767074,
            0.73720468450514,
            0.8615722692891622,
            0.7940265011837097,
            0.6641579030866394,
            0.1247923851208419,
            0.5516041402565558,
            0.2793882467037728,
            0.7702314074314024,
            0.6922472530519388,
            0.6583236889921952,
            0.36959351881712676,
            0.11647997232861351,
            0.4049078049991297,
            0.8437575633051015,
            0.3982481563651509,
            0.16352112573101374,
            0.17870251709912732,
            0.317082899855855,
            0.3293606152920274,
            0.6402955993405304,
            0.8587893106469677,
            0.47382408578926505,
            0.6889894723836297,
            0.08937785914241081,
            0.9979834263252008,
            0.17918948900844067,
            0.8222458007045474,
            0.7254046574055256,
            0.9449440646458904,
            0.36586949327462215,
            0.8474559288593324,
            0.2757188211270656,
            0.8298114484184984,
            0.7083717509106621,
            0.07929518951306291,
            0.48321822599635744,
            0.2272519595159478,
            0.3918179280858316,
            0.3256154934704405,
            0.5200668901608649,
            0.38839751303452497,
            0.3289868368785803,
            0.020472416332142918,
            0.9449724094836994,
            0.11017190730095217,
            0.681227046951888,
            0.22604262773022554,
            0.7957404591751001,
            0.8529316726754453,
            0.8625550353845358,
            0.3036797577106837,
            0.5253323287107735,
            0.1331213775067137,
            0.41784951416071314,
            0.0633457587325309,
            0.8615137605100734,
            0.3714749841359174,
            0.4273108309620661,
            0.9891190467888678,
            0.18895553206437266,
            0.08033293647774276,
            0.6698351297427197,
            0.557883135829446,
            0.1564016861135351,
            0.6998664919385046,
            0.8538103142808179,
            0.9830081626398472,
            0.6544615330814395,
            0.1751707665033898,
            0.7195983384070939,
            0.009578940811453274,
            0.060996207525998325,
            0.8966558698785311,
            0.9172497809804767,
            0.6873688306958446,
            0.8461795781798602,
            0.4434353998651901,
            0.16160926830894118,
            0.7608779077812889,
            0.2612803007485407,
            0.03627163288129809,
            0.3270400921485248,
            0.6613092314367686,
            0.46589759248285356,
            0.44256473260454343,
            0.08913781190338255,
            0.80822034274,
            0.8846639449091466,
            0.22864812439109916,
            0.8697472098723937,
            0.26314336919958015,
            0.8173663428989125,
            0.8293801724866913,
            0.28533981678060083,
            0.9361793340803706,
            0.4588308804884669,
            0.6851390702168644,
            0.5159597693666642,
            0.8647633365166663,
            0.405283209000318,
            0.027013286373555934,
            0.9623182977687629,
            0.16633674271558063,
            0.6148433925350465,
            0.8894403989278813,
            0.9414537113154291,
            0.3329517421302839,
            0.5584046048283866,
            0.6138441851283173,
            0.3672954104967674,
            0.9068487833899797,
            0.2140042802726292,
            0.10805830998778587,
            0.3887830673353594,
            0.39893555200760966,
            0.5531239611221972,
            0.56643888338218,
            0.5864515298969905,
            0.15517481344436945,
            0.9383688715231527,
            0.88830372506353,
            0.5784085409774496,
            0.9276595243228851,
            0.27111586165368606,
            0.5006044275146428,
            0.6331628980172147,
            0.9750822982018904,
            0.6855301033436515,
            0.5533306407110252,
            0.33825986290153,
            0.006820891121309902,
            0.8032214387541726,
            0.3176604985012764,
            0.049414569339980896,
            0.8677227856133868,
            0.26452110384636995,
            0.15734728938218956,
            0.6721247090218879,
            0.8627918479553528,
            0.36353359365652005,
            0.7484654134801864,
            0.6260170209718011,
            0.15895149943217135,
            0.7022369619363362,
            0.5220046031823239,
            0.11674761246146037,
            0.2024226815976462,
            0.07292223582344015,
            0.14461791684809144,
            0.08644560373638244,
            0.5387359740742176,
            0.6300586886740117,
            0.4057844378749569,
            0.7204713889581421,
            0.7479654421518374,
            0.3128880290942384,
            0.9282875419782163,
            0.560828505060497,
            0.9707560873205695,
            0.4386996884639184,
            0.3835104468287137,
            0.25564892884355095,
            0.23452071349649306,
            0.7200620690910834,
            0.4538568158711135,
            0.23195468469984248,
            0.4140255685825196,
            0.8155933304472726,
            0.33574638660575196,
            0.17042501175720404,
            0.42948054793395385,
            0.6933792315324623,
            0.24174220895381737,
            0.2672483475574948,
            0.6945084860982718,
            0.9275548318994762,
            0.26757586438687997,
            0.1619725127790732,
            0.014767600641227063,
            0.6359354498577273,
            0.8517399554895826,
            0.41624134970902693,
            0.10242733254165592,
            0.36104142527332184,
            0.21275847204613352,
            0.5637450717576823,
            0.13916190578251542,
            0.7642373174311216,
            0.3108392595731607,
            0.6613875139089719,
            0.4202825811939228,
            0.6569247307124084,
            0.5929979545340879,
            0.5491675725486057,
            0.9816430988477762,
            0.6280149354048197,
            0.9960805959877523,
            0.3913962798915156,
            0.5989283751994087,
            0.957563231808907,
            0.5566318631741984,
            0.7590884258289505,
            0.7777927567803292,
            0.49879933995873915,
            0.13353354552999153,
            0.9279229048314036,
            0.2370664160854331,
            0.3510296469819706,
            0.6614446504780367,
            0.986995881089584,
            0.6973681945581292,
            0.9272301178162013,
            0.4527254678745297,
            0.29621108710215505,
            0.5441479506379149,
            0.7373102777056667,
            0.7630322089608429,
            0.6912193908815816,
            0.3409212588445083,
            0.7142201141212429,
            0.4690050003663112,
            0.3253414883275987,
            0.4003397400512472,
            0.15071106349450858,
            0.8519426756215787,
            0.4196617799974046,
            0.9557702496092806,
            0.06105672386731875,
            0.3372164624662439,
            0.5366733512854408,
            0.7831324884479661,
            0.36575048862779125,
            0.9614231500820175,
            0.6348316369529686,
            0.5189922460952806,
            0.43800269472918196,
            0.09287327769255926,
            0.8816930914615431,
            0.9695644530635001,
            0.42309303342042304,
            0.0012699929753404726,
            0.21221769351629183,
            0.9021205975761899,
            0.6718567677715109,
            0.8894802935361854,
            0.7687159013589389,
            0.5246866706411499,
            0.2951922938306577,
            0.24337824966550925,
            0.038609396544701746,
            0.4969552905404657,
            0.2853376539175514,
            0.8193272050289364,
            0.9515656458313096,
            0.3833180809316754,
            0.6118741034216085,
            0.9560406914414782,
            0.4813281471290535,
            0.5660544075891027,
            0.48014403501339475,
            0.1548624827537909,
            0.2521215571496669,
            0.6792840060616092,
            0.8966820825244106,
            0.2233361916963894,
            0.47539483434104635,
            0.9933580750729148,
            0.5232142823854049,
            0.30330953285277495,
            0.4058193065525306,
            0.2841078625647636,
            0.9341790098766897,
            0.1844181164360984,
            0.35113145930152423,
            0.39468750562892807,
            0.978093650069611,
            0.4033005486478469,
            0.4090282086121735,
            0.05888555568029086,
            0.7706142435447999,
            0.7149431707453946,
            0.7032011655647592,
            0.5679884348784789,
            0.3334100055674273,
            0.6543687226565188,
            0.21165227713912926,
            0.12424003312270016,
            0.502616199069063,
            0.8046386046332458,
            0.5622554164631756,
            0.5310159317291049,
            0.813670000128855,
            0.43090246072072824,
            0.9727042492678563,
            0.4492543115901886,
            0.21494773052646476,
            0.5280022440554604,
            0.29149725837578766,
            0.8585274366066961,
            0.981397936907793,
            0.2881824919622863,
            0.3717187945468997,
            0.8167581560936864,
            0.34176262363631205,
            0.8061071889740702,
            0.5864116405249786,
            0.12933809062648616,
            0.0841525975831483,
            0.2237657554438418,
            0.36213906832143095,
            0.11525863992975649,
            0.3387717747454808,
            0.05246834019590696,
            0.5577469961367724,
            0.1353807285445956,
            0.04454492294653434,
            0.2611664141702966,
            0.666586605589649,
            0.8683790924295971,
            0.9604668193300366,
            0.8669054523817452,
            0.9971381379732814,
            0.6386911071588334,
            0.8311337256395229,
            0.8209913154744679,
            0.3190205246006165,
            0.2379959670172317,
            0.7214135144475256,
            0.8034105348519563,
            0.050593239766461684,
            0.04870128578090538,
            0.8371953861484251,
            0.31657779041187806,
            0.3412248019187545,
            0.6018970320812442,
            0.7063485489318,
            0.2406313450555253,
            0.3253448530947093,
            0.22697053320241345,
            0.8490166625447791,
            0.07366299950306732,
            0.5971864677253229,
            0.4519164325671219,
            0.8218696785033014,
            0.3303704398020546,
            0.9583059847171592,
            0.8354158809892106,
            0.8810447438269624,
            0.9045871684892826,
            0.6925508360981624,
            0.6077576016278133,
            0.6088293135501034,
            0.21385868193798574,
            0.19343969581617215,
            0.15672502612858574,
            0.5532311798469141,
            0.5300364357130998,
            0.27159475457077165,
            0.9348642681236323,
            0.3671248687211117,
            0.3740413010729394,
            0.6909583957099296,
            0.9522616005453355,
            0.664103296944027,
            0.833386084520864,
            0.7894168877475225,
            0.4934039694008986,
            0.7756872658671982,
            0.10931746758020133,
            0.3782261905250338,
            0.2653770788545289,
            0.025750514391701618,
            0.41786708920340376,
            0.4445013259100543,
            0.02755806713338249,
            0.11114897970709003,
            0.3389713415719606,
            0.24596825687380852,
            0.7874159024263606,
            0.8030986997454017,
            0.9649975194476174,
            0.5290499406201007,
            0.8722943870828824,
            0.8263547374972402,
            0.9470973510998094,
            0.5473412791356321,
            0.8908993185771172,
            0.23299689134732116,
            0.09041787191408546,
            0.5244710602962636,
            0.11245788690392011,
            0.17019840789772578,
            0.4299568630502757,
            0.7709856340368121,
            0.48890239489902176,
            0.4237067351055731,
            0.5973036231788943,
            0.5630310177456401,
            0.9623258188665121,
            0.049427585328416956,
            0.3029247983535698,
            0.9187264687620156,
            0.6259883637405863,
            0.6911307729340133,
            0.26243880256238195,
            0.8669786797759261,
            0.5199008290479468,
            0.11364227563827245,
            0.1587889533012118,
            0.11917059811133768,
            0.9876348990402241,
            0.2816379543970394,
            0.3774479042644483,
            0.9803913529335654,
            0.7475933057756593,
            0.6047642575908447,
            0.5115961178516851,
            0.06733865191629473,
            0.6362095024761912,
            0.670692413644084,
            0.38035722808548056,
            0.6154355086185057,
            0.8150637801548247,
            0.6063176352992721,
            0.2281752443145849,
            0.8712819449743555,
            0.25447796924090993,
            0.4605781905477859,
            0.5233577780424773,
            0.5073646546719424,
            0.49702335292328514,
            0.5132644676629659,
            0.4770163116001642,
            0.02655449787775499,
            0.28766709165060056,
            0.508359223156368,
            0.8949931458897772,
            0.7779079108472795,
            0.005019029094739613,
            0.33164214257546276,
            0.39789103613618493,
            0.3095264062331551,
            0.5327821483780961,
            0.11436420687012117,
            0.6852966208239392,
            0.49772127290514623,
            0.10211654922592406,
            0.3972579771366622,
            0.28453313720732587,
            0.013929500950643603,
            0.029937409958349703,
            0.04621775428910202,
            0.021052279123123685,
            0.5814271598674993,
            0.5020099699833803,
            0.7513611882505998,
            0.7112448275406529,
            0.9992223287186419,
            0.24765460428313746,
            0.32024493308425306,
            0.9483440396875484,
            0.011771246312387906,
            0.13473111297059515,
            0.6672521700736386,
            0.6789990537636947,
            0.749430158016849,
            0.26398821952741314,
            0.36018019638703547,
            0.7560173499866586,
            0.24355872228963882,
            0.9924526833749547,
            0.843308724088044,
            0.9813316233618194,
            0.10818135413275565,
            0.4471459509398701,
            0.9442599202387488,
            0.6060404660698585,
            0.1814646977550629,
            0.8649342817408586,
            0.3902694990283756,
            0.30119079026866713,
            0.9359151443469435,
            0.4296802227927854,
            0.1440795304598126,
            0.0029626406210598866,
            0.5688862708448951,
            0.42916841659263927,
            0.10286861835515715,
            0.09489630681314054,
            0.599430715740142,
            0.09905401250588186,
            0.4334715227960926,
            0.773084362687934,
            0.725693714464473,
            0.20645922850668774,
            0.7108051479044221,
            0.5563573713232006,
            0.07644956513294543,
            0.6318848963020304,
            0.5895363680721534,
            0.3604365247436039,
            0.7979935684389827,
            0.7858662360648364,
            0.2483777201846279,
            0.5538197285437602,
            0.542872180242656,
            0.02854719954577678,
            0.5924952885170911,
            0.09967391268211212,
            0.49534377435246,
            0.017722877162099082,
            0.46466479542448413,
            0.8403659118839767,
            0.799203563321978,
            0.3182577133756542,
            0.5878796409162204,
            0.9038514527942686,
            0.1622723074640542,
            0.8875253681184995,
            0.18012516371216303,
            0.990475584179511,
            0.7683801632321193,
            0.3693918587553383,
            0.42872271341028245,
            0.6793259521280779,
            0.7196521337586772,
            0.556518398665591,
            0.2990729958370584,
            0.7296538595907839,
            0.07392904702697445,
            0.2160195279748658,
            0.3458138602470161,
            0.9409195821706856,
            0.19866004336521748,
            0.42115509464562206,
            0.29596918379420456,
            0.4973471236197752,
            0.9564915081843459,
            0.49468321466200305,
            0.47421952136327705,
            0.17570143978888475,
            0.3281225417913891,
            0.9801142365323164,
            0.7432371502476758,
            0.11913621789604223,
            0.24515947786224623,
            0.27955772593125616,
            0.442486833652855,
            0.11454659326371142,
            0.6905917123768031,
            0.22137787314912405,
            0.05823394611009125,
            0.280703869284638,
            0.787912492685659,
            0.9566228212143926,
            0.1390183212902535,
            0.8140036172456262,
            0.9941495143054285,
            0.5803353332164393,
            0.2883997318878393,
            0.48799604157962617,
            0.6004620061795726,
            0.7627925092036979,
            0.13031597406158968,
            0.5191922758205741,
            0.8883635372191154,
            0.6798261595876015,
            0.6063564698611218,
            0.9168145218042657,
            0.3412967966793048,
            0.4993948120828091,
            0.19164032930858954,
            0.10806153006085184,
            0.3905306417111176,
            0.6166552905518544,
            0.6385354655768402,
        ];
        let offdiag = [
            0.364079660012705,
            0.8619364437773025,
            0.9569501997297541,
            0.4179943342384389,
            0.19569625589690154,
            0.1616807868843807,
            0.6750138741151056,
            0.503415475813367,
            0.5584832495496974,
            0.03173784735329499,
            0.5419064588924857,
            0.4066854757909856,
            0.9747211521383907,
            0.43237667006998526,
            0.7162243501980841,
            0.6467177127382803,
            0.528078968378295,
            0.5934486033306634,
            0.38511097167598785,
            0.7671081105559301,
            0.7256643442804025,
            0.06212402806508943,
            0.6445487757056096,
            0.5984677238003592,
            0.6310287618902886,
            0.3618828521124692,
            0.16238315421500438,
            0.9857388913524263,
            0.7938332750385858,
            0.8521663125427988,
            0.3674197429047408,
            0.37690981583177063,
            0.3651608627147268,
            0.4755868799015808,
            0.2985960271427759,
            0.16755776251302013,
            0.5127013716873778,
            0.30924456674706846,
            0.4416855032326584,
            0.6901977361895484,
            0.8203471817211889,
            0.7828709322328309,
            0.13071969989192445,
            0.587941031937881,
            0.41800778429425334,
            0.5480052417733599,
            0.18635150037753645,
            0.733149240890125,
            0.01861180781765781,
            0.5892261521314609,
            0.9673331004362133,
            0.05459052165153344,
            0.616462120310779,
            0.5912603221899997,
            0.13157386898410772,
            0.46425181907567337,
            0.007531976855268185,
            0.9747961983428572,
            0.9292965166434303,
            0.23484121616795428,
            0.4071442125095137,
            0.6930292324095262,
            0.9723674546801709,
            0.8963571277875497,
            0.21839606861661032,
            0.41364344099739414,
            0.6566202401409047,
            0.19456699795213006,
            0.8652993225070896,
            0.45408803740138515,
            0.03947588960330506,
            0.4568340605982598,
            0.4152142414737656,
            0.7085345472186575,
            0.8546078718415318,
            0.45137877102829815,
            0.6854571822007316,
            0.6986230036319465,
            0.10082768352420224,
            0.3882956889214374,
            0.2869372701498005,
            0.7882406457366057,
            0.006857292973851181,
            0.46424600569610064,
            0.3955444199734879,
            0.1401911903510299,
            0.8414966786525334,
            0.7277904609264436,
            0.505257415996501,
            0.33105362472411404,
            0.4478616319743841,
            0.8019738031090926,
            0.40301094861181885,
            0.3329256398002143,
            0.1905753237529617,
            0.11991658280424633,
            0.20032736537844287,
            0.9872563784055227,
            0.1693261204291121,
            0.883203308478453,
            0.9389668279838239,
            0.1287607257966149,
            0.20854886123576943,
            0.595547187033482,
            0.4940217676440165,
            0.5385152843822042,
            0.8568211113309893,
            0.1796288594490878,
            0.5296487856229674,
            0.0002150456498174469,
            0.8046118201850267,
            0.19412031965388932,
            0.18096127895134861,
            0.3580740374875896,
            0.750535625120133,
            0.2248114135513124,
            0.12247333796381032,
            0.8325951902815765,
            0.36993565746014545,
            0.0378655900989765,
            0.7701477883407127,
            0.008569681057469913,
            0.18239639305827748,
            0.017463073434318743,
            0.006223134170755484,
            0.4986636600376254,
            0.45748793385026376,
            0.28083281158639306,
            0.6647964473318215,
            0.9744708096641638,
            0.07638777233158311,
            0.8574191852575663,
            0.1463867999800229,
            0.3034759474349039,
            0.6091590890794221,
            0.27292981352324874,
            0.3525794544550954,
            0.5731571442423465,
            0.16425755314521107,
            0.20027184092520067,
            0.598247481947006,
            0.7390910098360797,
            0.7809315788930499,
            0.19872585743578441,
            0.6572170312919539,
            0.9593418968697944,
            0.5171409546178073,
            0.31423900590354004,
            0.8794454252267141,
            0.2793857421512965,
            0.30121822812940835,
            0.05005439821641888,
            0.3251264459046529,
            0.9755006446269698,
            0.04880605953584927,
            0.699799716751045,
            0.9742355965196517,
            0.21013845262046627,
            0.9920292531023821,
            0.618009072174111,
            0.6044179132322718,
            0.28798206818000316,
            0.8951313360922258,
            0.22414679674858173,
            0.18113445244997417,
            0.3228575279551015,
            0.46332642794732926,
            0.145551431303105,
            0.04509499143183804,
            0.4410185457204817,
            0.5065049665710109,
            0.7925366355873809,
            0.8920856063763323,
            0.4492347792351562,
            0.9370786999534714,
            0.6609710131149104,
            0.16819110640197754,
            0.5303503047163793,
            0.2255098600457438,
            0.37978472328233326,
            0.6604462019770025,
            0.9093252741071447,
            0.72890884448107,
            0.6307625936024871,
            0.22223180194442016,
            0.3906777359822561,
            0.6877057548936782,
            0.7160399786024957,
            0.06978195295869094,
            0.6045251774117154,
            0.7355489440503573,
            0.9871516465081379,
            0.2219237778961698,
            0.03750860553733659,
            0.8187029963391927,
            0.04603013356702723,
            0.09114615037770857,
            0.0009353728193471911,
            0.0001519744281250901,
            0.5983870679831816,
            0.16617149038347612,
            0.8605958337403734,
            0.3120404622484497,
            0.3852567818112579,
            0.6607126350780094,
            0.4407576101110686,
            0.3716076109337677,
            0.5719768248354734,
            0.2764999162093553,
            0.6189571658158419,
            0.2655953155211087,
            0.2893534225060703,
            0.04661438819103403,
            0.45216874384255545,
            0.04464107669830031,
            0.23878566184511085,
            0.8030387034674387,
            0.8733038306176847,
            0.605136289212071,
            0.14006719994086836,
            0.36880744029997403,
            0.938896345309875,
            0.7795138493234997,
            0.28975322187045127,
            0.32401983129803447,
            0.14393156105066385,
            0.4650547450443836,
            0.8015973422589429,
            0.2889066332386969,
            0.7548749965039905,
            0.6063531921640652,
            0.6728109740550963,
            0.014810337655810857,
            0.7035169836478895,
            0.7711512449259663,
            0.0227950416464886,
            0.8931891035254396,
            0.08336043889650913,
            0.8819802115417786,
            0.897865616056799,
            0.9052704314751564,
            0.07910887643706788,
            0.6851763677901087,
            0.848876417415287,
            0.6578144418367593,
            0.8042864187089538,
            0.8841022993678701,
            0.8228773805377525,
            0.808396424621782,
            0.023871966326144745,
            0.06091774529439098,
            0.9095411680088167,
            0.13793653217087698,
            0.07598504194009403,
            0.45420214909609846,
            0.4336268818337997,
            0.5029666674724202,
            0.93816825750109,
            0.814734975799812,
            0.1396486878455615,
            0.036230841199559105,
            0.49615497702579714,
            0.9283901005928664,
            0.5909605677227489,
            0.025360103350414387,
            0.6161155941417487,
            0.6668428012756875,
            0.7777371240217217,
            0.7903834944315103,
            0.31521717475745503,
            0.5168240272012035,
            0.9021258555246173,
            0.42196641468668883,
            0.3800388089940059,
            0.7366491056848764,
            0.42943284279964655,
            0.03291338668626409,
            0.5900087776262125,
            0.9258807657062089,
            0.7749079727003585,
            0.7413822825952217,
            0.045188775102580014,
            0.9156084769512414,
            0.454567630924573,
            0.3923157459116753,
            0.7775413279881932,
            0.6158958318126028,
            0.02182073566111753,
            0.03015826560805257,
            0.4011260917660313,
            0.48999742666912405,
            0.931002799367698,
            0.7223441054026664,
            0.022588285538463815,
            0.6142164299135794,
            0.9142958212693235,
            0.8472008162609926,
            0.14154774963307593,
            0.39804235938831467,
            0.04265133894968731,
            0.8976868518458463,
            0.1963902536245068,
            0.03351862084622026,
            0.003023434917818313,
            0.2790848214099678,
            0.693008064736397,
            0.4303044937245357,
            0.015926380253050043,
            0.4974781659013894,
            0.23939958830895713,
            0.8170477324547134,
            0.07692723124838274,
            0.04085782872229837,
            0.5857251318641874,
            0.06209398537415256,
            0.8779084717034803,
            0.2540965876018111,
            0.5410800210655744,
            0.589334084545765,
            0.041434100224223225,
            0.7798449334442401,
            0.016629249066081653,
            0.5893297051276523,
            0.6873529895142195,
            0.02291804677607645,
            0.49952504489129,
            0.4545219812958874,
            0.17761338662457016,
            0.2630512472633051,
            0.7546700239389786,
            0.007869992490377342,
            0.14236507102910767,
            0.13163488175342264,
            0.8960785207982166,
            0.47057311516631006,
            0.538498727695148,
            0.6678328829579107,
            0.8618140685067799,
            0.7594804956200164,
            0.29571327755880816,
            0.5959364751567997,
            0.6238454107171207,
            0.8788909551279358,
            0.8950478597848254,
            0.6940381075735362,
            0.7972403698501959,
            0.7346658711501034,
            0.44469685117821944,
            0.5501954477332992,
            0.44745529217331437,
            0.9179395008707721,
            0.33960303430677663,
            0.42497842979380984,
            0.729195877020768,
            0.3583817681533752,
            0.5409059281368742,
            0.10012636701523914,
            0.6510402876270114,
            0.9637876577783409,
            0.14661150779789067,
            0.45514934792245,
            0.15117207584514447,
            0.8841015424825794,
            0.7832395548815446,
            0.44260655841988816,
            0.797855724623169,
            0.5423955364367172,
            0.7900789587590238,
            0.6593099681417133,
            0.6244177932614147,
            0.6850891384627068,
            0.6674755260678876,
            0.5063165745484319,
            0.4142886169347809,
            0.3138332069496984,
            0.07607529791944323,
            0.7477204527143245,
            0.7528938643559445,
            0.8092269507658472,
            0.07549523378492762,
            0.4089494118652073,
            0.1819477366609078,
            0.5730093886951714,
            0.11472462066936828,
            0.14128976804760174,
            0.796272271030552,
            0.8558576615297402,
            0.5930294671738413,
            0.010190832510698855,
            0.2763689645650791,
            0.8159107540186498,
            0.36504285479199394,
            0.43775549709977024,
            0.8062347103947695,
            0.5890984260043509,
            0.9566187182061902,
            0.06359511210919133,
            0.265381894032363,
            0.5968141623837152,
            0.06568865277274394,
            0.8506277764025182,
            0.5274907371351217,
            0.011209318485873476,
            0.48012247864583524,
            0.2789970428722339,
            0.9697981951525741,
            0.5180273710082225,
            0.8858776860640896,
            0.8073613618559373,
            0.296238108208277,
            0.4606881823828842,
            0.19057559377543776,
            0.6302892454305035,
            0.9057624374063609,
            0.9753214742101239,
            0.4035035132542898,
            0.8036602359327839,
            0.8180762714807589,
            0.4006483532123818,
            0.6286801932753878,
            0.3464684390593167,
            0.9726421429855053,
            0.27933808693553175,
            0.22232095186342826,
            0.1321698471108982,
            0.9823204539823956,
            0.20734362842202725,
            0.3054112419541267,
            0.9738341034027908,
            0.7980847615925393,
            0.9261643208106606,
            0.49849214705993916,
            0.6750511730291295,
            0.5436349520321724,
            0.1866575034759651,
            0.4958909170969059,
            0.584755600055311,
            0.9553617818187332,
            0.25788872328481927,
            0.2885312711240896,
            0.9522289598474378,
            0.3215743155124523,
            0.06985372318579819,
            0.00960508156505524,
            0.40541648772426997,
            0.08472665971248605,
            0.1311733793758193,
            0.8621888285195718,
            0.14778581624049614,
            0.3860940179551059,
            0.7086268437490412,
            0.839008064294021,
            0.831667971162734,
            0.9412050466437062,
            0.6769821532594413,
            0.9802223803504255,
            0.8402778108176598,
            0.04272744142122398,
            0.8234018083084985,
            0.8483002966699806,
            0.4783834736624557,
            0.08019345467425332,
            0.06624695988749718,
            0.33130756304878284,
            0.7473445523204167,
            0.19204761437188023,
            0.31456997112460183,
            0.3197052704630253,
            0.9155398251226773,
            0.10920179355872262,
            0.8850835997784308,
            0.7415884159212427,
            0.6008720914747677,
            0.5966404568919287,
            0.6678082899773534,
            0.6647042276791859,
            0.13653199747277434,
            0.1590269386585248,
            0.24351142578286422,
            0.05351526271499507,
            0.480404582606065,
            0.9639607399619593,
            0.8437693697675417,
            0.31622338226595115,
            0.6146779434277841,
            0.8042521190100095,
            0.8804339304389627,
            0.14344184854253839,
            0.8269692995557392,
            0.0665691524558194,
            0.8526628226479506,
            0.767049536304247,
            0.3461964135682115,
            0.36856172017867295,
            0.10946481225091964,
            0.6296343389533717,
            0.44381598426926994,
            0.9258013284974322,
            0.32694281872017805,
            0.2726560102225223,
            0.15238554752581734,
            0.40100456153256014,
            0.0962876678026432,
            0.5421348324045241,
            0.17654825676340358,
            0.7991046206402563,
            0.5097043935096769,
            0.3625876559363187,
            0.17002347824871733,
            0.9142559695612397,
            0.6240546428034344,
            0.47418384105881983,
            0.4328050863529749,
            0.37694500070621684,
            0.3823893687138509,
            0.9474183094572671,
            0.0425700448592643,
            0.45681932073669707,
            0.8459580863343021,
            0.7281252096968317,
            0.13245508285507057,
            0.4544859650317712,
            0.795436118097436,
            0.3220559285377542,
            0.930382958202653,
            0.8606925995767063,
            0.10744292057296811,
            0.3804285216765506,
            0.07346077082697344,
            0.9051407634731666,
            0.48868563512865093,
            0.7561652947259722,
            0.32451329465326695,
            0.1683048709780709,
            0.4490566842260103,
            0.5145075784678194,
            0.10099744613697426,
            0.2934481988878076,
            0.5916845644262974,
            0.4832541607264088,
            0.5712971328446554,
            0.5577363874857717,
            0.2713509285275142,
            0.8178625736500287,
            0.6235688959393447,
            0.4041635457992726,
            0.41785052165100733,
            0.8636165697901461,
            0.612333179187261,
            0.3659818662708364,
            0.4838222151542816,
            0.6022043392049404,
            0.38569465632238853,
            0.7381654114757978,
            0.49106416021615007,
            0.5631976677422218,
            0.5838695419764669,
            0.2618470512266927,
            0.10584941032301265,
            0.49451897305663906,
            0.11656986085353915,
            0.6915071688677712,
            0.3209906708852702,
            0.392125323036854,
            0.14413801154530104,
            0.9131432256464062,
            0.35902930762233853,
            0.1074989345751205,
            0.9189592533371945,
            0.7518654692311804,
            0.47058052083207524,
            0.6863033342591917,
            0.7267698462825074,
            0.997752716436612,
            0.33941761300770323,
            0.4739026629645866,
            0.3384268643397502,
            0.5890468309990068,
            0.6928666643657653,
            0.5607667525331231,
            0.3835145567898476,
            0.04809805878637352,
            0.024530177444005652,
            0.7275808347614324,
            0.7323420610359163,
            0.2464138909253859,
            0.0867953399213629,
            0.065221150738069,
            0.2284068251951885,
            0.6141801156246481,
            0.028652170297960078,
            0.01536260222327468,
            0.5365065832411604,
            0.8307372815730425,
            0.8686352461606127,
            0.6816976554607935,
            0.22181760130671813,
            0.833322498831465,
            0.6206948722987908,
            0.7650361387869229,
            0.155813512075756,
            0.8223468951841878,
            0.3781403611278582,
            0.7655941934440494,
            0.6197346344470033,
            0.3909432728463075,
            0.44940621450772567,
            0.14402655813413134,
            0.9490160147912361,
            0.45059020259164384,
            0.8288304516905692,
            0.22154812072970065,
            0.7638904244419664,
            0.9140165531792395,
            0.9729128604771908,
            0.40663074418674383,
            0.8247314226757199,
            0.8435702052822057,
            0.23412305844853143,
            0.3910635586305843,
            0.5284823935598155,
            0.7130483590003284,
            0.6301763288002147,
            0.8528811526860188,
            0.35303060105548534,
            0.9009384201155841,
            0.4597228160343465,
            0.22713033822380213,
            0.43270711895875424,
            0.15327031186182827,
            0.07187499206089087,
            0.3460010898106064,
            0.40656354932943595,
            0.9471974604707348,
            0.2897567830693354,
            0.45171682518039835,
            0.7176364350751925,
            0.910810386548481,
            0.6534177747615387,
            0.38220765185486394,
            0.21556098627492126,
            0.2839639908993533,
            0.6063781685160827,
            0.5427938618032221,
            0.09231768460656165,
            0.24123374020246036,
            0.2789957093709534,
            0.7875391246881193,
            0.058856936622483924,
            0.020843494856079947,
            0.7590608490503631,
            0.9056329924775921,
            0.6970582637002153,
            0.5420389610682061,
            0.17313958162495258,
            0.44796200984191836,
            0.3275157128924281,
            0.6381130437156012,
            0.3144713755287085,
            0.5532941324443039,
            0.32041572677539465,
            0.9295958900883042,
            0.6546155967937327,
            0.9247163066747905,
            0.38603606171650495,
            0.707879408013021,
            0.9356583899091294,
            0.3488211065630509,
            0.5312538131390944,
            0.8490708932350948,
            0.4145168004898151,
            0.15897022178768472,
            0.8663630363336752,
            0.0007266826019544093,
            0.7196044787538562,
            0.17732968770967172,
            0.18956146079767966,
            0.9950867397987938,
            0.27016493475429104,
            0.38218321070707284,
            0.37158147404756714,
            0.7736106486880241,
            0.7539313656603518,
            0.4431266603328273,
            0.9409719538270968,
            0.6767272794871879,
            0.5556958137607451,
            0.3620466326683718,
            0.9003462669276908,
            0.5204830336881501,
            0.07019917316405,
            0.13622376053200003,
            0.7741224978513409,
            0.051353131582466016,
            0.3241431791300702,
            0.8035241807115899,
            0.4533080237475894,
            0.7950580501176364,
            0.285558157147243,
            0.6768731418263189,
            0.8788945793423171,
            0.15279016874964224,
            0.9696641644751026,
            0.024786299116959398,
            0.21449437687649509,
            0.8898679068501143,
            0.15773477782673806,
            0.4318372608363338,
            0.5963106313604036,
            0.7985488380382819,
            0.3174049952767162,
            0.9473212711674646,
            0.5994900844979442,
            0.41962557676492707,
            0.9602824072495657,
            0.7296720964947125,
            0.22260977057384457,
            0.057283889189107984,
            0.21587316769730436,
            0.7644234027546577,
            0.39861406369802277,
            0.7482602094308441,
            0.6241699815753742,
            0.06924175494663865,
            0.6002682971955638,
            0.9626769121746738,
            0.16189269454485922,
            0.5521740498562089,
            0.7820102411428099,
            0.7805233168819733,
            0.4830376583870286,
            0.8047598360701574,
            0.3191892814201236,
            0.8214044199169485,
            0.6397936117500288,
            0.9334125978255867,
            0.46759510944126936,
            0.01753060926717831,
            0.8187973036485883,
            0.156575715133476,
            0.8633196058933529,
            0.8064842630470342,
            0.9494678449984685,
            0.9878656212714773,
            0.6060044678477192,
            0.6876381948673684,
            0.5473737020103868,
            0.048390299228290634,
            0.8628503606300357,
            0.11230515858370227,
            0.5199762646961817,
            0.2673933307911175,
            0.07615622158213453,
            0.1857836708569922,
            0.8427782157325414,
            0.7926972018287994,
            0.6922037702306143,
            0.684605981099623,
            0.8833539226268813,
            0.6709645992347512,
            0.9622411733346549,
            0.8089220068400531,
            0.20626963868127146,
            0.8331974960806043,
            0.1307470024105779,
            0.33257961647475454,
            0.028632300296805968,
            0.19625091953750762,
            0.9750811602264828,
            0.8128941954898586,
            0.4719369076955766,
            0.3973196263119575,
            0.7721667252207919,
            0.9093380853691181,
            0.7749890711864922,
            0.27472412210651076,
            0.11455214222256116,
            0.7766943013339849,
            0.45007600705013007,
            0.40453854475231943,
            0.42531444515314487,
            0.49134550927800136,
            0.1565758903072042,
            0.15634088753872533,
            0.23001924923563977,
            0.3951598801146685,
            0.4535360962793058,
            0.3235624685093821,
            0.4133703556887912,
            0.9172857033598014,
            0.6474346315025084,
            0.17682587108400294,
            0.20601304508096907,
            0.5817795169408485,
            0.7667248510501843,
            0.5612967204134253,
            0.18699359405457272,
            0.1175593789698437,
            0.8375189080063993,
            0.2181522622344787,
            0.05871530324168128,
            0.29816354103718257,
            0.5417413084238404,
            0.5303918562554532,
            0.26450265450797417,
            0.10977463737931958,
            0.1495128815539314,
            0.1740353798335914,
            0.885501099362763,
            0.5773277942326036,
            0.0333919975215774,
            0.6652137671141506,
            0.13651414250456706,
            0.33202749695299083,
            0.23320297383105015,
            0.8481149173152761,
            0.6684300354156307,
            0.2513991924309523,
            0.7045139473400985,
            0.8292734146900084,
            0.053012076029575694,
            0.46803028708576466,
            0.894775201868527,
            0.5688145514665283,
            0.6970216285120504,
            0.6000282200917396,
            0.15277513974426282,
            0.6312901225844111,
            0.4696062377427428,
            0.003496665328008075,
            0.5042337813717614,
            0.1317585392789894,
            0.34493833875345004,
            0.5312377325883146,
            0.8699804341815669,
            0.3663909495338684,
            0.5329582228849704,
            0.9233537437256378,
            0.30201584980816876,
            0.8575053038140589,
            0.5941846776085685,
            0.4989407646035138,
            0.07992568191701421,
            0.38365847879263093,
            0.18028969141197237,
            0.938621102866954,
            0.11366400387635955,
            0.85273366286219,
            0.5587835226927677,
            0.3202538487702684,
            0.7318355361438076,
            0.11430099456796539,
            0.036789857855900276,
            0.03500866089703325,
            0.00920214520861129,
            0.15208117034916369,
            0.37441820143493976,
            0.2836977651713585,
            0.7503670722827566,
            0.3560120774287575,
            0.18986644024169197,
            0.49948463331449877,
            0.31897694772138807,
            0.6659232450424366,
            0.7880685272283637,
            0.2907423345938278,
            0.1317563340042952,
            0.5460567205385465,
            0.5533638605503894,
            0.42131148438940025,
            0.020182994043714486,
            0.8679359407524343,
            0.02325563225414251,
            0.4430617347658935,
            0.4340414048563358,
            0.14724898407813913,
            0.10167348885050442,
            0.2984969122987462,
            0.4092345988165035,
            0.8660809416174372,
            0.9292992929315136,
            0.139338605611934,
            0.16716401448888885,
            0.7089647527894133,
            0.41717601163011264,
            0.8964691666983746,
            0.8762219492995926,
            0.7741304233715472,
            0.3114711267282779,
            0.7766091081692695,
            0.32485626703923487,
            0.9250182727932481,
            0.1999905710590797,
            0.6462233017859472,
            0.4044757161812671,
            0.6714356638885354,
            0.1626518315353208,
            0.05120527743234227,
            0.6193802844130676,
            0.4480955231037772,
            0.2929352633762058,
            0.10101222640667562,
            0.5989539532124533,
            0.8289473420446114,
            0.790614708071105,
            0.9762636788560104,
            0.8203411244857088,
            0.901046346419268,
            0.856600343272874,
            0.4237549899523545,
            0.5240003448932211,
            0.11128151197182246,
            0.020530898636691663,
            0.08026381681681394,
            0.14312600839138245,
            0.9400988322818357,
            0.48649169301439166,
            0.9432895696864381,
            0.6474285676830914,
            0.8152650000791534,
            0.5185864233021551,
            0.3432979051937519,
            0.4893594344822948,
            0.3338744969314835,
            0.750397141517703,
            0.20792826370002604,
            0.5983351631229993,
            0.7885777617875244,
            0.658109121407257,
            0.0232352578061088,
            0.18065398838149405,
            0.6901131438235127,
            0.1740587191848625,
            0.1168492498457161,
            0.08846015570552923,
            0.3964085245630855,
            0.2696360847691305,
            0.15580161767187872,
            0.24353643984806417,
            0.8241918200198697,
            0.9233225421219916,
            0.7628162780085825,
            0.2367014390159462,
            0.7101796108499744,
            0.015649450012932276,
            0.9905744887560415,
            0.9957622931952277,
            0.05009176243195279,
            0.7843172222589199,
            0.20425531790171436,
            0.27448600566305736,
            0.8361652652578457,
            0.3864335166587488,
            0.6095387214917096,
            0.20916825623499147,
            0.9867562118766336,
            0.4853993280851947,
            0.4566880180910684,
            0.34182565067284365,
            0.55074291701347,
            0.8677983101877876,
            0.32529366325354037,
            0.5351366045332135,
            0.13363856278546826,
            0.7853325878475015,
            0.03956799465142302,
            0.45852391602588627,
            0.36699494355748663,
            0.07279288149149299,
            0.6838076927742218,
            0.5069875583028298,
            0.9021589282088052,
            0.5894522139755841,
            0.8596098479678158,
            0.9736316299280551,
            0.9268336974709885,
            0.7382839398245667,
            0.29500062511065717,
            0.1457769872497532,
            0.8961235277346835,
            0.09983866438684164,
            0.5138557528845914,
            0.040456544737760725,
            0.1744716454683204,
            0.3217540688347269,
            0.8033467436950723,
            0.5823041384792214,
            0.26213922327964856,
            0.8869436479448788,
            0.8390650003954797,
            0.721144120980615,
            0.034441457342738735,
            0.004696754245278489,
            0.3859610500049727,
            0.8355107023117931,
            0.12878726500581128,
            0.1223625204700417,
            0.8058009144376403,
            0.4201174804458172,
            0.2792822122060553,
            0.05557022745948215,
            0.8010997326466298,
            0.032565274579125836,
            0.9851635113275754,
            0.4597516296889842,
            0.04946595046060631,
            0.7927446863645953,
            0.13410069046234374,
            0.1894772933728146,
            0.03674743269655245,
            0.34046783255903634,
            0.32595422603501145,
            0.1028058256505886,
            0.12914249890522211,
            0.7581272488295554,
            0.2041382850575718,
            0.13754724689178432,
            0.9865152878339165,
            0.06538887132273463,
            0.9259539692398506,
            0.04836049977420065,
            0.7182687497070794,
            0.08130031153873629,
        ];

        test_evd(&diag, &offdiag);
    }
}
