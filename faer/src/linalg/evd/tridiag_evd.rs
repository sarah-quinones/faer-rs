// QR algorithm ported from Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use linalg::jacobi::JacobiRotation;

use crate::{internal_prelude::*, perm::swap_cols_idx};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvdError {
    NoConvergence,
}

#[math]
#[allow(dead_code)]
fn tridiag_to_mat<C: RealContainer, T: RealField<C, MathCtx: Default>>(
    diag: ColRef<'_, C, T, usize, ContiguousFwd>,
    offdiag: ColRef<'_, C, T, usize, ContiguousFwd>,
) -> Mat<C, T, usize, usize> {
    let n = diag.nrows();
    let ctx: &Ctx<C, T> = &ctx();
    let mut m = Mat::zeros_with(ctx, n, n);

    help!(C);
    {
        let mut m = m.as_mut();
        for i in 0..n {
            write1!(m[(i, i)] = math(copy(diag[i])));
            if i + 1 < n {
                write1!(m[(i + 1, i)] = math(copy(offdiag[i])));
                write1!(m[(i, i + 1)] = math(copy(offdiag[i])));
            }
        }
    }
    m
}

#[math]
fn qr_algorithm<C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    diag: ColMut<'_, C, T, usize, ContiguousFwd>,
    offdiag: ColMut<'_, C, T, usize, ContiguousFwd>,
    u: Option<MatMut<'_, C, T, usize, usize>>,
) -> Result<(), EvdError> {
    help!(C);
    let n = diag.nrows();

    let mut u = u;
    let mut diag = diag;
    let mut offdiag = offdiag;

    if let Some(mut u) = u.rb_mut() {
        u.fill(math(zero()));
        u.diagonal_mut().fill(math(one()));
    }

    if n <= 1 {
        return Ok(());
    }

    let max = math.max(diag.norm_max_with(ctx), offdiag.norm_max_with(ctx));
    if math.is_zero(max) {
        return Ok(());
    }
    let max_inv = math(recip(max));

    for mut x in diag.rb_mut().iter_mut() {
        write1!(x, math(x * max_inv));
    }
    for mut x in offdiag.rb_mut().iter_mut() {
        write1!(x, math(x * max_inv));
    }

    let eps = math.eps();
    let sml = math.min_positive();
    let mut end = n - 1;
    let mut start = 0;

    let max_iter = Ord::max(30, ctx.nbits() / 2)
        .saturating_mul(n)
        .saturating_mul(n);

    for iter in 0..max_iter {
        for i in start..end {
            if math(abs(offdiag[i]) < sml || abs(offdiag[i]) < eps * hypot(diag[i], diag[i + 1])) {
                write1!(offdiag[i] = math.zero());
            }
        }

        while end > 0 && math(is_zero(offdiag[end - 1])) {
            end -= 1;
        }

        if end == 0 {
            break;
        }

        if iter + 1 == max_iter {
            for mut x in diag.rb_mut().iter_mut() {
                write1!(x, math(x * max));
            }
            for mut x in offdiag.rb_mut().iter_mut() {
                write1!(x, math(x * max));
            }

            return Err(EvdError::NoConvergence);
        }

        start = end - 1;
        while start > 0 && math(!is_zero(offdiag[start - 1])) {
            start -= 1;
        }

        {
            // Wilkinson shift
            let td = math((diag[end - 1] - diag[end]) * from_f64(0.5));
            let e = math(copy(offdiag[end - 1]));
            let mut mu = math(copy(diag[end]));

            if math(is_zero(td)) {
                mu = math(mu - abs(e));
            } else if math(!is_zero(e)) {
                let e2 = math(abs2(e));
                let h = math(hypot(td, e));
                let h = math(if gt_zero(td) { copy(h) } else { -h });
                math(if is_zero(e2) {
                    mu = mu - e / ((td + h) / e)
                } else {
                    mu = mu - e2 / (td + h)
                });
            }

            let mut x = math(diag[start] - mu);
            let mut z = math(copy(offdiag[start]));

            let mut k = start;
            while k < end && math(!is_zero(z)) {
                let rot = JacobiRotation::make_givens(ctx, math.copy(x), math.copy(z));

                // T = G' T G
                let sdk = math(rot.s * diag[k] + rot.c * offdiag[k]);
                let dkp1 = math(rot.s * offdiag[k] + rot.c * diag[k + 1]);

                write1!(
                    diag[k] = math(
                        rot.c * (rot.c * diag[k] - rot.s * offdiag[k])
                            - rot.s * (rot.c * offdiag[k] - rot.s * diag[k + 1])
                    )
                );
                write1!(diag[k + 1] = math(rot.s * sdk + rot.c * dkp1));
                write1!(offdiag[k] = math(rot.c * sdk - rot.s * dkp1));

                if k > start {
                    write1!(offdiag[k - 1] = math(rot.c * offdiag[k - 1] - rot.s * z));
                }

                x = math(copy(offdiag[k]));

                if k < end - 1 {
                    z = math(-rot.s * offdiag[k + 1]);
                    write1!(offdiag[k + 1] = math(rot.c * offdiag[k + 1]));
                }

                if let Some(u) = u.rb_mut() {
                    let (x, y) = u.two_cols_mut(k, k + 1);
                    rot.apply_on_the_right_in_place(ctx, x, y);
                }
                k += 1;
            }
        }
    }

    for i in 0..n - 1 {
        let mut idx = i;
        let mut min = math(copy(diag[i]));

        for k in i + 1..n {
            if math(diag[k] < min) {
                idx = k;
                min = math(copy(diag[k]));
            }
        }

        if idx != i {
            let (a, b) = math((copy(diag[i]), copy(diag[idx])));
            write1!(diag[i] = b);
            write1!(diag[idx] = a);
            if let Some(mut u) = u.rb_mut() {
                swap_cols_idx(u.rb_mut(), i, idx);
            }
        }
    }

    for mut x in diag.rb_mut().iter_mut() {
        write1!(x, math(x * max));
    }

    Ok(())
}

#[cfg(test)]
mod evd_qr_tests {
    use faer_traits::Unit;

    use super::*;
    use crate::{assert, utils::approx::*, ColMut, Mat};

    #[track_caller]
    fn test_evd(diag: &[f64], offdiag: &[f64]) {
        let n = diag.len();
        let mut u = Mat::full(n, n, f64::NAN);

        let s = {
            let mut diag = diag.to_vec();
            let mut offdiag = offdiag.to_vec();

            qr_algorithm(
                &ctx(),
                ColMut::from_slice_mut(&mut diag)
                    .try_as_col_major_mut()
                    .unwrap(),
                ColMut::from_slice_mut(&mut offdiag)
                    .try_as_col_major_mut()
                    .unwrap(),
                Some(u.as_mut()),
            )
            .unwrap();

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

                let approx_eq = ApproxEq::<Unit, f64>::eps();
                assert!(reconstructed[(i, j)] ~ target);
            }
        }
    }

    #[test]
    fn test_evd_2_0() {
        let diag = [1.0, 1.0];
        let offdiag = [0.0];
        test_evd(&diag, &offdiag);
    }

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

    // https://github.com/sarah-ek/faer-rs/issues/82
    #[test]
    fn test_gh_82() {
        let diag = [
            0.0,
            0.0,
            1.0769230769230773,
            -0.4290761869709236,
            -0.8278050499098524,
            0.07994922044020283,
            -0.35579623371016944,
            0.6487378508167678,
            -0.9347442346214521,
            -0.08624745233962683,
            -0.4999243909534632,
            1.3708277457481026,
            -0.2592167303689501,
            -0.5929351972647323,
            -0.5863220906879729,
            0.15069873027683844,
            0.2449309426221532,
            0.5599151389028441,
            0.440084861097156,
            9.811634162559901e-17,
        ];
        let offdiag = [
            1.7320508075688772,
            2.081665999466133,
            2.0303418353670932,
            1.2463948607107287,
            1.5895840148470526,
            1.3810057029812097,
            1.265168346300635,
            0.8941431038915991,
            1.007512301091709,
            0.5877505835309086,
            1.0370970338888965,
            0.8628932798233644,
            1.1935059937001073,
            1.1614143449715744,
            0.41040224297074174,
            0.561318309959268,
            3.1807090401145072e-15,
            0.4963971959331084,
            1.942890293094024e-16,
        ];

        test_evd(&diag, &offdiag);
    }

    #[test]
    fn test_gh_82_mini() {
        let diag = [1.0000000000000002, 1.0000000000000002];
        let offdiag = [7.216449660063518e-16];

        test_evd(&diag, &offdiag);
    }
}
