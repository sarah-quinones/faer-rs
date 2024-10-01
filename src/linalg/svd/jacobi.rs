// Algorithm ported from Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2013-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::{
    assert, linalg::jacobi::JacobiRotation, perm::swap_cols_idx as swap_cols, MatMut, RealField,
};
use reborrow::*;

fn compute_2x2<E: RealField>(
    m00: E,
    m01: E,
    m10: E,
    m11: E,
) -> (JacobiRotation<E>, JacobiRotation<E>) {
    let t = m00.faer_add(m11);
    let d = m10.faer_sub(m01);

    let rot1 = if d == E::faer_zero() {
        JacobiRotation {
            c: E::faer_one(),
            s: E::faer_zero(),
        }
    } else {
        let u = t.faer_mul(d.faer_inv());
        let tmp = (E::faer_one().faer_add(u.faer_mul(u)))
            .faer_sqrt()
            .faer_inv();
        let tmp = if tmp == E::faer_zero() {
            u.faer_abs().faer_inv()
        } else {
            tmp
        };
        JacobiRotation {
            c: u.faer_mul(tmp),
            s: tmp,
        }
    };
    let j_right = {
        let (m00, m01, _, m11) = rot1.apply_on_the_left_2x2(m00, m01, m10, m11);
        JacobiRotation::from_triplet(m00, m01, m11)
    };
    let j_left = rot1 * j_right.transpose();

    (j_left, j_right)
}

pub enum Skip {
    None,
    First,
    Last,
}

pub fn jacobi_svd<E: RealField>(
    matrix: MatMut<'_, E>,
    u: Option<MatMut<'_, E>>,
    v: Option<MatMut<'_, E>>,
    skip: Skip,
    epsilon: E,
    consider_zero_threshold: E,
) -> usize {
    assert!(matrix.nrows() == matrix.ncols());
    let n = matrix.nrows();

    if let Some(u) = u.rb() {
        assert!(n == u.nrows());
        assert!(n == u.ncols());
    };
    if let Some(v) = v.rb() {
        assert!(n == v.ncols());
    }

    let mut matrix = matrix;
    let mut u = u;
    let mut v = v;

    if let Some(mut u) = u.rb_mut() {
        for j in 0..n {
            for i in 0..j {
                u.rb_mut().write(i, j, E::faer_zero());
            }
            u.rb_mut().write(j, j, E::faer_one());
            for i in j + 1..n {
                u.rb_mut().write(i, j, E::faer_zero());
            }
        }
    }

    if let Some(mut v) = v.rb_mut() {
        if matches!(skip, Skip::First) {
            for i in 0..n - 1 {
                v.rb_mut().write(i, 0, E::faer_zero());
            }
            v = v.submatrix_mut(0, 1, n - 1, n - 1);
        }

        let m = v.nrows();
        let n = v.ncols();
        for j in 0..n {
            for i in 0..j {
                v.rb_mut().write(i, j, E::faer_zero());
            }
            if j == m {
                break;
            }
            v.rb_mut().write(j, j, E::faer_one());
            for i in j + 1..m {
                v.rb_mut().write(i, j, E::faer_zero());
            }
        }
    }

    let mut max_diag = E::faer_zero();
    {
        let diag = matrix.rb().diagonal().column_vector();
        for idx in 0..diag.nrows() {
            let d = diag.read(idx).faer_abs();
            max_diag = if d > max_diag { d } else { max_diag };
        }
    }

    let precision = epsilon.faer_scale_power_of_two(E::faer_one().faer_add(E::faer_one()));
    loop {
        let mut failed = false;
        for p in 1..n {
            for q in 0..p {
                let threshold = precision.faer_mul(max_diag);
                let threshold = if threshold > consider_zero_threshold {
                    threshold
                } else {
                    consider_zero_threshold
                };

                if (matrix.read(p, q).faer_abs() > threshold)
                    || (matrix.read(q, p).faer_abs() > threshold)
                {
                    failed = true;
                    let (j_left, j_right) = compute_2x2(
                        matrix.read(p, p),
                        matrix.read(p, q),
                        matrix.read(q, p),
                        matrix.read(q, q),
                    );

                    let (top, bottom) = matrix.rb_mut().split_at_row_mut(p);
                    j_left.apply_on_the_left_in_place(bottom.row_mut(0), top.row_mut(q));
                    let (left, right) = matrix.rb_mut().split_at_col_mut(p);
                    j_right.apply_on_the_right_in_place(right.col_mut(0), left.col_mut(q));

                    if let Some(u) = u.rb_mut() {
                        let (left, right) = u.split_at_col_mut(p);
                        j_left
                            .transpose()
                            .apply_on_the_right_in_place(right.col_mut(0), left.col_mut(q))
                    }
                    if let Some(v) = v.rb_mut() {
                        let (left, right) = v.split_at_col_mut(p);
                        j_right.apply_on_the_right_in_place(right.col_mut(0), left.col_mut(q))
                    }

                    for idx in [p, q] {
                        let d = matrix.read(idx, idx).faer_abs();
                        max_diag = if d > max_diag { d } else { max_diag };
                    }
                }
            }
        }
        if !failed {
            break;
        }
    }

    // make diagonal elements positive
    for j in 0..n {
        let d = matrix.read(j, j);
        if d < E::faer_zero() {
            matrix.write(j, j, d.faer_neg());
            if let Some(mut u) = u.rb_mut() {
                for i in 0..n {
                    u.write(i, j, u.read(i, j).faer_neg());
                }
            }
        }
    }

    // sort singular values and count nonzero ones
    let (start, new_n) = match skip {
        Skip::None => (0, n),
        Skip::First => (1, n - 1),
        Skip::Last => (0, n - 1),
    };

    let mut matrix = matrix.submatrix_mut(start, start, new_n, new_n);
    let mut u = u.map(|u| u.submatrix_mut(0, start, n, new_n));
    let mut v = v.map(|v| {
        let vn = v.nrows();
        v.submatrix_mut(0, start, vn, new_n)
    });

    let n = new_n;
    let mut nnz_count = n;
    for i in 0..n {
        let mut largest_elem = E::faer_zero();
        let mut largest_pos = i;

        for j in i..n {
            let mjj = matrix.read(j, j);
            (largest_elem, largest_pos) = if mjj > largest_elem {
                (mjj, j)
            } else {
                (largest_elem, largest_pos)
            };
        }

        if largest_elem == E::faer_zero() {
            nnz_count = i;
        }

        if largest_pos > i {
            let mii = matrix.read(i, i);
            matrix.write(i, i, largest_elem);
            matrix.write(largest_pos, largest_pos, mii);
            if let Some(u) = u.rb_mut() {
                swap_cols(u, i, largest_pos);
            }
            if let Some(v) = v.rb_mut() {
                swap_cols(v, i, largest_pos);
            }
        }
    }
    nnz_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, Mat, MatRef};
    use assert_approx_eq::assert_approx_eq;

    #[track_caller]
    fn check_svd(mat: MatRef<'_, f64>, u: MatRef<'_, f64>, v: MatRef<'_, f64>, s: MatRef<'_, f64>) {
        let m = mat.nrows();
        let n = mat.ncols();
        let reconstructed = u * s * v.transpose();

        for i in 0..m {
            for j in 0..n {
                if i == j {
                    assert!(s.read(i, j) >= 0.0);
                } else {
                    assert_approx_eq!(s.read(i, j), 0.0);
                }
            }
        }

        for o in [u * u.transpose(), v * v.transpose()] {
            let m = o.nrows();
            for i in 0..m {
                for j in 0..m {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(o.read(i, j), target);
                }
            }
        }
        for i in 0..m {
            for j in 0..n {
                assert_approx_eq!(reconstructed.read(i, j), mat.read(i, j));
            }
        }

        let size = m.min(n);
        if size > 1 {
            for i in 0..size - 1 {
                assert!(s.read(i, i) >= s.read(i + 1, i + 1));
            }
        }
    }

    #[test]
    fn test_jacobi() {
        for n in [0, 1, 2, 4, 8, 15, 16, 31, 32] {
            let mat = Mat::<f64>::from_fn(n, n, |_, _| rand::random::<f64>());

            let mut s = mat.clone();
            let mut u = Mat::<f64>::zeros(n, n);
            let mut v = Mat::<f64>::zeros(n, n);

            jacobi_svd(
                s.as_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                Skip::None,
                f64::EPSILON,
                f64::MIN_POSITIVE,
            );
            check_svd(mat.as_ref(), u.as_ref(), v.as_ref(), s.as_ref());
        }
    }

    #[test]
    fn test_skip_first() {
        for n in [2, 4, 8, 15, 16, 31, 32] {
            let mat = Mat::<f64>::from_fn(
                n,
                n,
                |_, j| if j == 0 { 0.0 } else { rand::random::<f64>() },
            );

            let mut s = mat.clone();
            let mut u = Mat::<f64>::zeros(n, n);
            let mut v = Mat::<f64>::zeros(n, n);

            jacobi_svd(
                s.as_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                Skip::First,
                f64::EPSILON,
                f64::MIN_POSITIVE,
            );
            let mut u_shifted = Mat::<f64>::zeros(n, n);
            for j in 1..n {
                for i in 0..n {
                    u_shifted.write(i, j - 1, u.read(i, j));
                }

                s.write(j - 1, j, s.read(j, j));
                s.write(j, j, 0.0);
            }
            for i in 0..n {
                u_shifted.write(i, n - 1, u.read(i, 0));
            }
            check_svd(
                mat.as_ref().submatrix(0, 1, n, n - 1),
                u_shifted.as_ref(),
                v.as_ref().submatrix(0, 1, n - 1, n - 1),
                s.as_ref().submatrix(0, 1, n, n - 1),
            );
        }
    }

    #[test]
    fn test_skip_last() {
        for n in [2, 4, 8, 15, 16, 31, 32] {
            let mat = Mat::<f64>::from_fn(n, n, |_, j| {
                if j == n - 1 {
                    0.0
                } else {
                    rand::random::<f64>()
                }
            });

            let mut s = mat.clone();
            let mut u = Mat::<f64>::zeros(n, n);
            let mut v = Mat::<f64>::zeros(n, n);

            jacobi_svd(
                s.as_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                Skip::Last,
                f64::EPSILON,
                f64::MIN_POSITIVE,
            );
            assert!(v.read(n - 1, n - 1) == 1.0);
            for j in 0..n - 1 {
                assert_approx_eq!(v.read(n - 1, j), 0.0);
                assert_approx_eq!(v.read(j, n - 1), 0.0);
            }
            check_svd(
                mat.as_ref().submatrix(0, 0, n, n - 1),
                u.as_ref(),
                v.as_ref().submatrix(0, 0, n - 1, n - 1),
                s.as_ref().submatrix(0, 0, n, n - 1),
            );
        }
    }

    #[test]
    fn eigen_286() {
        let mat = crate::mat![[-7.90884e-313, -4.94e-324], [0.0, 5.60844e-313]];
        let n = 2;
        let mut s = mat.clone();
        let mut u = Mat::<f64>::zeros(n, n);
        let mut v = Mat::<f64>::zeros(n, n);
        jacobi_svd(
            s.as_mut(),
            Some(u.as_mut()),
            Some(v.as_mut()),
            Skip::None,
            f64::EPSILON,
            f64::MIN_POSITIVE,
        );
        check_svd(mat.as_ref(), u.as_ref(), v.as_ref(), s.as_ref());
    }
}
