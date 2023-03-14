// Algorithm ported from Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2013-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use assert2::assert as fancy_assert;
use faer_core::{permutation::swap_cols, ColMut, MatMut, RealField, RowMut};
use reborrow::*;

#[derive(Copy, Clone, Debug)]
pub struct JacobiRotation<T> {
    pub c: T,
    pub s: T,
}

impl<T: RealField> JacobiRotation<T> {
    pub fn from_triplet(x: T, y: T, z: T) -> Self {
        let abs_y = y.abs();
        let two_abs_y = abs_y + abs_y;
        if two_abs_y == T::zero() {
            Self {
                c: T::one(),
                s: T::zero(),
            }
        } else {
            let tau = (x - z) / two_abs_y;
            let w = (tau * tau + T::one()).sqrt();
            let t = if tau > T::zero() {
                (tau + w).inv()
            } else {
                (tau - w).inv()
            };

            let neg_sign_y = if y > T::zero() { -T::one() } else { T::one() };
            let n = (t * t + T::one()).sqrt().inv();

            Self {
                c: n,
                s: neg_sign_y * t * n,
            }
        }
    }

    pub fn apply_on_the_left_2x2(&self, m00: T, m01: T, m10: T, m11: T) -> (T, T, T, T) {
        let Self { c, s } = *self;
        (
            m00 * c + m10 * s,
            m01 * c + m11 * s,
            -s * m00 + c * m10,
            -s * m01 + c * m11,
        )
    }

    pub fn apply_on_the_right_2x2(&self, m00: T, m01: T, m10: T, m11: T) -> (T, T, T, T) {
        let (r00, r01, r10, r11) = self.transpose().apply_on_the_left_2x2(m00, m10, m01, m11);
        (r00, r10, r01, r11)
    }

    pub fn apply_on_the_left_in_place(&self, x: RowMut<'_, T>, y: RowMut<'_, T>) {
        pulp::Arch::new().dispatch(
            #[inline(always)]
            move || {
                let Self { c, s } = *self;
                if c == T::one() && s == T::zero() {
                    return;
                }

                x.cwise().zip(y).for_each(move |x, y| {
                    let x_ = *x;
                    let y_ = *y;
                    *x = c * x_ + s * y_;
                    *y = -s * x_ + c * y_;
                });
            },
        )
    }

    pub fn apply_on_the_right_in_place(&self, x: ColMut<'_, T>, y: ColMut<'_, T>) {
        self.transpose()
            .apply_on_the_left_in_place(x.transpose(), y.transpose());
    }

    pub fn transpose(&self) -> Self {
        Self {
            c: self.c,
            s: -self.s,
        }
    }
}

impl<T: RealField> core::ops::Mul for JacobiRotation<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            c: self.c * rhs.c - self.s * rhs.s,
            s: self.c * rhs.s + self.s * rhs.c,
        }
    }
}

fn compute_2x2<T: RealField>(
    m00: T,
    m01: T,
    m10: T,
    m11: T,
) -> (JacobiRotation<T>, JacobiRotation<T>) {
    let t = m00 + m11;
    let d = m10 - m01;

    let rot1 = if d == T::zero() {
        JacobiRotation {
            c: T::one(),
            s: T::zero(),
        }
    } else {
        let u = t / d;
        let tmp = (T::one() + u * u).sqrt().inv();
        let tmp = if tmp == T::zero() { u.abs().inv() } else { tmp };
        JacobiRotation { c: u * tmp, s: tmp }
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

pub fn jacobi_svd<T: RealField>(
    matrix: MatMut<'_, T>,
    u: Option<MatMut<'_, T>>,
    v: Option<MatMut<'_, T>>,
    skip: Skip,
    epsilon: T,
    consider_zero_threshold: T,
) -> usize {
    fancy_assert!(matrix.nrows() == matrix.ncols());
    let n = matrix.nrows();

    if let Some(u) = u.rb() {
        fancy_assert!(n == u.nrows());
        fancy_assert!(n == u.ncols());
    };
    if let Some(v) = v.rb() {
        fancy_assert!(n == v.ncols());
    }

    let mut matrix = matrix;
    let mut u = u;
    let mut v = v;

    if let Some(mut u) = u.rb_mut() {
        for j in 0..n {
            for i in 0..j {
                u.rb_mut().write_at(i, j, T::zero());
            }
            u.rb_mut().write_at(j, j, T::one());
            for i in j + 1..n {
                u.rb_mut().write_at(i, j, T::zero());
            }
        }
    }

    if let Some(mut v) = v.rb_mut() {
        if matches!(skip, Skip::First) {
            for i in 0..n - 1 {
                v.rb_mut().write_at(i, 0, T::zero());
            }
            v = v.submatrix(0, 1, n - 1, n - 1);
        }

        let m = v.nrows();
        let n = v.ncols();
        for j in 0..n {
            for i in 0..j {
                v.rb_mut().write_at(i, j, T::zero());
            }
            if j == m {
                break;
            }
            v.rb_mut().write_at(j, j, T::one());
            for i in j + 1..m {
                v.rb_mut().write_at(i, j, T::zero());
            }
        }
    }

    let mut max_diag = T::zero();
    for d in matrix.rb().diagonal() {
        let d = d.abs();
        max_diag = if d > max_diag { d } else { max_diag };
    }

    let precision = epsilon + epsilon;
    loop {
        let mut failed = false;
        for p in 1..n {
            for q in 0..p {
                let threshold = precision * max_diag;
                let threshold = if threshold > consider_zero_threshold {
                    threshold
                } else {
                    consider_zero_threshold
                };

                if (matrix[(p, q)].abs() > threshold) || (matrix[(q, p)].abs() > threshold) {
                    failed = true;
                    let (j_left, j_right) = compute_2x2(
                        matrix[(p, p)],
                        matrix[(p, q)],
                        matrix[(q, p)],
                        matrix[(q, q)],
                    );

                    let (top, bottom) = matrix.rb_mut().split_at_row(p);
                    j_left.apply_on_the_left_in_place(bottom.row(0), top.row(q));
                    let (left, right) = matrix.rb_mut().split_at_col(p);
                    j_right.apply_on_the_right_in_place(right.col(0), left.col(q));

                    if let Some(u) = u.rb_mut() {
                        let (left, right) = u.split_at_col(p);
                        j_left
                            .transpose()
                            .apply_on_the_right_in_place(right.col(0), left.col(q))
                    }
                    if let Some(v) = v.rb_mut() {
                        let (left, right) = v.split_at_col(p);
                        j_right.apply_on_the_right_in_place(right.col(0), left.col(q))
                    }

                    for idx in [p, q] {
                        let d = matrix[(idx, idx)].abs();
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
        let d = matrix[(j, j)];
        if d < T::zero() {
            matrix[(j, j)] = -d;
            if let Some(mut u) = u.rb_mut() {
                for i in 0..n {
                    u[(i, j)] = -u[(i, j)];
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

    let mut matrix = matrix.submatrix(start, start, new_n, new_n);
    let mut u = u.map(|u| u.submatrix(0, start, n, new_n));
    let mut v = v.map(|v| {
        let vn = v.nrows();
        v.submatrix(0, start, vn, new_n)
    });

    let n = new_n;
    let mut nnz_count = n;
    for i in 0..n {
        let mut largest_elem = T::zero();
        let mut largest_pos = i;

        for j in i..n {
            let mjj = matrix[(j, j)];
            (largest_elem, largest_pos) = if mjj > largest_elem {
                (mjj, j)
            } else {
                (largest_elem, largest_pos)
            };
        }

        if largest_elem == T::zero() {
            nnz_count = i;
        }

        if largest_pos > i {
            let mii = matrix[(i, i)];
            matrix[(i, i)] = largest_elem;
            matrix[(largest_pos, largest_pos)] = mii;
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
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{Mat, MatRef};

    fn check_svd(mat: MatRef<'_, f64>, u: MatRef<'_, f64>, v: MatRef<'_, f64>, s: MatRef<'_, f64>) {
        let m = mat.nrows();
        let n = mat.ncols();
        let reconstructed = u * s * v.transpose();

        for i in 0..m {
            for j in 0..n {
                if i == j {
                    fancy_assert!(s[(i, j)] >= 0.0);
                } else {
                    assert_approx_eq!(s[(i, j)], 0.0);
                }
            }
        }

        for o in [u * u.transpose(), v * v.transpose()] {
            let m = o.nrows();
            for i in 0..m {
                for j in 0..m {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(o[(i, j)], target);
                }
            }
        }
        for i in 0..m {
            for j in 0..n {
                assert_approx_eq!(reconstructed[(i, j)], mat[(i, j)]);
            }
        }

        let size = m.min(n);
        if size > 1 {
            for i in 0..size - 1 {
                fancy_assert!(s[(i, i)] >= s[(i + 1, i + 1)]);
            }
        }
    }

    #[test]
    fn test_jacobi() {
        for n in [0, 1, 2, 4, 8, 15, 16, 31, 32] {
            let mat = Mat::<f64>::with_dims(|_, _| rand::random::<f64>(), n, n);

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
            let mat = Mat::<f64>::with_dims(
                |_, j| if j == 0 { 0.0 } else { rand::random::<f64>() },
                n,
                n,
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
            fancy_assert!(v[(0, 0)] == 1.0);
            for j in 1..n {
                assert_approx_eq!(v[(0, j)], 0.0);
                assert_approx_eq!(v[(j, 0)], 0.0);
            }
            let mut u_shifted = Mat::<f64>::zeros(n, n);
            for j in 1..n {
                for i in 0..n {
                    u_shifted[(i, j - 1)] = u[(i, j)];
                }

                s[(j - 1, j)] = s[(j, j)];
                s[(j, j)] = 0.0;
            }
            for i in 0..n {
                u_shifted[(i, n - 1)] = u[(i, 0)];
            }
            check_svd(
                mat.as_ref().submatrix(0, 1, n, n - 1),
                u_shifted.as_ref(),
                v.as_ref().submatrix(1, 1, n - 1, n - 1),
                s.as_ref().submatrix(0, 1, n, n - 1),
            );
        }
    }

    #[test]
    fn test_skip_last() {
        for n in [2, 4, 8, 15, 16, 31, 32] {
            let mat = Mat::<f64>::with_dims(
                |_, j| {
                    if j == n - 1 {
                        0.0
                    } else {
                        rand::random::<f64>()
                    }
                },
                n,
                n,
            );

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
            fancy_assert!(v[(n - 1, n - 1)] == 1.0);
            for j in 0..n - 1 {
                assert_approx_eq!(v[(n - 1, j)], 0.0);
                assert_approx_eq!(v[(j, n - 1)], 0.0);
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
        let mat = faer_core::mat![[-7.90884e-313, -4.94e-324], [0.0, 5.60844e-313]];
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
