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
    assert, perm::swap_cols_idx as swap_cols, unzipped, utils::simd::*, utils::slice::*, zipped,
    MatMut, RealField,
};
use faer_entity::{pulp, SimdCtx, SimdGroupFor};
use reborrow::*;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct JacobiRotation<T> {
    pub c: T,
    pub s: T,
}

unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for JacobiRotation<T> {}
unsafe impl<T: bytemuck::Pod> bytemuck::Pod for JacobiRotation<T> {}

impl<E: RealField> JacobiRotation<E> {
    #[inline]
    pub fn make_givens(p: E, q: E) -> Self {
        if q == E::faer_zero() {
            Self {
                c: if p < E::faer_zero() {
                    E::faer_one().faer_neg()
                } else {
                    E::faer_one()
                },
                s: E::faer_zero(),
            }
        } else if p == E::faer_zero() {
            Self {
                c: E::faer_zero(),
                s: if q < E::faer_zero() {
                    E::faer_one().faer_neg()
                } else {
                    E::faer_one()
                },
            }
        } else if p.faer_abs() > q.faer_abs() {
            let t = q.faer_div(p);
            let mut u = E::faer_one().faer_add(t.faer_abs2()).faer_sqrt();
            if p < E::faer_zero() {
                u = u.faer_neg();
            }
            let c = u.faer_inv();
            let s = t.faer_neg().faer_mul(c);

            Self { c, s }
        } else {
            let t = p.faer_div(q);
            let mut u = E::faer_one().faer_add(t.faer_abs2()).faer_sqrt();
            if q < E::faer_zero() {
                u = u.faer_neg();
            }
            let s = u.faer_inv().faer_neg();
            let c = t.faer_neg().faer_mul(s);

            Self { c, s }
        }
    }

    #[inline]
    pub fn from_triplet(x: E, y: E, z: E) -> Self {
        let abs_y = y.faer_abs();
        let two_abs_y = abs_y.faer_add(abs_y);
        if two_abs_y == E::faer_zero() {
            Self {
                c: E::faer_one(),
                s: E::faer_zero(),
            }
        } else {
            let tau = (x.faer_sub(z)).faer_mul(two_abs_y.faer_inv());
            let w = ((tau.faer_mul(tau)).faer_add(E::faer_one())).faer_sqrt();
            let t = if tau > E::faer_zero() {
                (tau.faer_add(w)).faer_inv()
            } else {
                (tau.faer_sub(w)).faer_inv()
            };

            let neg_sign_y = if y > E::faer_zero() {
                E::faer_one().faer_neg()
            } else {
                E::faer_one()
            };
            let n = (t.faer_mul(t).faer_add(E::faer_one()))
                .faer_sqrt()
                .faer_inv();

            Self {
                c: n,
                s: neg_sign_y.faer_mul(t).faer_mul(n),
            }
        }
    }

    #[inline]
    pub fn apply_on_the_left_2x2(&self, m00: E, m01: E, m10: E, m11: E) -> (E, E, E, E) {
        let Self { c, s } = *self;
        (
            m00.faer_mul(c).faer_add(m10.faer_mul(s)),
            m01.faer_mul(c).faer_add(m11.faer_mul(s)),
            s.faer_neg().faer_mul(m00).faer_add(c.faer_mul(m10)),
            s.faer_neg().faer_mul(m01).faer_add(c.faer_mul(m11)),
        )
    }

    #[inline]
    pub fn apply_on_the_right_2x2(&self, m00: E, m01: E, m10: E, m11: E) -> (E, E, E, E) {
        let (r00, r01, r10, r11) = self.transpose().apply_on_the_left_2x2(m00, m10, m01, m11);
        (r00, r10, r01, r11)
    }

    #[inline]
    pub fn apply_on_the_left_in_place(&self, x: MatMut<'_, E>, y: MatMut<'_, E>) {
        self.apply_on_the_left_in_place_arch(E::Simd::default(), x, y);
    }

    #[inline(never)]
    fn apply_on_the_left_in_place_fallback(&self, x: MatMut<'_, E>, y: MatMut<'_, E>) {
        let Self { c, s } = *self;
        zipped!(x, y).for_each(move |unzipped!(mut x, mut y)| {
            let x_ = x.read();
            let y_ = y.read();
            x.write(c.faer_mul(x_).faer_add(s.faer_mul(y_)));
            y.write(s.faer_neg().faer_mul(x_).faer_add(c.faer_mul(y_)));
        });
    }

    #[inline(always)]
    pub fn apply_on_the_right_in_place_with_simd_and_offset<S: pulp::Simd>(
        &self,
        simd: S,
        offset: pulp::Offset<E::SimdMask<S>>,
        x: MatMut<'_, E>,
        y: MatMut<'_, E>,
    ) {
        self.transpose()
            .apply_on_the_left_in_place_with_simd_and_offset(
                simd,
                offset,
                x.transpose_mut(),
                y.transpose_mut(),
            );
    }

    #[inline(always)]
    pub fn apply_on_the_left_in_place_with_simd_and_offset<S: pulp::Simd>(
        &self,
        simd: S,
        offset: pulp::Offset<E::SimdMask<S>>,
        x: MatMut<'_, E>,
        y: MatMut<'_, E>,
    ) {
        let Self { c, s } = *self;
        assert!(all(x.nrows() == 1, y.nrows() == 1, x.ncols() == y.ncols()));

        if c == E::faer_one() && s == E::faer_zero() {
            return;
        }

        if x.col_stride() != 1 || y.col_stride() != 1 {
            self.apply_on_the_left_in_place_fallback(x, y);
            return;
        }

        let simd = SimdFor::<E, S>::new(simd);

        let x = SliceGroupMut::<'_, E>::new(x.transpose_mut().try_get_contiguous_col_mut(0));
        let y = SliceGroupMut::<'_, E>::new(y.transpose_mut().try_get_contiguous_col_mut(0));

        let c = simd.splat(c);
        let s = simd.splat(s);

        let (x_head, x_body, x_tail) = simd.as_aligned_simd_mut(x, offset);
        let (y_head, y_body, y_tail) = simd.as_aligned_simd_mut(y, offset);

        #[inline(always)]
        fn process<E: RealField, S: pulp::Simd>(
            simd: SimdFor<E, S>,
            mut x: impl Write<Output = SimdGroupFor<E, S>>,
            mut y: impl Write<Output = SimdGroupFor<E, S>>,
            c: SimdGroupFor<E, S>,
            s: SimdGroupFor<E, S>,
        ) {
            let zero = simd.splat(E::faer_zero());
            let x_ = x.read_or(zero);
            let y_ = y.read_or(zero);
            x.write(simd.mul_add_e(c, x_, simd.mul(s, y_)));
            y.write(simd.mul_add_e(c, y_, simd.neg(simd.mul(s, x_))));
        }

        process(simd, x_head, y_head, c, s);
        for (x, y) in x_body.into_mut_iter().zip(y_body.into_mut_iter()) {
            process(simd, x, y, c, s);
        }
        process(simd, x_tail, y_tail, c, s);
    }

    #[inline]
    pub fn apply_on_the_left_in_place_arch(
        &self,
        arch: E::Simd,
        x: MatMut<'_, E>,
        y: MatMut<'_, E>,
    ) {
        struct ApplyOnLeft<'a, E: RealField> {
            c: E,
            s: E,
            x: MatMut<'a, E>,
            y: MatMut<'a, E>,
        }

        impl<E: RealField> pulp::WithSimd for ApplyOnLeft<'_, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self { x, y, c, s } = self;
                assert!(all(x.nrows() == 1, y.nrows() == 1, x.ncols() == y.ncols()));

                if c == E::faer_one() && s == E::faer_zero() {
                    return;
                }

                let simd = SimdFor::<E, S>::new(simd);

                let x =
                    SliceGroupMut::<'_, E>::new(x.transpose_mut().try_get_contiguous_col_mut(0));
                let y =
                    SliceGroupMut::<'_, E>::new(y.transpose_mut().try_get_contiguous_col_mut(0));

                let offset = simd.align_offset(x.rb());

                let c = simd.splat(c);
                let s = simd.splat(s);

                let (x_head, x_body, x_tail) = simd.as_aligned_simd_mut(x, offset);
                let (y_head, y_body, y_tail) = simd.as_aligned_simd_mut(y, offset);

                #[inline(always)]
                fn process<E: RealField, S: pulp::Simd>(
                    simd: SimdFor<E, S>,
                    mut x: impl Write<Output = SimdGroupFor<E, S>>,
                    mut y: impl Write<Output = SimdGroupFor<E, S>>,
                    c: SimdGroupFor<E, S>,
                    s: SimdGroupFor<E, S>,
                ) {
                    let zero = simd.splat(E::faer_zero());
                    let x_ = x.read_or(zero);
                    let y_ = y.read_or(zero);
                    x.write(simd.mul_add_e(c, x_, simd.mul(s, y_)));
                    y.write(simd.mul_add_e(c, y_, simd.neg(simd.mul(s, x_))));
                }

                process(simd, x_head, y_head, c, s);
                for (x, y) in x_body.into_mut_iter().zip(y_body.into_mut_iter()) {
                    process(simd, x, y, c, s);
                }
                process(simd, x_tail, y_tail, c, s);
            }
        }

        let Self { c, s } = *self;

        let mut x = x;
        let mut y = y;

        if x.col_stride() == 1 && y.col_stride() == 1 {
            arch.dispatch(ApplyOnLeft::<'_, E> { c, s, x, y });
        } else {
            zipped!(x, y).for_each(move |unzipped!(mut x, mut y)| {
                let x_ = x.read();
                let y_ = y.read();
                x.write(c.faer_mul(x_).faer_add(s.faer_mul(y_)));
                y.write(s.faer_neg().faer_mul(x_).faer_add(c.faer_mul(y_)));
            });
        }
    }

    #[inline]
    pub fn apply_on_the_right_in_place(&self, x: MatMut<'_, E>, y: MatMut<'_, E>) {
        self.transpose()
            .apply_on_the_left_in_place(x.transpose_mut(), y.transpose_mut());
    }

    #[inline]
    pub fn apply_on_the_right_in_place_arch(
        &self,
        arch: E::Simd,
        x: MatMut<'_, E>,
        y: MatMut<'_, E>,
    ) {
        self.transpose().apply_on_the_left_in_place_arch(
            arch,
            x.transpose_mut(),
            y.transpose_mut(),
        );
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        Self {
            c: self.c,
            s: self.s.faer_neg(),
        }
    }
}

impl<E: RealField> core::ops::Mul for JacobiRotation<E> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            c: self.c.faer_mul(rhs.c).faer_sub(self.s.faer_mul(rhs.s)),
            s: self.c.faer_mul(rhs.s).faer_add(self.s.faer_mul(rhs.c)),
        }
    }
}

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
                    j_left.apply_on_the_left_in_place(
                        bottom.row_mut(0).as_2d_mut(),
                        top.row_mut(q).as_2d_mut(),
                    );
                    let (left, right) = matrix.rb_mut().split_at_col_mut(p);
                    j_right.apply_on_the_right_in_place(
                        right.col_mut(0).as_2d_mut(),
                        left.col_mut(q).as_2d_mut(),
                    );

                    if let Some(u) = u.rb_mut() {
                        let (left, right) = u.split_at_col_mut(p);
                        j_left.transpose().apply_on_the_right_in_place(
                            right.col_mut(0).as_2d_mut(),
                            left.col_mut(q).as_2d_mut(),
                        )
                    }
                    if let Some(v) = v.rb_mut() {
                        let (left, right) = v.split_at_col_mut(p);
                        j_right.apply_on_the_right_in_place(
                            right.col_mut(0).as_2d_mut(),
                            left.col_mut(q).as_2d_mut(),
                        )
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
