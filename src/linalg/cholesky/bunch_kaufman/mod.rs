//! The Bunch Kaufman decomposition of a hermitian matrix $A$ is such that:
//! $$P A P^\top = LBL^H,$$
//! where $B$ is a block diagonal matrix, with $1\times 1$ or $2 \times 2 $ diagonal blocks, and
//! $L$ is a unit lower triangular matrix.

use crate::{
    linalg::{
        matmul::triangular::{self, BlockStructure},
        temp_mat_req, temp_mat_uninit,
        triangular_solve::{
            solve_unit_lower_triangular_in_place_with_conj,
            solve_unit_upper_triangular_in_place_with_conj,
        },
    },
    perm::{permute_rows, swap_cols_idx as swap_cols, swap_rows_idx as swap_rows, PermRef},
    unzipped, zipped, Conj, Index, MatMut, MatRef, Parallelism, SignedIndex,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_entity::{ComplexField, Entity, RealField};
use reborrow::*;

/// Computing the decomposition.
pub mod compute {
    use super::*;
    use crate::assert;

    /// Pivoting strategy for choosing the pivots.
    #[derive(Copy, Clone)]
    #[non_exhaustive]
    pub enum PivotingStrategy {
        /// Diagonal pivoting.
        Diagonal,
    }

    /// Tuning parameters for the decomposition.
    #[derive(Copy, Clone)]
    #[non_exhaustive]
    pub struct BunchKaufmanParams {
        /// Pivoting strategy.
        pub pivoting: PivotingStrategy,
        /// Block size of the algorithm.
        pub blocksize: usize,
    }

    /// Dynamic Bunch-Kaufman regularization.
    /// Values below `epsilon` in absolute value, or with the wrong sign are set to `delta` with
    /// their corrected sign.
    #[derive(Debug)]
    pub struct BunchKaufmanRegularization<'a, E: ComplexField> {
        /// Expected signs for the diagonal at each step of the decomposition.
        pub dynamic_regularization_signs: Option<&'a mut [i8]>,
        /// Regularized value.
        pub dynamic_regularization_delta: E::Real,
        /// Regularization threshold.
        pub dynamic_regularization_epsilon: E::Real,
    }

    impl<E: ComplexField> Default for BunchKaufmanRegularization<'_, E> {
        fn default() -> Self {
            Self {
                dynamic_regularization_signs: None,
                dynamic_regularization_delta: E::Real::faer_zero(),
                dynamic_regularization_epsilon: E::Real::faer_zero(),
            }
        }
    }

    impl Default for BunchKaufmanParams {
        fn default() -> Self {
            Self {
                pivoting: PivotingStrategy::Diagonal,
                blocksize: 64,
            }
        }
    }

    fn best_score_idx<E: ComplexField>(a: MatRef<'_, E>) -> Option<(usize, usize, E::Real)> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return None;
        }

        let mut best_row = 0usize;
        let mut best_col = 0usize;
        let mut best_score = E::Real::faer_zero();

        for j in 0..n {
            for i in 0..m {
                let score = a.read(i, j).faer_abs();
                if score > best_score {
                    best_row = i;
                    best_col = j;
                    best_score = score;
                }
            }
        }

        Some((best_row, best_col, best_score))
    }

    fn assign_col<E: ComplexField>(a: MatMut<'_, E>, i: usize, j: usize) {
        if i < j {
            let (ai, aj) = a.subcols_mut(i, j - i + 1).split_at_col_mut(1);
            ai.col_mut(0).copy_from(aj.rb().col(j - i - 1));
        } else if j < i {
            let (aj, ai) = a.subcols_mut(j, i - j + 1).split_at_col_mut(1);
            ai.col_mut(i - j - 1).copy_from(aj.rb().col(0));
        }
    }

    fn best_score<E: ComplexField>(a: MatRef<'_, E>) -> E::Real {
        let m = a.nrows();
        let n = a.ncols();

        let mut best_score = E::Real::faer_zero();

        for j in 0..n {
            for i in 0..m {
                let score = a.read(i, j).faer_abs();
                if score > best_score {
                    best_score = score;
                }
            }
        }

        best_score
    }

    #[inline(always)]
    fn max<E: RealField>(a: E, b: E) -> E {
        if a > b {
            a
        } else {
            b
        }
    }

    fn swap_elems_conj<E: ComplexField>(
        a: MatMut<'_, E>,
        i0: usize,
        j0: usize,
        i1: usize,
        j1: usize,
    ) {
        let mut a = a;
        let tmp = a.read(i0, j0).faer_conj();
        a.write(i0, j0, a.read(i1, j1).faer_conj());
        a.write(i1, j1, tmp);
    }
    fn swap_elems<E: ComplexField>(a: MatMut<'_, E>, i0: usize, j0: usize, i1: usize, j1: usize) {
        let mut a = a;
        let tmp = a.read(i0, j0);
        a.write(i0, j0, a.read(i1, j1));
        a.write(i1, j1, tmp);
    }

    fn cholesky_diagonal_pivoting_blocked_step<I: Index, E: ComplexField>(
        mut a: MatMut<'_, E>,
        regularization: BunchKaufmanRegularization<'_, E>,
        mut w: MatMut<'_, E>,
        pivots: &mut [I],
        alpha: E::Real,
        parallelism: Parallelism,
    ) -> (usize, usize, usize) {
        assert!(a.nrows() == a.ncols());
        let n = a.nrows();
        let nb = w.ncols();
        assert!(nb < n);
        if n == 0 {
            return (0, 0, 0);
        }

        let eps = regularization.dynamic_regularization_epsilon.faer_abs();
        let delta = regularization.dynamic_regularization_delta.faer_abs();
        let mut signs = regularization.dynamic_regularization_signs;
        let has_eps = delta > E::Real::faer_zero();
        let mut dynamic_regularization_count = 0usize;
        let mut pivot_count = 0usize;

        let truncate = <I::Signed as SignedIndex>::truncate;

        let mut k = 0;
        while k < n && k + 1 < nb {
            let make_real = |mut mat: MatMut<'_, E>, i, j| {
                mat.write(i, j, E::faer_from_real(mat.read(i, j).faer_real()))
            };

            w.rb_mut()
                .subrows_mut(k, n - k)
                .col_mut(k)
                .copy_from(a.rb().subrows(k, n - k).col(k));

            let (w_left, w_right) = w
                .rb_mut()
                .submatrix_mut(k, 0, n - k, k + 1)
                .split_at_col_mut(k);
            let w_row = w_left.rb().row(0);
            let w_col = w_right.col_mut(0);
            crate::linalg::matmul::matmul(
                w_col.as_2d_mut(),
                a.rb().submatrix(k, 0, n - k, k),
                w_row.rb().transpose().as_2d(),
                Some(E::faer_one()),
                E::faer_one().faer_neg(),
                parallelism,
            );
            make_real(w.rb_mut(), k, k);

            let mut k_step = 1;

            let abs_akk = w.read(k, k).faer_real().faer_abs();
            let imax;
            let colmax;

            if k + 1 < n {
                (imax, _, colmax) =
                    best_score_idx(w.rb().col(k).as_2d().subrows(k + 1, n - k - 1)).unwrap();
            } else {
                imax = 0;
                colmax = E::Real::faer_zero();
            }
            let imax = imax + k + 1;

            let kp;
            if max(abs_akk, colmax) == E::Real::faer_zero() {
                kp = k;

                let mut d11 = w.read(k, k).faer_real();
                if has_eps {
                    if let Some(signs) = signs.rb_mut() {
                        if signs[k] > 0 && d11 <= eps {
                            d11 = delta;
                            dynamic_regularization_count += 1;
                        } else if signs[k] < 0 && d11 >= eps.faer_neg() {
                            d11 = delta.faer_neg();
                            dynamic_regularization_count += 1;
                        }
                    }
                }
                let d11 = d11.faer_inv();
                a.write(k, k, E::faer_from_real(d11));
            } else {
                if abs_akk >= colmax.faer_mul(alpha) {
                    kp = k;
                } else {
                    zipped!(
                        w.rb_mut()
                            .subrows_mut(k, imax - k)
                            .col_mut(k + 1)
                            .as_2d_mut(),
                        a.rb().row(imax).subcols(k, imax - k).transpose().as_2d(),
                    )
                    .for_each(|unzipped!(mut dst, src)| dst.write(src.read().faer_conj()));

                    w.rb_mut()
                        .subrows_mut(imax, n - imax)
                        .col_mut(k + 1)
                        .copy_from(a.rb().subrows(imax, n - imax).col(imax));

                    let (w_left, w_right) = w
                        .rb_mut()
                        .submatrix_mut(k, 0, n - k, nb)
                        .split_at_col_mut(k + 1);
                    let w_row = w_left.rb().row(imax - k).subcols(0, k);
                    let w_col = w_right.col_mut(0);

                    crate::linalg::matmul::matmul(
                        w_col.as_2d_mut(),
                        a.rb().submatrix(k, 0, n - k, k),
                        w_row.rb().transpose().as_2d(),
                        Some(E::faer_one()),
                        E::faer_one().faer_neg(),
                        parallelism,
                    );
                    make_real(w.rb_mut(), imax, k + 1);

                    let rowmax = max(
                        best_score(w.rb().subrows(k, imax - k).col(k + 1).as_2d()),
                        best_score(w.rb().subrows(imax + 1, n - imax - 1).col(k + 1).as_2d()),
                    );

                    if abs_akk >= alpha.faer_mul(colmax).faer_mul(colmax.faer_div(rowmax)) {
                        kp = k;
                    } else if w.read(imax, k + 1).faer_real().faer_abs() >= alpha.faer_mul(rowmax) {
                        kp = imax;
                        assign_col(w.rb_mut().subrows_mut(k, n - k), k, k + 1);
                    } else {
                        kp = imax;
                        k_step = 2;
                    }
                }

                let kk = k + k_step - 1;

                if kp != kk {
                    pivot_count += 1;
                    if let Some(signs) = signs.rb_mut() {
                        signs.swap(kp, kk);
                    }
                    a.write(kp, kp, a.read(kk, kk));
                    for j in kk + 1..kp {
                        a.write(kp, j, a.read(j, kk).faer_conj());
                    }
                    assign_col(a.rb_mut().subrows_mut(kp + 1, n - kp - 1), kp, kk);

                    swap_rows(a.rb_mut().subcols_mut(0, k), kk, kp);
                    swap_rows(w.rb_mut().subcols_mut(0, kk + 1), kk, kp);
                }

                if k_step == 1 {
                    a.rb_mut()
                        .subrows_mut(k, n - k)
                        .col_mut(k)
                        .copy_from(w.rb().subrows(k, n - k).col(k));

                    let mut d11 = w.read(k, k).faer_real();
                    if has_eps {
                        if let Some(signs) = signs.rb_mut() {
                            if signs[k] > 0 && d11 <= eps {
                                d11 = delta;
                                dynamic_regularization_count += 1;
                            } else if signs[k] < 0 && d11 >= eps.faer_neg() {
                                d11 = delta.faer_neg();
                                dynamic_regularization_count += 1;
                            }
                        } else if d11.faer_abs() <= eps {
                            if d11 < E::Real::faer_zero() {
                                d11 = delta.faer_neg();
                            } else {
                                d11 = delta;
                            }
                            dynamic_regularization_count += 1;
                        }
                    }
                    let d11 = d11.faer_inv();
                    a.write(k, k, E::faer_from_real(d11));

                    let x = a.rb_mut().subrows_mut(k + 1, n - k - 1).col_mut(k);
                    zipped!(x.as_2d_mut())
                        .for_each(|unzipped!(mut x)| x.write(x.read().faer_scale_real(d11)));
                    zipped!(w
                        .rb_mut()
                        .subrows_mut(k + 1, n - k - 1)
                        .col_mut(k)
                        .as_2d_mut())
                    .for_each(|unzipped!(mut x)| x.write(x.read().faer_conj()));
                } else {
                    let d21 = w.read(k + 1, k).faer_abs();
                    let d21_inv = d21.faer_inv();
                    let mut d11 = d21_inv.faer_scale_real(w.read(k + 1, k + 1).faer_real());
                    let mut d22 = d21_inv.faer_scale_real(w.read(k, k).faer_real());

                    let eps = eps.faer_mul(d21_inv);
                    let delta = delta.faer_mul(d21_inv);
                    if has_eps {
                        if let Some(signs) = signs.rb_mut() {
                            if signs[k] > 0 && signs[k + 1] > 0 {
                                if d11 <= eps {
                                    d11 = delta;
                                }
                                if d22 <= eps {
                                    d22 = delta;
                                }
                            } else if signs[k] < 0 && signs[k + 1] < 0 {
                                if d11 >= eps.faer_neg() {
                                    d11 = delta.faer_neg();
                                }
                                if d22 >= eps.faer_neg() {
                                    d22 = delta.faer_neg();
                                }
                            }
                        }
                    }

                    // t = (d11/|d21| * d22/|d21| - 1.0)
                    let mut t = d11.faer_mul(d22).faer_sub(E::Real::faer_one());
                    if has_eps {
                        if let Some(signs) = signs.rb_mut() {
                            if ((signs[k] > 0 && signs[k + 1] > 0)
                                || (signs[k] < 0 && signs[k + 1] < 0))
                                && t <= eps
                            {
                                t = delta;
                            } else if ((signs[k] > 0 && signs[k + 1] < 0)
                                || (signs[k] < 0 && signs[k + 1] > 0))
                                && t >= eps.faer_neg()
                            {
                                t = delta.faer_neg();
                            }
                        }
                    }

                    let t = t.faer_inv();
                    let d21 = w.read(k + 1, k).faer_scale_real(d21_inv);
                    let d = t.faer_mul(d21_inv);

                    a.write(k, k, E::faer_from_real(d11.faer_mul(d)));
                    a.write(k + 1, k, d21.faer_scale_real(d.faer_neg()));
                    a.write(k + 1, k + 1, E::faer_from_real(d22.faer_mul(d)));

                    for j in k + 2..n {
                        let wk = (w
                            .read(j, k)
                            .faer_scale_real(d11)
                            .faer_sub(w.read(j, k + 1).faer_mul(d21)))
                        .faer_scale_real(d);
                        let wkp1 = (w
                            .read(j, k + 1)
                            .faer_scale_real(d22)
                            .faer_sub(w.read(j, k).faer_mul(d21.faer_conj())))
                        .faer_scale_real(d);

                        a.write(j, k, wk);
                        a.write(j, k + 1, wkp1);
                    }

                    zipped!(w
                        .rb_mut()
                        .subrows_mut(k + 1, n - k - 1)
                        .col_mut(k)
                        .as_2d_mut())
                    .for_each(|unzipped!(mut x)| x.write(x.read().faer_conj()));
                    zipped!(w
                        .rb_mut()
                        .subrows_mut(k + 2, n - k - 2)
                        .col_mut(k + 1)
                        .as_2d_mut())
                    .for_each(|unzipped!(mut x)| x.write(x.read().faer_conj()));
                }
            }

            if k_step == 1 {
                pivots[k] = I::from_signed(truncate(kp));
            } else {
                pivots[k] = I::from_signed(truncate(!kp));
                pivots[k + 1] = I::from_signed(truncate(!kp));
            }

            k += k_step;
        }

        let (a_left, mut a_right) = a.rb_mut().subrows_mut(k, n - k).split_at_col_mut(k);
        triangular::matmul(
            a_right.rb_mut(),
            BlockStructure::TriangularLower,
            a_left.rb(),
            BlockStructure::Rectangular,
            w.rb().submatrix(k, 0, n - k, k).transpose(),
            BlockStructure::Rectangular,
            Some(E::faer_one()),
            E::faer_one().faer_neg(),
            parallelism,
        );

        zipped!(a_right.diagonal_mut().column_vector_mut().as_2d_mut())
            .for_each(|unzipped!(mut x)| x.write(E::faer_from_real(x.read().faer_real())));

        let mut j = k - 1;
        loop {
            let jj = j;
            let mut jp = pivots[j].to_signed().sx();
            if (jp as isize) < 0 {
                jp = !jp;
                j -= 1;
            }

            if j == 0 {
                return (k, pivot_count, dynamic_regularization_count);
            }
            j -= 1;

            if jp != jj {
                swap_rows(a.rb_mut().subcols_mut(0, j + 1), jp, jj);
            }
            if j == 0 {
                return (k, pivot_count, dynamic_regularization_count);
            }
        }
    }

    fn cholesky_diagonal_pivoting_unblocked<I: Index, E: ComplexField>(
        mut a: MatMut<'_, E>,
        regularization: BunchKaufmanRegularization<'_, E>,
        pivots: &mut [I],
        alpha: E::Real,
    ) -> (usize, usize) {
        let truncate = <I::Signed as SignedIndex>::truncate;

        assert!(a.nrows() == a.ncols());
        let n = a.nrows();
        if n == 0 {
            return (0, 0);
        }

        let eps = regularization.dynamic_regularization_epsilon.faer_abs();
        let delta = regularization.dynamic_regularization_delta.faer_abs();
        let mut signs = regularization.dynamic_regularization_signs;
        let has_eps = delta > E::Real::faer_zero();
        let mut dynamic_regularization_count = 0usize;
        let mut pivot_count = 0usize;

        let mut k = 0;
        while k < n {
            let make_real = |mut mat: MatMut<'_, E>, i, j| {
                mat.write(i, j, E::faer_from_real(mat.read(i, j).faer_real()))
            };

            let mut k_step = 1;

            let abs_akk = a.read(k, k).faer_abs();
            let imax;
            let colmax;

            if k + 1 < n {
                (imax, _, colmax) =
                    best_score_idx(a.rb().col(k).subrows(k + 1, n - k - 1).as_2d()).unwrap();
            } else {
                imax = 0;
                colmax = E::Real::faer_zero();
            }
            let imax = imax + k + 1;

            let kp;
            if max(abs_akk, colmax) == E::Real::faer_zero() {
                kp = k;

                let mut d11 = a.read(k, k).faer_real();
                if has_eps {
                    if let Some(signs) = signs.rb_mut() {
                        if signs[k] > 0 && d11 <= eps {
                            d11 = delta;
                            dynamic_regularization_count += 1;
                        } else if signs[k] < 0 && d11 >= eps.faer_neg() {
                            d11 = delta.faer_neg();
                            dynamic_regularization_count += 1;
                        }
                    }
                }
                let d11 = d11.faer_inv();
                a.write(k, k, E::faer_from_real(d11));
            } else {
                if abs_akk >= colmax.faer_mul(alpha) {
                    kp = k;
                } else {
                    let rowmax = max(
                        best_score(a.rb().row(imax).subcols(k, imax - k).as_2d()),
                        best_score(a.rb().subrows(imax + 1, n - imax - 1).col(imax).as_2d()),
                    );

                    if abs_akk >= alpha.faer_mul(colmax).faer_mul(colmax.faer_div(rowmax)) {
                        kp = k;
                    } else if a.read(imax, imax).faer_abs() >= alpha.faer_mul(rowmax) {
                        kp = imax
                    } else {
                        kp = imax;
                        k_step = 2;
                    }
                }

                let kk = k + k_step - 1;

                if kp != kk {
                    pivot_count += 1;
                    swap_cols(a.rb_mut().subrows_mut(kp + 1, n - kp - 1), kk, kp);
                    for j in kk + 1..kp {
                        swap_elems_conj(a.rb_mut(), j, kk, kp, j);
                    }

                    a.write(kp, kk, a.read(kp, kk).faer_conj());
                    swap_elems(a.rb_mut(), kk, kk, kp, kp);

                    if k_step == 2 {
                        swap_elems(a.rb_mut(), k + 1, k, kp, k);
                    }
                }

                if k_step == 1 {
                    let mut d11 = a.read(k, k).faer_real();
                    if has_eps {
                        if let Some(signs) = signs.rb_mut() {
                            if signs[k] > 0 && d11 <= eps {
                                d11 = delta;
                                dynamic_regularization_count += 1;
                            } else if signs[k] < 0 && d11 >= eps.faer_neg() {
                                d11 = delta.faer_neg();
                                dynamic_regularization_count += 1;
                            }
                        } else if d11.faer_abs() <= eps {
                            if d11 < E::Real::faer_zero() {
                                d11 = delta.faer_neg();
                            } else {
                                d11 = delta;
                            }
                            dynamic_regularization_count += 1;
                        }
                    }
                    let d11 = d11.faer_inv();
                    a.write(k, k, E::faer_from_real(d11));

                    let (x, mut trailing) = a
                        .rb_mut()
                        .subrows_mut(k + 1, n - k - 1)
                        .subcols_mut(k, n - k)
                        .split_at_col_mut(1);

                    for j in 0..n - k - 1 {
                        let d11xj = x.read(j, 0).faer_conj().faer_scale_real(d11);
                        for i in j..n - k - 1 {
                            let xi = x.read(i, 0);
                            trailing.write(i, j, trailing.read(i, j).faer_sub(d11xj.faer_mul(xi)));
                        }
                        make_real(trailing.rb_mut(), j, j);
                    }
                    zipped!(x).for_each(|unzipped!(mut x)| x.write(x.read().faer_scale_real(d11)));
                } else {
                    let d21 = a.read(k + 1, k).faer_abs();
                    let d21_inv = d21.faer_inv();
                    let mut d11 = d21_inv.faer_scale_real(a.read(k + 1, k + 1).faer_real());
                    let mut d22 = d21_inv.faer_scale_real(a.read(k, k).faer_real());

                    let eps = eps.faer_mul(d21_inv);
                    let delta = delta.faer_mul(d21_inv);
                    if has_eps {
                        if let Some(signs) = signs.rb_mut() {
                            if signs[k] > 0 && signs[k + 1] > 0 {
                                if d11 <= eps {
                                    d11 = delta;
                                }
                                if d22 <= eps {
                                    d22 = delta;
                                }
                            } else if signs[k] < 0 && signs[k + 1] < 0 {
                                if d11 >= eps.faer_neg() {
                                    d11 = delta.faer_neg();
                                }
                                if d22 >= eps.faer_neg() {
                                    d22 = delta.faer_neg();
                                }
                            }
                        }
                    }

                    // t = (d11/|d21| * d22/|d21| - 1.0)
                    let mut t = d11.faer_mul(d22).faer_sub(E::Real::faer_one());
                    if has_eps {
                        if let Some(signs) = signs.rb_mut() {
                            if ((signs[k] > 0 && signs[k + 1] > 0)
                                || (signs[k] < 0 && signs[k + 1] < 0))
                                && t <= eps
                            {
                                t = delta;
                            } else if ((signs[k] > 0 && signs[k + 1] < 0)
                                || (signs[k] < 0 && signs[k + 1] > 0))
                                && t >= eps.faer_neg()
                            {
                                t = delta.faer_neg();
                            }
                        }
                    }

                    let t = t.faer_inv();
                    let d21 = a.read(k + 1, k).faer_scale_real(d21_inv);
                    let d = t.faer_mul(d21_inv);

                    a.write(k, k, E::faer_from_real(d11.faer_mul(d)));
                    a.write(k + 1, k, d21.faer_scale_real(d.faer_neg()));
                    a.write(k + 1, k + 1, E::faer_from_real(d22.faer_mul(d)));

                    for j in k + 2..n {
                        let wk = (a
                            .read(j, k)
                            .faer_scale_real(d11)
                            .faer_sub(a.read(j, k + 1).faer_mul(d21)))
                        .faer_scale_real(d);
                        let wkp1 = (a
                            .read(j, k + 1)
                            .faer_scale_real(d22)
                            .faer_sub(a.read(j, k).faer_mul(d21.faer_conj())))
                        .faer_scale_real(d);

                        for i in j..n {
                            a.write(
                                i,
                                j,
                                a.read(i, j)
                                    .faer_sub(a.read(i, k).faer_mul(wk.faer_conj()))
                                    .faer_sub(a.read(i, k + 1).faer_mul(wkp1.faer_conj())),
                            );
                        }
                        make_real(a.rb_mut(), j, j);

                        a.write(j, k, wk);
                        a.write(j, k + 1, wkp1);
                    }
                }
            }

            if k_step == 1 {
                pivots[k] = I::from_signed(truncate(kp));
            } else {
                pivots[k] = I::from_signed(truncate(!kp));
                pivots[k + 1] = I::from_signed(truncate(!kp));
            }

            k += k_step;
        }

        (pivot_count, dynamic_regularization_count)
    }

    fn convert<I: Index, E: ComplexField>(
        mut a: MatMut<'_, E>,
        pivots: &[I],
        mut subdiag: MatMut<'_, E>,
    ) {
        assert!(a.nrows() == a.ncols());
        let n = a.nrows();

        let mut i = 0;
        while i < n {
            if (pivots[i].to_signed().sx() as isize) < 0 {
                subdiag.write(i, 0, a.read(i + 1, i));
                subdiag.write(i + 1, 0, E::faer_zero());
                a.write(i + 1, i, E::faer_zero());
                i += 2;
            } else {
                subdiag.write(i, 0, E::faer_zero());
                i += 1;
            }
        }

        let mut i = 0;
        while i < n {
            let p = pivots[i].to_signed().sx();
            if (p as isize) < 0 {
                let p = !p;
                swap_rows(a.rb_mut().subcols_mut(0, i), i + 1, p);
                i += 2;
            } else {
                swap_rows(a.rb_mut().subcols_mut(0, i), i, p);
                i += 1;
            }
        }
    }

    /// Computes the size and alignment of required workspace for performing a Cholesky
    /// decomposition with Bunch-Kaufman pivoting.
    pub fn cholesky_in_place_req<I: Index, E: Entity>(
        dim: usize,
        parallelism: Parallelism,
        params: BunchKaufmanParams,
    ) -> Result<StackReq, SizeOverflow> {
        let _ = parallelism;
        let mut bs = params.blocksize;
        if bs < 2 || dim <= bs {
            bs = 0;
        }
        StackReq::try_new::<I>(dim)?.try_and(temp_mat_req::<E>(dim, bs)?)
    }

    /// Info about the result of the Bunch-Kaufman factorization.
    #[derive(Copy, Clone, Debug)]
    pub struct BunchKaufmanInfo {
        /// Number of pivots whose value or sign had to be corrected.
        pub dynamic_regularization_count: usize,
        /// Number of pivoting transpositions.
        pub transposition_count: usize,
    }

    /// Computes the Cholesky factorization with Bunch-Kaufman  pivoting of the input matrix and
    /// stores the factorization in `matrix` and `subdiag`.
    ///
    /// The inverses of the diagonal blocks of the block diagonal matrix are stored on the diagonal
    /// of `matrix`, while the subdiagonal elements of those inverses are stored in `subdiag`.
    ///
    /// # Panics
    ///
    /// Panics if the input matrix is not square.
    ///
    /// This can also panic if the provided memory in `stack` is insufficient (see
    /// [`cholesky_in_place_req`]).
    #[track_caller]
    pub fn cholesky_in_place<'out, I: Index, E: ComplexField>(
        matrix: MatMut<'_, E>,
        subdiag: MatMut<'_, E>,
        regularization: BunchKaufmanRegularization<'_, E>,
        perm: &'out mut [I],
        perm_inv: &'out mut [I],
        parallelism: Parallelism,
        stack: PodStack<'_>,
        params: BunchKaufmanParams,
    ) -> (BunchKaufmanInfo, PermRef<'out, I>) {
        let truncate = <I::Signed as SignedIndex>::truncate;
        let mut regularization = regularization;

        let n = matrix.nrows();
        assert!(all(
            matrix.nrows() == matrix.ncols(),
            subdiag.nrows() == n,
            subdiag.ncols() == 1,
            perm.len() == n,
            perm_inv.len() == n
        ));

        #[cfg(feature = "perf-warn")]
        if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(CHOLESKY_WARN) {
            if matrix.col_stride().unsigned_abs() == 1 {
                log::warn!(target: "faer_perf", "Bunch-Kaufman decomposition prefers column-major matrix. Found row-major matrix.");
            } else {
                log::warn!(target: "faer_perf", "Bunch-Kaufman decomposition prefers column-major matrix. Found matrix with generic strides.");
            }
        }

        let _ = parallelism;
        let mut matrix = matrix;

        let alpha = E::Real::faer_one()
            .faer_add(E::Real::faer_from_f64(17.0).faer_sqrt())
            .faer_scale_power_of_two(E::Real::faer_from_f64(1.0 / 8.0));

        let (pivots, stack) = stack.make_raw::<I>(n);

        let mut bs = params.blocksize;
        if bs < 2 || n <= bs {
            bs = 0;
        }
        let mut work = temp_mat_uninit(n, bs, stack).0;

        let mut k = 0;
        let mut dynamic_regularization_count = 0;
        let mut transposition_count = 0;
        while k < n {
            let regularization = BunchKaufmanRegularization {
                dynamic_regularization_signs: regularization
                    .dynamic_regularization_signs
                    .rb_mut()
                    .map(|signs| &mut signs[k..]),
                dynamic_regularization_delta: regularization.dynamic_regularization_delta,
                dynamic_regularization_epsilon: regularization.dynamic_regularization_epsilon,
            };

            let kb;
            let reg_count;
            let piv_count;
            if bs >= 2 && bs < n - k {
                (kb, piv_count, reg_count) = cholesky_diagonal_pivoting_blocked_step(
                    matrix.rb_mut().submatrix_mut(k, k, n - k, n - k),
                    regularization,
                    work.rb_mut(),
                    &mut pivots[k..],
                    alpha,
                    parallelism,
                );
            } else {
                (piv_count, reg_count) = cholesky_diagonal_pivoting_unblocked(
                    matrix.rb_mut().submatrix_mut(k, k, n - k, n - k),
                    regularization,
                    &mut pivots[k..],
                    alpha,
                );
                kb = n - k;
            }
            dynamic_regularization_count += reg_count;
            transposition_count += piv_count;

            for pivot in &mut pivots[k..k + kb] {
                let pv = (*pivot).to_signed().sx();
                if pv as isize >= 0 {
                    *pivot = I::from_signed(truncate(pv + k));
                } else {
                    *pivot = I::from_signed(truncate(pv - k));
                }
            }

            k += kb;
        }

        convert(matrix.rb_mut(), pivots, subdiag);

        for (i, p) in perm.iter_mut().enumerate() {
            *p = I::from_signed(truncate(i));
        }
        let mut i = 0;
        while i < n {
            let p = pivots[i].to_signed().sx();
            if (p as isize) < 0 {
                let p = !p;
                perm.swap(i + 1, p);
                i += 2;
            } else {
                perm.swap(i, p);
                i += 1;
            }
        }
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[p.to_signed().zx()] = I::from_signed(truncate(i));
        }

        (
            BunchKaufmanInfo {
                dynamic_regularization_count,
                transposition_count,
            },
            unsafe { PermRef::new_unchecked(perm, perm_inv) },
        )
    }
}

/// Solving a linear system using the decomposition.
pub mod solve {
    use super::*;
    use crate::assert;

    /// Computes the size and alignment of required workspace for solving a linear system defined by
    /// a matrix in place, given its Bunch-Kaufman decomposition.
    #[track_caller]
    pub fn solve_in_place_req<I: Index, E: Entity>(
        dim: usize,
        rhs_ncols: usize,
        parallelism: Parallelism,
    ) -> Result<StackReq, SizeOverflow> {
        let _ = parallelism;
        temp_mat_req::<E>(dim, rhs_ncols)
    }

    /// Given the Bunch-Kaufman factors of a matrix $A$ and a matrix $B$ stored in `rhs`, this
    /// function computes the solution of the linear system:
    /// $$\text{Op}_A(A)X = B.$$
    ///
    /// $\text{Op}_A$ is either the identity or the conjugation depending on the value of
    /// `conj`.
    ///
    /// The solution of the linear system is stored in `rhs`.
    ///
    /// # Panics
    ///
    /// - Panics if `lb_factors` is not a square matrix.
    /// - Panics if `subdiag` is not a column vector with the same number of rows as the dimension
    ///   of `lb_factors`.
    /// - Panics if `rhs` doesn't have the same number of rows as the dimension of `lb_factors`.
    /// - Panics if the provided memory in `stack` is insufficient (see [`solve_in_place_req`]).
    #[track_caller]
    pub fn solve_in_place_with_conj<I: Index, E: ComplexField>(
        lb_factors: MatRef<'_, E>,
        subdiag: MatRef<'_, E>,
        conj: Conj,
        perm: PermRef<'_, I>,
        rhs: MatMut<'_, E>,
        parallelism: Parallelism,
        stack: PodStack<'_>,
    ) {
        let n = lb_factors.nrows();
        let k = rhs.ncols();

        assert!(all(
            lb_factors.nrows() == lb_factors.ncols(),
            rhs.nrows() == n,
            subdiag.nrows() == n,
            subdiag.ncols() == 1,
            perm.len() == n
        ));

        let a = lb_factors;
        let par = parallelism;
        let not_conj = conj.compose(Conj::Yes);

        let mut rhs = rhs;
        let mut x = temp_mat_uninit::<E>(n, k, stack).0;

        permute_rows(x.rb_mut(), rhs.rb(), perm);
        solve_unit_lower_triangular_in_place_with_conj(a, conj, x.rb_mut(), par);

        let mut i = 0;
        while i < n {
            if subdiag.read(i, 0) == E::faer_zero() {
                let d_inv = a.read(i, i).faer_real();
                for j in 0..k {
                    x.write(i, j, x.read(i, j).faer_scale_real(d_inv));
                }
                i += 1;
            } else {
                if conj == Conj::Yes {
                    let akp1k = subdiag.read(i, 0);
                    let ak = a.read(i, i).faer_real();
                    let akp1 = a.read(i + 1, i + 1).faer_real();

                    for j in 0..k {
                        let xk = x.read(i, j);
                        let xkp1 = x.read(i + 1, j);

                        x.write(i, j, xk.faer_scale_real(ak).faer_add(xkp1.faer_mul(akp1k)));
                        x.write(
                            i + 1,
                            j,
                            xkp1.faer_scale_real(akp1)
                                .faer_add(xk.faer_mul(akp1k.faer_conj())),
                        );
                    }
                } else {
                    let akp1k = subdiag.read(i, 0);
                    let ak = a.read(i, i).faer_real();
                    let akp1 = a.read(i + 1, i + 1).faer_real();

                    for j in 0..k {
                        let xk = x.read(i, j);
                        let xkp1 = x.read(i + 1, j);

                        x.write(
                            i,
                            j,
                            xk.faer_scale_real(ak)
                                .faer_add(xkp1.faer_mul(akp1k.faer_conj())),
                        );
                        x.write(
                            i + 1,
                            j,
                            xkp1.faer_scale_real(akp1).faer_add(xk.faer_mul(akp1k)),
                        );
                    }
                }
                i += 2;
            }
        }

        solve_unit_upper_triangular_in_place_with_conj(a.transpose(), not_conj, x.rb_mut(), par);
        permute_rows(rhs.rb_mut(), x.rb(), perm.inverse());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, complex_native::c64, Mat};
    use compute::BunchKaufmanParams;
    use dyn_stack::GlobalPodBuffer;
    use rand::random;

    #[test]
    fn test_real() {
        for n in [3, 6, 19, 100, 421] {
            let a = Mat::<f64>::from_fn(n, n, |_, _| random());
            let a = &a + a.adjoint();
            let rhs = Mat::<f64>::from_fn(n, 2, |_, _| random());

            let mut ldl = a.clone();
            let mut subdiag = Mat::<f64>::zeros(n, 1);

            let mut perm = vec![0usize; n];
            let mut perm_inv = vec![0; n];

            let params = Default::default();
            let mut mem = GlobalPodBuffer::new(
                compute::cholesky_in_place_req::<usize, f64>(n, Parallelism::None, params).unwrap(),
            );
            let (_, perm) = compute::cholesky_in_place(
                ldl.as_mut(),
                subdiag.as_mut(),
                Default::default(),
                &mut perm,
                &mut perm_inv,
                Parallelism::None,
                PodStack::new(&mut mem),
                params,
            );

            let mut mem = GlobalPodBuffer::new(
                solve::solve_in_place_req::<usize, f64>(n, rhs.ncols(), Parallelism::None).unwrap(),
            );
            let mut x = rhs.clone();
            solve::solve_in_place_with_conj(
                ldl.as_ref(),
                subdiag.as_ref(),
                Conj::No,
                perm.rb(),
                x.as_mut(),
                Parallelism::None,
                PodStack::new(&mut mem),
            );

            let err = &a * &x - &rhs;
            let mut max = 0.0;
            zipped!(err.as_ref()).for_each(|unzipped!(err)| {
                let err = err.read().abs();
                if err > max {
                    max = err
                }
            });
            assert!(max < 1e-9);
        }
    }

    #[test]
    fn test_cplx() {
        for n in [3, 6, 19, 100, 421] {
            let a = Mat::<c64>::from_fn(n, n, |_, _| c64::new(random(), random()));
            let a = &a + a.adjoint();
            let rhs = Mat::<c64>::from_fn(n, 2, |_, _| c64::new(random(), random()));

            let mut ldl = a.clone();
            let mut subdiag = Mat::<c64>::zeros(n, 1);

            let mut perm = vec![0usize; n];
            let mut perm_inv = vec![0; n];

            let params = BunchKaufmanParams {
                pivoting: compute::PivotingStrategy::Diagonal,
                blocksize: 32,
            };
            let mut mem = GlobalPodBuffer::new(
                compute::cholesky_in_place_req::<usize, c64>(n, Parallelism::None, params).unwrap(),
            );
            let (_, perm) = compute::cholesky_in_place(
                ldl.as_mut(),
                subdiag.as_mut(),
                Default::default(),
                &mut perm,
                &mut perm_inv,
                Parallelism::None,
                PodStack::new(&mut mem),
                params,
            );

            let mut x = rhs.clone();
            let mut mem = GlobalPodBuffer::new(
                solve::solve_in_place_req::<usize, c64>(n, rhs.ncols(), Parallelism::None).unwrap(),
            );
            solve::solve_in_place_with_conj(
                ldl.as_ref(),
                subdiag.as_ref(),
                Conj::Yes,
                perm.rb(),
                x.as_mut(),
                Parallelism::None,
                PodStack::new(&mut mem),
            );

            let err = a.conjugate() * &x - &rhs;
            let mut max = 0.0;
            zipped!(err.as_ref()).for_each(|unzipped!(err)| {
                let err = err.read().abs();
                if err > max {
                    max = err
                }
            });
            for i in 0..n {
                assert!(ldl[(i, i)].faer_imag() == 0.0);
            }
            assert!(max < 1e-9);
        }
    }
}
