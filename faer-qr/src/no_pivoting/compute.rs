use core::ops::{Add, Mul, Neg};
// TODO: remove

use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use dyn_stack::DynStack;
use faer_core::float_traits::Sqrt;
use faer_core::householder::{
    apply_block_househodler_on_the_left, apply_househodler_on_the_left,
    make_householder_in_place_unchecked,
};
use faer_core::mul::{matmul, triangular};
use faer_core::{temp_mat_uninit, ColMut, MatMut, MatRef};
use num_traits::{Inv, One, Zero};
use reborrow::*;

unsafe fn qr_in_place_unblocked<T>(
    mut matrix: MatMut<'_, T>,
    mut householder_factor: MatMut<'_, T>,
    mut stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Sqrt + PartialOrd + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Inv<Output = T> + Neg<Output = T>,
{
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = n.min(m);

    fancy_debug_assert!(householder_factor.nrows() == size);
    fancy_debug_assert!(householder_factor.ncols() == size);

    for k in 0..size {
        let mat_rem = matrix.rb_mut().submatrix_unchecked(k, k, m - k, n - k);
        let (_, _, first_col, last_cols) = mat_rem.split_at_unchecked(0, 1);
        let (mut first_col_head, mut first_col_tail) =
            first_col.col_unchecked(0).split_at_unchecked(1);

        let mut tail_squared_norm = T::zero();
        for elem in first_col_tail.rb() {
            tail_squared_norm = tail_squared_norm + elem * elem;
        }

        let (tau, beta) = make_householder_in_place_unchecked(
            first_col_tail.rb_mut(),
            first_col_head.rb_mut().get_unchecked(0),
            &tail_squared_norm,
        );

        *householder_factor.rb_mut().ptr_in_bounds_at_unchecked(k, k) = tau;
        *first_col_head.rb_mut().get_unchecked(0) = beta;

        if last_cols.ncols() > 0 {
            apply_househodler_on_the_left(
                last_cols,
                first_col_tail.rb(),
                householder_factor.rb().get_unchecked(k, k),
                stack.rb_mut(),
            );
        }
    }

    make_householder_factor_unblocked(householder_factor, matrix.rb(), stack);
}

unsafe fn make_householder_factor_unblocked<T>(
    mut householder_factor: MatMut<'_, T>,
    matrix: MatRef<'_, T>,
    mut stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Sqrt + PartialOrd + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Inv<Output = T> + Neg<Output = T>,
{
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = m.min(n);

    fancy_debug_assert!(householder_factor.nrows() == size);
    fancy_debug_assert!(householder_factor.ncols() == size);

    for i in (0..size).rev() {
        let rs = m - i - 1;
        let rt = size - i - 1;

        if rt > 0 {
            let factor = -householder_factor.rb().get_unchecked(i, i);

            let mut tail_row = householder_factor
                .rb_mut()
                .row_unchecked(i)
                .split_at_unchecked(size - rt)
                .1;

            use faer_core::mul::triangular::BlockStructure::*;

            let mut dst = tail_row.rb_mut().as_2d();
            let lhs = matrix
                .col(i)
                .split_at_unchecked(m - rs)
                .1
                .transpose()
                .as_2d();

            let rhs = matrix.submatrix_unchecked(m - rs, n - rt, rs, rt);
            triangular::matmul(
                dst.rb_mut(),
                Rectangular,
                lhs.split_at_unchecked(0, rt).2,
                Rectangular,
                rhs.split_at_unchecked(rt, 0).1,
                UnitTriangularLower,
                None,
                &factor,
                1,
                stack.rb_mut(),
            );
            matmul(
                dst.rb_mut(),
                lhs.split_at_unchecked(0, rt).3,
                rhs.split_at_unchecked(rt, 0).3,
                Some(&T::one()),
                &factor,
                1,
                stack.rb_mut(),
            );

            temp_mat_uninit! {
                let (mut tmp, stack) = unsafe { temp_mat_uninit::<T>(rt, 1, stack.rb_mut()) };
            }

            triangular::matmul(
                tmp.rb_mut().transpose(),
                Rectangular,
                householder_factor
                    .rb()
                    .submatrix_unchecked(i, size - rt, 1, rt),
                Rectangular,
                householder_factor
                    .rb()
                    .submatrix_unchecked(size - rt, size - rt, rt, rt),
                TriangularUpper,
                None,
                &T::one(),
                1,
                stack,
            );
            householder_factor
                .rb_mut()
                .submatrix_unchecked(i, size - rt, 1, rt)
                .row_unchecked(0)
                .cwise()
                .zip_unchecked(tmp.transpose().row_unchecked(0))
                .for_each(|a, b| *a = b.clone());
        }
    }
}

unsafe fn make_householder_top_right<T>(
    t01: MatMut<'_, T>,
    t00: MatRef<'_, T>,
    t11: MatRef<'_, T>,
    basis0: MatRef<'_, T>,
    basis1: MatRef<'_, T>,
    n_threads: usize,
    mut stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    let m = basis0.nrows();
    let n = basis0.ncols() + basis1.ncols();
    let bs = basis0.ncols();

    temp_mat_uninit! {
        let (mut tmp0, stack) = unsafe { temp_mat_uninit::<T>(m - bs, bs, stack.rb_mut()) };
        let (mut tmp1, mut stack) = unsafe { temp_mat_uninit::<T>(m - bs, n - bs, stack) };
    }

    let v0 = basis0.submatrix_unchecked(bs, 0, m - bs, bs);
    let v1 = basis1;

    let (_, v1_top, _, v1_bot) = v1.split_at_unchecked(n - bs, 0);
    let (_, tmp1_top, _, tmp1_bot) = tmp1.rb_mut().split_at_unchecked(n - bs, 0);

    use triangular::BlockStructure::*;
    triangular::matmul(
        tmp0.rb_mut(),
        Rectangular,
        v0,
        Rectangular,
        t00.transpose(),
        TriangularLower,
        None,
        &T::one(),
        n_threads,
        stack.rb_mut(),
    );
    triangular::matmul(
        tmp1_top,
        Rectangular,
        v1_top,
        UnitTriangularLower,
        t11,
        TriangularUpper,
        None,
        &T::one(),
        n_threads,
        stack.rb_mut(),
    );
    triangular::matmul(
        tmp1_bot,
        Rectangular,
        v1_bot,
        Rectangular,
        t11,
        TriangularUpper,
        None,
        &T::one(),
        n_threads,
        stack.rb_mut(),
    );

    matmul(
        t01,
        tmp0.rb().transpose(),
        tmp1.rb(),
        None,
        &-&T::one(),
        n_threads,
        stack.rb_mut(),
    );
}

fn recursion_threshold<T: 'static>(m: usize) -> usize {
    let _m = m;
    32
}

fn block_size<T: 'static>(n: usize) -> usize {
    n / 2
}

unsafe fn qr_in_place_recursive<T>(
    mut matrix: MatMut<'_, T>,
    mut householder_factor: MatMut<'_, T>,
    n_threads: usize,
    mut stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Sqrt + PartialOrd + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Inv<Output = T> + Neg<Output = T>,
{
    let m = matrix.nrows();
    let n = matrix.ncols();

    fancy_debug_assert!(m >= n);

    if n <= recursion_threshold::<T>(m) {
        return qr_in_place_unblocked(matrix, householder_factor, stack);
    }

    let bs = block_size::<T>(n);
    let (_, _, mut left, mut right) = matrix.rb_mut().split_at_unchecked(0, bs);

    let (mut t00, t01, _, mut t11) = householder_factor.rb_mut().split_at_unchecked(bs, bs);

    qr_in_place_recursive(left.rb_mut(), t00.rb_mut(), n_threads, stack.rb_mut());

    apply_block_househodler_on_the_left(
        right.rb_mut(),
        left.rb().submatrix_unchecked(0, 0, m, bs.min(m)),
        t00.rb(),
        false,
        n_threads,
        stack.rb_mut(),
    );

    let (_, _, _, mut bot_right) = right.rb_mut().split_at_unchecked(bs, 0);
    qr_in_place_recursive(bot_right.rb_mut(), t11.rb_mut(), n_threads, stack.rb_mut());

    make_householder_top_right(
        t01,
        t00.rb(),
        t11.rb(),
        left.rb(),
        bot_right.rb(),
        n_threads,
        stack,
    );
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::{GlobalMemBuffer, StackReq};
    use faer_core::mul::matmul;
    use faer_core::{Mat, MatRef};
    use rand::random;

    use super::*;

    macro_rules! placeholder_stack {
        () => {
            DynStack::new(&mut GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024)))
        };
    }

    fn reconstruct_factors(
        qr_factors: MatRef<'_, f64>,
        householder: MatRef<'_, f64>,
    ) -> (Mat<f64>, Mat<f64>) {
        let m = qr_factors.nrows();
        let n = qr_factors.ncols();
        let size = m.min(n);

        let mut q = Mat::zeros(m, m);
        let mut q2 = Mat::zeros(m, m);
        let mut r = Mat::zeros(m, n);

        r.as_mut()
            .cwise()
            .zip(qr_factors)
            .for_each_triangular_upper(false, |a, b| *a = *b);

        q.as_mut().diagonal().cwise().for_each(|a| *a = 1.0);
        q2.as_mut().diagonal().cwise().for_each(|a| *a = 1.0);

        let (basis_top, _, basis_bot, _) = qr_factors.split_at(size, size);
        let mut tmp = Mat::zeros(m, size);

        let (_, mut tmp_top, _, mut tmp_bot) = tmp.as_mut().split_at(size, 0);

        use triangular::BlockStructure::*;
        triangular::matmul(
            tmp_top.rb_mut(),
            Rectangular,
            basis_top,
            UnitTriangularLower,
            householder.transpose(),
            TriangularLower,
            None,
            &1.0,
            1,
            placeholder_stack!(),
        );
        triangular::matmul(
            tmp_bot.rb_mut(),
            Rectangular,
            basis_bot,
            Rectangular,
            householder.transpose(),
            TriangularLower,
            None,
            &1.0,
            1,
            placeholder_stack!(),
        );

        let (q_top_left, q_top_right, q_bot_left, q_bot_right) = q.as_mut().split_at(size, size);
        triangular::matmul(
            q_top_left,
            Rectangular,
            basis_top,
            UnitTriangularLower,
            tmp_top.rb().transpose(),
            Rectangular,
            Some(&1.0),
            &-1.0,
            1,
            placeholder_stack!(),
        );
        triangular::matmul(
            q_top_right,
            Rectangular,
            basis_top,
            UnitTriangularLower,
            tmp_bot.rb().transpose(),
            Rectangular,
            Some(&1.0),
            &-1.0,
            1,
            placeholder_stack!(),
        );
        matmul(
            q_bot_left,
            basis_bot,
            tmp_top.rb().transpose(),
            Some(&1.0),
            &-1.0,
            1,
            placeholder_stack!(),
        );
        matmul(
            q_bot_right,
            basis_bot,
            tmp_bot.rb().transpose(),
            Some(&1.0),
            &-1.0,
            1,
            placeholder_stack!(),
        );

        for k in (0..size).rev() {
            let tau = householder[(k, k)];
            let essential = qr_factors.col(k).split_at(k + 1).1;
            unsafe {
                apply_househodler_on_the_left(
                    q2.as_mut().submatrix(k, k, m - k, m - k),
                    essential,
                    &tau,
                    placeholder_stack!(),
                );
            }
        }

        (q, r)
    }

    #[test]
    fn test_unblocked() {
        for (m, n) in [(2, 2), (2, 4), (4, 2), (4, 4)] {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), m, n);
            let mat_orig = mat.clone();
            let size = m.min(n);
            let mut householder = Mat::zeros(size, size);
            unsafe {
                qr_in_place_unblocked(mat.as_mut(), householder.as_mut(), placeholder_stack!())
            }

            let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
            let mut qtq = Mat::zeros(m, m);
            let mut reconstructed = Mat::zeros(m, n);

            matmul(
                reconstructed.as_mut(),
                q.as_ref(),
                r.as_ref(),
                None,
                &1.0,
                1,
                placeholder_stack!(),
            );
            matmul(
                qtq.as_mut(),
                q.as_ref().transpose(),
                q.as_ref(),
                None,
                &1.0,
                1,
                placeholder_stack!(),
            );

            for i in 0..m {
                for j in 0..m {
                    assert_approx_eq!(qtq[(i, j)], if i == j { 1.0 } else { 0.0 });
                }
            }
            for i in 0..m {
                for j in 0..n {
                    assert_approx_eq!(reconstructed[(i, j)], mat_orig[(i, j)]);
                }
            }
        }
    }

    #[test]
    fn test_recursive() {
        for (m, n) in [(2, 2), (4, 2), (4, 4), (33, 33), (64, 64), (65, 65)] {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), m, n);
            let mat_orig = mat.clone();
            let size = m.min(n);
            let mut householder = Mat::zeros(size, size);
            unsafe {
                qr_in_place_recursive(mat.as_mut(), householder.as_mut(), 1, placeholder_stack!())
            }

            let (q, r) = reconstruct_factors(mat.as_ref(), householder.as_ref());
            let mut qtq = Mat::zeros(m, m);
            let mut reconstructed = Mat::zeros(m, n);

            matmul(
                reconstructed.as_mut(),
                q.as_ref(),
                r.as_ref(),
                None,
                &1.0,
                1,
                placeholder_stack!(),
            );
            matmul(
                qtq.as_mut(),
                q.as_ref().transpose(),
                q.as_ref(),
                None,
                &1.0,
                1,
                placeholder_stack!(),
            );

            for i in 0..m {
                for j in 0..m {
                    assert_approx_eq!(qtq[(i, j)], if i == j { 1.0 } else { 0.0 });
                }
            }
            for i in 0..m {
                for j in 0..n {
                    assert_approx_eq!(reconstructed[(i, j)], mat_orig[(i, j)]);
                }
            }
        }
    }
}
