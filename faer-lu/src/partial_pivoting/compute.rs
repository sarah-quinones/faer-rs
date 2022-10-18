use core::ops::{Add, Mul, Neg};

use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use dyn_stack::DynStack;
use faer_core::mul::matmul;
use faer_core::permutation::{
    permute_rows_unchecked, PermutationIndicesMut, PermutationIndicesRef,
};
use faer_core::solve::triangular::solve_unit_lower_triangular_in_place;
use faer_core::{temp_mat_uninit, MatMut};
use num_traits::{Inv, One, Signed, Zero};
use reborrow::*;

unsafe fn lu_in_place_unblocked<T>(
    mut matrix: MatMut<'_, T>,
    perm: &mut [usize],
    _perm_inv: &mut [usize],
    n_threads: usize,
    mut stack: DynStack<'_>,
) -> usize
where
    T: Zero + One + Clone + Send + Sync + Signed + PartialOrd + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    let m = matrix.nrows();
    let n = matrix.ncols();

    fancy_debug_assert!(m >= n);
    fancy_debug_assert!(perm.len() == m);

    if n == 0 {
        return 0;
    }

    let size = m.min(n);

    let mut n_transpositions = 0;

    for j in 0..size {
        let mut max = T::zero();
        let mut imax = j;

        for i in j..m {
            let abs = matrix.rb().get_unchecked(i, j).abs();
            if abs > max {
                imax = i;
                max = abs;
            }
        }

        if imax != j {
            n_transpositions += 1;

            let (_, top, _, bot) = matrix.rb_mut().split_at_unchecked(j + 1, 0);
            let mut row_j = top.row_unchecked(j);
            let mut row_imax = bot.row_unchecked(imax - j - 1);
            perm.swap(j, imax);

            for k in 0..n {
                core::mem::swap(
                    row_j.rb_mut().get_unchecked(k),
                    row_imax.rb_mut().get_unchecked(k),
                );
            }
        }

        let inv = matrix.rb().get_unchecked(j, j).inv();
        for i in j + 1..m {
            let elem = matrix.rb_mut().get_unchecked(i, j);
            *elem = &*elem * &inv;
        }

        let (_, top_right, bottom_left, bottom_right) =
            matrix.rb_mut().split_at_unchecked(j + 1, j + 1);

        matmul(
            bottom_right,
            bottom_left.rb().col(j).as_2d(),
            top_right.rb().row(j).as_2d(),
            Some(&T::one()),
            &-&T::one(),
            n_threads,
            stack.rb_mut(),
        )
    }

    n_transpositions
}

unsafe fn lu_in_place_impl<T>(
    mut matrix: MatMut<'_, T>,
    perm: &mut [usize],
    perm_inv: &mut [usize],
    n_threads: usize,
    mut stack: DynStack<'_>,
) -> usize
where
    T: Zero + One + Clone + Send + Sync + Signed + PartialOrd + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    let m = matrix.nrows();
    let n = matrix.ncols();

    fancy_debug_assert!(m >= n);
    fancy_debug_assert!(perm.len() == m);

    if n <= 24 {
        return lu_in_place_unblocked(matrix, perm, perm_inv, n_threads, stack);
    }

    let bs = n / 2;

    let (_, _, mut mat_left, mut mat_right) = matrix.rb_mut().split_at_unchecked(0, bs);

    let n_transpositions0 =
        lu_in_place_impl(mat_left.rb_mut(), perm, perm_inv, n_threads, stack.rb_mut());

    {
        temp_mat_uninit! {
            let (mut tmp_right, _) = unsafe { temp_mat_uninit::<T>(m, n - bs, stack.rb_mut()) };
        }

        tmp_right
            .rb_mut()
            .cwise()
            .zip_unchecked(mat_right.rb())
            .for_each(|a, b| *a = b.clone());

        permute_rows_unchecked(
            mat_right.rb_mut(),
            tmp_right.rb(),
            PermutationIndicesRef::new_unchecked(perm, perm_inv),
            n_threads,
        );
    }

    let (mat_top_left, mut mat_top_right, mat_bot_left, mut mat_bot_right) =
        matrix.rb_mut().split_at_unchecked(bs, bs);

    solve_unit_lower_triangular_in_place(
        mat_top_left.rb(),
        mat_top_right.rb_mut(),
        n_threads,
        stack.rb_mut(),
    );
    matmul(
        mat_bot_right.rb_mut(),
        mat_bot_left.rb(),
        mat_top_right.rb(),
        Some(&T::one()),
        &-&T::one(),
        n_threads,
        stack.rb_mut(),
    );

    let (mut tmp_perm, mut stack) = stack.make_with(m - bs, |i| i);
    let tmp_perm = &mut *tmp_perm;
    let tmp_perm_inv = perm_inv.split_at_mut(bs).1;
    let n_transpositions1 = lu_in_place_impl(
        mat_bot_right.rb_mut(),
        tmp_perm,
        tmp_perm_inv,
        n_threads,
        stack.rb_mut(),
    );

    {
        temp_mat_uninit! {
            let (mut tmp_bot_left, _) = unsafe { temp_mat_uninit::<T>(m - bs, bs, stack.rb_mut()) };
        }

        tmp_bot_left
            .rb_mut()
            .cwise()
            .zip_unchecked(mat_bot_left.rb())
            .for_each(|a, b| *a = b.clone());

        permute_rows_unchecked(
            mat_bot_left,
            tmp_bot_left.rb(),
            PermutationIndicesRef::new_unchecked(tmp_perm, tmp_perm_inv),
            n_threads,
        );
    }

    for idx in 0..m - bs {
        tmp_perm_inv[idx] = perm[bs + tmp_perm[idx]];
    }
    perm[bs..].copy_from_slice(tmp_perm_inv);

    n_transpositions0 + n_transpositions1
}

#[inline]
pub fn lu_in_place<'out, T>(
    matrix: MatMut<'_, T>,
    perm: &'out mut [usize],
    perm_inv: &'out mut [usize],
    n_threads: usize,
    stack: DynStack<'_>,
) -> (usize, PermutationIndicesMut<'out>)
where
    T: Zero + One + Clone + Send + Sync + Signed + PartialOrd + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    fancy_assert!(perm.len() == matrix.nrows());
    fancy_assert!(perm_inv.len() == matrix.nrows());
    let m = matrix.nrows();
    let n = matrix.ncols();
    let mut stack = stack;

    unsafe {
        for i in 0..m {
            *perm.get_unchecked_mut(i) = i;
        }
        let (_, _, mut left, mut right) = matrix.split_at_unchecked(0, n.min(m));
        let n_transpositions =
            lu_in_place_impl(left.rb_mut(), perm, perm_inv, n_threads, stack.rb_mut());

        if m < n {
            {
                temp_mat_uninit! {
                    let (mut tmp_right, _) = unsafe { temp_mat_uninit::<T>(m, n - m, stack.rb_mut()) };
                }
                tmp_right
                    .rb_mut()
                    .cwise()
                    .zip_unchecked(right.rb())
                    .for_each(|a, b| *a = b.clone());

                permute_rows_unchecked(
                    right.rb_mut(),
                    tmp_right.rb(),
                    PermutationIndicesRef::new_unchecked(perm, perm_inv),
                    n_threads,
                );
            }
            solve_unit_lower_triangular_in_place(left.rb(), right, n_threads, stack);
        }

        for i in 0..m {
            *perm_inv.get_unchecked_mut(*perm.get_unchecked(i)) = i;
        }

        (
            n_transpositions,
            PermutationIndicesMut::new_unchecked(perm, perm_inv),
        )
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::{GlobalMemBuffer, StackReq};
    use faer_core::{mul, Mat, MatRef};
    use rand::random;

    use super::*;

    fn reconstruct_matrix(lu_factors: MatRef<'_, f64>) -> Mat<f64> {
        let m = lu_factors.nrows();
        let n = lu_factors.ncols();

        let size = n.min(m);

        let mut a_reconstructed = Mat::zeros(m, n);

        let (_, l_top, _, l_bot) = lu_factors.submatrix(0, 0, m, size).split_at(size, 0);
        let (_, _, u_left, u_right) = lu_factors.submatrix(0, 0, size, n).split_at(0, size);

        use mul::triangular::BlockStructure::*;

        let (dst_top_left, dst_top_right, dst_bot_left, dst_bot_right) =
            a_reconstructed.as_mut().split_at(size, size);

        mul::triangular::matmul(
            dst_top_left,
            Rectangular,
            l_top,
            UnitTriangularLower,
            u_left,
            TriangularUpper,
            None,
            &1.0,
            12,
            DynStack::new(&mut dyn_stack::GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    UnitTriangularLower,
                    TriangularUpper,
                    size,
                    size,
                    size,
                    12,
                )
                .unwrap(),
            )),
        );
        mul::triangular::matmul(
            dst_top_right,
            Rectangular,
            l_top,
            UnitTriangularLower,
            u_right,
            Rectangular,
            None,
            &1.0,
            12,
            DynStack::new(&mut dyn_stack::GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    UnitTriangularLower,
                    Rectangular,
                    size,
                    n - size,
                    size,
                    12,
                )
                .unwrap(),
            )),
        );
        mul::triangular::matmul(
            dst_bot_left,
            Rectangular,
            l_bot,
            Rectangular,
            u_left,
            TriangularUpper,
            None,
            &1.0,
            12,
            DynStack::new(&mut dyn_stack::GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    Rectangular,
                    TriangularUpper,
                    m - size,
                    size,
                    size,
                    12,
                )
                .unwrap(),
            )),
        );
        mul::triangular::matmul(
            dst_bot_right,
            Rectangular,
            l_bot,
            Rectangular,
            u_right,
            Rectangular,
            None,
            &1.0,
            12,
            DynStack::new(&mut dyn_stack::GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    Rectangular,
                    Rectangular,
                    m - size,
                    n - size,
                    size,
                    12,
                )
                .unwrap(),
            )),
        );

        a_reconstructed
    }

    #[test]
    fn compute_lu() {
        for (m, n) in [
            (2, 2),
            (4, 4),
            (20, 20),
            (4, 2),
            (20, 2),
            (2, 4),
            (2, 20),
            (40, 20),
            (20, 40),
        ] {
            let mut mat = Mat::with_dims(|_i, _j| random::<f64>(), m, n);
            let mat_orig = mat.clone();
            let mut perm = vec![0; m];
            let mut perm_inv = vec![0; m];

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            lu_in_place(mat.as_mut(), &mut perm, &mut perm_inv, 1, stack.rb_mut());
            let reconstructed = reconstruct_matrix(mat.as_ref());

            for i in 0..m {
                for j in 0..n {
                    assert_approx_eq!(mat_orig[(perm[i], j)], reconstructed[(i, j)]);
                }
            }
        }
    }
}
