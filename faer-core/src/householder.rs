use core::ops::{Add, Mul, Neg};

use crate::float_traits::Sqrt;
use crate::mul::{matmul, triangular};
use crate::{temp_mat_uninit, ColMut, ColRef, MatMut, MatRef};

use assert2::debug_assert as fancy_debug_assert;
use dyn_stack::DynStack;
use num_traits::{Inv, One, Zero};
use reborrow::*;

pub fn make_householder_in_place_unchecked<T>(
    essential: ColMut<'_, T>,
    head: &T,
    tail_squared_norm: &T,
) -> (T, T)
where
    T: Zero + One + Sqrt + PartialOrd,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Inv<Output = T> + Neg<Output = T>,
{
    let mut beta = (&(head * head) + tail_squared_norm).sqrt();
    if *head >= T::zero() {
        beta = -&beta;
    }
    let head_minus_beta = head + &-&beta;
    let inv = head_minus_beta.inv();
    essential.cwise().for_each(|e| *e = &*e * &inv);
    let tau = -&head_minus_beta * beta.inv();
    (tau, beta)
}

pub unsafe fn apply_househodler_on_the_left<T>(
    matrix: MatMut<'_, T>,
    essential: ColRef<'_, T>,
    householder_coeff: &T,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + 'static + Send + Sync,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Inv<Output = T> + Neg<Output = T>,
{
    fancy_debug_assert!(matrix.nrows() == 1 + essential.nrows());
    let m = matrix.nrows();
    let n = matrix.ncols();
    if m == 1 {
        let factor = T::one() + -householder_coeff;
        matrix.cwise().for_each(|e| *e = &*e * &factor);
    } else {
        let (_, first_row, _, last_rows) = matrix.split_at_unchecked(1, 0);
        let mut first_row = first_row.row_unchecked(0);
        temp_mat_uninit! {
            let (tmp, mut stack) = unsafe { temp_mat_uninit::<T>(n, 1, stack) };
        }
        let mut tmp = tmp.transpose().row_unchecked(0);

        tmp.rb_mut()
            .cwise()
            .zip_unchecked(first_row.rb())
            .for_each(|a, b| *a = b.clone());

        matmul(
            tmp.rb_mut().as_2d(),
            essential.transpose().as_2d(),
            last_rows.rb(),
            Some(&T::one()),
            &T::one(),
            1,
            stack.rb_mut(),
        );

        first_row
            .rb_mut()
            .cwise()
            .zip_unchecked(tmp.rb())
            .for_each(|a, b| *a = &*a + &-&(householder_coeff * b));

        matmul(
            last_rows,
            essential.as_2d(),
            tmp.rb().as_2d(),
            Some(&T::one()),
            &-householder_coeff,
            1,
            stack,
        )
    }
}

pub unsafe fn apply_block_househodler_on_the_left<T>(
    matrix: MatMut<'_, T>,
    basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    forward: bool,
    n_threads: usize,
    mut stack: DynStack<'_>,
) where
    T: Zero + One + Clone + 'static + Send + Sync,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Inv<Output = T> + Neg<Output = T>,
{
    fancy_debug_assert!(matrix.nrows() == basis.nrows());
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = basis.ncols();

    let (_, basis_tri, _, basis_bot) = basis.split_at_unchecked(size, 0);

    temp_mat_uninit! {
        let (mut tmp1, mut stack) = unsafe { temp_mat_uninit::<T>(size, n, stack.rb_mut()) };
    }

    use triangular::BlockStructure::*;
    {
        temp_mat_uninit! {
            let (mut tmp0, mut stack) = unsafe { temp_mat_uninit::<T>(size, n, stack.rb_mut()) };
        }

        triangular::matmul(
            tmp0.rb_mut(),
            Rectangular,
            basis_tri.transpose(),
            UnitTriangularUpper,
            matrix.rb().submatrix_unchecked(0, 0, size, n),
            Rectangular,
            None,
            &T::one(),
            n_threads,
            stack.rb_mut(),
        );
        matmul(
            tmp0.rb_mut(),
            basis_bot.transpose(),
            matrix.rb().submatrix_unchecked(size, 0, m - size, n),
            Some(&T::one()),
            &T::one(),
            n_threads,
            stack.rb_mut(),
        );

        triangular::matmul(
            tmp1.rb_mut(),
            Rectangular,
            if forward {
                householder_factor
            } else {
                householder_factor.transpose()
            },
            if forward {
                TriangularUpper
            } else {
                TriangularLower
            },
            tmp0.rb(),
            Rectangular,
            None,
            &T::one(),
            n_threads,
            stack.rb_mut(),
        );
    }

    let (_, matrix_top, _, matrix_bot) = matrix.split_at_unchecked(size, 0);

    triangular::matmul(
        matrix_top,
        Rectangular,
        basis_tri,
        UnitTriangularLower,
        tmp1.rb(),
        Rectangular,
        Some(&T::one()),
        &-&T::one(),
        n_threads,
        stack.rb_mut(),
    );
    matmul(
        matrix_bot,
        basis_bot,
        tmp1.rb(),
        Some(&T::one()),
        &-&T::one(),
        n_threads,
        stack.rb_mut(),
    )
}
