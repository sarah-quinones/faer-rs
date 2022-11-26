use crate::{
    mul::{matmul, triangular},
    temp_mat_uninit, ColMut, ColRef, ComplexField, Conj, MatMut, MatRef, Parallelism,
};

use assert2::debug_assert as fancy_debug_assert;
use dyn_stack::DynStack;
use reborrow::*;

pub fn make_householder_in_place_unchecked<T: ComplexField>(
    essential: ColMut<'_, T>,
    head: T,
    tail_squared_norm: T::Real,
) -> (T, T) {
    let norm = ((head * head.conj()).real() + tail_squared_norm).sqrt();
    let sign = head / (head * head.conj()).sqrt();

    let signed_norm = sign * T::from_real(norm);
    let head_with_beta = head + signed_norm;
    let inv = head_with_beta.inv();
    essential.cwise().for_each(|e| *e = *e * inv);

    let two = T::Real::one() + T::Real::one();
    let tau = two / (T::Real::one() + tail_squared_norm * (inv * inv.conj()).real());
    (T::from_real(tau), -signed_norm)
}

pub unsafe fn apply_househodler_on_the_left<T: ComplexField>(
    matrix: MatMut<'_, T>,
    essential: ColRef<'_, T>,
    householder_coeff: T,
    stack: DynStack<'_>,
) {
    fancy_debug_assert!(matrix.nrows() == 1 + essential.nrows());
    let m = matrix.nrows();
    let n = matrix.ncols();
    if m == 1 {
        let factor = T::one() - householder_coeff;
        matrix.cwise().for_each(|e| *e = *e * factor);
    } else {
        let (first_row, last_rows) = matrix.split_at_row(1);
        let mut first_row = first_row.row_unchecked(0);
        temp_mat_uninit! {
            let (tmp, _) = unsafe { temp_mat_uninit::<T>(n, 1, stack) };
        }
        let mut tmp = tmp.transpose().row_unchecked(0);

        tmp.rb_mut()
            .cwise()
            .zip_unchecked(first_row.rb())
            .for_each(|a, b| *a = *b);

        matmul(
            tmp.rb_mut().as_2d(),
            Conj::No,
            essential.transpose().as_2d(),
            Conj::Yes,
            last_rows.rb(),
            Conj::No,
            Some(T::one()),
            T::one(),
            Parallelism::None,
        );

        first_row
            .rb_mut()
            .cwise()
            .zip_unchecked(tmp.rb())
            .for_each(|a, b| *a = *a - householder_coeff * *b);

        matmul(
            last_rows,
            Conj::No,
            essential.as_2d(),
            Conj::No,
            tmp.rb().as_2d(),
            Conj::No,
            Some(T::one()),
            -householder_coeff,
            Parallelism::None,
        )
    }
}

pub unsafe fn apply_block_househodler_on_the_left<T: ComplexField>(
    matrix: MatMut<'_, T>,
    basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    forward: bool,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    fancy_debug_assert!(matrix.nrows() == basis.nrows());
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = basis.ncols();

    let (basis_tri, basis_bot) = basis.split_at_row(size);

    temp_mat_uninit! {
        let (mut tmp0, stack) = unsafe { temp_mat_uninit::<T>(size, n, stack) };
        let (mut tmp1, _) = unsafe { temp_mat_uninit::<T>(size, n, stack) };
    }

    use triangular::BlockStructure::*;

    triangular::matmul(
        tmp0.rb_mut(),
        Rectangular,
        Conj::No,
        basis_tri.transpose(),
        UnitTriangularUpper,
        Conj::Yes,
        matrix.rb().submatrix_unchecked(0, 0, size, n),
        Rectangular,
        Conj::No,
        None,
        T::one(),
        parallelism,
    );
    matmul(
        tmp0.rb_mut(),
        Conj::No,
        basis_bot.transpose(),
        Conj::Yes,
        matrix.rb().submatrix_unchecked(size, 0, m - size, n),
        Conj::No,
        Some(T::one()),
        T::one(),
        parallelism,
    );

    triangular::matmul(
        tmp1.rb_mut(),
        Rectangular,
        Conj::No,
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
        if forward { Conj::No } else { Conj::Yes },
        tmp0.rb(),
        Rectangular,
        Conj::No,
        None,
        T::one(),
        parallelism,
    );

    let (matrix_top, matrix_bot) = matrix.split_at_row(size);

    triangular::matmul(
        matrix_top,
        Rectangular,
        Conj::No,
        basis_tri,
        UnitTriangularLower,
        Conj::No,
        tmp1.rb(),
        Rectangular,
        Conj::No,
        Some(T::one()),
        -T::one(),
        parallelism,
    );
    matmul(
        matrix_bot,
        Conj::No,
        basis_bot,
        Conj::No,
        tmp1.rb(),
        Conj::No,
        Some(T::one()),
        -T::one(),
        parallelism,
    )
}
