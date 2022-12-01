use crate::{
    mul::{
        dot, matmul,
        triangular::{self, BlockStructure},
    },
    temp_mat_uninit, ColMut, ColRef, ComplexField, Conj, MatMut, MatRef, Parallelism,
};

use assert2::assert as fancy_assert;
use dyn_stack::DynStack;
use reborrow::*;

pub fn make_householder_in_place<T: ComplexField>(
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

pub fn make_householder_factor_unblocked<T: ComplexField>(
    mut householder_factor: MatMut<'_, T>,
    matrix: MatRef<'_, T>,
    mut stack: DynStack<'_>,
) {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let size = m.min(n);

    fancy_assert!((householder_factor.nrows(), householder_factor.ncols()) == (size, size));
    fancy_assert!(householder_factor.col_stride() == 1);

    for i in (0..size).rev() {
        let rs = m - i - 1;
        let rt = size - i - 1;

        if rt > 0 {
            let factor = -*householder_factor.rb().get(i, i);

            let mut tail_row = householder_factor.rb_mut().row(i).split_at(size - rt).1;

            let mut dst = tail_row.rb_mut().as_2d();
            let lhs = matrix.col(i).split_at(m - rs).1.transpose().as_2d();

            let rhs = matrix.submatrix(m - rs, n - rt, rs, rt);
            triangular::matmul(
                dst.rb_mut(),
                BlockStructure::Rectangular,
                Conj::No,
                lhs.split_at(0, rt).2,
                BlockStructure::Rectangular,
                Conj::Yes,
                rhs.split_at(rt, 0).1,
                BlockStructure::UnitTriangularLower,
                Conj::No,
                None,
                factor,
                Parallelism::None,
            );

            let lhs = lhs.split_at(0, rt).3;
            let rhs = rhs.split_at(rt, 0).3;

            struct Gevm<'a, T> {
                dst: MatMut<'a, T>,
                lhs: MatRef<'a, T>,
                rhs: MatRef<'a, T>,
                beta: T,
            }
            impl<'a, T: ComplexField> pulp::WithSimd for Gevm<'a, T> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    let Self {
                        dst,
                        lhs,
                        rhs,
                        beta,
                    } = self;
                    let mut dst = dst.row(0);
                    let lhs = lhs.row(0).transpose();
                    fancy_assert!(rhs.row_stride() == 1);

                    for (i, rhs) in rhs.into_col_iter().enumerate() {
                        let dst = &mut dst[i];
                        *dst = *dst + beta * dot(simd, lhs, rhs);
                    }
                }
            }

            pulp::Arch::new().dispatch(Gevm {
                dst: dst.rb_mut(),
                lhs,
                rhs,
                beta: factor,
            });

            temp_mat_uninit! {
                let (mut tmp, _) = unsafe { temp_mat_uninit::<T>(rt, 1, stack.rb_mut()) };
            }

            triangular::matmul(
                tmp.rb_mut().transpose(),
                BlockStructure::Rectangular,
                Conj::No,
                householder_factor.rb().submatrix(i, size - rt, 1, rt),
                BlockStructure::Rectangular,
                Conj::No,
                householder_factor
                    .rb()
                    .submatrix(size - rt, size - rt, rt, rt),
                BlockStructure::TriangularUpper,
                Conj::No,
                None,
                T::one(),
                Parallelism::None,
            );
            householder_factor
                .rb_mut()
                .submatrix(i, size - rt, 1, rt)
                .row(0)
                .cwise()
                .zip(tmp.transpose().row(0))
                .for_each(|a, b| *a = *b);
        }
    }
}

pub fn apply_householder_on_the_left<T: ComplexField>(
    matrix: MatMut<'_, T>,
    essential: ColRef<'_, T>,
    householder_coeff: T,
    stack: DynStack<'_>,
) {
    fancy_assert!(matrix.nrows() == 1 + essential.nrows());
    let m = matrix.nrows();
    let n = matrix.ncols();
    if m == 1 {
        let factor = T::one() - householder_coeff;
        matrix.cwise().for_each(|e| *e = *e * factor);
    } else {
        let (first_row, last_rows) = matrix.split_at_row(1);
        let mut first_row = first_row.row(0);
        temp_mat_uninit! {
            let (tmp, _) = unsafe { temp_mat_uninit::<T>(n, 1, stack) };
        }
        let mut tmp = tmp.transpose().row(0);

        tmp.rb_mut()
            .cwise()
            .zip(first_row.rb())
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
            .zip(tmp.rb())
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

pub fn apply_block_householder_on_the_left<T: ComplexField>(
    matrix: MatMut<'_, T>,
    basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    forward: bool,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    fancy_assert!(matrix.nrows() == basis.nrows());
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
        matrix.rb().submatrix(0, 0, size, n),
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
        matrix.rb().submatrix(size, 0, m - size, n),
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

pub fn apply_householder_sequence_on_the_left<T: ComplexField>(
    mut matrix: MatMut<'_, T>,
    essentials: MatRef<'_, T>,
    householder_coeffs: ColRef<'_, T>,
    forward: bool,
    mut stack: DynStack<'_>,
) {
    let m = essentials.nrows();
    let n = matrix.ncols();

    if forward {
        for (k, (col, householder_coeff)) in essentials
            .into_col_iter()
            .zip(householder_coeffs)
            .enumerate()
        {
            let essential = col.subrows(k + 1, m - k - 1);
            apply_householder_on_the_left(
                matrix.rb_mut().submatrix(k, 0, m - k, n),
                essential,
                *householder_coeff,
                stack.rb_mut(),
            );
        }
    } else {
        for (k, (col, householder_coeff)) in essentials
            .into_col_iter()
            .zip(householder_coeffs)
            .enumerate()
            .rev()
        {
            let essential = col.subrows(k + 1, m - k - 1);
            apply_householder_on_the_left(
                matrix.rb_mut().submatrix(k, 0, m - k, n),
                essential,
                *householder_coeff,
                stack.rb_mut(),
            );
        }
    }
}
