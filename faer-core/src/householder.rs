use crate::{
    join_raw,
    mul::{
        matmul,
        triangular::{self, BlockStructure},
    },
    solve, temp_mat_uninit, ColMut, ColRef, ComplexField, Conj, MatMut, MatRef, Parallelism,
    RealField,
};
use num_traits::{Inv, One};

use assert2::assert as fancy_assert;
use dyn_stack::DynStack;
use reborrow::*;

pub fn make_householder_in_place<T: ComplexField>(
    essential: ColMut<'_, T>,
    head: T,
    tail_squared_norm: T::Real,
) -> (T, T) {
    let norm = ((head * head.conj()).real() + tail_squared_norm).sqrt();
    let sign = head.scale_real((head * head.conj()).real().sqrt().inv());

    let signed_norm = sign * T::from_real(norm);
    let head_with_beta = head + signed_norm;
    let inv = head_with_beta.inv();
    essential.cwise().for_each(|e| *e = *e * inv);

    let one_half = (T::Real::one() + T::Real::one()).inv();
    let tau = one_half * (T::Real::one() + tail_squared_norm * (inv * inv.conj()).real());
    (T::from_real(tau), -signed_norm)
}

pub fn apply_householder_on_the_left<T: ComplexField>(
    essential: ColRef<'_, T>,
    householder_coeff: T,
    conj_householder: Conj,
    matrix: MatMut<'_, T>,
    conj_mat: Conj,
    stack: DynStack<'_>,
) {
    fancy_assert!(matrix.nrows() == 1 + essential.nrows());
    let tau_inv = householder_coeff.inv();
    let m = matrix.nrows();
    let n = matrix.ncols();
    if m == 1 {
        let factor = T::one() - tau_inv;
        match conj_mat {
            Conj::No => matrix.cwise().for_each(|e| *e = *e * factor),
            Conj::Yes => matrix.cwise().for_each(|e| *e = (*e).conj() * factor),
        };
    } else {
        let (first_row, last_rows) = matrix.split_at_row(1);
        let mut first_row = first_row.row(0);
        temp_mat_uninit! {
            let (tmp, _) = unsafe { temp_mat_uninit::<T>(n, 1, stack) };
        }
        let mut tmp = tmp.transpose().row(0);

        match conj_mat {
            Conj::No => tmp
                .rb_mut()
                .cwise()
                .zip(first_row.rb())
                .for_each(|a, b| *a = *b),
            Conj::Yes => tmp
                .rb_mut()
                .cwise()
                .zip(first_row.rb())
                .for_each(|a, b| *a = (*b).conj()),
        }

        matmul(
            tmp.rb_mut().as_2d(),
            Conj::No,
            essential.transpose().as_2d(),
            Conj::Yes.compose(conj_householder),
            last_rows.rb(),
            conj_mat,
            Some(T::one()),
            T::one(),
            Parallelism::None,
        );

        match conj_mat {
            Conj::No => first_row
                .rb_mut()
                .cwise()
                .zip(tmp.rb())
                .for_each(|a, b| *a = *a - tau_inv * *b),
            Conj::Yes => first_row
                .rb_mut()
                .cwise()
                .zip(tmp.rb())
                .for_each(|a, b| *a = (*a).conj() - tau_inv * *b),
        };

        matmul(
            last_rows,
            conj_mat,
            essential.as_2d(),
            Conj::No.compose(conj_householder),
            tmp.rb().as_2d(),
            Conj::No,
            Some(T::one()),
            -tau_inv,
            Parallelism::None,
        )
    }
}

pub fn div_ceil(a: usize, b: usize) -> usize {
    let (div, rem) = (a / b, a % b);
    if rem == 0 {
        div
    } else {
        div + 1
    }
}

pub fn upgrade_householder_factor<T: ComplexField>(
    mut householder_factor: MatMut<'_, T>,
    essentials: MatRef<'_, T>,
    blocksize: usize,
    prev_blocksize: usize,
    parallelism: Parallelism,
) {
    if blocksize == prev_blocksize || householder_factor.nrows() <= prev_blocksize {
        return;
    }

    fancy_assert!(householder_factor.nrows() == householder_factor.ncols());

    let block_count = div_ceil(householder_factor.nrows(), blocksize);

    if block_count > 1 {
        fancy_assert!(blocksize > prev_blocksize);
        fancy_assert!(blocksize % prev_blocksize == 0);
        let idx = (block_count / 2) * blocksize;
        let (tau_tl, _, _, tau_br) = householder_factor.split_at(idx, idx);
        let (basis_left, basis_right) = essentials.split_at_col(idx);
        let basis_right =
            basis_right.submatrix(idx, 0, essentials.nrows() - idx, basis_right.ncols());
        join_raw(
            |parallelism| {
                upgrade_householder_factor(
                    tau_tl,
                    basis_left,
                    blocksize,
                    prev_blocksize,
                    parallelism,
                )
            },
            |parallelism| {
                upgrade_householder_factor(
                    tau_br,
                    basis_right,
                    blocksize,
                    prev_blocksize,
                    parallelism,
                )
            },
            parallelism,
        );
        return;
    }

    if prev_blocksize < 8 {
        // pretend that prev_blocksize == 1, recompute whole top half of matrix

        let (basis_top, basis_bot) = essentials.split_at_row(essentials.ncols());
        triangular::matmul(
            householder_factor.rb_mut(),
            BlockStructure::UnitTriangularUpper,
            Conj::No,
            basis_top.transpose(),
            BlockStructure::UnitTriangularUpper,
            Conj::Yes,
            basis_top,
            BlockStructure::UnitTriangularLower,
            Conj::No,
            None,
            T::one(),
            parallelism,
        );
        triangular::matmul(
            householder_factor.rb_mut(),
            BlockStructure::UnitTriangularUpper,
            Conj::No,
            basis_bot.transpose(),
            BlockStructure::Rectangular,
            Conj::Yes,
            basis_bot,
            BlockStructure::Rectangular,
            Conj::No,
            Some(T::one()),
            T::one(),
            parallelism,
        );
    } else {
        let prev_block_count = div_ceil(householder_factor.nrows(), prev_blocksize);

        let idx = (prev_block_count / 2) * prev_blocksize;
        let (tau_tl, mut tau_tr, _, tau_br) = householder_factor.split_at(idx, idx);
        let (basis_left, basis_right) = essentials.split_at_col(idx);
        let basis_right =
            basis_right.submatrix(idx, 0, essentials.nrows() - idx, basis_right.ncols());

        join_raw(
            |parallelism| {
                join_raw(
                    |parallelism| {
                        upgrade_householder_factor(
                            tau_tl,
                            basis_left,
                            blocksize,
                            prev_blocksize,
                            parallelism,
                        )
                    },
                    |parallelism| {
                        upgrade_householder_factor(
                            tau_br,
                            basis_right,
                            blocksize,
                            prev_blocksize,
                            parallelism,
                        )
                    },
                    parallelism,
                );
            },
            |parallelism| {
                let basis_left = basis_left.submatrix(idx, 0, basis_left.nrows() - idx, idx);
                let (basis_left_top, basis_left_bot) = basis_left.split_at_row(basis_right.ncols());
                let (basis_right_top, basis_right_bot) =
                    basis_right.split_at_row(basis_right.ncols());

                triangular::matmul(
                    tau_tr.rb_mut(),
                    BlockStructure::Rectangular,
                    Conj::No,
                    basis_left_top.transpose(),
                    BlockStructure::Rectangular,
                    Conj::Yes,
                    basis_right_top,
                    BlockStructure::UnitTriangularLower,
                    Conj::No,
                    None,
                    T::one(),
                    parallelism,
                );
                matmul(
                    tau_tr.rb_mut(),
                    Conj::No,
                    basis_left_bot.transpose(),
                    Conj::Yes,
                    basis_right_bot,
                    Conj::No,
                    Some(T::one()),
                    T::one(),
                    parallelism,
                );
            },
            parallelism,
        );
    }
}

pub fn apply_block_householder_on_the_left<T: ComplexField>(
    essentials: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    conj_householder: Conj,
    matrix: MatMut<'_, T>,
    conj_mat: Conj,
    forward: bool,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let (essentials_top, essentials_bot) = essentials.split_at_row(essentials.ncols());
    let bs = householder_factor.nrows();
    let n = matrix.ncols();

    let (mut matrix_top, mut matrix_bot) = matrix.split_at_row(bs);

    // essentials* × mat
    temp_mat_uninit! {
        let (mut tmp, _) = unsafe { temp_mat_uninit::<T>(bs, n, stack) };
    }

    triangular::matmul(
        tmp.rb_mut(),
        BlockStructure::Rectangular,
        Conj::No,
        essentials_top.transpose(),
        BlockStructure::UnitTriangularUpper,
        Conj::Yes.compose(conj_householder),
        matrix_top.rb(),
        BlockStructure::Rectangular,
        Conj::No.compose(conj_mat),
        None,
        T::one(),
        parallelism,
    );
    matmul(
        tmp.rb_mut(),
        Conj::No,
        essentials_bot.transpose(),
        Conj::Yes.compose(conj_householder),
        matrix_bot.rb(),
        Conj::No.compose(conj_mat),
        Some(T::one()),
        T::one(),
        parallelism,
    );

    // [T^-1|T^-*] × essentials* × tmp
    if forward {
        solve::solve_lower_triangular_in_place(
            householder_factor.transpose(),
            Conj::Yes.compose(conj_householder),
            tmp.rb_mut(),
            Conj::No,
            parallelism,
        );
    } else {
        solve::solve_upper_triangular_in_place(
            householder_factor,
            Conj::No.compose(conj_householder),
            tmp.rb_mut(),
            Conj::No,
            parallelism,
        );
    }

    // essentials × [T^-1|T^-*] × essentials* × tmp
    join_raw(
        |_| {
            triangular::matmul(
                matrix_top.rb_mut(),
                BlockStructure::Rectangular,
                Conj::No.compose(conj_mat),
                essentials_top,
                BlockStructure::UnitTriangularLower,
                Conj::No.compose(conj_householder),
                tmp.rb(),
                BlockStructure::Rectangular,
                Conj::No,
                Some(T::one()),
                -T::one(),
                parallelism,
            );
        },
        |_| {
            matmul(
                matrix_bot.rb_mut(),
                Conj::No.compose(conj_mat),
                essentials_bot,
                Conj::No.compose(conj_householder),
                tmp.rb(),
                Conj::No,
                Some(T::one()),
                -T::one(),
                parallelism,
            );
        },
        parallelism,
    );
}

pub fn apply_block_householder_sequence_on_the_left<T: ComplexField>(
    essentials: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    conj_householder: Conj,
    matrix: MatMut<'_, T>,
    conj_mat: Conj,
    forward: bool,
    parallelism: Parallelism,
    mut stack: DynStack<'_>,
) {
    let mut matrix = matrix;
    let blocksize = householder_factor.nrows();
    let m = essentials.nrows();
    let k = matrix.ncols();

    let size = householder_factor.ncols();

    let mut conj_mat = conj_mat;
    if size == 0 && conj_mat == Conj::Yes {
        matrix.cwise().for_each(|e| *e = (*e).conj());
        return;
    }

    if forward {
        let mut j = 0;
        while j < size {
            let bs = blocksize.min(size - j);
            let essentials = essentials.submatrix(j, j, m - j, bs);
            let householder = householder_factor.submatrix(0, j, bs, bs);

            apply_block_householder_on_the_left(
                essentials,
                householder,
                conj_householder,
                matrix.rb_mut().submatrix(j, 0, m - j, k),
                conj_mat,
                forward,
                parallelism,
                stack.rb_mut(),
            );

            conj_mat = Conj::No;
            j += bs;
        }
    } else {
        let mut j = size;
        let mut bs = size % blocksize;
        if bs == 0 {
            bs = blocksize
        }

        while j > 0 {
            j -= bs;

            let essentials = essentials.submatrix(j, j, m - j, bs);
            let householder = householder_factor.submatrix(0, j, bs, bs);

            apply_block_householder_on_the_left(
                essentials,
                householder,
                conj_householder,
                matrix.rb_mut().submatrix(j, 0, m - j, k),
                conj_mat,
                forward,
                parallelism,
                stack.rb_mut(),
            );

            conj_mat = Conj::No;
            bs = blocksize;
        }
    }
}
