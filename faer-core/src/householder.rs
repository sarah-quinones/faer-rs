//! Block Householder transformations.
//!
//! A Householder reflection is linear transformation that describes a relfection about a
//! hyperplane that crosses the origin of the space.
//!
//! Let $v$ be a unit vector that is orthogonal to the hyperplane. Then the corresponding
//! Householder transformation in matrix form is $I - 2vv^H$, where $I$ is the identity matrix.
//!
//! In practice, a non unit vector $v$ is used, so the transformation is written as
//! $$H = I - \frac{vv^H}{\tau}.$$
//!
//! A block Householder transformation is a sequence of such transformations
//! $H_0, H_1, \dots, H_{b -1 }$ applied one after the other, with the restriction that the first
//! $i$ components of the vector $v_i$ of the $i$-th transformation are zero, and the component at
//! index $i$ is one.
//!
//! The matrix $V = [v_0\ v_1\ \dots\ v_{b-1}]$ is thus a lower trapezoidal matrix with unit
//! diagonal. We call it the Householder basis.
//!
//! There exists a unique upper triangular matrix $T$, that we call the Householder factor, such
//! that $$H_0 \times \dots \times H_{b-1} = I - VT^{-1}V^H.$$
//!
//! A block Householder sequence is a sequence of such transformations, composed of two matrices:
//! - a lower trapezoidal matrix with unit diagonal, which is the horizontal concatenation of the
//! bases of each block Householder transformation,
//! - a horizontal concatenation of the Householder factors.
//!
//! Examples on how to create and manipulate block Householder sequences are provided in the
//! documentation of the QR module.

use crate::{
    join_raw,
    mul::{
        matmul,
        triangular::{self, BlockStructure},
    },
    solve, temp_mat_req, temp_mat_uninit, ColMut, ColRef, ComplexField, Conj, MatMut, MatRef,
    Parallelism, RealField,
};
use num_traits::{Inv, One};

use assert2::assert as fancy_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use reborrow::*;

#[doc(hidden)]
pub fn make_householder_in_place<T: ComplexField>(
    essential: Option<ColMut<'_, T>>,
    head: T,
    tail_squared_norm: T::Real,
) -> (T, T) {
    let norm = ((head * head.conj()).real() + tail_squared_norm).sqrt();

    let sign = if head != T::zero() {
        head.scale_real((head * head.conj()).real().sqrt().inv())
    } else {
        T::one()
    };

    let signed_norm = sign * T::from_real(norm);
    let head_with_beta = head + signed_norm;
    let inv = head_with_beta.inv();

    if let Some(essential) = essential {
        essential.cwise().for_each(|e| *e = *e * inv);
    }

    let one_half = (T::Real::one() + T::Real::one()).inv();
    let tau = one_half * (T::Real::one() + tail_squared_norm * (inv * inv.conj()).real());
    (T::from_real(tau), -signed_norm)
}

#[doc(hidden)]
pub fn apply_householder_on_the_left_in_place<T: ComplexField>(
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

fn div_ceil(a: usize, b: usize) -> usize {
    let (div, rem) = (a / b, a % b);
    if rem == 0 {
        div
    } else {
        div + 1
    }
}

#[doc(hidden)]
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

/// Computes the size and alignment of required workspace for applying a block Householder
/// transformation to a right-hand-side matrix in place.
pub fn apply_block_householder_on_the_left_in_place_req<T: 'static>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<T>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying the transpose of a block
/// Householder transformation to a right-hand-side matrix in place.
pub fn apply_block_householder_transpose_on_the_left_in_place_req<T: 'static>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<T>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying a block Householder
/// transformation to a left-hand-side matrix in place.
pub fn apply_block_householder_on_the_right_in_place_req<T: 'static>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<T>(blocksize, lhs_nrows)
}

/// Computes the size and alignment of required workspace for applying the transpose of a block
/// Householder transformation to a left-hand-side matrix in place.
pub fn apply_block_householder_transpose_on_the_right_in_place_req<T: 'static>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<T>(blocksize, lhs_nrows)
}

/// Computes the size and alignment of required workspace for applying the transpose of a sequence
/// of block Householder transformations to a right-hand-side matrix in place.
pub fn apply_block_householder_sequence_transpose_on_the_left_in_place_req<T: 'static>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<T>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying a sequence of block
/// Householder transformations to a right-hand-side matrix in place.
pub fn apply_block_householder_sequence_on_the_left_in_place_req<T: 'static>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<T>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying the transpose of a sequence
/// of block Householder transformations to a left-hand-side matrix in place.
pub fn apply_block_householder_sequence_transpose_on_the_right_in_place_req<T: 'static>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<T>(blocksize, lhs_nrows)
}

/// Computes the size and alignment of required workspace for applying a sequence of block
/// Householder transformations to a left-hand-side matrix in place.
pub fn apply_block_householder_sequence_on_the_right_in_place_req<T: 'static>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<T>(blocksize, lhs_nrows)
}

#[track_caller]
fn apply_block_householder_on_the_left_in_place_generic<T: ComplexField>(
    householder_basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    conj_lhs: Conj,
    matrix: MatMut<'_, T>,
    conj_rhs: Conj,
    forward: bool,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    fancy_assert!(householder_factor.nrows() == householder_factor.ncols());
    fancy_assert!(householder_basis.ncols() == householder_factor.nrows());
    fancy_assert!(matrix.nrows() == householder_basis.nrows());

    let (essentials_top, essentials_bot) =
        householder_basis.split_at_row(householder_basis.ncols());
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
        Conj::Yes.compose(conj_lhs),
        matrix_top.rb(),
        BlockStructure::Rectangular,
        Conj::No.compose(conj_rhs),
        None,
        T::one(),
        parallelism,
    );
    matmul(
        tmp.rb_mut(),
        Conj::No,
        essentials_bot.transpose(),
        Conj::Yes.compose(conj_lhs),
        matrix_bot.rb(),
        Conj::No.compose(conj_rhs),
        Some(T::one()),
        T::one(),
        parallelism,
    );

    // [T^-1|T^-*] × essentials* × tmp
    if forward {
        solve::solve_lower_triangular_in_place(
            householder_factor.transpose(),
            Conj::Yes.compose(conj_lhs),
            tmp.rb_mut(),
            Conj::No,
            parallelism,
        );
    } else {
        solve::solve_upper_triangular_in_place(
            householder_factor,
            Conj::No.compose(conj_lhs),
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
                Conj::No.compose(conj_rhs),
                essentials_top,
                BlockStructure::UnitTriangularLower,
                Conj::No.compose(conj_lhs),
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
                Conj::No.compose(conj_rhs),
                essentials_bot,
                Conj::No.compose(conj_lhs),
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

/// Computes the product of the matrix, multiplied by the given block Householder transformation,
/// and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_on_the_right_in_place<T: ComplexField>(
    householder_basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    conj_rhs: Conj,
    matrix: MatMut<'_, T>,
    conj_lhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_transpose_on_the_left_in_place(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose(),
        conj_lhs,
        parallelism,
        stack,
    )
}

/// Computes the product of the matrix, multiplied by the transpose of the given block Householder
/// transformation, and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_transpose_on_the_right_in_place<T: ComplexField>(
    householder_basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    conj_rhs: Conj,
    matrix: MatMut<'_, T>,
    conj_lhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_on_the_left_in_place(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose(),
        conj_lhs,
        parallelism,
        stack,
    )
}

/// Computes the product of the given block Householder transformation, multiplied by `matrix`, and
/// stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_on_the_left_in_place<T: ComplexField>(
    householder_basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    conj_lhs: Conj,
    matrix: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_on_the_left_in_place_generic(
        householder_basis,
        householder_factor,
        conj_lhs,
        matrix,
        conj_rhs,
        false,
        parallelism,
        stack,
    )
}

/// Computes the product of the transpose of the given block Householder transformation, multiplied
/// by `matrix`, and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_transpose_on_the_left_in_place<T: ComplexField>(
    householder_basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    conj_lhs: Conj,
    matrix: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_on_the_left_in_place_generic(
        householder_basis,
        householder_factor,
        conj_lhs.compose(Conj::Yes),
        matrix,
        conj_rhs,
        true,
        parallelism,
        stack,
    )
}

/// Computes the product of a sequence of block Householder transformations given by
/// `householder_basis` and `householder_factor`, multiplied by `matrix`, and stores the result in
/// `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_on_the_left_in_place<T: ComplexField>(
    householder_basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    conj_lhs: Conj,
    matrix: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut matrix = matrix;
    let mut stack = stack;

    let blocksize = householder_factor.nrows();

    fancy_assert!(blocksize > 0);
    let m = householder_basis.nrows();
    let k = matrix.ncols();

    let size = householder_factor.ncols();

    let mut conj_mat = conj_rhs;
    if size == 0 && conj_mat == Conj::Yes {
        matrix.cwise().for_each(|e| *e = (*e).conj());
        return;
    }

    let mut j = size;
    let mut bs = size % blocksize;
    if bs == 0 {
        bs = blocksize
    }

    while j > 0 {
        j -= bs;

        let essentials = householder_basis.submatrix(j, j, m - j, bs);
        let householder = householder_factor.submatrix(0, j, bs, bs);

        apply_block_householder_on_the_left_in_place(
            essentials,
            householder,
            conj_lhs,
            matrix.rb_mut().submatrix(j, 0, m - j, k),
            conj_mat,
            parallelism,
            stack.rb_mut(),
        );

        conj_mat = Conj::No;
        bs = blocksize;
    }
}

/// Computes the product of the transpose of a sequence block Householder transformations given by
/// `householder_basis` and `householder_factor`, multiplied by `matrix`, and stores the result in
/// `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_transpose_on_the_left_in_place<T: ComplexField>(
    householder_basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    conj_lhs: Conj,
    matrix: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut matrix = matrix;
    let mut stack = stack;
    let blocksize = householder_factor.nrows();
    fancy_assert!(blocksize > 0);
    let m = householder_basis.nrows();
    let k = matrix.ncols();

    let size = householder_factor.ncols();

    let mut conj_mat = conj_rhs;
    if size == 0 && conj_mat == Conj::Yes {
        matrix.cwise().for_each(|e| *e = (*e).conj());
        return;
    }

    let mut j = 0;
    while j < size {
        let bs = blocksize.min(size - j);
        let essentials = householder_basis.submatrix(j, j, m - j, bs);
        let householder = householder_factor.submatrix(0, j, bs, bs);

        apply_block_householder_transpose_on_the_left_in_place(
            essentials,
            householder,
            conj_lhs,
            matrix.rb_mut().submatrix(j, 0, m - j, k),
            conj_mat,
            parallelism,
            stack.rb_mut(),
        );

        conj_mat = Conj::No;
        j += bs;
    }
}

/// Computes the product of `matrix`, multiplied by a sequence of block Householder transformations
/// given by `householder_basis` and `householder_factor`, and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_on_the_right_in_place<T: ComplexField>(
    householder_basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    conj_rhs: Conj,
    matrix: MatMut<'_, T>,
    conj_lhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_sequence_transpose_on_the_left_in_place(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose(),
        conj_lhs,
        parallelism,
        stack,
    )
}

/// Computes the product of `matrix`, multiplied by the transpose of a sequence of block Householder
/// transformations given by `householder_basis` and `householder_factor`, and stores the result in
/// `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_transpose_on_the_right_in_place<T: ComplexField>(
    householder_basis: MatRef<'_, T>,
    householder_factor: MatRef<'_, T>,
    conj_rhs: Conj,
    matrix: MatMut<'_, T>,
    conj_lhs: Conj,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_sequence_on_the_left_in_place(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose(),
        conj_lhs,
        parallelism,
        stack,
    )
}
