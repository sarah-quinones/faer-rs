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
        inner_prod, matmul, matmul_with_conj,
        triangular::{self, BlockStructure},
    },
    solve, temp_mat_req, temp_mat_uninit, zipped, ComplexField, Conj, Entity, MatMut, MatRef,
    Parallelism,
};
use assert2::assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use reborrow::*;

#[doc(hidden)]
#[inline]
pub fn make_householder_in_place<E: ComplexField>(
    essential: Option<MatMut<'_, E>>,
    head: E,
    tail_squared_norm: E::Real,
) -> (E, E) {
    let one_half = E::Real::from_f64(0.5);
    let head_squared_norm = head.mul(&head.conj()).real();
    let norm = head_squared_norm.add(&tail_squared_norm).sqrt();

    let sign = if head != E::zero() {
        head.scale_real(&head_squared_norm.sqrt().inv())
    } else {
        E::one()
    };

    let signed_norm = sign.mul(&E::from_real(norm));
    let head_with_beta = head.add(&signed_norm);
    let head_with_beta_inv = head_with_beta.inv();

    if head_with_beta != E::zero() {
        if let Some(essential) = essential {
            assert!(essential.ncols() == 1);
            zipped!(essential).for_each(|mut e| e.write(e.read().mul(&head_with_beta_inv)));
        }
        let tau = one_half.mul(&E::Real::one().add(
            &tail_squared_norm.mul(&(head_with_beta_inv.mul(&head_with_beta_inv.conj())).real()),
        ));
        (E::from_real(tau), signed_norm.neg())
    } else {
        (E::from_real(E::Real::zero().inv()), E::zero())
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
pub fn upgrade_householder_factor<E: ComplexField>(
    mut householder_factor: MatMut<'_, E>,
    essentials: MatRef<'_, E>,
    blocksize: usize,
    prev_blocksize: usize,
    parallelism: Parallelism,
) {
    if blocksize == prev_blocksize || householder_factor.nrows() <= prev_blocksize {
        return;
    }

    assert!(householder_factor.nrows() == householder_factor.ncols());

    let block_count = div_ceil(householder_factor.nrows(), blocksize);

    if block_count > 1 {
        assert!(blocksize > prev_blocksize);
        assert!(blocksize % prev_blocksize == 0);
        let idx = (block_count / 2) * blocksize;
        let [tau_tl, _, _, tau_br] = householder_factor.split_at(idx, idx);
        let [basis_left, basis_right] = essentials.split_at_col(idx);
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

        let [basis_top, basis_bot] = essentials.split_at_row(essentials.ncols());
        triangular::matmul(
            householder_factor.rb_mut(),
            BlockStructure::UnitTriangularUpper,
            basis_top.adjoint(),
            BlockStructure::UnitTriangularUpper,
            basis_top,
            BlockStructure::UnitTriangularLower,
            None,
            E::one(),
            parallelism,
        );
        triangular::matmul(
            householder_factor.rb_mut(),
            BlockStructure::UnitTriangularUpper,
            basis_bot.adjoint(),
            BlockStructure::Rectangular,
            basis_bot,
            BlockStructure::Rectangular,
            Some(E::one()),
            E::one(),
            parallelism,
        );
    } else {
        let prev_block_count = div_ceil(householder_factor.nrows(), prev_blocksize);

        let idx = (prev_block_count / 2) * prev_blocksize;
        let [tau_tl, mut tau_tr, _, tau_br] = householder_factor.split_at(idx, idx);
        let [basis_left, basis_right] = essentials.split_at_col(idx);
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
                let [basis_left_top, basis_left_bot] = basis_left.split_at_row(basis_right.ncols());
                let [basis_right_top, basis_right_bot] =
                    basis_right.split_at_row(basis_right.ncols());

                triangular::matmul(
                    tau_tr.rb_mut(),
                    BlockStructure::Rectangular,
                    basis_left_top.adjoint(),
                    BlockStructure::Rectangular,
                    basis_right_top,
                    BlockStructure::UnitTriangularLower,
                    None,
                    E::one(),
                    parallelism,
                );
                matmul(
                    tau_tr.rb_mut(),
                    basis_left_bot.adjoint(),
                    basis_right_bot,
                    Some(E::one()),
                    E::one(),
                    parallelism,
                );
            },
            parallelism,
        );
    }
}

/// Computes the size and alignment of required workspace for applying a block Householder
/// transformation to a right-hand-side matrix in place.
pub fn apply_block_householder_on_the_left_in_place_req<E: Entity>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<E>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying the transpose of a block
/// Householder transformation to a right-hand-side matrix in place.
pub fn apply_block_householder_transpose_on_the_left_in_place_req<E: Entity>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<E>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying a block Householder
/// transformation to a left-hand-side matrix in place.
pub fn apply_block_householder_on_the_right_in_place_req<E: Entity>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<E>(blocksize, lhs_nrows)
}

/// Computes the size and alignment of required workspace for applying the transpose of a block
/// Householder transformation to a left-hand-side matrix in place.
pub fn apply_block_householder_transpose_on_the_right_in_place_req<E: Entity>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<E>(blocksize, lhs_nrows)
}

/// Computes the size and alignment of required workspace for applying the transpose of a sequence
/// of block Householder transformations to a right-hand-side matrix in place.
pub fn apply_block_householder_sequence_transpose_on_the_left_in_place_req<E: Entity>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<E>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying a sequence of block
/// Householder transformations to a right-hand-side matrix in place.
pub fn apply_block_householder_sequence_on_the_left_in_place_req<E: Entity>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<E>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying the transpose of a sequence
/// of block Householder transformations to a left-hand-side matrix in place.
pub fn apply_block_householder_sequence_transpose_on_the_right_in_place_req<E: Entity>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<E>(blocksize, lhs_nrows)
}

/// Computes the size and alignment of required workspace for applying a sequence of block
/// Householder transformations to a left-hand-side matrix in place.
pub fn apply_block_householder_sequence_on_the_right_in_place_req<E: Entity>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_req::<E>(blocksize, lhs_nrows)
}

#[track_caller]
fn apply_block_householder_on_the_left_in_place_generic<E: ComplexField>(
    householder_basis: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    matrix: MatMut<'_, E>,
    forward: bool,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    assert!(householder_factor.nrows() == householder_factor.ncols());
    assert!(householder_basis.ncols() == householder_factor.nrows());
    assert!(matrix.nrows() == householder_basis.nrows());

    let bs = householder_factor.nrows();
    if E::HAS_SIMD && householder_basis.row_stride() == 1 && matrix.row_stride() == 1 && bs == 1 {
        let arch = pulp::Arch::new();

        if matches!(conj_lhs, Conj::No) {
            struct ApplyOnLeftNoConj<'a, E: ComplexField> {
                tau_inv: E,
                essential: MatRef<'a, E>,
                rhs: MatMut<'a, E>,
            }

            impl<E: ComplexField> pulp::WithSimd for ApplyOnLeftNoConj<'_, E> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    let Self {
                        tau_inv,
                        essential,
                        mut rhs,
                    } = self;
                    debug_assert_eq!(essential.row_stride(), 1);
                    debug_assert_eq!(rhs.row_stride(), 1);

                    let n = rhs.ncols();

                    if rhs.nrows() == 0 {
                        return;
                    }

                    let m = rhs.nrows() - 1;
                    let lane_count =
                        core::mem::size_of::<E::SimdUnit<S>>() / core::mem::size_of::<E::Unit>();
                    let prefix = m % lane_count;

                    let essential = E::map(
                        essential.as_ptr(),
                        #[inline(always)]
                        |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
                    );

                    let (essential_scalar, essential_simd) = E::unzip(E::map(
                        E::copy(&essential),
                        #[inline(always)]
                        |slice| slice.split_at(prefix),
                    ));
                    let essential_simd = crate::simd::slice_as_simd::<E, S>(essential_simd).0;

                    for idx in 0..n {
                        let col = rhs.rb_mut().col(idx);
                        let [mut col_head, col_tail] = col.split_at_row(1);

                        let col_head_ = col_head.read(0, 0);
                        let col_tail = E::map(
                            col_tail.as_ptr(),
                            #[inline(always)]
                            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
                        );

                        let dot = col_head_.add(
                            &inner_prod::AccConjAxB::<'_, E> {
                                a: E::rb(E::as_ref(&essential)),
                                b: E::rb(E::as_ref(&col_tail)),
                            }
                            .with_simd(simd),
                        );

                        let k = (dot.mul(&tau_inv)).neg();
                        col_head.write(0, 0, col_head_.add(&k));

                        let (col_tail_scalar, col_tail_simd) = E::unzip(E::map(
                            col_tail,
                            #[inline(always)]
                            |slice| slice.split_at_mut(prefix),
                        ));
                        let col_tail_simd = crate::simd::slice_as_mut_simd::<E, S>(col_tail_simd).0;

                        for (a, b) in E::into_iter(col_tail_scalar)
                            .zip(E::into_iter(E::copy(&essential_scalar)))
                        {
                            let mut a_ = E::from_units(E::deref(E::rb(E::as_ref(&a))));
                            let b = E::from_units(E::deref(b));
                            a_ = a_.add(&k.mul(&b));

                            E::map(
                                E::zip(a, a_.into_units()),
                                #[inline(always)]
                                |(a, a_)| *a = a_,
                            );
                        }

                        let k = E::simd_splat(simd, k);
                        for (a, b) in
                            E::into_iter(col_tail_simd).zip(E::into_iter(E::copy(&essential_simd)))
                        {
                            let mut a_ = E::deref(E::rb(E::as_ref(&a)));
                            let b = E::deref(b);
                            a_ = E::simd_mul_adde(simd, E::copy(&k), E::copy(&b), a_);

                            E::map(
                                E::zip(a, a_),
                                #[inline(always)]
                                |(a, a_)| *a = a_,
                            );
                        }
                    }
                }
            }

            arch.dispatch(ApplyOnLeftNoConj {
                tau_inv: E::from_real(householder_factor.read(0, 0).real().inv()),
                essential: householder_basis.split_at_row(1)[1],
                rhs: matrix,
            });
        } else {
            struct ApplyOnLeftConj<'a, E: ComplexField> {
                tau_inv: E,
                essential: MatRef<'a, E>,
                rhs: MatMut<'a, E>,
            }

            impl<E: ComplexField> pulp::WithSimd for ApplyOnLeftConj<'_, E> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    let Self {
                        tau_inv,
                        essential,
                        mut rhs,
                    } = self;
                    debug_assert_eq!(essential.row_stride(), 1);
                    debug_assert_eq!(rhs.row_stride(), 1);

                    let n = rhs.ncols();

                    if rhs.nrows() == 0 {
                        return;
                    }

                    let m = rhs.nrows() - 1;
                    let lane_count =
                        core::mem::size_of::<E::SimdUnit<S>>() / core::mem::size_of::<E::Unit>();
                    let prefix = m % lane_count;

                    let essential = E::map(
                        essential.as_ptr(),
                        #[inline(always)]
                        |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
                    );

                    let (essential_scalar, essential_simd) = E::unzip(E::map(
                        E::copy(&essential),
                        #[inline(always)]
                        |slice| slice.split_at(prefix),
                    ));
                    let essential_simd = crate::simd::slice_as_simd::<E, S>(essential_simd).0;

                    for idx in 0..n {
                        let col = rhs.rb_mut().col(idx);
                        let [mut col_head, col_tail] = col.split_at_row(1);

                        let col_head_ = col_head.read(0, 0);
                        let col_tail = E::map(
                            col_tail.as_ptr(),
                            #[inline(always)]
                            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
                        );

                        let dot = col_head_.add(
                            &inner_prod::AccNoConjAxB::<'_, E> {
                                a: E::rb(E::as_ref(&essential)),
                                b: E::rb(E::as_ref(&col_tail)),
                            }
                            .with_simd(simd),
                        );

                        let k = (dot.mul(&tau_inv)).neg();
                        col_head.write(0, 0, col_head_.add(&k));

                        let (col_tail_scalar, col_tail_simd) = E::unzip(E::map(
                            col_tail,
                            #[inline(always)]
                            |slice| slice.split_at_mut(prefix),
                        ));
                        let col_tail_simd = crate::simd::slice_as_mut_simd::<E, S>(col_tail_simd).0;

                        for (a, b) in E::into_iter(col_tail_scalar)
                            .zip(E::into_iter(E::copy(&essential_scalar)))
                        {
                            let mut a_ = E::from_units(E::deref(E::rb(E::as_ref(&a))));
                            let b = E::from_units(E::deref(b));
                            a_ = a_.add(&k.mul(&b.conj()));

                            E::map(
                                E::zip(a, a_.into_units()),
                                #[inline(always)]
                                |(a, a_)| *a = a_,
                            );
                        }

                        let k = E::simd_splat(simd, k);
                        for (a, b) in
                            E::into_iter(col_tail_simd).zip(E::into_iter(E::copy(&essential_simd)))
                        {
                            let mut a_ = E::deref(E::rb(E::as_ref(&a)));
                            let b = E::deref(b);
                            a_ = E::simd_conj_mul_adde(simd, E::copy(&b), E::copy(&k), a_);

                            E::map(
                                E::zip(a, a_),
                                #[inline(always)]
                                |(a, a_)| *a = a_,
                            );
                        }
                    }
                }
            }

            arch.dispatch(ApplyOnLeftConj {
                tau_inv: E::from_real(householder_factor.read(0, 0).real().inv()),
                essential: householder_basis.split_at_row(1)[1],
                rhs: matrix,
            });
        }
    } else {
        let [essentials_top, essentials_bot] = householder_basis.split_at_row(bs);
        let n = matrix.ncols();

        let [mut matrix_top, mut matrix_bot] = matrix.split_at_row(bs);

        // essentials* × mat
        let (mut tmp, _) = unsafe { temp_mat_uninit::<E>(bs, n, stack) };
        let mut tmp = tmp.as_mut();

        triangular::matmul_with_conj(
            tmp.rb_mut(),
            BlockStructure::Rectangular,
            essentials_top.transpose(),
            BlockStructure::UnitTriangularUpper,
            Conj::Yes.compose(conj_lhs),
            matrix_top.rb(),
            BlockStructure::Rectangular,
            Conj::No,
            None,
            E::one(),
            parallelism,
        );
        matmul_with_conj(
            tmp.rb_mut(),
            essentials_bot.transpose(),
            Conj::Yes.compose(conj_lhs),
            matrix_bot.rb(),
            Conj::No,
            Some(E::one()),
            E::one(),
            parallelism,
        );

        // [T^-1|T^-*] × essentials* × tmp
        if forward {
            solve::solve_lower_triangular_in_place_with_conj(
                householder_factor.transpose(),
                Conj::Yes.compose(conj_lhs),
                tmp.rb_mut(),
                parallelism,
            );
        } else {
            solve::solve_upper_triangular_in_place_with_conj(
                householder_factor,
                Conj::No.compose(conj_lhs),
                tmp.rb_mut(),
                parallelism,
            );
        }

        // essentials × [T^-1|T^-*] × essentials* × tmp
        triangular::matmul_with_conj(
            matrix_top.rb_mut(),
            BlockStructure::Rectangular,
            essentials_top,
            BlockStructure::UnitTriangularLower,
            Conj::No.compose(conj_lhs),
            tmp.rb(),
            BlockStructure::Rectangular,
            Conj::No,
            Some(E::one()),
            E::one().neg(),
            parallelism,
        );
        matmul_with_conj(
            matrix_bot.rb_mut(),
            essentials_bot,
            Conj::No.compose(conj_lhs),
            tmp.rb(),
            Conj::No,
            Some(E::one()),
            E::one().neg(),
            parallelism,
        );
    }
}

/// Computes the product of the matrix, multiplied by the given block Householder transformation,
/// and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_on_the_right_in_place_with_conj<E: ComplexField>(
    householder_basis: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_rhs: Conj,
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_transpose_on_the_left_in_place_with_conj(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose(),
        parallelism,
        stack,
    )
}

/// Computes the product of the matrix, multiplied by the transpose of the given block Householder
/// transformation, and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_transpose_on_the_right_in_place_with_conj<E: ComplexField>(
    householder_basis: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_rhs: Conj,
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_on_the_left_in_place_with_conj(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose(),
        parallelism,
        stack,
    )
}

/// Computes the product of the given block Householder transformation, multiplied by `matrix`, and
/// stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_on_the_left_in_place_with_conj<E: ComplexField>(
    householder_basis: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_on_the_left_in_place_generic(
        householder_basis,
        householder_factor,
        conj_lhs,
        matrix,
        false,
        parallelism,
        stack,
    )
}

/// Computes the product of the transpose of the given block Householder transformation, multiplied
/// by `matrix`, and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_transpose_on_the_left_in_place_with_conj<E: ComplexField>(
    householder_basis: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_on_the_left_in_place_generic(
        householder_basis,
        householder_factor,
        conj_lhs.compose(Conj::Yes),
        matrix,
        true,
        parallelism,
        stack,
    )
}

/// Computes the product of a sequence of block Householder transformations given by
/// `householder_basis` and `householder_factor`, multiplied by `matrix`, and stores the result in
/// `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_on_the_left_in_place_with_conj<E: ComplexField>(
    householder_basis: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut matrix = matrix;
    let mut stack = stack;

    let blocksize = householder_factor.nrows();

    assert!(blocksize > 0);
    let m = householder_basis.nrows();
    let k = matrix.ncols();

    let size = householder_factor.ncols();

    let mut j = size;
    let mut bs = size % blocksize;
    if bs == 0 {
        bs = blocksize
    }

    while j > 0 {
        j -= bs;

        let essentials = householder_basis.submatrix(j, j, m - j, bs);
        let householder = householder_factor.submatrix(0, j, bs, bs);

        apply_block_householder_on_the_left_in_place_with_conj(
            essentials,
            householder,
            conj_lhs,
            matrix.rb_mut().submatrix(j, 0, m - j, k),
            parallelism,
            stack.rb_mut(),
        );

        bs = blocksize;
    }
}

/// Computes the product of the transpose of a sequence block Householder transformations given by
/// `householder_basis` and `householder_factor`, multiplied by `matrix`, and stores the result in
/// `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj<
    E: ComplexField,
>(
    householder_basis: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_lhs: Conj,
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    let mut matrix = matrix;
    let mut stack = stack;
    let blocksize = householder_factor.nrows();
    assert!(blocksize > 0);
    let m = householder_basis.nrows();
    let k = matrix.ncols();

    let size = householder_factor.ncols();

    let mut j = 0;
    while j < size {
        let bs = <usize as Ord>::min(blocksize, size - j);
        let essentials = householder_basis.submatrix(j, j, m - j, bs);
        let householder = householder_factor.submatrix(0, j, bs, bs);

        apply_block_householder_transpose_on_the_left_in_place_with_conj(
            essentials,
            householder,
            conj_lhs,
            matrix.rb_mut().submatrix(j, 0, m - j, k),
            parallelism,
            stack.rb_mut(),
        );

        j += bs;
    }
}

/// Computes the product of `matrix`, multiplied by a sequence of block Householder transformations
/// given by `householder_basis` and `householder_factor`, and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_on_the_right_in_place_with_conj<E: ComplexField>(
    householder_basis: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_rhs: Conj,
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose(),
        parallelism,
        stack,
    )
}

/// Computes the product of `matrix`, multiplied by the transpose of a sequence of block Householder
/// transformations given by `householder_basis` and `householder_factor`, and stores the result in
/// `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj<
    E: ComplexField,
>(
    householder_basis: MatRef<'_, E>,
    householder_factor: MatRef<'_, E>,
    conj_rhs: Conj,
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose(),
        parallelism,
        stack,
    )
}
