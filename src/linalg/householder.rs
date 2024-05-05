//! Block Householder transformations.
//!
//! A Householder reflection is linear transformation that describes a reflection about a
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
    assert,
    linalg::{
        matmul::{
            inner_prod, matmul, matmul_with_conj,
            triangular::{self, BlockStructure},
        },
        temp_mat_req, temp_mat_uninit, triangular_solve as solve,
    },
    unzipped,
    utils::{simd::*, slice::*, thread::join_raw, DivCeil},
    zipped, Conj, MatMut, MatRef, Parallelism,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_entity::*;
use num_complex::Complex;
use reborrow::*;

/// Computes the Householder reflection $I - \frac{v v^H}{\tau}$ such that when multiplied by $x$
/// from the left, The result is $\beta e_0$. $\tau$ and $\beta$ are returned and $\tau$ is
/// real-valued.
///
/// $x$ is determined by $x_0$, contained in `head`, and $|x_{1\dots}|$, contained in `tail_norm`.
/// The vector $v$ is such that $v_0 = 1$ and $v_{1\dots}$ is stored in `essential` (when provided).
#[inline]
pub fn make_householder_in_place<E: ComplexField>(
    essential: Option<MatMut<'_, E>>,
    head: E,
    tail_norm: E::Real,
) -> (E, E) {
    if tail_norm == E::Real::faer_zero() {
        return (E::faer_from_real(E::Real::faer_zero().faer_inv()), head);
    }

    let one_half = E::Real::faer_from_f64(0.5);

    let head_norm = head.faer_abs();
    let norm = Complex {
        re: head_norm,
        im: tail_norm,
    }
    .faer_abs();

    let sign = if head_norm != E::Real::faer_zero() {
        head.faer_scale_real(head_norm.faer_inv())
    } else {
        E::faer_one()
    };

    let signed_norm = sign.faer_mul(E::faer_from_real(norm));
    let head_with_beta = head.faer_add(signed_norm);
    let head_with_beta_inv = head_with_beta.faer_inv();

    if head_with_beta != E::faer_zero() {
        if let Some(essential) = essential {
            assert!(essential.ncols() == 1);
            zipped!(essential)
                .for_each(|unzipped!(mut e)| e.write(e.read().faer_mul(head_with_beta_inv)));
        }
        let tau = one_half.faer_mul(
            E::Real::faer_one().faer_add(
                tail_norm
                    .faer_mul(head_with_beta_inv.faer_abs())
                    .faer_abs2(),
            ),
        );
        (E::faer_from_real(tau), signed_norm.faer_neg())
    } else {
        (
            E::faer_from_real(E::Real::faer_zero().faer_inv()),
            E::faer_zero(),
        )
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

    let block_count = householder_factor.nrows().msrv_div_ceil(blocksize);

    if block_count > 1 {
        assert!(all(
            blocksize > prev_blocksize,
            blocksize % prev_blocksize == 0,
        ));
        let idx = (block_count / 2) * blocksize;
        let (tau_tl, _, _, tau_br) = householder_factor.split_at_mut(idx, idx);
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
            basis_top.adjoint(),
            BlockStructure::UnitTriangularUpper,
            basis_top,
            BlockStructure::UnitTriangularLower,
            None,
            E::faer_one(),
            parallelism,
        );
        triangular::matmul(
            householder_factor.rb_mut(),
            BlockStructure::UnitTriangularUpper,
            basis_bot.adjoint(),
            BlockStructure::Rectangular,
            basis_bot,
            BlockStructure::Rectangular,
            Some(E::faer_one()),
            E::faer_one(),
            parallelism,
        );
    } else {
        let prev_block_count = householder_factor.nrows().msrv_div_ceil(prev_blocksize);

        let idx = (prev_block_count / 2) * prev_blocksize;
        let (tau_tl, mut tau_tr, _, tau_br) = householder_factor.split_at_mut(idx, idx);
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
                    basis_left_top.adjoint(),
                    BlockStructure::Rectangular,
                    basis_right_top,
                    BlockStructure::UnitTriangularLower,
                    None,
                    E::faer_one(),
                    parallelism,
                );
                matmul(
                    tau_tr.rb_mut(),
                    basis_left_bot.adjoint(),
                    basis_right_bot,
                    Some(E::faer_one()),
                    E::faer_one(),
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
    stack: PodStack<'_>,
) {
    assert!(all(
        householder_factor.nrows() == householder_factor.ncols(),
        householder_basis.ncols() == householder_factor.nrows(),
        matrix.nrows() == householder_basis.nrows(),
    ));

    let bs = householder_factor.nrows();
    if householder_basis.row_stride() == 1 && matrix.row_stride() == 1 && bs == 1 {
        let arch = E::Simd::default();

        struct ApplyOnLeft<'a, C: ConjTy, E: ComplexField> {
            tau_inv: E,
            essential: MatRef<'a, E>,
            rhs: MatMut<'a, E>,
            conj: C,
        }

        impl<C: ConjTy, E: ComplexField> pulp::WithSimd for ApplyOnLeft<'_, C, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self {
                    tau_inv,
                    essential,
                    mut rhs,
                    conj,
                } = self;
                debug_assert_eq!(essential.row_stride(), 1);
                debug_assert_eq!(rhs.row_stride(), 1);

                let n = rhs.ncols();

                if rhs.nrows() == 0 {
                    return;
                }

                let simd = SimdFor::<E, S>::new(simd);

                let essential = SliceGroup::<'_, E>::new(essential.try_get_contiguous_col(0));
                let offset = simd.align_offset(essential.rb());
                let (essential_head, essential_body, essential_tail) =
                    simd.as_aligned_simd(essential, offset);
                for idx in 0..n {
                    let col =
                        SliceGroupMut::<'_, E>::new(rhs.rb_mut().try_get_contiguous_col_mut(idx));

                    let (col0, col) = col.split_at(1);
                    let mut col0 = col0.get_mut(0);

                    let dot = col0.read().faer_add(inner_prod::with_simd_and_offset(
                        simd,
                        conj.flip(),
                        essential,
                        col.rb(),
                        offset,
                    ));

                    let k = dot.faer_mul(tau_inv).faer_neg();
                    col0.write(col0.read().faer_add(k));

                    let (col_head, col_body, col_tail) = simd.as_aligned_simd_mut(col, offset);

                    #[inline(always)]
                    fn process<C: ConjTy, E: ComplexField, S: pulp::Simd>(
                        simd: SimdFor<E, S>,
                        conj: C,
                        mut a: impl Write<Output = SimdGroupFor<E, S>>,
                        b: impl Read<Output = SimdGroupFor<E, S>>,
                        k: SimdGroupFor<E, S>,
                    ) {
                        a.write(simd.conditional_conj_mul_add_e(
                            conj,
                            b.read_or(simd.splat(E::faer_zero())),
                            k,
                            a.read_or(simd.splat(E::faer_zero())),
                        ));
                    }

                    let k = simd.splat(k);
                    process(simd, conj, col_head, essential_head, k);
                    for (a, b) in col_body.into_mut_iter().zip(essential_body.into_ref_iter()) {
                        process(simd, conj, a, b, k);
                    }
                    process(simd, conj, col_tail, essential_tail, k);
                }
            }
        }

        if coe::is_same::<E, E::Real>() || matches!(conj_lhs, Conj::No) {
            arch.dispatch(ApplyOnLeft {
                tau_inv: E::faer_from_real(householder_factor.read(0, 0).faer_real().faer_inv()),
                essential: householder_basis.split_at_row(1).1,
                rhs: matrix,
                conj: NoConj,
            });
        } else {
            arch.dispatch(ApplyOnLeft {
                tau_inv: E::faer_from_real(householder_factor.read(0, 0).faer_real().faer_inv()),
                essential: householder_basis.split_at_row(1).1,
                rhs: matrix,
                conj: YesConj,
            });
        }
    } else {
        let (essentials_top, essentials_bot) = householder_basis.split_at_row(bs);
        let m = matrix.nrows();
        let n = matrix.ncols();

        // essentials* × mat
        let (tmp, _) = temp_mat_uninit::<E>(bs, n, stack);

        let mut n_tasks = Ord::min(
            Ord::min(crate::utils::thread::parallelism_degree(parallelism), n),
            4,
        );
        if (m * n).saturating_mul(4 * bs) < gemm::get_threading_threshold() {
            n_tasks = 1;
        }

        let inner_parallelism = match parallelism {
            Parallelism::None => Parallelism::None,
            #[cfg(feature = "rayon")]
            Parallelism::Rayon(mut par) => {
                if par == 0 {
                    par = rayon::current_num_threads();
                }

                if par >= 2 * n_tasks {
                    Parallelism::Rayon(par / n_tasks)
                } else {
                    Parallelism::None
                }
            }
        };

        crate::utils::thread::for_each_raw(
            n_tasks,
            |tid| {
                let (tid_col, tid_n) = crate::utils::thread::par_split_indices(n, tid, n_tasks);

                let mut tmp = unsafe { tmp.rb().subcols(tid_col, tid_n).const_cast() };
                let (mut matrix_top, mut matrix_bot) = unsafe {
                    matrix
                        .rb()
                        .subcols(tid_col, tid_n)
                        .const_cast()
                        .split_at_row_mut(bs)
                };

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
                    E::faer_one(),
                    inner_parallelism,
                );
                matmul_with_conj(
                    tmp.rb_mut(),
                    essentials_bot.transpose(),
                    Conj::Yes.compose(conj_lhs),
                    matrix_bot.rb(),
                    Conj::No,
                    Some(E::faer_one()),
                    E::faer_one(),
                    inner_parallelism,
                );

                // [T^-1|T^-*] × essentials* × tmp
                if forward {
                    solve::solve_lower_triangular_in_place_with_conj(
                        householder_factor.transpose(),
                        Conj::Yes.compose(conj_lhs),
                        tmp.rb_mut(),
                        inner_parallelism,
                    );
                } else {
                    solve::solve_upper_triangular_in_place_with_conj(
                        householder_factor,
                        Conj::No.compose(conj_lhs),
                        tmp.rb_mut(),
                        inner_parallelism,
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
                    Some(E::faer_one()),
                    E::faer_one().faer_neg(),
                    inner_parallelism,
                );
                matmul_with_conj(
                    matrix_bot.rb_mut(),
                    essentials_bot,
                    Conj::No.compose(conj_lhs),
                    tmp.rb(),
                    Conj::No,
                    Some(E::faer_one()),
                    E::faer_one().faer_neg(),
                    inner_parallelism,
                );
            },
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
    stack: PodStack<'_>,
) {
    apply_block_householder_transpose_on_the_left_in_place_with_conj(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose_mut(),
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
    stack: PodStack<'_>,
) {
    apply_block_householder_on_the_left_in_place_with_conj(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose_mut(),
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
    stack: PodStack<'_>,
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
    stack: PodStack<'_>,
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
    stack: PodStack<'_>,
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
            matrix.rb_mut().submatrix_mut(j, 0, m - j, k),
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
    stack: PodStack<'_>,
) {
    let mut matrix = matrix;
    let mut stack = stack;
    let blocksize = householder_factor.nrows();
    assert!(all(
        blocksize > 0,
        matrix.nrows() == householder_basis.nrows()
    ));
    let m = householder_basis.nrows();
    let k = matrix.ncols();

    let size = householder_factor.ncols();

    let mut j = 0;
    while j < size {
        let bs = Ord::min(blocksize, size - j);
        let essentials = householder_basis.submatrix(j, j, m - j, bs);
        let householder = householder_factor.submatrix(0, j, bs, bs);

        apply_block_householder_transpose_on_the_left_in_place_with_conj(
            essentials,
            householder,
            conj_lhs,
            matrix.rb_mut().submatrix_mut(j, 0, m - j, k),
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
    stack: PodStack<'_>,
) {
    apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose_mut(),
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
    stack: PodStack<'_>,
) {
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose_mut(),
        parallelism,
        stack,
    )
}
