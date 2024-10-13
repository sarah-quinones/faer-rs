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
            dot, matmul, matmul_with_conj,
            triangular::{self, BlockStructure},
        },
        triangular_solve as solve,
    },
    utils::{simd::SimdCtx, thread::join_raw},
    ContiguousFwd, Stride,
};

use crate::internal_prelude::*;

/// Computes the Householder reflection $I - \frac{v v^H}{\tau}$ such that when multiplied by $x$
/// from the left, The result is $\beta e_0$. $\tau$ and $\beta$ are returned and $\tau$ is
/// real-valued.
///
/// $x$ is determined by $x_0$, contained in `head`, and $|x_{1\dots}|$, contained in `tail_norm`.
/// The vector $v$ is such that $v_0 = 1$ and $v_{1\dots}$ is stored in `essential` (when provided).
#[inline]
#[math]
pub fn make_householder_in_place<'M, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    essential: Option<ColMut<'_, C, T, Dim<'M>>>,
    head: C::Of<&T>,
    tail_norm: <C::Real as Container>::Of<&T::RealUnit>,
) -> (C::Of<T>, C::Of<T>) {
    if math.re.is_zero(tail_norm) {
        return math((infinity(), copy(head)));
    }

    let one_half = math.re.from_f64(0.5);

    let head_norm = math.abs(head);
    let norm = math(hypot(head_norm, re.copy(tail_norm)));

    let sign = if !math.re.is_zero(head_norm) {
        math(mul_real(head, re.recip(head_norm)))
    } else {
        math.one()
    };

    let signed_norm = math(sign * from_real(norm));
    let head_with_beta = math(head + signed_norm);
    let head_with_beta_inv = math.recip(head_with_beta);

    help!(C);
    if !math.is_zero(head_with_beta) {
        if let Some(essential) = essential {
            zipped!(essential).for_each(|unzipped!(mut e)| {
                write1!(e, math(e * head_with_beta_inv));
            });
        }
        let tau = math.re(one_half * (one() + abs2(tail_norm * cx.abs(head_with_beta_inv))));
        math((from_real(tau), -signed_norm))
    } else {
        math((infinity(), zero()))
    }
}

#[doc(hidden)]
#[math]
pub fn upgrade_householder_factor<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    mut householder_factor: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    essentials: MatRef<'_, C, T, Dim<'M>, Dim<'N>>,
    blocksize: usize,
    prev_blocksize: usize,
    par: Parallelism,
) {
    if blocksize == prev_blocksize || householder_factor.nrows().unbound() <= prev_blocksize {
        return;
    }
    let M: Dim<'M> = essentials.nrows();
    let N: Dim<'N> = essentials.ncols();

    assert!(householder_factor.nrows() == householder_factor.ncols());

    let block_count = householder_factor.nrows().div_ceil(blocksize);
    if block_count > 1 {
        assert!(all(
            blocksize > prev_blocksize,
            blocksize % prev_blocksize == 0,
        ));
        make_guard!(HEAD);
        make_guard!(N_TAIL);
        make_guard!(M_TAIL);

        let N_idx = N.partition(
            IdxInc::<'N>::new_checked((block_count / 2) * blocksize, N),
            HEAD,
            N_TAIL,
        );
        let M_idx = M.head_partition(N_idx.head, M_TAIL);

        let (tau_tl, _, _, tau_br) = householder_factor.split_with_mut(N_idx, N_idx);
        let (basis_left, basis_right) = essentials.split_cols_with(N_idx);
        let basis_right = basis_right.split_rows_with(M_idx).1;
        join_raw(
            |parallelism| {
                upgrade_householder_factor(
                    ctx,
                    tau_tl,
                    basis_left,
                    blocksize,
                    prev_blocksize,
                    parallelism,
                )
            },
            |parallelism| {
                upgrade_householder_factor(
                    ctx,
                    tau_br,
                    basis_right,
                    blocksize,
                    prev_blocksize,
                    parallelism,
                )
            },
            par,
        );
        return;
    }

    make_guard!(TAIL);
    let midpoint = M.head_partition(N, TAIL);

    if prev_blocksize < 8 {
        // pretend that prev_blocksize == 1, recompute whole top half of matrix

        let (basis_top, basis_bot) = essentials.split_rows_with(midpoint);
        let acc_structure = BlockStructure::UnitTriangularUpper;

        help!(C);
        triangular::matmul(
            ctx,
            householder_factor.rb_mut(),
            acc_structure,
            None,
            basis_top.adjoint(),
            BlockStructure::UnitTriangularUpper,
            basis_top,
            BlockStructure::UnitTriangularLower,
            as_ref!(math.one()),
            par,
        );
        triangular::matmul(
            ctx,
            householder_factor.rb_mut(),
            acc_structure,
            Some(as_ref!(math.one())),
            basis_bot.adjoint(),
            BlockStructure::Rectangular,
            basis_bot,
            BlockStructure::Rectangular,
            as_ref!(math.one()),
            par,
        );
    } else {
        let prev_block_count = householder_factor.nrows().div_ceil(prev_blocksize);

        make_guard!(HEAD);
        make_guard!(N_TAIL);

        let N_idx = N.partition(
            IdxInc::new_checked((prev_block_count / 2) * prev_blocksize, N),
            HEAD,
            N_TAIL,
        );
        make_guard!(M_TAIL);
        let M_idx = M.head_partition(N_idx.head, M_TAIL);

        let (tau_tl, mut tau_tr, _, tau_br) = householder_factor.split_with_mut(N_idx, N_idx);
        let (basis_left, basis_right) = essentials.split_cols_with(N_idx);
        let basis_right = basis_right.split_rows_with(M_idx).1;

        join_raw(
            |parallelism| {
                join_raw(
                    |parallelism| {
                        upgrade_householder_factor(
                            ctx,
                            tau_tl,
                            basis_left,
                            blocksize,
                            prev_blocksize,
                            parallelism,
                        )
                    },
                    |parallelism| {
                        upgrade_householder_factor(
                            ctx,
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
                let basis_left = basis_left.split_rows_with(M_idx).1;
                make_guard!(M_TAIL);
                let M_idx = M_idx.tail.head_partition(basis_right.ncols(), M_TAIL);

                let (basis_left_top, basis_left_bot) = basis_left.split_rows_with(M_idx);
                let (basis_right_top, basis_right_bot) = basis_right.split_rows_with(M_idx);

                help!(C);
                triangular::matmul(
                    ctx,
                    tau_tr.rb_mut(),
                    BlockStructure::Rectangular,
                    None,
                    basis_left_top.adjoint(),
                    BlockStructure::Rectangular,
                    basis_right_top,
                    BlockStructure::UnitTriangularLower,
                    as_ref!(math.one()),
                    parallelism,
                );
                matmul(
                    ctx,
                    tau_tr.rb_mut(),
                    Some(as_ref!(math.one())),
                    basis_left_bot.adjoint(),
                    basis_right_bot,
                    as_ref!(math.one()),
                    parallelism,
                );
            },
            par,
        );
    }
}

/// Computes the size and alignment of required workspace for applying a block Householder
/// transformation to a right-hand-side matrix in place.
pub fn apply_block_householder_on_the_left_in_place_scratch<
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_scratch::<C, T>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying the transpose of a block
/// Householder transformation to a right-hand-side matrix in place.
pub fn apply_block_householder_transpose_on_the_left_in_place_scratch<
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_scratch::<C, T>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying a block Householder
/// transformation to a left-hand-side matrix in place.
pub fn apply_block_householder_on_the_right_in_place_scratch<
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_scratch::<C, T>(blocksize, lhs_nrows)
}

/// Computes the size and alignment of required workspace for applying the transpose of a block
/// Householder transformation to a left-hand-side matrix in place.
pub fn apply_block_householder_transpose_on_the_right_in_place_scratch<
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_scratch::<C, T>(blocksize, lhs_nrows)
}

/// Computes the size and alignment of required workspace for applying the transpose of a sequence
/// of block Householder transformations to a right-hand-side matrix in place.
pub fn apply_block_householder_sequence_transpose_on_the_left_in_place_scratch<
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_scratch::<C, T>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying a sequence of block
/// Householder transformations to a right-hand-side matrix in place.
pub fn apply_block_householder_sequence_on_the_left_in_place_scratch<
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    householder_basis_nrows: usize,
    blocksize: usize,
    rhs_ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_scratch::<C, T>(blocksize, rhs_ncols)
}

/// Computes the size and alignment of required workspace for applying the transpose of a sequence
/// of block Householder transformations to a left-hand-side matrix in place.
pub fn apply_block_householder_sequence_transpose_on_the_right_in_place_scratch<
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_scratch::<C, T>(blocksize, lhs_nrows)
}

/// Computes the size and alignment of required workspace for applying a sequence of block
/// Householder transformations to a left-hand-side matrix in place.
pub fn apply_block_householder_sequence_on_the_right_in_place_scratch<
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    householder_basis_nrows: usize,
    blocksize: usize,
    lhs_nrows: usize,
) -> Result<StackReq, SizeOverflow> {
    let _ = householder_basis_nrows;
    temp_mat_scratch::<C, T>(blocksize, lhs_nrows)
}

#[track_caller]
#[math]
fn apply_block_householder_on_the_left_in_place_generic<
    'M,
    'N,
    'K,
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    ctx: &Ctx<C, T>,
    householder_basis: MatRef<'_, C, T, Dim<'M>, Dim<'N>>,
    householder_factor: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
    conj_lhs: Conj,
    matrix: MatMut<'_, C, T, Dim<'M>, Dim<'K>>,
    forward: bool,
    par: Parallelism,
    stack: &mut DynStack,
) {
    assert!(all(
        householder_factor.nrows() == householder_factor.ncols(),
        householder_basis.ncols() == householder_factor.nrows(),
        matrix.nrows() == householder_basis.nrows(),
    ));

    let mut matrix = matrix;

    let M = householder_basis.nrows();
    let N = householder_basis.ncols();

    make_guard!(TAIL);
    let midpoint = M.head_partition(N, TAIL);

    if let (Some(householder_basis), Some(matrix), 1, true) = (
        householder_basis.try_as_col_major(),
        matrix.rb_mut().try_as_col_major_mut(),
        N.unbound(),
        T::SIMD_CAPABILITIES.is_simd(),
    ) {
        let arch = T::Arch::default();

        struct ApplyOnLeft<'a, 'TAIL, 'K, C: ComplexContainer, T: ComplexField<C>, const CONJ: bool> {
            ctx: &'a Ctx<C, T>,
            tau_inv: C::Of<&'a T>,
            essential: ColRef<'a, C, T, Dim<'TAIL>, ContiguousFwd>,
            rhs0: RowMut<'a, C, T, Dim<'K>>,
            rhs: MatMut<'a, C, T, Dim<'TAIL>, Dim<'K>, ContiguousFwd>,
        }

        impl<'TAIL, 'K, C: ComplexContainer, T: ComplexField<C>, const CONJ: bool> pulp::WithSimd
            for ApplyOnLeft<'_, 'TAIL, 'K, C, T, CONJ>
        {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self {
                    ctx,
                    tau_inv,
                    essential,
                    mut rhs,
                    mut rhs0,
                } = self;

                if rhs.nrows().unbound() == 0 {
                    return;
                }

                let N = rhs.nrows();
                let K = rhs.ncols();
                let simd = SimdCtx::<C, T, S>::new(T::simd_ctx(ctx, simd), N);
                let (head, indices, tail) = simd.indices();

                help!(C);
                for idx in K.indices() {
                    let mut col0 = rhs0.rb_mut().at_mut(idx);
                    let mut col = rhs.rb_mut().col_mut(idx);
                    let essential = essential;

                    let dot = if const { CONJ } {
                        math(col0 + dot::inner_prod_no_conj_simd(simd, essential.rb(), col.rb()))
                    } else {
                        math(col0 + dot::inner_prod_conj_lhs_simd(simd, essential.rb(), col.rb()))
                    };

                    let k = math(-dot * tau_inv);
                    write1!(col0, math(col0 + k));

                    let k = simd.splat(as_ref!(k));
                    macro_rules! simd {
                        ($i: expr) => {{
                            let i = $i;
                            let mut a = simd.read(col.rb(), i);
                            let b = simd.read(essential.rb(), i);

                            if const { CONJ } {
                                a = simd.conj_mul_add(b, k, a);
                            } else {
                                a = simd.mul_add(b, k, a);
                            }

                            simd.write(col.rb_mut(), i, a);
                        }};
                    }

                    if let Some(i) = head {
                        simd!(i);
                    }
                    for i in indices.clone() {
                        simd!(i);
                    }
                    if let Some(i) = tail {
                        simd!(i);
                    }
                }
            }
        }

        let N0 = N.check(0);

        let essential = householder_basis.col(N0).split_rows_with(midpoint).1;
        let (rhs0, rhs) = matrix.split_rows_with_mut(midpoint);
        let rhs0 = rhs0.row_mut(N0);

        let tau_inv = math(from_real(re.recip(real(householder_factor[(N0, N0)]))));

        if const { T::IS_REAL } || matches!(conj_lhs, Conj::No) {
            arch.dispatch(ApplyOnLeft::<_, _, false> {
                ctx,
                tau_inv: math.id(tau_inv),
                essential,
                rhs,
                rhs0,
            });
        } else {
            arch.dispatch(ApplyOnLeft::<_, _, true> {
                ctx,
                tau_inv: math.id(tau_inv),
                essential,
                rhs,
                rhs0,
            });
        }
    } else {
        let (essentials_top, essentials_bot) = householder_basis.split_rows_with(midpoint);
        let M = matrix.nrows();
        let K = matrix.ncols();

        // essentials* × mat
        let (mut tmp, _) = unsafe { temp_mat_uninit::<C, T, _, _>(ctx, N, K, stack) };
        let mut tmp = tmp.as_mat_mut();

        let mut n_tasks = Ord::min(
            Ord::min(crate::utils::thread::parallelism_degree(par), K.unbound()),
            4,
        );
        if (M.unbound() * K.unbound()).saturating_mul(4 * M.unbound())
            < gemm::get_threading_threshold()
        {
            n_tasks = 1;
        }

        let inner_parallelism = match par {
            Parallelism::None => Parallelism::None,
            #[cfg(feature = "rayon")]
            Parallelism::Rayon(par) => {
                let par = par.get();

                if par >= 2 * n_tasks {
                    Parallelism::rayon(par / n_tasks)
                } else {
                    Parallelism::None
                }
            }
        };

        let func =
            |(mut tmp, mut matrix): (MatMut<'_, C, T, Dim<'N>>, MatMut<'_, C, T, Dim<'M>>)| {
                let (mut top, mut bot) = matrix.rb_mut().split_rows_with_mut(midpoint);
                help!(C);

                triangular::matmul_with_conj(
                    ctx,
                    tmp.rb_mut(),
                    BlockStructure::Rectangular,
                    None,
                    essentials_top.transpose(),
                    BlockStructure::UnitTriangularUpper,
                    Conj::Yes.compose(conj_lhs),
                    top.rb(),
                    BlockStructure::Rectangular,
                    Conj::No,
                    as_ref!(math.one()),
                    inner_parallelism,
                );

                matmul_with_conj(
                    ctx,
                    tmp.rb_mut(),
                    Some(as_ref!(math.one())),
                    essentials_bot.transpose(),
                    Conj::Yes.compose(conj_lhs),
                    bot.rb(),
                    Conj::No,
                    as_ref!(math.one()),
                    inner_parallelism,
                );

                // [T^-1|T^-*] × essentials* × tmp
                if forward {
                    solve::solve_lower_triangular_in_place_with_conj(
                        ctx,
                        householder_factor.transpose(),
                        Conj::Yes.compose(conj_lhs),
                        tmp.rb_mut(),
                        inner_parallelism,
                    );
                } else {
                    solve::solve_upper_triangular_in_place_with_conj(
                        ctx,
                        householder_factor,
                        Conj::No.compose(conj_lhs),
                        tmp.rb_mut(),
                        inner_parallelism,
                    );
                }

                // essentials × [T^-1|T^-*] × essentials* × tmp
                triangular::matmul_with_conj(
                    ctx,
                    top.rb_mut(),
                    BlockStructure::Rectangular,
                    Some(as_ref!(math.one())),
                    essentials_top,
                    BlockStructure::UnitTriangularLower,
                    Conj::No.compose(conj_lhs),
                    tmp.rb(),
                    BlockStructure::Rectangular,
                    Conj::No,
                    math(id(-one())),
                    inner_parallelism,
                );
                matmul_with_conj(
                    ctx,
                    bot.rb_mut(),
                    Some(as_ref!(math.one())),
                    essentials_bot,
                    Conj::No.compose(conj_lhs),
                    tmp.rb(),
                    Conj::No,
                    math(id(-one())),
                    inner_parallelism,
                );
            };

        if n_tasks <= 1 {
            func((tmp.as_dyn_cols_mut(), matrix.as_dyn_cols_mut()));
            return;
        } else {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                tmp.rb_mut()
                    .par_col_partition_mut(n_tasks)
                    .zip_eq(matrix.rb_mut().par_col_partition_mut(n_tasks))
                    .for_each(func);
            }
        }
    }
}

/// Computes the product of the matrix, multiplied by the given block Householder transformation,
/// and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_on_the_right_in_place_with_conj<
    C: ComplexContainer,
    T: ComplexField<C>,
    M: Shape,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    householder_basis: MatRef<'_, C, T, M, N, impl Stride, impl Stride>,
    householder_factor: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    conj_rhs: Conj,
    matrix: MatMut<'_, C, T, K, M, impl Stride, impl Stride>,
    par: Parallelism,
    stack: &mut DynStack,
) {
    apply_block_householder_transpose_on_the_left_in_place_with_conj(
        ctx,
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose_mut(),
        par,
        stack,
    )
}

/// Computes the product of the matrix, multiplied by the transpose of the given block Householder
/// transformation, and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_transpose_on_the_right_in_place_with_conj<
    M: Shape,
    N: Shape,
    K: Shape,
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    ctx: &Ctx<C, T>,
    householder_basis: MatRef<'_, C, T, M, N, impl Stride, impl Stride>,
    householder_factor: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    conj_rhs: Conj,
    matrix: MatMut<'_, C, T, K, M, impl Stride, impl Stride>,
    par: Parallelism,
    stack: &mut DynStack,
) {
    apply_block_householder_on_the_left_in_place_with_conj(
        ctx,
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose_mut(),
        par,
        stack,
    )
}

/// Computes the product of the given block Householder transformation, multiplied by `matrix`, and
/// stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_on_the_left_in_place_with_conj<
    C: ComplexContainer,
    T: ComplexField<C>,
    M: Shape,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    householder_basis: MatRef<'_, C, T, M, N, impl Stride, impl Stride>,
    householder_factor: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    conj_lhs: Conj,
    matrix: MatMut<'_, C, T, M, K, impl Stride, impl Stride>,
    par: Parallelism,
    stack: &mut DynStack,
) {
    make_guard!(M);
    make_guard!(N);
    make_guard!(K);
    let M = householder_basis.nrows().bind(M);
    let N = householder_basis.ncols().bind(N);
    let K = matrix.ncols().bind(K);

    apply_block_householder_on_the_left_in_place_generic(
        ctx,
        householder_basis.as_shape(M, N).as_dyn_stride(),
        householder_factor.as_shape(N, N).as_dyn_stride(),
        conj_lhs,
        matrix.as_shape_mut(M, K).as_dyn_stride_mut(),
        false,
        par,
        stack,
    )
}

/// Computes the product of the transpose of the given block Householder transformation, multiplied
/// by `matrix`, and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_transpose_on_the_left_in_place_with_conj<
    C: ComplexContainer,
    T: ComplexField<C>,
    M: Shape,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    householder_basis: MatRef<'_, C, T, M, N, impl Stride, impl Stride>,
    householder_factor: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    conj_lhs: Conj,
    matrix: MatMut<'_, C, T, M, K, impl Stride, impl Stride>,
    par: Parallelism,
    stack: &mut DynStack,
) {
    make_guard!(M);
    make_guard!(N);
    make_guard!(K);
    let M = householder_basis.nrows().bind(M);
    let N = householder_basis.ncols().bind(N);
    let K = matrix.ncols().bind(K);

    apply_block_householder_on_the_left_in_place_generic(
        ctx,
        householder_basis.as_shape(M, N).as_dyn_stride(),
        householder_factor.as_shape(N, N).as_dyn_stride(),
        conj_lhs.compose(Conj::Yes),
        matrix.as_shape_mut(M, K).as_dyn_stride_mut(),
        true,
        par,
        stack,
    )
}

/// Computes the product of a sequence of block Householder transformations given by
/// `householder_basis` and `householder_factor`, multiplied by `matrix`, and stores the result in
/// `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_on_the_left_in_place_with_conj<
    C: ComplexContainer,
    T: ComplexField<C>,
    M: Shape,
    N: Shape,
    K: Shape,
    H: Shape,
>(
    ctx: &Ctx<C, T>,
    householder_basis: MatRef<'_, C, T, M, N, impl Stride, impl Stride>,
    householder_factor: MatRef<'_, C, T, H, N, impl Stride, impl Stride>,
    conj_lhs: Conj,
    matrix: MatMut<'_, C, T, M, K, impl Stride, impl Stride>,
    par: Parallelism,
    stack: &mut DynStack,
) {
    #[track_caller]
    pub fn imp<'M, 'N, 'K, 'H, C: ComplexContainer, T: ComplexField<C>>(
        ctx: &Ctx<C, T>,
        householder_basis: MatRef<'_, C, T, Dim<'M>, Dim<'N>>,
        householder_factor: MatRef<'_, C, T, Dim<'H>, Dim<'N>>,
        conj_lhs: Conj,
        matrix: MatMut<'_, C, T, Dim<'M>, Dim<'K>>,
        par: Parallelism,
        stack: &mut DynStack,
    ) {
        let mut matrix = matrix;
        let mut stack = stack;

        assert!(*householder_factor.nrows() > 0);
        let M = householder_basis.nrows();

        let size = householder_factor.ncols();

        let mut j = size.end();

        let mut blocksize = *size % *householder_factor.nrows();
        if blocksize == 0 {
            blocksize = *householder_factor.nrows();
        }

        while *j > 0 {
            let j_prev = size.idx(*j - blocksize);
            blocksize = *householder_factor.nrows();

            {
                let jm = M.checked_idx_inc(*j_prev);

                let essentials = householder_basis.submatrix_range((jm, M.end()), (j_prev, j));

                let householder = householder_factor
                    .subcols_range((j_prev, j))
                    .subrows(zero(), *j - *j_prev);

                let matrix = matrix.rb_mut().subrows_range_mut((jm, M.end()));
                make_guard!(M);
                make_guard!(N);
                let M = essentials.nrows().bind(M);
                let N = essentials.ncols().bind(N);

                apply_block_householder_on_the_left_in_place_with_conj(
                    ctx,
                    essentials.as_shape(M, N),
                    householder.as_shape(N, N),
                    conj_lhs,
                    matrix.as_row_shape_mut(M),
                    par,
                    stack.rb_mut(),
                );
            }

            j = j_prev.to_incl();
        }
    }
    make_guard!(M);
    make_guard!(N);
    make_guard!(K);
    make_guard!(H);
    let M = householder_basis.nrows().bind(M);
    let N = householder_basis.ncols().bind(N);
    let H = householder_factor.nrows().bind(H);
    let K = matrix.ncols().bind(K);
    imp(
        ctx,
        householder_basis.as_dyn_stride().as_shape(M, N),
        householder_factor.as_dyn_stride().as_shape(H, N),
        conj_lhs,
        matrix.as_dyn_stride_mut().as_shape_mut(M, K),
        par,
        stack,
    )
}

/// Computes the product of the transpose of a sequence block Householder transformations given by
/// `householder_basis` and `householder_factor`, multiplied by `matrix`, and stores the result in
/// `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj<
    C: ComplexContainer,
    T: ComplexField<C>,
    M: Shape,
    N: Shape,
    K: Shape,
    H: Shape,
>(
    ctx: &Ctx<C, T>,
    householder_basis: MatRef<'_, C, T, M, N, impl Stride, impl Stride>,
    householder_factor: MatRef<'_, C, T, H, N, impl Stride, impl Stride>,
    conj_lhs: Conj,
    matrix: MatMut<'_, C, T, M, K, impl Stride, impl Stride>,
    par: Parallelism,
    stack: &mut DynStack,
) {
    #[track_caller]
    pub fn imp<'M, 'N, 'K, 'H, C: ComplexContainer, T: ComplexField<C>>(
        ctx: &Ctx<C, T>,
        householder_basis: MatRef<'_, C, T, Dim<'M>, Dim<'N>>,
        householder_factor: MatRef<'_, C, T, Dim<'H>, Dim<'N>>,
        conj_lhs: Conj,
        matrix: MatMut<'_, C, T, Dim<'M>, Dim<'K>>,
        par: Parallelism,
        stack: &mut DynStack,
    ) {
        let mut matrix = matrix;
        let mut stack = stack;

        let blocksize = householder_factor.nrows();

        assert!(blocksize.unbound() > 0);
        let M = householder_basis.nrows();

        let size = householder_factor.ncols();

        let mut J = Dim::start();

        while let Some(j) = size.try_check(*J) {
            let j_next = size.advance(j, *blocksize);

            {
                let jn = j.to_incl();
                let jm = M.checked_idx_inc(*jn);

                let essentials = householder_basis.submatrix_range((jm, M.end()), (jn, j_next));
                let householder = householder_factor
                    .subcols_range((jn, j_next))
                    .subrows(zero(), *j_next - *jn);

                let matrix = matrix.rb_mut().subrows_range_mut((jm, M.end()));
                make_guard!(M);
                make_guard!(N);
                let M = essentials.nrows().bind(M);
                let N = essentials.ncols().bind(N);

                apply_block_householder_transpose_on_the_left_in_place_with_conj(
                    ctx,
                    essentials.as_shape(M, N),
                    householder.as_shape(N, N),
                    conj_lhs,
                    matrix.as_row_shape_mut(M),
                    par,
                    stack.rb_mut(),
                );
            }

            J = j_next;
        }
    }
    make_guard!(M);
    make_guard!(N);
    make_guard!(K);
    make_guard!(H);
    let M = householder_basis.nrows().bind(M);
    let N = householder_basis.ncols().bind(N);
    let H = householder_factor.nrows().bind(H);
    let K = matrix.ncols().bind(K);
    imp(
        ctx,
        householder_basis.as_dyn_stride().as_shape(M, N),
        householder_factor.as_dyn_stride().as_shape(H, N),
        conj_lhs,
        matrix.as_dyn_stride_mut().as_shape_mut(M, K),
        par,
        stack,
    )
}

/// Computes the product of `matrix`, multiplied by a sequence of block Householder transformations
/// given by `householder_basis` and `householder_factor`, and stores the result in `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_on_the_right_in_place_with_conj<
    C: ComplexContainer,
    T: ComplexField<C>,
    M: Shape,
    N: Shape,
    K: Shape,
    H: Shape,
>(
    ctx: &Ctx<C, T>,
    householder_basis: MatRef<'_, C, T, M, N, impl Stride, impl Stride>,
    householder_factor: MatRef<'_, C, T, H, N, impl Stride, impl Stride>,
    conj_rhs: Conj,
    matrix: MatMut<'_, C, T, K, M, impl Stride, impl Stride>,
    par: Parallelism,
    stack: &mut DynStack,
) {
    apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        ctx,
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose_mut(),
        par,
        stack,
    )
}

/// Computes the product of `matrix`, multiplied by the transpose of a sequence of block Householder
/// transformations given by `householder_basis` and `householder_factor`, and stores the result in
/// `matrix`.
#[track_caller]
pub fn apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj<
    C: ComplexContainer,
    T: ComplexField<C>,
    M: Shape,
    N: Shape,
    K: Shape,
    H: Shape,
>(
    ctx: &Ctx<C, T>,
    householder_basis: MatRef<'_, C, T, M, N, impl Stride, impl Stride>,
    householder_factor: MatRef<'_, C, T, H, N, impl Stride, impl Stride>,
    conj_rhs: Conj,
    matrix: MatMut<'_, C, T, K, M, impl Stride, impl Stride>,
    par: Parallelism,
    stack: &mut DynStack,
) {
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        ctx,
        householder_basis,
        householder_factor,
        conj_rhs,
        matrix.transpose_mut(),
        par,
        stack,
    )
}
