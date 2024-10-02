use super::*;
use crate::{assert, debug_assert, linalg::zip::Diag, utils::thread::join_raw};

#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub(crate) enum DiagonalKind {
    Zero,
    Unit,
    Generic,
}

unsafe fn copy_lower<E: ComplexField>(
    mut dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    src_diag: DiagonalKind,
) {
    let n = dst.nrows();
    debug_assert!(n == dst.nrows());
    debug_assert!(n == dst.ncols());
    debug_assert!(n == src.nrows());
    debug_assert!(n == src.ncols());

    for j in 0..n {
        for i in 0..j {
            dst.write_unchecked(i, j, E::faer_zero());
        }
        match src_diag {
            DiagonalKind::Zero => dst.write_unchecked(j, j, E::faer_zero()),
            DiagonalKind::Unit => dst.write_unchecked(j, j, E::faer_one()),
            DiagonalKind::Generic => dst.write_unchecked(j, j, src.read(j, j)),
        };
        for i in j + 1..n {
            dst.write_unchecked(i, j, src.read_unchecked(i, j));
        }
    }
}

unsafe fn accum_lower<E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    skip_diag: bool,
    alpha: Option<E>,
) {
    let n = dst.nrows();
    debug_assert!(n == dst.nrows());
    debug_assert!(n == dst.ncols());
    debug_assert!(n == src.nrows());
    debug_assert!(n == src.ncols());

    match alpha {
        Some(alpha) => {
            zipped!(__rw, dst, src).for_each_triangular_lower(
                if skip_diag { Diag::Skip } else { Diag::Include },
                |unzipped!(mut dst, src)| {
                    dst.write(alpha.faer_mul(dst.read()).faer_add(src.read()))
                },
            );
        }
        None => {
            zipped!(__rw, dst, src).for_each_triangular_lower(
                if skip_diag { Diag::Skip } else { Diag::Include },
                |unzipped!(mut dst, src)| dst.write(src.read()),
            );
        }
    }
}

#[inline]
unsafe fn copy_upper<E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    src_diag: DiagonalKind,
) {
    copy_lower(dst.transpose_mut(), src.transpose(), src_diag)
}

#[inline]
unsafe fn mul<E: ComplexField>(
    dst: MatMut<'_, E>,
    lhs: MatRef<'_, E>,
    rhs: MatRef<'_, E>,
    alpha: Option<E>,
    beta: E,
    conj_lhs: Conj,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    super::matmul_with_conj(dst, lhs, conj_lhs, rhs, conj_rhs, alpha, beta, parallelism);
}

unsafe fn mat_x_lower_into_lower_impl_unchecked<E: ComplexField>(
    dst: MatMut<'_, E>,
    skip_diag: bool,
    lhs: MatRef<'_, E>,
    rhs: MatRef<'_, E>,
    rhs_diag: DiagonalKind,
    alpha: Option<E>,
    beta: E,
    conj_lhs: Conj,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    let n = dst.nrows();
    debug_assert!(n == dst.nrows());
    debug_assert!(n == dst.ncols());
    debug_assert!(n == lhs.nrows());
    debug_assert!(n == lhs.ncols());
    debug_assert!(n == rhs.nrows());
    debug_assert!(n == rhs.ncols());

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat!(
                    [16, 16],
                    temp_dst,
                    n,
                    n,
                    dst.row_stride(),
                    dst.col_stride(),
                    E
                );
                stack_mat!(
                    [16, 16],
                    temp_rhs,
                    n,
                    n,
                    rhs.row_stride(),
                    rhs.col_stride(),
                    E
                );

                copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);
                mul(
                    temp_dst.rb_mut(),
                    lhs,
                    temp_rhs.rb(),
                    None,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                );
                accum_lower(dst, temp_dst.rb(), skip_diag, alpha);
            }
        };
        op();
    } else {
        let bs = n / 2;

        let (mut dst_top_left, _, mut dst_bot_left, dst_bot_right) = dst.split_at_mut(bs, bs);
        let (lhs_top_left, lhs_top_right, lhs_bot_left, lhs_bot_right) = lhs.split_at(bs, bs);
        let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at(bs, bs);

        // lhs_bot_right × rhs_bot_left  => dst_bot_left  | mat × mat => mat |   1
        // lhs_bot_right × rhs_bot_right => dst_bot_right | mat × low => low |   X
        //
        // lhs_top_left  × rhs_top_left  => dst_top_left  | mat × low => low |   X
        // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
        // lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2

        mul(
            dst_bot_left.rb_mut(),
            lhs_bot_right,
            rhs_bot_left,
            alpha,
            beta,
            conj_lhs,
            conj_rhs,
            parallelism,
        );
        mat_x_lower_into_lower_impl_unchecked(
            dst_bot_right,
            skip_diag,
            lhs_bot_right,
            rhs_bot_right,
            rhs_diag,
            alpha,
            beta,
            conj_lhs,
            conj_rhs,
            parallelism,
        );

        mat_x_lower_into_lower_impl_unchecked(
            dst_top_left.rb_mut(),
            skip_diag,
            lhs_top_left,
            rhs_top_left,
            rhs_diag,
            alpha,
            beta,
            conj_lhs,
            conj_rhs,
            parallelism,
        );
        mat_x_mat_into_lower_impl_unchecked(
            dst_top_left,
            skip_diag,
            lhs_top_right,
            rhs_bot_left,
            Some(E::faer_one()),
            beta,
            conj_lhs,
            conj_rhs,
            parallelism,
        );
        mat_x_lower_impl_unchecked(
            dst_bot_left,
            lhs_bot_left,
            rhs_top_left,
            rhs_diag,
            Some(E::faer_one()),
            beta,
            conj_lhs,
            conj_rhs,
            parallelism,
        );
    }
}

unsafe fn mat_x_lower_impl_unchecked<E: ComplexField>(
    dst: MatMut<'_, E>,
    lhs: MatRef<'_, E>,
    rhs: MatRef<'_, E>,
    rhs_diag: DiagonalKind,
    alpha: Option<E>,
    beta: E,
    conj_lhs: Conj,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    let n = rhs.nrows();
    let m = lhs.nrows();
    debug_assert!(m == lhs.nrows());
    debug_assert!(n == lhs.ncols());
    debug_assert!(n == rhs.nrows());
    debug_assert!(n == rhs.ncols());
    debug_assert!(m == dst.nrows());
    debug_assert!(n == dst.ncols());

    let join_parallelism = if n * n * m < 128 * 128 * 64 {
        Parallelism::None
    } else {
        parallelism
    };

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat!(
                    [16, 16],
                    temp_rhs,
                    n,
                    n,
                    rhs.row_stride(),
                    rhs.col_stride(),
                    E
                );

                copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

                mul(
                    dst,
                    lhs,
                    temp_rhs.rb(),
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                );
            }
        };
        op();
    } else {
        // split rhs into 3 sections
        // split lhs and dst into 2 sections

        let bs = n / 2;

        let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at(bs, bs);
        let (lhs_left, lhs_right) = lhs.split_at_col(bs);
        let (mut dst_left, mut dst_right) = dst.split_at_col_mut(bs);

        join_raw(
            |parallelism| {
                mat_x_lower_impl_unchecked(
                    dst_left.rb_mut(),
                    lhs_left,
                    rhs_top_left,
                    rhs_diag,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            },
            |parallelism| {
                mat_x_lower_impl_unchecked(
                    dst_right.rb_mut(),
                    lhs_right,
                    rhs_bot_right,
                    rhs_diag,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            },
            join_parallelism,
        );
        mul(
            dst_left,
            lhs_right,
            rhs_bot_left,
            Some(E::faer_one()),
            beta,
            conj_lhs,
            conj_rhs,
            parallelism,
        );
    }
}

unsafe fn lower_x_lower_into_lower_impl_unchecked<E: ComplexField>(
    dst: MatMut<'_, E>,
    skip_diag: bool,
    lhs: MatRef<'_, E>,
    lhs_diag: DiagonalKind,
    rhs: MatRef<'_, E>,
    rhs_diag: DiagonalKind,
    alpha: Option<E>,
    beta: E,
    conj_lhs: Conj,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    let n = dst.nrows();
    debug_assert!(n == lhs.nrows());
    debug_assert!(n == lhs.ncols());
    debug_assert!(n == rhs.nrows());
    debug_assert!(n == rhs.ncols());
    debug_assert!(n == dst.nrows());
    debug_assert!(n == dst.ncols());

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat!(
                    [16, 16],
                    temp_dst,
                    n,
                    n,
                    dst.row_stride(),
                    dst.col_stride(),
                    E
                );
                stack_mat!(
                    [16, 16],
                    temp_lhs,
                    n,
                    n,
                    lhs.row_stride(),
                    lhs.col_stride(),
                    E
                );
                stack_mat!(
                    [16, 16],
                    temp_rhs,
                    n,
                    n,
                    rhs.row_stride(),
                    rhs.col_stride(),
                    E
                );

                copy_lower(temp_lhs.rb_mut(), lhs, lhs_diag);
                copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

                mul(
                    temp_dst.rb_mut(),
                    temp_lhs.rb(),
                    temp_rhs.rb(),
                    None,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                );
                accum_lower(dst, temp_dst.rb(), skip_diag, alpha);
            }
        };
        op();
    } else {
        let bs = n / 2;

        let (dst_top_left, _, mut dst_bot_left, dst_bot_right) = dst.split_at_mut(bs, bs);
        let (lhs_top_left, _, lhs_bot_left, lhs_bot_right) = lhs.split_at(bs, bs);
        let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at(bs, bs);

        // lhs_top_left  × rhs_top_left  => dst_top_left  | low × low => low |   X
        // lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2
        // lhs_bot_right × rhs_bot_left  => dst_bot_left  | low × mat => mat | 1/2
        // lhs_bot_right × rhs_bot_right => dst_bot_right | low × low => low |   X

        lower_x_lower_into_lower_impl_unchecked(
            dst_top_left,
            skip_diag,
            lhs_top_left,
            lhs_diag,
            rhs_top_left,
            rhs_diag,
            alpha,
            beta,
            conj_lhs,
            conj_rhs,
            parallelism,
        );
        mat_x_lower_impl_unchecked(
            dst_bot_left.rb_mut(),
            lhs_bot_left,
            rhs_top_left,
            rhs_diag,
            alpha,
            beta,
            conj_lhs,
            conj_rhs,
            parallelism,
        );
        mat_x_lower_impl_unchecked(
            dst_bot_left.reverse_rows_and_cols_mut().transpose_mut(),
            rhs_bot_left.reverse_rows_and_cols().transpose(),
            lhs_bot_right.reverse_rows_and_cols().transpose(),
            lhs_diag,
            Some(E::faer_one()),
            beta,
            conj_rhs,
            conj_lhs,
            parallelism,
        );
        lower_x_lower_into_lower_impl_unchecked(
            dst_bot_right,
            skip_diag,
            lhs_bot_right,
            lhs_diag,
            rhs_bot_right,
            rhs_diag,
            alpha,
            beta,
            conj_lhs,
            conj_rhs,
            parallelism,
        )
    }
}

unsafe fn upper_x_lower_impl_unchecked<E: ComplexField>(
    dst: MatMut<'_, E>,
    lhs: MatRef<'_, E>,
    lhs_diag: DiagonalKind,
    rhs: MatRef<'_, E>,
    rhs_diag: DiagonalKind,
    alpha: Option<E>,
    beta: E,
    conj_lhs: Conj,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    let n = dst.nrows();
    debug_assert!(n == lhs.nrows());
    debug_assert!(n == lhs.ncols());
    debug_assert!(n == rhs.nrows());
    debug_assert!(n == rhs.ncols());
    debug_assert!(n == dst.nrows());
    debug_assert!(n == dst.ncols());

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat!(
                    [16, 16],
                    temp_lhs,
                    n,
                    n,
                    lhs.row_stride(),
                    lhs.col_stride(),
                    E
                );
                stack_mat!(
                    [16, 16],
                    temp_rhs,
                    n,
                    n,
                    rhs.row_stride(),
                    rhs.col_stride(),
                    E
                );

                copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
                copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

                mul(
                    dst,
                    temp_lhs.rb(),
                    temp_rhs.rb(),
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                );
            }
        };
        op();
    } else {
        let bs = n / 2;

        let (mut dst_top_left, dst_top_right, dst_bot_left, dst_bot_right) =
            dst.split_at_mut(bs, bs);
        let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_at(bs, bs);
        let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at(bs, bs);

        // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => mat |   1
        // lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => mat |   X
        //
        // lhs_top_right × rhs_bot_right => dst_top_right | mat × low => mat | 1/2
        // lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
        // lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => mat |   X

        join_raw(
            |_| {
                mul(
                    dst_top_left.rb_mut(),
                    lhs_top_right,
                    rhs_bot_left,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                );
                upper_x_lower_impl_unchecked(
                    dst_top_left,
                    lhs_top_left,
                    lhs_diag,
                    rhs_top_left,
                    rhs_diag,
                    Some(E::faer_one()),
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            },
            |_| {
                join_raw(
                    |_| {
                        mat_x_lower_impl_unchecked(
                            dst_top_right,
                            lhs_top_right,
                            rhs_bot_right,
                            rhs_diag,
                            alpha,
                            beta,
                            conj_lhs,
                            conj_rhs,
                            parallelism,
                        )
                    },
                    |_| {
                        mat_x_lower_impl_unchecked(
                            dst_bot_left.transpose_mut(),
                            rhs_bot_left.transpose(),
                            lhs_bot_right.transpose(),
                            lhs_diag,
                            alpha,
                            beta,
                            conj_rhs,
                            conj_lhs,
                            parallelism,
                        )
                    },
                    parallelism,
                );

                upper_x_lower_impl_unchecked(
                    dst_bot_right,
                    lhs_bot_right,
                    lhs_diag,
                    rhs_bot_right,
                    rhs_diag,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            },
            parallelism,
        );
    }
}

unsafe fn upper_x_lower_into_lower_impl_unchecked<E: ComplexField>(
    dst: MatMut<'_, E>,
    skip_diag: bool,
    lhs: MatRef<'_, E>,
    lhs_diag: DiagonalKind,
    rhs: MatRef<'_, E>,
    rhs_diag: DiagonalKind,
    alpha: Option<E>,
    beta: E,
    conj_lhs: Conj,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    let n = dst.nrows();
    debug_assert!(n == lhs.nrows());
    debug_assert!(n == lhs.ncols());
    debug_assert!(n == rhs.nrows());
    debug_assert!(n == rhs.ncols());
    debug_assert!(n == dst.nrows());
    debug_assert!(n == dst.ncols());

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat!(
                    [16, 16],
                    temp_dst,
                    n,
                    n,
                    dst.row_stride(),
                    dst.col_stride(),
                    E
                );
                stack_mat!(
                    [16, 16],
                    temp_lhs,
                    n,
                    n,
                    lhs.row_stride(),
                    lhs.col_stride(),
                    E
                );
                stack_mat!(
                    [16, 16],
                    temp_rhs,
                    n,
                    n,
                    rhs.row_stride(),
                    rhs.col_stride(),
                    E
                );

                copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
                copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

                mul(
                    temp_dst.rb_mut(),
                    temp_lhs.rb(),
                    temp_rhs.rb(),
                    None,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                );

                accum_lower(dst, temp_dst.rb(), skip_diag, alpha);
            }
        };
        op();
    } else {
        let bs = n / 2;

        let (mut dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_at_mut(bs, bs);
        let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_at(bs, bs);
        let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at(bs, bs);

        // lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => low |   X
        // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
        //
        // lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
        // lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => low |   X

        join_raw(
            |_| {
                mat_x_mat_into_lower_impl_unchecked(
                    dst_top_left.rb_mut(),
                    skip_diag,
                    lhs_top_right,
                    rhs_bot_left,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                );
                upper_x_lower_into_lower_impl_unchecked(
                    dst_top_left,
                    skip_diag,
                    lhs_top_left,
                    lhs_diag,
                    rhs_top_left,
                    rhs_diag,
                    Some(E::faer_one()),
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            },
            |_| {
                mat_x_lower_impl_unchecked(
                    dst_bot_left.transpose_mut(),
                    rhs_bot_left.transpose(),
                    lhs_bot_right.transpose(),
                    lhs_diag,
                    alpha,
                    beta,
                    conj_rhs,
                    conj_lhs,
                    parallelism,
                );
                upper_x_lower_into_lower_impl_unchecked(
                    dst_bot_right,
                    skip_diag,
                    lhs_bot_right,
                    lhs_diag,
                    rhs_bot_right,
                    rhs_diag,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            },
            parallelism,
        );
    }
}

unsafe fn mat_x_mat_into_lower_impl_unchecked<E: ComplexField>(
    dst: MatMut<'_, E>,
    skip_diag: bool,
    lhs: MatRef<'_, E>,
    rhs: MatRef<'_, E>,
    alpha: Option<E>,
    beta: E,
    conj_lhs: Conj,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    debug_assert!(dst.nrows() == dst.ncols());
    debug_assert!(dst.nrows() == lhs.nrows());
    debug_assert!(dst.ncols() == rhs.ncols());
    debug_assert!(lhs.ncols() == rhs.nrows());

    let n = dst.nrows();
    let k = lhs.ncols();

    let join_parallelism = if n * n * k < 128 * 128 * 128 {
        Parallelism::None
    } else {
        parallelism
    };

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat!(
                    [16, 16],
                    temp_dst,
                    n,
                    n,
                    dst.row_stride(),
                    dst.col_stride(),
                    E
                );

                mul(
                    temp_dst.rb_mut(),
                    lhs,
                    rhs,
                    None,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                );
                accum_lower(dst, temp_dst.rb(), skip_diag, alpha);
            }
        };
        op();
    } else {
        let bs = n / 2;
        let (dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_at_mut(bs, bs);
        let (lhs_top, lhs_bot) = lhs.split_at_row(bs);
        let (rhs_left, rhs_right) = rhs.split_at_col(bs);

        join_raw(
            |_| {
                mul(
                    dst_bot_left,
                    lhs_bot,
                    rhs_left,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            },
            |_| {
                join_raw(
                    |_| {
                        mat_x_mat_into_lower_impl_unchecked(
                            dst_top_left,
                            skip_diag,
                            lhs_top,
                            rhs_left,
                            alpha,
                            beta,
                            conj_lhs,
                            conj_rhs,
                            parallelism,
                        )
                    },
                    |_| {
                        mat_x_mat_into_lower_impl_unchecked(
                            dst_bot_right,
                            skip_diag,
                            lhs_bot,
                            rhs_right,
                            alpha,
                            beta,
                            conj_lhs,
                            conj_rhs,
                            parallelism,
                        )
                    },
                    join_parallelism,
                )
            },
            join_parallelism,
        );
    }
}

/// Describes the parts of the matrix that must be accessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockStructure {
    /// The full matrix is accessed.
    Rectangular,
    /// The lower triangular half (including the diagonal) is accessed.
    TriangularLower,
    /// The lower triangular half (excluding the diagonal) is accessed.
    StrictTriangularLower,
    /// The lower triangular half (excluding the diagonal, which is assumed to be equal to
    /// `1.0`) is accessed.
    UnitTriangularLower,
    /// The upper triangular half (including the diagonal) is accessed.
    TriangularUpper,
    /// The upper triangular half (excluding the diagonal) is accessed.
    StrictTriangularUpper,
    /// The upper triangular half (excluding the diagonal, which is assumed to be equal to
    /// `1.0`) is accessed.
    UnitTriangularUpper,
}

impl BlockStructure {
    /// Checks if `self` is full.
    #[inline]
    pub fn is_dense(self) -> bool {
        matches!(self, BlockStructure::Rectangular)
    }

    /// Checks if `self` is triangular lower (either inclusive or exclusive).
    #[inline]
    pub fn is_lower(self) -> bool {
        use BlockStructure::*;
        matches!(
            self,
            TriangularLower | StrictTriangularLower | UnitTriangularLower
        )
    }

    /// Checks if `self` is triangular upper (either inclusive or exclusive).
    #[inline]
    pub fn is_upper(self) -> bool {
        use BlockStructure::*;
        matches!(
            self,
            TriangularUpper | StrictTriangularUpper | UnitTriangularUpper
        )
    }

    /// Returns the block structure corresponding to the transposed matrix.
    #[inline]
    pub fn transpose(self) -> Self {
        use BlockStructure::*;
        match self {
            Rectangular => Rectangular,
            TriangularLower => TriangularUpper,
            StrictTriangularLower => StrictTriangularUpper,
            UnitTriangularLower => UnitTriangularUpper,
            TriangularUpper => TriangularLower,
            StrictTriangularUpper => StrictTriangularLower,
            UnitTriangularUpper => UnitTriangularLower,
        }
    }

    #[inline]
    pub(crate) fn diag_kind(self) -> DiagonalKind {
        use BlockStructure::*;
        match self {
            Rectangular | TriangularLower | TriangularUpper => DiagonalKind::Generic,
            StrictTriangularLower | StrictTriangularUpper => DiagonalKind::Zero,
            UnitTriangularLower | UnitTriangularUpper => DiagonalKind::Unit,
        }
    }
}

/// Computes the matrix product `[alpha * acc] + beta * lhs * rhs` (while optionally conjugating
/// either or both of the input matrices) and stores the result in `acc`.
///
/// Performs the operation:
/// - `acc = beta * Op_lhs(lhs) * Op_rhs(rhs)` if `alpha` is `None` (in this case, the preexisting
///   values in `acc` are not read, so it is allowed to be a view over uninitialized values if `E:
///   Copy`),
/// - `acc = alpha * acc + beta * Op_lhs(lhs) * Op_rhs(rhs)` if `alpha` is `Some(_)`,
///
/// The left hand side and right hand side may be interpreted as triangular depending on the
/// given corresponding matrix structure.  
///
/// For the destination matrix, the result is:
/// - fully computed if the structure is rectangular,
/// - only the triangular half (including the diagonal) is computed if the structure is
/// triangular,
/// - only the strict triangular half (excluding the diagonal) is computed if the structure is
/// strictly triangular or unit triangular.
///
/// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
/// `Op_rhs` is the identity if `conj_rhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
///
/// # Panics
///
/// Panics if the matrix dimensions are not compatible for matrix multiplication.  
/// i.e.  
///  - `acc.nrows() == lhs.nrows()`
///  - `acc.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
///
///  Additionally, matrices that are marked as triangular must be square, i.e., they must have
///  the same number of rows and columns.
///
/// # Example
///
/// ```
/// use faer::{
///     linalg::matmul::triangular::{matmul_with_conj, BlockStructure},
///     mat, unzipped, zipped, Conj, Mat, Parallelism,
/// };
///
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
///
/// let mut acc = Mat::<f64>::zeros(2, 2);
/// let target = mat![
///     [
///         2.5 * (lhs.read(0, 0) * rhs.read(0, 0) + lhs.read(0, 1) * rhs.read(1, 0)),
///         0.0,
///     ],
///     [
///         2.5 * (lhs.read(1, 0) * rhs.read(0, 0) + lhs.read(1, 1) * rhs.read(1, 0)),
///         2.5 * (lhs.read(1, 0) * rhs.read(0, 1) + lhs.read(1, 1) * rhs.read(1, 1)),
///     ],
/// ];
///
/// matmul_with_conj(
///     acc.as_mut(),
///     BlockStructure::TriangularLower,
///     lhs.as_ref(),
///     BlockStructure::Rectangular,
///     Conj::No,
///     rhs.as_ref(),
///     BlockStructure::Rectangular,
///     Conj::No,
///     None,
///     2.5,
///     Parallelism::None,
/// );
///
/// zipped!(__rw, acc.as_ref(), target.as_ref())
///     .for_each(|unzipped!(acc, target)| assert!((acc.read() - target.read()).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn matmul_with_conj<E: ComplexField>(
    acc: impl As2DMut<E>,
    acc_structure: BlockStructure,
    lhs: impl As2D<E>,
    lhs_structure: BlockStructure,
    conj_lhs: Conj,
    rhs: impl As2D<E>,
    rhs_structure: BlockStructure,
    conj_rhs: Conj,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
) {
    let mut acc = acc;
    let acc = acc.as_2d_mut();
    let lhs = lhs.as_2d_ref();
    let rhs = rhs.as_2d_ref();

    assert!(all(
        acc.nrows() == lhs.nrows(),
        acc.ncols() == rhs.ncols(),
        lhs.ncols() == rhs.nrows(),
    ));

    if !acc_structure.is_dense() {
        assert!(acc.nrows() == acc.ncols());
    }
    if !lhs_structure.is_dense() {
        assert!(lhs.nrows() == lhs.ncols());
    }
    if !rhs_structure.is_dense() {
        assert!(rhs.nrows() == rhs.ncols());
    }

    unsafe {
        matmul_unchecked(
            acc,
            acc_structure,
            lhs,
            lhs_structure,
            conj_lhs,
            rhs,
            rhs_structure,
            conj_rhs,
            alpha,
            beta,
            parallelism,
        )
    }
}

/// Computes the matrix product `[alpha * acc] + beta * lhs * rhs` and stores the result in
/// `acc`.
///
/// Performs the operation:
/// - `acc = beta * lhs * rhs` if `alpha` is `None` (in this case, the preexisting values in `acc`
///   are not read, so it is allowed to be a view over uninitialized values if `E: Copy`),
/// - `acc = alpha * acc + beta * lhs * rhs` if `alpha` is `Some(_)`,
///
/// The left hand side and right hand side may be interpreted as triangular depending on the
/// given corresponding matrix structure.  
///
/// For the destination matrix, the result is:
/// - fully computed if the structure is rectangular,
/// - only the triangular half (including the diagonal) is computed if the structure is
/// triangular,
/// - only the strict triangular half (excluding the diagonal) is computed if the structure is
/// strictly triangular or unit triangular.
///
/// # Panics
///
/// Panics if the matrix dimensions are not compatible for matrix multiplication.  
/// i.e.  
///  - `acc.nrows() == lhs.nrows()`
///  - `acc.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
///
///  Additionally, matrices that are marked as triangular must be square, i.e., they must have
///  the same number of rows and columns.
///
/// # Example
///
/// ```
/// use faer::{
///     linalg::matmul::triangular::{matmul, BlockStructure},
///     mat, unzipped, zipped, Conj, Mat, Parallelism,
/// };
///
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
///
/// let mut acc = Mat::<f64>::zeros(2, 2);
/// let target = mat![
///     [
///         2.5 * (lhs.read(0, 0) * rhs.read(0, 0) + lhs.read(0, 1) * rhs.read(1, 0)),
///         0.0,
///     ],
///     [
///         2.5 * (lhs.read(1, 0) * rhs.read(0, 0) + lhs.read(1, 1) * rhs.read(1, 0)),
///         2.5 * (lhs.read(1, 0) * rhs.read(0, 1) + lhs.read(1, 1) * rhs.read(1, 1)),
///     ],
/// ];
///
/// matmul(
///     acc.as_mut(),
///     BlockStructure::TriangularLower,
///     lhs.as_ref(),
///     BlockStructure::Rectangular,
///     rhs.as_ref(),
///     BlockStructure::Rectangular,
///     None,
///     2.5,
///     Parallelism::None,
/// );
///
/// zipped!(__rw, acc.as_ref(), target.as_ref())
///     .for_each(|unzipped!(acc, target)| assert!((acc.read() - target.read()).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn matmul<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
    acc: impl As2DMut<E>,
    acc_structure: BlockStructure,
    lhs: impl As2D<LhsE>,
    lhs_structure: BlockStructure,
    rhs: impl As2D<RhsE>,
    rhs_structure: BlockStructure,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
) {
    let mut acc = acc;
    let acc = acc.as_2d_mut();
    let lhs = lhs.as_2d_ref();
    let rhs = rhs.as_2d_ref();

    let (lhs, conj_lhs) = lhs.canonicalize();
    let (rhs, conj_rhs) = rhs.canonicalize();
    matmul_with_conj(
        acc,
        acc_structure,
        lhs,
        lhs_structure,
        conj_lhs,
        rhs,
        rhs_structure,
        conj_rhs,
        alpha,
        beta,
        parallelism,
    );
}

unsafe fn matmul_unchecked<E: ComplexField>(
    acc: MatMut<'_, E>,
    acc_structure: BlockStructure,
    lhs: MatRef<'_, E>,
    lhs_structure: BlockStructure,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    rhs_structure: BlockStructure,
    conj_rhs: Conj,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
) {
    debug_assert!(acc.nrows() == lhs.nrows());
    debug_assert!(acc.ncols() == rhs.ncols());
    debug_assert!(lhs.ncols() == rhs.nrows());

    if !acc_structure.is_dense() {
        debug_assert!(acc.nrows() == acc.ncols());
    }
    if !lhs_structure.is_dense() {
        debug_assert!(lhs.nrows() == lhs.ncols());
    }
    if !rhs_structure.is_dense() {
        debug_assert!(rhs.nrows() == rhs.ncols());
    }

    let mut acc = acc;
    let mut lhs = lhs;
    let mut rhs = rhs;

    let mut acc_structure = acc_structure;
    let mut lhs_structure = lhs_structure;
    let mut rhs_structure = rhs_structure;

    let mut conj_lhs = conj_lhs;
    let mut conj_rhs = conj_rhs;

    // if either the lhs or the rhs is triangular
    if rhs_structure.is_lower() {
        // do nothing
        false
    } else if rhs_structure.is_upper() {
        // invert acc, lhs and rhs
        acc = acc.reverse_rows_and_cols_mut();
        lhs = lhs.reverse_rows_and_cols();
        rhs = rhs.reverse_rows_and_cols();
        acc_structure = acc_structure.transpose();
        lhs_structure = lhs_structure.transpose();
        rhs_structure = rhs_structure.transpose();
        false
    } else if lhs_structure.is_lower() {
        // invert and transpose
        acc = acc.reverse_rows_and_cols_mut().transpose_mut();
        (lhs, rhs) = (
            rhs.reverse_rows_and_cols().transpose(),
            lhs.reverse_rows_and_cols().transpose(),
        );
        (conj_lhs, conj_rhs) = (conj_rhs, conj_lhs);
        (lhs_structure, rhs_structure) = (rhs_structure, lhs_structure);
        true
    } else if lhs_structure.is_upper() {
        // transpose
        acc_structure = acc_structure.transpose();
        acc = acc.transpose_mut();
        (lhs, rhs) = (rhs.transpose(), lhs.transpose());
        (conj_lhs, conj_rhs) = (conj_rhs, conj_lhs);
        (lhs_structure, rhs_structure) = (rhs_structure.transpose(), lhs_structure.transpose());
        true
    } else {
        // do nothing
        false
    };

    let clear_upper = |acc: MatMut<'_, E>, skip_diag: bool| match &alpha {
        &Some(alpha) => zipped!(__rw, acc).for_each_triangular_upper(
            if skip_diag { Diag::Skip } else { Diag::Include },
            |unzipped!(mut acc)| acc.write(alpha.faer_mul(acc.read())),
        ),

        None => zipped!(__rw, acc).for_each_triangular_upper(
            if skip_diag { Diag::Skip } else { Diag::Include },
            |unzipped!(mut acc)| acc.write(E::faer_zero()),
        ),
    };

    let skip_diag = matches!(
        acc_structure,
        BlockStructure::StrictTriangularLower
            | BlockStructure::StrictTriangularUpper
            | BlockStructure::UnitTriangularLower
            | BlockStructure::UnitTriangularUpper
    );
    let lhs_diag = lhs_structure.diag_kind();
    let rhs_diag = rhs_structure.diag_kind();

    if acc_structure.is_dense() {
        if lhs_structure.is_dense() && rhs_structure.is_dense() {
            mul(acc, lhs, rhs, alpha, beta, conj_lhs, conj_rhs, parallelism);
        } else {
            debug_assert!(rhs_structure.is_lower());

            if lhs_structure.is_dense() {
                mat_x_lower_impl_unchecked(
                    acc,
                    lhs,
                    rhs,
                    rhs_diag,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            } else if lhs_structure.is_lower() {
                clear_upper(acc.rb_mut(), true);
                lower_x_lower_into_lower_impl_unchecked(
                    acc,
                    false,
                    lhs,
                    lhs_diag,
                    rhs,
                    rhs_diag,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                );
            } else {
                debug_assert!(lhs_structure.is_upper());
                upper_x_lower_impl_unchecked(
                    acc,
                    lhs,
                    lhs_diag,
                    rhs,
                    rhs_diag,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            }
        }
    } else if acc_structure.is_lower() {
        if lhs_structure.is_dense() && rhs_structure.is_dense() {
            mat_x_mat_into_lower_impl_unchecked(
                acc,
                skip_diag,
                lhs,
                rhs,
                alpha,
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            )
        } else {
            debug_assert!(rhs_structure.is_lower());
            if lhs_structure.is_dense() {
                mat_x_lower_into_lower_impl_unchecked(
                    acc,
                    skip_diag,
                    lhs,
                    rhs,
                    rhs_diag,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                );
            } else if lhs_structure.is_lower() {
                lower_x_lower_into_lower_impl_unchecked(
                    acc,
                    skip_diag,
                    lhs,
                    lhs_diag,
                    rhs,
                    rhs_diag,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            } else {
                upper_x_lower_into_lower_impl_unchecked(
                    acc,
                    skip_diag,
                    lhs,
                    lhs_diag,
                    rhs,
                    rhs_diag,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            }
        }
    } else if lhs_structure.is_dense() && rhs_structure.is_dense() {
        mat_x_mat_into_lower_impl_unchecked(
            acc.transpose_mut(),
            skip_diag,
            rhs.transpose(),
            lhs.transpose(),
            alpha,
            beta,
            conj_rhs,
            conj_lhs,
            parallelism,
        )
    } else {
        debug_assert!(rhs_structure.is_lower());
        if lhs_structure.is_dense() {
            // lower part of lhs does not contribute to result
            upper_x_lower_into_lower_impl_unchecked(
                acc.transpose_mut(),
                skip_diag,
                rhs.transpose(),
                rhs_diag,
                lhs.transpose(),
                lhs_diag,
                alpha,
                beta,
                conj_rhs,
                conj_lhs,
                parallelism,
            )
        } else if lhs_structure.is_lower() {
            if !skip_diag {
                match &alpha {
                    &Some(alpha) => {
                        zipped!(
                            __rw,
                            acc.rb_mut().diagonal_mut().column_vector_mut().as_2d_mut(),
                            lhs.diagonal().column_vector().as_2d(),
                            rhs.diagonal().column_vector().as_2d(),
                        )
                        .for_each(|unzipped!(mut acc, lhs, rhs)| {
                            acc.write(
                                (alpha.faer_mul(acc.read()))
                                    .faer_add(beta.faer_mul(lhs.read().faer_mul(rhs.read()))),
                            )
                        });
                    }
                    None => {
                        zipped!(
                            __rw,
                            acc.rb_mut().diagonal_mut().column_vector_mut().as_2d_mut(),
                            lhs.diagonal().column_vector().as_2d(),
                            rhs.diagonal().column_vector().as_2d(),
                        )
                        .for_each(|unzipped!(mut acc, lhs, rhs)| {
                            acc.write(beta.faer_mul(lhs.read().faer_mul(rhs.read())))
                        });
                    }
                }
            }
            clear_upper(acc.rb_mut(), true);
        } else {
            debug_assert!(lhs_structure.is_upper());
            upper_x_lower_into_lower_impl_unchecked(
                acc.transpose_mut(),
                skip_diag,
                rhs.transpose(),
                rhs_diag,
                lhs.transpose(),
                lhs_diag,
                alpha,
                beta,
                conj_rhs,
                conj_lhs,
                parallelism,
            )
        }
    }
}
