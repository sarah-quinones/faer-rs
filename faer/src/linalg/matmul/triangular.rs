use faer_traits::ComplexContainer;

use super::*;
use crate::{
    assert, debug_assert,
    linalg::{temp_mat_uninit, zip::Diag},
    mat::{AsMatMut, MatMutGeneric, MatRefGeneric},
    unzipped,
    utils::thread::join_raw,
    zipped,
};

macro_rules! sync_opt {
    ($x: expr) => {
        $x.as_ref().map(|val| sync!(copy!(*val)))
    };
}
macro_rules! unsync_opt {
    ($x: expr) => {
        $x.map(|val| unsync!(val))
    };
}

macro_rules! copy_opt {
    ($x: expr) => {
        $x.as_ref().map(|val| copy!(*val))
    };
}

#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub(crate) enum DiagonalKind {
    Zero,
    Unit,
    Generic,
}

#[faer_macros::math]
fn copy_lower<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    src: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    src_diag: DiagonalKind,
) {
    help!(C);
    let N = dst.nrows();
    let mut dst = dst;
    match src_diag {
        DiagonalKind::Zero => {
            dst.copy_from_strict_triangular_lower_with_ctx(ctx, src);
            let ctx = Ctx::<C, T>::new(ctx);
            for j in N.indices() {
                let zero = math(zero());
                write1!(dst.write(j, j), zero);
            }
        }
        DiagonalKind::Unit => {
            dst.copy_from_strict_triangular_lower_with_ctx(ctx, src);
            let ctx = Ctx::<C, T>::new(ctx);
            for j in N.indices() {
                let one = math(one());
                write1!(dst.write(j, j), one);
            }
        }
        DiagonalKind::Generic => dst.copy_from_triangular_lower_with_ctx(ctx, src),
    }

    zipped!(dst).for_each_triangular_upper(Diag::Skip, |unzipped!(mut dst)| {
        let zero = math(zero());
        write1!(dst, zero)
    });
}

#[faer_macros::math]
fn accum_lower<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    src: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    skip_diag: bool,
    beta: Option<C::Of<&T>>,
) {
    let N = dst.nrows();
    debug_assert!(N == dst.nrows());
    debug_assert!(N == dst.ncols());
    debug_assert!(N == src.nrows());
    debug_assert!(N == src.ncols());
    help!(C);

    match beta {
        Some(alpha) => {
            zipped!(dst, src).for_each_triangular_lower(
                if skip_diag { Diag::Skip } else { Diag::Include },
                |unzipped!(mut dst, src)| {
                    let val = math(alpha * rb!(dst) + src);
                    write1!(dst, val);
                },
            );
        }
        None => {
            zipped!(dst, src).for_each_triangular_lower(
                if skip_diag { Diag::Skip } else { Diag::Include },
                |unzipped!(mut dst, src)| {
                    let src = math(copy(src));
                    write1!(dst, src)
                },
            );
        }
    }
}

#[faer_macros::math]
fn copy_upper<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    src: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    src_diag: DiagonalKind,
) {
    copy_lower(ctx, dst.transpose_mut(), src.transpose(), src_diag)
}

#[repr(align(64))]
struct Storage<T>([T; 16 * 16]);

macro_rules! stack_mat_16x16 {
    ($ctx: expr, $name: ident, $n: expr, $rs: expr, $cs: expr, $C: ty, $T: ty $(,)?) => {
        let mut __tmp = core::mem::MaybeUninit::<C::Of<Storage<T>>>::uninit();
        let __stack = DynStack::new_any(core::slice::from_mut(&mut __tmp));
        let mut $name = unsafe { temp_mat_uninit($ctx, $n, $n, __stack) }.0;
        let mut $name = $name.as_mat_mut();
        if $cs.unsigned_abs() == 1 {
            $name = $name.transpose_mut();
            if $cs == 1 {
                $name = $name.transpose_mut().reverse_cols_mut();
            }
        } else if $rs == -1 {
            $name = $name.reverse_rows_mut();
        }
    };
}

#[faer_macros::math]
fn mat_x_lower_impl_unchecked<'M, 'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, Dim<'M>, Dim<'N>>,
    beta: Option<C::Of<&T>>,
    lhs: MatRefGeneric<'_, C, T, Dim<'M>, Dim<'N>>,
    rhs: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    rhs_diag: DiagonalKind,
    alpha: C::Of<&T>,
    conj_lhs: Conj,
    conj_rhs: Conj,
    par: Parallelism,
) {
    let N = rhs.nrows();
    let M = lhs.nrows();
    let n = N.unbound();
    let m = M.unbound();
    debug_assert!(M == lhs.nrows());
    debug_assert!(N == lhs.ncols());
    debug_assert!(N == rhs.nrows());
    debug_assert!(N == rhs.ncols());
    debug_assert!(M == dst.nrows());
    debug_assert!(N == dst.ncols());

    let join_parallelism = if n * n * m < 128 * 128 * 64 {
        Parallelism::None
    } else {
        par
    };

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat_16x16!(ctx, temp_rhs, N, rhs.row_stride(), rhs.col_stride(), C, T);

                copy_lower(ctx, temp_rhs.rb_mut(), rhs, rhs_diag);

                super::matmul_with_conj(
                    ctx,
                    dst,
                    beta,
                    lhs,
                    conj_lhs,
                    temp_rhs.rb(),
                    conj_rhs,
                    alpha,
                    par,
                );
            }
        };
        op();
    } else {
        // split rhs into 3 sections
        // split lhs and dst into 2 sections

        make_guard!(HEAD);
        make_guard!(TAIL);
        let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

        let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_with(bs, bs);
        let (lhs_left, lhs_right) = lhs.split_cols_with(bs);
        let (mut dst_left, mut dst_right) = dst.split_cols_with_mut(bs);

        {
            help!(C);
            let alpha = sync!(copy!(alpha));
            let beta = beta.as_ref().map(|beta| sync!(copy!(*beta)));
            join_raw(
                |par| {
                    mat_x_lower_impl_unchecked(
                        ctx,
                        dst_left.rb_mut(),
                        beta.map(|beta| unsync!(beta)),
                        lhs_left,
                        rhs_top_left,
                        rhs_diag,
                        unsync!(alpha),
                        conj_lhs,
                        conj_rhs,
                        par,
                    )
                },
                |par| {
                    mat_x_lower_impl_unchecked(
                        ctx,
                        dst_right.rb_mut(),
                        beta.map(|beta| unsync!(beta)),
                        lhs_right,
                        rhs_bot_right,
                        rhs_diag,
                        unsync!(alpha),
                        conj_lhs,
                        conj_rhs,
                        par,
                    )
                },
                join_parallelism,
            )
        };
        help!(C);
        super::matmul_with_conj(
            ctx,
            dst_left,
            Some(as_ref!(math.one())),
            lhs_right,
            conj_lhs,
            rhs_bot_left,
            conj_rhs,
            alpha,
            par,
        );
    }
}

#[faer_macros::math]
fn lower_x_lower_into_lower_impl_unchecked<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    beta: Option<C::Of<&T>>,
    skip_diag: bool,
    lhs: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    lhs_diag: DiagonalKind,
    rhs: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    rhs_diag: DiagonalKind,
    alpha: C::Of<&T>,
    conj_lhs: Conj,
    conj_rhs: Conj,
    par: Parallelism,
) {
    let N = dst.nrows();
    let n = N.unbound();
    debug_assert!(N == lhs.nrows());
    debug_assert!(N == lhs.ncols());
    debug_assert!(N == rhs.nrows());
    debug_assert!(N == rhs.ncols());
    debug_assert!(N == dst.nrows());
    debug_assert!(N == dst.ncols());

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat_16x16!(ctx, temp_dst, N, dst.row_stride(), dst.col_stride(), C, T);
                stack_mat_16x16!(ctx, temp_lhs, N, lhs.row_stride(), lhs.col_stride(), C, T);
                stack_mat_16x16!(ctx, temp_rhs, N, rhs.row_stride(), rhs.col_stride(), C, T);

                copy_lower(ctx, temp_lhs.rb_mut(), lhs, lhs_diag);
                copy_lower(ctx, temp_rhs.rb_mut(), rhs, rhs_diag);

                super::matmul_with_conj(
                    ctx,
                    temp_dst.rb_mut(),
                    None,
                    temp_lhs.rb(),
                    conj_lhs,
                    temp_rhs.rb(),
                    conj_rhs,
                    alpha,
                    par,
                );
                accum_lower(ctx, dst, temp_dst.rb(), skip_diag, beta);
            }
        };
        op();
    } else {
        make_guard!(HEAD);
        make_guard!(TAIL);
        let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

        let (dst_top_left, _, mut dst_bot_left, dst_bot_right) = dst.split_with_mut(bs, bs);
        let (lhs_top_left, _, lhs_bot_left, lhs_bot_right) = lhs.split_with(bs, bs);
        let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_with(bs, bs);

        // lhs_top_left  × rhs_top_left  => dst_top_left  | low × low => low |   X
        // lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2
        // lhs_bot_right × rhs_bot_left  => dst_bot_left  | low × mat => mat | 1/2
        // lhs_bot_right × rhs_bot_right => dst_bot_right | low × low => low |   X

        help!(C);
        lower_x_lower_into_lower_impl_unchecked(
            ctx,
            dst_top_left,
            copy_opt!(beta),
            skip_diag,
            lhs_top_left,
            lhs_diag,
            rhs_top_left,
            rhs_diag,
            copy!(alpha),
            conj_lhs,
            conj_rhs,
            par,
        );
        mat_x_lower_impl_unchecked(
            ctx,
            dst_bot_left.rb_mut(),
            copy_opt!(beta),
            lhs_bot_left,
            rhs_top_left,
            rhs_diag,
            copy!(alpha),
            conj_lhs,
            conj_rhs,
            par,
        );
        mat_x_lower_impl_unchecked(
            ctx,
            dst_bot_left.reverse_rows_and_cols_mut().transpose_mut(),
            Some(as_ref!(math.one())),
            rhs_bot_left.reverse_rows_and_cols().transpose(),
            lhs_bot_right.reverse_rows_and_cols().transpose(),
            lhs_diag,
            copy!(alpha),
            conj_rhs,
            conj_lhs,
            par,
        );
        lower_x_lower_into_lower_impl_unchecked(
            ctx,
            dst_bot_right,
            copy_opt!(beta),
            skip_diag,
            lhs_bot_right,
            lhs_diag,
            rhs_bot_right,
            rhs_diag,
            copy!(alpha),
            conj_lhs,
            conj_rhs,
            par,
        )
    }
}

#[math]
fn upper_x_lower_impl_unchecked<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    beta: Option<C::Of<&T>>,
    lhs: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    lhs_diag: DiagonalKind,
    rhs: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    rhs_diag: DiagonalKind,
    alpha: C::Of<&T>,
    conj_lhs: Conj,
    conj_rhs: Conj,
    par: Parallelism,
) {
    let N = dst.nrows();
    let n = N.unbound();
    debug_assert!(N == lhs.nrows());
    debug_assert!(N == lhs.ncols());
    debug_assert!(N == rhs.nrows());
    debug_assert!(N == rhs.ncols());
    debug_assert!(N == dst.nrows());
    debug_assert!(N == dst.ncols());

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat_16x16!(ctx, temp_lhs, N, lhs.row_stride(), lhs.col_stride(), C, T);
                stack_mat_16x16!(ctx, temp_rhs, N, rhs.row_stride(), rhs.col_stride(), C, T);

                copy_upper(ctx, temp_lhs.rb_mut(), lhs, lhs_diag);
                copy_lower(ctx, temp_rhs.rb_mut(), rhs, rhs_diag);

                super::matmul_with_conj(
                    ctx,
                    dst,
                    beta,
                    temp_lhs.rb(),
                    conj_lhs,
                    temp_rhs.rb(),
                    conj_rhs,
                    alpha,
                    par,
                );
            }
        };
        op();
    } else {
        make_guard!(HEAD);
        make_guard!(TAIL);
        let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

        let (mut dst_top_left, dst_top_right, dst_bot_left, dst_bot_right) =
            dst.split_with_mut(bs, bs);
        let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_with(bs, bs);
        let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_with(bs, bs);

        // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => mat |   1
        // lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => mat |   X
        //
        // lhs_top_right × rhs_bot_right => dst_top_right | mat × low => mat | 1/2
        // lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
        // lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => mat |   X

        help!(C);
        let beta = sync_opt!(beta);
        let alpha = sync!(alpha);
        join_raw(
            |par| {
                super::matmul_with_conj(
                    ctx,
                    dst_top_left.rb_mut(),
                    unsync_opt!(beta),
                    lhs_top_right,
                    conj_lhs,
                    rhs_bot_left,
                    conj_rhs,
                    unsync!(alpha),
                    par,
                );
                upper_x_lower_impl_unchecked(
                    ctx,
                    dst_top_left,
                    Some(as_ref!(math.one())),
                    lhs_top_left,
                    lhs_diag,
                    rhs_top_left,
                    rhs_diag,
                    unsync!(alpha),
                    conj_lhs,
                    conj_rhs,
                    par,
                )
            },
            |par| {
                join_raw(
                    |par| {
                        mat_x_lower_impl_unchecked(
                            ctx,
                            dst_top_right,
                            unsync_opt!(beta),
                            lhs_top_right,
                            rhs_bot_right,
                            rhs_diag,
                            unsync!(alpha),
                            conj_lhs,
                            conj_rhs,
                            par,
                        )
                    },
                    |par| {
                        mat_x_lower_impl_unchecked(
                            ctx,
                            dst_bot_left.transpose_mut(),
                            unsync_opt!(beta),
                            rhs_bot_left.transpose(),
                            lhs_bot_right.transpose(),
                            lhs_diag,
                            unsync!(alpha),
                            conj_rhs,
                            conj_lhs,
                            par,
                        )
                    },
                    par,
                );

                upper_x_lower_impl_unchecked(
                    ctx,
                    dst_bot_right,
                    unsync_opt!(beta),
                    lhs_bot_right,
                    lhs_diag,
                    rhs_bot_right,
                    rhs_diag,
                    unsync!(alpha),
                    conj_lhs,
                    conj_rhs,
                    par,
                )
            },
            par,
        );
    }
}

#[math]
fn upper_x_lower_into_lower_impl_unchecked<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    beta: Option<C::Of<&T>>,
    skip_diag: bool,
    lhs: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    lhs_diag: DiagonalKind,
    rhs: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    rhs_diag: DiagonalKind,
    alpha: C::Of<&T>,
    conj_lhs: Conj,
    conj_rhs: Conj,
    par: Parallelism,
) {
    let N = dst.nrows();
    let n = N.unbound();
    debug_assert!(N == lhs.nrows());
    debug_assert!(N == lhs.ncols());
    debug_assert!(N == rhs.nrows());
    debug_assert!(N == rhs.ncols());
    debug_assert!(N == dst.nrows());
    debug_assert!(N == dst.ncols());

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat_16x16!(ctx, temp_dst, N, dst.row_stride(), dst.col_stride(), C, T);
                stack_mat_16x16!(ctx, temp_lhs, N, lhs.row_stride(), lhs.col_stride(), C, T);
                stack_mat_16x16!(ctx, temp_rhs, N, rhs.row_stride(), rhs.col_stride(), C, T);

                copy_upper(ctx, temp_lhs.rb_mut(), lhs, lhs_diag);
                copy_lower(ctx, temp_rhs.rb_mut(), rhs, rhs_diag);

                super::matmul_with_conj(
                    ctx,
                    temp_dst.rb_mut(),
                    None,
                    temp_lhs.rb(),
                    conj_lhs,
                    temp_rhs.rb(),
                    conj_rhs,
                    alpha,
                    par,
                );

                accum_lower(ctx, dst, temp_dst.rb(), skip_diag, beta);
            }
        };
        op();
    } else {
        make_guard!(HEAD);
        make_guard!(TAIL);
        let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

        let (mut dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_with_mut(bs, bs);
        let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_with(bs, bs);
        let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_with(bs, bs);

        // lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => low |   X
        // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
        //
        // lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
        // lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => low |   X

        help!(C);
        let beta = sync_opt!(beta);
        let alpha = sync!(alpha);

        join_raw(
            |par| {
                mat_x_mat_into_lower_impl_unchecked(
                    ctx,
                    dst_top_left.rb_mut(),
                    unsync_opt!(beta),
                    skip_diag,
                    lhs_top_right,
                    rhs_bot_left,
                    unsync!(alpha),
                    conj_lhs,
                    conj_rhs,
                    par,
                );
                upper_x_lower_into_lower_impl_unchecked(
                    ctx,
                    dst_top_left,
                    Some(as_ref!(math.one())),
                    skip_diag,
                    lhs_top_left,
                    lhs_diag,
                    rhs_top_left,
                    rhs_diag,
                    unsync!(alpha),
                    conj_lhs,
                    conj_rhs,
                    par,
                )
            },
            |par| {
                mat_x_lower_impl_unchecked(
                    ctx,
                    dst_bot_left.transpose_mut(),
                    unsync_opt!(beta),
                    rhs_bot_left.transpose(),
                    lhs_bot_right.transpose(),
                    lhs_diag,
                    unsync!(alpha),
                    conj_rhs,
                    conj_lhs,
                    par,
                );
                upper_x_lower_into_lower_impl_unchecked(
                    ctx,
                    dst_bot_right,
                    unsync_opt!(beta),
                    skip_diag,
                    lhs_bot_right,
                    lhs_diag,
                    rhs_bot_right,
                    rhs_diag,
                    unsync!(alpha),
                    conj_lhs,
                    conj_rhs,
                    par,
                )
            },
            par,
        );
    }
}

#[math]
fn mat_x_mat_into_lower_impl_unchecked<'N, 'K, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    beta: Option<C::Of<&T>>,
    skip_diag: bool,
    lhs: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'K>>,
    rhs: MatRefGeneric<'_, C, T, Dim<'K>, Dim<'N>>,
    alpha: C::Of<&T>,
    conj_lhs: Conj,
    conj_rhs: Conj,
    par: Parallelism,
) {
    let N = dst.nrows();
    let K = lhs.ncols();
    let n = N.unbound();
    let k = K.unbound();
    debug_assert!(dst.nrows() == dst.ncols());
    debug_assert!(dst.nrows() == lhs.nrows());
    debug_assert!(dst.ncols() == rhs.ncols());
    debug_assert!(lhs.ncols() == rhs.nrows());

    let par = if n * n * k < 128 * 128 * 128 {
        Parallelism::None
    } else {
        par
    };

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat_16x16!(ctx, temp_dst, N, dst.row_stride(), dst.col_stride(), C, T);

                super::matmul_with_conj(
                    ctx,
                    temp_dst.rb_mut(),
                    None,
                    lhs,
                    conj_lhs,
                    rhs,
                    conj_rhs,
                    alpha,
                    par,
                );
                accum_lower(ctx, dst, temp_dst.rb(), skip_diag, beta);
            }
        };
        op();
    } else {
        make_guard!(HEAD);
        make_guard!(TAIL);
        let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

        let (dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_with_mut(bs, bs);
        let (lhs_top, lhs_bot) = lhs.split_rows_with(bs);
        let (rhs_left, rhs_right) = rhs.split_cols_with(bs);

        help!(C);
        let beta = sync_opt!(beta);
        let alpha = sync!(alpha);
        join_raw(
            |par| {
                super::matmul_with_conj(
                    ctx,
                    dst_bot_left,
                    unsync_opt!(beta),
                    lhs_bot,
                    conj_lhs,
                    rhs_left,
                    conj_rhs,
                    unsync!(alpha),
                    par,
                )
            },
            |par| {
                join_raw(
                    |par| {
                        mat_x_mat_into_lower_impl_unchecked(
                            ctx,
                            dst_top_left,
                            unsync_opt!(beta),
                            skip_diag,
                            lhs_top,
                            rhs_left,
                            unsync!(alpha),
                            conj_lhs,
                            conj_rhs,
                            par,
                        )
                    },
                    |par| {
                        mat_x_mat_into_lower_impl_unchecked(
                            ctx,
                            dst_bot_right,
                            unsync_opt!(beta),
                            skip_diag,
                            lhs_bot,
                            rhs_right,
                            unsync!(alpha),
                            conj_lhs,
                            conj_rhs,
                            par,
                        )
                    },
                    par,
                )
            },
            par,
        );
    }
}

#[math]
fn mat_x_lower_into_lower_impl_unchecked<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    beta: Option<C::Of<&T>>,
    skip_diag: bool,
    lhs: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    rhs: MatRefGeneric<'_, C, T, Dim<'N>, Dim<'N>>,
    rhs_diag: DiagonalKind,
    alpha: C::Of<&T>,
    conj_lhs: Conj,
    conj_rhs: Conj,
    par: Parallelism,
) {
    let N = dst.nrows();
    let n = N.unbound();
    debug_assert!(N == dst.nrows());
    debug_assert!(N == dst.ncols());
    debug_assert!(N == lhs.nrows());
    debug_assert!(N == lhs.ncols());
    debug_assert!(N == rhs.nrows());
    debug_assert!(N == rhs.ncols());

    if n <= 16 {
        let op = {
            #[inline(never)]
            || {
                stack_mat_16x16!(ctx, temp_dst, N, dst.row_stride(), dst.col_stride(), C, T);
                stack_mat_16x16!(ctx, temp_rhs, N, rhs.row_stride(), rhs.col_stride(), C, T);

                copy_lower(ctx, temp_rhs.rb_mut(), rhs, rhs_diag);
                super::matmul_with_conj(
                    ctx,
                    temp_dst.rb_mut(),
                    None,
                    lhs,
                    conj_lhs,
                    temp_rhs.rb(),
                    conj_rhs,
                    alpha,
                    par,
                );
                accum_lower(ctx, dst, temp_dst.rb(), skip_diag, beta);
            }
        };
        op();
    } else {
        make_guard!(HEAD);
        make_guard!(TAIL);
        let bs = N.partition(N.checked_idx_inc(N.unbound() / 2), HEAD, TAIL);

        let (mut dst_top_left, _, mut dst_bot_left, dst_bot_right) = dst.split_with_mut(bs, bs);
        let (lhs_top_left, lhs_top_right, lhs_bot_left, lhs_bot_right) = lhs.split_with(bs, bs);
        let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_with(bs, bs);

        // lhs_bot_right × rhs_bot_left  => dst_bot_left  | mat × mat => mat |   1
        // lhs_bot_right × rhs_bot_right => dst_bot_right | mat × low => low |   X
        //
        // lhs_top_left  × rhs_top_left  => dst_top_left  | mat × low => low |   X
        // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
        // lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2
        help!(C);

        super::matmul_with_conj(
            ctx,
            dst_bot_left.rb_mut(),
            copy_opt!(beta),
            lhs_bot_right,
            conj_lhs,
            rhs_bot_left,
            conj_rhs,
            copy!(alpha),
            par,
        );
        mat_x_lower_into_lower_impl_unchecked(
            ctx,
            dst_bot_right,
            copy_opt!(beta),
            skip_diag,
            lhs_bot_right,
            rhs_bot_right,
            rhs_diag,
            copy!(alpha),
            conj_lhs,
            conj_rhs,
            par,
        );

        mat_x_lower_into_lower_impl_unchecked(
            ctx,
            dst_top_left.rb_mut(),
            copy_opt!(beta),
            skip_diag,
            lhs_top_left,
            rhs_top_left,
            rhs_diag,
            copy!(alpha),
            conj_lhs,
            conj_rhs,
            par,
        );
        mat_x_mat_into_lower_impl_unchecked(
            ctx,
            dst_top_left,
            Some(as_ref!(math.one())),
            skip_diag,
            lhs_top_right,
            rhs_bot_left,
            copy!(alpha),
            conj_lhs,
            conj_rhs,
            par,
        );
        mat_x_lower_impl_unchecked(
            ctx,
            dst_bot_left,
            Some(as_ref!(math.one())),
            lhs_bot_left,
            rhs_top_left,
            rhs_diag,
            copy!(alpha),
            conj_lhs,
            conj_rhs,
            par,
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

#[track_caller]
fn precondition<M: Shape, N: Shape, K: Shape>(
    dst_nrows: M,
    dst_ncols: N,
    dst_structure: BlockStructure,
    lhs_nrows: M,
    lhs_ncols: K,
    lhs_structure: BlockStructure,
    rhs_nrows: K,
    rhs_ncols: N,
    rhs_structure: BlockStructure,
) {
    assert!(all(
        dst_nrows == lhs_nrows,
        dst_ncols == rhs_ncols,
        lhs_ncols == rhs_nrows,
    ));

    let dst_nrows = dst_nrows.unbound();
    let dst_ncols = dst_ncols.unbound();
    let lhs_nrows = lhs_nrows.unbound();
    let lhs_ncols = lhs_ncols.unbound();
    let rhs_nrows = rhs_nrows.unbound();
    let rhs_ncols = rhs_ncols.unbound();

    if !dst_structure.is_dense() {
        assert!(dst_nrows == dst_ncols);
    }
    if !lhs_structure.is_dense() {
        assert!(lhs_nrows == lhs_ncols);
    }
    if !rhs_structure.is_dense() {
        assert!(rhs_nrows == rhs_ncols);
    }
}

#[track_caller]
#[inline]
pub fn matmul_with_conj<C: ComplexContainer, T: ComplexField<C>, M: Shape, N: Shape, K: Shape>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, M, N, impl Stride, impl Stride>,
    dst_structure: BlockStructure,
    beta: Option<C::Of<&T>>,
    lhs: MatRefGeneric<'_, C, T, M, K, impl Stride, impl Stride>,
    lhs_structure: BlockStructure,
    conj_lhs: Conj,
    rhs: MatRefGeneric<'_, C, T, K, N, impl Stride, impl Stride>,
    rhs_structure: BlockStructure,
    conj_rhs: Conj,
    alpha: C::Of<&T>,
    par: Parallelism,
) {
    precondition(
        dst.nrows(),
        dst.ncols(),
        dst_structure,
        lhs.nrows(),
        lhs.ncols(),
        lhs_structure,
        rhs.nrows(),
        rhs.ncols(),
        rhs_structure,
    );

    make_guard!(M);
    make_guard!(N);
    make_guard!(K);
    let M = dst.nrows().bind(M);
    let N = dst.ncols().bind(N);
    let K = lhs.ncols().bind(K);

    matmul_imp(
        ctx,
        dst.as_dyn_stride_mut().as_shape_mut(M, N),
        dst_structure,
        beta,
        lhs.as_dyn_stride().canonical().as_shape(M, K),
        lhs_structure,
        conj_lhs,
        rhs.as_dyn_stride().canonical().as_shape(K, N),
        rhs_structure,
        conj_rhs,
        alpha,
        par,
    );
}

#[track_caller]
#[inline]
pub fn matmul<
    C: ComplexContainer,
    LhsC: Container<Canonical = C>,
    RhsC: Container<Canonical = C>,
    T: ComplexField<C>,
    LhsT: ConjUnit<Canonical = T>,
    RhsT: ConjUnit<Canonical = T>,
    M: Shape,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, M, N, impl Stride, impl Stride>,
    dst_structure: BlockStructure,
    beta: Option<C::Of<&T>>,
    lhs: MatRefGeneric<'_, LhsC, LhsT, M, K, impl Stride, impl Stride>,
    lhs_structure: BlockStructure,
    rhs: MatRefGeneric<'_, RhsC, RhsT, K, N, impl Stride, impl Stride>,
    rhs_structure: BlockStructure,
    alpha: C::Of<&T>,
    par: Parallelism,
) {
    precondition(
        dst.nrows(),
        dst.ncols(),
        dst_structure,
        lhs.nrows(),
        lhs.ncols(),
        lhs_structure,
        rhs.nrows(),
        rhs.ncols(),
        rhs_structure,
    );

    make_guard!(M);
    make_guard!(N);
    make_guard!(K);
    let M = dst.nrows().bind(M);
    let N = dst.ncols().bind(N);
    let K = lhs.ncols().bind(K);

    matmul_imp(
        ctx,
        dst.as_dyn_stride_mut().as_shape_mut(M, N),
        dst_structure,
        beta,
        lhs.as_dyn_stride().canonical().as_shape(M, K),
        lhs_structure,
        const { Conj::get::<LhsC, LhsT>() },
        rhs.as_dyn_stride().canonical().as_shape(K, N),
        rhs_structure,
        const { Conj::get::<RhsC, RhsT>() },
        alpha,
        par,
    );
}

#[math]
fn matmul_imp<'M, 'N, 'K, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMutGeneric<'_, C, T, Dim<'M>, Dim<'N>>,
    dst_structure: BlockStructure,
    beta: Option<C::Of<&T>>,
    lhs: MatRefGeneric<'_, C, T, Dim<'M>, Dim<'K>>,
    lhs_structure: BlockStructure,
    conj_lhs: Conj,
    rhs: MatRefGeneric<'_, C, T, Dim<'K>, Dim<'N>>,
    rhs_structure: BlockStructure,
    conj_rhs: Conj,
    alpha: C::Of<&T>,
    par: Parallelism,
) {
    let mut acc = dst.as_dyn_mut();
    let mut lhs = lhs.as_dyn();
    let mut rhs = rhs.as_dyn();

    let mut acc_structure = dst_structure;
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

    help!(C);

    make_guard!(M);
    make_guard!(N);
    make_guard!(K);
    let M = acc.nrows().bind(M);
    let N = acc.ncols().bind(N);
    let K = lhs.ncols().bind(K);

    let clear_upper = |acc: MatMutGeneric<'_, C, T>, skip_diag: bool| match &beta {
        Some(beta) => zipped!(acc).for_each_triangular_upper(
            if skip_diag { Diag::Skip } else { Diag::Include },
            |unzipped!(mut acc)| {
                let tmp = math(acc * *beta);
                write1!(acc, tmp);
            },
        ),

        None => zipped!(acc).for_each_triangular_upper(
            if skip_diag { Diag::Skip } else { Diag::Include },
            |unzipped!(mut acc)| {
                let tmp = math(zero());
                write1!(acc, tmp);
            },
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
            super::matmul_with_conj(ctx, acc, beta, lhs, conj_lhs, rhs, conj_rhs, alpha, par);
        } else {
            debug_assert!(rhs_structure.is_lower());

            if lhs_structure.is_dense() {
                mat_x_lower_impl_unchecked(
                    ctx,
                    acc.as_shape_mut(M, N),
                    beta,
                    lhs.as_shape(M, N),
                    rhs.as_shape(N, N),
                    rhs_diag,
                    alpha,
                    conj_lhs,
                    conj_rhs,
                    par,
                )
            } else if lhs_structure.is_lower() {
                clear_upper(acc.rb_mut(), true);
                lower_x_lower_into_lower_impl_unchecked(
                    ctx,
                    acc.as_shape_mut(N, N),
                    beta,
                    false,
                    lhs.as_shape(N, N),
                    lhs_diag,
                    rhs.as_shape(N, N),
                    rhs_diag,
                    alpha,
                    conj_lhs,
                    conj_rhs,
                    par,
                );
            } else {
                debug_assert!(lhs_structure.is_upper());
                upper_x_lower_impl_unchecked(
                    ctx,
                    acc.as_shape_mut(N, N),
                    beta,
                    lhs.as_shape(N, N),
                    lhs_diag,
                    rhs.as_shape(N, N),
                    rhs_diag,
                    alpha,
                    conj_lhs,
                    conj_rhs,
                    par,
                )
            }
        }
    } else if acc_structure.is_lower() {
        if lhs_structure.is_dense() && rhs_structure.is_dense() {
            mat_x_mat_into_lower_impl_unchecked(
                ctx,
                acc.as_shape_mut(N, N),
                beta,
                skip_diag,
                lhs.as_shape(N, K),
                rhs.as_shape(K, N),
                alpha,
                conj_lhs,
                conj_rhs,
                par,
            )
        } else {
            debug_assert!(rhs_structure.is_lower());
            if lhs_structure.is_dense() {
                mat_x_lower_into_lower_impl_unchecked(
                    ctx,
                    acc.as_shape_mut(N, N),
                    beta,
                    skip_diag,
                    lhs.as_shape(N, N),
                    rhs.as_shape(N, N),
                    rhs_diag,
                    alpha,
                    conj_lhs,
                    conj_rhs,
                    par,
                );
            } else if lhs_structure.is_lower() {
                lower_x_lower_into_lower_impl_unchecked(
                    ctx,
                    acc.as_shape_mut(N, N),
                    beta,
                    skip_diag,
                    lhs.as_shape(N, N),
                    lhs_diag,
                    rhs.as_shape(N, N),
                    rhs_diag,
                    alpha,
                    conj_lhs,
                    conj_rhs,
                    par,
                )
            } else {
                upper_x_lower_into_lower_impl_unchecked(
                    ctx,
                    acc.as_shape_mut(N, N),
                    beta,
                    skip_diag,
                    lhs.as_shape(N, N),
                    lhs_diag,
                    rhs.as_shape(N, N),
                    rhs_diag,
                    alpha,
                    conj_lhs,
                    conj_rhs,
                    par,
                )
            }
        }
    } else if lhs_structure.is_dense() && rhs_structure.is_dense() {
        mat_x_mat_into_lower_impl_unchecked(
            ctx,
            acc.as_shape_mut(N, N).transpose_mut(),
            beta,
            skip_diag,
            rhs.transpose().as_shape(N, K),
            lhs.transpose().as_shape(K, N),
            alpha,
            conj_rhs,
            conj_lhs,
            par,
        )
    } else {
        debug_assert!(rhs_structure.is_lower());
        if lhs_structure.is_dense() {
            // lower part of lhs does not contribute to result
            upper_x_lower_into_lower_impl_unchecked(
                ctx,
                acc.as_shape_mut(N, N).transpose_mut(),
                beta,
                skip_diag,
                rhs.transpose().as_shape(N, N),
                rhs_diag,
                lhs.transpose().as_shape(N, N),
                lhs_diag,
                alpha,
                conj_rhs,
                conj_lhs,
                par,
            )
        } else if lhs_structure.is_lower() {
            if !skip_diag {
                match &beta {
                    Some(beta) => {
                        for j in 0..N.unbound() {
                            let tmp = math(beta * acc[(j, j)] + alpha * lhs[(j, j)] * rhs[(j, j)]);
                            write1!(acc.write(j, j), tmp);
                        }
                    }
                    None => {
                        for j in 0..N.unbound() {
                            let tmp = math(alpha * lhs[(j, j)] * rhs[(j, j)]);
                            write1!(acc.write(j, j), tmp);
                        }
                    }
                }
            }
            clear_upper(acc.rb_mut(), true);
        } else {
            debug_assert!(lhs_structure.is_upper());
            upper_x_lower_into_lower_impl_unchecked(
                ctx,
                acc.as_shape_mut(N, N).transpose_mut(),
                beta,
                skip_diag,
                rhs.transpose().as_shape(N, N),
                rhs_diag,
                lhs.transpose().as_shape(N, N),
                lhs_diag,
                alpha,
                conj_rhs,
                conj_lhs,
                par,
            )
        }
    }
}