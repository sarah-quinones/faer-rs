//! Triangular solve module.

use crate::{assert, debug_assert, internal_prelude::*, utils::thread::join_raw};
use faer_macros::math;
use faer_traits::{help, ComplexContainer, ConjUnit, Container, Ctx, SimdArch};
use generativity::make_guard;
use reborrow::*;

#[inline(always)]
#[math]
fn identity<C: ComplexContainer, T: ComplexField<C>>(ctx: &Ctx<C, T>, x: C::Of<T>) -> C::Of<T> {
    math.copy(x)
}

#[inline(always)]
#[math]
fn conj<C: ComplexContainer, T: ComplexField<C>>(ctx: &Ctx<C, T>, x: C::Of<T>) -> C::Of<T> {
    math.conj(x)
}

#[inline(always)]
#[math]
fn solve_unit_lower_triangular_in_place_base_case_generic_unchecked<
    'N,
    'K,
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    ctx: &Ctx<C, T>,
    tril: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
    rhs: MatMut<'_, C, T, Dim<'N>, Dim<'K>>,
    maybe_conj_lhs: impl Fn(&Ctx<C, T>, C::Of<T>) -> C::Of<T>,
) {
    let N = tril.nrows();
    let n = N.unbound();

    help!(C);
    match n {
        0 | 1 => (),
        2 => {
            let i0 = N.check(0);
            let i1 = N.check(1);

            let nl10_div_l11 = maybe_conj_lhs(ctx, math(-tril[(i1, i0)]));

            let (x0, rhs) = rhs.split_first_row_mut().unwrap();
            let (x1, rhs) = rhs.split_first_row_mut().unwrap();
            _ = rhs;

            zipped!(x0, x1).for_each(|unzipped!(x0, mut x1)| {
                math({ write1!(x1, x1 + nl10_div_l11 * x0) });
            });
        }
        3 => {
            let i0 = N.check(0);
            let i1 = N.check(1);
            let i2 = N.check(2);

            let nl10_div_l11 = maybe_conj_lhs(ctx, math(-tril[(i1, i0)]));
            let nl20_div_l22 = maybe_conj_lhs(ctx, math(-tril[(i2, i0)]));
            let nl21_div_l22 = maybe_conj_lhs(ctx, math(-tril[(i2, i1)]));

            let (x0, rhs) = rhs.split_first_row_mut().unwrap();
            let (x1, rhs) = rhs.split_first_row_mut().unwrap();
            let (x2, rhs) = rhs.split_first_row_mut().unwrap();
            _ = rhs;

            zipped!(x0, x1, x2).for_each(|unzipped!(mut x0, mut x1, mut x2)| {
                math({
                    let y0 = copy(x0);
                    let mut y1 = copy(x1);
                    let mut y2 = copy(x2);
                    y1 = y1 + nl10_div_l11 * y0;
                    y2 = y2 + nl20_div_l22 * y0 + nl21_div_l22 * y1;
                    write1!(x0, y0);
                    write1!(x1, y1);
                    write1!(x2, y2);
                })
            });
        }
        4 => {
            let i0 = N.check(0);
            let i1 = N.check(1);
            let i2 = N.check(2);
            let i3 = N.check(3);
            let nl10_div_l11 = maybe_conj_lhs(ctx, math(-tril[(i1, i0)]));
            let nl20_div_l22 = maybe_conj_lhs(ctx, math(-tril[(i2, i0)]));
            let nl21_div_l22 = maybe_conj_lhs(ctx, math(-tril[(i2, i1)]));
            let nl30_div_l33 = maybe_conj_lhs(ctx, math(-tril[(i3, i0)]));
            let nl31_div_l33 = maybe_conj_lhs(ctx, math(-tril[(i3, i1)]));
            let nl32_div_l33 = maybe_conj_lhs(ctx, math(-tril[(i3, i2)]));

            let (x0, rhs) = rhs.split_first_row_mut().unwrap();
            let (x1, rhs) = rhs.split_first_row_mut().unwrap();
            let (x2, rhs) = rhs.split_first_row_mut().unwrap();
            let (x3, rhs) = rhs.split_first_row_mut().unwrap();
            _ = rhs;

            zipped!(x0, x1, x2, x3).for_each(|unzipped!(mut x0, mut x1, mut x2, mut x3)| {
                math({
                    let y0 = copy(x0);
                    let mut y1 = copy(x1);
                    let mut y2 = copy(x2);
                    let mut y3 = copy(x3);
                    y1 = y1 + nl10_div_l11 * y0;
                    y2 = y2 + nl20_div_l22 * y0 + nl21_div_l22 * y1;
                    y3 = y3 + nl30_div_l33 * y0 + nl31_div_l33 * y1 + nl32_div_l33 * y2;
                    write1!(x0, y0);
                    write1!(x1, y1);
                    write1!(x2, y2);
                    write1!(x3, y3);
                })
            });
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
#[math]
fn solve_lower_triangular_in_place_base_case_generic_unchecked<
    'N,
    'K,
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    ctx: &Ctx<C, T>,
    tril: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
    rhs: MatMut<'_, C, T, Dim<'N>, Dim<'K>>,
    maybe_conj_lhs: impl Fn(&Ctx<C, T>, C::Of<T>) -> C::Of<T>,
) {
    let N = tril.nrows();
    let n = N.unbound();

    help!(C);
    match n {
        0 => (),
        1 => {
            let i0 = N.check(0);

            let inv = maybe_conj_lhs(ctx, math(recip(tril[(i0, i0)])));

            let (x0, rhs) = rhs.split_first_row_mut().unwrap();
            _ = rhs;

            zipped!(x0).for_each(|unzipped!(mut x0)| math(write1!(x0, x0 * inv)));
        }
        2 => {
            let i0 = N.check(0);
            let i1 = N.check(1);

            let l00_inv = maybe_conj_lhs(ctx, math(recip(tril[(i0, i0)])));
            let l11_inv = maybe_conj_lhs(ctx, math(recip(tril[(i1, i1)])));
            let nl10_div_l11 = maybe_conj_lhs(ctx, math(-tril[(i1, i0)] * l11_inv));

            let (x0, rhs) = rhs.split_first_row_mut().unwrap();
            let (x1, rhs) = rhs.split_first_row_mut().unwrap();
            _ = rhs;

            zipped!(x0, x1).for_each(|unzipped!(mut x0, mut x1)| {
                math({
                    write1!(x0, x0 * l00_inv);
                    write1!(x1, x1 * l11_inv + nl10_div_l11 * x0);
                })
            });
        }
        3 => {
            let i0 = N.check(0);
            let i1 = N.check(1);
            let i2 = N.check(2);

            let l00_inv = maybe_conj_lhs(ctx, math(recip(tril[(i0, i0)])));
            let l11_inv = maybe_conj_lhs(ctx, math(recip(tril[(i1, i1)])));
            let l22_inv = maybe_conj_lhs(ctx, math(recip(tril[(i2, i2)])));
            let nl10_div_l11 = maybe_conj_lhs(ctx, math(-tril[(i1, i0)] * l11_inv));
            let nl20_div_l22 = maybe_conj_lhs(ctx, math(-tril[(i2, i0)] * l22_inv));
            let nl21_div_l22 = maybe_conj_lhs(ctx, math(-tril[(i2, i1)] * l22_inv));

            let (x0, rhs) = rhs.split_first_row_mut().unwrap();
            let (x1, rhs) = rhs.split_first_row_mut().unwrap();
            let (x2, rhs) = rhs.split_first_row_mut().unwrap();
            _ = rhs;

            zipped!(x0, x1, x2).for_each(|unzipped!(mut x0, mut x1, mut x2)| {
                math({
                    let mut y0 = copy(x0);
                    let mut y1 = copy(x1);
                    let mut y2 = copy(x2);
                    y0 = y0 * l00_inv;
                    y1 = y1 * l11_inv + nl10_div_l11 * y0;
                    y2 = y2 * l22_inv + nl20_div_l22 * y0 + nl21_div_l22 * y1;
                    write1!(x0, y0);
                    write1!(x1, y1);
                    write1!(x2, y2);
                })
            });
        }
        4 => {
            let i0 = N.check(0);
            let i1 = N.check(1);
            let i2 = N.check(2);
            let i3 = N.check(3);

            let l00_inv = maybe_conj_lhs(ctx, math(recip(tril[(i0, i0)])));
            let l11_inv = maybe_conj_lhs(ctx, math(recip(tril[(i1, i1)])));
            let l22_inv = maybe_conj_lhs(ctx, math(recip(tril[(i2, i2)])));
            let l33_inv = maybe_conj_lhs(ctx, math(recip(tril[(i3, i3)])));
            let nl10_div_l11 = maybe_conj_lhs(ctx, math(-tril[(i1, i0)] * l11_inv));
            let nl20_div_l22 = maybe_conj_lhs(ctx, math(-tril[(i2, i0)] * l22_inv));
            let nl21_div_l22 = maybe_conj_lhs(ctx, math(-tril[(i2, i1)] * l22_inv));
            let nl30_div_l33 = maybe_conj_lhs(ctx, math(-tril[(i3, i0)] * l33_inv));
            let nl31_div_l33 = maybe_conj_lhs(ctx, math(-tril[(i3, i1)] * l33_inv));
            let nl32_div_l33 = maybe_conj_lhs(ctx, math(-tril[(i3, i2)] * l33_inv));

            let (x0, rhs) = rhs.split_first_row_mut().unwrap();
            let (x1, rhs) = rhs.split_first_row_mut().unwrap();
            let (x2, rhs) = rhs.split_first_row_mut().unwrap();
            let (x3, rhs) = rhs.split_first_row_mut().unwrap();
            _ = rhs;

            zipped!(x0, x1, x2, x3).for_each(|unzipped!(mut x0, mut x1, mut x2, mut x3)| {
                math({
                    let mut y0 = copy(x0);
                    let mut y1 = copy(x1);
                    let mut y2 = copy(x2);
                    let mut y3 = copy(x3);
                    y0 = y0 * l00_inv;
                    y1 = y1 * l11_inv + nl10_div_l11 * y0;
                    y2 = y2 * l22_inv + nl20_div_l22 * y0 + nl21_div_l22 * y1;
                    y3 = y3 * l33_inv + nl30_div_l33 * y0 + nl31_div_l33 * y1 + nl32_div_l33 * y2;
                    write1!(x0, y0);
                    write1!(x1, y1);
                    write1!(x2, y2);
                    write1!(x3, y3);
                })
            });
        }
        _ => unreachable!(),
    }
}

#[inline]
fn blocksize(n: usize) -> usize {
    // we want remainder to be a multiple of register size
    let base_rem = n / 2;
    n - if n >= 32 {
        (base_rem + 15) / 16 * 16
    } else if n >= 16 {
        (base_rem + 7) / 8 * 8
    } else if n >= 8 {
        (base_rem + 3) / 4 * 4
    } else {
        base_rem
    }
}

#[inline]
fn recursion_threshold() -> usize {
    4
}

#[track_caller]
#[inline]
pub fn solve_lower_triangular_in_place_with_conj<
    C: ComplexContainer,
    T: ComplexField<C>,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    triangular_lower: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    conj_lhs: Conj,
    rhs: MatMut<'_, C, T, N, K, impl Stride, impl Stride>,
    par: Par,
) {
    assert!(all(
        triangular_lower.nrows() == triangular_lower.ncols(),
        rhs.nrows() == triangular_lower.ncols(),
    ));

    make_guard!(N);
    make_guard!(K);
    let N = rhs.nrows().bind(N);
    let K = rhs.ncols().bind(K);

    solve_lower_triangular_in_place_unchecked(
        ctx,
        triangular_lower.as_dyn_stride().as_shape(N, N),
        conj_lhs,
        rhs.as_dyn_stride_mut().as_shape_mut(N, K),
        par,
    );
}

#[inline]
#[track_caller]
pub fn solve_lower_triangular_in_place<
    C: ComplexContainer,
    LhsC: Container<Canonical = C>,
    T: ComplexField<C>,
    LhsT: ConjUnit<Canonical = T>,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    triangular_lower: MatRef<'_, LhsC, LhsT, N, N, impl Stride, impl Stride>,
    rhs: MatMut<'_, C, T, N, K, impl Stride, impl Stride>,
    par: Par,
) {
    let tri = triangular_lower.canonical();
    solve_lower_triangular_in_place_with_conj(ctx, tri, Conj::get::<LhsC, LhsT>(), rhs, par)
}

#[track_caller]
#[inline]
pub fn solve_unit_lower_triangular_in_place_with_conj<
    C: ComplexContainer,
    T: ComplexField<C>,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    triangular_unit_lower: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    conj_lhs: Conj,
    rhs: MatMut<'_, C, T, N, K, impl Stride, impl Stride>,
    par: Par,
) {
    assert!(all(
        triangular_unit_lower.nrows() == triangular_unit_lower.ncols(),
        rhs.nrows() == triangular_unit_lower.ncols(),
    ));

    make_guard!(N);
    make_guard!(K);
    let N = rhs.nrows().bind(N);
    let K = rhs.ncols().bind(K);

    solve_unit_lower_triangular_in_place_unchecked(
        ctx,
        triangular_unit_lower.as_dyn_stride().as_shape(N, N),
        conj_lhs,
        rhs.as_dyn_stride_mut().as_shape_mut(N, K),
        par,
    );
}

#[inline]
#[track_caller]
pub fn solve_unit_lower_triangular_in_place<
    C: ComplexContainer,
    LhsC: Container<Canonical = C>,
    T: ComplexField<C>,
    LhsT: ConjUnit<Canonical = T>,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    triangular_unit_lower: MatRef<'_, LhsC, LhsT, N, N, impl Stride, impl Stride>,
    rhs: MatMut<'_, C, T, N, K, impl Stride, impl Stride>,
    par: Par,
) {
    let tri = triangular_unit_lower.canonical();
    solve_unit_lower_triangular_in_place_with_conj(ctx, tri, Conj::get::<LhsC, LhsT>(), rhs, par)
}

#[track_caller]
#[inline]
pub fn solve_upper_triangular_in_place_with_conj<
    C: ComplexContainer,
    T: ComplexField<C>,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    triangular_upper: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    conj_lhs: Conj,
    rhs: MatMut<'_, C, T, N, K, impl Stride, impl Stride>,
    par: Par,
) {
    assert!(all(
        triangular_upper.nrows() == triangular_upper.ncols(),
        rhs.nrows() == triangular_upper.ncols(),
    ));

    make_guard!(N);
    make_guard!(K);
    let N = rhs.nrows().bind(N);
    let K = rhs.ncols().bind(K);

    solve_upper_triangular_in_place_unchecked(
        ctx,
        triangular_upper.as_dyn_stride().as_shape(N, N),
        conj_lhs,
        rhs.as_dyn_stride_mut().as_shape_mut(N, K),
        par,
    );
}

#[inline]
#[track_caller]
pub fn solve_upper_triangular_in_place<
    C: ComplexContainer,
    LhsC: Container<Canonical = C>,
    T: ComplexField<C>,
    LhsT: ConjUnit<Canonical = T>,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    triangular_upper: MatRef<'_, LhsC, LhsT, N, N, impl Stride, impl Stride>,
    rhs: MatMut<'_, C, T, N, K, impl Stride, impl Stride>,
    par: Par,
) {
    let tri = triangular_upper.canonical();
    solve_upper_triangular_in_place_with_conj(ctx, tri, Conj::get::<LhsC, LhsT>(), rhs, par)
}

#[track_caller]
#[inline]
pub fn solve_unit_upper_triangular_in_place_with_conj<
    C: ComplexContainer,
    T: ComplexField<C>,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    triangular_unit_upper: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    conj_lhs: Conj,
    rhs: MatMut<'_, C, T, N, K, impl Stride, impl Stride>,
    par: Par,
) {
    assert!(all(
        triangular_unit_upper.nrows() == triangular_unit_upper.ncols(),
        rhs.nrows() == triangular_unit_upper.ncols(),
    ));

    make_guard!(N);
    make_guard!(K);
    let N = rhs.nrows().bind(N);
    let K = rhs.ncols().bind(K);

    solve_unit_upper_triangular_in_place_unchecked(
        ctx,
        triangular_unit_upper.as_dyn_stride().as_shape(N, N),
        conj_lhs,
        rhs.as_dyn_stride_mut().as_shape_mut(N, K),
        par,
    );
}

#[inline]
#[track_caller]
pub fn solve_unit_upper_triangular_in_place<
    C: ComplexContainer,
    LhsC: Container<Canonical = C>,
    T: ComplexField<C>,
    LhsT: ConjUnit<Canonical = T>,
    N: Shape,
    K: Shape,
>(
    ctx: &Ctx<C, T>,
    triangular_unit_upper: MatRef<'_, LhsC, LhsT, N, N, impl Stride, impl Stride>,
    rhs: MatMut<'_, C, T, N, K, impl Stride, impl Stride>,
    par: Par,
) {
    let tri = triangular_unit_upper.canonical();
    solve_unit_upper_triangular_in_place_with_conj(ctx, tri, Conj::get::<LhsC, LhsT>(), rhs, par)
}

#[math]
fn solve_unit_lower_triangular_in_place_unchecked<
    'N,
    'K,
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    ctx: &Ctx<C, T>,
    tril: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
    conj_lhs: Conj,
    rhs: MatMut<'_, C, T, Dim<'N>, Dim<'K>>,
    par: Par,
) {
    let N = tril.nrows();
    let K = rhs.ncols();
    let n = N.unbound();
    let k = K.unbound();

    if k > 64 && n <= 128 {
        make_guard!(LEFT);
        make_guard!(RIGHT);

        let mid = K.partition(IdxInc::new_checked(k / 2, K), LEFT, RIGHT);

        let (rhs_left, rhs_right) = rhs.split_cols_with_mut(mid);
        join_raw(
            |_| solve_unit_lower_triangular_in_place_unchecked(ctx, tril, conj_lhs, rhs_left, par),
            |_| solve_unit_lower_triangular_in_place_unchecked(ctx, tril, conj_lhs, rhs_right, par),
            par,
        );
        return;
    }

    debug_assert!(all(
        tril.nrows() == tril.ncols(),
        rhs.nrows() == tril.ncols(),
    ));

    if n <= recursion_threshold() {
        T::Arch::default().dispatch(
            #[inline(always)]
            || match conj_lhs {
                Conj::Yes => solve_unit_lower_triangular_in_place_base_case_generic_unchecked(
                    ctx, tril, rhs, conj,
                ),
                Conj::No => solve_unit_lower_triangular_in_place_base_case_generic_unchecked(
                    ctx, tril, rhs, identity,
                ),
            },
        );
        return;
    }

    make_guard!(HEAD);
    make_guard!(TAIL);
    let bs = N.partition(IdxInc::new_checked(blocksize(n), N), HEAD, TAIL);

    let (tril_top_left, _, tril_bot_left, tril_bot_right) = tril.split_with(bs, bs);
    let (mut rhs_top, mut rhs_bot) = rhs.split_rows_with_mut(bs);

    //       (A00    )   X0         (B0)
    // ConjA?(A10 A11)   X1 = ConjB?(B1)
    //
    //
    // 1. ConjA?(A00) X0 = ConjB?(B0)
    //
    // 2. ConjA?(A10) X0 + ConjA?(A11) X1 = ConjB?(B1)
    // => ConjA?(A11) X1 = ConjB?(B1) - ConjA?(A10) X0

    solve_unit_lower_triangular_in_place_unchecked(
        ctx,
        tril_top_left,
        conj_lhs,
        rhs_top.rb_mut(),
        par,
    );

    help!(C);
    crate::linalg::matmul::matmul_with_conj(
        ctx,
        rhs_bot.rb_mut(),
        Accum::Add,
        tril_bot_left,
        conj_lhs,
        rhs_top.into_const(),
        Conj::No,
        math(id(-one())),
        par,
    );

    solve_unit_lower_triangular_in_place_unchecked(ctx, tril_bot_right, conj_lhs, rhs_bot, par);
}

#[math]
fn solve_lower_triangular_in_place_unchecked<'N, 'K, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    tril: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
    conj_lhs: Conj,
    rhs: MatMut<'_, C, T, Dim<'N>, Dim<'K>>,
    par: Par,
) {
    let N = tril.nrows();
    let K = rhs.ncols();
    let n = N.unbound();
    let k = K.unbound();

    if k > 64 && n <= 128 {
        make_guard!(LEFT);
        make_guard!(RIGHT);

        let mid = K.partition(IdxInc::new_checked(k / 2, K), LEFT, RIGHT);

        let (rhs_left, rhs_right) = rhs.split_cols_with_mut(mid);
        join_raw(
            |_| solve_lower_triangular_in_place_unchecked(ctx, tril, conj_lhs, rhs_left, par),
            |_| solve_lower_triangular_in_place_unchecked(ctx, tril, conj_lhs, rhs_right, par),
            par,
        );
        return;
    }

    debug_assert!(all(
        tril.nrows() == tril.ncols(),
        rhs.nrows() == tril.ncols(),
    ));

    if n <= recursion_threshold() {
        T::Arch::default().dispatch(
            #[inline(always)]
            || match conj_lhs {
                Conj::Yes => solve_lower_triangular_in_place_base_case_generic_unchecked(
                    ctx, tril, rhs, conj,
                ),
                Conj::No => solve_lower_triangular_in_place_base_case_generic_unchecked(
                    ctx, tril, rhs, identity,
                ),
            },
        );
        return;
    }

    make_guard!(HEAD);
    make_guard!(TAIL);
    let bs = N.partition(IdxInc::new_checked(blocksize(n), N), HEAD, TAIL);

    let (tril_top_left, _, tril_bot_left, tril_bot_right) = tril.split_with(bs, bs);
    let (mut rhs_top, mut rhs_bot) = rhs.split_rows_with_mut(bs);

    //       (A00    )   X0         (B0)
    // ConjA?(A10 A11)   X1 = ConjB?(B1)
    //
    //
    // 1. ConjA?(A00) X0 = ConjB?(B0)
    //
    // 2. ConjA?(A10) X0 + ConjA?(A11) X1 = ConjB?(B1)
    // => ConjA?(A11) X1 = ConjB?(B1) - ConjA?(A10) X0

    solve_lower_triangular_in_place_unchecked(ctx, tril_top_left, conj_lhs, rhs_top.rb_mut(), par);

    help!(C);
    crate::linalg::matmul::matmul_with_conj(
        ctx,
        rhs_bot.rb_mut(),
        Accum::Add,
        tril_bot_left,
        conj_lhs,
        rhs_top.into_const(),
        Conj::No,
        math(id(-one())),
        par,
    );

    solve_lower_triangular_in_place_unchecked(ctx, tril_bot_right, conj_lhs, rhs_bot, par);
}

#[inline]
fn solve_unit_upper_triangular_in_place_unchecked<
    'N,
    'K,
    C: ComplexContainer,
    T: ComplexField<C>,
>(
    ctx: &Ctx<C, T>,
    triu: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
    conj_lhs: Conj,
    rhs: MatMut<'_, C, T, Dim<'N>, Dim<'K>>,
    par: Par,
) {
    solve_unit_lower_triangular_in_place_unchecked(
        ctx,
        triu.reverse_rows_and_cols(),
        conj_lhs,
        rhs.reverse_rows_mut(),
        par,
    );
}

#[inline]
fn solve_upper_triangular_in_place_unchecked<'N, 'K, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    triu: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
    conj_lhs: Conj,
    rhs: MatMut<'_, C, T, Dim<'N>, Dim<'K>>,
    par: Par,
) {
    solve_lower_triangular_in_place_unchecked(
        ctx,
        triu.reverse_rows_and_cols(),
        conj_lhs,
        rhs.reverse_rows_mut(),
        par,
    );
}
