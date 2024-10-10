use linalg::matmul::triangular::BlockStructure;

use crate::{internal_prelude::*, utils::thread::join_raw};

#[math]
fn invert_lower_triangular_impl_small<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    mut dst: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    src: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
) {
    help!(C);
    let N = dst.nrows();
    math(match *N {
        0 => {}
        1 => {
            let i0 = N.check(0);
            write1!(dst.write(i0, i0), recip(src[(i0, i0)]))
        }
        2 => {
            let i0 = N.check(0);
            let i1 = N.check(1);
            let dst00 = recip(src[(i0, i0)]);
            let dst11 = recip(src[(i1, i1)]);
            let dst10 = -dst11 * src[(i1, i0)] * dst00;

            write1!(dst.write(i0, i0), dst00);
            write1!(dst.write(i1, i1), dst11);
            write1!(dst.write(i1, i0), dst10);
        }
        _ => unreachable!(),
    })
}

#[math]
fn invert_unit_lower_triangular_impl_small<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    mut dst: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    src: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
) {
    let N = dst.nrows();
    match *N {
        0 | 1 => {}
        2 => {
            help!(C);
            let i0 = N.check(0);
            let i1 = N.check(1);
            math(write1!(dst.write(i1, i0), -src[(i1, i0)]));
        }
        _ => unreachable!(),
    }
}

#[math]
fn invert_lower_triangular_impl<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    src: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
    parallelism: Parallelism,
) {
    // m must be equal to n
    let N = dst.ncols();

    if *N <= 2 {
        invert_lower_triangular_impl_small(ctx, dst, src);
        return;
    }

    make_guard!(HEAD);
    make_guard!(TAIL);
    let mid = N.partition(N.checked_idx_inc(*N / 2), HEAD, TAIL);

    let (mut dst_tl, _, mut dst_bl, mut dst_br) = { dst.split_with_mut(mid, mid) };
    let (src_tl, _, src_bl, src_br) = { src.split_with(mid, mid) };

    join_raw(
        |parallelism| invert_lower_triangular_impl(ctx, dst_tl.rb_mut(), src_tl, parallelism),
        |parallelism| invert_lower_triangular_impl(ctx, dst_br.rb_mut(), src_br, parallelism),
        parallelism,
    );

    linalg::matmul::triangular::matmul(
        ctx,
        dst_bl.rb_mut(),
        BlockStructure::Rectangular,
        None,
        src_bl,
        BlockStructure::Rectangular,
        dst_tl.rb(),
        BlockStructure::TriangularLower,
        math(id(-one())),
        parallelism,
    );
    linalg::triangular_solve::solve_lower_triangular_in_place(ctx, src_br, dst_bl, parallelism);
}

#[math]
fn invert_unit_lower_triangular_impl<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    dst: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    src: MatRef<'_, C, T, Dim<'N>, Dim<'N>>,
    parallelism: Parallelism,
) {
    // m must be equal to n
    let N = dst.ncols();

    if *N <= 2 {
        invert_unit_lower_triangular_impl_small(ctx, dst, src);
        return;
    }

    make_guard!(HEAD);
    make_guard!(TAIL);
    let mid = N.partition(N.checked_idx_inc(*N / 2), HEAD, TAIL);

    let (mut dst_tl, _, mut dst_bl, mut dst_br) = { dst.split_with_mut(mid, mid) };
    let (src_tl, _, src_bl, src_br) = { src.split_with(mid, mid) };

    join_raw(
        |parallelism| invert_unit_lower_triangular_impl(ctx, dst_tl.rb_mut(), src_tl, parallelism),
        |parallelism| invert_unit_lower_triangular_impl(ctx, dst_br.rb_mut(), src_br, parallelism),
        parallelism,
    );

    linalg::matmul::triangular::matmul(
        ctx,
        dst_bl.rb_mut(),
        BlockStructure::Rectangular,
        None,
        src_bl,
        BlockStructure::Rectangular,
        dst_tl.rb(),
        BlockStructure::UnitTriangularLower,
        math(id(-one())),
        parallelism,
    );
    linalg::triangular_solve::solve_unit_lower_triangular_in_place(
        ctx,
        src_br,
        dst_bl,
        parallelism,
    );
}

/// Computes the inverse of the lower triangular matrix `src` (with implicit unit
/// diagonal) and stores the strictly lower triangular part of the result to `dst`.
///
/// # Panics
///
/// Panics if `src` and `dst` have mismatching dimensions, or if they are not square.
#[track_caller]
pub fn invert_unit_lower_triangular<C: ComplexContainer, T: ComplexField<C>, N: Shape>(
    ctx: &Ctx<C, T>,
    dst: MatMut<'_, C, T, N, N, impl Stride, impl Stride>,
    src: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    parallelism: Parallelism,
) {
    Assert!(all(
        dst.nrows() == src.nrows(),
        dst.ncols() == src.ncols(),
        dst.nrows() == dst.ncols()
    ));

    with_dim!(N, dst.nrows().unbound());

    invert_unit_lower_triangular_impl(
        ctx,
        dst.as_shape_mut(N, N).as_dyn_stride_mut(),
        src.as_shape(N, N).as_dyn_stride(),
        parallelism,
    )
}

/// Computes the inverse of the lower triangular matrix `src` and stores the
/// lower triangular part of the result to `dst`.
///
/// # Panics
///
/// Panics if `src` and `dst` have mismatching dimensions, or if they are not square.
#[track_caller]
pub fn invert_lower_triangular<C: ComplexContainer, T: ComplexField<C>, N: Shape>(
    ctx: &Ctx<C, T>,
    dst: MatMut<'_, C, T, N, N, impl Stride, impl Stride>,
    src: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    parallelism: Parallelism,
) {
    Assert!(all(
        dst.nrows() == src.nrows(),
        dst.ncols() == src.ncols(),
        dst.nrows() == dst.ncols()
    ));

    with_dim!(N, dst.nrows().unbound());

    invert_lower_triangular_impl(
        ctx,
        dst.as_shape_mut(N, N).as_dyn_stride_mut(),
        src.as_shape(N, N).as_dyn_stride(),
        parallelism,
    )
}

/// Computes the inverse of the upper triangular matrix `src` (with implicit unit
/// diagonal) and stores the strictly upper triangular part of the result to `dst`.
///
/// # Panics
///
/// Panics if `src` and `dst` have mismatching dimensions, or if they are not square.
#[track_caller]
pub fn invert_unit_upper_triangular<C: ComplexContainer, T: ComplexField<C>, N: Shape>(
    ctx: &Ctx<C, T>,
    dst: MatMut<'_, C, T, N, N, impl Stride, impl Stride>,
    src: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    parallelism: Parallelism,
) {
    invert_unit_lower_triangular(
        ctx,
        dst.reverse_rows_and_cols_mut(),
        src.reverse_rows_and_cols(),
        parallelism,
    )
}

/// Computes the inverse of the upper triangular matrix `src` and stores the
/// upper triangular part of the result to `dst`.
///
/// # Panics
///
/// Panics if `src` and `dst` have mismatching dimensions, or if they are not square.
#[track_caller]
pub fn invert_upper_triangular<C: ComplexContainer, T: ComplexField<C>, N: Shape>(
    ctx: &Ctx<C, T>,
    dst: MatMut<'_, C, T, N, N, impl Stride, impl Stride>,
    src: MatRef<'_, C, T, N, N, impl Stride, impl Stride>,
    parallelism: Parallelism,
) {
    invert_lower_triangular(
        ctx,
        dst.reverse_rows_and_cols_mut(),
        src.reverse_rows_and_cols(),
        parallelism,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, Mat, MatRef};
    use assert_approx_eq::assert_approx_eq;
    use linalg::matmul::triangular;
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, StandardNormal};

    #[test]
    fn test_invert_lower() {
        let rng = &mut StdRng::seed_from_u64(0);
        (0..32).for_each(|n| {
            let mut a: Mat<f64> = crate::stats::CwiseMatDistribution {
                nrows: n,
                ncols: n,
                dist: StandardNormal,
            }
            .sample(rng);
            a += MatRef::from_repeated_ref(&2.0, n, n);
            let ctx = &default();
            let mut inv = Mat::zeros_with_ctx(ctx, n, n);

            invert_lower_triangular(ctx, inv.as_mut(), a.as_ref(), Parallelism::rayon(0));

            let mut prod = Mat::zeros_with_ctx(ctx, n, n);
            triangular::matmul(
                ctx,
                prod.as_mut(),
                BlockStructure::Rectangular,
                None,
                a.as_ref(),
                BlockStructure::TriangularLower,
                inv.as_ref(),
                BlockStructure::TriangularLower,
                &1.0,
                Parallelism::rayon(0),
            );

            for i in 0..n {
                for j in 0..n {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(prod[(i, j)], target, 1e-4);
                }
            }
        });
    }

    #[test]
    fn test_invert_unit_lower() {
        let rng = &mut StdRng::seed_from_u64(0);
        (0..32).for_each(|n| {
            let mut a: Mat<f64> = crate::stats::CwiseMatDistribution {
                nrows: n,
                ncols: n,
                dist: StandardNormal,
            }
            .sample(rng);
            a += MatRef::from_repeated_ref(&2.0, n, n);
            let ctx = &Default::default();
            let mut inv = Mat::zeros_with_ctx(ctx, n, n);

            invert_unit_lower_triangular(ctx, inv.as_mut(), a.as_ref(), Parallelism::rayon(0));

            let mut prod = Mat::zeros_with_ctx(ctx, n, n);
            triangular::matmul(
                ctx,
                prod.as_mut(),
                BlockStructure::Rectangular,
                None,
                a.as_ref(),
                BlockStructure::UnitTriangularLower,
                inv.as_ref(),
                BlockStructure::UnitTriangularLower,
                &1.0,
                Parallelism::rayon(0),
            );

            for i in 0..n {
                for j in 0..n {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(prod[(i, j)], target, 1e-4);
                }
            }
        });
    }

    #[test]
    fn test_invert_upper() {
        let rng = &mut StdRng::seed_from_u64(0);
        (0..32).for_each(|n| {
            let mut a: Mat<f64> = crate::stats::CwiseMatDistribution {
                nrows: n,
                ncols: n,
                dist: StandardNormal,
            }
            .sample(rng);
            a += MatRef::from_repeated_ref(&2.0, n, n);
            let ctx = &Default::default();
            let mut inv = Mat::zeros_with_ctx(ctx, n, n);

            invert_upper_triangular(ctx, inv.as_mut(), a.as_ref(), Parallelism::rayon(0));

            let mut prod = Mat::zeros_with_ctx(ctx, n, n);
            triangular::matmul(
                ctx,
                prod.as_mut(),
                BlockStructure::Rectangular,
                None,
                a.as_ref(),
                BlockStructure::TriangularUpper,
                inv.as_ref(),
                BlockStructure::TriangularUpper,
                &1.0,
                Parallelism::rayon(0),
            );

            for i in 0..n {
                for j in 0..n {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(prod[(i, j)], target, 1e-4);
                }
            }
        });
    }

    #[test]
    fn test_invert_unit_upper() {
        let rng = &mut StdRng::seed_from_u64(0);
        (0..32).for_each(|n| {
            let mut a: Mat<f64> = crate::stats::CwiseMatDistribution {
                nrows: n,
                ncols: n,
                dist: StandardNormal,
            }
            .sample(rng);
            a += MatRef::from_repeated_ref(&2.0, n, n);

            let ctx = &Default::default();
            let mut inv = Mat::zeros_with_ctx(ctx, n, n);

            invert_unit_upper_triangular(ctx, inv.as_mut(), a.as_ref(), Parallelism::rayon(0));

            let mut prod = Mat::zeros_with_ctx(ctx, n, n);
            triangular::matmul(
                ctx,
                prod.as_mut(),
                BlockStructure::Rectangular,
                None,
                a.as_ref(),
                BlockStructure::UnitTriangularUpper,
                inv.as_ref(),
                BlockStructure::UnitTriangularUpper,
                &1.0,
                Parallelism::rayon(0),
            );

            for i in 0..n {
                for j in 0..n {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(prod[(i, j)], target, 1e-4);
                }
            }
        });
    }
}
