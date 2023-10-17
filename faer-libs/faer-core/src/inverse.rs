//! Triangular matrix inversion.

use crate::{
    join_raw,
    mul::triangular::{self, BlockStructure},
    solve, ComplexField, MatMut, MatRef, Parallelism,
};
#[cfg(feature = "std")]
use assert2::assert;
use reborrow::*;

unsafe fn invert_lower_triangular_impl_small<E: ComplexField>(
    mut dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
) {
    let m = dst.nrows();
    let src = {
        #[inline(always)]
        |i: usize, j: usize| src.read_unchecked(i, j)
    };
    match m {
        0 => {}
        1 => dst.write_unchecked(0, 0, src(0, 0).faer_inv()),
        2 => {
            let dst00 = src(0, 0).faer_inv();
            let dst11 = src(1, 1).faer_inv();
            let dst10 = (dst11.faer_mul(src(1, 0)).faer_mul(dst00)).faer_neg();

            dst.write_unchecked(0, 0, dst00);
            dst.write_unchecked(1, 1, dst11);
            dst.write_unchecked(1, 0, dst10);
        }
        _ => unreachable!(),
    }
}

unsafe fn invert_unit_lower_triangular_impl_small<E: ComplexField>(
    mut dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
) {
    let m = dst.nrows();
    let src = |i: usize, j: usize| src.read_unchecked(i, j);
    match m {
        0 | 1 => {}
        2 => {
            dst.write_unchecked(1, 0, src(1, 0).faer_neg());
        }
        _ => unreachable!(),
    }
}

unsafe fn invert_lower_triangular_impl<E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    parallelism: Parallelism,
) {
    // m must be equal to n
    let m = dst.nrows();
    let n = dst.ncols();

    if m <= 2 {
        invert_lower_triangular_impl_small(dst, src);
        return;
    }

    let [mut dst_tl, _, mut dst_bl, mut dst_br] = { dst.split_at(m / 2, n / 2) };

    let m = src.nrows();
    let n = src.ncols();
    let [src_tl, _, src_bl, src_br] = { src.split_at(m / 2, n / 2) };

    join_raw(
        |parallelism| invert_lower_triangular_impl(dst_tl.rb_mut(), src_tl, parallelism),
        |parallelism| invert_lower_triangular_impl(dst_br.rb_mut(), src_br, parallelism),
        parallelism,
    );

    triangular::matmul(
        dst_bl.rb_mut(),
        BlockStructure::Rectangular,
        src_bl,
        BlockStructure::Rectangular,
        dst_tl.rb(),
        BlockStructure::TriangularLower,
        None,
        E::faer_one().faer_neg(),
        parallelism,
    );
    solve::solve_lower_triangular_in_place(src_br, dst_bl, parallelism);
}

unsafe fn invert_unit_lower_triangular_impl<E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    parallelism: Parallelism,
) {
    // m must be equal to n
    let m = dst.nrows();
    let n = dst.ncols();

    if m <= 2 {
        invert_unit_lower_triangular_impl_small(dst, src);
        return;
    }

    let [mut dst_tl, _, mut dst_bl, mut dst_br] = { dst.split_at(m / 2, n / 2) };

    let m = src.nrows();
    let n = src.ncols();
    let [src_tl, _, src_bl, src_br] = { src.split_at(m / 2, n / 2) };

    join_raw(
        |parallelism| invert_unit_lower_triangular_impl(dst_tl.rb_mut(), src_tl, parallelism),
        |parallelism| invert_unit_lower_triangular_impl(dst_br.rb_mut(), src_br, parallelism),
        parallelism,
    );

    triangular::matmul(
        dst_bl.rb_mut(),
        BlockStructure::Rectangular,
        src_bl,
        BlockStructure::Rectangular,
        dst_tl.rb(),
        BlockStructure::UnitTriangularLower,
        None,
        E::faer_one().faer_neg(),
        parallelism,
    );
    solve::solve_unit_lower_triangular_in_place(src_br, dst_bl, parallelism);
}

/// Computes the inverse of the lower triangular matrix `src` (with implicit unit
/// diagonal) and stores the strictly lower triangular part of the result to `dst`.
///
/// # Panics
///
/// Panics if `src` and `dst` have mismatching dimensions, or if they are not square.
#[track_caller]
pub fn invert_unit_lower_triangular<E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    parallelism: Parallelism,
) {
    assert!(dst.nrows() == src.nrows());
    assert!(dst.ncols() == src.ncols());
    assert!(dst.nrows() == dst.ncols());

    unsafe { invert_unit_lower_triangular_impl(dst, src, parallelism) }
}

/// Computes the inverse of the lower triangular matrix `src` and stores the
/// lower triangular part of the result to `dst`.
///
/// # Panics
///
/// Panics if `src` and `dst` have mismatching dimensions, or if they are not square.
#[track_caller]
pub fn invert_lower_triangular<E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    parallelism: Parallelism,
) {
    assert!(dst.nrows() == src.nrows());
    assert!(dst.ncols() == src.ncols());
    assert!(dst.nrows() == dst.ncols());

    unsafe { invert_lower_triangular_impl(dst, src, parallelism) }
}

/// Computes the inverse of the upper triangular matrix `src` (with implicit unit
/// diagonal) and stores the strictly upper triangular part of the result to `dst`.
///
/// # Panics
///
/// Panics if `src` and `dst` have mismatching dimensions, or if they are not square.
#[track_caller]
pub fn invert_unit_upper_triangular<E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    parallelism: Parallelism,
) {
    invert_unit_lower_triangular(
        dst.reverse_rows_and_cols(),
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
pub fn invert_upper_triangular<E: ComplexField>(
    dst: MatMut<'_, E>,
    src: MatRef<'_, E>,
    parallelism: Parallelism,
) {
    invert_lower_triangular(
        dst.reverse_rows_and_cols(),
        src.reverse_rows_and_cols(),
        parallelism,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mat;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use rand::random;

    #[test]
    fn test_invert_lower() {
        (0..32).for_each(|n| {
            let a = Mat::from_fn(n, n, |_, _| 2.0 + random::<f64>());
            let mut inv = Mat::zeros(n, n);
            invert_lower_triangular(inv.as_mut(), a.as_ref(), Parallelism::Rayon(0));

            let mut prod = Mat::zeros(n, n);
            triangular::matmul(
                prod.as_mut(),
                BlockStructure::Rectangular,
                a.as_ref(),
                BlockStructure::TriangularLower,
                inv.as_ref(),
                BlockStructure::TriangularLower,
                None,
                1.0,
                Parallelism::Rayon(0),
            );

            for i in 0..n {
                for j in 0..n {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(prod.read(i, j), target, 1e-4);
                }
            }
        });
    }

    #[test]
    fn test_invert_unit_lower() {
        (0..32).for_each(|n| {
            let a = Mat::from_fn(n, n, |_, _| 2.0 + random::<f64>());
            let mut inv = Mat::zeros(n, n);
            invert_unit_lower_triangular(inv.as_mut(), a.as_ref(), Parallelism::Rayon(0));

            let mut prod = Mat::zeros(n, n);
            triangular::matmul(
                prod.as_mut(),
                BlockStructure::Rectangular,
                a.as_ref(),
                BlockStructure::UnitTriangularLower,
                inv.as_ref(),
                BlockStructure::UnitTriangularLower,
                None,
                1.0,
                Parallelism::Rayon(0),
            );
            for i in 0..n {
                for j in 0..n {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(prod.read(i, j), target, 1e-4);
                }
            }
        });
    }

    #[test]
    fn test_invert_upper() {
        (0..32).for_each(|n| {
            let a = Mat::from_fn(n, n, |_, _| 2.0 + random::<f64>());
            let mut inv = Mat::zeros(n, n);
            invert_upper_triangular(inv.as_mut(), a.as_ref(), Parallelism::Rayon(0));

            let mut prod = Mat::zeros(n, n);
            triangular::matmul(
                prod.as_mut(),
                BlockStructure::Rectangular,
                a.as_ref(),
                BlockStructure::TriangularUpper,
                inv.as_ref(),
                BlockStructure::TriangularUpper,
                None,
                1.0,
                Parallelism::Rayon(0),
            );
            for i in 0..n {
                for j in 0..n {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(prod.read(i, j), target, 1e-4);
                }
            }
        });
    }

    #[test]
    fn test_invert_unit_upper() {
        (0..32).for_each(|n| {
            let a = Mat::from_fn(n, n, |_, _| 2.0 + random::<f64>());
            let mut inv = Mat::zeros(n, n);
            invert_unit_upper_triangular(inv.as_mut(), a.as_ref(), Parallelism::Rayon(0));

            let mut prod = Mat::zeros(n, n);
            triangular::matmul(
                prod.as_mut(),
                BlockStructure::Rectangular,
                a.as_ref(),
                BlockStructure::UnitTriangularUpper,
                inv.as_ref(),
                BlockStructure::UnitTriangularUpper,
                None,
                1.0,
                Parallelism::Rayon(0),
            );
            for i in 0..n {
                for j in 0..n {
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert_approx_eq!(prod.read(i, j), target, 1e-4);
                }
            }
        });
    }
}
