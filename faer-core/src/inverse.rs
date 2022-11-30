//! Triangular matrix inversion.

use crate::{
    join_raw,
    mul::triangular::{self, BlockStructure},
    solve, ComplexField, Conj, MatMut, MatRef, Parallelism,
};
use assert2::assert as fancy_assert;
use reborrow::*;

unsafe fn invert_lower_triangular_impl_small<const DO_CONJ: bool, T: ComplexField>(
    mut dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
) {
    let m = dst.nrows();
    let mut dst = |i: usize, j: usize| dst.rb_mut().ptr_in_bounds_at_unchecked(i, j);
    let src = |i: usize, j: usize| {
        if !DO_CONJ {
            *src.ptr_in_bounds_at_unchecked(i, j)
        } else {
            (*src.ptr_in_bounds_at_unchecked(i, j)).conj()
        }
    };
    match m {
        0 => {}
        1 => *dst(0, 0) = src(0, 0).inv(),
        2 => {
            let dst00 = src(0, 0).inv();
            let dst11 = src(1, 1).inv();
            let dst10 = -dst11 * src(1, 0) * dst00;

            *dst(0, 0) = dst00;
            *dst(1, 1) = dst11;
            *dst(1, 0) = dst10;
        }
        _ => unreachable!(),
    }
}

unsafe fn invert_unit_lower_triangular_impl_small<const DO_CONJ: bool, T: ComplexField>(
    mut dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
) {
    let m = dst.nrows();
    let mut dst = |i: usize, j: usize| dst.rb_mut().ptr_in_bounds_at_unchecked(i, j);
    let src = |i: usize, j: usize| {
        if !DO_CONJ {
            *src.ptr_in_bounds_at_unchecked(i, j)
        } else {
            (*src.ptr_in_bounds_at_unchecked(i, j)).conj()
        }
    };
    match m {
        0 | 1 => {}
        2 => {
            *dst(1, 0) = -src(1, 0);
        }
        _ => unreachable!(),
    }
}

unsafe fn invert_lower_triangular_impl<T: ComplexField>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    conj: Conj,
    parallelism: Parallelism,
) {
    // m must be equal to n
    let m = dst.nrows();
    let n = dst.ncols();

    if m <= 2 {
        match conj {
            Conj::No => invert_lower_triangular_impl_small::<false, _>(dst, src),
            Conj::Yes => invert_lower_triangular_impl_small::<true, _>(dst, src),
        }
        return;
    }

    let (mut dst_tl, _, mut dst_bl, mut dst_br) = { dst.split_at(m / 2, n / 2) };

    let m = src.nrows();
    let n = src.ncols();
    let (src_tl, _, src_bl, src_br) = { src.split_at(m / 2, n / 2) };

    join_raw(
        |parallelism| invert_lower_triangular_impl(dst_tl.rb_mut(), src_tl, conj, parallelism),
        |parallelism| invert_lower_triangular_impl(dst_br.rb_mut(), src_br, conj, parallelism),
        parallelism,
    );

    triangular::matmul(
        dst_bl.rb_mut(),
        BlockStructure::Rectangular,
        Conj::No,
        src_bl,
        BlockStructure::Rectangular,
        conj,
        dst_tl.rb(),
        BlockStructure::TriangularLower,
        Conj::No,
        None,
        -T::one(),
        parallelism,
    );
    solve::solve_lower_triangular_in_place(src_br, conj, dst_bl, Conj::No, parallelism);
}

unsafe fn invert_unit_lower_triangular_impl<T: ComplexField>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    conj: Conj,
    parallelism: Parallelism,
) {
    // m must be equal to n
    let m = dst.nrows();
    let n = dst.ncols();

    if m <= 2 {
        match conj {
            Conj::No => invert_unit_lower_triangular_impl_small::<false, _>(dst, src),
            Conj::Yes => invert_unit_lower_triangular_impl_small::<true, _>(dst, src),
        }
        return;
    }

    let (mut dst_tl, _, mut dst_bl, mut dst_br) = { dst.split_at(m / 2, n / 2) };

    let m = src.nrows();
    let n = src.ncols();
    let (src_tl, _, src_bl, src_br) = { src.split_at(m / 2, n / 2) };

    join_raw(
        |parallelism| invert_unit_lower_triangular_impl(dst_tl.rb_mut(), src_tl, conj, parallelism),
        |parallelism| invert_unit_lower_triangular_impl(dst_br.rb_mut(), src_br, conj, parallelism),
        parallelism,
    );

    triangular::matmul(
        dst_bl.rb_mut(),
        BlockStructure::Rectangular,
        Conj::No,
        src_bl,
        BlockStructure::Rectangular,
        conj,
        dst_tl.rb(),
        BlockStructure::UnitTriangularLower,
        Conj::No,
        None,
        -T::one(),
        parallelism,
    );
    solve::solve_unit_lower_triangular_in_place(src_br, conj, dst_bl, Conj::No, parallelism);
}

/// Computes the \[conjugate\] inverse of the lower triangular matrix `src` (with implicit unit
/// diagonal) and stores the strictly lower triangular part of the result to `dst`.
///
/// # Panics
///
/// Panics if `src` and `dst` have mismatching dimensions, or if they are not square.
#[track_caller]
pub fn invert_unit_lower_triangular_to<T: ComplexField>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    conj: Conj,
    parallelism: Parallelism,
) {
    fancy_assert!(dst.nrows() == src.nrows());
    fancy_assert!(dst.ncols() == src.ncols());
    fancy_assert!(dst.nrows() == dst.ncols());

    unsafe { invert_unit_lower_triangular_impl(dst, src, conj, parallelism) }
}

/// Computes the \[conjugate\] inverse of the lower triangular matrix `src` and stores the
/// lower triangular part of the result to `dst`.
///
/// # Panics
///
/// Panics if `src` and `dst` have mismatching dimensions, or if they are not square.
#[track_caller]
pub fn invert_lower_triangular_to<T: ComplexField>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    conj: Conj,
    parallelism: Parallelism,
) {
    fancy_assert!(dst.nrows() == src.nrows());
    fancy_assert!(dst.ncols() == src.ncols());
    fancy_assert!(dst.nrows() == dst.ncols());

    unsafe { invert_lower_triangular_impl(dst, src, conj, parallelism) }
}

/// Computes the \[conjugate\] inverse of the upper triangular matrix `src` (with implicit unit
/// diagonal) and stores the strictly upper triangular part of the result to `dst`.
///
/// # Panics
///
/// Panics if `src` and `dst` have mismatching dimensions, or if they are not square.
#[track_caller]
pub fn invert_unit_upper_triangular_to<T: ComplexField>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    conj: Conj,
    parallelism: Parallelism,
) {
    invert_unit_lower_triangular_to(
        dst.reverse_rows_and_cols(),
        src.reverse_rows_and_cols(),
        conj,
        parallelism,
    )
}

/// Computes the \[conjugate\] inverse of the upper triangular matrix `src` and stores the
/// upper triangular part of the result to `dst`.
///
/// # Panics
///
/// Panics if `src` and `dst` have mismatching dimensions, or if they are not square.
#[track_caller]
pub fn invert_upper_triangular_to<T: ComplexField>(
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    conj: Conj,
    parallelism: Parallelism,
) {
    invert_lower_triangular_to(
        dst.reverse_rows_and_cols(),
        src.reverse_rows_and_cols(),
        conj,
        parallelism,
    )
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use rand::random;

    use crate::Mat;

    use super::*;

    #[test]
    fn test_invert_lower() {
        (0..32).for_each(|n| {
            for conj in [Conj::No, Conj::Yes] {
                let a = Mat::with_dims(|_, _| 2.0 + random::<f64>(), n, n);
                let mut inv = Mat::zeros(n, n);
                invert_lower_triangular_to(inv.as_mut(), a.as_ref(), conj, Parallelism::Rayon(0));

                let mut prod = Mat::zeros(n, n);
                triangular::matmul(
                    prod.as_mut(),
                    BlockStructure::Rectangular,
                    Conj::No,
                    a.as_ref(),
                    BlockStructure::TriangularLower,
                    conj,
                    inv.as_ref(),
                    BlockStructure::TriangularLower,
                    conj,
                    None,
                    1.0,
                    Parallelism::Rayon(0),
                );

                for i in 0..n {
                    for j in 0..n {
                        let target = if i == j { 1.0 } else { 0.0 };
                        assert_approx_eq!(prod[(i, j)], target, 1e-4);
                    }
                }
            }
        });
    }

    #[test]
    fn test_invert_unit_lower() {
        (0..32).for_each(|n| {
            for conj in [Conj::No, Conj::Yes] {
                dbg!(n, conj);
                let a = Mat::with_dims(|_, _| 2.0 + random::<f64>(), n, n);
                let mut inv = Mat::zeros(n, n);
                invert_unit_lower_triangular_to(
                    inv.as_mut(),
                    a.as_ref(),
                    conj,
                    Parallelism::Rayon(0),
                );

                let mut prod = Mat::zeros(n, n);
                triangular::matmul(
                    prod.as_mut(),
                    BlockStructure::Rectangular,
                    Conj::No,
                    a.as_ref(),
                    BlockStructure::UnitTriangularLower,
                    conj,
                    inv.as_ref(),
                    BlockStructure::UnitTriangularLower,
                    conj,
                    None,
                    1.0,
                    Parallelism::Rayon(0),
                );
                for i in 0..n {
                    for j in 0..n {
                        let target = if i == j { 1.0 } else { 0.0 };
                        assert_approx_eq!(prod[(i, j)], target, 1e-4);
                    }
                }
            }
        });
    }

    #[test]
    fn test_invert_upper() {
        (0..32).for_each(|n| {
            for conj in [Conj::No, Conj::Yes] {
                let a = Mat::with_dims(|_, _| 2.0 + random::<f64>(), n, n);
                let mut inv = Mat::zeros(n, n);
                invert_upper_triangular_to(inv.as_mut(), a.as_ref(), conj, Parallelism::Rayon(0));

                let mut prod = Mat::zeros(n, n);
                triangular::matmul(
                    prod.as_mut(),
                    BlockStructure::Rectangular,
                    Conj::No,
                    a.as_ref(),
                    BlockStructure::TriangularUpper,
                    conj,
                    inv.as_ref(),
                    BlockStructure::TriangularUpper,
                    conj,
                    None,
                    1.0,
                    Parallelism::Rayon(0),
                );
                for i in 0..n {
                    for j in 0..n {
                        let target = if i == j { 1.0 } else { 0.0 };
                        assert_approx_eq!(prod[(i, j)], target, 1e-4);
                    }
                }
            }
        });
    }

    #[test]
    fn test_invert_unit_upper() {
        (0..32).for_each(|n| {
            for conj in [Conj::No, Conj::Yes] {
                let a = Mat::with_dims(|_, _| 2.0 + random::<f64>(), n, n);
                let mut inv = Mat::zeros(n, n);
                invert_unit_upper_triangular_to(
                    inv.as_mut(),
                    a.as_ref(),
                    conj,
                    Parallelism::Rayon(0),
                );

                let mut prod = Mat::zeros(n, n);
                triangular::matmul(
                    prod.as_mut(),
                    BlockStructure::Rectangular,
                    Conj::No,
                    a.as_ref(),
                    BlockStructure::UnitTriangularUpper,
                    conj,
                    inv.as_ref(),
                    BlockStructure::UnitTriangularUpper,
                    conj,
                    None,
                    1.0,
                    Parallelism::Rayon(0),
                );
                for i in 0..n {
                    for j in 0..n {
                        let target = if i == j { 1.0 } else { 0.0 };
                        assert_approx_eq!(prod[(i, j)], target, 1e-4);
                    }
                }
            }
        });
    }
}
