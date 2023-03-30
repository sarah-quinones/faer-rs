//! Triangular solve module.

use crate::{join_raw, ComplexField, Conj, MatMut, MatRef, Parallelism};
use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use reborrow::*;

#[inline(always)]
fn identity<T>(x: T) -> T {
    x
}

#[inline(always)]
fn conj<T: ComplexField>(x: T) -> T {
    x.conj()
}

#[inline(always)]
unsafe fn solve_unit_lower_triangular_in_place_base_case_generic_unchecked<T: ComplexField>(
    tril: MatRef<'_, T>,
    rhs: MatMut<'_, T>,
    maybe_conj_lhs: impl Fn(T) -> T,
    maybe_conj_rhs: impl Fn(T) -> T,
) {
    let n = tril.nrows();
    match n {
        0 => (),
        1 => {
            let x0 = rhs.row_unchecked(0);
            x0.cwise().for_each(|x0| {
                *x0 = maybe_conj_rhs(*x0);
            });
        }
        2 => {
            let nl10_div_l11 = -maybe_conj_lhs(*tril.get_unchecked(1, 0));

            let (_, x0, _, x1) = rhs.split_at_unchecked(1, 0);
            let x0 = x0.row_unchecked(0);
            let x1 = x1.row_unchecked(0);

            x0.cwise().zip_unchecked(x1).for_each(|x0, x1| {
                *x0 = maybe_conj_rhs(*x0);
                *x1 = maybe_conj_rhs(*x1) + nl10_div_l11 * *x0;
            });
        }
        3 => {
            let nl10_div_l11 = -maybe_conj_lhs(*tril.get_unchecked(1, 0));
            let nl20_div_l22 = -maybe_conj_lhs(*tril.get_unchecked(2, 0));
            let nl21_div_l22 = -maybe_conj_lhs(*tril.get_unchecked(2, 1));

            let (_, x0, _, x1_2) = rhs.split_at_unchecked(1, 0);
            let (_, x1, _, x2) = x1_2.split_at_unchecked(1, 0);
            let x0 = x0.row_unchecked(0);
            let x1 = x1.row_unchecked(0);
            let x2 = x2.row_unchecked(0);

            x0.cwise()
                .zip_unchecked(x1)
                .zip_unchecked(x2)
                .for_each(|x0, x1, x2| {
                    let y0 = maybe_conj_rhs(*x0);
                    let mut y1 = maybe_conj_rhs(*x1);
                    let mut y2 = maybe_conj_rhs(*x2);
                    y1 = y1 + nl10_div_l11 * y0;
                    y2 = y2 + nl20_div_l22 * y0 + nl21_div_l22 * y1;
                    *x0 = y0;
                    *x1 = y1;
                    *x2 = y2;
                });
        }
        4 => {
            let nl10_div_l11 = -maybe_conj_lhs(*tril.get_unchecked(1, 0));
            let nl20_div_l22 = -maybe_conj_lhs(*tril.get_unchecked(2, 0));
            let nl21_div_l22 = -maybe_conj_lhs(*tril.get_unchecked(2, 1));
            let nl30_div_l33 = -maybe_conj_lhs(*tril.get_unchecked(3, 0));
            let nl31_div_l33 = -maybe_conj_lhs(*tril.get_unchecked(3, 1));
            let nl32_div_l33 = -maybe_conj_lhs(*tril.get_unchecked(3, 2));

            let (_, x0, _, x1_2_3) = rhs.split_at_unchecked(1, 0);
            let (_, x1, _, x2_3) = x1_2_3.split_at_unchecked(1, 0);
            let (_, x2, _, x3) = x2_3.split_at_unchecked(1, 0);
            let x0 = x0.row_unchecked(0);
            let x1 = x1.row_unchecked(0);
            let x2 = x2.row_unchecked(0);
            let x3 = x3.row_unchecked(0);

            x0.cwise()
                .zip_unchecked(x1)
                .zip_unchecked(x2)
                .zip_unchecked(x3)
                .for_each(|x0, x1, x2, x3| {
                    let y0 = maybe_conj_rhs(*x0);
                    let mut y1 = maybe_conj_rhs(*x1);
                    let mut y2 = maybe_conj_rhs(*x2);
                    let mut y3 = maybe_conj_rhs(*x3);
                    y1 = y1 + nl10_div_l11 * y0;
                    y2 = y2 + (nl20_div_l22 * y0 + nl21_div_l22 * y1);
                    y3 = (y3 + nl30_div_l33 * y0) + (nl31_div_l33 * y1 + nl32_div_l33 * y2);
                    *x0 = y0;
                    *x1 = y1;
                    *x2 = y2;
                    *x3 = y3;
                });
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
unsafe fn solve_lower_triangular_in_place_base_case_generic_unchecked<T: ComplexField>(
    tril: MatRef<'_, T>,
    rhs: MatMut<'_, T>,
    maybe_conj_lhs: impl Fn(T) -> T,
    maybe_conj_rhs: impl Fn(T) -> T,
) {
    let n = tril.nrows();
    match n {
        0 => (),
        1 => {
            let inv = maybe_conj_lhs(*tril.get_unchecked(0, 0)).inv();
            let x0 = rhs.row_unchecked(0);
            x0.cwise().for_each(|x0| *x0 = maybe_conj_rhs(*x0) * inv);
        }
        2 => {
            let l00_inv = maybe_conj_lhs(*tril.get_unchecked(0, 0)).inv();
            let l11_inv = maybe_conj_lhs(*tril.get_unchecked(1, 1)).inv();
            let nl10_div_l11 = -(maybe_conj_lhs(*tril.get_unchecked(1, 0)) * l11_inv);

            let (_, x0, _, x1) = rhs.split_at_unchecked(1, 0);
            let x0 = x0.row_unchecked(0);
            let x1 = x1.row_unchecked(0);

            x0.cwise().zip_unchecked(x1).for_each(|x0, x1| {
                *x0 = maybe_conj_rhs(*x0) * l00_inv;
                *x1 = maybe_conj_rhs(*x1) * l11_inv + nl10_div_l11 * *x0;
            });
        }
        3 => {
            let l00_inv = maybe_conj_lhs(*tril.get_unchecked(0, 0)).inv();
            let l11_inv = maybe_conj_lhs(*tril.get_unchecked(1, 1)).inv();
            let l22_inv = maybe_conj_lhs(*tril.get_unchecked(2, 2)).inv();
            let nl10_div_l11 = -(maybe_conj_lhs(*tril.get_unchecked(1, 0)) * l11_inv);
            let nl20_div_l22 = -(maybe_conj_lhs(*tril.get_unchecked(2, 0)) * l22_inv);
            let nl21_div_l22 = -(maybe_conj_lhs(*tril.get_unchecked(2, 1)) * l22_inv);

            let (_, x0, _, x1_2) = rhs.split_at_unchecked(1, 0);
            let (_, x1, _, x2) = x1_2.split_at_unchecked(1, 0);
            let x0 = x0.row_unchecked(0);
            let x1 = x1.row_unchecked(0);
            let x2 = x2.row_unchecked(0);

            x0.cwise()
                .zip_unchecked(x1)
                .zip_unchecked(x2)
                .for_each(|x0, x1, x2| {
                    let mut y0 = maybe_conj_rhs(*x0);
                    let mut y1 = maybe_conj_rhs(*x1);
                    let mut y2 = maybe_conj_rhs(*x2);
                    y0 = y0 * l00_inv;
                    y1 = y1 * l11_inv + nl10_div_l11 * y0;
                    y2 = y2 * l22_inv + nl20_div_l22 * y0 + nl21_div_l22 * y1;
                    *x0 = y0;
                    *x1 = y1;
                    *x2 = y2;
                });
        }
        4 => {
            let l00_inv = maybe_conj_lhs(*tril.get_unchecked(0, 0)).inv();
            let l11_inv = maybe_conj_lhs(*tril.get_unchecked(1, 1)).inv();
            let l22_inv = maybe_conj_lhs(*tril.get_unchecked(2, 2)).inv();
            let l33_inv = maybe_conj_lhs(*tril.get_unchecked(3, 3)).inv();
            let nl10_div_l11 = -(maybe_conj_lhs(*tril.get_unchecked(1, 0)) * l11_inv);
            let nl20_div_l22 = -(maybe_conj_lhs(*tril.get_unchecked(2, 0)) * l22_inv);
            let nl21_div_l22 = -(maybe_conj_lhs(*tril.get_unchecked(2, 1)) * l22_inv);
            let nl30_div_l33 = -(maybe_conj_lhs(*tril.get_unchecked(3, 0)) * l33_inv);
            let nl31_div_l33 = -(maybe_conj_lhs(*tril.get_unchecked(3, 1)) * l33_inv);
            let nl32_div_l33 = -(maybe_conj_lhs(*tril.get_unchecked(3, 2)) * l33_inv);

            let (_, x0, _, x1_2_3) = rhs.split_at_unchecked(1, 0);
            let (_, x1, _, x2_3) = x1_2_3.split_at_unchecked(1, 0);
            let (_, x2, _, x3) = x2_3.split_at_unchecked(1, 0);
            let x0 = x0.row_unchecked(0);
            let x1 = x1.row_unchecked(0);
            let x2 = x2.row_unchecked(0);
            let x3 = x3.row_unchecked(0);

            x0.cwise()
                .zip_unchecked(x1)
                .zip_unchecked(x2)
                .zip_unchecked(x3)
                .for_each(|x0, x1, x2, x3| {
                    let mut y0 = maybe_conj_rhs(*x0);
                    let mut y1 = maybe_conj_rhs(*x1);
                    let mut y2 = maybe_conj_rhs(*x2);
                    let mut y3 = maybe_conj_rhs(*x3);
                    y0 = y0 * l00_inv;
                    y1 = y1 * l11_inv + nl10_div_l11 * y0;
                    y2 = y2 * l22_inv + (nl20_div_l22 * y0 + nl21_div_l22 * y1);
                    y3 = (y3 * l33_inv + nl30_div_l33 * y0)
                        + (nl31_div_l33 * y1 + nl32_div_l33 * y2);
                    *x0 = y0;
                    *x1 = y1;
                    *x2 = y2;
                    *x3 = y3;
                });
        }
        _ => unreachable!(),
    }
}

#[inline]
fn blocksize<T: 'static>(n: usize) -> usize {
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
fn recursion_threshold<T: 'static>() -> usize {
    4
}

/// Computes the solution of `Op_lhs(triangular_lower)×X = Op(rhs)`, and stores the result in
/// `rhs`.
///
/// `triangular_lower` is interpreted as a lower triangular matrix (diagonal included).
/// Its strictly upper triangular part is not accessed.
///
/// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
/// `Op_rhs` is the identity if `conj_rhs` is `Conj::No`, and the conjugation operation if it
/// is `Conj::Yes`.
///
/// # Panics
///
///  - Panics if `triangular_lower` is not a square matrix.
///  - Panics if `rhs.nrows() != triangular_lower.ncols()`
///
/// # Example
///
/// ```
/// use faer_core::{
///     mat,
///     mul::triangular::{matmul, BlockStructure},
///     solve::solve_lower_triangular_in_place,
///     Conj, Mat, Parallelism,
/// };
///
/// let m = mat![[1.0, 0.0], [2.0, 3.0]];
/// let rhs = mat![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let mut sol = rhs.clone();
/// solve_lower_triangular_in_place(
///     m.as_ref(),
///     Conj::No,
///     sol.as_mut(),
///     Conj::No,
///     Parallelism::None,
/// );
///
/// let mut m_times_sol = Mat::<f64>::zeros(2, 3);
/// matmul(
///     m_times_sol.as_mut(),
///     BlockStructure::Rectangular,
///     Conj::No,
///     m.as_ref(),
///     BlockStructure::TriangularLower,
///     Conj::No,
///     sol.as_ref(),
///     BlockStructure::Rectangular,
///     Conj::No,
///     None,
///     1.0,
///     Parallelism::None,
/// );
///
/// m_times_sol
///     .as_ref()
///     .cwise()
///     .zip(rhs.as_ref())
///     .for_each(|x, target| assert!((x - target).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn solve_lower_triangular_in_place<T: ComplexField>(
    triangular_lower: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    fancy_assert!(triangular_lower.nrows() == triangular_lower.ncols());
    fancy_assert!(rhs.nrows() == triangular_lower.ncols());

    unsafe {
        solve_lower_triangular_in_place_unchecked(
            triangular_lower,
            conj_lhs,
            rhs,
            conj_rhs,
            parallelism,
        );
    }
}

/// Computes the solution of `Op_lhs(triangular_upper)×X = Op(rhs)`, and stores the result in
/// `rhs`.
///
/// `triangular_upper` is interpreted as a upper triangular matrix (diagonal included).
/// Its strictly lower triangular part is not accessed.
///
/// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
/// `Op_rhs` is the identity if `conj_rhs` is `Conj::No`, and the conjugation operation if it
/// is `Conj::Yes`.
///
/// # Panics
///
///  - Panics if `triangular_upper` is not a square matrix.
///  - Panics if `rhs.nrows() != triangular_lower.ncols()`
///
/// # Example
///
/// ```
/// use faer_core::{
///     mat,
///     mul::triangular::{matmul, BlockStructure},
///     solve::solve_upper_triangular_in_place,
///     Conj, Mat, Parallelism,
/// };
///
/// let m = mat![[1.0, 2.0], [0.0, 3.0]];
/// let rhs = mat![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let mut sol = rhs.clone();
/// solve_upper_triangular_in_place(
///     m.as_ref(),
///     Conj::No,
///     sol.as_mut(),
///     Conj::No,
///     Parallelism::None,
/// );
///
/// let mut m_times_sol = Mat::<f64>::zeros(2, 3);
/// matmul(
///     m_times_sol.as_mut(),
///     BlockStructure::Rectangular,
///     Conj::No,
///     m.as_ref(),
///     BlockStructure::TriangularUpper,
///     Conj::No,
///     sol.as_ref(),
///     BlockStructure::Rectangular,
///     Conj::No,
///     None,
///     1.0,
///     Parallelism::None,
/// );
///
/// m_times_sol
///     .as_ref()
///     .cwise()
///     .zip(rhs.as_ref())
///     .for_each(|x, target| assert!((x - target).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn solve_upper_triangular_in_place<T: ComplexField>(
    triangular_upper: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    fancy_assert!(triangular_upper.nrows() == triangular_upper.ncols());
    fancy_assert!(rhs.nrows() == triangular_upper.ncols());

    unsafe {
        solve_upper_triangular_in_place_unchecked(
            triangular_upper,
            conj_lhs,
            rhs,
            conj_rhs,
            parallelism,
        );
    }
}

/// Computes the solution of `Op_lhs(triangular_lower)×X = Op(rhs)`, and stores the result in
/// `rhs`.
///
/// `triangular_lower` is interpreted as a lower triangular matrix, and its diagonal elements are
/// implicitly considered to be `1.0`. Its upper triangular part is not accessed.
///
/// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
/// `Op_rhs` is the identity if `conj_rhs` is `Conj::No`, and the conjugation operation if it
/// is `Conj::Yes`.
///
/// # Panics
///
///  - Panics if `triangular_lower` is not a square matrix.
///  - Panics if `rhs.nrows() != triangular_lower.ncols()`
///
/// # Example
///
/// ```
/// use faer_core::{
///     mat,
///     mul::triangular::{matmul, BlockStructure},
///     solve::solve_unit_lower_triangular_in_place,
///     Conj, Mat, Parallelism,
/// };
///
/// let m = mat![[0.0, 0.0], [2.0, 0.0]];
/// let rhs = mat![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let mut sol = rhs.clone();
/// solve_unit_lower_triangular_in_place(
///     m.as_ref(),
///     Conj::No,
///     sol.as_mut(),
///     Conj::No,
///     Parallelism::None,
/// );
///
/// let mut m_times_sol = Mat::<f64>::zeros(2, 3);
/// matmul(
///     m_times_sol.as_mut(),
///     BlockStructure::Rectangular,
///     Conj::No,
///     m.as_ref(),
///     BlockStructure::UnitTriangularLower,
///     Conj::No,
///     sol.as_ref(),
///     BlockStructure::Rectangular,
///     Conj::No,
///     None,
///     1.0,
///     Parallelism::None,
/// );
///
/// m_times_sol
///     .as_ref()
///     .cwise()
///     .zip(rhs.as_ref())
///     .for_each(|x, target| assert!((x - target).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn solve_unit_lower_triangular_in_place<T: ComplexField>(
    triangular_lower: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    fancy_assert!(triangular_lower.nrows() == triangular_lower.ncols());
    fancy_assert!(rhs.nrows() == triangular_lower.ncols());

    unsafe {
        solve_unit_lower_triangular_in_place_unchecked(
            triangular_lower,
            conj_lhs,
            rhs,
            conj_rhs,
            parallelism,
        );
    }
}

/// Computes the solution of `Op_lhs(triangular_upper)×X = Op(rhs)`, and stores the result in
/// `rhs`.
///
/// `triangular_upper` is interpreted as a upper triangular matrix, and its diagonal elements are
/// implicitly considered to be `1.0`. Its lower triangular part is not accessed.
///
/// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
/// `Op_rhs` is the identity if `conj_rhs` is `Conj::No`, and the conjugation operation if it
/// is `Conj::Yes`.
///
/// # Panics
///
///  - Panics if `triangular_upper` is not a square matrix.
///  - Panics if `rhs.nrows() != triangular_lower.ncols()`
///
/// ```
/// use faer_core::{
///     mat,
///     mul::triangular::{matmul, BlockStructure},
///     solve::solve_unit_upper_triangular_in_place,
///     Conj, Mat, Parallelism,
/// };
///
/// let m = mat![[0.0, 2.0], [0.0, 0.0]];
/// let rhs = mat![[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let mut sol = rhs.clone();
/// solve_unit_upper_triangular_in_place(
///     m.as_ref(),
///     Conj::No,
///     sol.as_mut(),
///     Conj::No,
///     Parallelism::None,
/// );
///
/// let mut m_times_sol = Mat::<f64>::zeros(2, 3);
/// matmul(
///     m_times_sol.as_mut(),
///     BlockStructure::Rectangular,
///     Conj::No,
///     m.as_ref(),
///     BlockStructure::UnitTriangularUpper,
///     Conj::No,
///     sol.as_ref(),
///     BlockStructure::Rectangular,
///     Conj::No,
///     None,
///     1.0,
///     Parallelism::None,
/// );
///
/// m_times_sol
///     .as_ref()
///     .cwise()
///     .zip(rhs.as_ref())
///     .for_each(|x, target| assert!((x - target).abs() < 1e-10));
/// ```
#[track_caller]
#[inline]
pub fn solve_unit_upper_triangular_in_place<T: ComplexField>(
    triangular_upper: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    fancy_assert!(triangular_upper.nrows() == triangular_upper.ncols());
    fancy_assert!(rhs.nrows() == triangular_upper.ncols());

    unsafe {
        solve_unit_upper_triangular_in_place_unchecked(
            triangular_upper,
            conj_lhs,
            rhs,
            conj_rhs,
            parallelism,
        );
    }
}

/// # Safety
///
/// Same as [`solve_unit_lower_triangular_in_place`], except that panics become undefined behavior.
///
/// # Example
///
/// See [`solve_unit_lower_triangular_in_place`].
pub unsafe fn solve_unit_lower_triangular_in_place_unchecked<T: ComplexField>(
    tril: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    let n = tril.nrows();
    let k = rhs.ncols();

    if k > 64 && n <= 128 {
        let (_, _, rhs_left, rhs_right) = rhs.split_at_unchecked(0, k / 2);
        join_raw(
            |_| {
                solve_unit_lower_triangular_in_place_unchecked(
                    tril,
                    conj_lhs,
                    rhs_left,
                    conj_rhs,
                    parallelism,
                )
            },
            |_| {
                solve_unit_lower_triangular_in_place_unchecked(
                    tril,
                    conj_lhs,
                    rhs_right,
                    conj_rhs,
                    parallelism,
                )
            },
            parallelism,
        );
        return;
    }

    fancy_debug_assert!(tril.nrows() == tril.ncols());
    fancy_debug_assert!(rhs.nrows() == tril.ncols());

    if n <= recursion_threshold::<T>() {
        pulp::Arch::new().dispatch(
            #[inline(always)]
            || match (conj_lhs, conj_rhs) {
                (Conj::Yes, Conj::Yes) => {
                    solve_unit_lower_triangular_in_place_base_case_generic_unchecked(
                        tril, rhs, conj, conj,
                    )
                }
                (Conj::Yes, Conj::No) => {
                    solve_unit_lower_triangular_in_place_base_case_generic_unchecked(
                        tril, rhs, conj, identity,
                    )
                }
                (Conj::No, Conj::Yes) => {
                    solve_unit_lower_triangular_in_place_base_case_generic_unchecked(
                        tril, rhs, identity, conj,
                    )
                }
                (Conj::No, Conj::No) => {
                    solve_unit_lower_triangular_in_place_base_case_generic_unchecked(
                        tril, rhs, identity, identity,
                    )
                }
            },
        );
        return;
    }

    let bs = blocksize::<T>(n);

    let (tril_top_left, _, tril_bot_left, tril_bot_right) = tril.split_at_unchecked(bs, bs);
    let (_, mut rhs_top, _, mut rhs_bot) = rhs.split_at_unchecked(bs, 0);

    //       (A00    )   X0         (B0)
    // ConjA?(A10 A11)   X1 = ConjB?(B1)
    //
    //
    // 1. ConjA?(A00) X0 = ConjB?(B0)
    //
    // 2. ConjA?(A10) X0 + ConjA?(A11) X1 = ConjB?(B1)
    // => ConjA?(A11) X1 = ConjB?(B1) - ConjA?(A10) X0

    solve_unit_lower_triangular_in_place_unchecked(
        tril_top_left,
        conj_lhs,
        rhs_top.rb_mut(),
        conj_rhs,
        parallelism,
    );

    crate::mul::matmul(
        rhs_bot.rb_mut(),
        conj_rhs,
        tril_bot_left,
        conj_lhs,
        rhs_top.into_const(),
        Conj::No,
        Some(T::one()),
        -T::one(),
        parallelism,
    );

    solve_unit_lower_triangular_in_place_unchecked(
        tril_bot_right,
        conj_lhs,
        rhs_bot,
        Conj::No,
        parallelism,
    );
}

/// # Safety
///
/// Same as [`solve_unit_upper_triangular_in_place`], except that panics become undefined behavior.
///
/// # Example
///
/// See [`solve_unit_upper_triangular_in_place`].
#[inline]
pub unsafe fn solve_unit_upper_triangular_in_place_unchecked<T: ComplexField>(
    triu: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    solve_unit_lower_triangular_in_place_unchecked(
        triu.reverse_rows_and_cols(),
        conj_lhs,
        rhs.reverse_rows(),
        conj_rhs,
        parallelism,
    );
}

/// # Safety
///
/// Same as [`solve_lower_triangular_in_place`], except that panics become undefined behavior.
///
/// # Example
///
/// See [`solve_lower_triangular_in_place`].
pub unsafe fn solve_lower_triangular_in_place_unchecked<T: ComplexField>(
    tril: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    let n = tril.nrows();
    let k = rhs.ncols();

    if k > 64 && n <= 128 {
        let (_, _, rhs_left, rhs_right) = rhs.split_at_unchecked(0, k / 2);
        join_raw(
            |_| {
                solve_lower_triangular_in_place_unchecked(
                    tril,
                    conj_lhs,
                    rhs_left,
                    conj_rhs,
                    parallelism,
                )
            },
            |_| {
                solve_lower_triangular_in_place_unchecked(
                    tril,
                    conj_lhs,
                    rhs_right,
                    conj_rhs,
                    parallelism,
                )
            },
            parallelism,
        );
        return;
    }

    fancy_debug_assert!(tril.nrows() == tril.ncols());
    fancy_debug_assert!(rhs.nrows() == tril.ncols());

    let n = tril.nrows();

    if n <= recursion_threshold::<T>() {
        pulp::Arch::new().dispatch(
            #[inline(always)]
            || match (conj_lhs, conj_rhs) {
                (Conj::Yes, Conj::Yes) => {
                    solve_lower_triangular_in_place_base_case_generic_unchecked(
                        tril, rhs, conj, conj,
                    )
                }
                (Conj::Yes, Conj::No) => {
                    solve_lower_triangular_in_place_base_case_generic_unchecked(
                        tril, rhs, conj, identity,
                    )
                }
                (Conj::No, Conj::Yes) => {
                    solve_lower_triangular_in_place_base_case_generic_unchecked(
                        tril, rhs, identity, conj,
                    )
                }
                (Conj::No, Conj::No) => {
                    solve_lower_triangular_in_place_base_case_generic_unchecked(
                        tril, rhs, identity, identity,
                    )
                }
            },
        );
        return;
    }

    let bs = blocksize::<T>(n);

    let (tril_top_left, _, tril_bot_left, tril_bot_right) = tril.split_at_unchecked(bs, bs);
    let (_, mut rhs_top, _, mut rhs_bot) = rhs.split_at_unchecked(bs, 0);

    solve_lower_triangular_in_place_unchecked(
        tril_top_left,
        conj_lhs,
        rhs_top.rb_mut(),
        conj_rhs,
        parallelism,
    );

    crate::mul::matmul(
        rhs_bot.rb_mut(),
        conj_rhs,
        tril_bot_left,
        conj_lhs,
        rhs_top.into_const(),
        Conj::No,
        Some(T::one()),
        -T::one(),
        parallelism,
    );

    solve_lower_triangular_in_place_unchecked(
        tril_bot_right,
        conj_lhs,
        rhs_bot,
        Conj::No,
        parallelism,
    );
}

/// # Safety
///
/// Same as [`solve_upper_triangular_in_place`], except that panics become undefined behavior.
///
/// # Example
///
/// See [`solve_upper_triangular_in_place`].
#[inline]
pub unsafe fn solve_upper_triangular_in_place_unchecked<T: ComplexField>(
    triu: MatRef<'_, T>,
    conj_lhs: Conj,
    rhs: MatMut<'_, T>,
    conj_rhs: Conj,
    parallelism: Parallelism,
) {
    solve_lower_triangular_in_place_unchecked(
        triu.reverse_rows_and_cols(),
        conj_lhs,
        rhs.reverse_rows(),
        conj_rhs,
        parallelism,
    );
}
