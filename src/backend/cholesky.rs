use core::cmp::Ordering;
use core::ops::Add;
use core::ops::Div;
use core::ops::Mul;
use core::ops::Neg;

use crate::backend::mul;
use crate::backend::mul::triangular::BlockStructure;
use crate::backend::permutation::PermutationIndicesMut;
use crate::backend::solve;
use crate::izip;
use crate::temp_mat_req;
use crate::temp_mat_uninit;
use crate::MatMut;
use crate::MatRef;

use assert2::assert as fancy_assert;
use assert2::debug_assert as fancy_debug_assert;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use num_traits::sign::Signed;
use num_traits::{Inv, One, Zero};
use reborrow::*;

fn cholesky_in_place_left_looking_req<T: 'static>(
    dim: usize,
    block_size: usize,
    n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    let n = dim;
    let bs = block_size.min(dim);

    match n {
        0 | 1 => return Ok(StackReq::default()),
        _ => (),
    }

    use mul::triangular::BlockStructure::*;
    StackReq::try_all_of([
        temp_mat_req::<T>(bs, n - bs)?,
        StackReq::try_any_of([
            mul::triangular::matmul_req::<T>(
                TriangularLower,
                Rectangular,
                Rectangular,
                bs,
                bs,
                n - bs,
                n_threads,
            )?,
            cholesky_in_place_left_looking_req::<T>(bs, bs / 2, n_threads)?,
            mul::matmul_req::<T>(n - bs, bs, n - bs, n_threads)?,
            solve::triangular::solve_unit_lower_triangular_in_place_req::<T>(bs, bs, n_threads)?,
        ])?,
    ])
}

unsafe fn cholesky_in_place_left_looking_unchecked<T>(
    matrix: MatMut<'_, T>,
    block_size: usize,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    let mut matrix = matrix;

    fancy_debug_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );

    let n = matrix.nrows();

    match n {
        0 | 1 => return,
        _ => (),
    };

    let mut idx = 0;
    let mut stack = stack;
    loop {
        let block_size = (n - idx).min(block_size);
        let stack = stack.rb_mut();

        // we split L/D rows/cols into 3 sections each
        //     ┌             ┐
        //     | L00         |
        // L = | L10 A11     |
        //     | L20 A21 A22 |
        //     └             ┘
        //     ┌          ┐
        //     | D0       |
        // D = |    D1    |
        //     |       D2 |
        //     └          ┘
        //
        // we already computed L00, L10, L20, and D0. we now compute L11, L21, and D1

        let (top_left, _, bottom_left, bottom_right) = matrix.rb_mut().split_at_unchecked(idx, idx);
        let l00 = top_left.into_const();
        let d0 = l00.diagonal_unchecked();
        let (_, l10, _, l20) = bottom_left.into_const().split_at_unchecked(block_size, 0);
        let (mut a11, _, mut a21, _) = bottom_right.split_at_unchecked(block_size, block_size);

        // reserve space for L10×D0
        crate::temp_mat_uninit! {
            let (mut l10xd0, mut stack) = unsafe { temp_mat_uninit::<T>(block_size, idx, stack) };
        };

        for (l10xd0_col, l10_col, d_factor) in izip!(
            l10xd0.rb_mut().into_col_iter(),
            l10.rb().into_col_iter(),
            d0.into_iter(),
        ) {
            for (l10xd0_elem, l) in izip!(l10xd0_col, l10_col) {
                *l10xd0_elem = l * d_factor;
            }
        }

        let l10xd0 = l10xd0.into_const();

        mul::triangular::matmul(
            a11.rb_mut(),
            BlockStructure::TriangularLower,
            l10xd0,
            BlockStructure::Rectangular,
            l10.transpose(),
            BlockStructure::Rectangular,
            Some(&T::one()),
            &-&T::one(),
            n_threads,
            stack.rb_mut(),
        );

        cholesky_in_place_left_looking_unchecked(
            a11.rb_mut(),
            block_size / 2,
            n_threads,
            stack.rb_mut(),
        );

        if idx + block_size == n {
            break;
        }

        let ld11 = a11.into_const();
        let l11 = ld11;
        let d1 = ld11.diagonal_unchecked();

        mul::matmul(
            a21.rb_mut(),
            l20,
            l10xd0.transpose(),
            Some(&T::one()),
            &-&T::one(),
            n_threads,
            stack.rb_mut(),
        );

        solve::triangular::solve_unit_lower_triangular_in_place_unchecked(
            l11,
            a21.rb_mut().transpose(),
            n_threads,
            stack,
        );

        let l21xd1 = a21;
        for (l21xd1_col, d1_elem) in izip!(l21xd1.into_col_iter(), d1) {
            let d1_elem_inv = d1_elem.inv();
            for l21xd1_elem in l21xd1_col {
                *l21xd1_elem = &*l21xd1_elem * &d1_elem_inv;
            }
        }

        idx += block_size;
    }
}

/// Computes the memory requirements for a cholesky decomposition of a square matrix of dimension
/// `dim`.
pub fn raw_cholesky_in_place_req<T: 'static>(
    dim: usize,
    n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    if dim < 32 {
        cholesky_in_place_left_looking_req::<T>(dim, 16, n_threads)
    } else {
        let bs = dim / 2;
        let rem = dim - bs;
        StackReq::try_any_of([
            solve::triangular::solve_unit_lower_triangular_in_place_req::<T>(bs, rem, n_threads)?,
            StackReq::try_all_of([
                temp_mat_req::<T>(rem, bs)?,
                mul::triangular::matmul_req::<T>(
                    BlockStructure::TriangularLower,
                    BlockStructure::Rectangular,
                    BlockStructure::Rectangular,
                    rem,
                    rem,
                    bs,
                    n_threads,
                )?,
            ])?,
        ])
    }
}

unsafe fn cholesky_in_place_unchecked<T>(
    matrix: MatMut<'_, T>,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Neg<Output = T> + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    // right looking cholesky

    fancy_debug_assert!(matrix.nrows() == matrix.ncols());
    let mut matrix = matrix;
    let mut stack = stack;

    let n = matrix.nrows();
    if n < 32 {
        cholesky_in_place_left_looking_unchecked(matrix, 16, n_threads, stack);
    } else {
        let block_size = n / 2;
        let rem = n - block_size;
        let (mut l00, _, mut a10, mut a11) =
            matrix.rb_mut().split_at_unchecked(block_size, block_size);

        cholesky_in_place_unchecked(l00.rb_mut(), n_threads, stack.rb_mut());

        let l00 = l00.into_const();
        let d0 = l00.diagonal_unchecked();

        solve::triangular::solve_unit_lower_triangular_in_place_unchecked(
            l00,
            a10.rb_mut().transpose(),
            n_threads,
            stack.rb_mut(),
        );

        {
            temp_mat_uninit! {
                let (mut l10xd0, stack) = unsafe { temp_mat_uninit::<T>(rem, block_size, stack.rb_mut()) };
            };

            for (l10xd0_col, a10_col, d0_elem) in izip!(
                l10xd0.rb_mut().into_col_iter(),
                a10.rb_mut().into_col_iter(),
                d0,
            ) {
                let d0_elem_inv = d0_elem.inv();
                for (l10xd0_elem, a10_elem) in izip!(l10xd0_col, a10_col) {
                    *l10xd0_elem = a10_elem.clone();
                    *a10_elem = &*a10_elem * &d0_elem_inv;
                }
            }

            mul::triangular::matmul(
                a11.rb_mut(),
                BlockStructure::TriangularLower,
                a10.into_const(),
                BlockStructure::Rectangular,
                l10xd0.transpose().into_const(),
                BlockStructure::Rectangular,
                Some(&T::one()),
                &-&T::one(),
                n_threads,
                stack,
            );
        }

        cholesky_in_place_unchecked(a11, n_threads, stack);
    }
}

/// Computes the cholesky factors `L` and `D` of the input matrix such that `L` is strictly lower
/// triangular, `D` is diagonal, and `L×D×L.transpose() == matrix`, then stores them back in the
/// same matrix.
///
/// The input matrix is interpreted as symmetric and only the lower triangular part is read.
///
/// The matrix `L` is stored in the strictly lower triangular part of the input matrix, and the
/// diagonal elements of `D` are stored on the diagonal.
///
/// The strictly upper triangular part of the matrix is untouched.
///
/// # Warning
///
/// The cholesky decomposition may have poor numerical stability properties when used with non
/// positive definite matrices. In the general case, it is recommended to first permute the matrix
/// using [`compute_cholesky_permutation`] and
/// [`apply_symmetric_permutation`](crate::backend::symmetric::apply_symmetric_permutation).
///
/// # Panics
///
/// Panics if the input matrix is not square.
#[track_caller]
#[inline]
pub fn raw_cholesky_in_place<T>(matrix: MatMut<'_, T>, n_threads: usize, stack: DynStack<'_>)
where
    T: Zero + One + Clone + Neg<Output = T> + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Inv<Output = T>,
{
    fancy_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );
    unsafe { cholesky_in_place_unchecked(matrix, n_threads, stack) }
}

/// Computes a permutation that reduces the chances of numerical errors during the cholesky
/// factorization, then stores the result in `perm_indices` and `perm_inv_indices`.
pub fn compute_cholesky_permutation<'a, T>(
    perm_indices: &'a mut [usize],
    perm_inv_indices: &'a mut [usize],
    matrix: MatRef<'_, T>,
) -> PermutationIndicesMut<'a>
where
    T: Signed + PartialOrd,
{
    let n = matrix.nrows();
    fancy_assert!(
        matrix.nrows() == matrix.ncols(),
        "input matrix must be square",
    );
    fancy_assert!(
        perm_indices.len() == n,
        "length of permutation must be equal to the matrix dimension",
    );
    fancy_assert!(
        perm_inv_indices.len() == n,
        "length of inverse permutation must be equal to the matrix dimension",
    );

    let diag = matrix.diagonal();
    for (i, p) in perm_indices.iter_mut().enumerate() {
        *p = i;
    }

    perm_indices.sort_unstable_by(move |&i, &j| {
        let lhs = unsafe { diag.get_unchecked(i) }.abs();
        let rhs = unsafe { diag.get_unchecked(j) }.abs();
        let cmp = rhs.partial_cmp(&lhs);
        if let Some(cmp) = cmp {
            cmp
        } else {
            Ordering::Equal
        }
    });

    for (i, p) in perm_indices.iter().copied().enumerate() {
        *unsafe { perm_inv_indices.get_unchecked_mut(p) } = i;
    }

    unsafe { PermutationIndicesMut::new_unchecked(perm_indices, perm_inv_indices) }
}

pub fn solve_in_place_req<T: 'static>(
    cholesky_dim: usize,
    rhs_ncols: usize,
    n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    use super::solve::triangular::*;
    StackReq::try_any_of([
        solve_unit_lower_triangular_in_place_req::<T>(cholesky_dim, rhs_ncols, n_threads)?,
        solve_unit_upper_triangular_in_place_req::<T>(cholesky_dim, rhs_ncols, n_threads)?,
    ])
}

#[track_caller]
pub fn solve_in_place<T>(
    cholesky_factors: MatRef<'_, T>,
    rhs: MatMut<'_, T>,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + One + Clone + Send + Sync + 'static,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + Div<Output = T>,
{
    let n = cholesky_factors.nrows();
    let k = rhs.ncols();

    fancy_assert!(cholesky_factors.nrows() == cholesky_factors.ncols());
    fancy_assert!(rhs.nrows() == n);

    let mut rhs = rhs;
    let mut stack = stack;

    crate::backend::solve::triangular::solve_unit_lower_triangular_in_place(
        cholesky_factors,
        rhs.rb_mut(),
        n_threads,
        stack.rb_mut(),
    );

    for j in 0..k {
        for i in 0..n {
            let d = unsafe { cholesky_factors.get_unchecked(i, i) };
            let rhs = unsafe { rhs.rb_mut().get_unchecked(i, j) };
            *rhs = &*rhs / d;
        }
    }

    crate::backend::solve::triangular::solve_unit_upper_triangular_in_place(
        cholesky_factors.transpose(),
        rhs.rb_mut(),
        n_threads,
        stack.rb_mut(),
    );
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use dyn_stack::GlobalMemBuffer;
    use rand::random;

    use super::*;
    use crate::Mat;

    #[test]
    pub fn test_roundtrip() {
        let n = 511;
        let mut a = Mat::with_dims(|_, _| random::<f64>(), n, n);
        let a_orig = a.clone();

        raw_cholesky_in_place(
            a.as_mut(),
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
            )),
        );

        let mut lxd = Mat::zeros(n, n);
        for j in 0..n {
            let dj = a[(j, j)];
            lxd[(j, j)] = dj;
            for i in j + 1..n {
                lxd[(i, j)] = a[(i, j)] * dj;
            }
        }

        let mut a_reconstructed = Mat::zeros(n, n);

        use mul::triangular::BlockStructure::*;
        mul::triangular::matmul(
            a_reconstructed.as_mut(),
            Rectangular,
            lxd.as_ref(),
            TriangularLower,
            a.as_ref().transpose(),
            UnitTriangularUpper,
            None,
            &1.0,
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    TriangularLower,
                    TriangularUpper,
                    n,
                    n,
                    n,
                    12,
                )
                .unwrap(),
            )),
        );

        for j in 0..n {
            for i in j..n {
                assert_approx_eq!(a_reconstructed[(i, j)], a_orig[(i, j)]);
            }
        }
    }

    #[test]
    pub fn test_solve() {
        let n = 4;
        let k = 5;
        let mut a = Mat::with_dims(|_, _| random::<f64>(), n, n);
        let mut rhs = Mat::with_dims(|_, _| random::<f64>(), n, k);
        let a_orig = a.clone();
        let rhs_orig = rhs.clone();

        raw_cholesky_in_place(
            a.as_mut(),
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                raw_cholesky_in_place_req::<f64>(n, 12).unwrap(),
            )),
        );

        solve_in_place(
            a.as_ref(),
            rhs.as_mut(),
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                solve_in_place_req::<f64>(n, k, 12).unwrap(),
            )),
        );

        let mut result = Mat::zeros(n, k);
        use mul::triangular::BlockStructure::*;
        mul::triangular::matmul(
            result.as_mut(),
            Rectangular,
            a_orig.as_ref(),
            TriangularLower,
            rhs.as_ref(),
            Rectangular,
            None,
            &1.0,
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    TriangularLower,
                    Rectangular,
                    n,
                    k,
                    n,
                    12,
                )
                .unwrap(),
            )),
        );

        mul::triangular::matmul(
            result.as_mut(),
            Rectangular,
            a_orig.as_ref().transpose(),
            StrictTriangularUpper,
            rhs.as_ref(),
            Rectangular,
            Some(&1.0),
            &1.0,
            12,
            DynStack::new(&mut GlobalMemBuffer::new(
                mul::triangular::matmul_req::<f64>(
                    Rectangular,
                    StrictTriangularUpper,
                    Rectangular,
                    n,
                    k,
                    n,
                    12,
                )
                .unwrap(),
            )),
        );

        for j in 0..k {
            for i in 0..n {
                assert_approx_eq!(result[(i, j)], rhs_orig[(i, j)]);
            }
        }
    }
}
