use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::float_traits::Sqrt;
use faer_core::mul::triangular::BlockStructure;
use faer_core::{mul, solve, ComplexField, MatMut, Parallelism};
use num_traits::{Inv, One, Zero};
use reborrow::*;

use core::ops::{Add, Mul, Neg};

#[derive(Debug)]
pub struct CholeskyError;

unsafe fn cholesky_in_place_left_looking_unchecked<T: ComplexField>(
    matrix: MatMut<'_, T>,
    block_size: usize,
    parallelism: Parallelism,
) -> Result<(), CholeskyError> {
    let mut matrix = matrix;

    fancy_debug_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );

    let n = matrix.nrows();

    match n {
        0 => return Ok(()),
        1 => {
            let elem = matrix.get_unchecked(0, 0);
            return if (*elem).into_real_imag().0 > T::Real::zero() {
                *elem = elem.sqrt();
                Ok(())
            } else {
                Err(CholeskyError)
            };
        }
        _ => (),
    };

    let mut idx = 0;
    loop {
        let block_size = (n - idx).min(block_size);

        let (_, _, bottom_left, bottom_right) = matrix.rb_mut().split_at_unchecked(idx, idx);
        let (_, l10, _, l20) = bottom_left.into_const().split_at_unchecked(block_size, 0);
        let (mut a11, _, mut a21, _) = bottom_right.split_at_unchecked(block_size, block_size);

        faer_core::mul::triangular::matmul_unchecked(
            a11.rb_mut(),
            BlockStructure::TriangularLower,
            l10,
            BlockStructure::Rectangular,
            l10.transpose(),
            BlockStructure::Rectangular,
            Some(T::one()),
            -T::one(),
            false,
            false,
            false,
            parallelism,
        );

        cholesky_in_place_left_looking_unchecked(a11.rb_mut(), block_size / 2, parallelism)?;

        if idx + block_size == n {
            break;
        }

        let ld11 = a11.into_const();
        let l11 = ld11;

        faer_core::mul::matmul_unchecked(
            a21.rb_mut(),
            l20,
            l10.transpose(),
            Some(T::one()),
            -T::one(),
            false,
            false,
            false,
            parallelism,
        );

        solve::triangular::solve_lower_triangular_in_place_unchecked(
            l11,
            a21.rb_mut().transpose(),
            parallelism,
        );

        idx += block_size;
    }
    Ok(())
}

/// Computes the memory requirements for a cholesky decomposition of a square matrix of dimension
/// `dim`.
pub fn raw_cholesky_in_place_req<T: 'static>(
    dim: usize,
    n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    Ok(StackReq::default())
}

unsafe fn cholesky_in_place_unchecked<T: ComplexField>(
    matrix: MatMut<'_, T>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) -> Result<(), CholeskyError> {
    // right looking cholesky

    fancy_debug_assert!(matrix.nrows() == matrix.ncols());
    let mut matrix = matrix;
    let mut stack = stack;

    let n = matrix.nrows();
    if n < 32 {
        cholesky_in_place_left_looking_unchecked(matrix, 16, parallelism)
    } else {
        let block_size = (n / 2).min(128);
        let (mut l00, _, mut a10, mut a11) =
            matrix.rb_mut().split_at_unchecked(block_size, block_size);

        cholesky_in_place_unchecked(l00.rb_mut(), parallelism, stack.rb_mut())?;

        let l00 = l00.into_const();

        solve::triangular::solve_lower_triangular_in_place_unchecked(
            l00,
            a10.rb_mut().transpose(),
            parallelism,
        );

        faer_core::mul::triangular::matmul(
            a11.rb_mut(),
            BlockStructure::TriangularLower,
            a10.rb(),
            BlockStructure::Rectangular,
            a10.rb().transpose(),
            BlockStructure::Rectangular,
            Some(T::one()),
            -T::one(),
            false,
            false,
            false,
            parallelism,
        );

        cholesky_in_place_unchecked(a11, parallelism, stack)
    }
}

/// Computes the cholesky factor `L` of the input matrix such that `L` is lower triangular, and
/// `L×L.transpose() == matrix`, then stores it back in the same matrix, or returns an error if the
/// matrix is not positive definite.
///
/// The input matrix is interpreted as symmetric and only the lower triangular part is read.
///
/// The strictly upper triangular part of the matrix is not accessed.
///
/// # Panics
///
/// Panics if the input matrix is not square.
#[track_caller]
#[inline]
pub fn raw_cholesky_in_place<T: ComplexField>(
    matrix: MatMut<'_, T>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) -> Result<(), CholeskyError> {
    fancy_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );
    unsafe { cholesky_in_place_unchecked(matrix, parallelism, stack) }
}
