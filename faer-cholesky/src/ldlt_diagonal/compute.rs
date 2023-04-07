use assert2::{assert as fancy_assert, debug_assert as fancy_debug_assert};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    mul::triangular::BlockStructure, solve, zip, zip::RowUninit, ComplexField, Conj, MatMut,
    Parallelism,
};
use reborrow::*;

fn cholesky_in_place_left_looking_impl<T: ComplexField>(
    matrix: MatMut<'_, T>,
    parallelism: Parallelism,
) {
    let mut matrix = matrix;
    let _ = parallelism;

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
    let arch = pulp::Arch::new();
    loop {
        let block_size = 1;

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

        let (top_left, top_right, bottom_left, bottom_right) = matrix.rb_mut().split_at(idx, idx);
        let l00 = top_left.into_const();
        let d0 = l00.diagonal();
        let (_, l10, _, l20) = bottom_left.into_const().split_at(block_size, 0);
        let (mut a11, _, a21, _) = bottom_right.split_at(block_size, block_size);

        // reserve space for L10×D0
        let mut l10xd0 = top_right.submatrix(0, 0, idx, block_size).transpose();

        {
            let l10xd0 = l10xd0.rb_mut().row(0);
            let l10 = l10.rb().row(0);

            zip!(RowUninit(l10xd0), l10, d0.transpose())
                .for_each(|dst, src, factor| unsafe { *dst = src.mul(factor) });
        }

        for ((l10xd0_col, l10_col), d_factor) in l10xd0
            .rb_mut()
            .into_col_iter()
            .zip(l10.rb().into_col_iter())
            .zip(d0.into_iter())
        {
            let d_factor = d_factor.clone();
            for (l10xd0_elem, l) in l10xd0_col.into_iter().zip(l10_col) {
                *l10xd0_elem = l.mul(&d_factor);
            }
        }

        let l10xd0 = l10xd0.into_const();

        a11.write_at(
            0,
            0,
            a11[(0, 0)].sub(&faer_core::mul::dot(
                arch,
                l10xd0.row(0).transpose(),
                l10.row(0).transpose(),
            )),
        );

        if idx + block_size == n {
            break;
        }

        let ld11 = a11.into_const();
        let l11 = ld11;

        let mut a21 = a21.col(0);

        for (l20_col, l10) in l20.into_col_iter().zip(l10xd0.row(0)) {
            let l10_conj = l10.conj();
            zip!(a21.rb_mut(), l20_col.rb())
                .for_each(|dst, src| *dst = (*dst).sub(&src.mul(&l10_conj)));
        }

        let r = l11[(0, 0)].real().inv();
        zip!(a21.rb_mut()).for_each(|x| *x = (*x).scale_real(&r));

        idx += block_size;
    }
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct LdltDiagParams {}

/// Computes the size and alignment of required workspace for performing a Cholesky
/// decomposition with partial pivoting.
pub fn raw_cholesky_in_place_req<T: 'static>(
    dim: usize,
    parallelism: Parallelism,
    params: LdltDiagParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = dim;
    let _ = parallelism;
    let _ = params;
    Ok(StackReq::default())
}

fn cholesky_in_place_impl<T: ComplexField>(
    matrix: MatMut<'_, T>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) {
    // right looking cholesky

    fancy_debug_assert!(matrix.nrows() == matrix.ncols());
    let mut matrix = matrix;
    let mut stack = stack;

    let n = matrix.nrows();
    if n < 32 {
        cholesky_in_place_left_looking_impl(matrix, parallelism);
    } else {
        let block_size = (n / 2).min(128);
        let rem = n - block_size;
        let (mut l00, top_right, mut a10, mut a11) =
            matrix.rb_mut().split_at(block_size, block_size);

        cholesky_in_place_impl(l00.rb_mut(), parallelism, stack.rb_mut());

        let l00 = l00.into_const();
        let d0 = l00.diagonal();

        solve::solve_unit_lower_triangular_in_place(
            l00,
            Conj::Yes,
            a10.rb_mut().transpose(),
            Conj::No,
            parallelism,
        );

        {
            // reserve space for L10×D0
            let mut l10xd0 = top_right.submatrix(0, 0, block_size, rem).transpose();

            for ((l10xd0_col, a10_col), d0_elem) in l10xd0
                .rb_mut()
                .into_col_iter()
                .zip(a10.rb_mut().into_col_iter())
                .zip(d0)
            {
                let d0_elem_inv = d0_elem.inv();
                for (l10xd0_elem, a10_elem) in l10xd0_col.into_iter().zip(a10_col) {
                    *l10xd0_elem = a10_elem.clone();
                    *a10_elem = a10_elem.mul(&d0_elem_inv);
                }
            }

            faer_core::mul::triangular::matmul(
                a11.rb_mut(),
                BlockStructure::TriangularLower,
                Conj::No,
                a10.into_const(),
                BlockStructure::Rectangular,
                Conj::No,
                l10xd0.transpose().into_const(),
                BlockStructure::Rectangular,
                Conj::Yes,
                Some(T::one()),
                T::one().neg(),
                parallelism,
            );
        }

        cholesky_in_place_impl(a11, parallelism, stack);
    }
}

/// Computes the Cholesky factors $L$ and $D$ of the input matrix such that $L$ is strictly lower
/// triangular, $D$ is real-valued diagonal, and
/// $$LDL^H = A.$$
///
/// The result is stored back in the same matrix.
///
/// The input matrix is interpreted as symmetric and only the lower triangular part is read.
///
/// The matrix $L$ is stored in the strictly lower triangular part of the input matrix, and the
/// diagonal elements of $D$ are stored on the diagonal.
///
/// The strictly upper triangular part of the matrix is clobbered and may be filled with garbage
/// values.
///
/// # Warning
///
/// The Cholesky decomposition may have poor numerical stability properties when used with non
/// positive definite matrices. In the general case, it is recommended to first permute the matrix
/// using [`crate::compute_cholesky_permutation`] and
/// [`permute_rows_and_cols_symmetric`](faer_core::permutation::permute_rows_and_cols_symmetric_lower).
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
    params: LdltDiagParams,
) {
    let _ = params;
    fancy_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );
    cholesky_in_place_impl(matrix, parallelism, stack)
}
