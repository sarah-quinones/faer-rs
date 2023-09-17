use super::CholeskyError;
use crate::ldlt_diagonal::compute::RankUpdate;
use assert2::{assert, debug_assert};
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    mul::triangular::BlockStructure, parallelism_degree, solve, zipped, ComplexField, Entity,
    MatMut, Parallelism,
};
use reborrow::*;

fn cholesky_in_place_left_looking_impl<E: ComplexField>(
    matrix: MatMut<'_, E>,
    _parallelism: Parallelism,
) -> Result<(), CholeskyError> {
    let mut matrix = matrix;
    assert_eq!(matrix.ncols(), matrix.nrows());

    let n = matrix.nrows();

    match n {
        0 => return Ok(()),
        1 => {
            let real = matrix.read(0, 0).real();
            return if real > E::Real::zero() {
                matrix.write(0, 0, E::from_real(real.sqrt()));
                Ok(())
            } else {
                Err(CholeskyError)
            };
        }
        _ => (),
    };

    let mut idx = 0;
    let arch = pulp::Arch::new();
    loop {
        let block_size = 1;

        let [_, _, bottom_left, bottom_right] = matrix.rb_mut().split_at(idx, idx);
        let [_, l10, _, l20] = bottom_left.into_const().split_at(block_size, 0);
        let [mut a11, _, a21, _] = bottom_right.split_at(block_size, block_size);

        let l10 = l10.row(0);
        let mut a21 = a21.col(0);

        //
        //      L00
        // A =  L10  A11
        //      L20  A21  A22
        //
        // the first column block is already computed
        // we now compute A11 and A21
        //
        // L00           L00^H L10^H L20^H
        // L10 L11             L11^H L21^H
        // L20 L21 L22 ×             L22^H
        //
        //
        // L00×L00^H
        // L10×L00^H  L10×L10^H + L11×L11^H
        // L20×L00^H  L20×L10^H + L21×L11^H  L20×L20^H + L21×L21^H + L22×L22^H

        // A11 -= L10 × L10^H
        let mut dot = E::Real::zero();
        for j in 0..idx {
            dot = dot.add(&l10.read(0, j).abs2());
        }
        a11.write(0, 0, E::from_real(a11.read(0, 0).real().sub(&dot)));

        let real = a11.read(0, 0).real();
        if real > E::Real::zero() {
            a11.write(0, 0, E::from_real(real.sqrt()));
        } else {
            return Err(CholeskyError);
        };

        if idx + block_size == n {
            break;
        }

        let l11 = a11.read(0, 0);

        // A21 -= L20 × L10^H
        if E::HAS_SIMD && a21.row_stride() == 1 {
            arch.dispatch(RankUpdate {
                a21: a21.rb_mut(),
                l20,
                l10,
            });
        } else {
            for j in 0..idx {
                let l20_col = l20.col(j);
                let l10_conj = l10.read(0, j).conj();

                zipped!(a21.rb_mut(), l20_col)
                    .for_each(|mut dst, src| dst.write(dst.read().sub(&src.read().mul(&l10_conj))));
            }
        }

        // A21 is now L21×L11^H
        // find L21
        //
        // conj(L11) L21^T = A21^T

        let r = l11.real().inv();
        zipped!(a21.rb_mut()).for_each(|mut x| x.write(x.read().scale_real(&r)));

        idx += block_size;
    }
    Ok(())
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct LltParams {}

/// Computes the size and alignment of required workspace for performing a Cholesky
/// decomposition with partial pivoting.
pub fn cholesky_in_place_req<E: Entity>(
    dim: usize,
    parallelism: Parallelism,
    params: LltParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = dim;
    let _ = parallelism;
    let _ = params;
    Ok(StackReq::default())
}

fn cholesky_in_place_impl<E: ComplexField>(
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
) -> Result<(), CholeskyError> {
    // right looking cholesky

    debug_assert!(matrix.nrows() == matrix.ncols());
    let mut matrix = matrix;
    let mut stack = stack;

    let n = matrix.nrows();
    if n < 32 {
        cholesky_in_place_left_looking_impl(matrix, parallelism)
    } else {
        let block_size = Ord::min(n / 2, 128 * parallelism_degree(parallelism));
        let [mut l00, _, mut a10, mut a11] = matrix.rb_mut().split_at(block_size, block_size);

        cholesky_in_place_impl(l00.rb_mut(), parallelism, stack.rb_mut())?;

        let l00 = l00.into_const();

        solve::solve_lower_triangular_in_place(
            l00.conjugate(),
            a10.rb_mut().transpose(),
            parallelism,
        );

        faer_core::mul::triangular::matmul(
            a11.rb_mut(),
            BlockStructure::TriangularLower,
            a10.rb(),
            BlockStructure::Rectangular,
            a10.rb().adjoint(),
            BlockStructure::Rectangular,
            Some(E::one()),
            E::one().neg(),
            parallelism,
        );

        cholesky_in_place_impl(a11, parallelism, stack)
    }
}

/// Computes the Cholesky factor $L$ of a hermitian positive definite input matrix $A$ such that
/// $L$ is lower triangular, and
/// $$LL^H == A.$$
///
/// The result is stored back in the lower half of the same matrix, or an error is returned if the
/// matrix is not positive definite.
///
/// The input matrix is interpreted as symmetric and only the lower triangular part is read.
///
/// The strictly upper triangular part of the matrix is clobbered and may be filled with garbage
/// values.
///
/// # Panics
///
/// Panics if the input matrix is not square.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`cholesky_in_place_req`]).
#[track_caller]
#[inline]
pub fn cholesky_in_place<E: ComplexField>(
    matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: DynStack<'_>,
    params: LltParams,
) -> Result<(), CholeskyError> {
    let _ = params;
    assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );
    cholesky_in_place_impl(matrix, parallelism, stack)
}
