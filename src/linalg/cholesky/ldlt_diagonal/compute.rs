use crate::{
    assert, debug_assert,
    linalg::{
        matmul::triangular::BlockStructure, temp_mat_req, temp_mat_uninit,
        triangular_solve as solve,
    },
    unzipped,
    utils::{simd::*, slice::*},
    zipped, ColMut, Conj, MatMut, MatRef, Parallelism, RowRef,
};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_entity::*;
use reborrow::*;

pub(crate) struct RankUpdate<'a, E: ComplexField> {
    pub a21: ColMut<'a, E>,
    pub l20: MatRef<'a, E>,
    pub l10: RowRef<'a, E>,
}

impl<E: ComplexField> pulp::WithSimd for RankUpdate<'_, E> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self { a21, l20, l10 } = self;

        debug_assert_eq!(a21.row_stride(), 1);
        debug_assert_eq!(l20.row_stride(), 1);
        debug_assert_eq!(l20.nrows(), a21.nrows());
        debug_assert_eq!(l20.ncols(), l10.ncols());

        let m = l20.nrows();
        let n = l20.ncols();

        if m == 0 {
            return;
        }

        let simd = SimdFor::<E, S>::new(simd);
        let acc = SliceGroupMut::<'_, E>::new(a21.try_get_contiguous_col_mut());
        let offset = simd.align_offset(acc.rb());

        let (mut acc_head, mut acc_body, mut acc_tail) = simd.as_aligned_simd_mut(acc, offset);

        for j in 0..n {
            let l10 = simd.splat(l10.read(j).faer_neg().faer_conj());
            let l20 = SliceGroup::<'_, E>::new(l20.try_get_contiguous_col(j));

            let (l20_head, l20_body, l20_tail) = simd.as_aligned_simd(l20, offset);

            #[inline(always)]
            fn process<E: ComplexField, S: pulp::Simd>(
                simd: SimdFor<E, S>,
                mut acc: impl Write<Output = SimdGroupFor<E, S>>,
                l20: impl Read<Output = SimdGroupFor<E, S>>,
                l10: SimdGroupFor<E, S>,
            ) {
                let zero = simd.splat(E::faer_zero());
                acc.write(simd.mul_add_e(l10, l20.read_or(zero), acc.read_or(zero)));
            }

            process(simd, acc_head.rb_mut(), l20_head, l10);
            for (acc, l20) in acc_body
                .rb_mut()
                .into_mut_iter()
                .zip(l20_body.into_ref_iter())
            {
                process(simd, acc, l20, l10)
            }
            process(simd, acc_tail.rb_mut(), l20_tail, l10);
        }
    }
}

fn cholesky_in_place_left_looking_impl<E: ComplexField>(
    matrix: MatMut<'_, E>,
    regularization: LdltRegularization<'_, E>,
    parallelism: Parallelism,
    params: LdltDiagParams,
) -> usize {
    let mut matrix = matrix;
    let _ = parallelism;
    let _ = params;

    debug_assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );

    let n = matrix.nrows();

    if n == 0 {
        return 0;
    }

    let mut idx = 0;
    let arch = E::Simd::default();

    let eps = regularization.dynamic_regularization_epsilon.faer_abs();
    let delta = regularization.dynamic_regularization_delta.faer_abs();
    let has_eps = delta > E::Real::faer_zero();
    let mut dynamic_regularization_count = 0usize;
    loop {
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

        let (top_left, top_right, bottom_left, bottom_right) =
            matrix.rb_mut().split_at_mut(idx, idx);
        let l00 = top_left.into_const();
        let d0 = l00.diagonal().column_vector();
        let (_, l10, _, l20) = bottom_left.into_const().split_at(1, 0);
        let l10 = l10.row(0);

        let (mut a11, _, a21, _) = bottom_right.split_at_mut(1, 1);

        // reserve space for L10×D0
        let mut l10xd0 = top_right
            .submatrix_mut(0, 0, idx, 1)
            .transpose_mut()
            .row_mut(0);

        zipped!(l10xd0.rb_mut(), l10, d0.transpose()).for_each(
            |unzipped!(mut dst, src, factor)| {
                dst.write(
                    src.read()
                        .faer_scale_real(factor.read().faer_real().faer_inv()),
                )
            },
        );

        let l10xd0 = l10xd0.into_const();

        let mut d = a11
            .read(0, 0)
            .faer_sub(
                crate::linalg::matmul::inner_prod::inner_prod_with_conj_arch(
                    arch,
                    l10xd0.transpose(),
                    Conj::Yes,
                    l10.transpose(),
                    Conj::No,
                ),
            )
            .faer_real();

        // dynamic regularization code taken from clarabel.rs with modifications
        if has_eps {
            if let Some(signs) = regularization.dynamic_regularization_signs {
                if signs[idx] > 0 && d <= eps {
                    d = delta;
                    dynamic_regularization_count += 1;
                } else if signs[idx] < 0 && d >= eps.faer_neg() {
                    d = delta.faer_neg();
                    dynamic_regularization_count += 1;
                }
            } else if d.faer_abs() <= eps {
                if d < E::Real::faer_zero() {
                    d = delta.faer_neg();
                } else {
                    d = delta;
                }
                dynamic_regularization_count += 1;
            }
        }

        let d = d.faer_inv();
        a11.write(0, 0, E::faer_from_real(d));

        if idx + 1 == n {
            break;
        }

        let mut a21 = a21.col_mut(0);

        // A21 -= L20 × L10^H
        if a21.row_stride() == 1 {
            arch.dispatch(RankUpdate {
                a21: a21.rb_mut(),
                l20,
                l10: l10xd0,
            });
        } else {
            for j in 0..idx {
                let l20_col = l20.col(j);
                let l10_conj = l10xd0.read(j).faer_conj();

                zipped!(a21.rb_mut().as_2d_mut(), l20_col.as_2d()).for_each(
                    |unzipped!(mut dst, src)| {
                        dst.write(dst.read().faer_sub(src.read().faer_mul(l10_conj)))
                    },
                );
            }
        }

        zipped!(a21.rb_mut().as_2d_mut())
            .for_each(|unzipped!(mut x)| x.write(x.read().faer_scale_real(d)));

        idx += 1;
    }
    dynamic_regularization_count
}

/// LDLT factorization tuning parameters.
#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct LdltDiagParams {}

/// Computes the size and alignment of required workspace for performing a Cholesky decomposition.
pub fn raw_cholesky_in_place_req<E: Entity>(
    dim: usize,
    parallelism: Parallelism,
    params: LdltDiagParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    let _ = params;
    temp_mat_req::<E>(dim, dim)
}

// uses an out parameter for tail recursion
fn cholesky_in_place_impl<E: ComplexField>(
    count: &mut usize,
    matrix: MatMut<'_, E>,
    regularization: LdltRegularization<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: LdltDiagParams,
) {
    // right looking cholesky

    debug_assert!(matrix.nrows() == matrix.ncols());
    let mut matrix = matrix;
    let mut stack = stack;

    let n = matrix.nrows();
    if n < 32 {
        *count += cholesky_in_place_left_looking_impl(matrix, regularization, parallelism, params)
    } else {
        let block_size = Ord::min(n / 2, 128);
        let rem = n - block_size;
        let (mut l00, _, mut a10, mut a11) = matrix.rb_mut().split_at_mut(block_size, block_size);

        cholesky_in_place_impl(
            count,
            l00.rb_mut(),
            regularization,
            parallelism,
            stack.rb_mut(),
            params,
        );

        let l00 = l00.into_const();
        let d0 = l00.diagonal().column_vector();

        solve::solve_unit_lower_triangular_in_place(
            l00.conjugate(),
            a10.rb_mut().transpose_mut(),
            parallelism,
        );

        {
            // reserve space for L10×D0
            let (mut l10xd0, _) = temp_mat_uninit::<E>(rem, block_size, stack.rb_mut());
            let mut l10xd0 = l10xd0.as_mut();

            for j in 0..block_size {
                let l10xd0_col = l10xd0.rb_mut().col_mut(j);
                let a10_col = a10.rb_mut().col_mut(j);
                let d0_elem = d0.read(j);

                zipped!(l10xd0_col.as_2d_mut(), a10_col.as_2d_mut()).for_each(
                    |unzipped!(mut l10xd0_elem, mut a10_elem)| {
                        let a10_elem_read = a10_elem.read();
                        a10_elem.write(a10_elem_read.faer_mul(d0_elem));
                        l10xd0_elem.write(a10_elem_read);
                    },
                );
            }

            crate::linalg::matmul::triangular::matmul(
                a11.rb_mut(),
                BlockStructure::TriangularLower,
                a10.into_const(),
                BlockStructure::Rectangular,
                l10xd0.adjoint_mut().into_const(),
                BlockStructure::Rectangular,
                Some(E::faer_one()),
                E::faer_one().faer_neg(),
                parallelism,
            );
        }

        cholesky_in_place_impl(
            count,
            a11,
            LdltRegularization {
                dynamic_regularization_signs: regularization
                    .dynamic_regularization_signs
                    .map(|signs| &signs[block_size..]),
                dynamic_regularization_delta: regularization.dynamic_regularization_delta,
                dynamic_regularization_epsilon: regularization.dynamic_regularization_epsilon,
            },
            parallelism,
            stack,
            params,
        )
    }
}

/// Dynamic LDLT regularization.
/// Values below `epsilon` in absolute value, or with the wrong sign are set to `delta` with
/// their corrected sign.
#[derive(Copy, Clone, Debug)]
pub struct LdltRegularization<'a, E: ComplexField> {
    /// Expected signs for the diagonal at each step of the decomposition.
    pub dynamic_regularization_signs: Option<&'a [i8]>,
    /// Regularized value.
    pub dynamic_regularization_delta: E::Real,
    /// Regularization threshold.
    pub dynamic_regularization_epsilon: E::Real,
}

/// Info about the result of the LDLT factorization.
#[derive(Copy, Clone, Debug)]
pub struct LdltInfo {
    /// Number of pivots whose value or sign had to be corrected.
    pub dynamic_regularization_count: usize,
}

impl<E: ComplexField> Default for LdltRegularization<'_, E> {
    fn default() -> Self {
        Self {
            dynamic_regularization_signs: None,
            dynamic_regularization_delta: E::Real::faer_zero(),
            dynamic_regularization_epsilon: E::Real::faer_zero(),
        }
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
/// inverses of the diagonal elements of $D$ are stored on the diagonal.
///
/// The strictly upper triangular part of the matrix is clobbered and may be filled with garbage
/// values.
///
/// # Warning
///
/// The Cholesky decomposition with diagonal may have poor numerical stability properties when used
/// with non positive definite matrices. In the general case, it is recommended to first permute
/// (and conjugate when necessary) the rows and columns of the matrix using the permutation obtained
/// from [`faer::linalg::cholesky::compute_cholesky_permutation`](crate::linalg::cholesky::compute_cholesky_permutation).
///
/// # Panics
///
/// Panics if the input matrix is not square.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`raw_cholesky_in_place_req`]).
#[track_caller]
#[inline]
pub fn raw_cholesky_in_place<E: ComplexField>(
    matrix: MatMut<'_, E>,
    regularization: LdltRegularization<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
    params: LdltDiagParams,
) -> LdltInfo {
    assert!(matrix.ncols() == matrix.nrows());
    #[cfg(feature = "perf-warn")]
    if matrix.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(CHOLESKY_WARN) {
        if matrix.col_stride().unsigned_abs() == 1 {
            log::warn!(target: "faer_perf", "LDLT prefers column-major matrix. Found row-major matrix.");
        } else {
            log::warn!(target: "faer_perf", "LDLT prefers column-major matrix. Found matrix with generic strides.");
        }
    }

    let mut count = 0;
    cholesky_in_place_impl(
        &mut count,
        matrix,
        regularization,
        parallelism,
        stack,
        params,
    );
    LdltInfo {
        dynamic_regularization_count: count,
    }
}
