use assert2::{assert, debug_assert};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    mul::triangular::BlockStructure, solve, temp_mat_req, temp_mat_uninit, zipped, ComplexField,
    Conj, Entity, MatMut, MatRef, Parallelism, SimdCtx,
};
use reborrow::*;

pub(crate) struct RankUpdate<'a, E: ComplexField> {
    pub a21: MatMut<'a, E>,
    pub l20: MatRef<'a, E>,
    pub l10: MatRef<'a, E>,
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
        debug_assert_eq!(a21.ncols(), 1);
        debug_assert_eq!(l10.nrows(), 1);

        let m = l20.nrows();
        let n = l20.ncols();

        if m == 0 {
            return;
        }

        let lane_count = core::mem::size_of::<E::SimdUnit<S>>() / core::mem::size_of::<E::Unit>();
        let prefix = m % lane_count;

        let acc = a21.as_ptr();
        let acc = E::map(
            acc,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
        );

        let (mut acc_head, acc_tail) = E::unzip(E::map(
            acc,
            #[inline(always)]
            |slice| slice.split_at_mut(prefix),
        ));
        let mut acc_tail = faer_core::simd::slice_as_mut_simd::<E, S>(acc_tail).0;

        for j in 0..n {
            let l10_ = unsafe { l10.read_unchecked(0, j).neg().conj() };
            let l10 = E::simd_splat(simd, l10_);

            let l20 = E::map(
                l20.ptr_at(0, j),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
            );
            let (l20_head, l20_tail) = E::unzip(E::map(
                l20,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));
            let l20_tail = faer_core::simd::slice_as_simd::<E, S>(l20_tail).0;

            for (acc, l20) in
                E::into_iter(E::rb_mut(E::as_mut(&mut acc_head))).zip(E::into_iter(l20_head))
            {
                let mut acc_ = E::from_units(E::deref(E::rb(E::as_ref(&acc))));
                let l20 = E::from_units(E::deref(l20));
                acc_ = E::simd_scalar_mul_adde(simd, l10_, l20, acc_);
                E::map(
                    E::zip(acc, acc_.into_units()),
                    #[inline(always)]
                    |(acc, acc_)| *acc = acc_,
                );
            }

            for (acc, l20) in
                E::into_iter(E::rb_mut(E::as_mut(&mut acc_tail))).zip(E::into_iter(l20_tail))
            {
                let mut acc_ = E::deref(E::rb(E::as_ref(&acc)));
                let l20 = E::deref(l20);
                acc_ = E::simd_mul_adde(simd, E::copy(&l10), E::copy(&l20), acc_);
                E::map(
                    E::zip(acc, acc_),
                    #[inline(always)]
                    |(acc, acc_)| *acc = acc_,
                );
            }
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

    let eps = regularization.dynamic_regularization_epsilon.abs();
    let delta = regularization.dynamic_regularization_delta.abs();
    let has_eps = delta > E::Real::zero();
    let mut dynamic_regularization_count = 0usize;
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

        let [top_left, top_right, bottom_left, bottom_right] = matrix.rb_mut().split_at(idx, idx);
        let l00 = top_left.into_const();
        let d0 = l00.diagonal();
        let [_, l10, _, l20] = bottom_left.into_const().split_at(block_size, 0);
        let [mut a11, _, a21, _] = bottom_right.split_at(block_size, block_size);

        // reserve space for L10×D0
        let mut l10xd0 = top_right.submatrix(0, 0, idx, block_size).transpose();

        zipped!(l10xd0.rb_mut(), l10, d0.transpose())
            .for_each(|mut dst, src, factor| dst.write(src.read().mul(factor.read())));

        let l10xd0 = l10xd0.into_const();

        let mut d = a11
            .read(0, 0)
            .sub(faer_core::mul::inner_prod::inner_prod_with_conj_arch(
                arch,
                l10xd0.row(0).transpose(),
                Conj::Yes,
                l10.row(0).transpose(),
                Conj::No,
            ))
            .real();

        // dynamic regularization code taken from clarabel.rs with modifications
        if has_eps {
            if let Some(signs) = regularization.dynamic_regularization_signs {
                if signs[idx] > 0 && d <= delta {
                    d = eps;
                    dynamic_regularization_count += 1;
                } else if signs[idx] < 0 && d >= delta.neg() {
                    d = eps.neg();
                    dynamic_regularization_count += 1;
                }
            } else if d.abs() <= delta {
                if d < E::Real::zero() {
                    d = eps.neg();
                    dynamic_regularization_count += 1;
                } else {
                    d = eps;
                    dynamic_regularization_count += 1;
                }
            }
        }

        a11.write(0, 0, E::from_real(d));

        if idx + block_size == n {
            break;
        }

        let ld11 = a11.into_const();
        let l11 = ld11;

        let mut a21 = a21.col(0);

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
                let l10_conj = l10xd0.read(0, j).conj();

                zipped!(a21.rb_mut(), l20_col)
                    .for_each(|mut dst, src| dst.write(dst.read().sub(src.read().mul(l10_conj))));
            }
        }

        let r = l11.read(0, 0).real().inv();
        zipped!(a21.rb_mut()).for_each(|mut x| x.write(x.read().scale_real(r)));

        idx += block_size;
    }
    dynamic_regularization_count
}

#[derive(Default, Copy, Clone)]
#[non_exhaustive]
pub struct LdltDiagParams {}

/// Computes the size and alignment of required workspace for performing a Cholesky
/// decomposition with partial pivoting.
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
        let [mut l00, _, mut a10, mut a11] = matrix.rb_mut().split_at(block_size, block_size);

        cholesky_in_place_impl(
            count,
            l00.rb_mut(),
            regularization,
            parallelism,
            stack.rb_mut(),
            params,
        );

        let l00 = l00.into_const();
        let d0 = l00.diagonal();

        solve::solve_unit_lower_triangular_in_place(
            l00.conjugate(),
            a10.rb_mut().transpose(),
            parallelism,
        );

        {
            // reserve space for L10×D0
            let (mut l10xd0, _) = temp_mat_uninit(rem, block_size, stack.rb_mut());
            let mut l10xd0 = l10xd0.as_mut();

            for j in 0..block_size {
                let l10xd0_col = l10xd0.rb_mut().col(j);
                let a10_col = a10.rb_mut().col(j);
                let d0_elem = d0.read(j, 0);

                let d0_elem_inv = d0_elem.inv();

                zipped!(l10xd0_col, a10_col).for_each(|mut l10xd0_elem, mut a10_elem| {
                    let a10_elem_read = a10_elem.read();
                    a10_elem.write(a10_elem_read.mul(d0_elem_inv));
                    l10xd0_elem.write(a10_elem_read);
                });
            }

            faer_core::mul::triangular::matmul(
                a11.rb_mut(),
                BlockStructure::TriangularLower,
                a10.into_const(),
                BlockStructure::Rectangular,
                l10xd0.adjoint().into_const(),
                BlockStructure::Rectangular,
                Some(E::one()),
                E::one().neg(),
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

#[derive(Copy, Clone, Debug)]
pub struct LdltRegularization<'a, E: ComplexField> {
    pub dynamic_regularization_signs: Option<&'a [i8]>,
    pub dynamic_regularization_delta: E::Real,
    pub dynamic_regularization_epsilon: E::Real,
}

impl<E: ComplexField> Default for LdltRegularization<'_, E> {
    fn default() -> Self {
        Self {
            dynamic_regularization_signs: None,
            dynamic_regularization_delta: E::Real::zero(),
            dynamic_regularization_epsilon: E::Real::zero(),
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
/// diagonal elements of $D$ are stored on the diagonal.
///
/// The strictly upper triangular part of the matrix is clobbered and may be filled with garbage
/// values.
///
/// # Warning
///
/// The Cholesky decomposition with diagonal may have poor numerical stability properties when used
/// with non positive definite matrices. In the general case, it is recommended to first permute
/// (and conjugate when necessary) the rows and columns of the matrix using the permutation obtained
/// from [`crate::compute_cholesky_permutation`].
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
) -> usize {
    assert!(
        matrix.ncols() == matrix.nrows(),
        "only square matrices can be decomposed into cholesky factors",
    );
    let mut count = 0;
    cholesky_in_place_impl(
        &mut count,
        matrix,
        regularization,
        parallelism,
        stack,
        params,
    );
    count
}
