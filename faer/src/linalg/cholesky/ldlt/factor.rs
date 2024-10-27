use crate::{assert, internal_prelude::*};
use core::num::NonZero;
use faer_traits::RealValue;
use linalg::matmul::triangular::BlockStructure;
use pulp::Simd;

#[inline(always)]
#[math]
fn simd_cholesky_row_batch<'N, T: ComplexField, S: Simd>(
    simd: T::SimdCtx<S>,
    A: MatMut<'_, T, Dim<'N>, Dim<'N>, ContiguousFwd>,
    D: RowMut<'_, T, Dim<'N>>,

    start: IdxInc<'N>,

    is_llt: bool,
    regularize: bool,
    eps: T::Real,
    delta: T::Real,
    signs: Option<&Array<'N, i8>>,
) -> Result<usize, usize> {
    let mut A = A;
    let mut D = D;

    let n = A.ncols();

    ghost_tree!(ROW(HEAD, TAIL), {
        let (l![head, tail], (disjoint, ..)) = n.split(l![..start, ..], ROW);

        let simd = SimdCtx::<T, S>::new_force_mask(simd, tail.len());
        let (idx_head, indices, idx_tail) = simd.indices();
        assert!(idx_head.is_none());
        let Some(idx_tail) = idx_tail else { panic!() };

        let mut count = 0usize;

        for j in n.indices() {
            ghost_tree!(COL(LEFT, RIGHT), {
                let (l![left, right], (disjoint_col, ..)) = n.split(l![..j.to_incl(), j], COL);

                let l![A_0, mut Aj] = A.rb_mut().col_segments_mut(l![left, right], disjoint_col);
                let A_0 = A_0.rb();
                let A10 = A_0.row_segment(tail);

                let l![_, mut Aj] = Aj.rb_mut().row_segments_mut(l![head, tail], disjoint);

                {
                    let mut Aj = Aj.rb_mut();
                    let mut iter = indices.clone();
                    let i0 = iter.next();
                    let i1 = iter.next();
                    let i2 = iter.next();

                    match (i0, i1, i2) {
                        (None, None, None) => {
                            let mut Aij = simd.read(Aj.rb(), idx_tail);

                            for k in left {
                                let Ak = A10.col(left.local(k));

                                let D = real(D[k.local()]);
                                let D = if is_llt { one() } else { D };

                                let Ajk = simd.splat(&mul_real(conj(A_0[(j, left.local(k))]), -D));

                                let Aik = simd.read(Ak, idx_tail);
                                Aij = simd.mul_add(Ajk, Aik, Aij);
                            }
                            simd.write(Aj.rb_mut(), idx_tail, Aij);
                        }
                        (Some(i0), None, None) => {
                            let mut A0j = simd.read(Aj.rb(), i0);
                            let mut Aij = simd.read(Aj.rb(), idx_tail);

                            for k in left {
                                let Ak = A10.col(left.local(k));

                                let D = real(D[k.local()]);
                                let D = if is_llt { one() } else { D };

                                let Ajk = simd.splat(&mul_real(conj(A_0[(j, left.local(k))]), -D));

                                let A0k = simd.read(Ak, i0);
                                let Aik = simd.read(Ak, idx_tail);
                                A0j = simd.mul_add(Ajk, A0k, A0j);
                                Aij = simd.mul_add(Ajk, Aik, Aij);
                            }
                            simd.write(Aj.rb_mut(), i0, A0j);
                            simd.write(Aj.rb_mut(), idx_tail, Aij);
                        }
                        (Some(i0), Some(i1), None) => {
                            let mut A0j = simd.read(Aj.rb(), i0);
                            let mut A1j = simd.read(Aj.rb(), i1);
                            let mut Aij = simd.read(Aj.rb(), idx_tail);

                            for k in left {
                                let Ak = A10.col(left.local(k));

                                let D = real(D[k.local()]);
                                let D = if is_llt { one() } else { D };

                                let Ajk = simd.splat(&mul_real(conj(A_0[(j, left.local(k))]), -D));

                                let A0k = simd.read(Ak, i0);
                                let A1k = simd.read(Ak, i1);
                                let Aik = simd.read(Ak, idx_tail);
                                A0j = simd.mul_add(Ajk, A0k, A0j);
                                A1j = simd.mul_add(Ajk, A1k, A1j);
                                Aij = simd.mul_add(Ajk, Aik, Aij);
                            }
                            simd.write(Aj.rb_mut(), i0, A0j);
                            simd.write(Aj.rb_mut(), i1, A1j);
                            simd.write(Aj.rb_mut(), idx_tail, Aij);
                        }
                        (Some(i0), Some(i1), Some(i2)) => {
                            let mut A0j = simd.read(Aj.rb(), i0);
                            let mut A1j = simd.read(Aj.rb(), i1);
                            let mut A2j = simd.read(Aj.rb(), i2);
                            let mut Aij = simd.read(Aj.rb(), idx_tail);

                            for k in left {
                                let Ak = A10.col(left.local(k));

                                let D = real(D[k.local()]);
                                let D = if is_llt { one() } else { D };

                                let Ajk = simd.splat(&mul_real(conj(A_0[(j, left.local(k))]), -D));

                                let A0k = simd.read(Ak, i0);
                                let A1k = simd.read(Ak, i1);
                                let A2k = simd.read(Ak, i2);
                                let Aik = simd.read(Ak, idx_tail);
                                A0j = simd.mul_add(Ajk, A0k, A0j);
                                A1j = simd.mul_add(Ajk, A1k, A1j);
                                A2j = simd.mul_add(Ajk, A2k, A2j);
                                Aij = simd.mul_add(Ajk, Aik, Aij);
                            }
                            simd.write(Aj.rb_mut(), i0, A0j);
                            simd.write(Aj.rb_mut(), i1, A1j);
                            simd.write(Aj.rb_mut(), i2, A2j);
                            simd.write(Aj.rb_mut(), idx_tail, Aij);
                        }
                        _ => {
                            unreachable!();
                        }
                    }
                }

                let D = D.rb_mut().at_mut(j);

                if let Some(j_row) = tail.try_idx(*j) {
                    let mut diag = real(Aj[tail.from_global(j_row)]);

                    if regularize {
                        let sign = if is_llt {
                            1
                        } else {
                            if let Some(signs) = signs {
                                signs[j_row.local()]
                            } else {
                                0
                            }
                        };

                        let small_or_negative = diag <= eps;
                        let minus_small_or_positive = diag >= -eps;

                        if sign == 1 && small_or_negative {
                            diag = copy(delta);
                            count += 1;
                        } else if sign == -1 && minus_small_or_positive {
                            diag = neg(delta);
                        } else {
                            if small_or_negative && minus_small_or_positive {
                                if diag < zero() {
                                    diag = neg(delta);
                                } else {
                                    diag = copy(delta);
                                }
                            }
                        }
                    }

                    let j = j;
                    let diag = if is_llt {
                        if !(diag > zero()) {
                            *D = from_real(diag);
                            return Err(*j);
                        }
                        sqrt(diag)
                    } else {
                        copy(diag)
                    };

                    *D = from_real(diag);

                    if diag == zero() || !is_finite(diag) {
                        return Err(*j);
                    }
                }

                let diag = real(*D);

                {
                    let mut Aj = Aj.rb_mut();
                    let inv = simd.splat_real(&recip(diag));

                    for i in indices.clone() {
                        let mut Aij = simd.read(Aj.rb(), i);
                        Aij = simd.mul_real(Aij, inv);
                        simd.write(Aj.rb_mut(), i, Aij);
                    }
                    {
                        let mut Aij = simd.read(Aj.rb(), idx_tail);
                        Aij = simd.mul_real(Aij, inv);
                        simd.write(Aj.rb_mut(), idx_tail, Aij);
                    }
                }
            });
        }

        Ok(count)
    })
}

#[inline(always)]
#[math]
fn simd_cholesky_matrix<'N, T: ComplexField, S: Simd>(
    simd: T::SimdCtx<S>,
    A: MatMut<'_, T, Dim<'N>, Dim<'N>, ContiguousFwd>,
    D: RowMut<'_, T, Dim<'N>>,

    is_llt: bool,
    regularize: bool,
    eps: T::Real,
    delta: T::Real,
    signs: Option<&Array<'N, i8>>,
) -> Result<usize, usize> {
    let N = A.ncols();

    let blocksize = 4 * (size_of::<T::SimdVec<S>>() / size_of::<T>());

    let mut A = A;
    let mut D = D;

    let mut count = 0;

    let mut J = 0usize;
    while let Some(j) = N.try_check(J) {
        let J_next = N.advance(j, blocksize);

        ghost_tree!(FULL(HEAD), {
            let (l![head], _) = N.split(l![..J_next], FULL);

            let A = A.rb_mut().row_segment_mut(head).col_segment_mut(head);
            let D = D.rb_mut().col_segment_mut(head);

            let signs = signs.map(|signs| signs.segment(head));

            count += simd_cholesky_row_batch(
                simd,
                A,
                D,
                head.len().idx_inc(*j),
                is_llt,
                regularize,
                eps.clone(),
                delta.clone(),
                signs,
            )?;
        });

        J = *J_next;
    }

    Ok(count)
}

pub fn simd_cholesky<'N, T: ComplexField>(
    A: MatMut<'_, T, Dim<'N>, Dim<'N>>,
    D: RowMut<'_, T, Dim<'N>>,

    is_llt: bool,
    regularize: bool,
    eps: T::Real,
    delta: T::Real,
    signs: Option<&Array<'N, i8>>,
) -> Result<usize, usize> {
    struct Impl<'a, 'N, T: ComplexField> {
        A: MatMut<'a, T, Dim<'N>, Dim<'N>, ContiguousFwd>,
        D: RowMut<'a, T, Dim<'N>>,
        is_llt: bool,
        regularize: bool,
        eps: T::Real,
        delta: T::Real,
        signs: Option<&'a Array<'N, i8>>,
    }

    impl<'a, 'N, T: ComplexField> pulp::WithSimd for Impl<'a, 'N, T> {
        type Output = Result<usize, usize>;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self {
                A,
                D,
                is_llt,
                regularize,
                eps,
                delta,
                signs,
            } = self;
            let simd = T::simd_ctx(simd);
            if *A.nrows() > 0 {
                simd_cholesky_matrix(simd, A, D, is_llt, regularize, eps, delta, signs)
            } else {
                Ok(0)
            }
        }
    }

    let mut A = A;
    if const { T::SIMD_CAPABILITIES.is_simd() } {
        if let Some(A) = A.rb_mut().try_as_col_major_mut() {
            T::Arch::default().dispatch(Impl {
                A,
                D,
                is_llt,
                regularize,
                eps,
                delta,
                signs,
            })
        } else {
            cholesky_fallback(A, D, is_llt, regularize, eps.clone(), delta.clone(), signs)
        }
    } else {
        cholesky_fallback(A, D, is_llt, regularize, eps.clone(), delta.clone(), signs)
    }
}

#[math]
fn cholesky_fallback<'N, T: ComplexField>(
    A: MatMut<'_, T, Dim<'N>, Dim<'N>>,
    D: RowMut<'_, T, Dim<'N>>,

    is_llt: bool,
    regularize: bool,
    eps: T::Real,
    delta: T::Real,
    signs: Option<&Array<'N, i8>>,
) -> Result<usize, usize> {
    let N = A.nrows();
    let mut count = 0;
    let mut A = A;
    let mut D = D;

    for j in N.indices() {
        for i in j.to_incl().to(N.end()) {
            let mut sum = zero();
            for k in IdxInc::ZERO.to(j.excl()) {
                let D = real(D[k]);
                let D = if is_llt { one() } else { D };

                sum = sum + mul_real(conj(A[(j, k)]) * A[(i, k)], D);
            }
            A[(i, j)] = A[(i, j)] - sum;
        }

        let D = D.rb_mut().at_mut(j);
        let mut diag = real(A[(j, j)]);

        if regularize {
            let sign = if is_llt {
                1
            } else {
                if let Some(signs) = signs {
                    signs[j]
                } else {
                    0
                }
            };

            let small_or_negative = diag <= eps;
            let minus_small_or_positive = diag >= -eps;

            if sign == 1 && small_or_negative {
                diag = copy(delta);
                count += 1;
            } else if sign == -1 && minus_small_or_positive {
                diag = neg(delta);
            } else {
                if small_or_negative && minus_small_or_positive {
                    if diag < zero() {
                        diag = neg(delta);
                    } else {
                        diag = copy(delta);
                    }
                }
            }
        }

        let diag = if is_llt {
            if !(diag > zero()) {
                *D = from_real(diag);
                return Err(*j);
            }
            sqrt(diag)
        } else {
            copy(diag)
        };
        *D = from_real(diag);
        drop(D);

        if diag == zero() || !is_finite(diag) {
            return Err(*j);
        }

        let inv = recip(diag);

        for i in j.to_incl().to(N.end()) {
            A[(i, j)] = mul_real(A[(i, j)], inv);
        }
    }

    Ok(count)
}

#[math]
pub(crate) fn cholesky_recursion<'N, T: ComplexField>(
    A: MatMut<'_, T, Dim<'N>, Dim<'N>>,
    D: RowMut<'_, T, Dim<'N>>,

    recursion_threshold: usize,
    blocksize: usize,
    is_llt: bool,
    regularize: bool,
    eps: &T::Real,
    delta: &T::Real,
    signs: Option<&Array<'N, i8>>,
    par: Par,
) -> Result<usize, usize> {
    let N = A.ncols();
    if *N <= recursion_threshold {
        cholesky_fallback(A, D, is_llt, regularize, eps.clone(), delta.clone(), signs)
    } else {
        let mut count = 0;
        let blocksize = Ord::min(N.next_power_of_two() / 2, blocksize);
        let mut A = A;
        let mut D = D;

        let mut j_next = IdxInc::ZERO;
        while let Some(j) = N.try_check(*j_next) {
            j_next = N.advance(j, blocksize);

            ghost_tree!(FULL(HEAD, TAIL), {
                let (l![head, tail], (disjoint, ..)) = N.split(l![j.to_incl()..j_next, ..], FULL);

                let l![mut A_0, A_1] = A.rb_mut().col_segments_mut(l![head, tail], disjoint);
                let l![mut A00, mut A10] = A_0.rb_mut().row_segments_mut(l![head, tail], disjoint);
                let l![A01, mut A11] = A_1.row_segments_mut(l![head, tail], disjoint);

                let mut D0 = D.rb_mut().col_segment_mut(head);

                let mut L10xD0 = A01.transpose_mut();

                let signs = signs.map(|signs| signs.segment(head));

                match cholesky_recursion(
                    A00.rb_mut(),
                    D0.rb_mut(),
                    recursion_threshold,
                    blocksize,
                    is_llt,
                    regularize,
                    eps,
                    delta,
                    signs,
                    par,
                ) {
                    Ok(local_count) => count += local_count,
                    Err(fail_idx) => return Err(*j + fail_idx),
                }
                let A00 = A00.rb();

                if is_llt {
                    linalg::triangular_solve::solve_lower_triangular_in_place(
                        A00.conjugate(),
                        A10.rb_mut().transpose_mut(),
                        par,
                    )
                } else {
                    linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        A00.conjugate(),
                        A10.rb_mut().transpose_mut(),
                        par,
                    )
                }
                let mut A10 = A10.rb_mut();

                let (A10, L10xD0) = if is_llt {
                    (A10.rb(), A10.rb())
                } else {
                    for j in head {
                        let j = head.from_global(j);
                        let d = real(D0[j]);
                        let d = recip(d);

                        for i in tail {
                            let i = tail.from_global(i);
                            let a = copy(A10[(i, j)]);
                            A10[(i, j)] = mul_real(A10[(i, j)], d);
                            L10xD0[(i, j)] = a;
                        }
                    }

                    (A10.rb(), L10xD0.rb())
                };

                linalg::matmul::triangular::matmul(
                    A11.rb_mut(),
                    BlockStructure::TriangularLower,
                    Accum::Add,
                    A10,
                    BlockStructure::Rectangular,
                    L10xD0.adjoint(),
                    BlockStructure::Rectangular,
                    -one(),
                    par,
                );
            });
        }

        Ok(count)
    }
}

/// Dynamic LDLT regularization.
/// Values below `epsilon` in absolute value, or with the wrong sign are set to `delta` with
/// their corrected sign.
pub struct LdltRegularization<'a, T: ComplexField> {
    /// Expected signs for the diagonal at each step of the decomposition.
    pub dynamic_regularization_signs: Option<&'a [i8]>,
    /// Regularized value.
    pub dynamic_regularization_delta: RealValue<T>,
    /// Regularization threshold.
    pub dynamic_regularization_epsilon: RealValue<T>,
}

/// Info about the result of the LDLT factorization.
#[derive(Copy, Clone, Debug)]
pub struct LdltInfo {
    /// Number of pivots whose value or sign had to be corrected.
    pub dynamic_regularization_count: usize,
}

/// Error in the LDLT factorization.
#[derive(Copy, Clone, Debug)]
pub enum LdltError {
    ZeroPivot { index: usize },
}

impl<T: ComplexField> Default for LdltRegularization<'_, T> {
    fn default() -> Self {
        Self {
            dynamic_regularization_signs: None,
            dynamic_regularization_delta: zero(),
            dynamic_regularization_epsilon: zero(),
        }
    }
}

#[non_exhaustive]
pub struct LdltParams {
    pub recursion_threshold: NonZero<usize>,
    pub blocksize: NonZero<usize>,
}

impl Default for LdltParams {
    #[inline]
    fn default() -> Self {
        Self {
            recursion_threshold: NonZero::new(2).unwrap(),
            blocksize: NonZero::new(2).unwrap(),
        }
    }
}

#[inline]
pub fn cholesky_in_place_scratch<T: ComplexField>(
    dim: usize,
    par: Par,
    params: LdltParams,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    _ = params;
    temp_mat_scratch::<T>(dim, 1)
}

#[math]
pub fn cholesky_in_place<'N, T: ComplexField>(
    A: MatMut<'_, T, Dim<'N>, Dim<'N>>,
    regularization: LdltRegularization<'_, T>,
    par: Par,
    stack: &mut DynStack,
    params: LdltParams,
) -> Result<LdltInfo, LdltError> {
    let N = A.nrows();
    let mut D = unsafe { temp_mat_uninit(N, 1, stack).0 };
    let D = D.as_mat_mut();
    let mut D = D.col_mut(0).transpose_mut();
    let mut A = A;

    let ret = match cholesky_recursion(
        A.rb_mut(),
        D.rb_mut(),
        params.recursion_threshold.get(),
        params.blocksize.get(),
        false,
        regularization.dynamic_regularization_delta > zero()
            && regularization.dynamic_regularization_epsilon > zero(),
        &regularization.dynamic_regularization_epsilon,
        &regularization.dynamic_regularization_delta,
        regularization
            .dynamic_regularization_signs
            .map(|signs| Array::from_ref(signs, N)),
        par,
    ) {
        Ok(count) => Ok(LdltInfo {
            dynamic_regularization_count: count,
        }),
        Err(index) => Err(LdltError::ZeroPivot { index }),
    };
    let init = if let Err(LdltError::ZeroPivot { index }) = ret {
        N.idx(index).next()
    } else {
        N.end()
    };

    for i in IdxInc::ZERO.to(init) {
        A[(i, i)] = copy(D[i]);
    }

    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert, c64, stats::prelude::*, utils::approx::*, Mat, Row};

    #[test]
    fn test_simd_cholesky() {
        let rng = &mut StdRng::seed_from_u64(0);

        type T = c64;

        for n in 0..=64 {
            with_dim!(N, n);
            for f in [cholesky_fallback::<T>, simd_cholesky::<T>] {
                for llt in [true, false] {
                    let approx_eq = CwiseMat(ApproxEq {
                        abs_tol: 1e-12,
                        rel_tol: 1e-12,
                    });

                    let A = CwiseMatDistribution {
                        nrows: N,
                        ncols: N,
                        dist: ComplexDistribution::new(StandardNormal, StandardNormal),
                    }
                    .rand::<Mat<c64, Dim, Dim>>(rng);

                    let A = &A * &A.adjoint();
                    let A = A.as_ref().as_shape(N, N);

                    let mut L = A.cloned();
                    let mut L = L.as_mut();
                    let mut D = Row::zeros(N);
                    let mut D = D.as_mut();

                    f(L.rb_mut(), D.rb_mut(), llt, false, 0.0, 0.0, None).unwrap();

                    for j in N.indices() {
                        for i in IdxInc::ZERO.to(j.into()) {
                            L[(i, j)] = c64::ZERO;
                        }
                    }
                    let L = L.rb().as_dyn_stride();

                    if llt {
                        assert!(L * L.adjoint() ~ A);
                    } else {
                        assert!(L * D.as_diagonal() * L.adjoint() ~ A);
                    };
                }
            }
        }
    }

    #[test]
    fn test_cholesky() {
        let rng = &mut StdRng::seed_from_u64(0);

        type T = c64;

        for n in [2, 4, 8, 31, 127, 240] {
            with_dim!(N, n);

            for llt in [false, true] {
                let approx_eq = CwiseMat(ApproxEq {
                    abs_tol: 1e-12,
                    rel_tol: 1e-12,
                });

                let A = CwiseMatDistribution {
                    nrows: N,
                    ncols: N,
                    dist: ComplexDistribution::new(StandardNormal, StandardNormal),
                }
                .rand::<Mat<c64, Dim, Dim>>(rng);

                let A = &A * &A.adjoint();
                let A = A.as_ref();

                let mut L = A.cloned();
                let mut L = L.as_mut();
                let mut D = Row::zeros(N);
                let mut D = D.as_mut();

                cholesky_recursion(
                    L.rb_mut(),
                    D.rb_mut(),
                    32,
                    32,
                    llt,
                    false,
                    &0.0,
                    &0.0,
                    None,
                    Par::Seq,
                )
                .unwrap();

                for j in N.indices() {
                    for i in IdxInc::ZERO.to(j.into()) {
                        L[(i, j)] = c64::ZERO;
                    }
                }
                let L = L.rb().as_dyn_stride();

                if llt {
                    assert!(L * L.adjoint() ~ A);
                } else {
                    assert!(L * D.as_diagonal() * L.adjoint() ~ A);
                };
            }
        }
    }
}
