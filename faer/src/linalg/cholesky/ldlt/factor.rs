use crate::{assert, internal_prelude::*};
use core::num::NonZero;
use faer_traits::RealValue;
use linalg::matmul::triangular::BlockStructure;
use pulp::Simd;

#[inline(always)]
#[math]
fn simd_cholesky_row_batch<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd>(
    simd: T::SimdCtx<S>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>, ContiguousFwd>,
    D: RowMut<'_, C, T, Dim<'N>>,

    start: IdxInc<'N>,

    is_llt: bool,
    regularize: bool,
    eps: <C::Real as Container>::Of<&T::RealUnit>,
    delta: <C::Real as Container>::Of<&T::RealUnit>,
    signs: Option<&Array<'N, i8>>,
) -> Result<usize, usize> {
    help!(C);
    help2!(C::Real);
    let mut A = A;
    let mut D = D;

    let n = A.ncols();

    ghost_tree2!(ROW(HEAD, TAIL), {
        let (list![head, tail], disjoint) = n.split(list![..start, ..], ROW);

        let simd = SimdCtx::<C, T, S>::new_force_mask(simd, tail.len());
        let (idx_head, indices, idx_tail) = simd.indices();
        assert!(idx_head.is_none());
        let Some(idx_tail) = idx_tail else { panic!() };

        let ctx = &Ctx::<C, T>(T::ctx_from_simd(&simd.ctx).0);

        let mut count = 0usize;

        for j in n.indices() {
            ghost_tree2!(COL(LEFT, RIGHT), {
                let (list![left, right], disjoint_col) = n.split(list![..j.to_incl(), j], COL);

                let list![A_0, mut Aj] = A
                    .rb_mut()
                    .any_col_segments_mut(list![left, right], disjoint_col);
                let A_0 = A_0.rb();
                let A10 = A_0.row_segments(head, tail).1;

                let list![_, mut Aj] = Aj
                    .rb_mut()
                    .any_row_segments_mut(list![head, tail], disjoint);

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

                                let D = math(real(D[k.local()]));
                                let D = if is_llt { math.re.one() } else { D };

                                let Ajk = simd.splat(as_ref!(math(mul_real(
                                    conj(A_0[(j, left.local(k))]),
                                    re.neg(D)
                                ))));

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

                                let D = math(real(D[k.local()]));
                                let D = if is_llt { math.re.one() } else { D };

                                let Ajk = simd.splat(as_ref!(math(mul_real(
                                    conj(A_0[(j, left.local(k))]),
                                    re.neg(D)
                                ))));

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

                                let D = math(real(D[k.local()]));
                                let D = if is_llt { math.re.one() } else { D };

                                let Ajk = simd.splat(as_ref!(math(mul_real(
                                    conj(A_0[(j, left.local(k))]),
                                    re.neg(D)
                                ))));

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

                                let D = math(real(D[k.local()]));
                                let D = if is_llt { math.re.one() } else { D };

                                let Ajk = simd.splat(as_ref!(math(mul_real(
                                    conj(A_0[(j, left.local(k))]),
                                    re.neg(D)
                                ))));

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

                let mut D = D.rb_mut().at_mut(j);

                if let Some(j_row) = tail.try_idx(*j) {
                    let mut diag = math(real(Aj[tail.from_global(j_row)]));

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

                        let small_or_negative = math.re(diag <= eps);
                        let minus_small_or_positive = math.re(diag >= -eps);

                        if sign == 1 && small_or_negative {
                            diag = math.re.copy(delta);
                            count += 1;
                        } else if sign == -1 && minus_small_or_positive {
                            diag = math.re.neg(delta);
                        } else {
                            if small_or_negative && minus_small_or_positive {
                                if math.re.lt_zero(diag) {
                                    diag = math.re.neg(delta);
                                } else {
                                    diag = math.re.copy(delta);
                                }
                            }
                        }
                    }

                    let j = j;
                    let diag = if is_llt {
                        if !math.re.gt_zero(diag) {
                            write1!(D, math.from_real(diag));
                            return Err(*j);
                        }
                        math.re.sqrt(diag)
                    } else {
                        math.re.copy(diag)
                    };

                    write1!(D, math.from_real(diag));

                    if math.re.is_zero(diag) || !math.re.is_finite(diag) {
                        return Err(*j);
                    }
                }

                let diag = math(real(D));

                {
                    let mut Aj = Aj.rb_mut();
                    let inv = simd.splat_real(as_ref2!(math.re.recip(diag)));

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
fn simd_cholesky_matrix<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd>(
    simd: T::SimdCtx<S>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>, ContiguousFwd>,
    D: RowMut<'_, C, T, Dim<'N>>,

    is_llt: bool,
    regularize: bool,
    eps: <C::Real as Container>::Of<&T::RealUnit>,
    delta: <C::Real as Container>::Of<&T::RealUnit>,
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
        help!(C::Real);

        ghost_tree2!(FULL(HEAD), {
            let (list![head], _) = N.split(list![..J_next], FULL);

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
                copy!(eps),
                copy!(delta),
                signs,
            )?;
        });

        J = *J_next;
    }

    Ok(count)
}

pub fn simd_cholesky<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    D: RowMut<'_, C, T, Dim<'N>>,

    is_llt: bool,
    regularize: bool,
    eps: <C::Real as Container>::Of<&T::RealUnit>,
    delta: <C::Real as Container>::Of<&T::RealUnit>,
    signs: Option<&Array<'N, i8>>,
) -> Result<usize, usize> {
    struct Impl<'a, 'N, C: ComplexContainer, T: ComplexField<C>> {
        ctx: &'a Ctx<C, T>,
        A: MatMut<'a, C, T, Dim<'N>, Dim<'N>, ContiguousFwd>,
        D: RowMut<'a, C, T, Dim<'N>>,
        is_llt: bool,
        regularize: bool,
        eps: <C::Real as Container>::Of<&'a T::RealUnit>,
        delta: <C::Real as Container>::Of<&'a T::RealUnit>,
        signs: Option<&'a Array<'N, i8>>,
    }

    impl<'a, 'N, C: ComplexContainer, T: ComplexField<C>> pulp::WithSimd for Impl<'a, 'N, C, T> {
        type Output = Result<usize, usize>;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self {
                ctx,
                A,
                D,
                is_llt,
                regularize,
                eps,
                delta,
                signs,
            } = self;
            let simd = T::simd_ctx(ctx, simd);
            if *A.nrows() > 0 {
                simd_cholesky_matrix(simd, A, D, is_llt, regularize, eps, delta, signs)
            } else {
                Ok(0)
            }
        }
    }

    help!(C::Real);
    let mut A = A;
    if const { T::SIMD_CAPABILITIES.is_simd() } {
        if let Some(A) = A.rb_mut().try_as_col_major_mut() {
            T::Arch::default().dispatch(Impl {
                ctx,
                A,
                D,
                is_llt,
                regularize,
                eps: rb!(eps),
                delta: rb!(delta),
                signs,
            })
        } else {
            cholesky_fallback(ctx, A, D, is_llt, regularize, eps, delta, signs)
        }
    } else {
        cholesky_fallback(ctx, A, D, is_llt, regularize, eps, delta, signs)
    }
}

#[math]
fn cholesky_fallback<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    D: RowMut<'_, C, T, Dim<'N>>,

    is_llt: bool,
    regularize: bool,
    eps: <C::Real as Container>::Of<&T::RealUnit>,
    delta: <C::Real as Container>::Of<&T::RealUnit>,
    signs: Option<&Array<'N, i8>>,
) -> Result<usize, usize> {
    let N = A.nrows();
    let mut count = 0;
    let mut A = A;
    let mut D = D;
    help!(C);

    for j in N.indices() {
        for i in j.to_incl().to(N.end()) {
            let mut sum = math.zero();
            for k in zero().to(j.excl()) {
                let D = math(real(D[k]));
                let D = if is_llt { math.re.one() } else { D };

                sum = math(sum + mul_real(conj(A[(j, k)]) * A[(i, k)], D));
            }
            write1!(A[(i, j)] = math(A[(i, j)] - sum));
        }

        let mut D = D.rb_mut().at_mut(j);
        let mut diag = math(real(A[(j, j)]));

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

            let small_or_negative = math.re(diag <= eps);
            let minus_small_or_positive = math.re(diag >= -eps);

            if sign == 1 && small_or_negative {
                diag = math.re.copy(delta);
                count += 1;
            } else if sign == -1 && minus_small_or_positive {
                diag = math.re.neg(delta);
            } else {
                if small_or_negative && minus_small_or_positive {
                    if math.re.lt_zero(diag) {
                        diag = math.re.neg(delta);
                    } else {
                        diag = math.re.copy(delta);
                    }
                }
            }
        }

        let diag = if is_llt {
            if !math.re.gt_zero(diag) {
                write1!(D, math.from_real(diag));
                return Err(*j);
            }
            math.re.sqrt(diag)
        } else {
            math.re.copy(diag)
        };
        write1!(D, math.from_real(diag));
        drop(D);

        if math.re.is_zero(diag) || !math.re.is_finite(diag) {
            return Err(*j);
        }

        let inv = math.re.recip(diag);

        for i in j.to_incl().to(N.end()) {
            write1!(A[(i, j)] = math(mul_real(A[(i, j)], inv)));
        }
    }

    Ok(count)
}

#[math]
pub(crate) fn cholesky_recursion<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    D: RowMut<'_, C, T, Dim<'N>>,

    recursion_threshold: usize,
    blocksize: usize,
    is_llt: bool,
    regularize: bool,
    eps: <C::Real as Container>::Of<&T::RealUnit>,
    delta: <C::Real as Container>::Of<&T::RealUnit>,
    signs: Option<&Array<'N, i8>>,
    par: Parallelism,
) -> Result<usize, usize> {
    let N = A.ncols();
    if *N <= recursion_threshold {
        cholesky_fallback(ctx, A, D, is_llt, regularize, eps, delta, signs)
    } else {
        let mut count = 0;
        let blocksize = Ord::min(N.next_power_of_two() / 2, blocksize);
        let mut A = A;
        let mut D = D;
        help!(C);
        help2!(C::Real);

        let mut j_next = zero();
        while let Some(j) = N.try_check(*j_next) {
            j_next = N.advance(j, blocksize);

            ghost_tree2!(FULL(HEAD, TAIL), {
                let (list![head, tail], disjoint) = N.split(list![j.to_incl()..j_next, ..], FULL);

                let list![mut A_0, A_1] =
                    A.rb_mut().any_col_segments_mut(list![head, tail], disjoint);
                let list![mut A00, mut A10] = A_0
                    .rb_mut()
                    .any_row_segments_mut(list![head, tail], disjoint);
                let list![A01, mut A11] = A_1.any_row_segments_mut(list![head, tail], disjoint);

                let mut D0 = D.rb_mut().col_segment_mut(head);

                let mut L10xD0 = A01.transpose_mut();

                let signs = signs.map(|signs| signs.segment(head));

                match cholesky_recursion(
                    ctx,
                    A00.rb_mut(),
                    D0.rb_mut(),
                    recursion_threshold,
                    blocksize,
                    is_llt,
                    regularize,
                    rb2!(eps),
                    rb2!(delta),
                    signs,
                    par,
                ) {
                    Ok(local_count) => count += local_count,
                    Err(fail_idx) => return Err(*j + fail_idx),
                }
                let A00 = A00.rb();

                if is_llt {
                    linalg::triangular_solve::solve_lower_triangular_in_place(
                        ctx,
                        A00.conjugate(),
                        A10.rb_mut().transpose_mut(),
                        par,
                    )
                } else {
                    linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        ctx,
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
                        let d = math(real(D0[j]));
                        let d = math.re.recip(d);

                        for i in tail {
                            let i = tail.from_global(i);
                            let a = math(copy(A10[(i, j)]));
                            write1!(A10[(i, j)] = math(mul_real(A10[(i, j)], d)));
                            write1!(L10xD0[(i, j)] = a);
                        }
                    }

                    (A10.rb(), L10xD0.rb())
                };

                linalg::matmul::triangular::matmul(
                    ctx,
                    A11.rb_mut(),
                    BlockStructure::TriangularLower,
                    Some(as_ref!(math.one())),
                    A10,
                    BlockStructure::Rectangular,
                    L10xD0.adjoint(),
                    BlockStructure::Rectangular,
                    as_ref!(math(-one())),
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
pub struct LdltRegularization<'a, C: ComplexContainer, T: ComplexField<C>> {
    /// Expected signs for the diagonal at each step of the decomposition.
    pub dynamic_regularization_signs: Option<&'a [i8]>,
    /// Regularized value.
    pub dynamic_regularization_delta: RealValue<C, T>,
    /// Regularization threshold.
    pub dynamic_regularization_epsilon: RealValue<C, T>,
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

impl<C: ComplexContainer, T: ComplexField<C>> LdltRegularization<'_, C, T> {
    #[math]
    pub fn default_with(ctx: &Ctx<C, T>) -> Self {
        Self {
            dynamic_regularization_signs: None,
            dynamic_regularization_delta: math.re(zero()),
            dynamic_regularization_epsilon: math.re(zero()),
        }
    }
}

impl<C: ComplexContainer, T: ComplexField<C, MathCtx: Default>> Default
    for LdltRegularization<'_, C, T>
{
    fn default() -> Self {
        Self::default_with(&ctx())
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
pub fn cholesky_in_place_scratch<C: ComplexContainer, T: ComplexField<C>>(
    dim: usize,
    par: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    _ = par;
    temp_mat_scratch::<C, T>(dim, 1)
}

#[math]
pub fn cholesky_in_place<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>>,
    regularization: LdltRegularization<'_, C, T>,
    par: Parallelism,
    stack: &mut DynStack,
    params: LdltParams,
) -> Result<LdltInfo, LdltError> {
    let N = A.nrows();
    let mut D = unsafe { temp_mat_uninit(ctx, N, 1, stack).0 };
    let D = D.as_mat_mut();
    let mut D = D.col_mut(0).transpose_mut();
    let mut A = A;

    help!(C::Real);
    let ret = match cholesky_recursion(
        ctx,
        A.rb_mut(),
        D.rb_mut(),
        params.recursion_threshold.get(),
        params.blocksize.get(),
        false,
        math.gt_zero(regularization.dynamic_regularization_delta)
            && math.gt_zero(regularization.dynamic_regularization_epsilon),
        as_ref!(regularization.dynamic_regularization_epsilon),
        as_ref!(regularization.dynamic_regularization_delta),
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

    help2!(C);
    for i in zero().to(init) {
        write2!(A[(i, i)] = math(copy(D[i])));
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

        type C = faer_traits::Unit;
        type T = c64;

        for n in 0..=64 {
            with_dim!(N, n);
            for f in [cholesky_fallback::<C, T>, simd_cholesky::<C, T>] {
                for llt in [true, false] {
                    let approx_eq = CwiseMat(ApproxEq {
                        ctx: ctx::<Ctx<C, T>>(),
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
                    let mut D = Row::zeros_with_ctx(&default(), N);
                    let mut D = D.as_mut();

                    f(
                        &default(),
                        L.rb_mut(),
                        D.rb_mut(),
                        llt,
                        false,
                        &0.0,
                        &0.0,
                        None,
                    )
                    .unwrap();

                    for j in N.indices() {
                        for i in zero().to(j.into()) {
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

        type C = faer_traits::Unit;
        type T = c64;

        for n in [2, 4, 8, 31, 127, 240] {
            with_dim!(N, n);

            for llt in [false, true] {
                let approx_eq = CwiseMat(ApproxEq {
                    ctx: ctx::<Ctx<C, T>>(),
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
                let mut D = Row::zeros_with_ctx(&default(), N);
                let mut D = D.as_mut();

                cholesky_recursion(
                    &default(),
                    L.rb_mut(),
                    D.rb_mut(),
                    32,
                    32,
                    llt,
                    false,
                    &0.0,
                    &0.0,
                    None,
                    Parallelism::None,
                )
                .unwrap();

                for j in N.indices() {
                    for i in zero().to(j.into()) {
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
