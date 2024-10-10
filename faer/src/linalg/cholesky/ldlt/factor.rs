use crate::internal_prelude::*;
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
    ghost_tree!(MAT(HEAD, TAIL), {
        help!(C);
        help2!(C::Real);

        let mut A = A;
        let mut D = D;

        let (mat, MAT) = A.ncols().full(MAT);
        let start = mat.from_local_inc(start);

        let (disjoint, head, tail, _, _) = mat.split_inc(start, MAT.HEAD, MAT.TAIL);

        let simd = SimdCtx::<C, T, S>::new_force_mask(simd, tail.len());
        let indices = simd.indices();
        let ctx = &Ctx::<C, T>(T::ctx_from_simd(&simd.ctx).0);

        let mut count = 0usize;

        for j in mat {
            ghost_tree!(COL(LEFT, RIGHT), {
                let (mat, COL) = mat.len().full(COL);

                let (disjoint_col, j, left, right, _, _) = mat.split(j, COL.LEFT, COL.RIGHT);

                let (A_0, mut A_1) = A.rb_mut().col_segments_mut(left, right, disjoint_col);
                let A_0 = A_0.rb();
                let A10 = A_0.row_segments(head, tail).1;

                let mut A11 = A_1.rb_mut().row_segments_mut(head, tail, disjoint).1;

                let mut Aj = A11.rb_mut().col_mut(right.from_global(j));
                {
                    let mut Aj = Aj.rb_mut().as_array_mut();
                    let mut iter = indices.clone();
                    let i0 = iter.next();
                    let i1 = iter.next();
                    let i2 = iter.next();

                    match (i0, i1, i2) {
                        (None, None, None) => {
                            let mut Aij = simd.read_tail(rb!(Aj));

                            for k in left {
                                let Ak = A10.col(left.from_global(k)).as_array();

                                let Ajk = simd.splat(as_ref!(math(
                                    -D[mat.from_global(k)]
                                        * conj(A_0[(mat.from_global(j), left.from_global(k))])
                                )));

                                let Aik = simd.read_tail(rb!(Ak));
                                Aij = simd.mul_add(Ajk, Aik, Aij);
                            }
                            simd.write_tail(rb_mut!(Aj), Aij);
                        }
                        (Some(i0), None, None) => {
                            let mut A0j = simd.read(rb!(Aj), i0);
                            let mut Aij = simd.read_tail(rb!(Aj));

                            for k in left {
                                let Ak = A10.col(left.from_global(k)).as_array();

                                let Ajk = simd.splat(as_ref!(math(
                                    -D[mat.from_global(k)]
                                        * conj(A_0[(mat.from_global(j), left.from_global(k))])
                                )));

                                let A0k = simd.read(rb!(Ak), i0);
                                let Aik = simd.read_tail(rb!(Ak));
                                A0j = simd.mul_add(Ajk, A0k, A0j);
                                Aij = simd.mul_add(Ajk, Aik, Aij);
                            }
                            simd.write(rb_mut!(Aj), i0, A0j);
                            simd.write_tail(rb_mut!(Aj), Aij);
                        }
                        (Some(i0), Some(i1), None) => {
                            let mut A0j = simd.read(rb!(Aj), i0);
                            let mut A1j = simd.read(rb!(Aj), i1);
                            let mut Aij = simd.read_tail(rb!(Aj));

                            for k in left {
                                let Ak = A10.col(left.from_global(k)).as_array();

                                let Ajk = simd.splat(as_ref!(math(
                                    -D[mat.from_global(k)]
                                        * conj(A_0[(mat.from_global(j), left.from_global(k))])
                                )));

                                let A0k = simd.read(rb!(Ak), i0);
                                let A1k = simd.read(rb!(Ak), i1);
                                let Aik = simd.read_tail(rb!(Ak));
                                A0j = simd.mul_add(Ajk, A0k, A0j);
                                A1j = simd.mul_add(Ajk, A1k, A1j);
                                Aij = simd.mul_add(Ajk, Aik, Aij);
                            }
                            simd.write(rb_mut!(Aj), i0, A0j);
                            simd.write(rb_mut!(Aj), i1, A1j);
                            simd.write_tail(rb_mut!(Aj), Aij);
                        }
                        (Some(i0), Some(i1), Some(i2)) => {
                            let mut A0j = simd.read(rb!(Aj), i0);
                            let mut A1j = simd.read(rb!(Aj), i1);
                            let mut A2j = simd.read(rb!(Aj), i2);
                            let mut Aij = simd.read_tail(rb!(Aj));

                            for k in left {
                                let Ak = A10.col(left.from_global(k)).as_array();

                                let Ajk = simd.splat(as_ref!(math(
                                    -D[mat.from_global(k)]
                                        * conj(A_0[(mat.from_global(j), left.from_global(k))])
                                )));

                                let A0k = simd.read(rb!(Ak), i0);
                                let A1k = simd.read(rb!(Ak), i1);
                                let A2k = simd.read(rb!(Ak), i2);
                                let Aik = simd.read_tail(rb!(Ak));
                                A0j = simd.mul_add(Ajk, A0k, A0j);
                                A1j = simd.mul_add(Ajk, A1k, A1j);
                                A2j = simd.mul_add(Ajk, A2k, A2j);
                                Aij = simd.mul_add(Ajk, Aik, Aij);
                            }
                            simd.write(rb_mut!(Aj), i0, A0j);
                            simd.write(rb_mut!(Aj), i1, A1j);
                            simd.write(rb_mut!(Aj), i2, A2j);
                            simd.write_tail(rb_mut!(Aj), Aij);
                        }
                        _ => {
                            for k in left {
                                let Ak = A10.col(left.from_global(k)).as_array();

                                let Ajk = simd.splat(as_ref!(math(
                                    -D[mat.from_global(k)]
                                        * conj(A_0[(mat.from_global(j), left.from_global(k))])
                                )));

                                for i in indices.clone() {
                                    let Aik = simd.read(rb!(Ak), i);
                                    let mut Aij = simd.read(rb!(Aj), i);
                                    Aij = simd.mul_add(Ajk, Aik, Aij);
                                    simd.write(rb_mut!(Aj), i, Aij);
                                }
                                {
                                    let Aik = simd.read_tail(rb!(Ak));
                                    let mut Aij = simd.read_tail(rb!(Aj));
                                    Aij = simd.mul_add(Ajk, Aik, Aij);
                                    simd.write_tail(rb_mut!(Aj), Aij);
                                }
                            }
                        }
                    }
                }

                let mut D = D.rb_mut().at_mut(mat.from_global(j));

                if let Some(j_row) = tail.try_idx(*j) {
                    let mut diag = math(real(Aj[tail.from_global(j_row)]));

                    if regularize {
                        let sign = if is_llt {
                            1
                        } else {
                            if let Some(signs) = signs {
                                signs[mat.from_global(j_row)]
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

                    let j = mat.from_global(j);
                    let diag = if is_llt {
                        if !math.re.gt_zero(diag) {
                            write1!(D, math.from_real(diag));
                            return Err(*j);
                        }
                        math.re.sqrt(diag)
                    } else {
                        math.re.copy(diag)
                    };

                    if is_llt {
                        write1!(D, math.from_real(diag));
                        // write1!(D, math.one());
                    } else {
                        write1!(D, math.from_real(diag));
                    }

                    if math.re.is_zero(diag) || !math.re.is_finite(diag) {
                        return Err(*j);
                    }
                }

                let diag = math(real(D));

                {
                    let mut Aj = Aj.rb_mut().as_array_mut();
                    let inv = simd.splat_real(as_ref2!(math.re.recip(diag)));

                    for i in indices.clone() {
                        let mut Aij = simd.read(rb!(Aj), i);
                        Aij = simd.mul_real(Aij, inv);
                        simd.write(rb_mut!(Aj), i, Aij);
                    }
                    {
                        let mut Aij = simd.read_tail(rb!(Aj));
                        Aij = simd.mul_real(Aij, inv);
                        simd.write_tail(rb_mut!(Aj), Aij);
                    }
                }
                if is_llt {
                    write1!(D, math.one());
                }
            });
        }

        Ok(count)
    })
}

#[inline(always)]
#[math]
fn simd_cholesky<'N, C: ComplexContainer, T: ComplexField<C>, S: Simd>(
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

        ghost_tree!(FULL(HEAD, TAIL), {
            let (full, FULL) = N.full(FULL);

            let (disjoint, head, tail, _, _) =
                full.split_inc(full.from_local_inc(J_next), FULL.HEAD, FULL.TAIL);

            let A = A
                .rb_mut()
                .row_segments_mut(head, tail, disjoint)
                .0
                .col_segments_mut(head, tail, disjoint)
                .0;
            let D = D.rb_mut().col_segments_mut(head, tail, disjoint).0;

            let signs = signs.map(|signs| signs.segments(head, tail).0);

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

pub fn cholesky_in_place<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'N>, Dim<'N>, ContiguousFwd>,
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
            simd_cholesky(simd, A, D, is_llt, regularize, eps, delta, signs)
        }
    }

    help!(C::Real);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{c64, Mat, Row};
    use rand::prelude::*;

    #[test]
    fn test_simd_cholesky() {
        if let Some(simd) = pulp::x86::V3::try_new() {
            let rng = &mut StdRng::seed_from_u64(0);

            let n = 7;
            let A = Mat::from_fn(n, n, |_, _| c64::new(rng.gen(), rng.gen()));
            let A = &A * &A.adjoint();

            with_dim!(N, A.nrows());
            let A = A.as_ref().as_shape(N, N);

            let mut L = A.cloned();
            let mut L = L.as_mut().try_as_col_major_mut().unwrap();
            let mut D = Row::zeros_with_ctx(&default(), N);
            let mut D = D.as_mut();

            simd_cholesky(simd, L.rb_mut(), D.rb_mut(), false, false, &0.0, &0.0, None).unwrap();

            for j in N.indices() {
                for i in zero().to(j.into()) {
                    L[(i, j)] = c64::ZERO;
                }
            }
            let L = L.rb().as_dyn_stride();

            __dbg!(L.rb() * D.rb().as_diagonal() * L.adjoint() - &A);
        }
    }
}
