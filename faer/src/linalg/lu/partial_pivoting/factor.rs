use crate::{internal_prelude::*, utils::bound::Segment};
use core::num::NonZero;

#[math]
fn swap_elems<'N, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    mut col: ColMut<'_, C, T, Dim<'N>>,
    i: Idx<'N>,
    j: Idx<'N>,
) {
    help!(C);
    let a = math(copy(col[i]));
    let b = math(copy(col[j]));

    write1!(col[i] = b);
    write1!(col[j] = a);
}

#[math]
fn lu_in_place_unblocked<'M, 'NCOLS, 'N, I: Index, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    mut matrix: MatMut<'_, C, T, Dim<'M>, Dim<'NCOLS>>,
    range: Segment<'_, 'NCOLS, 'N>,
    trans: &mut Array<'N, I>,
) -> usize {
    let M = matrix.nrows();
    let NCOLS = matrix.ncols();
    let N = range.len();

    if *N == 0 {
        return 0;
    }

    let mut n_trans = 0;

    for k in range {
        let t = &mut trans[range.from_global(k)];

        let k_row = M.idx(*range.from_global(k));
        let mut imax = k_row;
        let mut max = math.re(zero());

        for i in imax.to_incl().to(M.end()) {
            let abs = math(abs1(
                matrix.rb().col_segment(range)[(i, range.from_global(k))],
            ));
            if math(abs > max) {
                max = abs;
                imax = i;
            }
        }

        *t = I::truncate(*imax - *k_row);

        if imax != k_row {
            n_trans += 1;

            for j in NCOLS.indices() {
                swap_elems(ctx, matrix.rb_mut().col_mut(j), k_row, imax);
            }
        }

        let mut matrix = matrix.rb_mut().col_segment_mut(range);

        let j = range.from_global(k);
        help!(C);

        let inv = math(recip(matrix[(k_row, j)]));
        for i in k_row.next().to(M.end()) {
            write1!(matrix[(i, j)] = math(matrix[(i, j)] * inv));
        }

        ghost_tree!(FULL_ROWS(TOP, BOT), FULL_COLS(LEFT, RIGHT), {
            let (_, rows @ list![top, bot], disjoint_rows) =
                M.split(list![..k_row.next(), ..], FULL_ROWS);

            let (_, cols @ list![left, right], disjoint_cols) =
                N.split(list![..j.next(), ..], FULL_COLS);

            let j = left.idx(*j);
            let k_row = top.idx(*k_row);

            let list![A0, mut A1] = matrix.rb_mut().row_segments_mut(rows, disjoint_rows);
            let list![_, A01] = A0.rb().col_segments(cols);
            let list![A10, mut A11] = A1.rb_mut().col_segments_mut(cols, disjoint_cols);
            let A10 = A10.rb();

            let lhs = A10.col(left.from_global(j));
            let rhs = A01.row(top.from_global(k_row));
            for j in right {
                let mut col = A11.rb_mut().col_mut(right.from_global(j));
                let rhs = math(rhs[right.from_global(j)]);

                for i in bot {
                    write1!(
                        col[bot.from_global(i)] =
                            math(col[bot.from_global(i)] - lhs[bot.from_global(i)] * rhs)
                    );
                }
            }
        });
    }

    n_trans
}

#[math]
fn lu_in_place_recursion<'M, 'NCOLS, 'N, I: Index, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    A: MatMut<'_, C, T, Dim<'M>, Dim<'NCOLS>>,
    range: Segment<'_, 'NCOLS, 'N>,
    trans: &mut Array<'N, I>,
    par: Parallelism,
    params: PartialPivLuParams,
) -> usize {
    help!(C);
    let mut A = A;
    let M = A.nrows();
    let NCOLS = A.ncols();
    let N = range.len();

    if *N <= params.recursion_threshold.get() {
        return lu_in_place_unblocked(ctx, A, range, trans);
    }

    let blocksize = Ord::min(
        params.recursion_threshold.get(),
        Ord::max(params.blocksize.get(), N.next_power_of_two() / 2),
    );
    let mut n_trans = 0;

    let j = range.len().check(0);
    let j_next = N.advance(j, blocksize);
    let i = M.check(0);
    let i_next = M.advance(i, blocksize);

    ghost_tree!(FULL_COLS(BLOCK_COLS), {
        let (_, list![block], _) = range.len().split(list![..j_next], FULL_COLS);

        n_trans += lu_in_place_recursion(
            ctx,
            A.rb_mut().col_segment_mut(range),
            block,
            trans.segment_mut(block),
            par,
            params,
        );
    });

    ghost_tree!(ROWS(TOP, BOT), COLS(LEFT, RIGHT), {
        let (_, list![top, bot], disjoint_rows) = M.split(list![..i_next, ..], ROWS);
        let (_, list![left, right], disjoint_cols) = N.split(list![..j_next, ..], COLS);

        {
            let mut A = A.rb_mut().col_segment_mut(range);

            let list![A0, A1] = A.rb_mut().row_segments_mut(list![top, bot], disjoint_rows);
            let list![A00, mut A01] = A0.col_segments_mut(list![left, right], disjoint_cols);
            let list![A10, mut A11] = A1.col_segments_mut(list![left, right], disjoint_cols);

            let A00 = A00.rb().as_col_shape(A00.nrows());
            let A10 = A10.rb().as_col_shape(A00.nrows());
            {
                linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                    ctx,
                    A00.rb(),
                    A01.rb_mut(),
                    par,
                );
            }

            linalg::matmul::matmul(
                ctx,
                A11.rb_mut(),
                Some(as_ref!(math(one()))),
                A10.rb(),
                A01.rb(),
                as_ref!(math(-one())),
                par,
            );

            n_trans += lu_in_place_recursion(
                ctx,
                A.rb_mut().row_segment_mut(bot),
                right,
                trans.segment_mut(right),
                par,
                params,
            );
        }

        let swap = |mat: MatMut<'_, C, T, Dim<'M>, Dim<'_>>| {
            let mut mat = mat;
            for j in mat.ncols().indices() {
                let mut col = mat.rb_mut().col_mut(j);

                for (i, j) in core::iter::zip(top, left) {
                    let t = trans[j.local()];
                    swap_elems(ctx, col.rb_mut(), i.local(), M.check(t.zx() + *i.local()));
                }

                for (i, j) in core::iter::zip(bot, right) {
                    let t = trans[j.local()];

                    swap_elems(ctx, col.rb_mut(), i.local(), M.check(t.zx() + *i.local()));
                }
            }
        };

        ghost_tree!(COLS(BEFORE, AFTER), {
            let (_, list![before, after], disjoint) = NCOLS.split(
                list![..range.global(j.to_incl()).local(), range.end()..],
                COLS,
            );

            let list![A_left, A_right] =
                A.rb_mut().col_segments_mut(list![before, after], disjoint);

            match par {
                Parallelism::None => {
                    swap(A_left);
                    swap(A_right);
                }
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(nthreads) => {
                    let nthreads = nthreads.get();
                    let len = (*before.len() + *after.len()) as f64;
                    let left_threads = Ord::min(
                        (nthreads as f64 * (*A_left.ncols() as f64 / len)) as usize,
                        nthreads,
                    );
                    let right_threads = nthreads - left_threads;

                    use rayon::prelude::*;
                    rayon::join(
                        || {
                            A_left.par_col_partition_mut(left_threads).for_each(|A| {
                                swap(A.bind_c(unique!()));
                            })
                        },
                        || {
                            A_right.par_col_partition_mut(right_threads).for_each(|A| {
                                swap(A.bind_c(unique!()));
                            })
                        },
                    );
                }
            }
        });
    });

    n_trans
}

/// LUfactorization tuning parameters.
#[derive(Copy, Clone)]
#[non_exhaustive]
pub struct PartialPivLuParams {
    pub recursion_threshold: NonZero<usize>,
    pub blocksize: NonZero<usize>,
}

/// Information about the resulting LU factorization.
#[derive(Copy, Clone, Debug)]
pub struct PartialPivLuInfo {
    /// Number of transpositions that were performed, can be used to compute the determinant of
    /// $P$.
    pub transposition_count: usize,
}

/// Error in the LDLT factorization.
#[derive(Copy, Clone, Debug)]
pub enum LdltError {
    ZeroPivot { index: usize },
}

impl Default for PartialPivLuParams {
    #[inline]
    fn default() -> Self {
        Self {
            recursion_threshold: NonZero::new(16).unwrap(),
            blocksize: NonZero::new(64).unwrap(),
        }
    }
}

#[inline]
pub fn lu_in_place_scratch<I: Index, C: ComplexContainer, T: ComplexField<C>>(
    nrows: usize,
    ncols: usize,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_new::<I>(Ord::min(nrows, ncols))
}

pub fn lu_in_place<'out, 'M, 'N, I: Index, C: ComplexContainer, T: ComplexField<C>>(
    ctx: &Ctx<C, T>,
    matrix: MatMut<'_, C, T, Dim<'M>, Dim<'N>>,
    perm: &'out mut Array<'M, I>,
    perm_inv: &'out mut Array<'M, I>,
    par: Parallelism,
    stack: &mut DynStack,
    params: PartialPivLuParams,
) -> (PartialPivLuInfo, PermRef<'out, I, Dim<'M>>) {
    let _ = &params;
    let truncate = <I::Signed as SignedIndex>::truncate;

    #[cfg(feature = "perf-warn")]
    if (matrix.col_stride().unsigned_abs() == 1 || matrix.row_stride().unsigned_abs() != 1)
        && crate::__perf_warn!(LU_WARN)
    {
        log::warn!(target: "faer_perf", "LU with partial pivoting prefers column-major or row-major matrix. Found matrix with generic strides.");
    }

    let mut matrix = matrix;
    let mut stack = stack;
    let M = matrix.nrows();
    let N = matrix.ncols();

    let size = Ord::min(*N, *M);

    ghost_tree!(FULL_COLS(LEFT, RIGHT), {
        let (_, list![left, right], disjoint) = N.split(list![..N.idx_inc(size), ..], FULL_COLS);

        for i in M.indices() {
            let p = &mut perm[i];
            *p = I::from_signed(truncate(*i));
        }

        let (mut transpositions, _) = stack
            .rb_mut()
            .make_with(size, |_| I::from_signed(truncate(0)));
        let transpositions = transpositions.as_mut();
        let transpositions = Array::from_mut(transpositions, left.len());

        let n_transpositions =
            lu_in_place_recursion(ctx, matrix.rb_mut(), left, transpositions, par, params);

        for idx in left {
            let t = transpositions[left.from_global(idx)];
            perm.as_mut().swap(*idx, *idx + t.zx());
        }

        if *M < *N {
            let list![left, right] = matrix.col_segments_mut(list![left, right], disjoint);
            linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                ctx,
                left.rb().as_shape(M, M),
                right,
                par,
            );
        }

        for i in M.indices() {
            perm_inv[M.check(perm[i].zx())] = I::from_signed(truncate(*i));
        }

        (
            PartialPivLuInfo {
                transposition_count: n_transpositions,
            },
            unsafe {
                PermRef::new_unchecked(
                    Idx::from_slice_ref_checked(perm.as_ref(), M),
                    Idx::from_slice_ref_checked(perm_inv.as_ref(), M),
                    M,
                )
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use dyn_stack::GlobalMemBuffer;
    use faer_traits::Unit;

    use super::*;
    use crate::{assert, stats::prelude::*, utils::approx::*, Mat};

    #[test]
    fn test_plu() {
        let rng = &mut StdRng::seed_from_u64(0);

        let approx_eq = CwiseMat(ApproxEq {
            ctx: ctx::<Ctx<Unit, f64>>(),
            abs_tol: 1e-13,
            rel_tol: 1e-13,
        });

        for n in [3, 128, 255, 256, 257] {
            with_dim!(N, n);

            let A = CwiseMatDistribution {
                nrows: N,
                ncols: N,
                dist: StandardNormal,
            }
            .rand::<Mat<f64, Dim, Dim>>(rng);
            let A = A.as_ref();

            let mut LU = A.cloned();
            let perm = &mut *vec![0usize; n];
            let perm_inv = &mut *vec![0usize; n];
            let perm = Array::from_mut(perm, N);
            let perm_inv = Array::from_mut(perm_inv, N);

            let p = lu_in_place(
                &ctx(),
                LU.as_mut(),
                perm,
                perm_inv,
                Parallelism::None,
                DynStack::new(&mut GlobalMemBuffer::new(
                    lu_in_place_scratch::<usize, Unit, f64>(*N, *N).unwrap(),
                )),
                PartialPivLuParams {
                    recursion_threshold: NonZero::new(2).unwrap(),
                    blocksize: NonZero::new(2).unwrap(),
                },
            )
            .1;

            let mut L = LU.as_ref().cloned();
            let mut U = LU.as_ref().cloned();

            for j in N.indices() {
                for i in zero().to(j.excl()) {
                    L[(i, j)] = 0.0;
                }
                L[(j, j)] = 1.0;
            }
            for j in N.indices() {
                for i in j.next().to(N.end()) {
                    U[(i, j)] = 0.0;
                }
            }
            let L = L.as_ref();
            let U = U.as_ref();

            assert!(p.inverse() * L * U ~ A);
        }

        for m in [8, 128, 255, 256, 257] {
            with_dim!(M, m);
            with_dim!(N, 8);

            let A = CwiseMatDistribution {
                nrows: M,
                ncols: N,
                dist: StandardNormal,
            }
            .rand::<Mat<f64, Dim, Dim>>(rng);
            let A = A.as_ref();

            let mut LU = A.cloned();
            let perm = &mut *vec![0usize; *M];
            let perm_inv = &mut *vec![0usize; *M];
            let perm = Array::from_mut(perm, M);
            let perm_inv = Array::from_mut(perm_inv, M);

            let p = lu_in_place(
                &ctx(),
                LU.as_mut(),
                perm,
                perm_inv,
                Parallelism::None,
                DynStack::new(&mut GlobalMemBuffer::new(
                    lu_in_place_scratch::<usize, Unit, f64>(*N, *N).unwrap(),
                )),
                PartialPivLuParams::default(),
            )
            .1;

            let mut L = LU.as_ref().cloned();
            let mut U = LU.as_ref().cloned();

            for j in N.indices() {
                let i = M.check(*j);
                for i in zero().to(i.excl()) {
                    if *i >= *j {
                        break;
                    }
                    L[(i, j)] = 0.0;
                }
                L[(i, j)] = 1.0;
            }
            for j in N.indices() {
                let i = M.check(*j);
                for i in i.next().to(M.end()) {
                    U[(i, j)] = 0.0;
                }
            }
            let L = L.as_ref();
            let U = U.as_ref();

            let U = U.subrows(zero(), N);

            assert!(p.inverse() * L * U ~ A);
        }
    }
}
