use crate::{
    assert,
    internal_prelude::*,
    perm::{swap_cols_idx, swap_rows_idx},
};
use linalg::matmul::triangular::{self, BlockStructure};

/// Pivoting strategy for choosing the pivots.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub enum PivotingStrategy {
    /// Diagonal pivoting.
    Diagonal,
}

/// Tuning parameters for the decomposition.
#[derive(Copy, Clone, Debug)]
pub struct BunchKaufmanParams {
    /// Pivoting strategy.
    pub pivoting: PivotingStrategy,
    /// Block size of the algorithm.
    pub blocksize: usize,

    pub non_exhaustive: NonExhaustive,
}

/// Dynamic Bunch-Kaufman regularization.
/// Values below `epsilon` in absolute value, or with the wrong sign are set to `delta` with
/// their corrected sign.
pub struct BunchKaufmanRegularization<'a, T: ComplexField> {
    /// Expected signs for the diagonal at each step of the decomposition.
    pub dynamic_regularization_signs: Option<&'a mut [i8]>,
    /// Regularized value.
    pub dynamic_regularization_delta: T::Real,
    /// Regularization threshold.
    pub dynamic_regularization_epsilon: T::Real,
}

impl<T: ComplexField> Default for BunchKaufmanRegularization<'_, T> {
    fn default() -> Self {
        Self {
            dynamic_regularization_signs: None,
            dynamic_regularization_delta: zero(),
            dynamic_regularization_epsilon: zero(),
        }
    }
}

impl<T: ComplexField> Auto<T> for BunchKaufmanParams {
    fn auto() -> Self {
        Self {
            pivoting: PivotingStrategy::Diagonal,
            blocksize: 64,
            non_exhaustive: NonExhaustive(()),
        }
    }
}

#[math]
fn best_score_idx_skip<T: ComplexField>(a: ColRef<'_, T>, skip: usize) -> (Option<usize>, T::Real) {
    let m = a.nrows();

    if m <= skip {
        return (None, zero());
    }

    let mut best_row = skip;
    let mut best_score = zero();

    for i in best_row..m {
        let score = abs(a[i]);
        if score > best_score {
            best_row = i;
            best_score = score;
        }
    }

    (Some(best_row), best_score)
}

fn assign_col<T: ComplexField>(a: MatMut<'_, T>, i: usize, j: usize) {
    if i != j {
        let (ai, aj) = a.two_cols_mut(i, j);
        { ai }.copy_from(aj);
    }
}

#[math]
fn best_score<T: ComplexField>(a: ColRef<'_, T>) -> T::Real {
    let M = a.nrows();

    let mut best_score = zero();

    for i in 0..M {
        let score = abs(a[i]);
        if score > best_score {
            best_score = score;
        }
    }

    best_score
}

#[math]
fn swap_elems_conj<T: ComplexField>(
    a: MatMut<'_, T>,
    (i0, j0): (usize, usize),
    (i1, j1): (usize, usize),
) {
    let mut a = a;
    let (x, y) = (conj(a[(i0, j0)]), conj(a[(i1, j1)]));

    a[(i0, j0)] = y;
    a[(i1, j1)] = x;
}
#[math]
fn swap_elems<T: ComplexField>(
    a: MatMut<'_, T>,
    (i0, j0): (usize, usize),
    (i1, j1): (usize, usize),
) {
    let mut a = a;
    let (x, y) = (copy(a[(i0, j0)]), copy(a[(i1, j1)]));

    a[(i0, j0)] = y;
    a[(i1, j1)] = x;
}

#[math]
fn make_real<T: ComplexField>(mut a: MatMut<'_, T>, (i0, j0): (usize, usize)) {
    a[(i0, j0)] = from_real(real(a[(i0, j0)]));
}

#[math]
fn cholesky_diagonal_pivoting_blocked_step<I: Index, T: ComplexField>(
    mut a: MatMut<'_, T>,
    regularization: BunchKaufmanRegularization<'_, T>,
    mut w: MatMut<'_, T>,
    pivots: &mut [I],
    alpha: T::Real,
    par: Par,
) -> (usize, usize, usize) {
    let n = a.nrows();
    let nb = w.ncols();

    assert!(nb < n);
    if n == 0 {
        return (0, 0, 0);
    }

    let eps = abs(regularization.dynamic_regularization_epsilon);
    let delta = abs(regularization.dynamic_regularization_delta);
    let mut signs = regularization.dynamic_regularization_signs;
    let has_eps = delta > zero();
    let mut dynamic_regularization_count = 0usize;
    let mut pivot_count = 0usize;

    let truncate = <I::Signed as SignedIndex>::truncate;

    let mut k = 0;
    while k < n && k + 1 < nb {
        let k0 = k;
        let j0 = k;
        let j1 = k + 1;

        w.rb_mut().get_mut(k0.., j0).copy_from(a.rb().get(k0.., k0));
        let (w_left, w_right) = w.rb_mut().get_mut(k0.., ..).split_at_col_mut(j0);

        let w_row = w_left.rb().row(0);
        let w_col = w_right.col_mut(0);
        crate::linalg::matmul::matmul(
            w_col.as_mat_mut(),
            Accum::Add,
            a.rb().get(k0..n, ..k0),
            w_row.rb().transpose().as_mat(),
            -one::<T>(),
            par,
        );
        make_real(w.rb_mut(), (k0, j0));

        let mut k_step = 1;

        let abs_akk = abs(real(w[(k0, j0)]));

        let (imax, colmax) = best_score_idx_skip(w.rb().col(j0), k0 + 1);

        let kp;
        if max(abs_akk, colmax) == zero() {
            kp = k0;

            let mut d11 = real(w[(k0, j0)]);
            if has_eps {
                if let Some(signs) = signs.rb_mut() {
                    if signs[k0] > 0 && d11 <= eps {
                        d11 = copy(delta);
                        dynamic_regularization_count += 1;
                    } else if signs[k0] < 0 && d11 >= -eps {
                        d11 = neg(delta);
                        dynamic_regularization_count += 1;
                    }
                }
            }
            a[(k0, k0)] = from_real(d11);
        } else {
            if abs_akk >= colmax * alpha {
                kp = k0;
            } else {
                let imax = imax.unwrap();
                z!(
                    w.rb_mut().get_mut(k0..imax, j1),
                    a.rb().get(imax, k0..imax).transpose(),
                )
                .for_each(|uz!(dst, src)| *dst = conj(src));

                w.rb_mut()
                    .get_mut(imax.., j1)
                    .copy_from(a.rb().get(imax.., imax));

                let (w_left, w_right) = w.rb_mut().get_mut(k0.., ..).split_at_col_mut(j1);

                let w_row = w_left.rb().row(imax - k).subcols(0, k);
                let w_col = w_right.col_mut(0);

                crate::linalg::matmul::matmul(
                    w_col.as_mat_mut(),
                    Accum::Add,
                    a.rb().get(k0.., ..k0),
                    w_row.rb().transpose().as_mat(),
                    -one::<T>(),
                    par,
                );
                make_real(w.rb_mut(), (imax, j1));

                let rowmax = max(
                    best_score(w.rb().get(k0..imax, j1)),
                    best_score(w.rb().get(imax + 1.., j1)),
                );

                if abs_akk >= (alpha * colmax) * (colmax / rowmax) {
                    kp = k0;
                } else if abs(real(w[(imax, j1)])) >= alpha * rowmax {
                    kp = imax;
                    assign_col(w.rb_mut().get_mut(k0.., ..), j0, j1);
                } else {
                    kp = imax;
                    k_step = 2;
                }
            }

            let kk = k + k_step - 1;
            let jk = kk;

            if kp != kk {
                pivot_count += 1;
                if let Some(signs) = signs.rb_mut() {
                    signs.swap(kp, kk);
                }
                a[(kp, kp)] = copy(a[(kk, kk)]);
                for j in kk + 1..kp {
                    a[(kp, j)] = conj(a[(j, kk)]);
                }
                assign_col(a.rb_mut().get_mut(kp + 1.., ..), kp, kk);

                swap_rows_idx(a.rb_mut().split_at_col_mut(k0).0, kk, kp);
                swap_rows_idx(w.rb_mut().split_at_col_mut(jk + 1).0, kk, kp);
            }

            if k_step == 1 {
                a.rb_mut().get_mut(k0.., k0).copy_from(w.rb().get(k0.., j0));

                let mut d11 = real(w[(k0, j0)]);
                if has_eps {
                    if let Some(signs) = signs.rb_mut() {
                        if signs[k0] > 0 && d11 <= eps {
                            d11 = copy(delta);
                            dynamic_regularization_count += 1;
                        } else if signs[k0] < 0 && d11 >= -eps {
                            d11 = neg(delta);
                            dynamic_regularization_count += 1;
                        }
                    } else if abs(d11) <= eps {
                        if d11 < zero() {
                            d11 = neg(delta);
                        } else {
                            d11 = copy(delta);
                        }
                        dynamic_regularization_count += 1;
                    }
                }
                a[(k0, k0)] = from_real(d11);
                let d11 = recip(d11);

                let x = a.rb_mut().get_mut(k0 + 1.., k0);
                z!(x).for_each(|uz!(x)| *x = mul_real(x, d11));
                z!(w.rb_mut().get_mut(k0 + 1.., j0)).for_each(|uz!(x)| *x = conj(x));
            } else {
                let k1 = k + 1;

                let dd = abs(w[(k1, j0)]);
                let dd_inv = recip(dd);
                let mut d11 = dd_inv * real(w[(k1, j1)]);
                let mut d22 = dd_inv * real(w[(k0, j0)]);

                let eps = eps * dd_inv;
                let delta = delta * dd_inv;
                if has_eps {
                    if let Some(signs) = signs.rb_mut() {
                        if signs[k0] > 0 && signs[k1] > 0 {
                            {
                                if d11 <= eps {
                                    d11 = copy(delta);
                                    dynamic_regularization_count += 1;
                                }
                                if d22 <= eps {
                                    d22 = copy(delta);
                                    dynamic_regularization_count += 1;
                                }
                            }
                        } else if signs[k0] < 0 && signs[k1] < 0 {
                            {
                                if d11 >= -eps {
                                    d11 = -delta;
                                    dynamic_regularization_count += 1;
                                }
                                if d22 >= -eps {
                                    d22 = -delta;
                                    dynamic_regularization_count += 1;
                                }
                            }
                        }
                    }
                }

                // t = (d11/|d21| * d22/|d21| - 1.0)
                let mut t = d11 * d22 - one();
                if has_eps {
                    if let Some(signs) = signs.rb_mut() {
                        if ((signs[k0] > 0 && signs[k1] > 0) || (signs[k0] < 0 && signs[k1] < 0))
                            && t <= eps
                        {
                            t = copy(delta);
                        } else if ((signs[k0] > 0 && signs[k1] < 0)
                            || (signs[k0] < 0 && signs[k1] > 0))
                            && t >= -eps
                        {
                            t = -delta;
                        }
                    }
                }

                let t = recip(t);
                let d21 = mul_real(w[(k1, j0)], dd_inv);
                let d = t * dd_inv;

                a[(k0, k0)] = copy(w[(k0, j0)]);
                a[(k1, k0)] = copy(w[(k1, j0)]);
                a[(k1, k1)] = copy(w[(k1, j1)]);
                make_real(a.rb_mut(), (k0, k0));
                make_real(a.rb_mut(), (k1, k1));

                for j in k1 + 1..n {
                    let wk = mul_real(
                        //
                        mul_real(w[(j, j0)], d11) - (w[(j, j1)] * d21),
                        d,
                    );
                    let wkp1 = mul_real(
                        //
                        mul_real(w[(j, j1)], d22) - (w[(j, j0)] * conj(d21)),
                        d,
                    );

                    a[(j, k0)] = wk;
                    a[(j, k1)] = wkp1;
                }

                z!(w.rb_mut().get_mut(k1.., j0)).for_each(|uz!(x)| *x = conj(x));
                z!(w.rb_mut().get_mut(k1 + 1.., j1)).for_each(|uz!(x)| *x = conj(x));
            }
        }

        if k_step == 1 {
            pivots[k0] = I::from_signed(truncate(kp));
        } else {
            let k1 = k + 1;
            pivots[k0] = I::from_signed(truncate(!kp));
            pivots[k1] = I::from_signed(truncate(!kp));
        }

        k += k_step;
    }

    let k0 = n.checked_idx_inc(k);
    let j0 = nb.checked_idx_inc(k);

    let (a_left, mut a_right) = a.rb_mut().get_mut(k0.., ..).split_at_col_mut(k0);
    triangular::matmul(
        a_right.rb_mut(),
        BlockStructure::TriangularLower,
        Accum::Add,
        a_left.rb(),
        BlockStructure::Rectangular,
        w.rb().get(k0.., ..j0).transpose(),
        BlockStructure::Rectangular,
        -one::<T>(),
        par,
    );

    z!(a_right.diagonal_mut().column_vector_mut()).for_each(|uz!(x)| *x = from_real(real(*x)));

    let mut j = k - 1;
    loop {
        let jj = j;
        let mut jp = pivots[j].to_signed().sx();
        if (jp as isize) < 0 {
            jp = !jp;
            j -= 1;
        }

        if j == 0 {
            return (k, pivot_count, dynamic_regularization_count);
        }
        j -= 1;

        if jp != jj {
            swap_rows_idx(a.rb_mut().get_mut(.., ..j + 1), jp, jj);
        }
        if j == 0 {
            return (k, pivot_count, dynamic_regularization_count);
        }
    }
}

#[math]
fn cholesky_diagonal_pivoting_unblocked<I: Index, T: ComplexField>(
    mut a: MatMut<'_, T>,
    regularization: BunchKaufmanRegularization<'_, T>,
    pivots: &mut [I],
    alpha: T::Real,
) -> (usize, usize) {
    let truncate = <I::Signed as SignedIndex>::truncate;

    assert!(a.nrows() == a.ncols());
    let n = a.nrows();

    if n == 0 {
        return (0, 0);
    }

    let eps = abs(regularization.dynamic_regularization_epsilon);
    let delta = abs(regularization.dynamic_regularization_delta);
    let mut signs = regularization.dynamic_regularization_signs;

    let has_eps = delta > zero();
    let mut dynamic_regularization_count = 0usize;
    let mut pivot_count = 0usize;

    let mut k = 0;
    while k < n {
        let k0 = k;

        // let (l![after_k], _) = N.split(l![k0.to_incl()..], FULL);
        // let k0_ = after_k.idx(*k0);

        let mut k_step = 1;
        let abs_akk = abs(a[(k0, k0)]);
        let (imax, colmax) = best_score_idx_skip(a.rb().col(k0).split_at_row(k0).1, 0);

        let imax = imax.map(|imax| imax + k0);

        let kp;
        if max(abs_akk, colmax) == zero() {
            kp = k0;

            let mut d11 = real(a[(k0, k0)]);
            if has_eps {
                if let Some(signs) = signs.rb_mut() {
                    if signs[k0] > 0 && d11 <= eps {
                        d11 = copy(delta);
                        dynamic_regularization_count += 1;
                    } else if signs[k0] < 0 && d11 >= -eps {
                        d11 = neg(delta);
                        dynamic_regularization_count += 1;
                    }
                }
            }
            a[(k0, k0)] = from_real(d11);
        } else {
            {
                if abs_akk >= colmax * alpha {
                    kp = k0;
                } else {
                    let imax_global = imax.unwrap();
                    let imax = imax_global;

                    // let (l![k_imax], _) = N.split(l![k0.to_incl()..imax.to_incl()], AFTER_K0);
                    // let (l![imax_end], _) = N.split(l![imax.next()..], AFTER_K1);

                    let rowmax = max(
                        best_score(a.rb().get(imax, k0..imax).transpose()),
                        best_score(a.rb().get(imax.., imax)),
                    );

                    if abs_akk >= (alpha * colmax) * (colmax / rowmax) {
                        kp = k0;
                    } else if abs(real(a[(imax, imax)])) >= alpha * rowmax {
                        kp = imax_global;
                    } else {
                        kp = imax_global;
                        k_step = 2;
                    }
                }
            }

            let kp = kp;
            let kk = k + k_step - 1;

            if kp != kk {
                let k1 = k + 1;

                pivot_count += 1;
                swap_cols_idx(a.rb_mut().get_mut(kp + 1.., ..), kk, kp);
                for j in kk + 1..kp {
                    swap_elems_conj(a.rb_mut(), (j, kk), (kp, j));
                }

                a[(kp, kk)] = conj(a[(kp, kk)]);
                swap_elems(a.rb_mut(), (kk, kk), (kp, kp));

                if k_step == 2 {
                    swap_elems(a.rb_mut(), (k1, k0), (kp, k0));
                }
            }

            if k_step == 1 {
                let mut d11 = real(a[(k0, k0)]);
                if has_eps {
                    if let Some(signs) = signs.rb_mut() {
                        if signs[k0] > 0 && d11 <= eps {
                            d11 = copy(delta);
                            dynamic_regularization_count += 1;
                        } else if signs[k0] < 0 && d11 >= -eps {
                            d11 = neg(delta);
                            dynamic_regularization_count += 1;
                        }
                    } else if abs(d11) <= eps {
                        if d11 < zero() {
                            d11 = neg(delta);
                        } else {
                            d11 = copy(delta);
                        }
                        dynamic_regularization_count += 1;
                    }
                }
                a[(k0, k0)] = from_real(d11);
                let d11 = recip(d11);

                for j in k0 + 1..n {
                    let d11xj = mul_real(conj(a[(j, k0)]), d11);
                    for i in j..n {
                        let xi = copy(a[(i, k0)]);
                        a[(i, j)] = a[(i, j)] - d11xj * xi;
                    }
                    make_real(a.rb_mut(), (j, j));
                }
                z!(a.rb_mut().get_mut(k0 + 1.., k0)).for_each(|uz!(x)| *x = mul_real(x, d11));
            } else {
                let k1 = k + 1;
                let d21 = abs(a[(k1, k0)]);
                let d21_inv = recip(d21);
                let mut d11 = d21_inv * real(a[(k1, k1)]);
                let mut d22 = d21_inv * real(a[(k0, k0)]);

                let eps = eps * d21_inv;
                let delta = delta * d21_inv;
                if has_eps {
                    if let Some(signs) = signs.rb_mut() {
                        if signs[k0] > 0 && signs[k1] > 0 {
                            {
                                if d11 <= eps {
                                    d11 = copy(delta);
                                    dynamic_regularization_count += 1;
                                }
                                if d22 <= eps {
                                    d22 = copy(delta);
                                    dynamic_regularization_count += 1;
                                }
                            }
                        } else if signs[k0] < 0 && signs[k1] < 0 {
                            {
                                if d11 >= -eps {
                                    d11 = -delta;
                                    dynamic_regularization_count += 1;
                                }
                                if d22 >= -eps {
                                    d22 = -delta;
                                    dynamic_regularization_count += 1;
                                }
                            }
                        }
                    }
                }

                // t = (d11/|d21| * d22/|d21| - 1.0)
                let mut t = d11 * d22 - one();
                if has_eps {
                    if let Some(signs) = signs.rb_mut() {
                        if ((signs[k0] > 0 && signs[k1] > 0) || (signs[k0] < 0 && signs[k1] < 0))
                            && t <= eps
                        {
                            t = copy(delta);
                        } else if ((signs[k0] > 0 && signs[k1] < 0)
                            || (signs[k0] < 0 && signs[k1] > 0))
                            && t >= -eps
                        {
                            t = neg(delta);
                        }
                    }
                }

                let t = recip(t);
                let d21 = mul_real(a[(k1, k0)], d21_inv);
                let d = t * d21_inv;

                for j in k1 + 1..n {
                    let wk = mul_real(
                        //
                        mul_real(a[(j, k0)], d11) - (a[(j, k1)] * d21),
                        d,
                    );
                    let wkp1 = mul_real(
                        //
                        mul_real(a[(j, k1)], d22) - (a[(j, k0)] * conj(d21)),
                        d,
                    );

                    for i in j..n {
                        a[(i, j)] = a[(i, j)] - a[(i, k0)] * conj(wk) - a[(i, k1)] * conj(wkp1);
                    }
                    make_real(a.rb_mut(), (j, j));

                    a[(j, k0)] = wk;
                    a[(j, k1)] = wkp1;
                }
            }
        }

        if k_step == 1 {
            pivots[k0] = I::from_signed(truncate(kp));
        } else {
            let k1 = k + 1;
            pivots[k0] = I::from_signed(truncate(!kp));
            pivots[k1] = I::from_signed(truncate(!kp));
        }
        k += k_step;
    }

    (pivot_count, dynamic_regularization_count)
}

#[math]
fn convert<'N, I: Index, T: ComplexField>(
    mut a: MatMut<'_, T, Dim<'N>, Dim<'N>>,
    pivots: &Array<'N, I>,
    mut subdiag: ColMut<'_, T, Dim<'N>>,
) {
    assert!(a.nrows() == a.ncols());
    let N = a.nrows();

    let mut i = 0;
    while let Some(i0) = N.try_check(i) {
        if (pivots[i0].to_signed().sx() as isize) < 0 {
            let i1 = N.check(i + 1);

            subdiag[i0] = copy(a[(i1, i0)]);
            subdiag[i1] = zero();
            a[(i1, i0)] = zero();
            i += 2;
        } else {
            subdiag[i0] = zero();
            i += 1;
        }
    }

    let mut i = 0;
    while let Some(i0) = N.try_check(i) {
        guards!(head);
        let a = a.rb_mut().get_mut(.., IdxInc::ZERO..i0.into()).bind_c(head);

        let p = pivots[i0].to_signed().sx();
        if (p as isize) < 0 {
            let p = !p;
            let i1 = N.check(i + 1);
            let p = N.check(p);

            swap_rows_idx(a, i1, p);
            i += 2;
        } else {
            let p = N.check(p);
            swap_rows_idx(a, i0, p);
            i += 1;
        }
    }
}

/// Computes the size and alignment of required workspace for performing a Cholesky
/// decomposition with Bunch-Kaufman pivoting.
pub fn cholesky_in_place_scratch<I: Index, T: ComplexField>(
    dim: usize,
    par: Par,
    params: BunchKaufmanParams,
) -> Result<StackReq, SizeOverflow> {
    let _ = par;
    let mut bs = params.blocksize;
    if bs < 2 || dim <= bs {
        bs = 0;
    }
    StackReq::try_new::<I>(dim)?.try_and(temp_mat_scratch::<T>(dim, bs)?)
}

/// Info about the result of the Bunch-Kaufman factorization.
#[derive(Copy, Clone, Debug)]
pub struct BunchKaufmanInfo {
    /// Number of pivots whose value or sign had to be corrected.
    pub dynamic_regularization_count: usize,
    /// Number of pivoting transpositions.
    pub transposition_count: usize,
}

/// Computes the Cholesky factorization with Bunch-Kaufman  pivoting of the input matrix and
/// stores the factorization in `matrix` and `subdiag`.
///
/// The diagonal of the block diagonal matrix is stored on the diagonal
/// of `matrix`, while the subdiagonal elements of the blocks are stored in `subdiag`.
///
/// # Panics
///
/// Panics if the input matrix is not squa
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`cholesky_in_place_scratch`]).

#[track_caller]
#[math]
pub fn cholesky_in_place<'out, I: Index, T: ComplexField>(
    A: MatMut<'_, T>,
    subdiag: ColMut<'_, T>,
    regularization: BunchKaufmanRegularization<'_, T>,
    perm: &'out mut [I],
    perm_inv: &'out mut [I],
    par: Par,
    stack: &mut DynStack,
    params: BunchKaufmanParams,
) -> (BunchKaufmanInfo, PermRef<'out, I>) {
    let truncate = <I::Signed as SignedIndex>::truncate;
    let mut regularization = regularization;

    let n = A.nrows();
    assert!(all(
        A.nrows() == A.ncols(),
        subdiag.nrows() == n,
        perm.len() == n,
        perm_inv.len() == n
    ));

    #[cfg(feature = "perf-warn")]
    if A.row_stride().unsigned_abs() != 1 && crate::__perf_warn!(CHOLESKY_WARN) {
        if A.col_stride().unsigned_abs() == 1 {
            log::warn!(target: "faer_perf", "Bunch-Kaufman decomposition prefers column-major
    matrix. Found row-major matrix.");
        } else {
            log::warn!(target: "faer_perf", "Bunch-Kaufman decomposition prefers column-major
    matrix. Found matrix with generic strides.");
        }
    }

    let _ = par;
    let mut matrix = A;

    let alpha = mul_pow2(
        one::<T::Real>() + sqrt(from_f64::<T::Real>(17.0)),
        from_f64::<T::Real>(0.125),
    );

    let (mut pivots, stack) = stack.make_with::<I>(n, |_| I::truncate(0));
    let pivots = &mut *pivots;

    let mut bs = params.blocksize;
    if bs < 2 || n <= bs {
        bs = 0;
    }
    let mut work = unsafe { temp_mat_uninit(n, bs, stack) }.0;
    let mut work = work.as_mat_mut();

    let mut k = 0;
    let mut dynamic_regularization_count = 0;
    let mut transposition_count = 0;
    while k < n {
        let regularization = BunchKaufmanRegularization {
            dynamic_regularization_signs: regularization
                .dynamic_regularization_signs
                .rb_mut()
                .map(|signs| &mut signs[k..]),
            dynamic_regularization_delta: copy(regularization.dynamic_regularization_delta),
            dynamic_regularization_epsilon: copy(regularization.dynamic_regularization_epsilon),
        };

        let kb;
        let reg_count;
        let piv_count;

        let rem = n - k;

        let alpha = copy(alpha);
        if bs >= 2 && bs < n - k {
            (kb, piv_count, reg_count) = cholesky_diagonal_pivoting_blocked_step(
                matrix.rb_mut().submatrix_mut(k, k, rem, rem),
                regularization,
                work.rb_mut().subrows_mut(k, rem),
                &mut pivots[k..],
                alpha,
                par,
            );
        } else {
            (piv_count, reg_count) = cholesky_diagonal_pivoting_unblocked(
                matrix.rb_mut().submatrix_mut(k, k, rem, rem),
                regularization,
                &mut pivots[k..],
                alpha,
            );
            kb = n - k;
        }
        dynamic_regularization_count += reg_count;
        transposition_count += piv_count;

        for pivot in &mut pivots[k..k + kb] {
            let pv = (*pivot).to_signed().sx();
            if pv as isize >= 0 {
                *pivot = I::from_signed(truncate(pv + k));
            } else {
                *pivot = I::from_signed(truncate(pv - k));
            }
        }

        k += kb;
    }

    with_dim!(N, n);
    convert(
        matrix.rb_mut().as_shape_mut(N, N),
        Array::from_mut(pivots, N),
        subdiag.as_row_shape_mut(N),
    );

    for (i, p) in perm.iter_mut().enumerate() {
        *p = I::from_signed(truncate(i));
    }
    let mut i = 0;
    while i < n {
        let p = pivots[i].to_signed().sx();
        if (p as isize) < 0 {
            let p = !p;
            perm.swap(i + 1, p);
            i += 2;
        } else {
            perm.swap(i, p);
            i += 1;
        }
    }
    for (i, &p) in perm.iter().enumerate() {
        perm_inv[p.to_signed().zx()] = I::from_signed(truncate(i));
    }

    (
        BunchKaufmanInfo {
            dynamic_regularization_count,
            transposition_count,
        },
        unsafe { PermRef::new_unchecked(perm, perm_inv, n) },
    )
}
