use crate::{linalg, Col, ColRef, ComplexField, Mat, MatMut, MatRef, Parallelism, RealField};
use dyn_stack::{GlobalPodBuffer, PodStack};
use reborrow::*;

type E = f64;

fn iterate_arnoldi(A: MatRef<'_, E>, H: MatMut<'_, E>, V: MatMut<'_, E>, start: usize, end: usize) {
    let mut V = V;
    let mut H = H;
    for j in start..end + 1 {
        let mut H = H.rb_mut().col_mut(j - 1);
        H.fill_zero();

        let (V, Vnext) = V.rb_mut().split_at_col_mut(j);
        let V = V.rb();
        let Vlast = V.col(j - 1);
        let mut Vnext = Vnext.col_mut(0);

        Vnext.copy_from(A * Vlast);
        let mut h = H.rb_mut().get_mut(..j);
        h.copy_from(V.adjoint() * &Vnext);
        Vnext -= V * h;

        let norm = Vnext.norm_l2();
        Vnext /= norm;

        H[j] = norm;
    }
}

fn reorder_schur(
    mut A: MatMut<'_, E>,
    mut Q: Option<MatMut<'_, E>>,
    mut ifst: usize,
    mut ilst: usize,
) {
    let epsilon = E::faer_epsilon();
    let zero_threshold = E::faer_zero_threshold();
    use linalg::evd::hessenberg_real_evd::schur_swap;

    let zero = E::faer_zero();
    let n = A.nrows();

    // *
    // * Determine the first row of the specified block and find out
    // * if it is 1-by-1 or 2-by-2.
    // *
    if ifst > 0 {
        if A.read(ifst, ifst - 1) != zero {
            ifst -= 1;
        }
    }
    let mut nbf = 1;
    if ifst < n - 1 {
        if A.read(ifst + 1, ifst) != zero {
            nbf = 2;
        }
    }

    // *
    // * Determine the first row of the final block
    // * and find out if it is 1-by-1 or 2-by-2.
    // *
    if ilst > 0 {
        if A.read(ilst, ilst - 1) != zero {
            ilst = ilst - 1;
        }
    }
    let mut nbl = 1;
    if ilst < n - 1 {
        if A.read(ilst + 1, ilst) != zero {
            nbl = 2
        }
    }
    if ifst == ilst {
        return;
    }

    if ifst < ilst {
        if nbf == 2 && nbl == 1 {
            ilst -= 1;
        }
        if nbf == 1 && nbl == 2 {
            ilst += 1;
        }

        let mut here = ifst;
        // * Swap with next one below.
        loop {
            if nbf == 1 || nbf == 2 {
                // * Current block either 1-by-1 or 2-by-2.
                let mut nbnext = 1;
                if here + nbf + 1 <= n - 1 {
                    if A.read(here + nbf + 1, here + nbf) != zero {
                        nbnext = 2;
                    }
                }

                schur_swap(
                    A.rb_mut(),
                    Q.rb_mut(),
                    here,
                    nbf,
                    nbnext,
                    epsilon,
                    zero_threshold,
                );

                here += nbnext;
                // * Test if 2-by-2 block breaks into two 1-by-1 blocks.
                if nbf == 2 {
                    if A.read(here + 1, here) == zero {
                        nbf = 3;
                    }
                }
            } else {
                // * Current block consists of two 1-by-1 blocks, each of which
                // * must be swapped individually.
                let mut nbnext = 1;
                if here + 3 <= n - 1 {
                    if A.read(here + 3, here + 2) != zero {
                        nbnext = 2;
                    }
                }

                schur_swap(
                    A.rb_mut(),
                    Q.rb_mut(),
                    here + 1,
                    1,
                    nbnext,
                    epsilon,
                    zero_threshold,
                );

                if nbnext == 1 {
                    // * Swap two 1-by-1 blocks.
                    schur_swap(A.rb_mut(), Q.rb_mut(), here, 1, 1, epsilon, zero_threshold);
                    here += 1;
                } else {
                    // * Recompute NBNEXT in case of 2-by-2 split.
                    if A.read(here + 2, here + 1) == zero {
                        nbnext = 1;
                    }

                    if nbnext == 2 {
                        // * 2-by-2 block did not split.
                        schur_swap(
                            A.rb_mut(),
                            Q.rb_mut(),
                            here,
                            1,
                            nbnext,
                            epsilon,
                            zero_threshold,
                        );
                        here += 2;
                    } else {
                        // * 2-by-2 block did split.
                        schur_swap(A.rb_mut(), Q.rb_mut(), here, 1, 1, epsilon, zero_threshold);
                        here += 1;
                        schur_swap(A.rb_mut(), Q.rb_mut(), here, 1, 1, epsilon, zero_threshold);
                        here += 1;
                    }
                }
            }

            if here >= ilst {
                break;
            }
        }
    } else {
        let mut here = ifst;

        loop {
            // * Swap with next one below.
            if nbf == 1 || nbf == 2 {
                // * Current block either 1-by-1 or 2-by-2.
                let mut nbnext = 1;
                if here >= 2 {
                    if A.read(here - 1, here - 2) != zero {
                        nbnext = 2;
                    }
                }

                schur_swap(
                    A.rb_mut(),
                    Q.rb_mut(),
                    here - nbnext,
                    nbnext,
                    nbf,
                    epsilon,
                    zero_threshold,
                );
                here -= nbnext;

                // * Test if 2-by-2 block breaks into two 1-by-1 blocks.
                if nbf == 2 {
                    if A.read(here + 1, here) == zero {
                        nbf = 3;
                    }
                }
            } else {
                // * Current block consists of two 1-by-1 blocks, each of which
                // * must be swapped individually.
                let mut nbnext = 1;
                if here >= 2 {
                    if A.read(here - 1, here - 2) != zero {
                        nbnext = 2;
                    }
                }

                schur_swap(
                    A.rb_mut(),
                    Q.rb_mut(),
                    here - nbnext,
                    nbnext,
                    1,
                    epsilon,
                    zero_threshold,
                );
                if nbnext == 1 {
                    // * Swap two 1-by-1 blocks.
                    schur_swap(
                        A.rb_mut(),
                        Q.rb_mut(),
                        here,
                        nbnext,
                        1,
                        epsilon,
                        zero_threshold,
                    );
                    here -= 1;
                } else {
                    // * Recompute NBNEXT in case of 2-by-2 split.
                    if A.read(here, here - 1) == zero {
                        nbnext = 1;
                    }
                    if nbnext == 2 {
                        // * 2-by-2 block did not split.
                        schur_swap(
                            A.rb_mut(),
                            Q.rb_mut(),
                            here - 1,
                            2,
                            1,
                            epsilon,
                            zero_threshold,
                        );
                        here -= 2;
                    } else {
                        // * 2-by-2 block did split.
                        schur_swap(A.rb_mut(), Q.rb_mut(), here, 1, 1, epsilon, zero_threshold);
                        here -= 1;
                        schur_swap(A.rb_mut(), Q.rb_mut(), here, 1, 1, epsilon, zero_threshold);
                        here -= 1;
                    }
                }
            }

            if here <= ilst {
                break;
            }
        }
    }
}

fn partial_schur(
    A: MatRef<'_, E>,
    v0: ColRef<'_, E>,
    min_dim: usize,
    max_dim: usize,
    n_eigval: usize,
    tol: E,
    restarts: usize,
) {
    use linalg::evd::hessenberg_real_evd as schur;

    let n = A.nrows();

    let mut stack = GlobalPodBuffer::new(
        schur::multishift_qr_req::<E>(
            max_dim,
            max_dim,
            true,
            true,
            crate::Parallelism::None,
            Default::default(),
        )
        .unwrap(),
    );
    let mut stack = PodStack::new(&mut stack);

    let mut eigval = Mat::<E>::zeros(max_dim, 2);
    let mut residual = Col::<E>::zeros(max_dim);

    let mut H = Mat::<E>::zeros(max_dim + 1, max_dim);
    let mut V = Mat::<E>::zeros(n, max_dim + 1);
    let mut Q = Mat::<E>::zeros(max_dim, max_dim);

    V.col_mut(0).copy_from(v0 / v0.norm_l2());

    iterate_arnoldi(A.as_ref(), H.as_mut(), V.as_mut(), 1, min_dim);
    let mut active = 0usize;
    let mut k = min_dim;

    for iter in 0..restarts {
        dbg!(iter);
        _ = iter;

        iterate_arnoldi(A.as_ref(), H.as_mut(), V.as_mut(), k + 1, max_dim);

        let Hmm = H[(max_dim, max_dim - 1)];

        let n = max_dim - active;
        let (mut w_re, mut w_im) = eigval.get_mut(active..max_dim, ..).two_cols_mut(0, 1);

        Q.fill_zero();
        Q.diagonal_mut().column_vector_mut().fill(1.0);

        let mut Q_slice = Q.get_mut(active..max_dim, active..max_dim);
        let mut H_slice = H.get_mut(active..max_dim, active..max_dim);

        {
            let n = max_dim - active;

            let mut householder = Mat::<f64>::zeros(1, n - 1);
            let mut householder = householder.as_mut();

            linalg::evd::hessenberg::make_hessenberg_in_place(
                H_slice.rb_mut(),
                householder.rb_mut().transpose_mut(),
                Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(
                    linalg::evd::hessenberg::make_hessenberg_in_place_req::<f64>(
                        n,
                        1,
                        Parallelism::None,
                    )
                    .unwrap(),
                )),
            );

            linalg::householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
                H_slice.rb().submatrix(1, 0, n - 1, n - 1),
                householder.rb(),
                crate::Conj::No,
                Q_slice.rb_mut().submatrix_mut(1, 1, n - 1, n - 1),
                Parallelism::None,
                PodStack::new(&mut GlobalPodBuffer::new(
                    linalg::householder::apply_block_householder_sequence_on_the_right_in_place_req::<f64>(
                        n,
                        1,
                        n,
                    )
                    .unwrap(),
                )),
            );

            for j in 0..n {
                for i in j + 2..n {
                    H_slice.write(i, j, E::faer_zero());
                }
            }
        }

        schur::multishift_qr(
            true,
            H_slice.rb_mut(),
            Some(Q_slice.rb_mut()),
            w_re.rb_mut(),
            w_im.rb_mut(),
            0,
            n,
            E::EPSILON,
            E::MIN_POSITIVE,
            crate::Parallelism::None,
            stack.rb_mut(),
            Default::default(),
        );

        let mut j = 0usize;
        while j < n {
            let mut i = j;
            let mut idx = i;
            let mut max = 0.0;
            while i < n {
                let cplx = i + 1 < n && H_slice[(i + 1, i)] != 0.0;
                let (v, bs) = if cplx {
                    (E::hypot(w_re[i], w_im[i]), 2)
                } else {
                    (w_re[i].abs(), 1)
                };

                if v > max {
                    max = v;
                    idx = i;
                }

                i += bs;
            }

            let i = idx;
            let cplx = i + 1 < n && H_slice[(i + 1, i)] != 0.0;
            let bs = if cplx { 2 } else { 1 };
            if i != j {
                reorder_schur(H_slice.rb_mut(), Some(Q_slice.rb_mut()), i, j);

                let x_re = w_re.rb_mut().try_as_slice_mut().unwrap();
                let x_im = w_im.rb_mut().try_as_slice_mut().unwrap();

                for x in [x_re, x_im] {
                    x[j..i + bs].rotate_right(bs)
                }
            }

            j += bs;
        }

        let mut X = Mat::<E>::zeros(n, n);
        linalg::evd::real_schur_to_eigen(H_slice.rb(), X.as_mut(), crate::Parallelism::None);
        let vecs = &Q_slice * &X;

        let H_tmp = H.get(..active, ..) * &Q;
        H.get_mut(..active, ..).copy_from(&H_tmp);

        // AV = VH
        // x in span(V)
        // Ax = AV y = (VH + f e*) y = k V y + f e* y = kx + f * y[-1]
        let mut j = 0usize;
        while j < n {
            let re = vecs[(max_dim - active - 1, j)];
            if w_im[j] != 0.0 {
                let im = vecs[(max_dim - active - 1, j + 1)];
                let res = E::hypot(Hmm * re, Hmm * im);
                residual[active + j] = res;
                residual[active + j + 1] = res;
                j += 2;
            } else {
                residual[active + j] = (Hmm * re).abs();
                j += 1;
            }
        }

        #[derive(Copy, Clone, PartialEq, Eq, Debug)]
        enum Group {
            Lock,
            Retain,
            Purge,
        }

        let mut groups = vec![Group::Purge; max_dim];

        let mut nev = n_eigval;
        if w_im[nev - active - 1] > 0.0 {
            nev += 1;
        }

        let mut nlock = 0usize;
        for j in 0..nev {
            if residual[j] <= tol {
                groups[j] = Group::Lock;
                nlock += 1;
            } else {
                groups[j] = Group::Retain;
            }
        }

        let ideal_size = Ord::min(nlock + min_dim, (min_dim + max_dim) / 2);
        k = nev;

        let mut i = nev;
        while i < max_dim {
            let cplx = eigval[(i, 1)] != 0.0;
            let bs = if cplx { 2 } else { 1 };

            let group;
            if k < ideal_size && residual[i] > tol {
                group = Group::Retain;
                k += bs;
            } else {
                group = Group::Purge;
            }

            for k in 0..bs {
                groups[i + k] = group;
            }
            i += bs;
        }

        let mut purge = 0usize;
        while purge < active && groups[purge] == Group::Lock {
            purge += 1;
        }

        let mut lo = 0usize;
        let mut mi = 0usize;
        let mut hi = 0usize;

        while hi < max_dim {
            let cplx = hi + 1 < max_dim && H[(hi + 1, hi)] != 0.0;
            let bs = if cplx { 2 } else { 1 };

            match groups[hi] {
                Group::Lock => {
                    reorder_schur(H.get_mut(..max_dim, ..max_dim), Some(Q.as_mut()), hi, lo);
                    for k in 0..bs {
                        residual[lo + k] = residual[hi + k];
                    }

                    lo += bs;
                    mi += bs;
                    hi += bs;
                }
                Group::Retain => {
                    reorder_schur(H.get_mut(..max_dim, ..max_dim), Some(Q.as_mut()), hi, mi);
                    mi += bs;
                    hi += bs;
                }
                Group::Purge => {
                    hi += bs;
                }
            }
        }

        let V_tmp = &V.get(.., purge..max_dim) * &Q.get(purge..max_dim, purge..k);
        V.get_mut(.., purge..k).copy_from(&V_tmp);
        let b_tmp = &H.get(max_dim, ..) * &Q;
        H.get_mut(max_dim, ..).copy_from(b_tmp);

        let (mut x, y) = V.two_cols_mut(k, max_dim);
        x.copy_from(&y);

        let (mut x, mut y) = H.two_rows_mut(k, max_dim);
        x.copy_from(&y);
        y.fill_zero();

        active = nlock;
        if nlock >= n_eigval {
            dbg!(iter);
            break;
        }
    }

    if true {
        let n = active;

        let H = H.get(..n, ..n);
        let V = V.get(.., ..n);
        let mut X = Mat::<E>::zeros(n, n);
        linalg::evd::real_schur_to_eigen(H, X.as_mut(), crate::Parallelism::None);
        let V = V * &X;

        let mut j = 0usize;
        while j < n {
            let cplx = j + 1 < n && H[(j + 1, j)] != 0.0;
            let bs = if cplx { 2 } else { 1 };

            if cplx {
                use num_complex::Complex;
                let v_re = V.col(j);
                let v_im = V.col(j + 1);

                let Complex { re, im } = H.submatrix(j, j, bs, bs).eigenvalues::<Complex<E>>()[0];

                dbg!((A * v_re - (re * v_re - im * v_im)).norm_l2());
                dbg!((A * v_im - (re * v_im + im * v_re)).norm_l2());
            } else {
                let v = V.col(j);
                dbg!((A * v - H[(j, j)] * v).norm_l2());
            }

            j += bs;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_arnoldi() {
        let rng = &mut StdRng::seed_from_u64(1);
        let n = 2000;
        let n_eigval = 20;
        let min_dim = 30;
        let max_dim = 60;
        let restarts = 1000;

        let mat = crate::stats::StandardNormalMat { nrows: n, ncols: n };
        let col = crate::stats::StandardNormalCol { nrows: n };
        let A: Mat<f64> = mat.sample(rng);
        let A = &A + A.adjoint();

        let mut v0: Col<f64> = col.sample(rng);
        v0 /= v0.norm_l2();
        let A = A.as_ref();
        let v0 = v0.as_ref();

        partial_schur(
            A,
            v0,
            min_dim,
            max_dim,
            n_eigval,
            f64::EPSILON * 128.0,
            restarts,
        );
    }
}
