// Algorithm ported from Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
// Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
// Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
// Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>
// Copyright (C) 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2014-2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use linalg::jacobi::JacobiRotation;

use crate::{internal_prelude::*, perm::swap_cols_idx};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SvdError {
    NoConvergence,
}

#[math]
fn bidiag_to_mat<'N, C: RealContainer, T: RealField<C>>(
    diag: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    subdiag: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
) -> Mat<C, T, Dim<'N>, Dim<'N>> {
    let n = diag.nrows();
    let ctx: &Ctx<C, T> = &ctx();
    let mut m = Mat::zeros_with(ctx, n, n);

    help!(C);
    {
        let mut m = m.as_mut();
        for i in n.indices() {
            write1!(m[(i, i)] = math(copy(diag[i])));
            if let Some(i1) = n.try_check(*i + 1) {
                write1!(m[(i1, i)] = math(copy(subdiag[i])));
            }
        }
    }
    m
}

#[math]
fn qr_algorithm<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,

    diag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    subdiag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    u: Option<MatMut<'_, C, T, Dim<'N>, Dim<'N>>>,
    v: Option<MatMut<'_, C, T, Dim<'N>, Dim<'N>>>,

    max_iters: usize,
) -> Result<(), SvdError> {
    let n = diag.nrows();
    let eps = math(eps());
    let sml = math(min_positive());

    if *n == 0 {
        return Ok(());
    }

    let last = n.idx(*n - 1);

    let max = math.max(diag.norm_max_with(ctx), subdiag.norm_max_with(ctx));
    if math.is_zero(max) {
        return Ok(());
    }
    help!(C);

    let max_inv = math(recip(max));

    let mut diag = diag;
    let mut subdiag = subdiag;
    let mut u = u;
    let mut v = v;

    for mut x in diag.rb_mut().iter_mut() {
        write1!(x, math(x * max_inv));
    }
    for mut x in subdiag.rb_mut().iter_mut() {
        write1!(x, math(x * max_inv));
    }

    {
        let eps2 = math(eps * eps);

        for iter in 0..max_iters {
            for i0 in zero().to(last.excl()) {
                let i1 = n.idx(*i0 + 1);
                if math(abs2(subdiag[i0]) <= eps2 * abs(diag[i0] * diag[i1]) + sml) {
                    write1!(subdiag[i0] = math.zero());
                }
            }

            for i in n.indices() {
                if math(abs(diag[i]) <= eps) {
                    write1!(diag[i] = math.zero());
                }
            }

            let mut end = n.end();
            while *end >= 2 && math(abs2(subdiag[n.idx(*end - 2)]) <= sml) {
                end = n.idx_inc(*end - 1)
            }

            if *end == 1 {
                break;
            }

            let mut start = n.idx(*end - 1);
            while *start >= 1 && !math(is_zero(subdiag[n.idx(*start - 1)])) {
                start = n.idx(*start - 1);
            }

            let mut found_zero_diag = false;

            for i in start.to_incl().to(n.idx_inc(*end - 1)) {
                if math(is_zero(diag[i])) {
                    found_zero_diag = true;

                    let mut val = math(copy(subdiag[i]));
                    write1!(subdiag[i] = math(zero()));
                    for j in i.next().to(end) {
                        let rot = math(JacobiRotation::make_givens(ctx, copy(diag[j]), copy(val)));
                        write1!(diag[j] = math(rot.c * diag[j] - rot.s * val));
                        if j.next() < end {
                            val = math(-rot.s * subdiag[j]);
                            write1!(subdiag[j] = math(rot.c * subdiag[j]));
                        }
                        if let Some(v) = v.rb_mut() {
                            let (i, j) = v.two_cols_mut(i, j);
                            rot.apply_on_the_right_in_place(ctx, i, j);
                        }
                    }
                }
            }

            if found_zero_diag {
                continue;
            }

            let end2 = n.idx(*end - 2);
            let end1 = n.idx(*end - 1);

            let t00 = if *end - *start == 2 {
                math(abs2(diag[end2]))
            } else {
                math(abs2(diag[end2] + abs2(subdiag[n.idx(*end - 3)])))
            };
            let t11 = math(abs2(diag[end1]) + abs2(subdiag[end2]));
            let t01 = math(diag[end2] * subdiag[end2]);

            let t01_2 = math(abs2(t01));

            let mu;
            if math(t01_2 > sml) {
                let d = math(mul_pow2((t00 - t11), from_f64(0.5)));
                let mut delta = math(sqrt(abs2(d) + t01_2));
                if math(lt_zero(d)) {
                    delta = math(-delta);
                }
                mu = math(t11 - (t01_2 / (d + delta)));
            } else {
                mu = math(copy(t11));
            }

            let mut y = math(abs2(diag[start]) - mu);
            let mut z = math(diag[start] * subdiag[start]);

            for k in start.to_incl().to(end1.excl()) {
                let rot = JacobiRotation::make_givens(ctx, math(copy(y)), math(copy(z)));
                if k > start {
                    write1!(subdiag[n.idx(*k - 1)] = math(abs(rot.c * y - rot.s * z)));
                }

                let mut diag_k = math(copy(diag[k]));

                let tmp = math((
                    rot.c * diag_k - rot.s * subdiag[k],
                    rot.s * diag_k + rot.c * subdiag[k],
                ));

                diag_k = tmp.0;
                write1!(subdiag[k] = tmp.1);

                let k1 = n.idx(*k + 1);

                y = math(copy(diag_k));
                z = math(-rot.s * diag[k1]);
                write1!(diag[k1] = math(rot.c * diag[k1]));

                if let Some(u) = u.rb_mut() {
                    let (k, k1) = u.two_cols_mut(k, k1);
                    rot.apply_on_the_right_in_place(ctx, k, k1);
                }

                let rot = JacobiRotation::make_givens(ctx, math(copy(y)), math(copy(z)));

                diag_k = math(abs(rot.c * y - rot.s * z));
                write1!(diag[k] = diag_k);

                let tmp = math((
                    rot.c * subdiag[k] - rot.s * diag[k1],
                    rot.s * subdiag[k] + rot.c * diag[k1],
                ));
                write1!(subdiag[k] = tmp.0);
                write1!(diag[k1] = tmp.1);

                if *k < *end - 2 {
                    y = math(copy(subdiag[k]));
                    z = math(-rot.s * subdiag[k1]);
                    write1!(subdiag[k1] = math(rot.c * subdiag[k1]));
                }

                if let Some(v) = v.rb_mut() {
                    let (k, k1) = v.two_cols_mut(k, k1);
                    rot.apply_on_the_right_in_place(ctx, k, k1);
                }
            }

            if iter + 1 == max_iters {
                for mut x in diag.rb_mut().iter_mut() {
                    write1!(x, math(x * max));
                }
                for mut x in subdiag.rb_mut().iter_mut() {
                    write1!(x, math(x * max));
                }

                return Err(SvdError::NoConvergence);
            }
        }
    }

    for j in n.indices() {
        let mut d = diag.rb_mut().at_mut(j);

        if math(lt_zero(d)) {
            write1!(d, math(-d));
            if let Some(mut v) = v.rb_mut() {
                for i in n.indices() {
                    write1!(v[(i, j)] = math(-v[(i, j)]));
                }
            }
        }
    }

    for k in n.indices() {
        let mut max = math(zero());
        let mut idx = k;

        for kk in k.to_incl().to(n.end()) {
            if math(diag[kk] > max) {
                max = math(copy(diag[kk]));
                idx = kk;
            }
        }

        if k != idx {
            let dk = math(copy(diag[k]));
            let di = math(copy(diag[idx]));

            write1!(diag[idx] = dk);
            write1!(diag[k] = di);

            if let Some(u) = u.rb_mut() {
                swap_cols_idx(ctx, u, k, idx);
            }
            if let Some(v) = v.rb_mut() {
                swap_cols_idx(ctx, v, k, idx);
            }
        }
    }

    for mut x in diag.rb_mut().iter_mut() {
        write1!(x, math(x * max));
    }
    for mut x in subdiag.rb_mut().iter_mut() {
        write1!(x, math(x * max));
    }
    Ok(())
}

#[math]
fn secular_eq<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    mu: C::Of<T>,
    col0_perm: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    diag_perm: ColRef<'_, C, T, Dim<'N>, ContiguousFwd>,
    shift: C::Of<T>,
) -> C::Of<T> {
    let mut res = math(one());

    let n = diag_perm.nrows();
    for i in n.indices() {
        let c = math(col0_perm[i]);
        let d = math(diag_perm[i]);
        res = math(res + (c / ((d - shift) - mu)) * (c / ((d + shift) + mu)));
    }

    res
}

#[math]
fn deflate<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    diag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    col0: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    jacobi_coeff: &mut [JacobiRotation<C, T>],
    jacobi_idx: &mut [Idx<'N>],
    transpositions: &mut [Idx<'N>],
    perm: &mut [Idx<'N>],
    k: usize,
    stack: &mut DynStack,
) -> (usize, usize) {
    help!(C);
    let mut diag = diag;
    let mut col0 = col0;

    let n = diag.nrows();
    let first = n.idx(0);

    let mut jacobi_0i = 0;
    let mut jacobi_ij = 0;

    let (max_diag, max_col0) = (
        diag.rb()
            .subrows_range((first.next(), n.end()))
            .norm_max_with(ctx),
        col0.norm_max_with(ctx),
    );
    let max = math(max(max_diag, max_col0));

    let eps = math(eps());
    let sml = math(min_positive());

    let eps_strict = math(max(eps * max_diag, sml));

    let eps_coarse = math(from_f64(8.0) * eps * max);

    // condition 4.1
    if math(diag[first] < eps_coarse) {
        write1!(diag[first] = math.copy(eps_coarse));
        write1!(col0[first] = math.copy(eps_coarse));
    }

    // condition 4.2
    for i in first.next().to(n.end()) {
        if math(abs(col0[i]) < eps_strict) {
            write1!(col0[i] = math(zero()));
        }
    }

    // condition 4.3
    for i in first.next().to(n.end()) {
        if math(diag[i] < eps_coarse) {
            if let Some(rot) = deflation_43(ctx, diag.rb_mut(), col0.rb_mut(), i) {
                jacobi_coeff[jacobi_0i] = rot;
                jacobi_idx[jacobi_0i] = i;
                jacobi_0i += 1;
            }
        }
    }

    let mut total_deflation = true;
    for i in first.next().to(n.end()) {
        if math(!(abs(col0[i]) < sml)) {
            total_deflation = false;
            break;
        }
    }

    perm[0] = first;
    let mut p = 1;
    for i in first.next().to(n.end()) {
        if math(abs(diag[i]) < sml) {
            perm[p] = i;
            p += 1;
        }
    }

    let mut i = 1;
    let mut j = k + 1;

    for p in &mut perm[p..] {
        if i >= k + 1 {
            *p = n.idx(j);
            j += 1;
        } else if j >= *n {
            *p = n.idx(i);
            i += 1;
        } else if math(diag[n.idx(i)] < diag[n.idx(j)]) {
            *p = n.idx(j);
            j += 1;
        } else {
            *p = n.idx(i);
            i += 1;
        }
    }

    if total_deflation {
        for i in first.next().to(n.end()) {
            let pi = perm[*i];
            if math(abs(diag[pi]) < sml || diag[pi] > diag[first]) {
                perm[*i - 1] = perm[*i];
            } else {
                perm[*i - 1] = first;
                break;
            }
        }
    }

    let (mut real_ind, stack) = stack.collect(n.indices());
    let (mut real_col, _) = stack.collect(n.indices());

    for i in (if total_deflation {
        zero()
    } else {
        first.next()
    })
    .to(n.end())
    {
        let pi = perm[*n - if total_deflation { *i + 1 } else { *i }];
        let j = real_col[*pi];

        let (a, b) = math((copy(diag[i]), copy(diag[j])));
        write1!(diag[i] = b);
        write1!(diag[j] = a);

        if *i != 0 && *j != 0 {
            let (a, b) = math((copy(col0[i]), copy(col0[j])));
            write1!(col0[i] = b);
            write1!(col0[j] = a);
        }

        transpositions[*i] = j;
        let real_i = real_ind[*i];
        real_col[*real_i] = j;
        real_col[*pi] = i;
        real_ind[*j] = real_i;
        real_ind[*i] = pi;
    }

    write1!(col0[first] = math(copy(diag[first])));

    for i in n.indices() {
        perm[*i] = i;
    }

    for (i, &j) in transpositions.iter().enumerate() {
        perm.swap(i, *j);
    }

    // condition 4.4
    let mut i = n.idx(*n - 1);
    while *i > 0 && math((abs(diag[i]) < sml || abs(col0[i]) < sml)) {
        i = n.idx(*i - 1);
    }

    while *i > 1 {
        let i1 = n.idx(*i - 1);
        if math(diag[i] - diag[i1] < eps_strict) {
            if let Some(rot) = deflation_44(ctx, diag.rb_mut(), col0.rb_mut(), i1, i) {
                jacobi_coeff[jacobi_0i + jacobi_ij] = rot;
                jacobi_idx[jacobi_0i + jacobi_ij] = i;
                jacobi_ij += 1;
            }
        }

        i = i1;
    }

    (jacobi_0i, jacobi_ij)
}

#[math]
fn deflation_43<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    diag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    col0: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    i: Idx<'N>,
) -> Option<JacobiRotation<C, T>> {
    let mut diag = diag;
    let mut col0 = col0;

    let n = diag.nrows();
    let first = n.idx(0);
    help!(C);

    let p = math(copy(col0[first]));
    let q = math(copy(col0[i]));

    if math(is_zero(p) && is_zero(q)) {
        write1!(diag[i] = math(zero()));
        return None;
    }

    let rot = JacobiRotation::make_givens(ctx, math.copy(p), math.copy(q));

    let r = math(rot.c * p - rot.s * q);
    write1!(col0[first] = math.copy(r));
    write1!(diag[first] = math.copy(r));
    write1!(col0[i] = math.zero());
    write1!(diag[i] = math.zero());

    Some(rot)
}

#[math]
fn deflation_44<'N, C: RealContainer, T: RealField<C>>(
    ctx: &Ctx<C, T>,
    diag: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    col0: ColMut<'_, C, T, Dim<'N>, ContiguousFwd>,
    i: Idx<'N>,
    j: Idx<'N>,
) -> Option<JacobiRotation<C, T>> {
    let mut diag = diag;
    let mut col0 = col0;

    help!(C);

    let p = math(copy(col0[i]));
    let q = math(copy(col0[j]));

    if math(is_zero(p) && is_zero(q)) {
        write1!(diag[i] = math(copy(diag[j])));
        return None;
    }

    let rot = JacobiRotation::make_givens(ctx, math.copy(p), math.copy(q));

    let r = math(rot.c * p - rot.s * q);
    write1!(col0[i] = math.copy(r));
    write1!(col0[j] = math.zero());
    write1!(diag[i] = math(copy(diag[j])));

    Some(rot)
}

#[cfg(test)]
mod tests {
    use faer_traits::Unit;

    use super::*;
    use crate::{assert, utils::approx::*, Col, Mat, MatMut};
    use std::{
        mem::MaybeUninit,
        path::{Path, PathBuf},
    };

    #[math]
    fn bidiag_to_mat<'N>(
        diag: ColRef<'_, Unit, f64, Dim<'N>, ContiguousFwd>,
        subdiag: ColRef<'_, Unit, f64, Dim<'N>, ContiguousFwd>,
    ) -> Mat<f64, Dim<'N>, Dim<'N>> {
        let n = diag.nrows();
        let ctx: &Ctx<Unit, f64> = &ctx();
        let mut m = Mat::zeros_with(ctx, n, n);

        help!(Unit);
        {
            let mut m = m.as_mut();
            for i in n.indices() {
                write1!(m[(i, i)] = math(copy(diag[i])));
                if let Some(i1) = n.try_check(*i + 1) {
                    write1!(m[(i1, i)] = math(copy(subdiag[i])));
                }
            }
        }
        m
    }

    fn parse_bidiag(path: &Path) -> (Col<f64>, Col<f64>) {
        let file = &*std::fs::read_to_string(path).unwrap();
        let mut diag = vec![];
        let mut subdiag = vec![];

        let mut iter = file.lines();
        for line in &mut iter {
            if line.starts_with("diag") {
                continue;
            }
            if line.starts_with("subdiag") {
                break;
            }

            let line = line.trim();
            let line = line.strip_suffix(",").unwrap_or(line);

            if line.is_empty() {
                continue;
            }

            diag.push(line.parse::<f64>().unwrap());
        }
        for line in iter {
            let line = line.trim();
            let line = line.strip_suffix(",").unwrap_or(line);

            if line.is_empty() {
                continue;
            }
            subdiag.push(line.parse::<f64>().unwrap());
        }

        assert!(diag.len() == subdiag.len());

        subdiag[diag.len() - 1] = 0.0;

        (
            Col::from_fn(diag.len(), |i| diag[i]),
            Col::from_fn(diag.len(), |i| subdiag[i]),
        )
    }

    #[test]
    fn test_qr_algorithm() {
        for file in
            std::fs::read_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/svd/"))
                .unwrap()
        {
            let (diag, subdiag) = parse_bidiag(&file.unwrap().path());
            with_dim!(n, diag.nrows());
            if *n > 512 {
                continue;
            }

            let diag = diag.as_ref().as_row_shape(n).try_as_col_major().unwrap();
            let subdiag = subdiag.as_ref().as_row_shape(n).try_as_col_major().unwrap();

            let mut d = diag.to_owned();
            let mut subd = subdiag.to_owned();

            let mut u = Mat::zeros_with(&ctx(), n, n);
            let mut v = Mat::zeros_with(&ctx(), n, n);

            for i in n.indices() {
                u[(i, i)] = 1.0;
                v[(i, i)] = 1.0;
            }

            qr_algorithm(
                &ctx(),
                d.as_mut().try_as_col_major_mut().unwrap(),
                subd.as_mut().try_as_col_major_mut().unwrap(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                30 * *n * *n,
            )
            .unwrap();

            for &x in subd.iter() {
                assert!(x == 0.0);
            }

            let mut approx_eq = CwiseMat(ApproxEq::<Unit, f64>::eps());
            approx_eq.0.abs_tol *=
                f64::max(diag.norm_max(), subdiag.norm_max()) * (*n as f64).sqrt();
            approx_eq.0.rel_tol *=
                f64::max(diag.norm_max(), subdiag.norm_max()) * (*n as f64).sqrt();
            let reconstructed = &u * d.as_diagonal() * v.adjoint();

            assert!(reconstructed ~ bidiag_to_mat(diag, subdiag));
        }
    }

    #[test]
    fn test_deflation43() {
        let approx_eq = CwiseMat(ApproxEq::<Unit, f32>::eps());

        let n = 7;
        let k = 3;
        let diag_orig = &*vec![1.0, 5.0, 3.0, 1e-7, 4.0, 2.0, 2e-7_f32];
        let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];

        with_dim!(n, n);

        let n_jacobi = 2;
        let jacobi_coeffs = &mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
        let jacobi_indices = &mut *vec![n.idx(0); n_jacobi];
        let perm = &mut *vec![n.idx(0); *n];
        let diag = &mut *(diag_orig.to_vec());
        let col = &mut *(col_orig.to_vec());

        let mut diag = MatMut::from_column_major_slice_mut(diag, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();
        let mut col = MatMut::from_column_major_slice_mut(col, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();

        let (jacobi_0i, jacobi_ij) = deflate(
            &ctx(),
            diag.rb_mut(),
            col.rb_mut(),
            jacobi_coeffs,
            jacobi_indices,
            &mut *vec![n.idx(0); *n],
            perm,
            k,
            DynStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * *n]),
        );
        assert!(all(jacobi_0i == 2, jacobi_ij == 0));

        let perm_inv = &mut *vec![n.idx(0); *n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[*p] = n.idx(i);
        }

        let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);

        let mut M_orig = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M_orig[(i, i)] = diag_orig[*i];
            M_orig[(i, n.idx(0))] = col_orig[*i];
        }

        let mut M = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M[(i, i)] = diag[i];
            M[(i, n.idx(0))] = col[i];
        }

        M = P.inverse() * M * P;

        for (&rot, &i) in core::iter::zip(&*jacobi_coeffs, &*jacobi_indices).rev() {
            let (x, y) = M.two_rows_mut(n.idx(0), i);
            rot.apply_on_the_left_in_place(&ctx(), x, y);
        }

        assert!(M ~ M_orig);
    }

    #[test]
    fn test_deflation44() {
        let approx_eq = CwiseMat(ApproxEq::<Unit, f32>::eps());

        let n = 7;
        let k = 3;
        let diag_orig = &*vec![1.0, 5.0, 3.0, 1.0, 4.0, 2.0, 1.0];
        let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];

        with_dim!(n, n);

        let n_jacobi = 1;
        let jacobi_coeffs = &mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
        let jacobi_indices = &mut *vec![n.idx(0); n_jacobi];
        let perm = &mut *vec![n.idx(0); *n];
        let diag = &mut *(diag_orig.to_vec());
        let col = &mut *(col_orig.to_vec());

        let mut diag = MatMut::from_column_major_slice_mut(diag, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();
        let mut col = MatMut::from_column_major_slice_mut(col, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();

        let (jacobi_0i, jacobi_ij) = deflate(
            &ctx(),
            diag.rb_mut(),
            col.rb_mut(),
            jacobi_coeffs,
            jacobi_indices,
            &mut *vec![n.idx(0); *n],
            perm,
            k,
            DynStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * *n]),
        );
        assert!(all(jacobi_0i == 0, jacobi_ij == 1));

        let perm_inv = &mut *vec![n.idx(0); *n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[*p] = n.idx(i);
        }

        let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);

        let mut M_orig = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M_orig[(i, i)] = diag_orig[*i];
            M_orig[(i, n.idx(0))] = col_orig[*i];
        }

        let mut M = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M[(i, i)] = diag[i];
            M[(i, n.idx(0))] = col[i];
        }

        M = P.inverse() * M * P;

        for (&rot, &i) in core::iter::zip(&*jacobi_coeffs, &*jacobi_indices).rev() {
            let (i, j) = (n.idx(*i - 1), i);
            let (pi, pj) = (perm[*i], perm[*j]);

            let (x, y) = M.two_rows_mut(pi, pj);
            rot.apply_on_the_left_in_place(&ctx(), x, y);

            let (x, y) = M.two_cols_mut(pi, pj);
            rot.transpose(&ctx())
                .apply_on_the_right_in_place(&ctx(), x, y);
        }

        assert!(M ~ M_orig);
    }

    #[test]
    fn test_both_deflation() {
        let approx_eq = CwiseMat(ApproxEq::<Unit, f32>::eps());

        let n = 7;
        let k = 3;
        let diag_orig = &*vec![1.0, 5.0, 3.0, 2.0, 4.0, 2.0, 0.0];
        let col_orig = &*vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32];

        with_dim!(n, n);

        let n_jacobi = 2;
        let jacobi_coeffs = &mut *vec![JacobiRotation { c: 0.0, s: 0.0 }; n_jacobi];
        let jacobi_indices = &mut *vec![n.idx(0); n_jacobi];
        let perm = &mut *vec![n.idx(0); *n];
        let diag = &mut *(diag_orig.to_vec());
        let col = &mut *(col_orig.to_vec());

        let mut diag = MatMut::from_column_major_slice_mut(diag, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();
        let mut col = MatMut::from_column_major_slice_mut(col, n, 1)
            .col_mut(0)
            .try_as_col_major_mut()
            .unwrap();

        let (jacobi_0i, jacobi_ij) = deflate(
            &ctx(),
            diag.rb_mut(),
            col.rb_mut(),
            jacobi_coeffs,
            jacobi_indices,
            &mut *vec![n.idx(0); *n],
            perm,
            k,
            DynStack::new_any(&mut *vec![MaybeUninit::new(0usize); 2 * *n]),
        );
        assert!(all(jacobi_0i == 1, jacobi_ij == 1));

        let perm_inv = &mut *vec![n.idx(0); *n];
        for (i, &p) in perm.iter().enumerate() {
            perm_inv[*p] = n.idx(i);
        }

        let P = crate::perm::PermRef::new_checked(perm, perm_inv, n);

        let mut M_orig = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M_orig[(i, i)] = diag_orig[*i];
            M_orig[(i, n.idx(0))] = col_orig[*i];
        }

        let mut M = Mat::zeros_with(&ctx(), n, n);
        for i in n.indices() {
            M[(i, i)] = diag[i];
            M[(i, n.idx(0))] = col[i];
        }

        M = P.inverse() * M * P;

        for (&rot, &i) in core::iter::zip(&jacobi_coeffs[1..], &jacobi_indices[1..]).rev() {
            let (i, j) = (n.idx(*i - 1), i);
            let (pi, pj) = (perm[*i], perm[*j]);

            let (x, y) = M.two_rows_mut(pi, pj);
            rot.apply_on_the_left_in_place(&ctx(), x, y);

            let (x, y) = M.two_cols_mut(pi, pj);
            rot.transpose(&ctx())
                .apply_on_the_right_in_place(&ctx(), x, y);
        }

        for (&rot, &i) in core::iter::zip(&jacobi_coeffs[..1], &jacobi_indices[..1]).rev() {
            let (x, y) = M.two_rows_mut(n.idx(0), i);
            rot.apply_on_the_left_in_place(&ctx(), x, y);
        }

        assert!(M ~ M_orig);
    }
}
