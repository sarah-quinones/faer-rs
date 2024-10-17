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
#[cfg(test)]
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
                let d = math((t00 - t11) * from_f64(0.5));
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

#[cfg(test)]
mod tests {
    use faer_traits::Unit;

    use super::*;
    use crate::{assert, utils::approx::*, Col, Mat};
    use std::path::{Path, PathBuf};

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
}
