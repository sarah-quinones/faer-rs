//! Computes the pseudo inverse; see:
//! https://en.wikipedia.org//wiki/Singular_value_decomposition#Pseudoinverse

use crate::{ColRef, Mat, MatRef};
use faer_entity::{ComplexField, RealField};

/// See: https://en.wikipedia.org//wiki/Singular_value_decomposition#Pseudoinverse
pub(crate) fn compute_pseudo_inverse<E: ComplexField + RealField>(
    s: ColRef<E>,
    u: MatRef<E>,
    v: MatRef<E>,
) -> Mat<E> {
    let epsilon = E::Real::faer_epsilon();
    let (m, n) = (u.nrows(), v.nrows());
    let min_mn = if m < n { m } else { n };
    let vt = v.transpose();
    let s_max = {
        let mut max = E::faer_zero();
        for i in 0..s.ncols() {
            // f already non-negative
            let f: E = s.read(i);
            if f > max {
                max = f;
            }
        }
        max
    };
    let sv_tolerance = epsilon.faer_mul(s_max);
    // now compute the pseudo-inverse
    let mut ai: Mat<E> = Mat::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            for k in 0..min_mn {
                let sv_k = s.read(k);
                if sv_k > sv_tolerance {
                    let val = vt.read(k, i).faer_mul(u.read(j, k)).faer_div(sv_k);
                    unsafe { ai.write_unchecked(i, j, ai.read(i, j).faer_add(val)) };
                }
            }
        }
    }
    ai
}
