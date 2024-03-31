//! Computes the pseudo inverse; see:
//! https://en.wikipedia.org//wiki/Singular_value_decomposition#Pseudoinverse

use crate::prelude::*;
use faer_entity::{ComplexField, RealField};

/// See: https://en.wikipedia.org//wiki/Singular_value_decomposition#Pseudoinverse
pub(crate) fn compute_pseudoinverse<E: ComplexField>(
    s: ColRef<'_, E>,
    u: MatRef<'_, E>,
    v: MatRef<'_, E>,
) -> Mat<E> {
    if s.nrows() == 0 {
        return Mat::zeros(v.nrows(), u.nrows());
    }

    let epsilon = E::Real::faer_epsilon().faer_scale_power_of_two(E::Real::faer_from_f64(8.0));

    let s_max = s.read(0).faer_real();
    let sv_tolerance = epsilon.faer_mul(s_max);

    let mut r = 0usize;
    let mut s_r = s_max;
    let mut s_inv = Mat::<E>::zeros(v.nrows(), u.nrows());
    while s_r > sv_tolerance {
        s_inv.write(r, r, E::faer_from_real(s_r.faer_inv()));
        r += 1;
        if r < s.nrows() {
            s_r = s.read(r).faer_real();
        } else {
            break;
        }
    }

    (v.get(.., ..r) * s_inv.get(..r, ..r)) * u.get(.., ..r).adjoint()
}
