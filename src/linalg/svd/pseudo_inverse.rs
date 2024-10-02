//! Computes the pseudo inverse; see:
//! https://en.wikipedia.org//wiki/Singular_value_decomposition#Pseudoinverse

use crate::prelude::*;
use faer_entity::{ComplexField, RealField};

/// See: <https://en.wikipedia.org//wiki/Singular_value_decomposition#Pseudoinverse>
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
    while r < s.nrows() && s.read(r).faer_real() > sv_tolerance {
        r += 1;
    }

    let s_inv = zipped!(__rw, s.get(..r))
        .map(|unzipped!(s)| E::faer_from_real(s.read().faer_real().faer_inv()));

    (v.get(.., ..r) * s_inv.as_ref().column_vector_as_diagonal()) * u.get(.., ..r).adjoint()
}
