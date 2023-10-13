use super::CholeskyError;
use crate::{
    ldlt_diagonal::update::{delete_rows_and_cols_triangular, rank_update_indices},
    llt::compute::{cholesky_in_place, cholesky_in_place_req},
};
#[cfg(feature = "std")]
use assert2::{assert, debug_assert};
use core::{iter::zip, slice};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    mul, mul::triangular::BlockStructure, simd::slice_as_mut_simd, solve, temp_mat_req,
    temp_mat_uninit, zipped, ComplexField, Entity, MatMut, Parallelism, SimdCtx,
};
use reborrow::*;

struct RankUpdateStepImpl<'a, E: Entity, const R: usize> {
    l_col: E::Group<&'a mut [E::Unit]>,
    w: [E::Group<&'a mut [E::Unit]>; R],
    neg_wj_over_ljj_array: [E; R],
    alpha_wj_over_nljj_array: [E; R],
    nljj_over_ljj_array: [E; R],
}

impl<'a, E: ComplexField> pulp::WithSimd for RankUpdateStepImpl<'a, E, 4> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            l_col,
            w,
            neg_wj_over_ljj_array,
            alpha_wj_over_nljj_array,
            nljj_over_ljj_array,
        } = self;

        let [w0, w1, w2, w3] = w;
        let [neg_wj_over_ljj0, neg_wj_over_ljj1, neg_wj_over_ljj2, neg_wj_over_ljj3] =
            neg_wj_over_ljj_array;
        let [nljj_over_ljj0, nljj_over_ljj1, nljj_over_ljj2, nljj_over_ljj3] = nljj_over_ljj_array;
        let [alpha_wj_over_nljj0, alpha_wj_over_nljj1, alpha_wj_over_nljj2, alpha_wj_over_nljj3] =
            alpha_wj_over_nljj_array;

        let (l_head, l_tail) = slice_as_mut_simd::<E, S>(l_col);
        let (w0_head, w0_tail) = slice_as_mut_simd::<E, S>(w0);
        let (w1_head, w1_tail) = slice_as_mut_simd::<E, S>(w1);
        let (w2_head, w2_tail) = slice_as_mut_simd::<E, S>(w2);
        let (w3_head, w3_tail) = slice_as_mut_simd::<E, S>(w3);

        {
            let neg_wj_over_ljj0 = E::faer_simd_splat(simd, neg_wj_over_ljj0);
            let neg_wj_over_ljj1 = E::faer_simd_splat(simd, neg_wj_over_ljj1);
            let neg_wj_over_ljj2 = E::faer_simd_splat(simd, neg_wj_over_ljj2);
            let neg_wj_over_ljj3 = E::faer_simd_splat(simd, neg_wj_over_ljj3);
            let nljj_over_ljj0 = E::faer_simd_splat(simd, nljj_over_ljj0);
            let nljj_over_ljj1 = E::faer_simd_splat(simd, nljj_over_ljj1);
            let nljj_over_ljj2 = E::faer_simd_splat(simd, nljj_over_ljj2);
            let nljj_over_ljj3 = E::faer_simd_splat(simd, nljj_over_ljj3);
            let alpha_wj_over_nljj0 = E::faer_simd_splat(simd, alpha_wj_over_nljj0);
            let alpha_wj_over_nljj1 = E::faer_simd_splat(simd, alpha_wj_over_nljj1);
            let alpha_wj_over_nljj2 = E::faer_simd_splat(simd, alpha_wj_over_nljj2);
            let alpha_wj_over_nljj3 = E::faer_simd_splat(simd, alpha_wj_over_nljj3);

            for (l, (w0, (w1, (w2, w3)))) in zip(
                E::faer_into_iter(l_head),
                zip(
                    E::faer_into_iter(w0_head),
                    zip(
                        E::faer_into_iter(w1_head),
                        zip(E::faer_into_iter(w2_head), E::faer_into_iter(w3_head)),
                    ),
                ),
            ) {
                let mut local_l = E::faer_deref(E::faer_rb(E::faer_as_ref(&l)));
                let mut local_w0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w0)));
                let mut local_w1 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w1)));
                let mut local_w2 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w2)));
                let mut local_w3 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w3)));

                local_w0 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&neg_wj_over_ljj0),
                    E::faer_copy(&local_l),
                    local_w0,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&alpha_wj_over_nljj0),
                    E::faer_copy(&local_w0),
                    E::faer_simd_mul(simd, E::faer_copy(&nljj_over_ljj0), local_l),
                );

                local_w1 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&neg_wj_over_ljj1),
                    E::faer_copy(&local_l),
                    local_w1,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&alpha_wj_over_nljj1),
                    E::faer_copy(&local_w1),
                    E::faer_simd_mul(simd, E::faer_copy(&nljj_over_ljj1), local_l),
                );

                local_w2 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&neg_wj_over_ljj2),
                    E::faer_copy(&local_l),
                    local_w2,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&alpha_wj_over_nljj2),
                    E::faer_copy(&local_w2),
                    E::faer_simd_mul(simd, E::faer_copy(&nljj_over_ljj2), local_l),
                );

                local_w3 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&neg_wj_over_ljj3),
                    E::faer_copy(&local_l),
                    local_w3,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&alpha_wj_over_nljj3),
                    E::faer_copy(&local_w3),
                    E::faer_simd_mul(simd, E::faer_copy(&nljj_over_ljj3), local_l),
                );

                E::faer_map(E::faer_zip(l, local_l), |(l, local_l)| *l = local_l);
                E::faer_map(E::faer_zip(w0, local_w0), |(w0, local_w0)| *w0 = local_w0);
                E::faer_map(E::faer_zip(w1, local_w1), |(w1, local_w1)| *w1 = local_w1);
                E::faer_map(E::faer_zip(w2, local_w2), |(w2, local_w2)| *w2 = local_w2);
                E::faer_map(E::faer_zip(w3, local_w3), |(w3, local_w3)| *w3 = local_w3);
            }
        }

        for (l, (w0, (w1, (w2, w3)))) in zip(
            E::faer_into_iter(l_tail),
            zip(
                E::faer_into_iter(w0_tail),
                zip(
                    E::faer_into_iter(w1_tail),
                    zip(E::faer_into_iter(w2_tail), E::faer_into_iter(w3_tail)),
                ),
            ),
        ) {
            let mut local_l = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&l))));
            let mut local_w0 = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&w0))));
            let mut local_w1 = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&w1))));
            let mut local_w2 = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&w2))));
            let mut local_w3 = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&w3))));

            local_w0 = local_w0.faer_add(E::faer_mul(neg_wj_over_ljj0, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj0, local_w0),
                E::faer_mul(nljj_over_ljj0, local_l),
            );

            local_w1 = local_w1.faer_add(E::faer_mul(neg_wj_over_ljj1, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj1, local_w1),
                E::faer_mul(nljj_over_ljj1, local_l),
            );

            local_w2 = local_w2.faer_add(E::faer_mul(neg_wj_over_ljj2, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj2, local_w2),
                E::faer_mul(nljj_over_ljj2, local_l),
            );

            local_w3 = local_w3.faer_add(E::faer_mul(neg_wj_over_ljj3, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj3, local_w3),
                E::faer_mul(nljj_over_ljj3, local_l),
            );

            E::faer_map(
                E::faer_zip(l, E::faer_into_units(local_l)),
                |(l, local_l)| *l = local_l,
            );
            E::faer_map(
                E::faer_zip(w0, E::faer_into_units(local_w0)),
                |(w0, local_w0)| *w0 = local_w0,
            );
            E::faer_map(
                E::faer_zip(w1, E::faer_into_units(local_w1)),
                |(w1, local_w1)| *w1 = local_w1,
            );
            E::faer_map(
                E::faer_zip(w2, E::faer_into_units(local_w2)),
                |(w2, local_w2)| *w2 = local_w2,
            );
            E::faer_map(
                E::faer_zip(w3, E::faer_into_units(local_w3)),
                |(w3, local_w3)| *w3 = local_w3,
            );
        }
    }
}

impl<'a, E: ComplexField> pulp::WithSimd for RankUpdateStepImpl<'a, E, 3> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            l_col,
            w,
            neg_wj_over_ljj_array,
            alpha_wj_over_nljj_array,
            nljj_over_ljj_array,
        } = self;

        let [w0, w1, w2] = w;
        let [neg_wj_over_ljj0, neg_wj_over_ljj1, neg_wj_over_ljj2] = neg_wj_over_ljj_array;
        let [nljj_over_ljj0, nljj_over_ljj1, nljj_over_ljj2] = nljj_over_ljj_array;
        let [alpha_wj_over_nljj0, alpha_wj_over_nljj1, alpha_wj_over_nljj2] =
            alpha_wj_over_nljj_array;

        let (l_head, l_tail) = slice_as_mut_simd::<E, S>(l_col);
        let (w0_head, w0_tail) = slice_as_mut_simd::<E, S>(w0);
        let (w1_head, w1_tail) = slice_as_mut_simd::<E, S>(w1);
        let (w2_head, w2_tail) = slice_as_mut_simd::<E, S>(w2);

        {
            let neg_wj_over_ljj0 = E::faer_simd_splat(simd, neg_wj_over_ljj0);
            let neg_wj_over_ljj1 = E::faer_simd_splat(simd, neg_wj_over_ljj1);
            let neg_wj_over_ljj2 = E::faer_simd_splat(simd, neg_wj_over_ljj2);
            let nljj_over_ljj0 = E::faer_simd_splat(simd, nljj_over_ljj0);
            let nljj_over_ljj1 = E::faer_simd_splat(simd, nljj_over_ljj1);
            let nljj_over_ljj2 = E::faer_simd_splat(simd, nljj_over_ljj2);
            let alpha_wj_over_nljj0 = E::faer_simd_splat(simd, alpha_wj_over_nljj0);
            let alpha_wj_over_nljj1 = E::faer_simd_splat(simd, alpha_wj_over_nljj1);
            let alpha_wj_over_nljj2 = E::faer_simd_splat(simd, alpha_wj_over_nljj2);

            for (l, (w0, (w1, w2))) in zip(
                E::faer_into_iter(l_head),
                zip(
                    E::faer_into_iter(w0_head),
                    zip(E::faer_into_iter(w1_head), E::faer_into_iter(w2_head)),
                ),
            ) {
                let mut local_l = E::faer_deref(E::faer_rb(E::faer_as_ref(&l)));
                let mut local_w0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w0)));
                let mut local_w1 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w1)));
                let mut local_w2 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w2)));

                local_w0 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&neg_wj_over_ljj0),
                    E::faer_copy(&local_l),
                    local_w0,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&alpha_wj_over_nljj0),
                    E::faer_copy(&local_w0),
                    E::faer_simd_mul(simd, E::faer_copy(&nljj_over_ljj0), local_l),
                );

                local_w1 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&neg_wj_over_ljj1),
                    E::faer_copy(&local_l),
                    local_w1,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&alpha_wj_over_nljj1),
                    E::faer_copy(&local_w1),
                    E::faer_simd_mul(simd, E::faer_copy(&nljj_over_ljj1), local_l),
                );

                local_w2 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&neg_wj_over_ljj2),
                    E::faer_copy(&local_l),
                    local_w2,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&alpha_wj_over_nljj2),
                    E::faer_copy(&local_w2),
                    E::faer_simd_mul(simd, E::faer_copy(&nljj_over_ljj2), local_l),
                );

                E::faer_map(E::faer_zip(l, local_l), |(l, local_l)| *l = local_l);
                E::faer_map(E::faer_zip(w0, local_w0), |(w0, local_w0)| *w0 = local_w0);
                E::faer_map(E::faer_zip(w1, local_w1), |(w1, local_w1)| *w1 = local_w1);
                E::faer_map(E::faer_zip(w2, local_w2), |(w2, local_w2)| *w2 = local_w2);
            }
        }

        for (l, (w0, (w1, w2))) in zip(
            E::faer_into_iter(l_tail),
            zip(
                E::faer_into_iter(w0_tail),
                zip(E::faer_into_iter(w1_tail), E::faer_into_iter(w2_tail)),
            ),
        ) {
            let mut local_l = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&l))));
            let mut local_w0 = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&w0))));
            let mut local_w1 = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&w1))));
            let mut local_w2 = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&w2))));

            local_w0 = local_w0.faer_add(E::faer_mul(neg_wj_over_ljj0, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj0, local_w0),
                E::faer_mul(nljj_over_ljj0, local_l),
            );

            local_w1 = local_w1.faer_add(E::faer_mul(neg_wj_over_ljj1, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj1, local_w1),
                E::faer_mul(nljj_over_ljj1, local_l),
            );

            local_w2 = local_w2.faer_add(E::faer_mul(neg_wj_over_ljj2, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj2, local_w2),
                E::faer_mul(nljj_over_ljj2, local_l),
            );

            E::faer_map(
                E::faer_zip(l, E::faer_into_units(local_l)),
                |(l, local_l)| *l = local_l,
            );
            E::faer_map(
                E::faer_zip(w0, E::faer_into_units(local_w0)),
                |(w0, local_w0)| *w0 = local_w0,
            );
            E::faer_map(
                E::faer_zip(w1, E::faer_into_units(local_w1)),
                |(w1, local_w1)| *w1 = local_w1,
            );
            E::faer_map(
                E::faer_zip(w2, E::faer_into_units(local_w2)),
                |(w2, local_w2)| *w2 = local_w2,
            );
        }
    }
}

impl<'a, E: ComplexField> pulp::WithSimd for RankUpdateStepImpl<'a, E, 2> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            l_col,
            w,
            neg_wj_over_ljj_array,
            alpha_wj_over_nljj_array,
            nljj_over_ljj_array,
        } = self;

        let [w0, w1] = w;
        let [neg_wj_over_ljj0, neg_wj_over_ljj1] = neg_wj_over_ljj_array;
        let [nljj_over_ljj0, nljj_over_ljj1] = nljj_over_ljj_array;
        let [alpha_wj_over_nljj0, alpha_wj_over_nljj1] = alpha_wj_over_nljj_array;

        let (l_head, l_tail) = slice_as_mut_simd::<E, S>(l_col);
        let (w0_head, w0_tail) = slice_as_mut_simd::<E, S>(w0);
        let (w1_head, w1_tail) = slice_as_mut_simd::<E, S>(w1);

        {
            let neg_wj_over_ljj0 = E::faer_simd_splat(simd, neg_wj_over_ljj0);
            let neg_wj_over_ljj1 = E::faer_simd_splat(simd, neg_wj_over_ljj1);
            let nljj_over_ljj0 = E::faer_simd_splat(simd, nljj_over_ljj0);
            let nljj_over_ljj1 = E::faer_simd_splat(simd, nljj_over_ljj1);
            let alpha_wj_over_nljj0 = E::faer_simd_splat(simd, alpha_wj_over_nljj0);
            let alpha_wj_over_nljj1 = E::faer_simd_splat(simd, alpha_wj_over_nljj1);

            for (l, (w0, w1)) in zip(
                E::faer_into_iter(l_head),
                zip(E::faer_into_iter(w0_head), E::faer_into_iter(w1_head)),
            ) {
                let mut local_l = E::faer_deref(E::faer_rb(E::faer_as_ref(&l)));
                let mut local_w0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w0)));
                let mut local_w1 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w1)));

                local_w0 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&neg_wj_over_ljj0),
                    E::faer_copy(&local_l),
                    local_w0,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&alpha_wj_over_nljj0),
                    E::faer_copy(&local_w0),
                    E::faer_simd_mul(simd, E::faer_copy(&nljj_over_ljj0), local_l),
                );

                local_w1 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&neg_wj_over_ljj1),
                    E::faer_copy(&local_l),
                    local_w1,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&alpha_wj_over_nljj1),
                    E::faer_copy(&local_w1),
                    E::faer_simd_mul(simd, E::faer_copy(&nljj_over_ljj1), local_l),
                );

                E::faer_map(E::faer_zip(l, local_l), |(l, local_l)| *l = local_l);
                E::faer_map(E::faer_zip(w0, local_w0), |(w0, local_w0)| *w0 = local_w0);
                E::faer_map(E::faer_zip(w1, local_w1), |(w1, local_w1)| *w1 = local_w1);
            }
        }

        for (l, (w0, w1)) in zip(
            E::faer_into_iter(l_tail),
            zip(E::faer_into_iter(w0_tail), E::faer_into_iter(w1_tail)),
        ) {
            let mut local_l = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&l))));
            let mut local_w0 = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&w0))));
            let mut local_w1 = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&w1))));

            local_w0 = local_w0.faer_add(E::faer_mul(neg_wj_over_ljj0, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj0, local_w0),
                E::faer_mul(nljj_over_ljj0, local_l),
            );

            local_w1 = local_w1.faer_add(E::faer_mul(neg_wj_over_ljj1, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj1, local_w1),
                E::faer_mul(nljj_over_ljj1, local_l),
            );

            E::faer_map(
                E::faer_zip(l, E::faer_into_units(local_l)),
                |(l, local_l)| *l = local_l,
            );
            E::faer_map(
                E::faer_zip(w0, E::faer_into_units(local_w0)),
                |(w0, local_w0)| *w0 = local_w0,
            );
            E::faer_map(
                E::faer_zip(w1, E::faer_into_units(local_w1)),
                |(w1, local_w1)| *w1 = local_w1,
            );
        }
    }
}

impl<'a, E: ComplexField> pulp::WithSimd for RankUpdateStepImpl<'a, E, 1> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            l_col,
            w,
            neg_wj_over_ljj_array,
            alpha_wj_over_nljj_array,
            nljj_over_ljj_array,
        } = self;

        let [w0] = w;
        let [neg_wj_over_ljj0] = neg_wj_over_ljj_array;
        let [nljj_over_ljj0] = nljj_over_ljj_array;
        let [alpha_wj_over_nljj0] = alpha_wj_over_nljj_array;

        let (l_head, l_tail) = slice_as_mut_simd::<E, S>(l_col);
        let (w0_head, w0_tail) = slice_as_mut_simd::<E, S>(w0);

        {
            let neg_wj_over_ljj0 = E::faer_simd_splat(simd, neg_wj_over_ljj0);
            let nljj_over_ljj0 = E::faer_simd_splat(simd, nljj_over_ljj0);
            let alpha_wj_over_nljj0 = E::faer_simd_splat(simd, alpha_wj_over_nljj0);

            for (l, w0) in zip(E::faer_into_iter(l_head), E::faer_into_iter(w0_head)) {
                let mut local_l = E::faer_deref(E::faer_rb(E::faer_as_ref(&l)));
                let mut local_w0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w0)));

                local_w0 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&neg_wj_over_ljj0),
                    E::faer_copy(&local_l),
                    local_w0,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&alpha_wj_over_nljj0),
                    E::faer_copy(&local_w0),
                    E::faer_simd_mul(simd, E::faer_copy(&nljj_over_ljj0), local_l),
                );

                E::faer_map(E::faer_zip(l, local_l), |(l, local_l)| *l = local_l);
                E::faer_map(E::faer_zip(w0, local_w0), |(w0, local_w0)| *w0 = local_w0);
            }
        }

        for (l, w0) in zip(E::faer_into_iter(l_tail), E::faer_into_iter(w0_tail)) {
            let mut local_l = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&l))));
            let mut local_w0 = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&w0))));

            local_w0 = local_w0.faer_add(E::faer_mul(neg_wj_over_ljj0, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj0, local_w0),
                E::faer_mul(nljj_over_ljj0, local_l),
            );

            E::faer_map(
                E::faer_zip(l, E::faer_into_units(local_l)),
                |(l, local_l)| *l = local_l,
            );
            E::faer_map(
                E::faer_zip(w0, E::faer_into_units(local_w0)),
                |(w0, local_w0)| *w0 = local_w0,
            );
        }
    }
}

fn rank_update_step_impl4<E: ComplexField>(
    arch: E::Simd,
    l_col: MatMut<'_, E>,
    w: MatMut<'_, E>,
    neg_wj_over_ljj_array: [E; 4],
    alpha_wj_over_nljj_array: [E; 4],
    nljj_over_ljj_array: [E; 4],
) {
    let m = l_col.nrows();
    let w = w.into_const();
    let w0 = unsafe { w.col(0).const_cast() };
    let w1 = unsafe { w.col(1).const_cast() };
    let w2 = unsafe { w.col(2).const_cast() };
    let w3 = unsafe { w.col(3).const_cast() };
    if l_col.row_stride() == 1 && w.row_stride() == 1 {
        arch.dispatch(RankUpdateStepImpl::<'_, E, 4> {
            l_col: unsafe { E::faer_map(l_col.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)) },
            w: unsafe {
                [
                    E::faer_map(w0.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)),
                    E::faer_map(w1.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)),
                    E::faer_map(w2.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)),
                    E::faer_map(w3.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)),
                ]
            },
            neg_wj_over_ljj_array,
            alpha_wj_over_nljj_array,
            nljj_over_ljj_array,
        });
    } else {
        let [neg_wj_over_ljj0, neg_wj_over_ljj1, neg_wj_over_ljj2, neg_wj_over_ljj3] =
            neg_wj_over_ljj_array;
        let [nljj_over_ljj0, nljj_over_ljj1, nljj_over_ljj2, nljj_over_ljj3] = nljj_over_ljj_array;
        let [alpha_wj_over_nljj0, alpha_wj_over_nljj1, alpha_wj_over_nljj2, alpha_wj_over_nljj3] =
            alpha_wj_over_nljj_array;

        zipped!(l_col, w0, w1, w2, w3).for_each(|mut l, mut w0, mut w1, mut w2, mut w3| {
            let mut local_l = l.read();
            let mut local_w0 = w0.read();
            let mut local_w1 = w1.read();
            let mut local_w2 = w2.read();
            let mut local_w3 = w3.read();

            local_w0 = local_w0.faer_add(E::faer_mul(neg_wj_over_ljj0, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj0, local_w0),
                E::faer_mul(nljj_over_ljj0, local_l),
            );
            local_w1 = local_w1.faer_add(E::faer_mul(neg_wj_over_ljj1, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj1, local_w1),
                E::faer_mul(nljj_over_ljj1, local_l),
            );
            local_w2 = local_w2.faer_add(E::faer_mul(neg_wj_over_ljj2, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj2, local_w2),
                E::faer_mul(nljj_over_ljj2, local_l),
            );
            local_w3 = local_w3.faer_add(E::faer_mul(neg_wj_over_ljj3, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj3, local_w3),
                E::faer_mul(nljj_over_ljj3, local_l),
            );

            l.write(local_l);
            w0.write(local_w0);
            w1.write(local_w1);
            w2.write(local_w2);
            w3.write(local_w3);
        });
    }
}

fn rank_update_step_impl3<E: ComplexField>(
    arch: E::Simd,
    l_col: MatMut<'_, E>,
    w: MatMut<'_, E>,
    neg_wj_over_ljj_array: [E; 4],
    alpha_wj_over_nljj_array: [E; 4],
    nljj_over_ljj_array: [E; 4],
) {
    let m = l_col.nrows();
    let w = w.into_const();
    let w0 = unsafe { w.col(0).const_cast() };
    let w1 = unsafe { w.col(1).const_cast() };
    let w2 = unsafe { w.col(2).const_cast() };

    let [neg_wj_over_ljj_array @ .., _] = neg_wj_over_ljj_array;
    let [alpha_wj_over_nljj_array @ .., _] = alpha_wj_over_nljj_array;
    let [nljj_over_ljj_array @ .., _] = nljj_over_ljj_array;

    if l_col.row_stride() == 1 && w.row_stride() == 1 {
        arch.dispatch(RankUpdateStepImpl::<'_, E, 3> {
            l_col: unsafe { E::faer_map(l_col.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)) },
            w: unsafe {
                [
                    E::faer_map(w0.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)),
                    E::faer_map(w1.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)),
                    E::faer_map(w2.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)),
                ]
            },
            neg_wj_over_ljj_array,
            alpha_wj_over_nljj_array,
            nljj_over_ljj_array,
        });
    } else {
        let [neg_wj_over_ljj0, neg_wj_over_ljj1, neg_wj_over_ljj2] = neg_wj_over_ljj_array;
        let [nljj_over_ljj0, nljj_over_ljj1, nljj_over_ljj2] = nljj_over_ljj_array;
        let [alpha_wj_over_nljj0, alpha_wj_over_nljj1, alpha_wj_over_nljj2] =
            alpha_wj_over_nljj_array;

        zipped!(l_col, w0, w1, w2).for_each(|mut l, mut w0, mut w1, mut w2| {
            let mut local_l = l.read();
            let mut local_w0 = w0.read();
            let mut local_w1 = w1.read();
            let mut local_w2 = w2.read();

            local_w0 = local_w0.faer_add(E::faer_mul(neg_wj_over_ljj0, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj0, local_w0),
                E::faer_mul(nljj_over_ljj0, local_l),
            );
            local_w1 = local_w1.faer_add(E::faer_mul(neg_wj_over_ljj1, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj1, local_w1),
                E::faer_mul(nljj_over_ljj1, local_l),
            );
            local_w2 = local_w2.faer_add(E::faer_mul(neg_wj_over_ljj2, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj2, local_w2),
                E::faer_mul(nljj_over_ljj2, local_l),
            );

            l.write(local_l);
            w0.write(local_w0);
            w1.write(local_w1);
            w2.write(local_w2);
        });
    }
}

fn rank_update_step_impl2<E: ComplexField>(
    arch: E::Simd,
    l_col: MatMut<'_, E>,
    w: MatMut<'_, E>,
    neg_wj_over_ljj_array: [E; 4],
    alpha_wj_over_nljj_array: [E; 4],
    nljj_over_ljj_array: [E; 4],
) {
    let m = l_col.nrows();
    let w = w.into_const();
    let w0 = unsafe { w.col(0).const_cast() };
    let w1 = unsafe { w.col(1).const_cast() };
    let [neg_wj_over_ljj_array @ .., _, _] = neg_wj_over_ljj_array;
    let [alpha_wj_over_nljj_array @ .., _, _] = alpha_wj_over_nljj_array;
    let [nljj_over_ljj_array @ .., _, _] = nljj_over_ljj_array;

    if l_col.row_stride() == 1 && w.row_stride() == 1 {
        arch.dispatch(RankUpdateStepImpl::<'_, E, 2> {
            l_col: unsafe { E::faer_map(l_col.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)) },
            w: unsafe {
                [
                    E::faer_map(w0.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)),
                    E::faer_map(w1.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)),
                ]
            },
            neg_wj_over_ljj_array,
            alpha_wj_over_nljj_array,
            nljj_over_ljj_array,
        });
    } else {
        let [neg_wj_over_ljj0, neg_wj_over_ljj1] = neg_wj_over_ljj_array;
        let [nljj_over_ljj0, nljj_over_ljj1] = nljj_over_ljj_array;
        let [alpha_wj_over_nljj0, alpha_wj_over_nljj1] = alpha_wj_over_nljj_array;

        zipped!(l_col, w0, w1).for_each(|mut l, mut w0, mut w1| {
            let mut local_l = l.read();
            let mut local_w0 = w0.read();
            let mut local_w1 = w1.read();

            local_w0 = local_w0.faer_add(E::faer_mul(neg_wj_over_ljj0, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj0, local_w0),
                E::faer_mul(nljj_over_ljj0, local_l),
            );
            local_w1 = local_w1.faer_add(E::faer_mul(neg_wj_over_ljj1, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj1, local_w1),
                E::faer_mul(nljj_over_ljj1, local_l),
            );

            l.write(local_l);
            w0.write(local_w0);
            w1.write(local_w1);
        });
    }
}

fn rank_update_step_impl1<E: ComplexField>(
    arch: E::Simd,
    l_col: MatMut<'_, E>,
    w: MatMut<'_, E>,
    neg_wj_over_ljj_array: [E; 4],
    alpha_wj_over_nljj_array: [E; 4],
    nljj_over_ljj_array: [E; 4],
) {
    let m = l_col.nrows();
    let w_rs = w.row_stride();

    let w0 = w.col(0);
    let [neg_wj_over_ljj_array @ .., _, _, _] = neg_wj_over_ljj_array;
    let [alpha_wj_over_nljj_array @ .., _, _, _] = alpha_wj_over_nljj_array;
    let [nljj_over_ljj_array @ .., _, _, _] = nljj_over_ljj_array;

    if l_col.row_stride() == 1 && w_rs == 1 {
        arch.dispatch(RankUpdateStepImpl::<'_, E, 1> {
            l_col: unsafe { E::faer_map(l_col.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)) },
            w: unsafe {
                [E::faer_map(w0.as_ptr(), |ptr| {
                    slice::from_raw_parts_mut(ptr, m)
                })]
            },
            neg_wj_over_ljj_array,
            alpha_wj_over_nljj_array,
            nljj_over_ljj_array,
        });
    } else {
        let [neg_wj_over_ljj0] = neg_wj_over_ljj_array;
        let [nljj_over_ljj0] = nljj_over_ljj_array;
        let [alpha_wj_over_nljj0] = alpha_wj_over_nljj_array;

        zipped!(l_col, w0).for_each(|mut l, mut w0| {
            let mut local_l = l.read();
            let mut local_w0 = w0.read();

            local_w0 = local_w0.faer_add(E::faer_mul(neg_wj_over_ljj0, local_l));
            local_l = E::faer_add(
                E::faer_mul(alpha_wj_over_nljj0, local_w0),
                E::faer_mul(nljj_over_ljj0, local_l),
            );

            l.write(local_l);
            w0.write(local_w0);
        });
    }
}

struct RankRUpdate<'a, E: Entity> {
    l: MatMut<'a, E>,
    w: MatMut<'a, E>,
    alpha: MatMut<'a, E>,
    r: &'a mut dyn FnMut() -> usize,
}

impl<'a, E: ComplexField> RankRUpdate<'a, E> {
    fn run(self) -> Result<(), CholeskyError> {
        // On the Modification of LDLT Factorizations
        // By R. Fletcher and M. J. D. Powell
        // https://www.ams.org/journals/mcom/1974-28-128/S0025-5718-1974-0359297-1/S0025-5718-1974-0359297-1.pdf

        let RankRUpdate {
            mut l,
            mut w,
            mut alpha,
            r,
        } = self;
        let n = l.nrows();
        let k = w.ncols();

        debug_assert!(l.ncols() == n);
        debug_assert!(w.nrows() == n);
        debug_assert!(alpha.nrows() == k);

        let arch = E::Simd::default();
        unsafe {
            for j in 0..n {
                let r = Ord::min((*r)(), k);

                let mut r_idx = 0;
                while r_idx < r {
                    let r_chunk = Ord::min(r - r_idx, 4);
                    let mut neg_wj_over_ljj_array = [
                        E::faer_zero(),
                        E::faer_zero(),
                        E::faer_zero(),
                        E::faer_zero(),
                    ];
                    let mut alpha_wj_over_nljj_array = [
                        E::faer_zero(),
                        E::faer_zero(),
                        E::faer_zero(),
                        E::faer_zero(),
                    ];
                    let mut nljj_over_ljj_array = [
                        E::faer_zero(),
                        E::faer_zero(),
                        E::faer_zero(),
                        E::faer_zero(),
                    ];

                    let mut ljj = l.read_unchecked(j, j);
                    for k in 0..r_chunk {
                        let neg_wj_over_ljj = neg_wj_over_ljj_array.get_unchecked_mut(k);
                        let alpha_conj_wj_over_nljj = alpha_wj_over_nljj_array.get_unchecked_mut(k);
                        let nljj_over_ljj = nljj_over_ljj_array.get_unchecked_mut(k);

                        let local_alpha = alpha.read_unchecked(r_idx + k, 0);
                        let wj = w.read_unchecked(j, r_idx + k);
                        let alpha_conj_wj = local_alpha.faer_mul(wj.faer_conj());

                        let sqr_nljj = ljj.faer_mul(ljj).faer_add(alpha_conj_wj.faer_mul(wj));
                        match PartialOrd::partial_cmp(&sqr_nljj.faer_real(), &E::Real::faer_zero())
                        {
                            Some(core::cmp::Ordering::Greater) => (),
                            _ => return Err(CholeskyError),
                        }
                        let nljj = E::faer_from_real(sqr_nljj.faer_real().faer_sqrt());
                        let inv_ljj = ljj.faer_inv();
                        let inv_nljj = nljj.faer_inv();

                        *neg_wj_over_ljj = (wj.faer_mul(inv_ljj)).faer_neg();
                        *nljj_over_ljj = nljj.faer_mul(inv_ljj);
                        *alpha_conj_wj_over_nljj = alpha_conj_wj.faer_mul(inv_nljj);
                        alpha.write_unchecked(
                            r_idx + k,
                            0,
                            local_alpha.faer_sub(
                                (*alpha_conj_wj_over_nljj)
                                    .faer_mul((*alpha_conj_wj_over_nljj).faer_conj()),
                            ),
                        );

                        ljj = nljj;
                    }
                    l.write_unchecked(j, j, ljj);

                    let rem = n - j - 1;

                    let l_col = l.rb_mut().col(j).subrows(j + 1, rem);
                    let w = w.rb_mut().subcols(r_idx, r_chunk).subrows(j + 1, rem);

                    match r_chunk {
                        1 => rank_update_step_impl1(
                            arch,
                            l_col,
                            w,
                            neg_wj_over_ljj_array,
                            alpha_wj_over_nljj_array,
                            nljj_over_ljj_array,
                        ),
                        2 => rank_update_step_impl2(
                            arch,
                            l_col,
                            w,
                            neg_wj_over_ljj_array,
                            alpha_wj_over_nljj_array,
                            nljj_over_ljj_array,
                        ),
                        3 => rank_update_step_impl3(
                            arch,
                            l_col,
                            w,
                            neg_wj_over_ljj_array,
                            alpha_wj_over_nljj_array,
                            nljj_over_ljj_array,
                        ),
                        4 => rank_update_step_impl4(
                            arch,
                            l_col,
                            w,
                            neg_wj_over_ljj_array,
                            alpha_wj_over_nljj_array,
                            nljj_over_ljj_array,
                        ),
                        _ => unreachable!(),
                    }

                    r_idx += r_chunk;
                }
            }
        }
        Ok(())
    }
}

/// Performs a rank-r update in place, while clobbering the inputs.
///
/// Takes the Cholesky factor $L$ of a matrix $A$, i.e., $LL^H = A$, a matrix $W$ and a column
/// vector $\alpha$, which is interpreted as a diagonal matrix.
///
/// This function computes the Cholesky factor of $A + W\text{Diag}(\alpha)W^H$, and stores the
/// result in the storage of the original cholesky factors.
///
/// The matrix $W$ and the vector $\alpha$ are clobbered, meaning that the values they contain after
/// the function is called are unspecified.
///
/// # Panics
///
/// Panics if any of these conditions is violated:
/// * `cholesky_factor` must be square of dimension `n`.
/// * `w` must have `n` rows.
/// * `alpha` must have one column.
/// * `alpha` must have the same number of rows as the number of columns in `w`.
#[track_caller]
pub fn rank_r_update_clobber<E: ComplexField>(
    cholesky_factor: MatMut<'_, E>,
    w: MatMut<'_, E>,
    alpha: MatMut<'_, E>,
) -> Result<(), CholeskyError> {
    let n = cholesky_factor.nrows();
    let k = w.ncols();

    assert!(cholesky_factor.ncols() == n);
    assert!(w.nrows() == n);
    assert!(alpha.nrows() == k);
    assert!(alpha.ncols() == 1);

    if n == 0 {
        return Ok(());
    }

    RankRUpdate {
        l: cholesky_factor,
        w,
        alpha,
        r: &mut || k,
    }
    .run()
}

/// Computes the size and alignment of required workspace for deleting the rows and columns from a
/// matrix, given its Cholesky decomposition.
#[track_caller]
pub fn delete_rows_and_cols_clobber_req<E: Entity>(
    dim: usize,
    number_of_rows_to_remove: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    let r = number_of_rows_to_remove;
    StackReq::try_all_of([temp_mat_req::<E>(dim, r)?, temp_mat_req::<E>(r, 1)?])
}

/// Deletes `r` rows and columns at the provided indices from the Cholesky factor.
///
/// Takes the Cholesky factor $L$ of a matrix $A$ of dimension `n`, and `r` indices, then computes
/// the Cholesky factor of $A$ with the provided rows and columns deleted from it.
///
/// The result is stored in the top left corner (with dimension `n - r`) of `cholesky_factor`.
///
/// The indices are clobbered, meaning that the values in the slice after the function
/// is called are unspecified.
///
/// # Panics
///
/// Panics if the matrix is not square, the indices are out of bounds, or the list of indices
/// contains duplicates.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`delete_rows_and_cols_clobber_req`]).
#[track_caller]
pub fn delete_rows_and_cols_clobber<E: ComplexField>(
    cholesky_factor: MatMut<'_, E>,
    indices: &mut [usize],
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let _ = parallelism;
    let n = cholesky_factor.nrows();
    let r = indices.len();
    assert!(cholesky_factor.ncols() == n);
    assert!(indices.len() < n);

    if r == 0 {
        return;
    }

    indices.sort_unstable();
    for i in 0..r - 1 {
        assert!(indices[i + 1] > indices[i]);
    }
    assert!(indices[r - 1] < n);

    let first = indices[0];

    let (mut w, stack) = temp_mat_uninit::<E>(n - first - r, r, stack);
    let mut w = w.as_mut();
    let (mut alpha, _) = temp_mat_uninit::<E>(r, 1, stack);
    let alpha = alpha.as_mut();
    let mut alpha = alpha.col(0);

    E::Simd::default().dispatch(|| {
        for k in 0..r {
            let j = indices[k];
            unsafe {
                alpha.write_unchecked(k, 0, E::faer_one());
            }

            for chunk_i in k..r {
                let chunk_i = chunk_i + 1;
                let i_start = indices[chunk_i - 1] + 1;
                #[rustfmt::skip]
                let i_finish = if chunk_i == r { n } else { indices[chunk_i] };

                for i in i_start..i_finish {
                    unsafe {
                        w.write_unchecked(
                            i - chunk_i - first,
                            k,
                            cholesky_factor.read_unchecked(i, j),
                        );
                    }
                }
            }
        }
    });
    let mut cholesky_factor = cholesky_factor;
    delete_rows_and_cols_triangular(cholesky_factor.rb_mut(), indices);

    RankRUpdate {
        l: cholesky_factor.submatrix(first, first, n - first - r, n - first - r),
        w,
        alpha,
        r: &mut rank_update_indices(first, indices),
    }
    .run()
    .unwrap();
}

/// Computes the size and alignment of the required workspace for inserting the rows and columns at
/// the index in the Cholesky factor..
pub fn insert_rows_and_cols_clobber_req<E: Entity>(
    inserted_matrix_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    cholesky_in_place_req::<E>(inserted_matrix_ncols, parallelism, Default::default())
}

/// Inserts `r` rows and columns at the provided index in the Cholesky factor.
///
/// Takes a matrix, `cholesky_factor_extended`, of dimension `n + r`, containing the Cholesky factor
/// $L$ of a matrix $A$ in its top left corner, of dimension `n`, and computes the Cholesky factor
/// of $A$ with the provided `inserted_matrix` inserted at the position starting at
/// `insertion_index`.
///
/// The inserted matrix is clobbered, meaning that the values it contains after the function
/// is called are unspecified.
///
/// # Panics
///
/// Panics if the index is out of bounds or the matrix dimensions are invalid.
///
/// This can also panic if the provided memory in `stack` is insufficient (see
/// [`insert_rows_and_cols_clobber_req`]).
#[track_caller]
pub fn insert_rows_and_cols_clobber<E: ComplexField>(
    cholesky_factor_extended: MatMut<'_, E>,
    insertion_index: usize,
    inserted_matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) -> Result<(), CholeskyError> {
    let new_n = cholesky_factor_extended.nrows();
    let r = inserted_matrix.ncols();

    assert!(cholesky_factor_extended.nrows() == cholesky_factor_extended.ncols());
    assert!(cholesky_factor_extended.ncols() == new_n);
    assert!(r < new_n);
    let old_n = new_n - r;

    assert!(insertion_index <= old_n);

    if r == 0 {
        return Ok(());
    }

    let mut current_col = old_n;

    let mut ld = cholesky_factor_extended;

    while current_col != insertion_index {
        current_col -= 1;

        unsafe {
            for i in (current_col..old_n).rev() {
                ld.write_unchecked(i + r, current_col + r, ld.read_unchecked(i, current_col));
            }
        }
    }

    while current_col != 0 {
        current_col -= 1;
        unsafe {
            for i in (insertion_index..old_n).rev() {
                ld.write_unchecked(i + r, current_col, ld.read_unchecked(i, current_col));
            }
        }
    }

    let [l00, _, l_bot_left, ld_bot_right] = ld.split_at(insertion_index, insertion_index);
    let l00 = l00.into_const();

    let [_, mut l10, _, l20] = l_bot_left.split_at(r, 0);
    let [mut l11, _, mut l21, ld22] = ld_bot_right.split_at(r, r);

    let [_, mut a01, _, a_bottom] = inserted_matrix.split_at(insertion_index, 0);
    let [_, a11, _, a21] = a_bottom.split_at(r, 0);

    let mut stack = stack;

    solve::solve_lower_triangular_in_place(l00, a01.rb_mut(), parallelism);

    let a10 = a01.rb().transpose();

    for j in 0..insertion_index {
        for i in 0..r {
            unsafe {
                l10.write_unchecked(i, j, a10.read_unchecked(i, j).faer_conj());
            }
        }
    }

    for j in 0..r {
        for i in j..r {
            unsafe {
                l11.write_unchecked(i, j, a11.read_unchecked(i, j));
            }
        }
    }

    mul::triangular::matmul(
        l11.rb_mut(),
        BlockStructure::TriangularLower,
        l10.rb(),
        BlockStructure::Rectangular,
        a01.rb(),
        BlockStructure::Rectangular,
        Some(E::faer_one()),
        E::faer_one().faer_neg(),
        parallelism,
    );

    cholesky_in_place(
        l11.rb_mut(),
        Default::default(),
        parallelism,
        stack.rb_mut(),
        Default::default(),
    )?;
    let l11 = l11.into_const();

    let rem = l21.nrows();

    for j in 0..r {
        for i in 0..rem {
            unsafe {
                l21.write_unchecked(i, j, a21.read_unchecked(i, j));
            }
        }
    }

    mul::matmul(
        l21.rb_mut(),
        l20.rb(),
        a01.rb(),
        Some(E::faer_one()),
        E::faer_one().faer_neg(),
        parallelism,
    );

    solve::solve_lower_triangular_in_place(l11.conjugate(), l21.rb_mut().transpose(), parallelism);

    let mut alpha = a11.col(0);
    let mut w = a21;

    for j in 0..r {
        unsafe {
            alpha.write_unchecked(j, 0, E::faer_one().faer_neg());

            for i in 0..rem {
                w.write_unchecked(i, j, l21.read_unchecked(i, j));
            }
        }
    }

    rank_r_update_clobber(ld22, w, alpha)
}
