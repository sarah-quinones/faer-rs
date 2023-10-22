use crate::ldlt_diagonal::compute::{raw_cholesky_in_place, raw_cholesky_in_place_req};
#[cfg(feature = "std")]
use assert2::{assert, debug_assert};
use core::{iter::zip, slice};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{
    mul, mul::triangular::BlockStructure, simd::slice_as_mut_simd, solve, temp_mat_req,
    temp_mat_uninit, zipped, ComplexField, Entity, MatMut, Parallelism, SimdCtx,
};
use faer_entity::*;
use reborrow::*;

struct RankRUpdate<'a, E: Entity> {
    ld: MatMut<'a, E>,
    w: MatMut<'a, E>,
    alpha: MatMut<'a, E>,
    r: &'a mut dyn FnMut() -> usize,
}

struct RankUpdateStepImpl<'a, E: Entity, const R: usize> {
    l_col: GroupFor<E, &'a mut [E::Unit]>,
    w: [GroupFor<E, &'a mut [E::Unit]>; R],
    p_array: [E; R],
    beta_array: [E; R],
}

impl<'a, E: ComplexField> pulp::WithSimd for RankUpdateStepImpl<'a, E, 4> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            l_col,
            w,
            p_array,
            beta_array,
        } = self;

        let [p0, p1, p2, p3] = p_array;
        let [beta0, beta1, beta2, beta3] = beta_array;
        let [w0, w1, w2, w3] = w;

        let (l_head, l_tail) = slice_as_mut_simd::<E, S>(l_col);
        let (w0_head, w0_tail) = slice_as_mut_simd::<E, S>(w0);
        let (w1_head, w1_tail) = slice_as_mut_simd::<E, S>(w1);
        let (w2_head, w2_tail) = slice_as_mut_simd::<E, S>(w2);
        let (w3_head, w3_tail) = slice_as_mut_simd::<E, S>(w3);

        {
            let p0 = E::faer_simd_splat(simd, p0);
            let p1 = E::faer_simd_splat(simd, p1);
            let p2 = E::faer_simd_splat(simd, p2);
            let p3 = E::faer_simd_splat(simd, p3);
            let beta0 = E::faer_simd_splat(simd, beta0);
            let beta1 = E::faer_simd_splat(simd, beta1);
            let beta2 = E::faer_simd_splat(simd, beta2);
            let beta3 = E::faer_simd_splat(simd, beta3);

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
                    E::faer_copy(&p0),
                    E::faer_copy(&local_l),
                    local_w0,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&beta0),
                    E::faer_copy(&local_w0),
                    local_l,
                );

                local_w1 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&p1),
                    E::faer_copy(&local_l),
                    local_w1,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&beta1),
                    E::faer_copy(&local_w1),
                    local_l,
                );

                local_w2 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&p2),
                    E::faer_copy(&local_l),
                    local_w2,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&beta2),
                    E::faer_copy(&local_w2),
                    local_l,
                );

                local_w3 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&p3),
                    E::faer_copy(&local_l),
                    local_w3,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&beta3),
                    E::faer_copy(&local_w3),
                    local_l,
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

            local_w0 = local_w0.faer_add(E::faer_mul(p0, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta0, local_w0));

            local_w1 = local_w1.faer_add(E::faer_mul(p1, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta1, local_w1));

            local_w2 = local_w2.faer_add(E::faer_mul(p2, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta2, local_w2));

            local_w3 = local_w3.faer_add(E::faer_mul(p3, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta3, local_w3));

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
            p_array,
            beta_array,
        } = self;

        let [p0, p1, p2] = p_array;
        let [beta0, beta1, beta2] = beta_array;
        let [w0, w1, w2] = w;

        let (l_head, l_tail) = slice_as_mut_simd::<E, S>(l_col);
        let (w0_head, w0_tail) = slice_as_mut_simd::<E, S>(w0);
        let (w1_head, w1_tail) = slice_as_mut_simd::<E, S>(w1);
        let (w2_head, w2_tail) = slice_as_mut_simd::<E, S>(w2);

        {
            let p0 = E::faer_simd_splat(simd, p0);
            let p1 = E::faer_simd_splat(simd, p1);
            let p2 = E::faer_simd_splat(simd, p2);
            let beta0 = E::faer_simd_splat(simd, beta0);
            let beta1 = E::faer_simd_splat(simd, beta1);
            let beta2 = E::faer_simd_splat(simd, beta2);

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
                    E::faer_copy(&p0),
                    E::faer_copy(&local_l),
                    local_w0,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&beta0),
                    E::faer_copy(&local_w0),
                    local_l,
                );

                local_w1 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&p1),
                    E::faer_copy(&local_l),
                    local_w1,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&beta1),
                    E::faer_copy(&local_w1),
                    local_l,
                );

                local_w2 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&p2),
                    E::faer_copy(&local_l),
                    local_w2,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&beta2),
                    E::faer_copy(&local_w2),
                    local_l,
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

            local_w0 = local_w0.faer_add(E::faer_mul(p0, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta0, local_w0));

            local_w1 = local_w1.faer_add(E::faer_mul(p1, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta1, local_w1));

            local_w2 = local_w2.faer_add(E::faer_mul(p2, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta2, local_w2));

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
            p_array,
            beta_array,
        } = self;

        let [p0, p1] = p_array;
        let [beta0, beta1] = beta_array;
        let [w0, w1] = w;

        let (l_head, l_tail) = slice_as_mut_simd::<E, S>(l_col);
        let (w0_head, w0_tail) = slice_as_mut_simd::<E, S>(w0);
        let (w1_head, w1_tail) = slice_as_mut_simd::<E, S>(w1);

        {
            let p0 = E::faer_simd_splat(simd, p0);
            let p1 = E::faer_simd_splat(simd, p1);
            let beta0 = E::faer_simd_splat(simd, beta0);
            let beta1 = E::faer_simd_splat(simd, beta1);

            for (l, (w0, w1)) in zip(
                E::faer_into_iter(l_head),
                zip(E::faer_into_iter(w0_head), E::faer_into_iter(w1_head)),
            ) {
                let mut local_l = E::faer_deref(E::faer_rb(E::faer_as_ref(&l)));
                let mut local_w0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w0)));
                let mut local_w1 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w1)));

                local_w0 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&p0),
                    E::faer_copy(&local_l),
                    local_w0,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&beta0),
                    E::faer_copy(&local_w0),
                    local_l,
                );

                local_w1 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&p1),
                    E::faer_copy(&local_l),
                    local_w1,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&beta1),
                    E::faer_copy(&local_w1),
                    local_l,
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

            local_w0 = local_w0.faer_add(E::faer_mul(p0, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta0, local_w0));

            local_w1 = local_w1.faer_add(E::faer_mul(p1, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta1, local_w1));

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
            p_array,
            beta_array,
        } = self;

        let [p0] = p_array;
        let [beta0] = beta_array;
        let [w0] = w;

        let (l_head, l_tail) = slice_as_mut_simd::<E, S>(l_col);
        let (w0_head, w0_tail) = slice_as_mut_simd::<E, S>(w0);

        {
            let p0 = E::faer_simd_splat(simd, p0);
            let beta0 = E::faer_simd_splat(simd, beta0);

            for (l, w0) in zip(E::faer_into_iter(l_head), E::faer_into_iter(w0_head)) {
                let mut local_l = E::faer_deref(E::faer_rb(E::faer_as_ref(&l)));
                let mut local_w0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&w0)));

                local_w0 = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&p0),
                    E::faer_copy(&local_l),
                    local_w0,
                );
                local_l = E::faer_simd_mul_adde(
                    simd,
                    E::faer_copy(&beta0),
                    E::faer_copy(&local_w0),
                    local_l,
                );

                E::faer_map(E::faer_zip(l, local_l), |(l, local_l)| *l = local_l);
                E::faer_map(E::faer_zip(w0, local_w0), |(w0, local_w0)| *w0 = local_w0);
            }
        }

        for (l, w0) in zip(E::faer_into_iter(l_tail), E::faer_into_iter(w0_tail)) {
            let mut local_l = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&l))));
            let mut local_w0 = E::faer_from_units(E::faer_deref(E::faer_rb(E::faer_as_ref(&w0))));

            local_w0 = local_w0.faer_add(E::faer_mul(p0, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta0, local_w0));

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
    p_array: [E; 4],
    beta_array: [E; 4],
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
            p_array,
            beta_array,
        });
    } else {
        let [p0, p1, p2, p3] = p_array;
        let [beta0, beta1, beta2, beta3] = beta_array;

        zipped!(l_col, w0, w1, w2, w3).for_each(|mut l, mut w0, mut w1, mut w2, mut w3| {
            let mut local_l = l.read();
            let mut local_w0 = w0.read();
            let mut local_w1 = w1.read();
            let mut local_w2 = w2.read();
            let mut local_w3 = w3.read();

            local_w0 = local_w0.faer_add(E::faer_mul(p0, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta0, local_w0));

            local_w1 = local_w1.faer_add(E::faer_mul(p1, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta1, local_w1));

            local_w2 = local_w2.faer_add(E::faer_mul(p2, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta2, local_w2));

            local_w3 = local_w3.faer_add(E::faer_mul(p3, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta3, local_w3));

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
    p_array: [E; 4],
    beta_array: [E; 4],
) {
    let m = l_col.nrows();
    let w = w.into_const();
    let w0 = unsafe { w.col(0).const_cast() };
    let w1 = unsafe { w.col(1).const_cast() };
    let w2 = unsafe { w.col(2).const_cast() };

    let [p_array @ .., _] = p_array;
    let [beta_array @ .., _] = beta_array;

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
            p_array,
            beta_array,
        });
    } else {
        let [p0, p1, p2] = p_array;
        let [beta0, beta1, beta2] = beta_array;

        zipped!(l_col, w0, w1, w2).for_each(|mut l, mut w0, mut w1, mut w2| {
            let mut local_l = l.read();
            let mut local_w0 = w0.read();
            let mut local_w1 = w1.read();
            let mut local_w2 = w2.read();

            local_w0 = local_w0.faer_add(E::faer_mul(p0, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta0, local_w0));

            local_w1 = local_w1.faer_add(E::faer_mul(p1, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta1, local_w1));

            local_w2 = local_w2.faer_add(E::faer_mul(p2, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta2, local_w2));

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
    p_array: [E; 4],
    beta_array: [E; 4],
) {
    let m = l_col.nrows();
    let w = w.into_const();
    let w0 = unsafe { w.col(0).const_cast() };
    let w1 = unsafe { w.col(1).const_cast() };
    let [p_array @ .., _, _] = p_array;
    let [beta_array @ .., _, _] = beta_array;

    if l_col.row_stride() == 1 && w.row_stride() == 1 {
        arch.dispatch(RankUpdateStepImpl::<'_, E, 2> {
            l_col: unsafe { E::faer_map(l_col.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)) },
            w: unsafe {
                [
                    E::faer_map(w0.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)),
                    E::faer_map(w1.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)),
                ]
            },
            p_array,
            beta_array,
        });
    } else {
        let [p0, p1] = p_array;
        let [beta0, beta1] = beta_array;

        zipped!(l_col, w0, w1).for_each(|mut l, mut w0, mut w1| {
            let mut local_l = l.read();
            let mut local_w0 = w0.read();
            let mut local_w1 = w1.read();

            local_w0 = local_w0.faer_add(E::faer_mul(p0, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta0, local_w0));

            local_w1 = local_w1.faer_add(E::faer_mul(p1, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta1, local_w1));

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
    p_array: [E; 4],
    beta_array: [E; 4],
) {
    let m = l_col.nrows();
    let w_rs = w.row_stride();

    let w0 = w.col(0);
    let [p_array @ .., _, _, _] = p_array;
    let [beta_array @ .., _, _, _] = beta_array;

    if l_col.row_stride() == 1 && w_rs == 1 {
        arch.dispatch(RankUpdateStepImpl::<'_, E, 1> {
            l_col: unsafe { E::faer_map(l_col.as_ptr(), |ptr| slice::from_raw_parts_mut(ptr, m)) },
            w: unsafe {
                [E::faer_map(w0.as_ptr(), |ptr| {
                    slice::from_raw_parts_mut(ptr, m)
                })]
            },
            p_array,
            beta_array,
        });
    } else {
        let [p0] = p_array;
        let [beta0] = beta_array;

        zipped!(l_col, w0).for_each(|mut l, mut w0| {
            let mut local_l = l.read();
            let mut local_w0 = w0.read();

            local_w0 = local_w0.faer_add(E::faer_mul(p0, local_l));
            local_l = local_l.faer_add(E::faer_mul(beta0, local_w0));

            l.write(local_l);
            w0.write(local_w0);
        });
    }
}

impl<'a, E: ComplexField> RankRUpdate<'a, E> {
    fn run(self) {
        // On the Modification of LDLT Factorizations
        // By R. Fletcher and M. J. D. Powell
        // https://www.ams.org/journals/mcom/1974-28-128/S0025-5718-1974-0359297-1/S0025-5718-1974-0359297-1.pdf

        let RankRUpdate {
            mut ld,
            mut w,
            mut alpha,
            r,
        } = self;
        let n = ld.nrows();
        let k = w.ncols();

        debug_assert!(ld.ncols() == n);
        debug_assert!(w.nrows() == n);
        debug_assert!(alpha.nrows() == k);

        let arch = E::Simd::default();
        for j in 0..n {
            let r = Ord::min((*r)(), k);

            let mut r_idx = 0;
            while r_idx < r {
                let r_chunk = Ord::min(r - r_idx, 4);
                let mut p_array = [
                    E::faer_zero(),
                    E::faer_zero(),
                    E::faer_zero(),
                    E::faer_zero(),
                ];
                let mut beta_array = [
                    E::faer_zero(),
                    E::faer_zero(),
                    E::faer_zero(),
                    E::faer_zero(),
                ];

                let mut dj = ld.rb().read(j, j).faer_inv();
                for k in 0..r_chunk {
                    let p = &mut p_array[k];
                    let beta = &mut beta_array[k];
                    let local_alpha = alpha.read(r_idx + k, 0);

                    *p = w.rb().read(j, r_idx + k);
                    let alpha_conj_p = local_alpha.faer_mul(p.faer_conj());
                    let new_dj = dj.faer_add(alpha_conj_p.faer_mul(*p));
                    *beta = alpha_conj_p.faer_mul(new_dj.faer_inv());
                    alpha.write(
                        r_idx + k,
                        0,
                        local_alpha.faer_sub(new_dj.faer_mul(beta.faer_mul(beta.faer_conj()))),
                    );

                    dj = new_dj;
                    *p = (*p).faer_neg();
                }
                ld.write(j, j, dj.faer_inv());

                let rem = n - j - 1;

                let l_col = ld.rb_mut().col(j).subrows(j + 1, rem);
                let w = w.rb_mut().subcols(r_idx, r_chunk).subrows(j + 1, rem);

                match r_chunk {
                    1 => rank_update_step_impl1(arch, l_col, w, p_array, beta_array),
                    2 => rank_update_step_impl2(arch, l_col, w, p_array, beta_array),
                    3 => rank_update_step_impl3(arch, l_col, w, p_array, beta_array),
                    4 => rank_update_step_impl4(arch, l_col, w, p_array, beta_array),
                    _ => unreachable!(),
                }

                r_idx += r_chunk;
            }
        }
    }
}

/// Performs a rank-r update in place, while clobbering the inputs.
///
/// Takes the Cholesky factors $L$ and $D$ of a matrix $A$, meaning that $LDL^H = A$,
/// a matrix $W$ and a column vector $\alpha$, which is interpreted as a diagonal matrix.
///
/// This function computes the cholesky factors of $A + W\text{Diag}(\alpha)W^H$, and stores the
/// result in the storage of the original cholesky factors.
///
/// The matrix $W$ and the vector $\alpha$ are clobbered, meaning that the values they contain after
/// the function is called are unspecified.
///
/// # Panics
///
/// Panics if any of these conditions is violated:
/// * `cholesky_factors` must be square of dimension `n`.
/// * `w` must have `n` rows.
/// * `alpha` must have one column.
/// * `alpha` must have the same number of rows as the number of columns in `w`.
#[track_caller]
pub fn rank_r_update_clobber<E: ComplexField>(
    cholesky_factors: MatMut<'_, E>,
    w: MatMut<'_, E>,
    alpha: MatMut<'_, E>,
) {
    let n = cholesky_factors.nrows();
    let k = w.ncols();

    assert!(cholesky_factors.ncols() == n);
    assert!(w.nrows() == n);
    assert!(alpha.nrows() == k);

    if n == 0 {
        return;
    }

    RankRUpdate {
        ld: cholesky_factors,
        w,
        alpha,
        r: &mut || k,
    }
    .run();
}

pub(crate) fn delete_rows_and_cols_triangular<E: ComplexField>(mat: MatMut<'_, E>, idx: &[usize]) {
    let mut mat = mat;
    let n = mat.nrows();
    let r = idx.len();
    debug_assert!(mat.ncols() == n);
    debug_assert!(r <= n);

    E::Simd::default().dispatch(|| {
        (0..=r).for_each(|chunk_j| {
            #[rustfmt::skip]
            let j_start = if chunk_j == 0 { 0 } else { idx[chunk_j - 1] + 1 };
            let j_finish = if chunk_j == r { n } else { idx[chunk_j] };

            for j in j_start..j_finish {
                (chunk_j..=r).for_each(|chunk_i| {
                    #[rustfmt::skip]
                    let i_start = if chunk_i == chunk_j { j } else { idx[chunk_i - 1] + 1 };
                    let i_finish = if chunk_i == r { n } else { idx[chunk_i] };

                    if chunk_i != 0 || chunk_j != 0 {
                        for i in i_start..i_finish {
                            unsafe {
                                let value = mat.rb().read_unchecked(i, j);
                                mat.rb_mut()
                                    .write_unchecked(i - chunk_i, j - chunk_j, value)
                            }
                        }
                    }
                })
            }
        });
    });
}

pub(crate) fn rank_update_indices(
    start_col: usize,
    indices: &[usize],
) -> impl FnMut() -> usize + '_ {
    let mut current_col = start_col;
    let mut current_r = 0;
    move || {
        if current_r == indices.len() {
            current_r
        } else {
            while current_col == indices[current_r] - current_r {
                current_r += 1;
                if current_r == indices.len() {
                    return current_r;
                }
            }

            current_col += 1;
            current_r
        }
    }
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
/// Takes the Cholesky factors $L$ and $D$ of a matrix $A$ of dimension `n`, and `r` indices, then
/// computes the Cholesky factor of $A$ with the provided rows and columns deleted from it.
///
/// The result is stored in the top left corner (with dimension `n - r`) of `cholesky_factor`.
///
/// The indices are clobbered, meaning that the values that the slice contains after the function
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
    cholesky_factors: MatMut<'_, E>,
    indices: &mut [usize],
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let _ = parallelism;
    let n = cholesky_factors.nrows();
    let r = indices.len();
    assert!(cholesky_factors.ncols() == n);
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
    let (mut alpha, _) = temp_mat_uninit::<E>(r, 1, stack);
    let mut w = w.as_mut();
    let alpha = alpha.as_mut();
    let mut alpha = alpha.col(0);

    E::Simd::default().dispatch(|| {
        for k in 0..r {
            let j = indices[k];
            unsafe {
                alpha.write_unchecked(k, 0, cholesky_factors.read_unchecked(j, j).faer_inv());
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
                            cholesky_factors.read_unchecked(i, j),
                        );
                    }
                }
            }
        }
    });
    let mut cholesky_factors = cholesky_factors;
    delete_rows_and_cols_triangular(cholesky_factors.rb_mut(), indices);

    RankRUpdate {
        ld: cholesky_factors.submatrix(first, first, n - first - r, n - first - r),
        w,
        alpha,
        r: &mut rank_update_indices(first, indices),
    }
    .run();
}

/// Computes the size and alignment of the required workspace for inserting the rows and columns at
/// the index in the Cholesky factor..
pub fn insert_rows_and_cols_clobber_req<E: Entity>(
    inserted_matrix_ncols: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    raw_cholesky_in_place_req::<E>(inserted_matrix_ncols, parallelism, Default::default())
}

/// Inserts `r` rows and columns at the provided index in the Cholesky factor.
///
/// Takes a matrix, `cholesky_factor_extended`, of dimension `n + r`, containing the Cholesky
/// factors $L$ and $D$ of a matrix $A$ in its top left corner, of dimension `n`, and computes the
/// Cholesky factor of $A$ with the provided `inserted_matrix` inserted at the position starting at
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
    cholesky_factors_extended: MatMut<'_, E>,
    insertion_index: usize,
    inserted_matrix: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    let new_n = cholesky_factors_extended.nrows();
    let r = inserted_matrix.ncols();

    assert!(cholesky_factors_extended.ncols() == new_n);
    assert!(r <= new_n);
    let old_n = new_n - r;

    assert!(insertion_index <= old_n);

    if r == 0 {
        return;
    }

    let mut current_col = old_n;

    let mut ld = cholesky_factors_extended;

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

    let [ld00, _, l_bot_left, ld_bot_right] = ld.split_at(insertion_index, insertion_index);
    let ld00 = ld00.into_const();
    let d0 = ld00.diagonal().into_column_vector();

    let [_, mut l10, _, l20] = l_bot_left.split_at(r, 0);
    let [mut ld11, _, mut l21, ld22] = ld_bot_right.split_at(r, r);

    let [_, mut a01, _, a_bottom] = inserted_matrix.split_at(insertion_index, 0);
    let [_, a11, _, a21] = a_bottom.split_at(r, 0);

    let mut stack = stack;

    solve::solve_unit_lower_triangular_in_place(ld00, a01.rb_mut(), parallelism);

    let a01 = a01.rb();

    for j in 0..insertion_index {
        let d0_inv = unsafe { d0.read_unchecked(j, 0) };
        for i in 0..r {
            unsafe {
                l10.write_unchecked(i, j, a01.read_unchecked(j, i).faer_conj().faer_mul(d0_inv));
            }
        }
    }

    for j in 0..r {
        for i in j..r {
            unsafe {
                ld11.write_unchecked(i, j, a11.read_unchecked(i, j));
            }
        }
    }

    mul::triangular::matmul(
        ld11.rb_mut(),
        BlockStructure::TriangularLower,
        l10.rb(),
        BlockStructure::Rectangular,
        a01,
        BlockStructure::Rectangular,
        Some(E::faer_one()),
        E::faer_one().faer_neg(),
        parallelism,
    );

    raw_cholesky_in_place(
        ld11.rb_mut(),
        Default::default(),
        parallelism,
        stack.rb_mut(),
        Default::default(),
    );
    let ld11 = ld11.into_const();

    let rem = l21.nrows();

    for j in 0..r {
        for i in 0..rem {
            unsafe { l21.write_unchecked(i, j, a21.read_unchecked(i, j)) }
        }
    }

    mul::matmul(
        l21.rb_mut(),
        l20.rb(),
        a01,
        Some(E::faer_one()),
        E::faer_one().faer_neg(),
        parallelism,
    );

    solve::solve_unit_lower_triangular_in_place(
        ld11.conjugate(),
        l21.rb_mut().transpose(),
        parallelism,
    );

    let d1 = ld11.into_const().diagonal().into_column_vector();

    for j in 0..r {
        unsafe {
            let d1_inv = d1.read_unchecked(j, 0);
            for i in 0..rem {
                let value = l21.rb_mut().read_unchecked(i, j).faer_mul(d1_inv);
                l21.write_unchecked(i, j, value);
            }
        }
    }

    let mut alpha = a11.col(0);
    let mut w = a21;

    for j in 0..r {
        unsafe {
            alpha
                .rb_mut()
                .write_unchecked(j, 0, ld11.read_unchecked(j, j).faer_inv().faer_neg());

            for i in 0..rem {
                w.rb_mut()
                    .write_unchecked(i, j, l21.read_unchecked(i, j).faer_neg());
            }
        }
    }

    rank_r_update_clobber(ld22, w, alpha);
}
