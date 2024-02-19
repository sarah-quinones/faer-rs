use crate::{
    assert, debug_assert,
    linalg::{matmul::inner_prod::inner_prod_with_conj, temp_mat_req, temp_mat_zeroed},
    unzipped,
    utils::thread::parallelism_degree,
    zipped, ComplexField, Conj, Entity, MatMut, MatRef, Parallelism,
};
use core::iter::zip;
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_entity::*;
use reborrow::*;

pub fn tridiagonalize_in_place_req<E: Entity>(
    n: usize,
    parallelism: Parallelism,
) -> Result<StackReq, SizeOverflow> {
    let _ = parallelism;
    StackReq::try_all_of([
        temp_mat_req::<E>(n, 1)?,
        temp_mat_req::<E>(n, 1)?,
        temp_mat_req::<E>(n, parallelism_degree(parallelism))?,
        temp_mat_req::<E>(n, parallelism_degree(parallelism))?,
    ])
}

struct SymMatVecWithLhsUpdate<'a, E: Entity> {
    acc: MatMut<'a, E>,
    acct: MatMut<'a, E>,
    lhs: MatMut<'a, E>,
    rhs: MatRef<'a, E>,
    u: MatRef<'a, E>,
    y: MatRef<'a, E>,
    first_col: usize,
    last_col: usize,
}

struct SymMatVec<'a, E: Entity> {
    acc: MatMut<'a, E>,
    lhs: MatRef<'a, E>,
    rhs: MatRef<'a, E>,
    first_col: usize,
    last_col: usize,
}

impl<E: ComplexField> pulp::WithSimd for SymMatVecWithLhsUpdate<'_, E> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            acc,
            acct,
            mut lhs,
            rhs,
            first_col,
            last_col,
            u,
            y,
        } = self;

        debug_assert!(lhs.nrows() == lhs.ncols());
        let n = lhs.nrows();
        debug_assert!(acc.nrows() == n);
        debug_assert!(acc.ncols() == 1);
        debug_assert!(acct.nrows() == n);
        debug_assert!(acct.ncols() == 1);
        debug_assert!(rhs.nrows() == n);
        debug_assert!(rhs.ncols() == 1);
        debug_assert!(u.nrows() == n);
        debug_assert!(u.ncols() == 1);
        debug_assert!(y.nrows() == n);
        debug_assert!(y.ncols() == 1);
        debug_assert!(first_col <= last_col);
        debug_assert!(last_col <= n);
        debug_assert!(lhs.row_stride() == 1);
        debug_assert!(rhs.row_stride() == 1);
        debug_assert!(acc.row_stride() == 1);
        debug_assert!(acct.row_stride() == 1);
        debug_assert!(u.row_stride() == 1);
        debug_assert!(y.row_stride() == 1);

        let lane_count =
            core::mem::size_of::<SimdUnitFor<E, S>>() / core::mem::size_of::<UnitFor<E>>();

        let mut acc = unsafe {
            E::faer_map(acc.as_ptr_mut(), |ptr| {
                core::slice::from_raw_parts_mut(ptr, n)
            })
        };
        let mut acct = unsafe {
            E::faer_map(acct.as_ptr_mut(), |ptr| {
                core::slice::from_raw_parts_mut(ptr, n)
            })
        };
        let rhs = unsafe { E::faer_map(rhs.as_ptr(), |ptr| core::slice::from_raw_parts(ptr, n)) };
        let u = unsafe { E::faer_map(u.as_ptr(), |ptr| core::slice::from_raw_parts(ptr, n)) };
        let y = unsafe { E::faer_map(y.as_ptr(), |ptr| core::slice::from_raw_parts(ptr, n)) };

        let zero = E::faer_zero();

        let rem = (last_col - first_col) % 2;

        for j in first_col..first_col + rem {
            let rhs_single_j = into_copy::<E, _>(E::faer_map(
                E::faer_copy(&rhs),
                #[inline(always)]
                |slice| E::faer_simd_splat_unit(simd, slice[j]),
            ));
            let y_single_j = E::faer_simd_neg(
                simd,
                into_copy::<E, _>(E::faer_map(
                    E::faer_copy(&y),
                    #[inline(always)]
                    |slice| E::faer_simd_splat_unit(simd, slice[j]),
                )),
            );
            let u_single_j = E::faer_simd_neg(
                simd,
                into_copy::<E, _>(E::faer_map(
                    E::faer_copy(&u),
                    #[inline(always)]
                    |slice| E::faer_simd_splat_unit(simd, slice[j]),
                )),
            );

            let start = j;
            let len = n - start;
            debug_assert!(len > 0);

            let acc_j = E::faer_map(
                E::faer_rb_mut(E::faer_as_mut(&mut acc)),
                #[inline(always)]
                |slice| &mut slice[start..],
            );
            let lhs_j = unsafe {
                E::faer_map(
                    lhs.rb_mut().ptr_at_mut(0, j),
                    #[inline(always)]
                    |ptr| core::slice::from_raw_parts_mut(ptr.wrapping_add(start), len),
                )
            };
            let rhs_j = E::faer_map(
                E::faer_copy(&rhs),
                #[inline(always)]
                |slice| &slice[start..],
            );
            let y_j = E::faer_map(
                E::faer_copy(&y),
                #[inline(always)]
                |slice| &slice[start..],
            );
            let u_j = E::faer_map(
                E::faer_copy(&u),
                #[inline(always)]
                |slice| &slice[start..],
            );

            let prefix = ((len - 1) % lane_count) + 1;

            let (acc_prefix, acc_suffix) = E::faer_unzip(E::faer_map(
                acc_j,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let (lhs_prefix, lhs_suffix) = E::faer_unzip(E::faer_map(
                lhs_j,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let (rhs_prefix, rhs_suffix) = E::faer_unzip(E::faer_map(
                rhs_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));
            let (u_prefix, u_suffix) = E::faer_unzip(E::faer_map(
                u_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));
            let (y_prefix, y_suffix) = E::faer_unzip(E::faer_map(
                y_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));

            let acc_suffix = faer_entity::slice_as_mut_simd::<E, S>(acc_suffix).0;
            let lhs_suffix = faer_entity::slice_as_mut_simd::<E, S>(lhs_suffix).0;
            let rhs_suffix = faer_entity::slice_as_simd::<E, S>(rhs_suffix).0;
            let u_suffix = faer_entity::slice_as_simd::<E, S>(u_suffix).0;
            let y_suffix = faer_entity::slice_as_simd::<E, S>(y_suffix).0;

            let mut sum0 = E::faer_simd_splat(simd, zero);
            let mut sum1 = E::faer_simd_splat(simd, zero);
            let mut sum2 = E::faer_simd_splat(simd, zero);
            let mut sum3 = E::faer_simd_splat(simd, zero);

            let u_prefix = E::faer_partial_load_last(simd, u_prefix);
            let y_prefix = E::faer_partial_load_last(simd, y_prefix);

            let mut lhs_prefix_ =
                E::faer_partial_load_last(simd, E::faer_rb(E::faer_as_ref(&lhs_prefix)));

            lhs_prefix_ = E::faer_simd_conj_mul_adde(simd, u_single_j, y_prefix, lhs_prefix_);
            lhs_prefix_ = E::faer_simd_conj_mul_adde(simd, y_single_j, u_prefix, lhs_prefix_);

            E::faer_partial_store_last(simd, lhs_prefix, lhs_prefix_);

            let rhs_prefix = E::faer_partial_load_last(simd, rhs_prefix);
            let acc_prefix = E::faer_map(acc_prefix, |slice| &mut slice[1..]);
            let mut acc_prefix_ =
                E::faer_partial_load_last(simd, E::faer_rb(E::faer_as_ref(&acc_prefix)));

            acc_prefix_ = E::faer_simd_mul_adde(simd, lhs_prefix_, rhs_single_j, acc_prefix_);
            E::faer_partial_store_last(simd, acc_prefix, acc_prefix_);

            sum0 = E::faer_simd_conj_mul_adde(simd, lhs_prefix_, rhs_prefix, sum0);

            let (acc_head, acc_tail) = E::faer_as_arrays_mut::<4, _>(acc_suffix);
            let (lhs_head, lhs_tail) = E::faer_as_arrays_mut::<4, _>(lhs_suffix);
            let (rhs_head, rhs_tail) = E::faer_as_arrays::<4, _>(rhs_suffix);
            let (u_head, u_tail) = E::faer_as_arrays::<4, _>(u_suffix);
            let (y_head, y_tail) = E::faer_as_arrays::<4, _>(y_suffix);

            for (acc, (lhs, (rhs, (u, y)))) in zip(
                E::faer_into_iter(acc_head),
                zip(
                    E::faer_into_iter(lhs_head),
                    zip(
                        E::faer_into_iter(rhs_head),
                        zip(E::faer_into_iter(u_head), E::faer_into_iter(y_head)),
                    ),
                ),
            ) {
                let [mut lhs0, mut lhs1, mut lhs2, mut lhs3] =
                    E::faer_unzip4(E::faer_deref(E::faer_rb(E::faer_as_ref(&lhs))));

                let [u0, u1, u2, u3] = E::faer_unzip4(E::faer_deref(u));
                let [y0, y1, y2, y3] = E::faer_unzip4(E::faer_deref(y));

                lhs0 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    u_single_j,
                    into_copy::<E, _>(y0),
                    into_copy::<E, _>(lhs0),
                ));
                lhs0 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    y_single_j,
                    into_copy::<E, _>(u0),
                    into_copy::<E, _>(lhs0),
                ));
                lhs1 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    u_single_j,
                    into_copy::<E, _>(y1),
                    into_copy::<E, _>(lhs1),
                ));
                lhs1 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    y_single_j,
                    into_copy::<E, _>(u1),
                    into_copy::<E, _>(lhs1),
                ));
                lhs2 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    u_single_j,
                    into_copy::<E, _>(y2),
                    into_copy::<E, _>(lhs2),
                ));
                lhs2 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    y_single_j,
                    into_copy::<E, _>(u2),
                    into_copy::<E, _>(lhs2),
                ));
                lhs3 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    u_single_j,
                    into_copy::<E, _>(y3),
                    into_copy::<E, _>(lhs3),
                ));
                lhs3 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    y_single_j,
                    into_copy::<E, _>(u3),
                    into_copy::<E, _>(lhs3),
                ));

                E::faer_map(
                    E::faer_zip(
                        lhs,
                        E::faer_zip(
                            E::faer_zip(E::faer_copy(&lhs0), E::faer_copy(&lhs1)),
                            E::faer_zip(E::faer_copy(&lhs2), E::faer_copy(&lhs3)),
                        ),
                    ),
                    #[inline(always)]
                    |(lhs, ((lhs0, lhs1), (lhs2, lhs3)))| {
                        lhs[0] = lhs0;
                        lhs[1] = lhs1;
                        lhs[2] = lhs2;
                        lhs[3] = lhs3;
                    },
                );

                let [rhs0, rhs1, rhs2, rhs3] = E::faer_unzip4(E::faer_deref(rhs));
                let [mut acc0, mut acc1, mut acc2, mut acc3] =
                    E::faer_unzip4(E::faer_deref(E::faer_rb(E::faer_as_ref(&acc))));
                acc0 = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs0)),
                    rhs_single_j,
                    into_copy::<E, _>(acc0),
                ));
                acc1 = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs1)),
                    rhs_single_j,
                    into_copy::<E, _>(acc1),
                ));
                acc2 = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs2)),
                    rhs_single_j,
                    into_copy::<E, _>(acc2),
                ));
                acc3 = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs3)),
                    rhs_single_j,
                    into_copy::<E, _>(acc3),
                ));
                E::faer_map(
                    E::faer_zip(
                        acc,
                        E::faer_zip(E::faer_zip(acc0, acc1), E::faer_zip(acc2, acc3)),
                    ),
                    #[inline(always)]
                    |(acc, ((acc0, acc1), (acc2, acc3)))| {
                        acc[0] = acc0;
                        acc[1] = acc1;
                        acc[2] = acc2;
                        acc[3] = acc3;
                    },
                );
                sum0 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs0)),
                    into_copy::<E, _>(rhs0),
                    sum0,
                );
                sum1 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs1)),
                    into_copy::<E, _>(rhs1),
                    sum1,
                );
                sum2 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs2)),
                    into_copy::<E, _>(rhs2),
                    sum2,
                );
                sum3 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs3)),
                    into_copy::<E, _>(rhs3),
                    sum3,
                );
            }

            sum0 = E::faer_simd_add(simd, sum0, sum1);
            sum2 = E::faer_simd_add(simd, sum2, sum3);

            sum0 = E::faer_simd_add(simd, sum0, sum2);

            for (acc, (lhs, (rhs, (u, y)))) in zip(
                E::faer_into_iter(acc_tail),
                zip(
                    E::faer_into_iter(lhs_tail),
                    zip(
                        E::faer_into_iter(rhs_tail),
                        zip(E::faer_into_iter(u_tail), E::faer_into_iter(y_tail)),
                    ),
                ),
            ) {
                let mut lhs0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&lhs)));
                let u0 = E::faer_deref(u);
                let y0 = E::faer_deref(y);
                lhs0 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    u_single_j,
                    into_copy::<E, _>(y0),
                    into_copy::<E, _>(lhs0),
                ));
                lhs0 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    y_single_j,
                    into_copy::<E, _>(u0),
                    into_copy::<E, _>(lhs0),
                ));

                E::faer_map(
                    E::faer_zip(lhs, E::faer_copy(&lhs0)),
                    #[inline(always)]
                    |(lhs, lhs0)| {
                        *lhs = lhs0;
                    },
                );

                let rhs0 = E::faer_deref(rhs);
                let mut acc0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&acc)));
                acc0 = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs0)),
                    rhs_single_j,
                    into_copy::<E, _>(acc0),
                ));
                E::faer_map(
                    E::faer_zip(acc, acc0),
                    #[inline(always)]
                    |(acc, acc0)| {
                        *acc = acc0;
                    },
                );
                sum0 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs0)),
                    into_copy::<E, _>(rhs0),
                    sum0,
                );
            }

            let sum = E::faer_simd_reduce_add(simd, sum0);
            let sum = E::faer_into_units(sum);

            E::faer_map(
                E::faer_zip(E::faer_rb_mut(E::faer_as_mut(&mut acct)), sum),
                #[inline(always)]
                |(slice, sum)| slice[j] = sum,
            );
        }

        for j in (first_col + rem..last_col).step_by(2) {
            let sum0_scalar = {
                let u = E::faer_from_units(E::faer_map(
                    E::faer_copy(&u),
                    #[inline(always)]
                    |slice| slice[j],
                ));
                let y = E::faer_from_units(E::faer_map(
                    E::faer_copy(&y),
                    #[inline(always)]
                    |slice| slice[j],
                ));
                let rhs = E::faer_from_units(E::faer_map(
                    E::faer_copy(&rhs),
                    #[inline(always)]
                    |slice| slice[j],
                ));

                let mut ljj = lhs.read(j, j);
                ljj = ljj
                    .faer_sub(u.faer_mul(y.faer_conj()))
                    .faer_sub(y.faer_mul(u.faer_conj()));
                lhs.write(j, j, ljj);

                ljj.faer_conj().faer_mul(rhs)
            };

            let rhs_single_j0 = into_copy::<E, _>(E::faer_map(
                E::faer_copy(&rhs),
                #[inline(always)]
                |slice| E::faer_simd_splat_unit(simd, slice[j]),
            ));
            let y_single_j0 = E::faer_simd_neg(
                simd,
                into_copy::<E, _>(E::faer_map(
                    E::faer_copy(&y),
                    #[inline(always)]
                    |slice| E::faer_simd_splat_unit(simd, slice[j]),
                )),
            );
            let u_single_j0 = E::faer_simd_neg(
                simd,
                into_copy::<E, _>(E::faer_map(
                    E::faer_copy(&u),
                    #[inline(always)]
                    |slice| E::faer_simd_splat_unit(simd, slice[j]),
                )),
            );

            let rhs_single_j1 = into_copy::<E, _>(E::faer_map(
                E::faer_copy(&rhs),
                #[inline(always)]
                |slice| E::faer_simd_splat_unit(simd, slice[j + 1]),
            ));
            let y_single_j1 = E::faer_simd_neg(
                simd,
                into_copy::<E, _>(E::faer_map(
                    E::faer_copy(&y),
                    #[inline(always)]
                    |slice| E::faer_simd_splat_unit(simd, slice[j + 1]),
                )),
            );
            let u_single_j1 = E::faer_simd_neg(
                simd,
                into_copy::<E, _>(E::faer_map(
                    E::faer_copy(&u),
                    #[inline(always)]
                    |slice| E::faer_simd_splat_unit(simd, slice[j + 1]),
                )),
            );

            let start = j + 1;
            let len = n - start;
            debug_assert!(len > 0);

            let acc_j = E::faer_map(
                E::faer_rb_mut(E::faer_as_mut(&mut acc)),
                #[inline(always)]
                |slice| &mut slice[start..],
            );
            let lhs_j0 = unsafe {
                E::faer_map(
                    lhs.rb_mut().ptr_at_mut(0, j),
                    #[inline(always)]
                    |ptr| core::slice::from_raw_parts_mut(ptr.wrapping_add(start), len),
                )
            };
            let lhs_j1 = unsafe {
                E::faer_map(
                    lhs.rb_mut().ptr_at_mut(0, j + 1),
                    #[inline(always)]
                    |ptr| core::slice::from_raw_parts_mut(ptr.wrapping_add(start), len),
                )
            };
            let rhs_j = E::faer_map(
                E::faer_copy(&rhs),
                #[inline(always)]
                |slice| &slice[start..],
            );
            let y_j = E::faer_map(
                E::faer_copy(&y),
                #[inline(always)]
                |slice| &slice[start..],
            );
            let u_j = E::faer_map(
                E::faer_copy(&u),
                #[inline(always)]
                |slice| &slice[start..],
            );

            let prefix = ((len - 1) % lane_count) + 1;

            let (mut acc_prefix, acc_suffix) = E::faer_unzip(E::faer_map(
                acc_j,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let (lhs0_prefix, lhs0_suffix) = E::faer_unzip(E::faer_map(
                lhs_j0,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let (lhs1_prefix, lhs1_suffix) = E::faer_unzip(E::faer_map(
                lhs_j1,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let (rhs_prefix, rhs_suffix) = E::faer_unzip(E::faer_map(
                rhs_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));
            let (u_prefix, u_suffix) = E::faer_unzip(E::faer_map(
                u_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));
            let (y_prefix, y_suffix) = E::faer_unzip(E::faer_map(
                y_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));

            let acc_suffix = faer_entity::slice_as_mut_simd::<E, S>(acc_suffix).0;
            let lhs0_suffix = faer_entity::slice_as_mut_simd::<E, S>(lhs0_suffix).0;
            let lhs1_suffix = faer_entity::slice_as_mut_simd::<E, S>(lhs1_suffix).0;
            let rhs_suffix = faer_entity::slice_as_simd::<E, S>(rhs_suffix).0;
            let u_suffix = faer_entity::slice_as_simd::<E, S>(u_suffix).0;
            let y_suffix = faer_entity::slice_as_simd::<E, S>(y_suffix).0;

            let mut sum0 = E::faer_simd_splat(simd, zero);
            let mut sum1 = E::faer_simd_splat(simd, zero);

            let u_prefix = E::faer_partial_load_last(simd, u_prefix);
            let y_prefix = E::faer_partial_load_last(simd, y_prefix);

            let mut lhs0_prefix_ =
                E::faer_partial_load_last(simd, E::faer_rb(E::faer_as_ref(&lhs0_prefix)));
            let mut lhs1_prefix_ =
                E::faer_partial_load_last(simd, E::faer_rb(E::faer_as_ref(&lhs1_prefix)));

            lhs0_prefix_ = E::faer_simd_conj_mul_adde(simd, u_single_j0, y_prefix, lhs0_prefix_);
            lhs0_prefix_ = E::faer_simd_conj_mul_adde(simd, y_single_j0, u_prefix, lhs0_prefix_);

            lhs1_prefix_ = E::faer_simd_conj_mul_adde(simd, y_single_j1, u_prefix, lhs1_prefix_);
            lhs1_prefix_ = E::faer_simd_conj_mul_adde(simd, u_single_j1, y_prefix, lhs1_prefix_);

            E::faer_partial_store_last(simd, lhs0_prefix, lhs0_prefix_);
            E::faer_partial_store_last(simd, lhs1_prefix, lhs1_prefix_);

            let rhs_prefix = E::faer_partial_load_last(simd, rhs_prefix);
            {
                let acc_prefix = E::faer_rb_mut(E::faer_as_mut(&mut acc_prefix));
                let mut acc_prefix_ =
                    E::faer_partial_load_last(simd, E::faer_rb(E::faer_as_ref(&acc_prefix)));

                acc_prefix_ = E::faer_simd_mul_adde(simd, lhs0_prefix_, rhs_single_j0, acc_prefix_);
                E::faer_partial_store_last(simd, acc_prefix, acc_prefix_);
            }
            {
                let acc_prefix = E::faer_map(acc_prefix, |slice| &mut slice[1..]);
                let mut acc_prefix_ =
                    E::faer_partial_load_last(simd, E::faer_rb(E::faer_as_ref(&acc_prefix)));

                acc_prefix_ = E::faer_simd_mul_adde(simd, lhs1_prefix_, rhs_single_j1, acc_prefix_);
                E::faer_partial_store_last(simd, acc_prefix, acc_prefix_);
            }

            sum0 = E::faer_simd_conj_mul_adde(simd, lhs0_prefix_, rhs_prefix, sum0);
            sum1 = E::faer_simd_conj_mul_adde(simd, lhs1_prefix_, rhs_prefix, sum1);

            for (acc, ((lhs_j0, lhs_j1), (rhs, (u, y)))) in zip(
                E::faer_into_iter(acc_suffix),
                zip(
                    zip(
                        E::faer_into_iter(lhs0_suffix),
                        E::faer_into_iter(lhs1_suffix),
                    ),
                    zip(
                        E::faer_into_iter(rhs_suffix),
                        zip(E::faer_into_iter(u_suffix), E::faer_into_iter(y_suffix)),
                    ),
                ),
            ) {
                let mut lhs0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&lhs_j0)));
                let mut lhs1 = E::faer_deref(E::faer_rb(E::faer_as_ref(&lhs_j1)));
                let u = E::faer_deref(u);
                let y = E::faer_deref(y);
                lhs0 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    u_single_j0,
                    into_copy::<E, _>(E::faer_copy(&y)),
                    into_copy::<E, _>(lhs0),
                ));
                lhs0 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    y_single_j0,
                    into_copy::<E, _>(E::faer_copy(&u)),
                    into_copy::<E, _>(lhs0),
                ));
                lhs1 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    u_single_j1,
                    into_copy::<E, _>(y),
                    into_copy::<E, _>(lhs1),
                ));
                lhs1 = from_copy::<E, _>(E::faer_simd_conj_mul_adde(
                    simd,
                    y_single_j1,
                    into_copy::<E, _>(u),
                    into_copy::<E, _>(lhs1),
                ));

                E::faer_map(
                    E::faer_zip(lhs_j0, E::faer_copy(&lhs0)),
                    #[inline(always)]
                    |(lhs, lhs0)| {
                        *lhs = lhs0;
                    },
                );
                E::faer_map(
                    E::faer_zip(lhs_j1, E::faer_copy(&lhs1)),
                    #[inline(always)]
                    |(lhs, lhs1)| {
                        *lhs = lhs1;
                    },
                );

                let rhs = E::faer_deref(rhs);
                let mut acc_ = E::faer_deref(E::faer_rb(E::faer_as_ref(&acc)));
                acc_ = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs0)),
                    rhs_single_j0,
                    into_copy::<E, _>(acc_),
                ));
                acc_ = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs1)),
                    rhs_single_j1,
                    into_copy::<E, _>(acc_),
                ));
                E::faer_map(
                    E::faer_zip(acc, acc_),
                    #[inline(always)]
                    |(acc, acc_)| {
                        *acc = acc_;
                    },
                );
                sum0 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(lhs0),
                    into_copy::<E, _>(E::faer_copy(&rhs)),
                    sum0,
                );
                sum1 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(lhs1),
                    into_copy::<E, _>(E::faer_copy(&rhs)),
                    sum1,
                );
            }

            {
                let sum = E::faer_simd_reduce_add(simd, sum0).faer_add(sum0_scalar);
                let sum = E::faer_into_units(sum);

                E::faer_map(
                    E::faer_zip(E::faer_rb_mut(E::faer_as_mut(&mut acct)), sum),
                    #[inline(always)]
                    |(slice, sum)| slice[j] = sum,
                );
            }
            {
                let sum = E::faer_simd_reduce_add(simd, sum1);
                let sum = E::faer_into_units(sum);

                E::faer_map(
                    E::faer_zip(E::faer_rb_mut(E::faer_as_mut(&mut acct)), sum),
                    #[inline(always)]
                    |(slice, sum)| slice[j + 1] = sum,
                );
            }
        }
    }
}

impl<E: ComplexField> pulp::WithSimd for SymMatVec<'_, E> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            acc,
            lhs,
            rhs,
            first_col,
            last_col,
        } = self;

        debug_assert!(lhs.nrows() == lhs.ncols());
        let n = lhs.nrows();
        debug_assert!(acc.nrows() == n);
        debug_assert!(acc.ncols() == 1);
        debug_assert!(rhs.nrows() == n);
        debug_assert!(rhs.ncols() == 1);
        debug_assert!(first_col < n);
        debug_assert!(last_col <= n);
        debug_assert!(lhs.row_stride() == 1);
        debug_assert!(rhs.row_stride() == 1);
        debug_assert!(acc.row_stride() == 1);

        let lane_count =
            core::mem::size_of::<SimdUnitFor<E, S>>() / core::mem::size_of::<UnitFor<E>>();

        let mut acc = unsafe {
            E::faer_map(acc.as_ptr_mut(), |ptr| {
                core::slice::from_raw_parts_mut(ptr, n)
            })
        };
        let rhs = unsafe { E::faer_map(rhs.as_ptr(), |ptr| core::slice::from_raw_parts(ptr, n)) };

        let zero = E::faer_zero();
        for j in first_col..last_col {
            let start = j + 1;
            let len = n - start;

            let acc_j = E::faer_map(
                E::faer_rb_mut(E::faer_as_mut(&mut acc)),
                #[inline(always)]
                |slice| &mut slice[start..],
            );
            let lhs_j = unsafe {
                E::faer_map(
                    lhs.ptr_at(start, j),
                    #[inline(always)]
                    |ptr| core::slice::from_raw_parts(ptr, len),
                )
            };
            let rhs_j = E::faer_map(
                E::faer_copy(&rhs),
                #[inline(always)]
                |slice| &slice[start..],
            );

            let (acc_prefix, acc_suffix) = E::faer_unzip(E::faer_map(
                acc_j,
                #[inline(always)]
                |slice| slice.split_at_mut(len % lane_count),
            ));
            let (lhs_prefix, lhs_suffix) = E::faer_unzip(E::faer_map(
                lhs_j,
                #[inline(always)]
                |slice| slice.split_at(len % lane_count),
            ));
            let (rhs_prefix, rhs_suffix) = E::faer_unzip(E::faer_map(
                rhs_j,
                #[inline(always)]
                |slice| slice.split_at(len % lane_count),
            ));

            let acc_suffix = faer_entity::slice_as_mut_simd::<E, S>(acc_suffix).0;
            let lhs_suffix = faer_entity::slice_as_simd::<E, S>(lhs_suffix).0;
            let rhs_suffix = faer_entity::slice_as_simd::<E, S>(rhs_suffix).0;

            let rhs_single_j = into_copy::<E, _>(E::faer_map(
                E::faer_copy(&rhs),
                #[inline(always)]
                |slice| E::faer_simd_splat_unit(simd, slice[j]),
            ));

            let mut sum0 = E::faer_simd_splat(simd, zero);
            let mut sum1 = E::faer_simd_splat(simd, zero);
            let mut sum2 = E::faer_simd_splat(simd, zero);
            let mut sum3 = E::faer_simd_splat(simd, zero);

            let lhs_prefix = E::faer_partial_load_last(simd, lhs_prefix);
            let rhs_prefix = E::faer_partial_load_last(simd, rhs_prefix);
            let mut acc_prefix_ =
                E::faer_partial_load_last(simd, E::faer_rb(E::faer_as_ref(&acc_prefix)));

            acc_prefix_ = E::faer_simd_mul_adde(simd, lhs_prefix, rhs_single_j, acc_prefix_);
            E::faer_partial_store_last(simd, acc_prefix, acc_prefix_);

            sum0 = E::faer_simd_conj_mul_adde(simd, lhs_prefix, rhs_prefix, sum0);

            let (acc_head, acc_tail) = E::faer_as_arrays_mut::<4, _>(acc_suffix);
            let (lhs_head, lhs_tail) = E::faer_as_arrays::<4, _>(lhs_suffix);
            let (rhs_head, rhs_tail) = E::faer_as_arrays::<4, _>(rhs_suffix);

            for (acc, (lhs, rhs)) in zip(
                E::faer_into_iter(acc_head),
                zip(E::faer_into_iter(lhs_head), E::faer_into_iter(rhs_head)),
            ) {
                let [lhs0, lhs1, lhs2, lhs3] = E::faer_unzip4(E::faer_deref(lhs));
                let [rhs0, rhs1, rhs2, rhs3] = E::faer_unzip4(E::faer_deref(rhs));
                let [mut acc0, mut acc1, mut acc2, mut acc3] =
                    E::faer_unzip4(E::faer_deref(E::faer_rb(E::faer_as_ref(&acc))));
                acc0 = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs0)),
                    rhs_single_j,
                    into_copy::<E, _>(acc0),
                ));
                acc1 = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs1)),
                    rhs_single_j,
                    into_copy::<E, _>(acc1),
                ));
                acc2 = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs2)),
                    rhs_single_j,
                    into_copy::<E, _>(acc2),
                ));
                acc3 = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs3)),
                    rhs_single_j,
                    into_copy::<E, _>(acc3),
                ));
                E::faer_map(
                    E::faer_zip(
                        acc,
                        E::faer_zip(E::faer_zip(acc0, acc1), E::faer_zip(acc2, acc3)),
                    ),
                    #[inline(always)]
                    |(acc, ((acc0, acc1), (acc2, acc3)))| {
                        acc[0] = acc0;
                        acc[1] = acc1;
                        acc[2] = acc2;
                        acc[3] = acc3;
                    },
                );
                sum0 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(lhs0),
                    into_copy::<E, _>(rhs0),
                    sum0,
                );
                sum1 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(lhs1),
                    into_copy::<E, _>(rhs1),
                    sum1,
                );
                sum2 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(lhs2),
                    into_copy::<E, _>(rhs2),
                    sum2,
                );
                sum3 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(lhs3),
                    into_copy::<E, _>(rhs3),
                    sum3,
                );
            }

            sum0 = E::faer_simd_add(simd, sum0, sum1);
            sum2 = E::faer_simd_add(simd, sum2, sum3);

            sum0 = E::faer_simd_add(simd, sum0, sum2);

            for (acc, (lhs, rhs)) in zip(
                E::faer_into_iter(acc_tail),
                zip(E::faer_into_iter(lhs_tail), E::faer_into_iter(rhs_tail)),
            ) {
                let lhs0 = E::faer_deref(lhs);
                let rhs0 = E::faer_deref(rhs);
                let mut acc0 = E::faer_deref(E::faer_rb(E::faer_as_ref(&acc)));
                acc0 = from_copy::<E, _>(E::faer_simd_mul_adde(
                    simd,
                    into_copy::<E, _>(E::faer_copy(&lhs0)),
                    rhs_single_j,
                    into_copy::<E, _>(acc0),
                ));
                E::faer_map(
                    E::faer_zip(acc, acc0),
                    #[inline(always)]
                    |(acc, acc0)| {
                        *acc = acc0;
                    },
                );
                sum0 = E::faer_simd_conj_mul_adde(
                    simd,
                    into_copy::<E, _>(lhs0),
                    into_copy::<E, _>(rhs0),
                    sum0,
                );
            }

            let mut sum = E::faer_simd_reduce_add(simd, sum0);
            let acc_ = E::faer_from_units(E::faer_map(
                E::faer_rb(E::faer_as_ref(&acc)),
                #[inline(always)]
                |slice| slice[j],
            ));

            sum = sum.faer_add(acc_);
            let sum = E::faer_into_units(sum);

            E::faer_map(
                E::faer_zip(E::faer_rb_mut(E::faer_as_mut(&mut acc)), sum),
                #[inline(always)]
                |(slice, sum)| slice[j] = sum,
            );
        }
    }
}

pub fn tridiagonalize_in_place<E: ComplexField>(
    mut a: MatMut<'_, E>,
    mut householder: MatMut<'_, E>,
    parallelism: Parallelism,
    stack: PodStack<'_>,
) {
    assert!(a.nrows() == a.ncols());
    assert!(a.row_stride() == 1);

    let n = a.nrows();
    if n < 2 {
        return;
    }

    let (mut u, stack) = temp_mat_zeroed::<E>(n, 1, stack);
    let (mut y, stack) = temp_mat_zeroed::<E>(n, 1, stack);

    let (mut v, stack) = temp_mat_zeroed::<E>(n, parallelism_degree(parallelism), stack);
    let (mut w, _) = temp_mat_zeroed::<E>(n, parallelism_degree(parallelism), stack);

    let mut u = u.as_mut();
    let mut y = y.as_mut();
    let mut v = v.as_mut();
    let mut w = w.as_mut();

    let arch = E::Simd::default();
    for k in 0..n - 1 {
        let a_cur = a.rb_mut().submatrix_mut(k, k, n - k, n - k);
        let (mut a11, _, mut a21, a22) = a_cur.split_at_mut(1, 1);

        let parallelism = if n - k <= 256 {
            Parallelism::None
        } else {
            parallelism
        };

        let (_, u) = u.rb_mut().split_at_row_mut(k);
        let (nu, mut u21) = u.split_at_row_mut(1);
        let (_, y) = y.rb_mut().split_at_row_mut(k);
        let (psi, mut y21) = y.split_at_row_mut(1);

        let (_, v) = v.rb_mut().split_at_row_mut(k);
        let (_, v21) = v.split_at_row_mut(1);
        let mut v21 = v21.subcols_mut(0, parallelism_degree(parallelism));

        let (_, w) = w.rb_mut().split_at_row_mut(k);
        let (_, w21) = w.split_at_row_mut(1);
        let w21 = w21.subcols_mut(0, parallelism_degree(parallelism));

        if k > 0 {
            let nu = nu.read(0, 0);
            let psi = psi.read(0, 0);
            a11.write(
                0,
                0,
                a11.read(0, 0).faer_sub(
                    nu.faer_mul(psi.faer_conj())
                        .faer_add(psi.faer_mul(nu.faer_conj())),
                ),
            );

            zipped!(a21.rb_mut(), u21.rb(), y21.rb()).for_each(|unzipped!(mut a, u, y)| {
                let u = u.read();
                let y = y.read();
                a.write(
                    a.read().faer_sub(
                        u.faer_mul(psi.faer_conj())
                            .faer_add(y.faer_mul(nu.faer_conj())),
                    ),
                )
            });
        }

        let (tau, new_head) = {
            let (head, tail) = a21.rb_mut().split_at_row_mut(1);
            let norm = tail.rb().norm_l2();
            crate::linalg::householder::make_householder_in_place(Some(tail), head.read(0, 0), norm)
        };
        a21.write(0, 0, E::faer_one());
        let tau_inv = tau.faer_inv();
        householder.write(k, 0, tau);

        if k > 0 {
            let ncols = (n - k - 1) as f64;
            let n_threads = parallelism_degree(parallelism) as f64;

            const TWO_POW_50: f64 = 1125899906842624.0;
            assert!(ncols < TWO_POW_50); // to check that integers can be
                                         // represented exactly as floats

            let idx_to_col_start = |idx: usize| {
                let idx_as_percent = idx as f64 / n_threads;
                let col_start_percent = 1.0f64 - (1.0f64 - idx_as_percent).faer_sqrt();
                (col_start_percent * ncols) as usize
            };

            crate::utils::thread::for_each_raw(
                parallelism_degree(parallelism),
                |idx| {
                    let first_col = idx_to_col_start(idx);
                    let last_col = idx_to_col_start(idx + 1);

                    let mut v21 = unsafe { v21.rb().col(idx).const_cast().as_2d_mut() };
                    let mut w21 = unsafe { w21.rb().col(idx).const_cast().as_2d_mut() };

                    zipped!(v21.rb_mut()).for_each(|unzipped!(mut z)| z.write(E::faer_zero()));
                    zipped!(w21.rb_mut()).for_each(|unzipped!(mut z)| z.write(E::faer_zero()));

                    let acc = v21.rb_mut();
                    let acct = w21.rb_mut();
                    let lhs = unsafe { a22.rb().const_cast() };
                    let rhs = a21.rb();

                    let u = u21.rb();
                    let y = y21.rb();

                    arch.dispatch(SymMatVecWithLhsUpdate {
                        acc,
                        acct,
                        lhs,
                        rhs,
                        first_col,
                        last_col,
                        u,
                        y,
                    });
                },
                parallelism,
            );

            zipped!(
                y21.rb_mut(),
                v21.rb().col(0).as_2d(),
                w21.rb().col(0).as_2d()
            )
            .for_each(|unzipped!(mut y, v, w)| y.write(v.read().faer_add(w.read())));
            for i in 1..n_threads as usize {
                zipped!(
                    y21.rb_mut(),
                    v21.rb().col(i).as_2d(),
                    w21.rb().col(i).as_2d()
                )
                .for_each(|unzipped!(mut y, v, w)| {
                    y.write(y.read().faer_add(v.read().faer_add(w.read())))
                });
            }
        } else {
            let mut acc = v21.rb_mut().col_mut(0).as_2d_mut();
            let lhs = a22.rb();
            let rhs = a21.rb();

            let first_col = 0;
            let last_col = n - k - 1;

            arch.dispatch(SymMatVec {
                acc: acc.rb_mut(),
                lhs,
                rhs,
                first_col,
                last_col,
            });

            zipped!(
                y21.rb_mut(),
                acc.rb(),
                a22.rb().diagonal().column_vector().as_2d(),
                a21.rb(),
            )
            .for_each(|unzipped!(mut y, v, a, u)| {
                y.write(v.read().faer_add(a.read().faer_mul(u.read())))
            });
        }

        zipped!(u21.rb_mut(), a21.rb()).for_each(|unzipped!(mut dst, src)| dst.write(src.read()));
        a21.write(0, 0, new_head);

        let beta = inner_prod_with_conj(u21.rb(), Conj::Yes, y21.rb(), Conj::No)
            .faer_scale_power_of_two(E::Real::faer_from_f64(0.5));

        zipped!(y21.rb_mut(), u21.rb()).for_each(|unzipped!(mut y, u)| {
            let u = u.read();
            y.write(
                y.read()
                    .faer_sub(beta.faer_mul(u.faer_mul(tau_inv)))
                    .faer_mul(tau_inv),
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert,
        complex_native::c64,
        linalg::householder::{
            apply_block_householder_sequence_on_the_right_in_place_req,
            apply_block_householder_sequence_on_the_right_in_place_with_conj,
            apply_block_householder_sequence_transpose_on_the_left_in_place_req,
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
        },
        Mat,
    };
    use assert_approx_eq::assert_approx_eq;

    macro_rules! make_stack {
        ($req: expr $(,)?) => {
            ::dyn_stack::PodStack::new(&mut ::dyn_stack::GlobalPodBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn tridiag_f64() {
        for (n, parallelism) in [(10, Parallelism::None), (512, Parallelism::Rayon(4))] {
            let mut mat = Mat::from_fn(n, n, |_, _| rand::random::<f64>());
            let transpose = mat.transpose().to_owned();

            zipped!(mat.as_mut(), transpose.as_ref())
                .for_each(|unzipped!(mut x, y)| x.write(x.read() + y.read()));

            let n = mat.ncols();

            let mut trid = mat.clone();
            let mut tau_left = Mat::zeros(n - 1, 1);

            tridiagonalize_in_place(
                trid.as_mut(),
                tau_left.as_mut().col_mut(0).as_2d_mut(),
                parallelism,
                make_stack!(tridiagonalize_in_place_req::<f64>(n, parallelism)),
            );

            let mut copy = mat.clone();
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                trid.as_ref().submatrix(1, 0, n - 1, n - 1),
                tau_left.as_ref().transpose(),
                Conj::No,
                copy.as_mut().submatrix_mut(1, 0, n - 1, n),
                Parallelism::None,
                make_stack!(
                    apply_block_householder_sequence_transpose_on_the_left_in_place_req::<f64>(
                        n - 1,
                        1,
                        n
                    ),
                ),
            );

            apply_block_householder_sequence_on_the_right_in_place_with_conj(
                trid.as_ref().submatrix(1, 0, n - 1, n - 1),
                tau_left.as_ref().transpose(),
                Conj::No,
                copy.as_mut().submatrix_mut(0, 1, n, n - 1),
                Parallelism::None,
                make_stack!(
                    apply_block_householder_sequence_on_the_right_in_place_req::<f64>(n - 1, 1, n,)
                ),
            );

            for j in 0..n {
                for i in (0..j.saturating_sub(1)).chain(j + 2..n) {
                    assert_approx_eq!(copy.read(i, j), 0.0);
                }
            }
        }
    }

    #[test]
    fn tridiag_c64() {
        for (n, parallelism) in [(10, Parallelism::None), (64, Parallelism::Rayon(4))] {
            let mut mat = Mat::from_fn(n, n, |_, _| c64::new(rand::random(), rand::random()));
            let transpose = mat.adjoint().to_owned();
            zipped!(mat.as_mut(), transpose.as_ref())
                .for_each(|unzipped!(mut x, y)| x.write(x.read() + y.read()));

            let n = mat.ncols();

            let mut trid = mat.clone();
            let mut tau_left = Mat::zeros(n - 1, 1);

            tridiagonalize_in_place(
                trid.as_mut(),
                tau_left.as_mut().col_mut(0).as_2d_mut(),
                parallelism,
                make_stack!(tridiagonalize_in_place_req::<c64>(n, parallelism)),
            );

            let mut copy = mat.clone();
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                trid.as_ref().submatrix(1, 0, n - 1, n - 1),
                tau_left.as_ref().transpose(),
                Conj::Yes,
                copy.as_mut().submatrix_mut(1, 0, n - 1, n),
                Parallelism::None,
                make_stack!(
                    apply_block_householder_sequence_transpose_on_the_left_in_place_req::<c64>(
                        n - 1,
                        1,
                        n
                    ),
                ),
            );

            apply_block_householder_sequence_on_the_right_in_place_with_conj(
                trid.as_ref().submatrix(1, 0, n - 1, n - 1),
                tau_left.as_ref().transpose(),
                Conj::No,
                copy.as_mut().submatrix_mut(0, 1, n, n - 1),
                Parallelism::None,
                make_stack!(
                    apply_block_householder_sequence_on_the_right_in_place_req::<c64>(n - 1, 1, n,)
                ),
            );

            for j in 0..n {
                for i in (0..j.saturating_sub(1)).chain(j + 2..n) {
                    assert_approx_eq!(copy.read(i, j), c64::faer_zero());
                }
            }
        }
    }
}
