use crate::tridiag_real_evd::norm2;
use assert2::{assert, debug_assert};
use core::iter::zip;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use faer_core::{
    mul::{
        inner_prod::inner_prod_with_conj,
        triangular::{self, BlockStructure},
    },
    parallelism_degree, temp_mat_req, temp_mat_zeroed, zipped, ComplexField, Conj, Entity, MatMut,
    MatRef, Parallelism,
};
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

        let lane_count = core::mem::size_of::<E::SimdUnit<S>>() / core::mem::size_of::<E::Unit>();

        let mut acc =
            unsafe { E::map(acc.as_ptr(), |ptr| core::slice::from_raw_parts_mut(ptr, n)) };
        let mut acct =
            unsafe { E::map(acct.as_ptr(), |ptr| core::slice::from_raw_parts_mut(ptr, n)) };
        let rhs = unsafe { E::map(rhs.as_ptr(), |ptr| core::slice::from_raw_parts(ptr, n)) };
        let u = unsafe { E::map(u.as_ptr(), |ptr| core::slice::from_raw_parts(ptr, n)) };
        let y = unsafe { E::map(y.as_ptr(), |ptr| core::slice::from_raw_parts(ptr, n)) };

        let zero = E::zero();

        let rem = (last_col - first_col) % 2;

        for j in first_col..first_col + rem {
            let rhs_single_j = E::map(
                E::copy(&rhs),
                #[inline(always)]
                |slice| E::simd_splat_unit(simd, slice[j].clone()),
            );
            let y_single_j = E::simd_neg(
                simd,
                E::map(
                    E::copy(&y),
                    #[inline(always)]
                    |slice| E::simd_splat_unit(simd, slice[j].clone()),
                ),
            );
            let u_single_j = E::simd_neg(
                simd,
                E::map(
                    E::copy(&u),
                    #[inline(always)]
                    |slice| E::simd_splat_unit(simd, slice[j].clone()),
                ),
            );

            let start = j;
            let len = n - start;
            debug_assert!(len > 0);

            let acc_j = E::map(
                E::rb_mut(E::as_mut(&mut acc)),
                #[inline(always)]
                |slice| &mut slice[start..],
            );
            let lhs_j = unsafe {
                E::map(
                    lhs.rb_mut().ptr_at(0, j),
                    #[inline(always)]
                    |ptr| core::slice::from_raw_parts_mut(ptr.wrapping_add(start), len),
                )
            };
            let rhs_j = E::map(
                E::copy(&rhs),
                #[inline(always)]
                |slice| &slice[start..],
            );
            let y_j = E::map(
                E::copy(&y),
                #[inline(always)]
                |slice| &slice[start..],
            );
            let u_j = E::map(
                E::copy(&u),
                #[inline(always)]
                |slice| &slice[start..],
            );

            let prefix = ((len - 1) % lane_count) + 1;

            let (acc_prefix, acc_suffix) = E::unzip(E::map(
                acc_j,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let (lhs_prefix, lhs_suffix) = E::unzip(E::map(
                lhs_j,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let (rhs_prefix, rhs_suffix) = E::unzip(E::map(
                rhs_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));
            let (u_prefix, u_suffix) = E::unzip(E::map(
                u_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));
            let (y_prefix, y_suffix) = E::unzip(E::map(
                y_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));

            let acc_suffix = faer_core::simd::slice_as_mut_simd::<E, S>(acc_suffix).0;
            let lhs_suffix = faer_core::simd::slice_as_mut_simd::<E, S>(lhs_suffix).0;
            let rhs_suffix = faer_core::simd::slice_as_simd::<E, S>(rhs_suffix).0;
            let u_suffix = faer_core::simd::slice_as_simd::<E, S>(u_suffix).0;
            let y_suffix = faer_core::simd::slice_as_simd::<E, S>(y_suffix).0;

            let mut sum0 = E::simd_splat(simd, zero.clone());
            let mut sum1 = E::simd_splat(simd, zero.clone());
            let mut sum2 = E::simd_splat(simd, zero.clone());
            let mut sum3 = E::simd_splat(simd, zero.clone());

            let u_prefix = E::partial_load_last(simd, u_prefix);
            let y_prefix = E::partial_load_last(simd, y_prefix);

            let mut lhs_prefix_ = E::partial_load_last(simd, E::rb(E::as_ref(&lhs_prefix)));

            lhs_prefix_ = E::simd_conj_mul_adde(simd, E::copy(&u_single_j), y_prefix, lhs_prefix_);
            lhs_prefix_ = E::simd_conj_mul_adde(simd, E::copy(&y_single_j), u_prefix, lhs_prefix_);

            E::partial_store_last(simd, lhs_prefix, E::copy(&lhs_prefix_));

            let rhs_prefix = E::partial_load_last(simd, rhs_prefix);
            let acc_prefix = E::map(acc_prefix, |slice| &mut slice[1..]);
            let mut acc_prefix_ = E::partial_load_last(simd, E::rb(E::as_ref(&acc_prefix)));

            acc_prefix_ = E::simd_mul_adde(
                simd,
                E::copy(&lhs_prefix_),
                E::copy(&rhs_single_j),
                acc_prefix_,
            );
            E::partial_store_last(simd, acc_prefix, acc_prefix_);

            sum0 = E::simd_conj_mul_adde(simd, E::copy(&lhs_prefix_), E::copy(&rhs_prefix), sum0);

            let (acc_head, acc_tail) = E::as_arrays_mut::<4, _>(acc_suffix);
            let (lhs_head, lhs_tail) = E::as_arrays_mut::<4, _>(lhs_suffix);
            let (rhs_head, rhs_tail) = E::as_arrays::<4, _>(rhs_suffix);
            let (u_head, u_tail) = E::as_arrays::<4, _>(u_suffix);
            let (y_head, y_tail) = E::as_arrays::<4, _>(y_suffix);

            for (acc, (lhs, (rhs, (u, y)))) in zip(
                E::into_iter(acc_head),
                zip(
                    E::into_iter(lhs_head),
                    zip(
                        E::into_iter(rhs_head),
                        zip(E::into_iter(u_head), E::into_iter(y_head)),
                    ),
                ),
            ) {
                let [mut lhs0, mut lhs1, mut lhs2, mut lhs3] =
                    E::unzip4(E::deref(E::rb(E::as_ref(&lhs))));

                let [u0, u1, u2, u3] = E::unzip4(E::deref(u));
                let [y0, y1, y2, y3] = E::unzip4(E::deref(y));

                lhs0 = E::simd_conj_mul_adde(simd, E::copy(&u_single_j), y0, lhs0);
                lhs0 = E::simd_conj_mul_adde(simd, E::copy(&y_single_j), u0, lhs0);
                lhs1 = E::simd_conj_mul_adde(simd, E::copy(&u_single_j), y1, lhs1);
                lhs1 = E::simd_conj_mul_adde(simd, E::copy(&y_single_j), u1, lhs1);
                lhs2 = E::simd_conj_mul_adde(simd, E::copy(&u_single_j), y2, lhs2);
                lhs2 = E::simd_conj_mul_adde(simd, E::copy(&y_single_j), u2, lhs2);
                lhs3 = E::simd_conj_mul_adde(simd, E::copy(&u_single_j), y3, lhs3);
                lhs3 = E::simd_conj_mul_adde(simd, E::copy(&y_single_j), u3, lhs3);

                E::map(
                    E::zip(
                        lhs,
                        E::zip(
                            E::zip(E::copy(&lhs0), E::copy(&lhs1)),
                            E::zip(E::copy(&lhs2), E::copy(&lhs3)),
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

                let [rhs0, rhs1, rhs2, rhs3] = E::unzip4(E::deref(rhs));
                let [mut acc0, mut acc1, mut acc2, mut acc3] =
                    E::unzip4(E::deref(E::rb(E::as_ref(&acc))));
                acc0 = E::simd_mul_adde(simd, E::copy(&lhs0), E::copy(&rhs_single_j), acc0);
                acc1 = E::simd_mul_adde(simd, E::copy(&lhs1), E::copy(&rhs_single_j), acc1);
                acc2 = E::simd_mul_adde(simd, E::copy(&lhs2), E::copy(&rhs_single_j), acc2);
                acc3 = E::simd_mul_adde(simd, E::copy(&lhs3), E::copy(&rhs_single_j), acc3);
                E::map(
                    E::zip(acc, E::zip(E::zip(acc0, acc1), E::zip(acc2, acc3))),
                    #[inline(always)]
                    |(acc, ((acc0, acc1), (acc2, acc3)))| {
                        acc[0] = acc0;
                        acc[1] = acc1;
                        acc[2] = acc2;
                        acc[3] = acc3;
                    },
                );
                sum0 = E::simd_conj_mul_adde(simd, E::copy(&lhs0), rhs0, sum0);
                sum1 = E::simd_conj_mul_adde(simd, E::copy(&lhs1), rhs1, sum1);
                sum2 = E::simd_conj_mul_adde(simd, E::copy(&lhs2), rhs2, sum2);
                sum3 = E::simd_conj_mul_adde(simd, E::copy(&lhs3), rhs3, sum3);
            }

            sum0 = E::simd_add(simd, sum0, sum1);
            sum2 = E::simd_add(simd, sum2, sum3);

            sum0 = E::simd_add(simd, sum0, sum2);

            for (acc, (lhs, (rhs, (u, y)))) in zip(
                E::into_iter(acc_tail),
                zip(
                    E::into_iter(lhs_tail),
                    zip(
                        E::into_iter(rhs_tail),
                        zip(E::into_iter(u_tail), E::into_iter(y_tail)),
                    ),
                ),
            ) {
                let mut lhs0 = E::deref(E::rb(E::as_ref(&lhs)));
                let u0 = E::deref(u);
                let y0 = E::deref(y);
                lhs0 = E::simd_conj_mul_adde(simd, E::copy(&u_single_j), y0, lhs0);
                lhs0 = E::simd_conj_mul_adde(simd, E::copy(&y_single_j), u0, lhs0);

                E::map(
                    E::zip(lhs, E::copy(&lhs0)),
                    #[inline(always)]
                    |(lhs, lhs0)| {
                        *lhs = lhs0;
                    },
                );

                let rhs0 = E::deref(rhs);
                let mut acc0 = E::deref(E::rb(E::as_ref(&acc)));
                acc0 = E::simd_mul_adde(simd, E::copy(&lhs0), E::copy(&rhs_single_j), acc0);
                E::map(
                    E::zip(acc, acc0),
                    #[inline(always)]
                    |(acc, acc0)| {
                        *acc = acc0;
                    },
                );
                sum0 = E::simd_conj_mul_adde(simd, E::copy(&lhs0), rhs0, sum0);
            }

            let sum = E::simd_reduce_add(simd, sum0);
            let sum = E::into_units(sum);

            E::map(
                E::zip(E::rb_mut(E::as_mut(&mut acct)), sum),
                #[inline(always)]
                |(slice, sum)| slice[j] = sum,
            );
        }

        for j in (first_col + rem..last_col).step_by(2) {
            let sum0_scalar = {
                let u = E::from_units(E::map(
                    E::copy(&u),
                    #[inline(always)]
                    |slice| slice[j].clone(),
                ));
                let y = E::from_units(E::map(
                    E::copy(&y),
                    #[inline(always)]
                    |slice| slice[j].clone(),
                ));
                let rhs = E::from_units(E::map(
                    E::copy(&rhs),
                    #[inline(always)]
                    |slice| slice[j].clone(),
                ));

                let mut ljj = lhs.read(j, j);
                ljj = ljj.sub(&u.mul(&y.conj())).sub(&y.mul(&u.conj()));
                lhs.write(j, j, ljj.clone());

                ljj.conj().mul(&rhs)
            };

            let rhs_single_j0 = E::map(
                E::copy(&rhs),
                #[inline(always)]
                |slice| E::simd_splat_unit(simd, slice[j].clone()),
            );
            let y_single_j0 = E::simd_neg(
                simd,
                E::map(
                    E::copy(&y),
                    #[inline(always)]
                    |slice| E::simd_splat_unit(simd, slice[j].clone()),
                ),
            );
            let u_single_j0 = E::simd_neg(
                simd,
                E::map(
                    E::copy(&u),
                    #[inline(always)]
                    |slice| E::simd_splat_unit(simd, slice[j].clone()),
                ),
            );

            let rhs_single_j1 = E::map(
                E::copy(&rhs),
                #[inline(always)]
                |slice| E::simd_splat_unit(simd, slice[j + 1].clone()),
            );
            let y_single_j1 = E::simd_neg(
                simd,
                E::map(
                    E::copy(&y),
                    #[inline(always)]
                    |slice| E::simd_splat_unit(simd, slice[j + 1].clone()),
                ),
            );
            let u_single_j1 = E::simd_neg(
                simd,
                E::map(
                    E::copy(&u),
                    #[inline(always)]
                    |slice| E::simd_splat_unit(simd, slice[j + 1].clone()),
                ),
            );

            let start = j + 1;
            let len = n - start;
            debug_assert!(len > 0);

            let acc_j = E::map(
                E::rb_mut(E::as_mut(&mut acc)),
                #[inline(always)]
                |slice| &mut slice[start..],
            );
            let lhs_j0 = unsafe {
                E::map(
                    lhs.rb_mut().ptr_at(0, j),
                    #[inline(always)]
                    |ptr| core::slice::from_raw_parts_mut(ptr.wrapping_add(start), len),
                )
            };
            let lhs_j1 = unsafe {
                E::map(
                    lhs.rb_mut().ptr_at(0, j + 1),
                    #[inline(always)]
                    |ptr| core::slice::from_raw_parts_mut(ptr.wrapping_add(start), len),
                )
            };
            let rhs_j = E::map(
                E::copy(&rhs),
                #[inline(always)]
                |slice| &slice[start..],
            );
            let y_j = E::map(
                E::copy(&y),
                #[inline(always)]
                |slice| &slice[start..],
            );
            let u_j = E::map(
                E::copy(&u),
                #[inline(always)]
                |slice| &slice[start..],
            );

            let prefix = ((len - 1) % lane_count) + 1;

            let (mut acc_prefix, acc_suffix) = E::unzip(E::map(
                acc_j,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let (lhs0_prefix, lhs0_suffix) = E::unzip(E::map(
                lhs_j0,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let (lhs1_prefix, lhs1_suffix) = E::unzip(E::map(
                lhs_j1,
                #[inline(always)]
                |slice| slice.split_at_mut(prefix),
            ));
            let (rhs_prefix, rhs_suffix) = E::unzip(E::map(
                rhs_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));
            let (u_prefix, u_suffix) = E::unzip(E::map(
                u_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));
            let (y_prefix, y_suffix) = E::unzip(E::map(
                y_j,
                #[inline(always)]
                |slice| slice.split_at(prefix),
            ));

            let acc_suffix = faer_core::simd::slice_as_mut_simd::<E, S>(acc_suffix).0;
            let lhs0_suffix = faer_core::simd::slice_as_mut_simd::<E, S>(lhs0_suffix).0;
            let lhs1_suffix = faer_core::simd::slice_as_mut_simd::<E, S>(lhs1_suffix).0;
            let rhs_suffix = faer_core::simd::slice_as_simd::<E, S>(rhs_suffix).0;
            let u_suffix = faer_core::simd::slice_as_simd::<E, S>(u_suffix).0;
            let y_suffix = faer_core::simd::slice_as_simd::<E, S>(y_suffix).0;

            let mut sum0 = E::simd_splat(simd, zero.clone());
            let mut sum1 = E::simd_splat(simd, zero.clone());

            let u_prefix = E::partial_load_last(simd, u_prefix);
            let y_prefix = E::partial_load_last(simd, y_prefix);

            let mut lhs0_prefix_ = E::partial_load_last(simd, E::rb(E::as_ref(&lhs0_prefix)));
            let mut lhs1_prefix_ = E::partial_load_last(simd, E::rb(E::as_ref(&lhs1_prefix)));

            lhs0_prefix_ = E::simd_conj_mul_adde(
                simd,
                E::copy(&u_single_j0),
                E::copy(&y_prefix),
                lhs0_prefix_,
            );
            lhs0_prefix_ = E::simd_conj_mul_adde(
                simd,
                E::copy(&y_single_j0),
                E::copy(&u_prefix),
                lhs0_prefix_,
            );

            lhs1_prefix_ = E::simd_conj_mul_adde(
                simd,
                E::copy(&y_single_j1),
                E::copy(&u_prefix),
                lhs1_prefix_,
            );
            lhs1_prefix_ = E::simd_conj_mul_adde(
                simd,
                E::copy(&u_single_j1),
                E::copy(&y_prefix),
                lhs1_prefix_,
            );

            E::partial_store_last(simd, lhs0_prefix, E::copy(&lhs0_prefix_));
            E::partial_store_last(simd, lhs1_prefix, E::copy(&lhs1_prefix_));

            let rhs_prefix = E::partial_load_last(simd, rhs_prefix);
            {
                let acc_prefix = E::rb_mut(E::as_mut(&mut acc_prefix));
                let mut acc_prefix_ = E::partial_load_last(simd, E::rb(E::as_ref(&acc_prefix)));

                acc_prefix_ = E::simd_mul_adde(
                    simd,
                    E::copy(&lhs0_prefix_),
                    E::copy(&rhs_single_j0),
                    acc_prefix_,
                );
                E::partial_store_last(simd, acc_prefix, acc_prefix_);
            }
            {
                let acc_prefix = E::map(acc_prefix, |slice| &mut slice[1..]);
                let mut acc_prefix_ = E::partial_load_last(simd, E::rb(E::as_ref(&acc_prefix)));

                acc_prefix_ = E::simd_mul_adde(
                    simd,
                    E::copy(&lhs1_prefix_),
                    E::copy(&rhs_single_j1),
                    acc_prefix_,
                );
                E::partial_store_last(simd, acc_prefix, acc_prefix_);
            }

            sum0 = E::simd_conj_mul_adde(simd, E::copy(&lhs0_prefix_), E::copy(&rhs_prefix), sum0);
            sum1 = E::simd_conj_mul_adde(simd, E::copy(&lhs1_prefix_), E::copy(&rhs_prefix), sum1);

            for (acc, ((lhs_j0, lhs_j1), (rhs, (u, y)))) in zip(
                E::into_iter(acc_suffix),
                zip(
                    zip(E::into_iter(lhs0_suffix), E::into_iter(lhs1_suffix)),
                    zip(
                        E::into_iter(rhs_suffix),
                        zip(E::into_iter(u_suffix), E::into_iter(y_suffix)),
                    ),
                ),
            ) {
                let mut lhs0 = E::deref(E::rb(E::as_ref(&lhs_j0)));
                let mut lhs1 = E::deref(E::rb(E::as_ref(&lhs_j1)));
                let u = E::deref(u);
                let y = E::deref(y);
                lhs0 = E::simd_conj_mul_adde(simd, E::copy(&u_single_j0), E::copy(&y), lhs0);
                lhs0 = E::simd_conj_mul_adde(simd, E::copy(&y_single_j0), E::copy(&u), lhs0);
                lhs1 = E::simd_conj_mul_adde(simd, E::copy(&u_single_j1), E::copy(&y), lhs1);
                lhs1 = E::simd_conj_mul_adde(simd, E::copy(&y_single_j1), E::copy(&u), lhs1);

                E::map(
                    E::zip(lhs_j0, E::copy(&lhs0)),
                    #[inline(always)]
                    |(lhs, lhs0)| {
                        *lhs = lhs0;
                    },
                );
                E::map(
                    E::zip(lhs_j1, E::copy(&lhs1)),
                    #[inline(always)]
                    |(lhs, lhs1)| {
                        *lhs = lhs1;
                    },
                );

                let rhs = E::deref(rhs);
                let mut acc_ = E::deref(E::rb(E::as_ref(&acc)));
                acc_ = E::simd_mul_adde(simd, E::copy(&lhs0), E::copy(&rhs_single_j0), acc_);
                acc_ = E::simd_mul_adde(simd, E::copy(&lhs1), E::copy(&rhs_single_j1), acc_);
                E::map(
                    E::zip(acc, acc_),
                    #[inline(always)]
                    |(acc, acc_)| {
                        *acc = acc_;
                    },
                );
                sum0 = E::simd_conj_mul_adde(simd, E::copy(&lhs0), E::copy(&rhs), sum0);
                sum1 = E::simd_conj_mul_adde(simd, E::copy(&lhs1), E::copy(&rhs), sum1);
            }

            {
                let sum = E::simd_reduce_add(simd, sum0).add(&sum0_scalar);
                let sum = E::into_units(sum);

                E::map(
                    E::zip(E::rb_mut(E::as_mut(&mut acct)), sum),
                    #[inline(always)]
                    |(slice, sum)| slice[j] = sum,
                );
            }
            {
                let sum = E::simd_reduce_add(simd, sum1);
                let sum = E::into_units(sum);

                E::map(
                    E::zip(E::rb_mut(E::as_mut(&mut acct)), sum),
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

        let lane_count = core::mem::size_of::<E::SimdUnit<S>>() / core::mem::size_of::<E::Unit>();

        let mut acc =
            unsafe { E::map(acc.as_ptr(), |ptr| core::slice::from_raw_parts_mut(ptr, n)) };
        let rhs = unsafe { E::map(rhs.as_ptr(), |ptr| core::slice::from_raw_parts(ptr, n)) };

        let zero = E::zero();
        for j in first_col..last_col {
            let start = j + 1;
            let len = n - start;

            let acc_j = E::map(
                E::rb_mut(E::as_mut(&mut acc)),
                #[inline(always)]
                |slice| &mut slice[start..],
            );
            let lhs_j = unsafe {
                E::map(
                    lhs.ptr_at(start, j),
                    #[inline(always)]
                    |ptr| core::slice::from_raw_parts(ptr, len),
                )
            };
            let rhs_j = E::map(
                E::copy(&rhs),
                #[inline(always)]
                |slice| &slice[start..],
            );

            let (acc_prefix, acc_suffix) = E::unzip(E::map(
                acc_j,
                #[inline(always)]
                |slice| slice.split_at_mut(len % lane_count),
            ));
            let (lhs_prefix, lhs_suffix) = E::unzip(E::map(
                lhs_j,
                #[inline(always)]
                |slice| slice.split_at(len % lane_count),
            ));
            let (rhs_prefix, rhs_suffix) = E::unzip(E::map(
                rhs_j,
                #[inline(always)]
                |slice| slice.split_at(len % lane_count),
            ));

            let acc_suffix = faer_core::simd::slice_as_mut_simd::<E, S>(acc_suffix).0;
            let lhs_suffix = faer_core::simd::slice_as_simd::<E, S>(lhs_suffix).0;
            let rhs_suffix = faer_core::simd::slice_as_simd::<E, S>(rhs_suffix).0;

            let rhs_single_j = E::map(
                E::copy(&rhs),
                #[inline(always)]
                |slice| E::simd_splat_unit(simd, slice[j].clone()),
            );

            let mut sum0 = E::simd_splat(simd, zero.clone());
            let mut sum1 = E::simd_splat(simd, zero.clone());
            let mut sum2 = E::simd_splat(simd, zero.clone());
            let mut sum3 = E::simd_splat(simd, zero.clone());

            let lhs_prefix = E::partial_load_last(simd, lhs_prefix);
            let rhs_prefix = E::partial_load_last(simd, rhs_prefix);
            let mut acc_prefix_ = E::partial_load_last(simd, E::rb(E::as_ref(&acc_prefix)));

            acc_prefix_ = E::simd_mul_adde(
                simd,
                E::copy(&lhs_prefix),
                E::copy(&rhs_single_j),
                acc_prefix_,
            );
            E::partial_store_last(simd, acc_prefix, acc_prefix_);

            sum0 = E::simd_conj_mul_adde(simd, E::copy(&lhs_prefix), rhs_prefix, sum0);

            let (acc_head, acc_tail) = E::as_arrays_mut::<4, _>(acc_suffix);
            let (lhs_head, lhs_tail) = E::as_arrays::<4, _>(lhs_suffix);
            let (rhs_head, rhs_tail) = E::as_arrays::<4, _>(rhs_suffix);

            for (acc, (lhs, rhs)) in zip(
                E::into_iter(acc_head),
                zip(E::into_iter(lhs_head), E::into_iter(rhs_head)),
            ) {
                let [lhs0, lhs1, lhs2, lhs3] = E::unzip4(E::deref(lhs));
                let [rhs0, rhs1, rhs2, rhs3] = E::unzip4(E::deref(rhs));
                let [mut acc0, mut acc1, mut acc2, mut acc3] =
                    E::unzip4(E::deref(E::rb(E::as_ref(&acc))));
                acc0 = E::simd_mul_adde(simd, E::copy(&lhs0), E::copy(&rhs_single_j), acc0);
                acc1 = E::simd_mul_adde(simd, E::copy(&lhs1), E::copy(&rhs_single_j), acc1);
                acc2 = E::simd_mul_adde(simd, E::copy(&lhs2), E::copy(&rhs_single_j), acc2);
                acc3 = E::simd_mul_adde(simd, E::copy(&lhs3), E::copy(&rhs_single_j), acc3);
                E::map(
                    E::zip(acc, E::zip(E::zip(acc0, acc1), E::zip(acc2, acc3))),
                    #[inline(always)]
                    |(acc, ((acc0, acc1), (acc2, acc3)))| {
                        acc[0] = acc0;
                        acc[1] = acc1;
                        acc[2] = acc2;
                        acc[3] = acc3;
                    },
                );
                sum0 = E::simd_conj_mul_adde(simd, E::copy(&lhs0), rhs0, sum0);
                sum1 = E::simd_conj_mul_adde(simd, E::copy(&lhs1), rhs1, sum1);
                sum2 = E::simd_conj_mul_adde(simd, E::copy(&lhs2), rhs2, sum2);
                sum3 = E::simd_conj_mul_adde(simd, E::copy(&lhs3), rhs3, sum3);
            }

            sum0 = E::simd_add(simd, sum0, sum1);
            sum2 = E::simd_add(simd, sum2, sum3);

            sum0 = E::simd_add(simd, sum0, sum2);

            for (acc, (lhs, rhs)) in zip(
                E::into_iter(acc_tail),
                zip(E::into_iter(lhs_tail), E::into_iter(rhs_tail)),
            ) {
                let lhs0 = E::deref(lhs);
                let rhs0 = E::deref(rhs);
                let mut acc0 = E::deref(E::rb(E::as_ref(&acc)));
                acc0 = E::simd_mul_adde(simd, E::copy(&lhs0), E::copy(&rhs_single_j), acc0);
                E::map(
                    E::zip(acc, acc0),
                    #[inline(always)]
                    |(acc, acc0)| {
                        *acc = acc0;
                    },
                );
                sum0 = E::simd_conj_mul_adde(simd, E::copy(&lhs0), rhs0, sum0);
            }

            let mut sum = E::simd_reduce_add(simd, sum0);
            let acc_ = E::from_units(E::map(
                E::rb(E::as_ref(&acc)),
                #[inline(always)]
                |slice| slice[j].clone(),
            ));

            sum = sum.add(&acc_);
            let sum = E::into_units(sum);

            E::map(
                E::zip(E::rb_mut(E::as_mut(&mut acc)), sum),
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
    stack: DynStack<'_>,
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

    let arch = pulp::Arch::new();
    for k in 0..n - 1 {
        let a_cur = a.rb_mut().submatrix(k, k, n - k, n - k);
        let [mut a11, _, mut a21, mut a22] = a_cur.split_at(1, 1);

        let parallelism = if n - k <= 256 {
            Parallelism::None
        } else {
            parallelism
        };

        let [_, u] = u.rb_mut().split_at_row(k);
        let [nu, mut u21] = u.split_at_row(1);
        let [_, y] = y.rb_mut().split_at_row(k);
        let [psi, mut y21] = y.split_at_row(1);

        let [_, v] = v.rb_mut().split_at_row(k);
        let [_, v21] = v.split_at_row(1);
        let mut v21 = v21.subcols(0, parallelism_degree(parallelism));

        let [_, w] = w.rb_mut().split_at_row(k);
        let [_, w21] = w.split_at_row(1);
        let w21 = w21.subcols(0, parallelism_degree(parallelism));

        if k > 0 {
            let nu = nu.read(0, 0);
            let psi = psi.read(0, 0);
            a11.write(
                0,
                0,
                a11.read(0, 0)
                    .sub(&nu.mul(&psi.conj()).add(&psi.mul(&nu.conj()))),
            );

            zipped!(a21.rb_mut(), u21.rb(), y21.rb()).for_each(|mut a, u, y| {
                let u = u.read();
                let y = y.read();
                a.write(a.read().sub(&u.mul(&psi.conj()).add(&y.mul(&nu.conj()))))
            });
        }

        let (tau, new_head) = {
            let [head, tail] = a21.rb_mut().split_at_row(1);
            let norm2 = norm2(tail.rb());
            faer_core::householder::make_householder_in_place(Some(tail), head.read(0, 0), norm2)
        };
        a21.write(0, 0, E::one());
        let tau_inv = tau.inv();
        householder.write(k, 0, tau);

        if k > 0 {
            if E::HAS_SIMD {
                let ncols = (n - k - 1) as f64;
                let n_threads = parallelism_degree(parallelism) as f64;

                assert!(ncols < 2.0f64.powi(50)); // to check that integers can be
                                                  // represented exactly as floats

                let idx_to_col_start = |idx: usize| {
                    let idx_as_percent = idx as f64 / n_threads;
                    let col_start_percent = 1.0 - (1.0 - idx_as_percent).sqrt();
                    (col_start_percent * ncols) as usize
                };

                faer_core::for_each_raw(
                    parallelism_degree(parallelism),
                    |idx| {
                        let first_col = idx_to_col_start(idx);
                        let last_col = idx_to_col_start(idx + 1);

                        let mut v21 = unsafe { v21.rb().col(idx).const_cast() };
                        let mut w21 = unsafe { w21.rb().col(idx).const_cast() };

                        zipped!(v21.rb_mut()).for_each(|mut z| z.write(E::zero()));
                        zipped!(w21.rb_mut()).for_each(|mut z| z.write(E::zero()));

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

                zipped!(y21.rb_mut(), v21.rb().col(0), w21.rb().col(0))
                    .for_each(|mut y, v, w| y.write(v.read().add(&w.read())));
                for i in 1..n_threads as usize {
                    zipped!(y21.rb_mut(), v21.rb().col(i), w21.rb().col(i))
                        .for_each(|mut y, v, w| y.write(y.read().add(&v.read().add(&w.read()))));
                }
            } else {
                triangular::matmul(
                    a22.rb_mut(),
                    BlockStructure::TriangularLower,
                    u21.rb(),
                    BlockStructure::Rectangular,
                    y21.rb().adjoint(),
                    BlockStructure::Rectangular,
                    Some(E::one()),
                    E::one().neg(),
                    parallelism,
                );
                triangular::matmul(
                    a22.rb_mut(),
                    BlockStructure::TriangularLower,
                    y21.rb(),
                    BlockStructure::Rectangular,
                    u21.rb().adjoint(),
                    BlockStructure::Rectangular,
                    Some(E::one()),
                    E::one().neg(),
                    parallelism,
                );

                zipped!(y21.rb_mut(), a22.rb().diagonal(), a21.rb())
                    .for_each(|mut y, a, u| y.write(a.read().mul(&u.read())));

                triangular::matmul(
                    y21.rb_mut(),
                    BlockStructure::Rectangular,
                    a22.rb(),
                    BlockStructure::StrictTriangularLower,
                    a21.rb(),
                    BlockStructure::Rectangular,
                    Some(E::one()),
                    E::one(),
                    parallelism,
                );
                triangular::matmul(
                    y21.rb_mut(),
                    BlockStructure::Rectangular,
                    a22.rb().adjoint(),
                    BlockStructure::StrictTriangularUpper,
                    a21.rb(),
                    BlockStructure::Rectangular,
                    Some(E::one()),
                    E::one(),
                    parallelism,
                );
            }
        } else if E::HAS_SIMD {
            let mut acc = v21.rb_mut().col(0);
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

            zipped!(y21.rb_mut(), acc.rb(), a22.rb().diagonal(), a21.rb())
                .for_each(|mut y, v, a, u| y.write(v.read().add(&a.read().mul(&u.read()))));
        } else {
            zipped!(y21.rb_mut(), a22.rb().diagonal(), a21.rb())
                .for_each(|mut y, a, u| y.write(a.read().mul(&u.read())));
            triangular::matmul(
                y21.rb_mut(),
                BlockStructure::Rectangular,
                a22.rb(),
                BlockStructure::StrictTriangularLower,
                a21.rb(),
                BlockStructure::Rectangular,
                Some(E::one()),
                E::one(),
                parallelism,
            );
            triangular::matmul(
                y21.rb_mut(),
                BlockStructure::Rectangular,
                a22.rb().adjoint(),
                BlockStructure::StrictTriangularUpper,
                a21.rb(),
                BlockStructure::Rectangular,
                Some(E::one()),
                E::one(),
                parallelism,
            );
        }

        zipped!(u21.rb_mut(), a21.rb()).for_each(|mut dst, src| dst.write(src.read()));
        a21.write(0, 0, new_head);

        let beta = inner_prod_with_conj(u21.rb(), Conj::Yes, y21.rb(), Conj::No)
            .scale_power_of_two(&E::Real::from_f64(0.5));

        zipped!(y21.rb_mut(), u21.rb()).for_each(|mut y, u| {
            let u = u.read();
            y.write(y.read().sub(&beta.mul(&u.mul(&tau_inv))).mul(&tau_inv));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;
    use assert_approx_eq::assert_approx_eq;
    use faer_core::{
        c64,
        householder::{
            apply_block_householder_sequence_on_the_right_in_place_req,
            apply_block_householder_sequence_on_the_right_in_place_with_conj,
            apply_block_householder_sequence_transpose_on_the_left_in_place_req,
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
        },
        Mat,
    };

    macro_rules! make_stack {
        ($req: expr $(,)?) => {
            ::dyn_stack::DynStack::new(&mut ::dyn_stack::GlobalMemBuffer::new($req.unwrap()))
        };
    }

    #[test]
    fn tridiag_f64() {
        for (n, parallelism) in [(10, Parallelism::None), (512, Parallelism::Rayon(4))] {
            let mut mat = Mat::from_fn(n, n, |_, _| rand::random::<f64>());
            let transpose = mat.transpose().to_owned();

            zipped!(mat.as_mut(), transpose.as_ref())
                .for_each(|mut x, y| x.write(x.read() + y.read()));

            let n = mat.ncols();

            let mut trid = mat.clone();
            let mut tau_left = Mat::zeros(n - 1, 1);

            tridiagonalize_in_place(
                trid.as_mut(),
                tau_left.as_mut().col(0),
                parallelism,
                make_stack!(tridiagonalize_in_place_req::<f64>(n, parallelism)),
            );

            let mut copy = mat.clone();
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                trid.as_ref().submatrix(1, 0, n - 1, n - 1),
                tau_left.as_ref().transpose(),
                Conj::No,
                copy.as_mut().submatrix(1, 0, n - 1, n),
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
                copy.as_mut().submatrix(0, 1, n, n - 1),
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
                .for_each(|mut x, y| x.write(x.read() + y.read()));

            let n = mat.ncols();

            let mut trid = mat.clone();
            let mut tau_left = Mat::zeros(n - 1, 1);

            tridiagonalize_in_place(
                trid.as_mut(),
                tau_left.as_mut().col(0),
                parallelism,
                make_stack!(tridiagonalize_in_place_req::<c64>(n, parallelism)),
            );

            let mut copy = mat.clone();
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                trid.as_ref().submatrix(1, 0, n - 1, n - 1),
                tau_left.as_ref().transpose(),
                Conj::Yes,
                copy.as_mut().submatrix(1, 0, n - 1, n),
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
                copy.as_mut().submatrix(0, 1, n, n - 1),
                Parallelism::None,
                make_stack!(
                    apply_block_householder_sequence_on_the_right_in_place_req::<c64>(n - 1, 1, n,)
                ),
            );

            for j in 0..n {
                for i in (0..j.saturating_sub(1)).chain(j + 2..n) {
                    assert_approx_eq!(copy.read(i, j), c64::zero());
                }
            }
        }
    }
}
