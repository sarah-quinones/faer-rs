use crate::{
    linalg::entity::{pulp, SimdCtx, SimdGroupFor, SimdIndexFor},
    prelude::*,
    utils::{
        simd::SimdFor,
        slice::{RefGroup, SliceGroup, SliceGroupMut},
    },
    ComplexField, RealField,
};
use coe::Coerce;
use core::iter::zip;
use equator::assert;
use num_complex::Complex;
use pulp::{Read, Write};
use reborrow::*;

// FIXME: reduce_add determinism

/// Specifies how missing values should be handled in mean and variance computations.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NanHandling {
    /// NaNs are passed as-is to arithmetic operators.
    Propagate,
    /// NaNs are skipped, and they're not included in the total count of entries.
    Ignore,
}

#[inline(always)]
fn from_usize<E: RealField>(n: usize) -> E {
    E::faer_from_f64(n as u32 as f64)
        .faer_add(E::faer_from_f64((n as u64 - (n as u32 as u64)) as f64))
}

#[inline(always)]
fn reduce<E: RealField, S: pulp::Simd>(non_nan_count: SimdIndexFor<E, S>) -> usize {
    let slice: &[E::Index] = bytemuck::cast_slice(core::slice::from_ref(&non_nan_count));

    let mut acc = 0usize;
    for &count in slice {
        acc += E::faer_index_to_usize(count);
    }
    acc
}

fn col_mean_row_major_ignore_nan_real<E: RealField>(out: ColMut<'_, E>, mat: MatRef<'_, E>) {
    struct Impl<'a, E: RealField> {
        out: ColMut<'a, E>,
        mat: MatRef<'a, E>,
    }

    impl<E: RealField> pulp::WithSimd for Impl<'_, E> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { mut out, mat } = self;
            let simd = SimdFor::<E, S>::new(simd);

            let m = mat.nrows();
            let chunk_size = if core::mem::size_of::<E::Index>() < core::mem::size_of::<usize>() {
                1usize << (core::mem::size_of::<E::Index>() * 8)
            } else {
                usize::MAX
            } / 4;

            let offset = simd.align_offset_ptr(mat.as_ptr(), mat.ncols());
            for i in 0..m {
                let row = SliceGroup::<'_, E>::new(mat.row(i).try_as_slice().unwrap());
                let (head, body, tail) = simd.as_aligned_simd(row, offset);

                let mut non_nan_count_total = 0usize;

                #[inline(always)]
                fn process<E: RealField, S: pulp::Simd>(
                    simd: SimdFor<E, S>,
                    acc: SimdGroupFor<E, S>,
                    non_nan_count: SimdIndexFor<E, S>,
                    val: impl Read<Output = SimdGroupFor<E, S>>,
                ) -> (SimdGroupFor<E, S>, SimdIndexFor<E, S>) {
                    let val = val.read_or(simd.splat(E::faer_nan()));
                    let is_not_nan = simd.less_than_or_equal(val, val);

                    (
                        simd.select(is_not_nan, simd.add(acc, val), acc),
                        simd.index_select(
                            is_not_nan,
                            simd.index_add(
                                non_nan_count,
                                simd.index_splat(E::faer_usize_to_index(1)),
                            ),
                            non_nan_count,
                        ),
                    )
                }

                let mut sum0 = simd.splat(E::faer_zero());
                let mut sum1 = simd.splat(E::faer_zero());
                let mut sum2 = simd.splat(E::faer_zero());
                let mut sum3 = simd.splat(E::faer_zero());
                let mut non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count1 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count2 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count3 = simd.index_splat(E::faer_usize_to_index(0));

                (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, head);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);
                non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));

                let (body4, body1) = body.as_arrays::<4>();

                let mut start = 0usize;
                while start < body4.len() {
                    let len = Ord::min(body4.len() - start, chunk_size);

                    for [x0, x1, x2, x3] in body4
                        .subslice(start..start + len)
                        .into_ref_iter()
                        .map(RefGroup::unzip)
                    {
                        (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, x0);
                        (sum1, non_nan_count1) = process(simd, sum1, non_nan_count1, x1);
                        (sum2, non_nan_count2) = process(simd, sum2, non_nan_count2, x2);
                        (sum3, non_nan_count3) = process(simd, sum3, non_nan_count3, x3);
                    }
                    non_nan_count0 = simd.index_add(non_nan_count0, non_nan_count1);
                    non_nan_count2 = simd.index_add(non_nan_count2, non_nan_count3);
                    non_nan_count0 = simd.index_add(non_nan_count0, non_nan_count2);
                    non_nan_count_total += reduce::<E, S>(non_nan_count0);
                    non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count1 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count2 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count3 = simd.index_splat(E::faer_usize_to_index(0));

                    start += len;
                }

                for x0 in body1.into_ref_iter() {
                    (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, x0);
                }

                (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, tail);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);

                sum0 = simd.add(sum0, sum1);
                sum2 = simd.add(sum2, sum3);
                sum0 = simd.add(sum0, sum2);

                sum0 = simd.rotate_left(sum0, offset.rotate_left_amount());
                let sum = simd.reduce_add(sum0);

                out.write(
                    i,
                    sum.faer_scale_real(from_usize::<E>(non_nan_count_total).faer_inv()),
                );
            }
        }
    }

    E::Simd::default().dispatch(Impl { out, mat });
}

fn col_varm_row_major_ignore_nan_real<E: RealField>(
    out: ColMut<'_, E>,
    mat: MatRef<'_, E>,
    col_mean: ColRef<'_, E>,
) {
    struct Impl<'a, E: RealField> {
        out: ColMut<'a, E>,
        mat: MatRef<'a, E>,
        col_mean: ColRef<'a, E>,
    }

    impl<E: RealField> pulp::WithSimd for Impl<'_, E> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self {
                mut out,
                mat,
                col_mean,
            } = self;
            let simd = SimdFor::<E, S>::new(simd);

            let m = mat.nrows();
            let chunk_size = if core::mem::size_of::<E::Index>() < core::mem::size_of::<usize>() {
                1usize << (core::mem::size_of::<E::Index>() * 8)
            } else {
                usize::MAX
            } / 4;

            let offset = simd.align_offset_ptr(mat.as_ptr(), mat.ncols());
            for i in 0..m {
                let mean = simd.splat(col_mean.read(i));
                let row = SliceGroup::<'_, E>::new(mat.row(i).try_as_slice().unwrap());
                let (head, body, tail) = simd.as_aligned_simd(row, offset);

                let mut non_nan_count_total = 0usize;

                #[inline(always)]
                fn process<E: RealField, S: pulp::Simd>(
                    simd: SimdFor<E, S>,
                    acc: SimdGroupFor<E, S>,
                    mean: SimdGroupFor<E, S>,
                    non_nan_count: SimdIndexFor<E, S>,
                    val: impl Read<Output = SimdGroupFor<E, S>>,
                ) -> (SimdGroupFor<E, S>, SimdIndexFor<E, S>) {
                    let val = val.read_or(simd.splat(E::faer_nan()));
                    let is_not_nan = simd.less_than_or_equal(val, val);
                    let diff = simd.sub(val, mean);

                    (
                        simd.select(is_not_nan, simd.mul_add_e(diff, diff, acc), acc),
                        simd.index_select(
                            is_not_nan,
                            simd.index_add(
                                non_nan_count,
                                simd.index_splat(E::faer_usize_to_index(1)),
                            ),
                            non_nan_count,
                        ),
                    )
                }

                let mut sum0 = simd.splat(E::faer_zero());
                let mut sum1 = simd.splat(E::faer_zero());
                let mut sum2 = simd.splat(E::faer_zero());
                let mut sum3 = simd.splat(E::faer_zero());
                let mut non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count1 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count2 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count3 = simd.index_splat(E::faer_usize_to_index(0));

                (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, head);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);
                non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));

                let (body4, body1) = body.as_arrays::<4>();

                let mut start = 0usize;
                while start < body4.len() {
                    let len = Ord::min(body4.len() - start, chunk_size);

                    for [x0, x1, x2, x3] in body4
                        .subslice(start..start + len)
                        .into_ref_iter()
                        .map(RefGroup::unzip)
                    {
                        (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, x0);
                        (sum1, non_nan_count1) = process(simd, sum1, mean, non_nan_count1, x1);
                        (sum2, non_nan_count2) = process(simd, sum2, mean, non_nan_count2, x2);
                        (sum3, non_nan_count3) = process(simd, sum3, mean, non_nan_count3, x3);
                    }
                    non_nan_count0 = simd.index_add(non_nan_count0, non_nan_count1);
                    non_nan_count2 = simd.index_add(non_nan_count2, non_nan_count3);
                    non_nan_count0 = simd.index_add(non_nan_count0, non_nan_count2);
                    non_nan_count_total += reduce::<E, S>(non_nan_count0);
                    non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count1 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count2 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count3 = simd.index_splat(E::faer_usize_to_index(0));

                    start += len;
                }

                for x0 in body1.into_ref_iter() {
                    (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, x0);
                }

                (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, tail);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);

                sum0 = simd.add(sum0, sum1);
                sum2 = simd.add(sum2, sum3);
                sum0 = simd.add(sum0, sum2);

                sum0 = simd.rotate_left(sum0, offset.rotate_left_amount());
                let sum = simd.reduce_add(sum0);

                let var = if non_nan_count_total == 0 {
                    E::faer_nan()
                } else if non_nan_count_total == 1 {
                    E::faer_zero()
                } else {
                    sum.faer_scale_real(from_usize::<E>(non_nan_count_total - 1).faer_inv())
                };

                out.write(i, var);
            }
        }
    }

    E::Simd::default().dispatch(Impl { out, mat, col_mean });
}

fn col_mean_row_major_ignore_nan_cplx<E: RealField>(
    out: ColMut<'_, Complex<E>>,
    mat: MatRef<'_, Complex<E>>,
) {
    struct Impl<'a, E: RealField> {
        out: ColMut<'a, Complex<E>>,
        mat: MatRef<'a, Complex<E>>,
    }

    impl<E: RealField> pulp::WithSimd for Impl<'_, E> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { mut out, mat } = self;
            let simd_cplx = SimdFor::<Complex<E>, S>::new(simd);
            let simd = SimdFor::<E, S>::new(simd);

            let m = mat.nrows();
            let chunk_size = if core::mem::size_of::<E::Index>() < core::mem::size_of::<usize>() {
                1usize << (core::mem::size_of::<E::Index>() * 8)
            } else {
                usize::MAX
            } / 4;

            let offset = simd_cplx.align_offset_ptr(mat.as_ptr(), mat.ncols());
            for i in 0..m {
                let row = SliceGroup::<'_, Complex<E>>::new(mat.row(i).try_as_slice().unwrap());
                let (head, body, tail) = simd_cplx.as_aligned_simd(row, offset);

                let mut non_nan_count_total = 0usize;

                #[inline(always)]
                fn process<E: RealField, S: pulp::Simd>(
                    simd: SimdFor<E, S>,
                    acc: SimdGroupFor<Complex<E>, S>,
                    non_nan_count: SimdIndexFor<E, S>,
                    val: impl Read<Output = SimdGroupFor<Complex<E>, S>>,
                ) -> (SimdGroupFor<Complex<E>, S>, SimdIndexFor<E, S>) {
                    let simd_cplx = SimdFor::<Complex<E>, S>::new(simd.simd);

                    let val = val.read_or(simd_cplx.splat(Complex::<E>::faer_nan()));
                    let val_re = val.re;
                    let val_im = val.im;
                    let re_is_not_nan = simd.less_than_or_equal(val.re, val.re);
                    let im_is_not_nan = simd.less_than_or_equal(val.im, val.im);

                    (
                        Complex {
                            re: simd.select(
                                im_is_not_nan,
                                simd.select(re_is_not_nan, simd.add(acc.re, val_re), acc.re),
                                acc.re,
                            ),
                            im: simd.select(
                                im_is_not_nan,
                                simd.select(re_is_not_nan, simd.add(acc.im, val_im), acc.im),
                                acc.im,
                            ),
                        },
                        simd.index_select(
                            im_is_not_nan,
                            simd.index_select(
                                re_is_not_nan,
                                simd.index_add(
                                    non_nan_count,
                                    simd.index_splat(E::faer_usize_to_index(1)),
                                ),
                                non_nan_count,
                            ),
                            non_nan_count,
                        ),
                    )
                }

                let mut sum0 = simd_cplx.splat(Complex::<E>::faer_zero());
                let mut sum1 = simd_cplx.splat(Complex::<E>::faer_zero());
                let mut sum2 = simd_cplx.splat(Complex::<E>::faer_zero());
                let mut sum3 = simd_cplx.splat(Complex::<E>::faer_zero());
                let mut non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count1 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count2 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count3 = simd.index_splat(E::faer_usize_to_index(0));

                (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, head);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);
                non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));

                let (body4, body1) = body.as_arrays::<4>();

                let mut start = 0usize;
                while start < body4.len() {
                    let len = Ord::min(body4.len() - start, chunk_size);

                    for [x0, x1, x2, x3] in body4
                        .subslice(start..start + len)
                        .into_ref_iter()
                        .map(RefGroup::unzip)
                    {
                        (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, x0);
                        (sum1, non_nan_count1) = process(simd, sum1, non_nan_count1, x1);
                        (sum2, non_nan_count2) = process(simd, sum2, non_nan_count2, x2);
                        (sum3, non_nan_count3) = process(simd, sum3, non_nan_count3, x3);
                    }
                    non_nan_count0 = simd.index_add(non_nan_count0, non_nan_count1);
                    non_nan_count2 = simd.index_add(non_nan_count2, non_nan_count3);
                    non_nan_count0 = simd.index_add(non_nan_count0, non_nan_count2);
                    non_nan_count_total += reduce::<E, S>(non_nan_count0);
                    non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count1 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count2 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count3 = simd.index_splat(E::faer_usize_to_index(0));

                    start += len;
                }

                for x0 in body1.into_ref_iter() {
                    (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, x0);
                }

                (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, tail);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);

                sum0 = simd_cplx.add(sum0, sum1);
                sum2 = simd_cplx.add(sum2, sum3);
                sum0 = simd_cplx.add(sum0, sum2);

                sum0 = simd_cplx.rotate_left(sum0, offset.rotate_left_amount());
                let sum = simd_cplx.reduce_add(sum0);

                out.write(
                    i,
                    sum.faer_scale_real(from_usize::<E>(non_nan_count_total).faer_inv()),
                );
            }
        }
    }

    E::Simd::default().dispatch(Impl { out, mat });
}

fn col_varm_row_major_ignore_nan_cplx<E: RealField>(
    out: ColMut<'_, E>,
    mat: MatRef<'_, Complex<E>>,
    col_mean: ColRef<'_, Complex<E>>,
) {
    struct Impl<'a, E: RealField> {
        out: ColMut<'a, E>,
        mat: MatRef<'a, Complex<E>>,
        col_mean: ColRef<'a, Complex<E>>,
    }

    impl<E: RealField> pulp::WithSimd for Impl<'_, E> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self {
                mut out,
                mat,
                col_mean,
            } = self;
            let simd_cplx = SimdFor::<Complex<E>, S>::new(simd);
            let simd = SimdFor::<E, S>::new(simd);

            let m = mat.nrows();
            let chunk_size = if core::mem::size_of::<E::Index>() < core::mem::size_of::<usize>() {
                1usize << (core::mem::size_of::<E::Index>() * 8)
            } else {
                usize::MAX
            } / 4;

            let offset = simd_cplx.align_offset_ptr(mat.as_ptr(), mat.ncols());
            for i in 0..m {
                let mean = simd_cplx.splat(col_mean.read(i));
                let row = SliceGroup::<'_, Complex<E>>::new(mat.row(i).try_as_slice().unwrap());
                let (head, body, tail) = simd_cplx.as_aligned_simd(row, offset);

                let mut non_nan_count_total = 0usize;

                #[inline(always)]
                fn process<E: RealField, S: pulp::Simd>(
                    simd: SimdFor<E, S>,
                    acc: SimdGroupFor<E, S>,
                    mean: SimdGroupFor<Complex<E>, S>,
                    non_nan_count: SimdIndexFor<E, S>,
                    val: impl Read<Output = SimdGroupFor<Complex<E>, S>>,
                ) -> (SimdGroupFor<E, S>, SimdIndexFor<E, S>) {
                    let simd_cplx = SimdFor::<Complex<E>, S>::new(simd.simd);

                    let val = val.read_or(simd_cplx.splat(Complex::<E>::faer_nan()));
                    let val_re = val.re;
                    let val_im = val.im;
                    let re_is_not_nan = simd.less_than_or_equal(val.re, val.re);
                    let im_is_not_nan = simd.less_than_or_equal(val.im, val.im);

                    let diff = simd_cplx.sub(
                        Complex {
                            re: val_re,
                            im: val_im,
                        },
                        mean,
                    );

                    (
                        simd.select(
                            im_is_not_nan,
                            simd.select(re_is_not_nan, simd_cplx.abs2_add_e(diff, acc), acc),
                            acc,
                        ),
                        simd.index_select(
                            im_is_not_nan,
                            simd.index_select(
                                re_is_not_nan,
                                simd.index_add(
                                    non_nan_count,
                                    simd.index_splat(E::faer_usize_to_index(1)),
                                ),
                                non_nan_count,
                            ),
                            non_nan_count,
                        ),
                    )
                }

                let mut sum0 = simd.splat(E::faer_zero());
                let mut sum1 = simd.splat(E::faer_zero());
                let mut sum2 = simd.splat(E::faer_zero());
                let mut sum3 = simd.splat(E::faer_zero());
                let mut non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count1 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count2 = simd.index_splat(E::faer_usize_to_index(0));
                let mut non_nan_count3 = simd.index_splat(E::faer_usize_to_index(0));

                (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, head);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);
                non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));

                let (body4, body1) = body.as_arrays::<4>();

                let mut start = 0usize;
                while start < body4.len() {
                    let len = Ord::min(body4.len() - start, chunk_size);

                    for [x0, x1, x2, x3] in body4
                        .subslice(start..start + len)
                        .into_ref_iter()
                        .map(RefGroup::unzip)
                    {
                        (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, x0);
                        (sum1, non_nan_count1) = process(simd, sum1, mean, non_nan_count1, x1);
                        (sum2, non_nan_count2) = process(simd, sum2, mean, non_nan_count2, x2);
                        (sum3, non_nan_count3) = process(simd, sum3, mean, non_nan_count3, x3);
                    }
                    non_nan_count0 = simd.index_add(non_nan_count0, non_nan_count1);
                    non_nan_count2 = simd.index_add(non_nan_count2, non_nan_count3);
                    non_nan_count0 = simd.index_add(non_nan_count0, non_nan_count2);
                    non_nan_count_total += reduce::<E, S>(non_nan_count0);
                    non_nan_count0 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count1 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count2 = simd.index_splat(E::faer_usize_to_index(0));
                    non_nan_count3 = simd.index_splat(E::faer_usize_to_index(0));

                    start += len;
                }

                for x0 in body1.into_ref_iter() {
                    (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, x0);
                }

                (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, tail);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);

                sum0 = simd.add(sum0, sum1);
                sum2 = simd.add(sum2, sum3);
                sum0 = simd.add(sum0, sum2);

                sum0 = simd.rotate_left(sum0, offset.rotate_left_amount());
                let sum = simd.reduce_add(sum0);

                let var = if non_nan_count_total == 0 {
                    E::faer_nan()
                } else if non_nan_count_total == 1 {
                    E::faer_zero()
                } else {
                    sum.faer_scale_real(from_usize::<E>(non_nan_count_total - 1).faer_inv())
                };

                out.write(i, var);
            }
        }
    }

    E::Simd::default().dispatch(Impl { out, mat, col_mean });
}

fn col_mean_row_major_ignore_nan_c32(out: ColMut<'_, c32>, mat: MatRef<'_, c32>) {
    type E = f32;

    struct Impl<'a> {
        out: ColMut<'a, c32>,
        mat: MatRef<'a, c32>,
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { mut out, mat } = self;

            let m = mat.nrows();
            let chunk_size = if core::mem::size_of::<u32>() < core::mem::size_of::<usize>() {
                1usize << (core::mem::size_of::<u32>() * 8)
            } else {
                usize::MAX
            } / 4;

            let offset = simd.c32s_align_offset(mat.as_ptr() as _, mat.ncols());
            for i in 0..m {
                let row = mat.row(i).try_as_slice().unwrap();
                let (head, body, tail) =
                    simd.c32s_as_aligned_simd(bytemuck::cast_slice(row), offset);

                let mut non_nan_count_total = 0usize;

                #[inline(always)]
                fn process<S: pulp::Simd>(
                    simd: S,
                    acc: S::c32s,
                    non_nan_count: S::u32s,
                    val: impl Read<Output = S::c32s>,
                ) -> (S::c32s, S::u32s) {
                    let val = val.read_or(simd.c32s_splat(Complex::<E>::faer_nan()));

                    if coe::is_same::<S, pulp::Scalar>() {
                        let acc: c32 = bytemuck::cast(acc);
                        let val: c32 = bytemuck::cast(val);
                        let non_nan_count: u32 = bytemuck::cast(non_nan_count);

                        let is_nan = val.re.is_nan() || val.im.is_nan();
                        let val = if is_nan { c32::faer_zero() } else { val };

                        (
                            bytemuck::cast(acc + val),
                            bytemuck::cast(non_nan_count + is_nan as u32 * 2),
                        )
                    } else {
                        let acc: S::f32s = bytemuck::cast(acc);
                        let val_swap: S::f32s = bytemuck::cast(simd.c32s_swap_re_im(val));
                        let val: S::f32s = bytemuck::cast(val);

                        let is_not_nan = simd.m32s_and(
                            simd.f32s_equal(val, val),
                            simd.f32s_equal(val_swap, val_swap),
                        );

                        (
                            bytemuck::cast(simd.m32s_select_f32s(
                                is_not_nan,
                                simd.f32s_add(acc, val),
                                acc,
                            )),
                            simd.m32s_select_u32s(
                                is_not_nan,
                                simd.u32s_add(non_nan_count, simd.u32s_splat(1)),
                                non_nan_count,
                            ),
                        )
                    }
                }

                let mut sum0 = simd.c32s_splat(Complex::<E>::faer_zero());
                let mut sum1 = simd.c32s_splat(Complex::<E>::faer_zero());
                let mut sum2 = simd.c32s_splat(Complex::<E>::faer_zero());
                let mut sum3 = simd.c32s_splat(Complex::<E>::faer_zero());
                let mut non_nan_count0 = simd.u32s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count1 = simd.u32s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count2 = simd.u32s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count3 = simd.u32s_splat(E::faer_usize_to_index(0));

                (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, head);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);
                non_nan_count0 = simd.u32s_splat(E::faer_usize_to_index(0));

                let (body4, body1) = pulp::as_arrays::<4, _>(body);

                let mut start = 0usize;
                while start < body4.len() {
                    let len = Ord::min(body4.len() - start, chunk_size);

                    for [x0, x1, x2, x3] in &body4[start..start + len] {
                        (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, x0);
                        (sum1, non_nan_count1) = process(simd, sum1, non_nan_count1, x1);
                        (sum2, non_nan_count2) = process(simd, sum2, non_nan_count2, x2);
                        (sum3, non_nan_count3) = process(simd, sum3, non_nan_count3, x3);
                    }
                    non_nan_count0 = simd.u32s_add(non_nan_count0, non_nan_count1);
                    non_nan_count2 = simd.u32s_add(non_nan_count2, non_nan_count3);
                    non_nan_count0 = simd.u32s_add(non_nan_count0, non_nan_count2);
                    non_nan_count_total += reduce::<E, S>(non_nan_count0);
                    non_nan_count0 = simd.u32s_splat(E::faer_usize_to_index(0));
                    non_nan_count1 = simd.u32s_splat(E::faer_usize_to_index(0));
                    non_nan_count2 = simd.u32s_splat(E::faer_usize_to_index(0));
                    non_nan_count3 = simd.u32s_splat(E::faer_usize_to_index(0));

                    start += len;
                }

                for x0 in body1 {
                    (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, x0);
                }

                (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, tail);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);

                sum0 = simd.c32s_add(sum0, sum1);
                sum2 = simd.c32s_add(sum2, sum3);
                sum0 = simd.c32s_add(sum0, sum2);

                sum0 = simd.c32s_rotate_left(sum0, offset.rotate_left_amount());
                let sum: c32 = simd.c32s_reduce_sum(sum0).into();

                out.write(
                    i,
                    sum.faer_scale_real(from_usize::<E>(non_nan_count_total / 2).faer_inv()),
                );
            }
        }
    }

    <c32 as ComplexField>::Simd::default().dispatch(Impl { out, mat });
}

fn col_mean_row_major_ignore_nan_c64(out: ColMut<'_, c64>, mat: MatRef<'_, c64>) {
    type E = f64;

    struct Impl<'a> {
        out: ColMut<'a, c64>,
        mat: MatRef<'a, c64>,
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { mut out, mat } = self;

            let m = mat.nrows();
            let chunk_size = if core::mem::size_of::<u64>() < core::mem::size_of::<usize>() {
                1usize << (core::mem::size_of::<u64>() * 8)
            } else {
                usize::MAX
            } / 4;

            let offset = simd.c64s_align_offset(mat.as_ptr() as _, mat.ncols());
            for i in 0..m {
                let row = mat.row(i).try_as_slice().unwrap();
                let (head, body, tail) =
                    simd.c64s_as_aligned_simd(bytemuck::cast_slice(row), offset);

                let mut non_nan_count_total = 0usize;

                #[inline(always)]
                fn process<S: pulp::Simd>(
                    simd: S,
                    acc: S::c64s,
                    non_nan_count: S::u64s,
                    val: impl Read<Output = S::c64s>,
                ) -> (S::c64s, S::u64s) {
                    let val = val.read_or(simd.c64s_splat(Complex::<E>::faer_nan()));

                    if coe::is_same::<S, pulp::Scalar>() {
                        let acc: c64 = bytemuck::cast(acc);
                        let val: c64 = bytemuck::cast(val);
                        let non_nan_count: u64 = bytemuck::cast(non_nan_count);

                        let is_nan = val.re.is_nan() || val.im.is_nan();
                        let val = if is_nan { c64::faer_zero() } else { val };

                        (
                            bytemuck::cast(acc + val),
                            bytemuck::cast(non_nan_count + is_nan as u64 * 2),
                        )
                    } else {
                        let acc: S::f64s = bytemuck::cast(acc);
                        let val_swap: S::f64s = bytemuck::cast(simd.c64s_swap_re_im(val));
                        let val: S::f64s = bytemuck::cast(val);

                        let is_not_nan = simd.m64s_and(
                            simd.f64s_equal(val, val),
                            simd.f64s_equal(val_swap, val_swap),
                        );

                        (
                            bytemuck::cast(simd.m64s_select_f64s(
                                is_not_nan,
                                simd.f64s_add(acc, val),
                                acc,
                            )),
                            simd.m64s_select_u64s(
                                is_not_nan,
                                simd.u64s_add(non_nan_count, simd.u64s_splat(1)),
                                non_nan_count,
                            ),
                        )
                    }
                }

                let mut sum0 = simd.c64s_splat(Complex::<E>::faer_zero());
                let mut sum1 = simd.c64s_splat(Complex::<E>::faer_zero());
                let mut sum2 = simd.c64s_splat(Complex::<E>::faer_zero());
                let mut sum3 = simd.c64s_splat(Complex::<E>::faer_zero());
                let mut non_nan_count0 = simd.u64s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count1 = simd.u64s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count2 = simd.u64s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count3 = simd.u64s_splat(E::faer_usize_to_index(0));

                (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, head);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);
                non_nan_count0 = simd.u64s_splat(E::faer_usize_to_index(0));

                let (body4, body1) = pulp::as_arrays::<4, _>(body);

                let mut start = 0usize;
                while start < body4.len() {
                    let len = Ord::min(body4.len() - start, chunk_size);

                    for [x0, x1, x2, x3] in &body4[start..start + len] {
                        (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, x0);
                        (sum1, non_nan_count1) = process(simd, sum1, non_nan_count1, x1);
                        (sum2, non_nan_count2) = process(simd, sum2, non_nan_count2, x2);
                        (sum3, non_nan_count3) = process(simd, sum3, non_nan_count3, x3);
                    }
                    non_nan_count0 = simd.u64s_add(non_nan_count0, non_nan_count1);
                    non_nan_count2 = simd.u64s_add(non_nan_count2, non_nan_count3);
                    non_nan_count0 = simd.u64s_add(non_nan_count0, non_nan_count2);
                    non_nan_count_total += reduce::<E, S>(non_nan_count0);
                    non_nan_count0 = simd.u64s_splat(E::faer_usize_to_index(0));
                    non_nan_count1 = simd.u64s_splat(E::faer_usize_to_index(0));
                    non_nan_count2 = simd.u64s_splat(E::faer_usize_to_index(0));
                    non_nan_count3 = simd.u64s_splat(E::faer_usize_to_index(0));

                    start += len;
                }

                for x0 in body1 {
                    (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, x0);
                }

                (sum0, non_nan_count0) = process(simd, sum0, non_nan_count0, tail);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);

                sum0 = simd.c64s_add(sum0, sum1);
                sum2 = simd.c64s_add(sum2, sum3);
                sum0 = simd.c64s_add(sum0, sum2);

                sum0 = simd.c64s_rotate_left(sum0, offset.rotate_left_amount());
                let sum: c64 = simd.c64s_reduce_sum(sum0).into();

                out.write(
                    i,
                    sum.faer_scale_real(from_usize::<E>(non_nan_count_total / 2).faer_inv()),
                );
            }
        }
    }

    <c64 as ComplexField>::Simd::default().dispatch(Impl { out, mat });
}

fn col_varm_row_major_ignore_nan_c32(
    out: ColMut<'_, f32>,
    mat: MatRef<'_, c32>,
    col_mean: ColRef<'_, c32>,
) {
    type E = f32;

    struct Impl<'a> {
        out: ColMut<'a, f32>,
        mat: MatRef<'a, c32>,
        col_mean: ColRef<'a, c32>,
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self {
                mut out,
                mat,
                col_mean,
            } = self;

            let m = mat.nrows();
            let chunk_size = if core::mem::size_of::<u32>() < core::mem::size_of::<usize>() {
                1usize << (core::mem::size_of::<u32>() * 8)
            } else {
                usize::MAX
            } / 4;

            let offset = simd.c32s_align_offset(mat.as_ptr() as _, mat.ncols());
            for i in 0..m {
                let mean = simd.c32s_splat(bytemuck::cast(col_mean.read(i)));
                let row = mat.row(i).try_as_slice().unwrap();
                let (head, body, tail) =
                    simd.c32s_as_aligned_simd(bytemuck::cast_slice(row), offset);

                let mut non_nan_count_total = 0usize;

                #[inline(always)]
                fn process<S: pulp::Simd>(
                    simd: S,
                    acc: S::f32s,
                    mean: S::c32s,
                    non_nan_count: S::u32s,
                    val: impl Read<Output = S::c32s>,
                ) -> (S::f32s, S::u32s) {
                    let val = val.read_or(simd.c32s_splat(Complex::<E>::faer_nan()));

                    if coe::is_same::<S, pulp::Scalar>() {
                        let acc: f32 = bytemuck::cast(acc);
                        let mean: c32 = bytemuck::cast(mean);
                        let val: c32 = bytemuck::cast(val);
                        let non_nan_count: u32 = bytemuck::cast(non_nan_count);

                        let is_nan = val.re.is_nan() || val.im.is_nan();
                        let val = if is_nan { mean } else { val };
                        let diff = val - mean;

                        (
                            bytemuck::cast(acc + diff.faer_abs2()),
                            bytemuck::cast(non_nan_count + is_nan as u32 * 2),
                        )
                    } else {
                        let acc: S::f32s = bytemuck::cast(acc);
                        let mean: S::f32s = bytemuck::cast(mean);
                        let val_swap: S::f32s = bytemuck::cast(simd.c32s_swap_re_im(val));
                        let val: S::f32s = bytemuck::cast(val);

                        let is_not_nan = simd.m32s_and(
                            simd.f32s_equal(val, val),
                            simd.f32s_equal(val_swap, val_swap),
                        );

                        let diff = simd.f32s_sub(val, mean);

                        (
                            simd.m32s_select_f32s(
                                is_not_nan,
                                simd.f32s_mul_add_e(diff, diff, acc),
                                acc,
                            ),
                            simd.m32s_select_u32s(
                                is_not_nan,
                                simd.u32s_add(non_nan_count, simd.u32s_splat(1)),
                                non_nan_count,
                            ),
                        )
                    }
                }

                let mut sum0 = simd.f32s_splat(0.0);
                let mut sum1 = simd.f32s_splat(0.0);
                let mut sum2 = simd.f32s_splat(0.0);
                let mut sum3 = simd.f32s_splat(0.0);
                let mut non_nan_count0 = simd.u32s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count1 = simd.u32s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count2 = simd.u32s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count3 = simd.u32s_splat(E::faer_usize_to_index(0));

                (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, head);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);
                non_nan_count0 = simd.u32s_splat(E::faer_usize_to_index(0));

                let (body4, body1) = pulp::as_arrays::<4, _>(body);

                let mut start = 0usize;
                while start < body4.len() {
                    let len = Ord::min(body4.len() - start, chunk_size);

                    for [x0, x1, x2, x3] in &body4[start..start + len] {
                        (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, x0);
                        (sum1, non_nan_count1) = process(simd, sum1, mean, non_nan_count1, x1);
                        (sum2, non_nan_count2) = process(simd, sum2, mean, non_nan_count2, x2);
                        (sum3, non_nan_count3) = process(simd, sum3, mean, non_nan_count3, x3);
                    }
                    non_nan_count0 = simd.u32s_add(non_nan_count0, non_nan_count1);
                    non_nan_count2 = simd.u32s_add(non_nan_count2, non_nan_count3);
                    non_nan_count0 = simd.u32s_add(non_nan_count0, non_nan_count2);
                    non_nan_count_total += reduce::<E, S>(non_nan_count0);
                    non_nan_count0 = simd.u32s_splat(E::faer_usize_to_index(0));
                    non_nan_count1 = simd.u32s_splat(E::faer_usize_to_index(0));
                    non_nan_count2 = simd.u32s_splat(E::faer_usize_to_index(0));
                    non_nan_count3 = simd.u32s_splat(E::faer_usize_to_index(0));

                    start += len;
                }

                for x0 in body1 {
                    (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, x0);
                }

                (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, tail);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);

                sum0 = simd.f32s_add(sum0, sum1);
                sum2 = simd.f32s_add(sum2, sum3);
                sum0 = simd.f32s_add(sum0, sum2);

                sum0 = simd.f32s_rotate_left(sum0, offset.rotate_left_amount());
                let sum = simd.f32s_reduce_sum(sum0);

                non_nan_count_total /= 2;

                let var = if non_nan_count_total == 0 {
                    E::faer_nan()
                } else if non_nan_count_total == 1 {
                    E::faer_zero()
                } else {
                    sum.faer_scale_real(from_usize::<E>(non_nan_count_total - 1).faer_inv())
                };

                out.write(i, var);
            }
        }
    }

    <c32 as ComplexField>::Simd::default().dispatch(Impl { out, mat, col_mean });
}

fn col_varm_row_major_ignore_nan_c64(
    out: ColMut<'_, f64>,
    mat: MatRef<'_, c64>,
    col_mean: ColRef<'_, c64>,
) {
    type E = f64;

    struct Impl<'a> {
        out: ColMut<'a, f64>,
        mat: MatRef<'a, c64>,
        col_mean: ColRef<'a, c64>,
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self {
                mut out,
                mat,
                col_mean,
            } = self;

            let m = mat.nrows();
            let chunk_size = if core::mem::size_of::<u64>() < core::mem::size_of::<usize>() {
                1usize << (core::mem::size_of::<u64>() * 8)
            } else {
                usize::MAX
            } / 4;

            let offset = simd.c64s_align_offset(mat.as_ptr() as _, mat.ncols());
            for i in 0..m {
                let mean = simd.c64s_splat(bytemuck::cast(col_mean.read(i)));
                let row = mat.row(i).try_as_slice().unwrap();
                let (head, body, tail) =
                    simd.c64s_as_aligned_simd(bytemuck::cast_slice(row), offset);

                let mut non_nan_count_total = 0usize;

                #[inline(always)]
                fn process<S: pulp::Simd>(
                    simd: S,
                    acc: S::f64s,
                    mean: S::c64s,
                    non_nan_count: S::u64s,
                    val: impl Read<Output = S::c64s>,
                ) -> (S::f64s, S::u64s) {
                    let val = val.read_or(simd.c64s_splat(Complex::<E>::faer_nan()));

                    if coe::is_same::<S, pulp::Scalar>() {
                        let acc: f64 = bytemuck::cast(acc);
                        let mean: c64 = bytemuck::cast(mean);
                        let val: c64 = bytemuck::cast(val);
                        let non_nan_count: u64 = bytemuck::cast(non_nan_count);

                        let is_nan = val.re.is_nan() || val.im.is_nan();
                        let val = if is_nan { mean } else { val };
                        let diff = val - mean;

                        (
                            bytemuck::cast(acc + diff.faer_abs2()),
                            bytemuck::cast(non_nan_count + is_nan as u64 * 2),
                        )
                    } else {
                        let acc: S::f64s = bytemuck::cast(acc);
                        let mean: S::f64s = bytemuck::cast(mean);
                        let val_swap: S::f64s = bytemuck::cast(simd.c64s_swap_re_im(val));
                        let val: S::f64s = bytemuck::cast(val);

                        let is_not_nan = simd.m64s_and(
                            simd.f64s_equal(val, val),
                            simd.f64s_equal(val_swap, val_swap),
                        );

                        let diff = simd.f64s_sub(val, mean);

                        (
                            simd.m64s_select_f64s(
                                is_not_nan,
                                simd.f64s_mul_add_e(diff, diff, acc),
                                acc,
                            ),
                            simd.m64s_select_u64s(
                                is_not_nan,
                                simd.u64s_add(non_nan_count, simd.u64s_splat(1)),
                                non_nan_count,
                            ),
                        )
                    }
                }

                let mut sum0 = simd.f64s_splat(0.0);
                let mut sum1 = simd.f64s_splat(0.0);
                let mut sum2 = simd.f64s_splat(0.0);
                let mut sum3 = simd.f64s_splat(0.0);
                let mut non_nan_count0 = simd.u64s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count1 = simd.u64s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count2 = simd.u64s_splat(E::faer_usize_to_index(0));
                let mut non_nan_count3 = simd.u64s_splat(E::faer_usize_to_index(0));

                (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, head);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);
                non_nan_count0 = simd.u64s_splat(E::faer_usize_to_index(0));

                let (body4, body1) = pulp::as_arrays::<4, _>(body);

                let mut start = 0usize;
                while start < body4.len() {
                    let len = Ord::min(body4.len() - start, chunk_size);

                    for [x0, x1, x2, x3] in &body4[start..start + len] {
                        (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, x0);
                        (sum1, non_nan_count1) = process(simd, sum1, mean, non_nan_count1, x1);
                        (sum2, non_nan_count2) = process(simd, sum2, mean, non_nan_count2, x2);
                        (sum3, non_nan_count3) = process(simd, sum3, mean, non_nan_count3, x3);
                    }
                    non_nan_count0 = simd.u64s_add(non_nan_count0, non_nan_count1);
                    non_nan_count2 = simd.u64s_add(non_nan_count2, non_nan_count3);
                    non_nan_count0 = simd.u64s_add(non_nan_count0, non_nan_count2);
                    non_nan_count_total += reduce::<E, S>(non_nan_count0);
                    non_nan_count0 = simd.u64s_splat(E::faer_usize_to_index(0));
                    non_nan_count1 = simd.u64s_splat(E::faer_usize_to_index(0));
                    non_nan_count2 = simd.u64s_splat(E::faer_usize_to_index(0));
                    non_nan_count3 = simd.u64s_splat(E::faer_usize_to_index(0));

                    start += len;
                }

                for x0 in body1 {
                    (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, x0);
                }

                (sum0, non_nan_count0) = process(simd, sum0, mean, non_nan_count0, tail);
                non_nan_count_total += reduce::<E, S>(non_nan_count0);

                sum0 = simd.f64s_add(sum0, sum1);
                sum2 = simd.f64s_add(sum2, sum3);
                sum0 = simd.f64s_add(sum0, sum2);

                sum0 = simd.f64s_rotate_left(sum0, offset.rotate_left_amount());
                let sum = simd.f64s_reduce_sum(sum0);

                non_nan_count_total /= 2;

                let var = if non_nan_count_total == 0 {
                    E::faer_nan()
                } else if non_nan_count_total == 1 {
                    E::faer_zero()
                } else {
                    sum.faer_scale_real(from_usize::<E>(non_nan_count_total - 1).faer_inv())
                };

                out.write(i, var);
            }
        }
    }

    <c64 as ComplexField>::Simd::default().dispatch(Impl { out, mat, col_mean });
}

fn col_mean_propagate<E: ComplexField>(out: ColMut<'_, E>, mat: MatRef<'_, E>) {
    fn col_mean_row_major<E: ComplexField>(out: ColMut<'_, E>, mat: MatRef<'_, E>) {
        struct Impl<'a, E: ComplexField> {
            out: ColMut<'a, E>,
            mat: MatRef<'a, E>,
        }

        impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self { mut out, mat } = self;
                let simd = SimdFor::<E, S>::new(simd);

                let m = mat.nrows();
                let n = mat.ncols();
                let one_n = from_usize::<E::Real>(n).faer_inv();

                let offset = simd.align_offset_ptr(mat.as_ptr(), mat.ncols());
                for i in 0..m {
                    let row = SliceGroup::<'_, E>::new(mat.row(i).try_as_slice().unwrap());
                    let (head, body, tail) = simd.as_aligned_simd(row, offset);
                    let mut sum0 = head.read_or(simd.splat(E::faer_zero()));
                    let mut sum1 = simd.splat(E::faer_zero());
                    let mut sum2 = simd.splat(E::faer_zero());
                    let mut sum3 = simd.splat(E::faer_zero());

                    let (body4, body1) = body.as_arrays::<4>();
                    for [x0, x1, x2, x3] in body4.into_ref_iter().map(RefGroup::unzip) {
                        sum0 = simd.add(sum0, x0.get());
                        sum1 = simd.add(sum1, x1.get());
                        sum2 = simd.add(sum2, x2.get());
                        sum3 = simd.add(sum3, x3.get());
                    }
                    for x0 in body1.into_ref_iter() {
                        sum0 = simd.add(sum0, x0.get());
                    }
                    sum0 = simd.add(sum0, tail.read_or(simd.splat(E::faer_zero())));

                    sum0 = simd.add(sum0, sum1);
                    sum2 = simd.add(sum2, sum3);
                    sum0 = simd.add(sum0, sum2);

                    sum0 = simd.rotate_left(sum0, offset.rotate_left_amount());
                    let sum = simd.reduce_add(sum0);

                    out.write(i, sum.faer_scale_real(one_n));
                }
            }
        }

        E::Simd::default().dispatch(Impl { out, mat });
    }

    fn col_mean_col_major<E: ComplexField>(out: ColMut<'_, E>, mat: MatRef<'_, E>) {
        struct Impl<'a, E: ComplexField> {
            out: ColMut<'a, E>,
            mat: MatRef<'a, E>,
        }

        impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self { out, mat } = self;
                let simd_real = SimdFor::<E::Real, S>::new(simd);
                let simd = SimdFor::<E, S>::new(simd);

                let m = mat.nrows();
                let n = mat.ncols();
                let one_n = simd_real.splat(from_usize::<E::Real>(n).faer_inv());

                let offset = simd.align_offset_ptr(mat.as_ptr(), m);
                let mut out = SliceGroupMut::<'_, E>::new(out.try_as_slice_mut().unwrap());

                out.fill_zero();
                for j in 0..n {
                    let col = SliceGroup::<'_, E>::new(mat.col(j).try_as_slice().unwrap());
                    let (head, body, tail) = simd.as_aligned_simd(col, offset);
                    let (out_head, out_body, out_tail) =
                        simd.as_aligned_simd_mut(out.rb_mut(), offset);

                    #[inline(always)]
                    fn process<E: ComplexField, S: pulp::Simd>(
                        simd: SimdFor<E, S>,
                        mut out: impl Write<Output = SimdGroupFor<E, S>>,
                        val: impl Read<Output = SimdGroupFor<E, S>>,
                    ) {
                        out.write(simd.add(
                            out.read_or(simd.splat(E::faer_zero())),
                            val.read_or(simd.splat(E::faer_zero())),
                        ))
                    }

                    process(simd, out_head, head);
                    for (out, x) in zip(out_body.into_mut_iter(), body.into_ref_iter()) {
                        process(simd, out, x)
                    }
                    process(simd, out_tail, tail);
                }

                #[inline(always)]
                fn process<E: ComplexField, S: pulp::Simd>(
                    simd: SimdFor<E, S>,
                    mut out: impl Write<Output = SimdGroupFor<E, S>>,
                    one_n: SimdGroupFor<E::Real, S>,
                ) {
                    out.write(simd.scale_real(one_n, out.read_or(simd.splat(E::faer_zero()))))
                }
                let (out_head, out_body, out_tail) = simd.as_aligned_simd_mut(out.rb_mut(), offset);
                process(simd, out_head, one_n);
                for out in out_body.into_mut_iter() {
                    process(simd, out, one_n);
                }
                process(simd, out_tail, one_n);
            }
        }

        E::Simd::default().dispatch(Impl { out, mat });
    }

    let mut mat = mat;
    let mut out = out;

    if mat.ncols() == 0 {
        out.fill(E::faer_nan());
        return;
    }

    if mat.col_stride() < 0 {
        mat = mat.reverse_cols();
    };
    if mat.row_stride() < 0 {
        mat = mat.reverse_rows();
        out = out.reverse_rows_mut();
    };

    if mat.col_stride() == 1 {
        col_mean_row_major(out, mat)
    } else if mat.row_stride() == 1 && out.row_stride() == 1 {
        col_mean_col_major(out, mat)
    } else {
        let n = mat.ncols();
        let one_n = from_usize::<E::Real>(n).faer_inv();

        out.fill_zero();
        for j in 0..n {
            out += mat.col(j);
        }
        zipped!(out).for_each(|unzipped!(mut x)| x.write(x.read().faer_scale_real(one_n)));
    }
}

fn row_mean_propagate<E: ComplexField>(out: RowMut<'_, E>, mat: MatRef<'_, E>) {
    col_mean_propagate(out.transpose_mut(), mat.transpose());
}

fn col_varm_propagate<E: ComplexField>(
    out: ColMut<'_, E::Real>,
    mat: MatRef<'_, E>,
    col_mean: ColRef<'_, E>,
) {
    fn col_varm_row_major<E: ComplexField>(
        out: ColMut<'_, E::Real>,
        mat: MatRef<'_, E>,
        col_mean: ColRef<'_, E>,
    ) {
        struct Impl<'a, E: ComplexField> {
            out: ColMut<'a, E::Real>,
            mat: MatRef<'a, E>,
            col_mean: ColRef<'a, E>,
        }

        impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self {
                    mut out,
                    mat,
                    col_mean,
                } = self;

                let simd_real = SimdFor::<E::Real, S>::new(simd);
                let simd = SimdFor::<E, S>::new(simd);

                let m = mat.nrows();
                let n = mat.ncols();
                let one_n1 = from_usize::<E::Real>(n - 1).faer_inv();

                let offset = simd.align_offset_ptr(mat.as_ptr(), mat.ncols());
                for i in 0..m {
                    let mean = simd.splat(col_mean.read(i));
                    let row = SliceGroup::<'_, E>::new(mat.row(i).try_as_slice().unwrap());
                    let (head, body, tail) = simd.as_aligned_simd(row, offset);

                    #[inline(always)]
                    fn process<E: ComplexField, S: pulp::Simd>(
                        simd: SimdFor<E, S>,
                        acc: SimdGroupFor<E::Real, S>,
                        mean: SimdGroupFor<E, S>,
                        val: impl Read<Output = SimdGroupFor<E, S>>,
                    ) -> SimdGroupFor<E::Real, S> {
                        let diff = simd.sub(val.read_or(mean), mean);
                        if const { E::IS_C32 } {
                            let diff = coe::coerce_static::<SimdGroupFor<E, S>, SimdGroupFor<c32, S>>(
                                diff,
                            );
                            let acc = coe::coerce_static::<
                                SimdGroupFor<E::Real, S>,
                                SimdGroupFor<f32, S>,
                            >(acc);

                            if coe::is_same::<S, pulp::Scalar>() {
                                let diff: c32 = bytemuck::cast(diff);
                                let acc: f32 = bytemuck::cast(acc);

                                coe::coerce_static::<
                                    SimdGroupFor<f32, pulp::Scalar>,
                                    SimdGroupFor<E::Real, S>,
                                >(diff.faer_abs2() + acc)
                            } else {
                                let diff: S::f32s = bytemuck::cast(diff);
                                coe::coerce_static::<SimdGroupFor<f32, S>, SimdGroupFor<E::Real, S>>(
                                    simd.simd.f32s_mul_add_e(diff, diff, bytemuck::cast(acc)),
                                )
                            }
                        } else if const { E::IS_C64 } {
                            let diff = coe::coerce_static::<SimdGroupFor<E, S>, SimdGroupFor<c64, S>>(
                                diff,
                            );
                            let acc = coe::coerce_static::<
                                SimdGroupFor<E::Real, S>,
                                SimdGroupFor<f64, S>,
                            >(acc);

                            if coe::is_same::<S, pulp::Scalar>() {
                                let diff: c64 = bytemuck::cast(diff);
                                let acc: f64 = bytemuck::cast(acc);

                                coe::coerce_static::<
                                    SimdGroupFor<f64, pulp::Scalar>,
                                    SimdGroupFor<E::Real, S>,
                                >(diff.faer_abs2() + acc)
                            } else {
                                let diff: S::f64s = bytemuck::cast(diff);
                                simd.simd.f64s_mul_add_e(diff, diff, bytemuck::cast(acc));
                                coe::coerce_static::<SimdGroupFor<f64, S>, SimdGroupFor<E::Real, S>>(
                                    simd.simd.f64s_mul_add_e(diff, diff, bytemuck::cast(acc)),
                                )
                            }
                        } else {
                            simd.abs2_add_e(diff, acc)
                        }
                    }

                    let mut sum0 = simd_real.splat(E::Real::faer_zero());
                    let mut sum1 = simd_real.splat(E::Real::faer_zero());
                    let mut sum2 = simd_real.splat(E::Real::faer_zero());
                    let mut sum3 = simd_real.splat(E::Real::faer_zero());

                    sum0 = process(simd, sum0, mean, head);
                    let (body4, body1) = body.as_arrays::<4>();
                    for [x0, x1, x2, x3] in body4.into_ref_iter().map(RefGroup::unzip) {
                        sum0 = process(simd, sum0, mean, x0);
                        sum1 = process(simd, sum1, mean, x1);
                        sum2 = process(simd, sum2, mean, x2);
                        sum3 = process(simd, sum3, mean, x3);
                    }
                    for x0 in body1.into_ref_iter() {
                        sum0 = process(simd, sum0, mean, x0);
                    }
                    sum0 = process(simd, sum0, mean, tail);

                    sum0 = simd_real.add(sum0, sum1);
                    sum2 = simd_real.add(sum2, sum3);
                    sum0 = simd_real.add(sum0, sum2);

                    sum0 = simd_real.rotate_left(sum0, offset.rotate_left_amount());
                    let sum = simd_real.reduce_add(sum0);

                    out.write(i, sum.faer_scale_real(one_n1));
                }
            }
        }

        E::Simd::default().dispatch(Impl { out, mat, col_mean });
    }

    fn col_varm_col_major_real<E: RealField>(
        out: ColMut<'_, E>,
        mat: MatRef<'_, E>,
        col_mean: ColRef<'_, E>,
    ) {
        struct Impl<'a, E: RealField> {
            out: ColMut<'a, E>,
            mat: MatRef<'a, E>,
            col_mean: ColRef<'a, E>,
        }

        impl<E: RealField> pulp::WithSimd for Impl<'_, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self { out, mat, col_mean } = self;

                let simd = SimdFor::<E, S>::new(simd);

                let n = mat.ncols();
                let one_n1 = simd.splat(from_usize::<E::Real>(n - 1).faer_inv());

                let offset = simd.align_offset_ptr(mat.as_ptr(), mat.nrows());

                let mut out = SliceGroupMut::<'_, E>::new(out.try_as_slice_mut().unwrap());
                let col_mean = SliceGroup::<'_, E>::new(col_mean.try_as_slice().unwrap());

                out.fill_zero();
                for j in 0..n {
                    let col = SliceGroup::<'_, E>::new(mat.col(j).try_as_slice().unwrap());
                    let (head, body, tail) = simd.as_aligned_simd(col, offset);
                    let (out_head, out_body, out_tail) =
                        simd.as_aligned_simd_mut(out.rb_mut(), offset);
                    let (mean_head, mean_body, mean_tail) = simd.as_aligned_simd(col_mean, offset);

                    #[inline(always)]
                    fn process<E: RealField, S: pulp::Simd>(
                        simd: SimdFor<E, S>,
                        mut out: impl Write<Output = SimdGroupFor<E, S>>,
                        val: impl Read<Output = SimdGroupFor<E, S>>,
                        mean: impl Read<Output = SimdGroupFor<E, S>>,
                    ) {
                        let zero = simd.splat(E::faer_zero());
                        let diff = simd.sub(val.read_or(zero), mean.read_or(zero));
                        out.write(simd.mul_add_e(diff, diff, out.read_or(zero)))
                    }

                    process(simd, out_head, head, mean_head);
                    for (out, (x, mean)) in zip(
                        out_body.into_mut_iter(),
                        zip(body.into_ref_iter(), mean_body.into_ref_iter()),
                    ) {
                        process(simd, out, x, mean);
                    }
                    process(simd, out_tail, tail, mean_tail);
                }

                #[inline(always)]
                fn process<E: RealField, S: pulp::Simd>(
                    simd: SimdFor<E, S>,
                    mut out: impl Write<Output = SimdGroupFor<E, S>>,
                    one_n1: SimdGroupFor<E, S>,
                ) {
                    out.write(simd.scale_real(one_n1, out.read_or(simd.splat(E::faer_zero()))))
                }
                let (out_head, out_body, out_tail) = simd.as_aligned_simd_mut(out.rb_mut(), offset);
                process(simd, out_head, one_n1);
                for out in out_body.into_mut_iter() {
                    process(simd, out, one_n1);
                }
                process(simd, out_tail, one_n1);
            }
        }

        E::Simd::default().dispatch(Impl { out, mat, col_mean });
    }

    fn col_varm_col_major_cplx<E: RealField>(
        out: ColMut<'_, E>,
        mat: MatRef<'_, Complex<E>>,
        col_mean: ColRef<'_, Complex<E>>,
    ) {
        struct Impl<'a, E: RealField> {
            out: ColMut<'a, E>,
            mat: MatRef<'a, Complex<E>>,
            col_mean: ColRef<'a, Complex<E>>,
        }

        impl<E: RealField> pulp::WithSimd for Impl<'_, E> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self { out, mat, col_mean } = self;

                let simd_cplx = SimdFor::<Complex<E>, S>::new(simd);
                let simd = SimdFor::<E, S>::new(simd);

                let n = mat.ncols();
                let one_n1 = simd.splat(from_usize::<E::Real>(n - 1).faer_inv());

                let offset = simd_cplx.align_offset_ptr(mat.as_ptr(), mat.nrows());

                let mut out = SliceGroupMut::<'_, E>::new(out.try_as_slice_mut().unwrap());
                let col_mean = SliceGroup::<'_, Complex<E>>::new(col_mean.try_as_slice().unwrap());

                out.fill_zero();
                for j in 0..n {
                    let col = SliceGroup::<'_, Complex<E>>::new(mat.col(j).try_as_slice().unwrap());
                    let (head, body, tail) = simd_cplx.as_aligned_simd(col, offset);
                    let (out_head, out_body, out_tail) =
                        simd.as_aligned_simd_mut(out.rb_mut(), offset);
                    let (mean_head, mean_body, mean_tail) =
                        simd_cplx.as_aligned_simd(col_mean, offset);

                    #[inline(always)]
                    fn process<E: RealField, S: pulp::Simd>(
                        simd: SimdFor<E, S>,
                        mut out: impl Write<Output = SimdGroupFor<E, S>>,
                        val: impl Read<Output = SimdGroupFor<Complex<E>, S>>,
                        mean: impl Read<Output = SimdGroupFor<Complex<E>, S>>,
                    ) {
                        let simd_cplx = SimdFor::<Complex<E>, S>::new(simd.simd);
                        let zero = simd_cplx.splat(Complex::<E>::faer_zero());
                        let zero_r = simd.splat(E::faer_zero());
                        let val = val.read_or(zero);
                        let mean = mean.read_or(zero);
                        let diff_re = simd.sub(val.re, mean.re);
                        let diff_im = simd.sub(val.im, mean.im);
                        out.write(simd.mul_add_e(
                            diff_im,
                            diff_im,
                            simd.mul_add_e(diff_re, diff_re, out.read_or(zero_r)),
                        ))
                    }

                    process(simd, out_head, head, mean_head);
                    for (out, (x, mean)) in zip(
                        out_body.into_mut_iter(),
                        zip(body.into_ref_iter(), mean_body.into_ref_iter()),
                    ) {
                        process(simd, out, x, mean);
                    }
                    process(simd, out_tail, tail, mean_tail);
                }

                #[inline(always)]
                fn process<E: RealField, S: pulp::Simd>(
                    simd: SimdFor<E, S>,
                    mut out: impl Write<Output = SimdGroupFor<E, S>>,
                    one_n1: SimdGroupFor<E, S>,
                ) {
                    out.write(simd.scale_real(one_n1, out.read_or(simd.splat(E::faer_zero()))))
                }
                let (out_head, out_body, out_tail) = simd.as_aligned_simd_mut(out.rb_mut(), offset);
                process(simd, out_head, one_n1);
                for out in out_body.into_mut_iter() {
                    process(simd, out, one_n1);
                }
                process(simd, out_tail, one_n1);
            }
        }

        E::Simd::default().dispatch(Impl { out, mat, col_mean });
    }

    let mut out = out;
    let mut mat = mat;
    let mut col_mean = col_mean;

    if mat.ncols() == 0 {
        out.fill(E::Real::faer_nan());
        return;
    }
    if mat.ncols() == 1 {
        out.fill_zero();
        return;
    }

    if mat.col_stride() < 0 {
        mat = mat.reverse_cols();
    };
    if mat.row_stride() < 0 {
        out = out.reverse_rows_mut();
        mat = mat.reverse_rows();
        col_mean = col_mean.reverse_rows();
    };

    if mat.col_stride() == 1 {
        col_varm_row_major(out, mat, col_mean)
    } else if mat.row_stride() == 1 && out.row_stride() == 1 && col_mean.row_stride() == 1 {
        if const { E::IS_REAL } {
            col_varm_col_major_real::<E::Real>(out, mat.coerce(), col_mean.coerce())
        } else if const { E::IS_NUM_COMPLEX } {
            col_varm_col_major_cplx::<E::Real>(out, mat.coerce(), col_mean.coerce())
        } else if const { E::IS_C32 } {
            let m = mat.nrows();

            let mut tmp = Col::<f32>::zeros(2 * m);
            let mut out: ColMut<'_, f32> = out.coerce();
            let mat: MatRef<'_, c32> = mat.coerce();
            let col_mean: ColRef<'_, c32> = col_mean.coerce();

            let mat = unsafe {
                mat::from_raw_parts::<f32>(
                    mat.as_ptr() as *const f32,
                    2 * m,
                    mat.ncols(),
                    1,
                    mat.col_stride() * 2,
                )
            };
            let col_mean =
                unsafe { col::from_raw_parts::<f32>(col_mean.as_ptr() as *const f32, 2 * m, 1) };

            col_varm_col_major_real::<f32>(tmp.as_mut(), mat, col_mean);
            for i in 0..m {
                out.write(i, tmp.read(2 * i) + tmp.read(2 * i + 1));
            }
        } else if const { E::IS_C64 } {
            let m = mat.nrows();

            let mut tmp = Col::<f64>::zeros(2 * m);
            let mut out: ColMut<'_, f64> = out.coerce();
            let mat: MatRef<'_, c64> = mat.coerce();
            let col_mean: ColRef<'_, c64> = col_mean.coerce();

            let mat = unsafe {
                mat::from_raw_parts::<f64>(
                    mat.as_ptr() as *const f64,
                    2 * m,
                    mat.ncols(),
                    1,
                    mat.col_stride() * 2,
                )
            };
            let col_mean =
                unsafe { col::from_raw_parts::<f64>(col_mean.as_ptr() as *const f64, 2 * m, 1) };

            col_varm_col_major_real::<f64>(tmp.as_mut(), mat, col_mean);
            for i in 0..m {
                out.write(i, tmp.read(2 * i) + tmp.read(2 * i + 1));
            }
        } else {
            panic!()
        }
    } else {
        let n = mat.ncols();
        let one_n1 = from_usize::<E::Real>(n - 1).faer_inv();

        out.fill_zero();
        for j in 0..n {
            zipped!(&mut out, col_mean, mat.col(j)).for_each(|unzipped!(mut out, mean, x)| {
                let diff = x.read().faer_sub(mean.read());
                out.write(out.read().faer_add(diff.faer_abs2()))
            });
        }
        zipped!(out).for_each(|unzipped!(mut x)| x.write(x.read().faer_scale_real(one_n1)));
    }
}

fn row_varm_propagate<E: ComplexField>(
    out: RowMut<'_, E::Real>,
    mat: MatRef<'_, E>,
    row_mean: RowRef<'_, E>,
) {
    col_varm_propagate(out.transpose_mut(), mat.transpose(), row_mean.transpose());
}

fn col_mean_ignore<E: ComplexField>(out: ColMut<'_, E>, mat: MatRef<'_, E>) {
    let mut mat = mat;
    let mut out = out;
    if mat.col_stride() < 0 {
        mat = mat.reverse_cols();
    };
    if mat.row_stride() < 0 {
        mat = mat.reverse_rows();
        out = out.reverse_rows_mut();
    };

    if mat.ncols() == 0 {
        out.fill(E::faer_nan());
        return;
    }

    let mat = if mat.col_stride() >= 0 {
        mat
    } else {
        mat.reverse_cols()
    };

    if mat.col_stride() == 1 {
        if const { E::IS_C32 } {
            col_mean_row_major_ignore_nan_c32(out.coerce(), mat.coerce())
        } else if const { E::IS_C64 } {
            col_mean_row_major_ignore_nan_c64(out.coerce(), mat.coerce())
        } else if const { E::IS_REAL } {
            col_mean_row_major_ignore_nan_real::<E::Real>(out.coerce(), mat.coerce())
        } else if const { E::IS_NUM_COMPLEX } {
            col_mean_row_major_ignore_nan_cplx::<E::Real>(out.coerce(), mat.coerce())
        } else {
            panic!()
        }
    } else {
        let m = mat.nrows();
        let n = mat.ncols();
        let mut valid_count = vec![0usize; m];

        out.fill_zero();
        if mat.row_stride() == 1 && out.row_stride() == 1 {
            struct Impl<'a, E: ComplexField> {
                out: ColMut<'a, E>,
                mat: MatRef<'a, E>,
                valid_count: &'a mut [usize],
            }

            impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    _ = simd;
                    let Self {
                        mut out,
                        mat,
                        valid_count,
                    } = self;

                    for j in 0..mat.ncols() {
                        let col = mat.col(j);
                        if col.row_stride() != 1 {
                            unsafe { core::hint::unreachable_unchecked() };
                        }
                        if out.row_stride() != 1 {
                            unsafe { core::hint::unreachable_unchecked() };
                        }

                        for i in 0..mat.nrows() {
                            let elem = unsafe { col.read_unchecked(i) };
                            let is_nan = elem.faer_is_nan();
                            *unsafe { valid_count.get_unchecked_mut(i) } += (!is_nan) as usize;
                            let acc = unsafe { out.read_unchecked(i) };
                            unsafe {
                                out.write_unchecked(
                                    i,
                                    if is_nan {
                                        acc.faer_add(E::faer_zero())
                                    } else {
                                        acc.faer_add(elem)
                                    },
                                )
                            };
                        }
                    }
                }
            }

            E::Simd::default().dispatch(Impl {
                out: out.rb_mut(),
                mat,
                valid_count: &mut valid_count,
            })
        } else {
            for j in 0..n {
                let col = mat.col(j);

                for i in 0..m {
                    let elem = unsafe { col.read_unchecked(i) };
                    let is_nan = elem.faer_is_nan();
                    *unsafe { valid_count.get_unchecked_mut(i) } += (!is_nan) as usize;
                    let acc = unsafe { out.read_unchecked(i) };
                    unsafe {
                        out.write_unchecked(i, if is_nan { acc } else { acc.faer_add(elem) })
                    };
                }
            }
        }

        for i in 0..m {
            out.write(
                i,
                out.read(i)
                    .faer_scale_real(from_usize::<E::Real>(valid_count[i]).faer_inv()),
            );
        }
    }
}

fn row_mean_ignore<E: ComplexField>(out: RowMut<'_, E>, mat: MatRef<'_, E>) {
    col_mean_ignore(out.transpose_mut(), mat.transpose())
}

fn col_varm_ignore<E: ComplexField>(
    out: ColMut<'_, E::Real>,
    mat: MatRef<'_, E>,
    col_mean: ColRef<'_, E>,
) {
    let mut out = out;
    if mat.ncols() == 0 {
        out.fill(E::Real::faer_nan());
        return;
    }

    let mat = if mat.col_stride() >= 0 {
        mat
    } else {
        mat.reverse_cols()
    };

    if mat.col_stride() == 1 {
        if const { E::IS_C32 } {
            col_varm_row_major_ignore_nan_c32(out.coerce(), mat.coerce(), col_mean.coerce())
        } else if const { E::IS_C64 } {
            col_varm_row_major_ignore_nan_c64(out.coerce(), mat.coerce(), col_mean.coerce())
        } else if const { E::IS_REAL } {
            col_varm_row_major_ignore_nan_real::<E::Real>(
                out.coerce(),
                mat.coerce(),
                col_mean.coerce(),
            )
        } else if const { E::IS_NUM_COMPLEX } {
            col_varm_row_major_ignore_nan_cplx::<E::Real>(
                out.coerce(),
                mat.coerce(),
                col_mean.coerce(),
            )
        } else {
            panic!()
        }
    } else {
        let m = mat.nrows();
        let n = mat.ncols();
        let mut valid_count = vec![0usize; m];

        out.fill_zero();
        if mat.row_stride() == 1 && out.row_stride() == 1 && col_mean.row_stride() == 1 {
            struct Impl<'a, E: ComplexField> {
                out: ColMut<'a, E::Real>,
                mat: MatRef<'a, E>,
                col_mean: ColRef<'a, E>,
                valid_count: &'a mut [usize],
            }

            impl<E: ComplexField> pulp::WithSimd for Impl<'_, E> {
                type Output = ();

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                    _ = simd;
                    let Self {
                        mut out,
                        mat,
                        col_mean,
                        valid_count,
                    } = self;

                    for j in 0..mat.ncols() {
                        let col = mat.col(j);
                        if col.row_stride() != 1 {
                            unsafe { core::hint::unreachable_unchecked() };
                        }
                        if col_mean.row_stride() != 1 {
                            unsafe { core::hint::unreachable_unchecked() };
                        }
                        if out.row_stride() != 1 {
                            unsafe { core::hint::unreachable_unchecked() };
                        }

                        for i in 0..mat.nrows() {
                            let elem = unsafe { col.read_unchecked(i) };
                            let diff = elem.faer_sub(unsafe { col_mean.read_unchecked(i) });
                            let is_nan = elem.faer_is_nan();
                            *unsafe { valid_count.get_unchecked_mut(i) } += (!is_nan) as usize;
                            let acc = unsafe { out.read_unchecked(i) };
                            unsafe {
                                out.write_unchecked(
                                    i,
                                    if is_nan {
                                        acc
                                    } else {
                                        acc.faer_add(diff.faer_abs2())
                                    },
                                )
                            };
                        }
                    }
                }
            }

            E::Simd::default().dispatch(Impl {
                out: out.rb_mut(),
                mat,
                col_mean,
                valid_count: &mut valid_count,
            })
        } else {
            for j in 0..n {
                for i in 0..m {
                    let elem = unsafe { mat.read_unchecked(i, j) };
                    let diff = elem.faer_sub(unsafe { col_mean.read_unchecked(i) });
                    let is_nan = elem.faer_is_nan();
                    *unsafe { valid_count.get_unchecked_mut(i) } += (!is_nan) as usize;
                    let acc = unsafe { out.read_unchecked(i) };
                    unsafe {
                        out.write_unchecked(
                            i,
                            if is_nan {
                                acc
                            } else {
                                acc.faer_add(diff.faer_abs2())
                            },
                        )
                    };
                }
            }
        }

        for i in 0..m {
            let non_nan_count = valid_count[i];
            let var = if non_nan_count == 0 {
                E::Real::faer_nan()
            } else if non_nan_count == 1 {
                E::Real::faer_zero()
            } else {
                out.read(i)
                    .faer_scale_real(from_usize::<E::Real>(non_nan_count - 1).faer_inv())
            };
            out.write(i, var);
        }
    }
}

fn row_varm_ignore<E: ComplexField>(
    out: RowMut<'_, E::Real>,
    mat: MatRef<'_, E>,
    row_mean: RowRef<'_, E>,
) {
    col_varm_ignore(out.transpose_mut(), mat.transpose(), row_mean.transpose())
}

/// Computes the mean of the columns of `mat` and stores the result in `out`.
#[track_caller]
pub fn col_mean<E: ComplexField>(out: ColMut<'_, E>, mat: MatRef<'_, E>, nan: NanHandling) {
    assert!(all(out.nrows() == mat.nrows()));

    match nan {
        NanHandling::Propagate => col_mean_propagate(out, mat),
        NanHandling::Ignore => col_mean_ignore(out, mat),
    }
}

/// Computes the mean of the rows of `mat` and stores the result in `out`.
#[track_caller]
pub fn row_mean<E: ComplexField>(out: RowMut<'_, E>, mat: MatRef<'_, E>, nan: NanHandling) {
    assert!(all(out.ncols() == mat.ncols()));

    match nan {
        NanHandling::Propagate => row_mean_propagate(out, mat),
        NanHandling::Ignore => row_mean_ignore(out, mat),
    }
}

/// Computes the variance of the columns of `mat` given their mean, and stores the result in `out`.
#[track_caller]
pub fn col_varm<E: ComplexField>(
    out: ColMut<'_, E::Real>,
    mat: MatRef<'_, E>,
    col_mean: ColRef<'_, E>,
    nan: NanHandling,
) {
    assert!(all(
        out.nrows() == mat.nrows(),
        col_mean.nrows() == mat.nrows()
    ));

    match nan {
        NanHandling::Propagate => col_varm_propagate(out, mat, col_mean),
        NanHandling::Ignore => col_varm_ignore(out, mat, col_mean),
    }
}

/// Computes the variance of the rows of `mat` given their mean, and stores the result in `out`.
#[track_caller]
pub fn row_varm<E: ComplexField>(
    out: RowMut<'_, E::Real>,
    mat: MatRef<'_, E>,
    row_mean: RowRef<'_, E>,
    nan: NanHandling,
) {
    assert!(all(
        out.ncols() == mat.ncols(),
        row_mean.ncols() == mat.ncols(),
    ));

    match nan {
        NanHandling::Propagate => row_varm_propagate(out, mat, row_mean),
        NanHandling::Ignore => row_varm_ignore(out, mat, row_mean),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use equator::assert;
    use num_complex::{Complex32, Complex64};

    #[test]
    fn test_meanvar() {
        let c32 = c32::new;
        let A = mat![
            [c32(1.2, 2.3), c32(3.4, 1.2)],
            [c32(1.7, -1.0), c32(-3.8, 1.95)],
        ];

        let mut row_mean = Row::zeros(A.ncols());
        let mut row_var = Row::zeros(A.ncols());
        super::row_mean_propagate(row_mean.as_mut(), A.as_ref());
        super::row_varm_propagate(row_var.as_mut(), A.as_ref(), row_mean.as_ref());

        let mut col_mean = Col::zeros(A.nrows());
        let mut col_var = Col::zeros(A.nrows());
        super::col_mean_propagate(col_mean.as_mut(), A.as_ref());
        super::col_varm_propagate(col_var.as_mut(), A.as_ref(), col_mean.as_ref());

        assert!(row_mean == row![(A[(0, 0)] + A[(1, 0)]) / 2.0, (A[(0, 1)] + A[(1, 1)]) / 2.0,]);
        assert!(
            row_var
                == row![
                    (A[(0, 0)] - row_mean[0]).faer_abs2() + (A[(1, 0)] - row_mean[0]).faer_abs2(),
                    (A[(0, 1)] - row_mean[1]).faer_abs2() + (A[(1, 1)] - row_mean[1]).faer_abs2(),
                ]
        );

        assert!(col_mean == col![(A[(0, 0)] + A[(0, 1)]) / 2.0, (A[(1, 0)] + A[(1, 1)]) / 2.0,]);
        assert!(
            col_var
                == col![
                    (A[(0, 0)] - col_mean[0]).faer_abs2() + (A[(0, 1)] - col_mean[0]).faer_abs2(),
                    (A[(1, 0)] - col_mean[1]).faer_abs2() + (A[(1, 1)] - col_mean[1]).faer_abs2(),
                ]
        );
    }

    #[test]
    fn test_meanvar_ignore_nan_nonan_c32() {
        let c32 = c32::new;
        let A = mat![
            [c32(1.2, 2.3), c32(3.4, 1.2)],
            [c32(1.7, -1.0), c32(-3.8, 1.95)],
        ];

        let mut row_mean = Row::zeros(A.ncols());
        let mut row_var = Row::zeros(A.ncols());
        super::row_mean_ignore(row_mean.as_mut(), A.as_ref());
        super::row_varm_ignore(row_var.as_mut(), A.as_ref(), row_mean.as_ref());

        let mut col_mean = Col::zeros(A.nrows());
        let mut col_var = Col::zeros(A.nrows());
        super::col_mean_ignore(col_mean.as_mut(), A.as_ref());
        super::col_varm_ignore(col_var.as_mut(), A.as_ref(), col_mean.as_ref());

        assert!(row_mean == row![(A[(0, 0)] + A[(1, 0)]) / 2.0, (A[(0, 1)] + A[(1, 1)]) / 2.0,]);
        assert!(
            row_var
                == row![
                    (A[(0, 0)] - row_mean[0]).faer_abs2() + (A[(1, 0)] - row_mean[0]).faer_abs2(),
                    (A[(0, 1)] - row_mean[1]).faer_abs2() + (A[(1, 1)] - row_mean[1]).faer_abs2(),
                ]
        );

        assert!(col_mean == col![(A[(0, 0)] + A[(0, 1)]) / 2.0, (A[(1, 0)] + A[(1, 1)]) / 2.0,]);
        assert!(
            col_var
                == col![
                    (A[(0, 0)] - col_mean[0]).faer_abs2() + (A[(0, 1)] - col_mean[0]).faer_abs2(),
                    (A[(1, 0)] - col_mean[1]).faer_abs2() + (A[(1, 1)] - col_mean[1]).faer_abs2(),
                ]
        );
    }

    #[test]
    fn test_meanvar_ignore_nan_yesnan_c32() {
        let c32 = c32::new;
        let nan = f32::NAN;
        let A = mat![
            [c32(1.2, nan), c32(3.4, 1.2)],
            [c32(1.7, -1.0), c32(-3.8, 1.95)],
        ];

        let mut row_mean = Row::zeros(A.ncols());
        let mut row_var = Row::zeros(A.ncols());
        super::row_mean_ignore(row_mean.as_mut(), A.as_ref());
        super::row_varm_ignore(row_var.as_mut(), A.as_ref(), row_mean.as_ref());

        let mut col_mean = Col::zeros(A.nrows());
        let mut col_var = Col::zeros(A.nrows());
        super::col_mean_ignore(col_mean.as_mut(), A.as_ref());
        super::col_varm_ignore(col_var.as_mut(), A.as_ref(), col_mean.as_ref());

        assert!(row_mean == row![A[(1, 0)] / 1.0, (A[(0, 1)] + A[(1, 1)]) / 2.0,]);
        assert!(
            row_var
                == row![
                    (A[(1, 0)] - row_mean[0]).faer_abs2(),
                    (A[(0, 1)] - row_mean[1]).faer_abs2() + (A[(1, 1)] - row_mean[1]).faer_abs2(),
                ]
        );

        assert!(col_mean == col![A[(0, 1)] / 1.0, (A[(1, 0)] + A[(1, 1)]) / 2.0,]);
        assert!(
            col_var
                == col![
                    (A[(0, 1)] - col_mean[0]).faer_abs2(),
                    (A[(1, 0)] - col_mean[1]).faer_abs2() + (A[(1, 1)] - col_mean[1]).faer_abs2(),
                ]
        );
    }

    #[test]
    fn test_meanvar_ignore_nan_nonan_c64() {
        let c64 = c64::new;
        let A = mat![
            [c64(1.2, 2.3), c64(3.4, 1.2)],
            [c64(1.7, -1.0), c64(-3.8, 1.95)],
        ];

        let mut row_mean = Row::zeros(A.ncols());
        let mut row_var = Row::zeros(A.ncols());
        super::row_mean_ignore(row_mean.as_mut(), A.as_ref());
        super::row_varm_ignore(row_var.as_mut(), A.as_ref(), row_mean.as_ref());

        let mut col_mean = Col::zeros(A.nrows());
        let mut col_var = Col::zeros(A.nrows());
        super::col_mean_ignore(col_mean.as_mut(), A.as_ref());
        super::col_varm_ignore(col_var.as_mut(), A.as_ref(), col_mean.as_ref());

        assert!(row_mean == row![(A[(0, 0)] + A[(1, 0)]) / 2.0, (A[(0, 1)] + A[(1, 1)]) / 2.0,]);
        assert!(
            row_var
                == row![
                    (A[(0, 0)] - row_mean[0]).faer_abs2() + (A[(1, 0)] - row_mean[0]).faer_abs2(),
                    (A[(0, 1)] - row_mean[1]).faer_abs2() + (A[(1, 1)] - row_mean[1]).faer_abs2(),
                ]
        );

        assert!(col_mean == col![(A[(0, 0)] + A[(0, 1)]) / 2.0, (A[(1, 0)] + A[(1, 1)]) / 2.0,]);
        assert!(
            col_var
                == col![
                    (A[(0, 0)] - col_mean[0]).faer_abs2() + (A[(0, 1)] - col_mean[0]).faer_abs2(),
                    (A[(1, 0)] - col_mean[1]).faer_abs2() + (A[(1, 1)] - col_mean[1]).faer_abs2(),
                ]
        );
    }

    #[test]
    fn test_meanvar_ignore_nan_yesnan_c64() {
        let c64 = c64::new;
        let nan = f64::NAN;
        let A = mat![
            [c64(1.2, nan), c64(3.4, 1.2)],
            [c64(1.7, -1.0), c64(-3.8, 1.95)],
        ];

        let mut row_mean = Row::zeros(A.ncols());
        let mut row_var = Row::zeros(A.ncols());
        super::row_mean_ignore(row_mean.as_mut(), A.as_ref());
        super::row_varm_ignore(row_var.as_mut(), A.as_ref(), row_mean.as_ref());

        let mut col_mean = Col::zeros(A.nrows());
        let mut col_var = Col::zeros(A.nrows());
        super::col_mean_ignore(col_mean.as_mut(), A.as_ref());
        super::col_varm_ignore(col_var.as_mut(), A.as_ref(), col_mean.as_ref());

        assert!(row_mean == row![A[(1, 0)] / 1.0, (A[(0, 1)] + A[(1, 1)]) / 2.0,]);
        assert!(
            row_var
                == row![
                    (A[(1, 0)] - row_mean[0]).faer_abs2(),
                    (A[(0, 1)] - row_mean[1]).faer_abs2() + (A[(1, 1)] - row_mean[1]).faer_abs2(),
                ]
        );

        assert!(col_mean == col![A[(0, 1)] / 1.0, (A[(1, 0)] + A[(1, 1)]) / 2.0,]);
        assert!(
            col_var
                == col![
                    (A[(0, 1)] - col_mean[0]).faer_abs2(),
                    (A[(1, 0)] - col_mean[1]).faer_abs2() + (A[(1, 1)] - col_mean[1]).faer_abs2(),
                ]
        );
    }

    #[test]
    fn test_meanvar_ignore_nan_nonan_cplx32() {
        let c32 = Complex32::new;
        let A = mat![
            [c32(1.2, 2.3), c32(3.4, 1.2)],
            [c32(1.7, -1.0), c32(-3.8, 1.95)],
        ];

        let mut row_mean = Row::zeros(A.ncols());
        let mut row_var = Row::zeros(A.ncols());
        super::row_mean_ignore(row_mean.as_mut(), A.as_ref());
        super::row_varm_ignore(row_var.as_mut(), A.as_ref(), row_mean.as_ref());

        let mut col_mean = Col::zeros(A.nrows());
        let mut col_var = Col::zeros(A.nrows());
        super::col_mean_ignore(col_mean.as_mut(), A.as_ref());
        super::col_varm_ignore(col_var.as_mut(), A.as_ref(), col_mean.as_ref());

        assert!(
            row_mean
                == row![
                    (A.read(0, 0) + A.read(1, 0)) / 2.0,
                    (A.read(0, 1) + A.read(1, 1)) / 2.0,
                ]
        );
        assert!(
            row_var
                == row![
                    (A.read(0, 0) - row_mean.read(0)).faer_abs2()
                        + (A.read(1, 0) - row_mean.read(0)).faer_abs2(),
                    (A.read(0, 1) - row_mean.read(1)).faer_abs2()
                        + (A.read(1, 1) - row_mean.read(1)).faer_abs2(),
                ]
        );

        assert!(
            col_mean
                == col![
                    (A.read(0, 0) + A.read(0, 1)) / 2.0,
                    (A.read(1, 0) + A.read(1, 1)) / 2.0,
                ]
        );
        assert!(
            col_var
                == col![
                    (A.read(0, 0) - col_mean.read(0)).faer_abs2()
                        + (A.read(0, 1) - col_mean.read(0)).faer_abs2(),
                    (A.read(1, 0) - col_mean.read(1)).faer_abs2()
                        + (A.read(1, 1) - col_mean.read(1)).faer_abs2(),
                ]
        );
    }

    #[test]
    fn test_meanvar_ignore_nan_yesnan_cplx32() {
        let c32 = Complex32::new;
        let nan = f32::NAN;
        let A = mat![
            [c32(1.2, nan), c32(3.4, 1.2)],
            [c32(1.7, -1.0), c32(-3.8, 1.95)],
        ];

        let mut row_mean = Row::zeros(A.ncols());
        let mut row_var = Row::zeros(A.ncols());
        super::row_mean_ignore(row_mean.as_mut(), A.as_ref());
        super::row_varm_ignore(row_var.as_mut(), A.as_ref(), row_mean.as_ref());

        let mut col_mean = Col::zeros(A.nrows());
        let mut col_var = Col::zeros(A.nrows());
        super::col_mean_ignore(col_mean.as_mut(), A.as_ref());
        super::col_varm_ignore(col_var.as_mut(), A.as_ref(), col_mean.as_ref());

        assert!(row_mean == row![A.read(1, 0) / 1.0, (A.read(0, 1) + A.read(1, 1)) / 2.0,]);
        assert!(
            row_var
                == row![
                    (A.read(1, 0) - row_mean.read(0)).faer_abs2(),
                    (A.read(0, 1) - row_mean.read(1)).faer_abs2()
                        + (A.read(1, 1) - row_mean.read(1)).faer_abs2(),
                ]
        );

        assert!(col_mean == col![A.read(0, 1) / 1.0, (A.read(1, 0) + A.read(1, 1)) / 2.0,]);
        assert!(
            col_var
                == col![
                    (A.read(0, 1) - col_mean.read(0)).faer_abs2(),
                    (A.read(1, 0) - col_mean.read(1)).faer_abs2()
                        + (A.read(1, 1) - col_mean.read(1)).faer_abs2(),
                ]
        );
    }

    #[test]
    fn test_meanvar_ignore_nan_nonan_cplx64() {
        let c64 = Complex64::new;
        let A = mat![
            [c64(1.2, 2.3), c64(3.4, 1.2)],
            [c64(1.7, -1.0), c64(-3.8, 1.95)],
        ];

        let mut row_mean = Row::zeros(A.ncols());
        let mut row_var = Row::zeros(A.ncols());
        super::row_mean_ignore(row_mean.as_mut(), A.as_ref());
        super::row_varm_ignore(row_var.as_mut(), A.as_ref(), row_mean.as_ref());

        let mut col_mean = Col::zeros(A.nrows());
        let mut col_var = Col::zeros(A.nrows());
        super::col_mean_ignore(col_mean.as_mut(), A.as_ref());
        super::col_varm_ignore(col_var.as_mut(), A.as_ref(), col_mean.as_ref());

        assert!(
            row_mean
                == row![
                    (A.read(0, 0) + A.read(1, 0)) / 2.0,
                    (A.read(0, 1) + A.read(1, 1)) / 2.0,
                ]
        );
        assert!(
            row_var
                == row![
                    (A.read(0, 0) - row_mean.read(0)).faer_abs2()
                        + (A.read(1, 0) - row_mean.read(0)).faer_abs2(),
                    (A.read(0, 1) - row_mean.read(1)).faer_abs2()
                        + (A.read(1, 1) - row_mean.read(1)).faer_abs2(),
                ]
        );

        assert!(
            col_mean
                == col![
                    (A.read(0, 0) + A.read(0, 1)) / 2.0,
                    (A.read(1, 0) + A.read(1, 1)) / 2.0,
                ]
        );
        assert!(
            col_var
                == col![
                    (A.read(0, 0) - col_mean.read(0)).faer_abs2()
                        + (A.read(0, 1) - col_mean.read(0)).faer_abs2(),
                    (A.read(1, 0) - col_mean.read(1)).faer_abs2()
                        + (A.read(1, 1) - col_mean.read(1)).faer_abs2(),
                ]
        );
    }

    #[test]
    fn test_meanvar_ignore_nan_yesnan_cplx64() {
        let c64 = Complex64::new;
        let nan = f64::NAN;
        let A = mat![
            [c64(1.2, nan), c64(3.4, 1.2)],
            [c64(1.7, -1.0), c64(-3.8, 1.95)],
        ];

        let mut row_mean = Row::zeros(A.ncols());
        let mut row_var = Row::zeros(A.ncols());
        super::row_mean_ignore(row_mean.as_mut(), A.as_ref());
        super::row_varm_ignore(row_var.as_mut(), A.as_ref(), row_mean.as_ref());

        let mut col_mean = Col::zeros(A.nrows());
        let mut col_var = Col::zeros(A.nrows());
        super::col_mean_ignore(col_mean.as_mut(), A.as_ref());
        super::col_varm_ignore(col_var.as_mut(), A.as_ref(), col_mean.as_ref());

        assert!(row_mean == row![A.read(1, 0) / 1.0, (A.read(0, 1) + A.read(1, 1)) / 2.0,]);
        assert!(
            row_var
                == row![
                    (A.read(1, 0) - row_mean.read(0)).faer_abs2(),
                    (A.read(0, 1) - row_mean.read(1)).faer_abs2()
                        + (A.read(1, 1) - row_mean.read(1)).faer_abs2(),
                ]
        );

        assert!(col_mean == col![A.read(0, 1) / 1.0, (A.read(1, 0) + A.read(1, 1)) / 2.0,]);
        assert!(
            col_var
                == col![
                    (A.read(0, 1) - col_mean.read(0)).faer_abs2(),
                    (A.read(1, 0) - col_mean.read(1)).faer_abs2()
                        + (A.read(1, 1) - col_mean.read(1)).faer_abs2(),
                ]
        );
    }
}
