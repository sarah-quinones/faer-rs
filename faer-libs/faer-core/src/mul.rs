//! Matrix multiplication.

use crate::{
    assert, c32, c64, group_helpers::*, transmute_unchecked, unzipped, zipped, ComplexField, Conj,
    Conjugate, DivCeil, MatMut, MatRef, Parallelism, SimdGroupFor,
};
use core::{iter::zip, marker::PhantomData, mem::MaybeUninit};
use faer_entity::{SimdCtx, *};
use pulp::Simd;
use reborrow::*;

#[doc(hidden)]
pub mod inner_prod {
    use super::*;
    use crate::assert;

    #[inline(always)]
    fn a_x_b_accumulate1<C: ConjTy, E: ComplexField, S: Simd>(
        simd: SimdFor<E, S>,
        conj: C,
        a: SliceGroup<E>,
        b: SliceGroup<E>,
        offset: pulp::Offset<E::SimdMask<S>>,
    ) -> SimdGroupFor<E, S> {
        let (a_head, a_body, a_tail) = simd.as_aligned_simd(a, offset);
        let (b_head, b_body, b_tail) = simd.as_aligned_simd(b, offset);
        let zero = simd.splat(E::faer_zero());
        let mut acc0 = simd.conditional_conj_mul(conj, a_head.read_or(zero), b_head.read_or(zero));

        let a_body1 = a_body;
        let b_body1 = b_body;
        for (a, b) in zip(a_body1.into_ref_iter(), b_body1.into_ref_iter()) {
            acc0 = simd.conditional_conj_mul_add_e(conj, a.read_or(zero), b.read_or(zero), acc0);
        }
        simd.conditional_conj_mul_add_e(conj, a_tail.read_or(zero), b_tail.read_or(zero), acc0)
    }

    #[inline(always)]
    fn a_x_b_accumulate2<C: ConjTy, E: ComplexField, S: Simd>(
        simd: SimdFor<E, S>,
        conj: C,
        a: SliceGroup<E>,
        b: SliceGroup<E>,
        offset: pulp::Offset<E::SimdMask<S>>,
    ) -> SimdGroupFor<E, S> {
        let (a_head, a_body, a_tail) = simd.as_aligned_simd(a, offset);
        let (b_head, b_body, b_tail) = simd.as_aligned_simd(b, offset);
        let zero = simd.splat(E::faer_zero());
        let mut acc0 = simd.conditional_conj_mul(conj, a_head.read_or(zero), b_head.read_or(zero));
        let mut acc1 = zero;

        let (a_body2, a_body1) = a_body.as_arrays::<2>();
        let (b_body2, b_body1) = b_body.as_arrays::<2>();
        for ([a0, a1], [b0, b1]) in zip(
            a_body2.into_ref_iter().map(RefGroup::unzip),
            b_body2.into_ref_iter().map(RefGroup::unzip),
        ) {
            acc0 = simd.conditional_conj_mul_add_e(conj, a0.read_or(zero), b0.read_or(zero), acc0);
            acc1 = simd.conditional_conj_mul_add_e(conj, a1.read_or(zero), b1.read_or(zero), acc1);
        }
        for (a, b) in zip(a_body1.into_ref_iter(), b_body1.into_ref_iter()) {
            acc0 = simd.conditional_conj_mul_add_e(conj, a.read_or(zero), b.read_or(zero), acc0);
        }
        acc0 =
            simd.conditional_conj_mul_add_e(conj, a_tail.read_or(zero), b_tail.read_or(zero), acc0);
        simd.add(acc0, acc1)
    }

    #[inline(always)]
    fn a_x_b_accumulate4<C: ConjTy, E: ComplexField, S: Simd>(
        simd: SimdFor<E, S>,
        conj: C,
        a: SliceGroup<E>,
        b: SliceGroup<E>,
        offset: pulp::Offset<E::SimdMask<S>>,
    ) -> SimdGroupFor<E, S> {
        let (a_head, a_body, a_tail) = simd.as_aligned_simd(a, offset);
        let (b_head, b_body, b_tail) = simd.as_aligned_simd(b, offset);
        let zero = simd.splat(E::faer_zero());
        let mut acc0 = simd.conditional_conj_mul(conj, a_head.read_or(zero), b_head.read_or(zero));
        let mut acc1 = zero;
        let mut acc2 = zero;
        let mut acc3 = zero;

        let (a_body4, a_body1) = a_body.as_arrays::<4>();
        let (b_body4, b_body1) = b_body.as_arrays::<4>();
        for ([a0, a1, a2, a3], [b0, b1, b2, b3]) in zip(
            a_body4.into_ref_iter().map(RefGroup::unzip),
            b_body4.into_ref_iter().map(RefGroup::unzip),
        ) {
            acc0 = simd.conditional_conj_mul_add_e(conj, a0.read_or(zero), b0.read_or(zero), acc0);
            acc1 = simd.conditional_conj_mul_add_e(conj, a1.read_or(zero), b1.read_or(zero), acc1);
            acc2 = simd.conditional_conj_mul_add_e(conj, a2.read_or(zero), b2.read_or(zero), acc2);
            acc3 = simd.conditional_conj_mul_add_e(conj, a3.read_or(zero), b3.read_or(zero), acc3);
        }
        for (a, b) in zip(a_body1.into_ref_iter(), b_body1.into_ref_iter()) {
            acc0 = simd.conditional_conj_mul_add_e(conj, a.read_or(zero), b.read_or(zero), acc0);
        }
        acc0 =
            simd.conditional_conj_mul_add_e(conj, a_tail.read_or(zero), b_tail.read_or(zero), acc0);
        simd.add(simd.add(acc0, acc1), simd.add(acc2, acc3))
    }

    #[inline(always)]
    fn a_x_b_accumulate8<C: ConjTy, E: ComplexField, S: Simd>(
        simd: SimdFor<E, S>,
        conj: C,
        a: SliceGroup<E>,
        b: SliceGroup<E>,
        offset: pulp::Offset<E::SimdMask<S>>,
    ) -> SimdGroupFor<E, S> {
        let (a_head, a_body, a_tail) = simd.as_aligned_simd(a, offset);
        let (b_head, b_body, b_tail) = simd.as_aligned_simd(b, offset);
        let zero = simd.splat(E::faer_zero());
        let mut acc0 = simd.conditional_conj_mul(conj, a_head.read_or(zero), b_head.read_or(zero));
        let mut acc1 = zero;
        let mut acc2 = zero;
        let mut acc3 = zero;
        let mut acc4 = zero;
        let mut acc5 = zero;
        let mut acc6 = zero;
        let mut acc7 = zero;

        let (a_body8, a_body1) = a_body.as_arrays::<8>();
        let (b_body8, b_body1) = b_body.as_arrays::<8>();
        for ([a0, a1, a2, a3, a4, a5, a6, a7], [b0, b1, b2, b3, b4, b5, b6, b7]) in zip(
            a_body8.into_ref_iter().map(RefGroup::unzip),
            b_body8.into_ref_iter().map(RefGroup::unzip),
        ) {
            acc0 = simd.conditional_conj_mul_add_e(conj, a0.read_or(zero), b0.read_or(zero), acc0);
            acc1 = simd.conditional_conj_mul_add_e(conj, a1.read_or(zero), b1.read_or(zero), acc1);
            acc2 = simd.conditional_conj_mul_add_e(conj, a2.read_or(zero), b2.read_or(zero), acc2);
            acc3 = simd.conditional_conj_mul_add_e(conj, a3.read_or(zero), b3.read_or(zero), acc3);
            acc4 = simd.conditional_conj_mul_add_e(conj, a4.read_or(zero), b4.read_or(zero), acc4);
            acc5 = simd.conditional_conj_mul_add_e(conj, a5.read_or(zero), b5.read_or(zero), acc5);
            acc6 = simd.conditional_conj_mul_add_e(conj, a6.read_or(zero), b6.read_or(zero), acc6);
            acc7 = simd.conditional_conj_mul_add_e(conj, a7.read_or(zero), b7.read_or(zero), acc7);
        }
        for (a, b) in zip(a_body1.into_ref_iter(), b_body1.into_ref_iter()) {
            acc0 = simd.conditional_conj_mul_add_e(conj, a.read_or(zero), b.read_or(zero), acc0);
        }
        acc0 =
            simd.conditional_conj_mul_add_e(conj, a_tail.read_or(zero), b_tail.read_or(zero), acc0);
        simd.add(
            simd.add(simd.add(acc0, acc1), simd.add(acc2, acc3)),
            simd.add(simd.add(acc4, acc5), simd.add(acc6, acc7)),
        )
    }

    #[inline(always)]
    pub fn with_simd_and_offset<C: ConjTy, E: ComplexField, S: Simd>(
        simd: SimdFor<E, S>,
        conj: C,
        a: SliceGroup<E>,
        b: SliceGroup<E>,
        offset: pulp::Offset<E::SimdMask<S>>,
    ) -> E {
        {
            let prologue = if E::N_COMPONENTS == 1 {
                a_x_b_accumulate8(simd, conj, a, b, offset)
            } else if E::N_COMPONENTS == 2 {
                a_x_b_accumulate4(simd, conj, a, b, offset)
            } else if E::N_COMPONENTS == 4 {
                a_x_b_accumulate2(simd, conj, a, b, offset)
            } else {
                a_x_b_accumulate1(simd, conj, a, b, offset)
            };

            simd.reduce_add(simd.rotate_left(prologue, offset.rotate_left_amount()))
        }
    }

    pub struct Impl<'a, C: ConjTy, E: ComplexField> {
        pub a: SliceGroup<'a, E>,
        pub b: SliceGroup<'a, E>,
        pub conj: C,
    }

    impl<C: ConjTy, E: ComplexField> pulp::WithSimd for Impl<'_, C, E> {
        type Output = E;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let simd = SimdFor::new(simd);
            with_simd_and_offset(simd, self.conj, self.a, self.b, simd.align_offset(self.a))
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn inner_prod_with_conj_arch<E: ComplexField>(
        arch: E::Simd,
        lhs: MatRef<'_, E>,
        conj_lhs: Conj,
        rhs: MatRef<'_, E>,
        conj_rhs: Conj,
    ) -> E {
        assert!(all(
            lhs.nrows() == rhs.nrows(),
            lhs.ncols() == 1,
            rhs.ncols() == 1,
        ));
        let nrows = lhs.nrows();
        let mut a = lhs;
        let mut b = rhs;
        if a.row_stride() < 0 {
            a = a.reverse_rows();
            b = b.reverse_rows();
        }

        let res = if a.row_stride() == 1 && b.row_stride() == 1 {
            let a = SliceGroup::<'_, E>::new(a.try_get_contiguous_col(0));
            let b = SliceGroup::<'_, E>::new(b.try_get_contiguous_col(0));
            if conj_lhs == conj_rhs {
                arch.dispatch(Impl { a, b, conj: NoConj })
            } else {
                arch.dispatch(Impl {
                    a,
                    b,
                    conj: YesConj,
                })
            }
        } else {
            crate::constrained::Size::with2(
                nrows,
                1,
                #[inline(always)]
                |nrows, ncols| {
                    let zero_idx = ncols.check(0);

                    let a = crate::constrained::MatRef::new(a, nrows, ncols);
                    let b = crate::constrained::MatRef::new(b, nrows, ncols);
                    let mut acc = E::faer_zero();
                    if conj_lhs == conj_rhs {
                        for i in nrows.indices() {
                            acc =
                                acc.faer_add(E::faer_mul(a.read(i, zero_idx), b.read(i, zero_idx)));
                        }
                    } else {
                        for i in nrows.indices() {
                            acc = acc.faer_add(E::faer_mul(
                                a.read(i, zero_idx).faer_conj(),
                                b.read(i, zero_idx),
                            ));
                        }
                    }
                    acc
                },
            )
        };

        match conj_rhs {
            Conj::Yes => res.faer_conj(),
            Conj::No => res,
        }
    }

    #[inline]
    #[track_caller]
    pub fn inner_prod_with_conj<E: ComplexField>(
        lhs: MatRef<'_, E>,
        conj_lhs: Conj,
        rhs: MatRef<'_, E>,
        conj_rhs: Conj,
    ) -> E {
        inner_prod_with_conj_arch(E::Simd::default(), lhs, conj_lhs, rhs, conj_rhs)
    }
}

#[doc(hidden)]
pub mod matvec_rowmajor {
    use super::*;
    use crate::assert;

    fn matvec_with_conj_impl<E: ComplexField>(
        acc: MatMut<'_, E>,
        a: MatRef<'_, E>,
        conj_a: Conj,
        b: MatRef<'_, E>,
        conj_b: Conj,
        alpha: Option<E>,
        beta: E,
    ) {
        let m = a.nrows();
        let n = a.ncols();

        assert!(all(
            b.nrows() == n,
            b.ncols() == 1,
            acc.nrows() == m,
            acc.ncols() == 1,
            a.col_stride() == 1,
            b.row_stride() == 1,
        ));

        let mut acc = acc;

        for i in 0..m {
            let a = a.submatrix(i, 0, 1, n);
            let res = inner_prod::inner_prod_with_conj(a.transpose(), conj_a, b, conj_b);
            match alpha {
                Some(alpha) => acc.write(
                    i,
                    0,
                    E::faer_add(alpha.faer_mul(acc.read(i, 0)), beta.faer_mul(res)),
                ),
                None => acc.write(i, 0, beta.faer_mul(res)),
            }
        }
    }

    pub fn matvec_with_conj<E: ComplexField>(
        acc: MatMut<'_, E>,
        lhs: MatRef<'_, E>,
        conj_lhs: Conj,
        rhs: MatRef<'_, E>,
        conj_rhs: Conj,
        alpha: Option<E>,
        beta: E,
    ) {
        if rhs.row_stride() == 1 {
            matvec_with_conj_impl(acc, lhs, conj_lhs, rhs, conj_rhs, alpha, beta);
        } else {
            matvec_with_conj_impl(
                acc,
                lhs,
                conj_lhs,
                rhs.to_owned().as_ref(),
                conj_rhs,
                alpha,
                beta,
            );
        }
    }
}

#[doc(hidden)]
pub mod matvec_colmajor {
    use super::*;
    use crate::assert;

    pub struct Impl<'a, C: ConjTy, E: ComplexField> {
        pub conj: C,
        pub acc: SliceGroupMut<'a, E>,
        pub a: SliceGroup<'a, E>,
        pub b: E,
    }

    #[inline(always)]
    pub fn with_simd_and_offset<C: ConjTy, E: ComplexField, S: Simd>(
        simd: SimdFor<E, S>,
        conj: C,
        acc: SliceGroupMut<'_, E>,
        a: SliceGroup<'_, E>,
        b: E,
        offset: pulp::Offset<SimdMaskFor<E, S>>,
    ) {
        let (a_head, a_body, a_tail) = simd.as_aligned_simd(a, offset);
        let (acc_head, acc_body, acc_tail) = simd.as_aligned_simd_mut(acc, offset);
        let b = simd.splat(b);

        #[inline(always)]
        pub fn process<C: ConjTy, E: ComplexField, S: Simd>(
            simd: SimdFor<E, S>,
            conj: C,
            mut acc: impl Write<Output = SimdGroupFor<E, S>>,
            a: impl Read<Output = SimdGroupFor<E, S>>,
            b: SimdGroupFor<E, S>,
        ) {
            acc.write(simd.conditional_conj_mul_add_e(
                conj,
                a.read_or(simd.splat(E::faer_zero())),
                b,
                acc.read_or(simd.splat(E::faer_zero())),
            ))
        }

        process(simd, conj, acc_head, a_head, b);
        for (acc, a) in acc_body.into_mut_iter().zip(a_body.into_ref_iter()) {
            process(simd, conj, acc, a, b);
        }
        process(simd, conj, acc_tail, a_tail, b);
    }

    impl<C: ConjTy, E: ComplexField> pulp::WithSimd for Impl<'_, C, E> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let simd = SimdFor::new(simd);
            with_simd_and_offset(
                simd,
                self.conj,
                self.acc,
                self.a,
                self.b,
                simd.align_offset(self.a),
            )
        }
    }

    fn matvec_with_conj_impl<E: ComplexField>(
        acc: MatMut<'_, E>,
        a: MatRef<'_, E>,
        conj_a: Conj,
        b: MatRef<'_, E>,
        conj_b: Conj,
        beta: E,
    ) {
        let m = a.nrows();
        let n = a.ncols();

        assert!(all(
            b.nrows() == n,
            b.ncols() == 1,
            acc.nrows() == m,
            acc.ncols() == 1,
            a.row_stride() == 1,
            acc.row_stride() == 1,
        ));

        let mut acc = SliceGroupMut::<'_, E>::new(acc.try_get_contiguous_col_mut(0));

        let arch = E::Simd::default();
        for j in 0..n {
            let acc = acc.rb_mut();
            let a = SliceGroup::<'_, E>::new(a.try_get_contiguous_col(j));
            let b = b.read(j, 0);
            let b = match conj_b {
                Conj::Yes => b.faer_conj(),
                Conj::No => b,
            };
            let b = b.faer_mul(beta);

            match conj_a {
                Conj::Yes => arch.dispatch(Impl {
                    conj: YesConj,
                    acc,
                    a,
                    b,
                }),
                Conj::No => arch.dispatch(Impl {
                    conj: NoConj,
                    acc,
                    a,
                    b,
                }),
            }
        }
    }

    pub fn matvec_with_conj<E: ComplexField>(
        acc: MatMut<'_, E>,
        lhs: MatRef<'_, E>,
        conj_lhs: Conj,
        rhs: MatRef<'_, E>,
        conj_rhs: Conj,
        alpha: Option<E>,
        beta: E,
    ) {
        let m = acc.nrows();
        let mut acc = acc;
        if acc.row_stride() == 1 {
            match alpha {
                Some(alpha) if alpha == E::faer_one() => {}
                Some(alpha) => {
                    for i in 0..m {
                        acc.write(i, 0, acc.read(i, 0).faer_mul(alpha));
                    }
                }
                None => {
                    for i in 0..m {
                        acc.write(i, 0, E::faer_zero());
                    }
                }
            }

            matvec_with_conj_impl(acc, lhs, conj_lhs, rhs, conj_rhs, beta);
        } else {
            let mut tmp = crate::Mat::<E>::zeros(m, 1);
            matvec_with_conj_impl(tmp.as_mut(), lhs, conj_lhs, rhs, conj_rhs, beta);
            match alpha {
                Some(alpha) => {
                    for i in 0..m {
                        acc.write(
                            i,
                            0,
                            (acc.read(i, 0).faer_mul(alpha)).faer_add(tmp.read(i, 0)),
                        )
                    }
                }
                None => {
                    for i in 0..m {
                        acc.write(i, 0, tmp.read(i, 0))
                    }
                }
            }
        }
    }
}

#[doc(hidden)]
pub mod matvec {
    use super::*;

    pub fn matvec_with_conj<E: ComplexField>(
        acc: MatMut<'_, E>,
        lhs: MatRef<'_, E>,
        conj_lhs: Conj,
        rhs: MatRef<'_, E>,
        conj_rhs: Conj,
        alpha: Option<E>,
        beta: E,
    ) {
        let mut acc = acc;
        let mut a = lhs;
        let mut b = rhs;

        if a.row_stride() < 0 {
            a = a.reverse_rows();
            acc = acc.reverse_rows_mut();
        }
        if a.col_stride() < 0 {
            a = a.reverse_cols();
            b = b.reverse_rows();
        }

        if a.row_stride() == 1 {
            return matvec_colmajor::matvec_with_conj(acc, a, conj_lhs, b, conj_rhs, alpha, beta);
        }
        if a.col_stride() == 1 {
            return matvec_rowmajor::matvec_with_conj(acc, a, conj_lhs, b, conj_rhs, alpha, beta);
        }

        let m = a.nrows();
        let n = a.ncols();

        match alpha {
            Some(alpha) => {
                for i in 0..m {
                    acc.write(i, 0, acc.read(i, 0).faer_mul(alpha));
                }
            }
            None => {
                for i in 0..m {
                    acc.write(i, 0, E::faer_zero());
                }
            }
        }

        for j in 0..n {
            let b = b.read(j, 0);
            let b = match conj_rhs {
                Conj::Yes => b.faer_conj(),
                Conj::No => b,
            };
            let b = b.faer_mul(beta);
            for i in 0..m {
                let mul = a.read(i, j).faer_mul(b);
                acc.write(i, 0, acc.read(i, 0).faer_add(mul));
            }
        }
    }
}

#[doc(hidden)]
pub mod outer_prod {
    use super::*;
    use crate::assert;

    pub struct Impl<'a, C: ConjTy, E: ComplexField> {
        pub conj: C,
        pub acc: SliceGroupMut<'a, E>,
        pub a: SliceGroup<'a, E>,
        pub b: E,
        pub alpha: Option<E>,
    }

    #[inline(always)]
    pub fn with_simd_and_offset<C: ConjTy, E: ComplexField, S: Simd>(
        simd: SimdFor<E, S>,
        conj: C,
        acc: SliceGroupMut<'_, E>,
        a: SliceGroup<'_, E>,
        b: E,
        alpha: Option<E>,
        offset: pulp::Offset<SimdMaskFor<E, S>>,
    ) {
        match alpha {
            Some(alpha) => {
                if alpha == E::faer_one() {
                    return matvec_colmajor::with_simd_and_offset(simd, conj, acc, a, b, offset);
                }

                let (a_head, a_body, a_tail) = simd.as_aligned_simd(a, offset);
                let (acc_head, acc_body, acc_tail) = simd.as_aligned_simd_mut(acc, offset);
                let b = simd.splat(b);
                let alpha = simd.splat(alpha);

                #[inline(always)]
                pub fn process<C: ConjTy, E: ComplexField, S: Simd>(
                    simd: SimdFor<E, S>,
                    conj: C,
                    mut acc: impl Write<Output = SimdGroupFor<E, S>>,
                    a: impl Read<Output = SimdGroupFor<E, S>>,
                    b: SimdGroupFor<E, S>,
                    alpha: SimdGroupFor<E, S>,
                ) {
                    acc.write(simd.conditional_conj_mul_add_e(
                        conj,
                        a.read_or(simd.splat(E::faer_zero())),
                        b,
                        simd.mul(alpha, acc.read_or(simd.splat(E::faer_zero()))),
                    ))
                }

                process(simd, conj, acc_head, a_head, b, alpha);
                for (acc, a) in acc_body.into_mut_iter().zip(a_body.into_ref_iter()) {
                    process(simd, conj, acc, a, b, alpha);
                }
                process(simd, conj, acc_tail, a_tail, b, alpha);
            }
            None => {
                let (a_head, a_body, a_tail) = simd.as_aligned_simd(a, offset);
                let (acc_head, acc_body, acc_tail) = simd.as_aligned_simd_mut(acc, offset);
                let b = simd.splat(b);

                #[inline(always)]
                pub fn process<C: ConjTy, E: ComplexField, S: Simd>(
                    simd: SimdFor<E, S>,
                    conj: C,
                    mut acc: impl Write<Output = SimdGroupFor<E, S>>,
                    a: impl Read<Output = SimdGroupFor<E, S>>,
                    b: SimdGroupFor<E, S>,
                ) {
                    acc.write(simd.conditional_conj_mul(
                        conj,
                        a.read_or(simd.splat(E::faer_zero())),
                        b,
                    ))
                }

                process(simd, conj, acc_head, a_head, b);
                for (acc, a) in acc_body.into_mut_iter().zip(a_body.into_ref_iter()) {
                    process(simd, conj, acc, a, b);
                }
                process(simd, conj, acc_tail, a_tail, b);
            }
        }
    }

    impl<C: ConjTy, E: ComplexField> pulp::WithSimd for Impl<'_, C, E> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let simd = SimdFor::new(simd);
            with_simd_and_offset(
                simd,
                self.conj,
                self.acc,
                self.a,
                self.b,
                self.alpha,
                simd.align_offset(self.a),
            )
        }
    }

    fn outer_prod_with_conj_impl<E: ComplexField>(
        acc: MatMut<'_, E>,
        a: MatRef<'_, E>,
        conj_a: Conj,
        b: MatRef<'_, E>,
        conj_b: Conj,
        alpha: Option<E>,
        beta: E,
    ) {
        let m = acc.nrows();
        let n = acc.ncols();

        assert!(all(
            a.nrows() == m,
            a.ncols() == 1,
            b.nrows() == n,
            b.ncols() == 1,
            acc.row_stride() == 1,
            a.row_stride() == 1,
        ));

        let mut acc = acc;

        let arch = E::Simd::default();

        let a = SliceGroup::new(a.try_get_contiguous_col(0));

        for j in 0..n {
            let acc = SliceGroupMut::new(acc.rb_mut().try_get_contiguous_col_mut(j));
            let b = b.read(j, 0);
            let b = match conj_b {
                Conj::Yes => b.faer_conj(),
                Conj::No => b,
            };
            let b = b.faer_mul(beta);
            match conj_a {
                Conj::Yes => arch.dispatch(Impl {
                    conj: YesConj,
                    acc,
                    a,
                    b,
                    alpha,
                }),
                Conj::No => arch.dispatch(Impl {
                    conj: NoConj,
                    acc,
                    a,
                    b,
                    alpha,
                }),
            }
        }
    }

    pub fn outer_prod_with_conj<E: ComplexField>(
        acc: MatMut<'_, E>,
        lhs: MatRef<'_, E>,
        conj_lhs: Conj,
        rhs: MatRef<'_, E>,
        conj_rhs: Conj,
        alpha: Option<E>,
        beta: E,
    ) {
        let mut acc = acc;
        let mut a = lhs;
        let mut b = rhs;
        let mut conj_a = conj_lhs;
        let mut conj_b = conj_rhs;

        if acc.row_stride() < 0 {
            acc = acc.reverse_rows_mut();
            a = a.reverse_rows();
        }
        if acc.col_stride() < 0 {
            acc = acc.reverse_cols_mut();
            b = b.reverse_rows();
        }

        if acc.row_stride() > a.col_stride() {
            acc = acc.transpose_mut();
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut conj_a, &mut conj_b);
        }

        if acc.row_stride() == 1 {
            if a.row_stride() == 1 {
                outer_prod_with_conj_impl(acc, a, conj_a, b, conj_b, alpha, beta);
            } else {
                outer_prod_with_conj_impl(
                    acc,
                    a.to_owned().as_ref(),
                    conj_a,
                    b,
                    conj_b,
                    alpha,
                    beta,
                );
            }
        } else {
            let m = acc.nrows();
            let n = acc.ncols();
            match alpha {
                Some(alpha) => {
                    for j in 0..n {
                        let b = b.read(j, 0);
                        let b = match conj_b {
                            Conj::Yes => b.faer_conj(),
                            Conj::No => b,
                        };
                        let b = b.faer_mul(beta);
                        match conj_a {
                            Conj::Yes => {
                                for i in 0..m {
                                    let ab = a.read(i, 0).faer_conj().faer_mul(b);
                                    acc.write(
                                        i,
                                        j,
                                        E::faer_add(acc.read(i, j).faer_mul(alpha), ab),
                                    );
                                }
                            }
                            Conj::No => {
                                for i in 0..m {
                                    let ab = a.read(i, 0).faer_mul(b);
                                    acc.write(
                                        i,
                                        j,
                                        E::faer_add(acc.read(i, j).faer_mul(alpha), ab),
                                    );
                                }
                            }
                        }
                    }
                }
                None => {
                    for j in 0..n {
                        let b = b.read(j, 0);
                        let b = match conj_b {
                            Conj::Yes => b.faer_conj(),
                            Conj::No => b,
                        };
                        let b = b.faer_mul(beta);
                        match conj_a {
                            Conj::Yes => {
                                for i in 0..m {
                                    acc.write(i, j, a.read(i, 0).faer_conj().faer_mul(b));
                                }
                            }
                            Conj::No => {
                                for i in 0..m {
                                    acc.write(i, j, a.read(i, 0).faer_mul(b));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

const NC: usize = 2048;
const MC: usize = 48;
const KC: usize = 64;

struct SimdLaneCount<E: ComplexField> {
    __marker: PhantomData<fn() -> E>,
}
impl<E: ComplexField> pulp::WithSimd for SimdLaneCount<E> {
    type Output = usize;

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let _ = simd;
        core::mem::size_of::<SimdUnitFor<E, S>>() / core::mem::size_of::<UnitFor<E>>()
    }
}

struct Ukr<'a, const MR_DIV_N: usize, const NR: usize, CB: ConjTy, E: ComplexField> {
    conj_b: CB,
    acc: MatMut<'a, E>,
    a: MatRef<'a, E>,
    b: MatRef<'a, E>,
}

impl<const MR_DIV_N: usize, const NR: usize, CB: ConjTy, E: ComplexField> pulp::WithSimd
    for Ukr<'_, MR_DIV_N, NR, CB, E>
{
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self {
            mut acc,
            a,
            b,
            conj_b,
        } = self;
        let lane_count =
            core::mem::size_of::<SimdUnitFor<E, S>>() / core::mem::size_of::<UnitFor<E>>();

        let mr = MR_DIV_N * lane_count;
        let nr = NR;

        assert!(all(
            acc.nrows() == mr,
            acc.ncols() == nr,
            a.nrows() == mr,
            b.ncols() == nr,
            a.ncols() == b.nrows(),
            a.row_stride() == 1,
            b.row_stride() == 1,
            acc.row_stride() == 1
        ));

        let k = a.ncols();
        let mut local_acc = [[E::faer_simd_splat(simd, E::faer_zero()); MR_DIV_N]; NR];
        let simd = SimdFor::<E, S>::new(simd);

        unsafe {
            let mut one_iter = {
                #[inline(always)]
                |depth| {
                    let a = a.ptr_inbounds_at(0, depth);

                    let mut a_uninit = [MaybeUninit::<SimdGroupFor<E, S>>::uninit(); MR_DIV_N];

                    let mut i = 0usize;
                    loop {
                        if i == MR_DIV_N {
                            break;
                        }
                        a_uninit[i] = MaybeUninit::new(into_copy::<E, _>(E::faer_map(
                            E::faer_copy(&a),
                            #[inline(always)]
                            |ptr| *(ptr.add(i * lane_count) as *const SimdUnitFor<E, S>),
                        )));
                        i += 1;
                    }
                    let a: [SimdGroupFor<E, S>; MR_DIV_N] = transmute_unchecked(a_uninit);

                    let mut j = 0usize;
                    loop {
                        if j == NR {
                            break;
                        }
                        let b = simd.splat(E::faer_from_units(E::faer_map(
                            b.ptr_at(depth, j),
                            #[inline(always)]
                            |ptr| *ptr,
                        )));
                        let mut i = 0;
                        loop {
                            if i == MR_DIV_N {
                                break;
                            }
                            let local_acc = &mut local_acc[j][i];
                            *local_acc =
                                simd.conditional_conj_mul_add_e(conj_b, b, a[i], *local_acc);
                            i += 1;
                        }
                        j += 1;
                    }
                }
            };

            let mut depth = 0;
            while depth < k / 4 * 4 {
                one_iter(depth);
                one_iter(depth + 1);
                one_iter(depth + 2);
                one_iter(depth + 3);
                depth += 4;
            }
            while depth < k {
                one_iter(depth);
                depth += 1;
            }

            let mut j = 0usize;
            loop {
                if j == NR {
                    break;
                }
                let mut i = 0usize;
                loop {
                    if i == MR_DIV_N {
                        break;
                    }
                    let acc = acc.rb_mut().ptr_inbounds_at_mut(i * lane_count, j);
                    let mut acc_value = into_copy::<E, _>(E::faer_map(E::faer_copy(&acc), |acc| {
                        *(acc as *const SimdUnitFor<E, S>)
                    }));
                    acc_value = simd.add(acc_value, local_acc[j][i]);
                    E::faer_map(
                        E::faer_zip(acc, from_copy::<E, _>(acc_value)),
                        #[inline(always)]
                        |(acc, new_acc)| *(acc as *mut SimdUnitFor<E, S>) = new_acc,
                    );
                    i += 1;
                }
                j += 1;
            }
        }
    }
}

#[inline]
fn min(a: usize, b: usize) -> usize {
    a.min(b)
}

struct MicroKernelShape<E: ComplexField> {
    __marker: PhantomData<fn() -> E>,
}

impl<E: ComplexField> MicroKernelShape<E> {
    const SHAPE: (usize, usize) = {
        if E::N_COMPONENTS <= 2 {
            (2, 2)
        } else if E::N_COMPONENTS == 4 {
            (2, 1)
        } else {
            (1, 1)
        }
    };

    const MAX_MR_DIV_N: usize = Self::SHAPE.0;
    const MAX_NR: usize = Self::SHAPE.1;

    const IS_2X2: bool = Self::MAX_MR_DIV_N == 2 && Self::MAX_NR == 2;
    const IS_2X1: bool = Self::MAX_MR_DIV_N == 2 && Self::MAX_NR == 1;
    const IS_1X1: bool = Self::MAX_MR_DIV_N == 2 && Self::MAX_NR == 1;
}

/// acc += a * maybe_conj(b)
///
/// acc, a, b are colmajor
/// m is a multiple of simd lane count
fn matmul_with_conj_impl<E: ComplexField>(
    acc: MatMut<'_, E>,
    a: MatRef<'_, E>,
    b: MatRef<'_, E>,
    conj_b: Conj,
    parallelism: Parallelism,
) {
    use coe::Coerce;
    use num_complex::Complex;
    if coe::is_same::<E, Complex<E::Real>>() {
        let acc: MatMut<'_, Complex<E::Real>> = acc.coerce();
        let a: MatRef<'_, Complex<E::Real>> = a.coerce();
        let b: MatRef<'_, Complex<E::Real>> = b.coerce();

        let Complex {
            re: mut acc_re,
            im: mut acc_im,
        } = acc.real_imag_mut();
        let Complex { re: a_re, im: a_im } = a.real_imag();
        let Complex { re: b_re, im: b_im } = b.real_imag();

        let real_matmul = |acc: MatMut<'_, E::Real>,
                           a: MatRef<'_, E::Real>,
                           b: MatRef<'_, E::Real>,
                           beta: E::Real| {
            matmul_with_conj(
                acc,
                a,
                Conj::No,
                b,
                Conj::No,
                Some(E::Real::faer_one()),
                beta,
                parallelism,
            )
        };

        match conj_b {
            Conj::Yes => {
                real_matmul(acc_re.rb_mut(), a_re, b_re, E::Real::faer_one());
                real_matmul(acc_re.rb_mut(), a_im, b_im, E::Real::faer_one());
                real_matmul(acc_im.rb_mut(), a_re, b_im, E::Real::faer_one().faer_neg());
                real_matmul(acc_im.rb_mut(), a_im, b_re, E::Real::faer_one());
            }
            Conj::No => {
                real_matmul(acc_re.rb_mut(), a_re, b_re, E::Real::faer_one());
                real_matmul(acc_re.rb_mut(), a_im, b_im, E::Real::faer_one().faer_neg());
                real_matmul(acc_im.rb_mut(), a_re, b_im, E::Real::faer_one());
                real_matmul(acc_im.rb_mut(), a_im, b_re, E::Real::faer_one());
            }
        }

        return;
    }

    let m = acc.nrows();
    let n = acc.ncols();
    let k = a.ncols();

    let arch = E::Simd::default();
    let lane_count = arch.dispatch(SimdLaneCount::<E> {
        __marker: PhantomData,
    });

    let nr = MicroKernelShape::<E>::MAX_NR;
    let mr_div_n = MicroKernelShape::<E>::MAX_MR_DIV_N;
    let mr = mr_div_n * lane_count;

    assert!(all(
        acc.row_stride() == 1,
        a.row_stride() == 1,
        b.row_stride() == 1,
        m % lane_count == 0,
    ));

    let mut acc = acc;

    let mut col_outer = 0usize;
    while col_outer < n {
        let n_chunk = min(NC, n - col_outer);

        let b_panel = b.submatrix(0, col_outer, k, n_chunk);
        let acc = acc.rb_mut().submatrix_mut(0, col_outer, m, n_chunk);

        let mut depth_outer = 0usize;
        while depth_outer < k {
            let k_chunk = min(KC, k - depth_outer);

            let a_panel = a.submatrix(0, depth_outer, m, k_chunk);
            let b_block = b_panel.submatrix(depth_outer, 0, k_chunk, n_chunk);

            let n_job_count = n_chunk.msrv_div_ceil(nr);
            let chunk_count = m.msrv_div_ceil(MC);

            let job_count = n_job_count * chunk_count;

            let job = |idx: usize| {
                assert!(all(
                    acc.row_stride() == 1,
                    a.row_stride() == 1,
                    b.row_stride() == 1,
                ));

                let col_inner = (idx % n_job_count) * nr;
                let row_outer = (idx / n_job_count) * MC;
                let m_chunk = min(MC, m - row_outer);

                let mut row_inner = 0;
                let ncols = min(nr, n_chunk - col_inner);
                let ukr_j = ncols;

                while row_inner < m_chunk {
                    let nrows = min(mr, m_chunk - row_inner);

                    let ukr_i = nrows / lane_count;

                    let a = a_panel.submatrix(row_outer + row_inner, 0, nrows, k_chunk);
                    let b = b_block.submatrix(0, col_inner, k_chunk, ncols);
                    let acc = acc
                        .rb()
                        .submatrix(row_outer + row_inner, col_inner, nrows, ncols);
                    let acc = unsafe { acc.const_cast() };

                    match conj_b {
                        Conj::Yes => {
                            let conj_b = YesConj;
                            if MicroKernelShape::<E>::IS_2X2 {
                                match (ukr_i, ukr_j) {
                                    (2, 2) => {
                                        arch.dispatch(Ukr::<2, 2, _, E> { conj_b, acc, a, b })
                                    }
                                    (2, 1) => {
                                        arch.dispatch(Ukr::<2, 1, _, E> { conj_b, acc, a, b })
                                    }
                                    (1, 2) => {
                                        arch.dispatch(Ukr::<1, 2, _, E> { conj_b, acc, a, b })
                                    }
                                    (1, 1) => {
                                        arch.dispatch(Ukr::<1, 1, _, E> { conj_b, acc, a, b })
                                    }
                                    _ => unreachable!(),
                                }
                            } else if MicroKernelShape::<E>::IS_2X1 {
                                match (ukr_i, ukr_j) {
                                    (2, 1) => {
                                        arch.dispatch(Ukr::<2, 1, _, E> { conj_b, acc, a, b })
                                    }
                                    (1, 1) => {
                                        arch.dispatch(Ukr::<1, 1, _, E> { conj_b, acc, a, b })
                                    }
                                    _ => unreachable!(),
                                }
                            } else if MicroKernelShape::<E>::IS_1X1 {
                                match (ukr_i, ukr_j) {
                                    (1, 1) => {
                                        arch.dispatch(Ukr::<1, 1, _, E> { conj_b, acc, a, b })
                                    }
                                    _ => unreachable!(),
                                }
                            } else {
                                unreachable!()
                            }
                        }
                        Conj::No => {
                            let conj_b = NoConj;
                            if MicroKernelShape::<E>::IS_2X2 {
                                match (ukr_i, ukr_j) {
                                    (2, 2) => {
                                        arch.dispatch(Ukr::<2, 2, _, E> { conj_b, acc, a, b })
                                    }
                                    (2, 1) => {
                                        arch.dispatch(Ukr::<2, 1, _, E> { conj_b, acc, a, b })
                                    }
                                    (1, 2) => {
                                        arch.dispatch(Ukr::<1, 2, _, E> { conj_b, acc, a, b })
                                    }
                                    (1, 1) => {
                                        arch.dispatch(Ukr::<1, 1, _, E> { conj_b, acc, a, b })
                                    }
                                    _ => unreachable!(),
                                }
                            } else if MicroKernelShape::<E>::IS_2X1 {
                                match (ukr_i, ukr_j) {
                                    (2, 1) => {
                                        arch.dispatch(Ukr::<2, 1, _, E> { conj_b, acc, a, b })
                                    }
                                    (1, 1) => {
                                        arch.dispatch(Ukr::<1, 1, _, E> { conj_b, acc, a, b })
                                    }
                                    _ => unreachable!(),
                                }
                            } else if MicroKernelShape::<E>::IS_1X1 {
                                match (ukr_i, ukr_j) {
                                    (1, 1) => {
                                        arch.dispatch(Ukr::<1, 1, _, E> { conj_b, acc, a, b })
                                    }
                                    _ => unreachable!(),
                                }
                            } else {
                                unreachable!()
                            }
                        }
                    }
                    row_inner += nrows;
                }
            };

            crate::for_each_raw(job_count, job, parallelism);

            depth_outer += k_chunk;
        }

        col_outer += n_chunk;
    }
}

#[doc(hidden)]
pub fn matmul_with_conj_gemm_dispatch<E: ComplexField>(
    mut acc: MatMut<'_, E>,
    lhs: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    conj_rhs: Conj,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
    _use_gemm: bool,
) {
    assert!(all(
        acc.nrows() == lhs.nrows(),
        acc.ncols() == rhs.ncols(),
        lhs.ncols() == rhs.nrows(),
    ));

    let m = acc.nrows();
    let n = acc.ncols();
    let k = lhs.ncols();

    if m == 0 || n == 0 {
        return;
    }

    if m == 1 && n == 1 {
        let mut acc = acc;
        let ab = inner_prod::inner_prod_with_conj(lhs.transpose(), conj_lhs, rhs, conj_rhs);
        match alpha {
            Some(alpha) => {
                acc.write(
                    0,
                    0,
                    E::faer_add(acc.read(0, 0).faer_mul(alpha), ab.faer_mul(beta)),
                );
            }
            None => {
                acc.write(0, 0, ab.faer_mul(beta));
            }
        }
        return;
    }

    if k == 1 {
        outer_prod::outer_prod_with_conj(
            acc,
            lhs,
            conj_lhs,
            rhs.transpose(),
            conj_rhs,
            alpha,
            beta,
        );
        return;
    }
    if n == 1 {
        matvec::matvec_with_conj(acc, lhs, conj_lhs, rhs, conj_rhs, alpha, beta);
        return;
    }
    if m == 1 {
        matvec::matvec_with_conj(
            acc.transpose_mut(),
            rhs.transpose(),
            conj_rhs,
            lhs.transpose(),
            conj_lhs,
            alpha,
            beta,
        );
        return;
    }

    unsafe {
        if m + n < 32 && k <= 6 {
            macro_rules! small_gemm {
                ($term: expr) => {
                    let term = $term;
                    match k {
                        0 => match alpha {
                            Some(alpha) => {
                                for i in 0..m {
                                    for j in 0..n {
                                        acc.write_unchecked(
                                            i,
                                            j,
                                            acc.read_unchecked(i, j).faer_mul(alpha),
                                        )
                                    }
                                }
                            }
                            None => {
                                for i in 0..m {
                                    for j in 0..n {
                                        acc.write_unchecked(i, j, E::faer_zero())
                                    }
                                }
                            }
                        },
                        1 => match alpha {
                            Some(alpha) => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = term(i, j, 0);
                                        acc.write_unchecked(
                                            i,
                                            j,
                                            E::faer_add(
                                                acc.read_unchecked(i, j).faer_mul(alpha),
                                                dot.faer_mul(beta),
                                            ),
                                        )
                                    }
                                }
                            }
                            None => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = term(i, j, 0);
                                        acc.write_unchecked(i, j, dot.faer_mul(beta))
                                    }
                                }
                            }
                        },
                        2 => match alpha {
                            Some(alpha) => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = term(i, j, 0).faer_add(term(i, j, 1));
                                        acc.write_unchecked(
                                            i,
                                            j,
                                            E::faer_add(
                                                acc.read_unchecked(i, j).faer_mul(alpha),
                                                dot.faer_mul(beta),
                                            ),
                                        )
                                    }
                                }
                            }
                            None => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = term(i, j, 0).faer_add(term(i, j, 1));
                                        acc.write_unchecked(i, j, dot.faer_mul(beta))
                                    }
                                }
                            }
                        },
                        3 => match alpha {
                            Some(alpha) => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = term(i, j, 0)
                                            .faer_add(term(i, j, 1))
                                            .faer_add(term(i, j, 2));
                                        acc.write_unchecked(
                                            i,
                                            j,
                                            E::faer_add(
                                                acc.read_unchecked(i, j).faer_mul(alpha),
                                                dot.faer_mul(beta),
                                            ),
                                        )
                                    }
                                }
                            }
                            None => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = term(i, j, 0)
                                            .faer_add(term(i, j, 1))
                                            .faer_add(term(i, j, 2));
                                        acc.write_unchecked(i, j, dot.faer_mul(beta))
                                    }
                                }
                            }
                        },
                        4 => match alpha {
                            Some(alpha) => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = E::faer_add(
                                            E::faer_add(term(i, j, 0), term(i, j, 1)),
                                            E::faer_add(term(i, j, 2), term(i, j, 3)),
                                        );

                                        acc.write_unchecked(
                                            i,
                                            j,
                                            E::faer_add(
                                                acc.read_unchecked(i, j).faer_mul(alpha),
                                                dot.faer_mul(beta),
                                            ),
                                        )
                                    }
                                }
                            }
                            None => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = E::faer_add(
                                            E::faer_add(term(i, j, 0), term(i, j, 1)),
                                            E::faer_add(term(i, j, 2), term(i, j, 3)),
                                        );
                                        acc.write_unchecked(i, j, dot.faer_mul(beta))
                                    }
                                }
                            }
                        },
                        5 => match alpha {
                            Some(alpha) => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = E::faer_add(
                                            E::faer_add(term(i, j, 0), term(i, j, 1))
                                                .faer_add(term(i, j, 2)),
                                            E::faer_add(term(i, j, 3), term(i, j, 4)),
                                        );

                                        acc.write_unchecked(
                                            i,
                                            j,
                                            E::faer_add(
                                                acc.read_unchecked(i, j).faer_mul(alpha),
                                                dot.faer_mul(beta),
                                            ),
                                        )
                                    }
                                }
                            }
                            None => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = E::faer_add(
                                            E::faer_add(term(i, j, 0), term(i, j, 1))
                                                .faer_add(term(i, j, 2)),
                                            E::faer_add(term(i, j, 3), term(i, j, 4)),
                                        );
                                        acc.write_unchecked(i, j, dot.faer_mul(beta))
                                    }
                                }
                            }
                        },
                        6 => match alpha {
                            Some(alpha) => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = E::faer_add(
                                            E::faer_add(term(i, j, 0), term(i, j, 1))
                                                .faer_add(term(i, j, 2)),
                                            E::faer_add(term(i, j, 3), term(i, j, 4))
                                                .faer_add(term(i, j, 5)),
                                        );

                                        acc.write_unchecked(
                                            i,
                                            j,
                                            E::faer_add(
                                                acc.read_unchecked(i, j).faer_mul(alpha),
                                                dot.faer_mul(beta),
                                            ),
                                        )
                                    }
                                }
                            }
                            None => {
                                for i in 0..m {
                                    for j in 0..n {
                                        let dot = E::faer_add(
                                            E::faer_add(term(i, j, 0), term(i, j, 1))
                                                .faer_add(term(i, j, 2)),
                                            E::faer_add(term(i, j, 3), term(i, j, 4))
                                                .faer_add(term(i, j, 5)),
                                        );
                                        acc.write_unchecked(i, j, dot.faer_mul(beta))
                                    }
                                }
                            }
                        },
                        _ => unreachable!(),
                    }
                };
            }

            match (conj_lhs, conj_rhs) {
                (Conj::Yes, Conj::Yes) => {
                    let term = {
                        #[inline(always)]
                        |i, j, depth| {
                            (lhs.read_unchecked(i, depth)
                                .faer_mul(rhs.read_unchecked(depth, j)))
                            .faer_conj()
                        }
                    };
                    small_gemm!(term);
                }
                (Conj::Yes, Conj::No) => {
                    let term = {
                        #[inline(always)]
                        |i, j, depth| {
                            lhs.read_unchecked(i, depth)
                                .faer_conj()
                                .faer_mul(rhs.read_unchecked(depth, j))
                        }
                    };
                    small_gemm!(term);
                }
                (Conj::No, Conj::Yes) => {
                    let term = {
                        #[inline(always)]
                        |i, j, depth| {
                            lhs.read_unchecked(i, depth)
                                .faer_mul(rhs.read_unchecked(depth, j).faer_conj())
                        }
                    };
                    small_gemm!(term);
                }
                (Conj::No, Conj::No) => {
                    let term = {
                        #[inline(always)]
                        |i, j, depth| {
                            lhs.read_unchecked(i, depth)
                                .faer_mul(rhs.read_unchecked(depth, j))
                        }
                    };
                    small_gemm!(term);
                }
            }
            return;
        }
    }

    #[cfg(not(test))]
    let _use_gemm = true;

    if _use_gemm {
        let gemm_parallelism = match parallelism {
            Parallelism::None => gemm::Parallelism::None,
            #[cfg(feature = "rayon")]
            Parallelism::Rayon(0) => gemm::Parallelism::Rayon(rayon::current_num_threads()),
            #[cfg(feature = "rayon")]
            Parallelism::Rayon(n_threads) => gemm::Parallelism::Rayon(n_threads),
        };
        if coe::is_same::<f32, E>() {
            let mut acc: MatMut<'_, f32> = coe::coerce(acc);
            let a: MatRef<'_, f32> = coe::coerce(lhs);
            let b: MatRef<'_, f32> = coe::coerce(rhs);
            let alpha: Option<f32> = coe::coerce_static(alpha);
            let beta: f32 = coe::coerce_static(beta);
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    acc.rb_mut().as_ptr_mut(),
                    acc.col_stride(),
                    acc.row_stride(),
                    alpha.is_some(),
                    a.as_ptr(),
                    a.col_stride(),
                    a.row_stride(),
                    b.as_ptr(),
                    b.col_stride(),
                    b.row_stride(),
                    alpha.unwrap_or(0.0),
                    beta,
                    false,
                    conj_lhs == Conj::Yes,
                    conj_rhs == Conj::Yes,
                    gemm_parallelism,
                )
            };
            return;
        }
        if coe::is_same::<f64, E>() {
            let mut acc: MatMut<'_, f64> = coe::coerce(acc);
            let a: MatRef<'_, f64> = coe::coerce(lhs);
            let b: MatRef<'_, f64> = coe::coerce(rhs);
            let alpha: Option<f64> = coe::coerce_static(alpha);
            let beta: f64 = coe::coerce_static(beta);
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    acc.rb_mut().as_ptr_mut(),
                    acc.col_stride(),
                    acc.row_stride(),
                    alpha.is_some(),
                    a.as_ptr(),
                    a.col_stride(),
                    a.row_stride(),
                    b.as_ptr(),
                    b.col_stride(),
                    b.row_stride(),
                    alpha.unwrap_or(0.0),
                    beta,
                    false,
                    conj_lhs == Conj::Yes,
                    conj_rhs == Conj::Yes,
                    gemm_parallelism,
                )
            };
            return;
        }
        if coe::is_same::<c32, E>() {
            let mut acc: MatMut<'_, c32> = coe::coerce(acc);
            let a: MatRef<'_, c32> = coe::coerce(lhs);
            let b: MatRef<'_, c32> = coe::coerce(rhs);
            let alpha: Option<c32> = coe::coerce_static(alpha);
            let beta: c32 = coe::coerce_static(beta);
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    acc.rb_mut().as_ptr_mut() as *mut gemm::c32,
                    acc.col_stride(),
                    acc.row_stride(),
                    alpha.is_some(),
                    a.as_ptr() as *const gemm::c32,
                    a.col_stride(),
                    a.row_stride(),
                    b.as_ptr() as *const gemm::c32,
                    b.col_stride(),
                    b.row_stride(),
                    alpha.unwrap_or(c32 { re: 0.0, im: 0.0 }).into(),
                    beta.into(),
                    false,
                    conj_lhs == Conj::Yes,
                    conj_rhs == Conj::Yes,
                    gemm_parallelism,
                )
            };
            return;
        }
        if coe::is_same::<c64, E>() {
            let mut acc: MatMut<'_, c64> = coe::coerce(acc);
            let a: MatRef<'_, c64> = coe::coerce(lhs);
            let b: MatRef<'_, c64> = coe::coerce(rhs);
            let alpha: Option<c64> = coe::coerce_static(alpha);
            let beta: c64 = coe::coerce_static(beta);
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    acc.rb_mut().as_ptr_mut() as *mut gemm::c64,
                    acc.col_stride(),
                    acc.row_stride(),
                    alpha.is_some(),
                    a.as_ptr() as *const gemm::c64,
                    a.col_stride(),
                    a.row_stride(),
                    b.as_ptr() as *const gemm::c64,
                    b.col_stride(),
                    b.row_stride(),
                    alpha.unwrap_or(c64 { re: 0.0, im: 0.0 }).into(),
                    beta.into(),
                    false,
                    conj_lhs == Conj::Yes,
                    conj_rhs == Conj::Yes,
                    gemm_parallelism,
                )
            };
            return;
        }
    }

    let arch = E::Simd::default();
    let lane_count = arch.dispatch(SimdLaneCount::<E> {
        __marker: PhantomData,
    });

    let mut a = lhs;
    let mut b = rhs;
    let mut conj_a = conj_lhs;
    let mut conj_b = conj_rhs;

    if n < m {
        (a, b) = (b.transpose(), a.transpose());
        core::mem::swap(&mut conj_a, &mut conj_b);
        acc = acc.transpose_mut();
    }

    if b.row_stride() < 0 {
        a = a.reverse_cols();
        b = b.reverse_rows();
    }

    let m = acc.nrows();
    let n = acc.ncols();

    let padded_m = m.msrv_checked_next_multiple_of(lane_count).unwrap();

    let mut a_copy = a.to_owned();
    a_copy.resize_with(padded_m, k, |_, _| E::faer_zero());
    let a_copy = a_copy.as_ref();
    let mut tmp = crate::Mat::<E>::zeros(padded_m, n);
    let tmp_conj_b = match (conj_a, conj_b) {
        (Conj::Yes, Conj::Yes) | (Conj::No, Conj::No) => Conj::No,
        (Conj::Yes, Conj::No) | (Conj::No, Conj::Yes) => Conj::Yes,
    };
    if b.row_stride() == 1 {
        matmul_with_conj_impl(tmp.as_mut(), a_copy, b, tmp_conj_b, parallelism);
    } else {
        let b = b.to_owned();
        matmul_with_conj_impl(tmp.as_mut(), a_copy, b.as_ref(), tmp_conj_b, parallelism);
    }

    let tmp = tmp.as_ref().subrows(0, m);

    match alpha {
        Some(alpha) => match conj_a {
            Conj::Yes => zipped!(acc, tmp).for_each(|unzipped!(mut acc, tmp)| {
                acc.write(E::faer_add(
                    acc.read().faer_mul(alpha),
                    tmp.read().faer_conj().faer_mul(beta),
                ))
            }),
            Conj::No => zipped!(acc, tmp).for_each(|unzipped!(mut acc, tmp)| {
                acc.write(E::faer_add(
                    acc.read().faer_mul(alpha),
                    tmp.read().faer_mul(beta),
                ))
            }),
        },
        None => match conj_a {
            Conj::Yes => {
                zipped!(acc, tmp).for_each(|unzipped!(mut acc, tmp)| {
                    acc.write(tmp.read().faer_conj().faer_mul(beta))
                });
            }
            Conj::No => {
                zipped!(acc, tmp)
                    .for_each(|unzipped!(mut acc, tmp)| acc.write(tmp.read().faer_mul(beta)));
            }
        },
    }
}

/// Computes the matrix product `[alpha * acc] + beta * lhs * rhs` (while optionally conjugating
/// either or both of the input matrices) and stores the result in `acc`.
///
/// Performs the operation:
/// - `acc = beta * Op_lhs(lhs) * Op_rhs(rhs)` if `alpha` is `None` (in this case, the preexisting
/// values in `acc` are not read, so it is allowed to be a view over uninitialized values if `E:
/// Copy`),
/// - `acc = alpha * acc + beta * Op_lhs(lhs) * Op_rhs(rhs)` if `alpha` is `Some(_)`,
///
/// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
/// `Op_rhs` is the identity if `conj_rhs` is `Conj::No`, and the conjugation operation if it is
/// `Conj::Yes`.  
///
/// # Panics
///
/// Panics if the matrix dimensions are not compatible for matrix multiplication.  
/// i.e.  
///  - `acc.nrows() == lhs.nrows()`
///  - `acc.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
///
/// # Example
///
/// ```
/// use faer_core::{mat, mul::matmul_with_conj, unzipped, zipped, Conj, Mat, Parallelism};
///
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
///
/// let mut acc = Mat::<f64>::zeros(2, 2);
/// let target = mat![
///     [
///         2.5 * (lhs.read(0, 0) * rhs.read(0, 0) + lhs.read(0, 1) * rhs.read(1, 0)),
///         2.5 * (lhs.read(0, 0) * rhs.read(0, 1) + lhs.read(0, 1) * rhs.read(1, 1)),
///     ],
///     [
///         2.5 * (lhs.read(1, 0) * rhs.read(0, 0) + lhs.read(1, 1) * rhs.read(1, 0)),
///         2.5 * (lhs.read(1, 0) * rhs.read(0, 1) + lhs.read(1, 1) * rhs.read(1, 1)),
///     ],
/// ];
///
/// matmul_with_conj(
///     acc.as_mut(),
///     lhs.as_ref(),
///     Conj::No,
///     rhs.as_ref(),
///     Conj::No,
///     None,
///     2.5,
///     Parallelism::None,
/// );
///
/// zipped!(acc.as_ref(), target.as_ref())
///     .for_each(|unzipped!(acc, target)| assert!((acc.read() - target.read()).abs() < 1e-10));
/// ```
#[inline]
#[track_caller]
pub fn matmul_with_conj<E: ComplexField>(
    acc: MatMut<'_, E>,
    lhs: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    conj_rhs: Conj,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
) {
    assert!(all(
        acc.nrows() == lhs.nrows(),
        acc.ncols() == rhs.ncols(),
        lhs.ncols() == rhs.nrows(),
    ));
    matmul_with_conj_gemm_dispatch(
        acc,
        lhs,
        conj_lhs,
        rhs,
        conj_rhs,
        alpha,
        beta,
        parallelism,
        true,
    );
}

/// Computes the matrix product `[alpha * acc] + beta * lhs * rhs` and
/// stores the result in `acc`.
///
/// Performs the operation:
/// - `acc = beta * lhs * rhs` if `alpha` is `None` (in this case, the preexisting values in `acc`
///   are not read, so it is allowed to be a view over uninitialized values if `E: Copy`),
/// - `acc = alpha * acc + beta * lhs * rhs` if `alpha` is `Some(_)`,
///
/// # Panics
///
/// Panics if the matrix dimensions are not compatible for matrix multiplication.  
/// i.e.  
///  - `acc.nrows() == lhs.nrows()`
///  - `acc.ncols() == rhs.ncols()`
///  - `lhs.ncols() == rhs.nrows()`
///
/// # Example
///
/// ```
/// use faer_core::{mat, mul::matmul, unzipped, zipped, Mat, Parallelism};
///
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
///
/// let mut acc = Mat::<f64>::zeros(2, 2);
/// let target = mat![
///     [
///         2.5 * (lhs.read(0, 0) * rhs.read(0, 0) + lhs.read(0, 1) * rhs.read(1, 0)),
///         2.5 * (lhs.read(0, 0) * rhs.read(0, 1) + lhs.read(0, 1) * rhs.read(1, 1)),
///     ],
///     [
///         2.5 * (lhs.read(1, 0) * rhs.read(0, 0) + lhs.read(1, 1) * rhs.read(1, 0)),
///         2.5 * (lhs.read(1, 0) * rhs.read(0, 1) + lhs.read(1, 1) * rhs.read(1, 1)),
///     ],
/// ];
///
/// matmul(
///     acc.as_mut(),
///     lhs.as_ref(),
///     rhs.as_ref(),
///     None,
///     2.5,
///     Parallelism::None,
/// );
///
/// zipped!(acc.as_ref(), target.as_ref())
///     .for_each(|unzipped!(acc, target)| assert!((acc.read() - target.read()).abs() < 1e-10));
/// ```
#[track_caller]
pub fn matmul<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
    acc: MatMut<'_, E>,
    lhs: MatRef<'_, LhsE>,
    rhs: MatRef<'_, RhsE>,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
) {
    let (lhs, conj_lhs) = lhs.canonicalize();
    let (rhs, conj_rhs) = rhs.canonicalize();
    matmul_with_conj::<E>(acc, lhs, conj_lhs, rhs, conj_rhs, alpha, beta, parallelism);
}

macro_rules! stack_mat_16x16_begin {
    ($name: ident, $nrows: expr, $ncols: expr, $rs: expr, $cs: expr, $ty: ty) => {
        let __nrows: usize = $nrows;
        let __ncols: usize = $ncols;
        let __rs: isize = $rs;
        let __cs: isize = $cs;
        let mut __data = <$ty as $crate::Entity>::faer_map(
            <$ty as $crate::Entity>::UNIT,
            #[inline(always)]
            |()| unsafe {
                $crate::transmute_unchecked::<
                    ::core::mem::MaybeUninit<[<$ty as $crate::Entity>::Unit; 16 * 16]>,
                    [::core::mem::MaybeUninit<<$ty as $crate::Entity>::Unit>; 16 * 16],
                >(::core::mem::MaybeUninit::<
                    [<$ty as $crate::Entity>::Unit; 16 * 16],
                >::uninit())
            },
        );

        <$ty as $crate::Entity>::faer_map(
            <$ty as $crate::Entity>::faer_zip(
                <$ty as $crate::Entity>::faer_as_mut(&mut __data),
                <$ty as $crate::Entity>::faer_into_units(<$ty as $crate::ComplexField>::faer_zero()),
            ),
            #[inline(always)]
            |(__data, zero)| {
                let __data: &mut _ = __data;
                for __data in __data {
                    let __data : &mut _ = __data;
                    *__data = ::core::mem::MaybeUninit::new(::core::clone::Clone::clone(&zero));
                }
            },
        );
        let mut __data =
            <$ty as $crate::Entity>::faer_map(<$ty as $crate::Entity>::faer_as_mut(&mut __data), |__data: &mut _| {
                (__data as *mut [::core::mem::MaybeUninit<<$ty as $crate::Entity>::Unit>; 16 * 16]
                    as *mut <$ty as $crate::Entity>::Unit)
            });

        let mut $name = unsafe {
            $crate::mat::from_raw_parts_mut::<'_, $ty>(__data, __nrows, __ncols, 1isize, 16isize)
        };

        if __cs.unsigned_abs() < __rs.unsigned_abs() {
            $name = $name.transpose_mut();
        }
        if __rs == -1 {
            $name = $name.reverse_rows_mut();
        }
        if __cs == -1 {
            $name = $name.reverse_cols_mut();
        }
    };
}

/// Triangular matrix multiplication module, where some of the operands are treated as triangular
/// matrices.
pub mod triangular {
    use super::*;
    use crate::{assert, debug_assert, join_raw, zip::Diag};

    #[repr(u8)]
    #[derive(Copy, Clone, Debug)]
    pub(crate) enum DiagonalKind {
        Zero,
        Unit,
        Generic,
    }

    unsafe fn copy_lower<E: ComplexField>(
        mut dst: MatMut<'_, E>,
        src: MatRef<'_, E>,
        src_diag: DiagonalKind,
    ) {
        let n = dst.nrows();
        debug_assert!(n == dst.nrows());
        debug_assert!(n == dst.ncols());
        debug_assert!(n == src.nrows());
        debug_assert!(n == src.ncols());

        let strict = match src_diag {
            DiagonalKind::Zero => {
                for j in 0..n {
                    dst.write_unchecked(j, j, E::faer_zero());
                }
                true
            }
            DiagonalKind::Unit => {
                for j in 0..n {
                    dst.write_unchecked(j, j, E::faer_one());
                }
                true
            }
            DiagonalKind::Generic => false,
        };

        zipped!(dst.rb_mut())
            .for_each_triangular_upper(Diag::Skip, |unzipped!(mut dst)| dst.write(E::faer_zero()));
        zipped!(dst, src).for_each_triangular_lower(
            if strict { Diag::Skip } else { Diag::Include },
            |unzipped!(mut dst, src)| dst.write(src.read()),
        );
    }

    unsafe fn accum_lower<E: ComplexField>(
        dst: MatMut<'_, E>,
        src: MatRef<'_, E>,
        skip_diag: bool,
        alpha: Option<E>,
    ) {
        let n = dst.nrows();
        debug_assert!(n == dst.nrows());
        debug_assert!(n == dst.ncols());
        debug_assert!(n == src.nrows());
        debug_assert!(n == src.ncols());

        match alpha {
            Some(alpha) => {
                zipped!(dst, src).for_each_triangular_lower(
                    if skip_diag { Diag::Skip } else { Diag::Include },
                    |unzipped!(mut dst, src)| {
                        dst.write(alpha.faer_mul(dst.read().faer_add(src.read())))
                    },
                );
            }
            None => {
                zipped!(dst, src).for_each_triangular_lower(
                    if skip_diag { Diag::Skip } else { Diag::Include },
                    |unzipped!(mut dst, src)| dst.write(src.read()),
                );
            }
        }
    }

    #[inline]
    unsafe fn copy_upper<E: ComplexField>(
        dst: MatMut<'_, E>,
        src: MatRef<'_, E>,
        src_diag: DiagonalKind,
    ) {
        copy_lower(dst.transpose_mut(), src.transpose(), src_diag)
    }

    #[inline]
    unsafe fn mul<E: ComplexField>(
        dst: MatMut<'_, E>,
        lhs: MatRef<'_, E>,
        rhs: MatRef<'_, E>,
        alpha: Option<E>,
        beta: E,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        super::matmul_with_conj(dst, lhs, conj_lhs, rhs, conj_rhs, alpha, beta, parallelism);
    }

    unsafe fn mat_x_lower_into_lower_impl_unchecked<E: ComplexField>(
        dst: MatMut<'_, E>,
        skip_diag: bool,
        lhs: MatRef<'_, E>,
        rhs: MatRef<'_, E>,
        rhs_diag: DiagonalKind,
        alpha: Option<E>,
        beta: E,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        let n = dst.nrows();
        debug_assert!(n == dst.nrows());
        debug_assert!(n == dst.ncols());
        debug_assert!(n == lhs.nrows());
        debug_assert!(n == lhs.ncols());
        debug_assert!(n == rhs.nrows());
        debug_assert!(n == rhs.ncols());

        if n <= 16 {
            let op = {
                #[inline(never)]
                || {
                    stack_mat_16x16_begin!(temp_dst, n, n, dst.row_stride(), dst.col_stride(), E);
                    stack_mat_16x16_begin!(temp_rhs, n, n, rhs.row_stride(), rhs.col_stride(), E);

                    copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);
                    mul(
                        temp_dst.rb_mut(),
                        lhs,
                        temp_rhs.rb(),
                        None,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                    accum_lower(dst, temp_dst.rb(), skip_diag, alpha);
                }
            };
            op();
        } else {
            let bs = n / 2;

            let (mut dst_top_left, _, mut dst_bot_left, dst_bot_right) = dst.split_at_mut(bs, bs);
            let (lhs_top_left, lhs_top_right, lhs_bot_left, lhs_bot_right) = lhs.split_at(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at(bs, bs);

            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | mat × mat => mat |   1
            // lhs_bot_right × rhs_bot_right => dst_bot_right | mat × low => low |   X
            //
            // lhs_top_left  × rhs_top_left  => dst_top_left  | mat × low => low |   X
            // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
            // lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2

            mul(
                dst_bot_left.rb_mut(),
                lhs_bot_right,
                rhs_bot_left,
                alpha,
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_lower_into_lower_impl_unchecked(
                dst_bot_right,
                skip_diag,
                lhs_bot_right,
                rhs_bot_right,
                rhs_diag,
                alpha,
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );

            mat_x_lower_into_lower_impl_unchecked(
                dst_top_left.rb_mut(),
                skip_diag,
                lhs_top_left,
                rhs_top_left,
                rhs_diag,
                alpha,
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_mat_into_lower_impl_unchecked(
                dst_top_left,
                skip_diag,
                lhs_top_right,
                rhs_bot_left,
                Some(E::faer_one()),
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_lower_impl_unchecked(
                dst_bot_left,
                lhs_bot_left,
                rhs_top_left,
                rhs_diag,
                Some(E::faer_one()),
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
        }
    }

    unsafe fn mat_x_lower_impl_unchecked<E: ComplexField>(
        dst: MatMut<'_, E>,
        lhs: MatRef<'_, E>,
        rhs: MatRef<'_, E>,
        rhs_diag: DiagonalKind,
        alpha: Option<E>,
        beta: E,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        let n = rhs.nrows();
        let m = lhs.nrows();
        debug_assert!(m == lhs.nrows());
        debug_assert!(n == lhs.ncols());
        debug_assert!(n == rhs.nrows());
        debug_assert!(n == rhs.ncols());
        debug_assert!(m == dst.nrows());
        debug_assert!(n == dst.ncols());

        let join_parallelism = if n * n * m < 128 * 128 * 64 {
            Parallelism::None
        } else {
            parallelism
        };

        if n <= 16 {
            let op = {
                #[inline(never)]
                || {
                    stack_mat_16x16_begin!(temp_rhs, n, n, rhs.row_stride(), rhs.col_stride(), E);

                    copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

                    mul(
                        dst,
                        lhs,
                        temp_rhs.rb(),
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                }
            };
            op();
        } else {
            // split rhs into 3 sections
            // split lhs and dst into 2 sections

            let bs = n / 2;

            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at(bs, bs);
            let (lhs_left, lhs_right) = lhs.split_at_col(bs);
            let (mut dst_left, mut dst_right) = dst.split_at_col_mut(bs);

            join_raw(
                |parallelism| {
                    mat_x_lower_impl_unchecked(
                        dst_left.rb_mut(),
                        lhs_left,
                        rhs_top_left,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                |parallelism| {
                    mat_x_lower_impl_unchecked(
                        dst_right.rb_mut(),
                        lhs_right,
                        rhs_bot_right,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                join_parallelism,
            );
            mul(
                dst_left,
                lhs_right,
                rhs_bot_left,
                Some(E::faer_one()),
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
        }
    }

    unsafe fn lower_x_lower_into_lower_impl_unchecked<E: ComplexField>(
        dst: MatMut<'_, E>,
        skip_diag: bool,
        lhs: MatRef<'_, E>,
        lhs_diag: DiagonalKind,
        rhs: MatRef<'_, E>,
        rhs_diag: DiagonalKind,
        alpha: Option<E>,
        beta: E,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        let n = dst.nrows();
        debug_assert!(n == lhs.nrows());
        debug_assert!(n == lhs.ncols());
        debug_assert!(n == rhs.nrows());
        debug_assert!(n == rhs.ncols());
        debug_assert!(n == dst.nrows());
        debug_assert!(n == dst.ncols());

        if n <= 16 {
            let op = {
                #[inline(never)]
                || {
                    stack_mat_16x16_begin!(temp_dst, n, n, dst.row_stride(), dst.col_stride(), E);
                    stack_mat_16x16_begin!(temp_lhs, n, n, lhs.row_stride(), lhs.col_stride(), E);
                    stack_mat_16x16_begin!(temp_rhs, n, n, rhs.row_stride(), rhs.col_stride(), E);

                    copy_lower(temp_lhs.rb_mut(), lhs, lhs_diag);
                    copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

                    mul(
                        temp_dst.rb_mut(),
                        temp_lhs.rb(),
                        temp_rhs.rb(),
                        None,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                    accum_lower(dst, temp_dst.rb(), skip_diag, alpha);
                }
            };
            op();
        } else {
            let bs = n / 2;

            let (dst_top_left, _, mut dst_bot_left, dst_bot_right) = dst.split_at_mut(bs, bs);
            let (lhs_top_left, _, lhs_bot_left, lhs_bot_right) = lhs.split_at(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at(bs, bs);

            // lhs_top_left  × rhs_top_left  => dst_top_left  | low × low => low |   X
            // lhs_bot_left  × rhs_top_left  => dst_bot_left  | mat × low => mat | 1/2
            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | low × mat => mat | 1/2
            // lhs_bot_right × rhs_bot_right => dst_bot_right | low × low => low |   X

            lower_x_lower_into_lower_impl_unchecked(
                dst_top_left,
                skip_diag,
                lhs_top_left,
                lhs_diag,
                rhs_top_left,
                rhs_diag,
                alpha,
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_lower_impl_unchecked(
                dst_bot_left.rb_mut(),
                lhs_bot_left,
                rhs_top_left,
                rhs_diag,
                alpha,
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_lower_impl_unchecked(
                dst_bot_left.reverse_rows_and_cols_mut().transpose_mut(),
                rhs_bot_left.reverse_rows_and_cols().transpose(),
                lhs_bot_right.reverse_rows_and_cols().transpose(),
                lhs_diag,
                Some(E::faer_one()),
                beta,
                conj_rhs,
                conj_lhs,
                parallelism,
            );
            lower_x_lower_into_lower_impl_unchecked(
                dst_bot_right,
                skip_diag,
                lhs_bot_right,
                lhs_diag,
                rhs_bot_right,
                rhs_diag,
                alpha,
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            )
        }
    }

    unsafe fn upper_x_lower_impl_unchecked<E: ComplexField>(
        dst: MatMut<'_, E>,
        lhs: MatRef<'_, E>,
        lhs_diag: DiagonalKind,
        rhs: MatRef<'_, E>,
        rhs_diag: DiagonalKind,
        alpha: Option<E>,
        beta: E,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        let n = dst.nrows();
        debug_assert!(n == lhs.nrows());
        debug_assert!(n == lhs.ncols());
        debug_assert!(n == rhs.nrows());
        debug_assert!(n == rhs.ncols());
        debug_assert!(n == dst.nrows());
        debug_assert!(n == dst.ncols());

        if n <= 16 {
            let op = {
                #[inline(never)]
                || {
                    stack_mat_16x16_begin!(temp_lhs, n, n, lhs.row_stride(), lhs.col_stride(), E);
                    stack_mat_16x16_begin!(temp_rhs, n, n, rhs.row_stride(), rhs.col_stride(), E);

                    copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
                    copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

                    mul(
                        dst,
                        temp_lhs.rb(),
                        temp_rhs.rb(),
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                }
            };
            op();
        } else {
            let bs = n / 2;

            let (mut dst_top_left, dst_top_right, dst_bot_left, dst_bot_right) =
                dst.split_at_mut(bs, bs);
            let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_at(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at(bs, bs);

            // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => mat |   1
            // lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => mat |   X
            //
            // lhs_top_right × rhs_bot_right => dst_top_right | mat × low => mat | 1/2
            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
            // lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => mat |   X

            join_raw(
                |_| {
                    mul(
                        dst_top_left.rb_mut(),
                        lhs_top_right,
                        rhs_bot_left,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                    upper_x_lower_impl_unchecked(
                        dst_top_left,
                        lhs_top_left,
                        lhs_diag,
                        rhs_top_left,
                        rhs_diag,
                        Some(E::faer_one()),
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                |_| {
                    join_raw(
                        |_| {
                            mat_x_lower_impl_unchecked(
                                dst_top_right,
                                lhs_top_right,
                                rhs_bot_right,
                                rhs_diag,
                                alpha,
                                beta,
                                conj_lhs,
                                conj_rhs,
                                parallelism,
                            )
                        },
                        |_| {
                            mat_x_lower_impl_unchecked(
                                dst_bot_left.transpose_mut(),
                                rhs_bot_left.transpose(),
                                lhs_bot_right.transpose(),
                                lhs_diag,
                                alpha,
                                beta,
                                conj_rhs,
                                conj_lhs,
                                parallelism,
                            )
                        },
                        parallelism,
                    );

                    upper_x_lower_impl_unchecked(
                        dst_bot_right,
                        lhs_bot_right,
                        lhs_diag,
                        rhs_bot_right,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                parallelism,
            );
        }
    }

    unsafe fn upper_x_lower_into_lower_impl_unchecked<E: ComplexField>(
        dst: MatMut<'_, E>,
        skip_diag: bool,
        lhs: MatRef<'_, E>,
        lhs_diag: DiagonalKind,
        rhs: MatRef<'_, E>,
        rhs_diag: DiagonalKind,
        alpha: Option<E>,
        beta: E,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        let n = dst.nrows();
        debug_assert!(n == lhs.nrows());
        debug_assert!(n == lhs.ncols());
        debug_assert!(n == rhs.nrows());
        debug_assert!(n == rhs.ncols());
        debug_assert!(n == dst.nrows());
        debug_assert!(n == dst.ncols());

        if n <= 16 {
            let op = {
                #[inline(never)]
                || {
                    stack_mat_16x16_begin!(temp_dst, n, n, dst.row_stride(), dst.col_stride(), E);
                    stack_mat_16x16_begin!(temp_lhs, n, n, lhs.row_stride(), lhs.col_stride(), E);
                    stack_mat_16x16_begin!(temp_rhs, n, n, rhs.row_stride(), rhs.col_stride(), E);

                    copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
                    copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

                    mul(
                        temp_dst.rb_mut(),
                        temp_lhs.rb(),
                        temp_rhs.rb(),
                        None,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );

                    accum_lower(dst, temp_dst.rb(), skip_diag, alpha);
                }
            };
            op();
        } else {
            let bs = n / 2;

            let (mut dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_at_mut(bs, bs);
            let (lhs_top_left, lhs_top_right, _, lhs_bot_right) = lhs.split_at(bs, bs);
            let (rhs_top_left, _, rhs_bot_left, rhs_bot_right) = rhs.split_at(bs, bs);

            // lhs_top_left  × rhs_top_left  => dst_top_left  | upp × low => low |   X
            // lhs_top_right × rhs_bot_left  => dst_top_left  | mat × mat => low | 1/2
            //
            // lhs_bot_right × rhs_bot_left  => dst_bot_left  | upp × mat => mat | 1/2
            // lhs_bot_right × rhs_bot_right => dst_bot_right | upp × low => low |   X

            join_raw(
                |_| {
                    mat_x_mat_into_lower_impl_unchecked(
                        dst_top_left.rb_mut(),
                        skip_diag,
                        lhs_top_right,
                        rhs_bot_left,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                    upper_x_lower_into_lower_impl_unchecked(
                        dst_top_left,
                        skip_diag,
                        lhs_top_left,
                        lhs_diag,
                        rhs_top_left,
                        rhs_diag,
                        Some(E::faer_one()),
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                |_| {
                    mat_x_lower_impl_unchecked(
                        dst_bot_left.transpose_mut(),
                        rhs_bot_left.transpose(),
                        lhs_bot_right.transpose(),
                        lhs_diag,
                        alpha,
                        beta,
                        conj_rhs,
                        conj_lhs,
                        parallelism,
                    );
                    upper_x_lower_into_lower_impl_unchecked(
                        dst_bot_right,
                        skip_diag,
                        lhs_bot_right,
                        lhs_diag,
                        rhs_bot_right,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                parallelism,
            );
        }
    }

    unsafe fn mat_x_mat_into_lower_impl_unchecked<E: ComplexField>(
        dst: MatMut<'_, E>,
        skip_diag: bool,
        lhs: MatRef<'_, E>,
        rhs: MatRef<'_, E>,
        alpha: Option<E>,
        beta: E,
        conj_lhs: Conj,
        conj_rhs: Conj,
        parallelism: Parallelism,
    ) {
        debug_assert!(dst.nrows() == dst.ncols());
        debug_assert!(dst.nrows() == lhs.nrows());
        debug_assert!(dst.ncols() == rhs.ncols());
        debug_assert!(lhs.ncols() == rhs.nrows());

        let n = dst.nrows();
        let k = lhs.ncols();

        let join_parallelism = if n * n * k < 128 * 128 * 128 {
            Parallelism::None
        } else {
            parallelism
        };

        if n <= 16 {
            let op = {
                #[inline(never)]
                || {
                    stack_mat_16x16_begin!(temp_dst, n, n, dst.row_stride(), dst.col_stride(), E);

                    mul(
                        temp_dst.rb_mut(),
                        lhs,
                        rhs,
                        None,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                    accum_lower(dst, temp_dst.rb(), skip_diag, alpha);
                }
            };
            op();
        } else {
            let bs = n / 2;
            let (dst_top_left, _, dst_bot_left, dst_bot_right) = dst.split_at_mut(bs, bs);
            let (lhs_top, lhs_bot) = lhs.split_at_row(bs);
            let (rhs_left, rhs_right) = rhs.split_at_col(bs);

            join_raw(
                |_| {
                    mul(
                        dst_bot_left,
                        lhs_bot,
                        rhs_left,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                |_| {
                    join_raw(
                        |_| {
                            mat_x_mat_into_lower_impl_unchecked(
                                dst_top_left,
                                skip_diag,
                                lhs_top,
                                rhs_left,
                                alpha,
                                beta,
                                conj_lhs,
                                conj_rhs,
                                parallelism,
                            )
                        },
                        |_| {
                            mat_x_mat_into_lower_impl_unchecked(
                                dst_bot_right,
                                skip_diag,
                                lhs_bot,
                                rhs_right,
                                alpha,
                                beta,
                                conj_lhs,
                                conj_rhs,
                                parallelism,
                            )
                        },
                        join_parallelism,
                    )
                },
                join_parallelism,
            );
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub enum BlockStructure {
        Rectangular,
        TriangularLower,
        StrictTriangularLower,
        UnitTriangularLower,
        TriangularUpper,
        StrictTriangularUpper,
        UnitTriangularUpper,
    }

    impl BlockStructure {
        #[inline]
        pub fn is_dense(self) -> bool {
            matches!(self, BlockStructure::Rectangular)
        }

        #[inline]
        pub fn is_lower(self) -> bool {
            use BlockStructure::*;
            matches!(
                self,
                TriangularLower | StrictTriangularLower | UnitTriangularLower
            )
        }

        #[inline]
        pub fn is_upper(self) -> bool {
            use BlockStructure::*;
            matches!(
                self,
                TriangularUpper | StrictTriangularUpper | UnitTriangularUpper
            )
        }

        #[inline]
        pub fn transpose(self) -> Self {
            use BlockStructure::*;
            match self {
                Rectangular => Rectangular,
                TriangularLower => TriangularUpper,
                StrictTriangularLower => StrictTriangularUpper,
                UnitTriangularLower => UnitTriangularUpper,
                TriangularUpper => TriangularLower,
                StrictTriangularUpper => StrictTriangularLower,
                UnitTriangularUpper => UnitTriangularLower,
            }
        }

        #[inline]
        pub(crate) fn diag_kind(self) -> DiagonalKind {
            use BlockStructure::*;
            match self {
                Rectangular | TriangularLower | TriangularUpper => DiagonalKind::Generic,
                StrictTriangularLower | StrictTriangularUpper => DiagonalKind::Zero,
                UnitTriangularLower | UnitTriangularUpper => DiagonalKind::Unit,
            }
        }
    }

    /// Computes the matrix product `[alpha * acc] + beta * lhs * rhs` (while optionally conjugating
    /// either or both of the input matrices) and stores the result in `acc`.
    ///
    /// Performs the operation:
    /// - `acc = beta * Op_lhs(lhs) * Op_rhs(rhs)` if `alpha` is `None` (in this case, the
    ///   preexisting values in `acc` are not read, so it is allowed to be a view over uninitialized
    ///   values if `E: Copy`),
    /// - `acc = alpha * acc + beta * Op_lhs(lhs) * Op_rhs(rhs)` if `alpha` is `Some(_)`,
    ///
    /// The left hand side and right hand side may be interpreted as triangular depending on the
    /// given corresponding matrix structure.  
    ///
    /// For the destination matrix, the result is:
    /// - fully computed if the structure is rectangular,
    /// - only the triangular half (including the diagonal) is computed if the structure is
    /// triangular,
    /// - only the strict triangular half (excluding the diagonal) is computed if the structure is
    /// strictly triangular or unit triangular.
    ///
    /// `Op_lhs` is the identity if `conj_lhs` is `Conj::No`, and the conjugation operation if it is
    /// `Conj::Yes`.  
    /// `Op_rhs` is the identity if `conj_rhs` is `Conj::No`, and the conjugation operation if it is
    /// `Conj::Yes`.  
    ///
    /// # Panics
    ///
    /// Panics if the matrix dimensions are not compatible for matrix multiplication.  
    /// i.e.  
    ///  - `acc.nrows() == lhs.nrows()`
    ///  - `acc.ncols() == rhs.ncols()`
    ///  - `lhs.ncols() == rhs.nrows()`
    ///
    ///  Additionally, matrices that are marked as triangular must be square, i.e., they must have
    ///  the same number of rows and columns.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::{
    ///     mat,
    ///     mul::triangular::{matmul_with_conj, BlockStructure},
    ///     unzipped, zipped, Conj, Mat, Parallelism,
    /// };
    ///
    /// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
    /// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
    ///
    /// let mut acc = Mat::<f64>::zeros(2, 2);
    /// let target = mat![
    ///     [
    ///         2.5 * (lhs.read(0, 0) * rhs.read(0, 0) + lhs.read(0, 1) * rhs.read(1, 0)),
    ///         0.0,
    ///     ],
    ///     [
    ///         2.5 * (lhs.read(1, 0) * rhs.read(0, 0) + lhs.read(1, 1) * rhs.read(1, 0)),
    ///         2.5 * (lhs.read(1, 0) * rhs.read(0, 1) + lhs.read(1, 1) * rhs.read(1, 1)),
    ///     ],
    /// ];
    ///
    /// matmul_with_conj(
    ///     acc.as_mut(),
    ///     BlockStructure::TriangularLower,
    ///     lhs.as_ref(),
    ///     BlockStructure::Rectangular,
    ///     Conj::No,
    ///     rhs.as_ref(),
    ///     BlockStructure::Rectangular,
    ///     Conj::No,
    ///     None,
    ///     2.5,
    ///     Parallelism::None,
    /// );
    ///
    /// zipped!(acc.as_ref(), target.as_ref())
    ///     .for_each(|unzipped!(acc, target)| assert!((acc.read() - target.read()).abs() < 1e-10));
    /// ```
    #[track_caller]
    #[inline]
    pub fn matmul_with_conj<E: ComplexField>(
        acc: MatMut<'_, E>,
        acc_structure: BlockStructure,
        lhs: MatRef<'_, E>,
        lhs_structure: BlockStructure,
        conj_lhs: Conj,
        rhs: MatRef<'_, E>,
        rhs_structure: BlockStructure,
        conj_rhs: Conj,
        alpha: Option<E>,
        beta: E,
        parallelism: Parallelism,
    ) {
        assert!(all(
            acc.nrows() == lhs.nrows(),
            acc.ncols() == rhs.ncols(),
            lhs.ncols() == rhs.nrows(),
        ));

        if !acc_structure.is_dense() {
            assert!(acc.nrows() == acc.ncols());
        }
        if !lhs_structure.is_dense() {
            assert!(lhs.nrows() == lhs.ncols());
        }
        if !rhs_structure.is_dense() {
            assert!(rhs.nrows() == rhs.ncols());
        }

        unsafe {
            matmul_unchecked(
                acc,
                acc_structure,
                lhs,
                lhs_structure,
                conj_lhs,
                rhs,
                rhs_structure,
                conj_rhs,
                alpha,
                beta,
                parallelism,
            )
        }
    }

    /// Computes the matrix product `[alpha * acc] + beta * lhs * rhs` and stores the result in
    /// `acc`.
    ///
    /// Performs the operation:
    /// - `acc = beta * lhs * rhs` if `alpha` is `None` (in this case, the preexisting values in
    ///   `acc` are not read, so it is allowed to be a view over uninitialized values if `E: Copy`),
    /// - `acc = alpha * acc + beta * lhs * rhs` if `alpha` is `Some(_)`,
    ///
    /// The left hand side and right hand side may be interpreted as triangular depending on the
    /// given corresponding matrix structure.  
    ///
    /// For the destination matrix, the result is:
    /// - fully computed if the structure is rectangular,
    /// - only the triangular half (including the diagonal) is computed if the structure is
    /// triangular,
    /// - only the strict triangular half (excluding the diagonal) is computed if the structure is
    /// strictly triangular or unit triangular.
    ///
    /// # Panics
    ///
    /// Panics if the matrix dimensions are not compatible for matrix multiplication.  
    /// i.e.  
    ///  - `acc.nrows() == lhs.nrows()`
    ///  - `acc.ncols() == rhs.ncols()`
    ///  - `lhs.ncols() == rhs.nrows()`
    ///
    ///  Additionally, matrices that are marked as triangular must be square, i.e., they must have
    ///  the same number of rows and columns.
    ///
    /// # Example
    ///
    /// ```
    /// use faer_core::{
    ///     mat,
    ///     mul::triangular::{matmul, BlockStructure},
    ///     unzipped, zipped, Conj, Mat, Parallelism,
    /// };
    ///
    /// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
    /// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
    ///
    /// let mut acc = Mat::<f64>::zeros(2, 2);
    /// let target = mat![
    ///     [
    ///         2.5 * (lhs.read(0, 0) * rhs.read(0, 0) + lhs.read(0, 1) * rhs.read(1, 0)),
    ///         0.0,
    ///     ],
    ///     [
    ///         2.5 * (lhs.read(1, 0) * rhs.read(0, 0) + lhs.read(1, 1) * rhs.read(1, 0)),
    ///         2.5 * (lhs.read(1, 0) * rhs.read(0, 1) + lhs.read(1, 1) * rhs.read(1, 1)),
    ///     ],
    /// ];
    ///
    /// matmul(
    ///     acc.as_mut(),
    ///     BlockStructure::TriangularLower,
    ///     lhs.as_ref(),
    ///     BlockStructure::Rectangular,
    ///     rhs.as_ref(),
    ///     BlockStructure::Rectangular,
    ///     None,
    ///     2.5,
    ///     Parallelism::None,
    /// );
    ///
    /// zipped!(acc.as_ref(), target.as_ref())
    ///     .for_each(|unzipped!(acc, target)| assert!((acc.read() - target.read()).abs() < 1e-10));
    /// ```
    #[track_caller]
    #[inline]
    pub fn matmul<
        E: ComplexField,
        LhsE: Conjugate<Canonical = E>,
        RhsE: Conjugate<Canonical = E>,
    >(
        acc: MatMut<'_, E>,
        acc_structure: BlockStructure,
        lhs: MatRef<'_, LhsE>,
        lhs_structure: BlockStructure,
        rhs: MatRef<'_, RhsE>,
        rhs_structure: BlockStructure,
        alpha: Option<E>,
        beta: E,
        parallelism: Parallelism,
    ) {
        let (lhs, conj_lhs) = lhs.canonicalize();
        let (rhs, conj_rhs) = rhs.canonicalize();
        matmul_with_conj(
            acc,
            acc_structure,
            lhs,
            lhs_structure,
            conj_lhs,
            rhs,
            rhs_structure,
            conj_rhs,
            alpha,
            beta,
            parallelism,
        );
    }

    unsafe fn matmul_unchecked<E: ComplexField>(
        acc: MatMut<'_, E>,
        acc_structure: BlockStructure,
        lhs: MatRef<'_, E>,
        lhs_structure: BlockStructure,
        conj_lhs: Conj,
        rhs: MatRef<'_, E>,
        rhs_structure: BlockStructure,
        conj_rhs: Conj,
        alpha: Option<E>,
        beta: E,
        parallelism: Parallelism,
    ) {
        debug_assert!(acc.nrows() == lhs.nrows());
        debug_assert!(acc.ncols() == rhs.ncols());
        debug_assert!(lhs.ncols() == rhs.nrows());

        if !acc_structure.is_dense() {
            debug_assert!(acc.nrows() == acc.ncols());
        }
        if !lhs_structure.is_dense() {
            debug_assert!(lhs.nrows() == lhs.ncols());
        }
        if !rhs_structure.is_dense() {
            debug_assert!(rhs.nrows() == rhs.ncols());
        }

        let mut acc = acc;
        let mut lhs = lhs;
        let mut rhs = rhs;

        let mut acc_structure = acc_structure;
        let mut lhs_structure = lhs_structure;
        let mut rhs_structure = rhs_structure;

        let mut conj_lhs = conj_lhs;
        let mut conj_rhs = conj_rhs;

        // if either the lhs or the rhs is triangular
        if rhs_structure.is_lower() {
            // do nothing
            false
        } else if rhs_structure.is_upper() {
            // invert acc, lhs and rhs
            acc = acc.reverse_rows_and_cols_mut();
            lhs = lhs.reverse_rows_and_cols();
            rhs = rhs.reverse_rows_and_cols();
            acc_structure = acc_structure.transpose();
            lhs_structure = lhs_structure.transpose();
            rhs_structure = rhs_structure.transpose();
            false
        } else if lhs_structure.is_lower() {
            // invert and transpose
            acc = acc.reverse_rows_and_cols_mut().transpose_mut();
            (lhs, rhs) = (
                rhs.reverse_rows_and_cols().transpose(),
                lhs.reverse_rows_and_cols().transpose(),
            );
            (conj_lhs, conj_rhs) = (conj_rhs, conj_lhs);
            (lhs_structure, rhs_structure) = (rhs_structure, lhs_structure);
            true
        } else if lhs_structure.is_upper() {
            // transpose
            acc_structure = acc_structure.transpose();
            acc = acc.transpose_mut();
            (lhs, rhs) = (rhs.transpose(), lhs.transpose());
            (conj_lhs, conj_rhs) = (conj_rhs, conj_lhs);
            (lhs_structure, rhs_structure) = (rhs_structure.transpose(), lhs_structure.transpose());
            true
        } else {
            // do nothing
            false
        };

        let clear_upper = |acc: MatMut<'_, E>, skip_diag: bool| match &alpha {
            &Some(alpha) => zipped!(acc).for_each_triangular_upper(
                if skip_diag { Diag::Skip } else { Diag::Include },
                |unzipped!(mut acc)| acc.write(alpha.faer_mul(acc.read())),
            ),

            None => zipped!(acc).for_each_triangular_upper(
                if skip_diag { Diag::Skip } else { Diag::Include },
                |unzipped!(mut acc)| acc.write(E::faer_zero()),
            ),
        };

        let skip_diag = matches!(
            acc_structure,
            BlockStructure::StrictTriangularLower
                | BlockStructure::StrictTriangularUpper
                | BlockStructure::UnitTriangularLower
                | BlockStructure::UnitTriangularUpper
        );
        let lhs_diag = lhs_structure.diag_kind();
        let rhs_diag = rhs_structure.diag_kind();

        if acc_structure.is_dense() {
            if lhs_structure.is_dense() && rhs_structure.is_dense() {
                mul(acc, lhs, rhs, alpha, beta, conj_lhs, conj_rhs, parallelism);
            } else {
                debug_assert!(rhs_structure.is_lower());

                if lhs_structure.is_dense() {
                    mat_x_lower_impl_unchecked(
                        acc,
                        lhs,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                } else if lhs_structure.is_lower() {
                    clear_upper(acc.rb_mut(), true);
                    lower_x_lower_into_lower_impl_unchecked(
                        acc,
                        false,
                        lhs,
                        lhs_diag,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                } else {
                    debug_assert!(lhs_structure.is_upper());
                    upper_x_lower_impl_unchecked(
                        acc,
                        lhs,
                        lhs_diag,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                }
            }
        } else if acc_structure.is_lower() {
            if lhs_structure.is_dense() && rhs_structure.is_dense() {
                mat_x_mat_into_lower_impl_unchecked(
                    acc,
                    skip_diag,
                    lhs,
                    rhs,
                    alpha,
                    beta,
                    conj_lhs,
                    conj_rhs,
                    parallelism,
                )
            } else {
                debug_assert!(rhs_structure.is_lower());
                if lhs_structure.is_dense() {
                    mat_x_lower_into_lower_impl_unchecked(
                        acc,
                        skip_diag,
                        lhs,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    );
                } else if lhs_structure.is_lower() {
                    lower_x_lower_into_lower_impl_unchecked(
                        acc,
                        skip_diag,
                        lhs,
                        lhs_diag,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                } else {
                    upper_x_lower_into_lower_impl_unchecked(
                        acc,
                        skip_diag,
                        lhs,
                        lhs_diag,
                        rhs,
                        rhs_diag,
                        alpha,
                        beta,
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                }
            }
        } else if lhs_structure.is_dense() && rhs_structure.is_dense() {
            mat_x_mat_into_lower_impl_unchecked(
                acc.transpose_mut(),
                skip_diag,
                rhs.transpose(),
                lhs.transpose(),
                alpha,
                beta,
                conj_rhs,
                conj_lhs,
                parallelism,
            )
        } else {
            debug_assert!(rhs_structure.is_lower());
            if lhs_structure.is_dense() {
                // lower part of lhs does not contribute to result
                upper_x_lower_into_lower_impl_unchecked(
                    acc.transpose_mut(),
                    skip_diag,
                    rhs.transpose(),
                    rhs_diag,
                    lhs.transpose(),
                    lhs_diag,
                    alpha,
                    beta,
                    conj_rhs,
                    conj_lhs,
                    parallelism,
                )
            } else if lhs_structure.is_lower() {
                if !skip_diag {
                    match &alpha {
                        &Some(alpha) => {
                            zipped!(
                                acc.rb_mut().diagonal_mut().column_vector_mut().as_2d_mut(),
                                lhs.diagonal().column_vector().as_2d(),
                                rhs.diagonal().column_vector().as_2d(),
                            )
                            .for_each(
                                |unzipped!(mut acc, lhs, rhs)| {
                                    acc.write(
                                        (alpha.faer_mul(acc.read())).faer_add(
                                            beta.faer_mul(lhs.read().faer_mul(rhs.read())),
                                        ),
                                    )
                                },
                            );
                        }
                        None => {
                            zipped!(
                                acc.rb_mut().diagonal_mut().column_vector_mut().as_2d_mut(),
                                lhs.diagonal().column_vector().as_2d(),
                                rhs.diagonal().column_vector().as_2d(),
                            )
                            .for_each(
                                |unzipped!(mut acc, lhs, rhs)| {
                                    acc.write(beta.faer_mul(lhs.read().faer_mul(rhs.read())))
                                },
                            );
                        }
                    }
                }
                clear_upper(acc.rb_mut(), true);
            } else {
                debug_assert!(lhs_structure.is_upper());
                upper_x_lower_into_lower_impl_unchecked(
                    acc.transpose_mut(),
                    skip_diag,
                    rhs.transpose(),
                    rhs_diag,
                    lhs.transpose(),
                    lhs_diag,
                    alpha,
                    beta,
                    conj_rhs,
                    conj_lhs,
                    parallelism,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        triangular::{BlockStructure, DiagonalKind},
        *,
    };
    use crate::{assert, Mat};
    use assert_approx_eq::assert_approx_eq;
    use num_complex::Complex32;

    #[test]
    fn test_stack_mat() {
        stack_mat_16x16_begin!(m, 3, 3, 1, 3, f64);
        {
            let _ = &mut m;
            dbg!(&m);
        }
    }

    #[test]
    #[ignore = "takes too long in CI"]
    fn test_matmul() {
        let random = |_, _| c32 {
            re: rand::random(),
            im: rand::random(),
        };

        let alphas = [
            None,
            Some(c32::faer_one()),
            Some(c32::faer_zero()),
            Some(random(0, 0)),
        ];

        #[cfg(not(miri))]
        let bools = [false, true];
        #[cfg(not(miri))]
        let betas = [c32::faer_one(), c32::faer_zero(), random(0, 0)];
        #[cfg(not(miri))]
        let par = [Parallelism::None, Parallelism::Rayon(0)];
        #[cfg(not(miri))]
        let conjs = [Conj::Yes, Conj::No];

        #[cfg(miri)]
        let bools = [true];
        #[cfg(miri)]
        let betas = [random(0, 0)];
        #[cfg(miri)]
        let par = [Parallelism::None];
        #[cfg(miri)]
        let conjs = [Conj::Yes];

        let big0 = 127;
        let big1 = 128;
        let big2 = 129;

        let mid0 = 15;
        let mid1 = 16;
        let mid2 = 17;
        for (m, n, k) in [
            (mid0, mid0, KC + 1),
            (big0, big1, 5),
            (big1, big0, 5),
            (big0, big2, 5),
            (big2, big0, 5),
            (mid0, mid0, 5),
            (mid1, mid1, 5),
            (mid2, mid2, 5),
            (mid0, mid1, 5),
            (mid1, mid0, 5),
            (mid0, mid2, 5),
            (mid2, mid0, 5),
            (mid0, 1, 1),
            (1, mid0, 1),
            (1, 1, mid0),
            (1, mid0, mid0),
            (mid0, 1, mid0),
            (mid0, mid0, 1),
            (1, 1, 1),
        ] {
            let a = Mat::from_fn(m, k, random);
            let b = Mat::from_fn(k, n, random);
            let acc_init = Mat::from_fn(m, n, random);

            for reverse_acc_cols in bools {
                for reverse_acc_rows in bools {
                    for reverse_b_cols in bools {
                        for reverse_b_rows in bools {
                            for reverse_a_cols in bools {
                                for reverse_a_rows in bools {
                                    for a_colmajor in bools {
                                        for b_colmajor in bools {
                                            for acc_colmajor in bools {
                                                let a = if a_colmajor {
                                                    a.to_owned()
                                                } else {
                                                    a.transpose().to_owned()
                                                };
                                                let mut a = if a_colmajor {
                                                    a.as_ref()
                                                } else {
                                                    a.as_ref().transpose()
                                                };

                                                let b = if b_colmajor {
                                                    b.to_owned()
                                                } else {
                                                    b.transpose().to_owned()
                                                };
                                                let mut b = if b_colmajor {
                                                    b.as_ref()
                                                } else {
                                                    b.as_ref().transpose()
                                                };

                                                if reverse_a_rows {
                                                    a = a.reverse_rows();
                                                }
                                                if reverse_a_cols {
                                                    a = a.reverse_cols();
                                                }
                                                if reverse_b_rows {
                                                    b = b.reverse_rows();
                                                }
                                                if reverse_b_cols {
                                                    b = b.reverse_cols();
                                                }
                                                for conj_a in conjs {
                                                    for conj_b in conjs {
                                                        for parallelism in par {
                                                            for alpha in alphas {
                                                                for beta in betas {
                                                                    for use_gemm in [true, false] {
                                                                        test_matmul_impl(
                                                                            reverse_acc_cols,
                                                                            reverse_acc_rows,
                                                                            acc_colmajor,
                                                                            m,
                                                                            n,
                                                                            conj_a,
                                                                            conj_b,
                                                                            parallelism,
                                                                            alpha,
                                                                            beta,
                                                                            use_gemm,
                                                                            &acc_init,
                                                                            a,
                                                                            b,
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn matmul_with_conj_fallback<E: ComplexField>(
        acc: MatMut<'_, E>,
        a: MatRef<'_, E>,
        conj_a: Conj,
        b: MatRef<'_, E>,
        conj_b: Conj,
        alpha: Option<E>,
        beta: E,
        parallelism: Parallelism,
    ) {
        let m = acc.nrows();
        let n = acc.ncols();
        let k = a.ncols();

        let job = |idx: usize| {
            let i = idx % m;
            let j = idx / m;
            let acc = acc.rb().submatrix(i, j, 1, 1);
            let mut acc = unsafe { acc.const_cast() };

            let mut local_acc = E::faer_zero();
            for depth in 0..k {
                let a = a.read(i, depth);
                let b = b.read(depth, j);
                local_acc = local_acc.faer_add(E::faer_mul(
                    match conj_a {
                        Conj::Yes => a.faer_conj(),
                        Conj::No => a,
                    },
                    match conj_b {
                        Conj::Yes => b.faer_conj(),
                        Conj::No => b,
                    },
                ))
            }
            match alpha {
                Some(alpha) => acc.write(
                    0,
                    0,
                    E::faer_add(acc.read(0, 0).faer_mul(alpha), local_acc.faer_mul(beta)),
                ),
                None => acc.write(0, 0, local_acc.faer_mul(beta)),
            }
        };

        crate::for_each_raw(m * n, job, parallelism);
    }

    fn test_matmul_impl(
        reverse_acc_cols: bool,
        reverse_acc_rows: bool,
        acc_colmajor: bool,
        m: usize,
        n: usize,
        conj_a: Conj,
        conj_b: Conj,
        parallelism: Parallelism,
        alpha: Option<c32>,
        beta: c32,
        use_gemm: bool,
        acc_init: &Mat<c32>,
        a: MatRef<c32>,
        b: MatRef<c32>,
    ) {
        let mut acc = if acc_colmajor {
            acc_init.to_owned()
        } else {
            acc_init.transpose().to_owned()
        };

        let mut acc = if acc_colmajor {
            acc.as_mut()
        } else {
            acc.as_mut().transpose_mut()
        };
        if reverse_acc_rows {
            acc = acc.reverse_rows_mut();
        }
        if reverse_acc_cols {
            acc = acc.reverse_cols_mut();
        }
        let mut target = acc.to_owned();

        matmul_with_conj_gemm_dispatch(
            acc.rb_mut(),
            a,
            conj_a,
            b,
            conj_b,
            alpha,
            beta,
            parallelism,
            use_gemm,
        );
        matmul_with_conj_fallback(
            target.as_mut(),
            a,
            conj_a,
            b,
            conj_b,
            alpha,
            beta,
            parallelism,
        );

        for j in 0..n {
            for i in 0..m {
                let acc: Complex32 = acc.read(i, j).into();
                let target: Complex32 = target.read(i, j).into();
                assert_approx_eq!(acc.re, target.re, 1e-3);
                assert_approx_eq!(acc.im, target.im, 1e-3);
            }
        }
    }

    fn generate_structured_matrix(
        is_dst: bool,
        nrows: usize,
        ncols: usize,
        structure: BlockStructure,
    ) -> Mat<f64> {
        let mut mat = Mat::new();
        mat.resize_with(nrows, ncols, |_, _| rand::random());

        if !is_dst {
            let kind = structure.diag_kind();
            if structure.is_lower() {
                for j in 0..ncols {
                    for i in 0..j {
                        mat.write(i, j, 0.0);
                    }
                }
            } else if structure.is_upper() {
                for j in 0..ncols {
                    for i in j + 1..nrows {
                        mat.write(i, j, 0.0);
                    }
                }
            }

            match kind {
                triangular::DiagonalKind::Zero => {
                    for i in 0..nrows {
                        mat.write(i, i, 0.0);
                    }
                }
                triangular::DiagonalKind::Unit => {
                    for i in 0..nrows {
                        mat.write(i, i, 1.0);
                    }
                }
                triangular::DiagonalKind::Generic => (),
            }
        }
        mat
    }

    fn run_test_problem(
        m: usize,
        n: usize,
        k: usize,
        dst_structure: BlockStructure,
        lhs_structure: BlockStructure,
        rhs_structure: BlockStructure,
    ) {
        let mut dst = generate_structured_matrix(true, m, n, dst_structure);
        let mut dst_target = dst.to_owned();
        let dst_orig = dst.to_owned();
        let lhs = generate_structured_matrix(false, m, k, lhs_structure);
        let rhs = generate_structured_matrix(false, k, n, rhs_structure);

        for parallelism in [Parallelism::None, Parallelism::Rayon(8)] {
            triangular::matmul_with_conj(
                dst.as_mut(),
                dst_structure,
                lhs.as_ref(),
                lhs_structure,
                Conj::No,
                rhs.as_ref(),
                rhs_structure,
                Conj::No,
                None,
                2.5,
                parallelism,
            );

            matmul_with_conj(
                dst_target.as_mut(),
                lhs.as_ref(),
                Conj::No,
                rhs.as_ref(),
                Conj::No,
                None,
                2.5,
                parallelism,
            );

            if dst_structure.is_dense() {
                for j in 0..n {
                    for i in 0..m {
                        assert_approx_eq!(dst.read(i, j), dst_target.read(i, j));
                    }
                }
            } else if dst_structure.is_lower() {
                for j in 0..n {
                    if matches!(dst_structure.diag_kind(), DiagonalKind::Generic) {
                        for i in 0..j {
                            assert_eq!(dst.read(i, j), dst_orig.read(i, j));
                        }
                        for i in j..n {
                            assert_approx_eq!(dst.read(i, j), dst_target.read(i, j));
                        }
                    } else {
                        for i in 0..=j {
                            assert_eq!(dst.read(i, j), dst_orig.read(i, j));
                        }
                        for i in j + 1..n {
                            assert_approx_eq!(dst.read(i, j), dst_target.read(i, j));
                        }
                    }
                }
            } else {
                for j in 0..n {
                    if matches!(dst_structure.diag_kind(), DiagonalKind::Generic) {
                        for i in 0..=j {
                            assert_approx_eq!(dst.read(i, j), dst_target.read(i, j));
                        }
                        for i in j + 1..n {
                            assert_eq!(dst.read(i, j), dst_orig.read(i, j));
                        }
                    } else {
                        for i in 0..j {
                            assert_approx_eq!(dst.read(i, j), dst_target.read(i, j));
                        }
                        for i in j..n {
                            assert_eq!(dst.read(i, j), dst_orig.read(i, j));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_triangular() {
        use BlockStructure::*;
        let structures = [
            Rectangular,
            TriangularLower,
            TriangularUpper,
            StrictTriangularLower,
            StrictTriangularUpper,
            UnitTriangularLower,
            UnitTriangularUpper,
        ];

        for dst in structures {
            for lhs in structures {
                for rhs in structures {
                    #[cfg(not(miri))]
                    let big = 100;

                    #[cfg(miri)]
                    let big = 31;
                    for _ in 0..3 {
                        let m = rand::random::<usize>() % big;
                        let mut n = rand::random::<usize>() % big;
                        let mut k = rand::random::<usize>() % big;

                        // for keeping track of miri progress
                        #[cfg(miri)]
                        dbg!(m, n, k);

                        match (!dst.is_dense(), !lhs.is_dense(), !rhs.is_dense()) {
                            (true, true, _) | (true, _, true) | (_, true, true) => {
                                n = m;
                                k = m;
                            }
                            _ => (),
                        }

                        if !dst.is_dense() {
                            n = m;
                        }

                        if !lhs.is_dense() {
                            k = m;
                        }

                        if !rhs.is_dense() {
                            k = n;
                        }

                        run_test_problem(m, n, k, dst, lhs, rhs);
                    }
                }
            }
        }
    }
}
