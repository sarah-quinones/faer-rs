use crate::{
    c32, c64, simd::*, transmute_unchecked, zipped, ComplexField, Conj, Conjugate, MatMut, MatRef,
    Parallelism, SimdGroup,
};
use core::{iter::zip, marker::PhantomData, mem::MaybeUninit};
use pulp::Simd;
use reborrow::*;

pub mod inner_prod {
    use super::*;
    use assert2::assert;

    #[inline(always)]
    pub fn conditional_conj_mul_adde<const CONJ_A: bool, E: ComplexField, S: Simd>(
        simd: S,
        a: SimdGroup<E, S>,
        b: SimdGroup<E, S>,
        acc: SimdGroup<E, S>,
    ) -> SimdGroup<E, S> {
        if CONJ_A {
            E::simd_conj_mul_adde(simd, a, b, acc)
        } else {
            E::simd_mul_adde(simd, a, b, acc)
        }
    }

    #[inline(always)]
    fn a_x_b_accumulate_prologue1<const CONJ_A: bool, E: ComplexField, S: Simd>(
        simd: S,
        a: E::Group<&[E::SimdUnit<S>]>,
        b: E::Group<&[E::SimdUnit<S>]>,
    ) -> SimdGroup<E, S> {
        assert!(E::N_COMPONENTS > 0);
        let mut len = usize::MAX;
        E::map(E::as_ref(&a), |slice| len = (**slice).len());

        let zero = E::simd_splat(simd, E::zero());
        let mut acc = E::copy(&zero);
        for (a, b) in zip(E::into_iter(a), E::into_iter(b)) {
            acc = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, E::deref(a), E::deref(b), acc);
        }
        acc
    }

    #[inline(always)]
    fn a_x_b_accumulate_prologue2<const CONJ_A: bool, E: ComplexField, S: Simd>(
        simd: S,
        a: E::Group<&[E::SimdUnit<S>]>,
        b: E::Group<&[E::SimdUnit<S>]>,
    ) -> SimdGroup<E, S> {
        assert!(E::N_COMPONENTS > 0);
        let mut len = usize::MAX;
        E::map(E::as_ref(&a), |slice| len = (**slice).len());

        let zero = E::simd_splat(simd, E::zero());

        let mut acc0 = E::copy(&zero);
        let mut acc1 = E::copy(&zero);

        let (a_head, a_tail) = E::as_arrays::<2, _>(a);
        let (b_head, b_tail) = E::as_arrays::<2, _>(b);

        for (a, b) in zip(E::into_iter(a_head), E::into_iter(b_head)) {
            let [a0, a1] = E::unzip2(E::deref(a));
            let [b0, b1] = E::unzip2(E::deref(b));
            acc0 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a0, b0, acc0);
            acc1 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a1, b1, acc1);
        }
        acc0 = E::simd_add(simd, acc0, acc1);

        for (a, b) in zip(E::into_iter(a_tail), E::into_iter(b_tail)) {
            acc0 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, E::deref(a), E::deref(b), acc0);
        }

        acc0
    }

    #[inline(always)]
    fn a_x_b_accumulate_prologue4<const CONJ_A: bool, E: ComplexField, S: Simd>(
        simd: S,
        a: E::Group<&[E::SimdUnit<S>]>,
        b: E::Group<&[E::SimdUnit<S>]>,
    ) -> SimdGroup<E, S> {
        assert!(E::N_COMPONENTS > 0);
        let mut len = usize::MAX;
        E::map(E::as_ref(&a), |slice| len = (**slice).len());

        let zero = E::simd_splat(simd, E::zero());

        let mut acc0 = E::copy(&zero);
        let mut acc1 = E::copy(&zero);
        let mut acc2 = E::copy(&zero);
        let mut acc3 = E::copy(&zero);

        let (a_head, a_tail) = E::as_arrays::<4, _>(a);
        let (b_head, b_tail) = E::as_arrays::<4, _>(b);

        for (a, b) in zip(E::into_iter(a_head), E::into_iter(b_head)) {
            let [a0, a1, a2, a3] = E::unzip4(E::deref(a));
            let [b0, b1, b2, b3] = E::unzip4(E::deref(b));
            acc0 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a0, b0, acc0);
            acc1 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a1, b1, acc1);
            acc2 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a2, b2, acc2);
            acc3 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a3, b3, acc3);
        }
        acc0 = E::simd_add(simd, acc0, acc1);
        acc2 = E::simd_add(simd, acc2, acc3);

        acc0 = E::simd_add(simd, acc0, acc2);

        for (a, b) in zip(E::into_iter(a_tail), E::into_iter(b_tail)) {
            acc0 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, E::deref(a), E::deref(b), acc0);
        }

        acc0
    }

    #[inline(always)]
    fn a_x_b_accumulate_prologue8<const CONJ_A: bool, E: ComplexField, S: Simd>(
        simd: S,
        a: E::Group<&[E::SimdUnit<S>]>,
        b: E::Group<&[E::SimdUnit<S>]>,
    ) -> SimdGroup<E, S> {
        assert!(E::N_COMPONENTS > 0);
        let mut len = usize::MAX;
        E::map(E::as_ref(&a), |slice| len = (**slice).len());

        let zero = E::simd_splat(simd, E::zero());

        let mut acc0 = E::copy(&zero);
        let mut acc1 = E::copy(&zero);
        let mut acc2 = E::copy(&zero);
        let mut acc3 = E::copy(&zero);
        let mut acc4 = E::copy(&zero);
        let mut acc5 = E::copy(&zero);
        let mut acc6 = E::copy(&zero);
        let mut acc7 = E::copy(&zero);

        let (a_head, a_tail) = E::as_arrays::<8, _>(a);
        let (b_head, b_tail) = E::as_arrays::<8, _>(b);

        for (a, b) in zip(E::into_iter(a_head), E::into_iter(b_head)) {
            let [a0, a1, a2, a3, a4, a5, a6, a7] = E::unzip8(E::deref(a));
            let [b0, b1, b2, b3, b4, b5, b6, b7] = E::unzip8(E::deref(b));
            acc0 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a0, b0, acc0);
            acc1 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a1, b1, acc1);
            acc2 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a2, b2, acc2);
            acc3 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a3, b3, acc3);
            acc4 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a4, b4, acc4);
            acc5 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a5, b5, acc5);
            acc6 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a6, b6, acc6);
            acc7 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, a7, b7, acc7);
        }
        acc0 = E::simd_add(simd, acc0, acc1);
        acc2 = E::simd_add(simd, acc2, acc3);
        acc4 = E::simd_add(simd, acc4, acc5);
        acc6 = E::simd_add(simd, acc6, acc7);

        acc0 = E::simd_add(simd, acc0, acc2);
        acc4 = E::simd_add(simd, acc4, acc6);

        acc0 = E::simd_add(simd, acc0, acc4);

        for (a, b) in zip(E::into_iter(a_tail), E::into_iter(b_tail)) {
            acc0 = conditional_conj_mul_adde::<CONJ_A, E, S>(simd, E::deref(a), E::deref(b), acc0);
        }

        acc0
    }

    #[inline(always)]
    fn reduce_add<E: ComplexField, S: Simd>(a: E::Group<&[E::Unit]>) -> E {
        let mut acc = E::zero();
        for a in E::into_iter(a) {
            acc = E::add(&acc, &E::from_units(E::deref(a)));
        }
        acc
    }

    #[inline(always)]
    fn a_x_b_accumulate_simd<const CONJ_A: bool, E: ComplexField, S: Simd>(
        simd: S,
        a: E::Group<&[E::Unit]>,
        b: E::Group<&[E::Unit]>,
    ) -> E {
        let (a_head, a_tail) = slice_as_simd::<E, S>(a);
        let (b_head, b_tail) = slice_as_simd::<E, S>(b);
        let prologue = if E::N_COMPONENTS == 1 {
            a_x_b_accumulate_prologue8::<CONJ_A, E, S>(simd, a_head, b_head)
        } else if E::N_COMPONENTS == 2 {
            a_x_b_accumulate_prologue4::<CONJ_A, E, S>(simd, a_head, b_head)
        } else if E::N_COMPONENTS == 4 {
            a_x_b_accumulate_prologue2::<CONJ_A, E, S>(simd, a_head, b_head)
        } else {
            a_x_b_accumulate_prologue1::<CONJ_A, E, S>(simd, a_head, b_head)
        };

        let mut acc = reduce_add::<E, S>(one_simd_as_slice::<E, S>(E::as_ref(&prologue)));

        for (a, b) in zip(E::into_iter(a_tail), E::into_iter(b_tail)) {
            let a = E::from_units(E::deref(a));
            let a = if CONJ_A { a.conj() } else { a };
            let b = E::from_units(E::deref(b));
            acc = E::add(&acc, &E::mul(&a, &b));
        }

        acc
    }

    pub struct AccNoConjAxB<'a, E: ComplexField> {
        pub a: E::Group<&'a [E::Unit]>,
        pub b: E::Group<&'a [E::Unit]>,
    }
    pub struct AccConjAxB<'a, E: ComplexField> {
        pub a: E::Group<&'a [E::Unit]>,
        pub b: E::Group<&'a [E::Unit]>,
    }

    impl<E: ComplexField> pulp::WithSimd for AccNoConjAxB<'_, E> {
        type Output = E;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            a_x_b_accumulate_simd::<false, E, S>(simd, self.a, self.b)
        }
    }
    impl<E: ComplexField> pulp::WithSimd for AccConjAxB<'_, E> {
        type Output = E;

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            a_x_b_accumulate_simd::<true, E, S>(simd, self.a, self.b)
        }
    }

    #[inline]
    pub fn inner_prod_with_conj<E: ComplexField>(
        lhs: MatRef<'_, E>,
        conj_lhs: Conj,
        rhs: MatRef<'_, E>,
        conj_rhs: Conj,
    ) -> E {
        assert!(lhs.nrows() == rhs.nrows());
        assert!(lhs.ncols() == 1);
        assert!(rhs.ncols() == 1);
        let nrows = lhs.nrows();
        let mut a = lhs;
        let mut b = rhs;
        if a.row_stride() < 0 {
            a = a.reverse_rows();
            b = b.reverse_rows();
        }

        let res = if E::HAS_SIMD && a.row_stride() == 1 && b.row_stride() == 1 {
            let a = E::map(a.as_ptr(), |ptr| unsafe {
                core::slice::from_raw_parts(ptr, nrows)
            });
            let b = E::map(b.as_ptr(), |ptr| unsafe {
                core::slice::from_raw_parts(ptr, nrows)
            });

            if conj_lhs == conj_rhs {
                pulp::Arch::new().dispatch(AccNoConjAxB::<E> { a, b })
            } else {
                pulp::Arch::new().dispatch(AccConjAxB::<E> { a, b })
            }
        } else {
            let mut acc = E::zero();
            if conj_lhs == conj_rhs {
                for i in 0..nrows {
                    acc = acc.add(&E::mul(&a.read(i, 0), &b.read(i, 0)));
                }
            } else {
                for i in 0..nrows {
                    acc = acc.add(&E::mul(&a.read(i, 0).conj(), &b.read(i, 0)));
                }
            }
            acc
        };

        match conj_rhs {
            Conj::Yes => res.conj(),
            Conj::No => res,
        }
    }
}

mod matvec_rowmajor {
    use super::*;
    use assert2::assert;

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

        assert!(b.nrows() == n);
        assert!(b.ncols() == 1);
        assert!(acc.nrows() == m);
        assert!(acc.ncols() == 1);

        assert!(a.col_stride() == 1);
        assert!(b.row_stride() == 1);

        let mut acc = acc;

        for i in 0..m {
            let a = a.submatrix(i, 0, 1, n);
            let res = inner_prod::inner_prod_with_conj(a.transpose(), conj_a, b, conj_b);
            match &alpha {
                Some(alpha) => {
                    acc.write(i, 0, E::add(&alpha.mul(&acc.read(i, 0)), &beta.mul(&res)))
                }
                None => acc.write(i, 0, beta.mul(&res)),
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

mod matvec_colmajor {
    use super::*;
    use assert2::assert;

    pub struct NoConjImpl<'a, E: ComplexField> {
        pub acc: E::Group<&'a mut [E::Unit]>,
        pub a: E::Group<&'a [E::Unit]>,
        pub b: E,
    }
    pub struct ConjImpl<'a, E: ComplexField> {
        pub acc: E::Group<&'a mut [E::Unit]>,
        pub a: E::Group<&'a [E::Unit]>,
        pub b: E,
    }

    impl<E: ComplexField> pulp::WithSimd for NoConjImpl<'_, E> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (a_head, a_tail) = slice_as_simd::<E, S>(self.a);
            let (acc_head, acc_tail) = slice_as_mut_simd::<E, S>(self.acc);
            {
                let b = E::simd_splat(simd, self.b.clone());

                for (acc_, a) in zip(E::into_iter(acc_head), E::into_iter(a_head)) {
                    let mut acc = E::deref(E::rb(E::as_ref(&acc_)));
                    let a = E::deref(a);
                    acc = E::simd_mul_adde(simd, E::copy(&b), a, acc);
                    E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
                }
            }

            let b = self.b;
            for (acc_, a) in zip(E::into_iter(acc_tail), E::into_iter(a_tail)) {
                let mut acc = E::from_units(E::deref(E::rb(E::as_ref(&acc_))));
                let a = E::from_units(E::deref(a));
                acc = E::mul_adde(&b, &a, &acc);
                let acc = E::into_units(acc);
                E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
            }
        }
    }

    impl<E: ComplexField> pulp::WithSimd for ConjImpl<'_, E> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let (a_head, a_tail) = slice_as_simd::<E, S>(self.a);
            let (acc_head, acc_tail) = slice_as_mut_simd::<E, S>(self.acc);
            {
                let b = E::simd_splat(simd, self.b.clone());

                for (acc_, a) in zip(E::into_iter(acc_head), E::into_iter(a_head)) {
                    let mut acc = E::deref(E::rb(E::as_ref(&acc_)));
                    let a = E::deref(a);
                    acc = E::simd_conj_mul_adde(simd, a, E::copy(&b), acc);
                    E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
                }
            }

            let b = self.b;
            for (acc_, a) in zip(E::into_iter(acc_tail), E::into_iter(a_tail)) {
                let mut acc = E::from_units(E::deref(E::rb(E::as_ref(&acc_))));
                let a = E::from_units(E::deref(a));
                acc = E::conj_mul_adde(&a, &b, &acc);
                let acc = E::into_units(acc);
                E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
            }
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

        assert!(b.nrows() == n);
        assert!(b.ncols() == 1);
        assert!(acc.nrows() == m);
        assert!(acc.ncols() == 1);

        assert!(a.row_stride() == 1);
        assert!(acc.row_stride() == 1);

        let mut acc = E::map(
            acc.as_ptr(),
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
        );

        let arch = pulp::Arch::new();
        for j in 0..n {
            let a = a.submatrix(0, j, m, 1);
            let acc = E::rb_mut(E::as_mut(&mut acc));

            let a = E::map(
                a.as_ptr(),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
            );
            let b = b.read(j, 0);
            let b = match conj_b {
                Conj::Yes => b.conj(),
                Conj::No => b,
            };
            let b = b.mul(&beta);

            match conj_a {
                Conj::Yes => arch.dispatch(ConjImpl { acc, a, b }),
                Conj::No => arch.dispatch(NoConjImpl { acc, a, b }),
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
                Some(alpha) => {
                    for i in 0..m {
                        acc.write(i, 0, acc.read(i, 0).mul(&alpha));
                    }
                }
                None => {
                    for i in 0..m {
                        acc.write(i, 0, E::zero());
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
                        acc.write(i, 0, (acc.read(i, 0).mul(&alpha)).add(&tmp.read(i, 0)))
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
            acc = acc.reverse_rows();
        }
        if a.col_stride() < 0 {
            a = a.reverse_cols();
            b = b.reverse_rows();
        }

        if E::HAS_SIMD {
            if a.row_stride() == 1 {
                return matvec_colmajor::matvec_with_conj(
                    acc, a, conj_lhs, b, conj_rhs, alpha, beta,
                );
            }
            if a.col_stride() == 1 {
                return matvec_rowmajor::matvec_with_conj(
                    acc, a, conj_lhs, b, conj_rhs, alpha, beta,
                );
            }
        }

        let m = a.nrows();
        let n = a.ncols();

        match alpha {
            Some(alpha) => {
                for i in 0..m {
                    acc.write(i, 0, acc.read(i, 0).mul(&alpha));
                }
            }
            None => {
                for i in 0..m {
                    acc.write(i, 0, E::zero());
                }
            }
        }

        for j in 0..n {
            let b = b.read(j, 0);
            let b = match conj_rhs {
                Conj::Yes => b.conj(),
                Conj::No => b,
            };
            let b = b.mul(&beta);
            for i in 0..m {
                let mul = a.read(i, j).mul(&b);
                acc.write(i, 0, acc.read(i, 0).add(&mul));
            }
        }
    }
}

pub mod outer_prod {
    use super::*;
    use assert2::assert;

    struct NoConjImpl<'a, E: ComplexField> {
        acc: E::Group<&'a mut [E::Unit]>,
        alpha: Option<E>,
        a: E::Group<&'a [E::Unit]>,
        b: E,
    }
    struct ConjImpl<'a, E: ComplexField> {
        acc: E::Group<&'a mut [E::Unit]>,
        alpha: Option<E>,
        a: E::Group<&'a [E::Unit]>,
        b: E,
    }

    impl<E: ComplexField> pulp::WithSimd for NoConjImpl<'_, E> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            match self.alpha {
                Some(alpha) => {
                    if alpha == E::one() {
                        matvec_colmajor::NoConjImpl {
                            acc: self.acc,
                            a: self.a,
                            b: self.b,
                        }
                        .with_simd(simd);
                    } else {
                        let (a_head, a_tail) = slice_as_simd::<E, S>(self.a);
                        let (acc_head, acc_tail) = slice_as_mut_simd::<E, S>(self.acc);
                        {
                            let alpha = E::simd_splat(simd, alpha.clone());
                            let b = E::simd_splat(simd, self.b.clone());

                            for (acc_, a) in zip(E::into_iter(acc_head), E::into_iter(a_head)) {
                                let mut acc = E::deref(E::rb(E::as_ref(&acc_)));
                                let a = E::deref(a);
                                acc = E::simd_mul_adde(
                                    simd,
                                    E::copy(&b),
                                    a,
                                    E::simd_mul(simd, E::copy(&alpha), acc),
                                );
                                E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
                            }
                        }

                        let b = self.b;
                        for (acc_, a) in zip(E::into_iter(acc_tail), E::into_iter(a_tail)) {
                            let mut acc = E::from_units(E::deref(E::rb(E::as_ref(&acc_))));
                            let a = E::from_units(E::deref(a));
                            acc = E::mul_adde(&b, &a, &E::mul(&alpha, &acc));
                            let acc = E::into_units(acc);
                            E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
                        }
                    }
                }
                None => {
                    let (a_head, a_tail) = slice_as_simd::<E, S>(self.a);
                    let (acc_head, acc_tail) = slice_as_mut_simd::<E, S>(self.acc);
                    {
                        let b = E::simd_splat(simd, self.b.clone());

                        for (acc_, a) in zip(E::into_iter(acc_head), E::into_iter(a_head)) {
                            let a = E::deref(a);
                            let acc = E::simd_mul(simd, E::copy(&b), a);
                            E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
                        }
                    }

                    let b = self.b;
                    for (acc_, a) in zip(E::into_iter(acc_tail), E::into_iter(a_tail)) {
                        let a = E::from_units(E::deref(a));
                        let acc = E::mul(&b, &a);
                        let acc = E::into_units(acc);
                        E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
                    }
                }
            }
        }
    }

    impl<E: ComplexField> pulp::WithSimd for ConjImpl<'_, E> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            match self.alpha {
                Some(alpha) => {
                    if alpha == E::one() {
                        matvec_colmajor::ConjImpl {
                            acc: self.acc,
                            a: self.a,
                            b: self.b,
                        }
                        .with_simd(simd);
                    } else {
                        let (a_head, a_tail) = slice_as_simd::<E, S>(self.a);
                        let (acc_head, acc_tail) = slice_as_mut_simd::<E, S>(self.acc);
                        {
                            let alpha = E::simd_splat(simd, alpha.clone());
                            let b = E::simd_splat(simd, self.b.clone());

                            for (acc_, a) in zip(E::into_iter(acc_head), E::into_iter(a_head)) {
                                let mut acc = E::deref(E::rb(E::as_ref(&acc_)));
                                let a = E::deref(a);
                                acc = E::simd_conj_mul_adde(
                                    simd,
                                    a,
                                    E::copy(&b),
                                    E::simd_mul(simd, E::copy(&alpha), acc),
                                );
                                E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
                            }
                        }

                        let b = self.b;
                        for (acc_, a) in zip(E::into_iter(acc_tail), E::into_iter(a_tail)) {
                            let mut acc = E::from_units(E::deref(E::rb(E::as_ref(&acc_))));
                            let a = E::from_units(E::deref(a));
                            acc = E::conj_mul_adde(&a, &b, &E::mul(&alpha, &acc));
                            let acc = E::into_units(acc);
                            E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
                        }
                    }
                }
                None => {
                    let (a_head, a_tail) = slice_as_simd::<E, S>(self.a);
                    let (acc_head, acc_tail) = slice_as_mut_simd::<E, S>(self.acc);
                    {
                        let b = E::simd_splat(simd, self.b.clone());

                        for (acc_, a) in zip(E::into_iter(acc_head), E::into_iter(a_head)) {
                            let a = E::deref(a);
                            let acc = E::simd_conj_mul(simd, a, E::copy(&b));
                            E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
                        }
                    }

                    let b = self.b;
                    for (acc_, a) in zip(E::into_iter(acc_tail), E::into_iter(a_tail)) {
                        let a = E::from_units(E::deref(a));
                        let acc = E::mul(&a.conj(), &b);
                        let acc = E::into_units(acc);
                        E::map(E::zip(acc_, acc), |(acc_, acc)| *acc_ = acc);
                    }
                }
            }
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

        assert!(a.nrows() == m);
        assert!(a.ncols() == 1);
        assert!(b.nrows() == n);
        assert!(b.ncols() == 1);

        assert!(acc.row_stride() == 1);
        assert!(a.row_stride() == 1);

        let mut acc = acc;

        let arch = pulp::Arch::new();

        let a = E::map(
            a.as_ptr(),
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, m) },
        );

        for j in 0..n {
            let acc = acc.rb_mut();
            let acc = E::map(
                acc.ptr_at(0, j),
                #[inline(always)]
                |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, m) },
            );

            let a = E::copy(&a);
            let b = b.read(j, 0);
            let b = match conj_b {
                Conj::Yes => b.conj(),
                Conj::No => b,
            };
            let b = b.mul(&beta);

            let alpha = alpha.clone();
            match conj_a {
                Conj::Yes => arch.dispatch(ConjImpl { acc, a, b, alpha }),
                Conj::No => arch.dispatch(NoConjImpl { acc, a, b, alpha }),
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
            acc = acc.reverse_rows();
            a = a.reverse_rows();
        }
        if acc.col_stride() < 0 {
            acc = acc.reverse_cols();
            b = b.reverse_rows();
        }

        if acc.row_stride() > a.col_stride() {
            acc = acc.transpose();
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut conj_a, &mut conj_b);
        }

        if E::HAS_SIMD && acc.row_stride() == 1 {
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
                            Conj::Yes => b.conj(),
                            Conj::No => b,
                        };
                        let b = b.mul(&beta);
                        match conj_a {
                            Conj::Yes => {
                                for i in 0..m {
                                    let ab = a.read(i, 0).conj().mul(&b);
                                    acc.write(i, j, E::add(&acc.read(i, j).mul(&alpha), &ab));
                                }
                            }
                            Conj::No => {
                                for i in 0..m {
                                    let ab = a.read(i, 0).mul(&b);
                                    acc.write(i, j, E::add(&acc.read(i, j).mul(&alpha), &ab));
                                }
                            }
                        }
                    }
                }
                None => {
                    for j in 0..n {
                        let b = b.read(j, 0);
                        let b = match conj_b {
                            Conj::Yes => b.conj(),
                            Conj::No => b,
                        };
                        let b = b.mul(&beta);
                        match conj_a {
                            Conj::Yes => {
                                for i in 0..m {
                                    acc.write(i, j, a.read(i, 0).conj().mul(&b));
                                }
                            }
                            Conj::No => {
                                for i in 0..m {
                                    acc.write(i, j, a.read(i, 0).mul(&b));
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
        core::mem::size_of::<E::SimdUnit<S>>() / core::mem::size_of::<E::Unit>()
    }
}

struct Ukr<'a, const MR_DIV_N: usize, const NR: usize, const CONJ_B: bool, E: ComplexField> {
    acc: MatMut<'a, E>,
    a: MatRef<'a, E>,
    b: MatRef<'a, E>,
}

impl<const MR_DIV_N: usize, const NR: usize, const CONJ_B: bool, E: ComplexField> pulp::WithSimd
    for Ukr<'_, MR_DIV_N, NR, CONJ_B, E>
{
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { mut acc, a, b } = self;
        let lane_count = core::mem::size_of::<E::SimdUnit<S>>() / core::mem::size_of::<E::Unit>();

        let mr = MR_DIV_N * lane_count;
        let nr = NR;

        assert!(acc.nrows() == mr);
        assert!(acc.ncols() == nr);
        assert!(a.nrows() == mr);
        assert!(b.ncols() == nr);
        assert!(a.ncols() == b.nrows());
        assert!(a.row_stride() == 1);
        assert!(b.row_stride() == 1);
        assert!(acc.row_stride() == 1);

        let k = a.ncols();
        let mut local_acc = [[E::into_copy(E::simd_splat(simd, E::zero())); MR_DIV_N]; NR];

        unsafe {
            let mut one_iter = {
                #[inline(always)]
                |depth| {
                    let a = a.ptr_inbounds_at(0, depth);

                    let mut a_uninit =
                        [MaybeUninit::<E::GroupCopy<E::SimdUnit<S>>>::uninit(); MR_DIV_N];
                    for i in 0..MR_DIV_N {
                        a_uninit[i] = MaybeUninit::new(E::into_copy(E::map(
                            E::copy(&a),
                            #[inline(always)]
                            |ptr| *(ptr.add(i * lane_count) as *const E::SimdUnit<S>),
                        )));
                    }
                    let a: [E::Group<E::SimdUnit<S>>; MR_DIV_N] = transmute_unchecked(a_uninit);

                    for j in 0..NR {
                        let b = E::map(
                            b.ptr_at(depth, j),
                            #[inline(always)]
                            |ptr| E::simd_splat_unit(simd, (*ptr).clone()),
                        );
                        for i in 0..MR_DIV_N {
                            let local_acc = &mut local_acc[j][i];
                            *local_acc =
                                E::into_copy(
                                    inner_prod::conditional_conj_mul_adde::<CONJ_B, E, S>(
                                        simd,
                                        E::copy(&b),
                                        E::copy(&a[i]),
                                        E::from_copy(*local_acc),
                                    ),
                                );
                        }
                    }
                }
            };

            let mut depth = 0;
            while depth < k / 4 * 4 {
                one_iter(depth + 0);
                one_iter(depth + 1);
                one_iter(depth + 2);
                one_iter(depth + 3);
                depth += 4;
            }
            while depth < k {
                one_iter(depth);
                depth += 1;
            }

            for j in 0..NR {
                for i in 0..MR_DIV_N {
                    let acc = acc.rb_mut().ptr_inbounds_at(i * lane_count, j);
                    let mut acc_value =
                        E::map(E::copy(&acc), |acc| *(acc as *const E::SimdUnit<S>));
                    acc_value = E::simd_add(simd, acc_value, E::from_copy(local_acc[j][i]));
                    E::map(E::zip(acc, acc_value), |(acc, new_acc)| {
                        *(acc as *mut E::SimdUnit<S>) = new_acc
                    });
                }
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
/// acc, a, b are colmaajor
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
        } = acc.real_imag();
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
                Some(E::Real::one()),
                beta,
                parallelism,
            )
        };

        real_matmul(acc_re.rb_mut(), a_re, b_re, E::Real::one());
        real_matmul(acc_re.rb_mut(), a_im, b_im, E::Real::one().neg());
        real_matmul(acc_im.rb_mut(), a_re, b_im, E::Real::one());
        real_matmul(acc_im.rb_mut(), a_im, b_re, E::Real::one());

        return;
    }

    let m = acc.nrows();
    let n = acc.ncols();
    let k = a.ncols();

    let arch = pulp::Arch::new();
    let lane_count = arch.dispatch(SimdLaneCount::<E> {
        __marker: PhantomData,
    });

    let nr = MicroKernelShape::<E>::MAX_NR;
    let mr_div_n = MicroKernelShape::<E>::MAX_MR_DIV_N;
    let mr = mr_div_n * lane_count;

    assert!(acc.row_stride() == 1);
    assert!(a.row_stride() == 1);
    assert!(b.row_stride() == 1);
    assert!(m % lane_count == 0);

    let mut acc = acc;

    let mut col_outer = 0usize;
    while col_outer < n {
        let n_chunk = min(NC, n - col_outer);

        let b_panel = b.submatrix(0, col_outer, k, n_chunk);
        let acc = acc.rb_mut().submatrix(0, col_outer, m, n_chunk);

        let mut depth_outer = 0usize;
        while depth_outer < k {
            let k_chunk = min(KC, k - depth_outer);

            let a_panel = a.submatrix(0, depth_outer, m, k_chunk);
            let b_block = b_panel.submatrix(depth_outer, 0, k_chunk, n_chunk);

            let n_job_count = div_ceil(n_chunk, nr);
            let chunk_count = div_ceil(m, MC);

            let job_count = n_job_count * chunk_count;

            let job = |idx: usize| {
                assert!(acc.row_stride() == 1);
                assert!(a.row_stride() == 1);
                assert!(b.row_stride() == 1);

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
                            if MicroKernelShape::<E>::IS_2X2 {
                                match (ukr_i, ukr_j) {
                                    (2, 2) => arch.dispatch(Ukr::<2, 2, true, E> { acc, a, b }),
                                    (2, 1) => arch.dispatch(Ukr::<2, 1, true, E> { acc, a, b }),
                                    (1, 2) => arch.dispatch(Ukr::<1, 2, true, E> { acc, a, b }),
                                    (1, 1) => arch.dispatch(Ukr::<1, 1, true, E> { acc, a, b }),
                                    _ => unreachable!(),
                                }
                            } else if MicroKernelShape::<E>::IS_2X1 {
                                match (ukr_i, ukr_j) {
                                    (2, 1) => arch.dispatch(Ukr::<2, 1, true, E> { acc, a, b }),
                                    (1, 1) => arch.dispatch(Ukr::<1, 1, true, E> { acc, a, b }),
                                    _ => unreachable!(),
                                }
                            } else if MicroKernelShape::<E>::IS_1X1 {
                                match (ukr_i, ukr_j) {
                                    (1, 1) => arch.dispatch(Ukr::<1, 1, true, E> { acc, a, b }),
                                    _ => unreachable!(),
                                }
                            } else {
                                unreachable!()
                            }
                        }
                        Conj::No => {
                            if MicroKernelShape::<E>::IS_2X2 {
                                match (ukr_i, ukr_j) {
                                    (2, 2) => arch.dispatch(Ukr::<2, 2, false, E> { acc, a, b }),
                                    (2, 1) => arch.dispatch(Ukr::<2, 1, false, E> { acc, a, b }),
                                    (1, 2) => arch.dispatch(Ukr::<1, 2, false, E> { acc, a, b }),
                                    (1, 1) => arch.dispatch(Ukr::<1, 1, false, E> { acc, a, b }),
                                    _ => unreachable!(),
                                }
                            } else if MicroKernelShape::<E>::IS_2X1 {
                                match (ukr_i, ukr_j) {
                                    (2, 1) => arch.dispatch(Ukr::<2, 1, false, E> { acc, a, b }),
                                    (1, 1) => arch.dispatch(Ukr::<1, 1, false, E> { acc, a, b }),
                                    _ => unreachable!(),
                                }
                            } else if MicroKernelShape::<E>::IS_1X1 {
                                match (ukr_i, ukr_j) {
                                    (1, 1) => arch.dispatch(Ukr::<1, 1, false, E> { acc, a, b }),
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

fn div_ceil(a: usize, b: usize) -> usize {
    let d = a / b;
    let r = a % b;
    if r > 0 && b > 0 {
        d + 1
    } else {
        d
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

        let mut local_acc = E::zero();
        for depth in 0..k {
            let a = a.read(i, depth);
            let b = b.read(depth, j);
            local_acc = E::mul_adde(
                &match conj_a {
                    Conj::Yes => a.conj(),
                    Conj::No => a,
                },
                &match conj_b {
                    Conj::Yes => b.conj(),
                    Conj::No => b,
                },
                &local_acc,
            )
        }
        match &alpha {
            Some(alpha) => acc.write(
                0,
                0,
                E::add(&acc.read(0, 0).mul(alpha), &local_acc.mul(&beta)),
            ),
            None => acc.write(0, 0, local_acc.mul(&beta)),
        }
    };

    crate::for_each_raw(m * n, job, parallelism);
}

#[doc(hidden)]
pub fn matmul_with_conj_gemm_dispatch<E: ComplexField>(
    acc: MatMut<'_, E>,
    lhs: MatRef<'_, E>,
    conj_lhs: Conj,
    rhs: MatRef<'_, E>,
    conj_rhs: Conj,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
    use_gemm: bool,
) {
    assert!(acc.nrows() == lhs.nrows());
    assert!(acc.ncols() == rhs.ncols());
    assert!(lhs.ncols() == rhs.nrows());

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
                acc.write(0, 0, E::add(&acc.read(0, 0).mul(&alpha), &ab.mul(&beta)));
            }
            None => {
                acc.write(0, 0, ab.mul(&beta));
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
            acc.transpose(),
            rhs.transpose(),
            conj_rhs,
            lhs.transpose(),
            conj_lhs,
            alpha,
            beta,
        );
        return;
    }

    if use_gemm {
        let gemm_parallelism = match parallelism {
            Parallelism::None => gemm::Parallelism::None,
            Parallelism::Rayon(0) => gemm::Parallelism::Rayon(rayon::current_num_threads()),
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
                    acc.rb_mut().as_ptr(),
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
                    acc.rb_mut().as_ptr(),
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
                    acc.rb_mut().as_ptr() as *mut gemm::c32,
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
                    acc.rb_mut().as_ptr() as *mut gemm::c64,
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

    if E::HAS_SIMD {
        let arch = pulp::Arch::new();
        let lane_count = arch.dispatch(SimdLaneCount::<E> {
            __marker: PhantomData,
        });

        let mut a = lhs;
        let mut b = rhs;
        let mut conj_a = conj_lhs;
        let mut conj_b = conj_rhs;
        let mut acc = acc;

        if n < m {
            (a, b) = (b.transpose(), a.transpose());
            core::mem::swap(&mut conj_a, &mut conj_b);
            acc = acc.transpose();
        }

        if b.row_stride() < 0 {
            a = a.reverse_cols();
            b = b.reverse_rows();
        }

        let m = acc.nrows();
        let n = acc.ncols();

        let padded_m = div_ceil(m, lane_count).checked_mul(lane_count).unwrap();

        let mut a_copy = a.to_owned();
        a_copy.resize_with(padded_m, k, |_, _| E::zero());
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
                Conj::Yes => zipped!(acc, tmp).for_each(|mut acc, tmp| {
                    acc.write(E::add(
                        &acc.read().mul(&alpha),
                        &tmp.read().conj().mul(&beta),
                    ))
                }),
                Conj::No => zipped!(acc, tmp).for_each(|mut acc, tmp| {
                    acc.write(E::add(&acc.read().mul(&alpha), &tmp.read().mul(&beta)))
                }),
            },
            None => match conj_a {
                Conj::Yes => {
                    zipped!(acc, tmp)
                        .for_each(|mut acc, tmp| acc.write(tmp.read().conj().mul(&beta)));
                }
                Conj::No => {
                    zipped!(acc, tmp).for_each(|mut acc, tmp| acc.write(tmp.read().mul(&beta)));
                }
            },
        }
    } else {
        matmul_with_conj_fallback(acc, lhs, conj_lhs, rhs, conj_rhs, alpha, beta, parallelism);
    }
}

#[inline]
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
    assert!(acc.nrows() == lhs.nrows());
    assert!(acc.ncols() == rhs.ncols());
    assert!(lhs.ncols() == rhs.nrows());
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

/// Triangular matrix multiplication module, where some of the operands are treated as triangular
/// matrices.
pub mod triangular {
    use super::*;
    use crate::{join_raw, zip::Diag};
    use assert2::{assert, debug_assert};

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
                    dst.write_unchecked(j, j, E::zero());
                }
                true
            }
            DiagonalKind::Unit => {
                for j in 0..n {
                    dst.write_unchecked(j, j, E::one());
                }
                true
            }
            DiagonalKind::Generic => false,
        };

        zipped!(dst.rb_mut()).for_each_triangular_upper(Diag::Skip, |mut dst| dst.write(E::zero()));
        zipped!(dst, src).for_each_triangular_lower(
            if strict { Diag::Skip } else { Diag::Include },
            |mut dst, src| dst.write(src.read()),
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
                    |mut dst, src| dst.write(alpha.mul(&dst.read().add(&src.read()))),
                );
            }
            None => {
                zipped!(dst, src).for_each_triangular_lower(
                    if skip_diag { Diag::Skip } else { Diag::Include },
                    |mut dst, src| dst.write(src.read()),
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
        copy_lower(dst.transpose(), src.transpose(), src_diag)
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
            let mut dst_buffer = crate::Mat::zeros(n, n);
            let mut temp_dst = dst_buffer.as_mut();
            let mut rhs_buffer = crate::Mat::zeros(n, n);
            let mut temp_rhs = rhs_buffer.as_mut();
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);
            mul(
                temp_dst.rb_mut(),
                lhs,
                temp_rhs.into_const(),
                None,
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            accum_lower(dst, temp_dst.into_const(), skip_diag, alpha);
        } else {
            let bs = n / 2;

            let [mut dst_top_left, _, mut dst_bot_left, dst_bot_right] = dst.split_at(bs, bs);
            let [lhs_top_left, lhs_top_right, lhs_bot_left, lhs_bot_right] = lhs.split_at(bs, bs);
            let [rhs_top_left, _, rhs_bot_left, rhs_bot_right] = rhs.split_at(bs, bs);

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
                alpha.clone(),
                beta.clone(),
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
                alpha.clone(),
                beta.clone(),
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
                beta.clone(),
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_mat_into_lower_impl_unchecked(
                dst_top_left,
                skip_diag,
                lhs_top_right,
                rhs_bot_left,
                Some(E::one()),
                beta.clone(),
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_lower_impl_unchecked(
                dst_bot_left,
                lhs_bot_left,
                rhs_top_left,
                rhs_diag,
                Some(E::one()),
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
                    let mut rhs_buffer = crate::Mat::zeros(n, n);
                    let mut temp_rhs = rhs_buffer.as_mut();

                    copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);
                    let temp_rhs = temp_rhs.into_const();

                    mul(
                        dst,
                        lhs,
                        temp_rhs,
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

            let [rhs_top_left, _, rhs_bot_left, rhs_bot_right] = rhs.split_at(bs, bs);
            let [lhs_left, lhs_right] = lhs.split_at_col(bs);
            let [mut dst_left, mut dst_right] = dst.split_at_col(bs);

            join_raw(
                |parallelism| {
                    mat_x_lower_impl_unchecked(
                        dst_left.rb_mut(),
                        lhs_left,
                        rhs_top_left,
                        rhs_diag,
                        alpha.clone(),
                        beta.clone(),
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
                        alpha.clone(),
                        beta.clone(),
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
                Some(E::one()),
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
            let mut dst_buffer = crate::Mat::zeros(n, n);
            let mut temp_dst = dst_buffer.as_mut();
            let mut lhs_buffer = crate::Mat::zeros(n, n);
            let mut temp_lhs = lhs_buffer.as_mut();
            let mut rhs_buffer = crate::Mat::zeros(n, n);
            let mut temp_rhs = rhs_buffer.as_mut();

            copy_lower(temp_lhs.rb_mut(), lhs, lhs_diag);
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

            let temp_lhs = temp_lhs.into_const();
            let temp_rhs = temp_rhs.into_const();
            mul(
                temp_dst.rb_mut(),
                temp_lhs,
                temp_rhs,
                None,
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            accum_lower(dst, temp_dst.into_const(), skip_diag, alpha);
        } else {
            let bs = n / 2;

            let [dst_top_left, _, mut dst_bot_left, dst_bot_right] = dst.split_at(bs, bs);
            let [lhs_top_left, _, lhs_bot_left, lhs_bot_right] = lhs.split_at(bs, bs);
            let [rhs_top_left, _, rhs_bot_left, rhs_bot_right] = rhs.split_at(bs, bs);

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
                alpha.clone(),
                beta.clone(),
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_lower_impl_unchecked(
                dst_bot_left.rb_mut(),
                lhs_bot_left,
                rhs_top_left,
                rhs_diag,
                alpha.clone(),
                beta.clone(),
                conj_lhs,
                conj_rhs,
                parallelism,
            );
            mat_x_lower_impl_unchecked(
                dst_bot_left.reverse_rows_and_cols().transpose(),
                rhs_bot_left.reverse_rows_and_cols().transpose(),
                lhs_bot_right.reverse_rows_and_cols().transpose(),
                lhs_diag,
                Some(E::one()),
                beta.clone(),
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
            let mut lhs_buffer = crate::Mat::zeros(n, n);
            let mut temp_lhs = lhs_buffer.as_mut();
            let mut rhs_buffer = crate::Mat::zeros(n, n);
            let mut temp_rhs = rhs_buffer.as_mut();

            copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

            let temp_lhs = temp_lhs.into_const();
            let temp_rhs = temp_rhs.into_const();
            mul(
                dst,
                temp_lhs,
                temp_rhs,
                alpha,
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );
        } else {
            let bs = n / 2;

            let [mut dst_top_left, dst_top_right, dst_bot_left, dst_bot_right] =
                dst.split_at(bs, bs);
            let [lhs_top_left, lhs_top_right, _, lhs_bot_right] = lhs.split_at(bs, bs);
            let [rhs_top_left, _, rhs_bot_left, rhs_bot_right] = rhs.split_at(bs, bs);

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
                        alpha.clone(),
                        beta.clone(),
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
                        Some(E::one()),
                        beta.clone(),
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
                                alpha.clone(),
                                beta.clone(),
                                conj_lhs,
                                conj_rhs,
                                parallelism,
                            )
                        },
                        |_| {
                            mat_x_lower_impl_unchecked(
                                dst_bot_left.transpose(),
                                rhs_bot_left.transpose(),
                                lhs_bot_right.transpose(),
                                lhs_diag,
                                alpha.clone(),
                                beta.clone(),
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
                        alpha.clone(),
                        beta.clone(),
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
        mut dst: MatMut<'_, E>,
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
            let mut dst_buffer = crate::Mat::zeros(n, n);
            let mut temp_dst = dst_buffer.as_mut();
            let mut lhs_buffer = crate::Mat::zeros(n, n);
            let mut temp_lhs = lhs_buffer.as_mut();
            let mut rhs_buffer = crate::Mat::zeros(n, n);
            let mut temp_rhs = rhs_buffer.as_mut();

            copy_upper(temp_lhs.rb_mut(), lhs, lhs_diag);
            copy_lower(temp_rhs.rb_mut(), rhs, rhs_diag);

            let temp_lhs = temp_lhs.into_const();
            let temp_rhs = temp_rhs.into_const();
            mul(
                temp_dst.rb_mut(),
                temp_lhs,
                temp_rhs,
                None,
                beta,
                conj_lhs,
                conj_rhs,
                parallelism,
            );

            accum_lower(dst.rb_mut(), temp_dst.into_const(), skip_diag, alpha);
        } else {
            let bs = n / 2;

            let [mut dst_top_left, _, dst_bot_left, dst_bot_right] = dst.split_at(bs, bs);
            let [lhs_top_left, lhs_top_right, _, lhs_bot_right] = lhs.split_at(bs, bs);
            let [rhs_top_left, _, rhs_bot_left, rhs_bot_right] = rhs.split_at(bs, bs);

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
                        alpha.clone(),
                        beta.clone(),
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
                        Some(E::one()),
                        beta.clone(),
                        conj_lhs,
                        conj_rhs,
                        parallelism,
                    )
                },
                |_| {
                    mat_x_lower_impl_unchecked(
                        dst_bot_left.transpose(),
                        rhs_bot_left.transpose(),
                        lhs_bot_right.transpose(),
                        lhs_diag,
                        alpha.clone(),
                        beta.clone(),
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
                        alpha.clone(),
                        beta.clone(),
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
            let mut dst_buffer = crate::Mat::zeros(n, n);
            let mut temp_dst = dst_buffer.as_mut();
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
            accum_lower(dst, temp_dst.rb(), skip_diag, alpha)
        } else {
            let bs = n / 2;
            let [dst_top_left, _, dst_bot_left, dst_bot_right] = dst.split_at(bs, bs);
            let [lhs_top, lhs_bot] = lhs.split_at_row(bs);
            let [rhs_left, rhs_right] = rhs.split_at_col(bs);

            join_raw(
                |_| {
                    mul(
                        dst_bot_left,
                        lhs_bot,
                        rhs_left,
                        alpha.clone(),
                        beta.clone(),
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
                                alpha.clone(),
                                beta.clone(),
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
                                alpha.clone(),
                                beta.clone(),
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

    /// Computes the matrix product `[alpha * acc] + beta * Op_lhs(lhs) * Op_rhs(rhs)` and
    /// stores the result in `acc`.
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
    /// If `alpha` is not provided, he preexisting values in `acc` are not read so it is allowed to
    /// be a view over uninitialized values if `E: Copy`.
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
    ///     zipped, Conj, Mat, Parallelism,
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
    ///     .for_each(|acc, target| assert!((acc.read() - target.read()).abs() < 1e-10));
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
        assert!(acc.nrows() == lhs.nrows());
        assert!(acc.ncols() == rhs.ncols());
        assert!(lhs.ncols() == rhs.nrows());

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
            acc = acc.reverse_rows_and_cols();
            lhs = lhs.reverse_rows_and_cols();
            rhs = rhs.reverse_rows_and_cols();
            acc_structure = acc_structure.transpose();
            lhs_structure = lhs_structure.transpose();
            rhs_structure = rhs_structure.transpose();
            false
        } else if lhs_structure.is_lower() {
            // invert and transpose
            acc = acc.reverse_rows_and_cols().transpose();
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
            acc = acc.transpose();
            (lhs, rhs) = (rhs.transpose(), lhs.transpose());
            (conj_lhs, conj_rhs) = (conj_rhs, conj_lhs);
            (lhs_structure, rhs_structure) = (rhs_structure.transpose(), lhs_structure.transpose());
            true
        } else {
            // do nothing
            false
        };

        let clear_upper = |acc: MatMut<'_, E>, skip_diag: bool| match &alpha {
            Some(alpha) => zipped!(acc).for_each_triangular_upper(
                if skip_diag { Diag::Skip } else { Diag::Include },
                |mut acc| acc.write(alpha.mul(&acc.read())),
            ),

            None => zipped!(acc).for_each_triangular_upper(
                if skip_diag { Diag::Skip } else { Diag::Include },
                |mut acc| acc.write(E::zero()),
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
                acc.transpose(),
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
                    acc.transpose(),
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
                        Some(alpha) => {
                            zipped!(acc.rb_mut().diagonal(), lhs.diagonal(), rhs.diagonal())
                                .for_each(|mut acc, lhs, rhs| {
                                    acc.write(
                                        (alpha.mul(&acc.read()))
                                            .add(&beta.mul(&lhs.read().mul(&rhs.read()))),
                                    )
                                });
                        }
                        None => {
                            zipped!(acc.rb_mut().diagonal(), lhs.diagonal(), rhs.diagonal())
                                .for_each(|mut acc, lhs, rhs| {
                                    acc.write(beta.mul(&lhs.read().mul(&rhs.read())))
                                });
                        }
                    }
                }
                clear_upper(acc.rb_mut(), true);
            } else {
                debug_assert!(lhs_structure.is_upper());
                upper_x_lower_into_lower_impl_unchecked(
                    acc.transpose(),
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
    use crate::Mat;
    use assert_approx_eq::assert_approx_eq;
    use num_complex::Complex32;

    #[test]
    #[ignore = "this takes too long to launch in CI"]
    fn test_matmul() {
        let random = |_, _| c32 {
            re: rand::random(),
            im: rand::random(),
        };

        let alphas = [
            None,
            Some(c32::one()),
            Some(c32::zero()),
            Some(random(0, 0)),
        ];

        #[cfg(not(miri))]
        let bools = [false, true];
        #[cfg(not(miri))]
        let betas = [c32::one(), c32::zero(), random(0, 0)];
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
            let a = Mat::with_dims(m, k, random);
            let b = Mat::with_dims(k, n, random);
            let acc_init = Mat::with_dims(m, n, random);

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
            acc.as_mut().transpose()
        };
        if reverse_acc_rows {
            acc = acc.reverse_rows();
        }
        if reverse_acc_cols {
            acc = acc.reverse_cols();
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
                assert_approx_eq!(acc, target, 1e-3);
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
