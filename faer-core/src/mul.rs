use crate::{
    c32, c64, simd::*, transmute_unchecked, ComplexField, Conj, Conjugate, MatMut, MatRef,
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
        a: MatRef<'_, E>,
        conj_a: Conj,
        b: MatRef<'_, E>,
        conj_b: Conj,
    ) -> E {
        assert!(a.nrows() == b.nrows());
        assert!(a.ncols() == 1);
        assert!(b.ncols() == 1);
        let nrows = a.nrows();
        let mut a = a;
        let mut b = b;
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

            if conj_a == conj_b {
                pulp::Arch::new().dispatch(AccNoConjAxB::<E> { a, b })
            } else {
                pulp::Arch::new().dispatch(AccConjAxB::<E> { a, b })
            }
        } else {
            let mut acc = E::zero();
            if conj_a == conj_b {
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

        match conj_b {
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
        a: MatRef<'_, E>,
        conj_a: Conj,
        b: MatRef<'_, E>,
        conj_b: Conj,
        alpha: Option<E>,
        beta: E,
    ) {
        if b.row_stride() == 1 {
            matvec_with_conj_impl(acc, a, conj_a, b, conj_b, alpha, beta);
        } else {
            matvec_with_conj_impl(acc, a, conj_a, b.to_owned().as_ref(), conj_b, alpha, beta);
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
        a: MatRef<'_, E>,
        conj_a: Conj,
        b: MatRef<'_, E>,
        conj_b: Conj,
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

            matvec_with_conj_impl(acc, a, conj_a, b, conj_b, beta);
        } else {
            let mut tmp = crate::Mat::<E>::zeros(m, 1);
            matvec_with_conj_impl(tmp.as_mut(), a, conj_a, b, conj_b, beta);
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
        a: MatRef<'_, E>,
        conj_a: Conj,
        b: MatRef<'_, E>,
        conj_b: Conj,
        alpha: Option<E>,
        beta: E,
    ) {
        let mut acc = acc;
        let mut a = a;
        let mut b = b;

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
                return matvec_colmajor::matvec_with_conj(acc, a, conj_a, b, conj_b, alpha, beta);
            }
            if a.col_stride() == 1 {
                return matvec_rowmajor::matvec_with_conj(acc, a, conj_a, b, conj_b, alpha, beta);
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
            let b = match conj_b {
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
        a: MatRef<'_, E>,
        conj_a: Conj,
        b: MatRef<'_, E>,
        conj_b: Conj,
        alpha: Option<E>,
        beta: E,
    ) {
        let mut acc = acc;
        let mut a = a;
        let mut b = b;
        let mut conj_a = conj_a;
        let mut conj_b = conj_b;

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

const NC: usize = 256;
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
    a: MatRef<'_, E>,
    conj_a: Conj,
    b: MatRef<'_, E>,
    conj_b: Conj,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
    use_gemm: bool,
) {
    assert!(acc.nrows() == a.nrows());
    assert!(acc.ncols() == b.ncols());
    assert!(a.ncols() == b.nrows());

    let m = acc.nrows();
    let n = acc.ncols();
    let k = a.ncols();

    if m == 1 && n == 1 {
        let mut acc = acc;
        let ab = inner_prod::inner_prod_with_conj(a.transpose(), conj_a, b, conj_b);
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
        outer_prod::outer_prod_with_conj(acc, a, conj_a, b.transpose(), conj_b, alpha, beta);
        return;
    }
    if n == 1 {
        matvec::matvec_with_conj(acc, a, conj_a, b, conj_b, alpha, beta);
        return;
    }
    if m == 1 {
        matvec::matvec_with_conj(
            acc.transpose(),
            b.transpose(),
            conj_b,
            a.transpose(),
            conj_a,
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
            let a: MatRef<'_, f32> = coe::coerce(a);
            let b: MatRef<'_, f32> = coe::coerce(b);
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
                    conj_a == Conj::Yes,
                    conj_b == Conj::Yes,
                    gemm_parallelism,
                )
            };
            return;
        }
        if coe::is_same::<f64, E>() {
            let mut acc: MatMut<'_, f64> = coe::coerce(acc);
            let a: MatRef<'_, f64> = coe::coerce(a);
            let b: MatRef<'_, f64> = coe::coerce(b);
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
                    conj_a == Conj::Yes,
                    conj_b == Conj::Yes,
                    gemm_parallelism,
                )
            };
            return;
        }
        if coe::is_same::<c32, E>() {
            let mut acc: MatMut<'_, c32> = coe::coerce(acc);
            let a: MatRef<'_, c32> = coe::coerce(a);
            let b: MatRef<'_, c32> = coe::coerce(b);
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
                    conj_a == Conj::Yes,
                    conj_b == Conj::Yes,
                    gemm_parallelism,
                )
            };
            return;
        }
        if coe::is_same::<c64, E>() {
            let mut acc: MatMut<'_, c64> = coe::coerce(acc);
            let a: MatRef<'_, c64> = coe::coerce(a);
            let b: MatRef<'_, c64> = coe::coerce(b);
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
                    conj_a == Conj::Yes,
                    conj_b == Conj::Yes,
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

        let mut a = a;
        let mut b = b;
        let mut conj_a = conj_a;
        let mut conj_b = conj_b;
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

        // let a_copy = crate::Mat::<E>::zeros(padded_m, k);
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

        let tmp = tmp.as_ref();

        unsafe {
            match alpha {
                Some(alpha) => match conj_a {
                    Conj::Yes => {
                        for j in 0..n {
                            for i in 0..m {
                                acc.write_unchecked(
                                    i,
                                    j,
                                    E::add(
                                        &acc.read_unchecked(i, j).mul(&alpha),
                                        &tmp.read_unchecked(i, j).conj().mul(&beta),
                                    ),
                                )
                            }
                        }
                    }
                    Conj::No => {
                        for j in 0..n {
                            for i in 0..m {
                                acc.write_unchecked(
                                    i,
                                    j,
                                    E::add(
                                        &acc.read_unchecked(i, j).mul(&alpha),
                                        &tmp.read_unchecked(i, j).mul(&beta),
                                    ),
                                )
                            }
                        }
                    }
                },
                None => match conj_a {
                    Conj::Yes => {
                        for j in 0..n {
                            for i in 0..m {
                                acc.write_unchecked(
                                    i,
                                    j,
                                    tmp.read_unchecked(i, j).conj().mul(&beta),
                                )
                            }
                        }
                    }
                    Conj::No => {
                        for j in 0..n {
                            for i in 0..m {
                                acc.write_unchecked(i, j, tmp.read_unchecked(i, j).mul(&beta))
                            }
                        }
                    }
                },
            }
        }
    } else {
        matmul_with_conj_fallback(acc, a, conj_a, b, conj_b, alpha, beta, parallelism);
    }
}

pub fn matmul_with_conj<E: ComplexField>(
    acc: MatMut<'_, E>,
    a: MatRef<'_, E>,
    conj_a: Conj,
    b: MatRef<'_, E>,
    conj_b: Conj,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
) {
    assert!(acc.nrows() == a.nrows());
    assert!(acc.ncols() == b.ncols());
    assert!(a.ncols() == b.nrows());
    matmul_with_conj_gemm_dispatch(acc, a, conj_a, b, conj_b, alpha, beta, parallelism, true)
}

pub fn matmul<E: ComplexField, AE: Conjugate<Canonical = E>, BE: Conjugate<Canonical = E>>(
    acc: MatMut<'_, E>,
    a: MatRef<'_, AE>,
    b: MatRef<'_, BE>,
    alpha: Option<E>,
    beta: E,
    parallelism: Parallelism,
) {
    let (a, conj_a) = a.canonicalize();
    let (b, conj_b) = b.canonicalize();
    matmul_with_conj_gemm_dispatch::<E>(acc, a, conj_a, b, conj_b, alpha, beta, parallelism, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mat;
    use assert_approx_eq::assert_approx_eq;
    use num_complex::Complex32;

    #[test]
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
            dbg!(m, n, k);
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
}
