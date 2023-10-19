//! addition and subtraction of matrices

use super::*;
#[cfg(feature = "std")]
use assert2::assert;

pub struct Dense {
    __private: (),
}
pub struct Diag {
    __private: (),
}
pub struct Scale {
    __private: (),
}

pub trait MatrixKind {
    type Ref<'a, E: Entity>: Copy;
    type Mut<'a, E: Entity>;
    type Own<E: Entity>;
}
type KindRef<'a, E, K> = <K as MatrixKind>::Ref<'a, E>;
type KindMut<'a, E, K> = <K as MatrixKind>::Mut<'a, E>;
type KindOwn<E, K> = <K as MatrixKind>::Own<E>;

impl MatrixKind for Dense {
    type Ref<'a, E: Entity> = MatRef<'a, E>;
    type Mut<'a, E: Entity> = MatMut<'a, E>;
    type Own<E: Entity> = Mat<E>;
}
impl MatrixKind for Scale {
    type Ref<'a, E: Entity> = &'a MatScale<E>;
    type Mut<'a, E: Entity> = &'a mut MatScale<E>;
    type Own<E: Entity> = MatScale<E>;
}
impl MatrixKind for Diag {
    type Ref<'a, E: Entity> = Matrix<DiagRef<'a, E>>;
    type Mut<'a, E: Entity> = Matrix<DiagMut<'a, E>>;
    type Own<E: Entity> = Matrix<DiagOwn<E>>;
}

pub trait GenericMatrix: Sized {
    type Kind: MatrixKind;
    type Elem: Entity;

    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem>;
}
pub trait GenericMatrixMut: GenericMatrix {
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem>;
}

impl<E: Entity> GenericMatrix for inner::DenseRef<'_, E> {
    type Kind = Dense;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        *this
    }
}
impl<E: Entity> GenericMatrix for inner::DenseMut<'_, E> {
    type Kind = Dense;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.rb()
    }
}
impl<E: Entity> GenericMatrix for inner::DenseOwn<E> {
    type Kind = Dense;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.as_ref()
    }
}
impl<E: Entity> GenericMatrixMut for inner::DenseMut<'_, E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.rb_mut()
    }
}
impl<E: Entity> GenericMatrixMut for inner::DenseOwn<E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.as_mut()
    }
}

impl<E: Entity> GenericMatrix for inner::DiagRef<'_, E> {
    type Kind = Diag;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        *this
    }
}
impl<E: Entity> GenericMatrix for inner::DiagMut<'_, E> {
    type Kind = Diag;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.rb()
    }
}
impl<E: Entity> GenericMatrix for inner::DiagOwn<E> {
    type Kind = Diag;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.as_ref()
    }
}
impl<E: Entity> GenericMatrixMut for inner::DiagMut<'_, E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.rb_mut()
    }
}
impl<E: Entity> GenericMatrixMut for inner::DiagOwn<E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.as_mut()
    }
}

impl<E: Entity> GenericMatrix for inner::Scale<E> {
    type Kind = Scale;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this
    }
}
impl<E: Entity> GenericMatrixMut for inner::Scale<E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this
    }
}

mod __matmul_assign {
    use super::*;

    impl MatMulAssign<Scale> for Dense {
        #[track_caller]
        fn mat_mul_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
            lhs: KindMut<'_, E, Dense>,
            rhs: KindRef<'_, RhsE, Scale>,
        ) {
            let rhs = rhs.value().canonicalize();
            zipped!(lhs).for_each(|mut lhs| lhs.write(lhs.read().faer_mul(rhs)));
        }
    }
    impl MatMulAssign<Scale> for Scale {
        #[track_caller]
        fn mat_mul_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
            lhs: KindMut<'_, E, Scale>,
            rhs: KindRef<'_, RhsE, Scale>,
        ) {
            let rhs = rhs.value().canonicalize();
            *lhs = scale((*lhs).value().faer_mul(rhs));
        }
    }

    impl MatMulAssign<Diag> for Diag {
        #[track_caller]
        fn mat_mul_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
            lhs: KindMut<'_, E, Diag>,
            rhs: KindRef<'_, RhsE, Diag>,
        ) {
            zipped!(lhs.inner.inner, rhs.inner.inner)
                .for_each(|mut lhs, rhs| lhs.write(lhs.read().faer_mul(rhs.read().canonicalize())));
        }
    }
}

mod __matmul {

    use super::*;
    #[cfg(feature = "std")]
    use assert2::assert;

    impl MatMul<Dense> for Dense {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Self>,
            rhs: KindRef<'_, RhsE, Self>,
        ) -> KindOwn<E, Self::Output> {
            assert!(lhs.ncols() == rhs.nrows());
            let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
            mul::matmul(
                out.as_mut(),
                lhs,
                rhs,
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }
    impl MatMul<Dense> for Scale {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Scale>,
            rhs: KindRef<'_, RhsE, Dense>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Mat::zeros(rhs.nrows(), rhs.ncols());
            let lhs = lhs.inner.0.canonicalize();
            zipped!(out.as_mut(), rhs)
                .for_each(|mut out, rhs| out.write(E::faer_mul(lhs, rhs.read().canonicalize())));
            out
        }
    }
    impl MatMul<Scale> for Dense {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Dense>,
            rhs: KindRef<'_, RhsE, Scale>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Mat::zeros(lhs.nrows(), lhs.ncols());
            let rhs = rhs.inner.0.canonicalize();
            zipped!(out.as_mut(), lhs)
                .for_each(|mut out, lhs| out.write(E::faer_mul(lhs.read().canonicalize(), rhs)));
            out
        }
    }
    impl MatMul<Scale> for Scale {
        type Output = Scale;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Scale>,
            rhs: KindRef<'_, RhsE, Scale>,
        ) -> KindOwn<E, Self::Output> {
            scale(E::faer_mul(
                lhs.inner.0.canonicalize(),
                rhs.inner.0.canonicalize(),
            ))
        }
    }

    impl MatMul<Diag> for Dense {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Dense>,
            rhs: KindRef<'_, RhsE, Diag>,
        ) -> KindOwn<E, Self::Output> {
            let lhs_ncols = lhs.ncols();
            let rhs_dim = rhs.inner.inner.nrows();
            assert!(lhs_ncols == rhs_dim);

            Mat::from_fn(lhs.nrows(), lhs.ncols(), |i, j| unsafe {
                E::faer_mul(
                    lhs.read_unchecked(i, j).canonicalize(),
                    rhs.inner.inner.read_unchecked(j, 0).canonicalize(),
                )
            })
        }
    }

    impl MatMul<Dense> for Diag {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Diag>,
            rhs: KindRef<'_, RhsE, Dense>,
        ) -> KindOwn<E, Self::Output> {
            let lhs_dim = lhs.inner.inner.nrows();
            let rhs_nrows = rhs.nrows();
            assert!(lhs_dim == rhs_nrows);

            Mat::from_fn(rhs.nrows(), rhs.ncols(), |i, j| unsafe {
                E::faer_mul(
                    lhs.inner.inner.read_unchecked(i, 0).canonicalize(),
                    rhs.read_unchecked(i, j).canonicalize(),
                )
            })
        }
    }

    impl MatMul<Diag> for Diag {
        type Output = Diag;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Diag>,
            rhs: KindRef<'_, RhsE, Diag>,
        ) -> KindOwn<E, Self::Output> {
            let lhs_dim = lhs.inner.inner.nrows();
            let rhs_dim = rhs.inner.inner.nrows();
            assert!(lhs_dim == rhs_dim);

            Matrix {
                inner: DiagOwn {
                    inner: Mat::from_fn(lhs_dim, 1, |i, _| unsafe {
                        E::faer_mul(
                            lhs.inner.inner.read_unchecked(i, 0).canonicalize(),
                            rhs.inner.inner.read_unchecked(i, 0).canonicalize(),
                        )
                    }),
                },
            }
        }
    }
}

pub trait MatSized: MatrixKind {
    fn nrows<E: Entity>(this: KindRef<'_, E, Self>) -> usize;
    fn ncols<E: Entity>(this: KindRef<'_, E, Self>) -> usize;
}

pub trait MatDenseStorage: MatSized {
    fn row_stride<E: Entity>(this: KindRef<'_, E, Self>) -> isize;
    fn col_stride<E: Entity>(this: KindRef<'_, E, Self>) -> isize;

    fn as_ptr<E: Entity>(this: KindRef<'_, E, Self>) -> GroupFor<E, *const E::Unit>;
    fn as_mut_ptr<E: Entity>(this: KindMut<'_, E, Self>) -> GroupFor<E, *mut E::Unit>;
}

pub trait MatMulAssign<Rhs: MatrixKind>: MatrixKind {
    fn mat_mul_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    );
}
pub trait MatAddAssign<Rhs: MatrixKind>: MatrixKind {
    fn mat_add_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    );
}
pub trait MatSubAssign<Rhs: MatrixKind>: MatrixKind {
    fn mat_sub_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    );
}

pub trait MatEq<Rhs: MatrixKind>: MatrixKind {
    fn mat_eq<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    ) -> bool;
}

pub trait MatMul<Rhs: MatrixKind>: MatrixKind {
    type Output: MatrixKind;

    fn mat_mul<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    ) -> KindOwn<E, Self::Output>;
}
pub trait MatAdd<Rhs: MatrixKind>: MatrixKind {
    type Output: MatrixKind;

    fn mat_add<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    ) -> KindOwn<E, Self::Output>;
}
pub trait MatSub<Rhs: MatrixKind>: MatrixKind {
    type Output: MatrixKind;

    fn mat_sub<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    ) -> KindOwn<E, Self::Output>;
}
pub trait MatNeg: MatrixKind {
    type Output: MatrixKind;

    fn mat_neg<E: Conjugate>(mat: KindRef<'_, E, Self>) -> KindOwn<E::Canonical, Self::Output>
    where
        E::Canonical: ComplexField;
}

impl MatSized for Dense {
    #[inline(always)]
    fn nrows<E: Entity>(this: KindRef<'_, E, Self>) -> usize {
        this.inner.inner.nrows
    }

    #[inline(always)]
    fn ncols<E: Entity>(this: KindRef<'_, E, Self>) -> usize {
        this.inner.inner.ncols
    }
}

impl MatSized for Diag {
    #[inline(always)]
    fn nrows<E: Entity>(this: KindRef<'_, E, Self>) -> usize {
        this.inner.inner.nrows()
    }

    #[inline(always)]
    fn ncols<E: Entity>(this: KindRef<'_, E, Self>) -> usize {
        Diag::nrows(this)
    }
}

impl MatEq<Dense> for Dense {
    #[track_caller]
    fn mat_eq<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> bool {
        if (lhs.nrows(), lhs.ncols()) != (rhs.nrows(), rhs.ncols()) {
            return false;
        }
        let m = lhs.nrows();
        let n = lhs.ncols();
        for j in 0..n {
            for i in 0..m {
                if !(lhs.read(i, j).canonicalize() == rhs.read(i, j).canonicalize()) {
                    return false;
                }
            }
        }

        true
    }
}

impl MatAdd<Dense> for Dense {
    type Output = Dense;

    #[track_caller]
    fn mat_add<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> KindOwn<E, Self::Output> {
        assert!((lhs.nrows(), lhs.ncols()) == (rhs.nrows(), rhs.ncols()));
        let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
        zipped!(out.as_mut(), lhs, rhs).for_each(|mut out, lhs, rhs| {
            out.write(E::faer_add(
                lhs.read().canonicalize(),
                rhs.read().canonicalize(),
            ))
        });
        out
    }
}
impl MatSub<Dense> for Dense {
    type Output = Dense;

    #[track_caller]
    fn mat_sub<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> KindOwn<E, Self::Output> {
        assert!((lhs.nrows(), lhs.ncols()) == (rhs.nrows(), rhs.ncols()));
        let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
        zipped!(out.as_mut(), lhs, rhs).for_each(|mut out, lhs, rhs| {
            out.write(E::faer_sub(
                lhs.read().canonicalize(),
                rhs.read().canonicalize(),
            ))
        });
        out
    }
}
impl MatAddAssign<Dense> for Dense {
    #[track_caller]
    fn mat_add_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, Dense>,
        rhs: KindRef<'_, RhsE, Dense>,
    ) {
        zipped!(lhs, rhs)
            .for_each(|mut lhs, rhs| lhs.write(lhs.read().faer_add(rhs.read().canonicalize())));
    }
}
impl MatSubAssign<Dense> for Dense {
    #[track_caller]
    fn mat_sub_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, Dense>,
        rhs: KindRef<'_, RhsE, Dense>,
    ) {
        zipped!(lhs, rhs)
            .for_each(|mut lhs, rhs| lhs.write(lhs.read().faer_sub(rhs.read().canonicalize())));
    }
}

impl MatNeg for Dense {
    type Output = Dense;

    fn mat_neg<E: Conjugate>(mat: KindRef<'_, E, Self>) -> KindOwn<E::Canonical, Self::Output>
    where
        E::Canonical: ComplexField,
    {
        let mut out = Mat::zeros(mat.nrows(), mat.ncols());
        zipped!(out.as_mut(), mat)
            .for_each(|mut out, src| out.write(src.read().canonicalize().faer_neg()));
        out
    }
}

#[inline(always)]
pub fn scale<E: Entity>(value: E) -> Matrix<inner::Scale<E>> {
    Matrix {
        inner: inner::Scale(value),
    }
}

const _: () = {
    use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Mul<&Matrix<Rhs>> for &Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatMul<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatMul<Rhs::Kind>>::Output>;

        fn mul(self, rhs: &Matrix<Rhs>) -> Self::Output {
            <Lhs::Kind as MatMul<Rhs::Kind>>::mat_mul(
                GenericMatrix::as_ref(self),
                GenericMatrix::as_ref(rhs),
            )
        }
    }
    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Mul<&Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatMul<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatMul<Rhs::Kind>>::Output>;

        fn mul(self, rhs: &Matrix<Rhs>) -> Self::Output {
            &self * rhs
        }
    }
    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Mul<Matrix<Rhs>> for &Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatMul<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatMul<Rhs::Kind>>::Output>;

        fn mul(self, rhs: Matrix<Rhs>) -> Self::Output {
            self * &rhs
        }
    }

    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Mul<Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatMul<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatMul<Rhs::Kind>>::Output>;

        fn mul(self, rhs: Matrix<Rhs>) -> Self::Output {
            &self * &rhs
        }
    }

    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Add<&Matrix<Rhs>> for &Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatAdd<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatAdd<Rhs::Kind>>::Output>;

        fn add(self, rhs: &Matrix<Rhs>) -> Self::Output {
            <Lhs::Kind as MatAdd<Rhs::Kind>>::mat_add(
                GenericMatrix::as_ref(self),
                GenericMatrix::as_ref(rhs),
            )
        }
    }
    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Add<&Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatAdd<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatAdd<Rhs::Kind>>::Output>;

        fn add(self, rhs: &Matrix<Rhs>) -> Self::Output {
            &self + rhs
        }
    }
    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Add<Matrix<Rhs>> for &Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatAdd<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatAdd<Rhs::Kind>>::Output>;

        fn add(self, rhs: Matrix<Rhs>) -> Self::Output {
            self + &rhs
        }
    }
    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Add<Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatAdd<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatAdd<Rhs::Kind>>::Output>;

        fn add(self, rhs: Matrix<Rhs>) -> Self::Output {
            &self + &rhs
        }
    }

    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Sub<&Matrix<Rhs>> for &Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatSub<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatSub<Rhs::Kind>>::Output>;

        fn sub(self, rhs: &Matrix<Rhs>) -> Self::Output {
            <Lhs::Kind as MatSub<Rhs::Kind>>::mat_sub(
                GenericMatrix::as_ref(self),
                GenericMatrix::as_ref(rhs),
            )
        }
    }

    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Sub<&Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatSub<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatSub<Rhs::Kind>>::Output>;

        fn sub(self, rhs: &Matrix<Rhs>) -> Self::Output {
            &self - rhs
        }
    }
    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Sub<Matrix<Rhs>> for &Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatSub<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatSub<Rhs::Kind>>::Output>;

        fn sub(self, rhs: Matrix<Rhs>) -> Self::Output {
            self - &rhs
        }
    }
    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> Sub<Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatSub<Rhs::Kind>,
    {
        type Output =
            KindOwn<<Lhs::Elem as Conjugate>::Canonical, <Lhs::Kind as MatSub<Rhs::Kind>>::Output>;

        fn sub(self, rhs: Matrix<Rhs>) -> Self::Output {
            &self - &rhs
        }
    }

    impl<Mat: GenericMatrix> Neg for &Matrix<Mat>
    where
        Mat::Elem: Conjugate,
        <Mat::Elem as Conjugate>::Canonical: ComplexField,
        Mat::Kind: MatNeg,
    {
        type Output = KindOwn<<Mat::Elem as Conjugate>::Canonical, <Mat::Kind as MatNeg>::Output>;
        fn neg(self) -> Self::Output {
            <Mat::Kind as MatNeg>::mat_neg(GenericMatrix::as_ref(self))
        }
    }
    impl<Mat: GenericMatrix> Neg for Matrix<Mat>
    where
        Mat::Elem: Conjugate,
        <Mat::Elem as Conjugate>::Canonical: ComplexField,
        Mat::Kind: MatNeg,
    {
        type Output = KindOwn<<Mat::Elem as Conjugate>::Canonical, <Mat::Kind as MatNeg>::Output>;
        fn neg(self) -> Self::Output {
            -&self
        }
    }

    impl<Lhs: GenericMatrix, Rhs: GenericMatrix> PartialEq<Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: Conjugate,
        Rhs::Elem: Conjugate<Canonical = <Lhs::Elem as Conjugate>::Canonical>,
        <Lhs::Elem as Conjugate>::Canonical: ComplexField,
        Lhs::Kind: MatEq<Rhs::Kind>,
    {
        fn eq(&self, rhs: &Matrix<Rhs>) -> bool {
            <Lhs::Kind as MatEq<Rhs::Kind>>::mat_eq(
                GenericMatrix::as_ref(self),
                GenericMatrix::as_ref(rhs),
            )
        }
    }

    impl<Lhs: GenericMatrixMut, Rhs: GenericMatrix> MulAssign<&Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: ComplexField,
        Rhs::Elem: Conjugate<Canonical = Lhs::Elem>,
        Lhs::Kind: MatMulAssign<Rhs::Kind>,
    {
        fn mul_assign(&mut self, rhs: &Matrix<Rhs>) {
            <Lhs::Kind as MatMulAssign<Rhs::Kind>>::mat_mul_assign(
                GenericMatrixMut::as_mut(self),
                GenericMatrix::as_ref(rhs),
            );
        }
    }
    impl<Lhs: GenericMatrixMut, Rhs: GenericMatrix> MulAssign<Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: ComplexField,
        Rhs::Elem: Conjugate<Canonical = Lhs::Elem>,
        Lhs::Kind: MatMulAssign<Rhs::Kind>,
    {
        fn mul_assign(&mut self, rhs: Matrix<Rhs>) {
            *self *= &rhs;
        }
    }

    impl<Lhs: GenericMatrixMut, Rhs: GenericMatrix> AddAssign<&Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: ComplexField,
        Rhs::Elem: Conjugate<Canonical = Lhs::Elem>,
        Lhs::Kind: MatAddAssign<Rhs::Kind>,
    {
        fn add_assign(&mut self, rhs: &Matrix<Rhs>) {
            <Lhs::Kind as MatAddAssign<Rhs::Kind>>::mat_add_assign(
                GenericMatrixMut::as_mut(self),
                GenericMatrix::as_ref(rhs),
            );
        }
    }
    impl<Lhs: GenericMatrixMut, Rhs: GenericMatrix> AddAssign<Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: ComplexField,
        Rhs::Elem: Conjugate<Canonical = Lhs::Elem>,
        Lhs::Kind: MatAddAssign<Rhs::Kind>,
    {
        fn add_assign(&mut self, rhs: Matrix<Rhs>) {
            *self += &rhs;
        }
    }

    impl<Lhs: GenericMatrixMut, Rhs: GenericMatrix> SubAssign<&Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: ComplexField,
        Rhs::Elem: Conjugate<Canonical = Lhs::Elem>,
        Lhs::Kind: MatSubAssign<Rhs::Kind>,
    {
        fn sub_assign(&mut self, rhs: &Matrix<Rhs>) {
            <Lhs::Kind as MatSubAssign<Rhs::Kind>>::mat_sub_assign(
                GenericMatrixMut::as_mut(self),
                GenericMatrix::as_ref(rhs),
            );
        }
    }
    impl<Lhs: GenericMatrixMut, Rhs: GenericMatrix> SubAssign<Matrix<Rhs>> for Matrix<Lhs>
    where
        Lhs::Elem: ComplexField,
        Rhs::Elem: Conjugate<Canonical = Lhs::Elem>,
        Lhs::Kind: MatSubAssign<Rhs::Kind>,
    {
        fn sub_assign(&mut self, rhs: Matrix<Rhs>) {
            *self -= &rhs;
        }
    }
};

#[cfg(test)]
#[allow(non_snake_case)]
mod test {
    use crate::{mat, Mat};
    use assert_approx_eq::assert_approx_eq;

    fn matrices() -> (Mat<f64>, Mat<f64>) {
        let A = mat![[2.8, -3.3], [-1.7, 5.2], [4.6, -8.3],];

        let B = mat![[-7.9, 8.3], [4.7, -3.2], [3.8, -5.2],];
        (A, B)
    }

    #[test]
    #[should_panic]
    fn test_adding_matrices_of_different_sizes_should_panic() {
        let A = mat![[1.0, 2.0], [3.0, 4.0]];
        let B = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        _ = A + B;
    }

    #[test]
    #[should_panic]
    fn test_subtracting_two_matrices_of_different_sizes_should_panic() {
        let A = mat![[1.0, 2.0], [3.0, 4.0]];
        let B = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        _ = A - B;
    }

    #[test]
    fn test_add() {
        let (A, B) = matrices();

        let expected = mat![[-5.1, 5.0], [3.0, 2.0], [8.4, -13.5],];

        assert_matrix_approx_eq(A.as_ref() + B.as_ref(), &expected);
        assert_matrix_approx_eq(&A + &B, &expected);
        assert_matrix_approx_eq(A.as_ref() + &B, &expected);
        assert_matrix_approx_eq(&A + B.as_ref(), &expected);
        assert_matrix_approx_eq(A.as_ref() + B.clone(), &expected);
        assert_matrix_approx_eq(&A + B.clone(), &expected);
        assert_matrix_approx_eq(A.clone() + B.as_ref(), &expected);
        assert_matrix_approx_eq(A.clone() + &B, &expected);
        assert_matrix_approx_eq(A + B, &expected);
    }

    #[test]
    fn test_sub() {
        let (A, B) = matrices();

        let expected = mat![[10.7, -11.6], [-6.4, 8.4], [0.8, -3.1],];

        assert_matrix_approx_eq(A.as_ref() - B.as_ref(), &expected);
        assert_matrix_approx_eq(&A - &B, &expected);
        assert_matrix_approx_eq(A.as_ref() - &B, &expected);
        assert_matrix_approx_eq(&A - B.as_ref(), &expected);
        assert_matrix_approx_eq(A.as_ref() - B.clone(), &expected);
        assert_matrix_approx_eq(&A - B.clone(), &expected);
        assert_matrix_approx_eq(A.clone() - B.as_ref(), &expected);
        assert_matrix_approx_eq(A.clone() - &B, &expected);
        assert_matrix_approx_eq(A - B, &expected);
    }

    #[test]
    fn test_neg() {
        let (A, _) = matrices();

        let expected = mat![[-2.8, 3.3], [1.7, -5.2], [-4.6, 8.3],];

        assert_eq!(-A, expected);
    }

    #[test]
    fn test_scalar_mul() {
        use crate::scale;

        let (A, _) = matrices();
        let scale = scale(3.0);
        let expected = Mat::from_fn(A.nrows(), A.ncols(), |i, j| A.read(i, j) * scale.value());

        {
            assert_matrix_approx_eq(A.as_ref() * scale, &expected);
            assert_matrix_approx_eq(&A * scale, &expected);
            assert_matrix_approx_eq(A.as_ref() * scale, &expected);
            assert_matrix_approx_eq(&A * scale, &expected);
            assert_matrix_approx_eq(A.as_ref() * scale, &expected);
            assert_matrix_approx_eq(&A * scale, &expected);
            assert_matrix_approx_eq(A.clone() * scale, &expected);
            assert_matrix_approx_eq(A.clone() * scale, &expected);
            assert_matrix_approx_eq(A * scale, &expected);
        }

        let (A, _) = matrices();
        {
            assert_matrix_approx_eq(scale * A.as_ref(), &expected);
            assert_matrix_approx_eq(scale * &A, &expected);
            assert_matrix_approx_eq(scale * A.as_ref(), &expected);
            assert_matrix_approx_eq(scale * &A, &expected);
            assert_matrix_approx_eq(scale * A.as_ref(), &expected);
            assert_matrix_approx_eq(scale * &A, &expected);
            assert_matrix_approx_eq(scale * A.clone(), &expected);
            assert_matrix_approx_eq(scale * A.clone(), &expected);
            assert_matrix_approx_eq(scale * A, &expected);
        }
    }

    #[test]
    fn test_diag_mul() {
        let (A, _) = matrices();
        let diag_left = mat![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let diag_right = mat![[4.0, 0.0], [0.0, 5.0]];

        assert!(&diag_left * &A == diag_left.diagonal() * &A);
        assert!(&A * &diag_right == &A * diag_right.diagonal());
    }

    fn assert_matrix_approx_eq(given: Mat<f64>, expected: &Mat<f64>) {
        assert_eq!(given.nrows(), expected.nrows());
        assert_eq!(given.ncols(), expected.ncols());
        for i in 0..given.nrows() {
            for j in 0..given.ncols() {
                assert_approx_eq!(given.read(i, j), expected.read(i, j));
            }
        }
    }
}
