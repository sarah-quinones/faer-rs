//! addition and subtraction of matrices

use super::*;
use crate::{
    assert,
    permutation::{Index, SignedIndex},
    sparse,
};

/// Scalar value tag.
pub struct Scalar {
    __private: (),
}
/// Dense column vector tag.
pub struct DenseCol {
    __private: (),
}
/// Dense row vector tag.
pub struct DenseRow {
    __private: (),
}
/// Dense matrix tag.
pub struct Dense {
    __private: (),
}
/// Sparse column-major matrix tag.
pub struct SparseColMat<I: Index> {
    __private: PhantomData<I>,
}
/// Sparse row-major matrix tag.
pub struct SparseRowMat<I: Index> {
    __private: PhantomData<I>,
}
/// Diagonal matrix tag.
pub struct Diag {
    __private: (),
}
/// Scaling factor tag.
pub struct Scale {
    __private: (),
}
/// Permutation matrix tag.
pub struct Perm<I> {
    __private: PhantomData<I>,
}

/// Trait for describing the view and owning variants of a given matrix type tag.
pub trait MatrixKind {
    /// Immutable view type.
    type Ref<'a, E: Entity>: Copy;
    /// Mutable view type.
    type Mut<'a, E: Entity>;
    /// Owning type.
    type Own<E: Entity>;
}
type KindRef<'a, E, K> = <K as MatrixKind>::Ref<'a, E>;
type KindMut<'a, E, K> = <K as MatrixKind>::Mut<'a, E>;
type KindOwn<E, K> = <K as MatrixKind>::Own<E>;

impl MatrixKind for Scalar {
    type Ref<'a, E: Entity> = &'a E;
    type Mut<'a, E: Entity> = &'a mut E;
    type Own<E: Entity> = E;
}
impl MatrixKind for DenseCol {
    type Ref<'a, E: Entity> = ColRef<'a, E>;
    type Mut<'a, E: Entity> = ColMut<'a, E>;
    type Own<E: Entity> = Col<E>;
}
impl MatrixKind for DenseRow {
    type Ref<'a, E: Entity> = RowRef<'a, E>;
    type Mut<'a, E: Entity> = RowMut<'a, E>;
    type Own<E: Entity> = Row<E>;
}
impl MatrixKind for Dense {
    type Ref<'a, E: Entity> = MatRef<'a, E>;
    type Mut<'a, E: Entity> = MatMut<'a, E>;
    type Own<E: Entity> = Mat<E>;
}
impl<I: Index> MatrixKind for SparseColMat<I> {
    type Ref<'a, E: Entity> = sparse::SparseColMatRef<'a, I, E>;
    type Mut<'a, E: Entity> = sparse::SparseColMatMut<'a, I, E>;
    type Own<E: Entity> = sparse::SparseColMat<I, E>;
}
impl<I: Index> MatrixKind for SparseRowMat<I> {
    type Ref<'a, E: Entity> = sparse::SparseRowMatRef<'a, I, E>;
    type Mut<'a, E: Entity> = sparse::SparseRowMatMut<'a, I, E>;
    type Own<E: Entity> = sparse::SparseRowMat<I, E>;
}
impl MatrixKind for Scale {
    type Ref<'a, E: Entity> = &'a MatScale<E>;
    type Mut<'a, E: Entity> = &'a mut MatScale<E>;
    type Own<E: Entity> = MatScale<E>;
}
impl MatrixKind for Diag {
    type Ref<'a, E: Entity> = Matrix<inner::DiagRef<'a, E>>;
    type Mut<'a, E: Entity> = Matrix<inner::DiagMut<'a, E>>;
    type Own<E: Entity> = Matrix<inner::DiagOwn<E>>;
}
impl<I: Index> MatrixKind for Perm<I> {
    type Ref<'a, E: Entity> = Matrix<inner::PermRef<'a, I, E>>;
    type Mut<'a, E: Entity> = Matrix<inner::PermMut<'a, I, E>>;
    type Own<E: Entity> = Matrix<inner::PermOwn<I, E>>;
}

/// Generic matrix trait.
pub trait GenericMatrix: Sized {
    /// Tag type.
    type Kind: MatrixKind;
    /// Scalar element type.
    type Elem: Entity;

    /// Returns a view over the matrix.
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem>;
}
/// Generic mutable matrix trait.
pub trait GenericMatrixMut: GenericMatrix {
    /// Returns a mutable over the matrix.
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem>;
}

impl<I: Index, E: Entity> GenericMatrix for inner::PermRef<'_, I, E> {
    type Kind = Perm<I>;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        *this
    }
}
impl<I: Index, E: Entity> GenericMatrix for inner::PermMut<'_, I, E> {
    type Kind = Perm<I>;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.rb()
    }
}
impl<I: Index, E: Entity> GenericMatrix for inner::PermOwn<I, E> {
    type Kind = Perm<I>;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.as_ref()
    }
}

impl<E: Entity> GenericMatrix for inner::DenseRowRef<'_, E> {
    type Kind = DenseRow;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        *this
    }
}
impl<E: Entity> GenericMatrix for inner::DenseRowMut<'_, E> {
    type Kind = DenseRow;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.rb()
    }
}
impl<E: Entity> GenericMatrix for inner::DenseRowOwn<E> {
    type Kind = DenseRow;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.as_ref()
    }
}
impl<E: Entity> GenericMatrixMut for inner::DenseRowMut<'_, E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.rb_mut()
    }
}
impl<E: Entity> GenericMatrixMut for inner::DenseRowOwn<E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.as_mut()
    }
}

impl<E: Entity> GenericMatrix for inner::DenseColRef<'_, E> {
    type Kind = DenseCol;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        *this
    }
}
impl<E: Entity> GenericMatrix for inner::DenseColMut<'_, E> {
    type Kind = DenseCol;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.rb()
    }
}
impl<E: Entity> GenericMatrix for inner::DenseColOwn<E> {
    type Kind = DenseCol;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.as_ref()
    }
}
impl<E: Entity> GenericMatrixMut for inner::DenseColMut<'_, E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.rb_mut()
    }
}
impl<E: Entity> GenericMatrixMut for inner::DenseColOwn<E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.as_mut()
    }
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

impl<I: Index, E: Entity> GenericMatrix for inner::SparseColMatRef<'_, I, E> {
    type Kind = SparseColMat<I>;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        *this
    }
}

impl<I: Index, E: Entity> GenericMatrix for inner::SparseRowMatRef<'_, I, E> {
    type Kind = SparseRowMat<I>;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        *this
    }
}

impl<I: Index, E: Entity> GenericMatrix for inner::SparseColMatMut<'_, I, E> {
    type Kind = SparseColMat<I>;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.rb()
    }
}

impl<I: Index, E: Entity> GenericMatrix for inner::SparseRowMatMut<'_, I, E> {
    type Kind = SparseRowMat<I>;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.rb()
    }
}

impl<I: Index, E: Entity> GenericMatrixMut for inner::SparseColMatMut<'_, I, E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.rb_mut()
    }
}

impl<I: Index, E: Entity> GenericMatrixMut for inner::SparseRowMatMut<'_, I, E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.rb_mut()
    }
}

impl<I: Index, E: Entity> GenericMatrix for inner::SparseColMat<I, E> {
    type Kind = SparseColMat<I>;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.as_ref()
    }
}

impl<I: Index, E: Entity> GenericMatrix for inner::SparseRowMat<I, E> {
    type Kind = SparseRowMat<I>;
    type Elem = E;

    #[inline(always)]
    fn as_ref(this: &Matrix<Self>) -> <Self::Kind as MatrixKind>::Ref<'_, Self::Elem> {
        this.as_ref()
    }
}

impl<I: Index, E: Entity> GenericMatrixMut for inner::SparseColMat<I, E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.as_mut()
    }
}

impl<I: Index, E: Entity> GenericMatrixMut for inner::SparseRowMat<I, E> {
    #[inline(always)]
    fn as_mut(this: &mut Matrix<Self>) -> <Self::Kind as MatrixKind>::Mut<'_, Self::Elem> {
        this.as_mut()
    }
}

mod __matmul_assign {
    use super::*;

    impl MatMulAssign<Scale> for DenseCol {
        #[track_caller]
        fn mat_mul_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
            lhs: KindMut<'_, E, DenseCol>,
            rhs: KindRef<'_, RhsE, Scale>,
        ) {
            let rhs = rhs.value().canonicalize();
            zipped!(lhs.as_2d_mut())
                .for_each(|unzipped!(mut lhs)| lhs.write(lhs.read().faer_mul(rhs)));
        }
    }
    impl MatMulAssign<Scale> for DenseRow {
        #[track_caller]
        fn mat_mul_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
            lhs: KindMut<'_, E, DenseRow>,
            rhs: KindRef<'_, RhsE, Scale>,
        ) {
            let rhs = rhs.value().canonicalize();
            zipped!(lhs.as_2d_mut())
                .for_each(|unzipped!(mut lhs)| lhs.write(lhs.read().faer_mul(rhs)));
        }
    }
    impl MatMulAssign<Scale> for Dense {
        #[track_caller]
        fn mat_mul_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
            lhs: KindMut<'_, E, Dense>,
            rhs: KindRef<'_, RhsE, Scale>,
        ) {
            let rhs = rhs.value().canonicalize();
            zipped!(lhs).for_each(|unzipped!(mut lhs)| lhs.write(lhs.read().faer_mul(rhs)));
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
            zipped!(lhs.inner.inner.as_2d_mut(), rhs.inner.inner.as_2d()).for_each(
                |unzipped!(mut lhs, rhs)| lhs.write(lhs.read().faer_mul(rhs.read().canonicalize())),
            );
        }
    }
}

mod __matmul {
    use super::*;
    use crate::{assert, permutation::Permutation};

    impl<I: Index> MatMul<Perm<I>> for Perm<I> {
        type Output = Perm<I>;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Perm<I>>,
            rhs: KindRef<'_, RhsE, Perm<I>>,
        ) -> KindOwn<E, Self::Output> {
            assert!(lhs.len() == rhs.len());
            let truncate = <I::Signed as SignedIndex>::truncate;
            let mut fwd = alloc::vec![I::from_signed(truncate(0)); lhs.len()].into_boxed_slice();
            let mut inv = alloc::vec![I::from_signed(truncate(0)); lhs.len()].into_boxed_slice();

            for (fwd, rhs) in fwd.iter_mut().zip(rhs.inner.forward) {
                *fwd = lhs.inner.forward[rhs.to_signed().zx()];
            }
            for (i, fwd) in fwd.iter().enumerate() {
                inv[fwd.to_signed().zx()] = I::from_signed(I::Signed::truncate(i));
            }

            Permutation {
                inner: inner::PermOwn {
                    forward: fwd,
                    inverse: inv,
                    __marker: core::marker::PhantomData,
                },
            }
        }
    }

    impl<I: Index> MatMul<DenseCol> for Perm<I> {
        type Output = DenseCol;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Perm<I>>,
            rhs: KindRef<'_, RhsE, DenseCol>,
        ) -> KindOwn<E, Self::Output> {
            assert!(lhs.len() == rhs.nrows());
            let mut out = Col::zeros(rhs.nrows());
            let fwd = lhs.inner.forward;
            for (i, fwd) in fwd.iter().enumerate() {
                out.write(i, rhs.read(fwd.to_signed().zx()).canonicalize());
            }
            out
        }
    }
    impl<I: Index> MatMul<Dense> for Perm<I> {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Perm<I>>,
            rhs: KindRef<'_, RhsE, Dense>,
        ) -> KindOwn<E, Self::Output> {
            assert!(lhs.len() == rhs.nrows());
            let mut out = Mat::zeros(rhs.nrows(), rhs.ncols());
            let fwd = lhs.inner.forward;

            for j in 0..rhs.ncols() {
                for (i, fwd) in fwd.iter().enumerate() {
                    out.write(i, j, rhs.read(fwd.to_signed().zx(), j).canonicalize());
                }
            }
            out
        }
    }
    impl<I: Index> MatMul<Perm<I>> for DenseRow {
        type Output = DenseRow;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, DenseRow>,
            rhs: KindRef<'_, RhsE, Perm<I>>,
        ) -> KindOwn<E, Self::Output> {
            assert!(lhs.ncols() == rhs.len());
            let mut out = Row::zeros(lhs.ncols());
            let inv = rhs.inner.inverse;

            for (j, inv) in inv.iter().enumerate() {
                out.write(j, lhs.read(inv.to_signed().zx()).canonicalize());
            }
            out
        }
    }
    impl<I: Index> MatMul<Perm<I>> for Dense {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Dense>,
            rhs: KindRef<'_, RhsE, Perm<I>>,
        ) -> KindOwn<E, Self::Output> {
            assert!(lhs.ncols() == rhs.len());
            let mut out = Mat::zeros(lhs.nrows(), lhs.ncols());
            let inv = rhs.inner.inverse;

            for (j, inv) in inv.iter().enumerate() {
                for i in 0..lhs.nrows() {
                    out.write(i, j, lhs.read(i, inv.to_signed().zx()).canonicalize());
                }
            }
            out
        }
    }

    impl MatMul<DenseCol> for Scale {
        type Output = DenseCol;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Scale>,
            rhs: KindRef<'_, RhsE, DenseCol>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Col::<E>::zeros(rhs.nrows());
            let lhs = lhs.inner.0.canonicalize();
            zipped!(out.as_mut().as_2d_mut(), rhs.as_2d()).for_each(|unzipped!(mut out, rhs)| {
                out.write(E::faer_mul(lhs, rhs.read().canonicalize()))
            });
            out
        }
    }
    impl MatMul<Scale> for DenseCol {
        type Output = DenseCol;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, DenseCol>,
            rhs: KindRef<'_, RhsE, Scale>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Col::<E>::zeros(lhs.nrows());
            let rhs = rhs.inner.0.canonicalize();
            zipped!(out.as_mut().as_2d_mut(), lhs.as_2d()).for_each(|unzipped!(mut out, lhs)| {
                out.write(E::faer_mul(lhs.read().canonicalize(), rhs))
            });
            out
        }
    }
    impl MatMul<DenseRow> for Scale {
        type Output = DenseRow;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Scale>,
            rhs: KindRef<'_, RhsE, DenseRow>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Row::<E>::zeros(rhs.nrows());
            let lhs = lhs.inner.0.canonicalize();
            zipped!(out.as_mut().as_2d_mut(), rhs.as_2d()).for_each(|unzipped!(mut out, rhs)| {
                out.write(E::faer_mul(lhs, rhs.read().canonicalize()))
            });
            out
        }
    }
    impl MatMul<Scale> for DenseRow {
        type Output = DenseRow;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, DenseRow>,
            rhs: KindRef<'_, RhsE, Scale>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Row::<E>::zeros(lhs.nrows());
            let rhs = rhs.inner.0.canonicalize();
            zipped!(out.as_mut().as_2d_mut(), lhs.as_2d()).for_each(|unzipped!(mut out, lhs)| {
                out.write(E::faer_mul(lhs.read().canonicalize(), rhs))
            });
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
            let mut out = Mat::<E>::zeros(rhs.nrows(), rhs.ncols());
            let lhs = lhs.inner.0.canonicalize();
            zipped!(out.as_mut(), rhs).for_each(|unzipped!(mut out, rhs)| {
                out.write(E::faer_mul(lhs, rhs.read().canonicalize()))
            });
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
            let mut out = Mat::<E>::zeros(lhs.nrows(), lhs.ncols());
            let rhs = rhs.inner.0.canonicalize();
            zipped!(out.as_mut(), lhs).for_each(|unzipped!(mut out, lhs)| {
                out.write(E::faer_mul(lhs.read().canonicalize(), rhs))
            });
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

    impl MatMul<Diag> for DenseRow {
        type Output = DenseRow;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, DenseRow>,
            rhs: KindRef<'_, RhsE, Diag>,
        ) -> KindOwn<E, Self::Output> {
            let lhs_ncols = lhs.ncols();
            let rhs_dim = rhs.inner.inner.nrows();
            assert!(lhs_ncols == rhs_dim);

            Row::from_fn(lhs_ncols, |j| unsafe {
                E::faer_mul(
                    lhs.read_unchecked(j).canonicalize(),
                    rhs.inner.inner.read_unchecked(j).canonicalize(),
                )
            })
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
                    rhs.inner.inner.read_unchecked(j).canonicalize(),
                )
            })
        }
    }

    impl MatMul<DenseCol> for Diag {
        type Output = DenseCol;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Diag>,
            rhs: KindRef<'_, RhsE, DenseCol>,
        ) -> KindOwn<E, Self::Output> {
            let lhs_dim = lhs.inner.inner.nrows();
            let rhs_nrows = rhs.nrows();
            assert!(lhs_dim == rhs_nrows);

            Col::from_fn(rhs.nrows(), |i| unsafe {
                E::faer_mul(
                    lhs.inner.inner.read_unchecked(i).canonicalize(),
                    rhs.read_unchecked(i).canonicalize(),
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
                    lhs.inner.inner.read_unchecked(i).canonicalize(),
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
                inner: inner::DiagOwn {
                    inner: Col::from_fn(lhs_dim, |i| unsafe {
                        E::faer_mul(
                            lhs.inner.inner.read_unchecked(i).canonicalize(),
                            rhs.inner.inner.read_unchecked(i).canonicalize(),
                        )
                    }),
                },
            }
        }
    }

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

    impl MatMul<DenseCol> for Dense {
        type Output = DenseCol;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Dense>,
            rhs: KindRef<'_, RhsE, DenseCol>,
        ) -> KindOwn<E, Self::Output> {
            assert!(lhs.ncols() == rhs.nrows());
            let mut out = Col::zeros(lhs.nrows());
            mul::matmul(
                out.as_mut().as_2d_mut(),
                lhs,
                rhs.as_2d(),
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }
    impl MatMul<Dense> for DenseRow {
        type Output = DenseRow;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, DenseRow>,
            rhs: KindRef<'_, RhsE, Dense>,
        ) -> KindOwn<E, Self::Output> {
            assert!(lhs.ncols() == rhs.nrows());
            let mut out = Row::zeros(lhs.nrows());
            mul::matmul(
                out.as_mut().as_2d_mut(),
                lhs.as_2d(),
                rhs,
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }

    impl MatMul<DenseCol> for DenseRow {
        type Output = Scalar;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, DenseRow>,
            rhs: KindRef<'_, RhsE, DenseCol>,
        ) -> KindOwn<E, Self::Output> {
            assert!(lhs.ncols() == rhs.nrows());
            let (lhs, conj_lhs) = lhs.canonicalize();
            let (rhs, conj_rhs) = rhs.canonicalize();

            crate::mul::inner_prod::inner_prod_with_conj(
                lhs.transpose().as_2d(),
                conj_lhs,
                rhs.as_2d(),
                conj_rhs,
            )
        }
    }

    impl MatMul<DenseRow> for DenseCol {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, DenseCol>,
            rhs: KindRef<'_, RhsE, DenseRow>,
        ) -> KindOwn<E, Self::Output> {
            assert!(lhs.ncols() == rhs.nrows());
            let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
            mul::matmul(
                out.as_mut(),
                lhs.as_2d(),
                rhs.as_2d(),
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<I: Index> MatMul<SparseColMat<I>> for Dense {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Self>,
            rhs: KindRef<'_, RhsE, SparseColMat<I>>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
            sparse::mul::dense_sparse_matmul(
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
    impl<I: Index> MatMul<SparseColMat<I>> for DenseRow {
        type Output = DenseRow;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Self>,
            rhs: KindRef<'_, RhsE, SparseColMat<I>>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Row::zeros(rhs.ncols());
            sparse::mul::dense_sparse_matmul(
                out.as_mut().as_2d_mut(),
                lhs.as_2d(),
                rhs,
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }
    impl<I: Index> MatMul<SparseRowMat<I>> for Dense {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Self>,
            rhs: KindRef<'_, RhsE, SparseRowMat<I>>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
            sparse::mul::sparse_dense_matmul(
                out.as_mut().transpose_mut(),
                rhs.transpose(),
                lhs.transpose(),
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }
    impl<I: Index> MatMul<SparseRowMat<I>> for DenseRow {
        type Output = DenseRow;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Self>,
            rhs: KindRef<'_, RhsE, SparseRowMat<I>>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Row::zeros(rhs.ncols());
            sparse::mul::sparse_dense_matmul(
                out.as_mut().transpose_mut().as_2d_mut(),
                rhs.transpose(),
                lhs.transpose().as_2d(),
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }

    impl<I: Index> MatMul<Dense> for SparseColMat<I> {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Self>,
            rhs: KindRef<'_, RhsE, Dense>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
            sparse::mul::sparse_dense_matmul(
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
    impl<I: Index> MatMul<DenseCol> for SparseColMat<I> {
        type Output = DenseCol;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Self>,
            rhs: KindRef<'_, RhsE, DenseCol>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Col::zeros(lhs.nrows());
            sparse::mul::sparse_dense_matmul(
                out.as_mut().as_2d_mut(),
                lhs,
                rhs.as_2d(),
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }
    impl<I: Index> MatMul<Dense> for SparseRowMat<I> {
        type Output = Dense;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Self>,
            rhs: KindRef<'_, RhsE, Dense>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Mat::zeros(lhs.nrows(), rhs.ncols());
            sparse::mul::dense_sparse_matmul(
                out.as_mut().transpose_mut(),
                rhs.transpose(),
                lhs.transpose(),
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }
    impl<I: Index> MatMul<DenseCol> for SparseRowMat<I> {
        type Output = DenseCol;

        #[track_caller]
        fn mat_mul<
            E: ComplexField,
            LhsE: Conjugate<Canonical = E>,
            RhsE: Conjugate<Canonical = E>,
        >(
            lhs: KindRef<'_, LhsE, Self>,
            rhs: KindRef<'_, RhsE, DenseCol>,
        ) -> KindOwn<E, Self::Output> {
            let mut out = Col::zeros(lhs.nrows());
            sparse::mul::dense_sparse_matmul(
                out.as_mut().transpose_mut().as_2d_mut(),
                rhs.transpose().as_2d(),
                lhs.transpose(),
                None,
                E::faer_one(),
                get_global_parallelism(),
            );
            out
        }
    }
}

/// Matrix multiplication.
pub trait MatMulAssign<Rhs: MatrixKind>: MatrixKind {
    /// Computes `lhs * rhs` and assigns it to `lhs`.
    fn mat_mul_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    );
}
/// Matrix addition.
pub trait MatAddAssign<Rhs: MatrixKind>: MatrixKind {
    /// Computes `lhs + rhs` and assigns it to `lhs`.
    fn mat_add_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    );
}
/// Matrix subtraction.
pub trait MatSubAssign<Rhs: MatrixKind>: MatrixKind {
    /// Computes `lhs - rhs` and assigns it to `lhs`.
    fn mat_sub_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    );
}

/// Matrix equality comparison.
pub trait MatEq<Rhs: MatrixKind>: MatrixKind {
    /// Computes `lhs == rhs`.
    fn mat_eq<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    ) -> bool;
}

/// Matrix multiplication.
pub trait MatMul<Rhs: MatrixKind>: MatrixKind {
    /// Result matrix type.
    type Output: MatrixKind;

    /// Returns `lhs * rhs`.
    fn mat_mul<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    ) -> KindOwn<E, Self::Output>;
}
/// Matrix addition.
pub trait MatAdd<Rhs: MatrixKind>: MatrixKind {
    /// Result matrix type.
    type Output: MatrixKind;

    /// Returns `lhs + rhs`.
    fn mat_add<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    ) -> KindOwn<E, Self::Output>;
}
/// Matrix subtraction.
pub trait MatSub<Rhs: MatrixKind>: MatrixKind {
    /// Result matrix type.
    type Output: MatrixKind;

    /// Returns `lhs - rhs`.
    fn mat_sub<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Rhs>,
    ) -> KindOwn<E, Self::Output>;
}
/// Matrix negation.
pub trait MatNeg: MatrixKind {
    /// Result matrix type.
    type Output: MatrixKind;

    /// Returns `-mat`.
    fn mat_neg<E: Conjugate>(mat: KindRef<'_, E, Self>) -> KindOwn<E::Canonical, Self::Output>
    where
        E::Canonical: ComplexField;
}

impl<I: Index> MatEq<Perm<I>> for Perm<I> {
    #[track_caller]
    fn mat_eq<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> bool {
        lhs.inner.forward == rhs.inner.forward
    }
}

impl MatEq<DenseCol> for DenseCol {
    #[track_caller]
    fn mat_eq<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> bool {
        lhs.as_2d() == rhs.as_2d()
    }
}
impl MatEq<DenseRow> for DenseRow {
    #[track_caller]
    fn mat_eq<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> bool {
        lhs.as_2d() == rhs.as_2d()
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

impl<I: Index> MatEq<SparseColMat<I>> for SparseColMat<I> {
    fn mat_eq<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> bool {
        if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
            return false;
        }

        let n = lhs.ncols();
        let mut equal = true;
        for j in 0..n {
            equal &= lhs.row_indices_of_col_raw(j) == rhs.row_indices_of_col_raw(j);
            let lhs_val = SliceGroup::<'_, LhsE>::new(lhs.values_of_col(j));
            let rhs_val = SliceGroup::<'_, RhsE>::new(rhs.values_of_col(j));
            equal &= lhs_val
                .into_ref_iter()
                .map(|r| r.read().canonicalize())
                .eq(rhs_val.into_ref_iter().map(|r| r.read().canonicalize()))
        }

        equal
    }
}

impl<I: Index> MatEq<SparseRowMat<I>> for SparseRowMat<I> {
    fn mat_eq<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> bool {
        lhs.transpose() == rhs.transpose()
    }
}

impl MatAdd<DenseCol> for DenseCol {
    type Output = DenseCol;

    #[track_caller]
    fn mat_add<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> KindOwn<E, Self::Output> {
        assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        let mut out = Col::<E>::zeros(lhs.nrows());
        zipped!(out.as_mut().as_2d_mut(), lhs.as_2d(), rhs.as_2d()).for_each(
            |unzipped!(mut out, lhs, rhs)| {
                out.write(E::faer_add(
                    lhs.read().canonicalize(),
                    rhs.read().canonicalize(),
                ))
            },
        );
        out
    }
}
impl MatSub<DenseCol> for DenseCol {
    type Output = DenseCol;

    #[track_caller]
    fn mat_sub<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> KindOwn<E, Self::Output> {
        assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        let mut out = Col::<E>::zeros(lhs.nrows());
        zipped!(out.as_mut().as_2d_mut(), lhs.as_2d(), rhs.as_2d()).for_each(
            |unzipped!(mut out, lhs, rhs)| {
                out.write(E::faer_sub(
                    lhs.read().canonicalize(),
                    rhs.read().canonicalize(),
                ))
            },
        );
        out
    }
}
impl MatAddAssign<DenseCol> for DenseCol {
    #[track_caller]
    fn mat_add_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, DenseCol>,
        rhs: KindRef<'_, RhsE, DenseCol>,
    ) {
        zipped!(lhs.as_2d_mut(), rhs.as_2d()).for_each(|unzipped!(mut lhs, rhs)| {
            lhs.write(lhs.read().faer_add(rhs.read().canonicalize()))
        });
    }
}
impl MatSubAssign<DenseCol> for DenseCol {
    #[track_caller]
    fn mat_sub_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, DenseCol>,
        rhs: KindRef<'_, RhsE, DenseCol>,
    ) {
        zipped!(lhs.as_2d_mut(), rhs.as_2d()).for_each(|unzipped!(mut lhs, rhs)| {
            lhs.write(lhs.read().faer_sub(rhs.read().canonicalize()))
        });
    }
}

impl MatNeg for DenseCol {
    type Output = DenseCol;

    fn mat_neg<E: Conjugate>(mat: KindRef<'_, E, Self>) -> KindOwn<E::Canonical, Self::Output>
    where
        E::Canonical: ComplexField,
    {
        let mut out = Col::<E::Canonical>::zeros(mat.nrows());
        zipped!(out.as_mut().as_2d_mut(), mat.as_2d())
            .for_each(|unzipped!(mut out, src)| out.write(src.read().canonicalize().faer_neg()));
        out
    }
}

impl MatAdd<DenseRow> for DenseRow {
    type Output = DenseRow;

    #[track_caller]
    fn mat_add<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> KindOwn<E, Self::Output> {
        assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        let mut out = Row::<E>::zeros(lhs.nrows());
        zipped!(out.as_mut().as_2d_mut(), lhs.as_2d(), rhs.as_2d()).for_each(
            |unzipped!(mut out, lhs, rhs)| {
                out.write(E::faer_add(
                    lhs.read().canonicalize(),
                    rhs.read().canonicalize(),
                ))
            },
        );
        out
    }
}
impl MatSub<DenseRow> for DenseRow {
    type Output = DenseRow;

    #[track_caller]
    fn mat_sub<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> KindOwn<E, Self::Output> {
        assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        let mut out = Row::<E>::zeros(lhs.nrows());
        zipped!(out.as_mut().as_2d_mut(), lhs.as_2d(), rhs.as_2d()).for_each(
            |unzipped!(mut out, lhs, rhs)| {
                out.write(E::faer_sub(
                    lhs.read().canonicalize(),
                    rhs.read().canonicalize(),
                ))
            },
        );
        out
    }
}
impl MatAddAssign<DenseRow> for DenseRow {
    #[track_caller]
    fn mat_add_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, DenseRow>,
        rhs: KindRef<'_, RhsE, DenseRow>,
    ) {
        zipped!(lhs.as_2d_mut(), rhs.as_2d()).for_each(|unzipped!(mut lhs, rhs)| {
            lhs.write(lhs.read().faer_add(rhs.read().canonicalize()))
        });
    }
}
impl MatSubAssign<DenseRow> for DenseRow {
    #[track_caller]
    fn mat_sub_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, DenseRow>,
        rhs: KindRef<'_, RhsE, DenseRow>,
    ) {
        zipped!(lhs.as_2d_mut(), rhs.as_2d()).for_each(|unzipped!(mut lhs, rhs)| {
            lhs.write(lhs.read().faer_sub(rhs.read().canonicalize()))
        });
    }
}

impl MatNeg for DenseRow {
    type Output = DenseRow;

    fn mat_neg<E: Conjugate>(mat: KindRef<'_, E, Self>) -> KindOwn<E::Canonical, Self::Output>
    where
        E::Canonical: ComplexField,
    {
        let mut out = Row::<E::Canonical>::zeros(mat.nrows());
        zipped!(out.as_mut().as_2d_mut(), mat.as_2d())
            .for_each(|unzipped!(mut out, src)| out.write(src.read().canonicalize().faer_neg()));
        out
    }
}

impl MatAdd<Dense> for Dense {
    type Output = Dense;

    #[track_caller]
    fn mat_add<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> KindOwn<E, Self::Output> {
        assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        let mut out = Mat::<E>::zeros(lhs.nrows(), rhs.ncols());
        zipped!(out.as_mut(), lhs, rhs).for_each(|unzipped!(mut out, lhs, rhs)| {
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
        assert!(all(lhs.nrows() == rhs.nrows(), lhs.ncols() == rhs.ncols()));
        let mut out = Mat::<E>::zeros(lhs.nrows(), rhs.ncols());
        zipped!(out.as_mut(), lhs, rhs).for_each(|unzipped!(mut out, lhs, rhs)| {
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
        zipped!(lhs, rhs).for_each(|unzipped!(mut lhs, rhs)| {
            lhs.write(lhs.read().faer_add(rhs.read().canonicalize()))
        });
    }
}
impl MatSubAssign<Dense> for Dense {
    #[track_caller]
    fn mat_sub_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, Dense>,
        rhs: KindRef<'_, RhsE, Dense>,
    ) {
        zipped!(lhs, rhs).for_each(|unzipped!(mut lhs, rhs)| {
            lhs.write(lhs.read().faer_sub(rhs.read().canonicalize()))
        });
    }
}

impl MatNeg for Dense {
    type Output = Dense;

    fn mat_neg<E: Conjugate>(mat: KindRef<'_, E, Self>) -> KindOwn<E::Canonical, Self::Output>
    where
        E::Canonical: ComplexField,
    {
        let mut out = Mat::<E::Canonical>::zeros(mat.nrows(), mat.ncols());
        zipped!(out.as_mut(), mat)
            .for_each(|unzipped!(mut out, src)| out.write(src.read().canonicalize().faer_neg()));
        out
    }
}

impl<I: Index> MatAdd<SparseColMat<I>> for SparseColMat<I> {
    type Output = SparseColMat<I>;

    #[track_caller]
    fn mat_add<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> KindOwn<E, Self::Output> {
        crate::sparse::ops::add(lhs, rhs).unwrap()
    }
}
impl<I: Index> MatAdd<SparseRowMat<I>> for SparseRowMat<I> {
    type Output = SparseColMat<I>;

    #[track_caller]
    fn mat_add<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> KindOwn<E, Self::Output> {
        crate::sparse::ops::add(lhs.transpose(), rhs.transpose())
            .unwrap()
            .into_transpose()
            .to_col_major()
            .unwrap()
    }
}
impl<I: Index> MatAdd<SparseRowMat<I>> for SparseColMat<I> {
    type Output = SparseColMat<I>;

    #[track_caller]
    fn mat_add<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, SparseRowMat<I>>,
    ) -> KindOwn<E, Self::Output> {
        crate::sparse::ops::add(lhs, rhs.to_col_major().unwrap().as_ref()).unwrap()
    }
}
impl<I: Index> MatAdd<SparseColMat<I>> for SparseRowMat<I> {
    type Output = SparseColMat<I>;

    #[track_caller]
    fn mat_add<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, SparseColMat<I>>,
    ) -> KindOwn<E, Self::Output> {
        crate::sparse::ops::add(lhs.to_col_major().unwrap().as_ref(), rhs).unwrap()
    }
}

impl<I: Index> MatSub<SparseColMat<I>> for SparseColMat<I> {
    type Output = SparseColMat<I>;

    #[track_caller]
    fn mat_sub<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> KindOwn<E, Self::Output> {
        crate::sparse::ops::sub(lhs, rhs).unwrap()
    }
}
impl<I: Index> MatSub<SparseRowMat<I>> for SparseRowMat<I> {
    type Output = SparseColMat<I>;

    #[track_caller]
    fn mat_sub<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, Self>,
    ) -> KindOwn<E, Self::Output> {
        crate::sparse::ops::sub(lhs.transpose(), rhs.transpose())
            .unwrap()
            .into_transpose()
            .to_col_major()
            .unwrap()
    }
}
impl<I: Index> MatSub<SparseRowMat<I>> for SparseColMat<I> {
    type Output = SparseColMat<I>;

    #[track_caller]
    fn mat_sub<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, SparseRowMat<I>>,
    ) -> KindOwn<E, Self::Output> {
        crate::sparse::ops::sub(lhs, rhs.to_col_major().unwrap().as_ref()).unwrap()
    }
}
impl<I: Index> MatSub<SparseColMat<I>> for SparseRowMat<I> {
    type Output = SparseColMat<I>;

    #[track_caller]
    fn mat_sub<E: ComplexField, LhsE: Conjugate<Canonical = E>, RhsE: Conjugate<Canonical = E>>(
        lhs: KindRef<'_, LhsE, Self>,
        rhs: KindRef<'_, RhsE, SparseColMat<I>>,
    ) -> KindOwn<E, Self::Output> {
        crate::sparse::ops::sub(lhs.to_col_major().unwrap().as_ref(), rhs).unwrap()
    }
}

impl<I: Index> MatNeg for SparseColMat<I> {
    type Output = SparseColMat<I>;

    fn mat_neg<E: Conjugate>(mat: KindRef<'_, E, Self>) -> KindOwn<E::Canonical, Self::Output>
    where
        E::Canonical: ComplexField,
    {
        let mut out = mat.to_owned().unwrap();
        crate::group_helpers::SliceGroupMut::<E::Canonical>::new(out.as_mut().values_mut())
            .into_mut_iter()
            .for_each(|mut x| x.write(x.read().faer_neg()));
        out
    }
}

impl<I: Index> MatNeg for SparseRowMat<I> {
    type Output = SparseColMat<I>;

    fn mat_neg<E: Conjugate>(mat: KindRef<'_, E, Self>) -> KindOwn<E::Canonical, Self::Output>
    where
        E::Canonical: ComplexField,
    {
        let mut out = mat.to_col_major().unwrap();
        crate::group_helpers::SliceGroupMut::<E::Canonical>::new(out.as_mut().values_mut())
            .into_mut_iter()
            .for_each(|mut x| x.write(x.read().faer_neg()));
        out
    }
}

impl<I: Index> MatAddAssign<SparseColMat<I>> for SparseColMat<I> {
    #[track_caller]
    fn mat_add_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, SparseColMat<I>>,
        rhs: KindRef<'_, RhsE, SparseColMat<I>>,
    ) {
        crate::sparse::ops::add_assign(lhs, rhs);
    }
}
impl<I: Index> MatSubAssign<SparseColMat<I>> for SparseColMat<I> {
    #[track_caller]
    fn mat_sub_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, SparseColMat<I>>,
        rhs: KindRef<'_, RhsE, SparseColMat<I>>,
    ) {
        crate::sparse::ops::sub_assign(lhs, rhs);
    }
}

impl<I: Index> MatAddAssign<SparseRowMat<I>> for SparseRowMat<I> {
    #[track_caller]
    fn mat_add_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, SparseRowMat<I>>,
        rhs: KindRef<'_, RhsE, SparseRowMat<I>>,
    ) {
        crate::sparse::ops::add_assign(lhs.transpose_mut(), rhs.transpose());
    }
}
impl<I: Index> MatSubAssign<SparseRowMat<I>> for SparseRowMat<I> {
    #[track_caller]
    fn mat_sub_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, SparseRowMat<I>>,
        rhs: KindRef<'_, RhsE, SparseRowMat<I>>,
    ) {
        crate::sparse::ops::sub_assign(lhs.transpose_mut(), rhs.transpose());
    }
}

impl<I: Index> MatAddAssign<SparseColMat<I>> for SparseRowMat<I> {
    #[track_caller]
    fn mat_add_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, SparseRowMat<I>>,
        rhs: KindRef<'_, RhsE, SparseColMat<I>>,
    ) {
        crate::sparse::ops::add_assign(
            lhs.transpose_mut(),
            rhs.transpose().to_col_major().unwrap().as_ref(),
        );
    }
}
impl<I: Index> MatSubAssign<SparseColMat<I>> for SparseRowMat<I> {
    #[track_caller]
    fn mat_sub_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, SparseRowMat<I>>,
        rhs: KindRef<'_, RhsE, SparseColMat<I>>,
    ) {
        crate::sparse::ops::sub_assign(
            lhs.transpose_mut(),
            rhs.transpose().to_col_major().unwrap().as_ref(),
        );
    }
}

impl<I: Index> MatAddAssign<SparseRowMat<I>> for SparseColMat<I> {
    #[track_caller]
    fn mat_add_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, SparseColMat<I>>,
        rhs: KindRef<'_, RhsE, SparseRowMat<I>>,
    ) {
        crate::sparse::ops::add_assign(lhs, rhs.to_col_major().unwrap().as_ref());
    }
}
impl<I: Index> MatSubAssign<SparseRowMat<I>> for SparseColMat<I> {
    #[track_caller]
    fn mat_sub_assign<E: ComplexField, RhsE: Conjugate<Canonical = E>>(
        lhs: KindMut<'_, E, SparseColMat<I>>,
        rhs: KindRef<'_, RhsE, SparseRowMat<I>>,
    ) {
        crate::sparse::ops::sub_assign(lhs, rhs.to_col_major().unwrap().as_ref());
    }
}

/// Returns a scaling factor with the given value.
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

        #[track_caller]
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

        #[track_caller]
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

        #[track_caller]
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

        #[track_caller]
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

        #[track_caller]
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

        #[track_caller]
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

        #[track_caller]
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

        #[track_caller]
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

        #[track_caller]
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

        #[track_caller]
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

        #[track_caller]
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

        #[track_caller]
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
        #[track_caller]
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
        #[track_caller]
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
        #[track_caller]
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
        #[track_caller]
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
        #[track_caller]
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
        #[track_caller]
        fn sub_assign(&mut self, rhs: Matrix<Rhs>) {
            *self -= &rhs;
        }
    }
};

#[cfg(test)]
#[allow(non_snake_case)]
mod test {
    use crate::{
        assert, mat,
        permutation::{Permutation, PermutationRef},
        Col, Mat, Row,
    };
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

    #[test]
    fn test_perm_mul() {
        let A = Mat::from_fn(6, 5, |i, j| (j + 5 * i) as f64);
        let pl = Permutation::<usize, f64>::new_checked(
            Box::new([5, 1, 4, 0, 2, 3]),
            Box::new([3, 1, 4, 5, 2, 0]),
        );
        let pr = Permutation::<usize, f64>::new_checked(
            Box::new([1, 4, 0, 2, 3]),
            Box::new([2, 0, 3, 4, 1]),
        );

        let perm_left = mat![
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ];
        let perm_right = mat![
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ];

        assert!(
            &pl * pl.as_ref().inverse()
                == PermutationRef::<'_, usize, f64>::new_checked(
                    &[0, 1, 2, 3, 4, 5],
                    &[0, 1, 2, 3, 4, 5],
                )
        );
        assert!(&perm_left * &A == &pl * &A);
        assert!(&A * &perm_right == &A * &pr);
    }

    #[test]
    fn test_matmul_col_row() {
        let A = Col::from_fn(6, |i| i as f64);
        let B = Row::from_fn(6, |j| (5 * j + 1) as f64);

        // outer product
        assert_eq!(&A * &B, A.as_ref().as_2d() * B.as_ref().as_2d());
        // inner product
        assert_eq!(
            &B * &A,
            (B.as_ref().as_2d() * A.as_ref().as_2d()).read(0, 0),
        );
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
