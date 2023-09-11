//! addition and subtraction of matrices

use crate::{zipped, ComplexField, Conjugate, Mat, MatMut, MatRef};
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use reborrow::*;

impl<LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsE>> AddAssign<MatRef<'_, RhsE>>
    for MatMut<'_, LhsE>
{
    fn add_assign(&mut self, rhs: MatRef<'_, RhsE>) {
        assert_eq!((self.nrows(), self.ncols()), (rhs.nrows(), rhs.ncols()));
        zipped!(self.rb_mut(), rhs).for_each(|mut lhs, rhs| {
            lhs.write(lhs.read().add(&rhs.read().canonicalize()));
        });
    }
}

impl<LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsE>> SubAssign<MatRef<'_, RhsE>>
    for MatMut<'_, LhsE>
{
    fn sub_assign(&mut self, rhs: MatRef<'_, RhsE>) {
        assert_eq!((self.nrows(), self.ncols()), (rhs.nrows(), rhs.ncols()));
        zipped!(self.rb_mut(), rhs).for_each(|mut lhs, rhs| {
            lhs.write(lhs.read().sub(&rhs.read().canonicalize()));
        });
    }
}

impl<LhsE: Conjugate, RhsE: Conjugate<Canonical = LhsE::Canonical>> Add<MatRef<'_, RhsE>>
    for MatRef<'_, LhsE>
where
    LhsE::Canonical: ComplexField,
{
    type Output = Mat<LhsE::Canonical>;

    fn add(self, rhs: MatRef<'_, RhsE>) -> Self::Output {
        assert_eq!((self.nrows(), self.ncols()), (rhs.nrows(), rhs.ncols()));
        // SAFETY: we checked that the lhs and rhs dimensions are the same, so unchecked access is
        // fine
        unsafe {
            Self::Output::with_dims(self.nrows(), self.ncols(), |i, j| {
                self.read_unchecked(i, j)
                    .canonicalize()
                    .add(&rhs.read_unchecked(i, j).canonicalize())
            })
        }
    }
}

impl<LhsE: Conjugate, RhsE: Conjugate<Canonical = LhsE::Canonical>> Sub<MatRef<'_, RhsE>>
    for MatRef<'_, LhsE>
where
    LhsE::Canonical: ComplexField,
{
    type Output = Mat<LhsE::Canonical>;

    fn sub(self, rhs: MatRef<'_, RhsE>) -> Self::Output {
        assert_eq!((self.nrows(), self.ncols()), (rhs.nrows(), rhs.ncols()));
        // SAFETY: we checked that the lhs and rhs dimensions are the same, so unchecked access is
        // fine
        unsafe {
            Self::Output::with_dims(self.nrows(), self.ncols(), |i, j| {
                self.read_unchecked(i, j)
                    .canonicalize()
                    .sub(&rhs.read_unchecked(i, j).canonicalize())
            })
        }
    }
}

impl<E: Conjugate> Neg for MatRef<'_, E>
where
    E::Canonical: ComplexField,
{
    type Output = Mat<E::Canonical>;

    fn neg(self) -> Self::Output {
        // SAFETY: destination and input dimensions are the same
        unsafe {
            Self::Output::with_dims(self.nrows(), self.ncols(), |i, j| {
                self.read_unchecked(i, j).canonicalize().neg()
            })
        }
    }
}

// implement unary traits for cases where the operand is
// an owned matrix by deferring to the case where it's a reference
// @todo: this will allocate even if the operand could be reused
// and in the future we should consider adding an efficient Neg
// implementation that is used instead as a backend
macro_rules! impl_unary_op_single {
    ($trait_name: ident, $op: ident, $operand: ty) => {
        impl<E: Conjugate> $trait_name for $operand
        where
            E::Canonical: ComplexField,
        {
            type Output = Mat<E::Canonical>;
            fn $op(self) -> Self::Output {
                self.as_ref().$op()
            }
        }
    };
}

// implement binary traits for cases where one of the operands is
// an owned matrix by deferring to the case where both are references
// @todo: this will allocate even if one of the operands could be reused
// and in the future we should consider adding an efficient add- and sub-assign
// implementations that are used instead as backends
macro_rules! impl_binary_op_single {
    ($trait_name: ident, $op: ident, $lhs: ty, $rhs: ty) => {
        impl<LhsE: Conjugate, RhsE: Conjugate<Canonical = LhsE::Canonical>> $trait_name<$rhs>
            for $lhs
        where
            LhsE::Canonical: ComplexField,
        {
            type Output = Mat<LhsE::Canonical>;
            fn $op(self, rhs: $rhs) -> Self::Output {
                self.as_ref().$op(rhs.as_ref())
            }
        }
    };
}

macro_rules! impl_assign_op_single {
    ($trait_name: ident, $op: ident, $lhs: ty, $rhs: ty) => {
        impl<LhsE: ComplexField, RhsE: Conjugate<Canonical = LhsE>> $trait_name<$rhs> for $lhs {
            fn $op(&mut self, rhs: $rhs) {
                self.as_mut().$op(rhs.as_ref())
            }
        }
    };
}

macro_rules! impl_eq_single {
    ($lhs: ty, $rhs: ty) => {
        impl<LhsE: Conjugate, RhsE: Conjugate<Canonical = LhsE::Canonical>> PartialEq<$rhs> for $lhs
        where
            LhsE::Canonical: ComplexField,
        {
            fn eq(&self, rhs: &$rhs) -> bool {
                PartialEq::eq(&self.as_ref(), &rhs.as_ref())
            }
        }
    };
}

macro_rules! impl_unary_op {
    ($trait_name: ident, $op: ident) => {
        // possible operands:
        //
        // Mat
        // &Mat
        // MatRef
        // &MatRef
        // MatMut
        // &MatMut

        impl_unary_op_single!($trait_name, $op, Mat<E>);
        impl_unary_op_single!($trait_name, $op, &Mat<E>);
        // impl_unary_op_single!($trait_name, $op, MatRef<'_, E>);
        impl_unary_op_single!($trait_name, $op, &MatRef<'_, E>);
        impl_unary_op_single!($trait_name, $op, MatMut<'_, E>);
        impl_unary_op_single!($trait_name, $op, &MatMut<'_, E>);
    };
}

macro_rules! impl_binary_op {
    ($trait_name: ident, $op: ident) => {
        impl_binary_op_single!($trait_name, $op, Mat<LhsE>, Mat<RhsE>);
        impl_binary_op_single!($trait_name, $op, Mat<LhsE>, &Mat<RhsE>);
        impl_binary_op_single!($trait_name, $op, Mat<LhsE>, MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, Mat<LhsE>, &MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, Mat<LhsE>, MatMut<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, Mat<LhsE>, &MatMut<'_, RhsE>);

        impl_binary_op_single!($trait_name, $op, &Mat<LhsE>, Mat<RhsE>);
        impl_binary_op_single!($trait_name, $op, &Mat<LhsE>, &Mat<RhsE>);
        impl_binary_op_single!($trait_name, $op, &Mat<LhsE>, MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, &Mat<LhsE>, &MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, &Mat<LhsE>, MatMut<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, &Mat<LhsE>, &MatMut<'_, RhsE>);

        impl_binary_op_single!($trait_name, $op, MatRef<'_, LhsE>, Mat<RhsE>);
        impl_binary_op_single!($trait_name, $op, MatRef<'_, LhsE>, &Mat<RhsE>);
        // impl_binary_op_single!($trait_name, $op, MatRef<'_, LhsE>, MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, MatRef<'_, LhsE>, &MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, MatRef<'_, LhsE>, MatMut<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, MatRef<'_, LhsE>, &MatMut<'_, RhsE>);

        impl_binary_op_single!($trait_name, $op, &MatRef<'_, LhsE>, Mat<RhsE>);
        impl_binary_op_single!($trait_name, $op, &MatRef<'_, LhsE>, &Mat<RhsE>);
        impl_binary_op_single!($trait_name, $op, &MatRef<'_, LhsE>, MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, &MatRef<'_, LhsE>, &MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, &MatRef<'_, LhsE>, MatMut<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, &MatRef<'_, LhsE>, &MatMut<'_, RhsE>);

        impl_binary_op_single!($trait_name, $op, MatMut<'_, LhsE>, Mat<RhsE>);
        impl_binary_op_single!($trait_name, $op, MatMut<'_, LhsE>, &Mat<RhsE>);
        impl_binary_op_single!($trait_name, $op, MatMut<'_, LhsE>, MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, MatMut<'_, LhsE>, &MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, MatMut<'_, LhsE>, MatMut<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, MatMut<'_, LhsE>, &MatMut<'_, RhsE>);

        impl_binary_op_single!($trait_name, $op, &MatMut<'_, LhsE>, Mat<RhsE>);
        impl_binary_op_single!($trait_name, $op, &MatMut<'_, LhsE>, &Mat<RhsE>);
        impl_binary_op_single!($trait_name, $op, &MatMut<'_, LhsE>, MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, &MatMut<'_, LhsE>, &MatRef<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, &MatMut<'_, LhsE>, MatMut<'_, RhsE>);
        impl_binary_op_single!($trait_name, $op, &MatMut<'_, LhsE>, &MatMut<'_, RhsE>);
    };
}

macro_rules! impl_assign_op {
    ($trait_name: ident, $op: ident) => {
        impl_assign_op_single!($trait_name, $op, Mat<LhsE>, Mat<RhsE>);
        impl_assign_op_single!($trait_name, $op, Mat<LhsE>, &Mat<RhsE>);
        impl_assign_op_single!($trait_name, $op, Mat<LhsE>, MatRef<'_, RhsE>);
        impl_assign_op_single!($trait_name, $op, Mat<LhsE>, &MatRef<'_, RhsE>);
        impl_assign_op_single!($trait_name, $op, Mat<LhsE>, MatMut<'_, RhsE>);
        impl_assign_op_single!($trait_name, $op, Mat<LhsE>, &MatMut<'_, RhsE>);

        impl_assign_op_single!($trait_name, $op, MatMut<'_, LhsE>, Mat<RhsE>);
        impl_assign_op_single!($trait_name, $op, MatMut<'_, LhsE>, &Mat<RhsE>);
        // impl_assign_op_single!($trait_name, $op, MatMut<'_, LhsE>, MatRef<'_, RhsE>);
        impl_assign_op_single!($trait_name, $op, MatMut<'_, LhsE>, &MatRef<'_, RhsE>);
        impl_assign_op_single!($trait_name, $op, MatMut<'_, LhsE>, MatMut<'_, RhsE>);
        impl_assign_op_single!($trait_name, $op, MatMut<'_, LhsE>, &MatMut<'_, RhsE>);
    };
}

impl_eq_single!(Mat<LhsE>, Mat<RhsE>);
impl_eq_single!(Mat<LhsE>, &Mat<RhsE>);
impl_eq_single!(Mat<LhsE>, MatRef<'_, RhsE>);
impl_eq_single!(Mat<LhsE>, &MatRef<'_, RhsE>);
impl_eq_single!(Mat<LhsE>, MatMut<'_, RhsE>);
impl_eq_single!(Mat<LhsE>, &MatMut<'_, RhsE>);

impl_eq_single!(&Mat<LhsE>, Mat<RhsE>);
// impl_eq_single!(&Mat<LhsE>, &Mat<RhsE>);
impl_eq_single!(&Mat<LhsE>, MatRef<'_, RhsE>);
// impl_eq_single!(&Mat<LhsE>, &MatRef<'_, RhsE>);
impl_eq_single!(&Mat<LhsE>, MatMut<'_, RhsE>);
// impl_eq_single!(&Mat<LhsE>, &MatMut<'_, RhsE>);

impl_eq_single!(MatRef<'_, LhsE>, Mat<RhsE>);
impl_eq_single!(MatRef<'_, LhsE>, &Mat<RhsE>);
// impl_eq_single!(MatRef<'_, LhsE>, MatRef<'_, RhsE>);
impl_eq_single!(MatRef<'_, LhsE>, &MatRef<'_, RhsE>);
impl_eq_single!(MatRef<'_, LhsE>, MatMut<'_, RhsE>);
impl_eq_single!(MatRef<'_, LhsE>, &MatMut<'_, RhsE>);

impl_eq_single!(&MatRef<'_, LhsE>, Mat<RhsE>);
// impl_eq_single!(&MatRef<'_, LhsE>, &Mat<RhsE>);
impl_eq_single!(&MatRef<'_, LhsE>, MatRef<'_, RhsE>);
// impl_eq_single!(&MatRef<'_, LhsE>, &MatRef<'_, RhsE>);
impl_eq_single!(&MatRef<'_, LhsE>, MatMut<'_, RhsE>);
// impl_eq_single!(&MatRef<'_, LhsE>, &MatMut<'_, RhsE>);

impl_eq_single!(MatMut<'_, LhsE>, Mat<RhsE>);
impl_eq_single!(MatMut<'_, LhsE>, &Mat<RhsE>);
impl_eq_single!(MatMut<'_, LhsE>, MatRef<'_, RhsE>);
impl_eq_single!(MatMut<'_, LhsE>, &MatRef<'_, RhsE>);
impl_eq_single!(MatMut<'_, LhsE>, MatMut<'_, RhsE>);
impl_eq_single!(MatMut<'_, LhsE>, &MatMut<'_, RhsE>);

impl_eq_single!(&MatMut<'_, LhsE>, Mat<RhsE>);
// impl_eq_single!(&MatMut<'_, LhsE>, &Mat<RhsE>);
impl_eq_single!(&MatMut<'_, LhsE>, MatRef<'_, RhsE>);
// impl_eq_single!(&MatMut<'_, LhsE>, &MatRef<'_, RhsE>);
impl_eq_single!(&MatMut<'_, LhsE>, MatMut<'_, RhsE>);
// impl_eq_single!(&MatMut<'_, LhsE>, &MatMut<'_, RhsE>);

impl_unary_op!(Neg, neg);

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);

impl_assign_op!(AddAssign, add_assign);
impl_assign_op!(SubAssign, sub_assign);

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
