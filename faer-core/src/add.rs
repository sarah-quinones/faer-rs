//! addition and subtraction of matrices

use crate::{ComplexField, Mat, MatRef};
use core::ops::{Add, Sub};

// add two matrices together
impl<'a, T> Add<MatRef<'_, T>> for MatRef<'a, T>
where
    T: ComplexField,
{
    type Output = Mat<T>;
    /// create a new matrix corresponding to the addition of `rhs` to `self`.
    /// # Panics
    /// Panics if the matrix dimensions do not match.
    fn add(self, rhs: MatRef<'_, T>) -> Self::Output {
        assert_eq!(
            (self.nrows(), self.ncols()),
            (rhs.nrows(), rhs.ncols()),
            "Matrix dimensions must match"
        );
        Self::Output::with_dims(self.nrows(), self.ncols(), |i, j| {
            self.read(i, j).add(&rhs.read(i, j))
        })
    }
}

impl<'a, T> Sub<MatRef<'_, T>> for MatRef<'a, T>
where
    T: ComplexField,
{
    type Output = Mat<T>;
    /// create a new matrix corresponding to the subtraction of `rhs` from `self`.
    /// # Panics
    /// Panics if the matrix dimensions do not match.
    fn sub(self, rhs: MatRef<'_, T>) -> Self::Output {
        assert_eq!(
            (self.nrows(), self.ncols()),
            (rhs.nrows(), rhs.ncols()),
            "Matrix dimensions must match"
        );
        Self::Output::with_dims(self.nrows(), self.ncols(), |i, j| {
            self.read(i, j).sub(&rhs.read(i, j))
        })
    }
}

// implement the add trait for cases where one of the operands is
// an owned matrix by deferring to the case where both are references
// @todo: this will allocate even if one of the operands could be reuse
// and in future we should consider adding an efficient add- and sub-assign
// implementations that are used instead as backends
macro_rules! impl_binary_op {
    ($Op:ty) => {
        paste::paste! {
            impl<T> $Op<MatRef<'_,T>> for Mat<T>
            where
                T: ComplexField,
            {
                type Output = Mat<T>;
                fn [<$Op:lower>](self, rhs: MatRef<'_,T>) -> Self::Output {
                    self.as_ref().[<$Op:lower>] (rhs)
                }
            }

            impl<T> $Op<Mat<T>> for MatRef<'_,T>
            where
                T: ComplexField,
            {
                type Output = Mat<T>;
                fn [<$Op:lower>](self, rhs: Mat<T>) -> Self::Output {
                    self.[<$Op:lower>] (rhs.as_ref())
                }
            }

            impl<T> $Op<Mat<T>> for Mat<T>
            where
                T: ComplexField,
            {
                type Output = Mat<T>;
                fn [<$Op:lower>](self, rhs: Mat<T>) -> Self::Output {
                    self.as_ref().[<$Op:lower>] (rhs.as_ref())
                }
            }

            impl<T> $Op<&Mat<T>> for MatRef<'_,T>
            where
                T: ComplexField,
            {
                type Output = Mat<T>;
                fn [<$Op:lower>](self, rhs: &Mat<T>) -> Self::Output {
                    self.[<$Op:lower>] (rhs.as_ref())
                }
            }

            impl<T> $Op<MatRef<'_,T>> for &Mat<T>
            where
                T: ComplexField,
            {
                type Output = Mat<T>;
                fn [<$Op:lower>](self, rhs: MatRef<'_,T>) -> Self::Output {
                    self.as_ref().[<$Op:lower>] (rhs)
                }
            }

            impl<T> $Op<&Mat<T>> for &'_ Mat<T>
            where
                T: ComplexField,
            {
                type Output = Mat<T>;
                fn [<$Op:lower>](self, rhs: &Mat<T>) -> Self::Output {
                    self.as_ref().[<$Op:lower>] (rhs.as_ref())
                }
            }

            impl<T> $Op<Mat<T>> for &Mat<T>
            where
                T: ComplexField,
            {
                type Output = Mat<T>;
                fn [<$Op:lower>](self, rhs: Mat<T>) -> Self::Output {
                    self.as_ref().[<$Op:lower>] (rhs.as_ref())
                }
            }

            impl<T> $Op<&Mat<T>> for Mat<T>
            where
                T: ComplexField,
            {
                type Output = Mat<T>;
                fn [<$Op:lower>](self, rhs: &Mat<T>) -> Self::Output {
                    self.as_ref().[<$Op:lower>] (rhs.as_ref())
                }
            }
        }
    };
}

impl_binary_op!(Add);
impl_binary_op!(Sub);

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
    fn test_add() {
        let (A, B) = matrices();

        let expected = mat![[-5.1, 5.0], [3.0, 2.0], [8.4, -13.5],];

        assert_matrix_approx_eq(A.as_ref() + B.as_ref(), &expected);
        assert_matrix_approx_eq(&A + &B, &expected);
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
        assert_matrix_approx_eq(&A - B.as_ref(), &expected);
        assert_matrix_approx_eq(A.as_ref() - B.clone(), &expected);
        assert_matrix_approx_eq(&A - B.clone(), &expected);
        assert_matrix_approx_eq(A.clone() - B.as_ref(), &expected);
        assert_matrix_approx_eq(A.clone() - &B, &expected);
        assert_matrix_approx_eq(&A - &B, &expected);
        assert_matrix_approx_eq(A - B, &expected);
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
