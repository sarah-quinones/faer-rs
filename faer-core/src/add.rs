//! addition and subtraction of matrices

use crate::{ComplexField, Conjugate, Entity, Mat, MatRef};
use core::ops::{Add, AddAssign, Sub};

// add two matrices together
impl<'a, T> Add<MatRef<'_, T>> for MatRef<'a, T>
where
    T: ComplexField,
    T::Unit: ComplexField,
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
    T::Unit: ComplexField,
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
