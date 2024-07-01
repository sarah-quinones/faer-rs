use crate::{
    diag::{Diag, DiagMut, DiagRef},
    sparse::{
        SparseColMat, SparseColMatMut, SparseColMatRef, SparseRowMat, SparseRowMatMut,
        SparseRowMatRef,
    },
    Col, ColMut, ColRef, Mat, MatMut, MatRef, Row, RowMut, RowRef,
};
use core::ops::{Div, Mul};

trait MulInvertible: Sized {
    fn invert(self) -> Self;
}

impl MulInvertible for f32 {
    fn invert(self) -> f32 {
        1.0 / self
    }
}

impl MulInvertible for f64 {
    fn invert(self) -> f64 {
        1.0 / self
    }
}

macro_rules! impl_scalar_ops {
    ($scalar:ty, $vector:ty, $res:ty) => {
        impl Mul<$scalar> for $vector {
            type Output = $res;

            #[inline]
            fn mul(self, scalar: $scalar) -> $res {
                crate::scale(scalar) * self
            }
        }

        impl Mul<$vector> for $scalar {
            type Output = $res;

            #[inline]
            fn mul(self, vector: $vector) -> $res {
                crate::scale(self) * vector
            }
        }

        impl Div<$scalar> for $vector {
            type Output = $res;

            #[inline]
            fn div(self, scalar: $scalar) -> $res {
                crate::scale(scalar.invert()) * self
            }
        }
    };
}

// Col
impl_scalar_ops!(f32, Col<f32>, Col<f32>);
impl_scalar_ops!(f64, Col<f64>, Col<f64>);

impl_scalar_ops!(f32, ColMut<'_, f32>, Col<f32>);
impl_scalar_ops!(f64, ColMut<'_, f64>, Col<f64>);

impl_scalar_ops!(f32, ColRef<'_, f32>, Col<f32>);
impl_scalar_ops!(f64, ColRef<'_, f64>, Col<f64>);

// Diag
impl_scalar_ops!(f32, Diag<f32>, Diag<f32>);
impl_scalar_ops!(f64, Diag<f64>, Diag<f64>);

impl_scalar_ops!(f32, DiagMut<'_, f32>, Diag<f32>);
impl_scalar_ops!(f64, DiagMut<'_, f64>, Diag<f64>);

impl_scalar_ops!(f32, DiagRef<'_, f32>, Diag<f32>);
impl_scalar_ops!(f64, DiagRef<'_, f64>, Diag<f64>);

// Mat
impl_scalar_ops!(f32, Mat<f32>, Mat<f32>);
impl_scalar_ops!(f64, Mat<f64>, Mat<f64>);

impl_scalar_ops!(f32, MatMut<'_, f32>, Mat<f32>);
impl_scalar_ops!(f64, MatMut<'_, f64>, Mat<f64>);

impl_scalar_ops!(f32, MatRef<'_, f32>, Mat<f32>);
impl_scalar_ops!(f64, MatRef<'_, f64>, Mat<f64>);

// Row
impl_scalar_ops!(f32, Row<f32>, Row<f32>);
impl_scalar_ops!(f64, Row<f64>, Row<f64>);

impl_scalar_ops!(f32, RowMut<'_, f32>, Row<f32>);
impl_scalar_ops!(f64, RowMut<'_, f64>, Row<f64>);

impl_scalar_ops!(f32, RowRef<'_, f32>, Row<f32>);
impl_scalar_ops!(f64, RowRef<'_, f64>, Row<f64>);

// SparseColMat
impl_scalar_ops!(f32, SparseColMat<usize, f32>, SparseColMat<usize, f32>);
impl_scalar_ops!(f64, SparseColMat<usize, f64>, SparseColMat<usize, f64>);

impl_scalar_ops!(f32, SparseColMatMut<'_, usize, f32>, SparseColMat<usize, f32>);
impl_scalar_ops!(f64, SparseColMatMut<'_, usize, f64>, SparseColMat<usize, f64>);

impl_scalar_ops!(f32, SparseColMatRef<'_, usize, f32>, SparseColMat<usize, f32>);
impl_scalar_ops!(f64, SparseColMatRef<'_, usize, f64>, SparseColMat<usize, f64>);

// SparseRowMat
impl_scalar_ops!(f32, SparseRowMat<usize, f32>, SparseRowMat<usize, f32>);
impl_scalar_ops!(f64, SparseRowMat<usize, f64>, SparseRowMat<usize, f64>);

impl_scalar_ops!(f32, SparseRowMatMut<'_, usize, f32>, SparseRowMat<usize, f32>);
impl_scalar_ops!(f64, SparseRowMatMut<'_, usize, f64>, SparseRowMat<usize, f64>);

impl_scalar_ops!(f32, SparseRowMatRef<'_, usize, f32>, SparseRowMat<usize, f32>);
impl_scalar_ops!(f64, SparseRowMatRef<'_, usize, f64>, SparseRowMat<usize, f64>);
