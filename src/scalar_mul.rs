use crate::{
    diag::{Diag, DiagMut, DiagRef},
    sparse::{
        SparseColMat, SparseColMatMut, SparseColMatRef, SparseRowMat, SparseRowMatMut,
        SparseRowMatRef,
    },
    Col, ColMut, ColRef, Mat, MatMut, MatRef, Row, RowMut, RowRef,
};
use core::ops::Mul;

macro_rules! impl_scalar_mul {
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
    };
}

// Col
impl_scalar_mul!(f32, Col<f32>, Col<f32>);
impl_scalar_mul!(f64, Col<f64>, Col<f64>);

impl_scalar_mul!(f32, ColMut<'_, f32>, Col<f32>);
impl_scalar_mul!(f64, ColMut<'_, f64>, Col<f64>);

impl_scalar_mul!(f32, ColRef<'_, f32>, Col<f32>);
impl_scalar_mul!(f64, ColRef<'_, f64>, Col<f64>);

// Diag
impl_scalar_mul!(f32, Diag<f32>, Diag<f32>);
impl_scalar_mul!(f64, Diag<f64>, Diag<f64>);

impl_scalar_mul!(f32, DiagMut<'_, f32>, Diag<f32>);
impl_scalar_mul!(f64, DiagMut<'_, f64>, Diag<f64>);

impl_scalar_mul!(f32, DiagRef<'_, f32>, Diag<f32>);
impl_scalar_mul!(f64, DiagRef<'_, f64>, Diag<f64>);

// Mat
impl_scalar_mul!(f32, Mat<f32>, Mat<f32>);
impl_scalar_mul!(f64, Mat<f64>, Mat<f64>);

impl_scalar_mul!(f32, MatMut<'_, f32>, Mat<f32>);
impl_scalar_mul!(f64, MatMut<'_, f64>, Mat<f64>);

impl_scalar_mul!(f32, MatRef<'_, f32>, Mat<f32>);
impl_scalar_mul!(f64, MatRef<'_, f64>, Mat<f64>);

// Row
impl_scalar_mul!(f32, Row<f32>, Row<f32>);
impl_scalar_mul!(f64, Row<f64>, Row<f64>);

impl_scalar_mul!(f32, RowMut<'_, f32>, Row<f32>);
impl_scalar_mul!(f64, RowMut<'_, f64>, Row<f64>);

impl_scalar_mul!(f32, RowRef<'_, f32>, Row<f32>);
impl_scalar_mul!(f64, RowRef<'_, f64>, Row<f64>);

// SparseColMat
impl_scalar_mul!(f32, SparseColMat<usize, f32>, SparseColMat<usize, f32>);
impl_scalar_mul!(f64, SparseColMat<usize, f64>, SparseColMat<usize, f64>);

impl_scalar_mul!(f32, SparseColMatMut<'_, usize, f32>, SparseColMat<usize, f32>);
impl_scalar_mul!(f64, SparseColMatMut<'_, usize, f64>, SparseColMat<usize, f64>);

impl_scalar_mul!(f32, SparseColMatRef<'_, usize, f32>, SparseColMat<usize, f32>);
impl_scalar_mul!(f64, SparseColMatRef<'_, usize, f64>, SparseColMat<usize, f64>);

// SparseRowMat
impl_scalar_mul!(f32, SparseRowMat<usize, f32>, SparseRowMat<usize, f32>);
impl_scalar_mul!(f64, SparseRowMat<usize, f64>, SparseRowMat<usize, f64>);

impl_scalar_mul!(f32, SparseRowMatMut<'_, usize, f32>, SparseRowMat<usize, f32>);
impl_scalar_mul!(f64, SparseRowMatMut<'_, usize, f64>, SparseRowMat<usize, f64>);

impl_scalar_mul!(f32, SparseRowMatRef<'_, usize, f32>, SparseRowMat<usize, f32>);
impl_scalar_mul!(f64, SparseRowMatRef<'_, usize, f64>, SparseRowMat<usize, f64>);
