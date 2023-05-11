mod conversions;

use conversions::{from_blas, from_blas_vec, from_blas_vec_mut, Stride};
use faer_core::{mul::matvec::matvec_with_conj, ComplexField, Conj};
use gemm::{c32, c64, Parallelism};

type CblasInt = i32;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasLayout {
    CblasRowMajor = 101,
    CblasColMajor = 102,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasTranspose {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113,
}

impl CblasTranspose {
    fn has_trans(&self) -> bool {
        matches!(self, Self::CblasTrans | Self::CblasConjTrans)
    }

    fn conj(&self) -> Conj {
        match self {
            CblasTranspose::CblasNoTrans | CblasTranspose::CblasTrans => Conj::No,
            CblasTranspose::CblasConjTrans => Conj::Yes,
        }
    }
}

/*
    BLAS level 2 functions
*/

macro_rules! impl_gemv {
    ($t: ty, $f: ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $f(
            layout: CblasLayout,
            trans: CblasTranspose,
            m: CblasInt,
            n: CblasInt,
            alpha: $t,
            a: *const $t,
            lda: CblasInt,
            x: *const $t,
            incx: CblasInt,
            beta: $t,
            y: *mut $t,
            incy: CblasInt,
        ) {
            // The definitions of alpha and beta are swapped
            let (faer_alpha, faer_beta) = (beta, alpha);

            let mut a = unsafe { from_blas::<$t>(layout, a, m, n, lda) };
            if trans.has_trans() {
                a = a.transpose();
            }
            let x = unsafe { from_blas_vec::<$t>(x, n, incx) };
            let y = unsafe { from_blas_vec_mut::<$t>(y, m, incy) };

            let faer_alpha = (faer_alpha != <$t>::zero()).then_some(faer_alpha);
            matvec_with_conj(y, a, trans.conj(), x, Conj::No, faer_alpha, faer_beta);
        }
    };
}

impl_gemv!(f32, cblas_sgemv);
impl_gemv!(f64, cblas_dgemv);
impl_gemv!(faer_core::c32, cblas_cgemv);
impl_gemv!(faer_core::c64, cblas_zgemv);

/*
    BLAS level 3 functions
*/

macro_rules! impl_gemm {
    ($t: ty, $f: ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $f(
            layout: CblasLayout,
            trans_a: CblasTranspose,
            trans_b: CblasTranspose,
            m: CblasInt,
            n: CblasInt,
            k: CblasInt,
            alpha: $t,
            a: *const $t,
            lda: CblasInt,
            b: *const $t,
            ldb: CblasInt,
            beta: $t,
            c: *mut $t,
            ldc: CblasInt,
        ) {
            // The definitions of alpha and beta are swapped
            let (faer_alpha, faer_beta) = (beta, alpha);

            let a_stride = Stride::from_leading_dim(layout, lda);
            let b_stride = Stride::from_leading_dim(layout, ldb);
            let c_stride = Stride::from_leading_dim(layout, ldc);

            gemm::gemm(
                m as usize,
                n as usize,
                k as usize,
                c,
                c_stride.col,
                c_stride.row,
                false,
                a,
                a_stride.col,
                a_stride.row,
                b,
                b_stride.col,
                b_stride.row,
                faer_alpha,
                faer_beta,
                false,
                trans_a.conj() == Conj::Yes,
                trans_b.conj() == Conj::Yes,
                Parallelism::Rayon(0),
            );
        }
    };
}

impl_gemm!(f32, cblas_sgemm);
impl_gemm!(f64, cblas_dgemm);
impl_gemm!(c32, cblas_cgemm);
impl_gemm!(c64, cblas_zgemm);

#[cfg(test)]
mod tests {
    use crate::{cblas_sgemm, cblas_sgemv, CblasLayout, CblasTranspose};

    #[test]
    fn test_gemm() {
        /*
                | 1 2 3 |   | 7  8  |       | 58  64  |
            1 * | 4 5 6 | x | 9  10 | + 0 = | 139 154 |
                            | 11 12 |
        */
        let a = [1., 2., 3., 4., 5., 6.];
        let b = [7., 8., 9., 10., 11., 12.];

        let mut c = [0.; 4];

        unsafe {
            cblas_sgemm(
                CblasLayout::CblasRowMajor,
                CblasTranspose::CblasNoTrans,
                CblasTranspose::CblasNoTrans,
                2,
                2,
                3,
                1.,
                a.as_ptr(),
                3,
                b.as_ptr(),
                2,
                0.,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c, [58., 64., 139., 154.,])
    }

    #[test]
    fn test_gemv() {
        /*      | 1 3 5 |   | 1 |       | 22 |
            1 * | 2 4 6 | * | 2 | + 0 = | 28 |
                            | 3 |
        */
        let a = [1., 2., 0., 3., 4., 0., 5., 6., 0.]; // one excess
        let x = [1., 2., 3.];
        let mut y = [f32::NAN; 2];

        unsafe {
            cblas_sgemv(
                CblasLayout::CblasColMajor,
                CblasTranspose::CblasNoTrans,
                2,
                3,
                1.,
                a.as_ptr(),
                3,
                x.as_ptr(),
                1,
                0.,
                y.as_mut_ptr(),
                1,
            );
        }

        assert_eq!(y, [22., 28.]);
    }
}
