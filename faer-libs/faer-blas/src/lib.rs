mod conversions;
mod impls;

use paste::paste;

// This is macro is pretty cursed: we munch on the type list, then for each
// type we declare a mod to create a type alias TY to "monomorphise" the
// "generic" TY in the argument list. A mod is required to constaint the scope
// of type alias TY.
// But we avoid writing a shim for each function, so maybe worth it
macro_rules! impl_fn {
    ($f:ident($($arg:ident : $arg_ty:ty),*), []) => {};
    ($f:ident($($arg:ident : $arg_ty:ty),*), [$t:ty : $prefix:ident $($tail:tt)*]) => {
        paste!{
            mod [< m $prefix $f >] {
                use crate::impls::*;
                type TY = $t;
                #[no_mangle]
                pub unsafe extern "C" fn [< cblas_ $prefix $f >](
                    $($arg : $arg_ty),*
                ) {
                    crate::impls::$f::<$t>($($arg),*);
                }
            }
            pub use [< m $prefix $f >]::*;
        }
        impl_fn!($f($($arg : $arg_ty),*), [$($tail) *]);
    }
}

impl_fn!(gemv(layout: CblasLayout, trans: CblasTranspose, m: CblasInt, n: CblasInt, alpha: TY, a: *const TY, lda: CblasInt, x: *const TY, incx: CblasInt, beta: TY, y: *mut TY, incy: CblasInt), [f32:s f64:d faer_core::c32:c faer_core::c64:z]);

impl_fn!(symm(layout: CblasLayout, side: CblasSide, _uplo: CblasUpLo, m: CblasInt, n: CblasInt, alpha: TY, a: *const TY, lda: CblasInt, b: *const TY, ldb: CblasInt, beta: TY, c: *mut TY, ldc: CblasInt), [f32:s f64:d gemm::c32:c gemm::c64:z]);
impl_fn!(gemm(layout: CblasLayout, trans_a: CblasTranspose, trans_b: CblasTranspose, m: CblasInt, n: CblasInt, k: CblasInt, alpha: TY, a: *const TY, lda: CblasInt, b: *const TY, ldb: CblasInt, beta: TY, c: *mut TY, ldc: CblasInt), [f32:s f64:d gemm::c32:c gemm::c64:z]);

#[cfg(test)]
mod tests {
    use crate::{
        cblas_sgemm, cblas_sgemv,
        impls::{CblasLayout, CblasTranspose},
    };

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
                CblasLayout::RowMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::NoTrans,
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
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
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
