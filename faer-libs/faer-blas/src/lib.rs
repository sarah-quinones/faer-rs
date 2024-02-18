mod conversions;
mod impls;

pub use impls::*;
use paste::paste;

// Complex scalar values in CBLAS parameters are passed as *void,
// whereas real scalars are passed directly as float or double.
// We dereference the complex scalars before we call the generic
// implementations
macro_rules! deref_arg {
    (c $arg:ident : SCALAR_TY) => {
        *$arg
    };
    (z $arg:ident : SCALAR_TY) => {
        *$arg
    };

    ($_:ident $arg:ident : $__:ty) => {
        $arg
    };
}

macro_rules! elem_type {
    (s) => {
        f32
    };
    (d) => {
        f64
    };
    (c) => {
        faer_core::c32
    };
    (z) => {
        faer_core::c64
    };
}

macro_rules! scalar_type {
    (s) => {
        f32
    };
    (d) => {
        f64
    };
    (c) => {
        *const faer_core::c32
    };
    (z) => {
        *const faer_core::c64
    };
}

// This is macro is pretty cursed: we munch on the type list, then for each
// type we declare a mod to create type aliases ELEM_TY and SCALAR_TY to "monomorphise" the
// "generic" types in the parameter list. A mod is required to constaint the scope
// of type alias ELEM_TY and SCALAR_TY.
// But we avoid writing a shim for each function, so maybe worth it
macro_rules! impl_fn {
    ({} $f:ident($($arg:ident : $arg_ty:ty),*)) => {};
    ({$prefix:ident $($tail:tt)*} $f:ident($($arg:ident : $arg_ty:ty),*)) => {
        paste!{
            mod [< mod_ $prefix $f >] {
                #![allow(non_camel_case_types)]
                use crate::impls::*;
                type ELEM_TY = elem_type!($prefix);
                type SCALAR_TY = scalar_type!($prefix);
                #[no_mangle]
                pub unsafe extern "C" fn [< cblas_ $prefix $f >]($($arg:$arg_ty),*) {
                    $f::<ELEM_TY>($(
                        deref_arg!($prefix $arg:$arg_ty),
                    )*);
                }
            }
            pub use [< mod_ $prefix $f >]::*;
        }
        impl_fn!({$($tail) *} $f($($arg:$arg_ty),*));
    }
}

impl_fn!({s d c z}gemv(layout: CblasLayout, trans: CblasTranspose, m: CblasInt, n: CblasInt, alpha: SCALAR_TY, a: *const ELEM_TY, lda: CblasInt, x: *const ELEM_TY, incx: CblasInt, beta: SCALAR_TY, y: *mut ELEM_TY, incy: CblasInt));
impl_fn!({s d c z}symm(layout: CblasLayout, side: CblasSide, uplo: CblasUpLo, m: CblasInt, n: CblasInt, alpha: SCALAR_TY, a: *const ELEM_TY, lda: CblasInt, b: *const ELEM_TY, ldb: CblasInt, beta: SCALAR_TY, c: *mut ELEM_TY, ldc: CblasInt));
impl_fn!({s d c z}gemm(layout: CblasLayout, trans_a: CblasTranspose, trans_b: CblasTranspose, m: CblasInt, n: CblasInt, k: CblasInt, alpha: SCALAR_TY, a: *const ELEM_TY, lda: CblasInt, b: *const ELEM_TY, ldb: CblasInt, beta: SCALAR_TY, c: *mut ELEM_TY, ldc: CblasInt));

#[cfg(test)]
mod tests {
    use crate::{
        cblas_sgemm, cblas_sgemv, cblas_ssymm,
        impls::{CblasLayout, CblasTranspose},
        CblasSide, CblasUpLo,
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

        assert_eq!(c, [58., 64., 139., 154.,]);

        /*
                       T
                | 1 4 |   | 7  8  |       | 58  64  |
            1 * | 2 5 | x | 9  10 | + 0 = | 139 154 |
                | 3 6 |   | 11 12 |
        */
        // Trans a
        let a = [1., 4., 2., 5., 3., 6.];
        let mut c = [0.; 4];

        unsafe {
            cblas_sgemm(
                CblasLayout::RowMajor,
                CblasTranspose::Trans,
                CblasTranspose::NoTrans,
                2,
                2,
                3,
                1.,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                0.,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c, [58., 64., 139., 154.,]);
    }

    #[test]
    fn test_symm() {
        /*
                | 1 2 3 |   | 7  8  |       | 58 64 |
            1 * | 2 0 2 | x | 9  10 | + 0 = | 36 40 |
                | 3 2 1 |   | 11 12 |       | 50 56 |
        */

        let a = [1., 2., 3., 2., 0., 2., 3., 2., 1.];
        let b = [7., 8., 9., 10., 11., 12.];
        let mut c = [0.; 6];

        unsafe {
            cblas_ssymm(
                CblasLayout::RowMajor,
                CblasSide::Left,
                CblasUpLo::Upper,
                3,
                2,
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

        assert_eq!(c, [58., 64., 36., 40., 50., 56.]);

        /*
                | 7  8  |   | 1 2 |       | 23 22 |
            1 * | 9  10 | * | 2 1 | + 0 = | 29 28 |
                | 11 12 |                 | 35 34 |
        */

        let a = [1., 2., 2., 1.];
        let mut c = [0.; 6];

        unsafe {
            cblas_ssymm(
                CblasLayout::RowMajor,
                CblasSide::Right,
                CblasUpLo::Upper,
                3,
                2,
                1.,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                0.,
                c.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(c, [23., 22., 29., 28., 35., 34.]);
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
