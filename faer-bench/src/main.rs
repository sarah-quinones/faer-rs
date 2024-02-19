#![allow(dead_code)]
use coe::is_same;
use eyre::Result;
use faer_core::{c32, c64, Parallelism};
use human_repr::HumanDuration;
use std::{fs::File, io::Write, time::Duration};

// use double_f64::DoubleF64 as f128;
type f128 = f64;
#[allow(non_camel_case_types)]
type c128 = num_complex::Complex<f128>;

extern crate blas_src;
extern crate openmp_sys;

fn random<T: 'static>() -> T {
    if is_same::<f32, T>() {
        coe::coerce_static(rand::random::<f32>())
    } else if is_same::<f64, T>() {
        coe::coerce_static(rand::random::<f64>())
    // } else if is_same::<f128, T>() {
    //     coe::coerce_static(double_f64::Double(rand::random::<f64>(), 0.0))
    } else if is_same::<c32, T>() {
        coe::coerce_static(c32::new(rand::random(), rand::random()))
    } else if is_same::<c64, T>() {
        coe::coerce_static(c64::new(rand::random(), rand::random()))
    // } else if is_same::<c128, T>() {
    //     coe::coerce_static(c128::new(
    //         double_f64::Double(rand::random(), 0.0),
    //         double_f64::Double(rand::random(), 0.0),
    //     ))
    } else if is_same::<Cplx32, T>() {
        coe::coerce_static(Cplx32::new(rand::random(), rand::random()))
    } else if is_same::<Cplx64, T>() {
        coe::coerce_static(Cplx64::new(rand::random(), rand::random()))
    } else {
        unimplemented!()
    }
}

fn time(mut f: impl FnMut()) -> f64 {
    let instant = std::time::Instant::now();
    f();
    instant.elapsed().as_secs_f64()
}

fn timeit(f: impl FnMut()) -> f64 {
    let mut f = f;
    let min = 1e-0;
    let once = time(&mut f);
    if once > min {
        return once;
    }

    let ten = time(|| {
        for _ in 0..10 {
            f()
        }
    });

    if ten > min {
        return ten / 10.0;
    }

    let n = (min * 10.0 / ten).ceil() as u64;
    time(|| {
        for _ in 0..n {
            f()
        }
    }) / n as f64
}

// mod double_f64;

mod gemm;
mod tr_inverse;
mod trsm;

mod cholesky;
mod col_piv_qr;
mod evd;
mod full_piv_lu;
mod inverse;
mod no_piv_qr;
mod partial_piv_lu;
mod rectangular_svd;
mod svd;
mod symmetric_evd;

macro_rules! printwriteln {
    ($out: expr, $($arg:tt)*) => {
        {
            println!($($arg)*);
            writeln!($out, $($arg)*)
        }
    };
}

#[cfg(feature = "nalgebra")]
macro_rules! ifnalgebra {
    ($arg: expr) => {{
        &$arg
    }};
}
#[cfg(not(feature = "nalgebra"))]
macro_rules! ifnalgebra {
    ($arg: expr) => {{
        &vec![Duration::ZERO; N]
    }};
}

#[cfg(feature = "eigen")]
macro_rules! ifeigen {
    ($arg: expr) => {{
        &$arg
    }};
}
#[cfg(not(feature = "eigen"))]
macro_rules! ifeigen {
    ($arg: expr) => {{
        &vec![Duration::ZERO; N]
    }};
}

fn print_results(
    output: &mut dyn Write,
    input_sizes: &[usize],
    faer: &[Duration],
    faer_parallel: &[Duration],
    ndarray: &[Duration],
    nalgebra: &[Duration],
    eigen: &[Duration],
) -> Result<()> {
    let fmt = |d: Duration| {
        if d == Duration::ZERO {
            "-".to_string()
        } else {
            format!("{}", d.human_duration())
        }
    };
    printwriteln!(
        output,
        "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "n",
        "faer",
        "faer(par)",
        "ndarray",
        "nalgebra",
        "eigen",
    )?;

    for (i, n) in input_sizes.iter().copied().enumerate() {
        printwriteln!(
            output,
            "{:5} {:>10} {:>10} {:>10} {:>10} {:>10}",
            n,
            fmt(faer[i]),
            fmt(faer_parallel[i]),
            fmt(ndarray[i]),
            fmt(nalgebra[i]),
            fmt(eigen[i]),
        )?;
    }
    Ok(())
}

#[cfg(feature = "eigen")]
mod eigen {
    extern "C" {
        pub fn gemm_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trsm_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trinv_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn chol_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn plu_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn flu_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn qr_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn colqr_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn inverse_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn svd_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn rectangular_svd_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn symmetric_evd_f32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn evd_f32(out: *mut f64, inputs: *const usize, count: usize);

        pub fn gemm_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trsm_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trinv_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn chol_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn plu_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn flu_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn qr_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn colqr_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn inverse_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn svd_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn rectangular_svd_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn symmetric_evd_f64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn evd_f64(out: *mut f64, inputs: *const usize, count: usize);

        pub fn gemm_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trsm_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trinv_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn chol_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn plu_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn flu_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn qr_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn colqr_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn inverse_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn svd_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn rectangular_svd_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn symmetric_evd_f128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn evd_f128(out: *mut f64, inputs: *const usize, count: usize);

        pub fn gemm_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trsm_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trinv_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn chol_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn plu_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn flu_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn qr_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn colqr_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn inverse_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn svd_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn rectangular_svd_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn symmetric_evd_c32(out: *mut f64, inputs: *const usize, count: usize);
        pub fn evd_c32(out: *mut f64, inputs: *const usize, count: usize);

        pub fn gemm_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trsm_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trinv_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn chol_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn plu_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn flu_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn qr_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn colqr_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn inverse_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn svd_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn rectangular_svd_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn symmetric_evd_c64(out: *mut f64, inputs: *const usize, count: usize);
        pub fn evd_c64(out: *mut f64, inputs: *const usize, count: usize);

        pub fn gemm_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trsm_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trinv_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn chol_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn plu_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn flu_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn qr_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn colqr_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn inverse_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn svd_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn rectangular_svd_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn symmetric_evd_c128(out: *mut f64, inputs: *const usize, count: usize);
        pub fn evd_c128(out: *mut f64, inputs: *const usize, count: usize);
    }
}

type Cplx32 = num_complex::Complex32;
type Cplx64 = num_complex::Complex64;

fn eigen(
    f: unsafe extern "C" fn(*mut f64, *const usize, usize),
    input_sizes: &[usize],
) -> Vec<Duration> {
    let count = input_sizes.len();
    let mut v = vec![0.0; count];
    unsafe {
        f(v.as_mut_ptr(), input_sizes.as_ptr(), count);
    }
    v.into_iter().map(Duration::from_secs_f64).collect()
}

const INPUT_SIZES: &[usize] = &[
    4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 896, 1024,
];
const N: usize = INPUT_SIZES.len();
fn main() -> Result<()> {
    let input_sizes = INPUT_SIZES;

    {
        println!("f32");
        let mut file = File::create("f32.txt")?;
        printwriteln!(
            file,
            "
## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &gemm::faer::<f32>(&input_sizes, Parallelism::None),
            &gemm::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &gemm::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&gemm::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::gemm_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &trsm::faer::<f32>(&input_sizes, Parallelism::None),
            &trsm::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &trsm::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&trsm::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::trsm_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &tr_inverse::faer::<f32>(&input_sizes, Parallelism::None),
            &tr_inverse::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &tr_inverse::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&tr_inverse::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::trinv_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &cholesky::faer::<f32>(&input_sizes, Parallelism::None),
            &cholesky::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &cholesky::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&cholesky::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::chol_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &partial_piv_lu::faer::<f32>(&input_sizes, Parallelism::None),
            &partial_piv_lu::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &partial_piv_lu::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&partial_piv_lu::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::plu_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &full_piv_lu::faer::<f32>(&input_sizes, Parallelism::None),
            &full_piv_lu::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &full_piv_lu::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&full_piv_lu::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::flu_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &no_piv_qr::faer::<f32>(&input_sizes, Parallelism::None),
            &no_piv_qr::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &no_piv_qr::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&no_piv_qr::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::qr_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &col_piv_qr::faer::<f32>(&input_sizes, Parallelism::None),
            &col_piv_qr::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &col_piv_qr::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&col_piv_qr::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::colqr_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &inverse::faer::<f32>(&input_sizes, Parallelism::None),
            &inverse::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &inverse::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&inverse::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::inverse_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &svd::faer::<f32>(&input_sizes, Parallelism::None),
            &svd::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &svd::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&svd::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::svd_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Thin matrix singular value decomposition

Computing the SVD of a rectangular matrix with shape `(4096, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &rectangular_svd::faer::<f32>(&input_sizes, Parallelism::None),
            &rectangular_svd::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &rectangular_svd::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&rectangular_svd::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::rectangular_svd_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Hermitian matrix eigenvalue decomposition

Computing the EVD of a hermitian matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &symmetric_evd::faer::<f32>(&input_sizes, Parallelism::None),
            &symmetric_evd::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &symmetric_evd::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&symmetric_evd::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::symmetric_evd_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Non Hermitian matrix eigenvalue decomposition

Computing the EVD of a matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &evd::faer::<f32>(&input_sizes, Parallelism::None),
            &evd::faer::<f32>(&input_sizes, Parallelism::Rayon(0)),
            &evd::ndarray::<f32>(&input_sizes),
            ifnalgebra!(&evd::nalgebra::<f32>(&input_sizes)),
            ifeigen!(&eigen(eigen::evd_f32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;
    }
    {
        println!("f64");
        let mut file = File::create("f64.txt")?;
        printwriteln!(
            file,
            "
## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &gemm::faer::<f64>(&input_sizes, Parallelism::None),
            &gemm::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &gemm::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&gemm::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::gemm_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &trsm::faer::<f64>(&input_sizes, Parallelism::None),
            &trsm::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &trsm::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&trsm::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::trsm_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &tr_inverse::faer::<f64>(&input_sizes, Parallelism::None),
            &tr_inverse::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &tr_inverse::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&tr_inverse::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::trinv_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &cholesky::faer::<f64>(&input_sizes, Parallelism::None),
            &cholesky::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &cholesky::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&cholesky::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::chol_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &partial_piv_lu::faer::<f64>(&input_sizes, Parallelism::None),
            &partial_piv_lu::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &partial_piv_lu::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&partial_piv_lu::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::plu_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &full_piv_lu::faer::<f64>(&input_sizes, Parallelism::None),
            &full_piv_lu::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &full_piv_lu::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&full_piv_lu::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::flu_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &no_piv_qr::faer::<f64>(&input_sizes, Parallelism::None),
            &no_piv_qr::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &no_piv_qr::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&no_piv_qr::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::qr_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &col_piv_qr::faer::<f64>(&input_sizes, Parallelism::None),
            &col_piv_qr::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &col_piv_qr::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&col_piv_qr::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::colqr_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &inverse::faer::<f64>(&input_sizes, Parallelism::None),
            &inverse::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &inverse::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&inverse::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::inverse_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &svd::faer::<f64>(&input_sizes, Parallelism::None),
            &svd::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &svd::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&svd::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::svd_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Thin matrix singular value decomposition

Computing the SVD of a rectangular matrix with shape `(4096, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &rectangular_svd::faer::<f64>(&input_sizes, Parallelism::None),
            &rectangular_svd::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &rectangular_svd::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&rectangular_svd::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::rectangular_svd_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Hermitian matrix eigenvalue decomposition

Computing the EVD of a hermitian matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &symmetric_evd::faer::<f64>(&input_sizes, Parallelism::None),
            &symmetric_evd::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &symmetric_evd::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&symmetric_evd::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::symmetric_evd_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Non Hermitian matrix eigenvalue decomposition

Computing the EVD of a matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &evd::faer::<f64>(&input_sizes, Parallelism::None),
            &evd::faer::<f64>(&input_sizes, Parallelism::Rayon(0)),
            &evd::ndarray::<f64>(&input_sizes),
            ifnalgebra!(&evd::nalgebra::<f64>(&input_sizes)),
            ifeigen!(&eigen(eigen::evd_f64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;
    }
    {
        println!("f128");
        let mut file = File::create("f128.txt")?;
        printwriteln!(
            file,
            "
## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &gemm::faer::<f128>(&input_sizes, Parallelism::None),
            &gemm::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::gemm_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A`
is a triangular matrix.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &trsm::faer::<f128>(&input_sizes, Parallelism::None),
            &trsm::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::trsm_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &tr_inverse::faer::<f128>(&input_sizes, Parallelism::None),
            &tr_inverse::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::trinv_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &cholesky::faer::<f128>(&input_sizes, Parallelism::None),
            &cholesky::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::chol_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix,
`L` is unit lower triangular and `U` is upper triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &partial_piv_lu::faer::<f128>(&input_sizes, Parallelism::None),
            &partial_piv_lu::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::plu_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are
permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &full_piv_lu::faer::<f128>(&input_sizes, Parallelism::None),
            &full_piv_lu::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::flu_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper
triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &no_piv_qr::faer::<f128>(&input_sizes, Parallelism::None),
            &no_piv_qr::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::qr_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix,
`Q` is unitary and `R` is upper triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &col_piv_qr::faer::<f128>(&input_sizes, Parallelism::None),
            &col_piv_qr::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::colqr_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &inverse::faer::<f128>(&input_sizes, Parallelism::None),
            &inverse::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::inverse_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &svd::faer::<f128>(&input_sizes, Parallelism::None),
            &svd::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::svd_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Thin matrix singular value decomposition

Computing the SVD of a rectangular matrix with shape `(4096, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &rectangular_svd::faer::<f128>(&input_sizes, Parallelism::None),
            &rectangular_svd::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::rectangular_svd_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Hermitian matrix eigenvalue decomposition

Computing the EVD of a hermitian matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &symmetric_evd::faer::<f128>(&input_sizes, Parallelism::None),
            &symmetric_evd::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::symmetric_evd_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Non Hermitian matrix eigenvalue decomposition

Computing the EVD of a matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &evd::faer::<f128>(&input_sizes, Parallelism::None),
            &evd::faer::<f128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::evd_f128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;
    }
    {
        println!("c32");
        let mut file = File::create("c32.txt")?;
        printwriteln!(
            file,
            "
## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &gemm::faer::<c32>(&input_sizes, Parallelism::None),
            &gemm::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &gemm::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&gemm::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::gemm_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &trsm::faer::<c32>(&input_sizes, Parallelism::None),
            &trsm::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &trsm::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&trsm::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::trsm_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &tr_inverse::faer::<c32>(&input_sizes, Parallelism::None),
            &tr_inverse::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &tr_inverse::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&tr_inverse::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::trinv_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &cholesky::faer::<c32>(&input_sizes, Parallelism::None),
            &cholesky::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &cholesky::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&cholesky::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::chol_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &partial_piv_lu::faer::<c32>(&input_sizes, Parallelism::None),
            &partial_piv_lu::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &partial_piv_lu::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&partial_piv_lu::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::plu_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &full_piv_lu::faer::<c32>(&input_sizes, Parallelism::None),
            &full_piv_lu::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &full_piv_lu::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&full_piv_lu::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::flu_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &no_piv_qr::faer::<c32>(&input_sizes, Parallelism::None),
            &no_piv_qr::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &no_piv_qr::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&no_piv_qr::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::qr_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &col_piv_qr::faer::<c32>(&input_sizes, Parallelism::None),
            &col_piv_qr::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &col_piv_qr::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&col_piv_qr::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::colqr_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &inverse::faer::<c32>(&input_sizes, Parallelism::None),
            &inverse::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &inverse::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&inverse::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::inverse_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &svd::faer::<c32>(&input_sizes, Parallelism::None),
            &svd::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &svd::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&svd::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::svd_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Thin matrix singular value decomposition

Computing the SVD of a rectangular matrix with shape `(4096, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &rectangular_svd::faer::<c32>(&input_sizes, Parallelism::None),
            &rectangular_svd::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &rectangular_svd::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&rectangular_svd::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::rectangular_svd_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Hermitian matrix eigenvalue decomposition

Computing the EVD of a hermitian matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &symmetric_evd::faer::<c32>(&input_sizes, Parallelism::None),
            &symmetric_evd::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &symmetric_evd::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&symmetric_evd::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::symmetric_evd_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Non Hermitian matrix eigenvalue decomposition

Computing the EVD of a matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &evd::faer::<c32>(&input_sizes, Parallelism::None),
            &evd::faer::<c32>(&input_sizes, Parallelism::Rayon(0)),
            &evd::ndarray::<Cplx32>(&input_sizes),
            ifnalgebra!(&evd::nalgebra::<Cplx32>(&input_sizes)),
            ifeigen!(&eigen(eigen::evd_c32, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;
    }
    {
        println!("c64");
        let mut file = File::create("c64.txt")?;
        printwriteln!(
            file,
            "
## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &gemm::faer::<c64>(&input_sizes, Parallelism::None),
            &gemm::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &gemm::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&gemm::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::gemm_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &trsm::faer::<c64>(&input_sizes, Parallelism::None),
            &trsm::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &trsm::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&trsm::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::trsm_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &tr_inverse::faer::<c64>(&input_sizes, Parallelism::None),
            &tr_inverse::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &tr_inverse::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&tr_inverse::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::trinv_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &cholesky::faer::<c64>(&input_sizes, Parallelism::None),
            &cholesky::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &cholesky::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&cholesky::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::chol_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &partial_piv_lu::faer::<c64>(&input_sizes, Parallelism::None),
            &partial_piv_lu::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &partial_piv_lu::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&partial_piv_lu::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::plu_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &full_piv_lu::faer::<c64>(&input_sizes, Parallelism::None),
            &full_piv_lu::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &full_piv_lu::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&full_piv_lu::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::flu_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &no_piv_qr::faer::<c64>(&input_sizes, Parallelism::None),
            &no_piv_qr::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &no_piv_qr::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&no_piv_qr::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::qr_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &col_piv_qr::faer::<c64>(&input_sizes, Parallelism::None),
            &col_piv_qr::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &col_piv_qr::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&col_piv_qr::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::colqr_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &inverse::faer::<c64>(&input_sizes, Parallelism::None),
            &inverse::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &inverse::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&inverse::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::inverse_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &svd::faer::<c64>(&input_sizes, Parallelism::None),
            &svd::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &svd::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&svd::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::svd_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Thin matrix singular value decomposition

Computing the SVD of a rectangular matrix with shape `(4096, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &rectangular_svd::faer::<c64>(&input_sizes, Parallelism::None),
            &rectangular_svd::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &rectangular_svd::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&rectangular_svd::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::rectangular_svd_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Hermitian matrix eigenvalue decomposition

Computing the EVD of a hermitian matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &symmetric_evd::faer::<c64>(&input_sizes, Parallelism::None),
            &symmetric_evd::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &symmetric_evd::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&symmetric_evd::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::symmetric_evd_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Non Hermitian matrix eigenvalue decomposition

Computing the EVD of a matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &evd::faer::<c64>(&input_sizes, Parallelism::None),
            &evd::faer::<c64>(&input_sizes, Parallelism::Rayon(0)),
            &evd::ndarray::<Cplx64>(&input_sizes),
            ifnalgebra!(&evd::nalgebra::<Cplx64>(&input_sizes)),
            ifeigen!(&eigen(eigen::evd_c64, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;
    }
    {
        println!("c128");
        let mut file = File::create("c128.txt")?;
        printwriteln!(
            file,
            "
## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &gemm::faer::<c128>(&input_sizes, Parallelism::None),
            &gemm::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::gemm_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &trsm::faer::<c128>(&input_sizes, Parallelism::None),
            &trsm::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::trsm_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &tr_inverse::faer::<c128>(&input_sizes, Parallelism::None),
            &tr_inverse::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::trinv_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &cholesky::faer::<c128>(&input_sizes, Parallelism::None),
            &cholesky::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::chol_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &partial_piv_lu::faer::<c128>(&input_sizes, Parallelism::None),
            &partial_piv_lu::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::plu_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &full_piv_lu::faer::<c128>(&input_sizes, Parallelism::None),
            &full_piv_lu::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::flu_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &no_piv_qr::faer::<c128>(&input_sizes, Parallelism::None),
            &no_piv_qr::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::qr_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(file,"
## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```")?;
        print_results(
            &mut file,
            &input_sizes,
            &col_piv_qr::faer::<c128>(&input_sizes, Parallelism::None),
            &col_piv_qr::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::colqr_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &inverse::faer::<c128>(&input_sizes, Parallelism::None),
            &inverse::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::inverse_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &svd::faer::<c128>(&input_sizes, Parallelism::None),
            &svd::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::svd_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Thin matrix singular value decomposition

Computing the SVD of a rectangular matrix with shape `(4096, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &rectangular_svd::faer::<c128>(&input_sizes, Parallelism::None),
            &rectangular_svd::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::rectangular_svd_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Hermitian matrix eigenvalue decomposition

Computing the EVD of a hermitian matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &symmetric_evd::faer::<c128>(&input_sizes, Parallelism::None),
            &symmetric_evd::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::symmetric_evd_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;

        printwriteln!(
            file,
            "
## Non Hermitian matrix eigenvalue decomposition

Computing the EVD of a matrix with shape `(n, n)`.

```"
        )?;
        print_results(
            &mut file,
            &input_sizes,
            &evd::faer::<c128>(&input_sizes, Parallelism::None),
            &evd::faer::<c128>(&input_sizes, Parallelism::Rayon(0)),
            &[Duration::ZERO; N],
            &[Duration::ZERO; N],
            ifeigen!(&eigen(eigen::evd_c128, &input_sizes)),
        )?;
        printwriteln!(file, "```")?;
    }
    Ok(())
}
