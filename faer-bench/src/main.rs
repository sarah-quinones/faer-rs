use coe::is_same;
use eyre::Result;
use faer_core::{c32, c64, Parallelism};
use human_repr::HumanDuration;
use std::{fs::File, io::Write, time::Duration};

extern crate blas_src;
extern crate openmp_sys;

fn random<T: 'static>() -> T {
    if is_same::<f32, T>() {
        coe::coerce_static(rand::random::<f32>())
    } else if is_same::<f64, T>() {
        coe::coerce_static(rand::random::<f64>())
    } else if is_same::<c32, T>() {
        coe::coerce_static(c32::new(rand::random(), rand::random()))
    } else if is_same::<c64, T>() {
        coe::coerce_static(c64::new(rand::random(), rand::random()))
    } else {
        unimplemented!()
    }
}

fn epsilon<T: faer_core::ComplexField>() -> T::Real {
    if is_same::<f32, T>() {
        coe::coerce_static(f32::EPSILON)
    } else if is_same::<f64, T>() {
        coe::coerce_static(f64::EPSILON)
    } else if is_same::<c32, T>() {
        coe::coerce_static(f32::EPSILON)
    } else if is_same::<c64, T>() {
        coe::coerce_static(f64::EPSILON)
    } else {
        unimplemented!()
    }
}

fn min_positive<T: faer_core::ComplexField>() -> T::Real {
    if is_same::<f32, T>() {
        coe::coerce_static(f32::MIN_POSITIVE)
    } else if is_same::<f64, T>() {
        coe::coerce_static(f64::MIN_POSITIVE)
    } else if is_same::<c32, T>() {
        coe::coerce_static(f32::MIN_POSITIVE)
    } else if is_same::<c64, T>() {
        coe::coerce_static(f64::MIN_POSITIVE)
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

mod gemm;
mod tr_inverse;
mod trsm;

mod cholesky;
mod col_piv_qr;
mod full_piv_lu;
mod inverse;
mod no_piv_qr;
mod partial_piv_lu;
mod rectangular_svd;
mod svd;

macro_rules! printwriteln {
    ($out: expr, $($arg:tt)*) => {
        {
            println!($($arg)*);
            writeln!($out, $($arg)*)
        }
    };
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
    }
}

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

fn main() -> Result<()> {
    let input_sizes = vec![32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 896, 1024];

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
            &gemm::nalgebra::<f32>(&input_sizes),
            &eigen(eigen::gemm_f32, &input_sizes),
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
            &trsm::nalgebra::<f32>(&input_sizes),
            &eigen(eigen::trsm_f32, &input_sizes),
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
            &tr_inverse::nalgebra::<f32>(&input_sizes),
            &eigen(eigen::trinv_f32, &input_sizes),
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
            &cholesky::nalgebra::<f32>(&input_sizes),
            &eigen(eigen::chol_f32, &input_sizes),
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
            &partial_piv_lu::nalgebra::<f32>(&input_sizes),
            &eigen(eigen::plu_f32, &input_sizes),
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
            &full_piv_lu::nalgebra::<f32>(&input_sizes),
            &eigen(eigen::flu_f32, &input_sizes),
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
            &no_piv_qr::nalgebra::<f32>(&input_sizes),
            &eigen(eigen::qr_f32, &input_sizes),
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
            &col_piv_qr::nalgebra::<f32>(&input_sizes),
            &eigen(eigen::colqr_f32, &input_sizes),
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
            &inverse::nalgebra::<f32>(&input_sizes),
            &eigen(eigen::inverse_f32, &input_sizes),
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
            &svd::nalgebra::<f32>(&input_sizes),
            &eigen(eigen::svd_f32, &input_sizes),
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
            &rectangular_svd::nalgebra::<f32>(&input_sizes),
            &eigen(eigen::rectangular_svd_f32, &input_sizes),
        )?;
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
            &gemm::nalgebra::<f64>(&input_sizes),
            &eigen(eigen::gemm_f64, &input_sizes),
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
            &trsm::nalgebra::<f64>(&input_sizes),
            &eigen(eigen::trsm_f64, &input_sizes),
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
            &tr_inverse::nalgebra::<f64>(&input_sizes),
            &eigen(eigen::trinv_f64, &input_sizes),
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
            &cholesky::nalgebra::<f64>(&input_sizes),
            &eigen(eigen::chol_f64, &input_sizes),
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
            &partial_piv_lu::nalgebra::<f64>(&input_sizes),
            &eigen(eigen::plu_f64, &input_sizes),
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
            &full_piv_lu::nalgebra::<f64>(&input_sizes),
            &eigen(eigen::flu_f64, &input_sizes),
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
            &no_piv_qr::nalgebra::<f64>(&input_sizes),
            &eigen(eigen::qr_f64, &input_sizes),
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
            &col_piv_qr::nalgebra::<f64>(&input_sizes),
            &eigen(eigen::colqr_f64, &input_sizes),
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
            &inverse::nalgebra::<f64>(&input_sizes),
            &eigen(eigen::inverse_f64, &input_sizes),
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
            &svd::nalgebra::<f64>(&input_sizes),
            &eigen(eigen::svd_f64, &input_sizes),
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
            &rectangular_svd::nalgebra::<f64>(&input_sizes),
            &eigen(eigen::rectangular_svd_f64, &input_sizes),
        )?;
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
            &gemm::ndarray::<c32>(&input_sizes),
            &gemm::nalgebra::<c32>(&input_sizes),
            &eigen(eigen::gemm_c32, &input_sizes),
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
            &trsm::ndarray::<c32>(&input_sizes),
            &trsm::nalgebra::<c32>(&input_sizes),
            &eigen(eigen::trsm_c32, &input_sizes),
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
            &tr_inverse::ndarray::<c32>(&input_sizes),
            &tr_inverse::nalgebra::<c32>(&input_sizes),
            &eigen(eigen::trinv_c32, &input_sizes),
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
            &cholesky::ndarray::<c32>(&input_sizes),
            &cholesky::nalgebra::<c32>(&input_sizes),
            &eigen(eigen::chol_c32, &input_sizes),
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
            &partial_piv_lu::ndarray::<c32>(&input_sizes),
            &partial_piv_lu::nalgebra::<c32>(&input_sizes),
            &eigen(eigen::plu_c32, &input_sizes),
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
            &full_piv_lu::ndarray::<c32>(&input_sizes),
            &full_piv_lu::nalgebra::<c32>(&input_sizes),
            &eigen(eigen::flu_c32, &input_sizes),
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
            &no_piv_qr::ndarray::<c32>(&input_sizes),
            &no_piv_qr::nalgebra::<c32>(&input_sizes),
            &eigen(eigen::qr_c32, &input_sizes),
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
            &col_piv_qr::ndarray::<c32>(&input_sizes),
            &col_piv_qr::nalgebra::<c32>(&input_sizes),
            &eigen(eigen::colqr_c32, &input_sizes),
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
            &inverse::ndarray::<c32>(&input_sizes),
            &inverse::nalgebra::<c32>(&input_sizes),
            &eigen(eigen::inverse_c32, &input_sizes),
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
            &svd::ndarray::<c32>(&input_sizes),
            &svd::nalgebra::<c32>(&input_sizes),
            &eigen(eigen::svd_c32, &input_sizes),
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
            &rectangular_svd::ndarray::<c32>(&input_sizes),
            &rectangular_svd::nalgebra::<c32>(&input_sizes),
            &eigen(eigen::rectangular_svd_c32, &input_sizes),
        )?;
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
            &gemm::ndarray::<c64>(&input_sizes),
            &gemm::nalgebra::<c64>(&input_sizes),
            &eigen(eigen::gemm_c64, &input_sizes),
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
            &trsm::ndarray::<c64>(&input_sizes),
            &trsm::nalgebra::<c64>(&input_sizes),
            &eigen(eigen::trsm_c64, &input_sizes),
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
            &tr_inverse::ndarray::<c64>(&input_sizes),
            &tr_inverse::nalgebra::<c64>(&input_sizes),
            &eigen(eigen::trinv_c64, &input_sizes),
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
            &cholesky::ndarray::<c64>(&input_sizes),
            &cholesky::nalgebra::<c64>(&input_sizes),
            &eigen(eigen::chol_c64, &input_sizes),
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
            &partial_piv_lu::ndarray::<c64>(&input_sizes),
            &partial_piv_lu::nalgebra::<c64>(&input_sizes),
            &eigen(eigen::plu_c64, &input_sizes),
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
            &full_piv_lu::ndarray::<c64>(&input_sizes),
            &full_piv_lu::nalgebra::<c64>(&input_sizes),
            &eigen(eigen::flu_c64, &input_sizes),
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
            &no_piv_qr::ndarray::<c64>(&input_sizes),
            &no_piv_qr::nalgebra::<c64>(&input_sizes),
            &eigen(eigen::qr_c64, &input_sizes),
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
            &col_piv_qr::ndarray::<c64>(&input_sizes),
            &col_piv_qr::nalgebra::<c64>(&input_sizes),
            &eigen(eigen::colqr_c64, &input_sizes),
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
            &inverse::ndarray::<c64>(&input_sizes),
            &inverse::nalgebra::<c64>(&input_sizes),
            &eigen(eigen::inverse_c64, &input_sizes),
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
            &svd::ndarray::<c64>(&input_sizes),
            &svd::nalgebra::<c64>(&input_sizes),
            &eigen(eigen::svd_c64, &input_sizes),
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
            &rectangular_svd::ndarray::<c64>(&input_sizes),
            &rectangular_svd::nalgebra::<c64>(&input_sizes),
            &eigen(eigen::rectangular_svd_c64, &input_sizes),
        )?;
    }
    Ok(())
}
