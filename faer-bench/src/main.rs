use std::time::Duration;

use eyre::Result;
use faer_core::Parallelism;
use human_repr::HumanDuration;

extern crate blas_src;
extern crate openmp_sys;

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

mod col_piv_qr;
mod full_piv_lu;
mod inverse;
mod no_piv_qr;
mod partial_piv_lu;

mod cholesky;

fn print_results(
    input_sizes: &[usize],
    faer: &[Duration],
    faer_parallel: &[Duration],
    ndarray: &[Duration],
    nalgebra: &[Duration],
    eigen: &[Duration],
) {
    let fmt = |d: Duration| {
        if d == Duration::ZERO {
            "-".to_string()
        } else {
            format!("{}", d.human_duration())
        }
    };
    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "n", "faer", "faer(par)", "ndarray", "nalgebra", "eigen",
    );

    for (i, n) in input_sizes.iter().copied().enumerate() {
        println!(
            "{:5} {:>10} {:>10} {:>10} {:>10} {:>10}",
            n,
            fmt(faer[i]),
            fmt(faer_parallel[i]),
            fmt(ndarray[i]),
            fmt(nalgebra[i]),
            fmt(eigen[i]),
        );
    }
}

mod eigen {
    extern "C" {
        pub fn gemm(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trsm(out: *mut f64, inputs: *const usize, count: usize);
        pub fn trinv(out: *mut f64, inputs: *const usize, count: usize);
        pub fn chol(out: *mut f64, inputs: *const usize, count: usize);
        pub fn plu(out: *mut f64, inputs: *const usize, count: usize);
        pub fn flu(out: *mut f64, inputs: *const usize, count: usize);
        pub fn qr(out: *mut f64, inputs: *const usize, count: usize);
        pub fn colqr(out: *mut f64, inputs: *const usize, count: usize);
        pub fn inverse(out: *mut f64, inputs: *const usize, count: usize);
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

    println!(
        "
## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```"
    );
    print_results(
        &input_sizes,
        &gemm::faer(&input_sizes, Parallelism::None),
        &gemm::faer(&input_sizes, Parallelism::Rayon(0)),
        &gemm::ndarray(&input_sizes),
        &gemm::nalgebra(&input_sizes),
        &eigen(eigen::gemm, &input_sizes),
    );
    println!("```");

    println!("
## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```");
    print_results(
        &input_sizes,
        &trsm::faer(&input_sizes, Parallelism::None),
        &trsm::faer(&input_sizes, Parallelism::Rayon(0)),
        &trsm::ndarray(&input_sizes),
        &trsm::nalgebra(&input_sizes),
        &eigen(eigen::trsm, &input_sizes),
    );
    println!("```");

    println!(
        "
## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```"
    );
    print_results(
        &input_sizes,
        &tr_inverse::faer(&input_sizes, Parallelism::None),
        &tr_inverse::faer(&input_sizes, Parallelism::Rayon(0)),
        &tr_inverse::ndarray(&input_sizes),
        &tr_inverse::nalgebra(&input_sizes),
        &eigen(eigen::trinv, &input_sizes),
    );
    println!("```");

    println!(
        "
## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```"
    );
    print_results(
        &input_sizes,
        &cholesky::faer(&input_sizes, Parallelism::None),
        &cholesky::faer(&input_sizes, Parallelism::Rayon(0)),
        &cholesky::ndarray(&input_sizes),
        &cholesky::nalgebra(&input_sizes),
        &eigen(eigen::chol, &input_sizes),
    );
    println!("```");

    println!("
## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```");
    print_results(
        &input_sizes,
        &partial_piv_lu::faer(&input_sizes, Parallelism::None),
        &partial_piv_lu::faer(&input_sizes, Parallelism::Rayon(0)),
        &partial_piv_lu::ndarray(&input_sizes),
        &partial_piv_lu::nalgebra(&input_sizes),
        &eigen(eigen::plu, &input_sizes),
    );
    println!("```");

    println!("
## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```");
    print_results(
        &input_sizes,
        &full_piv_lu::faer(&input_sizes, Parallelism::None),
        &full_piv_lu::faer(&input_sizes, Parallelism::Rayon(0)),
        &full_piv_lu::ndarray(&input_sizes),
        &full_piv_lu::nalgebra(&input_sizes),
        &eigen(eigen::flu, &input_sizes),
    );
    println!("```");

    println!("
## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```");
    print_results(
        &input_sizes,
        &no_piv_qr::faer(&input_sizes, Parallelism::None),
        &no_piv_qr::faer(&input_sizes, Parallelism::Rayon(0)),
        &no_piv_qr::ndarray(&input_sizes),
        &no_piv_qr::nalgebra(&input_sizes),
        &eigen(eigen::qr, &input_sizes),
    );
    println!("```");

    println!("
## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```");
    print_results(
        &input_sizes,
        &col_piv_qr::faer(&input_sizes, Parallelism::None),
        &col_piv_qr::faer(&input_sizes, Parallelism::Rayon(0)),
        &col_piv_qr::ndarray(&input_sizes),
        &col_piv_qr::nalgebra(&input_sizes),
        &eigen(eigen::colqr, &input_sizes),
    );
    println!("```");

    println!(
        "
## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```"
    );
    print_results(
        &input_sizes,
        &inverse::faer(&input_sizes, Parallelism::None),
        &inverse::faer(&input_sizes, Parallelism::Rayon(0)),
        &inverse::ndarray(&input_sizes),
        &inverse::nalgebra(&input_sizes),
        &eigen(eigen::inverse, &input_sizes),
    );
    println!("```");

    Ok(())
}
