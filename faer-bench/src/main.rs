use std::time::Duration;

use eyre::Result;
use faer_core::Parallelism;
use human_repr::HumanDuration;

extern crate blas_src;

fn time(mut f: impl FnMut()) -> f64 {
    let instant = std::time::Instant::now();
    f();
    instant.elapsed().as_secs_f64()
}

fn timeit(f: impl FnMut()) -> f64 {
    let mut f = f;
    let min = 1e-2;
    let once = time(&mut f);
    if once > min {
        once
    } else {
        let n = (min / once).ceil() as u64;
        time(|| {
            for _ in 0..n {
                f()
            }
        }) / n as f64
    }
}

mod gemm;
mod tr_inverse;
mod trsm;

mod full_piv_lu;
mod inverse;
mod partial_piv_lu;

fn print_results(
    input_sizes: &[usize],
    faer: &[Duration],
    faer_parallel: &[Duration],
    ndarray: &[Duration],
    nalgebra: &[Duration],
) {
    let fmt = |d: Duration| format!("{}", d.human_duration());
    println!(
        "{:5} {:>15} {:>20} {:>20} {:>30}",
        "", "faer (serial)", "faer (parallel)", "ndarray (openblas)", "nalgebra (matrixmultiply)",
    );

    for (i, n) in input_sizes.iter().copied().enumerate() {
        println!(
            "{:5} {:>15} {:>20} {:>20} {:>30}",
            n,
            fmt(faer[i]),
            fmt(faer_parallel[i]),
            fmt(ndarray[i]),
            fmt(nalgebra[i]),
        );
    }
}

fn main() -> Result<()> {
    let input_sizes = vec![32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 896, 1024];

    println!("gemm");
    print_results(
        &input_sizes,
        &gemm::faer(&input_sizes, Parallelism::None),
        &gemm::faer(&input_sizes, Parallelism::Rayon(0)),
        &gemm::ndarray(&input_sizes),
        &gemm::nalgebra(&input_sizes),
    );
    println!("trsm");
    print_results(
        &input_sizes,
        &trsm::faer(&input_sizes, Parallelism::None),
        &trsm::faer(&input_sizes, Parallelism::Rayon(0)),
        &trsm::ndarray(&input_sizes),
        &trsm::nalgebra(&input_sizes),
    );

    println!("triangular inverse");
    print_results(
        &input_sizes,
        &tr_inverse::faer(&input_sizes, Parallelism::None),
        &tr_inverse::faer(&input_sizes, Parallelism::Rayon(0)),
        &tr_inverse::ndarray(&input_sizes),
        &tr_inverse::nalgebra(&input_sizes),
    );

    println!("lu decomposition with partial pivoting");
    print_results(
        &input_sizes,
        &partial_piv_lu::faer(&input_sizes, Parallelism::None),
        &partial_piv_lu::faer(&input_sizes, Parallelism::Rayon(0)),
        &partial_piv_lu::ndarray(&input_sizes),
        &partial_piv_lu::nalgebra(&input_sizes),
    );

    println!("lu decomposition with full pivoting");
    print_results(
        &input_sizes,
        &full_piv_lu::faer(&input_sizes, Parallelism::None),
        &full_piv_lu::faer(&input_sizes, Parallelism::Rayon(0)),
        &full_piv_lu::ndarray(&input_sizes),
        &full_piv_lu::nalgebra(&input_sizes),
    );

    println!("matrix inverse");
    print_results(
        &input_sizes,
        &inverse::faer(&input_sizes, Parallelism::None),
        &inverse::faer(&input_sizes, Parallelism::Rayon(0)),
        &inverse::ndarray(&input_sizes),
        &inverse::nalgebra(&input_sizes),
    );

    Ok(())
}
