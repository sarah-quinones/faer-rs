use std::time::Duration;

use eyre::Result;
use faer_core::Parallelism;
use human_repr::HumanDuration;

extern crate blas_src;

mod gemm;
mod trsm;

fn main() -> Result<()> {
    let input_sizes = vec![32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 896, 1024];

    let fmt = |d: Duration| format!("{}", d.human_duration());
    {
        let faer = gemm::faer(&input_sizes, Parallelism::None);
        let faer_parallel = gemm::faer(&input_sizes, Parallelism::Rayon(0));
        let ndarray = gemm::ndarray(&input_sizes);
        let nalgebra = gemm::nalgebra(&input_sizes);

        println!("gemm");
        println!(
            "{:5} {:>15} {:>20} {:>20} {:>30}",
            "",
            "faer (serial)",
            "faer (parallel)",
            "ndarray (openblas)",
            "nalgebra (matrixmultiply)",
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
    {
        let faer = trsm::faer(&input_sizes, Parallelism::None);
        let faer_parallel = trsm::faer(&input_sizes, Parallelism::Rayon(0));
        let ndarray = trsm::ndarray(&input_sizes);
        let nalgebra = trsm::nalgebra(&input_sizes);

        println!("trsm");
        println!(
            "{:5} {:>15} {:>20} {:>20} {:>30}",
            "",
            "faer (serial)",
            "faer (parallel)",
            "ndarray (openblas)",
            "nalgebra (matrixmultiply)",
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

    Ok(())
}
