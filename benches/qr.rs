use diol::prelude::*;
use dyn_stack::{GlobalPodBuffer, PodStack};
use rand::prelude::*;
use reborrow::*;

fn faer<E: faer::ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
where
    rand_distr::StandardNormal: rand::distributions::Distribution<E>,
{
    let rng = &mut StdRng::seed_from_u64(0);
    let mat: faer::Mat<E> = faer::stats::StandardNormalMat { nrows: n, ncols: n }.sample(rng);
    let bs = faer::linalg::qr::no_pivoting::compute::recommended_blocksize::<E>(n, n);
    let mut copy = mat.clone();
    let mut householder = faer::Mat::zeros(bs, n);
    let mut mem = GlobalPodBuffer::new(
        faer::linalg::qr::no_pivoting::compute::qr_in_place_req::<E>(
            n,
            n,
            bs,
            faer::Parallelism::None,
            Default::default(),
        )
        .unwrap(),
    );
    let mut stack = PodStack::new(&mut mem);

    bencher.bench(|| {
        copy.copy_from(&mat);
        faer::linalg::qr::no_pivoting::compute::qr_in_place(
            copy.as_mut(),
            householder.as_mut(),
            faer::Parallelism::None,
            stack.rb_mut(),
            Default::default(),
        );
    })
}

fn nalgebra<E: faer::ComplexField + nalgebra::ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
where
    rand_distr::StandardNormal: rand::distributions::Distribution<E>,
{
    let rng = &mut StdRng::seed_from_u64(0);
    let mat: faer::Mat<E> = faer::stats::StandardNormalMat { nrows: n, ncols: n }.sample(rng);
    let mat = nalgebra::DMatrix::from_fn(n, n, |i, j| mat.read(i, j));

    bencher.bench(|| {
        mat.clone().qr();
    })
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    let args = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128].map(PlotArg);
    bench.register_many(list![faer::<f32>, nalgebra::<f32>], args);
    bench.register_many(list![faer::<f64>, nalgebra::<f64>], args);
    bench.run()?;

    Ok(())
}
