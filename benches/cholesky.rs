#![allow(non_snake_case)]

use diol::prelude::*;
use dyn_stack::{GlobalPodBuffer, PodStack};
use faer::{prelude::*, ComplexField};
use rand::{distributions::Distribution, prelude::*};
use rand_distr::Standard;

type C32 = num_complex::Complex32;
type C64 = num_complex::Complex64;

trait TypeDispatch: faer::ComplexField {
    type Type: faer::ComplexField + nalgebra::ComplexField;
    type Cplx: faer::ComplexField<Real = Self::Real>;
}

impl TypeDispatch for f32 {
    type Type = f32;
    type Cplx = c32;
}
impl TypeDispatch for f64 {
    type Type = f64;
    type Cplx = c64;
}
impl TypeDispatch for c32 {
    type Type = C32;
    type Cplx = c32;
}
impl TypeDispatch for c64 {
    type Type = C64;
    type Cplx = c64;
}

fn random_mat<E: ComplexField>(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> faer::Mat<E>
where
    Standard: Distribution<E>,
{
    let normal = faer::stats::StandardMat { nrows, ncols };
    let mut sample = || -> faer::Mat<E> { normal.sample(rng) };
    sample()
}

fn faer<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
where
    Standard: Distribution<E>,
{
    let par = faer::Parallelism::None;

    let rng = &mut StdRng::seed_from_u64(0);
    let H = random_mat::<E>(rng, n, n);
    let H = &H * H.adjoint() + Mat::<E>::identity(n, n);
    let mut llt = H.clone();
    let mut mem = GlobalPodBuffer::new(
        faer::linalg::cholesky::llt::compute::cholesky_in_place_req::<E>(
            n,
            par,
            Default::default(),
        )
        .unwrap(),
    );
    bencher.bench(|| {
        llt.copy_from_triangular_lower(&H);
        faer::linalg::cholesky::llt::compute::cholesky_in_place(
            llt.as_mut(),
            Default::default(),
            par,
            PodStack::new(&mut mem),
            Default::default(),
        )
        .unwrap();
    })
}

fn nalgebra<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
where
    Standard: Distribution<E::Type>,
{
    let rng = &mut StdRng::seed_from_u64(0);
    let H = random_mat::<E::Type>(rng, n, n);
    let H = &H * H.adjoint() + Mat::<E::Type>::identity(n, n);
    let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
    bencher.bench(|| H.clone().cholesky().unwrap())
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![faer::<f32>, nalgebra::<f32>],
        [1, 2, 4, 8, 16, 32, 64, 128, 256].map(PlotArg),
    );
    bench.register_many(
        list![faer::<f64>, nalgebra::<f64>],
        [1, 2, 4, 8, 16, 32, 64, 128, 256].map(PlotArg),
    );
    bench.run()?;

    Ok(())
}
