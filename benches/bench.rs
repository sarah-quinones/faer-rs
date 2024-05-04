#![allow(non_snake_case)]

use diol::{
    config::{PlotMetric, PlotName},
    prelude::*,
};
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

const NALGEBRA_LIMIT: usize = 2048;

#[derive(Copy, Clone, Debug)]
struct ApproxFlops;

impl diol::traits::PlotMetric for ApproxFlops {
    fn compute(&self, arg: PlotArg, time: diol::Picoseconds) -> f64 {
        (arg.0 as f64).powi(3) / (time.0 as f64 / 1e12)
    }
    fn name(&self) -> &'static str {
        "n³/s"
    }

    fn monotonicity(&self) -> diol::traits::Monotonicity {
        diol::traits::Monotonicity::HigherIsBetter
    }
}

fn random_mat<E: ComplexField>(rng: &mut dyn RngCore, nrows: usize, ncols: usize) -> faer::Mat<E>
where
    Standard: Distribution<E>,
{
    let normal = faer::stats::StandardMat { nrows, ncols };
    let mut sample = || -> faer::Mat<E> { normal.sample(rng) };
    sample()
}

mod bench_cholesky {
    use super::*;

    pub fn faer_cholesky<E: ComplexField>(bencher: Bencher, n: usize, par: faer::Parallelism)
    where
        Standard: Distribution<E>,
    {
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

    pub fn faer_seq_cholesky<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        faer_cholesky::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn faer_par_cholesky<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        faer_cholesky::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn faer_api_cholesky<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        let H = &H * H.adjoint() + Mat::<E>::identity(n, n);
        bencher.bench(|| H.cholesky(faer::Side::Lower))
    }

    pub fn nalgebra_cholesky<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        if n > NALGEBRA_LIMIT {
            return bencher.skip();
        }
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = &H * H.adjoint() + Mat::<E::Type>::identity(n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().cholesky().unwrap())
    }
}

mod bench_col_qr {
    use super::*;

    pub fn piv_qr_faer<E: ComplexField>(bencher: Bencher, n: usize, par: faer::Parallelism)
    where
        Standard: Distribution<E>,
    {
        let blocksize = faer::linalg::qr::col_pivoting::compute::recommended_blocksize::<E>(n, n);

        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        let mut qr = H.clone();
        let mut householder = Mat::<E>::zeros(blocksize, n);
        let mut mem = GlobalPodBuffer::new(
            faer::linalg::qr::col_pivoting::compute::qr_in_place_req::<usize, E>(
                n,
                n,
                blocksize,
                par,
                Default::default(),
            )
            .unwrap(),
        );
        let col_perm = &mut *vec![0usize; n];
        let col_perm_inv = &mut *vec![0usize; n];
        bencher.bench(|| {
            qr.copy_from(&H);
            faer::linalg::qr::col_pivoting::compute::qr_in_place(
                qr.as_mut(),
                householder.as_mut(),
                col_perm,
                col_perm_inv,
                par,
                PodStack::new(&mut mem),
                Default::default(),
            );
        })
    }

    pub fn faer_seq_piv_qr<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        piv_qr_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn faer_par_piv_qr<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        piv_qr_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn faer_api_piv_qr<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        bencher.bench(|| H.col_piv_qr())
    }

    pub fn nalgebra_piv_qr<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        if n > NALGEBRA_LIMIT {
            return bencher.skip();
        }
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().col_piv_qr())
    }
}
mod bench_qr {
    use super::*;

    pub fn qr_faer<E: ComplexField>(bencher: Bencher, n: usize, par: faer::Parallelism)
    where
        Standard: Distribution<E>,
    {
        let blocksize = faer::linalg::qr::no_pivoting::compute::recommended_blocksize::<E>(n, n);

        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        let mut qr = H.clone();
        let mut householder = Mat::<E>::zeros(blocksize, n);
        let mut mem = GlobalPodBuffer::new(
            faer::linalg::qr::no_pivoting::compute::qr_in_place_req::<E>(
                n,
                n,
                blocksize,
                par,
                Default::default(),
            )
            .unwrap(),
        );
        bencher.bench(|| {
            qr.copy_from(&H);
            faer::linalg::qr::no_pivoting::compute::qr_in_place(
                qr.as_mut(),
                householder.as_mut(),
                par,
                PodStack::new(&mut mem),
                Default::default(),
            );
        })
    }

    pub fn faer_seq_qr<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        qr_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn faer_par_qr<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        qr_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn faer_api_qr<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        bencher.bench(|| H.qr())
    }

    pub fn nalgebra_qr<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        if n > NALGEBRA_LIMIT {
            return bencher.skip();
        }
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().qr())
    }
}

mod bench_lu {
    use super::*;

    pub fn lu_faer<E: ComplexField>(bencher: Bencher, n: usize, par: faer::Parallelism)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E>(rng, n, n);
        let mut lu = A.clone();
        let mut mem = GlobalPodBuffer::new(
            faer::linalg::lu::partial_pivoting::compute::lu_in_place_req::<usize, E>(
                n,
                n,
                par,
                Default::default(),
            )
            .unwrap(),
        );
        let perm = &mut *vec![0usize; n];
        let perm_inv = &mut *vec![0usize; n];
        bencher.bench(|| {
            lu.copy_from(&A);
            faer::linalg::lu::partial_pivoting::compute::lu_in_place(
                lu.as_mut(),
                perm,
                perm_inv,
                par,
                PodStack::new(&mut mem),
                Default::default(),
            );
        })
    }

    pub fn faer_seq_lu<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        lu_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn faer_par_lu<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        lu_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn faer_api_lu<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        bencher.bench(|| H.partial_piv_lu())
    }

    pub fn nalgebra_lu<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        if n > NALGEBRA_LIMIT {
            return bencher.skip();
        }
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().lu())
    }
}
mod bench_piv_lu {
    use super::*;

    pub fn piv_lu_faer<E: ComplexField>(bencher: Bencher, n: usize, par: faer::Parallelism)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E>(rng, n, n);
        let mut lu = A.clone();
        let mut mem = GlobalPodBuffer::new(
            faer::linalg::lu::full_pivoting::compute::lu_in_place_req::<usize, E>(
                n,
                n,
                par,
                Default::default(),
            )
            .unwrap(),
        );
        let row_perm = &mut *vec![0usize; n];
        let row_perm_inv = &mut *vec![0usize; n];
        let col_perm = &mut *vec![0usize; n];
        let col_perm_inv = &mut *vec![0usize; n];
        bencher.bench(|| {
            lu.copy_from(&A);
            faer::linalg::lu::full_pivoting::compute::lu_in_place(
                lu.as_mut(),
                row_perm,
                row_perm_inv,
                col_perm,
                col_perm_inv,
                par,
                PodStack::new(&mut mem),
                Default::default(),
            );
        })
    }

    pub fn faer_seq_piv_lu<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        piv_lu_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn faer_par_piv_lu<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        piv_lu_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn faer_api_piv_lu<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        bencher.bench(|| H.full_piv_lu())
    }

    pub fn nalgebra_piv_lu<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        if n > NALGEBRA_LIMIT {
            return bencher.skip();
        }
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().full_piv_lu())
    }
}

mod bench_svd {
    use super::*;

    pub fn svd_faer<E: ComplexField>(bencher: Bencher, n: usize, par: faer::Parallelism)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E>(rng, n, n);
        let mut S = faer::Col::zeros(n);
        let mut U = faer::Mat::zeros(n, n);
        let mut V = faer::Mat::zeros(n, n);
        let mut mem = GlobalPodBuffer::new(
            faer::linalg::svd::compute_svd_req::<E>(
                n,
                n,
                faer::linalg::svd::ComputeVectors::Full,
                faer::linalg::svd::ComputeVectors::Full,
                par,
                Default::default(),
            )
            .unwrap(),
        );
        bencher.bench(|| {
            faer::linalg::svd::compute_svd(
                A.as_ref(),
                S.as_mut().as_2d_mut(),
                Some(U.as_mut()),
                Some(V.as_mut()),
                par,
                PodStack::new(&mut mem),
                Default::default(),
            );
        })
    }

    pub fn faer_seq_svd<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        svd_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn faer_par_svd<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        svd_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn faer_api_svd<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        bencher.bench(|| H.svd())
    }

    pub fn nalgebra_svd<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        if n > NALGEBRA_LIMIT {
            return bencher.skip();
        }
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().svd(true, true))
    }
}
mod bench_thin_svd {
    use super::*;

    pub fn thin_svd_faer<E: ComplexField>(bencher: Bencher, n: usize, par: faer::Parallelism)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let m = 4096;
        let A = random_mat::<E>(rng, m, n);
        let mut S = faer::Col::zeros(Ord::min(m, n));
        let mut U = faer::Mat::zeros(m, Ord::min(m, n));
        let mut V = faer::Mat::zeros(n, Ord::min(m, n));
        let mut mem = GlobalPodBuffer::new(
            faer::linalg::svd::compute_svd_req::<E>(
                m,
                n,
                faer::linalg::svd::ComputeVectors::Full,
                faer::linalg::svd::ComputeVectors::Full,
                par,
                Default::default(),
            )
            .unwrap(),
        );
        bencher.bench(|| {
            faer::linalg::svd::compute_svd(
                A.as_ref(),
                S.as_mut().as_2d_mut(),
                Some(U.as_mut()),
                Some(V.as_mut()),
                par,
                PodStack::new(&mut mem),
                Default::default(),
            );
        })
    }

    pub fn faer_seq_thin_svd<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        thin_svd_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn faer_par_thin_svd<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        thin_svd_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn faer_api_thin_svd<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let m = 4096;
        let A = random_mat::<E>(rng, m, n);
        bencher.bench(|| A.thin_svd())
    }

    pub fn nalgebra_thin_svd<E: TypeDispatch>(_: Bencher, PlotArg(_): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
    }
}

mod bench_evd {
    use coe::Coerce;

    use super::*;

    pub fn evd_faer<E: ComplexField>(bencher: Bencher, n: usize, par: faer::Parallelism)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E>(rng, n, n);
        let mut S_re = faer::Col::zeros(n);
        let mut S_im = faer::Col::<E>::zeros(n);
        let mut U = faer::Mat::zeros(n, n);
        let mut mem = GlobalPodBuffer::new(
            faer::linalg::evd::compute_evd_req::<E>(
                n,
                faer::linalg::evd::ComputeVectors::Yes,
                par,
                Default::default(),
            )
            .unwrap(),
        );
        bencher.bench(|| {
            if coe::is_same::<E, E::Real>() {
                faer::linalg::evd::compute_evd_real::<E::Real>(
                    A.as_ref().coerce(),
                    S_re.as_mut().as_2d_mut().coerce(),
                    S_im.as_mut().as_2d_mut().coerce(),
                    Some(U.as_mut().coerce()),
                    par,
                    PodStack::new(&mut mem),
                    Default::default(),
                );
            } else {
                faer::linalg::evd::compute_evd_complex(
                    A.as_ref(),
                    S_re.as_mut().as_2d_mut(),
                    Some(U.as_mut()),
                    par,
                    PodStack::new(&mut mem),
                    Default::default(),
                );
            }
        })
    }

    pub fn faer_seq_eig<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        evd_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn faer_par_eig<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        evd_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn faer_api_eig<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E>(rng, n, n);
        bencher.bench(|| A.eigendecomposition::<E::Cplx>())
    }

    pub fn nalgebra_eig<E: TypeDispatch>(_: Bencher, PlotArg(_): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
    }
}
mod bench_selfadjoint_evd {
    use super::*;

    pub fn selfadjoint_evd_faer<E: ComplexField>(bencher: Bencher, n: usize, par: faer::Parallelism)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E>(rng, n, n);
        let H = &A + A.adjoint();
        let mut S = faer::Col::zeros(n);
        let mut U = faer::Mat::zeros(n, n);
        let mut mem = GlobalPodBuffer::new(
            faer::linalg::evd::compute_hermitian_evd_req::<E>(
                n,
                faer::linalg::evd::ComputeVectors::Yes,
                par,
                Default::default(),
            )
            .unwrap(),
        );
        bencher.bench(|| {
            faer::linalg::evd::compute_hermitian_evd(
                H.as_ref(),
                S.as_mut().as_2d_mut(),
                Some(U.as_mut()),
                par,
                PodStack::new(&mut mem),
                Default::default(),
            );
        })
    }

    pub fn faer_seq_eigh<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        selfadjoint_evd_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn faer_par_eigh<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        selfadjoint_evd_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn faer_api_eigh<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E>(rng, n, n);
        let H = &A + A.adjoint();
        bencher.bench(|| H.selfadjoint_eigendecomposition(faer::Side::Lower))
    }

    pub fn nalgebra_eigh<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        if n > NALGEBRA_LIMIT {
            return bencher.skip();
        }
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E::Type>(rng, n, n);
        let H = &A + A.adjoint();
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().symmetric_eigen())
    }
}

fn register_for<E: TypeDispatch>(bench: &mut Bench)
where
    Standard: Distribution<E> + Distribution<E::Type>,
{
    let args = [
        4, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096,
    ]
    .map(PlotArg);

    bench.register_many(
        list![
            bench_cholesky::faer_seq_cholesky::<E>,
            bench_cholesky::faer_par_cholesky::<E>,
            bench_cholesky::faer_api_cholesky::<E>,
            bench_cholesky::nalgebra_cholesky::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_qr::faer_seq_qr::<E>,
            bench_qr::faer_par_qr::<E>,
            bench_qr::faer_api_qr::<E>,
            bench_qr::nalgebra_qr::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_col_qr::faer_seq_piv_qr::<E>,
            bench_col_qr::faer_par_piv_qr::<E>,
            bench_col_qr::faer_api_piv_qr::<E>,
            bench_col_qr::nalgebra_piv_qr::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_lu::faer_seq_lu::<E>,
            bench_lu::faer_par_lu::<E>,
            bench_lu::faer_api_lu::<E>,
            bench_lu::nalgebra_lu::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_piv_lu::faer_seq_piv_lu::<E>,
            bench_piv_lu::faer_par_piv_lu::<E>,
            bench_piv_lu::faer_api_piv_lu::<E>,
            bench_piv_lu::nalgebra_piv_lu::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_svd::faer_seq_svd::<E>,
            bench_svd::faer_par_svd::<E>,
            bench_svd::faer_api_svd::<E>,
            bench_svd::nalgebra_svd::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_thin_svd::faer_seq_thin_svd::<E>,
            bench_thin_svd::faer_par_thin_svd::<E>,
            bench_thin_svd::faer_api_thin_svd::<E>,
            bench_thin_svd::nalgebra_thin_svd::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_selfadjoint_evd::faer_seq_eigh::<E>,
            bench_selfadjoint_evd::faer_par_eigh::<E>,
            bench_selfadjoint_evd::faer_api_eigh::<E>,
            bench_selfadjoint_evd::nalgebra_eigh::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_evd::faer_seq_eig::<E>,
            bench_evd::faer_par_eig::<E>,
            bench_evd::faer_api_eig::<E>,
            bench_evd::nalgebra_eig::<E>,
        ],
        args,
    );
}

fn main() -> std::io::Result<()> {
    let config = BenchConfig {
        plot_metric: PlotMetric(Box::new(ApproxFlops)),
        ..BenchConfig::from_args()?
    };
    let bench = &mut Bench::new(BenchConfig {
        plot_name: PlotName("plot".into()),
        ..config.clone()
    });

    register_for::<f32>(bench);
    register_for::<f64>(bench);
    register_for::<c32>(bench);
    register_for::<c64>(bench);

    bench.run()?;

    Ok(())
}
