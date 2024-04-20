#![allow(non_snake_case)]

extern crate blas_src;

use diol::{
    config::{PlotMetric, PlotName},
    prelude::*,
};
use dyn_stack::{GlobalPodBuffer, PodStack};
use faer::{prelude::*, ComplexField};
use ndarray_linalg::Scalar;
use rand::{distributions::Distribution, prelude::*};
use rand_distr::Standard;

type C32 = num_complex::Complex32;
type C64 = num_complex::Complex64;

trait TypeDispatch: faer::ComplexField {
    type Type: faer::ComplexField + nalgebra::ComplexField + ndarray_linalg::Lapack;
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

#[derive(Copy, Clone, Debug)]
struct ApproxFlops;

impl diol::traits::PlotMetric for ApproxFlops {
    fn compute(&self, arg: PlotArg, time: diol::Picoseconds) -> f64 {
        (arg.0 as f64).powi(3) / (time.0 as f64 / 1e12)
    }
    fn name(&self) -> &'static str {
        "n³/s"
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

    pub fn cholesky_faer<E: ComplexField>(bencher: Bencher, n: usize, par: faer::Parallelism)
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

    pub fn cholesky_faer_seq<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        cholesky_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn cholesky_faer_par<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        cholesky_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn cholesky_faer_api<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        let H = &H * H.adjoint() + Mat::<E>::identity(n, n);
        bencher.bench(|| H.cholesky(faer::Side::Lower))
    }

    pub fn cholesky_nalgebra<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = &H * H.adjoint() + Mat::<E::Type>::identity(n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().cholesky().unwrap())
    }

    pub fn cholesky_ndarray<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = ndarray::Array2::from_shape_fn((n, n), |(i, j)| H.read(i, j));
        let H = &H.dot(&H.view().reversed_axes().mapv(|x| x.conj())) + ndarray::Array2::eye(n);
        bencher
            .bench(|| ndarray_linalg::Cholesky::cholesky(&H, ndarray_linalg::UPLO::Lower).unwrap())
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

    pub fn piv_qr_faer_seq<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        piv_qr_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn piv_qr_faer_par<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        piv_qr_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn piv_qr_faer_api<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        bencher.bench(|| H.col_piv_qr())
    }

    pub fn piv_qr_nalgebra<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().col_piv_qr())
    }

    pub fn piv_qr_ndarray<E: TypeDispatch>(_: Bencher, PlotArg(_): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
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

    pub fn qr_faer_seq<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        qr_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn qr_faer_par<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        qr_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn qr_faer_api<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        bencher.bench(|| H.qr())
    }

    pub fn qr_nalgebra<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().qr())
    }

    pub fn qr_ndarray<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = ndarray::Array2::from_shape_fn((n, n), |(i, j)| H.read(i, j));
        bencher.bench(|| ndarray_linalg::QR::qr(&H).unwrap())
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

    pub fn lu_faer_seq<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        lu_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn lu_faer_par<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        lu_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn lu_faer_api<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        bencher.bench(|| H.partial_piv_lu())
    }

    pub fn lu_nalgebra<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().lu())
    }

    pub fn lu_ndarray<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = ndarray::Array2::from_shape_fn((n, n), |(i, j)| H.read(i, j));
        bencher.bench(|| ndarray_linalg::Factorize::factorize(&H).unwrap())
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

    pub fn piv_lu_faer_seq<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        piv_lu_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn piv_lu_faer_par<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        piv_lu_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn piv_lu_faer_api<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        bencher.bench(|| H.full_piv_lu())
    }

    pub fn piv_lu_nalgebra<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().full_piv_lu())
    }

    pub fn piv_lu_ndarray<E: TypeDispatch>(_: Bencher, PlotArg(_): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
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

    pub fn svd_faer_seq<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        svd_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn svd_faer_par<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        svd_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn svd_faer_api<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E>(rng, n, n);
        bencher.bench(|| H.svd())
    }

    pub fn svd_nalgebra<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().svd(true, true))
    }

    pub fn svd_ndarray<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let H = random_mat::<E::Type>(rng, n, n);
        let H = ndarray::Array2::from_shape_fn((n, n), |(i, j)| H.read(i, j));
        bencher.bench(|| ndarray_linalg::SVDDC::svddc(&H, ndarray_linalg::JobSvd::All).unwrap())
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

    pub fn thin_svd_faer_seq<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        thin_svd_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn thin_svd_faer_par<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        thin_svd_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn thin_svd_faer_api<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let m = 4096;
        let A = random_mat::<E>(rng, m, n);
        bencher.bench(|| A.thin_svd())
    }

    pub fn thin_svd_nalgebra<E: TypeDispatch>(_: Bencher, PlotArg(_): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
    }

    pub fn thin_svd_ndarray<E: TypeDispatch>(_: Bencher, PlotArg(_): PlotArg)
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

    pub fn eig_faer_seq<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        evd_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn eig_faer_par<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        evd_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn eig_faer_api<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E>(rng, n, n);
        bencher.bench(|| A.eigendecomposition::<E::Cplx>())
    }

    pub fn eig_nalgebra<E: TypeDispatch>(_: Bencher, PlotArg(_): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
    }

    pub fn eig_ndarray<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E::Type>(rng, n, n);
        let H = ndarray::Array2::from_shape_fn((n, n), |(i, j)| A.read(i, j));
        bencher.bench(|| ndarray_linalg::Eig::eig(&H).unwrap())
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

    pub fn eigh_faer_seq<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        selfadjoint_evd_faer::<E>(bencher, n, faer::Parallelism::None)
    }

    pub fn eigh_faer_par<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: rand_distr::Distribution<E>,
    {
        selfadjoint_evd_faer::<E>(bencher, n, faer::Parallelism::Rayon(0))
    }

    pub fn eigh_faer_api<E: ComplexField>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E>(rng, n, n);
        let H = &A + A.adjoint();
        bencher.bench(|| H.selfadjoint_eigendecomposition(faer::Side::Lower))
    }

    pub fn eigh_nalgebra<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E::Type>(rng, n, n);
        let H = &A + A.adjoint();
        let H = nalgebra::DMatrix::from_fn(n, n, |i, j| H.read(i, j));
        bencher.bench(|| H.clone().symmetric_eigen())
    }

    pub fn eigh_ndarray<E: TypeDispatch>(bencher: Bencher, PlotArg(n): PlotArg)
    where
        Standard: Distribution<E::Type>,
    {
        let rng = &mut StdRng::seed_from_u64(0);
        let A = random_mat::<E::Type>(rng, n, n);
        let A = &A + A.adjoint();
        let H = ndarray::Array2::from_shape_fn((n, n), |(i, j)| A.read(i, j));
        bencher.bench(|| ndarray_linalg::Eigh::eigh(&H, ndarray_linalg::UPLO::Lower).unwrap())
    }
}

fn register_for<E: TypeDispatch>(bench: &mut Bench)
where
    Standard: Distribution<E> + Distribution<E::Type>,
{
    let args = [4, 8, 12, 16, 24, 32, 48, 64, 128, 256].map(PlotArg);

    bench.register_many(
        list![
            bench_cholesky::cholesky_faer_seq::<E>,
            bench_cholesky::cholesky_faer_par::<E>,
            bench_cholesky::cholesky_faer_api::<E>,
            bench_cholesky::cholesky_nalgebra::<E>,
            bench_cholesky::cholesky_ndarray::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_qr::qr_faer_seq::<E>,
            bench_qr::qr_faer_par::<E>,
            bench_qr::qr_faer_api::<E>,
            bench_qr::qr_nalgebra::<E>,
            bench_qr::qr_ndarray::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_col_qr::piv_qr_faer_seq::<E>,
            bench_col_qr::piv_qr_faer_par::<E>,
            bench_col_qr::piv_qr_faer_api::<E>,
            bench_col_qr::piv_qr_nalgebra::<E>,
            bench_col_qr::piv_qr_ndarray::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_lu::lu_faer_seq::<E>,
            bench_lu::lu_faer_par::<E>,
            bench_lu::lu_faer_api::<E>,
            bench_lu::lu_nalgebra::<E>,
            bench_lu::lu_ndarray::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_piv_lu::piv_lu_faer_seq::<E>,
            bench_piv_lu::piv_lu_faer_par::<E>,
            bench_piv_lu::piv_lu_faer_api::<E>,
            bench_piv_lu::piv_lu_nalgebra::<E>,
            bench_piv_lu::piv_lu_ndarray::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_svd::svd_faer_seq::<E>,
            bench_svd::svd_faer_par::<E>,
            bench_svd::svd_faer_api::<E>,
            bench_svd::svd_nalgebra::<E>,
            bench_svd::svd_ndarray::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_thin_svd::thin_svd_faer_seq::<E>,
            bench_thin_svd::thin_svd_faer_par::<E>,
            bench_thin_svd::thin_svd_faer_api::<E>,
            bench_thin_svd::thin_svd_nalgebra::<E>,
            bench_thin_svd::thin_svd_ndarray::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_selfadjoint_evd::eigh_faer_seq::<E>,
            bench_selfadjoint_evd::eigh_faer_par::<E>,
            bench_selfadjoint_evd::eigh_faer_api::<E>,
            bench_selfadjoint_evd::eigh_nalgebra::<E>,
            bench_selfadjoint_evd::eigh_ndarray::<E>,
        ],
        args,
    );

    bench.register_many(
        list![
            bench_evd::eig_faer_seq::<E>,
            bench_evd::eig_faer_par::<E>,
            bench_evd::eig_faer_api::<E>,
            bench_evd::eig_nalgebra::<E>,
            bench_evd::eig_ndarray::<E>,
        ],
        args,
    );
}

fn main() -> std::io::Result<()> {
    let config = BenchConfig {
        plot_metric: PlotMetric(Box::new(ApproxFlops)),
        ..BenchConfig::from_args()
    };
    let bench = &mut Bench::new(BenchConfig {
        output: Some("./target/diol.json".into()),
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
