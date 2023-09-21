use criterion::*;
use dyn_stack::{DynStack, GlobalMemBuffer};
use faer_core::{c32, c64, zipped, ComplexField, Mat, Parallelism, RealField};
use faer_evd::{
    tridiag::{tridiagonalize_in_place, tridiagonalize_in_place_req},
    tridiag_real_evd::{compute_tridiag_real_evd, compute_tridiag_real_evd_req},
};
use reborrow::*;
use std::any::type_name;

fn random<E: 'static>() -> E {
    if coe::is_same::<f32, E>() {
        coe::coerce_static(rand::random::<f32>())
    } else if coe::is_same::<f64, E>() {
        coe::coerce_static(rand::random::<f64>())
    } else if coe::is_same::<c32, E>() {
        coe::coerce_static(c32::new(rand::random(), rand::random()))
    } else if coe::is_same::<c64, E>() {
        coe::coerce_static(c64::new(rand::random(), rand::random()))
    } else if coe::is_same::<num_complex::Complex32, E>() {
        coe::coerce_static(num_complex::Complex32::new(rand::random(), rand::random()))
    } else if coe::is_same::<num_complex::Complex64, E>() {
        coe::coerce_static(num_complex::Complex64::new(rand::random(), rand::random()))
    } else {
        panic!()
    }
}

fn epsilon<E: 'static>() -> E {
    if coe::is_same::<f32, E>() {
        coe::coerce_static(f32::EPSILON)
    } else if coe::is_same::<f64, E>() {
        coe::coerce_static(f64::EPSILON)
    } else {
        panic!()
    }
}

fn min_positive<E: 'static>() -> E {
    if coe::is_same::<f32, E>() {
        coe::coerce_static(f32::MIN_POSITIVE)
    } else if coe::is_same::<f64, E>() {
        coe::coerce_static(f64::MIN_POSITIVE)
    } else {
        panic!()
    }
}

fn tridiagonalization<E: ComplexField>(criterion: &mut Criterion) {
    for n in [32, 64, 128, 256, 512, 1024] {
        let mut mat = Mat::from_fn(n, n, |_, _| random::<E>());
        let adjoint = mat.adjoint().to_owned();

        zipped!(mat.as_mut(), adjoint.as_ref())
            .for_each(|mut x, y| x.write(x.read().add(&y.read())));

        let mut trid = mat.clone();
        let mut tau_left = Mat::zeros(n - 1, 1);

        {
            let parallelism = Parallelism::None;
            let mut mem =
                GlobalMemBuffer::new(tridiagonalize_in_place_req::<E>(n, parallelism).unwrap());
            let mut stack = DynStack::new(&mut mem);

            criterion.bench_function(
                &format!("tridiag-st-{}-{}", type_name::<E>(), n),
                |bencher| {
                    bencher.iter(|| {
                        zipped!(trid.as_mut(), mat.as_ref())
                            .for_each(|mut dst, src| dst.write(src.read()));
                        tridiagonalize_in_place(
                            trid.as_mut(),
                            tau_left.as_mut().col(0),
                            parallelism,
                            stack.rb_mut(),
                        );
                    });
                },
            );
        }
        {
            let parallelism = Parallelism::Rayon(0);
            let mut mem =
                GlobalMemBuffer::new(tridiagonalize_in_place_req::<E>(n, parallelism).unwrap());
            let mut stack = DynStack::new(&mut mem);

            criterion.bench_function(
                &format!("tridiag-mt-{}-{}", type_name::<E>(), n),
                |bencher| {
                    bencher.iter(|| {
                        zipped!(trid.as_mut(), mat.as_ref())
                            .for_each(|mut dst, src| dst.write(src.read()));
                        tridiagonalize_in_place(
                            trid.as_mut(),
                            tau_left.as_mut().col(0),
                            parallelism,
                            stack.rb_mut(),
                        );
                    });
                },
            );
        }
    }
}

fn tridiagonal_evd<E: RealField>(criterion: &mut Criterion) {
    for n in [32, 64, 128, 256, 512, 1024, 4096] {
        let diag = (0..n).map(|_| random::<E>()).collect::<Vec<_>>();
        let offdiag = (0..n - 1).map(|_| random::<E>()).collect::<Vec<_>>();
        let mut u = Mat::<E>::zeros(n, n);

        let parallelism = Parallelism::None;
        let mut mem =
            GlobalMemBuffer::new(compute_tridiag_real_evd_req::<E>(n, parallelism).unwrap());
        let mut stack = DynStack::new(&mut mem);

        criterion.bench_function(
            &format!("tridiag-evd-st-{}-{}", type_name::<E>(), n),
            |bencher| {
                bencher.iter(|| {
                    let mut diag = diag.clone();
                    let mut offdiag = offdiag.clone();
                    compute_tridiag_real_evd(
                        &mut diag,
                        &mut offdiag,
                        u.as_mut(),
                        epsilon(),
                        min_positive(),
                        parallelism,
                        stack.rb_mut(),
                    );
                });
            },
        );
        let parallelism = Parallelism::Rayon(0);
        let mut mem =
            GlobalMemBuffer::new(compute_tridiag_real_evd_req::<E>(n, parallelism).unwrap());
        let mut stack = DynStack::new(&mut mem);
        criterion.bench_function(
            &format!("tridiag-evd-mt-{}-{}", type_name::<E>(), n),
            |bencher| {
                bencher.iter(|| {
                    let mut diag = diag.clone();
                    let mut offdiag = offdiag.clone();
                    compute_tridiag_real_evd(
                        &mut diag,
                        &mut offdiag,
                        u.as_mut(),
                        epsilon(),
                        min_positive(),
                        parallelism,
                        stack.rb_mut(),
                    );
                });
            },
        );
    }
}

fn evd<E: ComplexField>(criterion: &mut Criterion) {
    for n in [4, 6, 8, 10, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 4096] {
        let mut mat = Mat::from_fn(n, n, |_, _| random::<E>());
        let adjoint = mat.adjoint().to_owned();

        zipped!(mat.as_mut(), adjoint.as_ref())
            .for_each(|mut x, y| x.write(x.read().add(&y.read())));

        let mut s = Mat::zeros(n, n);
        let mut u = Mat::zeros(n, n);

        {
            let parallelism = Parallelism::None;
            let mut mem = GlobalMemBuffer::new(
                faer_evd::compute_hermitian_evd_req::<E>(
                    n,
                    faer_evd::ComputeVectors::Yes,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            criterion.bench_function(
                &format!("sym-evd-st-{}-{}", type_name::<E>(), n),
                |bencher| {
                    bencher.iter(|| {
                        faer_evd::compute_hermitian_evd(
                            mat.as_ref(),
                            s.as_mut().diagonal(),
                            Some(u.as_mut()),
                            parallelism,
                            stack.rb_mut(),
                            Default::default(),
                        );
                    });
                },
            );
        }
        {
            let parallelism = Parallelism::Rayon(0);
            let mut mem = GlobalMemBuffer::new(
                faer_evd::compute_hermitian_evd_req::<E>(
                    n,
                    faer_evd::ComputeVectors::Yes,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            criterion.bench_function(
                &format!("sym-evd-mt-{}-{}", type_name::<E>(), n),
                |bencher| {
                    bencher.iter(|| {
                        faer_evd::compute_hermitian_evd(
                            mat.as_ref(),
                            s.as_mut().diagonal(),
                            Some(u.as_mut()),
                            parallelism,
                            stack.rb_mut(),
                            Default::default(),
                        );
                    });
                },
            );
        }
    }
}

fn evd_nalgebra(criterion: &mut Criterion) {
    for n in [4, 6, 8, 10, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 4096] {
        let mat = nalgebra::DMatrix::<f64>::from_fn(n, n, |_, _| random::<f64>());
        criterion.bench_function(
            &format!("sym-evd-nalgebra-{}-{}", type_name::<f64>(), n),
            |bencher| {
                bencher.iter(|| {
                    mat.clone().symmetric_eigen();
                });
            },
        );
    }
}

fn cplx_schur<E: ComplexField>(criterion: &mut Criterion) {
    for n in [32, 64, 128, 256, 512, 1024, 4096] {
        let mat = Mat::from_fn(n, n, |_, _| random::<E>());
        let mut t = mat.clone();
        let mut z = mat.clone();
        let mut w = Mat::zeros(n, 1);

        {
            let parallelism = Parallelism::None;
            let mut mem = GlobalMemBuffer::new(
                faer_evd::hessenberg_cplx_evd::multishift_qr_req::<E>(
                    n,
                    n,
                    true,
                    true,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            criterion.bench_function(&format!("schur-st-{}-{}", type_name::<E>(), n), |bencher| {
                bencher.iter(|| {
                    zipped!(t.as_mut(), mat.as_ref())
                        .for_each(|mut dst, src| dst.write(src.read()));
                    zipped!(z.as_mut()).for_each(|mut x| x.write(E::zero()));
                    zipped!(z.as_mut().diagonal()).for_each(|mut x| x.write(E::one()));

                    faer_evd::hessenberg_cplx_evd::multishift_qr(
                        true,
                        t.as_mut(),
                        Some(z.as_mut()),
                        w.as_mut(),
                        0,
                        n,
                        epsilon(),
                        min_positive(),
                        parallelism,
                        stack.rb_mut(),
                        Default::default(),
                    );
                });
            });
        }
        {
            let parallelism = Parallelism::Rayon(0);
            let mut mem = GlobalMemBuffer::new(
                faer_evd::hessenberg_cplx_evd::multishift_qr_req::<E>(
                    n,
                    n,
                    true,
                    true,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            criterion.bench_function(&format!("schur-mt-{}-{}", type_name::<E>(), n), |bencher| {
                bencher.iter(|| {
                    zipped!(t.as_mut(), mat.as_ref())
                        .for_each(|mut dst, src| dst.write(src.read()));
                    zipped!(z.as_mut()).for_each(|mut x| x.write(E::zero()));
                    zipped!(z.as_mut().diagonal()).for_each(|mut x| x.write(E::one()));

                    faer_evd::hessenberg_cplx_evd::multishift_qr(
                        true,
                        t.as_mut(),
                        Some(z.as_mut()),
                        w.as_mut(),
                        0,
                        n,
                        epsilon(),
                        min_positive(),
                        parallelism,
                        stack.rb_mut(),
                        Default::default(),
                    );
                });
            });
        }
    }
}

fn real_schur<E: RealField>(criterion: &mut Criterion) {
    for n in [32, 64, 128, 256, 512, 1024, 4096] {
        let mat = Mat::from_fn(n, n, |_, _| random::<E>());
        let mut t = mat.clone();
        let mut z = mat.clone();
        let mut w_re = Mat::zeros(n, 1);
        let mut w_im = Mat::zeros(n, 1);

        {
            let parallelism = Parallelism::None;
            let mut mem = GlobalMemBuffer::new(
                faer_evd::hessenberg_real_evd::multishift_qr_req::<E>(
                    n,
                    n,
                    true,
                    true,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            criterion.bench_function(&format!("schur-st-{}-{}", type_name::<E>(), n), |bencher| {
                bencher.iter(|| {
                    zipped!(t.as_mut(), mat.as_ref())
                        .for_each(|mut dst, src| dst.write(src.read()));
                    zipped!(z.as_mut()).for_each(|mut x| x.write(E::zero()));
                    zipped!(z.as_mut().diagonal()).for_each(|mut x| x.write(E::one()));

                    faer_evd::hessenberg_real_evd::multishift_qr(
                        true,
                        t.as_mut(),
                        Some(z.as_mut()),
                        w_re.as_mut(),
                        w_im.as_mut(),
                        0,
                        n,
                        epsilon(),
                        min_positive(),
                        parallelism,
                        stack.rb_mut(),
                        Default::default(),
                    );
                });
            });
        }
        {
            let parallelism = Parallelism::Rayon(0);
            let mut mem = GlobalMemBuffer::new(
                faer_evd::hessenberg_real_evd::multishift_qr_req::<E>(
                    n,
                    n,
                    true,
                    true,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            criterion.bench_function(&format!("schur-mt-{}-{}", type_name::<E>(), n), |bencher| {
                bencher.iter(|| {
                    zipped!(t.as_mut(), mat.as_ref())
                        .for_each(|mut dst, src| dst.write(src.read()));
                    zipped!(z.as_mut()).for_each(|mut x| x.write(E::zero()));
                    zipped!(z.as_mut().diagonal()).for_each(|mut x| x.write(E::one()));

                    faer_evd::hessenberg_real_evd::multishift_qr(
                        true,
                        t.as_mut(),
                        Some(z.as_mut()),
                        w_re.as_mut(),
                        w_im.as_mut(),
                        0,
                        n,
                        epsilon(),
                        min_positive(),
                        parallelism,
                        stack.rb_mut(),
                        Default::default(),
                    );
                });
            });
        }
    }
}

fn hessenberg<E: ComplexField>(criterion: &mut Criterion) {
    for n in [32, 64, 128, 256, 512, 1024, 4096] {
        let mat = Mat::from_fn(n, n, |_, _| random::<E>());
        let mut t = mat.clone();
        let bs = faer_qr::no_pivoting::compute::recommended_blocksize::<E>(n - 1, n - 1);
        let mut householder = Mat::zeros(n - 1, bs);

        let parallelism = Parallelism::Rayon(0);
        let mut mem = GlobalMemBuffer::new(
            faer_evd::hessenberg::make_hessenberg_in_place_req::<E>(n, bs, Parallelism::None)
                .unwrap(),
        );
        let mut stack = DynStack::new(&mut mem);

        criterion.bench_function(
            &format!("hessenberg-mt-{}-{}", type_name::<E>(), n),
            |bencher| {
                bencher.iter(|| {
                    zipped!(t.as_mut(), mat.as_ref())
                        .for_each(|mut dst, src| dst.write(src.read()));

                    faer_evd::hessenberg::make_hessenberg_in_place(
                        t.as_mut(),
                        householder.as_mut(),
                        parallelism,
                        stack.rb_mut(),
                    );
                });
            },
        );
    }
}

fn unsym_evd<E: RealField>(criterion: &mut Criterion) {
    for n in [32, 64, 128, 256, 512, 1024, 4096] {
        let mat = Mat::from_fn(n, n, |_, _| random::<E>());
        let mut z = mat.clone();
        let mut w_re = Mat::zeros(n, 1);
        let mut w_im = Mat::zeros(n, 1);

        {
            let parallelism = Parallelism::None;
            let mut mem = GlobalMemBuffer::new(
                faer_evd::compute_evd_req::<E>(
                    n,
                    faer_evd::ComputeVectors::Yes,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            criterion.bench_function(
                &format!("unsym-evd-st-{}-{}", type_name::<E>(), n),
                |bencher| {
                    bencher.iter(|| {
                        faer_evd::compute_evd_real(
                            mat.as_ref(),
                            w_re.as_mut(),
                            w_im.as_mut(),
                            Some(z.as_mut()),
                            parallelism,
                            stack.rb_mut(),
                            Default::default(),
                        );
                    });
                },
            );
        }
        {
            let parallelism = Parallelism::Rayon(0);
            let mut mem = GlobalMemBuffer::new(
                faer_evd::compute_evd_req::<E>(
                    n,
                    faer_evd::ComputeVectors::Yes,
                    parallelism,
                    Default::default(),
                )
                .unwrap(),
            );
            let mut stack = DynStack::new(&mut mem);

            criterion.bench_function(
                &format!("unsym-evd-mt-{}-{}", type_name::<E>(), n),
                |bencher| {
                    bencher.iter(|| {
                        faer_evd::compute_evd_real(
                            mat.as_ref(),
                            w_re.as_mut(),
                            w_im.as_mut(),
                            Some(z.as_mut()),
                            parallelism,
                            stack.rb_mut(),
                            Default::default(),
                        );
                    });
                },
            );
        }
    }
}

fn cplx_schur_nalgebra(criterion: &mut Criterion) {
    type Cplx64 = num_complex::Complex64;
    for n in [32, 64, 128, 256, 512, 1024, 4096] {
        let mat = nalgebra::DMatrix::<Cplx64>::from_fn(n, n, |_, _| random::<Cplx64>());
        criterion.bench_function(
            &format!("schur-nalgebra-{}-{}", type_name::<Cplx64>(), n),
            |bencher| {
                bencher.iter(|| mat.clone().schur());
            },
        );
    }
}

criterion_group!(
    benches,
    tridiagonalization::<f32>,
    tridiagonalization::<f64>,
    tridiagonalization::<c64>,
    tridiagonal_evd::<f32>,
    tridiagonal_evd::<f64>,
    evd::<f32>,
    evd::<f64>,
    evd::<c64>,
    cplx_schur::<c64>,
    real_schur::<f64>,
    hessenberg::<f64>,
    unsym_evd::<f64>,
    cplx_schur_nalgebra,
    evd_nalgebra,
);
criterion_main!(benches);
