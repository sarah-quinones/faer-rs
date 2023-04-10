use criterion::{criterion_group, criterion_main, Criterion};
use faer_core::{
    c32, c64, is_vectorizable, mul::matmul_with_conj_gemm_dispatch as matmul_with_conj,
    ComplexField, Conj, Mat, Parallelism,
};
use std::time::Duration;

pub fn matmul<E: ComplexField>(criterion: &mut Criterion) {
    let name = core::any::type_name::<E>();
    for n in [32, 64, 128, 256, 512, 1024] {
        let mut acc = Mat::<E>::zeros(n, n);
        let a = Mat::<E>::zeros(n, n);
        let b = Mat::<E>::zeros(n, n);
        criterion.bench_function(&format!("matmul-{name}-st-{n}"), |bencher| {
            bencher.iter(|| {
                matmul_with_conj(
                    acc.as_mut(),
                    a.as_ref(),
                    Conj::No,
                    b.as_ref(),
                    Conj::No,
                    None,
                    E::one(),
                    Parallelism::None,
                    false,
                );
            });
        });
        criterion.bench_function(&format!("matmul-{name}-mt-{n}"), |bencher| {
            bencher.iter(|| {
                matmul_with_conj(
                    acc.as_mut(),
                    a.as_ref(),
                    Conj::No,
                    b.as_ref(),
                    Conj::No,
                    None,
                    E::one(),
                    Parallelism::Rayon(0),
                    false,
                );
            });
        });

        if is_vectorizable::<E>() {
            criterion.bench_function(&format!("gemm-{name}-mt-{n}"), |bencher| {
                bencher.iter(|| {
                    matmul_with_conj(
                        acc.as_mut(),
                        a.as_ref(),
                        Conj::No,
                        b.as_ref(),
                        Conj::No,
                        None,
                        E::one(),
                        Parallelism::Rayon(0),
                        true,
                    );
                });
            });
            criterion.bench_function(&format!("gemm-{name}-st-{n}"), |bencher| {
                bencher.iter(|| {
                    matmul_with_conj(
                        acc.as_mut(),
                        a.as_ref(),
                        Conj::No,
                        b.as_ref(),
                        Conj::No,
                        None,
                        E::one(),
                        Parallelism::None,
                        true,
                    );
                });
            });
        }
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);
    targets =
        matmul::<num_complex::Complex64>,
        matmul::<c64>,
        matmul::<f64>,
        matmul::<num_complex::Complex32>,
        matmul::<c32>,
        matmul::<f32>,
);
criterion_main!(benches);
