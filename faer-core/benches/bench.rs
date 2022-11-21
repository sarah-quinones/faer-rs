use criterion::{criterion_group, criterion_main, Criterion};
use faer_core::{Mat, Parallelism};
use std::time::Duration;

pub fn solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_secs(1));

    for n in [64, 128, 256, 512, 1024] {
        group.bench_function(&format!("faer-st-gemmt-{n}"), |b| {
            let mut dst = Mat::<f64>::zeros(n, n);
            let lhs = Mat::<f64>::zeros(n, n);
            let rhs = Mat::<f64>::zeros(n, n);

            use faer_core::mul::triangular::BlockStructure::*;
            b.iter(|| {
                faer_core::mul::triangular::matmul(
                    dst.as_mut(),
                    TriangularLower,
                    lhs.as_ref(),
                    Rectangular,
                    rhs.as_ref(),
                    Rectangular,
                    Some(1.0),
                    2.5,
                    false,
                    false,
                    false,
                    Parallelism::None,
                );
            })
        });

        group.bench_function(&format!("faer-mt-gemmt-{n}"), |b| {
            let mut dst = Mat::<f64>::zeros(n, n);
            let lhs = Mat::<f64>::zeros(n, n);
            let rhs = Mat::<f64>::zeros(n, n);

            use faer_core::mul::triangular::BlockStructure::*;
            b.iter(|| {
                faer_core::mul::triangular::matmul(
                    dst.as_mut(),
                    TriangularLower,
                    lhs.as_ref(),
                    Rectangular,
                    rhs.as_ref(),
                    Rectangular,
                    Some(1.0),
                    2.5,
                    false,
                    false,
                    false,
                    Parallelism::Rayon(0),
                );
            })
        });

        group.bench_function(&format!("faer-st-colsolve-{n}"), |b| {
            let mut tri = Mat::<f64>::zeros(n, n);
            tri.as_mut().diagonal().cwise().for_each(|x| *x = 1.0);
            let mut rhs = Mat::<f64>::zeros(n, n);

            b.iter(|| {
                faer_core::solve::solve_lower_triangular_in_place(
                    tri.as_ref(),
                    rhs.as_mut(),
                    false,
                    false,
                    Parallelism::None,
                );
            })
        });
        group.bench_function(&format!("faer-st-rowsolve-{n}"), |b| {
            let mut tri = Mat::<f64>::zeros(n, n);
            tri.as_mut().diagonal().cwise().for_each(|x| *x = 1.0);
            let mut rhs = Mat::<f64>::zeros(n, n);

            b.iter(|| {
                faer_core::solve::solve_lower_triangular_in_place(
                    tri.as_ref(),
                    rhs.as_mut().transpose(),
                    false,
                    false,
                    Parallelism::None,
                );
            })
        });
        group.bench_function(&format!("faer-mt-colsolve-{n}"), |b| {
            let mut tri = Mat::<f64>::zeros(n, n);
            tri.as_mut().diagonal().cwise().for_each(|x| *x = 1.0);
            let mut rhs = Mat::<f64>::zeros(n, n);

            b.iter(|| {
                faer_core::solve::solve_lower_triangular_in_place(
                    tri.as_ref(),
                    rhs.as_mut(),
                    false,
                    false,
                    Parallelism::Rayon(rayon::current_num_threads()),
                );
            })
        });
        group.bench_function(&format!("faer-mt-rowsolve-{n}"), |b| {
            let mut tri = Mat::<f64>::zeros(n, n);
            tri.as_mut().diagonal().cwise().for_each(|x| *x = 1.0);
            let mut rhs = Mat::<f64>::zeros(n, n);

            b.iter(|| {
                faer_core::solve::solve_lower_triangular_in_place(
                    tri.as_ref(),
                    rhs.as_mut().transpose(),
                    false,
                    false,
                    Parallelism::Rayon(rayon::current_num_threads()),
                );
            })
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);
    targets = solve
);
criterion_main!(benches);
