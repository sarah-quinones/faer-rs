use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut, StackReq};
use faer_core::Mat;

pub fn solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_secs(1));

    for n in [64, 128, 256, 512, 1024] {
        group.bench_function(&format!("faer-st-colsolve-{n}"), |b| {
            let mut tri = Mat::<f64>::zeros(n, n);
            tri.as_mut().diagonal().cwise().for_each(|x| *x = 1.0);
            let mut rhs = Mat::<f64>::zeros(n, n);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);
            b.iter(|| {
                faer_core::solve::triangular::solve_lower_triangular_in_place(
                    tri.as_ref(),
                    rhs.as_mut(),
                    1,
                    stack.rb_mut(),
                );
            })
        });
        group.bench_function(&format!("faer-st-rowsolve-{n}"), |b| {
            let mut tri = Mat::<f64>::zeros(n, n);
            tri.as_mut().diagonal().cwise().for_each(|x| *x = 1.0);
            let mut rhs = Mat::<f64>::zeros(n, n);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);
            b.iter(|| {
                faer_core::solve::triangular::solve_lower_triangular_in_place(
                    tri.as_ref(),
                    rhs.as_mut().transpose(),
                    1,
                    stack.rb_mut(),
                );
            })
        });
        group.bench_function(&format!("faer-mt-colsolve-{n}"), |b| {
            let mut tri = Mat::<f64>::zeros(n, n);
            tri.as_mut().diagonal().cwise().for_each(|x| *x = 1.0);
            let mut rhs = Mat::<f64>::zeros(n, n);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);
            b.iter(|| {
                faer_core::solve::triangular::solve_lower_triangular_in_place(
                    tri.as_ref(),
                    rhs.as_mut(),
                    12,
                    stack.rb_mut(),
                );
            })
        });
        group.bench_function(&format!("faer-mt-rowsolve-{n}"), |b| {
            let mut tri = Mat::<f64>::zeros(n, n);
            tri.as_mut().diagonal().cwise().for_each(|x| *x = 1.0);
            let mut rhs = Mat::<f64>::zeros(n, n);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);
            b.iter(|| {
                faer_core::solve::triangular::solve_lower_triangular_in_place(
                    tri.as_ref(),
                    rhs.as_mut().transpose(),
                    12,
                    stack.rb_mut(),
                );
            })
        });
    }
}

criterion_group!(benches, solve);
criterion_main!(benches);
