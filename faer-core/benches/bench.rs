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
        group.bench_function(&format!("faer-mt-solve-{n}"), |b| {
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
    }
}

pub fn complex_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex-mul");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_secs(1));

    for n in [64, 128, 256, 512, 1024] {
        group.bench_function(&format!("faer-mt-cmul-{n}"), |b| {
            let mut dst_real = Mat::<f64>::zeros(n, n);
            let mut dst_imag = Mat::<f64>::zeros(n, n);
            let lhs_real = Mat::<f64>::zeros(n, n);
            let lhs_imag = Mat::<f64>::zeros(n, n);
            let rhs_real = Mat::<f64>::zeros(n, n);
            let rhs_imag = Mat::<f64>::zeros(n, n);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);
            b.iter(|| unsafe {
                faer_core::complex_mul::matmul_unchecked(
                    dst_real.as_mut(),
                    dst_imag.as_mut(),
                    lhs_real.as_ref(),
                    lhs_imag.as_ref(),
                    false,
                    rhs_real.as_ref(),
                    rhs_imag.as_ref(),
                    false,
                    None,
                    &1.0,
                    12,
                    stack.rb_mut(),
                );
            })
        });

        group.bench_function(&format!("faer-mt-cmul-tri-{n}"), |b| {
            let mut dst_real = Mat::<f64>::zeros(n, n);
            let mut dst_imag = Mat::<f64>::zeros(n, n);
            let lhs_real = Mat::<f64>::zeros(n, n);
            let lhs_imag = Mat::<f64>::zeros(n, n);
            let rhs_real = Mat::<f64>::zeros(n, n);
            let rhs_imag = Mat::<f64>::zeros(n, n);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);
            b.iter(|| unsafe {
                faer_core::complex_mul::triangular::matmul_unchecked(
                    dst_real.as_mut(),
                    dst_imag.as_mut(),
                    faer_core::mul::triangular::BlockStructure::TriangularLower,
                    lhs_real.as_ref(),
                    lhs_imag.as_ref(),
                    faer_core::mul::triangular::BlockStructure::Rectangular,
                    false,
                    rhs_real.as_ref(),
                    rhs_imag.as_ref(),
                    faer_core::mul::triangular::BlockStructure::Rectangular,
                    false,
                    None,
                    &1.0,
                    12,
                    stack.rb_mut(),
                );
            })
        });
    }
}

criterion_group!(benches, complex_mul, solve);
criterion_main!(benches);
