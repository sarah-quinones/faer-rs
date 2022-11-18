use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

use dyn_stack::*;
use faer_core::Parallelism;
use rand::random;

use faer_core::Mat;

pub fn qr(c: &mut Criterion) {
    use faer_qr::no_pivoting;
    for n in [64, 128, 256, 512, 1024, 4096] {
        c.bench_function(&format!("faer-st-qr-{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), n, n);
            let mut householder = Mat::with_dims(|_, _| random::<f64>(), n, n);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                unsafe {
                    no_pivoting::compute::qr_in_place_recursive(
                        mat.as_mut(),
                        householder.as_mut(),
                        Parallelism::None,
                        0,
                        stack.rb_mut(),
                    )
                };
            })
        });

        c.bench_function(&format!("faer-mt-qr-{n}"), |b| {
            let mut mat = Mat::with_dims(|_, _| random::<f64>(), n, n);
            let mut householder = Mat::with_dims(|_, _| random::<f64>(), n, n);

            let mut mem = GlobalMemBuffer::new(StackReq::new::<f64>(1024 * 1024 * 1024));
            let mut stack = DynStack::new(&mut mem);

            b.iter(|| {
                unsafe {
                    no_pivoting::compute::qr_in_place_recursive(
                        mat.as_mut(),
                        householder.as_mut(),
                        Parallelism::Rayon,
                        0,
                        stack.rb_mut(),
                    )
                };
            })
        });
    }

    let _c = c;
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);
    targets = qr
);
criterion_main!(benches);
