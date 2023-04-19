use criterion::*;
use dyn_stack::{DynStack, GlobalMemBuffer};
use faer_core::{c32, c64, zipped, ComplexField, Mat, Parallelism};
use faer_evd::tridiag::{tridiagonalize_in_place, tridiagonalize_in_place_req};
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
    } else {
        panic!()
    }
}

fn tridiagonalization<E: ComplexField>(criterion: &mut Criterion) {
    for n in [32, 64, 128, 256, 512, 1024] {
        let mut mat = Mat::with_dims(n, n, |_, _| random::<E>());
        let adjoint = mat.adjoint().to_owned();

        zipped!(mat.as_mut(), adjoint.as_ref())
            .for_each(|mut x, y| x.write(x.read().add(&y.read())));

        let mut trid = mat.clone();
        let mut tau_left = Mat::zeros(n - 1, 1);

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
}

criterion_group!(
    benches,
    tridiagonalization::<f32>,
    tridiagonalization::<f64>,
    tridiagonalization::<c64>,
);
criterion_main!(benches);
