use criterion::{criterion_group, criterion_main, Criterion};

pub fn qr(c: &mut Criterion) {
    let _c = c;
}

criterion_group!(benches, qr);
criterion_main!(benches);
