// CONFIDENTIAL
// (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

//! VSA bench scaffold — populated during implementation phase.

use criterion::{Criterion, criterion_group, criterion_main};

fn placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| 1 + 1));
}

criterion_group!(benches, placeholder);
criterion_main!(benches);
