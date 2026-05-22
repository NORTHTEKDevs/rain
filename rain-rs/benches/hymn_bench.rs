// CONFIDENTIAL - PATENT PENDING
// (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rain_rs::hymn::{HymnConfig, HymnModel};

fn bench_hymn_forward_10k_4k(c: &mut Criterion) {
    let config = HymnConfig {
        in_dim: 10000,
        hidden_dim: 4096,
        out_dim: 10000,
        dropout: 0.0,
        seed: 42,
    };
    let model = HymnModel::new(config);
    let state: Vec<i16> = (0..10000).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
    let input: Vec<i16> = (0..10000).map(|i| if i % 3 == 0 { 1 } else { -1 }).collect();
    c.bench_function("hymn_forward_10k_4k", |b| {
        b.iter(|| model.forward(black_box(&state), black_box(&input)))
    });
}

criterion_group!(benches, bench_hymn_forward_10k_4k);
criterion_main!(benches);
