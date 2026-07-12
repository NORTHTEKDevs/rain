// (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

use rain_rs::hymn::{HymnConfig, HymnModel};

#[test]
fn forward_output_is_bipolar_and_correct_length() {
    let config = HymnConfig {
        in_dim: 32,
        hidden_dim: 16,
        out_dim: 32,
        dropout: 0.0,
        seed: 7,
    };
    let model = HymnModel::new(config);
    let state: Vec<i16> = (0..32).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
    let input: Vec<i16> = (0..32).map(|i| if i % 3 == 0 { 1 } else { -1 }).collect();
    let out = model.forward(&state, &input);
    assert_eq!(out.len(), 32);
    for v in &out {
        assert!(*v == 1 || *v == -1, "output not bipolar: {v}");
    }
}

#[test]
fn forward_is_deterministic_with_same_seed() {
    let config = HymnConfig {
        in_dim: 16,
        hidden_dim: 8,
        out_dim: 16,
        dropout: 0.0,
        seed: 42,
    };
    let m1 = HymnModel::new(config);
    let m2 = HymnModel::new(config);
    let state = vec![1_i16; 16];
    let input = vec![-1_i16; 16];
    let o1 = m1.forward(&state, &input);
    let o2 = m2.forward(&state, &input);
    assert_eq!(o1, o2);
}
