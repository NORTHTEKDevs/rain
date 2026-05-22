// CONFIDENTIAL - PATENT PENDING
// (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

//! HYMN MLP forward — v0 minimal port of Hyperion's HYMN/FERN architecture.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Configuration for a HYMN forward layer pair.
#[derive(Debug, Clone, Copy)]
pub struct HymnConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
    pub dropout: f32,
    pub seed: u64,
}

impl Default for HymnConfig {
    fn default() -> Self {
        HymnConfig {
            in_dim: 10000,
            hidden_dim: 4096,
            out_dim: 10000,
            dropout: 0.0,
            seed: 42,
        }
    }
}

/// HYMN MLP — two-layer bipolar forward with sign activation.
pub struct HymnModel {
    config: HymnConfig,
    w1: Vec<f32>, // shape (in_dim, hidden_dim), row-major
    w2: Vec<f32>, // shape (hidden_dim, out_dim), row-major
}

impl HymnModel {
    pub fn new(config: HymnConfig) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let std1 = (1.0_f32 / config.in_dim as f32).sqrt();
        let dist1 = Normal::new(0.0_f32, std1).expect("valid normal dist");
        let w1 = (0..config.in_dim * config.hidden_dim)
            .map(|_| dist1.sample(&mut rng))
            .collect();
        let std2 = (1.0_f32 / config.hidden_dim as f32).sqrt();
        let dist2 = Normal::new(0.0_f32, std2).expect("valid normal dist");
        let w2 = (0..config.hidden_dim * config.out_dim)
            .map(|_| dist2.sample(&mut rng))
            .collect();
        HymnModel { config, w1, w2 }
    }

    /// Forward pass. `state` and `input` are bipolar i16. Returns bipolar i16.
    pub fn forward(&self, state: &[i16], input: &[i16]) -> Vec<i16> {
        assert_eq!(state.len(), self.config.in_dim, "state dim mismatch");
        assert_eq!(input.len(), self.config.in_dim, "input dim mismatch");

        // Bipolar combine: sign(state + input)
        let combined: Vec<i16> = state
            .iter()
            .zip(input.iter())
            .map(|(&s, &i)| if s + i >= 0 { 1 } else { -1 })
            .collect();

        // Linear 1: in_dim -> hidden_dim
        let h = self.config.hidden_dim;
        let mut hidden_pre = vec![0.0_f32; h];
        for d in 0..self.config.in_dim {
            let c = combined[d] as f32;
            let row = &self.w1[d * h..(d + 1) * h];
            for (hp, &w) in hidden_pre.iter_mut().zip(row.iter()) {
                *hp += w * c;
            }
        }

        // sign() activation
        let hidden_signed: Vec<i16> = hidden_pre
            .iter()
            .map(|&x| if x >= 0.0 { 1 } else { -1 })
            .collect();

        // Linear 2: hidden_dim -> out_dim
        let d_out = self.config.out_dim;
        let mut out_pre = vec![0.0_f32; d_out];
        for hi in 0..h {
            let c = hidden_signed[hi] as f32;
            let row = &self.w2[hi * d_out..(hi + 1) * d_out];
            for (op, &w) in out_pre.iter_mut().zip(row.iter()) {
                *op += w * c;
            }
        }

        // sign() output
        out_pre
            .into_iter()
            .map(|x| if x >= 0.0 { 1 } else { -1 })
            .collect()
    }

    pub fn config(&self) -> &HymnConfig {
        &self.config
    }
}
