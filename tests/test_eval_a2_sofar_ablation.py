# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

import numpy as np

from evals.tier3_soundness.sofar_ablation import (
    _build_codebook_and_dirs,
    _val_mse,
    run_benchmark,
)
from scripts.pretrain_hymn import HymnSurrogate


def test_a2_runs_and_emits_finite_metrics():
    """Smoke: both branches return finite MSE; result carries the spec fields."""
    r = run_benchmark(
        dim=256,
        vocab_size=32,
        routing_k=16,
        num_roles=8,
        hidden_dim=128,
        n_eval_steps=100,
        seed=0,
    )
    assert r.benchmark == "A2_sofar_on_vsa_ablation"
    assert np.isfinite(r.mse_routing_off)
    assert np.isfinite(r.mse_routing_on)
    assert np.isfinite(r.improvement)
    assert r.notes == []
    # Pass/kill are complementary booleans by spec construction
    assert r.overall_pass != r.kill_triggered


def test_routing_at_identity_init_is_noop():
    """Architectural invariant: with the adapter at strict identity-at-init
    (lora_up=0, beam_gate=0), routing-on and routing-off are bit-identical
    -- the BeamSteeringAdapter short-circuit returns the state unchanged."""
    dim, k = 256, 16
    cb, mapper, beam, cb_arr, role_arr = _build_codebook_and_dirs(
        dim=dim,
        vocab_size=32,
        k=k,
        num_roles=8,
        seed=0,
    )
    # Force STRICT identity-at-init (undo the benchmark's post-warmup seeding)
    beam.lora_up = np.zeros_like(beam.lora_up)
    beam.beam_gate = np.float32(0.0)
    directions = mapper.get(cb_arr, role_arr)
    model = HymnSurrogate(in_dim=dim, hidden_dim=128, out_dim=dim, seed=0)
    corpus = "the quick brown fox " * 50
    off = _val_mse(model, cb, corpus, 50, seed=42, route=False)
    on = _val_mse(model, cb, corpus, 50, seed=42, route=True, beam=beam, directions=directions)
    assert abs(off - on) < 1e-6, f"identity-at-init should be a no-op; got off={off}, on={on}"


def test_a2_improvement_sign_is_consistent():
    """Same seed -> deterministic improvement value. Doesn't assert pass/fail
    because v0 routing (untrained LoRA) is not expected to clear the 3% bar;
    we only assert the metric is computed correctly and reproducibly."""
    r1 = run_benchmark(
        dim=256,
        vocab_size=32,
        routing_k=16,
        num_roles=8,
        hidden_dim=128,
        n_eval_steps=100,
        seed=7,
    )
    r2 = run_benchmark(
        dim=256,
        vocab_size=32,
        routing_k=16,
        num_roles=8,
        hidden_dim=128,
        n_eval_steps=100,
        seed=7,
    )
    assert r1.mse_routing_off == r2.mse_routing_off
    assert r1.mse_routing_on == r2.mse_routing_on
    assert r1.improvement == r2.improvement
