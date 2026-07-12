import pytest

from rain.spine import _RUST_AVAILABLE

if not _RUST_AVAILABLE:
    pytest.skip("rain._rust not available", allow_module_level=True)

from evals.tier3_soundness.dispatch_correctness import run_benchmark


def test_a5_py_matches_rust_100_pairs():
    r = run_benchmark(n_pairs=100, in_dim=64, hidden_dim=32, out_dim=64, seed=0)
    assert r.match_rate == 1.0, f"diffs at: {r.diff_positions}"
    assert r.overall_pass


def test_a5_works_at_larger_dims():
    """Verify scaling -- D=512 should still produce identical outputs."""
    r = run_benchmark(n_pairs=20, in_dim=512, hidden_dim=128, out_dim=512, seed=1)
    assert r.match_rate == 1.0, f"diffs at: {r.diff_positions}"
