import json
import tempfile
from pathlib import Path

from raincg.bench.run_benchmark import _seeded_subsample, write_result_json
from raincg.bench.common import ExactMatchResult


def test_seeded_subsample_is_deterministic_for_same_seed():
    pairs = [(f"src {i}", [str(i)]) for i in range(50)]
    a = _seeded_subsample(pairs, 10, seed=42)
    b = _seeded_subsample(pairs, 10, seed=42)
    assert a == b
    assert len(a) == 10


def test_seeded_subsample_differs_for_different_seed():
    pairs = [(f"src {i}", [str(i)]) for i in range(50)]
    a = _seeded_subsample(pairs, 10, seed=1)
    b = _seeded_subsample(pairs, 10, seed=2)
    assert a != b


def test_seeded_subsample_noop_when_k_none_or_ge_len():
    pairs = [(f"src {i}", [str(i)]) for i in range(5)]
    assert _seeded_subsample(pairs, None, seed=0) == pairs
    assert _seeded_subsample(pairs, 100, seed=0) == pairs


def test_seeded_subsample_pairing_applies_same_indices():
    # Simulates the paired-comparison contract: subsampling once and reusing
    # the result for both systems keeps VSA and transformer on the same items.
    pairs = [(f"src {i}", [str(i)]) for i in range(30)]
    subset = _seeded_subsample(pairs, 7, seed=5)
    vsa_input = list(subset)
    transformer_input = list(subset)
    assert vsa_input == transformer_input


def test_write_result_json_matches_contract_schema():
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        result = ExactMatchResult(correct=8, total=10)
        path = write_result_json(
            out_dir, bench="pcfg_set", system="pure_vsa", split="test",
            result=result, params=0, train_s=1.5, eval_s=2.5, seed=0,
            config={"vsa_d": 8192}, notes="unit test",
        )
        assert path.exists()
        assert path.name == "pcfg_set__pure_vsa__seed0.json"
        data = json.loads(path.read_text())
        for key in ("bench", "system", "split", "n", "correct", "accuracy",
                    "ci95", "params", "train_s", "eval_s", "seed", "config",
                    "timestamp", "evidence_tier", "notes"):
            assert key in data
        assert data["bench"] == "pcfg_set"
        assert data["system"] == "pure_vsa"
        assert data["n"] == 10
        assert data["correct"] == 8
        assert abs(data["accuracy"] - 0.8) < 1e-9
        assert len(data["ci95"]) == 2
        assert data["ci95"][0] < data["accuracy"] < data["ci95"][1]
        assert data["evidence_tier"] == "measured-fresh"
