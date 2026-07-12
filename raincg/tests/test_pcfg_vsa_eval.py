from pathlib import Path
import pytest
from raincg.bench.pcfg_vsa_eval import run_vsa_pcfg

REPO = Path(__file__).resolve().parents[2]
PCFG = REPO / "data" / "pcfg"


@pytest.mark.skipif(not (PCFG / "test.src").exists(), reason="PCFG data missing")
def test_vsa_subset_high_accuracy():
    res = run_vsa_pcfg(d=4096, limit=200, falsify=False)
    assert res.result.total == 200
    assert res.result.accuracy >= 0.99
    assert res.fit_seconds >= 0.0
    assert res.eval_seconds >= 0.0


@pytest.mark.skipif(not (PCFG / "test.src").exists(), reason="PCFG data missing")
def test_vsa_falsify_near_zero():
    res = run_vsa_pcfg(d=4096, limit=200, falsify=True)
    assert res.result.accuracy <= 0.05
