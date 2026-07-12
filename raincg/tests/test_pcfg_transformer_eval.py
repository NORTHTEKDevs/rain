import tempfile
import os
from raincg.bench.transformer_baseline import train_baseline
from raincg.bench.pcfg_transformer_eval import eval_transformer_on_pairs


def test_eval_returns_result():
    pairs = [("copy A B", ["A", "B"]), ("reverse A B", ["B", "A"])]
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "m.pt")
        train_baseline(pairs, d_model=32, nhead=2, num_layers=1, ff=64,
                       epochs=60, batch_size=2, ckpt_path=ckpt, max_minutes=1.0)
        res = eval_transformer_on_pairs(ckpt, pairs, max_output_len=10)
        assert res.result.total == 2
        assert 0.0 <= res.result.accuracy <= 1.0
        assert res.eval_seconds >= 0.0
