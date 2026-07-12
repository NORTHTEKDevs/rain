#!/usr/bin/env bash
# RAIN-Net v0.1 reproducibility check.
# Runs the test suite + full benchmark and reports against expected numbers.
# Anyone who clones the repo should be able to run this and see the same results.

set -euo pipefail

echo "=== RAIN-Net v0.1 reproducibility check ==="
echo

# 1. Tests
echo "--- 1. Test suite ---"
python -m pytest tests/ -q 2>&1 | tail -3
echo

# 2. Benchmark report
echo "--- 2. Full benchmark report ---"
python scripts/rain_net_full_report.py 2>&1 | grep -E "^\[|complete"
echo

# 3. Skill REPL smoke test
echo "--- 3. Skill REPL smoke test ---"
printf "what is 47 * 38\n100 km to miles\n30 days from 2026-05-24\nextract emails from contact foo@bar.com\nexit\n" \
  | python scripts/rain_chat_v2.py 2>/dev/null \
  | grep -E "^ANSWER" \
  | head -4
echo

# 4. Expected numbers (sanity check)
echo "--- 4. Expected vs measured ---"
python << 'PYEOF'
import json
from pathlib import Path
expected = {
    "synthetic_retrieval.rain_net.top1": 0.21,
    "synthetic_retrieval.rain_net.top3": 0.485,
    "distilled_kb_self_eval.top1": 0.7,
    "distilled_kb_self_eval.top3": 0.85,
    "routing_accuracy.accuracy": 0.85,
}
p = Path("docs/RESULTS-v0.1.json")
if not p.exists():
    print("(report not found; run rain-report first)")
else:
    d = json.loads(p.read_text())
    def dig(obj, path):
        for k in path.split("."):
            obj = obj.get(k, None)
            if obj is None:
                return None
        return obj
    print(f'{"metric":<55} {"expected":>10}  {"got":>10}  {"verdict":>10}')
    for k, ev in expected.items():
        got = dig(d, k)
        if got is None:
            verdict = "MISSING"
        elif got >= ev * 0.9:
            verdict = "OK"
        else:
            verdict = "REGRESSION"
        print(f"{k:<55} {ev:>10.3f}  {(got or 0):>10.3f}  {verdict:>10}")
PYEOF
echo
echo "=== Done ==="
