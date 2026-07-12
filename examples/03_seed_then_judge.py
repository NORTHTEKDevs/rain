"""RAIN with Ollama-seeded KB AND Ollama judge -- broke-mode Tracks 2+3 live.

Walks through the full no-paid-RLHF feedback loop:
  1. Seed RAIN's KB from a JSONL produced by `scripts.seed_kb_from_ollama`.
  2. Ask RAIN about a sample of subjects -- get answers + citations + epistemic.
  3. Have a separate Ollama model judge each answer.
  4. Feed the verdicts back through `agent.feedback(...)` to update the
     per-relation Bayesian calibration tally.
  5. Re-query and show how the epistemic classification shifted.

The judge calls are LIVE Ollama requests (~1 s each); the script is
slow by design -- this is offline-batch infrastructure, not a per-token
feedback loop.

Usage:
    python examples/03_seed_then_judge.py [seed.jsonl] [N=5]

Defaults to data/kb_seed/llama3b_default.jsonl + 5 sample subjects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rain.agent import ConsciousAgent
from rain.data.kb_seed import seed_from_jsonl
from rain.feedback.ollama_judge import OllamaJudge, judge_agent_session


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _pick_top_subjects(path: Path, n: int) -> list[str]:
    """Surface the top-N subjects by fact count in the seed JSONL."""
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                fact = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = fact.get("subject") or fact.get("s")
            if s:
                counts[s] = counts.get(s, 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


def main(seed_path: str, n_subjects: int) -> int:
    path = Path(seed_path)
    if not path.exists():
        print(f"seed file not found: {path}", file=sys.stderr)
        return 2

    banner("1. Load Ollama-distilled facts into a fresh ConsciousAgent")
    agent = ConsciousAgent(dim=2048, num_shards=32, seed=0)
    n_loaded = seed_from_jsonl(agent.kb, str(path))
    print(f"loaded {n_loaded} facts from {path.name}.")

    banner(f"2. Initial calibration on the top {n_subjects} subjects (pre-feedback)")
    top_subjects = _pick_top_subjects(path, n_subjects)
    probe_relations = ["isa", "lives_in", "has_part", "color", "made_of",
                       "kind", "has_property", "wrote", "located_in"]
    qa_pairs: list[tuple[str, str]] = []
    for s in top_subjects:
        for r in probe_relations:
            if agent.ask(s, r).inference_source is not None:
                qa_pairs.append((s, r))
                break  # one good relation per subject

    print(f"selected {len(qa_pairs)} (subject, relation) probes:")
    for s, r in qa_pairs:
        ans = agent.ask(s, r)
        cal = agent.calibration.calibration(r)
        print(f"  {s:20s}.{r:15s} -> {ans.text[:60]!r}")
        print(f"    pre-feedback: epistemic={ans.epistemic}, "
              f"confidence={ans.confidence:.2f}, calibration[{r}]={cal:.3f}")

    banner("3. Ask Ollama (llama3.2:3b) to judge each answer")
    judge = OllamaJudge(model="llama3.2:3b")
    print(f"running {len(qa_pairs)} judgments (live LLM calls -- ~1 s each)...")
    # Run judgments one at a time + print each verdict; then call the
    # session helper to actually fire the feedback update with the same logic.
    for s, r in qa_pairs:
        ans = agent.ask(s, r)
        verdict = judge.judge(f"What is the {r} of {s}?", ans.text)
        if verdict is None:
            print(f"  {s}.{r}: judge returned unparseable response")
        else:
            print(f"  {s}.{r}: correct={verdict.correct}, "
                  f"conf={verdict.confidence:.2f}, reasoning={verdict.reasoning!r}")
    stats = judge_agent_session(agent, qa_pairs, judge, min_confidence_for_update=0.5)
    print()
    print("session stats:")
    print(json.dumps(stats, indent=2))

    banner("4. Post-feedback calibration on the same probes")
    for s, r in qa_pairs:
        ans = agent.ask(s, r)
        cal = agent.calibration.calibration(r)
        print(f"  {s:20s}.{r:15s} -> epistemic={ans.epistemic}, "
              f"confidence={ans.confidence:.2f}, calibration[{r}]={cal:.3f}")

    banner("5. Per-relation tally snapshot")
    # Surface every relation the agent has feedback on.
    rels = sorted({r for _, r in qa_pairs})
    for r in rels:
        cal = agent.calibration.calibration(r)
        print(f"  calibration[{r:15s}] = {cal:.3f}")

    print()
    print("Done. No paid LLM API was called. No human label was provided.")
    print("Tracks 2 + 3 of the broke-mode plan, end-to-end, in one process.")
    return 0


if __name__ == "__main__":
    default_path = "data/kb_seed/llama3b_default.jsonl"
    path_arg = sys.argv[1] if len(sys.argv) > 1 else default_path
    n_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    sys.exit(main(path_arg, n_arg))
