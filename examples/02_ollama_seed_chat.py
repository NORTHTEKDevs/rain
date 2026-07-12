"""RAIN with Ollama-distilled KB -- the broke-mode end-to-end demo.

This is the visible payoff of Track 2 of the broke-mode training plan
(docs/plans/2026-05-22-broke-mode-training.md): a RAIN agent whose
world knowledge was produced for $0 by a local Ollama model and is
queryable through every normal cognitive surface (ask, describe,
think_aloud, self_describe), with the usual RAIN guarantees
(calibrated epistemic, grounded citations, refusal on unknowns).

Usage:
    # First seed a JSONL via scripts/seed_kb_from_ollama (one-time).
    python -m scripts.seed_kb_from_ollama \\
        --model llama3.2:3b --n-per-topic 25 \\
        --out data/kb_seed/llama3b_default.jsonl

    # Then walk through this demo:
    python examples/02_ollama_seed_chat.py data/kb_seed/llama3b_default.jsonl

The path is the only argument; defaults to the broke-mode-plan default
JSONL location.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rain.agent import ConsciousAgent
from rain.data.kb_seed import seed_from_jsonl


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main(seed_path: str) -> int:
    path = Path(seed_path)
    if not path.exists():
        print(f"seed file not found: {path}", file=sys.stderr)
        print("Generate one first:", file=sys.stderr)
        print("  python -m scripts.seed_kb_from_ollama --model llama3.2:3b "
              "--out " + str(path), file=sys.stderr)
        return 2

    banner(f"1. Load Ollama-distilled facts from {path.name}")
    agent = ConsciousAgent(dim=2048, num_shards=32, seed=0)
    n = seed_from_jsonl(agent.kb, str(path))
    print(f"loaded {n} facts into the ConsciousAgent KB.")

    # Surface every unique subject in the loaded JSONL for the curious
    import json
    subjects_seen: dict[str, int] = {}
    with open(path, encoding="utf-8") as fh:
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
                subjects_seen[s] = subjects_seen.get(s, 0) + 1
    top = sorted(subjects_seen.items(), key=lambda kv: -kv[1])[:15]
    print(f"top subjects (by fact count): {[f'{s}({n})' for s, n in top]}")

    banner("2. Ask a few questions across the loaded subjects")
    # Use the top subjects + the most-likely relations RAIN's KB shape supports.
    probe_relations = ["isa", "lives_in", "has_part", "color", "made_of",
                       "kind", "has_property", "wrote", "located_in"]
    asked = 0
    answered = 0
    for s, _ in top[:5]:
        for r in probe_relations:
            ans = agent.ask(s, r)
            asked += 1
            if ans.inference_source is not None:
                answered += 1
                print(f"ask({s!r}, {r!r}) -> {ans.text}")
                print(f"   citations: {ans.citations}")
                print(f"   epistemic: {ans.epistemic}  confidence: {ans.confidence}")
                print()
                break  # one answered relation per subject is enough
    print(f"answered {answered}/{asked} probes across the loaded KB.")

    banner("3. describe() composes multiple stored facts into prose")
    for s, _ in top[:3]:
        print(f"--- describe({s!r}) ---")
        print(agent.describe(s))
        print()

    banner("4. self_describe()")
    print(agent.self_describe())

    banner("5. unanswered queries -> calibrated refusal")
    for s, r in [("aardvark", "isa"), ("rain", "writes"), ("foo", "bar")]:
        ans = agent.ask(s, r)
        print(f"ask({s!r}, {r!r}) -> {ans.text}  [epistemic={ans.epistemic}]")
    print()
    print("RAIN refuses rather than hallucinating -- the broke-mode demo of "
          "what \"trained like an LLM but doesn't lie\" looks like.")
    return 0


if __name__ == "__main__":
    default_path = "data/kb_seed/llama3b_default.jsonl"
    arg = sys.argv[1] if len(sys.argv) > 1 else default_path
    sys.exit(main(arg))
