# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Phase-2 continual-feedback loop: seed KB, ask, judge, calibrate.

Closes the Track-3 (LLM-as-judge) wiring at scale: load an Ollama-seeded
KB into a fresh ConsciousAgent, generate a probe of (subject, relation)
questions across the loaded facts, run a local Ollama model as the judge
on every RAIN answer, and feed verdicts back through the per-relation
Bayesian calibration tally. Captures pre/post calibration deltas.

This is the "free RLAIF" path from docs/plans/2026-05-22-broke-mode-training.md
Track 3, exercised end-to-end on real seed data instead of a toy smoke.

Usage:
    python -m scripts.run_phase2_feedback \\
        --seed data/kb_seed/llama3b_expanded.jsonl \\
        --judge-model llama3.2:3b \\
        --n-probes 100 \\
        --min-confidence 0.0 \\
        --out evals/results/phase2_feedback_<date>.json

`--min-confidence 0.0` accepts every verdict, including low-confidence
ones; set to 0.5 in production to gate fuzzy verdicts out.
"""

from __future__ import annotations
import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

from rain.agent import ConsciousAgent
from rain.data.kb_seed import seed_from_jsonl
from rain.feedback.ollama_judge import OllamaJudge


@dataclass
class Phase2Run:
    seed_path: str
    judge_model: str
    n_probes: int
    n_facts_loaded: int
    min_confidence: float
    started: str
    wall_seconds: float
    total_probes: int
    answered_probes: int
    judged_probes: int
    judge_parse_failures: int
    correct_count: int
    feedback_updates: int
    relations_touched: int
    pre_calibration: dict[str, float] = field(default_factory=dict)
    post_calibration: dict[str, float] = field(default_factory=dict)
    delta_calibration: dict[str, float] = field(default_factory=dict)
    per_relation_correct: dict[str, int] = field(default_factory=dict)
    per_relation_total: dict[str, int] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)


def _read_facts(path: Path) -> list[dict]:
    facts: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            facts.append(obj)
    return facts


def _normalize_triple(fact: dict) -> tuple[str, str, str] | None:
    s = fact.get("subject") or fact.get("s")
    r = fact.get("relation") or fact.get("r")
    o = fact.get("object") or fact.get("o")
    if not (s and r and o):
        return None
    return (str(s), str(r), str(o))


def _build_probes(
    facts: list[dict], n: int, rng: random.Random
) -> list[tuple[str, str, str]]:
    """Pick (subject, relation, expected_object) triples to ask + judge."""
    triples: list[tuple[str, str, str]] = []
    for f in facts:
        t = _normalize_triple(f)
        if t is not None:
            triples.append(t)
    if not triples:
        return []
    if n >= len(triples):
        rng.shuffle(triples)
        return triples
    return rng.sample(triples, n)


def run(args: argparse.Namespace) -> Phase2Run:
    seed_path = Path(args.seed_path)
    if not seed_path.is_file():
        raise SystemExit(f"seed file not found: {seed_path}")

    facts = _read_facts(seed_path)
    rng = random.Random(args.rng_seed)
    probes = _build_probes(facts, args.n_probes, rng)
    if not probes:
        raise SystemExit(f"no valid (subject, relation, object) triples in {seed_path}")

    agent = ConsciousAgent(dim=args.dim, num_shards=args.num_shards, seed=args.rng_seed)
    n_loaded = seed_from_jsonl(agent.kb, str(seed_path))

    judge = OllamaJudge(
        model=args.judge_model, url=args.url,
        timeout=args.timeout, temperature=0.0,
    )

    # Pre-feedback calibration on the touched relations.
    touched_relations = sorted({r for _, r, _ in probes})
    pre = {r: round(agent.calibration.calibration(r), 4) for r in touched_relations}

    run_state = Phase2Run(
        seed_path=str(seed_path),
        judge_model=args.judge_model,
        n_probes=args.n_probes,
        n_facts_loaded=n_loaded,
        min_confidence=args.min_confidence,
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_seconds=0.0,
        total_probes=len(probes),
        answered_probes=0,
        judged_probes=0,
        judge_parse_failures=0,
        correct_count=0,
        feedback_updates=0,
        relations_touched=len(touched_relations),
        pre_calibration=pre,
    )
    per_rel_correct: Counter[str] = Counter()
    per_rel_total: Counter[str] = Counter()
    examples: list[dict] = []

    t0 = time.perf_counter()
    for i, (s, r, expected) in enumerate(probes):
        ans = agent.ask(s, r)
        if ans.inference_source is not None:
            run_state.answered_probes += 1
        question = f"What is the {r} of {s}?"
        verdict = judge.judge(question, ans.text)
        if verdict is None:
            run_state.judge_parse_failures += 1
            continue
        run_state.judged_probes += 1
        per_rel_total[r] += 1
        if verdict.correct:
            run_state.correct_count += 1
            per_rel_correct[r] += 1
        if verdict.confidence >= args.min_confidence:
            agent.feedback(r, verdict.correct)
            run_state.feedback_updates += 1
        if len(examples) < args.n_examples:
            examples.append({
                "subject": s, "relation": r, "expected": expected,
                "rain_text": ans.text,
                "rain_inference_source": ans.inference_source,
                "rain_confidence": ans.confidence,
                "judge_correct": verdict.correct,
                "judge_confidence": verdict.confidence,
                "judge_reasoning": verdict.reasoning,
            })
        if args.verbose and (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"[{i + 1}/{len(probes)}] judged={run_state.judged_probes}  "
                f"correct={run_state.correct_count}  "
                f"feedback={run_state.feedback_updates}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    run_state.wall_seconds = time.perf_counter() - t0

    post = {r: round(agent.calibration.calibration(r), 4) for r in touched_relations}
    run_state.post_calibration = post
    run_state.delta_calibration = {
        r: round(post[r] - pre[r], 4) for r in touched_relations
    }
    run_state.per_relation_correct = dict(per_rel_correct)
    run_state.per_relation_total = dict(per_rel_total)
    run_state.examples = examples
    return run_state


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 LLM-judge feedback loop")
    p.add_argument("--seed", dest="seed_path", required=True,
                   help="path to KB seed JSONL")
    p.add_argument("--judge-model", default="llama3.2:3b")
    p.add_argument("--n-probes", type=int, default=100)
    p.add_argument("--n-examples", type=int, default=10,
                   help="how many (probe + verdict) examples to capture in the report")
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--num-shards", type=int, default=32)
    p.add_argument("--rng-seed", type=int, default=0,
                   help="deterministic seed for ConsciousAgent + probe sampling")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="judge confidence threshold to fire a feedback update")
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rs = run(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(rs), indent=2))
    summary = {k: v for k, v in asdict(rs).items()
               if k not in ("examples", "pre_calibration", "post_calibration",
                            "delta_calibration", "per_relation_correct",
                            "per_relation_total")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
