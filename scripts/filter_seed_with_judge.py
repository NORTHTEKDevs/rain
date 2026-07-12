"""Filter a KB seed JSONL through a local Ollama judge.

The Phase-2 RLAIF run on the unfiltered llama3.2:3b-distilled seed
(2711 facts) showed the judge rejecting 50/50 sampled probes. The error
class is genuine semantic mistakes in the distilled triples
('gcc is not part of c', 'soccer is played with feet but not made of them').

Fix at the source: ask the judge to vet every triple BEFORE it lands in
the KB, dropping the bad ones. This is the same primitive as Track-3
RLAIF, just invoked at seed time instead of query time.

Usage:
    python -m scripts.filter_seed_with_judge \\
        --in  data/kb_seed/llama3b_expanded.jsonl \\
        --out data/kb_seed/llama3b_expanded_filtered.jsonl \\
        --rejected data/kb_seed/llama3b_expanded_rejected.jsonl \\
        --judge-model llama3.2:3b \\
        --min-confidence 0.5 \\
        --verbose

Slow: each triple = one Ollama call (~1 s on this workstation).
On 2711 facts that's ~45 minutes wall time.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rain.feedback.ollama_judge import OllamaJudge


@dataclass
class FilterRun:
    in_path: str
    out_path: str
    rejected_path: str | None
    judge_model: str
    min_confidence: float
    accept_unsure: bool
    total: int = 0
    judged: int = 0
    judge_parse_failures: int = 0
    kept: int = 0
    rejected: int = 0
    wall_seconds: float = 0.0
    per_relation_kept: dict[str, int] = field(default_factory=dict)
    per_relation_rejected: dict[str, int] = field(default_factory=dict)


def _triple_from(fact: dict) -> tuple[str, str, str] | None:
    s = fact.get("subject") or fact.get("s")
    r = fact.get("relation") or fact.get("r")
    o = fact.get("object") or fact.get("o")
    if not (s and r and o):
        return None
    return (str(s), str(r), str(o))


def _format_question(subject: str, relation: str) -> str:
    return f"What is the {relation} of {subject}?"


def _format_answer(subject: str, relation: str, obj: str) -> str:
    return f"The {relation} of {subject} is {obj}."


def run(args: argparse.Namespace) -> FilterRun:
    in_path = Path(args.in_path)
    if not in_path.is_file():
        raise SystemExit(f"in file not found: {in_path}")

    judge = OllamaJudge(model=args.judge_model, url=args.url,
                       timeout=args.timeout, temperature=0.0)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = Path(args.rejected_path) if args.rejected_path else None
    if rejected_path:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)

    state = FilterRun(
        in_path=str(in_path),
        out_path=str(out_path),
        rejected_path=str(rejected_path) if rejected_path else None,
        judge_model=args.judge_model,
        min_confidence=args.min_confidence,
        accept_unsure=args.accept_unsure,
    )
    kept_counter: Counter[str] = Counter()
    rej_counter: Counter[str] = Counter()

    t0 = time.perf_counter()
    with in_path.open(encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout, \
         (rejected_path.open("w", encoding="utf-8")
          if rejected_path else _NullSink()) as frej:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                fact = json.loads(line)
            except json.JSONDecodeError:
                continue
            triple = _triple_from(fact)
            if triple is None:
                continue
            state.total += 1
            s, r, o = triple

            verdict = judge.judge(_format_question(s, r), _format_answer(s, r, o))
            if verdict is None:
                state.judge_parse_failures += 1
                # Tie-break: treat unparseable as a "keep" (don't lose data
                # over judge stutter). Mark in metadata for downstream filter.
                fact.setdefault("judge", {"status": "parse_failure"})
                fout.write(json.dumps(fact) + "\n")
                state.kept += 1
                kept_counter[r] += 1
                continue

            state.judged += 1
            confident_enough = verdict.confidence >= state.min_confidence
            if verdict.correct and confident_enough:
                accept = True
                reason = "judge:correct"
            elif (not verdict.correct) and confident_enough:
                accept = False
                reason = "judge:incorrect"
            else:
                # Below the confidence floor -- judge isn't sure.
                accept = state.accept_unsure
                reason = "judge:unsure"

            enriched = dict(fact)
            enriched["judge"] = {
                "correct": verdict.correct,
                "confidence": round(verdict.confidence, 3),
                "reasoning": verdict.reasoning,
                "decision": "kept" if accept else "rejected",
                "reason": reason,
            }
            if accept:
                fout.write(json.dumps(enriched) + "\n")
                state.kept += 1
                kept_counter[r] += 1
            else:
                if rejected_path:
                    frej.write(json.dumps(enriched) + "\n")
                state.rejected += 1
                rej_counter[r] += 1

            if args.verbose and (i + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                print(
                    f"[{i + 1}/{state.total}] kept={state.kept} "
                    f"rejected={state.rejected} parse_fail={state.judge_parse_failures} "
                    f"elapsed={elapsed:.0f}s rate={rate:.2f}/s",
                    flush=True,
                )

    state.wall_seconds = time.perf_counter() - t0
    state.per_relation_kept = dict(kept_counter)
    state.per_relation_rejected = dict(rej_counter)
    return state


class _NullSink:
    """Context-manager no-op for the rejected file when not requested."""
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def write(self, *args, **kwargs): pass


def main() -> None:
    p = argparse.ArgumentParser(description="Filter KB seed JSONL via Ollama judge")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--rejected", dest="rejected_path", default=None,
                   help="optional path to write rejected triples (for analysis)")
    p.add_argument("--judge-model", default="llama3.2:3b")
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="judge confidence threshold for a decision to count")
    p.add_argument("--accept-unsure", action="store_true",
                   help="when judge confidence < min-confidence, keep the fact "
                        "(default: drop on low-confidence verdicts)")
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    rs = run(args)

    summary = {k: v for k, v in asdict(rs).items()
               if k not in ("per_relation_kept", "per_relation_rejected")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
