"""Reads raincg/results/*.json + raincg/bench/citations.py, renders RESULTS.md.

One table per bench: System | Acc (95% CI) | n | Params | Train s | Eval s |
Evidence. Evidence is measured-fresh / measured-prior-session /
cited-not-reproduced. Honesty footnotes below each table:
  - cogs_gen: ALWAYS show full-denominator accuracy; annotate any
    in-scope-slice number with its coverage %.
  - any row whose config used a test-limit is flagged "subsampled, paired".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from raincg.bench.citations import CITATIONS, EVIDENCE_TIER

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "raincg" / "results"
DEFAULT_OUT = REPO / "raincg" / "RESULTS.md"


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    results_dir = Path(results_dir)
    out = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def all_benches(results: list[dict], citations: list[dict] = CITATIONS) -> list[str]:
    seen: list[str] = []
    for r in results:
        if r["bench"] not in seen:
            seen.append(r["bench"])
    for c in citations:
        if c["bench"] not in seen:
            seen.append(c["bench"])
    return seen


def _fmt_pct(x: float) -> str:
    return f"{x:.2%}"


def _measured_row(r: dict) -> tuple[str, str, str, str, str, str, str]:
    ci = r.get("ci95") or [r.get("accuracy", 0.0), r.get("accuracy", 0.0)]
    acc_ci = f"{_fmt_pct(r.get('accuracy', 0.0))} ({_fmt_pct(ci[0])}-{_fmt_pct(ci[1])})"
    return (
        r.get("system", "?"),
        acc_ci,
        str(r.get("n", "-")),
        f"{r.get('params', 0):,}",
        f"{r.get('train_s', 0):.1f}",
        f"{r.get('eval_s', 0):.1f}",
        r.get("evidence_tier", "measured-fresh"),
    )


def _cited_row(c: dict) -> tuple[str, str, str, str, str, str, str]:
    acc_ci = _fmt_pct(c.get("accuracy", 0.0))
    return (
        c.get("system", "?"),
        acc_ci,
        "-",
        "-",
        "-",
        "-",
        f"{EVIDENCE_TIER} ({c.get('source', 'unknown source')})",
    )


NOTES_TRUNCATE_CHARS = 200


def _footnotes_for_bench(bench: str, rows: list[dict]) -> list[str]:
    notes: list[str] = []
    for r in rows:
        cfg = r.get("config") or {}
        system = r.get("system", "?")
        split = r.get("split", "?")

        if bench == "cogs_gen":
            full_n = cfg.get("full_n", r.get("n", 0))
            full_correct = cfg.get("full_correct", r.get("correct", 0))
            full_acc = (full_correct / full_n) if full_n else 0.0
            notes.append(
                f"- {system} ({split}): full-denominator accuracy = "
                f"{_fmt_pct(full_acc)} (n={full_n})"
            )
            n = r.get("n", 0)
            if full_n and n and n < full_n:
                coverage = n / full_n
                notes.append(
                    f"  in-scope slice n={n} is {coverage:.1%} coverage of the "
                    f"full {full_n}-example denominator"
                )

        test_limit = cfg.get("test_limit")
        if test_limit:
            notes.append(
                f"- {system} ({split}): subsampled, paired (test_limit={test_limit})"
            )

        result_notes = (r.get("notes") or "").strip()
        if result_notes:
            truncated = result_notes[:NOTES_TRUNCATE_CHARS]
            if len(result_notes) > NOTES_TRUNCATE_CHARS:
                truncated = truncated.rstrip() + "..."
            notes.append(f"- {system} ({split}) notes: {truncated}")
    return notes


METHODOLOGY_AND_CAVEATS = """## Methodology & caveats

- **Params column** counts gradient-trained floats only. Symbolic solvers
  (e.g. `template_learner_symbolic`, `template_learner_reinforce_role`)
  legitimately show `params: 0` -- that means zero gradient-trained floats,
  NOT zero learned structure. Their structural knowledge lives in symbolic
  lookup tables (templates, verb maps, role tables, etc.), disclosed in each
  result's `notes` and `config` (see `config.symbolic_table_entries` where
  present).
- **LLM rows** in this document are small, LOCAL models doing few-shot
  DIRECT-MAPPING prompting (sentence -> answer, no decomposition). Published
  work using least-to-most / decomposition prompting on FRONTIER LLMs
  (GPT-3 code-davinci-002, PaLM) solves these same splits at 97-99%+ -- see
  the `cited-not-reproduced` rows below. The differentiator this repo
  demonstrates is compute cost, determinism, and auditability of the exact
  symbolic/hybrid solvers, NOT raw task capability that a frontier LLM lacks.
- **`grammar_induction_reinforce` 41.7%** (measured-fresh in this document)
  SUPERSEDES the 80.7% figure recorded in `design/RESONATOR-FINDINGS.md`
  (finding F25). The discrepancy is unreconciled -- likely seed/config
  variance between runs -- and is flagged here rather than silently
  overwritten; treat the 41.7% figure in this document as authoritative
  until reconciled.
- **Hybrid rows** (`hybrid_tagger_supervised`, `hybrid_outputonly_reinforce`)
  are evaluated on the full 7706-example `scan_addprim_jump` test split, not
  a subsample.
"""


def render_results_md(results: list[dict], citations: list[dict] = CITATIONS) -> str:
    benches: list[str] = []
    for r in results:
        if r["bench"] not in benches:
            benches.append(r["bench"])
    for c in citations:
        if c["bench"] not in benches:
            benches.append(c["bench"])

    lines = ["# RAINCG Results", "", METHODOLOGY_AND_CAVEATS]
    header = "| System | Acc (95% CI) | n | Params | Train s | Eval s | Evidence |"
    sep = "|---|---|---|---|---|---|---|"

    for bench in benches:
        lines.append(f"## {bench}")
        lines.append("")
        lines.append(header)
        lines.append(sep)

        bench_results = [r for r in results if r["bench"] == bench]
        bench_citations = [c for c in citations if c["bench"] == bench]

        combined = (
            [(r.get("accuracy", 0.0), _measured_row(r)) for r in bench_results]
            + [(c.get("accuracy", 0.0), _cited_row(c)) for c in bench_citations]
        )
        combined.sort(key=lambda t: t[0], reverse=True)

        for _, row in combined:
            lines.append("| " + " | ".join(row) + " |")

        footnotes = _footnotes_for_bench(bench, bench_results)
        if footnotes:
            lines.append("")
            lines.extend(footnotes)

        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render raincg/RESULTS.md")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    results = load_results(Path(args.results_dir))
    md = render_results_md(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path} ({len(results)} measured results, "
          f"{len(CITATIONS)} citations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
