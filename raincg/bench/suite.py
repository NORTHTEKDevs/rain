"""Declarative stage registry + runner for the RAINCG benchmark suite.

Each stage is a plain dict (see STAGES below). The runner skips a stage when
a result JSON already matches its `expected_result_glob` in raincg/results/
(unless --force), otherwise runs `cmd` via subprocess with a timeout,
streaming combined stdout/stderr to raincg/results/logs/<stage>.log.

Usage:
  python -m raincg.bench.suite                       # list all stages, exit (safe default)
  python -m raincg.bench.suite --stages a,b --dry-run # show plan for a,b, exit
  python -m raincg.bench.suite --stages a,b           # run a,b (skip if result exists)
  python -m raincg.bench.suite --stages a,b --force   # run a,b even if result exists
"""
from __future__ import annotations

import argparse
import glob as globmod
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "raincg" / "results"
LOGS_DIR = RESULTS_DIR / "logs"
DEFAULT_PY = sys.executable


def _safe_system(system: str) -> str:
    """Filesystem-safe form of a `system` value (Windows disallows ':'; '/'
    shows up in OpenRouter model slugs like 'anthropic/claude-sonnet-5' and
    would otherwise be read as a path separator)."""
    return system.replace(":", "-").replace("/", "-")


def _out_path(bench: str, system: str, seed: int) -> str:
    return f"{bench}__{_safe_system(system)}__seed{seed}.json"


def _stage(name, bench, system, cmd_tail, timeout_s, needs=None, seed=0,
           out_flag="--out"):
    """Build a stage dict. cmd_tail is the module-invocation tail (after
    {py} -m). out_flag="--out" substitutes the per-stage result path {out};
    out_flag="--out-dir" passes the results directory instead, for tools like
    run_benchmark that write contract JSONs themselves.
    """
    out_name = _out_path(bench, system, seed)
    out_args = (["--out-dir", "raincg/results"] if out_flag == "--out-dir"
                else ["--out", "{out}"])
    return {
        "name": name,
        "bench": bench,
        "system": system,
        "seed": seed,
        "cmd": ["{py}", "-m", *cmd_tail, *out_args],
        "expected_result_glob": f"{bench}__{_safe_system(system)}__seed*.json",
        "out_name": out_name,
        "timeout_s": timeout_s,
        "needs": needs,
    }


STAGES: list[dict] = [
    _stage("pcfg_vsa", "pcfg_set", "pure_vsa",
           ["raincg.bench.run_benchmark"],
           timeout_s=1800, out_flag="--out-dir"),
    # run_benchmark --all evaluates VSA + Transformer paired and writes both
    # contract JSONs itself; checkpointed so a killed run resumes. The long
    # training is normally driven chunked by the orchestrator, not the suite.
    _stage("pcfg_transformer", "pcfg_set", "transformer_d512x2",
           ["raincg.bench.run_benchmark", "--all",
            "--ckpt", "raincg/results/pcfg_transformer_d512x2.pt", "--resume"],
           timeout_s=14400, needs="long-cpu", out_flag="--out-dir"),
    _stage("scan_transformer_baseline", "scan_addprim_jump", "transformer_d128x3",
           ["experiments.scan_transformer_baseline"],
           timeout_s=3600),
    _stage("scan_hybrid_supervised", "scan_addprim_jump", "hybrid_tagger_supervised",
           ["experiments.scan_hybrid"],
           timeout_s=1800),
    _stage("scan_hybrid_outputonly", "scan_addprim_jump", "hybrid_outputonly_reinforce",
           ["experiments.scan_hybrid_outputonly"],
           timeout_s=1800),
    _stage("scan_llm_llama", "scan_addprim_jump", "llm_llama3.2:3b",
           ["experiments.scan_llm_baseline", "--model", "llama3.2:3b", "--limit", "100"],
           timeout_s=1800, needs="ollama:llama3.2:3b"),
    _stage("scan_llm_qwen", "scan_addprim_jump", "llm_qwen2.5:14b",
           ["experiments.scan_llm_baseline", "--model", "qwen2.5:14b", "--limit", "50"],
           timeout_s=3600, needs="ollama:qwen2.5:14b"),
    _stage("cogs_llm_llama", "cogs_gen", "llm_llama3.2:3b",
           ["experiments.cogs_llm_baseline", "--model", "llama3.2:3b", "--limit", "100"],
           timeout_s=1800, needs="ollama:llama3.2:3b"),
    _stage("grammar_induction", "scan_addprim_jump", "grammar_induction_reinforce",
           ["experiments.scan_grammar_induction"],
           timeout_s=1800),
    _stage("cogs_template_symbolic", "cogs_gen", "template_learner_symbolic",
           ["experiments.cogs_template_symbolic_eval"],
           timeout_s=300),
    _stage("cogs_role_reinforce", "cogs_gen", "template_learner_reinforce_role",
           ["experiments.cogs_role_reinforce"],
           timeout_s=1800),
    _stage("scan_llm_sonnet5", "scan_addprim_jump", "llm_anthropic/claude-sonnet-5",
           ["experiments.scan_llm_baseline", "--backend", "openrouter",
            "--model", "anthropic/claude-sonnet-5", "--limit", "100"],
           timeout_s=3600, needs="openrouter-api-key"),
    _stage("cogs_llm_sonnet5", "cogs_gen", "llm_anthropic/claude-sonnet-5",
           ["experiments.cogs_llm_baseline", "--backend", "openrouter",
            "--model", "anthropic/claude-sonnet-5", "--limit", "100"],
           timeout_s=3600, needs="openrouter-api-key"),
    _stage("scan_llm_gpt55", "scan_addprim_jump", "llm_openai/gpt-5.5",
           ["experiments.scan_llm_baseline", "--backend", "openrouter",
            "--model", "openai/gpt-5.5", "--limit", "100"],
           timeout_s=3600, needs="openrouter-api-key"),
    _stage("cogs_llm_gpt55", "cogs_gen", "llm_openai/gpt-5.5",
           ["experiments.cogs_llm_baseline", "--backend", "openrouter",
            "--model", "openai/gpt-5.5", "--limit", "100"],
           timeout_s=3600, needs="openrouter-api-key"),
]


def stage_by_name(name: str, stages: list[dict] = STAGES) -> dict | None:
    for s in stages:
        if s["name"] == name:
            return s
    return None


def existing_results(stage: dict, results_dir: Path) -> list[str]:
    pattern = str(results_dir / stage["expected_result_glob"])
    return sorted(globmod.glob(pattern))


@dataclass
class StageOutcome:
    name: str
    status: str  # "skipped" | "ran" | "failed" | "timeout"
    returncode: int | None = None
    log_path: str | None = None
    detail: str = ""
    cmd: list[str] = field(default_factory=list)


def resolve_cmd(stage: dict, py: str, repo: Path) -> tuple[list[str], str]:
    out_path = str(Path(repo) / "raincg" / "results" / stage["out_name"])
    cmd = [tok.replace("{py}", py).replace("{repo}", str(repo)).replace("{out}", out_path)
           for tok in stage["cmd"]]
    return cmd, out_path


def run_stage(stage: dict, *, py: str = DEFAULT_PY, repo: Path = REPO,
              results_dir: Path = RESULTS_DIR, logs_dir: Path = LOGS_DIR,
              force: bool = False) -> StageOutcome:
    results_dir = Path(results_dir)
    logs_dir = Path(logs_dir)
    existing = existing_results(stage, results_dir)
    if existing and not force:
        return StageOutcome(stage["name"], "skipped",
                             detail=f"existing result: {existing[0]}")

    cmd, _out_path = resolve_cmd(stage, py, repo)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{stage['name']}.log"

    try:
        with open(log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.run(cmd, cwd=str(repo), stdout=log_fh,
                                   stderr=subprocess.STDOUT,
                                   timeout=stage["timeout_s"])
        status = "ran" if proc.returncode == 0 else "failed"
        return StageOutcome(stage["name"], status, returncode=proc.returncode,
                             log_path=str(log_path), cmd=cmd)
    except subprocess.TimeoutExpired:
        return StageOutcome(stage["name"], "timeout", log_path=str(log_path),
                             detail=f"exceeded {stage['timeout_s']}s", cmd=cmd)


def render_stage_table(stages: list[dict], results_dir: Path = RESULTS_DIR) -> str:
    results_dir = Path(results_dir)
    head = f"{'name':<26}{'bench':<20}{'system':<28}{'needs':<20}{'timeout_s':>10}  status"
    lines = [head, "-" * len(head)]
    for s in stages:
        existing = existing_results(s, results_dir)
        will = "SKIP (has result)" if existing else "WOULD RUN"
        needs = s.get("needs") or "-"
        lines.append(f"{s['name']:<26}{s['bench']:<20}{s['system']:<28}{needs:<20}"
                     f"{s['timeout_s']:>10}  {will}")
    return "\n".join(lines)


def render_plan(stages: list[dict], results_dir: Path, py: str, repo: Path) -> str:
    lines = []
    for s in stages:
        cmd, out_path = resolve_cmd(s, py, repo)
        existing = existing_results(s, results_dir)
        action = f"SKIP (has {existing[0]})" if existing else "RUN"
        lines.append(f"[{s['name']}] {action}\n  cmd: {' '.join(cmd)}\n  out: {out_path}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, stages: list[dict] = STAGES,
         results_dir: Path = RESULTS_DIR, logs_dir: Path = LOGS_DIR,
         py: str = DEFAULT_PY, repo: Path = REPO) -> int:
    ap = argparse.ArgumentParser(description="RAINCG bench suite runner")
    ap.add_argument("--stages", default=None,
                    help="comma-separated stage names to run/plan; omit to list all")
    ap.add_argument("--force", action="store_true", help="rerun even if result exists")
    ap.add_argument("--dry-run", action="store_true", help="print plan, do not execute")
    args = ap.parse_args(argv)

    if args.stages is None:
        print(render_stage_table(stages, results_dir))
        return 0

    wanted = [n.strip() for n in args.stages.split(",") if n.strip()]
    selected = []
    unknown = []
    for n in wanted:
        s = stage_by_name(n, stages)
        if s is None:
            unknown.append(n)
        else:
            selected.append(s)
    if unknown:
        print(f"unknown stage(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(render_plan(selected, results_dir, py, repo))
        return 0

    exit_code = 0
    for s in selected:
        outcome = run_stage(s, py=py, repo=repo, results_dir=results_dir,
                            logs_dir=logs_dir, force=args.force)
        print(f"[{outcome.name}] {outcome.status} {outcome.detail}".rstrip())
        if outcome.status in ("failed", "timeout"):
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
