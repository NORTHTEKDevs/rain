# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""RAIN-Net interactive demo: REPL that exercises the full pipeline.

Usage:
    python scripts/rain_net_demo.py                  # offline, KB+routing only
    python scripts/rain_net_demo.py --ollama         # adds active distillation
    python scripts/rain_net_demo.py --ollama --claude  # adds Claude escalation
    python scripts/rain_net_demo.py --seed-kb FILE   # bulk-load JSONL of facts

Commands inside the REPL:
    > ask <question>          ask the model
    > learn <fact text>       add a fact to semantic memory
    > stats                   print system stats
    > save <path>             save current KB to JSONL
    > load <path>             load KB from JSONL
    > help                    show commands
    > exit                    quit

Every answer shows the full audit trail: cited facts, binding score,
verifier score, routed experts, warnings. This is the investor-facing
demo: the value proposition (audited, continual, cheap) is visible
in every response.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rain.core.rain_net import RainNet, RainNetConfig


def build_loop_or_net(args: argparse.Namespace):
    """Return either a RainNet (offline) or ActiveLearningLoop (online).

    Imports active_learning lazily so the offline mode doesn't need
    urllib networking imports at startup.
    """
    cfg = RainNetConfig(dim=args.dim, n_candidates=args.candidates)
    net = RainNet(config=cfg)
    if not args.ollama:
        return net, None
    from rain.training.active_learning import (
        ActiveLearningConfig,
        ActiveLearningLoop,
    )
    from rain.training.distillation import ClaudeTeacher, OllamaTeacher

    al_cfg = ActiveLearningConfig(
        confidence_threshold=args.confidence,
        distill_jsonl_path=args.distill_path,
    )
    claude = ClaudeTeacher() if args.claude else None
    loop = ActiveLearningLoop(
        net=net,
        config=al_cfg,
        ollama=OllamaTeacher(model=args.ollama_model),
        claude=claude,
    )
    return net, loop


def seed_kb(net: RainNet, path: Path) -> int:
    """Bulk-load a JSONL file of facts. Each line is either:
       {"text": "..."} or {"text": "...", "source": "...", "id": "..."}.
    """
    if not path.exists():
        print(f"(seed) file not found: {path}")
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            net.ingest_fact(
                text=d["text"],
                source=d.get("source", ""),
                fact_id=d.get("id"),
            )
            n += 1
    return n


def save_kb(net: RainNet, path: Path) -> int:
    """Persist current semantic memory to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for fact in net.memory.semantic._facts.values():
            f.write(
                json.dumps(
                    {"id": fact.fact_id, "text": fact.text, "source": fact.source}
                )
                + "\n"
            )
            n += 1
    return n


def repl(net: RainNet, loop) -> None:
    print("RAIN-Net v0.1 demo. Type 'help' for commands.")
    print(f"Mode: {'active-learning (Ollama)' if loop else 'offline'}")
    print(f"KB facts: {len(net.memory.semantic)}")
    print()
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"exit", "quit", ":q"}:
            break
        if line == "help":
            print(
                "Commands:\n"
                "  ask <question>          ask RAIN-Net\n"
                "  learn <fact text>       add a fact to semantic memory\n"
                "  stats                   print system stats\n"
                "  save <path>             save current KB to JSONL\n"
                "  load <path>             load KB from JSONL\n"
                "  help                    show this help\n"
                "  exit                    quit"
            )
            continue
        if line == "stats":
            print(json.dumps(net.stats(), indent=2, default=str))
            if loop is not None:
                print(json.dumps(loop.stats(), indent=2, default=str))
            continue
        if line.startswith("learn "):
            text = line[6:].strip()
            if not text:
                print("(empty fact)")
                continue
            fid = net.ingest_fact(text=text, source="user")
            print(f"(learned as {fid})")
            continue
        if line.startswith("save "):
            n = save_kb(net, Path(line[5:].strip()))
            print(f"(saved {n} facts)")
            continue
        if line.startswith("load "):
            n = seed_kb(net, Path(line[5:].strip()))
            print(f"(loaded {n} facts)")
            continue
        if line.startswith("ask "):
            q = line[4:].strip()
        else:
            # Bare line is treated as a question.
            q = line
        if not q:
            continue
        report = (loop.handle(q) if loop else net.answer(q))
        print(report.human_format())
        print()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=10_000, help="HV dimension")
    p.add_argument("--candidates", type=int, default=4, help="TTC candidates per query")
    p.add_argument("--ollama", action="store_true", help="enable Ollama active learning")
    p.add_argument(
        "--ollama-model", default="qwen2.5:7b", help="Ollama model name (must be pulled)"
    )
    p.add_argument("--claude", action="store_true", help="enable Claude escalation")
    p.add_argument("--confidence", type=float, default=0.5, help="confidence threshold")
    p.add_argument(
        "--distill-path",
        default="data/distill/active.jsonl",
        help="path to persist distillation examples",
    )
    p.add_argument("--seed-kb", default=None, help="JSONL file of facts to bulk-load")
    args = p.parse_args(argv)

    net, loop = build_loop_or_net(args)
    if args.seed_kb:
        n = seed_kb(net, Path(args.seed_kb))
        print(f"(seeded {n} facts from {args.seed_kb})")
    repl(net, loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
