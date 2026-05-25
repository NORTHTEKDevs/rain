# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""RAIN-Net chat v2: investor-demo-grade interactive REPL.

Shows in one screen what makes RAIN-Net different from an LLM:
    - Multi-turn session memory (episodic recall across many turns)
    - Skill routing visible (math queries -> math_solver skill)
    - Active distillation (low-confidence queries -> Ollama teacher)
    - Audit trail (cited facts, binding score, routed experts)
    - Continual learning (add a fact mid-conversation; it's usable next turn)

Run (offline mode, no Ollama):
    python scripts/rain_chat_v2.py

Run (with active distillation via local Ollama):
    python scripts/rain_chat_v2.py --ollama

Run (load a starter KB):
    python scripts/rain_chat_v2.py --seed-kb data/distill/v0_general.jsonl

Commands inside the REPL:
    > <question>             ask RAIN-Net
    > learn <fact>            add a fact to semantic memory
    > recall <topic>          show what prior turns RAIN-Net would recall
    > history                 show session history
    > stats                   show system + session stats
    > help                    show help
    > exit                    quit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rain.core.rain_net import RainNet, RainNetConfig
from rain.core.session import Session


def load_seed_jsonl(net: RainNet, path: Path) -> int:
    """Bulk-load a JSONL file of facts (one per line, with text key)."""
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = d.get("text") or (
                d.get("teacher_answer") and f"Q: {d.get('query', '')} A: {d['teacher_answer']}"
            )
            if not text:
                continue
            net.ingest_fact(text=text, source=d.get("source", path.stem))
            n += 1
    return n


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=10_000)
    p.add_argument("--candidates", type=int, default=2)
    p.add_argument(
        "--skills",
        default="skills",
        help="path to skills dir (default: skills/); pass empty string to disable",
    )
    p.add_argument(
        "--verifier",
        default="data/checkpoints/verifier_head_v0.npz",
        help="path to trained verifier head; not loaded if missing",
    )
    p.add_argument(
        "--hymn",
        default=None,
        help="path to trained HYMN-Plus .npz for live LM generation",
    )
    p.add_argument("--seed-kb", default=None, help="JSONL of facts to bulk-load")
    p.add_argument("--ollama", action="store_true", help="enable active-learning loop")
    p.add_argument("--ollama-model", default="llama3.2:3b")
    args = p.parse_args(argv)

    # Build RainNet with all v0.1 components wired.
    cfg = RainNetConfig(
        dim=args.dim,
        n_candidates=args.candidates,
        skills_dir=args.skills or None,
        verifier_checkpoint_path=args.verifier or None,
        hymn_checkpoint_path=args.hymn,
        semantic_top_k=4,
    )
    net = RainNet(config=cfg)
    session = Session(net=net, recall_top_k=3)

    # Active-learning loop (optional).
    loop = None
    if args.ollama:
        from rain.training.active_learning import (
            ActiveLearningConfig,
            ActiveLearningLoop,
        )
        from rain.training.distillation import OllamaTeacher

        loop = ActiveLearningLoop(
            net=net,
            config=ActiveLearningConfig(
                confidence_threshold=0.4,
                distill_jsonl_path="data/distill/session_live.jsonl",
            ),
            ollama=OllamaTeacher(model=args.ollama_model),
        )

    # Optional seed KB.
    if args.seed_kb:
        n = load_seed_jsonl(net, Path(args.seed_kb))
        print(f"(seeded {n} facts from {args.seed_kb})")

    print("=" * 60)
    print("RAIN-Net chat v2")
    print("=" * 60)
    print(f"  dim:          {cfg.dim}")
    print(f"  candidates:   {cfg.n_candidates}")
    print(f"  skills:       {len(net.skill_registry)} loaded")
    print(f"  verifier:     {'loaded' if net.verifier.n_updates > 0 else 'fresh'} (updates={net.verifier.n_updates})")
    print(f"  hymn LM:      {'available' if args.hymn else 'disabled'}")
    print(f"  active learn: {'on (Ollama ' + args.ollama_model + ')' if loop else 'off'}")
    print(f"  KB size:      {len(net.memory.semantic)} facts")
    print(f"  Type 'help' for commands; 'exit' to quit.")
    print()

    while True:
        try:
            line = input(f"[t{len(session.turns)}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"exit", "quit", ":q"}:
            break
        if line == "help":
            print(
                "  <question>           ask RAIN-Net (multi-turn session)\n"
                "  learn <fact>          add a fact to semantic memory\n"
                "  recall <topic>        peek at episodic-memory recall\n"
                "  history               show session history\n"
                "  stats                 show system + session stats\n"
                "  exit                  quit"
            )
            continue
        if line == "history":
            print(session.history_summary())
            continue
        if line == "stats":
            print(json.dumps({"net": net.stats(), "session": session.stats()}, indent=2, default=str))
            continue
        if line.startswith("learn "):
            text = line[6:].strip()
            if not text:
                print("(empty)")
                continue
            fid = net.ingest_fact(text=text, source="user")
            print(f"(learned as {fid})")
            continue
        if line.startswith("recall "):
            topic = line[7:].strip()
            recall = session.recall(topic, top_k=5)
            if not recall:
                print("(no prior turns)")
            else:
                for score, t in recall:
                    print(f"  sim={score:+.3f}  [t{t.turn_id}] {t.query[:70]}")
            continue
        # Default: treat as a question.
        if loop is not None:
            # Active-learning path: escalate to Ollama on low confidence.
            report = loop.handle(line)
            # Also record the turn into the session manually.
            session.turns.append(__import__("rain.core.session", fromlist=["Turn"]).Turn(
                turn_id=session._next_id,
                query=line,
                answer_text=report.answer_text,
                confidence=report.confidence,
                cited_fact_ids=[f.fact_id for f in report.cited_facts],
                routed_experts=report.provenance,
            ))
            session._next_id += 1
        else:
            report = session.turn(line)
        print()
        print(report.human_format())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
