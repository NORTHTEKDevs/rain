# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""RAIN v0 chat REPL -- KB-grounded answer + HYMN sampling + judge feedback.

Wires together every working piece of the v0 broke-mode stack into one
interactive surface so RAIN is actually usable as a chat agent end-to-end
on the workstation:

  * Pre-loads an Ollama-distilled KB into a ConsciousAgent
    (`agent.tell()` for every fact in the seed JSONL).
  * Loads a trained HYMN checkpoint for char-level autoregressive sampling.
  * Optional local-Ollama judge for RLAIF-style feedback after each turn.

REPL commands:
  <subject> <relation>            ask KB; show citation chain + epistemic
  /describe <subject>             multi-fact prose description
  /sample <prompt>                HYMN free-text sampling (continuation)
  /judge <subject> <relation>     ask + judge + feed verdict back
  /selfdescribe                   RAIN's structural self-description
  /tally [relation]               per-relation Bayesian calibration tally
  /save <path>                    save the current calibration tally to JSON
  /help                           list commands
  /quit                           exit

Anything else is treated as a free-text HYMN prompt (autoregressive completion).
"""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rain.agent import ConsciousAgent
from rain.data.kb_seed import seed_from_jsonl
from rain.feedback.ollama_judge import OllamaJudge, build_triple_prompt
from rain.train.checkpoint import load_checkpoint
from rain.core.relational import Codebook
from scripts.sample_hymn import _hymn_forward, _sample_from_logits


class _HymnSampler:
    """Encapsulates a trained HYMN checkpoint + its char vocab so we can
    sample inside the REPL without reloading on every turn."""

    def __init__(self, checkpoint_path: str, corpus_path: str) -> None:
        self.W1, self.W2, self.meta = load_checkpoint(checkpoint_path)
        corpus = Path(corpus_path).read_text(encoding="utf-8")
        self.vocab = sorted(set(corpus))
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        cb = Codebook(vocab_size=256, dim=self.meta.in_dim, seed=self.meta.seed)
        self.codebook_matrix = np.stack(
            [cb.vector(c).astype(np.float32) for c in self.vocab]
        )
        self._cb = cb

    def sample(self, prompt: str, n_tokens: int, temperature: float,
               top_k: int | None, seed: int,
               repetition_penalty: float = 1.1,
               repetition_window: int = 16) -> str:
        rng = np.random.default_rng(seed)
        state = np.zeros(self.meta.in_dim, dtype=np.float32)
        for c in prompt:
            if c not in self.char_to_idx:
                c = self.vocab[0]
            state = _hymn_forward(state, self._cb.vector(c).astype(np.float32),
                                  self.W1, self.W2)
        out_chars: list[str] = []
        recent_ids: list[int] = []
        for _ in range(n_tokens):
            logits = self.codebook_matrix @ state
            idx = _sample_from_logits(
                logits, temperature, rng, top_k=top_k,
                recent_ids=recent_ids,
                repetition_penalty=repetition_penalty,
            )
            out_chars.append(self.vocab[idx])
            recent_ids.append(idx)
            if len(recent_ids) > repetition_window:
                recent_ids = recent_ids[-repetition_window:]
            state = _hymn_forward(state, self.codebook_matrix[idx], self.W1, self.W2)
        return "".join(out_chars)


def _help() -> None:
    print(__doc__)


def _handle_ask(agent: ConsciousAgent, line: str) -> bool:
    """Try to parse `<subject> <relation>`. Return True if handled.

    Tag the printed source clearly: [KB] for citation-grounded answers,
    [HYMN-guess] for fallback completions (epistemic=guess, no citations).
    """
    tokens = line.strip().split()
    if len(tokens) < 2:
        return False
    subject = tokens[0]
    relation = "_".join(tokens[1:])
    ans = agent.ask(subject, relation)
    if ans.inference_source is None:
        return False  # let caller fall back to HYMN sampling
    cal = agent.calibration.calibration(relation)
    tag = "[HYMN-guess]" if ans.inference_source == "hymn" else "[KB]"
    print(f"{tag} {ans.text}")
    if ans.citations:
        print(f"     citations: {ans.citations}")
    print(f"     epistemic: {ans.epistemic}  "
          f"confidence: {ans.confidence:.2f}  "
          f"relation calibration: {cal:.3f}")
    return True


def repl(agent: ConsciousAgent, sampler: _HymnSampler | None,
         judge: OllamaJudge | None, args: argparse.Namespace) -> None:
    print()
    print("RAIN v0 chat. /help for commands. /quit to exit.")
    if sampler:
        print(f"  HYMN: {Path(args.checkpoint).name}")
        print(f"  KB:   {Path(args.kb).name}  ({len(list(_ for _ in open(args.kb)))} lines)")
    print()
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", ":q"):
            break
        if line in ("/help", "?"):
            _help()
            continue
        if line.startswith("/describe "):
            entity = line[len("/describe "):].strip()
            print("[KB]", agent.describe(entity))
            continue
        if line == "/selfdescribe":
            print("[RAIN]", agent.self_describe())
            continue
        if line.startswith("/sample"):
            prompt = line[len("/sample"):].strip()
            if sampler is None:
                print("(no HYMN checkpoint loaded)")
                continue
            text = sampler.sample(prompt, n_tokens=args.sample_tokens,
                                  temperature=args.temperature,
                                  top_k=(args.top_k if args.top_k > 0 else None),
                                  seed=np.random.randint(0, 1 << 31),
                                  repetition_penalty=args.repetition_penalty)
            print(f"[HYMN, temp={args.temperature}, top-k={args.top_k}]")
            print(prompt + text)
            continue
        if line.startswith("/judge "):
            rest = line[len("/judge "):].strip().split()
            if len(rest) < 2:
                print("(usage: /judge <subject> <relation>)")
                continue
            subject = rest[0]
            relation = "_".join(rest[1:])
            ans = agent.ask(subject, relation)
            print(f"[KB] {ans.text}")
            if judge is None:
                print("(no judge configured; pass --judge-model to enable)")
                continue
            # Use the clean-form judge prompt so the verdict tracks the
            # underlying fact, not RAIN's surface "I know that ... directly
            # from a stored fact" framing or snake_case formatting.
            stored_obj = ans.citations[0][2] if ans.citations else None
            if stored_obj is None:
                print("(no citation -- nothing to judge)")
                continue
            q, judge_answer = build_triple_prompt(subject, relation, stored_obj)
            verdict = judge.judge(q, judge_answer)
            if verdict is None:
                print("(judge returned unparseable verdict)")
                continue
            print(f"[judge] correct={verdict.correct} conf={verdict.confidence:.2f}")
            print(f"        reasoning: {verdict.reasoning}")
            if verdict.confidence >= args.min_confidence:
                agent.feedback(relation, verdict.correct)
                print(f"        -> feedback fired, calibration[{relation}] = "
                      f"{agent.calibration.calibration(relation):.3f}")
            continue
        if line == "/tally":
            tally = {}
            for rel in sorted(agent.calibration._total):
                tally[rel] = round(agent.calibration.calibration(rel), 4)
            print("[tally]", json.dumps(tally, indent=2))
            continue
        if line.startswith("/tally "):
            rel = line[len("/tally "):].strip()
            print(f"[tally] calibration[{rel}] = "
                  f"{agent.calibration.calibration(rel):.4f}")
            continue
        if line.startswith("/save "):
            path = Path(line[len("/save "):].strip())
            data = {rel: round(agent.calibration.calibration(rel), 4)
                    for rel in sorted(agent.calibration._total)}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
            print(f"saved tally -> {path}")
            continue

        # Fall through: try KB ask if it parses as <subject> <relation>;
        # otherwise treat the line as a HYMN sampling prompt.
        if _handle_ask(agent, line):
            continue
        if sampler is None:
            print("(no KB hit; no HYMN checkpoint -- nothing to say)")
            continue
        text = sampler.sample(line + "\n", n_tokens=args.sample_tokens,
                              temperature=args.temperature,
                              top_k=(args.top_k if args.top_k > 0 else None),
                              seed=np.random.randint(0, 1 << 31))
        print(f"[HYMN, no-kb-hit fallback, temp={args.temperature}]")
        print(line + "\n" + text)


def main() -> int:
    p = argparse.ArgumentParser(description="RAIN v0 chat REPL")
    p.add_argument("--kb", default="data/kb_seed/llama3b_expanded.jsonl",
                   help="path to KB seed JSONL")
    p.add_argument("--checkpoint",
                   default="data/checkpoints/hymn_carry16_30k.npz",
                   help="trained HYMN checkpoint; empty string to disable HYMN")
    p.add_argument("--corpus",
                   default="data/corpora/tiny_shakespeare.txt",
                   help="training corpus (needed by sampler for char vocab)")
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--num-shards", type=int, default=32)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--sample-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.1,
                   help="repetition penalty for /sample (default 1.1, 1.0 disables)")
    p.add_argument("--judge-model", default="",
                   help="if set, enables /judge command; e.g. llama3.2:3b")
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="judge confidence threshold to fire calibration update")
    p.add_argument("--noninteractive", default="",
                   help="for smoke testing: run a single command and exit")
    args = p.parse_args()

    agent = ConsciousAgent(dim=args.dim, num_shards=args.num_shards, seed=args.rng_seed)
    n = 0
    if args.kb and Path(args.kb).is_file():
        n = seed_from_jsonl(agent.kb, args.kb)
    sampler = (_HymnSampler(args.checkpoint, args.corpus)
               if args.checkpoint and Path(args.checkpoint).is_file()
               else None)
    # If we have a sampler, also attach it to the agent so agent.ask itself
    # gains the HYMN-fallback behavior (used by integrations beyond this REPL).
    if sampler is not None:
        def _agent_sampler(prompt: str, n_tokens: int) -> str:
            return sampler.sample(prompt, n_tokens=n_tokens,
                                  temperature=args.temperature,
                                  top_k=(args.top_k if args.top_k > 0 else None),
                                  seed=np.random.randint(0, 1 << 31),
                                  repetition_penalty=args.repetition_penalty)
        agent.attach_hymn_sampler(_agent_sampler, n_tokens=args.sample_tokens)
    judge = OllamaJudge(model=args.judge_model) if args.judge_model else None

    if args.noninteractive:
        # Simulate a single REPL line then exit -- useful for smoke tests.
        class _StdinPatch:
            def __init__(self, lines: list[str]):
                self._lines = lines + ["/quit"]
                self._i = 0
            def readline(self):
                if self._i >= len(self._lines):
                    return ""
                ln = self._lines[self._i] + "\n"
                self._i += 1
                return ln
        # We rely on input() which reads from sys.stdin, so swap it out.
        import builtins
        feeder = iter(args.noninteractive.split(";") + ["/quit"])
        builtins.input = lambda prompt="": next(feeder)
        repl(agent, sampler, judge, args)
        return 0

    print(f"loaded {n} KB facts from {args.kb}")
    repl(agent, sampler, judge, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
