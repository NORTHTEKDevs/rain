# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Preference fine-tuning for HYMN-Plus v2 using the local Ollama judge.

DPO (Direct Preference Optimization) without an explicit reward model.
For each prompt:

  1. Sample TWO completions from the model at different temperatures
  2. Ask the local Ollama judge which is better
  3. Build a (prompt, chosen, rejected) tuple
  4. Do one DPO step:
       loss = -log(sigmoid(beta * (log_pi_chosen - log_pi_rejected)
                                  - (log_ref_chosen - log_ref_rejected)))
     where pi = current policy and ref = frozen reference (copy at start)

Output: a HYMN-Plus v2 checkpoint fine-tuned for whatever the judge
prefers. Beats vanilla SFT in many cases because preferences are denser
signal than next-token likelihood.

Honest scope:
  * Reference model is loaded once and kept frozen
  * Single-GPU/CPU, single-batch DPO (one preference pair per step)
  * Judge runs locally via Ollama; default llama3.2:3b
  * Prompt source is a JSONL of {"prompt": "..."} lines or a corpus

Usage:
    python -m scripts.preference_finetune \
        --base data/checkpoints/hymn_plus_v5_qa.npz \
        --prompts data/prompts/dpo_seeds.jsonl \
        --judge-model llama3.2:3b \
        --steps 200 --beta 0.1 \
        --out data/checkpoints/hymn_plus_v5_dpo.npz
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _load_v2(ckpt_path: Path):
    from rain.core.hymn_plus_v2 import HymnPlusV2, HymnPlusV2Config
    from rain.tokenize.bpe import BPETokenizer

    meta = json.loads(ckpt_path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("arch") != "hymn_plus_v2":
        raise ValueError(f"checkpoint arch={meta.get('arch')!r}; need hymn_plus_v2")
    cfg = HymnPlusV2Config(
        vocab_size=meta["vocab_size"],
        dim=meta["dim"],
        n_layers=meta["n_layers"],
        mlp_mult=meta["mlp_mult"],
        kb_size=meta["kb_size"],
        kb_top_k=meta["kb_top_k"],
        kb_attn_in_layers=(
            tuple(meta["kb_attn_in_layers"]) if meta.get("kb_attn_in_layers") else None
        ),
        tie_weights=meta.get("tie_weights", True),
        seed=meta.get("seed", 42),
    )
    model = HymnPlusV2(cfg)
    sd = dict(np.load(ckpt_path))
    model.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
    bpe_path = meta.get("bpe_model_path") or str(ckpt_path.with_suffix(".bpe.model"))
    tok = BPETokenizer(vocab_size=meta.get("bpe_vocab", cfg.vocab_size))
    tok.load(bpe_path)
    return model, tok, meta


def _sequence_logp(model, ids: torch.Tensor) -> torch.Tensor:
    """Sum of log-probabilities of `ids` under `model` (autoregressive).
    ids: (1, T) -- the FULL sequence (prompt + completion).
    Returns a scalar tensor.
    """
    logits, _ = model(ids[:, :-1])
    log_probs = F.log_softmax(logits, dim=-1)
    targets = ids[:, 1:]
    chosen = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return chosen.sum()


def _judge_pair(judge_model: str, prompt: str, a: str, b: str) -> str:
    """Ask Ollama which of A or B is a better completion. Returns 'a', 'b',
    or 'tie'. Tie if the judge is unsure or unreachable."""
    import urllib.error
    import urllib.request

    sys_prompt = (
        "You are a strict completion judge. Given a prompt and two candidate "
        "completions A and B, output exactly one of: A, B, TIE. No other text."
    )
    user = (
        f"PROMPT:\n{prompt}\n\n"
        f"COMPLETION A:\n{a}\n\n"
        f"COMPLETION B:\n{b}\n\n"
        f"Output exactly A, B, or TIE."
    )
    payload = {
        "model": judge_model,
        "prompt": sys_prompt + "\n\n" + user,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 8},
    }
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read().decode("utf-8"))
        verdict = out.get("response", "").strip().upper()
        if verdict.startswith("A"):
            return "a"
        if verdict.startswith("B"):
            return "b"
        return "tie"
    except (urllib.error.URLError, TimeoutError, OSError):
        return "tie"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="v2 checkpoint to fine-tune")
    p.add_argument(
        "--prompts", required=True, help="JSONL of {'prompt': str} or .txt one-prompt-per-line"
    )
    p.add_argument("--judge-model", default="llama3.2:3b")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--temp-a", type=float, default=0.3)
    p.add_argument("--temp-b", type=float, default=1.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"loading base checkpoint: {args.base}")
    pi, tok, meta = _load_v2(Path(args.base))
    print(f"  dim={meta['dim']} layers={meta['n_layers']} kb_size={meta['kb_size']}")

    # Frozen reference -- deep copy of the initial weights
    ref = copy.deepcopy(pi)
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()
    pi.train()

    opt = torch.optim.AdamW(pi.parameters(), lr=args.lr)

    # Load prompts
    prompts_path = Path(args.prompts)
    prompts: list[str] = []
    if prompts_path.suffix == ".jsonl":
        with prompts_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if "prompt" in r:
                        prompts.append(r["prompt"])
                except json.JSONDecodeError:
                    continue
    else:
        prompts = [
            ln.strip() for ln in prompts_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
    print(f"  prompts: {len(prompts)}")

    losses: list[float] = []
    chosen_a = 0
    chosen_b = 0
    tied = 0
    start = time.time()

    for step in range(args.steps):
        prompt = prompts[step % len(prompts)]
        # Encode prompt
        p_ids = tok.encode(prompt)
        if not p_ids:
            continue
        p_t = torch.tensor([p_ids], dtype=torch.long)

        # Sample A (conservative) and B (creative)
        pi.eval()
        with torch.no_grad():
            out_a = pi.sample(p_t, max_new=args.max_new, temperature=args.temp_a, top_k=30)
            out_b = pi.sample(p_t, max_new=args.max_new, temperature=args.temp_b, top_k=50)
        a_text = tok.decode(out_a[0, len(p_ids) :].tolist())
        b_text = tok.decode(out_b[0, len(p_ids) :].tolist())

        # Judge
        verdict = _judge_pair(args.judge_model, prompt, a_text, b_text)
        if verdict == "tie":
            tied += 1
            continue
        if verdict == "a":
            chosen_a += 1
            chosen_ids, rejected_ids = out_a, out_b
        else:
            chosen_b += 1
            chosen_ids, rejected_ids = out_b, out_a

        pi.train()
        # DPO loss
        with torch.no_grad():
            ref_chosen_logp = _sequence_logp(ref, chosen_ids)
            ref_rejected_logp = _sequence_logp(ref, rejected_ids)
        pi_chosen_logp = _sequence_logp(pi, chosen_ids)
        pi_rejected_logp = _sequence_logp(pi, rejected_ids)

        logits = args.beta * (
            (pi_chosen_logp - pi_rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
        )
        loss = -F.logsigmoid(logits)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

        if (step + 1) % 10 == 0:
            avg = sum(losses[-10:]) / max(1, len(losses[-10:]))
            print(
                f"step {step + 1}/{args.steps}  avg_loss={avg:.4f}  "
                f"votes A={chosen_a} B={chosen_b} TIE={tied}"
            )

    wall = time.time() - start

    # Save fine-tuned checkpoint
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().numpy() for k, v in pi.state_dict().items()}
    np.savez_compressed(out, **sd)

    # Reuse the BPE sidecar
    import shutil

    base_bpe = Path(args.base).with_suffix(".bpe.model")
    if base_bpe.is_file():
        shutil.copy(base_bpe, out.with_suffix(".bpe.model"))

    new_meta = {
        **meta,
        "arch": "hymn_plus_v2",
        "dpo_finetuned": True,
        "dpo_steps": args.steps,
        "dpo_beta": args.beta,
        "dpo_lr": args.lr,
        "dpo_judge_model": args.judge_model,
        "dpo_votes": {"a": chosen_a, "b": chosen_b, "tie": tied},
        "dpo_wall_seconds": wall,
        "dpo_mean_loss": (sum(losses) / len(losses)) if losses else None,
    }
    out.with_suffix(".json").write_text(json.dumps(new_meta, indent=2, default=str))
    final_loss = (sum(losses[-10:]) / len(losses[-10:])) if losses else float("nan")
    print(
        f"\nsaved {out} + sidecars"
        f"\nDPO summary: votes A={chosen_a} B={chosen_b} TIE={tied}"
        f"  steps_with_loss={len(losses)}  final_avg_loss={final_loss:.4f}"
        f"  wall={wall:.1f}s"
    )
    if math.isnan(final_loss):
        print("WARN: no preference pairs produced loss (judge always tied or unreachable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
