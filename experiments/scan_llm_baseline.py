"""Gap-1c: a real pretrained LLM, few-shot, on real SCAN addprim_jump.

Externalizes the baseline beyond our from-scratch transformer: prompt a local
pretrained LLM (Ollama) with few-shot IN->OUT examples (including the bare `jump`),
then ask for held-out `jump`-compositions. SCAN compositional splits are a
documented LLM weakness. Compare to the hybrid's 100% (Findings 21-22) and
pure_vsa's 100%.

Honest scope: a small/mid LOCAL model (CPU/iGPU), not a frontier API model with
heavy chain-of-thought; that stronger baseline is the remaining gap-1c-full step.

  python experiments/scan_llm_baseline.py --model llama3.2:3b --n 40 --shots 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _result_io import emit_result  # noqa: E402
from pure_vsa.scan_runner import load_scan_split

OPENROUTER_MAX_TOKENS = 300


def ollama(model: str, prompt: str) -> str:
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                         "options": {"temperature": 0.0, "num_predict": 160}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["response"]


def openrouter(model: str, prompt: str, max_tokens: int = OPENROUTER_MAX_TOKENS,
               retries: int = 3, timeout: int = 90) -> tuple[str, bool]:
    """Call OpenRouter's chat/completions endpoint. Returns (text, ok).

    ok=False if the response is empty/refused after all retries are exhausted;
    the key is read from OPENROUTER_API_KEY and never logged or returned.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()

    last_err = "unknown error"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content and content.strip():
                return content, True
            last_err = "empty response"
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:
                last_err = f"HTTP {e.code}"
            else:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = str(e)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    print(f"  [openrouter] giving up after {retries} attempts: {last_err}", file=sys.stderr)
    return "", False


def parse_actions(resp: str) -> list[str]:
    """Robust: extract the action tokens (I_*), ignoring any prose wrapper."""
    return [t for t in resp.replace(",", " ").split() if t.startswith("I_")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama", "openrouter"], default="ollama",
                    help="ollama: local pretrained LLM (default, unchanged behavior). "
                         "openrouter: frontier API model via OpenRouter (needs OPENROUTER_API_KEY).")
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--limit", type=int, default=None,
                    help="alias for --n (eval on first N examples, deterministic order)")
    p.add_argument("--shots", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="write contract result JSON to this path")
    a = p.parse_args()
    if a.limit is not None:
        a.n = a.limit
    rng = random.Random(a.seed)

    base = Path(__file__).resolve().parents[1] / "data" / "scan" / "addprim_jump"
    train = load_scan_split(base / "train.txt")
    test = load_scan_split(base / "test.txt")

    # few-shot: the bare jump example + diverse train examples
    shots = [(i, o) for (i, o) in train if i == "jump"][:1]
    pool = [(i, o) for (i, o) in train if i != "jump"]
    rng.shuffle(pool)
    shots += pool[: a.shots - len(shots)]
    preamble = ("Map each command to its action sequence. Reply with ONLY the "
                "space-separated action tokens (each starts with I_), no other text.\n\n"
                + "\n".join(f"IN: {i} OUT: {' '.join(o)}" for i, o in shots) + "\n")

    test_sample = test[:]
    rng.shuffle(test_sample)
    test_sample = test_sample[: a.n]
    ok = 0
    errors = 0
    t0 = time.time()
    for k, (inp, gold) in enumerate(test_sample, 1):
        prompt = preamble + f"IN: {inp} OUT:"
        if a.backend == "openrouter":
            resp, call_ok = openrouter(a.model, prompt, max_tokens=OPENROUTER_MAX_TOKENS)
            if not call_ok:
                errors += 1
        else:
            resp = ollama(a.model, prompt)
        pred = parse_actions(resp)
        if pred == gold:
            ok += 1
        if k <= 3:
            print(f"  IN: {inp}\n    gold: {' '.join(gold[:10])}{'...' if len(gold)>10 else ''}\n"
                  f"    llm:  {' '.join(pred[:10])}{'...' if len(pred)>10 else ''}  {'OK' if pred==gold else 'X'}")
    eval_s = time.time() - t0
    print(f"\n{a.model}  few-shot={a.shots}  held-out EM = {ok}/{len(test_sample)} = "
          f"{ok/len(test_sample)*100:.1f}%   ({eval_s:.0f}s)")
    print("(hybrid Findings 21-22 = 100%, pure_vsa = 100%, from-scratch transformer = 0%)")

    if a.out:
        if a.backend == "openrouter":
            notes = ("frontier API model via OpenRouter, direct few-shot mapping (same "
                      "prompt as local rows), no decomposition/least-to-most prompting - "
                      "see cited rows for that technique")
        else:
            notes = "local pretrained LLM (Ollama), few-shot prompting, real held-out addprim_jump"
        emit_result(
            a.out,
            bench="scan_addprim_jump",
            system=f"llm_{a.model}",
            split="test_heldout_jump",
            n=len(test_sample), correct=ok,
            params=0, train_s=0.0, eval_s=eval_s, seed=a.seed,
            config={"backend": a.backend, "model": a.model, "n": a.n, "shots": a.shots,
                    "test_limit": a.n, "errors": errors},
            notes=notes,
        )


if __name__ == "__main__":
    main()
