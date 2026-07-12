"""Gap-1b/1c on a SECOND real benchmark: a pretrained LLM on COGS gen split.

A from-scratch transformer on COGS is a GPU job (544-token logical-form
sequences). But a local pretrained LLM few-shot is CPU-feasible. COGS gen is the
compositional-generalization split (prim_to_subj_proper, obj_to_subj, etc.).
Measure exact logical-form match. Documented to be hard for LLMs.

  python experiments/cogs_llm_baseline.py --model llama3.2:3b --n 30 --shots 12
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _result_io import emit_result  # noqa: E402

OPENROUTER_MAX_TOKENS = 600


def load_tsv(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip() if len(parts) > 2 else ""))
    return rows


def ollama(model: str, prompt: str) -> str:
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                         "options": {"temperature": 0.0, "num_predict": 200}}).encode(),
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


def norm(s: str) -> str:
    return " ".join(s.replace("\n", " ").split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama", "openrouter"], default="ollama",
                    help="ollama: local pretrained LLM (default, unchanged behavior). "
                         "openrouter: frontier API model via OpenRouter (needs OPENROUTER_API_KEY).")
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--limit", type=int, default=None,
                    help="alias for --n (eval on first N examples, deterministic order)")
    p.add_argument("--shots", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="write contract result JSON to this path")
    a = p.parse_args()
    if a.limit is not None:
        a.n = a.limit
    rng = random.Random(a.seed)

    base = Path(__file__).resolve().parents[1] / "data" / "cogs" / "raw"
    train = load_tsv(base / "train.tsv")
    gen = load_tsv(base / "gen.tsv")
    rng.shuffle(train)
    shots = train[: a.shots]
    preamble = ("Translate each sentence into its logical form. Reply with ONLY the "
                "logical form, no other text.\n\n"
                + "\n".join(f"SENT: {s} LF: {lf}" for s, lf, _ in shots) + "\n")

    rng.shuffle(gen)
    sample = gen[: a.n]
    ok = 0
    errors = 0
    by_cond: dict[str, list[int]] = {}
    t0 = time.time()
    for k, (sent, gold, cond) in enumerate(sample, 1):
        prompt = preamble + f"SENT: {sent} LF:"
        if a.backend == "openrouter":
            resp, call_ok = openrouter(a.model, prompt, max_tokens=OPENROUTER_MAX_TOKENS)
            if not call_ok:
                errors += 1
        else:
            resp = ollama(a.model, prompt)
        pred = norm(resp)
        # strip a leading "LF:" if echoed
        if pred.upper().startswith("LF:"):
            pred = pred[3:].strip()
        hit = int(pred == norm(gold))
        ok += hit
        by_cond.setdefault(cond, [0, 0])
        by_cond[cond][0] += hit; by_cond[cond][1] += 1
        if k <= 2:
            print(f"  SENT: {sent}\n    gold: {norm(gold)[:90]}...\n    llm:  {pred[:90]}...  {'OK' if hit else 'X'}")
    eval_s = time.time() - t0
    print(f"\n{a.model}  COGS gen  few-shot={a.shots}  EM = {ok}/{len(sample)} = "
          f"{ok/len(sample)*100:.1f}%   ({eval_s:.0f}s)")
    hard = [c for c in by_cond if by_cond[c][1] >= 2]
    if hard:
        print("by condition (>=2 samples):  " +
              "  ".join(f"{c}:{by_cond[c][0]}/{by_cond[c][1]}" for c in hard[:6]))
    print("(pure_vsa COGS: in-dist ~99.7%, gen ~73.9% on the 6 covered constructions)")

    if a.out:
        if a.backend == "openrouter":
            notes = ("frontier API model via OpenRouter, direct few-shot mapping (same "
                      "prompt as local rows), no decomposition/least-to-most prompting - "
                      "see cited rows for that technique")
        else:
            notes = "local pretrained LLM (Ollama), few-shot prompting, real COGS gen split"
        emit_result(
            a.out,
            bench="cogs_gen",
            system=f"llm_{a.model}",
            split="gen",
            n=len(sample), correct=ok,
            params=0, train_s=0.0, eval_s=eval_s, seed=a.seed,
            config={"backend": a.backend, "model": a.model, "n": a.n, "shots": a.shots,
                    "test_limit": a.n, "errors": errors,
                    "by_condition": {c: v[0] / v[1] for c, v in by_cond.items() if v[1] >= 2}},
            notes=notes,
        )


if __name__ == "__main__":
    main()
