"""Gap-1b, output-only on REAL data: learn the tagger from outputs alone (RL).

Finding 21's hybrid hit 100% on real addprim_jump but used SUPERVISED per-token
category labels. This removes that hand-holding: the per-token tagger is trained
ONLY from gold action sequences -- no category labels -- via REINFORCE, with the
exact symbolic executor as a black-box reward (the NeSS/LANE approach). The tagger
is content-independent (a per-token-type category table, the thing Finding 21
showed is what generalizes); REINFORCE must discover which category each token
type should take so that exact execution reproduces the output.

Expected (if it converges): high held-out EM on real addprim_jump with NO
structural supervision. RL is finicky -- reported honestly either way.

  python experiments/scan_hybrid_outputonly.py --steps 4000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_hybrid import assemble, CATS, MAXLEN  # noqa: E402
from pure_vsa.scan_runner import load_scan_split  # noqa: E402
from _result_io import emit_result  # noqa: E402


def shaped_reward(pred: list[str], gold: list[str]) -> float:
    if pred == gold:
        return 1.0
    if not gold:
        return 0.0
    m = sum(1 for a, b in zip(pred, gold) if a == b)
    return 0.4 * m / max(len(gold), len(pred), 1)   # dense partial credit < exact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="write contract result JSON to this path")
    a = p.parse_args()
    torch.manual_seed(a.seed)

    base = Path(__file__).resolve().parents[1] / "data" / "scan" / "addprim_jump"
    train = load_scan_split(base / "train.txt")
    test = load_scan_split(base / "test.txt")
    vocab = {tok: i for i, tok in enumerate(sorted(CATS))}
    V = len(vocab)
    table = torch.zeros(V, 5, requires_grad=True)        # per-token-type category logits
    opt = torch.optim.Adam([table], lr=a.lr)

    def toks_ids(inp):
        t = inp.split()[:MAXLEN]
        return t, [vocab.get(x, 0) for x in t]

    @torch.no_grad()
    def greedy_em(split, n):
        ok = 0
        m = min(n, len(split))
        for inp, gold in split[:n]:
            t, ids = toks_ids(inp)
            cats = table[torch.tensor(ids)].argmax(-1).tolist()
            try:
                pred = assemble(t, cats).expected_output()
            except Exception:
                pred = []
            ok += (pred == gold)
        return ok, m

    baseline = 0.0
    t0 = time.monotonic()
    tr_ok = tr_n = te_ok = te_n = 0
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(train), (a.batch,))
        logps, rewards = [], []
        for j in idx.tolist():
            inp, gold = train[j]
            t, ids = toks_ids(inp)
            logits = table[torch.tensor(ids)]                # (L,5)
            dist = torch.distributions.Categorical(logits=logits)
            sample = dist.sample()                           # (L,)
            logps.append(dist.log_prob(sample).sum())
            try:
                pred = assemble(t, sample.tolist()).expected_output()
            except Exception:
                pred = []
            rewards.append(shaped_reward(pred, gold))
        r = torch.tensor(rewards)
        baseline = 0.95 * baseline + 0.05 * r.mean().item()
        adv = r - baseline
        loss = -(torch.stack(logps) * adv).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == a.steps:
            tr_ok, tr_n = greedy_em(train, a.n_eval); te_ok, te_n = greedy_em(test, a.n_eval)
            print(f"  step {step:4d}  meanR={r.mean():.3f}  baseline={baseline:.3f}  "
                  f"train_EM={tr_ok/tr_n*100:.1f}%  HELD-OUT_EM={te_ok/te_n*100:.1f}%  "
                  f"elapsed={time.monotonic()-t0:.0f}s", flush=True)
    train_s = time.monotonic() - t0
    print("DONE — OUTPUT-ONLY (REINFORCE) tagger + exact executor on REAL addprim_jump.")

    if a.out:
        if te_n == 0:   # steps too small to ever hit an eval checkpoint
            te_ok, te_n = greedy_em(test, a.n_eval)
        notes = ("content-independent per-token category table learned via REINFORCE "
                 "(no structural supervision) + exact symbolic executor")
        if te_n >= len(test):
            notes += (
                f"; evaluated on the full {len(test)}-example test split "
                "(not a subsample); train_s/eval_s may be inflated by a concurrent "
                "PCFG transformer training job sharing this machine's CPU"
            )
        emit_result(
            a.out,
            bench="scan_addprim_jump",
            system="hybrid_outputonly_reinforce",
            split="test_heldout_jump",
            n=te_n, correct=te_ok,
            params=table.numel(), train_s=train_s, eval_s=0.0, seed=a.seed,
            config={"steps": a.steps, "batch": a.batch, "n_eval": a.n_eval, "lr": a.lr},
            notes=notes,
        )


if __name__ == "__main__":
    main()
