"""Grammar-rule induction from OUTPUT ALONE (gap-1a frontier attempt).

The complement of F22 (which learned token categories but used the hand-coded
executor). Here the parse SKELETON is given (which token is verb/dir/spatial/
modifier/connective), but EVERY grammar rule -- the lexicon (verb->action,
dir->turn), the atom-expansion TEMPLATES per spatial form (the variable-length
SHAPE: which output slots are turns vs actions), the modifier counts, and the
connective order -- is INDUCED from (input, output) pairs only, via REINFORCE over
the rule parameters (an evolutionary/RL search; the rule-set search space is
small). No rule contents are given.

If the induced rule-set generalizes to held-out addprim_jump, the grammar rules
were learned from outputs alone. Honest boundary: the abstract rule TYPES (atoms
have a template; clauses repeat; connectives order) are given; discovering those
from scratch is the deeper LANE/NeSS-scale problem.

  python experiments/scan_grammar_induction.py --steps 3000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _result_io import emit_result  # noqa: E402
from pure_vsa.scan_runner import load_scan_split

VERBS = ["walk", "look", "run", "jump", "turn"]
DIRS = ["left", "right"]
SPATIAL = ["opposite", "around"]
MODS = ["twice", "thrice"]
CONJ = ["and", "after"]
OUT = ["I_WALK", "I_LOOK", "I_RUN", "I_JUMP", "I_TURN_LEFT", "I_TURN_RIGHT"]
CASES = ["plain", "dir", "opposite", "around"]   # atom-expansion cases
MAXATOM = 8
SLOT = ["ACTION", "TURN", "END"]


def parse_skeleton(inp: str):
    """Given categories (skeleton), parse into (clause1, clause2, conn)."""
    toks = inp.split()
    conn = next((t for t in toks if t in CONJ), None)
    def atom_clause(ts):
        verb = next((t for t in ts if t in VERBS), None)
        dirn = next((t for t in ts if t in DIRS), None)
        spat = next((t for t in ts if t in SPATIAL), None)
        mod = next((t for t in ts if t in MODS), None)
        case = spat if spat else ("dir" if dirn else "plain")
        return {"verb": verb, "dir": dirn, "case": case, "mod": mod}
    if conn:
        i = toks.index(conn)
        return atom_clause(toks[:i]), atom_clause(toks[i + 1:]), conn
    return atom_clause(toks), None, None


class Rules:
    """Sampled rule-set from the learnable logits."""
    def __init__(self, A, T, TM, CN, OR):
        self.A, self.T, self.TM, self.CN, self.OR = A, T, TM, CN, OR

    def atom_out(self, c):
        if c["verb"] is None:
            return []
        act = OUT[self.A[VERBS.index(c["verb"])]]
        turn = OUT[self.T[DIRS.index(c["dir"])]] if c["dir"] else None
        seq = []
        for s in self.TM[CASES.index(c["case"])]:
            if s == 2:        # END
                break
            if s == 0:        # ACTION
                seq.append(act)
            elif turn is not None:   # TURN
                seq.append(turn)
        return seq

    def clause_out(self, c):
        base = self.atom_out(c)
        cnt = (self.CN[MODS.index(c["mod"])] + 1) if c["mod"] else 1
        return base * cnt

    def execute(self, parse):
        c1, c2, conn = parse
        o1 = self.clause_out(c1)
        if c2 is None:
            return o1
        o2 = self.clause_out(c2)
        keep = (self.OR[CONJ.index(conn)] == 0)
        return o1 + o2 if keep else o2 + o1


def reward(pred, gold):
    if pred == gold:
        return 1.0
    if not gold:
        return 0.0
    m = sum(1 for a, b in zip(pred, gold) if a == b)
    return 0.4 * m / max(len(gold), len(pred), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="write contract result JSON to this path")
    a = p.parse_args()
    torch.manual_seed(a.seed)

    base = Path(__file__).resolve().parents[1] / "data" / "scan" / "addprim_jump"
    train = [(parse_skeleton(i), o) for i, o in load_scan_split(base / "train.txt")]
    test = [(parse_skeleton(i), o) for i, o in load_scan_split(base / "test.txt")]
    train.sort(key=lambda x: len(x[1]))                 # curriculum: simple first

    # learnable rule logits
    A = torch.zeros(len(VERBS), len(OUT), requires_grad=True)
    T = torch.zeros(len(DIRS), len(OUT), requires_grad=True)
    TM = torch.zeros(len(CASES), MAXATOM, 3, requires_grad=True)
    CN = torch.zeros(len(MODS), 3, requires_grad=True)
    OR = torch.zeros(len(CONJ), 2, requires_grad=True)
    params = [A, T, TM, CN, OR]
    opt = torch.optim.Adam(params, lr=a.lr)

    def sample():
        dists = {"A": torch.distributions.Categorical(logits=A),
                 "T": torch.distributions.Categorical(logits=T),
                 "TM": torch.distributions.Categorical(logits=TM),
                 "CN": torch.distributions.Categorical(logits=CN),
                 "OR": torch.distributions.Categorical(logits=OR)}
        s = {k: d.sample() for k, d in dists.items()}
        logp = sum(dists[k].log_prob(s[k]).sum() for k in s)
        rules = Rules(s["A"].tolist(), s["T"].tolist(), s["TM"].tolist(),
                      s["CN"].tolist(), s["OR"].tolist())
        return rules, logp

    def greedy_rules():
        return Rules(A.argmax(-1).tolist(), T.argmax(-1).tolist(), TM.argmax(-1).tolist(),
                     CN.argmax(-1).tolist(), OR.argmax(-1).tolist())

    def em(split, n):
        r = greedy_rules()
        ok = 0
        m = min(n, len(split))
        for parse, gold in split[:n]:
            try:
                ok += (r.execute(parse) == gold)
            except Exception:
                pass
        return ok, m

    # the lexicon-defining examples: each verb's bare/shortest occurrence. The
    # addprim_jump holdout makes `jump` appear only here, so these rules get almost
    # no RL signal unless surfaced. Force-include them every batch (still output-only).
    bare = {}
    for k, (parse, gold) in enumerate(train):
        c1, c2, _ = parse
        if c2 is None and c1["dir"] is None and c1["mod"] is None and c1["verb"] is not None:
            bare.setdefault(c1["verb"], k)
    bare_idx = list(bare.values())

    baseline = 0.0
    t0 = time.monotonic()
    te_ok = te_n = 0
    cur = 200                                            # curriculum window (grows)
    for step in range(1, a.steps + 1):
        hi = min(len(train), cur + step * 6)
        idx = torch.cat([torch.randint(0, hi, (a.batch - len(bare_idx),)),
                         torch.tensor(bare_idx)])
        logps, rewards = [], []
        for j in idx.tolist():
            parse, gold = train[j]
            rules, logp = sample()
            try:
                pred = rules.execute(parse)
            except Exception:
                pred = []
            logps.append(logp); rewards.append(reward(pred, gold))
        r = torch.tensor(rewards)
        baseline = 0.95 * baseline + 0.05 * r.mean().item()
        loss = -(torch.stack(logps) * (r - baseline)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == a.steps:
            tr_ok, tr_n = em(train, a.n_eval); te_ok, te_n = em(test, a.n_eval)
            print(f"  step {step:4d}  meanR={r.mean():.3f}  train_EM={tr_ok/tr_n*100:.1f}%  "
                  f"HELD-OUT_EM={te_ok/te_n*100:.1f}%  elapsed={time.monotonic()-t0:.0f}s", flush=True)
    train_s = time.monotonic() - t0
    print("DONE — grammar-RULE induction from output (skeleton given) on real addprim_jump.")

    if a.out:
        if te_n == 0:
            te_ok, te_n = em(test, a.n_eval)
        nparams = sum(p.numel() for p in params)
        emit_result(
            a.out,
            bench="scan_addprim_jump",
            system="grammar_induction_reinforce",
            split="test_heldout_jump",
            n=te_n, correct=te_ok,
            params=nparams, train_s=train_s, eval_s=0.0, seed=a.seed,
            config={"steps": a.steps, "batch": a.batch, "n_eval": a.n_eval, "lr": a.lr},
            notes="grammar-rule (lexicon/templates/counts/order) induced via REINFORCE from "
                  "output alone; parse skeleton given, executor learned",
        )


if __name__ == "__main__":
    main()
