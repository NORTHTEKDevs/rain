"""Gap-1b hybrid: NEURAL perception + EXACT composition on REAL SCAN addprim_jump.

Finding 20 showed a free transformer gets 0% on the real held-out split. This is
the hybrid: a learned NEURAL tagger reads the raw input tokens and predicts each
token's grammatical category (verb / direction / spatial / modifier / connective);
a deterministic assembler turns those tags into the exact SCAN parse
(pure_vsa.scan_runner Atom/Clause/ParsedSCAN); the EXACT symbolic executor
(`ParsedSCAN.expected_output`, verified at 100% with the hand parser) produces the
output. Perception is learned by gradient; composition stays exact.

Trained ONLY on addprim_jump train (where `jump` appears solely as the bare atom).
The tagger learns `jump -> verb` (a per-token-type fact) and so generalizes to all
held-out `jump`-compositions; the exact executor then composes them correctly.
Expected: high held-out EM where the free transformer (Finding 20) got 0%.

  python experiments/scan_hybrid.py --steps 1500
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from pure_vsa.scan_runner import (
    Atom, Clause, ParsedSCAN, load_scan_split,
    VERBS, MODIFIERS, SPATIAL, CONJUNCTIONS,
)
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _result_io import emit_result  # noqa: E402

DIRECTIONS = ["left", "right"]
# categories: 0 verb, 1 direction, 2 spatial, 3 modifier, 4 connective
CATS = {**{v: 0 for v in VERBS}, **{d: 1 for d in DIRECTIONS},
        **{s: 2 for s in SPATIAL}, **{m: 3 for m in MODIFIERS},
        **{c: 4 for c in CONJUNCTIONS}}
MAXLEN = 12


def category(tok: str) -> int:
    return CATS.get(tok, 0)


class Tagger(nn.Module):
    """Neural token-category tagger. per_token=True -> content-independent
    (token embedding only, no context); False -> a context-dependent transformer
    encoder. The thesis predicts only the content-independent one generalizes."""
    def __init__(self, vocab: int, d: int = 64, nhead: int = 4, layers: int = 2,
                 per_token: bool = False):
        super().__init__()
        self.per_token = per_token
        self.emb = nn.Embedding(vocab + 1, d)        # +1 pad id = vocab
        if per_token:
            self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        else:
            self.pos = nn.Embedding(MAXLEN, d)
            layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=128,
                                               batch_first=True, dropout=0.0, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 5)
        self.vocab = vocab

    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        if self.per_token:                            # context-free: per-token only
            return self.head(self.net(self.emb(ids)))
        L = ids.shape[1]
        h = self.emb(ids) + self.pos(torch.arange(L))
        return self.head(self.enc(h, src_key_padding_mask=pad_mask))

    @torch.no_grad()
    def tag(self, ids_row: list[int]) -> list[int]:
        ids = torch.tensor([ids_row], dtype=torch.long)
        mask = torch.zeros(1, len(ids_row), dtype=torch.bool)
        return self.forward(ids, mask)[0].argmax(-1).tolist()


def assemble(tokens: list[str], cats: list[int]) -> ParsedSCAN:
    def clause(toks: list[str], cs: list[int]) -> Clause:
        verb = direction = spatial = modifier = None
        for t, c in zip(toks, cs):
            if c == 0:
                verb = t
            elif c == 1:
                direction = t
            elif c == 2:
                spatial = t
            elif c == 3:
                modifier = t
        return Clause(Atom(verb or "walk", direction, spatial), modifier)

    conn_pos = [i for i, c in enumerate(cats) if c == 4]
    if conn_pos:
        p = conn_pos[0]
        return ParsedSCAN(clause(tokens[:p], cats[:p]),
                          clause(tokens[p + 1:], cats[p + 1:]), tokens[p])
    return ParsedSCAN(clause(tokens, cats))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="write contract result JSON to this path")
    a = p.parse_args()
    torch.manual_seed(a.seed)

    base = Path(__file__).resolve().parents[1] / "data" / "scan" / "addprim_jump"
    train = load_scan_split(base / "train.txt")
    test = load_scan_split(base / "test.txt")
    vocab = {tok: i for i, tok in enumerate(sorted(CATS))}
    PAD = len(vocab)

    def encode(inp: str):
        toks = inp.split()[:MAXLEN]
        ids = [vocab.get(t, 0) for t in toks]
        return toks, ids

    # training tensors (pad to MAXLEN)
    N = len(train)
    Xids = torch.full((N, MAXLEN), PAD, dtype=torch.long)
    Ycat = torch.full((N, MAXLEN), -100, dtype=torch.long)
    pad = torch.ones(N, MAXLEN, dtype=torch.bool)
    for i, (inp, _out) in enumerate(train):
        toks, ids = encode(inp)
        for k in range(len(ids)):
            Xids[i, k] = ids[k]; Ycat[i, k] = category(toks[k]); pad[i, k] = False
    print(f"REAL addprim_jump hybrid: train={N} test={len(test)} vocab={len(vocab)}")

    def run_mode(per_token: bool):
        torch.manual_seed(a.seed)
        model = Tagger(len(vocab), per_token=per_token)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        nparams = sum(p.numel() for p in model.parameters())

        def em(split, n):
            ok = 0
            m = min(n, len(split))
            for inp, gold in split[:n]:
                toks, ids = encode(inp)
                cats = model.tag(ids)[:len(toks)]
                try:
                    pred = assemble(toks, cats).expected_output()
                except Exception:
                    pred = []
                ok += (pred == gold)
            return ok, m

        for step in range(1, a.steps + 1):
            idx = torch.randint(0, N, (a.batch,))
            logits = model(Xids[idx], pad[idx])
            loss = F.cross_entropy(logits.reshape(-1, 5), Ycat[idx].reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        tr_ok, tr_n = em(train, a.n_eval)
        te_ok, te_n = em(test, a.n_eval)
        return tr_ok, tr_n, te_ok, te_n, nparams

    t0 = time.monotonic()
    tt_tr_ok, tt_tr_n, tt_te_ok, tt_te_n, tt_params = run_mode(per_token=False)   # context-dependent
    pt_tr_ok, pt_tr_n, pt_te_ok, pt_te_n, pt_params = run_mode(per_token=True)    # content-indep.
    tt_tr, tt_te = tt_tr_ok / tt_tr_n, tt_te_ok / tt_te_n
    pt_tr, pt_te = pt_tr_ok / pt_tr_n, pt_te_ok / pt_te_n
    train_s = time.monotonic() - t0
    print(f"{'tagger':>28} {'train_EM':>9} {'HELD-OUT_EM':>12}")
    print(f"{'transformer (context-dependent)':>28} {tt_tr*100:8.1f}% {tt_te*100:11.1f}%")
    print(f"{'per-token (content-indep.)':>28} {pt_tr*100:8.1f}% {pt_te*100:11.1f}%")
    print(f"(free transformer, Finding 20: 0% held-out)   elapsed={train_s:.0f}s")

    if a.out:
        notes = "hybrid: supervised per-token tagger (content-independent) + exact symbolic executor"
        if pt_te_n >= len(test):
            notes += (
                f"; evaluated on the full {len(test)}-example test split "
                "(not a subsample); train_s/eval_s may be inflated by a concurrent "
                "PCFG transformer training job sharing this machine's CPU"
            )
        # content-independent per-token tagger is the one the thesis predicts generalizes
        emit_result(
            a.out,
            bench="scan_addprim_jump",
            system="hybrid_tagger_supervised",
            split="test_heldout_jump",
            n=pt_te_n, correct=pt_te_ok,
            params=pt_params, train_s=train_s, eval_s=0.0, seed=a.seed,
            config={"steps": a.steps, "batch": a.batch, "n_eval": a.n_eval,
                    "per_token": True,
                    "context_dependent_train_EM": tt_tr, "context_dependent_test_EM": tt_te,
                    "context_dependent_params": tt_params},
            notes=notes,
        )


if __name__ == "__main__":
    main()
