"""Gap-1b: a from-scratch transformer on REAL SCAN addprim_jump.

A standard decoder-only transformer trained on the actual `addprim_jump`
split (14,670 train / 7,706 held-out jump-compositions), tokenized directly
from the raw SCAN text files (see data/scan/addprim_jump/{train,test}.txt),
using the same tokenization scheme as pure_vsa's SCAN pipeline — only the
model differs.

Expected (the documented transformer failure, on the real benchmark): the
transformer fits the train distribution but gets ~0% exact-match on the
held-out compositional split, while pure_vsa is verified at 100% on the same
split. CPU, time-boxed; evals every 500 steps so partial results survive.

This module is self-contained: it tokenizes data/scan/addprim_jump/train.txt
and test.txt directly (no cached .bin/meta.json, no dependency on any
sequence-model package outside this file).

  python -m experiments.scan_transformer_baseline --steps 3000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments._result_io import emit_result

MAXPOS = 64

# Special tokens (low IDs so they don't collide with the small natural
# vocab). Order matters: it determines token ids, which must stay stable
# across runs so results are reproducible.
SPECIALS = ["<PAD>", "<BOS>", "<SEP>", "<EOS>", "<UNK>"]


def _parse_lines(raw: str) -> list[tuple[list[str], list[str]]]:
    """Parse 'IN: a b c OUT: X Y Z' into (input_tokens, output_tokens) pairs."""
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if "IN:" not in line or "OUT:" not in line:
            continue
        in_part, out_part = line.split("OUT:", 1)
        in_tokens = in_part.replace("IN:", "").strip().split()
        out_tokens = out_part.strip().split()
        out.append((in_tokens, out_tokens))
    return out


def _build_vocab(examples: list[tuple[list[str], list[str]]]) -> dict[str, int]:
    """Build a unified vocab. Input and output tokens share a namespace; specials first."""
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    next_id = len(vocab)
    for in_tok, out_tok in examples:
        for t in in_tok + out_tok:
            if t not in vocab:
                vocab[t] = next_id
                next_id += 1
    return vocab


def _encode(examples: list[tuple[list[str], list[str]]], vocab: dict[str, int]) -> list[list[int]]:
    """Encode each example as <BOS> in_tokens <SEP> out_tokens <EOS>."""
    bos, sep, eos, unk = vocab["<BOS>"], vocab["<SEP>"], vocab["<EOS>"], vocab["<UNK>"]
    seqs = []
    for in_tok, out_tok in examples:
        ids = [bos]
        ids += [vocab.get(t, unk) for t in in_tok]
        ids.append(sep)
        ids += [vocab.get(t, unk) for t in out_tok]
        ids.append(eos)
        seqs.append(ids)
    return seqs


def _pad_and_pack(seqs: list[list[int]], pad_id: int, max_len: int) -> np.ndarray:
    arr = np.full((len(seqs), max_len), pad_id, dtype=np.uint16)
    for i, s in enumerate(seqs):
        L = min(len(s), max_len)
        arr[i, :L] = s[:L]
    return arr


def load_scan_simple(data_dir: Path):
    """Tokenize data_dir/{train,test}.txt directly, matching the tokenization
    scheme recorded in the original meta.json (specials first, then tokens in
    first-seen order across train+test).
    """
    train_ex = _parse_lines((data_dir / "train.txt").read_text(encoding="utf-8"))
    test_ex = _parse_lines((data_dir / "test.txt").read_text(encoding="utf-8"))

    vocab = _build_vocab(train_ex + test_ex)
    train_enc = _encode(train_ex, vocab)
    test_enc = _encode(test_ex, vocab)
    max_len = max(max(len(s) for s in train_enc), max(len(s) for s in test_enc))

    pad_id = vocab["<PAD>"]
    train_arr = _pad_and_pack(train_enc, pad_id=pad_id, max_len=max_len)
    test_arr = _pad_and_pack(test_enc, pad_id=pad_id, max_len=max_len)

    meta = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "max_len": int(max_len),
        "n_train": int(train_arr.shape[0]),
        "n_test": int(test_arr.shape[0]),
        "pad_id": int(pad_id),
        "bos_id": int(vocab["<BOS>"]),
        "sep_id": int(vocab["<SEP>"]),
        "eos_id": int(vocab["<EOS>"]),
    }
    return train_arr, test_arr, meta


def build_targets(seqs: np.ndarray, sep_id: int, pad_id: int) -> np.ndarray:
    targets = np.full(seqs.shape, fill_value=-100, dtype=np.int64)
    for i in range(seqs.shape[0]):
        row = seqs[i]
        sep_arr = np.where(row == sep_id)[0]
        if len(sep_arr) == 0:
            continue
        sep_idx = sep_arr[0]
        for t in range(sep_idx, len(row) - 1):
            nxt = row[t + 1]
            if nxt == pad_id:
                break
            targets[i, t] = nxt
    return targets


def exact_match(model, test_seqs: np.ndarray, meta: dict,
                 n_examples: int = 100, max_new: int = 40) -> dict:
    model.eval()
    sep_id, eos_id, pad_id = meta["sep_id"], meta["eos_id"], meta["pad_id"]
    n_correct = 0
    n_total = 0
    for idx in range(min(n_examples, test_seqs.shape[0])):
        row = torch.tensor(test_seqs[idx], dtype=torch.long)
        sep_arr = (row == sep_id).nonzero(as_tuple=True)[0]
        eos_arr = (row == eos_id).nonzero(as_tuple=True)[0]
        if len(sep_arr) == 0 or len(eos_arr) == 0:
            continue
        sep_pos, eos_pos = int(sep_arr[0]), int(eos_arr[0])
        prompt = row[: sep_pos + 1]
        gt_out = row[sep_pos + 1 : eos_pos + 1]
        with torch.no_grad():
            generated = model.generate(prompt, max_new=max_new, eos_id=eos_id)
        pred_out = generated[len(prompt):]
        eos_in_pred = (pred_out == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_in_pred) > 0:
            pred_out = pred_out[: int(eos_in_pred[0]) + 1]
        if pred_out.tolist() == gt_out.tolist():
            n_correct += 1
        n_total += 1
    return {"em": n_correct / max(1, n_total),
            "n_correct": n_correct, "n_total": n_total}


class TinyLM(nn.Module):
    """Decoder-only transformer LM (causal) over BOS in SEP out EOS sequences."""
    def __init__(self, vocab: int, d: int = 128, nhead: int = 4, layers: int = 3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(MAXPOS, d)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=4 * d,
                                           batch_first=True, dropout=0.0, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B,T) -> (B,T,V)
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T))
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        return self.head(self.enc(h, mask=mask, is_causal=False))

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int = 55, eos_id: int = 3) -> torch.Tensor:
        seq = prompt.tolist()
        for _ in range(max_new):
            x = torch.tensor(seq[-MAXPOS:], dtype=torch.long).unsqueeze(0)
            nxt = int(self(x)[0, -1].argmax())
            seq.append(nxt)
            if nxt == eos_id:
                break
        return torch.tensor(seq, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="write contract result JSON to this path")
    a = p.parse_args()
    torch.manual_seed(a.seed)

    data_dir = Path(__file__).resolve().parents[1] / "data" / "scan" / "addprim_jump"
    train_seqs, test_seqs, meta = load_scan_simple(data_dir)
    V, pad_id, sep_id = meta["vocab_size"], meta["pad_id"], meta["eos_id"]
    # truncate to MAXPOS positions
    train_seqs = train_seqs[:, :MAXPOS].astype(np.int64)
    test_seqs = test_seqs[:, :MAXPOS].astype(np.int64)
    targets = build_targets(train_seqs, meta["sep_id"], meta["pad_id"])
    X = torch.tensor(train_seqs); Y = torch.tensor(targets)
    n = X.shape[0]
    print(f"REAL addprim_jump: train={n} test={test_seqs.shape[0]} vocab={V}")

    model = TinyLM(V, d=a.d, layers=a.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"transformer params={nparams:,}  steps={a.steps} batch={a.batch}")
    t0 = time.monotonic()
    tr = te = None
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, n, (a.batch,))
        logits = model(X[idx])
        loss = F.cross_entropy(logits.reshape(-1, V), Y[idx].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == a.steps:
            model.eval()
            tr = exact_match(model, train_seqs[:a.n_eval], meta, n_examples=a.n_eval, max_new=55)
            te = exact_match(model, test_seqs[:a.n_eval], meta, n_examples=a.n_eval, max_new=55)
            print(f"  step {step:4d}  ce={loss.item():.3f}  "
                  f"train_EM={tr['em']*100:.1f}% ({tr['n_correct']}/{tr['n_total']})  "
                  f"HELD-OUT_EM={te['em']*100:.1f}% ({te['n_correct']}/{te['n_total']})  "
                  f"elapsed={time.monotonic()-t0:.0f}s", flush=True)
            model.train()
    train_s = time.monotonic() - t0
    print("DONE — transformer on REAL addprim_jump; compare to pure_vsa = 100% (verified).")

    if a.out:
        emit_result(
            a.out,
            bench="scan_addprim_jump",
            system=f"transformer_d{a.d}x{a.layers}",
            split="test_heldout_jump",
            n=te["n_total"], correct=te["n_correct"],
            params=nparams, train_s=train_s, eval_s=0.0, seed=a.seed,
            config={"steps": a.steps, "batch": a.batch, "d": a.d, "layers": a.layers,
                    "lr": a.lr, "n_eval": a.n_eval},
            notes="from-scratch decoder-only transformer on REAL addprim_jump held-out split",
        )


if __name__ == "__main__":
    main()
