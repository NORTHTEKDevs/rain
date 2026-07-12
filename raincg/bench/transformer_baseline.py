"""From-scratch seq2seq Transformer baseline for PCFG SET."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass
class Vocab:
    stoi: dict[str, int] = field(default_factory=dict)
    itos: list[str] = field(default_factory=list)

    def encode(self, toks: list[str]) -> list[int]:
        u = self.stoi["<unk>"]
        return [self.stoi.get(t, u) for t in toks]


def build_vocab(pairs: list[tuple[str, list[str]]]) -> Vocab:
    toks: set[str] = set()
    for src, tgt in pairs:
        toks.update(src.split())
        toks.update(tgt)
    itos = list(SPECIALS) + sorted(toks)
    return Vocab(stoi={t: i for i, t in enumerate(itos)}, itos=itos)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=4, num_layers=2, ff=2048, max_len=820):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=ff, batch_first=True,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def _embed(self, x):
        L = x.size(1)
        return self.emb(x) * math.sqrt(self.d_model) + self.pos[:L]

    def forward(self, src, tgt_in, src_kpm, tgt_kpm):
        L = tgt_in.size(1)
        cmask = nn.Transformer.generate_square_subsequent_mask(L).to(tgt_in.device)
        h = self.transformer(
            self._embed(src), self._embed(tgt_in),
            tgt_mask=cmask,
            src_key_padding_mask=src_kpm, tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
        )
        return self.out(h)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def greedy_decode(model, vocab: Vocab, src_str: str, max_len: int = 820) -> list[str]:
    model.eval()
    dev = next(model.parameters()).device
    # cap source length to the model's positional budget
    src_ids = vocab.encode(src_str.split())[: model.max_len]
    src = torch.tensor([src_ids], device=dev)
    src_kpm = (src == 0)
    bos, eos = vocab.stoi["<bos>"], vocab.stoi["<eos>"]
    ys = torch.tensor([[bos]], device=dev)
    out: list[str] = []
    cap = min(max_len, model.max_len - 1)
    for _ in range(cap):
        logits = model(src, ys, src_kpm, (ys == 0))
        nxt = int(logits[0, -1].argmax())
        if nxt == eos:
            break
        out.append(vocab.itos[nxt])
        ys = torch.cat([ys, torch.tensor([[nxt]], device=dev)], dim=1)
    return out


@dataclass
class TrainResult:
    params: int
    epochs_run: int
    final_loss: float
    train_seconds: float


def _batchify(pairs, vocab, bos, eos, pad, device, max_len):
    src_ids = [vocab.encode(s.split())[:max_len] for s, _ in pairs]
    tgt_ids = [([bos] + vocab.encode(t) + [eos])[:max_len] for _, t in pairs]
    sm = max(len(x) for x in src_ids)
    tm = max(len(x) for x in tgt_ids)
    src = torch.full((len(pairs), sm), pad, dtype=torch.long)
    tgt = torch.full((len(pairs), tm), pad, dtype=torch.long)
    for i, x in enumerate(src_ids):
        src[i, : len(x)] = torch.tensor(x)
    for i, x in enumerate(tgt_ids):
        tgt[i, : len(x)] = torch.tensor(x)
    return src.to(device), tgt.to(device)


def train_baseline(pairs, *, d_model=512, nhead=4, num_layers=2, ff=2048,
                   epochs=20, lr=3e-4, batch_size=128,
                   ckpt_path="raincg_pcfg_baseline.pt", max_minutes=60.0,
                   seed=0, device="cpu", log_every=1, max_len=820,
                   resume=False) -> TrainResult:
    """Train with per-epoch checkpointing. If resume=True and ckpt_path exists,
    continues from the next epoch after the checkpointed one, with cumulative
    train_seconds preserved across the kill/resume boundary."""
    torch.manual_seed(seed)
    vocab = build_vocab(pairs)
    pad, bos, eos = vocab.stoi["<pad>"], vocab.stoi["<bos>"], vocab.stoi["<eos>"]
    model = Seq2SeqTransformer(len(vocab.stoi), d_model, nhead, num_layers, ff,
                              max_len=max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    elapsed_prior = 0.0
    final_loss = float("nan")
    if resume and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", 0) + 1
        elapsed_prior = ck.get("elapsed_train_s", 0.0)
        final_loss = ck.get("final_loss", float("nan"))

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)
    n = len(pairs)
    t0 = time.perf_counter()
    ep = start_epoch - 1
    for ep in range(start_epoch, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            batch = [pairs[j] for j in perm[i:i + batch_size].tolist()]
            src, tgt = _batchify(batch, vocab, bos, eos, pad, device, max_len)
            tin, tout = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tin, (src == pad), (tin == pad))
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        final_loss = total / max(nb, 1)
        elapsed_total = elapsed_prior + (time.perf_counter() - t0)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": ep, "elapsed_train_s": elapsed_total,
                    "final_loss": final_loss, "itos": vocab.itos,
                    "cfg": dict(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                ff=ff, max_len=max_len)},
                   ckpt_path)
        if ep % log_every == 0:
            print(f"  epoch {ep}/{epochs} loss={final_loss:.4f} "
                  f"elapsed={elapsed_total / 60:.1f}min", flush=True)
        if elapsed_total / 60.0 >= max_minutes:
            print(f"  hit max_minutes={max_minutes}; stopping at epoch {ep}", flush=True)
            break
    train_seconds = elapsed_prior + (time.perf_counter() - t0)
    return TrainResult(params=param_count(model), epochs_run=ep,
                       final_loss=final_loss, train_seconds=train_seconds)


def load_checkpoint(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab = Vocab(stoi={t: i for i, t in enumerate(ck["itos"])}, itos=ck["itos"])
    c = ck["cfg"]
    model = Seq2SeqTransformer(len(vocab.itos), c["d_model"], c["nhead"],
                               c["num_layers"], c["ff"],
                               max_len=c.get("max_len", 820)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, vocab
