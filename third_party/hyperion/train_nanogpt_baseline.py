"""Matched 85M-param nanoGPT-char baseline for Tiny Shakespeare.

Used to give the VSA-LM v0.2 experiment an honest comparison point. Same
data, same training budget, matched FLOPs. Per the v0.2 audit, do NOT
compare to the stock 10.65M nanoGPT-char -- match parameter count.

Architecture knob to hit ~85M params at vocab=65, seq=256:
  layers=6, n_head=8, n_embed=512   -> ~10M  (default nano-GPT char)
  layers=12, n_head=12, n_embed=768 -> ~85M  (matched)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).parent.resolve()


class CharDataset(Dataset):
    def __init__(self, ids: np.ndarray, seq_len: int) -> None:
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.ids) - self.seq_len - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.ids[idx : idx + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(seq[:-1]), torch.from_numpy(seq[1:])


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.n_embed = n_embed
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.proj = nn.Linear(n_embed, n_embed)
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, C // self.n_head)
        q, k, v = qkv.unbind(dim=2)                  # each (B, T, H, D/H)
        q = q.transpose(1, 2)                        # (B, H, T, D/H)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embed: int, n_head: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, seq_len, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, n_layer: int, n_head: int,
                 n_embed: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, n_embed))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head, seq_len, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok = self.tok_emb(idx)                        # (B, T, C)
        x = self.drop(tok + self.pos_emb[:, :T])
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)                          # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def evaluate(model: NanoGPT, loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    p = argparse.ArgumentParser()
    # ~85M params at vocab=65, seq=256, n_layer=12, n_head=12, n_embed=768.
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-embed", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=5_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(HERE / "checkpoints" / "nanogpt_baseline.pt"))
    args = p.parse_args()

    device = torch.device(args.device)
    data_dir = HERE / "data" / "tinyshakespeare"
    train_ids = np.fromfile(data_dir / "train.bin", dtype=np.uint8)
    val_ids = np.fromfile(data_dir / "val.bin", dtype=np.uint8)
    print(f"Train tokens: {len(train_ids):,}. Val tokens: {len(val_ids):,}.")

    train_ds = CharDataset(train_ids, args.seq)
    val_ds = CharDataset(val_ids, args.seq)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = NanoGPT(
        vocab_size=65, seq_len=args.seq,
        n_layer=args.n_layer, n_head=args.n_head, n_embed=args.n_embed,
        dropout=args.dropout,
    ).to(device)
    print(f"NanoGPT params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    step = 0
    t0 = time.time()
    best_val = float("inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    while step < args.steps:
        for x, y in train_loader:
            if step >= args.steps:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % args.log_every == 0:
                dt = time.time() - t0
                bpc = loss.item() / math.log(2)
                print(
                    f"step {step:5d} | loss {loss.item():.4f} (bpc {bpc:.3f}) | "
                    f"lr {sched.get_last_lr()[0]:.2e} | {dt:.1f}s"
                )

            if step > 0 and step % args.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                bpc = val_loss / math.log(2)
                print(f"  val_loss {val_loss:.4f} (bpc {bpc:.3f})")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model": model.state_dict(), "step": step,
                                "val_loss": val_loss}, args.out)
                    print(f"  saved {args.out}")

            step += 1

    val_loss = evaluate(model, val_loader, device, max_batches=200)
    bpc = val_loss / math.log(2)
    print("=" * 60)
    print(f"NanoGPT FINAL val loss {val_loss:.4f} (bpc {bpc:.3f})")
    print("Expected ~1.40-1.50 for matched 85M params on Tiny Shakespeare.")


if __name__ == "__main__":
    main()
