"""Minimal decoder-only transformer for TinySCAN.

Format:
  Input: "<bos> walk twice <sep> W W <eos>"
  Train: predict every token after position 0 (standard LM).
  Eval:  feed prefix up to <sep>, greedy-decode to <eos>, compare with target.

Tiny model: 2 layers, 64-dim embed, 4 heads. ~25k params. The task is small,
so overfitting the training set is fine and expected.

The interesting question is NOT training perplexity. It is whether the model
generalizes to (swim, MODIFIER) compositions it never saw during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pure_vsa.tinyscan import (
    ACTIONS,
    MODIFIERS,
    OUTPUT_TOKENS,
    TinySCANExample,
    make_dataset,
)
from torch import Tensor, nn

# Vocabulary: input tokens, output tokens, plus special tokens.
SPECIAL_TOKENS = ["<pad>", "<bos>", "<sep>", "<eos>"]
PAD, BOS, SEP, EOS = 0, 1, 2, 3
INPUT_TOKENS = ACTIONS + MODIFIERS
ALL_TOKENS = SPECIAL_TOKENS + INPUT_TOKENS + OUTPUT_TOKENS
TOKEN_TO_ID = {t: i for i, t in enumerate(ALL_TOKENS)}
ID_TO_TOKEN = dict(enumerate(ALL_TOKENS))
VOCAB_SIZE = len(ALL_TOKENS)


def encode_example(ex: TinySCANExample) -> tuple[list[int], list[int]]:
    """Return (full_sequence, label_mask) for an example.

    full_sequence: <bos> action [modifier] <sep> out1 out2 ... <eos>
    label_mask:    0 for tokens we don't compute loss on (input + sep),
                   1 for tokens after <sep> we DO compute loss on.
    """
    tokens = [BOS, TOKEN_TO_ID[ex.action]]
    if ex.modifier is not None:
        tokens.append(TOKEN_TO_ID[ex.modifier])
    sep_pos = len(tokens)
    tokens.append(SEP)
    for t in ex.output_tokens:
        tokens.append(TOKEN_TO_ID[t])
    tokens.append(EOS)
    mask = [0] * (sep_pos + 1) + [1] * (len(tokens) - sep_pos - 1)
    return tokens, mask


def collate(batch: list[tuple[list[int], list[int]]], max_len: int) -> tuple[Tensor, Tensor]:
    """Pad to max_len, return (input_ids, label_mask)."""
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (tok, m) in enumerate(batch):
        L = len(tok)
        ids[i, :L] = torch.tensor(tok, dtype=torch.long)
        mask[i, :L] = torch.tensor(m, dtype=torch.long)
    return ids, mask


@dataclass
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    max_len: int = 16
    dropout: float = 0.0


class TinyDecoderLM(nn.Module):
    """Decoder-only transformer for the TinySCAN seq2seq task."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, ids: Tensor) -> Tensor:
        """ids: (B, T) -> logits (B, T, V)."""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        h = self.embed(ids) + self.pos(pos)
        # causal mask
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1
        )
        h = self.blocks(h, mask=causal, is_causal=True)
        h = self.norm(h)
        return self.head(h)

    @torch.no_grad()
    def generate(self, prefix: list[int], max_new: int = 8) -> list[int]:
        """Greedy decode until <eos> or max_new tokens."""
        self.eval()
        ids = torch.tensor([prefix], dtype=torch.long)
        for _ in range(max_new):
            logits = self.forward(ids)
            next_id = logits[0, -1].argmax().item()
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
            if next_id == EOS:
                break
        return ids[0, len(prefix):].tolist()


def train_and_eval(
    seed: int = 0,
    n_epochs: int = 2000,
    lr: float = 3e-3,
    held_out_action: str = "swim",
    verbose: bool = False,
) -> dict:
    """Train the model on TinySCAN train split, evaluate exact-match on test."""
    torch.manual_seed(seed)

    train_examples, test_examples = make_dataset(held_out_action=held_out_action)
    train_data = [encode_example(ex) for ex in train_examples]
    max_len = max(len(t[0]) for t in train_data) + 2

    cfg = TransformerConfig(max_len=max_len)
    model = TinyDecoderLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())

    ids, mask = collate(train_data, max_len)
    targets = ids.clone()
    inputs = ids.clone()
    # standard LM: predict token t+1 from tokens 0..t
    # so shift: input = ids[:, :-1], target = ids[:, 1:], mask shifted similarly
    inp = inputs[:, :-1]
    tgt = targets[:, 1:]
    tgt_mask = mask[:, 1:]

    for epoch in range(n_epochs):
        model.train()
        logits = model(inp)  # (B, T-1, V)
        loss_per_token = nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            tgt.reshape(-1),
            reduction="none",
        ).reshape(tgt.shape)
        # only compute loss on output tokens (after <sep>)
        loss = (loss_per_token * tgt_mask).sum() / (tgt_mask.sum() + 1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and epoch % 200 == 0:
            print(f"  epoch {epoch:4d} loss {loss.item():.4f}")

    # eval on training set (memorization check)
    train_correct = 0
    for ex in train_examples:
        prefix = _build_prefix(ex)
        generated = model.generate(prefix, max_new=8)
        predicted_tokens = _decode_generated(generated)
        if predicted_tokens == ex.output_tokens:
            train_correct += 1

    # eval on held-out (compositional generalization)
    test_correct = 0
    test_results = []
    for ex in test_examples:
        prefix = _build_prefix(ex)
        generated = model.generate(prefix, max_new=8)
        predicted_tokens = _decode_generated(generated)
        test_results.append((ex.action, ex.modifier, predicted_tokens, ex.output_tokens))
        if predicted_tokens == ex.output_tokens:
            test_correct += 1

    return {
        "n_params": n_params,
        "train_size": len(train_examples),
        "test_size": len(test_examples),
        "train_acc": train_correct / len(train_examples),
        "test_acc": test_correct / len(test_examples),
        "test_results": test_results,
        "final_loss": loss.item(),
    }


def _build_prefix(ex: TinySCANExample) -> list[int]:
    """Return tokens up to and including <sep>."""
    tokens = [BOS, TOKEN_TO_ID[ex.action]]
    if ex.modifier is not None:
        tokens.append(TOKEN_TO_ID[ex.modifier])
    tokens.append(SEP)
    return tokens


def _decode_generated(generated_ids: list[int]) -> list[str]:
    """Trim at <eos>, filter specials, map to token strings."""
    out = []
    for tid in generated_ids:
        if tid == EOS:
            break
        if tid >= 4:  # skip specials
            out.append(ID_TO_TOKEN[tid])
    return out


if __name__ == "__main__":
    result = train_and_eval(verbose=True)
    print(f"\nparams:        {result['n_params']}")
    print(f"train size:    {result['train_size']}")
    print(f"test size:     {result['test_size']}")
    print(f"train acc:     {result['train_acc']:.3f}")
    print(f"test acc:      {result['test_acc']:.3f} (compositional generalization)")
    print(f"final loss:    {result['final_loss']:.4f}")
    print("\ntest details:")
    for a, m, p, e in result["test_results"]:
        marker = "OK  " if p == e else "FAIL"
        print(f"  {marker} ({a}, {m}) -> {p}  expected {e}")
