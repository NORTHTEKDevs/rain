# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""HYMN-Mamba: HYMN-Plus stack with PROPER Mamba S6 instead of our simple
selective gated recurrence.

The architecture: identical to HYMN-Plus v1's block layout, with the
sequence-mixing layer swapped from SelectiveGatedRecurrence -> MambaS6Block.

Per block:
  x -> LN -> MambaS6Block -> +x
    -> LN -> SwiGLU MLP   -> +residual

This is the cleanest A/B test of "does the simplified recurrence cap
HYMN-Plus's performance, or does proper Mamba S6 do better".

If HYMN-Mamba meaningfully beats HYMN-Plus at the same model size and
training budget, the recurrence design was the bottleneck and HYMN-Mamba
becomes our default sequence-mixing layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rain.core.mamba_s6 import MambaS6Block, MambaS6Config


@dataclass
class HymnMambaConfig:
    """Configuration for HYMN-Mamba."""

    vocab_size: int
    dim: int = 256
    n_layers: int = 4
    mlp_mult: int = 4
    dropout: float = 0.0
    tie_weights: bool = True
    # Mamba S6 sub-config
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    seed: int = 42


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = mult * dim
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class HymnMambaBlock(nn.Module):
    def __init__(self, config: HymnMambaConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        s6_cfg = MambaS6Config(
            dim=config.dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
        self.mamba = MambaS6Block(s6_cfg)
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = SwiGLU(config.dim, mult=config.mlp_mult, dropout=config.dropout)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mamba(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class HymnMamba(nn.Module):
    def __init__(self, config: HymnMambaConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList([HymnMambaBlock(config) for _ in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.dim)
        if config.tie_weights:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return x @ self.embed.weight.T if self.lm_head is None else self.lm_head(x)

    @torch.no_grad()
    def sample(
        self,
        prompt_tokens: torch.Tensor,
        max_new: int,
        temperature: float = 0.8,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.eval()
        out = prompt_tokens
        for _ in range(max_new):
            logits = self.forward(out)[:, -1]
            if temperature <= 0.0:
                next_tok = logits.argmax(dim=-1)
            else:
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    min_v = v[..., -1:]
                    logits = torch.where(
                        logits < min_v,
                        torch.tensor(-float("inf"), device=logits.device),
                        logits,
                    )
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            out = torch.cat([out, next_tok.unsqueeze(-1)], dim=1)
        return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
