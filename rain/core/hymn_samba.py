"""HYMN-Samba: alternating Mamba S6 + Sliding-Window Causal Attention.

Based on Samba (Microsoft Research, ICLR 2025, arXiv:2406.07522):
"Simple Hybrid State Space Models for Efficient Unlimited Context
Language Modeling". The single architecture from our deep research
push (Nov 2025) with peer-reviewed exact numbers showing wins over
both pure Mamba and pure Transformer at matched compute.

Verified Samba results (from the paper):
  438M params, trained on the Pile, at increasing eval context:
                  4k        8k        16k
    Llama-2      11.14     47.23    249.03   (catastrophic degradation
                                              past training context)
    Mamba        10.70     10.30     10.24
    Samba         9.65      9.65      9.57   (wins both)

  1.3B params, 100B tokens:
                  4k        8k        16k
    Llama-2       7.60     44.32    249.64
    Mamba         7.47      7.26      7.15
    Samba         7.32      7.11      6.96   (wins both)

The recipe: alternate every layer between
  (1) Mamba S6 block         (sequence mixing via selective SSM)
  (2) Sliding-Window MHA     (local exact attention, window ~1024-2048)
each followed by a SwiGLU MLP + residuals + pre-norm.

This composition gives:
  - O(T) per-token compute (Mamba layers + SWA is O(T*W))
  - Long-range memory via the SSM state
  - Local exact retrieval via SWA (the thing Mamba struggles at:
    associative recall over the recent window)
  - Graceful degradation: doesn't blow up at 10x training context

Honest scope: this is the BEST published hybrid at the time of our
research push. Expected payoff at our small CPU scale is modest
(0.1-0.3 nats/char improvement over pure Mamba S6 or pure attention),
because architectural advantages of Samba shine at long context (8k+),
which we can't even train at on CPU. The real win would land at
WT-103-scale + GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rain.core.mamba_s6 import MambaS6Block, MambaS6Config


@dataclass
class HymnSambaConfig:
    """Configuration for HYMN-Samba (alternating Mamba + SWA layers)."""

    vocab_size: int
    dim: int = 256
    n_layers: int = 6
    mlp_mult: int = 4
    n_heads: int = 4
    max_seq_len: int = 512
    window_size: int = 128  # Sliding-window size for the attention layers
    dropout: float = 0.1
    tie_weights: bool = True
    # Mamba S6 sub-config
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    # Layer pattern: "MSMS..." (Mamba/SWA alternating). Other patterns:
    #   "MS" = Mamba then SWA, repeat
    #   "MMS" = 2 Mamba then 1 SWA, repeat
    layer_pattern: str = "MS"
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


class SlidingWindowAttention(nn.Module):
    """Causal multi-head self-attention restricted to a sliding window.

    Each token attends only to the previous `window_size` tokens (plus
    itself). The Samba paper shows this is the right amount of local
    "exact memory" to complement the long-range SSM state -- larger
    windows don't help much because Mamba already covers long-range,
    and smaller windows lose associative-recall ability.

    Implementation: builds an additive (T, T) attention mask combining
    causal + window constraints, passes it to scaled_dot_product_attention.
    SDPA uses FlashAttention when available so this stays efficient on GPU.
    """

    def __init__(self, dim: int, n_heads: int, window_size: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def _build_window_mask(self, T: int, device, dtype) -> torch.Tensor:
        """Causal + sliding-window mask. Position i can attend to positions
        max(0, i-window+1)..i. Cells outside that range get -inf."""
        # Start with full -inf, then set the allowed band to 0.
        mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        idx = torch.arange(T, device=device)
        # Allow positions where (i - j) is in [0, window-1] inclusive
        rows = idx.unsqueeze(1)
        cols = idx.unsqueeze(0)
        diff = rows - cols
        allowed = (diff >= 0) & (diff < self.window_size)
        mask = torch.where(allowed, torch.zeros_like(mask), mask)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # If the window covers the whole sequence, the mask is just causal.
        if self.window_size >= T:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            mask = self._build_window_mask(T, x.device, q.dtype)
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )

        out = attn.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.resid_drop(self.proj(out))


class MambaLayer(nn.Module):
    """Pre-norm Mamba S6 block + SwiGLU MLP."""

    def __init__(self, config: HymnSambaConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        s6_cfg = MambaS6Config(
            dim=config.dim, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand
        )
        self.mamba = MambaS6Block(s6_cfg)
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = SwiGLU(config.dim, mult=config.mlp_mult, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SwaLayer(nn.Module):
    """Pre-norm Sliding-Window-Attention block + SwiGLU MLP."""

    def __init__(self, config: HymnSambaConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = SlidingWindowAttention(
            config.dim, config.n_heads, config.window_size, dropout=config.dropout
        )
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = SwiGLU(config.dim, mult=config.mlp_mult, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HymnSamba(nn.Module):
    """The model: alternating MambaLayer + SwaLayer per the layer_pattern."""

    def __init__(self, config: HymnSambaConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.drop = nn.Dropout(config.dropout)

        # Build the layer stack from the pattern.
        # "MS" repeated -> M, S, M, S, ...
        # "MMS" repeated -> M, M, S, M, M, S, ...
        pattern = config.layer_pattern.upper()
        if not pattern or not all(c in "MS" for c in pattern):
            raise ValueError(f"layer_pattern must be a non-empty string of M/S; got {pattern!r}")
        layers: list[nn.Module] = []
        for i in range(config.n_layers):
            kind = pattern[i % len(pattern)]
            if kind == "M":
                layers.append(MambaLayer(config))
            else:
                layers.append(SwaLayer(config))
        self.blocks = nn.ModuleList(layers)
        self.norm_f = nn.LayerNorm(config.dim)
        if config.tie_weights:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device, dtype=torch.long).unsqueeze(0).expand(B, T)
        x = self.embed(tokens) + self.pos_embed(pos)
        x = self.drop(x)
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
            ctx = out[:, -self.config.max_seq_len :]
            logits = self.forward(ctx)[:, -1]
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
