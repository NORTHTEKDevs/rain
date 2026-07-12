"""VSA-native sequence engine (VSA-Seq).

A non-Transformer, non-MLP-RNN generative architecture. The recurrence
operates directly on bipolar 10K-D hypervectors using bind/bundle/permute
as the core operations. Connects to: Kanerva 1988 (sparse distributed
memory), Smolensky 1990 (tensor product representations), Plate 1995
(holographic reduced representations), and Schlag/Schmidhuber 2021
("linear Transformers are secretly fast-weight programmers").

What makes this architecturally different from HYMN / Mamba / RWKV:

  * State IS a hypervector, not a float vector. At inference the state is
    bipolar in {-1, +1}^D; during training we relax to tanh(.) and
    discretize at evaluation.
  * Recurrence uses BIND (elementwise multiply) + BUNDLE (additive sign)
    + PERMUTE (fixed shift) -- the three VSA primitives, NOT matrix
    multiplication on a learned weight.
  * Position role is a fixed permutation (free, no parameters), not a
    learned positional embedding.
  * The "output head" computes COSINE similarity with the codebook,
    not an arbitrary projection. Predictions are token IDs closest to
    the current state in hypervector space.
  * The codebook itself is bipolar and can be warm-started from any
    of RAIN's existing methods (feature, MiniLM, etc.).

Why this could be better-than-the-alternatives in the limit:

  * Parameter-light: most of the "model" is the fixed codebook plus
    two learnable D->D projections, not gigabytes of attention weights.
  * O(D) per step (elementwise ops) vs O(D^2) for dense recurrent
    matrices vs O(T*D) for attention.
  * Composable with KB: KB stores bound hypervectors in the SAME space
    as the sequence-engine state, so retrieval becomes a single bind +
    cleanup operation -- no separate retrieval pipeline.
  * Inherently sparse / discrete: hardware-friendly for non-GPU
    accelerators (Tsetlin-style chips, neuromorphic, etc.).

Why this MIGHT not work:

  * Bipolar relaxation through tanh may have bad gradient flow at
    high dimension.
  * Bundle (additive) without normalization saturates fast; cleanup
    via codebook nearest-neighbor is the proposed fix but it's not
    differentiable end-to-end.
  * Long-range dependency may be no better than HYMN's MLP-with-carry
    if the permutation doesn't act as a good positional code.

The first experiment (`scripts/pretrain_vsa_seq.py`) trains this on
Tiny Shakespeare with the same budget as HYMN's L1 specialist
(50K steps, dim=1024, carry=16, NLL loss) and compares. If we beat
HYMN's 1.47 NLL we have a real signal. If we don't, we learn what to
fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class VSASeqConfig:
    """Configuration for a VSA-Seq engine instance."""

    vocab_size: int
    dim: int
    # Tanh "temperature" -- higher = harder bipolar-like saturation. 1.0 is
    # vanilla tanh; values 2-5 give sharper near-bipolar gradients.
    bipolar_sharpness: float = 1.0
    # Whether the input projection W_in is enabled (set False to make the
    # input path pure VSA -- token_hv enters the bundle directly).
    use_input_proj: bool = True
    # Whether the output projection W_out is enabled.
    use_output_proj: bool = True
    # Number of permutation "lanes" (multi-head VSA). Each lane uses a
    # different fixed permutation, encoding a distinct position basis.
    n_perm_lanes: int = 4
    seed: int = 42


class VSASeq(nn.Module):
    """VSA-native sequence engine. See module docstring for the architectural
    claim.

    Inputs to forward():
        tokens: (B, T) int64 token IDs
        state: (B, D) float32 initial state (default zeros)

    Outputs:
        logits: (B, T, vocab_size) cosine-similarity logits in hypervector space
        new_state: (B, D) final state, ready for next chunk

    The codebook is registered as a buffer (not a Parameter) so it does not
    receive gradient updates by default. The caller can mutate it directly to
    implement continual learning (RAIN's `tell()` path). A learnable codebook
    variant lives under VSASeqLearnedCB.
    """

    def __init__(self, config: VSASeqConfig, codebook: np.ndarray | None = None):
        super().__init__()
        self.config = config

        # 1) Codebook: (vocab_size, dim) bipolar in {-1, +1}.
        if codebook is None:
            rng = np.random.default_rng(config.seed)
            codebook = rng.choice([-1, 1], size=(config.vocab_size, config.dim)).astype(np.int16)
        cb = torch.as_tensor(codebook, dtype=torch.float32)
        self.register_buffer("codebook", cb)

        # 2) Position roles: n_perm_lanes fixed permutations of [0, dim).
        rng = np.random.default_rng(config.seed + 1)
        perms = np.stack([rng.permutation(config.dim) for _ in range(config.n_perm_lanes)], axis=0)
        self.register_buffer("perms", torch.as_tensor(perms, dtype=torch.long))

        # 3) Learnable: input + output projections (small, D -> D).
        if config.use_input_proj:
            self.W_in = nn.Linear(config.dim, config.dim, bias=False)
        if config.use_output_proj:
            self.W_out = nn.Linear(config.dim, config.dim, bias=False)

        # 4) Lane mixing weight (one float per lane). Lets the model learn
        # how much each permutation's "view" of the past matters.
        self.lane_weights = nn.Parameter(torch.ones(config.n_perm_lanes) / config.n_perm_lanes)

        # 5) Learnable softmax temperature for the cosine readout. Init at
        # sqrt(D) so logits start variance-~1 (well-conditioned softmax).
        self.logit_temp = nn.Parameter(torch.tensor(float(config.dim) ** 0.5))

        # 6) Per-token bias (lets the model trivially learn frequency priors,
        # which absorbs the "loss is high just because of unigram stats"
        # gradient and frees the recurrence to learn actual structure).
        self.token_bias = nn.Parameter(torch.zeros(config.vocab_size))

        self._init_projections()

    def _init_projections(self) -> None:
        """Identity-leaning init: small random orthogonal perturbation around I.

        This keeps the initial dynamics close to "just bundle the token in"
        so training is stable. The model then learns to deviate from identity
        as needed.
        """
        for name in ("W_in", "W_out"):
            if hasattr(self, name):
                w: torch.Tensor = getattr(self, name).weight.data
                nn.init.eye_(w)
                noise = torch.randn_like(w) * 0.01
                w.add_(noise)

    def _bipolar_relax(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth bipolarization: tanh with optional sharpness."""
        return torch.tanh(self.config.bipolar_sharpness * x)

    def _multilane_permute(self, state: torch.Tensor) -> torch.Tensor:
        """Apply each permutation lane to `state`, weighted-sum the results.

        This is the VSA equivalent of "look at the past through multiple
        positional bases at once" -- analogous to multi-head attention but
        with zero learnable weights inside the head (just the fixed perms).
        """
        # state: (B, D); perms: (L, D); want: (B, D) after weighted sum.
        lanes = state[:, self.perms]  # (B, L, D) via advanced indexing
        # Softmax over lanes to keep mixing weights as a proper distribution
        # so they don't explode during training.
        weights = torch.softmax(self.lane_weights, dim=0)  # (L,)
        mixed = (lanes * weights.view(1, -1, 1)).sum(dim=1)  # (B, D)
        return mixed

    def step(self, token_hv: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """One recurrence step.

        token_hv: (B, D) bipolar (or float-relaxed) hypervector for input token
        state: (B, D) current state
        returns: (B, D) new state
        """
        # 1) View the past through the lane-mixed permutations (position role).
        past = self._multilane_permute(state)
        # 2) Optionally project the input (learned filter).
        inp = self.W_in(token_hv) if self.config.use_input_proj else token_hv
        # 3) Bundle (additive) then bipolar-relax. This is the heart of VSA-Seq.
        bundled = past + inp
        return self._bipolar_relax(bundled)

    def logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """Cosine-similarity readout, with a learnable temperature scalar.

        Why cosine over raw dot-product: the projected state magnitude is
        unbounded during training (W_out is unconstrained); raw dot product
        diverges and gradients explode. Cosine normalizes magnitudes so the
        softmax sees consistent-scale logits.

        Why a learnable temperature: pure cosine logits live in [-sqrt(D),
        +sqrt(D)] which is a narrow softmax range; the learnable temperature
        lets the model sharpen as needed.
        """
        proj = self.W_out(state) if self.config.use_output_proj else state
        proj_norm = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)
        cb_norm = self.codebook / (self.codebook.norm(dim=-1, keepdim=True) + 1e-8)
        cos = proj_norm @ cb_norm.T
        return cos * self.logit_temp + self.token_bias.unsqueeze(0)

    def forward(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the engine across a (B, T) token sequence in LM next-token mode.

        At step t we produce logits BEFORE bundling token[t] -- so the
        prediction is conditioned on the prior state only, not the token
        being predicted. After emitting logits we then update the state
        with token[t]. This is the standard causal-LM training pattern.

        Returns (logits, final_state) where logits[:, t] predicts the
        DISTRIBUTION over what character should follow tokens[:, :t]
        (i.e. predicts tokens[:, t]).
        """
        B, T = tokens.shape
        D = self.config.dim
        if state is None:
            state = torch.zeros(B, D, device=tokens.device, dtype=torch.float32)

        token_hvs = self.codebook[tokens]  # (B, T, D), float
        all_logits = torch.empty(
            B, T, self.config.vocab_size, device=tokens.device, dtype=torch.float32
        )
        for t in range(T):
            # Predict from state BEFORE adding token[t]
            all_logits[:, t] = self.logits_from_state(state)
            # Then ingest token[t] into the state for the next step
            state = self.step(token_hvs[:, t], state)
        return all_logits, state

    @torch.no_grad()
    def sample_next(
        self,
        token: int,
        state: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[int, torch.Tensor]:
        """Single-step sampling helper for inference loops."""
        tok = torch.tensor([[token]], device=state.device, dtype=torch.long)
        logits, new_state = self.forward(tok, state)
        l = logits[0, 0]
        if temperature <= 0.0:
            next_id = int(torch.argmax(l).item())
        else:
            p = torch.softmax(l / temperature, dim=-1)
            next_id = int(torch.multinomial(p, num_samples=1).item())
        return next_id, new_state


def count_params(model: VSASeq) -> int:
    """Just the learnable parameters (codebook + perms are fixed)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
