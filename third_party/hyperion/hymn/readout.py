"""Readout heads for HYMN.

The default `output_logits` in `HYMNMini` does a flat cosine similarity
between the field bundle and the codebook. That's adequate for very
small vocabularies but throws away the VSA inductive bias: the field
is bipolar, the codebook is bipolar, and the natural operation between
them is *bind + cleanup*, not just dot-product.

This module provides:

  BindQueryHead       Bind the field bundle with a learnable query vector
                       before cleanup. Lets the model learn `what` to
                       extract from the field independently of `how`
                       the field stores information.
  VSACleanupHead     Cosine-sim against the codebook + temperature.
                       Same as the inline default but as a swappable module.
  SuperposCleanup    Multi-query attention-style cleanup that returns a
                       soft mixture of codebook tokens (useful when the
                       next token is genuinely ambiguous given context).

All heads accept a `field_bundle: (B, D)` and return logits `(B, V)`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from vsa_core.ops import bind


class VSACleanupHead(nn.Module):
    """Plain cosine-sim cleanup with a learnable temperature.

    Same algebra as the inline readout in `HYMNMini.output_logits`, packaged
    as a module so the variants below can compose with it.
    """

    def __init__(self, vocab_size: int, d: int, init_log_scale: float = math.log(10.0)) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.log_logit_scale = nn.Parameter(torch.tensor(init_log_scale))

    def forward(self, field_bundle: Tensor, codebook: Tensor) -> Tensor:
        # field_bundle: (B, D), codebook: (V, D)
        b_norm = field_bundle / (field_bundle.norm(dim=-1, keepdim=True) + 1e-8)
        c_norm = codebook / (codebook.norm(dim=-1, keepdim=True) + 1e-8)
        return self.log_logit_scale.exp() * (b_norm @ c_norm.T)


class BindQueryHead(nn.Module):
    """Bind-then-cleanup: learnable query HV gets bound with the field bundle
    before cleanup. The query acts as a "what am I looking for?" question that
    the field stores compositionally.

    The query starts as a random bipolar vector (so binding is well-conditioned
    in the VSA sense) and is then learned as a continuous-relaxed parameter.
    Binding is element-wise (Hadamard) product in our bipolar regime, so
    bind(field_bundle, query) is differentiable.
    """

    def __init__(
        self,
        vocab_size: int,
        d: int,
        n_queries: int = 1,
        query_seed: int = 7,
        init_log_scale: float = math.log(10.0),
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.n_queries = n_queries
        # Bipolar init -> well-conditioned binds.
        g = torch.Generator(device="cpu").manual_seed(query_seed)
        q_init = torch.randint(0, 2, (n_queries, d), generator=g, dtype=torch.int8)
        q_init = (q_init.float() * 2 - 1)
        self.queries = nn.Parameter(q_init)
        # Per-query weight (so the model can soft-weight which query is dominant).
        self.query_weights = nn.Parameter(torch.full((n_queries,), 1.0 / n_queries))
        self.log_logit_scale = nn.Parameter(torch.tensor(init_log_scale))

    def forward(self, field_bundle: Tensor, codebook: Tensor) -> Tensor:
        # field_bundle: (B, D), codebook: (V, D)
        B = field_bundle.shape[0]
        # Bind field_bundle with each query and weight-sum.
        # bind is Hadamard product for bipolar VSAs. Use broadcasting:
        #   queries: (Q, D), field_bundle: (B, D) -> (B, Q, D) by outer hadamard
        fb = field_bundle.unsqueeze(1).expand(B, self.n_queries, self.d)
        q = self.queries.unsqueeze(0).expand(B, self.n_queries, self.d)
        bound = bind(fb, q)                                # (B, Q, D)
        # Weighted bundle across queries.
        w = torch.softmax(self.query_weights, dim=0)      # (Q,)
        mixed = (bound * w.view(1, -1, 1)).sum(dim=1)     # (B, D)
        mixed = torch.tanh(mixed)                         # keep bipolar-ish
        b_norm = mixed / (mixed.norm(dim=-1, keepdim=True) + 1e-8)
        c_norm = codebook / (codebook.norm(dim=-1, keepdim=True) + 1e-8)
        return self.log_logit_scale.exp() * (b_norm @ c_norm.T)


class ContextConditionedHead(nn.Module):
    """Readout that uses the most recent input as a query key.

    Identified in `experiments/hymn_recall_bench.py` as the single biggest
    known architectural gap. The flat cleanup heads have no idea WHAT to
    extract from the field -- they always extract the same projection. For
    any task where the answer depends on the most recent context token
    (recall, position-conditioned lookup, "what was the i-th X"), this
    head is the difference between marginal-above-random and actually
    solving the task.

    Mechanism:
      1. Take the field bundle `fb: (B, D)`.
      2. Take the most recent input HV `query: (B, D)` (positionally permuted).
      3. Bind: `bound = bind(fb, query)` -- this is VSA-style unbinding when
          `query` was bound INTO the field at write time.
      4. Mix with a learnable linear projection of (fb, query) -- gives the
          model freedom to learn non-VSA readouts when bind+cleanup isn't
          sufficient.
      5. Cleanup against codebook with learnable scale.

    Usage: call as `head(field_bundle, codebook, query_hv)`. The third
    argument is what makes it context-conditioned.
    """

    def __init__(
        self,
        vocab_size: int,
        d: int,
        init_log_scale: float = math.log(10.0),
        mix_weight: float = 0.5,
        init_address_temp: float = 0.5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.log_logit_scale = nn.Parameter(torch.tensor(init_log_scale))
        # Learnable mix between pure-bind readout and a learned projection.
        # alpha=1.0 -> pure VSA; alpha=0.0 -> pure learned. Defaults to 0.5.
        self.mix_logit = nn.Parameter(torch.tensor(math.log(mix_weight / max(1e-6, 1 - mix_weight))))
        # Learned projection of (field_bundle, query) for the non-VSA path.
        self.proj = nn.Linear(2 * d, d, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
        # Address-decode (used only when key_store is passed to forward()).
        # Low temp -> sharp argmax-like lookup; high temp -> blurred mix.
        self.log_address_temp = nn.Parameter(
            torch.tensor(math.log(init_address_temp)))
        # Linear projection of the query before scoring against position
        # keys. This is the "address decoder" -- learns the
        # (query_token -> position_index) mapping that the recall task
        # requires.
        self.query_proj = nn.Linear(d, d, bias=False)
        nn.init.xavier_uniform_(self.query_proj.weight)

    def forward(
        self,
        field_bundle: Tensor,             # (B, D)
        codebook: Tensor,                 # (V, D)
        query_hv: Tensor,                 # (B, D)
        key_store: Tensor | None = None,  # (N_pos, D); enables address decode
    ) -> Tensor:
        if key_store is not None:
            # Address-decode mode: the query token chooses which position
            # key to use for unbinding.
            q_proj = self.query_proj(query_hv)
            q_norm = q_proj / (q_proj.norm(dim=-1, keepdim=True) + 1e-8)
            k_norm = key_store / (key_store.norm(dim=-1, keepdim=True) + 1e-8)
            attn = torch.softmax(
                (q_norm @ k_norm.T) / self.log_address_temp.exp(),
                dim=-1,
            )                                              # (B, N_pos)
            unbinding_key = attn @ key_store               # (B, D)
            unbinding_key = torch.tanh(unbinding_key)
        else:
            unbinding_key = query_hv

        # VSA path: unbind whatever was bound at write time.
        bound = bind(field_bundle, unbinding_key)          # (B, D)
        bound = torch.tanh(bound)
        # Learned path: linear projection of concat(fb, query).
        projected = self.proj(torch.cat([field_bundle, query_hv], dim=-1))
        projected = torch.tanh(projected)
        # Convex mix.
        alpha = torch.sigmoid(self.mix_logit)
        mixed = alpha * bound + (1.0 - alpha) * projected
        b_norm = mixed / (mixed.norm(dim=-1, keepdim=True) + 1e-8)
        c_norm = codebook / (codebook.norm(dim=-1, keepdim=True) + 1e-8)
        return self.log_logit_scale.exp() * (b_norm @ c_norm.T)


class HopfieldCleanup(nn.Module):
    """Iterative associative cleanup using the codebook as attractor set.

    Modern Hopfield networks (Ramsauer et al. 2020) reformulate retrieval
    as a softmax-weighted superposition that converges in a few iterations.
    We apply that to the cleanup step: given the (noisy) field bundle,
    iteratively project onto the codebook manifold via softmax-weighted
    mixture, K times.

    Sharpens outputs that the cosine-sim cleanup alone would smear across
    near-similar codebook entries. Useful as a final stage after the field
    has produced a "messy" output bundle.

    Cost: K * (D * V) per forward pass, where K is typically 2-4.
    """

    def __init__(
        self,
        vocab_size: int,
        d: int,
        n_iterations: int = 3,
        init_log_scale: float = math.log(10.0),
        init_attn_temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.n_iterations = n_iterations
        self.log_logit_scale = nn.Parameter(torch.tensor(init_log_scale))
        self.log_attn_temp = nn.Parameter(torch.tensor(math.log(init_attn_temp)))

    def forward(self, field_bundle: Tensor, codebook: Tensor) -> Tensor:
        c_norm = codebook / (codebook.norm(dim=-1, keepdim=True) + 1e-8)
        state = field_bundle
        for _ in range(self.n_iterations):
            s_norm = state / (state.norm(dim=-1, keepdim=True) + 1e-8)
            sims = s_norm @ c_norm.T                       # (B, V)
            attn = torch.softmax(sims / self.log_attn_temp.exp(), dim=-1)
            state = attn @ codebook                        # superposition
            state = torch.tanh(state)
        # Final cleanup logits against codebook.
        s_norm = state / (state.norm(dim=-1, keepdim=True) + 1e-8)
        return self.log_logit_scale.exp() * (s_norm @ c_norm.T)


class SuperposCleanup(nn.Module):
    """Multi-head attention-style cleanup.

    Instead of returning the single best codebook match, we compute attention
    over the codebook (softmax of cosine sims) and return a *superposition*
    cleanup -- the weighted bundle of candidate tokens, then cosine-sim'd
    against the codebook again. This second pass lets the model express
    "I'm not sure if it's X or Y; the answer is somewhere between" rather
    than collapsing to a single token early.

    Useful when training on inherently ambiguous next-token distributions
    (most language tasks).
    """

    def __init__(
        self,
        vocab_size: int,
        d: int,
        attention_temperature: float = 1.0,
        init_log_scale: float = math.log(10.0),
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.log_attn_temp = nn.Parameter(
            torch.tensor(math.log(attention_temperature)))
        self.log_logit_scale = nn.Parameter(torch.tensor(init_log_scale))

    def forward(self, field_bundle: Tensor, codebook: Tensor) -> Tensor:
        # First pass: attention weights over codebook.
        b_norm = field_bundle / (field_bundle.norm(dim=-1, keepdim=True) + 1e-8)
        c_norm = codebook / (codebook.norm(dim=-1, keepdim=True) + 1e-8)
        sims = b_norm @ c_norm.T                           # (B, V)
        attn = torch.softmax(sims / self.log_attn_temp.exp(), dim=-1)
        # Superposition cleanup: weighted bundle of codebook tokens.
        cleaned = attn @ codebook                          # (B, D)
        cleaned = torch.tanh(cleaned)
        # Second-pass cosine sim with the cleaned bundle.
        cl_norm = cleaned / (cleaned.norm(dim=-1, keepdim=True) + 1e-8)
        return self.log_logit_scale.exp() * (cl_norm @ c_norm.T)
