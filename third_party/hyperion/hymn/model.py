"""HYMN-Mini: the three-way fusion (VSA + NCA + DEQ).

Phase 1b experiment target. 8x8 toroidal field, D=1000, ~12.5M params for the
update MLP (2K -> 2048 -> 2048 -> 1K).

Per v1.1 audit: NO active inference component in training. AI deferred to
Phase 3 with a proper variational posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from vsa_core import Codebook, permute

from .field import FieldConfig, ToroidalField
from .update import LocalUpdateRule
from .deq import tbptt_forward, deq_forward


@dataclass
class HYMNConfig:
    """Configuration for the HYMN-Mini three-way fusion model.

    Defaults are calibrated to the Phase 1b experiment: 8x8 toroidal field with
    D=1000 hypervector cells, ~12.5M trainable parameters. The two pilot-
    discovered flags `learnable_codebook` and `readout_all_cells` are off by
    default for the audit-strict spec but should be enabled at sub-1M-param
    scales (see design/HYPERION-v1.2-DESIGN-UPDATE.md).

    Field geometry: cells [0:n_input) are written to from each new token,
    cells [n_input:n_input+n_comp) run the dynamics, and the remaining cells
    form the output zone. The update rule is uniform across all cells (NCA-
    style); zone behavior emerges from the fixed per-cell type embedding.
    """
    # Field
    height: int = 8
    width: int = 8
    d: int = 1_000
    n_input: int = 16     # 25%
    n_comp: int = 32      # 50%
    # n_output computed = H*W - n_input - n_comp = 16 in default 8x8

    # Vocabulary
    vocab_size: int = 65

    # MLP
    hidden: int = 2_048
    dropout: float = 0.0

    # Training curriculum
    phase1_steps: int = 50_000
    phase1_unroll: int = 5
    phase2_max_iter: int = 128
    phase2_eps: float = 1e-4

    # Annealing
    beta_start: float = 1.0
    beta_end: float = 20.0
    beta_anneal_steps: int = 50_000

    # Codebook
    learnable_codebook: bool = False
    # Output read mode: if True, bundle all cells; if False, only output zone.
    readout_all_cells: bool = False

    # Readout head: "default" | "vsa_cleanup" | "bind_query" | "superpos"
    #              | "context_conditioned" | "hopfield"
    readout: str = "default"
    n_queries: int = 1                  # for bind_query head
    hopfield_iters: int = 3             # for hopfield head

    # VSA key-value writes: when True, each input token is bound with a
    # learnable per-position key before being written into the field. The
    # field becomes a literal associative memory: read with bind(field, key_t)
    # to recover the token at position t. Required for true VSA recall.
    kv_writes: bool = False
    kv_max_positions: int = 64

    # Initial value of the input-gate parameter (learnable). Default 0.5
    # means each token write overwrites half the input-zone cell. For
    # recall-heavy tasks where older tokens must persist, init lower
    # (e.g. 0.1) so the field acts more like a bundle than an overwrite.
    input_gate_init: float = 0.5

    # Zone-enforcement aux loss
    zone_loss_start: float = 1.0
    zone_loss_anneal_steps: int = 50_000

    # Bipolar regularization: pushes field cells toward {-1, +1}.
    # Loss = bipolar_weight * mean((1 - h^2)^2). Encourages crisp VSA states.
    bipolar_weight: float = 0.0

    # Seeds
    codebook_seed: int = 0
    type_embed_seed: int = 1


class HYMNMini(nn.Module):
    """Three-way fusion language model: VSA substrate + NCA dynamics + DEQ iteration.

    Implements the architecture described in design/HYPERION-v1.1-DESIGN.md.
    The model maintains a (B, N_cells, D) field of bipolar hypervectors. Each
    new token is written to designated input cells via a learned gate; a
    uniform MLP local update rule then iterates the field for K steps (Phase 1
    TBPTT) or to a fixed point (Phase 2 phantom-grad DEQ). Predictions are
    extracted by bundling the field (or just the output zone) and computing
    cosine similarity against the token codebook.

    Training phases:
      step < phase1_steps: TBPTT with phase1_unroll iterations per token,
        PyTorch autograd through the explicit unroll.
      step >= phase1_steps: DEQ with phantom-gradient backward (Geng 2021),
        Anderson acceleration in the tanh-soft space.

    Inference: hard sign(), plain fixed-point iteration, no Anderson.

    Args:
      cfg: HYMNConfig instance. See its docstring for parameter semantics.
    """

    def __init__(self, cfg: HYMNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        fc = FieldConfig(
            height=cfg.height, width=cfg.width, d=cfg.d,
            n_input=cfg.n_input, n_comp=cfg.n_comp,
            type_embed_seed=cfg.type_embed_seed,
        )
        self.field = ToroidalField(fc, device="cpu")
        self.update_rule = LocalUpdateRule(
            d=cfg.d, hidden=cfg.hidden, n_zones=3, dropout=cfg.dropout,
        )

        # Codebook (frozen by default; optionally learnable as a relaxation)
        cb = Codebook(cfg.vocab_size, cfg.d, seed=cfg.codebook_seed)
        if cfg.learnable_codebook:
            self.codebook = nn.Parameter(cb.all().clone())
        else:
            self.register_buffer("codebook", cb.all())

        # Per-input-cell learned gate for write_input_zone.
        self.input_gates = nn.Parameter(
            torch.full((cfg.n_input,), cfg.input_gate_init))

        # Per-position bipolar keys for VSA key-value writes.
        if cfg.kv_writes:
            g = torch.Generator(device="cpu").manual_seed(cfg.codebook_seed + 17)
            raw = torch.randint(0, 2, (cfg.kv_max_positions, cfg.d),
                                 generator=g, dtype=torch.int8)
            self.position_keys = nn.Parameter((raw.float() * 2 - 1))
        else:
            self.register_buffer("position_keys", torch.empty(0))

        # CLIP-style learnable inverse-temperature for the cosine-sim logits.
        # Without this, logits are stuck in [-1, +1] and softmax is near-uniform,
        # which prevents cross-entropy from producing any meaningful gradient.
        # Kept for backwards compatibility with the default readout path; the
        # swappable heads in `hymn.readout` carry their own log_logit_scale.
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        # Optional swappable readout head.
        self.readout_head = None
        # Track whether the readout takes a query_hv (third arg).
        self.readout_needs_query = False
        if cfg.readout != "default":
            from .readout import (
                BindQueryHead, ContextConditionedHead, HopfieldCleanup,
                SuperposCleanup, VSACleanupHead,
            )
            if cfg.readout == "vsa_cleanup":
                self.readout_head = VSACleanupHead(cfg.vocab_size, cfg.d)
            elif cfg.readout == "bind_query":
                self.readout_head = BindQueryHead(
                    cfg.vocab_size, cfg.d, n_queries=cfg.n_queries)
            elif cfg.readout == "superpos":
                self.readout_head = SuperposCleanup(cfg.vocab_size, cfg.d)
            elif cfg.readout == "context_conditioned":
                self.readout_head = ContextConditionedHead(cfg.vocab_size, cfg.d)
                self.readout_needs_query = True
            elif cfg.readout == "hopfield":
                self.readout_head = HopfieldCleanup(
                    cfg.vocab_size, cfg.d, n_iterations=cfg.hopfield_iters)
            else:
                raise ValueError(f"unknown readout: {cfg.readout!r}")

        self._step = 0

    def beta(self) -> float:
        cfg = self.cfg
        frac = min(1.0, self._step / max(1, cfg.beta_anneal_steps))
        return cfg.beta_start + (cfg.beta_end - cfg.beta_start) * frac

    def zone_loss_weight(self) -> float:
        cfg = self.cfg
        frac = min(1.0, self._step / max(1, cfg.zone_loss_anneal_steps))
        return cfg.zone_loss_start * (1.0 - frac)

    def step_anneal(self) -> None:
        self._step += 1

    def _bind_static_to_device(self, device: torch.device) -> None:
        """Move static tables (neighbors, zones, type embeddings) to device."""
        self.field.neighbors = self.field.neighbors.to(device)
        self.field.zones = self.field.zones.to(device)
        self.field.type_embeddings = self.field.type_embeddings.to(device)
        self.field.input_mask = self.field.input_mask.to(device)
        self.field.output_mask = self.field.output_mask.to(device)
        self.field.comp_mask = self.field.comp_mask.to(device)

    def _update_closure(self, beta: float, soft: bool):
        """Build a state -> state closure for use in TBPTT / DEQ."""
        neighbors = self.field.neighbors
        type_emb = self.field.type_embeddings
        zones = self.field.zones

        def update(state: Tensor) -> Tensor:
            return self.update_rule(
                state, neighbors=neighbors, type_embeddings=type_emb,
                zones=zones, beta=beta, soft=soft,
            )

        return update

    def output_logits(
        self, state: Tensor, query_hv: Tensor | None = None,
    ) -> Tensor:
        """Scaled cosine sim of field bundle against codebook -> (B, V).

        Args:
          state:    (B, N, D) current field state.
          query_hv: optional (B, D) context query for context_conditioned
                    readout. Ignored by other heads.
        """
        out_hv = self.field.output_bundle(
            state, beta=self.beta(), use_all_cells=self.cfg.readout_all_cells,
        )
        if self.readout_head is not None:
            if self.readout_needs_query:
                if query_hv is None:
                    raise ValueError(
                        f"readout={self.cfg.readout!r} requires query_hv")
                # When kv_writes is on, hand the position_keys to the
                # readout as a key_store -- enables address-decode mode.
                if self.cfg.kv_writes:
                    return self.readout_head(
                        out_hv, self.codebook, query_hv,
                        key_store=self.position_keys,
                    )
                return self.readout_head(out_hv, self.codebook, query_hv)
            return self.readout_head(out_hv, self.codebook)
        out_norm = out_hv / (out_hv.norm(dim=-1, keepdim=True) + 1e-8)
        c_norm = self.codebook / (self.codebook.norm(dim=-1, keepdim=True) + 1e-8)
        scale = self.log_logit_scale.exp()
        return scale * (out_norm @ c_norm.T)

    def bipolar_loss(self, state: Tensor) -> Tensor:
        """Push field cells toward {-1, +1}. (1 - h^2)^2 = 0 iff h in {-1, +1}.

        Returns scalar. Pre-multiplied by cfg.bipolar_weight in the caller.
        """
        return ((1.0 - state * state) ** 2).mean()

    def forward(
        self,
        token_ids: Tensor,        # (B, T) long
        targets: Tensor | None = None,  # (B, T) long
    ) -> tuple[Tensor, Tensor | None, dict]:
        """Process a sequence; one DEQ/TBPTT solve per token.

        Returns:
          logits: (B, T, V)
          loss:   scalar (cross-entropy + zone aux), or None if no targets
          metrics: dict with diagnostics
        """
        cfg = self.cfg
        device = token_ids.device
        self._bind_static_to_device(device)
        B, T = token_ids.shape
        beta = self.beta()

        state = self.field.init_state(B).to(device)
        all_logits = []
        zone_aux_losses = []
        bipolar_aux_losses = []

        for t in range(T):
            tok_ids = token_ids[:, t]
            tok_hv_raw = self.codebook[tok_ids]                             # (B, D)
            # Positional permutation by t.
            tok_hv = permute(tok_hv_raw, shift=t % cfg.d)

            # VSA key-value write: bind the (permuted) token with a learnable
            # per-position key before writing. Field becomes an associative
            # memory: bind(field, key_t) cleans up to permute(token_t, t).
            if cfg.kv_writes:
                from vsa_core.ops import bind as _vsa_bind
                key_idx = t % cfg.kv_max_positions
                key = self.position_keys[key_idx].unsqueeze(0).expand(B, -1)
                tok_to_write = _vsa_bind(tok_hv, key)
            else:
                tok_to_write = tok_hv

            # Write the new token's HV into the input zone.
            state = self.field.write_input_zone(state, tok_to_write, self.input_gates, beta=beta)

            update_fn = self._update_closure(beta=beta, soft=True)
            if self._step < cfg.phase1_steps:
                state = tbptt_forward(state, update_fn, n_steps=cfg.phase1_unroll)
            else:
                state = deq_forward(
                    state, update_fn,
                    max_iter=cfg.phase2_max_iter, eps=cfg.phase2_eps,
                    use_anderson=True,
                )

            # For context_conditioned readout: query = the position key (when
            # kv_writes is on, this unbinds the stored token at position t)
            # OR the current token HV (when kv_writes is off, this is just
            # the most-recent-context signal the readout uses).
            if self.readout_needs_query:
                if cfg.kv_writes:
                    key_idx = t % cfg.kv_max_positions
                    query = self.position_keys[key_idx].unsqueeze(0).expand(B, -1)
                else:
                    query = tok_hv
            else:
                query = None
            logits = self.output_logits(state, query_hv=query)
            all_logits.append(logits)

            # Zone-enforcement aux: input zone mean should track current token HV.
            if self.zone_loss_weight() > 0 and targets is not None:
                input_mean = state[:, self.field.input_mask, :].mean(dim=1)  # (B, D)
                input_target = tok_hv
                zone_aux_losses.append(((input_mean - input_target) ** 2).mean())

            # Bipolar regularization aux: push cells toward {-1, +1}.
            if cfg.bipolar_weight > 0 and targets is not None:
                bipolar_aux_losses.append(self.bipolar_loss(state))

        logits = torch.stack(all_logits, dim=1)  # (B, T, V)

        loss = None
        metrics: dict[str, float] = {"beta": beta, "step": self._step}
        if targets is not None:
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                targets.reshape(-1), reduction="mean",
            )
            loss = ce
            metrics["ce_loss"] = ce.item()
            if zone_aux_losses:
                zone_loss = torch.stack(zone_aux_losses).mean()
                loss = loss + self.zone_loss_weight() * zone_loss
                metrics["zone_loss"] = zone_loss.item()
                metrics["zone_loss_weight"] = self.zone_loss_weight()
            if bipolar_aux_losses:
                bp_loss = torch.stack(bipolar_aux_losses).mean()
                loss = loss + cfg.bipolar_weight * bp_loss
                metrics["bipolar_loss"] = bp_loss.item()
                metrics["bipolar_weight"] = cfg.bipolar_weight

        return logits, loss, metrics

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate_from_prompt(
        self,
        prompt: Tensor,            # (T_prompt,) long
        max_new: int = 64,
        eos_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Process a prompt, then autoregressively sample new tokens.

        Returns the full token sequence including the prompt and the generated
        suffix. Halts when eos_id is sampled or max_new is reached.

        Uses hard sign at inference, plain fixed-point iteration (no Anderson),
        max_iter = phase2_max_iter.
        """
        self.eval()
        cfg = self.cfg
        device = prompt.device
        self._bind_static_to_device(device)
        beta = self.beta()

        state = self.field.init_state(batch_size=1).to(device)
        tokens = prompt.tolist()

        # Run the prompt through to warm up the field.
        for t, tok_id in enumerate(tokens):
            tok_hv_raw = self.codebook[tok_id]
            tok_hv = permute(tok_hv_raw, shift=t % cfg.d).unsqueeze(0)
            state = self.field.write_input_zone(state, tok_hv, self.input_gates, beta=beta)
            update_fn = self._update_closure(beta=beta, soft=False)
            from .deq import deq_forward
            state = deq_forward(
                state, update_fn,
                max_iter=cfg.phase2_max_iter, eps=cfg.phase2_eps,
                use_anderson=False,
            )

        # Sample. For context_conditioned readout, the query is the most
        # recently written token's positionally-permuted HV.
        last_tok_hv = None
        if self.readout_needs_query:
            last_tok_hv = permute(
                self.codebook[tokens[-1]], shift=(len(tokens) - 1) % cfg.d
            ).unsqueeze(0)
        for step in range(max_new):
            logits = self.output_logits(
                state, query_hv=last_tok_hv,
            ).squeeze(0) / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = int(torch.multinomial(probs, num_samples=1).item())
            tokens.append(next_tok)
            if eos_id is not None and next_tok == eos_id:
                break
            t = len(tokens) - 1
            tok_hv = permute(self.codebook[next_tok], shift=t % cfg.d).unsqueeze(0)
            state = self.field.write_input_zone(state, tok_hv, self.input_gates, beta=beta)
            update_fn = self._update_closure(beta=beta, soft=False)
            state = deq_forward(
                state, update_fn,
                max_iter=cfg.phase2_max_iter, eps=cfg.phase2_eps,
                use_anderson=False,
            )
            if self.readout_needs_query:
                last_tok_hv = tok_hv

        return torch.tensor(tokens, device=device, dtype=torch.long)
