"""HYMN-Seq: redesigned HYMN for sequence-to-sequence tasks.

Key change from the original HYMN-Mini: explicit per-position memory
bank instead of the overwrite-and-bundle field. See
`design/HYMN-SEQ-DESIGN.md` for the rationale.

Architecture summary:
  Encoder:   write each input token to its own bank slot via
             slot[t] = bind(permute(token_hv, t), position_key_t)
  Iteration: K local-update steps over the bank with linear neighbor
             structure (slot[t-1], slot[t+1])
  Decoder:   VSA-flavored cross-attention -- query attends over bank,
             retrieved value is unbound + cleanup'd against codebook

Preserves the VSA substrate (bipolar HVs, bind/unbind/cleanup, the
LocalUpdateRule MLP) while fixing the no-per-position-memory failure
mode that crippled the original HYMN on M2/M3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from vsa_core import Codebook, permute
from vsa_core.ops import bind


@dataclass
class HYMNSeqConfig:
    # Vocabulary
    vocab_size: int = 24
    # Hypervector dimension
    d: int = 256
    # Encoder update MLP hidden dim
    hidden: int = 256
    # Number of field-iteration steps over the bank
    n_iters: int = 3
    # Max sequence length (bound for position_keys / pos_embeds)
    max_positions: int = 64
    # Whether to apply the local NCA update between slots.
    # Off = pure write-then-attend baseline (no inter-slot mixing).
    use_field_update: bool = True
    # Whether the codebook is learnable (typically True for seq2seq).
    learnable_codebook: bool = True
    # Seeds
    codebook_seed: int = 0
    pos_key_seed: int = 7
    # CLIP-style logit scale init.
    init_log_logit_scale: float = math.log(10.0)
    # Annealing for the soft-sign nonlinearity (kept as VSA-flavored bookkeeping).
    beta: float = 5.0


class MemoryBank(nn.Module):
    """Per-token bipolar memory bank with VSA key-value writes.

    Writes:
      slot[t] = bind(permute(token_hv, t), position_key_t)
    Reads (external):
      bank.slots is a (B, T, D) tensor of bipolar-ish HVs.
    """

    def __init__(self, cfg: HYMNSeqConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Learnable bipolar position keys.
        g = torch.Generator(device="cpu").manual_seed(cfg.pos_key_seed)
        raw = torch.randint(0, 2, (cfg.max_positions, cfg.d),
                             generator=g, dtype=torch.int8)
        self.position_keys = nn.Parameter((raw.float() * 2 - 1))

    def write(self, token_hvs: Tensor) -> Tensor:
        """Build the memory bank from a sequence of token HVs.

        Args:
          token_hvs: (B, T, D) -- one HV per input position, already
                     positionally permuted by the caller.

        Returns:
          slots: (B, T, D) bipolar-ish memory bank.
        """
        B, T, D = token_hvs.shape
        assert T <= self.cfg.max_positions, (
            f"sequence length {T} exceeds max_positions {self.cfg.max_positions}")
        keys = self.position_keys[:T].unsqueeze(0).expand(B, T, D)
        return torch.tanh(self.cfg.beta * bind(token_hvs, keys))

    def get_keys(self, T: int) -> Tensor:
        return self.position_keys[:T]


class InterSlotUpdate(nn.Module):
    """Local NCA-style update across bank slots (linear neighbor structure).

    Each slot attends to itself + previous + next, mixed via a small MLP.
    Applied for `n_iters` iterations. Optional (cfg.use_field_update).
    """

    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        # Per-slot input: self + prev + next + type-embed = 4D
        self.fc1 = nn.Linear(4 * d, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # Residual gate per slot type (computation vs boundary).
        self.alpha = nn.Parameter(torch.full((1,), 0.5))

    def forward(self, slots: Tensor, type_embed: Tensor,
                 beta: float = 5.0) -> Tensor:
        """One iteration of inter-slot mixing.

        Args:
          slots: (B, T, D)
          type_embed: (T, D) frozen random bipolar
        """
        B, T, D = slots.shape
        # Build neighbor tensors with cyclic-pad on the ends.
        prev = torch.roll(slots, shifts=1, dims=1)
        nxt = torch.roll(slots, shifts=-1, dims=1)
        te = type_embed.unsqueeze(0).expand(B, T, D)
        x = torch.cat([slots, prev, nxt, te], dim=-1)        # (B, T, 4D)
        h = torch.relu(self.fc1(x))
        delta = self.fc2(h)                                  # (B, T, D)
        alpha = torch.sigmoid(self.alpha)
        return torch.tanh(beta * (alpha * slots + (1 - alpha) * delta))


class CrossAttendDecoder(nn.Module):
    """VSA-flavored cross-attention readout.

    For each output query, soft-attend over bank slots, then unbind the
    retrieved value with the looked-up position key and cleanup against
    the codebook.

    This is structurally attention, but the value path is
    `bind(slot, key)` rather than a learned `W_v @ slot`. That keeps the
    VSA inductive bias: outputs must lie close to the codebook manifold.
    """

    def __init__(self, cfg: HYMNSeqConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Decoder query projection.
        self.query_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        nn.init.xavier_uniform_(self.query_proj.weight)
        # Output position embeddings for the decoder (added to query HVs).
        g = torch.Generator(device="cpu").manual_seed(cfg.pos_key_seed + 31)
        raw = torch.randint(0, 2, (cfg.max_positions, cfg.d),
                             generator=g, dtype=torch.int8)
        self.dec_pos_embed = nn.Parameter((raw.float() * 2 - 1))
        # Logit scale for cleanup.
        self.log_logit_scale = nn.Parameter(
            torch.tensor(cfg.init_log_logit_scale))
        # Attention temperature.
        self.log_attn_temp = nn.Parameter(torch.tensor(math.log(1.0)))

    def forward(
        self,
        query_hv: Tensor,        # (B, D) current decoder state
        slots: Tensor,           # (B, T_in, D) memory bank
        keys: Tensor,            # (T_in, D) position keys
        codebook: Tensor,        # (V, D)
        dec_pos: int,            # decoder position index
    ) -> Tensor:
        B = query_hv.shape[0]
        T_in, D = slots.shape[1], slots.shape[2]

        # Add decoder position embedding to query.
        pos = self.dec_pos_embed[dec_pos % self.cfg.max_positions]
        q = self.query_proj(query_hv + pos)
        q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

        # Attention: (B, T_in)
        slot_norm = slots / (slots.norm(dim=-1, keepdim=True) + 1e-8)
        sims = torch.einsum("bd,btd->bt", q_norm, slot_norm)
        attn = torch.softmax(
            sims / (self.log_attn_temp.exp() * math.sqrt(D)),
            dim=-1,
        )                                                   # (B, T_in)

        # Unbind each slot with its position key, weighted by attention.
        keys_b = keys.unsqueeze(0).expand(B, T_in, D)
        unbound = bind(slots, keys_b)                        # (B, T_in, D)
        unbound = torch.tanh(unbound)
        retrieved = torch.einsum("bt,btd->bd", attn, unbound)  # (B, D)

        # Cleanup against codebook.
        r_norm = retrieved / (retrieved.norm(dim=-1, keepdim=True) + 1e-8)
        c_norm = codebook / (codebook.norm(dim=-1, keepdim=True) + 1e-8)
        return self.log_logit_scale.exp() * (r_norm @ c_norm.T)


class HYMNSeq(nn.Module):
    """End-to-end encoder + iteration + decoder. VSA throughout.

    Two operating modes:
      forward(token_ids, targets) -> logits, loss, metrics
        Trains on a teacher-forced output. The first <sep_id> token (if
        provided) marks the start of the decoder; positions before that
        are encoder context only.
      generate(prompt, max_new, eos_id) -> generated token sequence
        Greedy decode given a prompt ending in <sep>.
    """

    def __init__(self, cfg: HYMNSeqConfig) -> None:
        super().__init__()
        self.cfg = cfg
        cb = Codebook(cfg.vocab_size, cfg.d, seed=cfg.codebook_seed)
        if cfg.learnable_codebook:
            self.codebook = nn.Parameter(cb.all().clone())
        else:
            self.register_buffer("codebook", cb.all())

        self.bank = MemoryBank(cfg)
        if cfg.use_field_update:
            self.update_rule = InterSlotUpdate(cfg.d, cfg.hidden)
            # Frozen random bipolar type embeddings per slot.
            g = torch.Generator(device="cpu").manual_seed(cfg.pos_key_seed + 99)
            raw = torch.randint(0, 2, (cfg.max_positions, cfg.d),
                                 generator=g, dtype=torch.int8)
            self.register_buffer("type_embed", (raw.float() * 2 - 1))
        else:
            self.update_rule = None
            self.register_buffer("type_embed", torch.empty(0))
        self.decoder = CrossAttendDecoder(cfg)

    def _build_bank(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Encode token_ids into a memory bank + return position keys."""
        cfg = self.cfg
        B, T = token_ids.shape
        # Codebook lookup + positional permute.
        raw_hvs = self.codebook[token_ids]                   # (B, T, D)
        permuted = torch.stack([
            permute(raw_hvs[:, t, :], shift=t % cfg.d) for t in range(T)
        ], dim=1)                                            # (B, T, D)
        # Write into bank.
        slots = self.bank.write(permuted)                    # (B, T, D)
        # Iterate (NCA update over bank slots).
        if self.update_rule is not None:
            type_emb = self.type_embed[:T]
            for _ in range(cfg.n_iters):
                slots = self.update_rule(slots, type_emb, beta=cfg.beta)
        keys = self.bank.get_keys(T)
        return slots, keys

    def forward(
        self,
        token_ids: Tensor,                # (B, T)
        targets: Tensor | None = None,    # (B, T) with -100 to ignore
        encoder_until: int | None = None, # build bank only from [0:encoder_until]
    ) -> tuple[Tensor, Tensor | None, dict]:
        """Train-time forward. Loss is CE on non-(-100) target positions.

        If `encoder_until` is set, the bank is built only from tokens
        [0:encoder_until], matching generation-time behavior where the
        decoder must produce outputs from only the input context. Without
        this, the decoder can trivially copy answers from output-portion
        slots, leading to perfect train CE + zero generalization (the
        exposure-bias / train-test-bank-mismatch failure we hit).
        """
        B, T = token_ids.shape
        cfg = self.cfg
        # Encoder: build the bank from input portion only (or full seq).
        bank_T = encoder_until if encoder_until is not None else T
        bank_T = max(1, min(bank_T, T))
        slots, keys = self._build_bank(token_ids[:, :bank_T])
        # Decoder: at position t, the query is token_hv[t] (the input we
        # are conditioning on); we predict token at t+1.
        token_hvs = self.codebook[token_ids]                 # (B, T, D)
        all_logits = []
        for t in range(T):
            logits = self.decoder(
                query_hv=token_hvs[:, t, :],
                slots=slots, keys=keys,
                codebook=self.codebook, dec_pos=t,
            )                                                # (B, V)
            all_logits.append(logits)
        logits = torch.stack(all_logits, dim=1)              # (B, T, V)

        loss = None
        metrics = {}
        if targets is not None:
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            loss = ce
            metrics["ce"] = ce.item()
        return logits, loss, metrics

    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,                   # (T_p,) long
        max_new: int = 40,
        eos_id: int | None = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """Greedy decode given a prompt ending in <sep>."""
        self.eval()
        cfg = self.cfg
        device = prompt.device
        tokens = prompt.tolist()
        # Build bank from prompt.
        prompt_b = prompt.unsqueeze(0)
        slots, keys = self._build_bank(prompt_b)
        # Generate.
        for step in range(max_new):
            t = len(tokens) - 1
            last_hv = self.codebook[tokens[-1]].unsqueeze(0)  # (1, D)
            logits = self.decoder(
                query_hv=last_hv, slots=slots, keys=keys,
                codebook=self.codebook, dec_pos=t,
            ).squeeze(0) / max(temperature, 1e-6)
            next_tok = int(logits.argmax(dim=-1).item())
            tokens.append(next_tok)
            if eos_id is not None and next_tok == eos_id:
                break
            # NOTE: we don't update the bank with newly-generated tokens
            # for this v1 -- the bank is built from the prompt only.
            # This matches a "encoder-only context, decoder-only output"
            # design like an encoder-decoder transformer. Updating the
            # bank with generated tokens is a v2 feature.
        return torch.tensor(tokens, device=device, dtype=torch.long)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
