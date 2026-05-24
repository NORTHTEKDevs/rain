# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Adapter that wraps a trained HYMN-Plus v2 checkpoint as a ConsciousAgent
fluency engine -- the v2 version of HymnPlusSampler.

The big v2 difference: when the agent calls tell(), we can also push the
new fact's hypervector into the model's KB-attention buffer so generation
immediately reflects the new knowledge -- no retraining, no fine-tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class HymnPlusV2Sampler:
    """(prompt, n_tokens) -> str fluency engine backed by HYMN-Plus v2."""

    def __init__(
        self,
        model,
        tokenizer,
        meta: dict,
        *,
        temperature: float = 0.7,
        top_k: int = 30,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.meta = meta
        self.temperature = temperature
        self.top_k = top_k

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        temperature: float = 0.7,
        top_k: int = 30,
    ) -> HymnPlusV2Sampler:
        import torch

        from rain.core.hymn_plus_v2 import HymnPlusV2, HymnPlusV2Config
        from rain.tokenize.bpe import BPETokenizer

        ckpt = Path(checkpoint_path)
        meta_path = ckpt.with_suffix(".json")
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("arch") != "hymn_plus_v2":
            raise ValueError(f"checkpoint arch={meta.get('arch')!r} is not 'hymn_plus_v2'")
        cfg = HymnPlusV2Config(
            vocab_size=meta["vocab_size"],
            dim=meta["dim"],
            n_layers=meta["n_layers"],
            mlp_mult=meta["mlp_mult"],
            kb_size=meta["kb_size"],
            kb_top_k=meta["kb_top_k"],
            kb_attn_in_layers=(
                tuple(meta["kb_attn_in_layers"]) if meta.get("kb_attn_in_layers") else None
            ),
            tie_weights=meta.get("tie_weights", True),
            seed=meta.get("seed", 42),
        )
        model = HymnPlusV2(cfg)
        sd = dict(np.load(ckpt))
        model.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
        model.eval()

        bpe_path = meta.get("bpe_model_path") or str(ckpt.with_suffix(".bpe.model"))
        tok = BPETokenizer(vocab_size=meta.get("bpe_vocab", cfg.vocab_size))
        tok.load(bpe_path)

        return cls(model, tok, meta, temperature=temperature, top_k=top_k)

    def set_kb_from_facts(self, facts: list[tuple[str, str, str]], seed: int = 0) -> int:
        """Build a KB matrix from (subject, relation, object) triples using
        RAIN's codebook + bind/bundle, then push it into all KB-attention
        blocks. Returns the number of facts loaded.

        Killer feature: call this after agent.tell(...) and the next sample()
        call reflects the new knowledge -- zero retraining.
        """
        import torch

        from rain.core.relational import Codebook, bind, bundle

        kb_size = self.meta["kb_size"]
        dim = self.meta["dim"]
        cb = Codebook(vocab_size=8192, dim=dim, seed=seed)

        rows: list[np.ndarray] = []
        for s, r, o in facts:
            sv = cb.vector(str(s))
            rv = cb.vector(str(r))
            ov = cb.vector(str(o))
            rows.append(bundle([bind(sv, rv), ov]).astype(np.float32))
            if len(rows) >= kb_size:
                break

        if not rows:
            return 0
        if len(rows) < kb_size:
            rng = np.random.default_rng(seed + 1)
            pad = (rng.integers(0, 2, size=(kb_size - len(rows), dim)) * 2 - 1).astype(np.float32)
            rows = rows + list(pad)

        kb = np.stack(rows[:kb_size], axis=0)
        self.model.set_kb(torch.as_tensor(kb, dtype=torch.float32))
        return min(len(facts), kb_size)

    def __call__(self, prompt: str, n_tokens: int) -> str:
        import torch

        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [0]
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = self.model.sample(
                x,
                max_new=int(n_tokens),
                temperature=self.temperature,
                top_k=self.top_k,
            )
        generated_ids = out[0, len(ids) :].tolist()
        return self.tokenizer.decode(generated_ids)
