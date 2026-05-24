# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Adapter that wraps a trained HYMN-Plus checkpoint as the fluency engine
for ConsciousAgent.

Same interface as the old `_HymnSampler` in `scripts/rain_chat.py` but
backed by the new HYMN-Plus architecture (selective gated recurrence +
SwiGLU MLP + pre-norm). Use via:

    sampler = HymnPlusSampler.from_checkpoint("data/checkpoints/foo.npz")
    agent = ConsciousAgent(enable_continual=True)
    agent.attach_hymn_sampler(sampler, n_tokens=120)

The sampler exposes the (prompt: str, n_tokens: int) -> str signature so
the agent code stays architecture-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class HymnPlusSampler:
    """Stateless wrapper around a HYMN-Plus model that exposes a
    `(prompt, n_tokens) -> str` sampling callable.
    """

    def __init__(
        self,
        model,
        char_to_id: dict[str, int],
        id_to_char: dict[int, str],
        *,
        temperature: float = 0.7,
        top_k: int = 30,
    ) -> None:
        self.model = model
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char
        self.temperature = temperature
        self.top_k = top_k

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        temperature: float = 0.7,
        top_k: int = 30,
    ) -> HymnPlusSampler:
        """Load a HYMN-Plus checkpoint (npz + json) from disk.

        Heavy imports (torch, the model module) are kept local so callers
        that don't use HYMN-Plus don't pay the import cost.
        """
        import torch

        from rain.core.hymn_plus import HymnPlus, HymnPlusConfig

        ckpt = Path(checkpoint_path)
        meta_path = ckpt.with_suffix(".json")
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        if not meta_path.exists():
            raise FileNotFoundError(f"meta json not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if meta.get("arch") != "hymn_plus_v1":
            raise ValueError(
                f"checkpoint arch={meta.get('arch')!r} is not 'hymn_plus_v1'; "
                "use a sampler matching the architecture."
            )

        char_vocab = meta["char_vocab"]
        char_to_id = {c: i for i, c in enumerate(char_vocab)}
        id_to_char = {i: c for c, i in char_to_id.items()}

        cfg = HymnPlusConfig(
            vocab_size=len(char_vocab),
            dim=meta["dim"],
            n_layers=meta["n_layers"],
            mlp_mult=meta["mlp_mult"],
            seed=meta.get("seed", 42),
        )
        model = HymnPlus(cfg)
        sd = dict(np.load(ckpt))
        model.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
        model.eval()
        return cls(model, char_to_id, id_to_char, temperature=temperature, top_k=top_k)

    def __call__(self, prompt: str, n_tokens: int) -> str:
        """The HymnSamplerFn callable. Always returns a string; never raises
        on unknown prompt characters (they're skipped).
        """
        import torch

        prompt_ids = [self.char_to_id[c] for c in prompt if c in self.char_to_id]
        if not prompt_ids:
            # Empty / out-of-vocab prompt: seed with the first vocab token
            prompt_ids = [0]
        x = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.no_grad():
            out = self.model.sample(
                x,
                max_new=int(n_tokens),
                temperature=self.temperature,
                top_k=self.top_k,
            )
        generated = out[0, len(prompt_ids) :].tolist()
        return "".join(self.id_to_char[int(i)] for i in generated)
