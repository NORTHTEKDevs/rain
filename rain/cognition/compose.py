# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/compose.py — adapted from RCK FHRR.
"""CompositionalReasoner: slot-based composition over orthogonal feature axes.

Each feature axis (e.g., 'color', 'shape', 'quantity', 'size', 'material')
is a separate role hypervector. A composed concept is the bundle of bind(role, value)
pairs across all slot-value assignments. Querying for a slot value unbinds and
finds the codebook nearest neighbor.

This is the mechanism behind RCK v1.1's 100% SCAN compositional generalization."""

from __future__ import annotations

import numpy as np

from rain.core.relational import Codebook, bind, bundle


class CompositionalReasoner:
    def __init__(self, codebook: Codebook, slot_names: list[str]) -> None:
        self.codebook = codebook
        self.slot_names = slot_names
        # Role hypervectors: one per slot
        self.roles: dict[str, np.ndarray] = {
            slot: codebook.vector(f"__role_{slot}__") for slot in slot_names
        }

    def compose(self, slot_values: dict[str, str]) -> np.ndarray:
        """Build a composite HV from slot->value bindings."""
        if not slot_values:
            raise ValueError("at least one slot required")
        binds: list[np.ndarray] = []
        for slot, value in slot_values.items():
            if slot not in self.roles:
                raise ValueError(f"unknown slot: {slot}")
            role_hv = self.roles[slot]
            value_hv = self.codebook.vector(value)
            binds.append(bind(role_hv, value_hv))
        return bundle(binds)

    def extract(self, composite: np.ndarray, slot: str, candidates: list[str]) -> str:
        """Given a composite HV, unbind by slot role, find nearest candidate."""
        if slot not in self.roles:
            raise ValueError(f"unknown slot: {slot}")
        role_hv = self.roles[slot]
        # Unbind = bind for bipolar
        unbound = bind(composite, role_hv)
        # Nearest candidate by cosine
        D = self.codebook.dim
        best_token = candidates[0]
        best_sim = -np.inf
        for tok in candidates:
            cand = self.codebook.vector(tok)
            sim = float(np.dot(unbound.astype(np.float32), cand.astype(np.float32)) / D)
            if sim > best_sim:
                best_sim = sim
                best_token = tok
        return best_token

    def compose_sum(self, slot_values: dict[str, str]) -> np.ndarray:
        """Like compose() but returns the UNSIGNED real superposition of binds.

        Required for resonant (explaining-away) readout, which needs the linear
        superposition rather than the sign-quantised bundle.
        """
        if not slot_values:
            raise ValueError("at least one slot required")
        acc = np.zeros(self.codebook.dim, dtype=np.float64)
        for slot, value in slot_values.items():
            if slot not in self.roles:
                raise ValueError(f"unknown slot: {slot}")
            acc += bind(self.roles[slot], self.codebook.vector(value)).astype(np.float64)
        return acc

    def extract_all(self, composite_sum: np.ndarray, slots: list[str],
                    candidates: list[str], max_iters: int = 25) -> dict[str, str]:
        """Jointly decode ALL slots via resonant explaining-away.

        Decodes every slot at once, re-reading each after subtracting the others'
        reconstructions. Recovers materially more slots at a given dimension than
        per-slot extract() once cross-talk dominates (Hyperion Finding 5). Operates
        on the unsigned superposition from compose_sum().
        """
        from rain.core.resonator import resonant_extract  # noqa: PLC0415

        roles = np.stack([self.roles[s] for s in slots])
        cb = np.stack([self.codebook.vector(c) for c in candidates])
        idx = resonant_extract(composite_sum, roles, cb, max_iters=max_iters)
        return {s: candidates[idx[i]] for i, s in enumerate(slots)}
