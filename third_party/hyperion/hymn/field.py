"""Toroidal grid of VSA cells with zone partitioning.

A field is a (H, W, D) tensor of cell hypervectors on a torus (wraparound
both axes). Cells are partitioned into three functional zones via fixed
random type embeddings that are concatenated to the MLP input. The update
rule is uniform across all cells (NCA-style) -- zone behavior emerges from
the type embedding bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class Zone(Enum):
    INPUT = 0
    COMPUTATION = 1
    OUTPUT = 2


@dataclass
class FieldConfig:
    height: int = 8
    width: int = 8
    d: int = 1_000
    # zone partition: cells [0:n_input) are input, [n_input:n_input+n_comp)
    # are computation, the rest are output. Defaults: 25% / 50% / 25%.
    n_input: int = 16
    n_comp: int = 32
    # n_output = h*w - n_input - n_comp (computed)
    type_embed_seed: int = 1


def make_type_embeddings(cfg: FieldConfig) -> Tensor:
    """Frozen random bipolar HV per cell, used as a type embedding."""
    n_cells = cfg.height * cfg.width
    g = torch.Generator(device="cpu").manual_seed(cfg.type_embed_seed)
    raw = torch.randint(0, 2, (n_cells, cfg.d), generator=g, dtype=torch.int8)
    return (raw * 2 - 1).to(dtype=torch.float32)


def zone_assignment(cfg: FieldConfig) -> Tensor:
    """Returns (n_cells,) long with zone index per cell."""
    n_cells = cfg.height * cfg.width
    assert cfg.n_input + cfg.n_comp < n_cells, "zone sizes exceed field"
    zones = torch.full((n_cells,), Zone.OUTPUT.value, dtype=torch.long)
    zones[: cfg.n_input] = Zone.INPUT.value
    zones[cfg.n_input : cfg.n_input + cfg.n_comp] = Zone.COMPUTATION.value
    return zones


def neighbor_indices(h: int, w: int) -> Tensor:
    """For each cell on a (h, w) torus, return its 4 Von Neumann neighbors
    (up, down, left, right) as flat indices. Shape: (h*w, 4).
    """
    n = h * w
    rows = torch.arange(n) // w
    cols = torch.arange(n) % w
    up    = ((rows - 1) % h) * w + cols
    down  = ((rows + 1) % h) * w + cols
    left  = rows * w + ((cols - 1) % w)
    right = rows * w + ((cols + 1) % w)
    return torch.stack([up, down, left, right], dim=-1)  # (n, 4)


class ToroidalField:
    """Holds a batch of field states plus the static neighbor table and zones.

    The field state itself is just a tensor; this class is the glue around it.
    """

    def __init__(self, cfg: FieldConfig, device: torch.device | str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.n_cells = cfg.height * cfg.width
        self.neighbors = neighbor_indices(cfg.height, cfg.width).to(self.device)  # (N, 4)
        self.zones = zone_assignment(cfg).to(self.device)  # (N,)
        self.type_embeddings = make_type_embeddings(cfg).to(self.device)  # (N, D)
        self.input_mask = (self.zones == Zone.INPUT.value)
        self.output_mask = (self.zones == Zone.OUTPUT.value)
        self.comp_mask = (self.zones == Zone.COMPUTATION.value)

    def init_state(self, batch_size: int, seed: int | None = None) -> Tensor:
        """Random bipolar initial state. (B, N, D)."""
        g = None
        if seed is not None:
            g = torch.Generator(device="cpu").manual_seed(seed)
        raw = torch.randint(
            0, 2, (batch_size, self.n_cells, self.cfg.d),
            generator=g, dtype=torch.int8,
        )
        return (raw * 2 - 1).to(dtype=torch.float32, device=self.device)

    def write_input_zone(
        self,
        state: Tensor,         # (B, N, D)
        token_hv: Tensor,      # (B, D)
        input_gates: Tensor,   # (N_input,) in [0,1], learned
        beta: float = 1.0,
    ) -> Tensor:
        """Mix the new token's HV into input-zone cells via a learned per-cell gate.

        Vectorized: state[input_cells] <- tanh(beta * (gate * tok + (1-gate) * old))
        """
        input_idx = torch.where(self.input_mask)[0]    # (n_input,)
        gates = input_gates.clamp(0.0, 1.0)            # (n_input,)
        input_cells = state[:, input_idx, :]           # (B, n_input, D)
        tok = token_hv.unsqueeze(1)                    # (B, 1, D)
        g = gates.view(1, -1, 1)                       # (1, n_input, 1)
        mixed = g * tok + (1 - g) * input_cells        # (B, n_input, D)
        new_input_cells = torch.tanh(beta * mixed)     # (B, n_input, D)
        new_state = state.clone()
        new_state[:, input_idx, :] = new_input_cells
        return new_state

    def output_mean(self, state: Tensor) -> Tensor:
        """Mean of output-zone cells. (B, N, D) -> (B, D)."""
        return state[:, self.output_mask, :].mean(dim=1)

    def output_bundle(self, state: Tensor, beta: float = 1.0,
                      use_all_cells: bool = False) -> Tensor:
        """VSA bundle of output-zone cells: tanh(beta * sum). (B, N, D) -> (B, D).

        Bundle (sum + soft-sign) preserves bipolar structure better than mean
        when the number of cells is small (e.g., 4-16). Mean of N tanh-outputs
        has per-component magnitude ~1/sqrt(N), washing out the sign pattern
        that the cosine-sim cleanup needs to match against the bipolar codebook.

        With use_all_cells=True, bundles over the entire field instead of just
        the output zone. This relaxes the zone bottleneck at small scale.
        """
        if use_all_cells:
            s = state.sum(dim=1)
        else:
            s = state[:, self.output_mask, :].sum(dim=1)
        return torch.tanh(beta * s)
