"""Uniform local update rule for HYMN cells.

For each cell i at iteration t:
    perception[i] = soft_sign( rho_0 * h[i] + sum_j rho_j * h[N(i,j)] )    # bundle of neighbors
    attended[i]   = bind(h[i], perception[i]) / sqrt(D)                    # F11 fix: normalize
    phi[i]        = MLP_theta( concat([attended_norm[i], type_embed[i]]) )
    h_new[i]      = soft_sign( phi[i] + alpha_zone(i) * h[i] )

The same MLP is applied to every cell. The type embedding (concatenated with the
attended hypervector) breaks the symmetry so that zone behavior can emerge.

In training: soft_sign = tanh(beta * x). At inference: soft_sign = sign(x).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from vsa_core.ops import bind


class LocalUpdateRule(nn.Module):
    """Uniform MLP local update rule for HYMN cells.

    Applied to all cells of the field at every iteration. Parameters are shared
    across cells.
    """

    def __init__(
        self,
        d: int,
        hidden: int = 2048,
        n_zones: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d = d
        self.hidden = hidden

        # Neighbor mixing weights: self + 4 neighbors -> 5 scalars.
        self.rho = nn.Parameter(torch.full((5,), 0.2))

        # MLP: input is concat([attended/sqrt(D), type_embed]) = 2D.
        self.fc1 = nn.Linear(2 * d, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Per-zone residual gate (3 zones).
        self.alpha = nn.Parameter(torch.full((n_zones,), 1.0))

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def _soft_sign(self, x: Tensor, beta: float, soft: bool) -> Tensor:
        if soft:
            return torch.tanh(beta * x)
        return torch.sign(x)

    def forward(
        self,
        state: Tensor,             # (B, N, D)
        neighbors: Tensor,         # (N, 4) long; static per field config
        type_embeddings: Tensor,   # (N, D); static
        zones: Tensor,             # (N,) long; static
        beta: float = 1.0,
        soft: bool = True,
    ) -> Tensor:
        """One iteration of the uniform update rule.

        Returns the new field state of shape (B, N, D).
        """
        B, N, D = state.shape
        # Gather neighbors: (B, N, 4, D)
        neighbor_hvs = state[:, neighbors, :]  # (B, N, 4, D)
        # Bundle self + 4 neighbors with learned scalar weights.
        # rho[0] for self, rho[1:5] for neighbors.
        self_w = self.rho[0]
        nbr_w = self.rho[1:5]  # (4,)
        # weighted neighbors sum
        weighted_nbrs = (neighbor_hvs * nbr_w.view(1, 1, 4, 1)).sum(dim=2)  # (B, N, D)
        perception_pre = self_w * state + weighted_nbrs  # (B, N, D)
        perception = self._soft_sign(perception_pre, beta=beta, soft=soft)

        # BIND state with perception (circular conv), normalized by sqrt(D) (F11).
        # bind() returns float-valued result of order sqrt(D), so divide.
        attended = bind(state, perception) / math.sqrt(D)  # (B, N, D)

        # MLP input: concat attended with type embeddings (broadcast across batch).
        te = type_embeddings.unsqueeze(0).expand(B, N, D)  # (B, N, D)
        mlp_in = torch.cat([attended, te], dim=-1)         # (B, N, 2D)

        h = self.fc1(mlp_in)
        h = torch.relu(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = torch.relu(h)
        h = self.drop(h)
        phi = self.fc3(h)                                  # (B, N, D)

        # Per-zone residual gate.
        alpha_per_cell = self.alpha[zones]                 # (N,)
        pre = phi + alpha_per_cell.view(1, N, 1) * state   # (B, N, D)

        return self._soft_sign(pre, beta=beta, soft=soft)
