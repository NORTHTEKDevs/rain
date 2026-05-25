# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Proper Mamba S6 selective state-space block.

What our previous SelectiveGatedRecurrence (in hymn_plus_v2.py) was:
  h_t = sigmoid(W_g @ x_t) * h_{t-1} + (1 - sigmoid(W_g @ x_t)) * (W_z @ x_t)

That's a learned input-dependent forget gate -- a useful approximation
but NOT the actual Mamba S6 mechanism.

Mamba S6 (Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces") does this per channel d:

  Δ_t      = softplus(linear_input_dependent_delta(x_t))   (B, T, D)
  A_bar_t  = exp(Δ_t * A)                                   (B, T, D, N)
  B_bar_t  = Δ_t * B_t  ;  B_t = linear_B(x_t)             (B, T, D, N)
  C_t      = linear_C(x_t)                                  (B, T, N)
  h_t      = A_bar_t * h_{t-1} + B_bar_t * x_t              (B, T, D, N)
  y_t      = (C_t @ h_t) + D * x_t                          (B, T, D)

The CRITICAL differences from gated recurrence:
  1. h is (B, T, D, N) -- N hidden state dims PER input channel D
  2. A is per-channel (D, N) -- learned dynamics per channel
  3. Δ, B, C are ALL input-dependent (selective)
  4. Proper discretization via Δ*A inside the exp

This expressiveness is what gives Mamba its scaling behavior. Our
gated recurrence was throwing it away.

Implementation: pure PyTorch with a sequential scan. SLOW on CPU
(O(T) sequential ops) but mathematically correct. For real training
at scale we'd want the parallel-scan kernel or use `mamba-ssm`'s
official implementation. For validation experiments at our current
scale (Tiny Shakespeare, dim<=256, T<=128), the pure-PyTorch path
is fast enough to A/B test against gated recurrence.

If this S6 block dramatically beats the gated recurrence at the same
parameter count, the bottleneck WAS the simplified recurrence and we
should make this our default. If not, the recurrence wasn't the
problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MambaS6Config:
    """Configuration for a single S6 block."""

    dim: int  # input/output channels (D in Mamba notation)
    d_state: int = 16  # state size per channel (N in Mamba notation)
    d_conv: int = 4  # conv kernel size for local context before SSM
    expand: int = 2  # inner expansion factor (Mamba uses 2x by default)
    dt_rank: int | str = "auto"  # rank of dt projection; "auto" = ceil(dim/16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    conv_bias: bool = True
    bias: bool = False


class MambaS6Block(nn.Module):
    """One proper Mamba S6 block.

    Structure: input -> in_proj (expand) -> (1D conv, SiLU) -> SSM -> SiLU(gate) * y -> out_proj
    Same layout as the official Mamba reference implementation.
    """

    def __init__(self, config: MambaS6Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.d_inner = config.expand * config.dim
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        if config.dt_rank == "auto":
            self.dt_rank = math.ceil(config.dim / 16)
        else:
            self.dt_rank = int(config.dt_rank)

        # Input projection: x -> [x_proj, gate]; both d_inner wide
        self.in_proj = nn.Linear(self.dim, 2 * self.d_inner, bias=config.bias)

        # Local 1-D conv along the time axis (groups=d_inner -> per-channel)
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            bias=config.conv_bias,
            padding=self.d_conv - 1,
        )

        # Projection: x_conv -> (dt, B, C) all input-dependent
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        # dt low-rank projection: dt_rank -> d_inner (with bias for init range)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # Initialize dt bias so that softplus(dt_bias) is in [dt_min, dt_max]
        dt_init_std = self.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=1e-4)
        # Inverse of softplus: log(exp(x) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Don't reinit the bias on subsequent layers' inits
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        # The A matrix is parameterized as -exp(A_log) per channel x state.
        # This guarantees A is negative -> stable continuous-time system.
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # D is the residual "skip" coefficient per channel
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        # Output projection: d_inner -> dim
        self.out_proj = nn.Linear(self.d_inner, self.dim, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = x.shape
        # 1) Input projection + split into (x_path, gate)
        xz = self.in_proj(x)  # (B, T, 2 * d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # 2) 1D causal conv along time. The conv1d expects (B, C, T).
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv(x_conv)[:, :, :T]  # crop right-side padding for causality
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_conv = F.silu(x_conv)

        # 3) SSM step. Compute dt, B, C from x_conv.
        x_dbl = self.x_proj(x_conv)  # (B, T, dt_rank + 2 * d_state)
        dt_proj, B_in, C_in = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # dt = softplus(dt_proj @ W_dt + b_dt). Shape: (B, T, d_inner)
        dt = F.softplus(self.dt_proj(dt_proj))

        # A is -exp(A_log), shape (d_inner, d_state). Negative for stability.
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Discretize: A_bar = exp(dt * A); B_bar = dt * B
        # dt: (B, T, d_inner); A: (d_inner, d_state); need (B, T, d_inner, d_state)
        A_bar = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, T, d_inner, d_state)
        B_bar = dt.unsqueeze(-1) * B_in.unsqueeze(2)  # (B, T, d_inner, d_state)

        # Sequential scan along T. h_t = A_bar_t * h_{t-1} + B_bar_t * x_conv_t
        # Then y_t = (C_t @ h_t) + D * x_conv_t
        # h is per-batch per-channel per-state: (B, d_inner, d_state)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            # broadcast x_conv[:, t]: (B, d_inner) -> (B, d_inner, 1)
            xt = x_conv[:, t].unsqueeze(-1)  # (B, d_inner, 1)
            h = A_bar[:, t] * h + B_bar[:, t] * xt
            # C_t: (B, d_state); h: (B, d_inner, d_state) -> y_t (B, d_inner)
            yt = torch.einsum("bin,bn->bi", h, C_in[:, t])
            yt = yt + self.D * x_conv[:, t]
            ys.append(yt)
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)

        # 4) Apply SiLU gate (the "selective" part of selective-SSM)
        y = y * F.silu(z)

        # 5) Output projection back to dim
        return self.out_proj(y)
