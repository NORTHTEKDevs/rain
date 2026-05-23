# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""A2 -- SOFAR-on-VSA ablation benchmark.

Tier-3 architectural soundness. Measures whether routing the bipolar VSA
state through SOFAR's projection-reconstruction (SVD beam-steering over the
codebook + role directions) helps a next-token MSE-on-HV objective.

Procedure (per docs/architecture/benchmark-suite.md A2):

    1. Build a deterministic char-level corpus + Codebook + HymnSurrogate
       (same random init for both branches; same val sample positions).
    2. Branch A (routing OFF):  forward(state, zero_input) -> MSE on HV target.
    3. Branch B (routing ON):   forward(beam.apply(state, dirs, cfg), zero_input)
                                -> MSE on HV target.
    4. improvement = (mse_off - mse_on) / mse_off.
    5. Acceptance: improvement >= 0.03 (>=3% relative reduction with routing on).
    6. Kill trigger: improvement < 0.03 -- drop SOFAR routing per design plan.

NOTE: the beam adapter ships with `identity-at-init` (lora_up=0, beam_gate=0),
which would make Branch B trivially identical to Branch A. The ablation needs
to exercise the routing math itself, so the benchmark seeds non-trivial
`lora_up` weights and `beam_gate` BEFORE measuring -- this simulates a
'post-warmup' routing configuration. With identity-at-init both branches
are bit-identical and `improvement = 0` exactly (verified in
test_routing_at_identity_init_is_noop).
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from rain.core.relational import Codebook
from rain.routing.mapper import RoutingMapper
from rain.routing.beam import BeamSteeringAdapter, BeamConfig, BeamMode
from scripts.pretrain_hymn import HymnSurrogate


@dataclass
class A2Result:
    benchmark: str
    n_eval_steps: int
    dim: int
    routing_k: int
    mse_routing_off: float
    mse_routing_on: float
    improvement: float
    pass_threshold: float
    overall_pass: bool
    kill_triggered: bool
    seed: int
    notes: list[str] = field(default_factory=list)


_DEFAULT_CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "she sells sea shells down by the sea shore. "
    "rain falls mainly on the plain in spain. "
    "to be or not to be that is the question. "
    "a stitch in time saves nine fortnights from grief. "
) * 100


def _build_codebook_and_dirs(
    dim: int, vocab_size: int, k: int, num_roles: int, seed: int
) -> tuple[Codebook, "RoutingMapper", "BeamSteeringAdapter", np.ndarray, np.ndarray]:
    """Codebook + role bank + routing mapper + beam adapter + source matrices."""
    cb = Codebook(vocab_size=vocab_size, dim=dim, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    codebook_arr = np.stack([cb.vector(f"tok_{i}") for i in range(vocab_size)]).astype(np.float32)
    role_arr = np.stack([
        np.where(rng.integers(0, 2, size=dim) == 1, 1, -1).astype(np.float32)
        for _ in range(num_roles)
    ])
    mapper = RoutingMapper(k=k)
    beam = BeamSteeringAdapter(k=k, D=dim, lora_rank=4, seed=seed + 2000)
    # Break identity-at-init so the routing path actually runs (see module docstring).
    beam.lora_up = rng.standard_normal(beam.lora_up.shape).astype(np.float32) * 0.01
    beam.beam_gate = np.float32(0.1)
    return cb, mapper, beam, codebook_arr, role_arr


def _val_mse(
    model: HymnSurrogate,
    codebook: Codebook,
    corpus: str,
    n_eval_steps: int,
    seed: int,
    *,
    route: bool = False,
    beam: "BeamSteeringAdapter | None" = None,
    directions=None,
) -> float:
    """Mean MSE-on-HV across n_eval_steps random positions.

    When route=True, `beam.apply(state, directions, cfg)` is applied to the
    state before HYMN forward.
    """
    rng = np.random.default_rng(seed)
    chars = list(corpus)
    if len(chars) < 2:
        return 0.0
    losses: list[float] = []
    cfg = BeamConfig(mode=BeamMode.FOCUS, center=float(beam.k // 2) if beam else 0.0,
                     width=float(max(1, (beam.k // 4))) if beam else 1.0)
    for _ in range(n_eval_steps):
        pos = int(rng.integers(0, len(chars) - 1))
        prev_char = chars[pos]
        next_char = chars[pos + 1]
        state = codebook.vector(prev_char).astype(np.float32)
        if route and beam is not None and directions is not None:
            state = beam.apply(state, directions, cfg)
        input_ = np.zeros_like(state)
        out, _ = model.forward(state, input_)
        target = codebook.vector(next_char).astype(np.float32)
        losses.append(float(np.mean((out - target) ** 2)))
    return float(np.mean(losses))


def run_benchmark(
    corpus: str | None = None,
    dim: int = 512,
    vocab_size: int = 64,
    routing_k: int = 32,
    num_roles: int = 16,
    hidden_dim: int = 256,
    n_eval_steps: int = 500,
    seed: int = 0,
) -> A2Result:
    """Run the A2 ablation. Returns A2Result with both branches' MSE + improvement."""
    text = corpus if corpus is not None else _DEFAULT_CORPUS
    cb, mapper, beam, cb_arr, role_arr = _build_codebook_and_dirs(
        dim=dim, vocab_size=vocab_size, k=routing_k, num_roles=num_roles, seed=seed,
    )
    model = HymnSurrogate(in_dim=dim, hidden_dim=hidden_dim, out_dim=dim, seed=seed)
    directions = mapper.get(cb_arr, role_arr)

    mse_off = _val_mse(model, cb, text, n_eval_steps, seed=seed + 99, route=False)
    mse_on = _val_mse(model, cb, text, n_eval_steps, seed=seed + 99,
                      route=True, beam=beam, directions=directions)

    if mse_off <= 0.0:
        improvement = 0.0
    else:
        improvement = (mse_off - mse_on) / mse_off

    pass_threshold = 0.03

    notes: list[str] = []
    if not np.isfinite(mse_off) or not np.isfinite(mse_on):
        notes.append(f"non-finite MSE: off={mse_off}, on={mse_on}")

    return A2Result(
        benchmark="A2_sofar_on_vsa_ablation",
        n_eval_steps=n_eval_steps,
        dim=dim,
        routing_k=routing_k,
        mse_routing_off=mse_off,
        mse_routing_on=mse_on,
        improvement=improvement,
        pass_threshold=pass_threshold,
        overall_pass=(improvement >= pass_threshold),
        kill_triggered=(improvement < pass_threshold),
        seed=seed,
        notes=notes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="A2 -- SOFAR-on-VSA ablation")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--routing-k", type=int, default=32)
    parser.add_argument("--num-roles", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-eval-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus", type=str, default=None,
                        help="path to text corpus (defaults to built-in toy)")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    corpus_text = Path(args.corpus).read_text() if args.corpus else None
    r = run_benchmark(
        corpus=corpus_text,
        dim=args.dim,
        vocab_size=args.vocab,
        routing_k=args.routing_k,
        num_roles=args.num_roles,
        hidden_dim=args.hidden_dim,
        n_eval_steps=args.n_eval_steps,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(r), indent=2))
    print(json.dumps(asdict(r), indent=2))


if __name__ == "__main__":
    main()
