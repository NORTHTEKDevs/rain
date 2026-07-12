"""A5 -- Python-vs-Rust HYMN forward correctness benchmark.

Asserts that the Rust HYMN forward and a Python reference produce bit-
identical bipolar outputs on the SAME (W1, W2, state, input) inputs, modulo
float-epsilon in intermediate accumulators (which does not affect sign).

A5 acceptance: 100/100 random bipolar (state, input) pairs match exactly.

Full WASM/Go/TS dispatch correctness deferred until those bindings exist.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from rain.spine import _RUST_AVAILABLE, HymnModel, _python_reference_forward


@dataclass
class A5Result:
    benchmark: str
    n_pairs: int
    n_matched: int
    match_rate: float
    overall_pass: bool
    diff_positions: list = field(default_factory=list)


def run_benchmark(
    n_pairs: int = 100,
    in_dim: int = 64,
    hidden_dim: int = 32,
    out_dim: int = 64,
    seed: int = 0,
) -> A5Result:
    if not _RUST_AVAILABLE:
        raise RuntimeError("rain._rust not available")
    model = HymnModel(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, seed=seed)
    W1, W2 = model.weights()
    rng = np.random.default_rng(seed)
    matched = 0
    diff_positions: list[int] = []
    for i in range(n_pairs):
        state = rng.choice([-1, 1], size=in_dim).astype(np.int16)
        input_ = rng.choice([-1, 1], size=in_dim).astype(np.int16)
        rust_out = model.forward(state, input_)
        py_out = _python_reference_forward(W1, W2, state, input_)
        if np.array_equal(rust_out, py_out):
            matched += 1
        else:
            diff_positions.append(i)
    rate = matched / n_pairs
    return A5Result(
        benchmark="A5_dispatch_correctness_py_vs_rust",
        n_pairs=n_pairs,
        n_matched=matched,
        match_rate=rate,
        overall_pass=(rate == 1.0),
        diff_positions=diff_positions[:10],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="A5 -- dispatch correctness")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--in-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    r = run_benchmark(
        n_pairs=args.n,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(r), indent=2))
    print(json.dumps(asdict(r), indent=2))


if __name__ == "__main__":
    main()
