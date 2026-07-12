"""A1 — EFE decoder source-mix sanity benchmark.

100-prompt probe across a small synthetic environment. Each of the 7
candidate sources is seeded with non-trivial data so its contribution is
detectable. Asserts each source contributes >= 5% of the total weighted
fused-score across the 100 prompts (excluding HYMN which is a future hookup
point — for v0 we exclude it from the bar but still measure)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from rain.core.bigram import BigramMemory
from rain.core.efe import MultiSignalEFEDecoder
from rain.core.fep import LowRankA
from rain.core.knowledge_base import ShardedKB
from rain.core.liquid_state import LiquidStateMachine
from rain.core.relational import Codebook


@dataclass
class A1Result:
    benchmark: str
    n_prompts: int
    source_contribution_pct: dict[str, float] = field(default_factory=dict)
    sources_passing: list[str] = field(default_factory=list)
    sources_below_threshold: list[str] = field(default_factory=list)
    overall_pass: bool = False


def run_benchmark(n_prompts: int = 100, seed: int = 0) -> A1Result:
    rng = np.random.default_rng(seed)
    D = 1024
    vocab = [f"tok_{i}" for i in range(32)]
    cb = Codebook(vocab_size=64, dim=D, seed=seed)

    # Seed each component with distinct synthetic data
    kb = ShardedKB(num_shards=8, dim=D, seed=seed)
    # KB facts: (tok_0, rel_0) -> tok_5, etc. Cycle through pairs.
    for i in range(0, 28, 2):
        kb.write(vocab[i], "next", vocab[(i + 5) % 32])

    lsm = LiquidStateMachine(input_dim=D, reservoir_dim=64, output_dim=D, seed=seed)
    # Step LSM a few times so its state is non-trivial
    for _ in range(10):
        lsm.step(cb.vector(vocab[int(rng.integers(32))]).astype(np.float32))

    bigram = BigramMemory(cb, order=2)
    for i in range(0, 28, 2):
        bigram.add([vocab[i], "next"], vocab[(i + 3) % 32])

    fep = LowRankA(D=D, R=8, seed=seed)
    # Update FEP a few times so it's not pure-noise
    for _ in range(5):
        s = cb.vector(vocab[int(rng.integers(32))]).astype(np.float32)
        t = cb.vector(vocab[int(rng.integers(32))]).astype(np.float32)
        fep.update(s, t, alpha=0.05)

    # Synthetic Tsetlin vote: prefers vocabulary half determined by sum of state
    def tsetlin_vote_fn(state: np.ndarray, vocab_: list[str]) -> list[tuple[str, float]]:
        sign = float(state.sum())
        out: list[tuple[str, float]] = []
        for tok in vocab_:
            hv = cb.vector(tok).astype(np.float32)
            score = float(hv.sum() * sign) / D
            out.append((tok, score))
        return out

    # Synthetic crystal recall: uses cosine to a few stored states
    crystals: list[tuple[np.ndarray, str]] = [
        (cb.vector(vocab[i]).astype(np.float32), vocab[(i + 7) % 32])
        for i in range(0, 32, 4)
    ]

    def crystal_recall_fn(state: np.ndarray, vocab_: list[str]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        s = state.astype(np.float32)
        best_tok, best_sim = None, -np.inf
        for stored_state, stored_tok in crystals:
            sim = float(np.dot(s, stored_state) / D)
            if sim > best_sim:
                best_sim = sim
                best_tok = stored_tok
        if best_tok is None:
            return []
        # Return all vocab tokens scored by similarity-to-best (target gets 1.0)
        return [(t, 1.0 if t == best_tok else 0.0) for t in vocab_]

    decoder = MultiSignalEFEDecoder(
        codebook=cb, kb=kb, lsm=lsm, bigram=bigram, fep=fep,
        tsetlin_vote_fn=tsetlin_vote_fn,
        crystal_recall_fn=crystal_recall_fn,
    )

    totals: dict[str, float] = dict.fromkeys(("hymn", "kb", "lsm", "bigram", "fep", "tsetlin", "crystal"), 0.0)
    grand_total = 0.0
    for i in range(n_prompts):
        state = cb.vector(vocab[i % 32]).astype(np.float32) * (1.0 + 0.01 * rng.standard_normal(D))
        recent = [vocab[(i - 1) % 32], "next"]
        result = decoder.decode(state, vocab=vocab, recent_context=recent, rng=rng)
        for src, contrib in result.source_contributions.items():
            totals[src] += contrib
            grand_total += contrib

    if grand_total <= 0.0:
        return A1Result(
            benchmark="A1_efe_source_mix",
            n_prompts=n_prompts,
            source_contribution_pct=dict.fromkeys(totals, 0.0),
            overall_pass=False,
        )

    pct = {src: 100.0 * v / grand_total for src, v in totals.items()}
    # For v0 acceptance, exclude HYMN (future hookup); require each other source >= 5%
    bar_sources = [s for s in totals if s != "hymn"]
    passing = [s for s in bar_sources if pct[s] >= 5.0]
    failing = [s for s in bar_sources if pct[s] < 5.0]
    overall = len(failing) == 0

    return A1Result(
        benchmark="A1_efe_source_mix",
        n_prompts=n_prompts,
        source_contribution_pct=pct,
        sources_passing=passing,
        sources_below_threshold=failing,
        overall_pass=overall,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="A1 — EFE source-mix sanity benchmark")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    r = run_benchmark(n_prompts=args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(r), indent=2))
    print(json.dumps(asdict(r), indent=2))


if __name__ == "__main__":
    main()
