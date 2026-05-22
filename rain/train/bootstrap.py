# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Phase 1 bootstrap orchestrator.

Composes:
  1.1 Codebook warm-start (scripts/bootstrap_warm_start)
  1.2 KB seeding (scripts/seed_kb)
  1.3 HYMN gradient pre-train (scripts/pretrain_hymn) -- OPTIONAL here; this
      orchestrator only INITIALIZES HYMN structure. Real gradient pre-train
      is a separate long-running call.
  1.4 SOFAR routing init (mapper + encoder + beam) -- identity-at-init
  1.5 Tsetlin clause seeding -- empty population, ready for Type-I feedback
  1.6 LSM RLS init -- zero readout, default reservoir
  1.7 FEP A init -- random low-rank factors

Returns a RainBootstrap dataclass holding all initialized components.

The structural init is fast (seconds). The expensive part is 1.3 HYMN
gradient pre-train which runs separately via scripts/pretrain_hymn.py.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

from rain.core.relational import Codebook
from rain.core.knowledge_base import ShardedKB
from rain.core.tsetlin import TsetlinMachine
from rain.core.liquid_state import LiquidStateMachine
from rain.core.fep import LowRankA
from rain.routing.mapper import RoutingMapper
from rain.routing.beam import BeamSteeringAdapter
from rain.tokenize.bpe import BPETokenizer
from rain.train.warm_start import warm_start_from_vectors
from rain.data.kb_seed import seed_from_jsonl


@dataclass
class RainBootstrap:
    """Container for all bootstrapped components."""
    config: dict
    tokenizer: BPETokenizer
    codebook: Codebook
    kb: ShardedKB
    tsetlin: TsetlinMachine
    lsm: LiquidStateMachine
    fep: LowRankA
    routing_mapper: RoutingMapper
    beam_adapter: BeamSteeringAdapter

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the bootstrap state."""
        return {
            "config": self.config,
            "codebook": {"vocab_size": self.codebook.vocab_size, "dim": self.codebook.dim},
            "kb": {"num_shards": self.kb.num_shards, "dim": self.kb.dim},
            "tsetlin": {
                "num_classes": self.tsetlin.num_classes,
                "num_clauses_per_class": self.tsetlin.num_clauses_per_class,
                "num_features": self.tsetlin.num_features,
            },
            "lsm": {
                "input_dim": self.lsm.input_dim,
                "reservoir_dim": self.lsm.reservoir_dim,
                "output_dim": self.lsm.output_dim,
            },
            "fep": {"D": self.fep.D, "R": self.fep.R},
            "beam": {"k": self.beam_adapter.k, "D": self.beam_adapter.D},
        }


def bootstrap_phase1(
    corpus_texts: list[str],
    kb_jsonl_path: str | None = None,
    warm_start_vectors: dict[str, np.ndarray] | None = None,
    *,
    D: int = 10000,
    vocab_size: int = 256,
    num_shards: int = 64,
    tsetlin_classes: int = 16,
    tsetlin_clauses_per_class: int = 32,
    lsm_reservoir: int = 512,
    fep_rank: int = 128,
    routing_k: int = 64,
    bpe_vocab: int = 32000,
    seed: int = 42,
) -> RainBootstrap:
    """End-to-end Phase 1 init. Fast -- does NOT run gradient pre-training.

    Parameters mirror the design doc's defaults; override per workload.
    """
    config = {
        "D": D,
        "vocab_size": vocab_size,
        "num_shards": num_shards,
        "tsetlin_classes": tsetlin_classes,
        "tsetlin_clauses_per_class": tsetlin_clauses_per_class,
        "lsm_reservoir": lsm_reservoir,
        "fep_rank": fep_rank,
        "routing_k": routing_k,
        "bpe_vocab": bpe_vocab,
        "seed": seed,
    }

    # 1.0 Tokenizer
    tokenizer = BPETokenizer(vocab_size=bpe_vocab)
    tokenizer.train(corpus_texts)

    # 1.1 Codebook
    codebook = Codebook(vocab_size=vocab_size, dim=D, seed=seed)
    if warm_start_vectors:
        warm_start_from_vectors(codebook, warm_start_vectors)

    # 1.2 KB
    kb = ShardedKB(num_shards=num_shards, dim=D, seed=seed)
    if kb_jsonl_path:
        seed_from_jsonl(kb, kb_jsonl_path)

    # 1.5 Tsetlin -- empty clauses, ready for feedback
    tsetlin = TsetlinMachine(
        num_classes=tsetlin_classes,
        num_clauses_per_class=tsetlin_clauses_per_class,
        num_features=D,
        seed=seed,
    )

    # 1.6 LSM
    lsm = LiquidStateMachine(
        input_dim=D, reservoir_dim=lsm_reservoir, output_dim=D, seed=seed,
    )

    # 1.7 FEP
    fep = LowRankA(D=D, R=fep_rank, seed=seed)

    # 1.4 SOFAR routing -- mapper builds SVD over codebook + roles; beam adapter identity-at-init
    routing_mapper = RoutingMapper(k=routing_k)
    beam_adapter = BeamSteeringAdapter(k=routing_k, D=D, lora_rank=4, seed=seed)

    return RainBootstrap(
        config=config,
        tokenizer=tokenizer,
        codebook=codebook,
        kb=kb,
        tsetlin=tsetlin,
        lsm=lsm,
        fep=fep,
        routing_mapper=routing_mapper,
        beam_adapter=beam_adapter,
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run RAIN Phase 1 bootstrap")
    parser.add_argument("--corpus", type=str, required=True, help="path to corpus text")
    parser.add_argument("--kb", type=str, default=None, help="path to KB JSONL")
    parser.add_argument("--D", type=int, default=10000)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-out", type=str, default="bootstrap_summary.json")
    args = parser.parse_args()

    corpus_text = Path(args.corpus).read_text()
    boot = bootstrap_phase1(
        corpus_texts=[corpus_text],
        kb_jsonl_path=args.kb,
        D=args.D,
        vocab_size=args.vocab,
        seed=args.seed,
    )
    summary = boot.summary()
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
