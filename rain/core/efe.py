"""Multi-signal Expected Free Energy decoder.

Fuses 7 candidate sources into a single next-token distribution:
  1. HYMN prediction (via state-to-codebook cosine — optional, future hookup)
  2. LSM recurrent prediction
  3. Bigram VSA n-gram
  4. FEP-acted (low-rank A predicted state then cosine)
  5. KB lookup (sharded HRR fact recall)
  6. Tsetlin clause vote (categorical prior — optional)
  7. Crystal recall (embedding-similarity from a crystal store — optional)

See docs/design/multi-signal-efe-decoder.md for the full math.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rain.core.bigram import BigramMemory
from rain.core.fep import LowRankA
from rain.core.knowledge_base import ShardedKB
from rain.core.liquid_state import LiquidStateMachine
from rain.core.relational import Codebook


class EpistemicClass(enum.StrEnum):
    KNOW = "know"
    THINK = "think"
    GUESS = "guess"
    UNKNOWN = "unknown"


DEFAULT_WEIGHTS: dict[str, float] = {
    "hymn": 4.0,
    "kb": 3.0,
    "lsm": 2.0,
    "crystal": 2.5,
    "fep": 1.5,
    "tsetlin": 1.0,
    "bigram": 0.5,
}


@dataclass
class DecodeResult:
    chosen_token: str | None
    confidence: float
    epistemic: EpistemicClass
    source_contributions: dict[str, float] = field(default_factory=dict)
    fusion_distribution: dict[str, float] = field(default_factory=dict)
    candidates_per_source: dict[str, list[tuple[str, float]]] = field(default_factory=dict)


def _rank_normalize(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Map scores to [0, 1] by rank position (best=1, worst=0)."""
    if not items:
        return []
    sorted_items = sorted(items, key=lambda x: -x[1])
    n = len(sorted_items)
    if n == 1:
        return [(sorted_items[0][0], 1.0)]
    return [(tok, 1.0 - (i / (n - 1))) for i, (tok, _) in enumerate(sorted_items)]


def _anti_repetition_penalty(
    token: str,
    last_token: str | None,
    last_two: tuple[str, str] | None,
    single_penalty: float = 0.3,
    bigram_cycle_penalty: float = 0.5,
) -> float:
    penalty = 0.0
    if last_token == token:
        penalty += single_penalty
    if last_two and last_two[0] == token:
        penalty += bigram_cycle_penalty
    return penalty


class MultiSignalEFEDecoder:
    def __init__(
        self,
        codebook: Codebook,
        kb: ShardedKB | None = None,
        lsm: LiquidStateMachine | None = None,
        bigram: BigramMemory | None = None,
        fep: LowRankA | None = None,
        tsetlin_vote_fn: Callable[[np.ndarray, list[str]], list[tuple[str, float]]] | None = None,
        crystal_recall_fn: Callable[[np.ndarray, list[str]], list[tuple[str, float]]] | None = None,
        hymn_predict_fn: Callable[[np.ndarray, list[str]], list[tuple[str, float]]] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.codebook = codebook
        self.kb = kb
        self.lsm = lsm
        self.bigram = bigram
        self.fep = fep
        self.tsetlin_vote_fn = tsetlin_vote_fn
        self.crystal_recall_fn = crystal_recall_fn
        self.hymn_predict_fn = hymn_predict_fn
        self.weights = weights or dict(DEFAULT_WEIGHTS)

    # ---- per-source candidate generators ----

    def _candidates_hymn(self, state: np.ndarray, vocab: list[str]) -> list[tuple[str, float]]:
        if self.hymn_predict_fn is None:
            return []
        return self.hymn_predict_fn(state, vocab)

    def _candidates_kb(
        self, recent_context: list[str], vocab: list[str]
    ) -> list[tuple[str, float]]:
        if self.kb is None or not recent_context:
            return []
        if len(recent_context) >= 2:
            s, r = recent_context[-2], recent_context[-1]
            o = self.kb.query(s, r)
            if o and o in vocab:
                # Spread scores across all vocab: retrieved token = 1.0, rest = 0.0
                return [(t, 1.0 if t == o else 0.0) for t in vocab]
        return []

    def _candidates_lsm(self, state: np.ndarray, vocab: list[str]) -> list[tuple[str, float]]:
        if self.lsm is None:
            return []
        pred = self.lsm.predict()
        # Cosine of LSM prediction vs each vocab codebook entry
        pred_f = pred.astype(np.float32)
        pred_norm = np.linalg.norm(pred_f)
        out: list[tuple[str, float]] = []
        for tok in vocab:
            hv = self.codebook.vector(tok).astype(np.float32)
            hv_norm = np.linalg.norm(hv)
            if pred_norm > 0 and hv_norm > 0:
                # pred may be shorter than hv (reservoir output_dim vs codebook dim)
                min_len = min(len(pred_f), len(hv))
                sim = float(np.dot(pred_f[:min_len], hv[:min_len]) / (pred_norm * hv_norm))
            else:
                sim = 0.0
            out.append((tok, sim))
        return out

    def _candidates_bigram(
        self, recent_context: list[str], vocab: list[str]
    ) -> list[tuple[str, float]]:
        if self.bigram is None or len(recent_context) < self.bigram.order:
            return []
        return self.bigram.query(recent_context, candidate_tokens=vocab, top_k=len(vocab))

    def _candidates_fep(self, state: np.ndarray, vocab: list[str]) -> list[tuple[str, float]]:
        if self.fep is None:
            return []
        pred = self.fep.predict(state.astype(np.float32))
        pred_f = pred.astype(np.float32)
        pred_norm = np.linalg.norm(pred_f)
        out: list[tuple[str, float]] = []
        for tok in vocab:
            hv = self.codebook.vector(tok).astype(np.float32)
            hv_norm = np.linalg.norm(hv)
            if pred_norm > 0 and hv_norm > 0:
                sim = float(np.dot(pred_f, hv) / (pred_norm * hv_norm))
            else:
                sim = 0.0
            out.append((tok, sim))
        return out

    def _candidates_tsetlin(self, state: np.ndarray, vocab: list[str]) -> list[tuple[str, float]]:
        if self.tsetlin_vote_fn is None:
            return []
        return self.tsetlin_vote_fn(state, vocab)

    def _candidates_crystal(self, state: np.ndarray, vocab: list[str]) -> list[tuple[str, float]]:
        if self.crystal_recall_fn is None:
            return []
        return self.crystal_recall_fn(state, vocab)

    # ---- fusion + decode ----

    def decode(
        self,
        state: np.ndarray,
        vocab: list[str],
        recent_context: list[str] | None = None,
        top_p: float = 0.9,
        rng: np.random.Generator | None = None,
    ) -> DecodeResult:
        """Decode the next token from the multi-signal candidate fusion.

        Args:
            state: current hypervector state (D-dim).
            vocab: candidate token strings.
            recent_context: recent token history for bigram + KB + anti-rep.
            top_p: nucleus sampling mass threshold.
            rng: optional. If None, a fresh wall-clock-seeded generator is used
                (non-deterministic). Pass an explicit Generator for reproducibility.
        """
        if rng is None:
            rng = np.random.default_rng()
        recent_context = recent_context or []

        raw = {
            "hymn": self._candidates_hymn(state, vocab),
            "kb": self._candidates_kb(recent_context, vocab),
            "lsm": self._candidates_lsm(state, vocab),
            "bigram": self._candidates_bigram(recent_context, vocab),
            "fep": self._candidates_fep(state, vocab),
            "tsetlin": self._candidates_tsetlin(state, vocab),
            "crystal": self._candidates_crystal(state, vocab),
        }
        normalized = {src: _rank_normalize(c) for src, c in raw.items()}

        # Fuse: total score per token = sum_src(weight[src] * rank_norm[tok, src])
        fused: dict[str, float] = {}
        source_contributions: dict[str, float] = dict.fromkeys(raw, 0.0)
        for src, items in normalized.items():
            w = self.weights.get(src, 0.0)
            for tok, score in items:
                fused[tok] = fused.get(tok, 0.0) + w * score
                source_contributions[src] += w * score

        # Anti-repetition
        last_token = recent_context[-1] if recent_context else None
        last_two = tuple(recent_context[-2:]) if len(recent_context) >= 2 else None
        for tok in list(fused.keys()):
            fused[tok] -= _anti_repetition_penalty(tok, last_token, last_two)

        if not fused:
            return DecodeResult(
                chosen_token=None,
                confidence=0.0,
                epistemic=EpistemicClass.UNKNOWN,
                source_contributions=source_contributions,
                fusion_distribution=fused,
                candidates_per_source=raw,
            )

        # Softmax + top-p sample
        items = sorted(fused.items(), key=lambda x: -x[1])
        scores = np.array([s for _, s in items], dtype=np.float64)
        # Numerical stability
        scores = scores - scores.max()
        probs = np.exp(scores) / np.exp(scores).sum()
        cumulative = np.cumsum(probs)
        cutoff_idx = int(np.searchsorted(cumulative, top_p))
        cutoff_idx = max(cutoff_idx, 0)
        nucleus = items[: cutoff_idx + 1]
        nucleus_probs = probs[: cutoff_idx + 1]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()
        choice = rng.choice(len(nucleus), p=nucleus_probs)
        chosen_token = nucleus[choice][0]
        confidence = float(nucleus_probs[choice])

        # Epistemic class
        if confidence > 0.85:
            ec = EpistemicClass.KNOW
        elif confidence > 0.5:
            ec = EpistemicClass.THINK
        elif confidence > 0.2:
            ec = EpistemicClass.GUESS
        else:
            ec = EpistemicClass.UNKNOWN

        return DecodeResult(
            chosen_token=chosen_token,
            confidence=confidence,
            epistemic=ec,
            source_contributions=source_contributions,
            fusion_distribution=fused,
            candidates_per_source=raw,
        )
