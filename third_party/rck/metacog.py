"""Meta-cognition: calibrated confidence + verbalized epistemic state.

A real generative AI should distinguish between:
  - "I know this with high confidence"
  - "I think so but I'm not certain"
  - "I have no relevant memory and would be guessing"

Rather than returning a single float, the model returns a category that
maps to a deliberate epistemic phrase. The thresholds are tunable and
the per-query category is tracked for downstream calibration analysis.

We also keep a running per-relation calibration tally: of the times we
said "I know", how often is the answer correct vs. wrong? This lets the
model improve its OWN confidence reports over time (a primitive form of
metacognitive learning).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


# Default confidence thresholds.
THRESHOLD_KNOW = 0.20
THRESHOLD_THINK = 0.10
THRESHOLD_GUESS = 0.05


def epistemic_category(confidence: float,
                       know: float = THRESHOLD_KNOW,
                       think: float = THRESHOLD_THINK,
                       guess: float = THRESHOLD_GUESS) -> str:
    """Map a raw confidence into one of {know, think, guess, unknown}."""
    if confidence >= know:
        return "know"
    if confidence >= think:
        return "think"
    if confidence >= guess:
        return "guess"
    return "unknown"


def verbalize(answer: str | None, confidence: float, source: str | None = None) -> str:
    """Render a natural-language response with calibrated hedging."""
    cat = epistemic_category(confidence)
    if cat == "unknown" or answer is None:
        return "I don't know."
    if cat == "guess":
        return f"I'm not really sure, but maybe {answer}."
    if cat == "think":
        return f"I think {answer}, but I'm not certain."
    # 'know'
    if source and source.startswith("structured-via"):
        return f"I'm pretty sure it's {answer} (inferred via the '{source.split('-via-')[1]}' relation)."
    return f"I know it's {answer}."


@dataclass
class CalibrationTally:
    """Tracks claims-vs-correctness so the model can self-monitor."""

    by_relation: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(
        lambda: {"know_right": 0, "know_wrong": 0,
                 "think_right": 0, "think_wrong": 0,
                 "guess_right": 0, "guess_wrong": 0,
                 "unknown_skipped": 0}
    ))

    def record(self, relation: str, confidence: float, correct: bool | None) -> None:
        cat = epistemic_category(confidence)
        bucket = self.by_relation[relation]
        if correct is None:
            bucket["unknown_skipped" if cat == "unknown" else f"{cat}_right"] += 1
            return
        suffix = "right" if correct else "wrong"
        bucket[f"{cat}_{suffix}"] += 1

    def calibration_score(self, relation: str) -> float:
        """Of the times we said 'know', what fraction were correct?"""
        b = self.by_relation.get(relation, {})
        denom = b.get("know_right", 0) + b.get("know_wrong", 0)
        if denom == 0:
            return 0.0
        return b["know_right"] / denom

    def summary(self) -> dict:
        out = {}
        for rel, bucket in self.by_relation.items():
            total_known = bucket.get("know_right", 0) + bucket.get("know_wrong", 0)
            out[rel] = {
                **bucket,
                "calibration_score": (bucket["know_right"] / total_known
                                      if total_known else None),
            }
        return out
