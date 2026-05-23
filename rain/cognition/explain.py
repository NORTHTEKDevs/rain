# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/explain.py — adapted from RCK FHRR.
"""Hallucination-free explanation: every claim cites the KB fact it came from."""

from __future__ import annotations

from dataclasses import dataclass

from rain.cognition.inference import InferenceResult


@dataclass
class Explanation:
    answer: str | None
    citations: list[tuple[str, str, str]]  # (subject, relation, object) triples
    text: str


def explain(infer_result: InferenceResult) -> Explanation:
    """Build a natural-language explanation citing each step of the inference chain."""
    if infer_result.answer is None:
        return Explanation(
            answer=None,
            citations=[],
            text="I don't have enough information to answer that.",
        )

    citations = list(infer_result.chain)
    if infer_result.source == "direct":
        s, r, o = citations[0]
        text = f"I know that {s} {r.replace('_', ' ')} {o} directly from a stored fact."
    elif infer_result.source == "inherited":
        steps = []
        for s, r, o in citations:
            steps.append(f"{s} {r.replace('_', ' ')} {o}")
        text = "Reasoning: " + "; ".join(steps) + "."
    elif infer_result.source == "transitive":
        steps = [f"{s} {r.replace('_', ' ')} {o}" for s, r, o in citations]
        text = "Transitive chain: " + " -> ".join(steps) + "."
    else:
        text = f"Answer: {infer_result.answer}. (No citation chain available.)"

    return Explanation(answer=infer_result.answer, citations=citations, text=text)
