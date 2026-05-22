# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/think_aloud.py — adapted from RCK FHRR.
"""Chain-of-thought narration: emit the reasoning trace before the answer."""

from __future__ import annotations
from rain.cognition.inference import InferenceResult


def narrate(question: str, infer_result: InferenceResult) -> str:
    """Produce a CoT narration. Question comes in, narrated steps + answer come out."""
    lines: list[str] = [f"Q: {question}"]
    if infer_result.answer is None:
        lines.append("Thought: I don't have a stored fact for that, and inheritance/transitive walks didn't help.")
        lines.append("A: I don't know.")
        return "\n".join(lines)

    if infer_result.source == "direct":
        s, r, o = infer_result.chain[0]
        lines.append(f"Thought: I recall directly that {s} {r.replace('_', ' ')} {o}.")
    elif infer_result.source == "inherited":
        lines.append("Thought: Following the inheritance chain...")
        for s, r, o in infer_result.chain:
            lines.append(f"  - {s} {r.replace('_', ' ')} {o}")
    elif infer_result.source == "transitive":
        lines.append("Thought: Following the transitive chain...")
        for s, r, o in infer_result.chain:
            lines.append(f"  - {s} {r.replace('_', ' ')} {o}")
    lines.append(f"A: {infer_result.answer}")
    return "\n".join(lines)
