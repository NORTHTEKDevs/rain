"""Think-aloud: explicit chain-of-thought rendered as natural-language.

When the user asks a question or invokes verbose mode, RCK can narrate
the steps it took to arrive at the answer. This is the equivalent of
LLM chain-of-thought prompting but ARCHITECTURAL: every step is a real
graph operation and the trace can be audited.

The narration follows a fixed structure:
  1. "Let me think about [QUESTION]..."
  2. For each step in the chain: "I know that [fact]."
  3. Connective: "So [intermediate conclusion]."
  4. Final: "Therefore [answer]."
"""
from __future__ import annotations

from rck.inference import InferenceResult
from rck.nlg import render


def narrate(question: str, result: InferenceResult) -> str:
    """Render an InferenceResult as a step-by-step explanation."""
    if result.answer is None:
        return ("Let me think about that... I don't have any facts that "
                "answer this question directly, and chain inference also "
                "comes up empty.")
    lines = [f"Let me think about \"{question}\"..."]
    if not result.chain:
        lines.append(f"I have it directly: the answer is {result.answer}.")
        return " ".join(lines)
    if result.source == "direct":
        lines.append(f"I know that {render(*result.chain[0])}")
        lines.append(f"So the answer is {result.answer}.")
        return " ".join(lines)
    if result.source == "transitive":
        # chain[0] = (S, R, mid), chain[1] = (mid, R, ans)
        s, r, mid = result.chain[0]
        _, _, ans = result.chain[1]
        lines.append(f"I know that {render(s, r, mid)}")
        lines.append(f"And I know that {render(mid, r, ans)}")
        lines.append(f"So by transitivity, {render(s, r, ans)}")
        return " ".join(lines)
    if result.source == "inherited":
        s, _, ancestor = result.chain[0]
        _, r, ans = result.chain[1]
        lines.append(f"I know that {render(s, 'isa', ancestor)}")
        lines.append(f"And I know that {render(ancestor, r, ans)}")
        lines.append(f"So I can infer that {render(s, r, ans)}")
        return " ".join(lines)
    # Fallback: dump the chain.
    for step in result.chain:
        lines.append(f"I know that {render(*step)}")
    lines.append(f"Therefore, the answer is {result.answer}.")
    return " ".join(lines)


def narrate_no_match(question: str, attempts: list[str]) -> str:
    """Render a 'I don't know' explanation with what was tried."""
    lines = [f"Let me think about \"{question}\"..."]
    if attempts:
        lines.append(f"I tried looking under: {', '.join(attempts)}.")
    lines.append("None of those gave me a confident answer, so I don't know.")
    return " ".join(lines)
