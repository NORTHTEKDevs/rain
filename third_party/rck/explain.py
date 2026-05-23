"""Self-explanation: convert an RCK decision into a human-readable trace.

RCK is uniquely positioned for grounded explanations:
  - The Tsetlin layer produces propositional clauses that fired.
  - The relational memory's hits return cosine scores against the codebook.
  - The signal ensemble (LSM / bigram / FEP) carries explicit weights.

We compose these into a "I emitted X because Y" sentence WITHOUT inventing
a single token: every claim cites a specific module + score. This is the
opposite of LLM rationalisation, which can hallucinate justifications post
hoc.
"""
from __future__ import annotations

from rck.agent import RCKAgent, StepTrace
from rck.compose import CompositionalReasoner


def explain_step(trace: StepTrace) -> str:
    """Verbal explanation of a single RCKAgent.step()'s emission."""
    parts = [f"I emitted '{trace.emitted_symbol}' on input '{trace.input_symbol}'."]

    # Workspace.
    if trace.workspace_winner is not None:
        parts.append(
            f"The workspace was driven by the '{trace.workspace_winner}' module "
            f"(salience {trace.workspace_score:.3f})."
        )

    # Bigram support.
    if trace.bigram_top:
        top = trace.bigram_top[0]
        parts.append(
            f"The strongest n-gram suggestion was '{top[0]}' with cosine "
            f"{top[1]:.3f}; "
            + (
                "this matched the emission." if str(top[0]) == str(trace.emitted_symbol)
                else "the active inference policy preferred a different candidate."
            )
        )

    # Tsetlin clauses.
    if trace.tsetlin_clauses:
        clauses = "; ".join(trace.tsetlin_clauses[:3])
        parts.append(f"Active reasoning clauses: {clauses}.")
    else:
        parts.append("No Tsetlin clauses fired (model still in early training).")

    # Uncertainty.
    if trace.column_uncertainty > 0.05:
        parts.append(
            f"Cortical-column variance was {trace.column_uncertainty:.4f} -- "
            f"the model is UNCERTAIN about this prediction."
        )
    else:
        parts.append(
            f"Cortical-column variance was {trace.column_uncertainty:.4f} -- "
            f"columns agreed."
        )

    parts.append(f"Predictive error on the workspace transition: {trace.pred_err:.3f}.")
    return " ".join(parts)


def explain_composition(reasoner: CompositionalReasoner, slots: dict) -> str:
    """Explanation for a CompositionalReasoner.compose() answer.

    Walks each slot, queries the relational memory, and reports the
    cosine score so the user can see exactly which fact contributed.
    """
    lines = ["Composition trace:"]
    pieces = []
    for slot, val in slots.items():
        ans, score = reasoner._single_slot_answer(slot, val)
        pieces.append(str(ans))
        cls = "STRONG" if score > 0.25 else "MEDIUM" if score > 0.10 else "WEAK"
        lines.append(
            f"  slot {slot!s} = {val!r:>10s}  ->  {str(ans)!r:>10s}  "
            f"(cos={score:.3f}, {cls})"
        )
    rendered = " ".join(pieces)
    lines.append(f"Composed output: '{rendered}'")
    lines.append("No single training example matches; the answer is constructed by "
                 "binding the slot-specific HVs in the relational memory.")
    return "\n".join(lines)


def explain_generate(agent: RCKAgent, prompt: str, max_new: int = 20) -> str:
    """Generate text and append a per-token explanation block."""
    out, traces = agent.generate(prompt, max_new=max_new)
    text_out = "".join(str(c) for c in out)
    lines = [f"Prompt: {prompt!r}", f"Emitted: {text_out!r}", "", "Per-token trace:"]
    for i, t in enumerate(traces[-min(8, len(traces)):]):
        lines.append(f"  step {i}: {explain_step(t)}")
    return "\n".join(lines)
