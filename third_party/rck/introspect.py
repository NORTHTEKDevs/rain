"""Introspection layer.

Two complementary capabilities:

1. **Broadcast history**: the Global Workspace Theory implementation in
   `rck/workspace.py` already runs a winner-take-all competition each
   step. We wrap it with a bounded ring buffer of the last N broadcasts
   so the model can report "what have I just been thinking about".

2. **`think()` summary**: a natural-language description of the model's
   CURRENT internal state -- recent workspace winners, last reasoning
   trace, confidence on the last query. This is not consciousness; it's
   grounded introspection: every line is a citation to a specific
   structural fact about the model's state.

Together with the self-model (`rck/self_model.py`) this gives RCK a
queryable "I" -- it knows what it is, knows what it just did, and can
report both in plain language.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rck.agent import RCKAgent, StepTrace


@dataclass
class IntrospectionBuffer:
    """Bounded ring of recent broadcasts + per-step diagnostics."""

    max_history: int = 32
    _history: deque = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.max_history)

    def record(self, trace: StepTrace) -> None:
        self._history.append({
            "input": trace.input_symbol,
            "emitted": trace.emitted_symbol,
            "winner": trace.workspace_winner,
            "winner_score": trace.workspace_score,
            "uncertainty": trace.column_uncertainty,
            "tsetlin_score": trace.tsetlin_score,
            "pred_err": trace.pred_err,
        })

    def recent(self, n: int = 8) -> list[dict]:
        return list(self._history)[-n:]

    def clear(self) -> None:
        self._history.clear()

    def stats(self) -> dict:
        if not self._history:
            return {"steps": 0}
        unc = [h["uncertainty"] for h in self._history]
        err = [h["pred_err"] for h in self._history]
        winners = [h["winner"] for h in self._history]
        winner_counts: dict[str, int] = {}
        for w in winners:
            if w is None:
                continue
            winner_counts[w] = winner_counts.get(w, 0) + 1
        return {
            "steps": len(self._history),
            "mean_uncertainty": sum(unc) / len(unc),
            "mean_pred_err": sum(err) / len(err),
            "winner_distribution": winner_counts,
        }


def think(agent: RCKAgent, buf: IntrospectionBuffer, last_query: str | None = None) -> str:
    """Return a natural-language summary of RCK's current internal state."""
    s = buf.stats()
    lines = ["My internal state right now:"]
    if last_query:
        lines.append(f"  - I was just asked: {last_query!r}")
    if s["steps"] == 0:
        lines.append("  - I haven't taken any steps yet.")
        return "\n".join(lines)
    lines.append(f"  - I've taken {s['steps']} recent steps.")
    lines.append(f"  - My mean prediction error is {s['mean_pred_err']:.3f}.")
    lines.append(f"  - My mean column-vote uncertainty is {s['mean_uncertainty']:.4f} "
                 f"({'agreement' if s['mean_uncertainty'] < 0.05 else 'some disagreement'}).")
    if s["winner_distribution"]:
        top = sorted(s["winner_distribution"].items(), key=lambda x: -x[1])[:3]
        winners_str = ", ".join(f"{name}({n})" for name, n in top)
        lines.append(f"  - Top broadcast modules: {winners_str}.")
    recent = buf.recent(4)
    if recent:
        last = recent[-1]
        lines.append(f"  - Most recent emission: {last['emitted']!r} on input {last['input']!r}.")
    lines.append(f"  - Codebook size: {agent.codebook.size()} symbols.")
    return "\n".join(lines)
