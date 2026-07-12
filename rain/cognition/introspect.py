"""Global-workspace introspection: ring buffer of recent broadcast events.

The 'broadcast' is the highest-confidence event in a turn: a question,
answer, learning event, refusal, tool call. The ring buffer stores recent
broadcasts so the agent can narrate 'what just happened in my head'."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Broadcast:
    kind: str  # "ask" | "answer" | "learn" | "refuse" | "tool" | "explain" | "introspect"
    payload: dict[str, Any] = field(default_factory=dict)


class Introspector:
    def __init__(self, capacity: int = 64) -> None:
        self._buffer: deque[Broadcast] = deque(maxlen=capacity)

    def record(self, kind: str, **payload: Any) -> None:
        self._buffer.append(Broadcast(kind=kind, payload=payload))

    def recent(self, n: int | None = None) -> list[Broadcast]:
        if n is None:
            return list(self._buffer)
        return list(self._buffer)[-n:]

    def narrate(self) -> str:
        """Plain-language summary of recent broadcasts."""
        lines = []
        for b in self._buffer:
            if b.kind == "ask":
                lines.append(f"I was asked: {b.payload.get('question', '?')}")
            elif b.kind == "answer":
                lines.append(f"I answered: {b.payload.get('answer', '?')}")
            elif b.kind == "learn":
                lines.append(f"I learned: {b.payload.get('fact', '?')}")
            elif b.kind == "refuse":
                lines.append(f"I refused: {b.payload.get('reason', '?')}")
            elif b.kind == "tool":
                lines.append(f"I used tool: {b.payload.get('name', '?')}")
            elif b.kind == "explain":
                lines.append(f"I explained: {b.payload.get('topic', '?')}")
            else:
                lines.append(f"I {b.kind}.")
        return "\n".join(lines)
