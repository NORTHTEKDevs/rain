"""Knowledge provenance: every fact carries metadata about its origin.

This is one of the structural advantages RCK has over LLMs. An LLM has
no record of WHERE it learned anything, WHEN, or WITH WHAT CONFIDENCE.
RCK tracks all three explicitly, attached to every triple.

Metadata schema (kept light to avoid bloat):
  - source:        the document/conversation/file the fact came from
  - timestamp:     when the fact was stored (unix epoch)
  - confidence:    0..1 -- how much we trust it
  - count:         how many times we've seen this fact reinforced
  - last_seen:     most recent timestamp it was confirmed
  - tags:          optional labels (e.g. {"medical", "user_provided"})

Storage: a dict keyed by (S, R, O) -> ProvenanceRecord. Separate from
the HRR memory so it doesn't blow the hypervector budget.

This module is intentionally simple. The big design decisions (decay,
consolidation, source ranking) are pushed up into higher-level modules
that consume provenance metadata.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProvenanceRecord:
    """Metadata attached to a single fact."""

    source: str = "user"
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    count: int = 1
    last_seen: float = field(default_factory=time.time)
    tags: set[str] = field(default_factory=set)
    # For source in {"induced", "rule"}: the chain of (s, r, o) tuples
    # that derived this fact. Empty for user-asserted facts.
    derivation: list[tuple[str, str, str]] = field(default_factory=list)

    def reinforce(self, source: str | None = None) -> None:
        """Mark the fact as seen again -- increment count, refresh time."""
        self.count += 1
        self.last_seen = time.time()
        if source and source != self.source:
            # Track multiple sources by switching to "multi" once we have more.
            self.source = "multi" if self.source != source else self.source
        # Reinforced facts increment confidence asymptotically toward 1.0.
        self.confidence = min(1.0, self.confidence + (1.0 - self.confidence) * 0.1)

    def decay(self, factor: float = 0.99) -> None:
        """Apply a small confidence decay -- used by the forgetting curve."""
        self.confidence = max(0.0, self.confidence * factor)


@dataclass
class ProvenanceStore:
    """Holds ProvenanceRecord per (S, R, O) triple.

    The store is independent from the HRR memory: removing a fact from
    HRR doesn't auto-remove the provenance entry (it just gets stale).
    Callers should call `forget()` here too when forgetting a fact.
    """

    _records: dict[tuple[str, str, str], ProvenanceRecord] = field(default_factory=dict)

    def store(self, s: str, r: str, o: str, *,
              source: str = "user", confidence: float = 1.0,
              tags: set[str] | None = None,
              derivation: list[tuple[str, str, str]] | None = None
              ) -> ProvenanceRecord:
        key = (s.lower(), r.lower(), o.lower())
        if key in self._records:
            self._records[key].reinforce(source)
            if tags:
                self._records[key].tags.update(tags)
            if derivation:
                self._records[key].derivation = list(derivation)
            return self._records[key]
        rec = ProvenanceRecord(
            source=source, confidence=confidence,
            tags=set(tags) if tags else set(),
            derivation=list(derivation) if derivation else [],
        )
        self._records[key] = rec
        return rec

    def get(self, s: str, r: str, o: str) -> ProvenanceRecord | None:
        return self._records.get((s.lower(), r.lower(), o.lower()))

    def forget(self, s: str, r: str, o: str) -> bool:
        return self._records.pop((s.lower(), r.lower(), o.lower()), None) is not None

    def all_for_subject(self, s: str) -> list[tuple[tuple[str, str, str], ProvenanceRecord]]:
        s_l = s.lower()
        return [(k, v) for k, v in self._records.items() if k[0] == s_l]

    def decay_all(self, factor: float = 0.99) -> int:
        """Apply uniform decay to every record. Returns number affected."""
        for rec in self._records.values():
            rec.decay(factor)
        return len(self._records)

    def low_confidence_facts(self, threshold: float = 0.1) -> list[tuple[str, str, str]]:
        """Return facts whose confidence has fallen below `threshold`.

        Higher layers can call this to garbage-collect stale memories.
        """
        return [k for k, v in self._records.items() if v.confidence < threshold]

    def size(self) -> int:
        return len(self._records)

    def summarize(self, s: str, r: str, o: str) -> str:
        """Human-readable provenance line."""
        rec = self.get(s, r, o)
        if rec is None:
            return "no record"
        age = time.time() - rec.timestamp
        return (f"source={rec.source}  confidence={rec.confidence:.2f}  "
                f"reinforced={rec.count}x  age={age:.0f}s")

    def save(self, path) -> int:
        """Persist the store to a JSONL file."""
        import json
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for (s, r, o), rec in self._records.items():
                obj = {
                    "subject": s,
                    "relation": r,
                    "object": o,
                    "source": rec.source,
                    "timestamp": rec.timestamp,
                    "confidence": rec.confidence,
                    "count": rec.count,
                    "last_seen": rec.last_seen,
                    "tags": sorted(rec.tags),
                    "derivation": [list(d) for d in rec.derivation],
                }
                f.write(json.dumps(obj) + "\n")
                n += 1
        return n

    def load(self, path, *, replace: bool = False) -> int:
        """Restore the store from a JSONL file produced by save()."""
        import json
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return 0
        if replace:
            self._records.clear()
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec_d = json.loads(line)
                key = (
                    str(rec_d["subject"]).lower(),
                    str(rec_d["relation"]).lower(),
                    str(rec_d["object"]).lower(),
                )
                self._records[key] = ProvenanceRecord(
                    source=str(rec_d.get("source", "unknown")),
                    timestamp=float(rec_d.get("timestamp", time.time())),
                    confidence=float(rec_d.get("confidence", 1.0)),
                    count=int(rec_d.get("count", 1)),
                    last_seen=float(rec_d.get("last_seen", time.time())),
                    tags=set(rec_d.get("tags", [])),
                    derivation=[tuple(d) for d in rec_d.get("derivation", [])],
                )
                n += 1
        return n
