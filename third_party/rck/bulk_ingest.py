"""High-throughput knowledge ingestion.

For real-scale knowledge bases (ConceptNet has ~3M assertions, Wikidata
has ~100M, DBpedia has ~3B triples). Pure-Python with the existing
ShardedKnowledgeBase substrate, this module pushes thousands of facts
per second on a single CPU core.

Two operations:
  1. `bulk_load(kb, source)` streams a JSONL or CSV file of triples.
     Each line has fields {s, r, o} (case-insensitive); other fields
     are ignored.
  2. `auto_symmetrize(kb)` walks the KB and adds the inverse of every
     fact in a curated set of inverse-relation pairs. So
     (shakespeare, wrote, hamlet) automatically gets paired with
     (hamlet, author, shakespeare). This is the cheap way to broaden
     question coverage without rewriting source data.

Inverse-relation table is small and explicit (no learning); add to it
as needed.
"""
from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterable
from pathlib import Path

from rck.knowledge_base import ShardedKnowledgeBase


# (forward_relation, inverse_relation) -- bidirectional auto-storage.
# We deliberately keep this conservative; entries here MUST be true
# inverses, otherwise the KB gets polluted with false symmetries.
INVERSE_PAIRS: list[tuple[str, str]] = [
    ("wrote", "author"),
    ("capital", "capitalof"),
    ("parent", "child"),
    ("spouse", "spouse"),     # symmetric
    ("sibling", "sibling"),   # symmetric
    ("friend", "friend"),     # symmetric
    ("partof", "haspart"),
    ("locatedin", "contains"),
    ("madeof", "componentof"),
    ("causes", "causedby"),
    ("isa", "hassubtype"),
    ("teaches", "studentof"),
    ("invented", "inventedby"),
    ("founded", "foundedby"),
    ("painted", "painter"),
    ("composed", "composer"),
    ("directed", "director"),
    ("starred_in", "starring"),
]

# A lookup so we can answer "what's the inverse of R?" in O(1).
_INVERSE_LOOKUP: dict[str, str] = {}
for fwd, inv in INVERSE_PAIRS:
    _INVERSE_LOOKUP[fwd] = inv
    _INVERSE_LOOKUP[inv] = fwd


def inverse_relation(relation: str) -> str | None:
    return _INVERSE_LOOKUP.get(relation.lower())


# ---------------------------------------------------------------------------
#  Bulk loaders
# ---------------------------------------------------------------------------

def bulk_load_jsonl(kb: ShardedKnowledgeBase, path: str | Path,
                    symmetrize: bool = True, max_facts: int | None = None) -> dict:
    """Stream a JSONL file of {s, r, o} into the KB."""
    path = Path(path)
    n = 0; n_sym = 0
    t0 = time.time()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = str(rec["s"]).lower()
            r = str(rec["r"]).lower()
            o = str(rec["o"]).lower()
            kb.store({"S": s, "R": r, "O": o})
            n += 1
            if symmetrize and (inv := inverse_relation(r)):
                if inv != r or s != o:  # avoid duplicate symmetric
                    kb.store({"S": o, "R": inv, "O": s})
                    n_sym += 1
            if max_facts is not None and n >= max_facts:
                break
    return {"facts": n, "symmetrized": n_sym, "elapsed_s": time.time() - t0}


def bulk_load_csv(kb: ShardedKnowledgeBase, path: str | Path,
                  symmetrize: bool = True, max_facts: int | None = None,
                  delimiter: str = ",") -> dict:
    """Stream a CSV file (s,r,o columns) into the KB."""
    path = Path(path)
    n = 0; n_sym = 0
    t0 = time.time()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            s = str(row.get("s", "")).strip().lower()
            r = str(row.get("r", "")).strip().lower()
            o = str(row.get("o", "")).strip().lower()
            if not (s and r and o):
                continue
            kb.store({"S": s, "R": r, "O": o})
            n += 1
            if symmetrize and (inv := inverse_relation(r)):
                if inv != r or s != o:
                    kb.store({"S": o, "R": inv, "O": s})
                    n_sym += 1
            if max_facts is not None and n >= max_facts:
                break
    return {"facts": n, "symmetrized": n_sym, "elapsed_s": time.time() - t0}


def bulk_load_triples(kb: ShardedKnowledgeBase,
                       triples: Iterable[tuple[str, str, str]],
                       symmetrize: bool = True) -> dict:
    """In-memory ingestion path -- useful for tests + programmatic data."""
    n = 0; n_sym = 0
    t0 = time.time()
    for s, r, o in triples:
        s, r, o = s.lower(), r.lower(), o.lower()
        kb.store({"S": s, "R": r, "O": o})
        n += 1
        if symmetrize and (inv := inverse_relation(r)):
            if inv != r or s != o:
                kb.store({"S": o, "R": inv, "O": s})
                n_sym += 1
    return {"facts": n, "symmetrized": n_sym, "elapsed_s": time.time() - t0}


# ---------------------------------------------------------------------------
#  Post-hoc symmetrization (when data was loaded without it)
# ---------------------------------------------------------------------------

def auto_symmetrize(kb: ShardedKnowledgeBase) -> int:
    """Walk every shard's fact list and add missing inverses.

    Returns the number of new inverse facts added.
    """
    added = 0
    seen: set[tuple[str, str, str]] = set()
    for shard in kb._shards:
        for fact in list(shard._facts):  # snapshot
            s = str(fact.get("S", "")).lower()
            r = str(fact.get("R", "")).lower()
            o = str(fact.get("O", "")).lower()
            if not (s and r and o):
                continue
            inv = inverse_relation(r)
            if not inv:
                continue
            key = (o, inv, s)
            if key in seen:
                continue
            seen.add(key)
            kb.store({"S": o, "R": inv, "O": s})
            added += 1
    return added
