# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/nlg.py — adapted from RCK FHRR.
"""Per-relation natural-language template renderer.

render(subject, relation, object) -> sentence. Deterministic template
selection: same triple always produces the same sentence (hash-based)."""

from __future__ import annotations
import hashlib


# Relation -> list of template strings with {s} and {o} slots
RELATION_TEMPLATES: dict[str, list[str]] = {
    "isa": ["{s} is a {o}.", "{s} is a kind of {o}."],
    "capital_of": ["{s} is the capital of {o}.", "the capital of {o} is {s}."],
    "locatedin": ["{s} is located in {o}.", "{s} is in {o}."],
    "color": ["the color of {s} is {o}.", "{s} is {o}."],
    "size": ["the size of {s} is {o}."],
    "made_of": ["{s} is made of {o}."],
    "has_part": ["{s} has a {o}.", "{o} is part of {s}."],
    "lives_in": ["{s} lives in {o}."],
    "category": ["{s} belongs to the category {o}."],
    "boils_at": ["{s} boils at {o}."],
    "creator": ["{s} was created by {o}.", "the creator of {s} is {o}."],
    "wrote": ["{s} wrote {o}."],
    "year": ["the year of {s} is {o}."],
}


def render(subject: str, relation: str, object_: str) -> str:
    """Deterministically render (s, r, o) as a sentence using a hashed template choice."""
    templates = RELATION_TEMPLATES.get(relation)
    if not templates:
        # Generic fallback
        return f"{subject} {relation.replace('_', ' ')} {object_}."
    h = hashlib.blake2b(f"{subject}|{relation}|{object_}".encode(), digest_size=4).digest()
    idx = int.from_bytes(h, "big") % len(templates)
    return templates[idx].format(s=subject, o=object_)
