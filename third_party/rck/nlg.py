"""Natural-language generation from (S, R, O) triples.

A small template renderer that turns retrieved structured answers into
conversational sentences. Each relation has 1-3 alternative templates
chosen by a deterministic hash of the input so the same question always
gets the same phrasing (no randomness mid-conversation).

The aim isn't GPT-quality prose. It's that "blue" becomes "The sky is blue."
"""
from __future__ import annotations

import hashlib


# Per-relation templates. {s} = subject, {o} = object.
TEMPLATES: dict[str, list[str]] = {
    "color":     ["The {s} is {o}.", "{s} is {o} in color."],
    "is":        ["The {s} is {o}.", "{s} is a {o}."],
    "isa":       ["A {s} is a {o}.", "The {s} is a kind of {o}."],
    "kind":      ["A {s} is a {o}.", "The {s} is a kind of {o}."],
    "has":       ["The {s} has {o}.", "{s} has {o}."],
    "wrote":     ["{s} wrote {o}.", "The author {s} is known for {o}."],
    "author":    ["{o} wrote {s}.", "{s} was written by {o}."],
    "capital":   ["The capital of {s} is {o}.", "{o} is the capital of {s}."],
    "lives_in":  ["{s} lives in {o}.", "{s} resides in {o}."],
    "locatedin": ["{s} is in {o}.", "{s} is located in {o}."],
    "continent": ["{s} is in {o}.", "{s} is on the continent of {o}."],
    "madeof":    ["{s} is made of {o}.", "The {s} is made of {o}."],
    "usedfor":   ["{s} is used for {o}.", "The {s} is used for {o}."],
    "causes":    ["{s} causes {o}.", "When there is {s}, {o} follows."],
    "field":     ["{s} worked in the field of {o}.",
                  "{s} is associated with {o}."],
    "value":     ["The value of {s} is {o}.", "{s} equals {o}."],
    "partof":    ["{s} is part of {o}.", "{s} is a component of {o}."],
    "category":  ["{s} is a {o}.", "The {s} is a kind of {o}."],
    "version":   ["The version of {s} is {o}.", "{s} is at version {o}."],
}

# Fallback template when relation isn't known.
DEFAULT_TEMPLATE = "The {r} of {s} is {o}."


def render(subject: str, relation: str, obj: str) -> str:
    """Choose a template and substitute. Deterministic on (s, r, o)."""
    subject = _human(subject); obj = _human(obj)
    rel = relation.lower()
    candidates = TEMPLATES.get(rel)
    if not candidates:
        return DEFAULT_TEMPLATE.format(s=subject, r=relation, o=obj)
    # Deterministic pick: hash chooses one of the templates.
    key = f"{subject}|{relation}|{obj}".encode("utf-8")
    idx = int.from_bytes(hashlib.blake2b(key, digest_size=1).digest(), "little") % len(candidates)
    return candidates[idx].format(s=subject, o=obj)


def render_chain(chain: list[tuple[str, str, str]]) -> str:
    """Render a multi-hop inference chain as prose."""
    if not chain:
        return ""
    parts = [render(s, r, o) for s, r, o in chain]
    return " ".join(parts)


def render_enumeration(items: list[str], relation: str, value: str,
                       max_items: int = 8) -> str:
    """Render an enumeration result: 'The animals that are mammals are X, Y, Z.'"""
    value = _human(value)
    if not items:
        return f"I don't know of any {value} {relation}."
    items = [_human(x) for x in items[:max_items]]
    if len(items) == 1:
        return f"{items[0]} is a {value}."
    body = ", ".join(items[:-1]) + f", and {items[-1]}"
    return f"Things that are a {value}: {body}."


def _human(token: str) -> str:
    """Render an internal symbol as a natural string."""
    s = str(token).replace("_", " ")
    return s
