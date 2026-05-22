"""Synonym normalization for relations + entities.

Allows users to phrase queries in different ways and have them route
to the same underlying KB facts. "What hue is the sky?" should hit the
same relation as "What color is the sky?".

We use a small static lookup for high-precision common cases and a
canonicalization function that lowercases + replaces underscores with
spaces, then matches.

Two levels:
  - RELATION synonyms: hue -> color, writer -> author, situated_in -> locatedin
  - ENTITY synonyms / aliases: e.g. uk -> england, usa -> unitedstates
"""
from __future__ import annotations


# Relation aliases. Map any common phrasing to the canonical relation name
# used in our KB schema.
RELATION_ALIASES: dict[str, str] = {
    # Color
    "hue": "color", "colour": "color", "shade": "color",
    # Size
    "magnitude": "size",
    # Has / parts
    "owns": "has", "possesses": "has",
    "contains": "haspart", "has_part": "haspart",
    # Authorship
    "writer": "author", "wrote_by": "author",
    "creator": "author", "created_by": "author",
    "writer_of": "author",
    "authored": "wrote", "written_by": "author",
    # Location
    "in": "locatedin", "located_in": "locatedin",
    "situated_in": "locatedin", "found_in": "locatedin",
    "place": "locatedin",
    # Lives in (people)
    "resides_in": "lives_in", "lives": "lives_in",
    # Capital
    "capital_of": "capitalof",
    # Material
    "material": "madeof", "made_of": "madeof", "consists_of": "madeof",
    "composition": "madeof",
    # Use / function
    "purpose": "usedfor", "used_for": "usedfor", "function": "usedfor",
    # Kind / category
    "type": "isa", "kind": "isa", "class": "isa",
    "species": "isa", "category": "category",
    # Time
    "preceded_by": "previousmonth", "comes_before": "previousmonth",
    # Causality
    "leads_to": "causes", "results_in": "causes",
    "caused_by": "causedby",
}

# Entity aliases.
ENTITY_ALIASES: dict[str, str] = {
    "uk": "england",
    "united_kingdom": "england",
    "britain": "england",
    "u.s.": "usa",
    "u.s.a.": "usa",
    "united_states": "usa",
    "america": "usa",
    "the_states": "usa",
    "deutschland": "germany",
    "espana": "spain",
    "nippon": "japan",
}


def canonical_relation(name: str) -> str:
    """Map any phrasing of a relation to the canonical form."""
    n = _normalize(name)
    return RELATION_ALIASES.get(n, n)


def canonical_entity(name: str) -> str:
    """Map any phrasing of an entity to the canonical form."""
    n = _normalize(name)
    return ENTITY_ALIASES.get(n, n)


def _normalize(s: str) -> str:
    return s.strip().lower().replace(" ", "_")
