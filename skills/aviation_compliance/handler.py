# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""aviation_compliance: AD applicability + airworthiness lookups.

A vertical-wedge skill for the AK bush-pilot use case. Queries it
handles:
    "is AD 2024-01-05 applicable to a Cessna 172"
    "what AD applies to my C172 engine"
    "list ADs for piston engines"
    "when is the next inspection for N12345"

The KB is expected to contain FAA Airworthiness Directives ingested
via scripts/ingest_faa_ads.py. The skill does lightweight matching
between the query (aircraft type, engine model, AD number) and the
ingested AD facts; returns matching citations.
"""

from __future__ import annotations

import re


_AD_NUMBER_RX = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_AIRCRAFT_RX = re.compile(r"\b(Cessna|Piper|Beechcraft|Boeing|Airbus|Cirrus|Mooney|C\d{3}|PA-\d{2,3}|N\d{1,5}[A-Z]?)\b", re.I)


def invoke(query_text: str, kb) -> str:
    if not query_text:
        return "aviation_compliance: empty query"
    if kb is None:
        return "aviation_compliance: no KB available"

    # Build the search HV for cosine-matching against KB.
    # Use a simple combined-search by encoding the query text.
    from rain.core.encoder_bank import EncoderBank

    enc = EncoderBank(dim=10_000)  # NB: must match KB dim; production uses RainNet.encoder_bank
    query_hv = enc.encode("text", query_text)

    # The KB here is the SemanticMemory instance.
    results = kb.search(query_hv, top_k=5)
    if not results:
        return "aviation_compliance: no relevant AD found in KB"

    # Format matches with AD number and tail.
    parts: list[str] = []
    ad_matches = _AD_NUMBER_RX.findall(query_text)
    aircraft_matches = [m for m in _AIRCRAFT_RX.findall(query_text)]

    found_in_kb: list[str] = []
    for score, fact in results:
        if score < 0.05:
            continue
        # Truncate long text.
        snippet = fact.text[:150] + ("..." if len(fact.text) > 150 else "")
        ad_in_fact = _AD_NUMBER_RX.search(fact.text)
        ad_no = ad_in_fact.group(1) if ad_in_fact else "(no AD#)"
        found_in_kb.append(f"[{ad_no} score={score:+.2f}] {snippet}")

    if not found_in_kb:
        return "aviation_compliance: low-confidence matches only; recommend manual review"

    header = "Matching ADs:"
    if ad_matches:
        header = f"Query referenced AD {', '.join(ad_matches)}; matches:"
    elif aircraft_matches:
        header = f"Query for {', '.join(aircraft_matches[:2])}; matches:"

    return header + "\n  " + "\n  ".join(found_in_kb[:3])
