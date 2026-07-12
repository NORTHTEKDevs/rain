"""regex_extractor: pull structured tokens from unstructured text.

Handles patterns like:
    "extract emails from contact me at foo@bar.com or baz@qux.io"
    "find all phone numbers in (907) 555-1234 and 555-9876"
    "get urls from check https://example.com and http://foo.io/path"
    "extract dates from 2026-05-24 and 12/31/2025"

Returns a comma-separated list of matches per category, or a "no matches"
notice. Deterministic; never raises on malformed input.
"""

from __future__ import annotations

import re


_EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RX = re.compile(r"https?://[^\s)>]+")
_PHONE_RX = re.compile(
    r"(?:\+?1[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})"
)
_ISO_DATE_RX = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_US_DATE_RX = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")


_KIND_MAP = {
    "email": ("email", _EMAIL_RX),
    "emails": ("email", _EMAIL_RX),
    "url": ("url", _URL_RX),
    "urls": ("url", _URL_RX),
    "link": ("url", _URL_RX),
    "links": ("url", _URL_RX),
    "phone": ("phone", _PHONE_RX),
    "phones": ("phone", _PHONE_RX),
    "number": ("phone", _PHONE_RX),
    "numbers": ("phone", _PHONE_RX),
    "date": ("date", None),  # special: tries both ISO + US
    "dates": ("date", None),
}


def _extract_dates(text: str) -> list[str]:
    out: list[str] = []
    for m in _ISO_DATE_RX.finditer(text):
        out.append(m.group(0))
    for m in _US_DATE_RX.finditer(text):
        out.append(m.group(0))
    return out


def invoke(query_text: str, kb) -> str:
    if not query_text:
        return "regex_extractor: empty query"
    tl = query_text.lower()
    # Detect which kind to extract.
    kind: str | None = None
    rx = None
    for key, (k, r) in _KIND_MAP.items():
        if re.search(rf"\b{key}\b", tl):
            kind = k
            rx = r
            break
    if kind is None:
        return "regex_extractor: which kind? (email, url, phone, date)"
    # Strip the query verb prefix; we extract from whatever's left,
    # which contains the actual content.
    # Heuristic: take everything after the first occurrence of 'from'.
    body = query_text
    m = re.search(r"\bfrom\b", query_text, re.I)
    if m:
        body = query_text[m.end():]
    body = body.strip()
    if not body:
        body = query_text  # fall back to whole query

    if kind == "date":
        matches = _extract_dates(body)
    else:
        assert rx is not None
        if kind == "phone":
            matches = [
                f"({m.group(1)}) {m.group(2)}-{m.group(3)}"
                for m in rx.finditer(body)
            ]
        else:
            matches = [m.group(0) for m in rx.finditer(body)]
    if not matches:
        return f"regex_extractor: no {kind} matches in input"
    return f"{kind}: " + ", ".join(matches)
