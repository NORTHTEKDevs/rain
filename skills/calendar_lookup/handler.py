# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""calendar_lookup: timezone math + common-date lookups.

Patterns:
    "what time is it in Tokyo when it's 9am in New York"
    "what is UTC offset for Anchorage"
    "is 2024 a leap year"
    "how many days in 2026"
    "what quarter is 2026-05-24 in"
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone


# Common timezone offsets (hours from UTC). Not DST-aware (v0.1 keeps simple).
_TZ_OFFSETS = {
    "utc": 0, "gmt": 0,
    "london": 0,
    "paris": 1, "berlin": 1, "rome": 1, "madrid": 1, "amsterdam": 1, "warsaw": 1,
    "athens": 2, "helsinki": 2, "cairo": 2,
    "moscow": 3,
    "dubai": 4,
    "karachi": 5, "mumbai": 5,
    "bangkok": 7,
    "beijing": 8, "shanghai": 8, "singapore": 8, "perth": 8,
    "tokyo": 9, "seoul": 9,
    "sydney": 10, "melbourne": 10,
    "auckland": 12,
    "honolulu": -10, "anchorage": -9,
    "los_angeles": -8, "san_francisco": -8, "seattle": -8, "vancouver": -8,
    "denver": -7, "phoenix": -7,
    "chicago": -6, "mexico_city": -6,
    "new_york": -5, "boston": -5, "toronto": -5, "atlanta": -5,
    "buenos_aires": -3, "sao_paulo": -3,
}


def _norm_city(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def invoke(query_text: str, kb) -> str:
    if not query_text:
        return "calendar_lookup: empty query"
    tl = query_text.lower()

    # Leap year query
    m = re.search(r"\bis\s+(\d{4})\s+a?\s*leap year", tl)
    if m:
        y = int(m.group(1))
        return f"{y} is {'a' if _is_leap(y) else 'not a'} leap year"

    # Days in year
    m = re.search(r"how many days in\s+(\d{4})", tl)
    if m:
        y = int(m.group(1))
        return f"{366 if _is_leap(y) else 365} days in {y}"

    # Quarter
    m = re.search(r"quarter.*(\d{4})-(\d{1,2})-(\d{1,2})", tl)
    if m:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        q = (d.month - 1) // 3 + 1
        return f"{d.isoformat()} is in Q{q} {d.year}"

    # UTC offset for city
    m = re.search(r"utc offset.*for\s+([a-zA-Z][a-zA-Z\s_-]+?)(?:\?|$)", tl)
    if m:
        c = _norm_city(m.group(1))
        if c in _TZ_OFFSETS:
            offset = _TZ_OFFSETS[c]
            sign = "+" if offset >= 0 else ""
            return f"{c.replace('_', ' ').title()}: UTC{sign}{offset}"
        return f"calendar_lookup: unknown city {c}"

    # Time-of-day conversion: "what time is it in X when its 9am in Y"
    m = re.search(
        r"time.*in\s+([a-zA-Z][a-zA-Z\s_-]+?)\s+when.*(\d{1,2})\s*(?:am|pm|:00)?\s+in\s+([a-zA-Z][a-zA-Z\s_-]+?)(?:\?|$)",
        tl,
    )
    if m:
        target = _norm_city(m.group(1))
        hour = int(m.group(2))
        source = _norm_city(m.group(3))
        if "pm" in tl[m.start() : m.end()]:
            hour = (hour % 12) + 12
        if target not in _TZ_OFFSETS or source not in _TZ_OFFSETS:
            return f"calendar_lookup: unknown city ({target} or {source})"
        delta = _TZ_OFFSETS[target] - _TZ_OFFSETS[source]
        new_hour = (hour + delta) % 24
        return (
            f"{target.replace('_', ' ').title()} is "
            f"{new_hour:02d}:00 when it's {hour:02d}:00 in "
            f"{source.replace('_', ' ').title()}"
        )

    return f"calendar_lookup: could not parse {query_text!r}"
