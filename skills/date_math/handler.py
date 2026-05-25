# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""date_math: arithmetic on dates.

Handles patterns like:
    "what is 30 days from 2026-05-24"
    "days between 2026-05-24 and 2026-12-31"
    "what day is 2026-12-25"
    "today plus 90 days"

Anchors on the current date (UTC) when "today" appears.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone


_ISO_RX = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_INTERVAL_RX = re.compile(r"(\d+)\s*(day|days|week|weeks|month|months|year|years)", re.I)


def _parse_iso(text: str) -> date | None:
    m = _ISO_RX.search(text)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _all_iso(text: str) -> list[date]:
    out: list[date] = []
    for m in _ISO_RX.finditer(text):
        try:
            out.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            continue
    return out


def _interval_days(text: str) -> int | None:
    m = _INTERVAL_RX.search(text)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("day"):
        return n
    if unit.startswith("week"):
        return n * 7
    if unit.startswith("month"):
        return n * 30  # approximation; sufficient for typical Q
    if unit.startswith("year"):
        return n * 365
    return None


def invoke(query_text: str, kb) -> str:
    if not query_text:
        return "date_math: empty query"
    tl = query_text.lower()
    today = datetime.now(timezone.utc).date()

    # "days between A and B"
    if "between" in tl:
        dates = _all_iso(query_text)
        if len(dates) >= 2:
            d = abs((dates[1] - dates[0]).days)
            return f"{d} days"
        return "date_math: need two ISO dates (YYYY-MM-DD) for 'between'"

    # "N days/weeks/months from <anchor>" or "<anchor> plus N days"
    if "from" in tl or "plus" in tl or "after" in tl:
        anchor = _parse_iso(query_text)
        if anchor is None and "today" in tl:
            anchor = today
        if anchor is None:
            return "date_math: need an anchor date (or 'today')"
        offset = _interval_days(query_text)
        if offset is None:
            return "date_math: need an interval like '30 days' or '4 weeks'"
        result = anchor + timedelta(days=offset)
        return result.isoformat()

    # "<anchor> minus N days/weeks" or "N days/weeks before <anchor>"
    if "before" in tl or "minus" in tl or "ago" in tl:
        anchor = _parse_iso(query_text)
        if anchor is None and "today" in tl:
            anchor = today
        if anchor is None:
            return "date_math: need an anchor date (or 'today')"
        offset = _interval_days(query_text)
        if offset is None:
            return "date_math: need an interval like '30 days'"
        result = anchor - timedelta(days=offset)
        return result.isoformat()

    # "what day is X" -> weekday name
    if "what day" in tl or "weekday" in tl:
        anchor = _parse_iso(query_text) or today
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return f"{anchor.isoformat()} is a {names[anchor.weekday()]}"

    # "today" alone
    if tl.strip() == "today" or "what is today" in tl:
        return today.isoformat()

    return f"date_math: could not parse {query_text!r}"
