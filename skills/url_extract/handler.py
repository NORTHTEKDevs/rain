# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""url_extract: pull just URLs + decompose them.

Patterns:
    "extract urls from check https://example.com/a/b?c=1"
    "what is the domain of https://foo.bar.example.com/path"
    "list all paths in https://a.com/x https://b.com/y/z"

Returns URLs with optional decomposition into scheme/host/path/query.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


_URL_RX = re.compile(r"https?://[^\s)>]+")


def invoke(query_text: str, kb) -> str:
    if not query_text:
        return "url_extract: empty query"
    tl = query_text.lower()
    matches = _URL_RX.findall(query_text)
    if not matches:
        return "url_extract: no URLs found"

    # Decomposition queries
    if "domain" in tl or "host" in tl:
        parts = [urlparse(u).hostname or "?" for u in matches]
        return "domains: " + ", ".join(parts)
    if "path" in tl:
        parts = [urlparse(u).path or "/" for u in matches]
        return "paths: " + ", ".join(parts)
    if "scheme" in tl or "protocol" in tl:
        parts = [urlparse(u).scheme for u in matches]
        return "schemes: " + ", ".join(parts)
    if "query" in tl or "params" in tl:
        parts = [urlparse(u).query or "(none)" for u in matches]
        return "queries: " + ", ".join(parts)

    # Default: full list
    if len(matches) == 1:
        u = matches[0]
        p = urlparse(u)
        return (
            f"url={u}; scheme={p.scheme}; host={p.hostname}; "
            f"path={p.path or '/'}; query={p.query or '(none)'}"
        )
    return "urls: " + ", ".join(matches)
