"""statistics: descriptive stats over a list of numbers in the query.

Patterns:
    "mean of 1, 2, 3, 4, 5"
    "median of [10, 20, 30]"
    "stdev of 5 10 15 20"
    "summary stats for 1.5, 2.5, 3.5"
    "what is the variance of 1 2 3"
"""

from __future__ import annotations

import re
import statistics as _stats


_NUM_RX = re.compile(r"-?\d+\.?\d*")


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUM_RX.findall(text)]


def invoke(query_text: str, kb) -> str:
    if not query_text:
        return "statistics: empty query"
    tl = query_text.lower()
    nums = _extract_numbers(query_text)
    if not nums:
        return "statistics: no numbers found in query"

    requested = []
    if "mean" in tl or "average" in tl:
        requested.append(("mean", _stats.mean(nums)))
    if "median" in tl:
        requested.append(("median", _stats.median(nums)))
    if "stdev" in tl or "standard deviation" in tl or "std dev" in tl:
        if len(nums) > 1:
            requested.append(("stdev", _stats.stdev(nums)))
        else:
            requested.append(("stdev", 0.0))
    if "variance" in tl:
        if len(nums) > 1:
            requested.append(("variance", _stats.variance(nums)))
        else:
            requested.append(("variance", 0.0))
    if "min" in tl or "minimum" in tl:
        requested.append(("min", min(nums)))
    if "max" in tl or "maximum" in tl:
        requested.append(("max", max(nums)))
    if "sum" in tl or "total" in tl:
        requested.append(("sum", sum(nums)))
    if "mode" in tl:
        try:
            requested.append(("mode", _stats.mode(nums)))
        except _stats.StatisticsError:
            requested.append(("mode", "no unique mode"))
    if "range" in tl:
        requested.append(("range", max(nums) - min(nums)))

    # "summary" / "stats" → all common ones
    if "summary" in tl or ("stats" in tl and not requested):
        requested = [
            ("n", len(nums)),
            ("mean", _stats.mean(nums)),
            ("median", _stats.median(nums)),
            ("stdev", _stats.stdev(nums) if len(nums) > 1 else 0.0),
            ("min", min(nums)),
            ("max", max(nums)),
        ]

    if not requested:
        return "statistics: which stat? (mean, median, stdev, variance, min, max, sum, summary)"

    parts = []
    for name, val in requested:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if isinstance(val, float) and val.is_integer():
                parts.append(f"{name}={int(val)}")
            else:
                parts.append(f"{name}={val:.4g}" if isinstance(val, float) else f"{name}={val}")
        else:
            parts.append(f"{name}={val}")
    return ", ".join(parts)
