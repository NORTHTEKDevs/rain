# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Render a KB seed JSONL into a char-level Q/A training corpus.

Closes the gap between 'HYMN trained on Shakespeare' (good at iambic
pentameter, bad at chat) and 'HYMN that can actually answer factual
questions in chat'. Per the design plan's broke-mode track 6.

For each fact (subject, relation, object) we emit several phrasings so
the model sees Q/A structure as a strong pattern. Snake_case is
de-underscored so the surface form is natural language.

Output: a single flat text file the existing pretrain drivers consume.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_QUESTION_TEMPLATES = [
    "Q: What is the {r} of {s}?\nA: {o}.\n",
    "Q: {S} {r}?\nA: {o}.\n",
    "Q: Tell me about {s}.\nA: {S} {r} {o}.\n",
    "Q: Where is {s}'s {r}?\nA: {o}.\n",  # natural for lives_in / located_in
    "Q: What does {s} have for {r}?\nA: {o}.\n",  # natural for has_property / has_part
]


def _deunderscore(s: str) -> str:
    return s.replace("_", " ").strip()


def _capitalize_first(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _render_fact(fact: dict, n_phrasings: int, rng: random.Random) -> list[str]:
    s = _deunderscore(fact.get("subject") or fact.get("s") or "")
    r = _deunderscore(fact.get("relation") or fact.get("r") or "")
    o = _deunderscore(fact.get("object") or fact.get("o") or "")
    if not (s and r and o):
        return []
    S = _capitalize_first(s)
    templates = rng.sample(_QUESTION_TEMPLATES, k=min(n_phrasings, len(_QUESTION_TEMPLATES)))
    rendered = []
    for tpl in templates:
        rendered.append(tpl.format(s=s, S=S, r=r, o=o))
    return rendered


def main() -> int:
    p = argparse.ArgumentParser(description="Render KB JSONL -> Q/A char corpus")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-phrasings", type=int, default=3, help="number of phrasings per fact (1..5)")
    p.add_argument(
        "--shuffle", action="store_true", help="shuffle the output so similar facts don't cluster"
    )
    p.add_argument("--rng-seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.rng_seed)
    in_path = Path(args.in_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rendered: list[str] = []
    n_facts = 0
    with in_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                fact = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_facts += 1
            rendered.extend(_render_fact(fact, args.n_phrasings, rng))

    if args.shuffle:
        rng.shuffle(rendered)

    # Join with blank lines so paragraph-level chunkers (compose_corpus,
    # shuffle pipelines) can split on `\n\n` cleanly.
    out_path.write_text("\n\n".join(s.rstrip() for s in rendered) + "\n", encoding="utf-8")
    chars = sum(len(r) for r in rendered)
    print(
        f"wrote {out_path}: {n_facts} facts -> {len(rendered)} Q/A pairs "
        f"({chars:,} chars, {len(set(out_path.read_text(encoding='utf-8'))):,} unique chars)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
