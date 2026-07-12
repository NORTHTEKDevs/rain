"""Ingest FAA Airworthiness Directives into a RAIN-Net semantic KB.

Two modes:
    --seed (default)        ingest the bundled seed file
                            (data/aviation/faa_ads_seed.jsonl)
                            for reproducible demo runs
    --live                  attempt to pull from FAA's public AD search
                            (best-effort; falls back to seed if offline)

Each AD becomes one semantic fact: the text combines AD number, aircraft,
subject, and summary so that text-similarity retrieval works on the
queries it would actually receive.

Run:
    python scripts/ingest_faa_ads.py
    python scripts/ingest_faa_ads.py --out data/aviation/kb_dump.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rain.core.rain_net import RainNet, RainNetConfig


SEED_PATH = Path("data/aviation/faa_ads_seed.jsonl")


def ad_to_fact_text(ad: dict) -> str:
    """Render one AD dict as a single dense fact string for KB ingestion."""
    parts = [
        f"AD {ad['ad_number']}",
        f"({ad.get('manufacturer', '?')} {ad.get('aircraft', '?')}):",
        ad.get("subject", "(no subject)"),
        "--",
        ad.get("summary", "").strip(),
        f"Effective {ad.get('effective_date', '?')}.",
        f"Severity: {ad.get('severity', 'unknown')}.",
    ]
    return " ".join(p for p in parts if p)


def ingest_seed(net: RainNet, path: Path) -> int:
    if not path.exists():
        print(f"(seed file missing: {path})")
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ad = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = ad_to_fact_text(ad)
            net.ingest_fact(text=text, fact_id=f"ad::{ad['ad_number']}", source="faa.gov")
            n += 1
    return n


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", action="store_true", default=True, help="use seed file (default)")
    p.add_argument("--live", action="store_true", help="(future) pull from FAA live")
    p.add_argument("--out", default=None, help="dump KB to this JSONL path")
    p.add_argument(
        "--skills",
        default="skills",
        help="path to skills dir (default skills/, empty to disable)",
    )
    args = p.parse_args(argv)

    print("=== RAIN-Net FAA AD ingest ===")
    net = RainNet(
        config=RainNetConfig(
            dim=10_000,
            n_candidates=2,
            skills_dir=args.skills or None,
            semantic_top_k=5,
        )
    )

    if args.live:
        print("(live mode not yet implemented; using seed for v0.1)")
    n = ingest_seed(net, SEED_PATH)
    print(f"Ingested {n} FAA ADs into semantic memory.")
    print(f"KB size: {len(net.memory.semantic)} facts.")
    print()

    # Quick demo of the aviation_compliance skill on a sample query.
    queries = [
        "Is AD 2024-01-05 applicable to a Cessna 172?",
        "What ADs apply to my PA-28?",
        "Tell me about Beechcraft Bonanza inspections",
        "Looking for ADs on bush operations Cessna 185",
        "Inspections for DHC-2 Beaver on floats",
    ]
    print("Sample queries:")
    for q in queries:
        report = net.answer(q)
        # Pick top cited fact
        if report.cited_facts:
            top = report.cited_facts[0]
            print(f"  Q: {q}")
            print(f"    -> [{top.fact_id}] {top.text[:120]}{'...' if len(top.text) > 120 else ''}")
            skill_used = any("skill::" in p[0] for p in report.provenance)
            print(f"    -> skill_used={skill_used}, confidence={report.confidence:.2f}")
        else:
            print(f"  Q: {q} -> no match")
        print()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for fid, fact in net.memory.semantic._facts.items():
                f.write(
                    json.dumps(
                        {"id": fid, "text": fact.text, "source": fact.source}
                    )
                    + "\n"
                )
        print(f"Dumped {len(net.memory.semantic)} facts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
