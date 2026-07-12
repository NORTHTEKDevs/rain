"""Multi-vertical swarm demo — the v0.2 horizontal-scale pitch made tangible.

Spins up 3 specialist RainNet instances inside one RainNetSwarm:
    aviation_compliance    -> FAA AD seed + airworthiness domain
    medical_basics         -> common medical facts (vaccines, conditions, drugs)
    legal_basics           -> contract law + common-law primer facts

Then runs a short interactive REPL. For each query:
    - HV-cosine router picks top-2 specialists in parallel
    - Each specialist runs its own RainNet (skills + experts + KB)
    - SwarmAnswer aggregates via verifier vote + HV bundle
    - Output shows which specialists were routed + their cited facts

This is the architectural piece that makes the Series A
\"horizontal scale via composition\" story concrete. Adding a new
vertical = new RainNet + one domain text. No router retraining.

Run:
    python scripts/swarm_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rain.core.rain_net import RainNet, RainNetConfig
from rain.core.swarm import RainNetSwarm


AVIATION_FACTS = [
    "AD 2024-01-05 applies to Cessna 172 R/S models for fuel-selector inspection within 100 hours.",
    "AD 2024-03-12 mandates wing-strut corrosion inspection on Cessna 172 (1996-2010) within 50 hours.",
    "AD 2023-11-22 requires aileron control-rod inspection on Piper PA-28 every 1000 hours.",
    "AD 2025-02-08 mandates V-tail bolt inspection on Beechcraft Bonanza 35-V35B every 500 hours.",
    "AD 2024-06-08 requires wing-spar inspection on DHC-2 Beaver float/ski aircraft within 200 hours.",
    "AD 2025-04-10 mandates lift-strut inspection on Piper PA-18 Super Cub on tundra tires every 100 hours.",
    "FAA Part 91 covers general aviation flight rules.",
    "FAA Part 135 governs on-demand commercial operations.",
    "FAA Part 23 covers airworthiness standards for normal-category airplanes.",
    "PT6A engines require preflight FOD inspection on Cessna 208 Caravan in bush operations.",
]

MEDICAL_FACTS = [
    "MMR vaccine prevents measles, mumps, and rubella; given at 12-15 months and 4-6 years.",
    "Aspirin reduces fever, pain, and inflammation by inhibiting COX-1 and COX-2 enzymes.",
    "Diabetes mellitus type 1 is autoimmune destruction of pancreatic beta cells; insulin-dependent.",
    "Type 2 diabetes is insulin resistance plus relative insulin deficiency; lifestyle and metformin are first-line.",
    "Hypertension is sustained blood pressure above 130/80 mmHg per AHA 2017 guidelines.",
    "Penicillin allergy is reported in ~10% of patients but verified in less than 1% by skin testing.",
    "Atherosclerosis is plaque buildup in arterial walls; LDL cholesterol is the primary modifiable risk.",
    "The Krebs cycle (citric acid cycle) generates NADH and FADH2 for the electron transport chain.",
    "Cardiac arrest differs from heart attack: arrest is electrical failure; attack is blood-flow blockage.",
    "Statins (atorvastatin, simvastatin) inhibit HMG-CoA reductase to lower LDL cholesterol.",
]

LEGAL_FACTS = [
    "A contract requires offer, acceptance, consideration, capacity, and lawful purpose.",
    "Promissory estoppel allows enforcement of a promise without consideration where reliance caused detriment.",
    "Statute of frauds requires writing for contracts over $500, real-estate transfers, and >1-year duration.",
    "Force majeure clauses excuse contract performance when extraordinary events prevent it.",
    "The Uniform Commercial Code (UCC) governs sales of goods in the US.",
    "Tort negligence has four elements: duty, breach, causation, damages.",
    "Strict liability applies to ultrahazardous activities and defective products regardless of fault.",
    "Mens rea (guilty mind) plus actus reus (guilty act) are required for most criminal liability.",
    "Miranda warnings must be given before custodial interrogation per Miranda v. Arizona (1966).",
    "The 4th Amendment requires a warrant for searches except via exigent circumstances or consent.",
]


def build_swarm(dim: int = 10_000) -> RainNetSwarm:
    """Construct the 3-specialist swarm with seeded KBs."""
    s = RainNetSwarm(dim=dim)

    aviation = s.add_member(
        "aviation_compliance",
        domain_text=(
            "aircraft airworthiness directive AD FAA Cessna Piper Beechcraft "
            "engine flight bush pilot inspection mandatory part 91 part 135 "
            "tail number maintenance"
        ),
        config=RainNetConfig(
            dim=dim,
            n_candidates=2,
            semantic_top_k=3,
            skills_dir="skills",
            verifier_checkpoint_path="data/checkpoints/verifier_head_v0.npz",
        ),
    )
    for f in AVIATION_FACTS:
        aviation.net.ingest_fact(f, source="faa.gov")

    medical = s.add_member(
        "medical_basics",
        domain_text=(
            "medical medicine doctor patient diagnosis treatment disease "
            "vaccine drug pharmaceutical condition symptom prescription "
            "cardiovascular diabetes hypertension blood"
        ),
        config=RainNetConfig(
            dim=dim,
            n_candidates=2,
            semantic_top_k=3,
            skills_dir="skills",
        ),
    )
    for f in MEDICAL_FACTS:
        medical.net.ingest_fact(f, source="medline+textbook")

    legal = s.add_member(
        "legal_basics",
        domain_text=(
            "legal law contract clause tort negligence liability statute "
            "constitutional court attorney defendant plaintiff criminal "
            "civil amendment warrant due process"
        ),
        config=RainNetConfig(
            dim=dim,
            n_candidates=2,
            semantic_top_k=3,
            skills_dir="skills",
        ),
    )
    for f in LEGAL_FACTS:
        legal.net.ingest_fact(f, source="caselaw+restatement")

    return s


SAMPLE_QUERIES = [
    "Which AD applies to my Cessna 172 right now?",
    "How does aspirin work?",
    "What are the requirements for a valid contract?",
    "Do I need a Miranda warning for a traffic stop?",
    "What is the difference between type 1 and type 2 diabetes?",
    "What FAA part covers bush pilot operations?",
    "Tell me about force majeure clauses.",
    "Are statin drugs safe?",
]


def banner() -> str:
    return """
============================================================
RAIN-Net Multi-Vertical Swarm Demo (v0.2)
3 specialist RainNets: aviation + medical + legal
HV-cosine router, parallel dispatch, verifier-voted aggregation
============================================================
"""


def main() -> int:
    print(banner())
    swarm = build_swarm()
    stats = swarm.stats()
    print(f"Swarm has {stats['n_members']} specialists:")
    for m in stats["members"]:
        print(f"  - {m['name']:<22} kb_size={m['kb_size']}")
    print()

    # Phase 1: route + answer the sample queries.
    print("=" * 60)
    print("Phase 1: 8 cross-vertical queries via the swarm")
    print("=" * 60)
    for q in SAMPLE_QUERIES:
        print(f"\nQ: {q}")
        # Show routing.
        q_hv = swarm._shared_encoder.encode("text", q)
        routing = swarm.route(q_hv, top_k=2)
        print(f"  routing: {[f'{n}={w:.2f}' for n, w in routing.members]}")
        # Run the answer.
        result = swarm.answer(q, top_k=2)
        print(f"  answer (consensus from {len(result.contributing_members)} members):")
        print(f"    {result.text[:180]}{'...' if len(result.text) > 180 else ''}")
        print(f"    confidence: {result.confidence:.2f}")
        # Show top cited fact per contributing member.
        for name, report in result.contributing_members:
            if report.cited_facts:
                top = report.cited_facts[0]
                print(f"    [{name}] cited: {top.text[:100]}")

    # Phase 2: ops stats.
    print()
    print("=" * 60)
    print("Phase 2: swarm operational stats")
    print("=" * 60)
    print(json.dumps(swarm.stats(), indent=2))

    print()
    print("=" * 60)
    print("Done. Adding a new specialist (e.g. real_estate) = one")
    print("RainNet + one domain text. No router retraining.")
    print("This is horizontal scale via composition, not training.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
