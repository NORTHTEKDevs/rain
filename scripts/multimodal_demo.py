# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Multimodal demo: text + image + numeric in the same HV space.

Demonstrates the architectural commitment of RAIN-Net: all modalities
produce HVs in the same 10K-D bipolar space, so multi-modal queries
are just HV ops (bind for compound queries, cosine for retrieval).

Scenario: a small KB of (caption, synthetic-image, timeseries) facts.
Query types:
    1. Text-only: "what is the apollo image about"
    2. Image-only: encode a query image, find similar
    3. Compound: bind(text_query_hv, image_query_hv) -> find similar
    4. Numeric: timeseries query against numeric facts in same KB

This proves: the same RainNet handles text/image/numeric queries via
the SAME architecture; no per-modality network swap needed. LLMs
require separate encoders + fusion training per modality.

Run:
    python scripts/multimodal_demo.py
"""

from __future__ import annotations

import sys

import numpy as np

from rain.core.encoder_bank import EncoderBank
from rain.core.hv_substrate import DEFAULT_DIM, similarity
from rain.core.rain_net import RainNet, RainNetConfig


def _synth_image(seed: int, shape: tuple[int, int, int] = (32, 32, 3)) -> np.ndarray:
    """Deterministic synthetic 'image' for demo purposes."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _synth_timeseries(seed: int, n: int = 64) -> np.ndarray:
    """Deterministic synthetic numeric series."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def main(argv: list[str]) -> int:
    print("=== RAIN-Net Multimodal Demo ===\n")

    dim = DEFAULT_DIM
    net = RainNet(config=RainNetConfig(dim=dim, n_candidates=2, semantic_top_k=5))

    # Ingest mixed-modality facts. Each fact text describes what the
    # accompanying modality data represents.
    facts = [
        ("Apollo 11 lunar surface photo", _synth_image(11)),
        ("Apollo 12 lunar surface photo", _synth_image(12)),
        ("Hubble deep-field telescope image", _synth_image(20)),
        ("Mars rover panorama", _synth_image(30)),
        ("ISS exterior photograph", _synth_image(40)),
    ]
    print(f"Ingesting {len(facts)} multimodal facts (text + image)...")
    bank = net.encoder_bank
    for text, img in facts:
        # Bind text HV with image HV to create a true multimodal fact HV.
        text_hv = bank.encode("text", text)
        img_hv = bank.encode("image", img)
        # Multimodal fact HV: bind text and image so retrieval needs both
        from rain.core.hv_substrate import bind

        fact_hv = bind(text_hv, img_hv)
        fid = net.ingest_fact(text=text, source="multimodal_demo")
        # Manually override the fact's HV to be the bound multimodal form.
        net.memory.semantic._facts[fid].hv = fact_hv
        net.memory.semantic._bank[net.memory.semantic._ids.index(fid)] = fact_hv

    # Add a few text-only and numeric-only facts for variety.
    for i, ts_label in enumerate(["AAPL daily close 2026", "BTC hourly volume", "patient heartrate"]):
        ts = _synth_timeseries(100 + i)
        ts_hv = bank.encode("ts", ts)
        fid = net.ingest_fact(text=ts_label, source="numeric_demo")
        net.memory.semantic._facts[fid].hv = ts_hv
        net.memory.semantic._bank[net.memory.semantic._ids.index(fid)] = ts_hv

    print(f"KB size: {net.memory.semantic.__len__()} facts\n")

    # --- Query 1: text-only ---
    print("--- Query 1: text-only ---")
    q1 = "tell me about the apollo moon photograph"
    print(f"  Q: {q1}")
    report1 = net.answer(q1)
    for f in report1.cited_facts[:3]:
        print(f"    -> [{f.fact_id}] {f.text}")
    print()

    # --- Query 2: image-only ---
    print("--- Query 2: image-only (query image close to Apollo 11) ---")
    # Use a noisy copy of Apollo 11's image as the query.
    query_img = _synth_image(11)
    # Add some noise so it's not an exact match.
    noisy = np.clip(
        query_img.astype(np.int16) + np.random.default_rng(0).integers(-20, 20, size=query_img.shape),
        0,
        255,
    ).astype(np.uint8)
    q_img_hv = bank.encode("image", noisy)
    # Manual semantic search since RainNet.answer expects text.
    results = net.memory.semantic.search(q_img_hv, top_k=5)
    for sim, fact in results[:3]:
        print(f"    -> sim={sim:+.3f} [{fact.fact_id}] {fact.text}")
    print()

    # --- Query 3: multimodal bind ---
    print("--- Query 3: COMPOUND text+image query (bind both) ---")
    text_part = "lunar surface"
    image_part = _synth_image(11)  # apollo 11 again
    compound = bank.encode_multi([("text", text_part), ("image", image_part)])
    results = net.memory.semantic.search(compound, top_k=5)
    for sim, fact in results[:3]:
        print(f"    -> sim={sim:+.3f} [{fact.fact_id}] {fact.text}")
    print()

    # --- Query 4: numeric query ---
    print("--- Query 4: numeric timeseries query ---")
    ts_query = _synth_timeseries(100)  # exact AAPL match
    q_ts_hv = bank.encode("ts", ts_query)
    results = net.memory.semantic.search(q_ts_hv, top_k=5)
    for sim, fact in results[:3]:
        print(f"    -> sim={sim:+.3f} [{fact.fact_id}] {fact.text}")
    print()

    print("=== Done ===")
    print()
    print("Key observations:")
    print("  - All four query modes use the SAME RainNet + the SAME")
    print("    semantic memory bank. No per-modality network swap.")
    print("  - Multimodal binding via bind(text_hv, image_hv) is one HV op.")
    print("  - LLMs require separate encoder networks per modality plus")
    print("    fusion training; RAIN-Net's HV substrate makes it native.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
