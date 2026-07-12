"""Literature numbers as data -- NOT reproduced on this machine.

Every entry here is a citation, not a measurement. results_table.py renders
each one with evidence_tier "cited-not-reproduced" plus its `source` string so
the honest table never confuses a paper's number with a fresh run.

Fields per entry:
  bench     -- matches the `bench` values used in raincg/results/*.json
  system    -- short label for the row (does not need to match our own
               `system` enum since these are external systems)
  accuracy  -- float 0..1, as reported in the source
  source    -- citation string (paper, table/section)
  note      -- optional free-form context
"""
from __future__ import annotations

CITATIONS: list[dict] = [
    {
        "bench": "scan_addprim_jump",
        "system": "seq2seq_lstm (Lake & Baroni)",
        "accuracy": 0.012,
        "source": "Lake & Baroni 2018, 'Generalization without systematicity', Table 3 (add jump)",
        "note": "~1.2% exact-match on the addprim_jump held-out split",
    },
    {
        "bench": "scan_addprim_jump",
        "system": "NeSS",
        "accuracy": 1.00,
        "source": "Chen et al., 'Compositional Generalization via Neural-Symbolic Stack Machines' (NeSS)",
        "note": "reported ~100% on addprim_jump",
    },
    {
        "bench": "scan_addprim_jump",
        "system": "LANE",
        "accuracy": 1.00,
        "source": "Liu et al., 'Learning Algebraic Recombination for Compositional Generalization' (LANE)",
        "note": "reported ~100% on addprim_jump",
    },
    {
        "bench": "pcfg_set",
        "system": "transformer (base split)",
        "accuracy": 0.85,
        "source": "Hupkes et al. 2020, 'Compositionality Decomposed', PCFG SET base/systematicity results",
        "note": "~0.85 accuracy on the base (non-compositional) split",
    },
    {
        "bench": "pcfg_set",
        "system": "transformer (productivity split)",
        "accuracy": 0.50,
        "source": "Hupkes et al. 2020, 'Compositionality Decomposed', PCFG SET productivity results",
        "note": "~0.50 accuracy on the productivity (length-generalization) split",
    },
    {
        "bench": "scan_addprim_jump",
        "system": "GPT-3 code-davinci-002 + least-to-most prompting",
        "accuracy": 0.997,
        "source": "Zhou et al. 2022/2023, 'Least-to-Most Prompting Enables Complex "
                  "Reasoning in Large Language Models', arXiv:2205.10625, Table 8 "
                  "(SCAN length split)",
        "note": "99.7% reported on the SCAN length split with 14 in-context "
                "demonstrations, no gradient training; the paper states the solving "
                "rate 'remains the same' across the other SCAN splits it tested "
                "(including the add-primitive splits), but does not itemize an "
                "addprim_jump-specific number in the excerpts we could verify -- cited "
                "here as the length-split figure, the one explicitly tabulated.",
    },
    {
        "bench": "cogs_gen",
        "system": "PaLM + least-to-most/decomposed prompting",
        "accuracy": 0.992,
        "source": "Drozdov et al. 2023 (ICLR), 'Compositional Semantic Parsing with "
                  "Large Language Models', arXiv:2209.15003",
        "note": "99.2% reported on COGS generalization (gen) accuracy via "
                "prompting-based syntactic decomposition + sequential semantic-parse "
                "generation, no gradient training on COGS.",
    },
    {
        "bench": "cogs_gen",
        "system": "LeAR (Liu et al. 2021)",
        "accuracy": 0.977,
        "source": "Liu et al. 2021, 'Learning Algebraic Recombination for "
                  "Compositional Generalization' (LeAR), Findings of ACL-IJCNLP 2021, "
                  "arXiv:2107.06516, Table 4",
        "note": "97.7% COGS generalization accuracy (COGS overall improved 35.0% -> "
                "97.7% over the prior seq2seq baseline), gradient-trained algebraic "
                "recombination parser.",
    },
]

EVIDENCE_TIER = "cited-not-reproduced"


def citations_for_bench(bench: str) -> list[dict]:
    return [c for c in CITATIONS if c["bench"] == bench]


def all_benches() -> list[str]:
    seen: list[str] = []
    for c in CITATIONS:
        if c["bench"] not in seen:
            seen.append(c["bench"])
    return seen
