# RAINCG Results

## Methodology & caveats

- **Params column** counts gradient-trained floats only. Symbolic solvers
  (e.g. `template_learner_symbolic`, `template_learner_reinforce_role`)
  legitimately show `params: 0` -- that means zero gradient-trained floats,
  NOT zero learned structure. Their structural knowledge lives in symbolic
  lookup tables (templates, verb maps, role tables, etc.), disclosed in each
  result's `notes` and `config` (see `config.symbolic_table_entries` where
  present).
- **LLM rows** in this document are small, LOCAL models doing few-shot
  DIRECT-MAPPING prompting (sentence -> answer, no decomposition). Published
  work using least-to-most / decomposition prompting on FRONTIER LLMs
  (GPT-3 code-davinci-002, PaLM) solves these same splits at 97-99%+ -- see
  the `cited-not-reproduced` rows below. The differentiator this repo
  demonstrates is compute cost, determinism, and auditability of the exact
  symbolic/hybrid solvers, NOT raw task capability that a frontier LLM lacks.
- **`grammar_induction_reinforce` 41.7%** (measured-fresh in this document)
  SUPERSEDES the 80.7% figure recorded in `design/RESONATOR-FINDINGS.md`
  (finding F25). The discrepancy is unreconciled -- likely seed/config
  variance between runs -- and is flagged here rather than silently
  overwritten; treat the 41.7% figure in this document as authoritative
  until reconciled.
- **Hybrid rows** (`hybrid_tagger_supervised`, `hybrid_outputonly_reinforce`)
  are evaluated on the full 7706-example `scan_addprim_jump` test split, not
  a subsample.

## cogs_gen

| System | Acc (95% CI) | n | Params | Train s | Eval s | Evidence |
|---|---|---|---|---|---|---|
| template_learner_reinforce_role | 99.75% (99.68%-99.81%) | 21000 | 122 | 1.6 | 1.2 | measured-fresh |
| template_learner_symbolic | 99.75% (99.68%-99.81%) | 21000 | 0 | 3.5 | 0.3 | measured-fresh |
| PaLM + least-to-most/decomposed prompting | 99.20% | - | - | - | - | cited-not-reproduced (Drozdov et al. 2023 (ICLR), 'Compositional Semantic Parsing with Large Language Models', arXiv:2209.15003) |
| LeAR (Liu et al. 2021) | 97.70% | - | - | - | - | cited-not-reproduced (Liu et al. 2021, 'Learning Algebraic Recombination for Compositional Generalization' (LeAR), Findings of ACL-IJCNLP 2021, arXiv:2107.06516, Table 4) |
| llm_openai/gpt-5.5 | 46.00% (36.56%-55.74%) | 100 | 0 | 0.0 | 1722.0 | measured-fresh |
| llm_anthropic/claude-sonnet-5 | 36.00% (27.27%-45.76%) | 100 | 0 | 0.0 | 709.9 | measured-fresh |
| llm_llama3.2:3b | 0.00% (0.00%-3.70%) | 100 | 0 | 0.0 | 289.1 | measured-fresh |

- llm_anthropic/claude-sonnet-5 (gen): full-denominator accuracy = 36.00% (n=100)
- llm_anthropic/claude-sonnet-5 (gen): subsampled, paired (test_limit=100)
- llm_anthropic/claude-sonnet-5 (gen) notes: frontier API model via OpenRouter, direct few-shot mapping (same prompt as local rows), no decomposition/least-to-most prompting - see cited rows for that technique
- llm_llama3.2:3b (gen): full-denominator accuracy = 0.00% (n=100)
- llm_llama3.2:3b (gen): subsampled, paired (test_limit=100)
- llm_llama3.2:3b (gen) notes: local pretrained LLM (Ollama), few-shot prompting, real COGS gen split
- llm_openai/gpt-5.5 (gen): full-denominator accuracy = 46.00% (n=100)
- llm_openai/gpt-5.5 (gen): subsampled, paired (test_limit=100)
- llm_openai/gpt-5.5 (gen) notes: frontier API model via OpenRouter, direct few-shot mapping (same prompt as local rows), no decomposition/least-to-most prompting - see cited rows for that technique
- template_learner_reinforce_role (gen): full-denominator accuracy = 99.75% (n=21000)
- template_learner_reinforce_role (gen) notes: output-only REINFORCE per-verb-type (agent/theme) table trained against the exact recursive-parser executor (emit_intransitive_output), reward = exact match on TRAIN predictions only, no gold role-lab...
- template_learner_symbolic (gen): full-denominator accuracy = 99.75% (n=21000)
- template_learner_symbolic (gen) notes: symbolic template metalearner, fit on COGS train split ONLY, zero gradient descent (params=0 = zero gradient-trained floats; structural knowledge lives in symbolic lookup tables, see config.symbolic_t...

## pcfg_set

| System | Acc (95% CI) | n | Params | Train s | Eval s | Evidence |
|---|---|---|---|---|---|---|
| pure_vsa | 100.00% (99.62%-100.00%) | 1000 | 0 | 0.4 | 17.6 | measured-fresh |
| transformer (base split) | 85.00% | - | - | - | - | cited-not-reproduced (Hupkes et al. 2020, 'Compositionality Decomposed', PCFG SET base/systematicity results) |
| transformer (productivity split) | 50.00% | - | - | - | - | cited-not-reproduced (Hupkes et al. 2020, 'Compositionality Decomposed', PCFG SET productivity results) |
| transformer_d512x2 | 0.50% (0.21%-1.17%) | 1000 | 15,683,095 | 6459.4 | 57.7 | measured-fresh |

- pure_vsa (test_tgt_le_40): subsampled, paired (test_limit=1000)
- transformer_d512x2 (test_tgt_le_40): subsampled, paired (test_limit=1000)

## scan_addprim_jump

| System | Acc (95% CI) | n | Params | Train s | Eval s | Evidence |
|---|---|---|---|---|---|---|
| hybrid_outputonly_reinforce | 100.00% (99.95%-100.00%) | 7706 | 65 | 150.0 | 0.0 | measured-fresh |
| hybrid_tagger_supervised | 100.00% (99.95%-100.00%) | 7706 | 9,541 | 37.3 | 0.0 | measured-fresh |
| NeSS | 100.00% | - | - | - | - | cited-not-reproduced (Chen et al., 'Compositional Generalization via Neural-Symbolic Stack Machines' (NeSS)) |
| LANE | 100.00% | - | - | - | - | cited-not-reproduced (Liu et al., 'Learning Algebraic Recombination for Compositional Generalization' (LANE)) |
| GPT-3 code-davinci-002 + least-to-most prompting | 99.70% | - | - | - | - | cited-not-reproduced (Zhou et al. 2022/2023, 'Least-to-Most Prompting Enables Complex Reasoning in Large Language Models', arXiv:2205.10625, Table 8 (SCAN length split)) |
| llm_openai/gpt-5.5 | 86.00% (77.86%-91.47%) | 100 | 0 | 0.0 | 486.3 | measured-fresh |
| llm_anthropic/claude-sonnet-5 | 72.00% (62.51%-79.86%) | 100 | 0 | 0.0 | 413.5 | measured-fresh |
| grammar_induction_reinforce | 41.70% (39.56%-43.87%) | 2000 | 148 | 251.9 | 0.0 | measured-fresh |
| llm_qwen2.5:14b | 12.00% (5.62%-23.81%) | 50 | 0 | 0.0 | 189.5 | measured-fresh |
| seq2seq_lstm (Lake & Baroni) | 1.20% | - | - | - | - | cited-not-reproduced (Lake & Baroni 2018, 'Generalization without systematicity', Table 3 (add jump)) |
| llm_llama3.2:3b | 0.00% (0.00%-3.70%) | 100 | 0 | 0.0 | 293.5 | measured-fresh |
| transformer_d128x3 | 0.00% (0.00%-3.70%) | 100 | 609,176 | 160.8 | 0.0 | measured-fresh |

- grammar_induction_reinforce (test_heldout_jump) notes: grammar-rule (lexicon/templates/counts/order) induced via REINFORCE from output alone; parse skeleton given, executor learned
- hybrid_outputonly_reinforce (test_heldout_jump) notes: content-independent per-token category table learned via REINFORCE (no structural supervision) + exact symbolic executor; evaluated on the full 7706-example test split (not a subsample); train_s/eval_...
- hybrid_tagger_supervised (test_heldout_jump) notes: hybrid: supervised per-token tagger (content-independent) + exact symbolic executor; evaluated on the full 7706-example test split (not a subsample); train_s/eval_s may be inflated by a concurrent PCF...
- llm_anthropic/claude-sonnet-5 (test_heldout_jump): subsampled, paired (test_limit=100)
- llm_anthropic/claude-sonnet-5 (test_heldout_jump) notes: frontier API model via OpenRouter, direct few-shot mapping (same prompt as local rows), no decomposition/least-to-most prompting - see cited rows for that technique
- llm_llama3.2:3b (test_heldout_jump): subsampled, paired (test_limit=100)
- llm_llama3.2:3b (test_heldout_jump) notes: local pretrained LLM (Ollama), few-shot prompting, real held-out addprim_jump
- llm_openai/gpt-5.5 (test_heldout_jump): subsampled, paired (test_limit=100)
- llm_openai/gpt-5.5 (test_heldout_jump) notes: frontier API model via OpenRouter, direct few-shot mapping (same prompt as local rows), no decomposition/least-to-most prompting - see cited rows for that technique
- llm_qwen2.5:14b (test_heldout_jump): subsampled, paired (test_limit=50)
- llm_qwen2.5:14b (test_heldout_jump) notes: local pretrained LLM (Ollama), few-shot prompting, real held-out addprim_jump
- transformer_d128x3 (test_heldout_jump) notes: from-scratch decoder-only transformer on REAL addprim_jump held-out split
