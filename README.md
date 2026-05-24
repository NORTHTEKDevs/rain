# RAIN — Resonant Active Inference Network

> A non-Transformer generative AI architecture with structured knowledge
> integrated as a per-block forward-pass primitive. CPU-trainable on a
> workstation. Auditable, continually-learning, multi-modal substrate.
> Architecturally distinct from any LLM.

**CONFIDENTIAL** — (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

---

## The architectural moat (validated as of v5)

| | LLMs (GPT-4, Claude, Llama, Mamba, RWKV) | **RAIN** |
|---|---|---|
| World knowledge lives in... | 10⁹–10¹² model parameters | **Explicit (subject, relation, object) KB + per-block KB-Attention** |
| Update one fact | Full retrain ($M, weeks) | **`tell()` in 8ms — generation reflects it on next call** |
| Audit which facts produced an answer | Impossible | **`/kb_attn` returns top-K facts per layer per token** |
| Generation architecture | Transformer / SSM / linear-attention | **Selective gated recurrence + SwiGLU + per-block KB-Attention. No attention. No convolution.** |
| Multi-modal | Separate encoder networks per modality | **One hypervector substrate; image/audio/timeseries encoders all produce same (D,) bipolar shape** |
| Training cost (validated) | $millions+ | **5K-30K steps on AMD CPU, hours** |
| Continual learning | Catastrophic forgetting | **By construction — LSM RLS + FEP rank-1 + Tsetlin Type-I + KB writes** |

The v5 KB-grounding validation experiment proves an in-distribution KB
built FRESH at inference (facts the model never saw during training)
measurably outperforms a random KB (NLL 3.81 vs 3.90, -2.3%). That's
the moat operating end-to-end. See `docs/RESULTS.md` for the full
table including v2/v3/v4 failures and how v5 succeeded.

## What's shipped (verified, on main, tagged)

### Architectures

- **HYMN** — original v0 MLP-with-carry char-level LM. L1 NLL = 1.47 on Tiny Shakespeare.
- **HYMN-Plus v1** (`rain.core.hymn_plus`) — selective gated recurrence + SwiGLU + pre-norm. L1 NLL 1.25 (-15%), OOD WT-2 NLL 2.77 (66% of uniform).
- **HYMN-Plus v1 on WT-103** — full 543MB corpus, 30K steps CPU, train NLL **1.16**, OOD WT-2 NLL **1.638** (86.7% compression of uniform baseline).
- **HYMN-Plus v2** (`rain.core.hymn_plus_v2`) — adds per-block KB-Attention + BPE tokenization. **The architectural moat lives here.**
- **HYMN-Plus v5** — v2 trained on Q/A hybrid corpus with real-fact KB pool init + KB-shuffle. First checkpoint where the swappable-KB story works end-to-end. Tag: `arch/hymn-plus-v5-moat-validated`.

### Tools and infrastructure

- `scripts/pretrain_hymn_plus_v2.py` — BPE training with KB-init, KB-shuffle, W_o init gain, val split, early stop
- `scripts/sample_hymn_plus_v2.py` — generation with optional `--kb-source` for inference-time KB swap
- `scripts/eval_hymn_plus.py` — held-out NLL on any corpus
- `scripts/validate_v2_kb_grounding.py` — the architectural-claim test (trained vs random vs in-dist vs OOD KB conditions)
- `scripts/benchmark_checkpoints.py` — apples-to-apples A/B
- `scripts/sample_quality.py` — verbatim-overlap + diversity (memorization detector)
- `scripts/calibrate_sampler.py` — temperature × top-k sweep
- `scripts/merge_kb_seeds.py` — multi-source KB consolidation + noun-phrase augmentation (11K facts in `data/kb_seed/merged_v1.jsonl`)
- `scripts/extract_alpaca.py` / `scripts/extract_code_corpus.py` / `scripts/extract_wikitext103.py` — corpus prep
- `scripts/demo_v2_tell_changes_generation.py` — BEFORE/AFTER demo for the killer feature
- `scripts/preference_finetune.py` — DPO using local Ollama judge

### Architectural extensions toward LLM-rival capability

- `rain/core/multimodal_kb.py` — image / audio / timeseries → bipolar hypervectors. Same KB-Attn layer handles all modalities.
- `rain/cognition/scratchpad.py` — explicit chain-of-thought via `scratch:*` KB writes that subsequent KB-Attention queries can attend to. Multi-step reasoning without LLM-style "thinking out loud" generation.
- `rain/cognition/sharded_kb_bridge.py` — bridge from RAIN's persistent ShardedKB (10K–1M facts) to the model's per-block KB-Attention buffer (1K–8K).
- `rain/cognition/hymn_plus_v2_sampler.py` — agent-pluggable adapter; `set_kb_from_facts()` is the killer-feature API.

### Agent / serving surface

- `rain.agent.ConsciousAgent` — KB-grounded continual-learning agent with cognitive surfaces (LSM / FEP / Tsetlin) wired
- `scripts/rain_chat.py` — REPL; auto-detects HYMN / HYMN-Plus v1 / HYMN-Plus v2 from sidecar JSON
- `scripts/rain_server.py` — aiohttp HTTP server: `/ask /tell /describe /sample /judge /tally /snapshot /self /kb_attn /`
- `rain/webui/chat.html` — vanilla-JS chat UI with KB / HYMN / cognitive-signal / KB-attention badges

## Get started

```bash
git clone git@github.com:NORTHTEKDevs/rain.git
cd rain
bash .shift/init.sh  # installs deps, downloads Tiny Shakespeare, runs tests
```

### Train HYMN-Plus v2 on a Q/A corpus with KB grounding

```bash
# 1. Build the super corpus (Alpaca + Code + KB-QA + Shakespeare = 27 MB)
python -m scripts.extract_alpaca   --out data/corpora/alpaca_qa_full.txt
python -m scripts.extract_code_corpus --out data/corpora/code_qa.txt
python -m scripts.compose_corpus \
    --part data/corpora/alpaca_qa_full.txt:1 \
    --part data/corpora/code_qa.txt:1 \
    --part data/corpora/kb_qa.txt:3 \
    --part data/corpora/tiny_shakespeare.txt:1 \
    --shuffle-chunks --out data/corpora/super_v1.txt

# 2. Train v2 with KB-grounding (CPU, ~30-60 min for v5-class)
python -m scripts.pretrain_hymn_plus_v2 \
    --corpus data/corpora/super_v1.txt \
    --bpe-vocab 4096 \
    --kb-init-jsonl data/kb_seed/merged_v1.jsonl \
    --kb-fact-pool-size 11000 \
    --steps 8000 --batch-size 16 --seq-len 128 \
    --dim 256 --n-layers 4 --mlp-mult 4 \
    --kb-size 1024 --kb-top-k 8 \
    --kb-shuffle-frac 0.05 --kb-shuffle-every 4 --kb-w-o-init-gain 0.3 \
    --lr 3e-4 --warmup-steps 500 --cosine-decay --weight-decay 0.05 \
    --val-split 0.03 --val-every 500 --early-stop-patience 5 \
    --out data/checkpoints/my_v2.npz

# 3. Validate the architectural moat
python -m scripts.validate_v2_kb_grounding \
    --checkpoint data/checkpoints/my_v2.npz \
    --train-corpus data/corpora/super_v1.txt \
    --eval-corpus data/corpora/super_v1.txt \
    --ood-corpus data/corpora/wikitext2_train.txt
# Success: in-distribution-KB NLL < random-KB NLL by >=2%

# 4. Run the demo
python -m scripts.demo_v2_tell_changes_generation \
    --checkpoint data/checkpoints/my_v2.npz \
    --prompt "Q: Where does the lion live?\nA:" \
    --facts "lion lives_in savanna" "wolf lives_in forest"

# 5. Talk to it via chat UI
python -m scripts.rain_server --checkpoint data/checkpoints/my_v2.npz \
    --kb data/kb_seed/merged_v1.jsonl --enable-continual
# browse to http://localhost:8721/
```

## Path to LLM-rival capability

The architecture is in place. Remaining work is **engineering and
training scale**, not research breakthroughs:

| Step | What | Capital |
|---|---|---|
| Scale checkpoint | dim=512–768, 8–12 layers, 100K steps on super_v1 | $15–50 cloud GPU |
| DPO preference fine-tune | `scripts/preference_finetune.py` with Ollama judge, 1K–10K prompts | $0 local |
| Bigger KB | Scale `data/kb_seed/merged_v1.jsonl` from 11K to 100K+ | $0 (Ollama) |
| Code corpus | `scripts/extract_code_corpus.py` already pulls 20K Q/A | $0 |
| Multi-modal training data | LAION subset, AudioSet samples | $50–500 storage |
| 1B param scale | Final-form model | $10K–100K |

Each step preserves the architectural moat. The model gets bigger, the
data richer, the KB deeper. The architecture stays the differentiator.

## Honest scoping

This is **not GPT-4 today**. It's a 3M–20M parameter non-Transformer
prototype with a measured architectural moat that LLMs structurally
cannot ship. Where pure scale matters (general-knowledge fluency,
open-domain chat), the path to parity goes through more compute, not
new research.

This **is** an architecture that:
- Trains on CPU in hours
- Can be updated in 8ms via `tell()`
- Tells you which facts produced which token
- Generalizes to fresh KB content at inference (validated)
- Composes multi-modal facts into one substrate
- Reasons in steps with auditable trace
- Runs offline on a workstation in ~50MB

That's a real category. No LLM lives there.

## Documents

- `docs/RESULTS.md` — empirical ledger (every run, every NLL, including negatives)
- `docs/RESEARCH_LOG.md` — v2→v5 architectural research log
- `docs/CAPABILITIES.md` — what works today, with reproduction commands
- `docs/quickstart.md` — 10-minute path from clone to talking agent
- `docs/training-runbook.md` — HYMN-Plus recipes and gotchas
- `docs/READY_TO_SCALE.md` — v0→v1.0 bridge plan
- `docs/ROADMAP_v1_5.md` — milestone plan with kill criteria
- `docs/plans/2026-05-22-rain-design.md` — original master design

## Tags

- `arch/hymn-plus-v1` — first working non-Transformer generative LM
- `arch/hymn-plus-v5-moat-validated` — KB-Attention architectural moat proven end-to-end

## License

Proprietary. SPDX NOASSERTION. See `LICENSE`.
