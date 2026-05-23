# RAIN v0 -- Quickstart

> The 10-minute path from a fresh clone to a talking RAIN agent on your
> own hardware, $0 cloud spend. Tested on the Corsair AI Workstation 300
> (Ryzen AI MAX+ 395, Radeon 8060S iGPU, 128 GB DDR5-8000, Windows 11).

## Prerequisites

- Python 3.11 or 3.12 (verified on 3.12.10).
- Git, Ollama (https://ollama.com/), curl.
- ~10 GB free disk (corpora + checkpoints).

## 1. Install (one-time)

```bash
git clone git@github.com:NORTHTEKDevs/rain.git
cd rain
bash .shift/init.sh          # creates .venv, installs rain[dev], torch+directml,
                             # downloads tiny_shakespeare, runs the test suite.
                             # prints `READY` when the environment is hot.
```

If you're on AMD on Windows, init.sh installs `torch-directml` so the iGPU
is usable. On NVIDIA or CPU-only systems, training still works (slower);
just ignore the DirectML log lines.

## 2. Train the HYMN core (~15 min on CPU; ~10 min on the iGPU)

This is the run that produced the L1-PASSING checkpoint
(L1 NLL = 1.5359, target 1.55):

```bash
.venv/Scripts/python -u -m scripts.pretrain_hymn_torch \
  --corpus data/corpora/tiny_shakespeare.txt \
  --steps 30000 --batch-size 64 \
  --carry-steps 16 --grad-clip 1.0 \
  --lr 5e-4 --weight-decay 1e-4 \
  --in-dim 1024 --hidden-dim 512 --out-dim 1024 \
  --seed 42 --device cpu --loss nll \
  --log-every 3000 \
  --out data/checkpoints/hymn_carry16_30k.npz
```

Validate it:

```bash
.venv/Scripts/python -m evals.tier2_llm_parity.tiny_shakespeare \
  --checkpoint data/checkpoints/hymn_carry16_30k.npz \
  --corpus data/corpora/tiny_shakespeare.txt \
  --n-eval-chars 5000 --out evals/results/L1.json
# Expect: "pass": true  with nll_loss ~ 1.5
```

Sample text from it:

```bash
.venv/Scripts/python -m scripts.sample_hymn \
  --checkpoint data/checkpoints/hymn_carry16_30k.npz \
  --corpus data/corpora/tiny_shakespeare.txt \
  --prompt "ROMEO:" --n-tokens 200 \
  --temperature 0.8 --top-k 20 \
  --repetition-penalty 1.1
```

## 3. Distill a knowledge base from a local Ollama LLM (~7 min)

Pull a small instruct model that follows JSON-output prompts reliably:

```bash
ollama pull llama3.2:3b
```

Generate the seed:

```bash
.venv/Scripts/python -u -m scripts.seed_kb_from_ollama \
  --model llama3.2:3b \
  --n-per-topic 25 \
  --out data/kb_seed/llama3b_default.jsonl
```

Output: ~600 facts in ~7 min on this workstation. For the 200-topic
expanded baseline use `--topics-file data/topics_expanded.txt` (gives
~2700 facts in ~25 min).

Optionally filter the seed through the judge (drops ~10-15% of bad
triples; takes ~2 hours wall on 2700 facts):

```bash
.venv/Scripts/python -u -m scripts.filter_seed_with_judge \
  --in  data/kb_seed/llama3b_default.jsonl \
  --out data/kb_seed/llama3b_filtered.jsonl \
  --rejected data/kb_seed/llama3b_rejected.jsonl \
  --judge-model llama3.2:3b \
  --accept-unsure --verbose
```

## 4. Chat with RAIN

```bash
.venv/Scripts/python -m scripts.rain_chat \
  --kb data/kb_seed/llama3b_default.jsonl \
  --checkpoint data/checkpoints/hymn_carry16_30k.npz \
  --corpus data/corpora/tiny_shakespeare.txt \
  --judge-model llama3.2:3b
```

REPL commands:
```
> lion lives_in              # KB lookup with citation + epistemic
> /describe lion              # multi-fact prose composition
> /sample ROMEO:              # HYMN free-text generation
> /judge lion lives_in        # ask + judge + (gated) feedback
> /selfdescribe               # RAIN's structural self-description
> /tally                      # per-relation calibration tally
> /save data/tally.json       # persist tally
> /help    /quit
```

Free-text input falls through: tries `<subject> <relation>` first; if no KB
hit, treats it as a HYMN sampling prompt. So `who is hamlet` just samples
HYMN as if the prompt were "who is hamlet ".

## 5. Run the full test suite

```bash
.venv/Scripts/python -m pytest tests/ -q
# Expect ~190+ passed in <15 s.
```

## What you have at this point

| Capability | How |
|---|---|
| Char-level next-token LM (passes L1 design-plan threshold) | `scripts.sample_hymn` / `scripts.pretrain_hymn_torch` |
| Structured knowledge base | `scripts.seed_kb_from_ollama` + `scripts.filter_seed_with_judge` |
| Grounded KB-cited answers | `scripts.rain_chat` `<subject> <relation>` |
| Calibrated uncertainty + epistemic class | every `agent.ask` return value |
| Theory of Mind primitives | `rain/cognition/theory_of_mind.py` (already shipped) |
| Free RLAIF feedback loop | `scripts.rain_chat /judge` or `scripts.run_phase2_feedback` |
| Compositional generalization | `examples/01_chat_e2e.py` section 4 |

Total dollar cost on the Corsair workstation: **$0** (electricity only).

## What you don't have yet (queued for future shifts)

- HYMN wired into the chat surface for token-by-token KB-grounded
  generation (currently chat is KB-first OR HYMN-fallback; not blended).
- WikiText-2-scale chat (the WT2-trained checkpoint exists at
  `data/checkpoints/hymn_wikitext2_carry16_30k.npz` with self-eval NLL
  1.72 but isn't wired into the chat REPL because its vocab doesn't
  match Tiny Shakespeare's).
- The Phase-6 cognitive integration (LSM + Tsetlin + FEP + SOFAR routing
  in the inference path).
- A Tauri/web UI (the chat REPL is the v0 surface).
- Multi-tenant deployment.

See `docs/plans/2026-05-22-broke-mode-training.md` for the four-track
roadmap that got us here.
