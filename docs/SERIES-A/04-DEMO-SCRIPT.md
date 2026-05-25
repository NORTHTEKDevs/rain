# RAIN-Net Investor Demo Script (15 minutes, live or recorded)

> Run on a laptop, no cloud needed. Demonstrates: continual learning,
> audit trail, multi-modal, active distillation, deterministic skills.
> The architecture's value props are visible on screen, not narrated.
>
> **Use `scripts/rain_chat_v2.py`** as the demo surface. It loads:
>   - 4 bundled procedural skills (math, date, unit, regex)
>   - Trained verifier head (auto-loaded from disk)
>   - Optional HYMN-Plus LM checkpoint
>   - Optional Ollama active-learning loop
>   - Multi-turn session with episodic recall

## Setup (do once, before any demo)

```bash
# 1. Clone + install
git clone github.com/NORTHTEKDevs/rain
cd rain
python -m venv .venv && .venv/Scripts/activate    # Windows
pip install -e .

# 2. (Optional) Start local Ollama for active learning
ollama pull qwen2.5:7b
ollama serve   # leave running

# 3. Verify tests pass
python -m pytest tests/test_rain_net.py -q
# Should print: 48 passed
```

---

## Demo Part 0 — Investor REPL with skills + sessions (NEW, 4 minutes)

**Goal**: show the entire system working as a single tool that a
human can drive in one minute.

```bash
python scripts/rain_chat_v2.py
```

You'll see:

```
============================================================
RAIN-Net chat v2
============================================================
  dim:          10000
  candidates:   2
  skills:       4 loaded
  verifier:     loaded (updates=960)
  active learn: off
  KB size:      0 facts
```

Narrate while typing:

```
[t0] > what is 47 * 38
```

```
ANSWER: 1786
ROUTED EXPERTS:
  - tsetlin: weight=0.51
  - samba_lm: weight=0.49
  - skill::math_solver: weight=0.72
```

> "The math_solver skill is on disk — `skills/math_solver/handler.py`.
> When I asked '47 * 38', the router matched the skill's domain HV +
> trigger pattern, fired the deterministic handler, and got 1786 — not
> a hallucinated answer, a computed one. The skill is in the
> provenance trail."

```
[t1] > 100 km to miles
ANSWER: 100 km = 62.14 miles
```

```
[t2] > 30 days from 2026-05-24
ANSWER: 2026-06-23
```

```
[t3] > extract emails from contact foo@bar.com and baz@qux.io
ANSWER: email: foo@bar.com, baz@qux.io
```

> "Four different deterministic skills, all on disk, all routed by HV
> similarity + trigger patterns. None went through an LLM. None cost
> anything. None can hallucinate. This is the procedural-memory layer
> of RAIN-Net's hierarchical memory."

```
[t4] > learn Apollo 11 landed on the Moon on July 20, 1969
(learned as f0)

[t5] > when did apollo 11 land
ANSWER: Apollo 11 landed on the Moon on July 20, 1969
WARNING: recalled prior turns [4] from episodic memory
```

> "I added a fact. No retraining. Next turn retrieved it AND noted the
> session's episodic memory recalled the learn turn. Multi-turn
> session works at arbitrary length — LLMs lose this past their
> context window; RAIN-Net's episodic memory is HV-addressable."

---

## Demo Part 1 — Architecture proof (3 minutes)

**Goal**: convince them the 9 modules exist and integrate.

```bash
python scripts/rain_net_demo.py
```

You should see:
```
RAIN-Net v0.1 demo. Type 'help' for commands.
Mode: offline
KB facts: 0
> _
```

**Narrate while you type**:

> "RAIN-Net is a composition engine that operates on hypervector
> representations. No facts in memory yet. Let me add some."

```
> learn Apollo 11 landed on the Moon on July 20, 1969.
(learned as f0)
> learn The speed of light in vacuum is approximately 299,792,458 m/s.
(learned as f1)
> learn Python was created by Guido van Rossum in 1991.
(learned as f2)
```

> "Three facts added. No retraining. No GPU. Now I ask a question."

```
> ask When did Apollo 11 land on the Moon?
```

You see the full **AuditReport** print:
- ANSWER (built from cited facts)
- CITED FACTS (with sources)
- BINDING SCORE
- VERIFIER SCORE
- ROUTED EXPERTS (which architecture answered, with weights)

> "Every output ships with the proof chain. The model is showing me
> WHICH facts it used, WHICH experts produced the answer, and a
> coherence score for whether the answer actually follows from the
> cited facts. An LLM cannot structurally produce this."

---

## Demo Part 2 — Continual learning, no forgetting (2 minutes)

**Goal**: prove the architectural continual-learning property.

```
> learn The 2026 Northtek Vertical OS v1.1 release date is May 12, 2026.
(learned as f3)
> ask When was Northtek Vertical OS v1.1 released?
```

Show the answer cites the new fact.

> "I just added a fact that did not exist anywhere in the model's
> training. No retraining. The model is using it immediately via the
> KB-Attention layer. Watch: I'll add 20 more facts, then re-ask."

```
> learn Random unrelated fact 1
> learn Random unrelated fact 2
... (repeat or paste a batch)
> ask When was Northtek Vertical OS v1.1 released?
```

The original fact is still cited.

> "Twenty new facts. The original fact is still retrievable. **No
> catastrophic forgetting** — there are no weight updates to forget.
> This is the structural property an LLM cannot match by tuning."

---

## Demo Part 3 — Active distillation (3 minutes, requires Ollama)

**Goal**: show the model trains itself by asking a smarter model.

```bash
# Exit the REPL with Ctrl-D, restart with Ollama mode.
python scripts/rain_net_demo.py --ollama
```

```
> ask What does FAA part 135 require for icing conditions?
```

The model sees the question, has no KB on it, confidence is low. The
active-learning loop kicks in:
- Sends to Ollama (local Qwen2.5:7b)
- Ollama answers
- The answer is ingested as a new fact in semantic memory
- The model re-answers, now citing the new fact
- A warning shows: "answer learned just now via Ollama"

> "The model just trained itself. The Ollama call was free (local).
> The fact is now in semantic memory; the next user asking a related
> question gets it instantly without another teacher call. Cost per
> query asymptotes toward zero as the KB grows."

---

## Demo Part 4 — Cost comparison (2 minutes)

Open `docs/TRAINING-STRATEGY.md` to the cost table. Walk through:

| Capability | LLM cost | RAIN-Net cost |
|---|---|---|
| Llama-1B | $500K-$2M | $5K-$50K |
| Llama-8B | $5M-$20M | $50K-$500K |
| Llama-70B | $50M-$200M | $1M-$10M |

Then open `docs/SERIES-A/03-FINANCIAL-MODEL.md` to the comparison:

> "Comparable LLM startup burns $400K+/month on compute alone. We
> project $25-35K/month. That's why a $7.5M Series A funds us to
> Tier-4 capability — the same amount funds only 9 months of compute
> for an LLM startup competitor."

---

## Demo Part 5 — Architectural paper (2 minutes)

Open `docs/RAIN-NET.md`. Scroll to the 9-module diagram.

> "Reference v0.1 of every module is shipped and tested. 48 tests
> green. This isn't a slide deck — it's a working system. Arxiv
> submission target NeurIPS 2027 or ICML 2027 once we've scaled past
> Tier 2 and have benchmark wins to report."

Open `tests/test_rain_net.py` briefly.

> "The 48 tests verify every architectural primitive: hypervector
> ops, encoder bank across modalities, MoA router, hierarchical
> memory, verifier head, symbolic verifier, end-to-end RainNet
> composition. Reproducible. Open source on close."

---

## Demo Part 6 — The wedge (2 minutes)

Open `docs/SERIES-A/02-PITCH-DECK.md` to the GTM slide.

> "First two verticals: aviation compliance and regulated knowledge
> work. Both have the four properties that exclude LLMs: stale-source
> pain, audit requirement, offline preference, hard-to-scrape data.
> ACV $150K-$500K, defensible via continual learning + audit trail
> as structural moats.
>
> Once those two verticals validate, the underlying RAIN-Net is the
> platform. Adding a third vertical = new encoder + new KB seed.
> 0-to-1 expensive, 1-to-N cheap."

---

## Demo Part 7 — The ask (1 minute)

> "$7.5M, 18 months, 11-person team, Tier-4 capability, 15+ customers,
> 2 papers, open-source release.
>
> Series B trigger: $10M ARR + Tier-4 paper + 7B-trained model with
> benchmark wins.
>
> Downside protection: vertical revenue still real at Tier-3, patent
> IP independently valuable, Northtek brand provides revenue
> diversification. The bet is asymmetric.
>
> Looking for a lead investor with a post-LLM thesis and 5-10y horizon."

---

## Notes for live demo

- **Pre-load Ollama before the meeting**. First-call latency is 10-30s
  on cold start; you don't want dead air.
- **Have docs/RAIN-NET.md, the financial model, and the demo terminal
  all open in tabs**. Switching is faster than fishing through windows.
- **If a question requires looking at code**, open
  `rain/core/rain_net.py` and walk the `answer()` method — it's 30
  lines and exercises every module. Show it ONCE per meeting.
- **Don't oversell**. The architecture is real; the scaling is
  unvalidated. Investors who've heard 100 AI pitches respect honesty.
  The risk slide is in deck for a reason.
- **End on the ask, not the demo**. The demo is the warm-up; the ask
  is the close.
