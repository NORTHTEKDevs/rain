# RAIN Funding Waypoints

**Companion to:** `docs/plans/2026-05-22-rain-design.md`
**Date:** 2026-05-22
**Status:** Working plan, revisit monthly

---

## Total dollar cost to v1.0

| Phase | Compute cost | Wall time | Self-fundable? |
|---|---|---|---|
| v0 (toy, 1-2 weeks training) | $200-500 | 12 weeks engineering | Yes — operating budget |
| v0.5 (~1 month training) | ~$2,000 | 12-16 weeks after v0 | Yes — single compute grant |
| v1.0 (~1-2 months training) | $20-50k | 12-26 weeks after v0.5 | Needs real fundraising |
| v2.0 federated (per specialist) | $5-10k × N specialists | concurrent | Mix of grants + revenue |

**Total path to v1.0:** ~$25-55k compute + 9-12 months engineering.

GPT-4 cost ~$100M+. RAIN to v1.0 is ~2,000-4,000x cheaper. This ratio is the
load-bearing claim for the entire fundraising story.

---

## Funding sources by stage

### v0 ($200-500) — operating budget only

No fundraising required. Train phase 1 on a workstation 4090 over 1-2 weeks.

### v0.5 ($2,000) — any single compute grant

| Source | Realistic ask | Effort | Notes |
|---|---|---|---|
| Google Cloud Research Credits | $5k cloud credit | Low | Renewable annually. |
| AWS Activate | $1k-5k cloud credit | Low | Startup tier easy to qualify for. |
| NVIDIA Inception | GPU access + credits | Medium | Application + interview; NVIDIA likes alt-arch ML. |
| Microsoft for Startups | $1-25k Azure credit | Low | Founders Hub. |
| Lambda Labs Research Credits | $1-10k | Low | Direct application; novel-architecture-friendly. |

**Action:** apply to 3-4 of these in parallel during v0 build weeks 1-4. Stack
the wins.

### v1.0 ($20-50k) — real fundraising window

#### Non-dilutive — preferred

| Source | Realistic ask | Effort | Notes |
|---|---|---|---|
| NSF SBIR Phase I | $275k | High | 6-9 month cycle. Confidential novel ML from small business is textbook fit. |
| NSF SBIR Phase II | $1-1.5M (after Phase I) | High | Sequential after Phase I. |
| DARPA AIE Open BAA | $250k-1M | High | Solicitation themes: assured autonomy, calibrated AI, robust ML — RAIN fits. |
| DOE / DARPA SBIR | $250k-1M | High | Various solicitations. |
| Schmidt Sciences | $50-500k | Medium | Funds "different paradigms of AI" outside LLM mainstream. |
| Open Philanthropy AI grants | $50-500k | Medium | Specifically interested in non-mainstream AI safety-positive architectures. |
| Astera Institute | $50-300k | Medium | Funds novel AI research; small grants are fast. |
| Manifund | $10-100k | Low | Faster cycle, smaller grants, public application. |
| ARIA (UK) | varies | Medium | If UK collaboration possible; novel-arch AI is in scope. |

#### Dilutive — only if non-dilutive doesn't close the gap

| Source | Realistic ask | Equity | Notes |
|---|---|---|---|
| Pre-seed angels (alt-arch AI) | $50-250k | 5-15% | RWKV, Liquid AI, Cartesia have backers actively scouting. |
| YC | $500k | 7% | Standard. |
| a16z Speedrun | $1M | 8% | AI-focused. |
| Conviction / Felicis / South Park Commons | $500k-2M | 5-15% | Novel-arch friendly. |

### v2.0 federated ($5-10k/specialist) — revenue-supported

By v2.0, v1.0 should be demonstrating commercial use cases. Specialist
training funded by early-customer revenue or by the v1.0 SBIR Phase II if it
landed.

---

## What to pitch to whom

### Compute-grant orgs (Google / AWS / NVIDIA / Microsoft / Lambda)

Lead with: "Novel non-transformer generative architecture. Sub-1% the training
cost of equivalent-scale LLMs. Confidential. Looking for compute credits to
run Phase 1 bootstrap." Show the cost-ratio chart. They love this story.

### NSF SBIR

Lead with: technical merit + commercialization potential. NSF specifically
funds the "alternative to transformer LLMs" angle because of national-AI-
strategy interest in not being dependent on one architectural class. Show:
- Confidential architecture (six novelty claims documented internally).
- Falsifiable benchmarks (Tier 1/2/3 in design doc §5).
- 70% of code already exists across 5 prior repos.
- Commercialization path (alaska-watchdog, kryos, fieldwork already use
  Northtek IP commercially).

### DARPA

Lead with: military / national-security applicable properties. RAIN's wins
align with DARPA themes:
- **Assured autonomy** — calibrated outputs, structural uncertainty.
- **Continual learning** — agents that adapt in deployment.
- **Resource constraint** — edge-deployable, cheap to train.
- **Explainability** — every output has a citation chain.

### Schmidt / Open Philanthropy / Astera

Lead with: research importance + alignment. RAIN is structurally aligned with
several alignment desiderata:
- Calibrated outputs reduce overconfidence.
- Structural transparency reduces deception risk.
- Continual learning without forgetting reduces retraining incentive (less
  data exfiltration risk).

### Alt-arch AI angels

Lead with: market timing + moat. RWKV and LFM2 are catching the "post-
transformer" wave. RAIN is the next-after-that wave: post-attention AND
post-backprop AND structurally cognitive. Investors looking for the next
RWKV/Liquid are looking for exactly this.

---

## Outreach timeline (rough)

| Month from today | Action |
|---|---|
| 0 (now) | v0 build kickoff. No fundraising required. |
| 1-2 | Compute-grant applications (Google / AWS / NVIDIA / MS / Lambda). Goal: $5-10k credits stacked. |
| 2-3 | v0 demo video produced; Tier 1/2/3 results green. |
| 3-4 | NSF SBIR Phase I application. Astera Institute small-grant application. Manifund post. |
| 4-6 | v0.5 build using stacked compute credits. |
| 6-9 | v0.5 results published internally. DARPA AIE application if relevant solicitation open. Schmidt Sciences + Open Philanthropy outreach. |
| 9-12 | v1.0 build. SBIR Phase I results land (or Phase II application). |
| 12+ | v1.0 demo to alt-arch angels if needed. Public-launch decision. |

---

## Critical fundraising-protective rules

1. **No public benchmark posts.** Even on private channels, anyone could
   screenshot.
2. **Sign NDAs before any pitch.** Standard, non-negotiable.
3. **One-line architecture descriptions only** in outreach until under NDA:
   "A new non-LLM generative AI based on Vector Symbolic Architecture and
   Active Inference. Confidential. Trained at <1% the cost of equivalent-
   scale LLMs."
4. **No code in pitch decks.** Demo videos + benchmark numbers + the
   one-sentence claim. Code is for under-NDA technical diligence.
5. **No co-presenting with anyone** who hasn't signed an NDA. No interns,
   contractors, accelerator-cohort-mates.
6. **Founder-friendly term sheets only.** No board control to early angels,
   no super-voting common to investors. SAFEs preferred over priced rounds
   for pre-seed.

---

## Open questions for the user

1. **Trademark strategy?** "RAIN" + "Resonant Active Inference Network" +
   "NORTHTEKDevs" should all be searched and reserved before launch.
3. **Domain?** rain.dev, rainai.dev, raincore.ai, rain.ai (likely taken),
   resonantai.dev. Pick + park ASAP.
4. **Entity structure?** Northtek LLC / Frostbyte LLC currently. For SBIR,
   founder must own >50% and be a US citizen — confirm both.
5. **Co-founder / first hire timing?** Solo-dev path works through v0.5;
   v1.0 + fundraising benefits from a technical co-founder or first ML
   engineer.
