# RAIN-Net Financial Model (18-month, Series A)

> Bottom-up. Conservative. Honest. Three scenarios: low / mid / high.

## Raise structure

**Target**: $7.5M (mid). Range $5M-$10M.
**Instrument**: SAFE or priced round depending on lead investor.
**Pre-money**: $25M-$40M depending on round structure.
**Dilution**: ~15-25% to investors.
**Runway**: 18 months at mid scenario.

## Use of funds (mid: $7.5M over 18 months)

| Category | $ | % | Notes |
|---|---|---|---|
| **Engineering salaries** | $3.3M | 44% | 5 senior ML/eng @ $220K + benefits |
| **Founder + 2 ops staff** | $0.6M | 8% | Founder $180K + 2 @ $150K |
| **Cloud compute** | $1.5M | 20% | Tier 2-4 training cycles |
| **Cloud inference** | $0.3M | 4% | Vertical customer pilots |
| **Vertical sales** | $0.5M | 7% | 1 sales lead + 1 SDR + travel + events |
| **Legal / IP / patents** | $0.4M | 5% | 4 patent filings + counsel + corp setup |
| **Office / equipment** | $0.2M | 3% | Workstation upgrades, lab gear |
| **Research / paper** | $0.3M | 4% | Conference travel, paper publication fees |
| **Marketing / brand** | $0.2M | 3% | Open-source launch, content, devrel |
| **Contingency** | $0.2M | 2% | |
| **Total** | $7.5M | 100% | |

## Headcount plan

| Month | Total | Hires that month |
|---|---|---|
| 0 | 1 | Founder only |
| 1 | 3 | +ML eng lead, +infra/devops |
| 2 | 5 | +ML eng 2, +ML eng 3 |
| 3 | 6 | +full-stack eng |
| 4 | 7 | +vertical sales lead |
| 6 | 8 | +applied scientist (multi-modal) |
| 8 | 9 | +SDR / customer success |
| 12 | 10 | +ops / hiring lead |
| 15 | 11 | +research scientist |
| 18 | 11 | (closeout / Series B prep) |

## Revenue plan (vertical pilots)

| Month | Customers | ARR | Cumulative cash |
|---|---|---|---|
| 0-5 | 0 | $0 | $0 (research mode) |
| 6 | 1 (aviation pilot) | $50K | $50K |
| 9 | 3 (aviation + medical pilot + legal pilot) | $250K | $350K |
| 12 | 6 | $700K | $1.3M |
| 15 | 10 | $1.5M | $3.0M |
| 18 | 15 | $2.5M | $5.5M |

Mid scenario assumes ~$150K ACV blended across verticals.

## Sensitivity (low / mid / high)

| | Low | **Mid** | High |
|---|---|---|---|
| Raise | $5M | **$7.5M** | $10M |
| Runway | 14 mo | **18 mo** | 22 mo |
| HC peak | 8 | **11** | 14 |
| Tier reached | 3 (3B base) | **4 (7B base)** | 4 + multi-modal |
| Customer count (18mo) | 8 | **15** | 25 |
| ARR (18mo) | $1.2M | **$2.5M** | $4.5M |
| Series B trigger | $5M ARR | **$10M ARR + Tier-4 paper** | $15M ARR |

## Cost structure: why we can run lean vs LLM startup competitors

Comparable LLM startup at Series A typically burns $2M-$3M/month on
compute alone. We project $80K-$100K/month.

| Cost item | LLM startup | RAIN-Net |
|---|---|---|
| Pretraining cluster | $200K+/mo | $0 (distill from teachers) |
| Inference GPU fleet | $100K+/mo | $20K-$30K/mo (small base + experts) |
| RLHF infrastructure | $50K+/mo | $0 (use existing judge LLMs) |
| Data acquisition | $50K+/mo | $5K/mo (synthetic curriculum) |
| Total typical compute | $400K+/mo | $25K-$35K/mo |
| Implied annual compute | $5M+ | $0.3M-$0.4M |
| 18-month compute total | $7.5M+ | $0.5M-$0.7M |

**This is why a $7.5M Series A can take RAIN-Net to Tier-4 capability.**
The same raise barely funds 9 months of frontier-LLM-style compute.

## Comparable companies (for valuation reference)

| Company | Stage | Raise | Pre-money | Architecture/Differentiation |
|---|---|---|---|---|
| Liquid AI | Series A | $250M | ~$2B | "Liquid neural nets" — limited benchmarks |
| Sakana AI | Series A | $30M | ~$200M | Evolutionary model merging |
| Together AI | Series A | $20M | ~$100M | Open-model inference + training infra |
| Anthropic seed (2021) | seed | $124M | ~$1B | Safety + RLHF research |
| Mistral seed (2023) | seed | $113M | ~$260M | Open European LLM lab |

**Our positioning**: research-stage non-LLM architecture bet. Comparable
to Liquid AI on novelty axis; comparable to early Mistral on
open-source-credibility axis. Pre-money $25M-$40M reasonable given
working v0.1 + paper-ready spec + clear cost story.

## Series B target metrics (18mo)

To raise a Series B at $50M+ pre-money in month 24:

- $10M+ ARR ($150K-$300K ACV × 30-60 customers)
- 1-2 published peer-reviewed papers (NeurIPS / ICML)
- 7B-param trained system with public benchmark wins on continual + audit
- Open-source v1.0 with >5K GitHub stars
- 2+ enterprise reference customers willing to be quoted

## Downside protection

If RAIN-Net does NOT scale to Tier-4 as projected:

- Vertical revenue still real at Tier-3 scale ($1M-$3M ARR achievable)
- Patent IP has independent licensing value
- Northtek umbrella brand (Vertical OS, NORTHTEKDevs products) provides
  revenue diversification
- Team is hireable at competitive AI lab valuations (no zero-out)

The downside scenario is "specialist AI company with $5-15M ARR + IP
portfolio" — not zero. This makes the bet asymmetric.
