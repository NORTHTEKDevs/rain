"""Reflexion loop: iterative self-critique using the verifier head.

Inspired by Shinn et al. 2023, "Reflexion: Language Agents with Verbal
Reinforcement Learning" (NeurIPS). LLM-based reflexion uses GPT-4 as
both generator and critic. RAIN-Net does it at much smaller scale:
the MoA router generates N candidates, the verifier head scores them,
the lowest-scored is regenerated with a critique HV that captures
what the high-scored candidate "had" that the low one didn't.

The critique HV is constructed in the substrate:
    critique_hv = unbind(query_hv, best_candidate_hv)
This isolates the "what's missing in the bad answer" component in HV
space. We then bias the regeneration by binding the critique HV with
the original query before re-routing.

Why this is novel against LLM reflexion:
    - No LLM-as-critic call required (zero API cost).
    - Critique is structured (an HV) not natural-language; reproducible.
    - Iterations are cheap: O(N candidates * router cost) per round.
    - Audit trail records every round's score progression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rain.core.hv_substrate import bind, similarity, unbind
from rain.core.rain_net import RainNet
from rain.core.symbolic_verifier import AuditReport
from rain.core.verifier_head import sample_and_select


@dataclass
class ReflexionRound:
    round_idx: int
    candidate_scores: list[float]
    best_score: float
    chosen_idx: int
    improvement_from_prev: float


@dataclass
class ReflexionResult:
    final_answer_hv: np.ndarray
    final_score: float
    rounds: list[ReflexionRound]
    n_candidates_per_round: int
    final_report: AuditReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def reflect(
    net: RainNet,
    query: str,
    modality: str = "text",
    n_candidates: int = 8,
    max_rounds: int = 3,
    improvement_threshold: float = 0.05,
) -> ReflexionResult:
    """Run a reflexion loop.

    1. Sample N candidates from the routed experts.
    2. Verifier scores them; pick best.
    3. If best score is high enough, return.
    4. Otherwise: form a critique HV (best_hv unbound from query_hv),
       bias the query by binding the critique, re-route, re-sample.
    5. Repeat up to max_rounds or until improvement stalls.

    Parameters
    ----------
    n_candidates : how many candidates to sample per round
    max_rounds   : hard cap on reflexion iterations
    improvement_threshold : if best-score improves by less than this
                            between rounds, stop early
    """
    query_hv = net.encoder_bank.encode(modality, query)
    rounds: list[ReflexionRound] = []

    def _sample_round(seed_query_hv: np.ndarray) -> tuple[np.ndarray, list[float], int]:
        cands: list[np.ndarray] = []
        for _ in range(n_candidates):
            ans, _prov = net.router.forward(seed_query_hv)
            cands.append(ans)
        chosen, idx, all_scores = sample_and_select(
            cands, query_hv, net.verifier, return_scores=True
        )
        return chosen, all_scores, idx

    # Round 0
    chosen, all_scores, idx = _sample_round(query_hv)
    best_score = float(all_scores[idx])
    rounds.append(
        ReflexionRound(
            round_idx=0,
            candidate_scores=all_scores,
            best_score=best_score,
            chosen_idx=idx,
            improvement_from_prev=0.0,
        )
    )
    best_hv = chosen

    # Iterative critique rounds.
    prev_score = best_score
    for r in range(1, max_rounds):
        # Critique HV: what does the best candidate carry that the original
        # query did not? unbind(query, best) isolates the "answer contribution."
        critique_hv = unbind(query_hv, best_hv)
        # New seed query: bias the original query by binding with critique.
        # In HV algebra this nudges routing toward the answer subspace.
        biased_query = bind(query_hv, critique_hv)
        # Re-sample with the biased query.
        new_chosen, new_scores, new_idx = _sample_round(biased_query)
        new_best = float(new_scores[new_idx])
        improvement = new_best - prev_score
        rounds.append(
            ReflexionRound(
                round_idx=r,
                candidate_scores=new_scores,
                best_score=new_best,
                chosen_idx=new_idx,
                improvement_from_prev=improvement,
            )
        )
        if new_best > best_score:
            best_score = new_best
            best_hv = new_chosen
        prev_score = new_best
        # Early stop if improvement stalled.
        if abs(improvement) < improvement_threshold:
            break

    return ReflexionResult(
        final_answer_hv=best_hv,
        final_score=best_score,
        rounds=rounds,
        n_candidates_per_round=n_candidates,
    )


def reflect_and_answer(
    net: RainNet,
    query: str,
    modality: str = "text",
    n_candidates: int = 8,
    max_rounds: int = 3,
) -> AuditReport:
    """Convenience: run reflexion + produce a full AuditReport.

    Returns the same AuditReport shape as RainNet.answer(), with the
    chosen answer being the highest-scored across all reflexion rounds.
    """
    result = reflect(
        net=net,
        query=query,
        modality=modality,
        n_candidates=n_candidates,
        max_rounds=max_rounds,
    )
    # Retrieve cited facts for the answer.
    query_hv = net.encoder_bank.encode(modality, query)
    cited = [f for _s, f in net.memory.semantic.search(
        query_hv, top_k=net.config.semantic_top_k
    )]
    # Build the audit report.
    from rain.core.symbolic_verifier import make_audit_report

    text = net._synthesise_answer_text(query, cited)
    rounds_summary = (
        f"reflexion: {len(result.rounds)} rounds, "
        f"final_score={result.final_score:+.3f}, "
        f"scores=" + ",".join(f"{r.best_score:+.3f}" for r in result.rounds)
    )
    report = make_audit_report(
        answer_text=text,
        answer_hv=result.final_answer_hv,
        cited_facts=cited,
        verifier_score=result.final_score,
        provenance=[(f"reflexion_round_{r.round_idx}", r.best_score) for r in result.rounds],
        tsetlin_model=getattr(net, "_tsetlin", None),
        confidence_thresh=net.config.confidence_threshold,
    )
    report.warnings.insert(0, rounds_summary)
    return report
