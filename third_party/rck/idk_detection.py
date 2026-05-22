"""Explicit "I don't know" detection.

Every retrieval returns SOME top candidate from the codebook -- the
HRR cleanup always picks one. Without a calibration step, "I don't
know" never gets reported. This module formalises the gap:

  * `EpistemicState.KNOWN`     -- top-1 above threshold, supported.
  * `EpistemicState.AMBIGUOUS` -- multiple candidates above threshold
                                  with similar scores; we have several
                                  plausible answers.
  * `EpistemicState.IDK`       -- no candidate clears the threshold;
                                  we genuinely don't know.

The classifier is conservative: we'd rather say IDK than hallucinate
a confident wrong answer. The thresholds are tunable via `IDKPolicy`.

Output shape:
    EpistemicAnswer(state, top_symbol, top_score, alternatives,
                    explanation, queried_path)

`alternatives` is the list of competing candidates when state is
AMBIGUOUS. `queried_path` records what was searched -- direct
KB, chain walker, chain induction -- so the IDK answer carries
context about what was tried.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Optional

from rck.knowledge_base import ShardedKnowledgeBase


class EpistemicState(enum.Enum):
    KNOWN = "known"
    AMBIGUOUS = "ambiguous"
    IDK = "idk"


@dataclass
class IDKPolicy:
    # Minimum score for top-1 to be considered KNOWN.
    known_threshold: float = 0.15
    # If top-2 score is within this fraction of top-1, declare ambiguous.
    ambiguity_ratio: float = 0.85
    # If even the top-1 is below this, declare IDK outright.
    idk_threshold: float = 0.08
    # How many alternatives to surface when AMBIGUOUS.
    max_alternatives: int = 3


@dataclass
class EpistemicAnswer:
    state: EpistemicState
    top_symbol: Optional[Hashable] = None
    top_score: float = 0.0
    alternatives: list[tuple[Hashable, float]] = field(default_factory=list)
    explanation: str = ""
    # When the agent layer compares this episode against history,
    # `drift_from_prior` is set to a brief description if state/answer
    # changed vs the most recent prior episode of the same signature.
    drift_from_prior: Optional[str] = None
    queried_path: str = "direct"

    def verbalize(self) -> str:
        if self.state is EpistemicState.KNOWN:
            return (f"I'm confident: the answer is {self.top_symbol!r} "
                    f"(score {self.top_score:.2f}).")
        if self.state is EpistemicState.AMBIGUOUS:
            opts = ", ".join(f"{s!r} ({sc:.2f})"
                              for s, sc in self.alternatives)
            return (f"I have multiple plausible answers: {opts}. "
                    f"{self.explanation}")
        return (f"I don't know. {self.explanation}")


def classify(candidates: Iterable[tuple[Hashable, float]],
             *, policy: IDKPolicy | None = None,
             queried_path: str = "direct") -> EpistemicAnswer:
    """Classify a ranked candidate list into KNOWN / AMBIGUOUS / IDK."""
    cfg = policy or IDKPolicy()
    cs = list(candidates)
    if not cs:
        return EpistemicAnswer(
            state=EpistemicState.IDK,
            explanation="No candidates returned by the KB query.",
            queried_path=queried_path,
        )
    top_sym, top_score = cs[0]
    top_score = float(top_score)
    if top_score < cfg.idk_threshold:
        return EpistemicAnswer(
            state=EpistemicState.IDK,
            top_symbol=top_sym, top_score=top_score,
            explanation=(f"top score {top_score:.3f} below IDK floor "
                         f"{cfg.idk_threshold}"),
            queried_path=queried_path,
        )
    # Ambiguity check: how close is top-2 to top-1?
    if len(cs) >= 2:
        second_score = float(cs[1][1])
        if second_score >= top_score * cfg.ambiguity_ratio:
            return EpistemicAnswer(
                state=EpistemicState.AMBIGUOUS,
                top_symbol=top_sym, top_score=top_score,
                alternatives=[(s, float(sc))
                               for s, sc in cs[:cfg.max_alternatives]],
                explanation=(f"top-2 ({second_score:.3f}) is within "
                             f"{cfg.ambiguity_ratio:.0%} of top-1 "
                             f"({top_score:.3f})."),
                queried_path=queried_path,
            )
    if top_score < cfg.known_threshold:
        # Above the IDK floor but below the KNOWN bar -- soft IDK.
        return EpistemicAnswer(
            state=EpistemicState.IDK,
            top_symbol=top_sym, top_score=top_score,
            explanation=(f"top score {top_score:.3f} below known "
                         f"threshold {cfg.known_threshold}"),
            queried_path=queried_path,
        )
    return EpistemicAnswer(
        state=EpistemicState.KNOWN,
        top_symbol=top_sym, top_score=top_score,
        queried_path=queried_path,
    )


def ask_with_idk(kb: ShardedKnowledgeBase,
                 known: dict[str, Hashable], unknown_role: str,
                 *, top_k: int = 5,
                 policy: IDKPolicy | None = None) -> EpistemicAnswer:
    """Query the KB and return an EpistemicAnswer instead of a raw
    (symbol, score) tuple."""
    candidates = kb.query(known, unknown_role, top_k=top_k)
    return classify(candidates, policy=policy, queried_path="direct")
