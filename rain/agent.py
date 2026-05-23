# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
# Original-source: third_party/rck/conscious_agent.py — adapted to use RAIN's
# bipolar 10K modules.
"""ConsciousAgent — top-level orchestrator.

Composes Codebook + ShardedKB + DialogueContext + Introspector + SelfModel +
TheoryOfMind + CalibrationTally + multi-hop inference + NL output. Exposes
the public interaction methods: ask, tell, describe, think_aloud, refute,
introspect.

This is the v0 cognitive surface. It does NOT use HYMN/LSM/FEP/Tsetlin yet
for token generation (those come online when the full EFE decoder integrates
in Phase 6). v0 ConsciousAgent operates over the symbolic-KB layer and
provides the ToM/self-model/calibration/transparency capabilities that
distinguish RAIN from LLMs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rain.cognition.compose_answer import describe as describe_fn
from rain.cognition.dialogue import DialogueContext
from rain.cognition.explain import explain
from rain.cognition.inference import InferenceResult, infer
from rain.cognition.introspect import Introspector
from rain.cognition.metacog import CalibrationTally, classify_epistemic
from rain.cognition.self_model import SelfModel
from rain.cognition.theory_of_mind import TheoryOfMind
from rain.cognition.think_aloud import narrate
from rain.core.fep import LowRankA
from rain.core.knowledge_base import ShardedKB
from rain.core.liquid_state import LiquidStateMachine
from rain.core.relational import Codebook
from rain.core.tsetlin import TsetlinMachine

# Callable signature: (prompt: str, n_tokens: int) -> str.
# Lets us attach any HYMN sampler (or any other fallback generator) without
# pulling a torch dep into rain.agent.
HymnSamplerFn = Callable[[str, int], str]


@dataclass
class Answer:
    """One agent reply with all the metadata RAIN exposes (calibration + epistemic + citations)."""
    text: str
    epistemic: str   # know / think / guess / unknown
    citations: list[tuple[str, str, str]] = field(default_factory=list)
    confidence: float = 0.0
    inference_source: str | None = None  # direct / inherited / transitive / None / hymn
    cognitive_signals: dict[str, float] | None = None  # populated when enable_continual


class ConsciousAgent:
    def __init__(
        self,
        dim: int = 10000,
        num_shards: int = 64,
        seed: int = 0,
        *,
        enable_continual: bool = False,
        lsm_reservoir: int = 256,
        fep_rank: int = 64,
        tsetlin_classes: int = 16,
        tsetlin_clauses_per_class: int = 16,
    ) -> None:
        self.dim = dim
        self.codebook = Codebook(vocab_size=4096, dim=dim, seed=seed)
        self.kb = ShardedKB(num_shards=num_shards, dim=dim, seed=seed)
        self.dialogue = DialogueContext()
        self.introspect = Introspector(capacity=128)
        self.self_model = SelfModel(dim=dim, num_shards=8, seed=seed + 1)
        self.tom = TheoryOfMind(dim=dim, num_shards=8, seed=seed + 2)
        self.calibration = CalibrationTally()
        # Optional HYMN sampling fallback for ask() on KB-miss. Set via
        # attach_hymn_sampler(); kept off by default so the structural-KB
        # behavior remains the v0 reference.
        self._hymn_sampler: HymnSamplerFn | None = None
        self._hymn_n_tokens: int = 120
        # Phase-2 continual-learning surfaces. OFF by default to preserve
        # the v0 reference behavior; opt in with enable_continual=True.
        if enable_continual:
            self.lsm = LiquidStateMachine(
                input_dim=dim, reservoir_dim=lsm_reservoir,
                output_dim=dim, seed=seed + 3,
            )
            self.fep = LowRankA(D=dim, R=fep_rank, seed=seed + 4)
            self.tsetlin = TsetlinMachine(
                num_classes=tsetlin_classes,
                num_clauses_per_class=tsetlin_clauses_per_class,
                num_features=dim, seed=seed + 5,
            )
            self._continual = True
        else:
            self.lsm = None
            self.fep = None
            self.tsetlin = None
            self._continual = False

    def attach_hymn_sampler(self, sampler: HymnSamplerFn, n_tokens: int = 120) -> None:
        """Wire a trained-HYMN sampler as the KB-miss fallback for ask().

        sampler(prompt, n_tokens) -> str. When ask() can't resolve a fact via
        the KB, it asks the sampler for a continuation and returns it with
        epistemic='guess' and inference_source='hymn'. The calibration tally
        for the relation is NOT updated automatically -- HYMN guesses are
        weaker than KB facts and Phase-2 feedback should evaluate them
        explicitly.
        """
        self._hymn_sampler = sampler
        self._hymn_n_tokens = n_tokens

    def tell(self, subject: str, relation: str, object_: str) -> None:
        """Teach a new fact. Records introspection + updates dialogue context.

        When `enable_continual=True` on the agent, this also drives the
        Phase-2 local-rule learners (LSM RLS, FEP rank-1 A, Tsetlin Type-I
        feedback) so the cognitive surfaces' weights evolve every time a
        new fact is taught. No global gradient, no backprop -- just the
        per-rule updates each module exposes.
        """
        self.kb.write(subject, relation, object_)
        self.dialogue.remember(entity=subject, relation=relation)
        self.introspect.record("learn", fact=f"{subject}.{relation}={object_}")
        if self._continual:
            self._drive_continual_update(subject, relation, object_)

    def _drive_continual_update(self, subject: str, relation: str, object_: str) -> None:
        """Run the local-rule learners off a new (s, r, o) fact. Best-effort
        -- any module-level exception is silently swallowed so a single
        bad input doesn't disrupt the KB write that already happened.
        """
        state_vec = self.codebook.vector(subject).astype(np.float32)
        target_vec = self.codebook.vector(object_).astype(np.float32)
        # LSM: reservoir step + RLS readout update.
        try:
            self.lsm.step(state_vec)
            self.lsm.update(target_vec)
        except Exception:
            pass
        # FEP: low-rank A update toward the target.
        try:
            self.fep.update(state_vec, target_vec, alpha=0.01)
        except Exception:
            pass
        # Tsetlin: per-relation Type-I feedback. Map relation -> stable class
        # index via a deterministic hash.
        try:
            cls = abs(hash(relation)) % self.tsetlin.num_classes
            bipolar = np.sign(state_vec + 1e-9).astype(np.int8)
            self.tsetlin.feedback(bipolar, target_class=cls)
        except Exception:
            pass

    def continual_state_snapshot(self) -> dict:
        """Quick diagnostic: per-module shape/parameter counts after N updates."""
        if not self._continual:
            return {"enabled": False}
        return {
            "enabled": True,
            "lsm": {
                "reservoir_dim": self.lsm.reservoir_dim,
                "output_dim": self.lsm.output_dim,
                "state_l2": float(np.linalg.norm(self.lsm.state)),
                "W_out_l2": float(np.linalg.norm(self.lsm.W_out)),
            },
            "fep": {
                "rank": self.fep.rank(),
                "U_l2": float(np.linalg.norm(self.fep.U)),
                "V_l2": float(np.linalg.norm(self.fep.V)),
            },
            "tsetlin": {
                "num_classes": self.tsetlin.num_classes,
                "num_clauses_per_class": self.tsetlin.num_clauses_per_class,
                "active_inclusions": int(np.sum(self.tsetlin.inclusion > 0)),
            },
        }

    def ask(self, subject: str, relation: str, think_aloud: bool = False) -> Answer:
        """Answer a fact-shaped question. Returns an Answer with epistemic class + citations."""
        # Resolve references from dialogue context
        subject_r, relation_r = self.dialogue.resolve_references(subject, relation)
        if subject_r is None or relation_r is None:
            ans = Answer(text="I don't have enough context to know what you're asking.",
                         epistemic="unknown")
            self.introspect.record("refuse", reason="missing context")
            return ans

        result = infer(self.kb, subject_r, relation_r)
        # Compute confidence: 1.0 for direct, 0.7 for inherited, 0.5 for transitive, 0.0 for None
        conf_map = {"direct": 1.0, "inherited": 0.7, "transitive": 0.5, None: 0.0}
        confidence = conf_map[result.source]
        cal = self.calibration.calibration(relation_r)
        epistemic = classify_epistemic(confidence, relation_calibration=cal)

        if result.answer is None:
            # KB miss. If a HYMN sampler is attached, use it as the fallback;
            # otherwise refuse cleanly. Either way, we DON'T pretend the
            # answer is grounded -- epistemic stays at the conservative end.
            if self._hymn_sampler is not None:
                prompt = f"{subject_r} {relation_r.replace('_', ' ')} "
                hymn_text = self._hymn_sampler(prompt, self._hymn_n_tokens)
                text = prompt + hymn_text
                self.introspect.record("hymn_guess", topic=f"{subject_r}.{relation_r}")
                self.dialogue.remember(entity=subject_r, relation=relation_r)
                self.dialogue.record_turn("user", f"ask({subject_r}, {relation_r})")
                self.dialogue.record_turn("rain", text)
                return Answer(
                    text=text,
                    epistemic="guess",
                    citations=[],
                    confidence=0.3,
                    inference_source="hymn",
                )
            text = "I don't know that yet."
            self.introspect.record("refuse", reason="no fact / no chain")
        elif think_aloud:
            text = narrate(f"what is {relation_r} of {subject_r}?", result)
            self.introspect.record("explain", topic=f"{subject_r}.{relation_r}")
        else:
            explanation = explain(result)
            text = explanation.text
            self.introspect.record("answer", answer=result.answer)

        # Update dialogue context with this turn's subject + relation
        self.dialogue.remember(entity=subject_r, relation=relation_r)
        self.dialogue.record_turn("user", f"ask({subject_r}, {relation_r})")
        self.dialogue.record_turn("rain", text)

        cognitive = self._cognitive_signals_for(subject_r, result) if self._continual else None
        return Answer(
            text=text,
            epistemic=epistemic,
            citations=list(result.chain),
            confidence=confidence,
            inference_source=result.source,
            cognitive_signals=cognitive,
        )

    def _cognitive_signals_for(self, subject: str, result: InferenceResult) -> dict[str, float]:
        """Read-side use of the LSM / FEP / Tsetlin surfaces.

        Returns a small diagnostic dict that exposes how strongly each
        cognitive surface 'agrees' with the KB answer. At v0 these are
        observability-only; future work can fold them into confidence/
        epistemic. The signals:
          * fep_cos: cosine similarity between FEP.predict(state) and
                     codebook(answer). Higher = FEP agrees.
          * lsm_state_l2: L2 norm of the LSM reservoir state after one
                          step on the state vector (rough 'familiarity').
          * tsetlin_votes_class: per-relation Tsetlin vote, normalized.
        """
        out: dict[str, float] = {}
        state_vec = self.codebook.vector(subject).astype(np.float32)
        try:
            self.lsm.step(state_vec)
            out["lsm_state_l2"] = float(np.linalg.norm(self.lsm.state))
        except Exception:
            out["lsm_state_l2"] = -1.0
        if result.answer is not None:
            try:
                pred = self.fep.predict(state_vec)
                target = self.codebook.vector(result.answer).astype(np.float32)
                num = float(pred @ target)
                den = float(np.linalg.norm(pred) * np.linalg.norm(target) + 1e-9)
                out["fep_cos"] = num / den
            except Exception:
                out["fep_cos"] = 0.0
        else:
            out["fep_cos"] = 0.0
        try:
            bipolar = np.sign(state_vec + 1e-9).astype(np.int8)
            votes = self.tsetlin.vote(bipolar)
            max_vote = int(np.max(np.abs(votes))) if votes.size else 0
            out["tsetlin_max_abs_vote"] = float(max_vote)
        except Exception:
            out["tsetlin_max_abs_vote"] = 0.0
        return out

    def describe(self, entity: str, relations: list[str] | None = None) -> str:
        """Multi-sentence description from stored facts."""
        if relations is None:
            relations = [
                "isa", "category", "color", "size", "capital_of", "locatedin",
                "lives_in", "made_of", "has_part", "wrote", "creator",
            ]
        out = describe_fn(self.kb, entity, relations)
        self.introspect.record("explain", topic=entity)
        return out

    def feedback(self, relation: str, was_correct: bool) -> None:
        """User feedback on a previous answer — updates per-relation calibration tally."""
        self.calibration.update(relation, was_correct)

    def self_describe(self) -> str:
        """Plain-language description of RAIN itself from self-model."""
        return " ".join(self.self_model.describe())

    def what_just_happened(self) -> str:
        """Introspection narration."""
        return self.introspect.narrate()
