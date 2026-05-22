"""ConsciousAgent -- RCK v1.3 top-level model.

Brings together:
  - ShardedKnowledgeBase (scales to tens of thousands of facts).
  - Self-model (RCK knows what RCK is).
  - Introspection (think() reports current internal state).
  - Meta-cognition (verbalised, calibrated confidence).
  - Theory-of-mind (belief tuples in a separate belief KB).
  - LM agent (free-form generation fallback).

We deliberately avoid making any claim about phenomenal consciousness.
What this class IMPLEMENTS is the operational subset of the Global
Workspace Theory and its higher-order variants:

  - A workspace that integrates module outputs (already in `rck.workspace`).
  - A broadcast history that the model can introspect over.
  - A self-model the model can query.
  - Meta-cognitive verbalisation of its own confidence.

These are the verifiable, scientifically-grounded capabilities that
consciousness theories propose. Whether THIS amounts to consciousness is
a separate philosophical question. We do not answer it here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable

from rck.actions import ActionRegistry, make_default_registry
from rck.agent import RCKAgent
from rck.bulk_ingest import inverse_relation
from rck.compose_answer import describe as compose_describe
from rck.corrections import detect_correction
from rck.dialogue import DialogueContext
from rck.personality import Personality
from rck.generative import GenerativeRCK, parse_question, _extract_simple_facts
from rck.inference import (
    boolean as kb_boolean, compare as kb_compare,
    enumerate_subjects, infer,
)
from rck.introspect import IntrospectionBuffer, think
from rck.knowledge_base import ShardedKnowledgeBase
from rck.metacog import CalibrationTally, epistemic_category, verbalize
from rck.multistep import two_step
from rck.negation import negate_boolean
from rck.nlg import render, render_chain, render_enumeration
from rck.numbers import evaluate_arithmetic
from rck.open_ie import extract_triples_from_text
from rck.self_model import (
    SELF_NAME, install_self_model, self_describe,
)
from rck.analogy import AnalogyResult, solve_analogy
from rck.chain_discover import Goal, discover_chains
from rck.confidence_calibration import (
    CalibratedAnswer, calibrated_lookup,
)
from rck.idk_detection import (
    EpistemicAnswer, EpistemicState, IDKPolicy, ask_with_idk,
)
from rck.episodic_consolidate import (
    ConsolidationReport, consolidate as _consolidate_episodes,
)
from rck.gap_detection import Gap, find_gaps as _find_gaps
from rck.skill_clustering import (
    SkillFamily, cluster_skills_by_prefix,
)
from rck.skill_promotion import (
    PromotionPolicy, promote_families as _promote_families,
)
from rck.negation_propagation import (
    PropagatedNegation, propagate_negations as _propagate_negations,
)
from rck.negative_facts import (
    deny as _deny_fact,
    filter_against_negatives as _filter_negatives,
)
from rck.query_memory import QueryMemory
from rck.chain_cache import ChainCache
from rck.belief_revision import ResolutionPlan, resolve_all
from rck.explain_why import ExplanationNode, explain as _explain_why
from rck.contradiction import (
    Conflict, ContradictionPolicy, detect_conflicts,
)
from rck.set_reasoning import (
    SetCandidate, intersect_queries, union_queries,
)
from rck.provenance import ProvenanceStore
from rck.chain_induction import (
    InducedFact, InductionPolicy, induce_from_chain,
)
from rck.chain_walker import Hop, walk_chain
from rck.rule_cascade import RuleCascadeResult, cascade_instantiate
from rck.rule_composition import compose_all
from rck.rule_extraction import Rule, RuleStore, extract_rules
from rck.rule_instantiation import InstantiatedFact, instantiate_all
from rck.confidence_propagation import PropagationConfig
from rck.shard_sizing import recommend_shards
from rck.skills import SkillLibrary
from rck.self_verify import verify as _verify_kb
from rck.synonyms import canonical_entity, canonical_relation
from rck.temporal import temporal_answer
from rck.theory_of_mind import (
    make_belief_kb, store_belief, what_does_x_think,
)
from rck.think_aloud import narrate, narrate_no_match
from rck.tokenizer import sentences, tokenize


@dataclass
class ConsciousAgent:
    """Top-level RCK agent with sharded KB + self-model + introspection."""

    dim: int = 4096
    n_shards: int = 64
    seed: int = 0
    install_self: bool = True
    # If set, overrides `n_shards` using the capacity-cliff recommendation
    # from `rck.shard_sizing` for the given expected fact count.
    expected_facts: int | None = None

    knowledge: ShardedKnowledgeBase = field(default=None, init=False)
    beliefs: ShardedKnowledgeBase = field(default=None, init=False)
    lm: RCKAgent = field(default=None, init=False)
    introspect_buf: IntrospectionBuffer = field(default=None, init=False)
    calibration: CalibrationTally = field(default=None, init=False)
    dialogue: DialogueContext = field(default=None, init=False)
    personality: Personality = field(default=None, init=False)
    actions: ActionRegistry = field(default=None, init=False)
    skills: SkillLibrary = field(default=None, init=False)
    provenance: ProvenanceStore = field(default=None, init=False)
    chain_cache: ChainCache = field(default=None, init=False)
    query_memory: QueryMemory = field(default=None, init=False)
    _last_query: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.expected_facts is not None:
            rec = recommend_shards(self.expected_facts, dim=self.dim)
            self.n_shards = rec.n_shards
        self.knowledge = ShardedKnowledgeBase(dim=self.dim, n_shards=self.n_shards, seed=self.seed)
        self.beliefs = make_belief_kb(dim=self.dim, n_shards=self.n_shards // 2 or 8, seed=self.seed + 1)
        self.lm = RCKAgent(
            vocab_size=8192, hv_dim=self.dim, n_columns=2, reservoir_dim=128,
            n_clauses=16, fep_rank=64, bigram_order=2, seed=self.seed + 2,
        )
        self.introspect_buf = IntrospectionBuffer(max_history=64)
        self.calibration = CalibrationTally()
        self.dialogue = DialogueContext(max_turns=16)
        self.personality = Personality()
        self.actions = make_default_registry()
        self.skills = SkillLibrary()
        self.provenance = ProvenanceStore()
        self.chain_cache = ChainCache(max_size=256)
        self.query_memory = QueryMemory(max_entries=4096)
        if self.install_self:
            install_self_model(self.knowledge)

    # ---- knowledge ingestion ----------------------------------------------

    def tell(self, subject: str, relation: str, obj: str) -> None:
        s = subject.lower(); r = relation.lower(); o = obj.lower()
        self.knowledge.store({"S": s, "R": r, "O": o})
        self.provenance.store(s, r, o, source="user")
        # Auto-symmetrize: also store the inverse so the user can query
        # in either direction.
        inv = inverse_relation(r)
        if inv and (inv != r or s != o):
            self.knowledge.store({"S": o, "R": inv, "O": s})
            self.provenance.store(o, inv, s, source="user",
                                   tags={"symmetrised"})

    def deny(self, subject: str, relation: str, obj: str) -> None:
        """Store a negative fact `(subject, NOT_relation, obj)`.
        Positive assertion that this triple does NOT hold."""
        _deny_fact(self.knowledge, subject, relation, obj,
                    provenance=self.provenance)

    def filter_negatives(self, subject: str, relation: str,
                          candidates) -> list:
        """Drop candidates that are explicitly denied by stored
        negative facts `(subject, NOT_relation, candidate)`."""
        return _filter_negatives(
            self.knowledge, subject, relation, candidates,
        )

    def drift_report(self, *, last_k: int = 200) -> dict:
        """Per-relation drift summary over the last `last_k` episodes."""
        if self.query_memory is None:
            return {"total_drift_events": 0, "by_signature": [],
                    "by_relation": {}}
        return self.query_memory.drift_report(last_k=last_k)

    def consolidate_episodes(self, *,
                              min_occurrences: int = 3,
                              stability_threshold: float = 0.9
                              ) -> ConsolidationReport:
        """Dreaming pass: walk query_memory and promote stable, frequent
        signatures into the chain cache; flag unstable and ambiguous
        signatures for the user."""
        if self.query_memory is None:
            return ConsolidationReport()
        def _discover(s: str, target: str):
            return self.discover(s, target, max_depth=3)
        return _consolidate_episodes(
            self.query_memory,
            min_occurrences=min_occurrences,
            stability_threshold=stability_threshold,
            kb=self.knowledge,
            chain_cache=self.chain_cache,
            discover_fn=_discover,
        )

    def shard_balance(self):
        """Per-shard load balancing report. Flags overloaded shards
        (above the capacity cliff) and suggests a reshard target."""
        from rck.shard_balance import report
        return report(self.knowledge)

    def concept_density(self, *, top_k: int = 10, stub_max: int = 1):
        """Per-subject fact-count histogram. Surfaces top entities and
        stubs (subjects with very few facts)."""
        from rck.concept_density import density_map
        return density_map(
            self.knowledge, top_k=top_k, stub_max=stub_max,
        )

    def relation_cooccurrence(self, *, min_co_occurrence: int = 2):
        """Pairs of relations that tend to appear together on the same
        subjects. Useful for chain-discovery priors + abstraction hints."""
        from rck.relation_cooccurrence import cooccurrence
        return cooccurrence(
            self.knowledge, min_co_occurrence=min_co_occurrence,
        )

    def similar_entities(self, subject: str, *,
                          top_k: int = 10,
                          min_overlap: int = 1):
        """Return entities most similar to `subject` by relation overlap
        (Jaccard similarity on their (R, O) attribute sets)."""
        from rck.entity_similarity import similar_entities
        return similar_entities(
            self.knowledge, subject,
            top_k=top_k, min_overlap=min_overlap,
        )

    def delta_replay(self, facts: list[tuple[str, str, str]],
                      *, max_rounds: int = 1) -> list[dict]:
        """Replay a list of candidate facts one at a time inside a
        counterfactual context. For each, run `what_changes` to
        capture what NEW derivations that ONE fact unlocks on top of
        the previous facts. Returns the per-fact summary list.

        The final state is rolled back -- nothing is permanently
        added to the agent.
        """
        results: list[dict] = []
        # Use a single counterfactual that accumulates facts; we add
        # one at a time and run cascade_induct after each.
        from rck.cascading_induction import cascade_induct
        cumulative: list[tuple[str, str, str]] = []
        with self.counterfactual([]):
            for fact in facts:
                pre_kb = self.knowledge.size()
                self.knowledge.store({
                    "S": fact[0].lower(),
                    "R": fact[1].lower(),
                    "O": fact[2].lower(),
                })
                if self.chain_cache is not None:
                    self.chain_cache.bump_version()
                cumulative.append(fact)
                res = cascade_induct(
                    self.knowledge, max_rounds=max_rounds,
                    probes_per_round=20,
                    policy=InductionPolicy(min_confidence=0.15),
                    skills=None, provenance=None,
                )
                results.append({
                    "fact": list(fact),
                    "kb_pre": pre_kb,
                    "kb_post": self.knowledge.size(),
                    "n_verified_inductions": res.total_verified,
                    "sample_derived": [
                        (f.subject, f.relation, f.obj)
                        for f in res.induced_facts[:5]
                    ],
                })
            # On exit, rollback. We added facts directly via kb.store,
            # so we must forget them manually.
            for s, r, o in cumulative:
                self.knowledge.forget({
                    "S": s.lower(), "R": r.lower(), "O": o.lower(),
                })
        if self.chain_cache is not None:
            self.chain_cache.bump_version()
        return results

    def abstract_facts(self, *, min_support: int = 3,
                        commit: bool = False) -> tuple[list, int]:
        """Find (parent, R, O) abstractions implied by sibling agreement.
        Returns (abstractions_list, n_committed). With commit=True, the
        new parent-level facts are stored with source="abstracted"."""
        from rck.hierarchical_abstraction import (
            commit_abstractions, find_abstractions,
        )
        found = find_abstractions(
            self.knowledge, min_support=min_support,
        )
        committed = 0
        if commit and found:
            committed = commit_abstractions(
                self.knowledge, found, provenance=self.provenance,
            )
            if committed > 0 and self.chain_cache is not None:
                self.chain_cache.bump_version()
        return found, committed

    def status_report(self) -> str:
        """Render `status()` as a multi-line human-readable report."""
        s = self.status()
        lines = ["RCK agent status:"]
        lines.append(f"  KB facts:           {s['kb_size']} "
                      f"(in {s['n_shards']} shards)")
        lines.append(f"  Provenance records: {s['provenance_records']}")
        if s["provenance_sources"]:
            srcs = ", ".join(f"{k}={v}"
                              for k, v in s["provenance_sources"].items())
            lines.append(f"  Sources:            {srcs}")
        if s["induced_via_hops"]:
            via = ", ".join(f"{k}h={v}"
                             for k, v in s["induced_via_hops"].items())
            lines.append(f"  Induced (hops):     {via}")
        sk = s["skills"]
        if sk:
            lines.append(f"  Skills:             n={sk.get('n', 0)}  "
                          f"uses={sk.get('total_uses', 0)}  "
                          f"high_conf={sk.get('high_conf', 0)}")
        rules = s["rules"]
        if rules.get("n_rules", 0):
            lines.append(f"  Rules:              {rules['n_rules']}")
            if rules.get("top_rule"):
                lines.append(f"    top: {rules['top_rule']}")
        qm = s["query_memory"]
        if qm.get("size", 0):
            lines.append(f"  Query memory:       {qm['size']} episodes")
            if qm.get("state_breakdown"):
                bd = ", ".join(f"{k}={v}"
                                for k, v in qm["state_breakdown"].items())
                lines.append(f"    states:           {bd}")
        cc = s["chain_cache"]
        if cc:
            lines.append(f"  Chain cache:        "
                          f"size={cc.get('size', 0)}  "
                          f"hits={cc.get('total_hits', 0)}  "
                          f"v={cc.get('kb_version', 0)}")
        cr = s.get("calibration_relations", [])
        if cr:
            lines.append(f"  Calibration:        {len(cr)} relations tracked")
        return "\n".join(lines)

    def what_if_user_says(self, text: str) -> dict:
        """Parse free-form text into (s, r, o) triples via Open IE,
        then preview the derivations they would unlock via
        `what_changes`. Returns the preview summary plus the
        extracted triples for inspection."""
        triples = extract_triples_from_text(text)
        if not triples:
            return {"extracted": [], "preview": None,
                    "note": "no triples extracted from text"}
        preview = self.what_changes(triples)
        preview["extracted"] = triples
        return preview

    def summarize_subject(self, subject: str, *, max_facts: int = 50):
        """Return a SubjectSummary covering everything the agent knows
        about `subject`: facts, denials, parents, siblings, source
        breakdown, query frequency."""
        from rck.subject_summary import summarize_subject
        return summarize_subject(
            self.knowledge, self.provenance, subject,
            query_memory=self.query_memory, max_facts=max_facts,
        )

    def what_changes(self, facts, *, max_rounds: int = 1) -> dict:
        """Preview what derivations a candidate fact set would enable.

        Adds `facts` (list of (s, r, o) tuples) inside a counterfactual
        context, runs cascade_induct, snapshots the new derived facts,
        then rolls back. Returns a summary dict.
        """
        with self.counterfactual(facts):
            pre_kb = self.knowledge.size()
            from rck.cascading_induction import cascade_induct
            res = cascade_induct(
                self.knowledge, max_rounds=max_rounds,
                probes_per_round=30,
                policy=InductionPolicy(min_confidence=0.15),
                skills=None,
                provenance=None,
            )
            verified = res.total_verified
            derived = [
                (f.subject, f.relation, f.obj)
                for f in res.induced_facts
            ]
            post_kb = self.knowledge.size()
        return {
            "candidate_facts": list(facts),
            "kb_pre": pre_kb,
            "kb_post": post_kb,
            "n_verified_inductions": verified,
            "sample_derived": derived[:10],
        }

    def prune_facts(self, *, min_confidence: float = 0.1,
                     prune_user_facts: bool = False,
                     prune_negative_facts: bool = False):
        """Drop low-confidence facts from the KB and provenance store."""
        from rck.fact_pruning import PruningPolicy, prune
        policy = PruningPolicy(
            min_confidence=min_confidence,
            prune_user_facts=prune_user_facts,
            prune_negative_facts=prune_negative_facts,
        )
        report = prune(self.knowledge, self.provenance, policy=policy)
        if report.dropped > 0 and self.chain_cache is not None:
            self.chain_cache.bump_version()
        return report

    def downstream_effects(self, cause: str, *,
                            max_depth: int = 3, beam_width: int = 5):
        """All entities reachable from `cause` via causes-chains."""
        from rck.causal import downstream_effects
        return downstream_effects(
            self.knowledge, cause,
            max_depth=max_depth, beam_width=beam_width,
        )

    def root_causes(self, effect: str, *,
                     max_depth: int = 3, beam_width: int = 5):
        """All entities that reach `effect` via causes-chains (backward)."""
        from rck.causal import root_causes
        return root_causes(
            self.knowledge, effect,
            max_depth=max_depth, beam_width=beam_width,
        )

    def rank_subjects(self, *, top_k: int = 20):
        """Return the top-K subjects by composite importance
        (fact count + object count + query count + derivation count)."""
        from rck.subject_importance import rank_subjects
        return rank_subjects(
            self.knowledge,
            provenance=self.provenance,
            query_memory=self.query_memory,
            top_k=top_k,
        )

    def diff_with(self, other):
        """Diff this agent against another. Returns an AgentDiffReport
        summarising facts/skills/rules unique to either side and shared."""
        from rck.agent_diff import diff_agents
        return diff_agents(self, other)

    def counterfactual(self, facts):
        """Context manager that temporarily adds `facts` (a list of
        (s, r, o) tuples) to the KB and rolls them back on exit.

        Usage:
            with agent.counterfactual([("dog", "isa", "fish")]):
                ...  # the KB thinks dogs are fish in this block
            # after the block, the temporary fact is gone
        """
        from rck.counterfactual import Counterfactual
        return Counterfactual(agent=self, facts=list(facts))

    def status(self) -> dict:
        """Comprehensive status dashboard. One call produces a summary
        of the agent's KB, derivation state, skills, rules, query memory,
        and cache. Useful for human-facing displays or cron logging."""
        kb_size = self.knowledge.size() if self.knowledge else 0
        sources: dict[str, int] = {}
        induced_via_hops: dict[int, int] = {}
        if self.provenance is not None:
            for rec in self.provenance._records.values():
                sources[rec.source] = sources.get(rec.source, 0) + 1
                if rec.source in {"induced", "rule"}:
                    for tag in rec.tags:
                        if tag.startswith("via_") and tag.endswith("_hops"):
                            try:
                                k = int(tag.split("_")[1])
                                induced_via_hops[k] = (
                                    induced_via_hops.get(k, 0) + 1
                                )
                                break
                            except (ValueError, IndexError):
                                pass
        rules_summary: dict = {"n_rules": 0, "top_rule": None}
        try:
            rs = self.extract_rules()
            rules_summary = {
                "n_rules": rs.size(),
                "top_rule": (rs.top_rules(n=1)[0].verbalize()
                              if rs.size() else None),
            }
        except Exception:
            pass
        return {
            "kb_size": kb_size,
            "n_shards": self.n_shards,
            "provenance_records": (
                self.provenance.size() if self.provenance else 0
            ),
            "provenance_sources": sources,
            "induced_via_hops": dict(sorted(induced_via_hops.items())),
            "skills": (self.skills.stats() if self.skills else {}),
            "rules": rules_summary,
            "query_memory": {
                "size": (self.query_memory.size()
                          if self.query_memory else 0),
                "state_breakdown": (
                    self.query_memory.state_breakdown()
                    if self.query_memory else {}
                ),
            },
            "chain_cache": (
                self.chain_cache.stats() if self.chain_cache else {}
            ),
            "calibration_relations": list(
                getattr(self.calibration, "by_relation", {}).keys()
            ) if self.calibration else [],
        }

    def merge_from(self, other) -> dict:
        """Fold another agent's auxiliary state into this one.

        Merges skills (counters add), provenance (counts add, source
        becomes "multi" on conflict), and the HRR KB (bipolar bundle
        sum). Does NOT merge query_memory, chain_cache, or calibration
        tally. Returns a MergeReport-as-dict for logging."""
        from rck.federated_merge import merge_agents
        report = merge_agents(self, other)
        return {
            "skills_added": report.skills_added,
            "skills_reinforced": report.skills_reinforced,
            "provenance_added": report.provenance_added,
            "provenance_reinforced": report.provenance_reinforced,
            "kb_facts_merged": report.kb_facts_merged,
        }

    def cluster_skills(self, *, prefix_length: int = 2,
                        min_members: int = 2) -> list[SkillFamily]:
        """Group the agent's SkillLibrary entries by shared prefix.
        Surfaces "families" of patterns sorted by total success."""
        return cluster_skills_by_prefix(
            self.skills,
            prefix_length=prefix_length, min_members=min_members,
        )

    def promote_skills(self, *,
                        promotion_policy: PromotionPolicy | None = None
                        ) -> tuple[RuleStore, list]:
        """Promote qualifying SkillFamilies to Rules.

        Returns a (RuleStore, new_rules) tuple. The store starts from
        extract_rules() so we don't lose anything; new_rules is just
        what was added by promotion this call.
        """
        store = self.extract_rules()
        new_rules = _promote_families(
            self.skills, store, policy=promotion_policy,
        )
        return store, new_rules

    def find_gaps(self, subject: str, *,
                   min_sibling_support: int = 2,
                   max_gaps: int = 10) -> list[Gap]:
        """For `subject`, find relations that peers (sibling isa entities)
        have facts for but the subject is missing. Useful for surfacing
        what to ask the user next about this entity."""
        return _find_gaps(
            self.knowledge, subject,
            min_sibling_support=min_sibling_support,
            max_gaps=max_gaps,
        )

    def propagate_negations(self) -> list[PropagatedNegation]:
        """Transitively propagate stored negative facts through lifting
        relations (isa, partof, ...). Returns the list of newly-stored
        negations along with skip/rejection records."""
        results = _propagate_negations(
            self.knowledge, provenance=self.provenance,
        )
        if any(r.stored for r in results) and self.chain_cache is not None:
            self.chain_cache.bump_version()
        return results

    def tell_belief(self, believer: str, subject: str, relation: str, obj: str) -> None:
        store_belief(self.beliefs, believer.lower(), subject.lower(),
                     relation.lower(), obj.lower())

    def load_jsonl(self, path: str | Path) -> int:
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.tell(rec["s"], rec["r"], rec["o"])
                n += 1
        # Bulk write may have created shorter chains; invalidate cache.
        if self.chain_cache is not None and n > 0:
            self.chain_cache.bump_version()
        return n

    def ingest_text(self, text: str) -> dict:
        """Ingest free-form text. Uses Open IE for richer extraction than
        the older _extract_simple_facts path."""
        n_facts = 0
        n_tokens = 0
        # Open IE pass -- broader coverage than the legacy regex set.
        for s, r, o in extract_triples_from_text(text):
            self.tell(s, r, o); n_facts += 1
        for sent in sentences(text):
            tokens = tokenize(sent)
            if tokens:
                self.lm.observe(tokens, learn=True)
                n_tokens += len(tokens)
        return {"facts": n_facts, "tokens": n_tokens}

    # ---- question answering ----------------------------------------------

    def correct(self, text: str) -> dict | None:
        """Try to interpret `text` as a user correction. If it matches, apply
        the change to the KB and return a confirmation."""
        result = detect_correction(text)
        if not result.matched:
            return None
        if result.forgot:
            s, r, o = result.forgot
            self.knowledge.forget({"S": s, "R": r, "O": o})
        if result.stored:
            s, r, o = result.stored
            self.tell(s, r, o)
        return {"verbal": result.summary, "stored": result.stored,
                "forgot": result.forgot, "source": "correction"}

    def ask(self, question: str, think_aloud: bool = False) -> dict:
        self._last_query = question

        # Correction short-circuit: "Actually, X is Y" updates the KB.
        corr = self.correct(question)
        if corr is not None:
            return corr

        # Tool/action short-circuit: try registered tools first.
        match = self.actions.match(question)
        if match is not None:
            tool, groups = match
            try:
                if groups:
                    result = self.actions.invoke(tool.name, **groups)
                else:
                    # No regex groups -- pass the question for tools that parse it.
                    result = self.actions.invoke(tool.name, question)
            except TypeError:
                # Tool doesn't take arguments.
                result = self.actions.invoke(tool.name)
            verbal = result.get("verbal") if isinstance(result, dict) else str(result)
            return {**(result if isinstance(result, dict) else {"result": result}),
                    "verbal": verbal or str(result),
                    "source": f"tool-{tool.name}", "category": "know"}

        # Arithmetic short-circuit.
        arith = evaluate_arithmetic(question)
        if arith is not None:
            return {**arith, "source": "arithmetic", "category": "know"}

        # Temporal short-circuit.
        tres = temporal_answer(question)
        if tres is not None:
            return {**tres, "source": "temporal", "category": "know",
                    "confidence": 1.0}

        # Multi-step decomposition.
        ms = two_step(self.knowledge, question)
        if ms is not None:
            return {**ms, "source": "multistep", "category": "know"}

        # Dialogue: resolve references + inherit topic.
        question_rewritten = self.dialogue.with_default_topic(question)
        question_rewritten = self.dialogue.resolve_references(question_rewritten)
        # Question-type dispatch.
        qtype = _classify_question(question_rewritten)
        if qtype == "boolean":
            return self._ask_boolean(question_rewritten)
        if qtype == "enumerate":
            return self._ask_enumeration(question_rewritten)
        if qtype == "compare":
            return self._ask_comparison(question_rewritten)
        parsed = parse_question(question_rewritten)
        if parsed is None:
            return self._lm_fallback(question_rewritten)
        entity, relation, _ = parsed
        # Apply synonym normalization.
        entity = canonical_entity(entity)
        relation = canonical_relation(relation)
        # Multi-relation cascade.
        cascade = [relation]
        if relation != "is":
            cascade.append("is")
        # Common-sense relation aliases.
        ALIASES = {
            "color": ["color"],
            "have": ["has"],
            "has": ["has"],
            "lives_in": ["lives_in", "locatedin"],
            "capital": ["capital"],
            "kind": ["isa", "kind"],
            "type": ["isa", "kind"],
            "category": ["category", "isa"],
            "is": ["is", "isa"],
            "causes": ["causes"],
            "madeof": ["madeof"],
            "usedfor": ["usedfor"],
            "field": ["field"],
            "wrote": ["wrote", "author"],
            "author": ["author", "wrote"],
            "value": ["value"],
            "version": ["version"],
            "use": ["uses"],
        }
        for alias in ALIASES.get(relation, []):
            if alias not in cascade:
                cascade.append(alias)
        # Generic inverse-relation fallback: "who composed Symphony" -> the
        # fact was stored as (mozart, composed, symphony) and auto-symmetrised
        # to (symphony, composer, mozart). Add the inverse to the cascade so
        # either direction of query works.
        inv = inverse_relation(relation)
        if inv and inv not in cascade:
            cascade.append(inv)

        # Try relations IN ORDER, accept the first that passes the
        # confident-enough bar. The `is` fallback is ONLY accepted if its
        # confidence is strong enough; otherwise we'd let self-model noise
        # answer "what color is the alien" with random fragments.
        STRONG_BAR = 0.10
        WEAK_BAR_SPECIFIC = 0.10   # weak hits below this are noise, not knowledge
        STRONG_BAR_IS = 0.15       # is-fallback needs higher confidence
        chosen = None
        weakest = None
        for rel in cascade:
            results = self.knowledge.query({"S": entity, "R": rel}, "O", top_k=3)
            if not results:
                continue
            score = float(results[0][1])
            top_sym = str(results[0][0])
            # 'is' fallback uses a stricter threshold.
            bar = STRONG_BAR_IS if rel == "is" and rel != relation else STRONG_BAR
            if score > bar:
                chosen = (top_sym, score, rel, results)
                break
            if rel == relation and score >= WEAK_BAR_SPECIFIC and weakest is None:
                weakest = (top_sym, score, rel, results)
        best = chosen or weakest
        if best is not None:
            answer, conf, used_rel, results = best
            cat = epistemic_category(conf)
            verbal_sentence = render(entity, used_rel, answer)
            res = {
                "answer": answer,
                "confidence": conf,
                "category": cat,
                "verbal": verbalize(answer, conf,
                                    source=("structured" if used_rel == relation
                                            else f"structured-via-{used_rel}")),
                "sentence": verbal_sentence,
                "source": "structured" if used_rel == relation else f"structured-via-{used_rel}",
                "candidates": [(str(s), float(c)) for s, c in results],
                "parsed": {"entity": entity, "relation": relation, "used_relation": used_rel},
            }
            self.dialogue.record(question, res["parsed"], answer)
            return res
        # No direct match -- try multi-hop inference.
        chain_res = infer(self.knowledge, entity, relation)
        if chain_res.answer is not None and chain_res.confidence > 0.05:
            cat = epistemic_category(chain_res.confidence)
            sentence = render(entity, relation, chain_res.answer)
            chain_text = render_chain(chain_res.chain)
            res = {
                "answer": chain_res.answer,
                "confidence": chain_res.confidence,
                "category": cat,
                "verbal": f"{sentence}",
                "sentence": sentence,
                "reasoning": chain_text,
                "source": f"inferred-{chain_res.source}",
                "candidates": [],
                "parsed": {"entity": entity, "relation": relation, "used_relation": relation},
            }
            if think_aloud:
                res["think_aloud"] = narrate(question, chain_res)
            self.dialogue.record(question, res["parsed"], chain_res.answer)
            return res
        fb = self._lm_fallback(question)
        if think_aloud:
            fb["think_aloud"] = narrate_no_match(question, cascade)
        self.dialogue.record(question, None, None)
        return fb

    # ---- multi-sentence description --------------------------------------

    def describe(self, entity: str, max_sentences: int = 6) -> str:
        """Return a multi-sentence description of `entity` from stored facts."""
        return compose_describe(self.knowledge,
                                canonical_entity(entity),
                                max_sentences=max_sentences)

    def think_aloud_about(self, question: str) -> str:
        """Force think-aloud narration for a query."""
        res = self.ask(question, think_aloud=True)
        if "think_aloud" in res:
            return res["think_aloud"]
        return res.get("verbal", "I don't know.")

    # ---- broader question types -------------------------------------------

    def _ask_boolean(self, question: str) -> dict:
        """Handle 'is X a Y?', 'does X have Y?' etc."""
        parsed = _parse_boolean(question)
        if parsed is None:
            return self._lm_fallback(question)
        s, r, o = parsed
        res = kb_boolean(self.knowledge, s, r, o)
        if res["answer"] is True:
            verbal = f"Yes, {render(s, r, o)}"
        elif res["answer"] is False:
            verbal = f"No, {render(s, r, o).rstrip('.')} -- I have evidence to the contrary."
        else:
            verbal = "I don't know."
        out = {
            "answer": res["answer"],
            "confidence": res["confidence"],
            "category": "boolean",
            "verbal": verbal,
            "source": f"boolean-{res['source']}",
            "chain": res.get("chain", []),
            "parsed": {"entity": s, "relation": r, "value": o},
        }
        self.dialogue.record(question, out["parsed"], str(res["answer"]))
        return out

    def _ask_enumeration(self, question: str) -> dict:
        """Handle 'list all X' / 'what are the X'."""
        parsed = _parse_enumeration(question)
        if parsed is None:
            return self._lm_fallback(question)
        relation, value = parsed
        # Try the parsed relation first, then common aliases so 'fruits'
        # works even though the underlying relation is `category`, not `isa`.
        relation_cascade = [relation]
        if relation == "isa":
            relation_cascade.append("category")
        all_items: list[tuple[str, float]] = []
        used_rel = relation
        for rel in relation_cascade:
            items = enumerate_subjects(self.knowledge, rel, value, top_k=12)
            if items:
                all_items = items; used_rel = rel; break
        names = [s for s, _ in all_items]
        verbal = render_enumeration(names, used_rel, value)
        return {
            "answer": names,
            "confidence": (max((c for _, c in all_items), default=0.0)),
            "category": "enumeration",
            "verbal": verbal,
            "source": "enumeration",
            "candidates": [(str(s), float(c)) for s, c in all_items],
            "parsed": {"relation": used_rel, "value": value},
        }

    def _ask_comparison(self, question: str) -> dict:
        parsed = _parse_comparison(question)
        if parsed is None:
            return self._lm_fallback(question)
        a, b, dim = parsed
        res = kb_compare(self.knowledge, a, b, dimension=dim)
        return {
            "answer": res["winner"],
            "confidence": 1.0 if res["winner"] else 0.0,
            "category": "comparison",
            "verbal": res["verbal"],
            "source": "comparison",
            "values": res["values"],
            "parsed": {"a": a, "b": b, "dimension": dim},
        }

    def _lm_fallback(self, question: str) -> dict:
        tokens = tokenize(question)
        emitted, _ = self.lm.generate(tokens, max_new=12) if tokens else ([], [])
        text = " ".join(str(t) for t in emitted)
        return {
            "answer": text,
            "confidence": 0.0,
            "category": "unknown",
            "verbal": "I don't know.",
            "source": "generated",
            "candidates": [],
            "parsed": None,
        }

    def what_does_x_think(self, believer: str, subject: str, relation: str) -> dict:
        results = what_does_x_think(self.beliefs, believer.lower(),
                                    subject.lower(), relation.lower())
        if not results or results[0][1] < 0.05:
            return {"answer": None, "verbal": f"I have no record of {believer}'s beliefs about {subject}.",
                    "candidates": []}
        ans = str(results[0][0])
        truth, _ = self.knowledge.answer({"S": subject.lower(), "R": relation.lower()}, "O")
        return {
            "answer": ans,
            "confidence": float(results[0][1]),
            "verbal": f"{believer.capitalize()} believes that the {relation} of {subject} is {ans}.",
            "ground_truth": truth,
            "matches_truth": (ans == truth),
            "candidates": [(str(s), float(c)) for s, c in results],
        }

    # ---- self awareness ---------------------------------------------------

    def who_am_i(self) -> str:
        return self_describe(self.knowledge)

    def think(self) -> str:
        return think(self.lm, self.introspect_buf, last_query=self._last_query)

    # ---- verification ----------------------------------------------------

    def verify_fact(self, subject: str, relation: str, obj: str,
                    *, methods: list[str] | None = None) -> dict:
        """Run self-verification on a (S, R, O) triple. See rck.self_verify."""
        return _verify_kb(self.knowledge, subject, relation, obj, methods=methods)

    def maintain(self, *, max_rounds: int = 2,
                  probes_per_round: int = 80,
                  warm_cache_top_k: int = 10,
                  resolve_conflicts: bool = True,
                  propagate_negations: bool = True,
                  promote_skills: bool = True,
                  consolidate_episodes: bool = True,
                  checkpoint_dir: str | Path | None = None) -> dict:
        """Run a single maintenance pass that exercises the v14 derivation
        stack end-to-end.

        Steps (in order):
          1. cascade_induct          -- chain-driven derivation
          2. cascade_instantiate     -- rule-driven derivation
          3. propagate_negations     -- lift NOT_R through isa/partof
          4. resolve_conflicts       -- drop losing facts from collisions
          5. promote_skills          -- skill families -> Rules
          6. consolidate_episodes    -- stable hot signatures -> cache
          7. warm_cache_from_history -- predictive prefetch
          8. checkpoint              -- optional save_state to disk

        Returns a summary dict suitable for logging in a cron job.
        Every flag defaults True; the cron job can flip them off for
        a lighter pass.
        """
        from rck.cascading_induction import cascade_induct
        summary: dict = {}

        casc = cascade_induct(
            self.knowledge, max_rounds=max_rounds,
            probes_per_round=probes_per_round,
            policy=InductionPolicy(min_confidence=0.15),
            skills=self.skills,
            provenance=self.provenance,
        )
        summary["chain_induction_verified"] = casc.total_verified
        summary["chain_induction_rounds"] = len(casc.rounds)

        rc = self.cascade_instantiate_rules(max_rounds=max_rounds)
        summary["rule_cascade_verified"] = rc.total_verified()
        summary["rule_cascade_rounds"] = len(rc.rounds)

        if propagate_negations:
            neg_results = self.propagate_negations()
            summary["negations_propagated"] = sum(
                1 for r in neg_results if r.stored
            )
        else:
            summary["negations_propagated"] = 0

        if resolve_conflicts:
            plans = self.resolve_conflicts(apply=True)
            summary["conflicts_resolved"] = len(plans)
        else:
            summary["conflicts_resolved"] = 0

        if promote_skills:
            _store, new_rules = self.promote_skills()
            summary["skills_promoted"] = len(new_rules)
        else:
            summary["skills_promoted"] = 0

        if consolidate_episodes and self.query_memory is not None:
            cr = self.consolidate_episodes()
            summary["episodes_consolidated"] = cr.total_promoted()
            summary["episodes_ambiguous_flagged"] = len(cr.ambiguous_flagged)
        else:
            summary["episodes_consolidated"] = 0
            summary["episodes_ambiguous_flagged"] = 0

        warmed = self.warm_cache_from_history(top_k=warm_cache_top_k)
        summary["cache_entries_warmed"] = warmed

        summary["final_kb_size"] = self.knowledge.size()
        summary["skill_library"] = self.skills.stats()

        if checkpoint_dir is not None:
            summary["checkpoint"] = self.save_state(checkpoint_dir)

        return summary

    def record_truth(self, known: dict, unknown_role: str,
                     correct_answer: str) -> dict:
        """Tell the agent the ground-truth answer for a previously
        asked query. Walks `query_memory` to find the latest episode
        matching the signature, then updates `calibration` with
        whether the system's predicted top_symbol was correct.

        Returns a dict summary of the update for logging.
        """
        if self.query_memory is None:
            return {"updated": False, "reason": "no query_memory"}
        history = self.query_memory.transitions_for_signature(
            known, unknown_role,
        )
        if not history:
            return {"updated": False, "reason": "no prior episode"}
        # Most recent episode for this signature.
        _ts, state, predicted = history[-1]
        # Pull the matching QueryEpisode to access top_score.
        target_sig = (tuple(sorted(known.items())), unknown_role)
        episode = None
        for e in reversed(self.query_memory.all()):
            sig = (tuple(sorted(e.known.items())), e.unknown_role)
            if sig == target_sig:
                episode = e
                break
        if episode is None:
            return {"updated": False, "reason": "no matching episode"}
        relation = str(known.get("R", "")).lower()
        correct = (str(predicted).lower() == str(correct_answer).lower()
                    if predicted is not None else False)
        self.calibration.record(relation, episode.top_score, correct)
        return {
            "updated": True,
            "relation": relation,
            "predicted": str(predicted) if predicted is not None else None,
            "correct_answer": str(correct_answer),
            "was_correct": correct,
            "confidence": episode.top_score,
        }

    def save_state(self, dir_path: str | Path) -> dict:
        """Persist the agent's auxiliary state (skills, provenance,
        query_memory) into a directory. Each component is written
        to a dedicated JSONL file. Returns counts per component.

        Does NOT persist the HRR knowledge base or LM -- those use
        separate persistence in rck.persist.
        """
        from pathlib import Path as _P
        d = _P(dir_path); d.mkdir(parents=True, exist_ok=True)
        return {
            "skills": self.save_skills(d / "skills.jsonl"),
            "provenance": self.save_provenance(d / "provenance.jsonl"),
            "query_memory": self.save_query_memory(d / "query_memory.jsonl"),
        }

    def load_state(self, dir_path: str | Path,
                    *, replace: bool = False) -> dict:
        """Restore agent auxiliary state from a directory previously
        written by save_state. Returns counts per component."""
        from pathlib import Path as _P
        d = _P(dir_path)
        return {
            "skills": self.load_skills(d / "skills.jsonl", replace=replace),
            "provenance": self.load_provenance(d / "provenance.jsonl",
                                                 replace=replace),
            "query_memory": self.load_query_memory(d / "query_memory.jsonl",
                                                    replace=replace),
        }

    def save_provenance(self, path: str | Path) -> int:
        """Persist the ProvenanceStore to JSONL."""
        if self.provenance is None:
            return 0
        return self.provenance.save(path)

    def load_provenance(self, path: str | Path,
                         *, replace: bool = False) -> int:
        """Restore the ProvenanceStore from JSONL."""
        if self.provenance is None:
            return 0
        return self.provenance.load(path, replace=replace)

    def save_skills(self, path: str | Path) -> int:
        """Persist the SkillLibrary to JSONL."""
        if self.skills is None:
            return 0
        return self.skills.save(path)

    def load_skills(self, path: str | Path, *, replace: bool = False) -> int:
        """Restore the SkillLibrary from JSONL."""
        if self.skills is None:
            return 0
        return self.skills.load(path, replace=replace)

    def save_query_memory(self, path: str | Path) -> int:
        """Persist the episodic query log to a JSONL file."""
        if self.query_memory is None:
            return 0
        return self.query_memory.save(path)

    def load_query_memory(self, path: str | Path,
                           *, replace: bool = False) -> int:
        """Re-hydrate the episodic query log from a JSONL file
        previously written by save_query_memory."""
        if self.query_memory is None:
            return 0
        return self.query_memory.load(path, replace=replace)

    def warm_cache_from_history(self, *, top_k: int = 10,
                                 max_depth: int = 3) -> int:
        """Pre-warm the chain cache for the most frequent recent
        queries. Uses query_memory.hot_signatures to pick (start,
        target) pairs whose query signature includes a known object,
        then runs discover_chains on each. Returns the number of
        chains added to the cache.

        For now we look at queries whose known dict contains both
        S and R, with O as the unknown. The "target" for warming is
        the historical top_symbol (the answer we returned). This
        lets us serve future repeats from cache.
        """
        if self.query_memory is None or self.chain_cache is None:
            return 0
        warmed = 0
        hot = self.query_memory.hot_signatures(top_k=top_k)
        for sig, _count in hot:
            known_pairs, unknown_role = sig
            known = dict(known_pairs)
            s = str(known.get("S", "")).lower()
            # Find a target by looking at the most recent successful
            # episode for this signature.
            history = self.query_memory.transitions_for_signature(
                known, unknown_role,
            )
            target = None
            for _ts, state, sym in reversed(history):
                if state == "known" and sym is not None:
                    target = str(sym).lower()
                    break
            if not s or not target:
                continue
            if self.chain_cache.get(s, target) is not None:
                continue  # already cached
            spec = self.discover(s, target, max_depth=max_depth)
            if spec is not None:
                warmed += 1
        return warmed

    def explain_why(self, subject: str, relation: str, obj: str,
                    *, max_depth: int = 5) -> ExplanationNode:
        """Recursively expand the provenance graph for `(s, r, o)`.
        Leaves are user-asserted facts; intermediate nodes are induced
        or rule-derived. Cycles are broken at the first repetition."""
        return _explain_why(self.provenance, subject, relation, obj,
                             max_depth=max_depth)

    def detect_conflicts(self, *, subjects: list[str] | None = None,
                          policy: ContradictionPolicy | None = None
                          ) -> list[Conflict]:
        """Scan the KB for contradictions on functional relations.
        Returns Conflict objects sorted by severity."""
        return detect_conflicts(self.knowledge, subjects=subjects,
                                 policy=policy)

    def resolve_conflicts(self, *, subjects: list[str] | None = None,
                           policy: ContradictionPolicy | None = None,
                           apply: bool = False
                           ) -> list[ResolutionPlan]:
        """Detect conflicts, then plan resolution by source priority +
        recency + count. If `apply=True`, drop losing facts from the KB.
        Bumps the chain cache version when facts are removed."""
        conflicts = detect_conflicts(self.knowledge, subjects=subjects,
                                      policy=policy)
        plans = resolve_all(self.knowledge, conflicts,
                             provenance=self.provenance, apply=apply)
        if apply and plans and self.chain_cache is not None:
            self.chain_cache.bump_version()
        return plans

    # ---- chain reasoning -------------------------------------------------

    def reason(self, start: str, relations: list[str],
               *, directions: list[str] | None = None,
               propagation: PropagationConfig | None = None) -> dict:
        """Walk an n-hop chain from `start` along `relations`.

        Returns a dict with the chain trace, answer, and confidence.
        Successful chains register a pattern in `self.skills`.
        Use this for compositional multi-hop queries that don't fit
        the natural-language templates handled by `multistep.two_step`.
        """
        dirs = directions or ["forward"] * len(relations)
        if len(dirs) != len(relations):
            raise ValueError("directions length must match relations")
        hops = [Hop(relation=r, direction=d) for r, d in zip(relations, dirs)]
        result = walk_chain(self.knowledge, start, hops,
                            config=propagation, skills=self.skills)
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "hedge": result.hedge,
            "trace": result.trace,
            "aborted_at": result.aborted_at,
            "verbal": result.verbalize(),
        }

    def intersect(self, constraints: list[dict], unknown_role: str,
                  *, top_k: int = 20, min_score: float = 0.10
                  ) -> list[SetCandidate]:
        """Set intersection across multiple KB constraints. Returns
        candidates that satisfy ALL constraints, ranked by geometric
        mean of per-constraint scores."""
        return intersect_queries(
            self.knowledge, constraints, unknown_role,
            top_k=top_k, min_score=min_score,
        )

    def union(self, constraints: list[dict], unknown_role: str,
              *, top_k: int = 20, min_score: float = 0.10
              ) -> list[SetCandidate]:
        """Set union across multiple KB constraints."""
        return union_queries(
            self.knowledge, constraints, unknown_role,
            top_k=top_k, min_score=min_score,
        )

    def ask_with_idk(self, known: dict, unknown_role: str,
                     *, policy: IDKPolicy | None = None) -> EpistemicAnswer:
        """Retrieve with explicit IDK detection. Returns an
        EpistemicAnswer that distinguishes KNOWN / AMBIGUOUS / IDK
        rather than always picking a top-1 answer.

        Compares against `query_memory` history for this signature
        and surfaces `drift_from_prior` if state or answer changed
        since the last identical query. Logs the new episode for
        future calibration / drift detection."""
        res = ask_with_idk(self.knowledge, known, unknown_role,
                            policy=policy)
        if self.query_memory is not None:
            history = self.query_memory.transitions_for_signature(
                known, unknown_role,
            )
            if history:
                prev_ts, prev_state, prev_sym = history[-1]
                if prev_state != res.state.value or prev_sym != res.top_symbol:
                    res.drift_from_prior = (
                        f"prev state={prev_state!r} sym={prev_sym!r}; "
                        f"now state={res.state.value!r} sym={res.top_symbol!r}"
                    )
            self.query_memory.record(
                known, unknown_role,
                state=res.state.value,
                top_symbol=res.top_symbol,
                top_score=res.top_score,
                notes=res.drift_from_prior or "",
            )
        return res

    def calibrated_ask(self, known: dict, unknown_role: str,
                       *, top_k: int = 3) -> list[CalibratedAnswer]:
        """Retrieve facts with provenance-aware confidence discounting.
        Induced facts are downweighted by chain depth; user-provided
        facts keep their full HRR score."""
        return calibrated_lookup(
            self.knowledge, self.provenance,
            known, unknown_role, top_k=top_k,
        )

    def analogy(self, a: str, b: str, c: str) -> AnalogyResult:
        """Solve `a : b :: c : ?`. Returns the inferred relation and
        the analog answer if both are findable in the KB."""
        return solve_analogy(self.knowledge, a, b, c)

    def extract_rules(self, *, min_support: int = 2,
                      min_confidence: float = 0.5) -> RuleStore:
        """Extract symbolic rules from the agent's skill library.
        Rules are universal implications instantiated by repeated
        chain patterns."""
        return extract_rules(
            self.skills,
            min_support=min_support, min_confidence=min_confidence,
        )

    def rule_effectiveness_report(self, *, top_k: int = 20) -> list[dict]:
        """Per-rule effectiveness summary: support, uses, successes,
        failures, confidence. Ranked by uses then confidence."""
        store = self.extract_rules()
        return store.effectiveness_report(top_k=top_k)

    def canonicalize(self, text: str, *, kind: str = "entity") -> str:
        """Normalise a user-supplied string to the canonical KB form.
        `kind` is 'entity' (default) or 'relation'."""
        from rck.synonyms import canonical_entity, canonical_relation
        if kind == "relation":
            return canonical_relation(text)
        return canonical_entity(text)

    def source_calibration(self) -> dict:
        """Per-source distribution of provenance records: per source,
        the count, average confidence, and average reinforcement count.
        Useful for deciding whether SOURCE_PRIORITY weights need
        adjustment."""
        if self.provenance is None:
            return {}
        from collections import defaultdict
        agg: dict[str, dict] = defaultdict(lambda: {
            "count": 0, "sum_confidence": 0.0, "sum_reinforce": 0,
        })
        for rec in self.provenance._records.values():
            r = agg[rec.source]
            r["count"] += 1
            r["sum_confidence"] += float(rec.confidence)
            r["sum_reinforce"] += int(rec.count)
        out: dict[str, dict] = {}
        for source, r in agg.items():
            n = r["count"]
            out[source] = {
                "count": n,
                "avg_confidence": r["sum_confidence"] / max(1, n),
                "avg_reinforce": r["sum_reinforce"] / max(1, n),
            }
        return out

    def instantiate_rules(self, *, min_support: int = 2,
                          min_confidence: float = 0.5
                          ) -> list[InstantiatedFact]:
        """Forward-chain: extract rules from skills then apply each
        one to the KB to derive new direct facts. Pairs with
        chain_induction / cascading_induction: rule instantiation is
        the BACKWARD direction (KB-driven) and runs faster on
        productive patterns."""
        store = self.extract_rules(
            min_support=min_support, min_confidence=min_confidence,
        )
        facts = instantiate_all(
            self.knowledge, store, provenance=self.provenance,
        )
        if facts and self.chain_cache is not None:
            self.chain_cache.bump_version()
        return facts

    def compose_rules(self, *, min_support: int = 2,
                      min_confidence: float = 0.5,
                      max_body_length: int = 5) -> list[Rule]:
        """Extract rules from the skill library, then enumerate all
        valid pairwise compositions. Returns the newly composed rules
        (also added to the underlying RuleStore)."""
        store = self.extract_rules(
            min_support=min_support, min_confidence=min_confidence,
        )
        return compose_all(store, max_body_length=max_body_length)

    def cascade_instantiate_rules(self, *, min_support: int = 2,
                                   min_confidence: float = 0.5,
                                   max_rounds: int = 4
                                   ) -> RuleCascadeResult:
        """Extract rules, then apply them in a loop until saturation.
        Pairs with cascade_induct (chain-driven) and instantiate_rules
        (single pass)."""
        store = self.extract_rules(
            min_support=min_support, min_confidence=min_confidence,
        )
        res = cascade_instantiate(
            self.knowledge, store, max_rounds=max_rounds,
            provenance=self.provenance,
        )
        if res.total_verified() > 0 and self.chain_cache is not None:
            self.chain_cache.bump_version()
        return res

    def induce(self, start: str, target: str, *,
               max_depth: int = 4,
               policy: InductionPolicy | None = None
               ) -> InducedFact | None:
        """Discover a chain to `target`, walk it, and induce the
        shortcut fact if it passes the policy and self-verification.

        Successful inductions also register a skill pattern.
        """
        spec = self.discover(start, target, max_depth=max_depth)
        if spec is None:
            return None
        hops = [Hop(r, d) for r, d in zip(spec["relations"], spec["directions"])]
        result = walk_chain(self.knowledge, start, hops,
                            config=None, skills=self.skills)
        if result.answer is None:
            return None
        induced = induce_from_chain(
            self.knowledge, result, start,
            policy=policy, skills=self.skills,
            provenance=self.provenance,
        )
        # If a verified shortcut was added, the cache may have a longer
        # chain for the same (start, target). Bump version so the next
        # discover() re-runs BFS and finds the new shortcut.
        if (induced is not None and induced.verified
                and self.chain_cache is not None):
            self.chain_cache.bump_version()
        return induced

    def discover(self, start: str, target: str, *,
                 max_depth: int = 4, beam_width: int = 4,
                 min_link_score: float = 0.10,
                 allow_reverse: bool = False,
                 use_cache: bool = True) -> dict | None:
        """Discover a chain from `start` to `target` via BFS over the KB.

        If found, returns the chain spec ready to be passed back to
        `reason()`. Returns None if no chain exists within max_depth.
        Caches the result by (start, target) for fast repeat lookups.
        """
        # Cache short-circuit.
        if use_cache:
            cached = self.chain_cache.get(start, target)
            if cached is not None:
                return {
                    "relations": list(cached.relations),
                    "directions": list(cached.directions),
                    "confidence": cached.discovery_confidence,
                    "trace": [],
                    "from_cache": True,
                }
        # Use the skill library as a prior once it has measurable data.
        prior = self.skills if self.skills.stats()["n"] >= 3 else None
        chains = discover_chains(
            self.knowledge, start, Goal.symbol(target),
            max_depth=max_depth, beam_width=beam_width,
            min_link_score=min_link_score, top_n=1,
            allow_reverse=allow_reverse,
            skills_prior=prior,
        )
        if not chains:
            return None
        ch = chains[0]
        if use_cache:
            self.chain_cache.put(start, target, ch.relations,
                                  ch.directions, ch.confidence)
        return {
            "relations": ch.relations,
            "directions": ch.directions,
            "confidence": ch.confidence,
            "trace": ch.trace,
            "from_cache": False,
        }

    # ---- state -----------------------------------------------------------

    def state(self) -> dict:
        return {
            "version": "15.0.0",
            "facts": self.knowledge.size(),
            "beliefs": self.beliefs.size(),
            "n_shards": self.n_shards,
            "lm_codebook": self.lm.codebook.size(),
            "calibration": self.calibration.summary(),
            "dialogue_turns": len(self.dialogue.history),
            "skills": self.skills.stats(),
        }


# ---------------------------------------------------------------------------
#  Question-type classifiers
# ---------------------------------------------------------------------------

def _classify_question(q: str) -> str:
    """Return 'boolean', 'enumerate', 'compare', or 'factual'."""
    toks = tokenize(q)
    if not toks:
        return "factual"
    # Comparison FIRST -- 'is X bigger than Y' begins with 'is' too.
    if "than" in toks or any(w in toks for w in (
        "bigger", "smaller", "larger", "younger", "older", "taller", "shorter"
    )):
        return "compare"
    first = toks[0]
    if first in {"is", "are", "does", "do", "was", "were", "has", "have", "can"}:
        return "boolean"
    if (first in {"list", "name"}
        or " ".join(toks[:2]) in {"what are", "which are"}
        or (first == "what" and "are" in toks[:5])):
        return "enumerate"
    return "factual"


def _parse_boolean(q: str) -> tuple[str, str, str] | None:
    """Extract (S, R, O) for boolean queries like 'is X a Y' or 'does X have Y'."""
    toks = [t for t in tokenize(q) if t not in {"?", "."}]
    if not toks:
        return None
    head = toks[0]
    if head in {"is", "are", "was", "were"}:
        # "is X a Y" / "is X Y"
        body = [t for t in toks[1:] if t not in {"a", "an", "the"}]
        if len(body) >= 2:
            return (body[0], "isa", body[-1])
    if head in {"does", "do"}:
        # "does X have Y" / "does X V Y"
        body = [t for t in toks[1:] if t not in {"a", "an", "the"}]
        if len(body) >= 3 and body[1] in {"have", "has"}:
            subject = body[0]
            obj = body[2] if len(body) > 2 else None
            return (subject, "has", obj)
        if len(body) >= 3:
            return (body[0], body[1], body[2])
    if head in {"has", "have"}:
        body = [t for t in toks[1:] if t not in {"a", "an", "the"}]
        if len(body) >= 2:
            return (body[0], "has", body[1])
    return None


def _parse_enumeration(q: str) -> tuple[str, str] | None:
    """Extract (relation, value) for 'what X are Y' / 'list all X that are Y'."""
    toks = [t for t in tokenize(q) if t not in {"?", "."}]
    if not toks:
        return None
    # "list all X that are Y" / "list X"
    if toks[0] in {"list", "name"}:
        words = [t for t in toks[1:] if t not in {"all", "the"}]
        if words:
            return ("isa", _singularize(words[0]))
    # "what animals are mammals" / "which countries are in europe"
    if toks[:2] in (["what", "are"], ["which", "are"]):
        rest = [t for t in toks[2:] if t not in {"a", "an", "the"}]
        if rest:
            return ("isa", _singularize(rest[-1]))
    if toks[0] in {"what", "which"} and "are" in toks:
        idx = toks.index("are")
        rest = [t for t in toks[idx + 1:] if t not in {"a", "an", "the"}]
        if rest:
            return ("isa", _singularize(rest[-1]))
    return None


def _singularize(word: str) -> str:
    """Crude English singularizer: cars -> car, mammals -> mammal, etc."""
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _parse_comparison(q: str) -> tuple[str, str, str] | None:
    """Extract (A, B, dimension) for 'is X bigger than Y' etc."""
    toks = [t for t in tokenize(q) if t not in {"?", "."}]
    if "than" not in toks:
        return None
    SIZE_WORDS = {"bigger", "larger", "huger", "taller"}
    DIM = "size"
    has_size_cmp = any(w in toks for w in SIZE_WORDS)
    if not has_size_cmp:
        return None
    than_idx = toks.index("than")
    before = [t for t in toks[:than_idx] if t not in
              {"is", "are", "a", "an", "the"} | SIZE_WORDS]
    after = [t for t in toks[than_idx + 1:] if t not in
             {"a", "an", "the"}]
    if not before or not after:
        return None
    return (before[-1], after[-1], DIM)
