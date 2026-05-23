# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Minimal RAG: query KB -> prepend retrieved facts -> sample HYMN.

The "blend HYMN + KB at generation time" is the Phase-6 design target
that the full EFE 7-source decoder is supposed to do. This module
ships a stepping-stone primitive that doesn't need any of that
plumbing: for each input question, pull KB facts that mention
candidate subjects, format them as 'Context: ...' lines, and prepend
them to the prompt before calling the HYMN sampler.

Trained-on-Q/A HYMN + this primitive = a chat surface where every
generated answer is conditioned on the KB facts it can actually see.

Usage:
    from rain.cognition.rag import KbAugmentedSampler
    sampler = KbAugmentedSampler(kb, base_sampler, max_facts=5)
    answer = sampler.sample("what does a lion eat", n_tokens=80)
"""

from __future__ import annotations

from collections.abc import Callable

from rain.core.knowledge_base import ShardedKB

# Same signature as the HymnSamplerFn alias in rain.agent.
HymnSamplerFn = Callable[[str, int], str]


class KbAugmentedSampler:
    """Wrap any base HYMN sampler with KB-context prepending."""

    def __init__(
        self,
        kb: ShardedKB,
        base_sampler: HymnSamplerFn,
        *,
        max_facts: int = 5,
        prefix: str = "Context: ",
        question_prefix: str = "\nQ: ",
        answer_prefix: str = "\nA:",
    ) -> None:
        self.kb = kb
        self.base = base_sampler
        self.max_facts = max_facts
        self.prefix = prefix
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix

    def candidate_subjects(self, question: str) -> list[str]:
        """Very simple subject extraction: every alphanumeric token of length
        >= 3 plus their two-token concatenation. The KB's `query()` returns
        None for unknown subjects so over-fetching is cheap."""
        words = [
            w.strip(".,?!;:'\"()[]{}").lower()
            for w in question.split()
            if w.strip(".,?!;:'\"()[]{}")
        ]
        words = [w for w in words if len(w) >= 3]
        candidates: list[str] = list(words)
        for a, b in zip(words, words[1:]):
            candidates.append(f"{a}_{b}")
        # De-dupe while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for c in candidates:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def retrieve(self, question: str) -> list[tuple[str, str, str]]:
        """Pull (s, r, o) facts from the KB that mention any candidate subject."""
        cands = self.candidate_subjects(question)
        facts: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        # The KB API exposes query(subject, relation) but no enumerate-by-subject.
        # We probe a small fixed relation set per candidate -- the same
        # relations the rendered Q/A corpus uses.
        probe_relations = [
            "isa",
            "is_a",
            "kind",
            "lives_in",
            "located_in",
            "born_in",
            "has_part",
            "has_property",
            "made_of",
            "color",
            "size",
            "is",
            "wrote",
            "is_used_for",
        ]
        for subj in cands:
            for rel in probe_relations:
                obj = self.kb.query(subj, rel)
                if obj is None:
                    continue
                triple = (subj, rel, obj)
                if triple not in seen:
                    facts.append(triple)
                    seen.add(triple)
                if len(facts) >= self.max_facts:
                    return facts
        return facts

    def _format_context(self, facts: list[tuple[str, str, str]]) -> str:
        if not facts:
            return ""
        lines = []
        for s, r, o in facts:
            s_clean = s.replace("_", " ")
            r_clean = r.replace("_", " ")
            o_clean = o.replace("_", " ")
            lines.append(f"{self.prefix}{s_clean} {r_clean} {o_clean}.")
        return "\n".join(lines)

    def sample(self, question: str, n_tokens: int) -> str:
        """Compose: [context lines][\nQ: question][\nA:] then HYMN continuation."""
        facts = self.retrieve(question)
        context = self._format_context(facts)
        prompt = (
            (context + self.question_prefix if context else self.question_prefix.lstrip())
            + question.strip()
            + self.answer_prefix
        )
        out = self.base(prompt, n_tokens)
        return out
