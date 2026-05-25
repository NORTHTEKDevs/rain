# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Teacher-LLM distillation: train RAIN-Net by asking a smarter model.

The cost story of RAIN-Net rests on this module. Instead of pretraining
the base model on web text (massive corpus, massive GPU cluster), we
distill from a frontier LLM as teacher.

Two teacher modes:

    OllamaTeacher    local LLM via Ollama HTTP. FREE. Good for v0.1.
                     Models: qwen2.5:7b, llama3.1:8b, mistral-nemo, etc.

    ClaudeTeacher    Anthropic API. Best quality. Costs API tokens.
                     Used selectively for hard / high-value examples.

The distillation training data is (query, teacher_answer, teacher_citations)
triples. We train the student on:
    L = KL(student || teacher) + lambda * L_grounding
where L_grounding rewards the student for using KB-Attention to produce
its answer (not memorising the fact in weights).

The result is a small student that has inherited the teacher's
knowledge compression at a fraction of the compute. Microsoft's Phi
series (3.5 Mini and below) demonstrated this works -- their small
models punch far above their parameter count by being distilled from
GPT-4 quality teachers on curated synthetic data.

Two new things RAIN-Net adds on top of standard distillation:
    1. KB-grounding loss: the student MUST use the KB-Attention layer
       to recover the teacher's answer, not memorise it.
    2. Multi-modal joint distillation: teachers across text + code +
       image-caption datasets all distill into the same HV-substrate
       student, training the encoder bank to produce consistent HVs.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rain.core.rain_net import RainNet


# -------------------- teacher backends --------------------


@dataclass
class TeacherResponse:
    """One teacher answer: text + optional citations + raw metadata."""

    answer: str
    citations: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    backend: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class OllamaTeacher:
    """Local LLM via Ollama HTTP. Free. Default for v0.1 distillation.

    Requires Ollama running at OLLAMA_HOST (default http://localhost:11434)
    with the named model pulled. Use `ollama pull qwen2.5:7b` first.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout

    def ask(self, prompt: str, system: str | None = None) -> TeacherResponse:
        """Send a chat completion request to Ollama; parse the response."""
        # Import lazily so the rest of the module imports without urllib3.
        import urllib.request

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if system is not None:
            payload["system"] = system
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.host}/api/generate"
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return TeacherResponse(
                answer="",
                backend="ollama",
                raw={"error": str(e), "model": self.model},
            )
        answer = body.get("response", "").strip()
        return TeacherResponse(
            answer=answer,
            tokens_in=int(body.get("prompt_eval_count", 0)),
            tokens_out=int(body.get("eval_count", 0)),
            backend="ollama",
            raw=body,
        )


class ClaudeTeacher:
    """Anthropic Claude via the API. Used for hard / high-value queries.

    Requires ANTHROPIC_API_KEY env var. Costs real money. Use selectively
    via the active-learning loop's verifier-gated escalation, not bulk
    pretraining.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def ask(self, prompt: str, system: str | None = None) -> TeacherResponse:
        if not self._api_key:
            return TeacherResponse(
                answer="",
                backend="claude",
                raw={"error": "ANTHROPIC_API_KEY not set"},
            )
        import urllib.request

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            body["system"] = system
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=data, method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("x-api-key", self._api_key)
        req.add_header("anthropic-version", "2023-06-01")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body_in = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return TeacherResponse(
                answer="",
                backend="claude",
                raw={"error": str(e)},
            )
        # Claude API response: content is a list of blocks; we take text.
        text = ""
        for block in body_in.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        usage = body_in.get("usage", {})
        return TeacherResponse(
            answer=text.strip(),
            tokens_in=int(usage.get("input_tokens", 0)),
            tokens_out=int(usage.get("output_tokens", 0)),
            backend="claude",
            raw=body_in,
        )


# -------------------- training-data record --------------------


@dataclass
class DistillExample:
    """One training example: query + teacher answer + citations.

    Saved to JSONL on disk; replayed during student training.
    """

    query: str
    teacher_answer: str
    teacher_backend: str
    tokens_in: int = 0
    tokens_out: int = 0
    citations: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "query": self.query,
                "teacher_answer": self.teacher_answer,
                "teacher_backend": self.teacher_backend,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "citations": self.citations,
                "timestamp": self.timestamp,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_json(cls, line: str) -> "DistillExample":
        d = json.loads(line)
        return cls(
            query=d["query"],
            teacher_answer=d["teacher_answer"],
            teacher_backend=d.get("teacher_backend", ""),
            tokens_in=int(d.get("tokens_in", 0)),
            tokens_out=int(d.get("tokens_out", 0)),
            citations=d.get("citations", []),
            timestamp=float(d.get("timestamp", 0.0)),
            metadata=d.get("metadata", {}),
        )


# -------------------- pipeline --------------------


class DistillationPipeline:
    """Coordinates teacher querying, dataset collection, and student
    ingestion.

    Standard use:

        pipe = DistillationPipeline(teacher=OllamaTeacher())
        for q in queries:
            ex = pipe.collect_example(q)
        pipe.save(Path("data/distill/v0.jsonl"))
        # ...later, in training:
        for ex in pipe.load(Path("data/distill/v0.jsonl")):
            student.ingest_fact(ex.teacher_answer, source=ex.teacher_backend)
    """

    def __init__(
        self,
        teacher: OllamaTeacher | ClaudeTeacher,
        system_prompt: str | None = None,
    ) -> None:
        self.teacher = teacher
        self.system_prompt = system_prompt or (
            "You are a teacher distilling knowledge into a smaller student "
            "model. Answer concisely. If the answer requires factual "
            "grounding, cite the source as 'source: ...' at the end."
        )
        self.examples: list[DistillExample] = []
        self.token_budget_used = 0

    def collect_example(
        self, query: str, metadata: dict[str, Any] | None = None
    ) -> DistillExample | None:
        """Query the teacher, build a DistillExample. None on failure."""
        resp = self.teacher.ask(query, system=self.system_prompt)
        if not resp.answer:
            return None
        # Extract trailing 'source:' citations if any
        citations: list[str] = []
        answer_text = resp.answer
        for line in answer_text.splitlines():
            ls = line.strip().lower()
            if ls.startswith("source:") or ls.startswith("citation:"):
                citations.append(line.strip())
        ex = DistillExample(
            query=query,
            teacher_answer=answer_text,
            teacher_backend=resp.backend,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            citations=citations,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self.examples.append(ex)
        self.token_budget_used += resp.tokens_in + resp.tokens_out
        return ex

    def collect_batch(
        self,
        queries: list[str],
        max_examples: int | None = None,
    ) -> list[DistillExample]:
        """Convenience: collect across a list of queries with optional cap."""
        out: list[DistillExample] = []
        for q in queries:
            ex = self.collect_example(q)
            if ex is not None:
                out.append(ex)
            if max_examples is not None and len(out) >= max_examples:
                break
        return out

    def save(self, path: str | Path) -> None:
        """Persist examples to JSONL on disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            for ex in self.examples:
                f.write(ex.to_json() + "\n")

    @staticmethod
    def load(path: str | Path) -> Iterator[DistillExample]:
        """Stream examples from a JSONL file."""
        p = Path(path)
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield DistillExample.from_json(line)

    def ingest_into_student(self, student: RainNet) -> int:
        """Pour collected examples into the student's semantic memory.

        Returns count ingested. Each (query, answer) pair becomes a
        semantic fact in the student. The student can then use the
        KB-Attention path to answer same-domain queries without re-asking
        the teacher.
        """
        count = 0
        for ex in self.examples:
            fact_text = f"Q: {ex.query} A: {ex.teacher_answer}"
            student.ingest_fact(
                text=fact_text,
                source=f"distill::{ex.teacher_backend}",
                confidence=0.9,
            )
            count += 1
        return count


# -------------------- curriculum builders --------------------


def synthetic_curriculum_queries(
    domain: str, n: int = 100, seed: int = 0
) -> list[str]:
    """Generate a curriculum of seed queries for a domain.

    Used to bootstrap a distillation dataset without pre-existing data.
    The queries themselves are synthetic but they cover the space the
    teacher should explain. Microsoft's textbooks-are-all-you-need
    approach: quality > quantity.

    Templates per domain:
        general    "explain X", "what is X used for", "compare X and Y"
        code       "implement X in Python", "what does X do in code"
        math       "compute X", "solve for X given Y"
        regulated  "what does regulation X require", "is X allowed under Y"
    """
    rng = np.random.default_rng(seed)
    topics = {
        "general": [
            "photosynthesis", "DNA replication", "supply and demand",
            "the French revolution", "general relativity", "machine learning",
            "the periodic table", "blood circulation", "plate tectonics",
            "the scientific method",
        ],
        "code": [
            "binary search", "depth-first search", "merge sort",
            "hash tables", "dynamic programming", "recursion",
            "object-oriented programming", "functional programming",
            "regular expressions", "memory management",
        ],
        "math": [
            "the quadratic formula", "the chain rule", "linear algebra",
            "the Pythagorean theorem", "the fundamental theorem of calculus",
            "Bayes theorem", "the law of large numbers",
            "the central limit theorem", "matrix multiplication", "eigenvalues",
        ],
        "regulated": [
            "FAA part 91", "FAA part 135", "OSHA fall protection",
            "HIPAA privacy rule", "GDPR consent", "SEC rule 10b-5",
            "EPA clean water act", "IRS section 179", "ADA accessibility",
            "FDA 510(k)",
        ],
    }
    pool = topics.get(domain, topics["general"])
    templates = [
        "Explain {t} in 3 sentences.",
        "What is {t} used for?",
        "Give a concrete example of {t}.",
        "What are common misconceptions about {t}?",
        "How does {t} relate to similar concepts?",
        "What are the prerequisites to understand {t}?",
        "What is the simplest possible description of {t}?",
        "What problem does {t} solve?",
        "Compare {t} with one alternative approach.",
        "What is the historical origin of {t}?",
    ]
    out: list[str] = []
    while len(out) < n:
        t = pool[int(rng.integers(0, len(pool)))]
        tmpl = templates[int(rng.integers(0, len(templates)))]
        out.append(tmpl.format(t=t))
    return out[:n]
