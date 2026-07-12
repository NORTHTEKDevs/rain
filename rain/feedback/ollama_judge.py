"""Use a local Ollama model as a free judge for RAIN answers.

Track 3 of the broke-mode training plan (docs/plans/2026-05-22-broke-mode-training.md).
RAIN normally takes user feedback via `ConsciousAgent.feedback(relation, was_correct)`
to update its per-relation Bayesian calibration tally. In broke mode there is no
human in the loop, so we substitute a local Ollama model: ask it to judge whether
RAIN's answer is correct, parse a structured verdict, and feed the verdict back.

Pipeline:
    OllamaJudge.judge(question_text, rain_answer_text)
        -> Judgment(correct: bool, confidence: float, reasoning: str)

The default judge model is llama3.2:3b -- fast, reliable JSON output, no VRAM
contention with bigger models loaded in parallel.

The judge call is NETWORK + LLM; it's slow (~1 s per call on this workstation).
Treat it as offline-batch infrastructure, not a per-token feedback loop.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_JUDGE_MODEL = "llama3.2:3b"
DEFAULT_TIMEOUT = 60


@dataclass
class Judgment:
    """One verdict from the judge."""

    correct: bool
    confidence: float
    reasoning: str
    raw: str  # the exact model output, kept for debugging / audit


SYSTEM_PROMPT = (
    "You are a strict fact-checker. Given a question and an AI's answer, "
    "you output ONLY a single JSON object with the keys: "
    '"correct" (true|false), "confidence" (0.0-1.0), "reasoning" (short string). '
    "No prose, no markdown fences, no other commentary."
)


def build_user_prompt(question: str, rain_answer: str) -> str:
    return (
        f"Question: {question}\n"
        f"Answer: {rain_answer}\n\n"
        "Is the answer factually correct?\n"
        'Output ONLY a JSON object like: {"correct": true, "confidence": 0.9, "reasoning": "matches commonly accepted fact"}'
    )


def build_triple_prompt(subject: str, relation: str, obj: str) -> tuple[str, str]:
    """Render a (subject, relation, object) triple as a clean (question, answer) pair
    for the judge, free of RAIN's surface artifacts ('I know that...directly from a
    stored fact') AND of snake_case formatting noise that triggers the judge to
    reject based on form rather than fact.

    Returns: (question_str, answer_str).
    """
    s_clean = subject.replace("_", " ")
    r_clean = relation.replace("_", " ")
    o_clean = obj.replace("_", " ")
    question = f"What is the {r_clean} of {s_clean}?"
    answer = f"The {r_clean} of {s_clean} is {o_clean}."
    return question, answer


def parse_judge_response(text: str) -> Judgment | None:
    """Extract the first JSON object from `text` and coerce to Judgment.

    Tolerates the same llama3.2 quirks the KB seeder handles:
    - Bare JSON
    - JSON inside ```...``` fences
    - JSON inside surrounding prose

    Returns None if no valid object is found OR required keys are missing.
    """
    candidate = None
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence:
        candidate = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if "correct" not in obj or "confidence" not in obj:
        return None
    try:
        correct = bool(obj["correct"])
        confidence = float(obj["confidence"])
    except (TypeError, ValueError):
        return None
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(obj.get("reasoning", ""))
    return Judgment(correct=correct, confidence=confidence, reasoning=reasoning, raw=text)


class OllamaJudge:
    """Synchronous wrapper around Ollama's /api/chat for one-shot judgments."""

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        url: str = DEFAULT_OLLAMA_URL,
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.url = url
        self.timeout = timeout
        self.temperature = temperature

    def judge(self, question: str, rain_answer: str) -> Judgment | None:
        """One judgment call. Returns None on parse failure or HTTP error."""
        try:
            resp = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(question, rain_answer)},
                    ],
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException:
            return None
        content = str(resp.json().get("message", {}).get("content", ""))
        return parse_judge_response(content)


def judge_agent_session(
    agent,  # rain.agent.ConsciousAgent -- avoiding circular import
    qa_pairs: Iterable[tuple[str, str]],  # iterable of (subject, relation)
    judge: OllamaJudge,
    *,
    min_confidence_for_update: float = 0.5,
) -> dict:
    """Run agent through each question, ask judge, update calibration.

    Returns a stats dict: total / answered / judged / correct / feedback_updates.
    Judgments with `confidence < min_confidence_for_update` are recorded but
    not fed back -- a fuzzy judge shouldn't drift the calibration tally.
    """
    stats = {
        "total": 0,
        "answered": 0,
        "judged": 0,
        "judge_parse_failures": 0,
        "correct": 0,
        "feedback_updates": 0,
    }
    for subject, relation in qa_pairs:
        stats["total"] += 1
        ans = agent.ask(subject, relation)
        if ans.inference_source is not None:
            stats["answered"] += 1
        question_text = f"What is the {relation} of {subject}?"
        verdict = judge.judge(question_text, ans.text)
        if verdict is None:
            stats["judge_parse_failures"] += 1
            continue
        stats["judged"] += 1
        if verdict.correct:
            stats["correct"] += 1
        if verdict.confidence >= min_confidence_for_update:
            agent.feedback(relation, verdict.correct)
            stats["feedback_updates"] += 1
    return stats
