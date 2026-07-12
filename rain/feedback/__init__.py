"""RAIN feedback substrate.

Wraps external judges (local Ollama models in the v0 broke-mode plan, an
agent-team in v1) and converts their verdicts into per-relation calibration
updates on ConsciousAgent. This is the no-paid-RLHF path to continual
quality improvement.
"""

from rain.feedback.ollama_judge import (
    Judgment,
    OllamaJudge,
    build_triple_prompt,
    judge_agent_session,
    parse_judge_response,
)

__all__ = [
    "OllamaJudge",
    "Judgment",
    "judge_agent_session",
    "parse_judge_response",
    "build_triple_prompt",
]
