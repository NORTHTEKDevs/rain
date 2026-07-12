"""Generate KB seed triples by distilling from a local Ollama model.

Track 2 of the broke-mode training plan. Uses one of the local Ollama
models (qwen3-coder-30B, devstral-24B, qwen2.5-coder-7B, llama3.2-3b)
to generate structured (subject, relation, object) triples for a list
of topics. Output is JSONL, one fact per line, compatible with
`rain.data.kb_seed.seed_from_jsonl`.

Usage:

    # Built-in topic list (~50 topics, ~1500 triples on default n-per-topic)
    python -m scripts.seed_kb_from_ollama \\
        --model huihui_ai/qwen3-coder-abliterated:30b \\
        --n-per-topic 30 \\
        --out data/kb_seed/qwen30b_default.jsonl

    # From a topic file (one topic per line)
    python -m scripts.seed_kb_from_ollama \\
        --model llama3.2:3b \\
        --topics-file my_topics.txt \\
        --n-per-topic 50 \\
        --out data/kb_seed/from_topics.jsonl

The Ollama HTTP API at http://localhost:11434 is assumed to be reachable.
Each fact is schema-validated (3-string tuple, non-empty, lowercase,
underscored relation) before being written. Invalid facts are dropped
with a per-batch count printed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 180  # 30B model + 30 triples can take a minute


# A modest default topic list. Real production runs should use a file
# (Wikipedia categories, MeSH terms, FrameNet, etc.).
DEFAULT_TOPICS = [
    "lions", "wolves", "eagles", "dolphins", "elephants",
    "the Roman Empire", "the French Revolution", "World War II",
    "ancient Egypt", "the Industrial Revolution",
    "water (chemistry)", "carbon (chemistry)", "the periodic table",
    "DNA and genetics", "photosynthesis",
    "the solar system", "black holes", "supernovae",
    "the Pacific Ocean", "Mount Everest",
    "the human heart", "the human brain", "the immune system",
    "blood circulation", "vitamins",
    "Shakespeare's plays", "Beethoven's symphonies", "Picasso's paintings",
    "jazz music", "Greek mythology",
    "the Internet", "the transistor", "machine learning",
    "the printing press", "the steam engine",
    "the United States Constitution", "the European Union",
    "the United Nations", "NATO", "the World Bank",
    "Python (programming language)", "C++ (programming language)",
    "the Linux kernel", "the TCP/IP stack", "relational databases",
    "Mount Fuji", "the Amazon rainforest", "the Sahara desert",
    "the Great Barrier Reef", "Antarctica",
]


SYSTEM_PROMPT = (
    "You are a factual knowledge extractor. You output ONLY valid JSON. "
    "No prose, no commentary, no markdown fences."
)


def build_user_prompt(topic: str, n: int) -> str:
    """Prompt template that asks for structured triples about a topic."""
    return (
        f"Output exactly {n} factual (subject, relation, object) triples about: {topic}.\n"
        "Each triple encodes one specific fact.\n"
        "Format: a single JSON array, where each element is a 3-element array "
        "[subject, relation, object].\n"
        "All values are lowercase, underscore-separated, no spaces.\n"
        "Use clean relation names like isa, lives_in, made_of, has_part, "
        "wrote, born_in, located_in, color, kind, has_property.\n"
        "Example output (do not copy these facts):\n"
        '[["lion","isa","mammal"],["lion","lives_in","savanna"],'
        '["lion","has_property","carnivorous"]]\n'
        f"Now output {n} triples about {topic}. JSON only:"
    )


@dataclass
class SeedRunStats:
    model: str
    topics_attempted: int = 0
    topics_with_any_facts: int = 0
    facts_written: int = 0
    facts_rejected: int = 0
    wall_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


_VALID_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,80}$")


def _normalize_token(s: str) -> str:
    """Lowercase, replace whitespace + dashes with underscores, strip junk.

    Leading/trailing underscores are stripped, and runs of underscores are
    collapsed so a clean alpha-numeric token surfaces even from inputs that
    start with garbage like a registered-mark symbol."""
    s = s.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _valid_triple(t: object) -> tuple[str, str, str] | None:
    """Coerce one parsed JSON item to a clean (s, r, o) tuple, or None."""
    if not isinstance(t, (list, tuple)) or len(t) != 3:
        return None
    s, r, o = t
    if not (isinstance(s, str) and isinstance(r, str) and isinstance(o, str)):
        return None
    s, r, o = _normalize_token(s), _normalize_token(r), _normalize_token(o)
    if not all((s, r, o)):
        return None
    if not (_VALID_TOKEN.match(s) and _VALID_TOKEN.match(r) and _VALID_TOKEN.match(o)):
        return None
    return (s, r, o)


def _balance_brackets(candidate: str) -> str:
    """Append the closing brackets needed to balance an unclosed JSON array.

    Small local LLMs often emit valid triples but truncate before writing
    the outer ']'. We count unmatched opens (ignoring brackets inside double-
    quoted strings) and append as many ']' as needed.
    """
    depth = 0
    in_string = False
    escape_next = False
    for ch in candidate:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
    if depth > 0:
        return candidate + ("]" * depth)
    return candidate


def _try_parse(text: str) -> list | None:
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, list) else None


def _extract_json_array(text: str) -> list | None:
    """Find the first JSON array in `text` and parse it.

    Robustness layers (tried in order until one succeeds):
      1. Markdown fence ```json ... ```.
      2. Bare ``[ ... ]`` slice (text.find('[') to text.rfind(']')).
      3. From-first-``[`` to end-of-text, with brackets balanced.
      4. Same as (3) after dropping any trailing unterminated value
         (everything past the last comma at depth-0-inside-string).
    """
    fence_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if fence_match:
        parsed = _try_parse(fence_match.group(1))
        if parsed is not None:
            return parsed

    start = text.find("[")
    if start < 0:
        return None
    end = text.rfind("]")
    if end > start:
        parsed = _try_parse(text[start:end + 1])
        if parsed is not None:
            return parsed

    # Truncated-array recovery: balance any missing closing brackets.
    candidate = text[start:]
    balanced = _balance_brackets(candidate)
    parsed = _try_parse(balanced)
    if parsed is not None:
        return parsed

    # Last resort: trim back to the last completed sub-array (the final
    # ',\n  [ ... ]' boundary), then balance.
    trim_end = candidate.rfind("]")
    if trim_end > 0:
        truncated = candidate[:trim_end + 1]
        parsed = _try_parse(_balance_brackets(truncated))
        if parsed is not None:
            return parsed

    return None


def ollama_generate(
    model: str,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = 0.4,
) -> str:
    """Single non-streaming /api/chat call; returns the assistant's content string."""
    resp = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    msg = payload.get("message") or {}
    return str(msg.get("content", ""))


def generate_triples_for_topic(
    topic: str,
    n: int,
    model: str,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[list[tuple[str, str, str]], int]:
    """Returns (valid_triples, n_rejected_for_schema)."""
    content = ollama_generate(model, build_user_prompt(topic, n), url=url, timeout=timeout)
    parsed = _extract_json_array(content)
    if parsed is None:
        return [], 0
    valid: list[tuple[str, str, str]] = []
    rejected = 0
    for item in parsed:
        v = _valid_triple(item)
        if v is None:
            rejected += 1
            continue
        valid.append(v)
    return valid, rejected


def _load_topics(args: argparse.Namespace) -> list[str]:
    if args.topics_file:
        return [
            line.strip()
            for line in Path(args.topics_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return list(DEFAULT_TOPICS)


def run(args: argparse.Namespace) -> SeedRunStats:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    topics = _load_topics(args)
    stats = SeedRunStats(model=args.model)
    t0 = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fh:
        for i, topic in enumerate(topics):
            stats.topics_attempted += 1
            try:
                triples, rejected = generate_triples_for_topic(
                    topic, args.n_per_topic, args.model,
                    url=args.url, timeout=args.timeout,
                )
            except Exception as exc:
                stats.errors.append(f"{topic}: {exc!r}")
                if args.verbose:
                    print(f"[{i + 1}/{len(topics)}] {topic}: ERROR {exc!r}", file=sys.stderr)
                continue
            if triples:
                stats.topics_with_any_facts += 1
            stats.facts_rejected += rejected
            for s, r, o in triples:
                fh.write(json.dumps(
                    {"subject": s, "relation": r, "object": o, "topic": topic, "source": args.model}
                ) + "\n")
                stats.facts_written += 1
            if args.verbose:
                print(
                    f"[{i + 1}/{len(topics)}] {topic}: "
                    f"{len(triples)} kept, {rejected} rejected"
                )
    stats.wall_seconds = time.perf_counter() - t0
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Seed RAIN KB from local Ollama LLM")
    p.add_argument("--model", default="llama3.2:3b",
                   help="Ollama model name. Default llama3.2:3b: small, fast, reliable JSON output. "
                        "Larger models like qwen3-coder-30B can be faster per-fact when they work, but "
                        "the abliterated qwen3 build silently returns empty content on this prompt -- "
                        "test any new model with a 3-topic smoke run before kicking off the full sweep.")
    p.add_argument("--topics-file", default=None,
                   help="Optional path to a topic list (one per line). "
                        "Default: built-in 50-topic baseline.")
    p.add_argument("--n-per-topic", type=int, default=30)
    p.add_argument("--url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    stats = run(args)

    print(json.dumps({
        "model": stats.model,
        "topics_attempted": stats.topics_attempted,
        "topics_with_any_facts": stats.topics_with_any_facts,
        "facts_written": stats.facts_written,
        "facts_rejected": stats.facts_rejected,
        "wall_seconds": round(stats.wall_seconds, 1),
        "errors": stats.errors[:5],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
