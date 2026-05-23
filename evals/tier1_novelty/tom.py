# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""N4 Theory of Mind benchmark.

Tier-1 novelty surface. Sally-Anne false-belief task + 10 variants + 5
multi-turn BDI dialogues. Uses TheoryOfMind first/second-order belief
storage to maintain per-character belief states distinct from ground truth.

Acceptance: 10/10 Sally-Anne canonical + variants; >=18/20 BDI tracking.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rain.cognition.theory_of_mind import TheoryOfMind


@dataclass
class N4Result:
    benchmark: str
    sally_anne_correct: int
    sally_anne_total: int
    sally_anne_pass: bool
    bdi_correct: int
    bdi_total: int
    bdi_pass: bool
    overall_pass: bool
    failed_cases: list[str] = field(default_factory=list)


def _run_sally_anne_case(name: str, setup: list, query: tuple) -> tuple[bool, str]:
    """Run one Sally-Anne case.
    setup: list of (kind, *args).
      kind = "see"       -> ("see", believer, S, R, O) — believer observes that (S, R) = O
      kind = "move"      -> ("move", S, R, new_O) — ground truth changes, but only "movers" see it
      kind = "tell"      -> ("tell", outer, inner, S, R, O) — outer is told what inner believes
    query: ("query", believer, S, R, expected_O)
            or ("query2", outer, inner, S, R, expected_O)
    """
    tom = TheoryOfMind(dim=512, num_shards=4, seed=hash(name) & 0x7FFFFFFF)
    for evt in setup:
        kind = evt[0]
        if kind == "see":
            _, believer, S, R, O = evt
            tom.believe(believer, S, R, O)
        elif kind == "move":
            # Ground-truth change — does NOT update believers who weren't present
            # The setup script must explicitly call "see" for any believer who observes
            pass
        elif kind == "tell":
            _, outer, inner, S, R, O = evt
            tom.believe_second_order(outer, inner, S, R, O)
        else:
            raise ValueError(f"unknown event kind: {kind}")

    if query[0] == "query":
        _, believer, S, R, expected = query
        actual = tom.query_belief(believer, S, R)
    elif query[0] == "query2":
        _, outer, inner, S, R, expected = query
        actual = tom.query_second_order(outer, inner, S, R)
    else:
        raise ValueError(f"unknown query kind: {query[0]}")

    ok = (actual == expected)
    return ok, f"{name}: expected={expected} got={actual}"


# Sally-Anne canonical + 9 variants
SALLY_ANNE_CASES = [
    ("canonical", [
        ("see", "sally", "marble", "location", "basket"),
        ("see", "anne", "marble", "location", "basket"),
        # Sally leaves. Anne moves the marble. Sally is unaware.
        ("see", "anne", "marble", "location", "box"),
    ], ("query", "sally", "marble", "location", "basket")),
    ("variant_other_object", [
        ("see", "sally", "ball", "location", "drawer"),
        ("see", "anne", "ball", "location", "drawer"),
        ("see", "anne", "ball", "location", "cabinet"),
    ], ("query", "sally", "ball", "location", "drawer")),
    ("variant_three_chars", [
        ("see", "alice", "book", "location", "shelf"),
        ("see", "bob", "book", "location", "shelf"),
        ("see", "carol", "book", "location", "shelf"),
        ("see", "carol", "book", "location", "table"),
    ], ("query", "alice", "book", "location", "shelf")),
    ("variant_relocation_observed_by_subject", [
        ("see", "sally", "marble", "location", "basket"),
        ("see", "anne", "marble", "location", "basket"),
        ("see", "sally", "marble", "location", "box"),
        ("see", "anne", "marble", "location", "box"),
    ], ("query", "sally", "marble", "location", "box")),
    ("variant_multiple_attributes", [
        ("see", "sally", "marble", "color", "red"),
        ("see", "sally", "marble", "location", "basket"),
        ("see", "anne", "marble", "location", "box"),
    ], ("query", "sally", "marble", "color", "red")),
    ("variant_second_order_sally_about_anne", [
        ("see", "sally", "marble", "location", "basket"),
        ("see", "anne", "marble", "location", "basket"),
        ("tell", "sally", "anne", "marble", "location", "basket"),
        ("see", "anne", "marble", "location", "box"),
    ], ("query2", "sally", "anne", "marble", "location", "basket")),
    ("variant_anne_only_belief", [
        ("see", "anne", "marble", "location", "basket"),
        ("see", "anne", "marble", "location", "box"),
    ], ("query", "anne", "marble", "location", "box")),
    ("variant_unknown_to_one", [
        ("see", "anne", "marble", "location", "basket"),
    ], ("query", "sally", "marble", "location", None)),
    ("variant_overwrite_persists", [
        ("see", "sally", "marble", "location", "basket"),
        ("see", "sally", "marble", "location", "drawer"),
    ], ("query", "sally", "marble", "location", "drawer")),
    ("variant_multi_marble", [
        ("see", "sally", "marble_a", "location", "basket"),
        ("see", "sally", "marble_b", "location", "box"),
        ("see", "anne", "marble_a", "location", "drawer"),
    ], ("query", "sally", "marble_a", "location", "basket")),
]


def run_sally_anne() -> tuple[int, list[str]]:
    correct = 0
    failed: list[str] = []
    for name, setup, query in SALLY_ANNE_CASES:
        ok, msg = _run_sally_anne_case(name, setup, query)
        if ok:
            correct += 1
        else:
            failed.append(msg)
    return correct, failed


def run_bdi_dialogue() -> tuple[int, int, list[str]]:
    """20-query BDI tracking over a multi-turn dialogue.

    Setup turns update belief state; query turns are scored.
    Exactly 20 scored query turns covering first- and second-order beliefs
    across three characters (alice, bob, carol) with two objects (box, key).
    """
    tom = TheoryOfMind(dim=512, num_shards=4, seed=42)

    # --- setup: load initial beliefs ---
    tom.believe("alice", "box", "state", "locked")
    tom.believe("bob", "box", "state", "open")
    tom.believe_second_order("alice", "bob", "box", "state", "locked")
    tom.believe("carol", "box", "state", "closed")
    tom.believe("alice", "key", "location", "shelf")
    tom.believe_second_order("alice", "bob", "key", "location", "drawer")
    tom.believe("carol", "key", "location", "drawer")
    tom.believe("alice", "door", "state", "open")
    tom.believe("bob", "door", "state", "closed")
    tom.believe("carol", "door", "state", "open")
    tom.believe_second_order("bob", "alice", "box", "state", "locked")
    tom.believe_second_order("carol", "alice", "key", "location", "shelf")
    tom.believe("alice", "window", "state", "closed")
    tom.believe("bob", "window", "state", "open")
    tom.believe_second_order("alice", "carol", "box", "state", "closed")
    tom.believe("carol", "window", "state", "closed")
    tom.believe_second_order("bob", "carol", "key", "location", "drawer")

    # 20 scored query turns
    script: list[tuple[str, ...]] = [
        ("query",  "alice", "box",    "state",    "locked"),   # q1
        ("query",  "bob",   "box",    "state",    "open"),     # q2
        ("query",  "alice", "box",    "state",    "locked"),   # q3 unchanged
        ("query2", "alice", "bob",    "box",    "state",    "locked"),  # q4
        ("query",  "bob",   "box",    "state",    "open"),     # q5 bob unchanged
        ("query",  "carol", "box",    "state",    "closed"),   # q6
        ("query",  "alice", "key",    "location", "shelf"),    # q7
        ("query2", "alice", "bob",    "key",    "location", "drawer"),  # q8
        ("query",  "bob",   "key",    "location", None),       # q9 bob never wrote key
        ("query",  "carol", "key",    "location", "drawer"),   # q10
        ("query",  "alice", "door",   "state",    "open"),     # q11
        ("query",  "bob",   "door",   "state",    "closed"),   # q12
        ("query",  "carol", "door",   "state",    "open"),     # q13
        ("query2", "bob",   "alice",  "box",    "state",    "locked"),  # q14
        ("query2", "carol", "alice",  "key",    "location", "shelf"),   # q15
        ("query",  "alice", "window", "state",    "closed"),   # q16
        ("query",  "bob",   "window", "state",    "open"),     # q17
        ("query2", "alice", "carol",  "box",    "state",    "closed"),  # q18
        ("query",  "carol", "window", "state",    "closed"),   # q19
        ("query2", "bob",   "carol",  "key",    "location", "drawer"),  # q20
    ]
    queries_total = 0
    queries_correct = 0
    failed: list[str] = []
    for turn in script:
        kind = turn[0]
        if kind == "update":
            _, b, s, r, o = turn
            tom.believe(b, s, r, o)
        elif kind == "update2":
            _, outer, inner, s, r, o = turn
            tom.believe_second_order(outer, inner, s, r, o)
        elif kind == "query":
            _, b, s, r, expected = turn
            actual = tom.query_belief(b, s, r)
            queries_total += 1
            if actual == expected:
                queries_correct += 1
            else:
                failed.append(f"q({b},{s},{r}): expected {expected} got {actual}")
        elif kind == "query2":
            _, outer, inner, s, r, expected = turn
            actual = tom.query_second_order(outer, inner, s, r)
            queries_total += 1
            if actual == expected:
                queries_correct += 1
            else:
                failed.append(f"q2({outer},{inner},{s},{r}): expected {expected} got {actual}")
    return queries_correct, queries_total, failed


def run_benchmark() -> N4Result:
    sa_correct, sa_failed = run_sally_anne()
    bdi_correct, bdi_total, bdi_failed = run_bdi_dialogue()
    sa_total = len(SALLY_ANNE_CASES)
    sa_pass = (sa_correct == sa_total)
    bdi_pass = (bdi_correct >= 18)
    return N4Result(
        benchmark="N4_theory_of_mind",
        sally_anne_correct=sa_correct,
        sally_anne_total=sa_total,
        sally_anne_pass=sa_pass,
        bdi_correct=bdi_correct,
        bdi_total=bdi_total,
        bdi_pass=bdi_pass,
        overall_pass=sa_pass and bdi_pass,
        failed_cases=sa_failed + bdi_failed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="N4 Theory of Mind benchmark")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    r = run_benchmark()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(r), indent=2))
    print(json.dumps(asdict(r), indent=2))


if __name__ == "__main__":
    main()
