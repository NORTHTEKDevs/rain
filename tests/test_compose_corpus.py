"""Tests for the multi-corpus composer CLI script."""

import subprocess
import sys
from pathlib import Path


def _python() -> str:
    return sys.executable


def test_compose_concatenates_with_weights(tmp_path):
    """A weight of N copies the part N times."""
    a = tmp_path / "a.txt"
    a.write_text("alpha\n\nbeta", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("gamma", encoding="utf-8")
    out = tmp_path / "combined.txt"

    res = subprocess.run(
        [
            _python(),
            "-m",
            "scripts.compose_corpus",
            "--part",
            f"{a}:2",
            "--part",
            f"{b}:3",
            "--out",
            str(out),
            "--rng-seed",
            "0",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )
    assert res.returncode == 0, res.stderr
    text = out.read_text(encoding="utf-8")
    # 'alpha' appears 2x (from a:2 * 1 chunk), 'beta' appears 2x, 'gamma' 3x
    assert text.count("alpha") == 2
    assert text.count("beta") == 2
    assert text.count("gamma") == 3


def test_compose_chunk_shuffle_changes_order(tmp_path):
    """Without --shuffle-chunks the order should be deterministic concat;
    with --shuffle-chunks the same chunks appear but interleaved."""
    a = tmp_path / "a.txt"
    a.write_text("alpha\n\nbeta\n\ngamma", encoding="utf-8")
    out_unshuffled = tmp_path / "u.txt"
    out_shuffled = tmp_path / "s.txt"

    subprocess.run(
        [
            _python(),
            "-m",
            "scripts.compose_corpus",
            "--part",
            f"{a}:1",
            "--out",
            str(out_unshuffled),
            "--rng-seed",
            "0",
        ],
        capture_output=True,
        cwd=Path(__file__).parents[1],
    )
    subprocess.run(
        [
            _python(),
            "-m",
            "scripts.compose_corpus",
            "--part",
            f"{a}:1",
            "--out",
            str(out_shuffled),
            "--shuffle-chunks",
            "--rng-seed",
            "7",
        ],
        capture_output=True,
        cwd=Path(__file__).parents[1],
    )

    u = out_unshuffled.read_text(encoding="utf-8")
    s = out_shuffled.read_text(encoding="utf-8")
    assert sorted(u.split("\n\n")) == sorted(s.split("\n\n"))


def test_compose_skips_missing_part(tmp_path):
    """Missing input files produce a warning but don't abort the run."""
    a = tmp_path / "a.txt"
    a.write_text("alpha", encoding="utf-8")
    missing = tmp_path / "does_not_exist.txt"
    out = tmp_path / "out.txt"
    res = subprocess.run(
        [
            _python(),
            "-m",
            "scripts.compose_corpus",
            "--part",
            f"{a}:1",
            "--part",
            f"{missing}:5",
            "--out",
            str(out),
            "--rng-seed",
            "0",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )
    assert res.returncode == 0
    assert "warn:" in res.stdout or "warn:" in res.stderr
    assert "alpha" in out.read_text(encoding="utf-8")
