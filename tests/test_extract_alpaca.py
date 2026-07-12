"""Tests for the Alpaca extractor's formatting helper.

The HF dataset download path is exercised by the CLI smoke run, not by
CI (datasets pulls 22 MB and we don't want CI to hit HuggingFace on
every push).
"""

from scripts.extract_alpaca import _format_one


def test_format_one_no_input():
    out = _format_one("List three colors.", "", "Red, green, blue.")
    assert out == "Q: List three colors.\nA: Red, green, blue.\n"


def test_format_one_with_input():
    out = _format_one("Translate to Spanish:", "Hello world", "Hola mundo")
    assert out == "Q: Translate to Spanish:\nHello world\nA: Hola mundo\n"


def test_format_one_strips_whitespace():
    out = _format_one("  Q  ", "  I  ", "  A  ")
    assert out == "Q: Q\nI\nA: A\n"
