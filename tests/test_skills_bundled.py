# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the 4 bundled on-disk skills."""

import pytest

from rain.core.encoder_bank import EncoderBank
from rain.core.procedural_adapter import ProceduralAdapter, SkillRegistry


D = 10_000


# ============================================================
# math_solver
# ============================================================


@pytest.fixture
def math_skill() -> ProceduralAdapter:
    return ProceduralAdapter.load("skills/math_solver")


def test_math_basic(math_skill):
    assert math_skill.invoke("what is 47 * 38", kb=None) == "1786"


def test_math_division(math_skill):
    assert math_skill.invoke("compute 100 / 4", kb=None) == "25"


def test_math_paren(math_skill):
    assert math_skill.invoke("evaluate (10 + 5) * 2", kb=None) == "30"


def test_math_power(math_skill):
    assert math_skill.invoke("find 2 ** 10", kb=None) == "1024"


def test_math_negative(math_skill):
    assert math_skill.invoke("999 - 100", kb=None) == "899"


def test_math_failure_gracefully(math_skill):
    out = math_skill.invoke("what color is the sky", kb=None)
    assert "could not parse" in out


# ============================================================
# date_math
# ============================================================


@pytest.fixture
def date_skill() -> ProceduralAdapter:
    return ProceduralAdapter.load("skills/date_math")


def test_date_between(date_skill):
    out = date_skill.invoke("days between 2026-05-24 and 2026-12-31", kb=None)
    assert "221 days" in out


def test_date_from(date_skill):
    out = date_skill.invoke("30 days from 2026-05-24", kb=None)
    assert "2026-06-23" in out


def test_date_weekday(date_skill):
    out = date_skill.invoke("what day is 2026-12-25", kb=None)
    assert "Friday" in out  # Dec 25 2026 is a Friday


def test_date_minus(date_skill):
    out = date_skill.invoke("7 days before 2026-05-24", kb=None)
    assert "2026-05-17" in out


def test_date_failure_gracefully(date_skill):
    out = date_skill.invoke("when does the world end", kb=None)
    assert "could not parse" in out or "need" in out


# ============================================================
# unit_converter
# ============================================================


@pytest.fixture
def unit_skill() -> ProceduralAdapter:
    return ProceduralAdapter.load("skills/unit_converter")


def test_unit_length(unit_skill):
    out = unit_skill.invoke("100 km to miles", kb=None)
    # 100 km = 62.1371 miles
    assert "62" in out


def test_unit_mass(unit_skill):
    out = unit_skill.invoke("5 lbs in kg", kb=None)
    # 5 lb = 2.26796 kg
    assert "2.26" in out or "2.27" in out


def test_unit_temp(unit_skill):
    out = unit_skill.invoke("32 F to C", kb=None)
    # 32 F = 0 C
    assert "0" in out


def test_unit_temp_kelvin(unit_skill):
    out = unit_skill.invoke("100 C to K", kb=None)
    # 100 C = 373.15 K
    assert "373" in out


def test_unit_volume(unit_skill):
    out = unit_skill.invoke("1 gal to L", kb=None)
    # 1 gal = 3.78541 L
    assert "3.78" in out or "3.79" in out


def test_unit_unknown(unit_skill):
    out = unit_skill.invoke("100 floops to bloops", kb=None)
    assert "unknown" in out.lower() or "mismatched" in out.lower()


def test_unit_temp_mismatch(unit_skill):
    out = unit_skill.invoke("100 C to miles", kb=None)
    assert "temp" in out.lower()


# ============================================================
# regex_extractor
# ============================================================


@pytest.fixture
def regex_skill() -> ProceduralAdapter:
    return ProceduralAdapter.load("skills/regex_extractor")


def test_regex_emails(regex_skill):
    out = regex_skill.invoke(
        "extract emails from contact me at foo@bar.com or baz@qux.io", kb=None
    )
    assert "foo@bar.com" in out
    assert "baz@qux.io" in out


def test_regex_urls(regex_skill):
    out = regex_skill.invoke(
        "find urls from check https://example.com and http://foo.io/path", kb=None
    )
    assert "https://example.com" in out
    assert "http://foo.io/path" in out


def test_regex_phones(regex_skill):
    out = regex_skill.invoke(
        "extract phones from call (907) 555-1234 or 555-9876", kb=None
    )
    # Should match at least the first; the second is a partial.
    assert "907" in out


def test_regex_dates(regex_skill):
    out = regex_skill.invoke(
        "extract dates from we met on 2026-05-24 and 12/31/2025", kb=None
    )
    assert "2026-05-24" in out
    assert "12/31/2025" in out


def test_regex_no_kind(regex_skill):
    out = regex_skill.invoke("hello world", kb=None)
    assert "which kind" in out


def test_regex_no_matches(regex_skill):
    out = regex_skill.invoke("extract emails from no emails here just text", kb=None)
    assert "no email" in out


# ============================================================
# Registry: all 4 skills load
# ============================================================


def test_registry_loads_core_four():
    """The 4 core skills must still all load (extended skills add to count)."""
    reg = SkillRegistry()
    n = reg.load_dir("skills")
    assert n >= 4
    names = {s.meta.name for s in reg.skills}
    core = {"math_solver", "date_math", "unit_converter", "regex_extractor"}
    assert core.issubset(names)


def test_registry_routes_correctly():
    enc = EncoderBank(dim=D)
    reg = SkillRegistry()
    reg.load_dir("skills")
    cases = [
        ("what is 47 * 38", "math_solver"),
        ("100 km to miles", "unit_converter"),
        ("30 days from 2026-05-24", "date_math"),
        ("extract emails from foo@bar.com", "regex_extractor"),
    ]
    correct = 0
    for query, expected in cases:
        qhv = enc.encode("text", query)
        matches = reg.match(qhv, threshold=0.1, query_text=query)
        if matches and matches[0][1].meta.name == expected:
            correct += 1
    assert correct >= 3, f"only {correct}/4 routed correctly"
