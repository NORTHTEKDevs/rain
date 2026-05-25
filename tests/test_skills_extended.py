# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for the 4 extended bundled skills (json, calendar, stats, url)."""

import pytest

from rain.core.procedural_adapter import ProceduralAdapter, SkillRegistry


# ============================================================
# json_parser
# ============================================================


@pytest.fixture
def json_skill() -> ProceduralAdapter:
    return ProceduralAdapter.load("skills/json_parser")


def test_json_extract_field(json_skill):
    out = json_skill.invoke('get name from {"name": "Apollo", "year": 1969}', kb=None)
    assert out == "Apollo"


def test_json_extract_nested(json_skill):
    out = json_skill.invoke(
        'value of users.0.email in {"users": [{"email": "a@b.com"}]}', kb=None
    )
    assert out == "a@b.com"


def test_json_array_length(json_skill):
    out = json_skill.invoke("[1, 2, 3, 4]", kb=None)
    assert "length 4" in out


def test_json_missing_path(json_skill):
    out = json_skill.invoke('get foo from {"bar": 1}', kb=None)
    assert "not found" in out


def test_json_invalid(json_skill):
    out = json_skill.invoke("get x from {invalid json}", kb=None)
    assert "invalid JSON" in out


def test_json_no_block(json_skill):
    out = json_skill.invoke("what is the meaning of life", kb=None)
    assert "no JSON block" in out


# ============================================================
# calendar_lookup
# ============================================================


@pytest.fixture
def cal_skill() -> ProceduralAdapter:
    return ProceduralAdapter.load("skills/calendar_lookup")


def test_calendar_leap_year(cal_skill):
    assert "is a leap" in cal_skill.invoke("is 2024 a leap year", kb=None)
    assert "not a leap" in cal_skill.invoke("is 2023 a leap year", kb=None)


def test_calendar_days_in_year(cal_skill):
    assert "366" in cal_skill.invoke("how many days in 2024", kb=None)
    assert "365" in cal_skill.invoke("how many days in 2023", kb=None)


def test_calendar_quarter(cal_skill):
    out = cal_skill.invoke("what quarter is 2026-05-24 in", kb=None)
    assert "Q2" in out


def test_calendar_utc_offset(cal_skill):
    out = cal_skill.invoke("what is UTC offset for Tokyo?", kb=None)
    assert "+9" in out
    out2 = cal_skill.invoke("what is UTC offset for Anchorage?", kb=None)
    assert "-9" in out2


def test_calendar_unknown_city(cal_skill):
    out = cal_skill.invoke("UTC offset for ZebraVille?", kb=None)
    assert "unknown" in out.lower()


# ============================================================
# statistics
# ============================================================


@pytest.fixture
def stats_skill() -> ProceduralAdapter:
    return ProceduralAdapter.load("skills/statistics")


def test_stats_mean(stats_skill):
    out = stats_skill.invoke("mean of 1, 2, 3, 4, 5", kb=None)
    assert "mean=3" in out


def test_stats_median(stats_skill):
    out = stats_skill.invoke("median of 10 20 30", kb=None)
    assert "median=20" in out


def test_stats_stdev(stats_skill):
    out = stats_skill.invoke("stdev of 2 4 4 4 5 5 7 9", kb=None)
    assert "stdev" in out and "2." in out  # ~2.138


def test_stats_summary(stats_skill):
    out = stats_skill.invoke("summary stats for 1, 2, 3, 4, 5", kb=None)
    assert "mean" in out and "median" in out and "min" in out and "max" in out


def test_stats_no_numbers(stats_skill):
    out = stats_skill.invoke("mean of these numbers", kb=None)
    assert "no numbers" in out


def test_stats_no_op(stats_skill):
    out = stats_skill.invoke("1 2 3 4 5", kb=None)
    assert "which stat" in out.lower()


# ============================================================
# url_extract
# ============================================================


@pytest.fixture
def url_skill() -> ProceduralAdapter:
    return ProceduralAdapter.load("skills/url_extract")


def test_url_domain(url_skill):
    out = url_skill.invoke(
        "what is the domain of https://foo.bar.example.com/path", kb=None
    )
    assert "foo.bar.example.com" in out


def test_url_path(url_skill):
    out = url_skill.invoke("path of https://a.com/x/y/z", kb=None)
    assert "/x/y/z" in out


def test_url_scheme(url_skill):
    out = url_skill.invoke("scheme of https://example.com", kb=None)
    assert "https" in out


def test_url_query(url_skill):
    out = url_skill.invoke("query of https://example.com?a=1&b=2", kb=None)
    assert "a=1" in out


def test_url_full_decompose(url_skill):
    # Bare URL with no decomposition keyword -> full decompose output.
    out = url_skill.invoke("https://example.com:8080/foo?q=1", kb=None)
    assert "example.com" in out
    assert "https" in out


def test_url_no_match(url_skill):
    out = url_skill.invoke("just some text without urls", kb=None)
    assert "no URLs" in out


# ============================================================
# Registry: 8 skills total
# ============================================================


def test_registry_loads_eight_core_skills():
    """At least the 8 core skills must load (additional vertical skills
    like aviation_compliance add to count)."""
    reg = SkillRegistry()
    n = reg.load_dir("skills")
    assert n >= 8
    names = {s.meta.name for s in reg.skills}
    core_eight = {
        "math_solver", "date_math", "unit_converter", "regex_extractor",
        "json_parser", "calendar_lookup", "statistics", "url_extract",
    }
    assert core_eight.issubset(names)
