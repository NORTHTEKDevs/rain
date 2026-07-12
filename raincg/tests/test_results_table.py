import json
from pathlib import Path

from raincg.bench import citations as citations_mod
from raincg.bench import results_table as rt


def _write_result(dir_: Path, name: str, **overrides) -> dict:
    row = {
        "bench": "scan_addprim_jump",
        "system": "pure_vsa",
        "split": "test_addprim_jump",
        "n": 7706,
        "correct": 7706,
        "accuracy": 1.0,
        "ci95": [0.9995, 1.0],
        "params": 0,
        "train_s": 12.0,
        "eval_s": 3.4,
        "seed": 0,
        "config": {},
        "timestamp": "2026-07-11T00:00:00Z",
        "evidence_tier": "measured-fresh",
        "notes": "",
    }
    row.update(overrides)
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / name).write_text(json.dumps(row), encoding="utf-8")
    return row


# --- citations shape ---------------------------------------------------

def test_citations_have_required_fields():
    for c in citations_mod.CITATIONS:
        assert {"bench", "system", "accuracy", "source"}.issubset(c.keys())
        assert isinstance(c["accuracy"], float)
        assert 0.0 <= c["accuracy"] <= 1.0


def test_citations_include_lake_baroni_and_hupkes():
    sources = " ".join(c["source"] for c in citations_mod.CITATIONS)
    assert "Lake & Baroni" in sources
    assert "Hupkes" in sources


def test_citations_for_bench_filters():
    scan_cites = citations_mod.citations_for_bench("scan_addprim_jump")
    assert all(c["bench"] == "scan_addprim_jump" for c in scan_cites)
    assert len(scan_cites) >= 3


# --- load_results --------------------------------------------------

def test_load_results_reads_fixture_jsons(tmp_path):
    _write_result(tmp_path, "scan_addprim_jump__pure_vsa__seed0.json")
    _write_result(tmp_path, "pcfg_set__transformer_d512x2__seed0.json",
                  bench="pcfg_set", system="transformer_d512x2", accuracy=0.86,
                  params=8_400_000)
    results = rt.load_results(tmp_path)
    assert len(results) == 2
    systems = {r["system"] for r in results}
    assert systems == {"pure_vsa", "transformer_d512x2"}


def test_load_results_skips_malformed_json(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")
    _write_result(tmp_path, "scan_addprim_jump__pure_vsa__seed0.json")
    results = rt.load_results(tmp_path)
    assert len(results) == 1


# --- rendering -------------------------------------------------------

def test_render_includes_measured_and_cited_rows():
    results = [_write_result_inmem()]
    md = rt.render_results_md(results)
    assert "measured-fresh" in md
    assert "cited-not-reproduced" in md
    assert "Lake & Baroni 2018" in md
    assert "## scan_addprim_jump" in md


def test_render_cogs_gen_always_shows_full_denominator():
    row = _write_result_inmem(bench="cogs_gen", system="llm_llama3.2:3b",
                              n=30, correct=6, accuracy=0.2,
                              config={"full_n": 300, "full_correct": 60})
    md = rt.render_results_md([row])
    assert "full-denominator accuracy" in md
    assert "20.00%" in md  # full_correct/full_n = 60/300 = 20%
    assert "coverage" in md
    assert "10.0%" in md  # n/full_n = 30/300 = 10%


def test_render_flags_subsampled_test_limit():
    row = _write_result_inmem(bench="scan_addprim_jump", system="llm_qwen2.5:14b",
                              config={"test_limit": 40})
    md = rt.render_results_md([row])
    assert "subsampled, paired" in md
    assert "test_limit=40" in md


def test_render_table_has_bench_sections_for_citation_only_benches():
    md = rt.render_results_md([])
    assert "## pcfg_set" in md
    assert "## scan_addprim_jump" in md


def test_main_writes_out_file(tmp_path):
    results_dir = tmp_path / "results"
    _write_result(results_dir, "scan_addprim_jump__pure_vsa__seed0.json")
    out_path = tmp_path / "RESULTS.md"
    rc = rt.main(["--results-dir", str(results_dir), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "# RAINCG Results" in text


def _write_result_inmem(**overrides) -> dict:
    row = {
        "bench": "scan_addprim_jump",
        "system": "pure_vsa",
        "split": "test_addprim_jump",
        "n": 7706,
        "correct": 7706,
        "accuracy": 1.0,
        "ci95": [0.9995, 1.0],
        "params": 0,
        "train_s": 12.0,
        "eval_s": 3.4,
        "seed": 0,
        "config": {},
        "timestamp": "2026-07-11T00:00:00Z",
        "evidence_tier": "measured-fresh",
        "notes": "",
    }
    row.update(overrides)
    return row
