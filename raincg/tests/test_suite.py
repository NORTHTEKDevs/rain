import json
import sys
from pathlib import Path

import pytest

from raincg.bench import suite


def _stub_stage(name="stub", timeout_s=30, glob_pat="pcfg_set__pure_vsa__seed*.json",
                cmd=None):
    return {
        "name": name,
        "bench": "pcfg_set",
        "system": "pure_vsa",
        "seed": 0,
        "cmd": cmd or ["{py}", "-c", "print('hello-from-stub')"],
        "expected_result_glob": glob_pat,
        "out_name": "pcfg_set__pure_vsa__seed0.json",
        "timeout_s": timeout_s,
        "needs": None,
    }


# --- registry shape ---------------------------------------------------

def test_stage_names_are_unique():
    names = [s["name"] for s in suite.STAGES]
    assert len(names) == len(set(names))


def test_expected_stages_present():
    expected = {
        "pcfg_vsa", "pcfg_transformer", "scan_transformer_baseline",
        "scan_hybrid_supervised", "scan_hybrid_outputonly", "scan_llm_llama",
        "scan_llm_qwen", "cogs_llm_llama", "grammar_induction",
        "cogs_template_symbolic", "cogs_role_reinforce",
        "scan_llm_sonnet5", "cogs_llm_sonnet5", "scan_llm_gpt55", "cogs_llm_gpt55",
    }
    assert expected == {s["name"] for s in suite.STAGES}


def test_openrouter_stages_need_api_key():
    for name in ("scan_llm_sonnet5", "cogs_llm_sonnet5", "scan_llm_gpt55", "cogs_llm_gpt55"):
        st = suite.stage_by_name(name)
        assert st["needs"] == "openrouter-api-key"
        assert "--backend" in st["cmd"] and "openrouter" in st["cmd"]


def test_all_stages_have_required_keys():
    required = {"name", "bench", "system", "cmd", "expected_result_glob",
               "timeout_s", "needs"}
    for s in suite.STAGES:
        assert required.issubset(s.keys())
        assert isinstance(s["cmd"], list)
        assert "{py}" in s["cmd"][0] or s["cmd"][0] == suite.DEFAULT_PY


def test_long_cpu_stage_flagged():
    st = suite.stage_by_name("pcfg_transformer")
    assert st["needs"] == "long-cpu"


def test_filesystem_safe_glob_no_colons():
    for s in suite.STAGES:
        assert ":" not in s["expected_result_glob"]
        assert "/" not in s["expected_result_glob"]


def test_safe_system_replaces_slash_and_colon():
    assert suite._safe_system("llm_anthropic/claude-sonnet-5") == "llm_anthropic-claude-sonnet-5"
    assert suite._safe_system("llm_llama3.2:3b") == "llm_llama3.2-3b"
    assert suite._safe_system("a/b:c") == "a-b-c"


def test_out_name_matches_run_filenames():
    # These must match the exact --out filenames used for the manual OpenRouter runs.
    assert (suite._out_path("scan_addprim_jump", "llm_anthropic/claude-sonnet-5", 0)
            == "scan_addprim_jump__llm_anthropic-claude-sonnet-5__seed0.json")
    assert (suite._out_path("cogs_gen", "llm_anthropic/claude-sonnet-5", 0)
            == "cogs_gen__llm_anthropic-claude-sonnet-5__seed0.json")
    assert (suite._out_path("scan_addprim_jump", "llm_openai/gpt-5.5", 0)
            == "scan_addprim_jump__llm_openai-gpt-5.5__seed0.json")
    assert (suite._out_path("cogs_gen", "llm_openai/gpt-5.5", 0)
            == "cogs_gen__llm_openai-gpt-5.5__seed0.json")


# --- skip / force logic -------------------------------------------------

def test_run_stage_skips_when_result_exists(tmp_path):
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"
    results_dir.mkdir(parents=True)
    (results_dir / "pcfg_set__pure_vsa__seed0.json").write_text("{}", encoding="utf-8")

    stage = _stub_stage()
    outcome = suite.run_stage(stage, py=sys.executable, repo=tmp_path,
                              results_dir=results_dir, logs_dir=logs_dir)
    assert outcome.status == "skipped"
    assert not logs_dir.exists() or not any(logs_dir.iterdir())


def test_run_stage_force_reruns_despite_existing_result(tmp_path):
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"
    results_dir.mkdir(parents=True)
    (results_dir / "pcfg_set__pure_vsa__seed0.json").write_text("{}", encoding="utf-8")

    stage = _stub_stage()
    outcome = suite.run_stage(stage, py=sys.executable, repo=tmp_path,
                              results_dir=results_dir, logs_dir=logs_dir, force=True)
    assert outcome.status == "ran"
    assert outcome.returncode == 0
    assert Path(outcome.log_path).exists()
    assert "hello-from-stub" in Path(outcome.log_path).read_text(encoding="utf-8")


def test_run_stage_executes_and_logs_when_no_result(tmp_path):
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"

    stage = _stub_stage()
    outcome = suite.run_stage(stage, py=sys.executable, repo=tmp_path,
                              results_dir=results_dir, logs_dir=logs_dir)
    assert outcome.status == "ran"
    assert outcome.returncode == 0
    log_text = Path(outcome.log_path).read_text(encoding="utf-8")
    assert "hello-from-stub" in log_text


def test_run_stage_reports_failure_on_nonzero_exit(tmp_path):
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"
    stage = _stub_stage(cmd=["{py}", "-c", "import sys; sys.exit(3)"])
    outcome = suite.run_stage(stage, py=sys.executable, repo=tmp_path,
                              results_dir=results_dir, logs_dir=logs_dir)
    assert outcome.status == "failed"
    assert outcome.returncode == 3


def test_run_stage_times_out(tmp_path):
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"
    stage = _stub_stage(timeout_s=1,
                        cmd=["{py}", "-c", "import time; time.sleep(10)"])
    outcome = suite.run_stage(stage, py=sys.executable, repo=tmp_path,
                              results_dir=results_dir, logs_dir=logs_dir)
    assert outcome.status == "timeout"


# --- CLI ------------------------------------------------------------

def test_main_default_lists_all_stages_and_does_not_run(tmp_path, capsys):
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"
    custom_stages = [_stub_stage(name="a"), _stub_stage(name="b")]

    rc = suite.main([], stages=custom_stages, results_dir=results_dir,
                    logs_dir=logs_dir, py=sys.executable, repo=tmp_path)
    out = capsys.readouterr().out
    assert rc == 0
    assert "a" in out and "b" in out
    assert not logs_dir.exists()


def test_main_dry_run_lists_plan_without_executing(tmp_path, capsys):
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"
    custom_stages = [_stub_stage(name="a")]

    rc = suite.main(["--stages", "a", "--dry-run"], stages=custom_stages,
                    results_dir=results_dir, logs_dir=logs_dir,
                    py=sys.executable, repo=tmp_path)
    out = capsys.readouterr().out
    assert rc == 0
    assert "RUN" in out
    assert not logs_dir.exists()


def test_main_runs_selected_stages(tmp_path, capsys):
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"
    custom_stages = [_stub_stage(name="a"), _stub_stage(name="b", glob_pat="never*.json")]

    rc = suite.main(["--stages", "a"], stages=custom_stages,
                    results_dir=results_dir, logs_dir=logs_dir,
                    py=sys.executable, repo=tmp_path)
    assert rc == 0
    assert (logs_dir / "a.log").exists()
    assert not (logs_dir / "b.log").exists()


def test_main_unknown_stage_errors(tmp_path, capsys):
    custom_stages = [_stub_stage(name="a")]
    rc = suite.main(["--stages", "nope"], stages=custom_stages,
                    results_dir=tmp_path / "results", logs_dir=tmp_path / "logs",
                    py=sys.executable, repo=tmp_path)
    assert rc == 2


def test_render_stage_table_marks_skip_when_result_present(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "pcfg_set__pure_vsa__seed0.json").write_text("{}", encoding="utf-8")
    table = suite.render_stage_table([_stub_stage()], results_dir=results_dir)
    assert "SKIP" in table
