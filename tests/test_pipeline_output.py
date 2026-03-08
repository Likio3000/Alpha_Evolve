from __future__ import annotations

from alpha_evolve.dashboard.api.pipeline_output import (
    line_to_pipeline_event,
    resolve_run_dir_hint,
)


def test_line_to_pipeline_event_strips_ansi_sequences() -> None:
    line = "\x1b[32mSharpe(best) = 1.25\x1b[0m"
    event = line_to_pipeline_event(line)
    assert event["type"] == "score"
    assert event["sharpe_best"] == 1.25
    assert "\x1b" not in event["raw"]


def test_line_to_pipeline_event_normalizes_nonfinite_values() -> None:
    line = 'PROGRESS {"type":"gen_progress","score":NaN,"limit":Infinity,"floor":-Infinity}'
    event = line_to_pipeline_event(line)
    assert event["type"] == "progress"
    assert event["data"]["score"] is None
    assert event["data"]["limit"] is None
    assert event["data"]["floor"] is None


def test_resolve_run_dir_hint_finds_output_paths() -> None:
    assert resolve_run_dir_hint("artefacts in artifacts/pipeline_runs_cs/run_demo") == "artifacts/pipeline_runs_cs/run_demo"
    assert resolve_run_dir_hint("outputs -> pipeline_runs_cs/run_demo") == "pipeline_runs_cs/run_demo"
