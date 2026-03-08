from __future__ import annotations

import queue
from pathlib import Path

from alpha_evolve.dashboard.api.job_controller import DashboardJobController
from alpha_evolve.dashboard.api.jobs import JobState
from alpha_evolve.dashboard.api.ml_lab_runtime import (
    parse_progress_line,
    update_activity_from_progress,
)


class _Regex:
    def __init__(self, payload_text: str) -> None:
        self._payload_text = payload_text

    def search(self, line: str):
        if "PROGRESS " not in line:
            return None

        class _Match:
            def __init__(self, payload_text: str) -> None:
                self._payload_text = payload_text

            def group(self, _index: int) -> str:
                return self._payload_text

        return _Match(self._payload_text)


def test_parse_progress_line_extracts_json_payload() -> None:
    payload = parse_progress_line('PROGRESS {"type":"ml_complete"}', _Regex('{"type":"ml_complete"}'))
    assert payload == {"type": "ml_complete"}


def test_update_activity_from_progress_tracks_best_sharpe() -> None:
    state = JobState()
    controller = DashboardJobController(state)
    state.init_activity("job-1", {"status": "running", "progress": {}, "sharpe_best": 0.4})

    update_activity_from_progress(
        controller=controller,
        job_id="job-1",
        payload={
            "type": "ml_model_end",
            "model_label": "HistGBM",
            "variant": "baseline",
            "sharpe": 0.8,
        },
    )

    activity = state.get_activity("job-1")
    assert activity is not None
    assert activity["sharpe_best"] == 0.8
    assert activity["last_message"] == "Finished HistGBM (baseline)"
