from __future__ import annotations

from datetime import datetime, timezone
import json

from alpha_evolve.dashboard.api.job_controller import DashboardJobController
from alpha_evolve.dashboard.api.jobs import JobState
from alpha_evolve.dashboard.api.ml_lab_runtime import MLLabLaunchRequest, launch_ml_lab_job
from alpha_evolve.dashboard.api.pipeline_runtime import initialize_pipeline_job


def test_initialize_pipeline_job_returns_context_and_records_log_path(tmp_path) -> None:
    state = JobState()
    controller = DashboardJobController(state)

    job = initialize_pipeline_job(
        controller=controller,
        root_dir=tmp_path,
        payload_dict={"generations": 3},
        full_args=["3", "--dataset", "sp500"],
        now_time=lambda: 123.0,
        now_datetime=lambda: datetime(2026, 3, 8, 5, 0, 0, tzinfo=timezone.utc),
    )

    assert job.job_id
    assert job.client_queue is controller.get_queue(job.job_id)
    assert job.log_path == tmp_path / "logs" / f"pipeline_{job.job_id}.log"

    activity = controller.get_activity(job.job_id)
    assert activity is not None
    assert activity["status"] == "running"
    assert activity["log_path"] == str(job.log_path)

    meta = state.meta[job.job_id]
    assert meta["payload"] == {"generations": 3}
    assert meta["pipeline_args"] == ["3", "--dataset", "sp500"]
    assert meta["submitted_at"] == "2026-03-08T05:00:00Z"

    queued = json.loads(job.client_queue.get_nowait())
    assert queued == {"type": "status", "msg": "started", "args": ["3", "--dataset", "sp500"]}


def test_launch_ml_lab_job_writes_spec_and_starts_runtime(tmp_path, monkeypatch) -> None:
    state = JobState()
    controller = DashboardJobController(state)
    run_dir = tmp_path / "runs" / "ml-run"
    spec_path = run_dir / "ml_spec.json"

    started = {}

    class DummyProc:
        pass

    proc = DummyProc()

    def fake_start_text_subprocess(*, cmd, cwd, env):  # noqa: ANN001
        started["cmd"] = cmd
        started["cwd"] = cwd
        started["env"] = env
        return proc

    class FakeThread:
        def __init__(self, *, target, kwargs, daemon):  # noqa: ANN001
            started["target"] = target
            started["kwargs"] = kwargs
            started["daemon"] = daemon

        def start(self) -> None:
            started["thread_started"] = True

    monkeypatch.setattr(
        "alpha_evolve.dashboard.api.ml_lab_runtime.start_text_subprocess",
        fake_start_text_subprocess,
    )
    monkeypatch.setattr(
        "alpha_evolve.dashboard.api.ml_lab_runtime.threading.Thread",
        FakeThread,
    )

    launch = launch_ml_lab_job(
        MLLabLaunchRequest(
            controller=controller,
            root_dir=tmp_path,
            run_dir=run_dir,
            run_dir_label="pipeline_runs_cs/ml_runs/ml-run",
            spec_payload={"dataset": "sp500", "seed": 7},
            spec_path=spec_path,
            progress_re=object(),
            cleanup_delay_seconds=300.0,
        ),
        now_time=lambda: 42.0,
    )

    assert launch.proc is proc
    assert launch.run_dir == run_dir
    assert launch.spec_path == spec_path
    assert launch.log_path == tmp_path / "logs" / f"ml_lab_{launch.job_id}.log"

    assert json.loads(spec_path.read_text(encoding="utf-8")) == {"dataset": "sp500", "seed": 7}
    assert started["cmd"][-2:] == ["--out_dir", str(run_dir)]
    assert started["cwd"] == tmp_path
    assert started["env"]["PYTHONPATH"].startswith(str(tmp_path / "src"))
    assert started["thread_started"] is True
    assert started["kwargs"]["job_id"] == launch.job_id
    assert started["kwargs"]["proc"] is proc
    assert started["kwargs"]["log_path"] == launch.log_path

    activity = controller.get_activity(launch.job_id)
    assert activity is not None
    assert activity["status"] == "running"
    assert activity["run_dir"] == "pipeline_runs_cs/ml_runs/ml-run"
    assert activity["log_path"] == str(launch.log_path)

    handle = controller.get_handle(launch.job_id)
    assert handle is not None
    assert handle.proc is proc
