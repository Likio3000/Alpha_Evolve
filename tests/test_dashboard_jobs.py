from __future__ import annotations

from alpha_evolve.dashboard.api.jobs import JobState


def test_initialize_job_sets_queue_meta_and_activity() -> None:
    state = JobState()
    job_id, queue = state.initialize_job(
        job_id="job-1",
        meta={"payload": {"generations": 1}},
        activity={"status": "running"},
    )
    assert job_id == "job-1"
    assert state.get_queue(job_id) is queue
    assert state.meta[job_id]["payload"]["generations"] == 1
    assert state.get_activity(job_id)["status"] == "running"


def test_touch_activity_adds_updated_at() -> None:
    state = JobState()
    state.init_activity("job-2", {"status": "running"})
    activity = state.touch_activity("job-2", last_message="hello")
    assert activity["last_message"] == "hello"
    assert "updated_at" in activity


def test_append_meta_sequence_enforces_limit() -> None:
    state = JobState()
    for value in [1, 2, 3]:
        state.append_meta_sequence("job-3", "gen_history", {"generation": value}, limit=2)
    history = state.meta["job-3"]["gen_history"]
    assert history == [{"generation": 2}, {"generation": 3}]
