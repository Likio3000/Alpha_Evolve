from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

from django.http import HttpRequest, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt
from pydantic import ValidationError

from alpha_evolve.cli.pipeline import PipelineOptions
from alpha_evolve.experiments import ExperimentRegistry, ExperimentSessionRunner, ExperimentSessionSpec
from alpha_evolve.self_play import load_base_configs

from ..helpers import PIPELINE_DIR, ROOT
from ..http import json_error, json_response
from ..models import ExperimentProposalDecisionRequest, ExperimentStartRequest

LOGGER = logging.getLogger(__name__)


def _resolve_experiments_db() -> Path:
    override = os.environ.get("AE_EXPERIMENTS_DB")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate
    return (ROOT / "artifacts" / "experiments" / "experiments.sqlite").resolve()


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    out = out.strip()
    return out or None


def _fingerprint_dataset(path: str | None) -> str | None:
    import hashlib

    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        return None

    h = hashlib.sha256()
    if p.is_file():
        try:
            st = p.stat()
            h.update(p.name.encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
            h.update(str(int(st.st_mtime)).encode("utf-8"))
        except Exception:
            return None
        return h.hexdigest()

    files = sorted([f for f in p.glob("*.csv") if f.is_file()])
    if not files:
        return None
    for f in files:
        try:
            st = f.stat()
        except Exception:
            continue
        h.update(f.name.encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()


def _to_toml(obj: dict[str, dict[str, Any]]) -> str:
    def _fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v)
        s = s.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'"{s}"'

    lines: list[str] = []
    for section in ("evolution", "backtest"):
        payload = obj.get(section)
        if not payload:
            continue
        lines.append(f"[{section}]")
        for key in sorted(payload.keys()):
            lines.append(f"{key} = {_fmt(payload[key])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class _ExperimentHandle:
    def __init__(self, task: asyncio.Task, stop_event: threading.Event) -> None:
        self.task = task
        self.stop_event = stop_event


_RUNNING: dict[str, _ExperimentHandle] = {}


def _registry() -> ExperimentRegistry:
    return ExperimentRegistry(_resolve_experiments_db())


def _list_search_spaces() -> list[str]:
    base = (ROOT / "configs" / "self_evolution").resolve()
    if not base.exists():
        return []
    paths: list[str] = []
    for suffix in (".json", ".toml", ".tml"):
        for p in sorted(base.glob(f"*{suffix}")):
            try:
                paths.append(str(p.relative_to(ROOT)))
            except ValueError:
                paths.append(str(p))
    return paths


def search_spaces(_request: HttpRequest):
    return json_response({"items": _list_search_spaces()})


@csrf_exempt
async def start_session(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    try:
        payload_data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return json_error("Invalid JSON body", 400)
    try:
        payload = ExperimentStartRequest.model_validate(payload_data)
    except ValidationError as exc:
        return json_response({"detail": exc.errors()}, status=422)

    req = payload.model_dump()
    search_space_raw = str(req["search_space"])
    search_space_path = Path(search_space_raw)
    if not search_space_path.is_absolute():
        search_space_path = (ROOT / search_space_path).resolve()
    else:
        search_space_path = search_space_path.resolve()
    if not search_space_path.exists():
        return json_error(f"Search space not found: {search_space_raw}", 404)
    search_space = str(search_space_path)

    base_config = req.get("config")
    if base_config:
        base_path = Path(str(base_config))
        if not base_path.is_absolute():
            base_path = (ROOT / base_path).resolve()
        else:
            base_path = base_path.resolve()
        if not base_path.exists():
            return json_error(f"Config not found: {base_config}", 404)
        base_config = str(base_path)

    try:
        evo_cfg, _bt_cfg = load_base_configs(base_config)
    except Exception as exc:
        return json_error(f"Failed to load base config: {exc}", 400)

    dataset_dir = str(evo_cfg.data_dir)
    dataset_hash = _fingerprint_dataset(dataset_dir)
    git_sha = _git_sha()

    pipeline_output_dir = req.get("pipeline_output_dir") or str(PIPELINE_DIR)
    out_path = Path(str(pipeline_output_dir))
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    else:
        out_path = out_path.resolve()
    pipeline_output_dir = str(out_path)
    opts = PipelineOptions(
        debug_prints=bool(req.get("debug_prints")),
        run_baselines=bool(req.get("run_baselines")),
        retrain_baselines=bool(req.get("retrain_baselines")),
        log_level=str(req.get("pipeline_log_level") or "INFO"),
        log_file=req.get("pipeline_log_file"),
        output_dir=str(pipeline_output_dir),
        disable_align_cache=bool(req.get("disable_align_cache")),
        align_cache_dir=req.get("align_cache_dir"),
    )

    spec = ExperimentSessionSpec(
        search_space_path=search_space,
        base_config_path=base_config,
        max_iterations=int(req.get("iterations", 5)),
        seed=int(req.get("seed", 0)),
        objective_metric=str(req.get("objective") or "Sharpe"),
        maximize=not bool(req.get("minimize", False)),
        exploration_probability=float(req.get("exploration_prob", 0.35)),
        auto_approve=bool(req.get("auto_approve", False)),
        approval_poll_interval=float(req.get("approval_poll_interval", 5.0)),
        approval_timeout=float(req.get("approval_timeout")) if req.get("approval_timeout") is not None else None,
        corr_gate_sharpe=float(req.get("corr_gate_sharpe", 1.0)),
        sharpe_close_epsilon=float(req.get("sharpe_close_epsilon", 0.05)),
        max_sharpe_sacrifice=float(req.get("max_sharpe_sacrifice", 0.05)),
        min_corr_improvement=float(req.get("min_corr_improvement", 0.05)),
        dataset_dir=dataset_dir,
        dataset_hash=dataset_hash,
        git_sha=git_sha,
    )

    registry = _registry()
    stop_event = threading.Event()
    runner = ExperimentSessionRunner(
        registry=registry,
        spec=spec,
        pipeline_options=opts,
        stop_flag=stop_event.is_set,
    )
    session_id = runner.session_id

    async def _run() -> None:
        try:
            await asyncio.to_thread(runner.run)
        except Exception:
            LOGGER.exception("Experiment session failed: %s", session_id)
        finally:
            handle = _RUNNING.pop(session_id, None)
            if handle and not handle.task.done():
                # Task is this coroutine, so it should already be done, but keep it safe.
                pass

    task = asyncio.create_task(_run())
    _RUNNING[session_id] = _ExperimentHandle(task=task, stop_event=stop_event)

    return json_response({"session_id": session_id, "db_path": str(_resolve_experiments_db())})


def list_sessions(request: HttpRequest):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    limit_param = request.GET.get("limit", "50")
    try:
        limit = max(1, min(1000, int(limit_param)))
    except Exception:
        return json_error("limit must be an integer", 400)
    reg = _registry()
    sessions = reg.list_sessions(limit=limit)
    for sess in sessions:
        sid = str(sess.get("session_id"))
        sess["running_task"] = sid in _RUNNING
    return json_response({"items": sessions, "db_path": str(_resolve_experiments_db())})


def get_session(request: HttpRequest, session_id: str):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    reg = _registry()
    sess = reg.get_session(session_id)
    if sess is None:
        return json_error("Unknown session id", 404)
    sess["running_task"] = session_id in _RUNNING
    return json_response(sess)


def list_iterations(request: HttpRequest, session_id: str):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    reg = _registry()
    sess = reg.get_session(session_id)
    if sess is None:
        return json_error("Unknown session id", 404)
    return json_response({"items": reg.list_iterations(session_id)})


def list_proposals(request: HttpRequest, session_id: str):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    reg = _registry()
    sess = reg.get_session(session_id)
    if sess is None:
        return json_error("Unknown session id", 404)
    return json_response({"items": reg.list_proposals(session_id, limit=200)})


@csrf_exempt
async def decide_proposal(request: HttpRequest, session_id: str, proposal_id: int):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    reg = _registry()
    sess = reg.get_session(session_id)
    if sess is None:
        return json_error("Unknown session id", 404)

    try:
        payload_data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return json_error("Invalid JSON body", 400)
    try:
        payload = ExperimentProposalDecisionRequest.model_validate(payload_data)
    except ValidationError as exc:
        return json_response({"detail": exc.errors()}, status=422)

    decision = payload.decision.strip().lower()
    if decision not in {"approved", "rejected"}:
        return json_error("decision must be approved|rejected", 400)
    try:
        reg.decide_proposal(
            session_id=session_id,
            proposal_id=int(proposal_id),
            decision=decision,
            decided_by=payload.decided_by,
            notes=payload.notes,
        )
    except ValueError as exc:
        return json_error(str(exc), 400)
    return json_response({"ok": True})


@csrf_exempt
async def stop_session(request: HttpRequest, session_id: str):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    reg = _registry()
    sess = reg.get_session(session_id)
    if sess is None:
        return json_error("Unknown session id", 404)
    handle = _RUNNING.get(session_id)
    if handle is None:
        reg.touch_session(session_id, status="stopped")
        return json_response({"stopped": True, "running_task": False})
    handle.stop_event.set()
    reg.touch_session(session_id, status="stop_requested")
    return json_response({"stopped": True, "running_task": True})


@csrf_exempt
def export_best_config(request: HttpRequest, session_id: str):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    reg = _registry()
    sess = reg.get_session(session_id)
    if sess is None:
        return json_error("Unknown session id", 404)

    best_id = sess.get("best_iteration_id")
    if best_id is None:
        return json_error("Session has no best iteration yet", 409)
    iteration = reg.get_iteration_by_id(session_id, int(best_id))
    if iteration is None:
        return json_error("Best iteration missing", 404)

    evo = iteration.get("evolution_json") or {}
    bt = iteration.get("backtest_json") or {}
    gens = evo.get("generations") if isinstance(evo, dict) else None
    try:
        generations = int(gens) if gens is not None else 5
    except Exception:
        generations = 5

    out_dir = (PIPELINE_DIR / "experiments" / session_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_config.toml"
    out_path.write_text(_to_toml({"evolution": evo, "backtest": bt}), encoding="utf-8")
    try:
        rel = out_path.relative_to(ROOT)
        out_str = str(rel)
    except ValueError:
        out_str = str(out_path)

    return json_response({"config_path": out_str, "generations": generations})
