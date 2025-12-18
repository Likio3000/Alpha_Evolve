from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True)


def _json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return None
    return json.loads(value)


class ExperimentRegistry:
    """SQLite-backed registry for self-evolution experiment sessions."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    @contextmanager
    def connect(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  session_id TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  status TEXT NOT NULL,
                  base_config_path TEXT,
                  search_space_path TEXT,
                  max_iterations INTEGER NOT NULL,
                  seed INTEGER NOT NULL,
                  exploration_probability REAL NOT NULL,
                  objective_metric TEXT NOT NULL,
                  maximize INTEGER NOT NULL,
                  corr_gate_sharpe REAL NOT NULL,
                  sharpe_close_epsilon REAL NOT NULL,
                  max_sharpe_sacrifice REAL NOT NULL,
                  min_corr_improvement REAL NOT NULL,
                  dataset_dir TEXT,
                  dataset_hash TEXT,
                  git_sha TEXT,
                  best_iteration_id INTEGER,
                  best_sharpe REAL,
                  best_corr REAL,
                  last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS iterations (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                  iteration_index INTEGER NOT NULL,
                  started_at TEXT NOT NULL,
                  finished_at TEXT,
                  status TEXT NOT NULL,
                  run_dir TEXT,
                  summary_path TEXT,
                  updates_json TEXT,
                  evolution_json TEXT,
                  backtest_json TEXT,
                  pipeline_options_json TEXT,
                  metrics_json TEXT,
                  objective_sharpe REAL,
                  objective_corr REAL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_iterations_session_iter ON iterations(session_id, iteration_index);
                CREATE TABLE IF NOT EXISTS proposals (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                  iteration_completed INTEGER NOT NULL,
                  next_iteration INTEGER NOT NULL,
                  created_at TEXT NOT NULL,
                  decided_at TEXT,
                  status TEXT NOT NULL,
                  decided_by TEXT,
                  notes TEXT,
                  proposed_updates_json TEXT NOT NULL,
                  base_snapshot_json TEXT,
                  candidate_snapshot_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_proposals_session_status ON proposals(session_id, status);
                """
            )

    def create_session(
        self,
        *,
        base_config_path: str | None,
        search_space_path: str,
        max_iterations: int,
        seed: int,
        exploration_probability: float,
        objective_metric: str,
        maximize: bool,
        corr_gate_sharpe: float,
        sharpe_close_epsilon: float,
        max_sharpe_sacrifice: float,
        min_corr_improvement: float,
        dataset_dir: str | None,
        dataset_hash: str | None,
        git_sha: str | None,
    ) -> str:
        session_id = str(uuid.uuid4())
        now = _utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                  session_id, created_at, updated_at, status,
                  base_config_path, search_space_path,
                  max_iterations, seed, exploration_probability,
                  objective_metric, maximize,
                  corr_gate_sharpe, sharpe_close_epsilon, max_sharpe_sacrifice, min_corr_improvement,
                  dataset_dir, dataset_hash, git_sha
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    now,
                    now,
                    "running",
                    base_config_path,
                    search_space_path,
                    int(max_iterations),
                    int(seed),
                    float(exploration_probability),
                    str(objective_metric),
                    1 if maximize else 0,
                    float(corr_gate_sharpe),
                    float(sharpe_close_epsilon),
                    float(max_sharpe_sacrifice),
                    float(min_corr_improvement),
                    dataset_dir,
                    dataset_hash,
                    git_sha,
                ),
            )
        return session_id

    def touch_session(self, session_id: str, *, status: str | None = None, last_error: str | None = None) -> None:
        updates: dict[str, Any] = {"updated_at": _utc_now_iso()}
        if status is not None:
            updates["status"] = status
        if last_error is not None:
            updates["last_error"] = last_error

        assignments = ", ".join([f"{k}=?" for k in updates])
        values = list(updates.values())
        values.append(session_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE sessions SET {assignments} WHERE session_id=?", values)

    def set_best(
        self,
        session_id: str,
        *,
        iteration_id: int,
        best_sharpe: float | None,
        best_corr: float | None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET updated_at=?, best_iteration_id=?, best_sharpe=?, best_corr=?
                WHERE session_id=?
                """,
                (_utc_now_iso(), int(iteration_id), best_sharpe, best_corr, session_id),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_iteration(
        self,
        *,
        session_id: str,
        iteration_index: int,
        status: str,
        updates: Mapping[str, Any] | None,
        evolution: Mapping[str, Any] | None,
        backtest: Mapping[str, Any] | None,
        pipeline_options: Mapping[str, Any] | None,
        started_at: str | None = None,
    ) -> int:
        started_at = started_at or _utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO iterations (
                  session_id, iteration_index, started_at, status,
                  updates_json, evolution_json, backtest_json, pipeline_options_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    int(iteration_index),
                    started_at,
                    str(status),
                    _json_dumps(dict(updates or {})),
                    _json_dumps(dict(evolution or {})),
                    _json_dumps(dict(backtest or {})),
                    _json_dumps(dict(pipeline_options or {})),
                ),
            )
            return int(cur.lastrowid)

    def finish_iteration(
        self,
        *,
        iteration_id: int,
        status: str,
        run_dir: str | None,
        summary_path: str | None,
        metrics: Mapping[str, Any] | None,
        objective_sharpe: float | None,
        objective_corr: float | None,
        finished_at: str | None = None,
    ) -> None:
        finished_at = finished_at or _utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE iterations
                SET finished_at=?, status=?, run_dir=?, summary_path=?, metrics_json=?, objective_sharpe=?, objective_corr=?
                WHERE id=?
                """,
                (
                    finished_at,
                    str(status),
                    run_dir,
                    summary_path,
                    _json_dumps(dict(metrics or {})),
                    objective_sharpe,
                    objective_corr,
                    int(iteration_id),
                ),
            )

    def list_iterations(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM iterations WHERE session_id=? ORDER BY iteration_index ASC",
                (session_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("updates_json", "evolution_json", "backtest_json", "pipeline_options_json", "metrics_json"):
                item[key] = _json_loads(item.get(key))
            out.append(item)
        return out

    def get_iteration(self, session_id: str, iteration_index: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM iterations WHERE session_id=? AND iteration_index=?",
                (session_id, int(iteration_index)),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        for key in ("updates_json", "evolution_json", "backtest_json", "pipeline_options_json", "metrics_json"):
            item[key] = _json_loads(item.get(key))
        return item

    def create_proposal(
        self,
        *,
        session_id: str,
        iteration_completed: int,
        next_iteration: int,
        proposed_updates: Mapping[str, Any],
        base_snapshot: Mapping[str, Any] | None,
        candidate_snapshot: Mapping[str, Any] | None,
    ) -> int:
        now = _utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO proposals (
                  session_id, iteration_completed, next_iteration, created_at, status,
                  proposed_updates_json, base_snapshot_json, candidate_snapshot_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    int(iteration_completed),
                    int(next_iteration),
                    now,
                    "pending",
                    _json_dumps(dict(proposed_updates)),
                    _json_dumps(dict(base_snapshot)) if base_snapshot is not None else None,
                    _json_dumps(dict(candidate_snapshot)) if candidate_snapshot is not None else None,
                ),
            )
            proposal_id = int(cur.lastrowid)
        self.touch_session(session_id, status="awaiting_approval")
        return proposal_id

    def list_proposals(self, session_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM proposals WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, max(1, int(limit))),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("proposed_updates_json", "base_snapshot_json", "candidate_snapshot_json"):
                item[key] = _json_loads(item.get(key))
            out.append(item)
        return out

    def get_proposal(self, session_id: str, proposal_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM proposals WHERE session_id=? AND id=?",
                (session_id, int(proposal_id)),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        for key in ("proposed_updates_json", "base_snapshot_json", "candidate_snapshot_json"):
            item[key] = _json_loads(item.get(key))
        return item

    def get_pending_proposal(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM proposals WHERE session_id=? AND status='pending' ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        for key in ("proposed_updates_json", "base_snapshot_json", "candidate_snapshot_json"):
            item[key] = _json_loads(item.get(key))
        return item

    def decide_proposal(
        self,
        *,
        session_id: str,
        proposal_id: int,
        decision: str,
        decided_by: str | None = None,
        notes: str | None = None,
    ) -> None:
        decision = decision.strip().lower()
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be 'approved' or 'rejected'")
        now = _utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE proposals
                SET decided_at=?, status=?, decided_by=?, notes=?
                WHERE session_id=? AND id=?
                """,
                (now, decision, decided_by, notes, session_id, int(proposal_id)),
            )
        self.touch_session(session_id, status="running")

    def get_iteration_by_id(self, session_id: str, iteration_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM iterations WHERE session_id=? AND id=?",
                (session_id, int(iteration_id)),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        for key in ("updates_json", "evolution_json", "backtest_json", "pipeline_options_json", "metrics_json"):
            item[key] = _json_loads(item.get(key))
        return item
