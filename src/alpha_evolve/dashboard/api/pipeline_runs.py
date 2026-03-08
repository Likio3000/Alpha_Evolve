from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_evolve.utils.run_artifacts import select_backtest_summary_csv

from . import helpers as dashboard_helpers


@dataclass(frozen=True)
class PipelineRunSummary:
    path: str
    name: str
    label: str | None
    sharpe_best: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "label": self.label,
            "sharpe_best": self.sharpe_best,
        }


@dataclass(frozen=True)
class LastPipelineRunSummary:
    run_dir: str | None
    sharpe_best: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "sharpe_best": self.sharpe_best,
        }


@dataclass(frozen=True)
class PipelineRunAssetsListing:
    items: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {"items": self.items}


@dataclass(frozen=True)
class PipelineRunAssetTarget:
    run_dir: Path
    file_path: Path


@dataclass(frozen=True)
class DashboardJobLogPayload:
    log: str

    def to_dict(self) -> dict[str, str]:
        return {"log": self.log}


@dataclass(frozen=True)
class DashboardJobStatusPayload:
    exists: bool
    running: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "exists": self.exists,
            "running": self.running,
        }


@dataclass(frozen=True)
class PipelineBacktestSummaryRow:
    alpha_id: str | None
    timeseries_name: str
    timeseries_file: str | None
    sharpe: float
    ann_return: float
    ann_vol: float
    max_dd: float
    turnover: float
    ops: str | None
    original_metric: float
    program: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "AlphaID": self.alpha_id,
            "TS": self.timeseries_name,
            "TimeseriesFile": self.timeseries_file,
            "Sharpe": self.sharpe,
            "AnnReturn": self.ann_return,
            "AnnVol": self.ann_vol,
            "MaxDD": self.max_dd,
            "Turnover": self.turnover,
            "Ops": self.ops,
            "OriginalMetric": self.original_metric,
            "Program": self.program,
        }


@dataclass(frozen=True)
class PipelineTimeseriesPayload:
    date: list[str]
    equity: list[float | None]
    ret_net: list[float | None]
    pending: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "date": self.date,
            "equity": self.equity,
            "ret_net": self.ret_net,
        }
        if self.pending:
            payload["pending"] = True
        return payload


@dataclass(frozen=True)
class PipelineRunLabelUpdate:
    ok: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {"ok": self.ok}


@dataclass(frozen=True)
class PipelineRunDetails:
    path: str
    name: str
    label: str | None
    sharpe_best: float | None
    summary: Any | None = None
    ui_context: Any | None = None
    meta: dict[str, Any] | None = None
    baseline_metrics: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "name": self.name,
            "label": self.label,
            "sharpe_best": self.sharpe_best,
        }
        if self.summary is not None:
            payload["summary"] = self.summary
        if self.ui_context is not None:
            payload["ui_context"] = self.ui_context
        if self.meta:
            payload["meta"] = self.meta
        if self.baseline_metrics is not None:
            payload["baseline_metrics"] = self.baseline_metrics
        return payload


def pipeline_run_labels_path() -> Path:
    return dashboard_helpers.PIPELINE_DIR / ".run_labels.json"


def load_pipeline_run_labels() -> dict[str, str]:
    path = pipeline_run_labels_path()
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_pipeline_run_labels(labels: dict[str, str]) -> None:
    path = pipeline_run_labels_path()
    path.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_pipeline_runs(*, pipeline_dir: Path | None = None) -> list[Path]:
    resolved_pipeline_dir = pipeline_dir or dashboard_helpers.PIPELINE_DIR
    runs = [path for path in resolved_pipeline_dir.glob("run_*") if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return runs


def format_pipeline_run_path_for_ui(
    path: Path,
    *,
    root_dir: Path | None = None,
    pipeline_dir: Path | None = None,
) -> str:
    resolved_root_dir = root_dir or dashboard_helpers.ROOT
    resolved_pipeline_dir = pipeline_dir or dashboard_helpers.PIPELINE_DIR
    try:
        return str(path.relative_to(resolved_root_dir))
    except ValueError:
        try:
            return str(path.relative_to(resolved_pipeline_dir))
        except ValueError:
            return str(path)


def safe_read_pipeline_run_json(path: Path) -> Any | None:
    try:
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def read_pipeline_run_codename(run_dir: Path) -> str | None:
    meta = safe_read_pipeline_run_json(run_dir / "meta" / "run_metadata.json")
    if isinstance(meta, dict):
        codename = meta.get("codename")
        if isinstance(codename, str):
            codename = codename.strip()
            if codename:
                return codename
    return None


def resolve_pipeline_run_dir(
    run_dir: str,
    *,
    pipeline_dir: Path | None = None,
    root_dir: Path | None = None,
) -> Path:
    candidate = Path(run_dir)
    base_candidates: list[Path] = []
    resolved_pipeline_dir = pipeline_dir or dashboard_helpers.PIPELINE_DIR
    resolved_root_dir = root_dir or dashboard_helpers.ROOT
    pipeline_root = resolved_pipeline_dir.resolve()
    if candidate.is_absolute():
        base_candidates.append(candidate.resolve())
    else:
        stripped = candidate
        if stripped.parts and stripped.parts[0] == pipeline_root.name:
            stripped = Path(*stripped.parts[1:]) if len(stripped.parts) > 1 else Path(".")
        base_candidates.append((pipeline_root / stripped).resolve())
        base_candidates.append((resolved_pipeline_dir.parent / candidate).resolve())
        base_candidates.append((resolved_root_dir / candidate).resolve())
    for resolved in base_candidates:
        try:
            resolved.relative_to(pipeline_root)
        except ValueError:
            continue
        if (
            resolved.exists()
            and resolved.is_dir()
            and resolved != pipeline_root
            and resolved.name.startswith("run_")
        ):
            return resolved
    raise ValueError("run_dir must resolve under pipeline_runs_cs/")


def pipeline_backtest_summary_csv(run_dir: Path) -> Path | None:
    backtest_dir = run_dir / "backtest_portfolio_csvs"
    if not backtest_dir.exists():
        return None
    return select_backtest_summary_csv(backtest_dir)


def build_pipeline_runs_listing(*, limit: int, labels: dict[str, str] | None = None) -> list[dict[str, Any]]:
    resolved_labels = labels or load_pipeline_run_labels()
    items: list[dict[str, Any]] = []
    for path in find_pipeline_runs()[:limit]:
        sharpe = dashboard_helpers.read_best_sharpe_from_run(path)
        items.append(
            PipelineRunSummary(
                path=format_pipeline_run_path_for_ui(path),
                name=read_pipeline_run_codename(path) or path.name,
                label=resolved_labels.get(path.name),
                sharpe_best=None if sharpe is None else float(sharpe),
            ).to_dict()
        )
    return items


def build_last_pipeline_run_summary() -> dict[str, Any]:
    run_dir = dashboard_helpers.resolve_latest_run_dir()
    if run_dir is None:
        return LastPipelineRunSummary(run_dir=None, sharpe_best=None).to_dict()
    sharpe = dashboard_helpers.read_best_sharpe_from_run(run_dir)
    return LastPipelineRunSummary(
        run_dir=format_pipeline_run_path_for_ui(run_dir),
        sharpe_best=None if sharpe is None else float(sharpe),
    ).to_dict()


def build_job_log_payload(*, log_text: str) -> dict[str, str]:
    return DashboardJobLogPayload(log=log_text).to_dict()


def build_job_status_payload(*, handle: Any | None) -> dict[str, bool]:
    if handle is None:
        return DashboardJobStatusPayload(exists=False, running=False).to_dict()
    return DashboardJobStatusPayload(exists=True, running=bool(handle.is_running())).to_dict()


def build_pipeline_backtest_summary_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _float_or_none(row: dict[str, str], key: str) -> float | None:
        try:
            return float(row.get(key, ""))
        except Exception:
            return None

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                PipelineBacktestSummaryRow(
                    alpha_id=row.get("AlphaID") or row.get("alpha_id"),
                    timeseries_name=Path(row.get("TS") or row.get("TimeseriesFile") or "").name,
                    timeseries_file=row.get("TimeseriesFile") or row.get("TS"),
                    sharpe=_float_or_none(row, "Sharpe") or 0.0,
                    ann_return=_float_or_none(row, "AnnReturn") or 0.0,
                    ann_vol=_float_or_none(row, "AnnVol") or 0.0,
                    max_dd=_float_or_none(row, "MaxDD") or 0.0,
                    turnover=_float_or_none(row, "Turnover") or 0.0,
                    ops=row.get("Ops"),
                    original_metric=_float_or_none(row, "OriginalMetric")
                    or _float_or_none(row, "original_metric")
                    or _float_or_none(row, "IC")
                    or 0.0,
                    program=row.get("Program") or row.get("PROGRAM"),
                ).to_dict()
            )
    return rows


def resolve_pipeline_timeseries_file(
    *,
    run_dir: Path,
    file: str | None,
    alpha_id: str | None,
) -> Path:
    backtest_dir = run_dir / "backtest_portfolio_csvs"
    if file:
        return backtest_dir / Path(file).name
    summary = pipeline_backtest_summary_csv(run_dir)
    if summary and alpha_id:
        try:
            with summary.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    resolved_alpha_id = row.get("AlphaID") or row.get("alpha_id")
                    if str(resolved_alpha_id) == str(alpha_id):
                        timeseries_name = row.get("TS") or row.get("TimeseriesFile")
                        if timeseries_name:
                            return backtest_dir / Path(timeseries_name).name
        except Exception:
            pass
    raise FileNotFoundError("Timeseries not found")


def build_pending_pipeline_timeseries_payload() -> dict[str, Any]:
    return PipelineTimeseriesPayload(date=[], equity=[], ret_net=[], pending=True).to_dict()


def build_pipeline_timeseries_payload(ts_path: Path) -> dict[str, Any]:
    dates: list[str] = []
    equity: list[float | None] = []
    ret_net: list[float | None] = []
    with ts_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            dates.append(str(row.get("date")))
            try:
                value = float(row.get("equity", "nan"))
                equity.append(None if value != value else value)
            except Exception:
                equity.append(None)
            try:
                value = float(row.get("ret_net", "nan"))
                ret_net.append(None if value != value else value)
            except Exception:
                ret_net.append(None)
    return PipelineTimeseriesPayload(date=dates, equity=equity, ret_net=ret_net).to_dict()


def resolve_pipeline_run_label_target(
    path: str,
    *,
    root_dir: Path | None = None,
    pipeline_dir: Path | None = None,
) -> Path:
    resolved_root_dir = root_dir or dashboard_helpers.ROOT
    resolved_pipeline_dir = pipeline_dir or dashboard_helpers.PIPELINE_DIR
    resolved_path = (resolved_root_dir / path).resolve()
    try:
        resolved_path.relative_to(resolved_pipeline_dir.resolve())
    except Exception as exc:
        raise ValueError("Path must be under pipeline_runs_cs/") from exc
    if not resolved_path.exists():
        raise FileNotFoundError("Run path not found")
    return resolved_path


def update_pipeline_run_label(*, path: str, label: str) -> dict[str, bool]:
    target = resolve_pipeline_run_label_target(path)
    labels = load_pipeline_run_labels()
    if label:
        labels[target.name] = label
    else:
        labels.pop(target.name, None)
    save_pipeline_run_labels(labels)
    return PipelineRunLabelUpdate().to_dict()


def resolve_pipeline_run_asset_path(
    *,
    run_path: Path,
    file: str,
    sub: str | None,
) -> Path:
    file_path = Path(file)
    if file_path.is_absolute():
        raise ValueError("file must be relative")
    if sub and file_path.parent == Path("."):
        file_path = Path(sub) / file_path
    resolved = (run_path / file_path).resolve()
    try:
        resolved.relative_to(run_path)
    except Exception as exc:
        raise ValueError("Invalid file path") from exc
    if not resolved.exists():
        raise FileNotFoundError("File not found")
    return resolved


def list_pipeline_run_assets(*, resolved_run_dir: Path, prefix: str, limit: int) -> dict[str, list[str]]:
    base = resolved_run_dir
    if prefix:
        candidate = (resolved_run_dir / prefix).resolve()
        try:
            candidate.relative_to(resolved_run_dir)
        except Exception as exc:
            raise ValueError("prefix must stay within run_dir") from exc
        base = candidate

    if not base.exists():
        return PipelineRunAssetsListing(items=[]).to_dict()

    allowed = {".csv", ".json", ".jsonl", ".log", ".md", ".png", ".txt"}
    items: list[str] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        try:
            rel = path.relative_to(resolved_run_dir)
        except ValueError:
            continue
        items.append(str(rel))
    items.sort()
    return PipelineRunAssetsListing(items=items[:limit]).to_dict()


def build_pipeline_run_details(*, resolved_run_dir: Path, labels: dict[str, str] | None = None) -> dict[str, Any]:
    sharpe = dashboard_helpers.read_best_sharpe_from_run(resolved_run_dir)
    resolved_labels = labels or load_pipeline_run_labels()

    summary = safe_read_pipeline_run_json(resolved_run_dir / "SUMMARY.json")
    meta_dir = resolved_run_dir / "meta"
    ui_context = None
    extra_meta: dict[str, Any] = {}
    if meta_dir.exists():
        ui_context = safe_read_pipeline_run_json(meta_dir / "ui_context.json")
        for key, filename in (
            ("evolution_config", "evolution_config.json"),
            ("backtest_config", "backtest_config.json"),
            ("run_metadata", "run_metadata.json"),
            ("data_alignment", "data_alignment.json"),
        ):
            data = safe_read_pipeline_run_json(meta_dir / filename)
            if data is not None:
                extra_meta[key] = data

    baseline_metrics = safe_read_pipeline_run_json(resolved_run_dir / "baseline_metrics.json")
    if baseline_metrics is None:
        backtest_config = extra_meta.get("backtest_config")
        if isinstance(backtest_config, dict):
            data_dir = backtest_config.get("data_dir")
            if isinstance(data_dir, str) and data_dir:
                base_path = Path(data_dir)
                if not base_path.is_absolute():
                    base_path = (dashboard_helpers.ROOT / base_path).resolve()
                baseline_metrics = safe_read_pipeline_run_json(base_path / "baseline_metrics.json")

    return PipelineRunDetails(
        path=format_pipeline_run_path_for_ui(resolved_run_dir),
        name=read_pipeline_run_codename(resolved_run_dir) or resolved_run_dir.name,
        label=resolved_labels.get(resolved_run_dir.name),
        sharpe_best=None if sharpe is None else float(sharpe),
        summary=summary,
        ui_context=ui_context,
        meta=extra_meta or None,
        baseline_metrics=baseline_metrics,
    ).to_dict()
