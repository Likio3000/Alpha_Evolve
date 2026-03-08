from __future__ import annotations

from pathlib import Path


def select_backtest_summary_csv(backtest_dir: Path) -> Path | None:
    candidates = sorted(backtest_dir.glob("backtest_summary_top*.csv"))
    if not candidates:
        return None

    def _rank(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        top_n = int(digits) if digits else -1
        return (top_n, path.name)

    return max(candidates, key=_rank)


def resolve_latest_run_dir(pipeline_dir: Path, *, project_root: Path | None = None) -> Path | None:
    latest = pipeline_dir / "LATEST"
    try:
        if latest.exists():
            raw = latest.read_text(encoding="utf-8").strip()
            if raw:
                raw_path = Path(raw).expanduser()
                candidates: list[Path] = []
                if raw_path.is_absolute():
                    candidates.append(raw_path.resolve())
                else:
                    if project_root is not None:
                        candidates.append((project_root / raw_path).resolve())
                    candidates.append((pipeline_dir / raw_path).resolve())
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
    except Exception:
        pass

    runs = [path for path in pipeline_dir.glob("run_*") if path.is_dir()]
    if not runs:
        return None

    def _rank(path: Path) -> tuple[int, str]:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = -1
        return (mtime_ns, path.name)

    return max(runs, key=_rank)
