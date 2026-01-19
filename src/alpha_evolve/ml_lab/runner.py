from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from alpha_evolve.backtesting.core import (
    _get_feature_bundle_cached,
    backtest_cross_sectional_alpha,
)
from alpha_evolve.backtesting.data import load_and_align_data_for_backtest
from alpha_evolve.config import BacktestConfig
from alpha_evolve.evolution.data import get_sector_groups
from alpha_evolve.ml_lab.models import build_model, get_model_spec, list_model_specs
from alpha_evolve.programs.types import (
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    SCALAR_FEATURE_NAMES,
)
from alpha_evolve.utils.errors import DataLoadError


_EXCLUDED_FEATURES = {"sector_id_vector"}


@dataclass
class ModelRequest:
    model_id: str
    preset: Optional[str] = None
    params: Optional[Dict[str, Any] | List[Dict[str, Any]]] = None


@dataclass
class ModelVariant:
    model_id: str
    model_label: str
    variant: str
    preset: str
    params: Dict[str, Any]


@dataclass
class MLRunSpec:
    models: List[ModelRequest]
    train_fraction: float = 0.7
    train_points: Optional[int] = None
    test_points: Optional[int] = None
    min_train_points: int = 200
    min_test_points: int = 100
    seed: int = 42
    exclude_features: Optional[List[str]] = None


@dataclass
class _MLSignalProgram:
    predictions: np.ndarray

    def new_state(self) -> Dict[str, Any]:
        return {"ml_step": 0}

    def eval(
        self,
        _features_at_t: Dict[str, Any],
        state: Dict[str, Any],
        n_stocks: int,
    ) -> np.ndarray:
        step = int(state.get("ml_step", 0))
        state["ml_step"] = step + 1
        if step < 0 or step >= self.predictions.shape[0]:
            return np.zeros(n_stocks, dtype=float)
        vec = self.predictions[step]
        if vec.shape[0] != n_stocks:
            return np.zeros(n_stocks, dtype=float)
        return vec


def _feature_names(exclude: Optional[Sequence[str]] = None) -> List[str]:
    excluded = set(_EXCLUDED_FEATURES)
    if exclude:
        excluded.update(exclude)
    return [f for f in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES if f not in excluded]


def _features_to_matrix(
    features_at_t: Dict[str, np.ndarray],
    feature_names: Sequence[str],
    n_stocks: int,
) -> np.ndarray:
    cols: List[np.ndarray] = []
    for name in feature_names:
        vec = features_at_t.get(name)
        if vec is None:
            cols.append(np.zeros(n_stocks, dtype=float))
            continue
        arr = np.asarray(vec, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != n_stocks:
            cols.append(np.zeros(n_stocks, dtype=float))
            continue
        cols.append(arr)
    mat = np.column_stack(cols) if cols else np.zeros((n_stocks, 0), dtype=float)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def _zscore_cross_section(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0, ddof=0)
    std = np.where(std < 1e-9, 1.0, std)
    out = (mat - mean) / std
    return np.clip(out, -6.0, 6.0)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    if np.std(a, ddof=0) < 1e-9 or np.std(b, ddof=0) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _build_ret_fwd_matrix(
    aligned_dfs,
    stock_symbols: Sequence[str],
    time_index: Sequence,
) -> np.ndarray:
    cols = []
    for sym in stock_symbols:
        series = aligned_dfs[sym]["ret_fwd"].loc[time_index].values
        cols.append(np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0))
    return np.column_stack(cols)


def _normalize_model_requests(raw: Any) -> List[ModelRequest]:
    if not raw:
        return []
    if isinstance(raw, list):
        entries = raw
    else:
        raise ValueError("models must be a list of objects")
    out: List[ModelRequest] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each model entry must be an object")
        model_id = str(entry.get("id") or entry.get("model_id") or "").strip()
        if not model_id:
            raise ValueError("Each model entry requires an id")
        preset = entry.get("preset")
        presets = entry.get("presets")
        params = entry.get("params")
        if preset is None and presets is not None:
            preset = presets
        if isinstance(preset, list):
            for p in preset:
                out.append(ModelRequest(model_id=model_id, preset=str(p), params=params))
        else:
            out.append(
                ModelRequest(
                    model_id=model_id,
                    preset=str(preset) if preset else None,
                    params=params,
                )
            )
    return out


def _expand_variants(requests: List[ModelRequest]) -> List[ModelVariant]:
    variants: List[ModelVariant] = []
    for req in requests:
        spec = get_model_spec(req.model_id)
        presets = spec.presets
        preset_names = {p.name for p in presets}
        fallback_presets: List[Tuple[str, Dict[str, Any]]] = [("default", {})]
        if req.preset == "all":
            selected = presets if presets else []
        elif req.preset:
            if req.preset not in preset_names:
                raise ValueError(
                    f"Unknown preset '{req.preset}' for model '{req.model_id}'"
                )
            selected = [next(p for p in presets if p.name == req.preset)]
        else:
            selected = [presets[0]] if presets else []

        param_sets: List[Dict[str, Any]]
        if isinstance(req.params, list):
            param_sets = [dict(p) for p in req.params if isinstance(p, dict)]
        elif isinstance(req.params, dict):
            param_sets = [dict(req.params)]
        else:
            param_sets = [{}]

        if selected:
            selected_presets = [(p.name, p.params) for p in selected]
        else:
            selected_presets = fallback_presets

        for preset_name, preset_params in selected_presets:
            for idx, override in enumerate(param_sets):
                merged = dict(preset_params)
                merged.update(override)
                suffix = ""
                if len(param_sets) > 1:
                    suffix = f"-custom{idx + 1}"
                variant_name = f"{preset_name}{suffix}"
                variants.append(
                    ModelVariant(
                        model_id=req.model_id,
                        model_label=spec.label,
                        variant=variant_name,
                        preset=preset_name,
                        params=merged,
                    )
                )
    return variants


def parse_run_spec(raw: Dict[str, Any]) -> MLRunSpec:
    raw_models = _normalize_model_requests(raw.get("models"))
    if not raw_models:
        raw_models = [ModelRequest(model_id=spec.id) for spec in list_model_specs()]

    def _as_float(key: str, default: float) -> float:
        try:
            value = raw.get(key, default)
            return float(value)
        except Exception:
            return float(default)

    def _as_int(key: str, default: Optional[int]) -> Optional[int]:
        if key not in raw:
            return default
        try:
            value = raw.get(key, default)
            if value is None:
                return None
            return int(value)
        except Exception:
            return default

    return MLRunSpec(
        models=raw_models,
        train_fraction=_as_float("train_fraction", 0.7),
        train_points=_as_int("train_points", None),
        test_points=_as_int("test_points", None),
        min_train_points=_as_int("min_train_points", 200) or 200,
        min_test_points=_as_int("min_test_points", 100) or 100,
        seed=_as_int("seed", 42) or 42,
        exclude_features=raw.get("exclude_features"),
    )


def _compute_split_points(
    total_points: int,
    spec: MLRunSpec,
) -> Tuple[int, int]:
    if total_points <= 0:
        return 0, 0
    if spec.train_points is not None and spec.test_points is not None:
        train_points = int(spec.train_points)
        test_points = int(spec.test_points)
        if train_points + test_points > total_points:
            test_points = max(0, total_points - train_points)
    elif spec.train_points is not None:
        train_points = int(spec.train_points)
        test_points = total_points - train_points
    elif spec.test_points is not None:
        test_points = int(spec.test_points)
        train_points = total_points - test_points
    else:
        train_points = int(total_points * float(spec.train_fraction))
        test_points = total_points - train_points

    train_points = max(0, min(train_points, total_points))
    test_points = max(0, min(test_points, total_points - train_points))

    if train_points < spec.min_train_points:
        train_points = min(total_points, spec.min_train_points)
        test_points = max(0, total_points - train_points)
    if test_points < spec.min_test_points and total_points - spec.min_test_points >= 1:
        test_points = min(total_points, spec.min_test_points)
        train_points = max(0, total_points - test_points)

    return train_points, test_points


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run_ml_lab(
    cfg: BacktestConfig,
    spec: MLRunSpec,
    out_dir: Path,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    log_fn = log or (lambda msg: print(msg, flush=True))

    def emit_progress(payload: Dict[str, Any]) -> None:
        log_fn(f"PROGRESS {json.dumps(payload, ensure_ascii=True)}")

    metadata = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": cfg.data_dir,
        "eval_lag": cfg.eval_lag,
        "hold": cfg.hold,
        "scale": cfg.scale,
        "net_exposure_target": cfg.net_exposure_target,
        "volatility_target": cfg.volatility_target,
        "max_leverage": cfg.max_leverage,
        "min_leverage": cfg.min_leverage,
        "dd_limit": cfg.dd_limit,
        "dd_reduction": cfg.dd_reduction,
    }
    _write_json(meta_dir / "run_metadata.json", metadata)

    results_path = out_dir / "ml_results.json"
    summary_path = out_dir / "ml_summary.json"

    try:
        aligned_dfs, common_index, stock_symbols = load_and_align_data_for_backtest(
            cfg.data_dir,
            cfg.max_lookback_data_option,
            cfg.min_common_points,
            cfg.eval_lag,
        )
    except DataLoadError as exc:
        metadata.update({"status": "error", "error": str(exc)})
        _write_json(meta_dir / "run_metadata.json", metadata)
        emit_progress({"type": "ml_error", "message": str(exc)})
        return {"error": str(exc)}

    n_stocks = len(stock_symbols)
    total_points = max(0, len(common_index) - int(cfg.eval_lag))
    train_points, test_points = _compute_split_points(total_points, spec)

    if train_points < 50 or test_points < 20:
        err = "Not enough data points for ML split."
        metadata.update({"status": "error", "error": err})
        _write_json(meta_dir / "run_metadata.json", metadata)
        emit_progress({"type": "ml_error", "message": err})
        return {"error": err}

    feature_names = _feature_names(spec.exclude_features)
    sector_groups_vec = get_sector_groups(list(stock_symbols)).astype(float)
    bundle = _get_feature_bundle_cached(
        aligned_dfs,
        list(stock_symbols),
        common_index,
        SCALAR_FEATURE_NAMES,
        CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
        sector_groups_vec,
    )
    ret_fwd = _build_ret_fwd_matrix(aligned_dfs, stock_symbols, common_index)

    X_train_parts: List[np.ndarray] = []
    y_train_parts: List[np.ndarray] = []
    for t in range(train_points):
        features_at_t = bundle.features_per_time[t]
        X_t = _zscore_cross_section(
            _features_to_matrix(features_at_t, feature_names, n_stocks)
        )
        y_t = ret_fwd[t]
        mask = np.isfinite(y_t) & np.all(np.isfinite(X_t), axis=1)
        if not np.any(mask):
            continue
        X_train_parts.append(X_t[mask])
        y_train_parts.append(y_t[mask])

    if not X_train_parts:
        err = "Training data is empty after filtering."
        metadata.update({"status": "error", "error": err})
        _write_json(meta_dir / "run_metadata.json", metadata)
        emit_progress({"type": "ml_error", "message": err})
        return {"error": err}

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)

    test_start = train_points
    test_end = train_points + test_points
    test_index = common_index[test_start:test_end]

    try:
        variants = _expand_variants(spec.models)
    except Exception as exc:
        err = str(exc)
        metadata.update({"status": "error", "error": err})
        _write_json(meta_dir / "run_metadata.json", metadata)
        emit_progress({"type": "ml_error", "message": err})
        return {"error": err}
    total_variants = len(variants)
    emit_progress(
        {
            "type": "ml_start",
            "total_models": total_variants,
            "train_points": train_points,
            "test_points": test_points,
            "feature_count": len(feature_names),
        }
    )

    results: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    best_sharpe: Optional[float] = None

    for idx, variant in enumerate(variants, start=1):
        emit_progress(
            {
                "type": "ml_model_start",
                "model_id": variant.model_id,
                "model_label": variant.model_label,
                "variant": variant.variant,
                "index": idx,
                "total": total_variants,
                "pct_complete": idx / total_variants if total_variants else 0.0,
            }
        )
        entry: Dict[str, Any] = {
            "model_id": variant.model_id,
            "model_label": variant.model_label,
            "variant": variant.variant,
            "preset": variant.preset,
            "params": variant.params,
            "train_points": train_points,
            "test_points": test_points,
            "feature_count": len(feature_names),
            "status": "ok",
        }
        try:
            model = build_model(variant.model_id, variant.params, spec.seed)
            fit_start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_sec = time.perf_counter() - fit_start
            entry["fit_seconds"] = float(fit_sec)

            preds_matrix = np.zeros((test_points, n_stocks), dtype=float)
            ic_values: List[float] = []
            pred_start = time.perf_counter()
            for offset, t in enumerate(range(test_start, test_end)):
                features_at_t = bundle.features_per_time[t]
                X_t = _zscore_cross_section(
                    _features_to_matrix(features_at_t, feature_names, n_stocks)
                )
                preds = model.predict(X_t)
                preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
                preds_matrix[offset] = preds
                corr = _safe_corr(preds, ret_fwd[t])
                if corr is not None:
                    ic_values.append(corr)
            pred_sec = time.perf_counter() - pred_start
            entry["predict_seconds"] = float(pred_sec)

            test_dfs = {sym: df.loc[test_index] for sym, df in aligned_dfs.items()}
            metrics = backtest_cross_sectional_alpha(
                prog=_MLSignalProgram(preds_matrix),
                aligned_dfs=test_dfs,
                common_time_index=test_index,
                stock_symbols=list(stock_symbols),
                n_stocks=n_stocks,
                fee_bps=cfg.fee,
                lag=cfg.eval_lag,
                hold=cfg.hold,
                scale_method=cfg.scale,
                long_short_n=cfg.long_short_n,
                net_exposure_target=cfg.net_exposure_target,
                winsor_p=cfg.winsor_p,
                debug_prints=False,
                annualization_factor=cfg.annualization_factor,
                stop_loss_pct=cfg.stop_loss_pct,
                sector_neutralize_positions=cfg.sector_neutralize_positions,
                volatility_target=cfg.volatility_target,
                volatility_lookback=cfg.volatility_lookback,
                max_leverage=cfg.max_leverage,
                min_leverage=cfg.min_leverage,
                dd_limit=cfg.dd_limit,
                dd_reduction=cfg.dd_reduction,
                initial_state_vars_config={"ml_step": "scalar"},
                scalar_feature_names=SCALAR_FEATURE_NAMES,
                cross_sectional_feature_vector_names=CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
            )

            ic = float(np.mean(ic_values)) if ic_values else 0.0
            entry.update(
                {
                    "IC": ic,
                    "Sharpe": float(metrics.get("Sharpe", 0.0)),
                    "AnnReturn": float(metrics.get("AnnReturn", 0.0)),
                    "AnnVol": float(metrics.get("AnnVol", 0.0)),
                    "MaxDD": float(metrics.get("MaxDD", 0.0)),
                    "Turnover": float(metrics.get("Turnover", 0.0)),
                    "NetExposureMean": float(metrics.get("NetExposureMean", 0.0)),
                    "NetExposureMedian": float(metrics.get("NetExposureMedian", 0.0)),
                    "GrossExposureMean": float(metrics.get("GrossExposureMean", 0.0)),
                }
            )
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)

        results.append(entry)
        _write_json(results_path, results)

        sharpe_val = entry.get("Sharpe")
        try:
            sharpe_num = float(sharpe_val)
        except Exception:
            sharpe_num = None
        if sharpe_num is not None and np.isfinite(sharpe_num):
            if best_sharpe is None or sharpe_num > best_sharpe:
                best_sharpe = sharpe_num
                best_result = entry

        summary_payload = {
            "best_sharpe": best_sharpe,
            "best_result": best_result,
            "completed": len(results),
            "total": total_variants,
        }
        _write_json(summary_path, summary_payload)

        emit_progress(
            {
                "type": "ml_model_end",
                "model_id": variant.model_id,
                "variant": variant.variant,
                "index": idx,
                "total": total_variants,
                "pct_complete": idx / total_variants if total_variants else 0.0,
                "sharpe": sharpe_num,
            }
        )

    metadata.update(
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_points": train_points,
            "test_points": test_points,
            "feature_count": len(feature_names),
            "best_sharpe": best_sharpe,
        }
    )
    _write_json(meta_dir / "run_metadata.json", metadata)
    emit_progress(
        {
            "type": "ml_complete",
            "best_sharpe": best_sharpe,
            "total": total_variants,
        }
    )
    return {"best_sharpe": best_sharpe, "results": results}
